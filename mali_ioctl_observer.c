/*
 * mali_ioctl_observer.c  (v2)
 *
 * LD_PRELOAD-shim для intercept'a kbase ioctl + mmap/mmap64. Цели:
 *   - открытие /dev/mali0, VERSION_CHECK/SET_FLAGS;
 *   - аллокации (MEM_ALLOC, MEM_IMPORT) — какие GPU VA возвращает kbase,
 *     и куда blob их потом mmap-ит (нужно для SAME_VA cookie -> user VA);
 *   - JOB_SUBMIT с BASE_JD_REQ_CS: hex-дамп Compute Job descriptor по `jc`,
 *     ПЛЮС сканирование `jc` на 8-байтные значения, попадающие внутрь
 *     известных живых аллокаций (помогает найти SSBO/shader_meta-указатели,
 *     включая tagged-pointers с битами 0..3),
 *     ПЛЮС дамп первых 128 байт каждой живой CPU-accessible аллокации
 *     (так находим SSBO src/dst по нашим маркерам 0xAABBxxxx / 0xDEADBEEF).
 *
 * Полагается на SAME_VA: для аллокаций с битом BASE_MEM_SAME_VA реальный
 * GPU VA равен адресу, который вернёт mmap (cookie → user VA). Для
 * non-SAME_VA gpu_va из MEM_ALLOC сразу годится как «GPU VA», а
 * соответствующий cpu_va получается из mmap по тому же offset.
 *
 * Лог: stderr или $MALI_OBS_LOG.
 *
 * Build (Android NDK).
 *
 *   # 32-bit:
 *   $NDK/.../armv7a-linux-androideabi28-clang \
 *       -Wall -O2 -fPIC -shared -o libmali_observer.so mali_ioctl_observer.c -ldl
 *
 *   # 64-bit:
 *   $NDK/.../aarch64-linux-android28-clang \
 *       -Wall -O2 -fPIC -shared -o libmali_observer.so mali_ioctl_observer.c -ldl
 *
 * Run:
 *   adb push libmali_observer.so mali_hello_compute /data/local/tmp/
 *   adb shell "cd /data/local/tmp && \
 *     LD_PRELOAD=./libmali_observer.so MALI_OBS_LOG=obs.log ./mali_hello_compute"
 *   adb pull /data/local/tmp/obs.log
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/*  Subset of the kbase ABI (UK 11.x, drivers/gpu/arm/midgard)         */
/* ------------------------------------------------------------------ */

#define KBASE_IOCTL_TYPE 0x80

struct kbase_ioctl_version_check { uint16_t major, minor; };
#define KBASE_IOCTL_VERSION_CHECK \
    _IOWR(KBASE_IOCTL_TYPE, 0, struct kbase_ioctl_version_check)

struct kbase_ioctl_set_flags { uint32_t create_flags; };
#define KBASE_IOCTL_SET_FLAGS \
    _IOW(KBASE_IOCTL_TYPE, 1, struct kbase_ioctl_set_flags)

struct kbase_ioctl_job_submit {
    uint64_t addr;
    uint32_t nr_atoms;
    uint32_t stride;
};
#define KBASE_IOCTL_JOB_SUBMIT \
    _IOW(KBASE_IOCTL_TYPE, 2, struct kbase_ioctl_job_submit)

union kbase_ioctl_mem_alloc {
    struct { uint64_t va_pages, commit_pages, extent, flags; } in;
    struct { uint64_t flags, gpu_va; } out;
};
#define KBASE_IOCTL_MEM_ALLOC \
    _IOWR(KBASE_IOCTL_TYPE, 5, union kbase_ioctl_mem_alloc)

struct kbase_ioctl_mem_free { uint64_t gpu_addr; };
#define KBASE_IOCTL_MEM_FREE \
    _IOW(KBASE_IOCTL_TYPE, 7, struct kbase_ioctl_mem_free)

union kbase_ioctl_mem_import {
    struct { uint64_t flags, phandle; uint32_t type, padding; } in;
    struct { uint64_t flags, gpu_va, va_pages; } out;
};
#define KBASE_IOCTL_MEM_IMPORT \
    _IOWR(KBASE_IOCTL_TYPE, 22, union kbase_ioctl_mem_import)

#define BASE_MEM_SAME_VA   (1ull << 13)
#define BASE_MEM_COOKIE_BASE  (64ul << 12)

/* base_jd_atom_v2 — same layout as in mali_write_value_demo.c */
typedef uint32_t base_jd_core_req;
#define BASE_JD_REQ_FS                  ((base_jd_core_req)1 << 0)
#define BASE_JD_REQ_CS                  ((base_jd_core_req)1 << 1)
#define BASE_JD_REQ_T                   ((base_jd_core_req)1 << 2)
#define BASE_JD_REQ_CF                  ((base_jd_core_req)1 << 3)
#define BASE_JD_REQ_V                   ((base_jd_core_req)1 << 4)
#define BASE_JD_REQ_PERMON              ((base_jd_core_req)1 << 7)
#define BASE_JD_REQ_EXTERNAL_RESOURCES  ((base_jd_core_req)1 << 8)
#define BASE_JD_REQ_SOFT_JOB            ((base_jd_core_req)1 << 9)
#define BASE_JD_REQ_ONLY_COMPUTE        ((base_jd_core_req)1 << 10)

typedef uint8_t base_atom_id;
typedef uint8_t base_jd_dep_type;
typedef uint8_t base_jd_prio;

struct base_dependency {
    base_atom_id     atom_id;
    base_jd_dep_type dependency_type;
};
struct base_jd_udata { uint64_t blob[2]; };

typedef struct base_jd_atom_v2 {
    uint64_t              jc;
    struct base_jd_udata  udata;
    uint64_t              extres_list;
    uint16_t              nr_extres;
    uint16_t              compat_core_req;
    struct base_dependency pre_dep[2];
    base_atom_id          atom_number;
    base_jd_prio          prio;
    uint8_t               device_nr;
    uint8_t               padding[1];
    base_jd_core_req      core_req;
} base_jd_atom_v2;

/* ------------------------------------------------------------------ */
/*  Logger                                                              */
/* ------------------------------------------------------------------ */

/* bionic's ioctl is declared as int ioctl(int, int, ...) */
static int   (*real_ioctl)(int, int, ...) = NULL;
static void *(*real_mmap)(void *, size_t, int, int, int, off_t)   = NULL;
static void *(*real_mmap64)(void *, size_t, int, int, int, off64_t) = NULL;
static FILE *g_log = NULL;
static pthread_mutex_t g_log_mtx = PTHREAD_MUTEX_INITIALIZER;

/* fd, через который blob говорит с kbase — запоминаем первый, на котором
 * проходит kbase ioctl. */
static int g_mali_fd = -1;

__attribute__((constructor))
static void obs_init(void) {
    real_ioctl  = dlsym(RTLD_NEXT, "ioctl");
    real_mmap   = dlsym(RTLD_NEXT, "mmap");
    real_mmap64 = dlsym(RTLD_NEXT, "mmap64");
    const char *path = getenv("MALI_OBS_LOG");
    if (path && *path) {
        g_log = fopen(path, "w");
        if (!g_log) g_log = stderr;
    } else {
        g_log = stderr;
    }
    setvbuf(g_log, NULL, _IONBF, 0);
    fprintf(g_log, "=== mali_ioctl_observer v2 attached pid=%d ===\n", getpid());
}

static void logf(const char *fmt, ...) {
    pthread_mutex_lock(&g_log_mtx);
    va_list ap; va_start(ap, fmt);
    vfprintf(g_log, fmt, ap);
    va_end(ap);
    pthread_mutex_unlock(&g_log_mtx);
}

static void hexdump(const char *tag, const void *p, size_t n) {
    if (!p) { logf("  %s: <null>\n", tag); return; }
    const uint8_t *b = (const uint8_t *)p;
    logf("  %s @ %p (%zu bytes):\n", tag, p, n);
    for (size_t i = 0; i < n; i += 16) {
        size_t end = i + 16 < n ? i + 16 : n;
        logf("    %04zx:", i);
        for (size_t j = i; j < end; j++) logf(" %02x", b[j]);
        for (size_t j = end; j < i + 16; j++) logf("   ");
        logf("  |");
        for (size_t j = i; j < end; j++) {
            uint8_t c = b[j];
            logf("%c", (c >= 0x20 && c < 0x7f) ? c : '.');
        }
        logf("|\n");
    }
}

static const char *core_req_str(base_jd_core_req r) {
    static char buf[256];
    buf[0] = 0;
    if (r & BASE_JD_REQ_FS)                 strcat(buf, "FS|");
    if (r & BASE_JD_REQ_CS)                 strcat(buf, "CS|");
    if (r & BASE_JD_REQ_T)                  strcat(buf, "T|");
    if (r & BASE_JD_REQ_CF)                 strcat(buf, "CF|");
    if (r & BASE_JD_REQ_V)                  strcat(buf, "V|");
    if (r & BASE_JD_REQ_PERMON)             strcat(buf, "PERMON|");
    if (r & BASE_JD_REQ_EXTERNAL_RESOURCES) strcat(buf, "EXTRES|");
    if (r & BASE_JD_REQ_SOFT_JOB)           strcat(buf, "SOFT|");
    if (r & BASE_JD_REQ_ONLY_COMPUTE)       strcat(buf, "ONLY_COMPUTE|");
    size_t n = strlen(buf);
    if (n && buf[n-1] == '|') buf[n-1] = 0;
    return buf;
}

/* ------------------------------------------------------------------ */
/*  Allocation tracking                                                 */
/* ------------------------------------------------------------------ */

#define MAX_ALLOCS 4096

struct alloc_entry {
    uint64_t cookie;     /* что вернул MEM_ALLOC: либо реальный gpu_va (no SAME_VA),
                          * либо cookie (>=BASE_MEM_COOKIE_BASE для SAME_VA) */
    uint64_t gpu_va;     /* финальный GPU VA: после mmap для SAME_VA, иначе == cookie */
    uint64_t va_pages;
    uint64_t flags;
    void    *cpu_va;     /* mmap'd адрес (CPU-видимый) */
    int      is_import;
    int      valid;
};

static struct alloc_entry g_allocs[MAX_ALLOCS];
static int g_n_allocs = 0;
static pthread_mutex_t g_allocs_mtx = PTHREAD_MUTEX_INITIALIZER;

static struct alloc_entry *alloc_add(uint64_t cookie, uint64_t va_pages,
                                     uint64_t flags, int is_import) {
    pthread_mutex_lock(&g_allocs_mtx);
    struct alloc_entry *e = NULL;
    if (g_n_allocs < MAX_ALLOCS) {
        e = &g_allocs[g_n_allocs++];
        e->cookie    = cookie;
        e->gpu_va    = cookie;     /* временно; обновим в mmap для SAME_VA */
        e->va_pages  = va_pages;
        e->flags     = flags;
        e->cpu_va    = NULL;
        e->is_import = is_import;
        e->valid     = 1;
    }
    pthread_mutex_unlock(&g_allocs_mtx);
    return e;
}

/* Найти наиболее недавнюю запись по «cookie/offset», у которой ещё не
 * установлен cpu_va. Cookie у SAME_VA переиспользуется, поэтому ищем
 * именно «висящие» (cpu_va == NULL). */
static struct alloc_entry *alloc_find_pending(uint64_t off) {
    pthread_mutex_lock(&g_allocs_mtx);
    struct alloc_entry *r = NULL;
    for (int i = g_n_allocs - 1; i >= 0; i--) {
        if (g_allocs[i].valid && g_allocs[i].cookie == off &&
            g_allocs[i].cpu_va == NULL) {
            r = &g_allocs[i];
            break;
        }
    }
    pthread_mutex_unlock(&g_allocs_mtx);
    return r;
}

static struct alloc_entry *alloc_find_containing(uint64_t addr) {
    pthread_mutex_lock(&g_allocs_mtx);
    struct alloc_entry *r = NULL;
    for (int i = 0; i < g_n_allocs; i++) {
        struct alloc_entry *e = &g_allocs[i];
        if (!e->valid) continue;
        uint64_t lo = e->gpu_va;
        uint64_t hi = e->gpu_va + e->va_pages * 4096ULL;
        if (addr >= lo && addr < hi) { r = e; break; }
    }
    pthread_mutex_unlock(&g_allocs_mtx);
    return r;
}

static void alloc_remove(uint64_t gpu_addr) {
    pthread_mutex_lock(&g_allocs_mtx);
    for (int i = 0; i < g_n_allocs; i++) {
        if (g_allocs[i].valid &&
            (g_allocs[i].gpu_va == gpu_addr ||
             g_allocs[i].cookie == gpu_addr)) {
            g_allocs[i].valid = 0;
            break;
        }
    }
    pthread_mutex_unlock(&g_allocs_mtx);
}

/* ------------------------------------------------------------------ */
/*  mmap / mmap64 interceptor                                           */
/* ------------------------------------------------------------------ */

static void mmap_post(void *r, int fd, off64_t off, size_t len, int prot, int flags) {
    if (r == MAP_FAILED || fd != g_mali_fd) return;
    uint64_t off_u = (uint64_t)off;
    struct alloc_entry *e = alloc_find_pending(off_u);
    if (!e) {
        /* Может быть, это не cookie, а реальный gpu_va non-SAME_VA-аллока,
         * который blob mmap-ит. Тогда entry уже есть и у него cookie == gpu_va. */
        pthread_mutex_lock(&g_allocs_mtx);
        for (int i = g_n_allocs - 1; i >= 0; i--) {
            if (g_allocs[i].valid && g_allocs[i].cookie == off_u) {
                e = &g_allocs[i];
                break;
            }
        }
        pthread_mutex_unlock(&g_allocs_mtx);
    }
    if (e) {
        e->cpu_va = r;
        if (e->flags & BASE_MEM_SAME_VA) {
            /* Для SAME_VA реальный GPU VA — это и есть результат mmap. */
            e->gpu_va = (uint64_t)(uintptr_t)r;
        }
        logf("[mmap fd=%d off=0x%llx len=%zu prot=0x%x flags=0x%x → %p  "
             "(matched cookie=0x%llx, gpu_va=0x%llx, %llu pages, flags=0x%llx, %s)]\n",
             fd, (unsigned long long)off_u, len, prot, flags, r,
             (unsigned long long)e->cookie,
             (unsigned long long)e->gpu_va,
             (unsigned long long)e->va_pages,
             (unsigned long long)e->flags,
             e->is_import ? "IMPORT" : "ALLOC");
    } else {
        logf("[mmap fd=%d off=0x%llx len=%zu prot=0x%x flags=0x%x → %p  "
             "(untracked)]\n",
             fd, (unsigned long long)off_u, len, prot, flags, r);
    }
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    if (!real_mmap) real_mmap = dlsym(RTLD_NEXT, "mmap");
    void *r = real_mmap(addr, len, prot, flags, fd, off);
    /* На 32-bit Android off_t — signed 32-bit, cookie'и небольшие — OK.
     * Большие offset'ы для non-SAME_VA blob берёт через mmap64. На 64 битах
     * off_t уже 64-битный, кастуем напрямую. */
#if defined(__LP64__) || defined(_LP64)
    off64_t off64 = (off64_t)off;
#else
    off64_t off64 = (off64_t)(uint32_t)off;  /* avoid sign extension on small cookies */
#endif
    mmap_post(r, fd, off64, len, prot, flags);
    return r;
}

void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off64_t off) {
    if (!real_mmap64) real_mmap64 = dlsym(RTLD_NEXT, "mmap64");
    void *r = real_mmap64(addr, len, prot, flags, fd, off);
    mmap_post(r, fd, off, len, prot, flags);
    return r;
}

/* ------------------------------------------------------------------ */
/*  ioctl interceptor                                                   */
/* ------------------------------------------------------------------ */

static void dump_live_allocations_head(void) {
    logf("  --- live allocations head dump (first 128 bytes each) ---\n");
    pthread_mutex_lock(&g_allocs_mtx);
    int n = g_n_allocs;
    pthread_mutex_unlock(&g_allocs_mtx);

    for (int i = 0; i < n; i++) {
        struct alloc_entry *e = &g_allocs[i];
        if (!e->valid || !e->cpu_va) continue;
        size_t bytes = e->va_pages * 4096ULL;
        if (bytes > 128) bytes = 128;
        char tag[160];
        snprintf(tag, sizeof(tag),
                 "alloc#%d %s gpu_va=0x%llx cpu_va=%p %llu pg flags=0x%llx",
                 i, e->is_import ? "IMPORT" : "ALLOC ",
                 (unsigned long long)e->gpu_va, e->cpu_va,
                 (unsigned long long)e->va_pages,
                 (unsigned long long)e->flags);
        hexdump(tag, e->cpu_va, bytes);
    }
}

static void scan_jc_pointers(const void *jc_cpu, size_t bytes) {
    logf("  --- pointer scan in jc (8-byte aligned, into known allocs) ---\n");
    const uint64_t *p = (const uint64_t *)jc_cpu;
    size_t words = bytes / 8;
    for (size_t k = 0; k < words; k++) {
        uint64_t val = p[k];
        if (val == 0) continue;
        struct alloc_entry *e = alloc_find_containing(val);
        const char *note = "";
        if (!e) {
            /* Попробуем как tagged-pointer: маска младших 4 бит. */
            struct alloc_entry *e2 = alloc_find_containing(val & ~0xfULL);
            if (e2) { e = e2; note = " (low-4-bits tagged)"; }
        }
        if (e) {
            uint64_t off_in = (val & ~0xfULL) - e->gpu_va;
            logf("    jc+0x%03zx: 0x%016llx → alloc#? gpu_va=0x%llx +0x%llx "
                 "(%llu pg, flags=0x%llx, %s)%s\n",
                 k * 8, (unsigned long long)val,
                 (unsigned long long)e->gpu_va,
                 (unsigned long long)off_in,
                 (unsigned long long)e->va_pages,
                 (unsigned long long)e->flags,
                 e->is_import ? "IMPORT" : "ALLOC", note);
        }
    }
}

static void dump_job_submit(int fd, struct kbase_ioctl_job_submit *sub) {
    logf("[JOB_SUBMIT fd=%d nr_atoms=%u stride=%u addr=0x%llx]\n",
         fd, sub->nr_atoms, sub->stride,
         (unsigned long long)sub->addr);

    int has_cs = 0;
    for (uint32_t i = 0; i < sub->nr_atoms; i++) {
        base_jd_atom_v2 *a = (base_jd_atom_v2 *)
            ((char *)(uintptr_t)sub->addr + (size_t)i * sub->stride);

        logf(" atom[%u]: atom#=%u core_req=0x%08x (%s) "
             "jc=0x%llx nr_extres=%u extres_list=0x%llx\n",
             i, a->atom_number, a->core_req, core_req_str(a->core_req),
             (unsigned long long)a->jc, a->nr_extres,
             (unsigned long long)a->extres_list);

        if ((a->core_req & BASE_JD_REQ_CS) && a->jc) {
            has_cs = 1;
            hexdump("jc", (void *)(uintptr_t)a->jc, 768);
            scan_jc_pointers((const void *)(uintptr_t)a->jc, 768);
        }

        if (a->nr_extres && a->extres_list) {
            size_t n = (size_t)a->nr_extres * sizeof(uint64_t);
            hexdump("extres_list", (void *)(uintptr_t)a->extres_list, n);
        }
    }

    if (has_cs) dump_live_allocations_head();
}

int ioctl(int fd, int req, ...) {
    va_list ap; va_start(ap, req);
    void *arg = va_arg(ap, void *);
    va_end(ap);

    unsigned ureq = (unsigned)req;
    int is_kbase = (ureq & 0xff00) == (KBASE_IOCTL_TYPE << 8);

    /* Запоминаем mali fd по первому kbase ioctl. */
    if (is_kbase && g_mali_fd < 0) g_mali_fd = fd;

    /* Pre-call: что хочет blob. Параллельно копируем «in»-поля, потому
     * что post-call мы видим уже перезаписанные «out»-поля. */
    uint64_t in_alloc_pages = 0;
    if (is_kbase) {
        unsigned nr = ureq & 0xff;
        switch (ureq) {
        case KBASE_IOCTL_VERSION_CHECK: {
            struct kbase_ioctl_version_check *v = arg;
            logf("[VERSION_CHECK fd=%d in major=%u minor=%u]\n",
                 fd, v->major, v->minor);
            break;
        }
        case KBASE_IOCTL_SET_FLAGS: {
            struct kbase_ioctl_set_flags *s = arg;
            logf("[SET_FLAGS fd=%d create_flags=0x%x]\n",
                 fd, s->create_flags);
            break;
        }
        case KBASE_IOCTL_MEM_ALLOC: {
            union kbase_ioctl_mem_alloc *m = arg;
            in_alloc_pages = m->in.va_pages;
            logf("[MEM_ALLOC>  fd=%d va_pages=%llu commit_pages=%llu "
                 "extent=%llu flags=0x%llx]\n",
                 fd,
                 (unsigned long long)m->in.va_pages,
                 (unsigned long long)m->in.commit_pages,
                 (unsigned long long)m->in.extent,
                 (unsigned long long)m->in.flags);
            break;
        }
        case KBASE_IOCTL_MEM_FREE: {
            struct kbase_ioctl_mem_free *m = arg;
            logf("[MEM_FREE fd=%d gpu_addr=0x%llx]\n",
                 fd, (unsigned long long)m->gpu_addr);
            break;
        }
        case KBASE_IOCTL_MEM_IMPORT: {
            union kbase_ioctl_mem_import *m = arg;
            logf("[MEM_IMPORT> fd=%d type=%u flags=0x%llx phandle=0x%llx]\n",
                 fd, m->in.type,
                 (unsigned long long)m->in.flags,
                 (unsigned long long)m->in.phandle);
            break;
        }
        case KBASE_IOCTL_JOB_SUBMIT:
            dump_job_submit(fd, arg);
            break;
        default:
            logf("[ioctl fd=%d nr=%u req=0x%x arg=%p]\n", fd, nr, ureq, arg);
            break;
        }
    }

    int rc = real_ioctl(fd, req, arg);

    if (is_kbase) {
        if (rc == 0) {
            switch (ureq) {
            case KBASE_IOCTL_VERSION_CHECK: {
                struct kbase_ioctl_version_check *v = arg;
                logf("  VERSION_CHECK< out major=%u minor=%u\n",
                     v->major, v->minor);
                break;
            }
            case KBASE_IOCTL_MEM_ALLOC: {
                union kbase_ioctl_mem_alloc *m = arg;
                logf("  MEM_ALLOC<   flags=0x%llx gpu_va=0x%llx\n",
                     (unsigned long long)m->out.flags,
                     (unsigned long long)m->out.gpu_va);
                alloc_add(m->out.gpu_va, in_alloc_pages, m->out.flags, 0);
                break;
            }
            case KBASE_IOCTL_MEM_IMPORT: {
                union kbase_ioctl_mem_import *m = arg;
                logf("  MEM_IMPORT<  flags=0x%llx gpu_va=0x%llx va_pages=%llu\n",
                     (unsigned long long)m->out.flags,
                     (unsigned long long)m->out.gpu_va,
                     (unsigned long long)m->out.va_pages);
                alloc_add(m->out.gpu_va, m->out.va_pages, m->out.flags, 1);
                break;
            }
            case KBASE_IOCTL_MEM_FREE: {
                struct kbase_ioctl_mem_free *m = arg;
                alloc_remove(m->gpu_addr);
                break;
            }
            default: break;
            }
        } else {
            logf("  -> rc=%d errno=%d %s\n", rc, errno, strerror(errno));
        }
    }

    return rc;
}
