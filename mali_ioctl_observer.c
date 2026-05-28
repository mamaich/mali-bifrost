/*
 * mali_ioctl_observer.c
 *
 * LD_PRELOAD-shim, который перехватывает ioctl() и для интересующих нас
 * операций kbase (`KBASE_IOCTL_*`) логирует параметры и сырые байты
 * структур в памяти. Цель — увидеть, как libGLES_mali.so:
 *   - открывает /dev/mali0, делает handshake;
 *   - аллоцирует GPU-память (MEM_ALLOC, MEM_IMPORT) — какие GPU VA он
 *     получает обратно;
 *   - подаёт compute-job (JOB_SUBMIT с BASE_JD_REQ_CS|ONLY_COMPUTE),
 *     и что физически лежит по адресу `jc` (Compute Job descriptor).
 *
 * Полагается на SAME_VA: указатели в kbase для 64-битного non-compat
 * процесса равны CPU-адресам в этом же процессе, поэтому `jc` и
 * `extres_list` можно разыменовывать напрямую.
 *
 * Лог идёт в stderr либо в файл из переменной окружения MALI_OBS_LOG.
 *
 * Build (Android NDK, aarch64):
 *   $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang \
 *       -Wall -O2 -fPIC -shared -o libmali_observer.so mali_ioctl_observer.c -ldl
 *
 * Run:
 *   adb push libmali_observer.so /data/local/tmp/
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
#include <sys/syscall.h>
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

static int  (*real_ioctl)(int, unsigned long, ...) = NULL;
static FILE *g_log = NULL;
static pthread_mutex_t g_log_mtx = PTHREAD_MUTEX_INITIALIZER;

__attribute__((constructor))
static void obs_init(void) {
    real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    const char *path = getenv("MALI_OBS_LOG");
    if (path && *path) {
        g_log = fopen(path, "w");
        if (!g_log) g_log = stderr;
    } else {
        g_log = stderr;
    }
    setvbuf(g_log, NULL, _IONBF, 0);
    fprintf(g_log, "=== mali_ioctl_observer attached pid=%d ===\n", getpid());
}

static void logf(const char *fmt, ...) {
    pthread_mutex_lock(&g_log_mtx);
    va_list ap; va_start(ap, fmt);
    vfprintf(g_log, fmt, ap);
    va_end(ap);
    pthread_mutex_unlock(&g_log_mtx);
}

/* Безопасно дампит до `n` байт по адресу `p`. Защищается от падений
 * через /proc/self/maps-readability — но проще просто оборачивать в
 * try-catch через signal: тут оставим без защиты, потому что blob
 * у нас в одном AS и адрес валиден. */
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
    if (r & BASE_JD_REQ_FS)               strcat(buf, "FS|");
    if (r & BASE_JD_REQ_CS)               strcat(buf, "CS|");
    if (r & BASE_JD_REQ_T)                strcat(buf, "T|");
    if (r & BASE_JD_REQ_CF)               strcat(buf, "CF|");
    if (r & BASE_JD_REQ_V)                strcat(buf, "V|");
    if (r & BASE_JD_REQ_PERMON)           strcat(buf, "PERMON|");
    if (r & BASE_JD_REQ_EXTERNAL_RESOURCES) strcat(buf, "EXTRES|");
    if (r & BASE_JD_REQ_SOFT_JOB)         strcat(buf, "SOFT|");
    if (r & BASE_JD_REQ_ONLY_COMPUTE)     strcat(buf, "ONLY_COMPUTE|");
    size_t n = strlen(buf);
    if (n && buf[n-1] == '|') buf[n-1] = 0;
    return buf;
}

/* ------------------------------------------------------------------ */
/*  ioctl interceptor                                                   */
/* ------------------------------------------------------------------ */

static void dump_job_submit(int fd, struct kbase_ioctl_job_submit *sub) {
    logf("[JOB_SUBMIT fd=%d nr_atoms=%u stride=%u addr=0x%llx]\n",
         fd, sub->nr_atoms, sub->stride,
         (unsigned long long)sub->addr);

    for (uint32_t i = 0; i < sub->nr_atoms; i++) {
        base_jd_atom_v2 *a = (base_jd_atom_v2 *)
            ((char *)(uintptr_t)sub->addr + (size_t)i * sub->stride);

        logf(" atom[%u]: atom#=%u core_req=0x%08x (%s) "
             "jc=0x%llx nr_extres=%u extres_list=0x%llx\n",
             i, a->atom_number, a->core_req, core_req_str(a->core_req),
             (unsigned long long)a->jc, a->nr_extres,
             (unsigned long long)a->extres_list);

        /* Дампим Compute Job header + следующие 256 байт. У compute
         * descriptor'а тело начинается с 32-байтного Job Header,
         * дальше Invocation (8) + Parameters (24) + Draw (... 120+).
         * Дампим 512 байт, чтобы захватить и descriptor, и часть
         * содержимого по ссылке. */
        if ((a->core_req & BASE_JD_REQ_CS) && a->jc) {
            hexdump("jc", (void *)(uintptr_t)a->jc, 512);
        }

        /* Если есть external resources — это массив base_external_resource
         * (по 8 байт каждый: gpu_va | access_bit). */
        if (a->nr_extres && a->extres_list) {
            size_t n = (size_t)a->nr_extres * sizeof(uint64_t);
            hexdump("extres_list", (void *)(uintptr_t)a->extres_list, n);
        }
    }
}

int ioctl(int fd, unsigned long req, ...) {
    /* Always extract the pointer argument; kbase ioctls are all _IOW/_IOWR/_IOR
     * with a pointer payload. */
    va_list ap; va_start(ap, req);
    void *arg = va_arg(ap, void *);
    va_end(ap);

    /* Pre-call: что хочет blob */
    if ((req & 0xff00) == (KBASE_IOCTL_TYPE << 8)) {
        unsigned nr = req & 0xff;
        switch (req) {
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
            logf("[ioctl fd=%d nr=%u req=0x%lx arg=%p]\n", fd, nr, req, arg);
            break;
        }
    }

    int rc = real_ioctl(fd, req, arg);

    /* Post-call: что вернулось */
    if (rc == 0 && (req & 0xff00) == (KBASE_IOCTL_TYPE << 8)) {
        switch (req) {
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
            break;
        }
        case KBASE_IOCTL_MEM_IMPORT: {
            union kbase_ioctl_mem_import *m = arg;
            logf("  MEM_IMPORT<  flags=0x%llx gpu_va=0x%llx va_pages=%llu\n",
                 (unsigned long long)m->out.flags,
                 (unsigned long long)m->out.gpu_va,
                 (unsigned long long)m->out.va_pages);
            break;
        }
        default: break;
        }
    } else if (rc != 0 && (req & 0xff00) == (KBASE_IOCTL_TYPE << 8)) {
        logf("  -> rc=%d errno=%d %s\n", rc, errno, strerror(errno));
    }

    return rc;
}
