/*
 * mali_compute_patcher.c
 *
 * LD_PRELOAD shim, который перехватывает kbase JOB_SUBMIT для compute-job'а,
 * собранного proprietary libGLES_mali.so из шейдера mali_hello_compute, и
 * подменяет в Compute Job descriptor (jc) base-адреса SSBO src/dst на
 * GPU VA, которые приложение публикует в наши глобалы. Шейдер при этом
 * начинает читать из чужого GPU VA и/или писать в чужой — то, что хотелось
 * «без буферов и без шейдера руками».
 *
 * Передача целевых VA из приложения
 * ---------------------------------
 * Патчер ЭКСПОРТИРУЕТ две глобальные переменные:
 *
 *     volatile uint64_t mali_patch_src;   // GPU VA для SSBO binding=0
 *     volatile uint64_t mali_patch_dst;   // GPU VA для SSBO binding=1
 *
 * Приложение (mali_hello_compute) находит их через
 * dlsym(RTLD_DEFAULT, "mali_patch_src"/"mali_patch_dst") и пишет туда
 * нужные адреса ПЕРЕД glDispatchCompute. Значение 0 = «не подменять».
 * Если патчер не загружен — dlsym вернёт NULL, и приложение поймёт, что
 * идёт «контрольный» прогон без подмены.
 *
 * Раскладка JC (зафиксирована наблюдением observer'ом на Mali-G31,
 * kbase UK 11.21, шейдер `dst[i] = src[i]` с двумя SSBO по 1024 uint32):
 *
 *   jc + 0xc0 .. 0xc7   FAU mirror: dst base
 *   jc + 0xc8 .. 0xcf   FAU mirror: src base
 *   jc + 0xe0 .. 0xe7   SSBO desc #0: dst base
 *   jc + 0xe8 .. 0xeb   SSBO desc #0: size_minus_one  (НЕ трогаем)
 *   jc + 0xf0 .. 0xf7   SSBO desc #1: src base
 *   jc + 0xf8 .. 0xfb   SSBO desc #1: size_minus_one  (НЕ трогаем)
 *
 * Раскладка специфична для этого шейдера (порядок binding, число SSBO,
 * blob-версия). Под другой шейдер пересоберите наблюдением observer'ом.
 *
 * Env vars:
 *   MALI_PATCH_LOG=<path>   лог (по умолчанию stderr)
 *
 * Замечания:
 *   1. Адреса должны быть валидны в текущем GPU AS (т.е. аллоцированы через
 *      тот же FD /dev/mali0). Иначе шейдер фолтит на TRANSLATION_FAULT.
 *   2. Размер подменяемого буфера должен быть не меньше оригинального
 *      (4 KiB = 1024 uint32). Иначе будет out-of-bounds.
 *
 * Build (Android NDK, 32-bit):
 *   $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi28-clang \
 *       -Wall -O2 -fPIC -shared -o libmali_patcher.so mali_compute_patcher.c -ldl
 *
 * Run:
 *   adb shell "cd /data/local/tmp && \
 *     LD_PRELOAD=./libmali_patcher.so MALI_PATCH_LOG=patch.log ./mali_hello_compute"
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/*  Subset of kbase ABI                                                 */
/* ------------------------------------------------------------------ */

#define KBASE_IOCTL_TYPE 0x80

struct kbase_ioctl_job_submit {
    uint64_t addr;
    uint32_t nr_atoms;
    uint32_t stride;
};
#define KBASE_IOCTL_JOB_SUBMIT \
    _IOW(KBASE_IOCTL_TYPE, 2, struct kbase_ioctl_job_submit)

typedef uint32_t base_jd_core_req;
#define BASE_JD_REQ_CS  ((base_jd_core_req)1 << 1)

typedef uint8_t base_atom_id;
typedef uint8_t base_jd_dep_type;
typedef uint8_t base_jd_prio;
struct base_dependency { base_atom_id atom_id; base_jd_dep_type type; };
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
/*  JC offset map for `dst[i] = src[i]` shader (2 SSBO, 1024 uint32)    */
/* ------------------------------------------------------------------ */

static const size_t OFF_FAU_DST   = 0xc0;
static const size_t OFF_FAU_SRC   = 0xc8;
static const size_t OFF_DESC_DST  = 0xe0;
static const size_t OFF_DESC_SRC  = 0xf0;

/* ------------------------------------------------------------------ */
/*  Exported globals — приложение пишет сюда через dlsym(RTLD_DEFAULT)  */
/* ------------------------------------------------------------------ */

/* default visibility, чтобы dlsym(RTLD_DEFAULT, ...) их находил.
 * 0 = не подменять соответствующий base. */
__attribute__((visibility("default"))) volatile uint64_t mali_patch_src = 0;
__attribute__((visibility("default"))) volatile uint64_t mali_patch_dst = 0;

/* ------------------------------------------------------------------ */
/*  State                                                               */
/* ------------------------------------------------------------------ */

static int (*real_ioctl)(int, int, ...) = NULL;
static FILE *g_log = NULL;
static pthread_mutex_t g_log_mtx = PTHREAD_MUTEX_INITIALIZER;

static void plogf(const char *fmt, ...) {
    pthread_mutex_lock(&g_log_mtx);
    va_list ap; va_start(ap, fmt);
    vfprintf(g_log, fmt, ap);
    va_end(ap);
    pthread_mutex_unlock(&g_log_mtx);
}

__attribute__((constructor))
static void patcher_init(void) {
    real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    const char *path = getenv("MALI_PATCH_LOG");
    if (path && *path) {
        g_log = fopen(path, "w");
        if (!g_log) g_log = stderr;
    } else {
        g_log = stderr;
    }
    setvbuf(g_log, NULL, _IONBF, 0);
    fprintf(g_log, "=== mali_compute_patcher attached pid=%d\n", getpid());
    fprintf(g_log, "    waiting for app to publish mali_patch_src/dst "
                   "(0 = passthrough)\n");
}

/* ------------------------------------------------------------------ */
/*  JC patch                                                            */
/* ------------------------------------------------------------------ */

static void patch_one_jc(uint64_t jc_gpu) {
    /* SAME_VA: jc на 32-битном процессе совпадает с CPU pointer'ом. */
    volatile uint64_t *jc = (volatile uint64_t *)(uintptr_t)jc_gpu;
    if (!jc) return;

    uint64_t dst = mali_patch_dst;
    uint64_t src = mali_patch_src;

    if (dst) {
        uint64_t old1 = jc[OFF_FAU_DST  / 8];
        uint64_t old2 = jc[OFF_DESC_DST / 8];
        jc[OFF_FAU_DST  / 8] = dst;
        jc[OFF_DESC_DST / 8] = dst;
        plogf("  patch DST jc+0x%02zx: 0x%llx → 0x%llx\n", OFF_FAU_DST,
              (unsigned long long)old1, (unsigned long long)dst);
        plogf("  patch DST jc+0x%02zx: 0x%llx → 0x%llx\n", OFF_DESC_DST,
              (unsigned long long)old2, (unsigned long long)dst);
    }
    if (src) {
        uint64_t old1 = jc[OFF_FAU_SRC  / 8];
        uint64_t old2 = jc[OFF_DESC_SRC / 8];
        jc[OFF_FAU_SRC  / 8] = src;
        jc[OFF_DESC_SRC / 8] = src;
        plogf("  patch SRC jc+0x%02zx: 0x%llx → 0x%llx\n", OFF_FAU_SRC,
              (unsigned long long)old1, (unsigned long long)src);
        plogf("  patch SRC jc+0x%02zx: 0x%llx → 0x%llx\n", OFF_DESC_SRC,
              (unsigned long long)old2, (unsigned long long)src);
    }
}

static void maybe_patch(struct kbase_ioctl_job_submit *sub) {
    if (!mali_patch_src && !mali_patch_dst) return;
    for (uint32_t i = 0; i < sub->nr_atoms; i++) {
        base_jd_atom_v2 *a = (base_jd_atom_v2 *)
            ((char *)(uintptr_t)sub->addr + (size_t)i * sub->stride);
        if (!(a->core_req & BASE_JD_REQ_CS)) continue;
        if (!a->jc) continue;
        plogf("[JOB_SUBMIT compute atom=%u jc=0x%llx core_req=0x%x — patching]\n",
              a->atom_number, (unsigned long long)a->jc, a->core_req);
        patch_one_jc(a->jc);
    }
}

/* ------------------------------------------------------------------ */
/*  ioctl interceptor                                                   */
/* ------------------------------------------------------------------ */

int ioctl(int fd, int req, ...) {
    va_list ap; va_start(ap, req);
    void *arg = va_arg(ap, void *);
    va_end(ap);

    if (!real_ioctl) real_ioctl = dlsym(RTLD_NEXT, "ioctl");
    if (!real_ioctl) { errno = ENOSYS; return -1; }

    unsigned ureq = (unsigned)req;
    if (ureq == (unsigned)KBASE_IOCTL_JOB_SUBMIT && arg) {
        maybe_patch((struct kbase_ioctl_job_submit *)arg);
    }

    return real_ioctl(fd, req, arg);
}
