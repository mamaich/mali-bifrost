/*
 * mali_import.c
 *
 * Combines GLES 3.1 compute dispatch (mali_hello_compute) with direct kbase
 * ioctl operations on the same /dev/mali0 fd that EGL opened.  After the
 * GLES glFinish(), uses the blob's fd to:
 *
 *   1. Import a CPU buffer (Buff[]) as a GPU USER_BUFFER allocation.
 *   2. Allocate a SAME_VA job-descriptor page.
 *   3. Build and submit a WRITE_VALUE job that writes 0x12345678 into Buff[0].
 *   4. Read back the completion event and verify the result.
 *
 * KEY FIX: EGL/blob opens /dev/mali0 with O_NONBLOCK, so a bare read() for
 * the completion event returns EAGAIN immediately.  The fix is to use poll()
 * to wait until the event is readable before calling read() — this avoids
 * touching fd flags that the blob may rely on.
 *
 * Physical-memory import (mali_import_phys) requires a custom kernel UAPI
 * header (mali_intercept_uapi.h).  It is compiled only when
 * -DHAVE_INTERCEPT_UAPI is passed.
 *
 * Build (Android NDK, 32-bit ARM):
 *   $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi28-clang \
 *       -Wall -O2 mali_import.c -o mali_import \
 *       -lEGL -lGLESv2 -ldl
 *
 *   # With physical-memory import:
 *   $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/armv7a-linux-androideabi28-clang \
 *       -Wall -O2 -DHAVE_INTERCEPT_UAPI mali_import.c -o mali_import \
 *       -lEGL -lGLESv2 -ldl
 *
 * Run:
 *   adb push mali_import /data/local/tmp/
 *   adb shell "cd /data/local/tmp && ./mali_import"
 */

#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>

#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/ioctl.h>
#include <linux/types.h>

#ifdef HAVE_INTERCEPT_UAPI
#include "mali_intercept_uapi.h"
#endif

#define PAGE_SHIFT_4K 12
#define PAGE_SZ_4K    (1u << PAGE_SHIFT_4K)
#define N_U32         (PAGE_SZ_4K / sizeof(uint32_t))

#define DEMO_PHYS_ADDR  0x1080000ULL
#define DEMO_LENGTH     0x1000ULL

#define USE_SECURE_IMPORT 0

/* ------------------------------------------------------------------ */
/* GLES compute shader: dst[i] = src[i], two SSBOs, local_size_x=64   */
/* ------------------------------------------------------------------ */

static const char *cs_glsl =
    "#version 310 es\n"
    "layout(local_size_x = 64) in;\n"
    "layout(std430, binding = 0) readonly  buffer Src { uint s[]; };\n"
    "layout(std430, binding = 1) writeonly buffer Dst { uint d[]; };\n"
    "void main() {\n"
    "    uint i = gl_GlobalInvocationID.x;\n"
    "    d[i] = s[i];\n"
    "}\n";

static void die(const char *msg) {
    fprintf(stderr, "FATAL: %s (errno=%d %s)\n", msg, errno, strerror(errno));
    exit(1);
}

static void egl_die(const char *msg) {
    fprintf(stderr, "FATAL EGL: %s (egl_err=0x%x)\n", msg, eglGetError());
    exit(1);
}

static GLuint build_program(void) {
    GLuint sh = glCreateShader(GL_COMPUTE_SHADER);
    glShaderSource(sh, 1, &cs_glsl, NULL);
    glCompileShader(sh);

    GLint ok = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[2048] = {0};
        glGetShaderInfoLog(sh, sizeof(log) - 1, NULL, log);
        fprintf(stderr, "shader compile error:\n%s\n", log);
        exit(2);
    }

    GLuint prog = glCreateProgram();
    glAttachShader(prog, sh);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[2048] = {0};
        glGetProgramInfoLog(prog, sizeof(log) - 1, NULL, log);
        fprintf(stderr, "program link error:\n%s\n", log);
        exit(2);
    }
    glDeleteShader(sh);
    return prog;
}

static int find_mali_fd(void) {
    DIR *d = opendir("/proc/self/fd");
    if (!d) return -1;
    int found = -1;
    struct dirent *e;
    while ((e = readdir(d)) != NULL) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        char path[64], target[128];
        snprintf(path, sizeof(path), "/proc/self/fd/%s", e->d_name);
        ssize_t n = readlink(path, target, sizeof(target) - 1);
        if (n <= 0) continue;
        target[n] = 0;
        if (strstr(target, "/dev/mali0")) {
            found = atoi(e->d_name);
            fprintf(stderr, "mali0 fd = %d (%s)\n", found, target);
            break;
        }
    }
    closedir(d);
    return found;
}

/* ------------------------------------------------------------------ */
/* Target GPU VAs for the patcher (0 = no patch)                       */
/* ------------------------------------------------------------------ */

uint64_t g_target_src = 0;
uint64_t g_target_dst = 0;

static void publish_targets_to_patcher(void) {
    uint64_t *psrc = (uint64_t *)dlsym(RTLD_DEFAULT, "mali_patch_src");
    uint64_t *pdst = (uint64_t *)dlsym(RTLD_DEFAULT, "mali_patch_dst");
    if (psrc && pdst) {
        *psrc = g_target_src;
        *pdst = g_target_dst;
        printf("patcher detected: published src=0x%llx dst=0x%llx\n",
               (unsigned long long)g_target_src,
               (unsigned long long)g_target_dst);
    } else {
        printf("patcher NOT loaded — unmodified run\n");
    }
}

/* ------------------------------------------------------------------ */
/* kbase ABI subset                                                     */
/* ------------------------------------------------------------------ */

#define KBASE_IOCTL_TYPE 0x80

struct kbase_ioctl_version_check { __u16 major; __u16 minor; };
#define KBASE_IOCTL_VERSION_CHECK \
    _IOWR(KBASE_IOCTL_TYPE, 0, struct kbase_ioctl_version_check)

struct kbase_ioctl_set_flags { __u32 create_flags; };
#define KBASE_IOCTL_SET_FLAGS \
    _IOW(KBASE_IOCTL_TYPE, 1, struct kbase_ioctl_set_flags)

struct kbase_ioctl_job_submit { __u64 addr; __u32 nr_atoms; __u32 stride; };
#define KBASE_IOCTL_JOB_SUBMIT \
    _IOW(KBASE_IOCTL_TYPE, 2, struct kbase_ioctl_job_submit)

union kbase_ioctl_mem_alloc {
    struct { __u64 va_pages; __u64 commit_pages; __u64 extent; __u64 flags; } in;
    struct { __u64 flags; __u64 gpu_va; } out;
};
#define KBASE_IOCTL_MEM_ALLOC \
    _IOWR(KBASE_IOCTL_TYPE, 5, union kbase_ioctl_mem_alloc)

struct kbase_ioctl_mem_free { __u64 gpu_addr; };
#define KBASE_IOCTL_MEM_FREE \
    _IOW(KBASE_IOCTL_TYPE, 7, struct kbase_ioctl_mem_free)

union kbase_ioctl_mem_import {
    struct { __u64 flags; __u64 phandle; __u32 type; __u32 padding; } in;
    struct { __u64 flags; __u64 gpu_va; __u64 va_pages; } out;
};
#define KBASE_IOCTL_MEM_IMPORT \
    _IOWR(KBASE_IOCTL_TYPE, 22, union kbase_ioctl_mem_import)

struct base_mem_import_user_buffer { __u64 ptr; __u64 length; };
#define BASE_MEM_IMPORT_TYPE_USER_BUFFER  3

#define BASE_MEM_PROT_CPU_RD  (1ull << 0)
#define BASE_MEM_PROT_CPU_WR  (1ull << 1)
#define BASE_MEM_PROT_GPU_RD  (1ull << 2)
#define BASE_MEM_PROT_GPU_WR  (1ull << 3)
#define BASE_MEM_SAME_VA      (1ull << 13)
#define BASE_MEM_SECURE       (1ull << 16)

#define BASE_MEM_MAP_TRACKING_HANDLE  (3ull << 12)

typedef uint32_t base_jd_core_req;
#define BASE_JD_REQ_CS                 ((base_jd_core_req)1 << 1)
#define BASE_JD_REQ_EXTERNAL_RESOURCES ((base_jd_core_req)1 << 8)

struct base_external_resource { uint64_t ext_resource; };
#define BASE_EXT_RES_ACCESS_EXCLUSIVE  1u

typedef uint8_t base_atom_id;
typedef uint8_t base_jd_dep_type;
struct base_dependency { base_atom_id atom_id; base_jd_dep_type dependency_type; };
struct base_jd_udata { uint64_t blob[2]; };
typedef uint8_t base_jd_prio;
#define BASE_JD_PRIO_MEDIUM ((base_jd_prio)0)

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

typedef struct base_jd_event_v2 {
    uint32_t             event_code;
    base_atom_id         atom_number;
    uint8_t              pad[3];
    struct base_jd_udata udata;
} base_jd_event_v2;

#define BASE_JD_EVENT_DONE  0x01

/* ------------------------------------------------------------------ */
/* Bifrost WRITE_VALUE job descriptor                                   */
/* ------------------------------------------------------------------ */

struct mali_job_descriptor_header {
    uint32_t exception_status;
    uint32_t first_incomplete_task;
    uint64_t fault_pointer;
    uint8_t  flags0;   /* bit0: job_desc_size=1; bits1..7: job_type */
    uint8_t  flags1;
    uint16_t job_index;
    uint16_t job_dependency_index_1;
    uint16_t job_dependency_index_2;
    uint64_t next_job;
} __attribute__((packed));

#define MALI_JOB_TYPE_WRITE_VALUE  2

struct mali_payload_write_value {
    uint64_t address;
    uint32_t type;
    uint32_t reserved;
    uint64_t immediate;
} __attribute__((packed));

#define MALI_WRITE_VALUE_TYPE_IMMEDIATE_32  6

/* ------------------------------------------------------------------ */
/* Helpers                                                              */
/* ------------------------------------------------------------------ */

static void hex16(const char *tag, const volatile void *p)
{
    const volatile uint8_t *b = (const volatile uint8_t *)p;
    printf("%s @ %p:", tag, (void *)p);
    for (int i = 0; i < 16; i++)
        printf(" %02x", b[i]);
    printf("\n");
}

/* ------------------------------------------------------------------ */
/* kbase operations                                                     */
/* ------------------------------------------------------------------ */

static int mali_open_and_handshake(void)
{
    int fd = open("/dev/mali0", O_RDWR | O_CLOEXEC);
    if (fd < 0) { fprintf(stderr, "open(/dev/mali0): %s\n", strerror(errno)); return -1; }

    struct kbase_ioctl_version_check v = { .major = 11, .minor = 13 };
    if (ioctl(fd, KBASE_IOCTL_VERSION_CHECK, &v) < 0) {
        fprintf(stderr, "VERSION_CHECK: %s\n", strerror(errno)); close(fd); return -1;
    }
    printf("kbase ABI = %u.%u\n", v.major, v.minor);

    if (mmap(NULL, PAGE_SZ_4K, PROT_NONE, MAP_SHARED, fd,
             BASE_MEM_MAP_TRACKING_HANDLE) == MAP_FAILED) {
        fprintf(stderr, "mmap(tracking page): %s\n", strerror(errno)); close(fd); return -1;
    }

    struct kbase_ioctl_set_flags sf = { .create_flags = 0 };
    if (ioctl(fd, KBASE_IOCTL_SET_FLAGS, &sf) < 0) {
        fprintf(stderr, "SET_FLAGS: %s\n", strerror(errno)); close(fd); return -1;
    }
    return fd;
}

static int mali_import_user_buffer(int fd, void *host_ptr, size_t length,
                                   uint64_t *gpu_va_out, void **cookie_map_out)
{
    struct base_mem_import_user_buffer ub = {
        .ptr    = (uint64_t)(uintptr_t)host_ptr,
        .length = length,
    };
    union kbase_ioctl_mem_import imp;
    memset(&imp, 0, sizeof(imp));
    imp.in.flags   = BASE_MEM_PROT_CPU_RD |
                     BASE_MEM_PROT_GPU_RD | BASE_MEM_PROT_GPU_WR;
    imp.in.phandle = (uint64_t)(uintptr_t)&ub;
    imp.in.type    = BASE_MEM_IMPORT_TYPE_USER_BUFFER;

    if (ioctl(fd, KBASE_IOCTL_MEM_IMPORT, &imp) < 0) {
        fprintf(stderr, "MEM_IMPORT: %s\n", strerror(errno)); return -1;
    }

    uint64_t cookie   = imp.out.gpu_va;
    uint64_t va_pages = imp.out.va_pages;

    void *cm = mmap(NULL, (size_t)(va_pages << PAGE_SHIFT_4K),
                    PROT_READ, MAP_SHARED, fd, (off_t)cookie);
    if (cm == MAP_FAILED) {
        fprintf(stderr, "mmap(import cookie=%#llx): %s\n",
                (unsigned long long)cookie, strerror(errno));
        struct kbase_ioctl_mem_free mf = { .gpu_addr = cookie };
        ioctl(fd, KBASE_IOCTL_MEM_FREE, &mf);
        return -1;
    }

    *gpu_va_out     = (uint64_t)(uintptr_t)cm;
    *cookie_map_out = cm;
    return 0;
}

static void *mali_alloc_page(int fd, uint64_t *gpu_va_out)
{
    union kbase_ioctl_mem_alloc a;
    memset(&a, 0, sizeof(a));
    a.in.va_pages     = 1;
    a.in.commit_pages = 1;
    a.in.extent       = 0;
    a.in.flags        = BASE_MEM_PROT_CPU_RD | BASE_MEM_PROT_CPU_WR |
                        BASE_MEM_PROT_GPU_RD | BASE_MEM_PROT_GPU_WR |
                        BASE_MEM_SAME_VA;

    if (ioctl(fd, KBASE_IOCTL_MEM_ALLOC, &a) < 0) {
        fprintf(stderr, "MEM_ALLOC: %s\n", strerror(errno)); return NULL;
    }

    uint64_t cookie = a.out.gpu_va;
    void *cpu = mmap(NULL, PAGE_SZ_4K, PROT_READ | PROT_WRITE,
                     MAP_SHARED, fd, (off_t)cookie);
    if (cpu == MAP_FAILED) {
        fprintf(stderr, "mmap(cookie=%#llx): %s\n",
                (unsigned long long)cookie, strerror(errno));
        struct kbase_ioctl_mem_free mf = { .gpu_addr = cookie };
        ioctl(fd, KBASE_IOCTL_MEM_FREE, &mf);
        return NULL;
    }

    *gpu_va_out = (uint64_t)(uintptr_t)cpu;
    return cpu;
}

static void mali_free_page(int fd, void *cpu, uint64_t gpu_va)
{
    (void)fd; (void)gpu_va;
    if (cpu) munmap(cpu, PAGE_SZ_4K);
}

static void build_write_value_job(void *job_page, uint64_t target, uint32_t value)
{
    memset(job_page, 0,
           sizeof(struct mali_job_descriptor_header) +
           sizeof(struct mali_payload_write_value));

    struct mali_job_descriptor_header *h = job_page;
    h->flags0    = (MALI_JOB_TYPE_WRITE_VALUE << 1) | 1;
    h->job_index = 1;
    h->next_job  = 0;

    struct mali_payload_write_value *p =
        (struct mali_payload_write_value *)
        ((uint8_t *)job_page + sizeof(struct mali_job_descriptor_header));
    p->address   = target;
    p->type      = MALI_WRITE_VALUE_TYPE_IMMEDIATE_32;
    p->reserved  = 0;
    p->immediate = value;
}

static int mali_submit_and_wait(int fd, uint64_t jc_gpu_va, uint64_t target_gpu_va)
{
    struct base_external_resource extres = {
        .ext_resource = (target_gpu_va & ~0xFFFull) | BASE_EXT_RES_ACCESS_EXCLUSIVE,
    };

    base_jd_atom_v2 atom;
    memset(&atom, 0, sizeof(atom));
    atom.jc          = jc_gpu_va;
    atom.nr_extres   = 1;
    atom.extres_list = (uintptr_t)&extres;
    atom.atom_number = 1;
    atom.prio        = BASE_JD_PRIO_MEDIUM;
    atom.core_req    = BASE_JD_REQ_CS | BASE_JD_REQ_EXTERNAL_RESOURCES;

    struct kbase_ioctl_job_submit sub = {
        .addr     = (uintptr_t)&atom,
        .nr_atoms = 1,
        .stride   = sizeof(atom),
    };
    if (ioctl(fd, KBASE_IOCTL_JOB_SUBMIT, &sub) < 0) {
        fprintf(stderr, "JOB_SUBMIT: %s\n", strerror(errno));
        return -1;
    }

    /*
     * EGL opens /dev/mali0 with O_NONBLOCK, so a bare read() would return
     * EAGAIN immediately.  poll() waits without touching the fd flags that
     * the blob relies on for its own non-blocking I/O.
     */
    struct pollfd pfd = { .fd = fd, .events = POLLIN };
    int pr;
    do { pr = poll(&pfd, 1, 10000); } while (pr < 0 && errno == EINTR);
    if (pr <= 0) {
        fprintf(stderr, pr < 0 ? "poll(event): %s\n" : "poll(event): timeout\n",
                strerror(errno));
        return -1;
    }

    base_jd_event_v2 ev;
    ssize_t n = read(fd, &ev, sizeof(ev));
    if (n != (ssize_t)sizeof(ev)) {
        fprintf(stderr, "read(event): n=%zd %s\n", n, strerror(errno));
        return -1;
    }
    printf("event: atom=%u code=0x%x (%s)\n",
           ev.atom_number, ev.event_code,
           ev.event_code == BASE_JD_EVENT_DONE ? "DONE" : "FAULT");
    return ev.event_code == BASE_JD_EVENT_DONE ? 0 : -1;
}

#ifdef HAVE_INTERCEPT_UAPI
static int mali_import_phys(int fd, uint64_t phys, uint64_t length,
                             uint64_t *gpu_va_out, size_t *map_size_out)
{
    union mali_intercept_import_phys p;
    memset(&p, 0, sizeof(p));
    p.in.phys_addr = phys;
    p.in.length    = length;
#if USE_SECURE_IMPORT
    p.in.flags = BASE_MEM_SECURE | BASE_MEM_PROT_GPU_RD | BASE_MEM_PROT_GPU_WR;
#else
    p.in.flags = BASE_MEM_PROT_CPU_RD | BASE_MEM_PROT_GPU_RD | BASE_MEM_PROT_GPU_WR;
#endif

    if (ioctl(fd, MALI_INTERCEPT_IOCTL_IMPORT_PHYS, &p) < 0) {
        fprintf(stderr, "IMPORT_PHYS: %s\n", strerror(errno)); return -1;
    }
    printf("IMPORT_PHYS: gpu_va=%#llx va_pages=%llu flags=%#llx\n",
           (unsigned long long)p.out.gpu_va,
           (unsigned long long)p.out.va_pages,
           (unsigned long long)p.out.flags);
    *gpu_va_out   = p.out.gpu_va;
    *map_size_out = (size_t)(p.out.va_pages << PAGE_SHIFT_4K);
    return 0;
}
#endif /* HAVE_INTERCEPT_UAPI */

/* ------------------------------------------------------------------ */
/* Static buffer to import into GPU AS                                  */
/* ------------------------------------------------------------------ */

static uint64_t __attribute__((aligned(4096)))
    Buff[PAGE_SZ_4K / sizeof(uint64_t)];

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */

int main(void) {
    /* ---- EGL/GLES setup ---- */
    EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy == EGL_NO_DISPLAY) egl_die("eglGetDisplay");
    if (!eglInitialize(dpy, NULL, NULL)) egl_die("eglInitialize");
    eglBindAPI(EGL_OPENGL_ES_API);

    static const EGLint cfg_attrs[] = {
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT_KHR,
        EGL_SURFACE_TYPE,    EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_RED_SIZE, 8,
        EGL_NONE
    };
    EGLConfig cfg; EGLint nc = 0;
    if (!eglChooseConfig(dpy, cfg_attrs, &cfg, 1, &nc) || nc < 1)
        egl_die("eglChooseConfig");

    static const EGLint pbuf_attrs[] = { EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE };
    EGLSurface surf = eglCreatePbufferSurface(dpy, cfg, pbuf_attrs);
    if (surf == EGL_NO_SURFACE) egl_die("eglCreatePbufferSurface");

    static const EGLint ctx_attrs[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctx_attrs);
    if (ctx == EGL_NO_CONTEXT) egl_die("eglCreateContext");

    if (!eglMakeCurrent(dpy, surf, surf, ctx)) egl_die("eglMakeCurrent");

    printf("GL_VENDOR   : %s\n", glGetString(GL_VENDOR));
    printf("GL_RENDERER : %s\n", glGetString(GL_RENDERER));
    printf("GL_VERSION  : %s\n", glGetString(GL_VERSION));

    /* Use the fd EGL opened — same GPU AS as all GLES allocations. */
    int fd = find_mali_fd();
    if (fd < 0) die("find_mali_fd");
    /* Alternative: open own fd with full handshake (separate AS):
     *   fd = mali_open_and_handshake(); */

    /* ---- GLES buffers ---- */
    uint32_t *init = (uint32_t *)malloc(PAGE_SZ_4K);
    if (!init) die("malloc init");
    for (uint32_t i = 0; i < N_U32; i++)
        init[i] = 0xAABB0000u | (uint16_t)i;

    GLuint bos[2];
    glGenBuffers(2, bos);

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bos[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, PAGE_SZ_4K, init, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bos[0]);

    uint32_t *zero = (uint32_t *)malloc(PAGE_SZ_4K);
    for (uint32_t i = 0; i < N_U32; i++) zero[i] = 0xDEADBEEFu;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bos[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, PAGE_SZ_4K, zero, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bos[1]);
    free(zero);

    /* ---- GLES compute dispatch ---- */
    GLuint prog = build_program();
    glUseProgram(prog);

    publish_targets_to_patcher();

    fprintf(stderr, "dispatching: %u work items in %u groups\n",
            (unsigned)N_U32, (unsigned)(N_U32 / 64));
    glDispatchCompute(N_U32 / 64, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);
    glFinish();

    /* ---- Direct kbase: WRITE_VALUE job into Buff[] ---- */
    uint64_t data_gpu = 0;
    void *import_cookie_map = NULL;
    if (mali_import_user_buffer(fd, Buff, sizeof(Buff),
                                &data_gpu, &import_cookie_map) < 0) {
        close(fd); return 1;
    }
    printf("imported Buff[%zu] (cpu=%p) -> gpu_va=%#llx\n",
           sizeof(Buff), (void *)Buff, (unsigned long long)data_gpu);

    uint64_t job_gpu = 0;
    void *job = mali_alloc_page(fd, &job_gpu);
    if (!job) {
        munmap(import_cookie_map, sizeof(Buff));
        struct kbase_ioctl_mem_free mf = { .gpu_addr = data_gpu };
        ioctl(fd, KBASE_IOCTL_MEM_FREE, &mf);
        close(fd); return 1;
    }

    build_write_value_job(job, data_gpu, 0x12345678u);

    printf("submit: jc=%#llx -> [%#llx] = 0x12345678\n",
           (unsigned long long)job_gpu, (unsigned long long)data_gpu);

    int rc = mali_submit_and_wait(fd, job_gpu, data_gpu);

    hex16("AFTER ", Buff);
    uint32_t v;
    memcpy(&v, Buff, sizeof(v));
    printf("write_value result: 0x%08x %s\n", v,
           v == 0x12345678u ? "[OK]" : "[MISMATCH]");

    mali_free_page(fd, job, job_gpu);
    munmap(import_cookie_map, sizeof(Buff));

    /* ---- GLES verify ---- */
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bos[1]);
    void *map = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, PAGE_SZ_4K,
                                 GL_MAP_READ_BIT);
    if (!map) { fprintf(stderr, "glMapBufferRange failed: 0x%x\n", glGetError()); return 3; }

    int ok = memcmp(map, init, PAGE_SZ_4K) == 0;
    printf("compute copy: %s\n", ok ? "OK" : "MISMATCH");
    uint32_t *got = (uint32_t *)map;
    for (int i = 0; i < 8; i++)
        printf("  dst[%d] = 0x%08x  (expected 0x%08x)\n", i, got[i], init[i]);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    /* ---- Cleanup ---- */
    glDeleteBuffers(2, bos);
    glDeleteProgram(prog);
    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(dpy, ctx);
    eglDestroySurface(dpy, surf);
    eglTerminate(dpy);
    free(init);
    return (rc == 0 && ok) ? 0 : 4;
}
