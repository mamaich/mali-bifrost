/*
 * mali_hello_compute.c
 *
 * Минимальный GLES 3.1 compute: копирует одну 4 KiB-страницу uint32 из
 * SSBO_src в SSBO_dst через шейдер вида
 *     dst[gid] = src[gid];
 *
 * Цели:
 *   - валидировать, что libGLES_mali.so на устройстве собирает и
 *     запускает compute-шейдеры;
 *   - подсунуть в src легкоузнаваемые маркеры (0xAABB_<i>), чтобы потом
 *     находить эти данные в дампах GPU-памяти;
 *   - залогировать кое-что про FD /dev/mali0 и про BO, чтобы в логе
 *     observer'а можно было сопоставить GPU VA с буферами.
 *
 * Build (Android NDK, aarch64):
 *   $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang \
 *       -Wall -O2 mali_hello_compute.c -o mali_hello_compute \
 *       -lEGL -lGLESv3
 *
 * Run:
 *   adb push mali_hello_compute /data/local/tmp/
 *   adb shell "cd /data/local/tmp && ./mali_hello_compute"
 */

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PAGE_SZ_4K 4096u
#define N_U32      (PAGE_SZ_4K / sizeof(uint32_t))

/* GLES compute: одна работа на uint32, local_size_x=64.
 * Размещение SSBO 0 (src, readonly) и 1 (dst, writeonly) — std430 */
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

/* Найти текущий FD /dev/mali0 в /proc/self/fd. Чисто диагностика: чтобы
 * понять, какой FD держит blob после eglMakeCurrent. */
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

int main(void) {
    /* ------------------------- EGL/GLES setup --------------------------- */
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
    EGLConfig cfg;
    EGLint nc = 0;
    if (!eglChooseConfig(dpy, cfg_attrs, &cfg, 1, &nc) || nc < 1)
        egl_die("eglChooseConfig");

    static const EGLint pbuf_attrs[] = {
        EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE
    };
    EGLSurface surf = eglCreatePbufferSurface(dpy, cfg, pbuf_attrs);
    if (surf == EGL_NO_SURFACE) egl_die("eglCreatePbufferSurface");

    static const EGLint ctx_attrs[] = {
        EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE
    };
    EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctx_attrs);
    if (ctx == EGL_NO_CONTEXT) egl_die("eglCreateContext");

    if (!eglMakeCurrent(dpy, surf, surf, ctx)) egl_die("eglMakeCurrent");

    printf("GL_VENDOR   : %s\n", glGetString(GL_VENDOR));
    printf("GL_RENDERER : %s\n", glGetString(GL_RENDERER));
    printf("GL_VERSION  : %s\n", glGetString(GL_VERSION));

    find_mali_fd();

    /* ------------------------- Buffers ---------------------------------- */
    /* Маркерный паттерн: верхние 16 бит = 0xAABB (узнаваемая «шапка»),
     * нижние 16 бит — индекс. В дампе будут идти 0xAABB0000, 0xAABB0001,
     * 0xAABB0002, ... — это легко глазами увидеть. */
    uint32_t *init = (uint32_t *)malloc(PAGE_SZ_4K);
    if (!init) die("malloc init");
    for (uint32_t i = 0; i < N_U32; i++)
        init[i] = 0xAABB0000u | (uint16_t)i;

    GLuint bos[2];
    glGenBuffers(2, bos);

    /* src (binding=0) */
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bos[0]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, PAGE_SZ_4K, init, GL_STATIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, bos[0]);

    /* dst (binding=1), пред-заполняем шумом 0xDEADBEEF, чтобы был
     * виден факт перезаписи */
    uint32_t *zero = (uint32_t *)malloc(PAGE_SZ_4K);
    for (uint32_t i = 0; i < N_U32; i++) zero[i] = 0xDEADBEEFu;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bos[1]);
    glBufferData(GL_SHADER_STORAGE_BUFFER, PAGE_SZ_4K, zero, GL_DYNAMIC_COPY);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, bos[1]);
    free(zero);

    /* ------------------------- Build + dispatch ------------------------- */
    GLuint prog = build_program();
    glUseProgram(prog);

    /* 4 KiB / 4 = 1024 work items, local_size_x=64 -> 16 workgroups */
    fprintf(stderr, "dispatching: %u work items in %u groups\n",
            (unsigned)N_U32, (unsigned)(N_U32 / 64));
    glDispatchCompute(N_U32 / 64, 1, 1);
    glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT |
                    GL_SHADER_STORAGE_BARRIER_BIT);
    glFinish();

    /* ------------------------- Verify ----------------------------------- */
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, bos[1]);
    void *map = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, PAGE_SZ_4K,
                                 GL_MAP_READ_BIT);
    if (!map) { fprintf(stderr, "glMapBufferRange failed: 0x%x\n", glGetError()); return 3; }

    int ok = memcmp(map, init, PAGE_SZ_4K) == 0;
    printf("compute copy: %s\n", ok ? "OK" : "MISMATCH");

    /* первые 8 элементов для глаз */
    uint32_t *got = (uint32_t *)map;
    for (int i = 0; i < 8; i++) printf("  dst[%d] = 0x%08x  (expected 0x%08x)\n",
                                       i, got[i], init[i]);

    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

    /* ------------------------- Cleanup ---------------------------------- */
    glDeleteBuffers(2, bos);
    glDeleteProgram(prog);
    eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(dpy, ctx);
    eglDestroySurface(dpy, surf);
    eglTerminate(dpy);
    free(init);
    return ok ? 0 : 4;
}
