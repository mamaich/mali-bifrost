/*
 * mali_write_value_demo.c
 *
 * Demonstrates how to:
 *   1. open /dev/mali0 (kbase) on a Bifrost device (S905X2/X3, S922X, A311D);
 *   2. expose a static, page-aligned CPU array (Buff[]) to the GPU through
 *      KBASE_IOCTL_MEM_IMPORT (type = USER_BUFFER);
 *   3. print first 16 bytes of Buff[];
 *   4. build a WRITE_VALUE job-chain that writes 0x12345678 into the first
 *      4 bytes of the imported region and submit it via KBASE_IOCTL_JOB_SUBMIT;
 *   5. wait for completion by read()'ing a base_jd_event_v2 from /dev/mali0;
 *   6. print first 16 bytes of Buff[] again.
 *
 * Build (Android NDK, aarch64):
 *   $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android30-clang \
 *       -DSHELL -O2 -static mali_write_value_demo.c -o mali_gpu_wr
 *
 * The program speaks ABI version 11.x (UK 11.13 in this tree).
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

/*
 * mali_base_jm_kernel.h is included first because it and mali.h both define
 * BASE_MEM_MAP_TRACKING_HANDLE to the same value; the undef below suppresses
 * the harmless redefinition warning.
 */
#include "mali_base_jm_kernel.h"
#undef BASE_MEM_MAP_TRACKING_HANDLE
#include "mali.h"
#include "midgard.h"

/* ------------------------------------------------------------------ */
/* base_external_resource is not exposed in the provided headers       */
/* ------------------------------------------------------------------ */

struct base_external_resource {
	uint64_t ext_resource;
};
#define BASE_EXT_RES_ACCESS_SHARED    0u
#define BASE_EXT_RES_ACCESS_EXCLUSIVE 1u

/* ------------------------------------------------------------------ */

#define PAGE_SHIFT_4K 12
#define PAGE_SZ_4K    (1u << PAGE_SHIFT_4K)

static void hex16(const char *tag, const volatile void *p)
{
	const volatile uint8_t *b = (const volatile uint8_t *)p;
	printf("%s @ %p:", tag, (void *)p);
	for (int i = 0; i < 16; i++)
		printf(" %02x", b[i]);
	printf("\n");
}

static int mali_open_and_handshake(void)
{
	int fd = open("/dev/mali0", O_RDWR | O_CLOEXEC);
	if (fd < 0) {
		fprintf(stderr, "open(/dev/mali0): %s\n", strerror(errno));
		return -1;
	}

	struct kbase_ioctl_version_check v = { .major = 11, .minor = 13 };
	if (ioctl(fd, KBASE_IOCTL_VERSION_CHECK, &v) < 0) {
		fprintf(stderr, "VERSION_CHECK: %s\n", strerror(errno));
		close(fd);
		return -1;
	}
	printf("kbase ABI = %u.%u\n", v.major, v.minor);

	/* The kernel refuses any other ioctls until the "tracking page"
	 * has been mmap'd.  Its only purpose is to anchor the kbase context
	 * to current->mm. */
	if (mmap(NULL, PAGE_SZ_4K, PROT_NONE, MAP_SHARED, fd,
		 BASE_MEM_MAP_TRACKING_HANDLE) == MAP_FAILED) {
		fprintf(stderr, "mmap(tracking page): %s\n", strerror(errno));
		close(fd);
		return -1;
	}

	struct kbase_ioctl_set_flags sf = { .create_flags = 0 };
	if (ioctl(fd, KBASE_IOCTL_SET_FLAGS, &sf) < 0) {
		fprintf(stderr, "SET_FLAGS: %s\n", strerror(errno));
		close(fd);
		return -1;
	}
	return fd;
}

/*
 * Import an existing CPU buffer into the GPU address space.
 *
 * Returns 0 on success and fills *gpu_va_out with the GPU virtual address.
 */
static int mali_import_user_buffer(int fd, void *host_ptr, size_t length,
				   uint64_t *gpu_va_out,
				   void **cookie_map_out)
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
		fprintf(stderr, "MEM_IMPORT: %s\n", strerror(errno));
		return -1;
	}

	uint64_t cookie   = imp.out.gpu_va;
	uint64_t va_pages = imp.out.va_pages;

	void *cm = mmap(NULL, va_pages << PAGE_SHIFT_4K,
			PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)cookie);
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

/*
 * Allocate one VA-page (4 KiB) with SAME_VA semantics and mmap() it.
 *
 * Returns the CPU pointer (which equals the GPU virtual address) and stores
 * the GPU VA in *gpu_va_out.
 */
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
		fprintf(stderr, "MEM_ALLOC: %s\n", strerror(errno));
		return NULL;
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

	*gpu_va_out = (uint64_t)(uintptr_t)cpu; /* SAME_VA */
	return cpu;
}

static void mali_free_page(int fd, void *cpu, uint64_t gpu_va)
{
	if (cpu)
		munmap(cpu, PAGE_SZ_4K);
	if (gpu_va) {
		struct kbase_ioctl_mem_free mf = { .gpu_addr = gpu_va };
		ioctl(fd, KBASE_IOCTL_MEM_FREE, &mf);
	}
}

/* Build a single WRITE_VALUE job that writes `value` (32-bit) to `target`.
 * The descriptor is laid out at the beginning of `job_page`. */
static void build_write_value_job(void *job_page, uint64_t target,
				  uint32_t value)
{
	memset(job_page, 0, MALI_WRITE_VALUE_JOB_LENGTH);

	pan_section_pack(job_page, WRITE_VALUE_JOB, HEADER, hdr) {
		hdr.type  = MALI_JOB_TYPE_WRITE_VALUE;
		hdr.index = 1;   /* must be non-zero */
		hdr.next  = 0;   /* end of chain */
	}

	pan_section_pack(job_page, WRITE_VALUE_JOB, PAYLOAD, pl) {
		pl.address         = target;
		pl.type            = MALI_WRITE_VALUE_TYPE_IMMEDIATE_32;
		pl.immediate_value = value;
	}
}

static int mali_submit_and_wait(int fd, uint64_t jc_gpu_va,
				uint64_t target_gpu_va)
{
	/* The imported region is referenced as an external resource so that
	 * the kernel pins Buff[]'s pages and maps them in the GPU MMU for
	 * the duration of this atom. */
	struct base_external_resource extres = {
		.ext_resource = (target_gpu_va & ~0xFFFull) |
		                BASE_EXT_RES_ACCESS_EXCLUSIVE,
	};

	base_jd_atom_v2 atom;
	memset(&atom, 0, sizeof(atom));
	atom.jc          = jc_gpu_va;
	atom.nr_extres   = 1;
	atom.extres_list = (uintptr_t)&extres;
	atom.atom_number = 1;             /* arbitrary, must be unique */
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

	struct base_jd_event_v2 ev;
	ssize_t n = read(fd, &ev, sizeof(ev));
	if (n != (ssize_t)sizeof(ev)) {
		fprintf(stderr, "read(event): n=%zd %s\n", n, strerror(errno));
		return -1;
	}
	printf("event: atom=%u code=0x%x (%s)\n",
	       ev.atom_number, (unsigned)ev.event_code,
	       ev.event_code == BASE_JD_EVENT_DONE ? "DONE" : "FAULT");
	return ev.event_code == BASE_JD_EVENT_DONE ? 0 : -1;
}

/*
 * Static, page-aligned, page-sized array exposed to the GPU as an
 * imported user buffer.
 */
static uint64_t __attribute__((aligned(4096)))
	Buff[4096 / sizeof(uint64_t)];

int main(void)
{
	int fd = mali_open_and_handshake();
	if (fd < 0)
		return 1;

	/* Pre-fill with a recognisable pattern so the BEFORE/AFTER dump is
	 * unambiguous. */
	memset(Buff, 0xAA, 16);
	hex16("BEFORE", Buff);

	/* Hand Buff[] to the GPU. */
	uint64_t data_gpu = 0;
	void *import_cookie_map = NULL;
	if (mali_import_user_buffer(fd, Buff, sizeof(Buff),
				    &data_gpu, &import_cookie_map) < 0) {
		close(fd);
		return 1;
	}
	printf("imported Buff[%zu] (cpu=%p) -> gpu_va=%#llx\n",
	       sizeof(Buff), (void *)Buff, (unsigned long long)data_gpu);

	/* Job descriptor lives in a normal SAME_VA allocation. */
	uint64_t job_gpu = 0;
	void *job = mali_alloc_page(fd, &job_gpu);
	if (!job) {
		munmap(import_cookie_map, sizeof(Buff));
		struct kbase_ioctl_mem_free mf = { .gpu_addr = data_gpu };
		ioctl(fd, KBASE_IOCTL_MEM_FREE, &mf);
		close(fd);
		return 1;
	}

	build_write_value_job(job, data_gpu, 0x12345678u);

	printf("submit: jc=%#llx -> [%#llx] = 0x12345678\n",
	       (unsigned long long)job_gpu, (unsigned long long)data_gpu);

	int rc = mali_submit_and_wait(fd, job_gpu, data_gpu);

	hex16("AFTER ", Buff);

	uint32_t v;
	memcpy(&v, Buff, sizeof(v));
	printf("result: first u32 = 0x%08x %s\n", v,
	       v == 0x12345678u ? "[OK]" : "[MISMATCH]");

	mali_free_page(fd, job, job_gpu);

	/* Release the import. */
	munmap(import_cookie_map, sizeof(Buff));
	struct kbase_ioctl_mem_free mf = { .gpu_addr = data_gpu };
	ioctl(fd, KBASE_IOCTL_MEM_FREE, &mf);

	close(fd);
	return rc;
}
