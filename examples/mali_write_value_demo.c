/*
 * mali_write_value_demo.c
 *
 * Demonstrates how to:
 *   1. open /dev/mali0 (kbase) on a Bifrost device (S905X2/X3, S922X, A311D);
 *   2. expose a static, page-aligned CPU array (Buff[]) to the GPU through
 *      KBASE_IOCTL_MEM_IMPORT (type = USER_BUFFER).  No GPU page is
 *      allocated for the data — the kernel pins the existing user pages
 *      and maps them into the GPU MMU for the lifetime of the job;
 *   3. print first 16 bytes of Buff[];
 *   4. build a WRITE_VALUE job-chain that writes 0x12345678 into the first
 *      4 bytes of the imported region and submit it to the GPU through
 *      KBASE_IOCTL_JOB_SUBMIT.  The atom uses BASE_JD_REQ_EXTERNAL_RESOURCES
 *      so the kernel pins Buff[]'s pages and inserts them in the GPU MMU
 *      for the duration of the job;
 *   5. wait for completion by read()'ing a base_jd_event_v2 from /dev/mali0;
 *   6. print first 16 bytes of Buff[] again.
 *
 * Build (Android NDK, aarch64):
 *   $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang \
 *       -Wall -O2 mali_write_value_demo.c -o mali_demo
 *
 * The program speaks ABI version 11.x (UK 11.13 in this tree).
 *
 * Notes on portability of the WRITE_VALUE job descriptor
 * ------------------------------------------------------
 * The kbase driver does not parse job-chain descriptors itself, it only
 * passes the GPU VA of the chain to the job-manager hardware.  The exact
 * layout of the WRITE_VALUE job (job_type = 2) and the values of its
 * "write value type" field are taken from public Mesa/panfrost XML for the
 * v6 (Bifrost) hardware generation.  If on a particular silicon revision
 * the job aborts with WRITE_FAULT, try IMMEDIATE_32 == 1 instead of 6 (the
 * older numbering still used in some panfrost branches).
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/ioctl.h>
#include <linux/types.h>

/* ------------------------------------------------------------------ */
/* Subset of the kbase ABI (from mali_kbase_ioctl.h / mali_base_kernel.h) */
/* ------------------------------------------------------------------ */

#define KBASE_IOCTL_TYPE 0x80

struct kbase_ioctl_version_check {
	__u16 major;
	__u16 minor;
};
#define KBASE_IOCTL_VERSION_CHECK \
	_IOWR(KBASE_IOCTL_TYPE, 0, struct kbase_ioctl_version_check)

struct kbase_ioctl_set_flags {
	__u32 create_flags;
};
#define KBASE_IOCTL_SET_FLAGS \
	_IOW(KBASE_IOCTL_TYPE, 1, struct kbase_ioctl_set_flags)

struct kbase_ioctl_job_submit {
	__u64 addr;
	__u32 nr_atoms;
	__u32 stride;
};
#define KBASE_IOCTL_JOB_SUBMIT \
	_IOW(KBASE_IOCTL_TYPE, 2, struct kbase_ioctl_job_submit)

union kbase_ioctl_mem_alloc {
	struct {
		__u64 va_pages;
		__u64 commit_pages;
		__u64 extent;
		__u64 flags;
	} in;
	struct {
		__u64 flags;
		__u64 gpu_va;
	} out;
};
#define KBASE_IOCTL_MEM_ALLOC \
	_IOWR(KBASE_IOCTL_TYPE, 5, union kbase_ioctl_mem_alloc)

struct kbase_ioctl_mem_free {
	__u64 gpu_addr;
};
#define KBASE_IOCTL_MEM_FREE \
	_IOW(KBASE_IOCTL_TYPE, 7, struct kbase_ioctl_mem_free)

union kbase_ioctl_mem_import {
	struct {
		__u64 flags;
		__u64 phandle;
		__u32 type;
		__u32 padding;
	} in;
	struct {
		__u64 flags;
		__u64 gpu_va;
		__u64 va_pages;
	} out;
};
#define KBASE_IOCTL_MEM_IMPORT \
	_IOWR(KBASE_IOCTL_TYPE, 22, union kbase_ioctl_mem_import)

/* phandle payload for type = USER_BUFFER */
struct base_mem_import_user_buffer {
	__u64 ptr;
	__u64 length;
};

#define BASE_MEM_IMPORT_TYPE_USER_BUFFER  3

struct kbase_ioctl_mem_sync {
	__u64 handle;
	__u64 user_addr;
	__u64 size;
	__u8  type;
	__u8  padding[7];
};
#define KBASE_IOCTL_MEM_SYNC \
	_IOW(KBASE_IOCTL_TYPE, 15, struct kbase_ioctl_mem_sync)

/* base_mem_alloc_flags */
#define BASE_MEM_PROT_CPU_RD             (1ull << 0)
#define BASE_MEM_PROT_CPU_WR             (1ull << 1)
#define BASE_MEM_PROT_GPU_RD             (1ull << 2)
#define BASE_MEM_PROT_GPU_WR             (1ull << 3)
#define BASE_MEM_SAME_VA                 (1ull << 13)

/* special pgoff handle for the "tracking page" that must be mapped first */
#define BASE_MEM_MAP_TRACKING_HANDLE     (3ull << 12)
/* cookies returned by MEM_ALLOC live in [BASE_MEM_COOKIE_BASE,FIRST_FREE) */
#define BASE_MEM_COOKIE_BASE             (64ul << 12)

/* core_req bits */
typedef uint32_t base_jd_core_req;
#define BASE_JD_REQ_CS                   ((base_jd_core_req)1 << 1)
#define BASE_JD_REQ_EXTERNAL_RESOURCES   ((base_jd_core_req)1 << 8)

/* base_external_resource: gpu_va | access bit (LSB) */
struct base_external_resource {
	uint64_t ext_resource;
};
#define BASE_EXT_RES_ACCESS_SHARED       0u
#define BASE_EXT_RES_ACCESS_EXCLUSIVE    1u

/* atom dependency */
typedef uint8_t base_atom_id;
typedef uint8_t base_jd_dep_type;
#define BASE_JD_DEP_TYPE_INVALID         (0)

struct base_dependency {
	base_atom_id     atom_id;
	base_jd_dep_type dependency_type;
};

struct base_jd_udata {
	uint64_t blob[2];
};

/* base_jd_atom_v2 — the ioctl payload of KBASE_IOCTL_JOB_SUBMIT.
 * Layout must exactly match mali_base_kernel.h. */
typedef uint8_t base_jd_prio;
#define BASE_JD_PRIO_MEDIUM ((base_jd_prio)0)

typedef struct base_jd_atom_v2 {
	uint64_t              jc;             /* GPU VA of the job chain        */
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

/* event returned by read() */
typedef struct base_jd_event_v2 {
	uint32_t            event_code;       /* base_jd_event_code             */
	base_atom_id        atom_number;
	uint8_t             pad[3];
	struct base_jd_udata udata;
} base_jd_event_v2;

#define BASE_JD_EVENT_DONE   0x01

/* ------------------------------------------------------------------ */
/* Mali job-chain descriptor (architecture documentation / Mesa)       */
/* ------------------------------------------------------------------ */

/* The first 32 bytes of every Mali job (when job_descriptor_size == 1, i.e.
 * 64-bit pointers). */
struct mali_job_descriptor_header {
	uint32_t exception_status;
	uint32_t first_incomplete_task;
	uint64_t fault_pointer;
	/* byte 16: job_descriptor_size:1, job_type:7 */
	uint8_t  flags0;
	/* byte 17: job_barrier:1, reserved:7 */
	uint8_t  flags1;
	uint16_t job_index;
	uint16_t job_dependency_index_1;
	uint16_t job_dependency_index_2;
	uint64_t next_job;                    /* 0 = end of chain               */
} __attribute__((packed));

#define MALI_JOB_TYPE_NULL          1
#define MALI_JOB_TYPE_WRITE_VALUE   2
#define MALI_JOB_TYPE_CACHE_FLUSH   3

/* Payload that immediately follows the header for a WRITE_VALUE job. */
struct mali_payload_write_value {
	uint64_t address;                     /* GPU VA to write                */
	uint32_t type;                        /* see MALI_WRITE_VALUE_TYPE_*    */
	uint32_t reserved;
	uint64_t immediate;                   /* value to write                 */
} __attribute__((packed));

/* Bifrost v6 numbering (Mesa genxml/common.xml). */
#define MALI_WRITE_VALUE_TYPE_IMMEDIATE_32  6

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
 * The kernel always returns a "cookie" for a 64-bit non-compat task; we have
 * to mmap() it once to anchor the region to a real GPU virtual address.  The
 * returned CPU VA points at a PROT_NONE mapping (it has no physical pages
 * attached until an atom that references the region as an external resource
 * actually runs) and should NOT be dereferenced — all CPU access must go
 * through the original `host_ptr`.
 *
 * Returns 0 on success and fills *gpu_va_out with the GPU virtual address
 * (which equals the placeholder CPU VA returned by mmap, due to BASE_MEM_
 * NEED_MMAP being forced on by the kernel for 64-bit tasks).
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
	imp.in.flags   = BASE_MEM_PROT_CPU_RD | BASE_MEM_PROT_CPU_WR |
	                 BASE_MEM_PROT_GPU_RD | BASE_MEM_PROT_GPU_WR;
	imp.in.phandle = (uint64_t)(uintptr_t)&ub;
	imp.in.type    = BASE_MEM_IMPORT_TYPE_USER_BUFFER;

	if (ioctl(fd, KBASE_IOCTL_MEM_IMPORT, &imp) < 0) {
		fprintf(stderr, "MEM_IMPORT: %s\n", strerror(errno));
		return -1;
	}

	uint64_t cookie   = imp.out.gpu_va;
	uint64_t va_pages = imp.out.va_pages;

	/* mmap the cookie to convert it into a real GPU VA */
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
 * On success returns the CPU pointer (which is also the GPU virtual address)
 * and stores the GPU VA in *gpu_va_out.
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

	/* For UK >= 10.5 the kernel returns a "cookie" in gpu_va that must be
	 * fed back to mmap() as the offset; the returned CPU address becomes
	 * the real, shared CPU/GPU virtual address. */
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
	memset(job_page, 0, sizeof(struct mali_job_descriptor_header) +
			    sizeof(struct mali_payload_write_value));

	struct mali_job_descriptor_header *h = job_page;
	/* job_descriptor_size = 1 (bit 0), job_type = 2 (bits 1..7) */
	h->flags0     = (MALI_JOB_TYPE_WRITE_VALUE << 1) | 1;
	h->flags1     = 0;
	h->job_index  = 1;          /* must be non-zero */
	h->next_job   = 0;          /* end of chain     */

	struct mali_payload_write_value *p =
		(struct mali_payload_write_value *)((uint8_t *)job_page +
		    sizeof(struct mali_job_descriptor_header));
	p->address   = target;
	p->type      = MALI_WRITE_VALUE_TYPE_IMMEDIATE_32;
	p->reserved  = 0;
	p->immediate = value;
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
	atom.jc              = jc_gpu_va;
	atom.nr_extres       = 1;
	atom.extres_list     = (uintptr_t)&extres;
	atom.atom_number     = 1;             /* arbitrary, must be unique    */
	atom.prio            = BASE_JD_PRIO_MEDIUM;
	atom.core_req        = BASE_JD_REQ_CS |
	                       BASE_JD_REQ_EXTERNAL_RESOURCES;

	struct kbase_ioctl_job_submit sub = {
		.addr     = (uintptr_t)&atom,
		.nr_atoms = 1,
		.stride   = sizeof(atom),
	};
	if (ioctl(fd, KBASE_IOCTL_JOB_SUBMIT, &sub) < 0) {
		fprintf(stderr, "JOB_SUBMIT: %s\n", strerror(errno));
		return -1;
	}

	/* Wait for the completion event. */
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

	/* Hand Buff[] to the GPU.  The kernel pins these very pages and
	 * inserts them in the GPU MMU on demand, when the atom that lists
	 * the region as an external resource starts running. */
	uint64_t data_gpu = 0;
	void *import_cookie_map = NULL;
	if (mali_import_user_buffer(fd, Buff, sizeof(Buff),
				    &data_gpu, &import_cookie_map) < 0) {
		close(fd);
		return 1;
	}
	printf("imported Buff[%zu] (cpu=%p) -> gpu_va=%#llx\n",
	       sizeof(Buff), (void *)Buff, (unsigned long long)data_gpu);

	/* Job descriptor still lives in a normal SAME_VA allocation. */
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

	/* Release the import: drop our CPU placeholder mapping and ask the
	 * kernel to forget about the region. */
	munmap(import_cookie_map, sizeof(Buff));
	struct kbase_ioctl_mem_free mf = { .gpu_addr = data_gpu };
	ioctl(fd, KBASE_IOCTL_MEM_FREE, &mf);

	close(fd);
	return rc;
}
