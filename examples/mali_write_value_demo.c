/*
 * mali_write_value_demo.c
 *
 * Demonstrates how to:
 *   1. open /dev/mali0 (kbase) on a Bifrost device (S905X2/X3, S922X, A311D);
 *   2. allocate one page of GPU memory and map it into the CPU address space
 *      (BASE_MEM_SAME_VA, so the CPU pointer == GPU VA);
 *   3. print first 16 bytes;
 *   4. build a WRITE_VALUE job-chain that writes 0x12345678 into the first
 *      4 bytes of that page and submit it to the GPU through
 *      KBASE_IOCTL_JOB_SUBMIT;
 *   5. wait for completion by read()'ing a base_jd_event_v2 from /dev/mali0;
 *   6. print first 16 bytes again.
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

/* Size of the data buffer the GPU will write into. */
#define ALLOC_SIZE    0x10000000ull   /* 256 MiB */

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
 * Allocate `size` bytes (rounded up to a multiple of 4 KiB) with SAME_VA
 * semantics and mmap() the result.
 *
 * On success returns the CPU pointer (which is also the GPU virtual address)
 * and stores the GPU VA in *gpu_va_out.
 */
static void *mali_alloc(int fd, uint64_t size, uint64_t *gpu_va_out)
{
	uint64_t va_pages = (size + PAGE_SZ_4K - 1) >> PAGE_SHIFT_4K;

	union kbase_ioctl_mem_alloc a;
	memset(&a, 0, sizeof(a));
	a.in.va_pages     = va_pages;
	a.in.commit_pages = va_pages;
	a.in.extent       = 0;
	a.in.flags        = BASE_MEM_PROT_CPU_RD | BASE_MEM_PROT_CPU_WR |
	                    BASE_MEM_PROT_GPU_RD | BASE_MEM_PROT_GPU_WR |
	                    BASE_MEM_SAME_VA;

	if (ioctl(fd, KBASE_IOCTL_MEM_ALLOC, &a) < 0) {
		fprintf(stderr, "MEM_ALLOC(%llu pages): %s\n",
			(unsigned long long)va_pages, strerror(errno));
		return NULL;
	}

	/* For UK >= 10.5 the kernel returns a "cookie" in gpu_va that must be
	 * fed back to mmap() as the offset; the returned CPU address becomes
	 * the real, shared CPU/GPU virtual address. */
	uint64_t cookie = a.out.gpu_va;
	void *cpu = mmap(NULL, va_pages << PAGE_SHIFT_4K,
			 PROT_READ | PROT_WRITE, MAP_SHARED, fd, (off_t)cookie);
	if (cpu == MAP_FAILED) {
		fprintf(stderr, "mmap(cookie=%#llx, %llu pages): %s\n",
			(unsigned long long)cookie,
			(unsigned long long)va_pages, strerror(errno));
		struct kbase_ioctl_mem_free mf = { .gpu_addr = cookie };
		ioctl(fd, KBASE_IOCTL_MEM_FREE, &mf);
		return NULL;
	}

	*gpu_va_out = (uint64_t)(uintptr_t)cpu; /* SAME_VA */
	return cpu;
}

static void mali_free(int fd, void *cpu, uint64_t size, uint64_t gpu_va)
{
	uint64_t va_pages = (size + PAGE_SZ_4K - 1) >> PAGE_SHIFT_4K;
	if (cpu)
		munmap(cpu, va_pages << PAGE_SHIFT_4K);
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

static int mali_submit_and_wait(int fd, uint64_t jc_gpu_va)
{
	base_jd_atom_v2 atom;
	memset(&atom, 0, sizeof(atom));
	atom.jc              = jc_gpu_va;
	atom.atom_number     = 1;             /* arbitrary, must be unique    */
	atom.prio            = BASE_JD_PRIO_MEDIUM;
	atom.core_req        = BASE_JD_REQ_CS; /* vertex/compute/tiler slot 1 */

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

int main(void)
{
	int fd = mali_open_and_handshake();
	if (fd < 0)
		return 1;

	/* Data buffer: GPU will write 0x12345678 here.  ALLOC_SIZE bytes
	 * (rounded up to a page) are allocated and committed up-front. */
	uint64_t data_gpu = 0;
	void *data = mali_alloc(fd, ALLOC_SIZE, &data_gpu);
	if (!data) { close(fd); return 1; }
	printf("data: cpu=%p gpu_va=%#llx size=%#llx\n",
	       data, (unsigned long long)data_gpu,
	       (unsigned long long)ALLOC_SIZE);

	/* Pre-fill with a recognisable pattern so the BEFORE/AFTER dump is
	 * unambiguous. */
	memset(data, 0xAA, 16);
	hex16("BEFORE", data);

	/* Job descriptor — one page is plenty. */
	uint64_t job_gpu = 0;
	void *job = mali_alloc(fd, PAGE_SZ_4K, &job_gpu);
	if (!job) { mali_free(fd, data, ALLOC_SIZE, data_gpu); close(fd); return 1; }

	build_write_value_job(job, data_gpu, 0x12345678u);

	printf("submit: jc=%#llx -> [%#llx] = 0x12345678\n",
	       (unsigned long long)job_gpu, (unsigned long long)data_gpu);

	int rc = mali_submit_and_wait(fd, job_gpu);

	/* Make sure any GPU writes that may have lingered in caches are
	 * visible to the CPU.  We only need to look at the first few bytes
	 * the GPU touched, so a single-page sync is enough. */
	struct kbase_ioctl_mem_sync ms = {
		.handle    = data_gpu,
		.user_addr = (uintptr_t)data,
		.size      = PAGE_SZ_4K,
		.type      = 1,            /* sync_from_device (invalidate) */
	};
	ioctl(fd, KBASE_IOCTL_MEM_SYNC, &ms);

	hex16("AFTER ", data);

	uint32_t v = *(volatile uint32_t *)data;
	printf("result: first u32 = 0x%08x %s\n", v,
	       v == 0x12345678u ? "[OK]" : "[MISMATCH]");

	mali_free(fd, job, PAGE_SZ_4K, job_gpu);
	mali_free(fd, data, ALLOC_SIZE, data_gpu);
	close(fd);
	return rc;
}
