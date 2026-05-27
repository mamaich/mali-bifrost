/*
 * mali_phys_import_demo.c — тестовая утилита для
 * MALI_INTERCEPT_IOCTL_IMPORT_PHYS.
 *
 * Делает почти то же самое, что mali_write_value_demo.c из вложения:
 *   1. открывает /dev/mali0 и проходит handshake (VERSION_CHECK,
 *      mmap tracking page, SET_FLAGS);
 *   2. вместо импорта пользовательского массива Buff[] через
 *      KBASE_IOCTL_MEM_IMPORT type=USER_BUFFER импортирует
 *      демонстрационный физический буфер DEMO_PHYS_ADDR/DEMO_LENGTH
 *      через MALI_INTERCEPT_IOCTL_IMPORT_PHYS, добавленный в
 *      mali_intercept.ko. IOCTL возвращает СРАЗУ готовый gpu_va —
 *      шаг mmap(cookie) делается изнутри ядра через vm_mmap;
 *   3. собирает WRITE_VALUE job-chain (как в исходнике), который
 *      пытается записать 0x12345678 в первые 4 байта импортированной
 *      области;
 *   4. отправляет атом через KBASE_IOCTL_JOB_SUBMIT и ждёт события.
 *
 * ВАЖНО: DEMO_PHYS_ADDR — фейковый адрес. На реальной системе его
 * должен предоставить специализированный драйвер. С фейковым адресом
 * GPU почти наверняка вернёт MMU/WRITE_FAULT — это нормально, тест
 * проверяет цепочку IOCTL → dma_buf → импорт → mmap → подача job,
 * а не корректность самой памяти.
 *
 * Дамп памяти до/после (BEFORE/AFTER), как в оригинальном demo,
 * здесь УБРАН — мы не должны CPU-читать память, владелец которой —
 * спецдрайвер (и для фейкового адреса это сразу свалит процесс).
 *
 * Сборка (Android NDK, aarch64) — путь до NDK и triple подставьте свои:
 *   aarch64-linux-gnu-gcc -Wall -O2 mali_phys_import_demo.c \
 *       -I.. -o mali_phys_import_demo
 *
 * Запуск на устройстве (с правами root, после insmod mali_intercept.ko):
 *   ./mali_phys_import_demo
 */

#define _FILE_OFFSET_BITS 64
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

/*
 * Демонстрационные значения. На реальной системе передаются от
 * специализированного драйвера. Должны быть выровнены на страницу и
 * длина кратна PAGE_SIZE (для 4K-страниц).
 */
#define DEMO_PHYS_ADDR  0x12345000ULL
#define DEMO_LENGTH     0x1000ULL

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

/* base_mem_alloc_flags */
#define BASE_MEM_PROT_CPU_RD             (1ull << 0)
#define BASE_MEM_PROT_CPU_WR             (1ull << 1)
#define BASE_MEM_PROT_GPU_RD             (1ull << 2)
#define BASE_MEM_PROT_GPU_WR             (1ull << 3)
#define BASE_MEM_SAME_VA                 (1ull << 13)

/* special pgoff handle for the "tracking page" that must be mapped first */
#define BASE_MEM_MAP_TRACKING_HANDLE     (3ull << 12)

/* core_req bits */
typedef uint32_t base_jd_core_req;
#define BASE_JD_REQ_CS                   ((base_jd_core_req)1 << 1)
#define BASE_JD_REQ_EXTERNAL_RESOURCES   ((base_jd_core_req)1 << 8)

struct base_external_resource {
	uint64_t ext_resource;
};
#define BASE_EXT_RES_ACCESS_EXCLUSIVE    1u

typedef uint8_t  base_atom_id;
typedef uint8_t  base_jd_dep_type;
struct base_dependency {
	base_atom_id     atom_id;
	base_jd_dep_type dependency_type;
};
struct base_jd_udata {
	uint64_t blob[2];
};

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
	uint32_t            event_code;
	base_atom_id        atom_number;
	uint8_t             pad[3];
	struct base_jd_udata udata;
} base_jd_event_v2;

#define BASE_JD_EVENT_DONE   0x01

/* ------------------------------------------------------------------ */
/* Mali job-chain descriptor (Bifrost v6, from Mesa/panfrost XML)      */
/* ------------------------------------------------------------------ */

struct mali_job_descriptor_header {
	uint32_t exception_status;
	uint32_t first_incomplete_task;
	uint64_t fault_pointer;
	uint8_t  flags0;             /* bit0: job_desc_size, bits1..7: type */
	uint8_t  flags1;
	uint16_t job_index;
	uint16_t job_dependency_index_1;
	uint16_t job_dependency_index_2;
	uint64_t next_job;
} __attribute__((packed));

#define MALI_JOB_TYPE_WRITE_VALUE   2

struct mali_payload_write_value {
	uint64_t address;
	uint32_t type;
	uint32_t reserved;
	uint64_t immediate;
} __attribute__((packed));

#define MALI_WRITE_VALUE_TYPE_IMMEDIATE_32  6

/* ------------------------------------------------------------------ */
/* Наш UAPI                                                            */
/* ------------------------------------------------------------------ */

#include "../mali_intercept_uapi.h"

/* ------------------------------------------------------------------ */

#define PAGE_SHIFT_4K 12
#define PAGE_SZ_4K    (1u << PAGE_SHIFT_4K)

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

	/* Tracking page обязан быть замаплен до любого другого IOCTL —
	 * привязывает kbase_context к current->mm. */
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
 * Импорт физической памяти через MALI_INTERCEPT_IOCTL_IMPORT_PHYS.
 *
 * Возвращает 0 и заполняет *gpu_va_out / *map_size_out. IOCTL уже сам
 * сделал mmap внутри ядра, поэтому gpu_va_out — готовый GPU VA (для
 * 64-битной задачи он же CPU VA, SAME_VA). При очистке нужно сделать
 * munmap(gpu_va, map_size) и KBASE_IOCTL_MEM_FREE(gpu_va).
 */
static int mali_import_phys(int fd, uint64_t phys, uint64_t length,
			    uint64_t *gpu_va_out,
			    size_t *map_size_out)
{
	union mali_intercept_import_phys p;
	memset(&p, 0, sizeof(p));
	p.in.phys_addr = phys;
	p.in.length    = length;
	p.in.flags     = BASE_MEM_PROT_CPU_RD |
	                 BASE_MEM_PROT_GPU_RD | BASE_MEM_PROT_GPU_WR;

	if (ioctl(fd, MALI_INTERCEPT_IOCTL_IMPORT_PHYS, &p) < 0) {
		fprintf(stderr, "IMPORT_PHYS: %s\n", strerror(errno));
		return -1;
	}

	printf("IMPORT_PHYS: gpu_va=%#llx va_pages=%llu flags=%#llx\n",
	       (unsigned long long)p.out.gpu_va,
	       (unsigned long long)p.out.va_pages,
	       (unsigned long long)p.out.flags);

	*gpu_va_out   = p.out.gpu_va;
	*map_size_out = (size_t)(p.out.va_pages << PAGE_SHIFT_4K);
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

	*gpu_va_out = (uint64_t)(uintptr_t)cpu;
	return cpu;
}

static void mali_free_page(int fd, void *cpu, uint64_t gpu_va)
{
	/*
	 * SAME_VA-аллокация: munmap дёрнет kbase_cpu_vm_close, тот
	 * декрементнёт refcount у cpu_alloc и при ref→0 mali сам
	 * вычистит регион из reg_rbtree. Звать KBASE_IOCTL_MEM_FREE
	 * поверх — лишний шаг, mali бы выдала warning "called with
	 * nonexistent gpu_addr".
	 */
	(void)fd;
	(void)gpu_va;
	if (cpu)
		munmap(cpu, PAGE_SZ_4K);
}

static void build_write_value_job(void *job_page, uint64_t target,
				   uint32_t value)
{
	memset(job_page, 0,
	       sizeof(struct mali_job_descriptor_header) +
	       sizeof(struct mali_payload_write_value));

	struct mali_job_descriptor_header *h = job_page;
	/* job_descriptor_size = 1 (bit 0), job_type = 2 (bits 1..7) */
	h->flags0    = (MALI_JOB_TYPE_WRITE_VALUE << 1) | 1;
	h->flags1    = 0;
	h->job_index = 1;
	h->next_job  = 0;

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
	/* Имортированная область идёт как external resource — это
	 * заставляет kbase запиннить страницы dma_buf и положить их в
	 * GPU MMU на время атома. */
	struct base_external_resource extres = {
		.ext_resource = (target_gpu_va & ~0xFFFull) |
				BASE_EXT_RES_ACCESS_EXCLUSIVE,
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

	/* Импортируем демонстрационный физический адрес. На реальном
	 * устройстве адрес отдаёт специализированный драйвер. */
	uint64_t data_gpu = 0;
	size_t   data_map_size = 0;
	if (mali_import_phys(fd, DEMO_PHYS_ADDR, DEMO_LENGTH,
			     &data_gpu, &data_map_size) < 0) {
		close(fd);
		return 1;
	}
	printf("imported phys=%#llx len=%#llx -> gpu_va=%#llx (готовый, без mmap)\n",
	       (unsigned long long)DEMO_PHYS_ADDR,
	       (unsigned long long)DEMO_LENGTH,
	       (unsigned long long)data_gpu);

	/* job descriptor — обычная SAME_VA аллокация */
	uint64_t job_gpu = 0;
	void *job = mali_alloc_page(fd, &job_gpu);
	if (!job) {
		munmap((void *)(uintptr_t)data_gpu, data_map_size);
		close(fd);
		return 1;
	}

	build_write_value_job(job, data_gpu, 0x12345678u);

	printf("submit: jc=%#llx -> [%#llx] = 0x12345678\n",
	       (unsigned long long)job_gpu,
	       (unsigned long long)data_gpu);
	printf("ПРИМЕЧАНИЕ: DEMO_PHYS_ADDR (%#llx) — фейковый, на этой\n"
	       "            записи GPU вероятно вернёт WRITE_FAULT/MMU_FAULT.\n"
	       "            Это нормально — тест проверяет цепочку IOCTL+импорт,\n"
	       "            а не корректность физической памяти.\n",
	       (unsigned long long)DEMO_PHYS_ADDR);

	int rc = mali_submit_and_wait(fd, job_gpu, data_gpu);

	/* Cleanup — для SAME_VA одного munmap достаточно: kbase_cpu_vm_close
	 * сам отстреливает регион. KBASE_IOCTL_MEM_FREE поверх не нужен. */
	mali_free_page(fd, job, job_gpu);
	munmap((void *)(uintptr_t)data_gpu, data_map_size);

	close(fd);
	return rc;
}
