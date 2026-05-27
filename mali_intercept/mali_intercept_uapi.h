/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * mali_intercept_uapi.h — UAPI заголовок модуля mali_intercept.
 *
 * Этот файл подключается из пользовательского пространства (системный
 * сервис, использующий /dev/mali0). Никаких kernel-only типов или
 * макросов в нём быть не должно.
 */

#ifndef _MALI_INTERCEPT_UAPI_H_
#define _MALI_INTERCEPT_UAPI_H_

#include <linux/types.h>
#include <linux/ioctl.h>

/*
 * Magic для IOCTL, добавленных модулем mali_intercept. Отличается от
 * KBASE_IOCTL_TYPE (0x80), что позволяет в обёртке однозначно
 * различать наши IOCTL от штатных IOCTL mali_kbase.
 */
#define MALI_INTERCEPT_IOCTL_TYPE 'M'  /* 0x4D */

/**
 * union mali_intercept_import_phys — импортировать физический буфер
 * в GPU VA пространство контекста, привязанного к открытому
 * /dev/mali0 файловому дескриптору.
 *
 * Буфер должен быть выровнен на страницу, длина кратна PAGE_SIZE.
 * Память управляется внешним (специализированным) драйвером —
 * mali_intercept только проксирует её GPU через временный dma_buf.
 * Спецдрайвер отвечает за время жизни и cache maintenance;
 * mali_intercept ничего о ней не освобождает.
 *
 * @in.phys_addr:  Физический адрес начала буфера в CPU RAM.
 *                 Должен быть page-aligned.
 * @in.length:     Длина буфера в байтах. Должна быть кратна PAGE_SIZE.
 * @in.flags:      Биты BASE_MEM_* (PROT_GPU_RD/WR и т.д.) — те же,
 *                 что принимает KBASE_IOCTL_MEM_IMPORT.
 *
 * @out.gpu_va:    Готовый GPU VA, по которому замаплен буфер.
 *                 В отличие от штатного KBASE_IOCTL_MEM_IMPORT, это
 *                 НЕ cookie — модуль сам делает mmap(cookie) изнутри
 *                 ядра через vm_mmap, поэтому пользователю не нужно
 *                 ничего домапывать. Для 64-битного non-compat
 *                 вызывающего эта же VA является CPU VA (SAME_VA).
 * @out.va_pages:  Размер маппинга в страницах PAGE_SIZE.
 * @out.flags:     Итоговые флаги, как их выставил mali_kbase
 *                 (может убрать неподдерживаемые биты).
 *
 * Очистка:
 *   munmap(out.gpu_va, out.va_pages * PAGE_SIZE);
 *
 *   Этого достаточно: при unmap последней VMA mali сам отстреливает
 *   регион (kbase_cpu_vm_close → kbase_mem_free_region), убирает
 *   GPU MMU маппинг и dma_buf_put-ит наш phys-dmabuf. Звать ещё
 *   KBASE_IOCTL_MEM_FREE(out.gpu_va) поверх munmap НЕ нужно — он
 *   найдёт регион уже снятым и выдаст warning "called with
 *   nonexistent gpu_addr" в dmesg.
 */
union mali_intercept_import_phys {
	struct {
		__u64 phys_addr;
		__u64 length;
		__u64 flags;
	} in;
	struct {
		__u64 gpu_va;
		__u64 va_pages;
		__u64 flags;
	} out;
};

#define MALI_INTERCEPT_IOCTL_IMPORT_PHYS \
	_IOWR(MALI_INTERCEPT_IOCTL_TYPE, 0, union mali_intercept_import_phys)

#endif /* _MALI_INTERCEPT_UAPI_H_ */
