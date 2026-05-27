// SPDX-License-Identifier: GPL-2.0
/*
 * mali_intercept.c — перехватчик IOCTL устройства /dev/maliN
 *
 * Подход (без CONFIG_KPROBES):
 *   1. Через kallsyms_lookup_name() находим адрес статической
 *      file_operations 'kbase_fops' в загруженном модуле mali_kbase.
 *   2. Сохраняем оригинальные указатели .unlocked_ioctl и .compat_ioctl.
 *   3. Через set_memory_rw() делаем страницу с kbase_fops пишимой
 *      (она в .rodata модуля и закрыта на запись, когда ядро собрано
 *      с CONFIG_DEBUG_SET_MODULE_RONX).
 *   4. Подменяем указатели на свои обёртки, которые сначала логируют
 *      IOCTL, потом вызывают оригинальную функцию.
 *   5. Возвращаем страницу в R/O.
 *
 * Поскольку мы патчим разделяемую struct file_operations, перехват
 * срабатывает для всех уже открытых и будущих файловых дескрипторов
 * любого /dev/maliN — никакого перехвата на уровне open() не нужно.
 *
 * Зависимость от mali_kbase создаётся через вызов экспортированной
 * функции kbase_find_device() — это заставляет модульный загрузчик
 * подгружать mali_kbase раньше и не даёт его выгрузить, пока
 * mali_intercept загружен.
 *
 * Требования к ядру:
 *   - CONFIG_KALLSYMS=y (стандартно для embedded-ядер)
 *   - set_memory_rw/ro экспортированы (на ARM64 в 4.9 — да,
 *     EXPORT_SYMBOL_GPL); если в вендорном ядре экспорта нет —
 *     обнаружится через kallsyms-фоллбэк
 *   - mali_kbase должен быть собран как модуль (его статические
 *     символы попадают в kallsyms только так)
 *
 * Целевая платформа: ARM64, Linux 4.9.113 (Amlogic Meson G12A/SM1/G12B).
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/kallsyms.h>
#include <linux/fs.h>
#include <linux/ioctl.h>
#include <linux/sched.h>
#include <linux/mm.h>
#include <linux/delay.h>
#include <linux/atomic.h>

#define TARGET_FOPS_SYMBOL "kbase_fops"

/*
 * Forward declaration экспортируемых из mali_kbase символов. Используем
 * их единственно для того, чтобы создать модульную зависимость:
 * mali_intercept нельзя загрузить без mali_kbase, и mali_kbase нельзя
 * выгрузить пока mali_intercept жив. Сами функции дают нам ещё и
 * sanity-check: убедиться, что mali_kbase действительно
 * проинициализировался хотя бы для одного /dev/maliN.
 */
struct kbase_device;
extern struct kbase_device *kbase_find_device(int minor);
extern void kbase_release_device(struct kbase_device *kbdev);

static struct file_operations *target_fops;

static long (*orig_unlocked_ioctl)(struct file *, unsigned int, unsigned long);
static long (*orig_compat_ioctl)(struct file *, unsigned int, unsigned long);

/*
 * set_memory_rw/ro ищем через kallsyms, чтобы не залипать на
 * EXPORT_SYMBOL_GPL: в стоковом v4.9.113 они есть, но вендорное
 * ядро Amlogic могло их выпилить.
 */
typedef int (*set_memory_attr_fn)(unsigned long addr, int numpages);
static set_memory_attr_fn set_memory_rw_fn;
static set_memory_attr_fn set_memory_ro_fn;

/*
 * Счётчик потоков, находящихся сейчас внутри наших обёрток.
 * Нужен для безопасной выгрузки: после восстановления указателей
 * ждём, пока не выйдут все, кто успел зайти по старому указателю.
 */
static atomic_t in_wrapper = ATOMIC_INIT(0);

static inline void log_ioctl(const char *tag, unsigned int cmd)
{
	pr_info("mali_intercept[%s]: pid=%d comm=%s cmd=0x%08x "
		"(dir=%u type=0x%02x nr=%u size=%u)\n",
		tag, current->pid, current->comm, cmd,
		_IOC_DIR(cmd), _IOC_TYPE(cmd),
		_IOC_NR(cmd), _IOC_SIZE(cmd));
}

static long wrapper_unlocked_ioctl(struct file *filp, unsigned int cmd,
				   unsigned long arg)
{
	long ret;

	atomic_inc(&in_wrapper);
	log_ioctl("unlocked", cmd);
	ret = orig_unlocked_ioctl ?
		orig_unlocked_ioctl(filp, cmd, arg) : -ENOTTY;
	atomic_dec(&in_wrapper);
	return ret;
}

static long wrapper_compat_ioctl(struct file *filp, unsigned int cmd,
				 unsigned long arg)
{
	long ret;

	atomic_inc(&in_wrapper);
	log_ioctl("compat  ", cmd);
	ret = orig_compat_ioctl ?
		orig_compat_ioctl(filp, cmd, arg) : -ENOTTY;
	atomic_dec(&in_wrapper);
	return ret;
}

/*
 * Снимает или ставит R/O защиту со страниц, в которых лежит target_fops.
 * Если kallsyms не нашёл set_memory_*, тихо ничего не делаем — значит
 * либо ядро без CONFIG_DEBUG_SET_MODULE_RONX (rodata уже writable),
 * либо запись просто не получится и init упадёт с понятной ошибкой.
 */
static int set_fops_writable(int writable)
{
	unsigned long base, top;
	int npages;
	set_memory_attr_fn fn;

	fn = writable ? set_memory_rw_fn : set_memory_ro_fn;
	if (!fn)
		return 0;

	base = (unsigned long)target_fops & PAGE_MASK;
	top  = ((unsigned long)target_fops + sizeof(*target_fops) - 1) &
	       PAGE_MASK;
	npages = ((top - base) >> PAGE_SHIFT) + 1;

	return fn(base, npages);
}

static int __init mali_intercept_init(void)
{
	struct kbase_device *probe;
	int ret;

	/*
	 * Sanity-check: mali_kbase загружен и хотя бы один /dev/maliN
	 * существует. Заодно фиксируем модульную зависимость от
	 * mali_kbase (через сам факт вызова его символа).
	 */
	probe = kbase_find_device(-1);
	if (!probe) {
		pr_err("mali_intercept: kbase_find_device(-1) вернул NULL — "
		       "ни одного /dev/maliN не зарегистрировано\n");
		return -ENODEV;
	}
	kbase_release_device(probe);

	target_fops = (struct file_operations *)
		kallsyms_lookup_name(TARGET_FOPS_SYMBOL);
	if (!target_fops) {
		pr_err("mali_intercept: символ '%s' не найден через kallsyms "
		       "(включен ли CONFIG_KALLSYMS? mali_kbase собран как модуль?)\n",
		       TARGET_FOPS_SYMBOL);
		return -ENOENT;
	}

	set_memory_rw_fn = (set_memory_attr_fn)
		kallsyms_lookup_name("set_memory_rw");
	set_memory_ro_fn = (set_memory_attr_fn)
		kallsyms_lookup_name("set_memory_ro");
	if (!set_memory_rw_fn || !set_memory_ro_fn)
		pr_warn("mali_intercept: set_memory_rw/ro не найдены через "
			"kallsyms; пишу в .rodata напрямую — сработает только "
			"если в ядре выключен CONFIG_DEBUG_SET_MODULE_RONX\n");

	orig_unlocked_ioctl = target_fops->unlocked_ioctl;
	orig_compat_ioctl   = target_fops->compat_ioctl;

	ret = set_fops_writable(1);
	if (ret) {
		pr_err("mali_intercept: set_memory_rw failed: %d\n", ret);
		return ret;
	}

	WRITE_ONCE(target_fops->unlocked_ioctl, wrapper_unlocked_ioctl);
	WRITE_ONCE(target_fops->compat_ioctl,   wrapper_compat_ioctl);

	(void)set_fops_writable(0);

	pr_info("mali_intercept: установлен на kbase_fops @ %p "
		"(original ioctl @ %pS)\n",
		target_fops, orig_unlocked_ioctl);
	return 0;
}

static void __exit mali_intercept_exit(void)
{
	int waited_ms = 0;

	if (!target_fops)
		return;

	(void)set_fops_writable(1);
	WRITE_ONCE(target_fops->unlocked_ioctl, orig_unlocked_ioctl);
	WRITE_ONCE(target_fops->compat_ioctl,   orig_compat_ioctl);
	(void)set_fops_writable(0);

	/*
	 * После восстановления указателей новых заходов в обёртки не
	 * будет. Но кто-то мог успеть прочитать указатель wrapper_* до
	 * подмены и сейчас находится между чтением указателя и
	 * atomic_inc. Даём небольшую фору и потом ждём, пока счётчик
	 * не опустится до нуля.
	 */
	msleep(100);
	while (atomic_read(&in_wrapper) > 0) {
		if (waited_ms % 1000 == 0)
			pr_info("mali_intercept: ждём выхода обёрток "
				"(%d in flight, ждём %d мс)\n",
				atomic_read(&in_wrapper), waited_ms);
		msleep(100);
		waited_ms += 100;
	}

	pr_info("mali_intercept: снят\n");
}

module_init(mali_intercept_init);
module_exit(mali_intercept_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("mamaich");
MODULE_DESCRIPTION("Перехват IOCTL для /dev/maliN через подмену kbase_fops");
MODULE_VERSION("0.2");
