// SPDX-License-Identifier: GPL-2.0
/*
 * mali_intercept.c — перехватчик IOCTL устройства /dev/maliN
 *
 * Модуль ставит kprobe на статическую функцию kbase_ioctl() из модуля
 * mali_kbase и для каждого вызова печатает в системный лог номер IOCTL
 * вместе с его разобранными полями (dir/type/nr/size) и PID/именем
 * процесса, который этот IOCTL вызвал.
 *
 * Цель — kprobe на kbase_ioctl(), а не на vfs_ioctl/sys_ioctl, чтобы
 * срабатывать только на /dev/maliN и не зашумлять лог посторонним
 * IOCTL-трафиком в системе.
 *
 * Требования к ядру:
 *   - CONFIG_KPROBES=y
 *   - модуль mali_kbase должен быть загружен ДО mali_intercept,
 *     иначе символ kbase_ioctl не будет найден через kallsyms.
 *
 * Целевая платформа: ARM64, Linux 4.9.113 (Amlogic Meson G12A/SM1/G12B).
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/kprobes.h>
#include <linux/ptrace.h>
#include <linux/sched.h>
#include <linux/ioctl.h>

#define INTERCEPT_TARGET "kbase_ioctl"

static struct kprobe kbase_ioctl_kp = {
	.symbol_name = INTERCEPT_TARGET,
};

/*
 * pre-handler срабатывает в начале kbase_ioctl, до её тела.
 *
 * Сигнатура цели:
 *   static long kbase_ioctl(struct file *filp,
 *                           unsigned int cmd,
 *                           unsigned long arg);
 *
 * Соглашение о вызовах AArch64 (AAPCS64): первые 8 целочисленных/
 * указательных аргументов передаются в x0..x7. Значит:
 *   regs->regs[0] == filp
 *   regs->regs[1] == cmd
 *   regs->regs[2] == arg
 */
static int handler_pre(struct kprobe *p, struct pt_regs *regs)
{
	unsigned int cmd = (unsigned int)regs->regs[1];

	pr_info("mali_intercept: pid=%d comm=%s cmd=0x%08x "
		"(dir=%u type=0x%02x nr=%u size=%u)\n",
		current->pid, current->comm, cmd,
		_IOC_DIR(cmd), _IOC_TYPE(cmd),
		_IOC_NR(cmd), _IOC_SIZE(cmd));

	return 0;
}

static int __init mali_intercept_init(void)
{
	int ret;

	kbase_ioctl_kp.pre_handler = handler_pre;

	ret = register_kprobe(&kbase_ioctl_kp);
	if (ret < 0) {
		pr_err("mali_intercept: register_kprobe(%s) failed: %d\n",
		       INTERCEPT_TARGET, ret);
		if (ret == -EINVAL)
			pr_err("mali_intercept: символ '%s' не найден — "
			       "загружен ли модуль mali_kbase?\n",
			       INTERCEPT_TARGET);
		return ret;
	}

	pr_info("mali_intercept: kprobe установлен на %s (addr=%pK)\n",
		INTERCEPT_TARGET, kbase_ioctl_kp.addr);
	return 0;
}

static void __exit mali_intercept_exit(void)
{
	unregister_kprobe(&kbase_ioctl_kp);
	pr_info("mali_intercept: kprobe снят\n");
}

module_init(mali_intercept_init);
module_exit(mali_intercept_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("mamaich");
MODULE_DESCRIPTION("Перехват IOCTL для /dev/maliN через kprobe на kbase_ioctl");
MODULE_VERSION("0.1");
