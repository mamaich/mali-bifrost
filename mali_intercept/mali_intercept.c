// SPDX-License-Identifier: GPL-2.0
/*
 * mali_intercept.c — перехватчик IOCTL устройства /dev/maliN
 *                    плюс собственный IOCTL для импорта физической
 *                    памяти из reserved-memory.
 *
 * Часть 1. Перехват IOCTL (без CONFIG_KPROBES):
 *
 *   1. Через kallsyms_lookup_name() находим адрес статической
 *      file_operations 'kbase_fops' в загруженном модуле mali_kbase.
 *   2. Сохраняем оригинальные указатели .unlocked_ioctl и .compat_ioctl.
 *   3. Через set_memory_rw() делаем страницу с kbase_fops пишимой.
 *   4. Подменяем указатели на свои обёртки. Обёртка:
 *        - если IOCTL имеет наш magic ('M'), идёт в свой диспетчер;
 *        - иначе логирует IOCTL и делегирует в оригинал.
 *
 * Часть 2. IOCTL MALI_INTERCEPT_IOCTL_IMPORT_PHYS:
 *
 *   Пользователь передаёт {phys_addr, length, flags}, получает обратно
 *   {gpu_va, va_pages, flags} — как у штатного KBASE_IOCTL_MEM_IMPORT,
 *   но источник памяти — физический адрес от стороннего драйвера, а не
 *   user VA / dma_buf fd.
 *
 *   Реализация — гибрид через UMM/dma_buf:
 *     - оборачиваем физический буфер во временный dma_buf с минимальным
 *       exporter-ом, который отдаёт sg_table с одним элементом
 *       (page=NULL, dma_address=phys_addr). mali_kbase в kbase_mem.c
 *       читает только sg_dma_address(), так что struct page не нужен —
 *       это важно для reserved-memory, где pfn_valid()=false.
 *     - получаем для этого dma_buf транзитный fd в таблице вызывающего
 *       процесса, вызываем оригинальный kbase_ioctl с
 *       KBASE_IOCTL_MEM_IMPORT type=UMM, после чего fd закрываем —
 *       kbase уже сделал dma_buf_get() внутри и держит свой ref.
 *     - так как мы вызываем kbase_ioctl с указателями на kernel-память
 *       (наш kparam и наш fd-holder), переключаемся в KERNEL_DS на
 *       время вызова. В 4.9 это штатный приём.
 *
 * Зависимость от mali_kbase создаётся через вызов экспортированной
 * функции kbase_find_device() — это заставляет модульный загрузчик
 * подгружать mali_kbase раньше и не даёт его выгрузить, пока
 * mali_intercept загружен.
 *
 * Требования к ядру:
 *   - CONFIG_KALLSYMS=y
 *   - CONFIG_DMA_SHARED_BUFFER=y
 *   - mali_kbase собран как модуль (его static-символы попадают в
 *     kallsyms только в этом случае)
 *
 * Целевая платформа: ARM64, Linux 4.9.113 (Amlogic Meson G12A/SM1/G12B).
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/kallsyms.h>
#include <linux/fs.h>
#include <linux/file.h>
#include <linux/fdtable.h>
#include <linux/ioctl.h>
#include <linux/sched.h>
#include <linux/mm.h>
#include <linux/delay.h>
#include <linux/atomic.h>
#include <linux/uaccess.h>
#include <linux/mman.h>
#include <linux/dma-buf.h>
#include <linux/scatterlist.h>
#include <linux/slab.h>
#include <linux/errno.h>

#include "mali_intercept_uapi.h"

#define TARGET_FOPS_SYMBOL "kbase_fops"

/*
 * Воспроизведение нужных констант и структур из mali_kbase, чтобы не
 * подключать всю кучу его заголовков. Эти значения — часть стабильного
 * ABI mali_kbase r16p0 и должны совпадать с тем, что определено в
 * mali_kbase_ioctl.h / mali_base_kernel.h.
 */
#define KBASE_IOCTL_TYPE_LOCAL          0x80
#define KBASE_MEM_IMPORT_TYPE_UMM_LOCAL 2 /* enum base_mem_import_type */

/*
 * base_mem_alloc_flags bits, нужные для маппинга прав VMA на права
 * импортируемого региона. Должны совпадать с mali_base_kernel.h.
 */
#define BASE_MEM_PROT_CPU_RD_LOCAL (1ull << 0)
#define BASE_MEM_PROT_CPU_WR_LOCAL (1ull << 1)
#define BASE_MEM_SECURE_LOCAL      (1ull << 16)

union kbase_ioctl_mem_import_local {
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

#define KBASE_IOCTL_MEM_IMPORT_LOCAL \
	_IOWR(KBASE_IOCTL_TYPE_LOCAL, 22, union kbase_ioctl_mem_import_local)

struct kbase_ioctl_mem_free_local {
	__u64 gpu_addr;
};

#define KBASE_IOCTL_MEM_FREE_LOCAL \
	_IOW(KBASE_IOCTL_TYPE_LOCAL, 7, struct kbase_ioctl_mem_free_local)

/*
 * Forward declaration экспортируемых из mali_kbase символов. Используем
 * для модульной зависимости (mali_kbase обязан быть загружен раньше и
 * не может быть выгружен пока жив mali_intercept) и для проверки, что
 * хотя бы один /dev/maliN зарегистрирован.
 */
struct kbase_device;
extern struct kbase_device *kbase_find_device(int minor);
extern void kbase_release_device(struct kbase_device *kbdev);

static struct file_operations *target_fops;

static long (*orig_unlocked_ioctl)(struct file *, unsigned int, unsigned long);
static long (*orig_compat_ioctl)(struct file *, unsigned int, unsigned long);

typedef int (*set_memory_attr_fn)(unsigned long addr, int numpages);
static set_memory_attr_fn set_memory_rw_fn;
static set_memory_attr_fn set_memory_ro_fn;

/*
 * __close_fd в вендорном ядре Amlogic 4.9 не экспортирован, как и
 * replace_fd. Ищем через kallsyms. Это требует, чтобы либо символ
 * был EXPORT_SYMBOL (в стоковом 4.9 — да), либо ядро собрано с
 * CONFIG_KALLSYMS_ALL=y (тогда виден и не экспортированный
 * статический).
 */
typedef int (*close_fd_fn_t)(struct files_struct *files, unsigned fd);
static close_fd_fn_t close_fd_fn;

static atomic_t in_wrapper = ATOMIC_INIT(0);

/* =========================================================== */
/* dma_buf exporter для физического (reserved-memory) буфера   */
/* =========================================================== */

struct phys_dmabuf_priv {
	phys_addr_t phys_addr;
	size_t length;
};

static struct sg_table *phys_dmabuf_map(struct dma_buf_attachment *attach,
					enum dma_data_direction dir)
{
	struct phys_dmabuf_priv *priv = attach->dmabuf->priv;
	struct sg_table *sgt;
	int ret;

	sgt = kzalloc(sizeof(*sgt), GFP_KERNEL);
	if (!sgt)
		return ERR_PTR(-ENOMEM);

	ret = sg_alloc_table(sgt, 1, GFP_KERNEL);
	if (ret) {
		kfree(sgt);
		return ERR_PTR(ret);
	}

	/*
	 * mali_kbase читает только sg_dma_address()/sg_dma_len() в
	 * kbase_mem.c — см. цикл for_each_sg вокруг 3529. struct page
	 * не нужен. NULL допустим: sg_assign_page проверяет только
	 * выравнивание адреса страницы на 4 (NULL == 0 — пройдёт).
	 *
	 * Спецдрайвер сам делает cache maintenance, поэтому dma_map_sg
	 * нам не нужен. На платформе без SMMU между CPU и Mali GPU видят
	 * один и тот же физический адрес — отдаём его напрямую.
	 */
	sg_set_page(sgt->sgl, NULL, priv->length, 0);
	sg_dma_address(sgt->sgl) = priv->phys_addr;
	sg_dma_len(sgt->sgl) = priv->length;

	return sgt;
}

static void phys_dmabuf_unmap(struct dma_buf_attachment *attach,
			      struct sg_table *sgt,
			      enum dma_data_direction dir)
{
	sg_free_table(sgt);
	kfree(sgt);
}

static void phys_dmabuf_release(struct dma_buf *dbuf)
{
	struct phys_dmabuf_priv *priv = dbuf->priv;

	pr_debug("mali_intercept: phys_dmabuf release phys=0x%llx len=%zu\n",
		 (unsigned long long)priv->phys_addr, priv->length);
	kfree(priv);
}

/*
 * CPU-сторонний mmap нужен для cookie→gpu_va перехода в kbase_mmap
 * для UMM-импортов 64-битных задач (BASE_MEM_NEED_MMAP форсируется в
 * kbase_mem_from_umm). Пользователь обычно не трогает эту память —
 * её владелец спецдрайвер — но mmap всё равно должен пройти, иначе
 * у вызвавшего IMPORT_PHYS не будет валидного gpu_va.
 *
 * remap_pfn_range кладёт PTE напрямую из PFN, не требуя struct page
 * (это критично для reserved-memory с pfn_valid()=false). Защита
 * страниц используется та, что выставил kbase_mmap до dma_buf_mmap.
 */
static int phys_dmabuf_mmap(struct dma_buf *dbuf, struct vm_area_struct *vma)
{
	struct phys_dmabuf_priv *priv = dbuf->priv;
	size_t length = vma->vm_end - vma->vm_start;
	unsigned long pfn = (unsigned long)(priv->phys_addr >> PAGE_SHIFT);

	if (vma->vm_pgoff) {
		pr_warn("mali_intercept: phys_dmabuf mmap vm_pgoff=%lu, поддерживается только 0\n",
			vma->vm_pgoff);
		return -EINVAL;
	}
	if (length > priv->length) {
		pr_warn("mali_intercept: phys_dmabuf mmap len=%zu > buf=%zu\n",
			length, priv->length);
		return -EINVAL;
	}

	return remap_pfn_range(vma, vma->vm_start, pfn,
			       length, vma->vm_page_prot);
}

/*
 * dma_buf_export в ядре 4.9 имеет WARN_ON, требующий, чтобы у ops
 * были непустые .kmap_atomic, .kmap и .mmap — иначе он возвращает
 * -EINVAL. Это требование сняли только в 4.19. Мы реально не
 * поддерживаем CPU-kmap на reserved-memory (нет struct page), но
 * stub'ы должны существовать. Возвращаем NULL — для kmap это
 * легальный ответ "не могу замапить", потребитель должен это
 * корректно обработать. mali_kbase в нашем UMM-пути не зовёт ни
 * kmap, ни kmap_atomic — он использует только sg_dma_address из
 * map_dma_buf.
 */
static void *phys_dmabuf_kmap_atomic(struct dma_buf *dbuf,
				     unsigned long page_num)
{
	return NULL;
}

static void *phys_dmabuf_kmap(struct dma_buf *dbuf, unsigned long page_num)
{
	return NULL;
}

static const struct dma_buf_ops phys_dmabuf_ops = {
	.map_dma_buf   = phys_dmabuf_map,
	.unmap_dma_buf = phys_dmabuf_unmap,
	.release       = phys_dmabuf_release,
	.mmap          = phys_dmabuf_mmap,
	.kmap_atomic   = phys_dmabuf_kmap_atomic,
	.kmap          = phys_dmabuf_kmap,
};

static struct dma_buf *phys_dmabuf_create(phys_addr_t phys, size_t length)
{
	struct phys_dmabuf_priv *priv;
	struct dma_buf *dbuf;
	DEFINE_DMA_BUF_EXPORT_INFO(exp_info);

	priv = kzalloc(sizeof(*priv), GFP_KERNEL);
	if (!priv)
		return ERR_PTR(-ENOMEM);

	priv->phys_addr = phys;
	priv->length = length;

	exp_info.ops   = &phys_dmabuf_ops;
	exp_info.size  = length;
	exp_info.flags = O_RDWR;
	exp_info.priv  = priv;

	dbuf = dma_buf_export(&exp_info);
	if (IS_ERR(dbuf)) {
		kfree(priv);
		return dbuf;
	}

	return dbuf;
}

/* =========================================================== */
/* Обработчик MALI_INTERCEPT_IOCTL_IMPORT_PHYS                 */
/* =========================================================== */

static long handle_import_phys(struct file *filp, unsigned long arg)
{
	union mali_intercept_import_phys param;
	union kbase_ioctl_mem_import_local kparam;
	struct dma_buf *dbuf;
	int fd, holder_fd;
	long ret;
	mm_segment_t old_fs;

	if (!close_fd_fn) {
		pr_err("mali_intercept: __close_fd недоступен — "
		       "IMPORT_PHYS не может работать. Соберите ядро "
		       "с CONFIG_KALLSYMS_ALL=y или экспортируйте "
		       "__close_fd.\n");
		return -ENOSYS;
	}

	if (copy_from_user(&param, (void __user *)arg, sizeof(param)))
		return -EFAULT;

	if (!param.in.length)
		return -EINVAL;
	if (param.in.phys_addr & ~PAGE_MASK)
		return -EINVAL;
	if (param.in.length & ~PAGE_MASK)
		return -EINVAL;

	pr_info("mali_intercept: IMPORT_PHYS pid=%d phys=0x%llx len=0x%llx flags=0x%llx%s\n",
		current->pid,
		(unsigned long long)param.in.phys_addr,
		(unsigned long long)param.in.length,
		(unsigned long long)param.in.flags,
		(param.in.flags & BASE_MEM_SECURE_LOCAL) ? " [SECURE]" : "");

	dbuf = phys_dmabuf_create((phys_addr_t)param.in.phys_addr,
				  (size_t)param.in.length);
	if (IS_ERR(dbuf))
		return PTR_ERR(dbuf);

	/*
	 * Получаем транзитный fd в таблице дескрипторов вызывающего
	 * процесса. Он нужен, потому что mali kbase_mem_from_umm()
	 * использует dma_buf_get(fd) → fget(fd), что ищет file в
	 * current->files. Альтернативного пути через struct dma_buf *
	 * в public API mali_kbase нет.
	 */
	fd = dma_buf_fd(dbuf, O_CLOEXEC);
	if (fd < 0) {
		dma_buf_put(dbuf);
		return fd;
	}

	holder_fd = fd;
	kparam.in.flags   = param.in.flags;
	kparam.in.phandle = (u64)(uintptr_t)&holder_fd;
	kparam.in.type    = KBASE_MEM_IMPORT_TYPE_UMM_LOCAL;
	kparam.in.padding = 0;

	/*
	 * kbase_ioctl делает copy_from_user/copy_to_user относительно
	 * uarg и get_user(fd, phandle). Мы передаём kernel-указатели
	 * (&kparam и &holder_fd) — на время вызова переключаемся в
	 * KERNEL_DS, чтобы access_ok пропустил, а копирование
	 * вырождалось в memcpy. В 4.9 этот механизм ещё штатный.
	 */
	old_fs = get_fs();
	set_fs(KERNEL_DS);
	ret = orig_unlocked_ioctl(filp, KBASE_IOCTL_MEM_IMPORT_LOCAL,
				  (unsigned long)&kparam);
	set_fs(old_fs);

	/*
	 * Закрываем транзитный fd. kbase_mem_from_umm() уже сделал
	 * dma_buf_get → fget внутри, увеличив refcount нашего dma_buf;
	 * сам dma_buf останется жить, пока mali не вызовет dma_buf_put
	 * в kbase_mem_phy_alloc_free для KBASE_MEM_TYPE_IMPORTED_UMM —
	 * тогда сработает наш phys_dmabuf_release.
	 *
	 * __close_fd/replace_fd в вендорном Amlogic-ядре 4.9 не
	 * экспортированы, поэтому вызов идёт через указатель,
	 * полученный из kallsyms_lookup_name в mali_intercept_init.
	 */
	close_fd_fn(current->files, fd);

	if (ret < 0)
		return ret;

	/*
	 * Для 64-битного non-compat вызывающего kbase_mem_from_umm форсит
	 * BASE_MEM_NEED_MMAP, и kparam.out.gpu_va — не реальный GPU VA,
	 * а cookie. Превращение cookie → real gpu_va делается одной
	 * операцией mmap на наш filp: kbase_mmap по этому cookie
	 * подхватывает pending_region, отдаёт ему стабильную GPU VA и
	 * (для UMM) делегирует в dma_buf_mmap нашего phys-dmabuf.
	 * Так как SAME_VA: real_gpu_va == cpu_va. Делаем эту mmap сразу
	 * из ядра, чтобы пользователю не пришлось.
	 *
	 * vm_mmap кладёт VMA в адресное пространство вызывающего процесса
	 * (то самое, чей kbase_context лежит в filp->private_data) — это
	 * именно то, что нам нужно.
	 *
	 * Права в prot выводим строго из BASE_MEM_PROT_CPU_RD/WR флагов
	 * импорта: kbase_mmap проверяет, что VMA не запрашивает больше
	 * прав, чем у региона (KBASE_REG_CPU_RD/WR из reg->flags). При
	 * несовпадении возвращает -EPERM с "inconsistent VM flags".
	 * Берём из kparam.out.flags (после kbase_check_import_flags —
	 * там, например, BASE_MEM_PROT_CPU_WR может быть отрезан).
	 *
	 * Для BASE_MEM_SECURE-импорта пользователь обязан не запрашивать
	 * BASE_MEM_PROT_CPU_RD/WR (контракт mali: "secure нельзя CPU
	 * читать"), поэтому prot тут естественно становится PROT_NONE.
	 * vm_mmap всё равно вызывается — он нужен, чтобы превратить
	 * cookie в реальный gpu_va. Получившаяся VMA технически
	 * существует в адресном пространстве процесса, но никаких PTE
	 * с доступом не имеет (PAGE_NONE) — любое CPU обращение
	 * прилетит SIGSEGV. Это то, что и просили: "secure без mmap
	 * на CPU".
	 */
	{
		unsigned long ua;
		unsigned long prot = 0;
		unsigned long map_len =
			(unsigned long)kparam.out.va_pages << PAGE_SHIFT;

		if (kparam.out.flags & BASE_MEM_PROT_CPU_RD_LOCAL)
			prot |= PROT_READ;
		if (kparam.out.flags & BASE_MEM_PROT_CPU_WR_LOCAL)
			prot |= PROT_WRITE;
		if (!prot)
			prot = PROT_NONE;

		ua = vm_mmap(filp, 0, map_len, prot, MAP_SHARED,
			     (unsigned long)kparam.out.gpu_va);

		if (IS_ERR_VALUE(ua)) {
			struct kbase_ioctl_mem_free_local mf;

			mf.gpu_addr = kparam.out.gpu_va;
			pr_warn("mali_intercept: vm_mmap(cookie=0x%llx, len=%lu) failed: %ld; rollback\n",
				(unsigned long long)kparam.out.gpu_va,
				map_len, (long)ua);

			old_fs = get_fs();
			set_fs(KERNEL_DS);
			orig_unlocked_ioctl(filp, KBASE_IOCTL_MEM_FREE_LOCAL,
					    (unsigned long)&mf);
			set_fs(old_fs);

			return (long)ua;
		}

		param.out.gpu_va = (u64)ua;
	}
	param.out.va_pages = kparam.out.va_pages;
	param.out.flags    = kparam.out.flags;

	if (copy_to_user((void __user *)arg, &param, sizeof(param)))
		return -EFAULT;

	pr_info("mali_intercept: IMPORT_PHYS OK pid=%d -> gpu_va=0x%llx va_pages=%llu flags=0x%llx\n",
		current->pid,
		(unsigned long long)param.out.gpu_va,
		(unsigned long long)param.out.va_pages,
		(unsigned long long)param.out.flags);

	return 0;
}

static long mali_intercept_dispatch(struct file *filp, unsigned int cmd,
				    unsigned long arg)
{
	switch (cmd) {
	case MALI_INTERCEPT_IOCTL_IMPORT_PHYS:
		return handle_import_phys(filp, arg);
	default:
		pr_warn("mali_intercept: unknown intercept IOCTL nr=%u\n",
			_IOC_NR(cmd));
		return -ENOTTY;
	}
}

/* =========================================================== */
/* ioctl wrappers                                              */
/* =========================================================== */

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

	if (_IOC_TYPE(cmd) == MALI_INTERCEPT_IOCTL_TYPE) {
		ret = mali_intercept_dispatch(filp, cmd, arg);
	} else {
		log_ioctl("unlocked", cmd);
		ret = orig_unlocked_ioctl ?
			orig_unlocked_ioctl(filp, cmd, arg) : -ENOTTY;
	}

	atomic_dec(&in_wrapper);
	return ret;
}

static long wrapper_compat_ioctl(struct file *filp, unsigned int cmd,
				 unsigned long arg)
{
	long ret;

	atomic_inc(&in_wrapper);

	if (_IOC_TYPE(cmd) == MALI_INTERCEPT_IOCTL_TYPE) {
		ret = mali_intercept_dispatch(filp, cmd, arg);
	} else {
		log_ioctl("compat  ", cmd);
		ret = orig_compat_ioctl ?
			orig_compat_ioctl(filp, cmd, arg) : -ENOTTY;
	}

	atomic_dec(&in_wrapper);
	return ret;
}

/* =========================================================== */
/* Подмена / восстановление kbase_fops                         */
/* =========================================================== */

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

	close_fd_fn = (close_fd_fn_t)kallsyms_lookup_name("__close_fd");
	if (!close_fd_fn)
		pr_warn("mali_intercept: __close_fd не найден через kallsyms; "
			"IMPORT_PHYS IOCTL будет возвращать -ENOSYS. Возможно, "
			"нужен CONFIG_KALLSYMS_ALL=y.\n");

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
		"(original ioctl @ %pS); IMPORT_PHYS IOCTL = 0x%08lx\n",
		target_fops, orig_unlocked_ioctl,
		(unsigned long)MALI_INTERCEPT_IOCTL_IMPORT_PHYS);
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
	 * подмены и сейчас находится между чтением и atomic_inc. Даём
	 * небольшую фору и потом ждём, пока счётчик не опустится до нуля.
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
MODULE_DESCRIPTION("Перехват IOCTL + IMPORT_PHYS для /dev/maliN");
MODULE_VERSION("0.3");
