# SPDX-License-Identifier: GPL-2.0
#
# Out-of-tree сборка модуля mali_intercept.
#
# mali_intercept зависит от символа kbase_find_device, экспортируемого
# модулем mali_kbase. Чтобы линковщик модулей увидел этот символ при
# сборке, нужно подсунуть ему Module.symvers от mali_kbase через
# KBUILD_EXTRA_SYMBOLS.
#
# Для нативной сборки на устройстве (mali_kbase уже собран рядом):
#   make
#
# Для кросс-сборки под ARM64 (с указанием путей):
#   make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- \
#        KDIR=/path/to/linux-4.9.113 \
#        MALI_SYMVERS=/path/to/mali-bifrost/driver/product/kernel/drivers/gpu/arm/midgard/Module.symvers
#
# Установка на устройстве:
#   sudo insmod /path/to/mali_kbase.ko       # если ещё не загружен
#   sudo insmod mali_intercept.ko
#   dmesg -w
#   sudo rmmod mali_intercept
#
# ВАЖНО: не выгружайте mali_kbase, пока загружен mali_intercept.
# Модульная зависимость через kbase_find_device этого не даст, но
# полагаться на это можно только если модуль был загружен через
# modprobe с правильным depmod.

obj-m += mali_intercept.o

KDIR ?= /lib/modules/$(shell uname -r)/build

# Где взять Module.symvers от mali_kbase. По умолчанию — соседний
# каталог в этом репозитории (после успешного `make` в корне).
MALI_KBASE_DIR ?= $(CURDIR)/../driver/product/kernel/drivers/gpu/arm/midgard
MALI_SYMVERS   ?= $(MALI_KBASE_DIR)/Module.symvers

PWD := $(shell pwd)

all:
	@if [ ! -f "$(MALI_SYMVERS)" ]; then \
		echo "ERROR: $(MALI_SYMVERS) not found."; \
		echo "       Сначала соберите mali_kbase (make в корне репо),"; \
		echo "       либо передайте MALI_SYMVERS=<путь к Module.symvers>."; \
		exit 1; \
	fi
	$(MAKE) -C $(KDIR) M=$(PWD) \
		KBUILD_EXTRA_SYMBOLS="$(MALI_SYMVERS)" \
		modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

.PHONY: all clean
