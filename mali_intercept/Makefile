# SPDX-License-Identifier: GPL-2.0
#
# Out-of-tree сборка модуля mali_intercept.
#
# Для нативной сборки на устройстве:
#   make
#
# Для кросс-сборки под ARM64:
#   make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- \
#        KDIR=/path/to/linux-4.9.113
#
# Установка:
#   sudo insmod mali_intercept.ko
#   dmesg -w
#   sudo rmmod mali_intercept

obj-m += mali_intercept.o

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

.PHONY: all clean
