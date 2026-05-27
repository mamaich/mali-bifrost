# SPDX-License-Identifier: GPL-2.0
#
# Сборка тестовой утилиты mali_phys_import_demo (userspace).
#
# Нативно на устройстве:
#   make
#
# Кросс с GNU toolchain:
#   make CC=aarch64-linux-gnu-gcc
#
# Кросс с NDK:
#   make CC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang

CC      ?= aarch64-linux-gnu-gcc
CFLAGS  ?= -Wall -O2
CPPFLAGS = -I..

BIN = mali_phys_import_demo
SRC = mali_phys_import_demo.c

all: $(BIN)

$(BIN): $(SRC) ../mali_intercept_uapi.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $(SRC) -o $@

clean:
	rm -f $(BIN)

.PHONY: all clean
