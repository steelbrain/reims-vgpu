# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
# The hello bound consumer, built through the SDK. Run on a Mac with Xcode.
# x86-64 by default; `make ARCH='-arch arm64e'` builds the arm64e variant for a
# vmapple guest, with an arch-suffixed output so both can coexist. The bundle
# that results is dormant: Graft binds it at boot (src/Hello.cpp).
NAME = GraftHelloConsumer
VERSION = 1.0.0
SDK = $(shell xcrun --sdk macosx --show-sdk-path)
ARCH ?= -arch x86_64
ARCHNAME = $(lastword $(ARCH))
# x86-64 keeps the plain name tools/internal/boot-core.py expects; other arches
# suffix so the two builds do not overwrite each other.
SUFFIX = $(if $(filter x86_64,$(ARCHNAME)),,-$(ARCHNAME))
BUILD = $(CURDIR)/../../build/hello-consumer$(SUFFIX)
KEXT = $(CURDIR)/../../build/GraftHelloConsumer$(SUFFIX).kext
BINARY = $(KEXT)/Contents/MacOS/$(NAME)
OBJECTS = $(BUILD)/Hello.o
HEADERS = src/Hello.h
KERNEL = $(SDK)/System/Library/Frameworks/Kernel.framework/Headers
# The red zone is x86-only; arm64 has none.
REDZONE = $(if $(filter x86_64,$(ARCHNAME)),-mno-red-zone,)
CXXFLAGS = $(ARCH) -isysroot $(SDK) -mmacosx-version-min=11.0 \
    -mkernel -nostdinc -isystem $(KERNEL) -DKERNEL -std=c++17 \
    -fapple-kext -fno-rtti -fno-exceptions -fno-stack-protector \
    -fno-builtin $(REDZONE) -Os -Wall -Wextra -Werror -I.

.PHONY: all
all: $(BINARY)

$(BUILD)/Hello.o: src/Hello.cpp $(HEADERS) Makefile
	@mkdir -p $(@D)
	xcrun --sdk macosx clang++ $(CXXFLAGS) -c $< -o $@

include ../../sdk/consumer.mk
