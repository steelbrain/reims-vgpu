# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
# A test-only consumer that exercises the activation gate across several
# IOMatchCategory values on one nub: Graft sets one property key on the nub and
# every gated personality matches together. Run on the Mac. x86-64 by default;
# `make ARCH='-arch arm64e'` builds the vmapple variant with a suffixed output.
NAME = GraftGateProbe
VERSION = 1.0.0
SDK = $(shell xcrun --sdk macosx --show-sdk-path)
ARCH ?= -arch x86_64
ARCHNAME = $(lastword $(ARCH))
# x86-64 keeps the plain name; other arches suffix so both can coexist.
SUFFIX = $(if $(filter x86_64,$(ARCHNAME)),,-$(ARCHNAME))
# The forced-refuse variant (REFUSE=1): a declaration naming an absent provider
# anchor, so Graft refuses the bind and never opens the gate. Same binary; the
# bundle is suffixed so it sits beside the normal one.
ifdef REFUSE
VARIANT = -refuse
GRAFT_DECL = src/graft-refuse.json
endif
BUILD = $(CURDIR)/../../../build/gate-probe$(VARIANT)$(SUFFIX)
KEXT = $(CURDIR)/../../../build/GraftGateProbe$(VARIANT)$(SUFFIX).kext
BINARY = $(KEXT)/Contents/MacOS/$(NAME)
OBJECTS = $(BUILD)/Probe.o
HEADERS = src/Probe.h
KERNEL = $(SDK)/System/Library/Frameworks/Kernel.framework/Headers
# The red zone is x86-only; arm64 has none.
REDZONE = $(if $(filter x86_64,$(ARCHNAME)),-mno-red-zone,)
CXXFLAGS = $(ARCH) -isysroot $(SDK) -mmacosx-version-min=11.0 \
    -mkernel -nostdinc -isystem $(KERNEL) -DKERNEL -std=c++17 \
    -fapple-kext -fno-rtti -fno-exceptions -fno-stack-protector \
    -fno-builtin $(REDZONE) -Os -Wall -Wextra -Werror -I.

.PHONY: all
all: $(BINARY)

$(BUILD)/Probe.o: src/Probe.cpp $(HEADERS) Makefile
	@mkdir -p $(@D)
	xcrun --sdk macosx clang++ $(CXXFLAGS) -c $< -o $@

include ../../../sdk/consumer.mk
