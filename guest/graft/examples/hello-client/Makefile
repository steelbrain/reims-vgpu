# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
# The hello owned client. Run on a Mac with Xcode. An owned client is an
# ordinary kext: it compiles its sources and the SDK start glue with kext flags
# and links them directly, with no manifest and no consumer.mk. x86-64 by
# default; `make ARCH='-arch arm64e'` builds the arm64e variant for a vmapple
# guest, with an arch-suffixed output so both can coexist.
NAME = GraftHelloClient
CLIENT_ID = ai.aneesiqbal.graft.HelloClient
VERSION = 1.0.0
SDK = $(shell xcrun --sdk macosx --show-sdk-path)
ARCH ?= -arch x86_64
ARCHNAME = $(lastword $(ARCH))
SUFFIX = $(if $(filter x86_64,$(ARCHNAME)),,-$(ARCHNAME))
BUILD = $(CURDIR)/../../build/hello-client$(SUFFIX)
KEXT = $(CURDIR)/../../build/GraftHelloClient$(SUFFIX).kext
BINARY = $(KEXT)/Contents/MacOS/$(NAME)
KERNEL = $(SDK)/System/Library/Frameworks/Kernel.framework/Headers
# The red zone is x86-only; arm64 has none.
REDZONE = $(if $(filter x86_64,$(ARCHNAME)),-mno-red-zone,)
CXXFLAGS = $(ARCH) -isysroot $(SDK) -mmacosx-version-min=11.0 \
    -mkernel -nostdinc -isystem $(KERNEL) -DKERNEL -std=c++17 \
    -fapple-kext -fno-rtti -fno-exceptions -fno-stack-protector \
    -fno-builtin $(REDZONE) -Os -Wall -Wextra -Werror -I../../sdk/include

.PHONY: all
all: $(BINARY)

$(BUILD)/HelloClient.o: src/HelloClient.cpp Makefile
	@mkdir -p $(@D)
	xcrun --sdk macosx clang++ $(CXXFLAGS) -c $< -o $@

# The SDK start glue: it declares the kmod and calls graft_client_initialize().
$(BUILD)/Start.o: ../../sdk/Start.cpp Makefile
	@mkdir -p $(@D)
	xcrun --sdk macosx clang++ $(CXXFLAGS) \
	    -DGRAFT_CLIENT_ID=$(CLIENT_ID) -DGRAFT_CLIENT_VERSION='"$(VERSION)"' -c $< -o $@

$(BINARY): $(BUILD)/HelloClient.o $(BUILD)/Start.o src/Info.plist
	@mkdir -p $(@D)
	xcrun --sdk macosx ld $(ARCH) -kext -undefined dynamic_lookup $(BUILD)/HelloClient.o $(BUILD)/Start.o -o $@
	cp src/Info.plist $(KEXT)/Contents/Info.plist
	plutil -lint $(KEXT)/Contents/Info.plist
