# The reims-graphics display kext for x86-64 macOS 11 to 26 guests, and, with
# ARCH_NAME=arm64e, for the arm64 macOS 14 and later guests Ambrosia boots:
# ReimsGraphics, an IOFramebuffer on the device's PCI function
# (src/ReimsGraphics.hpp; the protocol is docs/graphics-kext/design.md).
# Builds on a Mac with Xcode's clang and ld, against the macOS SDK's
# Kernel.framework, for macOS 11 and later, and stages it with Graft's
# consumer rules (../graft/sdk/consumer.mk, src/graft.json): dormant, for
# Graft to bind against the System kernel collection's libraries.
#
#   make        build/ReimsGraphics.kext
#   make ARCH_NAME=arm64e BUILD=build/arm64e
#               the arm64e kext, in a build directory of its own
#   make check  check its imports and IOFramebuffer's layout against a
#               guest's KDK (tools/check-imports.sh, tools/check-layout.py):
#                 KDK=<the guest release's KDK's Extensions directory>
#               The OpenCore link of the whole set is guest/'s set-check.
#   make test   build and run the native tests of the kext's own rules
#               (tests/*.cpp, against the headers they test)
#   make DTRACE=1 BUILD=build/dtrace
#               the kext for an image that traces it, its Info.plist setting
#               OSBundleForceDTraceInit (docs/qemu.md, "fbt probes in the
#               guest's kexts"); in a build directory of its own, as the
#               Info.plist is written only as the kext is linked
#   make clean  remove build/
#
# Everything lands in build/, never committed.

NAME     = ReimsGraphics
VERSION  = 1.0.0
SDK     ?= $(shell xcrun --sdk macosx --show-sdk-path)
KERNEL   = $(SDK)/System/Library/Frameworks/Kernel.framework/Headers

BUILD   ?= build
KEXT     = $(BUILD)/$(NAME).kext
BINARY   = $(KEXT)/Contents/MacOS/$(NAME)

KDK        ?=

TRACE   ?= 0
DTRACE  ?= 0

# The guest's architecture: x86_64 (macOS 11 and later) or arm64e (the arm64 guests Ambrosia
# boots, macOS 14 and later, built for a deployment target of 13.0). Each builds in a BUILD of
# its own.
ARCH_NAME ?= x86_64
ifeq ($(ARCH_NAME),arm64e)
MIN_OS = 13.0
else
MIN_OS = 11.0
endif
ARCH = -arch $(ARCH_NAME) -isysroot $(SDK) -mmacosx-version-min=$(MIN_OS)
# Graft's consumer rules, and the declaration and Info.plist they stage this kext from: one of
# each per architecture. An arm64 kernel collection has neither AppleGraphicsDeviceControl nor
# IOAcceleratorFamily2, and its WindowServer drives an IOMobileFramebuffer, so the arm64e kext
# (src/graft-arm64e.json, src/Info-arm64e.plist) takes its display's parent from
# IOMobileGraphicsFamily, its accelerator's from IOGraphicsFamily, and has no device control.
GRAFT_SDK ?= ../graft/sdk
ifeq ($(ARCH_NAME),arm64e)
GRAFT_DECL = src/graft-arm64e.json
GRAFT_INFO_PLIST = src/Info-arm64e.plist
# The arm64e display subclasses the Graft SDK's IOMobileFramebufferLegacy stand-in
# (<graft/IOMobileFramebufferLegacy.h>); the SDK's include directory holds it. x86-64 takes no
# Graft stand-in, so this is empty there and its flags are unchanged.
GRAFT_INC = -I$(GRAFT_SDK)/include
# How an arm64 process maps the region: its 16 KiB page, and no write-combining
# (src/RgfxRegionMapping.hpp).
GUEST_FLAGS = -DRGFX_GUEST_ARM64=1
endif

# src/shim: our own stand-ins for the two headers the SDK's IOHIDDevice.h includes and the SDK
# does not ship (RgfxPointerDevice.hpp).
FLAGS = $(ARCH) $(GRAFT_INC) $(GUEST_FLAGS) -mkernel -nostdinc -isystem $(KERNEL) -Isrc/shim \
	-I../../crates/reims-vgpu-graphics/include \
	-DKERNEL -DBUNDLE_VERSION='"$(VERSION)"' -DRGFX_FB_TRACE=$(TRACE) \
	-Os -Wall -Wextra -Wshadow-all -Werror -Wno-unused-parameter \
	-Wno-deprecated-declarations

CXXFLAGS = $(FLAGS) -std=c++17 -fapple-kext -fno-rtti -fno-exceptions
CFLAGS   = $(FLAGS) -std=c11

HEADERS = $(wildcard src/*.hpp src/shim/*/*.h src/shim/*/*/*.h)

# The Stage-2 framebuffer (ReimsGraphics), the Stage-3 Metal accelerator
# (RgfxAccelerator) and its region-map + doorbell user client, the guest's own mouse for
# POINTER_MOTION (RgfxPointerDevice), the pointer's moves (ReimsGraphicsPointer), on x86-64 the
# device control (RgfxDeviceControl) and the guest PAT's entry 2 (RgfxPat), and on arm64e the
# front end (ReimsGraphicsMobile) and the VideoToolbox service (RgfxVtService), all one kext: the
# objects both architectures build, with each architecture's own (OBJECTS_x86_64,
# OBJECTS_arm64e) third in the link.
OBJECTS_x86_64 = $(BUILD)/obj/RgfxDeviceControl.o $(BUILD)/obj/ReimsGraphicsPointer.o \
	$(BUILD)/obj/RgfxPat.o
OBJECTS_arm64e = $(BUILD)/obj/ReimsGraphicsMobile.o $(BUILD)/obj/ReimsGraphicsPointer.o \
	$(BUILD)/obj/RgfxVtService.o
OBJECTS = $(BUILD)/obj/ReimsGraphics.o \
	$(BUILD)/obj/RgfxAccelerator.o \
	$(OBJECTS_$(ARCH_NAME)) \
	$(BUILD)/obj/RgfxPointerDevice.o \
	$(BUILD)/obj/RgfxPointerReporting.o \
	$(BUILD)/obj/RgfxAcceleratorUserClient.o \
	$(BUILD)/obj/RgfxDisplayPipeUserClient.o \
	$(BUILD)/obj/kmod_info.o \
	$(BUILD)/obj/RgfxAcceleratorContexts.o \
	$(BUILD)/obj/RgfxAcceleratorUserClientContexts.o \
	$(BUILD)/obj/RgfxAcceleratorSurfaces.o \
	$(BUILD)/obj/RgfxSurfaceUserClient.o \
	$(BUILD)/obj/RgfxAcceleratorUserClientWindow.o \
	$(BUILD)/obj/RgfxAcceleratorLocks.o \
	$(BUILD)/obj/RgfxSharedEventUserClient.o \
	$(BUILD)/obj/RgfxGLUserClients.o

all: $(BINARY)

$(BUILD)/obj/%.o: src/%.cpp $(HEADERS) ../../crates/reims-vgpu-graphics/include/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	xcrun clang++ $(CXXFLAGS) -c $< -o $@

# RgfxPointerDevice's two IOReporting forwards call IOHIDDevice's private overrides through its
# vtable (src/RgfxPointerReporting.cpp): that file alone compiles with access control off.
$(BUILD)/obj/RgfxPointerReporting.o: CXXFLAGS += -fno-access-control

$(BUILD)/obj/%.o: src/%.c Makefile
	@mkdir -p $(@D)
	xcrun clang $(CFLAGS) -c $< -o $@

# The driver's source with its debug information, which check-layout.py reads
# for the SDK's IOFramebuffer.
$(BUILD)/obj/layout.o: src/ReimsGraphics.cpp src/ReimsGraphics.hpp ../../crates/reims-vgpu-graphics/include/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	xcrun clang++ $(CXXFLAGS) -g -fstandalone-debug -c $< -o $@

# Graft owns the two-pass link and dormant staging of this consumer.
include $(GRAFT_SDK)/consumer.mk

check: $(BINARY) $(BUILD)/obj/layout.o
	bash tools/check-imports.sh "$(KDK)" $(KEXT)
	python3 -I tools/check-layout.py "$(KDK)/IOGraphicsFamily.kext.dSYM" $(BUILD)/obj/layout.o

# The native tests: header-only rules of the kext's, built for this Mac.
TESTS = $(BUILD)/tests/spans_test $(BUILD)/tests/blocks_test $(BUILD)/tests/surface_blocks_test \
	$(BUILD)/tests/surface_test $(BUILD)/tests/pipe_clients_test $(BUILD)/tests/pipe_transactions_test \
	$(BUILD)/tests/window_test $(BUILD)/tests/scanout_owner_test \
	$(BUILD)/tests/display_modes_test $(BUILD)/tests/display_identity_test \
	$(BUILD)/tests/surface_locks_test \
	$(BUILD)/tests/bundle_paths_test $(BUILD)/tests/surface_caches_test \
	$(BUILD)/tests/vblank_clock_test $(BUILD)/tests/submit_gate_test $(BUILD)/tests/memory_size_test \
	$(BUILD)/tests/device_name_test $(BUILD)/tests/device_name_arm64_test \
	$(BUILD)/tests/queue_limits_test $(BUILD)/tests/user_client_words_test \
	$(BUILD)/tests/features_test $(BUILD)/tests/latency_test $(BUILD)/tests/release_table_test \
	$(BUILD)/tests/answer_look_test $(BUILD)/tests/answer_wait_test \
	$(BUILD)/tests/deferred_answers_test $(BUILD)/tests/shared_pipelines_test \
	$(BUILD)/tests/video_codecs_test $(BUILD)/tests/video_codecs_arm64_test \
	$(BUILD)/tests/ring_room_test $(BUILD)/tests/context_slots_test $(BUILD)/tests/pages_block_test \
	$(BUILD)/tests/close_table_test $(BUILD)/tests/ask_gate_test \
	$(BUILD)/tests/pipe_color_test $(BUILD)/tests/id_chunks_test $(BUILD)/tests/boot_console_test \
	$(BUILD)/tests/gamma_capability_test \
	$(BUILD)/tests/shared_event_ports_test $(BUILD)/tests/gl_surfaces_test \
	$(BUILD)/tests/surface_rows_test $(BUILD)/tests/surface_limits_test $(BUILD)/tests/accel_types_test \
	$(BUILD)/tests/pci_class_test \
	$(BUILD)/tests/defer_carry_test $(BUILD)/tests/arm64_region_test $(BUILD)/tests/features_arm64_test \
	$(BUILD)/tests/legacy_overrides_test $(BUILD)/tests/legacy_swap_queue_test \
	$(BUILD)/tests/mobile_modes_test $(BUILD)/tests/swap_rate_test \
	$(BUILD)/tests/legacy_swap_info_test $(BUILD)/tests/mobile_color_test $(BUILD)/tests/mobile_icc_test \
	$(BUILD)/tests/mobile_cursor_test $(BUILD)/tests/display_power_test \
	$(BUILD)/tests/control_ring_test $(BUILD)/tests/control_ring_arm64_test \
	$(BUILD)/tests/completion_arm_test $(BUILD)/tests/queue_wakes_test $(BUILD)/tests/kept_service_test \
	$(BUILD)/tests/queue_publish_test $(BUILD)/tests/doorbell_pages_test $(BUILD)/tests/top_down_test \
	$(BUILD)/tests/display_table_test $(BUILD)/tests/agdc_displays_test \
	$(BUILD)/tests/pointer_over_test $(BUILD)/tests/display_bounds_test $(BUILD)/tests/display_places_test \
	$(BUILD)/tests/pointer_motion_test $(BUILD)/tests/pointer_motion_arm64_test \
	$(BUILD)/tests/register_hold_test $(BUILD)/tests/vt_routes_test \
	$(BUILD)/tests/vt_ring_test $(BUILD)/tests/pat_test \
	$(BUILD)/tests/block_zero_test $(BUILD)/tests/block_zero_arm64_test $(BUILD)/tests/pool_blocks_test \
	$(BUILD)/tests/lifecycle_answers_test $(BUILD)/tests/lifecycle_answers_arm64_test

$(BUILD)/tests/%: tests/%.cpp $(HEADERS) ../../crates/reims-vgpu-graphics/include/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	xcrun clang++ -std=c++17 -Wall -Wextra -Werror -Isrc -I../../crates/reims-vgpu-graphics/include $< -o $@

# The region's rules as the arm64e kext builds them (RgfxRegionMapping.hpp).
$(BUILD)/tests/arm64_region_test: tests/arm64_region_test.cpp $(HEADERS) ../../crates/reims-vgpu-graphics/include/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	xcrun clang++ -std=c++17 -Wall -Wextra -Werror -DRGFX_GUEST_ARM64=1 -Isrc -I../../crates/reims-vgpu-graphics/include $< -o $@

# Ring 0's writes as the arm64e kext makes them (RgfxControlRing.hpp): no write-combined part.
$(BUILD)/tests/control_ring_arm64_test: tests/control_ring_test.cpp $(HEADERS) ../../crates/reims-vgpu-graphics/include/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	xcrun clang++ -std=c++17 -Wall -Wextra -Werror -DRGFX_GUEST_ARM64=1 -Isrc -I../../crates/reims-vgpu-graphics/include $< -o $@

# The guest's mouse as the arm64e kext builds it (RgfxPointerMotion.hpp): no pan.
$(BUILD)/tests/pointer_motion_arm64_test: tests/pointer_motion_test.cpp $(HEADERS) ../../crates/reims-vgpu-graphics/include/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	xcrun clang++ -std=c++17 -Wall -Wextra -Werror -DRGFX_GUEST_ARM64=1 -Isrc -I../../crates/reims-vgpu-graphics/include $< -o $@

# The GPU's `model` as the arm64e kext states it (RgfxDeviceName.hpp): the host's name whole.
$(BUILD)/tests/device_name_arm64_test: tests/device_name_test.cpp $(HEADERS) ../../crates/reims-vgpu-graphics/include/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	xcrun clang++ -std=c++17 -Wall -Wextra -Werror -DRGFX_GUEST_ARM64=1 -Isrc -I../../crates/reims-vgpu-graphics/include $< -o $@

# The features as the arm64e kext takes them (RgfxFeatures.hpp's PAGES_16K).
$(BUILD)/tests/features_arm64_test: tests/features_test.cpp $(HEADERS) ../../crates/reims-vgpu-graphics/include/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	xcrun clang++ -std=c++17 -Wall -Wextra -Werror -DRGFX_GUEST_ARM64=1 -Isrc -I../../crates/reims-vgpu-graphics/include $< -o $@

# What the arm64e kext zeroes of a new block (RgfxBlockZero.hpp): every span write-back, in its
# 16 KiB pages.
$(BUILD)/tests/block_zero_arm64_test: tests/block_zero_test.cpp $(HEADERS) ../../crates/reims-vgpu-graphics/include/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	xcrun clang++ -std=c++17 -Wall -Wextra -Werror -DRGFX_GUEST_ARM64=1 -Isrc -I../../crates/reims-vgpu-graphics/include $< -o $@

# What the arm64e kext publishes of the host's video (RgfxVideoCodecs.hpp): nothing.
$(BUILD)/tests/video_codecs_arm64_test: tests/video_codecs_test.cpp $(HEADERS) ../../crates/reims-vgpu-graphics/include/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	xcrun clang++ -std=c++17 -Wall -Wextra -Werror -DRGFX_GUEST_ARM64=1 -Isrc -I../../crates/reims-vgpu-graphics/include $< -o $@

# The answers the kext's waiters take (RgfxLifecycleAnswers.hpp), checked against the reply types
# the postLifecycle calls in src/ wait for, which the test reads from RGFX_KEXT_SRC as it runs.
$(BUILD)/tests/lifecycle_answers_test: tests/lifecycle_answers_test.cpp $(HEADERS) ../../crates/reims-vgpu-graphics/include/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	xcrun clang++ -std=c++17 -Wall -Wextra -Werror '-DRGFX_KEXT_SRC="$(CURDIR)/src"' -Isrc -I../../crates/reims-vgpu-graphics/include $< -o $@

# The answers the arm64e kext's waiters take: VT_STATUS too.
$(BUILD)/tests/lifecycle_answers_arm64_test: tests/lifecycle_answers_test.cpp $(HEADERS) ../../crates/reims-vgpu-graphics/include/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	xcrun clang++ -std=c++17 -Wall -Wextra -Werror -DRGFX_GUEST_ARM64=1 '-DRGFX_KEXT_SRC="$(CURDIR)/src"' -Isrc -I../../crates/reims-vgpu-graphics/include $< -o $@

test: $(TESTS)
	@for t in $(TESTS); do $$t || exit 1; done

clean:
	rm -rf $(BUILD)

.PHONY: all check test clean
