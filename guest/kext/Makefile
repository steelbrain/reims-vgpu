# The reims-graphics display kext for x86-64 macOS 26 guests: ReimsGraphics,
# an IOFramebuffer on the device's PCI function (src/ReimsGraphics.hpp; the
# protocol is docs/graphics-kext/design.md). Builds on a Mac with Xcode's
# clang and ld, against the macOS SDK's Kernel.framework, for macOS 26.
#
#   make        build/ReimsGraphics.kext
#   make check  check its imports and IOFramebuffer's layout against the
#               guest's KDK, and link it with OpenCore's own linker in user
#               space (tools/check-imports.sh, tools/check-layout.py,
#               tools/oc-link-check.sh):
#                 KDK=<the guest release's KDK's Extensions directory>
#                 KEXTINJECT=<OpenCorePkg's TestKextInject binary>
#                 BOOT_KC=<the guest's x86_64 BootKernelExtensions.kc>
#               Without KEXTINJECT and BOOT_KC the link is skipped, saying so.
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
KEXTINJECT ?=
BOOT_KC    ?=

TRACE   ?= 0
DTRACE  ?= 0

ARCH = -arch x86_64 -isysroot $(SDK) -mmacosx-version-min=26.0

FLAGS = $(ARCH) -mkernel -nostdinc -isystem $(KERNEL) -I../../crates/reims-graphics/include \
	-DKERNEL -DBUNDLE_VERSION='"$(VERSION)"' -DRGFX_FB_TRACE=$(TRACE) \
	-Os -Wall -Wextra -Wshadow-all -Werror -Wno-unused-parameter \
	-Wno-deprecated-declarations

CXXFLAGS = $(FLAGS) -std=c++17 -fapple-kext -fno-rtti -fno-exceptions
CFLAGS   = $(FLAGS) -std=c11

HEADERS = $(wildcard src/*.hpp)

# The Stage-2 framebuffer (ReimsGraphics), the Stage-3 Metal accelerator
# (RgfxAccelerator) and its region-map + doorbell user client, and the device
# control (RgfxDeviceControl), all one kext.
OBJECTS = $(BUILD)/obj/ReimsGraphics.o \
	$(BUILD)/obj/RgfxAccelerator.o \
	$(BUILD)/obj/RgfxDeviceControl.o \
	$(BUILD)/obj/RgfxAcceleratorUserClient.o \
	$(BUILD)/obj/RgfxDisplayPipeUserClient.o \
	$(BUILD)/obj/kmod_info.o \
	$(BUILD)/obj/RgfxAcceleratorContexts.o \
	$(BUILD)/obj/RgfxAcceleratorUserClientContexts.o \
	$(BUILD)/obj/RgfxAcceleratorSurfaces.o \
	$(BUILD)/obj/RgfxSurfaceUserClient.o \
	$(BUILD)/obj/RgfxAcceleratorUserClientWindow.o \
	$(BUILD)/obj/RgfxAcceleratorLocks.o \
	$(BUILD)/obj/RgfxSharedEventUserClient.o

all: $(BINARY)

$(BUILD)/obj/%.o: src/%.cpp $(HEADERS) ../../crates/reims-graphics/include/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	xcrun clang++ $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/%.o: src/%.c Makefile
	@mkdir -p $(@D)
	xcrun clang $(CFLAGS) -c $< -o $@

# The driver's source with its debug information, which check-layout.py reads
# for the SDK's IOFramebuffer.
$(BUILD)/obj/layout.o: src/ReimsGraphics.cpp src/ReimsGraphics.hpp ../../crates/reims-graphics/include/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	xcrun clang++ $(CXXFLAGS) -g -fstandalone-debug -c $< -o $@

$(BINARY): $(OBJECTS) src/Info.plist
	@mkdir -p $(@D)
	xcrun clang++ $(ARCH) -nostdlib -Xlinker -kext $(OBJECTS) -lcc_kext -o $@
	sed 's/@VERSION@/$(VERSION)/g' src/Info.plist > $(KEXT)/Contents/Info.plist
	$(if $(filter 1,$(DTRACE)),plutil -insert OSBundleForceDTraceInit -bool YES $(KEXT)/Contents/Info.plist)
	plutil -lint $(KEXT)/Contents/Info.plist > /dev/null

check: $(BINARY) $(BUILD)/obj/layout.o
	bash tools/check-imports.sh "$(KDK)" $(KEXT)
	python3 -I tools/check-layout.py "$(KDK)/IOGraphicsFamily.kext.dSYM" $(BUILD)/obj/layout.o
	bash tools/oc-link-check.sh "$(KEXTINJECT)" "$(BOOT_KC)" "$(KDK)" $(KEXT) $(BUILD)/oc-link

# The native tests: header-only rules of the kext's, built for this Mac.
TESTS = $(BUILD)/tests/spans_test $(BUILD)/tests/blocks_test $(BUILD)/tests/surface_blocks_test \
	$(BUILD)/tests/surface_test $(BUILD)/tests/pipe_clients_test $(BUILD)/tests/pipe_transactions_test \
	$(BUILD)/tests/window_test $(BUILD)/tests/scanout_owner_test \
	$(BUILD)/tests/display_modes_test $(BUILD)/tests/surface_locks_test \
	$(BUILD)/tests/bundle_paths_test $(BUILD)/tests/surface_caches_test \
	$(BUILD)/tests/vblank_clock_test $(BUILD)/tests/submit_gate_test $(BUILD)/tests/memory_size_test \
	$(BUILD)/tests/device_name_test \
	$(BUILD)/tests/queue_limits_test $(BUILD)/tests/user_client_words_test \
	$(BUILD)/tests/features_test $(BUILD)/tests/latency_test $(BUILD)/tests/release_table_test \
	$(BUILD)/tests/deferred_answers_test $(BUILD)/tests/video_codecs_test \
	$(BUILD)/tests/ring_room_test $(BUILD)/tests/context_slots_test $(BUILD)/tests/pages_block_test \
	$(BUILD)/tests/close_table_test $(BUILD)/tests/ask_gate_test \
	$(BUILD)/tests/pipe_color_test $(BUILD)/tests/id_chunks_test $(BUILD)/tests/boot_console_test \
	$(BUILD)/tests/gamma_capability_test \
	$(BUILD)/tests/shared_event_ports_test

$(BUILD)/tests/%: tests/%.cpp $(HEADERS) ../../crates/reims-graphics/include/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	xcrun clang++ -std=c++17 -Wall -Wextra -Werror -Isrc -I../../crates/reims-graphics/include $< -o $@

test: $(TESTS)
	@for t in $(TESTS); do $$t || exit 1; done

clean:
	rm -rf $(BUILD)

.PHONY: all check test clean
