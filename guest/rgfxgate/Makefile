# RgfxGate: a Graft client kext (docs/graphics-kext). It lets macOS load exactly
# our reims-graphics Metal bundle with SIP fully on and installs that bundle
# itself at first boot. Builds on a Mac with Xcode's clang/ld against the SDK's
# Kernel.framework and Graft's header and start glue (../graft), with nothing
# fetched. Output under build/, never committed.
#
#   make BUNDLE_DIR=<signed reims-graphics.bundle dir>   build/RgfxGate.kext
#   make test                                            native tests of the matching and preference logic,
#                                                        the tree removal, and when the payload is made again
#   make clean
#
# BUNDLE_DIR is the signed bundle to embed; packaging reads its cdhash with
# codesign and writes build/gen/RgfxGatePayload.h. The payload is made again
# whenever the bundle it would embed changes: a BUNDLE_DIR or a cdhash other
# than the last packed (build/gen/bundle-dir holds both), or a file of
# BUNDLE_DIR newer than it. A make with no BUNDLE_DIR keeps the payload made
# last.

NAME    = RgfxGate
VERSION = 1.0.0
SDK    ?= $(shell xcrun --sdk macosx --show-sdk-path)
GRAFT   = ../graft
KHEADERS = $(SDK)/System/Library/Frameworks/Kernel.framework/Headers
BUILD  ?= build
GEN     = $(BUILD)/gen
KEXT    = $(BUILD)/$(NAME).kext
BIN     = $(KEXT)/Contents/MacOS/$(NAME)
PLIST   = $(KEXT)/Contents/Info.plist
PAYLOAD = $(GEN)/RgfxGatePayload.h
BUNDLE_DIR ?=
BUNDLE_STAMP = $(GEN)/bundle-dir

# The guest's architecture: x86_64 (macOS 11 and later) or arm64e (macOS 14 and later, built for
# a deployment target of 13.0). The red zone is x86's alone.
ARCH_NAME ?= x86_64
ifeq ($(ARCH_NAME),arm64e)
MIN_OS = 13.0
REDZONE =
else
MIN_OS = 11.0
REDZONE = -mno-red-zone
endif
FLAGS = -arch $(ARCH_NAME) -mmacosx-version-min=$(MIN_OS) -isysroot $(SDK) -mkernel -nostdlib -fno-builtin -fno-stack-protector \
	$(REDZONE) -Os -Wall -Wextra -Werror -Wno-unused-parameter
CXXFLAGS = $(FLAGS) -MMD -MP -std=c++17 -fapple-kext -fno-rtti -fno-exceptions \
	-DKERNEL -DKERNEL_PRIVATE -nostdinc -isystem $(KHEADERS) \
	-DGRAFT_CLIENT_ID=ai.aneesiqbal.RgfxGate -DGRAFT_CLIENT_VERSION=\"$(VERSION)\" \
	-I$(GEN) -Isrc -I$(GRAFT)/sdk/include

OBJS = $(BUILD)/obj/RgfxGate.o $(BUILD)/obj/RgfxGateHooks.o \
	$(BUILD)/obj/RgfxGateInstall.o $(BUILD)/obj/RgfxGatePrefs.o $(BUILD)/obj/GraftStart.o

.PHONY: all clean test FORCE
all: $(PLIST)

-include $(OBJS:.o=.d)
$(OBJS): Makefile

# A payload over another bundle than the one given would gate-tag a bundle that is not
# installed, and the given one would load untagged. So the payload is packed again whenever
# BUNDLE_DIR's path and cdhash differ from those the stamp holds from the last pack, which no
# file time decides (make compares whole seconds), and whenever a file of BUNDLE_DIR is newer.
ifneq ($(BUNDLE_DIR),)
BUNDLE_KEY := $(abspath $(BUNDLE_DIR)) $(shell codesign -dvvv '$(BUNDLE_DIR)' 2>&1 | sed -n 's/^CDHash=//p')
$(PAYLOAD): $(shell find '$(BUNDLE_DIR)' -type f 2>/dev/null)
ifneq ($(BUNDLE_KEY),$(shell cat '$(BUNDLE_STAMP)' 2>/dev/null))
$(PAYLOAD): FORCE
endif
endif

$(PAYLOAD):
	@test -n "$(BUNDLE_DIR)" || { echo "set BUNDLE_DIR=<signed reims-graphics.bundle dir>" >&2; exit 2; }
	@mkdir -p $(GEN)
	python3 -I tools/pack-payload.py "$(BUNDLE_DIR)" $(PAYLOAD) $(ARCH_NAME)
	@echo '$(BUNDLE_KEY)' > $(BUNDLE_STAMP)

$(BUILD)/obj/RgfxGate.o: src/RgfxGate.cpp src/RgfxGate.hpp $(PAYLOAD)
	@mkdir -p $(BUILD)/obj
	xcrun --sdk macosx clang++ $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/RgfxGateHooks.o: src/RgfxGateHooks.cpp src/RgfxGate.hpp src/rgfx_match.h $(PAYLOAD)
	@mkdir -p $(BUILD)/obj
	xcrun --sdk macosx clang++ $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/RgfxGateInstall.o: src/RgfxGateInstall.cpp src/RgfxGate.hpp src/RgfxGatePrefs.hpp src/RgfxGateFile.hpp src/rgfx_tree.h $(PAYLOAD)
	@mkdir -p $(BUILD)/obj
	xcrun --sdk macosx clang++ $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/RgfxGatePrefs.o: src/RgfxGatePrefs.cpp src/RgfxGatePrefs.hpp src/RgfxGateFile.hpp src/rgfx_gva_pref.h
	@mkdir -p $(BUILD)/obj
	xcrun --sdk macosx clang++ $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/GraftStart.o: $(GRAFT)/sdk/Start.cpp
	@mkdir -p $(BUILD)/obj
	xcrun --sdk macosx clang++ $(CXXFLAGS) -c $< -o $@

$(BIN): $(OBJS)
	@mkdir -p $(KEXT)/Contents/MacOS
	xcrun --sdk macosx ld -arch $(ARCH_NAME) -kext -undefined dynamic_lookup $(OBJS) -o $(BIN)

$(KEXT): $(PLIST)

$(PLIST): $(BIN) Info.plist Makefile
	sed 's/@VERSION@/$(VERSION)/g' Info.plist > $(KEXT)/Contents/Info.plist
	@nm $(BIN) | grep ' _kmod_info$$' > /dev/null || { echo "no _kmod_info in $(BIN): OpenCore refuses it" >&2; exit 1; }
	@echo "built $(KEXT)"

test:
	@mkdir -p $(BUILD)/obj
	xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/match_test.cpp -o $(BUILD)/obj/match_test
	$(BUILD)/obj/match_test
	xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/amfi_test.cpp -o $(BUILD)/obj/amfi_test
	$(BUILD)/obj/amfi_test
	xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/gva_pref_test.cpp -framework CoreFoundation -o $(BUILD)/obj/gva_pref_test
	$(BUILD)/obj/gva_pref_test
	xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/tree_test.cpp -o $(BUILD)/obj/tree_test
	$(BUILD)/obj/tree_test
	bash tests/payload_test.sh $(BUILD)/payload-test

clean:
	rm -rf $(BUILD)/obj $(BUILD)/gen $(KEXT)
