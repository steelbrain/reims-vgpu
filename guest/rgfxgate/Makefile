# RgfxGate: a Lilu plugin kext (docs/graphics-kext). It lets macOS load exactly
# our reims-graphics Metal bundle with SIP fully on and installs that bundle
# itself at first boot. Builds on a Mac with Xcode's clang/ld against a DEBUG
# Lilu (its plugin headers and plugin_start.cpp) and MacKernelSDK, both in
# build/deps (tools/fetch-lilu.sh fetches them, pinned). Output under build/,
# never committed.
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
DEPS   ?= build/deps
RES     = $(DEPS)/Lilu.kext/Contents/Resources
KSDK    = $(DEPS)/MacKernelSDK
BUILD  ?= build
GEN     = $(BUILD)/gen
KEXT    = $(BUILD)/$(NAME).kext
BIN     = $(KEXT)/Contents/MacOS/$(NAME)
PAYLOAD = $(GEN)/RgfxGatePayload.h
BUNDLE_DIR ?=
BUNDLE_STAMP = $(GEN)/bundle-dir

FLAGS = -arch x86_64 -mkernel -nostdlib -fno-builtin -fno-stack-protector \
	-mno-red-zone -Os -Wall -Wextra -Werror -Wno-unused-parameter
CXXFLAGS = $(FLAGS) -std=c++17 -fapple-kext -fno-rtti -fno-exceptions \
	-DKERNEL -DKERNEL_PRIVATE -DLILU_KEXTPATCH_SUPPORT \
	-DPRODUCT_NAME=$(NAME) -DMODULE_VERSION=$(VERSION) \
	-I$(GEN) -Isrc -isystem $(RES) -isystem $(KSDK)/Headers

OBJS = $(BUILD)/obj/RgfxGate.o $(BUILD)/obj/RgfxGateHooks.o \
	$(BUILD)/obj/RgfxGateInstall.o $(BUILD)/obj/RgfxGatePrefs.o $(BUILD)/obj/plugin_start.o

.PHONY: all clean test FORCE
all: $(KEXT)

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
	@test -d "$(DEPS)/Lilu.kext" && test -d "$(KSDK)" || bash tools/fetch-lilu.sh "$(DEPS)"
	@mkdir -p $(GEN)
	python3 -I tools/pack-payload.py "$(BUNDLE_DIR)" $(PAYLOAD)
	@echo '$(BUNDLE_KEY)' > $(BUNDLE_STAMP)

$(BUILD)/obj/RgfxGate.o: src/RgfxGate.cpp src/RgfxGate.hpp $(PAYLOAD)
	@mkdir -p $(BUILD)/obj
	xcrun --sdk macosx clang++ $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/RgfxGateHooks.o: src/RgfxGateHooks.cpp src/RgfxGate.hpp src/rgfx_match.h $(PAYLOAD)
	@mkdir -p $(BUILD)/obj
	xcrun --sdk macosx clang++ $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/RgfxGateInstall.o: src/RgfxGateInstall.cpp src/RgfxGate.hpp src/RgfxGatePrefs.hpp src/rgfx_tree.h $(PAYLOAD)
	@mkdir -p $(BUILD)/obj
	xcrun --sdk macosx clang++ $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/RgfxGatePrefs.o: src/RgfxGatePrefs.cpp src/RgfxGatePrefs.hpp src/rgfx_gva_pref.h
	@mkdir -p $(BUILD)/obj
	xcrun --sdk macosx clang++ $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/plugin_start.o: $(RES)/Library/plugin_start.cpp
	@mkdir -p $(BUILD)/obj
	xcrun --sdk macosx clang++ $(CXXFLAGS) -c $< -o $@

$(BIN): $(OBJS)
	@mkdir -p $(KEXT)/Contents/MacOS
	xcrun --sdk macosx ld -arch x86_64 -kext -undefined dynamic_lookup $(OBJS) -o $(BIN)

$(KEXT): $(BIN) Info.plist
	sed 's/@VERSION@/$(VERSION)/g' Info.plist > $(KEXT)/Contents/Info.plist
	@nm $(BIN) | grep ' _kmod_info$$' > /dev/null || { echo "no _kmod_info in $(BIN): OpenCore refuses it" >&2; exit 1; }
	@echo "built $(KEXT)"

test:
	@mkdir -p $(BUILD)/obj
	xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/match_test.cpp -o $(BUILD)/obj/match_test
	$(BUILD)/obj/match_test
	xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/gva_pref_test.cpp -o $(BUILD)/obj/gva_pref_test
	$(BUILD)/obj/gva_pref_test
	xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror -Isrc tests/tree_test.cpp -o $(BUILD)/obj/tree_test
	$(BUILD)/obj/tree_test
	bash tests/payload_test.sh $(BUILD)/payload-test

clean:
	rm -rf $(BUILD)/obj $(BUILD)/gen $(KEXT)
