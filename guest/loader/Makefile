# ReimsGraphicsLoader: the Lilu plugin that binds ReimsGraphics, injected
# dormant into the Boot kernel collection, to IOGraphicsFamily and
# AppleGraphicsDeviceControl in the System collection (README.md). Builds on a
# Mac with Xcode's clang and ld against a DEBUG Lilu (its plugin headers and
# plugin_start.cpp) and MacKernelSDK in DEPS, which ../rgfxgate/tools/fetch-lilu.sh
# fetches, pinned, where DEPS has none.
#
#   make KDK=<a macOS 26 KDK's Extensions directory> [KEXT=<ReimsGraphics.kext>]
#       build/ReimsGraphicsLoader.kext, compiled with the tables of one
#       ReimsGraphics build (KEXT, by default ../kext/build/ReimsGraphics.kext),
#       and build/staged/: that kext as OpenCore injects it (no personalities,
#       its System-collection libraries replaced by its shims) and the shims,
#       RgfxIOGraphicsFamilyBridge.kext and
#       RgfxAppleGraphicsDeviceControlBridge.kext. The KDK's two libraries and
#       kernel are only read, for their exports and their classes' layouts.
#       Runs the layout checks (tests/layouts_test.cpp) against the tables
#       before it compiles the loader.
#   make test   the host test of the loader's site arithmetic (tests/fixups_test.cpp)
#   make clean  remove build/
#
# The loader holds the offset of every site, and the personalities, of the
# ReimsGraphics build it was made from, so it binds only that build: a new
# ReimsGraphics needs a new loader (README.md). Everything lands in build/,
# never committed.

NAME     = ReimsGraphicsLoader
VERSION  = 1.0.0
BUILD   ?= build
DEPS    ?= $(BUILD)/deps
KDK     ?=
KEXT    ?= ../kext/build/ReimsGraphics.kext
CONSUMER = ReimsGraphics
PREFIX   = Rgfx

RES      = $(DEPS)/Lilu.kext/Contents/Resources
KSDK     = $(DEPS)/MacKernelSDK
GEN      = $(BUILD)/gen
OBJ      = $(BUILD)/obj
STAGED   = $(BUILD)/staged
BUNDLE   = $(BUILD)/$(NAME).kext
BIN      = $(BUNDLE)/Contents/MacOS/$(NAME)

# The System-collection libraries ReimsGraphics links against, in the KDK.
LIBRARIES = $(KDK)/IOGraphicsFamily.kext/IOGraphicsFamily \
	$(KDK)/AppleGraphicsControl.kext/Contents/PlugIns/AppleGraphicsDeviceControl.kext/Contents/MacOS/AppleGraphicsDeviceControl

FLAGS = -arch x86_64 -mkernel -nostdlib -fno-builtin -fno-stack-protector \
	-mno-red-zone -Os -Wall -Wextra -Werror
CXXFLAGS = $(FLAGS) -std=c++17 -fapple-kext -fno-rtti -fno-exceptions \
	-DKERNEL -DKERNEL_PRIVATE -DLILU_KEXTPATCH_SUPPORT \
	-DPRODUCT_NAME=$(NAME) -DMODULE_VERSION=$(VERSION) \
	-I$(GEN) -Isrc -isystem $(RES) -isystem $(KSDK)/Headers
HOSTCXX = xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror

.PHONY: all test clean FORCE
all: $(BUNDLE)

# The shims, the staged consumer and the tables, made again on every build
# (they take a second): the KDK and the consumer are inputs make cannot
# date. tables.h is replaced only when it changes, so an unchanged table
# compiles nothing.
$(GEN)/tables.h: FORCE
	@test -n "$(KDK)" || { echo "set KDK=<a macOS 26 KDK's Extensions directory>" >&2; exit 2; }
	@test -f "$(KEXT)/Contents/MacOS/$(CONSUMER)" || { echo "no $(CONSUMER) kext at $(KEXT)" >&2; exit 2; }
	rm -rf $(BUILD)/shims $(STAGED)
	@mkdir -p $(GEN)
	python3 -I -B tools/gen-shims.py "$(KEXT)/Contents/MacOS/$(CONSUMER)" $(LIBRARIES) \
		--prefix $(PREFIX) --out $(BUILD)/shims
	python3 -I -B tools/stage-consumer.py "$(KEXT)" $(BUILD)/shims $(STAGED)
	python3 -I -B tools/gen-tables.py $(STAGED) $(GEN)/tables.h.new $(CONSUMER)
	@cmp -s $(GEN)/tables.h.new $@ && rm $(GEN)/tables.h.new || mv $(GEN)/tables.h.new $@

# The layout checks, on this Mac, against the tables just generated.
$(OBJ)/layouts.ok: tests/layouts_test.cpp src/Rows.h $(GEN)/tables.h
	@mkdir -p $(OBJ)
	$(HOSTCXX) -I$(GEN) -Isrc tests/layouts_test.cpp -o $(OBJ)/layouts_test
	$(OBJ)/layouts_test
	@touch $@

$(RES)/Library/plugin_start.cpp:
	bash ../rgfxgate/tools/fetch-lilu.sh "$(DEPS)"

$(OBJ)/Loader.o: src/Loader.cpp src/Fixup.h src/Rows.h src/Log.h $(GEN)/tables.h $(OBJ)/layouts.ok \
		$(RES)/Library/plugin_start.cpp Makefile
	xcrun --sdk macosx clang++ $(CXXFLAGS) -c $< -o $@

$(OBJ)/plugin_start.o: $(RES)/Library/plugin_start.cpp Makefile
	@mkdir -p $(OBJ)
	xcrun --sdk macosx clang++ $(CXXFLAGS) -Wno-unused-parameter -c $< -o $@

$(BIN): $(OBJ)/Loader.o $(OBJ)/plugin_start.o
	@mkdir -p $(@D)
	xcrun --sdk macosx ld -arch x86_64 -kext -undefined dynamic_lookup $^ -o $@

# grep reads all of nm's output: with -q, nm could die of SIGPIPE, which
# pipefail would report as a missing symbol.
$(BUNDLE): $(BIN) src/Info.plist
	sed 's/@VERSION@/$(VERSION)/g' src/Info.plist > $(BUNDLE)/Contents/Info.plist
	plutil -lint $(BUNDLE)/Contents/Info.plist > /dev/null
	@nm $(BIN) | grep ' _kmod_info$$' > /dev/null || { echo "no _kmod_info in $(BIN): OpenCore refuses it" >&2; exit 1; }
	@echo "built $(BUNDLE) for $(CONSUMER) $$(shasum -a 256 < "$(KEXT)/Contents/MacOS/$(CONSUMER)" | cut -c1-16)"

test:
	@mkdir -p $(OBJ)
	$(HOSTCXX) -Isrc tests/fixups_test.cpp -o $(OBJ)/fixups_test
	$(OBJ)/fixups_test

clean:
	rm -rf $(BUILD)
