# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
# Graft: the kext, its host tests and its release. Builds on a Mac with
# Xcode, against the Kernel.framework headers in Xcode's macOS SDK, and
# fetches nothing (CONTRIBUTING.md).
#
#   make        build/Graft.kext
#   make test   build and run every host test; the Python tests run under
#               python3 and again under SYSTEM_PYTHON (/usr/bin/python3)
#               (tests/README.md)
#   make kernel-check  compile a test kext carrying a manifest under macOS 11
#               kext flags and link it
#   make kernel-check-arm64e  compile the neutral core and the arm64e layer
#               for arm64e (compile only)
#   make graft-arm64e  build/Graft-arm64e.kext, the arm64e kext, linked with
#               its undefined symbols left for the boot firmware to bind
#   make core-probe   build/GraftCoreProbe.kext, the test-only owned client
#   make route-probe  build/GraftRouteProbe.kext, the arm64e test-only owned
#               client that exercises the AMFI route layer
#   make examples  build both examples (examples/README.md) for x86-64 and arm64e
#   make lint   check formatting, license lines, text hygiene, doc links, and
#               that src/ and sdk/ name no kext but Graft and Apple's
#   make format apply clang-format and ruff's formatting
#   make clean  remove build/
#   make release  build both kexts from the committed sources, with the Xcode
#               .xcode-version names, into dist/graft-$(VERSION)/ and
#               dist/graft-$(VERSION).zip (docs/guides/releasing.md)
#
# Build intermediates stay in build/, and releases in dist/.

BUILD   ?= build
VERSION  = 0.2.0
HOSTCXX  = xcrun --sdk macosx clang++ -std=c++17 -Wall -Wextra -Werror
HOSTCC   = xcrun --sdk macosx clang -Wall -Wextra -Werror
SDK      = $(shell xcrun --sdk macosx --show-sdk-path)
KERNEL   = $(SDK)/System/Library/Frameworks/Kernel.framework/Headers
KEXTFLAGS = -arch x86_64 -isysroot $(SDK) -mmacosx-version-min=11.0 \
	-mkernel -nostdinc -isystem $(KERNEL) -DKERNEL -std=c++17 \
	-fapple-kext -fno-rtti -fno-exceptions -fno-stack-protector \
	-fno-builtin -mno-red-zone -Os -Wall -Wextra -Werror
KEXTFLAGS += -Isdk/include -DGRAFT_VERSION='"$(VERSION)"'
# The architecture-neutral core (src/*.cpp) and the x86-64 layer
# (src/x86_64/*.cpp). HEADERS covers both, and the SDK headers the kext
# compiles (the client interface and the manifest format), for dependency
# tracking.
SOURCES = $(wildcard src/*.cpp) $(wildcard src/x86_64/*.cpp)
HEADERS = $(wildcard src/*.h) $(wildcard src/x86_64/*.h) sdk/include/graft/Graft.h \
	sdk/include/graft/ManifestFormat.h
CORE = $(patsubst src/%.cpp,$(BUILD)/kernel/%.o,$(SOURCES))
BUNDLE = $(BUILD)/Graft.kext
BINARY = $(BUNDLE)/Contents/MacOS/Graft
# The arm64e layer: the neutral core and src/arm64e/, with no x86-64 red zone.
# Every layer has an arm64e sibling, so the kext links. Its undefined symbols are
# left for the boot firmware's placement to bind to vmapple kernel exports.
ARMFLAGS = -arch arm64e -isysroot $(SDK) -mmacosx-version-min=11.0 \
	-mkernel -nostdinc -isystem $(KERNEL) -DKERNEL -std=c++17 \
	-fapple-kext -fno-rtti -fno-exceptions -fno-stack-protector \
	-fno-builtin -Os -Wall -Wextra -Werror
ARMFLAGS += -Isdk/include -DGRAFT_VERSION='"$(VERSION)"'
ARM_SOURCES = $(wildcard src/*.cpp) $(wildcard src/arm64e/*.cpp)
ARM_HEADERS = $(HEADERS) $(wildcard src/arm64e/*.h)
ARM_CORE = $(patsubst src/%.cpp,$(BUILD)/arm64e/%.o,$(ARM_SOURCES))
ARM_BUNDLE = $(BUILD)/Graft-arm64e.kext
ARM_BINARY = $(ARM_BUNDLE)/Contents/MacOS/Graft
# graft-kext's objects. Its tests link every one but its command line, and read
# the consumers in tests/consumer/ and tests/synthesized/ as graft-kext builds
# them.
TOOL_OBJECTS = $(patsubst sdk/tool/%.cpp,$(BUILD)/tool/%.o,$(wildcard sdk/tool/*.cpp))
TOOL_LIBRARY = $(filter-out $(BUILD)/tool/cli.o,$(TOOL_OBJECTS))
TOOL_HEADERS = $(wildcard sdk/tool/*.h) sdk/include/graft/ManifestFormat.h
TOOL_FIXTURES = $(BUILD)/consumer/x86_64/Fixture.kext/Contents/MacOS/Fixture \
	$(BUILD)/consumer/arm64e/Fixture.kext/Contents/MacOS/Fixture \
	$(BUILD)/synthesized/SynthDisplay.kext/Contents/MacOS/SynthDisplay

.PHONY: all test kernel-check kernel-check-arm64e graft-arm64e core-probe route-probe examples lint format clean release
all: $(BINARY)

# The Python tests. make test runs each under python3 and again under
# SYSTEM_PYTHON, macOS's own Python, which may be 3.9: a consumer runs
# tools/vendor.py with whatever python3 its Mac has, and the tools and tests
# keep to the Python 3.9 that CONTRIBUTING.md names.
PYTHON_TESTS = tests/graft_kext_test.py tests/consumer_mk_test.py tests/sdk_standin_test.py \
	tests/hook_symbols_test.py tests/release_test.py tests/lint_test.py tests/vendor_test.py
SYSTEM_PYTHON ?= /usr/bin/python3

CLANG_FORMAT ?= clang-format
RUFF ?= ruff
FORMATTED = $(shell git ls-files '*.c' '*.cpp' '*.h' '*.m')

lint:
	python3 -I -B tools/internal/lint.py
	$(CLANG_FORMAT) --dry-run -Werror $(FORMATTED)
	$(RUFF) check --quiet .
	$(RUFF) format --check --quiet .

format:
	$(CLANG_FORMAT) -i $(FORMATTED)
	$(RUFF) check --quiet --fix .
	$(RUFF) format --quiet .

release:
	python3 -I -B tools/internal/release.py

$(BINARY): $(CORE) src/Info.plist Makefile
	@mkdir -p $(@D)
	xcrun --sdk macosx ld -arch x86_64 -kext -undefined dynamic_lookup $(CORE) -o $@
	sed -e 's/@VERSION@/$(VERSION)/g' src/Info.plist > $(BUNDLE)/Contents/Info.plist
	plutil -lint $(BUNDLE)/Contents/Info.plist

test: $(BUILD)/macho_test $(BUILD)/binding_test $(BUILD)/manifest_test $(BUILD)/x86_route_test $(BUILD)/arm64e_fixup_test $(BUILD)/arm64e_route_test $(BUILD)/arm64e_synthesize_test $(BUILD)/arm64e_vtable_disc_test $(BUILD)/arm64e_synthesize_vtable_test $(BUILD)/write_plan_test $(BUILD)/fixture.dylib $(BUILD)/kernel/Fixture $(BUILD)/graft-kext $(BUILD)/staging_test $(BUILD)/synth_emit_test $(BUILD)/prelink_test $(TOOL_FIXTURES)
	$(BUILD)/macho_test $(BUILD)/fixture.dylib
	$(BUILD)/binding_test
	$(BUILD)/manifest_test $(BUILD)/kernel/Fixture
	$(BUILD)/x86_route_test
	$(BUILD)/arm64e_fixup_test
	$(BUILD)/arm64e_route_test
	$(BUILD)/arm64e_synthesize_test
	$(BUILD)/arm64e_vtable_disc_test
	$(BUILD)/arm64e_synthesize_vtable_test
	$(BUILD)/write_plan_test
	$(BUILD)/staging_test tests/consumer/graft.json $(BUILD)/consumer/x86_64 $(BUILD)/consumer/arm64e
	$(BUILD)/synth_emit_test tests/synthesized/graft.json $(BUILD)/synthesized
	$(BUILD)/prelink_test $(BUILD)/graft-kext $(BUILD)/consumer/x86_64
	for python in python3 $(SYSTEM_PYTHON); do \
		for test in $(PYTHON_TESTS); do \
			echo "$$python -I -B $$test"; \
			GRAFT_KEXT=$(BUILD)/graft-kext "$$python" -I -B "$$test" || exit 1; \
		done; \
	done

# The fixup is architecture-dispatched (src/Fixup.h), so this test, which checks
# the x86-64 pointer and branch arithmetic, forces x86-64 to pick that sibling.
$(BUILD)/binding_test: tests/binding_test.cpp src/Binding.h src/Fixup.h src/x86_64/Fixup.h src/FixupShared.h src/Layout.h
	@mkdir -p $(BUILD)
	$(HOSTCXX) -arch x86_64 -Isrc tests/binding_test.cpp -o $@

$(BUILD)/x86_route_test: tests/x86_route_test.cpp src/x86_64/X86Route.h src/x86_64/Fixup.h src/FixupShared.h src/Policy.h
	@mkdir -p $(BUILD)
	$(HOSTCXX) -arch x86_64 -Isrc $< -o $@

# The arm64e fixup signs pointers, so this test runs native arm64e to exercise
# the real pointer-authentication path (round-trip sign then authenticate).
$(BUILD)/arm64e_fixup_test: tests/arm64e_fixup_test.cpp src/arm64e/Fixup.h src/FixupShared.h
	@mkdir -p $(BUILD)
	$(HOSTCXX) -arch arm64e -Isrc $< -o $@

# The arm64e route planner re-signs each shadow slot, so this test runs native
# arm64e to exercise the real pointer-authentication path, as the fixup test does.
$(BUILD)/arm64e_route_test: tests/arm64e_route_test.cpp src/arm64e/Arm64Route.h src/arm64e/Fixup.h src/FixupShared.h
	@mkdir -p $(BUILD)
	$(HOSTCXX) -arch arm64e -Isrc $< -o $@

# The synthesis identity matcher is architecture-neutral string logic, but it
# belongs to the arm64e display mechanism; build it for the host.
$(BUILD)/arm64e_synthesize_test: tests/arm64e_synthesize_test.cpp src/arm64e/Synthesize.h
	@mkdir -p $(BUILD)
	$(HOSTCXX) -Isrc $< -o $@

# The donor vtable-disc check re-signs each live slot, so this test runs native
# arm64e to exercise the real pointer-authentication path, as the fixup test does.
$(BUILD)/arm64e_vtable_disc_test: tests/arm64e_vtable_disc_test.cpp src/arm64e/VtableDisc.h src/Fixup.h src/arm64e/Fixup.h src/FixupShared.h
	@mkdir -p $(BUILD)
	$(HOSTCXX) -arch arm64e -Isrc $< -o $@

# The synthesized-vtable builder re-signs each slot, so this test runs native
# arm64e to exercise the real pointer-authentication path, as the fixup test does.
$(BUILD)/arm64e_synthesize_vtable_test: tests/arm64e_synthesize_vtable_test.cpp src/arm64e/SynthesizeVtable.h src/arm64e/VtableDisc.h src/arm64e/Fixup.h src/FixupShared.h
	@mkdir -p $(BUILD)
	$(HOSTCXX) -arch arm64e -Isrc $< -o $@

$(BUILD)/write_plan_test: tests/write_plan_test.cpp src/WritePlan.h
	@mkdir -p $(BUILD)
	$(HOSTCXX) -Isrc $< -o $@

$(BUILD)/manifest_test: tests/manifest_test.cpp $(HEADERS)
	@mkdir -p $(BUILD)
	$(HOSTCXX) -arch x86_64 -Isrc -Isdk/include tests/manifest_test.cpp -o $@

kernel-check: $(BUILD)/kernel/Fixture

$(BUILD)/arm64e/%.o: src/%.cpp $(ARM_HEADERS) Makefile
	@mkdir -p $(@D)
	xcrun --sdk macosx clang++ $(ARMFLAGS) -Isrc -c $< -o $@

# Compile the neutral core and the arm64e layer for arm64e. It guards that the
# core stays architecture-neutral and the arm64e layer builds; graft-arm64e
# links them into the kext.
kernel-check-arm64e: $(ARM_CORE)
	@echo "arm64e: the neutral core and the arm64e layer compile"

# The arm64e Graft kext. It links with its undefined symbols unresolved, and
# the boot firmware's placement binds each to a vmapple kernel export. The
# arm64e layer reaches OSKext only through OSKextLoadKextWithIdentifier, an
# export, and uses no PE_get_kc_header, which it could not bind.
graft-arm64e: $(ARM_BINARY)

$(ARM_BINARY): $(ARM_CORE) src/Info.plist Makefile
	@mkdir -p $(@D)
	xcrun --sdk macosx ld -arch arm64e -kext -undefined dynamic_lookup $(ARM_CORE) -o $@
	sed -e 's/@VERSION@/$(VERSION)/g' src/Info.plist > $(ARM_BUNDLE)/Contents/Info.plist
	plutil -lint $(ARM_BUNDLE)/Contents/Info.plist

core-probe: $(BUILD)/GraftCoreProbe.kext/Contents/MacOS/GraftCoreProbe

$(BUILD)/kernel/CoreProbe.o: tests/kexts/core-probe/CoreProbe.cpp sdk/include/graft/Graft.h Makefile
	@mkdir -p $(@D)
	xcrun --sdk macosx clang++ $(KEXTFLAGS) -c $< -o $@

$(BUILD)/kernel/CoreProbeStart.o: sdk/Start.cpp Makefile
	@mkdir -p $(@D)
	xcrun --sdk macosx clang++ $(KEXTFLAGS) \
	    -DGRAFT_CLIENT_ID=ai.aneesiqbal.graft.CoreProbe \
	    -DGRAFT_CLIENT_VERSION='"1.0.0"' -c $< -o $@

$(BUILD)/GraftCoreProbe.kext/Contents/MacOS/GraftCoreProbe: $(BUILD)/kernel/CoreProbe.o $(BUILD)/kernel/CoreProbeStart.o tests/kexts/core-probe/Info.plist
	@mkdir -p $(@D)
	xcrun --sdk macosx ld -arch x86_64 -kext -undefined dynamic_lookup $(filter %.o,$^) -o $@
	cp tests/kexts/core-probe/Info.plist $(@D)/../Info.plist
	plutil -lint $(@D)/../Info.plist

# The arm64e route probe: the owned-client glue and the probe, built arm64e, so
# its asm wrappers and the authenticated tail-branch exercise the AMFI route
# layer on a vmapple arm64e boot.
route-probe: $(BUILD)/GraftRouteProbe.kext/Contents/MacOS/GraftRouteProbe

$(BUILD)/arm64e/RouteProbe.o: tests/kexts/route-probe/RouteProbe.cpp sdk/include/graft/Graft.h Makefile
	@mkdir -p $(@D)
	xcrun --sdk macosx clang++ $(ARMFLAGS) -c $< -o $@

$(BUILD)/arm64e/RouteProbeStart.o: sdk/Start.cpp Makefile
	@mkdir -p $(@D)
	xcrun --sdk macosx clang++ $(ARMFLAGS) \
	    -DGRAFT_CLIENT_ID=ai.aneesiqbal.graft.RouteProbe \
	    -DGRAFT_CLIENT_VERSION='"1.0.0"' -c $< -o $@

$(BUILD)/GraftRouteProbe.kext/Contents/MacOS/GraftRouteProbe: $(BUILD)/arm64e/RouteProbe.o $(BUILD)/arm64e/RouteProbeStart.o tests/kexts/route-probe/Info.plist
	@mkdir -p $(@D)
	xcrun --sdk macosx ld -arch arm64e -kext -undefined dynamic_lookup $(filter %.o,$^) -o $@
	cp tests/kexts/route-probe/Info.plist $(@D)/../Info.plist
	plutil -lint $(@D)/../Info.plist

# Build both examples (examples/README.md) for x86-64 and arm64e through their
# own Makefiles, the way a consumer repository would.
examples:
	$(MAKE) -C examples/hello-consumer
	$(MAKE) -C examples/hello-consumer ARCH='-arch arm64e'
	$(MAKE) -C examples/hello-client
	$(MAKE) -C examples/hello-client ARCH='-arch arm64e'

$(BUILD)/kernel/%.o: src/%.cpp $(HEADERS) Makefile
	@mkdir -p $(@D)
	xcrun --sdk macosx clang++ $(KEXTFLAGS) -Isrc -c $< -o $@

$(BUILD)/kernel/Fixture.o: tests/kernel_fixture.cpp src/Manifest.h sdk/include/graft/ManifestFormat.h Makefile
	@mkdir -p $(@D)
	xcrun --sdk macosx clang++ $(KEXTFLAGS) -Isrc -c $< -o $@

$(BUILD)/kernel/Fixture: $(BUILD)/kernel/Fixture.o
	xcrun --sdk macosx ld -arch x86_64 -kext -undefined dynamic_lookup $^ -o $@

$(BUILD)/fixture.dylib: tests/fixture.c
	@mkdir -p $(BUILD)
	$(HOSTCC) -dynamiclib -o $@ $<

# graft-kext, the SDK's consumer build tool, from the sources a consumer's build
# compiles it from, here with warnings as errors.
$(BUILD)/tool/%.o: sdk/tool/%.cpp $(TOOL_HEADERS) Makefile
	@mkdir -p $(@D)
	$(HOSTCXX) -pedantic -O2 -Isdk/include -c $< -o $@

$(BUILD)/graft-kext: $(TOOL_OBJECTS)
	$(HOSTCXX) $^ -o $@

$(BUILD)/staging_test $(BUILD)/synth_emit_test $(BUILD)/prelink_test: $(BUILD)/%: tests/%.cpp tests/Check.h tests/ToolTest.h $(TOOL_HEADERS) $(TOOL_LIBRARY)
	$(HOSTCXX) -pedantic -Isdk/include -Isdk/tool $< $(TOOL_LIBRARY) -o $@

# The consumers the tool's tests read, each built by graft-kext's build as a
# consumer's build runs it, with every step's output left in its work directory:
# the consumer in tests/consumer/ for x86-64 and arm64e, and the synthesized
# display in tests/synthesized/ for arm64e.
$(BUILD)/consumer/x86_64/Fixture.o: tests/consumer/Fixture.cpp tests/consumer/Fixture.h Makefile
	@mkdir -p $(@D)
	xcrun --sdk macosx clang++ $(KEXTFLAGS) -c $< -o $@

$(BUILD)/consumer/arm64e/Fixture.o: tests/consumer/Fixture.cpp tests/consumer/Fixture.h Makefile
	@mkdir -p $(@D)
	xcrun --sdk macosx clang++ $(ARMFLAGS) -c $< -o $@

$(BUILD)/consumer/%/Fixture.kext/Contents/MacOS/Fixture: $(BUILD)/consumer/%/Fixture.o tests/consumer/graft.json tests/consumer/Info.plist $(BUILD)/graft-kext
	$(BUILD)/graft-kext build --name=Fixture --version=1.0.0 --arch='-arch $*' \
		--declaration=tests/consumer/graft.json --info-plist=tests/consumer/Info.plist \
		--work=$(BUILD)/consumer/$*/graft --output=$(BUILD)/consumer/$*/Fixture.kext \
		$< -- -arch $* -Itests/consumer

$(BUILD)/synthesized/SynthDisplay.o: tests/synthesized/SynthDisplay.cpp tests/synthesized/SynthDisplay.h sdk/include/graft/IOMobileFramebufferLegacy.h Makefile
	@mkdir -p $(@D)
	xcrun --sdk macosx clang++ $(ARMFLAGS) -Itests/synthesized -c $< -o $@

$(BUILD)/synthesized/SynthDisplay.kext/Contents/MacOS/SynthDisplay: $(BUILD)/synthesized/SynthDisplay.o tests/synthesized/graft.json tests/synthesized/Info.plist $(BUILD)/graft-kext
	$(BUILD)/graft-kext build --name=SynthDisplay --version=1.0.0 --arch='-arch arm64e' \
		--declaration=tests/synthesized/graft.json --info-plist=tests/synthesized/Info.plist \
		--work=$(BUILD)/synthesized/graft --output=$(BUILD)/synthesized/SynthDisplay.kext \
		$< -- -arch arm64e -Isdk/include -Itests/synthesized

$(BUILD)/macho_test: tests/macho_test.cpp src/MachO.h
	@mkdir -p $(BUILD)
	$(HOSTCXX) -Isrc tests/macho_test.cpp -o $@

clean:
	rm -rf $(BUILD)
