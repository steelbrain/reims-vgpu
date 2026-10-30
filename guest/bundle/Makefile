# The reims-graphics guest Metal-plugin bundle: the serialiser core and the Metal
# shell that drives it.
#
# The serialiser core (src/Rgfx{ObjectModel,Caps,RingWriter,Serializer,Encoder,
# WireLayout}.*) is our own code, depends only on the committed wire header, needs
# no Apple Metal SPI, and builds for the guest's arch (x86_64).
#
# The Metal shell is the ObjC++ layer Apple's Metal loads as the guest's MTLDevice:
#   - the NSObject-based object wrappers (src/RgfxResource.*, src/RgfxBuffer.*, and
#     the resource/pipeline/… wrappers the later slices add) subclass only NSObject,
#     so they instantiate and are unit-tested on the Mac; and
#   - the device shell (src/RgfxMetalDevice.*) subclasses an Apple-private class
#     whose symbols resolve only at load time in the guest, so it is COMPILED into
#     the bundle but excluded from the user-space unit test.
#
#   make            build the serialiser objects (x86_64, the guest's arch)
#   make test       build + run both checks: the serialiser check (x86_64, under
#                   Rosetta) and the Metal-shell wrapper/routing check (this Mac's
#                   native arch). The device shell is not in the test binary.
#   make bundle     compile + bundle-link the whole x86_64 loadable bundle, device
#                   shell included (links clean with -undefined dynamic_lookup; it
#                   only needs to COMPILE off the guest, not run).
#   make clean      remove build/
#
# Everything lands in build/, never committed.

SDK     ?= $(shell xcrun --sdk macosx --show-sdk-path)
# ARCH is the guest's arch (serialiser core + the bundle); NATIVE is this Mac's
# arch (the shell wrapper unit test). No inline comments on these two: make keeps
# the spaces before a `#`, and a trailing space breaks the object paths below.
ARCH    ?= x86_64
NATIVE  ?= $(shell uname -m)
WIRE     = ../../crates/reims-graphics/include
BUILD    = build

CXX      = xcrun clang++

# Shared flags. The serialiser check already builds -Wall -Wextra -Werror; the
# shell holds to the same bar.
CXXFLAGS = -isysroot $(SDK) -std=c++17 -Os \
	-Wall -Wextra -Werror -Wno-unused-parameter \
	-I src -I $(WIRE)

# ObjC++ (.mm) also needs ARC and four warning suppressions: our classes conform to
# the full Metal protocols (MTLDevice/MTLBuffer/…) but deliberately implement only
# the selectors a slice owns — the Apple-private base supplies the device's protocol
# properties/methods at load, and the later slices fill the wrappers — so the
# protocol-completeness and protocol-property-synthesis diagnostics are expected and
# not errors. (unguarded-availability-new: we target the running OS's SPI;
# deprecated-declarations: the device protocol has members Apple deprecated on
# macOS 27 — location/peer*/locationNumber/didModifyRange — that we still implement
# for MTLDevice/MTLBuffer conformance and older guests.)
OBJCXXFLAGS = -fobjc-arc \
	-Wno-objc-protocol-property-synthesis -Wno-protocol \
	-Wno-incomplete-implementation -Wno-unguarded-availability-new \
	-Wno-deprecated-declarations
FRAMEWORKS  = -framework Metal -framework Foundation

HEADERS      = $(wildcard src/*.hpp) $(wildcard src/*.h)
TESTHEADERS  = $(wildcard tests/*.hpp)

# Serialiser core (C++).
CORE_SRCS = src/RgfxObjectModel.cpp src/RgfxCaps.cpp src/RgfxRingWriter.cpp \
	src/RgfxSerializer.cpp src/RgfxEncoder.cpp src/RgfxWireLayout.cpp

# The whole Metal shell (ObjC++): wrappers + the device. Goes in the bundle.
SHELL_SRCS = src/RgfxResource.mm src/RgfxBuffer.mm src/RgfxMetalDevice.mm
# The NSObject-based wrappers alone (no Apple-private base): goes in the unit test.
WRAP_SRCS  = src/RgfxResource.mm src/RgfxBuffer.mm

# Objects are kept per-arch (so the x86_64 and native builds never clobber) and
# split core/ (C++) vs shell/ (ObjC++) so each compile rule has a unique target
# directory — GNU make 3.81 mishandles two pattern rules sharing one target dir.
CORE_OBJ_ARCH  = $(CORE_SRCS:src/%.cpp=$(BUILD)/obj/$(ARCH)/core/%.o)
SHELL_OBJ_ARCH = $(SHELL_SRCS:src/%.mm=$(BUILD)/obj/$(ARCH)/shell/%.o)
CORE_OBJ_NAT   = $(CORE_SRCS:src/%.cpp=$(BUILD)/obj/$(NATIVE)/core/%.o)
WRAP_OBJ_NAT   = $(WRAP_SRCS:src/%.mm=$(BUILD)/obj/$(NATIVE)/shell/%.o)

SER_TESTBIN   = $(BUILD)/serializer_tests
SHELL_TESTBIN = $(BUILD)/shell_tests
BUNDLEBIN     = $(BUILD)/reims-graphics.bundle

all: $(CORE_OBJ_ARCH)

# --- compile rules (per arch; C++ and ObjC++) --------------------------------

$(BUILD)/obj/$(ARCH)/core/%.o: src/%.cpp $(HEADERS) $(WIRE)/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(ARCH) $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/$(ARCH)/shell/%.o: src/%.mm $(HEADERS) $(WIRE)/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(ARCH) $(CXXFLAGS) $(OBJCXXFLAGS) -c $< -o $@

$(BUILD)/obj/$(NATIVE)/core/%.o: src/%.cpp $(HEADERS) $(WIRE)/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(NATIVE) $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/$(NATIVE)/shell/%.o: src/%.mm $(HEADERS) $(WIRE)/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(NATIVE) $(CXXFLAGS) $(OBJCXXFLAGS) -c $< -o $@

# --- the serialiser check (x86_64, Rosetta) ----------------------------------

$(SER_TESTBIN): $(CORE_OBJ_ARCH) tests/serializer_tests.cpp $(HEADERS) $(TESTHEADERS) Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(ARCH) $(CXXFLAGS) -I tests $(CORE_OBJ_ARCH) tests/serializer_tests.cpp -o $@

# --- the Metal-shell wrapper/routing check (native arch) ---------------------
# The NSObject wrappers + serialiser core + the routing test; no device shell.

$(SHELL_TESTBIN): $(CORE_OBJ_NAT) $(WRAP_OBJ_NAT) tests/shell_tests.mm $(HEADERS) $(TESTHEADERS) Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(NATIVE) $(CXXFLAGS) $(OBJCXXFLAGS) -I tests \
		$(CORE_OBJ_NAT) $(WRAP_OBJ_NAT) tests/shell_tests.mm $(FRAMEWORKS) -o $@

test: $(SER_TESTBIN) $(SHELL_TESTBIN)
	arch -$(ARCH) $(SER_TESTBIN)
	$(SHELL_TESTBIN)

# --- the loadable bundle (x86_64, device shell included) ---------------------
# Private base/SPI symbols resolve at load in the guest, so -undefined
# dynamic_lookup; the bundle is built to prove it compiles + links, not to run.

bundle: $(BUNDLEBIN)

$(BUNDLEBIN): $(CORE_OBJ_ARCH) $(SHELL_OBJ_ARCH) Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(ARCH) $(CXXFLAGS) $(OBJCXXFLAGS) \
		-bundle -undefined dynamic_lookup $(FRAMEWORKS) \
		$(CORE_OBJ_ARCH) $(SHELL_OBJ_ARCH) -o $@

clean:
	rm -rf $(BUILD)

.PHONY: all test bundle clean
