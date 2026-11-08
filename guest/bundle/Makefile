# The reims-graphics guest Metal-plugin bundle: the serialiser core and the Metal
# shell that drives it.
#
# The serialiser core (src/Rgfx{ObjectModel,Caps,RingWriter,Serializer,Encoder,
# WireLayout}.* and the rest of CORE_SRCS) is our own code, depends only on the
# committed wire header (and CommonCrypto, for a library's key), needs no Apple
# Metal SPI, and builds for the guest's arch (x86_64).
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
#   make test       build + run the checks: the serialiser check (x86_64, under
#                   Rosetta), the AIR argument reader's check (x86_64, under Rosetta,
#                   over .metallibs built from tests/air_arguments.metal and
#                   tests/air_arguments_old.metal, and the repo's committed metallibs), the
#                   Metal-shell wrapper/routing check (this Mac's native arch; the
#                   device shell is not in its binary), and the device's base-class
#                   check (x86_64, under Rosetta), which loads the bundle
#                   (tests/inherited_tests.mm), the protocol members' check against
#                   their register (x86_64, under Rosetta, which loads the bundle:
#                   tests/protocol_members_tests.mm, tests/protocol_members.txt), and the
#                   video driver's H.264 and HEVC parameter-set checks and its entry
#                   points' check (x86_64, under Rosetta, tests/video_h264_tests.cpp,
#                   tests/video_h265_tests.cpp, tests/video_driver_tests.cpp).
#   make bundle     compile + bundle-link the whole x86_64 loadable bundle, device
#                   shell included (links clean with -undefined dynamic_lookup; it
#                   only needs to COMPILE off the guest, not run).
#   make sweep      the guest's correctness sweep of the system frameworks on the device
#                   (tools/sweep), build/sweep-x86_64; ARCH=arm64 for an Apple-silicon Mac.
#   make pipelines  shader and pipeline costs (tools/pipelines), build/pipelines-x86_64 and
#                   build/pipelines-metallibs/; ARCH=arm64 for an Apple-silicon Mac.
#   make bench      the GPU benchmark (tools/bench), build/bench-x86_64; ARCH=arm64 for an
#                   Apple-silicon Mac.
#   make res        what resources cost (tools/res), build/res-x86_64; ARCH=arm64 likewise.
#   make submit-probe
#                   a command buffer's fixed cost and latency (tools/submit_probe.m),
#                   build/submit_probe-x86_64; ARCH=arm64 likewise.
#   make upload-probe
#                   what the CPU's uploads into Managed resources cost at WindowServer's shapes
#                   (tools/upload_probe.m), build/upload_probe-x86_64; ARCH=arm64 likewise.
#   make reflection-check
#                   the stitchable functions' reflection the bundle builds against Metal's own,
#                   build/reflection_check-x86_64 (ARCH=arm64 likewise) over
#                   build/air_arguments.metallib.
#   make pipeline-reflection-check
#                   pipelines' reflection the bundle builds against Metal's own,
#                   build/pipeline_reflection_check-x86_64 (ARCH=arm64 likewise) over
#                   build/oracle_reflection.metallib (and its c_atomic_texture case over
#                   build/oracle_reflection_31.metallib).
#   make peer-test-build, make peer-test
#                   the wire round trip of the bundle's ring code against the host's library
#                   (tests/peer_tests.mm; native arch, this Mac's GPU, never in `make test`):
#                   build, then run holding the GPU lock. DIST names `cargo xtask dist`'s tree.
#   make spguard    what the kernel's VM guard makes of the device's superpages
#                   (tools/spguard), build/spguard-x86_64, run in a guest.
#   make clean      remove build/
#
# Everything lands in build/, never committed.

SDK     ?= $(shell xcrun --sdk macosx --show-sdk-path)
# ARCH is the guest's arch (serialiser core + the bundle); NATIVE is this Mac's
# arch (the shell wrapper unit test). No inline comments on these two: make keeps
# the spaces before a `#`, and a trailing space breaks the object paths below.
ARCH    ?= x86_64
NATIVE  ?= $(shell uname -m)
WIRE     = ../../crates/reims-vgpu-graphics/include
BUILD    = build

CXX      = xcrun clang++

# Shared flags. The serialiser check already builds -Wall -Wextra -Werror; the
# shell holds to the same bar. The deployment target is macOS 11.0, the oldest
# release the guest kexts serve: every symbol the SDK marks newer is then a weak
# import, which dyld leaves NULL on a release that lacks it, where a strong one
# would make dyld refuse the whole bundle. A tool that sets its own
# -mmacosx-version-min later on its line overrides it.
CXXFLAGS = -isysroot $(SDK) -mmacosx-version-min=11.0 -std=c++17 -Os \
	-Wall -Wextra -Werror -Wno-unused-parameter \
	-I src -I $(WIRE)

# ObjC++ (.mm) also needs ARC and three warning suppressions (unguarded-availability-new:
# we target the running OS's SPI; deprecated-declarations: the device protocol has members
# Apple deprecated on macOS 27 — location/peer*/locationNumber/didModifyRange — that we still
# implement for MTLDevice/MTLBuffer conformance and older guests; objc-protocol-method-
# implementation: the device's category split implements MTLDevice protocol methods in
# categories whose primary class declares the conformance).
OBJCXXFLAGS = -fobjc-arc -Wno-unguarded-availability-new -Wno-deprecated-declarations \
	-Wno-objc-protocol-method-implementation
# The protocol-completeness diagnostics: a Metal protocol member a wrapper class leaves out is
# a warning, never an error, so what is left shows on every build of the file. A member the
# device does not support is not left out: it is marked RGFX_UNSUPPORTED (src/RgfxUnsupported.h),
# and the bundle's build prints how many there are. The device's own files (RgfxMetalDevice*.mm)
# keep them quiet: its members live in its categories and in the IOGPU base, which these
# diagnostics do not see, so they would list every member. tools/protocol_audit.m checks
# the device at run time instead, through the base.
PROTOCOL_WARN  = -Wprotocol -Wincomplete-implementation -Wobjc-protocol-property-synthesis \
	-Wno-error=protocol -Wno-error=incomplete-implementation \
	-Wno-error=objc-protocol-property-synthesis
PROTOCOL_QUIET = -Wno-protocol -Wno-incomplete-implementation \
	-Wno-objc-protocol-property-synthesis
PROTOCOL_FLAGS = $(PROTOCOL_WARN)
FRAMEWORKS  = -framework Metal -framework Foundation -framework IOKit -framework IOSurface
# The device shell subclasses IOGPUMetalDevice, whose ObjC class symbol
# (_OBJC_CLASS_$_IOGPUMetalDevice) is a hard link-time dependency. It lives in
# the private IOGPU framework, which the SDK ships a .tbd stub for. We link it
# TWO-LEVEL (not dynamic_lookup) so the bundle records an LC_LOAD_DYLIB for
# IOGPU and dyld auto-loads it at the bundle's load in the guest, resolving the
# superclass — exactly as Apple's AppleParavirtGPUMetalIOGPUFamily.bundle does.
# A flat-namespace dynamic_lookup of this symbol does NOT resolve when Metal
# dlopens the plugin (Metal's process is two-level; IOGPU's symbols aren't in
# the flat namespace), so the class never realises and MTLCopyAllDevices=0.
PRIVFW      = -F $(SDK)/System/Library/PrivateFrameworks -framework IOGPU

HEADERS      = $(wildcard src/*.hpp) $(wildcard src/*.h) ../kext/src/RgfxUserClientABI.hpp
TESTHEADERS  = $(wildcard tests/*.hpp)

# Serialiser core (C++).
CORE_SRCS = src/RgfxObjectModel.cpp src/RgfxCaps.cpp src/RgfxRingWriter.cpp \
	src/RgfxSerializer.cpp src/RgfxEncoder.cpp src/RgfxWireLayout.cpp \
	src/RgfxRegion.cpp src/RgfxReplyReader.cpp src/RgfxLibraryKey.cpp \
	src/RgfxCompletionThread.cpp src/RgfxFunctionConstants.cpp \
	src/RgfxBinaryArchiveFile.cpp src/RgfxAirArguments.cpp src/RgfxKnownPipelines.cpp \
	src/RgfxSlabs.cpp src/RgfxSuperpages.cpp src/RgfxIcbSlotTable.cpp \
	src/RgfxLibraryStream.cpp src/RgfxMetallibWriter.cpp src/RgfxGLLowering.cpp \
	src/RgfxClMode.cpp

# The video driver AppleGVA loads from this bundle (C++, src/RgfxVideoDriver.cpp): bundle only.
VIDEO_SRCS = src/RgfxVideoDriver.cpp src/RgfxVideoH264.cpp src/RgfxVideoH265.cpp
VIDEO_OBJ_ARCH = $(VIDEO_SRCS:src/%.cpp=$(BUILD)/obj/$(ARCH)/core/%.o)
# The whole Metal shell (ObjC++): wrappers + the device. Goes in the bundle. Globbed
# so each later slice's wrappers + category are picked up with no Makefile edit.
SHELL_SRCS = $(wildcard src/*.mm)
# The NSObject-based wrappers alone: the device shell + its category slices subclass the
# Apple-private base, so they compile only into the bundle, never the native unit test.
WRAP_SRCS  = $(filter-out $(wildcard src/RgfxMetalDevice*.mm),$(wildcard src/*.mm))

# Objects are kept per-arch (so the x86_64 and native builds never clobber) and
# split core/ (C++) vs shell/ (ObjC++) so each compile rule has a unique target
# directory — GNU make 3.81 mishandles two pattern rules sharing one target dir.
CORE_OBJ_ARCH  = $(CORE_SRCS:src/%.cpp=$(BUILD)/obj/$(ARCH)/core/%.o)
SHELL_OBJ_ARCH = $(SHELL_SRCS:src/%.mm=$(BUILD)/obj/$(ARCH)/shell/%.o)
CORE_OBJ_NAT   = $(CORE_SRCS:src/%.cpp=$(BUILD)/obj/$(NATIVE)/core/%.o)
WRAP_OBJ_NAT   = $(WRAP_SRCS:src/%.mm=$(BUILD)/obj/$(NATIVE)/shell/%.o)

SER_TESTBIN   = $(BUILD)/serializer_tests
AIR_TESTBIN   = $(BUILD)/air_arguments_test
SHELL_TESTBIN = $(BUILD)/shell_tests
INHERIT_TESTBIN = $(BUILD)/inherited_tests
MEMBERS_TESTBIN = $(BUILD)/protocol_members_tests
VIDEO_TESTBIN = $(BUILD)/video_h264_tests
VIDEO_H265_TESTBIN = $(BUILD)/video_h265_tests
VIDEO_DRIVER_TESTBIN = $(BUILD)/video_driver_tests
BUNDLEBIN     = $(BUILD)/reims-graphics.bundle

all: $(CORE_OBJ_ARCH)

# --- compile rules (per arch; C++ and ObjC++) --------------------------------

$(BUILD)/obj/$(ARCH)/core/%.o: src/%.cpp $(HEADERS) $(WIRE)/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(ARCH) $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/$(ARCH)/shell/%.o: src/%.mm $(HEADERS) $(WIRE)/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(ARCH) $(CXXFLAGS) $(OBJCXXFLAGS) $(PROTOCOL_FLAGS) -c $< -o $@

$(BUILD)/obj/$(NATIVE)/core/%.o: src/%.cpp $(HEADERS) $(WIRE)/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(NATIVE) $(CXXFLAGS) -c $< -o $@

$(BUILD)/obj/$(NATIVE)/shell/%.o: src/%.mm $(HEADERS) $(WIRE)/reims_graphics_wire.h Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(NATIVE) $(CXXFLAGS) $(OBJCXXFLAGS) $(PROTOCOL_FLAGS) -c $< -o $@

# The device's files keep the protocol diagnostics quiet (above).
$(patsubst src/%.mm,$(BUILD)/obj/$(ARCH)/shell/%.o,$(wildcard src/RgfxMetalDevice*.mm)): \
	PROTOCOL_FLAGS = $(PROTOCOL_QUIET)

# --- the serialiser check (x86_64, Rosetta) ----------------------------------

$(SER_TESTBIN): $(CORE_OBJ_ARCH) tests/serializer_tests.cpp tests/slab_tests.cpp \
	tests/library_stream_tests.cpp tests/function_constants_tests.cpp $(HEADERS) $(TESTHEADERS) \
	Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(ARCH) $(CXXFLAGS) -I tests $(CORE_OBJ_ARCH) tests/serializer_tests.cpp \
		tests/slab_tests.cpp tests/library_stream_tests.cpp tests/function_constants_tests.cpp -o $@

# --- the Metal-shell wrapper/routing check (native arch) ---------------------
# The NSObject wrappers + serialiser core + the routing test; no device shell.

SHELL_TESTSRCS = tests/shell_tests.mm tests/protocol_tests.mm tests/archive_tests.mm \
	tests/resource_tests.mm tests/queue_tests.mm \
	tests/create_tests.mm tests/known_pipeline_tests.mm tests/icb_tests.mm tests/gl_tests.mm \
	tests/cl_mode_tests.mm

$(SHELL_TESTBIN): $(CORE_OBJ_NAT) $(WRAP_OBJ_NAT) $(SHELL_TESTSRCS) $(HEADERS) $(TESTHEADERS) Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(NATIVE) $(CXXFLAGS) $(OBJCXXFLAGS) $(PROTOCOL_QUIET) -I tests \
		$(CORE_OBJ_NAT) $(WRAP_OBJ_NAT) $(SHELL_TESTSRCS) $(FRAMEWORKS) -framework CoreVideo -o $@

# A Core Image kernel library, compiled from tests/ci_kernel.metal for the shell tests' extern
# functions; built here, never committed. Built for macOS 14 and Metal 3.1: a guest's Metal on
# macOS 26 refuses the macOS 27 SDK's default language version and deployment target.
CI_KERNEL = $(BUILD)/ci_kernel.metallib

$(CI_KERNEL): tests/ci_kernel.metal Makefile
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -c -fcikernel -std=metal3.1 -mmacosx-version-min=14.0 \
		tests/ci_kernel.metal -o $(BUILD)/ci_kernel.air
	xcrun -sdk macosx metallib -cikernel $(BUILD)/ci_kernel.air -o $@

# The stitchable functions of tests/stitchable.metal, for the shell tests' stitched library.
STITCHABLE = $(BUILD)/stitchable.metallib

$(STITCHABLE): tests/stitchable.metal Makefile
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -c -std=metal3.1 -mmacosx-version-min=14.0 tests/stitchable.metal \
		-o $(BUILD)/stitchable.air
	xcrun -sdk macosx metallib $(BUILD)/stitchable.air -o $@

# The stitchable functions of tests/stitch_constants.metal, specialised by constant values before
# the shell tests stitch them.
STITCH_CONSTANTS = $(BUILD)/stitch_constants.metallib

$(STITCH_CONSTANTS): tests/stitch_constants.metal Makefile
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -c -std=metal3.1 -mmacosx-version-min=14.0 tests/stitch_constants.metal \
		-o $(BUILD)/stitch_constants.air
	xcrun -sdk macosx metallib $(BUILD)/stitch_constants.air -o $@

# The post-tessellation vertex functions of tests/patch.metal, for the shell tests' patch answers.
PATCH_LIB = $(BUILD)/patch.metallib

$(PATCH_LIB): tests/patch.metal Makefile
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -c -std=metal3.1 -mmacosx-version-min=14.0 tests/patch.metal \
		-o $(BUILD)/patch.air
	xcrun -sdk macosx metallib $(BUILD)/patch.air -o $@

# --- the AIR argument reader's check (x86_64, Rosetta) ------------------------
# The reader (src/RgfxAirArguments.*) over a .metallib Apple's Metal compiler builds here from
# tests/air_arguments.metal, never committed; its function constants against the library's CNST
# tags (src/RgfxFunctionConstants.*).
AIR_LIB = $(BUILD)/air_arguments.metallib
# The same library built for macOS 26, whose functions Metal's library reflection answers
# (tests/protocol_tests.mm, test_argument_encoders_of_bindings).
AIR_REFLECTED = $(BUILD)/air_arguments_reflected.metallib

$(AIR_REFLECTED): tests/air_arguments.metal Makefile
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=26.0 tests/air_arguments.metal -o $@

$(AIR_LIB): tests/air_arguments.metal Makefile
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -c -std=metal3.1 -mmacosx-version-min=14.0 tests/air_arguments.metal \
		-o $(BUILD)/air_arguments.air
	xcrun -sdk macosx metallib $(BUILD)/air_arguments.air -o $@

# Pipelines' reflection (src/RgfxPipelineReflection.*) over the oracle probe's library
# (tools/oracle/oracle_reflection.metal), built for macOS 26 and Metal 3.0 as build.sh builds it,
# whose pipelines' reflection the x86 Macs' answers were measured over; never committed.
ORACLE_REFLECTION_LIB = $(BUILD)/oracle_reflection.metallib

$(ORACLE_REFLECTION_LIB): tools/oracle/oracle_reflection.metal Makefile
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -c -std=metal3.0 -mmacosx-version-min=26.0 tools/oracle/oracle_reflection.metal \
		-o $(BUILD)/oracle_reflection.air
	xcrun -sdk macosx metallib $(BUILD)/oracle_reflection.air -o $@

# The library of pipeline_reflection_check's c_atomic_texture case, of a language Metal 3.0 has not
# (tools/oracle/oracle_reflection_31.metal); never committed.
ORACLE_REFLECTION_31_LIB = $(BUILD)/oracle_reflection_31.metallib

$(ORACLE_REFLECTION_31_LIB): tools/oracle/oracle_reflection_31.metal Makefile
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -c -std=metal3.1 -mmacosx-version-min=26.0 tools/oracle/oracle_reflection_31.metal \
		-o $(BUILD)/oracle_reflection_31.air
	xcrun -sdk macosx metallib $(BUILD)/oracle_reflection_31.air -o $@

# The same reader over modules built for macOS 10.12, of LLVM 3.5's metadata nodes
# (tests/air_arguments_old.metal).
AIR_OLD_LIB = $(BUILD)/air_arguments_old.metallib

$(AIR_OLD_LIB): tests/air_arguments_old.metal Makefile
	@mkdir -p $(@D)
	xcrun -sdk macosx metal -c -std=macos-metal1.2 -mmacosx-version-min=10.12 \
		tests/air_arguments_old.metal -o $(BUILD)/air_arguments_old.air
	xcrun -sdk macosx metallib $(BUILD)/air_arguments_old.air -o $@

# The repo's committed metallibs, which Apple's compiler built from our own sources: the reader
# finds every function of them (tests/air_arguments_test.cpp).
COMMITTED_METALLIBS = ../../tests/suite/fixtures/shaders:../../crates/reims-vgpu-vulkan/metallibs

AIR_TESTOBJS = $(BUILD)/obj/$(ARCH)/core/RgfxAirArguments.o \
	$(BUILD)/obj/$(ARCH)/core/RgfxFunctionConstants.o

$(AIR_TESTBIN): $(AIR_TESTOBJS) tests/air_arguments_test.cpp $(HEADERS) Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(ARCH) $(CXXFLAGS) $(AIR_TESTOBJS) tests/air_arguments_test.cpp -o $@

# --- the device's base-class check (x86_64, Rosetta) --------------------------
# Loads the bundle and checks, at run time, that every creator of the device's base classes is
# answered by the device or listed as left to the base (tests/inherited_tests.mm).

$(INHERIT_TESTBIN): tests/inherited_tests.mm Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(ARCH) $(CXXFLAGS) $(OBJCXXFLAGS) tests/inherited_tests.mm -framework Foundation \
		-o $@

# --- the protocol members' check (x86_64, Rosetta) ----------------------------
# Loads the bundle and checks every member of every Metal protocol its classes adopt against
# the register tests/protocol_members.txt, reading the members' bodies from src/: a member the
# register does not list, or one listed as carried out whose body does nothing or answers a
# constant, fails (tests/protocol_members_tests.mm).

$(MEMBERS_TESTBIN): tests/protocol_members_tests.mm Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(ARCH) $(CXXFLAGS) $(OBJCXXFLAGS) tests/protocol_members_tests.mm \
		-framework Foundation -o $@

# --- the video driver's H.264 parameter-set check (x86_64, Rosetta) ----------

$(VIDEO_TESTBIN): $(BUILD)/obj/$(ARCH)/core/RgfxVideoH264.o tests/video_h264_tests.cpp \
	tests/video_h264_cases.hpp $(HEADERS) Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(ARCH) $(CXXFLAGS) -I tests $(BUILD)/obj/$(ARCH)/core/RgfxVideoH264.o \
		tests/video_h264_tests.cpp -o $@

# --- the video driver's HEVC parameter-set check (x86_64, Rosetta) ----------

$(VIDEO_H265_TESTBIN): $(BUILD)/obj/$(ARCH)/core/RgfxVideoH265.o tests/video_h265_tests.cpp \
	tests/video_h265_cases.hpp $(HEADERS) Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(ARCH) $(CXXFLAGS) -I tests $(BUILD)/obj/$(ARCH)/core/RgfxVideoH265.o \
		tests/video_h265_tests.cpp -o $@

# --- the video driver's entry points over a stand-in for the host's sessions (x86_64, Rosetta) ----

$(VIDEO_DRIVER_TESTBIN): $(VIDEO_OBJ_ARCH) tests/video_driver_tests.cpp $(HEADERS) Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(ARCH) $(CXXFLAGS) -I tests $(VIDEO_OBJ_ARCH) tests/video_driver_tests.cpp \
		-framework IOSurface -framework CoreFoundation -o $@

test: $(SER_TESTBIN) $(AIR_TESTBIN) $(AIR_LIB) $(AIR_OLD_LIB) $(AIR_REFLECTED) $(ORACLE_REFLECTION_LIB) \
	$(SHELL_TESTBIN) $(CI_KERNEL) $(STITCHABLE) $(STITCH_CONSTANTS) $(PATCH_LIB) $(INHERIT_TESTBIN) \
	$(MEMBERS_TESTBIN) $(VIDEO_TESTBIN) $(VIDEO_H265_TESTBIN) $(VIDEO_DRIVER_TESTBIN) $(BUNDLEBIN)
	arch -$(ARCH) $(SER_TESTBIN)
	RGFX_AIR_ARGUMENTS=$(AIR_LIB) RGFX_AIR_ARGUMENTS_OLD=$(AIR_OLD_LIB) \
		RGFX_ORACLE_REFLECTION=$(ORACLE_REFLECTION_LIB) RGFX_COMMITTED_METALLIBS=$(COMMITTED_METALLIBS) \
		arch -$(ARCH) $(AIR_TESTBIN)
	RGFX_CI_KERNEL=$(CI_KERNEL) RGFX_STITCHABLE=$(STITCHABLE) RGFX_STITCH_CONSTANTS=$(STITCH_CONSTANTS) \
		RGFX_AIR_ARGUMENTS=$(AIR_LIB) \
		RGFX_AIR_REFLECTED=$(AIR_REFLECTED) RGFX_ORACLE_REFLECTION=$(ORACLE_REFLECTION_LIB) \
		RGFX_PATCH=$(PATCH_LIB) $(SHELL_TESTBIN)
	arch -$(ARCH) $(INHERIT_TESTBIN) $(BUNDLEBIN)
	arch -$(ARCH) $(MEMBERS_TESTBIN) $(BUNDLEBIN) tests/protocol_members.txt src
	arch -$(ARCH) $(VIDEO_TESTBIN)
	arch -$(ARCH) $(VIDEO_H265_TESTBIN)
	arch -$(ARCH) $(VIDEO_DRIVER_TESTBIN)
	bash tests/bundle_files_check.sh $(BUNDLEBIN)

# --- the wire round trip against the host's library (native arch, this Mac's GPU) ----------
# The bundle's own ring code (the serialiser, a queue's rings, reader and completion thread, a
# command buffer's commit) against the host's library through its C API, with a stand-in for the
# kext (tests/peer_tests.mm), recording the wire's traffic against tests/peer_wire2.trace. Never in
# `make test`: it runs on the GPU, so build it first (`make peer-test-build`), then run
# `make peer-test` holding the Mac's GPU lock. DIST is the tree `cargo xtask dist` wrote (release):
# its archive is linked as QEMU links it, with its pkg-config file's private libraries.
# RGFX_PEER_RECORD=1 records the trace again.
DIST          ?= ../../target/dist-vulkan
PEER_TESTBIN   = $(BUILD)/peer_tests
PEER_TRACE     = tests/peer_wire2.trace
PEER_LIBS      = $(shell sed -n 's/^Libs.private: //p' $(DIST)/lib/pkgconfig/reims-vgpu.pc)

$(PEER_TESTBIN): $(CORE_OBJ_NAT) $(WRAP_OBJ_NAT) tests/peer_tests.mm $(HEADERS) \
	$(DIST)/lib/libreims_vgpu.a $(DIST)/include/reims_vgpu.h Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(NATIVE) $(CXXFLAGS) $(OBJCXXFLAGS) $(PROTOCOL_QUIET) -I $(DIST)/include \
		$(CORE_OBJ_NAT) $(WRAP_OBJ_NAT) tests/peer_tests.mm $(DIST)/lib/libreims_vgpu.a \
		$(FRAMEWORKS) -framework CoreVideo $(PEER_LIBS) -o $@

peer-test-build: $(PEER_TESTBIN)

peer-test: $(PEER_TESTBIN)
	$(PEER_TESTBIN) $(PEER_TRACE)

# --- the loadable bundle (x86_64, device shell included) ---------------------
# The IOGPUMetalDevice superclass is linked two-level against IOGPU (PRIVFW), so
# dyld auto-loads IOGPU and resolves the device shell's base class when Metal
# dlopens the plugin in the guest. The remaining private Metal SPI (instance
# methods, resolved via the base at runtime) stays -undefined dynamic_lookup.

bundle: $(BUNDLEBIN)

$(BUNDLEBIN): $(CORE_OBJ_ARCH) $(SHELL_OBJ_ARCH) $(VIDEO_OBJ_ARCH) Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(ARCH) $(CXXFLAGS) $(OBJCXXFLAGS) \
		-bundle -undefined dynamic_lookup $(FRAMEWORKS) $(PRIVFW) \
		$(CORE_OBJ_ARCH) $(SHELL_OBJ_ARCH) $(VIDEO_OBJ_ARCH) -o $@
	@echo "$(notdir $@): $$(cat src/*.mm | grep -o 'RGFX_UNSUPPORTED[A-Z_]*(' | wc -l | tr -d ' ')" \
	      "members marked RGFX_UNSUPPORTED (src/RgfxUnsupported.h)"

# --- the installable bundle: the directory Metal loads from /Library/GPUBundles -----
# Contents/Info.plist and Contents/MacOS/reims-graphics, ad-hoc signed: WindowServer is a
# platform process, whose Library Validation refuses an unsigned bundle (dlopen error 71),
# whatever amfi_get_out_of_my_way says; an ad-hoc signature it takes. The tarball is the same
# directory, as the guest install takes it.
#
# Contents/Resources/wire-version-N, empty, names the wire's version the bundle speaks
# (RGFX_VERSION in the wire header): the kext registers its accelerator only for a bundle
# carrying the marker of its own version (guest/kext/src/RgfxBundlePaths.hpp; design.md, "The
# wire's version"). It lies under the signature, as every file of the bundle does.

PACKAGE    = $(BUILD)/package/reims-graphics.bundle
PACKAGETGZ = $(BUILD)/reims-graphics.bundle.tgz
WIRE_H       = ../../crates/reims-vgpu-graphics/include/reims_graphics_wire.h
WIRE_VERSION = $(shell sed -n 's/^\#define RGFX_VERSION \([0-9][0-9]*\)$$/\1/p' $(WIRE_H))

package: $(PACKAGETGZ)

$(PACKAGETGZ): $(BUNDLEBIN) Info.plist $(WIRE_H) Makefile
	@test -n "$(WIRE_VERSION)" || { echo "no RGFX_VERSION in $(WIRE_H)"; exit 1; }
	rm -rf $(PACKAGE)
	@mkdir -p $(PACKAGE)/Contents/MacOS $(PACKAGE)/Contents/Resources
	plutil -lint Info.plist > /dev/null
	cp Info.plist $(PACKAGE)/Contents/Info.plist
	cp $(BUNDLEBIN) $(PACKAGE)/Contents/MacOS/reims-graphics
	: > $(PACKAGE)/Contents/Resources/wire-version-$(WIRE_VERSION)
	codesign --force --sign - $(PACKAGE)
	codesign -v --strict $(PACKAGE)
	COPYFILE_DISABLE=1 tar -C $(dir $(PACKAGE)) -czf $@ reims-graphics.bundle
	@echo "$(notdir $@): $(PACKAGE), ad-hoc signed"

# --- the guest's correctness sweep (tools/sweep; sweep.m says what it checks) ----------------
# One binary for ARCH (x86_64, the guest's; ARCH=arm64 for an Apple-silicon Mac), run in a guest
# or on a Mac, never in `make test`: it needs the system frameworks on a real device.
SWEEP_SRCS = $(wildcard tools/sweep/*.m)
SWEEP_FW   = -framework Foundation -framework CoreGraphics -framework ImageIO -framework Metal \
	-framework MetalPerformanceShaders -framework CoreImage -framework AVFoundation \
	-framework CoreMedia -framework CoreVideo -framework VideoToolbox -framework Accelerate \
	-framework QuartzCore
SWEEPBIN   = $(BUILD)/sweep-$(ARCH)

sweep: $(SWEEPBIN)

$(SWEEPBIN): $(SWEEP_SRCS) $(wildcard tools/sweep/*.h) Makefile
	@mkdir -p $(@D)
	xcrun clang -arch $(ARCH) -isysroot $(SDK) -mmacosx-version-min=14.0 -fobjc-arc -O2 \
		-Wall -Wextra -Werror -Wno-unused-parameter $(SWEEP_SRCS) $(SWEEP_FW) -o $@

# --- the GPU benchmark (tools/bench; bench.m says what each case measures) ---------------------
# One binary for ARCH (x86_64, a guest's or an Intel Mac's; ARCH=arm64 for an Apple-silicon Mac),
# run on a real device, never in `make test`. Its shaders compile from source at run time.
BENCHBIN = $(BUILD)/bench-$(ARCH)

bench: $(BENCHBIN)

$(BENCHBIN): tools/bench/bench.m Makefile
	@mkdir -p $(@D)
	xcrun clang -arch $(ARCH) -isysroot $(SDK) -mmacosx-version-min=14.0 -fobjc-arc -O2 \
		-Wall -Wextra -Werror -Wno-unused-parameter tools/bench/bench.m -framework Foundation \
		-framework Metal -framework MetalPerformanceShaders -framework QuartzCore -framework AppKit -o $@

# --- what resources cost (tools/res; res.m says what each case measures) -----------------------
# One binary for ARCH (x86_64, a guest's or an Intel Mac's; ARCH=arm64 for an Apple-silicon Mac),
# run on a real device, never in `make test`.
RESBIN = $(BUILD)/res-$(ARCH)

res: $(RESBIN)

$(RESBIN): tools/res/res.m Makefile
	@mkdir -p $(@D)
	xcrun clang -arch $(ARCH) -isysroot $(SDK) -mmacosx-version-min=14.0 -fobjc-arc -O2 \
		-Wall -Wextra -Werror -Wno-unused-parameter tools/res/res.m -framework Foundation \
		-framework Metal -framework IOSurface -o $@

# --- the command buffer's fixed cost and latency (tools/submit_probe.m says what each case does) --
# One binary for ARCH (x86_64, a guest's or an Intel Mac's; ARCH=arm64 for an Apple-silicon Mac),
# run on a real device, never in `make test`.
SUBMITBIN = $(BUILD)/submit_probe-$(ARCH)

submit-probe: $(SUBMITBIN)

$(SUBMITBIN): tools/submit_probe.m Makefile
	@mkdir -p $(@D)
	xcrun clang -arch $(ARCH) -isysroot $(SDK) -mmacosx-version-min=14.0 -fobjc-arc -O2 \
		-Wall -Wextra -Werror -Wno-unused-parameter tools/submit_probe.m -framework Foundation \
		-framework Metal -o $@

# --- the uploads into Managed resources (tools/upload_probe.m says what each case measures) -------
# One binary for ARCH (x86_64, a guest's or an Intel Mac's; ARCH=arm64 for an Apple-silicon Mac),
# run on a real device, never in `make test`.
UPLOADBIN = $(BUILD)/upload_probe-$(ARCH)

upload-probe: $(UPLOADBIN)

$(UPLOADBIN): tools/upload_probe.m Makefile
	@mkdir -p $(@D)
	xcrun clang -arch $(ARCH) -isysroot $(SDK) -mmacosx-version-min=14.0 -fobjc-arc -O2 \
		-Wall -Wextra -Werror -Wno-unused-parameter tools/upload_probe.m -framework Foundation \
		-framework Metal -o $@

# --- the shader and pipeline costs (tools/pipelines; pipelines.m says what each case measures) --
# One binary for ARCH (x86_64, a guest's or an Intel Mac's; ARCH=arm64 for an Apple-silicon Mac),
# run on a real device, never in `make test`. Its source compiles in (pipelines_source.h, from
# pipelines.metal) for newLibraryWithSource:, and into PIPELINES_LIBS metallibs, one nonce each,
# for newLibraryWithData:. Their nonces come from PIPELINES_SEED (default: the build's clock), so
# a new build's metallibs are new to every cache. Built for macOS 14 and Metal 3.0, which a
# guest's Metal on macOS 26 takes.
PIPELINESBIN  = $(BUILD)/pipelines-$(ARCH)
PIPELINES_LIBDIR = $(BUILD)/pipelines-metallibs
PIPELINES_LIBS = 8
PIPELINES_SEED ?= $(shell expr $$(date +%s) % 1000000 + 1000000)

pipelines: $(PIPELINESBIN) $(PIPELINES_LIBDIR)/stamp

$(BUILD)/pipelines_source.h: tools/pipelines/pipelines.metal Makefile
	@mkdir -p $(@D)
	cd tools/pipelines && xxd -i -n pipelines_metal pipelines.metal > ../../$@

$(PIPELINESBIN): tools/pipelines/pipelines.m $(BUILD)/pipelines_source.h Makefile
	@mkdir -p $(@D)
	xcrun clang -arch $(ARCH) -isysroot $(SDK) -mmacosx-version-min=14.0 -fobjc-arc -O2 \
		-Wall -Wextra -Werror -Wno-unused-parameter -I $(BUILD) tools/pipelines/pipelines.m \
		-framework Foundation -framework Metal -o $@

$(PIPELINES_LIBDIR)/stamp: tools/pipelines/pipelines.metal Makefile
	@mkdir -p $(@D)
	rm -f $(@D)/pipelines-*.metallib $(@D)/*.air
	for i in $$(seq 0 $$(($(PIPELINES_LIBS) - 1))); do \
		n=$$(( $(PIPELINES_SEED) * 16 + i )); \
		xcrun -sdk macosx metal -c -std=metal3.0 -mmacosx-version-min=14.0 -DRGFX_NONCE=$${n}u \
			tools/pipelines/pipelines.metal -o $(@D)/p$$i.air && \
		xcrun -sdk macosx metallib $(@D)/p$$i.air -o $(@D)/pipelines-$$(printf %02d $$i).metallib || exit 1; \
	done
	rm -f $(@D)/*.air
	echo $(PIPELINES_SEED) > $@

# --- the VM guard and the device's superpages (tools/spguard; spguard.cpp says what each case does) -
# One x86-64 binary, run in a guest, never in `make test`: no Metal.
SPGUARDBIN = $(BUILD)/spguard-x86_64

spguard: $(SPGUARDBIN)

$(SPGUARDBIN): tools/spguard/spguard.cpp src/RgfxSuperpages.cpp src/RgfxSuperpages.hpp Makefile
	@mkdir -p $(@D)
	$(CXX) -arch x86_64 $(CXXFLAGS) -mmacosx-version-min=13.0 tools/spguard/spguard.cpp \
		src/RgfxSuperpages.cpp -o $@

# --- the stitchable reflection's check against Metal's own (tools/reflection_check.mm) ---------
# One binary for ARCH (x86_64, a guest's or an Intel Mac's; ARCH=arm64 for an Apple-silicon Mac),
# run on a real device, never in `make test`: it compares with Metal's own compiler's answers.
# Its library is the AIR reader's (AIR_LIB).
REFLECTBIN = $(BUILD)/reflection_check-$(ARCH)

reflection-check: $(REFLECTBIN) $(AIR_LIB)

$(REFLECTBIN): tools/reflection_check.mm src/RgfxReflection.mm src/RgfxAirArguments.cpp $(HEADERS) Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(ARCH) $(CXXFLAGS) $(OBJCXXFLAGS) -mmacosx-version-min=14.0 -x objective-c++ \
		tools/reflection_check.mm src/RgfxReflection.mm -x c++ src/RgfxAirArguments.cpp \
		-framework Foundation -framework Metal -o $@

# --- pipelines' reflection's check against Metal's own (tools/pipeline_reflection_check.mm) ------
# As reflection-check: one binary for ARCH, run on a real device (an x86 Mac's: the answers the
# bundle's are of), never in `make test`. Its library is ORACLE_REFLECTION_LIB.
PIPEREFLECTBIN = $(BUILD)/pipeline_reflection_check-$(ARCH)

pipeline-reflection-check: $(PIPEREFLECTBIN) $(ORACLE_REFLECTION_LIB) $(ORACLE_REFLECTION_31_LIB)

$(PIPEREFLECTBIN): tools/pipeline_reflection_check.mm tools/oracle/oracle_walk.h src/RgfxPipelineReflection.mm \
	src/RgfxAirArguments.cpp $(HEADERS) Makefile
	@mkdir -p $(@D)
	$(CXX) -arch $(ARCH) $(CXXFLAGS) $(OBJCXXFLAGS) -I tools -mmacosx-version-min=26.0 -x objective-c++ \
		tools/pipeline_reflection_check.mm src/RgfxPipelineReflection.mm -x c++ src/RgfxAirArguments.cpp \
		-framework Foundation -framework Metal -o $@

clean:
	rm -rf $(BUILD)

.PHONY: all test bundle package sweep bench pipelines res submit-probe upload-probe reflection-check \
	pipeline-reflection-check peer-test peer-test-build \
	spguard clean

