/*
 * RgfxMetalDevice: see RgfxMetalDevice.h for the class's role and the category
 * split. This file is slice 0: init, identity, and the limit/feature getters,
 * answered from RgfxCaps. The resource/pipeline/command-buffer/sync selectors live
 * in the category files the header names.
 *
 * Our own ObjC++; the only Apple-private names are the forward-declarations in
 * RgfxMetalPrivate.h.
 */

#import "RgfxMetalDevice.h"

// RGFX DIAG (Wall-2 trace; remove before merge): the guest's logd is broken, so
// write diagnostics to a file any process (platform or not) can read.
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>

#include <IOKit/IOKitLib.h>

#include <cstring>
#include <memory>
#include <vector>

namespace {
// RGFX DIAG (remove before merge): append one line to /tmp/rgfx-diag.log.
void rgfxDiag(const char *fmt, ...)
{
	char line[512];
	va_list ap;
	va_start(ap, fmt);
	int n = vsnprintf(line, sizeof(line), fmt, ap);
	va_end(ap);
	if (n <= 0)
		return;
	if (static_cast<size_t>(n) > sizeof(line))
		n = sizeof(line);
	int fd = open("/tmp/rgfx-diag.log", O_WRONLY | O_CREAT | O_APPEND, 0666);
	if (fd < 0)
		return;
	(void)write(fd, line, static_cast<size_t>(n));
	close(fd);
}
} // namespace

#include "RgfxRegion.hpp"
#include "RgfxReplyReader.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"

// The accelerator's user-client ABI, mirroring guest/kext/src/RgfxAccelerator.hpp (that
// header is kernel-only and not included here, so the values are duplicated and kept in sync
// by hand): the only user-client type, the BAR-2 region memory id, and the doorbell selector.
enum {
	kRgfxAccelUserClientType = 0,
	kRgfxAccelMemoryRegion = 0,
	kRgfxAccelMethodDoorbell = 0,
};

// The device's doorbell trampoline the queue submission rings ring: it forwards the
// ring id to the device, which (once BAR 0's registers are mapped) writes it to the
// DOORBELL register. `context` is the device, valid for each ring writer's life (the
// device owns them).
@interface RgfxMetalDevice ()
- (void)rgfxRingDoorbell:(uint32_t)ringId;
@end

namespace {
void rgfxDeviceDoorbell(void *context, uint32_t ring_id)
{
	[(__bridge RgfxMetalDevice *)context rgfxRingDoorbell:ring_id];
}
} // namespace

// MTLGPUFamily raw values, compared numerically so the build does not depend on
// which Apple* families this SDK happens to name. The -1009 pseudo-family is the
// device-info bit the host reports among the families (AGENTS.md).
// measured: macOS 27 Metal.framework.
namespace {
constexpr long kGpuFamilyApple5 = 1005;
constexpr long kGpuFamilyApple6 = 1006;
constexpr long kGpuFamilyApple7 = 1007;
constexpr long kGpuFamilyApple8 = 1008;
constexpr long kGpuFamilyApple9 = 1009;
constexpr long kGpuFamilyApple10 = 1010;
constexpr long kGpuFamilyMinus1009 = -1009;
} // namespace

@implementation RgfxMetalDevice {
	RgfxCaps _caps;              // host device caps; invalid until the region is read
	RgfxIdSpace _ids;            // this device's id/address/resource-id lines
	RgfxSerializer *_serializer; // the create route; null until the ring transport lands
	mach_port_t _acceleratorPort;
	RgfxRegion *_region;         // BAR-region allocator; null until the handshake maps it
	RgfxReplyReader *_replyReader; // control-completion-ring reader; null until the handshake
	std::vector<std::unique_ptr<RgfxRingWriter>> _queueRings; // owned queue submit-ring writers
	io_connect_t _conn;          // our accelerator user client (BAR-2 region map + doorbell)
	std::unique_ptr<RgfxRingWriter> _controlRing; // control submission ring (ring id 0)
	mach_vm_address_t _mapBase;  // mapped device-region base (0 until the handshake maps it)
	mach_vm_size_t _mapSize;     // mapped device-region size
}

// RGFX DIAG (Wall-2 trace; remove before merge): fires when the bundle is loaded
// and this class is realized in the ObjC runtime. If this logs but no init below
// does, Metal loaded the bundle but did not instantiate us by initWithAcceleratorPort
// (wrong entry point or a pre-instantiation gate); if this never logs, the class is
// not being realized at all.
+ (void)load
{
	rgfxDiag("RGFX: RgfxMetalDevice +load (class realized in runtime)\n");
}

// RGFX DIAG (remove before merge): catch a plain -init in case Metal instantiates us
// that way rather than via initWithAcceleratorPort:.
- (instancetype)init
{
	rgfxDiag("RGFX: -init called (not initWithAcceleratorPort:)\n");
	NSLog(@"RGFX: -init called");
	return [super init];
}

- (instancetype)initWithAcceleratorPort:(mach_port_t)port
{
	// RGFX DIAG (Wall-2 trace; remove before merge): confirm Metal enters our init
	// and report whether the IOGPUMetalDevice base accepts or declines. To a file
	// (the guest's logd is broken) plus the probe's stderr (NSLog).
	rgfxDiag("RGFX: initWithAcceleratorPort entered, port=0x%x\n", (unsigned)port);
	NSLog(@"RGFX: initWithAcceleratorPort entered, port=0x%x", (unsigned)port);
	self = [super initWithAcceleratorPort:port];
	rgfxDiag("RGFX: [super initWithAcceleratorPort] -> %s\n", self ? "non-nil" : "nil");
	NSLog(@"RGFX: [super initWithAcceleratorPort] -> %@", self ? @"non-nil" : @"nil");
	if (self == nil) {
		return nil; // Apple's base declined (or the port is not ours)
	}
	_acceleratorPort = port;
	_conn = IO_OBJECT_NULL;
	_serializer = nullptr;
	_region = nullptr;
	_replyReader = nullptr;
	_mapBase = 0;
	_mapSize = 0;

	// Wire transport handshake: open our own user client on the accelerator and map the
	// device region (BAR 2) into this task. Read the caps blob the host wrote, build the
	// region allocator, and stand up the control submission ring (RgfxSerializer) + control
	// completion ring (RgfxReplyReader). With these non-null the resource / command-queue /
	// command-buffer categories route over our wire instead of falling back to the
	// IOGPUMetalDevice base (Apple's IOGPU kernel ABI). Any failure leaves the transport
	// stubbed (the getters answer from defaults) rather than crashing.
	kern_return_t kr = IOServiceOpen(port, mach_task_self(), kRgfxAccelUserClientType, &_conn);
	if (kr != KERN_SUCCESS) {
		rgfxDiag("RGFX: IOServiceOpen(type=%u) failed kr=0x%x; transport stubbed\n",
		         kRgfxAccelUserClientType, kr);
		return self;
	}
	mach_vm_address_t addr = 0;
	mach_vm_size_t mapSize = 0;
	kr = IOConnectMapMemory(_conn, kRgfxAccelMemoryRegion, mach_task_self(), &addr, &mapSize,
	                        kIOMapAnywhere);
	if (kr != KERN_SUCCESS || addr == 0 || mapSize < RGFX_ALLOCATIONS_OFFSET) {
		rgfxDiag("RGFX: IOConnectMapMemory failed kr=0x%x size=%llu; transport stubbed\n", kr,
		         (unsigned long long)mapSize);
		IOServiceClose(_conn);
		_conn = IO_OBJECT_NULL;
		return self;
	}
	_mapBase = addr;
	_mapSize = mapSize;
	volatile uint8_t *base = reinterpret_cast<volatile uint8_t *>(addr);

	// The caps blob the host wrote at RGFX_DEVICE_INFO_OFFSET (read once at init).
	RgfxDeviceInfo info;
	memcpy(&info, const_cast<const uint8_t *>(base) + RGFX_DEVICE_INFO_OFFSET, sizeof(info));
	if (!_caps.load(info))
		rgfxDiag("RGFX: caps blob invalid (version/zero); using defaults\n");

	_region = new RgfxRegion(base, mapSize);
	_controlRing = std::make_unique<RgfxRingWriter>(base + RGFX_CONTROL_SUBMIT_OFFSET,
	                                                RGFX_CONTROL_RING_DATA_SIZE, RGFX_CONTROL_RING_ID,
	                                                &rgfxDeviceDoorbell, (__bridge void *)self);
	_serializer = new RgfxSerializer(*_controlRing, _ids);
	_replyReader = new RgfxReplyReader(base + RGFX_CONTROL_COMPLETE_OFFSET,
	                                   RGFX_CONTROL_RING_DATA_SIZE);
	rgfxDiag("RGFX: transport up: region=%llu bytes, caps=%s\n", (unsigned long long)mapSize,
	         _caps.valid() ? "valid" : "default");
	return self;
}

// RGFX DIAG (remove before merge): -[_MTLDevice initLimits] and other Metal bring-up code send
// MTLDeviceSPI selectors straight to us without -respondsToSelector:; any that neither our shell
// nor the IOGPUMetalDevice base implements hit -doesNotRecognizeSelector: and throw an
// unhandled ObjC exception (crash). To enumerate the FULL set of missing selectors in one run
// (instead of one crash per selector), forward every unimplemented selector here, log it, and
// answer with a zero scalar so bring-up keeps going.
- (NSMethodSignature *)methodSignatureForSelector:(SEL)aSelector
{
	NSMethodSignature *sig = [super methodSignatureForSelector:aSelector];
	if (sig)
		return sig;
	rgfxDiag("RGFX: unimplemented selector (forwarded): %s\n", sel_getName(aSelector));
	return [NSMethodSignature signatureWithObjCTypes:"Q@:"]; // assume an 8-byte scalar return
}

- (void)forwardInvocation:(NSInvocation *)anInvocation
{
	unsigned long long zero = 0;
	[anInvocation setReturnValue:&zero];
}

// --- accessors the category slices route through -----------------------------

- (RgfxSerializer *)serializer
{
	return _serializer;
}

- (RgfxIdSpace *)ids
{
	return &_ids;
}

- (const RgfxCaps *)caps
{
	return &_caps;
}

- (RgfxRegion *)region
{
	return _region;
}

- (RgfxReplyReader *)replyReader
{
	return _replyReader;
}

- (RgfxRingWriter *)addQueueSubmitRingAtOffset:(uint64_t)regionOffset dataSize:(uint32_t)dataSize
{
	if (_region == nullptr) {
		return nullptr;
	}
	// The host keys a per-queue ring's doorbell by its submission ring's region offset
	// (device.rs `ring_doorbell` -> `by_ring`, keyed on `submit_offset`), the control
	// ring's id being its offset 0. So the doorbell id is the region offset itself, not a
	// serial counter, or the host finds no ring for the wake and the queue never runs.
	uint32_t ringId = static_cast<uint32_t>(regionOffset);
	auto writer = std::make_unique<RgfxRingWriter>(_region->base() + regionOffset, dataSize, ringId,
	                                               &rgfxDeviceDoorbell, (__bridge void *)self);
	RgfxRingWriter *raw = writer.get();
	_queueRings.push_back(std::move(writer));
	return raw;
}

- (void)rgfxRingDoorbell:(uint32_t)ringId
{
	// Ring the ring's doorbell through our accelerator user client: the kext writes `ringId`
	// to RGFX_REG_DOORBELL in BAR 0, waking the host's consumer of that ring.
	if (_conn == IO_OBJECT_NULL)
		return;
	uint64_t scalar = ringId;
	IOConnectCallScalarMethod(_conn, kRgfxAccelMethodDoorbell, &scalar, 1, nullptr, nullptr);
}

// --- identity ----------------------------------------------------------------

- (NSString *)name
{
	if (_caps.valid() && _caps.name()[0] != '\0') {
		return [NSString stringWithUTF8String:_caps.name()];
	}
	return @"reims-graphics";
}

- (uint64_t)registryID
{
	return _caps.valid() ? _caps.registryId() : 0;
}

- (MTLSize)maxThreadsPerThreadgroup
{
	if (_caps.valid()) {
		return MTLSizeMake(_caps.maxThreadsPerThreadgroup(0), _caps.maxThreadsPerThreadgroup(1),
		                   _caps.maxThreadsPerThreadgroup(2));
	}
	return MTLSizeMake(1024, 1024, 1024);
}

- (BOOL)hasUnifiedMemory
{
	return YES; // the guest's GPU memory is the host's; always unified
}

- (BOOL)isLowPower
{
	return NO;
}

- (BOOL)isHeadless
{
	return NO;
}

- (BOOL)isRemovable
{
	return NO;
}

- (MTLDeviceLocation)location
{
	return MTLDeviceLocationBuiltIn;
}

- (NSUInteger)locationNumber
{
	return 0;
}

- (uint64_t)peerGroupID
{
	return 0;
}

- (uint32_t)peerIndex
{
	return 0;
}

- (uint32_t)peerCount
{
	return 0;
}

- (NSUInteger)currentAllocatedSize
{
	return 0; // no allocation accounting yet
}

- (uint64_t)recommendedMaxWorkingSetSize
{
	// Placeholder until the host device-info carries it; a conservative 1 GiB so
	// clients that size work against it do not over-commit.
	return 1ull << 30;
}

// --- limits (RgfxCaps) -------------------------------------------------------

- (NSUInteger)maxBufferLength
{
	// Not yet a field in the caps blob; a safe large default until it is carried.
	return 1ull << 32;
}

- (NSUInteger)maxThreadgroupMemoryLength
{
	return _caps.valid() ? _caps.limit(kRgfxLimitMaxThreadgroupMemoryLength) : 32768;
}

- (NSUInteger)maxArgumentBufferSamplerCount
{
	// Not yet in the caps blob; Metal's common floor.
	return 16;
}

- (MTLArgumentBuffersTier)argumentBuffersSupport
{
	// Not yet in the caps blob; Apple-family GPUs are Tier 2.
	return MTLArgumentBuffersTier2;
}

// --- feature flags (RgfxCaps) ------------------------------------------------

- (BOOL)areProgrammableSamplePositionsSupported
{
	return _caps.valid() && _caps.flag(kRgfxFlagSupportsProgrammableSamplePositions);
}

- (BOOL)areRasterOrderGroupsSupported
{
	return _caps.valid() && _caps.flag(kRgfxFlagAreRasterOrderGroupsSupported);
}

- (BOOL)isDepth24Stencil8PixelFormatSupported
{
	return _caps.valid() && _caps.flag(kRgfxFlagIsDepth24Stencil8PixelFormatSupported);
}

- (BOOL)supportsFamily:(MTLGPUFamily)gpuFamily
{
	if (!_caps.valid()) {
		return NO;
	}
	switch (static_cast<long>(gpuFamily)) {
	case kGpuFamilyApple5:
		return _caps.supportsFamily(kRgfxFamilyApple5);
	case kGpuFamilyApple6:
		return _caps.supportsFamily(kRgfxFamilyApple6);
	case kGpuFamilyApple7:
		return _caps.supportsFamily(kRgfxFamilyApple7);
	case kGpuFamilyApple8:
		return _caps.supportsFamily(kRgfxFamilyApple8);
	case kGpuFamilyApple9:
		return _caps.supportsFamily(kRgfxFamilyApple9);
	case kGpuFamilyApple10:
		return _caps.supportsFamily(kRgfxFamilyApple10);
	case kGpuFamilyMinus1009:
		return _caps.supportsFamily(kRgfxFamilyMinus1009);
	default:
		return NO;
	}
}

- (BOOL)supportsFeatureSet:(MTLFeatureSet)featureSet
{
	// Feature sets are the legacy identity; families (above) are the live path and
	// what the host caps blob carries. Report none.
	(void)featureSet;
	return NO;
}

- (BOOL)supportsTextureSampleCount:(NSUInteger)sampleCount
{
	return _caps.valid() && _caps.supportsSampleCount(static_cast<uint32_t>(sampleCount));
}

// --- MTLDeviceSPI (overridden; forward-declared in RgfxMetalPrivate.h) --------

- (NSUInteger)maxVertexSamplers
{
	return _caps.valid() ? _caps.limit(kRgfxLimitMaxVertexSamplers) : 16;
}

- (NSUInteger)maxFragmentTextures
{
	return _caps.valid() ? _caps.limit(kRgfxLimitMaxFragmentTextures) : 128;
}

- (NSUInteger)maxComputeTextures
{
	return _caps.valid() ? _caps.limit(kRgfxLimitMaxComputeTextures) : 128;
}

- (NSUInteger)maxVertexAmplificationCount
{
	return _caps.valid() ? _caps.maxVertexAmplificationCount() : 1;
}

- (BOOL)supportsPrimitiveType:(NSUInteger)primitiveType
{
	return _caps.valid() && _caps.supportsPrimitiveType(static_cast<uint32_t>(primitiveType));
}

@end
