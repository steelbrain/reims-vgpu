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
	uint32_t _nextRingId;        // the next queue ring's doorbell id (0 is the control ring)
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
	_serializer = nullptr;
	_region = nullptr;
	_replyReader = nullptr;
	_nextRingId = 1;
	// Wire handshake stub: when the device-region/ring transport lands, this maps
	// the device region off `port`, reads the RgfxDeviceInfo caps blob into
	// `_caps` (RgfxCaps::load), stands up the control ring + RgfxSerializer and the
	// completion-ring RgfxReplyReader, and builds the RgfxRegion over the region's
	// base and size. Until then `_caps` stays invalid and the getters below answer
	// from their documented defaults, and `_serializer`/`_region`/`_replyReader` are
	// null (the resource and command-queue slices fall back while they are).
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
	uint32_t ringId = _nextRingId++;
	auto writer = std::make_unique<RgfxRingWriter>(_region->base() + regionOffset, dataSize, ringId,
	                                               &rgfxDeviceDoorbell, (__bridge void *)self);
	RgfxRingWriter *raw = writer.get();
	_queueRings.push_back(std::move(writer));
	return raw;
}

- (void)rgfxRingDoorbell:(uint32_t)ringId
{
	// Stub until BAR 0's registers are mapped (the wire handshake): then this writes
	// `ringId` to RGFX_REG_DOORBELL to wake the host's consumer of that ring.
	(void)ringId;
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
