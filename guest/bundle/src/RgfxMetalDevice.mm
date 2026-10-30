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

#include "RgfxSerializer.hpp"

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
}

- (instancetype)initWithAcceleratorPort:(mach_port_t)port
{
	self = [super initWithAcceleratorPort:port];
	if (self == nil) {
		return nil; // Apple's base declined (or the port is not ours)
	}
	_acceleratorPort = port;
	_serializer = nullptr;
	// Wire handshake stub: when the device-region/ring transport lands, this maps
	// the device region off `port`, reads the RgfxDeviceInfo caps blob into
	// `_caps` (RgfxCaps::load), and stands up the control ring + RgfxSerializer
	// over it. Until then `_caps` stays invalid and the getters below answer from
	// their documented defaults, and `_serializer` is null (the resource slice's
	// creates return nil until it exists).
	return self;
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
