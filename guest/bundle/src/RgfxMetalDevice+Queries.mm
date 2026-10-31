/*
 * RgfxMetalDevice (Queries): see the header. Our own ObjC++.
 */

#import "RgfxMetalDevice+Queries.h"

#import "RgfxUnsupported.h"

#include <mach/mach_time.h>

#import "RgfxHeap.h"

#include "RgfxCaps.hpp"
#include "RgfxHeapSizes.hpp"

@implementation RgfxMetalDevice (Queries)

// --- heap placement ------------------------------------------------------------------

- (MTLSizeAndAlign)heapBufferSizeAndAlignWithLength:(NSUInteger)length
                                            options:(MTLResourceOptions)options
{
	const RgfxCaps *caps = [self caps];
	const uint64_t hostAlign = caps != nullptr ? caps->heapBufferPageAlign() : 0;
	if (hostAlign == 0) {
		RGFX_UNSUPPORTED("the host's heap buffer alignment (none in the device info): "
		                 "Metal's 256 is answered");
	}
	RgfxHeapSizeAlign sa = rgfxHeapBufferSizeAndAlign(length, hostAlign);
	return {static_cast<NSUInteger>(sa.size), static_cast<NSUInteger>(sa.align)};
}

- (MTLSizeAndAlign)heapTextureSizeAndAlignWithDescriptor:(MTLTextureDescriptor *)desc
{
	return rgfxHeapTextureSizeAndAlign([self serializer], desc, [self class], _cmd);
}

// No acceleration structures, tensors or counter heaps: Metal's sizes of what a device makes
// none of.

- (MTLSizeAndAlign)heapAccelerationStructureSizeAndAlignWithSize:(NSUInteger)size
{
	RGFX_UNSUPPORTED("ray tracing");
	return {0, 0};
}

- (MTLSizeAndAlign)heapAccelerationStructureSizeAndAlignWithDescriptor:
    (MTLAccelerationStructureDescriptor *)descriptor
{
	RGFX_UNSUPPORTED("ray tracing");
	return {0, 0};
}

- (MTLAccelerationStructureSizes)accelerationStructureSizesWithDescriptor:
    (MTLAccelerationStructureDescriptor *)descriptor
{
	RGFX_UNSUPPORTED("ray tracing");
	return {0, 0, 0};
}

- (MTLSizeAndAlign)tensorSizeAndAlignWithDescriptor:(MTLTensorDescriptor *)descriptor
{
	RGFX_UNSUPPORTED("tensors");
	return {0, 0};
}

- (NSUInteger)sizeOfCounterHeapEntry:(MTL4CounterHeapType)type
{
	RGFX_UNSUPPORTED("Metal 4 counter heaps");
	return 0;
}

// --- timestamps ----------------------------------------------------------------------

- (void)sampleTimestamps:(MTLTimestamp *)cpuTimestamp gpuTimestamp:(MTLTimestamp *)gpuTimestamp
{
	const uint64_t now = mach_absolute_time();
	if (cpuTimestamp != nullptr)
		*cpuTimestamp = now;
	if (gpuTimestamp != nullptr)
		*gpuTimestamp = now;
}

// The GPU timestamps' ticks a second: mach_absolute_time()'s.
- (uint64_t)queryTimestampFrequency
{
	mach_timebase_info_data_t tb = {};
	if (mach_timebase_info(&tb) != KERN_SUCCESS || tb.numer == 0)
		return 1000000000ull;
	return 1000000000ull * tb.denom / tb.numer;
}

@end
