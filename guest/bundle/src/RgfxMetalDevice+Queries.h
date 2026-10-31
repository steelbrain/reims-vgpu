/*
 * RgfxMetalDevice (Queries): the MTLDevice queries the device answers itself, or with one
 * question to the host: the heap placement sizes (RgfxHeapSizes.hpp), the timestamps and the
 * sizes of what the device does not make.
 *
 * - heapBufferSizeAndAlignWithLength:options: answers the length at the host's alignment, as
 *   Metal does; heapTextureSizeAndAlignWithDescriptor: answers the host device's size and
 *   alignment (HEAP_TEXTURE_SIZE_AND_ALIGN), and a bound of Metal's where the host gave none
 *   (rgfxHeapTextureSizeAndAlign, RgfxHeap.h).
 * - sampleTimestamps:gpuTimestamp: answers the CPU's mach_absolute_time() for both: the device
 *   makes no counter sample buffer or counter heap, so nothing else stamps the GPU's domain,
 *   and queryTimestampFrequency answers that domain's ticks a second.
 * - The sizes of acceleration structures, tensors and counter heap entries, which the device
 *   makes none of, are 0, logged (RgfxUnsupported.h).
 *
 * Like the rest of the device it is compiled into the bundle but not linked into the unit
 * tests (RgfxMetalDevice.h). Our own ObjC++.
 */

#ifndef RGFX_METAL_DEVICE_QUERIES_H_
#define RGFX_METAL_DEVICE_QUERIES_H_

#import "RgfxMetalDevice.h"

NS_ASSUME_NONNULL_BEGIN

@interface RgfxMetalDevice (Queries)

- (MTLSizeAndAlign)heapBufferSizeAndAlignWithLength:(NSUInteger)length
                                            options:(MTLResourceOptions)options;
- (MTLSizeAndAlign)heapTextureSizeAndAlignWithDescriptor:(MTLTextureDescriptor *)desc;
- (void)sampleTimestamps:(MTLTimestamp *)cpuTimestamp gpuTimestamp:(MTLTimestamp *)gpuTimestamp;
- (uint64_t)queryTimestampFrequency;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_METAL_DEVICE_QUERIES_H_
