/*
 * RgfxInterop: what the bundle's objects share to answer the members Apple's frameworks send to
 * Metal objects beyond the public protocols (MPS, QuartzCore, SkyLight, CoreImage and CoreDisplay
 * send them to the objects Metal hands out).
 *
 * - Trace ids (`globalTraceObjectID`): Metal answers one on the device, its command queues and
 *   command buffers, every encoder (a parallel encoder's sub-encoders too), heaps and shared
 *   events, and on no other object. Each is an unsigned 64-bit value, stable for the object and
 *   unique in the process, taken in creation order from one counter all of those kinds share
 *   (measured on the M3 Max, macOS 27). RgfxNextTraceObjectId is that counter: each such object
 *   takes its id when it is made. The device answers its base class's, Metal's own.
 * - Unrecognised selectors: every class of the bundle logs a selector it does not answer, once per
 *   class and selector ("RGFX: unrecognized: -[RgfxComputeCommandEncoder sel]"), and then raises
 *   as NSObject does (RGFX_LOG_UNRECOGNIZED), so one boot shows every miss, not only the first
 *   that ends the process. The device does too: a selector neither it nor Metal's base classes
 *   answer raises, as Metal's own device does.
 *
 * Our own ObjC++; no Apple private SPI.
 */

#ifndef RGFX_INTEROP_H_
#define RGFX_INTEROP_H_

#import <Foundation/Foundation.h>
#import <IOSurface/IOSurfaceRef.h>
#import <Metal/Metal.h>

#include <stdint.h>

NS_ASSUME_NONNULL_BEGIN

#ifdef __cplusplus
extern "C" {
#endif

/// The next trace id of the process (see above): never 0, never repeated.
uint64_t RgfxNextTraceObjectId(void);

/// Logs, the first time `cls` is sent `sel` it does not answer, the class and selector.
/// Thread-safe.
void RgfxLogUnrecognized(Class cls, SEL sel);

/// Whether `cls` has been sent `sel` unanswered (for the tests).
BOOL RgfxUnrecognizedWasLogged(Class cls, SEL sel);

#ifdef __cplusplus
}
#endif

/// The members of Metal's buffer descriptor (MTLBufferDescriptor, which MTLDeviceSPI's and
/// MTLHeap's newBufferWithDescriptor: take) the bundle reads. measured: macOS 27 Metal.framework,
/// the class's own getters and their type encodings (`length`, `resourceOptions`, `alignment`,
/// `pinnedGPUAddress`, `parentGPUAddress`, `parentGPUSize`, `pointerTag` `Q16@0:8`; `contents`
/// `^v16@0:8`; `noCopy` `B16@0:8`; `iosurface` `^{__IOSurface=}16@0:8`; `addressRanges` `@16@0:8`;
/// `placementSparsePageSize` `q16@0:8`). A new descriptor answers length 0, alignment 1 and
/// options 0. Its deallocator has a setter and no getter.
@protocol RgfxBufferDescriptorFields <NSObject>
@property(readonly) NSUInteger length;
@property(readonly) MTLResourceOptions resourceOptions;
@property(readonly) NSUInteger alignment;
@property(readonly) uint64_t pinnedGPUAddress;
@property(readonly) uint64_t parentGPUAddress;
@property(readonly) uint64_t parentGPUSize;
@property(readonly) uint64_t pointerTag;
@property(readonly, nullable) void *contents;
@property(readonly) BOOL noCopy;
@property(readonly, nullable) IOSurfaceRef iosurface;
@property(readonly, nullable) id addressRanges;
@property(readonly) NSInteger placementSparsePageSize;
@end

NS_ASSUME_NONNULL_END

/// In a class's @implementation: log each selector the class does not answer, then raise as
/// NSObject does (RgfxLogUnrecognized). Subclasses inherit it, and log their own class.
#define RGFX_LOG_UNRECOGNIZED                                                                      \
	-(void)doesNotRecognizeSelector : (SEL)aSelector                                               \
	{                                                                                              \
		RgfxLogUnrecognized([self class], aSelector);                                              \
		[super doesNotRecognizeSelector:aSelector];                                                \
	}

#endif // RGFX_INTEROP_H_
