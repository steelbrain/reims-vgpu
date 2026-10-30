/*
 * RgfxResource: the base of every guest Metal object wrapper the reims-graphics
 * bundle vends (buffers, textures, samplers, heaps, queues, pipelines, events…).
 *
 * THE WRAPPER CONVENTION (established here; the resource/pipeline/command-buffer/
 * sync slices replicate it):
 *  - A wrapper is an `NSObject` subclass that conforms to the matching Metal
 *    *protocol* (`RgfxBuffer : RgfxResource <MTLBuffer>`, and so on). It is NOT a
 *    subclass of any Apple-private class, so — unlike the device shell — it
 *    instantiates and is unit-tested on the Mac in user space.
 *  - Every wrapper carries the guest-picked `objId` (RgfxObjId) that names its
 *    host handle, a weak back-pointer to the owning `id<MTLDevice>` (our device
 *    shell at runtime; nil in a unit test), and a raw pointer to that device's
 *    `RgfxSerializer`, which the wrapper routes its mutating Metal calls through.
 *    The serializer and the device own their lifetimes; the wrapper only borrows
 *    them, so the pointers are not retained/freed here.
 *  - A wrapper's getters answer from local state (length, gpuAddress, options,
 *    label…). A wrapper's mutating calls (create, modify, encode) serialise a wire
 *    message through the serializer. A create is done by the device category that
 *    makes the wrapper (it calls the serializer, then hands the RgfxCreated to the
 *    wrapper's initialiser), so a wrapper is never half-created.
 *
 * RgfxResource itself holds the three common fields and the MTLResource `label`.
 * Concrete wrappers add their protocol's state and methods. This is our own
 * ObjC++; it depends only on the serialiser core and the committed wire header,
 * never on Apple's private SPI.
 */

#ifndef RGFX_RESOURCE_H_
#define RGFX_RESOURCE_H_

#import <Foundation/Foundation.h>

#include "RgfxObjectModel.hpp" // RgfxObjId

#ifdef __cplusplus
class RgfxSerializer;
#endif

@protocol MTLDevice;

NS_ASSUME_NONNULL_BEGIN

@interface RgfxResource : NSObject {
@protected
	// The guest-picked id naming this object's host handle (RgfxObjId == uint64_t).
	uint64_t _objId;
	// Borrowed: the owning device's serialiser, the route for mutating calls.
	// Null in a unit test that drives a wrapper without a device.
	RgfxSerializer *_serializer;
}

/// The object's guest-picked id (0 names no object).
@property(nonatomic, readonly) uint64_t rgfxObjId;

/// The owning device (our shell at runtime). Weak: the device owns its wrappers'
/// lifetimes, not the other way round.
@property(nonatomic, readonly, weak, nullable) id<MTLDevice> device;

/// MTLResource's debug label, kept locally (diagnostics only, not on the wire).
@property(nonatomic, copy, nullable) NSString *label;

/// Designated initialiser: binds the wrapper to its host handle `objId`, owning
/// `device` (may be nil in a test), and that device's `serializer` (may be null).
- (instancetype)initWithObjId:(uint64_t)objId
                       device:(nullable id<MTLDevice>)device
                   serializer:(nullable RgfxSerializer *)serializer NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

/// The borrowed serialiser (null when the wrapper was made without one).
- (nullable RgfxSerializer *)serializer;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_RESOURCE_H_
