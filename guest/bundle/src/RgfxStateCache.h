/*
 * RgfxStateCache: the device's samplers and depth/stencil states, one object for equal
 * descriptors.
 *
 * Metal's answer (measured on the M3 Max under macOS 27):
 *  - newSamplerStateWithDescriptor: and newDepthStencilStateWithDescriptor: answer the object
 *    they made before for a descriptor equal to its.
 *  - What they compare is every property of the descriptor, the label among them, so labels
 *    that differ make objects that differ, each answering its own. An empty label counts as
 *    none (the object answers nil), and -0.0 as 0.0.
 *  - A depth/stencil descriptor's faces are never nil (null_resettable), so a face set to nil
 *    and one set to a new MTLStencilDescriptor are equal.
 *  - A sampler is freed once the app lets it go. A depth/stencil state outlives the app's last
 *    reference: Metal keeps it.
 * The AMD Radeon Pro 5300M and the Intel UHD 630 (macOS 26.7.1) make a new object each call,
 * in 0.2 us. Metal's documentation promises neither, so either is Metal's behaviour, and the
 * M3's is the cheaper one here: a sampler create waits for the host's answer (SAMPLER_STATE),
 * and a depth/stencil create is a kext call.
 *
 * So the cache answers an equal descriptor with the object it made, with no kext call. It
 * holds each object weakly while the app holds it, and the last kKept of each kind it made
 * strongly, whether or not the app still holds them. An app that makes one sampler again and
 * again, and lets each go, gets the same one, with one host create in all. Objects made past
 * those are freed as the app lets them go, as the AMD's and the Intel's are.
 *
 * The cache lets go of the object a create pushes out of the last kKept on a serial queue of its
 * own, in the order pushed out: where the app holds that object no more, its dealloc posts its
 * destroy, a kext call, and the create that pushed it out does not wait for it.
 *
 * A create that made nothing (one the device never got, a sampler the host refused, or one
 * whose create was not answered) answers nil and is not cached: the next equal create tries
 * again, as before.
 *
 * Thread-safe. A create runs outside the cache's lock, so a slow sampler create holds up no
 * other create; where two threads make objects of one descriptor at once, the first cached
 * wins and both answer it.
 *
 * Our own ObjC++; no Apple private SPI.
 */

#ifndef RGFX_STATE_CACHE_H_
#define RGFX_STATE_CACHE_H_

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#ifdef __cplusplus
class RgfxSerializer;
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxStateCache : NSObject

/// How many objects of each kind the cache keeps after the app lets them go: the last made.
@property(class, nonatomic, readonly) NSUInteger keptCount;

/// Returns once the objects every cache pushed out before the call are let go (for tests).
+ (void)waitForReleases;

#ifdef __cplusplus
/// A cache for the objects of `device` (weak; nil in a unit test), made through `serializer`
/// (null makes none).
- (instancetype)initWithDevice:(nullable id<MTLDevice>)device
                    serializer:(nullable RgfxSerializer *)serializer NS_DESIGNATED_INITIALIZER;
#endif

- (instancetype)init NS_UNAVAILABLE;

/// The sampler of shape `descriptor`: the one made for an equal descriptor where the cache
/// holds it, else a new one (RgfxSampler). Nil where none can be made.
- (nullable id<MTLSamplerState>)samplerWithDescriptor:(nullable MTLSamplerDescriptor *)descriptor;

/// The depth/stencil state of shape `descriptor`, answered as samplerWithDescriptor:.
- (nullable id<MTLDepthStencilState>)depthStencilStateWithDescriptor:
    (nullable MTLDepthStencilDescriptor *)descriptor;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_STATE_CACHE_H_
