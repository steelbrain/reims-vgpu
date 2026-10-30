/*
 * RgfxLibrary: the reims-graphics bundle's MTLLibrary wrapper. A library is created
 * from .metallib or AIR bytes, which the serialiser forwards to the host unchanged
 * (RGFX_CMD_LIBRARY_CREATE); the guest picks the objId. The library vends functions
 * by name (`-newFunctionWithName:`): each is an RgfxFunction holding this library's
 * objId and the entry-point name, resolved against the AIR host-side when a pipeline
 * create names it.
 *
 * Our own ObjC++: it conforms to MTLLibrary but subclasses only NSObject (via
 * RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_LIBRARY_H_
#define RGFX_LIBRARY_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#ifdef __cplusplus
class RgfxRegion;
class RgfxSerializer;
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxLibrary : RgfxResource <MTLLibrary>

/// Creates a library from `length` bytes of .metallib/AIR at `bytes`, serialising
/// an RGFX_CMD_LIBRARY_CREATE (the bytes forwarded unchanged; the serialiser picks
/// the objId). `device` is the owning shell (nil in a unit test). Returns nil if
/// `serializer` is null (a library must have a route) or the create did not post
/// (the message did not fit the ring).
+ (nullable instancetype)libraryWithData:(const void *)bytes
                                  length:(NSUInteger)length
                                  device:(nullable id<MTLDevice>)device
                              serializer:(nullable RgfxSerializer *)serializer;

/// Creates a library from `length` bytes of .metallib/AIR at `bytes` by copying
/// them into a block of `region` and serialising an RGFX_CMD_LIBRARY_FROM_REGION
/// naming it, for a library larger than one ring message. The library keeps the
/// block for its life. Returns nil if `serializer` or `region` is null, the region
/// has no room, or the create did not post (the ring had no room).
+ (nullable instancetype)libraryWithRegionCopyOf:(const void *)bytes
                                         length:(NSUInteger)length
                                         region:(nullable RgfxRegion *)region
                                         device:(nullable id<MTLDevice>)device
                                     serializer:(nullable RgfxSerializer *)serializer;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_LIBRARY_H_
