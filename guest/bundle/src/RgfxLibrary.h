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
/// an RGFX_CMD_LIBRARY_CREATE (the bytes forwarded unchanged, with their content
/// key where `key`, 32 bytes, is not null; the serialiser picks the objId).
/// `device` is the owning shell (nil in a unit test). Returns nil if `serializer`
/// is null (a library must have a route), the create did not post (the message did
/// not fit the ring: `*status` kRgfxNoAnswer, and the caller may retry), or the
/// device refused it (`*status` its answer).
+ (nullable instancetype)libraryWithData:(const void *)bytes
                                  length:(NSUInteger)length
                                     key:(nullable const uint8_t *)key
                                  device:(nullable id<MTLDevice>)device
                              serializer:(nullable RgfxSerializer *)serializer
                                  status:(uint32_t *)status;
/// The library the device holds under the content key of the `length` bytes at
/// `bytes` (`key`, 32 bytes), found by an RGFX_CMD_LIBRARY_LOOKUP, with no bytes sent;
/// its function table is read from the bytes. Returns nil if `serializer` is null or
/// the device holds no such library.
+ (nullable instancetype)libraryFoundWithKey:(const uint8_t *)key
                                       bytes:(const void *)bytes
                                      length:(NSUInteger)length
                                      device:(nullable id<MTLDevice>)device
                                  serializer:(nullable RgfxSerializer *)serializer;

/// Creates a library from `length` bytes of .metallib/AIR at `bytes` by streaming
/// them through the process's staging window in `region` (two blocks, used in
/// turn, so the copy of one chunk overlaps the host's copy-out of the one before),
/// one RGFX_CMD_LIBRARY_FROM_REGION per chunk, for a library larger than one ring
/// message. Waits for the host to have copied a block out before writing it again.
/// The chunks carry the content key where `key` (32 bytes) is not null. Returns nil
/// if `serializer` or `region` is null, the region has no room for the window, the
/// host did not take a chunk (`*status` kRgfxNoAnswer), or the device refused the
/// library (`*status` its answer to the last chunk).
+ (nullable instancetype)libraryStreamingBytes:(const void *)bytes
                                       length:(NSUInteger)length
                                          key:(nullable const uint8_t *)key
                                       region:(nullable RgfxRegion *)region
                                       device:(nullable id<MTLDevice>)device
                                   serializer:(nullable RgfxSerializer *)serializer
                                       status:(uint32_t *)status;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_LIBRARY_H_
