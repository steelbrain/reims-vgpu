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

/// Makes Metal's own library over `source` (its base device class's `newLibraryWithData:error:`),
/// or nil where Metal refuses it.
typedef id<MTLLibrary> _Nullable (^RgfxShadowMaker)(dispatch_data_t source);

/// The `functionType` Metal answers for a Core Image kernel library's functions (a TYPE 5 entry of
/// its function list), which no SDK enumerator names (measured: macOS 27 and 26.6.1).
static const MTLFunctionType kRgfxFunctionTypeExtern = (MTLFunctionType)4;

@interface RgfxLibrary : RgfxResource <MTLLibrary>

/// Keeps `source`, the bytes the library was made from (retained, not copied), and `maker`,
/// which makes the library's shadow from them the first time Core Image's link route needs it:
/// Metal's own library over the same bytes, whose extern functions are what Metal's link reads.
- (void)rgfxSetSource:(dispatch_data_t)source shadowMaker:(RgfxShadowMaker)maker;

/// Takes `shadow`, Metal's own library over the bytes the library was made from, as its shadow
/// (Core Image's link made both).
- (void)rgfxAdoptShadow:(id<MTLLibrary>)shadow;

/// The shadow, made on first use; nil where the library keeps no source or Metal refused it.
- (nullable id<MTLLibrary>)rgfxShadow;

/// The private MTLLibrary members Core Image calls (measured: macOS 27 and 26.6.1): the
/// shadow's extern function names (empty with no shadow), and the function of `name` whose
/// shadow is the shadow's own extern function of it, of its `functionType` (nil where the
/// shadow holds none).
- (NSArray<NSString *> *)externFunctionNames;
- (nullable id<MTLFunction>)newExternFunctionWithName:(NSString *)name;

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

/// Each of `functions` as Metal's own Core Image link takes it: an RgfxFunction as its shadow
/// (an extern function of its library's shadow, of its entry point, where it has none yet), any
/// other function as it is. nil, with `*error`, where an RgfxFunction has no shadow: its library
/// keeps no source, Metal refused it, or it holds no function of that name.
NSArray<id<MTLFunction>> *_Nullable RgfxShadowFunctions(NSArray<id<MTLFunction>> *functions,
                                                        NSError *__autoreleasing _Nullable *_Nullable error);

/// Core Image's link of `functions` (measured: macOS 27 and 26.6.1, Metal's base device class
/// links them on the CPU, reading Metal-internal library data no RgfxFunction holds): `link`,
/// Metal's own, given each function's shadow, and the library it answers made one of ours by
/// `makeLibrary` over the bytes Metal answers for it (`libraryDataContents`, a .metallib of the
/// one kernel), which keeps Metal's library as its shadow. A library `link` answers that is
/// already ours (Metal's link reached ours again, which made it) is answered as it is.
typedef id<MTLLibrary> _Nullable (^RgfxLinkBlock)(NSArray<id<MTLFunction>> *shadows,
                                                  NSError *__autoreleasing _Nullable *_Nullable error);
typedef id<MTLLibrary> _Nullable (^RgfxLibraryMaker)(dispatch_data_t data,
                                                     NSError *__autoreleasing _Nullable *_Nullable error);
id<MTLLibrary> _Nullable RgfxLinkImageFilterFunctions(NSArray<id<MTLFunction>> *functions,
                                                      RgfxLinkBlock link, RgfxLibraryMaker makeLibrary,
                                                      NSError *__autoreleasing _Nullable *_Nullable error);

NS_ASSUME_NONNULL_END

#endif // RGFX_LIBRARY_H_
