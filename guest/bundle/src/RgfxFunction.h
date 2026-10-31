/*
 * RgfxFunction: the reims-graphics bundle's MTLFunction wrapper. A function is a
 * pure guest-side reference — an entry point in a library, optionally specialised
 * by function constants — that a pipeline create names by (library objId +
 * entry-point name + constants). There is no FUNCTION_CREATE wire message and no
 * host handle of its own, so a function carries the sentinel objId RGFX_OBJ_NONE
 * (RgfxObjectModel.hpp: "0 names no object"); its identity is the library objId
 * and name it holds. RgfxLibrary's `-newFunctionWithName:` vends it.
 *
 * Our own ObjC++: it conforms to MTLFunction but subclasses only NSObject (via
 * RgfxResource), so it instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_FUNCTION_H_
#define RGFX_FUNCTION_H_

#import <Metal/Metal.h>

#import "RgfxResource.h"

#include "reims_graphics_wire.h" // RgfxFunctionConstant

#ifdef __cplusplus
class RgfxSerializer;
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxFunction : RgfxResource <MTLFunction>

/// A function reference naming `name` in `library`, specialised by `count` function
/// constants at `constants` (may be null when `count` is 0; the constants are copied).
/// The function keeps `library` alive, so the host still holds it when a pipeline is
/// made from the function, as Metal's functions outlive their library. `device`/
/// `serializer` are the owning library's (both may be nil in a unit test). Returns nil
/// if `name` is nil.
+ (nullable instancetype)functionWithLibrary:(nullable RgfxResource *)library
                                        name:(NSString *)name
                                   constants:(nullable const RgfxFunctionConstant *)constants
                                       count:(uint32_t)count
                                      device:(nullable id<MTLDevice>)device
                                  serializer:(nullable RgfxSerializer *)serializer;

/// Designated initialiser. Holds `library` (and its objId), the entry-point name
/// (copied) and a copy of the `count` function constants at `constants`.
- (nullable instancetype)initWithLibrary:(nullable RgfxResource *)library
                                    name:(NSString *)name
                               constants:(nullable const RgfxFunctionConstant *)constants
                                   count:(uint32_t)count
                                  device:(nullable id<MTLDevice>)device
                              serializer:(nullable RgfxSerializer *)serializer
    NS_DESIGNATED_INITIALIZER;

- (instancetype)initWithObjId:(uint64_t)objId
                       device:(nullable id<MTLDevice>)device
                   serializer:(nullable RgfxSerializer *)serializer NS_UNAVAILABLE;

/// The library this function's entry point lives in (the objId a pipeline create
/// names it by). RGFX_OBJ_NONE if the function was made without one.
@property(nonatomic, readonly) uint64_t rgfxLibraryObjId;

/// The entry-point name, as a NUL-terminated C string valid for the function's
/// lifetime (what a pipeline create copies to the wire).
- (const char *)rgfxEntryPointUTF8;

/// The function constants that specialise this function (null when none), and
/// their count. Borrowed: valid for the function's lifetime.
- (nullable const RgfxFunctionConstant *)rgfxConstants;
- (uint32_t)rgfxConstantCount;

/// Sets the stage `functionType` answers: the library reads it from the .metallib's
/// function list. A function made without it answers the placeholder stage.
- (void)rgfxSetFunctionType:(MTLFunctionType)type;

/// The 32 bytes of a function's bitcode hash (bitCodeHash), as the runtime's SPI types it.
typedef struct {
	uint8_t bytes[32];
} RgfxBitCodeHash;

/// Sets the hash bitCodeHash answers: the library reads the 32 bytes from the entry's HASH tag
/// in the .metallib's function list.
- (void)rgfxSetBitCodeHash:(const uint8_t *)hash;

/// The function SPI RenderBox reads as it keys its pipelines: the SHA-256 of the function's
/// bitcode, valid for the function's life (measured type: r^{?=[32C]}16@0:8).
- (const RgfxBitCodeHash *)bitCodeHash;

@end

NS_ASSUME_NONNULL_END

#endif // RGFX_FUNCTION_H_
