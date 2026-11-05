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
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "RgfxAirArguments.hpp" // RgfxAirFunction
#include "RgfxSerializer.hpp"   // RgfxLinkedFunctionRef
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

/// Sets the options `options` answers: those of the descriptor the function was made
/// from (`newFunctionWithDescriptor:`). A function made by name answers none.
- (void)rgfxSetOptions:(MTLFunctionOptions)options;

/// Sets the descriptor's `specializedName` (`newFunctionWithDescriptor:`): the name `name`
/// answers, and the one a pipeline that links the function resolves references to it by
/// (measured on an M3 Max, macOS 27.0.1). The entry point stays the one it was made from.
- (void)rgfxSetSpecializedName:(NSString *)specializedName;

/// The specialized name as a NUL-terminated C string valid for the function's lifetime;
/// null where the function was made with none.
- (nullable const char *)rgfxSpecializedNameUTF8;

/// The 32 bytes of a function's bitcode hash (bitCodeHash), as the runtime's SPI types it.
typedef struct {
	uint8_t bytes[32];
} RgfxBitCodeHash;

/// Sets the hash bitCodeHash answers: the library reads the 32 bytes from the entry's HASH tag
/// in the .metallib's function list.
- (void)rgfxSetBitCodeHash:(const uint8_t *)hash;

/// Sets what `patchType` and `patchControlPointCount` answer: the library reads them from the
/// .metallib's function list (a post-tessellation vertex function's TESS tag). A function made
/// without them answers MTLPatchTypeNone and -1, as Metal's answer for a function with no
/// [[patch]].
- (void)rgfxSetPatchType:(MTLPatchType)type controlPointCount:(NSInteger)count;

/// Sets the function Metal's own library made of the same entry point (`shadow`, an extern
/// function of the library's shadow, RgfxLibrary): what Metal's Core Image link reads in this
/// function's place, since it reads Metal's internal objects no RgfxFunction holds. A function
/// with a shadow hands each message it does not implement to the shadow, which answers it as
/// Metal's own function of the same entry point: Core Image reflects its kernels through Metal's
/// private getters (measured: macOS 26.6.1, `-returnType`). Each selector so handed is logged
/// once per class ("RGFX: answered by Metal's own function: -[RgfxFunction returnType]").
- (void)rgfxSetShadow:(id<MTLFunction>)shadow;

/// The shadow `rgfxSetShadow:` set; nil for none.
- (nullable id<MTLFunction>)rgfxShadow;

/// The shadow, made where there is none of the library's shadow by entry point, for a function made
/// with no constant values (its own, as Metal's library answers newFunctionWithName:, measured:
/// macOS 27); nil where it was made with values or its library has no shadow.
- (nullable id<MTLFunction>)rgfxShadowOrMake;

/// The library the function was made from; nil where it was made without one.
- (nullable RgfxResource *)rgfxLibrary;
/// The reflection of the buffer the function takes at `bufferIndex`: that binding of its library's
/// reflection of it (newArgumentEncoderWithBufferIndex:reflection:); nil, logged, where there is
/// none.
- (nullable MTLArgument *)rgfxArgumentOfBufferAtIndex:(NSUInteger)bufferIndex;

#ifdef __cplusplus
/// Sets the entry point's binding arguments, as its AIR module declares them: the library reads
/// them from the bitcode its .metallib holds for the entry point (RgfxAirArguments.hpp).
- (void)rgfxSetAirFunction:(std::shared_ptr<const RgfxAirFunction>)air;
/// The arguments `rgfxSetAirFunction:` set; null where the library could not read them, so
/// which textures the function writes is unknown.
- (std::shared_ptr<const RgfxAirFunction>)rgfxAirFunction;
/// Sets a visible function's signature, as its AIR module declares it: the library reads it from
/// the bitcode its .metallib holds for the function (RgfxAirArguments.hpp). The function then
/// answers -returnType, -arguments and -bindings from it (RgfxReflection.h) where Metal's own
/// function would make them with its device's compiler, which ours does not have: nil and an
/// empty array for one that is not stitchable; Metal's own function answers where it is null, or
/// of a type the reader does not know.
- (void)rgfxSetAirVisible:(std::shared_ptr<const RgfxAirVisible>)visible;
#endif

/// The function SPI RenderBox reads as it keys its pipelines: the SHA-256 of the function's
/// bitcode, valid for the function's life (measured type: r^{?=[32C]}16@0:8).
- (const RgfxBitCodeHash *)bitCodeHash;

/// Marks the function as made with no constant values (`-newFunctionWithName:`) from an entry
/// that declares the constants `declared` (by name, non-empty). Metal answers them from
/// `functionConstantsDictionary` and builds no pipeline of such a function (measured: its
/// pipeline creators assert); a function specialised by values answers none.
- (void)rgfxSetUnspecialisedConstants:(NSDictionary<NSString *, MTLFunctionConstant *> *)declared;

/// Whether the function declares constants and was made with no values for them: a pipeline of
/// it is refused.
- (BOOL)rgfxUnspecialised;

/// The refusal of a pipeline of `function` in the descriptor's `role` ("vertexFunction",
/// "fragmentFunction", "function"), where it is an unspecialised RgfxFunction: Metal's own
/// words at its assertion, as an NSError; nil for any other function.
+ (nullable NSError *)rgfxUnspecialisedError:(nullable id<MTLFunction>)function role:(NSString *)role;

@end

#ifdef __cplusplus
/// The functions a pipeline stage links, as its create sends them: their references,
/// and the group names those of a group point into.
struct RgfxLinkedFunctionList {
	std::vector<RgfxLinkedFunctionRef> refs;
	std::deque<std::string> groups;
};

/// Appends `linked`'s functions to `out` in the wire's order: its functions, its
/// binary functions, each group's in the order of their names, then its private
/// functions, each with the options it was made with. Answers the first that is not
/// an RgfxFunction (another device's library made it, so it names no library the host
/// holds), leaving `out` partly filled; nil when every one is, or `linked` is nil.
id<MTLFunction> _Nullable RgfxCollectLinkedFunctions(MTLLinkedFunctions *_Nullable linked,
                                                     RgfxLinkedFunctionList &out);
#endif

NS_ASSUME_NONNULL_END

#endif // RGFX_FUNCTION_H_
