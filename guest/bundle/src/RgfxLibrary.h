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
#include <memory>
#include <string>
#include <vector>

#include "RgfxAirArguments.hpp"

class RgfxRegion;
class RgfxSerializer;

/// A .metallib's install name and the install names of the dynamic libraries it was linked
/// against, in link order: its extended header's HDYN region, a tag list of NAME and one DYNL per
/// library (read from Apple's `metal -dynamiclib` and `-l` output, oracle_fnptr's libraries).
struct RgfxMetallibLinkage {
	std::string installName;
	std::vector<std::string> dependencies;
};

/// Reads `n` bytes at `p` as a .metallib's linkage into `out`: the extended header follows the
/// function list (the header's u64 offset at 0x18 and size at 0x20, which counts the entries and
/// not the count before them), each tag a 4-byte name, a u16 size and the data, up to ENDT; its
/// HDYN holds the u64 offset and size of the region. False, `out` empty, where the bytes are no
/// .metallib or a span runs past them; true with `out` empty for a library with no HDYN.
bool rgfxReadMetallibLinkage(const uint8_t *_Nullable p, size_t n, RgfxMetallibLinkage &out);
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

/// The shadow it was given (`rgfxAdoptShadow:`): Metal's own library of the compile, link or
/// stitching that made it; nil where it was given none (a shadow made over its bytes is not).
- (nullable id<MTLLibrary>)rgfxAdoptedShadow;

/// The private MTLLibrary members Core Image calls (measured: macOS 27 and 26.6.1): the
/// shadow's extern function names (empty with no shadow), and the function of `name` whose
/// shadow is the shadow's own extern function of it, of its `functionType` (nil where the
/// shadow holds none).
- (NSArray<NSString *> *)externFunctionNames;
- (nullable id<MTLFunction>)newExternFunctionWithName:(NSString *)name;

/// Marks the library as made from `url` (newLibraryWithURL:, and the calls that make theirs so),
/// which Metal does not serialize, whose last part an executable library answers as its install
/// name, and whose install names a pipeline loads (none of a library of no file).
- (void)rgfxSetMadeFromURL:(NSURL *)url;

/// The URL the library was made from (rgfxSetMadeFromURL:), nil for any other.
- (nullable NSURL *)rgfxURL;

/// The bytes the library was made from, nil where it kept none (one Metal's link or stitching
/// made).
- (nullable dispatch_data_t)rgfxSource;

/// The function table the library reads its functions from: one object for every library of the
/// process made from bytes of one content key (tests).
- (const void *)rgfxFunctionTable;

/// The private MTLLibrary members MPS and Core Image call, as Metal answers them (measured: macOS
/// 27): nil for the bitcode of a library made from bytes, and its shadow's for one Metal's link or
/// stitching made; and the library written to `url` by its shadow, or Metal's refusal of one made
/// from a URL.
- (nullable id)bitcodeData;
- (BOOL)serializeToURL:(NSURL *)url error:(NSError *_Nullable __autoreleasing *_Nullable)error;

#ifdef __cplusplus
/// The install names of the dynamic libraries the library was linked against
/// (rgfxReadMetallibLinkage of its bytes), read once: empty where it kept no bytes.
- (std::vector<std::string>)rgfxDependencies;

/// The entry point `name`'s nodes whole, with its module's function constants (rgfxReadAirEntries;
/// `entries` then holds that one entry), read once from the bitcode the library's bytes hold for
/// it: null where the library kept no bytes (one Metal's link or stitching made), or its module or
/// a node of the entry point does not read. A pipeline's reflection is made from it.
- (std::shared_ptr<const RgfxAirEntries>)rgfxAirEntryNamed:(const std::string &)name;
#endif

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
/// The library the device holds under `lookup` (32 bytes) for the `length` bytes at `bytes`, found
/// by an RGFX_CMD_LIBRARY_LOOKUP, with no bytes sent: their content key `key` (32 bytes), or the
/// key of the whole bytes (rgfxLibraryWholeKey), which its pipelines' keys still name it by `key`;
/// its function table is read from the bytes. Returns nil if `serializer` is null or the device
/// holds no such library.
+ (nullable instancetype)libraryFoundWithKey:(const uint8_t *)key
                                      lookup:(const uint8_t *)lookup
                                       bytes:(const void *)bytes
                                      length:(NSUInteger)length
                                      device:(nullable id<MTLDevice>)device
                                  serializer:(nullable RgfxSerializer *)serializer;

/// Creates a library from `length` bytes of .metallib/AIR at `bytes` by streaming
/// them through a staging window of `region` made for this library and given back
/// once the host has copied it out (RgfxLibraryStream.hpp), one
/// RGFX_CMD_LIBRARY_FROM_REGION per chunk, for a library larger than one ring
/// message. The chunks carry the content key where `key` (32 bytes) is not null.
/// Returns nil if `serializer` or `region` is null, the region has no room for a
/// window even of page-sized chunks, the host did not take a chunk (`*status`
/// kRgfxNoAnswer), or the device refused the library (`*status` its answer to the
/// last chunk).
+ (nullable instancetype)libraryStreamingBytes:(const void *)bytes
                                       length:(NSUInteger)length
                                          key:(nullable const uint8_t *)key
                                       region:(nullable RgfxRegion *)region
                                       device:(nullable id<MTLDevice>)device
                                   serializer:(nullable RgfxSerializer *)serializer
                                       status:(uint32_t *)status;

@end

/// A library of the functions given (newLibraryWithFunctionArray:error:, SPI), as both x86 Macs
/// make one (the AMD 5300M and the Intel 630 under macOS 26.7.1: oracle_bundle libforms2 and
/// libforms3). It holds the functions themselves, and nothing on the host: it answers each by its
/// name, the very object given, whatever constants or descriptor ask for it, and its names in the
/// reverse of the order given. It is executable, with no install name, label, reflection or
/// extern functions, and serializes nothing, with Metal's words.
@interface RgfxFunctionArrayLibrary : RgfxResource <MTLLibrary>

/// The library of `functions`; nil and Metal's error where two share a name, and the end of the
/// process with Metal's words where there are none.
+ (nullable instancetype)libraryOfFunctions:(nullable NSArray<id<MTLFunction>> *)functions
                                     device:(nullable id<MTLDevice>)device
                                      error:(NSError *_Nullable __autoreleasing *_Nullable)error;

/// The private MTLLibrary members the probe asks (oracle_bundle libforms3): no extern functions
/// (nil), and no file written, with Metal's refusal.
- (nullable NSArray<NSString *> *)externFunctionNames;
- (BOOL)serializeToURL:(NSURL *)url error:(NSError *_Nullable __autoreleasing *_Nullable)error;

@end

/// Each of `functions` as Metal's own links take it (Core Image's link and stitching): an
/// RgfxFunction as its shadow (its library's shadow's function of its entry point, where it has
/// none yet), any other function as it is. nil, with `*error`, where an RgfxFunction has no shadow:
/// it is specialised by constants, its library keeps no source, Metal refused it, or it holds no
/// function of that name.
NSArray<id<MTLFunction>> *_Nullable RgfxShadowFunctions(NSArray<id<MTLFunction>> *functions,
                                                        NSError *__autoreleasing _Nullable *_Nullable error);

/// A link Metal's base device class runs on the CPU over Metal's own functions, reading
/// Metal-internal library data no RgfxFunction holds (measured: macOS 27, and 26.6.1 in the guest,
/// for Core Image's link; stitching reads each function's bitcode hash): `link`, given each of
/// `functions`' shadow, and the library it answers made one of ours by `makeLibrary` over the
/// bytes Metal answers for it (`libraryDataContents`, a .metallib), which keeps Metal's library as
/// its shadow. A library `link` answers that is already ours (Metal's link reached one of ours
/// again, which made it) is answered as it is.
typedef id<MTLLibrary> _Nullable (^RgfxLinkBlock)(NSArray<id<MTLFunction>> *shadows,
                                                  NSError *__autoreleasing _Nullable *_Nullable error);
typedef id<MTLLibrary> _Nullable (^RgfxLibraryMaker)(dispatch_data_t data,
                                                     NSError *__autoreleasing _Nullable *_Nullable error);
id<MTLLibrary> _Nullable RgfxLinkOverShadows(NSArray<id<MTLFunction>> *functions, RgfxLinkBlock link,
                                             RgfxLibraryMaker makeLibrary,
                                             NSError *__autoreleasing _Nullable *_Nullable error);

/// A library of source compiled at run time (`newLibraryWithSource:` and its forms): `compile`,
/// Metal's own frontend over the device, and the library it answers made one of ours by
/// `makeLibrary` over the bytes Metal answers for it (`libraryDataContents`, a .metallib of AIR),
/// which keeps Metal's library as its shadow; the host takes the bytes as it takes an app's
/// .metallib. Metal's error passes through as Metal gave it: alone where it refuses the source
/// (MTLLibraryErrorDomain 3 and its log), and with the library where it compiled it (nil, measured:
/// macOS 27, a source with an unused variable among them). A library `compile` answers that is
/// already ours is answered as it is.
typedef id<MTLLibrary> _Nullable (^RgfxCompileBlock)(NSError *__autoreleasing _Nullable *_Nullable error);
id<MTLLibrary> _Nullable RgfxCompileOverMetal(RgfxCompileBlock compile, RgfxLibraryMaker makeLibrary,
                                              NSError *__autoreleasing _Nullable *_Nullable error);

/// A library Metal's own creator `make` answers on the device (a source compile, a Metal package's,
/// an MPS graph package's), made ours as `RgfxCompileOverMetal` makes it, with Metal's error passed
/// through as Metal gave it; `creator` names the creator in the refusal of a library Metal answers
/// no bytes for.
id<MTLLibrary> _Nullable RgfxLibraryOverMetal(RgfxCompileBlock make, RgfxLibraryMaker makeLibrary,
                                              NSString *creator,
                                              NSError *__autoreleasing _Nullable *_Nullable error);

/// A stitched library of a descriptor (`newLibraryWithStitchedDescriptor:` and Metal's other
/// descriptor entry points), by `stitch`, Metal's own, over a copy of `descriptor` whose functions
/// are their shadows and which names no binary archive (Metal would read ours as its own) nor fails
/// on a miss, made ours as `RgfxLinkOverShadows` makes it, and kept in `destination` where that is
/// one of ours. Where the descriptor fails on a binary-archive miss, its archives are looked up
/// first (RgfxBinaryArchive), and where none holds it, none named included, it answers nil and
/// Metal's error, MTLLibraryErrorDomain 3 "Failed to find library in binary archive." (measured:
/// macOS 27).
typedef id<MTLLibrary> _Nullable (^RgfxStitchBlock)(MTLStitchedLibraryDescriptor *descriptor,
                                                    NSError *__autoreleasing _Nullable *_Nullable error);
id<MTLLibrary> _Nullable RgfxStitchOverShadows(MTLStitchedLibraryDescriptor *descriptor,
                                               id<MTLBinaryArchive> _Nullable destination, RgfxStitchBlock stitch,
                                               RgfxLibraryMaker makeLibrary,
                                               NSError *__autoreleasing _Nullable *_Nullable error);

#ifdef __cplusplus
/// Whether this process is yet to log that the arguments of the entry point `name` were not read:
/// true the first time a name is asked, false after. A process can make a library of the same
/// bytes again and again (measured: macOS 26.6.1's HEIC decode makes one for nearly every
/// dispatch), and each would log its functions' line once.
bool rgfxAirUnreadFirstLog(const std::string &name);
#endif

NS_ASSUME_NONNULL_END

#endif // RGFX_LIBRARY_H_
