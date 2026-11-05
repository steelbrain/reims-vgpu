/*
 * RgfxTexture: the reims-graphics bundle's MTLTexture wrapper (resource slice,
 * following the RgfxBuffer worked example in RgfxBuffer.h / the wrapper convention
 * in RgfxResource.h).
 *
 * A device-placed texture is created by `+textureWithDescriptor:heap:heapOffset:
 * device:serializer:` with `heap` == RGFX_OBJ_NONE, which routes an
 * `RGFX_CMD_TEXTURE_CREATE` through the serialiser (picking the objId, the GPU
 * address and the resource id) and wraps the result. A heap-placed texture passes
 * the owning heap's objId and the placement offset; the host derives its address
 * from the heap it pinned, so no device address is picked. The device category
 * `newTextureWithDescriptor:` and the heap's `newTextureWithDescriptor:offset:`
 * are thin forwarders to this factory.
 *
 * The getters answer from the stored shape (the `RgfxTextureDescriptor` the create
 * carried) and the picked resource id. A view (`newTextureViewWithPixelFormat:` and its
 * longer forms, RGFX_CMD_TEXTURE_VIEW) is an RgfxTexture of its own shape and resource id
 * that keeps its texture and answers its memory (RgfxTexture.mm, "views"). Our own ObjC++: it conforms to the real
 * `MTLTexture` protocol but subclasses only NSObject (via RgfxResource), so it
 * instantiates and is unit-tested on the Mac.
 */

#ifndef RGFX_TEXTURE_H_
#define RGFX_TEXTURE_H_

#import <IOSurface/IOSurfaceRef.h>
#import <Metal/Metal.h>

#import "RgfxResource.h"

#include "RgfxObjectModel.hpp"   // RgfxObjId
#include "reims_graphics_wire.h" // RgfxTextureDescriptor (a typedef, not a tag)

#ifdef __cplusplus
class RgfxRegion;
class RgfxSerializer;
class RgfxSlabs;
#endif

#ifdef __cplusplus
/// The wire's RgfxTextureDescriptor of `d`: each Metal enum its raw value, the four swizzle
/// channels one byte each, red lowest (RgfxTexture.mm).
RgfxTextureDescriptor rgfxTextureDescriptorFromMTL(MTLTextureDescriptor *_Nonnull d);
#endif

NS_ASSUME_NONNULL_BEGIN

@interface RgfxTexture : RgfxResource <MTLTexture>

/// Creates a texture of shape `descriptor`. When `heap` is RGFX_OBJ_NONE the
/// texture is device-placed (a GPU address is picked); otherwise it lies at
/// `heapOffset` in the heap `heap`. A resource id is picked either way, and given
/// back as the texture goes. Serialises an `RGFX_CMD_TEXTURE_CREATE` through
/// `serializer`; `device` is the owning shell (nil in a unit test). Returns nil if
/// `serializer` is null or `descriptor` is nil, or where no resource id is left
/// (every chunk of the guest's pool is held), as Metal answers a texture it cannot
/// make.
+ (nullable instancetype)textureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                          heap:(RgfxObjId)heap
                                    heapOffset:(uint64_t)heapOffset
                                        device:(nullable id<MTLDevice>)device
                                    serializer:(nullable RgfxSerializer *)serializer;

/// Creates a texture of shape `descriptor` over `buffer` from `offset`, each row `bytesPerRow`
/// apart, serialising an `RGFX_CMD_TEXTURE_FROM_BUFFER`; the texture keeps `buffer`, which its
/// `buffer`, `bufferOffset` and `bufferBytesPerRow` answer. Nil where textureWithDescriptor:
/// answers nil.
+ (nullable instancetype)textureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                        buffer:(RgfxResource<MTLBuffer> *)buffer
                                        offset:(uint64_t)offset
                                   bytesPerRow:(uint64_t)bytesPerRow
                                        device:(nullable id<MTLDevice>)device
                                    serializer:(nullable RgfxSerializer *)serializer;

/// As textureWithDescriptor:buffer:offset:bytesPerRow:device:serializer:, by Metal's linear form
/// (newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:), which takes a 2D array:
/// serialises an `RGFX_CMD_LINEAR_TEXTURE_FROM_BUFFER` carrying `bytesPerImage`.
+ (nullable instancetype)linearTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                              buffer:(RgfxResource<MTLBuffer> *)buffer
                                              offset:(uint64_t)offset
                                         bytesPerRow:(uint64_t)bytesPerRow
                                       bytesPerImage:(uint64_t)bytesPerImage
                                              device:(nullable id<MTLDevice>)device
                                          serializer:(nullable RgfxSerializer *)serializer;

#ifdef __cplusplus
/// Creates a Managed texture of shape `descriptor` whose CPU copy is a Shared buffer of `slabs`
/// where given (the device took RGFX_FEATURE_MANAGED_TEXTURE_SLABS), as a device buffer of its
/// size takes (RgfxSlabs.hpp: the process's own pages where the device takes them), else over a
/// block of `region`, in the layout RgfxManagedLayout.hpp gives, which the guest maps:
/// `replaceRegion:` writes it and tells the host the region (RGFX_CMD_TEXTURE_REPLACE_REGION),
/// `getBytes:` reads it, and a blit encoder's synchronize has the host copy the GPU's copy into
/// it (docs/graphics-kext/design.md, Decided 4). Serialises the copy's Shared buffer and an
/// RGFX_CMD_MANAGED_TEXTURE_CREATE. Nil for a pixel format or shape no CPU copy holds, or where
/// the slabs or the region have no room or no resource id is left: the caller makes the texture
/// without a CPU copy then.
+ (nullable instancetype)managedTextureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                               region:(RgfxRegion *)region
                                                slabs:(nullable RgfxSlabs *)slabs
                                               device:(nullable id<MTLDevice>)device
                                           serializer:(RgfxSerializer *)serializer;

/// Designated initialiser: wraps an already-created host texture. `objId`,
/// `gpuAddress` (0 when heap-placed) and `resourceId` are the guest-picked values
/// the create returned; `desc` is the mapped shape the getters answer from. The
/// texture gives `resourceId` back to `serializer` as it goes.
- (instancetype)initWithObjId:(uint64_t)objId
                   gpuAddress:(uint64_t)gpuAddress
                   resourceId:(uint64_t)resourceId
                   descriptor:(const RgfxTextureDescriptor &)desc
                       device:(nullable id<MTLDevice>)device
                   serializer:(nullable RgfxSerializer *)serializer NS_DESIGNATED_INITIALIZER;
#endif

- (instancetype)initWithObjId:(uint64_t)objId
                       device:(nullable id<MTLDevice>)device
                   serializer:(nullable RgfxSerializer *)serializer NS_UNAVAILABLE;

/// Keeps the members of `descriptor` that the wire's texture descriptor does not carry
/// (allowGPUOptimizedContents, compressionType), which the texture answers. Each
/// factory calls it on the texture it made; until then the texture answers Metal's defaults.
- (void)rgfxTakeDescriptor:(nullable MTLTextureDescriptor *)descriptor;

/// Whether the texture lies over a Managed buffer of one copy (RgfxBuffer's rgfxOneCopy), itself
/// or as a view of one that does: a synchronize of it has nothing to bring over.
- (BOOL)rgfxOverOneCopy;

@end

/// Ends the process, with Metal's words, at a descriptor Metal's texture descriptor validation
/// (`-[MTLTextureDescriptorInternal validateWithDevice:]`) ends it at, as the AMD 5300M and the
/// Intel 630 under macOS 26.7.1 end it at every texture creator (oracle_texdesc at the device's, a
/// Private heap's and a Shared buffer's; oracle_surftex over an IOSurface; Q12; Q20): one
/// assertion listing every rule broken, a line each: a width, height, depth or array length of 0
/// or past what the type takes (16384 a side, 2048 for a 3D texture's and an array's), a format
/// the device lacks (Invalid, the three-channel formats, the packed 16-bit ones, sRGB R8 and RG8,
/// ASTC, ETC2, EAC, PVRTC, the XR10 formats, Unspecialized, every two-plane one, and
/// Depth24Unorm_Stencil8 and X24_Stencil8 as the lane answers no Depth24Stencil8, as the Intel 630
/// does), a depth, height, array length, sample count or level count the type does not take,
/// more levels than the size holds, a cube not square, a multisample texture of a format that has
/// none or a count the device does not make (`device`'s supportsTextureSampleCount:, else the x86
/// Macs' 2, 4 and 8), a compressed, depth or 4:2:2 format of a type it does not take, a 4:2:2
/// one of an odd width or with levels, a depth, stencil or multisample texture in Shared or
/// Managed storage, Memoryless storage and a storage mode Metal has no name for, and usage bit
/// 0x80. A type past MTLTextureTypeTextureBuffer ends it with no words, as there. The device's, the heaps' and a buffer's texture creators, by the public form and the
/// linear and tiled SPI, and the creator over an IOSurface, call it before they send the host
/// anything; it returns for any other descriptor.
void rgfxEndAtInvalidTextureDescriptor(MTLTextureDescriptor *descriptor, id<MTLDevice> device = nil);

/// Ends the process, with Metal's words, where a heap's texture creator ends it after the
/// descriptor's validation, as the AMD 5300M and the Intel 630 under macOS 26.7.1 end it at a
/// Private heap (oracle_texdesc heap): a descriptor of another storage mode than the heap's.
void rgfxEndAtHeapStorageMismatch(MTLTextureDescriptor *descriptor, MTLStorageMode heapMode);

/// Ends the process, with Metal's words, where `minimumLinearTextureAlignmentForPixelFormat:` ends
/// it on the x86-64 Macs, and returns for the 64 raw formats both answer (256), alike on the AMD
/// 5300M and the Intel 630 under macOS 26.7.1 with no validation layer: of every raw value from 0
/// to 799, and 1000, 1023, 4096 and 65535 (oracle_bundle linalign), they answer 46 of the SDK's
/// formats and 18 of no name there (84, 85, 300 to 304, 313 to 315, 323 to 325, 556, 586, 587, 628
/// and 629), and end the process at every other: the block-compressed and 4:2:2 formats ("Linear
/// textures do not support compressed pixel formats"), the depth and stencil ones ("...
/// depth/stencil pixel formats"), and every other value, a format the device lacks, which Metal
/// names MTLPixelFormatInvalid (Invalid itself, sRGB R8 and RG8, the three-channel formats, PVRTC,
/// EAC, ETC2, ASTC, the packed 16-bit ones, D24S8 and X24S8 as the lane answers no
/// Depth24Stencil8, as the Intel 630 does, Unspecialized, the XR10 formats, the Y'CbCr ones, and
/// values no format has).
void rgfxEndAtUnalignableLinearFormat(MTLPixelFormat format);

/// Ends the process, with Metal's words, where newTextureWithDescriptor:iosurface:plane: ends it on
/// the AMD 5300M and the Intel 630 under macOS 26.7.1 (oracle_surftex, docs/graphics-kext/
/// dgpu-oracle.md, Q20). At the descriptor's own validation, over any surface
/// (rgfxEndAtInvalidTextureDescriptor, every two-plane format among those the device lacks, Q13).
/// Then over a surface without planes: at a type other than 2D or TextureBuffer, a depth other than 1, more
/// than one level, rows wider than the surface's bytesPerRow (a compressed or 4:2:2 format's
/// counted in its blocks), a depth or stencil format, a plane other than 0, and more rows than
/// the surface's. The device's creator calls it before it sends the host anything; it returns for
/// any other texture, which the creator makes, or answers nil for (RgfxIOSurfaceTexture).
void rgfxEndAtInvalidSurfaceTexture(MTLTextureDescriptor *descriptor, IOSurfaceRef surface,
                                    NSUInteger plane, id<MTLDevice> device = nil);

/// Ends the process, with Metal's words, at a view of a texture in `from`, of `fromType`,
/// `levelCount` levels and `sliceCount` slices, that Metal's validation ends it at on the AMD 5300M
/// and the Intel 630 under macOS 26.7.1 with no validation layer (oracle_bundle view.*): in no
/// format (MTLPixelFormatInvalid, which the M3 Max under macOS 27.0.1 answers nil for); in a
/// format of one texel a block of other bytes than the texture's (RGBA16Float in RGBA8Unorm); and,
/// for a view naming its type, levels and slices (`subrange`), levels or slices past the
/// texture's, a type of another family (a 2D texture as 3D), or a cube's slices not a multiple of
/// six, as the M3 Max's does. It returns for any other view, among them those the oracles did not
/// measure (empty ranges, other slice counts, compressed formats), which the bundle answers nil.
void rgfxEndAtViewMetalEnds(MTLPixelFormat from, MTLTextureType fromType, uint64_t levelCount,
                            uint64_t sliceCount, MTLPixelFormat format, MTLTextureType type,
                            NSRange levels, NSRange slices, bool subrange);

/// A device whose linear texture alignment the bundle's own callers ask without
/// `minimumLinearTextureAlignmentForPixelFormat:`, which ends the process at some formats
/// (rgfxEndAtUnalignableLinearFormat), where Metal's own creators do not.
@protocol RgfxLinearAlignmentSource <NSObject>
- (NSUInteger)rgfxLinearTextureAlignmentForPixelFormat:(MTLPixelFormat)format;
@end

NS_ASSUME_NONNULL_END

#endif // RGFX_TEXTURE_H_
