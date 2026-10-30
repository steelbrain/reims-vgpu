/*
 * RgfxIOSurface: textures over IOSurfaces (`newTextureWithDescriptor:iosurface:plane:`).
 *
 * IOSurface.kext allocates a surface in guest RAM, which the host cannot reach, and
 * processes share a surface by its global IOSurfaceID (WindowServer and its clients
 * alias the same window and drawable surfaces). So a surface's own pages stay the one
 * copy every process and the CPU agree on, and each process that makes a texture over
 * a surface's plane keeps a copy of that plane in host memory the GPU reaches: a
 * block of the device region, one per (IOSurfaceID, plane) in the process, rows
 * kShadowRowAlign apart, with a Shared buffer over it and the texture over that buffer
 * (TEXTURE_FROM_IOSURFACE, which also names the texture by the surface's id on the
 * host, for presents).
 *
 * The copy syncs with the pages at two points:
 *   - before a commit, from the pages, for every copy whose surface's seed moved since
 *     the copy last synced (a CPU wrote it under a lock, or another process wrote its
 *     GPU results back);
 *   - when a command buffer that writes the copy completes, before it reports
 *     completion, back to the pages. A write is a render attachment, a blit's
 *     destination or a compute texture with ShaderWrite usage.
 * The seed after our own write-back is recorded, so a write-back is never read back.
 *
 * Left for later: zero copies (the host reaching the surface's pages, or one host
 * texture per IOSurfaceID that every process shares); CPU writes made without a lock;
 * a texture's own CPU reads and writes cover one-element-per-pixel formats only.
 *
 * Our own ObjC++.
 */

#ifndef RGFX_IOSURFACE_H_
#define RGFX_IOSURFACE_H_

#import <IOSurface/IOSurfaceRef.h>
#import <Metal/Metal.h>

#import "RgfxTexture.h"

#include <memory>
#include <vector>

class RgfxEncoder;
class RgfxRegion;
class RgfxSerializer;

NS_ASSUME_NONNULL_BEGIN

@interface RgfxIOSurfaceTexture : RgfxTexture

/// A texture of shape `descriptor` over plane `plane` of `surface`, in this process's
/// copy of the plane, which is made in `region` on first use and synced from the
/// surface's pages. Nil when the surface has no such plane, its plane has no CPU
/// mapping, the region has no room, or no texture resource id is left.
+ (nullable instancetype)textureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                     iosurface:(IOSurfaceRef)surface
                                         plane:(NSUInteger)plane
                                        region:(RgfxRegion *)region
                                        device:(nullable id<MTLDevice>)device
                                    serializer:(RgfxSerializer *)serializer;

@end

NS_ASSUME_NONNULL_END

/// Before a commit: syncs every copy in this process whose surface's seed moved from
/// the surface's pages.
void rgfxIOSurfaceSyncBeforeCommit(void);

/// Records on `enc` that its command buffer writes `texture`, when `texture` is one
/// over an IOSurface (otherwise nothing).
void rgfxIOSurfaceNoteWrite(RgfxEncoder *_Nullable enc, id _Nullable texture);

/// As rgfxIOSurfaceNoteWrite, for a texture a shader may write: only when its usage
/// has ShaderWrite.
void rgfxIOSurfaceNoteShaderWrite(RgfxEncoder *_Nullable enc, id _Nullable texture);

/// At a command buffer's completion: writes each copy in `writes` (taken from its
/// encoder) back to its surface's pages, once each.
void rgfxIOSurfaceWriteBack(const std::vector<std::shared_ptr<void>> &writes);

#endif // RGFX_IOSURFACE_H_
