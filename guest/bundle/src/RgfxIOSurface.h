/*
 * RgfxIOSurface: textures over IOSurfaces (`newTextureWithDescriptor:iosurface:plane:`).
 *
 * IOSurface.kext allocates a surface in guest RAM, and processes share a surface by its global
 * IOSurfaceID (WindowServer and its clients alias the same window and drawable surfaces). The GPU
 * works on guest memory in place, and copies a surface only where the host cannot use its pages
 * (docs/graphics-kext/design.md, Decided 14). Either way, a surface's GPU view holds every plane,
 * and every process that holds the surface shares it through the surface's shared block, the one
 * every holder gets from the accelerator kext. Each process makes a Shared buffer over the view
 * and each texture over its plane's rows in that buffer (TEXTURE_FROM_IOSURFACE, which also names
 * the texture by the surface's id on the host, for presents, and carries the surface's facts,
 * RgfxSurfaceFacts: its OSType, planes, size, plane 1's place in the view and its YCbCr matrix,
 * which a texture over plane 0 that reads plane 1, a two-plane YCbCr one, needs). What one
 * process's GPU draws there, another's samples there, with no copy.
 *
 * In place, the view is the surface's own pages. The process asks for the shared block in place
 * (RgfxRegion::acquireSurfacePages) with the surface's page-aligned base and its whole allocation,
 * which the kext wires and hands the device (SURFACE_PAGES), and makes its buffer over them
 * (BUFFER_OVER_SURFACE_PAGES); each texture lies over its plane's own offset from the base and
 * its own rows, and the block holds only the header and the writers (below). Nothing is copied
 * or synced: the GPU and every CPU read and write the same pages, and the texture's own CPU
 * access is the pages too. Such a buffer has no GPU address, and nor have its textures. A
 * surface goes in place only where the host's textures can lie over its planes: its base is a
 * page's, each plane's offset from it (CoreVideo's planar buffers start plane 0 past a header)
 * and its rows are multiples of the alignment the host's linear textures ask (the device info's,
 * RgfxCaps::linearTextureAlignment), and every plane lies within the allocation. In place, a
 * surface takes one page of the region however large it is.
 *
 * Where it does not, or the device uses no surface in place (no RGFX_FEATURE_SURFACE_PAGES), or
 * the host or the kext refuses the surface's pages, the view is a copy: the surface's own pages
 * stay its CPU copy, which every process and the CPU agree on, and its GPU view is the shared
 * block (RgfxRegion::acquireSurface), found by the pages of plane 0 the process maps, holding
 * every plane. Where the kext gives no shared block, the process keeps a block of its own for the
 * surface. The kext gives a surface's blocks in one mode, so every holder of a surface copied
 * copies it. Each view made, in place or a copy and why, is counted in the "surface copies" line
 * the process logs to /tmp/rgfx-surface-copies.log every 10 s while views are made or copied.
 *
 * The copy holds the planes in plane order, each from a kShadowRowAlign multiple, its rows
 * kShadowRowAlign apart, as the surface's own memory holds them one after another; a header past
 * the last (RgfxSurfaceHeader) holds the surface's seed when the copy last matched the pages,
 * which every holder reads and writes. The copy syncs with the pages, every plane, at two
 * points:
 *   - before a commit, from the pages, where the surface's seed is not the header's: a CPU
 *     wrote the surface under a lock since any holder synced it;
 *   - when a command buffer that writes the copy completes, before it reports completion,
 *     back to the pages, so a CPU lock in any process reads the GPU's results. A write is a
 *     render attachment, a blit's destination or a compute texture with ShaderWrite usage.
 * The seed after a write-back is the header's, so no holder reads that write back.
 *
 * After the header lie the view's writers (RgfxSurfaceWriters, reims_graphics_wire.h), which the
 * texture's create names to the host: a present of the surface shows the view once the command
 * buffers they name have finished. A commit of a buffer that writes the view names its queue and
 * the buffer's seqno there before the commit returns, so a present the process makes after it
 * waits for it on the host. Each process writes only the slots naming its own queues: it raises
 * its queue's seqno in the slot it holds, takes again one naming a queue of its own that is gone
 * (whose buffers have all finished), or takes an empty one (queue 0); it empties its slots as
 * its view goes, when the buffers that wrote it have finished. A slot given to another queue has
 * its seqno emptied first, then its queue written, then the seqno, so the host, which reads the
 * queue, the seqno and the queue again, never pairs one queue with another's seqno. A block made
 * again has its writers emptied by the process that made it, before its texture is created.
 *
 * IOSurface's use count is left as it is: Metal neither raises it nor reads it for GPU work
 * over a surface (measured, an M3 Max under macOS 27: IOSurfaceGetUseCount stays 0 and
 * IOSurfaceIsInUse false while a command buffer that renders into a surface, or blits from it,
 * is held in flight, and after it completes).
 *
 * A texture the creator cannot make is nil, as Metal answers one, and never silently: each cause
 * (no view in place and no room for a copy, a plane past the surface's, no texture resource id
 * left, ...) logs one "RGFX:" line per surface naming the surface and the checks that refused,
 * however often the texture is asked for (rgfxIOSurfaceNilLog).
 *
 * Left for later, for a copy: syncing the pages at the CPU's lock instead of at every GPU
 * write's completion (IOSurface's device caches); CPU writes made without a lock. In either
 * mode, a texture's own CPU reads and writes cover one-element-per-pixel formats only.
 *
 * Our own ObjC++.
 */

#ifndef RGFX_IOSURFACE_H_
#define RGFX_IOSURFACE_H_

#import <IOSurface/IOSurfaceRef.h>
#import <Metal/Metal.h>

#import "RgfxTexture.h"

#include <memory>
#include <string>
#include <vector>

class RgfxEncoder;
class RgfxRegion;
class RgfxSerializer;

NS_ASSUME_NONNULL_BEGIN

@interface RgfxIOSurfaceTexture : RgfxTexture

/// A texture of shape `descriptor` over plane `plane` of `surface`, over the plane's GPU
/// view, taken on first use: the surface's pages in place, else a copy in its shared block or a
/// block of this process's own in `region`, synced from the surface's pages. Nil when the
/// surface has no such plane, its plane has no CPU mapping, the region has no room, or no
/// texture resource id is left.
+ (nullable instancetype)textureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                     iosurface:(IOSurfaceRef)surface
                                         plane:(NSUInteger)plane
                                        region:(RgfxRegion *)region
                                        device:(nullable id<MTLDevice>)device
                                    serializer:(RgfxSerializer *)serializer;

@end

NS_ASSUME_NONNULL_END

/// Before a commit: syncs every copy this process uses from the surface's pages, where the
/// surface's seed moved since any holder synced it (a view in place needs none).
void rgfxIOSurfaceSyncBeforeCommit(void);

/// Records on `enc` that its command buffer writes `texture`, when `texture` is one
/// over an IOSurface or a view of one, through any views (otherwise nothing).
void rgfxIOSurfaceNoteWrite(RgfxEncoder *_Nullable enc, id _Nullable texture);

/// As rgfxIOSurfaceNoteWrite, for a texture a shader may write: only when its usage
/// has ShaderWrite.
void rgfxIOSurfaceNoteShaderWrite(RgfxEncoder *_Nullable enc, id _Nullable texture);

/// After a command buffer is posted at seqno `seqno` on queue `queue` (whose id is
/// `queueObjId`): names it among the writers of each copy in `writes` (taken from its encoder),
/// once each, so a present of the surface after this waits for it.
void rgfxIOSurfaceNoteCommitted(const std::vector<std::shared_ptr<void>> &writes, id _Nonnull queue,
                                uint64_t queueObjId, uint64_t seqno);

/// At a command buffer's completion: writes each copy in `writes` (taken from its
/// encoder) back to its surface's pages, once each (a view in place needs none).
void rgfxIOSurfaceWriteBack(const std::vector<std::shared_ptr<void>> &writes);

/// How many textures over `surface` were answered nil for `cause` ("arguments", "plane",
/// "plane shape", "region", "buffer", "resource id", "texture"), and the "RGFX:" line logged
/// for the first, without the prefix (for the tests): 0 and empty where none was.
uint64_t rgfxIOSurfaceNilLog(IOSurfaceRef _Nullable surface, const char *_Nonnull cause,
                             std::string *_Nonnull line);

#endif // RGFX_IOSURFACE_H_
