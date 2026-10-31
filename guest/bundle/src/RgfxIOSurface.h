/*
 * RgfxIOSurface: textures over IOSurfaces (`newTextureWithDescriptor:iosurface:plane:`).
 *
 * IOSurface.kext allocates a surface in guest RAM, which the host GPU does not use in place
 * (docs/graphics-kext/design.md, Decided 5 and 14), and processes share a surface by its global
 * IOSurfaceID (WindowServer and its clients alias the same window and drawable surfaces). So a
 * surface's own pages stay its CPU copy, which every process and the CPU agree on, and its GPU
 * view is one block of the device region per plane: the plane's shared block, the one every
 * process that holds the surface gets from the accelerator kext (RgfxRegion::acquireSurface),
 * found by the plane's pages the process maps. Each process makes a Shared buffer over the
 * block and its texture over that buffer (TEXTURE_FROM_IOSURFACE, which also names the texture
 * by the surface's id on the host, for presents). What one process's GPU draws there, another's
 * samples there, with no copy. Where the kext gives no shared block, the process keeps a block
 * of its own for the plane, as before.
 *
 * The block's rows are kShadowRowAlign apart, and a header past them (RgfxSurfaceHeader) holds
 * the surface's seed when the block last matched the pages, which every holder reads and
 * writes. The block syncs with the pages at two points:
 *   - before a commit, from the pages, where the surface's seed is not the header's: a CPU
 *     wrote the surface under a lock since any holder synced it;
 *   - when a command buffer that writes the block completes, before it reports completion,
 *     back to the pages, so a CPU lock in any process reads the GPU's results. A write is a
 *     render attachment, a blit's destination or a compute texture with ShaderWrite usage.
 * The seed after a write-back is the header's, so no holder reads that write back.
 *
 * After the header lie the copy's writers (RgfxSurfaceWriters, reims_graphics_wire.h), which the
 * texture's create names to the host: a present of the surface shows the copy once the command
 * buffers they name have finished. A commit of a buffer that writes the copy names its queue and
 * the buffer's seqno there before the commit returns, so a present the process makes after it
 * waits for it on the host. Each process writes only the slots naming its own queues: it raises
 * its queue's seqno in the slot it holds, takes again one naming a queue of its own that is gone
 * (whose buffers have all finished), or takes an empty one (queue 0); it empties its slots as
 * its copy goes, when the buffers that wrote it have finished. A slot given to another queue has
 * its seqno emptied first, then its queue written, then the seqno, so the host, which reads the
 * queue, the seqno and the queue again, never pairs one queue with another's seqno. A block made
 * again has its writers emptied by the process that made it, before its texture is created.
 *
 * IOSurface's use count is left as it is: Metal neither raises it nor reads it for GPU work
 * over a surface (measured, an M3 Max under macOS 27: IOSurfaceGetUseCount stays 0 and
 * IOSurfaceIsInUse false while a command buffer that renders into a surface, or blits from it,
 * is held in flight, and after it completes).
 *
 * Left for later: syncing the pages at the CPU's lock instead of at every GPU write's
 * completion (IOSurface's device caches); CPU writes made without a lock; a texture's own CPU
 * reads and writes cover one-element-per-pixel formats only.
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

/// A texture of shape `descriptor` over plane `plane` of `surface`, over the plane's GPU
/// view: its shared block, or a block of this process's own in `region`, taken on first use
/// and synced from the surface's pages. Nil when the surface has no such plane, its plane has
/// no CPU mapping, the region has no room, or no texture resource id is left.
+ (nullable instancetype)textureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                     iosurface:(IOSurfaceRef)surface
                                         plane:(NSUInteger)plane
                                        region:(RgfxRegion *)region
                                        device:(nullable id<MTLDevice>)device
                                    serializer:(RgfxSerializer *)serializer;

@end

NS_ASSUME_NONNULL_END

/// Before a commit: syncs every plane's block this process uses from the surface's pages,
/// where the surface's seed moved since any holder synced it.
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

/// At a command buffer's completion: writes each block in `writes` (taken from its
/// encoder) back to its surface's pages, once each.
void rgfxIOSurfaceWriteBack(const std::vector<std::shared_ptr<void>> &writes);

#endif // RGFX_IOSURFACE_H_
