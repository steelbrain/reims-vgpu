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
 * In place, a surface is asked cached, of any planes and format, but for a video decoder's
 * output where the host lands no picture in a device cache (RgfxAskUncachedScope): where the
 * host's GPU has memory of its own (RGFX_FEATURE_SURFACE_CACHE), the kext makes the block
 * cached, and the host keeps
 * the surface's GPU copy there, a device cache, apart from its pages (docs/graphics-kext/
 * design.md, Decided 14, "The device cache"). The pages and the cache meet at IOSurface's locks,
 * which the kext serves, and at the commit of each command buffer that uses the surface, which
 * names it before its first encoder with the surface's CPU generation from the block's header
 * (rgfxIOSurfaceNameUses), so the host fills the cache where the CPU wrote the surface under a
 * read-write lock since. The texture's own CPU access takes the surface's lock, as a CPU access
 * does. Whether a block is cached is the kext's answer, the same for every holder.
 *
 * Where it does not, or the device uses no surface in place (no RGFX_FEATURE_SURFACE_PAGES), or
 * the host or the kext refuses the surface's pages, the view is a copy: the surface's own pages
 * stay its CPU copy, which every process and the CPU agree on, and its GPU view is the shared
 * block (RgfxRegion::acquireSurface), found by the pages of plane 0 the process maps, holding
 * every plane. Where the kext gives no shared block, the process keeps a block of its own for the
 * surface. The kext gives a surface's blocks in one mode, so every holder of a surface copied
 * copies it. Each view made, in place or a copy and why, is counted in the "surface copies" line
 * the process logs (os_log, info) every 10 s while views are made or copied.
 *
 * The copy holds the planes in plane order, each from a kShadowRowAlign multiple, its rows
 * kShadowRowAlign apart, as the surface's own memory holds them one after another; a header past
 * the last (RgfxSurfaceHeader) holds the surface's seed when the copy last matched the pages,
 * which every holder reads and writes. The copy syncs with the pages, every plane, at two
 * points:
 *   - before a commit, from the pages, where the surface's seed is not the header's: a CPU
 *     wrote the surface under a lock since any holder synced it;
 *   - when a command buffer that writes the copy completes, before it reports completion,
 *     back to the pages, with no lock of the surface. A write is a render attachment, a
 *     blit's destination, a resource used with Write usage, or a texture bound at an index
 *     where a draw's or dispatch's function writes (RgfxSurfaceBindings): its AIR arguments
 *     declare a write or read_write texture there, or, where they are unknown, the texture's
 *     usage has ShaderWrite. A texture a function only reads or samples is no write.
 *
 * A CPU lock of a surface, in any process, waits for the GPU's pending writes to it, as Metal's
 * does (the kext's RgfxSurfaceLocks.hpp). A commit of a buffer that writes a surface's shared
 * block, in place or a copy, names the surface to the kext before the commit returns, or, for a
 * buffer held as Metal holds one from submission (one that waits on an event below its value, or
 * is behind such a one on its queue: RgfxCmdBuffer's rgfxPost), once it is scheduled
 * (rgfxIOSurfaceWritesPosted), and the kext keeps every lock of it waiting until the write
 * retires: at the buffer's completion, once a copy is written back (rgfxIOSurfaceRetire), or for
 * a view in place once the device has completed the buffer. The kext marks each surface named
 * the GPU's (IOSurface's device cache), which moves its seed once for the buffer as Metal's
 * submission does, and a copy records that seed in its header, so no holder takes the pages over
 * the GPU's results at its next commit. The seed after a write-back is that one.
 *
 * After the header lie the view's writers (RgfxSurfaceWriters, reims_graphics_wire.h), which the
 * texture's create names to the host: a present of the surface shows the view once the command
 * buffers they name have finished. A commit of a buffer that writes the view names its queue and
 * the buffer's seqno there before the commit returns, or, held, once it is scheduled, so a
 * present the process makes after that waits for it on the host. Each process writes only the
 * slots naming its own queues: it raises its queue's seqno in the slot it holds, takes again one
 * naming a queue of its own that is gone (whose buffers have all finished), or takes an empty one
 * (queue 0); it empties a slot once the host has finished the buffer it names
 * (rgfxIOSurfaceNoteFinished), so a slot is held only while a buffer it names runs, and empties
 * its slots as its view goes, when the buffers that wrote it have finished. A slot given to
 * another queue has its seqno emptied first, then its queue written, then the seqno, so the host,
 * which reads the queue, the seqno and the queue again, never pairs one queue with another's
 * seqno. A block made again has its writers emptied by the process that made it, before its
 * texture is created.
 *
 * Where the process names its command buffers' order (RGFX_FEATURE_SURFACE_ORDER in the
 * device's feature word; docs/graphics-kext/design.md, "Surface order"), the readers follow the
 * writers, in every block: the same record, each queue's latest buffer that only reads the view,
 * which only the bundles read. The post of a buffer that uses surfaces names
 * the buffers of other queues it runs after (rgfxIOSurfaceNameOrder), as a Mac's GPU orders a
 * process's access to an IOSurface after another's committed before it where one of the two
 * writes: the writers of each surface it uses, and the readers of each it writes, as the blocks
 * name them at its post (rgfxIOSurfaceNameOrder). Once posted it raises its own queue's slot, a
 * writer's or a reader's, so a post after it names it. A process
 * empties its reader slots as it does its writer slots.
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
 * write's completion (the device cache's sync); CPU writes made without a lock. In either
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

#include "RgfxAirArguments.hpp"

class RgfxEncoder;
class RgfxRegion;
class RgfxSerializer;

NS_ASSUME_NONNULL_BEGIN

@interface RgfxIOSurfaceTexture : RgfxTexture

/// A texture of shape `descriptor` over plane `plane` of `surface`, over the plane's GPU
/// view, taken on first use: the surface's pages in place, else a copy in its shared block or a
/// block of this process's own in `region`, synced from the surface's pages. Nil when the
/// surface has no such plane, its format is block-compressed (which Metal makes, but the host
/// makes no texture of over a surface's rows), its plane has no CPU mapping, the region has no
/// room, or
/// no texture resource id is left. The device's creator first ends the process where Metal does
/// (rgfxEndAtInvalidSurfaceTexture).
+ (nullable instancetype)textureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                     iosurface:(IOSurfaceRef)surface
                                         plane:(NSUInteger)plane
                                        region:(RgfxRegion *)region
                                        device:(nullable id<MTLDevice>)device
                                    serializer:(RgfxSerializer *)serializer;

/// The same from `origin` of the plane (elements across, rows down), in the plane's rows: the
/// video driver's landing of a picture inside a larger surface, as an HEIF grid's tile in its
/// canvas (RgfxMetalDevice+Video.mm). The caller keeps the origin's bytes across a multiple of
/// the alignment the host's linear textures take (the plane's rows are); nil where the texture
/// would reach past its plane's rows or elements, or the plane's elements are not pixels. Its own
/// CPU access (getBytes, replaceRegion) reads and writes nothing.
+ (nullable instancetype)textureWithDescriptor:(MTLTextureDescriptor *)descriptor
                                     iosurface:(IOSurfaceRef)surface
                                         plane:(NSUInteger)plane
                                        origin:(MTLOrigin)origin
                                        region:(RgfxRegion *)region
                                        device:(nullable id<MTLDevice>)device
                                    serializer:(RgfxSerializer *)serializer;

@end

NS_ASSUME_NONNULL_END

/// Before a commit: syncs each copy the command buffer of `enc` uses or writes (as its encoders
/// named them, rgfxIOSurfaceNoteUse and rgfxIOSurfaceNoteWrite) from the surface's pages, where
/// the surface's seed moved since any holder synced it (a view in place needs none). A copy the
/// buffer does not name is synced at the commit of one that does: Metal has a function reach a
/// resource only through a binding, an attachment, a blit, or useResource for one an argument
/// buffer names, and each of those names it.
void rgfxIOSurfaceSyncBeforeCommit(RgfxEncoder *_Nonnull enc);

/// While one lives on a thread, a surface's block that the thread's texture makes in place asks
/// no device cache: for a video decoder's outputs where the host's video caps do not say it lands
/// pictures in device caches (RGFX_VIDEO_CAPS_OUTPUT_CACHED), so the host's decode writes the
/// surface's pages. A block another holder made keeps the mode that holder's ask gave it.
class RgfxAskUncachedScope {
public:
	RgfxAskUncachedScope();
	~RgfxAskUncachedScope();
	RgfxAskUncachedScope(const RgfxAskUncachedScope &) = delete;
	RgfxAskUncachedScope &operator=(const RgfxAskUncachedScope &) = delete;
};

/// Lets go of every view the keep holds (the views in place kept past their textures,
/// RgfxIOSurface.mm): at the process's memory pressure, and for the tests.
void rgfxIOSurfaceKeepFlush(void);

/// Lets go of every view the keep holds, the least recently textured first, those another
/// process's hold of their surface keeps among them: the region's reclaim, where the kext has no
/// room for a block (RgfxRegion::setReclaim). True where it let any go.
bool rgfxIOSurfaceKeepReclaim(void);

/// Lets go of the views the keep holds that are due to go at `nowNs` (CLOCK_UPTIME_RAW): the
/// keep's own sweep, and the tests' passing of its time.
void rgfxIOSurfaceKeepSweepAt(uint64_t nowNs);

/// Whether the keep keeps a cached block's view while another process holds its surface, as
/// WindowServer's does (RgfxIOSurface.mm): for the tests.
void rgfxIOSurfaceKeepCountClients(bool on);

/// Records on `enc` that its command buffer writes `texture`, when `texture` is one
/// over an IOSurface or a view of one, through any views (otherwise nothing), and, where its
/// view is a copy, that its commit syncs it (rgfxIOSurfaceSyncBeforeCommit).
void rgfxIOSurfaceNoteWrite(RgfxEncoder *_Nullable enc, __unsafe_unretained id _Nullable texture);

/// Records on `enc` that its command buffer uses `resource`, when `resource` is a texture over an
/// IOSurface or a view of one: once a command buffer where its shared block is cached (a device
/// cache on the host, rgfxIOSurfaceNameUses) or the process names its buffers' order
/// (rgfxIOSurfaceSetOrder), and where its view is a copy, that its commit syncs it
/// (rgfxIOSurfaceSyncBeforeCommit); otherwise nothing.
void rgfxIOSurfaceNoteUse(RgfxEncoder *_Nullable enc, __unsafe_unretained id _Nullable resource);

/// At a commit: names in `enc`'s op stream, before its first encoder, each cached surface the
/// buffer uses (rgfxIOSurfaceNoteUse) or writes (`writes`, taken from `enc`), once each
/// (RGFX_OP_SURFACE_USE): its block, its CPU generation as the kext left it at the surface's last
/// read-write lock, and whether the buffer writes it. The host fills the surface's device cache
/// from its pages before the buffer's work where the cache holds an older generation. The answer
/// is every surface the buffer uses, taken from `enc`, for its post to name its order by
/// (rgfxIOSurfaceNameOrder).
std::vector<std::shared_ptr<void>> rgfxIOSurfaceNameUses(
    RgfxEncoder *_Nonnull enc, const std::vector<std::shared_ptr<void>> &writes);

/// At the post of the command buffer of queue `queueObjId`, just before it goes on its ring:
/// where the process names its buffers' order (rgfxIOSurfaceSetOrder) and the buffer uses or
/// writes a surface (`uses`, rgfxIOSurfaceNameUses' answer, and `writes`), names in `enc`'s op
/// stream, once (RGFX_OP_SURFACE_ORDER), the buffers of other queues it runs after: each queue
/// once, at its highest seqno, of the writers of every surface it uses, and the readers of every
/// surface it writes, as the blocks hold them now. The post is where the kext gets the buffer, as
/// a Mac's kernel gets it at its submission, which waits for every buffer enqueued before it on
/// its queue to be committed: a buffer committed behind one not yet committed runs after the
/// accesses of other queues posted meanwhile, as it would on a Mac. The answer is the surfaces it
/// only reads, whose readers it raises once posted (rgfxIOSurfaceNoteRead); empty without the
/// order.
std::vector<std::shared_ptr<void>> rgfxIOSurfaceNameOrder(
    RgfxEncoder *_Nonnull enc, const std::vector<std::shared_ptr<void>> &writes,
    const std::vector<std::shared_ptr<void>> &uses, uint64_t queueObjId);

/// Whether the process names its command buffers' order over IOSurfaces: where the device's
/// feature word holds RGFX_FEATURE_SURFACE_ORDER, as the device is made.
void rgfxIOSurfaceSetOrder(bool on);

/// The textures over IOSurfaces, or views of them, bound at the indexes of one stage of an
/// encoder, so that each draw or dispatch names as written those its function writes there. A
/// binding of any other texture, or of nil, drops the index's.
class RgfxSurfaceBindings {
public:
	/// `texture` bound at `index`, `overSurface` where it is a texture over an IOSurface or a
	/// view of one (as rgfxBinds answers it).
	void bind(NSUInteger index, __unsafe_unretained id _Nullable texture, bool overSurface)
	{
		if (m_bound.empty() && !overSurface)
			return; // the common case: no surface bound, and none binding
		rebind(index, texture, overSurface);
	}
	/// `resource` made resident (useResource:usage:) with no Write in its usage: a texture over an
	/// IOSurface or a resource over app memory, which a function may yet write through an
	/// argument buffer (noteWrites).
	void resident(__unsafe_unretained id _Nullable resource);
	/// At a draw or dispatch of the function whose arguments are `fn`, records on `enc` as
	/// written, once each while it stays bound or resident:
	/// - each bound texture over an IOSurface the function writes (rgfxAirWritesTexture), and,
	///   where `fn` is null, its arguments unknown, or has a written texture argument at an index
	///   function constants set (rgfxAirWritesTextureAnywhere), each whose usage has ShaderWrite
	///   or is MTLTextureUsageUnknown, which allows every use;
	/// - each resident texture (rgfxIOSurfaceNoteWrite, and its copy of app memory) where `fn` is
	///   null or one of its argument buffers may write a texture, and each resident buffer over
	///   app memory where `fn` is null or has an argument buffer. Metal takes the usage
	///   useResource gave for the app's promise, but a discrete Mac's driver lands a write the
	///   promise left out, and the app's pages then hold it: so does a surface's here.
	void noteWrites(RgfxEncoder *_Nullable enc, const RgfxAirFunction *_Nullable fn)
	{
		// What a draw notes follows from the function and what is bound or resident alone:
		// once both are as they were at the last draw, every write is noted already.
		if (m_changed || fn != m_lastFn)
			noteChangedWrites(enc, fn);
	}
	/// The encoder's pipeline changed: the next draw looks at every binding again, the
	/// function's arguments being another's.
	void pipelineChanged() { m_changed = true; }
	/// Drops every binding and resident resource, as the encoder ends.
	void clear()
	{
		m_bound.clear();
		m_resident.clear();
		m_changed = true;
	}

private:
	void rebind(NSUInteger index, __unsafe_unretained id _Nullable texture, bool overSurface);
	void noteChangedWrites(RgfxEncoder *_Nullable enc, const RgfxAirFunction *_Nullable fn);

	struct Bound {
		NSUInteger index;
		id _Nonnull texture;
		bool noted;
	};
	struct Resident {
		id _Nonnull resource;
		bool texture;
		bool noted;
	};
	std::vector<Bound> m_bound;
	std::vector<Resident> m_resident;
	// A binding, a resident resource or the pipeline changed since the last draw's notes, which
	// were for the function `m_lastFn`.
	bool m_changed = true;
	const RgfxAirFunction *_Nullable m_lastFn = nullptr;
};

/// Where the writes `writes` (taken from a command buffer's encoder) are named to the kext: the
/// region of their surfaces' shared blocks, null where none is of a shared block. A buffer with
/// one names them after it is posted (rgfxIOSurfaceWritesPosted), and retires them at its
/// completion (rgfxIOSurfaceRetire) whether or not the naming came first.
RgfxRegion *_Nullable rgfxIOSurfaceWritesRegion(const std::vector<std::shared_ptr<void>> &writes);

/// Once a command buffer is posted at seqno `seqno` on queue `queueObjId`, and, held, scheduled
/// (RgfxCmdBuffer's rgfxNameAccesses): names the surfaces
/// it writes, those with shared blocks in `region`, to the kext once each. A CPU lock of one of
/// them, in any process, then waits until the buffer's results are in its pages, as Metal's
/// does once a buffer is submitted. The kext marks each surface the GPU's, which moves its seed,
/// and each copy records that seed as its own, so no holder takes the pages over the GPU's
/// results at its next commit.
void rgfxIOSurfaceWritesPosted(RgfxRegion *_Nonnull region,
                               const std::vector<std::shared_ptr<void>> &writes,
                               uint64_t queueObjId, uint64_t seqno);

/// At the completion of queue `queueObjId`'s buffer at `seqno`, once its results are in the
/// surfaces' pages (after rgfxIOSurfaceWriteBack): its writes, and those of the queue's buffers
/// before it, retire, and the CPU locks waiting for them go on.
void rgfxIOSurfaceRetire(RgfxRegion *_Nonnull region, uint64_t queueObjId, uint64_t seqno);

/// Once a command buffer is posted at seqno `seqno` on queue `queue` (whose id is `queueObjId`),
/// and, held, scheduled: names it among the readers of each view in `reads`
/// (rgfxIOSurfaceNameOrder's answer), so a later post of another queue's buffer that writes the
/// surface runs after it.
void rgfxIOSurfaceNoteRead(const std::vector<std::shared_ptr<void>> &reads, id _Nonnull queue,
                           uint64_t queueObjId, uint64_t seqno);

/// Once the host has finished queue `queueObjId`'s buffer at `seqno`, and its accesses are named
/// (rgfxIOSurfaceNoteCommitted, rgfxIOSurfaceNoteRead): empties the queue's slots, of the writers
/// and the readers of each view in `writes` and `reads`, that still name that buffer. A buffer
/// that named it would wait for nothing, and the slot is free for a queue of any process.
void rgfxIOSurfaceNoteFinished(const std::vector<std::shared_ptr<void>> &writes,
                               const std::vector<std::shared_ptr<void>> &reads,
                               uint64_t queueObjId, uint64_t seqno);

/// Once a command buffer is posted at seqno `seqno` on queue `queue` (whose id is
/// `queueObjId`), and, held, scheduled: names it among the writers of each copy in `writes`
/// (taken from its encoder), once each, so a present of the surface after this waits for it.
void rgfxIOSurfaceNoteCommitted(const std::vector<std::shared_ptr<void>> &writes, id _Nonnull queue,
                                uint64_t queueObjId, uint64_t seqno);

/// At a command buffer's completion: writes each copy in `writes` (taken from its
/// encoder) back to its surface's pages, once each (a view in place needs none). It takes no
/// lock of the surface: a lock waits for this buffer's writes, which retire only after it, and
/// a GPU's write leaves the seed where the commit moved it.
void rgfxIOSurfaceWriteBack(const std::vector<std::shared_ptr<void>> &writes);

/// Whether every view in `writes` (taken from a command buffer's encoder) is in place, none a
/// copy rgfxIOSurfaceWriteBack writes back: the buffer's results are then in the surfaces' pages
/// once the device has completed it.
bool rgfxIOSurfaceWritesInPlace(const std::vector<std::shared_ptr<void>> &writes);

/// How many textures over `surface` were answered nil for `cause` ("arguments", "plane",
/// "descriptor", "plane shape", "region", "buffer", "resource id", "texture"), and the "RGFX:"
/// line logged for the first, without the prefix (for the tests): 0 and empty where none was.
uint64_t rgfxIOSurfaceNilLog(IOSurfaceRef _Nullable surface, const char *_Nonnull cause,
                             std::string *_Nonnull line);

#endif // RGFX_IOSURFACE_H_
