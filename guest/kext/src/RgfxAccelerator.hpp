/*
 * RgfxAccelerator: the reims-graphics device's Metal accelerator for x86-64 macOS
 * guests (docs/graphics-kext/design.md, "Stage 3 guest stack"). It is a SECOND
 * IOService on the same PCI device as the Stage-2 framebuffer (ReimsGraphics):
 * the framebuffer matches in category IOFramebuffer, this matches in category
 * IOAccelerator, and both are children of the one IOPCIDevice (the notes require
 * the accelerator and the framebuffer to share a provider).
 *
 * What it is for: Metal does not scan /Library/GPUBundles. It loads our guest GPU
 * bundle from two properties on THIS service's IOKit personality — MetalPluginName
 * (the bundle path) and MetalPluginClassName (the Obj-C class Metal instantiates as
 * the MTLDevice, RgfxMetalDevice) — which it finds because the service matches
 * IOServiceMatching("IOAccelerator") under IOMatchCategory IOAccelerator. Those two
 * keys live in src/Info.plist's accelerator personality (stage3-apple-metal-iface.md
 * section 1). This service then exposes, over OUR protocol (not Apple's IOGPU
 * user-client ABI), the two things the bundle's transport needs:
 *   - the process's part of the device's BAR-2 region, mapped into the Metal client's
 *     task in its window (RgfxAcceleratorUserClient.hpp), so the bundle's RgfxRegion /
 *     RgfxRingWriter write into real device memory, and
 *   - the doorbell: a write of a ring id to RGFX_REG_DOORBELL in BAR 0, to wake the
 *     host's consumer of a queue's submission ring (ringDoorbell, reached from the
 *     user client's external method).
 *
 * ========================= BOOT-GATED DECISION (UNKNOWN #1) =========================
 * BASE CLASS / KERNEL FAMILY. stage3-apple-metal-iface.md section 3 and
 * docs/graphics-kext/design.md Decided 15 record that our guest runs the ported
 * IOGPUFamily stack, that IOAcceleratorFamily2 cannot start in a kernel where
 * IOGPUFamily runs, and that our accelerator therefore aligns with IOGPU. The exact
 * IOGPU kernel base class (an IOGPU, itself an IOAccelerator) and its submission /
 * allocation user-client ABI are Apple-private and are NOT in the macOS SDK or the
 * KDK, so they cannot be subclassed off the guest — only a guest boot + IORegistry +
 * user-client trace settles them (notes UNKNOWN #1; design.md Open).
 *
 * This scaffold subclasses the base the SDK DOES expose, IOAccelerator
 * (IOKit/graphics/IOAccelerator.h), which is what Metal's IOServiceMatching(
 * "IOAccelerator") enumeration matches. That is enough to publish the Metal
 * discovery keys and to stand up OUR own region-map + doorbell user client, which is
 * all that compiles offline. On the guest the base very likely becomes the IOGPU
 * family class; the Metal keys, the PCI match, and our user-client protocol below are
 * expected to carry over unchanged. DO NOT fabricate the IOGPU base or its selectors
 * here.
 * ===================================================================================
 *
 * Our own code; it names only SDK-public IOKit classes.
 */

#ifndef RGFX_ACCELERATOR_HPP_
#define RGFX_ACCELERATOR_HPP_

#include <IOKit/IOLib.h>
#include <IOKit/graphics/IOAccelerator.h>
#include <IOKit/pci/IOPCIDevice.h>

#include "RgfxBlockTable.hpp"
#include "RgfxDeviceName.hpp"
#include "RgfxMemorySize.hpp"
#include "RgfxUserClientABI.hpp"
#include "reims_graphics_wire.h"

// OUR user-client protocol (not Apple's IOGPU ABI). The bundle's wire handshake opens the
// accelerator with IOServiceOpen(acceleratorPort, task, kRgfxAccelUserClientType, &connect),
// reads the device info (kRgfxAccelMethodDeviceInfo), opens its context and window
// (kRgfxAccelMethodOpenContext), and rings a ring's doorbell with
// IOConnectCallScalarMethod(kRgfxAccelMethodDoorbell, {ringId}).

// Our own user-client type (kRgfxAccelUserClientType), the selectors of our own protocol and
// the shared block header's CPU generation word are the bundle's too: RgfxUserClientABI.hpp.
// The IOAccel surface client, the public kIOAccelSurfaceClientType (RgfxSurfaceUserClient).
static const UInt32 kRgfxSurfaceUserClientType = 0;
// The type Apple's IOGPUMetalDevice base opens to build its device
// (RgfxAcceleratorUserClient::deviceMethod).
static const UInt32 kRgfxIOGPUDeviceUserClientType = 5;

// clientMemoryForType ids: the whole BAR-2 device region, which is refused
// (kIOReturnUnsupported): the region holds every process's blocks, the rings and the device
// page, and a process maps only its own part of it, in its window.
enum {
	kRgfxAccelMemoryRegion = 0,
};

// The doorbell's dispatch table (kRgfxAccelMethodDoorbell, RgfxUserClientABI.hpp): its one entry.
enum {
	kRgfxAccelMethodCount = 1,
};

// The kext's own features (kRgfxKextFeature*, RgfxUserClientABI.hpp) this kext carries out, the
// second word kRgfxAccelMethodFeatures answers: PagesBlock, Submit's deferred flag and
// TakeAnswers, OpenContext's span, shared events' ports, and Alloc's write-combined part. Each
// bit is set by the change that lands its call.
constexpr uint64_t kRgfxKextFeatureWord = kRgfxKextFeaturePagesBlock |
                                          kRgfxKextFeatureDeferredAnswers |
                                          kRgfxKextFeatureContextSpan |
                                          kRgfxKextFeatureSharedEvents |
                                          kRgfxKextFeatureWriteCombined;

class RgfxAccelerator;
class RgfxAcceleratorUserClient;
struct RgfxSurfaceEntry;
// How a SURFACE_PAGES naming a block's pages ended (RgfxSurfaceBlocks.hpp).
enum RgfxPagesNaming : uint32_t;
class RgfxSurfaceUserClient;
class ReimsGraphics;

// A surface's shared block goes (RgfxAcceleratorContexts.cpp): its device cache's pending writes
// go, waking the locks waiting on them, and its device cache and IOSurface are released
// (RgfxAcceleratorLocks.cpp), the cache's first. Either may be null. A cached block's surface
// stays cached until its BLOCK_RELEASE is answered where `releasing` (rgfxSurfaceCacheReleased),
// its locks waiting for the answer and its cache, its surface and its memory's wiring held until
// then; else it goes, or, where a lock is naming a deferred block's pages, goes once that attach
// is answered, which lets it go: the call waits for no host. A surface that goes takes the wiring
// of its memory with it, unless `wire` is given: that wiring is then the caller's (`*wire`, null
// where the surface was not cached, or its memory was never wired), to hand to a block the device
// may still map in place, or to unwire. The block is the one at `offset`: the surface's cached
// state is its cached block's alone, which another block of the surface leaves as it is
// (rgfxCacheBlockLetGo). `blockWired`: the block keeps a wire of the surface's pages of its own
// while it is in the table; decided in the same hold of the locks' state as the attach in flight,
// a block kept for good without one has that attach keep its pages' wiring for good
// (rgfxCacheLetGo).
void rgfxSurfaceLocksForget(void *cache, void *surface, UInt64 offset, bool releasing,
                            bool blockWired, void **wire = nullptr);
// The surface `surface` of device cache `cache` (rgfxSurfaceLocksAttach), IOSurfaceID
// `surfaceId`, has its block at `offset` cached (RgfxSurfaceCaches.hpp), the block's first page
// mapped at `header`, which the kext keeps (and releases once the surface is cached no more) to
// write its CPU generation at `genOffset`, 0 from here, and its clients at
// kRgfxSurfaceClientsOffset, its count now from here (rgfxCacheNoteClients). The surface's
// memory is wired until then. Where the surface's previous block is cached still, a let-go's
// attach of its pages in flight, waits for the answer first, as a lock does; a previous block's
// release unanswered is the ask's to wait for (rgfxSurfaceCacheAwaitWriteBack), and goes on at
// once. False, `header` the caller's and the table as it was, where it cannot be.
bool rgfxSurfaceCacheAdd(void *cache, void *surface, UInt64 offset, uint32_t surfaceId,
                         IOMemoryMap *header, UInt32 genOffset);
// Whether the pages of IOSurface `surface` are still to take a write-back: a cached block of it
// went, and the host copies its cache into them before it answers the block's release
// (rgfxCacheWritingBack). Takes the locks' lock alone, which may be under the contexts' lock.
bool rgfxSurfaceCacheWritingBack(OSObject *surface);
// Waits, holding no lock of the kext's, until the pages of IOSurface `surface` took their
// write-back, or until `deadline` (mach absolute time).
void rgfxSurfaceCacheAwaitWriteBack(OSObject *surface, uint64_t deadline);
// The cached block at `offset` was held again by a process, which holds its surface: the
// surface's clients written into the block's header anew (rgfxCacheNoteClients). Takes the
// locks' lock alone.
void rgfxSurfaceCacheNoteClients(UInt64 offset);
// The cached surface of device cache `cache` has its block dormant (RgfxSurfaceBlocks.hpp), or
// held again: false where it is not cached, made empty (it does not sleep), or, held again,
// where it is to be let go. `*purgeable` where the surface is volatile.
bool rgfxSurfaceCacheSleep(void *cache, bool *purgeable);
bool rgfxSurfaceCacheWake(void *cache);
// The dormant cached block at `offset` is to be let go: its surface had no client left at its
// sleep (rgfxCacheUnclaimed). Takes the locks' lock alone.
void rgfxSurfaceCacheUnclaimed(UInt64 offset);
// Why the dormant cached block at `offset` is to be let go (a RgfxDormantCause), else
// kRgfxCacheKept; `*written` where a buffer wrote its cache since its last copy, `*purgeable`
// where its surface is volatile.
uint32_t rgfxSurfaceCacheLetGoOf(UInt64 offset, bool *written, bool *purgeable);
// How many surfaces' blocks went uncached as the table of cached surfaces had no room
// (kRgfxCachedMaxSurfaces). Takes the locks' lock alone.
UInt64 rgfxSurfaceCacheNoRoom();
// How many releases posted unheld (RGFX_BLOCK_RELEASE_UNHELD) and answered OK had their surface
// taken again by a process before the answer (rgfxCacheRetaken). Takes the locks' lock alone.
UInt64 rgfxSurfaceCacheRetaken();
// The flags of the BLOCK_RELEASE of the cached block at `offset`, its surface `surface`
// (rgfxSurfaceCacheReleasingSurface's), whose counts are read out of every lock and decided on
// under the locks' lock (rgfxCacheReleaseFlags). Takes the locks' lock alone.
uint32_t rgfxSurfaceCacheReleaseFlags(UInt64 offset, OSObject *surface);
// The surface of the cached block at `offset` whose BLOCK_RELEASE is posted (rgfxSurfaceLocksForget
// with `releasing`), retained for the caller, who releases it; `*written` where a buffer wrote
// its cache since its last copy. Null where no cached surface is releasing the block.
OSObject *rgfxSurfaceCacheReleasingSurface(UInt64 offset, bool *written);
// What a cached surface held for its block's release, taken out by rgfxSurfaceCacheReleased: its
// header's mapping, its device cache's reference, the surface's and its memory, and whether that
// memory is wired (a deferred block's whose pages were never named is not).
struct RgfxReleasedSurface {
	void *map;
	void *cache;
	void *surface;
	void *wire;
	bool wired;
};
// The BLOCK_RELEASE of the block at `offset` was answered: a cached surface releasing it is
// cached no more and its locks go on, what it held in `*gone`, for the caller to let go with
// rgfxSurfaceCacheLetGo out of the contexts' lock (all null where none was releasing the block).
// Takes the locks' lock alone, so the caller may hold the contexts' lock.
void rgfxSurfaceCacheReleased(UInt64 offset, RgfxReleasedSurface *gone);
// Lets go of what rgfxSurfaceCacheReleased took out: the mapping, the device cache's reference,
// then the surface's, then the memory's wiring. Under none of the kext's locks: a release may
// free the cache or the surface, which call back into the kext.
void rgfxSurfaceCacheLetGo(const RgfxReleasedSurface &gone);
// The CPU generation of the cached surface of IOSurfaceID `surfaceId`, for a present of it;
// false where it is not cached.
bool rgfxSurfaceCacheGeneration(uint32_t surfaceId, uint64_t *gen);
// Makes the device cache of the surface of ID `surfaceId` that `task` may see, whose memory holds
// the physical byte `phys` (a shared block's first), with the kext's control function: its
// IOSurface and its cache, each retained, or false. Under the contexts' lock.
bool rgfxSurfaceLocksAttach(RgfxAccelerator *accel, task_t task, uint32_t surfaceId, UInt64 phys,
                            void **surface, void **cache);
// The device raised its vector: the locks waiting for a surface look at the queues' completions
// again.
void rgfxSurfaceLocksWake();
// The IOSurface of ID `surfaceId` that `task` may see, retained, or null. Under the contexts'
// lock.
OSObject *rgfxSurfaceLocksLookup(RgfxAccelerator *accel, task_t task, uint32_t surfaceId);
// Whether `task` maps `surface`'s memory from its first byte at `address`, `length` bytes of it
// or more: IOSurface's own mapping of it for the task's client, found, never made, and not wired.
bool rgfxSurfaceLocksMapsAt(OSObject *surface, task_t task, UInt64 address, UInt64 length);
// Whether `surface`'s memory, wired, is the `count` runs of `runs` from its first byte, over
// `length` bytes.
bool rgfxSurfaceLocksStartsAt(OSObject *surface, const RgfxPageRun *runs, UInt32 count,
                              UInt64 length);
// Whether `surface` reads fresh, to look at: its seed 0, none of its pages resident and its
// memory holding `length` bytes. Racy: the deferral's own read is rgfxSurfaceLocksDefer's.
bool rgfxSurfaceLocksFresh(OSObject *surface, UInt64 length);
// What rgfxSurfaceLocksDefer found.
enum RgfxDeferAnswer {
	// No device cache, or no state for it: the block is neither deferred nor cached.
	kRgfxDeferNoCache,
	// The cache made, the surface not fresh: the block's pages are named now, as any cached
	// block's, and rgfxSurfaceCacheSettle says so.
	kRgfxDeferStale,
	// The cache made, the surface fresh: the host is asked to defer the block, and
	// rgfxSurfaceCacheSettle gives its answer, locks waiting for it meanwhile.
	kRgfxDeferFresh,
};
// For the new deferred block at `offset` of surface `surface` (retained by its entry), IOSurfaceID
// `surfaceId`, made by context `context`, its holders mapping `size` bytes and the plane `length`:
// makes the surface's device cache with the kext's control function, and records the block
// cached and gating (RgfxSurfaceCaches.hpp), the block's first page mapped at `header`, which the
// table keeps, its memory retained but not wired; then reads whether the surface is fresh, under
// a read-only lock of it. `*cache` is the cache, retained, for the entry, where it answers other
// than kRgfxDeferNoCache, which leaves `header` the caller's. Under no lock of the kext's.
RgfxDeferAnswer rgfxSurfaceLocksDefer(RgfxAccelerator *accel, OSObject *surface,
                                      uint32_t surfaceId, UInt64 offset, UInt64 size,
                                      UInt64 length, UInt32 context, IOMemoryMap *header,
                                      void **cache);
// The host answered the ask of the block of device cache `cache`: deferred where `deferred`,
// else its pages are named, its surface's memory wired. Locks waiting for the answer go on.
void rgfxSurfaceCacheSettle(void *cache, bool deferred);
// What rgfxSurfaceCacheReleaseAttach did.
enum RgfxReleaseAttach {
	// Nothing: not a deferred block, or one whose pages were named before.
	kRgfxReleaseAsIs,
	// A deferred block going with its pages never named: nothing reads them.
	kRgfxReleaseUnattached,
	// Its pages were named now, or asked and not answered: the host may map them until the
	// block's release is answered OK, so the block takes a wire of them
	// (rgfxSurfaceCacheHostPages).
	kRgfxReleaseAttached,
	// Nothing yet, asked with `name` false: its pages are to be named first, or a lock is naming
	// them, each waiting on the host's answer. The release thread hands the release to the
	// naming worker, which asks again with `name` true (RgfxReleaseTable.hpp).
	kRgfxReleaseNameFirst,
};
// The BLOCK_RELEASE of the cached block at `offset`, of surface `surface` (retained by the
// caller), is to be posted: a deferred block's pages are named first where the GPU wrote it and
// the surface lives past the kext's own references (rgfxCacheReleaseNext), and a lock's naming
// in flight is waited for, where `name`; else neither is, and kRgfxReleaseNameFirst says one is
// owed. Under no lock of the kext's.
RgfxReleaseAttach rgfxSurfaceCacheReleaseAttach(RgfxAccelerator *accel, UInt64 offset,
                                                OSObject *surface, bool name);
// The pages of the deferred block of device cache `cache`, or else of the block at `offset`,
// where the host may map them (rgfxCacheHostPages): its surface's memory, retained and wired once
// more for the block to hold until its release is answered OK; else null. Takes the locks' lock
// alone, and wires out of it.
IOMemoryDescriptor *rgfxSurfaceCacheHostPages(void *cache, UInt64 offset);

// A command queue's completion ring the accelerator watches for its client
// (RgfxAcceleratorContexts.cpp): the ring's header, mapped into the kernel, and the producer
// count last seen there.
struct RgfxWatchedRing {
	RgfxAcceleratorUserClient *client;
	UInt64 objId;
	IOMemoryMap *map;
	volatile UInt8 *header;
	UInt64 lastProducer;
};

class RgfxAccelerator : public IOAccelerator {
	OSDeclareDefaultStructors(RgfxAccelerator);

public:
	// IOService
	bool start(IOService *provider) override;
	void stop(IOService *provider) override;

	// Our own user client: the region-map + doorbell transport the bundle opens.
	IOReturn newUserClient(task_t owningTask, void *securityID, UInt32 type,
	                       OSDictionary *properties, IOUserClient **handler) override;

	// Reached from the user client. The BAR-2 device region to map into the Metal
	// client's task (retained for the caller), and the doorbell write that wakes the
	// host's consumer of ring `ringId`.
	IODeviceMemory *copyRegion() const;
	// The region again, as a descriptor of its own that only write-combined spans are mapped
	// from (retained for the caller; null before the contexts' start, or where it could not be
	// made). A mapping's cache mode is set on its descriptor's one VM object for every page of
	// it (xnu's IOKit hands the mode to mach_make_memory_entry with MAP_MEM_ONLY, which changes
	// the object's WIMG bits), and the fault path maps a physically contiguous object's pages in
	// the object's mode (vm_fault's pmap_map_block_addr; both read from the public xnu source).
	// So a write-combined span mapped from the region's own descriptor would leave each region
	// page faulted after it write-combined, and each write-back span after it would walk the
	// object's resident pages to take the mode back. Measured on the Radeon 780M host:
	// draws.10k's encode took 7.7 ms with the spans write-back, 1.05-1.3 ms write-combined.
	// Defined with the contexts' state (RgfxAcceleratorContexts.cpp).
	IODeviceMemory *copyWriteCombinedRegion() const;
	// The region's first `size` bytes from RGFX_DEVICE_INFO_OFFSET (the device info, and the
	// video caps after it), mapped read-only into the kernel: the map, which the caller
	// releases, with *info its first byte; null where it cannot be mapped.
	IOMemoryMap *mapDeviceInfo(UInt64 size, const void **info) const;
	void ringDoorbell(UInt32 ringId);
	// BAR 2's size, as the device's REGION_SIZE registers give it (no more than the BAR's).
	UInt64 getRegionSize() const { return regionSize; }
	// The GPU memory the guest is told of, from the host's device info (RgfxMemorySize.hpp):
	// false where the region is not mapped.
	bool readMemorySize(RgfxMemorySize *out) const;
	// The GPU's `model` bytes, from the host's device name (RgfxDeviceName.hpp): how many, 0
	// where the host gave no name or the region is not mapped.
	size_t readDeviceModel(uint8_t (&out)[kRgfxDeviceModelMax]) const;

	// Shared events' ports (RgfxSharedEventUserClient.hpp): the client of ours that holds context
	// `ctx` open for task `caller`, retained for the caller, or null where none does; a name for an
	// export, unique from the kext's start, never 0; and a name's SHARED_EVENT_UNNAME, posted once
	// its port has gone (or an export posted it unanswered), waiting for no answer.
	RgfxAcceleratorUserClient *copyContextClient(task_t caller, UInt64 ctx);
	UInt64 takeSharedEventName();
	void unnameSharedEvent(UInt64 name);

	// The framebuffer on our PCI device (ReimsGraphics), retained for the caller, or nullptr
	// before it has started. The first lookup that finds it names this accelerator on it
	// (IOAccelTypes), as a display pipe's accelerator does.
	ReimsGraphics *copyFramebuffer();

	// The bundle (RgfxAccelerator.cpp, "The bundle"): the accelerator registers only once the
	// bundle MetalPluginName names is on disk, and opens no user client before. A check from
	// start repeats until the bundle is there; the framebuffer's first server connection,
	// WindowServer's, settles it with a last check of its own, before IOFramebuffer makes the
	// connection, and no check runs after. A bundle installed later is used from the next boot.
	void settleBundle();

	// The processes' contexts (RgfxAcceleratorContexts.cpp). A context number is handed out once
	// (from 1); a run of units of the GPU-address space (RgfxContextSlots.hpp), `*units` from
	// `*first`, is held until the close: the units a context holds, or RGFX_CONTEXT_VA_SPAN's
	// for a bundle that takes no span from the kext (`oldSpan`). WindowServer's
	// (`windowServer`) may take the last units free. kIOReturnNoResources where no run is free.
	IOReturn openContext(UInt32 *context, bool oldSpan, bool windowServer, UInt32 *first,
	                     UInt32 *units);
	// How a context's close ended (closeContext).
	enum CloseAnswer {
		// Answered OK in time: the device holds nothing of the context, and every command
		// buffer of it has finished. Its units are given back, and the caller lets go of the
		// rest (freeIdChunks, freeBlocks, dropSurfaceBlocks closed).
		kCloseAnswered,
		// Refused, or never posted: the device may hold all of it for good, its units too, so
		// the caller keeps the rest from use (keepIdChunks, keepBlocks, dropSurfaceBlocks not
		// closed).
		kCloseRefused,
		// Not answered in kCloseAnswerMs: the caller hands the rest to the close (parkClose
		// with `*late`), whose answer, however late, lets go of all of it as one in time does.
		kCloseLate,
	};
	CloseAnswer closeContext(UInt32 context, UInt32 first, UInt32 units, int32_t *late);
	// A late close's holdings: the client's resource-id chunks, blocks and holds of shared
	// surface blocks are the close's own (its token's, RgfxCloseTable.hpp) from here on, none of
	// them anyone else's, until its answer lets go of them on the release thread. Never answered,
	// they stay the device's for good, as its units do.
	void parkClose(const void *client, int32_t late);
	// Chunks of the texture resource ids, each owned by `owner` for context `context` and posted
	// to the device (RESOURCE_ID_SMALL_CHUNK, or RESOURCE_ID_CHUNK where the guest did not take
	// RGFX_FEATURE_SMALL_ID_CHUNKS) before takeIdChunk returns its first id; the owner's are freed
	// after its context's close is answered.
	IOReturn takeIdChunk(const void *owner, UInt32 context, UInt64 *firstId);
	void freeIdChunks(const void *owner);
	// The owner's chunks are kept from use for good: its close was refused.
	void keepIdChunks(const void *owner);
	// Blocks of the region past RGFX_CONTEXT_ALLOCATIONS_OFFSET (RgfxBlockTable.hpp), each owned
	// by `owner` of context `context`. freeBlock returns at once; the block is given out again
	// once the device answers its BLOCK_RELEASE OK, posted from a thread of the kext's own, and
	// is kIOReturnBusy while a create naming it is in flight (pinRange). freeBlocks frees the
	// owner's blocks after its close is answered; keepBlocks keeps them from use for good after
	// a close that was not.
	// allocBlock maps the block into the client's window before it returns, its bytes from
	// `writeCombinedFrom` on write-combined (none where that is past the block); freeBlock takes
	// it back from the window before the block is freed.
	IOReturn allocBlock(RgfxAcceleratorUserClient *owner, UInt32 context, UInt64 size,
	                    UInt64 align, UInt64 writeCombinedFrom, UInt64 *offset);
	IOReturn freeBlock(RgfxAcceleratorUserClient *owner, UInt64 offset);
	// A pages block (kRgfxAccelMethodPagesBlock) of `owner` of context `context` over the
	// `length` bytes at `address` in `task`, `writable` there: the block's offset and size. The
	// block is never in the owner's window; freeBlock frees it, and its pages are unwired once
	// its release is answered OK, or once freeBlocks frees it after the owner's close.
	IOReturn pagesBlock(RgfxAcceleratorUserClient *owner, UInt32 context, task_t task,
	                    UInt64 address, UInt64 length, bool writable, UInt64 *offset,
	                    UInt64 *size);
	void freeBlocks(const void *owner);
	// Before the owner's close is posted: none of its blocks is released from here on, a freed
	// or a refused one included, so no release of its comes behind the close, which the device
	// would take with the context gone and answer without waiting for its buffers. The close's
	// answer lets go of them (freeBlocks), or hands them on (parkClose).
	void holdBlocks(const void *owner);
	void keepBlocks(const void *owner);
	// Pins the owner's live block that holds the `length` bytes at `offset`, for a create that
	// names them, until unpinBlock: false where no block of the owner's does.
	bool pinRange(const void *owner, UInt64 offset, UInt64 length, RgfxBlockPin *pin);
	void unpinBlock(const RgfxBlockPin &pin);
	// The shared blocks of IOSurface planes (RgfxSurfaceBlocks.hpp), for our user client `client`
	// of context `context` in `task`: the block of the plane of `length` bytes at `address` in
	// the task, of `size` bytes, a copy block or one in place (`inPlace`,
	// kRgfxAccelMethodSurfaceBlock), and letting go of one ask of it. dropSurfaceBlocks lets go of every one of the client's after its close, the device
	// answered (`closed`) or not; pinRange pins a span of one the client holds. Each holder has
	// the block mapped in its window from its first ask until its hold ends, which takes it
	// back before the block can go to anyone else.
	// A block in place made for an ask with `surfaceId` and `cacheable` is cached where the
	// device has RGFX_FEATURE_SURFACE_CACHE (kRgfxAccelMethodSurfaceBlock).
	struct SurfaceBlock {
		UInt64 offset;
		UInt64 size;
		UInt64 serial;
		bool fresh;
		bool cached;
	};
	IOReturn acquireSurfaceBlock(RgfxAcceleratorUserClient *client, UInt32 context, task_t task,
	                             UInt64 address, UInt64 length, UInt64 size, bool inPlace,
	                             uint32_t surfaceId, bool cacheable, SurfaceBlock *block);
	// A dormant block's surface asked it to go (RgfxAcceleratorLocks.cpp): the release thread
	// lets it go (RgfxAcceleratorContexts.cpp). From any thread, under no lock of the kext's.
	void wakeReleases();
	// Posts again the release of every block the device refused (an object of some context lay in
	// it): a context close may have dropped that object. Called after a context closes, from the
	// releasing client's thread, under no lock of the kext's.
	void retryRefusedBlocks();
	// A CPU lock of the cached surface whose block is at `offset` follows a GPU write: the host
	// copies its device cache into its pages (SURFACE_DOWNLOAD). kIOReturnSuccess once answered
	// OK, waiting up to `timeoutMs`; kIOReturnTimeout where no answer came by then,
	// kIOReturnUnsupported where the host answered otherwise, else what the post returned
	// (RgfxAcceleratorContexts.cpp).
	IOReturn downloadSurfaceCache(UInt64 offset, UInt32 timeoutMs);
	// Names the pages of the in-place block at `offset` made by context `context`: wires the
	// first `length` bytes of the surface's memory `memory`, writes their runs past the `size`
	// bytes its holders map, and posts its SURFACE_PAGES with `flags` (RGFX_SURFACE_PAGES_*): a
	// deferred block's attach, or the pages of a block its ask meant to defer. Named, the device
	// answered OK, `memory` wired, and the runs in `*runs` (IOMalloc'd, `*count` of them, the
	// caller's) where asked for; not named, `memory` as it was; unanswered, `memory` left wired
	// once more, which the caller keeps until the block's release is answered OK, the host free to
	// take the pages yet (rgfxPagesNaming). Under no lock of the kext's.
	RgfxPagesNaming attachSurfacePages(UInt32 context, UInt64 offset, UInt64 size, UInt64 length,
	                                   IOMemoryDescriptor *memory, UInt32 flags,
	                                   RgfxPageRun **runs, UInt32 *count);
	IOReturn releaseSurfaceBlock(RgfxAcceleratorUserClient *client, UInt64 offset);
	void dropSurfaceBlocks(const void *client, bool closed);
	// The processes' object lifecycle, on the framebuffer's control ring
	// (ReimsGraphics::postLifecycle). openQueue's QUEUE_CREATE is answered once the device has
	// the queue's rings; closeContext's CONTEXT_CLOSE once the device has dropped the context's
	// objects.
	IOReturn postLifecycle(UInt32 type, const void *message, UInt32 size, UInt32 replyType,
	                       void *reply, UInt32 replyCapacity, UInt32 *replyLength, UInt64 *seqno,
	                       UInt32 timeoutMs);
	// The deferred pipeline creates (ReimsGraphics::postDeferred, takeDeferred, dropDeferred,
	// forgetDeferred).
	IOReturn postDeferred(UInt32 context, UInt32 type, const void *message, UInt32 size,
	                      UInt64 *seqno);
	IOReturn takeDeferred(UInt32 context, UInt64 waitFor, UInt32 timeoutMs, UInt8 *out,
	                      UInt32 capacity, UInt32 *length, UInt32 *count);
	void dropDeferred(UInt32 context);
	void forgetDeferred(UInt64 seqno);
	// `flags` (RGFX_QUEUE_*, granted by rgfxQueueFlagsGranted) go in the QUEUE_CREATE's tail;
	// with none, the message has no tail.
	UInt32 openQueue(const RgfxQueueCreate &create, UInt32 flags);
	// The device's feature word (kRgfxAccelMethodFeatures), read from its registers.
	uint64_t featureWord() const;
	void closeQueue(UInt64 objId);
	// The completion rings of the clients' queues: watchQueue maps queue `objId`'s ring header
	// at region offset `completeOffset`, and from then on each interrupt of the device's vector
	// that finds records the client has not taken there wakes the client's completion waiters
	// (queueInterrupt, on the framebuffer's work loop). unwatchQueue and unwatchClient stop it.
	IOReturn watchQueue(RgfxAcceleratorUserClient *client, UInt64 objId, UInt64 completeOffset);
	void unwatchQueue(RgfxAcceleratorUserClient *client, UInt64 objId);
	void unwatchClient(RgfxAcceleratorUserClient *client);
	void queueInterrupt();
	// The seqno the device last completed on queue `objId` (RgfxRingHeader::completed_seqno, 0
	// where it writes none), read from its watched completion ring; 0 where it is not watched.
	UInt64 queueCompleted(UInt64 objId);

	// CPU locks of the surfaces the GPU writes (RgfxAcceleratorLocks.cpp, RgfxSurfaceLocks.hpp):
	// queue `queue`'s buffer at `seqno`, of client `client` (context `context`, in `task`),
	// writes the surfaces of the `count` pairs (block offset, IOSurfaceID) at `pairs`; the
	// client's queue `queue` retired its writes up to `seqno`; and a queue gone, its writes with
	// it.
	void noteSurfaceWrites(RgfxAcceleratorUserClient *client, task_t task, UInt64 queue,
	                       UInt64 seqno, const UInt64 *pairs, UInt32 count);
	// The device cache of the shared block at `offset`, which `client` holds, for the surface of
	// ID `surfaceId` (RgfxAcceleratorContexts.cpp): made at the first ask and kept with the
	// block's entry. It and its IOSurface come retained, for the caller to release, as the
	// block's let-go may release the entry's meanwhile, and `*inPlace` says
	// whether the device draws it in place. Else why there is none: the client holds no such
	// block, its pages are not yet the device's, its surface is another, or no cache could be made
	// (logged where it was tried).
	enum SurfaceCacheAnswer : UInt32 {
		kSurfaceCacheMade,
		kSurfaceCacheNotHeld,
		kSurfaceCachePending,
		kSurfaceCacheOtherSurface,
		kSurfaceCacheNone,
	};
	SurfaceCacheAnswer surfaceDeviceCache(RgfxAcceleratorUserClient *client, task_t task,
	                                      UInt64 offset, uint32_t surfaceId, void **cache,
	                                      OSObject **surface, bool *inPlace);
	void retireSurfaceWrites(UInt64 queue, UInt64 seqno);
	void forgetQueueWrites(UInt64 queue);

	// The IOAccel surfaces (RgfxAcceleratorSurfaces.cpp). registerSurface names `surface`'s id
	// (kIOReturnExclusiveAccess where another surface holds it); unregisterSurface drops it and
	// the grants made for it.
	IOReturn registerSurface(RgfxSurfaceUserClient *surface, UInt32 surfaceId);
	void unregisterSurface(RgfxSurfaceUserClient *surface);
	// The framebuffer's scanout as it is now: its place and bytes in the region, its shape.
	struct Scanout {
		UInt64 offset;
		UInt64 length;
		UInt32 width;
		UInt32 height;
		UInt32 rowBytes;
	};
	bool currentScanout(Scanout *scanout);
	// Whether `task` owns the framebuffer's scanout (ReimsGraphics::ownsScanout): only the owner
	// locks a surface over it or is granted it.
	bool ownsScanout(task_t task);
	// For our user client `client` in `task`, of context `context`: the scanout surface
	// `surfaceId` of that task is drawn into, whose span `client` may name from then on
	// (grantedRange) until revokeGrants or the surface goes. The device is told the span is
	// `context`'s (BLOCK_GRANT) before this returns, and the span is mapped in the client's
	// window while the grant lasts. kIOReturnNotPermitted where `task` does not own the scanout.
	IOReturn grantSurfaceScanout(RgfxAcceleratorUserClient *client, UInt32 context, task_t task,
	                             UInt32 surfaceId, Scanout *scanout);
	void revokeGrants(const void *client);
	bool grantedRange(const void *client, UInt64 offset, UInt64 length);
	// The scanout's owner moved to `owner` (a new server connection, ReimsGraphics): every other
	// task's grant of the scanout goes, its span out of the client's window, and so does every
	// other task's lock mapping of it (RgfxScanoutOwner.hpp, rgfxKeepsScanout). Called with no
	// lock of the framebuffer's held.
	void scanoutOwnerMoved(task_t owner);

#if RGFX_FB_TRACE
	// A trace build logs the properties processes read of the accelerator.
	using IOAccelerator::copyProperty;
	OSObject *copyProperty(const OSSymbol *aKey) const override;
	OSObject *copyProperty(const OSString *aKey) const override;
	OSObject *copyProperty(const char *aKey) const override;
#endif

private:
	IOReturn newDisplayPipe(task_t owningTask, void *securityID, UInt32 type,
	                        OSDictionary *properties, IOUserClient **handler);
	IOReturn newSharedEventPort(task_t owningTask, void *securityID, UInt32 type,
	                            OSDictionary *properties, IOUserClient **handler);
	IOReturn newSurface(task_t owningTask, void *securityID, UInt32 type,
	                    OSDictionary *properties, IOUserClient **handler);
	IOReturn newGLClient(task_t owningTask, void *securityID, UInt32 type,
	                     OSDictionary *properties, IOUserClient **handler);
	UInt32 readRegister(UInt32 offset) const;
	void writeRegister(UInt32 offset, UInt32 value);
	void teardown();
	// Starts the bundle's check (settleBundle); false where it cannot.
	bool watchBundle();
	// Names the GPU's PCI provider GFX0, as AMD Macs do, so AppleGVA's video routing finds it
	// (its definition says what breaks without it); a no-op where another device is named GFX0.
	void nameProviderForVideo();
	// Publishes the host's hardware video that AppleGVA reads from the accelerator, from the
	// video caps the host wrote (RgfxVideoCodecs.hpp): HEVC decode, where the host has it.
	void publishVideoCodecs();
	// A fresh block in place's SURFACE_PAGES (RgfxAcceleratorContexts.cpp): its `count` runs of
	// the surface's `length` bytes written at `size` bytes into the block of `total` bytes at
	// `offset`, and posted for context `context`, its block cached where `cached`; true where
	// the device answered OK.
	bool postSurfacePages(UInt32 context, UInt64 offset, UInt64 total, UInt64 size,
	                      UInt64 length, const RgfxPageRun *runs, UInt32 count, bool cached);
	// SURFACE_PAGES of the block of `total` bytes at `offset` made for context `context`: its
	// `count` runs of `length` bytes at region offset `runsOffset`, already written, with
	// `flags` (RGFX_SURFACE_PAGES_*, none sent where 0). kIOReturnSuccess where the device
	// answered OK, kIOReturnUnsupported where it answered otherwise, its status in `*status`
	// where asked for, else why it did not answer.
	IOReturn postPages(UInt32 context, UInt64 offset, UInt64 total, UInt64 runsOffset,
	                   UInt64 length, UInt32 count, UInt32 flags, UInt32 *status = nullptr);
	// The new deferred block of entry `e`, pending, held by its asker of context `context` alone,
	// `total` bytes with its runs' room: makes its surface's device cache and reads the surface
	// (rgfxSurfaceLocksDefer); fresh, asks the host to defer it; not fresh, refused or with no
	// cache, names its pages now, the entry keyed by them from then on. Whether the device took
	// the block, and `*cached` where it is cached. Not under the contexts' lock.
	bool deferSurfaceBlock(RgfxSurfaceEntry &e, UInt32 context, uint32_t surfaceId, UInt64 total,
	                       bool *cached);
	// Holds the dormant block of entry `e`, deferred or cached in place, again for the ask of
	// IOSurfaceID `surfaceId`, its own surface, which found it: false, nothing changed, where its
	// cache asked it to go. Under the contexts' lock.
	bool reviveBlockOfSurface(RgfxSurfaceEntry &e, uint32_t surfaceId);
	// Makes the device cache of the surface of IOSurfaceID `surfaceId` in `task` for the fresh
	// block in place of entry `e` at `offset`, still pending, and records the block cached
	// (rgfxSurfaceCacheAdd): false, nothing made, where it cannot be.
	bool cacheSurface(RgfxSurfaceEntry &e, task_t task, uint32_t surfaceId, UInt64 offset);
	// Holds the dormant block of entry `e` again for the ask of the surface of IOSurfaceID
	// `surfaceId` in `task`, where it is that block's surface (RgfxSurfaceBlocks.hpp): false,
	// nothing changed, where it is not. Under the contexts' lock.
	bool reviveSurfaceBlock(RgfxSurfaceEntry &e, task_t task, uint32_t surfaceId);

	IOPCIDevice *pci;
	IOMemoryMap *registerMap;
	volatile UInt8 *registers;
	IODeviceMemory *region;
	UInt64 regionSize;

	// The contexts' state lives in RgfxAcceleratorContexts.cpp, not in this class, so the
	// class's layout and its own code stay as the loader's sites were taken from; it is made
	// at the first context's open.
	bool startContexts();
};

#endif // RGFX_ACCELERATOR_HPP_
