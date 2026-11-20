// RgfxUserClientABI: the accelerator user client's protocol between the kext and the guest's
// Metal bundle: the user-client type the bundle opens, the external-method selectors, and the
// shared block header's word the kext writes. One source for both sides: the kext
// (RgfxAccelerator.hpp) and the bundle (guest/bundle/src/RgfxMetalDevice.mm,
// RgfxIOSurface.mm, tools/window_probe.m) include it. Constants alone, in C as in C++, and free
// of the kernel's headers, so the native test (tests/user_client_words_test.cpp) builds it as
// both sides do.
//
// Our own code.
#ifndef RGFX_USER_CLIENT_ABI_HPP_
#define RGFX_USER_CLIENT_ABI_HPP_

// Our own user-client type: a private one ('RGFX'), clear of the public IOAccel surface type 0
// (kRgfxSurfaceUserClientType) and of the types Apple's frameworks open.
enum {
	kRgfxAccelUserClientType = 0x52474658,
	// A shared event's port ('RGSE', RgfxSharedEventUserClient.hpp): a connection the process
	// that hands its shared event to another opens on the accelerator. Its send right is the
	// event's eventPort, which a handle carries to another process, and which that process's
	// calls on it (kRgfxSharedEventMethodImport) prove it holds. Where the kext's word holds
	// kRgfxKextFeatureSharedEvents.
	kRgfxSharedEventUserClientType = 0x52475345,
};

// External-method selectors the user client dispatches.
//
// Every selector of ours, on the 'RGFX' client and on a shared event's port, is a number that
// every WebKit web-content sandbox lets its process call on an IOAccelerator connection, on every
// release from macOS 11 to 26: the web content process of Safari 15.1 to 17.6 on macOS 12 may
// call only 0 to 13, 15 to 18, 22 to 24, 28 to 31, 40, 42, 48, 256 to 265, 267, 268 and 512
// there, the numbers Apple's own drivers take, and its sandbox refuses any other before the kext
// sees it; on macOS 11, 13 and later it may call any number (design.md, "The accelerator's user
// client: its numbers"; tests/user_client_words_test.cpp holds the set and checks every selector
// the bundle calls against it). The numbers are free to move only because the bundle and the
// kext ship as one set: the features call stays at kRgfxAccelMethodFeatures in every map and
// answers the map's number, kRgfxSelectorMap, and a bundle that meets a kext of another map, or
// none, refuses the device rather than call a number that means something else there.
enum {
	// The selector map these numbers make, which the features call answers third. A change of
	// any number below takes the next map number.
	kRgfxSelectorMap = 1,
};

enum {
	// Ring a ring's doorbell: scalarInput[0] is the ring id, written to
	// RGFX_REG_DOORBELL. No output.
	kRgfxAccelMethodDoorbell = 0,
	// 1 is kept for Consumed, the completion ring's room doorbell (kRgfxKextFeatureConsumedDoorbell).
	// 2 is given to no selector of ours: it is the IOGPU base's device-info call on its own client
	// type (RgfxIOGPUReplies.hpp), so a trace names one call per number.
	// The features: no input; out: the device's feature word (the kext's RgfxFeatures.hpp), the
	// RGFX_FEATURE_* bits 0 to 2 as the host offers them and from bit 3 the ones the framebuffer
	// took for the kext and its bundle; then the kext's own word (kRgfxKextFeature*, below); then
	// the selector map, kRgfxSelectorMap. It keeps this number in every map. A kext of the map
	// before this one answers kIOReturnUnsupported here, and the bundle then has no device.
	kRgfxAccelMethodFeatures = 3,
	// A pages block over the process's own memory (kRgfxKextFeaturePagesBlock;
	// RgfxPagesBlock.hpp): in the memory's address in the process, its bytes, and how the process
	// maps it (kRgfxPagesWritable and, below, kRgfxPagesUnchanging and kRgfxPagesCopyOnWrite; 0
	// for memory it may not write); out the block's region offset and its size, and where asked for
	// (kRgfxKextFeatureImportAddresses), a third: the GPU address the device gave the block's
	// view, 0 where it gave none. The kext wires the pages that
	// hold the range, rounded out to whole pages, in the process for as long as the block lives,
	// lists their guest RAM runs in the block, and hands them to the device (SURFACE_PAGES, marked
	// RGFX_SURFACE_PAGES_PRIVATE) before its grant (BLOCK_GRANT): the device's view of them starts
	// at the range's first page, so a buffer over the range's first byte lies at that byte's
	// offset in its page (BUFFER_OVER_PAGES). The block is the client's, freed through
	// kRgfxAccelMethodFree; the process never maps it, so it never sees a guest physical address.
	// The pages are unwired once the device has answered the block's release, or the client's
	// close, and stay wired for good where neither is answered. Memory the process may not write
	// (0) is wired for reading alone and its SURFACE_PAGES marked RGFX_SURFACE_PAGES_READ_ONLY,
	// where the device took RGFX_FEATURE_READ_ONLY_PAGES: the host lets only objects whose GPU
	// writes it drops lie over it. Memory named unchanging (kRgfxPagesUnchanging,
	// kRgfxKextFeatureUnchangingPages) is read-only memory the process can never write and no one
	// else writes, a file's pages on a read-only volume or anonymous memory with a copy pending
	// (the bundle's RgfxFilePages.hpp), wired for reading as read-only memory is, and its
	// SURFACE_PAGES marked RGFX_SURFACE_PAGES_UNCHANGING too, and
	// RGFX_SURFACE_PAGES_COPY_ON_WRITE where it is also named so, where the device took
	// RGFX_FEATURE_UNCHANGING_PAGES (RgfxPagesBlock.hpp); read-only memory not named so may carry
	// why not (kRgfxPagesWhyShift), which its SURFACE_PAGES carries on where the device took
	// RGFX_FEATURE_UNCHANGING_PAGES. kIOReturnUnsupported where the device lacks
	// RGFX_FEATURE_PAGES_BUFFERS or refused the pages, for memory the process may not write
	// where it lacks RGFX_FEATURE_READ_ONLY_PAGES: the GPU's writes would land in pages the
	// process cannot write, and for memory named unchanging where it lacks
	// RGFX_FEATURE_UNCHANGING_PAGES. kIOReturnBadArgument where the range is not mapped in the
	// process, for memory named writable and unchanging, or copy-on-write and not unchanging, and
	// for a why with writable or unchanging memory;
	// kIOReturnNoSpace or kIOReturnNoResources where the region or the kext's table has no room
	// for the block.
	kRgfxAccelMethodPagesBlock = 4,
	// Posts the client's messages no one waits for that name no block, of its own objects, on
	// the control ring in one call (kRgfxKextFeatureSubmitBatch): struct in, inline, messages
	// back to back of the types RgfxSubmitGate's rgfxSubmitBatchSize names (OBJECT_DESTROY,
	// BUFFER_AT_OFFSET, TEXTURE_CREATE, TEXTURE_VIEW, MANAGED_TEXTURE_CREATE,
	// MANAGED_BUFFER_CREATE and BUFFER_DID_MODIFY_RANGE), each whole (its header's type and size
	// set, the size its type's; the kext stamps its seqno) and of an object in the client's
	// context, at most kRgfxSubmitBatchBytes in all; out: how many were posted, in their order,
	// each as kRgfxAccelMethodSubmit posts it (a destroy of one of the client's command queues
	// ends the queue as Submit's does, after the messages before it; an object made for a context
	// closed since is destroyed behind it). None is posted where any is not such a message
	// (kIOReturnBadArgument). Succeeds with fewer than all only where the control ring stayed full
	// or the device stopped: the rest were not posted, and go in a later call, before anything
	// else of the client's.
	// With three scalars in and two out (kRgfxKextFeatureQueuePublish): one of the client's
	// queues, its submission ring's new producer and the publish's flags (RgfxQueuePublish.hpp's
	// kRgfxPublish*): once every message is posted, the kext publishes the producer on the queue's
	// ring after them and rings its doorbell as the bundle's own publish would; the second output
	// says whether it did (0 where any message was not posted, or the queue is not one whose ring
	// header the kext keeps: the bundle then publishes the buffer itself, or fails it).
	kRgfxAccelMethodSubmitBatch = 5,
	// 6 is kept for WakeCompletions (kRgfxKextFeatureWakeCompletions).
	// The answers to the client's deferred pipeline creates (Submit's kRgfxSubmitDeferred) the
	// device has given (kRgfxKextFeatureDeferredAnswers): in the seqno of one to wait for first
	// (0: none), up to a minute; struct out, the answers taken, each a whole
	// RGFX_EVT_PIPELINE_REFLECTION, in their seqnos' order, as many as fit; out their count.
	// Returns success whether or not the one waited for came: the answers that did are out all
	// the same.
	kRgfxAccelMethodTakeAnswers = 7,
	// Lets go of one ask of the shared block at the region offset scalarInput[0]; the block goes
	// once no process holds it, through its BLOCK_RELEASE.
	kRgfxAccelMethodSurfaceBlockRelease = 8,
	// The client's queue scalarInput[0] has its buffers up to the seqno scalarInput[1] complete,
	// their copies written back: their writes retire.
	kRgfxAccelMethodSurfaceRetire = 9,
	// 10 and 11 are a shared event's port's own (below).
	// A pages block over memory of the kext's own (kRgfxKextFeaturePagesChunked;
	// RgfxPagesBlock.hpp): in the memory's length, a multiple of kRgfxPagesChunk (below); out its address
	// in the process, the block's region offset and its size. The kext takes the length in
	// physically contiguous chunks of kRgfxPagesChunk bytes each, each at a multiple of its bytes
	// in guest RAM, zero-filled, maps them into the process in order, read and write, and makes the
	// block of their runs as PagesBlock makes one of the process's pages: so a 2 MiB slab is a few
	// runs of the guest's RAM where the process's own pages are hundreds. The memory is the
	// block's: the process never unmaps it, and kRgfxAccelMethodFree of the block takes it with the
	// block, unmapped from the process and freed once the device has answered the block's release,
	// or the client's close.
	// kIOReturnUnsupported as for PagesBlock; kIOReturnBadArgument for a length of no chunks, not a
	// multiple of one, or past kRgfxPagesChunkedMaxBytes; kIOReturnNoMemory where the kernel has
	// not that many chunks to give, which the process then takes plain pages for.
	kRgfxAccelMethodPagesChunked = 12,
	// The two seeds of an IOSurface the client's process holds (kRgfxKextFeatureSurfaceSeeds): in
	// its IOSurfaceID; out its lock seed (IOSurfaceGetSeed's) and its unlock seed. IOSurface moves
	// the lock seed at a read-write lock and the unlock seed at that lock's unlock, and both at a
	// GPU write (IOSurface::getUnlockSeed, which the Mac's GPU families read: the kext's
	// RgfxIOSurfaceKPI.hpp), so the two differ while a CPU holds the surface locked for writing.
	// The bundle reads them for a copy of the surface synced while such a lock was held, which it
	// syncs again once the unlock seed moves, and, where the host took RGFX_FEATURE_SURFACE_REFILL,
	// for a cached surface a commit named at a new CPU generation, whose device cache it has the
	// host fill again once the unlock seed moves. kIOReturnNotFound where the process holds no such
	// surface; kIOReturnUnsupported where IOSurface has no unlock seed to read.
	kRgfxAccelMethodSurfaceSeeds = 13,
	// Every CPU's page attribute table, IA32_PAT, which the x86-64 kext reads on each online CPU
	// itself (kRgfxKextFeatureCpuPats; the kext's RgfxPat.hpp): no input; struct out, inline, a
	// 64-bit word per CPU number from 0, as many as fit, RGFX_PAT_NONE for a CPU that did not
	// answer (one offline); out the CPUs that answered, the kernel's CPU count, and the CPUs that
	// answered with a number past that count, which the struct does not hold. For an
	// administrator's process alone, as the read stops every CPU for a moment:
	// kIOReturnNotPrivileged for any other. kIOReturnUnsupported on arm64, which has no PAT.
	kRgfxAccelMethodCpuPats = 15,
	// A block of the device's Shared pool for the client's context (kRgfxKextFeaturePool; the
	// device's RGFX_FEATURE_POOL, design.md, "The Shared pool"): in the bytes asked for; out the
	// block's address in the process, its offset in the pool and its length, the bytes asked for
	// rounded up to RGFX_POOL_BLOCK_ALIGN, and, where four outputs are asked for
	// (kRgfxKextFeaturePoolAddresses), the GPU address the host gave the block, 0 where the
	// feature word lacks RGFX_FEATURE_POOL_ADDRESSES. The kext posts the context's POOL_TAKE and
	// waits for its answer however long the host takes, then maps the block into the process,
	// read and write and write-back, as its own range: every byte of it reads zero, the host's
	// zeroing at the take. The process names the block by its pool offset (HEAP_OVER_POOL) and
	// gives it back through PoolGive; the context's close gives back every block it still holds.
	// kIOReturnUnsupported where the device has no pool; kIOReturnBadArgument for 0 bytes or
	// past RGFX_POOL_TAKE_MAX, or where the context is not open; kIOReturnNoMemory where the
	// pool has no room for the block (RGFX_STATUS_REFUSED) or the process got no mapping of it,
	// the block then given back; kIOReturnNoResources where the kext could not list it.
	kRgfxAccelMethodPoolTake = 16,
	// Gives back the client's Shared pool block at the pool offset scalarInput[0], and returns
	// at once: the block's mapping goes from the process first, and its POOL_GIVE is posted
	// after every message the process posted before the call. The host takes the block back once
	// the context's command buffers committed before have finished, and refuses it while an
	// object it holds (a heap) lies in it: the process destroys its objects over the block first.
	// kIOReturnNotFound where the client holds no block at that offset.
	kRgfxAccelMethodPoolGive = 17,
	// A Private block for the client's context (kRgfxKextFeaturePrivateBlocks; the device's
	// RGFX_FEATURE_PRIVATE_BLOCKS, design.md, "Private blocks"): in the bytes asked for; out the
	// block's GPU address and its length, at least the bytes asked for rounded up to
	// RGFX_PRIVATE_BLOCK_ALIGN. A block is a range of GPU addresses with no memory the process
	// maps: the process places its Private buffers, its heaps that are not Shared and its Managed
	// buffers' GPU copies in it, and gives it back through PrivateGive; the context's close gives
	// back every block it still holds. The kext posts the context's PRIVATE_TAKE and waits for its
	// answer however long the host takes, and lists no block: the host knows each context's.
	// kIOReturnUnsupported where the device offers no Private blocks; kIOReturnBadArgument for 0
	// bytes or past RGFX_PRIVATE_TAKE_MAX, or where the context is not open; kIOReturnNoMemory
	// where the host had no block for it (RGFX_STATUS_REFUSED).
	kRgfxAccelMethodPrivateTake = 18,
	// Gives back the client's Private block at the GPU address scalarInput[0], of the length
	// scalarInput[1] PrivateTake answered, and returns at once: its PRIVATE_GIVE is posted after
	// every message the process posted before the call. The host takes the block back once the
	// context's command buffers committed before have finished, and refuses it while an object it
	// holds lies in it, and a block the context does not hold: the process destroys its objects in
	// the block first. kIOReturnUnsupported where the device offers no Private blocks.
	kRgfxAccelMethodPrivateGive = 22,
	// A new selector takes the next number design.md's registry gives out.
};

// The process's device context (RgfxAcceleratorContexts.cpp), on our own user client only: the
// selectors below, and the features call, pages blocks, answers, releases and retires above,
// which the kext routes to its context methods (rgfxAccelContextSelector).
enum {
	// Opens the client's context and its window. No input; out: the context number c, the
	// object-id base (c << RGFX_CONTEXT_OBJ_ID_SHIFT), the GPU-address base of the context's
	// range, and the window's base in the process and its size (the region's): region offset o
	// of what the client is given lies at the base plus o, and nothing else of the region is
	// mapped there (RgfxAcceleratorUserClient.hpp). A sixth output, where asked
	// (kRgfxKextFeatureContextSpan), is the range's span in bytes, from the base; a client that
	// asks for five holds RGFX_CONTEXT_VA_SPAN from the base. A seventh, where asked
	// (kRgfxKextFeatureDoorbellPage), is the address in the process of the context's doorbell
	// page (RGFX_FEATURE_DOORBELL_PAGES, RgfxDoorbellPages.hpp), mapped uncached for its whole
	// RGFX_DOORBELL_PAGE_SIZE bytes, where a 32-bit store at its first byte of one of the
	// context's queues' ring ids rings that ring as kRgfxAccelMethodDoorbell does; 0 where the
	// context has none, and the process rings through kRgfxAccelMethodDoorbell. A context opened
	// without asking for the seventh has no page. kIOReturnNoResources where the kext has no
	// range left (RgfxContextSlots.hpp).
	kRgfxAccelMethodOpenContext = 0x100,
	// A region block: in size, align (a power of two); out: its region offset. A third input,
	// where the kext's word holds kRgfxKextFeatureWriteCombined, is the byte of the block from
	// which the process's mapping of it is write-combined (kIOMapWriteCombineCache), a multiple of
	// a page; the bytes before it, and the whole block where it is the block's size or more, are
	// mapped write-back as any block's.
	kRgfxAccelMethodAlloc = 0x101,
	// Frees the client's block at the region offset scalarInput[0], and returns at once: the
	// block is no longer the client's to name, and is given out again once the device answers
	// its BLOCK_RELEASE OK (RgfxBlockTable.hpp). kIOReturnBusy while a create naming it is in
	// flight.
	kRgfxAccelMethodFree = 0x102,
	// Opens a command queue: in obj_id, submit offset, complete offset, submit data size,
	// complete data size, priority; out: the device's status (RGFX_STATUS_OK, or the
	// refusal). Each ring must lie in one of the client's blocks. The device has the queue's
	// rings when it returns. A seventh input, the RGFX_QUEUE_* flags the process wants, asks for
	// a second output, the flags granted: those whose feature the feature word holds
	// (kRgfxAccelMethodFeatures, the kext's RgfxFeatures.hpp), which the QUEUE_CREATE carries in
	// its tail where there are any.
	kRgfxAccelMethodQueueCreate = 0x103,
	// Closes the client's queue scalarInput[0].
	kRgfxAccelMethodQueueDestroy = 0x104,
	// Posts one object-lifecycle message on the control ring: struct in, the whole message
	// (its header's type set; the kext stamps its size and seqno), a create or destroy of an
	// object in the client's context (its obj_id, 16 bytes in, in the context's range), at most
	// ReimsGraphics::kLifecycleMessageMax bytes, naming only the client's own blocks of the region
	// (RgfxRegionSpans.hpp); out: the seqno, and for a create the device answers (a pipeline's,
	// a library's), its answer as struct out once the device has given it. A create not answered
	// in time fails, and the kext destroys its object behind it. One scalar in, of the
	// kRgfxSubmit* flags, may follow: kRgfxSubmitDeferred posts a pipeline create without
	// waiting, its answer kept for kRgfxAccelMethodTakeAnswers, and answers no struct out; where
	// the kext has no slot left for the answer, the create is posted and answered as without it.
	// kRgfxSubmitShared, with the create's pipeline key after the flags (kRgfxSubmitKeyWords
	// scalars, its bytes in order) and two outputs: where the kext keeps the device's answer to an
	// equal create of any process (RgfxSharedPipelines.hpp) and has a slot left, the create is
	// posted as with kRgfxSubmitDeferred and the kept answer is the struct out, stamped with this
	// create's seqno and obj_id, the second output 1; else it is posted and answered as without it,
	// the second output 0.
	kRgfxAccelMethodSubmit = 0x105,
	// Gives the client's context a chunk of texture resource ids from the guest's one pool:
	// RGFX_RESOURCE_ID_SMALL_CHUNK of them where the kext's feature word holds
	// RGFX_FEATURE_SMALL_ID_CHUNKS, else RGFX_RESOURCE_ID_CHUNK. No input; out: the chunk's first
	// id, and, where two outputs are asked, how many ids the chunk holds, which the bundle takes
	// as the chunk's size. kIOReturnNoResources once every chunk is held. The device has the grant
	// (RESOURCE_ID_SMALL_CHUNK or RESOURCE_ID_CHUNK) before it returns; the context holds the
	// chunk until its close.
	kRgfxAccelMethodResourceIds = 0x106,
	// Waits for the client's command buffers to complete: in the last count this returned (0
	// at first); out: the count now. Returns once the count is not the one given, which it
	// becomes when a completion ring of one of the client's queues holds records the client
	// has not taken (the device's interrupt, RgfxAccelerator::queueInterrupt). kIOReturnAborted
	// when the thread is interrupted, kIOReturnNotOpen once the client's context is released.
	kRgfxAccelMethodWaitCompletions = 0x107,
	// The scanout an IOAccel surface of the client's process is drawn into, for a texture over
	// it: in the surface's id and its buffer (0, the one buffer); out its region offset, its
	// bytes, its width, height and bytes per row (BGRA8 pixels). The surface must be the
	// framebuffer's whole scanout (RgfxSurfaceUserClient::isScanout); from then on the client's
	// lifecycle messages may name the span, until its context is released or the surface goes.
	kRgfxAccelMethodSurfaceScanout = 0x108,
	// The shared block of an IOSurface plane the client's process holds (RgfxSurfaceBlocks.hpp):
	// in the plane's CPU address in the process, its bytes there, and the block's size (whole
	// pages, at least those bytes); out the block's region offset, its size, its serial (unique
	// to the block) and 1 where this ask made it (its bytes are not yet the plane's), else 0.
	// Every process that holds the surface gets the same block for the plane, and its lifecycle
	// messages may name it from then on, until it lets go of its last ask. The device has the
	// hold (BLOCK_SHARE) before this returns. kIOReturnBadArgument where the range is not mapped
	// in the process, or the plane's block has another size or mode.
	// A fourth input of 1 asks for the block in place (RGFX_FEATURE_SURFACE_PAGES): the first
	// input is then the surface's page-aligned base, the second its whole allocation (whole
	// pages), and the size that of the part the processes map, which holds no pixels: the device
	// uses the surface's own pages, which the kext wires and hands it (SURFACE_PAGES) before the
	// first hold. The size out is that part's. kIOReturnUnsupported where the device lacks the
	// feature or refused the pages: the process copies the surface instead.
	// Six inputs, in place, add the surface's IOSurfaceID and whether the process may have its
	// GPU copy kept in a device cache (1: one plane, no YCbCr): where the device has
	// RGFX_FEATURE_SURFACE_CACHE, the block that ask makes is cached (RgfxSurfaceCaches.hpp), and
	// its header holds the surface's CPU generation at kRgfxSurfaceCpuGenOffset and its clients at
	// kRgfxSurfaceClientsOffset. A fifth output, where asked, is 1 where the block is cached, else
	// 0. With six inputs, a struct input of exactly sizeof(RgfxSurfacePagesLayout) is the
	// surface's planes as its memory holds them: where the device answers which surfaces need a
	// device cache (RGFX_FEATURE_SURFACE_CACHE_REFUSED), a new cacheable block asks it with them,
	// and is cached only where it answers so; with none, it stays in place. A struct input of
	// the layout followed by an RgfxSurfaceResident (RgfxDeferCarry.hpp) lists the surface's
	// resident pages the process found in its mapping, where the kext took
	// RGFX_FEATURE_SURFACE_DEFERRED_CARRY: a new block of a surface fresh but for those is
	// deferred carrying them, where the kext's read proves them; elsewhere the list is ignored.
	kRgfxAccelMethodSurfaceBlock = 0x109,
	// The device info the host wrote at RGFX_DEVICE_INFO_OFFSET (RgfxDeviceInfo): no input;
	// struct out, exactly sizeof(RgfxDeviceInfo) bytes, or those and the video caps after them
	// (RgfxVideoCaps). Its page holds the control completion ring's tail, so no process maps
	// it; the kext reads it for the process.
	kRgfxAccelMethodDeviceInfo = 0x10b,
	// The surfaces a command buffer writes, named after the buffer is posted and before its commit
	// returns (RgfxAcceleratorLocks.cpp): struct in, words of 64 bits: the client's queue, the
	// buffer's seqno there, a count n, then n pairs of a shared block's region offset (one the
	// client holds, kRgfxAccelMethodSurfaceBlock) and its surface's IOSurfaceID. A CPU lock of each
	// surface, in any process, waits from then until the write retires (below, or the device's
	// completion of the buffer for a surface drawn in place). No output; a pair the kext cannot
	// follow is logged and skipped.
	kRgfxAccelMethodSurfaceWrites = 0x10c,
};

// Whether the 'RGFX' client routes `selector` to its context methods (RgfxAcceleratorContexts.cpp):
// every selector of its own but the doorbell's.
static inline int rgfxAccelContextSelector(unsigned selector)
{
	switch (selector) {
	case kRgfxAccelMethodFeatures:
	case kRgfxAccelMethodPagesBlock:
	case kRgfxAccelMethodPagesChunked:
	case kRgfxAccelMethodSurfaceSeeds:
	case kRgfxAccelMethodCpuPats:
	case kRgfxAccelMethodPoolTake:
	case kRgfxAccelMethodPoolGive:
	case kRgfxAccelMethodPrivateTake:
	case kRgfxAccelMethodPrivateGive:
	case kRgfxAccelMethodSubmitBatch:
	case kRgfxAccelMethodTakeAnswers:
	case kRgfxAccelMethodSurfaceBlockRelease:
	case kRgfxAccelMethodSurfaceRetire:
	case kRgfxAccelMethodOpenContext:
	case kRgfxAccelMethodAlloc:
	case kRgfxAccelMethodFree:
	case kRgfxAccelMethodQueueCreate:
	case kRgfxAccelMethodQueueDestroy:
	case kRgfxAccelMethodSubmit:
	case kRgfxAccelMethodResourceIds:
	case kRgfxAccelMethodWaitCompletions:
	case kRgfxAccelMethodSurfaceScanout:
	case kRgfxAccelMethodSurfaceBlock:
	case kRgfxAccelMethodDeviceInfo:
	case kRgfxAccelMethodSurfaceWrites:
		return 1;
	default:
		return 0;
	}
}

// A shared event's port's selectors (kRgfxSharedEventUserClientType), numbered apart from the
// accelerator's own so that a call on a connection of another kind is refused there as unknown.
enum {
	// Names the caller's shared event for other processes, once per port: in the caller's
	// context (OpenContext's out[0]), the event's obj_id and the obj_id of the caller's queue
	// whose completion ring takes the values other holders give the event; out the device's
	// status (RGFX_STATUS_OK, or the refusal). The device has the name when it returns OK, and
	// the port then names the event until no process holds it (SHARED_EVENT_UNNAME).
	// kIOReturnBadArgument where the context is not one the calling task holds, or the ids are
	// not in it; kIOReturnExclusiveAccess where the port names an event already.
	kRgfxSharedEventMethodExport = 10,
	// Makes the caller's new shared event from the event the port names: in the caller's context,
	// the new event's obj_id and its listen queue, as Export's; out the device's status, and the
	// event's value, which the new event starts at. Any task that holds the port may call it, and
	// only one that does: that is what makes the port the capability. kIOReturnBadArgument where
	// the context is not one the calling task holds, or the ids are not in it;
	// kIOReturnNotReady where the port names no event yet.
	kRgfxSharedEventMethodImport = 11,
};

// How the process maps the memory of a PagesBlock (4), its third scalar in: writable; unchanging,
// read-only memory the process can never write and no one else writes, mapped with no write in
// the mapping's maximum protection, a file's pages on a read-only volume or anonymous memory with a
// copy pending (kRgfxKextFeatureUnchangingPages; never with kRgfxPagesWritable);
// copy-on-write, which only unchanging memory is named, where the process maps the file's pages
// private or the anonymous memory with a copy pending; and for read-only memory not named
// unchanging, in the bits from
// kRgfxPagesWhyShift (kRgfxPagesWhyMask), why not: one of the wire's RGFX_PAGES_WHY_* values, a
// diagnostic the block's SURFACE_PAGES carries to the host (RGFX_SURFACE_PAGES_WHY_SHIFT).
enum {
	kRgfxPagesWritable = 1 << 0,
	kRgfxPagesUnchanging = 1 << 1,
	kRgfxPagesCopyOnWrite = 1 << 2,
	kRgfxPagesHowBits = kRgfxPagesWritable | kRgfxPagesUnchanging | kRgfxPagesCopyOnWrite,
	kRgfxPagesWhyShift = 8,
	kRgfxPagesWhyMask = 0xf << kRgfxPagesWhyShift,
};

// PagesChunked's (12) chunk, 64 KiB, each at a multiple of 64 KiB of guest RAM
// (RgfxPagesBlock.hpp, kRgfxPagesChunkMask), and the most bytes one call takes, 1 GiB. A 64 KiB
// run of RAM is one the kernel finds at once, however long the guest has run: measured on an
// x86-64 macOS 26.7.2 guest of 8 GiB on the Radeon 780M host, with chunks at any multiple of
// 4 KiB, 4,096 such chunks in a row each took 0.033 ms, and 0.036 ms once 978 MiB of 4 KiB pages
// had been taken, none longer than 0.74 ms, none refused. Of 128 runs of 2 MiB asked, it gave 28
// on the quiet guest and 16 once memory had been taken, the slowest in 264 ms.
enum {
	kRgfxPagesChunk = 0x10000,
	kRgfxPagesChunkedMaxBytes = 0x40000000,
};

// The most bytes of messages one kRgfxAccelMethodSubmitBatch takes: an inline struct in, 170
// destroys, 73 heap-placed buffers or 39 textures.
enum {
	kRgfxSubmitBatchBytes = 4096,
};

// kRgfxAccelMethodSubmit's flags, its optional scalar in.
enum {
	// Post a pipeline create without waiting for its answer (kRgfxAccelMethodTakeAnswers;
	// kRgfxKextFeatureDeferredAnswers).
	kRgfxSubmitDeferred = 1 << 0,
	// Post a pipeline create without waiting where an equal create's answer is kept, by its
	// pipeline key (kRgfxKextFeatureSharedPipelines).
	kRgfxSubmitShared = 1 << 1,
	// The scalars a kRgfxSubmitShared key takes: 32 bytes.
	kRgfxSubmitKeyWords = 4,
};

// The kext's own features, the second word kRgfxAccelMethodFeatures answers: the user-client
// calls and flags a kext may lack, which the bundle uses only where the word holds their bit
// (design.md, "The accelerator's user client").
enum {
	// PagesBlock, 4.
	kRgfxKextFeaturePagesBlock = 1 << 0,
	// SubmitBatch, 5.
	kRgfxKextFeatureSubmitBatch = 1 << 1,
	// WakeCompletions, 6.
	kRgfxKextFeatureWakeCompletions = 1 << 2,
	// The doorbell table's method 1, Consumed.
	kRgfxKextFeatureConsumedDoorbell = 1 << 3,
	// Submit's (0x105) deferred-answer flag, and TakeAnswers, 7.
	kRgfxKextFeatureDeferredAnswers = 1 << 4,
	// OpenContext's (0x100) sixth output, the span of the context's GPU addresses.
	kRgfxKextFeatureContextSpan = 1 << 5,
	// A shared event's port (kRgfxSharedEventUserClientType) and its selectors, 10 and 11,
	// which the bundle uses where the feature word holds RGFX_FEATURE_SHARED_EVENT_NAMES too.
	kRgfxKextFeatureSharedEvents = 1 << 6,
	// Alloc's (0x101) third input, the part of the block mapped write-combined.
	kRgfxKextFeatureWriteCombined = 1 << 7,
	// PagesChunked, 12.
	kRgfxKextFeaturePagesChunked = 1 << 8,
	// Submit's (0x105) shared flag, kRgfxSubmitShared, which keeps answers only where the feature
	// word also holds RGFX_FEATURE_PIPELINE_KEYS.
	kRgfxKextFeatureSharedPipelines = 1 << 9,
	// SurfaceSeeds, 13.
	kRgfxKextFeatureSurfaceSeeds = 1 << 10,
	// SubmitBatch's (5) publish of a queue's ring with the messages (RgfxQueuePublish.hpp).
	kRgfxKextFeatureQueuePublish = 1 << 11,
	// OpenContext's (0x100) seventh output, the context's doorbell page in the process.
	kRgfxKextFeatureDoorbellPage = 1 << 12,
	// CpuPats, 15.
	kRgfxKextFeatureCpuPats = 1 << 13,
	// Every block Alloc (0x101) gives, and every new block SurfaceBlock (0x109) makes, reads zero
	// over every byte the process's window maps of it, whatever its last holder, of any context,
	// left there (the kext's RgfxBlockZero.hpp).
	kRgfxKextFeatureZeroedBlocks = 1 << 14,
	// PoolTake, 16, and PoolGive, 17, which the bundle uses where the feature word holds
	// RGFX_FEATURE_POOL too; the x86-64 and arm64e kexts.
	kRgfxKextFeaturePool = 1 << 15,
	// PoolTake's (16) fourth output, the block's GPU address, which the bundle asks for where the
	// feature word holds RGFX_FEATURE_POOL_ADDRESSES too; the x86-64 and arm64e kexts.
	kRgfxKextFeaturePoolAddresses = 1 << 16,
	// PrivateTake, 18, and PrivateGive, 22, which the bundle uses where the feature word holds
	// RGFX_FEATURE_PRIVATE_BLOCKS too; the x86-64 and arm64e kexts.
	kRgfxKextFeaturePrivateBlocks = 1 << 17,
	// PagesBlock's (4) third output, the GPU address the device gave the block's view, which the
	// bundle asks for where the feature word holds RGFX_FEATURE_IMPORT_ADDRESSES too; the x86-64
	// and arm64e kexts.
	kRgfxKextFeatureImportAddresses = 1 << 18,
	// PagesBlock's (4) memory named unchanging (kRgfxPagesUnchanging, kRgfxPagesCopyOnWrite),
	// which the bundle names where the feature word holds RGFX_FEATURE_UNCHANGING_PAGES too; the
	// x86-64 and arm64e kexts.
	kRgfxKextFeatureUnchangingPages = 1 << 19,
};

// Where a cached block's header holds its surface's CPU generation, a 64-bit count the kext
// writes at each read-write CPU lock (the bundle's RgfxSurfaceHeader::cpuGen), and its clients,
// a 32-bit count of the processes that hold the surface (IOSurface's client retain count), which
// the kext writes as it caches the block, as a process holds the block again and as each client
// goes; 0 where no count was written.
enum {
	kRgfxSurfaceClientsOffset = 12,
	kRgfxSurfaceCpuGenOffset = 16,
};

#endif // RGFX_USER_CLIENT_ABI_HPP_
