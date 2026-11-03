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
};

// External-method selectors the user client dispatches.
enum {
	// Ring a ring's doorbell: scalarInput[0] is the ring id, written to
	// RGFX_REG_DOORBELL. No output.
	kRgfxAccelMethodDoorbell = 0,
	// 1 is kept for Consumed, the completion ring's room doorbell (kRgfxKextFeatureConsumedDoorbell).
};

// The process's device context (RgfxAcceleratorContexts.cpp), on our own user client only,
// numbered clear of the selectors Apple's IOGPU base drives on its clients. They run from
// kRgfxAccelMethodContextFirst to kRgfxAccelMethodContextLast with no gap, which is the range the
// kext routes to its context methods.
enum {
	// Opens the client's context and its window. No input; out: the context number c, the
	// object-id base (c << RGFX_CONTEXT_OBJ_ID_SHIFT), the GPU-address base of the context's
	// range, and the window's base in the process and its size (the region's): region offset o
	// of what the client is given lies at the base plus o, and nothing else of the region is
	// mapped there (RgfxAcceleratorUserClient.hpp). A sixth output, where asked
	// (kRgfxKextFeatureContextSpan), is the range's span in bytes, from the base; a client that
	// asks for five holds RGFX_CONTEXT_VA_SPAN from the base. kIOReturnNoResources where the
	// kext has no range left (RgfxContextSlots.hpp).
	kRgfxAccelMethodOpenContext = 0x100,
	// A region block: in size, align (a power of two); out: its region offset.
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
	kRgfxAccelMethodSubmit = 0x105,
	// Gives the client's context a chunk of RGFX_RESOURCE_ID_CHUNK texture resource ids, from
	// the guest's one pool of RGFX_RESOURCE_ID_CHUNKS: no input; out: the chunk's first id.
	// kIOReturnNoResources once every chunk is held. The device has the grant
	// (RESOURCE_ID_CHUNK) before it returns; the context holds the chunk until its close.
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
	// its header holds the surface's CPU generation at kRgfxSurfaceCpuGenOffset. A fifth output,
	// where asked, is 1 where the block is cached, else 0.
	kRgfxAccelMethodSurfaceBlock = 0x109,
	// Lets go of one ask of the shared block at the region offset scalarInput[0]; the block goes
	// once no process holds it, through its BLOCK_RELEASE.
	kRgfxAccelMethodSurfaceBlockRelease = 0x10a,
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
	// The client's queue scalarInput[0] has its buffers up to the seqno scalarInput[1] complete,
	// their copies written back: their writes retire.
	kRgfxAccelMethodSurfaceRetire = 0x10d,
	// A pages block over the process's own memory (kRgfxKextFeaturePagesBlock;
	// RgfxPagesBlock.hpp): in the memory's address in the process, its bytes, and 1 for memory the
	// process may write; out the block's region offset and its size. The kext wires the pages that
	// hold the range, rounded out to whole pages, in the process for as long as the block lives,
	// lists their guest RAM runs in the block, and hands them to the device (SURFACE_PAGES, marked
	// RGFX_SURFACE_PAGES_PRIVATE) before its grant (BLOCK_GRANT): the device's view of them starts
	// at the range's first page, so a buffer over the range's first byte lies at that byte's
	// offset in its page (BUFFER_OVER_PAGES). The block is the client's, freed through
	// kRgfxAccelMethodFree; the process never maps it, so it never sees a guest physical address.
	// The pages are unwired once the device has answered the block's release, or the client's
	// close, and stay wired for good where neither is answered. kIOReturnUnsupported where the
	// device lacks RGFX_FEATURE_PAGES_BUFFERS or refused the pages, and for memory the process may
	// not write (0) until read-only pages have a user: the GPU's writes would land in pages the
	// process cannot write. kIOReturnBadArgument where the range is not mapped in the process;
	// kIOReturnNoSpace or kIOReturnNoResources where the region or the kext's table has no room
	// for the block.
	kRgfxAccelMethodPagesBlock = 0x10e,

	kRgfxAccelMethodContextFirst = kRgfxAccelMethodOpenContext,
	kRgfxAccelMethodContextLast = kRgfxAccelMethodPagesBlock,

	// The features: no input; out: the device's feature word (the kext's RgfxFeatures.hpp), the
	// RGFX_FEATURE_* bits 0 to 2 as the host offers them and from bit 3 the ones the framebuffer
	// took for the kext and its bundle; then the kext's own word (kRgfxKextFeature*, below). A
	// kext older than it answers kIOReturnUnsupported. The kext routes it to its context methods
	// beside the range above.
	kRgfxAccelMethodFeatures = 0x10f,
	// 0x110 is kept for SubmitBatch (kRgfxKextFeatureSubmitBatch), 0x111 for WakeCompletions
	// (kRgfxKextFeatureWakeCompletions).
	// The answers to the client's deferred pipeline creates (Submit's kRgfxSubmitDeferred) the
	// device has given (kRgfxKextFeatureDeferredAnswers): in the seqno of one to wait for first
	// (0: none), up to a minute; struct out, the answers taken, each a whole
	// RGFX_EVT_PIPELINE_REFLECTION, in their seqnos' order, as many as fit; out their count.
	// Returns success whether or not the one waited for came: the answers that did are out all
	// the same. The kext routes it to its context methods beside the range above.
	kRgfxAccelMethodTakeAnswers = 0x112,
	// A new selector takes the next number design.md's registry gives out.
};

// kRgfxAccelMethodSubmit's flags, its optional scalar in.
enum {
	// Post a pipeline create without waiting for its answer (kRgfxAccelMethodTakeAnswers;
	// kRgfxKextFeatureDeferredAnswers).
	kRgfxSubmitDeferred = 1 << 0,
};

// The kext's own features, the second word kRgfxAccelMethodFeatures answers: the user-client
// calls and flags a kext may lack, which the bundle uses only where the word holds their bit
// (design.md, "The accelerator's user client").
enum {
	// PagesBlock, 0x10e.
	kRgfxKextFeaturePagesBlock = 1 << 0,
	// SubmitBatch, 0x110.
	kRgfxKextFeatureSubmitBatch = 1 << 1,
	// WakeCompletions, 0x111.
	kRgfxKextFeatureWakeCompletions = 1 << 2,
	// The doorbell table's method 1, Consumed.
	kRgfxKextFeatureConsumedDoorbell = 1 << 3,
	// Submit's (0x105) deferred-answer flag, and TakeAnswers, 0x112.
	kRgfxKextFeatureDeferredAnswers = 1 << 4,
	// OpenContext's (0x100) sixth output, the span of the context's GPU addresses.
	kRgfxKextFeatureContextSpan = 1 << 5,
};

// Where a cached block's header holds its surface's CPU generation, a 64-bit count the kext
// writes at each read-write CPU lock (the bundle's RgfxSurfaceHeader::cpuGen).
enum {
	kRgfxSurfaceCpuGenOffset = 16,
};

#endif // RGFX_USER_CLIENT_ABI_HPP_
