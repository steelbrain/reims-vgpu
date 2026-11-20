// RgfxFeatures: the device's feature bits as the kext negotiates them (design.md, "The wire's
// version").
//
// The host offers its bits in FEATURES_LO and FEATURES_HI. Bits 0 to 2 need no answer: the host
// acts on them as it offers them, and the kext and the bundle use them where offered. From bit 3,
// the host acts on a bit only where the guest's driver took it: the framebuffer writes
// DRIVER_FEATURES_LO and _HI after its reset and before DRIVER_READY with the bits the kext and its
// bundle carry out, masked by the offer, and reads back the ones the host took. A host older than
// the registers reads them as 0, so nothing from bit 3 is taken there. Of those bits the kext and
// its bundle carry out these:
//  - PAGES_BUFFERS (bit 7): the kext makes pages blocks over a process's own memory
//    (kRgfxAccelMethodPagesBlock), and the bundle lays its no-copy buffers over them
//    (BUFFER_OVER_PAGES);
//  - IMPORT_ADDRESSES_LATER (bit 9), both kexts, read with IMPORT_ADDRESSES: the device answers a
//    pages block before its view has an address, with none (rgfxPagesAnswer), the bundle makes
//    its buffers over such a block at the host's address (BUFFER_OVER_PAGES at gpu_address 0),
//    and asks a buffer's address where it needs it (RGFX_CMD_PAGES_BUFFER_ADDRESS), a query the
//    kext admits as any (rgfxSubmitAdmitted) and answers with the device's COMPLETE;
//  - SURFACE_ORDER (bit 11): the bundle names, as it commits a command buffer that uses
//    IOSurfaces, the buffers of other queues it runs after (design.md, "Surface order"), which
//    needs nothing of the kext;
//  - INDIRECT_COMMAND_BUFFERS (bit 12): the kext admits ICB_CREATE and ICB_WRITE, answers a create
//    with its ICB_STATE and destroys behind one not answered in time, and the bundle makes
//    indirect command buffers only where the word holds the bit;
//  - UNSHARED_QUEUES (bit 13): the accelerator flags each queue WindowServer's process creates
//    RGFX_QUEUE_UNSHARED (rgfxQueueFlagsSent), so that the host gives it a GPU queue no other
//    process's queue shares (design.md, "WindowServer's queues are unshared"), which needs nothing
//    of the bundle;
//  - MANAGED_AT_OFFSET (bit 14): the bundle places its Managed buffers' CPU copies in heaps over
//    its blocks (BUFFER_MANAGED_AT_OFFSET, which Submit posts as it posts BUFFER_AT_OFFSET), which
//    needs nothing more of the kext;
//  - HEAP_OVER_PAGES (bit 15): the bundle makes its slabs' heaps over pages blocks of its own pages
//    (HEAP_OVER_PAGES, which Submit checks names a block of the process's, as it does
//    BUFFER_OVER_PAGES);
//  - SUBMIT_ROOM (bit 16): a post that finds a submission ring full sets the ring's
//    producer_waiting and waits for the host's SUBMIT_ROOM record: the framebuffer's on ring 0,
//    the bundle's on its queues' rings.
//  - ICB_SLOTS (bit 17), meaningful only with bit 12: the bundle writes its indirect command
//    buffers' CPU encodes as slots in its region blocks and posts ICB_SLOTS, a write into an
//    object made that the kext admits as any (rgfxSubmitAdmitted), its obj_id the first run's
//    buffer;
//  - COMMIT_ORDER (bit 18): the bundle names, as it posts a command buffer, the latest buffer each
//    other queue of its process posted since (design.md, "Commit order"), which needs nothing of
//    the kext;
//  - SURFACE_DEFERRED (bit 19), meaningful only with SURFACE_CACHE: the kext defers a fresh
//    cached surface's block in place, naming its pages at the first CPU lock that needs them
//    (RgfxSurfaceBlocks.hpp), within the budget the host gives in DEFERRED_BUDGET_LO and _HI,
//    which needs nothing of the bundle;
//  - HOST_TIME (bit 20): at each VBLANK the framebuffer reads the host's clock (HOST_TIME_HI, _LO
//    and _HI again, LO's read between two of mach_absolute_time) and maps the refresh's stamp by
//    the reading whose round trip bounds the two clocks' offset closest (RgfxVblankClock.hpp),
//    which needs nothing of the bundle;
//  - MANAGED_TEXTURE_SLABS (bit 21): the bundle lays its Managed textures' CPU copies in its
//    buffers' slabs, its own pages among them, and names a slab's buffer as each copy, which needs
//    nothing of the kext;
//  - MANAGED_BUFFER_SLABS (bit 22): the bundle lays its Managed buffers' CPU copies in its Shared
//    buffers' slabs, its own pages among them, and names a slab's buffer as each copy
//    (MANAGED_BUFFER_CREATE, which Submit posts as it posts MANAGED_TEXTURE_CREATE), which needs
//    nothing more of the kext;
//  - SMALL_ID_CHUNKS (bit 23): the kext grants the texture resource ids in chunks of
//    RGFX_RESOURCE_ID_SMALL_CHUNK (RESOURCE_ID_SMALL_CHUNK, RgfxIdChunks.hpp), and the bundle
//    takes its ids from chunks of that size;
//  - DISPLAY_COLOR (bit 24): the display pipe takes its gamma selectors (5 to 7) and sends each
//    transaction's gamma tables and colour matrix as DISPLAY_COLOR parts before its present, a
//    transaction that sets only a colour as a present that keeps the surface shown
//    (RgfxPipeColor.hpp), which needs nothing of the bundle;
//  - MESH_PIPELINES (bit 25): the bundle makes mesh render pipelines and draws them, whose
//    MESH_RENDER_PIPELINE_CREATE the kext admits as it does the other pipeline creates, its
//    PIPELINE_REFLECTION answer carrying the mesh stages' limits after the record
//    (RgfxSubmitGate.hpp);
//  - SHARED_EVENT_NAMES (bit 26): a process hands its shared event to another through a port of
//    the kext's (RgfxSharedEventUserClient.hpp), for which the kext posts SHARED_EVENT_NAME,
//    SHARED_EVENT_IMPORT and SHARED_EVENT_UNNAME, and the bundle's events take the values other
//    holders give them on a queue of their own;
//  - OPENCL_MODE (bit 27): the bundle makes the GL renderer's OpenCL kernels' pipelines of OpenCL
//    mode (RGFX_COMPUTE_PIPELINE_OPENCL_MODE), which needs nothing of the kext;
//  - PRIVATE_RENDER_STATE (bit 28): the bundle's render pipeline creates carry their descriptors'
//    private state (RGFX_RENDER_PIPELINE_PRIVATE_STATE), which needs nothing of the kext;
//  - SURFACE_CACHE_REFUSED (bit 29), which a host whose GPU shares the CPU's memory offers: the
//    kext asks the host, at a new cacheable block in place, whether its surface needs a device
//    cache (RGFX_SURFACE_PAGES_ASK_CACHE, with the layout the bundle gives with its ask), and
//    makes the block cached where the host answers RGFX_STATUS_CACHE, its cached blocks then kept
//    as SURFACE_CACHE's are (RgfxSurfaceCaches.hpp);
//  - UNCHANGING_PAGES (bit 31), both kexts, meaningful only with READ_ONLY_PAGES: the bundle names
//    read-only memory no one writes, a file's pages on a read-only volume, which the app maps
//    copy-on-write with no copy made yet or with no write allowed, or anonymous memory with a copy
//    pending and no write in its maximum protection (the bundle's RgfxFilePages.hpp), and the
//    kext makes a pages block of that mapping marked unchanging (RGFX_SURFACE_PAGES_UNCHANGING,
//    and RGFX_SURFACE_PAGES_COPY_ON_WRITE where the app's mapping is copy-on-write;
//    kRgfxPagesUnchanging, RgfxPagesBlock.hpp), for the host to read its pages once;
//  - SURFACE_DEFERRED_CARRY (bit 32), meaningful only with SURFACE_DEFERRED: the bundle lists, at
//    its ask of a cacheable block in place, the few resident pages of its surface it finds in its
//    mapping, and the kext defers a surface fresh but for those, carrying them in the deferral's
//    runs once its read of the surface proves them (RgfxDeferCarry.hpp);
//  - SCHEDULED_ASK (bit 33): a thread of the bundle's that waits for a command buffer to be
//    scheduled (waitUntilScheduled) posts RGFX_CMD_SCHEDULED_ASK on the buffer's queue's ring,
//    which the bundle writes itself, so it needs nothing of the kext;
//  - PAGES_16K (bit 34), the arm64e kext alone (RGFX_GUEST_ARM64): its runs of guest RAM are whole
//    16 KiB pages, as its kernel's pages are. A host whose page is 16 KiB shows its features over
//    guest RAM (SURFACE_PAGES, PAGES_BUFFERS, HEAP_OVER_PAGES and those that go with them) only
//    once the guest took it (design.md, "Negotiated features"), so the framebuffer first writes
//    DRIVER_FEATURES_HI with it where offered (rgfxDriverFeaturesFirst), reads the offer again,
//    and then writes both halves from that offer. The x86-64 kext never takes it, and reads the
//    offer once, as before the bit;
//  - DISPLAY_SCALE (bit 35): the device's DISPLAYS answer carries, after each display's record,
//    the scale of the host screen that shows it (RgfxDisplayScale), which the framebuffer gives
//    the OS as the display's density (RgfxDisplayIdentity.hpp: the x86-64 framebuffer's EDID and
//    HiDPI modes, the arm64e framebuffer's DisplayAttributes and dot pitch), which needs nothing
//    of the bundle;
//  - TEXTURE_DID_MODIFY_DATA (bit 36): the bundle tells the host a texture's didModifyData
//    (RGFX_CMD_TEXTURE_DID_MODIFY_DATA), whose message the kext admits as it does a Managed
//    range's (RgfxSubmitGate.hpp), so a mipmapped texture over a buffer takes the bytes again;
//  - DISPLAY_POWER (bit 37), the x86-64 kext alone: the framebuffer tells the host, with
//    SET_DISPLAY_POWER, as the display's panel goes dark at the display's sleep and lights at its
//    wake, from the display wrangler's power messages (RgfxDisplayPower.hpp), which needs nothing
//    of the bundle; an arm64 guest has no display wrangler;
//  - RATE_MAPS (bit 38): the bundle makes, resets and asks rasterization rate maps
//    (RGFX_CMD_RATE_MAP_CREATE, _RESET and _QUERY), whose messages the kext admits as any create
//    or query of the client's own objects (RgfxSubmitGate.hpp), waiting for their RATE_MAP_STATE
//    or RATE_MAP_ANSWER, and names a map in a render pass's begin;
//  - FUNCTION_POINTERS (bit 39): the bundle makes a compute pipeline's function handles and
//    visible function tables (RGFX_CMD_FUNCTION_HANDLE_CREATE and _TABLE_CREATE), waiting for
//    their FUNCTION_POINTER_STATE, and sets a table's entries (RGFX_CMD_FUNCTION_TABLE_SET,
//    unanswered), which the kext admits as any create or write of the client's own objects
//    (RgfxSubmitGate.hpp), and binds tables in a compute pass's ops;
//  - DYNAMIC_LIBRARIES (bit 40): the bundle makes dynamic libraries of its libraries
//    (RGFX_CMD_DYNAMIC_LIBRARY_CREATE), which the kext admits as any create of the client's own
//    objects (RgfxSubmitGate.hpp), waiting for its COMPLETE, and names them at the end of a
//    compute pipeline's create;
//  - PIPELINE_KEYS (bit 41): the bundle asks each render and compute pipeline create's answer
//    with its pipeline key, and the kext keeps the answers that carry one for every process
//    (RgfxSharedPipelines.hpp), so an equal create of another process need not wait;
//  - ICB_SLOTS_IN_PAGES (bit 42): the bundle writes an indirect command buffer's slots in a
//    chunked pages block of the kext's (PagesChunked), cached memory of the process's own, and
//    names them in RGFX_CMD_ICB_SLOTS_IN_PAGES, which the kext admits as it does ICB_SLOTS, a
//    write into the client's own objects (RgfxSubmitGate.hpp);
//  - DOORBELL_ASKS (bit 43): the bundle rings a queue's doorbell for a command buffer that asks
//    for its CB_SUBMITTED with the ring's id and RGFX_DOORBELL_ASKS, and for anything else with
//    the id alone (RgfxRingWriter::setDoorbellAsks), so the host may take the plain ones with no
//    exit to its process; the kext writes what the bundle hands its doorbell call, so it needs
//    nothing of the kext;
//  - GUEST_TIMES (bit 44), meaningful only with HOST_TIME: the bundle starts each command buffer's
//    stream with RGFX_OP_GUEST_TIMES, the guest's clock as the buffer was made, committed and put
//    on its ring, and the framebuffer writes, after each VBLANK's reading of the host's clock, the
//    reading's offset and bound in ring 0's header (RgfxVblankClock::offsetAt), for the host to
//    split the time before it takes each buffer: a diagnostic;
//  - READ_ONLY_PAGES (bit 45), meaningful only with PAGES_BUFFERS: the kext makes pages blocks of
//    memory the process maps read-only too, wired for reading alone and marked so in their
//    SURFACE_PAGES (RGFX_SURFACE_PAGES_READ_ONLY, rgfxPagesWiring), for the host to keep every
//    object over them from writing their pages, and the bundle lays its no-copy buffers over
//    read-only memory there with MTLResourceOptions bit 24, whose GPU writes the host drops;
//  - DOORBELL_PAGES (bit 46): the accelerator binds a doorbell page of BAR 4 to each context it
//    opens for a bundle that asks for one (RgfxDoorbellPages.hpp) and maps it into the context's
//    process, where the bundle writes its queues' ring ids with no call into the kext, and the
//    kext's own publish of a queue's ring (RgfxQueuePublish.hpp) rings there too;
//  - PIPELINE_ANSWERS (bit 47), meaningful only with PIPELINE_KEYS: the framebuffer's start
//    seeds the answers it keeps for every process (RgfxSharedPipelines.hpp) with those the device
//    kept of keyed creates of its earlier runs (RGFX_CMD_PIPELINE_ANSWERS), so a create of a key
//    an earlier boot made need not wait either;
//  - MULTI_DISPLAY (bit 48): where the device has two displays or more, its DISPLAYS answer lists
//    them all and it sends each one's events, and the framebuffer drives each display with a
//    framebuffer of its own (ReimsGraphics.hpp, RgfxDisplayTable.hpp), which needs nothing of the
//    bundle: an IOFramebuffer on x86-64, an IOMobileFramebuffer on arm64. The arm64e kext takes it
//    only where the firmware placed a node under /arm-io for display 1 as well as display 0
//    (rgfxDriverFeatures' `mobileNodes`): a process waits for a service of each display node it
//    counts, and counts nodes before the kext can add any (design.md, "Multiple displays"). With
//    display 0's node alone, the arm64e kext drives display 0 alone, as before the bit;
//  - POINTER_OVER (bit 49), meaningful only with MULTI_DISPLAY, and taken by the arm64e kext only
//    with it: on the device's POINTER_OVER the framebuffer moves the cursor onto the display the
//    host's pointer is over (ReimsGraphicsPointer.cpp, RgfxPointerOver.hpp; on arm64 by the
//    places its own moves showed, RgfxDisplayPlaces.hpp), which needs nothing of the bundle;
//  - DISPLAY_BOUNDS (bit 50), meaningful only with POINTER_OVER, and taken by the arm64e kext only
//    with MULTI_DISPLAY: each framebuffer's move of the cursor tells the host every display's
//    rectangle in WindowServer's global space that changed (RgfxDisplayBounds.hpp; on arm64 each
//    one a landing of its own moves confirmed), by which the host carries its captured pointer
//    past one display's edge onto the display the guest placed there, which needs nothing of the
//    bundle;
//  - POINTER_MOTION (bit 51), both kexts: the framebuffer publishes the guest's own mouse
//    (RgfxPointerDevice.hpp) and hands it each POINTER_MOTION's reports
//    (RgfxPointerMotion.hpp), which needs nothing of the bundle;
//  - VT_MESSAGES (bit 52), the arm64e kext alone: the framebuffer publishes the VideoToolbox
//    service where the host's caps list a decoder, carries its clients' messages to the host as
//    VT_MESSAGEs and hands them the host's VT_MESSAGE events (RgfxVtService.hpp), which needs
//    nothing of the bundle; an x86-64 guest's hardware video is AppleGVA's (RgfxVideoCodecs.hpp);
//  - UPLOADS_WRITTEN (bit 53), both kexts: where the bundle's upload chunks are Managed (its
//    device answers hasUnifiedMemory NO), each command buffer names the bytes its encoders wrote
//    in them in its own stream (RGFX_OP_UPLOADS_WRITTEN, RgfxUploads.h), in place of a
//    BUFFER_DID_MODIFY_RANGE of each on ring 0, which needs nothing of the kext;
//  - VT_RING (bit 54), the arm64e kext alone, with VT_MESSAGES: the framebuffer takes the host's
//    messages from the VideoToolbox ring through each VT_RING event's count (RgfxVtRing.hpp) and
//    hands them on as VT_MESSAGE events', which needs nothing of the bundle;
//  - CURSOR_COLOR (bit 55), the arm64e kext alone: set_block takes the cursor layer's ICC chain
//    and the display's swaps post it to the host (CURSOR_COLOR, RgfxMobileICC.hpp), which converts
//    the cursor's image through it, which needs nothing of the bundle; an x86-64 guest gives its
//    cursor no chain;
//  - BUFFER_ESCAPED (bit 56), both kexts: the bundle tells the host, once, that a no-copy buffer
//    the host reads in place handed its identity out with no message naming it
//    (RGFX_CMD_BUFFER_ESCAPED), whose message the kext admits, and takes in a batch, as it does a
//    Managed range's (RgfxSubmitGate.hpp);
//  - POOL (bit 57), both kexts: the accelerator reads the Shared pool's range
//    (POOL_BASE_LO to POOL_LENGTH_HI) and maps the blocks the host gives a context's POOL_TAKE
//    into its process (kRgfxAccelMethodPoolTake, RgfxPoolBlocks.hpp), and the bundle lays its
//    buffers' slabs in them, a heap over each (HEAP_OVER_POOL), which the kext admits as any
//    create of the client's own objects (RgfxSubmitGate.hpp);
//  - SURFACE_REFILL (bit 58), both kexts: the bundle tracks a cached surface's unlock seed from a
//    commit made while a CPU held it locked for writing (kRgfxAccelMethodSurfaceSeeds), and has
//    the next commit after the unlock ask the host to fill the surface's device cache again
//    (RGFX_SURFACE_USE_REFILL), an op of its command buffers', which needs nothing of the kext;
//  - POOL_ADDRESSES (bit 59), both kexts, with POOL: the kext answers the GPU address
//    the host gives each pool block (kRgfxAccelMethodPoolTake's fourth output), where the bundle
//    makes the block's heap;
//  - PRIVATE_BLOCKS (bit 60), both kexts: the kext posts a context's PRIVATE_TAKE and PRIVATE_GIVE
//    (kRgfxAccelMethodPrivateTake, kRgfxAccelMethodPrivateGive) and answers each block's GPU
//    address and length, where the bundle places its Private buffers, its heaps that are not
//    Shared and its Managed buffers' GPU copies;
//  - SURFACE_CACHE_EVERY (bit 62), both kexts, read with SURFACE_CACHE: the host's GPU form of
//    every IOSurface is its device cache, so the accelerator names every block in place cached and
//    refuses, before it posts anything, an ask in place it cannot cache; its tables of the
//    surfaces' blocks, of a block's holders and of the cached surfaces grow rather than refuse;
//    and it counts each copy block it makes, a surface's second copy beside its device cache, and
//    each ask in place it refuses, by why, as the bundle counts each copy view by why it is not in
//    place (RgfxIOSurface.mm);
//  - MANAGED_AT_COPY (bit 63), both kexts: on a GPU that shares the CPU's memory the bundle makes
//    each Managed buffer at its CPU copy's address, with no Private block pick of its own
//    (RgfxSerializer::createManagedBuffer), which needs nothing of the kext.
//
// The accelerator answers the bundle's Features call with the feature word (rgfxFeatureWord), and
// grants a queue's flags (kRgfxAccelMethodQueueCreate's seventh input) only where their feature
// was taken (rgfxQueueFlagsGranted). RGFX_QUEUE_UNSHARED is never granted to a process that asks
// for it: the kext alone sets it, from the kernel's own view of the client's task.
//
// Header-only and free of the kernel's headers, so its native test (tests/features_test.cpp)
// builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_FEATURES_HPP_
#define RGFX_FEATURES_HPP_

#include "reims_graphics_wire.h"

#ifndef RGFX_GUEST_ARM64
#define RGFX_GUEST_ARM64 0
#endif

// The bits the host acts on as it offers them, with no write: bits 0 to 2.
constexpr uint64_t kRgfxOfferedFeatures =
    RGFX_FEATURE_SURFACE_PAGES | RGFX_FEATURE_SURFACE_CACHE | RGFX_FEATURE_CB_SUBMITTED;

// The bits from bit 3 the kext and its bundle carry out, which the framebuffer writes to
// DRIVER_FEATURES where the host offers them.
constexpr uint64_t kRgfxCarriedFeatures =
    RGFX_FEATURE_PAGES_BUFFERS | RGFX_FEATURE_SURFACE_ORDER |
    RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS | RGFX_FEATURE_UNSHARED_QUEUES |
    RGFX_FEATURE_MANAGED_AT_OFFSET | RGFX_FEATURE_HEAP_OVER_PAGES | RGFX_FEATURE_SUBMIT_ROOM |
    RGFX_FEATURE_ICB_SLOTS | RGFX_FEATURE_COMMIT_ORDER | RGFX_FEATURE_SURFACE_DEFERRED |
    RGFX_FEATURE_HOST_TIME | RGFX_FEATURE_MANAGED_TEXTURE_SLABS |
    RGFX_FEATURE_MANAGED_BUFFER_SLABS | RGFX_FEATURE_SMALL_ID_CHUNKS | RGFX_FEATURE_DISPLAY_COLOR |
    RGFX_FEATURE_MESH_PIPELINES | RGFX_FEATURE_SHARED_EVENT_NAMES | RGFX_FEATURE_OPENCL_MODE |
    RGFX_FEATURE_PRIVATE_RENDER_STATE | RGFX_FEATURE_SURFACE_CACHE_REFUSED |
    RGFX_FEATURE_SURFACE_DEFERRED_ASK | RGFX_FEATURE_UNCHANGING_PAGES |
    RGFX_FEATURE_SURFACE_DEFERRED_CARRY |
    RGFX_FEATURE_SCHEDULED_ASK | RGFX_FEATURE_DISPLAY_SCALE |
    RGFX_FEATURE_TEXTURE_DID_MODIFY_DATA | RGFX_FEATURE_RATE_MAPS |
    RGFX_FEATURE_FUNCTION_POINTERS | RGFX_FEATURE_DYNAMIC_LIBRARIES | RGFX_FEATURE_PIPELINE_KEYS |
    RGFX_FEATURE_ICB_SLOTS_IN_PAGES | RGFX_FEATURE_DOORBELL_ASKS | RGFX_FEATURE_GUEST_TIMES |
    RGFX_FEATURE_READ_ONLY_PAGES | RGFX_FEATURE_DOORBELL_PAGES | RGFX_FEATURE_PIPELINE_ANSWERS |
    RGFX_FEATURE_POINTER_MOTION | RGFX_FEATURE_MULTI_DISPLAY |
    RGFX_FEATURE_POINTER_OVER | RGFX_FEATURE_DISPLAY_BOUNDS | RGFX_FEATURE_UPLOADS_WRITTEN |
    RGFX_FEATURE_BUFFER_ESCAPED | RGFX_FEATURE_SURFACE_REFILL |
    (RGFX_GUEST_ARM64 ? RGFX_FEATURE_PAGES_16K | RGFX_FEATURE_VT_MESSAGES | RGFX_FEATURE_VT_RING |
                            RGFX_FEATURE_CURSOR_COLOR
                      : RGFX_FEATURE_DISPLAY_POWER) |
    RGFX_FEATURE_POOL | RGFX_FEATURE_POOL_ADDRESSES | RGFX_FEATURE_PRIVATE_BLOCKS |
    RGFX_FEATURE_IMPORT_ADDRESSES | RGFX_FEATURE_IMPORT_ADDRESSES_LATER |
    RGFX_FEATURE_SURFACE_CACHE_EVERY | RGFX_FEATURE_MANAGED_AT_COPY;

// What the framebuffer writes to DRIVER_FEATURES, given the host's offer and, on arm64, the
// displays whose node the firmware placed under /arm-io, counted from display 0 (`mobileNodes`):
// MULTI_DISPLAY, and POINTER_OVER and DISPLAY_BOUNDS, which mean nothing without it, only where
// display 1 has one too. The x86-64 kext reads no nodes.
inline uint64_t rgfxDriverFeatures(uint64_t offered, uint32_t mobileNodes = RGFX_DISPLAYS_MAX)
{
	uint64_t carried = kRgfxCarriedFeatures;
	if (RGFX_GUEST_ARM64 && mobileNodes < 2)
		carried &= ~static_cast<uint64_t>(RGFX_FEATURE_MULTI_DISPLAY | RGFX_FEATURE_POINTER_OVER |
		                                  RGFX_FEATURE_DISPLAY_BOUNDS);
	return offered & carried;
}

// What the framebuffer writes to DRIVER_FEATURES_HI before it reads the offer again, given the
// host's first offer: PAGES_16K where the kext carries it (the arm64e kext) and the host offers
// it, else nothing, and the offer is read once.
inline uint64_t rgfxDriverFeaturesFirst(uint64_t offered)
{
	return offered & kRgfxCarriedFeatures & RGFX_FEATURE_PAGES_16K;
}

// The feature word the bundle works by: bits 0 to 2 as offered, and from bit 3 the ones the host
// took (DRIVER_FEATURES read back), of those the kext carries out.
inline uint64_t rgfxFeatureWord(uint64_t offered, uint64_t taken)
{
	return (offered & kRgfxOfferedFeatures) | (taken & kRgfxCarriedFeatures);
}

// The flags of `wanted` (RGFX_QUEUE_*) a queue gets, given the feature word: each one whose
// feature was taken. A queue granted none is created without the flags tail, as a wire-2 kext
// creates it.
inline uint32_t rgfxQueueFlagsGranted(uint32_t wanted, uint64_t word)
{
	uint32_t allowed = 0;
	if (word & RGFX_FEATURE_QUIET_SCHEDULED)
		allowed |= RGFX_QUEUE_QUIET_SCHEDULED;
	if (word & RGFX_FEATURE_CONSUMED_DOORBELL)
		allowed |= RGFX_QUEUE_CONSUMED_DOORBELL;
	return wanted & allowed;
}

// The flags a queue's QUEUE_CREATE carries: those `granted` to its process
// (rgfxQueueFlagsGranted), and RGFX_QUEUE_UNSHARED for a queue of WindowServer's (`windowServer`,
// the client's task as initWithTask found it) where the feature word holds UNSHARED_QUEUES. A
// host that does not offer the feature never sees the flag, and a create with no flag goes
// without the tail, as a wire-2 kext sends it.
inline uint32_t rgfxQueueFlagsSent(uint32_t granted, bool windowServer, uint64_t word)
{
	if (windowServer && (word & RGFX_FEATURE_UNSHARED_QUEUES))
		granted |= RGFX_QUEUE_UNSHARED;
	return granted;
}

#endif // RGFX_FEATURES_HPP_
