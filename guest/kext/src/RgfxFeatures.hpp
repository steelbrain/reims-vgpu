// RgfxFeatures: the device's feature bits as the kext negotiates them (design.md, "The wire's
// version").
//
// The host offers its bits in FEATURES_LO and FEATURES_HI. Bits 0 to 2 need no answer: the host
// acts on them as it offers them, and the kext and the bundle use them where offered. From bit 3,
// the host acts on a bit only where the guest's driver took it: the framebuffer writes
// DRIVER_FEATURES_LO and _HI after its reset and before DRIVER_READY with the bits the kext and its
// bundle carry out, masked by the offer, and reads back the ones the host took. A host older than
// the registers reads them as 0, so nothing from bit 3 is taken there. Of those bits the kext and
// its bundle carry out fifteen:
//  - PAGES_BUFFERS (bit 7): the kext makes pages blocks over a process's own memory
//    (kRgfxAccelMethodPagesBlock), and the bundle lays its no-copy buffers over them
//    (BUFFER_OVER_PAGES);
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
//  - HEAP_OVER_PAGES (bit 15): the bundle makes its slabs' heaps over pages blocks of one run, its
//    superpages (HEAP_OVER_PAGES, which Submit checks names a block of the process's, as it does
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
//    holders give them on a queue of their own.
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
    RGFX_FEATURE_MESH_PIPELINES | RGFX_FEATURE_SHARED_EVENT_NAMES;

// What the framebuffer writes to DRIVER_FEATURES, given the host's offer.
inline uint64_t rgfxDriverFeatures(uint64_t offered)
{
	return offered & kRgfxCarriedFeatures;
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
