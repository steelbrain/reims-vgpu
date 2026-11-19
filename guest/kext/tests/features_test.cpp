// The native test of RgfxFeatures.hpp: what the framebuffer writes to DRIVER_FEATURES, the feature
// word the accelerator answers the bundle, and the queue flags it grants; and the wire records of
// the negotiation as the kext compiles them.
#include "RgfxFeatures.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>

// The wire's layout as the kext reads it (design.md; wire.rs asserts the same).
static_assert(RGFX_REG_DRIVER_FEATURES_LO == 0x024, "DRIVER_FEATURES_LO");
static_assert(RGFX_REG_DRIVER_FEATURES_HI == 0x028, "DRIVER_FEATURES_HI");
static_assert(RGFX_REG_CONSUMED == 0x02c, "CONSUMED");
static_assert(RGFX_REG_DEFERRED_BUDGET_LO == 0x030, "DEFERRED_BUDGET_LO");
static_assert(RGFX_REG_DEFERRED_BUDGET_HI == 0x034, "DEFERRED_BUDGET_HI");
static_assert(RGFX_SURFACE_PAGES_DEFERRED == (1u << 2), "SURFACE_PAGES_DEFERRED");
static_assert(RGFX_SURFACE_PAGES_ATTACH == (1u << 3), "SURFACE_PAGES_ATTACH");
static_assert(RGFX_SURFACE_PAGES_ASK_CACHE == (1u << 4), "SURFACE_PAGES_ASK_CACHE");
static_assert(RGFX_FEATURE_SURFACE_CACHE_REFUSED == (1u << 29), "SURFACE_CACHE_REFUSED is bit 29");
static_assert(RGFX_FEATURE_SURFACE_DEFERRED_ASK == (1u << 30), "SURFACE_DEFERRED_ASK is bit 30");
static_assert(RGFX_FEATURE_SURFACE_DEFERRED_CARRY == (1ull << 32),
              "SURFACE_DEFERRED_CARRY is bit 32");
static_assert(RGFX_FEATURE_SCHEDULED_ASK == (1ull << 33), "SCHEDULED_ASK is bit 33");
static_assert(RGFX_FEATURE_PAGES_16K == (1ull << 34), "PAGES_16K is bit 34");
static_assert(RGFX_FEATURE_DISPLAY_SCALE == (1ull << 35), "DISPLAY_SCALE is bit 35");
static_assert(RGFX_FEATURE_TEXTURE_DID_MODIFY_DATA == (1ull << 36),
              "TEXTURE_DID_MODIFY_DATA is bit 36");
static_assert(RGFX_FEATURE_DISPLAY_POWER == (1ull << 37), "DISPLAY_POWER is bit 37");
static_assert(RGFX_FEATURE_RATE_MAPS == (1ull << 38), "RATE_MAPS is bit 38");
static_assert(RGFX_FEATURE_FUNCTION_POINTERS == (1ull << 39), "FUNCTION_POINTERS is bit 39");
static_assert(RGFX_FEATURE_DYNAMIC_LIBRARIES == (1ull << 40), "DYNAMIC_LIBRARIES is bit 40");
static_assert(RGFX_FEATURE_PIPELINE_KEYS == (1ull << 41), "PIPELINE_KEYS is bit 41");
static_assert(RGFX_FEATURE_ICB_SLOTS_IN_PAGES == (1ull << 42), "ICB_SLOTS_IN_PAGES is bit 42");
static_assert(RGFX_FEATURE_DOORBELL_ASKS == (1ull << 43), "DOORBELL_ASKS is bit 43");
static_assert(RGFX_FEATURE_GUEST_TIMES == (1ull << 44), "GUEST_TIMES is bit 44");
static_assert(RGFX_FEATURE_READ_ONLY_PAGES == (1ull << 45), "READ_ONLY_PAGES is bit 45");
static_assert(RGFX_FEATURE_DOORBELL_PAGES == (1ull << 46), "DOORBELL_PAGES is bit 46");
static_assert(RGFX_FEATURE_PIPELINE_ANSWERS == (1ull << 47), "PIPELINE_ANSWERS is bit 47");
static_assert(RGFX_CMD_PIPELINE_ANSWERS == 0x2f9, "PIPELINE_ANSWERS");
static_assert(sizeof(RgfxPipelineAnswersQuery) == 24, "PIPELINE_ANSWERS' query");
static_assert(offsetof(RgfxPipelineAnswersQuery, first) == 16, "the query's first");
static_assert(offsetof(RgfxPipelineAnswersQuery, most) == 20, "the query's most");
static_assert(RGFX_DOORBELL_ASKS == 1, "an asking doorbell sets bit 0 of the ring's id");
static_assert(sizeof(RgfxDisplayScale) == 8, "the display's scale record");
static_assert(RGFX_STATUS_CACHE == 7, "STATUS_CACHE");
static_assert(RGFX_SURFACE_PAGES_LAYOUT_OFFSET ==
                  RGFX_SURFACE_PAGES_CACHE_OFFSET + sizeof(RgfxSurfacePagesCache),
              "the layout follows the cache's part");
static_assert(sizeof(RgfxSurfacePagesLayout) == 136, "SURFACE_PAGES' layout");
static_assert(offsetof(RgfxRingHeader, producer_waiting) == 0x10, "producer_waiting");
static_assert(offsetof(RgfxRingHeader, raise_all) == 0xc4, "raise_all");
static_assert(RGFX_QUEUE_CREATE_FLAGS_OFFSET == sizeof(RgfxQueueCreate), "the tail's offset");
static_assert(sizeof(RgfxQueueCreate) + sizeof(RgfxQueueCreateFlags) == 64, "a create with a tail");
static_assert(RGFX_FEATURE_UNSHARED_QUEUES == (1u << 13), "UNSHARED_QUEUES is bit 13");
static_assert(RGFX_QUEUE_UNSHARED == (1u << 3), "RGFX_QUEUE_UNSHARED is flag 3");
static_assert(sizeof(RgfxBufferOverPages) == 72, "BUFFER_OVER_PAGES");
static_assert(sizeof(RgfxCommandBufferPart) == 56, "COMMAND_BUFFER_PART");
static_assert(offsetof(RgfxDeviceInfo, sampler_reduction_support) == 0x452,
              "sampler_reduction_support");

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "features_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

// PAGES_16K, which the arm64e kext carries (this test built with RGFX_GUEST_ARM64 set, as
// features_arm64_test) and the x86-64 kext does not.
constexpr uint64_t k16K = RGFX_FEATURE_PAGES_16K;
constexpr uint64_t k16KCarried = RGFX_GUEST_ARM64 ? k16K : 0;
// DISPLAY_POWER, which the x86-64 kext carries, its framebuffer telling the host when the
// display's panel goes dark and lights, and the arm64e kext does not.
constexpr uint64_t kDisplayPower = RGFX_FEATURE_DISPLAY_POWER;
constexpr uint64_t kDisplayPowerCarried = RGFX_GUEST_ARM64 ? 0 : kDisplayPower;
// MULTI_DISPLAY, which both kexts carry, a framebuffer of its own driving each display the device
// lists; the arm64e kext only where the firmware placed a node for display 1 too.
constexpr uint64_t kMultiDisplay = RGFX_FEATURE_MULTI_DISPLAY;
constexpr uint64_t kMultiDisplayCarried = kMultiDisplay;
// POINTER_OVER, which both kexts carry, moving the cursor onto the display the host's pointer is
// over; the arm64e kext only with MULTI_DISPLAY.
constexpr uint64_t kPointerOver = RGFX_FEATURE_POINTER_OVER;
constexpr uint64_t kPointerOverCarried = kPointerOver;
// DISPLAY_BOUNDS, which both kexts carry, telling the host where each display lies in
// WindowServer's global space; the arm64e kext only with MULTI_DISPLAY.
constexpr uint64_t kDisplayBounds = RGFX_FEATURE_DISPLAY_BOUNDS;
constexpr uint64_t kDisplayBoundsCarried = kDisplayBounds;
// POINTER_MOTION, which both kexts carry, publishing the guest's own mouse for it.
constexpr uint64_t kPointerMotion = RGFX_FEATURE_POINTER_MOTION;
// VT_MESSAGES, which the arm64e kext carries, publishing the VideoToolbox service for it, and
// the x86-64 kext does not.
constexpr uint64_t kVtMessages = RGFX_FEATURE_VT_MESSAGES;
constexpr uint64_t kVtMessagesCarried = RGFX_GUEST_ARM64 ? kVtMessages : 0;
// UPLOADS_WRITTEN, which both kexts carry, their bundle naming its Managed upload chunks' bytes in
// each command buffer's stream.
constexpr uint64_t kUploadsWritten = RGFX_FEATURE_UPLOADS_WRITTEN;
// VT_RING, which the arm64e kext carries with VT_MESSAGES, taking the host's messages from the
// VideoToolbox ring, and the x86-64 kext does not.
constexpr uint64_t kVtRing = RGFX_FEATURE_VT_RING;
constexpr uint64_t kVtRingCarried = RGFX_GUEST_ARM64 ? kVtRing : 0;
// CURSOR_COLOR, which the arm64e kext carries, posting the cursor layer's ICC chain, and the
// x86-64 kext does not.
constexpr uint64_t kCursorColor = RGFX_FEATURE_CURSOR_COLOR;
constexpr uint64_t kCursorColorCarried = RGFX_GUEST_ARM64 ? kCursorColor : 0;
// POOL, which both kexts carry, mapping the Shared pool's blocks into their processes.
constexpr uint64_t kPool = RGFX_FEATURE_POOL;
// POOL_ADDRESSES, which both kexts carry beside POOL, handing their processes the GPU address the
// host gave each block.
constexpr uint64_t kPoolAddresses = RGFX_FEATURE_POOL_ADDRESSES;
// PRIVATE_BLOCKS, which both kexts carry, handing their processes the Private blocks the host
// gives their contexts.
constexpr uint64_t kPrivateBlocks = RGFX_FEATURE_PRIVATE_BLOCKS;
// IMPORT_ADDRESSES, which both kexts carry, handing their processes the GPU address the host gave
// each pages block's view.
constexpr uint64_t kImportAddresses = RGFX_FEATURE_IMPORT_ADDRESSES;
// A bit the wire does not name.
constexpr uint64_t kUnnamed = 1ull << 63;
// Every bit the wire names from bit 3 that the kext does not carry out yet.
constexpr uint64_t kNotCarried = RGFX_FEATURE_QUIET_SCHEDULED | RGFX_FEATURE_CONSUMED_DOORBELL |
                                 RGFX_FEATURE_QUEUE_BARRIER | RGFX_FEATURE_CB_PARTS |
                                 RGFX_FEATURE_QUEUE_VECTORS | (k16K & ~k16KCarried) |
                                 (kDisplayPower & ~kDisplayPowerCarried) |
                                 (kMultiDisplay & ~kMultiDisplayCarried) |
                                 (kPointerOver & ~kPointerOverCarried) |
                                 (kDisplayBounds & ~kDisplayBoundsCarried) |
                                 (kVtMessages & ~kVtMessagesCarried) |
                                 (kVtRing & ~kVtRingCarried) |
                                 (kCursorColor & ~kCursorColorCarried);
// The ones it does: the bundle lays no-copy buffers over pages blocks the kext makes, orders its
// command buffers over IOSurfaces and places its Managed buffers' CPU copies in its heaps, neither
// of which needs anything of the kext, makes indirect command buffers, whose creates and writes,
// as records or as slots, the kext admits, makes its slabs' heaps over pages blocks, whose blocks
// the kext checks, a full submission ring's post waits for the host's room record, and each post
// names the buffers of the process's other queues posted before it, which needs nothing of the
// kext; the kext makes WindowServer's queues unshared, which needs nothing of the bundle, and
// defers fresh cached surfaces' blocks; the bundle lays its Managed textures' and buffers'
// CPU copies in its slabs, which needs nothing of the kext; and it makes mesh render pipelines,
// whose creates the kext admits as the other pipeline creates.
constexpr uint64_t kPages = RGFX_FEATURE_PAGES_BUFFERS;
constexpr uint64_t kOrder = RGFX_FEATURE_SURFACE_ORDER;
constexpr uint64_t kIcb = RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS;
constexpr uint64_t kUnshared = RGFX_FEATURE_UNSHARED_QUEUES;
constexpr uint64_t kManaged = RGFX_FEATURE_MANAGED_AT_OFFSET;
constexpr uint64_t kHeaps = RGFX_FEATURE_HEAP_OVER_PAGES;
constexpr uint64_t kRoom = RGFX_FEATURE_SUBMIT_ROOM;
constexpr uint64_t kSlots = RGFX_FEATURE_ICB_SLOTS;
constexpr uint64_t kCommits = RGFX_FEATURE_COMMIT_ORDER;
constexpr uint64_t kDeferred = RGFX_FEATURE_SURFACE_DEFERRED;
constexpr uint64_t kTime = RGFX_FEATURE_HOST_TIME;
constexpr uint64_t kTextureSlabs = RGFX_FEATURE_MANAGED_TEXTURE_SLABS;
constexpr uint64_t kBufferSlabs = RGFX_FEATURE_MANAGED_BUFFER_SLABS;
constexpr uint64_t kSmallIds = RGFX_FEATURE_SMALL_ID_CHUNKS;
constexpr uint64_t kColor = RGFX_FEATURE_DISPLAY_COLOR;
constexpr uint64_t kMesh = RGFX_FEATURE_MESH_PIPELINES;
constexpr uint64_t kShares = RGFX_FEATURE_SHARED_EVENT_NAMES;
constexpr uint64_t kOpenCL = RGFX_FEATURE_OPENCL_MODE;
constexpr uint64_t kPrivate = RGFX_FEATURE_PRIVATE_RENDER_STATE;
// The kext asks a host whose GPU shares the CPU's memory whether a new cacheable block in place
// needs a device cache, with the layout its bundle gives.
constexpr uint64_t kCacheRefused = RGFX_FEATURE_SURFACE_CACHE_REFUSED;
// And defers a fresh one there, asking with that layout.
constexpr uint64_t kDeferredAsk = RGFX_FEATURE_SURFACE_DEFERRED_ASK;
// And defers one fresh but for the resident pages its bundle lists, carrying them.
constexpr uint64_t kDeferredCarry = RGFX_FEATURE_SURFACE_DEFERRED_CARRY;
// Its bundle asks for a waited-for command buffer's scheduling.
constexpr uint64_t kScheduledAsk = RGFX_FEATURE_SCHEDULED_ASK;
// It gives the OS the display's density from its host screen's scale.
constexpr uint64_t kDisplayScale = RGFX_FEATURE_DISPLAY_SCALE;
// Its bundle tells the host a texture's didModifyData.
constexpr uint64_t kDidModify = RGFX_FEATURE_TEXTURE_DID_MODIFY_DATA;
// Its bundle makes, resets and asks rasterization rate maps.
constexpr uint64_t kRateMaps = RGFX_FEATURE_RATE_MAPS;
// Its bundle makes a compute pipeline's function handles and visible function tables.
constexpr uint64_t kFunctionPointers = RGFX_FEATURE_FUNCTION_POINTERS;
// Its bundle makes dynamic libraries and names them in a compute pipeline's create.
constexpr uint64_t kDynamicLibraries = RGFX_FEATURE_DYNAMIC_LIBRARIES;
// Its bundle asks pipeline creates' answers with their keys, which the kext keeps for every process.
constexpr uint64_t kPipelineKeys = RGFX_FEATURE_PIPELINE_KEYS;
// Its bundle writes indirect command buffers' slots in chunked pages blocks of the kext's.
constexpr uint64_t kSlotsInPages = RGFX_FEATURE_ICB_SLOTS_IN_PAGES;
// Its bundle sets bit 0 of a doorbell's value where the buffer asks for its scheduling.
constexpr uint64_t kDoorbellAsks = RGFX_FEATURE_DOORBELL_ASKS;
// Its bundle stamps each command buffer with the guest's clock, and its framebuffer gives the host
// the clocks' offset.
constexpr uint64_t kGuestTimes = RGFX_FEATURE_GUEST_TIMES;
// Its kext makes pages blocks of memory a process maps read-only, and its bundle lays its no-copy
// buffers over read-only memory there.
constexpr uint64_t kReadOnlyPages = RGFX_FEATURE_READ_ONLY_PAGES;
// Its kext binds a doorbell page to each context and maps it into the process, where its bundle
// rings its queues.
constexpr uint64_t kDoorbellPages = RGFX_FEATURE_DOORBELL_PAGES;
// Its framebuffer seeds the answers kept for every process with the device's of earlier runs.
constexpr uint64_t kPipelineAnswers = RGFX_FEATURE_PIPELINE_ANSWERS;
// Its bundle tells the host, once, that a no-copy buffer's identity left with no message.
constexpr uint64_t kBufferEscaped = RGFX_FEATURE_BUFFER_ESCAPED;
// Its bundle asks the host to fill a cached surface's device cache again after a CPU's unlock.
constexpr uint64_t kSurfaceRefill = RGFX_FEATURE_SURFACE_REFILL;
constexpr uint64_t kCarried = kPages | kOrder | kIcb | kUnshared | kManaged | kHeaps | kRoom |
                              kSlots | kCommits | kDeferred | kTime | kTextureSlabs |
                              kBufferSlabs | kSmallIds | kColor | kMesh | kShares | kOpenCL |
                              kPrivate | kCacheRefused | kDeferredAsk | kDeferredCarry |
                              kScheduledAsk | kDisplayScale | kDidModify | kRateMaps |
                              kFunctionPointers | kDynamicLibraries | kPipelineKeys |
                              kSlotsInPages | kDoorbellAsks | kGuestTimes | kReadOnlyPages |
                              kDoorbellPages | kPipelineAnswers | kDisplayPowerCarried |
                              kMultiDisplayCarried | kPointerOverCarried | kDisplayBoundsCarried |
                              kPointerMotion | k16KCarried | kVtMessagesCarried | kUploadsWritten |
                              kVtRingCarried | kCursorColorCarried | kBufferEscaped |
                              kPool | kSurfaceRefill | kPoolAddresses | kPrivateBlocks |
                              kImportAddresses;
// Every bit the wire names from bit 3.
constexpr uint64_t kNew = kNotCarried | kCarried;
} // namespace

int main()
{
	// Of the bits from 3 the kext carries out PAGES_BUFFERS, SURFACE_ORDER,
	// INDIRECT_COMMAND_BUFFERS, UNSHARED_QUEUES, MANAGED_AT_OFFSET, HEAP_OVER_PAGES, SUBMIT_ROOM,
	// ICB_SLOTS, COMMIT_ORDER, SURFACE_DEFERRED, HOST_TIME, MANAGED_TEXTURE_SLABS,
	// MANAGED_BUFFER_SLABS, SMALL_ID_CHUNKS, DISPLAY_COLOR, MESH_PIPELINES,
	// SHARED_EVENT_NAMES, OPENCL_MODE, PRIVATE_RENDER_STATE, SURFACE_CACHE_REFUSED,
	// SURFACE_DEFERRED_ASK, SURFACE_DEFERRED_CARRY, SCHEDULED_ASK, DISPLAY_SCALE,
	// TEXTURE_DID_MODIFY_DATA, RATE_MAPS, FUNCTION_POINTERS, DYNAMIC_LIBRARIES, PIPELINE_KEYS,
	// ICB_SLOTS_IN_PAGES, DOORBELL_ASKS, GUEST_TIMES, READ_ONLY_PAGES, DOORBELL_PAGES,
	// PIPELINE_ANSWERS, POINTER_MOTION, MULTI_DISPLAY, POINTER_OVER, DISPLAY_BOUNDS,
	// UPLOADS_WRITTEN, BUFFER_ESCAPED, SURFACE_REFILL, POOL, POOL_ADDRESSES, PRIVATE_BLOCKS,
	// IMPORT_ADDRESSES and, the x86-64 kext, DISPLAY_POWER, or, the arm64e kext, PAGES_16K,
	// VT_MESSAGES, VT_RING and CURSOR_COLOR, and writes each only where offered.
	check(kRgfxCarriedFeatures == kCarried,
	      "the kext takes PAGES_BUFFERS, SURFACE_ORDER, ICBs, UNSHARED_QUEUES, MANAGED_AT_OFFSET, "
	      "HEAP_OVER_PAGES, SUBMIT_ROOM, ICB slots, COMMIT_ORDER, SURFACE_DEFERRED, HOST_TIME, "
	      "MANAGED_TEXTURE_SLABS, MANAGED_BUFFER_SLABS, SMALL_ID_CHUNKS, DISPLAY_COLOR, "
	      "MESH_PIPELINES, SHARED_EVENT_NAMES, OPENCL_MODE, PRIVATE_RENDER_STATE, "
	      "SURFACE_CACHE_REFUSED, SURFACE_DEFERRED_ASK, SURFACE_DEFERRED_CARRY, SCHEDULED_ASK, "
	      "DISPLAY_SCALE, TEXTURE_DID_MODIFY_DATA, RATE_MAPS, FUNCTION_POINTERS, DYNAMIC_LIBRARIES, "
	      "PIPELINE_KEYS, ICB_SLOTS_IN_PAGES, DOORBELL_ASKS, GUEST_TIMES, READ_ONLY_PAGES, "
	      "DOORBELL_PAGES, PIPELINE_ANSWERS, POINTER_MOTION, MULTI_DISPLAY, POINTER_OVER, "
	      "DISPLAY_BOUNDS, UPLOADS_WRITTEN, BUFFER_ESCAPED, SURFACE_REFILL, POOL, POOL_ADDRESSES, "
	      "PRIVATE_BLOCKS, IMPORT_ADDRESSES and, on x86-64, DISPLAY_POWER, on arm64e PAGES_16K, "
	      "VT_MESSAGES, VT_RING and CURSOR_COLOR, from bit 3");
	check(RGFX_FEATURE_PAGES_BUFFERS == (1u << 7), "PAGES_BUFFERS is bit 7");
	check(RGFX_FEATURE_SURFACE_ORDER == (1u << 11), "SURFACE_ORDER is bit 11");
	check(RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS == (1u << 12), "INDIRECT_COMMAND_BUFFERS is bit 12");
	check(RGFX_FEATURE_UNSHARED_QUEUES == (1u << 13), "UNSHARED_QUEUES is bit 13");
	check(RGFX_FEATURE_MANAGED_AT_OFFSET == (1u << 14), "MANAGED_AT_OFFSET is bit 14");
	check(RGFX_FEATURE_HEAP_OVER_PAGES == (1u << 15), "HEAP_OVER_PAGES is bit 15");
	check(RGFX_FEATURE_SUBMIT_ROOM == (1u << 16), "SUBMIT_ROOM is bit 16");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kRoom) == kRoom,
	      "SUBMIT_ROOM offered, written");
	check(RGFX_FEATURE_ICB_SLOTS == (1u << 17), "ICB_SLOTS is bit 17");
	check(RGFX_FEATURE_COMMIT_ORDER == (1u << 18), "COMMIT_ORDER is bit 18");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kCommits) == kCommits,
	      "COMMIT_ORDER offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kCommits, kCommits) ==
	          (kRgfxOfferedFeatures | kCommits),
	      "COMMIT_ORDER taken is in the word");
	check(RGFX_FEATURE_SURFACE_DEFERRED == (1u << 19), "SURFACE_DEFERRED is bit 19");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kDeferred) == kDeferred,
	      "SURFACE_DEFERRED offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kDeferred, 0) == kRgfxOfferedFeatures,
	      "SURFACE_DEFERRED offered but not taken is not in the word");
	check(RGFX_FEATURE_HOST_TIME == (1u << 20), "HOST_TIME is bit 20");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kTime) == kTime,
	      "HOST_TIME offered, written");
	check(RGFX_FEATURE_MANAGED_TEXTURE_SLABS == (1u << 21), "MANAGED_TEXTURE_SLABS is bit 21");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kTextureSlabs) == kTextureSlabs,
	      "MANAGED_TEXTURE_SLABS offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kTextureSlabs, kTextureSlabs) ==
	          (kRgfxOfferedFeatures | kTextureSlabs),
	      "MANAGED_TEXTURE_SLABS taken is in the word");
	check(RGFX_FEATURE_MANAGED_BUFFER_SLABS == (1u << 22), "MANAGED_BUFFER_SLABS is bit 22");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kBufferSlabs) == kBufferSlabs,
	      "MANAGED_BUFFER_SLABS offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kBufferSlabs, kBufferSlabs) ==
	          (kRgfxOfferedFeatures | kBufferSlabs),
	      "MANAGED_BUFFER_SLABS taken is in the word");
	check(RGFX_FEATURE_SMALL_ID_CHUNKS == (1u << 23), "SMALL_ID_CHUNKS is bit 23");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kSmallIds) == kSmallIds,
	      "SMALL_ID_CHUNKS offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kSmallIds, kSmallIds) ==
	          (kRgfxOfferedFeatures | kSmallIds),
	      "SMALL_ID_CHUNKS taken is in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures, rgfxDriverFeatures(kRgfxOfferedFeatures)) ==
	          kRgfxOfferedFeatures,
	      "SMALL_ID_CHUNKS not offered is not in the word");
	check(RGFX_FEATURE_DISPLAY_COLOR == (1u << 24), "DISPLAY_COLOR is bit 24");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kColor) == kColor,
	      "DISPLAY_COLOR offered, written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried) == 0,
	      "DISPLAY_COLOR not offered, not written");
	check(RGFX_FEATURE_MESH_PIPELINES == (1u << 25), "MESH_PIPELINES is bit 25");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kMesh) == kMesh,
	      "MESH_PIPELINES offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kMesh, 0) == kRgfxOfferedFeatures,
	      "MESH_PIPELINES offered but not taken is not in the word");
	check(RGFX_FEATURE_SHARED_EVENT_NAMES == (1u << 26), "SHARED_EVENT_NAMES is bit 26");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kShares) == kShares,
	      "SHARED_EVENT_NAMES offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kShares, 0) == kRgfxOfferedFeatures,
	      "SHARED_EVENT_NAMES offered but not taken is not in the word");
	check(RGFX_FEATURE_OPENCL_MODE == (1u << 27), "OPENCL_MODE is bit 27");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kOpenCL) == kOpenCL,
	      "OPENCL_MODE offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kOpenCL, 0) == kRgfxOfferedFeatures,
	      "OPENCL_MODE offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kOpenCL, kOpenCL) == (kRgfxOfferedFeatures | kOpenCL),
	      "OPENCL_MODE taken is in the word");
	check(RGFX_FEATURE_PRIVATE_RENDER_STATE == (1u << 28), "PRIVATE_RENDER_STATE is bit 28");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kPrivate) == kPrivate,
	      "PRIVATE_RENDER_STATE offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kPrivate, 0) == kRgfxOfferedFeatures,
	      "PRIVATE_RENDER_STATE offered but not taken is not in the word");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kCacheRefused) == kCacheRefused,
	      "SURFACE_CACHE_REFUSED offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kCacheRefused, 0) == kRgfxOfferedFeatures,
	      "SURFACE_CACHE_REFUSED offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kCacheRefused, kCacheRefused) ==
	          (kRgfxOfferedFeatures | kCacheRefused),
	      "SURFACE_CACHE_REFUSED taken is in the word");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kCacheRefused | kDeferredAsk) ==
	          (kCacheRefused | kDeferredAsk),
	      "SURFACE_DEFERRED_ASK offered beside SURFACE_CACHE_REFUSED, both written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kCacheRefused | kDeferredAsk, kCacheRefused) ==
	          (kRgfxOfferedFeatures | kCacheRefused),
	      "SURFACE_DEFERRED_ASK offered but not taken is not in the word");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kDeferred | kDeferredCarry) ==
	          (kDeferred | kDeferredCarry),
	      "SURFACE_DEFERRED_CARRY offered beside SURFACE_DEFERRED, both written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kDeferred | kDeferredCarry, kDeferred) ==
	          (kRgfxOfferedFeatures | kDeferred),
	      "SURFACE_DEFERRED_CARRY offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kDeferred | kDeferredCarry,
	                      kDeferred | kDeferredCarry) ==
	          (kRgfxOfferedFeatures | kDeferred | kDeferredCarry),
	      "SURFACE_DEFERRED_CARRY taken is in the word");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kScheduledAsk) == kScheduledAsk,
	      "SCHEDULED_ASK offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kScheduledAsk, 0) == kRgfxOfferedFeatures,
	      "SCHEDULED_ASK offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kScheduledAsk, kScheduledAsk) ==
	          (kRgfxOfferedFeatures | kScheduledAsk),
	      "SCHEDULED_ASK taken is in the word");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kDisplayScale) == kDisplayScale,
	      "DISPLAY_SCALE offered, written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried) == 0,
	      "DISPLAY_SCALE not offered, not written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kDidModify) == kDidModify,
	      "TEXTURE_DID_MODIFY_DATA offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kDidModify, 0) == kRgfxOfferedFeatures,
	      "TEXTURE_DID_MODIFY_DATA offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kDidModify, kDidModify) ==
	          (kRgfxOfferedFeatures | kDidModify),
	      "TEXTURE_DID_MODIFY_DATA taken is in the word");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kRateMaps) == kRateMaps,
	      "RATE_MAPS offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kRateMaps, 0) == kRgfxOfferedFeatures,
	      "RATE_MAPS offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kRateMaps, kRateMaps) ==
	          (kRgfxOfferedFeatures | kRateMaps),
	      "RATE_MAPS taken is in the word");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kFunctionPointers) ==
	          kFunctionPointers,
	      "FUNCTION_POINTERS offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kFunctionPointers, 0) == kRgfxOfferedFeatures,
	      "FUNCTION_POINTERS offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kFunctionPointers, kFunctionPointers) ==
	          (kRgfxOfferedFeatures | kFunctionPointers),
	      "FUNCTION_POINTERS taken is in the word");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kDynamicLibraries) ==
	          kDynamicLibraries,
	      "DYNAMIC_LIBRARIES offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kDynamicLibraries, 0) == kRgfxOfferedFeatures,
	      "DYNAMIC_LIBRARIES offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kDynamicLibraries, kDynamicLibraries) ==
	          (kRgfxOfferedFeatures | kDynamicLibraries),
	      "DYNAMIC_LIBRARIES taken is in the word");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kPipelineKeys) == kPipelineKeys,
	      "PIPELINE_KEYS offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kPipelineKeys, 0) == kRgfxOfferedFeatures,
	      "PIPELINE_KEYS offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kPipelineKeys, kPipelineKeys) ==
	          (kRgfxOfferedFeatures | kPipelineKeys),
	      "PIPELINE_KEYS taken is in the word");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kSlotsInPages) == kSlotsInPages,
	      "ICB_SLOTS_IN_PAGES offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kSlotsInPages, 0) == kRgfxOfferedFeatures,
	      "ICB_SLOTS_IN_PAGES offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kSlotsInPages, kSlotsInPages) ==
	          (kRgfxOfferedFeatures | kSlotsInPages),
	      "ICB_SLOTS_IN_PAGES taken is in the word");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kDoorbellAsks) == kDoorbellAsks,
	      "DOORBELL_ASKS offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kDoorbellAsks, 0) == kRgfxOfferedFeatures,
	      "DOORBELL_ASKS offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kDoorbellAsks, kDoorbellAsks) ==
	          (kRgfxOfferedFeatures | kDoorbellAsks),
	      "DOORBELL_ASKS taken is in the word");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kGuestTimes) == kGuestTimes,
	      "GUEST_TIMES offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kGuestTimes, 0) == kRgfxOfferedFeatures,
	      "GUEST_TIMES offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kGuestTimes, kGuestTimes) ==
	          (kRgfxOfferedFeatures | kGuestTimes),
	      "GUEST_TIMES taken is in the word");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kReadOnlyPages) == kReadOnlyPages,
	      "READ_ONLY_PAGES offered, written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kReadOnlyPages, 0) == kRgfxOfferedFeatures,
	      "READ_ONLY_PAGES offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kReadOnlyPages, kReadOnlyPages) ==
	          (kRgfxOfferedFeatures | kReadOnlyPages),
	      "READ_ONLY_PAGES taken is in the word");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kDoorbellPages) == kDoorbellPages,
	      "DOORBELL_PAGES offered, written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried) == 0,
	      "DOORBELL_PAGES not offered (a host that maps no BAR 4), not written");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kDoorbellPages, 0) == kRgfxOfferedFeatures,
	      "DOORBELL_PAGES offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kDoorbellPages, kDoorbellPages) ==
	          (kRgfxOfferedFeatures | kDoorbellPages),
	      "DOORBELL_PAGES taken is in the word");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kPipelineAnswers) ==
	          kPipelineAnswers,
	      "PIPELINE_ANSWERS offered, written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kPipelineKeys) == kPipelineKeys,
	      "PIPELINE_ANSWERS not offered, not written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kDisplayPower) ==
	          kDisplayPowerCarried,
	      "DISPLAY_POWER offered, written by the x86-64 kext alone");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried) == 0,
	      "DISPLAY_POWER not offered, not written");
	check(RGFX_FEATURE_MULTI_DISPLAY == (1ull << 48), "MULTI_DISPLAY is bit 48");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kMultiDisplay) ==
	          kMultiDisplayCarried,
	      "MULTI_DISPLAY offered, written");
	// On arm64, only where the firmware placed nodes for displays 0 and 1: with display 0's alone,
	// or none, the word written is the one before the arm64e kext carried the bit. The x86-64 kext
	// counts no nodes and writes it wherever offered.
	{
		const uint64_t offered = kRgfxOfferedFeatures | kNotCarried | kMultiDisplay | kScheduledAsk;
		const uint64_t before = kScheduledAsk;
		const uint64_t lone = RGFX_GUEST_ARM64 ? before : before | kMultiDisplay;
		check(rgfxDriverFeatures(offered, 0) == lone, "no display node: MULTI_DISPLAY on x86-64 alone");
		check(rgfxDriverFeatures(offered, 1) == lone,
		      "display 0's node alone: MULTI_DISPLAY on x86-64 alone, the arm64 word as before");
		for (uint32_t nodes = 2; nodes <= RGFX_DISPLAYS_MAX; nodes++)
			check(rgfxDriverFeatures(offered, nodes) == (before | kMultiDisplay),
			      "nodes for displays 0 and 1: MULTI_DISPLAY written");
		check(rgfxDriverFeatures(offered & ~kMultiDisplay, 2) == before,
		      "MULTI_DISPLAY not offered: not written, whatever the nodes");
	}
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kMultiDisplay, kMultiDisplay) ==
	          (kRgfxOfferedFeatures | kMultiDisplayCarried),
	      "MULTI_DISPLAY taken is in the word where carried");
	check(RGFX_FEATURE_POINTER_OVER == (1ull << 49), "POINTER_OVER is bit 49");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kPointerOver) ==
	          kPointerOverCarried,
	      "POINTER_OVER offered, written");
	check(RGFX_FEATURE_DISPLAY_BOUNDS == (1ull << 50), "DISPLAY_BOUNDS is bit 50");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kDisplayBounds) ==
	          kDisplayBoundsCarried,
	      "DISPLAY_BOUNDS offered, written");
	check(RGFX_FEATURE_POINTER_MOTION == (1ull << 51), "POINTER_MOTION is bit 51");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kPointerMotion) == kPointerMotion,
	      "POINTER_MOTION offered, written by both kexts");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried) == 0,
	      "POINTER_MOTION not offered, not written: the guest publishes no mouse");
	// On arm64, POINTER_OVER and DISPLAY_BOUNDS go with MULTI_DISPLAY: with display 0's node
	// alone, or none, none of the three is written, so the word is the one before the arm64e kext
	// carried them. The x86-64 kext writes them wherever offered.
	{
		const uint64_t pointer = kMultiDisplay | kPointerOver | kDisplayBounds;
		const uint64_t offered = kRgfxOfferedFeatures | kNotCarried | pointer | kScheduledAsk;
		const uint64_t lone = RGFX_GUEST_ARM64 ? kScheduledAsk : kScheduledAsk | pointer;
		check(rgfxDriverFeatures(offered, 0) == lone,
		      "no display node: the pointer's bits on x86-64 alone");
		check(rgfxDriverFeatures(offered, 1) == lone,
		      "display 0's node alone: the pointer's bits on x86-64 alone");
		for (uint32_t nodes = 2; nodes <= RGFX_DISPLAYS_MAX; nodes++)
			check(rgfxDriverFeatures(offered, nodes) == (kScheduledAsk | pointer),
			      "nodes for displays 0 and 1: MULTI_DISPLAY, POINTER_OVER and "
			      "DISPLAY_BOUNDS written");
		check(rgfxDriverFeatures(kRgfxOfferedFeatures | kScheduledAsk, 2) == kScheduledAsk,
		      "none of them offered (a device of one display): the word as before");
	}
	check(RGFX_FEATURE_VT_MESSAGES == (1ull << 52), "VT_MESSAGES is bit 52");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kVtMessages) ==
	          kVtMessagesCarried,
	      "VT_MESSAGES offered, written by the arm64e kext alone");
	check(RGFX_FEATURE_UPLOADS_WRITTEN == (1ull << 53), "UPLOADS_WRITTEN is bit 53");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kUploadsWritten) ==
	          kUploadsWritten,
	      "UPLOADS_WRITTEN offered, written by both kexts");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kUploadsWritten, 0) == kRgfxOfferedFeatures,
	      "UPLOADS_WRITTEN offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kUploadsWritten, kUploadsWritten) ==
	          (kRgfxOfferedFeatures | kUploadsWritten),
	      "UPLOADS_WRITTEN taken is in the word");
	check(RGFX_FEATURE_VT_RING == (1ull << 54), "VT_RING is bit 54");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kVtMessages | kVtRing) ==
	          (kVtMessagesCarried | kVtRingCarried),
	      "VT_MESSAGES and VT_RING offered, both written by the arm64e kext alone");
	check(RGFX_FEATURE_CURSOR_COLOR == (1ull << 55), "CURSOR_COLOR is bit 55");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kCursorColor) ==
	          kCursorColorCarried,
	      "CURSOR_COLOR offered, written by the arm64e kext alone");
	check(RGFX_FEATURE_BUFFER_ESCAPED == (1ull << 56), "BUFFER_ESCAPED is bit 56");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kBufferEscaped) ==
	          kBufferEscaped,
	      "BUFFER_ESCAPED offered, written by both kexts");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kBufferEscaped, 0) == kRgfxOfferedFeatures,
	      "BUFFER_ESCAPED offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kBufferEscaped, kBufferEscaped) ==
	          (kRgfxOfferedFeatures | kBufferEscaped),
	      "BUFFER_ESCAPED taken is in the word");
	check(RGFX_FEATURE_POOL == (1ull << 57), "POOL is bit 57");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kPool) == kPool,
	      "POOL offered, written by both kexts");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kPool, 0) == kRgfxOfferedFeatures,
	      "POOL offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kPool, kPool) == (kRgfxOfferedFeatures | kPool),
	      "POOL taken is in the word");
	check(RGFX_FEATURE_SURFACE_REFILL == (1ull << 58), "SURFACE_REFILL is bit 58");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kSurfaceRefill) ==
	          kSurfaceRefill,
	      "SURFACE_REFILL offered, written by both kexts");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kSurfaceRefill, 0) == kRgfxOfferedFeatures,
	      "SURFACE_REFILL offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kSurfaceRefill, kSurfaceRefill) ==
	          (kRgfxOfferedFeatures | kSurfaceRefill),
	      "SURFACE_REFILL taken is in the word");
	check(RGFX_FEATURE_POOL_ADDRESSES == (1ull << 59), "POOL_ADDRESSES is bit 59");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kPoolAddresses) ==
	          kPoolAddresses,
	      "POOL_ADDRESSES offered, written by both kexts");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kPoolAddresses, 0) == kRgfxOfferedFeatures,
	      "POOL_ADDRESSES offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kPoolAddresses, kPoolAddresses) ==
	          (kRgfxOfferedFeatures | kPoolAddresses),
	      "POOL_ADDRESSES taken is in the word");
	check(RGFX_FEATURE_IMPORT_ADDRESSES == (1ull << 61), "IMPORT_ADDRESSES is bit 61");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kImportAddresses) ==
	          kImportAddresses,
	      "IMPORT_ADDRESSES offered, written by both kexts");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kImportAddresses, 0) == kRgfxOfferedFeatures,
	      "IMPORT_ADDRESSES offered but not taken is not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kImportAddresses, kImportAddresses) ==
	          (kRgfxOfferedFeatures | kImportAddresses),
	      "IMPORT_ADDRESSES taken is in the word");
	// PAGES_16K: the arm64e kext writes it first where offered, and then with the rest; the
	// x86-64 kext never writes it, and writes nothing first.
	check(rgfxDriverFeaturesFirst(~0ull) == k16KCarried, "PAGES_16K first where carried");
	check(rgfxDriverFeaturesFirst(~0ull & ~k16K) == 0, "nothing first where PAGES_16K is not offered");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kScheduledAsk | k16K) ==
	          (kScheduledAsk | k16KCarried),
	      "PAGES_16K offered, written where carried");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | k16K, k16K) == (kRgfxOfferedFeatures | k16KCarried),
	      "PAGES_16K taken is in the word where carried");
	check(rgfxDriverFeatures(0) == 0, "nothing offered, nothing written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kUnnamed) == 0,
	      "every other bit offered, none written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNew | kUnnamed) == kCarried,
	      "every bit offered, the carried bits written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kPages) == kPages,
	      "PAGES_BUFFERS offered, written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kOrder) == kOrder,
	      "SURFACE_ORDER offered, written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kIcb) == kIcb,
	      "ICBs offered, written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kUnshared) == kUnshared,
	      "UNSHARED_QUEUES offered, written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kManaged) == kManaged,
	      "MANAGED_AT_OFFSET offered, written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kHeaps) == kHeaps,
	      "HEAP_OVER_PAGES offered, written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kIcb | kSlots) == (kIcb | kSlots),
	      "ICBs and their slots offered, both written");
	check(rgfxDriverFeatures(~0ull) == kCarried, "everything offered, the carried bits written");
	check((rgfxDriverFeatures(~0ull) & kRgfxOfferedFeatures) == 0, "bits 0 to 2 are never written");

	// The word: bits 0 to 2 as offered, on a host older than the registers (DRIVER_FEATURES
	// reads 0) and on one offering every bit.
	check(rgfxFeatureWord(kRgfxOfferedFeatures, 0) == kRgfxOfferedFeatures,
	      "a wire-2 host's bits 0 to 2");
	check(rgfxFeatureWord(RGFX_FEATURE_CB_SUBMITTED, 0) == RGFX_FEATURE_CB_SUBMITTED,
	      "only the bits offered");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kNew | kOrder, 0) == kRgfxOfferedFeatures,
	      "bits from 3 offered but not taken are not in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kOrder, kOrder) == (kRgfxOfferedFeatures | kOrder),
	      "SURFACE_ORDER taken is in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kCarried, kManaged) ==
	          (kRgfxOfferedFeatures | kManaged),
	      "MANAGED_AT_OFFSET taken is in the word, the carried bits offered but not taken are not");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kUnshared, kUnshared) ==
	          (kRgfxOfferedFeatures | kUnshared),
	      "UNSHARED_QUEUES taken is in the word");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kRoom, kRoom) == (kRgfxOfferedFeatures | kRoom),
	      "SUBMIT_ROOM taken is in the word, for the bundle's queue rings");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kRoom, 0) == kRgfxOfferedFeatures,
	      "SUBMIT_ROOM offered but not taken is not");
	// A read-back of bits the kext never wrote, or of bits 0 to 2 the host does not offer, adds
	// nothing.
	check(rgfxFeatureWord(kRgfxOfferedFeatures, kNotCarried | kUnnamed) ==
	          kRgfxOfferedFeatures,
	      "taken bits the kext does not carry out are dropped");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kNew, kNew) == (kRgfxOfferedFeatures | kCarried),
	      "the carried bits taken are in the word, for the bundle");
	check(rgfxFeatureWord(0, kRgfxOfferedFeatures) == 0,
	      "bits 0 to 2 come from the offer, not the read-back");

	// Queue flags: granted only where the word holds their feature, so none: the kext takes
	// neither queue feature.
	const uint32_t both = RGFX_QUEUE_QUIET_SCHEDULED | RGFX_QUEUE_CONSUMED_DOORBELL;
	check(rgfxQueueFlagsGranted(both, rgfxFeatureWord(kRgfxOfferedFeatures | kNew, kNew)) == 0,
	      "no flag is granted while the kext takes no queue feature");
	check(rgfxQueueFlagsGranted(both, RGFX_FEATURE_QUIET_SCHEDULED) == RGFX_QUEUE_QUIET_SCHEDULED,
	      "QUIET_SCHEDULED with its feature");
	check(rgfxQueueFlagsGranted(both, RGFX_FEATURE_CONSUMED_DOORBELL) ==
	          RGFX_QUEUE_CONSUMED_DOORBELL,
	      "CONSUMED_DOORBELL with its feature");
	check(rgfxQueueFlagsGranted(both | (1u << 5), kNew) == both, "an unknown flag is not granted");
	check(rgfxQueueFlagsGranted(0, kNew) == 0, "nothing wanted, nothing granted");

	// RGFX_QUEUE_UNSHARED is the kext's own: a process that asks for it is not granted it, with or
	// without the feature, and the rest of what it asks is granted as before.
	const uint64_t taken = kRgfxOfferedFeatures | kUnshared;
	check(rgfxQueueFlagsGranted(RGFX_QUEUE_UNSHARED, taken) == 0,
	      "a process asking for RGFX_QUEUE_UNSHARED is not granted it");
	check(rgfxQueueFlagsGranted(RGFX_QUEUE_UNSHARED, ~0ull) == 0,
	      "nor with every feature in the word");
	check(rgfxQueueFlagsGranted(RGFX_QUEUE_UNSHARED | RGFX_QUEUE_QUIET_SCHEDULED,
	                            taken | RGFX_FEATURE_QUIET_SCHEDULED) == RGFX_QUEUE_QUIET_SCHEDULED,
	      "its other flags still granted");

	// What QUEUE_CREATE carries: WindowServer's queues unshared where the word holds the
	// feature; nothing for any other process's, whatever it asked; nothing on a host that did not
	// take it, so no tail goes there.
	check(rgfxQueueFlagsSent(0, true, taken) == RGFX_QUEUE_UNSHARED, "WindowServer's queue");
	check(rgfxQueueFlagsSent(0, false, taken) == 0, "another process's queue");
	check(rgfxQueueFlagsSent(rgfxQueueFlagsGranted(RGFX_QUEUE_UNSHARED, taken), false, taken) == 0,
	      "another process asking for it");
	check(rgfxQueueFlagsSent(0, true, kRgfxOfferedFeatures) == 0,
	      "a host that did not take UNSHARED_QUEUES");
	check(rgfxQueueFlagsSent(0, true, rgfxFeatureWord(kRgfxOfferedFeatures, 0)) == 0,
	      "a wire-2 host");
	check(rgfxQueueFlagsSent(RGFX_QUEUE_QUIET_SCHEDULED, true, taken) ==
	          (RGFX_QUEUE_QUIET_SCHEDULED | RGFX_QUEUE_UNSHARED),
	      "beside the flags granted");
	check(rgfxQueueFlagsSent(RGFX_QUEUE_QUIET_SCHEDULED, false, taken) ==
	          RGFX_QUEUE_QUIET_SCHEDULED,
	      "the flags granted alone for another process");

	std::printf("features_test: OK (%d checks)\n", checks);
	return 0;
}
