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

// Every bit the wire names from bit 3 that the kext does not carry out yet.
constexpr uint64_t kNotCarried = RGFX_FEATURE_QUIET_SCHEDULED | RGFX_FEATURE_CONSUMED_DOORBELL |
                                 RGFX_FEATURE_QUEUE_BARRIER | RGFX_FEATURE_CB_PARTS |
                                 RGFX_FEATURE_QUEUE_VECTORS;
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
constexpr uint64_t kCarried = kPages | kOrder | kIcb | kUnshared | kManaged | kHeaps | kRoom |
                              kSlots | kCommits | kDeferred | kTime | kTextureSlabs |
                              kBufferSlabs | kSmallIds | kColor | kMesh | kShares | kOpenCL |
                              kPrivate | kCacheRefused | kDeferredAsk | kDeferredCarry;
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
	// SURFACE_DEFERRED_ASK and SURFACE_DEFERRED_CARRY, and writes each only where offered.
	check(kRgfxCarriedFeatures == kCarried,
	      "the kext takes PAGES_BUFFERS, SURFACE_ORDER, ICBs, UNSHARED_QUEUES, MANAGED_AT_OFFSET, "
	      "HEAP_OVER_PAGES, SUBMIT_ROOM, ICB slots, COMMIT_ORDER, SURFACE_DEFERRED, HOST_TIME, "
	      "MANAGED_TEXTURE_SLABS, MANAGED_BUFFER_SLABS, SMALL_ID_CHUNKS, DISPLAY_COLOR, "
	      "MESH_PIPELINES, SHARED_EVENT_NAMES, OPENCL_MODE, PRIVATE_RENDER_STATE, "
	      "SURFACE_CACHE_REFUSED, SURFACE_DEFERRED_ASK and SURFACE_DEFERRED_CARRY from bit 3");
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
	check(rgfxDriverFeatures(0) == 0, "nothing offered, nothing written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | (1ull << 40)) == 0,
	      "every other bit offered, none written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNew | (1ull << 40)) == kCarried,
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
	check(rgfxFeatureWord(kRgfxOfferedFeatures, kNotCarried | (1ull << 40)) ==
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
