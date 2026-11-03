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
// the kext checks, and a full submission ring's post waits for the host's room record; the kext
// makes WindowServer's queues unshared, which needs nothing of the bundle.
constexpr uint64_t kPages = RGFX_FEATURE_PAGES_BUFFERS;
constexpr uint64_t kOrder = RGFX_FEATURE_SURFACE_ORDER;
constexpr uint64_t kIcb = RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS;
constexpr uint64_t kUnshared = RGFX_FEATURE_UNSHARED_QUEUES;
constexpr uint64_t kManaged = RGFX_FEATURE_MANAGED_AT_OFFSET;
constexpr uint64_t kHeaps = RGFX_FEATURE_HEAP_OVER_PAGES;
constexpr uint64_t kRoom = RGFX_FEATURE_SUBMIT_ROOM;
constexpr uint64_t kSlots = RGFX_FEATURE_ICB_SLOTS;
constexpr uint64_t kCarried =
    kPages | kOrder | kIcb | kUnshared | kManaged | kHeaps | kRoom | kSlots;
// Every bit the wire names from bit 3.
constexpr uint64_t kNew = kNotCarried | kCarried;
} // namespace

int main()
{
	// Of the bits from 3 the kext carries out PAGES_BUFFERS, SURFACE_ORDER,
	// INDIRECT_COMMAND_BUFFERS, UNSHARED_QUEUES, MANAGED_AT_OFFSET, HEAP_OVER_PAGES, SUBMIT_ROOM
	// and ICB_SLOTS, and writes each only where offered.
	check(kRgfxCarriedFeatures == kCarried,
	      "the kext takes PAGES_BUFFERS, SURFACE_ORDER, ICBs, UNSHARED_QUEUES, MANAGED_AT_OFFSET, "
	      "HEAP_OVER_PAGES, SUBMIT_ROOM and ICB slots from bit 3");
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
