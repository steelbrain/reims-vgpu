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
                                 RGFX_FEATURE_PAGES_BUFFERS | RGFX_FEATURE_QUEUE_VECTORS;
// The ones it does: the bundle orders its command buffers over IOSurfaces and places its Managed
// buffers' CPU copies in its heaps, neither of which needs anything of the kext, and makes
// indirect command buffers, whose creates and writes the kext admits.
constexpr uint64_t kOrder = RGFX_FEATURE_SURFACE_ORDER;
constexpr uint64_t kIcb = RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS;
constexpr uint64_t kManaged = RGFX_FEATURE_MANAGED_AT_OFFSET;
constexpr uint64_t kCarried = kOrder | kIcb | kManaged;
// Every bit the wire names from bit 3.
constexpr uint64_t kNew = kNotCarried | kCarried;
} // namespace

int main()
{
	// Of the bits from 3 the kext carries out SURFACE_ORDER, INDIRECT_COMMAND_BUFFERS and
	// MANAGED_AT_OFFSET, and writes each only where offered.
	check(kRgfxCarriedFeatures == kCarried,
	      "the kext takes SURFACE_ORDER, ICBs and MANAGED_AT_OFFSET from bit 3");
	check(RGFX_FEATURE_SURFACE_ORDER == (1u << 11), "SURFACE_ORDER is bit 11");
	check(RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS == (1u << 12), "INDIRECT_COMMAND_BUFFERS is bit 12");
	check(RGFX_FEATURE_MANAGED_AT_OFFSET == (1u << 14), "MANAGED_AT_OFFSET is bit 14");
	check(rgfxDriverFeatures(0) == 0, "nothing offered, nothing written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | (1ull << 40)) == 0,
	      "every other bit offered, none written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNew | (1ull << 40)) == kCarried,
	      "every bit offered, SURFACE_ORDER, ICBs and MANAGED_AT_OFFSET written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kOrder) == kOrder,
	      "SURFACE_ORDER offered, written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kIcb) == kIcb,
	      "ICBs offered, written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNotCarried | kManaged) == kManaged,
	      "MANAGED_AT_OFFSET offered, written");
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
	// A read-back of bits the kext never wrote, or of bits 0 to 2 the host does not offer, adds
	// nothing.
	check(rgfxFeatureWord(kRgfxOfferedFeatures, kNotCarried | (1ull << 40)) ==
	          kRgfxOfferedFeatures,
	      "taken bits the kext does not carry out are dropped");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kNew, kNew) == (kRgfxOfferedFeatures | kCarried),
	      "SURFACE_ORDER, ICBs and MANAGED_AT_OFFSET taken are in the word, for the bundle");
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

	std::printf("features_test: OK (%d checks)\n", checks);
	return 0;
}
