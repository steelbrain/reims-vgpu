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

// Every bit the wire names from bit 3.
constexpr uint64_t kNew = RGFX_FEATURE_QUIET_SCHEDULED | RGFX_FEATURE_CONSUMED_DOORBELL |
                          RGFX_FEATURE_QUEUE_BARRIER | RGFX_FEATURE_CB_PARTS |
                          RGFX_FEATURE_PAGES_BUFFERS | RGFX_FEATURE_QUEUE_VECTORS;
} // namespace

int main()
{
	// The kext carries out none of the bits from 3 yet, so it writes 0 whatever the offer.
	check(kRgfxKextFeatures == 0, "the kext takes no bit from 3 yet");
	check(rgfxDriverFeatures(0) == 0, "nothing offered, nothing written");
	check(rgfxDriverFeatures(kRgfxOfferedFeatures | kNew | (1ull << 40)) == 0,
	      "every bit offered, still none written");
	check((rgfxDriverFeatures(~0ull) & kRgfxOfferedFeatures) == 0, "bits 0 to 2 are never written");

	// The word: bits 0 to 2 as offered, on a host older than the registers (DRIVER_FEATURES
	// reads 0) and on one offering every bit.
	check(rgfxFeatureWord(kRgfxOfferedFeatures, 0) == kRgfxOfferedFeatures,
	      "a wire-2 host's bits 0 to 2");
	check(rgfxFeatureWord(RGFX_FEATURE_CB_SUBMITTED, 0) == RGFX_FEATURE_CB_SUBMITTED,
	      "only the bits offered");
	check(rgfxFeatureWord(kRgfxOfferedFeatures | kNew, 0) == kRgfxOfferedFeatures,
	      "bits from 3 offered but not taken are not in the word");
	// A read-back of bits the kext never wrote, or of bits 0 to 2 the host does not offer, adds
	// nothing.
	check(rgfxFeatureWord(kRgfxOfferedFeatures, kNew | (1ull << 40)) == kRgfxOfferedFeatures,
	      "taken bits the kext does not carry out are dropped");
	check(rgfxFeatureWord(0, kRgfxOfferedFeatures) == 0,
	      "bits 0 to 2 come from the offer, not the read-back");

	// Queue flags: granted only where the word holds their feature, so none yet.
	const uint32_t both = RGFX_QUEUE_QUIET_SCHEDULED | RGFX_QUEUE_CONSUMED_DOORBELL;
	check(rgfxQueueFlagsGranted(both, rgfxFeatureWord(kRgfxOfferedFeatures | kNew, kNew)) == 0,
	      "no flag is granted while the kext takes no feature");
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
