// Native tests of the AVX-512 state copy (src/rgfx_xstate.h): an XSAVE image
// in AMD's layout and in Intel's, the copy to the flavor's fields and back,
// the initial-state rule both ways, and which layouts the copy takes. Build and
// run with `make test`.
#include "rgfx_xstate.h"
#include <cassert>
#include <cstdio>

// AMD's layout, as a Zen 4 core's CPUID leaf 0xD reports it (no MPX
// components between YMM_Hi128 and the opmask).
static constexpr RgfxXstateLayout kAmd = {832, 64, 896, 512, 1408, 1024};

static constexpr uint32_t kImage = 4096;

static void fill(uint8_t *p, size_t n, uint8_t seed) {
	for (size_t i = 0; i < n; i++)
		p[i] = static_cast<uint8_t>(seed + i * 7 + (i >> 8));
}

static void set_bv(uint8_t *image, uint64_t bv) {
	memcpy(image + kRgfxXsaveHeader, &bv, sizeof bv);
}

// What the hardware would write for a thread whose k, zmmh and zmm16 hold the
// given bytes: each component at the layout's offset, its XSTATE_BV bit set.
static void hardware_image(uint8_t *image, const RgfxXstateLayout &l, const uint8_t *k, const uint8_t *zmmh,
                           const uint8_t *zmm16) {
	fill(image, kImage, 0x5a);  // what earlier saves left elsewhere
	memcpy(image + l.opmask, k, kRgfxOpmaskSize);
	memcpy(image + l.zmm_hi256, zmmh, kRgfxZmmHi256Size);
	memcpy(image + l.hi16_zmm, zmm16, kRgfxHi16ZmmSize);
	uint8_t hdr[64] = {};
	memcpy(image + kRgfxXsaveHeader, hdr, sizeof hdr);
	set_bv(image, 0x7 | kRgfxXstateAvx512);
}

static void round_trip(const RgfxXstateLayout &l) {
	uint8_t k[64], zmmh[512], zmm16[1024];
	fill(k, sizeof k, 1);
	fill(zmmh, sizeof zmmh, 2);
	fill(zmm16, sizeof zmm16, 3);
	static uint8_t image[kImage];
	hardware_image(image, l, k, zmmh, zmm16);

	uint8_t k2[64], zmmh2[512], zmm162[1024];
	rgfx_xstate_to_flavor(image, l, k2, zmmh2, zmm162);
	assert(memcmp(k, k2, sizeof k) == 0);
	assert(memcmp(zmmh, zmmh2, sizeof zmmh) == 0);
	assert(memcmp(zmm16, zmm162, sizeof zmm16) == 0);

	static uint8_t back[kImage];
	fill(back, sizeof back, 0x33);
	rgfx_xstate_from_flavor(back, l, k2, zmmh2, zmm162);
	assert(rgfx_xstate_bv(back) == kRgfxXstateAvx512);
	assert(memcmp(back + l.opmask, k, sizeof k) == 0);
	assert(memcmp(back + l.zmm_hi256, zmmh, sizeof zmmh) == 0);
	assert(memcmp(back + l.hi16_zmm, zmm16, sizeof zmm16) == 0);
	// XCOMP_BV and the header's reserved bytes are zero: XRSTOR faults on
	// anything else in the standard form.
	assert(rgfx_xstate_all_zero(back + kRgfxXsaveHeader + 8, 56));
	// Nothing below the first component is left from before: the legacy region
	// is outside XRSTOR's components 5-7, but the image starts clean.
	assert(rgfx_xstate_all_zero(back, kRgfxXsaveHeader));
}

int main() {
	// --- which layouts ------------------------------------------------------
	assert(rgfx_xstate_is_intel(kRgfxXstateIntel));
	assert(!rgfx_xstate_is_intel(kAmd));
	assert(rgfx_xstate_layout_ok(kAmd, kImage));
	assert(rgfx_xstate_layout_ok(kRgfxXstateIntel, kImage));
	assert(rgfx_xstate_end(kAmd) == 2432);
	assert(rgfx_xstate_end(kRgfxXstateIntel) == 2688);
	// Everything inside XNU's save area element, Intel's 2688 bytes.
	assert(rgfx_xstate_layout_ok(kAmd, 2688));
	assert(!rgfx_xstate_layout_ok(kAmd, 2431));
	// A component short of its size, inside the header, or over another: not
	// taken.
	RgfxXstateLayout bad = kAmd;
	bad.opmask_size = 0;
	assert(!rgfx_xstate_layout_ok(bad, kImage));
	bad = kAmd;
	bad.opmask = 512;
	assert(!rgfx_xstate_layout_ok(bad, kImage));
	bad = kAmd;
	bad.zmm_hi256 = 860;
	assert(!rgfx_xstate_layout_ok(bad, kImage));
	bad = kAmd;
	bad.hi16_zmm = 1000;
	assert(!rgfx_xstate_layout_ok(bad, kImage));
	// An offset whose end wraps in 32 bits.
	bad = kAmd;
	bad.hi16_zmm = 0xffffff00u;
	assert(!rgfx_xstate_layout_ok(bad, kImage));

	// --- the copy, both layouts ---------------------------------------------
	round_trip(kAmd);
	round_trip(kRgfxXstateIntel);

	// The bug the copy replaces: Intel's offsets read in an AMD image give
	// other bytes than the registers held.
	{
		uint8_t k[64], zmmh[512], zmm16[1024];
		fill(k, sizeof k, 9);
		fill(zmmh, sizeof zmmh, 10);
		fill(zmm16, sizeof zmm16, 11);
		static uint8_t image[kImage];
		hardware_image(image, kAmd, k, zmmh, zmm16);
		assert(memcmp(image + kRgfxXstateIntel.opmask, k, sizeof k) != 0);
		assert(memcmp(image + kRgfxXstateIntel.hi16_zmm, zmm16, sizeof zmm16) != 0);
		uint8_t k2[64], zmmh2[512], zmm162[1024];
		rgfx_xstate_to_flavor(image, kAmd, k2, zmmh2, zmm162);
		assert(memcmp(k, k2, sizeof k) == 0);
		assert(memcmp(zmm16, zmm162, sizeof zmm16) == 0);
	}

	// --- the initial-state rule ---------------------------------------------
	// A component whose XSTATE_BV bit is clear reads as zeros, whatever bytes
	// the image holds there (XSAVE may leave them as they were).
	{
		uint8_t k[64], zmmh[512], zmm16[1024];
		fill(k, sizeof k, 4);
		fill(zmmh, sizeof zmmh, 5);
		fill(zmm16, sizeof zmm16, 6);
		static uint8_t image[kImage];
		hardware_image(image, kAmd, k, zmmh, zmm16);
		set_bv(image, 0x7 | kRgfxXstateZmmHi256);
		uint8_t k2[64], zmmh2[512], zmm162[1024];
		fill(k2, sizeof k2, 0xee);
		fill(zmm162, sizeof zmm162, 0xee);
		rgfx_xstate_to_flavor(image, kAmd, k2, zmmh2, zmm162);
		assert(rgfx_xstate_all_zero(k2, sizeof k2));
		assert(memcmp(zmmh, zmmh2, sizeof zmmh) == 0);
		assert(rgfx_xstate_all_zero(zmm162, sizeof zmm162));
	}
	// An all-zero component goes back in its initial state: its XSTATE_BV bit
	// clear, so XRSTOR initializes it.
	{
		uint8_t k[64] = {}, zmmh[512] = {}, zmm16[1024];
		fill(zmm16, sizeof zmm16, 7);
		static uint8_t back[kImage];
		rgfx_xstate_from_flavor(back, kAmd, k, zmmh, zmm16);
		assert(rgfx_xstate_bv(back) == kRgfxXstateHi16Zmm);
		uint8_t none[64] = {}, noneh[512] = {}, none16[1024] = {};
		rgfx_xstate_from_flavor(back, kAmd, none, noneh, none16);
		assert(rgfx_xstate_bv(back) == 0);
		// One nonzero byte is enough to keep a component.
		k[63] = 1;
		rgfx_xstate_from_flavor(back, kAmd, k, noneh, none16);
		assert(rgfx_xstate_bv(back) == kRgfxXstateOpmask);
		assert(back[kAmd.opmask + 63] == 1);
	}

	printf("xstate_test: all passed\n");
	return 0;
}
