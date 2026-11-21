// Pure parts of RgfxGate's AVX-512 state copy (RgfxGateXstate.cpp), shared
// with the native tests: where an XSAVE image holds the AVX-512 state
// components, and the copy between such an image and the AVX-512 fields of a
// thread state flavor.
//
// XNU's fpu_get_fxstate and fpu_set_fxstate copy a thread's k0-7, the upper
// halves of zmm0-15 and zmm16-31 at the offsets of Intel's standard XSAVE
// layout (struct x86_avx512_thread_state: 1088, 1152 and 1664). The save area
// holds what the CPU's own XSAVE wrote, at the offsets CPUID leaf 0xD gives
// for components 5, 6 and 7, and an AMD core's differ (832, 896 and 1408: no
// MPX components before them). There the copy reads and writes the wrong
// bytes. RgfxGate takes those components with XSAVE and puts them back with
// XRSTOR at CPUID's offsets instead (RgfxGateXstate.cpp), through this
// file's copy.
//
// The image is the standard (non-compacted) XSAVE form: the legacy region at
// 0, the header at 512 (XSTATE_BV its first 8 bytes, XCOMP_BV the next 8, the
// rest reserved and zero), each component at its CPUID offset. A component
// whose XSTATE_BV bit is clear is in its initial state, all zeros, whatever
// the image holds there.
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef KERNEL
#include <libkern/libkern.h> // memcpy, memset in the kext
#else
#include <string.h>
#endif

// XSAVE state components 5 (opmask), 6 (ZMM_Hi256) and 7 (Hi16_ZMM).
static constexpr uint64_t kRgfxXstateOpmask = 1ull << 5;
static constexpr uint64_t kRgfxXstateZmmHi256 = 1ull << 6;
static constexpr uint64_t kRgfxXstateHi16Zmm = 1ull << 7;
static constexpr uint64_t kRgfxXstateAvx512 = kRgfxXstateOpmask | kRgfxXstateZmmHi256 | kRgfxXstateHi16Zmm;

// The components' sizes: k0-7 at 8 bytes, zmm0-15's upper 32 bytes, zmm16-31
// at 64 bytes.
static constexpr uint32_t kRgfxOpmaskSize = 64;
static constexpr uint32_t kRgfxZmmHi256Size = 512;
static constexpr uint32_t kRgfxHi16ZmmSize = 1024;

// The XSAVE header's place in the image, and the first byte past it.
static constexpr uint32_t kRgfxXsaveHeader = 512;
static constexpr uint32_t kRgfxXsaveHeaderEnd = 576;

// Where the image holds components 5-7: CPUID leaf 0xD, sub-leaves 5, 6 and 7,
// EBX (the offset) and EAX (the size).
struct RgfxXstateLayout {
	uint32_t opmask, opmask_size;
	uint32_t zmm_hi256, zmm_hi256_size;
	uint32_t hi16_zmm, hi16_zmm_size;
};

// Intel's layout, which XNU's copy assumes.
static constexpr RgfxXstateLayout kRgfxXstateIntel = {1088, kRgfxOpmaskSize, 1152, kRgfxZmmHi256Size,
                                                      1664, kRgfxHi16ZmmSize};

static inline bool rgfx_xstate_is_intel(const RgfxXstateLayout &l) {
	return l.opmask == kRgfxXstateIntel.opmask && l.zmm_hi256 == kRgfxXstateIntel.zmm_hi256 &&
	       l.hi16_zmm == kRgfxXstateIntel.hi16_zmm;
}

// The first byte past the last component.
static inline uint32_t rgfx_xstate_end(const RgfxXstateLayout &l) {
	uint32_t e = kRgfxXsaveHeaderEnd;
	if (l.opmask + l.opmask_size > e)
		e = l.opmask + l.opmask_size;
	if (l.zmm_hi256 + l.zmm_hi256_size > e)
		e = l.zmm_hi256 + l.zmm_hi256_size;
	if (l.hi16_zmm + l.hi16_zmm_size > e)
		e = l.hi16_zmm + l.hi16_zmm_size;
	return e;
}

// Whether the copy can use the layout: each component its full size, past the
// header, none overlapping another, all within limit bytes.
static inline bool rgfx_xstate_layout_ok(const RgfxXstateLayout &l, uint32_t limit) {
	if (l.opmask_size != kRgfxOpmaskSize || l.zmm_hi256_size != kRgfxZmmHi256Size ||
	    l.hi16_zmm_size != kRgfxHi16ZmmSize)
		return false;
	// In 64 bits: CPUID's offsets near 2^32 must not wrap past the limit.
	const uint64_t at[3] = {l.opmask, l.zmm_hi256, l.hi16_zmm};
	const uint64_t size[3] = {l.opmask_size, l.zmm_hi256_size, l.hi16_zmm_size};
	for (int i = 0; i < 3; i++) {
		if (at[i] < kRgfxXsaveHeaderEnd || at[i] + size[i] > limit)
			return false;
		for (int j = i + 1; j < 3; j++)
			if (at[i] < at[j] + size[j] && at[j] < at[i] + size[i])
				return false;
	}
	return true;
}

static inline uint64_t rgfx_xstate_bv(const uint8_t *image) {
	uint64_t bv;
	memcpy(&bv, image + kRgfxXsaveHeader, sizeof bv);
	return bv;
}

static inline bool rgfx_xstate_all_zero(const uint8_t *p, size_t n) {
	uint8_t d = 0;
	for (size_t i = 0; i < n; i++)
		d |= p[i];
	return d == 0;
}

// The flavor's three fields from an image XSAVE wrote: k0-7 (64 bytes), the
// upper halves of zmm0-15 (512) and zmm16-31 (1024). A component whose
// XSTATE_BV bit is clear reads as zeros.
static inline void rgfx_xstate_to_flavor(const uint8_t *image, const RgfxXstateLayout &l, uint8_t *k,
                                         uint8_t *zmmh, uint8_t *zmm16) {
	const uint64_t bv = rgfx_xstate_bv(image);
	if (bv & kRgfxXstateOpmask)
		memcpy(k, image + l.opmask, kRgfxOpmaskSize);
	else
		memset(k, 0, kRgfxOpmaskSize);
	if (bv & kRgfxXstateZmmHi256)
		memcpy(zmmh, image + l.zmm_hi256, kRgfxZmmHi256Size);
	else
		memset(zmmh, 0, kRgfxZmmHi256Size);
	if (bv & kRgfxXstateHi16Zmm)
		memcpy(zmm16, image + l.hi16_zmm, kRgfxHi16ZmmSize);
	else
		memset(zmm16, 0, kRgfxHi16ZmmSize);
}

// An image for XRSTOR with components 5-7 from the flavor's three fields, into
// an image of rgfx_xstate_end(l) bytes or more: the header's XSTATE_BV has a
// component's bit where its bytes are not all zero (XNU's rule in
// fpu_set_fxstate, which keeps an all-zero component in its initial state),
// XCOMP_BV and the reserved bytes are zero, and the rest of the image is zero.
static inline void rgfx_xstate_from_flavor(uint8_t *image, const RgfxXstateLayout &l, const uint8_t *k,
                                           const uint8_t *zmmh, const uint8_t *zmm16) {
	memset(image, 0, rgfx_xstate_end(l));
	uint64_t bv = 0;
	if (!rgfx_xstate_all_zero(k, kRgfxOpmaskSize)) {
		memcpy(image + l.opmask, k, kRgfxOpmaskSize);
		bv |= kRgfxXstateOpmask;
	}
	if (!rgfx_xstate_all_zero(zmmh, kRgfxZmmHi256Size)) {
		memcpy(image + l.zmm_hi256, zmmh, kRgfxZmmHi256Size);
		bv |= kRgfxXstateZmmHi256;
	}
	if (!rgfx_xstate_all_zero(zmm16, kRgfxHi16ZmmSize)) {
		memcpy(image + l.hi16_zmm, zmm16, kRgfxHi16ZmmSize);
		bv |= kRgfxXstateHi16Zmm;
	}
	memcpy(image + kRgfxXsaveHeader, &bv, sizeof bv);
}
