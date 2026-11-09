// RgfxMobileModes: the display's modes as an arm64 process reads them of the framebuffer
// (ReimsGraphicsMobile.cpp): IOAVFamily's timing and colour elements, which the framebuffer
// publishes as its TimingElements and ColorElements, and the mode set_digital_out_mode takes by
// their IDs. Apple's paravirtual display (AppleParavirtDisplay) publishes and takes its own the same
// way: it makes each element with IOAVFamily's constructors, writes its ID and Score, gives each
// timing its colour elements, and its set_digital_out_mode takes a colour ID and a timing ID (read
// from macOS 27.0.1's AppleParavirtGPU, createDisplayAttributes, init and set_digital_out_mode).
// Header-only and free of the kernel's headers, for the native test (tests/mobile_modes_test.cpp).
//
// Our own code.
#ifndef RGFX_MOBILE_MODES_HPP_
#define RGFX_MOBILE_MODES_HPP_

#include <stddef.h>
#include <stdint.h>

#include "RgfxDisplayModes.hpp"

// IOAVFamily's IOAVVideoTimingData, the 0x50 bytes IOAVVideoTimingElement's withData copies into the
// element it makes. Each field is named by the key IOAVVideoTimingElement's copyProperties, or
// IOAVVideoTimingAxisCopy for an axis, publishes it under, at the same offsets on every release
// from 14 to 27.0.1 (read from each release's IOAVFamily).
struct RgfxIOAVTimingAxis {
	uint32_t total;
	uint32_t active;
	uint32_t syncWidth;
	uint32_t backPorch;
	uint32_t frontPorch;
	// In 16.16 fixed point: the horizontal axis's in kilohertz, the vertical axis's in hertz, as
	// IOAVVideoTimingUpdateSyncRates computes them from the pixel clock and the totals. The axis's
	// SyncRate is this rounded to a whole unit.
	int32_t preciseSyncRate;
	uint32_t syncPolarity;
	uint32_t pixelRepetition;
};

struct RgfxIOAVTimingData {
	// IsInterlaced (bit 0) and IsSplit (bit 1).
	uint32_t flags;
	// The horizontal axis, then the vertical one (IOAVVideoAxis 0 and 1).
	RgfxIOAVTimingAxis axis[2];
	// In hertz: IOAVVideoTimingGetPixelClock answers it, or, where it is 0, the vertical rate times
	// both totals.
	uint32_t pixelClock;
	uint32_t aspectRatio;
	uint32_t scanInformation;
};

static_assert(sizeof(RgfxIOAVTimingAxis) == 0x20, "an axis is 0x20 bytes");
static_assert(offsetof(RgfxIOAVTimingData, axis) == 0x04, "the axes start at 4");
static_assert(offsetof(RgfxIOAVTimingData, pixelClock) == 0x44, "the pixel clock is at 0x44");
static_assert(offsetof(RgfxIOAVTimingData, aspectRatio) == 0x48, "AspectRatio is at 0x48");
static_assert(offsetof(RgfxIOAVTimingData, scanInformation) == 0x4c, "ScanInformation is at 0x4c");
static_assert(sizeof(RgfxIOAVTimingData) == 0x50, "withData copies 0x50 bytes");

// IOAVElement's ID (64 bits) and Score (32 bits), which its copyProperties publishes, on every
// release from 14 to 27.0.1. IOAVFamily declares no setter for either: the paravirtual display
// writes both at these offsets once withData or withCapabilities has made the element.
constexpr size_t kRgfxIOAVElementId = 0x10;
constexpr size_t kRgfxIOAVElementScore = 0x18;

// The IOAVStandardType of every element, the IOAVVideoTimingStandard of every timing and the
// IOAVVideoColorimetry of every colour mode, as the paravirtual display gives its own.
constexpr uint32_t kRgfxIOAVStandardType = 2;
constexpr uint32_t kRgfxIOAVTimingStandard = 4;
constexpr uint32_t kRgfxIOAVColorimetry = 16;

// The display's one colour mode: 8 bits a component, as the device shows BGRA8 surfaces and the
// x86-64 framebuffer's one pixel format is 32-bit direct pixels; PixelEncoding, DynamicRange and EOTF
// 0 and no DSC, as the paravirtual display's 8-bit colour mode, whose ID (100) and Score (the same)
// it keeps.
constexpr uint32_t kRgfxColorElementId = 100;
constexpr uint32_t kRgfxColorDepth = 8;

// A timing's ID: 1000, where the paravirtual display numbers its own from, and the mode's own ID
// (RgfxDisplayModes.hpp), which stays with its size and rate while the device lists it.
constexpr uint32_t kRgfxTimingIdBase = 1000;
// The Score of the first timing, which the paravirtual display gives its first.
constexpr uint32_t kRgfxTimingTopScore = 1000;

// A timing the display publishes: its mode, its ID and its Score.
struct RgfxMobileTiming {
	RgfxMode mode;
	uint32_t id;
	uint32_t score;
};

// The display's timings, into `out` (room for `n`): one for each of the device's modes
// `modes[0..n)`, its preferred one (`preferredId`) first and then the others in the device's order,
// each scored kRgfxTimingTopScore less its place, as the paravirtual display scores its own in its
// host's order.
// Returns their count.
inline uint32_t rgfxMobileTimings(const RgfxMode *modes, uint32_t n, int32_t preferredId,
                                  RgfxMobileTiming *out)
{
	uint32_t count = 0;
	for (int pass = 0; pass < 2; pass++) {
		for (uint32_t i = 0; i < n; i++) {
			const RgfxMode &m = modes[i];
			if ((m.id == preferredId) != (pass == 0) || m.id <= 0)
				continue;
			out[count] = RgfxMobileTiming{m, kRgfxTimingIdBase + static_cast<uint32_t>(m.id),
			                              kRgfxTimingTopScore - count};
			count++;
		}
	}
	return count;
}

// The timing a display starts in, before WindowServer sets one: the first rgfxMobileTimings lists,
// the preferred mode's, or the first mode's where none is preferred. 0 for a display with no mode.
inline uint32_t rgfxMobileStartTimingId(const RgfxMode *modes, uint32_t n, int32_t preferredId)
{
	uint32_t first = 0;
	for (uint32_t i = 0; i < n; i++) {
		if (modes[i].id <= 0)
			continue;
		const uint32_t id = kRgfxTimingIdBase + static_cast<uint32_t>(modes[i].id);
		if (modes[i].id == preferredId)
			return id;
		if (!first)
			first = id;
	}
	return first;
}

// The mode of the colour and timing IDs set_digital_out_mode takes, in that order: the mode in
// `modes[0..n)` whose timing has `timingId`, with the display's one colour mode. False for any other
// pair, which the paravirtual display leaves to its parent class to answer.
inline bool rgfxMobileModeFor(uint32_t colorId, uint32_t timingId, const RgfxMode *modes, uint32_t n,
                              RgfxMode *mode)
{
	if (colorId != kRgfxColorElementId || timingId <= kRgfxTimingIdBase)
		return false;
	const uint32_t id = timingId - kRgfxTimingIdBase;
	for (uint32_t i = 0; i < n; i++) {
		if (modes[i].id > 0 && static_cast<uint32_t>(modes[i].id) == id) {
			*mode = modes[i];
			return true;
		}
	}
	return false;
}

// A mode's IOAVVideoTimingData, from its timing `t` (ReimsGraphics' Timing: the reduced blanking
// the x86-64 framebuffer's detailed timing gives the same mode, its pixel clock in hertz) and the
// device's rate, `refreshMhz` thousandths of a hertz. The vertical rate is the device's own; the
// horizontal one is the line rate the pixel clock gives, as IOAVVideoTimingUpdateSyncRates rounds
// it. A pixel clock past 32 bits is left 0, from which IOAVFamily works it out of the vertical rate
// and the totals. The sync polarities, aspect ratio and scan information are 0, as the paravirtual
// display leaves them.
template <typename Timing>
inline RgfxIOAVTimingData rgfxIOAVTimingData(const Timing &t, uint32_t refreshMhz)
{
	RgfxIOAVTimingData d = {};
	const uint32_t hTotal = t.hActive + t.hBlank;
	const uint32_t vTotal = t.vActive + t.vBlank;
	const uint64_t clock = t.pixelClock;
	d.axis[0].total = hTotal;
	d.axis[0].active = t.hActive;
	d.axis[0].syncWidth = t.hSync;
	d.axis[0].backPorch = t.hBlank - t.hFront - t.hSync;
	d.axis[0].frontPorch = t.hFront;
	d.axis[0].preciseSyncRate =
	    hTotal ? static_cast<int32_t>(((clock << 16) + hTotal * 500ull) / (hTotal * 1000ull)) : 0;
	d.axis[1].total = vTotal;
	d.axis[1].active = t.vActive;
	d.axis[1].syncWidth = t.vSync;
	d.axis[1].backPorch = t.vBlank - t.vFront - t.vSync;
	d.axis[1].frontPorch = t.vFront;
	d.axis[1].preciseSyncRate =
	    static_cast<int32_t>(((static_cast<uint64_t>(refreshMhz) << 16) + 500) / 1000);
	d.pixelClock = clock <= 0xffffffffull ? static_cast<uint32_t>(clock) : 0;
	return d;
}

#endif // RGFX_MOBILE_MODES_HPP_
