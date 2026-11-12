// RgfxDisplayPower: whether the display's panel is lit, as the x86-64 framebuffer hears of the
// display's sleep and wake (design.md, "The display's sleep").
//
// The x86-64 framebuffer stays out of power management itself (ReimsGraphics.cpp, "Power"), so it
// hears of the sleep from the OS's display wrangler (IODisplayWrangler), in which it registers its
// general interest: as the display sleeps (`pmset displaysleepnow`, Energy Saver's timer) the
// wrangler tells its general-interest clients kIOMessageDeviceWillPowerOff with the state it goes
// to, below its lit states, and at the wake kIOMessageDeviceHasPoweredOn with a lit one. Its lit
// states are kRgfxWranglerLitState and above: 4, on, and 3, dimmed, which a dimmed panel shows.
// The panel goes off as the wrangler goes below them, and on once it is lit again; a message of
// any other type, or with no state, leaves the panel as it is (rgfxPanelAfter).
//
// An arm64 guest has no display wrangler, and the arm64e framebuffer does not take DISPLAY_POWER
// (RgfxFeatures.hpp): its panel stays lit through the display's sleep.
//
// The framebuffer tells the host with SET_DISPLAY_POWER, where the host took DISPLAY_POWER, each
// time the panel it wants differs from the one it last told (rgfxPanelToSend). The device is lit
// from its reset, as the framebuffer's record of what it told starts.
//
// Header-only and free of the kernel's headers, so its native test (tests/display_power_test.cpp)
// builds it as the kext does.
//
// Our own code.
#ifndef RGFX_DISPLAY_POWER_HPP_
#define RGFX_DISPLAY_POWER_HPP_

#include <stdint.h>

// IOKit's kIOMessageDeviceWillPowerOff and kIOMessageDeviceHasPoweredOn (iokit_common_msg(0x210)
// and (0x230)), which ReimsGraphics.cpp checks against the kernel's.
constexpr uint32_t kRgfxMessageDeviceWillPowerOff = 0xe0000210u;
constexpr uint32_t kRgfxMessageDeviceHasPoweredOn = 0xe0000230u;

// The wrangler's lowest lit state: 3, dimmed, below 4, on.
constexpr unsigned long kRgfxWranglerLitState = 3;

// What a wrangler message leaves the display's panel at.
enum class RgfxPanel { Unchanged, Off, On };

// What a wrangler message of `type` leaves the panel at, `state` the power state it names where
// it names one (`hasState`).
inline RgfxPanel rgfxPanelAfter(uint32_t type, bool hasState, unsigned long state)
{
	if (!hasState)
		return RgfxPanel::Unchanged;
	if (type == kRgfxMessageDeviceWillPowerOff && state < kRgfxWranglerLitState)
		return RgfxPanel::Off;
	if (type == kRgfxMessageDeviceHasPoweredOn && state >= kRgfxWranglerLitState)
		return RgfxPanel::On;
	return RgfxPanel::Unchanged;
}

// Whether the framebuffer posts SET_DISPLAY_POWER: where the host took DISPLAY_POWER (`taken`)
// and the panel it wants (`wanted`, true lit) is not the one it last told (`told`).
inline bool rgfxPanelToSend(bool taken, bool wanted, bool told)
{
	return taken && wanted != told;
}

#endif // RGFX_DISPLAY_POWER_HPP_
