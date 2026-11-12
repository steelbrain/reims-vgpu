// The native test of RgfxDisplayPower.hpp: the display's panel from the display wrangler's power
// messages, and when the framebuffer tells the host.
#include "RgfxDisplayPower.hpp"

#include "reims_graphics_wire.h"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "display_power_test: FAILED: %s\n", what);
		std::exit(1);
	}
}
} // namespace

int main()
{
	// iokit_common_msg(message) is sys_iokit | sub_iokit_common | message: 0xe0000000 | message.
	check(kRgfxMessageDeviceWillPowerOff == (0xe0000000u | 0x210u), "WillPowerOff is 0x210");
	check(kRgfxMessageDeviceHasPoweredOn == (0xe0000000u | 0x230u), "HasPoweredOn is 0x230");

	// The display sleeps: the wrangler goes from on (4) through dimmed (3) to a state below.
	check(rgfxPanelAfter(kRgfxMessageDeviceWillPowerOff, true, 3) == RgfxPanel::Unchanged,
	      "a dimmed display stays lit");
	for (unsigned long state = 0; state < 3; state++)
		check(rgfxPanelAfter(kRgfxMessageDeviceWillPowerOff, true, state) == RgfxPanel::Off,
		      "a display going below dimmed goes dark");
	// The display wakes: the wrangler is lit again, dimmed or on.
	check(rgfxPanelAfter(kRgfxMessageDeviceHasPoweredOn, true, 4) == RgfxPanel::On,
	      "a display woken to on is lit");
	check(rgfxPanelAfter(kRgfxMessageDeviceHasPoweredOn, true, 3) == RgfxPanel::On,
	      "a display woken to dimmed is lit");
	check(rgfxPanelAfter(kRgfxMessageDeviceHasPoweredOn, true, 2) == RgfxPanel::Unchanged,
	      "a power-up that stays below dimmed leaves the panel dark");
	// Nothing else moves the panel: a message with no state, or of another type.
	check(rgfxPanelAfter(kRgfxMessageDeviceWillPowerOff, false, 0) == RgfxPanel::Unchanged,
	      "a power-off with no state leaves the panel");
	check(rgfxPanelAfter(kRgfxMessageDeviceHasPoweredOn, false, 4) == RgfxPanel::Unchanged,
	      "a power-on with no state leaves the panel");
	check(rgfxPanelAfter(0xe0000220u /* kIOMessageDeviceWillPowerOn */, true, 4) ==
	          RgfxPanel::Unchanged,
	      "WillPowerOn leaves the panel until the power is on");
	check(rgfxPanelAfter(0xe0000200u /* kIOMessageCanDevicePowerOff */, true, 1) ==
	          RgfxPanel::Unchanged,
	      "the ask before a power-off leaves the panel");

	// The framebuffer tells the host each change, only where the host took DISPLAY_POWER.
	check(!rgfxPanelToSend(true, true, true), "lit and told lit: nothing to tell");
	check(rgfxPanelToSend(true, false, true), "dark, told lit: told");
	check(rgfxPanelToSend(true, true, false), "lit again, told dark: told");
	check(!rgfxPanelToSend(false, false, true), "a host that did not take it is told nothing");
	check(RGFX_FEATURE_DISPLAY_POWER == (1ull << 37), "DISPLAY_POWER is bit 37");
	check(RGFX_CMD_SET_DISPLAY_POWER == 8, "SET_DISPLAY_POWER is command 8");
	check(sizeof(RgfxSetDisplayPower) == 24, "SET_DISPLAY_POWER is 24 bytes");

	std::printf("display_power_test: %d checks passed\n", checks);
	return 0;
}
