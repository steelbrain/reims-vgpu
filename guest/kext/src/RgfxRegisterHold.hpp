// RgfxRegisterHold: the arm64e framebuffers' registrations, held while display 0's start starts
// the heads, so every display's IOMobileFramebuffer service registers back to back (design.md,
// "Multiple displays", "How an arm64 process pairs displays with services").
//
// A process's IOMobileFramebuffer.framework counts the displays it expects from the device tree,
// then takes the IOMobileFramebuffer services already registered and, where it has fewer than it
// counted, waits once, up to 30 seconds, for the next batch of matching notifications, and goes on
// with what it has (read from macOS 27.0.1's framework). A service that registers after that batch
// is never driven by the process. IOMobileFramebufferLegacy's start registers its service as it
// ends, so display 0's would register before any head starts, and each head's a head's start
// later. Display 0's framebuffer opens the hold before the class's start where it drives two
// displays or more; each framebuffer's registerService is then recorded instead of run, and display
// 0's start ends the hold once the heads have started, registering the held services, display 0's
// first, then the heads' by display. It ends the hold early, and the heads that start after
// register as they start, once a head has failed to start or the hold's deadline has passed, so no
// head's start keeps display 0's service from registering for long. With one display the hold is
// never opened, and display 0's service registers within the class's start, as before the hold.
//
// Registering back to back is not enough on its own. A process takes the services of one batch of
// notifications: the first service registered sends the batch, and the batch holds every service
// registered before the process has read its last one. Each registration matches on a thread of
// its own, so a second service can come after a process has read the first and found no more (an
// arm64 15.6.1 guest under TCG: its WindowServer took display 1's service alone, and gave up on
// display 0's). So the release opens a window (RgfxRegisterWindow) until every service released
// has matched, its notifications delivered, or the window's deadline has passed, and in the window
// a process's read of one of the properties the framework reads of each service it takes, before
// it asks for the next (read from macOS 27.0.1's iomfb_match_callback: "dfr", "AppleTV" and
// "external"), waits for the window to close, so the services it has not read yet are in its batch
// by then. The kernel's own reads, the services' matching among them, never wait, nor does a read
// of any other key, which a process's matching may make with the registry's notification lock held.
//
// Header-only and free of the kernel's headers, so its native test (tests/register_hold_test.cpp)
// builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_REGISTER_HOLD_HPP_
#define RGFX_REGISTER_HOLD_HPP_

#include <stdint.h>

#include "reims_graphics_wire.h"

// How long display 0's start holds the registrations at most, in nanoseconds: well inside the 30
// seconds a process waits for its services, which the heads' starts must leave room in.
constexpr uint64_t kRgfxRegisterHoldNs = 10000000000ull;

struct RgfxRegisterHold {
	// Whether the hold is open: from open to release.
	bool holding = false;
	// The displays whose registration is held, a bit each, and the options each was asked with.
	uint32_t held = 0;
	uint32_t options[RGFX_DISPLAYS_MAX] = {};
	// When the hold is due to end, in the caller's clock.
	uint64_t deadline = 0;

	// Opens the hold for a start of `count` displays, due at `due`: only where `count` is two or
	// more. Anything held before is dropped.
	void open(uint32_t count, uint64_t due)
	{
		holding = count > 1;
		held = 0;
		deadline = due;
	}

	// Whether display `display`'s registration, asked with `opts`, is held: recorded, for release
	// to hand back. False, and nothing recorded, where the hold is not open or the display is past
	// RGFX_DISPLAYS_MAX: the caller then registers the service at once.
	bool hold(uint32_t display, uint32_t opts)
	{
		if (!holding || display >= RGFX_DISPLAYS_MAX)
			return false;
		held |= 1u << display;
		options[display] = opts;
		return true;
	}

	// Whether the hold is to end before the rest of the heads start: a head has failed to start
	// (`headFailed`), or `now` is at or past the deadline.
	bool due(bool headFailed, uint64_t now) const { return holding && (headFailed || now >= deadline); }

	// Ends the hold: the displays held, a bit each, which the caller registers by display, display
	// 0's first. From here hold answers false.
	uint32_t release()
	{
		const uint32_t out = holding ? held : 0;
		holding = false;
		held = 0;
		return out;
	}
};

// How long the window after the release lasts at most, in nanoseconds: a process's property read
// waits no longer than this for the released services to match.
constexpr uint64_t kRgfxRegisterWindowNs = 5000000000ull;

struct RgfxRegisterWindow {
	// Whether the window is open: from the release to the services' matching or the deadline.
	bool open = false;
	// When the window closes at the latest, in the caller's clock.
	uint64_t deadline = 0;

	void begin(uint64_t due)
	{
		open = true;
		deadline = due;
	}
	void end() { open = false; }

	// Whether a read of a framebuffer's property waits at `now`: the window is open and before its
	// deadline, and the read is a process's (`byProcess`): the kernel's never waits.
	bool waits(bool byProcess, uint64_t now) const { return open && byProcess && now < deadline; }
};

#endif // RGFX_REGISTER_HOLD_HPP_
