// The native test of RgfxRegisterHold.hpp: the arm64e framebuffers' registrations, held while
// display 0's start starts the heads, with one display as before the hold, every display starting,
// a head that fails to start and a hold past its deadline; and the window after the release, in
// which a process's property reads wait for the services to match.
#include "RgfxRegisterHold.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "register_hold_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

// Display 0's start as ReimsGraphics runs it over the hold: the class's start registers display 0
// (registerService with options 0), then each head starts in turn, registering as its class start
// ends where `starts[i]` holds, and the hold ends once the heads are done or it is due. The order
// the services register in, display by display, goes into `order`; returns how many registered.
// `clock[i]` is the time as head i's start ends.
uint32_t runStart(uint32_t count, const bool *starts, const uint64_t *clock, uint64_t deadline,
                  uint32_t *order)
{
	RgfxRegisterHold hold;
	uint32_t n = 0;
	auto registerNow = [&](uint32_t display) { order[n++] = display; };
	auto releaseHeld = [&]() {
		const uint32_t held = hold.release();
		for (uint32_t d = 0; d < RGFX_DISPLAYS_MAX; d++)
			if (held & (1u << d))
				registerNow(d);
	};
	hold.open(count, deadline);
	if (!hold.hold(0, 0))
		registerNow(0);
	for (uint32_t i = 1; i < count; i++) {
		if (starts[i] && !hold.hold(i, 0))
			registerNow(i);
		if (hold.due(!starts[i], clock[i]))
			releaseHeld();
	}
	releaseHeld();
	return n;
}
} // namespace

int main()
{
	const bool all[4] = {true, true, true, true};
	const uint64_t early[4] = {0, 1, 2, 3};
	uint32_t order[8] = {};

	// One display: nothing is held, and display 0's service registers within the class's start,
	// as before the hold.
	{
		RgfxRegisterHold hold;
		hold.open(1, 100);
		check(!hold.holding, "one display opens no hold");
		check(!hold.hold(0, 0), "one display: display 0 registers at once");
		check(!hold.due(true, 1000), "one display: nothing is ever due");
		check(hold.release() == 0, "one display: nothing to release");
		check(runStart(1, all, early, 100, order) == 1 && order[0] == 0,
		      "one display: display 0 registers, alone");
	}

	// Two to four displays, every head starting: every service registers as the hold ends,
	// display 0's first, then the heads' by display.
	for (uint32_t count = 2; count <= RGFX_DISPLAYS_MAX; count++) {
		const uint32_t n = runStart(count, all, early, 100, order);
		check(n == count, "every display's service registers");
		for (uint32_t d = 0; d < count; d++)
			check(order[d] == d, "display 0's first, then the heads' by display");
	}

	// The options each was asked with come back with it.
	{
		RgfxRegisterHold hold;
		hold.open(2, 100);
		check(hold.hold(0, 0) && hold.hold(1, 7), "both held");
		check(hold.options[0] == 0 && hold.options[1] == 7, "each display's options kept");
		check(hold.release() == 0x3, "both released");
		check(!hold.hold(1, 0), "after the release, a registration runs at once");
		check(hold.release() == 0, "a second release hands back nothing");
	}

	// Head 1 of three fails to start: display 0's service registers as soon as it has failed, and
	// head 2's as its start ends, after display 0's.
	{
		const bool starts[4] = {true, false, true, true};
		const uint32_t n = runStart(3, starts, early, 100, order);
		check(n == 2 && order[0] == 0 && order[1] == 2,
		      "a failed head: display 0 registers at once, head 2 after it");
	}

	// The only head fails: display 0's service still registers.
	{
		const bool starts[4] = {true, false, false, false};
		check(runStart(2, starts, early, 100, order) == 1 && order[0] == 0,
		      "the only head failed: display 0 registers");
	}

	// Every head fails: display 0's alone registers.
	{
		const bool starts[4] = {true, false, false, false};
		check(runStart(4, starts, early, 100, order) == 1 && order[0] == 0,
		      "every head failed: display 0 registers alone");
	}

	// The heads' starts run past the deadline: the hold ends after the head whose start passed
	// it, and the later heads register as they start.
	{
		const uint64_t slow[4] = {0, 50, 150, 160};
		const uint32_t n = runStart(4, all, slow, 100, order);
		check(n == 4 && order[0] == 0 && order[1] == 1 && order[2] == 2 && order[3] == 3,
		      "past the deadline: displays 0 to 2 at head 2's end, head 3 as it starts");
		RgfxRegisterHold hold;
		hold.open(4, 100);
		hold.hold(0, 0);
		check(!hold.due(false, 99), "not due before the deadline");
		check(hold.due(false, 100), "due at the deadline");
	}

	// A display past RGFX_DISPLAYS_MAX is never held.
	{
		RgfxRegisterHold hold;
		hold.open(RGFX_DISPLAYS_MAX, 100);
		check(!hold.hold(RGFX_DISPLAYS_MAX, 0), "a display past the table registers at once");
		check(hold.release() == 0, "nothing held for it");
	}

	// The deadline leaves a process's 30 second wait room.
	check(kRgfxRegisterHoldNs < 30000000000ull, "the hold ends well inside a process's wait");

	// The window after the release: a process's read waits while it is open and before its
	// deadline; the kernel's reads never wait; a closed window, or one past its deadline, keeps no
	// read waiting.
	{
		RgfxRegisterWindow w;
		check(!w.waits(true, 0), "no window before a release: no read waits");
		w.begin(100);
		check(w.waits(true, 50), "an open window: a process's read waits");
		check(!w.waits(false, 50), "an open window: the kernel's read never waits");
		check(!w.waits(true, 100), "at the deadline no read waits");
		w.end();
		check(!w.waits(true, 50), "a closed window: no read waits");
		check(kRgfxRegisterHoldNs + kRgfxRegisterWindowNs < 30000000000ull,
		      "the hold and the window end inside a process's wait");
	}

	std::printf("register_hold_test: %d checks passed\n", checks);
	return 0;
}
