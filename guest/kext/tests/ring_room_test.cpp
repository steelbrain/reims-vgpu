// The native test of RgfxRingRoom.hpp: a post that found the control ring full sleeps only where
// no work-loop pass came since its mark, a pass wakes the posts counted waiting and only where
// one is, and a sleep ends at its backstop or the post's deadline, whichever comes first.
#include "RgfxRingRoom.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "ring_room_test: FAILED: %s\n", what);
		std::exit(1);
	}
}
} // namespace

int main()
{
	// No post waits: a pass wakes no one, and moves the passes on.
	{
		RgfxRingRoom r = {};
		const uint64_t mark = rgfxRingRoomMark(r);
		check(!rgfxRingRoomPass(r), "a pass with no post waiting wakes no one");
		check(rgfxRingRoomMark(r) == mark + 1, "a pass moves the passes on");
	}

	// A pass between a post's mark and its sleep: the post looks again at once.
	{
		RgfxRingRoom r = {};
		const uint64_t mark = rgfxRingRoomMark(r);
		rgfxRingRoomPass(r);
		rgfxRingRoomWait(r);
		check(!rgfxRingRoomSleeps(r, mark), "a post whose mark a pass passed does not sleep");
		rgfxRingRoomDone(r);
	}

	// A pass after the post counted itself waiting: the pass wakes it, and it does not sleep
	// past that pass.
	{
		RgfxRingRoom r = {};
		const uint64_t mark = rgfxRingRoomMark(r);
		rgfxRingRoomWait(r);
		check(rgfxRingRoomSleeps(r, mark), "a post with no pass since its mark sleeps");
		check(rgfxRingRoomPass(r), "a pass wakes the post counted waiting");
		check(!rgfxRingRoomSleeps(r, mark), "the woken post looks again");
		rgfxRingRoomDone(r);
		check(!rgfxRingRoomPass(r), "once the post looked again, a pass wakes no one");
	}

	// Two posts waiting: a pass wakes both, and wakes on until both looked again.
	{
		RgfxRingRoom r = {};
		const uint64_t a = rgfxRingRoomMark(r);
		rgfxRingRoomWait(r);
		const uint64_t b = rgfxRingRoomMark(r);
		rgfxRingRoomWait(r);
		check(rgfxRingRoomPass(r), "a pass wakes the posts waiting");
		check(!rgfxRingRoomSleeps(r, a) && !rgfxRingRoomSleeps(r, b), "each looks again");
		rgfxRingRoomDone(r);
		check(rgfxRingRoomPass(r), "a post still waiting is woken by the next pass");
		rgfxRingRoomDone(r);
		check(!rgfxRingRoomPass(r), "none left waiting");
	}

	// The sleep's end: the backstop, or the deadline where sooner, and the deadline where the
	// backstop would wrap.
	check(rgfxRingRoomSleepUntil(100, 10, 1000) == 110, "the backstop before the deadline");
	check(rgfxRingRoomSleepUntil(995, 10, 1000) == 1000, "the deadline before the backstop");
	check(rgfxRingRoomSleepUntil(UINT64_MAX - 5, 10, UINT64_MAX - 1) == UINT64_MAX - 1,
	      "a backstop past the clock's end is the deadline");
	check(rgfxRingRoomSleepUntil(1000, 10, 1000) == 1000, "at the deadline, the deadline");

	std::printf("ring_room_test: %d checks passed\n", checks);
	return 0;
}
