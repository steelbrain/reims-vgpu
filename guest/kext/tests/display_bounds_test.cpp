// The native test of RgfxDisplayBounds.hpp: which displays' rectangles are told the host (each
// display's first, and one that changed, never one without an area or past the device's
// displays), the command that tells one, and that a rectangle counts as told only once marked.
#include "RgfxDisplayBounds.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "display_bounds_test: FAILED: %s\n", what);
		std::exit(1);
	}
}
} // namespace

int main()
{
	// The wire's command: 40 bytes, the display then the rectangle past the 16-byte header.
	check(sizeof(RgfxDisplayBounds) == 40 && RGFX_CMD_DISPLAY_BOUNDS == 9, "the wire's");
	check(offsetof(RgfxDisplayBounds, display) == 16 && offsetof(RgfxDisplayBounds, min_x) == 20 &&
	          offsetof(RgfxDisplayBounds, max_y) == 32,
	      "its fields' offsets");
	check(RGFX_FEATURE_DISPLAY_BOUNDS == (1ull << 50), "DISPLAY_BOUNDS is feature bit 50");

	RgfxBoundsTold told = {};
	const RgfxGlobalRect left = {0, 0, 1920, 1080};
	const RgfxGlobalRect right = {1920, 200, 3840, 1280};
	// Nothing told yet: each display's is to be told.
	check(rgfxBoundsToTell(told, 0, left) && rgfxBoundsToTell(told, 1, right), "first, told");
	{
		const RgfxDisplayBounds command = rgfxBoundsCommand(1, right);
		check(command.display == 1 && command.min_x == 1920 && command.min_y == 200 &&
		          command.max_x == 3840 && command.max_y == 1280,
		      "the command holds the display and its rectangle");
		check(command.header.type == 0 && command.reserved_24 == 0, "the rest is zero");
	}
	// Until marked told, it stays to be told: a post the ring had no room for is tried again.
	check(rgfxBoundsToTell(told, 1, right), "unmarked, still to tell");
	rgfxBoundsWereTold(&told, 0, left);
	rgfxBoundsWereTold(&told, 1, right);
	check(!rgfxBoundsToTell(told, 0, left) && !rgfxBoundsToTell(told, 1, right),
	      "told, not again");
	// Display 1 moved left of display 0, as System Settings arranges it: told again.
	const RgfxGlobalRect moved = {-1920, 0, 0, 1080};
	check(rgfxBoundsToTell(told, 1, moved), "a moved rectangle, told");
	check(rgfxBoundsToTell(told, 1, {1920, 200, 3840, 1281}), "a resized one, told");
	// A rectangle of no area, and a display past the device's, never.
	check(!rgfxBoundsToTell(told, 1, {5, 5, 5, 10}) && !rgfxBoundsToTell(told, 1, {0, 9, 10, 3}),
	      "no area, not told");
	check(!rgfxBoundsToTell(told, RGFX_DISPLAYS_MAX, left), "past the device's displays");
	rgfxBoundsWereTold(&told, RGFX_DISPLAYS_MAX, left);
	// A reset forgets them all.
	told = {};
	check(rgfxBoundsToTell(told, 0, left), "after a reset, told again");

	std::printf("display_bounds_test: %d checks passed\n", checks);
	return 0;
}
