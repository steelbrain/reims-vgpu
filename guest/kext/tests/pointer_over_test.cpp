// The native test of RgfxPointerOver.hpp: a POINTER_OVER's display, point and buttons, the move
// its buttons make (a drag where one is held), where in the global space a display's pixel lies,
// at 1x and 2x, held inside the display, and that point as an event's 16-bit location.
#include "RgfxPointerOver.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "pointer_over_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

bool at(const RgfxGlobalRect &rect, uint32_t width, uint32_t height, int32_t x, int32_t y,
        int32_t wantX, int32_t wantY)
{
	int32_t gx = -1, gy = -1;
	return rgfxPointerGlobal(rect, width, height, x, y, &gx, &gy) && gx == wantX && gy == wantY;
}
} // namespace

int main()
{
	// The event's layout: the display, the point, then the buttons, past its 16-byte header.
	{
		RgfxPointerOver event = {};
		event.header.type = RGFX_EVT_POINTER_OVER;
		event.display = 1;
		event.x = 2400;
		event.y = -3;
		event.buttons = RGFX_POINTER_BUTTON_LEFT;
		uint8_t bytes[sizeof(event)];
		std::memcpy(bytes, &event, sizeof(event));
		RgfxPointerAt p = {};
		check(rgfxPointerOverRead(bytes + 16, sizeof(event) - 16, &p), "the event reads");
		check(p.display == 1 && p.x == 2400 && p.y == -3, "its display and point");
		check(p.buttons == RGFX_POINTER_BUTTON_LEFT, "its buttons");
		check(!rgfxPointerOverRead(bytes + 16, 12, &p),
		      "a payload short of the buttons is refused");
		check(sizeof(RgfxPointerOver) == 32 && RGFX_EVT_POINTER_OVER == 0x104, "the wire's");
		check(RGFX_FEATURE_POINTER_OVER == (1ull << 49), "POINTER_OVER is feature bit 49");
	}

	// No button held is a move; a held button's is its drag, the left's first, then the right's,
	// then any other's.
	check(rgfxPointerMoveOf(0) == RgfxPointerMove::moved, "no button, a move");
	check(rgfxPointerMoveOf(RGFX_POINTER_BUTTON_LEFT) == RgfxPointerMove::leftDragged, "left");
	check(rgfxPointerMoveOf(RGFX_POINTER_BUTTON_RIGHT) == RgfxPointerMove::rightDragged, "right");
	check(rgfxPointerMoveOf(RGFX_POINTER_BUTTON_LEFT | RGFX_POINTER_BUTTON_RIGHT) ==
	          RgfxPointerMove::leftDragged,
	      "left and right, the left's");
	check(rgfxPointerMoveOf(RGFX_POINTER_BUTTON_RIGHT | RGFX_POINTER_BUTTON_MIDDLE) ==
	          RgfxPointerMove::rightDragged,
	      "right and middle, the right's");
	check(rgfxPointerMoveOf(RGFX_POINTER_BUTTON_MIDDLE) == RgfxPointerMove::otherDragged, "middle");
	check(rgfxPointerMoveOf(RGFX_POINTER_BUTTON_SIDE) == RgfxPointerMove::otherDragged, "side");
	check(rgfxPointerMoveOf(RGFX_POINTER_BUTTON_EXTRA) == RgfxPointerMove::otherDragged, "extra");
	check(rgfxPointerMoveOf(1u << 5) == RgfxPointerMove::moved, "an unknown bit is no button");

	// Display 1 to the right of a 1920 x 1080 display 0, at 1x: its pixel is the space's point.
	const RgfxGlobalRect right = {1920, 0, 3840, 1080};
	check(at(right, 1920, 1080, 0, 0, 1920, 0), "the corner");
	check(at(right, 1920, 1080, 480, 270, 2400, 270), "a pixel at 1x");
	check(at(right, 1920, 1080, 1919, 1079, 3839, 1079), "the last pixel");
	// Held inside the display, as the pointer is held at the image's edge.
	check(at(right, 1920, 1080, 1920, 1080, 3839, 1079), "past the last pixel is the last");
	check(at(right, 1920, 1080, -5, -5, 1920, 0), "before the first is the first");
	// At 2x the rectangle is the mode's points, half its pixels.
	const RgfxGlobalRect scaled = {1440, -100, 2880, 710};
	check(at(scaled, 2880, 1620, 1000, 501, 1940, 150), "a pixel at 2x");
	// A display left of and above display 0.
	const RgfxGlobalRect left = {-1280, -800, 0, 0};
	check(at(left, 1280, 800, 640, 400, -640, -400), "negative corners");
	// Nothing for an empty mode or rectangle.
	int32_t gx = 7, gy = 7;
	check(!rgfxPointerGlobal(right, 0, 1080, 1, 1, &gx, &gy) && gx == 7 && gy == 7, "no mode");
	check(!rgfxPointerGlobal({5, 5, 5, 9}, 1920, 1080, 1, 1, &gx, &gy), "an empty rectangle");

	// An event's location is 16 bits: a point past them is held at their ends.
	check(rgfxPointerLocation(2880) == 2880 && rgfxPointerLocation(-640) == -640, "in range");
	check(rgfxPointerLocation(40000) == 32767 && rgfxPointerLocation(-40000) == -32768, "held");

	std::printf("pointer_over_test: %d checks passed\n", checks);
	return 0;
}
