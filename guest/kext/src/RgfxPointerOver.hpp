// RgfxPointerOver: where the guest's cursor goes for the host's POINTER_OVER (design.md, "Multiple
// displays", "The pointer across displays"): the host's pointer is over display `display` at
// (x, y) of that display's pixels, with `buttons` held, and macOS scales the one USB tablet over
// the display its cursor is on, so the kext moves the cursor onto that display, at that point,
// before the tablet's next position comes (ReimsGraphicsPointer.cpp).
//
// IOHIDSystem's post takes a point in WindowServer's virtual desktop. On x86-64 the framebuffer
// holds the virtual rectangle WindowServer gave it (IOFBSetBounds); on arm64 the kext learns
// each rectangle from its own moves (RgfxDisplayPlaces.hpp). Scale the scanout pixels into that
// rectangle. Measured on x86-64 macOS 11.7.11, 15.8.2 and 26.7.2 with two 1920 x 1080 displays
// side by side: a mouse-moved posted at a point in display 1's CGDisplayBounds puts the cursor
// there (lane ptrcross, potluck jobs m3-max-34583 to 34585).
//
// The move is a mouse-moved where no button is held, and the drag of the button held where one is,
// as IOHIDSystem's own moves are (the left button's before the right's, the right's before any
// other's), so a drag across displays stays a drag.
//
// Header-only and free of the kernel's headers, so its native test (tests/pointer_over_test.cpp)
// builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_POINTER_OVER_HPP_
#define RGFX_POINTER_OVER_HPP_

#include <stdint.h>

#include "RgfxDisplayTable.hpp"
#include "reims_graphics_wire.h"

// POINTER_OVER's display, point and buttons (RgfxPointerOver past its header).
struct RgfxPointerAt {
	uint32_t display;
	int32_t x;
	int32_t y;
	uint32_t buttons;
};

// The display, point and buttons of a POINTER_OVER whose payload (past its header) is `size`
// bytes at `payload`: false where it is too short to hold them.
inline bool rgfxPointerOverRead(const volatile uint8_t *payload, uint32_t size, RgfxPointerAt *at)
{
	if (size < 4 * sizeof(uint32_t))
		return false;
	at->display = rgfxPayload32(payload, 0);
	at->x = static_cast<int32_t>(rgfxPayload32(payload, 4));
	at->y = static_cast<int32_t>(rgfxPayload32(payload, 8));
	at->buttons = rgfxPayload32(payload, 12);
	return true;
}

// The kind of move the cursor's is, by the buttons held: a plain move, or the drag of the left,
// the right or another button, in that order, as IOHIDSystem picks a move's event type
// (NX_MOUSEMOVED, NX_LMOUSEDRAGGED, NX_RMOUSEDRAGGED, NX_OMOUSEDRAGGED).
enum class RgfxPointerMove {
	moved,
	leftDragged,
	rightDragged,
	otherDragged,
};

inline RgfxPointerMove rgfxPointerMoveOf(uint32_t buttons)
{
	if (buttons & RGFX_POINTER_BUTTON_LEFT)
		return RgfxPointerMove::leftDragged;
	if (buttons & RGFX_POINTER_BUTTON_RIGHT)
		return RgfxPointerMove::rightDragged;
	constexpr uint32_t kOthers =
	    RGFX_POINTER_BUTTON_MIDDLE | RGFX_POINTER_BUTTON_SIDE | RGFX_POINTER_BUTTON_EXTRA;
	if (buttons & kOthers)
		return RgfxPointerMove::otherDragged;
	return RgfxPointerMove::moved;
}

// A display's rectangle in the global space: its corner (minx, miny) and the corner past it
// (maxx, maxy), as IOGBounds holds it.
struct RgfxGlobalRect {
	int32_t minx;
	int32_t miny;
	int32_t maxx;
	int32_t maxy;
};

// The point of the global space where pixel (x, y) of a display whose mode is `width` by `height`
// pixels lies, the display at `rect`: the rectangle's corner, plus the pixel scaled from the
// mode's pixels to the rectangle's size (1:1 at 1x, half at 2x), rounded down, and held inside the
// rectangle as the pointer is held at the image's edge. False, with nothing written, where the
// mode or the rectangle is empty.
inline bool rgfxPointerGlobal(const RgfxGlobalRect &rect, uint32_t width, uint32_t height, int32_t x,
                              int32_t y, int32_t *globalX, int32_t *globalY)
{
	const int64_t spanX = static_cast<int64_t>(rect.maxx) - rect.minx;
	const int64_t spanY = static_cast<int64_t>(rect.maxy) - rect.miny;
	if (width == 0 || height == 0 || spanX <= 0 || spanY <= 0)
		return false;
	auto place = [](int64_t at, int64_t pixels, int64_t span) {
		if (at < 0)
			at = 0;
		int64_t to = at * span / pixels;
		return to < span ? to : span - 1;
	};
	*globalX = static_cast<int32_t>(rect.minx + place(x, width, spanX));
	*globalY = static_cast<int32_t>(rect.miny + place(y, height, spanY));
	return true;
}

// A global coordinate as an event's location holds it (IOGPoint's 16 bits), held in their range.
inline int16_t rgfxPointerLocation(int32_t global)
{
	constexpr int32_t kLeast = -32768, kMost = 32767;
	return static_cast<int16_t>(global < kLeast ? kLeast : global > kMost ? kMost : global);
}

#endif // RGFX_POINTER_OVER_HPP_
