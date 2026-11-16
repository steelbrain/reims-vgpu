// RgfxDisplayBounds: which displays' rectangles the x86-64 kext tells the host in DISPLAY_BOUNDS
// (design.md, "Multiple displays", "A drag across displays"). Each framebuffer holds its display's
// rectangle in WindowServer's global space as WindowServer placed it (IOFBSetBounds, which
// IOFramebuffer::getBoundingRect reads; RgfxPointerOver.hpp), and the host carries its captured
// pointer past one display's edge onto the display whose rectangle meets it there. Each move of the
// cursor, on any display, tells the host every display's rectangle that differs from the one the
// host was last told, so a rearrangement in System Settings reaches the host with the next move. A
// rectangle is counted told only once its command is on the ring, and the kext forgets them all at
// each reset of the device, which forgets them too.
//
// Header-only and free of the kernel's headers, so its native test (tests/display_bounds_test.cpp)
// builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_DISPLAY_BOUNDS_HPP_
#define RGFX_DISPLAY_BOUNDS_HPP_

#include <stdint.h>

#include "RgfxPointerOver.hpp"
#include "reims_graphics_wire.h"

// The rectangle the host was last told of each display, by id, where it was told one.
struct RgfxBoundsTold {
	RgfxGlobalRect rects[RGFX_DISPLAYS_MAX];
	bool told[RGFX_DISPLAYS_MAX];
};

// Whether display `display`'s rectangle `rect` is to be told: a display the device can have, a
// rectangle with an area, and the host told none of the display's or another.
inline bool rgfxBoundsToTell(const RgfxBoundsTold &told, uint32_t display, const RgfxGlobalRect &rect)
{
	if (display >= RGFX_DISPLAYS_MAX || rect.maxx <= rect.minx || rect.maxy <= rect.miny)
		return false;
	if (!told.told[display])
		return true;
	const RgfxGlobalRect &was = told.rects[display];
	return was.minx != rect.minx || was.miny != rect.miny || was.maxx != rect.maxx ||
	       was.maxy != rect.maxy;
}

// DISPLAY_BOUNDS for display `display` at `rect`; its header is the post's.
inline RgfxDisplayBounds rgfxBoundsCommand(uint32_t display, const RgfxGlobalRect &rect)
{
	RgfxDisplayBounds command = {};
	command.display = display;
	command.min_x = rect.minx;
	command.min_y = rect.miny;
	command.max_x = rect.maxx;
	command.max_y = rect.maxy;
	return command;
}

// Display `display`'s `rect` as told, once its command is on the ring.
inline void rgfxBoundsWereTold(RgfxBoundsTold *told, uint32_t display, const RgfxGlobalRect &rect)
{
	if (display >= RGFX_DISPLAYS_MAX)
		return;
	told->rects[display] = rect;
	told->told[display] = true;
}

#endif // RGFX_DISPLAY_BOUNDS_HPP_
