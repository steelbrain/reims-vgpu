// RgfxSurfacePlace: where an IOAccel surface with no backing of its own lies in the framebuffer's
// scanout (RgfxSurfaceUserClient::lock). Header-only and free of the kernel's headers, so its
// native test (tests/surface_test.cpp) builds it as the kext does.
//
// A surface's shape is a region of its framebuffer (SetShape: the region's bounds, in the
// display's pixels). WindowServer keeps one such surface for a display and reshapes it to each
// update's dirty rectangle before it locks it (measured on a macOS 26 guest: the whole 3840x2400
// display first, then 320x56 at 1760,2160, the login window's password field, about 120 times a
// second). With no backing, the surface's pixels are the scanout's under its shape: they start
// at the shape's top-left corner, each row the scanout's row bytes after the one before, so
// what the client writes there is what the display shows.
//
// Our own code.
#ifndef RGFX_SURFACE_PLACE_HPP_
#define RGFX_SURFACE_PLACE_HPP_

#include <stdint.h>

// The bytes a pixel of the scanout takes: the framebuffer's one mode depth, 32-bit BGRA.
constexpr uint32_t kRgfxScanoutBytesPerPixel = 4;

// The offset in the scanout of the first pixel of a shape at `x`, `y` of `w` by `h` pixels, for a
// scanout of `width` by `height` pixels, rows `rowBytes` apart. False for an empty shape, or one
// not wholly inside the scanout.
inline bool rgfxSurfaceInScanout(int32_t x, int32_t y, int32_t w, int32_t h, uint32_t width,
                                 uint32_t height, uint32_t rowBytes, uint64_t *offset)
{
	if (x < 0 || y < 0 || w <= 0 || h <= 0)
		return false;
	const uint64_t right = static_cast<uint64_t>(x) + static_cast<uint64_t>(w);
	const uint64_t bottom = static_cast<uint64_t>(y) + static_cast<uint64_t>(h);
	if (right > width || bottom > height ||
	    static_cast<uint64_t>(width) * kRgfxScanoutBytesPerPixel > rowBytes)
		return false;
	*offset = static_cast<uint64_t>(y) * rowBytes +
	          static_cast<uint64_t>(x) * kRgfxScanoutBytesPerPixel;
	return true;
}

#endif // RGFX_SURFACE_PLACE_HPP_
