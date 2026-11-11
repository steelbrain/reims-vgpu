// RgfxSurfaceLimits: the largest surface the kext has IOSurface answer to
// IOSurfaceGetPropertyMaximum, as a GPU's accelerator lowers IOSurfaceRoot's limits at its start:
// as wide and as tall as the device's 2D textures (the device info's max_texture_width_2d and
// max_texture_height_2d, which the bundle answers to Metal), and that width's row in the largest
// pixels a texture over a surface has, 16 bytes (RGBA32Float, RGBA32Uint, RGBA32Sint).
//
// IOSurfaceRoot starts the three at INT_MAX, and answers INT_MAX where no accelerator lowers them
// (measured: a macOS 13.7.8 guest). A Mac's GPUs lower them: 16384 wide and tall on an Intel Mac
// with a 5300M and a 630 (131072 bytes a row, macOS 26.7.1) and on an M3 Max (32768, macOS 27)
// (measured). Safari 16.6 under macOS 13 sizes the tiles of a page that scrolls only vertically
// by the maximum width over the screen's scale, in float (read from WebKit's code,
// TileController::computeTileSize): at 1x INT_MAX is 2^31 there, which no int holds, and such a
// page drew nothing but its scroll bar (measured, 13.7.8; at 2x it drew).
//
// Header-only and free of the kernel's headers, so its native test
// (tests/surface_limits_test.cpp) builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_SURFACE_LIMITS_HPP_
#define RGFX_SURFACE_LIMITS_HPP_

#include "reims_graphics_wire.h"

#include <stddef.h>
#include <stdint.h>

// The entries of RgfxDeviceInfo's `limits` the kext reads, in the host's `Limits` field order
// (the bundle's RgfxLimit): max_texture_height_2d and max_texture_width_2d.
static constexpr unsigned kRgfxSurfaceLimitsHeightEntry = 15;
static constexpr unsigned kRgfxSurfaceLimitsWidthEntry = 17;

// The bytes of RgfxDeviceInfo the kext reads for the limits: `limits` up to its width entry.
static constexpr uint64_t kRgfxSurfaceLimitsFirst = offsetof(RgfxDeviceInfo, limits);
static constexpr uint64_t kRgfxSurfaceLimitsEnd =
    kRgfxSurfaceLimitsFirst + (kRgfxSurfaceLimitsWidthEntry + 1) * sizeof(uint64_t);

// The largest bytes of one pixel of a texture over a surface.
static constexpr uint64_t kRgfxSurfaceLargestPixel = 16;

// What IOSurfaceRoot starts each limit at, which leaves the limit as it is.
static constexpr uint32_t kRgfxSurfaceNoLimit = 0x7fffffff;

// The limits the kext asks for, each kRgfxSurfaceNoLimit where the device answers none.
struct RgfxSurfaceLimits {
	uint32_t width;
	uint32_t height;
	uint32_t bytesPerRow;
};

// A limit from the device's answer: none for 0 or for one past IOSurface's own.
inline uint32_t rgfxSurfaceLimit(uint64_t answer)
{
	return answer == 0 || answer > kRgfxSurfaceNoLimit ? kRgfxSurfaceNoLimit
	                                                   : static_cast<uint32_t>(answer);
}

// The limits from the device's widest and tallest 2D texture.
inline RgfxSurfaceLimits rgfxSurfaceLimits(uint64_t width, uint64_t height)
{
	RgfxSurfaceLimits out;
	out.width = rgfxSurfaceLimit(width);
	out.height = rgfxSurfaceLimit(height);
	out.bytesPerRow = out.width == kRgfxSurfaceNoLimit
	                      ? kRgfxSurfaceNoLimit
	                      : rgfxSurfaceLimit(uint64_t(out.width) * kRgfxSurfaceLargestPixel);
	return out;
}

// The limits out of the device info's bytes from kRgfxSurfaceLimitsFirst to kRgfxSurfaceLimitsEnd.
inline RgfxSurfaceLimits rgfxSurfaceLimitsFrom(const uint8_t *span)
{
	auto entry = [span](unsigned k) {
		uint64_t v = 0;
		for (unsigned i = 0; i < sizeof(v); i++)
			v |= static_cast<uint64_t>(span[k * sizeof(v) + i]) << (8 * i);
		return v;
	};
	return rgfxSurfaceLimits(entry(kRgfxSurfaceLimitsWidthEntry),
	                         entry(kRgfxSurfaceLimitsHeightEntry));
}

#endif // RGFX_SURFACE_LIMITS_HPP_
