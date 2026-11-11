// The native test of RgfxSurfaceLimits.hpp: the largest surface the kext has IOSurface answer,
// from the device's widest and tallest 2D texture in the device info.
#include "RgfxSurfaceLimits.hpp"

#include "../../bundle/src/RgfxCaps.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// The entries are the bundle's, which answers them to Metal.
static_assert(kRgfxSurfaceLimitsWidthEntry == kRgfxLimitMaxTextureWidth2D, "width entry");
static_assert(kRgfxSurfaceLimitsHeightEntry == kRgfxLimitMaxTextureHeight2D, "height entry");
static_assert(kRgfxSurfaceLimitsEnd <= sizeof(RgfxDeviceInfo), "inside the device info");

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "surface_limits_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

// The limits as the kext reads them: out of a whole device info's bytes, the host's layout.
RgfxSurfaceLimits fromDeviceInfo(uint64_t width, uint64_t height)
{
	static RgfxDeviceInfo info;
	std::memset(&info, 0xa5, sizeof(info));
	info.limits[kRgfxLimitMaxTextureWidth2D] = width;
	info.limits[kRgfxLimitMaxTextureHeight2D] = height;
	return rgfxSurfaceLimitsFrom(reinterpret_cast<const uint8_t *>(&info) + kRgfxSurfaceLimitsFirst);
}

bool is(RgfxSurfaceLimits l, uint32_t width, uint32_t height, uint32_t bytesPerRow)
{
	return l.width == width && l.height == height && l.bytesPerRow == bytesPerRow;
}
} // namespace

int main()
{
	// The device's 16384 a side (Metal's answer for the GPU families it reports): surfaces as
	// wide and tall, and rows of 16384 RGBA32Float pixels.
	check(is(rgfxSurfaceLimits(16384, 16384), 16384, 16384, 262144), "16384 a side");
	check(is(fromDeviceInfo(16384, 16384), 16384, 16384, 262144),
	      "16384 a side, out of the device info");
	// Width and height each from its own entry.
	check(is(fromDeviceInfo(8192, 4096), 8192, 4096, 131072), "8192 wide, 4096 tall");
	// A device that answers none leaves IOSurface's own.
	check(is(rgfxSurfaceLimits(0, 0), kRgfxSurfaceNoLimit, kRgfxSurfaceNoLimit, kRgfxSurfaceNoLimit),
	      "no answer");
	check(is(fromDeviceInfo(0, 16384), kRgfxSurfaceNoLimit, 16384, kRgfxSurfaceNoLimit),
	      "no width, out of the device info");
	// Past what IOSurface holds: its own.
	check(is(rgfxSurfaceLimits(1ull << 32, 16384), kRgfxSurfaceNoLimit, 16384, kRgfxSurfaceNoLimit),
	      "4 Gi wide");
	// A row past IOSurface's own: its own row, the width kept.
	check(is(rgfxSurfaceLimits(1u << 28, 16384), 1u << 28, 16384, kRgfxSurfaceNoLimit),
	      "a row past INT_MAX");

	std::printf("surface_limits_test: OK (%d checks)\n", checks);
	return 0;
}
