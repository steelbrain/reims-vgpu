// The native test of RgfxSurfaceRows.hpp: the alignment the kext has IOSurface give the bytes per
// row it works out, from the host's answer in the device info.
#include "RgfxSurfaceRows.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static_assert(offsetof(RgfxDeviceInfo, linear_row_alignment) == 0x454, "linear_row_alignment");

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "surface_rows_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

// The alignment as the kext reads it: out of a whole device info's bytes, the host's layout.
uint32_t fromDeviceInfo(uint32_t host)
{
	static RgfxDeviceInfo info;
	std::memset(&info, 0xa5, sizeof(info));
	info.linear_row_alignment = host;
	return rgfxSurfaceRowAlignmentFrom(reinterpret_cast<const uint8_t *>(&info) + kRgfxSurfaceRowsFirst);
}
} // namespace

int main()
{
	// RADV's 256 bytes: IOSurface's rows at 256.
	check(rgfxSurfaceRowAlignment(256) == 256, "RADV's 256 bytes");
	check(fromDeviceInfo(256) == 256, "RADV's 256 bytes, out of the device info");
	// A host that answers none (an older host, a GPU of its own memory) or no more than IOSurface's
	// own 64: IOSurface's 64.
	check(rgfxSurfaceRowAlignment(0) == 64, "no answer");
	check(fromDeviceInfo(0) == 64, "no answer, out of the device info");
	check(rgfxSurfaceRowAlignment(16) == 64, "MoltenVK's 16 bytes");
	check(rgfxSurfaceRowAlignment(64) == 64, "64 bytes");
	// Up to a page.
	check(rgfxSurfaceRowAlignment(128) == 128, "128 bytes");
	check(rgfxSurfaceRowAlignment(4096) == 4096, "a page");
	// No power of two, or past a page: IOSurface's 64.
	check(rgfxSurfaceRowAlignment(192) == 64, "192 bytes");
	check(rgfxSurfaceRowAlignment(8192) == 64, "two pages");
	check(fromDeviceInfo(0x80000000u) == 64, "2 GiB, out of the device info");

	std::printf("surface_rows_test: OK (%d checks)\n", checks);
	return 0;
}
