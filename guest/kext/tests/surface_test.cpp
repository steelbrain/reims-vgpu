// The placement rule of RgfxSurfacePlace.hpp, built natively: a surface with no backing is the
// scanout's pixels under its shape, where the shape lies wholly inside the scanout.
#include <cstdio>

#include "RgfxSurfacePlace.hpp"

static int g_checks = 0, g_failed = 0;
#define CHECK(e)                                                                   \
	do {                                                                           \
		g_checks++;                                                                \
		if (!(e)) {                                                                \
			g_failed++;                                                            \
			std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #e); \
		}                                                                          \
	} while (0)

int main()
{
	const uint32_t width = 3840, height = 2400, rowBytes = 3840 * 4;
	uint64_t at = 1;

	// The whole display: its first pixel.
	CHECK(rgfxSurfaceInScanout(0, 0, 3840, 2400, width, height, rowBytes, &at) && at == 0);
	// A dirty rectangle (the login window's password field): its top-left pixel.
	CHECK(rgfxSurfaceInScanout(1760, 2160, 320, 56, width, height, rowBytes, &at));
	CHECK(at == 2160ull * rowBytes + 1760ull * 4);
	// Rows longer than the display's pixels: placed by the rows.
	CHECK(rgfxSurfaceInScanout(1, 1, 1, 1, width, height, rowBytes + 256, &at));
	CHECK(at == rowBytes + 256 + 4);
	// Touching the far edges is inside.
	CHECK(rgfxSurfaceInScanout(3839, 2399, 1, 1, width, height, rowBytes, &at));

	// Outside, partly outside, empty, or rows too short for the display: nowhere.
	at = 7;
	CHECK(!rgfxSurfaceInScanout(-1, 0, 10, 10, width, height, rowBytes, &at));
	CHECK(!rgfxSurfaceInScanout(0, -1, 10, 10, width, height, rowBytes, &at));
	CHECK(!rgfxSurfaceInScanout(3800, 0, 41, 10, width, height, rowBytes, &at));
	CHECK(!rgfxSurfaceInScanout(0, 2390, 10, 11, width, height, rowBytes, &at));
	CHECK(!rgfxSurfaceInScanout(0, 0, 0, 10, width, height, rowBytes, &at));
	CHECK(!rgfxSurfaceInScanout(0, 0, 10, -2, width, height, rowBytes, &at));
	CHECK(!rgfxSurfaceInScanout(0, 0, 10, 10, width, height, rowBytes - 4, &at));
	CHECK(at == 7);

	if (g_failed) {
		std::fprintf(stderr, "surface_test: %d of %d checks failed\n", g_failed, g_checks);
		return 1;
	}
	std::printf("surface_test: OK (%d checks)\n", g_checks);
	return 0;
}
