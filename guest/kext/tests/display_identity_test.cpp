// The native test of what the display says it is (RgfxDisplayIdentity.hpp): the x86-64
// framebuffer's connection flags and its EDID.
#include "RgfxDisplayIdentity.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

int checks = 0;

void check(bool ok, const char *what, const char *about)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "display_identity_test: FAILED (%s): %s\n", about, what);
		std::exit(1);
	}
}

// The connection is built in (kIOConnectionBuiltIn, 0x800) and nothing else: no DisplayPort or
// HDMI type, no backlight, no mirroring flags.
void connectionFlags()
{
	check(kRgfxDisplayConnectionFlags == 0x800, "built in, and no other flag", "connection");
}

// The host screens' scales the tests cover, in 120ths: none told yet, 1x, 1.25x, 1.5x, 2x, KDE's
// 2.1x and 3x.
constexpr uint32_t kScales[] = {0, 120, 150, 180, 240, 252, 360};
// Display ids the tests cover.
constexpr uint32_t kIds[] = {0, 1, 7};
// Window widths the tests cover.
constexpr uint32_t kWidths[] = {1024, 3888, 7680};

// The EDID is a valid EDID 1.4 base block saying who the display is: our manufacturer, product,
// a serial per display, the year, the product's name, sRGB, and no timings or extensions.
void edidBlock()
{
	for (uint32_t scale : kScales) {
		for (uint32_t id : kIds) {
			uint8_t b[kRgfxDisplayEdidBytes];
			std::memset(b, 0xee, sizeof(b));
			rgfxDisplayEdid(b, id, scale, 3888, 2072);
			const uint8_t header[8] = {0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00};
			check(std::memcmp(b, header, 8) == 0, "the EDID's header", "edid");
			uint8_t sum = 0;
			for (uint8_t v : b)
				sum = static_cast<uint8_t>(sum + v);
			check(sum == 0, "the bytes sum to 0", "edid");
			const uint32_t packed = (static_cast<uint32_t>(b[8]) << 8) | b[9];
			const char made[4] = {static_cast<char>('@' + ((packed >> 10) & 31)),
			                      static_cast<char>('@' + ((packed >> 5) & 31)),
			                      static_cast<char>('@' + (packed & 31)), 0};
			check(std::strcmp(made, "RVG") == 0, "the manufacturer is RVG", "edid");
			check(b[10] == 0x47 && b[11] == 0x52, "the product is 0x5247", "edid");
			const uint32_t serial = b[12] | (b[13] << 8) | (b[14] << 16) |
			                        (static_cast<uint32_t>(b[15]) << 24);
			check(serial == ((0x5247u << 16) | (id + 1)), "the serial names the display", "edid");
			check(b[17] == 36 && b[18] == 1 && b[19] == 4, "made in 2026, EDID 1.4", "edid");
			check(b[20] == 0xa5 && b[24] == 0x04, "digital, 8 bits, sRGB the default", "edid");
			for (int i = 35; i < 38; i++)
				check(b[i] == 0, "no established timings", "edid");
			for (int i = 38; i < 54; i++)
				check(b[i] == 0x01, "no standard timings", "edid");
			check(b[54 + 3] == 0x10 && b[108 + 3] == 0x10,
			      "no detailed timing: dummy descriptors", "edid");
			check(b[90 + 3] == 0xfe && std::memcmp(b + 90 + 5, "3888x2072\n   ", 13) == 0,
			      "the native size as text, padded", "edid");
			check(b[72 + 3] == 0xfc && std::memcmp(b + 72 + 5, "Reims Display", 13) == 0,
			      "the product's name", "edid");
			check(b[126] == 0, "no extensions", "edid");
		}
	}
}

// The EDID's size is that of a 1920 x 1080 display at the display's density, in whole
// centimetres, whatever the window's size, so it is under the 32 inches across from which macOS
// starts a display HiDPI at 3800 pixels wide.
void edidSize()
{
	struct Want {
		uint32_t scale, width, height;
	};
	const Want wants[] = {{0, 44, 25},   {120, 44, 25}, {150, 35, 20}, {180, 30, 17},
	                      {240, 22, 13}, {252, 21, 12}, {360, 15, 8}};
	for (const Want &w : wants) {
		for (uint32_t width : kWidths) {
			uint8_t b[kRgfxDisplayEdidBytes];
			rgfxDisplayEdid(b, 0, w.scale, width, 2072);
			check(b[21] == w.width && b[22] == w.height,
			      "the reference's size at the density, whatever the window's", "size");
		}
	}
	for (uint32_t scale : kScales) {
		uint8_t a[kRgfxDisplayEdidBytes];
		rgfxDisplayEdid(a, 0, scale, 3888, 2072);
		// The density the size gives is the display's, within a centimetre's rounding.
		const double ppi = 1920 * 2.54 / a[21];
		const double want = rgfxDisplayPixelsPerInch(scale);
		check(std::fabs(ppi - want) <= want * 0.05, "the EDID's density is the display's",
		      "size");
		const double inches = std::sqrt(a[21] * a[21] + a[22] * a[22]) / 2.54;
		check(inches < 32, "under 32 inches across", "size");
	}
}

// One EDID for one display, scale and window size; another for any other, so each window size
// is a display of its own to the guest, whose WindowServer then gives a new size its default mode
// rather than keep the old one.
void edidIdentity()
{
	const uint32_t sizes[][2] = {{3888, 2072}, {3840, 2160}, {3888, 2070}, {1600, 932},
	                             {16384, 16384}, {16, 2}};
	for (uint32_t scale : kScales) {
		for (const auto &one : sizes) {
			uint8_t a[kRgfxDisplayEdidBytes], b[kRgfxDisplayEdidBytes];
			rgfxDisplayEdid(a, 0, scale, one[0], one[1]);
			rgfxDisplayEdid(b, 0, scale, one[0], one[1]);
			check(std::memcmp(a, b, sizeof(a)) == 0, "one display, scale and size: one EDID",
			      "identity");
			uint8_t sum = 0;
			for (uint8_t v : a)
				sum = static_cast<uint8_t>(sum + v);
			check(sum == 0, "the bytes sum to 0", "identity");
			check(a[90 + 3] == 0xfe && (a[90 + 5 + 12] == 0x0a || a[90 + 5 + 12] == 0x20),
			      "the size's text fits its descriptor", "identity");
			for (const auto &other : sizes) {
				if (&other == &one)
					continue;
				rgfxDisplayEdid(b, 0, scale, other[0], other[1]);
				check(std::memcmp(a, b, sizeof(a)) != 0, "another size: another EDID",
				      "identity");
			}
		}
	}
	uint8_t one[kRgfxDisplayEdidBytes], two[kRgfxDisplayEdidBytes];
	rgfxDisplayEdid(one, 0, 120, 3888, 2072);
	rgfxDisplayEdid(two, 0, 252, 3888, 2072);
	check(std::memcmp(one, two, sizeof(one)) != 0, "another scale: another EDID", "identity");
	rgfxDisplayEdid(two, 1, 120, 3888, 2072);
	check(std::memcmp(one, two, sizeof(one)) != 0, "another display: another EDID", "identity");
}

} // namespace

int main()
{
	connectionFlags();
	edidBlock();
	edidSize();
	edidIdentity();
	std::printf("display_identity_test: %d checks passed\n", checks);
	return 0;
}
