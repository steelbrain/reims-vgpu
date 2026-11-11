// RgfxDisplayIdentity: what the display says it is, the same on both of the kext's architectures:
// the x86-64 framebuffer's EDID (ReimsGraphics.cpp, makeEdid) and the arm64e framebuffer's
// DisplayAttributes (ReimsGraphicsMobile.cpp) give the OS these values. Header-only and free of
// the kernel's headers.
//
// Our own code.
#ifndef RGFX_DISPLAY_IDENTITY_HPP_
#define RGFX_DISPLAY_IDENTITY_HPP_

#include <stdint.h>

// Our own three-letter PNP manufacturer ID, "RVG", and the EDID's packing of it: five bits a
// letter, 'A' being 1.
constexpr char kRgfxDisplayManufacturer[] = "RVG";
constexpr uint16_t kRgfxDisplayManufacturerPacked = static_cast<uint16_t>(
    ((kRgfxDisplayManufacturer[0] - '@') << 10) | ((kRgfxDisplayManufacturer[1] - '@') << 5) |
    (kRgfxDisplayManufacturer[2] - '@'));
// Our product code and the product's name.
constexpr uint16_t kRgfxDisplayProduct = 0x5247;
constexpr char kRgfxDisplayName[] = "Reims Display";
// The year of manufacture: 2026.
constexpr uint32_t kRgfxDisplayYear = 2026;

// The largest width or height of a mode the display takes.
constexpr uint32_t kRgfxDisplayMaxDimension = 16384;

// The serial: the product code over the device's display id, so each display keeps its own.
constexpr uint32_t rgfxDisplaySerial(uint32_t displayId)
{
	return (static_cast<uint32_t>(kRgfxDisplayProduct) << 16) | ((displayId + 1) & 0xffff);
}

// The pixels per inch of a display shown on a host screen of `scale120ths` pixels a point, in
// 120ths (the device's RgfxDisplayScale): 110 a point, as a Mac's 1x display has, so 110 on a
// screen at 1x, 220 on one at 2x and 231 at 2.1, rounded to nearest; 110 where the host gives no
// scale (0: no screen shows the display yet, or a host without DISPLAY_SCALE).
constexpr uint32_t kRgfxDisplayPointsPerInch = 110;
constexpr uint32_t rgfxDisplayPixelsPerInch(uint32_t scale120ths)
{
	return scale120ths ? (kRgfxDisplayPointsPerInch * scale120ths + 60) / 120
	                   : kRgfxDisplayPointsPerInch;
}

// Whether a display on a host screen of `scale120ths` is high-density: its pixels per inch above
// a point's, so the OS may run it HiDPI.
constexpr bool rgfxDisplayHighDensity(uint32_t scale120ths)
{
	return rgfxDisplayPixelsPerInch(scale120ths) > kRgfxDisplayPointsPerInch;
}

// The size, in millimetres, of `pixels` of a display on a host screen of `scale120ths`, at its
// pixels per inch, rounded to nearest.
constexpr uint32_t rgfxDisplayMillimetres(uint32_t pixels, uint32_t scale120ths)
{
	return (pixels * 254 + rgfxDisplayPixelsPerInch(scale120ths) * 5) /
	       (rgfxDisplayPixelsPerInch(scale120ths) * 10);
}

// sRGB's primaries and D65 white, the display's colours, in thousandths of CIE 1931 x and y.
struct RgfxDisplayChromaticity {
	uint32_t x, y;
};
constexpr RgfxDisplayChromaticity kRgfxDisplayRed = {640, 330};
constexpr RgfxDisplayChromaticity kRgfxDisplayGreen = {300, 600};
constexpr RgfxDisplayChromaticity kRgfxDisplayBlue = {150, 60};
constexpr RgfxDisplayChromaticity kRgfxDisplayWhite = {313, 329};

// The x86-64 framebuffer's connection flags (IOFramebuffer's kConnectionFlags): built in
// (IOGraphicsTypes.h's kIOConnectionBuiltIn), as a Mac's own panel is; the VM's window is the
// machine's one screen. x86-64 macOS picks the default scale of a built-in display by its
// density, and of any other by its width (read from macOS 26.7.2's CoreDisplay): a built-in
// display starts HiDPI where its default mode is at least 2048 x 1536 pixels at more than 160
// pixels per inch across its diagonal, so on a host screen at 1.5x or more it runs HiDPI at every
// window size from 2048 x 1536; any other starts at 1x below 3800 pixels wide whatever its
// density. Either starts HiDPI from 3800 pixels wide where its EDID is 32 inches across or more
// (rgfxDisplayEdidCentimetres keeps it under), and on a default mode of at least 3840 x 2160.
constexpr uint32_t kRgfxDisplayConnectionFlags = 0x800;

// The size the x86-64 framebuffer's EDID gives, in whole centimetres: that of a 1920 x 1080
// display at the display's density, whatever the window's size, so the EDID is at most 20 inches
// across (at 1x, or before the host gives a scale). x86-64 macOS starts a display HiDPI from 3800
// pixels wide where its EDID is 32 inches across or more, built in or not (read from macOS
// 26.7.2's CoreDisplay; measured: a 3888 x 2072 window on a host screen at 1x started HiDPI with
// an EDID of 90 x 48 cm). Each mode's own size, in millimetres, is still its native mode's at the
// density (ReimsGraphics::imageSize), and that is the size CoreGraphics reports.
constexpr uint32_t kRgfxDisplayEdidReferenceWidth = 1920;
constexpr uint32_t kRgfxDisplayEdidReferenceHeight = 1080;
constexpr uint32_t rgfxDisplayEdidCentimetres(uint32_t pixels, uint32_t scale120ths)
{
	return (rgfxDisplayMillimetres(pixels, scale120ths) + 5) / 10;
}

// One 10-bit chromaticity coordinate of the EDID, from thousandths.
constexpr uint32_t rgfxEdidChroma(uint32_t thousandths)
{
	return (thousandths * 1024 + 500) / 1000;
}

// The x86-64 framebuffer's EDID, one base block of 128 bytes: the identity, sRGB colour, the size
// (rgfxDisplayEdidCentimetres), the product name and, in an unspecified text descriptor, the
// native mode's size, `nativeWidth` x `nativeHeight`, with no timings. The EDID names the native
// size so each window size is a display of its own to the guest, as it was while the EDID's size
// followed the native mode: IODisplay keeps a display's settings under a hash of the whole EDID,
// and WindowServer's UUID for it changes with the EDID. WindowServer keeps the mode of a display
// it knows while the framebuffer lists that mode, and the framebuffer lists the mode set until
// another is set (RgfxDisplayModes.hpp), so an EDID that stayed the same across sizes left the
// guest on its old mode through every resize (measured on macOS 26.7.2). The display's modes are
// the driver's own (ReimsGraphics::getDisplayModes), whose flags give the default and native mode
// and the HiDPI ones (ReimsGraphics::modeFlags). For a timing an EDID lists, the OS programs a
// mode of its own, which lacked the HiDPI flag the driver gave it, and WindowServer started on
// that mode, at 1x (measured on the x86-64 guest, macOS 26.7.2).
constexpr uint32_t kRgfxDisplayEdidBytes = 128;
inline void rgfxDisplayEdid(uint8_t *b, uint32_t displayId, uint32_t scale120ths,
                            uint32_t nativeWidth, uint32_t nativeHeight)
{
	for (uint32_t i = 0; i < kRgfxDisplayEdidBytes; i++)
		b[i] = 0;
	static const uint8_t header[8] = {0x00, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00};
	for (uint32_t i = 0; i < 8; i++)
		b[i] = header[i];
	b[8] = static_cast<uint8_t>(kRgfxDisplayManufacturerPacked >> 8);
	b[9] = static_cast<uint8_t>(kRgfxDisplayManufacturerPacked);
	b[10] = static_cast<uint8_t>(kRgfxDisplayProduct);
	b[11] = static_cast<uint8_t>(kRgfxDisplayProduct >> 8);
	const uint32_t serial = rgfxDisplaySerial(displayId);
	b[12] = static_cast<uint8_t>(serial);
	b[13] = static_cast<uint8_t>(serial >> 8);
	b[14] = static_cast<uint8_t>(serial >> 16);
	b[15] = static_cast<uint8_t>(serial >> 24);
	b[16] = 0;                                            // week unspecified
	b[17] = static_cast<uint8_t>(kRgfxDisplayYear - 1990); // the year, counted from 1990
	b[18] = 1;                                            // EDID 1.4
	b[19] = 4;
	b[20] = 0xa5; // digital, 8 bits a component, DisplayPort
	b[21] = static_cast<uint8_t>(rgfxDisplayEdidCentimetres(kRgfxDisplayEdidReferenceWidth,
	                                                        scale120ths));
	b[22] = static_cast<uint8_t>(rgfxDisplayEdidCentimetres(kRgfxDisplayEdidReferenceHeight,
	                                                        scale120ths));
	b[23] = 120;  // gamma 2.2
	b[24] = 0x04; // RGB 4:4:4, sRGB the default colour space

	// sRGB's primaries and D65 white, 10 bits each.
	const uint32_t rx = rgfxEdidChroma(kRgfxDisplayRed.x), ry = rgfxEdidChroma(kRgfxDisplayRed.y);
	const uint32_t gx = rgfxEdidChroma(kRgfxDisplayGreen.x),
	               gy = rgfxEdidChroma(kRgfxDisplayGreen.y);
	const uint32_t bx = rgfxEdidChroma(kRgfxDisplayBlue.x), by = rgfxEdidChroma(kRgfxDisplayBlue.y);
	const uint32_t wx = rgfxEdidChroma(kRgfxDisplayWhite.x),
	               wy = rgfxEdidChroma(kRgfxDisplayWhite.y);
	b[25] = static_cast<uint8_t>(((rx & 3) << 6) | ((ry & 3) << 4) | ((gx & 3) << 2) | (gy & 3));
	b[26] = static_cast<uint8_t>(((bx & 3) << 6) | ((by & 3) << 4) | ((wx & 3) << 2) | (wy & 3));
	b[27] = static_cast<uint8_t>(rx >> 2);
	b[28] = static_cast<uint8_t>(ry >> 2);
	b[29] = static_cast<uint8_t>(gx >> 2);
	b[30] = static_cast<uint8_t>(gy >> 2);
	b[31] = static_cast<uint8_t>(bx >> 2);
	b[32] = static_cast<uint8_t>(by >> 2);
	b[33] = static_cast<uint8_t>(wx >> 2);
	b[34] = static_cast<uint8_t>(wy >> 2);
	// No established timings; no standard timings (01 01 each).
	for (uint32_t i = 38; i < 54; i++)
		b[i] = 0x01;

	// The descriptors: a dummy, the product name, the native size as text ("3888x2072"), each
	// padded as the descriptor asks, and a dummy.
	uint8_t *d = b + 54;
	d[3] = 0x10;
	d += 18;
	d[3] = 0xfc;
	const uint32_t len = sizeof(kRgfxDisplayName) - 1;
	for (uint32_t i = 0; i < 13; i++)
		d[5 + i] = i < len ? static_cast<uint8_t>(kRgfxDisplayName[i]) : (i == len ? 0x0a : 0x20);
	d += 18;
	d[3] = 0xfe;
	char text[13];
	uint32_t n = 0;
	const uint32_t sides[2] = {nativeWidth, nativeHeight};
	for (uint32_t side = 0; side < 2; side++) {
		if (side)
			text[n++] = 'x';
		char digits[10];
		uint32_t count = 0, v = sides[side] > 99999 ? 99999 : sides[side];
		do {
			digits[count++] = static_cast<char>('0' + v % 10);
			v /= 10;
		} while (v);
		while (count)
			text[n++] = digits[--count];
	}
	for (uint32_t i = 0; i < 13; i++)
		d[5 + i] = i < n ? static_cast<uint8_t>(text[i]) : (i == n ? 0x0a : 0x20);
	d += 18;
	d[3] = 0x10;

	b[126] = 0; // no extensions
	uint8_t sum = 0;
	for (uint32_t i = 0; i < 127; i++)
		sum = static_cast<uint8_t>(sum + b[i]);
	b[127] = static_cast<uint8_t>(-sum);
}

#endif // RGFX_DISPLAY_IDENTITY_HPP_
