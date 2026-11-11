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
// pixels per inch across its diagonal, at 1x otherwise, so on a host screen at 1.5x or more it
// runs HiDPI at every window size from 2048 x 1536 and on one at 1x it runs at 1x; any other starts
// at 1x below 3800 pixels wide whatever its density. Either starts HiDPI on a default mode of at
// least 3840 x 2160 pixels.
constexpr uint32_t kRgfxDisplayConnectionFlags = 0x800;

#endif // RGFX_DISPLAY_IDENTITY_HPP_
