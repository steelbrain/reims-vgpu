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

// The pixels per inch of a display whose native mode is `width` pixels wide: 220 for a native mode
// at least 2880 pixels wide, as a high-density panel, and 110 otherwise.
constexpr uint32_t rgfxDisplayPixelsPerInch(uint32_t width)
{
	return width >= 2880 ? 220 : 110;
}

// The size, in millimetres, of `pixels` of a display whose native mode is `width` pixels wide, at
// its pixels per inch.
constexpr uint32_t rgfxDisplayMillimetres(uint32_t pixels, uint32_t width)
{
	return (pixels * 254 + rgfxDisplayPixelsPerInch(width) * 5) /
	       (rgfxDisplayPixelsPerInch(width) * 10);
}

// sRGB's primaries and D65 white, the display's colours, in thousandths of CIE 1931 x and y.
struct RgfxDisplayChromaticity {
	uint32_t x, y;
};
constexpr RgfxDisplayChromaticity kRgfxDisplayRed = {640, 330};
constexpr RgfxDisplayChromaticity kRgfxDisplayGreen = {300, 600};
constexpr RgfxDisplayChromaticity kRgfxDisplayBlue = {150, 60};
constexpr RgfxDisplayChromaticity kRgfxDisplayWhite = {313, 329};

#endif // RGFX_DISPLAY_IDENTITY_HPP_
