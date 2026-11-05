// RgfxBootConsole: the boot framebuffer the OS had when ReimsGraphics started, as the platform
// expert's console info holds it then (IOPlatformExpert::getConsoleInfo, xnu's PE_state.video):
// the framebuffer the firmware handed the kernel (the option ROM's GOP, firmware/), before
// IOFramebuffer moves the console onto the display's own framebuffer. ReimsGraphics publishes it
// as its RgfxBootFramebuffer property, so a boot's check reads it from the registry long after the
// kernel's message ring has wrapped:
//   Answered  whether the platform expert gave console info
//   Address   the console's base address (0 on a boot with no boot framebuffer)
//   Width, Height, RowBytes, Depth  its size in pixels, its row in bytes, its bits a pixel
// A boot whose firmware set no GOP mode has no boot framebuffer: Address is 0.
//
// Header-only and free of the kernel's headers, so its native test
// (tests/boot_console_test.cpp) builds it as the kext does.
//
// Our own code.
#ifndef RGFX_BOOT_CONSOLE_HPP_
#define RGFX_BOOT_CONSOLE_HPP_

#include <stdint.h>

// The property's name, on ReimsGraphics.
#define kRgfxBootFramebufferKey "RgfxBootFramebuffer"

struct RgfxBootConsole {
	bool answered;
	uint64_t address;
	uint32_t width;
	uint32_t height;
	uint32_t rowBytes;
	uint32_t depth;
};

inline uint32_t rgfxBootConsoleWord(uint64_t v)
{
	return v > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(v);
}

// The boot framebuffer from the console info's fields (PE_Video's v_baseAddr, v_rowBytes,
// v_width, v_height and v_depth), where `answered`; all zeros where the platform expert gave none.
// A size past 32 bits saturates.
inline RgfxBootConsole rgfxBootConsole(bool answered, uint64_t base, uint64_t rowBytes,
                                       uint64_t width, uint64_t height, uint64_t depth)
{
	RgfxBootConsole c = {};
	if (!answered)
		return c;
	c.answered = true;
	c.address = base;
	c.width = rgfxBootConsoleWord(width);
	c.height = rgfxBootConsoleWord(height);
	c.rowBytes = rgfxBootConsoleWord(rowBytes);
	c.depth = rgfxBootConsoleWord(depth);
	return c;
}

#endif // RGFX_BOOT_CONSOLE_HPP_
