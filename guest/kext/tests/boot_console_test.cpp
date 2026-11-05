// The native test of RgfxBootConsole.hpp: the boot framebuffer ReimsGraphics publishes, from the
// platform expert's console info at its start.
#include "RgfxBootConsole.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "boot_console_test: FAILED: %s\n", what);
		std::exit(1);
	}
}
} // namespace

int main()
{
	check(std::strcmp(kRgfxBootFramebufferKey, "RgfxBootFramebuffer") == 0,
	      "the property's name is the one a boot's check reads");

	// The option ROM's GOP mode: a 1920 x 1080 framebuffer of 32 bits a pixel in BAR2.
	RgfxBootConsole gop = rgfxBootConsole(true, 0xc0000000ull, 7680, 1920, 1080, 32);
	check(gop.answered, "console info given");
	check(gop.address == 0xc0000000ull, "the boot framebuffer's address is the console's base");
	check(gop.width == 1920 && gop.height == 1080, "its size in pixels");
	check(gop.rowBytes == 7680 && gop.depth == 32, "its row and depth");

	// A boot with no GOP mode: console info with a base of 0, which a boot's check fails on.
	RgfxBootConsole none = rgfxBootConsole(true, 0, 0, 0, 0, 0);
	check(none.answered && none.address == 0, "no boot framebuffer: an address of 0");

	// No console info at all: every field 0, whatever the fields held.
	RgfxBootConsole unanswered = rgfxBootConsole(false, 0xc0000000ull, 7680, 1920, 1080, 32);
	check(!unanswered.answered && unanswered.address == 0 && unanswered.width == 0 &&
	          unanswered.height == 0 && unanswered.rowBytes == 0 && unanswered.depth == 0,
	      "no console info: all zeros");

	// A field past 32 bits saturates rather than wrapping to a small size.
	RgfxBootConsole wide = rgfxBootConsole(true, 1, 0x100000000ull, 0x1ffffffffull, 2, 3);
	check(wide.rowBytes == UINT32_MAX && wide.width == UINT32_MAX, "past 32 bits saturates");
	check(wide.height == 2 && wide.depth == 3, "the rest kept");

	std::printf("boot_console_test: %d checks passed\n", checks);
	return 0;
}
