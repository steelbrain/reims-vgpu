// The native test of what the display says it is (RgfxDisplayIdentity.hpp): the x86-64
// framebuffer's connection flags.
#include "RgfxDisplayIdentity.hpp"

#include <cstdio>
#include <cstdlib>

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

} // namespace

int main()
{
	connectionFlags();
	std::printf("display_identity_test: %d checks passed\n", checks);
	return 0;
}
