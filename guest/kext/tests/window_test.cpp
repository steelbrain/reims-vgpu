// The native test of RgfxWindowSpans.hpp: a window maps in only whole pages past the region's
// reserved head, inside the window.
#include "RgfxWindowSpans.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "window_test: FAILED: %s\n", what);
		std::exit(1);
	}
}
} // namespace

int main()
{
	const uint64_t window = 0x10000000; // 256 MiB
	const uint64_t head = RGFX_ALLOCATIONS_OFFSET;
	check(rgfxWindowSpan(window, head, 0x1000), "the first page past the head");
	check(rgfxWindowSpan(window, window - 0x1000, 0x1000), "the window's last page");
	check(rgfxWindowSpan(window, head, window - head), "everything past the head");
	check(!rgfxWindowSpan(window, head - 0x1000, 0x1000), "no page of the head");
	check(!rgfxWindowSpan(window, RGFX_DEVICE_INFO_OFFSET & ~0xfffull, 0x1000),
	      "nor the device info's page");
	check(!rgfxWindowSpan(window, 0, 0x1000), "nor the control rings");
	check(!rgfxWindowSpan(window, head - 0x1000, 0x2000), "nor a span that starts in it");
	check(!rgfxWindowSpan(window, head, 0), "no empty span");
	check(!rgfxWindowSpan(window, head + 0x800, 0x1000), "no span off a page");
	check(!rgfxWindowSpan(window, head, 0x1800), "nor a part of a page");
	check(!rgfxWindowSpan(window, window - 0x1000, 0x2000), "nor one past the window");
	check(!rgfxWindowSpan(window, window, 0x1000), "nor one after it");
	check(!rgfxWindowSpan(window, 0xfffffffffffff000ull, 0x2000), "nor one that wraps");
	check(!rgfxWindowSpan(0, head, 0x1000), "a closed window maps nothing");
	std::printf("window_test: OK (%d checks)\n", checks);
	return 0;
}
