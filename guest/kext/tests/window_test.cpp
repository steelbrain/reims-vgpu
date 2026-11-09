// The native test of RgfxWindowSpans.hpp: a window maps in only whole pages past the region's
// reserved head, inside the window, and its own memory costs the process only the pages it reads.
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
	// On x86-64 a window page is a guest page, 4 KiB: the span of whole guest pages is those
	// pages, and a part of one maps the page it is in.
	check(rgfxWindowSpanLength(0x1000) == 0x1000 && rgfxWindowSpanLength(0xfd2000) == 0xfd2000,
	      "whole 4 KiB pages are their own span");
	check(rgfxWindowSpanLength(0x1800) == 0x2000 && rgfxWindowSpanLength(1) == 0x1000,
	      "a part of a page maps the page");
	// The write-combined split of a block: from its byte where that lies in the asked bytes, a
	// page's offset; none, the whole block write-back, where it lies in the block's rounding or
	// past it.
	check(rgfxWindowWriteBackPart(0x3000, 0x3000, 0x1000) == 0x1000, "write-back to the split");
	check(rgfxWindowWriteBackPart(0x1800, 0x2000, 0x1900) == 0x2000,
	      "a split in the rounding leaves the block write-back, a span");
	check(rgfxWindowWriteBackPart(0x2000, 0x2000, UINT64_MAX) == 0x2000, "no split, no part");
	// xnu charges a mapping of pageable memory that is not purgeable at its whole size, the
	// region's, to the process's footprint as the context opens.
	check((kRgfxWindowMemoryOptions & kRgfxWindowPageable) != 0, "the window's memory is pageable");
	check((kRgfxWindowMemoryOptions & kRgfxWindowPurgeable) != 0,
	      "and purgeable, so a process's footprint counts only the pages it reads there");
	check((kRgfxWindowMemoryOptions & kRgfxWindowDirectionInOut) == kRgfxWindowDirectionInOut,
	      "and the kernel's to write, which purgeable memory must be");
	std::printf("window_test: OK (%d checks)\n", checks);
	return 0;
}
