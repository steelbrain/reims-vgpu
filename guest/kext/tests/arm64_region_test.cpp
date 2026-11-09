// The native test of RgfxRegionMapping.hpp as the arm64e kext builds it (RGFX_GUEST_ARM64, the
// Makefile's rule for this test): a process's page is 16 KiB, so a window's span and a block are
// whole 16 KiB pages, and no write-combined part is offered, so every span is mapped write-back.
#include "RgfxBlockTable.hpp"
#include "RgfxSurfaceBlocks.hpp"
#include "RgfxWindowSpans.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

#if !RGFX_GUEST_ARM64
#error "arm64_region_test is built with RGFX_GUEST_ARM64=1, as the arm64e kext is"
#endif

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "arm64_region_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

const char kA = 0, kB = 0;
const void *const a = &kA;
const void *const b = &kB;

std::string readFile(const char *path)
{
	std::ifstream in(path);
	check(in.good(), path);
	return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// The length each mapSpan and unmapSpan call of `path` passes, its text from the call's first
// comma to its closing parenthesis, is a span's: rounded by rgfxWindowSpanLength, or one of the
// lengths that are whole window pages already, a block's as allocated (`bytes`, `bytes -
// combined` and `combined`, split by rgfxWindowWriteBackPart) and a grant's of the scanout
// (`scanout->length`, `g.length`), which currentScanout rounds.
void spansRounded(const char *path)
{
	const std::string source = readFile(path);
	const char *const whole[] = {"bytes", "bytes - combined", "combined", "scanout->length",
	                             "g.length"};
	for (const char *call : {"mapSpan(", "unmapSpan("}) {
		for (size_t at = source.find(call); at != std::string::npos;
		     at = source.find(call, at + 1)) {
			// Not a declaration or a definition: those name the length's type.
			const size_t open = at + std::string(call).size();
			const size_t comma = source.find(',', open);
			const size_t close = source.find(')', open);
			if (comma == std::string::npos || close == std::string::npos || comma > close)
				continue;
			std::string length = source.substr(comma + 1, close - comma - 1);
			if (length.find("UInt64") != std::string::npos)
				continue;
			const size_t first = length.find_first_not_of(" \n\t");
			length.erase(0, first == std::string::npos ? length.size() : first);
			bool ok = length.rfind("rgfxWindowSpanLength(", 0) == 0;
			for (const char *w : whole)
				ok = ok || length == w || length == std::string(w) + ", true";
			if (!ok)
				std::fprintf(stderr, "arm64_region_test: %s: a span of \"%s\"\n", path,
				             length.c_str());
			check(ok, "every span a window maps or takes back is whole window pages");
		}
	}
}
} // namespace

int main()
{
	check(kRgfxProcessPage == 0x4000, "an arm64 process's page is 16 KiB");
	check(kRgfxWindowPage == kRgfxProcessPage && kRgfxBlockPage == kRgfxProcessPage,
	      "the window and the blocks are in the process's pages");
	check(!kRgfxRegionWriteCombines && kRgfxKextFeatureRegionWriteCombined == 0,
	      "no write-combined part is offered");
	check(RGFX_ALLOCATIONS_OFFSET % kRgfxProcessPage == 0 &&
	          RGFX_CONTEXT_ALLOCATIONS_OFFSET % kRgfxProcessPage == 0,
	      "the region's reserved head and the scanout end on a 16 KiB page");

	// A span is whole 16 KiB pages: a 4 KiB one, or one off a 16 KiB page, would map or take back
	// the rest of its page with it.
	const uint64_t window = 0x10000000; // 256 MiB
	const uint64_t head = RGFX_ALLOCATIONS_OFFSET;
	check(rgfxWindowSpan(window, head, 0x4000), "a 16 KiB page past the head");
	check(rgfxWindowSpan(window, head + 0x4000, 0x8000), "two pages");
	check(!rgfxWindowSpan(window, head, 0x1000), "no 4 KiB span");
	check(!rgfxWindowSpan(window, head + 0x1000, 0x4000), "no span off a 16 KiB page");
	check(!rgfxWindowSpan(window, head + 0x4000, 0x5000), "no span that ends off one");

	// A block is whole 16 KiB pages at a 16 KiB offset, whatever it asks for, so two processes'
	// blocks never share a page.
	const uint64_t start = RGFX_CONTEXT_ALLOCATIONS_OFFSET;
	RgfxBlock blocks[4] = {};
	RgfxBlockTable t = {blocks, 0, 4, start, start + 0x100000, 0, 0, 0, 0, 0, 0, 0};
	uint64_t first = 0, second = 0;
	check(rgfxBlockAlloc(t, a, 1, 0x100, 0x1000, &first) == kRgfxBlockOk &&
	          first % 0x4000 == 0 && t.blocks[0].size == 0x4000,
	      "a 256-byte block takes a 16 KiB page");
	check(rgfxBlockAlloc(t, b, 2, 0x5000, 0x1000, &second) == kRgfxBlockOk &&
	          second == first + 0x4000 && t.blocks[1].size == 0x8000,
	      "the next process's block starts on the next page, rounded to whole pages");
	check(rgfxWindowSpan(start + 0x1000000, second, t.blocks[1].size),
	      "a block is a span a window maps");


	// The spans of a surface's block, as the kext lays it out: its holders use `size` bytes, whole
	// 4 KiB guest pages, and their windows map those bytes' 16 KiB pages, the runs past them. The
	// sizes are two of the blocks a 27.0.1 guest's WindowServer asked for, which the windows
	// refused to map as they were (0xe00002c2): a 1920x1080 RGBA16Float plane's and a smaller one.
	for (const uint64_t size : {0xfd3000ull, 0x71000ull}) {
		const uint64_t length = size;
		const uint64_t runs = 4 * sizeof(RgfxPageRun);
		const uint64_t total = rgfxSurfaceRunsFrom(size) + rgfxSurfaceRunRoom(length);
		RgfxBlock surfaceBlocks[2] = {};
		RgfxBlockTable st = {surfaceBlocks, 0, 2, start, start + 0x4000000, 0, 0, 0, 0, 0, 0, 0};
		uint64_t at = 0;
		check(rgfxBlockAlloc(st, a, 1, total, 0x1000, &at) == kRgfxBlockOk, "the block is made");
		const uint64_t span = rgfxWindowSpanLength(size);
		check(!rgfxWindowSpan(start + 0x4000000, at, size),
		      "the bytes the holders use are no span on 16 KiB pages");
		check(span % 0x4000 == 0 && span >= size && span - size < 0x4000 &&
		          rgfxWindowSpan(start + 0x4000000, at, span),
		      "the span the windows map is those bytes' 16 KiB pages");
		check(rgfxSurfaceRunsFrom(size) >= span && rgfxSurfaceRunsFrom(size) + runs <= total &&
		          total <= st.blocks[0].size,
		      "the runs lie past the span, in the block");
	}
	// The scanout's grant: its rows of a mode, whole 16 KiB pages from the region's head, inside
	// the scanout's span.
	const uint64_t scanout = rgfxWindowSpanLength(7680ull * 1080);
	check(scanout == 0x7ec000 && rgfxWindowSpan(0x10000000, RGFX_ALLOCATIONS_OFFSET, scanout) &&
	          rgfxWindowSpanLength(RGFX_SCANOUT_SPAN) == RGFX_SCANOUT_SPAN,
	      "a 1920x1080 scanout's grant is a span, as is the largest");
	// A block's write-combined split, offered on x86-64 alone, falls on a block page or leaves
	// the block whole.
	check(rgfxWindowWriteBackPart(0x5000, 0x8000, 0x6000) == 0x8000 &&
	          rgfxWindowWriteBackPart(0x9000, 0xc000, 0x4000) == 0x4000,
	      "the write-back part of a block is a span");
	// And every caller of mapSpan and unmapSpan passes a span's length.
	spansRounded("src/RgfxAcceleratorContexts.cpp");
	spansRounded("src/RgfxAcceleratorSurfaces.cpp");

	std::printf("arm64_region_test: %d checks passed\n", checks);
	return 0;
}
