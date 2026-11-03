// The span rule of RgfxRegionSpans.hpp, built natively: which spans each message names, and
// that a short message of a type that names one is malformed.
#include <cstdio>
#include <cstring>

#include "RgfxRegionSpans.hpp"

static int g_checks = 0, g_failed = 0;
#define CHECK(e)                                                                   \
	do {                                                                           \
		g_checks++;                                                                \
		if (!(e)) {                                                                \
			g_failed++;                                                            \
			std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #e); \
		}                                                                          \
	} while (0)

template <typename T> static int spansOf(uint32_t type, const T &m, uint64_t size, RgfxRegionSpan *s)
{
	return rgfxRegionSpans(type, reinterpret_cast<const uint8_t *>(&m), size, s);
}

int main()
{
	RgfxRegionSpan s[kRgfxMaxRegionSpans] = {};

	RgfxBufferOverHostMemory buffer = {};
	buffer.region_offset = 0x8400000;
	buffer.length = 0x1000;
	CHECK(spansOf(RGFX_CMD_BUFFER_OVER_HOST_MEMORY, buffer, sizeof(buffer), s) == 1);
	CHECK(s[0].offset == 0x8400000 && s[0].length == 0x1000);
	CHECK(spansOf(RGFX_CMD_BUFFER_OVER_HOST_MEMORY, buffer, sizeof(buffer) - 8, s) == -1);

	// A buffer over pages names its pages block, not the bytes of the view it lies at.
	RgfxBufferOverPages pages = {};
	pages.block_offset = 0x8800000;
	pages.block_size = 0x2000;
	pages.offset = 0x123;
	pages.length = 0x40000;
	CHECK(spansOf(RGFX_CMD_BUFFER_OVER_PAGES, pages, sizeof(pages), s) == 1);
	CHECK(s[0].offset == 0x8800000 && s[0].length == 0x2000);
	CHECK(spansOf(RGFX_CMD_BUFFER_OVER_PAGES, pages, sizeof(pages) - 8, s) == -1);

	RgfxHeapOverHostMemory heap = {};
	heap.region_offset = 0x9000000;
	heap.size_bytes = 0x20000;
	CHECK(spansOf(RGFX_CMD_HEAP_OVER_HOST_MEMORY, heap, sizeof(heap), s) == 1);
	CHECK(s[0].offset == 0x9000000 && s[0].length == 0x20000);
	CHECK(spansOf(RGFX_CMD_HEAP_OVER_HOST_MEMORY, heap, 16, s) == -1);

	RgfxLibraryFromRegion chunk = {};
	chunk.region_offset = 0xa000000;
	chunk.byte_length = 0x4000;
	chunk.ack_offset = 0x9fff000;
	CHECK(spansOf(RGFX_CMD_LIBRARY_FROM_REGION, chunk, sizeof(chunk), s) == 2);
	CHECK(s[0].offset == 0xa000000 && s[0].length == 0x4000);
	CHECK(s[1].offset == 0x9fff000 && s[1].length == 8);
	CHECK(spansOf(RGFX_CMD_LIBRARY_FROM_REGION, chunk, 72, s) == -1);

	RgfxArgumentLayoutQuery layout = {};
	layout.reply_offset = 0xb000000;
	layout.reply_length = 0x4000;
	CHECK(spansOf(RGFX_CMD_ARGUMENT_LAYOUT, layout, sizeof(layout), s) == 1);
	CHECK(s[0].offset == 0xb000000 && s[0].length == 0x4000);
	CHECK(spansOf(RGFX_CMD_ARGUMENT_LAYOUT, layout, 40, s) == -1);

	// Messages that name no span of the region, whatever their size.
	RgfxObjectCreate event = {};
	CHECK(spansOf(RGFX_CMD_EVENT_CREATE, event, sizeof(event), s) == 0);
	RgfxBufferAtOffset atOffset = {};
	CHECK(spansOf(RGFX_CMD_BUFFER_AT_OFFSET, atOffset, sizeof(atOffset), s) == 0);
	RgfxLibraryCreate library = {};
	CHECK(spansOf(RGFX_CMD_LIBRARY_CREATE, library, 8, s) == 0);

	if (g_failed) {
		std::fprintf(stderr, "spans_test: %d of %d checks failed\n", g_failed, g_checks);
		return 1;
	}
	std::printf("spans_test: OK (%d checks)\n", g_checks);
	return 0;
}
