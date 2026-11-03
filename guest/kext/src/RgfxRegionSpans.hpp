// RgfxRegionSpans: the spans of the device's region a process's lifecycle message names, which
// the accelerator kext requires to lie in the process's own blocks before it posts the message
// (RgfxAcceleratorUserClient::submitMethod). Header-only and free of the kernel's headers, so its
// native test (tests/spans_test.cpp) builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_REGION_SPANS_HPP_
#define RGFX_REGION_SPANS_HPP_

#include "reims_graphics_wire.h"

// One span of the region: `length` bytes at `offset`.
struct RgfxRegionSpan {
	uint64_t offset;
	uint64_t length;
};

// The most spans one message names.
constexpr unsigned kRgfxMaxRegionSpans = 2;

// The spans of the region the `size`-byte `message` of `type` names: a heap's or a buffer's block
// over host memory, the pages block a buffer over pages lies over, a library chunk's bytes and its
// ack slot, and the block an argument layout's answer is written into. Writes them to `spans` and
// answers how many (0 for a message that names none); -1 for a message of one of those types too
// short to hold its record.
inline int rgfxRegionSpans(uint32_t type, const uint8_t *message, uint64_t size,
                           RgfxRegionSpan spans[kRgfxMaxRegionSpans])
{
	switch (type) {
	case RGFX_CMD_HEAP_OVER_HOST_MEMORY: {
		RgfxHeapOverHostMemory m;
		if (size < sizeof(m))
			return -1;
		__builtin_memcpy(&m, message, sizeof(m));
		spans[0] = { m.region_offset, m.size_bytes };
		return 1;
	}
	case RGFX_CMD_BUFFER_OVER_HOST_MEMORY: {
		RgfxBufferOverHostMemory m;
		if (size < sizeof(m))
			return -1;
		__builtin_memcpy(&m, message, sizeof(m));
		spans[0] = { m.region_offset, m.length };
		return 1;
	}
	case RGFX_CMD_BUFFER_OVER_PAGES: {
		RgfxBufferOverPages m;
		if (size < sizeof(m))
			return -1;
		__builtin_memcpy(&m, message, sizeof(m));
		spans[0] = { m.block_offset, m.block_size };
		return 1;
	}
	case RGFX_CMD_LIBRARY_FROM_REGION: {
		RgfxLibraryFromRegion m;
		if (size < sizeof(m))
			return -1;
		__builtin_memcpy(&m, message, sizeof(m));
		spans[0] = { m.region_offset, m.byte_length };
		spans[1] = { m.ack_offset, sizeof(uint64_t) };
		return 2;
	}
	case RGFX_CMD_ARGUMENT_LAYOUT: {
		RgfxArgumentLayoutQuery m;
		if (size < sizeof(m))
			return -1;
		__builtin_memcpy(&m, message, sizeof(m));
		spans[0] = { m.reply_offset, m.reply_length };
		return 1;
	}
	default:
		return 0;
	}
}

#endif // RGFX_REGION_SPANS_HPP_
