// RgfxPagesBlock: the arithmetic of a pages block (kRgfxAccelMethodPagesBlock, 4): a block of
// a process's own that holds nothing but the guest RAM runs of pages of that process's, wired by
// the kext for as long as the block lives, which the device maps in place for the process's
// buffers over them (RGFX_FEATURE_PAGES_BUFFERS, BUFFER_OVER_PAGES).
//
// The range a call names is rounded out to whole pages, as the wiring takes them; the device's
// view starts at the first page, so a buffer over the call's first byte lies at its offset in that
// page (rgfxPagesRange). Its runs are counted in one pass over the wired range's physical
// segments, then written in a second straight into the block, through the kernel's mapping of it,
// so no scratch copy of them is made however large the range (rgfxPagesRuns): the wiring holds the
// pages, so both passes see the same segments.
//
// Header-only and free of the kernel's headers, so its native test (tests/pages_block_test.cpp)
// builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_PAGES_BLOCK_HPP_
#define RGFX_PAGES_BLOCK_HPP_

#include <stdint.h>

#include "reims_graphics_wire.h"

// The most bytes one pages block covers, its pages rounded out: 64 GiB, more than a guest has.
constexpr uint64_t kRgfxPagesBlockMaxBytes = 1ull << 36;

// The pages a call's range lies in: `length` bytes from `base`, a page, and the call's first byte
// `offset` bytes into them.
struct RgfxPagesRange {
	uint64_t base;
	uint64_t length;
	uint64_t offset;
};

// The whole pages of `page` (a power of two) that hold the `length` bytes at `address`: false
// where the range is empty, wraps, or covers more than kRgfxPagesBlockMaxBytes.
inline bool rgfxPagesRange(uint64_t address, uint64_t length, uint64_t page, RgfxPagesRange *r)
{
	if (length == 0 || page == 0 || (page & (page - 1)) != 0)
		return false;
	const uint64_t mask = page - 1;
	if (address > UINT64_MAX - length || address + length > UINT64_MAX - mask)
		return false;
	const uint64_t base = address & ~mask;
	const uint64_t end = (address + length + mask) & ~mask;
	if (end - base > kRgfxPagesBlockMaxBytes)
		return false;
	*r = {base, end - base, address - base};
	return true;
}

// The runs of a wired range of `length` bytes whose physical segments `segment` answers:
// `segment(at, &bytes)` gives the physical address of the range's byte `at` and how many bytes
// from it are contiguous (an address of 0, or no bytes, where it has none). Segments that meet are
// one run. Where `runs` is not null each run is written to it in order, never read back (it may be
// device memory), at most `capacity` of them. How many runs there are; 0 where a byte has no
// segment, or `runs` would take more than `capacity`.
template <typename Segment>
inline uint32_t rgfxPagesRuns(uint64_t length, Segment segment, RgfxPageRun *runs,
                              uint32_t capacity)
{
	uint32_t count = 0;
	RgfxPageRun last = {0, 0};
	for (uint64_t at = 0; at < length;) {
		uint64_t bytes = 0;
		const uint64_t phys = segment(at, &bytes);
		if (phys == 0 || bytes == 0)
			return 0;
		if (bytes > length - at)
			bytes = length - at;
		if (count && last.address + last.length == phys) {
			last.length += bytes;
		} else {
			if (count && runs)
				runs[count - 1] = last;
			if (count == UINT32_MAX || (runs && count == capacity))
				return 0;
			count++;
			last = {phys, bytes};
		}
		at += bytes;
	}
	if (count && runs)
		runs[count - 1] = last;
	return count;
}

// The bytes of a block that holds `count` runs: whole pages of `page`.
inline uint64_t rgfxPagesBlockSize(uint32_t count, uint64_t page)
{
	const uint64_t bytes = static_cast<uint64_t>(count) * sizeof(RgfxPageRun);
	return (bytes + page - 1) & ~(page - 1);
}

#endif // RGFX_PAGES_BLOCK_HPP_
