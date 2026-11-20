// RgfxPagesBlock: the arithmetic of a pages block (kRgfxAccelMethodPagesBlock, 4): a block of
// a process's own that holds nothing but the guest RAM runs of pages of that process's, wired by
// the kext for as long as the block lives, which the device maps in place for the process's
// buffers over them (RGFX_FEATURE_PAGES_BUFFERS, BUFFER_OVER_PAGES). A chunked pages block
// (kRgfxAccelMethodPagesChunked, 12) is one over the kext's own chunks of kRgfxPagesChunk bytes
// instead, each one run of guest RAM at a multiple of its bytes (kRgfxPagesChunkMask), mapped into
// the process, its runs found the same way.
//
// Memory the process maps read-only takes a block where the guest took RGFX_FEATURE_READ_ONLY_PAGES
// alone: its pages are wired for reading, and the block's SURFACE_PAGES is marked
// RGFX_SURFACE_PAGES_READ_ONLY, for the host to keep every object over it from writing them
// (rgfxPagesWiring).
//
// Memory the bundle names unchanging (kRgfxPagesUnchanging) takes a block where the guest took
// RGFX_FEATURE_UNCHANGING_PAGES too: read-only memory its process can never write, a file's pages on
// a read-only volume mapped with no write in the mapping's maximum protection, as the bundle finds
// it at the call (its RgfxFilePages.hpp). It is the app's own memory, wired for reading as any
// read-only memory is: a shared mapping's pages are the file's own, and a private copy-on-write
// one's are copied for the process by the wire, as IOMemoryDescriptor::prepare's
// vm_map_create_upl resolves the entry's copy and vm_object_iopl_request faults each page into
// the process's own object (read from xnu's vm_map.c and vm_upl.c), as a Mac's driver's wire of an
// app's memory does; the process can write neither. The block's SURFACE_PAGES is marked
// RGFX_SURFACE_PAGES_UNCHANGING, and RGFX_SURFACE_PAGES_COPY_ON_WRITE where the bundle names the
// memory so too, for the host to read the pages once. Read-only memory not named unchanging may
// carry why not (kRgfxPagesWhyShift, the RGFX_PAGES_WHY_* values), which the block's SURFACE_PAGES
// carries on to the host in its RGFX_SURFACE_PAGES_WHY_SHIFT bits, where the guest took
// RGFX_FEATURE_UNCHANGING_PAGES: a diagnostic the host counts its copies by.
//
// Where the guest took RGFX_FEATURE_IMPORT_ADDRESSES, the device answers a pages block's
// SURFACE_PAGES with the GPU address of the block's view, at which the bundle places its buffers
// over the block (rgfxPagesAnswer). Where it took RGFX_FEATURE_IMPORT_ADDRESSES_LATER too, the
// device answers before the view has an address, with 0, and places the buffers itself.
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

#include "RgfxUserClientABI.hpp"
#include "reims_graphics_wire.h"

// The most bytes one pages block covers, its pages rounded out: 64 GiB, more than a guest has.
constexpr uint64_t kRgfxPagesBlockMaxBytes = 1ull << 36;

// The physical alignment of the memory IOBufferMemoryDescriptor::inTaskWithPhysicalMask gives
// under `mask`: one more than the mask's clear low bits, and at least a `page`. Read from xnu's
// source (iokit/Kernel/IOBufferMemoryDescriptor.cpp, initWithPhysicalMask, which
// inTaskWithPhysicalMask calls with an alignment of 1): `alignment = (~mask & (mask - 1)) + 1`,
// raised to the page; IOKernelAllocateWithPhysicalRestrict then asks kmem_alloc_contig for
// physical pages at a multiple of it. A mask of 0 asks for no alignment: the page.
constexpr uint64_t rgfxPhysicalMaskAlignment(uint64_t mask, uint64_t page)
{
	const uint64_t alignment = mask == 0 ? page : (~mask & (mask - 1)) + 1;
	return alignment < page ? page : alignment;
}

// The physical mask of the chunks of a chunked pages block (kRgfxAccelMethodPagesChunked): guest
// RAM below 2^48, each chunk at a multiple of its own bytes, so a block of chunks is runs of
// guest RAM that start and end at multiples of 64 KiB, one run a chunk at most: a host that
// imports pages as dma-bufs imports a block's view as one dma-buf of its runs, which a block of
// up to 1,024 chunks fits at udmabuf's stock limits (1,024 runs, 64 MiB).
constexpr uint64_t kRgfxPagesChunkMask = 0x0000'FFFF'FFFF'0000ull;
static_assert(rgfxPhysicalMaskAlignment(kRgfxPagesChunkMask, 0x1000) == kRgfxPagesChunk,
              "each chunk at a multiple of its own bytes");
static_assert(rgfxPhysicalMaskAlignment(kRgfxPagesChunkMask, 0x4000) == kRgfxPagesChunk,
              "on an arm64 guest's 16 KiB pages too");

// The pages a call's range lies in: `length` bytes from `base`, a page, and the call's first byte
// `offset` bytes into them.
struct RgfxPagesRange {
	uint64_t base;
	uint64_t length;
	uint64_t offset;
};

// How a pages block's pages are wired (kRgfxAccelMethodPagesBlock): not at all, for reading
// alone, or for reading and writing; kBadArgument where the call names them so no memory is
// mapped.
enum class RgfxPagesAccess { kRefused, kBadArgument, kRead, kReadWrite };

// How a pages block of memory the process maps writable or not is made: its wiring, and the
// RGFX_SURFACE_PAGES_* its SURFACE_PAGES carries.
struct RgfxPagesWiring {
	RgfxPagesAccess access;
	uint32_t flags;
};

// The wiring of a pages block over memory the process maps as `how` says (kRgfxPagesWritable,
// kRgfxPagesUnchanging, kRgfxPagesCopyOnWrite, and why not unchanging from kRgfxPagesWhyShift),
// given the feature word: writable memory for reading and writing, which resolves copy-on-write
// first, so a private mapping's pages are its own before the device maps them, a block of the
// process's (RGFX_SURFACE_PAGES_PRIVATE); read-only memory, where the word holds
// RGFX_FEATURE_READ_ONLY_PAGES, for reading alone, which asks of the VM system no access the
// process lacks, its block marked RGFX_SURFACE_PAGES_READ_ONLY too, for the host to keep every
// object over it from writing its pages (the process can post any message on its rings, so the
// mark, not the bundle's buffers' options, is what holds the line), with the why the bundle gave
// where the word holds RGFX_FEATURE_UNCHANGING_PAGES too, and without READ_ONLY_PAGES none, as a
// host that does not know the mark would not hold it; unchanging memory, where the word holds
// RGFX_FEATURE_UNCHANGING_PAGES too, as read-only memory, its block marked
// RGFX_SURFACE_PAGES_UNCHANGING too, and RGFX_SURFACE_PAGES_COPY_ON_WRITE where `how` says so, and
// without the bit none, as a host that does not know the mark could not tell its pages from any
// read-only memory's. Bits of no meaning, writable unchanging memory, copy-on-write memory not
// named unchanging and a why with writable or unchanging memory are no mapping at all
// (kBadArgument).
inline RgfxPagesWiring rgfxPagesWiring(uint64_t how, uint64_t word)
{
	const bool writable = (how & kRgfxPagesWritable) != 0;
	const bool unchanging = (how & kRgfxPagesUnchanging) != 0;
	const bool copyOnWrite = (how & kRgfxPagesCopyOnWrite) != 0;
	const uint64_t why = (how & kRgfxPagesWhyMask) >> kRgfxPagesWhyShift;
	if ((how & ~static_cast<uint64_t>(kRgfxPagesHowBits | kRgfxPagesWhyMask)) != 0 ||
	    (unchanging && writable) || (copyOnWrite && !unchanging) ||
	    (why != 0 && (writable || unchanging)))
		return {RgfxPagesAccess::kBadArgument, 0};
	if (writable)
		return {RgfxPagesAccess::kReadWrite, RGFX_SURFACE_PAGES_PRIVATE};
	if (!(word & RGFX_FEATURE_READ_ONLY_PAGES))
		return {RgfxPagesAccess::kRefused, 0};
	const uint32_t readOnly = RGFX_SURFACE_PAGES_PRIVATE | RGFX_SURFACE_PAGES_READ_ONLY;
	const bool marks = (word & RGFX_FEATURE_UNCHANGING_PAGES) != 0;
	if (!unchanging) {
		const uint32_t said = marks ? static_cast<uint32_t>(why) << RGFX_SURFACE_PAGES_WHY_SHIFT : 0;
		return {RgfxPagesAccess::kRead, readOnly | said};
	}
	if (!marks)
		return {RgfxPagesAccess::kRefused, 0};
	return {RgfxPagesAccess::kRead, readOnly | RGFX_SURFACE_PAGES_UNCHANGING |
	                                    (copyOnWrite ? RGFX_SURFACE_PAGES_COPY_ON_WRITE : 0u)};
}

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

// An arm64 guest's page, 16 KiB, which its kernel gives memory in and wires whole. Where the kext
// took RGFX_FEATURE_PAGES_16K (RgfxFeatures.hpp), every run of guest RAM it names to the device,
// and the length of each SURFACE_PAGES, is whole such pages, as a host whose page is 16 KiB maps
// guest RAM in no less (design.md, "Negotiated features").
constexpr uint64_t kRgfxPage16K = 0x4000;

// `length` bytes from a page's first byte, taken to the end of their last page of `page` (a power
// of two).
inline uint64_t rgfxToPageEnd(uint64_t length, uint64_t page)
{
	return (length + page - 1) & ~(page - 1);
}

// The `count` runs of a span of guest RAM, as the device is told of them in whole pages of `page`
// (a power of two): the last run taken to the end of its page. A span whose first byte is a
// page's then covers rgfxToPageEnd(its length) bytes. The bytes past the span's own in its last
// page are the same allocation's, which the kernel gives and wires in whole pages, so they are no
// other allocation's to read or write.
inline void rgfxRunsToPageEnd(RgfxPageRun *runs, uint32_t count, uint64_t page)
{
	if (count == 0)
		return;
	RgfxPageRun &last = runs[count - 1];
	last.length = rgfxToPageEnd(last.address + last.length, page) - last.address;
}

// What the `answered` bytes of a pages block's SURFACE_PAGES answer copied into `answer` say:
// false where they are fewer than a COMPLETE's, an answer not given; else its status, and the GPU
// address the host gave the block's view (RGFX_FEATURE_IMPORT_ADDRESSES) where the answer is an
// RgfxPagesAnswer and its status is OK, and 0 where it is a COMPLETE alone, as the device answers
// a guest that did not take the feature, or the host gave the block none.
inline bool rgfxPagesAnswer(const RgfxPagesAnswer &answer, uint32_t answered, uint32_t *status,
                            uint64_t *gpuAddress)
{
	if (answered < sizeof(RgfxComplete))
		return false;
	*status = answer.status;
	*gpuAddress = answered >= sizeof(RgfxPagesAnswer) && answer.status == RGFX_STATUS_OK
	                  ? answer.gpu_address
	                  : 0;
	return true;
}

#endif // RGFX_PAGES_BLOCK_HPP_
