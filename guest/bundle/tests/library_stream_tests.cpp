/*
 * library_stream_tests: a library streamed through a staging window made for it
 * (RgfxLibraryStream.hpp), against a host of the test's own on the kext's forwarding route: the
 * window's shape for each library, its fallbacks where the region refuses it, the chunks' bytes
 * and order, and the window going back once every chunk is copied out, or kept where one is not.
 * Part of the serialiser check (serializer_tests.cpp calls rgfxLibraryStreamTests). No Metal.
 */

#include <cstdint>
#include <cstring>
#include <vector>

#include "RgfxLibraryStream.hpp"
#include "RgfxObjectModel.hpp"
#include "RgfxRegion.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"
#include "reims_graphics_wire.h"
#include "rgfx_test_ring.hpp" // CHECK, g_checks

void rgfxLibraryStreamTests();

namespace {

constexpr uint64_t kPage = 0x1000;
constexpr uint64_t kMiB = 1ull << 20;

// The region's memory and the kext's blocks over it: first fit from the floor, each take and give
// logged; a take larger than `largest` refused, as a fragmented region refuses it.
struct Blocks {
	std::vector<uint8_t> mem;
	struct Held {
		uint64_t offset, size;
	};
	std::vector<Held> held;
	std::vector<uint64_t> takes, gives; // sizes asked for, offsets given back
	uint64_t largest = ~0ull;

	explicit Blocks(uint64_t bytes) : mem(RGFX_ALLOCATIONS_OFFSET + bytes, 0xee) {}

	static bool alloc(void *context, uint64_t size, uint64_t align, uint64_t *offset)
	{
		Blocks *b = static_cast<Blocks *>(context);
		b->takes.push_back(size);
		if (size > b->largest)
			return false;
		uint64_t at = RGFX_ALLOCATIONS_OFFSET;
		for (bool moved = true; moved;) {
			moved = false;
			at = (at + align - 1) & ~(align - 1);
			for (const Held &h : b->held)
				if (at < h.offset + h.size && h.offset < at + size) {
					at = h.offset + h.size;
					moved = true;
				}
		}
		if (at + size > b->mem.size())
			return false;
		b->held.push_back({at, size});
		*offset = at;
		return true;
	}

	static void release(void *context, uint64_t offset)
	{
		Blocks *b = static_cast<Blocks *>(context);
		b->gives.push_back(offset);
		for (size_t i = 0; i < b->held.size(); i++)
			if (b->held[i].offset == offset) {
				b->held.erase(b->held.begin() + i);
				return;
			}
	}
};

// The host: takes each LIBRARY_FROM_REGION off the forward, appends its chunk's bytes, and writes
// its ack (unless `ackless` names the chunk's place, which it never acks), answering the last
// chunk with `answer`.
struct Host {
	Blocks *blocks;
	std::vector<uint8_t> library;
	std::vector<RgfxLibraryFromRegion> chunks;
	uint32_t answer = RGFX_STATUS_OK;
	uint64_t ackless = ~0ull;
	uint64_t nextSeqno = 1;

	static bool forward(void *context, const void *message, uint32_t size, uint64_t *out_seqno,
	                    uint32_t *out_status)
	{
		Host *h = static_cast<Host *>(context);
		*out_seqno = h->nextSeqno++;
		if (out_status)
			*out_status = kRgfxNoAnswer;
		RgfxLibraryFromRegion m = {};
		CHECK(size == sizeof(m));
		std::memcpy(&m, message, sizeof(m));
		CHECK(m.header.type == RGFX_CMD_LIBRARY_FROM_REGION);
		CHECK(m.chunk_offset == h->library.size()); // in order
		h->chunks.push_back(m);
		const uint8_t *base = h->blocks->mem.data();
		h->library.insert(h->library.end(), base + m.region_offset,
		                  base + m.region_offset + m.byte_length);
		if (m.chunk_offset != h->ackless)
			std::memcpy(h->blocks->mem.data() + m.ack_offset, &m.ack_value, sizeof(m.ack_value));
		if (m.chunk_offset + m.byte_length >= m.total_length && out_status)
			*out_status = h->answer;
		return true;
	}
};

std::vector<uint8_t> pattern(uint64_t n)
{
	std::vector<uint8_t> v(n);
	for (uint64_t i = 0; i < n; i++)
		v[i] = static_cast<uint8_t>(i * 131 + (i >> 12));
	return v;
}

// Streams `length` bytes through a region of `regionBytes` whose largest take is `largest`:
// what the host made, the window's takes and gives, and the stream's answer.
struct Run {
	bool ok;
	uint32_t status;
	Host host;
	std::vector<uint64_t> takes, gives;
	size_t heldAfter;
};

Run stream(uint64_t length, uint64_t regionBytes, uint64_t largest, uint64_t ackless = ~0ull,
           uint64_t awaitUs = kRgfxStagingAwaitUs)
{
	Blocks blocks(regionBytes);
	blocks.largest = largest;
	RgfxRegion region(blocks.mem.data(), blocks.mem.size());
	region.setBlockSource(&Blocks::alloc, &Blocks::release, &blocks);
	Run run = {};
	run.host.blocks = &blocks;
	run.host.ackless = ackless;
	RgfxRingWriter writer(&Host::forward, &run.host, 4096);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	const std::vector<uint8_t> lib = pattern(length);
	RgfxObjId objId = 0;
	run.status = kRgfxNoAnswer;
	run.ok = rgfxStreamLibrary(region, ser, lib.data(), lib.size(), nullptr, &objId, &run.status,
	                           awaitUs);
	if (run.ok) {
		CHECK(objId != 0);
		CHECK(run.host.library == lib); // every byte, in order
		for (const RgfxLibraryFromRegion &c : run.host.chunks)
			CHECK(c.obj_id == objId && c.total_length == length);
	}
	run.takes = blocks.takes;
	run.gives = blocks.gives;
	run.heldAfter = blocks.held.size();
	return run;
}

// The window's shape: one chunk block for a library that fits a chunk, two for more, each chunk
// the library split evenly in whole pages, so the window is never more than the library and three
// pages; smaller chunk sizes in turn, down to a page.
void test_staging_shapes()
{
	RgfxStagingShape s = rgfxStagingShape(12296, kRgfxStagingChunkMax); // Photo Booth's library
	CHECK(s.chunk == 4 * kPage && s.blocks == 1 && s.bytes == 5 * kPage);
	s = rgfxStagingShape(kPage, kRgfxStagingChunkMax);
	CHECK(s.chunk == kPage && s.blocks == 1 && s.bytes == 2 * kPage);
	s = rgfxStagingShape(kRgfxStagingChunkMax, kRgfxStagingChunkMax);
	CHECK(s.chunk == kRgfxStagingChunkMax && s.blocks == 1);
	s = rgfxStagingShape(40 * kMiB, kRgfxStagingChunkMax); // two chunks of 20 MiB
	CHECK(s.chunk == 20 * kMiB && s.blocks == 2 && s.bytes == kPage + 40 * kMiB);
	s = rgfxStagingShape(160000000, kRgfxStagingChunkMax); // QuartzCore's: five chunks
	CHECK(s.blocks == 2 && s.chunk <= kRgfxStagingChunkMax && 5 * s.chunk >= 160000000 &&
	      4 * s.chunk < 160000000);
	for (uint64_t length = 1; length < 300 * kMiB; length = length * 3 + 7) {
		for (uint64_t max = kRgfxStagingChunkMax;;) {
			const RgfxStagingShape t = rgfxStagingShape(length, max);
			CHECK(t.chunk % kPage == 0 && t.chunk >= kPage && t.chunk <= max);
			CHECK(t.blocks == (length > t.chunk ? 2u : 1u));
			CHECK(t.bytes == kPage + t.blocks * t.chunk);
			CHECK(t.bytes <= length + 3 * kPage || t.bytes <= kPage + 2 * max);
			if (!rgfxStagingSmaller(&max))
				break;
		}
	}
	uint64_t max = kRgfxStagingChunkMax;
	std::vector<uint64_t> sizes = {max};
	while (rgfxStagingSmaller(&max))
		sizes.push_back(max);
	CHECK((sizes == std::vector<uint64_t>{32 * kMiB, 8 * kMiB, 2 * kMiB, 512 * 1024, 128 * 1024,
	                                      32 * 1024, 8 * 1024, kPage}));
}

// A library through a window made for it: the window is its size, not a process-long 64 MiB, and
// goes back once the stream is done, so a second stream takes a window of its own.
void test_window_made_and_given_back()
{
	Run small = stream(12296, 128 * kMiB, ~0ull);
	CHECK(small.ok && small.status == RGFX_STATUS_OK);
	CHECK((small.takes == std::vector<uint64_t>{5 * kPage}));
	CHECK(small.gives.size() == 1 && small.heldAfter == 0);
	CHECK(small.host.chunks.size() == 1 && small.host.chunks[0].byte_length == 12296);

	Run big = stream(70 * kMiB, 128 * kMiB, ~0ull); // three chunks through two blocks
	CHECK(big.ok && big.status == RGFX_STATUS_OK);
	const RgfxStagingShape s = rgfxStagingShape(70 * kMiB, kRgfxStagingChunkMax);
	CHECK((big.takes == std::vector<uint64_t>{s.bytes}) && s.blocks == 2);
	CHECK(big.host.chunks.size() == 3 && big.gives.size() == 1 && big.heldAfter == 0);
	// The two blocks in turn, each ack slot in the window's first page.
	CHECK(big.host.chunks[0].region_offset != big.host.chunks[1].region_offset);
	CHECK(big.host.chunks[0].region_offset == big.host.chunks[2].region_offset);
	CHECK(big.host.chunks[0].ack_offset == big.host.chunks[2].ack_offset);
	CHECK(big.host.chunks[1].ack_offset - big.host.chunks[0].ack_offset == 64);
	CHECK(big.host.chunks[2].ack_value > big.host.chunks[1].ack_value);
}

// Where the region refuses the window, smaller chunks are asked for in turn, and the library still
// streams whole; a region with no room even for page-sized chunks refuses the library, holding
// nothing.
void test_window_falls_back()
{
	Run frag = stream(10 * kMiB, 128 * kMiB, 80 * 1024); // gaps of 80 KiB at most
	CHECK(frag.ok && frag.status == RGFX_STATUS_OK);
	CHECK(frag.takes.size() > 1 && frag.takes.back() <= 80 * 1024);
	for (size_t i = 1; i < frag.takes.size(); i++)
		CHECK(frag.takes[i] < frag.takes[i - 1]);
	CHECK(frag.host.chunks.size() > 2 && frag.gives.size() == 1 && frag.heldAfter == 0);

	Run tiny = stream(12296, 128 * kMiB, 3 * kPage); // a page of acks and one page a chunk
	CHECK(tiny.ok && tiny.host.chunks.size() == 4 && tiny.takes.back() == 3 * kPage);

	Run none = stream(12296, 128 * kMiB, 2 * kPage - 1);
	CHECK(!none.ok && none.host.chunks.empty() && none.gives.empty() && none.heldAfter == 0);
	CHECK(none.takes.back() == 3 * kPage); // the last ask: two page-sized chunk blocks
}

// The device's refusal of the library is its answer; the window still goes back. A chunk the host
// never acks keeps the window, so nothing it may still read is given out again.
void test_refused_and_ackless()
{
	{
		Blocks blocks(16 * kMiB);
		RgfxRegion region(blocks.mem.data(), blocks.mem.size());
		region.setBlockSource(&Blocks::alloc, &Blocks::release, &blocks);
		Host host;
		host.blocks = &blocks;
		host.answer = RGFX_STATUS_REFUSED;
		RgfxRingWriter writer(&Host::forward, &host, 4096);
		RgfxIdSpace ids;
		RgfxSerializer ser(writer, ids);
		const std::vector<uint8_t> lib = pattern(20000);
		RgfxObjId objId = 0;
		uint32_t status = 0;
		CHECK(rgfxStreamLibrary(region, ser, lib.data(), lib.size(), nullptr, &objId, &status));
		CHECK(status == RGFX_STATUS_REFUSED && blocks.gives.size() == 1 && blocks.held.empty());
	}
	// Room for chunks of 512 KiB: six, the fifth (at 2 MiB) never acked. The stream is done, the
	// device made the library, and the window is kept.
	Run lastAckless = stream(3 * kMiB, 16 * kMiB, 2 * kMiB + kPage, /*ackless*/ 2 * kMiB,
	                         /*awaitUs*/ 2000);
	CHECK(lastAckless.ok && lastAckless.status == RGFX_STATUS_OK);
	CHECK(lastAckless.host.chunks.size() == 6 && lastAckless.host.chunks[0].byte_length == 512 * 1024);
	CHECK(lastAckless.gives.empty() && lastAckless.heldAfter == 1);
	// The first chunk never acked: its block is never written again, so the stream stops at the
	// third chunk, answers false and keeps the window.
	Run firstAckless = stream(3 * kMiB, 16 * kMiB, 2 * kMiB + kPage, /*ackless*/ 0,
	                          /*awaitUs*/ 2000);
	CHECK(!firstAckless.ok && firstAckless.host.chunks.size() == 2);
	CHECK(firstAckless.gives.empty() && firstAckless.heldAfter == 1);
}

} // namespace

void rgfxLibraryStreamTests()
{
	test_staging_shapes();
	test_window_made_and_given_back();
	test_window_falls_back();
	test_refused_and_ackless();
}
