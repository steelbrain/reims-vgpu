/*
 * serializer_tests: the reims-graphics guest serialiser's self-contained check,
 * run on the Mac (guest/bundle/Makefile `test`). It builds a plain-buffer region,
 * a ring writer over it, and a serialiser, posts every message the slice covers,
 * and reads each back with a minimal host-side consumer (the mirror of
 * crates/reims-vgpu-graphics/src/ring.rs), asserting the type, size and fields on the
 * wire. No Metal, no device: this validates the serialiser's bytes alone.
 *
 * Exits 0 and prints a one-line summary when every check passes; aborts on the
 * first failure.
 */

#include <IOKit/IOReturn.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/sysctl.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <vector>

#include "RgfxArgumentLayout.hpp"
#include "RgfxBinaryArchiveFile.hpp"
#include "RgfxCaps.hpp"
#include "../../kext/src/RgfxUserClientABI.hpp"
#include "../../kext/src/RgfxSubmitGate.hpp"
#include <utility>
#include "RgfxEncoder.hpp"
#include "RgfxFeatureProfile.hpp"
#include "RgfxFilePages.hpp"
#include "RgfxHeapRanges.hpp"
#include "RgfxLibraryKey.hpp"
#include "RgfxMemoryShadow.hpp"
#include "RgfxObjectModel.hpp"
#include <chrono>
#include <thread>

#include "RgfxReplyReader.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxRegion.hpp"
#include "RgfxSerializer.hpp"
#include "reims_graphics_wire.h"
#include "rgfx_argument_reply.hpp"
#include "rgfx_test_ring.hpp" // CHECK, g_checks, TestRing, expectMessage

void rgfxSlabTests();
void rgfxLibraryStreamTests();
void rgfxFunctionConstantsTests();

namespace {

void test_id_space()
{
	RgfxIdSpace ids;
	CHECK(ids.allocObjId() == 1);
	CHECK(ids.allocObjId() == 2);
	CHECK(ids.allocResourceId() == 1);
	CHECK(ids.allocResourceId() == 2);

	uint64_t a = ids.allocAddress(100, 256);
	uint64_t b = ids.allocAddress(1, 4096);
	CHECK(a >= RgfxIdSpace::kDefaultAddressBase);
	CHECK((a % RgfxIdSpace::kAddressGranule) == 0);
	CHECK((b % RgfxIdSpace::kAddressGranule) == 0);
	CHECK(b >= a + RgfxIdSpace::addressSpan(100)); // a's span, its guard page included
}

// An object id is given once in a context: ids only grow, and nothing gives one back (a destroy
// frees none), so the kext's destroy behind a create it stopped waiting for names no later object
// of the same context. Only a new context's bases start them again, from its base + 1: a later
// context under a number given again can name the id, which the host keeps in ring order
// (graphics/tests.rs, a_create_reusing_a_pending_icb_s_id_after_its_destroy_keeps_its_object).
void test_obj_ids_are_given_once()
{
	RgfxIdSpace ids;
	RgfxObjId last = 0;
	for (int i = 0; i < 1000; i++) {
		const RgfxObjId id = ids.allocObjId();
		CHECK(id > last);
		last = id;
	}
	const uint64_t base = uint64_t(7) << RGFX_CONTEXT_OBJ_ID_SHIFT;
	ids.setBases(base, RgfxIdSpace::kDefaultAddressBase);
	CHECK(ids.allocObjId() == base + 1);
	CHECK(ids.allocObjId() == base + 2);
}

// GPU addresses hold Metal's span of a pin: each starts a 32 KiB granule, holds the pages its
// bytes touch and one guard page, rounded up to a granule, and no two live spans meet, so the
// host pins every one. A span given back is picked again once the picks have gone round the
// context's range, joined with the free spans beside it.
void test_address_picks()
{
	const uint64_t page = RgfxIdSpace::kAddressPage, granule = RgfxIdSpace::kAddressGranule;
	CHECK(RgfxIdSpace::addressSpan(0) == 2 * page);
	CHECK(RgfxIdSpace::addressSpan(1) == 2 * page);
	CHECK(RgfxIdSpace::addressSpan(page) == 2 * page);
	CHECK(RgfxIdSpace::addressSpan(page + 1) == 4 * page);
	CHECK(RgfxIdSpace::addressSpan(3 * page) == 4 * page);
	CHECK(RgfxIdSpace::addressSpan(64 * page) == 66 * page);
	CHECK(RgfxIdSpace::addressSpan(65 * page) == 66 * page);

	// A context's range, as the kext hands it (a unit, the fourth), its picks granule-aligned
	// and their spans apart, whatever their sizes: one of a heap's 1 MiB, then a buffer's 16 KiB.
	RgfxIdSpace ids;
	const uint64_t base = RGFX_CONTEXT_VA_BASE + 3 * RGFX_CONTEXT_VA_UNIT;
	ids.setBases(uint64_t{4} << RGFX_CONTEXT_OBJ_ID_SHIFT, base, RGFX_CONTEXT_VA_UNIT);
	uint64_t prev = 0, prevSize = 0;
	const uint64_t sizes[] = {1u << 20, 0x4000, 256, 1, 0x4001, 4u << 20, 0x8000};
	for (uint64_t size : sizes) {
		uint64_t at = ids.allocAddress(size, 256);
		CHECK(at >= base && at % granule == 0);
		if (prev != 0)
			CHECK(at >= prev + RgfxIdSpace::addressSpan(prevSize));
		prev = at;
		prevSize = size;
	}
	// A larger alignment is honoured.
	CHECK(ids.allocAddress(1, 0x10'0000) % 0x10'0000 == 0);

	// Next fit: a span given back is not picked again at once.
	uint64_t a = ids.allocAddress(1, 1);
	ids.freeAddress(a);
	uint64_t b = ids.allocAddress(1, 1);
	CHECK(b > a);
	ids.freeAddress(0);       // no pick: ignored
	ids.freeAddress(a);       // given back already: ignored
	ids.freeAddress(b + 256); // no pick starts there: ignored

	// Round the range: 1 GiB picks till none is left, then two side by side given back, which
	// a pick of both their spans takes, at the first's address.
	RgfxIdSpace full;
	full.setBases(uint64_t{5} << RGFX_CONTEXT_OBJ_ID_SHIFT, base, RGFX_CONTEXT_VA_UNIT);
	const uint64_t gib = uint64_t{1} << 30;
	std::vector<uint64_t> picks;
	for (;;) {
		uint64_t at = full.allocAddress(gib, 1);
		if (at == 0)
			break;
		picks.push_back(at);
	}
	CHECK(picks.size() == RGFX_CONTEXT_VA_UNIT / RgfxIdSpace::addressSpan(gib));
	CHECK(picks.front() == base);
	CHECK(picks.back() + RgfxIdSpace::addressSpan(gib) <= base + RGFX_CONTEXT_VA_UNIT);
	full.freeAddress(picks[10]);
	full.freeAddress(picks[11]);
	uint64_t both = full.allocAddress(2 * RgfxIdSpace::addressSpan(gib) - 2 * page, 1);
	CHECK(both == picks[10]);
	CHECK(full.allocAddress(gib, 1) == 0); // no other 1 GiB span is free
	full.freeAddress(picks[3]);
	CHECK(full.allocAddress(gib, 1) == picks[3]);

	// Picks from several threads at once never share a span.
	RgfxIdSpace shared;
	std::vector<uint64_t> got[4];
	std::vector<std::thread> threads;
	for (auto &g : got)
		threads.emplace_back([&shared, &g] {
			for (int i = 0; i < 500; i++) {
				uint64_t at = shared.allocAddress(0x1000, 1);
				g.push_back(at);
				if (i % 3 == 0)
					shared.freeAddress(at);
			}
		});
	for (auto &t : threads)
		t.join();
	std::vector<uint64_t> all;
	for (auto &g : got)
		all.insert(all.end(), g.begin(), g.end());
	std::sort(all.begin(), all.end());
	CHECK(std::adjacent_find(all.begin(), all.end()) == all.end());
}

// A Private block source standing in for the kext: it hands blocks of the length asked for
// plus `extra`, at addresses from `next` on, each `gap` bytes past the last one's end, or
// refuses where `refuse` holds; and records what it took and was given back.
struct FakePrivate {
	uint64_t next = uint64_t{1} << 44; // the host's Private range on the Vulkan platform
	uint64_t extra = 0;
	uint64_t gap = uint64_t{1} << 30;
	bool refuse = false;
	std::vector<std::pair<uint64_t, uint64_t>> taken, given;
	std::mutex lock;
};

bool fakePrivateTake(void *context, uint64_t size, uint64_t *gpu_address, uint64_t *length)
{
	FakePrivate *f = static_cast<FakePrivate *>(context);
	std::lock_guard<std::mutex> hold(f->lock);
	if (f->refuse)
		return false;
	*gpu_address = f->next;
	*length = size + f->extra;
	f->taken.emplace_back(*gpu_address, *length);
	f->next += *length + f->gap;
	return true;
}

void fakePrivateGive(void *context, uint64_t gpu_address, uint64_t length)
{
	FakePrivate *f = static_cast<FakePrivate *>(context);
	std::lock_guard<std::mutex> hold(f->lock);
	f->given.emplace_back(gpu_address, length);
}

// Where `address` lies in one of the blocks `f` handed.
bool inPrivateBlock(FakePrivate &f, uint64_t address, uint64_t span)
{
	for (const auto &block : f.taken)
		if (address >= block.first && address - block.first <= block.second &&
		    span <= block.second - (address - block.first))
			return true;
	return false;
}

// Private objects' addresses (RGFX_FEATURE_PRIVATE_BLOCKS): picked as the context's are, but in
// blocks the host gives the context, a block of 16 MiB taken where none held has room, a pick past
// that a block of its own given back at its free, and one left empty given back past the one the
// space keeps.
void test_private_address_picks()
{
	const uint64_t page = RgfxIdSpace::kAddressPage, granule = RgfxIdSpace::kAddressGranule;
	const uint64_t base = RGFX_CONTEXT_VA_BASE + 3 * RGFX_CONTEXT_VA_UNIT;
	const uint64_t blockBytes = RgfxIdSpace::kPrivateBlockBytes;
	CHECK(blockBytes == uint64_t{16} << 20 && blockBytes % RGFX_PRIVATE_BLOCK_ALIGN == 0);

	// No source: the context's range, as allocAddress picks.
	{
		RgfxIdSpace ids;
		ids.setBases(uint64_t{4} << RGFX_CONTEXT_OBJ_ID_SHIFT, base, RGFX_CONTEXT_VA_UNIT);
		CHECK(!ids.privateBlocks());
		const uint64_t at = ids.allocPrivateAddress(0x4000, 256);
		CHECK(at == base);
		CHECK(ids.privateBlocksHeld() == 0 && ids.privatePicksLive() == 0);
		ids.freeAddress(at);
	}

	// The first pick takes a block of 16 MiB and lies at its start; the next, next fit after it;
	// a larger alignment is honoured inside the block.
	FakePrivate f;
	f.extra = 0x1'0000; // the host may answer more than asked: the space uses all of it
	RgfxIdSpace ids;
	ids.setBases(uint64_t{4} << RGFX_CONTEXT_OBJ_ID_SHIFT, base, RGFX_CONTEXT_VA_UNIT);
	ids.setPrivateBlockSource(&fakePrivateTake, &fakePrivateGive, &f);
	CHECK(ids.privateBlocks());
	const uint64_t a = ids.allocPrivateAddress(1u << 20, 256);
	CHECK(f.taken.size() == 1 && f.taken[0].second == blockBytes + f.extra);
	const uint64_t first = f.taken[0].first;
	CHECK(a == first);
	const uint64_t b = ids.allocPrivateAddress(0x4000, 1);
	CHECK(b == a + RgfxIdSpace::addressSpan(1u << 20));
	const uint64_t c = ids.allocPrivateAddress(1, 0x10'0000);
	CHECK(c % 0x10'0000 == 0 && c > b && inPrivateBlock(f, c, granule));
	CHECK(ids.privateBlocksHeld() == 1 && ids.privatePicksLive() == 3);
	// None of them is the context's own: its range's first pick is still its base.
	const uint64_t own = ids.allocAddress(1, 1);
	CHECK(own == base);
	ids.freeAddress(own);

	// A pick whose span and alignment are past a block's length takes a block of their length,
	// rounded up to the host's 64 KiB, which is the pick's own and lies at its first aligned byte.
	const uint64_t big = uint64_t{32} << 20;
	const uint64_t d = ids.allocPrivateAddress(big, 0x20'0000);
	CHECK(f.taken.size() == 2);
	CHECK(f.taken[1].second - f.extra ==
	      ((RgfxIdSpace::addressSpan(big) + 0x20'0000 + RGFX_PRIVATE_BLOCK_ALIGN - 1) &
	       ~(RGFX_PRIVATE_BLOCK_ALIGN - 1)));
	CHECK(d % 0x20'0000 == 0 && d - f.taken[1].first < 0x20'0000 &&
	      inPrivateBlock(f, d, RgfxIdSpace::addressSpan(big)));
	CHECK(ids.privateBlocksHeld() == 2 && ids.privatePicksLive() == 4);

	// Next fit: a freed span is picked again only once the picks have gone round, so the pick
	// after b's free lies past c, not at b; and the big pick's block, the room past it among it,
	// takes no other pick.
	ids.freeAddress(b);
	const uint64_t g = ids.allocPrivateAddress(0x4000, 1);
	CHECK(g == c + RgfxIdSpace::addressSpan(1) && f.taken.size() == 2);

	// Freed: an address no pick starts at is ignored; the big pick's block goes back with it, at
	// once; the first block, emptied, is kept, the only empty one.
	ids.freeAddress(0);
	ids.freeAddress(a + page);
	CHECK(f.given.empty());
	ids.freeAddress(d);
	CHECK(f.given.size() == 1 && f.given[0] == f.taken[1]);
	CHECK(ids.privateBlocksHeld() == 1 && ids.privatePicksLive() == 3);
	ids.freeAddress(d); // given back already: ignored
	ids.freeAddress(a);
	ids.freeAddress(g);
	ids.freeAddress(c);
	CHECK(f.given.size() == 1);
	CHECK(ids.privateBlocksHeld() == 1 && ids.privatePicksLive() == 0);
	ids.freeAddress(c); // freed already: ignored
	CHECK(f.given.size() == 1);
	// The kept block takes the next pick: no new block.
	const uint64_t e = ids.allocPrivateAddress(0x4000, 1);
	CHECK(f.taken.size() == 2 && e >= first && e - first < f.taken[0].second);
	ids.freeAddress(e);

	// Blocks side by side: their free ranges never join, so no pick spans two blocks. The first
	// block's last granule is left free and the second's whole; a pick of a block's length then
	// lies at the second's start, not one granule before it.
	{
		FakePrivate g;
		g.gap = 0;
		RgfxIdSpace side;
		side.setBases(uint64_t{5} << RGFX_CONTEXT_OBJ_ID_SHIFT, base, RGFX_CONTEXT_VA_UNIT);
		side.setPrivateBlockSource(&fakePrivateTake, &fakePrivateGive, &g);
		const uint64_t fill = side.allocPrivateAddress(blockBytes - 3 * page, 1);
		CHECK(fill == g.next - blockBytes && RgfxIdSpace::addressSpan(blockBytes - 3 * page) ==
		                                         blockBytes - granule);
		const uint64_t second = side.allocPrivateAddress(2 * page, 1);
		CHECK(g.taken.size() == 2 && second == g.taken[1].first &&
		      second == g.taken[0].first + blockBytes);
		side.freeAddress(second); // the second block, empty, kept
		CHECK(g.given.empty());
		const uint64_t whole = side.allocPrivateAddress(blockBytes - page, 1);
		CHECK(RgfxIdSpace::addressSpan(blockBytes - page) == blockBytes);
		CHECK(whole == g.taken[1].first && g.taken.size() == 2);
		side.freeAddress(whole);
		side.freeAddress(fill);
		CHECK(g.given.size() == 1);
	}

	// Refused: the context's range, and no block held.
	{
		FakePrivate r;
		r.refuse = true;
		RgfxIdSpace refused;
		refused.setBases(uint64_t{6} << RGFX_CONTEXT_OBJ_ID_SHIFT, base, RGFX_CONTEXT_VA_UNIT);
		refused.setPrivateBlockSource(&fakePrivateTake, &fakePrivateGive, &r);
		const uint64_t at = refused.allocPrivateAddress(0x4000, 1);
		CHECK(at == base && refused.privateBlocksHeld() == 0);
		const uint64_t next = refused.allocPrivateAddress(0x4000, 1);
		CHECK(next == base + RgfxIdSpace::addressSpan(0x4000));
		refused.freeAddress(at);
		refused.freeAddress(next);
		CHECK(r.given.empty());
	}

	// Picks from several threads at once never share a span, and each lies in a block taken. The
	// picks, 62.5 MiB of spans, go round several 16 MiB blocks, so a freed span is picked again:
	// the live picks are the ones that never meet.
	{
		FakePrivate t;
		RgfxIdSpace shared;
		shared.setBases(uint64_t{7} << RGFX_CONTEXT_OBJ_ID_SHIFT, base, RGFX_CONTEXT_VA_UNIT);
		shared.setPrivateBlockSource(&fakePrivateTake, &fakePrivateGive, &t);
		std::vector<uint64_t> got[4], kept[4];
		std::vector<std::thread> threads;
		for (int k = 0; k < 4; k++)
			threads.emplace_back([&shared, &g = got[k], &live = kept[k]] {
				for (int i = 0; i < 500; i++) {
					uint64_t at = shared.allocPrivateAddress(0x1000, 1);
					g.push_back(at);
					if (i % 3 == 0)
						shared.freeAddress(at);
					else
						live.push_back(at);
				}
			});
		for (auto &th : threads)
			th.join();
		std::vector<uint64_t> live;
		for (int k = 0; k < 4; k++) {
			for (uint64_t at : got[k])
				CHECK(inPrivateBlock(t, at, granule));
			live.insert(live.end(), kept[k].begin(), kept[k].end());
		}
		std::sort(live.begin(), live.end());
		CHECK(live.size() == 4 * 333);
		for (size_t i = 1; i < live.size(); i++)
			CHECK(live[i] - live[i - 1] >= RgfxIdSpace::addressSpan(0x1000));
		CHECK(t.taken.size() > 1);
	}
}

// A device heap in a Private block lies at a multiple of kHeapAddressAlign (256 KiB) from the
// block's first byte, which the host's 64 KiB need not make a multiple of it: the host binds the
// heap's textures at that offset in the block's memory, at their alignment. Here the block starts
// on an odd 64 KiB and a buffer's pick lies at its start: the heap lies 256 KiB into the block,
// not at the granule past the buffer (block + 32 KiB) nor at the next absolute 256 KiB
// (block + 192 KiB); HEAP_CREATE carries that address; a larger alignment asked counts from the
// block too; a buffer picked after it keeps its granule. With no source, the heap lies where any
// pick in the context's range does, at the granule past a buffer's.
void test_private_heaps_align_from_their_block()
{
	const uint64_t heapAlign = RgfxSerializer::kHeapAddressAlign;
	const uint64_t granule = RgfxIdSpace::kAddressGranule;
	CHECK(heapAlign == 0x40000 && heapAlign % granule == 0);
	const uint64_t base = RGFX_CONTEXT_VA_BASE + 3 * RGFX_CONTEXT_VA_UNIT;
	TestRing ring(0x8000);
	ring.setConsumerIdle(1);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);

	FakePrivate f;
	f.next += RGFX_PRIVATE_BLOCK_ALIGN; // the block's first byte on an odd 64 KiB
	RgfxIdSpace ids;
	ids.setBases(uint64_t{4} << RGFX_CONTEXT_OBJ_ID_SHIFT, base, RGFX_CONTEXT_VA_UNIT);
	ids.setPrivateBlockSource(&fakePrivateTake, &fakePrivateGive, &f);
	RgfxSerializer ser(writer, ids);
	const uint64_t buffer = ids.allocPrivateAddress(0x4000, 256);
	CHECK(f.taken.size() == 1);
	const uint64_t block = f.taken[0].first;
	CHECK(buffer == block && block % heapAlign == RGFX_PRIVATE_BLOCK_ALIGN);
	CHECK(RgfxIdSpace::addressSpan(0x4000) == granule);

	RgfxCreated heap = ser.createHeap(0x10000, 0, 1 /*Placement*/);
	CHECK(heap.posted && heap.gpu_address == block + heapAlign);
	CHECK((heap.gpu_address - block) % heapAlign == 0 && heap.gpu_address % heapAlign != 0);
	auto hm = expectMessage<RgfxHeapCreate>(ring, RGFX_CMD_HEAP_CREATE);
	CHECK(hm.obj_id == heap.obj_id && hm.gpu_address == heap.gpu_address);

	RgfxCreated wide = ser.createHeap(0x10000, 0, 1, 0x10'0000);
	CHECK(wide.posted && wide.gpu_address == block + 0x10'0000);
	CHECK(expectMessage<RgfxHeapCreate>(ring, RGFX_CMD_HEAP_CREATE).gpu_address ==
	      wide.gpu_address);
	const uint64_t after = ids.allocPrivateAddress(0x4000, 256);
	CHECK(after == wide.gpu_address + RgfxIdSpace::addressSpan(0x10000));
	CHECK(f.taken.size() == 1 && ids.privatePicksLive() == 4);
	// A heap past a block's length lies at the first byte of a block of its own, which goes back
	// at its free.
	RgfxCreated large = ser.createHeap(uint64_t{32} << 20, 0, 1);
	CHECK(f.taken.size() == 2 && f.taken[1].first % heapAlign != 0);
	CHECK(large.posted && large.gpu_address == f.taken[1].first);
	CHECK(expectMessage<RgfxHeapCreate>(ring, RGFX_CMD_HEAP_CREATE).gpu_address ==
	      large.gpu_address);
	ids.freeAddress(large.gpu_address);
	CHECK(f.given.size() == 1 && f.given[0] == f.taken[1]);
	for (uint64_t at : {buffer, heap.gpu_address, wide.gpu_address, after})
		ids.freeAddress(at);
	CHECK(ids.privatePicksLive() == 0);

	// No source: the context's range, at the granule past a buffer's pick, as before.
	RgfxIdSpace own;
	own.setBases(uint64_t{5} << RGFX_CONTEXT_OBJ_ID_SHIFT, base, RGFX_CONTEXT_VA_UNIT);
	RgfxSerializer ownSer(writer, own);
	const uint64_t first = own.allocPrivateAddress(0x4000, 256);
	CHECK(first == base);
	RgfxCreated ownHeap = ownSer.createHeap(0x10000, 0, 1);
	CHECK(ownHeap.posted && ownHeap.gpu_address == base + granule);
	CHECK(expectMessage<RgfxHeapCreate>(ring, RGFX_CMD_HEAP_CREATE).gpu_address ==
	      ownHeap.gpu_address);
}

// A chunk source standing in for the kext: it hands each first id of `firsts` in turn, with
// the grant's size from `sizes` where it holds one for it (else none, the space's own size),
// then answers none left, and counts its calls (the user-client calls a create would make).
struct FakeChunks {
	std::vector<uint64_t> firsts;
	std::vector<uint64_t> sizes;
	size_t next = 0;
	int calls = 0;
};

bool fakeChunk(void *context, uint64_t *first_id, uint64_t *ids)
{
	FakeChunks *f = static_cast<FakeChunks *>(context);
	f->calls++;
	if (f->next == f->firsts.size())
		return false;
	if (f->next < f->sizes.size())
		*ids = f->sizes[f->next];
	*first_id = f->firsts[f->next++];
	return true;
}

// The kext's grant says how many ids its chunk holds, and the space takes that many, whatever
// size it was set up for: a process whose features query a sandbox refused set chunks of 64
// up, while the kext granted 16 (Safari's WebContent on macOS 12.7.6), and named ids past its
// chunk that the device refused.
void test_resource_id_chunk_size_is_the_grant_s()
{
	const uint64_t small = RGFX_RESOURCE_ID_SMALL_CHUNK;
	const uint64_t chunk = RGFX_RESOURCE_ID_CHUNK;

	// Set up for chunks of 64; the kext grants small chunk 29 (ids 465 to 480), then small
	// chunk 3 (ids 49 to 64): 16 ids each, and the 17th id takes the next grant.
	FakeChunks f;
	f.firsts = {1 + 29 * small, 1 + 3 * small};
	f.sizes = {small, small};
	RgfxIdSpace ids;
	ids.setResourceIdSource(&fakeChunk, &f, chunk);
	for (uint64_t id = 465; id <= 480; id++)
		CHECK(ids.allocResourceId() == id);
	CHECK(f.calls == 1);
	CHECK(ids.allocResourceId() == 49);
	CHECK(f.calls == 2);
	for (uint64_t id = 50; id <= 64; id++)
		CHECK(ids.allocResourceId() == id);
	CHECK(ids.allocResourceId() == 0);
	CHECK(f.calls == 3);
	// Ids next to the grants, which a chunk of 64 would have held, are no chunk's.
	ids.freeResourceId(481);
	ids.freeResourceId(65);
	CHECK(ids.allocResourceId() == 0);

	// Set up for small chunks; the kext grants a chunk of 64 (ids 129 to 192): all 64 are the
	// process's.
	FakeChunks big;
	big.firsts = {1 + 2 * chunk};
	big.sizes = {chunk};
	RgfxIdSpace bigIds;
	bigIds.setResourceIdSource(&fakeChunk, &big, small);
	for (uint64_t id = 129; id <= 192; id++)
		CHECK(bigIds.allocResourceId() == id);
	CHECK(big.calls == 1);

	// A grant of a size that is not whole small chunks, or more than a chunk, is no chunk; one
	// whose first id is not a multiple of its own size past 1 is none either; one of 32 at a
	// multiple of 32 is two small chunks.
	FakeChunks bad;
	bad.firsts = {1 + 2 * small, 1 + 2 * chunk, 1 + small, 1 + 2 * small};
	bad.sizes = {small + 1, 2 * chunk, 2 * small, 2 * small};
	RgfxIdSpace badIds;
	badIds.setResourceIdSource(&fakeChunk, &bad, chunk);
	CHECK(badIds.allocResourceId() == 0);
	CHECK(badIds.allocResourceId() == 0);
	CHECK(badIds.allocResourceId() == 0);
	for (uint64_t id = 33; id <= 64; id++)
		CHECK(badIds.allocResourceId() == id);
	CHECK(badIds.allocResourceId() == 0);
	CHECK(bad.calls == 5);
}

// The kext's small chunks (RGFX_FEATURE_SMALL_ID_CHUNKS): 16 ids each, from the same pool as
// the chunks of 64, a chunk of 64 being four small ones, so a process takes no more ids than
// its textures need, 16 at a time.
void test_small_resource_id_chunks()
{
	const uint64_t small = RGFX_RESOURCE_ID_SMALL_CHUNK;
	CHECK(small == 16 && RGFX_RESOURCE_ID_SMALL_CHUNKS * small ==
	                         RGFX_RESOURCE_ID_CHUNKS * RGFX_RESOURCE_ID_CHUNK);

	// Small chunks 29 (ids 465 to 480), then 3 (ids 49 to 64), then none.
	FakeChunks f;
	f.firsts = {1 + 29 * small, 1 + 3 * small};
	RgfxIdSpace ids;
	ids.setResourceIdSource(&fakeChunk, &f, small);
	for (uint64_t id = 465; id <= 480; id++)
		CHECK(ids.allocResourceId() == id);
	CHECK(f.calls == 1);
	ids.freeResourceId(470);
	CHECK(ids.allocResourceId() == 470);
	CHECK(f.calls == 1);
	CHECK(ids.allocResourceId() == 49);
	CHECK(f.calls == 2);
	for (uint64_t id = 50; id <= 64; id++)
		CHECK(ids.allocResourceId() == id);
	CHECK(ids.allocResourceId() == 0);
	CHECK(f.calls == 3);
	// An id next to a held small chunk, in no chunk the process holds, is ignored.
	ids.freeResourceId(481);
	ids.freeResourceId(48);
	CHECK(ids.allocResourceId() == 0);
	ids.freeResourceId(64);
	CHECK(ids.allocResourceId() == 64);

	// A small chunk that is not one (not a small chunk's first id, past the pool), or one
	// held already, gives no id; the pool's last is one.
	FakeChunks bad;
	const uint64_t last = 1 + (RGFX_RESOURCE_ID_SMALL_CHUNKS - 1) * small;
	bad.firsts = {2 + 4 * small, last + small, last, last};
	RgfxIdSpace badIds;
	badIds.setResourceIdSource(&fakeChunk, &bad, small);
	CHECK(badIds.allocResourceId() == 0);
	CHECK(badIds.allocResourceId() == 0);
	CHECK(badIds.allocResourceId() == last);
	for (uint64_t i = 1; i < small; i++)
		CHECK(badIds.allocResourceId() == last + i);
	CHECK(badIds.allocResourceId() == 0);
	CHECK(last + small - 1 == RGFX_RESOURCE_ID_CHUNKS * RGFX_RESOURCE_ID_CHUNK);

	// Chunks of 64 are four small chunks each: a source of 64 whose first id is a small
	// chunk's but not a chunk's is no chunk, and a chunk of 64 over a small chunk held
	// already is none.
	FakeChunks smallFirst;
	smallFirst.firsts = {1 + 5 * small};
	FakeChunks mixed;
	mixed.firsts = {1 + 5 * small, 1 + 64, 1 + 2 * 64};
	RgfxIdSpace mixedIds;
	mixedIds.setResourceIdSource(&fakeChunk, &smallFirst, small);
	for (uint64_t id = 81; id <= 96; id++)
		CHECK(mixedIds.allocResourceId() == id);
	mixedIds.setResourceIdSource(&fakeChunk, &mixed);
	CHECK(mixedIds.allocResourceId() == 0);
	CHECK(mixedIds.allocResourceId() == 0);
	CHECK(mixedIds.allocResourceId() == 129);
	for (uint64_t id = 130; id <= 192; id++)
		CHECK(mixedIds.allocResourceId() == id);
	CHECK(mixed.calls == 3);

	// With no source, chunks of 16 hand out the whole pool in order, as chunks of 64 do.
	RgfxIdSpace local;
	local.setResourceIdSource(nullptr, nullptr, small);
	const uint64_t poolIds = RGFX_RESOURCE_ID_SMALL_CHUNKS * small;
	for (uint64_t id = 1; id <= poolIds; id++)
		CHECK(local.allocResourceId() == id);
	CHECK(local.allocResourceId() == 0);
}

void test_resource_id_chunks()
{
	const uint64_t chunk = RGFX_RESOURCE_ID_CHUNK;

	// With no source, the ids are the whole pool's, in order, to its last whole chunk below
	// the limit; then none, until one is given back.
	{
		RgfxIdSpace ids;
		const uint64_t last = RGFX_RESOURCE_ID_CHUNKS * chunk;
		CHECK(last < RGFX_RESOURCE_ID_LIMIT);
		for (uint64_t id = 1; id <= last; id++)
			CHECK(ids.allocResourceId() == id);
		CHECK(ids.allocResourceId() == 0);
		ids.freeResourceId(100);
		CHECK(ids.allocResourceId() == 100);
		CHECK(ids.allocResourceId() == 0);
	}

	// From the kext's chunks: 7 (ids 449 to 512), then 3 (ids 193 to 256), then none.
	FakeChunks f;
	f.firsts = {1 + 7 * chunk, 1 + 3 * chunk};
	RgfxIdSpace ids;
	ids.setResourceIdSource(&fakeChunk, &f);

	// Take: the first id takes a chunk; its other 63 take no other.
	CHECK(ids.allocResourceId() == 449);
	CHECK(f.calls == 1);
	for (uint64_t id = 450; id <= 512; id++)
		CHECK(ids.allocResourceId() == id);
	CHECK(f.calls == 1);

	// Free and reuse: the lowest id given back comes first, and no chunk is taken.
	ids.freeResourceId(460);
	ids.freeResourceId(455);
	CHECK(ids.allocResourceId() == 455);
	CHECK(ids.allocResourceId() == 460);
	CHECK(f.calls == 1);

	// Refill: with the chunk full, the next id is a new chunk's first.
	CHECK(ids.allocResourceId() == 193);
	CHECK(f.calls == 2);
	// The chunks are tried in the order taken, so an id given back to the first comes before
	// the second's next.
	ids.freeResourceId(500);
	CHECK(ids.allocResourceId() == 500);
	CHECK(ids.allocResourceId() == 194);

	// 0, an id of a chunk the process does not hold, and ids past the pool are ignored.
	ids.freeResourceId(0);
	ids.freeResourceId(1);
	ids.freeResourceId(RGFX_RESOURCE_ID_LIMIT - 1);
	ids.freeResourceId(RGFX_RESOURCE_ID_LIMIT);
	ids.freeResourceId(~0ull);

	// Exhaustion: once both chunks are full and the source has none, no id, each time asked.
	for (uint64_t id = 195; id <= 256; id++)
		CHECK(ids.allocResourceId() == id);
	CHECK(ids.allocResourceId() == 0);
	CHECK(f.calls == 3);
	CHECK(ids.allocResourceId() == 0);
	CHECK(f.calls == 4);
	// An id given back is taken again with no call.
	ids.freeResourceId(200);
	CHECK(ids.allocResourceId() == 200);
	CHECK(f.calls == 4);

	// A chunk that is not one of the pool's (not a chunk's first id, past the pool, 0), or
	// one held already, gives no id.
	FakeChunks bad;
	bad.firsts = {2, 1 + RGFX_RESOURCE_ID_CHUNKS * chunk, 0, 1 + 5 * chunk, 1 + 5 * chunk};
	RgfxIdSpace badIds;
	badIds.setResourceIdSource(&fakeChunk, &bad);
	CHECK(badIds.allocResourceId() == 0);
	CHECK(badIds.allocResourceId() == 0);
	CHECK(badIds.allocResourceId() == 0);
	CHECK(badIds.allocResourceId() == 1 + 5 * chunk);
	for (uint64_t i = 1; i < chunk; i++)
		CHECK(badIds.allocResourceId() == 1 + 5 * chunk + i);
	CHECK(badIds.allocResourceId() == 0);

	// A texture create with no id left posts nothing and answers no object; one given back
	// lets the next create go out with it.
	TestRing ring(0x8000);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	FakeChunks one;
	one.firsts = {1 + 9 * chunk};
	RgfxIdSpace serIds;
	serIds.setResourceIdSource(&fakeChunk, &one);
	RgfxSerializer ser(writer, serIds);
	RgfxTextureDescriptor desc = {};
	desc.pixel_format = 80;
	desc.texture_type = 2;
	desc.width = desc.height = desc.depth = 1;
	desc.mipmap_level_count = desc.array_length = desc.sample_count = 1;
	for (uint64_t i = 0; i < chunk; i++) {
		RgfxCreated t = ser.createTexture(desc);
		CHECK(t.posted && t.resource_id == 1 + 9 * chunk + i);
		auto tm = expectMessage<RgfxTextureCreate>(ring, RGFX_CMD_TEXTURE_CREATE);
		CHECK(tm.resource_id == t.resource_id);
	}
	RgfxCreated none = ser.createTexture(desc);
	CHECK(!none.posted && none.obj_id == 0 && none.resource_id == 0 && none.gpu_address == 0);
	RgfxCreated noneFromBuffer = ser.createTextureFromBuffer(1, 0, 4, desc);
	CHECK(!noneFromBuffer.posted && noneFromBuffer.resource_id == 0);
	RgfxCreated noneFromSurface = ser.createTextureFromIOSurface(1, 0, 4, 0x55, 0, desc, 0, RgfxSurfaceFacts{});
	CHECK(!noneFromSurface.posted && noneFromSurface.resource_id == 0);
	RgfxTextureView noView = {};
	noView.parent_obj_id = 1;
	RgfxCreated noneView = ser.createTextureView(noView);
	CHECK(!noneView.posted && noneView.resource_id == 0 && noneView.obj_id == 0);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(!ring.next(&h, &bytes));
	ser.releaseResourceId(1 + 9 * chunk + 3);
	RgfxCreated again = ser.createTexture(desc);
	CHECK(again.posted && again.resource_id == 1 + 9 * chunk + 3);
	auto am = expectMessage<RgfxTextureCreate>(ring, RGFX_CMD_TEXTURE_CREATE);
	CHECK(am.resource_id == again.resource_id && am.obj_id == again.obj_id);
}

// The negotiation's records as the bundle compiles them (wire.rs asserts the same).
static_assert(RGFX_REG_DRIVER_FEATURES_LO == 0x024 && RGFX_REG_DRIVER_FEATURES_HI == 0x028,
              "DRIVER_FEATURES");
static_assert(offsetof(RgfxRingHeader, producer_waiting) == 0x10, "producer_waiting");
static_assert(offsetof(RgfxRingHeader, raise_all) == 0xc4, "raise_all");
static_assert(RGFX_QUEUE_CREATE_FLAGS_OFFSET == sizeof(RgfxQueueCreate) &&
                  sizeof(RgfxQueueCreateFlags) == 8,
              "QUEUE_CREATE's tail");
static_assert(sizeof(RgfxOpHeader) + sizeof(RgfxOpQueueBarrier) == 24, "QUEUE_BARRIER");
static_assert(RGFX_SAMPLER_CREATE_EXT_OFFSET == sizeof(RgfxSamplerCreate) &&
                  sizeof(RgfxSamplerCreateExt) == 8,
              "SAMPLER_CREATE's tail");
static_assert(sizeof(RgfxCommandBufferPart) == 56, "COMMAND_BUFFER_PART");
static_assert(sizeof(RgfxBufferOverPages) == 72, "BUFFER_OVER_PAGES");
static_assert(offsetof(RgfxDeviceInfo, features_lo) == 0x451 &&
                  offsetof(RgfxDeviceInfo, sampler_reduction_support) == 0x452 &&
                  offsetof(RgfxDeviceInfo, integrated) == 0x453,
              "the device info's feature, reduction and integrated bytes");

// The features call's answer: taken where the kext gave its three words and its selector map is
// this bundle's, as they are, bits from 3 and 32 among them; refused, with both words 0, where it
// answered kIOReturnUnsupported (a kext of the map before, whose features call had another
// number), failed otherwise, gave fewer words, or named another map. The device info's offered
// bits stand in for none of them.
void test_caps_features()
{
	RgfxCaps caps;
	CHECK(caps.features() == 0 && caps.kextFeatures() == 0); // nothing before the call
	const uint64_t word = RGFX_FEATURE_CB_SUBMITTED | RGFX_FEATURE_CB_PARTS | (1ull << 40);
	const uint64_t ours[3] = {word, 1ull << 4, kRgfxSelectorMap};
	CHECK(caps.takeFeatures(KERN_SUCCESS, 3, ours));
	CHECK(caps.features() == word && caps.kextFeatures() == 1ull << 4);

	CHECK(!caps.takeFeatures(kIOReturnUnsupported, 3, ours));
	CHECK(caps.features() == 0 && caps.kextFeatures() == 0);
	CHECK(caps.takeFeatures(KERN_SUCCESS, 3, ours));
	CHECK(!caps.takeFeatures(kIOReturnBadArgument, 3, ours));
	CHECK(caps.features() == 0 && caps.kextFeatures() == 0);
	CHECK(!caps.takeFeatures(KERN_SUCCESS, 2, ours));
	const uint64_t before[3] = {word, 1ull << 4, kRgfxSelectorMap - 1};
	const uint64_t after[3] = {word, 1ull << 4, kRgfxSelectorMap + 1};
	CHECK(!caps.takeFeatures(KERN_SUCCESS, 3, before));
	CHECK(!caps.takeFeatures(KERN_SUCCESS, 3, after));
	CHECK(caps.features() == 0 && caps.kextFeatures() == 0);

	RgfxDeviceInfo info = {};
	info.version = RGFX_DEVICE_INFO_VERSION;
	info.features_lo = RGFX_FEATURE_SURFACE_PAGES | RGFX_FEATURE_CB_SUBMITTED |
	                   RGFX_FEATURE_QUIET_SCHEDULED;
	CHECK(caps.load(info));
	CHECK(!caps.takeFeatures(kIOReturnUnsupported, 0, ours));
	CHECK(caps.features() == 0 && caps.kextFeatures() == 0);
	const uint64_t none[3] = {0, 0, kRgfxSelectorMap};
	CHECK(caps.takeFeatures(KERN_SUCCESS, 3, none));
	CHECK(caps.features() == 0 && caps.kextFeatures() == 0);
}

// The feature profile the device answers, on every host: the x86 Macs' that the guest's Metal
// knows, 10002 where it has the MTLArchitecture class (macOS 13 and later), else 10001, the one
// macOS 11 and 12's Metal has limits for.
void test_feature_profile_for_guest()
{
	static_assert(rgfx_profile::kX86Mac == 10002);
	static_assert(rgfx_profile::kX86Mac12 == 10001);
	CHECK(rgfx_profile::forGuest(true) == 10002);  // macOS 13 to 26
	CHECK(rgfx_profile::forGuest(false) == 10001); // macOS 11 and 12
}

void test_caps()
{
	RgfxCaps caps;
	// A bad version is refused and leaves the caps invalid.
	RgfxDeviceInfo bad = {};
	bad.version = RGFX_DEVICE_INFO_VERSION + 1;
	CHECK(!caps.load(bad));
	CHECK(!caps.valid());

	RgfxDeviceInfo info = {};
	info.version = RGFX_DEVICE_INFO_VERSION;
	info.registry_id = 0xabcd;
	info.gpu_core_count = 16;
	info.max_threads_per_threadgroup[0] = 1024;
	info.max_threads_per_threadgroup[1] = 512;
	info.max_threads_per_threadgroup[2] = 64;
	info.max_vertex_amplification_count = 2;
	// The families and feature sets are looked up by value, not by position.
	info.family_count = 3;
	info.family_values[0] = 1005; // Apple5
	info.family_answers[0] = 1;
	info.family_values[1] = 1006; // Apple6, a present no
	info.family_values[2] = -1009;
	info.family_answers[2] = 1;
	info.family_values[3] = 1007; // past family_count: not read
	info.family_answers[3] = 1;
	info.feature_set_count = 2;
	info.feature_set_values[0] = 10005; // macOS_GPUFamily2_v1
	info.feature_set_answers[0] = 1;
	info.feature_set_values[1] = 10002; // macOS_ReadWriteTextureTier2, a present no
	info.sizes[kRgfxSizeRecommendedMaxWorkingSetSize] = 1ull << 34; // first by position
	info.sizes[kRgfxSizeLargestRasterizationRateMapLayerCount] = 2; // last by position
	info.identity_flags[kRgfxIdentityIsLowPower] = 1;              // first by position
	info.identity_flags[kRgfxIdentitySupportsPlacementSparse] = 1; // last by position
	info.counter_sampling[0] = 1; // at stage boundaries
	info.format_count = 3;
	info.formats[0].format = 80; // BGRA8Unorm
	info.formats[0].texture_buffer_alignment = 16;
	info.formats[0].linear_alignment = 256;
	info.formats[0].sparse_tiles[0][0] = 64; // 1 sample, 16 KiB pages
	info.formats[0].sparse_tiles[0][1] = 64;
	info.formats[0].sparse_tiles[0][2] = 1;
	info.formats[0].sparse_tiles[8][0] = 128; // 4 samples, 256 KiB pages
	info.formats[0].sparse_tiles[8][1] = 128;
	info.formats[0].sparse_tiles[8][2] = 1;
	// A host GPU whose linear images ask more of larger texels: a 16-byte texel's 1024 bytes
	// beside a 4-byte texel's 256, the largest the device's linearTextureAlignmentBytes.
	info.formats[1].format = 125; // RGBA32Float
	info.formats[1].linear_alignment = 1024;
	info.formats[2].format = 70; // RGBA8Unorm
	info.formats[2].linear_alignment = 256;
	info.formats[3].format = 115; // RGBA16Float, past format_count: not read
	info.formats[3].texture_buffer_alignment = 16;
	info.formats[3].linear_alignment = 512;
	const char *arch = "test_arch";
	info.architecture_name_len = static_cast<uint32_t>(std::strlen(arch));
	std::memcpy(info.architecture_name, arch, info.architecture_name_len);
	info.sample_counts[0] = 1;               // count 1
	info.sample_counts[2] = 1;               // count 4
	info.primitive_types[3] = 1;
	info.write_rounding_modes[1] = 1; // mode 2
	info.limits[kRgfxLimitDeviceLinearTextureAlignmentBytes] = 64;
	info.limits[kRgfxLimitLinearTextureAlignmentBytes] = 1024;
	info.limits[kRgfxLimitIOSurfaceReadOnlyTextureAlignmentBytes] = 32;
	info.limits[kRgfxLimitIOSurfaceTextureAlignmentBytes] = 128;
	info.limits[kRgfxLimitMaxViewportCount] = 16; // last by position
	info.flags[kRgfxFlagAreRasterOrderGroupsSupported] = 1; // first by position
	info.flags[kRgfxFlagSupportsTileShaders] = 1;           // last by position
	// Longer than the 32 bytes the first layout held.
	const char *nm = "NVIDIA GeForce RTX 5080 Laptop GPU";
	info.name_len = static_cast<uint32_t>(std::strlen(nm));
	std::memcpy(info.name, nm, info.name_len);

	CHECK(caps.load(info));
	CHECK(caps.valid());
	CHECK(caps.registryId() == 0xabcd);
	CHECK(caps.gpuCoreCount() == 16);
	CHECK(caps.maxThreadsPerThreadgroup(0) == 1024);
	CHECK(caps.maxThreadsPerThreadgroup(2) == 64);
	CHECK(caps.maxVertexAmplificationCount() == 2);
	CHECK(caps.supportsFamily(1005));
	CHECK(caps.supportsFamily(-1009));
	CHECK(!caps.supportsFamily(1006));
	CHECK(!caps.supportsFamily(1007)); // listed past the count
	CHECK(!caps.supportsFamily(42));   // names no family
	CHECK(caps.supportsFeatureSet(10005));
	CHECK(!caps.supportsFeatureSet(10002));
	CHECK(!caps.supportsFeatureSet(0)); // not listed
	CHECK(caps.size(kRgfxSizeRecommendedMaxWorkingSetSize) == 1ull << 34);
	CHECK(caps.size(kRgfxSizeLargestRasterizationRateMapLayerCount) == 2);
	CHECK(caps.size(kRgfxSizeMaxBufferLength) == 0); // a gap reads 0
	CHECK(caps.identityFlag(kRgfxIdentityIsLowPower));
	CHECK(caps.identityFlag(kRgfxIdentitySupportsPlacementSparse));
	CHECK(!caps.identityFlag(kRgfxIdentityHasUnifiedMemory));
	CHECK(caps.supportsCounterSampling(0));
	CHECK(!caps.supportsCounterSampling(1));
	CHECK(!caps.supportsCounterSampling(5));
	CHECK(std::strcmp(caps.architectureName(), "test_arch") == 0);
	CHECK(caps.textureBufferAlignment(80) == 16);
	CHECK(caps.textureBufferAlignment(115) == 0); // listed past the count
	CHECK(caps.textureBufferAlignment(130) == 0); // not listed
	// The linear alignment is each format's own where the blob answers it, and the device's
	// largest, never 0, where it does not: a format listed without one, listed past the count,
	// or not listed, where Metal ends the process.
	CHECK(caps.minimumLinearTextureAlignment(80) == 256);
	CHECK(caps.minimumLinearTextureAlignment(70) == 256);
	CHECK(caps.minimumLinearTextureAlignment(125) == 1024);
	CHECK(caps.minimumLinearTextureAlignment(115) == 1024); // listed past the count
	CHECK(caps.minimumLinearTextureAlignment(130) == 1024); // not listed (ASTC)
	CHECK(caps.minimumLinearTextureAlignment(0) == 1024);   // Invalid
	uint64_t tile[3];
	CHECK(caps.sparseTile(80, 1, 101, tile) && tile[0] == 64 && tile[1] == 64 && tile[2] == 1);
	CHECK(caps.sparseTile(80, 4, 103, tile) && tile[0] == 128 && tile[2] == 1);
	CHECK(!caps.sparseTile(80, 2, 102, tile) && tile[0] == 0); // a zero tile is no answer
	CHECK(!caps.sparseTile(80, 8, 101, tile));                  // no such sample count
	CHECK(!caps.sparseTile(80, 1, 104, tile));                  // no such page
	CHECK(!caps.sparseTile(130, 1, 101, tile));                 // not listed
	CHECK(caps.supportsSampleCount(1));
	CHECK(caps.supportsSampleCount(4));
	CHECK(!caps.supportsSampleCount(2));
	CHECK(!caps.supportsSampleCount(8));
	CHECK(!caps.supportsSampleCount(3)); // not a power the blob holds
	CHECK(caps.supportsPrimitiveType(3));
	CHECK(!caps.supportsPrimitiveType(0));
	CHECK(caps.supportsWriteRoundingMode(2));
	CHECK(!caps.supportsWriteRoundingMode(1));
	CHECK(caps.limit(kRgfxLimitDeviceLinearTextureAlignmentBytes) == 64);
	// The IOSurface alignments the device answers, at their places between the linear ones.
	CHECK(caps.limit(kRgfxLimitIOSurfaceReadOnlyTextureAlignmentBytes) == 32);
	CHECK(caps.limit(kRgfxLimitIOSurfaceTextureAlignmentBytes) == 128);
	CHECK(caps.limit(kRgfxLimitLinearTextureAlignmentBytes) == 1024);
	CHECK(caps.limit(kRgfxLimitMaxViewportCount) == 16);
	CHECK(caps.flag(kRgfxFlagAreRasterOrderGroupsSupported));
	CHECK(caps.flag(kRgfxFlagSupportsTileShaders));
	CHECK(!caps.flag(kRgfxFlagSupportsImageBlocks));
	CHECK(std::strcmp(caps.name(), "NVIDIA GeForce RTX 5080 Laptop GPU") == 0);
}

// The device answers Metal's 16 for both IOSurface alignments whatever the host answers
// (kRgfxIOSurfaceTextureAlignmentBytes): CoreAnimation makes no texture of a surface whose rows are
// not at a multiple of the answer, and draws nothing of its layer. The rows are those WindowServer
// made no texture of in macOS 26.7.2 guests answering an RTX 3090's and an RTX 2080's 128, where
// Safari drew its pages blank, and IOSurface's own alignment, 64 bytes.
void test_iosurface_alignments_divide_other_codes_surface_rows()
{
	CHECK(kRgfxIOSurfaceTextureAlignmentBytes == 16);
	const uint64_t rows[] = {64, 192, 448, 576, 704, 1088, 1216, 4160, 5184, 5312, 14272};
	for (uint64_t r : rows)
		CHECK(r % kRgfxIOSurfaceTextureAlignmentBytes == 0);
	// The host's answers stay in the caps, and only there.
	RgfxDeviceInfo info = {};
	info.version = RGFX_DEVICE_INFO_VERSION;
	info.limits[kRgfxLimitIOSurfaceReadOnlyTextureAlignmentBytes] = 128;
	info.limits[kRgfxLimitIOSurfaceTextureAlignmentBytes] = 256;
	RgfxCaps caps;
	CHECK(caps.load(info));
	CHECK(caps.limit(kRgfxLimitIOSurfaceReadOnlyTextureAlignmentBytes) == 128);
	CHECK(caps.limit(kRgfxLimitIOSurfaceTextureAlignmentBytes) == 256);
}

// The MTLDeviceSPI feature queries the device answers itself (RgfxCaps::spiFeature): each YES on
// a host of the Mac2 family whose flags and read-write tier say yes, as Metal's devices answer,
// and each NO on a host of no Mac2 family with no flags and no tier, and with no caps at all.
void test_spi_features()
{
	RgfxCaps none;
	for (int k = 0; k < kRgfxSpiCount; k++)
		CHECK(!none.spiFeature(static_cast<RgfxSpiFeature>(k)));

	RgfxDeviceInfo info = {};
	info.version = RGFX_DEVICE_INFO_VERSION;
	info.family_count = 2;
	info.family_values[0] = 2002; // Mac2
	info.family_answers[0] = 1;
	info.family_values[1] = 5001; // Metal3
	info.family_answers[1] = 1;
	info.flags[kRgfxFlagSupportsDynamicAttributeStride] = 1;
	info.flags[kRgfxFlagAreRasterOrderGroupsSupported] = 1;
	info.flags[kRgfxFlagSupportsSimdReduction] = 1;
	info.flags[kRgfxFlagSupportsTexture2DMultisampleArray] = 1;
	info.sizes[kRgfxSizeReadWriteTextureSupport] = 2;
	RgfxCaps mac;
	CHECK(mac.load(info));
	for (int k = 0; k < kRgfxSpiCount; k++)
		CHECK(mac.spiFeature(static_cast<RgfxSpiFeature>(k)));
	CHECK(!mac.spiFeature(kRgfxSpiCount));

	// The Mac2 family alone: every query but the four flags' and the tier's.
	RgfxDeviceInfo family = info;
	std::memset(family.flags, 0, sizeof family.flags);
	family.sizes[kRgfxSizeReadWriteTextureSupport] = 0;
	RgfxCaps fam;
	CHECK(fam.load(family));
	CHECK(fam.spiFeature(kRgfxSpiTessellation) && fam.spiFeature(kRgfxSpiSIMDGroup));
	CHECK(!fam.spiFeature(kRgfxSpiDynamicAttributeStride));
	CHECK(!fam.spiFeature(kRgfxSpiRasterOrderGroups));
	CHECK(!fam.spiFeature(kRgfxSpiSIMDReduction));
	CHECK(!fam.spiFeature(kRgfxSpiTexture2DMultisampleArray));
	CHECK(!fam.spiFeature(kRgfxSpiReadWriteTextureArguments));

	// The flags and the tier alone: only theirs.
	RgfxDeviceInfo flagsOnly = info;
	flagsOnly.family_answers[0] = 0;
	RgfxCaps fl;
	CHECK(fl.load(flagsOnly));
	CHECK(!fl.spiFeature(kRgfxSpiTessellation) && !fl.spiFeature(kRgfxSpiArgumentBuffers));
	CHECK(!fl.spiFeature(kRgfxSpiInt64));
	CHECK(fl.spiFeature(kRgfxSpiDynamicAttributeStride) && fl.spiFeature(kRgfxSpiSIMDReduction));
	CHECK(fl.spiFeature(kRgfxSpiReadWriteTextureArguments));
	int yes = 0;
	for (int k = 0; k < kRgfxSpiCount; k++)
		yes += fl.spiFeature(static_cast<RgfxSpiFeature>(k)) ? 1 : 0;
	CHECK(yes == 5);
}

// The SPI family name, from the device name, by the rule Metal's devices answer by: the three
// Macs' names give their own answers (measured), and the hosts' names the lane's.
void test_family_names()
{
	const std::pair<const char *, const char *> cases[] = {
	    {"Apple M3 Max", "M3 Max"},
	    {"AMD Radeon Pro 5300M", "Radeon Pro"},
	    {"Intel(R) UHD Graphics 630", "UHD Graphics"},
	    {"NVIDIA GeForce RTX 5080 Laptop GPU", "GeForce RTX"},
	    {"AMD Radeon RX 7900 XTX", "Radeon RX"},
	    {"reims-graphics", "reims-graphics"}, // the name with no caps: no vendor, no number
	    {"NVIDIA", "NVIDIA"},                 // nothing left: the whole name
	};
	for (const auto &[name, family] : cases)
		CHECK(rgfxFamilyNameOf(name) == family);
}

// The SPI vendor name, from the device name's first word: the two x86 Macs' names give their
// own answers (measured), the hosts' names their vendors' in those words, and a name of no vendor
// the GL renderer knows the AMD's.
void test_vendor_names()
{
	const std::pair<const char *, const char *> cases[] = {
	    {"AMD Radeon Pro 5300M", "AMD"},
	    {"Intel(R) UHD Graphics 630", "Intel(R)"},
	    {"NVIDIA GeForce RTX 5080 Laptop GPU", "NVIDIA"},
	    {"AMD Radeon 780M (RADV GFX1103_R1)", "AMD"},
	    {"Intel(R) Arc(tm) Graphics (MTL)", "Intel(R)"},
	    {"Intel Arc A770", "Intel(R)"},
	    {"Apple M3 Max", "AMD"},
	    {"llvmpipe (LLVM 20.1.8, 256 bits)", "AMD"},
	    {"reims-graphics", "AMD"},
	    {"", "AMD"},
	    {"NVIDIAX", "AMD"}, // a first word that only begins with a vendor's
	};
	for (const auto &[name, vendor] : cases)
		CHECK(rgfxVendorNameOf(name) == vendor);
}

// The x86-64 guest's Metal name: the host's name whole after "Intel - " where the host's GPU is
// an integrated one and "AMD - " for every other type, whatever the name holds (the owner's
// examples first); the host's spaces kept; "reims-graphics" after the prefix for no name.
void test_metal_names()
{
	const struct {
		const char *host;
		bool integrated;
		const char *metal;
	} cases[] = {
	    {"NVIDIA GeForce RTX 3090", false, "AMD - NVIDIA GeForce RTX 3090"},
	    {"AMD Radeon 780M Graphics (RADV PHOENIX)", true,
	     "Intel - AMD Radeon 780M Graphics (RADV PHOENIX)"},
	    {"llvmpipe (LLVM 20.1.8, 256 bits)", false, "AMD - llvmpipe (LLVM 20.1.8, 256 bits)"},
	    {"Intel(R) Iris(R) Xe Graphics (RPL-P)", true, "Intel - Intel(R) Iris(R) Xe Graphics (RPL-P)"},
	    {"Intel(R) Arc(tm) A770 Graphics (DG2)", false, "AMD - Intel(R) Arc(tm) A770 Graphics (DG2)"},
	    {"AMD Radeon RX 7900 XTX (RADV NAVI31)", false, "AMD - AMD Radeon RX 7900 XTX (RADV NAVI31)"},
	    {"", false, "AMD - reims-graphics"},
	    {"", true, "Intel - reims-graphics"},
	    {" GPU ", false, "AMD -  GPU "},
	    {"  ", true, "Intel -   "},
	};
	for (const auto &c : cases)
		CHECK(rgfxMetalNameOf(c.host, c.integrated) == c.metal);

	// The blob's byte: yes where the host wrote 1, no where it wrote 0 (an older host's), and no
	// with no caps.
	RgfxCaps none;
	CHECK(!none.hostIntegrated());
	CHECK(rgfxMetalNameOf(none.name(), none.hostIntegrated()) == "AMD - reims-graphics");
	RgfxDeviceInfo info = {};
	info.version = RGFX_DEVICE_INFO_VERSION;
	RgfxCaps older;
	CHECK(older.load(info) && !older.hostIntegrated());
	info.integrated = 1;
	RgfxCaps integrated;
	CHECK(integrated.load(info) && integrated.hostIntegrated());
	info.version = RGFX_DEVICE_INFO_VERSION + 1;
	RgfxCaps refused;
	CHECK(!refused.load(info) && !refused.hostIntegrated());
}

// maximumConcurrentCompilationTaskCount is the logical processor count of the system the bundle
// runs on, its kernel's hw.logicalcpu, read here beside it; never the host's compile threads,
// which the blob still carries.
void test_concurrent_compilation_task_count()
{
	int cpus = 0;
	size_t length = sizeof(cpus);
	CHECK(sysctlbyname("hw.logicalcpu", &cpus, &length, nullptr, 0) == 0 && cpus >= 1);
	CHECK(rgfxConcurrentCompilationTaskCount() == static_cast<uint64_t>(cpus));
}

// Heap and buffer creates carry the three modes and Metal's bit 24 to the host, and no other
// bit: the GL renderer's no-copy buffers' bit 20 (0x100000) among them.
void test_host_resource_options()
{
	CHECK(RgfxHostResourceOptions(0x100010u) == 0x10u);
	CHECK(RgfxHostResourceOptions(0x1000321u) == 0x1000321u);
	CHECK(RgfxHostResourceOptions(0xfffffcffu) == 0x10000ffu);
	TestRing ring(0x8000);
	ring.setConsumerIdle(1);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	CHECK(ser.createBuffer(0x1000, 0x100010).posted);
	CHECK(expectMessage<RgfxBufferCreate>(ring, RGFX_CMD_BUFFER_CREATE).options == 0x10);
	CHECK(ser.createBufferOverPages(0x1000, 0x100000, 0x60000, 0x1000, 0x40).posted);
	CHECK(expectMessage<RgfxBufferOverPages>(ring, RGFX_CMD_BUFFER_OVER_PAGES).options == 0);
	CHECK(ser.createHeap(0x10000, 0x1100020, 1).posted);
	CHECK(expectMessage<RgfxHeapCreate>(ring, RGFX_CMD_HEAP_CREATE).options == 0x1000020);
}

// A buffer over a pages block at the address the host gave the block's view
// (createBufferOverPagesAt, RGFX_FEATURE_IMPORT_ADDRESSES) picks no address: its message names
// that address plus the pointer's offset, and the answer the address, which no pick of the
// process's lands on; an address of 0, or a buffer whose end would wrap, is no object.
void test_buffer_over_pages_at_the_host_s_address()
{
	TestRing ring(0x8000);
	ring.setConsumerIdle(1);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	const uint64_t view = 1ull << 43;
	RgfxCreated at = ser.createBufferOverPagesAt(0x3000, 0, 0x60000, 0x1000, 0x123, view);
	CHECK(at.posted && at.obj_id != 0 && at.gpu_address == view);
	const RgfxBufferOverPages m =
	    expectMessage<RgfxBufferOverPages>(ring, RGFX_CMD_BUFFER_OVER_PAGES);
	CHECK(m.obj_id == at.obj_id && m.gpu_address == view + 0x123 && m.offset == 0x123 &&
	      m.length == 0x3000 && m.block_offset == 0x60000 && m.block_size == 0x1000);
	// A pick of the process's lies in its own range, not at the host's address.
	RgfxCreated picked = ser.createBufferOverPages(0x3000, 0, 0x60000, 0x1000, 0x123);
	CHECK(picked.posted && picked.gpu_address != 0 && picked.gpu_address != view);
	CHECK(expectMessage<RgfxBufferOverPages>(ring, RGFX_CMD_BUFFER_OVER_PAGES).gpu_address ==
	      picked.gpu_address + 0x123);
	RgfxCreated none = ser.createBufferOverPagesAt(0x3000, 0, 0x60000, 0x1000, 0x123, 0);
	RgfxCreated wraps =
	    ser.createBufferOverPagesAt(0x3000, 0, 0x60000, 0x1000, 0x123, UINT64_MAX - 0x2000);
	CHECK(none.obj_id == 0 && !none.posted && wraps.obj_id == 0 && !wraps.posted);
}

// The host's answers to PAGES_BUFFER_ADDRESS: `status` and `at`, or no answer where `answers` is
// false, or one `short_by` bytes short; the messages asked, in order.
struct PagesAddressAsker {
	uint32_t status = RGFX_STATUS_OK;
	uint64_t at = 0;
	bool answers = true;
	uint32_t short_by = 0;
	uint32_t type = RGFX_EVT_COMPLETE;
	std::vector<RgfxPagesBufferAddress> asked;
	static bool ask(void *context, const void *message, uint32_t size, void *reply,
	                uint32_t capacity, uint32_t *reply_size)
	{
		auto *self = static_cast<PagesAddressAsker *>(context);
		RgfxPagesBufferAddress m = {};
		CHECK(size == sizeof(m));
		std::memcpy(&m, message, sizeof(m));
		self->asked.push_back(m);
		if (!self->answers)
			return false;
		RgfxPagesAnswer answer = {};
		answer.header = {self->type, sizeof(answer), 9};
		answer.status = self->status;
		answer.gpu_address = self->at;
		CHECK(capacity >= sizeof(answer));
		std::memcpy(reply, &answer, sizeof(answer));
		*reply_size = sizeof(answer) - self->short_by;
		return true;
	}
};

// A buffer over a pages block the host answered before its view had an address
// (createBufferOverPagesLater, RGFX_FEATURE_IMPORT_ADDRESSES_LATER) picks no address: its message
// names 0 and the pointer's offset, for the host to place it, and the answer's address is 0; one
// whose end would wrap is no object. Its address is asked of the host (pagesBufferAddress,
// PAGES_BUFFER_ADDRESS naming the buffer): the answer's where it is OK, and 0 where there is no
// asker, no answer, a short one, another event, or a status but OK.
void test_buffer_over_pages_at_the_host_s_address_later()
{
	TestRing ring(0x8000);
	ring.setConsumerIdle(1);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	CHECK(!ser.pagesLater());
	ser.setPagesLater(true);
	CHECK(ser.pagesLater());
	RgfxCreated later = ser.createBufferOverPagesLater(0x3000, 0, 0x60000, 0x1000, 0x123);
	CHECK(later.posted && later.obj_id != 0 && later.gpu_address == 0);
	const RgfxBufferOverPages m =
	    expectMessage<RgfxBufferOverPages>(ring, RGFX_CMD_BUFFER_OVER_PAGES);
	CHECK(m.obj_id == later.obj_id && m.gpu_address == 0 && m.offset == 0x123 &&
	      m.length == 0x3000 && m.block_offset == 0x60000 && m.block_size == 0x1000);
	RgfxCreated wraps = ser.createBufferOverPagesLater(UINT64_MAX - 0x100, 0, 0x60000, 0x1000,
	                                                   0x123);
	CHECK(wraps.obj_id == 0 && !wraps.posted);

	CHECK(ser.pagesBufferAddress(later.obj_id) == 0);
	PagesAddressAsker asker;
	ser.setAsker(&PagesAddressAsker::ask, &asker);
	const uint64_t view = 1ull << 43;
	asker.at = view + 0x123;
	CHECK(ser.pagesBufferAddress(later.obj_id) == view + 0x123 && asker.asked.size() == 1);
	const RgfxPagesBufferAddress &asked = asker.asked[0];
	CHECK(asked.header.type == RGFX_CMD_PAGES_BUFFER_ADDRESS &&
	      asked.header.size == sizeof(RgfxPagesBufferAddress) && asked.obj_id == later.obj_id);
	asker.status = RGFX_STATUS_NOT_FOUND;
	CHECK(ser.pagesBufferAddress(later.obj_id) == 0);
	asker.status = RGFX_STATUS_REFUSED;
	CHECK(ser.pagesBufferAddress(later.obj_id) == 0);
	asker.status = RGFX_STATUS_OK;
	asker.short_by = 8;
	CHECK(ser.pagesBufferAddress(later.obj_id) == 0);
	asker.short_by = 0;
	asker.type = RGFX_EVT_SIZE_AND_ALIGN;
	CHECK(ser.pagesBufferAddress(later.obj_id) == 0);
	asker.type = RGFX_EVT_COMPLETE;
	asker.answers = false;
	CHECK(ser.pagesBufferAddress(later.obj_id) == 0 && asker.asked.size() == 6);
	// Nothing went on the ring.
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(!ring.next(&h, &bytes));
}

// Where the host makes Managed buffers at their copies (MANAGED_AT_COPY), a Managed buffer's
// create posts its CPU copy's address, with no pick of the process's: the answer's address less
// the offset its caller gives, so the buffer's is still the answer's plus the offset, and it
// shares the copy's address, so a refusal gives nothing of it back. No copy address, or one below
// the offset, is no object.
void test_managed_buffers_at_their_copies()
{
	TestRing ring(0x8000);
	ring.setConsumerIdle(1);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	FakePrivate f;
	// Room for the copy's pick alone.
	RgfxIdSpace ids(RGFX_CONTEXT_VA_BASE, RgfxIdSpace::addressSpan(0x3123));
	ids.setPrivateBlockSource(&fakePrivateTake, &fakePrivateGive, &f);
	RgfxSerializer ser(writer, ids);
	CHECK(!ser.managedAtCopy());
	ser.setManagedAtCopy(true);
	CHECK(ser.managedAtCopy());

	// Over a no-copy buffer's pages, 0x123 into its page: at the copy's own address.
	RgfxCreated copy = ser.createBufferOverPages(0x3000, 0, 0x60000, 0x1000, 0x123);
	CHECK(copy.posted && !copy.sharesAddress && ids.allocAddress(1, 1) == 0);
	(void)expectMessage<RgfxBufferOverPages>(ring, RGFX_CMD_BUFFER_OVER_PAGES);
	RgfxCreated m = ser.createManagedBuffer(copy.obj_id, 0x3000, 0x10, 0x3000,
	                                        RgfxSerializer::kDefaultAddressAlign, 0x123,
	                                        copy.gpu_address + 0x123);
	CHECK(m.posted && m.obj_id != 0 && m.sharesAddress && m.gpu_address == copy.gpu_address);
	auto mm = expectMessage<RgfxManagedBufferCreateWhole>(ring, RGFX_CMD_MANAGED_BUFFER_CREATE);
	CHECK(mm.create.obj_id == m.obj_id && mm.create.cpu_buffer_obj_id == copy.obj_id);
	CHECK(mm.create.gpu_address == copy.gpu_address + 0x123 && mm.create.length == 0x3000 &&
	      mm.zero.zero_from == 0x3000);
	CHECK(f.taken.empty() && ids.privateBlocksHeld() == 0 && ids.privatePicksLive() == 0);

	// A slab's copy, at its offset in a heap at the host's address.
	const uint64_t at = (1ull << 45) + 0x1000;
	RgfxCreated s = ser.createManagedBuffer(0x79, 0x800, 0x10, 0,
	                                        RgfxSerializer::kDefaultAddressAlign, 0, at);
	CHECK(s.posted && s.sharesAddress && s.gpu_address == at);
	CHECK(expectMessage<RgfxManagedBufferCreateWhole>(ring, RGFX_CMD_MANAGED_BUFFER_CREATE)
	          .create.gpu_address == at);
	CHECK(f.taken.empty() && ids.privatePicksLive() == 0);

	// No copy address, or one below the offset: no object, nothing posted.
	RgfxCreated none = ser.createManagedBuffer(0x79, 0x800, 0x10, 0);
	RgfxCreated below = ser.createManagedBuffer(0x79, 0x800, 0x10, 0,
	                                            RgfxSerializer::kDefaultAddressAlign, 0x123, 0x100);
	CHECK(none.obj_id == 0 && !none.posted && below.obj_id == 0 && !below.posted);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(!ring.next(&h, &bytes));

	// A refused one gives nothing back: the copy's pick stays held, and the range full. Given
	// back, the range has room again, so the check sees a release.
	RgfxCreated refused = m;
	refused.posted = false;
	CHECK(ser.refused(refused));
	CHECK(ids.allocAddress(1, 1) == 0);
	ser.releaseAddress(copy.gpu_address);
	CHECK(ids.allocAddress(1, 1) == copy.gpu_address);
}

// A heap, buffer or device-placed texture (Managed or not) whose GPU address finds no room in the
// context's range is no object: the create posts nothing and answers ids of 0, so the caller
// answers nil, as Metal does for one it cannot place, and a texture's resource id goes back; a
// smaller one that fits is made after it, and a texture in a heap, which takes no address.
void test_creates_with_no_address_post_nothing()
{
	TestRing ring(0x8000);
	ring.setConsumerIdle(1);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	const uint64_t past = RGFX_CONTEXT_VA_SPAN; // its span, with the guard page, passes the range
	RgfxCreated h = ser.createHeap(past, 0, 1);
	RgfxCreated hh = ser.createHeapOverHostMemory(past, 0, 1, 0x40000);
	RgfxCreated b = ser.createBuffer(past, 0);
	RgfxCreated bh = ser.createBufferOverHostMemory(past, 0, 0x50000);
	RgfxCreated bp = ser.createBufferOverPages(past, 0, 0x60000, 0x1000, 0x123);
	RgfxCreated wraps = ser.createBufferOverPages(UINT64_MAX - 0x100, 0, 0x60000, 0x1000, 0x123);
	RgfxCreated mp = ser.createManagedBuffer(1, past, 0x10, 0);
	RgfxCreated mwraps = ser.createManagedBuffer(1, UINT64_MAX - 0x100, 0x10, 0,
	                                             RgfxSerializer::kDefaultAddressAlign, 0x123);
	RgfxCreated hp = ser.createHeapOverPages(past, 0, 1, 0x60000, 0x4000);
	RgfxTextureDescriptor desc = {};
	desc.pixel_format = 80;
	desc.texture_type = 2;
	desc.width = desc.height = desc.depth = 1;
	desc.mipmap_level_count = desc.array_length = desc.sample_count = 1;
	RgfxCreated t = ser.createTexture(desc, RGFX_OBJ_NONE, 0, 256, past);
	RgfxCreated tm = ser.createManagedTexture(desc, 9, 4, 1, 1, 0, 256, past);
	for (const RgfxCreated &c : {h, hh, b, bh, bp, wraps, mp, mwraps, hp, t, tm})
		CHECK(!c.posted && c.obj_id == 0 && c.gpu_address == 0 && c.resource_id == 0);
	RgfxMsgHeader header = {};
	std::vector<uint8_t> bytes;
	CHECK(!ring.next(&header, &bytes));
	RgfxCreated fits = ser.createHeap(0x10000, 0, 1);
	CHECK(fits.posted && fits.obj_id == 1 && fits.gpu_address != 0);
	CHECK(expectMessage<RgfxHeapCreate>(ring, RGFX_CMD_HEAP_CREATE).obj_id == 1);
	// The textures gave their resource id back: the next texture takes the first.
	RgfxCreated placed = ser.createTexture(desc, fits.obj_id, 0, 256, past);
	CHECK(placed.posted && placed.obj_id == 2 && placed.resource_id == 1 &&
	      placed.gpu_address == 0);
	auto pm = expectMessage<RgfxTextureCreate>(ring, RGFX_CMD_TEXTURE_CREATE);
	CHECK(pm.obj_id == 2 && pm.resource_id == 1 && pm.heap_obj_id == fits.obj_id &&
	      pm.gpu_address == 0);
}

void test_lifecycle_messages()
{
	TestRing ring(0x8000);
	ring.setConsumerIdle(1); // the host is idle, so each post rings the doorbell
	int before = TestRing::doorbells.load();
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);

	// Queue.
	RgfxCreated q = ser.createQueue(0x20000, 0x30000, 0x1000, 0x1000, 2, 1);
	CHECK(q.posted);
	CHECK(q.obj_id == 1);
	CHECK(q.seqno == 1); // the first message on the ring
	auto qm = expectMessage<RgfxQueueCreate>(ring, RGFX_CMD_QUEUE_CREATE);
	CHECK(qm.header.seqno == 1);
	CHECK(qm.obj_id == 1);
	CHECK(qm.submit_offset == 0x20000 && qm.complete_offset == 0x30000);
	CHECK(qm.submit_data_size == 0x1000 && qm.complete_data_size == 0x1000);
	CHECK(qm.priority == 2 && qm.vector == 1);

	// Heap (device) and heap over host memory.
	RgfxCreated h = ser.createHeap(0x10000, 0, 1 /*Placement*/, 4096);
	CHECK(h.posted && (h.gpu_address % 4096) == 0);
	auto hm = expectMessage<RgfxHeapCreate>(ring, RGFX_CMD_HEAP_CREATE);
	CHECK(hm.obj_id == h.obj_id && hm.gpu_address == h.gpu_address);
	CHECK(hm.size_bytes == 0x10000 && hm.heap_type == 1);

	RgfxCreated hh = ser.createHeapOverHostMemory(0x2000, 0, 0, 0x40000);
	CHECK(hh.posted);
	auto hhm = expectMessage<RgfxHeapOverHostMemory>(ring, RGFX_CMD_HEAP_OVER_HOST_MEMORY);
	CHECK(hhm.obj_id == hh.obj_id && hhm.region_offset == 0x40000 && hhm.size_bytes == 0x2000);

	// Buffers: device, over host memory, at offset in a heap.
	RgfxCreated b = ser.createBuffer(0x1000, 0);
	CHECK(b.posted && b.gpu_address != 0);
	auto bm = expectMessage<RgfxBufferCreate>(ring, RGFX_CMD_BUFFER_CREATE);
	CHECK(bm.obj_id == b.obj_id && bm.gpu_address == b.gpu_address && bm.length == 0x1000);
	CHECK(bm.reserved_2c == 0);

	RgfxCreated bh = ser.createBufferOverHostMemory(0x800, 0, 0x50000);
	CHECK(bh.posted);
	auto bhm = expectMessage<RgfxBufferOverHostMemory>(ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
	CHECK(bhm.obj_id == bh.obj_id && bhm.region_offset == 0x50000 && bhm.length == 0x800);

	// Over a pages block, at the pointer's offset in its page: the pick holds the span from the
	// page's start, and the buffer's address keeps the offset.
	RgfxCreated bp = ser.createBufferOverPages(0x3000, 0, 0x60000, 0x1000, 0x123);
	CHECK(bp.posted && bp.gpu_address % RgfxIdSpace::kAddressGranule == 0);
	auto bpm = expectMessage<RgfxBufferOverPages>(ring, RGFX_CMD_BUFFER_OVER_PAGES);
	CHECK(bpm.obj_id == bp.obj_id && bpm.gpu_address == bp.gpu_address + 0x123);
	CHECK(bpm.block_offset == 0x60000 && bpm.block_size == 0x1000);
	CHECK(bpm.offset == 0x123 && bpm.length == 0x3000 && bpm.reserved_44 == 0);
	RgfxCreated next = ser.createBuffer(0x1000, 0);
	CHECK(next.gpu_address >= bp.gpu_address + RgfxIdSpace::addressSpan(0x3123));
	(void)expectMessage<RgfxBufferCreate>(ring, RGFX_CMD_BUFFER_CREATE);

	// A Managed buffer over that one, as a no-copy Managed buffer over the pages is: its pick
	// holds the span from the page's start too, and its address keeps the offset. The app's
	// bytes: nothing said of them, its zero bytes from its length.
	RgfxCreated mb = ser.createManagedBuffer(bp.obj_id, 0x3000, 0x10, 0x3000,
	                                         RgfxSerializer::kDefaultAddressAlign, 0x123);
	CHECK(mb.posted && mb.gpu_address % RgfxIdSpace::kAddressGranule == 0);
	auto mbm = expectMessage<RgfxManagedBufferCreateWhole>(ring, RGFX_CMD_MANAGED_BUFFER_CREATE);
	CHECK(mbm.create.obj_id == mb.obj_id && mbm.create.cpu_buffer_obj_id == bp.obj_id);
	CHECK(mbm.create.gpu_address == mb.gpu_address + 0x123 && mbm.create.length == 0x3000 &&
	      mbm.create.options == 0x10 && mbm.create.reserved_34 == 0);
	CHECK(mbm.zero.zero_from == 0x3000);
	RgfxCreated beyond = ser.createBuffer(0x1000, 0);
	CHECK(beyond.gpu_address >= mb.gpu_address + RgfxIdSpace::addressSpan(0x3123));
	(void)expectMessage<RgfxBufferCreate>(ring, RGFX_CMD_BUFFER_CREATE);

	// Its CPU copy's zero bytes, as the caller gives them, up to its length: from a byte before
	// the length, that byte; from the length or past it, the length, which says nothing.
	const uint64_t zeroes[][2] = {{0, 0},           {0x80, 0x80},     {0xfff, 0xfff},
	                              {0x1000, 0x1000}, {0x1001, 0x1000}, {UINT64_MAX, 0x1000}};
	for (const auto &zero : zeroes) {
		RgfxCreated z = ser.createManagedBuffer(bp.obj_id, 0x1000, 0x10, zero[0]);
		CHECK(z.posted);
		auto zm = expectMessage<RgfxManagedBufferCreateWhole>(ring, RGFX_CMD_MANAGED_BUFFER_CREATE);
		CHECK(zm.create.obj_id == z.obj_id && zm.create.length == 0x1000);
		CHECK(zm.zero.zero_from == zero[1]);
	}

	// A slab's heap over a pages block's guest RAM, at an address of its size.
	RgfxCreated slab = ser.createHeapOverPages(0x200000, 0x200, 1, 0x60000, 0x4000);
	CHECK(slab.posted && slab.gpu_address % RgfxIdSpace::kAddressGranule == 0);
	auto slabm = expectMessage<RgfxHeapOverPages>(ring, RGFX_CMD_HEAP_OVER_PAGES);
	CHECK(slabm.obj_id == slab.obj_id && slabm.gpu_address == slab.gpu_address);
	CHECK(slabm.block_offset == 0x60000 && slabm.block_size == 0x4000 && slabm.options == 0x200);
	CHECK(slabm.size_bytes == 0x200000 && slabm.heap_type == 1);
	RgfxCreated after = ser.createBuffer(0x1000, 0);
	CHECK(after.gpu_address >= slab.gpu_address + RgfxIdSpace::addressSpan(0x200000));
	(void)expectMessage<RgfxBufferCreate>(ring, RGFX_CMD_BUFFER_CREATE);

	RgfxCreated bo = ser.createBufferAtOffset(h.obj_id, 0x100, 0x200, 0);
	CHECK(bo.posted && bo.gpu_address == 0); // derived host-side from the heap
	auto bom = expectMessage<RgfxBufferAtOffset>(ring, RGFX_CMD_BUFFER_AT_OFFSET);
	CHECK(bom.obj_id == bo.obj_id && bom.heap_obj_id == h.obj_id);
	CHECK(bom.offset == 0x100 && bom.length == 0x200);

	// Textures: device-placed and heap-placed and from a buffer.
	RgfxTextureDescriptor desc = {};
	desc.pixel_format = 80; // MTLPixelFormatBGRA8Unorm
	desc.texture_type = 2;  // MTLTextureType2D
	desc.width = 1920;
	desc.height = 1080;
	desc.depth = 1;
	desc.mipmap_level_count = 1;
	desc.array_length = 1;
	desc.sample_count = 1;
	desc.usage = 4; // MTLTextureUsageRenderTarget

	RgfxCreated t = ser.createTexture(desc);
	CHECK(t.posted && t.resource_id != 0 && t.gpu_address != 0);
	auto tm = expectMessage<RgfxTextureCreate>(ring, RGFX_CMD_TEXTURE_CREATE);
	CHECK(tm.obj_id == t.obj_id && tm.resource_id == t.resource_id);
	CHECK(tm.gpu_address == t.gpu_address && tm.heap_obj_id == 0);
	CHECK(tm.desc.width == 1920 && tm.desc.height == 1080 && tm.desc.pixel_format == 80);

	RgfxCreated th = ser.createTexture(desc, h.obj_id, 0x400);
	CHECK(th.posted && th.gpu_address == 0); // heap-placed: no device address
	auto thm = expectMessage<RgfxTextureCreate>(ring, RGFX_CMD_TEXTURE_CREATE);
	CHECK(thm.heap_obj_id == h.obj_id && thm.heap_offset == 0x400);

	RgfxCreated tb = ser.createTextureFromBuffer(b.obj_id, 0x10, 7680, desc);
	CHECK(tb.posted);
	auto tbm = expectMessage<RgfxTextureFromBuffer>(ring, RGFX_CMD_TEXTURE_FROM_BUFFER);
	CHECK(tbm.buffer_obj_id == b.obj_id && tbm.offset == 0x10 && tbm.bytes_per_row == 7680);
	CHECK(tbm.resource_id == tb.resource_id && tbm.desc.width == 1920);

	// A view: the parent and shape as given, the ids picked.
	RgfxTextureView v = {};
	v.obj_id = 0x77;      // not read
	v.resource_id = 0x77; // not read
	v.parent_obj_id = tb.obj_id;
	v.pixel_format = 81;
	v.flags = RGFX_TEXTURE_VIEW_SUBRANGE | RGFX_TEXTURE_VIEW_SWIZZLE;
	v.texture_type = 2;
	v.level_count = 1;
	v.slice_start = 0;
	v.slice_count = 1;
	v.swizzle = 0x05020202;
	RgfxCreated tv = ser.createTextureView(v);
	CHECK(tv.posted && tv.resource_id != 0 && tv.resource_id != tb.resource_id);
	auto tvm = expectMessage<RgfxTextureView>(ring, RGFX_CMD_TEXTURE_VIEW);
	CHECK(tvm.obj_id == tv.obj_id && tvm.resource_id == tv.resource_id);
	CHECK(tvm.parent_obj_id == tb.obj_id && tvm.pixel_format == 81 && tvm.flags == 3);
	CHECK(tvm.texture_type == 2 && tvm.level_count == 1 && tvm.slice_count == 1);
	CHECK(tvm.swizzle == 0x05020202);

	// Sampler: the float clamps and bias are stored as bits, the reduction mode and the bias in
	// the create's extension.
	RgfxSamplerParams sp = {};
	sp.min_filter = 1;
	sp.mag_filter = 1;
	sp.mip_filter = 1;
	sp.max_anisotropy = 4;
	sp.lod_min_clamp = 0.0f;
	sp.lod_max_clamp = 7.5f;
	sp.flags = 0x1;
	sp.reduction_mode = 2;
	sp.lod_bias = -1.25f;
	RgfxCreated s = ser.createSampler(sp);
	CHECK(s.posted);
	auto sm = expectMessage<RgfxSamplerCreateWhole>(ring, RGFX_CMD_SAMPLER_CREATE);
	CHECK(sm.create.obj_id == s.obj_id && sm.create.max_anisotropy == 4 && sm.create.flags == 0x1);
	float lodMax;
	std::memcpy(&lodMax, &sm.create.lod_max_clamp_bits, 4);
	CHECK(lodMax == 7.5f);
	float bias;
	std::memcpy(&bias, &sm.ext.lod_bias_bits, 4);
	CHECK(sm.ext.reduction_mode == 2 && bias == -1.25f);

	// Depth/stencil.
	RgfxStencilDescriptor front = {};
	front.stencil_compare = 7;
	front.read_mask = 0xff;
	front.write_mask = 0xff;
	RgfxStencilDescriptor back = {};
	RgfxCreated ds = ser.createDepthStencil(4 /*compare*/, true, front, back);
	CHECK(ds.posted);
	auto dsm = expectMessage<RgfxDepthStencilCreate>(ring, RGFX_CMD_DEPTH_STENCIL_CREATE);
	CHECK(dsm.obj_id == ds.obj_id && dsm.depth_compare == 4 && dsm.depth_write_enabled == 1);
	CHECK(dsm.front.stencil_compare == 7 && dsm.front.read_mask == 0xff);

	// Sync objects.
	RgfxCreated ev = ser.createEvent();
	CHECK(ev.posted);
	auto evm = expectMessage<RgfxObjectCreate>(ring, RGFX_CMD_EVENT_CREATE);
	CHECK(evm.obj_id == ev.obj_id);

	RgfxCreated fe = ser.createFence();
	CHECK(fe.posted);
	auto fem = expectMessage<RgfxObjectCreate>(ring, RGFX_CMD_FENCE_CREATE);
	CHECK(fem.obj_id == fe.obj_id);

	RgfxCreated se = ser.createSharedEvent(42);
	CHECK(se.posted);
	auto sem = expectMessage<RgfxSharedEventCreate>(ring, RGFX_CMD_SHARED_EVENT_CREATE);
	CHECK(sem.obj_id == se.obj_id && sem.initial_value == 42);

	RgfxCreated vs = ser.createVideoSession(RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_DECODE);
	CHECK(vs.posted);
	auto vsm = expectMessage<RgfxVideoSessionCreate>(ring, RGFX_CMD_VIDEO_SESSION_CREATE);
	CHECK(vsm.obj_id == vs.obj_id && vsm.codec == RGFX_VIDEO_CODEC_H264 &&
	      vsm.direction == RGFX_VIDEO_DECODE);

	// Destroy carries the id alone.
	CHECK(ser.destroy(b.obj_id));
	auto dm = expectMessage<RgfxObjectDestroy>(ring, RGFX_CMD_OBJECT_DESTROY);
	CHECK(dm.obj_id == b.obj_id);

	// The ring is now drained and every post rang the idle host's doorbell.
	RgfxMsgHeader spare = {};
	std::vector<uint8_t> none;
	CHECK(!ring.next(&spare, &none));
	CHECK(TestRing::doorbells.load() > before);
}

void test_library_message()
{
	TestRing ring(0x8000);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, nullptr, nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);

	const uint8_t air[6] = { 1, 2, 3, 4, 5, 6 };
	RgfxCreated lib = ser.createLibrary(air, sizeof(air));
	CHECK(lib.posted);

	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(ring.next(&h, &bytes));
	CHECK(h.type == RGFX_CMD_LIBRARY_CREATE);
	// header struct (72) + 6 bytes, rounded up to a multiple of 8 = 80.
	CHECK(h.size == 80);
	RgfxLibraryCreate head;
	std::memcpy(&head, bytes.data(), sizeof(head));
	CHECK(head.obj_id == lib.obj_id && head.byte_length == 6);
	CHECK(head.flags == 0); // no key
	// A ring the process writes itself hands back no answer.
	CHECK(lib.status == kRgfxNoAnswer);
	CHECK(std::memcmp(bytes.data() + RGFX_LIBRARY_BYTES_OFFSET, air, sizeof(air)) == 0);

	// A library larger than the ring cannot be posted in one message.
	std::vector<uint8_t> huge(ring.dataSize, 0xcc);
	RgfxCreated big = ser.createLibrary(huge.data(), huge.size());
	CHECK(!big.posted);
}

void test_pipeline_messages()
{
	TestRing ring(0x8000);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, nullptr, nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);

	// A render pipeline: two function names, a label, one colour attachment with
	// source-over blend, and a one-entry vertex descriptor, all in the tail.
	RgfxRenderPipelineParams rp = {};
	rp.vertex.library = 10;
	rp.vertex.entry_point = "vmain";
	rp.fragment.library = 10;
	rp.fragment.entry_point = "frag_main";
	rp.label = "pipe";
	rp.raster_sample_count = 1;
	rp.color_attachment_count = 1;
	rp.colors[0].pixel_format = 80; // BGRA8Unorm
	rp.colors[0].blending_enabled = 1;
	rp.colors[0].source_rgb_blend_factor = 1;      // One
	rp.colors[0].destination_rgb_blend_factor = 5; // OneMinusSourceAlpha
	rp.colors[0].write_mask = 0xf;
	rp.depth_pixel_format = 252; // Depth32Float
	RgfxVertexBufferLayout layout = {};
	layout.buffer_index = 0;
	layout.stride = 32;
	layout.step_function = 1;
	rp.vertex_layouts = &layout;
	rp.vertex_layout_count = 1;
	RgfxVertexAttribute attr = {};
	attr.format = 29; // Float4
	rp.vertex_attributes = &attr;
	rp.vertex_attribute_count = 1;
	rp.flags = RGFX_RENDER_PIPELINE_RASTERIZATION_ENABLED |
	           RGFX_RENDER_PIPELINE_TESSELLATION_FACTOR_SCALE;
	// A tessellation state of no default but the factor format's (Half, Metal's only), and call
	// stack depths of their own.
	rp.tessellation_partition_mode = 3;           // FractionalEven
	rp.max_tessellation_factor = 32;
	rp.tessellation_factor_format = 0;            // Half
	rp.tessellation_control_point_index_type = 2; // UInt32
	rp.tessellation_factor_step_function = 1;     // PerPatch
	rp.tessellation_output_winding_order = 1;     // CounterClockwise
	rp.max_vertex_call_stack_depth = 2;
	rp.max_fragment_call_stack_depth = 3;

	RgfxCreated r = ser.createRenderPipeline(rp);
	CHECK(r.posted);

	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(ring.next(&h, &bytes));
	CHECK(h.type == RGFX_CMD_RENDER_PIPELINE_CREATE);
	CHECK(h.size == bytes.size());
	CHECK(h.size % 8 == 0);
	RgfxRenderPipelineCreate rc;
	std::memcpy(&rc, bytes.data(), sizeof(rc));
	CHECK(rc.obj_id == r.obj_id);
	CHECK(rc.vertex_library_obj_id == 10 && rc.fragment_library_obj_id == 10);
	CHECK(rc.color_attachment_count == 1 && rc.colors[0].pixel_format == 80);
	CHECK(rc.colors[0].destination_rgb_blend_factor == 5);
	CHECK(rc.depth_pixel_format == 252);
	CHECK(rc.flags & RGFX_RENDER_PIPELINE_TESSELLATION_FACTOR_SCALE);
	CHECK(rc.tessellation_partition_mode == 3 && rc.max_tessellation_factor == 32);
	CHECK(rc.tessellation_factor_format == 0 && rc.tessellation_control_point_index_type == 2);
	CHECK(rc.tessellation_factor_step_function == 1 && rc.tessellation_output_winding_order == 1);
	CHECK(rc.max_vertex_call_stack_depth == 2 && rc.max_fragment_call_stack_depth == 3);
	CHECK(rc.vertex_layout_count == 1 && rc.vertex_attribute_count == 1);
	CHECK(rc.vertex_name_len == 5);
	CHECK(std::memcmp(bytes.data() + rc.vertex_name_offset, "vmain", 5) == 0);
	CHECK(rc.fragment_name_len == 9);
	CHECK(std::memcmp(bytes.data() + rc.fragment_name_offset, "frag_main", 9) == 0);
	CHECK(rc.label_len == 4);
	CHECK(std::memcmp(bytes.data() + rc.label_offset, "pipe", 4) == 0);
	RgfxVertexBufferLayout gotLayout;
	std::memcpy(&gotLayout, bytes.data() + rc.vertex_layouts_offset, sizeof(gotLayout));
	CHECK(gotLayout.stride == 32 && gotLayout.step_function == 1);
	RgfxVertexAttribute gotAttr;
	std::memcpy(&gotAttr, bytes.data() + rc.vertex_attributes_offset, sizeof(gotAttr));
	CHECK(gotAttr.format == 29);

	// A compute pipeline specialised by one function constant.
	RgfxFunctionConstant k = {};
	k.index = 2;
	k.data_type = 84; // Long4, the widest
	for (size_t i = 0; i < sizeof(k.value); i++)
		k.value[i] = static_cast<uint8_t>(i + 1);
	RgfxComputePipelineParams cp = {};
	cp.function.library = 10;
	cp.function.entry_point = "cmain";
	cp.function.constants = &k;
	cp.function.constant_count = 1;
	cp.max_total_threads_per_threadgroup = 256;
	cp.flags = RGFX_COMPUTE_PIPELINE_SUPPORT_ICB;

	RgfxCreated c = ser.createComputePipeline(cp);
	CHECK(c.posted);
	CHECK(ring.next(&h, &bytes));
	CHECK(h.type == RGFX_CMD_COMPUTE_PIPELINE_CREATE);
	CHECK(h.size == bytes.size());
	RgfxComputePipelineCreate cc;
	std::memcpy(&cc, bytes.data(), sizeof(cc));
	CHECK(cc.obj_id == c.obj_id && cc.function_library_obj_id == 10);
	CHECK(cc.name_len == 5);
	CHECK(std::memcmp(bytes.data() + cc.name_offset, "cmain", 5) == 0);
	CHECK(cc.constant_count == 1);
	RgfxFunctionConstant gotK;
	std::memcpy(&gotK, bytes.data() + cc.constants_offset, sizeof(gotK));
	CHECK(gotK.index == 2 && gotK.data_type == 84);
	CHECK(std::memcmp(gotK.value, k.value, sizeof(k.value)) == 0);
	CHECK(cc.max_total_threads_per_threadgroup == 256);

	// The host's reflection reply, matched to the create by its seqno: built as
	// the host would and read back, asserting the reply layout the guest reads.
	RgfxPipelineReflection refl = {};
	refl.header.type = RGFX_EVT_PIPELINE_REFLECTION;
	refl.header.size = sizeof(refl);
	refl.header.seqno = c.seqno;
	refl.obj_id = c.obj_id;
	refl.status = RGFX_PIPELINE_OK;
	refl.kind = RGFX_PIPELINE_KIND_COMPUTE;
	refl.max_total_threads_per_threadgroup = 1024;
	refl.thread_execution_width = 32;
	std::vector<uint8_t> wire(sizeof(refl), 0);
	std::memcpy(wire.data(), &refl, sizeof(refl));
	RgfxPipelineReflection back;
	std::memcpy(&back, wire.data(), sizeof(back));
	CHECK(back.header.seqno == c.seqno && back.obj_id == c.obj_id);
	CHECK(back.status == RGFX_PIPELINE_OK && back.kind == RGFX_PIPELINE_KIND_COMPUTE);
	CHECK(back.thread_execution_width == 32);
}

// A mesh render pipeline's create: its three functions and label in the tail, its colours, flags,
// limits and required sizes in its head; and its answer, the reflection with the mesh stages'
// limits after it, which the reply reader hands over whole, as it does a render pipeline's without.
void test_mesh_pipeline_messages()
{
	TestRing ring(0x8000);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, nullptr, nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);

	RgfxFunctionConstant k = {};
	k.index = 3;
	k.data_type = 29;
	RgfxMeshRenderPipelineParams mp = {};
	mp.object.library = 11;
	mp.object.entry_point = "omain";
	mp.mesh.library = 12;
	mp.mesh.entry_point = "mesh_main";
	mp.mesh.constants = &k;
	mp.mesh.constant_count = 1;
	mp.fragment.library = 12;
	mp.fragment.entry_point = "fmain";
	mp.label = "meshes";
	mp.raster_sample_count = 4;
	mp.color_attachment_count = 2;
	mp.colors[1].pixel_format = 80;
	mp.colors[1].write_mask = 0xf;
	mp.depth_pixel_format = 252;
	mp.flags = RGFX_RENDER_PIPELINE_RASTERIZATION_ENABLED |
	           RGFX_MESH_PIPELINE_MESH_TG_MULTIPLE_OF_EXEC_WIDTH;
	mp.max_total_threads_per_object_threadgroup = 64;
	mp.max_total_threads_per_mesh_threadgroup = 32;
	mp.max_total_threadgroups_per_mesh_grid = 8;
	mp.payload_memory_length = 256;
	mp.required_threads_per_object_threadgroup[0] = 2;
	mp.required_threads_per_mesh_threadgroup[0] = 4;
	mp.required_threads_per_mesh_threadgroup[2] = 1;

	RgfxCreated r = ser.createMeshRenderPipeline(mp);
	CHECK(r.posted);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(ring.next(&h, &bytes));
	CHECK(h.type == RGFX_CMD_MESH_RENDER_PIPELINE_CREATE);
	CHECK(h.size == bytes.size() && h.size % 8 == 0);
	CHECK(h.seqno == r.seqno && r.seqno == 1); // the first message on the ring
	RgfxMeshRenderPipelineCreate mc;
	std::memcpy(&mc, bytes.data(), sizeof(mc));
	CHECK(mc.obj_id == r.obj_id);
	CHECK(mc.object_library_obj_id == 11 && mc.mesh_library_obj_id == 12 &&
	      mc.fragment_library_obj_id == 12);
	CHECK(mc.object_name_len == 5);
	CHECK(std::memcmp(bytes.data() + mc.object_name_offset, "omain", 5) == 0);
	CHECK(mc.mesh_name_len == 9);
	CHECK(std::memcmp(bytes.data() + mc.mesh_name_offset, "mesh_main", 9) == 0);
	CHECK(mc.fragment_name_len == 5);
	CHECK(std::memcmp(bytes.data() + mc.fragment_name_offset, "fmain", 5) == 0);
	CHECK(mc.label_len == 6);
	CHECK(std::memcmp(bytes.data() + mc.label_offset, "meshes", 6) == 0);
	CHECK(mc.object_constant_count == 0 && mc.mesh_constant_count == 1 &&
	      mc.fragment_constant_count == 0);
	RgfxFunctionConstant gotK;
	std::memcpy(&gotK, bytes.data() + mc.mesh_constants_offset, sizeof(gotK));
	CHECK(gotK.index == 3 && gotK.data_type == 29);
	CHECK(mc.object_linked_count == 0 && mc.mesh_linked_count == 0 &&
	      mc.fragment_linked_count == 0);
	CHECK(mc.raster_sample_count == 4 && mc.color_attachment_count == 2);
	CHECK(mc.colors[0].pixel_format == 0 && mc.colors[1].pixel_format == 80);
	CHECK(mc.depth_pixel_format == 252 && mc.stencil_pixel_format == 0);
	CHECK(mc.flags == (RGFX_RENDER_PIPELINE_RASTERIZATION_ENABLED |
	                   RGFX_MESH_PIPELINE_MESH_TG_MULTIPLE_OF_EXEC_WIDTH));
	CHECK(mc.max_total_threads_per_object_threadgroup == 64);
	CHECK(mc.max_total_threads_per_mesh_threadgroup == 32);
	CHECK(mc.max_total_threadgroups_per_mesh_grid == 8);
	CHECK(mc.payload_memory_length == 256);
	CHECK(mc.required_threads_per_object_threadgroup[0] == 2);
	CHECK(mc.required_threads_per_mesh_threadgroup[0] == 4 &&
	      mc.required_threads_per_mesh_threadgroup[1] == 0 &&
	      mc.required_threads_per_mesh_threadgroup[2] == 1);

	// The answer: 104 bytes, the mesh stages' limits at RGFX_PIPELINE_REFLECTION_MESH_OFFSET.
	CHECK(sizeof(RgfxPipelineReflection) == RGFX_PIPELINE_REFLECTION_MESH_OFFSET);
	uint8_t answer[sizeof(RgfxPipelineReflection) + sizeof(RgfxMeshPipelineReflection)] = {};
	CHECK(sizeof(answer) == 104);
	RgfxPipelineReflection refl = {};
	refl.header.type = RGFX_EVT_PIPELINE_REFLECTION;
	refl.header.size = sizeof(answer);
	refl.header.seqno = 21;
	refl.obj_id = r.obj_id;
	refl.status = RGFX_PIPELINE_OK;
	refl.kind = RGFX_PIPELINE_KIND_MESH;
	RgfxMeshPipelineReflection limits = {};
	limits.max_total_threads_per_object_threadgroup = 64;
	limits.max_total_threads_per_mesh_threadgroup = 32;
	limits.object_thread_execution_width = 32;
	limits.mesh_thread_execution_width = 32;
	limits.max_total_threadgroups_per_mesh_grid = UINT32_MAX;
	std::memcpy(answer, &refl, sizeof(refl));
	std::memcpy(answer + RGFX_PIPELINE_REFLECTION_MESH_OFFSET, &limits, sizeof(limits));

	RgfxReplyReader reader;
	RgfxPipelineReply got = {};
	reader.expectReflection(21, [&got](const RgfxPipelineReply &m) { got = m; });
	reader.deliver(answer, sizeof(answer));
	CHECK(got.obj_id == r.obj_id && got.kind == RGFX_PIPELINE_KIND_MESH);
	CHECK(got.hasMesh);
	CHECK(std::memcmp(&got.mesh, &limits, sizeof(limits)) == 0);

	// Before its handler, kept whole too.
	refl.header.seqno = 22;
	std::memcpy(answer, &refl, sizeof(refl));
	reader.deliver(answer, sizeof(answer));
	got = {};
	reader.expectReflection(22, [&got](const RgfxPipelineReply &m) { got = m; });
	CHECK(got.hasMesh && got.mesh.max_total_threadgroups_per_mesh_grid == UINT32_MAX);

	// A refusal carries no limits, and neither does a render pipeline's answer.
	refl.header.size = sizeof(refl);
	refl.header.seqno = 23;
	refl.status = RGFX_PIPELINE_REFUSED;
	reader.deliver(&refl, sizeof(refl));
	got = {};
	reader.expectReflection(23, [&got](const RgfxPipelineReply &m) { got = m; });
	CHECK(got.status == RGFX_PIPELINE_REFUSED && !got.hasMesh);
	refl.header.size = sizeof(answer);
	refl.header.seqno = 24;
	refl.status = RGFX_PIPELINE_OK;
	refl.kind = RGFX_PIPELINE_KIND_RENDER;
	std::memcpy(answer, &refl, sizeof(refl));
	reader.deliver(answer, sizeof(answer));
	got = {};
	reader.expectReflection(24, [&got](const RgfxPipelineReply &m) { got = m; });
	CHECK(got.kind == RGFX_PIPELINE_KIND_RENDER && !got.hasMesh);
}

void test_ring_wrap_and_no_room()
{
	// A small ring so a message must pad to the end and wrap.
	const uint32_t kData = 256;
	TestRing ring(kData);
	RgfxRingWriter writer(ring.base(), kData, RGFX_CONTROL_RING_ID, nullptr, nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);

	// Each OBJECT_DESTROY is 24 bytes. Post and drain to push the producer near
	// the ring's end, then post one that must pad over the tail.
	for (int i = 0; i < 10; i++) {
		CHECK(ser.destroy(100 + i));
		auto m = expectMessage<RgfxObjectDestroy>(ring, RGFX_CMD_OBJECT_DESTROY);
		CHECK(m.obj_id == static_cast<uint64_t>(100 + i));
	}
	// 10 * 24 = 240; 16 bytes to the end. The next 24-byte message pads and wraps.
	CHECK(ser.destroy(999));
	auto wrapped = expectMessage<RgfxObjectDestroy>(ring, RGFX_CMD_OBJECT_DESTROY);
	CHECK(wrapped.obj_id == 999);

	// Fill the ring without draining until a post finds no room.
	TestRing full(kData);
	RgfxRingWriter fw(full.base(), kData, RGFX_CONTROL_RING_ID, nullptr, nullptr);
	RgfxIdSpace fids;
	RgfxSerializer fser(fw, fids);
	int posted = 0;
	bool hitNoRoom = false;
	for (int i = 0; i < 100; i++) {
		if (fser.destroy(i)) {
			posted++;
		} else {
			hitNoRoom = true;
			break;
		}
	}
	CHECK(hitNoRoom);
	CHECK(posted == static_cast<int>(kData / sizeof(RgfxObjectDestroy))); // 256/24 = 10
}

// A minimal host-side op-stream consumer, the mirror of
// crates/reims-vgpu/src/graphics/decode/cmdbuf.rs's walk: it reads each
// RgfxOpHeader, checks `len`, splits the body, and tracks which encoder is open.
struct OpWalk {
	const uint8_t *ops;
	size_t size; // op_bytes
	size_t pos = 0;

	OpWalk(const uint8_t *p, size_t n) : ops(p), size(n) {}

	bool done() const { return pos >= size; }

	// Reads the next op; fills `tag`, `flags`, `len` and points `body`/`body_len`
	// at its body. Returns false (and does not advance) on a structural fault.
	bool next(uint16_t *tag, uint16_t *flags, uint32_t *len, const uint8_t **body,
	          size_t *body_len)
	{
		if (size - pos < sizeof(RgfxOpHeader))
			return false;
		RgfxOpHeader h;
		std::memcpy(&h, ops + pos, sizeof(h));
		if (h.len < RGFX_OP_BODY_OFFSET || (h.len % 8) != 0 || h.len > size - pos)
			return false;
		*tag = h.tag;
		*flags = h.flags;
		*len = h.len;
		*body = ops + pos + RGFX_OP_BODY_OFFSET;
		*body_len = h.len - RGFX_OP_BODY_OFFSET;
		pos += h.len;
		return true;
	}
};

// `tag >> 12`, the op's group, as the host splits it.
uint16_t opGroup(uint16_t tag) { return static_cast<uint16_t>(tag >> 12); }

// The object and mesh stages' ops and the mesh draws, each in its own body.
void test_mesh_ops()
{
	TestRing ring(0x8000);
	ring.setConsumerIdle(1);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxEncoder enc(writer, /*queue*/ 1, /*cb*/ 100);
	RgfxRenderPassDesc pass = {};
	RgfxRenderPassColorAttachment color = {};
	color.texture_obj_id = 4;
	enc.beginRenderPass(pass, &color, 1);
	RgfxBufferBinding b = {};
	b.buffer_obj_id = 9;
	b.offset = 16;
	enc.renderSetObjectBuffers(2, &b, 1);
	enc.renderSetMeshBuffers(3, &b, 1);
	enc.renderSetObjectBufferOffset(32, 2);
	enc.renderSetMeshBufferOffset(48, 3);
	RgfxTextureBinding t = {};
	t.texture_obj_id = 5;
	enc.renderSetObjectTextures(1, &t, 1);
	enc.renderSetMeshTextures(0, &t, 1);
	RgfxSamplerBinding sb = {};
	sb.sampler_obj_id = 6;
	enc.renderSetObjectSamplerStates(0, &sb, 1);
	enc.renderSetMeshSamplerStates(1, &sb, 1);
	RgfxSamplerLodBinding lb = {};
	lb.sampler_obj_id = 7;
	enc.renderSetObjectSamplerStatesLodClamps(2, &lb, 1);
	enc.renderSetMeshSamplerStatesLodClamps(3, &lb, 1);
	enc.renderSetObjectThreadgroupMemoryLength(256, 0);
	const uint64_t grid[3] = {4, 2, 1};
	const uint64_t object[3] = {32, 1, 1};
	const uint64_t mesh[3] = {64, 1, 1};
	enc.renderDrawMeshThreadgroups(grid, object, mesh);
	enc.renderDrawMeshThreads(grid, object, mesh);
	enc.renderDrawMeshThreadgroupsIndirect(9, 24, object, mesh);
	enc.endEncoder();
	uint64_t seqno = 0;
	CHECK(enc.finish(&seqno));

	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(ring.next(&h, &bytes));
	RgfxCommandBuffer cb = {};
	std::memcpy(&cb, bytes.data(), sizeof(cb));
	OpWalk walk(bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET, cb.op_bytes);
	uint16_t tag = 0, flags = 0;
	uint32_t len = 0;
	const uint8_t *body = nullptr;
	size_t body_len = 0;
	CHECK(walk.next(&tag, &flags, &len, &body, &body_len) && tag == RGFX_OP_ENCODER_BEGIN_RENDER);

	// The ranged sets: a range head, then the entries.
	struct Ranged {
		uint16_t tag;
		uint64_t location;
		size_t entry;
		const void *first;
	};
	const Ranged ranged[] = {
	    {RGFX_OP_RENDER_SET_OBJECT_BUFFERS, 2, sizeof(b), &b},
	    {RGFX_OP_RENDER_SET_MESH_BUFFERS, 3, sizeof(b), &b},
	};
	for (const Ranged &want : ranged) {
		CHECK(walk.next(&tag, &flags, &len, &body, &body_len) && tag == want.tag);
		RgfxOpSetObjectBuffers head = {};
		CHECK(body_len == sizeof(head) + want.entry);
		std::memcpy(&head, body, sizeof(head));
		CHECK(head.range_location == want.location && head.range_length == 1);
		CHECK(std::memcmp(body + sizeof(head), want.first, want.entry) == 0);
	}
	const uint16_t offsets[] = {RGFX_OP_RENDER_SET_OBJECT_BUFFER_OFFSET,
	                            RGFX_OP_RENDER_SET_MESH_BUFFER_OFFSET};
	const uint64_t offsetValues[][2] = {{32, 2}, {48, 3}};
	for (int i = 0; i < 2; i++) {
		CHECK(walk.next(&tag, &flags, &len, &body, &body_len) && tag == offsets[i]);
		RgfxOpSetMeshBufferOffset o = {};
		CHECK(body_len == sizeof(o));
		std::memcpy(&o, body, sizeof(o));
		CHECK(o.offset == offsetValues[i][0] && o.index == offsetValues[i][1]);
	}
	const Ranged more[] = {
	    {RGFX_OP_RENDER_SET_OBJECT_TEXTURES, 1, sizeof(t), &t},
	    {RGFX_OP_RENDER_SET_MESH_TEXTURES, 0, sizeof(t), &t},
	    {RGFX_OP_RENDER_SET_OBJECT_SAMPLER_STATES, 0, sizeof(sb), &sb},
	    {RGFX_OP_RENDER_SET_MESH_SAMPLER_STATES, 1, sizeof(sb), &sb},
	    {RGFX_OP_RENDER_SET_OBJECT_SAMPLER_STATES_LOD_CLAMPS, 2, sizeof(lb), &lb},
	    {RGFX_OP_RENDER_SET_MESH_SAMPLER_STATES_LOD_CLAMPS, 3, sizeof(lb), &lb},
	};
	for (const Ranged &want : more) {
		CHECK(walk.next(&tag, &flags, &len, &body, &body_len) && tag == want.tag);
		uint64_t head[2] = {};
		CHECK(body_len == sizeof(head) + want.entry);
		std::memcpy(head, body, sizeof(head));
		CHECK(head[0] == want.location && head[1] == 1);
		CHECK(std::memcmp(body + sizeof(head), want.first, want.entry) == 0);
	}
	CHECK(walk.next(&tag, &flags, &len, &body, &body_len) &&
	      tag == RGFX_OP_RENDER_SET_OBJECT_THREADGROUP_MEMORY_LENGTH);
	RgfxOpSetObjectThreadgroupMemoryLength tml = {};
	CHECK(body_len == sizeof(tml));
	std::memcpy(&tml, body, sizeof(tml));
	CHECK(tml.length == 256 && tml.index == 0);

	CHECK(walk.next(&tag, &flags, &len, &body, &body_len) &&
	      tag == RGFX_OP_RENDER_DRAW_MESH_THREADGROUPS);
	RgfxOpDrawMeshThreadgroups tg = {};
	CHECK(body_len == sizeof(tg));
	std::memcpy(&tg, body, sizeof(tg));
	CHECK(tg.threadgroups_per_grid[0] == 4 && tg.threadgroups_per_grid[1] == 2 &&
	      tg.threadgroups_per_grid[2] == 1);
	CHECK(tg.threads_per_object_threadgroup[0] == 32 && tg.threads_per_mesh_threadgroup[0] == 64);
	CHECK(walk.next(&tag, &flags, &len, &body, &body_len) && tag == RGFX_OP_RENDER_DRAW_MESH_THREADS);
	RgfxOpDrawMeshThreads th = {};
	CHECK(body_len == sizeof(th));
	std::memcpy(&th, body, sizeof(th));
	CHECK(th.threads_per_grid[0] == 4 && th.threads_per_mesh_threadgroup[0] == 64);
	CHECK(walk.next(&tag, &flags, &len, &body, &body_len) &&
	      tag == RGFX_OP_RENDER_DRAW_MESH_THREADGROUPS_INDIRECT);
	RgfxOpDrawMeshThreadgroupsIndirect ind = {};
	CHECK(body_len == sizeof(ind));
	std::memcpy(&ind, body, sizeof(ind));
	CHECK(ind.indirect_buffer_obj_id == 9 && ind.indirect_buffer_offset == 24);
	CHECK(ind.threads_per_object_threadgroup[0] == 32 && ind.threads_per_mesh_threadgroup[0] == 64);
	CHECK(walk.next(&tag, &flags, &len, &body, &body_len) && tag == RGFX_OP_ENCODER_END);
	CHECK(walk.done());
}

// post() waits while the ring is full, ringing the doorbell and calling its hook, until the
// host takes room; a message over half the ring does not fit() and is refused at once; and
// beforePublish sees the seqno before `producer` moves.
// The doorbell rings where the host marked the ring asleep, and where it marked it walking only
// for a command buffer whose stream ends in its ask for its submission, which the host answers
// inside the doorbell's write; a walking host takes the rest once its walk is done, and one that
// takes the ring is rung for nothing.
void test_ring_doorbell_follows_the_mark()
{
	TestRing ring(0x8000);
	RgfxRingWriter writer(ring.base(), ring.dataSize, 1, TestRing::doorbell, nullptr);
	const auto rang = [&](uint32_t mark, bool asks) {
		ring.setConsumerIdle(mark);
		const int before = TestRing::doorbells.load();
		RgfxEncoder enc(writer, /*queue*/ 1, /*cb*/ 100);
		enc.beginComputePass();
		enc.endEncoder();
		if (asks)
			enc.appendSubmitted();
		CHECK(enc.endsInAsk() == asks);
		CHECK(enc.finish());
		RgfxMsgHeader header = {};
		std::vector<uint8_t> body;
		CHECK(ring.next(&header, &body) && header.type == RGFX_CMD_COMMAND_BUFFER);
		return TestRing::doorbells.load() - before;
	};
	CHECK(rang(RGFX_CONSUMER_ASLEEP, false) == 1);
	CHECK(rang(RGFX_CONSUMER_ASLEEP, true) == 1);
	CHECK(rang(RGFX_CONSUMER_WALKING, false) == 0);
	CHECK(rang(RGFX_CONSUMER_WALKING, true) == 1);
	CHECK(rang(0, false) == 0);
	CHECK(rang(0, true) == 0);
}

// Where the device's word holds DOORBELL_ASKS, a command buffer that asks for its submission
// rings its ring's id with RGFX_DOORBELL_ASKS, for the host to answer inside the write, and
// every other doorbell writes the id alone; where it does not, every doorbell writes the id
// alone, as before the bit.
void test_ring_doorbell_asks_where_the_device_takes_them()
{
	constexpr uint32_t kRingId = 0x1000; // a region offset, as the device's ring ids are
	static std::vector<uint32_t> written;
	const auto record = [](void *, uint32_t value) { written.push_back(value); };
	for (const bool takes : { false, true }) {
		TestRing ring(0x8000);
		RgfxRingWriter writer(ring.base(), ring.dataSize, kRingId, record, nullptr);
		writer.setDoorbellAsks(takes);
		const auto rang = [&](uint32_t mark, bool asks) {
			ring.setConsumerIdle(mark);
			written.clear();
			RgfxEncoder enc(writer, /*queue*/ 1, /*cb*/ 100);
			enc.beginComputePass();
			enc.endEncoder();
			if (asks)
				enc.appendSubmitted();
			CHECK(enc.finish());
			RgfxMsgHeader header = {};
			std::vector<uint8_t> body;
			CHECK(ring.next(&header, &body) && header.type == RGFX_CMD_COMMAND_BUFFER);
			CHECK(written.size() == 1);
			return written.empty() ? 0u : written[0];
		};
		const uint32_t asking = takes ? kRingId | RGFX_DOORBELL_ASKS : kRingId;
		CHECK(rang(RGFX_CONSUMER_ASLEEP, false) == kRingId);
		CHECK(rang(RGFX_CONSUMER_ASLEEP, true) == asking);
		CHECK(rang(RGFX_CONSUMER_WALKING, true) == asking);
	}
}

void test_ring_post_waits_for_room()
{
	constexpr uint32_t kData = 0x1000;
	TestRing ring(kData);
	RgfxRingWriter writer(ring.base(), kData, 1, TestRing::doorbell, nullptr);
	CHECK(writer.fits(kData / 2));
	CHECK(!writer.fits(kData / 2 + 8));
	CHECK(!writer.fits(12));
	std::vector<uint8_t> big(kData / 2 + 8, 0);
	CHECK(!writer.post(RGFX_CMD_COMMAND_BUFFER, big.data(), static_cast<uint32_t>(big.size()),
	                   nullptr));
	CHECK(ring.producer() == 0);

	uint8_t msg[256] = {};
	int filled = 0;
	while (writer.tryPost(RGFX_CMD_COMMAND_BUFFER, msg, sizeof(msg), nullptr))
		filled++;
	CHECK(filled == kData / 256);
	uint64_t full = ring.producer();
	int before = TestRing::doorbells.load();

	// The host takes everything 20 ms from now. The post's time is counted from before the host's
	// thread starts, so it is at least those 20 ms however late this thread gets to the post.
	auto start = std::chrono::steady_clock::now();
	std::thread host([&ring, full] {
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		__atomic_store_n(reinterpret_cast<uint64_t *>(ring.mem.data() +
		                                              __builtin_offsetof(RgfxRingHeader, consumer)),
		                 full, __ATOMIC_RELEASE);
	});
	std::atomic<int> waits{0};
	uint64_t seen = 0, producerAtPublish = ~0ull, seqno = 0;
	CHECK(writer.post(RGFX_CMD_COMMAND_BUFFER, msg, sizeof(msg), &seqno, [&] { waits++; },
	                  [&](uint64_t s) {
		                  seen = s;
		                  producerAtPublish = ring.producer();
	                  }));
	auto took = std::chrono::steady_clock::now() - start;
	host.join();
	CHECK(took >= std::chrono::milliseconds(15));
	CHECK(waits.load() >= 1);
	CHECK(TestRing::doorbells.load() > before); // rung while it waited
	CHECK(seen == seqno && seqno == static_cast<uint64_t>(filled) + 1);
	CHECK(producerAtPublish == full);           // seen before the message was published
	CHECK(ring.producer() == full + sizeof(msg));
}

// With room events (SUBMIT_ROOM), a post that finds the ring full stores 1 in the ring's
// producer_waiting and waits for the host's room record, not a sleep's end: room the host takes
// with no record leaves it waiting (its backstop is 20 ms), and the record, which the queue's
// reader hands over as it reads RGFX_EVT_SUBMIT_ROOM, ends the wait at once. A post that finds
// room touches no flag.
//
// The backstop also ends the wait, 20 ms after it began, wherever a busy host keeps this test's
// threads off a CPU that long. So a run counts only where it ends inside the backstop from
// before its post began: its post can then have been ended by the record alone, and was still
// waiting when looked at. Runs repeat until one counts.
void test_ring_post_waits_for_the_room_record()
{
	// RgfxRingWriter.cpp's kRoomBackstopUs.
	constexpr auto kBackstop = std::chrono::milliseconds(20);
	constexpr uint32_t kData = 0x1000;
	TestRing ring(kData);
	RgfxRingWriter writer(ring.base(), kData, 1, TestRing::doorbell, nullptr);
	auto room = std::make_shared<RgfxRingRoomEvents>();
	writer.setRoomEvents(room);
	auto *waitingFlag = reinterpret_cast<uint32_t *>(
	    ring.mem.data() + __builtin_offsetof(RgfxRingHeader, producer_waiting));
	auto waiting = [waitingFlag] { return __atomic_load_n(waitingFlag, __ATOMIC_SEQ_CST); };
	uint8_t msg[256] = {};
	CHECK(writer.post(RGFX_CMD_COMMAND_BUFFER, msg, sizeof(msg), nullptr));
	CHECK(waiting() == 0);
	while (writer.tryPost(RGFX_CMD_COMMAND_BUFFER, msg, sizeof(msg), nullptr)) {
	}
	// Each run starts with the ring full: it takes one message and its post puts one back.
	bool counted = false;
	for (int run = 0; run < 50 && !counted; run++) {
		// The host clears the flag as it posts its record.
		__atomic_store_n(waitingFlag, 0u, __ATOMIC_SEQ_CST);
		const int rung = TestRing::doorbells.load();
		std::atomic<bool> posted{false};
		bool postedOK = false;
		const auto began = std::chrono::steady_clock::now();
		std::thread poster([&] {
			postedOK = writer.post(RGFX_CMD_COMMAND_BUFFER, msg, sizeof(msg), nullptr);
			posted = true;
		});
		// The post rings the doorbell once its look after storing the flag found no room: from
		// then on only the record or the backstop ends its wait.
		for (int i = 0; i < 5000 && TestRing::doorbells.load() == rung; i++)
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		CHECK(TestRing::doorbells.load() > rung);
		CHECK(waiting() == 1);
		// The host takes one message and posts no record: the post still waits.
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(ring.next(&h, &bytes));
		std::this_thread::sleep_for(std::chrono::milliseconds(3));
		const bool postedBeforeRecord = posted;
		// Its record: the post looks again, and finds the room.
		room->arrived();
		poster.join();
		CHECK(posted && postedOK);
		if (std::chrono::steady_clock::now() - began < kBackstop) {
			CHECK(!postedBeforeRecord);
			counted = true;
		}
	}
	CHECK(counted);

	// A queue's reader hands each RGFX_EVT_SUBMIT_ROOM on its completion ring to the room.
	TestRing completion(kData);
	RgfxReplyReader reader(completion.base(), kData);
	reader.setRoomEvents(room);
	const uint64_t before = room->seen();
	RgfxMsgHeader record = {RGFX_EVT_SUBMIT_ROOM, sizeof(RgfxMsgHeader), 0};
	CHECK(completion.postFromHost(&record, sizeof(record)));
	CHECK(reader.pump() == 1);
	CHECK(room->seen() == before + 1);
}

// A writer's room line counts each sleep for room, those a room record ended apart, and their
// time in all and at most; it is logged at the first sleep, then at most every interval, and once
// more as the writer goes where sleeps came since the last line.
void test_ring_room_sleeps_count_for_the_line()
{
	RgfxRingRoomSleeps s;
	CHECK(!s.unlogged());
	CHECK(s.note(300, true, 1000, 500));
	CHECK(!s.unlogged());
	CHECK(!s.note(900, false, 1200, 500));
	CHECK(s.unlogged());
	CHECK(s.note(100, true, 1500, 500));
	CHECK(!s.unlogged());
	CHECK(s.sleeps == 3 && s.woken == 2);
	CHECK(s.sleptNs == 1300 && s.mostNs == 900);
	RgfxRingRoomSleeps z;
	CHECK(z.note(1, false, 0, 500));
	CHECK(!z.note(1, false, 1, 500));
	CHECK(z.unlogged());
}

// Blocks of a plain buffer standing in for the device region, as a queue's pool hands them out
// (RgfxMessageBlocks): each taken block at the next free offset, its size the next power of two
// of at least 64 KiB, and what was given back.
struct TestBlocks final : RgfxMessageBlocks {
	std::vector<uint8_t> mem = std::vector<uint8_t>(0x800000, 0);
	uint64_t next = 0;
	std::vector<RgfxMessageBlock> taken, given;
	bool refuse = false;
	bool take(uint64_t size, RgfxMessageBlock *block) override
	{
		uint64_t bytes = 0x10000;
		while (bytes < size)
			bytes <<= 1;
		if (refuse || next + bytes > mem.size())
			return false;
		*block = {next, mem.data() + next, bytes};
		next += bytes;
		taken.push_back(*block);
		return true;
	}
	void giveBack(const RgfxMessageBlock &block) override { given.push_back(block); }
};

// A command buffer's message is built in the builder's own buffer and posted from it on the ring
// while it fits half of it, else copied at the commit into one block, from its start, which the
// COMMAND_BUFFER_IN_REGION names, the whole COMMAND_BUFFER in it; the block goes to whoever holds
// the posted message. The buffer grows to the queue's guess as the message outgrows the ring.
// The surface uses a commit puts first lead the op stream, in the order put, before every op,
// however many.
void test_command_buffer_storage()
{
	TestRing ring(0x1000); // up to 2 KiB a message on the ring
	ring.setConsumerIdle(1);
	RgfxRingWriter writer(ring.base(), ring.dataSize, 1, TestRing::doorbell, nullptr);
	auto blocks = std::make_shared<TestBlocks>();
	auto walkOps = [](const uint8_t *ops, uint64_t n, std::vector<uint16_t> *tags,
	                  std::vector<uint64_t> *firstWords) {
		OpWalk walk(ops, n);
		while (!walk.done()) {
			uint16_t tag = 0, flags = 0;
			uint32_t len = 0;
			const uint8_t *body = nullptr;
			size_t body_len = 0;
			if (!walk.next(&tag, &flags, &len, &body, &body_len))
				return false;
			uint64_t w = 0;
			if (body_len >= 8)
				std::memcpy(&w, body, 8);
			tags->push_back(tag);
			firstWords->push_back(w);
		}
		return true;
	};
	auto uses = [](size_t n, uint64_t base) {
		std::vector<RgfxOpSurfaceUse> u(n);
		for (size_t i = 0; i < n; i++)
			u[i].block_offset = base + i;
		return u;
	};

	// Small, with a few uses: on the ring, the uses first, then the ops.
	{
		RgfxEncoder enc(writer, 1, 2);
		enc.setMessageBlocks(blocks, 0);
		enc.beginBlitPass();
		enc.blitFillBuffer(9, 0, 4, 1);
		enc.endEncoder();
		std::vector<RgfxOpSurfaceUse> u = uses(3, 100);
		enc.prependSurfaceUses(u.data(), 2);
		enc.prependSurfaceUses(u.data() + 2, 1); // put later, so first
		CHECK(!enc.inBlock());
		uint64_t seqno = 0;
		RgfxMessageBlock held = {};
		CHECK(enc.finish(&seqno, {}, {}, &held) && held.ptr == nullptr);
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(ring.next(&h, &bytes) && h.type == RGFX_CMD_COMMAND_BUFFER && h.size == bytes.size());
		RgfxCommandBuffer cb = {};
		std::memcpy(&cb, bytes.data(), sizeof(cb));
		CHECK(cb.cb_obj_id == 2 && cb.op_bytes == bytes.size() - RGFX_COMMAND_BUFFER_OPS_OFFSET);
		std::vector<uint16_t> tags;
		std::vector<uint64_t> words;
		CHECK(walkOps(bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET, cb.op_bytes, &tags, &words));
		CHECK((tags == std::vector<uint16_t>{RGFX_OP_SURFACE_USE, RGFX_OP_SURFACE_USE,
		                                      RGFX_OP_SURFACE_USE, RGFX_OP_ENCODER_BEGIN_BLIT,
		                                      RGFX_OP_BLIT_FILL_BUFFER, RGFX_OP_ENCODER_END}));
		CHECK(words[0] == 102 && words[1] == 100 && words[2] == 101 && words[4] == 9);
		CHECK(blocks->taken.empty());
	}

	// Past the ring: built in the builder's buffer, which grows to the guess at the op that
	// outgrows the ring and doubles past it, with no block taken; more uses than the room before
	// the header move the ops; the commit copies the message into one block, from its start.
	{
		RgfxEncoder enc(writer, 1, 3, 0x100);
		enc.setMessageBlocks(blocks, 0x10000);
		enc.beginBlitPass();
		int fills = 0;
		auto fill = [&enc, &fills] {
			enc.blitFillBuffer(static_cast<RgfxObjId>(fills), 0, 4, 1);
			fills++;
		};
		while (enc.fitsRing())
			fill();
		CHECK(fills > 2048 / 48 && fills < 4096 / 32);
		// Its first growth past the ring takes the guess at once.
		const size_t fitted = enc.capacity();
		while (enc.capacity() == fitted)
			fill();
		CHECK(fitted < 0x10000 && enc.capacity() >= 0x10000);
		while (fills < 5000) // 5000 fills of 40 bytes: 200 KB, past the 64 KiB guess
			fill();
		enc.endEncoder();
		CHECK(blocks->taken.empty() && !enc.inBlock());
		std::vector<RgfxOpSurfaceUse> u = uses(100, 1000);
		enc.prependSurfaceUses(u.data(), u.size());
		uint64_t seqno = 0;
		RgfxMessageBlock held = {};
		CHECK(enc.finish(&seqno, {}, {}, &held));
		CHECK(blocks->taken.size() == 1 && held.ptr == blocks->taken.back().ptr &&
		      blocks->given.empty() && !enc.inBlock());
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(ring.next(&h, &bytes) && h.type == RGFX_CMD_COMMAND_BUFFER_IN_REGION);
		RgfxCommandBufferInRegion ref = {};
		std::memcpy(&ref, bytes.data(), sizeof(ref));
		CHECK(ref.region_offset == held.offset && ref.byte_length <= held.size);
		const uint8_t *message = blocks->mem.data() + ref.region_offset;
		RgfxCommandBuffer cb = {};
		std::memcpy(&cb, message, sizeof(cb));
		CHECK(cb.header.type == RGFX_CMD_COMMAND_BUFFER && cb.header.size == ref.byte_length &&
		      cb.header.seqno == 0 && cb.cb_obj_id == 3 &&
		      cb.op_bytes == ref.byte_length - RGFX_COMMAND_BUFFER_OPS_OFFSET);
		std::vector<uint16_t> tags;
		std::vector<uint64_t> words;
		CHECK(walkOps(message + RGFX_COMMAND_BUFFER_OPS_OFFSET, cb.op_bytes, &tags, &words));
		CHECK(tags.size() == 100 + 5002);
		bool inOrder = true;
		for (size_t i = 0; i < 100; i++)
			inOrder = inOrder && tags[i] == RGFX_OP_SURFACE_USE && words[i] == 1000 + i;
		CHECK(inOrder && tags[100] == RGFX_OP_ENCODER_BEGIN_BLIT && tags.back() == RGFX_OP_ENCODER_END);
		for (size_t i = 0; i < 5000; i++)
			inOrder = inOrder && tags[101 + i] == RGFX_OP_BLIT_FILL_BUFFER && words[101 + i] == i;
		CHECK(inOrder);
	}

	// Past the ring with no block to be had: built in a buffer of its own, and not posted.
	{
		blocks->refuse = true;
		RgfxEncoder enc(writer, 1, 4);
		enc.setMessageBlocks(blocks, 0);
		enc.beginBlitPass();
		for (int i = 0; i < 200; i++)
			enc.blitFillBuffer(9, 0, 4, 1);
		enc.endEncoder();
		CHECK(!enc.fitsRing() && enc.messageSize() > 2048);
		RgfxMessageBlock held = {};
		CHECK(!enc.finish(nullptr, {}, {}, &held) && held.ptr == nullptr && !enc.inBlock());
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(!ring.next(&h, &bytes));
		blocks->refuse = false;
	}
}

// A message's posted time (tryPost's stampAt) is written in the ring's copy, not the caller's
// bytes, after the copy and beforePublish and before the publish: no earlier than the clock read
// in beforePublish, no later than the doorbell the publish rings. A post with no stamp writes
// nothing there, and a forwarding writer none.
void test_ring_stamps_the_posted_time_before_the_publish()
{
	static uint64_t rang = 0;
	const auto doorbell = [](void *, uint32_t) { rang = rgfxGuestClockNs(); };
	TestRing ring(0x8000);
	ring.setConsumerIdle(RGFX_CONSUMER_ASLEEP);
	RgfxRingWriter writer(ring.base(), ring.dataSize, 1, doorbell, nullptr);
	uint8_t msg[96] = {};
	const uint64_t marker = 0x0123456789abcdefull;
	std::memcpy(msg + 64, &marker, sizeof(marker));
	uint64_t before = 0;
	const auto seen = [&](uint64_t) { before = rgfxGuestClockNs(); };
	CHECK(writer.tryPost(RGFX_CMD_COMMAND_BUFFER, msg, sizeof(msg), nullptr, nullptr, seen, false,
	                     RGFX_GUEST_TIMES_POSTED_OFFSET));
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(ring.next(&h, &bytes) && bytes.size() == sizeof(msg));
	uint64_t posted = 0;
	std::memcpy(&posted, bytes.data() + RGFX_GUEST_TIMES_POSTED_OFFSET, sizeof(posted));
	CHECK(before != 0 && rang != 0);
	CHECK(posted >= before && posted <= rang);
	uint64_t kept = 0;
	std::memcpy(&kept, msg + 64, sizeof(kept));
	CHECK(kept == marker); // the caller's bytes stay as they were
	// No stamp asked: the bytes go as given.
	CHECK(writer.tryPost(RGFX_CMD_COMMAND_BUFFER, msg, sizeof(msg), nullptr));
	CHECK(ring.next(&h, &bytes));
	std::memcpy(&posted, bytes.data() + RGFX_GUEST_TIMES_POSTED_OFFSET, sizeof(posted));
	CHECK(posted == marker);
	// Past the message: no stamp.
	uint8_t small[32] = {};
	CHECK(writer.tryPost(RGFX_CMD_COMMAND_BUFFER, small, sizeof(small), nullptr, nullptr, {}, false,
	                     RGFX_GUEST_TIMES_POSTED_OFFSET));
	CHECK(ring.next(&h, &bytes) && bytes.size() == sizeof(small));
}

// Where the buffer stamps the guest's times (stampGuestTimes), its stream starts with
// RGFX_OP_GUEST_TIMES, the last op put first, before its surface uses and its commit order: its
// making and its commit as noted, and its posted time stamped as it went on the ring, on the
// ring for a message there and in the block for one too large for the ring. A buffer that does
// not stamp carries none, and its message keeps its size.
void test_guest_times_go_first()
{
	TestRing ring(0x1000);
	ring.setConsumerIdle(1);
	RgfxRingWriter writer(ring.base(), ring.dataSize, 1, TestRing::doorbell, nullptr);
	auto blocks = std::make_shared<TestBlocks>();
	const auto build = [&](RgfxEncoder &enc, int fills) {
		enc.setMessageBlocks(blocks, 0);
		enc.beginBlitPass();
		for (int i = 0; i < fills; i++)
			enc.blitFillBuffer(9, 0, 4, 1);
		enc.endEncoder();
		RgfxOpSurfaceUse use = {};
		use.block_offset = 77;
		enc.prependSurfaceUses(&use, 1);
		RgfxSurfaceOrderWait wait = {2, 5};
		enc.prependCommitOrder(&wait, 1);
	};
	const auto tags = [](const uint8_t *ops, uint64_t n) {
		std::vector<uint16_t> out;
		OpWalk walk(ops, n);
		while (!walk.done()) {
			uint16_t tag = 0, flags = 0;
			uint32_t len = 0;
			const uint8_t *body = nullptr;
			size_t body_len = 0;
			CHECK(walk.next(&tag, &flags, &len, &body, &body_len));
			out.push_back(tag);
		}
		return out;
	};
	const std::vector<uint16_t> plain = {RGFX_OP_COMMIT_ORDER, RGFX_OP_SURFACE_USE,
	                                     RGFX_OP_ENCODER_BEGIN_BLIT, RGFX_OP_BLIT_FILL_BUFFER,
	                                     RGFX_OP_ENCODER_END};
	std::vector<uint16_t> timed = plain;
	timed.insert(timed.begin(), RGFX_OP_GUEST_TIMES);

	// Not stamped: no times.
	{
		RgfxEncoder enc(writer, 1, 2);
		build(enc, 1);
		CHECK(!enc.stampsGuestTimes());
		const uint64_t size = enc.messageSize();
		CHECK(enc.finish());
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(ring.next(&h, &bytes) && h.size == size);
		CHECK(tags(bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET,
		           bytes.size() - RGFX_COMMAND_BUFFER_OPS_OFFSET) == plain);
	}
	// Stamped, on the ring: first, its size counted before the finish.
	{
		RgfxEncoder enc(writer, 1, 3);
		enc.stampGuestTimes(1000);
		CHECK(enc.stampsGuestTimes());
		build(enc, 1);
		enc.noteCommitted(2000);
		const uint64_t size = enc.messageSize();
		const uint64_t before = rgfxGuestClockNs();
		CHECK(enc.fitsRing() && enc.finish());
		const uint64_t after = rgfxGuestClockNs();
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(ring.next(&h, &bytes) && h.size == size && h.type == RGFX_CMD_COMMAND_BUFFER);
		RgfxCommandBuffer cb = {};
		std::memcpy(&cb, bytes.data(), sizeof(cb));
		CHECK(cb.op_bytes == bytes.size() - RGFX_COMMAND_BUFFER_OPS_OFFSET);
		CHECK(tags(bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET, cb.op_bytes) == timed);
		RgfxOpHeader op = {};
		std::memcpy(&op, bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET, sizeof(op));
		CHECK(op.len == RGFX_OP_BODY_OFFSET + sizeof(RgfxOpGuestTimes));
		RgfxOpGuestTimes t = {};
		std::memcpy(&t, bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET + RGFX_OP_BODY_OFFSET,
		            sizeof(t));
		CHECK(t.began_ns == 1000 && t.committed_ns == 2000);
		CHECK(t.posted_ns >= before && t.posted_ns <= after);
	}
	// Stamped, too large for the ring: in the block, its posted time there too.
	{
		RgfxEncoder enc(writer, 1, 4);
		enc.stampGuestTimes(3000);
		build(enc, 200);
		enc.noteCommitted(4000);
		CHECK(!enc.fitsRing());
		const uint64_t size = enc.messageSize();
		RgfxMessageBlock held = {};
		const uint64_t before = rgfxGuestClockNs();
		CHECK(enc.finish(nullptr, {}, {}, &held));
		const uint64_t after = rgfxGuestClockNs();
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(ring.next(&h, &bytes) && h.type == RGFX_CMD_COMMAND_BUFFER_IN_REGION);
		RgfxCommandBufferInRegion ref = {};
		std::memcpy(&ref, bytes.data(), sizeof(ref));
		CHECK(ref.byte_length == size && ref.region_offset == held.offset);
		const uint8_t *message = blocks->mem.data() + ref.region_offset;
		RgfxCommandBuffer cb = {};
		std::memcpy(&cb, message, sizeof(cb));
		const std::vector<uint16_t> got = tags(message + RGFX_COMMAND_BUFFER_OPS_OFFSET, cb.op_bytes);
		CHECK(got.size() == timed.size() + 199 && got[0] == RGFX_OP_GUEST_TIMES &&
		      got[1] == RGFX_OP_COMMIT_ORDER);
		RgfxOpGuestTimes t = {};
		std::memcpy(&t, message + RGFX_COMMAND_BUFFER_OPS_OFFSET + RGFX_OP_BODY_OFFSET, sizeof(t));
		CHECK(t.began_ns == 3000 && t.committed_ns == 4000);
		CHECK(t.posted_ns >= before && t.posted_ns <= after);
	}
}

// A pass's sample positions follow its colour records, and its flag says so; a pass without
// them carries none, whatever flag its description held. The depth store action options are
// one op.
void test_render_pass_sample_positions()
{
	TestRing ring(0x8000);
	ring.setConsumerIdle(1);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxEncoder enc(writer, /*queue*/ 1, /*cb*/ 100);
	RgfxRenderPassDesc pass = {};
	RgfxRenderPassColorAttachment colors[2] = {};
	colors[0].texture_obj_id = 4;
	colors[1].texture_obj_id = 5;
	RgfxRenderPassSamplePositions positions = {};
	positions.count = 2;
	const float xy[4] = {0.25f, 0.75f, 0.5625f, 0.125f};
	std::memcpy(&positions.position_bits[0][0], &xy[0], 4);
	std::memcpy(&positions.position_bits[0][1], &xy[1], 4);
	std::memcpy(&positions.position_bits[1][0], &xy[2], 4);
	std::memcpy(&positions.position_bits[1][1], &xy[3], 4);
	enc.beginRenderPass(pass, colors, 2, &positions);
	enc.renderSetDepthStoreActionOptions(1);
	enc.endEncoder();
	pass.flags = RGFX_PASS_FLAG_SAMPLE_POSITIONS | RGFX_PASS_FLAG_RT_SIZE;
	enc.beginRenderPass(pass, colors, 1);
	enc.endEncoder();
	uint64_t seqno = 0;
	CHECK(enc.finish(&seqno));

	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(ring.next(&h, &bytes));
	RgfxCommandBuffer cb = {};
	std::memcpy(&cb, bytes.data(), sizeof(cb));
	OpWalk walk(bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET, cb.op_bytes);
	uint16_t tag = 0, flags = 0;
	uint32_t len = 0;
	const uint8_t *body = nullptr;
	size_t body_len = 0;

	CHECK(walk.next(&tag, &flags, &len, &body, &body_len));
	CHECK(tag == RGFX_OP_ENCODER_BEGIN_RENDER);
	const size_t colorEnd = RGFX_RENDER_PASS_COLOR_OFFSET + 2 * sizeof(RgfxRenderPassColorAttachment);
	CHECK(body_len == colorEnd + sizeof(RgfxRenderPassSamplePositions));
	RgfxRenderPassDesc got = {};
	std::memcpy(&got, body, sizeof(got));
	CHECK(got.flags == RGFX_PASS_FLAG_SAMPLE_POSITIONS && got.color_count == 2);
	RgfxRenderPassSamplePositions gotPositions = {};
	std::memcpy(&gotPositions, body + colorEnd, sizeof(gotPositions));
	CHECK(std::memcmp(&gotPositions, &positions, sizeof(positions)) == 0);

	CHECK(walk.next(&tag, &flags, &len, &body, &body_len));
	CHECK(tag == RGFX_OP_RENDER_SET_DEPTH_STORE_ACTION_OPTIONS);
	CHECK(body_len == sizeof(RgfxOpSetDepthStoreActionOptions));
	RgfxOpSetDepthStoreActionOptions options = {};
	std::memcpy(&options, body, sizeof(options));
	CHECK(options.store_action_options == 1);
	CHECK(walk.next(&tag, &flags, &len, &body, &body_len) && tag == RGFX_OP_ENCODER_END);

	CHECK(walk.next(&tag, &flags, &len, &body, &body_len));
	CHECK(tag == RGFX_OP_ENCODER_BEGIN_RENDER);
	CHECK(body_len == RGFX_RENDER_PASS_COLOR_OFFSET + sizeof(RgfxRenderPassColorAttachment));
	std::memcpy(&got, body, sizeof(got));
	CHECK(got.flags == RGFX_PASS_FLAG_RT_SIZE && got.color_count == 1);
	CHECK(walk.next(&tag, &flags, &len, &body, &body_len) && tag == RGFX_OP_ENCODER_END);
	CHECK(walk.done());
}

// A pass's visibility result record follows its colour records and its sample positions, in
// that order, and its flag says so; a pass without one carries none, whatever flag its
// description held.
void test_render_pass_visibility()
{
	TestRing ring(0x8000);
	ring.setConsumerIdle(1);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxEncoder enc(writer, /*queue*/ 1, /*cb*/ 100);
	RgfxRenderPassDesc pass = {};
	RgfxRenderPassColorAttachment colors[2] = {};
	colors[0].texture_obj_id = 4;
	colors[1].texture_obj_id = 5;
	RgfxRenderPassSamplePositions positions = {};
	positions.count = 2;
	RgfxRenderPassVisibility visibility = {};
	visibility.buffer_obj_id = 0x77;
	visibility.result_type = 1;
	enc.beginRenderPass(pass, colors, 2, nullptr, &visibility);
	enc.renderSetVisibilityResultMode(2, 8);
	enc.endEncoder();
	enc.beginRenderPass(pass, colors, 1, &positions, &visibility);
	enc.endEncoder();
	pass.flags = RGFX_PASS_FLAG_VISIBILITY_RESULT;
	enc.beginRenderPass(pass, colors, 1);
	enc.endEncoder();
	uint64_t seqno = 0;
	CHECK(enc.finish(&seqno));

	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(ring.next(&h, &bytes));
	RgfxCommandBuffer cb = {};
	std::memcpy(&cb, bytes.data(), sizeof(cb));
	OpWalk walk(bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET, cb.op_bytes);
	uint16_t tag = 0, flags = 0;
	uint32_t len = 0;
	const uint8_t *body = nullptr;
	size_t body_len = 0;
	RgfxRenderPassDesc got = {};
	RgfxRenderPassVisibility gotVisibility = {};

	CHECK(walk.next(&tag, &flags, &len, &body, &body_len));
	CHECK(tag == RGFX_OP_ENCODER_BEGIN_RENDER);
	const size_t twoColors = RGFX_RENDER_PASS_COLOR_OFFSET + 2 * sizeof(RgfxRenderPassColorAttachment);
	CHECK(body_len == twoColors + sizeof(RgfxRenderPassVisibility));
	std::memcpy(&got, body, sizeof(got));
	CHECK(got.flags == RGFX_PASS_FLAG_VISIBILITY_RESULT && got.color_count == 2);
	std::memcpy(&gotVisibility, body + twoColors, sizeof(gotVisibility));
	CHECK(std::memcmp(&gotVisibility, &visibility, sizeof(visibility)) == 0);
	CHECK(walk.next(&tag, &flags, &len, &body, &body_len));
	CHECK(tag == RGFX_OP_RENDER_SET_VISIBILITY_RESULT_MODE);
	RgfxOpSetVisibilityResultMode mode = {};
	CHECK(body_len == sizeof(mode));
	std::memcpy(&mode, body, sizeof(mode));
	CHECK(mode.mode == 2 && mode.offset == 8);
	CHECK(walk.next(&tag, &flags, &len, &body, &body_len) && tag == RGFX_OP_ENCODER_END);

	CHECK(walk.next(&tag, &flags, &len, &body, &body_len));
	CHECK(tag == RGFX_OP_ENCODER_BEGIN_RENDER);
	const size_t oneColor = RGFX_RENDER_PASS_COLOR_OFFSET + sizeof(RgfxRenderPassColorAttachment);
	CHECK(body_len ==
	      oneColor + sizeof(RgfxRenderPassSamplePositions) + sizeof(RgfxRenderPassVisibility));
	std::memcpy(&got, body, sizeof(got));
	CHECK(got.flags == (RGFX_PASS_FLAG_SAMPLE_POSITIONS | RGFX_PASS_FLAG_VISIBILITY_RESULT));
	RgfxRenderPassSamplePositions gotPositions = {};
	std::memcpy(&gotPositions, body + oneColor, sizeof(gotPositions));
	CHECK(gotPositions.count == 2);
	std::memcpy(&gotVisibility, body + oneColor + sizeof(RgfxRenderPassSamplePositions),
	            sizeof(gotVisibility));
	CHECK(std::memcmp(&gotVisibility, &visibility, sizeof(visibility)) == 0);
	CHECK(walk.next(&tag, &flags, &len, &body, &body_len) && tag == RGFX_OP_ENCODER_END);

	CHECK(walk.next(&tag, &flags, &len, &body, &body_len));
	CHECK(tag == RGFX_OP_ENCODER_BEGIN_RENDER);
	CHECK(body_len == oneColor);
	std::memcpy(&got, body, sizeof(got));
	CHECK(got.flags == 0 && got.color_count == 1);
	CHECK(walk.next(&tag, &flags, &len, &body, &body_len) && tag == RGFX_OP_ENCODER_END);
	CHECK(walk.done());
}

void test_command_buffer_encoder()
{
	TestRing ring(0x8000);
	ring.setConsumerIdle(1);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);

	RgfxEncoder enc(writer, /*queue*/ 1, /*cb*/ 100);

	// Render encoder: a pass with one colour attachment, then a scalar op, an
	// array-tailed op (setVertexBuffers), a no-arg op (setDepthCleared) and a draw.
	RgfxRenderPassDesc pass = {};
	pass.flags = RGFX_PASS_FLAG_RT_SIZE;
	pass.render_target_width = 1920;
	pass.render_target_height = 1080;
	RgfxRenderPassColorAttachment color = {};
	color.texture_obj_id = 4;
	color.load_action = 2;  // Clear
	color.store_action = 1; // Store
	double one = 1.0;
	std::memcpy(&color.clear_color_bits[3], &one, 8);
	enc.beginRenderPass(pass, &color, 1);
	enc.renderSetRenderPipelineState(7);
	enc.renderSetVertexBuffer(2, 0, 0);
	enc.renderSetViewport(0.0, 0.0, 1920.0, 1080.0, 0.0, 1.0);
	RgfxBufferBinding vbs[2] = {};
	vbs[0].buffer_obj_id = 2;
	vbs[0].offset = 0;
	vbs[1].buffer_obj_id = 3;
	vbs[1].offset = 256;
	enc.renderSetVertexBuffers(0, vbs, 2);
	enc.renderSetDepthCleared();
	enc.renderDrawPrimitives(3, 0, 3);
	enc.endEncoder();

	// Compute encoder: default pass, a scalar, a dispatch, an array-tailed op and a
	// no-body op.
	enc.beginComputePass();
	enc.computeSetComputePipelineState(8);
	enc.computeDispatchThreadgroups(16, 8, 1, 32, 1, 1);
	RgfxComputeBufferBinding cbs[2] = {};
	cbs[0].buffer_obj_id = 2;
	cbs[0].offset = 0;
	cbs[1].buffer_obj_id = 3;
	cbs[1].offset = 512;
	enc.computeSetBuffers(0, cbs, 2);
	enc.computeEncodeStartElse();
	enc.endEncoder();

	// Blit encoder: a copy, a fill-texture (trailing bytes), and a fence wait.
	enc.beginBlitPass();
	enc.blitCopyBufferToBuffer(2, 3, 16, 32, 4096);
	const uint8_t fill[5] = { 0xde, 0xad, 0xbe, 0xef, 0x01 };
	enc.blitFillTexture(4, 0, 0, 0, 0, 0, 32, 32, 1, fill, sizeof(fill));
	enc.blitWaitForFence(9);
	enc.endEncoder();

	uint64_t seqno = 0;
	CHECK(enc.finish(&seqno));
	CHECK(seqno == 1);

	// Read the command-buffer message back off the ring and check its framing.
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(ring.next(&h, &bytes));
	CHECK(h.type == RGFX_CMD_COMMAND_BUFFER);
	CHECK(h.size == bytes.size());
	CHECK(h.size % 8 == 0);
	CHECK(h.seqno == 1);

	RgfxCommandBuffer cb = {};
	std::memcpy(&cb, bytes.data(), sizeof(cb));
	CHECK(cb.queue_obj_id == 1);
	CHECK(cb.cb_obj_id == 100);
	CHECK(cb.op_bytes == bytes.size() - RGFX_COMMAND_BUFFER_OPS_OFFSET);
	CHECK(cb.op_bytes == enc.opBytes());

	// The op stream follows the header at RGFX_COMMAND_BUFFER_OPS_OFFSET.
	OpWalk walk(bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET, cb.op_bytes);

	// The exact op sequence the encoder wrote.
	const uint16_t want[] = {
	    RGFX_OP_ENCODER_BEGIN_RENDER,
	    RGFX_OP_RENDER_SET_RENDER_PIPELINE_STATE,
	    RGFX_OP_RENDER_SET_VERTEX_BUFFER,
	    RGFX_OP_RENDER_SET_VIEWPORT,
	    RGFX_OP_RENDER_SET_VERTEX_BUFFERS,
	    RGFX_OP_RENDER_SET_DEPTH_CLEARED,
	    RGFX_OP_RENDER_DRAW_PRIMITIVES,
	    RGFX_OP_ENCODER_END,
	    RGFX_OP_ENCODER_BEGIN_COMPUTE,
	    RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE,
	    RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS,
	    RGFX_OP_COMPUTE_SET_BUFFERS,
	    RGFX_OP_COMPUTE_ENCODE_START_ELSE,
	    RGFX_OP_ENCODER_END,
	    RGFX_OP_ENCODER_BEGIN_BLIT,
	    RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER,
	    RGFX_OP_BLIT_FILL_TEXTURE,
	    RGFX_OP_BLIT_WAIT_FOR_FENCE,
	    RGFX_OP_ENCODER_END,
	};
	const size_t want_n = sizeof(want) / sizeof(want[0]);

	int open = 0; // 0 none, 1 render, 2 compute, 3 blit
	int begins = 0, ends = 0;
	size_t i = 0;
	while (!walk.done()) {
		uint16_t tag = 0, flags = 0;
		uint32_t len = 0;
		const uint8_t *body = nullptr;
		size_t body_len = 0;
		CHECK(walk.next(&tag, &flags, &len, &body, &body_len));
		CHECK(len % 8 == 0);
		CHECK(i < want_n);
		CHECK(tag == want[i]);

		// Framing is balanced: a begin opens exactly one encoder, an end closes it,
		// and every body op's group matches the open encoder.
		if (opGroup(tag) == RGFX_OP_GROUP_FRAMING) {
			if (tag == RGFX_OP_ENCODER_BEGIN_RENDER) {
				CHECK(open == 0);
				open = 1;
				begins++;
			} else if (tag == RGFX_OP_ENCODER_BEGIN_COMPUTE) {
				CHECK(open == 0);
				open = 2;
				begins++;
			} else if (tag == RGFX_OP_ENCODER_BEGIN_BLIT) {
				CHECK(open == 0);
				open = 3;
				begins++;
			} else if (tag == RGFX_OP_ENCODER_END) {
				CHECK(open != 0);
				open = 0;
				ends++;
			}
		} else {
			CHECK(static_cast<int>(opGroup(tag)) == open);
		}

		// Spot-check the bodies of representative ops across all three encoders.
		if (tag == RGFX_OP_ENCODER_BEGIN_RENDER) {
			CHECK(len == 8 + sizeof(RgfxRenderPassDesc) + 1 * sizeof(RgfxRenderPassColorAttachment));
			RgfxRenderPassDesc gotPass;
			std::memcpy(&gotPass, body, sizeof(gotPass));
			CHECK(gotPass.color_count == 1);
			CHECK(gotPass.flags == RGFX_PASS_FLAG_RT_SIZE);
			CHECK(gotPass.render_target_width == 1920 && gotPass.render_target_height == 1080);
			RgfxRenderPassColorAttachment gotColor;
			std::memcpy(&gotColor, body + RGFX_RENDER_PASS_COLOR_OFFSET, sizeof(gotColor));
			CHECK(gotColor.texture_obj_id == 4);
			CHECK(gotColor.load_action == 2 && gotColor.store_action == 1);
			double a;
			std::memcpy(&a, &gotColor.clear_color_bits[3], 8);
			CHECK(a == 1.0);
		} else if (tag == RGFX_OP_RENDER_SET_VERTEX_BUFFER) {
			RgfxOpSetVertexBuffer b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.buffer_obj_id == 2 && b.offset == 0 && b.index == 0);
		} else if (tag == RGFX_OP_RENDER_SET_VERTEX_BUFFERS) {
			// Array-tailed: head's range then two RgfxBufferBinding records.
			CHECK(len == 8 + sizeof(RgfxOpSetVertexBuffers) + 2 * sizeof(RgfxBufferBinding));
			RgfxOpSetVertexBuffers head;
			std::memcpy(&head, body, sizeof(head));
			CHECK(head.range_location == 0 && head.range_length == 2);
			RgfxBufferBinding e0, e1;
			std::memcpy(&e0, body + sizeof(head), sizeof(e0));
			std::memcpy(&e1, body + sizeof(head) + sizeof(e0), sizeof(e1));
			CHECK(e0.buffer_obj_id == 2 && e0.offset == 0);
			CHECK(e1.buffer_obj_id == 3 && e1.offset == 256);
		} else if (tag == RGFX_OP_RENDER_SET_DEPTH_CLEARED) {
			CHECK(len == 8 && body_len == 0); // no-arg render op: header only
		} else if (tag == RGFX_OP_RENDER_DRAW_PRIMITIVES) {
			RgfxOpDrawPrimitives b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.primitive_type == 3 && b.vertex_start == 0 && b.vertex_count == 3);
		} else if (tag == RGFX_OP_ENCODER_BEGIN_COMPUTE) {
			CHECK(len == 8 && body_len == 0); // default pass: empty body
		} else if (tag == RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS) {
			RgfxComputeOpDispatchThreadgroups b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.threadgroups_width == 16 && b.threadgroups_height == 8);
			CHECK(b.threads_width == 32 && b.threads_height == 1 && b.threads_depth == 1);
		} else if (tag == RGFX_OP_COMPUTE_SET_BUFFERS) {
			CHECK(len == 8 + sizeof(RgfxComputeOpSetBuffers) +
			                 2 * sizeof(RgfxComputeBufferBinding));
			RgfxComputeOpSetBuffers head;
			std::memcpy(&head, body, sizeof(head));
			CHECK(head.location == 0 && head.length == 2);
			RgfxComputeBufferBinding e1;
			std::memcpy(&e1, body + sizeof(head) + sizeof(e1), sizeof(e1));
			CHECK(e1.buffer_obj_id == 3 && e1.offset == 512);
		} else if (tag == RGFX_OP_COMPUTE_ENCODE_START_ELSE) {
			CHECK(len == 8 && body_len == 0); // no-body compute op
		} else if (tag == RGFX_OP_BLIT_COPY_BUFFER_TO_BUFFER) {
			RgfxBlitOpCopyBufferToBuffer b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.source_buffer_obj_id == 2 && b.destination_buffer_obj_id == 3);
			CHECK(b.source_offset == 16 && b.destination_offset == 32 && b.size == 4096);
		} else if (tag == RGFX_OP_BLIT_FILL_TEXTURE) {
			// Head then `length` fill bytes at offset 80, padded to a multiple of 8.
			CHECK(len == 8 + 88); // 80-byte head + 5 fill bytes padded to 88
			RgfxBlitOpFillTexture head;
			std::memcpy(&head, body, sizeof(head));
			CHECK(head.texture_obj_id == 4 && head.length == 5);
			CHECK(head.region_size_width == 32 && head.region_size_height == 32);
			const uint8_t expect[5] = { 0xde, 0xad, 0xbe, 0xef, 0x01 };
			CHECK(std::memcmp(body + RGFX_OP_BLIT_FILL_TEXTURE_BYTES_OFFSET, expect, 5) == 0);
		} else if (tag == RGFX_OP_BLIT_WAIT_FOR_FENCE) {
			RgfxBlitOpWaitForFence b;
			std::memcpy(&b, body, sizeof(b));
			CHECK(b.fence_obj_id == 9);
		}
		i++;
	}

	// The whole stream was consumed: the tag sequence matched and the framing is
	// balanced (three encoders opened and closed).
	CHECK(i == want_n);
	CHECK(open == 0);
	CHECK(begins == 3 && ends == 3);

	// op_bytes is the stream the walk consumed, 8-aligned.
	CHECK(walk.pos == cb.op_bytes);
	CHECK(cb.op_bytes % 8 == 0);
}

} // namespace

// A forwarding writer hands each whole message, its type and size stamped, to its forward,
// whose seqno it returns; a refusal posts nothing. A reflection the kext hands back before its
// wrapper registers is kept for the registration.
struct ForwardLog {
	std::vector<std::vector<uint8_t>> messages;
	bool refuse = false;
	uint64_t nextSeqno = 41;
	uint32_t answer = kRgfxNoAnswer; // the COMPLETE status the "kext" hands back
};

bool forwardTo(void *context, const void *message, uint32_t size, uint64_t *out_seqno,
               uint32_t *out_status)
{
	ForwardLog *log = static_cast<ForwardLog *>(context);
	if (log->refuse)
		return false;
	const uint8_t *b = static_cast<const uint8_t *>(message);
	log->messages.emplace_back(b, b + size);
	*out_seqno = log->nextSeqno++;
	if (out_status)
		*out_status = log->answer;
	return true;
}

// A video command buffer: the video encoder's begin names its session, and a decode's
// parameter sets follow its body, padded to whole words; HEVC's decode op is laid out the same.
void test_video_command_buffer()
{
	TestRing ring(0x8000);
	ring.setConsumerIdle(1);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxEncoder enc(writer, /*queue*/ 1, /*cb*/ 200);
	enc.beginVideoPass(77);
	RgfxVideoOpDecodeH264 d = {};
	d.bitstream_buffer_obj_id = 5;
	d.bitstream_offset = 0x100;
	d.bitstream_size = 4321;
	d.luma_texture_obj_id = 6;
	d.chroma_texture_obj_id = 7;
	d.output_rows = 1088;
	const uint8_t sets[11] = {0, 0, 0, 1, 0x67, 0x42, 0, 0, 0, 1, 0x68};
	enc.videoDecodeH264(d, sets, sizeof(sets));
	enc.videoDecodeH264(d, nullptr, 0);
	RgfxVideoOpDecodeH265 e = {};
	e.bitstream_buffer_obj_id = 8;
	e.bitstream_size = 512;
	e.luma_texture_obj_id = 9;
	e.chroma_texture_obj_id = 10;
	const uint8_t hevcSets[18] = {0, 0, 0, 1, 0x42, 1, 1, 1, 0, 0, 0, 1, 0x44, 1, 0xc1, 0x72, 0xb4, 0x62};
	enc.videoDecodeH265(e, hevcSets, sizeof(hevcSets));
	enc.endEncoder();
	uint64_t seqno = 0;
	CHECK(enc.finish(&seqno));

	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(ring.next(&h, &bytes));
	CHECK(h.type == RGFX_CMD_COMMAND_BUFFER);
	RgfxCommandBuffer cb = {};
	std::memcpy(&cb, bytes.data(), sizeof(cb));
	CHECK(cb.cb_obj_id == 200 && cb.queue_obj_id == 1);
	const uint8_t *op = bytes.data() + RGFX_COMMAND_BUFFER_OPS_OFFSET;
	const uint8_t *end = op + cb.op_bytes;
	CHECK(end == bytes.data() + bytes.size());

	RgfxOpHeader oh = {};
	std::memcpy(&oh, op, sizeof(oh));
	CHECK(oh.tag == RGFX_OP_ENCODER_BEGIN_VIDEO && oh.len == 16);
	RgfxOpVideoPass pass = {};
	std::memcpy(&pass, op + RGFX_OP_BODY_OFFSET, sizeof(pass));
	CHECK(pass.session_obj_id == 77);
	op += oh.len;

	std::memcpy(&oh, op, sizeof(oh));
	CHECK(oh.tag == RGFX_OP_VIDEO_DECODE_H264);
	CHECK(oh.len == RGFX_OP_BODY_OFFSET + RGFX_VIDEO_DECODE_H264_PARAMETER_SETS_OFFSET + 16);
	RgfxVideoOpDecodeH264 got = {};
	std::memcpy(&got, op + RGFX_OP_BODY_OFFSET, sizeof(got));
	CHECK(got.bitstream_buffer_obj_id == 5 && got.bitstream_offset == 0x100 &&
	      got.bitstream_size == 4321 && got.luma_texture_obj_id == 6 &&
	      got.chroma_texture_obj_id == 7 && got.parameter_sets_size == sizeof(sets) &&
	      got.output_rows == 1088);
	const uint8_t *tail = op + RGFX_OP_BODY_OFFSET + RGFX_VIDEO_DECODE_H264_PARAMETER_SETS_OFFSET;
	CHECK(std::memcmp(tail, sets, sizeof(sets)) == 0);
	for (size_t i = sizeof(sets); i < 16; i++)
		CHECK(tail[i] == 0);
	op += oh.len;

	std::memcpy(&oh, op, sizeof(oh));
	CHECK(oh.tag == RGFX_OP_VIDEO_DECODE_H264);
	CHECK(oh.len == RGFX_OP_BODY_OFFSET + RGFX_VIDEO_DECODE_H264_PARAMETER_SETS_OFFSET);
	std::memcpy(&got, op + RGFX_OP_BODY_OFFSET, sizeof(got));
	CHECK(got.parameter_sets_size == 0);
	op += oh.len;

	// HEVC's op: its own tag, the same layout.
	std::memcpy(&oh, op, sizeof(oh));
	CHECK(oh.tag == RGFX_OP_VIDEO_DECODE_H265);
	CHECK(oh.len == RGFX_OP_BODY_OFFSET + RGFX_VIDEO_DECODE_H265_PARAMETER_SETS_OFFSET + 24);
	RgfxVideoOpDecodeH265 hevc = {};
	std::memcpy(&hevc, op + RGFX_OP_BODY_OFFSET, sizeof(hevc));
	CHECK(hevc.bitstream_buffer_obj_id == 8 && hevc.bitstream_offset == 0 &&
	      hevc.bitstream_size == 512 && hevc.luma_texture_obj_id == 9 &&
	      hevc.chroma_texture_obj_id == 10 && hevc.parameter_sets_size == sizeof(hevcSets) &&
	      hevc.output_rows == 0);
	tail = op + RGFX_OP_BODY_OFFSET + RGFX_VIDEO_DECODE_H265_PARAMETER_SETS_OFFSET;
	CHECK(std::memcmp(tail, hevcSets, sizeof(hevcSets)) == 0);
	for (size_t i = sizeof(hevcSets); i < 24; i++)
		CHECK(tail[i] == 0);
	op += oh.len;

	std::memcpy(&oh, op, sizeof(oh));
	CHECK(oh.tag == RGFX_OP_ENCODER_END && oh.len == 8);
	op += oh.len;
	CHECK(op == end);
}

void test_forwarded_lifecycle()
{
	ForwardLog log;
	RgfxRingWriter writer(&forwardTo, &log, 4096);
	CHECK(writer.dataSize() == 4096);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	RgfxCreated ev = ser.createEvent();
	CHECK(ev.posted);
	CHECK(ev.seqno == 41);
	CHECK(log.messages.size() == 1);
	RgfxObjectCreate m = {};
	std::memcpy(&m, log.messages[0].data(), sizeof(m));
	CHECK(m.header.type == RGFX_CMD_EVENT_CREATE);
	CHECK(m.header.size == sizeof(RgfxObjectCreate));
	CHECK(m.header.seqno == 0); // the kext stamps it
	CHECK(m.obj_id == ev.obj_id);
	log.refuse = true;
	CHECK(!ser.createEvent().posted);
	CHECK(log.messages.size() == 1);
	// Too large for the forward: refused before it is handed over.
	std::vector<uint8_t> big(8192, 0);
	CHECK(!writer.tryPost(RGFX_CMD_LIBRARY_CREATE, big.data(), 8192, nullptr));

	RgfxReplyReader reader;
	CHECK(reader.pump() == 0);
	RgfxPipelineReflection r = {};
	r.header.type = RGFX_EVT_PIPELINE_REFLECTION;
	r.header.size = sizeof(r);
	r.header.seqno = 7;
	r.obj_id = 99;
	reader.deliver(&r, sizeof(r));
	uint64_t got = 0;
	reader.expectReflection(7, [&got](const RgfxPipelineReflection &m) { got = m.obj_id; });
	CHECK(got == 99);
	CHECK(!reader.awaitingReflection(7));
	reader.expectReflection(8, [&got](const RgfxPipelineReflection &m) { got = m.obj_id + 1; });
	CHECK(reader.awaitingReflection(8));
	r.header.seqno = 8;
	reader.deliver(&r, sizeof(r));
	CHECK(got == 100);
	CHECK(!reader.awaitingReflection(8));
}

// The kext's two routes, one call log: each Submit (one message) and each SubmitBatch (its
// messages), in the order made, with the batch's calls apart. `batchTakes` caps how many of a
// batch's messages the "kext" posts (its ring stayed full past them).
struct KextLog {
	struct Call {
		bool batch;
		std::vector<RgfxObjId> objs; // each message's obj_id (the destroy's, the create's)
		std::vector<uint32_t> types;
	};
	// A timer posts from a thread of its own: the calls are made and read under `lock`.
	std::mutex lock;
	std::vector<Call> calls;
	uint32_t batchTakes = UINT32_MAX;
	uint64_t nextSeqno = 1;
	// Run once, at the next batch call, before it takes anything: the writer's held messages are
	// locked for the call then.
	std::function<void()> once;

	static RgfxObjId objOf(const uint8_t *m)
	{
		RgfxObjId obj = 0;
		std::memcpy(&obj, m + sizeof(RgfxMsgHeader), sizeof(obj));
		return obj;
	}
	// Every obj_id the "kext" posted, in ring order.
	std::vector<RgfxObjId> ring()
	{
		std::lock_guard<std::mutex> hold(lock);
		std::vector<RgfxObjId> out;
		for (const Call &c : calls)
			out.insert(out.end(), c.objs.begin(), c.objs.end());
		return out;
	}
};

bool kextSubmit(void *context, const void *message, uint32_t size, uint64_t *out_seqno,
                uint32_t *out_status)
{
	KextLog *log = static_cast<KextLog *>(context);
	std::lock_guard<std::mutex> hold(log->lock);
	RgfxMsgHeader header;
	std::memcpy(&header, message, sizeof(header));
	log->calls.push_back({false, {KextLog::objOf(static_cast<const uint8_t *>(message))},
	                      {header.type}});
	*out_seqno = log->nextSeqno++;
	if (out_status)
		*out_status = kRgfxNoAnswer;
	return true;
}

uint32_t kextSubmitBatch(void *context, const void *messages, uint32_t bytes)
{
	KextLog *log = static_cast<KextLog *>(context);
	if (std::function<void()> once = std::exchange(log->once, nullptr))
		once();
	std::lock_guard<std::mutex> hold(log->lock);
	const uint8_t *m = static_cast<const uint8_t *>(messages);
	KextLog::Call call = {true, {}, {}};
	uint32_t taken = 0;
	for (uint32_t at = 0; at < bytes && taken < log->batchTakes; taken++) {
		RgfxMsgHeader header;
		std::memcpy(&header, m + at, sizeof(header));
		CHECK(header.seqno == 0);
		// Each of a type and size the kext's SubmitBatch takes.
		CHECK(header.size != 0 && header.size == rgfxSubmitBatchSize(header.type));
		call.objs.push_back(KextLog::objOf(m + at));
		call.types.push_back(header.type);
		at += header.size;
		log->nextSeqno++;
	}
	if (taken != 0)
		log->calls.push_back(call);
	return taken;
}

// Destroys held to go together (RgfxRingWriter::postLater): each `after` runs at the destroy, as
// it did once the destroy was posted, and the destroys reach the "kext" in their order before
// anything the process posts after them, at most a batch's bytes a call, or once the timer fires,
// whose arm an RgfxRingLaterArms holds until it ends; a destroy the host orders other contexts by goes at once; a batch the kext takes only in part
// keeps the rest held, first, and the writer posts nothing past them until they go.
void test_held_destroys()
{
	// A timer far past the test, so only the steps below post; the timer's own step has its own.
	constexpr uint64_t kNever = 600ull * 1000000000;
	KextLog log;
	RgfxRingWriter writer(&kextSubmit, &log, 4096);
	CHECK(!writer.batches());
	writer.setBatch(&kextSubmitBatch, 4096, kNever);
	CHECK(writer.batches());
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);

	// Held, `after` run at once, nothing posted yet.
	int afters = 0;
	CHECK(ser.destroy(0x101, [&afters] { afters++; }));
	CHECK(ser.destroy(0x102, [&afters] { afters++; }));
	CHECK(afters == 2);
	CHECK(log.calls.empty());
	// A create goes after them: one batch of both, in order, then the create alone.
	RgfxCreated ev = ser.createEvent();
	CHECK(ev.posted);
	CHECK(log.calls.size() == 2);
	CHECK(log.calls[0].batch && log.calls[0].objs == std::vector<RgfxObjId>({0x101, 0x102}));
	CHECK(log.calls[0].types == std::vector<uint32_t>(2, RGFX_CMD_OBJECT_DESTROY));
	CHECK(!log.calls[1].batch && log.calls[1].objs[0] == ev.obj_id);
	CHECK(writer.flushLater());
	CHECK(log.calls.size() == 2);

	// 170 destroys fill a 4096-byte call (24 bytes each): the 171st sends them, and waits.
	log.calls.clear();
	for (RgfxObjId i = 0; i < 171; i++)
		CHECK(ser.destroy(0x1000 + i));
	CHECK(log.calls.size() == 1 && log.calls[0].batch && log.calls[0].objs.size() == 170);
	CHECK(log.calls[0].objs.front() == 0x1000 && log.calls[0].objs.back() == 0x1000 + 169);
	CHECK(writer.flushLater());
	CHECK(log.calls.size() == 2 && log.calls[1].objs == std::vector<RgfxObjId>({0x1000 + 170}));

	// The timer posts what is held, with no post behind it: kRgfxLaterNs after the first held.
	KextLog timed;
	RgfxRingWriter timedWriter(&kextSubmit, &timed, 4096);
	timedWriter.setBatch(&kextSubmitBatch, 4096);
	RgfxSerializer tser(timedWriter, ids);
	const auto from = std::chrono::steady_clock::now();
	CHECK(tser.destroy(0x2001));
	CHECK(tser.destroy(0x2002));
	for (int i = 0; i < 20000 && timed.ring().size() < 2; i++)
		std::this_thread::sleep_for(std::chrono::microseconds(100));
	CHECK(std::chrono::steady_clock::now() - from >= std::chrono::nanoseconds(kRgfxLaterNs));
	CHECK(timed.ring() == std::vector<RgfxObjId>({0x2001, 0x2002}));

	// Held inside an RgfxRingLaterArms, nothing goes while it lives, however long, and the timer
	// it arms as it ends posts kRgfxLaterNs after that.
	KextLog scoped;
	RgfxRingWriter scopedWriter(&kextSubmit, &scoped, 4096);
	scopedWriter.setBatch(&kextSubmitBatch, 4096);
	RgfxSerializer sser(scopedWriter, ids);
	std::chrono::steady_clock::time_point ended;
	{
		RgfxRingLaterArms arms;
		CHECK(sser.destroy(0x2101));
		CHECK(sser.destroy(0x2102));
		std::this_thread::sleep_for(std::chrono::nanoseconds(5 * kRgfxLaterNs));
		CHECK(scoped.ring().empty());
		ended = std::chrono::steady_clock::now();
	}
	for (int i = 0; i < 20000 && scoped.ring().size() < 2; i++)
		std::this_thread::sleep_for(std::chrono::microseconds(100));
	CHECK(std::chrono::steady_clock::now() - ended >= std::chrono::nanoseconds(kRgfxLaterNs));
	CHECK(scoped.ring() == std::vector<RgfxObjId>({0x2101, 0x2102}));

	// At once: the held first, then the destroy alone, its `after` run.
	log.calls.clear();
	CHECK(ser.destroy(0x3001));
	afters = 0;
	CHECK(ser.destroy(0x3002, [&afters] { afters++; }, /*atOnce*/ true));
	CHECK(afters == 1);
	CHECK(log.calls.size() == 2 && log.calls[0].batch && log.calls[0].objs[0] == 0x3001);
	CHECK(!log.calls[1].batch && log.calls[1].objs[0] == 0x3002);

	// The kext takes one of three: the other two stay held, first, and the create behind them is
	// refused, posting nothing; once the kext takes them, they go, then what follows.
	log.calls.clear();
	for (RgfxObjId i = 0; i < 3; i++)
		CHECK(ser.destroy(0x4000 + i));
	log.batchTakes = 1;
	CHECK(!ser.createEvent().posted);
	CHECK(log.ring() == std::vector<RgfxObjId>({0x4000}));
	CHECK(!writer.flushLater());
	CHECK(log.ring() == std::vector<RgfxObjId>({0x4000, 0x4001}));
	log.batchTakes = UINT32_MAX;
	RgfxCreated after = ser.createEvent();
	CHECK(after.posted);
	CHECK(log.ring() == std::vector<RgfxObjId>({0x4000, 0x4001, 0x4002, after.obj_id}));

	// A writer that does not batch posts each destroy at once, as before.
	KextLog plain;
	RgfxRingWriter direct(&kextSubmit, &plain, 4096);
	RgfxSerializer pser(direct, ids);
	afters = 0;
	CHECK(pser.destroy(0x5001, [&afters] { afters++; }));
	CHECK(afters == 1 && plain.calls.size() == 1 && !plain.calls[0].batch);

	// The writer's end posts what it holds. Its timer comes 1 ms after the post, and a busy host
	// can keep this thread off a CPU that long, letting the timer post first: a run counts only
	// where the look at the calls comes inside that 1 ms from before the post, so the end alone can
	// have posted the destroy. Runs repeat until one counts. Each run's log outlives its timer's
	// use of it: the timer posts only while the writer is there.
	constexpr auto kTimer = std::chrono::nanoseconds(1000000);
	bool counted = false;
	for (int run = 0; run < 50 && !counted; run++) {
		KextLog ending;
		bool noneBeforeEnd = false;
		std::chrono::steady_clock::duration took{};
		{
			RgfxRingWriter going(&kextSubmit, &ending, 4096);
			going.setBatch(&kextSubmitBatch, 4096, kTimer.count());
			RgfxObjectDestroy d = {};
			d.obj_id = 0x6001;
			const auto began = std::chrono::steady_clock::now();
			CHECK(going.postLater(RGFX_CMD_OBJECT_DESTROY, &d, sizeof(d)));
			noneBeforeEnd = ending.ring().empty();
			took = std::chrono::steady_clock::now() - began;
		}
		CHECK(ending.ring() == std::vector<RgfxObjId>({0x6001}));
		// Its timer, which outlives it, then finds it gone and posts nothing more.
		std::this_thread::sleep_for(std::chrono::milliseconds(3));
		CHECK(ending.ring().size() == 1);
		if (took < kTimer) {
			CHECK(noneBeforeEnd);
			counted = true;
		}
	}
	CHECK(counted);
}

// A destroy's give-back (its `after`) waits for the command buffers committed before the destroy
// was asked for (RgfxRetireGate): at once where none is in flight; held while one is, whatever
// the buffers committed after it do, and run in the order of their destroys once the last of
// those is done; a give-back that destroys more, run there, has its own destroy wait for the
// buffers committed by then. Where the destroy is held to go with others, its give-back waits
// the same.
void test_give_backs_wait_for_the_buffers_committed_before()
{
	TestRing ring(0x8000);
	ring.setConsumerIdle(1);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	std::vector<int> ran;
	CHECK(ser.destroy(0x101, [&ran] { ran.push_back(1); }));
	CHECK(ran == std::vector<int>({1}));
	const uint64_t a = ser.commandBufferCommitted();
	CHECK(ser.destroy(0x102, [&ser, &ran] {
		ran.push_back(2);
		// Run at a's completion, after c's commit: it waits for c.
		CHECK(ser.destroy(0x104, [&ran] { ran.push_back(4); }));
	}));
	const uint64_t b = ser.commandBufferCommitted();
	CHECK(ser.destroy(0x103, [&ran] { ran.push_back(3); }));
	CHECK(ran.size() == 1 && ser.heldGiveBacks() == 2);
	// The destroys themselves went at once.
	for (RgfxObjId obj : {0x101ull, 0x102ull, 0x103ull})
		CHECK(expectMessage<RgfxObjectDestroy>(ring, RGFX_CMD_OBJECT_DESTROY).obj_id == obj);
	// b done first: both still wait for a.
	ser.commandBufferDone(b);
	CHECK(ran.size() == 1 && ser.heldGiveBacks() == 2);
	// A buffer committed after the destroys holds neither back.
	const uint64_t c = ser.commandBufferCommitted();
	ser.commandBufferDone(a);
	CHECK(ran == std::vector<int>({1, 2, 3}) && ser.heldGiveBacks() == 1);
	ser.commandBufferDone(c);
	CHECK(ran == std::vector<int>({1, 2, 3, 4}) && ser.heldGiveBacks() == 0);
	CHECK(ser.heldGiveBacksPeak() == 2);
	// A buffer done twice changes nothing, and with none in flight a give-back runs at once.
	ser.commandBufferDone(c);
	CHECK(ser.destroy(0x105, [&ran] { ran.push_back(5); }));
	CHECK(ran.back() == 5 && ser.heldGiveBacks() == 0);

	KextLog log;
	RgfxRingWriter batching(&kextSubmit, &log, 4096);
	batching.setBatch(&kextSubmitBatch, 4096, 600ull * 1000000000);
	RgfxIdSpace batchIds;
	RgfxSerializer bser(batching, batchIds);
	const uint64_t d = bser.commandBufferCommitted();
	int held = 0;
	CHECK(bser.destroy(0x201, [&held] { held++; }));
	CHECK(held == 0 && bser.heldGiveBacks() == 1);
	bser.commandBufferDone(d);
	CHECK(held == 1 && bser.heldGiveBacks() == 0);
}

// The context's end, the serialiser's, runs every give-back still held for a buffer whose
// completion never came; the gate then runs each one released at once, and a completion that
// comes later runs nothing more.
void test_the_context_s_end_runs_the_give_backs_held()
{
	TestRing ring(0x8000);
	ring.setConsumerIdle(1);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxIdSpace ids;
	int ran = 0;
	{
		RgfxSerializer ser(writer, ids);
		(void)ser.commandBufferCommitted(); // its completion never comes
		CHECK(ser.destroy(0x301, [&ran] { ran++; }));
		CHECK(ran == 0 && ser.heldGiveBacks() == 1);
	}
	CHECK(ran == 1);

	RgfxRetireGate gate;
	const uint64_t serial = gate.committed();
	gate.release(gate.stamp(), [&ran] { ran++; });
	CHECK(ran == 1 && gate.held() == 1);
	gate.close();
	CHECK(ran == 2 && gate.held() == 0 && gate.peak() == 1);
	gate.release(gate.stamp(), [&ran] { ran++; });
	CHECK(ran == 3);
	gate.completed(serial);
	CHECK(ran == 3 && gate.held() == 0);
}

// Heap-placed buffers held to go together with the destroys around them
// (RgfxSerializer::createBufferAtOffset): each answered as made and held, in their order with the
// destroys, before anything posted after them; a commit's postHeldCreates() answers whether none
// stays held, destroys left aside; a buffer that needs the held to go first to make room is
// refused at once where they are refused, holding nothing more; a writer that does not batch
// posts each at once, as before.
void test_held_buffers()
{
	constexpr uint64_t kNever = 600ull * 1000000000;
	KextLog log;
	RgfxRingWriter writer(&kextSubmit, &log, 4096);
	writer.setBatch(&kextSubmitBatch, 4096, kNever);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);

	RgfxCreated a = ser.createBufferAtOffset(0x10, 0, 256, 0);
	CHECK(a.posted && a.obj_id != 0 && a.seqno == 0);
	CHECK(ser.destroy(0x101));
	RgfxCreated b = ser.createBufferAtOffset(0x10, 256, 256, 0);
	CHECK(b.posted);
	CHECK(log.calls.empty());
	CHECK(ser.postHeldCreates());
	CHECK(log.calls.size() == 1 && log.calls[0].batch);
	CHECK(log.calls[0].objs == std::vector<RgfxObjId>({a.obj_id, 0x101, b.obj_id}));
	CHECK(log.calls[0].types == std::vector<uint32_t>({RGFX_CMD_BUFFER_AT_OFFSET,
	                                                   RGFX_CMD_OBJECT_DESTROY,
	                                                   RGFX_CMD_BUFFER_AT_OFFSET}));

	// Before what is posted after them.
	log.calls.clear();
	RgfxCreated c = ser.createBufferAtOffset(0x10, 512, 256, 0);
	RgfxCreated ev = ser.createEvent();
	CHECK(c.posted && ev.posted);
	CHECK(log.ring() == std::vector<RgfxObjId>({c.obj_id, ev.obj_id}));

	// The kext takes none: a held buffer leaves the commit's answer false, held destroys alone
	// do not; once it takes them they go, in order.
	log.calls.clear();
	log.batchTakes = 0;
	CHECK(ser.destroy(0x201));
	CHECK(ser.postHeldCreates());
	RgfxCreated d = ser.createBufferAtOffset(0x10, 768, 256, 0);
	CHECK(d.posted);
	CHECK(!ser.postHeldCreates());
	CHECK(!ser.postHeldCreates());
	CHECK(log.calls.empty());
	// So is the next post refused, posting nothing.
	CHECK(!ser.createEvent().posted);
	CHECK(log.calls.empty());
	log.batchTakes = UINT32_MAX;
	CHECK(ser.postHeldCreates());
	CHECK(log.ring() == std::vector<RgfxObjId>({0x201, d.obj_id}));

	// 73 buffers fill a call (56 bytes each): with the kext taking none, the 74th, which needs
	// them gone to make room, is refused, and nothing more is held; once the kext takes them,
	// the 73 go in one call.
	log.calls.clear();
	std::vector<RgfxObjId> held;
	for (int i = 0; i < 73; i++) {
		RgfxCreated m = ser.createBufferAtOffset(0x10, 4096u * i, 64, 0);
		CHECK(m.posted);
		held.push_back(m.obj_id);
	}
	CHECK(log.calls.empty());
	log.batchTakes = 0;
	CHECK(!ser.createBufferAtOffset(0x10, 0, 64, 0).posted);
	log.batchTakes = UINT32_MAX;
	CHECK(writer.flushLater());
	CHECK(log.calls.size() == 1 && log.calls[0].batch && log.calls[0].objs == held);

	// A writer that does not batch posts each at once.
	KextLog plain;
	RgfxRingWriter direct(&kextSubmit, &plain, 4096);
	RgfxSerializer pser(direct, ids);
	RgfxCreated p = pser.createBufferAtOffset(0x10, 0, 64, 0);
	CHECK(p.posted && p.seqno != 0);
	CHECK(plain.calls.size() == 1 && !plain.calls[0].batch &&
	      plain.calls[0].types[0] == RGFX_CMD_BUFFER_AT_OFFSET);
	CHECK(pser.postHeldCreates());
}

// The other creates no one waits for that name no block, a Managed buffer's modified range, and
// a no-copy buffer's escape note (posted only where the device takes it), held as a heap-placed
// buffer is (RgfxSerializer::postUnwaited): each answered as made and held,
// in their order with the destroys, before anything posted after them; a commit's
// postHeldCreates() answers false while any of them stays held, each on its own; a texture over
// an IOSurface, which a present waits for, goes at once, behind the held; a writer that does not
// batch posts each at once, as before.
void test_held_creates()
{
	constexpr uint64_t kNever = 600ull * 1000000000;
	KextLog log;
	RgfxRingWriter writer(&kextSubmit, &log, 4096);
	writer.setBatch(&kextSubmitBatch, 4096, kNever);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	RgfxTextureDescriptor desc = {};
	desc.texture_type = 2;
	desc.pixel_format = 70;
	desc.width = 4;
	desc.height = 4;
	desc.depth = 1;
	desc.mipmap_level_count = 1;
	desc.sample_count = 1;
	desc.array_length = 1;
	RgfxTextureView view = {};
	view.parent_obj_id = 0x77;
	view.pixel_format = 70;

	// Each kind, made one at a time with the kext taking none: answered as made, held, and the
	// commit's answer false until the kext takes it, which posts it alone.
	auto heldAlone = [&](const char *what, uint32_t type, const std::function<RgfxObjId()> &make) {
		log.calls.clear();
		log.batchTakes = 0;
		const RgfxObjId obj = make();
		CHECK(log.calls.empty());
		CHECK(!ser.postHeldCreates());
		log.batchTakes = UINT32_MAX;
		CHECK(ser.postHeldCreates());
		const bool alone = log.calls.size() == 1 && log.calls[0].batch &&
		                   log.calls[0].objs == std::vector<RgfxObjId>({obj}) &&
		                   log.calls[0].types == std::vector<uint32_t>({type});
		if (!alone)
			std::fprintf(stderr, "held alone: %s\n", what);
		CHECK(alone);
	};
	heldAlone("texture", RGFX_CMD_TEXTURE_CREATE, [&] {
		RgfxCreated c = ser.createTexture(desc, RGFX_OBJ_NONE, 0, 256, 1024);
		CHECK(c.posted && c.obj_id != 0 && c.resource_id != 0 && c.gpu_address != 0 && c.seqno == 0);
		return c.obj_id;
	});
	heldAlone("view", RGFX_CMD_TEXTURE_VIEW, [&] {
		RgfxCreated c = ser.createTextureView(view);
		CHECK(c.posted && c.obj_id != 0 && c.resource_id != 0 && c.seqno == 0);
		return c.obj_id;
	});
	heldAlone("managed texture", RGFX_CMD_MANAGED_TEXTURE_CREATE, [&] {
		RgfxCreated c = ser.createManagedTexture(desc, 0x78, 4, 1, 1, 0, 256, 1024);
		CHECK(c.posted && c.obj_id != 0 && c.resource_id != 0 && c.gpu_address != 0 && c.seqno == 0);
		return c.obj_id;
	});
	heldAlone("managed buffer", RGFX_CMD_MANAGED_BUFFER_CREATE, [&] {
		RgfxCreated c = ser.createManagedBuffer(0x79, 256, 0x10, 0);
		CHECK(c.posted && c.obj_id != 0 && c.gpu_address != 0 && c.seqno == 0);
		return c.obj_id;
	});
	heldAlone("modified range", RGFX_CMD_BUFFER_DID_MODIFY_RANGE, [&] {
		CHECK(ser.bufferDidModifyRange(0x7a, 16, 32));
		return RgfxObjId{0x7a};
	});
	log.calls.clear();
	CHECK(!ser.carriesBufferEscaped() &&
	      !ser.bufferEscaped(0x7c, RGFX_BUFFER_ESCAPED_GPU_ADDRESS));
	CHECK(log.calls.empty() && ser.postHeldCreates() && log.calls.empty());
	ser.setCarriesBufferEscaped(true);
	heldAlone("escape note", RGFX_CMD_BUFFER_ESCAPED, [&] {
		CHECK(ser.bufferEscaped(0x7c, RGFX_BUFFER_ESCAPED_RESIDENCY_SET));
		return RgfxObjId{0x7c};
	});

	// Together with destroys and heap-placed buffers, in the order made, in one call.
	log.calls.clear();
	CHECK(ser.destroy(0x101));
	RgfxCreated t = ser.createTexture(desc, 0x10, 4096);
	RgfxCreated a = ser.createBufferAtOffset(0x10, 0, 256, 0);
	RgfxCreated v = ser.createTextureView(view);
	CHECK(ser.bufferDidModifyRange(0x7a, 0, 16));
	RgfxCreated m = ser.createManagedBuffer(a.obj_id, 256, 0x10, 0);
	CHECK(ser.bufferEscaped(0x7d, RGFX_BUFFER_ESCAPED_ARGUMENT_ENCODER));
	CHECK(t.posted && t.gpu_address == 0 && a.posted && v.posted && m.posted);
	CHECK(log.calls.empty());
	CHECK(ser.postHeldCreates());
	CHECK(log.calls.size() == 1 && log.calls[0].batch);
	CHECK(log.calls[0].objs ==
	      std::vector<RgfxObjId>({0x101, t.obj_id, a.obj_id, v.obj_id, 0x7a, m.obj_id, 0x7d}));
	CHECK(log.calls[0].types == std::vector<uint32_t>({RGFX_CMD_OBJECT_DESTROY,
	                                                   RGFX_CMD_TEXTURE_CREATE,
	                                                   RGFX_CMD_BUFFER_AT_OFFSET,
	                                                   RGFX_CMD_TEXTURE_VIEW,
	                                                   RGFX_CMD_BUFFER_DID_MODIFY_RANGE,
	                                                   RGFX_CMD_MANAGED_BUFFER_CREATE,
	                                                   RGFX_CMD_BUFFER_ESCAPED}));

	// A texture over an IOSurface goes at once, behind what is held.
	log.calls.clear();
	RgfxCreated before = ser.createTexture(desc);
	RgfxCreated surface = ser.createTextureFromIOSurface(0x7b, 0, 16, 0x55, 0, desc, 0,
	                                                     RgfxSurfaceFacts{});
	CHECK(before.posted && surface.posted && surface.seqno != 0);
	CHECK(log.calls.size() == 2 && log.calls[0].batch && !log.calls[1].batch);
	CHECK(log.ring() == std::vector<RgfxObjId>({before.obj_id, surface.obj_id}));
	CHECK(log.calls[1].types == std::vector<uint32_t>({RGFX_CMD_TEXTURE_FROM_IOSURFACE}));

	// A writer that does not batch posts each at once.
	KextLog plain;
	RgfxRingWriter direct(&kextSubmit, &plain, 4096);
	RgfxSerializer pser(direct, ids);
	RgfxCreated pt = pser.createTexture(desc);
	RgfxCreated pv = pser.createTextureView(view);
	CHECK(pt.posted && pt.seqno != 0 && pv.posted && pv.seqno != 0);
	CHECK(pser.bufferDidModifyRange(0x7a, 0, 16));
	pser.setCarriesBufferEscaped(true);
	CHECK(pser.bufferEscaped(0x7c, RGFX_BUFFER_ESCAPED_GPU_ADDRESS));
	CHECK(plain.calls.size() == 4 && !plain.calls[0].batch && !plain.calls[1].batch &&
	      !plain.calls[2].batch && !plain.calls[3].batch);
	CHECK(plain.calls[3].types == std::vector<uint32_t>({RGFX_CMD_BUFFER_ESCAPED}));
	CHECK(pser.postHeldCreates());
}

// A commit's postHeldCreates() counts only what the route refused of the held it posted: a
// texture another thread makes while the route's call refuses a held destroy, its create held
// once the call is done, leaves the answer true, as a buffer committed then cannot name it. The
// create stays held behind the destroy, counts for the commits after it, and goes with the
// destroy at the next post the route takes. Run in a process of its own on fair-share mutexes
// (rgfxRunFairShare): the other thread then takes the held messages between the post's hold and
// any later one.
void test_held_since_the_post_is_not_counted()
{
	CHECK(rgfxMutexesFairShare());
	constexpr uint64_t kNever = 600ull * 1000000000;
	KextLog log;
	RgfxRingWriter writer(&kextSubmit, &log, 4096);
	writer.setBatch(&kextSubmitBatch, 4096, kNever);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	RgfxTextureDescriptor desc = {};
	desc.texture_type = 2;
	desc.pixel_format = 70;
	desc.width = 4;
	desc.height = 4;
	desc.depth = 1;
	desc.mipmap_level_count = 1;
	desc.sample_count = 1;
	desc.array_length = 1;

	CHECK(ser.destroy(0x101));
	log.batchTakes = 0;
	RgfxCreated made = {};
	std::thread other;
	log.once = [&] {
		// The other thread's create waits for the held messages, which the call has locked.
		other = std::thread([&] { made = ser.createTexture(desc); });
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	};
	CHECK(ser.postHeldCreates());
	other.join();
	CHECK(made.posted && made.obj_id != 0 && made.seqno == 0);
	CHECK(log.calls.empty());
	CHECK(writer.holdsBut(RGFX_CMD_OBJECT_DESTROY));
	CHECK(!ser.postHeldCreates());
	log.batchTakes = UINT32_MAX;
	CHECK(ser.postHeldCreates());
	CHECK(log.ring() == std::vector<RgfxObjId>({0x101, made.obj_id}));
}

// A synthetic .metallib: its header (the bitcode section at 0x60, 16 bytes), a stand-in for the
// function list, the bitcode, and a tail.
std::vector<uint8_t> syntheticMetallib()
{
	std::vector<uint8_t> b(0x58, 0);
	std::memcpy(b.data(), "MTLB", 4);
	const uint64_t at = 0x60, size = 0x10;
	std::memcpy(b.data() + 0x48, &at, 8);
	std::memcpy(b.data() + 0x50, &size, 8);
	for (const char *part : { "funclist", "BITCODE-BITCODE!", "tail-end" })
		b.insert(b.end(), part, part + std::strlen(part));
	return b;
}

// The keys of a 640 KiB library, its bitcode section 128 KiB at 192 KiB: the content key's first
// piece spans the section and its second is a whole piece, the last; the whole bytes are three
// pieces, hashed on threads at once. Both are the digests computed independently and pinned on
// the host's side too (library.rs, keys_in_pieces_as_the_bundle_does).
void test_library_key_pieces()
{
	std::vector<uint8_t> lib(0xa0000);
	for (size_t i = 0; i < lib.size(); i++)
		lib[i] = static_cast<uint8_t>(i % 251);
	std::memset(lib.data(), 0, 0x58);
	std::memcpy(lib.data(), "MTLB", 4);
	const uint64_t at = 0x30000, size = 0x20000;
	std::memcpy(lib.data() + 0x48, &at, 8);
	std::memcpy(lib.data() + 0x50, &size, 8);
	static const uint8_t content[32] = {
		0x05, 0x34, 0x27, 0xb9, 0xa6, 0xd3, 0xe1, 0xdd, 0xba, 0x82, 0x49, 0x63, 0x1f, 0xd3, 0x9b, 0xa1,
		0xeb, 0x6f, 0x16, 0x1c, 0x98, 0xd2, 0xe2, 0x4f, 0x56, 0xb0, 0x2f, 0x67, 0xe1, 0x74, 0x3c, 0x4a,
	};
	static const uint8_t whole[32] = {
		0xe4, 0x9b, 0x7d, 0x62, 0x71, 0x76, 0xdb, 0x48, 0xe2, 0xd5, 0xd3, 0x5b, 0x03, 0xb2, 0x6b, 0x91,
		0xf0, 0x02, 0x7a, 0x33, 0x0e, 0xc3, 0x25, 0xfd, 0x26, 0x20, 0xc9, 0x86, 0xc3, 0xa0, 0x72, 0x8b,
	};
	uint8_t key[32] = {};
	CHECK(rgfxLibraryKey(lib.data(), lib.size(), key));
	CHECK(std::memcmp(key, content, sizeof(key)) == 0);
	CHECK(rgfxLibraryWholeKey(lib.data(), lib.size(), key));
	CHECK(std::memcmp(key, whole, sizeof(key)) == 0);
	// A byte past the section, in the last piece, changes the content key.
	lib[0x9ffff] ^= 0xff;
	CHECK(rgfxLibraryKey(lib.data(), lib.size(), key));
	CHECK(std::memcmp(key, content, sizeof(key)) != 0);
}

// A library's content key is the SHA-256 of the prefix and the SHA-256 of the bytes around the
// bitcode section, one piece: the digest computed independently (and pinned on the host's side
// too), the same whatever the section holds, and none for bytes that are no .metallib or whose
// section lies outside them.
void test_library_key()
{
	static const uint8_t expect[32] = {
		0xc2, 0x98, 0xaa, 0x26, 0x7e, 0x84, 0x06, 0x18, 0xa9, 0xde, 0x5c, 0xa5, 0x3b, 0x39, 0xf7, 0x12,
		0xeb, 0x30, 0xb7, 0x16, 0xee, 0xc2, 0xbc, 0xe7, 0x1f, 0x1c, 0xd4, 0x72, 0xb6, 0xbd, 0x2b, 0x30,
	};
	std::vector<uint8_t> lib = syntheticMetallib();
	CHECK(lib.size() == 0x78);
	uint8_t key[32] = {};
	CHECK(rgfxLibraryKey(lib.data(), lib.size(), key));
	CHECK(std::memcmp(key, expect, sizeof(key)) == 0);
	// The bitcode is not keyed (its HASH tags are); the bytes around it are.
	lib[0x65] ^= 0xff;
	CHECK(rgfxLibraryKey(lib.data(), lib.size(), key));
	CHECK(std::memcmp(key, expect, sizeof(key)) == 0);
	lib[0x75] ^= 0xff;
	CHECK(rgfxLibraryKey(lib.data(), lib.size(), key));
	CHECK(std::memcmp(key, expect, sizeof(key)) != 0);
	// No key: not a .metallib, a section past the end, too short for a header.
	std::vector<uint8_t> other = syntheticMetallib();
	other[0] = 'X';
	CHECK(!rgfxLibraryKey(other.data(), other.size(), key));
	other = syntheticMetallib();
	const uint64_t past = 0x70;
	std::memcpy(other.data() + 0x50, &past, 8);
	CHECK(!rgfxLibraryKey(other.data(), other.size(), key));
	CHECK(!rgfxLibraryKey(other.data(), 0x40, key));

	// The key of the whole bytes, the bitcode included, is the digest the host's
	// `whole_keys_as_the_bundle_does` pins: Core Image's libraries are kept under it.
	static const uint8_t whole[32] = {
		0x41, 0xa4, 0x8e, 0x77, 0x6f, 0xec, 0xa5, 0x8f, 0x52, 0x67, 0x0e, 0xd8, 0x93, 0x50, 0x17, 0x03,
		0xdb, 0x0f, 0xf3, 0x4e, 0xd8, 0x4e, 0x59, 0x78, 0x35, 0x85, 0x96, 0x16, 0xff, 0x1b, 0xf6, 0xcb,
	};
	lib = syntheticMetallib();
	CHECK(rgfxLibraryWholeKey(lib.data(), lib.size(), key));
	CHECK(std::memcmp(key, whole, sizeof(key)) == 0);
	lib[0x65] ^= 0xff;
	CHECK(rgfxLibraryWholeKey(lib.data(), lib.size(), key));
	CHECK(std::memcmp(key, whole, sizeof(key)) != 0);
	other = syntheticMetallib();
	other[0] = 'X';
	CHECK(!rgfxLibraryWholeKey(other.data(), other.size(), key));
}

// A file's library key is found once for the file's identity: a load of the same identity takes
// the key found before without reading the bytes, and an identity a rewrite changes (its size,
// its times to the nanosecond, its inode) reads them again.
void test_library_key_of_file()
{
	std::vector<uint8_t> lib = syntheticMetallib();
	std::vector<uint8_t> other = syntheticMetallib();
	other[0x75] ^= 0xff;
	uint8_t libKey[32] = {}, otherKey[32] = {}, key[32] = {};
	CHECK(rgfxLibraryKey(lib.data(), lib.size(), libKey));
	CHECK(rgfxLibraryKey(other.data(), other.size(), otherKey));
	CHECK(std::memcmp(libKey, otherKey, 32) != 0);
	const RgfxFileIdentity file = { 0x1001, 0x77, lib.size(), 1700000000, 5, 1700000000, 6 };
	CHECK(rgfxLibraryKeyOfFile(file, lib.data(), lib.size(), key));
	CHECK(std::memcmp(key, libKey, 32) == 0);
	// The same identity: the key found before, whatever the bytes now hold.
	CHECK(rgfxLibraryKeyOfFile(file, other.data(), other.size(), key));
	CHECK(std::memcmp(key, libKey, 32) == 0);
	// Each change a rewrite makes: the bytes keyed again.
	for (int field = 0; field < 4; field++) {
		RgfxFileIdentity rewritten = file;
		if (field == 0)
			rewritten.modified_ns++;
		else if (field == 1)
			rewritten.changed_ns++;
		else if (field == 2)
			rewritten.inode++;
		else
			rewritten.device++;
		CHECK(rgfxLibraryKeyOfFile(rewritten, other.data(), other.size(), key));
		CHECK(std::memcmp(key, otherKey, 32) == 0);
	}
	// Bytes that are no .metallib have no key, and none is kept for them.
	const RgfxFileIdentity notLib = { 0x1002, 0x78, 4, 1, 2, 3, 4 };
	const uint8_t junk[4] = { 1, 2, 3, 4 };
	CHECK(!rgfxLibraryKeyOfFile(notLib, junk, sizeof(junk), key));
	CHECK(rgfxLibraryKeyOfFile(notLib, lib.data(), lib.size(), key));
	CHECK(std::memcmp(key, libKey, 32) == 0);
}

// Keyed library messages through the kext's route: the create carries its key and hands back the
// device's answer; a lookup is one LIBRARY_LOOKUP of its own, answered; a streamed library's
// chunks carry the key and its answer.
void test_keyed_library_messages()
{
	ForwardLog log;
	RgfxRingWriter writer(&forwardTo, &log, 4096);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	std::vector<uint8_t> lib = syntheticMetallib();
	uint8_t key[32] = {};
	CHECK(rgfxLibraryKey(lib.data(), lib.size(), key));

	log.answer = RGFX_STATUS_NOT_FOUND;
	RgfxCreated found = ser.lookupLibrary(key, lib.size());
	CHECK(found.posted && found.status == RGFX_STATUS_NOT_FOUND);
	CHECK(log.messages.size() == 1);
	RgfxLibraryLookup lookup = {};
	CHECK(log.messages[0].size() == sizeof(lookup));
	std::memcpy(&lookup, log.messages[0].data(), sizeof(lookup));
	CHECK(lookup.header.type == RGFX_CMD_LIBRARY_LOOKUP);
	CHECK(lookup.header.size == sizeof(RgfxLibraryLookup));
	CHECK(lookup.obj_id == found.obj_id);
	CHECK(lookup.slice_size == lib.size());
	CHECK(std::memcmp(lookup.key, key, 32) == 0);

	log.answer = RGFX_STATUS_OK;
	RgfxCreated made = ser.createLibrary(lib.data(), lib.size(), key);
	CHECK(made.posted && made.status == RGFX_STATUS_OK);
	RgfxLibraryCreate head = {};
	std::memcpy(&head, log.messages[1].data(), sizeof(head));
	CHECK(head.header.type == RGFX_CMD_LIBRARY_CREATE);
	CHECK(head.flags == RGFX_LIBRARY_FLAG_KEYED);
	CHECK(std::memcmp(head.key, key, 32) == 0);
	CHECK(std::memcmp(log.messages[1].data() + RGFX_LIBRARY_BYTES_OFFSET, lib.data(), lib.size()) ==
	      0);

	log.answer = RGFX_STATUS_REFUSED;
	uint32_t status = 0;
	CHECK(ser.postLibraryChunk(made.obj_id, 0x1000, 8, 8, 0, 0x2000, 1, key, &status));
	CHECK(status == RGFX_STATUS_REFUSED);
	RgfxLibraryFromRegion chunk = {};
	std::memcpy(&chunk, log.messages[2].data(), sizeof(chunk));
	CHECK(chunk.header.type == RGFX_CMD_LIBRARY_FROM_REGION);
	CHECK(chunk.flags == RGFX_LIBRARY_FLAG_KEYED);
	CHECK(std::memcmp(chunk.key, key, 32) == 0);
}

// Dynamic libraries: none is asked for until the host takes RGFX_FEATURE_DYNAMIC_LIBRARIES;
// then DYNAMIC_LIBRARY_CREATE names the library and hands back the host's answer, and a compute
// create that links dynamic libraries ends with their run, each loaded one's install name in the
// tail before it.
void test_dynamic_library_messages()
{
	ForwardLog log;
	RgfxRingWriter writer(&forwardTo, &log, 4096);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	log.answer = RGFX_STATUS_OK;
	RgfxCreated none = ser.createDynamicLibrary(0x21);
	CHECK(!none.posted && none.status == kRgfxNoAnswer && log.messages.empty());

	ser.setCarriesDynamicLibraries(true);
	RgfxCreated made = ser.createDynamicLibrary(0x21);
	CHECK(made.posted && made.status == RGFX_STATUS_OK && made.obj_id != 0);
	CHECK(log.messages.size() == 1);
	RgfxDynamicLibraryCreate create = {};
	CHECK(log.messages[0].size() == sizeof(create));
	std::memcpy(&create, log.messages[0].data(), sizeof(create));
	CHECK(create.header.type == RGFX_CMD_DYNAMIC_LIBRARY_CREATE);
	CHECK(create.header.size == sizeof(RgfxDynamicLibraryCreate));
	CHECK(create.obj_id == made.obj_id && create.library_obj_id == 0x21);
	log.answer = RGFX_STATUS_REFUSED;
	CHECK(ser.createDynamicLibrary(0x22).status == RGFX_STATUS_REFUSED);

	const char *name = "@executable_path/libx.metallib";
	const RgfxPipelineLibraryRef libraries[2] = {{0x31, nullptr}, {0x32, name}};
	RgfxComputePipelineParams cp = {};
	cp.function.library = 10;
	cp.function.entry_point = "k";
	cp.libraries = libraries;
	cp.library_count = 2;
	RgfxCreated c = ser.createComputePipeline(cp);
	CHECK(c.posted);
	CHECK(log.messages.size() == 3);
	const std::vector<uint8_t> &bytes = log.messages[2];
	RgfxComputePipelineCreate cc = {};
	std::memcpy(&cc, bytes.data(), sizeof(cc));
	CHECK(cc.header.type == RGFX_CMD_COMPUTE_PIPELINE_CREATE);
	CHECK(cc.header.size == bytes.size() && bytes.size() % 8 == 0);
	CHECK(cc.library_count == 2);
	CHECK(bytes.size() >= sizeof(cc) + 2 * sizeof(RgfxPipelineLibrary));
	RgfxPipelineLibrary run[2] = {};
	std::memcpy(run, bytes.data() + bytes.size() - sizeof(run), sizeof(run));
	CHECK(run[0].obj_id == 0x31 && run[0].name_len == 0);
	CHECK(run[1].obj_id == 0x32 && run[1].name_len == std::strlen(name));
	CHECK(run[1].name_offset >= sizeof(cc));
	CHECK(run[1].name_offset + run[1].name_len <= bytes.size() - sizeof(run));
	CHECK(std::memcmp(bytes.data() + run[1].name_offset, name, std::strlen(name)) == 0);

	// A pipeline that links none leaves the count 0 and nothing after its tail.
	cp.libraries = nullptr;
	cp.library_count = 0;
	CHECK(ser.createComputePipeline(cp).posted);
	RgfxComputePipelineCreate plain = {};
	std::memcpy(&plain, log.messages[3].data(), sizeof(plain));
	CHECK(plain.library_count == 0);
	CHECK(log.messages[3].size() < bytes.size());
}

// The archive file: its sections round-trip, whatever does not parse whole is no archive, and a
// Metal archive is told by its fat header, every slice inside the file.
void test_archive_file()
{
	RgfxArchiveKey k1(kRgfxArchiveCompute), k2(kRgfxArchiveCompute), k3(kRgfxArchiveVertex);
	k1.text("a");
	k2.text("b");
	k3.text("a");
	RgfxArchiveEntry a = k1.finish(), b = k2.finish(), v = k3.finish();
	CHECK(!(a == b));
	CHECK(!(a == v)); // the kind is part of the key
	RgfxArchiveKey k4(kRgfxArchiveCompute);
	k4.text("a");
	CHECK(k4.finish() == a);
	// A field's length goes in with it: "ab" then "" is not "a" then "b".
	RgfxArchiveKey s1(kRgfxArchiveCompute), s2(kRgfxArchiveCompute);
	s1.text("ab");
	s1.text("");
	s2.text("a");
	s2.text("b");
	CHECK(!(s1.finish() == s2.finish()));

	std::vector<RgfxArchiveSection> in(2);
	in[0].architecture = "vulkan_10de_2c58_00";
	in[0].entries = {a, v};
	in[1].architecture = "applegpu_g15s";
	in[1].entries = {b};
	std::vector<uint8_t> file = rgfxWriteArchiveFile(in);
	std::vector<RgfxArchiveSection> out;
	CHECK(rgfxReadArchiveFile(file.data(), file.size(), out) == RgfxArchiveFileKind::Ours);
	CHECK(out.size() == 2);
	CHECK(out[0].architecture == in[0].architecture && out[0].entries == in[0].entries);
	CHECK(out[1].architecture == in[1].architecture && out[1].entries == in[1].entries);
	CHECK(file.size() == 8 + 4 + (4 + 19 + 4 + 2 * 33) + (4 + 13 + 4 + 33));

	for (size_t cut = 0; cut < file.size(); cut++)
		CHECK(rgfxReadArchiveFile(file.data(), cut, out) == RgfxArchiveFileKind::Invalid);
	std::vector<uint8_t> longer = file;
	longer.push_back(0);
	CHECK(rgfxReadArchiveFile(longer.data(), longer.size(), out) == RgfxArchiveFileKind::Invalid);
	CHECK(out.empty());
	std::vector<uint8_t> none = rgfxWriteArchiveFile({});
	CHECK(rgfxReadArchiveFile(none.data(), none.size(), out) == RgfxArchiveFileKind::Ours);
	CHECK(out.empty());

	// One slice of 4 bytes at 28: a Metal archive; its slice past the end: no archive.
	const uint8_t fat[] = {0xcb, 0xfe, 0xba, 0xbe, 0, 0, 0, 1, 1, 0, 0, 0x17, 0, 0, 0, 0xd,
	                       0,    0,    0,    28,   0, 0, 0, 4, 0, 0, 0, 0,    1, 2, 3, 4};
	CHECK(rgfxReadArchiveFile(fat, sizeof(fat), out) == RgfxArchiveFileKind::Metal);
	CHECK(rgfxReadArchiveFile(fat, sizeof(fat) - 1, out) == RgfxArchiveFileKind::Invalid);
	CHECK(rgfxReadArchiveFile(fat, 8, out) == RgfxArchiveFileKind::Invalid);
	// The same of the fat magic 0xcafebabe, as Metal reads a framework's binary.metallib; of no
	// slices, no archive.
	uint8_t cafe[sizeof(fat)];
	std::memcpy(cafe, fat, sizeof(fat));
	cafe[0] = 0xca;
	CHECK(rgfxReadArchiveFile(cafe, sizeof(cafe), out) == RgfxArchiveFileKind::Metal);
	CHECK(rgfxReadArchiveFile(cafe, sizeof(cafe) - 1, out) == RgfxArchiveFileKind::Invalid);
	cafe[7] = 0;
	CHECK(rgfxReadArchiveFile(cafe, sizeof(cafe), out) == RgfxArchiveFileKind::Invalid);
	const uint8_t other[] = {0xca, 0xfe, 0xba, 0xbf, 0, 0, 0, 1};
	CHECK(rgfxReadArchiveFile(other, sizeof(other), out) == RgfxArchiveFileKind::Invalid);
	// An AIR slice of junk beside another GPU's slice: an archive of no keys, as both x86 Macs
	// read one, for no slice is this device's (oracle_bundle archive.slices: a junk AIR slice is
	// "an invalid format" only beside the device's own slice).
	uint8_t pair[8 + 2 * 20 + 8] = {0xcb, 0xfe, 0xba, 0xbe, 0, 0, 0, 2};
	const uint8_t records[2][20] = {
	    {1, 0, 0, 0x17, 0, 0, 0, 0xc, 0, 0, 0, 48, 0, 0, 0, 4, 0, 0, 0, 4},
	    {1, 0, 0, 0x13, 0, 0, 1, 0x42, 0, 0, 0, 52, 0, 0, 0, 4, 0, 0, 0, 4}};
	std::memcpy(pair + 8, records, sizeof(records));
	std::memset(pair + 48, 0x5a, 8);
	CHECK(rgfxReadArchiveFile(pair, sizeof(pair), out) == RgfxArchiveFileKind::Metal);
	CHECK(out.empty());
	// A whole .metallib: an archive of none of this device's slices, as on both x86 Macs; cut
	// short, or of another magic, no archive.
	uint8_t metallib[0x20] = {'M', 'T', 'L', 'B', 1, 0x80, 2, 0, 9, 0, 0, 0x81};
	metallib[0x10] = sizeof(metallib);
	CHECK(rgfxReadArchiveFile(metallib, sizeof(metallib), out) == RgfxArchiveFileKind::Metal);
	CHECK(out.empty());
	CHECK(rgfxReadArchiveFile(metallib, sizeof(metallib) - 1, out) == RgfxArchiveFileKind::Invalid);
	metallib[3] = 'C';
	CHECK(rgfxReadArchiveFile(metallib, sizeof(metallib), out) == RgfxArchiveFileKind::Invalid);
}

// An Automatic heap's placements, in the order Metal's took them (measured: M3 Max, macOS 27,
// a Private heap of 1000000 bytes, 1015808 once rounded): each at the lowest offset at its
// alignment where all its bytes are free, the sizes summed as used, the free bytes joined as
// they come back, and the longest free run from an aligned offset as the most available.
void test_heap_ranges()
{
	RgfxHeapRanges r(1015808);
	uint64_t at = 1;
	CHECK(r.used() == 0 && r.maxAvailable(0) == 1015808 && r.maxAvailable(65536) == 1015808);
	uint64_t b1 = 0, b2 = 0, b3 = 0, t1 = 0;
	CHECK(r.take(100, 256, &b1) && b1 == 0);
	CHECK(r.used() == 100 && r.maxAvailable(1) == 1015708 && r.maxAvailable(256) == 1015552);
	CHECK(r.maxAvailable(4096) == 1011712 && r.maxAvailable(65536) == 950272);
	CHECK(r.take(100, 256, &b2) && b2 == 256);
	CHECK(r.take(70000, 256, &b3) && b3 == 512);
	CHECK(r.used() == 70200 && r.maxAvailable(1) == 945296 && r.maxAvailable(256) == 945152);
	CHECK(r.maxAvailable(4096) == 942080 && r.maxAvailable(65536) == 884736);
	CHECK(r.take(264192, 128, &t1) && t1 == 70528);
	CHECK(r.used() == 334392 && r.maxAvailable(1) == 681088 && r.maxAvailable(256) == 680960);
	r.give(b1, 100);
	CHECK(r.used() == 334292 && r.maxAvailable(1) == 681088);
	uint64_t b4 = 0, b5 = 0, b7 = 0;
	CHECK(r.take(50, 256, &b4) && b4 == 0);
	r.give(b2, 100); // made aliasable
	CHECK(r.take(60, 256, &b5) && b5 == 256);
	CHECK(r.used() == 334302 && r.maxAvailable(256) == 680960);
	CHECK(!r.take(680961, 256, &at) && at == 1);
	CHECK(r.take(680960, 256, &b7) && b7 == 334848);
	CHECK(r.used() == 1015262 && r.maxAvailable(1) == 206 && r.maxAvailable(256) == 0);
	CHECK(!r.take(16, 256, &at));
	// First fit, not best fit: with holes of 5000 and 1000 bytes in that order, 800 bytes go in
	// the first (Metal's order of a 5000, 256, 1000 and 256-byte buffer, the first and third
	// freed).
	RgfxHeapRanges f(1 << 20);
	uint64_t a = 0, b = 0, c = 0, d = 0, e = 0, g = 0, h = 0, i = 0;
	CHECK(f.take(5000, 256, &a) && f.take(256, 256, &b) && f.take(1000, 256, &c) &&
	      f.take(256, 256, &d));
	CHECK(a == 0 && b == 5120 && c == 5376 && d == 6400);
	f.give(a, 5000);
	f.give(c, 1000);
	CHECK(f.take(800, 256, &e) && e == 0);
	CHECK(f.take(3000, 256, &g) && g == 1024);
	CHECK(f.take(3000, 256, &h) && h == 6656);
	uint64_t k = 0;
	CHECK(f.take(100, 256, &k) && k == 4096);
	// B, E and F back: [0, 4096) and [4196, 6400) are free, neither holds 5500.
	f.give(b, 256);
	f.give(e, 800);
	f.give(g, 3000);
	CHECK(f.take(5500, 256, &i) && i == 9728);
	CHECK(f.maxAvailable(1) == (1 << 20) - 15228 && f.take(16, 256, &at) && at == 0);
	// Nothing for a size of 0, nor at an alignment that is not a power of two.
	CHECK(!f.take(0, 256, &at) && !f.take(16, 3, &at) && f.maxAvailable(3) == 0);
	RgfxHeapRanges none(0);
	CHECK(!none.take(1, 1, &at) && none.maxAvailable(1) == 0);
}

// What a fake device answers a heap texture size query, and what it was asked.
struct SizeAsker {
	RgfxHeapTextureSizeAndAlign asked = {};
	uint32_t status = RGFX_STATUS_OK;
	bool answers = true;
	int calls = 0;

	static bool ask(void *context, const void *message, uint32_t size, void *reply,
	                uint32_t capacity, uint32_t *reply_size)
	{
		SizeAsker *self = static_cast<SizeAsker *>(context);
		self->calls++;
		CHECK(size == sizeof(RgfxHeapTextureSizeAndAlign));
		std::memcpy(&self->asked, message, sizeof(self->asked));
		if (!self->answers)
			return false;
		RgfxSizeAndAlign answer = {};
		answer.header = {RGFX_EVT_SIZE_AND_ALIGN, sizeof(answer), 9};
		answer.status = self->status;
		answer.size = self->status == RGFX_STATUS_OK ? 264192 : 0;
		answer.align = self->status == RGFX_STATUS_OK ? 128 : 0;
		CHECK(capacity >= sizeof(answer));
		std::memcpy(reply, &answer, sizeof(answer));
		*reply_size = sizeof(answer);
		return true;
	}
};

// A heap texture size query asks the device with the texture's shape under a new id of the
// context and posts nothing on the ring; it answers the device's size and alignment, and
// nothing where no asker is set, nothing came back, or the device refused the shape.
void test_heap_texture_size_query()
{
	TestRing ring(0x4000);
	ring.setConsumerIdle(1);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxIdSpace ids;
	ids.setBases(2ull << RGFX_CONTEXT_OBJ_ID_SHIFT, RgfxIdSpace::kDefaultAddressBase);
	RgfxSerializer ser(writer, ids);
	RgfxTextureDescriptor desc = {};
	desc.pixel_format = 70;
	desc.texture_type = 2;
	desc.width = 256;
	desc.height = 256;
	desc.depth = 1;
	desc.mipmap_level_count = 1;
	desc.array_length = 1;
	desc.sample_count = 1;
	uint64_t size = 1, align = 1;
	CHECK(!ser.heapTextureSizeAndAlign(desc, &size, &align) && size == 1 && align == 1);

	SizeAsker asker;
	ser.setAsker(&SizeAsker::ask, &asker);
	CHECK(ser.heapTextureSizeAndAlign(desc, &size, &align));
	CHECK(size == 264192 && align == 128 && asker.calls == 1);
	CHECK(asker.asked.header.type == RGFX_CMD_HEAP_TEXTURE_SIZE_AND_ALIGN);
	CHECK(asker.asked.header.size == sizeof(RgfxHeapTextureSizeAndAlign));
	CHECK(asker.asked.obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT == 2);
	CHECK(std::memcmp(&asker.asked.desc, &desc, sizeof(desc)) == 0);
	const uint64_t first = asker.asked.obj_id;
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(!ring.next(&h, &bytes));

	// An equal shape is answered as the device answered it, with no second ask.
	asker.status = RGFX_STATUS_REFUSED;
	RgfxTextureDescriptor same = desc;
	size = align = 1;
	CHECK(ser.heapTextureSizeAndAlign(same, &size, &align));
	CHECK(size == 264192 && align == 128 && asker.calls == 1);

	// Another shape is asked; a refusal answers nothing and is asked again.
	RgfxTextureDescriptor other = desc;
	other.width = 512;
	size = align = 1;
	CHECK(!ser.heapTextureSizeAndAlign(other, &size, &align) && size == 1 && align == 1);
	CHECK(asker.asked.obj_id != first && asker.calls == 2);
	CHECK(asker.asked.desc.width == 512);
	asker.answers = false;
	CHECK(!ser.heapTextureSizeAndAlign(other, &size, &align) && asker.calls == 3);
	asker.answers = true;
	asker.status = RGFX_STATUS_OK;
	CHECK(ser.heapTextureSizeAndAlign(other, &size, &align) && asker.calls == 4);
	CHECK(ser.heapTextureSizeAndAlign(other, &size, &align) && asker.calls == 4);
	other.usage = 1;
	CHECK(ser.heapTextureSizeAndAlign(other, &size, &align) && asker.calls == 5);
}

// The device's answers to the rate map messages, and what it was last asked.
struct RateMapAsker {
	uint32_t calls = 0;
	uint32_t status = RGFX_STATUS_OK;
	std::vector<uint8_t> asked;
	static bool ask(void *context, const void *message, uint32_t size, void *reply,
	                uint32_t capacity, uint32_t *reply_size)
	{
		auto *self = static_cast<RateMapAsker *>(context);
		self->calls++;
		const auto *bytes = static_cast<const uint8_t *>(message);
		self->asked.assign(bytes, bytes + size);
		RgfxMsgHeader h = {};
		std::memcpy(&h, message, sizeof(h));
		if (h.type == RGFX_CMD_RATE_MAP_QUERY) {
			RgfxRateMapAnswer answer = {};
			answer.header = {RGFX_EVT_RATE_MAP_ANSWER, sizeof(answer), 11};
			answer.status = self->status;
			answer.values[0] = 0x3f800000; // 1.0f
			answer.values[1] = 0x40000000; // 2.0f
			answer.values[2] = 7;
			CHECK(capacity >= sizeof(answer));
			std::memcpy(reply, &answer, sizeof(answer));
			*reply_size = sizeof(answer);
			return true;
		}
		RgfxRateMapState answer = {};
		answer.header = {RGFX_EVT_RATE_MAP_STATE, sizeof(answer), 10};
		answer.status = self->status;
		answer.screen_width = 256;
		answer.screen_height = 128;
		answer.granularity_width = answer.granularity_height = 32;
		answer.granularity_depth = 1;
		answer.layer_count = 1;
		answer.mutability = 2;
		answer.parameter_size = 24624;
		answer.parameter_align = 4;
		CHECK(capacity >= sizeof(answer));
		std::memcpy(reply, &answer, sizeof(answer));
		*reply_size = sizeof(answer);
		return true;
	}
};

// A rate map's create, reset and queries ask the device and post nothing on the ring: the create
// under a new id of the context, carrying the descriptor's screen size, its mutability where set
// and each layer's record with its samples after the layers, each the bits of its float; a reset
// the same under the map's id. Nothing is asked where the host makes no maps, or where the
// message is past half the control ring, which the kext takes no lifecycle message past.
void test_rate_map_messages()
{
	TestRing ring(0x8000);
	ring.setConsumerIdle(1);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxIdSpace ids;
	ids.setBases(3ull << RGFX_CONTEXT_OBJ_ID_SHIFT, RgfxIdSpace::kDefaultAddressBase);
	RgfxSerializer ser(writer, ids);
	RateMapAsker asker;
	ser.setAsker(&RateMapAsker::ask, &asker);
	RgfxRateMapShape shape;
	shape.screen_width = 256;
	shape.screen_height = 128;
	shape.mutability_set = true;
	shape.mutability = 1;
	shape.layers.push_back({3, 2, {1.0f, 0.25f, 1.0f, 1.0f, 0.5f}});
	RgfxRateMapState state = {};
	RgfxCreated none = ser.createRateMap(shape, &state);
	CHECK(!none.posted && none.status == kRgfxNoAnswer && asker.calls == 0);

	ser.setCarriesRateMaps(true);
	RgfxCreated made = ser.createRateMap(shape, &state);
	CHECK(made.posted && made.status == RGFX_STATUS_OK && asker.calls == 1);
	CHECK(made.obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT == 3 && made.seqno == 10);
	CHECK(state.screen_width == 256 && state.parameter_size == 24624 && state.layer_count == 1);
	const size_t head = sizeof(RgfxRateMapCreate), layer = sizeof(RgfxRateMapLayer);
	CHECK(asker.asked.size() == ((head + layer + 5 * 4 + 7) & ~size_t(7)));
	RgfxRateMapCreate create = {};
	std::memcpy(&create, asker.asked.data(), sizeof(create));
	CHECK(create.header.type == RGFX_CMD_RATE_MAP_CREATE && create.header.size == asker.asked.size());
	CHECK(create.obj_id == made.obj_id && create.screen_width == 256 && create.screen_height == 128);
	CHECK(create.screen_depth == 0 && create.flags == RGFX_RATE_MAP_MUTABILITY_SET);
	CHECK(create.mutability == 1 && create.layer_count == 1 && create.layers_offset == head);
	RgfxRateMapLayer l = {};
	std::memcpy(&l, asker.asked.data() + head, sizeof(l));
	CHECK(l.horizontal_count == 3 && l.vertical_count == 2 && l.samples_offset == head + layer);
	float samples[5] = {};
	std::memcpy(samples, asker.asked.data() + l.samples_offset, sizeof(samples));
	CHECK(samples[0] == 1.0f && samples[1] == 0.25f && samples[2] == 1.0f && samples[3] == 1.0f &&
	      samples[4] == 0.5f);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(!ring.next(&h, &bytes));

	// A reset names the map; a refusal answers its status.
	shape.mutability_set = false;
	asker.status = RGFX_STATUS_REFUSED;
	CHECK(ser.resetRateMap(made.obj_id, shape, &state) == RGFX_STATUS_REFUSED);
	std::memcpy(&create, asker.asked.data(), sizeof(create));
	CHECK(create.header.type == RGFX_CMD_RATE_MAP_RESET && create.obj_id == made.obj_id);
	CHECK(create.flags == 0 && create.mutability == 0);

	// A query carries its kind, its argument and its coordinate's bits, and answers the values
	// where the device answers OK, zeros where it does not.
	uint64_t values[3] = {9, 9, 9};
	asker.status = RGFX_STATUS_OK;
	CHECK(ser.queryRateMap(made.obj_id, RGFX_RATE_MAP_QUERY_SCREEN_TO_PHYSICAL, 1, RGFX_OBJ_NONE,
	                       10.5f, 3.25f, values) == RGFX_STATUS_OK);
	CHECK(values[0] == 0x3f800000 && values[1] == 0x40000000 && values[2] == 7);
	RgfxRateMapQuery q = {};
	CHECK(asker.asked.size() == sizeof(q));
	std::memcpy(&q, asker.asked.data(), sizeof(q));
	float x = 0, y = 0;
	std::memcpy(&x, &q.x_bits, 4);
	std::memcpy(&y, &q.y_bits, 4);
	CHECK(q.header.type == RGFX_CMD_RATE_MAP_QUERY && q.obj_id == made.obj_id && q.at == 1);
	CHECK(q.query == RGFX_RATE_MAP_QUERY_SCREEN_TO_PHYSICAL && x == 10.5f && y == 3.25f);
	asker.status = RGFX_STATUS_REFUSED;
	CHECK(ser.queryRateMap(made.obj_id, RGFX_RATE_MAP_QUERY_COPY_PARAMETER_DATA, 4, 0x77, 0, 0,
	                       values) == RGFX_STATUS_REFUSED);
	CHECK(values[0] == 0 && values[1] == 0 && values[2] == 0);
	std::memcpy(&q, asker.asked.data(), sizeof(q));
	CHECK(q.buffer_obj_id == 0x77 && q.at == 4);

	// A descriptor past half the control ring is asked nothing.
	const uint32_t calls = asker.calls;
	shape.layers[0].horizontal_count = 4096;
	shape.layers[0].vertical_count = 2;
	shape.layers[0].samples.assign(4098, 1.0f);
	CHECK(ser.createRateMap(shape, &state).status == kRgfxNoAnswer && asker.calls == calls);
	CHECK(!ring.next(&h, &bytes));
}

// The device's answers to the function pointer creates, and what it was last asked.
struct FunctionPointerAsker {
	uint32_t calls = 0;
	uint32_t status = RGFX_STATUS_OK;
	std::vector<uint8_t> asked;
	static bool ask(void *context, const void *message, uint32_t size, void *reply,
	                uint32_t capacity, uint32_t *reply_size)
	{
		auto *self = static_cast<FunctionPointerAsker *>(context);
		self->calls++;
		const auto *bytes = static_cast<const uint8_t *>(message);
		self->asked.assign(bytes, bytes + size);
		RgfxFunctionPointerState answer = {};
		answer.header = {RGFX_EVT_FUNCTION_POINTER_STATE, sizeof(answer), 12};
		answer.status = self->status;
		answer.gpu_resource_id = 3;
		answer.allocated_size = 4096;
		answer.resource_options = 528;
		CHECK(capacity >= sizeof(answer));
		std::memcpy(reply, &answer, sizeof(answer));
		*reply_size = sizeof(answer);
		return true;
	}
};

// A function handle's and a visible function table's creates ask the device and post nothing on
// the ring: the handle under a new id of the context, naming its pipeline and carrying its
// function as a linked function's record, with the strings and constants after the head; the
// table naming its pipeline and its entry count. A table's set is posted on the ring unanswered,
// in runs of entries each message holds at most half the control ring of. Nothing is asked or
// posted where the host makes no function pointers.
void test_function_pointer_messages()
{
	TestRing ring(0x8000);
	ring.setConsumerIdle(1);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxIdSpace ids;
	ids.setBases(3ull << RGFX_CONTEXT_OBJ_ID_SHIFT, RgfxIdSpace::kDefaultAddressBase);
	RgfxSerializer ser(writer, ids);
	FunctionPointerAsker asker;
	ser.setAsker(&FunctionPointerAsker::ask, &asker);
	RgfxFunctionConstant constant = {};
	constant.index = 7;
	constant.data_type = 29;
	constant.value[0] = 0x5a;
	RgfxLinkedFunctionRef fn = {};
	fn.function.library = 0x55;
	fn.function.entry_point = "fp_add1";
	fn.function.constants = &constant;
	fn.function.constant_count = 1;
	fn.list = RGFX_LINKED_FUNCTIONS;
	fn.options = 4;
	fn.specialized_name = "add1_7";
	RgfxFunctionPointerState state = {};
	const RgfxObjId handles[3] = {0x31, 0, 0x32};
	CHECK(ser.createFunctionHandle(0x44, fn, &state).status == kRgfxNoAnswer);
	CHECK(ser.createFunctionTable(0x44, 3, &state).status == kRgfxNoAnswer);
	CHECK(!ser.setFunctionTable(0x66, 0, handles, 3) && asker.calls == 0);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(!ring.next(&h, &bytes));

	ser.setCarriesFunctionPointers(true);
	RgfxCreated handle = ser.createFunctionHandle(0x44, fn, &state);
	CHECK(handle.posted && handle.status == RGFX_STATUS_OK && asker.calls == 1);
	CHECK(handle.obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT == 3 && handle.seqno == 12);
	const size_t head = sizeof(RgfxFunctionHandleCreate);
	CHECK(asker.asked.size() == ((head + 7 + sizeof(constant) + 6 + 7) & ~size_t(7)));
	RgfxFunctionHandleCreate create = {};
	std::memcpy(&create, asker.asked.data(), sizeof(create));
	CHECK(create.header.type == RGFX_CMD_FUNCTION_HANDLE_CREATE &&
	      create.header.size == asker.asked.size());
	CHECK(create.obj_id == handle.obj_id && create.pipeline_obj_id == 0x44);
	const RgfxLinkedFunction &f = create.function;
	CHECK(f.library_obj_id == 0x55 && f.list == RGFX_LINKED_FUNCTIONS && f.options == 4);
	CHECK(f.name_offset == head && f.name_len == 7 &&
	      std::memcmp(asker.asked.data() + f.name_offset, "fp_add1", 7) == 0);
	RgfxFunctionConstant c = {};
	std::memcpy(&c, asker.asked.data() + f.constants_offset, sizeof(c));
	CHECK(f.constant_count == 1 && c.index == 7 && c.data_type == 29 && c.value[0] == 0x5a);
	CHECK(f.group_len == 0 && f.specialized_name_len == 6 &&
	      std::memcmp(asker.asked.data() + f.specialized_name_offset, "add1_7", 6) == 0);

	// Not found (a function the pipeline does not link) answers its status, no resource id.
	asker.status = RGFX_STATUS_NOT_FOUND;
	RgfxCreated none = ser.createFunctionHandle(0x44, fn, &state);
	CHECK(none.posted && none.status == RGFX_STATUS_NOT_FOUND && none.resource_id == 0);

	asker.status = RGFX_STATUS_OK;
	RgfxCreated table = ser.createFunctionTable(0x44, 3, &state);
	CHECK(table.posted && table.status == RGFX_STATUS_OK && table.resource_id == 3);
	CHECK(state.gpu_resource_id == 3 && state.allocated_size == 4096 &&
	      state.resource_options == 528);
	RgfxFunctionTableCreate tc = {};
	CHECK(asker.asked.size() == sizeof(tc));
	std::memcpy(&tc, asker.asked.data(), sizeof(tc));
	CHECK(tc.header.type == RGFX_CMD_FUNCTION_TABLE_CREATE && tc.header.size == sizeof(tc));
	CHECK(tc.obj_id == table.obj_id && tc.obj_id != handle.obj_id && tc.pipeline_obj_id == 0x44);
	CHECK(tc.function_count == 3);
	CHECK(!ring.next(&h, &bytes));

	// A set goes on the ring, unanswered, its handles after the head.
	const uint32_t calls = asker.calls;
	CHECK(ser.setFunctionTable(table.obj_id, 1, handles, 3) && asker.calls == calls);
	CHECK(ring.next(&h, &bytes) && h.type == RGFX_CMD_FUNCTION_TABLE_SET);
	RgfxFunctionTableSet set = {};
	CHECK(bytes.size() == sizeof(set) + 3 * 8);
	std::memcpy(&set, bytes.data(), sizeof(set));
	CHECK(set.obj_id == table.obj_id && set.first == 1 && set.count == 3);
	RgfxObjId sent[3] = {};
	std::memcpy(sent, bytes.data() + sizeof(set), sizeof(sent));
	CHECK(sent[0] == 0x31 && sent[1] == 0 && sent[2] == 0x32);

	// Entries past what one message holds go in runs, each from where the last ended.
	const uint64_t run = (writer.dataSize() / 2 - sizeof(set)) / 8;
	std::vector<RgfxObjId> many(run + 5);
	for (size_t i = 0; i < many.size(); i++)
		many[i] = 0x100 + i;
	CHECK(ser.setFunctionTable(table.obj_id, 2, many.data(), many.size()));
	CHECK(ring.next(&h, &bytes) && h.type == RGFX_CMD_FUNCTION_TABLE_SET);
	std::memcpy(&set, bytes.data(), sizeof(set));
	CHECK(set.first == 2 && set.count == run && bytes.size() == sizeof(set) + run * 8);
	CHECK(ring.next(&h, &bytes) && h.type == RGFX_CMD_FUNCTION_TABLE_SET);
	std::memcpy(&set, bytes.data(), sizeof(set));
	CHECK(set.first == 2 + run && set.count == 5);
	RgfxObjId last = 0;
	std::memcpy(&last, bytes.data() + sizeof(set) + 4 * 8, 8);
	CHECK(last == 0x100 + run + 4);
	CHECK(!ring.next(&h, &bytes));
}

// A no-copy buffer's block and the app's memory (RgfxMemoryShadow): the block starts as the
// memory; each sync moves the bytes one side changed since the two last agreed, byte by byte,
// so a CPU write and a GPU write beside it in one word both stand; memory mapped read-only is
// never written, and each sync in writes the whole of it into the block, the GPU's writes
// dropped; the encoder's notes sync each shadow once and keep the ones written; the commits,
// bytes and write-backs are counted by kind.
void test_memory_shadow()
{
	CHECK(!RgfxMemoryShadow::anyLive());
	const size_t n = 0x3003; // a tail past the last whole word and chunk
	std::vector<uint8_t> app(n), block(n, 0xee);
	for (size_t i = 0; i < n; i++)
		app[i] = static_cast<uint8_t>(i * 7);
	{
		auto s = std::make_shared<RgfxMemoryShadow>(app.data(), block.data(), n, true);
		CHECK(s->ok() && s->length() == n && s->writable() && RgfxMemoryShadow::anyLive());
		CHECK(block == app);

		// The CPU writes; the GPU writes elsewhere, before the commit's sync.
		app[0x10] = 0xa1;
		std::memset(&app[0x1000], 0xa2, 8); // a whole word
		app[0x3002] = 0xa3;                 // the tail
		block[0x2000] = 0xb1;               // a GPU write not yet written back
		block[0x400] = 0xb2;
		app[0x401] = 0xa4; // beside it, in the same word
		s->syncIn();
		CHECK(block[0x10] == 0xa1 && block[0x1000] == 0xa2 && block[0x1007] == 0xa2);
		CHECK(block[0x3002] == 0xa3 && block[0x401] == 0xa4);
		CHECK(block[0x2000] == 0xb1 && block[0x400] == 0xb2); // the GPU's stay
		CHECK(app[0x2000] == static_cast<uint8_t>(0x2000 * 7));

		// The completion writes the GPU's back, and keeps a CPU write made meanwhile.
		app[0x402] = 0xa5;
		s->writeBack();
		CHECK(app[0x2000] == 0xb1 && app[0x400] == 0xb2 && app[0x401] == 0xa4);
		CHECK(app[0x402] == 0xa5 && block[0x402] == static_cast<uint8_t>(0x402 * 7));
		s->syncIn();
		CHECK(block == app);
		s->writeBack();
		CHECK(block == app);
	}
	CHECK(!RgfxMemoryShadow::anyLive());

	// Read-only memory: written whole at each sync in, never written back.
	std::vector<uint8_t> ro(64, 3), roBlock(64, 0);
	auto r = std::make_shared<RgfxMemoryShadow>(ro.data(), roBlock.data(), ro.size(), false);
	CHECK(r->ok() && !r->writable() && roBlock == ro);
	roBlock[5] = 9; // the GPU's
	ro[6] = 4;      // another mapping's write of the memory
	r->writeBack();
	CHECK(ro[5] == 3 && roBlock[5] == 9 && roBlock[6] == 3);
	r->syncIn();
	CHECK(roBlock == ro); // the GPU's dropped, the memory's taken

	// The notes: each shadow synced once, the ones any use writes kept.
	std::vector<uint8_t> a2(16, 1), b2(16, 2), aBlock(16), bBlock(16);
	auto a = std::make_shared<RgfxMemoryShadow>(a2.data(), aBlock.data(), 16, true);
	auto b = std::make_shared<RgfxMemoryShadow>(b2.data(), bBlock.data(), 16, true);
	TestRing ring(0x1000);
	RgfxRingWriter writer(ring.base(), ring.dataSize, 1, TestRing::doorbell, nullptr);
	RgfxEncoder enc(writer, 1, 2);
	enc.noteShadowUse(a, false);
	enc.noteShadowUse(a, false); // the same as the last: one note
	enc.noteShadowUse(b, false);
	enc.noteShadowUse(a, true);
	enc.noteShadowUse(r, false);
	std::vector<RgfxShadowUse> uses = enc.takeShadowUses();
	CHECK(uses.size() == 4 && enc.takeShadowUses().empty());
	a2[0] = 5;
	b2[0] = 6;
	const RgfxShadowCounts before = rgfxShadowCounts();
	std::vector<std::shared_ptr<RgfxMemoryShadow>> written = rgfxShadowsSyncIn(uses);
	CHECK(aBlock[0] == 5 && bBlock[0] == 6);
	CHECK(written.size() == 1 && written[0] == a);
	CHECK(rgfxShadowsSyncIn({}).empty());
	const unsigned w = RgfxShadowCounts::kWritable, rd = RgfxShadowCounts::kReadOnly;
	const RgfxShadowCounts synced = rgfxShadowCounts();
	CHECK(synced.commits[w] == before.commits[w] + 1 && synced.bytes[w] == before.bytes[w] + 32);
	CHECK(synced.commits[rd] == before.commits[rd] + 1 &&
	      synced.bytes[rd] == before.bytes[rd] + 64);
	CHECK(synced.writeBacks == before.writeBacks);
	// The completion writes back the one written, counted, and lets go of them; a read-only
	// shadow among them writes nothing back and counts nothing.
	aBlock[1] = 7;
	written.push_back(r);
	rgfxShadowsWriteBack(written);
	CHECK(a2[1] == 7 && written.empty());
	const RgfxShadowCounts back = rgfxShadowCounts();
	CHECK(back.writeBacks == synced.writeBacks + 1 && back.bytes[w] == synced.bytes[w] + 16);
	CHECK(back.commits[w] == synced.commits[w] && back.bytes[rd] == synced.bytes[rd]);
	rgfxShadowsWriteBack(written);
	CHECK(rgfxShadowCounts().writeBacks == back.writeBacks);
}

// What a fake device answers an argument layout query, and what it was asked.
struct LayoutAsker {
	std::vector<uint8_t> asked;
	uint32_t status = RGFX_STATUS_OK;
	uint32_t answerType = RGFX_EVT_COMPLETE;
	bool answers = true;
	int calls = 0;

	static bool ask(void *context, const void *message, uint32_t size, void *reply,
	                uint32_t capacity, uint32_t *reply_size)
	{
		LayoutAsker *self = static_cast<LayoutAsker *>(context);
		self->calls++;
		const uint8_t *b = static_cast<const uint8_t *>(message);
		self->asked.assign(b, b + size);
		if (!self->answers)
			return false;
		RgfxComplete answer = {};
		answer.header = {self->answerType, sizeof(answer), 5};
		answer.status = self->status;
		CHECK(capacity >= sizeof(answer));
		std::memcpy(reply, &answer, sizeof(answer));
		*reply_size = sizeof(answer);
		return true;
	}

	RgfxArgumentLayoutQuery query() const
	{
		RgfxArgumentLayoutQuery q = {};
		CHECK(asked.size() >= sizeof(q));
		std::memcpy(&q, asked.data(), sizeof(q));
		return q;
	}
};

// An argument layout query asks the device under a new id of the context, with the
// descriptors, or the function's name and constants, after its head, and the reply block it
// names; it posts nothing on the ring, and answers the COMPLETE's status, or no answer where no
// asker is set, nothing came back, or the answer is not a COMPLETE.
void test_argument_layout_queries()
{
	TestRing ring(0x4000);
	ring.setConsumerIdle(1);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxIdSpace ids;
	ids.setBases(3ull << RGFX_CONTEXT_OBJ_ID_SHIFT, RgfxIdSpace::kDefaultAddressBase);
	RgfxSerializer ser(writer, ids);
	RgfxArgumentDescriptor d[2] = {};
	d[0].data_type = 58; // MTLDataTypeTexture
	d[0].index = 0;
	d[0].texture_type = 2;
	d[1].data_type = 60; // MTLDataTypePointer
	d[1].index = 1;
	d[1].array_length = 4;
	CHECK(ser.argumentLayoutOfDescriptors(d, 2, 0x10000, 4096) == kRgfxNoAnswer);

	LayoutAsker asker;
	ser.setAsker(&LayoutAsker::ask, &asker);
	CHECK(ser.argumentLayoutOfDescriptors(d, 2, 0x10000, 4096) == RGFX_STATUS_OK);
	RgfxArgumentLayoutQuery q = asker.query();
	CHECK(q.header.type == RGFX_CMD_ARGUMENT_LAYOUT);
	CHECK(q.header.size == asker.asked.size() && q.header.size % 8 == 0);
	CHECK(q.header.size == sizeof(q) + sizeof(d));
	CHECK(q.obj_id >> RGFX_CONTEXT_OBJ_ID_SHIFT == 3);
	CHECK(q.source == RGFX_ARGUMENT_SOURCE_DESCRIPTORS && q.count == 2);
	CHECK(q.reply_offset == 0x10000 && q.reply_length == 4096);
	CHECK(q.descriptors_offset == sizeof(q));
	CHECK(std::memcmp(asker.asked.data() + q.descriptors_offset, d, sizeof(d)) == 0);
	const uint64_t first = q.obj_id;
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(!ring.next(&h, &bytes));

	RgfxFunctionConstant c = {};
	c.index = 2;
	c.data_type = 53; // MTLDataTypeBool
	c.value[0] = 1;
	asker.status = RGFX_STATUS_TOO_SMALL;
	CHECK(ser.argumentLayoutOfFunction(0x77, "kern", &c, 1, 3, 0x20000, 8192) ==
	      RGFX_STATUS_TOO_SMALL);
	q = asker.query();
	CHECK(q.obj_id != first && q.source == RGFX_ARGUMENT_SOURCE_FUNCTION);
	CHECK(q.library_obj_id == 0x77 && q.buffer_index == 3);
	CHECK(q.reply_offset == 0x20000 && q.reply_length == 8192);
	CHECK(q.name_offset == sizeof(q) && q.name_len == 4);
	CHECK(std::memcmp(asker.asked.data() + q.name_offset, "kern", 4) == 0);
	CHECK(q.constants_offset == sizeof(q) + 4 && q.constant_count == 1);
	CHECK(std::memcmp(asker.asked.data() + q.constants_offset, &c, sizeof(c)) == 0);
	CHECK(q.header.size == ((sizeof(q) + 4 + sizeof(c) + 7) & ~size_t(7)));

	asker.status = RGFX_STATUS_REFUSED;
	CHECK(ser.argumentLayoutOfDescriptors(d, 2, 0x10000, 4096) == RGFX_STATUS_REFUSED);
	asker.answerType = RGFX_EVT_SIZE_AND_ALIGN;
	CHECK(ser.argumentLayoutOfDescriptors(d, 2, 0x10000, 4096) == kRgfxNoAnswer);
	asker.answers = false;
	CHECK(ser.argumentLayoutOfDescriptors(d, 2, 0x10000, 4096) == kRgfxNoAnswer);
	// Larger than the control ring takes: not asked.
	std::vector<RgfxArgumentDescriptor> many(ring.dataSize / sizeof(RgfxArgumentDescriptor));
	const int calls = asker.calls;
	CHECK(ser.argumentLayoutOfDescriptors(many.data(), static_cast<uint32_t>(many.size()), 0x10000,
	                                      4096) == kRgfxNoAnswer);
	CHECK(asker.calls == calls);
}

// A layout answer parses into its tree, and an index lands where Metal's encoder puts it: in
// the last member whose indices hold it, at its element's offset, inside a struct element as
// that element's layout says, with a nested argument buffer's layout; nowhere for an index no
// member holds. An answer that is not whole parses to nothing.
void test_argument_layout_parse()
{
	auto probe = probeReply().bytes();
	auto layout = RgfxArgLayout::parse(probe.data(), probe.size());
	CHECK(layout != nullptr);
	CHECK(layout->encodedLength == 64 && layout->alignment == 16);
	CHECK(layout->members.size() == 5 && layout->idSpan() == 5);
	RgfxArgPlace at = {};
	CHECK(layout->place(0, &at) && at.kind == RGFX_ARGUMENT_TEXTURE && at.offset == 0);
	CHECK(layout->place(1, &at) && at.kind == RGFX_ARGUMENT_BUFFER && at.offset == 8);
	CHECK(at.nested && at.nested->encodedLength == 16 && at.nested->members.size() == 2);
	RgfxArgPlace inner = {};
	CHECK(at.nested->place(1, &inner) && inner.kind == RGFX_ARGUMENT_SAMPLER && inner.offset == 8);
	CHECK(layout->place(3, &at) && at.kind == RGFX_ARGUMENT_CONSTANT && at.offset == 32 &&
	      at.size == 16);
	CHECK(layout->place(4, &at) && at.offset == 48 && !at.nested);
	CHECK(!layout->place(5, &at));

	// An array, a later member that claims one of its indices, and an array of structs.
	ArgReply r;
	r.layout(256, 16)
	    .member(0, RGFX_ARGUMENT_BUFFER, 0, 8, 4, 8)
	    .member(2, RGFX_ARGUMENT_TEXTURE, 40, 8)
	    .member(10, RGFX_ARGUMENT_STRUCT, 64, 32, 2, 32, 3, 1);
	r.layout(32, 16)
	    .member(0, RGFX_ARGUMENT_BUFFER, 0, 8)
	    .member(1, RGFX_ARGUMENT_CONSTANT, 16, 16)
	    .member(2, RGFX_ARGUMENT_TEXTURE, 8, 8);
	auto b = r.bytes();
	layout = RgfxArgLayout::parse(b.data(), b.size());
	CHECK(layout != nullptr && layout->idSpan() == 16);
	CHECK(layout->place(1, &at) && at.kind == RGFX_ARGUMENT_BUFFER && at.offset == 8);
	CHECK(layout->place(2, &at) && at.kind == RGFX_ARGUMENT_TEXTURE && at.offset == 40);
	CHECK(layout->place(3, &at) && at.kind == RGFX_ARGUMENT_BUFFER && at.offset == 24);
	CHECK(!layout->place(4, &at));
	CHECK(layout->place(10, &at) && at.kind == RGFX_ARGUMENT_BUFFER && at.offset == 64);
	CHECK(layout->place(12, &at) && at.kind == RGFX_ARGUMENT_TEXTURE && at.offset == 72);
	CHECK(layout->place(14, &at) && at.kind == RGFX_ARGUMENT_CONSTANT && at.offset == 112);
	CHECK(!layout->place(16, &at));

	// Not whole: no layouts, a run past the members, a layout index past the layouts, a layout
	// that holds itself, a reply shorter than its records or than it says it needs.
	CHECK(RgfxArgLayout::parse(nullptr, 0) == nullptr);
	ArgReply none;
	none.layouts.clear();
	auto nb = none.bytes();
	CHECK(RgfxArgLayout::parse(nb.data(), nb.size()) == nullptr);
	ArgReply run = probeReply();
	run.layouts[1].member_count = 3;
	auto rb = run.bytes();
	CHECK(RgfxArgLayout::parse(rb.data(), rb.size()) == nullptr);
	ArgReply past = probeReply();
	past.members[1].layout = 2;
	auto pb = past.bytes();
	CHECK(RgfxArgLayout::parse(pb.data(), pb.size()) == nullptr);
	ArgReply self = probeReply();
	self.members[1].layout = 0;
	auto sb = self.bytes();
	CHECK(RgfxArgLayout::parse(sb.data(), sb.size()) == nullptr);
	CHECK(RgfxArgLayout::parse(probe.data(), probe.size() - 1) == nullptr);
	RgfxArgumentLayoutReply head = {};
	std::memcpy(&head, probe.data(), sizeof(head));
	head.needed = probe.size() + 8;
	std::memcpy(probe.data(), &head, sizeof(head));
	CHECK(RgfxArgLayout::parse(probe.data(), probe.size()) == nullptr);
}

// A region's pages blocks (RgfxRegion::acquirePages): none without a source; with one, the
// block it gives, its address, bytes and writability as asked, and no pointer, as the process
// never maps one; a block the region does not hold goes back through the block source, and is
// none.
void test_region_pages_blocks()
{
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x10000, 0);
	RgfxRegion region(mem.data(), mem.size());
	int marker = 0;
	CHECK(!region.hasPagesBlocks() && !region.acquirePages(&marker, 4, true).ok);
	struct Source {
		const void *address = nullptr;
		uint64_t length = 0;
		bool writable = false;
		uint64_t offset = 0;
		std::vector<uint64_t> released;
		static bool block(void *c, const void *address, uint64_t length, uint32_t how,
		                  uint64_t *offset, uint64_t *size, uint64_t *gpuAddress)
		{
			const bool writable = (how & RgfxRegion::kPagesWritable) != 0;
			Source *s = static_cast<Source *>(c);
			s->address = address;
			s->length = length;
			s->writable = writable;
			*offset = s->offset;
			*size = 0x1000;
			*gpuAddress = 0;
			return true;
		}
		static bool alloc(void *, uint64_t, uint64_t, uint64_t *) { return false; }
		static void release(void *c, uint64_t offset)
		{
			static_cast<Source *>(c)->released.push_back(offset);
		}
	} src;
	region.setBlockSource(&Source::alloc, &Source::release, &src);
	region.setPagesBlockSource(&Source::block, &src);
	CHECK(region.hasPagesBlocks());
	src.offset = RGFX_ALLOCATIONS_OFFSET + 0x4000;
	RgfxRegion::Block b = region.acquirePages(&marker, 0x4000, true);
	CHECK(b.ok && b.offset == src.offset && b.size == 0x1000 && b.ptr == nullptr &&
	      b.gpuAddress == 0);
	CHECK(src.address == &marker && src.length == 0x4000 && src.writable);
	CHECK(src.released.empty());
	src.offset = mem.size() - 0x800;
	CHECK(!region.acquirePages(&marker, 0x4000, false).ok && !src.writable);
	CHECK(src.released.size() == 1 && src.released[0] == mem.size() - 0x800);
}

// A pages block asked with a join (RgfxRegion::acquirePages) is given again for the same bytes,
// writability and join while it is joinable, the kext asked once: it stays joinable until
// leavePages came for each time it was given, and goes back to the block source at the release
// for the last of them, whichever order those come in. Other bytes, another writability, another
// join and kPagesOwn each take a block of their own; once a block is joinable no longer, the same
// bytes take a new block though the old one is still held.
void test_region_pages_joins()
{
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion region(mem.data(), mem.size());
	struct Source {
		int asked = 0;
		uint64_t next = RGFX_ALLOCATIONS_OFFSET + 0x10000;
		std::vector<uint64_t> released;
		static bool block(void *c, const void *, uint64_t, uint32_t, uint64_t *offset,
		                  uint64_t *size, uint64_t *gpuAddress)
		{
			Source *s = static_cast<Source *>(c);
			s->asked++;
			*offset = s->next;
			*size = 0x4000;
			// The device's address of each block's view, as IMPORT_ADDRESSES gives one.
			*gpuAddress = (1ull << 43) + (s->next << 4);
			s->next += 0x4000;
			return true;
		}
		static bool alloc(void *, uint64_t, uint64_t, uint64_t *) { return false; }
		static void release(void *c, uint64_t offset)
		{
			static_cast<Source *>(c)->released.push_back(offset);
		}
	} src;
	region.setBlockSource(&Source::alloc, &Source::release, &src);
	region.setPagesBlockSource(&Source::block, &src);
	// Written, so every page has its object, as the kext's wire of a block faults them in.
	static uint8_t bytes[0x8000];
	std::memset(bytes, 0x5a, sizeof(bytes));
	const uint32_t join = 1;

	RgfxRegion::Block a = region.acquirePages(bytes, 0x4000, true, join);
	RgfxRegion::Block b = region.acquirePages(bytes, 0x4000, true, join);
	CHECK(a.ok && b.ok && a.offset == b.offset && a.size == b.size && b.ptr == nullptr);
	CHECK(src.asked == 1);
	// The address the device gave the block's view, which the join shares.
	CHECK(a.gpuAddress != 0 && b.gpuAddress == a.gpuAddress);
	// Each of these takes a block of its own.
	const RgfxRegion::Block others[] = {
	    region.acquirePages(bytes, 0x3000, true, join),
	    region.acquirePages(bytes + 0x1000, 0x4000, true, join),
	    region.acquirePages(bytes, 0x4000, false, join),
	    region.acquirePages(bytes, 0x4000, true, join + 1),
	    region.acquirePages(bytes, 0x4000, true, RgfxRegion::kPagesOwn),
	    region.acquirePages(bytes, 0x4000, true, RgfxRegion::kPagesOwn),
	};
	CHECK(src.asked == 7);
	for (const RgfxRegion::Block &o : others)
		CHECK(o.ok && o.offset != a.offset);
	CHECK(others[4].offset != others[5].offset);
	for (const RgfxRegion::Block &o : others)
		CHECK(o.gpuAddress != 0 && o.gpuAddress != a.gpuAddress);

	// The first holder's release before its leave: held still, and joinable.
	region.release(a.offset);
	CHECK(src.released.empty());
	RgfxRegion::Block c = region.acquirePages(bytes, 0x4000, true, join);
	CHECK(c.ok && c.offset == a.offset && src.asked == 7);
	region.leavePages(a.offset);
	region.leavePages(a.offset);
	CHECK(region.acquirePages(bytes, 0x4000, true, join).offset == a.offset && src.asked == 7);
	// The last leave: the same bytes take a new block while the old one is held.
	region.leavePages(a.offset);
	region.leavePages(a.offset);
	RgfxRegion::Block d = region.acquirePages(bytes, 0x4000, true, join);
	CHECK(d.ok && d.offset != a.offset && src.asked == 8);
	region.leavePages(a.offset); // past the last leave: nothing
	region.release(a.offset);
	region.release(a.offset);
	CHECK(src.released.empty());
	region.release(a.offset); // the last of the four
	CHECK(src.released.size() == 1 && src.released[0] == a.offset);
	region.release(d.offset);
	CHECK(src.released.size() == 2 && src.released[1] == d.offset);
	// A block of kPagesOwn goes back at its one release.
	region.leavePages(others[4].offset);
	region.release(others[4].offset);
	CHECK(src.released.size() == 3 && src.released[2] == others[4].offset);
}

// A joinable pages block is joined only while the memory at its bytes is the memory it was given
// for (RgfxRegion::acquirePages, rgfxPagesIdentity): new memory mapped at the same address and
// length, with other contents, takes a block of its own while the first is still held and
// joinable, and the first is joinable no longer, its leaves and releases still its own; the same
// memory mapped again at the same address and offset joins; the same memory at another offset,
// and memory with no pages yet, do not.
void test_region_pages_join_over_remapped_memory()
{
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion region(mem.data(), mem.size());
	struct Source {
		int asked = 0;
		uint64_t next = RGFX_ALLOCATIONS_OFFSET + 0x10000;
		std::vector<uint64_t> released;
		static bool block(void *c, const void *, uint64_t, uint32_t, uint64_t *offset,
		                  uint64_t *size, uint64_t *gpuAddress)
		{
			Source *s = static_cast<Source *>(c);
			s->asked++;
			*offset = s->next;
			*size = 0x4000;
			*gpuAddress = 0;
			s->next += 0x4000;
			return true;
		}
		static bool alloc(void *, uint64_t, uint64_t, uint64_t *) { return false; }
		static void release(void *c, uint64_t offset)
		{
			static_cast<Source *>(c)->released.push_back(offset);
		}
	} src;
	region.setBlockSource(&Source::alloc, &Source::release, &src);
	region.setPagesBlockSource(&Source::block, &src);
	const uint32_t join = 1;
	const mach_vm_size_t length = 0x8000;
	mach_vm_address_t at = 0;
	CHECK(mach_vm_allocate(mach_task_self(), &at, length, VM_FLAGS_ANYWHERE) == KERN_SUCCESS);
	uint8_t *bytes = reinterpret_cast<uint8_t *>(at);
	std::memset(bytes, 0xaa, length);
	std::vector<uint64_t> first;
	CHECK(rgfxPagesIdentity(bytes, 0x4000, &first) && first.size() == 3);

	RgfxRegion::Block a = region.acquirePages(bytes, 0x4000, true, join);
	RgfxRegion::Block b = region.acquirePages(bytes, 0x4000, true, join);
	CHECK(a.ok && b.ok && a.offset == b.offset && src.asked == 1);
	CHECK(region.pagesRejoinsRefused() == 0);

	// Other memory at the same address, other contents: a block of its own, the first still
	// held twice and never asked to join again. The kext wires a pages block's memory (its
	// IOMemoryDescriptor's prepare), which keeps the first memory's object alive while the block
	// is held, so the new memory's object cannot reuse its identity (measured: a macOS 26.7.2
	// guest's no-copy buffer kept it in 200 of 200 overwrites, where with no buffer the new
	// object took the old one's identity in 198 of 200). The fake source wires nothing: a memory
	// entry of the first memory, held to the end, keeps it alive as the kext does (measured on
	// macOS 27.0.1: the identity was the old one's in 0 of 5000 overwrites with the entry held,
	// and in 717 of 5000 without it).
	mach_port_t firstMemory = MACH_PORT_NULL;
	memory_object_size_t firstSize = length;
	CHECK(mach_make_memory_entry_64(mach_task_self(), &firstSize, at, VM_PROT_READ | VM_PROT_WRITE,
	                                &firstMemory, MACH_PORT_NULL) == KERN_SUCCESS);
	mach_vm_address_t again = at;
	CHECK(mach_vm_allocate(mach_task_self(), &again, length, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE) ==
	      KERN_SUCCESS);
	CHECK(again == at);
	std::memset(bytes, 0x55, length);
	std::vector<uint64_t> second;
	CHECK(rgfxPagesIdentity(bytes, 0x4000, &second) && second != first);
	RgfxRegion::Block c = region.acquirePages(bytes, 0x4000, true, join);
	CHECK(c.ok && c.offset != a.offset && src.asked == 2);
	CHECK(region.pagesRejoinsRefused() == 1);
	RgfxRegion::Block d = region.acquirePages(bytes, 0x4000, true, join);
	CHECK(d.ok && d.offset == c.offset && src.asked == 2 && region.pagesRejoinsRefused() == 1);
	// The first block's leaves and releases are its own: it goes back at its last release, and
	// the second stays joinable.
	region.leavePages(a.offset);
	region.leavePages(a.offset);
	region.release(a.offset);
	CHECK(src.released.empty());
	region.release(a.offset);
	CHECK(src.released.size() == 1 && src.released[0] == a.offset);
	CHECK(region.acquirePages(bytes, 0x4000, true, join).offset == c.offset && src.asked == 2);

	// The same memory mapped again at the same address and offset: joined.
	mach_vm_size_t entrySize = length;
	mach_port_t entry = MACH_PORT_NULL;
	CHECK(mach_make_memory_entry_64(mach_task_self(), &entrySize, at,
	                                VM_PROT_READ | VM_PROT_WRITE, &entry,
	                                MACH_PORT_NULL) == KERN_SUCCESS);
	mach_vm_address_t mapped = at;
	CHECK(mach_vm_map(mach_task_self(), &mapped, length, 0, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE,
	                  entry, 0, false, VM_PROT_READ | VM_PROT_WRITE, VM_PROT_READ | VM_PROT_WRITE,
	                  VM_INHERIT_NONE) == KERN_SUCCESS);
	CHECK(mapped == at && bytes[0] == 0x55);
	CHECK(region.acquirePages(bytes, 0x4000, true, join).offset == c.offset && src.asked == 2);
	CHECK(region.pagesRejoinsRefused() == 1);
	// The same memory at another offset: other memory for these bytes.
	CHECK(mach_vm_map(mach_task_self(), &mapped, length - 0x1000, 0,
	                  VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE, entry, 0x1000, false,
	                  VM_PROT_READ | VM_PROT_WRITE, VM_PROT_READ | VM_PROT_WRITE,
	                  VM_INHERIT_NONE) == KERN_SUCCESS);
	RgfxRegion::Block e = region.acquirePages(bytes, 0x4000, true, join);
	CHECK(e.ok && e.offset != c.offset && src.asked == 3 && region.pagesRejoinsRefused() == 2);
	mach_port_deallocate(mach_task_self(), entry);

	// Memory with no pages yet: no identity, so neither joined nor joinable.
	CHECK(mach_vm_allocate(mach_task_self(), &again, length, VM_FLAGS_FIXED | VM_FLAGS_OVERWRITE) ==
	      KERN_SUCCESS);
	std::vector<uint64_t> none;
	CHECK(!rgfxPagesIdentity(bytes, 0x4000, &none));
	RgfxRegion::Block f = region.acquirePages(bytes, 0x4000, true, join);
	RgfxRegion::Block g = region.acquirePages(bytes, 0x4000, true, join);
	CHECK(f.ok && g.ok && f.offset != e.offset && g.offset != f.offset && src.asked == 5);
	CHECK(region.pagesRejoinsRefused() == 3);
	mach_port_deallocate(mach_task_self(), firstMemory);
	mach_vm_deallocate(mach_task_self(), at, length);
}

// A file on a read-only volume for the unchanging pages tests (RgfxFilePages.hpp): /bin/bash, on
// the system volume, which every macOS since 10.15 mounts read-only, opened read-only, and the
// `length` bytes at `offset` in it as read.
struct ReadOnlyFile {
	int fd = -1;
	std::vector<uint8_t> bytes;
	ReadOnlyFile(off_t offset, size_t length) : bytes(length)
	{
		fd = open("/bin/bash", O_RDONLY | O_CLOEXEC);
		CHECK(fd >= 0);
		struct statfs volume = {};
		CHECK(fstatfs(fd, &volume) == 0 && (volume.f_flags & MNT_RDONLY) != 0);
		CHECK(pread(fd, bytes.data(), length, offset) == static_cast<ssize_t>(length));
	}
	~ReadOnlyFile() { close(fd); }
};

// The object and the offset the kernel finds the page at `address` in, below any private object
// over it (mach_vm_page_info).
static vm_page_info_basic_data_t pageAt(const void *address)
{
	vm_page_info_basic_data_t info = {};
	mach_msg_type_number_t count = VM_PAGE_INFO_BASIC_COUNT;
	CHECK(mach_vm_page_info(mach_task_self(), reinterpret_cast<uintptr_t>(address),
	                        VM_PAGE_INFO_BASIC, reinterpret_cast<vm_page_info_t>(&info),
	                        &count) == KERN_SUCCESS);
	return info;
}

// The region at `address` (mach_vm_region_recurse's short info).
static vm_region_submap_short_info_data_64_t regionAt(const void *address)
{
	mach_vm_address_t base = reinterpret_cast<uintptr_t>(address);
	mach_vm_size_t size = 0;
	natural_t depth = 0;
	vm_region_submap_short_info_data_64_t info = {};
	mach_msg_type_number_t count = VM_REGION_SUBMAP_SHORT_INFO_COUNT_64;
	CHECK(mach_vm_region_recurse(mach_task_self(), &base, &size, &depth,
	                             reinterpret_cast<vm_region_recurse_info_t>(&info),
	                             &count) == KERN_SUCCESS);
	return info;
}

// The region's shadow depth at `address`.
static unsigned shadowDepthAt(const void *address) { return regionAt(address).shadow_depth; }

// The classification of read-only memory at a no-copy buffer's make (rgfxFilePagesOf): a shared
// mapping of a file on a read-only volume, from a descriptor opened read-only, has no write in its
// maximum protection and the file's object at depth 0 (kRgfxFileMaxNoWrite). A private mapping
// has write in its maximum protection, as mmap makes it, and is neither (writable max
// protection); with its maximum protection taken down to read (vm_protect's set_maximum) it is
// copy-on-write, the delayed copy object xnu makes for it at depth 1, every page the file's
// (kRgfxFileCopyOnWrite). A private mapping written before its maximum protection lost write has
// a shadow over the file's pages, and is neither (shadowed), while its pages past the written one
// are copy-on-write still; so are a shared writable mapping (writable mapping), a file on a
// writable volume, and a range over two regions. Anonymous memory whose maximum protection has
// no write is unchanging where the kernel answers it copy-on-write with a copy pending on its own
// object (kRgfxAnonymousCopyOnWrite), as vm_copy leaves its destination, which keeps its bytes
// when the source is written after; it is neither, why not a file, with write in its maximum
// protection, held by the process alone (SM_PRIVATE) where the kext takes no sole holder's
// memory, mapped at a second address too, or copy-on-write over an object below its own. Where
// the kext takes it, anonymous memory the process's mapping alone holds, a count of 1, is
// unchanging (kRgfxAnonymousPrivate); a memory entry of it, or a second mapping, which takes no
// write the maximum lacks, holds it too, a count of 2, and it is neither until the holder goes; and
// a process that asked for its regions' footprint is answered a count of 1 for every page with an
// id of none of its objects, so neither.
void test_file_pages_classify()
{
	const uint64_t page = vm_page_size;
	const off_t at = static_cast<off_t>(8 * page);
	const size_t mapped = 8 * page;
	const uint64_t into = 0x180, length = 3 * page + 0x40;
	ReadOnlyFile file(at + static_cast<off_t>(into), length);
	auto lowerMax = [](void *address, size_t bytes) {
		return mach_vm_protect(mach_task_self(), reinterpret_cast<uintptr_t>(address), bytes,
		                       true, VM_PROT_READ) == KERN_SUCCESS;
	};

	// Shared, read-only: max-no-write, depth 0.
	uint8_t *shared = static_cast<uint8_t *>(mmap(nullptr, mapped, PROT_READ, MAP_SHARED, file.fd, at));
	CHECK(shared != MAP_FAILED && shadowDepthAt(shared) == 0);
	RgfxFilePages a = rgfxFilePagesOf(shared + into, length);
	CHECK(a.clause == kRgfxFileMaxNoWrite && a.why == kRgfxNeitherNone);
	CHECK(std::memcmp(shared + into, file.bytes.data(), length) == 0);

	// Private, as mmap makes it: write in its maximum protection.
	uint8_t *priv = static_cast<uint8_t *>(mmap(nullptr, mapped, PROT_READ, MAP_PRIVATE, file.fd, at));
	CHECK(priv != MAP_FAILED && shadowDepthAt(priv) == 1);
	RgfxFilePages b = rgfxFilePagesOf(priv + into, length);
	CHECK(b.clause == kRgfxFileNeither && b.why == kRgfxNeitherWritableMaxProtection);
	// Its maximum protection taken down: copy-on-write, every page the file's below the delayed
	// copy, and the process can no longer make it writable.
	CHECK(lowerMax(priv, mapped));
	RgfxFilePages c = rgfxFilePagesOf(priv + into, length);
	CHECK(c.clause == kRgfxFileCopyOnWrite && c.why == kRgfxNeitherNone);
	CHECK(mprotect(priv, mapped, PROT_READ | PROT_WRITE) != 0);
	CHECK(pageAt(priv + into).object_id == pageAt(shared + into).object_id);

	// Written before its maximum protection lost write: its written page is the process's own,
	// under a shadow.
	uint8_t *written = static_cast<uint8_t *>(mmap(nullptr, mapped, PROT_READ, MAP_PRIVATE, file.fd, at));
	CHECK(written != MAP_FAILED);
	CHECK(mprotect(written, mapped, PROT_READ | PROT_WRITE) == 0);
	written[into + page] ^= 0xff;
	CHECK(mprotect(written, mapped, PROT_READ) == 0 && lowerMax(written, mapped));
	RgfxFilePages d = rgfxFilePagesOf(written + into, length);
	CHECK(d.clause == kRgfxFileNeither && d.why == kRgfxNeitherShadowed);
	// The bytes of the range's other pages are still the file's, and so is the range past it.
	CHECK(rgfxFilePagesOf(written + 5 * page, page).clause == kRgfxFileCopyOnWrite);

	// Two regions: the second half of a shared mapping replaced by other pages of the file.
	uint8_t *split = static_cast<uint8_t *>(mmap(nullptr, mapped, PROT_READ, MAP_SHARED, file.fd, at));
	CHECK(split != MAP_FAILED);
	CHECK(mmap(split + 4 * page, 4 * page, PROT_READ, MAP_SHARED | MAP_FIXED, file.fd,
	           at + static_cast<off_t>(16 * page)) == split + 4 * page);
	CHECK(rgfxFilePagesOf(split + 3 * page, 2 * page).why == kRgfxNeitherSpansRegions);
	CHECK(rgfxFilePagesOf(split + 4 * page, page).clause == kRgfxFileMaxNoWrite);

	// Anonymous memory, read-only now: neither while its maximum protection has write, and still
	// neither, why not a file, once it has none and the process holds it alone (SM_PRIVATE).
	auto anonymous = [](size_t bytes, uint8_t fill) {
		mach_vm_address_t a = 0;
		CHECK(mach_vm_allocate(mach_task_self(), &a, bytes, VM_FLAGS_ANYWHERE) == KERN_SUCCESS);
		if (fill != 0)
			std::memset(reinterpret_cast<void *>(a), fill, bytes);
		return reinterpret_cast<uint8_t *>(a);
	};
	uint8_t *anon = anonymous(mapped, 0x11);
	CHECK(mprotect(anon, mapped, PROT_READ) == 0);
	RgfxFilePages e = rgfxFilePagesOf(anon + into, length);
	CHECK(e.clause == kRgfxFileNeither && e.why == kRgfxNeitherNotAFile);
	CHECK(lowerMax(anon, mapped) && regionAt(anon).share_mode == SM_PRIVATE);
	RgfxFilePages f = rgfxFilePagesOf(anon + into, length);
	CHECK(f.clause == kRgfxFileNeither && f.why == kRgfxNeitherNotAFile);
	CHECK(mprotect(anon, mapped, PROT_READ | PROT_WRITE) != 0);

	// Where the kext takes a sole holder's memory: the mapping alone holds it, and it is unchanging.
	auto soleHolder = [&]() {
		const RgfxFilePages p = rgfxFilePagesOf(anon + into, length, true);
		return p.clause == kRgfxAnonymousPrivate && p.why == kRgfxNeitherNone;
	};
	CHECK(pageAt(anon).ref_count == 1 && pageAt(anon).depth == 0);
	CHECK(static_cast<uint32_t>(pageAt(anon).object_id) == regionAt(anon).object_id);
	CHECK(soleHolder());
	// A memory entry of it holds it too, and a second mapping, which takes no write the maximum
	// lacks: neither while each lasts.
	mach_port_t entry = MACH_PORT_NULL;
	memory_object_size_t entryBytes = mapped;
	CHECK(mach_make_memory_entry_64(mach_task_self(), &entryBytes, reinterpret_cast<uintptr_t>(anon),
	                                VM_PROT_READ | MAP_MEM_VM_SHARE, &entry,
	                                MACH_PORT_NULL) == KERN_SUCCESS);
	CHECK(pageAt(anon).ref_count == 2);
	RgfxFilePages held = rgfxFilePagesOf(anon + into, length, true);
	CHECK(held.clause == kRgfxFileNeither && held.why == kRgfxNeitherNotAFile);
	mach_port_deallocate(mach_task_self(), entry);
	CHECK(pageAt(anon).ref_count == 1 && soleHolder());
	mach_vm_address_t remapped = 0;
	vm_prot_t remapCur = VM_PROT_NONE, remapMax = VM_PROT_NONE;
	CHECK(mach_vm_remap(mach_task_self(), &remapped, mapped, 0, VM_FLAGS_ANYWHERE, mach_task_self(),
	                    reinterpret_cast<uintptr_t>(anon), false, &remapCur, &remapMax,
	                    VM_INHERIT_NONE) == KERN_SUCCESS);
	CHECK((remapMax & VM_PROT_WRITE) == 0 && pageAt(anon).ref_count == 2);
	CHECK(rgfxFilePagesOf(anon + into, length, true).clause == kRgfxFileNeither);
	mach_vm_deallocate(mach_task_self(), remapped, mapped);
	CHECK(pageAt(anon).ref_count == 1 && soleHolder());
	// Asked for its regions' footprint: a count of 1 with an id of none of its objects, neither.
	int footprint = 1;
	CHECK(sysctlbyname("vm.self_region_footprint", nullptr, nullptr, &footprint,
	                   sizeof(footprint)) == 0);
	const vm_page_info_basic_data_t told = pageAt(anon);
	const bool fake = told.ref_count == 1 &&
	                  static_cast<uint32_t>(told.object_id) != regionAt(anon).object_id;
	const bool refusedThen = rgfxFilePagesOf(anon + into, length, true).clause == kRgfxFileNeither;
	footprint = 0;
	CHECK(sysctlbyname("vm.self_region_footprint", nullptr, nullptr, &footprint,
	                   sizeof(footprint)) == 0);
	CHECK(fake && refusedThen && soleHolder());

	// Mapped at a second address too, writable there: truly shared, neither.
	uint8_t *aliased = anonymous(mapped, 0x44);
	mach_vm_address_t alias = 0;
	vm_prot_t cur = VM_PROT_NONE, max = VM_PROT_NONE;
	CHECK(mach_vm_remap(mach_task_self(), &alias, mapped, 0, VM_FLAGS_ANYWHERE, mach_task_self(),
	                    reinterpret_cast<uintptr_t>(aliased), false, &cur, &max,
	                    VM_INHERIT_NONE) == KERN_SUCCESS);
	CHECK((max & VM_PROT_WRITE) != 0 && lowerMax(aliased, mapped));
	CHECK(regionAt(aliased).share_mode != SM_PRIVATE);
	RgfxFilePages g = rgfxFilePagesOf(aliased + into, length);
	CHECK(g.clause == kRgfxFileNeither && g.why == kRgfxNeitherNotAFile);

	// vm_copy's destination, large enough for a virtual copy: a copy pending on the source's
	// object, which has none below it; copy-on-write, and unchanging once its maximum protection
	// has no write. The source's write after goes to a copy of the source's own.
	const size_t large = 64 * page;
	auto copyOf = [&](const uint8_t *from) {
		uint8_t *to = anonymous(large, 0);
		CHECK(mach_vm_copy(mach_task_self(), reinterpret_cast<uintptr_t>(from), large,
		                   reinterpret_cast<uintptr_t>(to)) == KERN_SUCCESS);
		return to;
	};
	uint8_t *source = anonymous(large, 0x55);
	uint8_t *copied = copyOf(source);
	CHECK(lowerMax(copied, large));
	CHECK(regionAt(copied).share_mode == SM_COW && shadowDepthAt(copied) == 0);
	RgfxFilePages h = rgfxFilePagesOf(copied + into, length);
	CHECK(h.clause == kRgfxAnonymousCopyOnWrite && h.why == kRgfxNeitherNone);
	std::memset(source, 0x66, large);
	CHECK(copied[into] == 0x55 && copied[large - 1] == 0x55);
	CHECK(rgfxFilePagesOf(copied + into, length).clause == kRgfxAnonymousCopyOnWrite);

	// Written before its maximum protection lost write: copy-on-write over an object below its
	// own, which may hold another mapping's writes, so neither.
	uint8_t *second = anonymous(large, 0x77);
	uint8_t *rewritten = copyOf(second);
	rewritten[into] ^= 0xff;
	CHECK(lowerMax(rewritten, large));
	CHECK(regionAt(rewritten).share_mode == SM_COW && shadowDepthAt(rewritten) != 0);
	RgfxFilePages k = rgfxFilePagesOf(rewritten + into, length);
	CHECK(k.clause == kRgfxFileNeither && k.why == kRgfxNeitherNotAFile);

	// A file on a writable volume, shared from a read-only descriptor and private with its
	// maximum protection taken down; and shared writable.
	char path[] = "/tmp/rgfx-file-pages-XXXXXX";
	const int temp = mkstemp(path);
	CHECK(temp >= 0);
	std::vector<uint8_t> fill(mapped, 0x22);
	CHECK(write(temp, fill.data(), mapped) == static_cast<ssize_t>(mapped));
	const int readOnly = open(path, O_RDONLY | O_CLOEXEC);
	uint8_t *onData = static_cast<uint8_t *>(mmap(nullptr, mapped, PROT_READ, MAP_SHARED, readOnly, 0));
	uint8_t *onDataPriv =
	    static_cast<uint8_t *>(mmap(nullptr, mapped, PROT_READ, MAP_PRIVATE, readOnly, 0));
	uint8_t *writable =
	    static_cast<uint8_t *>(mmap(nullptr, mapped, PROT_READ, MAP_SHARED, temp, 0));
	CHECK(onData != MAP_FAILED && onDataPriv != MAP_FAILED && writable != MAP_FAILED);
	CHECK(lowerMax(onDataPriv, mapped));
	CHECK(rgfxFilePagesOf(onData + into, length).why == kRgfxNeitherWritableVolume);
	CHECK(rgfxFilePagesOf(onDataPriv + into, length).why == kRgfxNeitherWritableVolume);
	CHECK(rgfxFilePagesOf(writable + into, length).why == kRgfxNeitherWritableMapping);

	munmap(writable, mapped);
	munmap(onDataPriv, mapped);
	munmap(onData, mapped);
	close(readOnly);
	close(temp);
	unlink(path);
	for (uint8_t *a : {anon, aliased})
		mach_vm_deallocate(mach_task_self(), reinterpret_cast<uintptr_t>(a), mapped);
	for (uint8_t *a : {source, copied, second, rewritten})
		mach_vm_deallocate(mach_task_self(), reinterpret_cast<uintptr_t>(a), large);
	mach_vm_deallocate(mach_task_self(), alias, mapped);
	munmap(split, mapped);
	munmap(written, mapped);
	munmap(priv, mapped);
	munmap(shared, mapped);
}

// Read-only memory under a no-copy buffer (RgfxRegion::acquirePages with setUnchangingPages): the
// block is always of the app's own memory. Where it cannot change it is named unchanging, and
// copy-on-write for a private mapping whose maximum protection has no write, and the GPU's pages
// hold the file's bytes; a block of the same bytes joins while they are the same memory. Memory
// that is neither is asked read-only with why not in the bits from kPagesWhyShift: a private
// mapping as mmap makes it (writable max protection) and one written before its maximum
// protection lost write (shadowed), whose GPU's pages hold the process's bytes. Memory whose
// unchanging block the kext refused is asked again read-only, why RGFX_PAGES_WHY_REFUSED. Writable
// memory is not classified, and without setUnchangingPages nothing is. Anonymous memory whose
// maximum protection has no write is named unchanging and copy-on-write with a copy pending on its
// own object, the GPU's pages the copy's bytes, and is read-only, why not a file, where the
// process holds it alone, until the kext takes a sole holder's memory (setSoleHolderPages): then it
// is named unchanging and a sole holder's, the GPU's pages its bytes, and asked again read-only,
// why refused, where the kext refuses that. Each classification is counted.
void test_region_unchanging_file_pages()
{
	const uint64_t page = vm_page_size;
	const off_t at = static_cast<off_t>(8 * page);
	const size_t mapped = 8 * page;
	const uint64_t into = 0x180, length = 3 * page + 0x40;
	ReadOnlyFile file(at + static_cast<off_t>(into), length);
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0);
	RgfxRegion region(mem.data(), mem.size());
	struct Source {
		struct Ask {
			const void *address;
			uint64_t length;
			uint32_t how;
			std::vector<uint8_t> seen; // what the GPU reads at the call
			mach_port_t held;          // the pages, as the kext's wire holds them
		};
		std::vector<Ask> asked;
		uint64_t next = RGFX_ALLOCATIONS_OFFSET + 0x10000;
		bool refuseUnchanging = false;
		static bool block(void *c, const void *address, uint64_t length, uint32_t how,
		                  uint64_t *offset, uint64_t *size, uint64_t *gpuAddress)
		{
			Source *s = static_cast<Source *>(c);
			const uint8_t *bytes = static_cast<const uint8_t *>(address);
			Source::Ask ask{address, length, how, std::vector<uint8_t>(bytes, bytes + length),
			                MACH_PORT_NULL};
			// The pages the bytes lie in, shared, as the kext's wire of a block's pages holds the
			// process's pages.
			const uintptr_t start = reinterpret_cast<uintptr_t>(address);
			const uintptr_t base = start & ~(uintptr_t)(vm_page_size - 1);
			memory_object_size_t held =
			    ((start + length + vm_page_size - 1) & ~(uintptr_t)(vm_page_size - 1)) - base;
			CHECK(mach_make_memory_entry_64(mach_task_self(), &held, base,
			                                VM_PROT_READ | MAP_MEM_VM_SHARE, &ask.held,
			                                MACH_PORT_NULL) == KERN_SUCCESS);
			s->asked.push_back(std::move(ask));
			if (s->refuseUnchanging && (how & RgfxRegion::kPagesUnchanging) != 0)
				return false;
			*offset = s->next;
			*size = 0x4000;
			*gpuAddress = 0;
			s->next += 0x4000;
			return true;
		}
		static bool alloc(void *, uint64_t, uint64_t, uint64_t *) { return false; }
		static void release(void *, uint64_t) {}
		// What the GPU's pages of ask `i` hold now.
		std::vector<uint8_t> now(size_t i) const
		{
			// The entry starts at the address's page.
			const uint64_t in = reinterpret_cast<uintptr_t>(asked[i].address) & (vm_page_size - 1);
			const uint64_t span = (in + asked[i].length + vm_page_size - 1) & ~(vm_page_size - 1);
			mach_vm_address_t view = 0;
			CHECK(mach_vm_map(mach_task_self(), &view, span, 0, VM_FLAGS_ANYWHERE, asked[i].held,
			                  0, false, VM_PROT_READ, VM_PROT_READ,
			                  VM_INHERIT_NONE) == KERN_SUCCESS);
			const uint8_t *p = reinterpret_cast<const uint8_t *>(view) + in;
			std::vector<uint8_t> out(p, p + asked[i].length);
			mach_vm_deallocate(mach_task_self(), view, span);
			return out;
		}
	} src;
	region.setBlockSource(&Source::alloc, &Source::release, &src);
	region.setPagesBlockSource(&Source::block, &src);
	region.setUnchangingPages(true);
	CHECK(region.unchangingPages());
	const RgfxFilePagesCounts before = rgfxFilePagesCounts();
	const uint32_t join = 2;
	const uint32_t shift = RgfxRegion::kPagesWhyShift;
	auto privateOf = [&](bool lowerMax) {
		uint8_t *p = static_cast<uint8_t *>(mmap(nullptr, mapped, PROT_READ, MAP_PRIVATE, file.fd, at));
		CHECK(p != MAP_FAILED);
		if (lowerMax)
			CHECK(mach_vm_protect(mach_task_self(), reinterpret_cast<uintptr_t>(p), mapped, true,
			                      VM_PROT_READ) == KERN_SUCCESS);
		return p;
	};

	// Shared, max-no-write: the app's own pages, named unchanging, the file's bytes.
	uint8_t *shared = static_cast<uint8_t *>(mmap(nullptr, mapped, PROT_READ, MAP_SHARED, file.fd, at));
	CHECK(shared != MAP_FAILED);
	RgfxRegion::Block a = region.acquirePages(shared + into, length, false, join);
	CHECK(a.ok && src.asked.size() == 1);
	CHECK(src.asked[0].how == RgfxRegion::kPagesUnchanging &&
	      src.asked[0].address == shared + into);
	CHECK(src.asked[0].seen == file.bytes && src.now(0) == file.bytes);

	// Private as mmap makes it: neither, writable max protection, the app's own memory read-only.
	uint8_t *priv = privateOf(false);
	RgfxRegion::Block b = region.acquirePages(priv + into, length, false, join);
	CHECK(b.ok && b.offset != a.offset && src.asked.size() == 2);
	CHECK(src.asked[1].how == (RGFX_PAGES_WHY_WRITABLE_MAX_PROTECTION << shift) &&
	      src.asked[1].address == priv + into);

	// Private with its maximum protection taken down: copy-on-write, the GPU's pages the file's
	// bytes, which the process can never write; the same bytes join.
	uint8_t *cow = privateOf(true);
	RgfxRegion::Block c = region.acquirePages(cow + into, length, false, join);
	CHECK(c.ok && c.offset != b.offset && src.asked.size() == 3);
	CHECK(src.asked[2].how == (RgfxRegion::kPagesUnchanging | RgfxRegion::kPagesCopyOnWrite) &&
	      src.asked[2].address == cow + into);
	CHECK(src.asked[2].seen == file.bytes && src.now(2) == file.bytes);
	CHECK(mprotect(cow, mapped, PROT_READ | PROT_WRITE) != 0);
	RgfxRegion::Block again = region.acquirePages(cow + into, length, false, join);
	CHECK(again.ok && again.offset == c.offset && src.asked.size() == 3);

	// Written before its maximum protection lost write: neither, shadowed, the GPU's pages the
	// process's bytes.
	uint8_t *written = privateOf(false);
	CHECK(mprotect(written, mapped, PROT_READ | PROT_WRITE) == 0);
	for (uint64_t i = 0; i < page; i++)
		written[into + i] = file.bytes[i] ^ 0xff;
	CHECK(mprotect(written, mapped, PROT_READ) == 0);
	CHECK(mach_vm_protect(mach_task_self(), reinterpret_cast<uintptr_t>(written), mapped, true,
	                      VM_PROT_READ) == KERN_SUCCESS);
	RgfxRegion::Block d = region.acquirePages(written + into, length, false, join);
	CHECK(d.ok && src.asked.size() == 4);
	CHECK(src.asked[3].how == (RGFX_PAGES_WHY_SHADOWED << shift) &&
	      src.asked[3].address == written + into);
	CHECK(src.asked[3].seen[0] == (file.bytes[0] ^ 0xff) && src.now(3)[0] == (file.bytes[0] ^ 0xff));

	// The kext refuses the unchanging block: the app's own memory again, read-only, why refused.
	src.refuseUnchanging = true;
	RgfxRegion::Block e = region.acquirePages(shared + into, length, false, RgfxRegion::kPagesOwn);
	CHECK(e.ok && src.asked.size() == 6);
	CHECK(src.asked[4].how == RgfxRegion::kPagesUnchanging &&
	      src.asked[5].how == (RGFX_PAGES_WHY_REFUSED << shift) &&
	      src.asked[4].address == shared + into && src.asked[5].address == shared + into);
	src.refuseUnchanging = false;

	// Writable memory is never classified, and is asked writable.
	std::vector<uint8_t> heap(4 * page, 0x33);
	RgfxRegion::Block f = region.acquirePages(heap.data(), heap.size(), true, RgfxRegion::kPagesOwn);
	CHECK(f.ok && src.asked.size() == 7 && src.asked[6].how == RgfxRegion::kPagesWritable &&
	      src.asked[6].address == heap.data());

	const RgfxFilePagesCounts after = rgfxFilePagesCounts();
	auto grew = [&](const uint64_t (&now)[kRgfxFileClauses], const uint64_t (&was)[kRgfxFileClauses],
	                RgfxFileClause clause) { return now[clause] - was[clause]; };
	CHECK(after.classified - before.classified == 5);
	CHECK(grew(after.count, before.count, kRgfxFileMaxNoWrite) == 2 &&
	      grew(after.bytes, before.bytes, kRgfxFileMaxNoWrite) == 2 * length);
	CHECK(grew(after.count, before.count, kRgfxFileCopyOnWrite) == 1 &&
	      grew(after.bytes, before.bytes, kRgfxFileCopyOnWrite) == length);
	CHECK(grew(after.count, before.count, kRgfxFileNeither) == 2);
	for (RgfxFileNeither why : {kRgfxNeitherWritableMaxProtection, kRgfxNeitherShadowed})
		CHECK(after.neitherCount[why] - before.neitherCount[why] == 1 &&
		      after.neitherBytes[why] - before.neitherBytes[why] == length);
	CHECK(after.kextRefused - before.kextRefused == 1 &&
	      after.kextRefusedBytes - before.kextRefusedBytes == length);
	uint64_t timed = 0;
	for (unsigned i = 0; i < RgfxFilePagesCounts::kBuckets; i++)
		timed += after.buckets[i] - before.buckets[i];
	CHECK(timed == 5 && after.ns > before.ns);

	// Without setUnchangingPages nothing is classified, and read-only memory is asked as it is.
	RgfxRegion plain(mem.data(), mem.size());
	plain.setBlockSource(&Source::alloc, &Source::release, &src);
	plain.setPagesBlockSource(&Source::block, &src);
	RgfxRegion::Block g = plain.acquirePages(shared + into, length, false, RgfxRegion::kPagesOwn);
	CHECK(g.ok && src.asked.size() == 8 && src.asked[7].how == 0 &&
	      src.asked[7].address == shared + into);
	CHECK(rgfxFilePagesCounts().classified == after.classified);

	// Anonymous memory with a copy pending on its own object, vm_copy's destination, whose maximum
	// protection has no write: named unchanging and copy-on-write, the GPU's pages the copy's
	// bytes, which the source's write after leaves as they are. Anonymous memory the process holds
	// alone (SM_PRIVATE) is read-only, why not a file. Each counted by its clause.
	const size_t large = 64 * page;
	mach_vm_address_t from = 0, to = 0, alone = 0;
	for (mach_vm_address_t *a : {&from, &to})
		CHECK(mach_vm_allocate(mach_task_self(), a, large, VM_FLAGS_ANYWHERE) == KERN_SUCCESS);
	std::memset(reinterpret_cast<void *>(from), 0x5a, large);
	CHECK(mach_vm_copy(mach_task_self(), from, large, to) == KERN_SUCCESS);
	CHECK(mach_vm_protect(mach_task_self(), to, large, true, VM_PROT_READ) == KERN_SUCCESS);
	uint8_t *copied = reinterpret_cast<uint8_t *>(to);
	RgfxRegion::Block h = region.acquirePages(copied + into, length, false, RgfxRegion::kPagesOwn);
	CHECK(h.ok && src.asked.size() == 9 &&
	      src.asked[8].how == (RgfxRegion::kPagesUnchanging | RgfxRegion::kPagesCopyOnWrite) &&
	      src.asked[8].address == copied + into);
	std::memset(reinterpret_cast<void *>(from), 0x66, large);
	CHECK(src.asked[8].seen == std::vector<uint8_t>(length, 0x5a) &&
	      src.now(8) == src.asked[8].seen);
	CHECK(mach_vm_allocate(mach_task_self(), &alone, mapped, VM_FLAGS_ANYWHERE) == KERN_SUCCESS);
	std::memset(reinterpret_cast<void *>(alone), 0x5a, mapped);
	CHECK(mach_vm_protect(mach_task_self(), alone, mapped, true, VM_PROT_READ) == KERN_SUCCESS);
	uint8_t *own = reinterpret_cast<uint8_t *>(alone);
	RgfxRegion::Block k = region.acquirePages(own + into, length, false, RgfxRegion::kPagesOwn);
	CHECK(k.ok && src.asked.size() == 10 &&
	      src.asked[9].how == (RGFX_PAGES_WHY_NOT_A_FILE << shift) &&
	      src.asked[9].address == own + into);
	const RgfxFilePagesCounts last = rgfxFilePagesCounts();
	CHECK(last.classified - after.classified == 2 &&
	      grew(last.count, after.count, kRgfxAnonymousCopyOnWrite) == 1 &&
	      grew(last.bytes, after.bytes, kRgfxAnonymousCopyOnWrite) == length &&
	      last.neitherCount[kRgfxNeitherNotAFile] - after.neitherCount[kRgfxNeitherNotAFile] == 1);

	// Where the kext takes a sole holder's memory, anonymous memory the process's mapping alone
	// holds is named unchanging and a sole holder's, the GPU's pages its bytes; refused, it is
	// asked again read-only, why refused. Each counted by its clause.
	CHECK(!region.soleHolderPages());
	region.setSoleHolderPages(true);
	CHECK(region.soleHolderPages());
	mach_vm_address_t sole = 0, refusedSole = 0;
	for (mach_vm_address_t *a : {&sole, &refusedSole}) {
		CHECK(mach_vm_allocate(mach_task_self(), a, mapped, VM_FLAGS_ANYWHERE) == KERN_SUCCESS);
		std::memset(reinterpret_cast<void *>(*a), 0x3c, mapped);
		CHECK(mach_vm_protect(mach_task_self(), *a, mapped, true, VM_PROT_READ) == KERN_SUCCESS);
	}
	const uint32_t soleHow = RgfxRegion::kPagesUnchanging | RgfxRegion::kPagesSoleHolder;
	uint8_t *soleBytes = reinterpret_cast<uint8_t *>(sole);
	RgfxRegion::Block m = region.acquirePages(soleBytes + into, length, false, RgfxRegion::kPagesOwn);
	CHECK(m.ok && src.asked.size() == 11 && src.asked[10].how == soleHow &&
	      src.asked[10].address == soleBytes + into);
	CHECK(src.asked[10].seen == std::vector<uint8_t>(length, 0x3c) &&
	      src.now(10) == src.asked[10].seen);
	src.refuseUnchanging = true;
	uint8_t *refusedBytes = reinterpret_cast<uint8_t *>(refusedSole);
	RgfxRegion::Block n =
	    region.acquirePages(refusedBytes + into, length, false, RgfxRegion::kPagesOwn);
	CHECK(n.ok && src.asked.size() == 13 && src.asked[11].how == soleHow &&
	      src.asked[12].how == (RGFX_PAGES_WHY_REFUSED << shift) &&
	      src.asked[12].address == refusedBytes + into);
	src.refuseUnchanging = false;
	const RgfxFilePagesCounts soleCounts = rgfxFilePagesCounts();
	CHECK(soleCounts.classified - last.classified == 2 &&
	      grew(soleCounts.count, last.count, kRgfxAnonymousPrivate) == 2 &&
	      grew(soleCounts.bytes, last.bytes, kRgfxAnonymousPrivate) == 2 * length &&
	      soleCounts.kextRefused - last.kextRefused == 1);

	for (const Source::Ask &ask : src.asked)
		mach_port_deallocate(mach_task_self(), ask.held);
	for (mach_vm_address_t a : {from, to})
		mach_vm_deallocate(mach_task_self(), a, large);
	for (mach_vm_address_t a : {alone, sole, refusedSole})
		mach_vm_deallocate(mach_task_self(), a, mapped);
	munmap(written, mapped);
	munmap(cow, mapped);
	munmap(priv, mapped);
	munmap(shared, mapped);
}

// The bucket of a classification's time (RgfxFilePagesCounts::buckets): four an octave, each
// from its least time to the next one's.
void test_file_pages_buckets()
{
	CHECK(rgfxFilePagesBucketNs(0) == 0 && rgfxFilePagesBucketNs(3) == 3);
	CHECK(rgfxFilePagesBucketNs(8) == 4 && rgfxFilePagesBucketNs(11) == 7);
	CHECK(rgfxFilePagesBucketNs(12) == 8 && rgfxFilePagesBucketNs(13) == 10);
	CHECK(rgfxFilePagesBucketNs(4 * 10) == 1024 && rgfxFilePagesBucketNs(4 * 10 + 1) == 1280);
	CHECK(rgfxFilePagesBucketNs(255) == 7ull << 61);
	for (unsigned i = 8; i + 1 < RgfxFilePagesCounts::kBuckets; i++)
		CHECK(rgfxFilePagesBucketNs(i) < rgfxFilePagesBucketNs(i + 1));
}

// Where the kext maps parts of blocks write-combined, a block asked with a write-combined part
// takes it from the write-combined source, and one asked without from the block source; a queue's
// submission ring's data starts its block's second page, write-combined, with the ring's header
// ending the first, and the pair names the blocks to release. Without the source every block is
// the block source's, the rings at their blocks' starts.
void test_region_write_combined_parts()
{
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x100000, 0xee);
	struct Source {
		uint64_t next = RGFX_ALLOCATIONS_OFFSET;
		std::vector<uint64_t> combinedFrom; // UINT64_MAX for a plain block
		std::vector<uint64_t> released;
		bool place(uint64_t size, uint64_t align, uint64_t from, uint64_t *offset)
		{
			next = (next + align - 1) & ~(align - 1);
			*offset = next;
			next += size;
			combinedFrom.push_back(from);
			return true;
		}
		static bool alloc(void *c, uint64_t size, uint64_t align, uint64_t *offset)
		{
			return static_cast<Source *>(c)->place(size, align, UINT64_MAX, offset);
		}
		static bool combined(void *c, uint64_t size, uint64_t align, uint64_t from,
		                     uint64_t *offset)
		{
			return static_cast<Source *>(c)->place(size, align, from, offset);
		}
		static void release(void *c, uint64_t offset)
		{
			static_cast<Source *>(c)->released.push_back(offset);
		}
	};
	{
		RgfxRegion region(mem.data(), mem.size());
		Source src;
		region.setBlockSource(&Source::alloc, &Source::release, &src);
		CHECK(!region.writeCombines());
		RgfxRegion::Block b = region.allocate(0x2000, 0x1000, 0);
		CHECK(b.ok && src.combinedFrom == std::vector<uint64_t>({UINT64_MAX}));
		RgfxRegion::RingPair rings = region.allocateRingPair(0x8000, 0x8000);
		CHECK(rings.ok && rings.submit_offset == rings.submit_block);
		CHECK(rings.complete_offset == rings.complete_block && rings.complete_offset != 0);
	}
	std::fill(mem.begin(), mem.end(), 0xee); // the rings above zeroed theirs
	RgfxRegion region(mem.data(), mem.size());
	Source src;
	region.setBlockSource(&Source::alloc, &Source::release, &src);
	region.setWriteCombinedSource(&Source::combined);
	CHECK(region.writeCombines());
	RgfxRegion::Block plain = region.allocate(0x2000);
	RgfxRegion::Block whole = region.allocate(0x2000, 0x1000, 0);
	RgfxRegion::Block tail = region.allocate(0x3000, 0x1000, 0x1000);
	RgfxRegion::Block past = region.allocate(0x1000, 0x1000, 0x1000);
	CHECK(plain.ok && whole.ok && tail.ok && past.ok);
	CHECK(src.combinedFrom ==
	      std::vector<uint64_t>({UINT64_MAX, 0, 0x1000, UINT64_MAX}));
	src.combinedFrom.clear();
	RgfxRegion::RingPair rings = region.allocateRingPair(0x8000, 0x8000);
	CHECK(rings.ok && rings.submit_data_size == 0x8000);
	CHECK(src.combinedFrom == std::vector<uint64_t>({0x1000, UINT64_MAX}));
	CHECK(rings.submit_block % 0x1000 == 0);
	CHECK(rings.submit_offset == rings.submit_block + 0x1000 - RGFX_RING_HEADER_SIZE);
	CHECK((rings.submit_offset + RGFX_RING_HEADER_SIZE) % 0x1000 == 0);
	CHECK(rings.submit_offset % RgfxRegion::kRingAlign == 0);
	CHECK(rings.complete_block == rings.complete_offset);
	// The ring's header and data are zeroed; the head of the block before them is not touched.
	bool zeroed = true;
	for (uint64_t i = 0; i < RGFX_RING_HEADER_SIZE + 0x8000; i++)
		zeroed = zeroed && mem[rings.submit_offset + i] == 0;
	CHECK(zeroed && mem[rings.submit_block] == 0xee);
}

// A case rgfxRunFairShare runs in a process of its own: 0 where it passed.
static int fair_share_case(const char *name)
{
	if (std::strcmp(name, "held-since-the-post-is-not-counted") == 0) {
		test_held_since_the_post_is_not_counted();
		return 0;
	}
	std::fprintf(stderr, "no fair-share case %s\n", name);
	return 2;
}

int main(int argc, char **argv)
{
	if (argc == 3 && std::strcmp(argv[1], "--fair-share") == 0)
		return fair_share_case(argv[2]);
	test_id_space();
	test_private_address_picks();
	test_private_heaps_align_from_their_block();
	test_obj_ids_are_given_once();
	test_region_pages_blocks();
	test_region_pages_joins();
	test_region_pages_join_over_remapped_memory();
	test_file_pages_classify();
	test_region_unchanging_file_pages();
	test_file_pages_buckets();
	test_region_write_combined_parts();
	test_address_picks();
	test_resource_id_chunks();
	test_small_resource_id_chunks();
	test_resource_id_chunk_size_is_the_grant_s();
	test_caps();
	test_iosurface_alignments_divide_other_codes_surface_rows();
	test_spi_features();
	test_family_names();
	test_vendor_names();
	test_metal_names();
	test_concurrent_compilation_task_count();
	test_host_resource_options();
	test_buffer_over_pages_at_the_host_s_address();
	test_buffer_over_pages_at_the_host_s_address_later();
	test_caps_features();
	test_feature_profile_for_guest();
	test_archive_file();
	test_lifecycle_messages();
	test_managed_buffers_at_their_copies();
	test_creates_with_no_address_post_nothing();
	test_library_message();
	test_pipeline_messages();
	test_mesh_pipeline_messages();
	test_mesh_ops();
	test_ring_wrap_and_no_room();
	test_ring_doorbell_follows_the_mark();
	test_ring_doorbell_asks_where_the_device_takes_them();
	test_ring_post_waits_for_room();
	test_ring_post_waits_for_the_room_record();
	test_ring_room_sleeps_count_for_the_line();
	test_command_buffer_encoder();
	test_command_buffer_storage();
	test_ring_stamps_the_posted_time_before_the_publish();
	test_guest_times_go_first();
	test_video_command_buffer();
	test_render_pass_sample_positions();
	test_render_pass_visibility();
	test_forwarded_lifecycle();
	test_held_destroys();
	test_give_backs_wait_for_the_buffers_committed_before();
	test_the_context_s_end_runs_the_give_backs_held();
	test_held_buffers();
	test_held_creates();
	rgfxRunFairShare(argv[0], "--fair-share", "held-since-the-post-is-not-counted");
	test_library_key();
	test_library_key_pieces();
	test_library_key_of_file();
	test_keyed_library_messages();
	test_dynamic_library_messages();
	test_heap_ranges();
	test_heap_texture_size_query();
	test_rate_map_messages();
	test_function_pointer_messages();
	test_memory_shadow();
	test_argument_layout_queries();
	test_argument_layout_parse();
	rgfxSlabTests();
	rgfxLibraryStreamTests();
	rgfxFunctionConstantsTests();
	std::printf("serializer_tests: OK (%d checks)\n", g_checks.load());
	return 0;
}
