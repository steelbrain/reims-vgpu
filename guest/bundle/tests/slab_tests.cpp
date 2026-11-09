/*
 * slab_tests: the slab allocator of the device's Shared and Managed buffers (RgfxSlabs.hpp), over
 * a source of memory of the test's own and over the region's source, part of the serialiser check
 * (serializer_tests.cpp calls rgfxSlabTests; guest/bundle/Makefile `test`). No Metal.
 */

#include <mach/kern_return.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "RgfxHeapRanges.hpp"
#include "RgfxObjectModel.hpp"
#include "RgfxRegion.hpp"
#include "RgfxRingWriter.hpp"
#include "RgfxSerializer.hpp"
#include "RgfxSlabs.hpp"
#include "RgfxSuperpages.hpp"
#include "reims_graphics_wire.h"
#include "../../kext/src/RgfxSuperpageRecord.hpp"
#include "rgfx_test_ring.hpp" // CHECK, g_checks, TestRing, expectMessage

void rgfxSlabTests();

namespace {

constexpr uint32_t kShared = 0x200;          // Shared, default cache, Tracked
constexpr uint32_t kSharedUntracked = 0x100; // Shared, default cache, Untracked
constexpr uint32_t kManaged = 0x210;         // Managed, Tracked
constexpr uint64_t kMiB = 1ull << 20;

// Memory of the test's own, handed out first fit in units of `unitBytes` (16 KiB, as the region's),
// every byte 0xee as it first comes, with a log of what the allocator asked of it, in order. Its
// Shared classes are heaped where `heapedClasses`, as the region's are.
struct Source : RgfxSlabSource {
	std::vector<uint8_t> raw;
	uint8_t *base; // raw's first byte at a 16 KiB boundary
	uint64_t bytes;
	RgfxHeapRanges free;
	uint64_t gpu = 0x100000000ull;
	uint64_t nextObj = 1;
	uint64_t unitBytes;
	bool refuseHeaps = false;
	bool zeroed = false;
	bool heapedClasses = true;
	std::mutex lock;
	std::vector<std::string> log;
	uint64_t takes = 0, refusedTakes = 0, gives = 0, heaps = 0, heapDrops = 0;

	explicit Source(uint64_t size, uint64_t unit = 0x4000)
	    : raw(size + 0x4000, 0xee), bytes(size), free(size), unitBytes(unit)
	{
		base = raw.data() + (0x4000 - reinterpret_cast<uintptr_t>(raw.data()) % 0x4000) % 0x4000;
	}

	uint64_t unit() const override { return unitBytes; }
	uint64_t capacity() const override { return bytes; }
	bool heaped(uint32_t options) const override
	{
		return heapedClasses && ((options >> 4) & 0xf) == 0;
	}
	bool take(uint64_t size, Memory *out) override
	{
		std::lock_guard<std::mutex> hold(lock);
		uint64_t at = 0;
		if (!free.take(size, unit(), &at)) {
			refusedTakes++;
			log.push_back("refuse " + std::to_string(size));
			return false;
		}
		takes++;
		log.push_back("take " + std::to_string(size));
		out->bytes = base + at;
		out->size = size;
		out->regionOffset = at;
		out->zeroed = zeroed;
		if (zeroed)
			std::memset(out->bytes, 0, size);
		return true;
	}
	void give(const Memory &memory) override
	{
		std::lock_guard<std::mutex> hold(lock);
		gives++;
		log.push_back("give " + std::to_string(memory.size));
		free.give(memory.regionOffset, memory.size);
	}
	bool makeHeap(const Memory &memory, uint32_t options, Heap *out) override
	{
		std::lock_guard<std::mutex> hold(lock);
		if (refuseHeaps)
			return false;
		heaps++;
		out->objId = nextObj++;
		out->gpuAddress = gpu;
		gpu += memory.size + 0x8000;
		log.push_back("heap");
		return true;
	}
	void dropHeap(const Heap &heap) override
	{
		std::lock_guard<std::mutex> hold(lock);
		heapDrops++;
		log.push_back("drop heap");
	}
	bool makeBuffer(const Memory &memory, const Heap &heap, uint64_t offset, uint64_t length,
	                uint32_t options, Made *out) override
	{
		std::lock_guard<std::mutex> hold(lock);
		out->objId = nextObj++;
		out->gpuAddress = heap.objId != 0 ? heap.gpuAddress + offset : (gpu += 0x10000);
		out->ownsAddress = heap.objId == 0;
		return true;
	}
	// Whether the last `n` entries of the log are `want`.
	bool logEnds(const std::vector<std::string> &want)
	{
		std::lock_guard<std::mutex> hold(lock);
		return log.size() >= want.size() &&
		       std::equal(want.begin(), want.end(), log.end() - want.size());
	}
};

bool allBytes(const uint8_t *p, uint64_t n, uint8_t v)
{
	for (uint64_t i = 0; i < n; i++) {
		if (p[i] != v)
			return false;
	}
	return true;
}

// A buffer takes whole 4 KiB pages of a slab, page aligned, and buffers of one class share a slab
// (the first 128 KiB), first fit; other options take a slab of their own.
void test_slab_granule_and_classes()
{
	Source source(64 * kMiB);
	RgfxSlabs slabs(source);
	std::vector<RgfxSlabs::Range> held;
	for (int i = 0; i < 32; i++) {
		RgfxSlabs::Range r;
		CHECK(slabs.take(256, kShared, 0, &r));
		CHECK(r.size == 0x1000 && r.offset == 0x1000ull * i);
		CHECK(reinterpret_cast<uintptr_t>(r.contents) % 0x1000 == 0);
		CHECK(r.slab == (held.empty() ? r.slab : held[0].slab));
		held.push_back(r);
	}
	CHECK(source.takes == 1 && source.log[0] == "take 131072" && source.heaps == 1);
	// One more takes the class's next slab, twice as large.
	RgfxSlabs::Range next;
	CHECK(slabs.take(1, kShared, 0, &next) && next.slab != held[0].slab && next.offset == 0);
	CHECK(source.log.back() == "heap" && source.takes == 2 && source.log[2] == "take 262144");
	// The buffer's GPU address is its heap's plus its offset.
	RgfxSlabSource::Made a, b;
	CHECK(slabs.makeBuffer(held[0], 256, kShared, &a) && slabs.makeBuffer(held[3], 256, kShared, &b));
	CHECK(b.gpuAddress == a.gpuAddress + 0x3000 && !a.ownsAddress);
	// Untracked and Managed buffers lie in slabs of their own; a Managed one in no heap.
	RgfxSlabs::Range untracked, managed;
	CHECK(slabs.take(0x1000, kSharedUntracked, 0, &untracked) && untracked.slab != held[0].slab &&
	      untracked.slab != next.slab);
	CHECK(slabs.take(0x1000, kManaged, 0, &managed) && managed.slab != untracked.slab);
	CHECK(source.heaps == 3 && source.takes == 4);
	RgfxSlabSource::Made m;
	CHECK(slabs.makeBuffer(managed, 0x1000, kManaged, &m) && m.ownsAddress);
	// A buffer of 1 MiB still lies in a class's slab; one a byte past it takes its own.
	RgfxSlabs::Range mib, big;
	CHECK(slabs.take(kMiB, kShared, 0, &mib) && mib.size == kMiB);
	CHECK(slabs.take(kMiB + 1, kShared, 0, &big) && big.size == kMiB + 0x1000 && big.offset == 0);
	CHECK(source.log.back() == "heap" && source.log[source.log.size() - 2] == "take 1064960");
	for (auto &r : held)
		slabs.give(r);
	slabs.give(next);
	slabs.give(untracked);
	slabs.give(managed);
	slabs.give(mib);
	slabs.give(big);
	CHECK(slabs.stats().usedBytes == 0);
}

// A class's slabs grow 128 KiB, 256, 512, 1 MiB, then 2 MiB each, and a buffer never makes one
// smaller than twice itself.
void test_slab_growth()
{
	Source source(64 * kMiB);
	RgfxSlabs slabs(source);
	std::vector<RgfxSlabs::Range> held;
	while (source.takes < 7) {
		RgfxSlabs::Range r;
		CHECK(slabs.take(0x10000, kShared, 0, &r));
		held.push_back(r);
	}
	std::vector<std::string> sizes;
	for (const auto &e : source.log) {
		if (e.rfind("take ", 0) == 0)
			sizes.push_back(e.substr(5));
	}
	CHECK(sizes == std::vector<std::string>({"131072", "262144", "524288", "1048576", "2097152",
	                                         "2097152", "2097152"}));
	Source other(64 * kMiB);
	RgfxSlabs first(other);
	RgfxSlabs::Range r;
	CHECK(first.take(600 << 10, kShared, 0, &r) && other.log[0] == "take 2097152");
	first.give(r);
	for (auto &h : held)
		slabs.give(h);
}

// A range freed is free again, first fit: freed pages beside each other join, and a buffer takes
// the lowest hole that holds it, never one too small; another slab only where none does.
void test_slab_reuse_and_fragmentation()
{
	Source source(64 * kMiB);
	RgfxSlabs slabs(source);
	std::vector<RgfxSlabs::Range> held(32);
	for (auto &r : held)
		CHECK(slabs.take(0x1000, kShared, 0, &r));
	for (int i = 0; i < 32; i += 2)
		slabs.give(held[i]);
	// Every hole is one page: two pages go to the next slab.
	RgfxSlabs::Range two;
	CHECK(slabs.take(0x2000, kShared, 0, &two) && two.slab != held[0].slab);
	// A page goes in the lowest hole.
	RgfxSlabs::Range one;
	CHECK(slabs.take(0x1000, kShared, 0, &one) && one.slab == held[0].slab && one.offset == 0);
	// Pages 4, 5 and 6 free make a hole of three, which a buffer of three pages takes.
	slabs.give(held[5]);
	RgfxSlabs::Range three;
	CHECK(slabs.take(0x3000, kShared, 0, &three) && three.slab == held[0].slab &&
	      three.offset == 0x4000);
	// The slab's pages past the 32 taken are free too: 0x20000 - 0x20000 = none; a fourth page
	// lies in hole 2.
	RgfxSlabs::Range again;
	CHECK(slabs.take(0x1000, kShared, 0, &again) && again.slab == held[0].slab &&
	      again.offset == 0x2000);
	CHECK(source.takes == 2);
	slabs.give(two);
	slabs.give(one);
	slabs.give(three);
	slabs.give(again);
	for (int i = 1; i < 32; i += 2) {
		if (i != 5)
			slabs.give(held[i]);
	}
	CHECK(slabs.stats().usedBytes == 0 && slabs.stats().emptySlabs == 2);
}

// Every range reads zero as it is taken, whatever its memory held: the source's bytes, a freed
// buffer's. A create zeroes only past the bytes its caller writes; the background zeroes freed
// bytes, after which a create finds them clean.
void test_slab_zeroing()
{
	Source source(64 * kMiB);
	RgfxSlabs slabs(source);
	RgfxSlabs::Range r;
	CHECK(slabs.take(0x3000, kShared, 0, &r) && allBytes(r.contents, 0x3000, 0));
	// The rest of the slab is still the source's bytes until the background zeroes it.
	CHECK(r.contents[0x3000] == 0xee && slabs.stats().dirtyBytes == 0x20000 - 0x3000);
	std::memset(r.contents, 0xa5, 0x3000);
	slabs.give(r);
	CHECK(slabs.stats().dirtyBytes == 0x20000);
	RgfxSlabs::Range again;
	CHECK(slabs.take(0x2000, kShared, 0x100, &again) && again.offset == 0);
	// The caller writes its first 256 bytes: the rest reads zero.
	CHECK(again.contents[0] == 0xa5 && again.contents[0xff] == 0xa5 &&
	      allBytes(again.contents + 0x100, 0x1f00, 0));
	slabs.give(again);
	while (slabs.step()) {
	}
	CHECK(slabs.stats().dirtyBytes == 0);
	const RgfxSlabs::Range *any = &again;
	CHECK(allBytes(any->contents, 0x20000, 0));
	// Clean memory is taken as it is: a byte the background zeroed and nothing wrote since.
	RgfxSlabs::Range clean;
	CHECK(slabs.take(0x20000, kShared, 0, &clean) && allBytes(clean.contents, 0x20000, 0));
	slabs.give(clean);
	// A source's zeroed slabs are clean from the start.
	Source zeroed(64 * kMiB);
	zeroed.zeroed = true;
	RgfxSlabs z(zeroed);
	RgfxSlabs::Range zr;
	CHECK(z.take(0x1000, kShared, 0, &zr) && z.stats().dirtyBytes == 0);
	z.give(zr);
}

// A class keeps two empty slabs; a third empty one goes: its heap dropped, then its memory given
// back. Without the background thread it goes at once.
void test_slab_empty_slabs_go()
{
	Source source(64 * kMiB);
	RgfxSlabs slabs(source);
	std::vector<RgfxSlabs::Range> held;
	while (source.takes < 4) {
		RgfxSlabs::Range r;
		CHECK(slabs.take(0x20000, kShared, 0, &r));
		held.push_back(r);
	}
	CHECK(held.size() >= 4);
	for (auto &r : held)
		slabs.give(r);
	CHECK(slabs.stats().emptySlabs == 2 && source.heapDrops == 2 && source.gives == 2);
	// Slabs of 256 KiB, 512 KiB, 1 and 2 MiB emptied in turn: the last two go.
	CHECK(source.logEnds({"drop heap", "give 1048576", "drop heap", "give 2097152"}));
	// The kept ones serve the class again, with no new slab.
	RgfxSlabs::Range r;
	CHECK(slabs.take(0x1000, kShared, 0, &r) && source.takes == 4);
	slabs.give(r);
}

// The kept empty slabs go once they sat empty kCacheIdleMs, as a cached slab does, so a process
// at rest holds only the slabs its buffers lie in; one a buffer took from meanwhile stays. The
// background's step lets them go, its heap destroyed and its memory given back.
uint64_t g_testNowMs = 0;
uint64_t testClock(void *) { return g_testNowMs; }

void test_slab_idle_empty_slabs_go()
{
	Source source(64 * kMiB);
	RgfxSlabs slabs(source);
	g_testNowMs = 1000;
	slabs.setClock(&testClock, nullptr);
	auto drain = [&]() {
		while (slabs.step()) {
		}
	};
	// Two slabs of the class, both emptied at 1000 ms; a third holds a buffer.
	std::vector<RgfxSlabs::Range> held;
	while (source.takes < 3) {
		RgfxSlabs::Range r;
		CHECK(slabs.take(0x20000, kShared, 0, &r));
		held.push_back(r);
	}
	RgfxSlabs::Range last = held.back();
	held.pop_back();
	for (auto &r : held)
		slabs.give(r);
	drain();
	CHECK(slabs.stats().emptySlabs == 2 && slabs.stats().slabs == 3 && source.heapDrops == 0);
	// Not yet at 2999 ms.
	g_testNowMs = 1000 + RgfxSlabs::kCacheIdleMs - 1;
	drain();
	CHECK(slabs.stats().emptySlabs == 2 && source.heapDrops == 0);
	// At 3000 ms both go, their heaps destroyed and their memory given back; the slab with a
	// buffer stays.
	g_testNowMs = 1000 + RgfxSlabs::kCacheIdleMs;
	drain();
	CHECK(slabs.stats().emptySlabs == 0 && slabs.stats().slabs == 1);
	CHECK(source.heapDrops == 2 && source.gives == 2 && slabs.stats().usedBytes == last.size);
	// One emptied, then taken from again before its time: it stays, and goes once emptied
	// for its own kCacheIdleMs.
	slabs.give(last);
	g_testNowMs += 1000;
	RgfxSlabs::Range again;
	CHECK(slabs.take(0x1000, kShared, 0, &again) && again.slab == last.slab);
	g_testNowMs += RgfxSlabs::kCacheIdleMs;
	drain();
	CHECK(slabs.stats().slabs == 1 && source.heapDrops == 2);
	slabs.give(again);
	g_testNowMs += RgfxSlabs::kCacheIdleMs - 1;
	drain();
	CHECK(slabs.stats().slabs == 1);
	g_testNowMs += 1;
	drain();
	CHECK(slabs.stats().slabs == 0 && source.heapDrops == 3 && source.gives == 3);
	// A cached dedicated slab goes on the same clock.
	RgfxSlabs::Range big;
	CHECK(slabs.take(3 * kMiB, kShared, 0, &big));
	slabs.give(big);
	drain();
	CHECK(slabs.stats().cachedSlabs == 1);
	g_testNowMs += RgfxSlabs::kCacheIdleMs;
	drain();
	CHECK(slabs.stats().cachedSlabs == 0 && slabs.stats().slabs == 0);
}

// A buffer past 1 MiB takes a slab of its own; once freed the slab is cached, and a later buffer
// of its class that needs at least half of it lies there at offset 0, its bytes zeroed. The cache
// holds a sixteenth of the source, the oldest let go first.
void test_slab_dedicated_cache()
{
	Source source(256 * kMiB); // a cache of 16 MiB
	RgfxSlabs slabs(source);
	RgfxSlabs::Range big;
	CHECK(slabs.take(3 * kMiB, kShared, 0, &big) && big.offset == 0 && big.size == 3 * kMiB);
	std::memset(big.contents, 0xa5, big.size);
	const void *slab = big.slab;
	slabs.give(big);
	CHECK(slabs.stats().cachedSlabs == 1 && source.gives == 0);
	// Too small for it (under half) and of another class: slabs of their own.
	RgfxSlabs::Range small, other;
	CHECK(slabs.take(kMiB + 0x1000, kShared, 0, &small) && small.slab != slab);
	CHECK(slabs.take(3 * kMiB, kSharedUntracked, 0, &other) && other.slab != slab);
	RgfxSlabs::Range reuse;
	CHECK(slabs.take(2 * kMiB, kShared, 0, &reuse) && reuse.slab == slab && reuse.offset == 0);
	CHECK(allBytes(reuse.contents, 2 * kMiB, 0) && slabs.stats().cachedSlabs == 0);
	const uint64_t takes = source.takes;
	slabs.give(reuse);
	slabs.give(small);
	slabs.give(other);
	// 3 + 1.0039 + 3 MiB cached; four 4 MiB slabs more pass 16 MiB: the oldest go.
	std::vector<RgfxSlabs::Range> four(4);
	for (auto &r : four)
		CHECK(slabs.take(4 * kMiB, kSharedUntracked, 0, &r));
	CHECK(source.takes == takes + 4); // the cached Untracked one of 3 MiB holds no 4 MiB
	for (auto &r : four)
		slabs.give(r);
	CHECK(slabs.stats().cachedBytes <= 16 * kMiB && source.gives >= 3);
	// The background zeroes the cached slabs.
	while (slabs.step()) {
	}
	CHECK(slabs.stats().dirtyBytes == 0);
}

// Where the source has no room for a new slab: the process's own empty and cached slabs go and
// the slab is asked again; then a small buffer takes slabs half as large down to its own size.
// Nil only where the source has no room even for that, and a buffer is made again once one goes.
// A slab whose heap the device never got is nil at once, its memory given back.
void test_slab_fallbacks()
{
	{
		Source source(0x40000); // 256 KiB
		RgfxSlabs slabs(source);
		RgfxSlabs::Range a, b;
		CHECK(slabs.take(0x1000, kShared, 0, &a)); // 128 KiB slab
		CHECK(slabs.take(0x1000, kManaged, 0, &b)); // 128 KiB slab: the source is full
		std::vector<RgfxSlabs::Range> more;
		RgfxSlabs::Range r;
		for (int i = 0; i < 31; i++) {
			CHECK(slabs.take(0x1000, kShared, 0, &r));
			more.push_back(r);
		}
		// The Shared slab is full and the source too: nil.
		CHECK(!slabs.take(0x1000, kShared, 0, &r));
		CHECK(source.refusedTakes >= 2);
		// The Managed slab emptied gives the source room again.
		slabs.give(b);
		CHECK(slabs.take(0x1000, kShared, 0, &r) && r.slab != a.slab);
		CHECK(slabs.stats().slabs == 2 && source.gives == 1);
		slabs.give(r);
		for (auto &m : more)
			slabs.give(m);
		slabs.give(a);
	}
	{
		// Room for a 16 KiB slab only: a page takes it, after 256, 128, 64 and 32 KiB were refused.
		Source source(0x24000);
		RgfxSlabs slabs(source);
		RgfxSlabs::Range a, b;
		CHECK(slabs.take(0x20000, kShared, 0, &a) && a.size == 0x20000); // 256 KiB refused, 128 taken
		source.log.clear();
		CHECK(slabs.take(0x1000, kManaged, 0, &b));
		CHECK(source.log == std::vector<std::string>({"refuse 131072", "refuse 131072",
		                                              "refuse 65536", "refuse 32768",
		                                              "take 16384"}));
		// A cached slab goes before a create is refused.
		slabs.give(a);
		slabs.give(b);
		RgfxSlabs::Range big;
		CHECK(slabs.take(0x20000, kSharedUntracked, 0, &big) && slabs.stats().emptySlabs == 0);
		slabs.give(big);
	}
	{
		Source source(64 * kMiB);
		source.refuseHeaps = true;
		RgfxSlabs slabs(source);
		RgfxSlabs::Range r;
		CHECK(!slabs.take(0x1000, kShared, 0, &r));
		CHECK(source.takes == 1 && source.gives == 1 && slabs.stats().slabs == 0);
		// A Managed class makes no heap.
		CHECK(slabs.take(0x1000, kManaged, 0, &r));
		slabs.give(r);
	}
}

// With a first source (the process's own guest RAM: 2 MiB units, no heaps, zeroed pages), a new
// slab comes from it at its unit while it has memory, small and dedicated ones alike, and its
// buffers are its own; once it has none, the region's source makes the slab, heaped as its classes
// are. A slab goes back to the source it came from, and a cached one past a sixteenth of its own
// source's capacity goes at once. Where both refuse, the process's empty slabs go and the first
// source is asked again first.
void test_slab_first_source()
{
	{
		Source region(64 * kMiB);
		Source pages(8 * kMiB, 2 * kMiB);
		pages.heapedClasses = false;
		pages.zeroed = true;
		RgfxSlabs slabs(region);
		slabs.setFirst(&pages);
		RgfxSlabs::Range a, b, big, c;
		CHECK(slabs.take(0x1000, kShared, 0, &a) && pages.takes == 1 && region.takes == 0);
		CHECK(pages.logEnds({"take 2097152"})); // the unit, though a class's first slab is 128 KiB
		RgfxSlabSource::Made made;
		CHECK(slabs.makeBuffer(a, 0x1000, kShared, &made) && made.ownsAddress && pages.heaps == 0);
		std::vector<RgfxSlabs::Range> fill(511);
		for (auto &r : fill)
			CHECK(slabs.take(0x1000, kShared, 0, &r) && r.slab == a.slab);
		CHECK(slabs.take(0x1000, kShared, 0, &b) && b.slab != a.slab && pages.takes == 2);
		CHECK(allBytes(b.contents, b.size, 0) && slabs.stats().dirtyBytes == 0);
		// A dedicated 3 MiB buffer takes two units, the source's last.
		CHECK(slabs.take(3 * kMiB, kShared, 0, &big) && big.size == 3 * kMiB);
		CHECK(pages.takes == 3 && pages.logEnds({"take 4194304"}));
		// The pages are full: the region's source makes the next slab, with its heap.
		CHECK(slabs.take(0x1000, kSharedUntracked, 0, &c));
		CHECK(pages.logEnds({"refuse 2097152"}) && region.takes == 1 && region.heaps == 1);
		CHECK(slabs.makeBuffer(c, 0x1000, kSharedUntracked, &made) && !made.ownsAddress);
		RgfxSlabs::Stats st = slabs.stats();
		CHECK(st.slabs == 4 && st.firstSlabs == 3);
		// The dedicated slab is past a sixteenth of the pages (512 KiB): it goes back to them at
		// once, and the region's source gives nothing.
		slabs.give(big);
		CHECK(pages.gives == 1 && pages.logEnds({"give 4194304"}) && region.gives == 0);
		CHECK(slabs.stats().cachedSlabs == 0);
		slabs.give(c);
		slabs.give(b);
		for (auto &r : fill)
			slabs.give(r);
		slabs.give(a);
		CHECK(region.gives == 0 && slabs.stats().emptySlabs == 3);
	}
	{
		Source region(0x20000); // one 128 KiB slab
		Source pages(4 * kMiB, 2 * kMiB);
		pages.heapedClasses = false;
		RgfxSlabs slabs(region);
		slabs.setFirst(&pages);
		RgfxSlabs::Range a, b, c, d;
		CHECK(slabs.take(0x1000, kShared, 0, &a));
		CHECK(slabs.take(0x1000, kSharedUntracked, 0, &b) && pages.takes == 2);
		CHECK(slabs.take(0x1000, kManaged, 0, &c) && region.takes == 1);
		slabs.give(a); // kept, empty
		// A fourth class (Shared, write-combined): both refuse, the empty slab goes, and the pages
		// make the slab.
		pages.log.clear();
		CHECK(slabs.take(0x1000, kShared | 0x1, 0, &d));
		CHECK(pages.gives == 1 && pages.takes == 3 && region.refusedTakes >= 1);
		CHECK(pages.log == std::vector<std::string>({"refuse 2097152", "give 2097152",
		                                             "take 2097152"}));
		slabs.give(d);
		slabs.give(c);
		slabs.give(b);
	}
}

// With the background thread, a freed slab goes and freed bytes are zeroed on it, and buffers
// made and freed on several threads at once never share a byte.
void test_slab_thread()
{
	Source source(256 * kMiB);
	RgfxSlabs slabs(source);
	CHECK(slabs.start());
	std::vector<uint8_t> owner(source.bytes / 0x1000, 0);
	std::mutex ownerLock;
	std::atomic<int> overlaps{0}, bad{0};
	auto worker = [&](uint8_t id) {
		uint32_t seed = id * 2654435761u;
		std::vector<RgfxSlabs::Range> mine;
		for (int i = 0; i < 3000; i++) {
			seed = seed * 1664525u + 1013904223u;
			if (mine.size() > 40 || (!mine.empty() && (seed >> 28) < 6)) {
				size_t k = (seed >> 8) % mine.size();
				RgfxSlabs::Range r = mine[k];
				mine.erase(mine.begin() + k);
				if (r.contents[0] != id || r.contents[r.size - 1] != id)
					bad++;
				{
					std::lock_guard<std::mutex> hold(ownerLock);
					const uint64_t first = (r.contents - source.base) / 0x1000;
					for (uint64_t p = first; p < first + r.size / 0x1000; p++)
						owner[p] = 0;
				}
				slabs.give(r);
				continue;
			}
			const uint64_t sizes[] = {256, 0x1000, 0x3000, 0x10000, 0x80000, 3 * kMiB};
			const uint64_t length = sizes[(seed >> 12) % 6];
			const uint32_t options = (seed >> 20) & 1 ? kShared : kManaged;
			RgfxSlabs::Range r;
			if (!slabs.take(length, options, 0, &r)) {
				bad++;
				continue;
			}
			if (r.contents[0] != 0 || r.contents[r.size - 1] != 0)
				bad++;
			{
				std::lock_guard<std::mutex> hold(ownerLock);
				const uint64_t first = (r.contents - source.base) / 0x1000;
				for (uint64_t p = first; p < first + r.size / 0x1000; p++) {
					if (owner[p] != 0)
						overlaps++;
					owner[p] = id;
				}
			}
			r.contents[0] = id;
			r.contents[r.size - 1] = id;
			mine.push_back(r);
		}
		for (auto &r : mine) {
			std::lock_guard<std::mutex> hold(ownerLock);
			const uint64_t first = (r.contents - source.base) / 0x1000;
			for (uint64_t p = first; p < first + r.size / 0x1000; p++)
				owner[p] = 0;
			slabs.give(r);
		}
	};
	std::vector<std::thread> threads;
	for (uint8_t id = 1; id <= 4; id++)
		threads.emplace_back(worker, id);
	for (auto &t : threads)
		t.join();
	CHECK(overlaps == 0 && bad == 0);
	// The thread zeroes what was freed and lets the extra slabs go.
	bool settled = false;
	for (int i = 0; i < 2000 && !settled; i++) {
		RgfxSlabs::Stats s = slabs.stats();
		settled = s.dirtyBytes == 0 && s.dooming == 0;
		if (!settled)
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
	CHECK(settled && slabs.stats().usedBytes == 0);
}

// Over the region: a Shared slab is a Placement heap over host memory at its block, each buffer a
// BUFFER_AT_OFFSET there; a Managed one is the block, each buffer a BUFFER_OVER_HOST_MEMORY over
// its own range of it. A slab let go destroys its heap.
void test_region_slab_source()
{
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 0x800000, 0xee);
	TestRing ring(0x8000);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	RgfxRegion region(mem.data(), mem.size());
	RgfxRegionSlabSource source(region, ser);
	CHECK(source.unit() == 0x4000 && source.heaped(kShared) && !source.heaped(kManaged));
	RgfxSlabs slabs(source);
	RgfxSlabs::Range r;
	CHECK(slabs.take(256, kShared, 0, &r));
	auto heap = expectMessage<RgfxHeapOverHostMemory>(ring, RGFX_CMD_HEAP_OVER_HOST_MEMORY);
	CHECK(heap.region_offset == static_cast<uint64_t>(r.contents - mem.data()) &&
	      heap.size_bytes == 0x20000 && heap.options == kShared && heap.heap_type == 1);
	RgfxSlabSource::Made made;
	CHECK(slabs.makeBuffer(r, 256, 0, &made));
	auto at = expectMessage<RgfxBufferAtOffset>(ring, RGFX_CMD_BUFFER_AT_OFFSET);
	CHECK(at.heap_obj_id == heap.obj_id && at.offset == 0 && at.length == 256 && at.options == 0);
	CHECK(made.objId == at.obj_id && made.gpuAddress == heap.gpu_address && !made.ownsAddress);
	RgfxSlabs::Range m;
	CHECK(slabs.take(0x2000, kManaged, 0, &m) && slabs.makeBuffer(m, 0x2000, kManaged, &made));
	auto over = expectMessage<RgfxBufferOverHostMemory>(ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY);
	CHECK(over.region_offset == static_cast<uint64_t>(m.contents - mem.data()) &&
	      over.length == 0x2000 && made.ownsAddress && made.gpuAddress == over.gpu_address);
	slabs.give(r);
	slabs.give(m);
	// Three slabs of another class, two 1 MiB buffers each, emptied: the third goes, its heap
	// destroyed.
	std::vector<RgfxSlabs::Range> held;
	for (int i = 0; i < 6; i++) {
		RgfxSlabs::Range h;
		CHECK(slabs.take(kMiB, kSharedUntracked, 0, &h));
		held.push_back(h);
	}
	CHECK(slabs.stats().slabs == 5);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	while (ring.next(&h, &bytes)) {
	}
	for (auto &r : held)
		slabs.give(r);
	CHECK(ring.next(&h, &bytes) && h.type == RGFX_CMD_OBJECT_DESTROY);
	CHECK(!ring.next(&h, &bytes));
}

// A kext of the test's own: blocks first fit past the floor.
struct Kext {
	RgfxHeapRanges free;
	std::map<uint64_t, uint64_t> sizes; // each block's, by region offset
	int reclaims = 0, reclaimed = 0;
	RgfxSlabs *slabs = nullptr;

	explicit Kext(uint64_t size) : free(size) {}
	static bool alloc(void *context, uint64_t size, uint64_t align, uint64_t *offset)
	{
		Kext *k = static_cast<Kext *>(context);
		if (!k->free.take(size, align, offset))
			return false;
		*offset += RGFX_ALLOCATIONS_OFFSET;
		k->sizes[*offset] = size;
		return true;
	}
	static void release(void *context, uint64_t offset)
	{
		Kext *k = static_cast<Kext *>(context);
		auto it = k->sizes.find(offset);
		if (it == k->sizes.end())
			return;
		k->free.give(offset - RGFX_ALLOCATIONS_OFFSET, it->second);
		k->sizes.erase(it);
	}
	static bool reclaim(void *context)
	{
		Kext *k = static_cast<Kext *>(context);
		k->reclaims++;
		const bool went = k->slabs != nullptr ? k->slabs->reclaim() : false;
		k->reclaimed += went;
		return went;
	}
};

// Where the kext has no room for a block, the region lets the slabs the process keeps go (the
// cached and the empty ones) and asks once more: a block is refused only where nothing the
// process holds could make room.
void test_region_reclaims_slabs()
{
	constexpr uint64_t kRoom = 80 * kMiB; // a cache of 5 MiB
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + kRoom, 0xee);
	TestRing ring(0x8000);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	RgfxRegion region(mem.data(), mem.size());
	Kext kext(kRoom);
	region.setBlockSource(&Kext::alloc, &Kext::release, &kext);
	region.setReclaim(&Kext::reclaim, &kext);
	RgfxRegionSlabSource source(region, ser);
	RgfxSlabs slabs(source);
	kext.slabs = &slabs;
	// A 4 MiB buffer freed stays cached, a 256 KiB one's slab stays empty: 4.5 MiB kept.
	RgfxSlabs::Range big, small;
	CHECK(slabs.take(4 * kMiB, kShared, 0, &big) && slabs.take(256 << 10, kShared, 0, &small));
	slabs.give(big);
	slabs.give(small);
	CHECK(slabs.stats().cachedSlabs == 1 && slabs.stats().emptySlabs == 1);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	while (ring.next(&h, &bytes)) {
	}
	// Room for 3.5 MiB past the rest: a 4.5 MiB block is placed where both slabs lay, once they
	// went, their heaps destroyed.
	RgfxRegion::Block filler = region.allocate(72 * kMiB);
	CHECK(filler.ok && kext.reclaims == 0);
	RgfxRegion::Block block = region.allocate(4 * kMiB + (512 << 10));
	CHECK(block.ok && kext.reclaims == 1 && kext.reclaimed == 1);
	CHECK(slabs.stats().slabs == 0 && slabs.stats().cachedSlabs == 0);
	int destroys = 0;
	while (ring.next(&h, &bytes))
		destroys += h.type == RGFX_CMD_OBJECT_DESTROY;
	CHECK(destroys == 2);
	// Nothing left to give back: refused after one reclaim that let nothing go.
	CHECK(!region.allocate(4 * kMiB).ok && kext.reclaims == 2 && kext.reclaimed == 1);
	region.release(block.offset);
	// The slabs' own blocks ask through the region too: a buffer past the room is nil.
	RgfxSlabs::Range none;
	CHECK(!slabs.take(16 * kMiB, kShared, 0, &none));
}

// A kext of the test's own for the pages source: region blocks and pages blocks (16 KiB, which
// hold the runs) first fit from one room past the floor, each release logged.
struct PagesKext {
	RgfxHeapRanges free;
	std::map<uint64_t, uint64_t> blocks; // region blocks, by offset
	std::set<uint64_t> pages;            // pages blocks, by offset
	std::vector<std::string> *log;
	bool refusePages = false;
	const void *lastAddress = nullptr;
	uint64_t lastLength = 0;
	bool lastWritable = false;

	PagesKext(uint64_t size, std::vector<std::string> *events) : free(size), log(events) {}
	static bool alloc(void *context, uint64_t size, uint64_t align, uint64_t *offset)
	{
		PagesKext *k = static_cast<PagesKext *>(context);
		if (!k->free.take(size, align, offset))
			return false;
		*offset += RGFX_ALLOCATIONS_OFFSET;
		k->blocks[*offset] = size;
		return true;
	}
	static bool pagesBlock(void *context, const void *address, uint64_t length, bool writable,
	                       uint64_t *offset, uint64_t *size)
	{
		PagesKext *k = static_cast<PagesKext *>(context);
		k->lastAddress = address;
		k->lastLength = length;
		k->lastWritable = writable;
		if (k->refusePages || !k->free.take(0x4000, 0x4000, offset))
			return false;
		*offset += RGFX_ALLOCATIONS_OFFSET;
		*size = 0x4000;
		k->pages.insert(*offset);
		return true;
	}
	static void release(void *context, uint64_t offset)
	{
		PagesKext *k = static_cast<PagesKext *>(context);
		if (k->pages.erase(offset) != 0) {
			k->free.give(offset - RGFX_ALLOCATIONS_OFFSET, 0x4000);
			k->log->push_back("release pages");
			return;
		}
		auto it = k->blocks.find(offset);
		if (it == k->blocks.end())
			return;
		k->free.give(offset - RGFX_ALLOCATIONS_OFFSET, it->second);
		k->blocks.erase(it);
	}
};

// The kernel's memory for the pages source: `superpagesLeft` superpages of contiguous room, then
// KERN_NO_SPACE; past the wire budget (`budgetSpent`), KERN_RESOURCE_SHORTAGE. Plain pages
// (`pages`) where `refusePlain` is false, else KERN_NO_SPACE. And the guest's superpage record
// (`record`), the kext's own (RgfxSuperpageRecord.hpp) over `none`, which every process shares.
struct PagesVm {
	std::map<uint8_t *, std::vector<uint8_t>> held;
	uint64_t superpagesLeft = 0;
	bool budgetSpent = false;
	bool refusePlain = false;
	bool none = false;
	std::vector<std::string> *log;

	explicit PagesVm(std::vector<std::string> *events) : log(events) {}
	static int superpages(void *context, uint64_t size, uint8_t **bytes)
	{
		PagesVm *vm = static_cast<PagesVm *>(context);
		const std::string what = "superpages " + std::to_string(size);
		if (vm->budgetSpent || size / (2 * kMiB) > vm->superpagesLeft) {
			vm->log->push_back(what + " refused");
			return vm->budgetSpent ? KERN_RESOURCE_SHORTAGE : KERN_NO_SPACE;
		}
		vm->superpagesLeft -= size / (2 * kMiB);
		vm->log->push_back(what);
		std::vector<uint8_t> memory(size, 0);
		*bytes = memory.data();
		vm->held[*bytes] = std::move(memory);
		return KERN_SUCCESS;
	}
	static int pages(void *context, uint64_t size, uint8_t **bytes)
	{
		PagesVm *vm = static_cast<PagesVm *>(context);
		const std::string what = "pages " + std::to_string(size);
		if (vm->refusePlain) {
			vm->log->push_back(what + " refused");
			return KERN_NO_SPACE;
		}
		vm->log->push_back(what);
		std::vector<uint8_t> memory(size, 0);
		*bytes = memory.data();
		vm->held[*bytes] = std::move(memory);
		vm->plain.insert(*bytes);
		return KERN_SUCCESS;
	}
	static void unmap(void *context, uint8_t *bytes, uint64_t size)
	{
		PagesVm *vm = static_cast<PagesVm *>(context);
		vm->held.erase(bytes);
		if (vm->plain.erase(bytes) == 0)
			vm->superpagesLeft += size / (2 * kMiB);
		vm->log->push_back("unmap " + std::to_string(size));
	}
	static bool record(void *context, RgfxPagesMemory::Note note)
	{
		PagesVm *vm = static_cast<PagesVm *>(context);
		const int answer = rgfxSuperpagesNote(&vm->none, note);
		static const char *const kNotes[] = {"ask", "refused", "given back"};
		vm->log->push_back(std::string("record ") + kNotes[note] + " " + std::to_string(answer));
		return answer != 0;
	}
	std::set<uint8_t *> plain; // what pages made
};

bool endsWith(const std::vector<std::string> &log, const std::vector<std::string> &want)
{
	return log.size() >= want.size() && std::equal(want.begin(), want.end(), log.end() - want.size());
}

// The process's own pages as the first source: a heaped class's slab is the 2 MiB superpage the
// slabs' thread took ahead (prepare, a piece of its step), wired into a pages block of the kext's,
// made one HEAP_OVER_PAGES its buffers are placed in (BUFFER_AT_OFFSET, and
// BUFFER_MANAGED_AT_OFFSET where the device took MANAGED_AT_OFFSET); the thread takes the next
// after each slab of the source, and a take asks the kernel for none. A dedicated Shared slab
// larger than a superpage is plain pages with no heap, its buffer a BUFFER_OVER_PAGES over the
// block's view. A larger Managed one, a take with no superpage held and no plain pages to be had,
// and a kext that gives no block (the superpage given back) are the region's slabs, as are
// Managed buffers without MANAGED_AT_OFFSET. A slab's heap is destroyed, then its block released,
// then its pages go.
void test_pages_slab_source()
{
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 16 * kMiB, 0xee);
	TestRing ring(0x10000);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	RgfxRegion region(mem.data(), mem.size());
	std::vector<std::string> log;
	PagesKext kext(16 * kMiB, &log);
	region.setBlockSource(&PagesKext::alloc, &PagesKext::release, &kext);
	region.setPagesBlockSource(&PagesKext::pagesBlock, &kext);
	PagesVm vm(&log);
	vm.superpagesLeft = 3;
	RgfxPagesMemory calls;
	calls.context = &vm;
	calls.superpages = &PagesVm::superpages;
	calls.pages = &PagesVm::pages;
	calls.unmap = &PagesVm::unmap;
	RgfxRegionSlabSource regionSource(region, ser);
	RgfxPagesSlabSource pages(region, ser, calls, 16ull << 30);
	CHECK(pages.unit() == 2 * kMiB && pages.heaped(kShared) && !pages.heaped(kManaged));
	CHECK(pages.serves(kShared, 2 * kMiB) && pages.serves(kSharedUntracked, 2 * kMiB));
	CHECK(pages.serves(kShared, 4 * kMiB) && !pages.serves(kManaged, 2 * kMiB));
	RgfxSlabs slabs(regionSource);
	slabs.setFirst(&pages);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;

	// The thread's first piece of work: a superpage ahead, which the first take uses.
	CHECK(slabs.step() && pages.holdsSuperpage());
	CHECK(log == std::vector<std::string>({"superpages 2097152"}) && kext.pages.empty());
	RgfxSlabs::Range a, b;
	CHECK(slabs.take(256, kShared, 0, &a) && a.offset == 0 && a.size == 0x1000);
	CHECK(log.size() == 1 && kext.pages.size() == 1 && !pages.holdsSuperpage());
	CHECK(kext.lastAddress == a.contents && kext.lastLength == 2 * kMiB && kext.lastWritable);
	const uint64_t block = *kext.pages.begin();
	auto heap = expectMessage<RgfxHeapOverPages>(ring, RGFX_CMD_HEAP_OVER_PAGES);
	CHECK(heap.block_offset == block && heap.block_size == 0x4000);
	CHECK(heap.size_bytes == 2 * kMiB && heap.options == kShared && heap.heap_type == 1);
	CHECK(heap.gpu_address != 0);
	CHECK(slabs.take(0x3000, kShared, 0, &b) && b.slab == a.slab && b.offset == 0x1000);
	CHECK(allBytes(b.contents, b.size, 0));
	RgfxSlabSource::Made made;
	CHECK(slabs.makeBuffer(b, 0x2800, kShared, &made));
	auto at = expectMessage<RgfxBufferAtOffset>(ring, RGFX_CMD_BUFFER_AT_OFFSET);
	CHECK(at.heap_obj_id == heap.obj_id && at.offset == 0x1000 && at.length == 0x2800);
	CHECK(made.objId == at.obj_id && made.gpuAddress == heap.gpu_address + 0x1000);
	CHECK(!made.ownsAddress);
	// The next superpage, taken ahead after that slab; one held, nothing more to do.
	CHECK(slabs.step() && log == std::vector<std::string>({"superpages 2097152",
	                                                       "superpages 2097152"}));
	CHECK(!slabs.step() && log.size() == 2 && pages.holdsSuperpage());
	// A dedicated buffer of up to a superpage takes the one held, with its heap; a larger one
	// plain pages with no heap, the kernel asked for no superpage, its buffer over the block's view.
	RgfxSlabs::Range one, big;
	CHECK(slabs.take(3 * kMiB / 2, kShared, 0, &one) && one.size == 3 * kMiB / 2);
	CHECK(log.size() == 2 && kext.pages.size() == 2 && !pages.holdsSuperpage());
	CHECK(expectMessage<RgfxHeapOverPages>(ring, RGFX_CMD_HEAP_OVER_PAGES).size_bytes == 2 * kMiB);
	CHECK(slabs.take(3 * kMiB, kShared, 0, &big) && big.size == 3 * kMiB && big.offset == 0);
	CHECK(log.size() == 3 && endsWith(log, {"pages 4194304"}) && kext.pages.size() == 3);
	CHECK(kext.lastAddress == big.contents && kext.lastLength == 4 * kMiB);
	CHECK(!ring.next(&h, &bytes));
	CHECK(slabs.makeBuffer(big, 3 * kMiB, kShared, &made));
	auto bigOver = expectMessage<RgfxBufferOverPages>(ring, RGFX_CMD_BUFFER_OVER_PAGES);
	CHECK(bigOver.block_offset == *kext.pages.rbegin() && bigOver.block_size == 0x4000);
	CHECK(bigOver.offset == 0 && bigOver.length == 3 * kMiB && bigOver.options == kShared);
	CHECK(made.objId == bigOver.obj_id && made.gpuAddress == bigOver.gpu_address);
	CHECK(made.ownsAddress && made.gpuAddress != 0);
	const size_t asked = log.size();
	// Managed, with no MANAGED_AT_OFFSET: the region's slab, unasked.
	RgfxSlabs::Range m;
	CHECK(slabs.take(0x1000, kManaged, 0, &m) && slabs.makeBuffer(m, 0x1000, kManaged, &made));
	CHECK(log.size() == asked);
	CHECK(expectMessage<RgfxBufferOverHostMemory>(ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY).length ==
	      0x1000);
	// The last superpage, taken ahead, then none: the thread's ask refused, and the next take
	// plain pages at once; with no plain pages either, the region's heaped slab. (The thread's
	// work runs until it has none: the region's slab's bytes are zeroed first.)
	RgfxSlabs::Range c, d, d2;
	while (slabs.step()) {
	}
	CHECK(endsWith(log, {"superpages 2097152"}) && pages.holdsSuperpage());
	CHECK(slabs.take(0x1000, kSharedUntracked, 0, &c) && log.size() == asked + 1);
	CHECK(!slabs.step() && endsWith(log, {"superpages 2097152 refused"}));
	CHECK(slabs.take(0x1000, kShared | 0x1, 0, &d) && endsWith(log, {"pages 2097152"}));
	vm.refusePlain = true;
	CHECK(slabs.take(0x1000, kShared | 0x2, 0, &d2) && endsWith(log, {"pages 2097152 refused"}));
	vm.refusePlain = false;
	while (ring.next(&h, &bytes)) {
	}
	CHECK(h.type == RGFX_CMD_HEAP_OVER_HOST_MEMORY);
	CHECK(slabs.stats().slabs == 7 && slabs.stats().firstSlabs == 5);
	// Past the wire budget: the thread's ask refused, and the take plain pages.
	vm.superpagesLeft = 1;
	vm.budgetSpent = true;
	RgfxSlabs::Range e;
	while (slabs.step()) {
	}
	CHECK(endsWith(log, {"superpages 2097152 refused"}) && !pages.holdsSuperpage());
	CHECK(slabs.take(0x1000, kSharedUntracked | 0x1, 0, &e) && endsWith(log, {"pages 2097152"}));
	vm.budgetSpent = false;
	// No block from the kext: the superpage held goes back, and the region's slab.
	while (slabs.step()) {
	}
	CHECK(endsWith(log, {"superpages 2097152"}) && pages.holdsSuperpage());
	kext.refusePages = true;
	log.clear();
	RgfxSlabs::Range f;
	CHECK(slabs.take(0x1000, kSharedUntracked | 0x2, 0, &f));
	CHECK(log == std::vector<std::string>({"unmap 2097152"}) && !pages.holdsSuperpage());
	CHECK(vm.held.size() == 6 && slabs.stats().firstSlabs == 6);
	kext.refusePages = false;
	while (ring.next(&h, &bytes)) {
	}
	// Every slab let go: each heap destroyed, its pages block released, then its pages.
	for (RgfxSlabs::Range *r : {&a, &b, &one, &big, &m, &c, &d, &d2, &e, &f})
		slabs.give(*r);
	log.clear();
	CHECK(slabs.reclaim() && slabs.stats().slabs == 0 && kext.pages.empty() && vm.held.empty());
	CHECK(log.size() == 12 && std::count(log.begin(), log.end(), "unmap 4194304") == 1);
	for (size_t i = 0; i < log.size(); i += 2)
		CHECK(log[i] == "release pages" && log[i + 1].rfind("unmap ", 0) == 0);
	int destroys = 0;
	while (ring.next(&h, &bytes))
		destroys += h.type == RGFX_CMD_OBJECT_DESTROY;
	CHECK(destroys == 7); // every slab's heap: none for the Managed plain block or the large one

	// With MANAGED_AT_OFFSET, a Managed slab is a superpage too, its buffers' CPU copies placed
	// in its heap.
	RgfxPagesSlabSource managed(region, ser, calls, 16ull << 30, true);
	CHECK(managed.serves(kManaged, 2 * kMiB) && !managed.serves(kManaged, 4 * kMiB));
	CHECK(managed.heaped(kManaged));
	RgfxRegionSlabSource managedRegion(region, ser, true);
	RgfxSlabs both(managedRegion);
	both.setFirst(&managed);
	RgfxSlabs::Range g;
	CHECK(both.step() && endsWith(log, {"superpages 2097152"}));
	CHECK(both.take(0x1000, kManaged, 0, &g) && !managed.holdsSuperpage());
	auto mheap = expectMessage<RgfxHeapOverPages>(ring, RGFX_CMD_HEAP_OVER_PAGES);
	CHECK(mheap.options == kShared && mheap.size_bytes == 2 * kMiB);
	CHECK(both.makeBuffer(g, 0x800, kManaged, &made));
	auto mat = expectMessage<RgfxBufferManagedAtOffset>(ring, RGFX_CMD_BUFFER_MANAGED_AT_OFFSET);
	CHECK(mat.heap_obj_id == mheap.obj_id && mat.offset == 0 && mat.length == 0x800);
	CHECK(made.ownsAddress && made.gpuAddress == mat.gpu_address);
	// A Managed buffer past a superpage: the region's heaped slab, the kernel never asked.
	const size_t before = log.size();
	RgfxSlabs::Range bigManaged;
	CHECK(both.take(3 * kMiB, kManaged, 0, &bigManaged) && log.size() == before);
	CHECK(expectMessage<RgfxHeapOverHostMemory>(ring, RGFX_CMD_HEAP_OVER_HOST_MEMORY).size_bytes >=
	      3 * kMiB);
	both.give(g);
	both.give(bigManaged);
	CHECK(both.reclaim() && kext.pages.empty() && vm.held.empty());
}

// Where the kernel has no superpage left (it runs out minutes after a boot), the slabs' thread's
// ask is refused, and the pages source takes plain pages: a heaped class's 2 MiB slab is a pages
// block of them, one HEAP_OVER_PAGES its buffers are placed in, as over a superpage (the host
// makes the heap over the block's view); a dedicated Shared slab larger than 2 MiB is plain pages,
// as at any time, its buffer a BUFFER_OVER_PAGES over the view. Only where the kernel gives
// neither is the slab the region's. Plain pages go back as superpages do, after the block's
// release.
void test_pages_slab_source_plain_pages()
{
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 16 * kMiB, 0xee);
	TestRing ring(0x10000);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	RgfxRegion region(mem.data(), mem.size());
	std::vector<std::string> log;
	PagesKext kext(16 * kMiB, &log);
	region.setBlockSource(&PagesKext::alloc, &PagesKext::release, &kext);
	region.setPagesBlockSource(&PagesKext::pagesBlock, &kext);
	PagesVm vm(&log);
	vm.superpagesLeft = 0;
	RgfxPagesMemory calls;
	calls.context = &vm;
	calls.superpages = &PagesVm::superpages;
	calls.pages = &PagesVm::pages;
	calls.unmap = &PagesVm::unmap;
	RgfxRegionSlabSource regionSource(region, ser);
	RgfxPagesSlabSource pages(region, ser, calls, 16ull << 30);
	RgfxSlabs slabs(regionSource);
	slabs.setFirst(&pages);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;

	RgfxSlabs::Range a, b;
	CHECK(!slabs.step() && log == std::vector<std::string>({"superpages 2097152 refused"}));
	log.clear();
	CHECK(slabs.take(256, kShared, 0, &a) && a.offset == 0 && a.size == 0x1000);
	CHECK(log == std::vector<std::string>({"pages 2097152"}));
	CHECK(kext.pages.size() == 1 && kext.lastAddress == a.contents);
	CHECK(kext.lastLength == 2 * kMiB && kext.lastWritable);
	auto heap = expectMessage<RgfxHeapOverPages>(ring, RGFX_CMD_HEAP_OVER_PAGES);
	CHECK(heap.block_offset == *kext.pages.begin() && heap.size_bytes == 2 * kMiB);
	CHECK(slabs.take(0x3000, kShared, 0, &b) && b.slab == a.slab && b.offset == 0x1000);
	RgfxSlabSource::Made made;
	CHECK(slabs.makeBuffer(b, 0x2800, kShared, &made));
	auto at = expectMessage<RgfxBufferAtOffset>(ring, RGFX_CMD_BUFFER_AT_OFFSET);
	CHECK(at.heap_obj_id == heap.obj_id && at.offset == 0x1000 && !made.ownsAddress);
	// A dedicated Shared buffer past 2 MiB: plain pages with no heap, its buffer over the view.
	RgfxSlabs::Range big;
	CHECK(slabs.take(3 * kMiB, kShared, 0, &big) && big.size == 3 * kMiB);
	CHECK(log == std::vector<std::string>({"pages 2097152", "pages 4194304"}));
	CHECK(kext.pages.size() == 2 && kext.lastAddress == big.contents);
	CHECK(!ring.next(&h, &bytes));
	CHECK(slabs.makeBuffer(big, 3 * kMiB, kShared, &made));
	auto over = expectMessage<RgfxBufferOverPages>(ring, RGFX_CMD_BUFFER_OVER_PAGES);
	CHECK(over.block_offset == *kext.pages.rbegin() && over.length == 3 * kMiB);
	CHECK(made.ownsAddress && made.gpuAddress == over.gpu_address);
	// Neither superpages nor plain pages: the region's heaped slab.
	vm.refusePlain = true;
	RgfxSlabs::Range c;
	CHECK(slabs.take(0x1000, kSharedUntracked, 0, &c));
	CHECK(endsWith(log, {"pages 4194304", "pages 2097152 refused"}));
	CHECK(expectMessage<RgfxHeapOverHostMemory>(ring, RGFX_CMD_HEAP_OVER_HOST_MEMORY).size_bytes >=
	      0x1000);
	vm.refusePlain = false;
	// Let go: each pages block released, then its plain pages unmapped.
	for (RgfxSlabs::Range *r : {&a, &b, &big, &c})
		slabs.give(*r);
	log.clear();
	CHECK(slabs.reclaim() && slabs.stats().slabs == 0 && kext.pages.empty() && vm.held.empty());
	CHECK(log == std::vector<std::string>(
	                 {"release pages", "unmap 2097152", "release pages", "unmap 4194304"}) ||
	      log == std::vector<std::string>(
	                 {"release pages", "unmap 4194304", "release pages", "unmap 2097152"}));
	CHECK(vm.superpagesLeft == 0);
}

// The kernel's VM as xnu 26.4 and later answers RgfxSuperpages's calls, in 2 MiB pieces (read from
// the public xnu source, vm_map_enter and vm_map_delete): `free` superpages of contiguous RAM. A
// call for superpages makes them one by one; at one it cannot make it takes down the whole range
// it was asked for, the part it never made among it. Taking down a piece that is not mapped, there
// or in any deallocation, is the guard exception (DEALLOC_GAP) a process with its VM guard on dies
// of, here counted in `gaps`. A fixed call is refused where anything lies in its range. Another
// thread may take a hole the moment it is made (`foreignAt`): a piece of the test's own there.
struct GuardedVm {
	enum Piece { kReserved, kSuperpage, kForeign };
	std::map<uint64_t, Piece> pieces; // mapped 2 MiB pieces, by address
	uint64_t next = 0x200000000ull;   // where anywhere goes next
	uint64_t free = 0;
	uint64_t gaps = 0;
	uint64_t foreignAt = 0; // a hole made here is another thread's at once
	bool refuseReserve = false;
	std::vector<std::string> log;

	// Takes [at, at + size) down, a gap counted once where any piece of it is unmapped.
	void takeDown(uint64_t at, uint64_t size)
	{
		bool gap = false;
		for (uint64_t a = at; a < at + size; a += 2 * kMiB) {
			auto it = pieces.find(a);
			if (it == pieces.end()) {
				gap = true;
				continue;
			}
			if (it->second == kSuperpage)
				free++;
			pieces.erase(it);
		}
		gaps += gap;
	}
	uint64_t anywhere(uint64_t size)
	{
		const uint64_t at = next;
		next += size + 2 * kMiB;
		return at;
	}
	static int superpages(void *context, uint64_t *at, uint64_t size, bool fixed)
	{
		GuardedVm *vm = static_cast<GuardedVm *>(context);
		vm->log.push_back("superpages " + std::to_string(size) + (fixed ? " fixed" : ""));
		const uint64_t start = fixed ? *at : vm->anywhere(size);
		if (fixed)
			for (uint64_t a = start; a < start + size; a += 2 * kMiB)
				if (vm->pieces.count(a))
					return KERN_NO_SPACE;
		for (uint64_t a = start; a < start + size; a += 2 * kMiB) {
			if (vm->free == 0) {
				// The piece's entry is in, with no superpage under it; then [start, end) goes.
				vm->pieces[a] = kReserved;
				vm->takeDown(start, size);
				return KERN_NO_SPACE;
			}
			vm->free--;
			vm->pieces[a] = kSuperpage;
		}
		*at = start;
		return KERN_SUCCESS;
	}
	static int reserve(void *context, uint64_t *at, uint64_t size)
	{
		GuardedVm *vm = static_cast<GuardedVm *>(context);
		vm->log.push_back("reserve " + std::to_string(size));
		if (vm->refuseReserve)
			return KERN_NO_SPACE;
		*at = vm->anywhere(size);
		for (uint64_t a = *at; a < *at + size; a += 2 * kMiB)
			vm->pieces[a] = kReserved;
		return KERN_SUCCESS;
	}
	static void deallocate(void *context, uint64_t at, uint64_t size)
	{
		GuardedVm *vm = static_cast<GuardedVm *>(context);
		vm->log.push_back("deallocate " + std::to_string(size));
		vm->takeDown(at, size);
		if (vm->foreignAt != 0 && at <= vm->foreignAt && vm->foreignAt < at + size)
			vm->pieces[vm->foreignAt] = kForeign;
	}
	RgfxSuperpageVm calls()
	{
		RgfxSuperpageVm vm;
		vm.context = this;
		vm.superpages = &superpages;
		vm.reserve = &reserve;
		vm.deallocate = &deallocate;
		return vm;
	}
	bool runOfSuperpages(uint64_t at, uint64_t size) const
	{
		for (uint64_t a = at; a < at + size; a += 2 * kMiB) {
			auto it = pieces.find(a);
			if (it == pieces.end() || it->second != kSuperpage)
				return false;
		}
		return true;
	}
};

// Superpages are one kernel call each, so a refusal never takes down memory that is not mapped: a
// slab of several reserves its range and makes each superpage in the hole its piece leaves, and
// one the kernel cannot make gives back the superpages before it and the reservation after it,
// with no gap, and never touches the hole, which another thread may hold. One superpage is one
// call anywhere.
void test_superpages_one_call_each()
{
	GuardedVm vm;
	const RgfxSuperpageVm calls = vm.calls();
	uint8_t *bytes = nullptr;
	// Room for one of three: refused with no gap, nothing left mapped, the one made given back.
	vm.free = 1;
	CHECK(rgfxTakeSuperpages(calls, 6 * kMiB, &bytes) == KERN_NO_SPACE);
	CHECK(vm.gaps == 0);
	CHECK(vm.pieces.empty() && vm.free == 1);
	CHECK(vm.log == std::vector<std::string>({"reserve 6291456", "deallocate 2097152",
	                                          "superpages 2097152 fixed", "deallocate 2097152",
	                                          "superpages 2097152 fixed", "deallocate 2097152",
	                                          "deallocate 2097152"}));
	// Room for all three: one reservation, three superpages in it, at a 2 MiB boundary.
	vm.free = 10;
	vm.log.clear();
	CHECK(rgfxTakeSuperpages(calls, 6 * kMiB, &bytes) == KERN_SUCCESS);
	const uint64_t at = reinterpret_cast<uintptr_t>(bytes);
	CHECK(at % (2 * kMiB) == 0 && vm.runOfSuperpages(at, 6 * kMiB) && vm.pieces.size() == 3);
	CHECK(vm.log == std::vector<std::string>(
	                    {"reserve 6291456", "deallocate 2097152", "superpages 2097152 fixed",
	                     "deallocate 2097152", "superpages 2097152 fixed", "deallocate 2097152",
	                     "superpages 2097152 fixed"}));
	CHECK(vm.free == 7 && vm.gaps == 0);
	rgfxGiveSuperpages(calls, bytes, 6 * kMiB);
	CHECK(vm.pieces.empty() && vm.free == 10 && vm.gaps == 0);
	// Another thread takes the second piece's hole: refused there, the superpage before it and the
	// reservation after it given back, the other thread's piece left alone.
	vm.foreignAt = vm.next + 2 * kMiB;
	CHECK(rgfxTakeSuperpages(calls, 6 * kMiB, &bytes) == KERN_NO_SPACE);
	CHECK(vm.gaps == 0 && vm.free == 10);
	CHECK(vm.pieces.size() == 1 && vm.pieces.begin()->first == vm.foreignAt &&
	      vm.pieces.begin()->second == GuardedVm::kForeign);
	vm.pieces.clear();
	vm.foreignAt = 0;
	// None at all: refused at the first, no gap.
	vm.free = 0;
	CHECK(rgfxTakeSuperpages(calls, 4 * kMiB, &bytes) == KERN_NO_SPACE);
	CHECK(vm.pieces.empty() && vm.gaps == 0);
	// One superpage: one call anywhere, refused alike with none to give.
	vm.log.clear();
	CHECK(rgfxTakeSuperpages(calls, 2 * kMiB, &bytes) == KERN_NO_SPACE);
	vm.free = 1;
	CHECK(rgfxTakeSuperpages(calls, 2 * kMiB, &bytes) == KERN_SUCCESS);
	CHECK(vm.runOfSuperpages(reinterpret_cast<uintptr_t>(bytes), 2 * kMiB) && vm.free == 0);
	CHECK(vm.log == std::vector<std::string>({"superpages 2097152", "superpages 2097152"}));
	rgfxGiveSuperpages(calls, bytes, 2 * kMiB);
	CHECK(vm.pieces.empty() && vm.gaps == 0);
	// No reservation: its refusal, and no superpage asked.
	vm.free = 4;
	vm.refuseReserve = true;
	vm.log.clear();
	CHECK(rgfxTakeSuperpages(calls, 4 * kMiB, &bytes) == KERN_NO_SPACE);
	CHECK(vm.log == std::vector<std::string>({"reserve 4194304"}) && vm.pieces.empty());
	vm.refuseReserve = false;
	// A size that is not whole superpages: refused, uncalled.
	vm.log.clear();
	CHECK(rgfxTakeSuperpages(calls, 0, &bytes) == KERN_INVALID_ARGUMENT);
	CHECK(rgfxTakeSuperpages(calls, 3 * kMiB, &bytes) == KERN_INVALID_ARGUMENT);
	CHECK(vm.log.empty() && vm.gaps == 0);
}

// The guest's superpage record, which every process shares (the kext's, RgfxSuperpageRecord.hpp):
// a process's slabs' thread asks the kernel for its superpage ahead only while it is open. A
// superpage the kernel refuses (KERN_NO_SPACE, its search of the whole RAM, 140 to 190 ms on an
// 8 GiB guest once none are left) closes it, and from then on every process's thread asks the
// record alone, and its takes take plain pages. A slab of superpages given back to the kernel opens
// it, and the next ask goes to the kernel again. A refusal past the wire budget leaves it open, and
// plain pages given back tell it nothing.
void test_pages_source_superpage_record()
{
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 16 * kMiB, 0xee);
	TestRing ring(0x10000);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	RgfxRegion region(mem.data(), mem.size());
	std::vector<std::string> log;
	PagesKext kext(16 * kMiB, &log);
	region.setBlockSource(&PagesKext::alloc, &PagesKext::release, &kext);
	region.setPagesBlockSource(&PagesKext::pagesBlock, &kext);
	PagesVm vm(&log);
	vm.superpagesLeft = 1;
	RgfxPagesMemory calls;
	calls.context = &vm;
	calls.superpages = &PagesVm::superpages;
	calls.pages = &PagesVm::pages;
	calls.unmap = &PagesVm::unmap;
	calls.record = &PagesVm::record;
	// Two processes of the guest: each its own slabs over its own pages source, one kernel and
	// one record between them.
	RgfxRegionSlabSource regionSource(region, ser);
	RgfxPagesSlabSource pagesA(region, ser, calls, 16ull << 30);
	RgfxPagesSlabSource pagesB(region, ser, calls, 16ull << 30);
	RgfxSlabs a(regionSource), b(regionSource);
	a.setFirst(&pagesA);
	b.setFirst(&pagesB);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	auto drain = [&] {
		while (ring.next(&h, &bytes)) {
		}
	};

	// The last superpage, to process A's thread, which A's first slab uses.
	RgfxSlabs::Range a1, b1, a2, b2, b3;
	CHECK(a.step() && log == std::vector<std::string>({"record ask 1", "superpages 2097152"}));
	CHECK(a.take(0x1000, kShared, 0, &a1) && log.size() == 2);
	// None left: process B's thread's ask is refused, which closes the record; B's slab is plain
	// pages.
	log.clear();
	CHECK(!b.step());
	CHECK(log == std::vector<std::string>(
	                 {"record ask 1", "superpages 2097152 refused", "record refused 0"}));
	CHECK(b.take(0x1000, kShared, 0, &b1));
	CHECK(endsWith(log, {"pages 2097152"}) && vm.none && kext.pages.size() == 2);
	// Closed: each thread asks the record and not the kernel, and A's next slab, and B's large
	// one, are plain pages.
	log.clear();
	CHECK(!a.step() && !b.step());
	CHECK(a.take(0x1000, kSharedUntracked, 0, &a2));
	CHECK(b.take(3 * kMiB, kShared, 0, &b2) && b2.size == 3 * kMiB);
	CHECK(log == std::vector<std::string>(
	                 {"record ask 0", "record ask 0", "pages 2097152", "pages 4194304"}));
	CHECK(kext.pages.size() == 4 && a.stats().firstSlabs == 2 && b.stats().firstSlabs == 2);
	drain();
	// Process A's superpage slab let go: back to the kernel, which opens the record.
	a.give(a1);
	log.clear();
	CHECK(a.reclaim());
	CHECK(log == std::vector<std::string>(
	                 {"release pages", "unmap 2097152", "record given back 1"}));
	CHECK(!vm.none && vm.superpagesLeft == 1);
	// Open again: B's thread asks, and takes the superpage A gave back, which B's next slab uses.
	log.clear();
	CHECK(b.step() && log == std::vector<std::string>({"record ask 1", "superpages 2097152"}));
	CHECK(b.take(0x1000, kShared | 0x1, 0, &b3) && log.size() == 2);
	CHECK(vm.superpagesLeft == 0 && kext.pages.size() == 4);
	// Past the wire budget: refused, not for want of contiguous RAM, the record left open, and A's
	// next slab plain pages.
	vm.superpagesLeft = 1;
	vm.budgetSpent = true;
	RgfxSlabs::Range a3;
	log.clear();
	CHECK(!b.step());
	CHECK(log == std::vector<std::string>({"record ask 1", "superpages 2097152 refused"}));
	CHECK(!vm.none);
	CHECK(a.take(0x1000, kShared | 0x2, 0, &a3) && endsWith(log, {"pages 2097152"}));
	vm.budgetSpent = false;
	drain();
	// Plain pages given back tell the record nothing; B's superpage given back opens it again.
	vm.none = true;
	for (RgfxSlabs::Range *r : {&a2, &a3})
		a.give(*r);
	log.clear();
	CHECK(a.reclaim() && vm.none);
	CHECK(std::count(log.begin(), log.end(), "record given back 1") == 0);
	for (RgfxSlabs::Range *r : {&b1, &b2, &b3})
		b.give(*r);
	log.clear();
	CHECK(b.reclaim() && !vm.none);
	CHECK(std::count(log.begin(), log.end(), "record given back 1") == 1);
	CHECK(kext.pages.empty() && vm.held.empty());
	drain();
}

// The kernel's superpage ask on the slabs' thread alone, never on a thread that takes a slab, as a
// drawing thread does: the kernel's search for a contiguous 2 MiB runs on the thread that asks, 140
// to 190 ms on an 8 GiB guest once none are left, refused or not. With the slabs' thread running,
// takes on the test's thread, of a slab of one superpage and of a larger one: every ask the kernel
// saw came from one thread, not the test's; a take of a superpage's slab used the one held, and a
// larger one, or one after a refusal, took plain pages at once. The kernel, the kext and the record
// are called on both threads, so the test's fakes of them take a lock.
struct LockedPages {
	std::mutex lock;
	std::vector<std::string> log;
	PagesKext kext{16 * kMiB, &log};
	PagesVm vm{&log};
	std::set<std::thread::id> askers;

	static bool alloc(void *c, uint64_t size, uint64_t align, uint64_t *offset)
	{
		LockedPages *p = static_cast<LockedPages *>(c);
		std::lock_guard<std::mutex> hold(p->lock);
		return PagesKext::alloc(&p->kext, size, align, offset);
	}
	static bool pagesBlock(void *c, const void *address, uint64_t length, bool writable,
	                       uint64_t *offset, uint64_t *size)
	{
		LockedPages *p = static_cast<LockedPages *>(c);
		std::lock_guard<std::mutex> hold(p->lock);
		return PagesKext::pagesBlock(&p->kext, address, length, writable, offset, size);
	}
	static void release(void *c, uint64_t offset)
	{
		LockedPages *p = static_cast<LockedPages *>(c);
		std::lock_guard<std::mutex> hold(p->lock);
		PagesKext::release(&p->kext, offset);
	}
	static int superpages(void *c, uint64_t size, uint8_t **bytes)
	{
		LockedPages *p = static_cast<LockedPages *>(c);
		std::lock_guard<std::mutex> hold(p->lock);
		p->askers.insert(std::this_thread::get_id());
		return PagesVm::superpages(&p->vm, size, bytes);
	}
	static int pages(void *c, uint64_t size, uint8_t **bytes)
	{
		LockedPages *p = static_cast<LockedPages *>(c);
		std::lock_guard<std::mutex> hold(p->lock);
		return PagesVm::pages(&p->vm, size, bytes);
	}
	static void unmap(void *c, uint8_t *bytes, uint64_t size)
	{
		LockedPages *p = static_cast<LockedPages *>(c);
		std::lock_guard<std::mutex> hold(p->lock);
		PagesVm::unmap(&p->vm, bytes, size);
	}
	static bool record(void *c, RgfxPagesMemory::Note note)
	{
		LockedPages *p = static_cast<LockedPages *>(c);
		std::lock_guard<std::mutex> hold(p->lock);
		return PagesVm::record(&p->vm, note);
	}
	// Waits up to 5 s for `done`, read under the lock.
	template <typename F> bool waitFor(F done)
	{
		for (int i = 0; i < 5000; i++) {
			{
				std::lock_guard<std::mutex> hold(lock);
				if (done())
					return true;
			}
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		return false;
	}
	size_t count(const std::string &line)
	{
		return static_cast<size_t>(std::count(log.begin(), log.end(), line));
	}
};

void test_pages_source_asks_on_its_thread_alone()
{
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 16 * kMiB, 0xee);
	TestRing ring(0x10000);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	RgfxRegion region(mem.data(), mem.size());
	LockedPages p;
	region.setBlockSource(&LockedPages::alloc, &LockedPages::release, &p);
	region.setPagesBlockSource(&LockedPages::pagesBlock, &p);
	p.vm.superpagesLeft = 2;
	RgfxPagesMemory calls;
	calls.context = &p;
	calls.superpages = &LockedPages::superpages;
	calls.pages = &LockedPages::pages;
	calls.unmap = &LockedPages::unmap;
	calls.record = &LockedPages::record;
	RgfxRegionSlabSource regionSource(region, ser);
	RgfxPagesSlabSource pages(region, ser, calls, 16ull << 30);
	RgfxSlabs slabs(regionSource);
	slabs.setFirst(&pages);
	CHECK(slabs.start());
	const std::string superpage = "superpages 2097152";

	// The thread's superpage ahead, at its start; the first slab's take uses it.
	CHECK(p.waitFor([&] { return pages.holdsSuperpage() && p.count(superpage) == 1; }));
	RgfxSlabs::Range a, big, one, c;
	CHECK(slabs.take(0x1000, kShared, 0, &a) && a.offset == 0);
	{
		std::lock_guard<std::mutex> hold(p.lock);
		CHECK(p.vm.held.count(a.contents) == 1 && p.vm.plain.count(a.contents) == 0);
	}
	// The next, after that slab; a larger slab is plain pages at once.
	CHECK(p.waitFor([&] { return pages.holdsSuperpage() && p.count(superpage) == 2; }));
	CHECK(slabs.take(3 * kMiB, kShared, 0, &big));
	CHECK(slabs.take(3 * kMiB / 2, kShared, 0, &one) && one.offset == 0);
	{
		std::lock_guard<std::mutex> hold(p.lock);
		CHECK(p.count("pages 4194304") == 1 && p.vm.plain.count(big.contents) == 1);
		CHECK(p.vm.plain.count(one.contents) == 0);
	}
	// None left: the thread's ask refused, which closes the record, and the next take plain pages.
	CHECK(p.waitFor([&] { return p.count("record refused 0") == 1; }));
	CHECK(slabs.take(0x1000, kSharedUntracked, 0, &c));
	{
		std::lock_guard<std::mutex> hold(p.lock);
		CHECK(p.vm.plain.count(c.contents) == 1 && p.count("pages 2097152") == 1);
		CHECK(p.count(superpage) == 2 && p.count(superpage + " refused") == 1);
		CHECK(p.askers.size() == 1 && p.askers.count(std::this_thread::get_id()) == 0);
	}
}

// Where the device took MANAGED_BUFFER_SLABS, a Managed buffer lies in its Shared class: a range
// of the Shared buffers' slabs, the process's own pages first, at any size, its CPU copy a Shared
// buffer there (BUFFER_AT_OFFSET in a superpage's heap, BUFFER_OVER_PAGES over a larger slab's
// view of plain pages, BUFFER_AT_OFFSET in a region slab's heap where the kernel gives no pages),
// and the Managed buffer a MANAGED_BUFFER_CREATE over it at an address of its own. Its range goes
// back once the copy's destroy follows the buffer's, the copy's address given back where its
// create picked it.
void test_managed_buffers_over_shared_copies()
{
	std::vector<uint8_t> mem(RGFX_ALLOCATIONS_OFFSET + 16 * kMiB, 0xee);
	TestRing ring(0x10000);
	RgfxRingWriter writer(ring.base(), ring.dataSize, RGFX_CONTROL_RING_ID, TestRing::doorbell,
	                      nullptr);
	RgfxIdSpace ids;
	RgfxSerializer ser(writer, ids);
	RgfxRegion region(mem.data(), mem.size());
	std::vector<std::string> log;
	PagesKext kext(16 * kMiB, &log);
	region.setBlockSource(&PagesKext::alloc, &PagesKext::release, &kext);
	region.setPagesBlockSource(&PagesKext::pagesBlock, &kext);
	PagesVm vm(&log);
	vm.superpagesLeft = 3;
	RgfxPagesMemory calls;
	calls.context = &vm;
	calls.superpages = &PagesVm::superpages;
	calls.pages = &PagesVm::pages;
	calls.unmap = &PagesVm::unmap;
	RgfxRegionSlabSource regionSource(region, ser, false, true);
	RgfxPagesSlabSource pages(region, ser, calls, 16ull << 30, false, true);
	CHECK(regionSource.overCopy(kManaged) && !regionSource.overCopy(kShared));
	CHECK(regionSource.classOptions(kManaged) == kShared && pages.classOptions(kShared) == kShared);
	RgfxSlabs slabs(regionSource);
	slabs.setFirst(&pages);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;

	// A small one: the superpage the thread took ahead, a heap of the Shared class, shared with a
	// Shared buffer.
	RgfxSlabs::Range m, s;
	CHECK(slabs.step());
	CHECK(slabs.take(0x1000, kManaged, 0, &m) && m.offset == 0 && m.size == 0x1000);
	CHECK(log == std::vector<std::string>({"superpages 2097152"}));
	auto heap = expectMessage<RgfxHeapOverPages>(ring, RGFX_CMD_HEAP_OVER_PAGES);
	CHECK(heap.options == kShared && heap.size_bytes == 2 * kMiB);
	RgfxSlabSource::Made made;
	CHECK(slabs.makeBuffer(m, 0x800, kManaged, &made));
	auto copy = expectMessage<RgfxBufferAtOffset>(ring, RGFX_CMD_BUFFER_AT_OFFSET);
	CHECK(copy.heap_obj_id == heap.obj_id && copy.offset == 0 && copy.length == 0x800 &&
	      copy.options == kShared);
	auto over = expectMessage<RgfxManagedBufferCreate>(ring, RGFX_CMD_MANAGED_BUFFER_CREATE);
	CHECK(over.cpu_buffer_obj_id == copy.obj_id && over.length == 0x800 &&
	      over.options == kManaged && over.gpu_address != 0);
	CHECK(made.objId == over.obj_id && made.gpuAddress == over.gpu_address && made.ownsAddress);
	CHECK(slabs.take(0x1000, kShared, 0, &s) && s.slab == m.slab && s.offset == 0x1000);

	// A large one: plain pages with no heap, its copy over the block's view.
	RgfxSlabs::Range big;
	CHECK(slabs.take(3 * kMiB, kManaged, 0, &big) && big.size == 3 * kMiB && big.offset == 0);
	CHECK(endsWith(log, {"pages 4194304"}) && kext.pages.size() == 2);
	CHECK(!ring.next(&h, &bytes));
	CHECK(slabs.makeBuffer(big, 3 * kMiB, kManaged, &made));
	auto bigCopy = expectMessage<RgfxBufferOverPages>(ring, RGFX_CMD_BUFFER_OVER_PAGES);
	CHECK(bigCopy.offset == 0 && bigCopy.length == 3 * kMiB && bigCopy.options == kShared);
	auto bigOver = expectMessage<RgfxManagedBufferCreate>(ring, RGFX_CMD_MANAGED_BUFFER_CREATE);
	CHECK(bigOver.cpu_buffer_obj_id == bigCopy.obj_id && bigOver.length == 3 * kMiB);
	CHECK(made.objId == bigOver.obj_id && bigOver.gpu_address != bigCopy.gpu_address);
	CHECK(slabs.stats().slabs == 2 && slabs.stats().firstSlabs == 2);

	// No superpage held and no plain pages: a region slab's heap of the Shared class, nothing of
	// the region's own.
	RgfxSlabs::Range r;
	vm.refusePlain = true;
	CHECK(slabs.take(0x1000, kManaged | 0x1, 0, &r));
	CHECK(endsWith(log, {"pages 2097152 refused"}));
	vm.refusePlain = false;
	auto regionHeap = expectMessage<RgfxHeapOverHostMemory>(ring, RGFX_CMD_HEAP_OVER_HOST_MEMORY);
	CHECK(regionHeap.options == (kShared | 0x1));
	CHECK(slabs.makeBuffer(r, 0x1000, kManaged | 0x1, &made));
	auto regionCopy = expectMessage<RgfxBufferAtOffset>(ring, RGFX_CMD_BUFFER_AT_OFFSET);
	CHECK(regionCopy.heap_obj_id == regionHeap.obj_id && regionCopy.options == (kShared | 0x1));
	auto regionOver = expectMessage<RgfxManagedBufferCreate>(ring, RGFX_CMD_MANAGED_BUFFER_CREATE);
	CHECK(regionOver.cpu_buffer_obj_id == regionCopy.obj_id && regionOver.options == (kManaged | 0x1));

	// Each give destroys its copy, and only then is the range free again: the small Managed
	// buffer's range is taken again by the next buffer of the class.
	CHECK(!ring.next(&h, &bytes));
	slabs.give(m);
	auto destroyed = expectMessage<RgfxObjectDestroy>(ring, RGFX_CMD_OBJECT_DESTROY);
	CHECK(destroyed.obj_id == copy.obj_id && !ring.next(&h, &bytes));
	RgfxSlabs::Range again;
	CHECK(slabs.take(0x1000, kManaged, 0, &again) && again.slab == m.slab && again.offset == 0);
	// A Shared buffer's give destroys nothing of its own.
	slabs.give(s);
	CHECK(!ring.next(&h, &bytes));
	slabs.give(big);
	CHECK(expectMessage<RgfxObjectDestroy>(ring, RGFX_CMD_OBJECT_DESTROY).obj_id == bigCopy.obj_id);
	slabs.give(r);
	CHECK(expectMessage<RgfxObjectDestroy>(ring, RGFX_CMD_OBJECT_DESTROY).obj_id ==
	      regionCopy.obj_id);
	slabs.give(again);
	CHECK(slabs.reclaim() && slabs.stats().slabs == 0 && kext.pages.empty() && vm.held.empty());
	while (ring.next(&h, &bytes)) {
	}
}

} // namespace

void rgfxSlabTests()
{
	test_slab_granule_and_classes();
	test_slab_growth();
	test_slab_reuse_and_fragmentation();
	test_slab_zeroing();
	test_slab_empty_slabs_go();
	test_slab_idle_empty_slabs_go();
	test_slab_dedicated_cache();
	test_slab_fallbacks();
	test_slab_first_source();
	test_slab_thread();
	test_region_slab_source();
	test_region_reclaims_slabs();
	test_pages_slab_source();
	test_pages_slab_source_plain_pages();
	test_superpages_one_call_each();
	test_pages_source_superpage_record();
	test_pages_source_asks_on_its_thread_alone();
	test_managed_buffers_over_shared_copies();
}
