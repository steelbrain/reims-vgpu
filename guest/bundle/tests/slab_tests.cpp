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
#include "reims_graphics_wire.h"
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
// KERN_NO_SPACE; past the wire budget (`budgetSpent`), KERN_RESOURCE_SHORTAGE.
struct PagesVm {
	std::map<uint8_t *, std::vector<uint8_t>> held;
	uint64_t superpagesLeft = 0;
	bool budgetSpent = false;
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
	static void unmap(void *context, uint8_t *bytes, uint64_t size)
	{
		PagesVm *vm = static_cast<PagesVm *>(context);
		vm->held.erase(bytes);
		vm->superpagesLeft += size / (2 * kMiB);
		vm->log->push_back("unmap " + std::to_string(size));
	}
};

bool endsWith(const std::vector<std::string> &log, const std::vector<std::string> &want)
{
	return log.size() >= want.size() && std::equal(want.begin(), want.end(), log.end() - want.size());
}

// The process's own pages as the first source: a heaped class's slab is one 2 MiB superpage
// wired into a pages block of the kext's, made one HEAP_OVER_PAGES its buffers are placed in
// (BUFFER_AT_OFFSET, and BUFFER_MANAGED_AT_OFFSET where the device took MANAGED_AT_OFFSET). A
// dedicated slab larger than a superpage, a kernel with no superpage to give (no contiguous
// 2 MiB, or the wire budget spent), and a kext that gives no block (the superpage given back) are
// the region's slabs, as are Managed buffers without MANAGED_AT_OFFSET. A slab's heap is
// destroyed, then its block released, then its pages go.
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
	calls.unmap = &PagesVm::unmap;
	RgfxRegionSlabSource regionSource(region, ser);
	RgfxPagesSlabSource pages(region, ser, calls, 16ull << 30);
	CHECK(pages.unit() == 2 * kMiB && pages.heaped(kShared) && !pages.heaped(kManaged));
	CHECK(pages.serves(kShared) && pages.serves(kSharedUntracked) && !pages.serves(kManaged));
	RgfxSlabs slabs(regionSource);
	slabs.setFirst(&pages);
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;

	RgfxSlabs::Range a, b;
	CHECK(slabs.take(256, kShared, 0, &a) && a.offset == 0 && a.size == 0x1000);
	CHECK(log == std::vector<std::string>({"superpages 2097152"}) && kext.pages.size() == 1);
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
	// A dedicated buffer of up to a superpage takes one of its own; a larger one, which would be
	// more than one run, the region's slab.
	RgfxSlabs::Range one, big;
	CHECK(slabs.take(3 * kMiB / 2, kShared, 0, &one) && one.size == 3 * kMiB / 2);
	CHECK(endsWith(log, {"superpages 2097152"}) && kext.pages.size() == 2);
	CHECK(expectMessage<RgfxHeapOverPages>(ring, RGFX_CMD_HEAP_OVER_PAGES).size_bytes == 2 * kMiB);
	const size_t asked = log.size();
	CHECK(slabs.take(3 * kMiB, kShared, 0, &big) && big.size == 3 * kMiB);
	CHECK(log.size() == asked && kext.pages.size() == 2);
	CHECK(expectMessage<RgfxHeapOverHostMemory>(ring, RGFX_CMD_HEAP_OVER_HOST_MEMORY).size_bytes ==
	      3 * kMiB);
	// Managed, with no MANAGED_AT_OFFSET: the region's slab, unasked.
	RgfxSlabs::Range m;
	CHECK(slabs.take(0x1000, kManaged, 0, &m) && slabs.makeBuffer(m, 0x1000, kManaged, &made));
	CHECK(log.size() == asked);
	CHECK(expectMessage<RgfxBufferOverHostMemory>(ring, RGFX_CMD_BUFFER_OVER_HOST_MEMORY).length ==
	      0x1000);
	// The last superpage, then none: the region's heaped slab.
	RgfxSlabs::Range c, d;
	CHECK(slabs.take(0x1000, kSharedUntracked, 0, &c) && endsWith(log, {"superpages 2097152"}));
	CHECK(slabs.take(0x1000, kShared | 0x1, 0, &d));
	CHECK(endsWith(log, {"superpages 2097152 refused"}));
	while (ring.next(&h, &bytes)) {
	}
	CHECK(h.type == RGFX_CMD_HEAP_OVER_HOST_MEMORY);
	CHECK(slabs.stats().slabs == 6 && slabs.stats().firstSlabs == 3);
	// Past the wire budget: the region's slab.
	vm.budgetSpent = true;
	RgfxSlabs::Range e;
	CHECK(slabs.take(0x1000, kSharedUntracked | 0x1, 0, &e));
	CHECK(endsWith(log, {"superpages 2097152 refused"}) && slabs.stats().firstSlabs == 3);
	vm.budgetSpent = false;
	// No block from the kext: the superpage goes back, and the region's slab.
	slabs.give(c);
	CHECK(slabs.reclaim());
	kext.refusePages = true;
	log.clear();
	RgfxSlabs::Range f;
	CHECK(slabs.take(0x1000, kSharedUntracked | 0x2, 0, &f));
	CHECK(log == std::vector<std::string>({"superpages 2097152", "unmap 2097152"}));
	kext.refusePages = false;
	while (ring.next(&h, &bytes)) {
	}
	// Every slab let go: each heap destroyed, its pages block released, then its pages.
	for (RgfxSlabs::Range *r : {&a, &b, &one, &big, &m, &d, &e, &f})
		slabs.give(*r);
	log.clear();
	CHECK(slabs.reclaim() && slabs.stats().slabs == 0 && kext.pages.empty() && vm.held.empty());
	CHECK(log == std::vector<std::string>(
	                 {"release pages", "unmap 2097152", "release pages", "unmap 2097152"}));
	int destroys = 0;
	while (ring.next(&h, &bytes))
		destroys += h.type == RGFX_CMD_OBJECT_DESTROY;
	CHECK(destroys == 6); // every slab's heap but the Managed one's plain block

	// With MANAGED_AT_OFFSET, a Managed slab is a superpage too, its buffers' CPU copies placed
	// in its heap.
	RgfxPagesSlabSource managed(region, ser, calls, 16ull << 30, true);
	CHECK(managed.serves(kManaged) && managed.heaped(kManaged));
	RgfxRegionSlabSource managedRegion(region, ser, true);
	RgfxSlabs both(managedRegion);
	both.setFirst(&managed);
	RgfxSlabs::Range g;
	CHECK(both.take(0x1000, kManaged, 0, &g) && endsWith(log, {"superpages 2097152"}));
	auto mheap = expectMessage<RgfxHeapOverPages>(ring, RGFX_CMD_HEAP_OVER_PAGES);
	CHECK(mheap.options == kShared && mheap.size_bytes == 2 * kMiB);
	CHECK(both.makeBuffer(g, 0x800, kManaged, &made));
	auto mat = expectMessage<RgfxBufferManagedAtOffset>(ring, RGFX_CMD_BUFFER_MANAGED_AT_OFFSET);
	CHECK(mat.heap_obj_id == mheap.obj_id && mat.offset == 0 && mat.length == 0x800);
	CHECK(made.ownsAddress && made.gpuAddress == mat.gpu_address);
	both.give(g);
	CHECK(both.reclaim() && kext.pages.empty() && vm.held.empty());
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
}
