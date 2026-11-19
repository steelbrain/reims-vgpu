// The native test of RgfxBlockZero.hpp over RgfxBlockTable.hpp, driven as allocBlock and the
// surface path drive them: a block one process filled and let go of, by its free and the device's
// answer or by its close, reads zero in the next process given its offset, for each kind of block
// in the region (a command buffer's body, a library's staging chunk, a surface's copy), and so
// does the first block over the bytes an earlier boot left; the spans zeroed are the spans the
// window maps, each in its mode, and never a surface block's runs.
#include "RgfxBlockTable.hpp"
#include "RgfxBlockZero.hpp"
#include "RgfxSurfaceBlocks.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "block_zero_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

constexpr uint64_t kPage = kRgfxBlockPage;
constexpr uint64_t kStart = 0x100000;
constexpr uint64_t kEnd = kStart + (48ull << 20);
// What the region's bytes held before the kext's start: an earlier boot's.
constexpr uint8_t kStale = 0x5a;

// The kext's table and the region's bytes, and whether the kext has the region's write-combined
// descriptor, as the x86-64 kext does (kRgfxRegionWriteCombines).
struct Kext {
	RgfxBlock blocks[64] = {};
	RgfxBlockTable t = {blocks, 0, 64, kStart, kEnd, 0, 0, 0, 0, 0, 0, 0};
	std::vector<uint8_t> region = std::vector<uint8_t>(kEnd, kStale);
	bool combines = kRgfxRegionWriteCombines;
	RgfxZeroStats stats = {};
};

// A block a process was given: its offset, and the spans its window maps.
struct Grant {
	uint64_t offset = 0;
	uint64_t bytes = 0;
	RgfxZeroSpan spans[2] = {};
	uint32_t n = 0;
};

void zero(Kext &k, const RgfxZeroSpan &s)
{
	std::memset(&k.region[s.offset], 0, s.length);
}

// allocBlock: the block, its spans zeroed, then mapped from the same spans.
Grant alloc(Kext &k, const void *owner, uint32_t context, uint64_t size, uint64_t writeCombinedFrom)
{
	Grant g;
	RgfxBlockPin pin = {};
	check(rgfxBlockAlloc(k.t, owner, context, size, kPage, &g.offset, &pin) == kRgfxBlockOk,
	      "a block");
	g.bytes = rgfxBlockAlignUp(size, kPage);
	g.n = rgfxBlockZeroSpans(g.offset, size, g.bytes, writeCombinedFrom, k.combines, g.spans);
	for (uint32_t i = 0; i < g.n; i++)
		zero(k, g.spans[i]);
	rgfxZeroNote(k.stats, rgfxBlockZeroKind(size, writeCombinedFrom), g.bytes, 3000, 2000, 1000,
	             true);
	rgfxBlockUnpin(k.t, pin);
	return g;
}

// The surface path's new block: `total` bytes, the first `size` the ones its holders map.
Grant surfaceBlock(Kext &k, const void *entry, uint32_t context, uint64_t size, uint64_t total,
                   bool inPlace)
{
	Grant g;
	RgfxBlockPin pin = {};
	check(rgfxBlockAlloc(k.t, entry, context, total, kPage, &g.offset, &pin) == kRgfxBlockOk,
	      "a surface's block");
	g.bytes = rgfxBlockAlignUp(total, kPage);
	g.spans[0] = rgfxSurfaceZeroSpan(g.offset, size);
	g.n = 1;
	zero(k, g.spans[0]);
	rgfxZeroNote(k.stats, inPlace ? kRgfxZeroSurfaceHeader : kRgfxZeroSurfaceCopy,
	             g.spans[0].length, 3000, 2000, 1000, true);
	rgfxBlockUnpin(k.t, pin);
	return g;
}

// The process's stores over everything its window maps.
void fill(Kext &k, const Grant &g, uint8_t value)
{
	for (uint32_t i = 0; i < g.n; i++)
		std::memset(&k.region[g.spans[i].offset], value, g.spans[i].length);
}

bool readsZero(const Kext &k, const Grant &g)
{
	for (uint32_t i = 0; i < g.n; i++)
		for (uint64_t b = 0; b < g.spans[i].length; b++)
			if (k.region[g.spans[i].offset + b] != 0)
				return false;
	return true;
}

uint64_t mapped(const Grant &g)
{
	uint64_t n = 0;
	for (uint32_t i = 0; i < g.n; i++)
		n += g.spans[i].length;
	return n;
}

// The holder's free, the release posted, and the device's OK: the offset is given out again.
void freeAnswered(Kext &k, const void *owner, uint64_t offset)
{
	check(rgfxBlockFree(k.t, owner, offset) == kRgfxBlockOk, "a free");
	RgfxBlock release = {};
	check(rgfxBlockNextRelease(k.t, &release) && release.offset == offset, "its release");
	rgfxBlockReleased(k.t, offset, release.serial, true);
	check(rgfxBlockIndex(k.t, offset) < 0, "answered OK, the block is gone");
}

const char kA = 0, kB = 0, kEntryA = 0, kEntryB = 0;
const void *const a = &kA;
const void *const b = &kB;
const void *const entryA = &kEntryA;
const void *const entryB = &kEntryB;
} // namespace

int main()
{
	// The spans are what the window maps (allocBlock's comment, mapSpan): from the block's start
	// to its end, in order, write-back up to the asked write-combined byte and write-combined
	// past it where the kext has the descriptor, else all write-back.
	{
		const uint64_t sizes[] = {1, kPage, kPage + 1, 5 * kPage, 64 * kPage + 7};
		for (uint64_t size : sizes) {
			const uint64_t bytes = rgfxBlockAlignUp(size, kPage);
			const uint64_t froms[] = {0, kPage, 2 * kPage, bytes, UINT64_MAX};
			for (uint64_t from : froms) {
				for (int combines = 0; combines < 2; combines++) {
					RgfxZeroSpan s[2] = {};
					const uint32_t n =
					    rgfxBlockZeroSpans(kStart, size, bytes, from, combines != 0, s);
					check(n >= 1 && n <= 2, "one or two spans");
					check(s[0].offset == kStart, "from the block's start");
					uint64_t at = kStart;
					for (uint32_t i = 0; i < n; i++) {
						check(s[i].offset == at && s[i].length > 0, "the spans in order, none empty");
						at += s[i].length;
					}
					check(at == kStart + bytes, "to the block's end, its rounding too");
					const bool split = combines && from < size;
					check(n == (split && from ? 2u : 1u), "a span for each mode");
					check(s[n - 1].writeCombined == split, "write-combined only past the byte asked");
					check(!split || s[n - 1].offset == kStart + rgfxWindowWriteBackPart(size, bytes, from),
					      "at the window's write-back part");
					check(!split || from || (n == 1 && s[0].writeCombined),
					      "a body is write-combined whole");
				}
			}
		}
		RgfxZeroSpan none[2] = {};
		check(rgfxBlockZeroSpans(kStart, 0, 0, 0, true, none) == 0, "no bytes, no span");
		check(rgfxBlockZeroKind(64 << 10, 0) == kRgfxZeroBodies, "a body's kind");
		check(rgfxBlockZeroKind(64 << 10, kPage) == kRgfxZeroSplit, "a staging chunk's kind");
		check(rgfxBlockZeroKind(64 << 10, UINT64_MAX) == kRgfxZeroWriteBack, "a slab's kind");
		check(rgfxBlockZeroKind(kPage, kPage) == kRgfxZeroWriteBack,
		      "write-combined past the block's end: write-back whole");
	}

	// The bytes an earlier boot left: the first block reads zero.
	{
		Kext k;
		const Grant g = alloc(k, a, 1, 3 * kPage, UINT64_MAX);
		check(readsZero(k, g), "a block over an earlier boot's bytes reads zero");
	}

	// A command buffer's body (write-combined whole), freed and answered: the next process
	// given its offset reads zero, over a write-combined span.
	{
		Kext k;
		const Grant mine = alloc(k, a, 1, 64 << 10, 0);
		fill(k, mine, 0xa5);
		freeAnswered(k, a, mine.offset);
		const Grant theirs = alloc(k, b, 2, 64 << 10, 0);
		check(theirs.offset == mine.offset, "a body's offset given to another process");
		check(mapped(theirs) == theirs.bytes, "its window maps the whole body");
		check(theirs.spans[0].writeCombined == k.combines, "write-combined where the kext can");
		check(readsZero(k, theirs), "another process's body reads zero");
	}

	// A library's staging chunk (a write-back page, write-combined past it).
	{
		Kext k;
		const uint64_t size = kPage + (256 << 10);
		const Grant mine = alloc(k, a, 1, size, kPage);
		fill(k, mine, 0xa5);
		freeAnswered(k, a, mine.offset);
		const Grant theirs = alloc(k, b, 2, size, kPage);
		check(theirs.offset == mine.offset, "a staging chunk's offset given to another process");
		check(theirs.n == (k.combines ? 2u : 1u), "a write-back page and a write-combined rest");
		check(readsZero(k, theirs), "another process's staging chunk reads zero");
	}

	// A surface's copy block, let go of by its last holder and answered: another surface's copy
	// block at its offset reads zero over all its holders map.
	{
		Kext k;
		const uint64_t size = 1920ull * 1080 * 4 + 100;
		const Grant mine = surfaceBlock(k, entryA, 1, size, size, false);
		check(mapped(mine) == rgfxWindowSpanLength(size), "a copy's holders map the whole copy");
		fill(k, mine, 0xa5);
		freeAnswered(k, entryA, mine.offset);
		const Grant theirs = surfaceBlock(k, entryB, 2, size, size, false);
		check(theirs.offset == mine.offset, "a copy block's offset given to another surface");
		check(readsZero(k, theirs), "another surface's copy block reads zero");
	}

	// A surface's block in place: its header's pages are zeroed, the runs past them, the kext's
	// own and never mapped, are not.
	{
		Kext k;
		const uint64_t header = kPage;
		const uint64_t total = rgfxSurfaceRunsFrom(header) + 2 * kPage;
		const Grant g = surfaceBlock(k, entryA, 1, header, total, true);
		check(g.spans[0].length == rgfxSurfaceRunsFrom(header),
		      "the header's span ends where the runs start");
		check(readsZero(k, g), "the header reads zero");
		check(k.region[g.offset + rgfxSurfaceRunsFrom(header)] == kStale,
		      "the runs are left for the kext to write");
	}

	// A process's close: its blocks go, and the next process given the offset reads zero.
	{
		Kext k;
		const Grant mine = alloc(k, a, 1, 64 << 10, 0);
		const Grant staging = alloc(k, a, 1, kPage + (64 << 10), kPage);
		fill(k, mine, 0xa5);
		fill(k, staging, 0xa6);
		check(rgfxBlockFreeOwner(k.t, a) == 2, "the close frees both");
		const Grant theirs = alloc(k, b, 2, 64 << 10, 0);
		const Grant theirStaging = alloc(k, b, 2, kPage + (64 << 10), kPage);
		check(theirs.offset == mine.offset && theirStaging.offset == staging.offset,
		      "the closed process's offsets given to another");
		check(readsZero(k, theirs) && readsZero(k, theirStaging),
		      "the closed process's bytes read zero");
	}

	// Another kind at the offset: a write-back slab let go of, a body over its first bytes, then
	// a staging chunk past it.
	{
		Kext k;
		const Grant slab = alloc(k, a, 1, 1 << 20, UINT64_MAX);
		fill(k, slab, 0xa5);
		freeAnswered(k, a, slab.offset);
		const Grant body = alloc(k, b, 2, 64 << 10, 0);
		const Grant staging = alloc(k, b, 2, kPage + (128 << 10), kPage);
		check(body.offset == slab.offset && staging.offset == slab.offset + body.bytes,
		      "both inside the slab's bytes");
		check(readsZero(k, body) && readsZero(k, staging), "each reads zero");
	}

	// The log's counts.
	{
		RgfxZeroStats s = {};
		rgfxZeroNote(s, kRgfxZeroBodies, 64 << 10, 5000, 4000, 1000, true);
		rgfxZeroNote(s, kRgfxZeroBodies, 64 << 10, 7000, 5000, 2000, true);
		rgfxZeroNote(s, kRgfxZeroSurfaceCopy, 1 << 20, 90000, 10000, 80000, true);
		rgfxZeroNote(s, kRgfxZeroSplit, 0, 0, 0, 0, false);
		check(s.took[kRgfxZeroBodies].count == 2 && s.ns[kRgfxZeroBodies] == 12000 &&
		          s.bytes[kRgfxZeroBodies] == 128 << 10,
		      "the bodies counted");
		check(s.took[kRgfxZeroBodies].maxNs == 7000, "their most");
		check(s.took[kRgfxZeroSurfaceCopy].count == 1 && s.took[kRgfxZeroSplit].count == 0,
		      "each kind its own");
		check(s.mapNs == 19000 && s.storeNs == 83000, "the mappings' and the stores' time");
		check(s.failed == 1 && rgfxZeroCount(s) == 3, "a span not mapped is counted apart");
		rgfxZeroNote(s, kRgfxZeroKinds, 1, 1, 1, 1, true);
		check(rgfxZeroCount(s) == 3, "no kind past the last");
	}

	std::printf("block_zero_test: %d checks passed\n", checks);
	return 0;
}
