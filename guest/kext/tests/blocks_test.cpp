// The native test of RgfxBlockTable.hpp: a block is given out again only once the device has
// answered its release OK, a pinned block is not freed, and a client's blocks go at its close only
// where the close was answered.
#include "RgfxBlockTable.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "blocks_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

constexpr uint64_t kStart = 0x10000;
constexpr uint64_t kEnd = 0x20000;

struct Table {
	RgfxBlock blocks[8] = {};
	RgfxBlockTable t = {blocks, 0, 8, kStart, kEnd, 0, 0, 0, 0, 0, 0, 0};
};

const char kA = 0, kB = 0;
const void *const a = &kA;
const void *const b = &kB;
} // namespace

int main()
{
	{
		Table x;
		uint64_t first = 0, second = 0, third = 0;
		check(rgfxBlockAlloc(x.t, a, 1, 0x1000, 0x1000, &first) == kRgfxBlockOk &&
		          first == kStart,
		      "the first block starts the span");
		check(rgfxBlockAlloc(x.t, b, 2, 0x1800, 0x1000, &second) == kRgfxBlockOk &&
		          second == kStart + 0x1000,
		      "the next fits behind it, a whole number of pages");
		check(rgfxBlockFree(x.t, b, first) == kRgfxBlockBadArgument,
		      "a client frees only its own block");
		check(rgfxBlockFree(x.t, a, first + 0x10) == kRgfxBlockBadArgument,
		      "a free names a block's start");

		// Freed, the block is no one's and not given out again until the device answers OK.
		check(rgfxBlockFree(x.t, a, first) == kRgfxBlockOk, "a free");
		check(rgfxBlockFree(x.t, a, first) == kRgfxBlockBadArgument, "a second free of it");
		RgfxBlockPin pin;
		check(!rgfxBlockPin(x.t, a, first, 0x10, &pin), "a freed block is not named");
		check(rgfxBlockAlloc(x.t, a, 1, 0x1000, 0x1000, &third) == kRgfxBlockOk &&
		          third == kStart + 0x3000,
		      "a freed block is not given out");
		RgfxBlock release;
		check(rgfxBlockNextRelease(x.t, &release) && release.offset == first &&
		          release.context == 1 && release.size == 0x1000,
		      "its release is to be posted");
		check(!rgfxBlockNextRelease(x.t, &release), "once");
		// An answer for another allocation at the offset finds nothing.
		rgfxBlockReleased(x.t, first, release.serial + 100, true);
		check(rgfxBlockIndex(x.t, first) == 0, "a stale answer keeps the block");
		rgfxBlockReleased(x.t, first, release.serial, true);
		check(rgfxBlockIndex(x.t, first) < 0, "OK gives it out again");
		uint64_t again = 0;
		check(rgfxBlockAlloc(x.t, b, 2, 0x1000, 0x1000, &again) == kRgfxBlockOk && again == first,
		      "to the next client that asks");

		// A refused release keeps the block.
		check(rgfxBlockFree(x.t, a, third) == kRgfxBlockOk, "a free");
		check(rgfxBlockNextRelease(x.t, &release) && release.offset == third, "its release");
		rgfxBlockReleased(x.t, third, release.serial, false);
		check(rgfxBlockIndex(x.t, third) >= 0 &&
		          x.t.blocks[rgfxBlockIndex(x.t, third)].state == kRgfxBlockKept,
		      "a refused release keeps it");
		check(!rgfxBlockNextRelease(x.t, &release), "and is not posted again");
	}
	{
		// A pinned block is not freed; an unpin of a block gone since touches nothing.
		Table x;
		uint64_t at = 0;
		check(rgfxBlockAlloc(x.t, a, 1, 0x2000, 0x1000, &at) == kRgfxBlockOk, "a block");
		RgfxBlockPin pin;
		check(!rgfxBlockPin(x.t, b, at, 0x10, &pin), "another client does not pin it");
		check(!rgfxBlockPin(x.t, a, at + 0x1000, 0x1001, &pin), "nor a span past its end");
		check(rgfxBlockPin(x.t, a, at + 0x1000, 0x1000, &pin) && pin.offset == at,
		      "its owner pins a span in it");
		uint64_t size = 0;
		check(rgfxBlockFreeable(x.t, a, at, &size) == kRgfxBlockBusy && size == 0,
		      "pinned, it may not be freed");
		check(rgfxBlockFree(x.t, a, at) == kRgfxBlockBusy, "pinned, it is busy");
		rgfxBlockUnpin(x.t, pin);
		check(rgfxBlockFreeable(x.t, b, at, &size) == kRgfxBlockBadArgument,
		      "another client may not free it");
		check(rgfxBlockFreeable(x.t, a, at + 0x1000, &size) == kRgfxBlockBadArgument,
		      "nor its owner a span in it");
		check(rgfxBlockFreeable(x.t, a, at, &size) == kRgfxBlockOk && size == 0x2000,
		      "unpinned, its owner may free its bytes");
		check(rgfxBlockFree(x.t, a, at) == kRgfxBlockOk, "unpinned, it is freed");
		check(rgfxBlockFreeable(x.t, a, at, &size) == kRgfxBlockBadArgument,
		      "freed, it may not be freed again");
		RgfxBlock release;
		check(rgfxBlockNextRelease(x.t, &release), "its release");
		rgfxBlockReleased(x.t, at, release.serial, true);
		uint64_t next = 0;
		check(rgfxBlockAlloc(x.t, b, 2, 0x2000, 0x1000, &next) == kRgfxBlockOk && next == at,
		      "given out again");
		rgfxBlockUnpin(x.t, pin);
		check(x.t.blocks[0].pins == 0, "a stale unpin touches nothing");
		check(rgfxBlockPin(x.t, b, next, 0x10, &pin), "the new owner pins it");
		rgfxBlockUnpin(x.t, RgfxBlockPin{at, release.serial});
		check(x.t.blocks[0].pins == 1, "nor the new owner's pin");
	}
	{
		// A close answered frees every block of the client, whatever its state; one not
		// answered keeps them all from use, save a releasing one the device answers OK.
		Table x;
		uint64_t live = 0, freed = 0, releasing = 0, other = 0;
		rgfxBlockAlloc(x.t, a, 1, 0x1000, 0x1000, &live);
		rgfxBlockAlloc(x.t, a, 1, 0x1000, 0x1000, &freed);
		rgfxBlockAlloc(x.t, a, 1, 0x1000, 0x1000, &releasing);
		rgfxBlockAlloc(x.t, b, 2, 0x1000, 0x1000, &other);
		rgfxBlockFree(x.t, a, releasing);
		RgfxBlock release;
		check(rgfxBlockNextRelease(x.t, &release) && release.offset == releasing, "releasing");
		rgfxBlockFree(x.t, a, freed);

		// Pages blocks of each client's hold their wired pages: the answered close's are taken
		// out, each once, before its blocks go; the other client's stay with its block.
		int pagesA = 0, pagesA2 = 0, pagesB = 0;
		check(rgfxBlockSetWired(x.t, live, &pagesA) && rgfxBlockSetWired(x.t, releasing, &pagesA2) &&
		          rgfxBlockSetWired(x.t, other, &pagesB),
		      "pages blocks hold their wired pages");
		Table y = x;
		y.t.blocks = y.blocks;
		void *took[2] = {rgfxBlockTakeWired(x.t, a), rgfxBlockTakeWired(x.t, a)};
		check(took[0] == &pagesA && took[1] == &pagesA2, "the closed client's wired pages, in order");
		check(rgfxBlockTakeWired(x.t, a) == nullptr, "each once");
		check(rgfxBlockFreeOwner(x.t, a) == 3 && x.t.count == 1 && x.t.blocks[0].owner == b,
		      "an answered close frees the client's blocks");
		check(x.t.blocks[0].wired == &pagesB, "the other client's pages stay wired");
		// A releasing block taken out answers no wired pages to its release's answer.
		check(!rgfxBlockReleasingAt(x.t, releasing, release.serial),
		      "a releasing block taken out is its release's no more");
		// Its offset is given out again, and that block's release is posted: the old release's
		// late answer is not the new block's, whose release still holds its surface.
		uint64_t again = 0;
		check(rgfxBlockAlloc(x.t, b, 2, 0x1000, 0x1000, &again) == kRgfxBlockOk &&
		          again == kStart,
		      "the first freed offset is given out again");
		uint64_t reused = 0;
		check(rgfxBlockAlloc(x.t, b, 2, 0x1000, 0x1000, &reused) == kRgfxBlockOk &&
		          reused == kStart + 0x1000 && rgfxBlockAlloc(x.t, b, 2, 0x1000, 0x1000, &reused) ==
		                                           kRgfxBlockOk &&
		          reused == releasing,
		      "and the released block's too");
		rgfxBlockFree(x.t, b, reused);
		RgfxBlock newer;
		check(rgfxBlockNextRelease(x.t, &newer) && newer.offset == releasing &&
		          newer.serial != release.serial,
		      "the new block's release, of another serial");
		check(!rgfxBlockReleasingAt(x.t, releasing, release.serial) &&
		          rgfxBlockReleasingAt(x.t, releasing, newer.serial),
		      "the old release's answer finds the offset another's");
		void *wired = &pagesB;
		rgfxBlockReleased(x.t, releasing, release.serial, true, &wired);
		check(wired == nullptr && rgfxBlockReleasingAt(x.t, releasing, newer.serial),
		      "a release answered after the close unwires nothing and frees nothing of the new");

		check(rgfxBlockKeepOwner(y.t, a) == 3 && y.t.count == 4, "an unanswered close keeps them");
		RgfxBlockPin pin;
		check(!rgfxBlockPin(y.t, a, live, 0x10, &pin), "kept blocks are no one's");
		check(!rgfxBlockNextRelease(y.t, &release), "and not released");
		uint64_t at = 0;
		check(rgfxBlockAlloc(y.t, b, 2, 0x1000, 0x1000, &at) == kRgfxBlockOk &&
		          at == kStart + 0x4000,
		      "nor given out");
		rgfxBlockReleased(y.t, releasing, release.serial, true);
		check(rgfxBlockIndex(y.t, releasing) < 0, "a releasing one answered OK goes");
	}
	{
		// A block given out pinned is busy until its grant is posted, and dropped where it
		// could not be.
		Table x;
		uint64_t at = 0, next = 0;
		RgfxBlockPin pin;
		check(rgfxBlockAlloc(x.t, a, 1, 0x1000, 0x1000, &at, &pin) == kRgfxBlockOk &&
		          pin.offset == at,
		      "a block given out pinned");
		check(rgfxBlockFree(x.t, a, at) == kRgfxBlockBusy, "is busy before its grant");
		rgfxBlockUnpin(x.t, pin);
		check(rgfxBlockFree(x.t, a, at) == kRgfxBlockOk, "and freed after it");
		check(rgfxBlockAlloc(x.t, b, 2, 0x1000, 0x1000, &next, &pin) == kRgfxBlockOk &&
		          next == at + 0x1000,
		      "another");
		rgfxBlockDrop(x.t, RgfxBlockPin{next, pin.serial + 1});
		check(rgfxBlockIndex(x.t, next) >= 0, "a drop of another allocation keeps it");
		rgfxBlockDrop(x.t, pin);
		check(rgfxBlockIndex(x.t, next) < 0, "a drop takes it back");
	}
	{
		Table x;
		uint64_t at = 0;
		check(rgfxBlockAlloc(x.t, a, 1, 0, 0x1000, &at) == kRgfxBlockBadArgument, "size 0");
		check(rgfxBlockAlloc(x.t, a, 1, 0x1000, 0x3000, &at) == kRgfxBlockBadArgument,
		      "an alignment not a power of two");
		check(rgfxBlockAlloc(x.t, a, 1, kEnd, 0x1000, &at) == kRgfxBlockNoSpace,
		      "more than the span");
		for (int i = 0; i < 8; i++)
			rgfxBlockAlloc(x.t, a, 1, 0x1000, 0x1000, &at);
		check(rgfxBlockAlloc(x.t, a, 1, 0x1000, 0x1000, &at) == kRgfxBlockTableFull,
		      "a full table");
	}
	{
		// A shared block: pinned whoever owns it, released naming its last holder's context,
		// and kept for good where that holder's close was not answered.
		Table x;
		uint64_t at = 0;
		check(rgfxBlockAlloc(x.t, a, 1, 0x2000, 0x1000, &at) == kRgfxBlockOk, "a block");
		RgfxBlockPin pin;
		check(rgfxBlockPinAny(x.t, at + 0x1000, 0x1000, &pin) && pin.offset == at,
		      "any client's span in it pins it");
		check(!rgfxBlockPinAny(x.t, at + 0x1000, 0x1001, &pin), "a span past it does not");
		check(rgfxBlockFree(x.t, a, at) == kRgfxBlockBusy, "pinned, it is not freed");
		rgfxBlockUnpin(x.t, pin);
		rgfxBlockSetContext(x.t, at, 9);
		check(rgfxBlockFree(x.t, a, at) == kRgfxBlockOk, "unpinned, it is");
		check(!rgfxBlockPinAny(x.t, at, 0x10, &pin), "a freed block is not pinned");
		RgfxBlock release;
		check(rgfxBlockNextRelease(x.t, &release) && release.context == 9,
		      "its release names the context set");
		uint64_t kept = 0;
		check(rgfxBlockAlloc(x.t, b, 2, 0x1000, 0x1000, &kept) == kRgfxBlockOk, "another");
		rgfxBlockKeepAt(x.t, kept);
		check(rgfxBlockFree(x.t, b, kept) == kRgfxBlockBadArgument &&
		          !rgfxBlockPinAny(x.t, kept, 0x10, &pin),
		      "a kept block is no one's");
		check(!rgfxBlockNextRelease(x.t, &release), "nor released");
	}
	{
		// An in-place block's wired pages are held by the block until its release is answered
		// OK, then handed back for the caller to unwire; a refused release keeps the block and
		// is posted again after a context close.
		Table x;
		int pagesA = 0, pagesB = 0;
		void *wireA = &pagesA, *wireB = &pagesB;
		uint64_t a1 = 0, b1 = 0;
		check(rgfxBlockAlloc(x.t, a, 1, 0x1000, 0x1000, &a1) == kRgfxBlockOk, "a block");
		check(rgfxBlockAlloc(x.t, b, 2, 0x1000, 0x1000, &b1) == kRgfxBlockOk, "another");
		check(x.t.blocks[rgfxBlockIndex(x.t, a1)].wired == nullptr &&
		          !x.t.blocks[rgfxBlockIndex(x.t, a1)].refused,
		      "a fresh block holds no wire and is not refused");
		check(rgfxBlockSetWired(x.t, a1, wireA) && rgfxBlockSetWired(x.t, b1, wireB),
		      "an in-place block's wire attaches");
		check(!rgfxBlockSetWired(x.t, 0x999000, wireA), "no wire attaches to a block not there");

		// The first block's release is answered OK: the block goes and hands back its wire.
		check(rgfxBlockFree(x.t, a, a1) == kRgfxBlockOk, "a free");
		RgfxBlock rel;
		check(rgfxBlockNextRelease(x.t, &rel) && rel.offset == a1, "its release");
		void *got = reinterpret_cast<void *>(0x1);
		rgfxBlockReleased(x.t, a1, rel.serial, true, &got);
		check(got == wireA && rgfxBlockIndex(x.t, a1) < 0, "OK hands back the wire and frees it");

		// The second block's release is refused: it is kept, marked refused, and keeps its wire.
		check(rgfxBlockFree(x.t, b, b1) == kRgfxBlockOk, "a free");
		check(rgfxBlockNextRelease(x.t, &rel) && rel.offset == b1, "its release");
		got = reinterpret_cast<void *>(0x1);
		rgfxBlockReleased(x.t, b1, rel.serial, false, &got);
		int64_t bi = rgfxBlockIndex(x.t, b1);
		check(bi >= 0 && x.t.blocks[bi].state == kRgfxBlockKept && x.t.blocks[bi].refused &&
		          x.t.blocks[bi].wired == wireB,
		      "a refused release keeps the block, marks it refused and keeps its wire");
		check(got == nullptr, "and hands back no wire");
		check(!rgfxBlockNextRelease(x.t, &rel), "it is not posted again on its own");

		// A context close posts refused releases again; this time OK frees it and hands the wire.
		check(rgfxBlockRetryRefused(x.t) == 1, "a close posts the refused release again");
		bi = rgfxBlockIndex(x.t, b1);
		check(bi >= 0 && x.t.blocks[bi].state == kRgfxBlockFreed && !x.t.blocks[bi].refused,
		      "the refused block is freed again, no longer refused");
		check(rgfxBlockNextRelease(x.t, &rel) && rel.offset == b1, "its release once more");
		got = reinterpret_cast<void *>(0x1);
		rgfxBlockReleased(x.t, b1, rel.serial, true, &got);
		check(got == wireB && rgfxBlockIndex(x.t, b1) < 0, "OK hands back the wire and frees it");
		check(rgfxBlockRetryRefused(x.t) == 0, "no refused blocks are left");
	}
	{
		// A cached surface block kept instead of released (a create in flight names it) holds the
		// cache's wiring of its pages where it holds none of a holder's: the device may still map
		// them in place, and no answer ever hands the wiring back, however many come.
		Table x;
		int cachePages = 0, holderPages = 0, other = 0;
		uint64_t dormant = 0, held = 0;
		check(rgfxBlockAlloc(x.t, a, 1, 0x1000, 0x1000, &dormant) == kRgfxBlockOk &&
		          rgfxBlockAlloc(x.t, a, 1, 0x1000, 0x1000, &held) == kRgfxBlockOk,
		      "two surface blocks");
		RgfxBlockPin pin;
		check(rgfxBlockPinAny(x.t, dormant, 0x10, &pin) &&
		          rgfxBlockFree(x.t, a, dormant) == kRgfxBlockBusy,
		      "a create in flight names the first: it is not freed");
		rgfxBlockKeepAt(x.t, dormant);
		check(rgfxBlockHoldWire(x.t, dormant, &cachePages),
		      "kept with no holder's wiring, it holds the cache's");
		check(!rgfxBlockHoldWire(x.t, dormant, &other), "and keeps it: a second is not taken");
		rgfxBlockUnpin(x.t, pin);
		RgfxBlock rel;
		check(!rgfxBlockNextRelease(x.t, &rel), "its release is never posted");
		check(rgfxBlockRetryRefused(x.t) == 0, "nor posted again by a close");
		void *got = reinterpret_cast<void *>(0x1);
		for (uint64_t serial = 0; serial < 64; serial++)
			rgfxBlockReleased(x.t, dormant, serial, true, &got);
		const int64_t di = rgfxBlockIndex(x.t, dormant);
		check(got == nullptr && di >= 0 && x.t.blocks[di].wired == &cachePages,
		      "no answer hands its wiring back: its pages stay wired for good");
		// The other is kept holding its last holder's wiring: the cache's goes, the block's own
		// holding the same pages.
		check(rgfxBlockSetWired(x.t, held, &holderPages), "a holder's wiring");
		rgfxBlockKeepAt(x.t, held);
		check(!rgfxBlockHoldWire(x.t, held, &cachePages),
		      "a block holding a holder's wiring takes no cache's");
		check(!rgfxBlockHoldWire(x.t, 0x999000, &cachePages), "nor does a block not there");
		check(!rgfxBlockHoldWire(x.t, held, nullptr), "and no wiring is none to hold");
	}
	{
		// A block kept for good (a close not answered, or a create naming it) is not posted
		// again by a later close: it is not `refused`.
		Table x;
		uint64_t a1 = 0, a2 = 0;
		rgfxBlockAlloc(x.t, a, 1, 0x1000, 0x1000, &a1);
		rgfxBlockAlloc(x.t, a, 1, 0x1000, 0x1000, &a2);
		// One kept at (a shared block whose close was not answered), one kept across a close.
		rgfxBlockKeepAt(x.t, a1);
		check(rgfxBlockKeepOwner(x.t, a) == 1, "the owner's other block is kept across a close");
		check(rgfxBlockRetryRefused(x.t) == 0, "neither kept-for-good block is posted again");

		// But a block refused and only then kept-for-good (its owner's close not answered) is
		// cleared of refused, so it is never posted again either.
		uint64_t b1 = 0;
		rgfxBlockAlloc(x.t, b, 2, 0x1000, 0x1000, &b1);
		rgfxBlockFree(x.t, b, b1);
		RgfxBlock rel;
		rgfxBlockNextRelease(x.t, &rel);
		rgfxBlockReleased(x.t, b1, rel.serial, false);
		check(x.t.blocks[rgfxBlockIndex(x.t, b1)].refused, "refused first");
		check(rgfxBlockKeepOwner(x.t, b) == 1, "then its owner's close is not answered");
		check(!x.t.blocks[rgfxBlockIndex(x.t, b1)].refused &&
		          rgfxBlockRetryRefused(x.t) == 0,
		      "a kept-for-good block is no longer refused");
	}
	{
		// WindowServer's reserve: the last 4 pages and the last 2 entries are its alone.
		Table x;
		x.t.reserveBytes = 0x4000;
		x.t.reserveEntries = 2;
		uint64_t o[8] = {};
		check(rgfxBlockAlloc(x.t, a, 1, 0x4000, 0x1000, &o[0], nullptr, true) == kRgfxBlockOk &&
		          o[0] == kStart,
		      "WindowServer takes the first gap, outside the reserve, while there is room");
		check(rgfxBlockAlloc(x.t, b, 2, 0x4000, 0x1000, &o[1]) == kRgfxBlockOk &&
		          rgfxBlockAlloc(x.t, b, 2, 0x4000, 0x1000, &o[2]) == kRgfxBlockOk,
		      "another process takes the rest of the room");
		uint64_t none = 0;
		check(rgfxBlockAlloc(x.t, b, 2, 0x1000, 0x1000, &none) == kRgfxBlockNoSpace,
		      "but not the reserve");
		check(rgfxBlockAlloc(x.t, a, 1, 0x4000, 0x1000, &o[3], nullptr, true) == kRgfxBlockOk &&
		          o[3] == kEnd - 0x4000,
		      "which WindowServer reaches once the rest is full");
		check(x.t.reservedCount == 2 && x.t.reservedBytes == 0x8000,
		      "WindowServer's blocks are counted");
		check(rgfxBlockAlloc(x.t, a, 1, 0x1000, 0x1000, &none, nullptr, true) ==
		          kRgfxBlockNoSpace,
		      "and the reserve runs out too");
		// Its block leaves the table at its release: no longer counted, the peak kept.
		check(rgfxBlockFree(x.t, a, o[3]) == kRgfxBlockOk, "WindowServer frees its block");
		RgfxBlock rel;
		check(rgfxBlockNextRelease(x.t, &rel) && rel.offset == o[3], "its release is posted");
		rgfxBlockReleased(x.t, o[3], rel.serial, true);
		check(x.t.reservedCount == 1 && x.t.reservedBytes == 0x4000 &&
		          x.t.reservedPeakCount == 2 && x.t.reservedPeakBytes == 0x8000,
		      "a released block is no longer counted, the peak stays");
		check(rgfxBlockAlloc(x.t, b, 2, 0x1000, 0x1000, &none) == kRgfxBlockNoSpace,
		      "the freed reserve is still not another process's");
	}
	{
		// A block may not reach into the reserve, nor be larger than the room outside it.
		Table x;
		x.t.reserveBytes = 0x4000;
		uint64_t at = 0;
		check(rgfxBlockAlloc(x.t, b, 2, 0xd000, 0x1000, &at) == kRgfxBlockNoSpace,
		      "a block larger than the room outside the reserve is refused");
		check(rgfxBlockAlloc(x.t, b, 2, 0xc000, 0x1000, &at) == kRgfxBlockOk && at == kStart,
		      "one that fits beside it is not");
		Table y;
		y.t.reserveBytes = 0x4000;
		check(rgfxBlockAlloc(y.t, a, 1, 0xd000, 0x1000, &at, nullptr, true) == kRgfxBlockOk,
		      "WindowServer's may reach into it");
		Table z;
		z.t.reserveBytes = 0x20000;
		check(rgfxBlockAlloc(z.t, b, 2, 0x1000, 0x1000, &at) == kRgfxBlockNoSpace,
		      "a reserve past the range leaves the others no room");
		check(rgfxBlockAlloc(z.t, a, 1, 0x1000, 0x1000, &at, nullptr, true) == kRgfxBlockOk,
		      "and WindowServer all of it");
	}
	{
		// The last 2 of the 8 entries are WindowServer's.
		Table x;
		x.t.reserveEntries = 2;
		uint64_t at = 0;
		int made = 0;
		while (rgfxBlockAlloc(x.t, b, 2, 0x1000, 0x1000, &at) == kRgfxBlockOk)
			made++;
		check(made == 6, "another process gets all but the reserved entries");
		check(rgfxBlockAlloc(x.t, b, 2, 0x1000, 0x1000, &at) == kRgfxBlockTableFull,
		      "and then the table is full for it");
		check(rgfxBlockAlloc(x.t, a, 1, 0x1000, 0x1000, &at, nullptr, true) == kRgfxBlockOk &&
		          rgfxBlockAlloc(x.t, a, 1, 0x1000, 0x1000, &at, nullptr, true) == kRgfxBlockOk,
		      "but not for WindowServer, to the last entry");
		check(rgfxBlockAlloc(x.t, a, 1, 0x1000, 0x1000, &at, nullptr, true) ==
		          kRgfxBlockTableFull,
		      "past which it is full for WindowServer too");
		check(x.t.reservedPeakCount == 2, "the peak counts WindowServer's entries");
	}
	{
		// With no reserve, every allocation is as before.
		Table x;
		uint64_t at = 0;
		int made = 0;
		while (rgfxBlockAlloc(x.t, b, 2, 0x1000, 0x1000, &at) == kRgfxBlockOk)
			made++;
		check(made == 8 && x.t.reservedCount == 0, "no reserve: every entry is anyone's");
	}
	std::printf("blocks_test: OK (%d checks)\n", checks);
	return 0;
}
