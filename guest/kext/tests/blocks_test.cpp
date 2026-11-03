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
	RgfxBlockTable t = {blocks, 0, 8, kStart, kEnd, 0};
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

		Table y = x;
		y.t.blocks = y.blocks;
		check(rgfxBlockFreeOwner(x.t, a) == 3 && x.t.count == 1 && x.t.blocks[0].owner == b,
		      "an answered close frees the client's blocks");

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
	std::printf("blocks_test: OK (%d checks)\n", checks);
	return 0;
}
