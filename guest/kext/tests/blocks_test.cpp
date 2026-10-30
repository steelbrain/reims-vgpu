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
		check(rgfxBlockFree(x.t, a, at) == kRgfxBlockBusy, "pinned, it is busy");
		rgfxBlockUnpin(x.t, pin);
		check(rgfxBlockFree(x.t, a, at) == kRgfxBlockOk, "unpinned, it is freed");
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
	std::printf("blocks_test: OK (%d checks)\n", checks);
	return 0;
}
