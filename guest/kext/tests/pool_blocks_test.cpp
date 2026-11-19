// The native test of RgfxPoolBlocks.hpp: the take's rounded length, the check of the
// host's answer, and a client's list of its pool blocks growing to its most, each block taken off
// by its offset or at the close, and none listed twice.
#include "RgfxPoolBlocks.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

// The wire's numbers as the kext reads them (design.md; wire.rs asserts the same).
static_assert(RGFX_FEATURE_POOL == (1ull << 57), "POOL is bit 57");
static_assert(RGFX_POOL_BLOCK_ALIGN == 0x10000, "64 KiB blocks");
static_assert(RGFX_POOL_TAKE_MAX == (1ull << 32), "4 GiB at most");
static_assert(RGFX_REG_POOL_BASE_LO == 0x44 && RGFX_REG_POOL_BASE_HI == 0x48, "POOL_BASE");
static_assert(RGFX_REG_POOL_LENGTH_LO == 0x4c && RGFX_REG_POOL_LENGTH_HI == 0x50, "POOL_LENGTH");
static_assert(RGFX_CMD_POOL_TAKE == 763 && RGFX_CMD_POOL_GIVE == 764, "the pool's commands");
static_assert(RGFX_EVT_POOL_BLOCK == 1040, "POOL_BLOCK");
static_assert(sizeof(RgfxPoolTake) == 32 && sizeof(RgfxPoolGive) == 40, "the pool's messages");
static_assert(sizeof(RgfxPoolBlock) == 48, "POOL_BLOCK's record");
static_assert(offsetof(RgfxPoolBlock, gpu_address) == 40, "the block's GPU address");
static_assert(RGFX_FEATURE_POOL_ADDRESSES == (1ull << 59), "POOL_ADDRESSES is bit 59");
static_assert(kRgfxKextFeaturePool == (1 << 15), "the kext's pool bit");
static_assert(kRgfxKextFeaturePoolAddresses == (1 << 16), "the kext's pool addresses bit");

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "pool_blocks_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

constexpr uint64_t kAlign = RGFX_POOL_BLOCK_ALIGN;

// Grows the list as the kext does where it is full: false where it holds its most.
bool grow(RgfxPoolBlockList &list, std::vector<std::vector<RgfxHeldPoolBlock>> &arrays)
{
	const uint32_t to = rgfxPoolBlocksGrowTo(list);
	if (to == 0)
		return false;
	if (to == list.capacity)
		return true;
	arrays.emplace_back(to);
	RgfxHeldPoolBlock *old = rgfxPoolBlocksGrow(list, arrays.back().data(), to);
	check(old == nullptr || arrays.size() >= 2, "the old array is handed back");
	return true;
}
} // namespace

int main()
{
	// An answer is whole from its length on: a host's before POOL_ADDRESSES (40 bytes) too.
	check(!rgfxPoolAnswerWhole(32), "an answer cut before its length is not whole");
	check(!rgfxPoolAnswerWhole(39), "an answer cut in its length is not whole");
	check(rgfxPoolAnswerWhole(40), "an answer of a host before POOL_ADDRESSES is whole");
	check(rgfxPoolAnswerWhole(sizeof(RgfxPoolBlock)), "an answer with its GPU address is whole");
	// The take's length: rounded up to 64 KiB; none for 0 bytes or past 4 GiB.
	check(rgfxPoolTakeLength(0) == 0, "0 bytes take nothing");
	check(rgfxPoolTakeLength(1) == kAlign, "a byte takes a block");
	check(rgfxPoolTakeLength(kAlign) == kAlign, "a block takes a block");
	check(rgfxPoolTakeLength(kAlign + 1) == 2 * kAlign, "past a block takes two");
	check(rgfxPoolTakeLength(RGFX_POOL_TAKE_MAX) == RGFX_POOL_TAKE_MAX, "4 GiB at most");
	check(rgfxPoolTakeLength(RGFX_POOL_TAKE_MAX + 1) == 0, "past 4 GiB takes nothing");

	// The host's answer: the rounded length, aligned, within the pool.
	const uint64_t pool = 64ull << 30;
	check(rgfxPoolAnswerFits(pool, 1, 0, kAlign), "the first block");
	check(rgfxPoolAnswerFits(pool, 3 * kAlign, pool - 3 * kAlign, 3 * kAlign), "the last bytes");
	check(!rgfxPoolAnswerFits(pool, 3 * kAlign, pool - 2 * kAlign, 3 * kAlign), "past the pool");
	check(!rgfxPoolAnswerFits(pool, kAlign, pool, kAlign), "at the pool's end");
	check(!rgfxPoolAnswerFits(pool, kAlign, UINT64_MAX - kAlign + 1, kAlign), "an offset that wraps");
	check(!rgfxPoolAnswerFits(pool, kAlign, kAlign / 2, kAlign), "an offset not aligned");
	check(!rgfxPoolAnswerFits(pool, kAlign, 0, 2 * kAlign), "a length not the take's");
	check(!rgfxPoolAnswerFits(pool, 2 * kAlign, 0, kAlign), "a length short of the take");
	check(!rgfxPoolAnswerFits(pool, 0, 0, 0), "a take of nothing");
	check(!rgfxPoolAnswerFits(0, kAlign, 0, kAlign), "no pool");

	// A Private take's length: rounded up to 64 KiB; none for 0 bytes or past 1 TiB.
	check(rgfxPrivateTakeLength(0) == 0, "0 bytes take no Private block");
	check(rgfxPrivateTakeLength(1) == RGFX_PRIVATE_BLOCK_ALIGN, "a byte takes a Private block");
	check(rgfxPrivateTakeLength(RGFX_PRIVATE_BLOCK_ALIGN + 1) == 2 * RGFX_PRIVATE_BLOCK_ALIGN,
	      "past a Private block takes two");
	check(rgfxPrivateTakeLength(RGFX_PRIVATE_TAKE_MAX) == RGFX_PRIVATE_TAKE_MAX, "1 TiB at most");
	check(rgfxPrivateTakeLength(RGFX_PRIVATE_TAKE_MAX + 1) == 0, "past 1 TiB takes nothing");
	// The host's answer: at least the rounded length, a multiple of 64 KiB, at an address that
	// is not 0 and does not wrap.
	const uint64_t at = 1ull << 44;
	const uint64_t p = RGFX_PRIVATE_BLOCK_ALIGN;
	check(rgfxPrivateAnswerFits(1, at, p), "a Private block of the take's length");
	check(rgfxPrivateAnswerFits(1, at + 0x8000, 256ull << 20), "a longer one, anywhere");
	check(!rgfxPrivateAnswerFits(2 * p, at, p), "one short of the take");
	check(!rgfxPrivateAnswerFits(p, at, p + 0x1000), "one not a multiple of 64 KiB");
	check(!rgfxPrivateAnswerFits(p, 0, p), "one at address 0");
	check(!rgfxPrivateAnswerFits(p, UINT64_MAX - p + 2, p), "one that wraps");
	check(!rgfxPrivateAnswerFits(0, at, p), "a Private take of nothing");

	// The list: empty, grown at its first block, each block listed once, taken by its offset.
	std::vector<std::vector<RgfxHeldPoolBlock>> arrays;
	RgfxPoolBlockList list = {nullptr, 0, 0};
	int maps[4] = {};
	check(!rgfxPoolBlocksAdd(list, 0, kAlign, &maps[0]), "an empty list holds nothing");
	check(rgfxPoolBlocksGrowTo(list) == kRgfxPoolBlocksFirst, "the first array");
	check(grow(list, arrays) && list.capacity == kRgfxPoolBlocksFirst, "grown");
	check(rgfxPoolBlocksAdd(list, 0, 2 * kAlign, &maps[0]), "a block");
	check(rgfxPoolBlocksAdd(list, 4 * kAlign, kAlign, &maps[1]), "another");
	check(!rgfxPoolBlocksAdd(list, kAlign, kAlign, &maps[2]), "a block over the first");
	check(!rgfxPoolBlocksAdd(list, 3 * kAlign, 2 * kAlign, &maps[2]), "a block over the second");
	check(!rgfxPoolBlocksAdd(list, 8 * kAlign, 0, &maps[2]), "a block of nothing");
	check(rgfxPoolBlocksAdd(list, 2 * kAlign, kAlign, &maps[2]), "a block between them");
	check(list.count == 3, "three listed");
	RgfxHeldPoolBlock b = {};
	check(!rgfxPoolBlocksTake(list, kAlign, &b), "no block starts inside another");
	check(rgfxPoolBlocksTake(list, 0, &b) && b.length == 2 * kAlign && b.map == &maps[0],
	      "the first block, its length and its mapping");
	check(!rgfxPoolBlocksTake(list, 0, &b), "taken once");
	check(rgfxPoolBlocksAdd(list, 0, kAlign, &maps[3]), "its bytes listed again");
	check(list.count == 3, "three listed again");

	// The close takes every block, each once.
	int seen = 0;
	while (rgfxPoolBlocksTakeAny(list, &b))
		seen |= 1 << (static_cast<int *>(b.map) - maps);
	check(seen == 0b1110 && list.count == 0, "the close takes the rest, each once");

	// Grown up to its most, keeping every block; then full.
	for (uint32_t i = 0; i < kRgfxPoolBlocksMax; i++) {
		if (list.count == list.capacity)
			check(grow(list, arrays), "grows while below its most");
		check(rgfxPoolBlocksAdd(list, uint64_t(i) * kAlign, kAlign, &maps[i % 4]), "listed");
	}
	check(list.capacity == kRgfxPoolBlocksMax, "its most");
	check(rgfxPoolBlocksGrowTo(list) == 0 && !grow(list, arrays), "no more");
	check(!rgfxPoolBlocksAdd(list, uint64_t(kRgfxPoolBlocksMax) * kAlign, kAlign, &maps[0]),
	      "a full list lists no more");
	bool kept = true;
	for (uint32_t i = 0; i < kRgfxPoolBlocksMax; i += 997)
		kept = kept && rgfxPoolBlocksTake(list, uint64_t(i) * kAlign, &b) && b.length == kAlign;
	check(kept, "every block kept through each growth");
	check(rgfxPoolBlocksGrowTo(list) == kRgfxPoolBlocksMax, "room again, no growth");

	std::printf("pool_blocks_test: OK (%d checks)\n", checks);
	return 0;
}
