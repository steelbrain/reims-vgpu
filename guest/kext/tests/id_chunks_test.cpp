// The native test of RgfxIdChunks.hpp's grants (rgfxIdChunkGrant): a guest that took
// RGFX_FEATURE_SMALL_ID_CHUNKS takes its chunks a unit of RGFX_RESOURCE_ID_SMALL_CHUNK ids at a
// time, granted by RESOURCE_ID_SMALL_CHUNK, so the pool holds four times as many processes'
// first chunks; one that did not takes aligned chunks of RGFX_RESOURCE_ID_CHUNK ids, granted by
// RESOURCE_ID_CHUNK, and never posts a RESOURCE_ID_SMALL_CHUNK; and the two kinds of chunk, taken
// from the one table, never hand out the same id.
#include "RgfxFeatures.hpp"
#include "RgfxIdChunks.hpp"

#include <cstdio>
#include <cstdlib>
#include <set>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "id_chunks_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

constexpr uint32_t kUnits = RGFX_RESOURCE_ID_SMALL_CHUNKS;
constexpr uint32_t kRun = RGFX_RESOURCE_ID_CHUNK / RGFX_RESOURCE_ID_SMALL_CHUNK;

// The ids a grant hands out, as the device takes them: a RESOURCE_ID_CHUNK's chunk k holds ids
// 1 + 64k to 64 + 64k, a RESOURCE_ID_SMALL_CHUNK's chunk k ids 1 + 16k to 16 + 16k.
void idsOf(const RgfxIdGrant &g, std::set<uint64_t> *ids, bool *overlap)
{
	const uint64_t size = g.type == RGFX_CMD_RESOURCE_ID_SMALL_CHUNK ? RGFX_RESOURCE_ID_SMALL_CHUNK
	                                                                  : RGFX_RESOURCE_ID_CHUNK;
	for (uint64_t id = 1 + g.chunk * size; id <= size + g.chunk * size; id++)
		*overlap |= !ids->insert(id).second;
}

void test_small_chunks()
{
	const void *owners[kUnits] = {};
	int a = 0;
	int b = 0;
	// One unit at a time, in order, each its own small chunk.
	for (uint32_t k = 0; k < 3; k++) {
		const RgfxIdGrant g = rgfxIdChunkGrant(owners, RGFX_FEATURE_SMALL_ID_CHUNKS, &a);
		check(g.type == RGFX_CMD_RESOURCE_ID_SMALL_CHUNK,
		      "a guest of small chunks grants small ones");
		check(g.first == static_cast<int32_t>(k) && g.chunk == k && g.units == 1,
		      "a small chunk is one unit, numbered as its unit");
		check(g.firstId == 1 + k * RGFX_RESOURCE_ID_SMALL_CHUNK, "a small chunk's first id");
	}
	// The pool holds RGFX_RESOURCE_ID_SMALL_CHUNKS processes' first chunks, four times
	// RGFX_RESOURCE_ID_CHUNKS.
	uint32_t taken = 3;
	while (rgfxIdChunkGrant(owners, RGFX_FEATURE_SMALL_ID_CHUNKS, &b).first >= 0)
		taken++;
	check(taken == kUnits && kUnits == 4 * RGFX_RESOURCE_ID_CHUNKS,
	      "small chunks hold four times as many processes");
	check(rgfxIdChunksHeld(owners, kUnits) == kUnits, "every unit held");
	check(rgfxIdChunkGrant(owners, RGFX_FEATURE_SMALL_ID_CHUNKS, &b).first < 0,
	      "none once every unit is held");
	// A close's free gives its units back, the first of them taken next.
	check(rgfxIdChunksHand(owners, kUnits, &a, nullptr) == 3, "a's three units freed");
	const RgfxIdGrant again = rgfxIdChunkGrant(owners, RGFX_FEATURE_SMALL_ID_CHUNKS, &b);
	check(again.first == 0 && again.firstId == 1, "a freed unit is taken again");
}

void test_chunks_without_the_feature()
{
	const void *owners[kUnits] = {};
	int a = 0;
	// Every feature but SMALL_ID_CHUNKS: chunks of 64 ids, aligned runs of four units.
	const uint64_t features = ~static_cast<uint64_t>(RGFX_FEATURE_SMALL_ID_CHUNKS);
	uint32_t grants = 0;
	for (;;) {
		const RgfxIdGrant g = rgfxIdChunkGrant(owners, features, &a);
		check(g.type == RGFX_CMD_RESOURCE_ID_CHUNK,
		      "a guest that did not take SMALL_ID_CHUNKS never posts RESOURCE_ID_SMALL_CHUNK");
		if (g.first < 0)
			break;
		check(g.units == kRun && g.first == static_cast<int32_t>(grants * kRun) &&
		          g.chunk == grants,
		      "a chunk is four aligned units, numbered as before");
		check(g.firstId == 1 + grants * RGFX_RESOURCE_ID_CHUNK, "a chunk's first id, as before");
		grants++;
	}
	check(grants == RGFX_RESOURCE_ID_CHUNKS, "the pool holds RGFX_RESOURCE_ID_CHUNKS chunks");
	// A run of four free units that does not start at a multiple of four is no chunk.
	rgfxIdChunkGive(owners, 1, 4);
	check(rgfxIdChunkGrant(owners, features, &a).first < 0, "no misaligned chunk");
	rgfxIdChunkGive(owners, 0, 1);
	const RgfxIdGrant g = rgfxIdChunkGrant(owners, features, &a);
	check(g.first == 0 && g.chunk == 0, "the aligned run, once free");
}

// The kext grants from the feature word it negotiated (rgfxFeatureWord over rgfxDriverFeatures):
// from a host that does not offer SMALL_ID_CHUNKS, such as one from before it, only
// RESOURCE_ID_CHUNKs; from one that does, small chunks.
void test_grants_follow_the_offer()
{
	const void *owners[kUnits] = {};
	int a = 0;
	const uint64_t old =
	    kRgfxOfferedFeatures | (kRgfxCarriedFeatures & ~uint64_t{RGFX_FEATURE_SMALL_ID_CHUNKS});
	const uint64_t oldWord = rgfxFeatureWord(old, rgfxDriverFeatures(old));
	check((oldWord & RGFX_FEATURE_SMALL_ID_CHUNKS) == 0, "an old host's word has no small chunks");
	for (int i = 0; i < 8; i++)
		check(rgfxIdChunkGrant(owners, oldWord, &a).type == RGFX_CMD_RESOURCE_ID_CHUNK,
		      "an old host is granted RESOURCE_ID_CHUNKs only");
	const uint64_t now = old | RGFX_FEATURE_SMALL_ID_CHUNKS;
	const uint64_t word = rgfxFeatureWord(now, rgfxDriverFeatures(now));
	check(rgfxIdChunkGrant(owners, word, &a).type == RGFX_CMD_RESOURCE_ID_SMALL_CHUNK,
	      "a host that offers it is granted small chunks");
}

// The small chunks and the 64-id chunks, taken in turn from the one table, never share an id.
void test_kinds_never_share_an_id()
{
	const void *owners[kUnits] = {};
	int a = 0;
	int b = 0;
	std::set<uint64_t> ids;
	bool overlap = false;
	uint32_t i = 0;
	for (;; i++) {
		// Two small chunks for every 64-id one, so the small ones leave runs half taken.
		const bool small = i % 3 != 2;
		const RgfxIdGrant g =
		    rgfxIdChunkGrant(owners, small ? RGFX_FEATURE_SMALL_ID_CHUNKS : 0, small ? &a : &b);
		if (g.first < 0) {
			if (small)
				break;
			continue;
		}
		idsOf(g, &ids, &overlap);
	}
	check(!overlap, "no id is granted twice");
	check(ids.size() == static_cast<size_t>(kUnits) * RGFX_RESOURCE_ID_SMALL_CHUNK,
	      "every id of the pool is granted once");
	check(*ids.begin() == 1 && *ids.rbegin() == RGFX_RESOURCE_ID_CHUNKS * RGFX_RESOURCE_ID_CHUNK,
	      "the grants are the pool's ids, below RGFX_RESOURCE_ID_LIMIT");
	check(*ids.rbegin() < RGFX_RESOURCE_ID_LIMIT, "below the limit");
	// Freed in the middle, the kinds still share nothing.
	rgfxIdChunksHand(owners, kUnits, &b, nullptr);
	ids.clear();
	for (uint32_t u = 0; u < kUnits; u++) {
		if (owners[u] == nullptr)
			continue;
		RgfxIdGrant held = {RGFX_CMD_RESOURCE_ID_SMALL_CHUNK, u, static_cast<int32_t>(u), 1, 0};
		idsOf(held, &ids, &overlap);
	}
	while (true) {
		const RgfxIdGrant g = rgfxIdChunkGrant(owners, 0, &b);
		if (g.first < 0)
			break;
		idsOf(g, &ids, &overlap);
	}
	check(!overlap, "chunks taken after a free share no id with the small ones held");
}
} // namespace

int main()
{
	test_small_chunks();
	test_chunks_without_the_feature();
	test_grants_follow_the_offer();
	test_kinds_never_share_an_id();
	std::printf("id_chunks_test: %d checks passed\n", checks);
	return 0;
}
