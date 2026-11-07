/*
 * serializer_tests: the reims-graphics guest serialiser's self-contained check,
 * run on the Mac (guest/bundle/Makefile `test`). It builds a plain-buffer region,
 * a ring writer over it, and a serialiser, posts every message the slice covers,
 * and reads each back with a minimal host-side consumer (the mirror of
 * crates/reims-graphics/src/ring.rs), asserting the type, size and fields on the
 * wire. No Metal, no device: this validates the serialiser's bytes alone.
 *
 * Exits 0 and prints a one-line summary when every check passes; aborts on the
 * first failure.
 */

#include <IOKit/IOReturn.h>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "RgfxArgumentLayout.hpp"
#include "RgfxBinaryArchiveFile.hpp"
#include "RgfxCaps.hpp"
#include <utility>
#include "RgfxEncoder.hpp"
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

// A chunk source standing in for the kext: it hands each first id of `firsts` in turn, then
// answers none left, and counts its calls (the user-client calls a create would make).
struct FakeChunks {
	std::vector<uint64_t> firsts;
	size_t next = 0;
	int calls = 0;
};

bool fakeChunk(void *context, uint64_t *first_id)
{
	FakeChunks *f = static_cast<FakeChunks *>(context);
	f->calls++;
	if (f->next == f->firsts.size())
		return false;
	*first_id = f->firsts[f->next++];
	return true;
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
                  offsetof(RgfxDeviceInfo, sampler_reduction_support) == 0x452,
              "the device info's feature and reduction bytes");

// The feature words: the kext's where it answers the call, and where it answers
// kIOReturnUnsupported (a kext older than the call) or fails otherwise, the blob's bits 0 to 2
// and no kext feature.
void test_caps_features()
{
	RgfxCaps caps;
	caps.takeFeatures(kIOReturnUnsupported, 0, 0);
	CHECK(caps.features() == 0 && caps.kextFeatures() == 0); // no blob: nothing

	RgfxDeviceInfo info = {};
	info.version = RGFX_DEVICE_INFO_VERSION;
	info.features_lo = RGFX_FEATURE_SURFACE_PAGES | RGFX_FEATURE_CB_SUBMITTED |
	                   RGFX_FEATURE_QUIET_SCHEDULED; // bit 3 offered, but taken by no kext
	CHECK(caps.load(info));
	CHECK(caps.features() == 0); // nothing before the probe
	caps.takeFeatures(kIOReturnUnsupported, 0xffff, 0x1f);
	CHECK(caps.features() == (RGFX_FEATURE_SURFACE_PAGES | RGFX_FEATURE_CB_SUBMITTED));
	CHECK(caps.kextFeatures() == 0);
	caps.takeFeatures(kIOReturnBadArgument, 0xffff, 0x1f);
	CHECK(caps.features() == (RGFX_FEATURE_SURFACE_PAGES | RGFX_FEATURE_CB_SUBMITTED));
	CHECK(caps.kextFeatures() == 0);
	// A kext that answers gives both words as they are, bits from 3 and 32 among them.
	const uint64_t word = RGFX_FEATURE_CB_SUBMITTED | RGFX_FEATURE_CB_PARTS | (1ull << 40);
	caps.takeFeatures(KERN_SUCCESS, word, 1ull << 4);
	CHECK(caps.features() == word && caps.kextFeatures() == 1ull << 4);
	caps.takeFeatures(KERN_SUCCESS, 0, 0);
	CHECK(caps.features() == 0 && caps.kextFeatures() == 0);
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
	info.format_count = 1;
	info.formats[0].format = 80; // BGRA8Unorm
	info.formats[0].texture_buffer_alignment = 16;
	info.formats[0].sparse_tiles[0][0] = 64; // 1 sample, 16 KiB pages
	info.formats[0].sparse_tiles[0][1] = 64;
	info.formats[0].sparse_tiles[0][2] = 1;
	info.formats[0].sparse_tiles[8][0] = 128; // 4 samples, 256 KiB pages
	info.formats[0].sparse_tiles[8][1] = 128;
	info.formats[0].sparse_tiles[8][2] = 1;
	info.formats[1].format = 70; // past format_count: not read
	info.formats[1].texture_buffer_alignment = 16;
	const char *arch = "test_arch";
	info.architecture_name_len = static_cast<uint32_t>(std::strlen(arch));
	std::memcpy(info.architecture_name, arch, info.architecture_name_len);
	info.sample_counts[0] = 1;               // count 1
	info.sample_counts[2] = 1;               // count 4
	info.primitive_types[3] = 1;
	info.write_rounding_modes[1] = 1; // mode 2
	info.limits[kRgfxLimitDeviceLinearTextureAlignmentBytes] = 64;
	info.limits[kRgfxLimitLinearTextureAlignmentBytes] = 256;
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
	CHECK(caps.textureBufferAlignment(70) == 0); // listed past the count
	CHECK(caps.textureBufferAlignment(130) == 0); // not listed
	// The linear alignment is the one limit for every format, never 0: those with a
	// texture-buffer answer, and those without, where Metal ends the process.
	CHECK(caps.minimumLinearTextureAlignment(80) == 256);
	CHECK(caps.minimumLinearTextureAlignment(70) == 256);  // listed past the count
	CHECK(caps.minimumLinearTextureAlignment(130) == 256); // not listed (ASTC)
	CHECK(caps.minimumLinearTextureAlignment(0) == 256);   // Invalid
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
	CHECK(caps.limit(kRgfxLimitMaxViewportCount) == 16);
	CHECK(caps.flag(kRgfxFlagAreRasterOrderGroupsSupported));
	CHECK(caps.flag(kRgfxFlagSupportsTileShaders));
	CHECK(!caps.flag(kRgfxFlagSupportsImageBlocks));
	CHECK(std::strcmp(caps.name(), "NVIDIA GeForce RTX 5080 Laptop GPU") == 0);
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
	RgfxCreated hp = ser.createHeapOverPages(past, 0, 1, 0x60000, 0x4000);
	RgfxTextureDescriptor desc = {};
	desc.pixel_format = 80;
	desc.texture_type = 2;
	desc.width = desc.height = desc.depth = 1;
	desc.mipmap_level_count = desc.array_length = desc.sample_count = 1;
	RgfxCreated t = ser.createTexture(desc, RGFX_OBJ_NONE, 0, 256, past);
	RgfxCreated tm = ser.createManagedTexture(desc, 9, 4, 1, 1, 0, 256, past);
	for (const RgfxCreated &c : {h, hh, b, bh, bp, wraps, hp, t, tm})
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

	// Sampler: the float clamps are stored as bits.
	RgfxSamplerParams sp = {};
	sp.min_filter = 1;
	sp.mag_filter = 1;
	sp.mip_filter = 1;
	sp.max_anisotropy = 4;
	sp.lod_min_clamp = 0.0f;
	sp.lod_max_clamp = 7.5f;
	sp.flags = 0x1;
	RgfxCreated s = ser.createSampler(sp);
	CHECK(s.posted);
	auto sm = expectMessage<RgfxSamplerCreate>(ring, RGFX_CMD_SAMPLER_CREATE);
	CHECK(sm.obj_id == s.obj_id && sm.max_anisotropy == 4 && sm.flags == 0x1);
	float lodMax;
	std::memcpy(&lodMax, &sm.lod_max_clamp_bits, 4);
	CHECK(lodMax == 7.5f);

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
	rp.flags = RGFX_RENDER_PIPELINE_RASTERIZATION_ENABLED;

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

	// The host takes everything 20 ms from now.
	std::thread host([&ring, full] {
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		__atomic_store_n(reinterpret_cast<uint64_t *>(ring.mem.data() +
		                                              __builtin_offsetof(RgfxRingHeader, consumer)),
		                 full, __ATOMIC_RELEASE);
	});
	std::atomic<int> waits{0};
	uint64_t seen = 0, producerAtPublish = ~0ull, seqno = 0;
	auto start = std::chrono::steady_clock::now();
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
void test_ring_post_waits_for_the_room_record()
{
	constexpr uint32_t kData = 0x1000;
	TestRing ring(kData);
	RgfxRingWriter writer(ring.base(), kData, 1, TestRing::doorbell, nullptr);
	auto room = std::make_shared<RgfxRingRoomEvents>();
	writer.setRoomEvents(room);
	auto waiting = [&ring] {
		return __atomic_load_n(reinterpret_cast<uint32_t *>(
		                           ring.mem.data() +
		                           __builtin_offsetof(RgfxRingHeader, producer_waiting)),
		                       __ATOMIC_SEQ_CST);
	};
	uint8_t msg[256] = {};
	CHECK(writer.post(RGFX_CMD_COMMAND_BUFFER, msg, sizeof(msg), nullptr));
	CHECK(waiting() == 0);
	while (writer.tryPost(RGFX_CMD_COMMAND_BUFFER, msg, sizeof(msg), nullptr)) {
	}
	std::atomic<bool> posted{false};
	bool postedOK = false;
	std::thread poster([&] {
		postedOK = writer.post(RGFX_CMD_COMMAND_BUFFER, msg, sizeof(msg), nullptr);
		posted = true;
	});
	for (int i = 0; i < 5000 && waiting() == 0; i++)
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	CHECK(waiting() == 1);
	// The host takes one message and posts no record: the post still waits.
	RgfxMsgHeader h = {};
	std::vector<uint8_t> bytes;
	CHECK(ring.next(&h, &bytes));
	std::this_thread::sleep_for(std::chrono::milliseconds(3));
	CHECK(!posted);
	// Its record: the post looks again, and finds the room.
	const auto start = std::chrono::steady_clock::now();
	room->arrived();
	poster.join();
	CHECK(posted && postedOK);
	CHECK(std::chrono::steady_clock::now() - start < std::chrono::milliseconds(15));

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

// A command buffer's message is built where it is posted from: on the ring while it fits half of
// it, else in a block, which it moves to once as it outgrows the ring and grows from by doubling,
// and which the COMMAND_BUFFER_IN_REGION names, the whole COMMAND_BUFFER in it; the block goes to
// whoever holds the posted message, the ones it outgrew back to the pool. The surface uses a
// commit puts first lead the op stream, in the order put, before every op, however many.
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

	// Past the ring: built in a block from the op that outgrows it, grown block to block, the
	// outgrown ones back to the pool; more uses than the room before the header move the ops.
	{
		RgfxEncoder enc(writer, 1, 3, 0x100);
		enc.setMessageBlocks(blocks, 0);
		enc.beginBlitPass();
		int fills = 0;
		while (!enc.inBlock()) {
			enc.blitFillBuffer(static_cast<RgfxObjId>(fills), 0, 4, 1);
			fills++;
		}
		CHECK(blocks->taken.size() == 1 && fills > 2048 / 48 && fills < 4096 / 32);
		for (; fills < 5000; fills++) // 5000 fills of 40 bytes: 200 KB, past one 64 KiB block
			enc.blitFillBuffer(static_cast<RgfxObjId>(fills), 0, 4, 1);
		enc.endEncoder();
		CHECK(blocks->taken.size() == 3 && blocks->given.size() == 2);
		std::vector<RgfxOpSurfaceUse> u = uses(100, 1000);
		enc.prependSurfaceUses(u.data(), u.size());
		uint64_t seqno = 0;
		RgfxMessageBlock held = {};
		CHECK(enc.finish(&seqno, {}, {}, &held));
		CHECK(held.ptr == blocks->taken.back().ptr && blocks->given.size() == 2);
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(ring.next(&h, &bytes) && h.type == RGFX_CMD_COMMAND_BUFFER_IN_REGION);
		RgfxCommandBufferInRegion ref = {};
		std::memcpy(&ref, bytes.data(), sizeof(ref));
		CHECK(ref.region_offset >= held.offset && ref.region_offset % 8 == 0 &&
		      ref.region_offset + ref.byte_length <= held.offset + held.size);
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
		CHECK(!enc.inBlock() && enc.messageSize() > 2048);
		RgfxMessageBlock held = {};
		CHECK(!enc.finish(nullptr, {}, {}, &held) && held.ptr == nullptr);
		RgfxMsgHeader h = {};
		std::vector<uint8_t> bytes;
		CHECK(!ring.next(&h, &bytes));
		blocks->refuse = false;
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

// A library's content key is the SHA-256 of the prefix and the bytes around the bitcode section:
// the digest computed independently (and pinned on the host's side too), the same whatever the
// section holds, and none for bytes that are no .metallib or whose section lies outside them.
void test_library_key()
{
	static const uint8_t expect[32] = {
		0xc0, 0x99, 0x93, 0x08, 0x93, 0xa6, 0x91, 0xa1, 0xaa, 0xed, 0xfa, 0x6d, 0xee, 0xd4, 0x1a, 0x58,
		0x09, 0x95, 0x3e, 0x36, 0xa8, 0xe6, 0x9e, 0xd9, 0x86, 0xe1, 0xb2, 0x11, 0xff, 0x4e, 0x21, 0x7d,
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

// A no-copy buffer's block and the app's memory (RgfxMemoryShadow): the block starts as the
// memory; each sync moves the bytes one side changed since the two last agreed, byte by byte,
// so a CPU write and a GPU write beside it in one word both stand; memory mapped read-only is
// never written; the encoder's notes sync each shadow once and keep the ones written.
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

	// Read-only memory: synced in, never written.
	std::vector<uint8_t> ro(64, 3), roBlock(64, 0);
	auto r = std::make_shared<RgfxMemoryShadow>(ro.data(), roBlock.data(), ro.size(), false);
	CHECK(!r->writable() && roBlock == ro);
	roBlock[5] = 9;
	r->writeBack();
	CHECK(ro[5] == 3);

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
	std::vector<std::shared_ptr<RgfxMemoryShadow>> written = rgfxShadowsSyncIn(uses);
	CHECK(aBlock[0] == 5 && bBlock[0] == 6);
	CHECK(written.size() == 1 && written[0] == a);
	CHECK(rgfxShadowsSyncIn({}).empty());
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
		static bool block(void *c, const void *address, uint64_t length, bool writable,
		                  uint64_t *offset, uint64_t *size)
		{
			Source *s = static_cast<Source *>(c);
			s->address = address;
			s->length = length;
			s->writable = writable;
			*offset = s->offset;
			*size = 0x1000;
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
	CHECK(b.ok && b.offset == src.offset && b.size == 0x1000 && b.ptr == nullptr);
	CHECK(src.address == &marker && src.length == 0x4000 && src.writable);
	CHECK(src.released.empty());
	src.offset = mem.size() - 0x800;
	CHECK(!region.acquirePages(&marker, 0x4000, false).ok && !src.writable);
	CHECK(src.released.size() == 1 && src.released[0] == mem.size() - 0x800);
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

int main()
{
	test_id_space();
	test_obj_ids_are_given_once();
	test_region_pages_blocks();
	test_region_write_combined_parts();
	test_address_picks();
	test_resource_id_chunks();
	test_small_resource_id_chunks();
	test_caps();
	test_spi_features();
	test_family_names();
	test_vendor_names();
	test_host_resource_options();
	test_caps_features();
	test_archive_file();
	test_lifecycle_messages();
	test_creates_with_no_address_post_nothing();
	test_library_message();
	test_pipeline_messages();
	test_mesh_pipeline_messages();
	test_mesh_ops();
	test_ring_wrap_and_no_room();
	test_ring_post_waits_for_room();
	test_ring_post_waits_for_the_room_record();
	test_ring_room_sleeps_count_for_the_line();
	test_command_buffer_encoder();
	test_command_buffer_storage();
	test_video_command_buffer();
	test_render_pass_sample_positions();
	test_render_pass_visibility();
	test_forwarded_lifecycle();
	test_library_key();
	test_library_key_of_file();
	test_keyed_library_messages();
	test_heap_ranges();
	test_heap_texture_size_query();
	test_memory_shadow();
	test_argument_layout_queries();
	test_argument_layout_parse();
	rgfxSlabTests();
	rgfxLibraryStreamTests();
	rgfxFunctionConstantsTests();
	std::printf("serializer_tests: OK (%d checks)\n", g_checks.load());
	return 0;
}
