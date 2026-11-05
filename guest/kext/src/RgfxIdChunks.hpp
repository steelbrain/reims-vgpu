// RgfxIdChunks: who holds each unit of the guest's texture resource ids
// (RGFX_RESOURCE_ID_SMALL_CHUNKS units of RGFX_RESOURCE_ID_SMALL_CHUNK ids), each owner's slot
// null where none does. A client takes chunks as its context makes objects, and gives them back
// once its close is answered: the device then maps none of their ids to an object of the
// context. A close not answered in time hands its client's units to the close's token
// (RgfxCloseTable.hpp), so no other client takes one before the answer comes.
//
// A chunk is one unit where the guest took RGFX_FEATURE_SMALL_ID_CHUNKS (RESOURCE_ID_SMALL_CHUNK
// grants it), and otherwise an aligned run of four, chunk k of RGFX_RESOURCE_ID_CHUNK ids being
// units 4k to 4k + 3 (RESOURCE_ID_CHUNK grants it): every texture-making process holds at least
// one chunk until its close, so the smaller chunk lets four times as many processes make
// textures from the same pool (rgfxIdChunkGrant).
//
// The caller holds the contexts' lock around every call. Header-only and free of the kernel's
// headers, so its native tests (tests/close_table_test.cpp, tests/id_chunks_test.cpp) build it as
// the kext does. Our own code; it depends only on the committed wire header.
#ifndef RGFX_ID_CHUNKS_HPP_
#define RGFX_ID_CHUNKS_HPP_

#include <stdint.h>

#include "reims_graphics_wire.h"

// The first free run of `run` units of the `count` of `owners` that starts at a multiple of
// `run`, now `owner`'s: its first unit, or -1 where no such run is free.
inline int32_t rgfxIdChunkTake(const void **owners, uint32_t count, const void *owner,
                               uint32_t run = 1)
{
	for (uint32_t i = 0; run != 0 && i + run <= count; i += run) {
		uint32_t n = 0;
		while (n < run && owners[i + n] == nullptr)
			n++;
		if (n < run)
			continue;
		for (n = 0; n < run; n++)
			owners[i + n] = owner;
		return static_cast<int32_t>(i);
	}
	return -1;
}

// The `run` units of `owners` from `first` are free again.
inline void rgfxIdChunkGive(const void **owners, uint32_t first, uint32_t run)
{
	for (uint32_t n = 0; n < run; n++)
		owners[first + n] = nullptr;
}

// A chunk an ask takes (rgfxIdChunkGrant): the message that tells the device of it, RESOURCE_ID_
// SMALL_CHUNK or RESOURCE_ID_CHUNK, and its `chunk` field; its first unit of the table, and how
// many it holds; and its first id. `first` is -1 where none is free.
struct RgfxIdGrant {
	uint32_t type;
	uint32_t chunk;
	int32_t first;
	uint32_t units;
	uint64_t firstId;
};

// Takes a chunk of the RGFX_RESOURCE_ID_SMALL_CHUNKS units of `owners` for `owner`, as the
// guest's taken features `features` say: one unit, granted by RESOURCE_ID_SMALL_CHUNK, where they
// hold RGFX_FEATURE_SMALL_ID_CHUNKS; else four aligned, granted by RESOURCE_ID_CHUNK, which every
// host knows. A guest that did not take the feature never posts a RESOURCE_ID_SMALL_CHUNK.
inline RgfxIdGrant rgfxIdChunkGrant(const void **owners, uint64_t features, const void *owner)
{
	const bool small = (features & RGFX_FEATURE_SMALL_ID_CHUNKS) != 0;
	const uint32_t run =
	    small ? 1 : static_cast<uint32_t>(RGFX_RESOURCE_ID_CHUNK / RGFX_RESOURCE_ID_SMALL_CHUNK);
	RgfxIdGrant g = {};
	g.type = small ? RGFX_CMD_RESOURCE_ID_SMALL_CHUNK : RGFX_CMD_RESOURCE_ID_CHUNK;
	g.units = run;
	g.first = rgfxIdChunkTake(owners, RGFX_RESOURCE_ID_SMALL_CHUNKS, owner, run);
	if (g.first < 0)
		return g;
	g.chunk = static_cast<uint32_t>(g.first) / run;
	g.firstId = 1 + static_cast<uint64_t>(g.first) * RGFX_RESOURCE_ID_SMALL_CHUNK;
	return g;
}

// How many of the `count` chunks are held.
inline uint32_t rgfxIdChunksHeld(const void *const *owners, uint32_t count)
{
	uint32_t n = 0;
	for (uint32_t i = 0; i < count; i++)
		n += owners[i] != nullptr;
	return n;
}

// `from`'s chunks are `to`'s, or free where `to` is null: how many.
inline uint32_t rgfxIdChunksHand(const void **owners, uint32_t count, const void *from,
                                 const void *to)
{
	uint32_t n = 0;
	for (uint32_t i = 0; from != nullptr && i < count; i++) {
		if (owners[i] == from) {
			owners[i] = to;
			n++;
		}
	}
	return n;
}

#endif // RGFX_ID_CHUNKS_HPP_
