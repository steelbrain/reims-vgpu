// RgfxIdChunks: who holds each chunk of the guest's texture resource ids
// (RGFX_RESOURCE_ID_CHUNKS of RGFX_RESOURCE_ID_CHUNK ids), each owner's slot null where none
// does. A client takes chunks as its context makes objects, and gives them back once its close is
// answered: the device then maps none of their ids to an object of the context. A close not
// answered in time hands its client's chunks to the close's token (RgfxCloseTable.hpp), so no
// other client takes one before the answer comes.
//
// The caller holds the contexts' lock around every call. Header-only and free of the kernel's
// headers, so its native test (tests/close_table_test.cpp) builds it as the kext does.
//
// Our own code.
#ifndef RGFX_ID_CHUNKS_HPP_
#define RGFX_ID_CHUNKS_HPP_

#include <stdint.h>

// The first free of the `count` chunks of `owners`, now `owner`'s: its index, or -1 where every
// one is held.
inline int32_t rgfxIdChunkTake(const void **owners, uint32_t count, const void *owner)
{
	for (uint32_t i = 0; i < count; i++) {
		if (owners[i] == nullptr) {
			owners[i] = owner;
			return static_cast<int32_t>(i);
		}
	}
	return -1;
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
