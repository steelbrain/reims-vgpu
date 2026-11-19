/*
 * RgfxObjectModel: the guest bundle's object model for the reims-graphics lane
 * (docs/graphics-kext/design.md, "Stage 3 wire protocol: the Metal-call stream",
 * Decided 6 and 7).
 *
 * Every guest Metal object the bundle wraps carries a 64-bit ObjId the guest
 * picks, unique per device; the host keeps the ObjId -> Platform-handle table.
 * The guest also picks each buffer/texture/heap GPU address and each texture
 * resource id, and the host pins them in the platform's own address space, so a
 * create needs no round trip. Host device addresses never reach the guest.
 *
 * RgfxIdSpace hands out those three number lines for one device (one guest
 * process = one context, Decided 7). Object ids are monotonic and never reuse a
 * value while the device lives: host handles are refcounted, and reuse would
 * alias a live handle. GPU addresses lie in the context's range and follow
 * Metal's rules for a pinned address (allocAddress), since the host pins them;
 * a range comes back once its object's destroy has gone to the host, and is
 * picked again only once the picks have gone round the whole range (next fit),
 * so the host has long let the old pin go. Both start at the context's bases,
 * which the accelerator kext hands out. Where the device and the kext took
 * RGFX_FEATURE_PRIVATE_BLOCKS, a Private object's address lies instead in a
 * block of addresses the host gives the context (allocPrivateAddress): the
 * same rules hold inside the blocks. Texture resource ids are shared
 * by every process on the host's one device, whose ids end below
 * RGFX_RESOURCE_ID_LIMIT, so they come in chunks the kext hands out on demand
 * from one pool for the guest (of RGFX_RESOURCE_ID_SMALL_CHUNK ids where it
 * took RGFX_FEATURE_SMALL_ID_CHUNKS, else RGFX_RESOURCE_ID_CHUNK), and an id
 * goes back to the process's chunks when its texture goes: Metal gives a
 * texture's id back as the texture goes too.
 *
 * This is our own code: it knows nothing of Apple's Metal SPI. The Metal device
 * shell (built later) allocates an ObjId per object it creates and stores it on
 * the wrapper; here we only mint the numbers.
 */

#ifndef RGFX_OBJECT_MODEL_HPP_
#define RGFX_OBJECT_MODEL_HPP_

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

#include "reims_graphics_wire.h"

/// A guest-picked object id, unique per device. 0 names no object.
typedef uint64_t RgfxObjId;

/// The id that names no object, used where a message's field is optional (a
/// device-placed texture's `heap_obj_id`, for one).
static const RgfxObjId RGFX_OBJ_NONE = 0;

/// The three number lines one device's context hands out: object ids, GPU
/// addresses and texture resource ids. Every method is safe to call from any
/// thread (Metal creates objects on the app's threads).
class RgfxIdSpace {
public:
	/// A source of chunks of texture resource ids: it hands `*first_id`, the
	/// first id of a chunk the process holds from now on (the accelerator
	/// kext's), and `*ids`, how many ids the chunk holds, or leaves `*ids` 0 for
	/// a chunk of the source's size (setResourceIdSource); or it answers false
	/// where none is left. Called with `context`. The grant's own size is the
	/// chunk's, so a process names only the ids the kext granted it whatever it
	/// took the device's features to be.
	typedef bool (*ResourceIdChunkFn)(void *context, uint64_t *first_id, uint64_t *ids);

	/// A space whose GPU addresses are the `address_span` bytes from
	/// `address_base`. Object ids start at 1 (0 is [`RGFX_OBJ_NONE`]). Until a
	/// chunk source is set, resource ids come from every chunk of the pool in
	/// order, as for the device's one process.
	explicit RgfxIdSpace(uint64_t address_base = kDefaultAddressBase,
	                     uint64_t address_span = RGFX_CONTEXT_VA_SPAN);

	/// The next object id, never 0 and never reused.
	RgfxObjId allocObjId();

	/// A free texture resource id of the process's chunks: the lowest free one of
	/// the first chunk taken that has one, else the first of a new chunk from the
	/// source. 0 where the source has none left: the caller makes no texture.
	uint64_t allocResourceId();

	/// Gives texture resource id `id` back to the process's chunks, once its
	/// texture is gone. 0, and an id of no chunk the process holds, are ignored.
	void freeResourceId(uint64_t id);

	/// Where new chunks of texture resource ids come from from now on, each of
	/// `chunk_ids` ids where the source does not say how many its chunk holds:
	/// RGFX_RESOURCE_ID_SMALL_CHUNK, or RGFX_RESOURCE_ID_CHUNK. A chunk whose
	/// first id is not a multiple of its size past 1, whose size is not whole
	/// small chunks of at most RGFX_RESOURCE_ID_CHUNK ids, or that holds an id
	/// the process holds or one past the pool, is no chunk.
	void setResourceIdSource(ResourceIdChunkFn source, void *context,
	                         uint64_t chunk_ids = RGFX_RESOURCE_ID_CHUNK);

	/// The GPU address of an object of `size` bytes the host pins there, aligned up
	/// to `align` (a power of two, at least 1) and to a kAddressGranule. Each pick
	/// holds Metal's span of a pin (addressSpan): the pages its bytes touch and one
	/// guard page after them, rounded up to a granule, so no two live picks' spans
	/// meet and each starts a granule, as the host's pins must (measured on an M3
	/// Max and an M4; metal2vulkan's runtime holds pins the same). Picks go forward
	/// through the context's range and wrap round to the ranges given back behind
	/// them. 0 where no free range is left.
	uint64_t allocAddress(uint64_t size, uint64_t align);

	/// Gives the range allocAddress or allocPrivateAddress picked at `address` back,
	/// once its object's destroy has gone to the host (posted, or held to go before
	/// anything the process posts after it). An address no live pick starts at, 0
	/// included, is ignored. A Private block left with no pick is given back to the
	/// source, past the one empty block the space keeps for its next picks.
	void freeAddress(uint64_t address);

	/// A source of Private blocks (the accelerator kext's PrivateTake and
	/// PrivateGive, RGFX_FEATURE_PRIVATE_BLOCKS): `take` hands a block of at least
	/// `size` bytes, its GPU address and its length, or answers false where the
	/// host has none; `give` hands one back, once no object lies in it. Called with
	/// `context`, and never with the space's lock held.
	typedef bool (*PrivateTakeFn)(void *context, uint64_t size, uint64_t *gpu_address,
	                              uint64_t *length);
	typedef void (*PrivateGiveFn)(void *context, uint64_t gpu_address, uint64_t length);

	/// Where Private blocks come from from now on. Called once, before any
	/// Private address is picked.
	void setPrivateBlockSource(PrivateTakeFn take, PrivateGiveFn give, void *context);

	/// Whether a Private block source is set.
	bool privateBlocks() const { return m_privateTake != nullptr; }

	/// The GPU address of a Private object of `size` bytes (a Private buffer, a heap
	/// that is not Shared, a Managed buffer's GPU copy), as allocAddress picks one,
	/// but in a Private block the context holds where a block source is set: next
	/// fit through the blocks held, in address order, then round from the first;
	/// where none has room, a new block of kPrivateBlockBytes, or of the pick's span
	/// and alignment where that is more. Where no source is set, or the source has
	/// no block, allocAddress's pick in the context's range (the refusal logged
	/// once). 0 where no address is left.
	uint64_t allocPrivateAddress(uint64_t size, uint64_t align);

	/// The bytes of addresses the space asks a Private block for at least: 256 MiB,
	/// metal2vulkan-rt's pooled block, so a pick wraps round its blocks only after
	/// that many bytes of spans (the host lets an object's pin go only once the
	/// command buffers that use it have finished).
	static constexpr uint64_t kPrivateBlockBytes = uint64_t{256} << 20;

	/// The Private blocks held, and the picks in them (for the tests).
	size_t privateBlocksHeld() const;
	size_t privatePicksLive() const;

	/// The bytes a pick of `size` bytes holds from its first byte, a granule's start:
	/// the pages it touches and one guard page, rounded up to a granule (1 page and
	/// the guard for 1 byte: 32 KiB; 64 pages: 66 pages).
	static uint64_t addressSpan(uint64_t size);

	/// Metal's page on Apple silicon, and the granule a pin's first page starts.
	static constexpr uint64_t kAddressPage = 0x4000;
	static constexpr uint64_t kAddressGranule = 0x8000;

	/// Moves the object ids and GPU addresses to the process's context's ranges,
	/// which the accelerator kext hands out (its object-id base, and its GPU
	/// addresses' base and span: RGFX_CONTEXT_VA_SPAN where the kext names
	/// none), so no two processes' numbers meet on the host. Called once, before
	/// any number is minted.
	void setBases(uint64_t obj_id_base, uint64_t address_base,
	              uint64_t address_span = RGFX_CONTEXT_VA_SPAN);

	/// Where this space's GPU addresses begin by default: the guest's first, where
	/// no range the host places itself lies (RGFX_CONTEXT_VA_BASE).
	static constexpr uint64_t kDefaultAddressBase = RGFX_CONTEXT_VA_BASE;

private:
	/// The context's address range, `span` bytes from `base`, every byte free. Caller holds
	/// m_addressLock (or is the constructor).
	void resetAddresses(uint64_t base, uint64_t span);

	std::atomic<uint64_t> m_nextObjId;

	// The GPU addresses, under m_addressLock: the free ranges by start (coalesced), the live
	// picks' spans by start, and where the next pick looks first.
	mutable std::mutex m_addressLock;
	std::map<uint64_t, uint64_t> m_freeRanges;
	std::map<uint64_t, uint64_t> m_livePicks;
	uint64_t m_addressCursor = 0;
	// Whether a pick found no room, which is logged once.
	bool m_addressesFull = false;

	// The Private blocks, under m_addressLock: each held block by address, its length and its
	// live picks; their free ranges by start, never joined across two blocks; the live picks'
	// spans by start; and where the next pick looks first.
	struct PrivateBlock {
		uint64_t length = 0;
		uint64_t picks = 0;
	};
	// The block `address` lies in, or end. Caller holds m_addressLock.
	std::map<uint64_t, PrivateBlock>::iterator privateBlockOf(uint64_t address);
	std::map<uint64_t, PrivateBlock> m_privateBlocks;
	std::map<uint64_t, uint64_t> m_privateFree;
	std::map<uint64_t, uint64_t> m_privatePicks;
	uint64_t m_privateCursor = 0;
	PrivateTakeFn m_privateTake = nullptr;
	PrivateGiveFn m_privateGive = nullptr;
	void *m_privateContext = nullptr;
	// Whether a take found no block, which is logged once.
	std::atomic<bool> m_privateRefused{false};

	// The texture resource ids, under m_idsLock, by small chunk (small chunk k
	// holds ids 1 + k * 16 to 16 + k * 16; a chunk of 64 is four): those the
	// process holds, in the order taken, and each one's free ids, bit i for its
	// id 1 + k * 16 + i.
	std::mutex m_idsLock;
	std::vector<uint32_t> m_chunks;
	uint16_t m_freeIds[RGFX_RESOURCE_ID_SMALL_CHUNKS] = {};
	bool m_heldChunks[RGFX_RESOURCE_ID_SMALL_CHUNKS] = {};
	ResourceIdChunkFn m_chunkSource = nullptr;
	void *m_chunkSourceContext = nullptr;
	// How many ids each chunk from the source holds, or the pool's, with none.
	uint64_t m_chunkIds = RGFX_RESOURCE_ID_CHUNK;
	// With no source: the next chunk of the pool.
	uint32_t m_nextLocalChunk = 0;
};

#endif // RGFX_OBJECT_MODEL_HPP_
