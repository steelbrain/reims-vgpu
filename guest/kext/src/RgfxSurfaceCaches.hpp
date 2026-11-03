// RgfxSurfaceCaches: the IOSurfaces whose shared blocks have a device cache on the host
// (RGFX_SURFACE_PAGES_CACHED, where the device has RGFX_FEATURE_SURFACE_CACHE), and what a CPU
// lock of one does past waiting for the GPU's pending writes (RgfxSurfaceLocks.hpp).
//
// Such a surface's GPU copy is the host's device cache, in its GPU's own memory, and its pages
// hold the CPU's. The kext keeps them apart as a GPU of its own memory does:
//   - A CPU lock, of either mode, of a surface the GPU wrote since the pages last took the cache
//     has the host copy the cache into the pages (SURFACE_DOWNLOAD) and waits for the copy, once
//     the writes are done. One lock asks at a time; the others wait for its answer.
//   - A read-write CPU lock moves the surface's CPU generation, the count of its read-write
//     locks, which the kext writes into the block's header for the processes: each command
//     buffer that uses the surface names the generation it saw at its commit
//     (RGFX_OP_SURFACE_USE), and the host fills the cache from the pages before the first buffer
//     of a newer one. IOSurface calls the kext at every read-write lock of a surface whose device
//     cache exists, current or not (RgfxIOSurfaceKPI.hpp), so the count starts with the block.
//   - The block's last holder letting go posts its BLOCK_RELEASE, which the host answers once
//     the pages hold the cache: a lock waits for the answer, then the surface is no longer
//     cached. The kext holds the surface's device cache until then, so its locks still reach
//     the kext, and the surface's memory wired, so the host's copy lands in pages that are
//     still the surface's, past the holders' own wiring, which ends at their let-go: never
//     pages the guest has purged, paged out or given to another; a block that is not cached
//     lets its cache go with it (rgfxCacheLetGo, rgfxCacheReleased).
//   - The block's last holder letting go leaves it dormant (RgfxSurfaceBlocks.hpp), and the
//     surface cached, its memory wired and its device cache held: its locks go on as before, a
//     download among them, and its generation moves on. Its BLOCK_RELEASE, once the dormant
//     block is let go, has the host copy a cache the GPU wrote into the pages, and a lock waits
//     for the answer; then the surface is no longer cached.
//   - A dormant surface's device cache purged (op 5, as each client of the surface goes), or
//     the surface made empty (op 7), marks it to be let go (rgfxCachePurged,
//     rgfxCachePurgeable), and the kext's release thread lets its block go, the pages taking
//     what the GPU wrote; a surface made empty while held does not sleep at its last let-go.
//     A surface made volatile keeps its dormant block, the first of them to go under the caps
//     (RgfxSurfaceBlocks.hpp), as IOSurfaceRef.h lets a volatile surface's contents be
//     discarded; made nonvolatile again (op 6, or op 7), it is neither.
// A wait that does not end in kWaitMs goes on, as a lock waiting for a write does.
//
// A surface is named by its key, the device cache the kext made on it, as in RgfxSurfaceLocks.
// The caller holds one lock around every call. Header-only and free of the kernel's headers, so
// its native test (tests/surface_caches_test.cpp) builds it as the kext does.
//
// Our own code.
#ifndef RGFX_SURFACE_CACHES_HPP_
#define RGFX_SURFACE_CACHES_HPP_

#include <stdint.h>

#include "RgfxSurfaceBlocks.hpp"

// One cached surface.
struct RgfxCachedSurface {
	uint64_t key;
	// Its shared block's region offset, and its IOSurfaceID.
	uint64_t offset;
	uint32_t surfaceId;
	// The count of its read-write CPU locks: its CPU generation.
	uint64_t cpuGen;
	// How many GPU writes were named of it, and how many of them the pages hold: as many as
	// `writes` had when the last download was asked and answered.
	uint64_t writes;
	uint64_t copied;
	// A lock's download is asked and not yet answered.
	bool downloading;
	// The block's BLOCK_RELEASE is posted and not yet answered.
	bool releasing;
	// The block is dormant (no client holds it).
	bool dormant;
	// The dormant block is to be let go, and why (a RgfxDormantCause); kRgfxCacheKept for none.
	uint32_t letGo;
	// The surface was made volatile, or empty, and not nonvolatile since.
	bool purgeable;
	bool emptied;
	// Where the kext writes `cpuGen` for the processes (the block's header, mapped), the
	// mapping, the surface's memory, held wired while the block lives, and the surface's
	// reference, which the table holds from the block's let-go until its release is answered,
	// opaque here.
	volatile uint64_t *genAt;
	void *map;
	void *wire;
	void *surface;
	// Where the kext writes the surface's clients for the processes (the header's
	// kRgfxSurfaceClientsOffset), and the surface, whose reference the block's entry holds, or
	// this table's `surface` after the let-go, while this entry lives: opaque here.
	volatile uint32_t *clientsAt;
	const void *object;
};

// RgfxCachedSurface::letGo where nothing asks the dormant block to go.
constexpr uint32_t kRgfxCacheKept = ~0u;

struct RgfxCacheTable {
	RgfxCachedSurface *entries;
	uint32_t count;
	uint32_t capacity;
};

// The cached surface of key `key`, or null.
inline RgfxCachedSurface *rgfxCacheFind(RgfxCacheTable &t, uint64_t key)
{
	for (uint32_t i = 0; i < t.count; i++) {
		if (t.entries[i].key == key)
			return &t.entries[i];
	}
	return nullptr;
}

// The cached surface whose block is at `offset`, or null.
inline RgfxCachedSurface *rgfxCacheAtBlock(RgfxCacheTable &t, uint64_t offset)
{
	for (uint32_t i = 0; i < t.count; i++) {
		if (t.entries[i].offset == offset)
			return &t.entries[i];
	}
	return nullptr;
}

// The cached surface of IOSurfaceID `surfaceId` whose block is not going, or null.
inline RgfxCachedSurface *rgfxCacheOfSurface(RgfxCacheTable &t, uint32_t surfaceId)
{
	for (uint32_t i = 0; i < t.count; i++) {
		const RgfxCachedSurface &c = t.entries[i];
		if (c.surfaceId == surfaceId && !c.releasing)
			return &t.entries[i];
	}
	return nullptr;
}

// Adds the surface `key` (IOSurfaceID `surfaceId`), whose block at `offset` is cached, at CPU
// generation 0 and no write. Null where the table has no room or the key is there already.
inline RgfxCachedSurface *rgfxCacheAdd(RgfxCacheTable &t, uint64_t key, uint64_t offset,
                                       uint32_t surfaceId)
{
	if (t.count == t.capacity || rgfxCacheFind(t, key) != nullptr)
		return nullptr;
	RgfxCachedSurface &c = t.entries[t.count++];
	c = RgfxCachedSurface{};
	c.key = key;
	c.offset = offset;
	c.surfaceId = surfaceId;
	c.letGo = kRgfxCacheKept;
	return &c;
}

// Takes `c` out of the table: its mapping and its wired memory, which the caller lets go, in
// `*map` and `*wire`.
inline void rgfxCacheRemove(RgfxCacheTable &t, RgfxCachedSurface *c, void **map, void **wire)
{
	*map = c->map;
	*wire = c->wire;
	*c = t.entries[--t.count];
}

// The block of surface `key` goes: a cached one stays, its locks waiting, until the block's
// release is answered where `releasing`, else it is taken out (its mapping and wired memory in
// `*map` and `*wire`, for the caller to let go). True where the caller releases the device
// cache's reference and the surface's, `surface`, now; false where the table keeps both for the
// answer (rgfxCacheReleased).
inline bool rgfxCacheLetGo(RgfxCacheTable &t, uint64_t key, bool releasing, void *surface,
                           void **map, void **wire)
{
	*map = *wire = nullptr;
	RgfxCachedSurface *c = rgfxCacheFind(t, key);
	if (c == nullptr)
		return true;
	if (releasing) {
		c->releasing = true;
		c->surface = surface;
		return false;
	}
	rgfxCacheRemove(t, c, map, wire);
	return true;
}

// The surface of the cached block at `offset` whose BLOCK_RELEASE is posted and not yet
// answered, the reference the table holds until then (opaque here), and whether a buffer wrote
// its cache since its last copy (`*written`); null where no cached surface is releasing it.
inline void *rgfxCacheReleasingSurface(RgfxCacheTable &t, uint64_t offset, bool *written)
{
	RgfxCachedSurface *c = rgfxCacheAtBlock(t, offset);
	*written = c != nullptr && c->releasing && c->writes != c->copied;
	return c != nullptr && c->releasing ? c->surface : nullptr;
}

// The flags of a cached block's BLOCK_RELEASE, its surface having `clients` clients left
// (IOSurface's count, read as the release is posted): none, RGFX_BLOCK_RELEASE_NO_CLIENT, which
// the host only counts, copying the cache into the pages all the same (a mach port or an XPC
// object holds a surface with no client, and whoever looks it up from one reads its pages).
inline uint32_t rgfxCacheReleaseFlags(int32_t clients)
{
	return clients == 0 ? RGFX_BLOCK_RELEASE_NO_CLIENT : 0;
}

// The BLOCK_RELEASE of the block at `offset` was answered: the cached surface releasing it is
// taken out (its mapping, wired memory and surface's reference in `*map`, `*wire` and
// `*surface`), and its key, the device cache whose reference the caller releases now, before
// the surface's, returned; 0 where none was releasing it.
inline uint64_t rgfxCacheReleased(RgfxCacheTable &t, uint64_t offset, void **map, void **wire,
                                  void **surface)
{
	*map = *wire = *surface = nullptr;
	RgfxCachedSurface *c = rgfxCacheAtBlock(t, offset);
	if (c == nullptr || !c->releasing)
		return 0;
	const uint64_t key = c->key;
	*surface = c->surface;
	rgfxCacheRemove(t, c, map, wire);
	return key;
}

// The block of surface `key` is dormant (`dormant`), or held again: false where the surface is
// not cached, made empty (it does not sleep), or is to be let go (it is held again by no ask).
// `*purgeable` where the surface is volatile.
inline bool rgfxCacheSleep(RgfxCacheTable &t, uint64_t key, bool dormant, bool *purgeable)
{
	*purgeable = false;
	RgfxCachedSurface *c = rgfxCacheFind(t, key);
	if (c == nullptr || (dormant && c->emptied) || (!dormant && c->letGo != kRgfxCacheKept))
		return false;
	c->dormant = dormant;
	*purgeable = c->purgeable;
	return true;
}

// The new states op 7 carries, as IOMemoryDescriptor numbers them (kIOMemoryPurgeable*):
// nonvolatile, and the two that may lose a surface's bytes, volatile and empty.
constexpr uint64_t kRgfxPurgeableNonVolatile = 2;
constexpr uint64_t kRgfxPurgeableVolatile = 3;
constexpr uint64_t kRgfxPurgeableEmpty = 4;

// Surface `key` has `clients` processes holding it now, as a block of it is cached or as one
// goes (op 5): written into its block's header for the processes, which keep a texture's view of
// a surface another process holds (the bundle's keep). A count below 0 is written as 0, and one
// past 32 bits as their most. True where the surface is cached and the count was written.
inline bool rgfxCacheNoteClients(RgfxCacheTable &t, uint64_t key, int64_t clients)
{
	RgfxCachedSurface *c = rgfxCacheFind(t, key);
	if (c == nullptr || c->clientsAt == nullptr)
		return false;
	const uint32_t n = clients < 0              ? 0u
	                   : clients > 0xffffffffll ? 0xffffffffu
	                                            : static_cast<uint32_t>(clients);
	__atomic_store_n(c->clientsAt, n, __ATOMIC_RELEASE);
	return true;
}

// Surface `key`'s device cache was purged (op 5): a dormant block of it is to be let go. True
// where one is, not asked to go before.
inline bool rgfxCachePurged(RgfxCacheTable &t, uint64_t key)
{
	RgfxCachedSurface *c = rgfxCacheFind(t, key);
	if (c == nullptr || !c->dormant || c->letGo != kRgfxCacheKept)
		return false;
	c->letGo = kRgfxDormantPurged;
	return true;
}

// Surface `key` was made `state` (op 7): volatile marks it, its dormant block the first to go
// under the caps; empty marks it, its dormant block to be let go now, a held one not to sleep;
// nonvolatile ends both; reading the state changes nothing. True where a dormant block of it is
// to be looked at again, on the release thread.
inline bool rgfxCachePurgeable(RgfxCacheTable &t, uint64_t key, uint64_t state)
{
	RgfxCachedSurface *c = rgfxCacheFind(t, key);
	if (c == nullptr)
		return false;
	if (state == kRgfxPurgeableNonVolatile) {
		c->purgeable = c->emptied = false;
	} else if (state == kRgfxPurgeableVolatile) {
		c->purgeable = true;
	} else if (state == kRgfxPurgeableEmpty) {
		c->emptied = true;
		if (c->dormant && c->letGo == kRgfxCacheKept)
			c->letGo = kRgfxDormantEmptied;
	} else {
		return false;
	}
	return c->dormant;
}

// The dormant block at `offset` of a cached surface is to be let go: its surface had no client
// left at its sleep. True where it is, not asked to go before.
inline bool rgfxCacheUnclaimed(RgfxCacheTable &t, uint64_t offset)
{
	RgfxCachedSurface *c = rgfxCacheAtBlock(t, offset);
	if (c == nullptr || !c->dormant || c->releasing || c->letGo != kRgfxCacheKept)
		return false;
	c->letGo = kRgfxDormantNoClients;
	return true;
}

// Surface `key` was made nonvolatile by op 6 (IOSurfaceDeviceCache::unpurge): as op 7 with
// nonvolatile.
inline bool rgfxCacheUnpurged(RgfxCacheTable &t, uint64_t key)
{
	return rgfxCachePurgeable(t, key, kRgfxPurgeableNonVolatile);
}

// The block at `offset` of a cached surface, where it is dormant and asked to go: why, else
// kRgfxCacheKept. `*purgeable` where its surface is volatile.
inline uint32_t rgfxCacheLetGoOf(RgfxCacheTable &t, uint64_t offset, bool *purgeable)
{
	RgfxCachedSurface *c = rgfxCacheAtBlock(t, offset);
	*purgeable = c != nullptr && c->purgeable;
	return c != nullptr && c->dormant && !c->releasing ? c->letGo : kRgfxCacheKept;
}

// A GPU write of surface `key` was named. False where it is not cached.
inline bool rgfxCacheNoteWrite(RgfxCacheTable &t, uint64_t key)
{
	RgfxCachedSurface *c = rgfxCacheFind(t, key);
	if (c == nullptr)
		return false;
	c->writes++;
	return true;
}

// What a CPU lock of surface `key` does next, once no GPU write of it is pending.
enum RgfxCacheStep {
	// Nothing more: the pages hold every write, or the surface is not cached.
	kRgfxCacheGoOn,
	// Waits: another lock's download, or the block's release, is not yet answered.
	kRgfxCacheWait,
	// Asks the download of the block at `*offset` and waits for its answer, then calls
	// rgfxCacheDownloaded with `*mark`.
	kRgfxCacheDownload,
};

inline RgfxCacheStep rgfxCacheNext(RgfxCacheTable &t, uint64_t key, uint64_t *offset,
                                   uint64_t *mark)
{
	RgfxCachedSurface *c = rgfxCacheFind(t, key);
	if (c == nullptr)
		return kRgfxCacheGoOn;
	if (c->downloading || c->releasing)
		return kRgfxCacheWait;
	if (c->writes == c->copied)
		return kRgfxCacheGoOn;
	c->downloading = true;
	*offset = c->offset;
	*mark = c->writes;
	return kRgfxCacheDownload;
}

// The download asked at `mark` was answered, OK where `ok`: the pages hold the writes named up to
// it. Writes named since are still to copy. A download not answered OK copies nothing.
inline void rgfxCacheDownloaded(RgfxCacheTable &t, uint64_t key, uint64_t mark, bool ok)
{
	RgfxCachedSurface *c = rgfxCacheFind(t, key);
	if (c == nullptr)
		return;
	c->downloading = false;
	if (ok && mark > c->copied)
		c->copied = mark;
}

// A read-write CPU lock of surface `key` goes on: its CPU generation moves. The new generation,
// or 0 where the surface is not cached.
inline uint64_t rgfxCacheReadWriteLock(RgfxCacheTable &t, uint64_t key)
{
	RgfxCachedSurface *c = rgfxCacheFind(t, key);
	if (c == nullptr)
		return 0;
	return ++c->cpuGen;
}

// The BLOCK_RELEASE of the cached block at `offset` is to be posted: locks of its surface wait
// for its answer. False where no cached surface has that block.
inline bool rgfxCacheReleasing(RgfxCacheTable &t, uint64_t offset)
{
	RgfxCachedSurface *c = rgfxCacheAtBlock(t, offset);
	if (c == nullptr)
		return false;
	c->releasing = true;
	return true;
}

#endif // RGFX_SURFACE_CACHES_HPP_
