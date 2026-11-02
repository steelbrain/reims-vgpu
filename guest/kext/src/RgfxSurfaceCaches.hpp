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
	// Where the kext writes `cpuGen` for the processes (the block's header, mapped), the
	// mapping, and the surface's memory, which the kext holds wired from the block's caching
	// until it is cached no more, opaque here.
	volatile uint64_t *genAt;
	void *map;
	void *wire;
};

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
		if (t.entries[i].surfaceId == surfaceId && !t.entries[i].releasing)
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
	return &c;
}

// Takes `c` out of the table: its mapping and its surface's wired memory, which the caller lets
// go, in `*map` and `*wire`.
inline void rgfxCacheRemove(RgfxCacheTable &t, RgfxCachedSurface *c, void **map, void **wire)
{
	*map = c->map;
	*wire = c->wire;
	*c = t.entries[--t.count];
}

// The block of surface `key` goes: a cached one stays, its locks waiting and its memory wired,
// until the block's release is answered where `releasing`, else it is taken out (its mapping
// and wired memory in `*map` and `*wire`, for the caller to let go). True where the caller
// releases the device cache's reference now; false where the table keeps it, and the wiring,
// for the answer (rgfxCacheReleased).
inline bool rgfxCacheLetGo(RgfxCacheTable &t, uint64_t key, bool releasing, void **map,
                           void **wire)
{
	*map = *wire = nullptr;
	RgfxCachedSurface *c = rgfxCacheFind(t, key);
	if (c == nullptr)
		return true;
	if (releasing) {
		c->releasing = true;
		return false;
	}
	rgfxCacheRemove(t, c, map, wire);
	return true;
}

// The BLOCK_RELEASE of the block at `offset` was answered: the cached surface releasing it is
// taken out (its mapping and wired memory in `*map` and `*wire`, which the caller unwires only
// now), and its key, the device cache whose reference the caller releases now, returned; 0
// where none was releasing it.
inline uint64_t rgfxCacheReleased(RgfxCacheTable &t, uint64_t offset, void **map, void **wire)
{
	*map = *wire = nullptr;
	RgfxCachedSurface *c = rgfxCacheAtBlock(t, offset);
	if (c == nullptr || !c->releasing)
		return 0;
	const uint64_t key = c->key;
	rgfxCacheRemove(t, c, map, wire);
	return key;
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
