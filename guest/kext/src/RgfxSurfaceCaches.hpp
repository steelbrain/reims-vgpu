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
//   - A deferred block (RgfxSurfaceBlocks.hpp) has no pages until a lock needs them: the kext
//     holds its surface's memory retained, not wired. Its ask reads the surface's seed and its
//     resident pages once its device cache and control function are made (kRgfxCacheGating),
//     and a lock goes on then as for any block: its seed moves, and the ask names the pages.
//     Found fresh, a lock waits for the host's answer to the deferral (kRgfxCacheDeferring).
//     Deferred, a read-write lock, or a read-only one of a surface the GPU wrote, attaches the
//     pages first (RGFX_SURFACE_PAGES_ATTACH), one lock at a time, the others waiting; then it
//     goes on as for any block, a download among it. A naming not answered OK is made again,
//     backing off, until the lock's wait ends (rgfxAttachRetryMs). A read-only lock of a
//     surface the GPU never wrote attaches nothing: a fresh page reads as zeros, as the cache
//     does. A dormant one is kept at a client's going (op 5) while other clients are left; its
//     release, where the GPU wrote it and the surface lives past the kext's own references,
//     attaches its pages first, so the release copies the cache into them
//     (rgfxCacheReleaseNext), naming them again as a lock does; else nothing reads them.
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

// Where a cached surface's block stands with its pages (RgfxSurfaceBlocks.hpp's deferred
// blocks).
enum RgfxCacheDeferral : uint32_t {
	// Its pages were named at its ask, as every block's not deferred.
	kRgfxCacheNotDeferred = 0,
	// Its ask reads the surface's seed and pages, to defer it or name them: a lock goes on as
	// for a block not deferred.
	kRgfxCacheGating,
	// The surface was fresh, and the host has not answered the deferral: a lock waits.
	kRgfxCacheDeferring,
	// Deferred: no pages named yet.
	kRgfxCacheDeferred,
	// A lock, or the release, is naming its pages: the others wait.
	kRgfxCacheAttaching,
	// Its pages were named after its ask: as a block not deferred.
	kRgfxCacheAttached,
	// Its attach was not answered in time: the host may yet take its pages, so they stay wired,
	// as an attached block's, until its release is answered OK, and a lock goes on as for one
	// attached, never naming them again.
	kRgfxCacheAttachLate,
};

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
	// Its pages (RgfxCacheDeferral): `wire` holds the surface's memory retained but not prepared
	// while the block is deferred (rgfxCacheWired), and an attach prepares it. The block's
	// context, the bytes its holders map and the plane's bytes: what an attach's SURFACE_PAGES
	// names.
	RgfxCacheDeferral deferral;
	uint32_t context;
	uint64_t size;
	uint64_t length;
	// The block went, not released, while a lock or the release named its pages: the attach
	// lets the surface go once answered (rgfxCacheAttached), so the let-go waits for no host.
	bool forgotten;
	// That block, kept for good, took no wire of the pages of its own (its let-go read them
	// unnamed, before the attach began): the attach keeps its wiring of them for good where it
	// named them, as the host may map them through a block that never goes.
	bool keepWire;
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
		if (c.surfaceId == surfaceId && !c.releasing && !c.forgotten)
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

// Whether `c`'s `wire` is prepared: every block's but a deferred one's before its pages are
// named, whose memory is only retained, and which the caller then releases unprepared.
inline bool rgfxCacheWired(const RgfxCachedSurface &c)
{
	return c.deferral == kRgfxCacheNotDeferred || c.deferral == kRgfxCacheAttached ||
	       c.deferral == kRgfxCacheAttachLate;
}

// Whether the host may map a deferred block's pages: named, being named, or asked and not
// answered. Its holders wire none, so its block takes a wire of them as it goes, which it keeps
// until its release is answered OK.
inline bool rgfxCacheHostPages(const RgfxCachedSurface &c)
{
	return c.deferral == kRgfxCacheAttaching || c.deferral == kRgfxCacheAttached ||
	       c.deferral == kRgfxCacheAttachLate;
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
// answer (rgfxCacheReleased), or, where an attach of a deferred block's pages is in flight, for
// that attach to let go once answered (rgfxCacheAttached): the let-go never waits for the host.
// `blockWired`: the block keeps a wire of the surface's pages of its own while it is in the
// table. Decided here, in the same hold as the attach's state: a block kept for good without
// one, whose attach is in flight, has the attach keep the pages wired (`keepWire`).
inline bool rgfxCacheLetGo(RgfxCacheTable &t, uint64_t key, bool releasing, void *surface,
                           void **map, void **wire, bool blockWired)
{
	*map = *wire = nullptr;
	RgfxCachedSurface *c = rgfxCacheFind(t, key);
	if (c == nullptr)
		return true;
	if (releasing || c->deferral == kRgfxCacheAttaching) {
		c->releasing = releasing;
		c->forgotten = !releasing;
		c->keepWire = !releasing && !blockWired;
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

// Surface `key`'s device cache was purged (op 5), `clients` of the surface left: a dormant block
// of it is to be let go, but a deferred one whose pages are not named while a client is left,
// whose cache alone holds its bytes. True where one is, not asked to go before.
inline bool rgfxCachePurged(RgfxCacheTable &t, uint64_t key, int64_t clients = 0)
{
	RgfxCachedSurface *c = rgfxCacheFind(t, key);
	if (c == nullptr || !c->dormant || c->letGo != kRgfxCacheKept)
		return false;
	if (c->deferral == kRgfxCacheDeferred && clients > 0)
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
	// Names the deferred block's pages (RGFX_SURFACE_PAGES_ATTACH) and waits for the answer, then
	// calls rgfxCacheAttached.
	kRgfxCacheAttach,
};

// A lock, read-write where `readWrite`.
inline RgfxCacheStep rgfxCacheNext(RgfxCacheTable &t, uint64_t key, uint64_t *offset,
                                   uint64_t *mark, bool readWrite = false)
{
	RgfxCachedSurface *c = rgfxCacheFind(t, key);
	if (c == nullptr)
		return kRgfxCacheGoOn;
	if (c->downloading || c->releasing || c->deferral == kRgfxCacheDeferring ||
	    c->deferral == kRgfxCacheAttaching)
		return kRgfxCacheWait;
	if (c->deferral == kRgfxCacheDeferred) {
		if (!readWrite && c->writes == c->copied)
			return kRgfxCacheGoOn;
		c->deferral = kRgfxCacheAttaching;
		*offset = c->offset;
		return kRgfxCacheAttach;
	}
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

// How long a lock, or the release, waits to name a deferred block's pages again after a naming
// not answered OK (kRgfxPagesNotNamed: the surface's memory not wired, its runs not taken or not
// written, or the host refusing them), `tries` namings made so far: from kRgfxAttachRetryFirstMs,
// doubling to kRgfxAttachRetryMostMs. A naming that fails for a moment (the guest short of wired
// memory, the host's views full) is made again within the lock's wait, and one that keeps failing
// costs a dozen namings in its 2 s. The lock goes on over pages that do not hold the cache's bytes
// only at its deadline, as each wait of a lock does.
constexpr uint32_t kRgfxAttachRetryFirstMs = 10;
constexpr uint32_t kRgfxAttachRetryMostMs = 320;

inline uint32_t rgfxAttachRetryMs(uint32_t tries)
{
	uint32_t ms = kRgfxAttachRetryFirstMs;
	for (uint32_t i = 1; i < tries && ms < kRgfxAttachRetryMostMs; i++)
		ms *= 2;
	return ms < kRgfxAttachRetryMostMs ? ms : kRgfxAttachRetryMostMs;
}

// The attach a lock or the release asked of surface `key` (kRgfxCacheAttach) ended as `naming`:
// its pages are the cache's from here on, or are still to name, or, not answered, may yet be the
// host's (kRgfxCacheAttachLate). Where its block went meanwhile (`forgotten`), the surface is
// taken out (its mapping, its memory, wired where rgfxPagesNamingWired, and the surface's
// reference in `*map`, `*wire` and `*surface`) and its key returned, the device cache whose
// reference the caller releases, then the surface's; else 0. A block that took a wire of its own
// as it went (rgfxCacheHostPages) holds the host's pages past the cache's; one kept for good that
// took none (`keepWire`) has the caller keep the memory's wiring for good where the naming wired
// it (`*keepWired`): neither unwired nor released, as the host may map the pages for good.
inline uint64_t rgfxCacheAttached(RgfxCacheTable &t, uint64_t key, RgfxPagesNaming naming,
                                  void **map, void **wire, void **surface,
                                  bool *keepWired = nullptr)
{
	*map = *wire = *surface = nullptr;
	if (keepWired)
		*keepWired = false;
	RgfxCachedSurface *c = rgfxCacheFind(t, key);
	if (c == nullptr || c->deferral != kRgfxCacheAttaching)
		return 0;
	c->deferral = naming == kRgfxPagesNamed       ? kRgfxCacheAttached
	              : naming == kRgfxPagesUnanswered ? kRgfxCacheAttachLate
	                                               : kRgfxCacheDeferred;
	if (!c->forgotten)
		return 0;
	if (keepWired)
		*keepWired = c->keepWire && rgfxPagesNamingWired(naming);
	*surface = c->surface;
	rgfxCacheRemove(t, c, map, wire);
	return key;
}

// The ask of surface `key`'s deferred block read the surface fresh: it asks the host to defer
// the block, and locks wait for the answer. False where the surface is not gating.
inline bool rgfxCacheGated(RgfxCacheTable &t, uint64_t key)
{
	RgfxCachedSurface *c = rgfxCacheFind(t, key);
	if (c == nullptr || c->deferral != kRgfxCacheGating)
		return false;
	c->deferral = kRgfxCacheDeferring;
	return true;
}

// The host answered the ask of surface `key`'s block: deferred where `deferred`, else its pages
// were named at the ask (its surface not fresh, or the deferral refused), `wire` prepared.
inline void rgfxCacheSettled(RgfxCacheTable &t, uint64_t key, bool deferred)
{
	RgfxCachedSurface *c = rgfxCacheFind(t, key);
	if (c != nullptr && (c->deferral == kRgfxCacheGating || c->deferral == kRgfxCacheDeferring))
		c->deferral = deferred ? kRgfxCacheDeferred : kRgfxCacheNotDeferred;
}

// Whether the asker's hold of a block whose ask named its pages after all (its surface not fresh,
// the deferral refused, or no device cache: deferSurfaceBlock), `naming` the outcome, wires
// them again for itself, the block cached where `cached`. The naming's own wiring
// (rgfxPagesNamingWired) is its cache's where named, which the settle leaves wired
// (rgfxCacheSettled, rgfxCacheWired), so the hold takes a second; else the hold takes the
// naming's, which nothing else would let go.
inline bool rgfxAskHoldWiresAgain(bool cached, RgfxPagesNaming naming)
{
	return cached && naming == kRgfxPagesNamed;
}

// The references of a surface the kext holds itself as its release thread looks at a deferred
// block's release: the table's, which the block's entry handed it at the let-go
// (rgfxCacheLetGo); the device cache's, the kext holding the cache, which IOSurface's
// IOSurfaceDeviceCache::init retains its surface for (IOSurface 393.5.8); and the release
// thread's own, taken to read the surface (rgfxCacheReleasingSurface). A surface retained past
// these lives on: a process's client, a mach port or an XPC object holds it, and may read it.
constexpr int32_t kRgfxSurfaceTableRef = 1;
constexpr int32_t kRgfxSurfaceDeviceCacheRef = 1;
constexpr int32_t kRgfxSurfaceReleaseRef = 1;
constexpr int32_t kRgfxSurfaceKextRefs =
    kRgfxSurfaceTableRef + kRgfxSurfaceDeviceCacheRef + kRgfxSurfaceReleaseRef;

// What the release of the cached block at `offset` does before its BLOCK_RELEASE goes, its
// surface having `clients` clients and `retains` references (IOSurface's counts, read out of
// the lock).
enum RgfxCacheReleaseStep {
	// Goes as it is: a block not deferred, or attached, whose release copies the cache into its
	// pages; or a deferred one the GPU never wrote, or that nothing but the kext holds, whose
	// pages nothing reads.
	kRgfxCacheReleaseAsIs,
	// Waits: a lock is naming its pages.
	kRgfxCacheReleaseWait,
	// Names its pages (kRgfxCacheAttach), so the release copies what the GPU wrote into them.
	kRgfxCacheReleaseAttach,
};

// What rgfxCacheReleaseNext would answer, with nothing changed: for the release thread, which
// hands a release that waits or names to the naming worker, and goes on with the next.
inline RgfxCacheReleaseStep rgfxCacheReleasePeek(RgfxCacheTable &t, uint64_t offset,
                                                 int32_t clients, int32_t retains)
{
	const RgfxCachedSurface *c = rgfxCacheAtBlock(t, offset);
	if (c == nullptr || !c->releasing)
		return kRgfxCacheReleaseAsIs;
	if (c->deferral == kRgfxCacheAttaching)
		return kRgfxCacheReleaseWait;
	if (c->deferral != kRgfxCacheDeferred || c->writes == c->copied ||
	    (clients <= 0 && retains <= kRgfxSurfaceKextRefs))
		return kRgfxCacheReleaseAsIs;
	return kRgfxCacheReleaseAttach;
}

inline RgfxCacheReleaseStep rgfxCacheReleaseNext(RgfxCacheTable &t, uint64_t offset,
                                                 int32_t clients, int32_t retains)
{
	const RgfxCacheReleaseStep step = rgfxCacheReleasePeek(t, offset, clients, retains);
	if (step == kRgfxCacheReleaseAttach)
		rgfxCacheAtBlock(t, offset)->deferral = kRgfxCacheAttaching;
	return step;
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

// The let-go of the block at `offset`, which named surface `key`'s device cache: as
// rgfxCacheLetGo, where the surface's cached entry is this block's own. Another block of the
// surface, a copy or one in place uncached, whose locks follow the surface by the same device
// cache, holds no entry, and leaves the cached block's as it is: released and unanswered, its
// memory stays wired for the host's copy into the pages; living, or with an attach in flight,
// its locks go on. True, nothing in `*map` and `*wire`, where the caller releases its own
// references of the cache and the surface now.
inline bool rgfxCacheBlockLetGo(RgfxCacheTable &t, uint64_t key, uint64_t offset, bool releasing,
                                void *surface, void **map, void **wire, bool blockWired)
{
	*map = *wire = nullptr;
	const RgfxCachedSurface *c = rgfxCacheFind(t, key);
	if (c != nullptr && c->offset != offset)
		return true;
	return rgfxCacheLetGo(t, key, releasing, surface, map, wire, blockWired);
}

#endif // RGFX_SURFACE_CACHES_HPP_
