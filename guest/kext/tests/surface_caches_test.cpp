// The native test of RgfxSurfaceCaches.hpp: a lock of a cached surface the GPU wrote asks one
// download and goes on once it is answered, other locks waiting for it meanwhile; a write named
// during a download is copied by the next; a read-write lock moves the CPU generation; a lock
// waits for the block's release; and a surface that is not cached does nothing past the writes.
// A block's let-go hands back the device cache's reference at once, but a cached block's being
// released, which the table holds until the answer.
// A dormant block keeps its generation and its downloads, and is let go once its device cache
// goes, never named by that key again, or once its surface turns purgeable.
// A cached surface's clients are written where its header holds them, clamped to 32 bits, and
// nothing is written for a surface not cached or one with no header.
#include "RgfxSurfaceCaches.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "surface_caches_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

constexpr uint64_t kSurface = 0x1000, kOther = 0x2000, kNotCached = 0x3000;
constexpr uint64_t kBlock = 0x40000, kOtherBlock = 0x80000;
constexpr uint32_t kId = 42, kOtherId = 43;

void dormancy()
{
	RgfxCachedSurface entries[3] = {};
	RgfxCacheTable t = {entries, 0, 3};
	uint64_t offset = 0, mark = 0;
	RgfxCachedSurface *c = rgfxCacheAdd(t, kSurface, kBlock, kId);
	check(c != nullptr && c->letGo == kRgfxCacheKept, "a cached surface, asked to go nowhere");
	check(rgfxCacheAdd(t, kOther, kOtherBlock, kOtherId) != nullptr, "and another");
	check(rgfxCacheReadWriteLock(t, kSurface) == 1, "generation 1 while held");
	check(rgfxCacheNoteWrite(t, kSurface), "the GPU writes it");

	// Dormant: its locks download and move its generation as before.
	bool marked = true;
	check(rgfxCacheSleep(t, kSurface, true, &marked) && c->dormant && !marked,
	      "its block is dormant");
	check(!rgfxCacheSleep(t, kNotCached, true, &marked),
	      "a surface not cached has no dormant block");
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheDownload && offset == kBlock,
	      "a lock of a dormant surface the GPU wrote asks the download");
	rgfxCacheDownloaded(t, kSurface, mark, true);
	check(rgfxCacheReadWriteLock(t, kSurface) == 2, "and its generation moves on");
	check(rgfxCacheLetGoOf(t, kBlock, &marked) == kRgfxCacheKept, "nothing asks it to go");

	// Purgeable (op 7's states as IOMemoryDescriptor numbers them): volatile marks a dormant
	// block the first to go, and keeps it; nonvolatile, by op 7 or op 6, ends the mark; reading
	// the state changes nothing; empty lets a dormant block go.
	bool purgeable = true;
	check(!rgfxCachePurgeable(t, kSurface, 1) && rgfxCacheLetGoOf(t, kBlock, &purgeable) ==
	                                                  kRgfxCacheKept && !purgeable,
	      "reading the state keeps it, not volatile");
	check(rgfxCachePurgeable(t, kSurface, kRgfxPurgeableVolatile) &&
	          rgfxCacheLetGoOf(t, kBlock, &purgeable) == kRgfxCacheKept && purgeable,
	      "volatile keeps the dormant block, marked to go first");
	check(rgfxCachePurgeable(t, kSurface, kRgfxPurgeableNonVolatile) &&
	          rgfxCacheLetGoOf(t, kBlock, &purgeable) == kRgfxCacheKept && !purgeable,
	      "nonvolatile again ends the mark");
	check(rgfxCachePurgeable(t, kSurface, kRgfxPurgeableVolatile) && rgfxCacheUnpurged(t, kSurface) &&
	          rgfxCacheLetGoOf(t, kBlock, &purgeable) == kRgfxCacheKept && !purgeable,
	      "and so does op 6");
	check(!rgfxCachePurgeable(t, kOther, kRgfxPurgeableVolatile) &&
	          rgfxCacheFind(t, kOther)->purgeable,
	      "a held block is marked, with nothing to look at");
	check(rgfxCachePurgeable(t, kSurface, kRgfxPurgeableEmpty) &&
	          rgfxCacheLetGoOf(t, kBlock, &purgeable) == kRgfxDormantEmptied,
	      "empty lets the dormant block go");
	check(rgfxCachePurgeable(t, kSurface, kRgfxPurgeableEmpty) && !rgfxCachePurged(t, kSurface) &&
	          rgfxCacheLetGoOf(t, kBlock, &purgeable) == kRgfxDormantEmptied,
	      "it is asked once, for its first cause");
	check(!rgfxCacheSleep(t, kSurface, false, &purgeable), "one asked to go is held again by no ask");

	// Revival keeps the generation, and the volatile mark.
	check(rgfxCacheSleep(t, kOther, true, &purgeable) && purgeable &&
	          rgfxCacheSleep(t, kOther, false, &purgeable),
	      "the other, volatile, sleeps marked and is held again");
	check(rgfxCachePurgeable(t, kOther, kRgfxPurgeableNonVolatile) == false &&
	          !rgfxCacheFind(t, kOther)->purgeable,
	      "made nonvolatile while held");
	check(rgfxCacheReadWriteLock(t, kOther) == 1, "its generation counts on");

	// Op 5: the cache was purged, as a client of the surface went. A dormant block of it is let
	// go, a held one's not, and its cache, which the block holds, names it until its release is
	// answered.
	check(!rgfxCachePurged(t, kOther), "a held block's purge lets nothing go");
	check(rgfxCacheSleep(t, kOther, true, &purgeable), "dormant again");
	check(rgfxCachePurged(t, kOther) &&
	          rgfxCacheLetGoOf(t, kOtherBlock, &purgeable) == kRgfxDormantPurged,
	      "purged, the dormant block is let go");
	check(rgfxCacheFind(t, kOther) != nullptr && rgfxCacheOfSurface(t, kOtherId) != nullptr,
	      "its cache is the surface's key still");
	check(!rgfxCachePurged(t, kOther), "it is asked once");
	int surfaceObject = 0;
	void *map = nullptr, *wire = nullptr, *held = nullptr;
	bool written = true;
	check(rgfxCacheReleasingSurface(t, kOtherBlock, &written) == nullptr && !written,
	      "no release yet: no surface releasing");
	check(!rgfxCacheLetGo(t, kOther, true, &surfaceObject, &map, &wire) &&
	          rgfxCacheLetGoOf(t, kOtherBlock, &purgeable) == kRgfxCacheKept,
	      "its release posted, nothing asks it to go again");
	check(rgfxCacheReleasingSurface(t, kOtherBlock, &written) == &surfaceObject && !written,
	      "its surface is the release's, its cache copied");
	rgfxCacheFind(t, kOther)->writes++;
	check(rgfxCacheReleasingSurface(t, kOtherBlock, &written) == &surfaceObject && written,
	      "a buffer wrote its cache since: written");
	check(rgfxCacheReleaseFlags(0) == RGFX_BLOCK_RELEASE_NO_CLIENT &&
	          rgfxCacheReleaseFlags(1) == 0 && rgfxCacheReleaseFlags(2) == 0 &&
	          rgfxCacheReleaseFlags(-1) == 0,
	      "its release says no client only where the surface has none");
	check(rgfxCacheReleased(t, kOtherBlock, &map, &wire, &held) == kOther && t.count == 1 &&
	          rgfxCacheFind(t, kOther) == nullptr && held == &surfaceObject,
	      "its release answered, it is cached no more");
	check(rgfxCacheReleasingSurface(t, kOtherBlock, &written) == nullptr && !written,
	      "answered: no surface releasing");

	// A dormant block whose surface had no client left at its sleep is let go, once, and only
	// a dormant one.
	check(!rgfxCacheUnclaimed(t, kBlock) &&
	          rgfxCacheLetGoOf(t, kBlock, &purgeable) == kRgfxDormantEmptied,
	      "one asked to go already keeps its first cause");
	RgfxCachedSurface *u = rgfxCacheAdd(t, kNotCached, 0x777000, 77);
	check(u != nullptr && !rgfxCacheUnclaimed(t, 0x777000), "a held block is not let go");
	check(rgfxCacheSleep(t, kNotCached, true, &purgeable) && rgfxCacheUnclaimed(t, 0x777000) &&
	          rgfxCacheLetGoOf(t, 0x777000, &purgeable) == kRgfxDormantNoClients,
	      "a dormant one with no client is let go");
	check(!rgfxCacheUnclaimed(t, 0x777000), "once");
	check(!rgfxCacheUnclaimed(t, 0x999000), "no block there: nothing");
	void *m2 = nullptr, *w2 = nullptr;
	check(rgfxCacheLetGo(t, kNotCached, false, &surfaceObject, &m2, &w2), "and goes");

	// A surface made empty while held does not sleep at its last let-go; made nonvolatile
	// again, it does.
	RgfxCachedSurface *e = rgfxCacheAdd(t, kOther, kOtherBlock, kOtherId);
	check(e != nullptr && !rgfxCachePurgeable(t, kOther, kRgfxPurgeableEmpty) && e->emptied,
	      "a held surface made empty");
	check(!rgfxCacheSleep(t, kOther, true, &purgeable) && !e->dormant, "does not sleep");
	check(!rgfxCacheUnpurged(t, kOther) && rgfxCacheSleep(t, kOther, true, &purgeable) &&
	          e->dormant,
	      "made nonvolatile, it sleeps");
}

// A cached surface's clients go where its header holds them, clamped to 32 bits; a surface not
// cached, or one with no header, writes none.
void clients()
{
	RgfxCachedSurface entries[2] = {};
	RgfxCacheTable t = {entries, 0, 2};
	volatile uint32_t header = 0;
	RgfxCachedSurface *c = rgfxCacheAdd(t, kSurface, kBlock, kId);
	check(c != nullptr && c->clientsAt == nullptr && c->object == nullptr,
	      "a cached surface starts with no header for its clients");
	check(!rgfxCacheNoteClients(t, kSurface, 2) && header == 0, "nothing written with no header");
	c->clientsAt = &header;
	check(rgfxCacheNoteClients(t, kSurface, 2) && header == 2, "two processes hold it");
	check(rgfxCacheNoteClients(t, kSurface, 1) && header == 1, "one went");
	check(rgfxCacheNoteClients(t, kSurface, -1) && header == 0, "below 0 is 0");
	check(rgfxCacheNoteClients(t, kSurface, 1ll << 40) && header == 0xffffffffu,
	      "past 32 bits, their most");
	check(!rgfxCacheNoteClients(t, kNotCached, 3) && header == 0xffffffffu,
	      "a surface not cached writes nothing");
}

// The table the kext sizes (kRgfxCachedMaxSurfaces) has a state for every block the surface table
// holds, the dormant ones at their cap among them, and for as many again released and not yet
// answered: a new surface past the dormant cap is still cached.
void room()
{
	static RgfxCachedSurface entries[kRgfxCachedMaxSurfaces];
	RgfxCacheTable t = {entries, 0, kRgfxCachedMaxSurfaces};
	bool purgeable = false;
	uint64_t key = 0x10000;
	bool ok = true;
	for (uint32_t i = 0; i < kRgfxDormantMaxBlocks; i++, key++)
		ok &= rgfxCacheAdd(t, key, key << 12, static_cast<uint32_t>(key)) != nullptr &&
		      rgfxCacheSleep(t, key, true, &purgeable);
	check(ok, "the dormant blocks at their cap are cached");
	int surfaceObject = 0;
	void *map = nullptr, *wire = nullptr;
	for (uint32_t i = 0; i < kRgfxSurfaceMaxBlocks; i++, key++)
		ok &= rgfxCacheAdd(t, key, key << 12, static_cast<uint32_t>(key)) != nullptr &&
		      !rgfxCacheLetGo(t, key, true, &surfaceObject, &map, &wire);
	check(ok, "so are as many blocks as the surface table holds, released and unanswered");
	for (uint32_t i = kRgfxDormantMaxBlocks; i < kRgfxSurfaceMaxBlocks; i++, key++)
		ok &= rgfxCacheAdd(t, key, key << 12, static_cast<uint32_t>(key)) != nullptr;
	check(ok && t.count == t.capacity, "and every block held past the dormant ones");
	check(rgfxCacheAdd(t, key, key << 12, static_cast<uint32_t>(key)) == nullptr,
	      "past them, none");
}
} // namespace

int main()
{
	RgfxCachedSurface entries[2] = {};
	RgfxCacheTable t = {entries, 0, 2};
	uint64_t offset = 0, mark = 0;

	RgfxCachedSurface *c = rgfxCacheAdd(t, kSurface, kBlock, kId);
	check(c != nullptr && c->cpuGen == 0 && c->writes == 0, "a cached surface starts at 0");
	check(rgfxCacheAdd(t, kSurface, kOtherBlock, kOtherId) == nullptr, "a key is added once");
	check(rgfxCacheAdd(t, kOther, kOtherBlock, kOtherId) != nullptr, "a second surface");
	check(rgfxCacheAdd(t, 0x9000, 0x90000, 9) == nullptr, "no room past the capacity");

	// Nothing written: a lock goes on, asking nothing.
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheGoOn, "nothing to copy");
	check(rgfxCacheNext(t, kNotCached, &offset, &mark) == kRgfxCacheGoOn,
	      "a surface not cached goes on");
	check(!rgfxCacheNoteWrite(t, kNotCached), "a write of a surface not cached is not kept");

	// A read-write lock moves the generation, of its own surface only.
	check(rgfxCacheReadWriteLock(t, kSurface) == 1, "the first read-write lock: generation 1");
	check(rgfxCacheReadWriteLock(t, kSurface) == 2, "the second: generation 2");
	check(rgfxCacheFind(t, kOther)->cpuGen == 0, "the other surface's stays");
	check(rgfxCacheReadWriteLock(t, kNotCached) == 0, "a surface not cached has none");

	// The GPU writes it: the next lock asks a download, and another waits for its answer.
	check(rgfxCacheNoteWrite(t, kSurface), "a write is named");
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheDownload && offset == kBlock &&
	          mark == 1,
	      "a lock after a write asks the download of the surface's block");
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheWait,
	      "a second lock waits for the first's download");
	check(rgfxCacheNext(t, kOther, &offset, &mark) == kRgfxCacheGoOn,
	      "the other surface's lock does not");

	// A write named while the download is in flight is still to copy after its answer.
	check(rgfxCacheNoteWrite(t, kSurface), "a write named during the download");
	rgfxCacheDownloaded(t, kSurface, mark, true);
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheDownload && mark == 2,
	      "the later write asks a download of its own");
	rgfxCacheDownloaded(t, kSurface, mark, true);
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheGoOn,
	      "every write copied: the lock goes on");

	// A download not answered OK copied nothing: the next lock asks again.
	check(rgfxCacheNoteWrite(t, kSurface), "another write");
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheDownload, "its download");
	rgfxCacheDownloaded(t, kSurface, mark, false);
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheDownload,
	      "a refused download is asked again");
	rgfxCacheDownloaded(t, kSurface, mark, true);

	// The block goes: locks wait for its release's answer, and the surface leaves the table.
	check(!rgfxCacheReleasing(t, 0x123000), "a block of no cached surface");
	check(rgfxCacheOfSurface(t, kId) == rgfxCacheFind(t, kSurface), "found by IOSurfaceID");
	check(rgfxCacheReleasing(t, kBlock), "the cached block's release is posted");
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheWait,
	      "a lock waits for the release");
	check(rgfxCacheOfSurface(t, kId) == nullptr, "a surface whose block goes has no generation");
	void *map = nullptr, *wire = nullptr;
	rgfxCacheRemove(t, rgfxCacheAtBlock(t, kBlock), &map, &wire);
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheGoOn,
	      "answered: the lock goes on, the surface no longer cached");
	check(t.count == 1 && rgfxCacheFind(t, kOther) != nullptr, "the other surface stays");

	// The device cache's reference, the surface's and the surface's wired memory: let go with
	// the block not cached, or a cached one not released; held for the answer of a cached one's
	// release, so the memory is unwired only once the host's copy into it is done.
	int memory = 0, header = 0, surfaceObject = 0;
	void *surface = &surfaceObject, *held = nullptr;
	RgfxCachedSurface *other = rgfxCacheFind(t, kOther);
	other->wire = &memory;
	other->map = &header;
	check(rgfxCacheLetGo(t, kNotCached, true, surface, &map, &wire) && map == nullptr &&
	          wire == nullptr,
	      "a surface not cached lets its cache go with its block");
	check(rgfxCacheReleased(t, kOtherBlock, &map, &wire, &held) == 0 && wire == nullptr &&
	          held == nullptr,
	      "no release asked: nothing answered");
	check(!rgfxCacheLetGo(t, kOther, true, surface, &map, &wire) && map == nullptr &&
	          wire == nullptr,
	      "a cached block being released keeps its cache, its surface and its memory wired");
	check(rgfxCacheNext(t, kOther, &offset, &mark) == kRgfxCacheWait, "its locks wait");
	check(rgfxCacheReleased(t, 0x123000, &map, &wire, &held) == 0 && wire == nullptr &&
	          held == nullptr,
	      "an answer for another block lets none go, nor unwires");
	check(rgfxCacheReleased(t, kOtherBlock, &map, &wire, &held) == kOther && t.count == 0 &&
	          map == &header && wire == &memory && held == surface,
	      "the answer lets it go: the cache's reference, then the surface's, then the memory to "
	      "unwire");
	check(rgfxCacheReleased(t, kOtherBlock, &map, &wire, &held) == 0 && wire == nullptr, "once");
	other = rgfxCacheAdd(t, kOther, kOtherBlock, kOtherId);
	check(other != nullptr && other->wire == nullptr, "cached again, afresh");
	other->wire = &memory;
	check(rgfxCacheLetGo(t, kOther, false, surface, &map, &wire) && t.count == 0 &&
	          wire == &memory,
	      "a block let go with no release lets the cache, the surface and the wiring go at once");

	dormancy();
	clients();
	room();
	std::printf("surface_caches_test: OK (%d checks)\n", checks);
	return 0;
}
