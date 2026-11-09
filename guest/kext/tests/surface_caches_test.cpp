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
// A deferred block's locks go on while its ask reads the surface and wait for the host's answer;
// then a read-write lock, or a read-only one after a GPU write, names its pages once, the others
// waiting, and a read-only lock of a block the GPU never wrote names none. A dormant deferred
// block stays at a client's going while others are left; its release names its pages first only
// where the GPU wrote it and the surface lives past the kext's three references. A block let go
// while an attach is in flight hands the let-go to that attach, waiting for nothing. A naming the
// host never answers leaves its pages wired. A block whose ask names its pages after all lets go
// of each wiring it took once, cached or not. A block kept for good with no wire of its own, let
// go while an attach is in flight, has that attach keep the pages' wiring where it named them.
// A naming not taken leaves the block deferred, so the lock, or the release, names it again,
// backing off from 10 ms to 320 ms, a dozen times within a lock's 2 s wait. Another block of a
// cached surface, a copy or one uncached, leaves the cached block's entry as it goes. A new block
// in place of a surface whose pages are still to take a released block's write-back waits for it
// until the release is late, and so does a CPU lock of the surface. A release is flagged unheld
// only where nothing but the kext holds the surface, and a lock or a new block of it met before
// the answer marks the header's clients for the host.
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
	check(!rgfxCacheLetGo(t, kOther, true, &surfaceObject, &map, &wire, true) &&
	          rgfxCacheLetGoOf(t, kOtherBlock, &purgeable) == kRgfxCacheKept,
	      "its release posted, nothing asks it to go again");
	check(rgfxCacheReleasingSurface(t, kOtherBlock, &written) == &surfaceObject && !written,
	      "its surface is the release's, its cache copied");
	rgfxCacheFind(t, kOther)->writes++;
	check(rgfxCacheReleasingSurface(t, kOtherBlock, &written) == &surfaceObject && written,
	      "a buffer wrote its cache since: written");
	const int32_t ported = kRgfxSurfaceKextRefs + 1; // a mach port's reference past the kext's
	check(rgfxCacheReleaseFlags(t, kOtherBlock, 0, ported) == RGFX_BLOCK_RELEASE_NO_CLIENT &&
	          rgfxCacheReleaseFlags(t, kOtherBlock, 1, ported) == 0 &&
	          rgfxCacheReleaseFlags(t, kOtherBlock, 2, ported) == 0 &&
	          rgfxCacheReleaseFlags(t, kOtherBlock, -1, ported) == 0 &&
	          !rgfxCacheFind(t, kOther)->unheld,
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
	check(rgfxCacheLetGo(t, kNotCached, false, &surfaceObject, &m2, &w2, true), "and goes");

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
		      !rgfxCacheLetGo(t, key, true, &surfaceObject, &map, &wire, true);
	check(ok, "so are as many blocks as the surface table holds, released and unanswered");
	for (uint32_t i = kRgfxDormantMaxBlocks; i < kRgfxSurfaceMaxBlocks; i++, key++)
		ok &= rgfxCacheAdd(t, key, key << 12, static_cast<uint32_t>(key)) != nullptr;
	check(ok && t.count == t.capacity, "and every block held past the dormant ones");
	check(rgfxCacheAdd(t, key, key << 12, static_cast<uint32_t>(key)) == nullptr,
	      "past them, none");
}
// A deferred block: gating, deferring, deferred, attached.
void deferral()
{
	RgfxCachedSurface entries[3] = {};
	RgfxCacheTable t = {entries, 0, 3};
	uint64_t offset = 0, mark = 0;
	void *map = nullptr, *wire = nullptr, *held = nullptr;
	RgfxCachedSurface *c = rgfxCacheAdd(t, kSurface, kBlock, kId);
	check(c != nullptr && c->deferral == kRgfxCacheNotDeferred && rgfxCacheWired(*c),
	      "a cached surface's pages are named at its ask, its memory wired");
	check(!rgfxCacheGated(t, kSurface), "one not gating is not gated");
	c->deferral = kRgfxCacheGating;
	check(!rgfxCacheWired(*c), "a gating one's memory is not wired");

	// Its ask reads the surface: a lock meanwhile goes on, its seed the gate's to read.
	check(rgfxCacheNext(t, kSurface, &offset, &mark, true) == kRgfxCacheGoOn &&
	          rgfxCacheReadWriteLock(t, kSurface) == 1,
	      "a read-write lock while the ask reads the surface goes on, its generation moved");
	check(rgfxCacheGated(t, kSurface) && c->deferral == kRgfxCacheDeferring,
	      "the surface was fresh");
	check(rgfxCacheNext(t, kSurface, &offset, &mark, true) == kRgfxCacheWait &&
	          rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheWait,
	      "the deferral not answered, locks wait");
	rgfxCacheSettled(t, kSurface, true);
	check(c->deferral == kRgfxCacheDeferred && !rgfxCacheWired(*c), "deferred, no pages");

	// A read-only lock of a block the GPU never wrote names no pages.
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheGoOn &&
	          c->deferral == kRgfxCacheDeferred,
	      "a read-only lock of an unwritten one attaches nothing");
	// A read-write one names them, once: the others wait; an attach not answered leaves them to
	// name again.
	offset = 0;
	check(rgfxCacheNext(t, kSurface, &offset, &mark, true) == kRgfxCacheAttach &&
	          offset == kBlock && c->deferral == kRgfxCacheAttaching,
	      "a read-write lock attaches its pages");
	check(!rgfxCacheWired(*c), "not wired while the attach is asked");
	check(rgfxCacheNext(t, kSurface, &offset, &mark, true) == kRgfxCacheWait &&
	          rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheWait,
	      "the other locks wait for it");
	check(rgfxCacheAttached(t, kSurface, kRgfxPagesNotNamed, &map, &wire, &held) == 0, "kept");
	check(c->deferral == kRgfxCacheDeferred, "an attach not answered OK names them again later");
	check(rgfxCacheNext(t, kSurface, &offset, &mark, true) == kRgfxCacheAttach,
	      "the next read-write lock asks again");
	check(rgfxCacheAttached(t, kSurface, kRgfxPagesNamed, &map, &wire, &held) == 0, "kept");
	check(c->deferral == kRgfxCacheAttached && rgfxCacheWired(*c), "attached, its memory wired");
	check(rgfxCacheNext(t, kSurface, &offset, &mark, true) == kRgfxCacheGoOn,
	      "then a lock goes on as for any block");
	check(rgfxCacheAttached(t, kSurface, kRgfxPagesNotNamed, &map, &wire, &held) == 0, "kept");
	check(c->deferral == kRgfxCacheAttached, "an answer with no attach asked changes nothing");

	// Another, the GPU writes it: a read-only lock attaches, then downloads.
	RgfxCachedSurface *d = rgfxCacheAdd(t, kOther, kOtherBlock, kOtherId);
	d->deferral = kRgfxCacheDeferred;
	check(rgfxCacheNoteWrite(t, kOther), "the GPU writes the other");
	check(rgfxCacheNext(t, kOther, &offset, &mark) == kRgfxCacheAttach && offset == kOtherBlock,
	      "a read-only lock after a GPU write attaches its pages");
	check(rgfxCacheAttached(t, kOther, kRgfxPagesNamed, &map, &wire, &held) == 0, "kept");
	check(rgfxCacheNext(t, kOther, &offset, &mark) == kRgfxCacheDownload && mark == 1,
	      "then downloads what the GPU wrote");
	rgfxCacheDownloaded(t, kOther, mark, true);
	check(rgfxCacheNext(t, kOther, &offset, &mark) == kRgfxCacheGoOn, "and goes on");

	// Refused, or not fresh: its pages named at the ask after all.
	RgfxCachedSurface *r = rgfxCacheAdd(t, kNotCached, 0xc0000, 44);
	r->deferral = kRgfxCacheGating;
	rgfxCacheSettled(t, kNotCached, false);
	check(r->deferral == kRgfxCacheNotDeferred && rgfxCacheWired(*r),
	      "a surface not fresh is not deferred");
	r->deferral = kRgfxCacheDeferring;
	rgfxCacheSettled(t, kNotCached, false);
	check(r->deferral == kRgfxCacheNotDeferred, "nor one whose deferral the host refused");
	rgfxCacheSettled(t, kNotCached, true);
	check(r->deferral == kRgfxCacheNotDeferred, "an answer after it changes nothing");
}

// On a host that answers which surfaces need a cache, a deferral refused, or a surface read not
// fresh, has the block ask with its pages; answered in place, its entry goes.
void ungated()
{
	RgfxCachedSurface entries[3] = {};
	RgfxCacheTable t = {entries, 0, 3};
	void *map = nullptr, *wire = nullptr;
	int mapping = 0, memory = 0;
	RgfxCachedSurface *c = rgfxCacheAdd(t, kSurface, kBlock, kId);
	c->map = &mapping;
	c->wire = &memory;
	c->deferral = kRgfxCacheGating;
	check(rgfxCacheUngated(t, kSurface, &map, &wire) && map == &mapping && wire == &memory,
	      "a gating entry goes, its mapping and memory the caller's to let go");
	check(rgfxCacheFind(t, kSurface) == nullptr && t.count == 0, "nothing is left of it");

	c = rgfxCacheAdd(t, kSurface, kBlock, kId);
	c->deferral = kRgfxCacheGating;
	check(rgfxCacheGated(t, kSurface), "fresh, it asks to be deferred");
	check(rgfxCacheUngated(t, kSurface, &map, &wire) && t.count == 0,
	      "a deferring one goes too, its deferral refused and its pages in place");

	// One the host deferred, or one named cached, stays.
	c = rgfxCacheAdd(t, kSurface, kBlock, kId);
	c->deferral = kRgfxCacheGating;
	rgfxCacheSettled(t, kSurface, true);
	check(!rgfxCacheUngated(t, kSurface, &map, &wire) && map == nullptr && wire == nullptr &&
	          rgfxCacheFind(t, kSurface) == c,
	      "a deferred entry stays");
	RgfxCachedSurface *d = rgfxCacheAdd(t, kOther, kOtherBlock, kOtherId);
	d->deferral = kRgfxCacheGating;
	rgfxCacheSettled(t, kOther, false);
	check(!rgfxCacheUngated(t, kOther, &map, &wire) && rgfxCacheFind(t, kOther) == d,
	      "so does one whose pages were named cached");
	check(!rgfxCacheUngated(t, kNotCached, &map, &wire), "and a surface not cached has none");
}

// Op 5 keeps a dormant deferred block while clients are left, and its release names its pages
// only where the GPU wrote it and the surface lives on.
void deferredRelease()
{
	// The kext's own references of a surface at a release: the table's, the device cache's and
	// the release thread's.
	static_assert(kRgfxSurfaceTableRef == 1 && kRgfxSurfaceDeviceCacheRef == 1 &&
	                  kRgfxSurfaceReleaseRef == 1,
	              "one reference each");
	check(kRgfxSurfaceKextRefs == 3, "the kext holds three references of a surface it releases");

	RgfxCachedSurface entries[2] = {};
	RgfxCacheTable t = {entries, 0, 2};
	bool purgeable = false;
	void *held = nullptr;
	RgfxCachedSurface *c = rgfxCacheAdd(t, kSurface, kBlock, kId);
	c->deferral = kRgfxCacheDeferred;
	check(rgfxCacheSleep(t, kSurface, true, &purgeable), "a deferred block dormant");
	check(!rgfxCachePurged(t, kSurface, 1) && c->letGo == kRgfxCacheKept,
	      "a client going with one left keeps it");
	check(rgfxCachePurged(t, kSurface, 0) && c->letGo == kRgfxDormantPurged,
	      "the last client's going lets it go");
	RgfxCachedSurface *a = rgfxCacheAdd(t, kOther, kOtherBlock, kOtherId);
	a->deferral = kRgfxCacheAttached;
	check(rgfxCacheSleep(t, kOther, true, &purgeable) && rgfxCachePurged(t, kOther, 2),
	      "an attached one goes at any client's going, as any block");

	// Its release: not yet releasing, nothing to do.
	check(rgfxCacheReleaseNext(t, kBlock, 1, 9) == kRgfxCacheReleaseAsIs, "not released yet");
	int surfaceObject = 0;
	void *map = nullptr, *wire = nullptr;
	check(!rgfxCacheLetGo(t, kSurface, true, &surfaceObject, &map, &wire, true), "released");
	check(rgfxCacheReleaseNext(t, kBlock, 1, 9) == kRgfxCacheReleaseAsIs,
	      "the GPU never wrote it: its pages are fresh, nothing to copy");
	check(rgfxCacheNoteWrite(t, kSurface), "the GPU wrote it");
	check(rgfxCacheReleaseNext(t, kBlock, 0, kRgfxSurfaceKextRefs) == kRgfxCacheReleaseAsIs &&
	          c->deferral == kRgfxCacheDeferred,
	      "held by the kext alone: nothing reads its pages");
	check(rgfxCacheReleasePeek(t, kBlock, 0, kRgfxSurfaceKextRefs) == kRgfxCacheReleaseAsIs &&
	          rgfxCacheReleasePeek(t, kBlock, 0, kRgfxSurfaceKextRefs + 1) ==
	              kRgfxCacheReleaseAttach &&
	          c->deferral == kRgfxCacheDeferred,
	      "the release thread's look answers as the release would, and changes nothing");
	check(rgfxCacheReleaseNext(t, kBlock, 0, kRgfxSurfaceKextRefs + 1) ==
	              kRgfxCacheReleaseAttach &&
	          c->deferral == kRgfxCacheAttaching,
	      "with no client but a reference past the kext's (a port), its pages are named");
	check(rgfxCacheReleaseNext(t, kBlock, 0, kRgfxSurfaceKextRefs + 1) == kRgfxCacheReleaseWait,
	      "while they are named, the release waits");
	check(rgfxCacheReleasePeek(t, kBlock, 0, kRgfxSurfaceKextRefs + 1) == kRgfxCacheReleaseWait &&
	          c->deferral == kRgfxCacheAttaching,
	      "and a look then says it waits: the release thread hands it to the naming worker");
	uint64_t offset = 0, mark = 0;
	check(rgfxCacheNext(t, kSurface, &offset, &mark, true) == kRgfxCacheWait,
	      "and a lock waits for the release");
	check(rgfxCacheAttached(t, kSurface, kRgfxPagesNamed, &map, &wire, &held) == 0, "kept");
	check(rgfxCacheReleaseNext(t, kBlock, 1, 9) == kRgfxCacheReleaseAsIs,
	      "attached, it goes as any block, its release copying the cache");
	c->deferral = kRgfxCacheDeferred;
	check(rgfxCacheReleaseNext(t, kBlock, 1, kRgfxSurfaceKextRefs) == kRgfxCacheReleaseAttach,
	      "a client left names them too");
}
// A block let go, not released (its close unanswered, kept for good), while a lock attaches its
// pages: the let-go returns at once, the table keeping the surface for the attach, which lets it
// go once answered. Nothing waits for the host under the contexts' lock: another context's
// blocks are made and let go meanwhile, and their locks go on.
void forgottenAttach()
{
	RgfxCachedSurface entries[3] = {};
	RgfxCacheTable t = {entries, 0, 3};
	uint64_t offset = 0, mark = 0;
	int surfaceObject = 0, otherObject = 0;
	void *map = nullptr, *wire = nullptr, *held = nullptr;
	RgfxCachedSurface *c = rgfxCacheAdd(t, kSurface, kBlock, kId);
	c->deferral = kRgfxCacheDeferred;
	c->map = &map;
	c->wire = &wire;
	check(rgfxCacheNext(t, kSurface, &offset, &mark, true) == kRgfxCacheAttach,
	      "a lock attaches the pages, its SURFACE_PAGES unanswered");
	check(!rgfxCacheLetGo(t, kSurface, false, &surfaceObject, &map, &wire, true) &&
	          map == nullptr && wire == nullptr && c->forgotten && !c->releasing &&
	          c->surface == &surfaceObject,
	      "the block's let-go returns at once, the attach to let the surface go");
	check(rgfxCacheOfSurface(t, kId) == nullptr, "its IOSurfaceID names no cached surface now");
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheWait,
	      "another lock of it waits for the attach, not for the let-go");

	// Meanwhile another context makes a block, uses it and lets it go.
	check(rgfxCacheAdd(t, kOther, kOtherBlock, kOtherId) != nullptr, "another block is made");
	check(rgfxCacheNoteWrite(t, kOther) &&
	          rgfxCacheNext(t, kOther, &offset, &mark) == kRgfxCacheDownload,
	      "its locks go on");
	rgfxCacheDownloaded(t, kOther, mark, true);
	check(rgfxCacheLetGo(t, kOther, false, &otherObject, &map, &wire, true) && t.count == 1,
	      "and it goes, with the attach still in flight");

	// The attach answered: it takes the surface out, the let-go its own.
	map = wire = nullptr;
	check(rgfxCacheAttached(t, kSurface, kRgfxPagesNotNamed, &map, &wire, &held) == kSurface && t.count == 0,
	      "answered, the attach lets the surface go");
	check(held == &surfaceObject && map != nullptr && wire != nullptr,
	      "its mapping, its memory and the surface's reference for the caller");
	check(rgfxCacheNext(t, kSurface, &offset, &mark, true) == kRgfxCacheGoOn,
	      "a lock then finds it cached no more");
	check(rgfxCacheAttached(t, kSurface, kRgfxPagesNamed, &map, &wire, &held) == 0 && held == nullptr,
	      "an answer for a surface gone does nothing");
}
// A block kept for good (its close unanswered, or a create naming it in flight), let go while a
// lock attaches its pages: where the let-go took no wire of the pages of its own (it read them
// unnamed, before the attach began), the attach keeps the wiring it made for good, as the host
// may map the pages through the block; where the block holds a wire of its own, or the attach
// named nothing, or the block is released, the attach lets its wiring go as before.
void keptAttachWire()
{
	RgfxCachedSurface entries[1] = {};
	RgfxCacheTable t = {entries, 0, 1};
	uint64_t offset = 0, mark = 0;
	int surfaceObject = 0, memory = 0, header = 0;
	void *map = nullptr, *wire = nullptr, *held = nullptr;
	struct Case {
		bool releasing, blockWired;
		RgfxPagesNaming naming;
		bool keep;
		const char *what;
	};
	const Case cases[] = {
	    {false, false, kRgfxPagesNamed, true,
	     "kept for good with no wire of its own: the named pages stay wired"},
	    {false, false, kRgfxPagesUnanswered, true,
	     "and pages asked and not answered, which the host may yet map"},
	    {false, false, kRgfxPagesNotNamed, false,
	     "pages not named: nothing the host maps, the wiring goes"},
	    {false, true, kRgfxPagesNamed, false,
	     "a block with a wire of its own: the attach's wiring goes, the block's holds them"},
	};
	for (const Case &k : cases) {
		RgfxCachedSurface *c = rgfxCacheAdd(t, kSurface, kBlock, kId);
		c->deferral = kRgfxCacheDeferred;
		c->map = &header;
		c->wire = &memory;
		check(rgfxCacheNext(t, kSurface, &offset, &mark, true) == kRgfxCacheAttach,
		      "a read-write lock attaches its pages");
		check(!rgfxCacheLetGo(t, kSurface, k.releasing, &surfaceObject, &map, &wire,
		                      k.blockWired) &&
		          c->forgotten && c->keepWire == !k.blockWired,
		      "the block's let-go hands its going to the attach, deciding the wiring");
		bool keep = !k.keep;
		check(rgfxCacheAttached(t, kSurface, k.naming, &map, &wire, &held, &keep) == kSurface &&
		          t.count == 0 && wire == &memory && map == &header && held == &surfaceObject,
		      "answered, the attach lets the surface go");
		check(keep == k.keep, k.what);
	}
	// Released, its let-go waits for the release's answer, which unwires the pages once the host
	// is done with them: the attach keeps nothing.
	RgfxCachedSurface *c = rgfxCacheAdd(t, kSurface, kBlock, kId);
	c->deferral = kRgfxCacheDeferred;
	check(rgfxCacheNext(t, kSurface, &offset, &mark, true) == kRgfxCacheAttach, "attaching");
	check(!rgfxCacheLetGo(t, kSurface, true, &surfaceObject, &map, &wire, false) &&
	          c->releasing && !c->forgotten && !c->keepWire,
	      "a released block keeps no wiring past its answer");
	bool keep = true;
	check(rgfxCacheAttached(t, kSurface, kRgfxPagesNamed, &map, &wire, &held, &keep) == 0 &&
	          !keep && c->deferral == kRgfxCacheAttached,
	      "its attach, answered, leaves the release to let the pages go");
}
// A naming not taken (the surface's memory not wired, its runs not written, the host refusing
// them) is made again: by the lock that asked it, after a GPU write or for a read-write lock,
// and by the release, while the other locks wait; the waits between back off.
void attachRetry()
{
	check(rgfxAttachRetryMs(0) == 10 && rgfxAttachRetryMs(1) == 10 &&
	          rgfxAttachRetryMs(2) == 20 && rgfxAttachRetryMs(3) == 40 &&
	          rgfxAttachRetryMs(6) == 320 && rgfxAttachRetryMs(7) == 320 &&
	          rgfxAttachRetryMs(1000) == 320,
	      "the waits between namings double from 10 ms to 320 ms");
	// The namings a lock makes in its 2 s wait, each taking no time.
	uint32_t at = 0, tries = 1;
	while (at + rgfxAttachRetryMs(tries) < 2000)
		at += rgfxAttachRetryMs(tries++);
	check(tries >= 10 && tries <= 13, "a dozen namings in a lock's wait");

	RgfxCachedSurface entries[1] = {};
	RgfxCacheTable t = {entries, 0, 1};
	uint64_t offset = 0, mark = 0;
	void *map = nullptr, *wire = nullptr, *held = nullptr;
	RgfxCachedSurface *c = rgfxCacheAdd(t, kSurface, kBlock, kId);
	c->deferral = kRgfxCacheDeferred;
	check(rgfxCacheNoteWrite(t, kSurface), "the GPU writes it");
	for (int i = 0; i < 3; i++) {
		check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheAttach,
		      "a read-only lock after a GPU write names its pages");
		check(rgfxCacheNext(t, kSurface, &offset, &mark, true) == kRgfxCacheWait,
		      "another lock waits for the naming");
		check(rgfxCacheAttached(t, kSurface, kRgfxPagesNotNamed, &map, &wire, &held) == 0 &&
		          c->deferral == kRgfxCacheDeferred && !rgfxCacheWired(*c),
		      "not taken: deferred still, to name again, nothing wired");
	}
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheAttach, "named again");
	check(rgfxCacheAttached(t, kSurface, kRgfxPagesNamed, &map, &wire, &held) == 0 &&
	          c->deferral == kRgfxCacheAttached,
	      "taken at last");
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheDownload && mark == 1,
	      "then the lock downloads what the GPU wrote, never going on over pages without it");

	// The release, where the surface lives on: named again too, the locks waiting.
	c->deferral = kRgfxCacheDeferred;
	c->copied = 0;
	check(rgfxCacheReleasing(t, kBlock), "its release is posted");
	for (int i = 0; i < 2; i++) {
		check(rgfxCacheReleaseNext(t, kBlock, 1, kRgfxSurfaceKextRefs) ==
		          kRgfxCacheReleaseAttach,
		      "the release names its pages");
		check(rgfxCacheReleaseNext(t, kBlock, 1, kRgfxSurfaceKextRefs) == kRgfxCacheReleaseWait &&
		          rgfxCacheNext(t, kSurface, &offset, &mark, true) == kRgfxCacheWait,
		      "the release and the locks wait for the naming");
		check(rgfxCacheAttached(t, kSurface, kRgfxPagesNotNamed, &map, &wire, &held) == 0 &&
		          c->deferral == kRgfxCacheDeferred && c->releasing,
		      "not taken: deferred and releasing still");
	}
	check(rgfxCacheReleaseNext(t, kBlock, 1, kRgfxSurfaceKextRefs) == kRgfxCacheReleaseAttach,
	      "named again");
	check(rgfxCacheAttached(t, kSurface, kRgfxPagesNamed, &map, &wire, &held) == 0 &&
	          rgfxCacheHostPages(*c) &&
	          rgfxCacheReleaseNext(t, kBlock, 1, kRgfxSurfaceKextRefs) == kRgfxCacheReleaseAsIs,
	      "taken: the release goes, copying the cache into the pages");
}
// A naming the host never answers (its SURFACE_PAGES timed out): nothing the host may yet map
// goes back to the guest, where the old code unwired it.
void unanswered()
{
	RgfxCachedSurface entries[3] = {};
	RgfxCacheTable t = {entries, 0, 3};
	uint64_t offset = 0, mark = 0;
	int surfaceObject = 0;
	void *map = nullptr, *wire = nullptr, *held = nullptr;

	// An attach not answered: its pages stay wired as named ones, never named again.
	RgfxCachedSurface *c = rgfxCacheAdd(t, kSurface, kBlock, kId);
	c->deferral = kRgfxCacheDeferred;
	check(!rgfxCacheHostPages(*c) && !rgfxCacheWired(*c), "deferred: the host maps no pages");
	check(rgfxCacheNext(t, kSurface, &offset, &mark, true) == kRgfxCacheAttach &&
	          rgfxCacheHostPages(*c),
	      "being named, the host may map them");
	check(rgfxCacheAttached(t, kSurface, kRgfxPagesUnanswered, &map, &wire, &held) == 0 &&
	          c->deferral == kRgfxCacheAttachLate,
	      "the attach not answered");
	check(rgfxCacheWired(*c) && rgfxCacheHostPages(*c),
	      "its memory stays wired, the host free to map it yet");
	check(rgfxCacheNext(t, kSurface, &offset, &mark, true) == kRgfxCacheGoOn,
	      "a lock goes on, naming the pages no second time");
	check(rgfxCacheNoteWrite(t, kSurface) &&
	          rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheDownload,
	      "and a lock after a GPU write asks its download, as for an attached block");
	rgfxCacheDownloaded(t, kSurface, mark, false);
	// Its release names nothing again, and the table keeps the wired memory until the answer.
	check(!rgfxCacheLetGo(t, kSurface, true, &surfaceObject, &map, &wire, true) && map == nullptr &&
	          wire == nullptr,
	      "released, its memory stays with the table");
	check(rgfxCacheReleaseNext(t, kBlock, 1, 9) == kRgfxCacheReleaseAsIs,
	      "its release attaches nothing again");
	check(rgfxCacheWired(*c), "and its memory is unwired only at the release's answer");

	// An attach not answered while its block went (kept for good): the attach lets the surface
	// go, its memory wired, the block holding a wire of its own (rgfxCacheHostPages).
	RgfxCachedSurface *f = rgfxCacheAdd(t, kOther, kOtherBlock, kOtherId);
	f->deferral = kRgfxCacheDeferred;
	check(rgfxCacheNext(t, kOther, &offset, &mark, true) == kRgfxCacheAttach &&
	          rgfxCacheHostPages(*f),
	      "being named as its block goes: the block wires the pages itself");
	check(!rgfxCacheLetGo(t, kOther, false, &surfaceObject, &map, &wire, true), "forgotten");
	check(rgfxCacheAttached(t, kOther, kRgfxPagesUnanswered, &map, &wire, &held) == kOther &&
	          held == &surfaceObject,
	      "not answered, the attach takes the surface out");
	check(rgfxPagesNamingWired(kRgfxPagesUnanswered),
	      "its memory let go as wired, the block's own wire keeping the pages");

	// A deferral not answered: no page was named, so nothing is wired, and the table keeps the
	// surface until the block's release is answered.
	RgfxCachedSurface *d = rgfxCacheAdd(t, kNotCached, 0xc0000, 44);
	d->deferral = kRgfxCacheDeferring;
	rgfxCacheSettled(t, kNotCached, true);
	check(d->deferral == kRgfxCacheDeferred && !rgfxCacheWired(*d) && !rgfxCacheHostPages(*d),
	      "an unanswered deferral named no pages");
	check(!rgfxCacheLetGo(t, kNotCached, true, &surfaceObject, &map, &wire, true) && map == nullptr,
	      "its block goes through its release, the surface kept for the answer");
	check(rgfxCacheReleaseNext(t, 0xc0000, 1, 9) == kRgfxCacheReleaseAsIs,
	      "and its release names no pages");
}

// Another block of a cached surface goes: a copy, or one in place uncached, whose locks follow the
// surface by the same device cache. The surface's cached entry is the cached block's alone: where
// its release is unanswered, its memory stays wired until the answer, for the host's copy of the
// cache into the pages; where it lives, its locks go on, an attach in flight among them.
void otherBlocks()
{
	constexpr uint64_t kCopyBlock = 0x100000;
	RgfxCachedSurface entries[2] = {};
	RgfxCacheTable t = {entries, 0, 2};
	uint64_t offset = 0, mark = 0;
	int memory = 0, header = 0, surfaceObject = 0, copyObject = 0;
	void *map = nullptr, *wire = nullptr, *held = nullptr;

	// Released and unanswered: the copy block's let-go hands back nothing of the cached block's.
	RgfxCachedSurface *c = rgfxCacheAdd(t, kSurface, kBlock, kId);
	c->wire = &memory;
	c->map = &header;
	check(rgfxCacheNoteWrite(t, kSurface), "the GPU writes the cached block");
	check(!rgfxCacheBlockLetGo(t, kSurface, kBlock, true, &surfaceObject, &map, &wire, true) &&
	          map == nullptr && wire == nullptr,
	      "the cached block goes, its release unanswered");
	check(rgfxCacheBlockLetGo(t, kSurface, kCopyBlock, false, &copyObject, &map, &wire, true) &&
	          map == nullptr && wire == nullptr,
	      "a copy block of the surface goes after it: its own references go, none of the cached "
	      "block's");
	check(rgfxCacheFind(t, kSurface) == c && c->releasing && c->wire == &memory &&
	          c->surface == &surfaceObject && rgfxCacheWired(*c),
	      "the release's entry stays, its memory wired for the host's copy into the pages");
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheWait,
	      "and a lock of the surface still waits for the release");
	check(rgfxCacheReleased(t, kBlock, &map, &wire, &held) == kSurface && map == &header &&
	          wire == &memory && held == &surfaceObject && t.count == 0,
	      "the answer lets it go, its memory unwired only then");

	// Living, deferred: a copy block's let-go leaves it, and its attach in flight.
	RgfxCachedSurface *d = rgfxCacheAdd(t, kOther, kOtherBlock, kOtherId);
	d->deferral = kRgfxCacheDeferred;
	check(rgfxCacheBlockLetGo(t, kOther, kCopyBlock, false, &copyObject, &map, &wire, true) &&
	          map == nullptr && wire == nullptr && rgfxCacheFind(t, kOther) == d,
	      "a living cached block stays as a copy block of its surface goes");
	check(rgfxCacheNext(t, kOther, &offset, &mark, true) == kRgfxCacheAttach,
	      "a lock names its pages");
	check(rgfxCacheBlockLetGo(t, kOther, kCopyBlock, false, &copyObject, &map, &wire, true) &&
	          !d->forgotten && d->surface == nullptr,
	      "a copy block going meanwhile hands the attach nothing");
	check(rgfxCacheAttached(t, kOther, kRgfxPagesNamed, &map, &wire, &held) == 0 &&
	          d->deferral == kRgfxCacheAttached && t.count == 1,
	      "the attach answered, the cached block keeps its pages");
	// Its own let-go is rgfxCacheLetGo's.
	check(!rgfxCacheBlockLetGo(t, kOther, kOtherBlock, true, &surfaceObject, &map, &wire, true) &&
	          d->releasing,
	      "the cached block's own let-go releases it");
	check(rgfxCacheBlockLetGo(t, kNotCached, kCopyBlock, false, &copyObject, &map, &wire, true) &&
	          map == nullptr && wire == nullptr,
	      "a block of a surface not cached lets its references go");
}

// A new block in place of a surface whose cached block went, its release unanswered, waits for
// the host's copy of that block's cache into the pages, past a lock's 2 s, until the release is
// late; then it goes on. A CPU lock of the surface waits for the answer as long, and for another
// lock's download, or its own, as long too. A block kept for good, or one let go with its
// attach in flight, posts no release and leaves nothing to wait for; nor does another surface's
// release.
void writeBack()
{
	RgfxCachedSurface entries[3] = {};
	RgfxCacheTable t = {entries, 0, 3};
	uint64_t offset = 0, mark = 0;
	int surfaceObject = 0, otherObject = 0;
	void *map = nullptr, *wire = nullptr, *held = nullptr;
	bool purgeable = false;

	RgfxCachedSurface *c = rgfxCacheAdd(t, kSurface, kBlock, kId);
	c->object = &surfaceObject;
	check(!rgfxCacheWritingBack(t, &surfaceObject), "a held cached block: no write-back pending");
	check(rgfxCacheSleep(t, kSurface, true, &purgeable) &&
	          !rgfxCacheWritingBack(t, &surfaceObject),
	      "nor a dormant one");
	check(rgfxCacheNoteWrite(t, kSurface) &&
	          !rgfxCacheBlockLetGo(t, kSurface, kBlock, true, &surfaceObject, &map, &wire, true),
	      "the GPU wrote it, and its block goes, its release posted");
	check(rgfxCacheWritingBack(t, &surfaceObject), "its pages are still to take the write-back");
	check(!rgfxCacheWritingBack(t, &otherObject) && !rgfxCacheWritingBack(t, nullptr),
	      "another surface's pages are not");
	check(rgfxCacheNewBlockStep(true, 0) == kRgfxNewBlockWait, "a new block of it waits");
	check(rgfxCacheNewBlockStep(true, 2000) == kRgfxNewBlockWait,
	      "past a lock's 2 s, still: the release is answered once the buffers that used the cache "
	      "finish");
	check(rgfxCacheNewBlockStep(true, kRgfxWriteBackWaitMs - 1) == kRgfxNewBlockWait &&
	          kRgfxWriteBackWaitMs == 10000,
	      "until the release is late, as the kext counts it");
	check(rgfxCacheNewBlockStep(true, kRgfxWriteBackWaitMs) == kRgfxNewBlockLate,
	      "then it goes on, logged");
	check(rgfxCacheNewBlockStep(false, 0) == kRgfxNewBlockMake &&
	          rgfxCacheNewBlockStep(false, kRgfxWriteBackWaitMs) == kRgfxNewBlockMake,
	      "with no write-back pending, it is made");
	// A CPU lock of the surface meanwhile, the bundle's copy-path locks among them, waits for the
	// same answer as long: past it, it would read or write pages the copy then lands over.
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheWait,
	      "a lock of the surface waits for the release");
	check(rgfxCacheLockWaitMs(t, kSurface, 2000) == kRgfxWriteBackWaitMs,
	      "as long as a new block waits for it, past the lock's 2 s");
	check(rgfxCacheLockWaitMs(t, kSurface, kRgfxWriteBackWaitMs + 1) == kRgfxWriteBackWaitMs + 1,
	      "never shorter than the lock's own wait");
	check(rgfxCacheLockWaitMs(t, kOther, 2000) == 2000 &&
	          rgfxCacheLockWaitMs(t, kNotCached, 2000) == 2000,
	      "a surface with no release posted waits as the lock does");
	check(rgfxCacheReleased(t, kBlock, &map, &wire, &held) == kSurface &&
	          !rgfxCacheWritingBack(t, &surfaceObject),
	      "the release answered, the pages hold the cache: the new block is made");
	check(rgfxCacheLockWaitMs(t, kSurface, 2000) == 2000, "and a lock waits as it does");
	// A lock waiting for another lock's download waits as long as the download's answer, and
	// that as long as a release's: a lock that went on at 2 s read pages without the GPU's
	// writes, as the host's copy came later.
	c = rgfxCacheAdd(t, kSurface, kBlock, kId);
	check(rgfxCacheNoteWrite(t, kSurface) &&
	          rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheDownload &&
	          rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheWait,
	      "a second lock waits for the first's download");
	check(rgfxCacheLockWaitMs(t, kSurface, 2000) == kRgfxWriteBackWaitMs,
	      "as long as for a release, past the lock's 2 s");
	check(rgfxCacheDownloadWaitMs(0, 2000) == kRgfxWriteBackWaitMs &&
	          rgfxCacheDownloadWaitMs(1500, 2000) == kRgfxWriteBackWaitMs - 1500,
	      "the download's own answer: what is left of that wait from the lock's start");
	check(rgfxCacheDownloadWaitMs(kRgfxWriteBackWaitMs, 2000) == 0 &&
	          rgfxCacheDownloadWaitMs(kRgfxWriteBackWaitMs + 7, 2000) == 0,
	      "none left: the lock goes on without asking");
	check(rgfxCacheDownloadWaitMs(100, kRgfxWriteBackWaitMs + 100) == kRgfxWriteBackWaitMs,
	      "never shorter than the lock's own wait");
	rgfxCacheDownloaded(t, kSurface, mark, true);
	check(rgfxCacheReleased(t, kBlock, &map, &wire, &held) == 0 &&
	          rgfxCacheBlockLetGo(t, kSurface, kBlock, false, &surfaceObject, &map, &wire, true) &&
	          t.count == 0,
	      "that block goes, kept for good");

	// Kept for good: no release, nothing to wait for.
	c = rgfxCacheAdd(t, kSurface, kBlock, kId);
	c->object = &surfaceObject;
	check(rgfxCacheBlockLetGo(t, kSurface, kBlock, false, &surfaceObject, &map, &wire, true) &&
	          !rgfxCacheWritingBack(t, &surfaceObject),
	      "a block kept for good posts no release");
	// Kept for good with an attach in flight: the attach lets it go, and posts no release.
	c = rgfxCacheAdd(t, kSurface, kBlock, kId);
	c->object = &surfaceObject;
	c->deferral = kRgfxCacheDeferred;
	check(rgfxCacheNext(t, kSurface, &offset, &mark, true) == kRgfxCacheAttach,
	      "a lock names a deferred block's pages");
	check(!rgfxCacheBlockLetGo(t, kSurface, kBlock, false, &surfaceObject, &map, &wire, true) &&
	          c->forgotten && !rgfxCacheWritingBack(t, &surfaceObject),
	      "its block kept for good meanwhile: nothing to wait for");
}

// A block whose ask names its pages after all (deferSurfaceBlock): each wiring taken is let go
// once, by its cache or by the asker's hold, cached or not, whatever the naming's outcome.
void askNaming()
{
	const RgfxPagesNaming namings[] = {kRgfxPagesNamed, kRgfxPagesNotNamed,
	                                   kRgfxPagesUnanswered};
	const bool cachedness[] = {false, true};
	for (const bool cached : cachedness) {
		for (const RgfxPagesNaming naming : namings) {
			RgfxCachedSurface entries[1] = {};
			RgfxCacheTable t = {entries, 0, 1};
			// The naming's wiring, which it keeps but where the host answered it took none; the
			// hold's own, where named.
			const bool again = naming == kRgfxPagesNamed && rgfxAskHoldWiresAgain(cached, naming);
			const int taken = (rgfxPagesNamingWired(naming) ? 1 : 0) + (again ? 1 : 0);
			// The cache's, at its entry's removal, where its settle leaves it wired; the hold's,
			// where it holds the pages: named, or not answered.
			int letGo = 0;
			if (cached) {
				RgfxCachedSurface *c = rgfxCacheAdd(t, kSurface, kBlock, kId);
				c->deferral = kRgfxCacheGating;
				rgfxCacheSettled(t, kSurface, naming != kRgfxPagesNamed);
				letGo += rgfxCacheWired(*c) ? 1 : 0;
			}
			letGo += naming != kRgfxPagesNotNamed ? 1 : 0;
			check(taken == letGo, cached ? "a cached block's ask lets go of each wiring once"
			                             : "an uncached block's ask lets go of each wiring once");
		}
	}
}
} // namespace

// A cached block's release is flagged unheld (RGFX_BLOCK_RELEASE_UNHELD) only where the surface
// has no client and no reference past the kext's three, decided under the lock as a deferred
// block's naming is; a lock or a new block of the surface met while the release is unanswered,
// a process holding the surface again, marks the header's count of clients past 0 for the host,
// once, and a release posted again is posted held.
void unheldReleases()
{
	RgfxCachedSurface entries[2] = {};
	RgfxCacheTable t = {entries, 0, 2};
	uint64_t offset = 0, mark = 0;
	int surfaceObject = 0, otherObject = 0;
	void *map = nullptr, *wire = nullptr;
	volatile uint32_t clients = 0, otherClients = 0;
	const int32_t kext = kRgfxSurfaceKextRefs, ported = kRgfxSurfaceKextRefs + 1;
	const uint32_t both = RGFX_BLOCK_RELEASE_NO_CLIENT | RGFX_BLOCK_RELEASE_UNHELD;

	RgfxCachedSurface *c = rgfxCacheAdd(t, kSurface, kBlock, kId);
	c->object = &surfaceObject;
	c->clientsAt = &clients;
	check(rgfxCacheReleaseFlags(t, kBlock, 0, kext) == RGFX_BLOCK_RELEASE_NO_CLIENT && !c->unheld,
	      "a block not released: no client, never unheld");
	check(rgfxCacheNoteWrite(t, kSurface) &&
	          !rgfxCacheBlockLetGo(t, kSurface, kBlock, true, &surfaceObject, &map, &wire, true),
	      "the GPU wrote it, and its block goes");
	check(rgfxCacheReleaseFlags(t, kBlock, 1, kext) == 0 && !c->unheld, "a client left: held");
	check(rgfxCacheReleaseFlags(t, kBlock, 0, ported) == RGFX_BLOCK_RELEASE_NO_CLIENT &&
	          !c->unheld,
	      "no client but a port: held, no client");
	check(rgfxCacheReleaseFlags(t, kBlock, 0, kext) == both && c->unheld,
	      "held by the kext alone: unheld");
	check(rgfxCacheReleaseFlags(t, kOtherBlock, 0, kext) == RGFX_BLOCK_RELEASE_NO_CLIENT,
	      "a block no cached surface releases: no client only");

	// Taken again before the answer: a lock marks the header, once, and waits for the answer.
	check(rgfxCacheNext(t, kSurface, &offset, &mark, true) == kRgfxCacheWait && c->retaken &&
	          clients == 1,
	      "a lock of it, unanswered: the header counts a client, and it waits");
	clients = 0;
	check(rgfxCacheNext(t, kSurface, &offset, &mark) == kRgfxCacheWait && clients == 0,
	      "a second lock marks nothing more");
	check(rgfxCacheReleaseFlags(t, kBlock, 0, kext) == RGFX_BLOCK_RELEASE_NO_CLIENT && !c->unheld,
	      "taken again, the release posted again is held");

	// A new block of a surface whose unheld release is unanswered marks its header too.
	RgfxCachedSurface *o = rgfxCacheAdd(t, kOther, kOtherBlock, kOtherId);
	o->object = &otherObject;
	o->clientsAt = &otherClients;
	check(rgfxCacheNoteWrite(t, kOther) &&
	          !rgfxCacheBlockLetGo(t, kOther, kOtherBlock, true, &otherObject, &map, &wire, true) &&
	          rgfxCacheReleaseFlags(t, kOtherBlock, 0, kext) == both,
	      "another surface's release, unheld");
	check(rgfxCacheWritingBack(t, &otherObject) && o->retaken && otherClients == 1,
	      "a new block of it waits for the write-back, the header counting a client");
	otherClients = 3;
	check(rgfxCacheWritingBack(t, &otherObject) && otherClients == 3,
	      "a count the kext wrote stays");

	// No process reads the pages before the answer, and none after it: the entry keeps them wired
	// and the surface held until then, and only the answer hands them back, for the release
	// thread to unwire and let go (the kernel zero-fills pages it gives out again).
	int mapObject = 0, wireObject = 0;
	c->map = &mapObject;
	c->wire = &wireObject;
	bool written = false;
	check(rgfxCacheAtBlock(t, kBlock) == c && rgfxCacheWired(*c) &&
	          rgfxCacheReleasingSurface(t, kBlock, &written) == &surfaceObject && written,
	      "unanswered: the entry holds the surface and its pages, wired");
	void *gotMap = nullptr, *gotWire = nullptr, *gotSurface = nullptr;
	check(rgfxCacheReleased(t, kBlock, &gotMap, &gotWire, &gotSurface) == kSurface &&
	          gotMap == &mapObject && gotWire == &wireObject && gotSurface == &surfaceObject &&
	          rgfxCacheAtBlock(t, kBlock) == nullptr,
	      "the answer hands the mapping, the wire and the surface back to let go");

	// One posted held is taken again uncounted, and its header left as it was.
	RgfxCachedSurface entries2[1] = {};
	RgfxCacheTable h = {entries2, 0, 1};
	RgfxCachedSurface *k = rgfxCacheAdd(h, kSurface, kBlock, kId);
	volatile uint32_t heldClients = 0;
	k->clientsAt = &heldClients;
	check(!rgfxCacheBlockLetGo(h, kSurface, kBlock, true, &surfaceObject, &map, &wire, true) &&
	          rgfxCacheReleaseFlags(h, kBlock, 0, ported) == RGFX_BLOCK_RELEASE_NO_CLIENT &&
	          rgfxCacheNext(h, kSurface, &offset, &mark) == kRgfxCacheWait && k->retaken &&
	          heldClients == 0,
	      "a held release taken again: nothing for the host to read, nothing counted");
}

int main()
{
	askNaming();
	unheldReleases();
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
	check(rgfxCacheLetGo(t, kNotCached, true, surface, &map, &wire, true) && map == nullptr &&
	          wire == nullptr,
	      "a surface not cached lets its cache go with its block");
	check(rgfxCacheReleased(t, kOtherBlock, &map, &wire, &held) == 0 && wire == nullptr &&
	          held == nullptr,
	      "no release asked: nothing answered");
	check(!rgfxCacheLetGo(t, kOther, true, surface, &map, &wire, true) && map == nullptr &&
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
	check(rgfxCacheLetGo(t, kOther, false, surface, &map, &wire, true) && t.count == 0 &&
	          wire == &memory,
	      "a block let go with no release lets the cache, the surface and the wiring go at once");

	dormancy();
	clients();
	room();
	deferral();
	ungated();
	deferredRelease();
	forgottenAttach();
	keptAttachWire();
	attachRetry();
	unanswered();
	otherBlocks();
	writeBack();
	std::printf("surface_caches_test: OK (%d checks)\n", checks);
	return 0;
}
