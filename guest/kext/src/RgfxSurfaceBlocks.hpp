// RgfxSurfaceBlocks: the accelerator kext's shared blocks, one per IOSurface plane the processes
// holding the surface ask by, which they draw and sample on the GPU (the bundle asks by each
// surface's plane 0, the block holding every plane: guest/bundle/src/RgfxIOSurface.h).
//
// A process asks for a plane's block with its own CPU mapping of the plane. The kext wires that
// range in the process's task and reads its physical pages, and the pages are the block's key:
// a process that holds the surface maps the surface's own pages, and one that does not cannot
// present them, so a surface's block goes only to the processes that hold the surface, whatever
// id they claim. Each holder's wired pages stay wired while it holds the block, so the pages
// cannot be reused for another surface under its key.
//
// A block is made in one of two modes, which the asker picks and its entry keeps. A copy block
// holds a copy of the plane, keyed by the plane's pages. An in-place block holds only what the
// holders share past the pixels (the bundle's header and writers), keyed by the pages of the
// surface's whole allocation, which the device's GPU uses in place (SURFACE_PAGES): past the
// part the processes map, the block holds the pages' runs for the device. One surface's blocks
// are all of one mode: an ask in the other mode finds no block while one of the first lives.
//
// The block is the table's (its RgfxBlockTable owner is its entry), and each holding client is
// counted: one hold per client, with a count of its asks. The device learns of each holder's
// context before the process learns of the block (BLOCK_SHARE), and of a holder letting it go
// (BLOCK_UNSHARE). The last holder's let-go frees the block (its BLOCK_RELEASE, which the device
// refuses while an object of any context lies in it), but for a cached block in place
// (RgfxSurfaceCaches.hpp), which stays, dormant, with its host device cache, its runs and its
// header, its surface and the surface's device cache retained and the surface's pages wired:
// the surface's next ask, by the same device cache, holds it again (rgfxSurfaceRevives), and
// the cache's copy in the GPU's memory is used again rather than made and filled anew, as a GPU
// of its own memory keeps a surface's copy past the textures over it. The dormant blocks are
// let go past kRgfxDormantMaxBlocks of them or a byte cap the caller sets (rgfxSurfaceOverCaps),
// those of volatile surfaces first, then the oldest, and for the causes RgfxDormantCause names.
//
// A cached block in place may be deferred (RGFX_SURFACE_PAGES_DEFERRED): its surface had no
// pages at its first ask, a fresh surface only the GPU has touched, and the host's device cache
// alone holds its bytes until a CPU lock names its pages (RgfxSurfaceCaches.hpp). Its entry has
// no runs, so its key is its surface, the IOSurface object (rgfxSurfaceFindDeferred), which an
// ask looks up by the IOSurfaceID it gives in its own task, and its holders hold no wired pages.
// It stays keyed so for its life, its pages named or not. Its bytes count against the device's
// budget for such caches (rgfxSurfaceDeferFits), held and dormant, until it goes, and a dormant
// one's against half of it (rgfxSurfaceOverDeferredCap) rather than the cap of wired pages.
// The caller holds the accelerator's lock around every call. Header-only and free of the
// kernel's headers, so its native test (tests/surface_blocks_test.cpp) builds it as the kext
// does.
//
// Our own code.
#ifndef RGFX_SURFACE_BLOCKS_HPP_
#define RGFX_SURFACE_BLOCKS_HPP_

#include <stdint.h>

// RgfxPageRun, a run of physically contiguous bytes of a plane's pages, is the wire's: an
// in-place block hands its runs to the device as they are.
#include "reims_graphics_wire.h"

// The most clients that hold one plane's block.
constexpr uint32_t kRgfxSurfaceHolders = 16;

// One client's hold of a plane's block.
struct RgfxSurfaceHold {
	const void *client;
	uint32_t context;
	// How many of the client's asks it holds: the hold ends with its last let-go.
	uint32_t refs;
	// The client's wired pages of the plane (a kernel memory descriptor, opaque here), which the
	// caller unwires once the hold ends.
	void *pages;
	// The hold is new and its BLOCK_SHARE not yet posted (rgfxSurfaceSharing).
	bool sharing;
};

// A plane's shared block, where `used`.
struct RgfxSurfaceEntry {
	bool used;
	// The block: its region offset, the bytes its holders map and its RgfxBlockTable serial.
	uint64_t offset;
	uint64_t size;
	uint64_t serial;
	// Whether the device's GPU uses the surface's pages in place, the block holding no copy.
	bool inPlace;
	// An in-place block the device has not yet answered for its pages (SURFACE_PAGES): no other
	// client holds it until the answer, which keeps the block or lets it go.
	bool pending;
	// The key: the plane's bytes and their runs (the caller's memory, which it frees once the
	// entry goes).
	uint64_t length;
	RgfxPageRun *runs;
	uint32_t runCount;
	RgfxSurfaceHold holds[kRgfxSurfaceHolders];
	uint32_t holdCount;
	// The surface's IOSurface and its device cache (RgfxAcceleratorLocks.cpp), opaque here, made
	// at the first write a holder names and kept with the entry (the surface retained); null
	// before. `surfaceId` is the IOSurfaceID they were made for, and `noCache` says they could
	// not be, so no later write tries again.
	void *surface;
	void *cache;
	uint32_t surfaceId;
	bool noCache;
	// The block's GPU copy is the host's device cache (RGFX_SURFACE_PAGES_CACHED): its surface
	// and cache were made with it (RgfxSurfaceCaches.hpp).
	bool cached;
	// A cached block no client holds, kept for its surface (rgfxSurfaceSleep), `surface` and
	// `cache` retained still: the surface's IOSurfaceID, and its cache's address, are no other
	// surface's while the block lasts. `dormantSeq` orders the dormant blocks, the oldest the
	// lowest.
	bool dormant;
	uint64_t dormantSeq;
	// A dormant block whose surface was made volatile (RgfxSurfaceCaches.hpp): the first of
	// them to go.
	bool purgeable;
	// A deferred block (rgfxSurfaceDefer): `surface`, retained, is its key, from its ask on, and
	// `runCount` is 0.
	bool deferred;
};

struct RgfxSurfaceTable {
	RgfxSurfaceEntry *entries;
	uint32_t capacity;
	// The dormant entries: how many, the bytes of their surfaces' pages, all wired, and the last
	// `dormantSeq` given.
	uint32_t dormantCount = 0;
	uint64_t dormantBytes = 0;
	uint64_t dormantSeq = 0;
	// The deferred entries' bytes, held and dormant, and the dormant ones', which dormantBytes
	// does not count: their pages are not wired.
	uint64_t deferredBytes = 0;
	uint64_t dormantDeferredBytes = 0;
};

// The most dormant blocks kept at once.
constexpr uint32_t kRgfxDormantMaxBlocks = 1024;
// The most planes with a shared block at once, held or dormant: 512 held past the dormant ones.
constexpr uint32_t kRgfxSurfaceMaxBlocks = 512 + kRgfxDormantMaxBlocks;
// The most cached surfaces at once (RgfxSurfaceCaches.hpp): one for every block the table holds,
// any of them cached, and as many again whose release is posted and not yet answered, as a
// surface stays cached from its block's let-go until then. Fewer, and dormant blocks, which are
// all cached, would take the states of the surfaces made next. More releases unanswered at once
// than the table holds blocks (a host slow to answer a storm of let-gos) can still fill it: a
// surface made then has its block in place uncached, which is correct, only slower, and the
// kext counts it (the dormancy line's "not cached: no room").
constexpr uint32_t kRgfxCachedMaxSurfaces = 2 * kRgfxSurfaceMaxBlocks;

// Why a dormant block was let go: the kext counts each (RgfxAccelerator::dormantStats).
enum RgfxDormantCause : uint32_t {
	// Its surface's device cache was purged (IOSurface's op 5: a client of the surface went, or
	// the surface was made nonvolatile after empty).
	kRgfxDormantPurged = 0,
	// Its surface was made empty (op 7).
	kRgfxDormantEmptied,
	// Its surface had no client at its sleep: the last went while the block was held, its op 5
	// then letting nothing go.
	kRgfxDormantNoClients,
	// Past kRgfxDormantMaxBlocks dormant blocks, the oldest.
	kRgfxDormantCount,
	// Past the caller's cap of dormant bytes, the oldest.
	kRgfxDormantBytes,
	// The table had no free entry for a new block, the oldest.
	kRgfxDormantTableFull,
	// The region had no room for a new block, the oldest.
	kRgfxDormantNoSpace,
	// An ask of its pages it does not answer: by another device cache, or in the other mode.
	kRgfxDormantStale,
	// A deferred one, past half the device's budget for deferred blocks' caches, the oldest.
	kRgfxDormantDeferredBytes,
	kRgfxDormantCauses,
};

// The guest's page, by which the two modes' keys of one surface meet: a copy's starts at plane
// 0's first byte, which may lie past the start of the surface's first page (CoreVideo places a
// plane 512 bytes in), and one in place at the page's start.
constexpr uint64_t kRgfxSurfacePage = 0x1000;

// Whether a run starting at `a` and one starting at `b` start in the same guest page.
inline bool rgfxSurfaceSamePage(uint64_t a, uint64_t b)
{
	return a / kRgfxSurfacePage == b / kRgfxSurfacePage;
}

// The entry of mode `inPlace` whose key is the `count` runs of `runs` over `length` bytes, its
// block `size` bytes, or -1. A plane whose pages match but whose block size does not is no
// entry's, and nor is one whose first byte lies in the first page of an entry of the other mode:
// the asker gets no block (`*differs`), rather than one laid out apart from the other holders'.
inline int32_t rgfxSurfaceFind(const RgfxSurfaceTable &t, const RgfxPageRun *runs, uint32_t count,
                               uint64_t length, uint64_t size, bool inPlace, bool *differs)
{
	*differs = false;
	for (uint32_t i = 0; i < t.capacity; i++) {
		const RgfxSurfaceEntry &e = t.entries[i];
		if (!e.used)
			continue;
		if (e.inPlace != inPlace) {
			if (count && e.runCount && rgfxSurfaceSamePage(e.runs[0].address, runs[0].address)) {
				*differs = true;
				return -1;
			}
			continue;
		}
		if (e.length != length || e.runCount != count)
			continue;
		bool same = true;
		for (uint32_t r = 0; r < count && same; r++)
			same = e.runs[r].address == runs[r].address && e.runs[r].length == runs[r].length;
		if (!same)
			continue;
		if (e.size != size) {
			*differs = true;
			return -1;
		}
		return static_cast<int32_t>(i);
	}
	return -1;
}

// Whether an entry whose pages start in `runs[0]`'s page, of either mode, waits for the device's
// answer (`pending`): the asker waits for it before it looks for the plane's block.
inline bool rgfxSurfacePending(const RgfxSurfaceTable &t, const RgfxPageRun *runs, uint32_t count)
{
	for (uint32_t i = 0; i < t.capacity && count; i++) {
		const RgfxSurfaceEntry &e = t.entries[i];
		if (e.used && e.pending && e.runCount &&
		    rgfxSurfaceSamePage(e.runs[0].address, runs[0].address))
			return true;
	}
	return false;
}

// The deferred entry of `surface` (an IOSurface, opaque here), pending or not, or -1.
inline int32_t rgfxSurfaceFindDeferred(const RgfxSurfaceTable &t, const void *surface)
{
	for (uint32_t i = 0; i < t.capacity && surface != nullptr; i++) {
		const RgfxSurfaceEntry &e = t.entries[i];
		if (e.used && e.deferred && e.surface == surface)
			return static_cast<int32_t>(i);
	}
	return -1;
}

// The entry of the block at `offset`, or -1.
inline int32_t rgfxSurfaceAt(const RgfxSurfaceTable &t, uint64_t offset)
{
	for (uint32_t i = 0; i < t.capacity; i++) {
		if (t.entries[i].used && t.entries[i].offset == offset)
			return static_cast<int32_t>(i);
	}
	return -1;
}

// An unused entry, or -1 where every one is used.
inline int32_t rgfxSurfaceFreeEntry(const RgfxSurfaceTable &t)
{
	for (uint32_t i = 0; i < t.capacity; i++) {
		if (!t.entries[i].used)
			return static_cast<int32_t>(i);
	}
	return -1;
}

enum RgfxSurfaceHoldResult {
	// The client's first hold: the caller tells the device (BLOCK_SHARE), and keeps `pages`.
	kRgfxSurfaceHoldNew = 0,
	// The client holds it already, one ask more: the caller unwires `pages`.
	kRgfxSurfaceHoldAgain,
	// kRgfxSurfaceHolders clients hold it: no hold; the caller unwires `pages`.
	kRgfxSurfaceHoldFull,
};

// `client` of `context` holds the entry's block, with its wired `pages`.
inline RgfxSurfaceHoldResult rgfxSurfaceHold(RgfxSurfaceEntry &e, const void *client,
                                             uint32_t context, void *pages)
{
	for (uint32_t i = 0; i < e.holdCount; i++) {
		if (e.holds[i].client == client) {
			e.holds[i].refs++;
			return kRgfxSurfaceHoldAgain;
		}
	}
	if (e.holdCount == kRgfxSurfaceHolders)
		return kRgfxSurfaceHoldFull;
	e.holds[e.holdCount++] = {client, context, 1, pages, true};
	return kRgfxSurfaceHoldNew;
}

// Whether `client`'s hold of the entry's block is new, its BLOCK_SHARE not yet posted: the
// client's asks run beside each other (RgfxAskGate.hpp), and another of them waits for the share
// before it holds the block again, so none returns a block the device does not yet let the
// client's context name. A new block's maker posts its first share while the entry is pending,
// which every other ask waits for anyway.
inline bool rgfxSurfaceSharing(const RgfxSurfaceEntry &e, const void *client)
{
	for (uint32_t i = 0; i < e.holdCount; i++) {
		if (e.holds[i].client == client)
			return e.holds[i].sharing;
	}
	return false;
}

// `client`'s new hold of the entry's block had its BLOCK_SHARE posted: its other asks hold the
// block again from here.
inline void rgfxSurfaceShared(RgfxSurfaceEntry &e, const void *client)
{
	for (uint32_t i = 0; i < e.holdCount; i++) {
		if (e.holds[i].client == client)
			e.holds[i].sharing = false;
	}
}

enum RgfxSurfaceLetGo {
	// The client holds no such block: nothing.
	kRgfxSurfaceLetGoNotHeld = 0,
	// One ask fewer; the client holds it still.
	kRgfxSurfaceLetGoRef,
	// The client's hold ended, and other clients hold it: the caller tells the device
	// (BLOCK_UNSHARE) and unwires the pages.
	kRgfxSurfaceLetGoHold,
	// The client was its last holder: the caller frees the block (its BLOCK_RELEASE names
	// `context`), or keeps a cached one dormant (rgfxSurfaceSleep), unwires the hold's pages and
	// lets the entry go (rgfxSurfaceClear). A dormant block's pages stay wired all the same: its
	// device cache holds a wire of its own on the surface's memory until the block's release is
	// answered (rgfxSurfaceCacheAdd).
	kRgfxSurfaceLetGoLast,
};

// `client` lets one ask of the entry's block go, or every one where `all` (its close). Where the
// hold ends, `*pages` is its wired pages and `*context` its context.
inline RgfxSurfaceLetGo rgfxSurfaceLetGo(RgfxSurfaceEntry &e, const void *client, bool all,
                                         void **pages, uint32_t *context)
{
	for (uint32_t i = 0; i < e.holdCount; i++) {
		RgfxSurfaceHold &h = e.holds[i];
		if (h.client != client)
			continue;
		if (!all && h.refs > 1) {
			h.refs--;
			return kRgfxSurfaceLetGoRef;
		}
		*pages = h.pages;
		*context = h.context;
		e.holds[i] = e.holds[--e.holdCount];
		e.holds[e.holdCount] = {};
		return e.holdCount ? kRgfxSurfaceLetGoHold : kRgfxSurfaceLetGoLast;
	}
	return kRgfxSurfaceLetGoNotHeld;
}

// `from`'s close was not answered in time: its holds, each with its context, asks and wired pages,
// are `to`'s, a late close's token (RgfxCloseTable.hpp), until the answer lets them go as an
// answer in time would (rgfxSurfaceLetGo with `all`). How many.
inline uint32_t rgfxSurfaceHand(RgfxSurfaceTable &t, const void *from, const void *to)
{
	uint32_t n = 0;
	for (uint32_t i = 0; from != nullptr && t.entries && i < t.capacity; i++) {
		RgfxSurfaceEntry &e = t.entries[i];
		if (!e.used)
			continue;
		for (uint32_t h = 0; h < e.holdCount; h++) {
			if (e.holds[h].client == from) {
				e.holds[h].client = to;
				n++;
			}
		}
	}
	return n;
}

// Whether `client` holds an entry's block that holds the `length` bytes at `offset`.
inline bool rgfxSurfaceHeldBy(const RgfxSurfaceTable &t, const void *client, uint64_t offset,
                              uint64_t length)
{
	for (uint32_t i = 0; i < t.capacity; i++) {
		const RgfxSurfaceEntry &e = t.entries[i];
		if (!e.used || offset < e.offset || offset - e.offset > e.size ||
		    length > e.size - (offset - e.offset))
			continue;
		for (uint32_t h = 0; h < e.holdCount; h++) {
			if (e.holds[h].client == client)
				return true;
		}
	}
	return false;
}

// The entry's last holder let it go, and its block stays for its surface: dormant, the newest.
inline void rgfxSurfaceSleep(RgfxSurfaceTable &t, RgfxSurfaceEntry &e)
{
	if (e.dormant)
		return;
	e.dormant = true;
	e.dormantSeq = ++t.dormantSeq;
	t.dormantCount++;
	(e.deferred ? t.dormantDeferredBytes : t.dormantBytes) += e.length;
}

// The entry is dormant no more: held again, or let go.
inline void rgfxSurfaceWake(RgfxSurfaceTable &t, RgfxSurfaceEntry &e)
{
	if (!e.dormant)
		return;
	e.dormant = false;
	e.dormantSeq = 0;
	e.purgeable = false;
	t.dormantCount--;
	(e.deferred ? t.dormantDeferredBytes : t.dormantBytes) -= e.length;
}

// The new entry's block is deferred, keyed by its surface (`e.surface`, set by the caller), its
// `length` bytes counted against the budget.
inline void rgfxSurfaceDefer(RgfxSurfaceTable &t, RgfxSurfaceEntry &e)
{
	if (e.deferred)
		return;
	e.deferred = true;
	t.deferredBytes += e.length;
}

// The pending entry is deferred no more: its pages were named at its ask after all (its
// surface was not fresh, or the host refused the deferral), in the `count` runs of `runs`,
// which are its key from here on.
inline void rgfxSurfaceUndefer(RgfxSurfaceTable &t, RgfxSurfaceEntry &e, RgfxPageRun *runs,
                               uint32_t count)
{
	if (e.deferred) {
		e.deferred = false;
		t.deferredBytes -= e.length;
	}
	e.runs = runs;
	e.runCount = count;
}

// How a SURFACE_PAGES that names a block's pages ended, and so what the caller does with them:
// named (the host maps them until the block's release is answered OK), not named (the host
// answered otherwise, or nothing was posted: the caller unwires them), or unanswered (the host
// may yet take and map them: they stay wired, as named ones, until the block's release is
// answered OK, never given back to the guest before).
enum RgfxPagesNaming : uint32_t {
	kRgfxPagesNamed = 0,
	kRgfxPagesNotNamed,
	kRgfxPagesUnanswered,
};

// The naming of a SURFACE_PAGES posted, `answered` or not, OK where `ok`.
inline RgfxPagesNaming rgfxPagesNaming(bool answered, bool ok)
{
	if (!answered)
		return kRgfxPagesUnanswered;
	return ok ? kRgfxPagesNamed : kRgfxPagesNotNamed;
}

// Whether the pages stay wired after a naming: unless the host answered it did not take them.
inline bool rgfxPagesNamingWired(RgfxPagesNaming n)
{
	return n != kRgfxPagesNotNamed;
}

// Whether a new deferred block of `length` bytes fits the device's budget for deferred blocks'
// caches, `budget` bytes (DEFERRED_BUDGET_LO and _HI), beside those of the deferred blocks there
// are: a budget of 0 defers none.
inline bool rgfxSurfaceDeferFits(const RgfxSurfaceTable &t, uint64_t length, uint64_t budget)
{
	return budget != 0 && length <= budget && t.deferredBytes <= budget - length;
}

// The bytes an in-place block of a plane of `length` bytes keeps for its runs past the part its
// holders map: one run a page at most, in whole pages. A deferred block keeps that much, its
// runs unknown until they are named.
inline uint64_t rgfxSurfaceRunRoom(uint64_t length)
{
	const uint64_t pages = (length + kRgfxSurfacePage - 1) / kRgfxSurfacePage;
	const uint64_t bytes = pages * sizeof(RgfxPageRun);
	return (bytes + kRgfxSurfacePage - 1) / kRgfxSurfacePage * kRgfxSurfacePage;
}

// Whether a dormant entry is held again by an ask whose surface's device cache is `cache`, and
// `cacheKept` (nothing asked the block to go since it slept): only the same cache, which only
// the same surface has, the entry holding it.
inline bool rgfxSurfaceRevives(const RgfxSurfaceEntry &e, const void *cache, bool cacheKept)
{
	return e.used && e.dormant && cache != nullptr && e.cache == cache && cacheKept;
}

// Which dormant entries rgfxSurfaceOldestDormant looks at.
enum RgfxDormantKind : uint32_t {
	// Every one.
	kRgfxDormantAny = 0,
	// The deferred ones alone, whose bytes count against the deferred cap.
	kRgfxDormantDeferredOnly,
	// Those not deferred, whose pages are wired: a deferred one's going lets none go.
	kRgfxDormantWiredOnly,
};

// The dormant entry to go first, or -1: the oldest of those whose surfaces are volatile, else
// the oldest; of those of `kind` alone. One still pending, its last holder's BLOCK_UNSHARE not
// yet posted, is not let go, so no BLOCK_RELEASE overtakes that.
inline int32_t rgfxSurfaceOldestDormant(const RgfxSurfaceTable &t,
                                        RgfxDormantKind kind = kRgfxDormantAny)
{
	int32_t oldest = -1;
	for (uint32_t i = 0; i < t.capacity; i++) {
		const RgfxSurfaceEntry &e = t.entries[i];
		if (!e.used || !e.dormant || e.pending ||
		    (kind == kRgfxDormantDeferredOnly && !e.deferred) ||
		    (kind == kRgfxDormantWiredOnly && e.deferred))
			continue;
		const RgfxSurfaceEntry *o = oldest < 0 ? nullptr : &t.entries[oldest];
		if (o == nullptr || (e.purgeable && !o->purgeable) ||
		    (e.purgeable == o->purgeable && e.dormantSeq < o->dormantSeq))
			oldest = static_cast<int32_t>(i);
	}
	return oldest;
}

// The dormant entry to let go while the dormant blocks pass kRgfxDormantMaxBlocks or `maxBytes`
// of wired pages, the first to go (rgfxSurfaceOldestDormant), and why; -1 within both. Past the
// bytes alone, the first of those not deferred: a deferred one's bytes are not wired.
inline int32_t rgfxSurfaceOverCaps(const RgfxSurfaceTable &t, uint64_t maxBytes,
                                   RgfxDormantCause *why)
{
	if (t.dormantCount > kRgfxDormantMaxBlocks) {
		*why = kRgfxDormantCount;
		return rgfxSurfaceOldestDormant(t);
	}
	if (t.dormantBytes <= maxBytes)
		return -1;
	*why = kRgfxDormantBytes;
	return rgfxSurfaceOldestDormant(t, kRgfxDormantWiredOnly);
}

// The dormant deferred entry to let go while their bytes pass `maxBytes`, half the device's
// budget for deferred blocks' caches: the first of them to go; -1 within it.
inline int32_t rgfxSurfaceOverDeferredCap(const RgfxSurfaceTable &t, uint64_t maxBytes,
                                          RgfxDormantCause *why)
{
	if (t.dormantDeferredBytes <= maxBytes)
		return -1;
	*why = kRgfxDormantDeferredBytes;
	return rgfxSurfaceOldestDormant(t, kRgfxDormantDeferredOnly);
}

// What an ask whose new block found no room in the region does next. Before its first wait
// (`deadline` 0), it lets the oldest dormant blocks go and waits for their releases, or gives up
// where there are none. After it, a block that still finds no room waits on until `deadline`,
// whatever else woke it meanwhile (any block's release, any ask's answer), and gives up only
// past it: the releases it asked for are answered once their contexts' buffers finish.
enum RgfxSurfaceRoomStep {
	kRgfxSurfaceRoomGiveUp,
	kRgfxSurfaceRoomLetGoAndWait,
	kRgfxSurfaceRoomWait,
};

inline RgfxSurfaceRoomStep rgfxSurfaceRoomStep(uint64_t deadline, uint32_t dormantCount,
                                               uint64_t now)
{
	if (deadline == 0)
		return dormantCount ? kRgfxSurfaceRoomLetGoAndWait : kRgfxSurfaceRoomGiveUp;
	return now < deadline ? kRgfxSurfaceRoomWait : kRgfxSurfaceRoomGiveUp;
}

// A dormant entry whose pages start in the page of `runs[0]`, of either mode, or -1: one an ask
// it does not answer finds in its way (rgfxSurfaceFind's `*differs`, or a cache not its own).
inline int32_t rgfxSurfaceDormantMeeting(const RgfxSurfaceTable &t, const RgfxPageRun *runs,
                                         uint32_t count)
{
	for (uint32_t i = 0; i < t.capacity && count; i++) {
		const RgfxSurfaceEntry &e = t.entries[i];
		if (e.used && e.dormant && e.runCount &&
		    rgfxSurfaceSamePage(e.runs[0].address, runs[0].address))
			return static_cast<int32_t>(i);
	}
	return -1;
}

// Lets the entry go once its block has none holding it; its runs are the caller's to free. A
// dormant one leaves the table's count of them.
inline void rgfxSurfaceClear(RgfxSurfaceTable &t, RgfxSurfaceEntry &e)
{
	rgfxSurfaceWake(t, e);
	if (e.deferred)
		t.deferredBytes -= e.length;
	e = RgfxSurfaceEntry{};
}

// Appends the physical segment of `length` bytes at `addr`, the next of a plane's in its order,
// to the `*count` runs of `runs` (of `capacity`), merged with the last where they meet. False
// where it does not fit.
inline bool rgfxSurfaceAppendRun(RgfxPageRun *runs, uint32_t *count, uint32_t capacity,
                                 uint64_t addr, uint64_t length)
{
	if (length == 0)
		return true;
	if (*count && runs[*count - 1].address + runs[*count - 1].length == addr) {
		runs[*count - 1].length += length;
		return true;
	}
	if (*count == capacity)
		return false;
	runs[(*count)++] = {addr, length};
	return true;
}

#endif // RGFX_SURFACE_BLOCKS_HPP_
