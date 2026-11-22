// The native test of RgfxSurfaceBlocks.hpp: a plane's block is found by its pages and its mode
// alone, each client holds it once however often it asks, the hold ends with the client's last
// let-go, the block with its last holder's, and only a holder's span is named. A dormant block is
// held again only by its own device cache, live, and the dormant blocks past their caps go oldest
// first, and each once it has slept the age, timed from its sleep, a pending one not until it is no
// longer pending. A deferred block is found by its surface alone, never by pages, counts against
// the device's budget while it lives and while dormant against the smaller of half of it and the
// dormant bytes' cap, an eighth of the guest's memory, and keeps room for one run a page of its
// whole allocation, every plane's. A cached block in place is found by its surface too, laid out as
// the ask, by an ask that maps the surface where it asks, and by no other; a surface's memory is
// its block's runs only page for page. Pages a SURFACE_PAGES named and the host never answered for
// stay wired. A client's new hold makes its other asks wait until its share is posted. A let-go's
// unshare, posted out of the lock, holds back every ask of the block and the block's release until
// it is posted, so the device never ends a hold the kext gave after it. Each reason a copy block is
// refused has its own name, logged once.
#include "RgfxSurfaceBlocks.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "surface_blocks_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

const char kA = 0, kB = 0, kC = 0;
const void *const a = &kA;
const void *const b = &kB;
const void *const c = &kC;
char pagesA, pagesB, pagesC;
char cacheA, cacheB;

// Makes entry `i` of `t` a cached block in place of `length` bytes at `page`, held by no one.
RgfxSurfaceEntry &cachedEntry(RgfxSurfaceTable &t, uint32_t i, RgfxPageRun *run, uint64_t page,
                              uint64_t length, void *cache)
{
	*run = {page, length};
	RgfxSurfaceEntry &e = t.entries[i];
	e = RgfxSurfaceEntry{};
	e.used = true;
	e.offset = 0x100000 + i * 0x2000ull;
	e.size = 0x1000;
	e.inPlace = true;
	e.length = length;
	e.runs = run;
	e.runCount = 1;
	e.cache = cache;
	e.cached = true;
	return e;
}

void dormancy()
{
	RgfxSurfaceEntry entries[3] = {};
	RgfxSurfaceTable t = {entries, 3};
	RgfxPageRun runs[3];
	RgfxSurfaceEntry &e = cachedEntry(t, 0, &runs[0], 0x10000, 0x4000, &cacheA);
	RgfxSurfaceEntry &f = cachedEntry(t, 1, &runs[1], 0x20000, 0x8000, &cacheB);
	check(rgfxSurfaceOldestDormant(t) == -1, "no dormant block yet");

	// The last let-go keeps each dormant, in order, counted with its pages' bytes.
	rgfxSurfaceSleep(t, e, 1);
	rgfxSurfaceSleep(t, f, 1);
	rgfxSurfaceSleep(t, e, 1);
	check(t.dormantCount == 2 && t.dormantBytes == 0xc000, "two dormant, counted once each");
	check(rgfxSurfaceOldestDormant(t) == 0, "the first slept is the oldest");
	bool differs = false;
	RgfxPageRun same = {0x10000, 0x4000};
	check(rgfxSurfaceFind(t, &same, 1, 0x4000, 0x1000, true, &differs) == 0,
	      "a dormant block is found by its pages");

	// Revival: only the block's own device cache, and only while nothing asked it to go.
	check(rgfxSurfaceRevives(e, &cacheA, true), "its own cache holds it again");
	check(!rgfxSurfaceRevives(e, &cacheA, false),
	      "a block asked to go never revives, its own cache's ask included");
	check(!rgfxSurfaceRevives(e, &cacheB, true), "another surface's cache does not");
	check(!rgfxSurfaceRevives(e, nullptr, true), "nor an ask with no cache");
	rgfxSurfaceWake(t, e);
	check(!rgfxSurfaceRevives(e, &cacheA, true), "nor one of a block held");
	check(t.dormantCount == 1 && t.dormantBytes == 0x8000 && rgfxSurfaceOldestDormant(t) == 1,
	      "held again: the other is the oldest");
	rgfxSurfaceSleep(t, e, 1);
	check(rgfxSurfaceOldestDormant(t) == 1 && e.dormantSeq > f.dormantSeq,
	      "dormant again, it is the newest");

	// The ask of another mode, or of another cache, finds the dormant block in its way.
	RgfxPageRun inside = {0x10200, 0x3e00};
	check(rgfxSurfaceFind(t, &inside, 1, 0x3e00, 0x1000, false, &differs) == -1 && differs,
	      "a copy of its plane differs");
	check(rgfxSurfaceDormantMeeting(t, &inside, 1) == 0, "and meets the dormant block");
	RgfxPageRun elsewhere = {0x30000, 0x1000};
	check(rgfxSurfaceDormantMeeting(t, &elsewhere, 1) == -1, "other pages meet none");

	// Teardown: a dormant entry cleared leaves the counts.
	rgfxSurfaceClear(t, f);
	check(t.dormantCount == 1 && t.dormantBytes == 0x4000 && rgfxSurfaceOldestDormant(t) == 0,
	      "the one let go leaves the counts");
	rgfxSurfaceClear(t, e);
	check(t.dormantCount == 0 && t.dormantBytes == 0 && rgfxSurfaceOldestDormant(t) == -1,
	      "none dormant");

	// The caps let the oldest go first: the bytes past the caller's cap, the count past
	// kRgfxDormantMaxBlocks.
	static RgfxSurfaceEntry many[kRgfxDormantMaxBlocks + 2];
	static RgfxPageRun manyRuns[kRgfxDormantMaxBlocks + 2];
	RgfxSurfaceTable m = {many, kRgfxDormantMaxBlocks + 2};
	RgfxDormantCause why = kRgfxDormantCauses;
	for (uint32_t i = 0; i < 3; i++)
		rgfxSurfaceSleep(m, cachedEntry(m, i, &manyRuns[i], 0x100000 * (i + 1), 0x1000, &cacheA),
		                 1);
	check(rgfxSurfaceOverCaps(m, 0x3000, &why) == -1, "within the caps");
	check(rgfxSurfaceOverCaps(m, 0x2fff, &why) == 0 && why == kRgfxDormantBytes,
	      "past the bytes: the oldest");
	rgfxSurfaceClear(m, many[0]);
	check(rgfxSurfaceOverCaps(m, 0x1fff, &why) == 1 && why == kRgfxDormantBytes,
	      "then the next oldest");
	for (uint32_t i = 3; i < kRgfxDormantMaxBlocks + 2; i++)
		rgfxSurfaceSleep(m, cachedEntry(m, i, &manyRuns[i], 0x100000 * (i + 1), 0x1000, &cacheA),
		                 1);
	check(m.dormantCount == kRgfxDormantMaxBlocks + 1, "one past the count");
	check(rgfxSurfaceOverCaps(m, ~0ull, &why) == 1 && why == kRgfxDormantCount,
	      "past the count: the oldest");
	rgfxSurfaceClear(m, many[1]);
	check(rgfxSurfaceOverCaps(m, ~0ull, &why) == -1, "at the count: none");

	// A dormant block whose surface is volatile goes first, the oldest of those first; one held
	// again is no longer marked.
	many[700].purgeable = true;
	many[9].purgeable = true;
	check(rgfxSurfaceOldestDormant(m) == 9, "the oldest volatile one goes first");
	check(rgfxSurfaceOverCaps(m, 0x1000 * 4, &why) == 9 && why == kRgfxDormantBytes,
	      "past a cap too");
	rgfxSurfaceWake(m, many[9]);
	check(!many[9].purgeable && rgfxSurfaceOldestDormant(m) == 700,
	      "held again, it is not; the next volatile one");
	rgfxSurfaceClear(m, many[700]);
	check(rgfxSurfaceOldestDormant(m) == 2, "none volatile: the oldest");
	many[2].pending = true;
	check(rgfxSurfaceOldestDormant(m) == 3,
	      "one whose BLOCK_UNSHARE is not yet posted is passed over");
	many[2].pending = false;
}
} // namespace

// Deferred blocks (RGFX_SURFACE_PAGES_DEFERRED): keyed by their surface, budgeted, capped.
void deferred()
{
	RgfxSurfaceEntry entries[4] = {};
	RgfxSurfaceTable t = {entries, 4};
	RgfxPageRun runs[2];
	// A two-plane surface's allocation (NV12 at 1920x1080: 2073600 bytes of luma and 1036800 of
	// chroma, page aligned), and a 4K BGRA one's.
	constexpr uint64_t kTwoPlanes = 3112960, kBgra4k = 3840ull * 2160 * 4;
	constexpr uint64_t kBudget = 64ull << 20;
	char surfaceA = 0, surfaceB = 0;
	check(rgfxSurfaceDeferFits(t, kTwoPlanes, kBudget), "a budget with room defers");
	check(!rgfxSurfaceDeferFits(t, kTwoPlanes, 0), "a budget of 0 defers none");
	check(!rgfxSurfaceDeferFits(t, kBudget + 1, kBudget), "nor a block past the budget");

	// Made deferred: no runs, its surface its key, its bytes counted.
	RgfxSurfaceEntry &d = entries[0];
	d.used = true;
	d.offset = 0x200000;
	d.size = 0x1000;
	d.inPlace = d.pending = d.cached = true;
	d.length = kTwoPlanes;
	d.surface = &surfaceA;
	rgfxSurfaceDefer(t, d);
	rgfxSurfaceDefer(t, d);
	check(d.deferred && d.runCount == 0 && t.deferredBytes == kTwoPlanes,
	      "a deferred block counts its whole allocation once");
	check(rgfxSurfaceFindDeferred(t, &surfaceA) == 0, "found by its surface, pending or not");
	check(rgfxSurfaceFindDeferred(t, &surfaceB) == -1 && rgfxSurfaceFindDeferred(t, nullptr) == -1,
	      "and by no other");
	RgfxPageRun page = {0x40000, kTwoPlanes};
	bool differs = true;
	check(rgfxSurfaceFind(t, &page, 1, kTwoPlanes, 0x1000, true, &differs) == -1 && !differs,
	      "never by pages");
	check(!rgfxSurfacePending(t, &page, 1) && rgfxSurfaceDormantMeeting(t, &page, 1) == -1,
	      "nor waited for or met by an ask of pages");
	check(!rgfxSurfaceDeferFits(t, kBudget - kTwoPlanes + 1, kBudget) &&
	          rgfxSurfaceDeferFits(t, kBudget - kTwoPlanes, kBudget),
	      "the budget counts the blocks deferred");

	// Its runs' room: one run a page of the whole allocation, whole pages.
	check(rgfxSurfaceRunRoom(kTwoPlanes) == 3 * kRgfxSurfacePage,
	      "760 pages of two planes keep 760 runs' room, 3 pages");
	check(rgfxSurfaceRunRoom(kBgra4k) == 32 * kRgfxSurfacePage,
	      "a 4K surface's 8100 runs, 32 pages");
	check(rgfxSurfaceRunRoom(kRgfxSurfacePage) == kRgfxSurfacePage, "one page's run, a page");

	// Dormant: counted against the deferred cap, not the wired pages' bytes.
	d.pending = false;
	rgfxSurfaceSleep(t, d, 1);
	check(t.dormantCount == 1 && t.dormantBytes == 0 && t.dormantDeferredBytes == kTwoPlanes,
	      "a dormant deferred block's bytes are not wired");
	RgfxDormantCause why = kRgfxDormantCauses;
	check(rgfxSurfaceOverCaps(t, 0, &why) == -1, "the wired pages' cap passes it by");
	check(rgfxSurfaceOverDeferredCap(t, kTwoPlanes, &why) == -1, "within the deferred cap");

	// An older dormant block not deferred, and a newer deferred one: past the deferred cap the
	// oldest deferred goes, never the other.
	RgfxSurfaceEntry &old = cachedEntry(t, 1, &runs[0], 0x10000, 0x4000, &cacheA);
	rgfxSurfaceSleep(t, old, 1);
	old.dormantSeq = 0;
	RgfxSurfaceEntry &e = entries[2];
	e = d;
	e.offset = 0x300000;
	e.length = kBgra4k;
	e.surface = &surfaceB;
	e.deferred = e.dormant = false;
	rgfxSurfaceDefer(t, e);
	rgfxSurfaceSleep(t, e, 1);
	check(t.deferredBytes == kTwoPlanes + kBgra4k &&
	          t.dormantDeferredBytes == kTwoPlanes + kBgra4k && t.dormantBytes == 0x4000,
	      "each counted where it belongs");
	check(rgfxSurfaceOldestDormant(t) == 1 &&
	          rgfxSurfaceOldestDormant(t, kRgfxDormantDeferredOnly) == 0 &&
	          rgfxSurfaceOldestDormant(t, kRgfxDormantWiredOnly) == 1,
	      "the oldest of all, and of the deferred");
	check(rgfxSurfaceOverDeferredCap(t, kBgra4k, &why) == 0 && why == kRgfxDormantDeferredBytes,
	      "past half the budget, the oldest deferred goes");
	check(rgfxSurfaceOverCaps(t, 0x4000, &why) == -1, "the wired cap still holds");
	// The deferred ones older, or volatile: past the wired pages' cap the block not deferred
	// goes, the only one whose going lets wired pages go, never a deferred one first.
	old.dormantSeq = t.dormantSeq + 1;
	check(rgfxSurfaceOverCaps(t, 0x3fff, &why) == 1 && why == kRgfxDormantBytes,
	      "past the wired cap, the oldest wired block, not an older deferred one");
	d.purgeable = true;
	check(rgfxSurfaceOverCaps(t, 0x3fff, &why) == 1 && why == kRgfxDormantBytes,
	      "nor a volatile deferred one");
	d.purgeable = false;
	old.dormantSeq = 0;
	rgfxSurfaceClear(t, d);
	check(t.deferredBytes == kBgra4k && t.dormantDeferredBytes == kBgra4k && t.dormantCount == 2,
	      "a deferred block let go leaves both counts");
	check(rgfxSurfaceOverDeferredCap(t, kBgra4k, &why) == -1, "and the cap holds again");

	// Held again: its bytes leave the dormant count, not the budget's.
	rgfxSurfaceWake(t, e);
	check(t.deferredBytes == kBgra4k && t.dormantDeferredBytes == 0 && t.dormantCount == 1,
	      "a deferred block held again");

	// Undeferred at its ask (not fresh, or refused): keyed by its runs from then on.
	RgfxSurfaceEntry &u = entries[3];
	u = RgfxSurfaceEntry{};
	u.used = u.inPlace = u.pending = true;
	u.offset = 0x400000;
	u.size = 0x1000;
	u.length = 0x2000;
	u.surface = &surfaceA;
	rgfxSurfaceDefer(t, u);
	runs[1] = {0x80000, 0x2000};
	rgfxSurfaceUndefer(t, u, &runs[1], 1);
	check(!u.deferred && u.runCount == 1 && t.deferredBytes == kBgra4k,
	      "its bytes leave the budget");
	check(rgfxSurfaceFindDeferred(t, &surfaceA) == -1 &&
	          rgfxSurfaceFind(t, &runs[1], 1, 0x2000, 0x1000, true, &differs) == 3,
	      "found by its pages, not its surface");
	rgfxSurfaceClear(t, u);
	check(t.deferredBytes == kBgra4k, "and its let-go counts nothing twice");
}

// The dormant deferred blocks' cap is the smaller of half the device's budget for deferred
// blocks' caches and the dormant bytes' cap, an eighth of the guest's memory, and past it the
// oldest dormant deferred block goes, though half the budget holds them all.
void dormantDeferredCap()
{
	// A host GPU's budget of 10 GiB, half of it 5 GiB, and an 8 GiB guest's cap of 1 GiB.
	constexpr uint64_t kBudget = 10ull << 30, kGuestCap = (8ull << 30) / 8;
	check(rgfxSurfaceDormantDeferredCap(kBudget, kGuestCap) == kGuestCap,
	      "the guest's cap, where it is the smaller");
	check(rgfxSurfaceDormantDeferredCap(kBudget, kBudget) == kBudget / 2,
	      "half the budget, where it is the smaller");
	check(rgfxSurfaceDormantDeferredCap(0, kGuestCap) == 0, "a budget of 0 keeps none");

	// Three dormant deferred blocks of 400 MiB, 1200 MiB in all: within half the budget, past
	// the guest's cap.
	RgfxSurfaceEntry entries[3] = {};
	RgfxSurfaceTable t = {entries, 3};
	char surfaces[3] = {};
	constexpr uint64_t kLength = 400ull << 20;
	for (uint32_t i = 0; i < 3; i++) {
		RgfxSurfaceEntry &e = entries[i];
		e.used = e.inPlace = e.cached = true;
		e.offset = 0x200000 * (i + 1);
		e.size = 0x1000;
		e.length = kLength;
		e.surface = &surfaces[i];
		rgfxSurfaceDefer(t, e);
		rgfxSurfaceSleep(t, e, 1);
	}
	check(t.dormantDeferredBytes == 3 * kLength && t.dormantBytes == 0,
	      "three dormant deferred blocks");
	RgfxDormantCause why = kRgfxDormantCauses;
	check(rgfxSurfaceOverDeferredCap(t, kBudget / 2, &why) == -1, "within half the budget");
	const uint64_t cap = rgfxSurfaceDormantDeferredCap(kBudget, kGuestCap);
	check(rgfxSurfaceOverDeferredCap(t, cap, &why) == 0 && why == kRgfxDormantDeferredBytes,
	      "past the guest's cap, the oldest goes");
	rgfxSurfaceClear(t, entries[0]);
	check(rgfxSurfaceOverDeferredCap(t, cap, &why) == -1 && t.dormantDeferredBytes == 2 * kLength,
	      "then the cap holds the other two");
}

// A SURFACE_PAGES naming a block's pages: named, refused, or never answered, where the host may
// yet take them and they stay wired.
// A dormant block ages from its sleep: it goes once it has slept the age, and the next to age is
// the earliest of those not yet aged, not pending and not deferred; a block held again ages from
// its next sleep, and a deferred one never ages.
void aging()
{
	RgfxSurfaceEntry entries[3] = {};
	RgfxSurfaceTable t = {entries, 3};
	RgfxPageRun runs[3];
	const uint64_t age = 1000;
	RgfxSurfaceEntry &e = cachedEntry(t, 0, &runs[0], 0x10000, 0x4000, &cacheA);
	RgfxSurfaceEntry &f = cachedEntry(t, 1, &runs[1], 0x20000, 0x8000, &cacheB);
	RgfxSurfaceEntry &g = cachedEntry(t, 2, &runs[2], 0x30000, 0x1000, &cacheB);
	uint64_t at = 0;
	check(!rgfxSurfaceNextAge(t, 0, age, &at), "nothing dormant, nothing ages");
	check(!rgfxSurfaceAged(e, 5000, age), "a held block never ages");
	rgfxSurfaceSleep(t, e, 100);
	rgfxSurfaceSleep(t, f, 300);
	check(e.sleptAt == 100 && f.sleptAt == 300, "each timed from its sleep");
	check(!rgfxSurfaceAged(e, 1099, age) && rgfxSurfaceAged(e, 1100, age),
	      "a block goes once it has slept the age");
	check(rgfxSurfaceNextAge(t, 1050, age, &at) && at == 1100, "the first to age is the oldest");
	check(rgfxSurfaceNextAge(t, 1100, age, &at) && at == 1300,
	      "past it, the next not yet aged: an aged one waits on no timer");
	check(!rgfxSurfaceNextAge(t, 1300, age, &at), "both aged: none is next");
	// One pending (its last holder's BLOCK_UNSHARE not yet posted) neither ages nor is next.
	rgfxSurfaceSleep(t, g, 200);
	g.pending = true;
	check(!rgfxSurfaceAged(g, 5000, age), "a pending block does not age");
	check(rgfxSurfaceNextAge(t, 1050, age, &at) && at == 1100, "nor is it next");
	g.pending = false;
	check(rgfxSurfaceNextAge(t, 1150, age, &at) && at == 1200, "it is once posted");
	// A deferred one neither ages nor is next: it goes past its caps alone.
	g.deferred = true;
	check(!rgfxSurfaceAged(g, 5000, age), "a deferred block does not age");
	check(rgfxSurfaceNextAge(t, 1150, age, &at) && at == 1300, "nor is it next");
	g.deferred = false;
	// Held again, and asleep again later: timed from the new sleep.
	rgfxSurfaceWake(t, e);
	check(e.sleptAt == 0 && !rgfxSurfaceAged(e, 5000, age), "held again, it does not age");
	rgfxSurfaceSleep(t, e, 2000);
	check(!rgfxSurfaceAged(e, 2999, age) && rgfxSurfaceAged(e, 3000, age),
	      "asleep again, it ages from its new sleep");
	// Before the clock reaches the age, nothing has slept that long, a sleep at 0 included.
	rgfxSurfaceClear(t, f);
	check(t.dormantCount == 2, "a let-go entry is dormant no more");
	RgfxSurfaceEntry &h = cachedEntry(t, 1, &runs[1], 0x20000, 0x8000, &cacheB);
	rgfxSurfaceSleep(t, h, 0);
	check(!rgfxSurfaceAged(h, 999, age) && rgfxSurfaceAged(h, 1000, age),
	      "a sleep at the clock's 0 ages at the age");
	check(kRgfxDormantAged < kRgfxDormantCauses, "the age is a cause of its own");
}

void naming()
{
	check(rgfxPagesNaming(true, true) == kRgfxPagesNamed && rgfxPagesNamingWired(kRgfxPagesNamed),
	      "answered OK: named, wired");
	check(rgfxPagesNaming(true, false) == kRgfxPagesNotNamed &&
	          !rgfxPagesNamingWired(kRgfxPagesNotNamed),
	      "answered otherwise: not named, unwired");
	check(rgfxPagesNaming(false, false) == kRgfxPagesUnanswered &&
	          rgfxPagesNaming(false, true) == kRgfxPagesUnanswered &&
	          rgfxPagesNamingWired(kRgfxPagesUnanswered),
	      "never answered: kept wired, the host free to take them yet");
}

// A client's asks run beside each other (RgfxAskGate.hpp): a new hold's BLOCK_SHARE is posted
// out of the lock, and another ask of the same client waits for it before it holds the block
// again, so no ask returns a block the device does not yet let the client's context name.
void sharing()
{
	int a = 0, b = 0, pagesA = 0, pagesB = 0;
	RgfxSurfaceEntry e = {};
	e.used = true;
	check(!rgfxSurfaceSharing(e, &a), "a block no one holds is no one's to share");
	check(rgfxSurfaceHold(e, &a, 1, &pagesA) == kRgfxSurfaceHoldNew && rgfxSurfaceSharing(e, &a),
	      "a's new hold: its share not yet posted, a's other asks wait for it");
	check(!rgfxSurfaceSharing(e, &b), "another client's asks do not");
	check(rgfxSurfaceHold(e, &b, 2, &pagesB) == kRgfxSurfaceHoldNew && rgfxSurfaceSharing(e, &b),
	      "b's own new hold shares on its own");
	rgfxSurfaceShared(e, &a);
	check(!rgfxSurfaceSharing(e, &a) && rgfxSurfaceSharing(e, &b),
	      "a's share posted: a's asks hold it again, b's still wait for b's");
	check(rgfxSurfaceHold(e, &a, 1, &pagesA) == kRgfxSurfaceHoldAgain && !rgfxSurfaceSharing(e, &a),
	      "a's next ask holds it again, sharing nothing");
	void *pages = nullptr;
	uint32_t context = 0;
	check(rgfxSurfaceLetGo(e, &b, true, &pages, &context) == kRgfxSurfaceLetGoHold &&
	          !rgfxSurfaceSharing(e, &b),
	      "b's share failed and its hold went: b's other asks find the block no longer b's");
}

// The device's view of the shared blocks, as the host carries out ring 0's messages, in ring
// order (crates/reims-vgpu-graphics/src/region.rs). A share adds its context to the block's
// holders, making the block where none is, and is nothing for a context that holds it already.
// An unshare ends the context's hold, the block going with its last holder. A release ends every
// hold of the block. A message that names the block is refused for a context not among its
// holders. Contexts are below 32, a bit each.
struct Device {
	uint64_t offsets[4] = {};
	uint32_t holders[4] = {};
	int at(uint64_t offset) const
	{
		for (int i = 0; i < 4; i++) {
			if (holders[i] && offsets[i] == offset)
				return i;
		}
		return -1;
	}
	void share(uint32_t context, uint64_t offset)
	{
		int i = at(offset);
		for (int j = 0; i < 0 && j < 4; j++) {
			if (!holders[j]) {
				i = j;
				offsets[i] = offset;
			}
		}
		holders[i] |= 1u << context;
	}
	void unshare(uint32_t context, uint64_t offset)
	{
		if (int i = at(offset); i >= 0)
			holders[i] &= ~(1u << context);
	}
	void release(uint64_t offset)
	{
		if (int i = at(offset); i >= 0)
			holders[i] = 0;
	}
	bool holds(uint32_t context, uint64_t offset) const
	{
		int i = at(offset);
		return i >= 0 && (holders[i] & (1u << context)) != 0;
	}
};

// A shared block held by clients `a` of context 1 and `b` of context 2, each share posted, in a
// table of four blocks: its entry, and the device's view.
struct TwoHolders {
	RgfxBlock blocks[4] = {};
	RgfxBlockTable table = {blocks, 0, 4, 0x100000, 0x200000, 0, 0, 0, 0, 0, 0, 0};
	RgfxSurfaceEntry e = {};
	Device device;
	int pagesA = 0, pagesB = 0;
	TwoHolders(const void *a, const void *b)
	{
		uint64_t offset = 0;
		check(rgfxBlockAlloc(table, &e, 1, 0x1000, 0x1000, &offset) == kRgfxBlockOk,
		      "the shared block is made");
		e.used = true;
		e.offset = offset;
		e.size = 0x1000;
		e.serial = blocks[0].serial;
		check(rgfxSurfaceHold(e, a, 1, &pagesA) == kRgfxSurfaceHoldNew, "a's first hold");
		device.share(1, offset);
		rgfxSurfaceShared(e, a);
		check(rgfxSurfaceHold(e, b, 2, &pagesB) == kRgfxSurfaceHoldNew, "b's first hold");
		device.share(2, offset);
		rgfxSurfaceShared(e, b);
	}
};

// A let-go whose block other clients still hold ends the hold under the lock and posts its
// BLOCK_UNSHARE out of it (releaseSurfaceBlock). Until the unshare is posted, every ask of the
// block waits (rgfxSurfaceAskWaits), and so does the block's release (rgfxBlockNextRelease). The
// device carries out a context's share and unshare in ring order, so an unshare posted after
// that context's next share of the block, or of a block given the span again after the release,
// ends the new hold. The device then refuses the context's next message that names it (a
// BUFFER_OVER_SURFACE_PAGES, so the TEXTURE_FROM_IOSURFACE over it too), though the kext gave the
// hold.
void unshareOrder()
{
	int a = 0, b = 0;
	void *pages = nullptr;
	uint32_t context = 0;
	{
		// The client lets go and asks again before its unshare is posted.
		TwoHolders x(&a, &b);
		const uint64_t offset = x.e.offset;
		check(rgfxSurfaceLetGo(x.e, &a, false, &pages, &context) == kRgfxSurfaceLetGoHold &&
		          context == 1,
		      "a lets go, b holds on: a's unshare is to post");
		RgfxBlockPin mark = {};
		check(rgfxBlockBeginUnshare(x.table, offset, &mark), "the unshare is marked on the block");
		check(rgfxSurfaceAskWaits(x.e, x.table, &a),
		      "a's next ask of the block waits for a's unshare");
		check(rgfxSurfaceAskWaits(x.e, x.table, &b), "and so does every other ask of it");
		x.device.unshare(1, offset);
		check(!rgfxBlockEndUnshare(x.table, mark), "the unshare posted; b holds it: no release");
		check(!rgfxSurfaceAskWaits(x.e, x.table, &a), "a's ask goes on");
		check(rgfxSurfaceHold(x.e, &a, 1, &x.pagesA) == kRgfxSurfaceHoldNew,
		      "a holds the block anew");
		x.device.share(1, offset);
		rgfxSurfaceShared(x.e, &a);
		check(x.device.holds(1, offset) && rgfxSurfaceHeldBy({&x.e, 1}, &a, offset, 1),
		      "the device lets a name the block the kext gave a");
		// The order the wait keeps out: the new hold's share ahead of the old hold's unshare.
		Device ahead;
		ahead.share(1, offset);
		ahead.share(2, offset);
		ahead.share(1, offset);
		ahead.unshare(1, offset);
		check(!ahead.holds(1, offset), "a share ahead of the unshare: the unshare ends the hold");
	}
	{
		// The other holder lets go last, and the span is given out again, before the unshare is
		// posted.
		TwoHolders x(&a, &b);
		const uint64_t offset = x.e.offset;
		check(rgfxSurfaceLetGo(x.e, &a, false, &pages, &context) == kRgfxSurfaceLetGoHold,
		      "a lets go, b holds on");
		RgfxBlockPin mark = {};
		check(rgfxBlockBeginUnshare(x.table, offset, &mark), "a's unshare is marked");
		check(rgfxSurfaceLetGo(x.e, &b, false, &pages, &context) == kRgfxSurfaceLetGoLast &&
		          context == 2,
		      "b lets go last");
		check(rgfxBlockFree(x.table, &x.e, offset) == kRgfxBlockOk,
		      "the block is freed for its release (letGoBlock)");
		RgfxBlock next = {};
		check(!rgfxBlockNextRelease(x.table, &next),
		      "its release is not posted ahead of a's unshare");
		check(rgfxBlockUnsharing(x.table, offset), "the block still waits for it");
		x.device.unshare(1, offset);
		check(rgfxBlockEndUnshare(x.table, mark),
		      "a's unshare posted: the release is due, and its poster runs the release worker");
		check(rgfxBlockNextRelease(x.table, &next) && next.offset == offset,
		      "the release is posted");
		x.device.release(offset);
		rgfxBlockReleased(x.table, offset, next.serial, true);
		RgfxSurfaceEntry f = {};
		uint64_t again = 0;
		check(rgfxBlockAlloc(x.table, &f, 1, 0x1000, 0x1000, &again) == kRgfxBlockOk &&
		          again == offset,
		      "the span is given out again, to a new block");
		f.used = true;
		f.offset = again;
		f.size = 0x1000;
		check(!rgfxSurfaceAskWaits(f, x.table, &a), "an ask of the new block waits for nothing");
		check(!rgfxBlockEndUnshare(x.table, mark), "the old block's mark finds nothing there");
		check(rgfxSurfaceHold(f, &a, 1, &x.pagesA) == kRgfxSurfaceHoldNew, "a holds the new block");
		x.device.share(1, again);
		check(x.device.holds(1, again), "the device lets a name it");
		// The order the release's wait keeps out: the release, the new block's share, then the
		// old hold's unshare.
		Device late;
		late.share(1, offset);
		late.share(2, offset);
		late.release(offset);
		late.share(1, again);
		late.unshare(1, offset);
		check(!late.holds(1, again), "a late unshare ends the new block's hold");
	}
}

// A cached block in place is found by its surface too, held or dormant, pending or not, so an ask
// of the surface that maps it where it asks wires no pages of its own: only one cached in place,
// not deferred, keyed by its surface's pages from their first byte, laid out as the ask, of that
// surface.
void cachedBySurface()
{
	RgfxSurfaceEntry entries[4] = {};
	RgfxSurfaceTable t = {entries, 4};
	RgfxPageRun runs[4];
	char surfaceA = 0, surfaceB = 0;
	constexpr uint64_t kLength = 0x4000;
	RgfxSurfaceEntry &e = cachedEntry(t, 0, &runs[0], 0x10000, kLength, &cacheA);
	e.surface = &surfaceA;
	e.surfaceStart = true;
	check(rgfxSurfaceFindCached(t, &surfaceA, kLength, e.size, true) == 0,
	      "a held cached block is found by its surface");
	e.pending = true;
	check(rgfxSurfaceFindCached(t, &surfaceA, kLength, e.size, true) == 0,
	      "pending too: the ask waits for it");
	e.pending = false;
	rgfxSurfaceSleep(t, e, 1);
	check(rgfxSurfaceFindCached(t, &surfaceA, kLength, e.size, true) == 0, "and dormant");
	check(rgfxSurfaceFindCached(t, &surfaceB, kLength, e.size, true) == -1 &&
	          rgfxSurfaceFindCached(t, nullptr, kLength, e.size, true) == -1,
	      "another surface's ask does not find it");
	check(rgfxSurfaceFindCached(t, &surfaceA, kLength + 0x1000, e.size, true) == -1 &&
	          rgfxSurfaceFindCached(t, &surfaceA, kLength, e.size + 0x1000, true) == -1,
	      "nor an ask laid out otherwise, which wires its pages");
	rgfxSurfaceWake(t, e);
	e.surfaceStart = false;
	check(rgfxSurfaceFindCached(t, &surfaceA, kLength, e.size, true) == -1,
	      "nor a block keyed by pages past its surface's first: the ask's pages find what they "
	      "find");
	e.surfaceStart = true;
	e.cached = false;
	check(rgfxSurfaceFindCached(t, &surfaceA, kLength, e.size, true) == -1,
	      "an uncached block is not: its holders wire the pages it is keyed by");
	e.cached = true;
	e.inPlace = false;
	check(rgfxSurfaceFindCached(t, &surfaceA, kLength, e.size, true) == -1, "nor a copy");
	e.inPlace = true;
	rgfxSurfaceDefer(t, e);
	check(rgfxSurfaceFindCached(t, &surfaceA, kLength, e.size, true) == -1 &&
	          rgfxSurfaceFindDeferred(t, &surfaceA) == 0,
	      "nor a deferred one, which rgfxSurfaceFindDeferred finds");
	e.used = false;
	check(rgfxSurfaceFindCached(t, &surfaceA, kLength, e.size, true) == -1, "nor an entry gone");
	RgfxSurfaceEntry &f = cachedEntry(t, 2, &runs[2], 0x20000, kLength, &cacheB);
	f.surface = &surfaceB;
	f.surfaceStart = true;
	check(rgfxSurfaceFindCached(t, &surfaceB, kLength, f.size, true) == 2, "another surface's own");
}

// An ask whose address does not map its surface finds no block by the surface, whether or not
// the surface has a cached block: it wires its range as any ask does, which the wire refuses
// (BadArgument) where its task does not map it, and finds by its pages what they key.
void askNotMapping()
{
	RgfxSurfaceEntry entries[2] = {};
	RgfxSurfaceTable t = {entries, 2};
	RgfxPageRun run;
	char surface = 0;
	constexpr uint64_t kLength = 0x4000;
	check(rgfxSurfaceFindCached(t, &surface, kLength, 0x1000, false) == -1 &&
	          rgfxSurfaceFindCached(t, &surface, kLength, 0x1000, true) == -1,
	      "a surface with no cached block: none either way");
	RgfxSurfaceEntry &e = cachedEntry(t, 0, &run, 0x10000, kLength, &cacheA);
	e.surface = &surface;
	e.surfaceStart = true;
	check(rgfxSurfaceFindCached(t, &surface, kLength, e.size, false) == -1,
	      "a cached one: an ask not mapping the surface where it asks finds it not");
	rgfxSurfaceSleep(t, e, 1);
	check(rgfxSurfaceFindCached(t, &surface, kLength, e.size, false) == -1,
	      "nor dormant: no revive by a surface the ask does not show it maps");
	check(rgfxSurfaceFindCached(t, &surface, kLength, e.size, true) == 0,
	      "the same ask mapping it finds it");
}

// A surface's memory walked by its physical segments against a block's runs, as
// rgfxSurfaceAppendRun merged them: the same pages in the same order, each run whole, no more.
void runsWalk()
{
	const RgfxPageRun runs[2] = {{0x10000, 0x3000}, {0x40000, 0x1000}};
	auto walk = [&runs](std::initializer_list<RgfxPageRun> segments, uint32_t count = 2) {
		RgfxRunsWalk w = rgfxSurfaceRunsWalk(runs, count);
		for (const RgfxPageRun &s : segments)
			rgfxSurfaceRunsNext(w, s.address, s.length);
		return rgfxSurfaceRunsSame(w);
	};
	check(walk({{0x10000, 0x1000}, {0x11000, 0x2000}, {0x40000, 0x1000}}),
	      "segments that merge into the runs are them");
	check(walk({{0x10000, 0x3000}, {0x40000, 0x1000}}), "and so are the runs themselves");
	check(!walk({{0x10000, 0x3000}}), "a walk short of the last run is not");
	check(!walk({{0x10000, 0x2000}, {0x40000, 0x1000}}), "nor one that leaves a run short");
	check(!walk({{0x11000, 0x2000}, {0x40000, 0x1000}}), "nor one that starts past the first page");
	check(!walk({{0x10000, 0x3000}, {0x40000, 0x2000}}), "nor one past a run's end");
	check(!walk({{0x10000, 0x3000}, {0x40000, 0x1000}, {0x41000, 0x1000}}),
	      "nor one with more pages than the runs");
	check(!walk({{0x10000, 0x3000}, {0x50000, 0x1000}}), "nor other pages");
	check(!walk({{0x40000, 0x1000}, {0x10000, 0x3000}}), "nor the pages in another order");
	check(!walk({{0x10000, 0x3000}}, 0), "nor any walk of no runs");
}

int main()
{
	naming();
	aging();
	deferred();
	dormantDeferredCap();
	cachedBySurface();
	askNotMapping();
	runsWalk();
	dormancy();
	sharing();
	unshareOrder();
	RgfxSurfaceEntry entries[4] = {};
	RgfxSurfaceTable t = {entries, 4};
	RgfxPageRun plane[2] = {{0x10000, 0x3000}, {0x40000, 0x1000}};
	bool differs = false;
	check(rgfxSurfaceFind(t, plane, 2, 0x4000, 0x5000, false, &differs) == -1 && !differs,
	      "an empty table holds no plane");

	// The first ask makes the entry.
	int32_t at = rgfxSurfaceFreeEntry(t);
	check(at == 0, "a free entry");
	RgfxSurfaceEntry &e = entries[at];
	e.used = true;
	e.offset = 0x100000;
	e.size = 0x5000;
	e.serial = 7;
	e.length = 0x4000;
	e.runs = plane;
	e.runCount = 2;
	check(rgfxSurfaceHold(e, a, 1, &pagesA) == kRgfxSurfaceHoldNew, "a's first hold");
	check(rgfxSurfaceFreeEntry(t) == 1, "the next free entry");

	// Another process's same pages find it; other pages, or a part of them, do not.
	RgfxPageRun same[2] = {{0x10000, 0x3000}, {0x40000, 0x1000}};
	check(rgfxSurfaceFind(t, same, 2, 0x4000, 0x5000, false, &differs) == 0, "the same pages");
	RgfxPageRun other[2] = {{0x10000, 0x3000}, {0x41000, 0x1000}};
	check(rgfxSurfaceFind(t, other, 2, 0x4000, 0x5000, false, &differs) == -1 && !differs,
	      "other pages are another plane");
	check(rgfxSurfaceFind(t, same, 1, 0x3000, 0x5000, false, &differs) == -1 && !differs,
	      "a part of the pages is no plane's");
	check(rgfxSurfaceFind(t, same, 2, 0x4000, 0x6000, false, &differs) == -1 && differs,
	      "the same pages with another block size get no block");

	// A surface's blocks are of one mode: its whole allocation's pages, asked for in place, find
	// no copy block, nor its plane's pages a block in place.
	RgfxPageRun whole[2] = {{0x10000, 0x3000}, {0x40000, 0x3000}};
	check(rgfxSurfaceFind(t, whole, 2, 0x6000, 0x1000, true, &differs) == -1 && differs,
	      "an ask in place finds no copy block of the surface");
	check(rgfxSurfaceFind(t, other + 1, 1, 0x1000, 0x1000, true, &differs) == -1 && !differs,
	      "nor does it differ from another surface's");
	check(!rgfxSurfacePending(t, whole, 2), "a copy block waits for nothing");
	RgfxSurfaceEntry &in = entries[1];
	in.used = true;
	in.inPlace = true;
	in.pending = true;
	in.offset = 0x200000;
	in.size = 0x1000;
	in.length = 0x6000;
	RgfxPageRun inRuns[2] = {{0x80000, 0x2000}, {0x90000, 0x4000}};
	in.runs = inRuns;
	in.runCount = 2;
	RgfxPageRun inSame[2] = {{0x80000, 0x2000}, {0x90000, 0x4000}};
	check(rgfxSurfacePending(t, inSame, 2), "a block in place waits for the device's answer");
	check(rgfxSurfacePending(t, inSame, 1), "for every ask of its surface's pages");
	in.pending = false;
	check(!rgfxSurfacePending(t, inSame, 2), "until it has it");
	check(rgfxSurfaceFind(t, inSame, 2, 0x6000, 0x1000, true, &differs) == 1 && !differs,
	      "then an ask in place of the same pages finds it");
	check(rgfxSurfaceFind(t, inSame, 1, 0x2000, 0x3000, false, &differs) == -1 && differs,
	      "and an ask for a copy of its plane none");
	check(rgfxSurfaceFind(t, inSame, 2, 0x6000, 0x2000, true, &differs) == -1 && differs,
	      "nor one in place with another block size");
	// A plane that starts past its surface's first page's start (CoreVideo's, 512 bytes in) meets
	// the surface's block in place by that page.
	RgfxPageRun inPlane[1] = {{0x80200, 0x1e00}};
	check(rgfxSurfaceFind(t, inPlane, 1, 0x1e00, 0x3000, false, &differs) == -1 && differs,
	      "a copy of a plane starting inside the first page finds the block in place's mode");
	in.pending = true;
	check(rgfxSurfacePending(t, inPlane, 1), "and waits for its answer");
	in.pending = false;
	check(rgfxSurfaceHeldBy(t, a, 0x100000, 0x10) && !rgfxSurfaceHeldBy(t, a, 0x200000, 0x10),
	      "a holds only the copy block");
	rgfxSurfaceClear(t, in);
	check(rgfxSurfaceFreeEntry(t) == 1, "the block in place cleared");

	// Each client holds it once, however often it asks.
	check(rgfxSurfaceHold(e, b, 2, &pagesB) == kRgfxSurfaceHoldNew, "b's first hold");
	check(rgfxSurfaceHold(e, b, 2, &pagesB) == kRgfxSurfaceHoldAgain, "b asks again");
	check(e.holdCount == 2, "two holders");

	// Only a holder's span is named, and only inside the block.
	check(rgfxSurfaceHeldBy(t, a, 0x100000, 0x5000), "a holder names the block");
	check(rgfxSurfaceHeldBy(t, b, 0x101000, 0x1000), "and a span in it");
	check(!rgfxSurfaceHeldBy(t, c, 0x100000, 0x10), "no one else names it");
	check(!rgfxSurfaceHeldBy(t, a, 0x104000, 0x1001), "nor a span past it");
	check(!rgfxSurfaceHeldBy(t, a, 0xff000, 0x2000), "nor one before it");

	// A let-go ends a hold with the client's last ask, and the block with its last holder.
	void *pages = nullptr;
	uint32_t context = 0;
	check(rgfxSurfaceLetGo(e, c, false, &pages, &context) == kRgfxSurfaceLetGoNotHeld,
	      "c holds nothing");
	check(rgfxSurfaceLetGo(e, b, false, &pages, &context) == kRgfxSurfaceLetGoRef,
	      "b lets one ask go");
	check(rgfxSurfaceLetGo(e, b, false, &pages, &context) == kRgfxSurfaceLetGoHold &&
	          pages == &pagesB && context == 2,
	      "b's hold ends; a holds it still");
	check(!rgfxSurfaceHeldBy(t, b, 0x100000, 0x10), "b no longer names it");
	check(rgfxSurfaceHold(e, c, 3, &pagesC) == kRgfxSurfaceHoldNew, "c holds it");
	check(rgfxSurfaceHold(e, c, 3, &pagesC) == kRgfxSurfaceHoldAgain, "c asks again");
	check(rgfxSurfaceLetGo(e, c, true, &pages, &context) == kRgfxSurfaceLetGoHold &&
	          pages == &pagesC && context == 3,
	      "c's close lets every ask go");
	check(rgfxSurfaceLetGo(e, a, false, &pages, &context) == kRgfxSurfaceLetGoLast &&
	          pages == &pagesA && context == 1,
	      "a was its last holder");
	check(rgfxSurfaceAt(t, 0x100000) == 0, "found by its offset until cleared");
	rgfxSurfaceClear(t, e);
	check(rgfxSurfaceAt(t, 0x100000) == -1 && rgfxSurfaceFreeEntry(t) == 0, "cleared");

	// The holders are bounded.
	RgfxSurfaceEntry full = {};
	static char clients[kRgfxSurfaceHolders + 1];
	for (uint32_t i = 0; i < kRgfxSurfaceHolders; i++)
		check(rgfxSurfaceHold(full, &clients[i], i, nullptr) == kRgfxSurfaceHoldNew, "a holder");
	check(rgfxSurfaceHold(full, &clients[kRgfxSurfaceHolders], 99, nullptr) ==
	          kRgfxSurfaceHoldFull,
	      "past the most holders");

	// Where the kext grows a block's holds (RGFX_FEATURE_SURFACE_CACHE_EVERY), the holders past
	// the entry's own lie in the caller's memory, kept as they grow again, and stay with the
	// entry's slot once it is cleared.
	{
		RgfxSurfaceTable gt = {&full, 1};
		full.used = true;
		static_assert(sizeof(full.holds.first) / sizeof(full.holds.first[0]) ==
		                  kRgfxSurfaceHolders,
		              "the entry holds kRgfxSurfaceHolders itself");
		static RgfxSurfaceHold more1[kRgfxSurfaceHolders], more2[2 * kRgfxSurfaceHolders];
		static char extra[3 * kRgfxSurfaceHolders];
		uint32_t oldCapacity = 99;
		check(rgfxSurfaceGrowHolds(full, more1, kRgfxSurfaceHolders, &oldCapacity) == nullptr &&
		          oldCapacity == 0 && full.holds.capacity() == 2 * kRgfxSurfaceHolders,
		      "the first growth holds twice as many, with nothing to free");
		check(rgfxSurfaceHold(full, &clients[kRgfxSurfaceHolders], 99, nullptr) ==
		          kRgfxSurfaceHoldNew,
		      "a holder past the entry's own");
		for (uint32_t i = kRgfxSurfaceHolders + 1; i < 2 * kRgfxSurfaceHolders; i++)
			check(rgfxSurfaceHold(full, &extra[i], i, nullptr) == kRgfxSurfaceHoldNew,
			      "holders up to the grown room");
		check(rgfxSurfaceHold(full, &extra[0], 1, nullptr) == kRgfxSurfaceHoldFull,
		      "past the grown room: full again");
		check(rgfxSurfaceGrowHolds(full, more1, kRgfxSurfaceHolders, &oldCapacity) == more1 &&
		          oldCapacity == kRgfxSurfaceHolders,
		      "no smaller growth: the offered array goes back");
		check(rgfxSurfaceGrowHolds(full, more2, 2 * kRgfxSurfaceHolders, &oldCapacity) == more1 &&
		          oldCapacity == kRgfxSurfaceHolders && full.holds.capacity() == 3 * kRgfxSurfaceHolders,
		      "a second growth answers the old array to free");
		check(full.holds[kRgfxSurfaceHolders].client == &clients[kRgfxSurfaceHolders] &&
		          full.holds[kRgfxSurfaceHolders].context == 99,
		      "the grown holders kept");
		check(rgfxSurfaceHold(full, &extra[0], 1, nullptr) == kRgfxSurfaceHoldNew,
		      "a holder in the new room");
		check(rgfxSurfaceHold(full, &clients[kRgfxSurfaceHolders], 99, nullptr) ==
		          kRgfxSurfaceHoldAgain,
		      "a grown holder found again");
		void *p = nullptr;
		uint32_t ctx = 0;
		check(rgfxSurfaceLetGo(full, &clients[0], true, &p, &ctx) == kRgfxSurfaceLetGoHold &&
		          ctx == 0,
		      "a let-go of one of the entry's own");
		full.offset = 0x40000;
		full.size = 0x2000;
		check(rgfxSurfaceHeldBy(gt, &extra[0], 0x40000, 0x1000) &&
		          !rgfxSurfaceHeldBy(gt, &extra[1], 0x40000, 0x1000),
		      "held by a grown holder, and by no one else");
		for (uint32_t n = full.holdCount; n > 1; n--)
			check(rgfxSurfaceLetGo(full, full.holds[n - 1].client, true, &p, &ctx) ==
			          kRgfxSurfaceLetGoHold,
			      "holders let go, from the grown ones");
		check(rgfxSurfaceLetGo(full, full.holds[0].client, true, &p, &ctx) ==
		          kRgfxSurfaceLetGoLast,
		      "the last holder");
		rgfxSurfaceClear(gt, full);
		check(!full.used && full.holdCount == 0 && full.holds.more == more2 &&
		          full.holds.capacity() == 3 * kRgfxSurfaceHolders,
		      "a cleared entry keeps its grown holds for its slot's next block");
	}

	// Where the kext grows the table, it grows in chunks that never move its entries, past
	// as many as the region's block table holds.
	{
		static RgfxSurfaceEntry first[2];
		static RgfxSurfaceEntry chunks[kRgfxSurfaceChunks][kRgfxSurfaceChunkEntries];
		RgfxSurfaceTable gt = {first, 2};
		first[0].used = first[1].used = true;
		first[0].offset = 0x10000;
		first[1].offset = 0x20000;
		check(rgfxSurfaceFreeEntry(gt) == -1 && rgfxSurfaceMayGrow(gt), "full, and may grow");
		check(!rgfxSurfaceGrow(gt, nullptr), "no chunk, no growth");
		RgfxSurfaceEntry *const at1 = &gt.entries[1];
		check(rgfxSurfaceGrow(gt, chunks[0]) && gt.capacity == 2 + kRgfxSurfaceChunkEntries &&
		          gt.entries.moreFrom == 2,
		      "grown by a chunk past the first array");
		check(&gt.entries[1] == at1 && &gt.entries[2] == &chunks[0][0] &&
		          &gt.entries[1 + kRgfxSurfaceChunkEntries] ==
		              &chunks[0][kRgfxSurfaceChunkEntries - 1],
		      "the first array's entries stay where they were, the chunk's follow");
		check(rgfxSurfaceFreeEntry(gt) == 2, "the chunk's first entry is free");
		gt.entries[2].used = true;
		gt.entries[2].offset = 0x30000;
		check(rgfxSurfaceAt(gt, 0x30000) == 2 && rgfxSurfaceAt(gt, 0x20000) == 1,
		      "found by offset across the arrays");
		for (uint32_t i = 1; i < kRgfxSurfaceChunks; i++)
			check(rgfxSurfaceGrow(gt, chunks[i]), "a chunk more");
		check(&gt.entries[2 + kRgfxSurfaceChunkEntries] == &chunks[1][0] &&
		          &gt.entries[gt.capacity - 1] ==
		              &chunks[kRgfxSurfaceChunks - 1][kRgfxSurfaceChunkEntries - 1],
		      "each chunk follows the last");
		check(!rgfxSurfaceMayGrow(gt) && !rgfxSurfaceGrow(gt, chunks[0]),
		      "no more than kRgfxSurfaceChunks chunks");
		check(kRgfxSurfaceMaxBlocks + kRgfxSurfaceChunks * kRgfxSurfaceChunkEntries >=
		          kRgfxMaxBlocks,
		      "a grown table holds as many blocks as the block table");
		RgfxSurfaceTable none = {};
		check(!rgfxSurfaceMayGrow(none), "a table not yet made does not grow");
	}

	// Runs: adjacent segments merge, gaps start a run, and too many runs do not fit.
	RgfxPageRun runs[2];
	uint32_t count = 0;
	check(rgfxSurfaceAppendRun(runs, &count, 2, 0x1000, 0x800) &&
	          rgfxSurfaceAppendRun(runs, &count, 2, 0x1800, 0x800) &&
	          rgfxSurfaceAppendRun(runs, &count, 2, 0x5000, 0x0) &&
	          rgfxSurfaceAppendRun(runs, &count, 2, 0x9000, 0x1000) && count == 2 &&
	          runs[0].address == 0x1000 && runs[0].length == 0x1000 && runs[1].address == 0x9000,
	      "runs merge where they meet");
	check(!rgfxSurfaceAppendRun(runs, &count, 2, 0x20000, 0x1000), "a third run does not fit");

	// No room for a new block: the first time it lets dormant blocks go and waits, or gives up
	// with none; after that it waits on to its deadline, whatever wakes it, and gives up past it.
	check(rgfxSurfaceRoomStep(0, 3, 50) == kRgfxSurfaceRoomLetGoAndWait,
	      "no room: dormant blocks go, and the ask waits");
	check(rgfxSurfaceRoomStep(0, 0, 50) == kRgfxSurfaceRoomGiveUp,
	      "no room and none dormant: the ask gives up");
	check(rgfxSurfaceRoomStep(150, 3, 60) == kRgfxSurfaceRoomWait,
	      "woken with no room before the deadline: it waits on");
	check(rgfxSurfaceRoomStep(150, 0, 149) == kRgfxSurfaceRoomWait,
	      "it waits on once none is dormant: their releases are in flight");
	check(rgfxSurfaceRoomStep(150, 3, 150) == kRgfxSurfaceRoomGiveUp,
	      "at the deadline it gives up");

	// A refused copy block: each reason has its own name, and is logged the first time alone.
	{
		uint32_t logged = 0;
		for (uint32_t why = 0; why < kRgfxCopyRefusals; why++) {
			check(std::strcmp(rgfxCopyRefusalName(why), "unknown") != 0, "each reason is named");
			for (uint32_t other = 0; other < why; other++)
				check(std::strcmp(rgfxCopyRefusalName(why), rgfxCopyRefusalName(other)) != 0,
				      "no two reasons share a name");
		}
		check(kRgfxCopyRefusals <= 32, "a bit a reason");
		check(rgfxCopyRefusalFirst(&logged, kRgfxCopyRefusedRoom), "a reason's first refusal");
		check(!rgfxCopyRefusalFirst(&logged, kRgfxCopyRefusedRoom), "its second is not logged");
		check(rgfxCopyRefusalFirst(&logged, kRgfxCopyRefusedMap), "another reason's first is");
	}

	std::printf("surface_blocks_test: OK (%d checks)\n", checks);
	return 0;
}
