// RgfxReleaseTable: the BLOCK_RELEASEs on the control ring, waiting for their answers.
//
// The device answers a block's release once every command buffer its context committed before
// the release has finished (`crates/reims-vgpu/src/graphics.rs`, release_block). The kext's one
// release worker (RgfxAcceleratorContexts.cpp) posts every freed block's release up to the table's
// room, each in its own entry, and returns; the framebuffer's work loop hands each answer to its
// entry by seqno (rgfxReleaseAnswer), in whatever order the contexts' drains end, and enters the
// worker, which retires the answered entries (rgfxReleaseTakeDone). So releases posted together
// share their contexts' drains, and one slow context holds back only its own blocks.
//
// An entry not answered in time is late: the worker is told of it once, to log it, and it stays in
// flight until its answer, which retires it as any other. The device's answer is its word that it
// is done with the block: until then it may still run a buffer over the block, or copy a cached
// surface's GPU copy into the surface's pages (a host slowed by tens of seconds does both late),
// so nothing the release holds (the block, its wired pages, its surface and the surface's wire)
// is handed back before it.
//
// An entry is reserved before its release is posted, and the post writes its seqno into the
// entry before the message is published (ReimsGraphics::tryPost's `waiterSeqno`), so an answer
// that comes before the poster marks the entry posted still finds it.
//
// A deferred block whose pages are named before its release (RgfxAcceleratorLocks.cpp,
// rgfxSurfaceCacheReleaseAttach) waits for the host's answer to that naming, or for a lock's
// naming in flight: its entry, reserved and holding the block, is handed to the naming worker
// (kRgfxReleaseNaming), and the release worker goes on with the next block. Once the naming is
// done (kRgfxReleaseNamed), the release worker posts it as any other. So a slow naming holds
// back only the releases that name pages, never the others or the region's room they free.
//
// Header-only and free of the kernel's headers, so its native test (tests/release_table_test.cpp)
// builds it as the kext does.
//
// Our own code.
#ifndef RGFX_RELEASE_TABLE_HPP_
#define RGFX_RELEASE_TABLE_HPP_

#include <stdint.h>

enum RgfxReleaseState : uint8_t {
	kRgfxReleaseFree = 0,
	// Reserved, its release being posted: its seqno is set before the message is published.
	kRgfxReleasePosting,
	// On the ring, not yet answered.
	kRgfxReleasePosted,
	// On the ring, not answered in time (rgfxReleaseTakeDone told of it once): still waiting.
	kRgfxReleaseLate,
	// Answered, `status` the device's, to be retired.
	kRgfxReleaseAnswered,
	// Reserved and holding its block, whose deferred pages the naming worker names before the
	// release is posted: not on the ring, so no answer, lateness or timer looks at it.
	kRgfxReleaseNaming,
	// Its pages' naming done (`attach`): its release is the release worker's to post.
	kRgfxReleaseNamed,
};

struct RgfxReleaseEntry {
	// The release's seqno: written by the post, read by the answer.
	uint64_t seqno;
	// The block's, as rgfxBlockNextRelease gave it.
	uint64_t offset;
	uint64_t serial;
	uint64_t size;
	// When the block was freed, and when its release was posted (the caller's clock).
	uint64_t freedAt;
	uint64_t postedAt;
	uint32_t context;
	uint32_t status;
	// The release's flags, and whether its surface's cache was written since its last copy.
	uint32_t flags;
	// What the naming of its block's pages did (the kext's RgfxReleaseAttach), once named.
	uint32_t attach;
	bool written;
	RgfxReleaseState state;
};

struct RgfxReleaseTable {
	RgfxReleaseEntry *entries;
	uint32_t capacity;
	// Entries not free.
	uint32_t inFlight;
};

// A free entry, now posting, cleared: its index, or -1 where every entry is in flight.
inline int32_t rgfxReleaseReserve(RgfxReleaseTable &t)
{
	for (uint32_t i = 0; i < t.capacity; i++) {
		if (t.entries[i].state == kRgfxReleaseFree) {
			t.entries[i] = {};
			t.entries[i].state = kRgfxReleasePosting;
			t.inFlight++;
			return static_cast<int32_t>(i);
		}
	}
	return -1;
}

// The release of entry `i` is on the ring, at `now`: posted, unless its answer came already.
inline void rgfxReleasePosted(RgfxReleaseTable &t, uint32_t i, uint64_t now)
{
	RgfxReleaseEntry &e = t.entries[i];
	e.postedAt = now;
	if (e.state == kRgfxReleasePosting)
		e.state = kRgfxReleasePosted;
}

// The release of entry `i` was not posted (or no block was there to release): the entry is free
// again.
inline void rgfxReleaseUnposted(RgfxReleaseTable &t, uint32_t i)
{
	if (t.entries[i].state != kRgfxReleaseFree) {
		t.entries[i].state = kRgfxReleaseFree;
		t.inFlight--;
	}
}

// Entry `i`, reserved and holding its block, has its block's deferred pages named before its
// release is posted, by the naming worker (rgfxReleaseNextNaming).
inline void rgfxReleaseNaming(RgfxReleaseTable &t, uint32_t i)
{
	if (t.entries[i].state == kRgfxReleasePosting)
		t.entries[i].state = kRgfxReleaseNaming;
}

// The first entry whose block's pages are to be named, or -1 where none is: the naming worker,
// one at a time, names it, then marks it named.
inline int32_t rgfxReleaseNextNaming(const RgfxReleaseTable &t)
{
	for (uint32_t i = 0; i < t.capacity; i++) {
		if (t.entries[i].state == kRgfxReleaseNaming)
			return static_cast<int32_t>(i);
	}
	return -1;
}

// The naming of entry `i`'s block's pages is done, `attach` what it did: its release is to be
// posted.
inline void rgfxReleaseNamed(RgfxReleaseTable &t, uint32_t i, uint32_t attach)
{
	if (t.entries[i].state == kRgfxReleaseNaming) {
		t.entries[i].attach = attach;
		t.entries[i].state = kRgfxReleaseNamed;
	}
}

// An entry whose block's pages are named, posting again, for the release worker to post: its
// index, or -1 where none is.
inline int32_t rgfxReleaseTakeNamed(RgfxReleaseTable &t)
{
	for (uint32_t i = 0; i < t.capacity; i++) {
		if (t.entries[i].state == kRgfxReleaseNamed) {
			t.entries[i].state = kRgfxReleasePosting;
			return static_cast<int32_t>(i);
		}
	}
	return -1;
}

// The device answered the message of `seqno` with `status`: true where it is a release in the
// table, now answered. Another answer of that seqno, or one to no release, is not taken.
inline bool rgfxReleaseAnswer(RgfxReleaseTable &t, uint64_t seqno, uint32_t status)
{
	if (seqno == 0)
		return false;
	for (uint32_t i = 0; i < t.capacity; i++) {
		RgfxReleaseEntry &e = t.entries[i];
		if ((e.state == kRgfxReleasePosting || e.state == kRgfxReleasePosted ||
		     e.state == kRgfxReleaseLate) &&
		    __atomic_load_n(&e.seqno, __ATOMIC_ACQUIRE) == seqno) {
			e.status = status;
			e.state = kRgfxReleaseAnswered;
			return true;
		}
	}
	return false;
}

// Takes one entry done with, freeing it: an answered one (`*answered` true). Else one posted
// before `now - timeout` and not yet late becomes late, and a copy of it is given with
// `*answered` false, its entry kept in flight for its answer. False where neither is.
inline bool rgfxReleaseTakeDone(RgfxReleaseTable &t, uint64_t now, uint64_t timeout,
                                RgfxReleaseEntry *out, bool *answered)
{
	int32_t stale = -1;
	for (uint32_t i = 0; i < t.capacity; i++) {
		const RgfxReleaseEntry &e = t.entries[i];
		if (e.state == kRgfxReleaseAnswered) {
			*out = e;
			*answered = true;
			rgfxReleaseUnposted(t, i);
			return true;
		}
		if (stale < 0 && e.state == kRgfxReleasePosted && now - e.postedAt >= timeout)
			stale = static_cast<int32_t>(i);
	}
	if (stale < 0)
		return false;
	t.entries[stale].state = kRgfxReleaseLate;
	*out = t.entries[stale];
	*answered = false;
	return true;
}

// When the oldest release still waiting and not yet late was posted, or 0 where none is.
inline uint64_t rgfxReleaseOldest(const RgfxReleaseTable &t)
{
	uint64_t oldest = 0;
	for (uint32_t i = 0; i < t.capacity; i++) {
		const RgfxReleaseEntry &e = t.entries[i];
		if (e.state == kRgfxReleasePosted && (oldest == 0 || e.postedAt < oldest))
			oldest = e.postedAt;
	}
	return oldest;
}

#endif // RGFX_RELEASE_TABLE_HPP_
