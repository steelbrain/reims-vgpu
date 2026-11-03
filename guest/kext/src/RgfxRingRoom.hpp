// RgfxRingRoom: how a post that finds the control ring full waits for room (ReimsGraphics::post,
// postReleases and the display pipe's present). Header-only and free of the kernel's headers, so
// its native test (tests/ring_room_test.cpp) builds it as the kext does.
//
// The host takes ring 0 on a thread of its own, so a post that found no room looks again at the
// work loop's next pass: each device interrupt makes one (an answer, a queue's completion, a
// refresh, and, where the host took SUBMIT_ROOM, the record it posts once it has taken from a ring
// whose producer_waiting the post set), and so has the host moved on. A post reads the passes so far (its mark) before it looks at the ring; finding no room, it
// counts itself waiting, then, under the lock the work loop wakes it under, sleeps only where
// no pass came since its mark, until the next pass or its backstop. A pass counts itself, then
// wakes the waiters where any is counted. Each side writes its count before it reads the other's
// (sequentially consistent), so either the post sees the pass and looks again at once, or the
// pass sees the post and wakes it: none sleeps through a pass that came after its look.
//
// Our own code.
#ifndef RGFX_RING_ROOM_HPP_
#define RGFX_RING_ROOM_HPP_

#include <stdint.h>

struct RgfxRingRoom {
	// The work loop's passes so far.
	uint64_t passes;
	// The posts counted waiting for room.
	uint32_t waiters;
};

// How long a post that found no room sleeps at most before it looks again where no pass wakes
// it sooner, in microseconds: where the host took SUBMIT_ROOM, its record makes a pass once the
// host has taken from the ring, and the sleep's end is only a backstop; else the poll a post made
// before the passes woke it.
constexpr uint32_t kRgfxRingRoomBackstopUs = 1000;
constexpr uint32_t kRgfxRingRoomEventBackstopUs = 20000;

inline uint32_t rgfxRingRoomBackstopUs(bool submitRoom)
{
	return submitRoom ? kRgfxRingRoomEventBackstopUs : kRgfxRingRoomBackstopUs;
}

// A post's mark: the passes so far, read before it looks at the ring.
inline uint64_t rgfxRingRoomMark(const RgfxRingRoom &r)
{
	return __atomic_load_n(&r.passes, __ATOMIC_SEQ_CST);
}

// A post that found no room is waiting: from here a pass wakes it.
inline void rgfxRingRoomWait(RgfxRingRoom &r)
{
	__atomic_fetch_add(&r.waiters, 1, __ATOMIC_SEQ_CST);
}

// The post looked again: no longer waiting.
inline void rgfxRingRoomDone(RgfxRingRoom &r)
{
	__atomic_fetch_sub(&r.waiters, 1, __ATOMIC_SEQ_CST);
}

// Under the lock, after rgfxRingRoomWait: whether the post sleeps, no pass having come since its
// mark.
inline bool rgfxRingRoomSleeps(const RgfxRingRoom &r, uint64_t mark)
{
	return __atomic_load_n(&r.passes, __ATOMIC_SEQ_CST) == mark;
}

// The work loop's pass: whether a post waits, which the caller then wakes under the lock.
inline bool rgfxRingRoomPass(RgfxRingRoom &r)
{
	__atomic_fetch_add(&r.passes, 1, __ATOMIC_SEQ_CST);
	return __atomic_load_n(&r.waiters, __ATOMIC_SEQ_CST) != 0;
}

// Until when a post sleeps: `backstop` after `now`, or `deadline` where that comes first (each
// in the same clock's units).
inline uint64_t rgfxRingRoomSleepUntil(uint64_t now, uint64_t backstop, uint64_t deadline)
{
	const uint64_t at = now + backstop;
	return at < now || at > deadline ? deadline : at;
}

#endif // RGFX_RING_ROOM_HPP_
