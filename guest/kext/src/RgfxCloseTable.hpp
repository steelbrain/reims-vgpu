// RgfxCloseTable: the CONTEXT_CLOSEs on the control ring, waiting for their answers.
//
// A process's close is posted with its entry's seqno written before the device can see the
// message (ReimsGraphics::postAnswered), so the answer, however late, finds the entry: the
// framebuffer's work loop hands it over (rgfxCloseAnswer). The closing thread waits
// kCloseAnswerMs for it. Answered in time (rgfxCloseInTime), the entry is free again and the
// thread lets go of what the context held: its blocks, its holds of shared surface blocks, its
// resource-id chunks and its range of GPU addresses. Not answered in time, the entry is late: the
// device may still run the context's buffers over those blocks and still map the ids and the
// addresses to its objects, so none of them goes to anyone else. The thread hands them to the
// entry's token, an owner no client is, and parks the entry (rgfxClosePark). Its answer then lets
// the release worker take it (rgfxCloseTakeLate) and let go of all of it, as the thread does for
// an answer in time. A close never answered holds them for good.
//
// Header-only and free of the kernel's headers, so its native test (tests/close_table_test.cpp)
// builds it as the kext does.
//
// Our own code.
#ifndef RGFX_CLOSE_TABLE_HPP_
#define RGFX_CLOSE_TABLE_HPP_

#include <stdint.h>

enum RgfxCloseState : uint8_t {
	kRgfxCloseFree = 0,
	// Posted, or being posted: its seqno is written before the message is published.
	kRgfxClosePosted,
	// Answered while its thread waits, `status` the device's.
	kRgfxCloseAnswered,
	// Not answered in time: its context's holdings are its token's.
	kRgfxCloseLate,
	// Late, then answered, `status` the device's: for the release worker once parked.
	kRgfxCloseLateAnswered,
	// Taken by the release worker, which lets go of its token's holdings, then frees it.
	kRgfxCloseRetiring,
};

struct RgfxCloseEntry {
	// The close's seqno: written by the post, read by the answer.
	uint64_t seqno;
	uint32_t context;
	// The context's run of GPU address units (RgfxContextSlots.hpp).
	uint32_t firstUnit;
	uint32_t units;
	uint32_t status;
	RgfxCloseState state;
	// Late, and its context's holdings handed to its token: the worker may let go of them.
	bool parked;
	// Its address is the token, the owner of a late close's holdings: unique while the entry is
	// in use, and never a client's.
	char token;
};

struct RgfxCloseTable {
	RgfxCloseEntry *entries;
	uint32_t capacity;
};

// A free entry for the close of `context`, whose units are the `units` from `firstUnit`, posted
// from here on: its index, or -1 where every entry is in use.
inline int32_t rgfxCloseReserve(RgfxCloseTable &t, uint32_t context, uint32_t firstUnit,
                                uint32_t units)
{
	for (uint32_t i = 0; i < t.capacity; i++) {
		RgfxCloseEntry &e = t.entries[i];
		if (e.state != kRgfxCloseFree)
			continue;
		e.seqno = 0;
		e.context = context;
		e.firstUnit = firstUnit;
		e.units = units;
		e.status = 0;
		e.parked = false;
		e.state = kRgfxClosePosted;
		return static_cast<int32_t>(i);
	}
	return -1;
}

// The close of entry `i` was not posted: the entry is free again.
inline void rgfxCloseUnposted(RgfxCloseTable &t, uint32_t i)
{
	t.entries[i].state = kRgfxCloseFree;
}

// The device answered the message of `seqno` with `status`: true where it is a close in the
// table, now answered. Another answer of that seqno, or one to no close, is not taken.
inline bool rgfxCloseAnswer(RgfxCloseTable &t, uint64_t seqno, uint32_t status)
{
	if (seqno == 0)
		return false;
	for (uint32_t i = 0; i < t.capacity; i++) {
		RgfxCloseEntry &e = t.entries[i];
		if ((e.state != kRgfxClosePosted && e.state != kRgfxCloseLate) ||
		    __atomic_load_n(&e.seqno, __ATOMIC_ACQUIRE) != seqno)
			continue;
		e.status = status;
		e.state = e.state == kRgfxClosePosted ? kRgfxCloseAnswered : kRgfxCloseLateAnswered;
		return true;
	}
	return false;
}

// Whether entry `i` was answered.
inline bool rgfxCloseAnswered(const RgfxCloseTable &t, uint32_t i)
{
	return t.entries[i].state == kRgfxCloseAnswered;
}

// The closing thread's wait for entry `i` ends: answered, the entry is free again and true, its
// status in `*status`; else it is late from here on, and false.
inline bool rgfxCloseInTime(RgfxCloseTable &t, uint32_t i, uint32_t *status)
{
	RgfxCloseEntry &e = t.entries[i];
	if (e.state == kRgfxCloseAnswered) {
		*status = e.status;
		e.state = kRgfxCloseFree;
		return true;
	}
	e.state = kRgfxCloseLate;
	return false;
}

// The owner a late entry `i`'s holdings are handed to.
inline const void *rgfxCloseToken(const RgfxCloseTable &t, uint32_t i)
{
	return &t.entries[i].token;
}

// Late entry `i`'s holdings are its token's now: true where its answer came already, so the
// caller runs the worker to take it.
inline bool rgfxClosePark(RgfxCloseTable &t, uint32_t i)
{
	t.entries[i].parked = true;
	return t.entries[i].state == kRgfxCloseLateAnswered;
}

// Takes one late close answered and parked, its index in `*at`: the caller lets go of its
// token's holdings, then frees the entry (rgfxCloseRetired), so no other close's holdings join
// the token's meanwhile. False where none is.
inline bool rgfxCloseTakeLate(RgfxCloseTable &t, uint32_t *at)
{
	for (uint32_t i = 0; i < t.capacity; i++) {
		RgfxCloseEntry &e = t.entries[i];
		if (e.state != kRgfxCloseLateAnswered || !e.parked)
			continue;
		e.state = kRgfxCloseRetiring;
		*at = i;
		return true;
	}
	return false;
}

// The late close of entry `i` is done with: its token holds nothing now, and the entry is free.
inline void rgfxCloseRetired(RgfxCloseTable &t, uint32_t i)
{
	t.entries[i].state = kRgfxCloseFree;
	t.entries[i].parked = false;
}

// How many closes are late and not yet answered.
inline uint32_t rgfxCloseLateCount(const RgfxCloseTable &t)
{
	uint32_t n = 0;
	for (uint32_t i = 0; i < t.capacity; i++)
		n += t.entries[i].state == kRgfxCloseLate || t.entries[i].state == kRgfxCloseLateAnswered;
	return n;
}

#endif // RGFX_CLOSE_TABLE_HPP_
