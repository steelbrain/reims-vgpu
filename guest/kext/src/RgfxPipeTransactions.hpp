// RgfxPipeTransactions: a display pipe client's transactions from their submit to their
// completion (RgfxDisplayPipeUserClient), and which completions are notified. Header-only and
// free of the kernel's headers, so its native test (tests/pipe_transactions_test.cpp) builds it
// as the kext does.
//
// A transaction that shows an IOSurface goes to the host as a PRESENT_SURFACE and stays
// outstanding until the host answers it. An answer completes that transaction and every one
// before it, oldest first. A transaction with nothing to show completes with the one before it,
// or at once where none is outstanding. At most `Max` are outstanding; the caller waits for room.
//
// Which completions are notified: a transaction carries a cookie, which its completion's
// notification gives back, or 0 for none. One with no cookie completes for its waiters
// (selector 9) and the display only. Measured on a macOS 26 guest: WindowServer drives the display
// in the transactional model ('wsrv' 0x80000013, then 0x11), each transaction with a cookie
// (4461 of them over three WindowServers). At its shutdown, when the session restarts, it hands
// the display back to the CPU ('wsrv' 0x2) and submits its last frames with none, waiting on each
// with selector 9 (4 per shutdown). Where the kext notified those, WindowServer aborted in
// IOAccelerator's transaction-notification callback at the first one's notification.
//
// Our own code.
#ifndef RGFX_PIPE_TRANSACTIONS_HPP_
#define RGFX_PIPE_TRANSACTIONS_HPP_

#include <stdint.h>

// Whether a transaction with cookie `cookie` is notified when it completes.
inline bool rgfxPipeNotifies(uint64_t cookie)
{
	return cookie != 0;
}

template <uint32_t Max> class RgfxPipeTransactions {
public:
	// A completed transaction: its id, the cookie its notification carries, and whether it is
	// notified (rgfxPipeNotifies).
	struct Completion {
		uint32_t id;
		uint64_t cookie;
		bool notified;
	};

	bool hasRoom() const { return count_ < Max; }
	uint32_t outstanding() const { return count_; }

	// A transaction that shows a surface, sent to the host as `seqno`: outstanding until the
	// host answers it. Needs room.
	void sent(uint64_t seqno, uint32_t id, uint64_t cookie, bool notified)
	{
		if (count_ < Max)
			items_[count_++] = {seqno, {id, cookie, notified}};
	}

	// A transaction with nothing to show. Where one is outstanding, it completes with the last
	// of them: false. Where none is, it completes now: true, and `*now` is its completion.
	// Needs room.
	bool nothingToShow(uint32_t id, uint64_t cookie, bool notified, Completion *now)
	{
		Completion c = {id, cookie, notified};
		if (count_ == 0) {
			*now = c;
			return true;
		}
		if (count_ < Max) {
			items_[count_] = {items_[count_ - 1].seqno, c};
			count_++;
		}
		return false;
	}

	// The host answered `seqno`: the transactions it completes, oldest first, into `out` (room
	// for `Max`), and their count; 0 where `seqno` is none of this client's.
	uint32_t answered(uint64_t seqno, Completion *out)
	{
		uint32_t match = Max;
		for (uint32_t i = 0; i < count_; i++) {
			if (items_[i].seqno == seqno)
				match = i;
		}
		if (match == Max)
			return 0;
		uint32_t n = match + 1;
		for (uint32_t i = 0; i < n; i++)
			out[i] = items_[i].completion;
		for (uint32_t i = n; i < count_; i++)
			items_[i - n] = items_[i];
		count_ -= n;
		return n;
	}

private:
	struct Item {
		uint64_t seqno;
		Completion completion;
	};

	Item items_[Max] = {};
	uint32_t count_ = 0;
};

#endif // RGFX_PIPE_TRANSACTIONS_HPP_
