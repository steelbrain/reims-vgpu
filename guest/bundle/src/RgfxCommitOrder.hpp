// RgfxCommitOrder: the order of a process's command buffers across its queues, which the host
// commits them in (RGFX_FEATURE_COMMIT_ORDER; design.md, "Commit order").
//
// Metal orders a tracked resource across a process's queues by the order the command buffers were
// committed in (dgpu-oracle.md, Q6). Here a plain commit returns once the buffer is on its queue's
// ring, and the host takes each ring on a thread of its own, so the guest names the order: as a
// buffer goes on its ring, it names (RGFX_OP_COMMIT_ORDER) the latest buffer each other queue of
// the process put on its own ring since this queue last named that queue, and the host commits it
// after those.
//
// - **The table.** Each queue of the process holds a slot while it lives (join, leave): its id,
//   and `posted`, the seqno of its latest buffer on its ring, raised once the buffer is there
//   (posted). A slot given to another queue has its seqno emptied first, then its queue written,
//   so a reader, which reads the queue, the seqno and the queue again, never pairs one queue with
//   another's seqno, as the surfaces' writer slots are read (RgfxIOSurface.mm, nameSlots).
// - **Naming.** A post reads every other slot's `posted` (name) before its buffer goes on its ring,
//   and raises its own after. So a buffer it names was on its ring before the read, and a buffer
//   that names it read after its own raise: two buffers never name each other, and the host's
//   waits, from later posts to earlier ones, cannot meet in a cycle.
// - **Cheap.** A post loads two words a slot up to the highest slot ever held (a process has a
//   handful of queues), and names a queue only where its `posted` rose since this queue last
//   named it (RgfxCommitNamed, the queue's own, read and written only by its posts, which its
//   queue makes one at a time). A post that finds nothing new names nothing and carries no op.
// - **Past the table.** A queue that finds no free slot (more than kSlots live at once) names and
//   is named by no other: the host then commits its buffers as it takes them.
//
// Header-only and free of Metal, so the shell tests drive it as the queues do.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_COMMIT_ORDER_HPP_
#define RGFX_COMMIT_ORDER_HPP_

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "reims_graphics_wire.h"

// What one queue last named of each slot: the queue it named there, and the seqno.
struct RgfxCommitNamed {
	struct Entry {
		uint64_t queue = 0;
		uint64_t seqno = 0;
	};
	std::vector<Entry> bySlot;
};

class RgfxCommitOrder {
public:
	// The most queues of a process that take part at once.
	static constexpr size_t kSlots = 256;
	// The slot of a queue that holds none.
	static constexpr size_t kNoSlot = kSlots;

	// Queue `queue` (its id, never 0) takes a free slot, with nothing posted: its slot, or kNoSlot
	// where none is free.
	size_t join(uint64_t queue)
	{
		for (size_t i = 0; i < kSlots; i++) {
			Slot &s = m_slots[i];
			uint64_t free = 0;
			if (s.queue.load(std::memory_order_relaxed) != 0)
				continue;
			// Claimed first, so no other join empties its seqno; the seqno is emptied before the
			// queue is written (the header's order).
			if (!s.queue.compare_exchange_strong(free, kClaiming, std::memory_order_seq_cst))
				continue;
			s.posted.store(0, std::memory_order_seq_cst);
			s.queue.store(queue, std::memory_order_seq_cst);
			size_t high = m_high.load(std::memory_order_relaxed);
			while (high < i + 1 &&
			       !m_high.compare_exchange_weak(high, i + 1, std::memory_order_seq_cst)) {
			}
			return i;
		}
		return kNoSlot;
	}

	// The queue in `slot` goes: its slot is free again.
	void leave(size_t slot)
	{
		if (slot >= kSlots)
			return;
		m_slots[slot].queue.store(0, std::memory_order_seq_cst);
	}

	// Before a buffer of the queue in `slot` goes on its ring: each other queue whose `posted`
	// rose since this queue last named it (`named`, the queue's own), at that `posted`, appended
	// to `waits`, and noted in `named`.
	void name(size_t slot, RgfxCommitNamed &named, std::vector<RgfxSurfaceOrderWait> &waits) const
	{
		if (slot >= kSlots)
			return;
		const size_t high = m_high.load(std::memory_order_seq_cst);
		if (named.bySlot.size() < high)
			named.bySlot.resize(high);
		for (size_t i = 0; i < high; i++) {
			if (i == slot)
				continue;
			const Slot &s = m_slots[i];
			const uint64_t queue = s.queue.load(std::memory_order_seq_cst);
			if (queue == 0 || queue == kClaiming)
				continue;
			const uint64_t seqno = s.posted.load(std::memory_order_seq_cst);
			if (seqno == 0 || s.queue.load(std::memory_order_seq_cst) != queue)
				continue;
			RgfxCommitNamed::Entry &last = named.bySlot[i];
			if (last.queue == queue && last.seqno >= seqno)
				continue;
			last.queue = queue;
			last.seqno = seqno;
			waits.push_back({queue, seqno});
		}
	}

	// A buffer of the queue in `slot` is on its ring, at `seqno`.
	void posted(size_t slot, uint64_t seqno)
	{
		if (slot >= kSlots)
			return;
		m_slots[slot].posted.store(seqno, std::memory_order_seq_cst);
	}

private:
	// A slot's queue while a join empties its seqno: no queue's id (ids are a context's, under
	// RGFX_CONTEXT_OBJ_ID_SHIFT's bits).
	static constexpr uint64_t kClaiming = ~uint64_t(0);
	struct Slot {
		std::atomic<uint64_t> queue{0};
		std::atomic<uint64_t> posted{0};
	};
	std::array<Slot, kSlots> m_slots;
	// One past the highest slot ever taken.
	std::atomic<size_t> m_high{0};
};

#endif // RGFX_COMMIT_ORDER_HPP_
