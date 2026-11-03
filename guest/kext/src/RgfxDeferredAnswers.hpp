// RgfxDeferredAnswers: the device's answers to the pipeline creates a process posts without
// waiting (kRgfxAccelMethodSubmit with kRgfxSubmitDeferred), kept until the process takes them
// (kRgfxAccelMethodTakeAnswers).
//
// The bundle posts a create that way only where the device accepted an equal one of its process
// before: the create returns at once, its state taking that one's reflection, and only what the
// device assigns this create alone (its gpuResourceID) waits, at its first read, for the create's
// own answer. A slot is reserved before the create is posted and its seqno stamped before the
// device can see the message (ReimsGraphics::tryPost's `waiterSeqno`), so an answer always finds
// its slot. The table holds at most N slots, PerContext of them for one context: where none is free,
// the kext posts the create as it posts any other, and waits for its answer. A context's slots are
// dropped at its close, and the answers in them with them.
//
// The caller holds the framebuffer's state lock around every call; a slot's seqno is read and
// written atomically, as the post stamps it outside the lock. Header-only and free of the
// kernel's headers, so its native test (tests/deferred_answers_test.cpp) builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_DEFERRED_ANSWERS_HPP_
#define RGFX_DEFERRED_ANSWERS_HPP_

#include <stdint.h>
#include <string.h>

#include "reims_graphics_wire.h"

template <uint32_t N, uint32_t PerContext, uint32_t ReplyMax>
class RgfxDeferredAnswers {
public:
	// Where a create's answer stands.
	enum State {
		// No slot of the context's waits for or holds it: never deferred, taken, or dropped.
		kAbsent,
		// Posted, not yet answered.
		kWaiting,
		// Answered, not yet taken.
		kAnswered,
	};

	// A free slot for a create of context `context` (never 0), or -1 where every slot is held or
	// the context holds PerContext of them.
	int reserve(uint32_t context)
	{
		if (context == 0 || held(context) >= PerContext)
			return -1;
		for (uint32_t i = 0; i < N; i++) {
			Slot &s = m_slots[i];
			if (s.context != 0)
				continue;
			s.context = context;
			__atomic_store_n(&s.seqno, uint64_t{0}, __ATOMIC_RELEASE);
			s.answered = false;
			s.length = 0;
			return static_cast<int>(i);
		}
		return -1;
	}

	// Where the post of slot `slot`'s create stamps its seqno before the device sees it.
	uint64_t *seqnoOf(int slot) { return &m_slots[slot].seqno; }

	// Frees slot `slot`, whose create was not posted.
	void cancel(int slot) { m_slots[slot].context = 0; }

	// Takes the answer to the create at `seqno`: its `header` (type, size and seqno) and the
	// `size` payload bytes at `payload` (the ring's; read once each), kept whole up to ReplyMax
	// bytes. True where a slot waits for it.
	bool answer(const RgfxMsgHeader &header, const volatile uint8_t *payload, uint32_t size)
	{
		if (header.seqno == 0 || header.type != RGFX_EVT_PIPELINE_REFLECTION)
			return false;
		for (uint32_t i = 0; i < N; i++) {
			Slot &s = m_slots[i];
			if (s.context == 0 || s.answered ||
			    __atomic_load_n(&s.seqno, __ATOMIC_ACQUIRE) != header.seqno)
				continue;
			const uint32_t head = static_cast<uint32_t>(sizeof(RgfxMsgHeader));
			uint32_t n = head + size < ReplyMax ? head + size : ReplyMax;
			RgfxMsgHeader whole = header;
			whole.size = n;
			memcpy(s.reply, &whole, head);
			for (uint32_t b = head; b < n; b++)
				s.reply[b] = payload[b - head];
			s.length = n;
			s.answered = true;
			return true;
		}
		return false;
	}

	// Where context `context`'s create at `seqno` stands.
	State state(uint32_t context, uint64_t seqno) const
	{
		for (uint32_t i = 0; i < N; i++) {
			const Slot &s = m_slots[i];
			if (s.context == context && context != 0 &&
			    __atomic_load_n(&s.seqno, __ATOMIC_ACQUIRE) == seqno)
				return s.answered ? kAnswered : kWaiting;
		}
		return kAbsent;
	}

	// Copies context `context`'s answers, each whole, into the `capacity` bytes at `out`, in
	// their seqnos' order, while they fit, freeing each slot copied: the bytes written, and in
	// `*count` the answers. The rest stay for a later take.
	uint32_t take(uint32_t context, uint8_t *out, uint32_t capacity, uint32_t *count)
	{
		uint32_t used = 0, taken = 0;
		for (;;) {
			Slot *first = nullptr;
			for (uint32_t i = 0; i < N; i++) {
				Slot &s = m_slots[i];
				if (s.context == context && context != 0 && s.answered &&
				    (first == nullptr || s.seqno < first->seqno))
					first = &s;
			}
			if (first == nullptr || capacity - used < first->length)
				break;
			memcpy(out + used, first->reply, first->length);
			used += first->length;
			taken++;
			first->context = 0;
		}
		*count = taken;
		return used;
	}

	// Frees the slot of the create at `seqno`, answered or not, whichever context holds it: a
	// create posted after its context's close dropped the context's slots. Seqnos are the
	// device's, one a message, so no other create's slot goes with it.
	void forget(uint64_t seqno)
	{
		for (uint32_t i = 0; i < N; i++) {
			Slot &s = m_slots[i];
			if (s.context != 0 && seqno != 0 && __atomic_load_n(&s.seqno, __ATOMIC_ACQUIRE) == seqno)
				s.context = 0;
		}
	}

	// Frees every slot of context `context`, answered or not: its close.
	void drop(uint32_t context)
	{
		for (uint32_t i = 0; i < N; i++) {
			if (m_slots[i].context == context)
				m_slots[i].context = 0;
		}
	}

	// The slots context `context` holds.
	uint32_t held(uint32_t context) const
	{
		uint32_t n = 0;
		for (uint32_t i = 0; i < N; i++)
			n += m_slots[i].context == context ? 1 : 0;
		return n;
	}

private:
	struct Slot {
		// The context the slot is held for; 0 while it is free.
		uint32_t context;
		uint32_t length;
		// The create's seqno, 0 until its post stamps it.
		uint64_t seqno;
		bool answered;
		uint8_t reply[ReplyMax];
	};
	Slot m_slots[N] = {};
};

#endif // RGFX_DEFERRED_ANSWERS_HPP_
