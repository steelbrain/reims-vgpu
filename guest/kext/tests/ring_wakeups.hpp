// The race of a guest's publish against the host's look before it sleeps, on a submission ring
// in memory, for the native tests of the kext's publishes: queue_publish_test.cpp's of
// rgfxQueuePublish and control_ring_test.cpp's of rgfxControlPublishWantsBell. The guest stores
// the ring's producer, fences, and reads consumer_idle, ringing the doorbell where the host sleeps;
// the host sets consumer_idle, fences, and reads the producer once more before it sleeps. Either
// the host's look sees the publish or the guest's read sees the host asleep, but only while both
// fence between their store and their load: without the guest's fence, x86 lets its read pass
// its store (the store waits in the store buffer), and arm64 lets a load-acquire pass an earlier
// store-release, and the host sleeps with a message no doorbell comes for (issue #158, a guest
// driver without that fence). The host side mirrors crates/reims-vgpu-graphics/src/ring.rs step
// for step: Consumer::next's take, sleep_or_pending, and awake after the doorbell.
//
// A wake is lost where the host sleeps, no doorbell rung, while the guest has decided about a
// message the host has not taken. The guest stores how far it has decided after each publish, and
// the sleeping host reads that, then the doorbell, so it counts a sleep as lost only once nothing
// can end it; with the fences none is. It then ends the sleep, as a backstop would, and the race
// goes on. The guest publishes one message at a time, once the host has taken the last. Before its
// handshake each side spins for a random time, so the two handshakes overlap across the whole
// window, and stores to a line of memory not in its cache, so its handshake's store waits behind
// that one in the store buffer while its load is answered, as a store behind a command buffer's
// bodies does; the guest also reads consumer_idle before it writes the message, so its read after
// the publish is a hit (crates/reims-vgpu-graphics/src/ring/wakeups.rs races the host's own code
// the same way).
//
// Each test runs the race with the kext's own publish, which must lose nothing, and with
// `--control` instead runs it with the publish's full fence left out, which must lose wakes where
// the CPU reorders: `make -C guest/kext test` runs the first, and the control runs by hand, as
// `build/tests/queue_publish_test --control`. Wakes lost per million messages without the fence,
// queue_publish_test / control_ring_test, one run each on 2026-11-15 (with the fence, none):
//   Apple M3 Max (arm64, macOS 27): 452k / 346k (422k with the arm64 rules);
//   Ryzen 7 7840HS (x86-64, Zen 4, built with clang++ on Linux): 905 / 528;
//   Core i9-13900H (x86-64, Raptor Lake, likewise): 1646 / 1. There the control ring's control,
//   whose x86-64 rules put a store fence before the producer's store, barely finds the missing
//   fence: one wake lost in the run.
#ifndef RING_WAKEUPS_HPP_
#define RING_WAKEUPS_HPP_

#include <stdint.h>

#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

#include "reims_graphics_wire.h"

// What a race came to.
struct RingWakeups {
	// Messages the guest published.
	uint64_t published;
	// Doorbells it rang.
	uint64_t rung;
	// Sleeps the host's second look called off: its look found the message.
	uint64_t looked;
	// Sleeps no doorbell would have ended.
	uint64_t lost;
};

// The most each side spins before its handshake, in turns.
constexpr uint64_t kRingWakeupsSpin = 128;

// A ring's memory: its header, then its byte ring, as the region lays out ring 0's submit ring.
struct alignas(4096) RingWakeupsMemory {
	uint8_t bytes[RGFX_CONTROL_RING_SPAN];
};

// The host's end of the ring, as ring.rs's Consumer takes it.
struct RingWakeupsHost {
	volatile uint8_t *header;
	uint64_t dataSize;
	uint64_t count = 0;

	volatile uint64_t *word64(uint32_t at) const
	{
		return reinterpret_cast<volatile uint64_t *>(header + at);
	}
	volatile uint32_t *word32(uint32_t at) const
	{
		return reinterpret_cast<volatile uint32_t *>(header + at);
	}
	uint64_t producer() const
	{
		return __atomic_load_n(word64(__builtin_offsetof(RgfxRingHeader, producer)),
		                       __ATOMIC_ACQUIRE);
	}
	void idle(uint32_t value) const
	{
		__atomic_store_n(word32(__builtin_offsetof(RgfxRingHeader, consumer_idle)), value,
		                 __ATOMIC_SEQ_CST);
	}
	// Consumer::next until it answers none: the producer read (acquire), each message's size read
	// from its header, and the host's count stored (release) past it.
	void takeAll()
	{
		while (producer() != count) {
			uint32_t size = 0;
			std::memcpy(&size,
			            const_cast<const uint8_t *>(header) + RGFX_RING_HEADER_SIZE +
			                (count & (dataSize - 1)) + 4,
			            sizeof(size));
			count += size;
			__atomic_store_n(word64(__builtin_offsetof(RgfxRingHeader, consumer)), count,
			                 __ATOMIC_RELEASE);
		}
	}
	// Consumer::sleep_or_pending: consumer_idle set, a full fence, the producer read again, and
	// consumer_idle cleared where the guest published more.
	bool sleepOrPending() const
	{
		idle(RGFX_CONSUMER_ASLEEP);
		__atomic_thread_fence(__ATOMIC_SEQ_CST);
		const bool pending = producer() != count;
		if (pending)
			idle(0);
		return pending;
	}
	// Consumer::awake.
	void awake() const { idle(0); }
};

// What a side does before its handshake, as ring/wakeups.rs's Lead: a spin of a random length,
// up to kRingWakeupsSpin turns, and a store to a line of `far`, 64 MiB, picked at random, so it
// misses the cache and a store after it waits behind it in the store buffer.
struct RingWakeupsLead {
	// A xorshift sequence's state.
	uint64_t state;
	std::vector<uint64_t> far = std::vector<uint64_t>(64u << 17);

	uint64_t next()
	{
		state ^= state << 13;
		state ^= state >> 7;
		state ^= state << 17;
		return state;
	}
	void spin()
	{
		for (volatile uint64_t turn = next() % kRingWakeupsSpin; turn > 0; turn = turn - 1) {
		}
	}
	void storeFar()
	{
		const uint64_t line = (next() >> 7) % (far.size() / 8);
		__atomic_store_n(&far[line * 8], state, __ATOMIC_RELAXED);
	}
};

// A turn of a wait: a pause for the first thousand, then a yield, so a loaded Mac that runs both
// threads on one CPU still moves on.
inline void ringWakeupsWait(uint32_t &turns)
{
	if (++turns < 1024) {
#if defined(__x86_64__)
		__builtin_ia32_pause();
#else
		__asm__ __volatile__("yield");
#endif
	} else {
		std::this_thread::yield();
	}
}

// Races `publish(dataOffset, produced)` (the guest's write of a 16-byte message at `dataOffset`
// into the byte ring and its publish of `produced`: whether it rings the doorbell) against the
// host's take and sleep, on a ring of `dataSize` bytes whose header is at `header`, for at most
// `most` messages or `ms` milliseconds.
template <class Publish>
RingWakeups ringWakeupsRace(volatile uint8_t *header, uint64_t dataSize, Publish publish,
                            uint64_t most, int ms)
{
	RingWakeupsHost host{header, dataSize};
	host.idle(RGFX_CONSUMER_ASLEEP);
	// The doorbell, rung by the guest and taken by the host's sleep; how far the guest has decided
	// whether to ring, its producer once it rang or not; and the race's end.
	bool bell = false;
	uint64_t decided = 0;
	bool done = false;
	RingWakeups out = {};
	std::thread hostThread([&] {
		RingWakeupsLead lead{0x2545f4914f6cdd1dull};
		for (;;) {
			host.takeAll();
			lead.spin();
			lead.storeFar();
			if (host.sleepOrPending()) {
				out.looked++;
				continue;
			}
			// Asleep until the doorbell.
			uint32_t turns = 0;
			for (;;) {
				const uint64_t through = __atomic_load_n(&decided, __ATOMIC_SEQ_CST);
				if (__atomic_exchange_n(&bell, false, __ATOMIC_SEQ_CST))
					break;
				if (through > host.count) {
					out.lost++;
					break;
				}
				if (__atomic_load_n(&done, __ATOMIC_SEQ_CST))
					return;
				ringWakeupsWait(turns);
			}
			host.awake();
		}
	});
	const volatile uint64_t *consumer = reinterpret_cast<const volatile uint64_t *>(
	    header + __builtin_offsetof(RgfxRingHeader, consumer));
	uint64_t produced = 0;
	RingWakeupsLead lead{0x9e3779b97f4a7c15ull};
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
	while (out.published < most &&
	       (out.published % 1024 != 0 || std::chrono::steady_clock::now() < deadline)) {
		// The host took the last.
		uint32_t turns = 0;
		while (__atomic_load_n(consumer, __ATOMIC_ACQUIRE) != produced)
			ringWakeupsWait(turns);
		lead.spin();
		// consumer_idle into the guest's cache, so the read after the publish's store is answered
		// at once while the store waits for its line: the window the fence closes.
		(void)__atomic_load_n(reinterpret_cast<const volatile uint32_t *>(
		                          header + __builtin_offsetof(RgfxRingHeader, consumer_idle)),
		                      __ATOMIC_RELAXED);
		lead.storeFar();
		const uint64_t at = produced & (dataSize - 1);
		produced += sizeof(RgfxMsgHeader);
		out.published++;
		if (publish(static_cast<uint32_t>(at), produced)) {
			__atomic_store_n(&bell, true, __ATOMIC_SEQ_CST);
			out.rung++;
		}
		__atomic_store_n(&decided, produced, __ATOMIC_SEQ_CST);
	}
	uint32_t turns = 0;
	while (__atomic_load_n(consumer, __ATOMIC_ACQUIRE) != produced)
		ringWakeupsWait(turns);
	__atomic_store_n(&done, true, __ATOMIC_SEQ_CST);
	hostThread.join();
	return out;
}

// A 16-byte message's header, seqno `seqno`, as the host reads it.
inline RgfxMsgHeader ringWakeupsMessage(uint64_t seqno)
{
	RgfxMsgHeader h = {};
	h.type = 1;
	h.size = sizeof(RgfxMsgHeader);
	h.seqno = seqno;
	return h;
}

#endif // RING_WAKEUPS_HPP_
