// The native test of RgfxQueuePublish.hpp: the kext's publish of a queue's submission ring is the
// bundle's own (RgfxRingWriter::tryPost), step for step and word for word, and the client's table
// of its queues' ring headers lets go of each at the queue's destroy, at its block's free and at the
// context's close, after which the kext publishes on none of them. Raced against the host's look
// before it sleeps (ring_wakeups.hpp), the publish never leaves the host asleep with a buffer it
// was not rung for; `--control` races it without its full fence instead, which loses wakes.
#include "RgfxQueuePublish.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "ring_wakeups.hpp"

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "queue_publish_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

// A ring header's accesses, in their order: P the producer's store, F a full fence, I the
// consumer_idle load.
struct Ring {
	uint64_t producer = 0;
	uint32_t idle = 0;
	std::string order;
	void storeProducer(uint64_t p)
	{
		producer = p;
		order += 'P';
	}
	void fullFence() { order += 'F'; }
	uint32_t consumerIdle()
	{
		order += 'I';
		return idle;
	}
};

// The bundle's publish (RgfxRingWriter::tryPost): whether it rings, and the word it writes.
bool bundleRings(uint32_t idle, bool answered, bool asks, uint32_t ring, uint32_t *word)
{
	if (!(idle == RGFX_CONSUMER_ASLEEP || (idle == RGFX_CONSUMER_WALKING && answered)))
		return false;
	*word = answered && asks ? ring | RGFX_DOORBELL_ASKS : ring;
	return true;
}

struct Map {
	int id;
};

// A kept submission ring header's fields as the kext's KeptRing touches them
// (RgfxAcceleratorUserClientContexts.cpp), over the race's ring; without `Fenced`, its full fence
// left out, the control.
template <bool Fenced>
struct RacedRing {
	volatile uint8_t *header;
	void storeProducer(uint64_t producer)
	{
		__atomic_store_n(reinterpret_cast<volatile uint64_t *>(
		                     header + __builtin_offsetof(RgfxRingHeader, producer)),
		                 producer, __ATOMIC_RELEASE);
	}
	void fullFence()
	{
		if (Fenced)
			__atomic_thread_fence(__ATOMIC_SEQ_CST);
	}
	uint32_t consumerIdle()
	{
		return __atomic_load_n(reinterpret_cast<const volatile uint32_t *>(
		                           header + __builtin_offsetof(RgfxRingHeader, consumer_idle)),
		                       __ATOMIC_ACQUIRE);
	}
};

// The race of rgfxQueuePublish, on a queue ring of 256 bytes: the bundle's write of each buffer's
// message, then the kext's publish of it.
template <bool Fenced>
RingWakeups raceQueuePublish()
{
	RingWakeupsMemory *memory = new RingWakeupsMemory();
	volatile uint8_t *header = memory->bytes;
	RacedRing<Fenced> r{header};
	const RingWakeups raced = ringWakeupsRace(
	    header, 256,
	    [&](uint32_t at, uint64_t produced) {
		    const RgfxMsgHeader message = ringWakeupsMessage(produced / sizeof(RgfxMsgHeader));
		    std::memcpy(memory->bytes + RGFX_RING_HEADER_SIZE + at, &message, sizeof(message));
		    uint32_t word = 0;
		    return rgfxQueuePublish(r, 0x40000, produced, 0, &word);
	    },
	    1000000, 1000);
	delete memory;
	return raced;
}

void printRace(const char *what, const RingWakeups &raced)
{
	std::printf("queue_publish_test: %s: %llu published, %llu rung, %llu looks found the buffer, "
	            "%llu wakes lost\n",
	            what, static_cast<unsigned long long>(raced.published),
	            static_cast<unsigned long long>(raced.rung),
	            static_cast<unsigned long long>(raced.looked),
	            static_cast<unsigned long long>(raced.lost));
}
} // namespace

int main(int argc, char **argv)
{
	if (argc > 1 && std::strcmp(argv[1], "--control") == 0) {
		// The control: the publish without its full fence loses wakes where the CPU reorders.
		const RingWakeups raced = raceQueuePublish<false>();
		printRace("the race without the fence", raced);
		return raced.lost > 0 ? 0 : 1;
	}

	// The publish, for every consumer state and flag the bundle passes.
	const uint32_t ringId = 0x40000;
	const uint32_t states[] = {0, RGFX_CONSUMER_ASLEEP, RGFX_CONSUMER_WALKING, 3};
	for (uint32_t idle : states) {
		for (int answered = 0; answered < 2; answered++) {
			for (int asks = 0; asks < 2; asks++) {
				Ring r;
				r.idle = idle;
				const uint64_t flags = (answered ? kRgfxPublishAnswered : 0) |
				                       (asks ? kRgfxPublishAsks : 0);
				uint32_t word = 0xdead, expected = 0xbeef;
				const bool rings = rgfxQueuePublish(r, ringId, 0x1234, flags, &word);
				const bool bundle = bundleRings(idle, answered, asks, ringId, &expected);
				check(r.producer == 0x1234, "the producer is stored");
				check(r.order == "PFI", "producer, full fence, then consumer_idle, as the bundle's");
				check(rings == bundle, "rung where the bundle's own publish rings");
				check(!rings || word == expected, "with the word the bundle writes");
			}
		}
	}
	// The flags fit the mask the kext checks.
	check((kRgfxPublishFlags & ~uint64_t(3)) == 0 && kRgfxPublishAnswered != kRgfxPublishAsks,
	      "two flags, apart");

	// The table: kept at the create, found by the publish, each dropped once.
	RgfxQueueHeaders<Map, 3> t;
	Map a{1}, b{2}, c{3}, d{4};
	volatile uint8_t hdr[4][64] = {};
	check(rgfxQueueHeaderAdd(t, {0x10, 0x100000, 0x100000, &a, hdr[0]}), "a queue's header kept");
	check(!rgfxQueueHeaderAdd(t, {0x10, 0x200000, 0x200000, &d, hdr[3]}),
	      "a second for the same queue is not");
	check(rgfxQueueHeaderAdd(t, {0x11, 0x100000, 0x100800, &b, hdr[1]}),
	      "a second queue's, in the same block");
	check(rgfxQueueHeaderAdd(t, {0x12, 0x300000, 0x300000, &c, hdr[2]}), "a third's");
	check(!rgfxQueueHeaderAdd(t, {0x13, 0x400000, 0x400000, &d, hdr[3]}),
	      "none past the table's room: that queue's buffers the bundle publishes");
	const RgfxQueueHeader<Map> *q = rgfxQueueHeaderFind(t, 0x11);
	check(q && q->ring == 0x100800 && q->map == &b && q->header == hdr[1], "found by its queue");
	check(rgfxQueueHeaderFind(t, 0x13) == nullptr, "an unkept queue is not found");

	Map *out[3] = {};
	// The free of a block lets go of every ring's header in it, and of no other.
	check(rgfxQueueHeaderDropBlock(t, 0x100000, out) == 2, "the block's two headers dropped");
	check((out[0] == &a && out[1] == &b) || (out[0] == &b && out[1] == &a), "their mappings out");
	check(!rgfxQueueHeaderFind(t, 0x10) && !rgfxQueueHeaderFind(t, 0x11),
	      "the kext publishes on neither");
	check(rgfxQueueHeaderFind(t, 0x12) != nullptr, "the other block's stays");
	check(rgfxQueueHeaderDropBlock(t, 0x100000, out) == 0, "a second free drops nothing");
	// The queue's destroy.
	check(rgfxQueueHeaderDrop(t, 0x12, out) == 1 && out[0] == &c, "the destroy drops its header");
	check(rgfxQueueHeaderDrop(t, 0x12, out) == 0 && t.count == 0, "once");
	// The context's close: every one left.
	check(rgfxQueueHeaderAdd(t, {0x20, 0x500000, 0x500000, &a, hdr[0]}) &&
	          rgfxQueueHeaderAdd(t, {0x21, 0x600000, 0x600000, &b, hdr[1]}),
	      "kept again");
	check(rgfxQueueHeaderDropAll(t, out) == 2 && t.count == 0, "the close drops every one");
	check(!rgfxQueueHeaderFind(t, 0x20) && !rgfxQueueHeaderFind(t, 0x21), "and finds none");
	check(rgfxQueueHeaderAdd(t, {0x22, 0x700000, 0x700000, &c, hdr[2]}), "room again after");

	// The publish raced against the host's look before it sleeps.
	const RingWakeups raced = raceQueuePublish<true>();
	printRace("the race", raced);
	check(raced.lost == 0, "no buffer is left without its doorbell");

	std::printf("queue_publish_test: %d checks passed\n", checks);
	return 0;
}
