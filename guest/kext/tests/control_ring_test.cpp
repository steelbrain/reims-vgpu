// The native test of RgfxControlRing.hpp, built with each architecture's rules (the Makefile's
// control_ring_test, and control_ring_arm64_test with RGFX_GUEST_ARM64=1): every word of a message
// on ring 0's submit ring lands at its place, through the write-combined mapping exactly where
// x86-64's rules map the data's page so (never the header's page, never on arm64), and no store
// of the producer comes before a store of the data it covers: on x86-64 a store fence lies between
// the last word and the publish, on arm64 the publish is the release store after the last word.
// Raced against the host's look before it sleeps (ring_wakeups.hpp), the publish never leaves the
// host asleep with a message it was not rung for; `--control` races it without its full fence
// instead, which loses wakes.
#include "RgfxControlRing.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ring_wakeups.hpp"

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "control_ring_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

// The region's first ring span, and every access the writes made, in their order.
struct Recorder {
	enum Kind { kData, kFence, kPublish, kFullFence, kIdleRead };
	struct Access {
		Kind kind;
		uint32_t at;
		uint64_t value;
		bool combined;
	};
	std::vector<uint8_t> region = std::vector<uint8_t>(RGFX_CONTROL_RING_SPAN, 0);
	std::vector<Access> log;

	void data(uint32_t at, uint64_t word, bool combined)
	{
		std::memcpy(region.data() + at, &word, sizeof(word));
		log.push_back({kData, at, word, combined});
	}
	void storeFence() { log.push_back({kFence, 0, 0, false}); }
	void publish(uint64_t produced) { log.push_back({kPublish, 0, produced, false}); }
	// The header's consumer_idle, as the host left it.
	uint32_t idle = 0;
	void fullFence() { log.push_back({kFullFence, 0, 0, false}); }
	uint32_t consumerIdle()
	{
		log.push_back({kIdleRead, 0, idle, false});
		return idle;
	}
};

// Whether every data store comes before a fence (x86-64's rules) and the fence before the
// publish, which is last: nothing of the data is stored after the producer moves.
void publishedAfterData(const Recorder &r, const char *what)
{
	check(!r.log.empty() && r.log.back().kind == Recorder::kPublish, what);
	size_t lastData = 0, fence = 0;
	bool anyFence = false;
	for (size_t i = 0; i < r.log.size(); i++) {
		if (r.log[i].kind == Recorder::kData)
			lastData = i;
		if (r.log[i].kind == Recorder::kFence) {
			fence = i;
			anyFence = true;
		}
	}
	check(lastData < r.log.size() - 1, what);
	if (kRgfxControlWriteCombines)
		check(anyFence && fence > lastData && fence == r.log.size() - 2, what);
	else
		check(!anyFence, what);
}

// Ring 0's submit ring as the kext's ControlWords writes it (ReimsGraphics.cpp), over the race's
// ring, both mappings the same memory; without `Fenced`, the full fence left out, the control.
template <bool Fenced>
struct RacedWords {
	volatile uint8_t *rings;
	void data(uint32_t at, uint64_t word, bool)
	{
		*reinterpret_cast<volatile uint64_t *>(rings + at) = word;
	}
	void storeFence() { rgfxStoreFence(); }
	void publish(uint64_t count)
	{
		__atomic_store_n(reinterpret_cast<volatile uint64_t *>(
		                     rings + RGFX_CONTROL_SUBMIT_OFFSET +
		                     __builtin_offsetof(RgfxRingHeader, producer)),
		                 count, __ATOMIC_RELEASE);
	}
	void fullFence()
	{
		if (Fenced)
			__atomic_thread_fence(__ATOMIC_SEQ_CST);
	}
	uint32_t consumerIdle()
	{
		return __atomic_load_n(reinterpret_cast<const volatile uint32_t *>(
		                           rings + RGFX_CONTROL_SUBMIT_OFFSET +
		                           __builtin_offsetof(RgfxRingHeader, consumer_idle)),
		                       __ATOMIC_ACQUIRE);
	}
};

// The race of rgfxControlPublishWantsBell on ring 0's submit ring: each message written by
// rgfxControlWrite, then published.
template <bool Fenced>
RingWakeups raceControlPublish()
{
	RingWakeupsMemory *memory = new RingWakeupsMemory();
	RacedWords<Fenced> w{memory->bytes};
	const RingWakeups raced = ringWakeupsRace(
	    memory->bytes + RGFX_CONTROL_SUBMIT_OFFSET, RGFX_CONTROL_RING_DATA_SIZE,
	    [&](uint32_t at, uint64_t produced) {
		    const RgfxMsgHeader message = ringWakeupsMessage(produced / sizeof(RgfxMsgHeader));
		    rgfxControlWrite(w, at, &message, sizeof(message));
		    return rgfxControlPublishWantsBell(w, produced);
	    },
	    1000000, 1000);
	delete memory;
	return raced;
}

void printRace(const char *what, const RingWakeups &raced)
{
	std::printf("control_ring_test: %s (%s): %llu published, %llu rung, %llu looks found the "
	            "message, %llu wakes lost\n",
	            what, RGFX_GUEST_ARM64 ? "arm64" : "x86-64",
	            static_cast<unsigned long long>(raced.published),
	            static_cast<unsigned long long>(raced.rung),
	            static_cast<unsigned long long>(raced.looked),
	            static_cast<unsigned long long>(raced.lost));
}
} // namespace

int main(int argc, char **argv)
{
	if (argc > 1 && std::strcmp(argv[1], "--control") == 0) {
		// The control: the publish without its full fence loses wakes where the CPU reorders.
		const RingWakeups raced = raceControlPublish<false>();
		printRace("the race without the fence", raced);
		return raced.lost > 0 ? 0 : 1;
	}

	// The span: the data's whole 4 KiB pages past the header's, on x86-64's rules alone.
	check(kRgfxControlWcFrom == 0x1000 && kRgfxControlWcTo == 0x9000,
	      "the span is the data's pages past the header's");
	check(!rgfxControlWordCombined(RGFX_CONTROL_SUBMIT_OFFSET), "the header is never combined");
	check(!rgfxControlWordCombined(kRgfxControlDataAt), "the data in the header's page is uncached");
	check(rgfxControlWordCombined(kRgfxControlWcFrom) == kRgfxControlWriteCombines,
	      "the data's second page is combined where the rules map it so");
	check(!rgfxControlWordCombined(kRgfxControlWcTo), "nothing past the span is combined");
#if RGFX_GUEST_ARM64
	check(!kRgfxControlWriteCombines, "arm64 maps no write-combined part");
#else
	check(kRgfxControlWriteCombines, "x86-64 maps the data write-combined");
#endif

	// A message across the header's page and the next: each word at its place, through the
	// mapping its page takes, then the publish after a fence.
	{
		Recorder r;
		uint8_t message[64];
		for (uint32_t i = 0; i < sizeof(message); i++)
			message[i] = static_cast<uint8_t>(i + 1);
		const uint32_t dataOffset = kRgfxControlWcFrom - kRgfxControlDataAt - 24;
		rgfxControlWrite(r, dataOffset, message, sizeof(message));
		rgfxControlPublish(r, dataOffset + sizeof(message));
		check(std::memcmp(r.region.data() + kRgfxControlDataAt + dataOffset, message,
		                  sizeof(message)) == 0,
		      "the message lands at its place");
		check(r.log.size() == sizeof(message) / 8 + (kRgfxControlWriteCombines ? 2 : 1),
		      "one store a word, a fence where combined, one publish");
		for (const Recorder::Access &a : r.log) {
			if (a.kind == Recorder::kData)
				check(a.combined == rgfxControlWordCombined(a.at) &&
				          a.combined == (kRgfxControlWriteCombines && a.at >= kRgfxControlWcFrom),
				      "each word through its page's mapping");
		}
		publishedAfterData(r, "the producer is published after the message, fenced");
		check(r.log.back().value == dataOffset + sizeof(message), "the producer is the count");
	}

	// A batch of messages, the last ending at the ring's end: one publish after all of them.
	{
		Recorder r;
		const uint32_t size = 48;
		uint8_t message[size];
		std::memset(message, 0x5a, sizeof(message));
		uint32_t at = RGFX_CONTROL_RING_DATA_SIZE - 4 * size;
		for (int i = 0; i < 4; i++, at += size)
			rgfxControlWrite(r, at, message, size);
		rgfxControlPublish(r, RGFX_CONTROL_RING_DATA_SIZE);
		publishedAfterData(r, "a batch is published after its last message, fenced");
		check(r.log.front().at == kRgfxControlDataAt + RGFX_CONTROL_RING_DATA_SIZE - 4 * size,
		      "the batch's first word lands first");
		check(r.log[r.log.size() - (kRgfxControlWriteCombines ? 3 : 2)].at ==
		          kRgfxControlDataAt + RGFX_CONTROL_RING_DATA_SIZE - 8,
		      "the batch's last word is the ring's last");
	}

	// A pad's 8 bytes alone at the ring's end: one word.
	{
		Recorder r;
		const uint32_t pad[2] = {RGFX_MSG_PAD, 8};
		rgfxControlWrite(r, RGFX_CONTROL_RING_DATA_SIZE - 8, pad, 8);
		check(r.log.size() == 1 && r.log[0].at == kRgfxControlDataAt + RGFX_CONTROL_RING_DATA_SIZE - 8,
		      "a pad's header alone is one word");
		uint64_t word = 0;
		std::memcpy(&word, pad, sizeof(word));
		check(r.log[0].value == word, "the pad's type and size, as the host reads them");
	}

	// The publish that answers whether to ring: the producer's publish as above, then a full
	// fence, then the read of consumer_idle, last; the doorbell is wanted only where the host's
	// consumer sleeps, and the publish writes none itself, so the caller rings it out of the
	// ring's lock (ReimsGraphics::tryPost).
	for (uint32_t idle : {0u, 1u, 2u}) {
		Recorder r;
		r.idle = idle;
		const uint8_t message[32] = {1};
		rgfxControlWrite(r, 0, message, sizeof(message));
		const bool bell = rgfxControlPublishWantsBell(r, sizeof(message));
		check(bell == (idle == 1), "the doorbell is wanted where the consumer sleeps alone");
		const size_t n = r.log.size();
		check(n >= 3 && r.log[n - 1].kind == Recorder::kIdleRead &&
		          r.log[n - 2].kind == Recorder::kFullFence &&
		          r.log[n - 3].kind == Recorder::kPublish && r.log[n - 3].value == sizeof(message),
		      "published, then fenced, then consumer_idle read");
		r.log.resize(n - 2);
		publishedAfterData(r, "the producer is published after the message, fenced");
	}

	// The publish raced against the host's look before it sleeps.
	const RingWakeups raced = raceControlPublish<true>();
	printRace("the race", raced);
	check(raced.lost == 0, "no message is left without its doorbell");

	std::printf("control_ring_test: %d checks passed (%s)\n", checks,
	            RGFX_GUEST_ARM64 ? "arm64" : "x86-64");
	return 0;
}
