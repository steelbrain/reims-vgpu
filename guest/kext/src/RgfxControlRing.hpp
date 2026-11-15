// RgfxControlRing: how the kext writes the submit ring of ring 0, the control ring: through which
// mapping each 8-byte word of a message goes, and the order of the words and the publish of the
// ring's producer. Header-only and free of the kernel's headers, over the kext's accesses to the
// region (`Words`, ReimsGraphics's), so the native tests (tests/control_ring_test.cpp, built for
// both architectures' rules) check the order as the kext keeps it.
//
// The region is uncached to the guest's kernel on an AMD host (docs/graphics-kext/design.md, "The
// region"), where a store to it runs at 13 MB/s: a SubmitBatch call, a few creates and destroys
// posted here, took 33.6 us p50 on a 780M host against 15.2 us on an Iris Xe host, whose EPT
// ignores the guest's memory type (measured, 11-14, Geekbench's Metal Super Resolution). Where the region's write-combined mappings are ordinary memory
// (x86-64's, RgfxRegionMapping.hpp), the kext writes the ring's data through a write-combined
// mapping of its own instead: the whole 4 KiB pages of the data past the header's page, which
// stays uncached, as the header's words are the protocol's atomics and no page of it is mapped
// write-combined. A write-combined store is ordered with no later store, so a store fence comes
// before the producer's publish: the host never sees the producer before the data it covers.
// arm64 maps no write-combined part, and its release store of the producer orders the data's
// stores before it.
//
// Our own code.
#ifndef RGFX_CONTROL_RING_HPP_
#define RGFX_CONTROL_RING_HPP_

#include <stdint.h>

#include "RgfxRegionMapping.hpp"
#include "reims_graphics_wire.h"

// Whether the kext writes ring 0's submit data through a write-combined mapping.
constexpr bool kRgfxControlWriteCombines = kRgfxRegionWriteCombines;

// The write-combined mapping's span, in bytes of the region: the whole 4 KiB pages of ring 0's
// submit data past the page of its header.
constexpr uint32_t kRgfxControlWcPage = 0x1000;
constexpr uint32_t kRgfxControlDataAt = RGFX_CONTROL_SUBMIT_OFFSET + RGFX_RING_HEADER_SIZE;
constexpr uint32_t kRgfxControlWcFrom =
    (RGFX_CONTROL_SUBMIT_OFFSET + RGFX_RING_HEADER_SIZE + kRgfxControlWcPage - 1) &
    ~(kRgfxControlWcPage - 1);
constexpr uint32_t kRgfxControlWcTo =
    (kRgfxControlDataAt + RGFX_CONTROL_RING_DATA_SIZE + kRgfxControlWcPage - 1) &
    ~(kRgfxControlWcPage - 1);
static_assert(kRgfxControlWcFrom > RGFX_CONTROL_SUBMIT_OFFSET,
              "the header's page is not in the write-combined span");
static_assert(kRgfxControlWcTo <= RGFX_CONTROL_SUBMIT_OFFSET + RGFX_CONTROL_RING_SPAN,
              "the write-combined span stays in ring 0's submit ring");

// Whether the word at `regionOffset` goes through the write-combined mapping.
constexpr bool rgfxControlWordCombined(uint32_t regionOffset)
{
	return kRgfxControlWriteCombines && regionOffset >= kRgfxControlWcFrom &&
	       regionOffset < kRgfxControlWcTo;
}

// Writes `bytes` bytes (a multiple of 8) of `words` at `dataOffset` bytes into the ring's data,
// one 8-byte store a word, each through the mapping rgfxControlWordCombined names:
//   `Words::data(uint32_t regionOffset, uint64_t word, bool combined)`.
template <class Words>
inline void rgfxControlWrite(Words &w, uint32_t dataOffset, const void *words, uint32_t bytes)
{
	const uint8_t *src = static_cast<const uint8_t *>(words);
	for (uint32_t i = 0; i + 8 <= bytes; i += 8) {
		uint64_t word;
		__builtin_memcpy(&word, src + i, sizeof(word));
		const uint32_t at = kRgfxControlDataAt + dataOffset + i;
		w.data(at, word, rgfxControlWordCombined(at));
	}
}

// Publishes the ring's producer, `produced`, after every word written before it: a store fence
// first where the data's stores may be write-combined (`Words::storeFence()`), then the release
// store of the producer (`Words::publish(uint64_t)`).
template <class Words>
inline void rgfxControlPublish(Words &w, uint64_t produced)
{
	if (kRgfxControlWriteCombines)
		w.storeFence();
	w.publish(produced);
}

// Publishes `produced` as rgfxControlPublish does, then, after a full fence (`Words::fullFence()`),
// reads whether the host's consumer of the ring sleeps (`Words::consumerIdle()`, the header's
// consumer_idle): whether ring 0's doorbell is to be rung. The caller rings it once it has let go
// of the ring's lock (ReimsGraphics::tryPost): the doorbell's write exits to the host, and another
// CPU's post would wait out that exit under the lock. The order of the ring's messages is the
// producer's, which the lock keeps; a doorbell is only a wake, and two posts that both find the
// consumer asleep ring it twice, as they did under the lock.
template <class Words>
inline bool rgfxControlPublishWantsBell(Words &w, uint64_t produced)
{
	rgfxControlPublish(w, produced);
	w.fullFence();
	return w.consumerIdle() == 1;
}

// The store fence the kext's Words take: on x86-64, SFENCE, which lands every store before it,
// the write-combined ones among them, before any store after it. arm64 maps no write-combined
// part (kRgfxControlWriteCombines), so it is never asked for there.
inline void rgfxStoreFence()
{
#if defined(__x86_64__)
	__builtin_ia32_sfence();
#endif
}

#endif // RGFX_CONTROL_RING_HPP_
