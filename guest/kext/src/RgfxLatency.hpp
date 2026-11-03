// RgfxLatency: a histogram of latencies in nanoseconds, for the kext's own log lines: eight buckets
// an octave (each within an eighth of its lower edge), so a percentile reads within 12.5%.
//
// Header-only and free of the kernel's headers, so its native test (tests/latency_test.cpp)
// builds it as the kext does.
//
// Our own code.
#ifndef RGFX_LATENCY_HPP_
#define RGFX_LATENCY_HPP_

#include <stdint.h>

static constexpr uint32_t kRgfxLatencySubBuckets = 8;
// The 8 single nanoseconds, then octaves 3 to 63.
static constexpr uint32_t kRgfxLatencyBuckets = 62 * kRgfxLatencySubBuckets;

struct RgfxLatency {
	uint64_t count;
	uint64_t maxNs;
	uint32_t buckets[kRgfxLatencyBuckets];
};

// The bucket of `ns`: below 8 ns its own, else its octave's eighth.
inline uint32_t rgfxLatencyBucket(uint64_t ns)
{
	if (ns < kRgfxLatencySubBuckets)
		return static_cast<uint32_t>(ns);
	// Octaves 3 (8 to 15 ns) and up follow the 8 single nanoseconds.
	const uint32_t octave = 63 - static_cast<uint32_t>(__builtin_clzll(ns));
	const uint32_t sub = static_cast<uint32_t>(ns >> (octave - 3)) & (kRgfxLatencySubBuckets - 1);
	return (octave - 2) * kRgfxLatencySubBuckets + sub;
}

// The least latency of bucket `b`.
inline uint64_t rgfxLatencyFloor(uint32_t b)
{
	if (b < kRgfxLatencySubBuckets)
		return b;
	const uint32_t octave = b / kRgfxLatencySubBuckets + 2;
	const uint64_t sub = b % kRgfxLatencySubBuckets;
	return (kRgfxLatencySubBuckets + sub) << (octave - 3);
}

inline void rgfxLatencyAdd(RgfxLatency &l, uint64_t ns)
{
	l.count++;
	if (ns > l.maxNs)
		l.maxNs = ns;
	l.buckets[rgfxLatencyBucket(ns)]++;
}

// The latency under which `permille` thousandths of the samples lie: the next bucket's floor
// (the most measured where that is past it), 0 with no samples.
inline uint64_t rgfxLatencyPercentile(const RgfxLatency &l, uint32_t permille)
{
	if (!l.count)
		return 0;
	uint64_t want = (l.count * permille + 999) / 1000;
	if (want == 0)
		want = 1;
	uint64_t seen = 0;
	for (uint32_t b = 0; b < kRgfxLatencyBuckets; b++) {
		seen += l.buckets[b];
		if (seen >= want) {
			const uint64_t top = b + 1 < kRgfxLatencyBuckets ? rgfxLatencyFloor(b + 1) : l.maxNs;
			return top < l.maxNs ? top : l.maxNs;
		}
	}
	return l.maxNs;
}

#endif // RGFX_LATENCY_HPP_
