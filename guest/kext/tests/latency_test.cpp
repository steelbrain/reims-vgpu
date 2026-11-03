// The native test of RgfxLatency.hpp: a latency lands in a bucket within an eighth of it, and a
// percentile reads within that of the samples'.
#include "RgfxLatency.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "latency_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

RgfxLatency fresh()
{
	static RgfxLatency l;
	l = {};
	return l;
}
} // namespace

int main()
{
	// Every bucket's floor is in it, and the next bucket starts past it.
	bool rising = true;
	for (uint32_t b = 0; b + 1 < kRgfxLatencyBuckets; b++)
		rising &= rgfxLatencyFloor(b + 1) > rgfxLatencyFloor(b);
	check(rising, "the buckets' floors rise");
	check(rgfxLatencyBucket(~0ull) == kRgfxLatencyBuckets - 1, "the longest latency is the last's");
	const uint64_t samples[] = {0, 1, 7, 8, 9, 15, 16, 1000, 1023, 1024, 123456789, 1ull << 40,
	                            ~0ull >> 1};
	for (uint64_t ns : samples) {
		const uint32_t b = rgfxLatencyBucket(ns);
		check(b < kRgfxLatencyBuckets, "a bucket in range");
		check(rgfxLatencyFloor(b) <= ns, "a latency is at or past its bucket's floor");
		if (b + 1 < kRgfxLatencyBuckets)
			check(ns < rgfxLatencyFloor(b + 1), "and below the next bucket's");
		if (ns >= 8)
			check(ns - rgfxLatencyFloor(b) <= rgfxLatencyFloor(b) / 8,
			      "within an eighth of its floor");
	}

	RgfxLatency l = fresh();
	check(rgfxLatencyPercentile(l, 500) == 0, "no samples read 0");
	// 1000 samples, 1 us to 1000 us.
	for (uint64_t us = 1; us <= 1000; us++)
		rgfxLatencyAdd(l, us * 1000);
	check(l.count == 1000 && l.maxNs == 1000000, "every sample counted, the most kept");
	const uint64_t p50 = rgfxLatencyPercentile(l, 500), p99 = rgfxLatencyPercentile(l, 990);
	check(p50 >= 500000 && p50 <= 500000 + 500000 / 8 + 1000, "p50 within an eighth of 500 us");
	check(p99 >= 990000 && p99 <= 1000000, "p99 within an eighth of 990 us, no more than the most");
	check(rgfxLatencyPercentile(l, 1000) == 1000000, "p100 is the most measured");
	// One slow sample among fast ones shows at the top, not at p50.
	RgfxLatency m = fresh();
	for (int i = 0; i < 99; i++)
		rgfxLatencyAdd(m, 20000);
	rgfxLatencyAdd(m, 4000000);
	check(rgfxLatencyPercentile(m, 500) <= 20000 + 20000 / 8, "p50 is the fast ones'");
	check(rgfxLatencyPercentile(m, 999) == 4000000, "the slow one is past p99");

	std::printf("latency_test: OK (%d checks)\n", checks);
	return 0;
}
