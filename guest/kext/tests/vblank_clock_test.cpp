// The display's refreshes in the guest's clock (RgfxVblankClock.hpp), built natively: a refresh's
// time is its host stamp plus the clocks' offset with the shortest delivery, never after its
// receipt and never going back; the offset follows a drift; the period comes from the stamps;
// and an answer taken within a period after a VBLANK is that refresh's.
#include <cstdio>

#include "RgfxVblankClock.hpp"

static int g_checks = 0, g_failed = 0;
#define CHECK(e)                                                                   \
	do {                                                                           \
		g_checks++;                                                                \
		if (!(e)) {                                                                \
			g_failed++;                                                            \
			std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #e); \
		}                                                                          \
	} while (0)

static const uint64_t kPeriod = 8333333; // 120 Hz
static const uint64_t kOffset = 5000000000ull; // the guest's clock ahead of the host's

static void first()
{
	RgfxVblankClock c;
	CHECK(c.last() == 0 && c.next() == 0);
	CHECK(c.shownAt(123) == 123); // no VBLANK yet: the answer's own time
	// The first: its receipt's offset, 60 Hz assumed for the next.
	CHECK(c.vblank(1, 1000000, 1000000 + kOffset + 40000) == 1000000 + kOffset + 40000);
	CHECK(c.next() == c.last() + RgfxVblankClock::kDefaultPeriodNs);
}

static void shortestDelivery()
{
	RgfxVblankClock c;
	uint64_t host = 1000000;
	c.vblank(1, host, host + kOffset + 90000); // delivered 90 us late
	host += kPeriod;
	// Delivered 10 us late: the offset falls to it, and the time is the refresh's.
	CHECK(c.vblank(2, host, host + kOffset + 10000) == host + kOffset + 10000);
	CHECK(c.period() == kPeriod && c.next() == c.last() + kPeriod);
	host += kPeriod;
	// Delivered 300 us late: mapped by the 10 us offset (plus one refresh's drift), before its
	// receipt.
	uint64_t t = c.vblank(3, host, host + kOffset + 300000);
	CHECK(t == host + kOffset + 10000 + RgfxVblankClock::kDriftNs);
	CHECK(t < host + kOffset + 300000);
	// A count that skips (a VBLANK the ring had no room for): the period from the stamps.
	host += 2 * kPeriod;
	c.vblank(5, host, host + kOffset + 10000);
	CHECK(c.period() == kPeriod);
}

static void drift()
{
	// The guest's clock gains 50 us a refresh on the host's: the offset follows, by the drift
	// step until it reaches the samples.
	RgfxVblankClock c;
	uint64_t host = 1000000, gained = 0, last = 0;
	for (uint64_t n = 1; n <= 200; n++, host += kPeriod) {
		gained += RgfxVblankClock::kDriftNs / 2;
		uint64_t receipt = host + kOffset + gained + 20000;
		uint64_t t = c.vblank(n, host, receipt);
		CHECK(t <= receipt);
		CHECK(t > last); // never goes back
		last = t;
	}
	// Tracking: within a step of the true offset.
	CHECK(c.last() + RgfxVblankClock::kDriftNs >= host - kPeriod + kOffset + gained + 20000);
}

static void neverBack()
{
	// Stamps that would map behind the last time (the host's refresh thread late, then on
	// time): each later than the one before.
	RgfxVblankClock c;
	uint64_t a = c.vblank(1, 1000000, 1000000 + kOffset + 500000);
	uint64_t b = c.vblank(2, 1000001, 1000001 + kOffset);
	CHECK(b > a);
}

static void shown()
{
	RgfxVblankClock c;
	uint64_t host = 1000000;
	c.vblank(1, host, host + kOffset + 20000);
	uint64_t t = c.vblank(2, host + kPeriod, host + kPeriod + kOffset + 20000);
	uint64_t receipt = host + kPeriod + kOffset + 20000;
	// Taken with the VBLANK, or later within its period: that refresh's time.
	CHECK(c.shownAt(receipt) == t);
	CHECK(c.shownAt(receipt + kPeriod - 1) == t);
	// A period or more after it, or before it: the answer's own time.
	CHECK(c.shownAt(receipt + kPeriod) == receipt + kPeriod);
	CHECK(c.shownAt(receipt - 1) == receipt - 1);
}

int main()
{
	first();
	shortestDelivery();
	drift();
	neverBack();
	shown();
	if (g_failed) {
		std::fprintf(stderr, "vblank_clock_test: %d of %d checks failed\n", g_failed, g_checks);
		return 1;
	}
	std::printf("vblank_clock_test: OK (%d checks)\n", g_checks);
	return 0;
}
