// The display's refreshes in the guest's clock (RgfxVblankClock.hpp), built natively: a refresh's
// time is its host stamp plus the clocks' offset, the held reading of the host's clock's where
// one came (HOST_TIME), else the one with the shortest delivery, never after its receipt and never
// going back; either offset follows a drift; the period comes from the stamps; and an answer taken
// within a period after a VBLANK is that refresh's, one taken whole periods after it the refresh
// that many periods on, up to kOffPeriods, and the answers' times never go back.
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

// A clock with two VBLANKs at 120 Hz, the second's time in `*t` and its receipt in `*receipt`.
static RgfxVblankClock twoVblanks(uint64_t *t, uint64_t *receipt)
{
	RgfxVblankClock c;
	uint64_t host = 1000000;
	c.vblank(1, host, host + kOffset + 20000);
	*t = c.vblank(2, host + kPeriod, host + kPeriod + kOffset + 20000);
	*receipt = host + kPeriod + kOffset + 20000;
	return c;
}

static void shown()
{
	uint64_t t = 0, receipt = 0;
	// Taken with the VBLANK, or later within its period: that refresh's time.
	RgfxVblankClock c = twoVblanks(&t, &receipt);
	CHECK(c.shownAt(receipt) == t);
	CHECK(c.shownAt(receipt + kPeriod - 1) == t);
	// Whole periods after it, up to kOffPeriods: the refresh that many periods on, never after
	// the answer's receipt.
	c = twoVblanks(&t, &receipt);
	CHECK(c.shownAt(receipt + kPeriod) == t + kPeriod);
	c = twoVblanks(&t, &receipt);
	CHECK(c.shownAt(receipt + 2 * kPeriod + 5) == t + 2 * kPeriod);
	c = twoVblanks(&t, &receipt);
	const uint64_t lastOn = receipt + RgfxVblankClock::kOffPeriods * kPeriod - 1;
	CHECK(c.shownAt(lastOn) == t + (RgfxVblankClock::kOffPeriods - 1) * kPeriod);
	CHECK(t + (RgfxVblankClock::kOffPeriods - 1) * kPeriod <= lastOn);
	// kOffPeriods or more after it (the VBLANKs are off), or before it: the answer's own time.
	c = twoVblanks(&t, &receipt);
	const uint64_t off = receipt + RgfxVblankClock::kOffPeriods * kPeriod;
	CHECK(c.shownAt(off) == off);
	c = twoVblanks(&t, &receipt);
	CHECK(c.shownAt(receipt - 1) == receipt - 1);
}

static void shownNeverBack()
{
	// An answer taken a period and 3 ms after the last VBLANK's receipt, before the next VBLANK
	// came, is the next refresh's: the answer after that VBLANK, that refresh's too, does not
	// go back (mapped to its receipt, the first came 3 ms after the second).
	uint64_t t = 0, receipt = 0;
	RgfxVblankClock c = twoVblanks(&t, &receipt);
	const uint64_t first = c.shownAt(receipt + kPeriod + 3000000);
	CHECK(first == t + kPeriod);
	const uint64_t host = 1000000 + 2 * kPeriod;
	const uint64_t next = c.vblank(3, host, host + kOffset + 3500000);
	CHECK(next == t + kPeriod + RgfxVblankClock::kDriftNs); // its offset crept up a step
	CHECK(c.shownAt(host + kOffset + 3600000) == next && next >= first);
	// Each answer after one mapped by its own time, past the VBLANKs: the next refresh's time
	// would go back, so the last answer's is taken.
	c = twoVblanks(&t, &receipt);
	const uint64_t own = receipt + RgfxVblankClock::kOffPeriods * kPeriod + 1000;
	CHECK(c.shownAt(own) == own);
	const uint64_t late = c.vblank(3, 1000000 + 2 * kPeriod, own + 2000);
	CHECK(late < own);
	CHECK(c.shownAt(own + 3000) == own);
	// Two answers in one refresh: one time.
	c = twoVblanks(&t, &receipt);
	CHECK(c.shownAt(receipt + 100) == t && c.shownAt(receipt + 200) == t);
}

static void hostTimeWords()
{
	// HI, LO, HI again: no wrap between them, a wrap before LO was read, a wrap after it.
	CHECK(RgfxVblankClock::hostTime(1, 5, 1) == ((1ull << 32) | 5));
	CHECK(RgfxVblankClock::hostTime(1, 5, 2) == ((2ull << 32) | 5));
	CHECK(RgfxVblankClock::hostTime(1, 0xfffffff0u, 2) == ((1ull << 32) | 0xfffffff0u));
}

// A reading of the host's clock at host time `host`, `bound` either side of it in the guest's
// clock, whose offset is `offset`.
static void readAt(RgfxVblankClock &c, uint64_t host, uint64_t offset, uint64_t bound)
{
	c.reading(host + offset - bound, host, host + offset + bound);
}

static void readingsMapPastDelivery()
{
	// Deliveries of 170 us, but 20 us at every tenth refresh, as a vCPU awake for one takes it.
	// With a reading at each VBLANK, every refresh maps by the clocks' own offset; without, the
	// times move with the deliveries.
	RgfxVblankClock clock, inferred;
	uint64_t host = 1000000;
	bool wandered = false;
	for (uint64_t n = 1; n <= 300; n++, host += kPeriod) {
		uint64_t receipt = host + kOffset + (n % 10 ? 170000 : 20000);
		readAt(clock, host, kOffset, 2000);
		CHECK(clock.vblank(n, host, receipt) == host + kOffset);
		uint64_t t = inferred.vblank(n, host, receipt);
		CHECK(t <= receipt);
		if (n > 10 && t != host + kOffset + 20000)
			wandered = true;
	}
	CHECK(wandered);
}

static void shortestRoundTripHolds()
{
	RgfxVblankClock c;
	uint64_t host = 1000000;
	readAt(c, host, kOffset, 1000); // 1 us either side
	CHECK(c.vblank(1, host, host + kOffset + 90000) == host + kOffset);
	// 1 ms later, 5 us either side and 4 us off: the held one's bound has widened by 50 ns only.
	host += 1000000;
	readAt(c, host, kOffset + 4000, 5000);
	CHECK(c.vblank(2, host, host + kOffset + 90000) == host + kOffset);
	// 101 ms after the first: its bound has widened past 5 us, so the newer reading holds.
	host += 100000000;
	readAt(c, host, kOffset + 3000, 5000);
	CHECK(c.vblank(3, host, host + kOffset + 90000) == host + kOffset + 3000);
}

static void readingsFollowADrift()
{
	// The guest's clock gains 5 ppm on the host's (42 ns a refresh), its readings 2 or 2.5 us
	// either side: every refresh maps within a refresh's drift of the true offset.
	RgfxVblankClock c;
	uint64_t host = 1000000, gained = 0;
	for (uint64_t n = 1; n <= 1200; n++, host += kPeriod) {
		gained = host * 5 / 1000000;
		readAt(c, host, kOffset + gained, n % 2 ? 2000 : 2500);
		uint64_t t = c.vblank(n, host, host + kOffset + gained + 150000);
		CHECK(t <= host + kOffset + gained && t + 1000 >= host + kOffset + gained);
	}
}

static void readingsNeverAfterReceipt()
{
	// A reading that would put the refresh after its receipt: the receipt's time.
	RgfxVblankClock c;
	uint64_t host = 1000000, receipt = host + kOffset + 30000;
	readAt(c, host, kOffset + 500000, 1000);
	CHECK(c.vblank(1, host, receipt) == receipt);
	// A reading of 0 (the device reset) is not taken: a fresh clock maps by its delivery.
	RgfxVblankClock fresh;
	fresh.reading(5000, 0, 6000);
	CHECK(fresh.vblank(1, host, receipt) == receipt);
	CHECK(fresh.vblank(2, host + kPeriod, receipt + kPeriod - 10000) ==
	      receipt + kPeriod - 10000);
}

int main()
{
	first();
	shortestDelivery();
	drift();
	neverBack();
	shown();
	shownNeverBack();
	hostTimeWords();
	readingsMapPastDelivery();
	shortestRoundTripHolds();
	readingsFollowADrift();
	readingsNeverAfterReceipt();
	if (g_failed) {
		std::fprintf(stderr, "vblank_clock_test: %d of %d checks failed\n", g_failed, g_checks);
		return 1;
	}
	std::printf("vblank_clock_test: OK (%d checks)\n", g_checks);
	return 0;
}
