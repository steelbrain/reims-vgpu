// RgfxVblankClock: the display's refreshes in the guest's clock (ReimsGraphics). Header-only and
// free of the kernel's headers, so its native test (tests/vblank_clock_test.cpp) builds it as the
// kext does.
//
// The host stamps each VBLANK with the time of its refresh on its own monotonic clock, in
// nanoseconds; the guest's mach_absolute_time counts nanoseconds too (x86-64). The two clocks
// differ by an offset, and each refresh maps to its host time + that offset: a refresh's time is
// when it happened, not when its interrupt was taken.
//
// Where the device has HOST_TIME, the kext reads the host's clock between two reads of its own at
// each VBLANK (reading): the offset lies within half that round trip of the reading's midpoint
// less the host's time, its bound. The clock maps by the reading of the least bound, a bound that
// widens by kAgePpm of the reading's age, so a newer reading takes over within a few refreshes and
// a drift of the clocks is followed. The offset is then the clocks' own, whatever the VBLANKs'
// delivery takes.
//
// Without it (a host built before HOST_TIME), the offset is inferred from the VBLANKs' delivery:
// a VBLANK's receipt shows the offset plus the event's delivery, and the clock keeps the least
// (receipt - host time) seen, the offset with the shortest delivery. That offset may rise by
// kDriftNs a refresh, so it follows the clocks if they drift apart (at most 0.24 ms a second at
// 120 Hz). It moves with the delivery: a quicker one pulls it down at once and slower ones let it
// creep back, so under load the mapped times wander (measured: Safari's 1080p60 playback put the
// display link's times 38 to 76 us rms off a straight line, against 0 idle).
//
// Either way no mapped time lies after its receipt, and a mapped time never goes back.
//
// A display pipe transaction completes at the host's answer, which the host gives in the refresh
// that showed its surface, right after that refresh's VBLANK: its time is that refresh's
// (shownAt), where the answer comes within the refresh's period after the VBLANK's receipt.
//
// Our own code.
#ifndef RGFX_VBLANK_CLOCK_HPP_
#define RGFX_VBLANK_CLOCK_HPP_

#include <stdint.h>

class RgfxVblankClock {
public:
	// A frame at 60 Hz, the period before two refreshes have been seen.
	static constexpr uint64_t kDefaultPeriodNs = 16666667;
	// The gaps taken as the period.
	static constexpr uint64_t kMinPeriodNs = 1000000;
	static constexpr uint64_t kMaxPeriodNs = 100000000;
	// How far the offset inferred from delivery may rise a refresh.
	static constexpr uint64_t kDriftNs = 2000;
	// How fast a held reading's bound widens with its age, in parts per million of it: 1 us in
	// 20 ms, so a newer reading takes over within a few refreshes from a held one a microsecond
	// tighter, and the offset follows the clocks' drift (measured on an x86-64 guest: 0.7 ppm
	// between its clock and the host's raw monotonic one, which NTP does not slew).
	static constexpr uint64_t kAgePpm = 50;

	// The host's clock from HOST_TIME's three reads, HI, LO and HI again: the second HI where LO
	// is below 2^31 (where LO wrapped between the HI reads, it did so before LO was read), else
	// the first.
	static uint64_t hostTime(uint32_t hi, uint32_t lo, uint32_t hiAgain)
	{
		return (static_cast<uint64_t>(lo < 0x80000000u ? hiAgain : hi) << 32) | lo;
	}

	// A reading of the host's clock (HOST_TIME): `hostNs` read between the guest's `before` and
	// `after`. It is held where its bound is no more than the held one's, widened by its age; a
	// reading of 0 (a device reset) is not taken.
	void reading(uint64_t before, uint64_t hostNs, uint64_t after)
	{
		if (!hostNs || after < before)
			return;
		uint64_t bound = (after - before) / 2;
		if (haveReading_ && before >= readingAt_ &&
		    bound > readingBound_ + (before - readingAt_) * kAgePpm / 1000000)
			return;
		haveReading_ = true;
		readingOffset_ = static_cast<int64_t>(before + bound - hostNs);
		readingBound_ = bound;
		readingAt_ = before;
	}

	// The VBLANK of refresh `count`, stamped `hostNs` by the host, taken at `receipt` (the
	// guest's clock): the refresh's time in the guest's clock, by the held reading's offset where
	// one came, else by the offset with the shortest delivery.
	uint64_t vblank(uint64_t count, uint64_t hostNs, uint64_t receipt)
	{
		int64_t sample = static_cast<int64_t>(receipt - hostNs);
		if (!have_ || sample < offset_ + static_cast<int64_t>(kDriftNs))
			offset_ = sample;
		else
			offset_ += static_cast<int64_t>(kDriftNs);
		int64_t offset = haveReading_ ? readingOffset_ : offset_;
		uint64_t time = hostNs + static_cast<uint64_t>(offset);
		if (time > receipt)
			time = receipt;
		if (have_ && time <= time_)
			time = time_ + 1;
		if (have_ && count > count_ && hostNs > hostNs_) {
			// A gap outside a display's rates (10 to 1000 Hz) is not taken.
			uint64_t gap = (hostNs - hostNs_) / (count - count_);
			if (gap >= kMinPeriodNs && gap <= kMaxPeriodNs)
				period_ = gap;
		}
		have_ = true;
		count_ = count;
		hostNs_ = hostNs;
		time_ = time;
		receipt_ = receipt;
		return time;
	}

	// The last refresh's time (0 before any), and the next one's, a period after it.
	uint64_t last() const { return have_ ? time_ : 0; }
	uint64_t next() const { return have_ ? time_ + period_ : 0; }
	uint64_t period() const { return period_; }

	// The time of the refresh whose answer is taken at `now`: the last refresh's, where `now` is
	// within its period after that refresh's VBLANK was taken; else `now` (no VBLANK came
	// lately: the device's VBLANKs are off).
	uint64_t shownAt(uint64_t now) const
	{
		if (have_ && now >= receipt_ && now - receipt_ < period_)
			return time_;
		return now;
	}

private:
	bool have_ = false;
	int64_t offset_ = 0;
	uint64_t count_ = 0;
	uint64_t hostNs_ = 0;
	uint64_t time_ = 0;
	uint64_t receipt_ = 0;
	uint64_t period_ = kDefaultPeriodNs;
	// The held reading of the host's clock: its offset, its bound and when it was taken.
	bool haveReading_ = false;
	int64_t readingOffset_ = 0;
	uint64_t readingBound_ = 0;
	uint64_t readingAt_ = 0;
};

#endif // RGFX_VBLANK_CLOCK_HPP_
