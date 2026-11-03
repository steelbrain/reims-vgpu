// RgfxVblankClock: the display's refreshes in the guest's clock (ReimsGraphics). Header-only and
// free of the kernel's headers, so its native test (tests/vblank_clock_test.cpp) builds it as the
// kext does.
//
// The host stamps each VBLANK with the time of its refresh on its own monotonic clock, in
// nanoseconds; the guest's mach_absolute_time counts nanoseconds too (x86-64). The two clocks
// differ by an offset, which a VBLANK's receipt shows plus the event's delivery: the clock keeps
// the least (receipt - host time) seen, the offset with the shortest delivery, and maps each
// refresh to host time + that offset. So a refresh's time is when it happened, not when its
// interrupt was taken, and no mapped time lies after its receipt. The offset may rise by
// kDriftNs a refresh, so it follows the clocks if they drift apart (at most 0.24 ms a second at
// 120 Hz), and a mapped time never goes back.
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
	// How far the offset may rise a refresh.
	static constexpr uint64_t kDriftNs = 2000;

	// The VBLANK of refresh `count`, stamped `hostNs` by the host, taken at `receipt` (the
	// guest's clock): the refresh's time in the guest's clock.
	uint64_t vblank(uint64_t count, uint64_t hostNs, uint64_t receipt)
	{
		int64_t sample = static_cast<int64_t>(receipt - hostNs);
		if (!have_ || sample < offset_ + static_cast<int64_t>(kDriftNs))
			offset_ = sample;
		else
			offset_ += static_cast<int64_t>(kDriftNs);
		uint64_t time = hostNs + static_cast<uint64_t>(offset_);
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
};

#endif // RGFX_VBLANK_CLOCK_HPP_
