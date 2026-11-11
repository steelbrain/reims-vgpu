// RgfxSwapRate: what an arm64 display's swaps did over each window they ran (the caller's
// length, a minute), for one log line a window while swaps happen and none while the display is
// idle: the swaps begun, completed and dropped, those with no present, the distinct layer-0
// surfaces shown with their counts, and the swaps with a layer-1 surface, which is not shown. It
// tells a log reader the swap rate while an app runs and whether the app's own surface reaches a
// swap. Header-only and free of the
// kernel's headers, so its native test (tests/swap_rate_test.cpp) builds it as the kext does.
// The caller holds the display's gate around every call.
//
// Our own code.

#ifndef RGFX_SWAP_RATE_HPP_
#define RGFX_SWAP_RATE_HPP_

#include <stdint.h>

// The distinct layer-0 surfaces a window counts by ID; more are counted together.
constexpr uint32_t kRgfxSwapRateSurfaces = 4;

struct RgfxSwapRate {
	uint64_t start;      // the window's first swap, in the clock's ticks; 0 before any
	uint32_t begun;      // swaps begun
	uint32_t completed;  // swaps completed, shown or not
	uint32_t dropped;    // swaps dropped at their state, never begun
	uint32_t unshown;    // swaps begun with no layer-0 surface or a present that failed
	uint32_t layer1;     // swaps with a layer-1 surface
	uint32_t ids[kRgfxSwapRateSurfaces];
	uint32_t counts[kRgfxSwapRateSurfaces];
	uint32_t others;     // swaps of a layer-0 surface past the counted IDs

	// Opens the window at `now` where none is open.
	void touch(uint64_t now)
	{
		if (!start)
			start = now ? now : 1;
	}
	// A swap begun at `now`, of layer-0 surface `id` (0 for none) and with a layer-1 surface
	// where `layer1Surface`; `shown` where its present was posted.
	void begin(uint64_t now, uint32_t id, bool layer1Surface, bool shown)
	{
		touch(now);
		begun++;
		layer1 += layer1Surface;
		unshown += !shown;
		if (!id)
			return;
		for (uint32_t i = 0; i < kRgfxSwapRateSurfaces; i++) {
			if (ids[i] == id) {
				counts[i]++;
				return;
			}
			if (!ids[i]) {
				ids[i] = id;
				counts[i] = 1;
				return;
			}
		}
		others++;
	}
	void complete(uint64_t now)
	{
		touch(now);
		completed++;
	}
	void drop(uint64_t now)
	{
		touch(now);
		dropped++;
	}
	// Whether the window is `length` ticks old or more at `now`: its line is then due, and the
	// caller logs it and clears the window.
	bool due(uint64_t now, uint64_t length) const { return start && now - start >= length; }
	void clear() { *this = RgfxSwapRate(); }
};

#endif // RGFX_SWAP_RATE_HPP_
