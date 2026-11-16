// The native test of RgfxDisplayPlaces.hpp, against a model of WindowServer: each display a
// rectangle of the global space, a posted point held at the nearest display's edge where it lies
// past every display, and the cursor shown on the display holding it at its pixel. Displays side
// by side, left, below and above, one at 2x, one in a scaled mode, a rearrangement mid-session, a
// display never entered, and the crossing's limits: its posts, its hold and a landing that never
// comes.
#include "RgfxDisplayPlaces.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "display_places_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

bool same(const RgfxGlobalRect &a, int32_t minx, int32_t miny, int32_t maxx, int32_t maxy)
{
	return a.minx == minx && a.miny == miny && a.maxx == maxx && a.maxy == maxy;
}

// WindowServer, as the kext sees it: each display's rectangle and mode.
struct Desktop {
	RgfxGlobalRect rects[RGFX_DISPLAYS_MAX];
	RgfxPlaceMode modes[RGFX_DISPLAYS_MAX];
	uint32_t count;

	// The display a point posted shows the cursor on, and its pixel there.
	void land(int32_t gx, int32_t gy, uint32_t *display, int32_t *x, int32_t *y) const
	{
		int64_t best = -1;
		for (uint32_t d = 0; d < count; d++) {
			const RgfxGlobalRect &r = rects[d];
			const int32_t cx = gx < r.minx ? r.minx : gx > r.maxx - 1 ? r.maxx - 1 : gx;
			const int32_t cy = gy < r.miny ? r.miny : gy > r.maxy - 1 ? r.maxy - 1 : gy;
			const int64_t dx = cx - gx, dy = cy - gy, far = dx * dx + dy * dy;
			if (best >= 0 && far >= best)
				continue;
			best = far;
			*display = d;
			*x = static_cast<int32_t>(static_cast<int64_t>(cx - r.minx) * modes[d].width /
			                          (r.maxx - r.minx));
			*y = static_cast<int32_t>(static_cast<int64_t>(cy - r.miny) * modes[d].height /
			                          (r.maxy - r.miny));
		}
	}
};

// A crossing's outcome.
struct Crossed {
	bool arrived;
	uint32_t posts;
	uint32_t confirmations;
	uint32_t display; // where the cursor is after it
	int32_t x, y;
};

// The kext's crossing onto `display` at pixel `x`, `y`, each post landing at once, `step` ns apart.
Crossed cross(RgfxDisplayPlaces *places, RgfxCrossing *c, const Desktop &desk, uint32_t display,
              int32_t x, int32_t y, uint64_t *now, uint64_t step = 16000000)
{
	Crossed out = {};
	rgfxPlacesFit(places, desk.modes);
	rgfxCrossingStart(c, display, x, y, *now);
	int32_t gx = 0, gy = 0;
	while (rgfxCrossingNext(*places, c, *now, &gx, &gy)) {
		out.posts++;
		*now += step;
		desk.land(gx, gy, &out.display, &out.x, &out.y);
		bool confirmed = false;
		const RgfxCrossingStep s =
		    rgfxCrossingLanded(places, c, out.display, out.x, out.y, &confirmed);
		out.confirmations += confirmed;
		if (s == RgfxCrossingStep::arrived) {
			out.arrived = true;
			break;
		}
		check(s == RgfxCrossingStep::again, "a landing elsewhere posts again");
	}
	check(!c->active && !c->waiting, "a crossing ends with no post out");
	*now += 1000000000;
	return out;
}

Desktop twoSideBySide()
{
	Desktop d = {};
	d.count = 2;
	d.rects[0] = {0, 0, 1920, 1080};
	d.rects[1] = {1920, 0, 3840, 1080};
	d.modes[0] = {1920, 1080, 120};
	d.modes[1] = {1920, 1080, 120};
	return d;
}

uint32_t confirmedCount(const RgfxDisplayPlaces &places)
{
	RgfxGlobalRect rects[RGFX_DISPLAYS_MAX];
	bool placed[RGFX_DISPLAYS_MAX];
	rgfxPlacesConfirmed(places, rects, placed);
	uint32_t n = 0;
	for (bool p : placed)
		n += p;
	return n;
}
} // namespace

int main()
{
	uint64_t now = 1000000000;

	// The guess: WindowServer's own arrangement of new displays, side by side, tops at 0, at 1x
	// on a 1x screen and HiDPI above it; a display not driven has none.
	{
		RgfxDisplayPlaces places = {};
		RgfxPlaceMode modes[RGFX_DISPLAYS_MAX] = {};
		modes[0] = {1920, 1080, 120};
		modes[1] = {2560, 1440, 240};
		modes[3] = {1280, 800, 0};
		rgfxPlacesFit(&places, modes);
		check(same(places.at[0].rect, 0, 0, 1920, 1080), "display 0 at 0,0 at 1x");
		check(same(places.at[1].rect, 1920, 0, 3200, 720), "display 1 beside it, HiDPI at 2x");
		check(places.at[2].state == RgfxPlaceState::none, "display 2 is not driven");
		check(same(places.at[3].rect, 3200, 0, 4480, 800), "display 3 beside display 1");
		check(confirmedCount(places) == 0, "a guess is never told");
		// Its mode changed, a display keeps its corner and takes the guessed size.
		places.at[1].state = RgfxPlaceState::confirmed;
		modes[1] = {1920, 1080, 120};
		rgfxPlacesFit(&places, modes);
		check(same(places.at[1].rect, 1920, 0, 3840, 1080) &&
		          places.at[1].state == RgfxPlaceState::guessed,
		      "a new mode keeps the corner, guessed again");
		modes[3] = {};
		rgfxPlacesFit(&places, modes);
		check(places.at[3].state == RgfxPlaceState::none,
		      "a display no longer driven is forgotten");
	}

	// Side by side, as guessed: one post a crossing, each confirming the display it lands on.
	{
		const Desktop desk = twoSideBySide();
		RgfxDisplayPlaces places = {};
		RgfxCrossing c = {};
		Crossed r = cross(&places, &c, desk, 1, 3, 500, &now);
		check(r.arrived && r.posts == 1 && r.display == 1 && r.x == 3 && r.y == 500,
		      "side by side: onto display 1 at its pixel, in one post");
		check(r.confirmations == 1 && places.at[1].state == RgfxPlaceState::confirmed,
		      "the landing confirms display 1");
		check(places.at[0].state == RgfxPlaceState::guessed, "display 0 not yet");
		r = cross(&places, &c, desk, 0, 1910, 700, &now);
		check(r.arrived && r.posts == 1 && r.display == 0 && r.x == 1910 && r.y == 700,
		      "back onto display 0");
		check(confirmedCount(places) == 2, "both confirmed after a round trip");
		RgfxGlobalRect rects[RGFX_DISPLAYS_MAX];
		bool placed[RGFX_DISPLAYS_MAX];
		rgfxPlacesConfirmed(places, rects, placed);
		check(same(rects[0], 0, 0, 1920, 1080) && same(rects[1], 1920, 0, 3840, 1080),
		      "the rectangles told are WindowServer's");
		// A landing near the corner shows no size: it does not confirm.
		RgfxDisplayPlaces fresh = {};
		RgfxCrossing f = {};
		r = cross(&fresh, &f, desk, 1, 2, 5, &now);
		check(r.arrived && r.confirmations == 0 && fresh.at[1].state == RgfxPlaceState::guessed,
		      "a landing at the corner confirms nothing");
	}

	// Display 1 left of display 0: the first post is held at display 0's right edge; the next,
	// beside display 0's left, lands on display 1, which is learned there, and confirmed by the
	// next crossing.
	{
		Desktop desk = twoSideBySide();
		desk.rects[1] = {-1920, 0, 0, 1080};
		RgfxDisplayPlaces places = {};
		RgfxCrossing c = {};
		Crossed r = cross(&places, &c, desk, 0, 900, 400, &now);
		check(r.arrived && places.at[0].state == RgfxPlaceState::confirmed, "display 0 confirmed");
		r = cross(&places, &c, desk, 1, 1915, 300, &now);
		check(r.arrived && r.display == 1 && r.posts == 2, "left: onto display 1 in two posts");
		check(same(places.at[1].rect, -1920, 0, 0, 1080) &&
		          places.at[1].state == RgfxPlaceState::learned && confirmedCount(places) == 1,
		      "display 1 learned left of display 0, not yet told");
		r = cross(&places, &c, desk, 0, 10, 300, &now);
		r = cross(&places, &c, desk, 1, 1915, 600, &now);
		check(r.arrived && r.posts == 1 && r.confirmations == 1, "the next crossing confirms it");
		check(same(places.at[1].rect, -1920, 0, 0, 1080) && confirmedCount(places) == 2,
		      "and it is told");
	}

	// Display 1 below display 0, and above it, offset sideways.
	{
		Desktop desk = twoSideBySide();
		desk.rects[1] = {0, 1080, 1920, 2160};
		RgfxDisplayPlaces places = {};
		RgfxCrossing c = {};
		cross(&places, &c, desk, 0, 500, 500, &now);
		Crossed r = cross(&places, &c, desk, 1, 800, 3, &now);
		check(r.arrived && r.display == 1 && r.posts <= kRgfxCrossingPosts,
		      "below: onto display 1");
		check(same(places.at[1].rect, 0, 1080, 1920, 2160), "learned below display 0");
		r = cross(&places, &c, desk, 1, 900, 40, &now); // already there: a landing that agrees
		check(places.at[1].state == RgfxPlaceState::confirmed, "and confirmed");

		desk.rects[1] = {300, -1080, 2220, 0};
		RgfxDisplayPlaces above = {};
		cross(&above, &c, desk, 0, 500, 500, &now);
		r = cross(&above, &c, desk, 1, 900, 1070, &now);
		check(r.arrived && r.display == 1, "above, offset: onto display 1");
		check(same(above.at[1].rect, 300, -1080, 2220, 0), "learned at its offset");
	}

	// HiDPI: display 1 on a 2x screen runs at half its pixels, guessed so; display 0 at 1x.
	{
		Desktop desk = twoSideBySide();
		desk.modes[1] = {2880, 1800, 240};
		desk.rects[1] = {1920, 0, 3360, 900};
		RgfxDisplayPlaces places = {};
		RgfxCrossing c = {};
		Crossed r = cross(&places, &c, desk, 1, 4, 1200, &now);
		check(r.arrived && r.posts == 1 && r.x == 4 && r.y == 1200,
		      "2x: onto display 1 at its pixel");
		check(places.at[1].state == RgfxPlaceState::confirmed &&
		          same(places.at[1].rect, 1920, 0, 3360, 900),
		      "confirmed at its size in points");
		// The same display at 1x, though its screen is 2x (the guest picked it): the guess is
		// wrong in size; landings learn it, and two of them size it.
		desk.rects[1] = {1920, 0, 4800, 1800};
		RgfxDisplayPlaces wrong = {};
		cross(&wrong, &c, desk, 0, 500, 500, &now);
		uint32_t crossings = 0;
		while (wrong.at[1].state != RgfxPlaceState::confirmed && crossings < 8) {
			cross(&wrong, &c, desk, 1, 100 + 300 * crossings, 200 + 150 * crossings, &now);
			cross(&wrong, &c, desk, 0, 600, 600, &now);
			crossings++;
		}
		check(wrong.at[1].state == RgfxPlaceState::confirmed &&
		          same(wrong.at[1].rect, 1920, 0, 4800, 1800),
		      "a size the guess got wrong is learned and confirmed");
		check(crossings <= 3, "within three crossings");
	}

	// A scaled mode, 1920 x 1080 pixels showing 2560 x 1440 points, which the guest's user picks:
	// each crossing lands on the display, and its size is learned to within a few points, as
	// pixels only approximate points at that scale.
	{
		Desktop desk = twoSideBySide();
		desk.rects[1] = {1920, 0, 4480, 1440};
		RgfxDisplayPlaces places = {};
		RgfxCrossing c = {};
		cross(&places, &c, desk, 0, 500, 500, &now);
		for (uint32_t i = 0; i < 4; i++) {
			const Crossed r = cross(&places, &c, desk, 1, 200 + 400 * i, 100 + 300 * i, &now);
			check(r.arrived && r.display == 1, "a scaled mode: onto display 1");
			cross(&places, &c, desk, 0, 600, 600, &now);
		}
		const RgfxGlobalRect &r = places.at[1].rect;
		check(r.minx >= 1918 && r.minx <= 1922 && r.miny >= -2 && r.miny <= 2 &&
		          r.maxx - r.minx >= 2550 && r.maxx - r.minx <= 2570 && r.maxy - r.miny >= 1434 &&
		          r.maxy - r.miny <= 1446,
		      "its size learned to within a few points");
	}

	// A rearrangement mid-session: both confirmed side by side, then display 1 moved left. The
	// first crossing after it lands wrong and posts again; the rectangle told moves once a
	// landing confirms it.
	{
		Desktop desk = twoSideBySide();
		RgfxDisplayPlaces places = {};
		RgfxCrossing c = {};
		cross(&places, &c, desk, 1, 10, 400, &now);
		cross(&places, &c, desk, 0, 1900, 400, &now);
		check(confirmedCount(places) == 2, "both confirmed");
		desk.rects[1] = {-1920, 200, 0, 1280};
		Crossed r = cross(&places, &c, desk, 1, 1915, 600, &now);
		check(r.arrived && r.display == 1 && r.posts == 2, "moved: onto display 1 in two posts");
		check(same(places.at[1].rect, -1920, 200, 0, 1280), "its new place learned");
		check(places.at[1].state == RgfxPlaceState::learned,
		      "and no longer told as it was confirmed");
		RgfxGlobalRect rects[RGFX_DISPLAYS_MAX];
		bool placed[RGFX_DISPLAYS_MAX];
		rgfxPlacesConfirmed(places, rects, placed);
		check(placed[0] && !placed[1], "display 0's stands");
		cross(&places, &c, desk, 0, 4, 500, &now);
		r = cross(&places, &c, desk, 1, 1900, 900, &now);
		check(r.posts == 1 && r.confirmations == 1, "confirmed by the next crossing");
		rgfxPlacesConfirmed(places, rects, placed);
		check(placed[1] && same(rects[1], -1920, 200, 0, 1280), "the new place is told");
	}

	// Three displays, display 2 never entered: never confirmed, so never told.
	{
		Desktop desk = twoSideBySide();
		desk.count = 3;
		desk.rects[2] = {3840, 0, 5760, 1080};
		desk.modes[2] = {1920, 1080, 120};
		RgfxDisplayPlaces places = {};
		RgfxCrossing c = {};
		cross(&places, &c, desk, 1, 10, 400, &now);
		cross(&places, &c, desk, 0, 1900, 400, &now);
		RgfxGlobalRect rects[RGFX_DISPLAYS_MAX];
		bool placed[RGFX_DISPLAYS_MAX];
		rgfxPlacesConfirmed(places, rects, placed);
		check(placed[0] && placed[1] && !placed[2], "a display never entered is not told");
		check(places.at[2].state == RgfxPlaceState::guessed, "it keeps its guess");
	}

	// The crossing's limits.
	{
		// No placement lands on the display (it is not where any is tried): at most
		// kRgfxCrossingPosts posts, and the cursor left where the last one put it.
		Desktop desk = twoSideBySide();
		desk.rects[1] = {1920, 5000, 3840, 6080}; // apart, as WindowServer never places one
		RgfxDisplayPlaces places = {};
		RgfxCrossing c = {};
		cross(&places, &c, desk, 0, 100, 100, &now);
		Crossed r = cross(&places, &c, desk, 1, 10, 10, &now);
		check(!r.arrived && r.posts <= kRgfxCrossingPosts, "posts at most kRgfxCrossingPosts");
		// The hold: no post again past it.
		RgfxDisplayPlaces slow = {};
		desk.rects[1] = {-1920, 0, 0, 1080};
		cross(&slow, &c, desk, 0, 100, 100, &now);
		r = cross(&slow, &c, desk, 1, 1910, 300, &now, kRgfxCrossingHoldNs);
		check(!r.arrived && r.posts == 1, "past the hold, no post again");
		// A landing that never comes: the crossing ends after kRgfxLandingWaitNs; a landing after
		// that is no crossing's.
		RgfxCrossing w = {};
		rgfxPlacesFit(&slow, desk.modes);
		rgfxCrossingStart(&w, 1, 5, 5, now);
		int32_t gx = 0, gy = 0;
		check(rgfxCrossingNext(slow, &w, now, &gx, &gy), "a post");
		check(!rgfxCrossingNext(slow, &w, now + 1000, &gx, &gy) && w.active,
		      "none while one is out");
		check(!rgfxCrossingExpired(&w, now + kRgfxLandingWaitNs - 1), "waited for");
		check(rgfxCrossingExpired(&w, now + kRgfxLandingWaitNs) && !w.active && !w.waiting,
		      "then given up, the crossing over");
		bool confirmed = true;
		check(rgfxCrossingLanded(&slow, &w, 1, 5, 5, &confirmed) == RgfxCrossingStep::none &&
		          !confirmed,
		      "a late landing is no crossing's");
		// A new crossing while a post is out: that post's landing is taken first, for its own
		// point, and the new crossing posts after it.
		RgfxDisplayPlaces two = {};
		const Desktop side = twoSideBySide();
		rgfxPlacesFit(&two, side.modes);
		RgfxCrossing n = {};
		rgfxCrossingStart(&n, 1, 3, 500, now);
		check(rgfxCrossingNext(two, &n, now, &gx, &gy) && gx == 1923, "a post onto display 1");
		rgfxCrossingStart(&n, 0, 700, 500, now + 1000);
		check(!rgfxCrossingNext(two, &n, now + 1000, &gx, &gy), "waits for the post out");
		check(rgfxCrossingLanded(&two, &n, 1, 3, 500, &confirmed) == RgfxCrossingStep::again &&
		          confirmed,
		      "whose landing confirms display 1, and the new crossing goes on");
		check(rgfxCrossingNext(two, &n, now + 2000, &gx, &gy) && gx == 700 && gy == 500,
		      "onto display 0");
	}

	std::printf("display_places_test: %d checks passed\n", checks);
	return 0;
}
