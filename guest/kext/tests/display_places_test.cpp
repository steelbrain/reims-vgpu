// The native test of RgfxDisplayPlaces.hpp, against a model of WindowServer: each display a
// rectangle of the global space, a posted point held at the edge of the display the cursor is on
// where it lies past every display, and the cursor shown on the display holding it at its pixel,
// where it moved. Displays side by side, left, below and above, one at 2x, one in a scaled mode,
// a rearrangement mid-session, a display never entered, the tablet's move while a post is out,
// the cursor's image over an edge, a post held where the cursor was, a landing past the host's
// hold, and the crossing's limits: its posts, a slow guest and a landing that never comes.
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

// WindowServer, as the kext sees it: each display's rectangle and mode, and the display the cursor
// is on.
struct Desktop {
	RgfxGlobalRect rects[RGFX_DISPLAYS_MAX];
	RgfxPlaceMode modes[RGFX_DISPLAYS_MAX];
	uint32_t count;
	mutable uint32_t on;
	mutable bool shown; // the cursor's pixel on `on`, once shown
	mutable int32_t ax, ay;

	// The display a point posted shows the cursor on, and its pixel there: the display holding
	// it, or, where none does, the one the cursor is on, the point held at its edge. False where
	// the cursor did not move, which shows nothing.
	bool land(int32_t gx, int32_t gy, uint32_t *display, int32_t *x, int32_t *y) const
	{
		uint32_t d = on < count ? on : 0;
		for (uint32_t i = 0; i < count; i++)
			if (gx >= rects[i].minx && gx < rects[i].maxx && gy >= rects[i].miny &&
			    gy < rects[i].maxy)
				d = i;
		const RgfxGlobalRect &r = rects[d];
		const int32_t cx = gx < r.minx ? r.minx : gx > r.maxx - 1 ? r.maxx - 1 : gx;
		const int32_t cy = gy < r.miny ? r.miny : gy > r.maxy - 1 ? r.maxy - 1 : gy;
		*display = d;
		*x = static_cast<int32_t>(static_cast<int64_t>(cx - r.minx) * modes[d].width /
		                          (r.maxx - r.minx));
		*y = static_cast<int32_t>(static_cast<int64_t>(cy - r.miny) * modes[d].height /
		                          (r.maxy - r.miny));
		const bool moved = !shown || d != on || *x != ax || *y != ay;
		on = d;
		ax = *x;
		ay = *y;
		shown = true;
		return moved;
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

// The kext's crossing onto `display` at pixel `x`, `y`, each post's move shown `step` ns after it;
// a post that shows nothing, or is held, waited for until its deadline.
Crossed cross(RgfxDisplayPlaces *places, RgfxCrossing *c, const Desktop &desk, uint32_t display,
              int32_t x, int32_t y, uint64_t *now, uint64_t step = 16000000)
{
	Crossed out = {};
	rgfxPlacesFit(places, desk.modes);
	rgfxCrossingStart(c, display, x, y, *now);
	int32_t gx = 0, gy = 0;
	while (rgfxCrossingNext(*places, c, *now, &gx, &gy)) {
		out.posts++;
		uint32_t d = 0;
		int32_t lx = 0, ly = 0;
		RgfxCrossingStep s = RgfxCrossingStep::none;
		if (desk.land(gx, gy, &d, &lx, &ly)) {
			*now += step;
			bool confirmed = false;
			s = rgfxCrossingLanded(places, c, d, lx, ly, *now, &confirmed);
			out.confirmations += confirmed;
		}
		if (s == RgfxCrossingStep::none || s == RgfxCrossingStep::held) {
			check(c->waiting, "a post that shows nothing, or is held, is waited for");
			*now = rgfxCrossingDeadline(*c);
			RgfxCursorsShown shown = {};
			if (desk.shown && desk.on < RGFX_DISPLAYS_MAX)
				shown.at[desk.on] = {true, desk.ax, desk.ay, c->postSeq + 1};
			s = rgfxCrossingWaited(*places, c, *now, shown);
		}
		if (s == RgfxCrossingStep::arrived) {
			out.arrived = true;
			break;
		}
		check(s == RgfxCrossingStep::again, "the cursor not on the display posts again");
	}
	out.display = desk.on;
	out.x = desk.ax;
	out.y = desk.ay;
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

// The cursor shown on no display.
const RgfxCursorsShown kNoCursor = {};

// The cursor shown on display `d` alone, at pixel `x`, `y`, by a MOVE_CURSOR numbered `seq` (by
// default after any post, whose crossing's postSeq stays 0).
RgfxCursorsShown shownOn(uint32_t d, int32_t x, int32_t y, uint64_t seq = 1)
{
	RgfxCursorsShown s = {};
	s.at[d] = {true, x, y, seq};
	return s;
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
	// beside display 0's left, lands on display 1 where that placement puts it, which confirms
	// the placement as its rectangle.
	{
		Desktop desk = twoSideBySide();
		desk.rects[1] = {-1920, 0, 0, 1080};
		RgfxDisplayPlaces places = {};
		RgfxCrossing c = {};
		Crossed r = cross(&places, &c, desk, 0, 900, 400, &now);
		check(r.arrived && places.at[0].state == RgfxPlaceState::confirmed, "display 0 confirmed");
		r = cross(&places, &c, desk, 1, 1915, 300, &now);
		check(r.arrived && r.display == 1 && r.posts == 2 && r.x == 1915 && r.y == 300,
		      "left: onto display 1 at its pixel in two posts");
		check(same(places.at[1].rect, -1920, 0, 0, 1080) && r.confirmations == 1 &&
		          confirmedCount(places) == 2,
		      "display 1 confirmed left of display 0, and told");
		check(same(places.at[0].rect, 0, 0, 1920, 1080) &&
		          places.at[0].state == RgfxPlaceState::confirmed,
		      "display 0, where the first post was held, keeps its place");
		cross(&places, &c, desk, 0, 10, 300, &now);
		r = cross(&places, &c, desk, 1, 1915, 600, &now);
		check(r.arrived && r.posts == 1, "the next crossing takes one post");
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
		check(same(places.at[1].rect, 0, 1080, 1920, 2160) &&
		          places.at[1].state == RgfxPlaceState::confirmed,
		      "confirmed below display 0");

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
		check(same(places.at[0].rect, 0, 0, 1920, 1080), "display 0 keeps its place");
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

	// A post out while the host's tablet moves the cursor on the display it is on (its hold over,
	// a slow guest's landing not yet shown): the tablet's pixel is no landing, and the display it
	// showed on keeps its place.
	{
		Desktop desk = twoSideBySide();
		RgfxDisplayPlaces places = {};
		RgfxCrossing c = {};
		cross(&places, &c, desk, 1, 10, 400, &now);
		cross(&places, &c, desk, 0, 1900, 400, &now);
		cross(&places, &c, desk, 1, 500, 500, &now);
		check(confirmedCount(places) == 2, "both confirmed, the cursor on display 1");
		rgfxCrossingStart(&c, 0, 756, 472, now);
		int32_t gx = 0, gy = 0;
		check(rgfxCrossingNext(places, &c, now, &gx, &gy) && gx == 756 && gy == 472,
		      "a post onto display 0");
		bool confirmed = true;
		check(rgfxCrossingLanded(&places, &c, 1, 756, 472, now + 100000000, &confirmed) ==
		              RgfxCrossingStep::none &&
		          c.waiting && !confirmed,
		      "the tablet's move on display 1 is no landing");
		check(same(places.at[1].rect, 1920, 0, 3840, 1080) &&
		          places.at[1].state == RgfxPlaceState::confirmed,
		      "and display 1 keeps its place");
		check(rgfxCrossingLanded(&places, &c, 0, 756, 472, now + 120000000, &confirmed) ==
		              RgfxCrossingStep::arrived &&
		          confirmed && c.landedAfter == 120000000,
		      "display 0's is the landing");
		now += 1000000000;

		// Rearranged, display 1 left: the tablet's move on display 0, then the post held at its
		// right edge, far past it, which posts again at once.
		desk.rects[1] = {-1920, 0, 0, 1080};
		rgfxCrossingStart(&c, 1, 500, 500, now);
		check(rgfxCrossingNext(places, &c, now, &gx, &gy) && gx == 2420, "a post right of 0");
		check(rgfxCrossingLanded(&places, &c, 0, 500, 500, now + 100000000, &confirmed) ==
		              RgfxCrossingStep::none &&
		          c.waiting,
		      "the tablet's move on display 0 is no landing");
		check(rgfxCrossingLanded(&places, &c, 0, 1919, 500, now + 110000000, &confirmed) ==
		              RgfxCrossingStep::again &&
		          !c.waiting,
		      "held at display 0's right edge, far past it: posted again");
		check(rgfxCrossingNext(places, &c, now + 110000000, &gx, &gy) && gx == -1420 && gy == 500,
		      "left of display 0");
		check(rgfxCrossingLanded(&places, &c, 1, 500, 500, now + 130000000, &confirmed) ==
		              RgfxCrossingStep::arrived &&
		          confirmed && same(places.at[1].rect, -1920, 0, 0, 1080),
		      "onto display 1, confirmed there");
		check(same(places.at[0].rect, 0, 0, 1920, 1080) &&
		          places.at[0].state == RgfxPlaceState::confirmed,
		      "display 0 keeps its place");
		now += 1000000000;
	}

	// The cursor near display 0's right edge shows on display 1 too, its image over the edge: its
	// hot spot past display 1's left is no landing; at display 1's first pixel, where the post's
	// point is held at that edge, the crossing waits for display 0's landing.
	{
		const Desktop side = twoSideBySide();
		RgfxDisplayPlaces places = {};
		RgfxCrossing c = {};
		rgfxPlacesFit(&places, side.modes);
		rgfxCrossingStart(&c, 0, 1915, 600, now);
		int32_t gx = 0, gy = 0;
		check(rgfxCrossingNext(places, &c, now, &gx, &gy) && gx == 1915 && gy == 600, "a post");
		bool confirmed = true;
		check(rgfxCrossingLanded(&places, &c, 1, -5, 600, now, &confirmed) ==
		              RgfxCrossingStep::none &&
		          !confirmed && c.waiting && c.active,
		      "display 1's image over the edge is no landing");
		check(rgfxCrossingLanded(&places, &c, 1, 3, 1080, now, &confirmed) ==
		              RgfxCrossingStep::none &&
		          c.waiting,
		      "nor one past its bottom edge");
		check(rgfxCrossingLanded(&places, &c, 1, 0, 600, now + 1000, &confirmed) ==
		              RgfxCrossingStep::held &&
		          c.waiting && c.held && rgfxCrossingDeadline(c) == now + 1000 + kRgfxSpillWaitNs,
		      "its hot spot at display 1's first pixel holds the crossing a while");
		check(rgfxCrossingLanded(&places, &c, 1, 0, 600, now + 2000, &confirmed) ==
		              RgfxCrossingStep::none,
		      "once");
		check(rgfxCrossingLanded(&places, &c, 0, 1915, 600, now + 3000, &confirmed) ==
		              RgfxCrossingStep::arrived &&
		          confirmed && !c.waiting,
		      "display 0's is the landing");
		check(same(places.at[1].rect, 1920, 0, 3840, 1080) &&
		          places.at[1].state == RgfxPlaceState::guessed,
		      "display 1 learned nothing");
		now += 1000000000;
	}

	// Rearranged, display 1 left: a post just past display 0's right edge is held there, the
	// crossing waits kRgfxSpillWaitNs for a landing on display 1, then posts again.
	{
		Desktop desk = twoSideBySide();
		RgfxDisplayPlaces places = {};
		RgfxCrossing c = {};
		cross(&places, &c, desk, 1, 10, 400, &now);
		cross(&places, &c, desk, 0, 1000, 500, &now);
		desk.rects[1] = {-1920, 0, 0, 1080};
		rgfxCrossingStart(&c, 1, 3, 500, now);
		int32_t gx = 0, gy = 0;
		check(rgfxCrossingNext(places, &c, now, &gx, &gy) && gx == 1923, "a post right of 0");
		bool confirmed = true;
		check(rgfxCrossingLanded(&places, &c, 0, 1919, 500, now + 10000000, &confirmed) ==
		              RgfxCrossingStep::held,
		      "held at display 0's edge, within a cursor's reach: waited on");
		check(rgfxCrossingWaited(places, &c, now + 10000000 + kRgfxSpillWaitNs - 1, kNoCursor) ==
		              RgfxCrossingStep::none &&
		          c.waiting,
		      "for kRgfxSpillWaitNs");
		check(rgfxCrossingWaited(places, &c, now + 10000000 + kRgfxSpillWaitNs, kNoCursor) ==
		              RgfxCrossingStep::again &&
		          !c.waiting,
		      "then posted again");
		check(rgfxCrossingNext(places, &c, now + 70000000, &gx, &gy) && gx == -1917,
		      "left of display 0");
		check(rgfxCrossingLanded(&places, &c, 1, 3, 500, now + 80000000, &confirmed) ==
		              RgfxCrossingStep::arrived &&
		          confirmed && same(places.at[1].rect, -1920, 0, 0, 1080),
		      "onto display 1, confirmed there");
		now += 1000000000;
	}

	// Rearranged, display 1 left, the cursor at display 0's right edge (a captured pointer pushed
	// past its window's edge): the post right of display 0 is held at the very pixel the cursor
	// is at, which shows nothing; after kRgfxLandingWaitNs the crossing posts again.
	{
		Desktop desk = twoSideBySide();
		RgfxDisplayPlaces places = {};
		RgfxCrossing c = {};
		cross(&places, &c, desk, 1, 10, 400, &now);
		cross(&places, &c, desk, 0, 1919, 600, &now);
		desk.rects[1] = {-1920, 0, 0, 1080};
		const uint64_t before = now;
		Crossed r = cross(&places, &c, desk, 1, 0, 600, &now);
		check(r.arrived && r.posts == 2 && r.display == 1 && r.x == 0 && r.y == 600,
		      "onto display 1 at its first pixel in two posts");
		check(now - 1000000000 - before >= kRgfxLandingWaitNs, "after the first post's wait");
		check(same(places.at[1].rect, -1920, 0, 0, 1080) &&
		          places.at[1].state == RgfxPlaceState::confirmed,
		      "the placement it landed by is confirmed, and told");
	}

	// The crossing's limits.
	{
		// No placement lands on the display (it is not where any is tried): at most
		// kRgfxCrossingPosts posts, and the cursor left where the last one put it.
		Desktop desk = twoSideBySide();
		desk.rects[1] = {1920, 5000, 3840, 6080}; // apart, as WindowServer never places one
		desk.on = 0;
		RgfxDisplayPlaces places = {};
		RgfxCrossing c = {};
		cross(&places, &c, desk, 0, 100, 100, &now);
		Crossed r = cross(&places, &c, desk, 1, 10, 10, &now);
		check(!r.arrived && r.posts == kRgfxCrossingPosts && r.display == 0,
		      "posts kRgfxCrossingPosts, the cursor left on display 0");
		// A slow guest, each move shown 150 ms after its post (past the host's hold): the
		// crossing still posts again, and arrives.
		RgfxDisplayPlaces slow = {};
		desk.rects[1] = {-1920, 0, 0, 1080};
		cross(&slow, &c, desk, 0, 100, 100, &now);
		r = cross(&slow, &c, desk, 1, 1910, 300, &now, 150000000);
		check(r.arrived && r.posts == 2 && r.display == 1, "a slow guest crosses in two posts");
		// A post that never shows the cursor: posted again after each kRgfxLandingWaitNs, at
		// most kRgfxCrossingPosts, then over; a landing after that is no crossing's.
		RgfxCrossing w = {};
		rgfxPlacesFit(&slow, desk.modes);
		rgfxCrossingStart(&w, 1, 5, 5, now);
		int32_t gx = 0, gy = 0;
		check(rgfxCrossingNext(slow, &w, now, &gx, &gy), "a post");
		check(!rgfxCrossingNext(slow, &w, now + 1000, &gx, &gy) && w.active,
		      "none while one is out");
		check(rgfxCrossingWaited(slow, &w, now + kRgfxLandingWaitNs - 1, kNoCursor) ==
		              RgfxCrossingStep::none &&
		          w.waiting,
		      "waited for");
		uint64_t t = now + kRgfxLandingWaitNs;
		uint32_t posts = 1;
		check(rgfxCrossingWaited(slow, &w, t, kNoCursor) == RgfxCrossingStep::again && !w.waiting &&
		          w.active,
		      "then posted again");
		while (rgfxCrossingNext(slow, &w, t, &gx, &gy)) {
			posts++;
			t = rgfxCrossingDeadline(w);
			check(rgfxCrossingWaited(slow, &w, t, kNoCursor) == RgfxCrossingStep::again, "and again");
		}
		check(posts == kRgfxCrossingPosts && !w.active && !w.waiting,
		      "at most kRgfxCrossingPosts, then over");
		bool confirmed = true;
		check(rgfxCrossingLanded(&slow, &w, 1, 5, 5, t, &confirmed) == RgfxCrossingStep::none &&
		          !confirmed,
		      "a late landing is no crossing's");
		// A new crossing while a post is out: that post's landing is taken first, for its own
		// display, and the new crossing posts after it.
		RgfxDisplayPlaces two = {};
		const Desktop side = twoSideBySide();
		rgfxPlacesFit(&two, side.modes);
		RgfxCrossing n = {};
		rgfxCrossingStart(&n, 1, 3, 500, now);
		check(rgfxCrossingNext(two, &n, now, &gx, &gy) && gx == 1923, "a post onto display 1");
		rgfxCrossingStart(&n, 0, 700, 500, now + 1000);
		check(!rgfxCrossingNext(two, &n, now + 1000, &gx, &gy), "waits for the post out");
		check(rgfxCrossingLanded(&two, &n, 1, 3, 500, now + 2000, &confirmed) ==
		              RgfxCrossingStep::again &&
		          confirmed,
		      "whose landing confirms display 1, and the new crossing goes on");
		check(rgfxCrossingNext(two, &n, now + 2000, &gx, &gy) && gx == 700 && gy == 500,
		      "onto display 0");
		// A crossing told while the cursor is on its display already, at the post's pixel (the
		// host had not yet seen it there, say for the image's part over the edge on the display
		// beside it): the post shows nothing, and the display last showed the cursor there, so
		// the crossing is over, with no post elsewhere; shown at another pixel, it is not there.
		RgfxCrossing a = {};
		rgfxCrossingStart(&a, 1, 0, 500, now + 3000);
		check(rgfxCrossingNext(two, &a, now + 3000, &gx, &gy) && gx == 1920, "a post onto display 1");
		check(rgfxCrossingWaited(two, &a, now + 3000 + kRgfxLandingWaitNs, shownOn(1, 0, 500)) ==
		              RgfxCrossingStep::arrived &&
		          !a.active && !a.waiting && a.posts == 1,
		      "the cursor already there: over in one post");
		rgfxCrossingStart(&a, 1, 0, 500, now + 4000);
		check(rgfxCrossingNext(two, &a, now + 4000, &gx, &gy), "again a post onto display 1");
		check(rgfxCrossingWaited(two, &a, now + 4000 + kRgfxLandingWaitNs, shownOn(1, 900, 500)) ==
		              RgfxCrossingStep::again &&
		          a.active,
		      "shown elsewhere on it: not there, posted again");
	}

	// A post that shows nothing by its deadline with no display showing the cursor where its point
	// lies (measured on 27.0.1: the first post 2 s after the displays were rearranged): its landing
	// may yet come, at the pixel the next post aims at too. Both displays confirmed side by side,
	// display 1 then moved left, the cursor on display 0; a crossing onto display 0 at 400,300.
	{
		Desktop desk = twoSideBySide();
		RgfxDisplayPlaces places = {};
		RgfxCrossing c = {};
		cross(&places, &c, desk, 1, 10, 400, &now);
		cross(&places, &c, desk, 0, 959, 174, &now);
		desk.rects[1] = {-1920, 0, 0, 1080};
		bool confirmed = true;
		int32_t gx = 0, gy = 0;
		for (int late = 0; late < 2; late++) {
			rgfxCrossingStart(&c, 0, 400, 300, now);
			check(rgfxCrossingNext(places, &c, now, &gx, &gy) && gx == 400 && gy == 300,
			      "a post at display 0's own place");
			check(rgfxCrossingWaited(places, &c, now + kRgfxLandingWaitNs, shownOn(0, 959, 174)) ==
			              RgfxCrossingStep::again &&
			          c.unsure,
			      "nothing shown where it lies: posted again, unsure");
			check(rgfxCrossingNext(places, &c, now + kRgfxLandingWaitNs, &gx, &gy) && gx == 4240,
			      "beside display 1, right of it");
			// Shown together, the cursor held at display 0's edge; or the first post's landing,
			// late, at the very pixel both aim at.
			const int32_t x = late ? 400 : 1919;
			check(rgfxCrossingLanded(&places, &c, 0, x, 300, now + kRgfxLandingWaitNs + 5000000,
			                         &confirmed) == RgfxCrossingStep::arrived &&
			          !confirmed,
			      "on display 0: arrived, nothing confirmed");
			check(same(places.at[0].rect, 0, 0, 1920, 1080) &&
			          places.at[0].state == RgfxPlaceState::confirmed,
			      "display 0 keeps its place");
			now += 1000000000;
		}
		// A landing held at the display's edge, on the axis of the placement it was posted by:
		// the point lay past the display there, so the rectangle keeps its own on that axis.
		check(!rgfxPlacesLanded(&places, 0, {4240, 300, 1919, 300}, {3840, 0, 5760, 1080},
		                        false) &&
		          same(places.at[0].rect, 0, 0, 1920, 1080) &&
		          places.at[0].state == RgfxPlaceState::confirmed,
		      "held at an edge: the axis keeps the display's own place");
		// The next crossing is sure again.
		rgfxCrossingStart(&c, 1, 1910, 500, now);
		check(!c.unsure, "a new crossing is sure");
	}

	// A display that showed the cursor where the post's point is held at before the post went out
	// shows nothing of it (measured on 15.6.1 under TCG: a captured drag left the cursor at display
	// 1's left edge, and the crossing onto display 0 posted at display 0's own place; at the
	// deadline, that post not yet shown, display 1 still showed the cursor there). Both displays
	// confirmed side by side; a crossing onto display 0 at 1900,300; the cursor on display 1 at
	// pixel 0,300, where that post's point is held at display 1's left edge.
	{
		Desktop desk = twoSideBySide();
		RgfxDisplayPlaces places = {};
		RgfxCrossing c = {};
		cross(&places, &c, desk, 1, 10, 400, &now);
		cross(&places, &c, desk, 0, 959, 174, &now);
		check(confirmedCount(places) == 2, "both confirmed");
		bool confirmed = true;
		int32_t gx = 0, gy = 0;
		for (int fresh = 0; fresh < 2; fresh++) {
			RgfxDisplayPlaces p = places;
			rgfxCrossingStart(&c, 0, 1900, 300, now);
			check(rgfxCrossingNext(p, &c, now, &gx, &gy) && gx == 1900 && gy == 300,
			      "a post at display 0's own place");
			c.postSeq = 40;
			const RgfxCrossingStep s =
			    rgfxCrossingWaited(p, &c, now + kRgfxLandingWaitNs, shownOn(1, 0, 300, 40 + fresh));
			check(s == RgfxCrossingStep::again && c.unsure == !fresh,
			      fresh ? "shown there after the post: its outcome, sure"
			            : "shown there before the post: nothing of it, unsure");
			check(rgfxCrossingNext(p, &c, now + kRgfxLandingWaitNs, &gx, &gy) && gx == 5740,
			      "posted again beside display 1, right of it");
			c.postSeq = 41 + fresh;
			// The first post's landing, late, at the pixel both aim at.
			check(rgfxCrossingLanded(&p, &c, 0, 1900, 300, now + kRgfxLandingWaitNs + 90000000,
			                         &confirmed) == RgfxCrossingStep::arrived,
			      "on display 0: arrived");
			if (!fresh)
				check(!confirmed && same(p.at[0].rect, 0, 0, 1920, 1080) &&
				          p.at[0].state == RgfxPlaceState::confirmed,
				      "unsure: display 0 keeps its confirmed place");
			else
				check(same(p.at[0].rect, 3840, 0, 5760, 1080),
				      "sure: taken for the second post's landing, display 0 placed right of 1");
			now += 1000000000;
		}
	}

	// A landing kRgfxCrossingHoldNs or more after the POINTER_OVER, once the host passes its
	// pointer's positions on to the tablet again, may show the tablet's move (measured on 15.6.1
	// under TCG, at two displays side by side in a 2x mode, 2016 x 1260 pixels showing 1008 x 630
	// points: a captured drag back onto display 0 posted at its right edge, 1007,64, and the cursor
	// showed 130 ms later at pixel 1962,126, 26 points short in the drag's direction). Both
	// displays confirmed, display 0's last landing 969,315 at pixel 1938,630.
	{
		Desktop desk = {};
		desk.count = 2;
		desk.rects[0] = {0, 0, 1008, 630};
		desk.rects[1] = {1008, 0, 2016, 630};
		desk.modes[0] = {2016, 1260, 240};
		desk.modes[1] = {2016, 1260, 240};
		RgfxDisplayPlaces places = {};
		RgfxCrossing c = {};
		cross(&places, &c, desk, 1, 8, 128, &now);
		cross(&places, &c, desk, 0, 1938, 630, &now);
		check(confirmedCount(places) == 2 && places.at[0].last.lx == 1938,
		      "both confirmed, display 0's last landing at 1938,630");
		bool confirmed = true;
		int32_t gx = 0, gy = 0;
		for (int early = 0; early < 2; early++) {
			RgfxDisplayPlaces p = places;
			rgfxCrossingStart(&c, 0, 2014, 128, now);
			check(rgfxCrossingNext(p, &c, now + 2000000, &gx, &gy) && gx == 1007 && gy == 64,
			      "a post at display 0's right edge");
			const uint64_t at = early ? now + kRgfxCrossingHoldNs - 1 : now + kRgfxCrossingHoldNs;
			check(rgfxCrossingLanded(&p, &c, 0, 1962, 126, at, &confirmed) ==
			              RgfxCrossingStep::arrived &&
			          !confirmed,
			      "shown 26 points short: arrived, nothing confirmed");
			if (!early) {
				check(same(p.at[0].rect, 0, 0, 1008, 630) &&
				          p.at[0].state == RgfxPlaceState::confirmed && p.at[0].last.lx == 1938,
				      "past the hold: display 0 keeps its place, its size and its last landing");
				check(rgfxPlacesLanded(&p, 0, {967, 279, 1934, 558}, p.at[0].rect, false) &&
				          same(p.at[0].rect, 0, 0, 1008, 630),
				      "and the next landing finds it so");
				// One that agrees with the placement it was posted by still confirms it.
				p.at[0].state = RgfxPlaceState::learned;
				check(rgfxPlacesLanded(&p, 0, {967, 279, 1934, 558}, p.at[0].rect, true) &&
				          p.at[0].state == RgfxPlaceState::confirmed,
				      "past the hold, a landing that agrees confirms");
			} else {
				check(p.at[0].state == RgfxPlaceState::learned &&
				          same(p.at[0].rect, 29, 2, 1034, 630),
				      "within the hold: taken for the post's, it sizes and moves display 0");
			}
			now += 1000000000;
		}
	}

	std::printf("display_places_test: %d checks passed\n", checks);
	return 0;
}
