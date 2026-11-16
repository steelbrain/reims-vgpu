// RgfxDisplayPlaces: where the arm64e kext holds each display to lie in WindowServer's global
// space, learned from its own moves of the cursor, and the moves of a crossing (design.md,
// "Multiple displays", "The pointer across displays"; ReimsGraphicsPointer.cpp).
//
// An arm64 guest's displays are IOMobileFramebuffers, which WindowServer tells nothing of where it
// placed them, and it registers none of them with IOHIDSystem: measured on arm64 macOS 27.0.1 at
// two displays in three arrangements, IOHIDSystem's screen count stayed 0 and its work bounds
// 0,0 to 1920,1080 (lane a64ptr, potluck job m3-max-35345). What the kext does see is the cursor.
// A move it posts at a point of the global space (IOHIDSystem::extPostEvent, as the x86-64 kext's)
// puts WindowServer's cursor at that point, and the swap that shows it there names the display and
// the pixel (its MOVE_CURSOR): in the same job, posts at 2880,540, -960,540 and 960,1620 showed the
// cursor on display 1 at 960,540, with display 1 right of, left of and below display 0. A point
// past every display is held at the nearest display's edge (2880,540 with one 1920-wide display
// showed the cursor at its right edge, on 14.6.1, job m3-max-35287).
//
// So each driven display's rectangle is held here, and each landing of a move the kext posted
// checks it:
//  - a display starts from a guess, WindowServer's own arrangement of new displays: side by side
//    in their order, tops at 0 (measured on 27.0.1's first boot at two displays), each its mode's
//    size at 1x on a 1x screen and half that above 1x, as arm64's WindowServer runs a display
//    HiDPI above 1x (design.md, "The display's density"); a display's new mode keeps its corner
//    and takes the guessed size;
//  - a landing whose pixel lies where the rectangle puts the posted point, on both axes, with the
//    pixel far enough from the corner to show a wrong size, confirms the rectangle, and only a
//    confirmed one is told the host (DISPLAY_BOUNDS);
//  - a landing that disagrees moves the rectangle's corner on each axis where the pixel lies
//    inside the display, so the point was not held at an edge, and leaves an axis held at an edge
//    as it was; the rectangle so moved is learned, and confirmed by a later landing that agrees;
//  - two landings on a display, the second disagreeing with its size, each inside it at the size
//    held and at the size they give, and 64 pixels apart or more, size it by the points between
//    them over the pixels between them, the other axis alike.
//
// A crossing is the moves that put the cursor on the display the host's pointer is over, for its
// pixel there: first at that pixel of the display's rectangle; and, where the cursor landed on
// another display, at that pixel of the display placed beside each other display in turn (right,
// left, below and above it, top or left edges aligned), each placement that overlaps no learned
// display's. A crossing ends once the cursor lands on its display, wherever on it (the host's
// tablet then puts it at its pixel, scaled over the display the cursor is on), and with the
// cursor where it is after kRgfxCrossingPosts posts, or once a post has waited
// kRgfxLandingWaitNs for its landing, or once the host's hold of the pointer's positions for that
// display is over (kRgfxCrossingHoldNs, the device's CROSSING_HOLD_NS), so a guest never sees the
// cursor jump on.
//
// Header-only and free of the kernel's headers, so its native test (tests/display_places_test.cpp)
// builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_DISPLAY_PLACES_HPP_
#define RGFX_DISPLAY_PLACES_HPP_

#include <stdint.h>

#include "RgfxPointerOver.hpp"
#include "reims_graphics_wire.h"

// A crossing's posts: the first, and at most three more.
constexpr uint32_t kRgfxCrossingPosts = 4;
// How long a crossing posts again after its first: the host's hold of the positions for the
// display (CROSSING_HOLD_NS).
constexpr uint64_t kRgfxCrossingHoldNs = 100000000;
// How long a post waits for its landing before its crossing ends.
constexpr uint64_t kRgfxLandingWaitNs = 250000000;
// How far from the rectangle's corner, in pixels on either axis, a landing shows its size.
constexpr int32_t kRgfxPlaceProof = 16;
// How far apart two landings are, in pixels, to size a display by.
constexpr int32_t kRgfxPlaceSpan = 64;

enum class RgfxPlaceState : uint8_t {
	none,      // not driven
	guessed,   // as WindowServer places a new display
	learned,   // moved by a landing, not yet confirmed
	confirmed, // a landing agreed with it
};

// A driven display's mode, in pixels, and the scale of the host screen that shows it in 120ths
// (RgfxDisplayScale; 0 where the host told none). A width or height of 0 is a display not driven.
struct RgfxPlaceMode {
	uint32_t width;
	uint32_t height;
	uint32_t scale;
};

// A landing: the point of the global space posted and the pixel the cursor showed at.
struct RgfxPlaceLanding {
	int32_t gx;
	int32_t gy;
	int32_t lx;
	int32_t ly;
};

struct RgfxDisplayPlace {
	RgfxGlobalRect rect;
	uint32_t width; // the mode the rectangle is for
	uint32_t height;
	RgfxPlaceState state;
	bool landed; // `last` holds the latest landing on it
	RgfxPlaceLanding last;
};

// Each display's, by id.
struct RgfxDisplayPlaces {
	RgfxDisplayPlace at[RGFX_DISPLAYS_MAX];
};

// The size the guess gives a display of `mode`: its pixels at 1x, half of them above 1x.
inline void rgfxPlaceGuessSize(const RgfxPlaceMode &mode, int32_t *width, int32_t *height)
{
	const uint32_t s = mode.scale > 120 ? 2 : 1;
	*width = static_cast<int32_t>(mode.width / s ? mode.width / s : 1);
	*height = static_cast<int32_t>(mode.height / s ? mode.height / s : 1);
}

// Each display's rectangle brought to the displays' modes (`modes`, by id): a display not driven
// forgotten, a driven one with none guessed beside the display before it (or at 0,0), and one
// whose mode changed guessed at its corner.
inline void rgfxPlacesFit(RgfxDisplayPlaces *places, const RgfxPlaceMode *modes)
{
	for (uint32_t d = 0; d < RGFX_DISPLAYS_MAX; d++) {
		RgfxDisplayPlace &p = places->at[d];
		const RgfxPlaceMode &m = modes[d];
		if (!m.width || !m.height) {
			p = {};
			continue;
		}
		if (p.state == RgfxPlaceState::none || (p.width == m.width && p.height == m.height))
			continue;
		int32_t w = 0, h = 0;
		rgfxPlaceGuessSize(m, &w, &h);
		p.rect.maxx = p.rect.minx + w;
		p.rect.maxy = p.rect.miny + h;
		p.width = m.width;
		p.height = m.height;
		p.state = RgfxPlaceState::guessed;
		p.landed = false;
	}
	int32_t x = 0, y = 0;
	for (uint32_t d = 0; d < RGFX_DISPLAYS_MAX; d++) {
		RgfxDisplayPlace &p = places->at[d];
		const RgfxPlaceMode &m = modes[d];
		if (!m.width || !m.height)
			continue;
		if (p.state == RgfxPlaceState::none) {
			int32_t w = 0, h = 0;
			rgfxPlaceGuessSize(m, &w, &h);
			p = {};
			p.rect = {x, y, x + w, y + h};
			p.width = m.width;
			p.height = m.height;
			p.state = RgfxPlaceState::guessed;
		}
		x = p.rect.maxx;
		y = p.rect.miny;
	}
}

// The point of the global space `pixel` of `pixels` lies at, on one axis of a display whose
// rectangle spans `span` points from `corner`.
inline int64_t rgfxPlacePoint(int64_t corner, int64_t span, int64_t pixels, int64_t pixel)
{
	return corner + pixel * span / pixels;
}

// Whether `pixel` of `pixels` lies inside a display `span` points across on that axis, off both
// its edges, where a point past the display is never held: its point a point in from each edge,
// and the pixel two in from the first and three from the last, as an edge's point lies on the
// last pixel or the one before it at any size up to 2x.
inline bool rgfxPlaceInside(int64_t span, int64_t pixels, int64_t pixel)
{
	const int64_t at = pixel * span / pixels;
	return pixel >= 2 && pixel <= pixels - 3 && at >= 1 && at <= span - 2;
}

inline int64_t rgfxPlaceAbs(int64_t v) { return v < 0 ? -v : v; }

// Display `display`'s rectangle checked by a landing: `landing`'s point posted, and the cursor
// shown on the display at its pixel. True where it confirmed the rectangle (which the host is
// then told, where it differs from the one last told).
inline bool rgfxPlacesLanded(RgfxDisplayPlaces *places, uint32_t display,
                             const RgfxPlaceLanding &landing)
{
	if (display >= RGFX_DISPLAYS_MAX)
		return false;
	RgfxDisplayPlace &p = places->at[display];
	if (p.state == RgfxPlaceState::none || !p.width || !p.height)
		return false;
	const int64_t pixels[2] = {p.width, p.height};
	const int64_t posted[2] = {landing.gx, landing.gy};
	const int64_t pixel[2] = {landing.lx, landing.ly};
	int64_t corner[2] = {p.rect.minx, p.rect.miny};
	int64_t span[2] = {static_cast<int64_t>(p.rect.maxx) - p.rect.minx,
	                   static_cast<int64_t>(p.rect.maxy) - p.rect.miny};
	auto agrees = [&](int a) {
		const int64_t tolerance = 1 + span[a] / pixels[a];
		return rgfxPlaceAbs(rgfxPlacePoint(corner[a], span[a], pixels[a], pixel[a]) - posted[a]) <=
		       tolerance;
	};
	// The size, from this landing and the one before it where both disagree with the size held:
	// by the axis they lie furthest apart on, the other scaled alike.
	if (p.landed && !(agrees(0) && agrees(1))) {
		const int64_t before[2] = {p.last.lx, p.last.ly};
		const int64_t beforeAt[2] = {p.last.gx, p.last.gy};
		const int a =
		    rgfxPlaceAbs(pixel[0] - before[0]) >= rgfxPlaceAbs(pixel[1] - before[1]) ? 0 : 1;
		const int64_t apart = pixel[a] - before[a];
		const int64_t moved = posted[a] - beforeAt[a];
		if (rgfxPlaceAbs(apart) >= kRgfxPlaceSpan && (moved > 0) == (apart > 0) && moved != 0) {
			const int64_t sized = (moved * pixels[a] * 2 + apart) / (apart * 2);
			const int b = 1 - a;
			const int64_t sizedOther = (sized * pixels[b] * 2 + pixels[a]) / (pixels[a] * 2);
			if (sized > 1 && sizedOther > 1 && rgfxPlaceAbs(sized - span[a]) > 1 &&
			    rgfxPlaceInside(span[a], pixels[a], pixel[a]) &&
			    rgfxPlaceInside(span[a], pixels[a], before[a]) &&
			    rgfxPlaceInside(sized, pixels[a], pixel[a]) &&
			    rgfxPlaceInside(sized, pixels[a], before[a])) {
				span[a] = sized;
				span[b] = sizedOther;
			}
		}
	}
	p.landed = true;
	p.last = landing;
	int agreed = 0, moved = 0;
	for (int a = 0; a < 2; a++) {
		if (agrees(a)) {
			agreed++;
			continue;
		}
		if (!rgfxPlaceInside(span[a], pixels[a], pixel[a]))
			continue; // held at an edge: this axis says nothing
		corner[a] += posted[a] - rgfxPlacePoint(corner[a], span[a], pixels[a], pixel[a]);
		moved++;
	}
	const bool resized = span[0] != static_cast<int64_t>(p.rect.maxx) - p.rect.minx ||
	                     span[1] != static_cast<int64_t>(p.rect.maxy) - p.rect.miny;
	p.rect = {static_cast<int32_t>(corner[0]), static_cast<int32_t>(corner[1]),
	          static_cast<int32_t>(corner[0] + span[0]), static_cast<int32_t>(corner[1] + span[1])};
	if (agreed == 2 && !resized) {
		if (landing.lx < kRgfxPlaceProof && landing.ly < kRgfxPlaceProof)
			return false;
		p.state = RgfxPlaceState::confirmed;
		return true;
	}
	if (moved || resized)
		p.state = RgfxPlaceState::learned;
	return false;
}

// Each confirmed display's rectangle (`placed` set), as the host is told them.
inline void rgfxPlacesConfirmed(const RgfxDisplayPlaces &places, RgfxGlobalRect *rects,
                                bool *placed)
{
	for (uint32_t d = 0; d < RGFX_DISPLAYS_MAX; d++) {
		placed[d] = places.at[d].state == RgfxPlaceState::confirmed;
		if (placed[d])
			rects[d] = places.at[d].rect;
	}
}

// The moves of a crossing onto `display`, for its pixel `x`, `y`.
struct RgfxCrossing {
	bool active;
	uint32_t display;
	int32_t x;
	int32_t y;
	uint32_t posts;     // posted for this crossing
	uint32_t candidate; // the next placement beside another display to try
	uint64_t startedAt;
	// The post out, whose landing is not yet seen: it may be an earlier crossing's.
	bool waiting;
	int32_t gx;
	int32_t gy;
	uint64_t postedAt;
};

// What a landing does to the crossing.
enum class RgfxCrossingStep {
	none,    // no post was out
	arrived, // the cursor is on the crossing's display: it is over
	again,   // on another display: post again (rgfxCrossingNext)
};

inline void rgfxCrossingStart(RgfxCrossing *c, uint32_t display, int32_t x, int32_t y, uint64_t now)
{
	c->active = true;
	c->display = display;
	c->x = x;
	c->y = y;
	c->posts = 0;
	c->candidate = 0;
	c->startedAt = now;
}

inline bool rgfxPlaceOverlaps(const RgfxGlobalRect &a, const RgfxGlobalRect &b)
{
	return a.minx < b.maxx && b.minx < a.maxx && a.miny < b.maxy && b.miny < a.maxy;
}

inline bool rgfxPlaceSame(const RgfxGlobalRect &a, const RgfxGlobalRect &b)
{
	return a.minx == b.minx && a.miny == b.miny && a.maxx == b.maxx && a.maxy == b.maxy;
}

// The next placement of the crossing's display beside another, from `c->candidate` on: for each
// other display, learned or confirmed ones first, in their order, its right, left, bottom and top
// sides. False where none is left.
inline bool rgfxCrossingCandidate(const RgfxDisplayPlaces &places, RgfxCrossing *c,
                                  RgfxGlobalRect *rect)
{
	const RgfxDisplayPlace &k = places.at[c->display];
	const int32_t w = k.rect.maxx - k.rect.minx, h = k.rect.maxy - k.rect.miny;
	constexpr uint32_t kSides = 4;
	for (; c->candidate < 2 * RGFX_DISPLAYS_MAX * kSides; c->candidate++) {
		const uint32_t pass = c->candidate / (RGFX_DISPLAYS_MAX * kSides);
		const uint32_t anchor = c->candidate / kSides % RGFX_DISPLAYS_MAX;
		const uint32_t side = c->candidate % kSides;
		const RgfxDisplayPlace &a = places.at[anchor];
		if (anchor == c->display || a.state == RgfxPlaceState::none)
			continue;
		const bool known =
		    a.state == RgfxPlaceState::learned || a.state == RgfxPlaceState::confirmed;
		if (known != (pass == 0))
			continue;
		RgfxGlobalRect r = {};
		switch (side) {
		case 0:
			r = {a.rect.maxx, a.rect.miny, a.rect.maxx + w, a.rect.miny + h};
			break;
		case 1:
			r = {a.rect.minx - w, a.rect.miny, a.rect.minx, a.rect.miny + h};
			break;
		case 2:
			r = {a.rect.minx, a.rect.maxy, a.rect.minx + w, a.rect.maxy + h};
			break;
		default:
			r = {a.rect.minx, a.rect.miny - h, a.rect.minx + w, a.rect.miny};
			break;
		}
		if (rgfxPlaceSame(r, k.rect))
			continue; // the first post's
		bool overlaps = false;
		for (uint32_t d = 0; d < RGFX_DISPLAYS_MAX; d++) {
			const RgfxDisplayPlace &o = places.at[d];
			if (d != c->display &&
			    (o.state == RgfxPlaceState::learned || o.state == RgfxPlaceState::confirmed) &&
			    rgfxPlaceOverlaps(r, o.rect))
				overlaps = true;
		}
		if (overlaps)
			continue;
		c->candidate++;
		*rect = r;
		return true;
	}
	return false;
}

// The point of the crossing's next post at `now`, which is then out: false, with the crossing
// over, where it posted kRgfxCrossingPosts, its hold is over, or no placement is left; false with
// it going on while a post is out.
inline bool rgfxCrossingNext(const RgfxDisplayPlaces &places, RgfxCrossing *c, uint64_t now,
                             int32_t *gx, int32_t *gy)
{
	if (!c->active || c->waiting)
		return false;
	const RgfxDisplayPlace &k = places.at[c->display < RGFX_DISPLAYS_MAX ? c->display : 0];
	if (c->display >= RGFX_DISPLAYS_MAX || k.state == RgfxPlaceState::none ||
	    c->posts >= kRgfxCrossingPosts || (c->posts && now - c->startedAt >= kRgfxCrossingHoldNs)) {
		c->active = false;
		return false;
	}
	RgfxGlobalRect rect = k.rect;
	if (c->posts && !rgfxCrossingCandidate(places, c, &rect)) {
		c->active = false;
		return false;
	}
	int32_t px = 0, py = 0;
	if (!rgfxPointerGlobal(rect, k.width, k.height, c->x, c->y, &px, &py)) {
		c->active = false;
		return false;
	}
	c->posts++;
	c->waiting = true;
	c->gx = *gx = px;
	c->gy = *gy = py;
	c->postedAt = now;
	return true;
}

// The cursor shown on `display` at pixel `x`, `y` (a MOVE_CURSOR with the cursor visible): where a
// post was out, its landing, which checks the display's rectangle (`*confirmed` where it confirmed
// it), and the crossing's step.
inline RgfxCrossingStep rgfxCrossingLanded(RgfxDisplayPlaces *places, RgfxCrossing *c,
                                           uint32_t display, int32_t x, int32_t y, bool *confirmed)
{
	*confirmed = false;
	if (!c->waiting)
		return RgfxCrossingStep::none;
	c->waiting = false;
	*confirmed = rgfxPlacesLanded(places, display, {c->gx, c->gy, x, y});
	if (!c->active)
		return RgfxCrossingStep::none;
	if (display == c->display) {
		c->active = false;
		return RgfxCrossingStep::arrived;
	}
	return RgfxCrossingStep::again;
}

// At `now`, a post out for kRgfxLandingWaitNs with no landing: no longer waited for, and the
// crossing that posted it over. True where it was.
inline bool rgfxCrossingExpired(RgfxCrossing *c, uint64_t now)
{
	if (!c->waiting || now - c->postedAt < kRgfxLandingWaitNs)
		return false;
	c->waiting = false;
	if (c->posts)
		c->active = false;
	return true;
}

#endif // RGFX_DISPLAY_PLACES_HPP_
