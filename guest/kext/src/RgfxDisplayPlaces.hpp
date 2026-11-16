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
// past every display is held at the edge of the display the cursor is on (2880,540 with one
// 1920-wide display showed the cursor at its right edge, on 14.6.1, job m3-max-35287; 5755,600
// with the cursor on display 0 of two side by side held it at display 0's right edge, on 27.0.1,
// job m3-max-35450).
//
// So each driven display's rectangle is held here, and each landing of a move the kext posted at
// it checks it, by the placement the move was posted by (the rectangle held, or one beside
// another display):
//  - a display starts from a guess, WindowServer's own arrangement of new displays: side by side
//    in their order, tops at 0 (measured on 27.0.1's first boot at two displays), each its mode's
//    size at 1x on a 1x screen and half that above 1x, as arm64's WindowServer runs a display
//    HiDPI above 1x (design.md, "The display's density"); a display's new mode keeps its corner
//    and takes the guessed size;
//  - a landing whose pixel lies where the placement puts the posted point, on both axes, with
//    the pixel far enough from the corner to show a wrong size, confirms the placement as the
//    rectangle, and only a confirmed one is told the host (DISPLAY_BOUNDS);
//  - a landing that disagrees moves the rectangle's corner on each axis where the pixel lies
//    inside the display, so the point was not held at an edge; on an axis held at an edge the
//    point lay past the display, so the placement the move was posted by is not the display's
//    there, and the rectangle keeps its own (measured on 27.0.1: a post beside display 1 for
//    display 0, which the cursor was on, held the cursor at display 0's edge; taking that
//    placement on the axis put display 0 past display 1, potluck job m3-max-35865); the
//    rectangle so moved is learned, and confirmed by a later landing that agrees;
//  - two landings on a display, the second disagreeing with its size, each inside it at the size
//    held and at the size they give, and 64 pixels apart or more, size it by the points between
//    them over the pixels between them, the other axis alike.
//
// A crossing is the moves that put the cursor on the display the host's pointer is over, for its
// pixel there: first at that pixel of the display's rectangle; and, where the cursor did not reach
// the display, at that pixel of the display placed beside each other display in turn (right,
// left, below and above it, top or left edges aligned), each placement that overlaps no learned
// display's. A crossing ends once the cursor lands on its display, wherever on it (the host's
// tablet then puts it at its pixel, scaled over the display the cursor is on), and with the
// cursor where it is after kRgfxCrossingPosts posts, so a guest never sees the cursor jump on.
//
// Only the display a post aims at learns from its landing: a post lands on the display holding
// its point, or, where none does, on the display the cursor was on, held at that display's edge,
// which is never the display aimed at. What else shows the cursor while a post is out is not
// taken for its landing:
//  - on another display, a pixel other than the one the post's point lies at, or is held at, is
//    the tablet's move or the user's: the host passes the pointer's positions on to the tablet
//    once its hold is over (the device's CROSSING_HOLD_NS, 100 ms), and an arm64 guest under
//    TCG shows a post's landing 18 to 153 ms after the POINTER_OVER (host log, 14.6.1 at two
//    displays, potluck job m3-max-35541), so the tablet can move the cursor on the display it is
//    on before the post lands; taken for a landing there, it placed that display over the other;
//  - on another display, the pixel the post's point is held at, where the point lies within a
//    cursor's reach (RGFX_CURSOR_MAX_SIZE points) past that display's edge, is the cursor held
//    there, or its image reaching over the edge from the display beside it, whose hot spot was
//    shown at the edge's pixel too (measured on arm64 27.0.1: a post at 1915,650 of display 0,
//    1920 wide, showed the cursor on display 1 at 0,650 before display 0's landing; job
//    m3-max-35515): the crossing waits kRgfxSpillWaitNs for the landing on the display aimed at
//    before it posts again;
//  - a hot spot past the display's edge is the image over it alone.
// A post that shows no move in kRgfxLandingWaitNs left the cursor where it was: at the pixel it
// aimed at already, where the crossing's display last showed the cursor there (the host told the
// crossing before it saw the cursor there), which ends the crossing; otherwise its point was held
// at the very pixel the cursor was at (a captured pointer pushed past its window's edge after the
// displays were rearranged), and the crossing posts again. Where no display shows the cursor where
// that post's point lies or is held at, the post's landing may yet come (measured on 27.0.1: a
// post 2 s after the displays were rearranged showed nothing in kRgfxLandingWaitNs, potluck job
// m3-max-35865), at the very pixel each later post of the crossing aims at, where it cannot be
// told from theirs: the crossing still ends on its display's landing, and learns no place from
// one after it.
// Only a MOVE_CURSOR posted after the post shows what the post did: a display that showed the
// cursor where the post's point is held at before the post went out showed nothing of it
// (measured on arm64 15.6.1 under TCG, at two displays side by side in a 2x mode: a captured
// drag left the cursor at display 1's left edge, the crossing onto display 0 posted there, and
// at its deadline, the post not yet shown, display 1 still showed the cursor at the very pixel
// that post's point was held at; taken for its outcome, the crossing posted beside display 1,
// and the first post's late landing, taken for that one's, placed display 0 right of display 1,
// potluck job m3-max-36173).
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

// A crossing's posts: the first, and at most three more. The count is the crossing's one bound,
// with no cutoff at the host's hold (the device's CROSSING_HOLD_NS, 100 ms): an arm64 guest under
// TCG shows even a first post's landing 18 to 153 ms after the POINTER_OVER (14.6.1 at two
// displays, potluck job m3-max-35541), so a crossing cut off at the hold would end before its
// landing showed that a display had moved. Under HVF a whole crossing ends inside the hold (27.0.1:
// landings 5.8 to 35 ms after their post, job m3-max-35651).
constexpr uint32_t kRgfxCrossingPosts = 4;
// How long a post waits for its landing: past it, the cursor is taken to have stayed where it was.
constexpr uint64_t kRgfxLandingWaitNs = 250000000;
// How long a landing that may be the cursor's image over another display's edge waits for the
// landing on the display aimed at.
constexpr uint64_t kRgfxSpillWaitNs = 50000000;
// How far past a display's edge, in points, the cursor's image may reach over it.
constexpr int64_t kRgfxSpillReach = RGFX_CURSOR_MAX_SIZE;
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

inline bool rgfxPlaceSame(const RgfxGlobalRect &a, const RgfxGlobalRect &b)
{
	return a.minx == b.minx && a.miny == b.miny && a.maxx == b.maxx && a.maxy == b.maxy;
}

// Display `display`'s rectangle checked by the landing of a post aimed at it: `landing`'s point
// posted, placed by `aimed` (the rectangle held for the display, or a placement beside another),
// and the cursor shown on the display at its pixel. The landing is checked against `aimed`, which
// the rectangle becomes where the landing agrees with it. True where it confirmed the rectangle
// (which the host is then told, where it differs from the one last told).
inline bool rgfxPlacesLanded(RgfxDisplayPlaces *places, uint32_t display,
                             const RgfxPlaceLanding &landing, const RgfxGlobalRect &aimed)
{
	if (display >= RGFX_DISPLAYS_MAX)
		return false;
	RgfxDisplayPlace &p = places->at[display];
	if (p.state == RgfxPlaceState::none || !p.width || !p.height || aimed.maxx <= aimed.minx ||
	    aimed.maxy <= aimed.miny)
		return false;
	const int64_t pixels[2] = {p.width, p.height};
	const int64_t posted[2] = {landing.gx, landing.gy};
	const int64_t pixel[2] = {landing.lx, landing.ly};
	int64_t corner[2] = {aimed.minx, aimed.miny};
	int64_t span[2] = {static_cast<int64_t>(aimed.maxx) - aimed.minx,
	                   static_cast<int64_t>(aimed.maxy) - aimed.miny};
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
		if (!rgfxPlaceInside(span[a], pixels[a], pixel[a])) {
			// Held at an edge: the point lay past the display on this axis, so the placement
			// is not the display's there; the rectangle keeps its own.
			corner[a] = a == 0 ? p.rect.minx : p.rect.miny;
			span[a] = a == 0 ? static_cast<int64_t>(p.rect.maxx) - p.rect.minx
			                 : static_cast<int64_t>(p.rect.maxy) - p.rect.miny;
			continue;
		}
		corner[a] += posted[a] - rgfxPlacePoint(corner[a], span[a], pixels[a], pixel[a]);
		moved++;
	}
	const bool resized = span[0] != static_cast<int64_t>(aimed.maxx) - aimed.minx ||
	                     span[1] != static_cast<int64_t>(aimed.maxy) - aimed.miny;
	const RgfxGlobalRect rect = {
	    static_cast<int32_t>(corner[0]), static_cast<int32_t>(corner[1]),
	    static_cast<int32_t>(corner[0] + span[0]), static_cast<int32_t>(corner[1] + span[1])};
	const bool placed = !rgfxPlaceSame(rect, p.rect);
	p.rect = rect;
	if (agreed == 2 && !resized &&
	    (landing.lx >= kRgfxPlaceProof || landing.ly >= kRgfxPlaceProof)) {
		p.state = RgfxPlaceState::confirmed;
		return true;
	}
	if (moved || resized || placed)
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
	uint32_t aimedAt;     // the display it is aimed at
	RgfxGlobalRect aimed; // that display's placement it is posted by
	int32_t gx;
	int32_t gy;
	uint64_t postedAt;
	// Another display showed the cursor where the post's point is held at its edge, within a
	// cursor's reach of it, at `heldAt`: the landing aimed at is waited for kRgfxSpillWaitNs.
	bool held;
	uint64_t heldAt;
	// How long after its post the last landing on the display aimed at came.
	uint64_t landedAfter;
	// A post of this crossing showed nothing by its deadline, with no display showing the cursor
	// where its point lies or is held at: its landing may yet come, so no later one teaches a
	// place.
	bool unsure;
	// The count of MOVE_CURSORs the kext had posted when the post out went out
	// (RgfxCursorShown::seq): a display's last one at or below it is from before the post.
	uint64_t postSeq;
};

// What a landing, or a wait for one, does to the crossing.
enum class RgfxCrossingStep {
	none,    // nothing: no post was out, or it is still waited for
	arrived, // the cursor is on the crossing's display: it is over
	again,   // the cursor is not on it: post again (rgfxCrossingNext)
	held,    // wait for the landing aimed at until rgfxCrossingDeadline
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
	c->unsure = false;
}

inline bool rgfxPlaceOverlaps(const RgfxGlobalRect &a, const RgfxGlobalRect &b)
{
	return a.minx < b.maxx && b.minx < a.maxx && a.miny < b.maxy && b.miny < a.maxy;
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
// over, where it posted kRgfxCrossingPosts or no placement is left; false with it going on while
// a post is out.
inline bool rgfxCrossingNext(const RgfxDisplayPlaces &places, RgfxCrossing *c, uint64_t now,
                             int32_t *gx, int32_t *gy)
{
	if (!c->active || c->waiting)
		return false;
	const RgfxDisplayPlace &k = places.at[c->display < RGFX_DISPLAYS_MAX ? c->display : 0];
	if (c->display >= RGFX_DISPLAYS_MAX || k.state == RgfxPlaceState::none ||
	    c->posts >= kRgfxCrossingPosts) {
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
	c->aimedAt = c->display;
	c->aimed = rect;
	c->held = false;
	c->gx = *gx = px;
	c->gy = *gy = py;
	c->postedAt = now;
	return true;
}

// The pixel of `pixels` a display spanning `span` points from `corner` shows the cursor at for a
// point `g` on one axis: the point's, or, where it lies past the display (`*past`), the edge's;
// and how far past (`*by`, in points).
inline int64_t rgfxPlaceHeld(int64_t corner, int64_t span, int64_t pixels, int64_t g, bool *past,
                             int64_t *by)
{
	int64_t at = g - corner;
	if (at < 0) {
		*past = true;
		if (-at > *by)
			*by = -at;
		at = 0;
	} else if (at > span - 1) {
		*past = true;
		if (at - (span - 1) > *by)
			*by = at - (span - 1);
		at = span - 1;
	}
	return at * pixels / span;
}

// The cursor shown on `display` with its hot spot at pixel `x`, `y` (a MOVE_CURSOR with the cursor
// visible) at `now`, while a post is out: on the display aimed at, the post's landing, which
// checks the display's rectangle (`*confirmed` where it confirmed it); on another display at the
// pixel the post's point lies at or is held at, the cursor not where it was aimed; anything else
// no landing (the header's comment). The crossing's step.
inline RgfxCrossingStep rgfxCrossingLanded(RgfxDisplayPlaces *places, RgfxCrossing *c,
                                           uint32_t display, int32_t x, int32_t y, uint64_t now,
                                           bool *confirmed)
{
	*confirmed = false;
	if (!c->waiting || display >= RGFX_DISPLAYS_MAX)
		return RgfxCrossingStep::none;
	const RgfxDisplayPlace &p = places->at[display];
	if (p.state == RgfxPlaceState::none || !p.width || !p.height || x < 0 || y < 0 ||
	    static_cast<uint32_t>(x) >= p.width || static_cast<uint32_t>(y) >= p.height)
		return RgfxCrossingStep::none;
	if (display == c->aimedAt || (c->active && display == c->display)) {
		c->waiting = false;
		c->held = false;
		if (display == c->aimedAt) {
			c->landedAfter = now - c->postedAt;
			if (!c->unsure)
				*confirmed = rgfxPlacesLanded(places, display, {c->gx, c->gy, x, y}, c->aimed);
		}
		if (!c->active)
			return RgfxCrossingStep::none;
		if (display == c->display) {
			c->active = false;
			return RgfxCrossingStep::arrived;
		}
		return RgfxCrossingStep::again;
	}
	const int64_t spanX = static_cast<int64_t>(p.rect.maxx) - p.rect.minx;
	const int64_t spanY = static_cast<int64_t>(p.rect.maxy) - p.rect.miny;
	if (spanX <= 0 || spanY <= 0)
		return RgfxCrossingStep::none;
	bool past = false;
	int64_t by = 0;
	const int64_t hx = rgfxPlaceHeld(p.rect.minx, spanX, p.width, c->gx, &past, &by);
	const int64_t hy = rgfxPlaceHeld(p.rect.miny, spanY, p.height, c->gy, &past, &by);
	if (rgfxPlaceAbs(x - hx) > 1 + p.width / spanX || rgfxPlaceAbs(y - hy) > 1 + p.height / spanY)
		return RgfxCrossingStep::none;
	if (c->held)
		return RgfxCrossingStep::none;
	if (past && by <= kRgfxSpillReach) {
		c->held = true;
		c->heldAt = now;
		return RgfxCrossingStep::held;
	}
	c->waiting = false;
	return c->active ? RgfxCrossingStep::again : RgfxCrossingStep::none;
}

// When the post out is no longer waited for: kRgfxLandingWaitNs after it, or kRgfxSpillWaitNs
// after a landing held it.
inline uint64_t rgfxCrossingDeadline(const RgfxCrossing &c)
{
	return c.held ? c.heldAt + kRgfxSpillWaitNs : c.postedAt + kRgfxLandingWaitNs;
}

// The cursor as a display's last MOVE_CURSOR showed it: whether it is shown, and its hot spot's
// pixel.
struct RgfxCursorShown {
	bool visible;
	int32_t x;
	int32_t y;
	// The count of MOVE_CURSORs the kext had posted, every display's, with that one.
	uint64_t seq;
};

// Each display's, by its id.
struct RgfxCursorsShown {
	RgfxCursorShown at[RGFX_DISPLAYS_MAX];
};

// Whether display `d` shows the cursor where the post out's point lies or is held at, by the
// placement it was aimed by (the display aimed at) or the display's rectangle (another); `*past`
// where the point lies past it.
inline bool rgfxCrossingShownAt(const RgfxDisplayPlaces &places, const RgfxCrossing &c,
                                uint32_t d, const RgfxCursorShown &shown, bool *past)
{
	*past = false;
	const RgfxDisplayPlace &p = places.at[d];
	const RgfxGlobalRect &r = d == c.aimedAt ? c.aimed : p.rect;
	const int64_t spanX = static_cast<int64_t>(r.maxx) - r.minx;
	const int64_t spanY = static_cast<int64_t>(r.maxy) - r.miny;
	if (!shown.visible || p.state == RgfxPlaceState::none || !p.width || !p.height ||
	    spanX <= 0 || spanY <= 0)
		return false;
	int64_t by = 0;
	const int64_t hx = rgfxPlaceHeld(r.minx, spanX, p.width, c.gx, past, &by);
	const int64_t hy = rgfxPlaceHeld(r.miny, spanY, p.height, c.gy, past, &by);
	return rgfxPlaceAbs(shown.x - hx) <= 1 + p.width / spanX &&
	       rgfxPlaceAbs(shown.y - hy) <= 1 + p.height / spanY;
}

// At `now`, the post out waited for until its deadline with no landing on the display aimed at,
// given each display's last MOVE_CURSOR (`shown`): where the crossing's display shows the cursor
// at the pixel the post aimed at, the cursor was there already, and the post, which moved it
// nowhere, showed nothing: the crossing is over (`arrived`); otherwise the cursor is not there,
// and the crossing, where one goes on, posts again (`again`), unsure of its places where no
// display showed the cursor where the post's point lies or is held at with a MOVE_CURSOR after
// the post (the header's comment). `none` before the deadline or with no post out.
inline RgfxCrossingStep rgfxCrossingWaited(const RgfxDisplayPlaces &places, RgfxCrossing *c,
                                           uint64_t now, const RgfxCursorsShown &shown)
{
	if (!c->waiting || now < rgfxCrossingDeadline(*c))
		return RgfxCrossingStep::none;
	const bool held = c->held;
	c->waiting = false;
	c->held = false;
	if (!c->active)
		return RgfxCrossingStep::none;
	bool past = false;
	if (c->aimedAt == c->display && c->display < RGFX_DISPLAYS_MAX &&
	    rgfxCrossingShownAt(places, *c, c->display, shown.at[c->display], &past) && !past) {
		c->active = false;
		return RgfxCrossingStep::arrived;
	}
	bool seen = held;
	for (uint32_t d = 0; d < RGFX_DISPLAYS_MAX && !seen; d++)
		seen = shown.at[d].seq > c->postSeq &&
		       rgfxCrossingShownAt(places, *c, d, shown.at[d], &past);
	if (!seen)
		c->unsure = true;
	return RgfxCrossingStep::again;
}

#endif // RGFX_DISPLAY_PLACES_HPP_
