// The native test of RgfxMobileCursor.hpp: what a swap's layer-1 surface is to the cursor, as the
// paravirtual display's program_hardware_cursor_gated takes it, the bytes its image's rows lie in,
// their copy into the cursor's slot, and the moves the display posts.
#include "RgfxMobileCursor.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "mobile_cursor_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

constexpr uint32_t kBGRA = kRgfxCursorPixelFormat;

RgfxCursorTake take(uint32_t w, uint32_t h, const RgfxSurfaceImageExtents &e,
                    RgfxMobileCursorImage *image, uint32_t format = kBGRA, uint32_t bpe = 4)
{
	return rgfxMobileCursorImage(w, h, format, bpe, e, 256, image);
}

void testImage()
{
	RgfxMobileCursorImage i = {};
	// The arrow WindowServer gives: the whole surface, its hot spot inside.
	check(take(64, 64, {0, 0, 64, 64, 4, 5}, &i) == RgfxCursorTake::Image,
	      "a BGRA surface's image is a cursor");
	check(i.x == 0 && i.y == 0 && i.width == 64 && i.height == 64 && i.hotX == 4 && i.hotY == 5 &&
	          i.moveX == 4 && i.moveY == 5,
	      "the image is the extents at the origin, the hot spot in it");
	// An origin inside the surface: the hot spot is the surface's, the image's less the origin.
	check(take(100, 80, {10, 20, 32, 30, 15, 25}, &i) == RgfxCursorTake::Image, "an inner image");
	check(i.x == 10 && i.y == 20 && i.width == 32 && i.height == 30 && i.hotX == 5 && i.hotY == 5 &&
	          i.moveX == 15 && i.moveY == 25,
	      "the image's hot spot is the surface's less the origin; the move adds the surface's");
	// Extents past the largest cursor are cut to it, and past the surface to the surface.
	check(take(512, 512, {0, 0, 400, 300, 1, 1}, &i) == RgfxCursorTake::Image &&
	          i.width == 256 && i.height == 256,
	      "extents are cut to the largest cursor");
	check(take(100, 50, {90, 40, 64, 64, 95, 45}, &i) == RgfxCursorTake::Image && i.width == 10 &&
	          i.height == 10,
	      "extents are cut to the surface past the origin");
	// Both extents 0: no image, which the swap shows as its surface.
	check(take(64, 64, {0, 0, 0, 0, 0, 0}, &i) == RgfxCursorTake::None, "no extents, no image");
	check(take(64, 64, {0, 0, 0, 0, 0, 0}, &i, 0x34324241) == RgfxCursorTake::None,
	      "no image whatever the format, which is checked after");
	// Refusals, as the paravirtual display's.
	check(take(64, 64, {0, 0, 64, 64, 1, 1}, &i, 0x34324241) == RgfxCursorTake::Format,
	      "not BGRA is refused");
	check(take(64, 64, {0, 0, 64, 64, 1, 1}, &i, kBGRA, 2) == RgfxCursorTake::Format,
	      "not 4 bytes an element is refused");
	check(take(64, 64, {64, 0, 8, 8, 64, 1}, &i) == RgfxCursorTake::Origin,
	      "an origin past the width is refused");
	check(take(64, 64, {0, 64, 8, 8, 1, 64}, &i) == RgfxCursorTake::Origin,
	      "an origin past the height is refused");
	check(take(0, 0, {0, 0, 8, 8, 1, 1}, &i) == RgfxCursorTake::Origin,
	      "an empty surface is refused");
	check(take(64, 64, {10, 10, 8, 8, 9, 12}, &i) == RgfxCursorTake::HotSpot,
	      "a hot spot left of the image is refused");
	check(take(64, 64, {10, 10, 8, 8, 18, 12}, &i) == RgfxCursorTake::HotSpot,
	      "a hot spot right of the image is refused");
	check(take(64, 64, {10, 10, 8, 8, 12, 18}, &i) == RgfxCursorTake::HotSpot,
	      "a hot spot below the image is refused");
	check(take(64, 64, {0, 0, 8, 0, 1, 0}, &i) == RgfxCursorTake::HotSpot,
	      "one extent 0 leaves no row for the hot spot");
	check(take(20, 20, {0, 0, 64, 64, 30, 1}, &i) == RgfxCursorTake::HotSpot,
	      "a hot spot outside the image cut to the surface is refused");
}

void testSpanAndCopy()
{
	// A 10 x 6 surface of 48 bytes a row (4 bytes of padding), its image 3 x 2 at (4, 3).
	const uint64_t bpr = 48;
	std::vector<uint8_t> surface(bpr * 6);
	for (size_t k = 0; k < surface.size(); k++)
		surface[k] = static_cast<uint8_t>(k);
	RgfxMobileCursorImage i = {4, 3, 3, 2, 1, 1, 5, 4};
	uint64_t offset = 0, length = 0;
	check(rgfxMobileCursorSpan(i, bpr, surface.size(), &offset, &length), "the span fits");
	check(offset == 3 * bpr + 16 && length == bpr + 12, "the span runs from the first row's first "
	                                                    "pixel to the last row's last");
	std::vector<uint8_t> slot(3 * 2 * 4, 0xee);
	rgfxMobileCursorCopy(i, bpr, surface.data() + offset, slot.data());
	bool rows = true;
	for (uint32_t r = 0; r < 2; r++)
		for (uint32_t b = 0; b < 12; b++)
			rows = rows && slot[r * 12 + b] == surface[(3 + r) * bpr + 16 + b];
	check(rows, "the slot holds the image's rows, one after the other");
	check(!rgfxMobileCursorSpan(i, bpr, offset + length - 1, &offset, &length),
	      "a surface a byte short is refused");
	RgfxMobileCursorImage wide = {10, 0, 3, 1, 0, 0, 10, 0};
	check(!rgfxMobileCursorSpan(wide, bpr, surface.size(), &offset, &length),
	      "rows past the bytes per row are refused");
	RgfxMobileCursorImage none = {0, 0, 0, 0, 0, 0, 0, 0};
	check(!rgfxMobileCursorSpan(none, bpr, surface.size(), &offset, &length),
	      "an empty image has no span");
}

void testState()
{
	RgfxMobileCursor c = {};
	check(!c.same(7, 0), "nothing taken matches nothing");
	RgfxMobileCursorImage arrow = {0, 0, 32, 32, 3, 4, 3, 4};
	c.record(7, 1, RgfxCursorTake::Image, arrow);
	check(c.same(7, 1) && !c.same(7, 2) && !c.same(8, 1), "the surface and seed taken match");
	int32_t x = 0, y = 0;
	check(c.move(100, 50, true, &x, &y) && x == 103 && y == 54,
	      "the first move posts the hot spot's place");
	c.posted(x, y, true);
	check(!c.move(100, 50, true, &x, &y), "the same place posts nothing");
	check(c.move(101, 50, true, &x, &y) && x == 104, "a new place posts");
	c.posted(x, y, true);
	check(c.move(0, 0, false, &x, &y) && x == 104 && y == 54,
	      "a hide posts, at the place it had");
	c.posted(x, y, false);
	check(!c.move(9, 9, false, &x, &y), "hidden again posts nothing");
	check(c.move(101, 50, true, &x, &y), "shown again posts, at the same place");
	c.forget();
	check(!c.same(7, 1), "a surface forgotten is taken again");
}

} // namespace

int main()
{
	testImage();
	testSpanAndCopy();
	testState();
	std::printf("mobile_cursor_test: %d checks passed\n", checks);
	return 0;
}
