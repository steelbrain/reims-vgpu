// RgfxMobileCursor: the cursor an arm64 WindowServer gives a swap as its layer 1, which the display
// shows as the device's cursor (ReimsGraphicsMobile.cpp, beginMobileSwaps). Apple's paravirtual
// display (AppleParavirtDisplay) takes it so: where a swap's layer bit 1 is set, its
// program_hardware_gated calls its program_hardware_cursor_gated with layer 1's surface and the
// swap's two words at the release's cursor offset (RgfxLegacySwapLayout.hpp), and that
// - with no surface, hides the cursor;
// - with the surface and the surface's seed it took last, moves the cursor alone, to the two words
//   plus the hot spot it took then;
// - else takes the surface's image: IOSurface's image origin and extents
//   (getImageOriginAndExtents, RgfxSurfaceImageExtents), each extent cut to the host's largest
//   cursor. Both 0 is no cursor: it answers 2, and program_hardware_gated then shows layer 1's
//   surface as the swap's, in place of layer 0's. A surface that is not BGRA of 4 bytes an element,
//   an origin past the surface, or a hot spot outside the image cut to the surface past the
//   origin, is refused and logged, the cursor left as it was. Otherwise the image's rows go to the
//   host from the origin, with the hot spot less the origin, and the cursor moves to the two words
//   plus the hot spot.
// (Read from macOS 26.6.2's and 27.0.1's AppleParavirtGPU.) Unlike the paravirtual display, which
// records the surface before it checks it and so moves a refused or image-less surface's cursor at
// its next swap, the display here keeps what it found of a surface and seed, and answers the same
// for them until either changes.
//
// Header-only and free of the kernel's headers, for the native test (tests/mobile_cursor_test.cpp).
//
// Our own code.
#ifndef RGFX_MOBILE_CURSOR_HPP_
#define RGFX_MOBILE_CURSOR_HPP_

#include <stddef.h>
#include <stdint.h>
#include <string.h>

// What IOSurface's getImageOriginAndExtents answers (24 bytes, returned through x8): the image's
// origin in the surface's pixels, its extents, and the point the paravirtual display adds to the
// cursor's position and checks lies in the image, in the surface's pixels: the hot spot.
struct RgfxSurfaceImageExtents {
	uint32_t originX;
	uint32_t originY;
	uint32_t width;
	uint32_t height;
	uint32_t hotX;
	uint32_t hotY;
};
static_assert(sizeof(RgfxSurfaceImageExtents) == 24, "getImageOriginAndExtents answers 24 bytes");

// The pixel format and bytes per element of the only surface the display takes as a cursor:
// 'BGRA', 4 bytes.
constexpr uint32_t kRgfxCursorPixelFormat = 0x42475241;
constexpr uint32_t kRgfxCursorBytesPerElement = 4;

// The image a cursor surface gives the device: its rectangle in the surface, its hot spot in the
// image (SET_CURSOR's), and the hot spot in the surface, which MOVE_CURSOR's place adds to the
// swap's position.
struct RgfxMobileCursorImage {
	uint32_t x;
	uint32_t y;
	uint32_t width;
	uint32_t height;
	uint32_t hotX;
	uint32_t hotY;
	uint32_t moveX;
	uint32_t moveY;
};

// What a surface is to the cursor.
enum class RgfxCursorTake : uint32_t {
	// An image, which the cursor shows.
	Image,
	// No image (both extents 0): the swap shows the surface in place of layer 0's.
	None,
	// Refused: not BGRA of 4 bytes an element.
	Format,
	// Refused: the image's origin lies past the surface.
	Origin,
	// Refused: the hot spot lies outside the image.
	HotSpot,
};

// What a surface of `surfaceWidth` x `surfaceHeight` pixels, `pixelFormat` and `bytesPerElement`,
// whose image origin and extents are `e`, is to a cursor of at most `maxSize` pixels a side, and
// its image in `*image` for RgfxCursorTake::Image, as the paravirtual display takes it.
inline RgfxCursorTake rgfxMobileCursorImage(uint32_t surfaceWidth, uint32_t surfaceHeight,
                                            uint32_t pixelFormat, uint32_t bytesPerElement,
                                            const RgfxSurfaceImageExtents &e, uint32_t maxSize,
                                            RgfxMobileCursorImage *image)
{
	uint32_t width = e.width < maxSize ? e.width : maxSize;
	uint32_t height = e.height < maxSize ? e.height : maxSize;
	if (!width && !height)
		return RgfxCursorTake::None;
	if (pixelFormat != kRgfxCursorPixelFormat || bytesPerElement != kRgfxCursorBytesPerElement)
		return RgfxCursorTake::Format;
	if (!surfaceWidth || !surfaceHeight || e.originX >= surfaceWidth || e.originY >= surfaceHeight)
		return RgfxCursorTake::Origin;
	if (width > surfaceWidth - e.originX)
		width = surfaceWidth - e.originX;
	if (height > surfaceHeight - e.originY)
		height = surfaceHeight - e.originY;
	if (e.hotX < e.originX || e.hotX - e.originX >= width || e.hotY < e.originY ||
	    e.hotY - e.originY >= height)
		return RgfxCursorTake::HotSpot;
	*image = {e.originX, e.originY, width, height, e.hotX - e.originX, e.hotY - e.originY,
	          e.hotX,    e.hotY};
	return RgfxCursorTake::Image;
}

// The bytes of a surface of `bytesPerRow` an image's rows lie in: from `*offset`, `*length` of
// them. False where the rows do not fit `bytesPerRow`, or the span runs past `surfaceBytes`.
inline bool rgfxMobileCursorSpan(const RgfxMobileCursorImage &image, uint64_t bytesPerRow,
                                 uint64_t surfaceBytes, uint64_t *offset, uint64_t *length)
{
	const uint64_t rowEnd = (static_cast<uint64_t>(image.x) + image.width) * 4;
	if (!image.width || !image.height || rowEnd > bytesPerRow)
		return false;
	const uint64_t first = image.y * bytesPerRow + static_cast<uint64_t>(image.x) * 4;
	const uint64_t end = (static_cast<uint64_t>(image.y) + image.height - 1) * bytesPerRow + rowEnd;
	if (end > surfaceBytes)
		return false;
	*offset = first;
	*length = end - first;
	return true;
}

// Copies an image's rows from `span`, the surface's bytes rgfxMobileCursorSpan gives, into `slot`
// as SET_CURSOR takes them: `width * 4` bytes a row, one after the other.
inline void rgfxMobileCursorCopy(const RgfxMobileCursorImage &image, uint64_t bytesPerRow,
                                 const uint8_t *span, uint8_t *slot)
{
	const size_t row = static_cast<size_t>(image.width) * 4;
	for (uint32_t r = 0; r < image.height; r++)
		memcpy(slot + r * row, span + r * bytesPerRow, row);
}

// What the display holds of the cursor between swaps: the surface and seed it took last and what
// it found them to be, with their image; and the place and visibility MOVE_CURSOR last posted.
struct RgfxMobileCursor {
	bool taken;
	uint32_t surfaceId;
	uint32_t seed;
	RgfxCursorTake take;
	RgfxMobileCursorImage image;
	bool moved;
	int32_t x;
	int32_t y;
	bool visible;

	// Whether a swap's surface of `surfaceId` and `seed` is the one taken last, so that what was
	// found of it stands.
	bool same(uint32_t id, uint32_t s) const { return taken && surfaceId == id && seed == s; }
	// Records what a surface of `id` and `s` was found to be.
	void record(uint32_t id, uint32_t s, RgfxCursorTake t, const RgfxMobileCursorImage &i)
	{
		taken = true;
		surfaceId = id;
		seed = s;
		take = t;
		image = i;
	}
	// Forgets the surface taken last, so its next swap takes it again (its image not sent).
	void forget() { taken = false; }
	// The MOVE_CURSOR a swap's position `(px, py)` asks for: the hot spot's place, and whether
	// the cursor shows. False where it is the one posted last, which needs no post.
	bool move(int32_t px, int32_t py, bool show, int32_t *mx, int32_t *my) const
	{
		*mx = show ? px + static_cast<int32_t>(image.moveX) : x;
		*my = show ? py + static_cast<int32_t>(image.moveY) : y;
		return !moved || visible != show || (show && (*mx != x || *my != y));
	}
	// Records the MOVE_CURSOR posted.
	void posted(int32_t mx, int32_t my, bool show)
	{
		moved = true;
		x = mx;
		y = my;
		visible = show;
	}
};

#endif // RGFX_MOBILE_CURSOR_HPP_
