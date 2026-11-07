// The native test of RgfxGLSurfaces.hpp: a GL window's surface and drawable, as the guest's GL
// renderer drove them (the measured calls, end to end: the buffers made, their shape and format,
// the set surface's answer and the lookups' slots), and IOGPU's rules around them: what SetIDMode
// and SetShape refuse, which slots a drawable's mode asks for, what each slot is made as, and
// when slots go or are made again.
#include "RgfxGLSurfaces.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "gl_surfaces_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

// A region of no rectangles, its bounds 0, 0, `w`, `h`.
void region(uint8_t out[12], uint16_t w, uint16_t h)
{
	std::memset(out, 0, 12);
	out[8] = w & 0xff;
	out[9] = w >> 8;
	out[10] = h & 0xff;
	out[11] = h >> 8;
}

// A surface's slots as the kext holds them: which hold a buffer, and each one's shape.
struct Slots {
	uint32_t existing = 0;
	RgfxGLBuffer buffers[kRgfxGLSlots] = {};

	void prune(RgfxGLSurface &s) { existing &= ~rgfxGLPrune(s, existing); }
	void reshape(RgfxGLSurface &s)
	{
		const uint32_t remake = rgfxGLReshape(s);
		for (uint32_t slot = 0; slot < kRgfxGLSlots; slot++) {
			if (!(remake & (1u << slot)))
				continue;
			existing &= ~(1u << slot);
			if (rgfxGLBufferOf(slot, s.mode, s.width, s.height, buffers[slot]))
				existing |= 1u << slot;
		}
	}
};

// A drawable's set surface on `s`, which `others` other drawables (asking for `otherRequests`)
// hold: its request bits, as the kext attaches it.
uint32_t attach(RgfxGLSurface &s, Slots &slots, uint64_t mode, uint32_t others, uint32_t otherRequests)
{
	const uint64_t on = rgfxGLDrawableModeOn(mode, s.mode);
	rgfxGLResetRequests(s, otherRequests);
	s.mode = rgfxGLSurfaceModeWith(s.mode, on, others);
	const uint32_t bits = rgfxGLDrawableRequestBits(on);
	s.requested |= bits;
	slots.prune(s);
	slots.reshape(s);
	return bits;
}

void measuredWindow()
{
	RgfxGLSurface s;
	Slots slots;
	check(s.id == kRgfxGLNoId, "a new surface has no id");

	// SetIDMode(0x100b, 0x4024): mode 0x24 (0x4000 only chose the type), nothing made yet.
	check(rgfxGLSetIDMode(s, 0x100b, 0x4024, 0) == kRgfxGLSuccess, "the window's SetIDMode");
	check(s.id == 0x100b && s.mode == 0x24, "the id and the mode without 0x4000");
	rgfxGLResetRequests(s, 0);
	slots.prune(s);
	slots.reshape(s);
	check(slots.existing == 0 && s.requested == 0, "no buffer before a drawable asks");

	// SetShape(2, 0) with a region of no rectangles, bounds 256 x 256: the shape, still no
	// buffer (SetShape makes none).
	uint8_t r[12];
	region(r, 256, 256);
	check(rgfxGLSetShape(s, 2, 0, r, 12) == kRgfxGLSuccess, "the window's SetShape");
	check(s.shaped && s.width == 256 && s.height == 256, "the shape's size");
	check(slots.existing == 0, "SetShape makes no buffer");

	// The drawable's set surface, mode 0xc00: colour slots 0 and 1, 256 x 256 BGRA.
	const uint32_t bits = attach(s, slots, 0xc00, 0, 0);
	check(bits == 3, "mode 0xc00 asks for colour slots 0 and 1");
	check(s.mode == 0x24, "the surface keeps its mode: the drawable's 0xc00 is the context's");
	check(slots.existing == 3, "slots 0 and 1 made");
	for (uint32_t slot = 0; slot < 2; slot++) {
		const RgfxGLBuffer &b = slots.buffers[slot];
		check(b.width == 256 && b.height == 256 && b.bytesPerElement == 4 && b.hasFormat &&
		          b.format == 0x42475241,
		      "a colour buffer of the shape, BGRA");
	}

	// The lookups the renderer makes: index 0 is slot 1, index 1 is slot 0.
	check(rgfxGLSlotOfIndex(0) == 1 && rgfxGLSlotOfIndex(1) == 0, "the renderer's two lookups");

	// A resize: SetShape keeps the buffers; the next set surface makes them anew at the size.
	region(r, 300, 200);
	check(rgfxGLSetShape(s, 2, 0, r, 12) == kRgfxGLSuccess, "the resize's SetShape");
	check(slots.buffers[0].width == 256, "SetShape alone leaves the buffers");
	// The drawable's set surface again: off the surface (its requests start again, no prune,
	// as it attaches to the same one), then on.
	rgfxGLResetRequests(s, 0);
	attach(s, slots, 0xc00, 0, 0);
	check(slots.existing == 3 && slots.buffers[0].width == 300 && slots.buffers[1].height == 200,
	      "the buffers made again at the new size");

	// Another set surface at the same size and mode makes nothing.
	RgfxGLSurface again = s;
	rgfxGLResetRequests(again, 0);
	again.requested |= 3;
	check(rgfxGLPrune(again, 3) == 0, "nothing pruned");
	check(rgfxGLReshape(again) == 0, "the same size and mode remake nothing");

	// The drawable goes: its request bits go, but the colour slots stay (IOGPU keeps the
	// requests of slots 0 and 1 across a reset) until the surface goes.
	rgfxGLResetRequests(s, 0);
	slots.prune(s);
	check(slots.existing == 3, "the colour buffers outlive the drawable");
}

void setIDModeRules()
{
	RgfxGLSurface s;
	check(rgfxGLSetIDMode(s, 1, 0x4024 | 0x40, 0) == kRgfxGLBadArgument, "a refused mode bit");
	check(rgfxGLSetIDMode(s, 1, 0x4024 | 0x10000000, 0) == kRgfxGLBadArgument, "a high refused bit");
	check(rgfxGLSetIDMode(s, 1, 0x24 | 0xc00, 0) == kRgfxGLError, "both buffering bits");
	check(rgfxGLSetIDMode(s, 1, 0x4, 0) == kRgfxGLError, "a surface that is not windowed");
	check(rgfxGLSetIDMode(s, 1, 0x24 | 0x100000, 0) == kRgfxGLUnsupported, "shared buffers");
	check(s.id == kRgfxGLNoId && s.mode == 0, "a refusal changes nothing");
	check(rgfxGLSetIDMode(s, 7, 0x4024, 0) == kRgfxGLSuccess, "the first id");
	check(rgfxGLSetIDMode(s, 7, 0x4424, 0) == kRgfxGLSuccess, "the same id again, a new mode");
	check(s.mode == 0x424, "the new mode");
	check(rgfxGLSetIDMode(s, 8, 0x4024, 0) == kRgfxGLBadArgument, "another id");
	check(s.id == 7, "the id is kept");
	// With a drawable attached, the mode's kept bits may not change, and the others are kept.
	check(rgfxGLSetIDMode(s, 7, 0x4023, 1) == kRgfxGLError, "a new depth under a drawable");
	s.mode |= 0x1000;
	check(rgfxGLSetIDMode(s, 7, 0x4424, 1) == kRgfxGLSuccess, "the same kept bits under a drawable");
	check(s.mode == (0x424 | 0x1000), "the bits outside the kept ones stay");
}

void setShapeRules()
{
	RgfxGLSurface s;
	uint8_t r[20];
	region(r, 64, 32);
	check(rgfxGLSetShape(s, 2, 0, nullptr, 12) == kRgfxGLBadArgument, "no region");
	check(rgfxGLSetShape(s, 2, 0, r, 11) == kRgfxGLBadArgument, "a region under 12 bytes");
	check(rgfxGLSetShape(s, 0x1000, 0, r, 12) == kRgfxGLUnsupported, "option 0x1000");
	check(rgfxGLSetShape(s, 0x2000, 0, r, 12) == kRgfxGLUnsupported, "option 0x2000");
	check(rgfxGLSetShape(s, 2, 1, r, 12) == kRgfxGLError, "another framebuffer");
	check(rgfxGLSetShape(s, 2, 0, r, 20) == kRgfxGLError, "a length its count does not give");
	r[0] = 1;
	check(rgfxGLSetShape(s, 2, 0, r, 12) == kRgfxGLError, "one rectangle in 12 bytes");
	check(!s.shaped, "a refused shape sets nothing");
	check(rgfxGLSetShape(s, 2, 0, r, 20) == kRgfxGLSuccess, "one rectangle in 20 bytes");
	check(s.shaped && s.width == 64 && s.height == 32, "the bounds' size");
	region(r, 0x8000, 32);
	check(rgfxGLSetShape(s, 2, 0, r, 12) == kRgfxGLError, "a negative width");
	region(r, kRgfxGLMaxSurfaceSize + 1, 32);
	check(rgfxGLSetShape(s, 2, 0, r, 12) == kRgfxGLError, "past the largest surface");
	region(r, kRgfxGLMaxSurfaceSize, kRgfxGLMaxSurfaceSize);
	check(rgfxGLSetShape(s, 2, 0, r, 12) == kRgfxGLSuccess, "the largest surface");
	region(r, 0, 32);
	check(rgfxGLSetShape(s, 2, 0, r, 12) == kRgfxGLSuccess && s.width == 1 && s.height == 1,
	      "a size of 0 is 1 x 1");
}

void buffers()
{
	check(rgfxGLDepthOf(4).bytes == 4 && rgfxGLDepthOf(4).format == 0x42475241, "depth 4: BGRA");
	check(rgfxGLDepthOf(3).bytes == 2 && rgfxGLDepthOf(3).format == 0x35353531, "depth 3: 5551");
	check(rgfxGLDepthOf(11).bytes == 8 && rgfxGLDepthOf(11).format == 0x6c363472, "depth 11");
	check(rgfxGLDepthOf(12).bytes == 8 && rgfxGLDepthOf(12).format == 0x52476841, "depth 12");
	check(rgfxGLDepthOf(13).bytes == 16 && rgfxGLDepthOf(13).format == 0x52476641, "depth 13");
	check(rgfxGLDepthOf(15).bytes == 4 && rgfxGLDepthOf(15).format == 0x6c313072, "depth 15");
	check(rgfxGLDepthOf(5).bytes == 0 && rgfxGLDepthOf(0x10).bytes == 0, "depths with none");

	check(rgfxGLPowerOfTwo(0) == 1 && rgfxGLPowerOfTwo(1) == 1 && rgfxGLPowerOfTwo(2) == 2 &&
	          rgfxGLPowerOfTwo(3) == 4 && rgfxGLPowerOfTwo(256) == 256 && rgfxGLPowerOfTwo(257) == 512,
	      "powers of two");
	RgfxGLBuffer b;
	check(rgfxGLBufferOf(3, 0x24, 300, 200, b) && b.width == 300 && b.bytesPerElement == 4 &&
	          b.hasFormat,
	      "colour slot 3");
	check(rgfxGLBufferOf(4, 0x24, 300, 200, b) && b.width == 512 && b.height == 256 &&
	          b.bytesPerElement == 4 && !b.hasFormat,
	      "depth slot 4: powers of two, 4 bytes, no format");
	check(rgfxGLBufferOf(6, 0x24, 300, 200, b) && b.bytesPerElement == 4, "depth slot 6");
	check(rgfxGLBufferOf(7, 0x24, 300, 200, b) && b.bytesPerElement == 1 && !b.hasFormat,
	      "slot 7: 1 byte");
	check(rgfxGLBufferOf(9, 0x24, 1, 1, b) && b.width == 1 && b.height == 1, "slot 9 of 1 x 1");
	check(!rgfxGLBufferOf(10, 0x24, 300, 200, b) && !rgfxGLBufferOf(11, 0x24, 300, 200, b),
	      "slot 10 is never made, and there is no slot 11");
}

void drawableRules()
{
	check(rgfxGLDrawableModeTaken(0xc00) && !rgfxGLDrawableModeTaken(0xc01) &&
	          !rgfxGLDrawableModeTaken(0x400000),
	      "the drawable modes refused");
	check(rgfxGLDrawableModeOn(0xc00, 0x24) == 0xc00, "a surface without buffering bits");
	check(rgfxGLDrawableModeOn(0xc00, 0x824) == 0x800, "the surface's 0x800 wins");
	check(rgfxGLDrawableModeOn(0x800, 0x424) == 0x400, "the surface's 0x400 wins");
	check(rgfxGLSurfaceModeWith(0x24, 0x800, 1) == 0x24, "with others the surface keeps its mode");
	check(rgfxGLSurfaceModeWith(0x24 | 0x1000, 0x1000000000ull, 0) == (0x24 | 0x1000000000ull),
	      "alone, the drawable's bits outside the context's join the kept ones");

	check(rgfxGLDrawableRequestBits(0xc00) == 3, "0xc00");
	check(rgfxGLDrawableRequestBits(0x400) == 2, "0x400: slot 1");
	check(rgfxGLDrawableRequestBits(0x800) == 1, "0x800: slot 0");
	check(rgfxGLDrawableRequestBits(0x40) == 0x10, "0x40: slot 4");
	check(rgfxGLDrawableRequestBits(0x80) == 0x80, "0x80: slot 7");
	check(rgfxGLDrawableRequestBits(0x100) == 4, "0x100: slot 2");
	check(rgfxGLDrawableRequestBits(0x300) == 0xc, "0x300: slots 2 and 3");
	check(rgfxGLDrawableRequestBits(0x4300) == 0x30c, "0x4300: slots 2, 3, 8 and 9");
	check(rgfxGLDrawableRequestBits(0x2100) == 0x24, "0x2100: slot 5 with slot 2");
	check(rgfxGLDrawableRequestBits(0) == 0, "no bits");

	const struct {
		uint64_t index;
		int32_t slot;
	} table[] = {{0, 1}, {1, 0}, {2, -1}, {3, 2}, {4, 3}, {5, -1}, {6, 5}, {7, 6}, {8, -1},
	             {11, -1}, {12, 10}, {13, 4}, {14, 7}, {15, -1}, {0x7fffffff, 0}, {0x80000000, -1}};
	for (const auto &e : table)
		check(rgfxGLSlotOfIndex(e.index) == e.slot, "a lookup's slot");

	// Two drawables on one surface: the surface makes what either asks for.
	RgfxGLSurface s;
	Slots slots;
	uint8_t r[12];
	region(r, 64, 64);
	rgfxGLSetIDMode(s, 1, 0x4024, 0);
	rgfxGLSetShape(s, 2, 0, r, 12);
	const uint32_t first = attach(s, slots, 0x800, 0, 0);
	check(first == 1 && slots.existing == 1, "the first drawable's slot 0");
	attach(s, slots, 0x400, 1, first);
	check(slots.existing == 3, "the second's slot 1 joins it");
	check(s.mode == 0x24, "the second leaves the surface's mode");
}

void sizesAndAnswers()
{
	check(sizeof(RgfxGLSetSurface) == 0x30 && sizeof(RgfxGLConfigStatus) == 0x28, "the sizes");
	RgfxGLSetSurface in;
	std::memset(&in, 0, sizeof in);
	const uint8_t measured[12] = {0x00, 0x0c, 0, 0, 0, 0, 0, 0, 0x0b, 0x10, 0, 0};
	std::memcpy(&in, measured, sizeof measured);
	check(in.mode == 0xc00 && in.surfaceId == 0x100b, "the measured set surface's fields");
	check(__builtin_offsetof(RgfxGLSetSurface, colorDepthMode) == 0xc &&
	          __builtin_offsetof(RgfxGLSetSurface, fixedSource) == 0x1c &&
	          __builtin_offsetof(RgfxGLSetSurface, scaledHeight) == 0x28,
	      "the set surface's offsets");
	check(__builtin_offsetof(RgfxGLConfigStatus, width) == 8 &&
	          __builtin_offsetof(RgfxGLConfigStatus, surfaceHeight) == 0x14 &&
	          __builtin_offsetof(RgfxGLConfigStatus, field_18) == 0x18 &&
	          __builtin_offsetof(RgfxGLConfigStatus, field_20) == 0x20,
	      "the answer's offsets");
	check(kRgfxGLDrawableUserClientType == 7 && kRgfxGLSurfaceUserClientType == 0x20, "the types");
}
} // namespace

int main()
{
	measuredWindow();
	setIDModeRules();
	setShapeRules();
	buffers();
	drawableRules();
	sizesAndAnswers();
	std::printf("gl_surfaces_test: %d checks passed\n", checks);
	return 0;
}
