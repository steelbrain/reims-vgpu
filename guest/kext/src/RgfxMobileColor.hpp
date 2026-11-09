// RgfxMobileColor: the colour matrix an arm64 WindowServer sets on its IOMobileFramebuffer
// (ReimsGraphicsMobile.cpp's swap_set_color_matrix), as the DISPLAY_COLOR matrix stage the host
// applies (reims_graphics_wire.h, RGFX_COLOR_MATRIX). Header-only and free of the kernel's headers,
// so its native test (tests/mobile_color_test.cpp) builds it as the kext does.
//
// What the guest sends: IOMobileFramebufferSwapSetColorMatrix(fb, matrix, function, field) sends
// IOMobileFramebuffer's user client selector 69 a structure of 0x88 bytes: a byte at 0x00, set for
// no matrix; fifteen signed 64-bit words from 0x08, each a float of the caller's times 2^32
// rounded toward zero; the function at 0x80 and the field at 0x84. The user client calls the
// display's swap_set_color_matrix with the words (NULL where the byte is set), the function and
// the field, outside the display's gate. The words, as the display gets them:
//   0x00  field_00: three words
//   0x18  the matrix's nine words, row by row: row i is output channel i's red, green and blue
//   0x60  field_60: three words
// (read on macOS 14.6.1, 15.6.1, 26.6.2 and 27.0.1, which lay it out alike). WindowServer sets its
// accessibility matrix so at each swap until the call succeeds, with function 2, field 0, its
// matrix's three rows at 0x18 and zeros in field_00 and field_60, and logs "IOMFB was not able to
// set accessibility matrix." at each refusal (27.0.1). IOMobileFramebufferLegacy refuses the call
// (kIOReturnUnsupported, 0xe00002c7) on each of those releases where its subclass does not take
// it.
//
// What the display does with it: swap_set_color_matrix answers 0 for any function and field, and
// shows the nine words, or the identity for NULL, reading neither field_00 nor field_60.
// rgfxMobileColorMatrix makes them the wire's three rows of four floats, out_i = row_i[0] * r +
// row_i[1] * g + row_i[2] * b + row_i[3]: row i's first three entries each word converted to a
// double, times 2^-32, rounded to a float, and its fourth 0. The conversion is integer arithmetic
// with the same two roundings (to nearest, ties to even: a 64-bit integer to a double, then a
// double to a float), so the kext needs no floating point.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_MOBILE_COLOR_HPP_
#define RGFX_MOBILE_COLOR_HPP_

#include <stdint.h>

#include "reims_graphics_wire.h"

// The matrix's rows, the words before them, and the words in all.
constexpr uint32_t kRgfxMobileMatrixRows = 3;
constexpr uint32_t kRgfxMobileMatrixFirstWord = 3;
constexpr uint32_t kRgfxMobileMatrixWords = 15;

// The function QuartzCore sets its accessibility matrix with (read).
constexpr uint32_t kRgfxMobileAccessibilityMatrix = 2;

// The bits of `v` rounded to nearest, ties to even, at `drop` low bits (drop < 64): the kept bits.
inline uint64_t rgfxRoundNearestEven(uint64_t v, uint32_t drop)
{
	if (drop == 0)
		return v;
	const uint64_t kept = v >> drop;
	const uint64_t rest = v & ((uint64_t(1) << drop) - 1);
	const uint64_t half = uint64_t(1) << (drop - 1);
	return kept + ((rest > half || (rest == half && (kept & 1))) ? 1 : 0);
}

// The number of significant bits of a nonzero `v`.
inline uint32_t rgfxBitLength(uint64_t v)
{
	uint32_t n = 0;
	while (v) {
		n++;
		v >>= 1;
	}
	return n;
}

// The bits of the float `(float)((double)fixed * 2^-32)`: the signed 64-bit word converted to a
// double, scaled, and rounded to a float, each rounding to nearest, ties to even.
inline uint32_t rgfxFixed32ToFloatBits(int64_t fixed)
{
	if (fixed == 0)
		return 0;
	const uint32_t sign = fixed < 0 ? 0x80000000u : 0;
	uint64_t mag = fixed < 0 ? uint64_t(0) - uint64_t(fixed) : uint64_t(fixed);
	// To a double: 53 significant bits, value mag * 2^shift.
	int32_t shift = 0;
	uint32_t n = rgfxBitLength(mag);
	if (n > 53) {
		const uint32_t drop = n - 53;
		mag = rgfxRoundNearestEven(mag, drop);
		shift += static_cast<int32_t>(drop);
		if (mag >> 53) {
			mag >>= 1;
			shift++;
		}
		n = 53;
	}
	// To a float: 24 significant bits.
	if (n > 24) {
		const uint32_t drop = n - 24;
		mag = rgfxRoundNearestEven(mag, drop);
		shift += static_cast<int32_t>(drop);
		if (mag >> 24) {
			mag >>= 1;
			shift++;
		}
	} else {
		mag <<= 24 - n;
		shift -= static_cast<int32_t>(24 - n);
	}
	// mag * 2^(shift - 32), mag in [2^23, 2^24): its exponent is shift - 32 + 23, biased by 127.
	// A word's magnitude is 2^-32 to 2^31, inside a float's normal range.
	const uint32_t exponent = static_cast<uint32_t>(shift - 32 + 23 + 127);
	return sign | (exponent << 23) | static_cast<uint32_t>(mag & 0x7fffff);
}

// The DISPLAY_COLOR matrix rows for the words swap_set_color_matrix takes, or the identity for
// NULL: row i's first three entries the matrix's row i, its fourth 0.
inline void rgfxMobileColorMatrix(const int64_t *words, RgfxColorEntry rows[kRgfxMobileMatrixRows])
{
	constexpr uint32_t kOne = 0x3f800000u;
	for (uint32_t i = 0; i < kRgfxMobileMatrixRows; i++) {
		uint32_t row[4] = {0, 0, 0, 0};
		for (uint32_t j = 0; j < 3; j++)
			row[j] = words ? rgfxFixed32ToFloatBits(words[kRgfxMobileMatrixFirstWord + 3 * i + j])
			               : (i == j ? kOne : 0);
		rows[i].x = row[0];
		rows[i].y = row[1];
		rows[i].z = row[2];
		rows[i].w = row[3];
	}
}

#endif // RGFX_MOBILE_COLOR_HPP_
