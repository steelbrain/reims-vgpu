// RgfxMobileICC: the colour blocks an arm64 WindowServer sets on its IOMobileFramebuffer
// (ReimsGraphicsMobile.cpp's set_block), ICC matrices and curves, and which of them the display
// takes. Header-only and free of the kernel's headers, so its native test
// (tests/mobile_icc_test.cpp) builds it as the kext does. The kext does no floating point: every
// value is a signed 64-bit word with 32 fractional bits (Q32), products are taken in 128 bits, and
// powers go through a bitwise log2 and a table of 2^(2^-k).
//
// What the guest sends: QuartzCore's IOMobileFramebufferSwapSetICCMatrix and
// IOMobileFramebufferSwapSetICCCurve call IOMobileFramebufferSetBlock(fb, type, scalars, count,
// data, size), which the user client hands the display's set_block with offset 0, in WindowServer's
// thread and outside the display's gate (read on 27.0.1's QuartzCore, measured on 27.0.1). The
// blocks, as the display gets them (measured on 27.0.1), in 27.0.1's numbers:
//   matrix, type 0x33, 0x40 bytes; scalars {swap id, matrix location, layer}
//     0x00  field_00: a word, 1
//     0x04  field_04: three floats, 0
//     0x10  the 3x3 matrix's floats, row by row
//     0x34  field_34: three floats, 0
//   curve, type 0x34, 0x290 bytes; scalars {swap id, curve location, channel mask, layer}
//     0x00  field_00: a word, 1
//     0x04  field_04: a word, 0
//     0x08  the curve's type
//     0x0c  seven floats, the parameters g, a, b, c, d, e, f (the order QuartzCore prints them)
// The numbers are the release's (measured, each release's set_block calls at login): 14.6.1,
// 15.6.1 and 26.6.2 number each block one below 27.0.1, the matrix 0x32 and the curve 0x33 (and a
// block that 27.0.1 sends as 0x6c, 0x6b), the curve block is 0x248 bytes on 14.6.1 and 0x258 on
// 15.6.1, and 14.6.1's curve locations from Blend2XYZ on are one lower, with the same fields and
// values in the same order. rgfxICCReleaseOf gives each measured release's numbering by the
// kernel's major version, and the display refuses every block of another release, and every block
// at a location not measured on its own.
// The locations and curve types are QuartzCore's (its print tables): kRgfxICCMatrix* and
// kRgfxICCCurve* below. The parametric types are the ICC's, in the forms QuartzCore prints:
//   Type0  y = x^g
//   Type1  y = (a x + b)^g            where x >= -b/a, else 0
//   Type2  y = (a x + b)^g + c        where x >= -b/a, else c
//   Type3  y = (a x + b)^g            where x >= d, else c x
//   Type4  y = (a x + b)^g + e        where x >= d, else c x + f
// IOMobileFramebufferLegacy's own set_block answers every block kIOReturnUnsupported
// (0xe00002c7) on 14.6.1, 15.6.1, 26.6.2, 27.0.1 and 27.2.1 (read from each release's
// IOMobileGraphicsFamily), and WindowServer logs "Failed to set ICC matrix" or "Failed to set ICC
// curve" at each refused block of the four colour-space locations (Src2XYZ, XYZ2Blend, Blend2XYZ,
// XYZ2Device), and nothing at the others' (27.0.1: 12 lines a burst, one for each of those
// blocks).
//
// What WindowServer sets, a chain per layer: the layer's Src2XYZ curve and matrix, its XYZ2Blend
// matrix and curve, then, after the layers are blended, the display's Blend2XYZ curve and matrix
// and its XYZ2Device matrix and curve, with the tone map, user gamma and black point stages
// between.
// It sets each layer's half (Src2XYZ, XYZ2Blend) and the display's (Blend2XYZ, XYZ2Device) as a
// pair, the first before the second in one burst: at its start, at each change of its colour space
// and at each change of the display's colour profile. In every burst measured (27.0.1: three at
// login, one each after switching the profile to Generic RGB, to Display P3 and back), each pair
// of layer 0 and of the display is the identity: the matrices' product within 5.9e-6 of it and the
// curves' composition within 2.2e-5 of x. WindowServer composites in the profile's own space, so
// the profile changes what it draws and the display shows it as drawn. Layer 1, the cursor's, is
// the one that is not: its source is standard sRGB and its blend space the display's (the
// published primaries and white, 0.0024 off sRGB's product at login), and another profile's after a
// switch (0.25 to 0.35 off).
//
// What the display does: it applies no chain to layer 0 or to the display's half. The host shows
// layer 0's surface as it is, so the display takes a block there only where the chain it leaves is
// the identity, which is then what the display shows, and refuses the rest kIOReturnUnsupported, as
// the class does (rgfxICCTake). The cursor's layer, 1, is the exception where the host took
// RGFX_FEATURE_CURSOR_COLOR (RgfxICCCursor below): the host converts the cursor's image through
// the layer's chain, so each of its four colour-space blocks the display can read is taken and
// kept, and the chain goes to the host (CURSOR_COLOR, rgfxICCCursorColor) at the next swap. Where
// the host did not take it, layer 1 is judged as the others are. The rules:
// - layer 1's Src2XYZ curve and matrix and XYZ2Blend matrix and curve, where the host converts
//   the cursor, are taken where the display reads them, whatever they hold, and a Src2XYZ block it
//   cannot read is refused and drops its stage from the chain the host is sent;
// - a pair's first block (Src2XYZ, Blend2XYZ) is held, and taken;
// - its second (a layer's XYZ2Blend, the display's XYZ2Device) is taken where it inverts what is
//   held: the matrices' product within kRgfxICCTolerance of the identity, entry by entry, and the
//   curves' composition within it of x at kRgfxICCSamples + 1 inputs spread over [0, 1], channel by
//   channel. Judging the whole chain at each block would refuse each pair's first block, which
//   leaves the chain off the identity until the second comes;
// - a second block with nothing held is taken: no Src2XYZ is ever set for layer 2, nor for layer 1
//   until the cursor's first image, so no surface of the layer is drawn through a chain then
//   (inferred from the bursts at login, not read);
// - a first block the display cannot read is refused, and so is every second block of its pair
//   until a first block the display reads is held again;
// - user gamma (UserGammaAdjustment, the curves CGSetDisplayTransferByFormula and
//   CGSetDisplayTransferByTable set, measured) is refused, whatever it holds: where the display
//   would apply it is not known (WindowServer sends Type2 2.2/1/0/0 with the display's own profile
//   and 1/1/0/0 with Generic RGB, with no transfer set);
// - a block at any other location is taken where it is the identity by itself, within
//   kRgfxICCTolerance;
// - a block of another type than the release's two, any block of a release rgfxICCReleaseOf does
//   not know, a block at a location not measured on its release, a chunk (a nonzero offset), a
//   layer past kRgfxICCLayers, or one the display cannot read (rgfxICCParseMatrix,
//   rgfxICCParseCurve) is refused.
// The tolerance, 2^-14, is above the floats' rounding in what WindowServer sends (2.2e-5 at most,
// measured) and a sixteenth of an 8-bit step.
//
// Our own code; it depends on nothing but <stddef.h>, <stdint.h> and the wire header.
#ifndef RGFX_MOBILE_ICC_HPP_
#define RGFX_MOBILE_ICC_HPP_

#include <stddef.h>
#include <stdint.h>

#include "reims_graphics_wire.h"

// The matrix block's size, on every release measured.
constexpr uint32_t kRgfxICCMatrixBlockSize = 0x40;

// The matrix locations (QuartzCore's IOMFBICCMatLoc).
enum : uint32_t {
	kRgfxICCMatrixSrc2XYZ = 0,
	kRgfxICCMatrixXYZ2Blend = 1,
	kRgfxICCMatrixPreToneMap = 2,
	kRgfxICCMatrixPostToneMap = 3,
	kRgfxICCMatrixBlend2XYZ = 4,
	kRgfxICCMatrixXYZ2Device = 5,
	kRgfxICCMatrixHarmony = 6,
	kRgfxICCMatrixUserAdjust = 7,
	kRgfxICCMatrixLocations = 8,
};

// The curve locations (QuartzCore's IOMFBCurveLoc): 2 to 11 are the tone map's.
enum : uint32_t {
	kRgfxICCCurveSrc2XYZ = 0,
	kRgfxICCCurveXYZ2Blend = 1,
	kRgfxICCCurveBlend2XYZ = 12,
	kRgfxICCCurveDstMid = 13,
	kRgfxICCCurveSystemGamma = 14,
	kRgfxICCCurveXYZ2Device = 15,
	kRgfxICCCurveUserGammaAdjustment = 16,
	kRgfxICCCurveBlackpoint = 17,
	kRgfxICCCurveDICOM = 18,
	kRgfxICCCurveLocations = 19,
};

// The curve types (QuartzCore's) the display evaluates; the rest (6 PQEngamma, 7 PQDegamma,
// 8 Table, 9 NonUniformTable, 10 and 11 the fixed HLG and PQ, 12 Polynomial and the tone map's) it
// refuses.
enum : uint32_t {
	kRgfxICCCurveBypass = 0,
	kRgfxICCCurveType0 = 1,
	kRgfxICCCurveType1 = 2,
	kRgfxICCCurveType2 = 3,
	kRgfxICCCurveType3 = 4,
	kRgfxICCCurveType4 = 5,
};

// A release's numbering of the blocks: their types, the curve block's size, the curve location
// its Blend2XYZ takes (the later stages follow it in 27.0.1's order, each one lower on a release
// whose tone map has a stage fewer), and the locations measured on it, bit by bit, in its own
// numbers. The kRgfxICCMatrix* and kRgfxICCCurve* above are 27.0.1's; a block at a location not
// measured on its release is refused, as an unknown block is, until a capture shows it.
struct RgfxICCRelease {
	uint32_t matrixBlock;
	uint32_t curveBlock;
	uint32_t curveBlockSize;
	uint32_t curveBlend2XYZ;
	uint32_t matrixMeasured;
	uint32_t curveMeasured;
};

// The numbering of the release whose kernel's major version (version_major) is `kernelMajor`,
// measured from its set_block calls at login; false for a release not measured, whose blocks the
// display refuses. A location's bit is set only where a capture showed the stage on that release:
// 14.6.1 (job m3-max-36326) sent Src2XYZ, XYZ2Blend, the tone map's 3, 6 and 7, Blend2XYZ at 11,
// SystemGamma at 13, XYZ2Device at 14, UserGammaAdjustment at 15 and Blackpoint at 16, its stages
// from Blend2XYZ on one lower than 27.0.1's, and no DstMid or DICOM; 15.6.1 (m3-max-36411) and
// 26.6.2 (m3-max-36238) sent 27.0.1's numbers where they sent them: the tone map's 3, 6, 7 and 11
// (11 a tone map stage 14.6.1 sends none at), Blend2XYZ 12, SystemGamma 14 and XYZ2Device 15; all
// three sent the matrices at 0 to 5. The curve block is 0x248 bytes on 14.6.1, 0x258 on 15.6.1 and
// 0x290 on 26.6.2 and 27, with the same fields where the display reads them. 27's locations are
// read from 27.0.1's QuartzCore (its print tables); 27.2.1's kernel is Darwin 27 too and sends the
// same numbers (measured, m3-max-36378).
inline bool rgfxICCReleaseOf(int kernelMajor, RgfxICCRelease *release)
{
	constexpr uint32_t toneMap = (1u << 3) | (1u << 6) | (1u << 7);
	switch (kernelMajor) {
	case 23: // 14.6.1
		*release = {0x32, 0x33, 0x248, 11, 0x3f,
		            0x3u | toneMap | (1u << 11) | (1u << 13) | (1u << 14) | (1u << 15) | (1u << 16)};
		return true;
	case 24: // 15.6.1
		*release = {0x32, 0x33, 0x258, 12, 0x3f,
		            0x3u | toneMap | (1u << 11) | (1u << 12) | (1u << 14) | (1u << 15)};
		return true;
	case 25: // 26.6.2
		*release = {0x32, 0x33, 0x290, 12, 0x3f,
		            0x3u | toneMap | (1u << 11) | (1u << 12) | (1u << 14) | (1u << 15)};
		return true;
	case 27: // 27.0.1, 27.2.1
		*release = {0x33, 0x34, 0x290, kRgfxICCCurveBlend2XYZ, (1u << kRgfxICCMatrixLocations) - 1,
		            (1u << kRgfxICCCurveLocations) - 1};
		return true;
	default:
		return false;
	}
}

// A curve location of `release`, in its own numbers, as 27.0.1 numbers it; false where it was not
// measured on the release.
inline bool rgfxICCCurveLocation(const RgfxICCRelease &release, uint64_t location,
                                 uint32_t *canonical)
{
	if (location >= 32 || !(release.curveMeasured & (1u << location)))
		return false;
	*canonical = location < release.curveBlend2XYZ
	                 ? static_cast<uint32_t>(location)
	                 : static_cast<uint32_t>(location) - release.curveBlend2XYZ + kRgfxICCCurveBlend2XYZ;
	return *canonical < kRgfxICCCurveLocations;
}

// A matrix location of `release`; false where it was not measured on the release.
inline bool rgfxICCMatrixLocation(const RgfxICCRelease &release, uint64_t location)
{
	return location < kRgfxICCMatrixLocations && (release.matrixMeasured & (1u << location));
}

// The layers whose pairs the display keeps (WindowServer sets 0 to 2: 27.0.1).
constexpr uint32_t kRgfxICCLayers = 4;

// The cursor's layer: WindowServer gives its cursor as each swap's layer 1 (RgfxMobileCursor.hpp),
// the display's CursorPlane.
constexpr uint32_t kRgfxICCCursorLayer = 1;

// One, in Q32.
constexpr int64_t kRgfxQ32One = int64_t(1) << 32;

// How far from the identity a chain may be, in Q32: 2^-14.
constexpr int64_t kRgfxICCTolerance = int64_t(1) << 18;

// The intervals a curve is sampled over [0, 1] at.
constexpr uint32_t kRgfxICCSamples = 1024;

// ---- Fixed point ------------------------------------------------------------------------------

// The float with bits `bits` in Q32, rounded to nearest (ties away from zero), into `out`; false
// for a NaN, an infinity, or a magnitude of 2^31 or more.
inline bool rgfxFloatBitsToQ32(uint32_t bits, int64_t *out)
{
	const uint32_t exponent = (bits >> 23) & 0xff;
	const bool negative = (bits >> 31) != 0;
	if (exponent == 0xff)
		return false;
	if (exponent == 0) { // zero or subnormal: below 2^-126, 0 in Q32
		*out = 0;
		return true;
	}
	const uint64_t mantissa = (bits & 0x7fffff) | 0x800000;
	// value = mantissa * 2^(exponent - 150), in Q32 mantissa * 2^(exponent - 118).
	const int32_t shift = static_cast<int32_t>(exponent) - 118;
	uint64_t mag;
	if (shift >= 0) {
		if (shift > 39) // mantissa < 2^24: 2^(24 + 39) = 2^63
			return false;
		mag = mantissa << shift;
	} else if (shift > -64) {
		const uint32_t drop = static_cast<uint32_t>(-shift);
		mag = (mantissa + (uint64_t(1) << (drop - 1))) >> drop;
	} else {
		mag = 0;
	}
	if (mag >= (uint64_t(1) << 63))
		return false;
	*out = negative ? -static_cast<int64_t>(mag) : static_cast<int64_t>(mag);
	return true;
}

// a * b in Q32, rounded toward negative infinity.
inline int64_t rgfxQ32Mul(int64_t a, int64_t b)
{
	return static_cast<int64_t>((static_cast<__int128>(a) * b) >> 32);
}

// log2(x) in Q32 for x > 0 in Q32: the integer part from the bit length, then 32 fraction bits by
// squaring the mantissa in Q62. Its error is below 2^-30.
inline int64_t rgfxQ32Log2(int64_t x)
{
	const uint64_t v = static_cast<uint64_t>(x);
	const uint32_t n = 64 - static_cast<uint32_t>(__builtin_clzll(v));
	int64_t result = (static_cast<int64_t>(n) - 33) * kRgfxQ32One;
	// The mantissa in [1, 2), in Q62: its top bit at 62.
	uint64_t m = n > 63 ? v >> (n - 63) : v << (63 - n);
	for (int bit = 31; bit >= 0; bit--) {
		const unsigned __int128 sq = static_cast<unsigned __int128>(m) * m;
		m = static_cast<uint64_t>(sq >> 62);
		if (m >> 63) { // m >= 2
			m >>= 1;
			result += int64_t(1) << bit;
		}
	}
	return result;
}

// 2^(2^-k), k = 1 to 32, in Q62.
constexpr uint64_t kRgfxExp2Steps[32] = {
    0x5a827999fcef3242ull, 0x4c1bf828c6dc54b8ull, 0x45cae0f1f545eb73ull, 0x42d561b3e6243d8aull,
    0x4166c34c5615d0ecull, 0x40b268f9de0183baull, 0x4058f6a7ecccd5b6ull, 0x402c6be96af2fb58ull,
    0x4016321b687027a8ull, 0x400b18178ba33b14ull, 0x40058bce410147e8ull, 0x4002c5d7bff71dafull,
    0x400162e807ee7e5bull, 0x4000b1730df6a524ull, 0x400058b9497b8152ull, 0x40002c5c955dd701ull,
    0x4000162e46d6f26cull, 0x40000b1722757b1bull, 0x4000058b90fd3e0cull, 0x400002c5c86f3f26ull,
    0x40000162e433c79bull, 0x400000b17218edd0ull, 0x40000058b90c3968ull, 0x4000002c5c860d54ull,
    0x400000162e4302d2ull, 0x4000000b17218073ull, 0x400000058b90bffcull, 0x40000002c5c85fefull,
    0x4000000162e42ff3ull, 0x40000000b17217f9ull, 0x4000000058b90bfcull, 0x400000002c5c85feull,
};

// 2^y in Q32 for y in Q32 below 31: 2 to y's integer part times the product of 2^(2^-k) over
// y's fraction bits, in Q62. 0 where the result is below 2^-33.
inline int64_t rgfxQ32Exp2(int64_t y)
{
	const int64_t whole = y >> 32; // floor
	const uint32_t frac = static_cast<uint32_t>(y);
	uint64_t r = uint64_t(1) << 62;
	for (int k = 1; k <= 32; k++)
		if (frac & (uint32_t(1) << (32 - k)))
			r = static_cast<uint64_t>((static_cast<unsigned __int128>(r) * kRgfxExp2Steps[k - 1]) >>
			                          62);
	// r in [1, 2) in Q62; the result is r * 2^whole, in Q32 r >> (30 - whole).
	const int64_t shift = 30 - whole;
	if (shift >= 63)
		return 0;
	if (shift <= 0)
		return static_cast<int64_t>(r << -shift);
	return static_cast<int64_t>((r + (uint64_t(1) << (shift - 1))) >> shift);
}

// x^g in Q32 for x >= 0 and g > 0 in Q32; 0 for x <= 0, and INT64_MAX past 2^31.
inline int64_t rgfxQ32Pow(int64_t x, int64_t g)
{
	if (x <= 0)
		return 0;
	const int64_t y = rgfxQ32Mul(g, rgfxQ32Log2(x));
	if (y >= int64_t(31) * kRgfxQ32One)
		return INT64_MAX;
	return rgfxQ32Exp2(y);
}

// ---- The blocks -------------------------------------------------------------------------------

// A matrix block's matrix, row by row, in Q32.
struct RgfxICCMatrix {
	int64_t m[9];
};

// A curve block's curve: its type and its seven parameters in Q32 (g, a, b, c, d, e, f).
struct RgfxICCCurve {
	uint32_t type;
	int64_t p[7];
};

// A little-endian word of a block.
inline uint32_t rgfxICCWord(const uint8_t *data, size_t at)
{
	return uint32_t(data[at]) | uint32_t(data[at + 1]) << 8 | uint32_t(data[at + 2]) << 16 |
	       uint32_t(data[at + 3]) << 24;
}

// The matrix block `data` of `size` bytes with its `count` scalars: its location, layer and
// matrix. False for a block of another size, fewer than 3 scalars, field_00 other than 1, a
// nonzero field_04 or field_34 (what either means is not stated), an unknown location, or an entry
// that is not a finite float below 2^31.
inline bool rgfxICCParseMatrix(const RgfxICCRelease &release, const uint64_t *scalars,
                               uint32_t count, const uint8_t *data, size_t size,
                               uint32_t *location, uint32_t *layer, RgfxICCMatrix *matrix)
{
	if (!scalars || count < 3 || !data || size != kRgfxICCMatrixBlockSize)
		return false;
	if (rgfxICCWord(data, 0x00) != 1)
		return false;
	for (size_t at = 0x04; at < 0x10; at += 4)
		if (rgfxICCWord(data, at) != 0 || rgfxICCWord(data, at + 0x30) != 0)
			return false;
	for (uint32_t i = 0; i < 9; i++)
		if (!rgfxFloatBitsToQ32(rgfxICCWord(data, 0x10 + 4 * i), &matrix->m[i]))
			return false;
	if (!rgfxICCMatrixLocation(release, scalars[1]) || scalars[2] > 0xffffffffu)
		return false;
	*location = static_cast<uint32_t>(scalars[1]);
	*layer = static_cast<uint32_t>(scalars[2]);
	return true;
}

// The curve block `data` of `size` bytes with its `count` scalars: its location, channel mask
// (bit 0 red, 1 green, 2 blue), layer and curve. False for a block of another size, fewer than 4
// scalars, field_00 other than 1, a nonzero field_04, an unknown location, a mask of no channel or
// another bit, a type the display does not evaluate, a parameter that is not a finite float below
// 2^31, or a Type1 or Type2 whose a is not positive (so x >= -b/a is a x + b >= 0).
inline bool rgfxICCParseCurve(const RgfxICCRelease &release, const uint64_t *scalars,
                              uint32_t count, const uint8_t *data, size_t size, uint32_t *location,
                              uint32_t *mask, uint32_t *layer, RgfxICCCurve *curve)
{
	if (!scalars || count < 4 || !data || size != release.curveBlockSize || size < 0x28)
		return false;
	if (rgfxICCWord(data, 0x00) != 1 || rgfxICCWord(data, 0x04) != 0)
		return false;
	uint32_t canonical = 0;
	if (!rgfxICCCurveLocation(release, scalars[1], &canonical) || scalars[2] == 0 ||
	    scalars[2] > 7 || scalars[3] > 0xffffffffu)
		return false;
	curve->type = rgfxICCWord(data, 0x08);
	if (curve->type > kRgfxICCCurveType4)
		return false;
	for (uint32_t i = 0; i < 7; i++) {
		curve->p[i] = 0;
		if (curve->type != kRgfxICCCurveBypass &&
		    !rgfxFloatBitsToQ32(rgfxICCWord(data, 0x0c + 4 * i), &curve->p[i]))
			return false;
	}
	if ((curve->type == kRgfxICCCurveType1 || curve->type == kRgfxICCCurveType2) && curve->p[1] <= 0)
		return false;
	*location = canonical;
	*mask = static_cast<uint32_t>(scalars[2]);
	*layer = static_cast<uint32_t>(scalars[3]);
	return true;
}

// ---- Evaluation -------------------------------------------------------------------------------

// The curve at x in Q32, in QuartzCore's forms.
inline int64_t rgfxICCCurveAt(const RgfxICCCurve &c, int64_t x)
{
	const int64_t g = c.p[0], a = c.p[1], b = c.p[2], cc = c.p[3], d = c.p[4], e = c.p[5],
	              f = c.p[6];
	switch (c.type) {
	case kRgfxICCCurveType0:
		return rgfxQ32Pow(x, g);
	case kRgfxICCCurveType1:
	case kRgfxICCCurveType2: {
		const int64_t base = rgfxQ32Mul(a, x) + b;
		const int64_t add = c.type == kRgfxICCCurveType2 ? cc : 0;
		return base >= 0 ? rgfxQ32Pow(base, g) + add : add;
	}
	case kRgfxICCCurveType3:
		return x >= d ? rgfxQ32Pow(rgfxQ32Mul(a, x) + b, g) : rgfxQ32Mul(cc, x);
	case kRgfxICCCurveType4:
		return x >= d ? rgfxQ32Pow(rgfxQ32Mul(a, x) + b, g) + e : rgfxQ32Mul(cc, x) + f;
	default:
		return x;
	}
}

// Whether `second` after `first` (or `first` alone, where `second` is null) is within
// kRgfxICCTolerance of x at x = k / kRgfxICCSamples, k = 0 to kRgfxICCSamples.
inline bool rgfxICCCurvesAreIdentity(const RgfxICCCurve &first, const RgfxICCCurve *second)
{
	if (first.type == kRgfxICCCurveBypass && (!second || second->type == kRgfxICCCurveBypass))
		return true;
	for (uint32_t k = 0; k <= kRgfxICCSamples; k++) {
		const int64_t x = (kRgfxQ32One / kRgfxICCSamples) * k;
		int64_t y = rgfxICCCurveAt(first, x);
		if (second)
			y = rgfxICCCurveAt(*second, y);
		const int64_t diff = y - x;
		if (diff > kRgfxICCTolerance || diff < -kRgfxICCTolerance)
			return false;
	}
	return true;
}

// The product a * b of two matrices in Q32.
inline RgfxICCMatrix rgfxICCMatrixMul(const RgfxICCMatrix &a, const RgfxICCMatrix &b)
{
	RgfxICCMatrix r;
	for (uint32_t i = 0; i < 3; i++)
		for (uint32_t j = 0; j < 3; j++) {
			__int128 sum = 0;
			for (uint32_t k = 0; k < 3; k++)
				sum += static_cast<__int128>(a.m[3 * i + k]) * b.m[3 * k + j];
			r.m[3 * i + j] = static_cast<int64_t>(sum >> 32);
		}
	return r;
}

// Whether every entry of `m` is within kRgfxICCTolerance of the identity's.
inline bool rgfxICCMatrixIsIdentity(const RgfxICCMatrix &m)
{
	for (uint32_t i = 0; i < 9; i++) {
		const int64_t diff = m.m[i] - ((i % 4 == 0) ? kRgfxQ32One : 0);
		if (diff > kRgfxICCTolerance || diff < -kRgfxICCTolerance)
			return false;
	}
	return true;
}

// ---- What the display takes -------------------------------------------------------------------

// What the display holds of a pair's first block: nothing, a block it read, or one it refused.
enum RgfxICCHeld : uint8_t {
	kRgfxICCHeldNone,
	kRgfxICCHeldRead,
	kRgfxICCHeldRefused,
};

// A pair's first matrix, and its first curve channel by channel.
struct RgfxICCPair {
	RgfxICCHeld matrixHeld;
	RgfxICCHeld curveHeld[3];
	RgfxICCMatrix matrix;
	RgfxICCCurve curve[3];
};

// The cursor layer's chain as WindowServer set it, for the host, which converts the cursor's
// image through it (CURSOR_COLOR): each stage the display holds, its values as the block carried
// them (the floats as bits, for the host) and as the display reads them (Q32, to judge the
// identity). `host` says the host took RGFX_FEATURE_CURSOR_COLOR; `changes` counts the changes to
// the chain, and `posted` the count the last CURSOR_COLOR the host was sent carried.
struct RgfxICCCursor {
	bool host;
	uint32_t changes;
	uint32_t posted;
	bool toXYZHeld;
	bool fromXYZHeld;
	bool sourceHeld[3];
	bool blendHeld[3];
	uint32_t toXYZ[9];
	uint32_t fromXYZ[9];
	RgfxCursorCurve source[3];
	RgfxCursorCurve blend[3];
	RgfxICCMatrix toXYZQ;
	RgfxICCMatrix fromXYZQ;
	RgfxICCCurve sourceQ[3];
	RgfxICCCurve blendQ[3];
};

// A display's pairs: each layer's (Src2XYZ, XYZ2Blend) and the display's (Blend2XYZ,
// XYZ2Device), its release's numbering (rgfxICCReleaseOf), and the cursor layer's chain. Zeroed,
// it holds nothing and takes nothing.
struct RgfxICCState {
	RgfxICCPair layer[kRgfxICCLayers];
	RgfxICCPair display;
	RgfxICCRelease release;
	RgfxICCCursor cursor;
};

// Records a matrix block of the cursor's chain into `bits` and `q`, from its data as
// rgfxICCParseMatrix read it into `m`.
inline void rgfxICCCursorMatrix(uint32_t bits[9], RgfxICCMatrix *q, const uint8_t *data,
                                const RgfxICCMatrix &m)
{
	for (uint32_t i = 0; i < 9; i++)
		bits[i] = rgfxICCWord(data, 0x10 + 4 * i);
	*q = m;
}

// Records a curve block of the cursor's chain, for the channels of `mask`, into `wire` and `q`
// and their `held`, from its data as rgfxICCParseCurve read it into `c`; a Bypass carries no
// parameters.
inline void rgfxICCCursorCurve(RgfxCursorCurve wire[3], RgfxICCCurve q[3], bool held[3],
                               uint32_t mask, const uint8_t *data, const RgfxICCCurve &c)
{
	for (uint32_t ch = 0; ch < 3; ch++) {
		if (!(mask & (1u << ch)))
			continue;
		wire[ch].form = c.type;
		for (uint32_t i = 0; i < 7; i++)
			wire[ch].params[i] =
			    c.type == kRgfxICCCurveBypass ? 0 : rgfxICCWord(data, 0x0c + 4 * i);
		q[ch] = c;
		held[ch] = true;
	}
}

// Whether the cursor's chain has a change the host was not sent; read with no lock held, as each
// swap asks before it takes the lock (rgfxICCCursorColor).
inline bool rgfxICCCursorDue(const RgfxICCState *state)
{
	const RgfxICCCursor &c = state->cursor;
	return c.host && __atomic_load_n(&c.changes, __ATOMIC_RELAXED) !=
	                     __atomic_load_n(&c.posted, __ATOMIC_RELAXED);
}

// Whether the chain the cursor holds is whole and, within kRgfxICCTolerance, the identity: the
// matrices' product, entry by entry, and each channel's curves' composition at kRgfxICCSamples + 1
// inputs, a channel whose curves are another's checked once.
inline bool rgfxICCCursorIsIdentity(const RgfxICCCursor &c)
{
	if (!rgfxICCMatrixIsIdentity(rgfxICCMatrixMul(c.fromXYZQ, c.toXYZQ)))
		return false;
	for (uint32_t ch = 0; ch < 3; ch++) {
		bool same = false;
		for (uint32_t before = 0; before < ch && !same; before++) {
			same = c.sourceQ[before].type == c.sourceQ[ch].type &&
			       c.blendQ[before].type == c.blendQ[ch].type;
			for (uint32_t i = 0; same && i < 7; i++)
				same = c.sourceQ[before].p[i] == c.sourceQ[ch].p[i] &&
				       c.blendQ[before].p[i] == c.blendQ[ch].p[i];
		}
		if (!same && !rgfxICCCurvesAreIdentity(c.sourceQ[ch], &c.blendQ[ch]))
			return false;
	}
	return true;
}

// The CURSOR_COLOR the cursor's chain asks for, but its header and display, into `*message`, and
// the count of changes it carries into `*changes`, where the chain changed since the host was last
// sent one: `on` where every stage of the chain is held and the chain is not the identity within
// kRgfxICCTolerance, so an identity costs the host nothing, and the cursor shows as it is until
// the chain is whole (no Src2XYZ is set for layer 1 before the cursor's first image: 27.0.1).
// False where nothing is due. rgfxICCCursorPosted records the post once it went out.
inline bool rgfxICCCursorColor(RgfxICCState *state, RgfxCursorColor *message, uint32_t *changes)
{
	if (!rgfxICCCursorDue(state))
		return false;
	const RgfxICCCursor &c = state->cursor;
	*message = RgfxCursorColor{};
	*changes = c.changes;
	bool whole = c.toXYZHeld && c.fromXYZHeld;
	for (uint32_t ch = 0; ch < 3; ch++)
		whole = whole && c.sourceHeld[ch] && c.blendHeld[ch];
	if (!whole || rgfxICCCursorIsIdentity(c))
		return true;
	message->on = 1;
	for (uint32_t ch = 0; ch < 3; ch++) {
		message->source_curves[ch] = c.source[ch];
		message->blend_curves[ch] = c.blend[ch];
	}
	for (uint32_t i = 0; i < 9; i++) {
		message->to_xyz[i] = c.toXYZ[i];
		message->from_xyz[i] = c.fromXYZ[i];
	}
	return true;
}

// Records that the CURSOR_COLOR of `changes` changes went out to the host.
inline void rgfxICCCursorPosted(RgfxICCState *state, uint32_t changes)
{
	__atomic_store_n(&state->cursor.posted, changes, __ATOMIC_RELAXED);
}

// Counts a change to the cursor's chain.
inline void rgfxICCCursorChanged(RgfxICCCursor *c)
{
	__atomic_store_n(&c->changes, c->changes + 1, __ATOMIC_RELAXED);
}

// A pair's second matrix: taken where it inverts the first held, or nothing is held.
inline bool rgfxICCSecondMatrix(const RgfxICCPair &pair, const RgfxICCMatrix &second)
{
	if (pair.matrixHeld == kRgfxICCHeldNone)
		return true;
	return pair.matrixHeld == kRgfxICCHeldRead &&
	       rgfxICCMatrixIsIdentity(rgfxICCMatrixMul(second, pair.matrix));
}

// A pair's second curve, for the channels of `mask`: taken where it inverts each first held, or
// nothing is held. A first curve the channels share is checked once.
inline bool rgfxICCSecondCurve(const RgfxICCPair &pair, uint32_t mask, const RgfxICCCurve &second)
{
	const RgfxICCCurve *checked = nullptr;
	for (uint32_t ch = 0; ch < 3; ch++) {
		if (!(mask & (1u << ch)) || pair.curveHeld[ch] == kRgfxICCHeldNone)
			continue;
		if (pair.curveHeld[ch] != kRgfxICCHeldRead)
			return false;
		const RgfxICCCurve &first = pair.curve[ch];
		bool same = false;
		if (checked) {
			same = checked->type == first.type;
			for (uint32_t i = 0; same && i < 7; i++)
				same = checked->p[i] == first.p[i];
		}
		if (!same && !rgfxICCCurvesAreIdentity(first, &second))
			return false;
		checked = &first;
	}
	return true;
}

// The pair whose first block `type` and `scalars` name, read or not, and that block's channels
// (7 for a matrix); null for any other block.
inline RgfxICCPair *rgfxICCFirstOf(RgfxICCState *state, uint32_t type, const uint64_t *scalars,
                                   uint32_t count, uint32_t *channels)
{
	if (!scalars)
		return nullptr;
	uint32_t curveLocation = 0;
	if (type == state->release.matrixBlock && count >= 3 &&
	    rgfxICCMatrixLocation(state->release, scalars[1])) {
		*channels = 7;
		if (scalars[1] == kRgfxICCMatrixSrc2XYZ && scalars[2] < kRgfxICCLayers)
			return &state->layer[scalars[2]];
		if (scalars[1] == kRgfxICCMatrixBlend2XYZ)
			return &state->display;
	} else if (type == state->release.curveBlock && count >= 4 && scalars[2] >= 1 &&
	           scalars[2] <= 7 && rgfxICCCurveLocation(state->release, scalars[1], &curveLocation)) {
		*channels = static_cast<uint32_t>(scalars[2]);
		if (curveLocation == kRgfxICCCurveSrc2XYZ && scalars[3] < kRgfxICCLayers)
			return &state->layer[scalars[3]];
		if (curveLocation == kRgfxICCCurveBlend2XYZ)
			return &state->display;
	}
	return nullptr;
}

// Whether the display takes the block set_block hands it (the rules above), holding a pair's first
// block in `state`, or that it refused one.
inline bool rgfxICCTake(RgfxICCState *state, uint32_t type, uint32_t offset,
                        const uint64_t *scalars, uint32_t count, const uint8_t *data, size_t size)
{
	if (!state->release.matrixBlock || !state->release.curveBlock)
		return false;
	uint32_t channels = 0;
	RgfxICCPair *first = rgfxICCFirstOf(state, type, scalars, count, &channels);
	uint32_t location = 0, mask = 7, layer = 0;
	RgfxICCMatrix m{};
	RgfxICCCurve c{};
	const bool isMatrix = type == state->release.matrixBlock;
	const bool read =
	    offset == 0 &&
	    (isMatrix ? rgfxICCParseMatrix(state->release, scalars, count, data, size, &location,
	                                   &layer, &m)
	              : type == state->release.curveBlock &&
	                    rgfxICCParseCurve(state->release, scalars, count, data, size, &location,
	                                      &mask, &layer, &c)) &&
	    layer < kRgfxICCLayers;
	if (first) {
		// Held, read or refused: its pair's second block is judged against it.
		if (isMatrix) {
			first->matrixHeld = read ? kRgfxICCHeldRead : kRgfxICCHeldRefused;
			if (read)
				first->matrix = m;
		} else {
			for (uint32_t ch = 0; ch < 3; ch++)
				if (channels & (1u << ch)) {
					first->curveHeld[ch] = read ? kRgfxICCHeldRead : kRgfxICCHeldRefused;
					if (read)
						first->curve[ch] = c;
				}
		}
		// The cursor's source, kept for the host where it converts the cursor; one the display
		// cannot read leaves its stage out of the chain.
		RgfxICCCursor &cursor = state->cursor;
		if (cursor.host && first == &state->layer[kRgfxICCCursorLayer]) {
			if (isMatrix) {
				cursor.toXYZHeld = read;
				if (read)
					rgfxICCCursorMatrix(cursor.toXYZ, &cursor.toXYZQ, data, m);
			} else if (read) {
				rgfxICCCursorCurve(cursor.source, cursor.sourceQ, cursor.sourceHeld, mask, data, c);
			} else {
				for (uint32_t ch = 0; ch < 3; ch++)
					if (channels & (1u << ch))
						cursor.sourceHeld[ch] = false;
			}
			rgfxICCCursorChanged(&cursor);
		}
		return read;
	}
	if (!read)
		return false;
	// The cursor's blend space, kept for the host where it converts the cursor: taken, whatever it
	// holds.
	RgfxICCCursor &cursor = state->cursor;
	if (cursor.host && layer == kRgfxICCCursorLayer) {
		if (isMatrix && location == kRgfxICCMatrixXYZ2Blend) {
			cursor.fromXYZHeld = true;
			rgfxICCCursorMatrix(cursor.fromXYZ, &cursor.fromXYZQ, data, m);
			rgfxICCCursorChanged(&cursor);
			return true;
		}
		if (!isMatrix && location == kRgfxICCCurveXYZ2Blend) {
			rgfxICCCursorCurve(cursor.blend, cursor.blendQ, cursor.blendHeld, mask, data, c);
			rgfxICCCursorChanged(&cursor);
			return true;
		}
	}
	if (isMatrix) {
		switch (location) {
		case kRgfxICCMatrixXYZ2Blend:
			return rgfxICCSecondMatrix(state->layer[layer], m);
		case kRgfxICCMatrixXYZ2Device:
			return rgfxICCSecondMatrix(state->display, m);
		default:
			return rgfxICCMatrixIsIdentity(m);
		}
	}
	switch (location) {
	case kRgfxICCCurveXYZ2Blend:
		return rgfxICCSecondCurve(state->layer[layer], mask, c);
	case kRgfxICCCurveXYZ2Device:
		return rgfxICCSecondCurve(state->display, mask, c);
	case kRgfxICCCurveUserGammaAdjustment:
		return false;
	default:
		return rgfxICCCurvesAreIdentity(c, nullptr);
	}
}

#endif // RGFX_MOBILE_ICC_HPP_
