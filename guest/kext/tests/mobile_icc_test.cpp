// The native test of RgfxMobileICC.hpp: the fixed-point math the display judges WindowServer's ICC
// blocks with (ReimsGraphicsMobile.cpp's set_block), against the same math in floating point here;
// the blocks' parse, on blocks laid out as QuartzCore lays them out; which blocks the display
// takes, over the bursts a 27.0.1 login and its profile switches send, in WindowServer's order; and
// the override's source.
#include "RgfxMobileICC.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>

namespace {

int checks = 0;

void check(bool ok, const char *what, const char *about)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "mobile_icc_test: FAILED (%s): %s\n", about, what);
		std::exit(1);
	}
}

double real(int64_t q)
{
	return static_cast<double>(q) * 0x1p-32;
}

int64_t q32(double v)
{
	return static_cast<int64_t>(std::llround(v * 0x1p32));
}

uint32_t bitsOf(float f)
{
	uint32_t bits = 0;
	std::memcpy(&bits, &f, sizeof(bits));
	return bits;
}

// A pseudo-random sequence (xorshift).
uint64_t seed = 0x9e3779b97f4a7c15ull;
uint64_t next()
{
	seed ^= seed << 13;
	seed ^= seed >> 7;
	seed ^= seed << 17;
	return seed;
}

// A float converts to Q32 as rounding to nearest does; NaNs, infinities and magnitudes of 2^31 or
// more are refused.
void conversion()
{
	for (float f : {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 2.4f, 0.416667f, 12.92f, 0.0031308f, -0.055f,
	                1e-12f, 1e-30f, 1073741824.0f, -1073741824.0f, 2147483520.0f}) {
		int64_t v = 0;
		check(rgfxFloatBitsToQ32(bitsOf(f), &v), "a finite float below 2^31 converts", "edge");
		check(std::fabs(static_cast<double>(v) - static_cast<double>(f) * 0x1p32) <= 0.5,
		      "it rounds to nearest", "edge");
	}
	for (float f : {2147483648.0f, -2147483648.0f, 1e20f, INFINITY, -INFINITY, NAN}) {
		int64_t v = 0;
		check(!rgfxFloatBitsToQ32(bitsOf(f), &v), "a float of 2^31 or more is refused", "edge");
	}
	for (int i = 0; i < 1000000; i++) {
		const uint32_t bits = static_cast<uint32_t>(next());
		float f = 0;
		std::memcpy(&f, &bits, sizeof(f));
		int64_t v = 0;
		const bool ok = rgfxFloatBitsToQ32(bits, &v);
		check(ok == (std::isfinite(f) && std::fabs(f) < 0x1p31f), "refusal matches the range",
		      "spread");
		if (ok)
			check(std::fabs(static_cast<double>(v) - static_cast<double>(f) * 0x1p32) <= 0.5,
			      "it rounds to nearest", "spread");
	}
}

// log2, exp2 and powers within 2^-29 of the floating-point answers, and the table is 2^(2^-k).
void math()
{
	for (int k = 1; k <= 32; k++)
		check(std::fabs(static_cast<double>(kRgfxExp2Steps[k - 1]) * 0x1p-62 -
		                std::exp2(std::ldexp(1.0, -k))) <= 0x1p-52,
		      "the table is 2^(2^-k)", "table");
	for (int i = 0; i < 500000; i++) {
		const int64_t x = static_cast<int64_t>(next() >> (1 + next() % 63));
		if (x <= 0)
			continue;
		check(std::fabs(real(rgfxQ32Log2(x)) - std::log2(real(x))) <= 0x1p-30, "log2", "log2");
		const int64_t y = static_cast<int64_t>(next() % (uint64_t(60) << 32)) - (int64_t(32) << 32);
		if (y < int64_t(30) << 32) {
			const double want = std::exp2(real(y));
			const double got = real(rgfxQ32Exp2(y));
			check(std::fabs(got - want) <= std::fmax(want * 0x1p-40, 0x1p-32), "exp2", "exp2");
		}
		const int64_t base = static_cast<int64_t>(next() % ((uint64_t(1) << 32) + 1));
		const int64_t g = q32(0.1 + static_cast<double>(next() % 4000) / 1000.0);
		check(std::fabs(real(rgfxQ32Pow(base, g)) - std::pow(real(base), real(g))) <= 0x1p-29,
		      "x^g on [0, 1]", "pow");
	}
	check(rgfxQ32Pow(0, kRgfxQ32One) == 0 && rgfxQ32Pow(-5, kRgfxQ32One) == 0, "0^g is 0", "pow");
	check(rgfxQ32Pow(kRgfxQ32One, q32(2.4)) == kRgfxQ32One, "1^g is 1", "pow");
}

// QuartzCore's forms, in floating point.
double curveAt(uint32_t type, const double p[7], double x)
{
	const double g = p[0], a = p[1], b = p[2], c = p[3], d = p[4], e = p[5], f = p[6];
	switch (type) {
	case kRgfxICCCurveType0:
		return std::pow(x, g);
	case kRgfxICCCurveType1:
		return a * x + b >= 0 ? std::pow(a * x + b, g) : 0;
	case kRgfxICCCurveType2:
		return a * x + b >= 0 ? std::pow(a * x + b, g) + c : c;
	case kRgfxICCCurveType3:
		return x >= d ? std::pow(a * x + b, g) : c * x;
	case kRgfxICCCurveType4:
		return x >= d ? std::pow(a * x + b, g) + e : c * x + f;
	default:
		return x;
	}
}

void put32(uint8_t *data, size_t at, uint32_t v)
{
	for (int i = 0; i < 4; i++)
		data[at + i] = static_cast<uint8_t>(v >> (8 * i));
}

void putFloat(uint8_t *data, size_t at, float v)
{
	put32(data, at, bitsOf(v));
}

// The largest curve block measured, 0x290 bytes (27.0.1's and 26.6.2's).
constexpr size_t kCurveBytes = 0x290;

// A release's numbering, as rgfxICCReleaseOf gives it.
RgfxICCRelease releaseOf(int major)
{
	RgfxICCRelease release{};
	check(rgfxICCReleaseOf(major, &release), "the release's numbering is known", "releaseOf");
	return release;
}

// 27.0.1's numbering, the one the parse tests read blocks in.
const RgfxICCRelease k27 = releaseOf(27);

// A curve block as QuartzCore's IOMobileFramebufferSwapSetICCCurve lays it out.
void curveBlock(uint8_t data[kCurveBytes], uint32_t type, const float p[7])
{
	std::memset(data, 0, kCurveBytes);
	put32(data, 0x00, 1);
	put32(data, 0x08, type);
	for (int i = 0; i < 7; i++)
		putFloat(data, 0x0c + 4 * i, p[i]);
}

// A matrix block as IOMobileFramebufferSwapSetICCMatrix lays it out.
void matrixBlock(uint8_t data[kRgfxICCMatrixBlockSize], const float m[9])
{
	std::memset(data, 0, kRgfxICCMatrixBlockSize);
	put32(data, 0x00, 1);
	for (int i = 0; i < 9; i++)
		putFloat(data, 0x10 + 4 * i, m[i]);
}

// The parametric curves evaluate as QuartzCore's forms do, within 2^-28 over [0, 1].
void curves()
{
	// The sRGB decode and encode WindowServer sets (27.0.1), Generic RGB's 1.8, and a Type1 and a
	// Type2 of every part.
	const struct {
		uint32_t type;
		float p[7];
	} cases[] = {
	    {kRgfxICCCurveType3, {2.4f, 0.947867f, 0.0521327f, 0.0773994f, 0.04045f, 0, 0}},
	    {kRgfxICCCurveType4, {0.416667f, 1.13712f, 0, 12.92f, 0.0031308f, -0.055f, 0}},
	    {kRgfxICCCurveType0, {1.8f, 0, 0, 0, 0, 0, 0}},
	    {kRgfxICCCurveType0, {1.0f / 1.8f, 0, 0, 0, 0, 0, 0}},
	    {kRgfxICCCurveType1, {2.2f, 1.2f, -0.1f, 0, 0, 0, 0}},
	    {kRgfxICCCurveType2, {1.5f, 0.715542f, 0.05f, 0.1f, 0, 0, 0}},
	};
	const uint64_t scalars[4] = {0x4b, kRgfxICCCurveXYZ2Device, 7, 0};
	for (const auto &c : cases) {
		uint8_t data[kCurveBytes];
		curveBlock(data, c.type, c.p);
		uint32_t location = 99, mask = 99, layer = 99;
		RgfxICCCurve curve{};
		check(rgfxICCParseCurve(k27, scalars, 4, data, sizeof(data), &location, &mask, &layer, &curve),
		      "a parametric curve parses", "curve");
		check(location == kRgfxICCCurveXYZ2Device && mask == 7 && layer == 0,
		      "its location, mask and layer are scalars 1 to 3", "curve");
		double p[7];
		for (int i = 0; i < 7; i++)
			p[i] = static_cast<double>(c.p[i]);
		double worst = 0;
		for (int k = 0; k <= 4096; k++) {
			const int64_t x = (kRgfxQ32One * k) / 4096;
			worst = std::fmax(worst, std::fabs(real(rgfxICCCurveAt(curve, x)) -
			                                   curveAt(c.type, p, real(x))));
		}
		check(worst <= 0x1p-28, "the curve evaluates as QuartzCore's form", "curve");
	}

	// Bypass passes its input through, whatever its parameters.
	{
		const float p[7] = {1, 0, 0, 0, 0, 0, 0};
		uint8_t data[kCurveBytes];
		curveBlock(data, kRgfxICCCurveBypass, p);
		uint32_t location, mask, layer;
		RgfxICCCurve curve{};
		check(rgfxICCParseCurve(k27, scalars, 4, data, sizeof(data), &location, &mask, &layer, &curve),
		      "Bypass parses", "bypass");
		for (int64_t x : {int64_t(0), kRgfxQ32One / 3, kRgfxQ32One})
			check(rgfxICCCurveAt(curve, x) == x, "Bypass is the identity", "bypass");
	}

	// What the display does not take.
	{
		const float p[7] = {2.4f, 0.947867f, 0.0521327f, 0.0773994f, 0.04045f, 0, 0};
		uint8_t data[kCurveBytes];
		uint32_t location, mask, layer;
		RgfxICCCurve curve{};
		curveBlock(data, kRgfxICCCurveType3, p);
		check(!rgfxICCParseCurve(k27, scalars, 3, data, sizeof(data), &location, &mask, &layer, &curve),
		      "fewer than 4 scalars", "refused");
		check(!rgfxICCParseCurve(k27, scalars, 4, data, sizeof(data) - 4, &location, &mask, &layer,
		                         &curve),
		      "another size", "refused");
		for (uint32_t type : {6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 17u, 18u, 0xffffffffu}) {
			curveBlock(data, type, p);
			check(!rgfxICCParseCurve(k27, scalars, 4, data, sizeof(data), &location, &mask, &layer,
			                         &curve),
			      "a type the display does not evaluate", "refused");
		}
		curveBlock(data, kRgfxICCCurveType3, p);
		put32(data, 0x00, 2);
		check(!rgfxICCParseCurve(k27, scalars, 4, data, sizeof(data), &location, &mask, &layer, &curve),
		      "field_00 other than 1", "refused");
		curveBlock(data, kRgfxICCCurveType3, p);
		put32(data, 0x04, 1);
		check(!rgfxICCParseCurve(k27, scalars, 4, data, sizeof(data), &location, &mask, &layer, &curve),
		      "a nonzero field_04", "refused");
		curveBlock(data, kRgfxICCCurveType3, p);
		putFloat(data, 0x0c, NAN);
		check(!rgfxICCParseCurve(k27, scalars, 4, data, sizeof(data), &location, &mask, &layer, &curve),
		      "a parameter that is not finite", "refused");
		curveBlock(data, kRgfxICCCurveType3, p);
		for (uint64_t m : {uint64_t(0), uint64_t(8), uint64_t(15)}) {
			const uint64_t bad[4] = {1, kRgfxICCCurveXYZ2Device, m, 0};
			check(!rgfxICCParseCurve(k27, bad, 4, data, sizeof(data), &location, &mask, &layer,
			                         &curve),
			      "a mask of no channel or another bit", "refused");
		}
		const uint64_t far[4] = {1, kRgfxICCCurveLocations, 7, 0};
		check(!rgfxICCParseCurve(k27, far, 4, data, sizeof(data), &location, &mask, &layer, &curve),
		      "an unknown location", "refused");
		const float neg[7] = {2.2f, -1.0f, 1.0f, 0, 0, 0, 0};
		curveBlock(data, kRgfxICCCurveType1, neg);
		check(!rgfxICCParseCurve(k27, scalars, 4, data, sizeof(data), &location, &mask, &layer, &curve),
		      "a Type1 whose a is not positive", "refused");
	}
}

// A matrix block parses to its location, layer and matrix; what the display does not take is
// refused.
void matrices()
{
	const float m[9] = {0.437102f, 0.384124f, 0.142974f, 0.223083f, 0.716339f,
	                    0.0605774f, 0.0139922f, 0.0970453f, 0.713862f};
	uint8_t data[kRgfxICCMatrixBlockSize];
	matrixBlock(data, m);
	const uint64_t scalars[3] = {0x4b, kRgfxICCMatrixBlend2XYZ, 0};
	uint32_t location = 99, layer = 99;
	RgfxICCMatrix matrix{};
	check(rgfxICCParseMatrix(k27, scalars, 3, data, sizeof(data), &location, &layer, &matrix),
	      "a matrix parses", "matrix");
	check(location == kRgfxICCMatrixBlend2XYZ && layer == 0,
	      "its location and layer are scalars 1 and 2", "matrix");
	for (int i = 0; i < 9; i++)
		check(std::fabs(real(matrix.m[i]) - static_cast<double>(m[i])) <= 0x1p-33,
		      "entry i is the float at 0x10 + 4i", "matrix");

	// The product with its inverse is the identity.
	const float inv[9] = {3.12482f, -1.60934f, -0.489281f, -0.979213f, 1.91654f,
	                      0.033485f, 0.0718694f, -0.228997f, 1.40587f};
	RgfxICCMatrix inverse{};
	matrixBlock(data, inv);
	check(rgfxICCParseMatrix(k27, scalars, 3, data, sizeof(data), &location, &layer, &inverse),
	      "the inverse parses", "matrix");
	check(rgfxICCMatrixIsIdentity(rgfxICCMatrixMul(inverse, matrix)),
	      "the product with the inverse is the identity", "product");
	check(!rgfxICCMatrixIsIdentity(matrix), "the matrix alone is not", "product");
	RgfxICCMatrix one{};
	one.m[0] = one.m[4] = one.m[8] = kRgfxQ32One;
	check(rgfxICCMatrixIsIdentity(one), "the identity is", "product");
	const RgfxICCMatrix same = rgfxICCMatrixMul(one, matrix);
	for (int i = 0; i < 9; i++)
		check(same.m[i] == matrix.m[i], "the identity times a matrix is the matrix", "product");

	// Refused: fewer scalars, another size, field_00 other than 1, a nonzero field_04 or field_34,
	// an unknown location, a NaN.
	matrixBlock(data, m);
	check(!rgfxICCParseMatrix(k27, scalars, 2, data, sizeof(data), &location, &layer, &matrix),
	      "fewer than 3 scalars", "refused");
	check(!rgfxICCParseMatrix(k27, scalars, 3, data, sizeof(data) + 4, &location, &layer, &matrix),
	      "another size", "refused");
	put32(data, 0x00, 0);
	check(!rgfxICCParseMatrix(k27, scalars, 3, data, sizeof(data), &location, &layer, &matrix),
	      "field_00 other than 1", "refused");
	for (size_t at : {size_t(0x04), size_t(0x0c), size_t(0x34), size_t(0x3c)}) {
		matrixBlock(data, m);
		putFloat(data, at, 0.5f);
		check(!rgfxICCParseMatrix(k27, scalars, 3, data, sizeof(data), &location, &layer, &matrix),
		      "a nonzero field_04 or field_34", "refused");
	}
	matrixBlock(data, m);
	const uint64_t far[3] = {1, kRgfxICCMatrixLocations, 0};
	check(!rgfxICCParseMatrix(k27, far, 3, data, sizeof(data), &location, &layer, &matrix),
	      "an unknown location", "refused");
	putFloat(data, 0x20, NAN);
	check(!rgfxICCParseMatrix(k27, scalars, 3, data, sizeof(data), &location, &layer, &matrix),
	      "an entry that is not finite", "refused");
}

// ---- What the display takes ----

// A display's colour spaces, as WindowServer sends them (27.0.1): a colour space's primaries to
// XYZ D50, and back, row by row.
const float kDisplayP3[9] = {0.51512f, 0.291976f, 0.157104f, 0.241192f, 0.692235f,
                             0.0665731f, -0.00105285f, 0.0418851f, 0.784068f};
const float kDisplayP3Inverse[9] = {2.40398f, -0.989911f, -0.397637f, -0.842246f, 1.79888f,
                                    0.0160235f, 0.0482211f, -0.0974257f, 1.27401f};
// The display's own: its published primaries and white (RgfxDisplayIdentity.hpp).
const float kPublished[9] = {0.437102f, 0.384124f, 0.142974f, 0.223083f, 0.716339f,
                             0.0605774f, 0.0139922f, 0.0970453f, 0.713862f};
const float kPublishedInverse[9] = {3.12482f, -1.60934f, -0.489281f, -0.979213f, 1.91654f,
                                    0.033485f, 0.0718694f, -0.228997f, 1.40587f};
// Standard sRGB, the cursor's.
const float kSRGB[9] = {0.43603f, 0.385115f, 0.143055f, 0.222495f, 0.716895f,
                        0.0606098f, 0.0139128f, 0.0970542f, 0.713933f};
const float kGenericRGB[9] = {0.454255f, 0.353313f, 0.156631f, 0.241913f, 0.67363f,
                              0.0844574f, 0.0148889f, 0.0906148f, 0.719396f};
const float kGenericRGBInverse[9] = {3.0287f, -1.52389f, -0.480522f, -1.09713f, 2.06033f,
                                     -0.00301049f, 0.0755107f, -0.22798f, 1.40038f};
const float kIdentity[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

// The curves: sRGB's decode and encode, Generic RGB's decode with its toe and its inverse, and
// Generic RGB's pure 1.8 and its inverse.
const float kSRGBDecode[7] = {2.4f, 0.947867f, 0.0521327f, 0.0773994f, 0.04045f, 0, 0};
const float kSRGBEncode[7] = {0.416667f, 1.13712f, 0, 12.92f, 0.0031308f, -0.055f, 0};
const float kGenericDecode[7] = {1.8f, 1, 0, 0.0625f, 0.03125f, 0, 0};
const float kGenericEncode[7] = {0.555556f, 1, 0, 16, 0.00195312f, 0, 0};
const float kGamma18[7] = {1.8f, 1, 0, 0, 0, 0, 0};
const float kGamma18Inverse[7] = {0.555556f, 1, 0, 0, 0, 0, 0};
const float kZeros[7] = {};

RgfxICCState state;
uint32_t swapId = 1;

// A state holding nothing, with the block types of the release whose kernel's major version is
// `major`.
void fresh(int major, const char *about)
{
	state = RgfxICCState{};
	check(rgfxICCReleaseOf(major, &state.release), "the release's numbering is known", about);
}

bool matrix(uint32_t location, uint32_t layer, const float m[9])
{
	uint8_t data[kRgfxICCMatrixBlockSize];
	matrixBlock(data, m);
	const uint64_t scalars[3] = {swapId, location, layer};
	return rgfxICCTake(&state, state.release.matrixBlock, 0, scalars, 3, data, sizeof(data));
}

bool curve(uint32_t location, uint32_t mask, uint32_t layer, uint32_t type, const float p[7])
{
	uint8_t data[kCurveBytes];
	curveBlock(data, type, p);
	const uint64_t scalars[4] = {swapId, location, mask, layer};
	return rgfxICCTake(&state, state.release.curveBlock, 0, scalars, 4, data,
	                   state.release.curveBlockSize);
}

// The display's half of a burst, as WindowServer orders it: Blend2XYZ's curve and matrix, then
// XYZ2Device's matrix and curve. Each answer is checked against `secondTaken` for the second two.
void displayHalf(const float *toXYZ, const float *fromXYZ, uint32_t decodeType,
                 const float *decode, uint32_t encodeType, const float *encode, bool secondTaken,
                 const char *about)
{
	check(curve(kRgfxICCCurveBlend2XYZ, 7, 0, decodeType, decode), "Blend2XYZ's curve is held",
	      about);
	check(matrix(kRgfxICCMatrixBlend2XYZ, 0, toXYZ), "Blend2XYZ's matrix is held", about);
	check(matrix(kRgfxICCMatrixXYZ2Device, 0, fromXYZ) == secondTaken,
	      "XYZ2Device's matrix is judged against it", about);
	check(curve(kRgfxICCCurveXYZ2Device, 7, 0, encodeType, encode) == secondTaken,
	      "XYZ2Device's curve is judged against it", about);
}

// A layer's half: Src2XYZ's curve and matrix where `layer` has a source, then XYZ2Blend's
// matrices and curves for layers 0 to 2.
void layerHalf(int layer, const float *toXYZ, uint32_t decodeType, const float *decode,
               const float *fromXYZ, uint32_t encodeType, const float *encode,
               const bool matrixTaken[3], const bool curveTaken[3], const char *about)
{
	if (layer >= 0) {
		check(curve(kRgfxICCCurveSrc2XYZ, 7, layer, decodeType, decode),
		      "Src2XYZ's curve is held", about);
		check(matrix(kRgfxICCMatrixSrc2XYZ, layer, toXYZ), "Src2XYZ's matrix is held", about);
	}
	for (uint32_t l = 0; l < 3; l++)
		check(matrix(kRgfxICCMatrixXYZ2Blend, l, fromXYZ) == matrixTaken[l],
		      "XYZ2Blend's matrix is judged against its layer's", about);
	for (uint32_t l = 0; l < 3; l++)
		check(curve(kRgfxICCCurveXYZ2Blend, 7, l, encodeType, encode) == curveTaken[l],
		      "XYZ2Blend's curve is judged against its layer's", about);
}

// The bursts of a 27.0.1 login and of switching the profile, replayed in WindowServer's order.
void bursts()
{
	fresh(27, "27.0.1");
	const bool all[3] = {true, true, true};
	// At WindowServer's start: Display P3, then the tone map's defaults, then layer 0's half.
	// Layers 1 and 2 have no source yet: their second blocks are taken.
	swapId = 1;
	displayHalf(kDisplayP3, kDisplayP3Inverse, kRgfxICCCurveType3, kSRGBDecode,
	            kRgfxICCCurveType4, kSRGBEncode, true, "start");
	check(curve(kRgfxICCCurveSystemGamma, 7, 0, kRgfxICCCurveBypass, kZeros),
	      "SystemGamma's Bypass is taken", "start");
	for (uint32_t layer = 0; layer < 3; layer++) {
		for (uint32_t loc : {3u, 7u, 6u, 11u})
			check(curve(loc, 7, layer, kRgfxICCCurveBypass, kZeros),
			      "a tone map's Bypass is taken", "start");
		check(matrix(kRgfxICCMatrixPreToneMap, layer, kIdentity) &&
		          matrix(kRgfxICCMatrixPostToneMap, layer, kIdentity),
		      "the tone map's identity matrices are taken", "start");
	}
	layerHalf(0, kDisplayP3, kRgfxICCCurveType3, kSRGBDecode, kDisplayP3Inverse,
	          kRgfxICCCurveType4, kSRGBEncode, all, all, "start");

	// The display's own colour space: the halves change one block at a time, and each is taken.
	swapId = 0x4b;
	displayHalf(kPublished, kPublishedInverse, kRgfxICCCurveType3, kSRGBDecode,
	            kRgfxICCCurveType4, kSRGBEncode, true, "own");
	const float twoTwo[7] = {2.2f, 1, 0, 0, 0, 0, 0};
	for (uint32_t mask : {1u, 2u, 4u})
		check(!curve(kRgfxICCCurveUserGammaAdjustment, mask, 0, kRgfxICCCurveType2, twoTwo),
		      "user gamma is refused", "own");
	layerHalf(0, kPublished, kRgfxICCCurveType3, kSRGBDecode, kPublishedInverse,
	          kRgfxICCCurveType4, kSRGBEncode, all, all, "own");

	// The cursor's first image: layer 1's source is standard sRGB, whose matrix the display's
	// does not invert (0.0024 off), so layer 1's XYZ2Blend matrix is refused; the curves invert.
	swapId = 0x8c;
	const bool cursorMatrix[3] = {true, false, true};
	layerHalf(1, kSRGB, kRgfxICCCurveType3, kSRGBDecode, kPublishedInverse, kRgfxICCCurveType4,
	          kSRGBEncode, cursorMatrix, all, "cursor");
	const float blackpoint[7] = {1, 1, 0, 0, 0, 0, 0};
	check(curve(kRgfxICCCurveBlackpoint, 7, 0, kRgfxICCCurveType3, blackpoint),
	      "the identity black point is taken", "cursor");

	// Generic RGB: both halves are Generic RGB's own, with the toe in layer 0's and none in the
	// display's; layer 1 keeps standard sRGB's source, so its blocks are refused.
	swapId = 0xc5;
	displayHalf(kGenericRGB, kGenericRGBInverse, kRgfxICCCurveType3, kGamma18,
	            kRgfxICCCurveType3, kGamma18Inverse, true, "Generic RGB");
	const bool generic[3] = {true, false, true};
	layerHalf(0, kGenericRGB, kRgfxICCCurveType3, kGenericDecode, kGenericRGBInverse,
	          kRgfxICCCurveType3, kGenericEncode, generic, generic, "Generic RGB");

	// Display P3, then the display's own profile again.
	swapId = 0xc8;
	displayHalf(kDisplayP3, kDisplayP3Inverse, kRgfxICCCurveType3, kSRGBDecode,
	            kRgfxICCCurveType4, kSRGBEncode, true, "P3");
	const bool p3[3] = {true, false, true};
	layerHalf(0, kDisplayP3, kRgfxICCCurveType3, kSRGBDecode, kDisplayP3Inverse,
	          kRgfxICCCurveType4, kSRGBEncode, p3, all, "P3");
	swapId = 0xca;
	displayHalf(kPublished, kPublishedInverse, kRgfxICCCurveType3, kSRGBDecode,
	            kRgfxICCCurveType4, kSRGBEncode, true, "reset");
	layerHalf(0, kPublished, kRgfxICCCurveType3, kSRGBDecode, kPublishedInverse,
	          kRgfxICCCurveType4, kSRGBEncode, cursorMatrix, all, "reset");

	// The halves across each other are not inverses with Generic RGB: the display's curve (no toe)
	// after layer 0's (with the toe) is 6e-4 off near black, so pairing by curve type would refuse
	// what the halves take.
	RgfxICCCurve toe{}, pure{};
	uint8_t data[kCurveBytes];
	uint32_t location, mask, layer;
	const uint64_t scalars[4] = {1, kRgfxICCCurveXYZ2Device, 7, 0};
	curveBlock(data, kRgfxICCCurveType3, kGenericEncode);
	check(rgfxICCParseCurve(k27, scalars, 4, data, sizeof(data), &location, &mask, &layer, &toe), "toe",
	      "halves");
	curveBlock(data, kRgfxICCCurveType3, kGamma18);
	check(rgfxICCParseCurve(k27, scalars, 4, data, sizeof(data), &location, &mask, &layer, &pure),
	      "pure", "halves");
	check(!rgfxICCCurvesAreIdentity(toe, &pure), "the halves' curves across are not inverses",
	      "halves");
}

// A second block that does not invert what is held is refused, and so is every second block
// after a first the display cannot read; the next readable first is held again.
void pairs()
{
	fresh(27, "pairs");
	check(matrix(kRgfxICCMatrixBlend2XYZ, 0, kPublished), "a first matrix is held", "pairs");
	check(!matrix(kRgfxICCMatrixXYZ2Device, 0, kIdentity), "the identity does not invert it",
	      "pairs");
	check(!matrix(kRgfxICCMatrixXYZ2Device, 0, kDisplayP3Inverse),
	      "another colour space's inverse does not", "pairs");
	check(matrix(kRgfxICCMatrixXYZ2Device, 0, kPublishedInverse), "its own inverse does",
	      "pairs");
	check(curve(kRgfxICCCurveBlend2XYZ, 7, 0, kRgfxICCCurveType3, kSRGBDecode),
	      "a first curve is held", "pairs");
	check(!curve(kRgfxICCCurveXYZ2Device, 7, 0, kRgfxICCCurveType3, kGamma18Inverse),
	      "1/1.8 does not invert sRGB's decode", "pairs");
	check(!curve(kRgfxICCCurveXYZ2Device, 7, 0, kRgfxICCCurveBypass, kZeros),
	      "Bypass does not", "pairs");

	// Channel by channel: a red first curve of 1.8 and sRGB's on green and blue.
	check(curve(kRgfxICCCurveBlend2XYZ, 1, 0, kRgfxICCCurveType3, kGamma18), "red's is held",
	      "channels");
	check(curve(kRgfxICCCurveXYZ2Device, 6, 0, kRgfxICCCurveType4, kSRGBEncode),
	      "green's and blue's are judged alone", "channels");
	check(!curve(kRgfxICCCurveXYZ2Device, 7, 0, kRgfxICCCurveType4, kSRGBEncode),
	      "all three are judged channel by channel", "channels");
	check(curve(kRgfxICCCurveXYZ2Device, 1, 0, kRgfxICCCurveType3, kGamma18Inverse),
	      "red's own inverse is taken", "channels");

	// A first block the display cannot read: a PQ curve, a chunk, a NaN entry.
	check(!curve(kRgfxICCCurveSrc2XYZ, 7, 0, 7, kZeros), "a PQ first curve is refused",
	      "unread");
	check(!curve(kRgfxICCCurveXYZ2Blend, 7, 0, kRgfxICCCurveType4, kSRGBEncode),
	      "its second is refused", "unread");
	check(curve(kRgfxICCCurveSrc2XYZ, 7, 0, kRgfxICCCurveType3, kSRGBDecode),
	      "a readable first is held again", "unread");
	check(curve(kRgfxICCCurveXYZ2Blend, 7, 0, kRgfxICCCurveType4, kSRGBEncode),
	      "and its second is taken", "unread");
	{
		uint8_t data[kRgfxICCMatrixBlockSize];
		matrixBlock(data, kPublished);
		const uint64_t scalars[3] = {1, kRgfxICCMatrixSrc2XYZ, 0};
		check(!rgfxICCTake(&state, state.release.matrixBlock, 0x10, scalars, 3, data, sizeof(data)),
		      "a chunked first matrix is refused", "unread");
		check(!matrix(kRgfxICCMatrixXYZ2Blend, 0, kPublishedInverse), "its second is refused",
		      "unread");
		float nan[9];
		std::memcpy(nan, kPublished, sizeof(nan));
		nan[4] = NAN;
		check(!matrix(kRgfxICCMatrixSrc2XYZ, 0, nan), "a NaN first matrix is refused", "unread");
		check(!matrix(kRgfxICCMatrixXYZ2Blend, 0, kPublishedInverse), "its second is refused",
		      "unread");
		check(matrix(kRgfxICCMatrixSrc2XYZ, 0, kPublished) &&
		          matrix(kRgfxICCMatrixXYZ2Blend, 0, kPublishedInverse),
		      "a readable first and its inverse are taken again", "unread");
	}

	// Each layer's pair is its own.
	check(matrix(kRgfxICCMatrixSrc2XYZ, 3, kSRGB), "layer 3's first is held", "layers");
	check(!matrix(kRgfxICCMatrixXYZ2Blend, 3, kPublishedInverse), "layer 3's second is judged",
	      "layers");
	check(matrix(kRgfxICCMatrixXYZ2Blend, 0, kPublishedInverse),
	      "layer 0's is judged against layer 0's", "layers");
	check(!matrix(kRgfxICCMatrixSrc2XYZ, kRgfxICCLayers, kSRGB) &&
	          !matrix(kRgfxICCMatrixXYZ2Blend, kRgfxICCLayers, kPublishedInverse),
	      "a layer past kRgfxICCLayers is refused", "layers");
}

// The other locations: taken where the block is the identity by itself; user gamma never; other
// block types never.
void others()
{
	fresh(27, "others");
	const float scaled[9] = {1, 0, 0, 0, 1, 0, 0, 0, 0.5f};
	const uint32_t matrixLocations[] = {kRgfxICCMatrixPreToneMap, kRgfxICCMatrixPostToneMap,
	                                    kRgfxICCMatrixHarmony, kRgfxICCMatrixUserAdjust};
	for (uint32_t loc : matrixLocations) {
		check(matrix(loc, 0, kIdentity), "an identity matrix is taken", "others");
		check(!matrix(loc, 0, scaled), "another matrix is refused", "others");
	}
	const float blackpoint[7] = {1, 1, 0, 0, 0, 0, 0};
	const float lifted[7] = {1, 0.9f, 0.1f, 0, 0, 0, 0};
	const uint32_t locations[] = {2, 11, kRgfxICCCurveDstMid, kRgfxICCCurveSystemGamma,
	                              kRgfxICCCurveBlackpoint, kRgfxICCCurveDICOM};
	for (uint32_t loc : locations) {
		check(curve(loc, 7, 0, kRgfxICCCurveBypass, kZeros), "Bypass is taken", "others");
		check(curve(loc, 7, 0, kRgfxICCCurveType3, blackpoint), "an identity curve is taken",
		      "others");
		check(!curve(loc, 7, 0, kRgfxICCCurveType3, lifted), "another curve is refused",
		      "others");
		check(!curve(loc, 7, 0, kRgfxICCCurveType3, kSRGBDecode), "a decode is refused",
		      "others");
	}
	const float one[7] = {1, 1, 0, 0, 0, 0, 0};
	check(!curve(kRgfxICCCurveUserGammaAdjustment, 7, 0, kRgfxICCCurveType2, one),
	      "user gamma is refused, even the identity", "others");
	check(!curve(kRgfxICCCurveUserGammaAdjustment, 7, 0, kRgfxICCCurveBypass, kZeros),
	      "user gamma's Bypass is refused", "others");
	uint8_t data[0x4c] = {};
	const uint64_t layer = 0;
	check(!rgfxICCTake(&state, 0x6c, 0, &layer, 1, data, sizeof(data)),
	      "another block type is refused", "others");
	check(!rgfxICCTake(&state, state.release.curveBlock, 0, nullptr, 0, nullptr, 0),
	      "a block with nothing is refused", "others");
}

// The burst of a 15.6.1 or 26.6.2 login, replayed in WindowServer's order under the release's
// numbering (measured, jobs m3-max-36411 and m3-max-36238, the same blocks in the same order): the
// display's half in Display P3, the tone map's defaults for layers 1, 2 and 0, each after a block
// 0x6b, layer 0's half, and the cursor's half at its first image, about 57 s later. The answers are
// 27.0.1's: only the cursor's matrix is refused.
void burstsOf(int major, const char *about)
{
	fresh(major, about);
	const bool all[3] = {true, true, true};
	swapId = 1;
	displayHalf(kDisplayP3, kDisplayP3Inverse, kRgfxICCCurveType3, kSRGBDecode,
	            kRgfxICCCurveType4, kSRGBEncode, true, about);
	check(curve(kRgfxICCCurveSystemGamma, 7, 0, kRgfxICCCurveBypass, kZeros),
	      "SystemGamma's Bypass is taken", about);
	uint8_t data[0x4c] = {};
	for (uint32_t layer : {1u, 2u, 0u}) {
		const uint64_t which = layer;
		check(!rgfxICCTake(&state, 0x6b, 0, &which, 1, data, sizeof(data)),
		      "the block 0x6b is refused", about);
		for (uint32_t loc : {3u, 7u, 6u, 11u})
			check(curve(loc, 7, layer, kRgfxICCCurveBypass, kZeros),
			      "a tone map's Bypass is taken", about);
		check(matrix(kRgfxICCMatrixPreToneMap, layer, kIdentity) &&
		          matrix(kRgfxICCMatrixPostToneMap, layer, kIdentity),
		      "the tone map's identity matrices are taken", about);
	}
	layerHalf(0, kDisplayP3, kRgfxICCCurveType3, kSRGBDecode, kDisplayP3Inverse,
	          kRgfxICCCurveType4, kSRGBEncode, all, all, about);
	swapId = 0x169;
	const bool cursorMatrix[3] = {true, false, true};
	layerHalf(1, kSRGB, kRgfxICCCurveType3, kSRGBDecode, kDisplayP3Inverse, kRgfxICCCurveType4,
	          kSRGBEncode, cursorMatrix, all, "cursor");
}

// The burst of a 14.6.1 login, replayed in WindowServer's order under 14.6.1's numbering
// (measured, job m3-max-36326): the tone map's defaults for layers 1, 2 and 0, each after a 0x48-byte
// block 0x6b; layer 0's half and the display's in Display P3; then the display's own colour space,
// standard sRGB, whose layers' XYZ2Blend comes a swap before layer 0's Src2XYZ (so layer 0's is
// refused, as the profile switch's on 27.0.1 and 27.2.1 is); user gamma; and the identity black
// point. The curve blocks are 0x248 bytes, and the stages from Blend2XYZ on sit one lower.
void bursts14()
{
	fresh(23, "14.6.1");
	const bool all[3] = {true, true, true};
	swapId = 1;
	uint8_t data[0x48] = {};
	for (uint32_t layer : {1u, 2u, 0u}) {
		const uint64_t which = layer;
		check(!rgfxICCTake(&state, 0x6b, 0, &which, 1, data, sizeof(data)),
		      "the 0x48-byte block is refused", "14.6.1");
		for (uint32_t loc : {3u, 7u, 6u})
			check(curve(loc, 7, layer, kRgfxICCCurveBypass, kZeros),
			      "a tone map's Bypass is taken", "14.6.1");
		check(matrix(kRgfxICCMatrixPreToneMap, layer, kIdentity) &&
		          matrix(kRgfxICCMatrixPostToneMap, layer, kIdentity),
		      "the tone map's identity matrices are taken", "14.6.1");
	}
	layerHalf(0, kDisplayP3, kRgfxICCCurveType3, kSRGBDecode, kDisplayP3Inverse,
	          kRgfxICCCurveType4, kSRGBEncode, all, all, "14.6.1");
	check(curve(11, 7, 0, kRgfxICCCurveType3, kSRGBDecode), "Blend2XYZ's curve (11) is held",
	      "14.6.1");
	check(matrix(kRgfxICCMatrixBlend2XYZ, 0, kDisplayP3), "Blend2XYZ's matrix is held", "14.6.1");
	check(matrix(kRgfxICCMatrixXYZ2Device, 0, kDisplayP3Inverse),
	      "XYZ2Device's matrix inverts it", "14.6.1");
	check(curve(14, 7, 0, kRgfxICCCurveType4, kSRGBEncode), "XYZ2Device's curve (14) inverts it",
	      "14.6.1");
	const float one[7] = {1, 0, 0, 0, 0, 0, 0};
	check(curve(13, 7, 0, kRgfxICCCurveBypass, one), "SystemGamma's Bypass (13) is taken", "14.6.1");
	swapId = 2;
	check(curve(kRgfxICCCurveSrc2XYZ, 7, 0, kRgfxICCCurveType3, kSRGBDecode) &&
	          matrix(kRgfxICCMatrixSrc2XYZ, 0, kDisplayP3),
	      "layer 0's source is held again", "14.6.1");

	// The display's own colour space, standard sRGB: layer 0's source is still Display P3 for
	// this swap, so its XYZ2Blend matrix is refused; layers 1 and 2 have none.
	swapId = 0x27;
	const float sRGBInverse[9] = {3.13437f, -1.61734f, -0.490746f, -0.978868f, 1.91622f,
	                              0.0334624f, 0.0719889f, -0.228979f, 1.40571f};
	const bool own[3] = {false, true, true};
	layerHalf(-1, nullptr, 0, nullptr, sRGBInverse, kRgfxICCCurveType4, kSRGBEncode, own, all,
	          "14.6.1 own");
	check(curve(11, 7, 0, kRgfxICCCurveType3, kSRGBDecode) &&
	          matrix(kRgfxICCMatrixBlend2XYZ, 0, kSRGB) &&
	          matrix(kRgfxICCMatrixXYZ2Device, 0, sRGBInverse) &&
	          curve(14, 7, 0, kRgfxICCCurveType4, kSRGBEncode),
	      "the display's half in sRGB is taken", "14.6.1 own");
	const float gammaOne[7] = {1, 1, 0, 0, 0, 0, 0};
	for (uint32_t mask : {1u, 2u, 4u})
		check(!curve(15, mask, 0, kRgfxICCCurveType2, gammaOne), "user gamma (15) is refused",
		      "14.6.1 own");
	swapId = 0x28;
	check(curve(kRgfxICCCurveSrc2XYZ, 7, 0, kRgfxICCCurveType3, kSRGBDecode) &&
	          matrix(kRgfxICCMatrixSrc2XYZ, 0, kSRGB),
	      "layer 0's sRGB source is held", "14.6.1 own");
	check(matrix(kRgfxICCMatrixXYZ2Blend, 0, sRGBInverse),
	      "and an XYZ2Blend that inverts it is taken (not in the capture)", "14.6.1 own");
	swapId = 0x43;
	check(curve(16, 7, 0, kRgfxICCCurveType3, gammaOne), "the identity black point (16) is taken",
	      "14.6.1 own");
}

// Each measured release's numbering; a block of another release's numbers, a location not
// measured on the release, or any block of a release not measured, is refused, and so is every
// block of a state with no numbering.
void releases()
{
	const RgfxICCRelease r14 = releaseOf(23), r15 = releaseOf(24), r26 = releaseOf(25),
	                     r27 = releaseOf(27);
	check(r14.matrixBlock == 0x32 && r14.curveBlock == 0x33 && r14.curveBlockSize == 0x248 &&
	          r14.curveBlend2XYZ == 11,
	      "14.6.1's numbers", "releases");
	check(r15.matrixBlock == 0x32 && r15.curveBlock == 0x33 && r15.curveBlockSize == 0x258 &&
	          r15.curveBlend2XYZ == 12 && r15.curveMeasured == r26.curveMeasured &&
	          r15.matrixMeasured == r26.matrixMeasured,
	      "15.6.1's numbers", "releases");
	check(r26.matrixBlock == 0x32 && r26.curveBlock == 0x33 && r26.curveBlockSize == 0x290 &&
	          r26.curveBlend2XYZ == 12,
	      "26.6.2's numbers", "releases");
	check(r27.matrixBlock == 0x33 && r27.curveBlock == 0x34 && r27.curveBlockSize == 0x290 &&
	          r27.curveBlend2XYZ == 12,
	      "27.0.1's and 27.2.1's numbers", "releases");
	RgfxICCRelease none{};
	for (int major : {22, 26, 28})
		check(!rgfxICCReleaseOf(major, &none), "a release not measured has none", "releases");

	// The curve locations each release maps onto 27.0.1's, and those it leaves unmeasured.
	struct Expect {
		const RgfxICCRelease *release;
		uint32_t location;
		int canonical; // -1: not measured
	};
	const Expect expects[] = {
	    {&r14, 0, kRgfxICCCurveSrc2XYZ},      {&r14, 1, kRgfxICCCurveXYZ2Blend},
	    {&r14, 2, -1},                        {&r14, 3, 3},
	    {&r14, 6, 6},                         {&r14, 7, 7},
	    {&r14, 10, -1},                       {&r14, 11, kRgfxICCCurveBlend2XYZ},
	    {&r14, 12, -1},                       {&r14, 13, kRgfxICCCurveSystemGamma},
	    {&r14, 14, kRgfxICCCurveXYZ2Device},  {&r14, 15, kRgfxICCCurveUserGammaAdjustment},
	    {&r14, 16, kRgfxICCCurveBlackpoint},  {&r14, 17, -1},
	    {&r14, 18, -1},                       {&r26, 11, 11},
	    {&r26, 12, kRgfxICCCurveBlend2XYZ},   {&r26, 13, -1},
	    {&r26, 14, kRgfxICCCurveSystemGamma}, {&r26, 15, kRgfxICCCurveXYZ2Device},
	    {&r26, 16, -1},                       {&r26, 17, -1},
	    {&r26, 18, -1},                       {&r26, 2, -1},
	    {&r27, 2, 2},                         {&r27, 13, kRgfxICCCurveDstMid},
	    {&r27, 18, kRgfxICCCurveDICOM},       {&r27, 19, -1},
	    {&r27, 64, -1},                       {&r15, 11, 11},
	    {&r15, 12, kRgfxICCCurveBlend2XYZ},   {&r15, 13, -1},
	    {&r15, 16, -1},
	};
	for (const Expect &e : expects) {
		uint32_t canonical = 0;
		const bool known = rgfxICCCurveLocation(*e.release, e.location, &canonical);
		check(e.canonical < 0 ? !known : known && canonical == static_cast<uint32_t>(e.canonical),
		      "a curve location maps as measured", "releases");
	}
	for (uint32_t loc = 0; loc < 8; loc++) {
		check(rgfxICCMatrixLocation(r14, loc) == (loc < 6) &&
		          rgfxICCMatrixLocation(r15, loc) == (loc < 6) &&
		          rgfxICCMatrixLocation(r26, loc) == (loc < 6) && rgfxICCMatrixLocation(r27, loc),
		      "the matrix locations measured", "releases");
	}
	check(!rgfxICCMatrixLocation(r27, 8), "past the matrix locations", "releases");

	uint8_t m[kRgfxICCMatrixBlockSize], c[kCurveBytes];
	matrixBlock(m, kIdentity);
	curveBlock(c, kRgfxICCCurveBypass, kZeros);
	const uint64_t ms[3] = {1, kRgfxICCMatrixPreToneMap, 0};
	const uint64_t cs[4] = {1, kRgfxICCCurveSystemGamma, 7, 0};
	fresh(25, "releases");
	check(rgfxICCTake(&state, 0x32, 0, ms, 3, m, sizeof(m)) &&
	          rgfxICCTake(&state, 0x33, 0, cs, 4, c, 0x290),
	      "26.6.2's identity blocks are taken under its numbers", "releases");
	check(!rgfxICCTake(&state, 0x33, 0, ms, 3, m, sizeof(m)) &&
	          !rgfxICCTake(&state, 0x34, 0, cs, 4, c, 0x290) &&
	          !rgfxICCTake(&state, 0x33, 0, cs, 4, c, 0x248) &&
	          !rgfxICCTake(&state, 0x33, 0, cs, 4, c, 0x258),
	      "27.0.1's, 14.6.1's and 15.6.1's are refused under 26.6.2's", "releases");
	const uint64_t harmony[3] = {1, kRgfxICCMatrixHarmony, 0};
	const uint64_t dicom[4] = {1, kRgfxICCCurveDICOM, 7, 0};
	check(!rgfxICCTake(&state, 0x32, 0, harmony, 3, m, sizeof(m)) &&
	          !rgfxICCTake(&state, 0x33, 0, dicom, 4, c, 0x290),
	      "a location not measured on 26.6.2 is refused", "releases");
	fresh(23, "releases");
	const uint64_t cs14[4] = {1, 13, 7, 0};
	check(rgfxICCTake(&state, 0x33, 0, cs14, 4, c, 0x248),
	      "14.6.1's 0x248-byte SystemGamma (13) is taken", "releases");
	check(!rgfxICCTake(&state, 0x33, 0, cs14, 4, c, 0x290),
	      "a 0x290-byte curve block is refused under 14.6.1's", "releases");
	fresh(24, "releases");
	check(rgfxICCTake(&state, 0x33, 0, cs, 4, c, 0x258) &&
	          !rgfxICCTake(&state, 0x33, 0, cs, 4, c, 0x290),
	      "15.6.1 takes its 0x258-byte curve block and refuses a 0x290-byte one", "releases");
	fresh(27, "releases");
	check(rgfxICCTake(&state, 0x33, 0, ms, 3, m, sizeof(m)) &&
	          rgfxICCTake(&state, 0x34, 0, cs, 4, c, 0x290) &&
	          rgfxICCTake(&state, 0x33, 0, harmony, 3, m, sizeof(m)) &&
	          rgfxICCTake(&state, 0x34, 0, dicom, 4, c, 0x290),
	      "27.0.1's identity blocks are taken under its numbers", "releases");
	check(!rgfxICCTake(&state, 0x32, 0, ms, 3, m, sizeof(m)) &&
	          !rgfxICCTake(&state, 0x33, 0, cs, 4, c, 0x290),
	      "26.6.2's are refused under 27.0.1's", "releases");
	state = RgfxICCState{};
	check(!rgfxICCTake(&state, 0x33, 0, ms, 3, m, sizeof(m)) &&
	          !rgfxICCTake(&state, 0, 0, ms, 3, m, sizeof(m)),
	      "a state with no numbering takes nothing", "releases");
}

// The CURSOR_COLOR the cursor's chain asks for, where one is due; checks it carries `toXYZ`,
// `fromXYZ` and the curves as the blocks did, where `on`.
bool cursorPost(RgfxCursorColor *c)
{
	uint32_t changes = 0;
	if (!rgfxICCCursorColor(&state, c, &changes))
		return false;
	check(changes == state.cursor.changes, "it carries the count of changes", "cursor post");
	rgfxICCCursorPosted(&state, changes);
	check(!rgfxICCCursorDue(&state), "once posted, nothing is due", "cursor post");
	return true;
}

void checkCurves(const RgfxCursorCurve curves[3], uint32_t type, const float p[7], const char *about)
{
	for (int ch = 0; ch < 3; ch++) {
		check(curves[ch].form == type, "a channel's curve carries the block's type", about);
		for (int i = 0; i < 7; i++)
			check(curves[ch].params[i] == (type == kRgfxICCCurveBypass ? 0 : bitsOf(p[i])),
			      "and its parameters' bits", about);
	}
}

void checkMatrix(const uint32_t m[9], const float want[9], const char *about)
{
	for (int i = 0; i < 9; i++)
		check(m[i] == bitsOf(want[i]), "a matrix carries the block's floats' bits", about);
}

// Where the host converts the cursor (it took CURSOR_COLOR), each block of layer 1's chain the
// display reads is taken, whatever it holds, and the chain goes to the host whole: off until the
// cursor's first image completes it, on with the blocks' values after it and after each profile
// switch, nothing again until it changes, and off where it is the identity within the guest's
// precision. Layer 0's and the display's halves are judged as before, and so is layer 1 where
// the host did not take CURSOR_COLOR (bursts).
void cursorChain()
{
	fresh(27, "cursor");
	state.cursor.host = true;
	const bool all[3] = {true, true, true};
	RgfxCursorColor c{};
	check(!cursorPost(&c), "nothing is due before a block", "cursor");
	// The login's first burst: layers 1 and 2 get their XYZ2Blend halves alone.
	swapId = 1;
	displayHalf(kDisplayP3, kDisplayP3Inverse, kRgfxICCCurveType3, kSRGBDecode,
	            kRgfxICCCurveType4, kSRGBEncode, true, "cursor start");
	layerHalf(0, kDisplayP3, kRgfxICCCurveType3, kSRGBDecode, kDisplayP3Inverse,
	          kRgfxICCCurveType4, kSRGBEncode, all, all, "cursor start");
	check(cursorPost(&c) && c.on == 0, "half a chain is posted off", "cursor start");
	// The display's own profile.
	swapId = 0x4b;
	displayHalf(kPublished, kPublishedInverse, kRgfxICCCurveType3, kSRGBDecode,
	            kRgfxICCCurveType4, kSRGBEncode, true, "cursor own");
	layerHalf(0, kPublished, kRgfxICCCurveType3, kSRGBDecode, kPublishedInverse,
	          kRgfxICCCurveType4, kSRGBEncode, all, all, "cursor own");
	check(cursorPost(&c) && c.on == 0, "still half a chain", "cursor own");
	// The cursor's first image: its standard sRGB source, 0.0024 off the blend space's inverse, is
	// taken now, and the chain goes to the host on.
	swapId = 0x8c;
	layerHalf(1, kSRGB, kRgfxICCCurveType3, kSRGBDecode, kPublishedInverse, kRgfxICCCurveType4,
	          kSRGBEncode, all, all, "cursor image");
	check(cursorPost(&c) && c.on == 1, "the whole chain is posted on", "cursor image");
	checkCurves(c.source_curves, kRgfxICCCurveType3, kSRGBDecode, "cursor image");
	checkMatrix(c.to_xyz, kSRGB, "cursor image");
	checkMatrix(c.from_xyz, kPublishedInverse, "cursor image");
	checkCurves(c.blend_curves, kRgfxICCCurveType4, kSRGBEncode, "cursor image");
	check(!cursorPost(&c), "nothing again until it changes", "cursor image");
	// Generic RGB: layer 1 keeps its source, and its blend space's matrix and curve, Generic
	// RGB's, which the source's do not invert, are taken.
	swapId = 0xc5;
	displayHalf(kGenericRGB, kGenericRGBInverse, kRgfxICCCurveType3, kGamma18,
	            kRgfxICCCurveType3, kGamma18Inverse, true, "cursor Generic RGB");
	layerHalf(0, kGenericRGB, kRgfxICCCurveType3, kGenericDecode, kGenericRGBInverse,
	          kRgfxICCCurveType3, kGenericEncode, all, all, "cursor Generic RGB");
	check(cursorPost(&c) && c.on == 1, "Generic RGB's chain is posted", "cursor Generic RGB");
	checkCurves(c.source_curves, kRgfxICCCurveType3, kSRGBDecode, "cursor Generic RGB");
	checkMatrix(c.to_xyz, kSRGB, "cursor Generic RGB");
	checkMatrix(c.from_xyz, kGenericRGBInverse, "cursor Generic RGB");
	checkCurves(c.blend_curves, kRgfxICCCurveType3, kGenericEncode, "cursor Generic RGB");
	// Display P3.
	swapId = 0xc8;
	displayHalf(kDisplayP3, kDisplayP3Inverse, kRgfxICCCurveType3, kSRGBDecode,
	            kRgfxICCCurveType4, kSRGBEncode, true, "cursor P3");
	layerHalf(0, kDisplayP3, kRgfxICCCurveType3, kSRGBDecode, kDisplayP3Inverse,
	          kRgfxICCCurveType4, kSRGBEncode, all, all, "cursor P3");
	check(cursorPost(&c) && c.on == 1, "Display P3's chain is posted", "cursor P3");
	checkMatrix(c.from_xyz, kDisplayP3Inverse, "cursor P3");
	checkCurves(c.blend_curves, kRgfxICCCurveType4, kSRGBEncode, "cursor P3");
	// A source the blend space inverts: the chain is the identity, posted off.
	check(curve(kRgfxICCCurveSrc2XYZ, 7, 1, kRgfxICCCurveType3, kSRGBDecode) &&
	          matrix(kRgfxICCMatrixSrc2XYZ, 1, kDisplayP3),
	      "a Display P3 source is taken", "cursor identity");
	check(cursorPost(&c) && c.on == 0, "an identity chain is posted off", "cursor identity");
	// Channel by channel: green's blend curve alone changes, and the chain is on again.
	check(curve(kRgfxICCCurveXYZ2Blend, 2, 1, kRgfxICCCurveType3, kGamma18Inverse),
	      "a curve of green alone is taken", "cursor channels");
	check(cursorPost(&c) && c.on == 1, "the chain is posted on", "cursor channels");
	check(c.blend_curves[0].form == kRgfxICCCurveType4 &&
	          c.blend_curves[1].form == kRgfxICCCurveType3 &&
	          c.blend_curves[2].form == kRgfxICCCurveType4,
	      "with green's curve its own", "cursor channels");
	// A source the display cannot read leaves the chain without it: off.
	float nan[9];
	std::memcpy(nan, kSRGB, sizeof(nan));
	nan[4] = NAN;
	check(!matrix(kRgfxICCMatrixSrc2XYZ, 1, nan), "a NaN source matrix is refused",
	      "cursor unread");
	check(cursorPost(&c) && c.on == 0, "the chain without it is posted off", "cursor unread");
	check(!curve(kRgfxICCCurveXYZ2Blend, 7, 1, 7, kZeros), "a PQ blend curve is refused",
	      "cursor unread");
	check(!rgfxICCCursorDue(&state), "and changes nothing", "cursor unread");
	check(matrix(kRgfxICCMatrixSrc2XYZ, 1, kSRGB), "a readable source is taken again",
	      "cursor unread");
	check(cursorPost(&c) && c.on == 1, "and the chain is on again", "cursor unread");
	// Other layers are judged as before: layer 2's XYZ2Blend against its own source.
	check(matrix(kRgfxICCMatrixSrc2XYZ, 2, kSRGB) &&
	          !matrix(kRgfxICCMatrixXYZ2Blend, 2, kDisplayP3Inverse),
	      "layer 2's half is still judged", "cursor others");
	// A post that did not go out leaves the chain due.
	check(matrix(kRgfxICCMatrixXYZ2Blend, 1, kPublishedInverse), "a new blend matrix", "cursor due");
	uint32_t changes = 0;
	check(rgfxICCCursorColor(&state, &c, &changes) && rgfxICCCursorDue(&state),
	      "built but not posted, it stays due", "cursor due");
	check(matrix(kRgfxICCMatrixXYZ2Blend, 1, kDisplayP3Inverse), "another, meanwhile",
	      "cursor due");
	rgfxICCCursorPosted(&state, changes);
	check(rgfxICCCursorDue(&state), "a change after the one posted stays due", "cursor due");
	// A host that did not take CURSOR_COLOR is never sent the chain.
	fresh(27, "cursor no host");
	const bool judged[3] = {true, false, true};
	layerHalf(1, kSRGB, kRgfxICCCurveType3, kSRGBDecode, kPublishedInverse, kRgfxICCCurveType4,
	          kSRGBEncode, judged, all, "cursor no host");
	check(!rgfxICCCursorDue(&state) && !cursorPost(&c), "nothing is due", "cursor no host");
}

std::string readFile(const char *path)
{
	std::ifstream in(path);
	check(in.good(), "the file can be read", path);
	return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// The override takes a block as rgfxICCTake says, under the display's lock, and answers 0 or
// kIOReturnUnsupported, posting nothing.
void source()
{
	const std::string src = readFile("src/ReimsGraphicsMobile.cpp");
	const size_t at = src.find("RgfxIOMFBWord ReimsGraphics::set_block(");
	check(at != std::string::npos, "the override is defined", "source");
	const size_t end = src.find("\n}\n", at);
	check(end != std::string::npos, "the override ends", "source");
	const std::string body = src.substr(at, end - at);
	check(body.find("rgfxICCTake(mobileICC, type, offset, scalars, count, data, size)") !=
	          std::string::npos,
	      "it takes the block as rgfxICCTake says", "source");
	check(body.find("IOLockLock(mobileICCLock)") != std::string::npos &&
	          body.find("IOLockUnlock(mobileICCLock)") != std::string::npos,
	      "under the display's lock", "source");
	check(body.find("word(took ? kIOReturnSuccess : kIOReturnUnsupported)") != std::string::npos,
	      "it answers 0 or kIOReturnUnsupported", "source");
	check(body.find("post") == std::string::npos, "it posts nothing", "source");
	check(src.find("rgfxICCReleaseOf(version_major, &iccRelease)") != std::string::npos &&
	          src.find("mobileICC->release = iccRelease;") != std::string::npos,
	      "the display takes its release's block types by the kernel's major version", "source");
	check(src.find("mobileICC->cursor.host = cursorColorTaken;") != std::string::npos,
	      "the cursor's chain is the host's where it took CURSOR_COLOR", "source");
	// Each swap posts the cursor's chain where it changed, before its cursor's image.
	const size_t swaps = src.find("void ReimsGraphics::beginMobileSwaps()");
	const size_t swapsEnd = src.find("\n}\n", swaps);
	check(swaps != std::string::npos && swapsEnd != std::string::npos, "the swaps begin", "source");
	const std::string begin = src.substr(swaps, swapsEnd - swaps);
	const size_t posted = begin.find("postMobileCursorColor();");
	check(posted != std::string::npos && posted < begin.find("showMobileCursor("),
	      "the chain is posted before the swap's cursor", "source");
	const size_t post = src.find("void ReimsGraphics::postMobileCursorColor()");
	const size_t postEnd = src.find("\n}\n", post);
	check(post != std::string::npos && postEnd != std::string::npos, "the post is defined",
	      "source");
	const std::string postBody = src.substr(post, postEnd - post);
	check(postBody.find("rgfxICCCursorColor(mobileICC, &c, &changes)") != std::string::npos &&
	          postBody.find("tryPost(RGFX_CMD_CURSOR_COLOR, &c, sizeof(c), nullptr)") !=
	              std::string::npos &&
	          postBody.find("rgfxICCCursorPosted(mobileICC, changes)") > postBody.find("tryPost("),
	      "it posts the chain without waiting, and records it once it went out", "source");
}

} // namespace

int main()
{
	conversion();
	math();
	curves();
	matrices();
	bursts();
	burstsOf(24, "15.6.1");
	burstsOf(25, "26.6.2");
	bursts14();
	releases();
	pairs();
	others();
	cursorChain();
	source();
	std::printf("mobile_icc_test: %d checks passed\n", checks);
	return 0;
}
