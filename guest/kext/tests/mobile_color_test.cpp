// The native test of RgfxMobileColor.hpp: the DISPLAY_COLOR matrix rows the arm64 display posts for
// the colour matrix WindowServer sets with a swap (ReimsGraphicsMobile.cpp's
// swap_set_color_matrix), against the same conversion done in floating point here: each word
// converted to a double, times 2^-32, rounded to a float. Also the override's source: it converts
// the matrix, posts it as the matrix stage, and answers 0.
#include "RgfxMobileColor.hpp"

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
		std::fprintf(stderr, "mobile_color_test: FAILED (%s): %s\n", about, what);
		std::exit(1);
	}
}

// The conversion in floating point: to a double, times 2^-32, to a float.
uint32_t floatBits(int64_t word)
{
	const float f = static_cast<float>(static_cast<double>(word) * 0x1p-32);
	uint32_t bits = 0;
	std::memcpy(&bits, &f, sizeof(bits));
	return bits;
}

// A float's word as kern_SwapSetColorMatrix makes it: times 2^32, rounded toward zero.
int64_t wordOf(float f)
{
	return static_cast<int64_t>(static_cast<double>(f) * 0x1p32);
}

uint32_t bitsOf(float f)
{
	uint32_t bits = 0;
	std::memcpy(&bits, &f, sizeof(bits));
	return bits;
}

// Each word converts as floating point converts it: edges, the boundaries where a double
// and then a float round (ties among them), and a pseudo-random spread over every magnitude.
void conversion()
{
	const int64_t edges[] = {
	    0, 1, -1, 2, 3, 0x7fffffff, int64_t(1) << 32, -(int64_t(1) << 32), (int64_t(1) << 32) + 1,
	    INT64_MAX, INT64_MIN, INT64_MIN + 1, INT64_MAX - 1, int64_t(1) << 62, -(int64_t(1) << 62),
	};
	for (int64_t w : edges)
		check(rgfxFixed32ToFloatBits(w) == floatBits(w),
		      "an edge word converts as two roundings do", "edge");
	// Around each power of two, for the first rounding (past 53 bits) and the second (past 24):
	// halfway points and one either side, which a single rounding gets otherwise than two.
	for (int p = 1; p < 63; p++) {
		const int64_t base = int64_t(1) << p;
		for (int drop : {p - 53, p - 24, p - 25, p - 54}) {
			if (drop < 1)
				continue;
			const int64_t half = int64_t(1) << (drop - 1);
			for (int64_t k = 0; k < 6; k++) {
				const int64_t mid = base + (k << drop) + half;
				for (int64_t d = -2; d <= 2; d++) {
					for (int64_t w : {mid + d, -(mid + d)})
						check(rgfxFixed32ToFloatBits(w) == floatBits(w),
						      "a word near a boundary converts as two roundings do", "tie");
				}
			}
		}
	}
	uint64_t x = 0x9e3779b97f4a7c15ull;
	for (int i = 0; i < 2000000; i++) {
		x ^= x << 13;
		x ^= x >> 7;
		x ^= x << 17;
		const int64_t w = static_cast<int64_t>(x) >> (x % 63);
		check(rgfxFixed32ToFloatBits(w) == floatBits(w), "a word converts as two roundings do",
		      "spread");
	}
}

// NULL is the identity; a matrix gives its nine words' rows with a fourth 0, reading neither
// field_00 nor field_60.
void rows()
{
	RgfxColorEntry r[kRgfxMobileMatrixRows];
	rgfxMobileColorMatrix(nullptr, r);
	const uint32_t one = bitsOf(1.0f);
	for (uint32_t i = 0; i < kRgfxMobileMatrixRows; i++) {
		const uint32_t row[4] = {r[i].x, r[i].y, r[i].z, r[i].w};
		for (uint32_t j = 0; j < 4; j++)
			check(row[j] == (i == j ? one : 0), "NULL is the identity", "identity");
	}

	// A grey matrix, as WindowServer's words of floats carry it: each row the same weights.
	const float grey[3] = {0.2126f, 0.7152f, 0.0722f};
	int64_t words[kRgfxMobileMatrixWords];
	for (uint32_t k = 0; k < kRgfxMobileMatrixWords; k++)
		words[k] = int64_t(0x5a5a5a5a5a5a5a5a) - k; // field_00 and field_60: not read
	for (uint32_t i = 0; i < 3; i++)
		for (uint32_t j = 0; j < 3; j++)
			words[kRgfxMobileMatrixFirstWord + 3 * i + j] = wordOf(grey[j]);
	rgfxMobileColorMatrix(words, r);
	for (uint32_t i = 0; i < kRgfxMobileMatrixRows; i++) {
		const uint32_t row[4] = {r[i].x, r[i].y, r[i].z, r[i].w};
		for (uint32_t j = 0; j < 3; j++)
			check(row[j] == floatBits(wordOf(grey[j])), "row i is the matrix's row i", "grey");
		check(row[3] == 0, "the fourth entry is 0", "grey");
	}

	// Rows and columns apart: entry (i, j) is word 3 + 3i + j.
	for (uint32_t i = 0; i < 3; i++)
		for (uint32_t j = 0; j < 3; j++)
			words[kRgfxMobileMatrixFirstWord + 3 * i + j] =
			    wordOf(static_cast<float>(10 * i + j) - 4.5f);
	rgfxMobileColorMatrix(words, r);
	for (uint32_t i = 0; i < kRgfxMobileMatrixRows; i++) {
		const uint32_t row[3] = {r[i].x, r[i].y, r[i].z};
		for (uint32_t j = 0; j < 3; j++)
			check(row[j] == bitsOf(static_cast<float>(10 * i + j) - 4.5f),
			      "entry (i, j) is word 3 + 3i + j", "order");
	}
	// An identity matrix's words give the identity's rows, which the host stages off.
	for (uint32_t i = 0; i < 3; i++)
		for (uint32_t j = 0; j < 3; j++)
			words[kRgfxMobileMatrixFirstWord + 3 * i + j] = i == j ? int64_t(1) << 32 : 0;
	rgfxMobileColorMatrix(words, r);
	for (uint32_t i = 0; i < kRgfxMobileMatrixRows; i++) {
		const uint32_t row[4] = {r[i].x, r[i].y, r[i].z, r[i].w};
		for (uint32_t j = 0; j < 4; j++)
			check(row[j] == (i == j ? one : 0), "the identity's words are the identity", "ones");
	}
}

std::string readFile(const char *path)
{
	std::ifstream in(path);
	check(in.good(), "the file can be read", path);
	return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// The override converts the matrix, posts it as the matrix stage where the host took the colour,
// and answers 0 whatever the post did.
void source()
{
	const std::string src = readFile("src/ReimsGraphicsMobile.cpp");
	const size_t at = src.find("RgfxIOMFBWord ReimsGraphics::swap_set_color_matrix(");
	check(at != std::string::npos, "the override is defined", "source");
	const size_t end = src.find("\n}\n", at);
	check(end != std::string::npos, "the override ends", "source");
	const std::string body = src.substr(at, end - at);
	check(body.find("rgfxMobileColorMatrix(reinterpret_cast<const int64_t *>(matrix), rows);") !=
	          std::string::npos,
	      "it converts the matrix", "source");
	const char *post =
	    "displayColor() ? postDisplayColor(0, RGFX_COLOR_MATRIX, kRgfxMobileMatrixRows,";
	check(body.find(post) != std::string::npos,
	      "it posts the matrix stage where the host took the colour", "source");
	check(body.find("return word(kIOReturnSuccess);") != std::string::npos, "it answers 0",
	      "source");
	check(body.find("return word(posted)") == std::string::npos,
	      "the post's result is not the answer", "source");
}

} // namespace

int main()
{
	static_assert(kRgfxMobileMatrixFirstWord + 3 * kRgfxMobileMatrixRows + 3 ==
	                  kRgfxMobileMatrixWords,
	              "field_00, the matrix and field_60");
	static_assert(kRgfxMobileMatrixRows == 3, "the wire's matrix stage has three entries");
	conversion();
	rows();
	source();
	std::printf("mobile_color_test: %d checks passed\n", checks);
	return 0;
}
