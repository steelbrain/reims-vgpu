// The display pipe's colour (RgfxPipeColor.hpp), built natively: the gamma selectors' structures
// are decoded and refused as macOS 26's kernel refuses them, a table belongs to the transaction
// begun, plane 1's is accepted and not shown, and a stage goes to the host in parts of at most
// RGFX_COLOR_PART_ENTRIES entries, in order, its entries as they came.
#include <cstdio>
#include <cstring>
#include <vector>

#include "RgfxPipeColor.hpp"

static int g_checks = 0, g_failed = 0;
#define CHECK(e)                                                                   \
	do {                                                                           \
		g_checks++;                                                                \
		if (!(e)) {                                                                \
			g_failed++;                                                            \
			std::fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #e); \
		}                                                                          \
	} while (0)

static uint32_t bits(float f)
{
	uint32_t v;
	std::memcpy(&v, &f, sizeof(v));
	return v;
}

static void put32(std::vector<uint8_t> &b, uint32_t off, uint32_t v)
{
	std::memcpy(b.data() + off, &v, sizeof(v));
}

// A gamma selector's structure of `n` entries, entry k's red, green and blue k, 2k and 3k, as
// CoreDisplay sends it (field_10 0.0, field_14 1.0), `extra` bytes past its entries.
static std::vector<uint8_t> gamma(uint32_t txn, uint32_t plane, uint32_t n, uint32_t extra = 0)
{
	std::vector<uint8_t> b(kRgfxGammaHeaderSize + n * kRgfxGammaEntrySize + extra);
	put32(b, 0x00, txn);
	put32(b, 0x04, plane);
	put32(b, 0x08, n);
	put32(b, 0x10, bits(0.0f));
	put32(b, 0x14, bits(1.0f));
	for (uint32_t k = 0; k < n; k++) {
		const uint32_t at = kRgfxGammaHeaderSize + k * kRgfxGammaEntrySize;
		put32(b, at + 0, bits(static_cast<float>(k)));
		put32(b, at + 4, bits(static_cast<float>(2 * k)));
		put32(b, at + 8, bits(static_cast<float>(3 * k)));
	}
	return b;
}

static void decode()
{
	RgfxGammaArgs a = {};
	std::vector<uint8_t> b = gamma(7, 0, 1024);
	CHECK(rgfxGammaDecode(b.data(), b.size(), &a));
	CHECK(a.transaction == 7 && a.plane == 0 && a.count == 1024);
	CHECK(a.field_0c == 0 && a.field_10 == bits(0.0f) && a.field_14 == bits(1.0f));

	// Shorter than the header, or than its entries: refused.
	CHECK(!rgfxGammaDecode(b.data(), kRgfxGammaHeaderSize - 1, &a));
	CHECK(!rgfxGammaDecode(nullptr, 0, &a));
	CHECK(!rgfxGammaDecode(b.data(), b.size() - 1, &a));
	// Longer is taken.
	std::vector<uint8_t> longer = gamma(7, 0, 4, 32);
	CHECK(rgfxGammaDecode(longer.data(), longer.size(), &a) && a.count == 4);
	// None: the header alone.
	std::vector<uint8_t> none = gamma(7, 0, 0);
	CHECK(rgfxGammaDecode(none.data(), none.size(), &a) && a.count == 0);

	// 0x20000 entries at most, whatever the size says.
	std::vector<uint8_t> most(kRgfxGammaHeaderSize);
	put32(most, 0x08, kRgfxGammaMaxEntries);
	CHECK(!rgfxGammaDecode(most.data(), most.size(), &a));
	CHECK(rgfxGammaDecode(most.data(),
	                      kRgfxGammaHeaderSize + uint64_t(kRgfxGammaMaxEntries) * kRgfxGammaEntrySize,
	                      &a));
	put32(most, 0x08, kRgfxGammaMaxEntries + 1);
	CHECK(!rgfxGammaDecode(most.data(), uint64_t(1) << 32, &a));
	// A count whose bytes pass 2^32 is refused, not wrapped.
	put32(most, 0x08, 0x10000000);
	CHECK(!rgfxGammaDecode(most.data(), 0x100000018ull, &a));
}

static void actions()
{
	RgfxGammaArgs a = {};
	a.transaction = 5;

	// The begun transaction's: staged, by any of the three.
	CHECK(rgfxGammaAction(kRgfxPipePostGamma, a, 5) == RgfxGammaAction::Stage);
	CHECK(rgfxGammaAction(kRgfxPipePreGamma, a, 5) == RgfxGammaAction::Stage);
	CHECK(rgfxGammaAction(kRgfxPipePlaneGamma, a, 5) == RgfxGammaAction::Stage);
	// Another's, or none begun: success, nothing changed.
	CHECK(rgfxGammaAction(kRgfxPipePostGamma, a, 6) == RgfxGammaAction::Ignore);
	CHECK(rgfxGammaAction(kRgfxPipePostGamma, a, 0) == RgfxGammaAction::Ignore);
	a.transaction = 0;
	CHECK(rgfxGammaAction(kRgfxPipePostGamma, a, 0) == RgfxGammaAction::Ignore);

	// Selector 5's plane: 1 accepted and not shown, past 1 refused, but only for the begun
	// transaction's; 6 and 7 read no plane.
	a.transaction = 5;
	a.plane = 1;
	CHECK(rgfxGammaAction(kRgfxPipePlaneGamma, a, 5) == RgfxGammaAction::Ignore);
	a.plane = 2;
	CHECK(rgfxGammaAction(kRgfxPipePlaneGamma, a, 5) == RgfxGammaAction::BadArgument);
	CHECK(rgfxGammaAction(kRgfxPipePlaneGamma, a, 4) == RgfxGammaAction::Ignore);
	CHECK(rgfxGammaAction(kRgfxPipePostGamma, a, 5) == RgfxGammaAction::Stage);
	CHECK(rgfxGammaAction(kRgfxPipePreGamma, a, 5) == RgfxGammaAction::Stage);

	CHECK(rgfxGammaSlot(5) == 0 && rgfxGammaSlot(6) == 1 && rgfxGammaSlot(7) == 2);
	CHECK(rgfxGammaSlot(8) == -1 && rgfxGammaSlot(4) == -1);
	CHECK(rgfxGammaSlotStage(0) == RGFX_COLOR_PLANE_GAMMA);
	CHECK(rgfxGammaSlotStage(1) == RGFX_COLOR_PRE_GAMMA);
	CHECK(rgfxGammaSlotStage(2) == RGFX_COLOR_POST_GAMMA);
}

static void parts()
{
	CHECK(rgfxColorPartCount(0) == 1);
	CHECK(rgfxColorPartCount(1) == 1 && rgfxColorPartCount(256) == 1);
	CHECK(rgfxColorPartCount(257) == 2 && rgfxColorPartCount(8190) == 32);
	CHECK(rgfxColorPartCount(kRgfxGammaMaxEntries) == 512);
	CHECK(kRgfxColorPartMax == 40 + 256 * 16);

	// 300 entries: 256, then 44, each entry as it came.
	std::vector<uint8_t> b = gamma(9, 0, 300);
	const uint8_t *entries = b.data() + kRgfxGammaHeaderSize;
	std::vector<uint8_t> part(kRgfxColorPartMax);
	uint32_t size = rgfxColorPart(part.data(), 0, 9, RGFX_COLOR_POST_GAMMA, 300, entries, 0);
	RgfxDisplayColor m;
	std::memcpy(&m, part.data(), sizeof(m));
	CHECK(size == 40 + 256 * 16 && m.header.size == size && size % 8 == 0);
	CHECK(m.header.type == RGFX_CMD_DISPLAY_COLOR);
	CHECK(m.display == 0 && m.transaction_id == 9 && m.stage == RGFX_COLOR_POST_GAMMA);
	CHECK(m.entry_count == 300 && m.first_entry == 0 && m.part_entries == 256);
	CHECK(std::memcmp(part.data() + 40, entries, 256 * 16) == 0);

	size = rgfxColorPart(part.data(), 0, 9, RGFX_COLOR_POST_GAMMA, 300, entries, 1);
	std::memcpy(&m, part.data(), sizeof(m));
	CHECK(size == 40 + 44 * 16 && m.header.size == size && size % 8 == 0);
	CHECK(m.first_entry == 256 && m.part_entries == 44 && m.entry_count == 300);
	RgfxColorEntry last;
	std::memcpy(&last, part.data() + 40 + 43 * 16, sizeof(last));
	CHECK(last.x == bits(299.0f) && last.y == bits(598.0f) && last.z == bits(897.0f));

	// None: one part of no entries, the stage off.
	size = rgfxColorPart(part.data(), 0, 9, RGFX_COLOR_PRE_GAMMA, 0, nullptr, 0);
	std::memcpy(&m, part.data(), sizeof(m));
	CHECK(size == 40 && m.entry_count == 0 && m.part_entries == 0 && m.first_entry == 0);
}

static void matrix()
{
	// A transaction's matrix: three rows of four floats at 0xb0, carried as three entries.
	std::vector<uint8_t> t(0x118);
	const float rows[12] = {0.5f, 0, 0, 0.1f, 0, 0.25f, 0, 0.2f, 0, 0, 1, 0};
	std::memcpy(t.data() + kRgfxTxnMatrix, rows, sizeof(rows));
	CHECK(kRgfxTxnMatrix + kRgfxMatrixRows * 16 <= 0xe0);
	std::vector<uint8_t> part(kRgfxColorPartMax);
	uint32_t size = rgfxColorPart(part.data(), 0, 3, RGFX_COLOR_MATRIX, kRgfxMatrixRows,
	                              t.data() + kRgfxTxnMatrix, 0);
	RgfxDisplayColor m;
	std::memcpy(&m, part.data(), sizeof(m));
	CHECK(size == 40 + 3 * 16 && m.stage == RGFX_COLOR_MATRIX && m.entry_count == 3);
	RgfxColorEntry row0;
	std::memcpy(&row0, part.data() + 40, sizeof(row0));
	CHECK(row0.x == bits(0.5f) && row0.w == bits(0.1f));
}

static void sends()
{
	CHECK(rgfxPipeSends(true, false) == RgfxPipeSends::Surface);
	CHECK(rgfxPipeSends(true, true) == RgfxPipeSends::Surface);
	CHECK(rgfxPipeSends(false, true) == RgfxPipeSends::Keep);
	CHECK(rgfxPipeSends(false, false) == RgfxPipeSends::Nothing);
	// A surface's present and a colour's kept present complete at the host's answer, once the
	// display shows them; nothing sent, with the transaction before it.
	CHECK(rgfxPipeWaitsForAnswer(RgfxPipeSends::Surface));
	CHECK(rgfxPipeWaitsForAnswer(RgfxPipeSends::Keep));
	CHECK(!rgfxPipeWaitsForAnswer(RgfxPipeSends::Nothing));
}

int main()
{
	decode();
	actions();
	parts();
	matrix();
	sends();
	if (g_failed) {
		std::fprintf(stderr, "pipe_color_test: %d of %d checks failed\n", g_failed, g_checks);
		return 1;
	}
	std::printf("pipe_color_test: OK (%d checks)\n", g_checks);
	return 0;
}
