// RgfxPipeColor: the display pipe's colour (RgfxDisplayPipeUserClient), as its gamma selectors
// and its transactions' colour matrix set it, and the DISPLAY_COLOR parts that carry it to the
// host (reims_graphics_wire.h). Header-only and free of the kernel's headers, so its native test
// (tests/pipe_color_test.cpp) builds it as the kext does.
//
// The gamma selectors: 5 sets plane `plane`'s gamma table, 6 the pipe's table before its colour
// matrix, 7 the pipe's table after it. Each takes one variable-size structure, inline or through
// a descriptor:
//   0x00  the transaction's id
//   0x04  for selector 5, the plane (0 or 1); not read by 6 and 7
//   0x08  the entry count n, at most 0x20000
//   0x0c  field_0c
//   0x10  field_10, a float (CoreDisplay sends 0.0)
//   0x14  field_14, a float (CoreDisplay sends 1.0)
//   0x18  n entries of 16 bytes: red, green and blue as floats, then a fourth word (0)
// as macOS 26's IOAcceleratorFamily2 reads it (read). A structure shorter than its header or its
// entries, or with more than 0x20000 entries, is refused kIOReturnBadArgument. One whose
// transaction is not the one begun changes nothing and succeeds; for one that is, a plane past 1
// is refused kIOReturnBadArgument. The table belongs to that transaction: its submit sets it on
// the display, and a transaction begun after it without that submit drops it. A stage keeps its
// table across transactions until one sets it again.
//
// The colour matrix: a transaction whose dirty mask has kRgfxDirtyMatrix carries it at
// kRgfxTxnMatrix, three rows of four floats (out_i = row_i[0] * r + row_i[1] * g + row_i[2] * b +
// row_i[3]), as macOS 26's IOPresentment puts CoreDisplay's matrix there (read).
//
// The host gets each stage a transaction sets as DISPLAY_COLOR parts of at most
// RGFX_COLOR_PART_ENTRIES entries, in order, then the transaction's PRESENT_SURFACE, which shows
// them: the entries go as they came, 16 bytes each (RgfxColorEntry).
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_PIPE_COLOR_HPP_
#define RGFX_PIPE_COLOR_HPP_

#include <stdint.h>
#include <string.h>

#include "reims_graphics_wire.h"

// A gamma selector's structure: its header's size, an entry's, and the most entries it takes.
constexpr uint32_t kRgfxGammaHeaderSize = 0x18;
constexpr uint32_t kRgfxGammaEntrySize = 16;
constexpr uint32_t kRgfxGammaMaxEntries = 0x20000;

// The gamma selectors.
constexpr uint32_t kRgfxPipePlaneGamma = 5;
constexpr uint32_t kRgfxPipePreGamma = 6;
constexpr uint32_t kRgfxPipePostGamma = 7;

// Inside a transaction (selector 8's 0x118 bytes): the dirty mask's colour-matrix bit, and the
// matrix's three rows.
constexpr uint64_t kRgfxDirtyMatrix = 0x40;
constexpr uint32_t kRgfxTxnMatrix = 0xb0;
constexpr uint32_t kRgfxMatrixRows = 3;

static_assert(sizeof(RgfxColorEntry) == kRgfxGammaEntrySize, "a guest entry is a wire entry");
static_assert(kRgfxGammaMaxEntries == RGFX_COLOR_MAX_ENTRIES, "the host takes what the guest may");

// A gamma selector's header.
struct RgfxGammaArgs {
	uint32_t transaction;
	uint32_t plane;
	uint32_t count;
	uint32_t field_0c;
	uint32_t field_10;
	uint32_t field_14;
};

// What a gamma selector does with its structure.
enum class RgfxGammaAction {
	// The table is the begun transaction's: stage it.
	Stage,
	// Success with nothing changed: another transaction's, or plane 1's, which our display, with
	// one plane, never shows.
	Ignore,
	// kIOReturnBadArgument.
	BadArgument,
};

inline uint32_t rgfxColorRead32(const uint8_t *p, uint32_t off)
{
	uint32_t v;
	memcpy(&v, p + off, sizeof(v));
	return v;
}

// Decodes a gamma selector's `size` bytes at `in` into `*out`: false where the kernel refuses
// them, shorter than the header or its entries, or with more than kRgfxGammaMaxEntries.
inline bool rgfxGammaDecode(const uint8_t *in, uint64_t size, RgfxGammaArgs *out)
{
	if (!in || size < kRgfxGammaHeaderSize)
		return false;
	out->transaction = rgfxColorRead32(in, 0x00);
	out->plane = rgfxColorRead32(in, 0x04);
	out->count = rgfxColorRead32(in, 0x08);
	out->field_0c = rgfxColorRead32(in, 0x0c);
	out->field_10 = rgfxColorRead32(in, 0x10);
	out->field_14 = rgfxColorRead32(in, 0x14);
	if (out->count > kRgfxGammaMaxEntries)
		return false;
	return size >= kRgfxGammaHeaderSize + static_cast<uint64_t>(out->count) * kRgfxGammaEntrySize;
}

// What gamma selector `selector` does with arguments `args` (decoded), `begun` the transaction
// begun and not submitted (0: none).
inline RgfxGammaAction rgfxGammaAction(uint32_t selector, const RgfxGammaArgs &args, uint32_t begun)
{
	if (begun == 0 || args.transaction != begun)
		return RgfxGammaAction::Ignore;
	if (selector == kRgfxPipePlaneGamma) {
		if (args.plane > 1)
			return RgfxGammaAction::BadArgument;
		if (args.plane == 1)
			return RgfxGammaAction::Ignore;
	}
	return RgfxGammaAction::Stage;
}

// The wire's stage (RGFX_COLOR_*) a gamma selector sets, and its slot among the three gamma
// stages a pipe client stages (0 to 2); -1 for any other selector.
inline int rgfxGammaSlot(uint32_t selector)
{
	switch (selector) {
	case kRgfxPipePlaneGamma:
		return 0;
	case kRgfxPipePreGamma:
		return 1;
	case kRgfxPipePostGamma:
		return 2;
	default:
		return -1;
	}
}

inline uint32_t rgfxGammaSlotStage(int slot)
{
	static const uint32_t stages[3] = {RGFX_COLOR_PLANE_GAMMA, RGFX_COLOR_PRE_GAMMA,
	                                   RGFX_COLOR_POST_GAMMA};
	return stages[slot];
}

// How many DISPLAY_COLOR parts carry a stage of `count` entries: one for none (the stage off).
inline uint32_t rgfxColorPartCount(uint32_t count)
{
	return count == 0 ? 1 : (count + RGFX_COLOR_PART_ENTRIES - 1) / RGFX_COLOR_PART_ENTRIES;
}

// The largest DISPLAY_COLOR message: a part of RGFX_COLOR_PART_ENTRIES entries.
constexpr uint32_t kRgfxColorPartMax =
    sizeof(RgfxDisplayColor) + RGFX_COLOR_PART_ENTRIES * sizeof(RgfxColorEntry);

// Fills `*part` (room for kRgfxColorPartMax bytes) with part `k` of stage `stage`'s `count`
// entries at `entries` (16 bytes each), for transaction `transaction` on display `display`: its
// size in bytes, which its header carries. The header's type and seqno are the post's.
inline uint32_t rgfxColorPart(uint8_t *part, uint32_t display, uint32_t transaction,
                              uint32_t stage, uint32_t count, const uint8_t *entries, uint32_t k)
{
	const uint32_t first = k * RGFX_COLOR_PART_ENTRIES;
	const uint32_t left = count > first ? count - first : 0;
	const uint32_t carried = left < RGFX_COLOR_PART_ENTRIES ? left : RGFX_COLOR_PART_ENTRIES;
	const uint32_t size = sizeof(RgfxDisplayColor) + carried * sizeof(RgfxColorEntry);
	RgfxDisplayColor m = {};
	m.header.type = RGFX_CMD_DISPLAY_COLOR;
	m.header.size = size;
	m.display = display;
	m.transaction_id = transaction;
	m.stage = stage;
	m.entry_count = count;
	m.first_entry = first;
	m.part_entries = carried;
	memcpy(part, &m, sizeof(m));
	if (carried)
		memcpy(part + sizeof(m), entries + static_cast<uint64_t>(first) * kRgfxGammaEntrySize,
		       carried * sizeof(RgfxColorEntry));
	return size;
}

// What a submitted transaction sends the host: the IOSurface on its plane 0, a present that keeps
// the surface shown where it sets a colour and shows no surface (RGFX_PRESENT_KEEP), or nothing.
enum class RgfxPipeSends { Surface, Keep, Nothing };

inline RgfxPipeSends rgfxPipeSends(bool showsSurface, bool setsColor)
{
	if (showsSurface)
		return RgfxPipeSends::Surface;
	return setsColor ? RgfxPipeSends::Keep : RgfxPipeSends::Nothing;
}

// Whether a transaction that sends `sends` completes at the host's answer to it, once the display
// shows it: a surface's, and a colour's kept present, at the refresh that shows the colour. One
// that sends nothing completes with the transaction before it, or at once where none is
// outstanding (RgfxPipeTransactions.hpp).
inline bool rgfxPipeWaitsForAnswer(RgfxPipeSends sends)
{
	return sends != RgfxPipeSends::Nothing;
}

#endif // RGFX_PIPE_COLOR_HPP_
