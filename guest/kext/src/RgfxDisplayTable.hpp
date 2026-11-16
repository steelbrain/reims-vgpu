// RgfxDisplayTable: the device's displays as the framebuffers drive them, the code both front ends
// share (design.md, "Multiple displays"): how many displays the kext drives, each display's
// scanout room, the device's DISPLAYS answer read into a table of its displays, the display an
// event names, the displays a refresh is for, and the display a display pipe client's index names.
//
// A device has 1 to RGFX_DISPLAYS_MAX displays. One of two or more offers MULTI_DISPLAY; where the
// framebuffer took it, the device lists every display, ids 0 to count - 1, and sends each one's
// events, and the kext drives each with a framebuffer of its own. Where it did not (a device of
// one display, or a host that does not offer the bit), the kext drives display 0 alone, as before
// the bit: its one room is the whole scanout span, and every rule here answers as it did then.
//
// Header-only and free of the kernel's headers, so its native test (tests/display_table_test.cpp)
// builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_DISPLAY_TABLE_HPP_
#define RGFX_DISPLAY_TABLE_HPP_

#include <stdint.h>

#include "RgfxScanoutOwner.hpp"
#include "reims_graphics_wire.h"

// The displays the kext drives: every display the device lists where the framebuffer took
// MULTI_DISPLAY (`taken`), at most RGFX_DISPLAYS_MAX; else display 0 alone.
inline uint32_t rgfxDisplaysDriven(bool taken, uint32_t listed)
{
	if (!taken || listed < 1)
		return 1;
	return listed < RGFX_DISPLAYS_MAX ? listed : RGFX_DISPLAYS_MAX;
}

// Each display's scanout room on a device of `count` displays: RGFX_SCANOUT_SPAN / count, rounded
// down to a multiple of RGFX_SCANOUT_ROOM_ALIGN (the host's display::scanout_room): 128 MiB for
// one display, 64 MiB each for two, 0x2aa_0000 bytes each for three and 32 MiB each for four.
inline uint64_t rgfxScanoutRoomOf(uint32_t count)
{
	const uint64_t room = RGFX_SCANOUT_SPAN / (count ? count : 1);
	return room - room % RGFX_SCANOUT_ROOM_ALIGN;
}

// A display's room in the region: `length` bytes at `offset`.
struct RgfxRoom {
	uint64_t offset;
	uint64_t length;
};

// Display `display`'s scanout room on a device of `count` driven displays (rgfxDisplaysDriven), in
// a region of `regionSize` bytes whose last `cursorBytes` are the cursor's: display i's room
// starts at RGFX_ALLOCATIONS_OFFSET + i * rgfxScanoutRoomOf(count), as the host lays them out, and
// lies inside the scanout span the region holds (rgfxScanoutRoom). With one display it is that
// span itself, as it was before displays had rooms. A display past the count has no room.
inline RgfxRoom rgfxDisplayRoom(uint64_t regionSize, uint64_t cursorBytes, uint32_t display,
                                uint32_t count)
{
	const uint64_t whole = rgfxScanoutRoom(regionSize, cursorBytes);
	if (count <= 1)
		return {RGFX_ALLOCATIONS_OFFSET, display == 0 ? whole : 0};
	const uint64_t each = rgfxScanoutRoomOf(count);
	const uint64_t from = static_cast<uint64_t>(display) * each;
	if (display >= count || from >= whole)
		return {RGFX_ALLOCATIONS_OFFSET + from, 0};
	return {RGFX_ALLOCATIONS_OFFSET + from, whole - from < each ? whole - from : each};
}

// One display of a DISPLAYS answer: its record (RgfxDisplayEntry), its host screen's scale where
// the host took DISPLAY_SCALE (RgfxDisplayScale, 0 where none), and where its modes lie in the
// answer's payload (`modesAt`, the payload's offset of its first RgfxModeEntry; `modeCount` of
// them).
struct RgfxListedDisplay {
	uint32_t id;
	uint32_t connected;
	uint32_t preferred;
	uint32_t modeCount;
	uint32_t modesAt;
	uint32_t scale;
};

// A 32-bit field of an answer's payload, little-endian as both guests are.
inline uint32_t rgfxPayload32(const volatile uint8_t *payload, uint32_t offset)
{
	return *reinterpret_cast<const volatile uint32_t *>(payload + offset);
}

// Reads a DISPLAYS answer's payload, `size` bytes past its header: its count, then that many
// records, each followed by its scale where the host took DISPLAY_SCALE (`scaleTaken`) and then
// its modes. The first `max` displays go into `out`, all of them in *stored; *listed is the
// answer's count. False, the answer unreadable, where the payload cannot hold its count, or any
// record or mode list the count names, as the framebuffer has read it since its first display.
inline bool rgfxListDisplays(const volatile uint8_t *payload, uint32_t size, bool scaleTaken,
                             RgfxListedDisplay *out, uint32_t max, uint32_t *listed,
                             uint32_t *stored)
{
	*listed = 0;
	*stored = 0;
	if (size < sizeof(uint32_t))
		return false;
	const uint32_t count = rgfxPayload32(payload, 0);
	const uint32_t scaleBytes = scaleTaken ? sizeof(RgfxDisplayScale) : 0;
	// The first record follows the count, past the message header.
	uint32_t offset = RGFX_DISPLAYS_ENTRIES_OFFSET - sizeof(RgfxMsgHeader);
	for (uint32_t i = 0; i < count; i++) {
		if (size - offset < sizeof(RgfxDisplayEntry) + scaleBytes)
			return false;
		const uint32_t n = rgfxPayload32(payload, offset + 12);
		const uint32_t at = offset + sizeof(RgfxDisplayEntry) + scaleBytes;
		if (static_cast<uint64_t>(n) * sizeof(RgfxModeEntry) > size - at)
			return false;
		if (i < max) {
			RgfxListedDisplay &d = out[i];
			d.id = rgfxPayload32(payload, offset);
			d.connected = rgfxPayload32(payload, offset + 4);
			d.preferred = rgfxPayload32(payload, offset + 8);
			d.modeCount = n;
			d.modesAt = at;
			d.scale = scaleBytes ? rgfxPayload32(payload, offset + sizeof(RgfxDisplayEntry)) : 0;
		}
		offset = at + n * sizeof(RgfxModeEntry);
	}
	*listed = count;
	*stored = count < max ? count : max;
	return true;
}

// The display of `id` among the `n` listed, or -1 where none is.
inline int rgfxFindListed(const RgfxListedDisplay *listed, uint32_t n, uint32_t id)
{
	for (uint32_t i = 0; i < n; i++) {
		if (listed[i].id == id)
			return static_cast<int>(i);
	}
	return -1;
}

// The display a VBLANK or a DISPLAY_CHANGED names: its first payload field (RgfxVblank,
// RgfxDisplayChanged), 0 where the payload is too short to hold it.
inline uint32_t rgfxEventDisplay(const volatile uint8_t *payload, uint32_t size)
{
	return size >= sizeof(uint32_t) ? rgfxPayload32(payload, 0) : 0;
}

// The displays a refresh is for: a DISPLAY_CHANGED marks its display (changed), the work loop
// posts GET_DISPLAYS while one is marked (wanted, then posted with its seqno), and the answer to
// the latest refresh posted gives back every display marked for a refresh posted since the last
// answer (answered), whose connect interrupts then fire. An answer to an earlier refresh, or to
// another GET_DISPLAYS, is for none. With one display, that is display 0's bit, and a refresh as
// the framebuffer asked for it before displays were marked.
struct RgfxDisplayRefresh {
	uint32_t pending;
	uint32_t inFlight;
	uint64_t seqno;

	void changed(uint32_t display)
	{
		if (display < 32)
			pending |= 1u << display;
	}
	bool wanted() const { return pending != 0; }
	void posted(uint64_t s)
	{
		inFlight |= pending;
		pending = 0;
		seqno = s;
	}
	uint32_t answered(uint64_t s)
	{
		if (seqno == 0 || s != seqno)
			return 0;
		seqno = 0;
		const uint32_t displays = inFlight;
		inFlight = 0;
		return displays;
	}
};

// The index a display pipe client associates (RgfxDisplayPipeUserClient, selector 0) that names no
// framebuffer: IOPresentment's for a framebuffer it found without IOAccelIndex.
constexpr uint32_t kRgfxPipeAnyIndex = 0xffffffffu;

// The display a pipe client of `index` drives, of `count` driven: the index's own, or display 0 for
// kRgfxPipeAnyIndex; -1, refused, for an index past the displays.
inline int rgfxPipeDisplay(uint32_t index, uint32_t count)
{
	if (index == kRgfxPipeAnyIndex)
		return 0;
	return index < count ? static_cast<int>(index) : -1;
}

#endif // RGFX_DISPLAY_TABLE_HPP_
