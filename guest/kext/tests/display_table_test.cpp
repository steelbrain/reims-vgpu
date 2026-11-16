// The native test of RgfxDisplayTable.hpp: the displays the kext drives, their rooms, the DISPLAYS
// answer's table, the display of an event, the refreshes' displays and a pipe client's display.
#include "RgfxDisplayTable.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "display_table_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

// A DISPLAYS payload (past its header), as the device writes it.
struct Answer {
	std::vector<uint8_t> bytes;
	void u32(uint32_t v)
	{
		uint8_t b[4];
		std::memcpy(b, &v, 4);
		bytes.insert(bytes.end(), b, b + 4);
	}
	// A display: its record, its scale where `scaled`, then its modes (width, height, rate each).
	void display(uint32_t id, uint32_t connected, uint32_t preferred,
	             std::vector<RgfxModeEntry> modes, bool scaled, uint32_t scale)
	{
		u32(id);
		u32(connected);
		u32(preferred);
		u32(static_cast<uint32_t>(modes.size()));
		if (scaled) {
			u32(scale);
			u32(0);
		}
		for (const RgfxModeEntry &m : modes) {
			u32(m.width);
			u32(m.height);
			u32(m.refresh_mhz);
		}
	}
	const volatile uint8_t *data() const { return bytes.data(); }
	uint32_t size() const { return static_cast<uint32_t>(bytes.size()); }
};

uint32_t modeWidth(const Answer &a, const RgfxListedDisplay &d, uint32_t j)
{
	return rgfxPayload32(a.data(), d.modesAt + j * sizeof(RgfxModeEntry));
}
} // namespace

int main()
{
	// The kext drives every listed display only where it took MULTI_DISPLAY.
	check(rgfxDisplaysDriven(false, 1) == 1, "one display without the bit");
	check(rgfxDisplaysDriven(false, 4) == 1, "display 0 alone without the bit");
	check(rgfxDisplaysDriven(true, 2) == 2, "two displays with the bit");
	check(rgfxDisplaysDriven(true, 4) == 4, "four displays with the bit");
	check(rgfxDisplaysDriven(true, 9) == RGFX_DISPLAYS_MAX, "never past RGFX_DISPLAYS_MAX");
	check(rgfxDisplaysDriven(true, 0) == 1, "a device that lists none is display 0's");

	// The framebuffers of the displays driven: every one with a node, display 0's always.
	check(rgfxFramebuffersOf(1, RGFX_DISPLAYS_MAX) == 1, "one display, one framebuffer");
	check(rgfxFramebuffersOf(1, 0) == 1, "display 0's framebuffer with no node");
	check(rgfxFramebuffersOf(2, 1) == 1, "two driven, display 0's node alone: display 0's");
	check(rgfxFramebuffersOf(2, 2) == 2, "two driven, two nodes: two framebuffers");
	check(rgfxFramebuffersOf(4, 2) == 2, "four driven, two nodes: two framebuffers");
	check(rgfxFramebuffersOf(3, 4) == 3, "three driven, four nodes: three framebuffers");
	check(rgfxFramebuffersOf(4, RGFX_DISPLAYS_MAX) == 4, "x86-64: every display driven");
	check(RGFX_FEATURE_MULTI_DISPLAY == (1ull << 48), "MULTI_DISPLAY is bit 48");
	check(RGFX_DISPLAYS_MAX == 4, "four displays at most");

	// The rooms are the host's (display::scanout_room): the span split evenly, in 64 KiB steps.
	check(rgfxScanoutRoomOf(1) == RGFX_SCANOUT_SPAN, "one display's room is the span");
	check(rgfxScanoutRoomOf(2) == 64ull << 20, "two displays' rooms are 64 MiB");
	check(rgfxScanoutRoomOf(3) == 0x2aa0000ull, "three displays' rooms are 0x2aa_0000 bytes");
	check(rgfxScanoutRoomOf(4) == 32ull << 20, "four displays' rooms are 32 MiB");

	const uint64_t cursor = 256ull * 256 * 4;
	const uint64_t region = 1ull << 30;
	// One display: the room is the scanout span the region holds, as before rooms (rgfxScanoutRoom).
	for (uint64_t size : {region, RGFX_ALLOCATIONS_OFFSET + cursor + 0x10000ull,
	                      RGFX_ALLOCATIONS_OFFSET + cursor}) {
		const RgfxRoom r = rgfxDisplayRoom(size, cursor, 0, 1);
		check(r.offset == RGFX_ALLOCATIONS_OFFSET, "one display's room starts at the allocations");
		check(r.length == rgfxScanoutRoom(size, cursor), "one display's room is today's");
	}
	check(rgfxDisplayRoom(region, cursor, 1, 1).length == 0, "no display 1 on a device of one");
	// Several: display i's at the allocations plus i rooms, inside the span.
	for (uint32_t count = 2; count <= RGFX_DISPLAYS_MAX; count++) {
		for (uint32_t i = 0; i < count; i++) {
			const RgfxRoom r = rgfxDisplayRoom(region, cursor, i, count);
			check(r.offset == RGFX_ALLOCATIONS_OFFSET + i * rgfxScanoutRoomOf(count),
			      "display i's room starts i rooms in");
			check(r.length == rgfxScanoutRoomOf(count), "each room is the count's");
			check(r.offset + r.length <= RGFX_CONTEXT_ALLOCATIONS_OFFSET,
			      "every room ends before the accelerator's blocks");
		}
		check(rgfxDisplayRoom(region, cursor, count, count).length == 0,
		      "no room past the displays");
	}
	check(rgfxDisplayRoom(region, cursor, 3, 4).offset == RGFX_ALLOCATIONS_OFFSET + (96ull << 20),
	      "display 3 of 4 starts 96 MiB in");
	// A region too small for the whole span: the rooms past its end are cut or empty.
	{
		const uint64_t small = RGFX_ALLOCATIONS_OFFSET + cursor + (80ull << 20);
		check(rgfxDisplayRoom(small, cursor, 0, 2).length == 64ull << 20, "room 0 fits");
		check(rgfxDisplayRoom(small, cursor, 1, 2).length == 16ull << 20, "room 1 is cut");
		check(rgfxDisplayRoom(small, cursor, 2, 4).length == 16ull << 20, "room 2 of 4 is cut");
		check(rgfxDisplayRoom(small, cursor, 3, 4).length == 0, "room 3 of 4 is past the region");
	}

	// A one-display answer, as a device of one display or a guest without the bit gets it.
	RgfxListedDisplay table[RGFX_DISPLAYS_MAX];
	uint32_t listed = 0, stored = 0;
	{
		Answer a;
		a.u32(1);
		a.display(0, 1, 0, {{1920, 1080, 60000}, {1280, 720, 60000}}, false, 0);
		check(rgfxListDisplays(a.data(), a.size(), false, table, RGFX_DISPLAYS_MAX, &listed,
		                       &stored),
		      "one display reads");
		check(listed == 1 && stored == 1, "one display listed and stored");
		check(table[0].id == 0 && table[0].connected == 1 && table[0].preferred == 0 &&
		          table[0].modeCount == 2 && table[0].scale == 0,
		      "display 0's record");
		check(table[0].modesAt == RGFX_DISPLAYS_ENTRIES_OFFSET - sizeof(RgfxMsgHeader) +
		                              sizeof(RgfxDisplayEntry),
		      "display 0's modes follow its record");
		check(modeWidth(a, table[0], 1) == 1280, "display 0's second mode");
	}
	// Two displays with their scales, as a guest that took MULTI_DISPLAY and DISPLAY_SCALE gets it.
	{
		Answer a;
		a.u32(2);
		a.display(0, 1, 1, {{1920, 1080, 60000}, {3840, 2160, 60000}}, true, 240);
		a.display(1, 1, 0, {{2560, 1440, 120000}}, true, 120);
		check(rgfxListDisplays(a.data(), a.size(), true, table, RGFX_DISPLAYS_MAX, &listed,
		                       &stored),
		      "two displays read");
		check(listed == 2 && stored == 2, "two displays listed and stored");
		check(table[0].scale == 240 && table[0].preferred == 1, "display 0's scale and preference");
		check(table[1].id == 1 && table[1].modeCount == 1 && table[1].scale == 120,
		      "display 1's record and scale");
		check(modeWidth(a, table[1], 0) == 2560, "display 1's mode");
		check(rgfxFindListed(table, stored, 1) == 1, "display 1 is found by its id");
		check(rgfxFindListed(table, stored, 2) == -1, "display 2 is not listed");
		// A table of one keeps the first display, and checks the rest all the same.
		check(rgfxListDisplays(a.data(), a.size(), true, table, 1, &listed, &stored) &&
		          listed == 2 && stored == 1 && table[0].id == 0,
		      "a table of one keeps display 0");
	}
	// Unreadable answers.
	{
		Answer a;
		check(!rgfxListDisplays(a.data(), 0, false, table, RGFX_DISPLAYS_MAX, &listed, &stored),
		      "an answer with no count is unreadable");
		a.u32(2);
		a.display(0, 1, 0, {{1920, 1080, 60000}}, false, 0);
		check(!rgfxListDisplays(a.data(), a.size(), false, table, RGFX_DISPLAYS_MAX, &listed,
		                        &stored),
		      "a count past the records is unreadable");
		Answer b;
		b.u32(1);
		b.display(0, 1, 0, {{1920, 1080, 60000}, {1280, 720, 60000}}, false, 0);
		check(!rgfxListDisplays(b.data(), b.size() - 4, false, table, RGFX_DISPLAYS_MAX, &listed,
		                        &stored),
		      "a cut mode list is unreadable");
		Answer c;
		c.u32(0);
		check(rgfxListDisplays(c.data(), c.size(), false, table, RGFX_DISPLAYS_MAX, &listed,
		                       &stored) &&
		          listed == 0 && stored == 0,
		      "an answer of no display reads as none");
	}

	// An event's display: its first payload field.
	{
		uint8_t vblank[sizeof(RgfxVblank) - sizeof(RgfxMsgHeader)] = {};
		const uint32_t one = 1;
		std::memcpy(vblank, &one, 4);
		check(rgfxEventDisplay(vblank, sizeof(vblank)) == 1, "a VBLANK of display 1");
		check(rgfxEventDisplay(vblank, 2) == 0, "a payload too short names display 0");
		check(__builtin_offsetof(RgfxVblank, display) == sizeof(RgfxMsgHeader) &&
		          __builtin_offsetof(RgfxDisplayChanged, display) == sizeof(RgfxMsgHeader),
		      "both events name their display first");
	}

	// The refreshes' displays.
	{
		RgfxDisplayRefresh r = {};
		check(!r.wanted(), "no refresh at first");
		check(r.answered(0) == 0 && r.answered(5) == 0, "no answer is a refresh's at first");
		r.changed(0);
		check(r.wanted(), "a change of display 0 wants a refresh");
		r.posted(7);
		check(!r.wanted(), "posted, it is wanted no more");
		check(r.answered(6) == 0, "another answer is not the refresh's");
		check(r.answered(7) == 1, "the refresh's answer is display 0's");
		check(r.answered(7) == 0, "answered once");
		// A change while a refresh is in flight: a second refresh, whose answer gives both.
		r.changed(1);
		r.posted(8);
		r.changed(3);
		check(r.wanted(), "a change during a refresh wants another");
		r.posted(9);
		check(r.answered(8) == 0, "the earlier refresh's answer is for none");
		check(r.answered(9) == ((1u << 1) | (1u << 3)), "the latest gives every display marked");
		r.changed(40);
		check(!r.wanted(), "a display past the mask is never marked");
	}

	// A display pipe client's display.
	check(rgfxPipeDisplay(0, 1) == 0, "index 0 is display 0");
	check(rgfxPipeDisplay(kRgfxPipeAnyIndex, 1) == 0, "the index of no framebuffer is display 0");
	check(rgfxPipeDisplay(1, 1) == -1, "index 1 is refused with one display");
	check(rgfxPipeDisplay(1, 2) == 1, "index 1 is display 1 of two");
	check(rgfxPipeDisplay(3, 4) == 3 && rgfxPipeDisplay(4, 4) == -1, "indices past the count");

	std::printf("display_table_test: %d checks passed\n", checks);
	return 0;
}
