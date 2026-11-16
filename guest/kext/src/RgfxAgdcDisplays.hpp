// RgfxAgdcDisplays: the device control's answers about the displays (RgfxDeviceControl.hpp): its
// attribute 0x980, which lists the displays the framebuffers drive, and the display a request of
// attributes 0x925, 0x711 and 0x986 names.
//
// IOPresentment builds a framebuffer's display pipe only where 0x980 lists the framebuffer
// (macOS 26.7.1's IOPresentment, ioPresentmentPopulateIOFramebufferEntries, read): it reads the
// count at 0x30 (the fifth count) and that many framebuffers from 0x3c, 8 bytes each, by their
// index, the index each display's pipe clients then associate (RgfxDisplayTable.hpp's
// rgfxPipeDisplay). CoreDisplay shows nothing on a framebuffer without a pipe ("Unable to find
// display pipe for IOFB service", measured on 26.7.2 with two displays and one listed). With one
// display the answer is the one display's, as before the device had several.
//
// Header-only and free of the kernel's headers, so its native test (tests/agdc_displays_test.cpp)
// builds it as the kext does.
//
// Our own code.
#ifndef RGFX_AGDC_DISPLAYS_HPP_
#define RGFX_AGDC_DISPLAYS_HPP_

#include <stdint.h>

// Attribute 0x980, IOPresentment's "AGDC GPU capabilities": four masks of the display ports (bit
// 1 + i for display i), the display count five times, the control's provider, and each display's
// framebuffer, by its index. The pointers name the services; the AGDC library's user client hands
// IOPresentment their registry IDs.
struct __attribute__((packed)) RgfxAgdcDisplays {
	uint64_t field_00[4];
	uint32_t field_20[5];
	uint64_t field_34;
	uint64_t field_3c[20];
};
static_assert(sizeof(RgfxAgdcDisplays) == 0xdc, "attribute 0x980 is 0xdc bytes");
static_assert(__builtin_offsetof(RgfxAgdcDisplays, field_20) + 4 * sizeof(uint32_t) == 0x30,
              "IOPresentment reads the fifth count");
static_assert(__builtin_offsetof(RgfxAgdcDisplays, field_3c) == 0x3c,
              "IOPresentment reads the framebuffers from 0x3c");

// The most displays 0x980 can list.
constexpr uint32_t kRgfxAgdcDisplaysMax = sizeof(RgfxAgdcDisplays::field_3c) / sizeof(uint64_t);

// Attribute 0x980 for `count` displays (1 where the framebuffers have not read the device's
// displays yet), the control's `provider`, and display i's framebuffer at `framebuffers[i]` (0
// where none drives it yet).
inline RgfxAgdcDisplays rgfxAgdcDisplays(uint32_t count, uint64_t provider,
                                         const uint64_t *framebuffers)
{
	if (count < 1)
		count = 1;
	if (count > kRgfxAgdcDisplaysMax)
		count = kRgfxAgdcDisplaysMax;
	RgfxAgdcDisplays d = {};
	const uint64_t ports = ((1ull << count) - 1) << 1;
	for (uint64_t &mask : d.field_00)
		mask = ports;
	for (uint32_t &n : d.field_20)
		n = count;
	d.field_34 = provider;
	for (uint32_t i = 0; i < count; i++)
		d.field_3c[i] = framebuffers[i];
	return d;
}

// How a request names its display: 0x925's and 0x986's by its index plus 1, 0x711's by its index.
constexpr uint32_t kRgfxAgdcNamedFromOne = 1;
constexpr uint32_t kRgfxAgdcNamedFromZero = 0;

// The display a request names (`named`, counted from `first`) of `count` driven, or -1 where it
// names none: with one display, 0x925 and 0x986 name display 0 as 1 and 0x711 as 0, as before.
inline int rgfxAgdcDisplay(uint32_t named, uint32_t first, uint32_t count)
{
	if (count < 1)
		count = 1;
	if (named < first || named - first >= count)
		return -1;
	return static_cast<int>(named - first);
}

#endif // RGFX_AGDC_DISPLAYS_HPP_
