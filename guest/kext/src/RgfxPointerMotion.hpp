// RgfxPointerMotion: the guest's own relative pointing device, a mouse, which the host's
// POINTER_MOTION moves and clicks (design.md, "A pointer for games"; RgfxPointerDevice.hpp
// publishes it). The host's pointing device reaches the guest through QEMU's USB tablet, an
// absolute pointer, which holds at the image's edge: a game that hides the cursor and turns its
// camera by the mouse's motion gets none past it. While a game hides the cursor, the host's window
// sends its pointer's motion here instead, in the host device's own counts, unaccelerated, and
// this device hands it to macOS as a USB mouse's reports, so macOS accelerates it once, as its
// settings say, and a game that reads the device's counts (GCMouse, IOHIDManager) gets the host's.
//
// The device is a HID mouse of five buttons, 16-bit relative X and Y and an 8-bit wheel, in one
// input report of 6 bytes and no report ID: the buttons in bits 0 to 4 of byte 0 (left, right,
// middle, then the side buttons, the order of RGFX_POINTER_BUTTON_*), X and Y little-endian in
// bytes 1 to 4, the wheel in byte 5, above 0 up. It declares no resolution, as most USB mice
// declare none. A POINTER_MOTION past what one report holds goes as several, each with the
// buttons it leaves held: the host sends a press or a release in an event of its own, which the
// device joins to no motion of other buttons, so the guest takes it where the host made it.
//
// The x86-64 kext's mouse scrolls sideways too: an 8-bit AC Pan (consumer page 0x0C, usage
// 0x238) in byte 6, above 0 right, so 7 bytes a report, which IOHIDEventDriver makes the scroll
// event's sideways axis. Its SCROLL's notches go to it (design.md, "Scrolling"), and a SCROLL's
// distance to the precise scroller below. The arm64e kext's mouse has no pan and it has no
// scroller: its guest's tablet scrolls.
//
// Header-only and free of the kernel's headers, so its native test (tests/pointer_motion_test.cpp)
// builds it as the kext does.
//
// Our own code; it depends only on the committed wire header.
#ifndef RGFX_POINTER_MOTION_HPP_
#define RGFX_POINTER_MOTION_HPP_

#include <stddef.h>
#include <stdint.h>

#include "RgfxDisplayTable.hpp"
#include "reims_graphics_wire.h"

#ifndef RGFX_GUEST_ARM64
#define RGFX_GUEST_ARM64 0
#endif

// The device's HID report descriptor.
constexpr uint8_t kRgfxMouseReportDescriptor[] = {
    0x05, 0x01,       // Usage Page (Generic Desktop)
    0x09, 0x02,       // Usage (Mouse)
    0xa1, 0x01,       // Collection (Application)
    0x09, 0x01,       //   Usage (Pointer)
    0xa1, 0x00,       //   Collection (Physical)
    0x05, 0x09,       //     Usage Page (Button)
    0x19, 0x01,       //     Usage Minimum (1)
    0x29, 0x05,       //     Usage Maximum (5)
    0x15, 0x00,       //     Logical Minimum (0)
    0x25, 0x01,       //     Logical Maximum (1)
    0x95, 0x05,       //     Report Count (5)
    0x75, 0x01,       //     Report Size (1)
    0x81, 0x02,       //     Input (Data, Variable, Absolute)
    0x95, 0x01,       //     Report Count (1)
    0x75, 0x03,       //     Report Size (3)
    0x81, 0x01,       //     Input (Constant)
    0x05, 0x01,       //     Usage Page (Generic Desktop)
    0x09, 0x30,       //     Usage (X)
    0x09, 0x31,       //     Usage (Y)
    0x16, 0x01, 0x80, //     Logical Minimum (-32767)
    0x26, 0xff, 0x7f, //     Logical Maximum (32767)
    0x75, 0x10,       //     Report Size (16)
    0x95, 0x02,       //     Report Count (2)
    0x81, 0x06,       //     Input (Data, Variable, Relative)
    0x09, 0x38,       //     Usage (Wheel)
    0x15, 0x81,       //     Logical Minimum (-127)
    0x25, 0x7f,       //     Logical Maximum (127)
    0x75, 0x08,       //     Report Size (8)
    0x95, 0x01,       //     Report Count (1)
    0x81, 0x06,       //     Input (Data, Variable, Relative)
#if !RGFX_GUEST_ARM64
    0x05, 0x0c,       //     Usage Page (Consumer)
    0x0a, 0x38, 0x02, //     Usage (AC Pan)
    0x15, 0x81,       //     Logical Minimum (-127)
    0x25, 0x7f,       //     Logical Maximum (127)
    0x75, 0x08,       //     Report Size (8)
    0x95, 0x01,       //     Report Count (1)
    0x81, 0x06,       //     Input (Data, Variable, Relative)
#endif
    0xc0,             //   End Collection
    0xc0,             // End Collection
};

// The size of one report.
constexpr size_t kRgfxMouseReportBytes = RGFX_GUEST_ARM64 ? 6 : 7;

// The HID transport the mouse names (RgfxPointerDevice's newTransportString): "Virtual",
// IOHIDKeys.h's kIOHIDTransportVirtualValue, on arm64, where display 0's framebuffer, the mouse's
// provider, carries IOMobileFramebuffer's own "Transport", a dictionary, which a lookup of the
// mouse's transport through its parents finds where the mouse names none (RgfxPointerDevice.hpp);
// none on x86-64, whose framebuffer has no "Transport".
constexpr const char *kRgfxMouseTransport = RGFX_GUEST_ARM64 ? "Virtual" : nullptr;

// The most one report moves on an axis, and turns the wheel (or pans).
constexpr int32_t kRgfxMouseReportMotionMax = 32767;
constexpr int32_t kRgfxMouseReportWheelMax = 127;

// The most reports one POINTER_MOTION or SCROLL makes: a motion of i32's range on an axis takes
// 65538 of 32767, more than any host sends; the device sends at most this many, the motion past
// them dropped, as are notches past 8 x 127 either way.
constexpr uint32_t kRgfxMouseReportsMax = 8;

#if !RGFX_GUEST_ARM64
// The x86-64 kext's precise scroller (RgfxPointerDevice.hpp), which takes a SCROLL's distance, a
// touchpad's: the mouse's descriptor and reports, its buttons and motion always 0, with a physical
// range of -1 to 1 on its wheel and pan against their logical -127 to 127. Apple's
// IOHIDEventService::determineResolution reads that as 127 counts a unit (IOHIDEventService.cpp:701
// and :998, IOHIDFamily-2238.100.59) and publishes it as HIDScrollResolution, and at that macOS
// marks the scroller's events continuous and precise, in points, as a touchpad's (measured on
// x86-64 26.7.2 and 11.7.11, potluck jobs m3-max-75987 and m3-max-75994).
constexpr uint8_t kRgfxScrollerReportDescriptor[] = {
    0x05, 0x01, 0x09, 0x02, 0xa1, 0x01, 0x09, 0x01, 0xa1, 0x00, // the mouse's collections,
    0x05, 0x09, 0x19, 0x01, 0x29, 0x05, 0x15, 0x00, 0x25, 0x01, // buttons,
    0x95, 0x05, 0x75, 0x01, 0x81, 0x02, 0x95, 0x01, 0x75, 0x03, 0x81, 0x01,
    0x05, 0x01, 0x09, 0x30, 0x09, 0x31, 0x16, 0x01, 0x80, 0x26, 0xff, 0x7f, // X and Y
    0x75, 0x10, 0x95, 0x02, 0x81, 0x06,
    0x09, 0x38, 0x15, 0x81, 0x25, 0x7f, // Wheel, logical -127 to 127
    0x35, 0xff,                         // Physical Minimum (-1)
    0x45, 0x01,                         // Physical Maximum (1)
    0x75, 0x08, 0x95, 0x01, 0x81, 0x06,
    0x05, 0x0c, 0x0a, 0x38, 0x02, 0x15, 0x81, 0x25, 0x7f, // AC Pan, logical -127 to 127
    0x75, 0x08, 0x95, 0x01, 0x81, 0x06,
    0x35, 0x00, // Physical Minimum (0)
    0x45, 0x00, // Physical Maximum (0)
    0xc0, 0xc0,
};

// The 1/256 points of a SCROLL's distance one of the scroller's counts stands for. With its
// acceleration off (RgfxPointerDevice.cpp), a count scrolls 10 points, however fast the counts
// come: 120 counts made 1200 points slow, fast and in lumps, on each axis, at 127 and at 90 counts
// a unit (the jobs above). No descriptor makes a count worth less: Apple's driver passes each
// count whole, getValue() << 16 (IOHIDEventDriver.cpp:4300 and :4312), and reads no resolution
// multiplier.
constexpr int64_t kRgfxScrollerFinePerCount = 10 * 256;

// The scroller's counts on one axis for `fine` more of a SCROLL's distance, the rest kept in
// `*rest`: the distance joins the rest, which starts afresh where the scrolling turns back, and
// a count goes for each whole 10 points, toward zero. Counts past what one SCROLL's reports carry
// (8 x 127 either way) are dropped, as a mouse's notches are.
inline int32_t rgfxScrollerCounts(int64_t *rest, int32_t fine)
{
	if ((*rest > 0 && fine < 0) || (*rest < 0 && fine > 0))
		*rest = 0;
	const int64_t total = *rest + fine;
	const int64_t counts = total / kRgfxScrollerFinePerCount;
	*rest = total - counts * kRgfxScrollerFinePerCount;
	const int64_t most = int64_t(kRgfxMouseReportsMax) * kRgfxMouseReportWheelMax;
	return static_cast<int32_t>(counts > most ? most : (counts < -most ? -most : counts));
}
#endif

// POINTER_MOTION's motion, wheel and buttons (RgfxPointerMotion past its header), and on x86-64 a
// SCROLL's sideways notches, above 0 right.
struct RgfxMotion {
	int32_t dx;
	int32_t dy;
	int32_t wheel;
	uint32_t buttons;
#if !RGFX_GUEST_ARM64
	int32_t pan;
#endif
};

// SCROLL's notches and distance (RgfxScroll past its header): up and right, the distance in
// 1/256 point.
struct RgfxScrolled {
	int32_t notchesUp;
	int32_t notchesRight;
	int32_t fineUp;
	int32_t fineRight;
};

// The motion, wheel and buttons of a POINTER_MOTION whose payload (past its header) is `size`
// bytes at `payload`: false where it is too short to hold them.
inline bool rgfxPointerMotionRead(const volatile uint8_t *payload, uint32_t size, RgfxMotion *motion)
{
	if (size < 4 * sizeof(uint32_t))
		return false;
	motion->dx = static_cast<int32_t>(rgfxPayload32(payload, 0));
	motion->dy = static_cast<int32_t>(rgfxPayload32(payload, 4));
	motion->wheel = static_cast<int32_t>(rgfxPayload32(payload, 8));
	motion->buttons = rgfxPayload32(payload, 12);
	return true;
}

// The notches and distance of a SCROLL whose payload (past its header) is `size` bytes at
// `payload`: false where it is too short to hold them.
inline bool rgfxScrollRead(const volatile uint8_t *payload, uint32_t size, RgfxScrolled *scroll)
{
	if (size < 4 * sizeof(uint32_t))
		return false;
	scroll->notchesUp = static_cast<int32_t>(rgfxPayload32(payload, 0));
	scroll->notchesRight = static_cast<int32_t>(rgfxPayload32(payload, 4));
	scroll->fineUp = static_cast<int32_t>(rgfxPayload32(payload, 8));
	scroll->fineRight = static_cast<int32_t>(rgfxPayload32(payload, 12));
	return true;
}

namespace rgfx_pointer_motion_detail {
// The part of `*left` one report carries, at most `max` either way, taken off `*left`.
inline int32_t take(int32_t *left, int32_t max)
{
	const int32_t part = *left > max ? max : (*left < -max ? -max : *left);
	*left -= part;
	return part;
}
} // namespace rgfx_pointer_motion_detail

// The reports that carry `motion`, written to `reports` (kRgfxMouseReportBytes each, room for
// kRgfxMouseReportsMax): how many. At least one, so a POINTER_MOTION that only presses or lets go
// of a button is a report; every report holds its buttons, as the buttons stay held through it.
inline uint32_t rgfxMouseReports(const RgfxMotion &motion,
                                 uint8_t reports[kRgfxMouseReportsMax][kRgfxMouseReportBytes])
{
	using rgfx_pointer_motion_detail::take;
	int32_t dx = motion.dx, dy = motion.dy, wheel = motion.wheel;
#if !RGFX_GUEST_ARM64
	int32_t pan = motion.pan;
#endif
	const uint8_t buttons = static_cast<uint8_t>(motion.buttons & 0x1f);
	uint32_t n = 0;
	do {
		const int32_t x = take(&dx, kRgfxMouseReportMotionMax);
		const int32_t y = take(&dy, kRgfxMouseReportMotionMax);
		const int32_t w = take(&wheel, kRgfxMouseReportWheelMax);
		uint8_t *r = reports[n++];
		r[0] = buttons;
		r[1] = static_cast<uint8_t>(static_cast<uint16_t>(x) & 0xff);
		r[2] = static_cast<uint8_t>(static_cast<uint16_t>(x) >> 8);
		r[3] = static_cast<uint8_t>(static_cast<uint16_t>(y) & 0xff);
		r[4] = static_cast<uint8_t>(static_cast<uint16_t>(y) >> 8);
		r[5] = static_cast<uint8_t>(static_cast<int8_t>(w));
#if !RGFX_GUEST_ARM64
		r[6] = static_cast<uint8_t>(static_cast<int8_t>(take(&pan, kRgfxMouseReportWheelMax)));
	} while ((dx || dy || wheel || pan) && n < kRgfxMouseReportsMax);
#else
	} while ((dx || dy || wheel) && n < kRgfxMouseReportsMax);
#endif
	return n;
}

#endif // RGFX_POINTER_MOTION_HPP_
