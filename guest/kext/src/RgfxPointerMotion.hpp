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
    0xc0,             //   End Collection
    0xc0,             // End Collection
};

// The size of one report.
constexpr size_t kRgfxMouseReportBytes = 6;

// The most one report moves on an axis, and turns the wheel.
constexpr int32_t kRgfxMouseReportMotionMax = 32767;
constexpr int32_t kRgfxMouseReportWheelMax = 127;

// The most reports one POINTER_MOTION makes: a motion of i32's range on an axis takes 65538 of
// 32767, more than any host sends; the device sends at most this many, the motion past them
// dropped.
constexpr uint32_t kRgfxMouseReportsMax = 8;

// POINTER_MOTION's motion, wheel and buttons (RgfxPointerMotion past its header).
struct RgfxMotion {
	int32_t dx;
	int32_t dy;
	int32_t wheel;
	uint32_t buttons;
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
	} while ((dx || dy || wheel) && n < kRgfxMouseReportsMax);
	return n;
}

#endif // RGFX_POINTER_MOTION_HPP_
