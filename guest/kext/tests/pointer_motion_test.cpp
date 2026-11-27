// The native test of RgfxPointerMotion.hpp: a POINTER_MOTION's motion, wheel and buttons, the
// reports that carry it (one for a press alone, several for motion past one report's range, each
// with its buttons), and the report descriptor's shape: a mouse of five buttons, relative 16-bit
// X and Y and an 8-bit wheel, six bytes a report, and as the x86-64 kext builds it an 8-bit AC
// Pan after the wheel, seven bytes, which a SCROLL's notches turn, and the precise scroller's
// descriptor and counts, which a SCROLL's distance turns (pointer_motion_arm64_test builds it as
// the arm64e kext does, which has neither), and the HID transport the mouse names.
#include "RgfxPointerMotion.hpp"

#include <IOKit/hid/IOHIDKeys.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "pointer_motion_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

int16_t axis(const uint8_t *r, int at)
{
	return static_cast<int16_t>(static_cast<uint16_t>(r[at] | (r[at + 1] << 8)));
}

// A POINTER_MOTION's motion, wheel and buttons, with no pan.
RgfxMotion motion(int32_t dx, int32_t dy, int32_t wheel, uint32_t buttons)
{
	RgfxMotion m = {};
	m.dx = dx;
	m.dy = dy;
	m.wheel = wheel;
	m.buttons = buttons;
	return m;
}
} // namespace

int main()
{
	// The event's layout: the motion, the wheel, then the buttons, past its 16-byte header.
	{
		RgfxPointerMotion event = {};
		event.header.type = RGFX_EVT_POINTER_MOTION;
		event.dx = -300;
		event.dy = 7;
		event.wheel = -2;
		event.buttons = RGFX_POINTER_BUTTON_RIGHT;
		uint8_t bytes[sizeof(event)];
		std::memcpy(bytes, &event, sizeof(event));
		RgfxMotion m = {};
		check(rgfxPointerMotionRead(bytes + sizeof(RgfxMsgHeader), sizeof(event) - sizeof(RgfxMsgHeader), &m),
		      "a whole POINTER_MOTION reads");
		check(m.dx == -300 && m.dy == 7 && m.wheel == -2 && m.buttons == RGFX_POINTER_BUTTON_RIGHT,
		      "its fields in order");
		check(!rgfxPointerMotionRead(bytes + 16, 12, &m), "a short one reads nothing");
	}
	uint8_t r[kRgfxMouseReportsMax][kRgfxMouseReportBytes];
	// A press alone is one report of no motion.
	{
		const uint32_t n = rgfxMouseReports(motion(0, 0, 0, RGFX_POINTER_BUTTON_LEFT), r);
		check(n == 1, "a press alone is one report");
		check(r[0][0] == 1 && axis(r[0], 1) == 0 && axis(r[0], 3) == 0 && r[0][5] == 0,
		      "the left button, no motion");
	}
	// Motion and the wheel in range: one report, little-endian, the wheel signed.
	{
		const uint32_t n = rgfxMouseReports(motion(-5, 1234, -3, RGFX_POINTER_BUTTON_EXTRA), r);
		check(n == 1, "motion in range is one report");
		check(r[0][0] == 0x10, "the extra button is HID button 5");
		check(axis(r[0], 1) == -5 && axis(r[0], 3) == 1234, "X and Y");
		check(static_cast<int8_t>(r[0][5]) == -3, "the wheel, down");
	}
	// Every button maps to its HID button, nothing past the five.
	{
		rgfxMouseReports(motion(0, 0, 0, 0xffffffffu), r);
		check(r[0][0] == 0x1f, "five buttons and no more");
	}
	// Motion past a report's range goes as several, each with the buttons, the sum the motion.
	{
		const uint32_t n = rgfxMouseReports(motion(70000, -40000, 300, RGFX_POINTER_BUTTON_LEFT), r);
		check(n == 3, "70000 across takes three reports");
		long x = 0, y = 0, w = 0;
		for (uint32_t i = 0; i < n; i++) {
			check(r[i][0] == 1, "each report holds the button");
			x += axis(r[i], 1);
			y += axis(r[i], 3);
			w += static_cast<int8_t>(r[i][5]);
		}
		check(x == 70000 && y == -40000 && w == 300, "the reports sum to the motion");
		check(axis(r[0], 1) == 32767 && axis(r[0], 3) == -32767, "each axis's first is its most");
	}
	// Past every report the device sends, the rest is dropped.
	{
		const uint32_t n = rgfxMouseReports(motion(INT32_MAX, 0, 0, 0), r);
		check(n == kRgfxMouseReportsMax, "at most the reports the device sends");
	}
	// A SCROLL's layout: notches up and right, then the distance, past its header.
	{
		RgfxScroll event = {};
		event.notches_up = -2;
		event.notches_right = 3;
		event.fine_up = 256;
		event.fine_right = -512;
		uint8_t bytes[sizeof(event)];
		std::memcpy(bytes, &event, sizeof(event));
		RgfxScrolled s = {};
		check(rgfxScrollRead(bytes + 16, sizeof(event) - 16, &s), "a whole SCROLL reads");
		check(s.notchesUp == -2 && s.notchesRight == 3 && s.fineUp == 256 && s.fineRight == -512,
		      "its fields in order");
		check(!rgfxScrollRead(bytes + 16, 12, &s), "a short one reads nothing");
	}
#if !RGFX_GUEST_ARM64
	// Notches up and right in one report, the pan signed in byte 6, the buttons held.
	{
		RgfxMotion m = {};
		m.wheel = 2;
		m.pan = -3;
		m.buttons = RGFX_POINTER_BUTTON_LEFT;
		check(rgfxMouseReports(m, r) == 1, "notches in range are one report");
		check(r[0][0] == 1 && static_cast<int8_t>(r[0][5]) == 2 && static_cast<int8_t>(r[0][6]) == -3,
		      "the button, the wheel up, the pan left");
	}
	// The extremes: notches of i32's range either way fill every report the device sends at 127
	// each, and overrun no field.
	{
		RgfxMotion m = {};
		m.wheel = INT32_MIN;
		m.pan = INT32_MAX;
		const uint32_t n = rgfxMouseReports(m, r);
		check(n == kRgfxMouseReportsMax, "every report");
		for (uint32_t i = 0; i < n; i++)
			check(static_cast<int8_t>(r[i][5]) == -127 && static_cast<int8_t>(r[i][6]) == 127 &&
			          axis(r[i], 1) == 0 && axis(r[i], 3) == 0,
			      "127 notches a report each way, no motion");
	}
#endif
	// The descriptor: a mouse's application collection, its input report's fields.
	{
		check(kRgfxMouseReportDescriptor[0] == 0x05 && kRgfxMouseReportDescriptor[1] == 0x01 &&
		          kRgfxMouseReportDescriptor[2] == 0x09 && kRgfxMouseReportDescriptor[3] == 0x02,
		      "Generic Desktop, Mouse");
		// The input items' bits: 5 + 3 + 2 x 16 + 8 = 48, six bytes.
		int bits = 0, size = 0, count = 0;
		for (size_t i = 0; i < sizeof(kRgfxMouseReportDescriptor);) {
			const uint8_t item = kRgfxMouseReportDescriptor[i];
			const int len = (item & 3) == 3 ? 4 : (item & 3);
			int value = 0;
			for (int b = 0; b < len; b++)
				value |= kRgfxMouseReportDescriptor[i + 1 + b] << (8 * b);
			if ((item & 0xfc) == 0x74)
				size = value;
			else if ((item & 0xfc) == 0x94)
				count = value;
			else if ((item & 0xfc) == 0x80)
				bits += size * count;
			i += 1 + len;
		}
		check(bits == 8 * static_cast<int>(kRgfxMouseReportBytes), "the report's bytes");
		check(kRgfxMouseReportBytes == (RGFX_GUEST_ARM64 ? 6 : 7), "six bytes, seven with the pan");
		bool pan = false;
		for (size_t i = 0; i + 2 < sizeof(kRgfxMouseReportDescriptor); i++)
			pan |= kRgfxMouseReportDescriptor[i] == 0x0a && kRgfxMouseReportDescriptor[i + 1] == 0x38 &&
			       kRgfxMouseReportDescriptor[i + 2] == 0x02;
		check(pan == !RGFX_GUEST_ARM64, "AC Pan on x86-64 alone");
		check(sizeof(kRgfxMouseReportDescriptor) == (RGFX_GUEST_ARM64 ? 64u : 79u),
		      "the arm64e descriptor's bytes as before");
	}
#if !RGFX_GUEST_ARM64
	// The scroller's descriptor is the mouse's with a physical range of -1 to 1 from the wheel on,
	// set after the wheel's logical range and cleared before the collections end.
	{
		uint8_t want[sizeof(kRgfxMouseReportDescriptor) + 8];
		size_t n = 0;
		const uint8_t wheel[] = {0x09, 0x38, 0x15, 0x81, 0x25, 0x7f};
		for (size_t i = 0; i < sizeof(kRgfxMouseReportDescriptor); i++) {
			if (i == sizeof(kRgfxMouseReportDescriptor) - 2) {
				const uint8_t clear[] = {0x35, 0x00, 0x45, 0x00};
				std::memcpy(want + n, clear, sizeof(clear));
				n += sizeof(clear);
			}
			want[n++] = kRgfxMouseReportDescriptor[i];
			if (i >= sizeof(wheel) - 1 &&
			    !std::memcmp(kRgfxMouseReportDescriptor + i + 1 - sizeof(wheel), wheel, sizeof(wheel))) {
				const uint8_t set[] = {0x35, 0xff, 0x45, 0x01};
				std::memcpy(want + n, set, sizeof(set));
				n += sizeof(set);
			}
		}
		check(n == sizeof(kRgfxScrollerReportDescriptor) &&
		          !std::memcmp(want, kRgfxScrollerReportDescriptor, n),
		      "the scroller's descriptor is the mouse's at 127 counts a unit");
	}
	// The scroller's counts: one for each whole 10 points, toward zero, the rest carried on and
	// started afresh where the scrolling turns back.
	{
		int64_t rest = 0;
		check(rgfxScrollerCounts(&rest, 2560) == 1 && rest == 0, "10 points are a count");
		check(rgfxScrollerCounts(&rest, 1280) == 0 && rest == 1280, "5 points wait");
		check(rgfxScrollerCounts(&rest, 1280) == 1 && rest == 0, "and make one with 5 more");
		check(rgfxScrollerCounts(&rest, -3520) == -1 && rest == -960, "down, toward zero");
		check(rgfxScrollerCounts(&rest, 2000) == 0 && rest == 2000, "a turn starts afresh");
		check(rgfxScrollerCounts(&rest, 0) == 0 && rest == 2000, "nothing keeps the rest");
	}
	// The extremes: i32's range either way makes at most what one SCROLL's reports carry, 8 x 127,
	// the rest past whole counts kept, and a rest beside it overflows nothing.
	{
		int64_t rest = 2559;
		check(rgfxScrollerCounts(&rest, INT32_MAX) == 1016 &&
		          rest == (int64_t(INT32_MAX) + 2559) % 2560,
		      "up, at most 1016 counts");
		rest = -2559;
		check(rgfxScrollerCounts(&rest, INT32_MIN) == -1016 &&
		          rest == (int64_t(INT32_MIN) - 2559) % 2560,
		      "down, at most 1016 counts");
		RgfxMotion m = {};
		m.wheel = 1016;
		m.pan = -1016;
		check(rgfxMouseReports(m, r) == kRgfxMouseReportsMax &&
		          static_cast<int8_t>(r[7][5]) == 127 && static_cast<int8_t>(r[7][6]) == -127,
		      "which every report carries");
	}
#endif
	// The transport the mouse names: on arm64 IOHIDKeys.h's Virtual, a string, so a lookup of it
	// through the mouse's parents stops before display 0's framebuffer and its dictionary of the
	// same key (a desktop-killing exception in macOS 26.6.2's WindowServer, m3-max-35949); none on
	// x86-64, whose framebuffer has none.
	if (RGFX_GUEST_ARM64)
		check(kRgfxMouseTransport != nullptr &&
		          std::strcmp(kRgfxMouseTransport, kIOHIDTransportVirtualValue) == 0,
		      "the arm64 mouse names the Virtual transport");
	else
		check(kRgfxMouseTransport == nullptr, "the x86-64 mouse names no transport");
	std::printf("pointer_motion_test: %d checks passed\n", checks);
	return 0;
}
