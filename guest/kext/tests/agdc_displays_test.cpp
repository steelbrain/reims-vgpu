// The native test of RgfxAgdcDisplays.hpp: the device control's attribute 0x980 and the display
// its requests name, with one display as before the device had several, and with two to four.
#include "RgfxAgdcDisplays.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "agdc_displays_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

// The answer as the control made it for its one display, before displays were counted: the
// mask of port 1, the count 1, the provider and the one framebuffer at index 0.
RgfxAgdcDisplays oneDisplayAnswer(uint64_t provider, uint64_t framebuffer)
{
	RgfxAgdcDisplays d = {};
	const uint32_t kDisplayCount = 1, kDisplayIndex = 0;
	uint64_t ports = ((1ull << kDisplayCount) - 1) << 1;
	for (auto &mask : d.field_00)
		mask = ports;
	for (auto &count : d.field_20)
		count = kDisplayCount;
	d.field_34 = provider;
	if (framebuffer)
		d.field_3c[kDisplayIndex] = framebuffer;
	return d;
}

uint32_t u32At(const RgfxAgdcDisplays &d, size_t offset)
{
	uint32_t v;
	std::memcpy(&v, reinterpret_cast<const uint8_t *>(&d) + offset, sizeof v);
	return v;
}

uint64_t u64At(const RgfxAgdcDisplays &d, size_t offset)
{
	uint64_t v;
	std::memcpy(&v, reinterpret_cast<const uint8_t *>(&d) + offset, sizeof v);
	return v;
}
} // namespace

int main()
{
	const uint64_t provider = 0xffffff8012345000ull;
	const uint64_t fbs[4] = {0xffffff8020000000ull, 0xffffff8020001000ull, 0xffffff8020002000ull,
	                         0xffffff8020003000ull};

	// One display: the same bytes as before, with its framebuffer and before it attached.
	{
		const RgfxAgdcDisplays now = rgfxAgdcDisplays(1, provider, fbs);
		const RgfxAgdcDisplays before = oneDisplayAnswer(provider, fbs[0]);
		check(std::memcmp(&now, &before, sizeof now) == 0, "one display answers as before");
		const uint64_t none[1] = {0};
		const RgfxAgdcDisplays early = rgfxAgdcDisplays(1, provider, none);
		const RgfxAgdcDisplays earlyBefore = oneDisplayAnswer(provider, 0);
		check(std::memcmp(&early, &earlyBefore, sizeof early) == 0,
		      "one display before its framebuffer attached answers as before");
		const RgfxAgdcDisplays uncounted = rgfxAgdcDisplays(0, provider, fbs);
		check(std::memcmp(&now, &uncounted, sizeof now) == 0, "no count yet answers as one display");
		check(u32At(now, 0x30) == 1 && u64At(now, 0x3c) == fbs[0] && u64At(now, 0x44) == 0,
		      "IOPresentment reads one framebuffer");
	}

	// Two to four: every display's framebuffer by its index, the count and the ports.
	for (uint32_t count = 2; count <= 4; count++) {
		const RgfxAgdcDisplays d = rgfxAgdcDisplays(count, provider, fbs);
		check(u32At(d, 0x30) == count, "IOPresentment reads the count");
		for (uint32_t i = 0; i < 5; i++)
			check(d.field_20[i] == count, "the count five times");
		for (uint32_t i = 0; i < 4; i++)
			check(d.field_00[i] == (((1ull << count) - 1) << 1), "ports 1 to the count");
		for (uint32_t i = 0; i < count; i++)
			check(u64At(d, 0x3c + 8 * i) == fbs[i], "display i's framebuffer at its index");
		check(u64At(d, 0x3c + 8 * count) == 0, "nothing past the displays");
		check(d.field_34 == provider, "the provider");
	}
	check(rgfxAgdcDisplays(2, provider, fbs).field_00[0] == 0x6, "two displays are ports 1 and 2");
	{
		uint64_t many[kRgfxAgdcDisplaysMax];
		for (uint32_t i = 0; i < kRgfxAgdcDisplaysMax; i++)
			many[i] = fbs[0] + i;
		const RgfxAgdcDisplays d = rgfxAgdcDisplays(99, provider, many);
		check(d.field_20[4] == kRgfxAgdcDisplaysMax && d.field_3c[kRgfxAgdcDisplaysMax - 1] ==
		                                                   many[kRgfxAgdcDisplaysMax - 1],
		      "never past the list");
	}

	// The display a request names. One display: 0x925 and 0x986 name it 1, 0x711 0, and nothing
	// else, as before.
	check(rgfxAgdcDisplay(1, kRgfxAgdcNamedFromOne, 1) == 0, "0x925 and 0x986 name display 0 as 1");
	check(rgfxAgdcDisplay(0, kRgfxAgdcNamedFromOne, 1) == -1, "0 names none from one");
	check(rgfxAgdcDisplay(2, kRgfxAgdcNamedFromOne, 1) == -1, "2 names none of one display");
	check(rgfxAgdcDisplay(0, kRgfxAgdcNamedFromZero, 1) == 0, "0x711 names display 0 as 0");
	check(rgfxAgdcDisplay(1, kRgfxAgdcNamedFromZero, 1) == -1, "1 names none of one display");
	check(rgfxAgdcDisplay(1, kRgfxAgdcNamedFromOne, 0) == 0, "no count yet is one display");
	check(rgfxAgdcDisplay(0xffffffffu, kRgfxAgdcNamedFromOne, 1) == -1, "a wild index names none");
	// Two: each display by its index.
	check(rgfxAgdcDisplay(2, kRgfxAgdcNamedFromOne, 2) == 1, "2 names display 1 of two");
	check(rgfxAgdcDisplay(3, kRgfxAgdcNamedFromOne, 2) == -1, "3 names none of two");
	check(rgfxAgdcDisplay(1, kRgfxAgdcNamedFromZero, 2) == 1, "0x711 names display 1 as 1");
	check(rgfxAgdcDisplay(2, kRgfxAgdcNamedFromZero, 2) == -1, "2 names none of two from 0");
	check(rgfxAgdcDisplay(4, kRgfxAgdcNamedFromOne, 4) == 3, "4 names display 3 of four");

	std::printf("agdc_displays_test: %d checks passed\n", checks);
	return 0;
}
