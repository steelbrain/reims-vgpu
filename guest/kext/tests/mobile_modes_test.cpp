// The native test of the display's modes as an arm64 process reads them (RgfxMobileModes.hpp):
// the IOAVFamily element layouts the framebuffer writes, on each release the arm64e kexts run on,
// the timings' values, their order and scores, and the IDs set_digital_out_mode takes.
//
// kReleases records, for each release, what IOAVFamily reads of the data and the elements, from
// the release's own IOAVFamily: IOAVElement's copyProperties (ID, Score), IOAVVideoTimingAxisCopy
// (an axis's fields), IOAVVideoTimingElement's copyProperties (the flags, AspectRatio,
// ScanInformation, the 0x50 bytes of ElementData), IOAVVideoTimingGetPixelClock (the pixel clock),
// IOAVVideoTimingUpdateSyncRates (the rates' units) and IOAVVideoTimingElement's initWithData (the
// three fields it refuses at 0, and the 0x50 bytes it copies). The header's layout must be each
// release's.
#include "RgfxMobileModes.hpp"

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
		std::fprintf(stderr, "mobile_modes_test: FAILED (%s): %s\n", about, what);
		std::exit(1);
	}
}

// What a release's IOAVFamily reads, by byte offset.
struct Release {
	const char *name;
	// In an element (IOAVElement): ID and Score.
	uint32_t elementId, elementScore;
	// In the data: the size withData copies and ElementData publishes, the first axis and an axis's
	// size, and in an axis: Total, Active, SyncWidth, BackPorch, FrontPorch, PreciseSyncRate,
	// SyncPolarity and PixelRepetition.
	uint32_t dataSize, axisBase, axisSize;
	uint32_t total, active, syncWidth, backPorch, frontPorch, preciseSyncRate, syncPolarity,
	    pixelRepetition;
	// In the data: the flags, the pixel clock, AspectRatio and ScanInformation.
	uint32_t flags, pixelClock, aspectRatio, scanInformation;
	// The fields initWithData refuses at 0: the horizontal Active, the vertical Active and the
	// vertical PreciseSyncRate.
	uint32_t refusedAtZero[3];
};

constexpr Release kReleases[] = {
	{"14.6.1", 0x10, 0x18, 0x50, 0x04, 0x20, 0x0, 0x4, 0x8, 0xc, 0x10, 0x14, 0x18, 0x1c, 0x0, 0x44,
	 0x48, 0x4c, {0x08, 0x28, 0x38}},
	{"15.6.1", 0x10, 0x18, 0x50, 0x04, 0x20, 0x0, 0x4, 0x8, 0xc, 0x10, 0x14, 0x18, 0x1c, 0x0, 0x44,
	 0x48, 0x4c, {0x08, 0x28, 0x38}},
	{"26.6.2", 0x10, 0x18, 0x50, 0x04, 0x20, 0x0, 0x4, 0x8, 0xc, 0x10, 0x14, 0x18, 0x1c, 0x0, 0x44,
	 0x48, 0x4c, {0x08, 0x28, 0x38}},
	{"27.0.1", 0x10, 0x18, 0x50, 0x04, 0x20, 0x0, 0x4, 0x8, 0xc, 0x10, 0x14, 0x18, 0x1c, 0x0, 0x44,
	 0x48, 0x4c, {0x08, 0x28, 0x38}},
	{"27.2.1", 0x10, 0x18, 0x50, 0x04, 0x20, 0x0, 0x4, 0x8, 0xc, 0x10, 0x14, 0x18, 0x1c, 0x0, 0x44,
	 0x48, 0x4c, {0x08, 0x28, 0x38}},
};

uint32_t at(const RgfxIOAVTimingData &d, uint32_t offset)
{
	uint32_t v;
	std::memcpy(&v, reinterpret_cast<const uint8_t *>(&d) + offset, sizeof(v));
	return v;
}

void layouts()
{
	for (const Release &r : kReleases) {
		check(kRgfxIOAVElementId == r.elementId, "the element's ID is where IOAVFamily reads it",
		      r.name);
		check(kRgfxIOAVElementScore == r.elementScore,
		      "the element's Score is where IOAVFamily reads it", r.name);
		check(sizeof(RgfxIOAVTimingData) == r.dataSize, "the data is the size withData copies",
		      r.name);
		check(offsetof(RgfxIOAVTimingData, flags) == r.flags, "the flags", r.name);
		check(offsetof(RgfxIOAVTimingData, pixelClock) == r.pixelClock, "the pixel clock", r.name);
		check(offsetof(RgfxIOAVTimingData, aspectRatio) == r.aspectRatio, "AspectRatio", r.name);
		check(offsetof(RgfxIOAVTimingData, scanInformation) == r.scanInformation,
		      "ScanInformation", r.name);
		check(sizeof(RgfxIOAVTimingAxis) == r.axisSize, "an axis's size", r.name);
		for (uint32_t axis = 0; axis < 2; axis++) {
			const size_t base = offsetof(RgfxIOAVTimingData, axis) + axis * sizeof(RgfxIOAVTimingAxis);
			check(base == r.axisBase + axis * r.axisSize, "the axis's place", r.name);
		}
		check(offsetof(RgfxIOAVTimingAxis, total) == r.total, "Total", r.name);
		check(offsetof(RgfxIOAVTimingAxis, active) == r.active, "Active", r.name);
		check(offsetof(RgfxIOAVTimingAxis, syncWidth) == r.syncWidth, "SyncWidth", r.name);
		check(offsetof(RgfxIOAVTimingAxis, backPorch) == r.backPorch, "BackPorch", r.name);
		check(offsetof(RgfxIOAVTimingAxis, frontPorch) == r.frontPorch, "FrontPorch", r.name);
		check(offsetof(RgfxIOAVTimingAxis, preciseSyncRate) == r.preciseSyncRate,
		      "PreciseSyncRate", r.name);
		check(offsetof(RgfxIOAVTimingAxis, syncPolarity) == r.syncPolarity, "SyncPolarity", r.name);
		check(offsetof(RgfxIOAVTimingAxis, pixelRepetition) == r.pixelRepetition,
		      "PixelRepetition", r.name);
	}
}

// ReimsGraphics' Timing, as timingFor fills it: the reduced blanking of a mode.
struct Timing {
	uint64_t pixelClock;
	uint32_t hActive, hBlank, hFront, hSync;
	uint32_t vActive, vBlank, vFront, vSync;
};

void values()
{
	// 1920x1080 at 60 Hz as timingFor gives it: 80 pixels of horizontal blank (8 front, 32 sync),
	// 31 lines of vertical blank (17 front, 8 sync, 6 back), 2000 x 1111 at 60 Hz.
	const Timing t = {133320000, 1920, 80, 8, 32, 1080, 31, 17, 8};
	const RgfxIOAVTimingData d = rgfxIOAVTimingData(t, 60000);
	for (const Release &r : kReleases) {
		for (uint32_t offset : r.refusedAtZero)
			check(at(d, offset) != 0, "initWithData takes the data", r.name);
	}
	check(d.flags == 0, "progressive, not split", "1080p");
	check(d.axis[0].total == 2000 && d.axis[0].active == 1920, "the horizontal sizes", "1080p");
	check(d.axis[0].frontPorch == 8 && d.axis[0].syncWidth == 32 && d.axis[0].backPorch == 40,
	      "the horizontal blank", "1080p");
	check(d.axis[1].total == 1111 && d.axis[1].active == 1080, "the vertical sizes", "1080p");
	check(d.axis[1].frontPorch == 17 && d.axis[1].syncWidth == 8 && d.axis[1].backPorch == 6,
	      "the vertical blank", "1080p");
	check(d.axis[1].preciseSyncRate == 60 << 16, "the vertical rate is the device's, in 16.16 Hz",
	      "1080p");
	// 133.32 MHz over 2000 pixels: 66.66 kHz, in 16.16, as IOAVVideoTimingUpdateSyncRates rounds it.
	check(d.axis[0].preciseSyncRate == static_cast<int32_t>((66660ull * 65536 + 500) / 1000),
	      "the line rate is the pixel clock's, in 16.16 kHz", "1080p");
	check(d.pixelClock == 133320000, "the pixel clock in hertz", "1080p");
	check(d.axis[0].syncPolarity == 0 && d.axis[1].syncPolarity == 0 &&
	          d.axis[0].pixelRepetition == 0 && d.axis[1].pixelRepetition == 0,
	      "no polarity or repetition", "1080p");
	check(d.aspectRatio == 0 && d.scanInformation == 0, "aspect and scan left to IOAVFamily",
	      "1080p");

	// A rate that is not whole: 59.94 Hz.
	const RgfxIOAVTimingData ntsc = rgfxIOAVTimingData(t, 59940);
	check(ntsc.axis[1].preciseSyncRate == static_cast<int32_t>((59940ull * 65536 + 500) / 1000),
	      "a fractional rate in 16.16", "59.94");

	// A pixel clock past 32 bits is left to IOAVFamily.
	const Timing huge = {5000000000ull, 16384, 80, 8, 32, 16384, 900, 886, 8};
	const RgfxIOAVTimingData big = rgfxIOAVTimingData(huge, 120000);
	check(big.pixelClock == 0, "a clock past 32 bits is 0", "16384");
	check(big.axis[1].preciseSyncRate == 120 << 16, "its rate is still the device's", "16384");
}

void timings()
{
	const RgfxMode modes[] = {{3, 1280, 720, 60000}, {7, 1920, 1080, 60000}, {9, 2560, 1440, 59940}};
	RgfxMobileTiming out[3];
	check(rgfxMobileTimings(modes, 3, 7, out) == 3, "a timing for each mode", "order");
	check(out[0].mode.id == 7 && out[1].mode.id == 3 && out[2].mode.id == 9,
	      "the preferred mode first, then the device's order", "order");
	check(out[0].id == 1007 && out[1].id == 1003 && out[2].id == 1009,
	      "a timing's ID is 1000 and the mode's", "order");
	check(out[0].score == 1000 && out[1].score == 999 && out[2].score == 998,
	      "scores fall from 1000 by place", "order");
	check(rgfxMobileTimings(modes, 3, 42, out) == 3 && out[0].mode.id == 3,
	      "with no preferred mode listed, the device's order", "order");
	check(rgfxMobileTimings(modes, 0, 7, out) == 0, "no modes, no timings", "order");

	// The display starts in the timing listed first.
	check(rgfxMobileStartTimingId(modes, 3, 7) == 1007, "the preferred mode's timing", "start");
	check(rgfxMobileTimings(modes, 3, 42, out) == 3 &&
	          rgfxMobileStartTimingId(modes, 3, 42) == out[0].id,
	      "with none preferred, the first timing listed", "start");
	const RgfxMode unnumbered[] = {{0, 640, 480, 60000}, {5, 800, 600, 60000}};
	check(rgfxMobileStartTimingId(unnumbered, 2, 0) == 1005, "never mode 0, which has no timing",
	      "start");
	check(rgfxMobileStartTimingId(modes, 0, 7) == 0, "no modes, no timing", "start");

	RgfxMode m{};
	check(rgfxMobileModeFor(100, 1007, modes, 3, &m) && m.width == 1920 && m.height == 1080,
	      "the colour and timing IDs set a mode", "set");
	check(rgfxMobileModeFor(100, 1009, modes, 3, &m) && m.refreshMhz == 59940,
	      "each published timing sets its mode", "set");
	check(!rgfxMobileModeFor(1007, 100, modes, 3, &m), "the IDs are colour first", "set");
	check(!rgfxMobileModeFor(102, 1007, modes, 3, &m), "no colour mode but the one", "set");
	check(!rgfxMobileModeFor(100, 1004, modes, 3, &m), "no timing the device does not list",
	      "set");
	check(!rgfxMobileModeFor(100, 1000, modes, 3, &m), "no timing of mode 0", "set");
	check(!rgfxMobileModeFor(100, 7, modes, 3, &m), "no mode by its own ID", "set");
}

std::string readFile(const char *path)
{
	std::ifstream in(path);
	check(in.good(), "the file can be read", path);
	return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// The elements' constructors are IOAVFamily's: Graft binds them from it, so the declaration names
// it as a library, with an anchor, and the kext as one of its OSBundleLibraries.
void imports()
{
	const std::string graft = readFile("src/graft-arm64e.json");
	const std::string plist = readFile("src/Info-arm64e.plist");
	const std::string source = readFile("src/ReimsGraphicsMobile.cpp");
	const size_t library = graft.find("\"id\": \"com.apple.iokit.IOAVFamily\"");
	check(library != std::string::npos, "graft-arm64e.json names IOAVFamily", "imports");
	check(graft.find("\"anchor\": \"IOAVVideoTimingElement\"", library) != std::string::npos,
	      "anchored on its timing element", "imports");
	check(plist.find("<key>com.apple.iokit.IOAVFamily</key>") != std::string::npos,
	      "the kext's libraries name IOAVFamily", "imports");
	// The third library: index 2.
	check(graft.find("com.apple.iokit.IOGraphicsFamily") < graft.find("IOMobileGraphicsFamily") &&
	          graft.find("IOMobileGraphicsFamily") < library,
	      "IOAVFamily is the third library", "imports");
	for (const char *name :
	     {"__ZN22IOAVVideoTimingElement8withDataEPK19IOAVVideoTimingData16IOAVStandardType23"
	      "IOAVVideoTimingStandard",
	      "__ZN21IOAVVideoColorElement16withCapabilitiesE16IOAVStandardTypej22IOAVVideoPixel"
	      "Encoding26IOAVVideoColorDynamicRange18IOAVVideoColorEOTF20IOAVVideoColorimetrybPK12OS"
	      "DictionaryS7_S7_PK5OSSet",
	      "__ZN22IOAVVideoTimingElement16setColorElementsEP7OSArray"}) {
		const size_t key = graft.find(std::string("\"") + name + "\"");
		check(key != std::string::npos, "graft-arm64e.json imports it", name);
		check(graft.find("\"library\": 2", key) == graft.find("\"library\"", key),
		      "from IOAVFamily", name);
		check(source.find(std::string("__asm__(\"") + name + "\")") != std::string::npos,
		      "ReimsGraphicsMobile.cpp calls it by that name", name);
	}
}

} // namespace

int main()
{
	layouts();
	values();
	timings();
	imports();
	std::printf("mobile_modes_test: %d checks passed (%zu releases)\n", checks,
	            sizeof(kReleases) / sizeof(kReleases[0]));
	return 0;
}
