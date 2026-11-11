// The native test of what ReimsGraphics overrides of IOMobileFramebufferLegacy's own methods
// (ReimsGraphicsMobile.cpp): get_dot_pitch, because the class's start leaves its state unset for a
// PCI device; set_digital_out_mode, which the class answers kIOReturnUnsupported and the
// paravirtual display overrides to set its modes; swap_set_color_matrix, which the class answers
// kIOReturnUnsupported and the display answers 0 to show the matrix; and set_parameter, whose
// colour filter scale of 1.0 the display answers 0 and hands the rest to the class
// (RgfxMobileColor.hpp, tests/mobile_color_test.cpp). start keeps its provider as an
// AppleARMIODevice in one field, NULL where the provider is not one, and the class's get_dot_pitch
// calls through that field unchecked.
//
// For each release the arm64e kexts run on, kReleases records that field and the vtable slot of
// each method ReimsGraphics overrides for it, read from the release's IOMobileGraphicsFamily. Graft
// binds an override by its method, not its slot, so the test checks what the bind needs on every
// release: the vendored SDK stand-in declares each method, ReimsGraphics overrides it, and
// graft-arm64e.json requires it, so a release without the method refuses the bind rather than keep
// the class's own.
#include "RgfxDisplayIdentity.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>

namespace {

int checks = 0;

void check(bool ok, const char *what, const char *about)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "legacy_overrides_test: FAILED (%s): %s\n", about, what);
		std::exit(1);
	}
}

// A method ReimsGraphics overrides: its member name, as graft-arm64e.json keys it, its declaration
// in the SDK stand-in and in ReimsGraphics.hpp, and its Itanium name in the class.
struct Method {
	const char *name;
	const char *standIn;
	const char *override;
	const char *mangled;
};

constexpr Method kMethods[] = {
	{"get_dot_pitch", "virtual graft_iomfb_word get_dot_pitch(uint32_t *pitch);",
	 "RgfxIOMFBWord get_dot_pitch(uint32_t *pitch) override;",
	 "__ZN25IOMobileFramebufferLegacy13get_dot_pitchEPj"},
	{"set_digital_out_mode",
	 "virtual graft_iomfb_word set_digital_out_mode(uint32_t field_0, uint32_t field_1);",
	 "RgfxIOMFBWord set_digital_out_mode(uint32_t colorId, uint32_t timingId) override;",
	 "__ZN25IOMobileFramebufferLegacy20set_digital_out_modeEjj"},
	{"swap_set_color_matrix",
	 "virtual graft_iomfb_word swap_set_color_matrix(IOMFBColorFixedMatrix *matrix,\n"
	 "                                                   IOMFBColorMatrixFunction function,\n"
	 "                                                   uint32_t field_2);",
	 "RgfxIOMFBWord swap_set_color_matrix(IOMFBColorFixedMatrix *matrix,\n"
	 "\t                                    IOMFBColorMatrixFunction function, uint32_t field_2) "
	 "override;",
	 "__ZN25IOMobileFramebufferLegacy21swap_set_color_matrixEP21IOMFBColorFixedMatrix"
	 "24IOMFBColorMatrixFunctionj"},
	{"set_parameter",
	 "virtual graft_iomfb_word set_parameter(IOMFBParameterName name, const uint64_t *values,\n"
	 "                                           uint32_t count);",
	 "RgfxIOMFBWord set_parameter(IOMFBParameterName name, const uint64_t *values,\n"
	 "\t                            uint32_t count) override;",
	 "__ZN25IOMobileFramebufferLegacy13set_parameterE18IOMFBParameterNamePKyj"},
};
constexpr size_t kMethodCount = sizeof(kMethods) / sizeof(kMethods[0]);

// A release's class: its size, the field start keeps its provider in as an AppleARMIODevice, and
// each of kMethods' vtable slots, in kMethods' order.
struct Release {
	const char *name;
	uint32_t classSize;
	uint32_t armIODevice;
	uint32_t slots[kMethodCount];
};

constexpr Release kReleases[] = {
	{"14.6.1", 0x18a0, 0x1640, {292, 289, 331, 332}},
	{"15.6.1", 0x19c0, 0x1758, {292, 289, 334, 335}},
	{"26.6.2", 0x1af8, 0x1890, {293, 290, 335, 336}},
	{"27.0.1", 0x1dd8, 0x1b80, {294, 291, 343, 344}},
	{"27.2.1", 0x1de0, 0x1b80, {294, 291, 343, 344}},
};

// The SDK stand-in's virtual slots past its own: GRAFT_IOMFB_VTABLE_RESERVE.
constexpr uint32_t kStandInSlots = 512;

std::string readFile(const char *path)
{
	std::ifstream in(path);
	check(in.good(), "the file can be read", path);
	return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

// The text of the JSON object that follows "key": in `text`, from its brace to its own closing one.
std::string object(const std::string &text, const std::string &key)
{
	const size_t at = text.find("\"" + key + "\"");
	check(at != std::string::npos, "the key is present", key.c_str());
	const size_t open = text.find('{', at);
	check(open != std::string::npos, "the key holds an object", key.c_str());
	int depth = 0;
	for (size_t i = open; i < text.size(); i++) {
		if (text[i] == '{')
			depth++;
		else if (text[i] == '}' && --depth == 0)
			return text.substr(open, i - open + 1);
	}
	check(false, "the object closes", key.c_str());
	return "";
}

void releases()
{
	for (const Release &r : kReleases) {
		check(r.armIODevice % 8 == 0, "the provider field is a pointer's", r.name);
		check(r.armIODevice + 8 <= r.classSize, "the provider field is inside the class", r.name);
		for (size_t m = 0; m < kMethodCount; m++)
			check(r.slots[m] < kStandInSlots, "the slot is inside the stand-in's vtable", r.name);
	}
	for (size_t i = 1; i < sizeof(kReleases) / sizeof(kReleases[0]); i++) {
		const Release &a = kReleases[i - 1], &b = kReleases[i];
		check(a.classSize < b.classSize, "each release's class is larger than the last's", b.name);
		check(a.armIODevice <= b.armIODevice, "the provider field never moves down", b.name);
		for (size_t m = 0; m < kMethodCount; m++)
			check(a.slots[m] <= b.slots[m], "a method's slot never moves down", b.name);
	}
}

void declarations()
{
	const std::string standIn = readFile("../graft/sdk/include/graft/IOMobileFramebufferLegacy.h");
	const std::string header = readFile("src/ReimsGraphics.hpp");
	const std::string overrides = object(object(readFile("src/graft-arm64e.json"), "synthesized"),
	                                     "overrides");
	for (const Method &m : kMethods) {
		check(standIn.find(m.standIn) != std::string::npos, "the SDK stand-in declares it", m.name);
		check(header.find(m.override) != std::string::npos, "ReimsGraphics overrides it", m.name);
		const std::string spec = object(overrides, m.name);
		check(spec.find("\"required\": true") != std::string::npos,
		      "graft-arm64e.json requires the override", m.name);
		check(std::string(m.mangled).find(std::string(m.name) + "E") != std::string::npos,
		      "the Itanium name is the method's", m.name);
	}
}

// get_dot_pitch answers the display's pixels per inch, the density its published size is at: 110
// a point of the host screen that shows it, whatever the display's width.
void dotPitch()
{
	check(rgfxDisplayPixelsPerInch(0) == 110, "no scale from the host: 110 ppi", "pitch");
	check(rgfxDisplayPixelsPerInch(120) == 110, "a screen at 1x: 110 ppi", "pitch");
	check(rgfxDisplayPixelsPerInch(180) == 165, "a screen at 1.5x: 165 ppi", "pitch");
	check(rgfxDisplayPixelsPerInch(240) == 220, "a screen at 2x: 220 ppi", "pitch");
	check(rgfxDisplayPixelsPerInch(252) == 231, "KDE's 2.1x: 231 ppi", "pitch");
	check(rgfxDisplayPixelsPerInch(150) == 138, "1.25x: 137.5 ppi rounds up", "pitch");
	check(!rgfxDisplayHighDensity(0) && !rgfxDisplayHighDensity(120), "1x is not high-density",
	      "pitch");
	check(rgfxDisplayHighDensity(150) && rgfxDisplayHighDensity(240), "above 1x is", "pitch");
	for (uint32_t scale : {0u, 120u, 150u, 180u, 240u, 252u, 360u}) {
		const uint32_t ppi = rgfxDisplayPixelsPerInch(scale);
		for (uint32_t width : {1024u, 1280u, 1920u, 2560u, 2880u, 3840u, 5120u, 7680u, 16384u}) {
			const uint32_t mm = rgfxDisplayMillimetres(width, scale);
			// The size, rounded to the millimetre, is the width at the pitch: within half a
			// millimetre.
			const long twice = static_cast<long>(width) * 254 * 2 / ppi - static_cast<long>(mm) * 20;
			check(twice >= -10 && twice <= 10, "the published size is the width at the pitch",
			      "pitch");
		}
	}
}

} // namespace

int main()
{
	releases();
	declarations();
	dotPitch();
	std::printf("legacy_overrides_test: %d checks passed (%zu releases, %zu methods)\n", checks,
	            sizeof(kReleases) / sizeof(kReleases[0]), kMethodCount);
	return 0;
}
