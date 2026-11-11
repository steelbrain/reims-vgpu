// The native test of RgfxDisplayModes.hpp: across a change of the device's modes at any
// moment of WindowServer's display setup, the framebuffer's answers stay consistent: the mode
// it answers as current is always one it lists, the startup mode follows the device, a change
// before IOFramebuffer registers its connect interrupt fires once it does, and the mode
// configuration a WindowServer left goes at the next server connection after a change.
#include "RgfxDisplayModes.hpp"

#include <cstdio>
#include <cstdlib>

namespace {
int checks = 0;

void check(bool ok, const char *what)
{
	checks++;
	if (!ok) {
		std::fprintf(stderr, "display_modes_test: FAILED: %s\n", what);
		std::exit(1);
	}
}

// The OS's timing, which the rules only keep.
struct Timing {
	uint32_t tag;
};

constexpr uint32_t kMax = 4;
using Modes = RgfxDisplayModes<Timing, kMax>;
constexpr int32_t kReservedBase = static_cast<int32_t>(0x80000000u);
constexpr int32_t kProgrammed = static_cast<int32_t>(0x80001000u);

// The host's modes: the window before its first draw, and its size after, at 120 Hz.
constexpr RgfxMode kOffered = {0, 1920, 1080, 120000};
constexpr RgfxMode kWindow = {0, 3840, 2400, 120000};
constexpr RgfxMode kResized = {0, 2560, 1600, 120000};

bool lists(const Modes &m, int32_t id)
{
	int32_t ids[kMax + 1];
	uint32_t n = m.list(ids);
	for (uint32_t i = 0; i < n; i++) {
		if (ids[i] == id)
			return true;
	}
	return false;
}

// What WindowServer reads at its display setup: the mode set (getCurrentDisplayMode), which
// the list holds (getDisplayModes) and whose information the framebuffer gives.
bool consistent(const Modes &m)
{
	int32_t current = m.haveCurrent ? m.current.id : m.startupId;
	RgfxMode info;
	return lists(m, current) && m.modeFor(current, &info) && m.modeFor(m.startupId, &info);
}

// The framebuffer's start: the device's first list, its startup mode set (enableController).
Modes started(const RgfxMode &first)
{
	Modes m;
	m.reset();
	check(m.take(&first, 1, 0), "the first list is a change");
	check(m.startupId == m.preferredId && m.preferredId != 0, "the start takes the preferred mode");
	RgfxMode startup;
	check(m.modeFor(m.startupId, &startup), "the startup mode has its information");
	m.set(startup);
	check(consistent(m), "the started framebuffer's answers are consistent");
	return m;
}

void aChangeBeforeTheConnectInterruptIsRegistered()
{
	// The window draws only after the start's GET_DISPLAYS, so the device's list changes right
	// after the start's first mode is set, before IOFramebuffer registers its connect
	// interrupt; WindowServer is already waiting to open the framebuffer.
	Modes m = started(kOffered);
	int32_t offered = m.current.id;
	RgfxConnectChange connect = {};
	check(m.take(&kWindow, 1, 0), "the window's size is a change");
	check(!connect.changed(false), "no interrupt registered: the change waits");

	int32_t window = m.preferredId;
	check(window != offered, "the window's mode has an ID of its own");
	check(m.current.id == offered && lists(m, offered), "the mode set stays listed");
	check(lists(m, window), "the device's mode is listed");
	check(m.list(nullptr) == 2, "both, and nothing else");
	check(m.startupId == window, "the startup mode follows the device");
	check(consistent(m), "the answers stay consistent");
	RgfxMode info;
	check(m.modeFor(offered, &info) && !m.isPreferred(info), "the mode set is not preferred");
	check(m.modeFor(window, &info) && m.isPreferred(info), "the device's mode is");

	// IOFramebuffer registers the interrupt, off, then turns it on: the change fires once.
	check(!connect.armed(false), "registered off: it still waits");
	check(connect.armed(true), "on: the change fires");
	check(!connect.armed(true), "once");

	// WindowServer sets the device's mode: the old one is no longer listed.
	RgfxMode shown;
	check(m.modeFor(window, &shown), "the window's mode has its information");
	m.set(shown);
	check(!lists(m, offered) && m.list(nullptr) == 1, "the old mode drops once another is set");
	check(consistent(m), "and the answers stay consistent");
}

void aChangeBetweenTheCurrentModeAndTheList()
{
	// WindowServer reads the count and the mode set, then the list changes before it reads
	// the list: the list it is given in the count it was told still names the mode set.
	Modes m = started(kWindow);
	int32_t window = m.current.id;
	uint32_t reported = m.list(nullptr);
	check(reported == 1, "one mode reported");
	RgfxConnectChange connect = {};
	check(m.take(&kResized, 1, 0), "the resize is a change");
	check(connect.changed(true), "the registered interrupt fires at once");
	int32_t out[kMax + 1] = {};
	m.fill(out, reported);
	check(out[0] == window, "the reported list holds the mode set");

	// The connect interrupt has WindowServer ask again: both, the device's preferred.
	reported = m.list(nullptr);
	check(reported == 2, "the mode set and the device's mode");
	m.fill(out, reported);
	check((out[0] == window || out[1] == window) && out[0] != out[1], "each once");
	check(m.startupId == m.preferredId && m.preferredId != window, "the startup follows");
	check(consistent(m), "the answers stay consistent");
}

void aChangeAfterTheStartupModeIsSet()
{
	// WindowServer programs the window's timing, sets it and makes it the startup mode; then
	// the window is resized, and that WindowServer ends before it takes the change in.
	Modes m = started(kWindow);
	RgfxModeConfig config = {};
	config.listChanged(); // the start's first list
	check(config.serverOpens(), "the first server connection: nothing left to keep");

	Timing timing = {7};
	m.clearProgrammed();
	check(m.addProgrammed(kProgrammed, kWindow, timing, kReservedBase), "the timing is programmed");
	check(!m.addProgrammed(kProgrammed, kWindow, timing, kReservedBase), "an ID once");
	check(!m.addProgrammed(5, kWindow, timing, kReservedBase), "only from the reserved base");
	m.followProgrammed();
	check(m.current.id == kProgrammed, "the mode set takes the programmed ID");
	check(m.list(nullptr) == 1 && lists(m, kProgrammed), "the list is the programmed timing");
	RgfxMode set;
	check(m.modeFor(kProgrammed, &set) && m.isPreferred(set), "which is the preferred mode's");
	m.set(set);
	m.startupId = kProgrammed;
	check(consistent(m), "consistent before the change");

	check(m.take(&kResized, 1, 0), "the resize is a change");
	config.listChanged();
	check(lists(m, kProgrammed), "the programmed mode set stays listed");
	check(lists(m, m.preferredId), "the device's new mode is listed");
	check(m.list(nullptr) == 2, "both, and nothing else");
	check(m.startupId == m.preferredId, "the startup mode follows the device");
	check(consistent(m), "the answers stay consistent");
	const Modes::Programmed *p = m.programmedFor(kProgrammed);
	check(p && p->timing.tag == 7, "the programmed timing is kept");

	// The next WindowServer's server connection drops the configuration made from the old list.
	check(config.serverOpens(), "a changed list: the configuration goes");
	check(!config.serverOpens(), "an unchanged one: it stays");
}

void idsStayWithTheirModes()
{
	Modes m = started(kWindow);
	int32_t window = m.current.id;
	RgfxMode both[2] = {kResized, kWindow};
	check(m.take(both, 2, 0), "a new mode listed is a change");
	int32_t resized = m.preferredId;
	check(resized != window, "the new mode has a new ID");
	check(lists(m, window) && m.list(nullptr) == 2, "a listed mode keeps its ID");
	check(!m.take(both, 2, 0), "the same list again is no change");
	check(m.take(both, 2, 1), "another preferred mode is a change");
	check(m.preferredId == window, "the preferred mode is the one named");
	check(m.take(&kResized, 1, 7), "a dropped mode is a change");
	check(m.preferredId == resized, "no preferred mode named: the first");
	check(m.modeCount == 1 && m.current.id == window, "the mode set stays set");
	check(m.take(both, 2, 0), "listed again");
	check(m.modes[1].id == window, "the mode set's mode takes back its ID");
}

// The HiDPI flag is for the modes of a display on a host screen above 1x whose pixels halve
// exactly, at any width: the window's on a screen at twice its points keeps it as it shrinks, one
// on a screen at 1x never has it, and one of an odd size never does, which WindowServer would run
// at half pixels.
void hiDpiModesHalveExactly()
{
	constexpr uint32_t k1x = 120, k2x = 240, kKde = 252;
	check(RgfxMode{0, 2880, 1800, 60000}.hiDpi(k2x), "2880x1800 at 2x");
	check(RgfxMode{0, 3840, 2160, 60000}.hiDpi(k2x), "3840x2160 at 2x");
	check(kWindow.hiDpi(kKde), "the window's 3840x2400 at 2.1x");
	check(RgfxMode{0, 2878, 1800, 60000}.hiDpi(k2x), "below 2880 pixels wide at 2x");
	check(RgfxMode{0, 2720, 1536, 60000}.hiDpi(kKde), "2720x1536 at 2.1x");
	check(!RgfxMode{0, 3840, 2160, 60000}.hiDpi(k1x), "not 3840x2160 at 1x");
	check(!kWindow.hiDpi(0), "not with no scale from the host");
	check(!kOffered.hiDpi(0), "not the 1920x1080 offered before the window draws");
	check(!RgfxMode{0, 3841, 2161, 60000}.hiDpi(k2x), "not an odd width and height");
	check(!RgfxMode{0, 3662, 2039, 60000}.hiDpi(k2x), "not an odd height");
	check(!RgfxMode{0, 3663, 2040, 60000}.hiDpi(k2x), "not an odd width");
}

// A mode or depth the framebuffer does not hold is answered as Apple's paravirtual framebuffer
// answers it: kIOReturnUnsupportedMode for the pixel information, kIOReturnError elsewhere.
void anUnknownModeIsAnsweredAsApplesFramebufferDoes()
{
	check(rgfxNoSuchModeStatus(RgfxModeCall::PixelInformation) == 0xe00002e6u,
	      "the pixel information of no mode: kIOReturnUnsupportedMode");
	check(rgfxNoSuchModeStatus(RgfxModeCall::ModeInformation) == 0xe00002bcu,
	      "the mode information of no mode: kIOReturnError");
	check(rgfxNoSuchModeStatus(RgfxModeCall::TimingInformation) == 0xe00002bcu,
	      "the timing information of no mode: kIOReturnError");
	check(rgfxNoSuchModeStatus(RgfxModeCall::SetMode) == 0xe00002bcu,
	      "setting no mode: kIOReturnError");
	check(rgfxNoSuchModeStatus(RgfxModeCall::SetStartupMode) == 0xe00002bcu,
	      "setting no startup mode: kIOReturnError");
	Modes m = started(kWindow);
	RgfxMode none;
	check(!m.modeFor(0, &none), "mode 0, which WindowServer asks about, is none of ours");
}
} // namespace

int main()
{
	aChangeBeforeTheConnectInterruptIsRegistered();
	aChangeBetweenTheCurrentModeAndTheList();
	aChangeAfterTheStartupModeIsSet();
	idsStayWithTheirModes();
	hiDpiModesHalveExactly();
	anUnknownModeIsAnsweredAsApplesFramebufferDoes();
	std::printf("display_modes_test: OK (%d checks)\n", checks);
	return 0;
}
