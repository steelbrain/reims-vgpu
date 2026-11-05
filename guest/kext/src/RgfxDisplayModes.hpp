// RgfxDisplayModes: the display's modes as the framebuffer gives them to IOFramebuffer, and the
// mode set. Header-only and free of the kernel's headers, so its native test
// (tests/display_modes_test.cpp) builds it as the kext does; `Timing` is the detailed timing
// the OS programs (IODetailedTimingInformationV2 in the kext), which the rules only keep.
//
// The modes are the device's (its DISPLAYS answer: the host window's size at its screen's
// rate), each under an ID that stays with its size and rate while the device lists it, and
// the timings the OS programmed (setDetailedTimings), under the IDs the OS gave them. The list
// IOFramebuffer sees is the programmed timings the device lists, or, while there are none, the
// device's own modes; and the mode set, wherever that list does not hold it.
//
// Why the mode set stays listed: the host's mode can change at any time (its window draws its
// first frame, or is resized), and WindowServer can be reading the framebuffer's modes when it
// does: WindowServer waits for the framebuffer and opens it during the framebuffer's own start.
// The framebuffer answers the mode set as current, and the list it gives must hold that mode,
// with its information and timing, until a mode is set again; it is never the preferred one
// where the device no longer lists it.
//
// The startup mode follows the device: where the device no longer shows it, it becomes the
// device's preferred mode, as the framebuffer's start chose it.
//
// Our own code.
#ifndef RGFX_DISPLAY_MODES_HPP_
#define RGFX_DISPLAY_MODES_HPP_

#include <stdint.h>

// A display mode: its ID, pixels each way and refresh rate in thousandths of a hertz.
struct RgfxMode {
	int32_t id;
	uint32_t width;
	uint32_t height;
	uint32_t refreshMhz;
};

template <typename Timing, uint32_t Max> struct RgfxDisplayModes {
	// A timing the OS programmed: its mode, under the ID the OS gave it, and the timing.
	struct Programmed {
		RgfxMode mode;
		Timing timing;
	};

	RgfxMode modes[Max];
	uint32_t modeCount;
	int32_t preferredId;
	int32_t nextModeId;
	bool haveCurrent;
	RgfxMode current;
	int32_t startupId;
	Programmed programmed[Max];
	uint32_t programmedCount;

	// No modes, no mode set; IDs from 1.
	void reset()
	{
		modeCount = 0;
		preferredId = 0;
		nextModeId = 1;
		haveCurrent = false;
		current = RgfxMode{0, 0, 0, 0};
		startupId = 0;
		programmedCount = 0;
	}

	// Takes the device's modes, `fresh[0..n)` (their IDs ignored), `preferred` the index of the
	// one it prefers (past the list: none). A mode of a size and rate listed before, or set,
	// keeps its ID; another gets an ID none had. The preferred mode is the one named, else the
	// first. The startup mode becomes the preferred one where the device no longer shows it,
	// and on the first list. Returns whether the list or its preferred mode changed (the first
	// list is a change).
	bool take(const RgfxMode *fresh, uint32_t n, uint32_t preferred)
	{
		RgfxMode list[Max];
		uint32_t count = 0;
		int32_t named = 0;
		for (uint32_t j = 0; j < n && count < Max; j++) {
			RgfxMode m = fresh[j];
			m.id = 0;
			for (uint32_t k = 0; k < modeCount && !m.id; k++) {
				if (same(modes[k], m))
					m.id = modes[k].id;
			}
			if (!m.id && haveCurrent && same(current, m))
				m.id = current.id;
			if (!m.id)
				m.id = nextModeId++;
			list[count++] = m;
			if (j == preferred)
				named = m.id;
		}
		bool first = !startupId;
		int32_t newPreferred = named ? named : (count ? list[0].id : 0);
		bool changed = first || count != modeCount || newPreferred != preferredId;
		for (uint32_t i = 0; i < count; i++) {
			changed = changed || list[i].id != modes[i].id;
			modes[i] = list[i];
		}
		modeCount = count;
		preferredId = newPreferred;
		if (first || !shows(startupId))
			startupId = preferredId;
		return changed;
	}

	// The mode of an ID: one listed, a programmed timing's, or the mode set.
	bool modeFor(int32_t id, RgfxMode *mode) const
	{
		for (uint32_t i = 0; i < modeCount; i++) {
			if (modes[i].id == id) {
				*mode = modes[i];
				return true;
			}
		}
		if (const Programmed *p = programmedFor(id)) {
			*mode = p->mode;
			return true;
		}
		if (haveCurrent && current.id == id) {
			*mode = current;
			return true;
		}
		return false;
	}

	// Whether the device lists a mode of this size and rate (within half a hertz), and which.
	bool deviceMode(uint32_t width, uint32_t height, uint32_t refreshMhz, RgfxMode *mode) const
	{
		for (uint32_t i = 0; i < modeCount; i++) {
			const RgfxMode &m = modes[i];
			if (m.width == width && m.height == height && near(m.refreshMhz, refreshMhz)) {
				if (mode)
					*mode = m;
				return true;
			}
		}
		return false;
	}

	const Programmed *programmedFor(int32_t id) const
	{
		for (uint32_t i = 0; i < programmedCount; i++) {
			if (programmed[i].mode.id == id)
				return &programmed[i];
		}
		return nullptr;
	}

	// The device's preferred mode, or a programmed timing of its size and rate.
	bool isPreferred(const RgfxMode &m) const
	{
		if (m.id == preferredId)
			return true;
		RgfxMode host;
		return programmedFor(m.id) && deviceMode(m.width, m.height, m.refreshMhz, &host) &&
		       host.id == preferredId;
	}

	// Whether the device shows the mode of an ID: one it lists, or a programmed timing of one.
	bool shows(int32_t id) const
	{
		for (uint32_t i = 0; i < modeCount; i++) {
			if (modes[i].id == id)
				return true;
		}
		const Programmed *p = programmedFor(id);
		return p && deviceMode(p->mode.width, p->mode.height, p->mode.refreshMhz, nullptr);
	}

	// The modes IOFramebuffer sees, into `ids` (Max + 1 of room) when not null: the programmed
	// timings the device lists, or, while there are none, the device's own; then the mode set,
	// where they do not hold it. Returns their count.
	uint32_t list(int32_t *ids) const
	{
		uint32_t n = 0;
		bool holdsCurrent = false;
		for (uint32_t i = 0; i < programmedCount; i++) {
			const RgfxMode &m = programmed[i].mode;
			if (!deviceMode(m.width, m.height, m.refreshMhz, nullptr))
				continue;
			if (ids)
				ids[n] = m.id;
			holdsCurrent = holdsCurrent || (haveCurrent && m.id == current.id);
			n++;
		}
		if (!n) {
			for (uint32_t i = 0; i < modeCount; i++) {
				if (ids)
					ids[n] = modes[i].id;
				holdsCurrent = holdsCurrent || (haveCurrent && modes[i].id == current.id);
				n++;
			}
		}
		if (haveCurrent && !holdsCurrent) {
			if (ids)
				ids[n] = current.id;
			n++;
		}
		return n;
	}

	// `reported` IDs into `out`, for a caller that was told the list holds `reported`: the
	// list's, the mode set among them where the list holds it, so a list that grew since still
	// names the mode set; where the list is shorter, the rest repeat its first (the preferred
	// mode where it is empty). The connect interrupt that follows a change has the caller ask
	// again.
	void fill(int32_t *out, uint32_t reported) const
	{
		int32_t ids[Max + 1];
		uint32_t n = list(ids);
		for (uint32_t i = 0; i < reported; i++)
			out[i] = i < n ? ids[i] : (n ? ids[0] : preferredId);
		if (!haveCurrent || !reported || n <= reported)
			return;
		for (uint32_t i = 0; i < reported; i++) {
			if (out[i] == current.id)
				return;
		}
		out[reported - 1] = current.id;
	}

	// The mode is set: the scanout shows it.
	void set(const RgfxMode &mode)
	{
		current = mode;
		haveCurrent = true;
	}

	// The OS's programmed timings: none, then each added. False where `id` is below the
	// reserved base the OS numbers them from, or already programmed, or there is no room.
	void clearProgrammed() { programmedCount = 0; }
	bool addProgrammed(int32_t id, const RgfxMode &mode, const Timing &timing, int32_t reservedBase)
	{
		if (static_cast<uint32_t>(id) < static_cast<uint32_t>(reservedBase) || programmedFor(id) ||
		    programmedCount >= Max)
			return false;
		Programmed &p = programmed[programmedCount++];
		p.mode = mode;
		p.mode.id = id;
		p.timing = timing;
		return true;
	}

	// The mode set takes the ID of a programmed timing of its size and rate the device lists,
	// as the list now names it.
	void followProgrammed()
	{
		if (!haveCurrent || programmedFor(current.id))
			return;
		for (uint32_t i = 0; i < programmedCount; i++) {
			const RgfxMode &m = programmed[i].mode;
			if (m.width == current.width && m.height == current.height &&
			    near(m.refreshMhz, current.refreshMhz) &&
			    deviceMode(m.width, m.height, m.refreshMhz, nullptr)) {
				current.id = m.id;
				return;
			}
		}
	}

private:
	static bool same(const RgfxMode &a, const RgfxMode &b)
	{
		return a.width == b.width && a.height == b.height && a.refreshMhz == b.refreshMhz;
	}
	static bool near(uint32_t a, uint32_t b) { return (a > b ? a - b : b - a) <= 500; }
};

// The mode configuration a WindowServer leaves on the framebuffer (kIOFBConfigKey, which it
// writes through IOFramebuffer's setProperties and the next WindowServer reads back), made from
// the list it saw. Where the device's list changed since the last server connection opened,
// the next one drops it, so a WindowServer starting after a change, its predecessor gone before
// it took the change in, builds its configuration from the list, as at boot, where there is
// none.
struct RgfxModeConfig {
	uint32_t changes;
	uint32_t atOpen;

	// The device's list changed (Modes::take said so).
	void listChanged() { changes++; }

	// A server connection opens: true where the configuration left on the framebuffer is to go.
	bool serverOpens()
	{
		bool stale = changes != atOpen;
		atOpen = changes;
		return stale;
	}
};

// The connect interrupt for a change of the device's display: IOFramebuffer senses the
// connection again when it fires. A change while IOFramebuffer has no connect interrupt
// registered, or has it off (IOFramebuffer registers it during its start, after the start's
// first mode is set), waits, and fires once the interrupt is registered and on, so
// IOFramebuffer and WindowServer learn of every change whenever it lands.
struct RgfxConnectChange {
	bool pending;

	// The display changed: true where the interrupt fires now (it is registered and on);
	// else the change waits.
	bool changed(bool canFire)
	{
		pending = !canFire;
		return canFire;
	}

	// The interrupt may fire: true where a change waits for it, which it then takes.
	bool armed(bool canFire)
	{
		if (!pending || !canFire)
			return false;
		pending = false;
		return true;
	}
};

// The framebuffer's calls that name a mode and a depth.
enum class RgfxModeCall {
	PixelInformation,
	ModeInformation,
	TimingInformation,
	SetMode,
	SetStartupMode,
};

// IOReturn's kIOReturnError and kIOReturnUnsupportedMode, which the kext checks against
// IOReturn.h.
constexpr uint32_t kRgfxIOReturnError = 0xe00002bc;
constexpr uint32_t kRgfxIOReturnUnsupportedMode = 0xe00002e6;

// What `call` answers for a mode the framebuffer does not list, or a depth past its one (0), as
// Apple's paravirtual framebuffer answers each (AppleParavirtFramebuffer, macOS 26.7.1, read from
// its code): the pixel information kIOReturnUnsupportedMode, as for an aperture other than the
// system's, and the others kIOReturnError. Apple's lists its modes as 0 to its count - 1, and
// WindowServer asks the pixel information of mode 0, which this framebuffer's IDs never are.
inline uint32_t rgfxNoSuchModeStatus(RgfxModeCall call)
{
	return call == RgfxModeCall::PixelInformation ? kRgfxIOReturnUnsupportedMode
	                                              : kRgfxIOReturnError;
}

#endif // RGFX_DISPLAY_MODES_HPP_
