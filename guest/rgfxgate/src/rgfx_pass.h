// Pure decisions of RgfxGate's install pass (RgfxGateInstall.cpp), shared with
// the native tests: what a pass does where it runs, and when the watch that
// follows a pass in arm64's BaseSystem passes again.
//
// A base system is one with /System/Installation. On x86-64 that is Recovery
// and the installer, whose base system ships no IOGPU.framework, so the bundle
// could not give them Metal and the pass writes nothing there (design.md,
// "Registration waits for the bundle"). On arm64 every boot starts in the
// paired recoveryOS's BaseSystem, and the boots that stay there (FileVault's
// unlock, Recovery, the obliteration boots) draw their windows through Metal:
// that BaseSystem ships IOGPU.framework and Apple's own GPU bundles, and its
// Data volume is a temporary one (rc.temporaryDataVolume's tmpfs) whose
// template carries /Library/GPUBundles. So there the pass installs the bundle
// into that temporary Data volume, and then watches for the next Data volume
// the boot moves to without restarting the kernel: the full system's, once
// FileVault's password unlocks it, or another BaseSystem's, whose
// rc.temporaryDataVolume makes a new temporary one. Each gets a pass of its own,
// and while the watch runs each pass publishes the root it ended under
// (RgfxGate.hpp, kPassRootKey), for which the accelerator holds a process's read
// of its MetalPluginName on any other root.
#pragma once
#include <stdint.h>

enum RgfxPassKind : int {
	// A base system where the bundle is not installed: nothing is written.
	kRgfxPassSkip,
	// A full system's Data volume: the install, and nothing after it.
	kRgfxPassInstall,
	// arm64's BaseSystem: the install into its temporary Data volume, then the
	// watch for the next Data volume.
	kRgfxPassInstallThenWatch,
};

// What a pass does: `base_system` where /System/Installation is there,
// `installs_in_base_system` where the build installs into a base system's Data
// volume (arm64e).
static inline RgfxPassKind rgfx_pass_kind(bool base_system, bool installs_in_base_system) {
	if (!base_system)
		return kRgfxPassInstall;
	return installs_in_base_system ? kRgfxPassInstallThenWatch : kRgfxPassSkip;
}

// Whether the watch passes again: a Data volume is there (`have`, its file
// system's id `id`), and it is not the one the last pass ran on (`last`).
static inline bool rgfx_watch_passes(bool have, uint64_t id, uint64_t last) {
	return have && id != last;
}

// Whether the watch ends, looked at after any pass it made: the root is a full
// system (`base_system` false) and the Data volume there (`have`, `id`) is the
// one the last pass ran on (`last`). A boot that has reached its full system
// stays there; until its Data volume has had a pass, the watch goes on.
static inline bool rgfx_watch_ends(bool base_system, bool have, uint64_t id, uint64_t last) {
	return !base_system && have && id == last;
}
