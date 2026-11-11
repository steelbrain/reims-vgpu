// RgfxGate's AppleGVA preference (RgfxGatePrefs.cpp): gvaForceAMDKE kept true
// in /Library/Preferences/com.apple.AppleGVA.plist and in the global domain's
// /Library/Preferences/.GlobalPreferences.plist, so VideoToolbox's hardware
// sessions reach our bundle's video driver in every process, a sandboxed one's
// too.
#pragma once

namespace rgfxgate {

// Run on the install thread once the Data volume is up, every boot: sets the
// key where a file lacks it (rgfx_gva_pref.h). No-op in the installer/Recovery.
void keepVideoPreference();

} // namespace rgfxgate
