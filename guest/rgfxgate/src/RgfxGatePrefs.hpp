// RgfxGate's AppleGVA preference (RgfxGatePrefs.cpp): gvaForceAMDKE kept true
// in /Library/Preferences/com.apple.AppleGVA.plist, so VideoToolbox's hardware
// sessions reach our bundle's video driver.
#pragma once

namespace rgfxgate {

// Run on the install thread once the Data volume is up, every boot: sets the
// key where the file lacks it (rgfx_gva_pref.h). No-op in the installer/Recovery.
void keepVideoPreference();

} // namespace rgfxgate
