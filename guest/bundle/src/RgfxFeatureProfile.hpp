// The feature profile the device answers (`featureProfile`, SPI). At device creation Metal reads
// it before anything else and keys three things on it: the device's limits, its GPU families and
// its feature queries. A guest's Metal that has no limits for the profile it reads leaves every
// limit 0 and ends the process at its first check of one. WindowServer is that process, so the
// guest has no desktop.
//
// The host answers its own Metal device's profile, or the M3 Max's (10) where no Metal device
// stands for its GPU (the Vulkan platform). The device keeps that answer wherever the guest's
// Metal knows it (docs/graphics-kext/dgpu-oracle.md, Q11, says why it is not the x86 Macs' 10002
// there). Metal of macOS 12 does not know 10: the device answers 10001 there, as 12's drivers for
// the oracle's two GPUs do (AMDRadeonX6000's, the Radeon Pro 5300M's, and AppleIntelKBLGraphics',
// the UHD 630's; read from 12.7.6's drivers), and 12's Metal has limits for it. macOS 11's Metal
// is taken to be 12's here: its boots stop in the kernel before WindowServer (Graft's to fix).
//
// Whether the guest's Metal knows the host's profile is told by what it has, not by its version:
// Metal that has the MTLArchitecture class (macOS 13 and later) knows 10 (13.7.8, 14.8.9 and
// 26.7.2 boot to their desktops with it), and Metal without it (macOS 11 and 12) does not (12.7.6's
// WindowServer ends at device creation with it).

#ifndef RGFX_FEATURE_PROFILE_HPP
#define RGFX_FEATURE_PROFILE_HPP

#include <cstdint>

namespace rgfx_profile {

// The profile macOS 11 and 12's x86 GPU drivers answer.
constexpr uint64_t kX86Mac = 10001;

// The profile the device answers: the host's where the guest's Metal knows it (it has the
// MTLArchitecture class), else the x86 Macs' that Metal knows.
constexpr uint64_t forGuest(uint64_t host, bool metalHasArchitecture)
{
	return metalHasArchitecture ? host : kX86Mac;
}

} // namespace rgfx_profile

#endif
