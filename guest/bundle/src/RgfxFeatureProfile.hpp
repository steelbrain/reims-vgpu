// The feature profile the device answers (`featureProfile`, SPI). At device creation Metal reads
// it before anything else and keys three things on it: the device's limits, its GPU families and
// its feature queries. A guest's Metal that has no limits for the profile it reads leaves every
// limit 0 and ends the process at its first check of one. WindowServer is that process, so the
// guest has no desktop.
//
// The device answers the x86 Macs' profile, which the guest's Metal knows, on every host. Under
// macOS 26.7.1 both of the oracle's GPUs answer 10002 (the Radeon Pro 5300M and the UHD 630;
// measured, docs/graphics-kext/dgpu-oracle.md, Q11), and under macOS 12 both of their drivers
// answer 10001 (AMDRadeonX6000's and AppleIntelKBLGraphics'; read from 12.7.6's drivers), which
// 12's Metal has limits for (it has none for the M3 Max's 10: 12.7.6's WindowServer ends at
// device creation under it). macOS 11's Metal is taken to be 12's here: its boots stop in the
// kernel before WindowServer (Graft's to fix).
//
// Which of the two the guest's Metal knows is told by what it has, not by its version: Metal that
// has the MTLArchitecture class (macOS 13 and later) gets 10002, under which 26.7.2 boots to its
// desktop, and Metal without it (macOS 11 and 12) gets 10001. Under 10002, macOS 26's WindowServer
// asks for tiled textures of 6 and 7 levels over a buffer, which the host makes (RgfxBuffer.mm).

#ifndef RGFX_FEATURE_PROFILE_HPP
#define RGFX_FEATURE_PROFILE_HPP

#include <cstdint>

namespace rgfx_profile {

// The profile macOS 11 and 12's x86 GPU drivers answer.
constexpr uint64_t kX86Mac12 = 10001;

// The profile macOS 26's x86 GPU drivers answer.
constexpr uint64_t kX86Mac = 10002;

// The profile the device answers: the x86 Macs' that the guest's Metal knows, 10002 where it has
// the MTLArchitecture class, else 10001.
constexpr uint64_t forGuest(bool metalHasArchitecture)
{
	return metalHasArchitecture ? kX86Mac : kX86Mac12;
}

} // namespace rgfx_profile

#endif
