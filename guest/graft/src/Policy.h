// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The policy constants that pin Graft to our own kexts. Graft serves only
// these and offers no general plugin interface: one consumer it binds against
// the System KC, one owned client it routes for, our device's PCI match, and
// the fixed route capacities. Gathering the identities here keeps them out of
// the mechanism and in one place to review. To serve a different kext, rebuild
// Graft with this file changed.
#pragma once
#include <stddef.h>

namespace graft {
namespace policy {

// The consumer Graft binds by name once its manifest passes every check:
// ReimsGraphics, injected with its personalities gated by an IOPropertyMatch
// until Graft binds it and sets the gate property on the device nub. The SDK
// stages the gate, checks the personalities name our device, and records it in
// the manifest, so Graft finds the nub with no device constant of its own.
constexpr const char *ConsumerIdentifier = "ai.aneesiqbal.ReimsGraphics";

// The owned client Graft serves directly: RgfxGate, which installs the Metal
// bundle and routes its file-system checks through Graft.
constexpr const char *OwnedClientIdentifier = "ai.aneesiqbal.RgfxGate";

// Apple's Mobile File Integrity kext, whose MAC policy is the one an owned
// client's admission routes shadow on arm64 (src/arm64e/Route.cpp). This is the
// public bundle identifier, used to find AMFI's image in the Boot collection and
// cross-validate the policy-list element whose ops it carries; it is not derived
// from any Apple binary.
constexpr const char *AmfiIdentifier = "com.apple.driver.AppleMobileFileIntegrity";

// The most kernel functions an owned client may route in one request:
// RgfxGate's six MAC dispatchers.
constexpr size_t MaxClientRoutes = 6;

// Graft's own executable route slots: the two catalogue-send entries plus an
// owned client's MaxClientRoutes.
constexpr size_t RouteCapacity = 2 + MaxClientRoutes;

} // namespace policy
} // namespace graft
