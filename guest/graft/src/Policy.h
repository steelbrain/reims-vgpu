// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The policy constants of Graft's generic mechanism, in one place to review.
// A client is accepted when it is a placed kext (src/Start.cpp), so no kext's
// identifier is named here but Apple's own: the MAC policies the arm64e route
// mechanism can shadow (src/arm64e/Route.cpp), AMFI by default and Sandbox when
// a client names it. The route limits the installers enforce are here too.
#pragma once
#include <stddef.h>

namespace graft {
namespace policy {

// Apple's Mobile File Integrity kext, whose MAC policy a client's admission
// routes shadow on arm64 by default (src/arm64e/Route.cpp). This is the public
// bundle identifier, used to find AMFI's image in the Boot collection and
// cross-validate the policy-list element whose ops it carries; it is not derived
// from any Apple binary.
constexpr const char *AmfiIdentifier = "com.apple.driver.AppleMobileFileIntegrity";

// Apple's Sandbox kext, whose MAC policy (registered name "Sandbox") a client
// can route on arm64 by naming it, the same way it routes AMFI's. Its public
// bundle identifier locates its image in the Boot collection, where the policy's
// conf and ops must lie; like AMFI's, it is not derived from any Apple binary.
constexpr const char *SandboxIdentifier = "com.apple.security.sandbox";

// The most kernel functions one route request may carry together. A request
// that asks for more is refused whole; the shared pool below still bounds how
// many install across every client.
constexpr size_t MaxRequestRoutes = 16;

// Graft's executable route slots on x86-64, shared first come first served: its
// own two catalogue-send entries plus up to thirty more across all clients.
constexpr size_t RouteCapacity = 32;

} // namespace policy
} // namespace graft
