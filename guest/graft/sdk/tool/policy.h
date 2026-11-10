// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The device a bound consumer serves. Staging requires every personality to
// name it, and records it in the manifest, from which Graft finds the PCI nub
// to open the consumer's activation gate on; Graft holds no device of its own.
// Serving another device means changing this match.
#pragma once

namespace graft::tool {

// The IOPCIPrimaryMatch of the paravirtual GPU Graft serves, PCI 1234:5247:
// the device ID above the vendor ID.
constexpr const char *DeviceMatch = "0x52471234";

} // namespace graft::tool
