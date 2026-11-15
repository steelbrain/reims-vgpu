// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The arm64e branch islands (src/Island.h): there are none. Every arm64e import
// site is an eight-byte pointer, which reaches any address, so no plan asks for
// an island, and there is nothing to write or drop.
#include "../Island.h"

namespace graft {

uint64_t reserveIsland(uint64_t) {
    return 0;
}

uint64_t islandDestination(uint64_t) {
    return 0;
}

WriteError commitIslands() {
    return WriteError::None;
}

void discardIslands() {}

} // namespace graft
