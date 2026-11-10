// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// x86-64 start-path points: the Boot kernel collection from the kernel's own
// accessor, and IOLog's running address as its plain linked value, since
// x86-64 does not sign pointers. openKernel (RuntimeImage.cpp) calls these.
#include "../RuntimeImage.h"
#include <IOKit/IOLib.h>
#include <pexpert/pexpert.h>

namespace graft {

const mach_header_64 *bootCollectionHeader() {
    return macho::header(PE_get_kc_header(KCKindPrimary));
}

uint64_t ioLogAddress() {
    return reinterpret_cast<uint64_t>(&IOLog);
}

} // namespace graft
