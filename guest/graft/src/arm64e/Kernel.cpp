// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// arm64e start-path points. The Boot kernel collection's slid header comes
// from the boot firmware's device-tree handover, /chosen/ambrosia/boot-kc-base,
// read from the IODeviceTree plane with SIP on. IOLog's calibration address is
// the plain value the kernel's symbol table records for it, read as a data
// symbol (below). Neither reads nor writes kernel memory beyond the registry, so
// this needs nothing from the write layer.
#include "../RuntimeImage.h"
#include <IOKit/IODeviceTreeSupport.h>
#include <IOKit/IORegistryEntry.h>
#include <libkern/c++/OSData.h>
#include <string.h>

// _IOLog as a data symbol, not the function. Its address is the unsigned GOT
// slot the loader binds to IOLog's raw address, so taking it is a plain load.
// Taking &IOLog as a function pointer would instead force a signed-pointer
// materialization whose authentication faults on a CPU with FEAT_FPAC, because
// the loader fills IOLog's data slot plain, not signed.
extern "C" const uint8_t graftIOLogSymbol __asm__("_IOLog");

namespace graft {

const mach_header_64 *bootCollectionHeader() {
    auto node = IORegistryEntry::fromPath("/chosen/ambrosia", gIODTPlane);
    if (!node)
        return nullptr;
    auto data = OSDynamicCast(OSData, node->getProperty("boot-kc-base"));
    const mach_header_64 *header = nullptr;
    if (data && data->getLength() == sizeof(uint64_t)) {
        uint64_t base = 0;
        memcpy(&base, data->getBytesNoCopy(), sizeof(base));
        header = macho::header(reinterpret_cast<const void *>(base));
    }
    node->release();
    return header;
}

uint64_t ioLogAddress() {
    return reinterpret_cast<uint64_t>(&graftIOLogSymbol);
}

} // namespace graft
