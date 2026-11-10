// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The arm64e binding placement (src/BindingPlacement.h). The boot firmware places
// every image before Graft's kmod start, so there is nothing to route and nothing
// to wait for: the bind runs once. The Apple providers (IOGraphicsFamily and the
// libkern/IOKit base classes) stay in the Boot collection openKernel holds, so a
// provider is still its fileset entry there; this placement uses no OSKext and no
// started gating, since a placed entry is already present. The kexts Graft serves,
// the consumer and the owned client, land in the Auxiliary collection instead, so
// each is read by bundle identifier from the boot firmware's per-kext device-tree
// node.
#include "BindingPlacement.h"
#include "MachO.h"
#include <IOKit/IODeviceTreeSupport.h>
#include <IOKit/IOLib.h>
#include <IOKit/IORegistryEntry.h>
#include <libkern/c++/OSData.h>
#include <string.h>

#if !defined(__arm64__)
#error Graft's arm64e binding placement requires the arm64e architecture implementation
#endif

namespace graft {

bool installBindingTrigger(const RuntimeImage &, void (*)()) {
    return true;
}

Placement locate(const mach_header_64 *collection, const char *identifier, char *reason,
                 size_t reasonSize) {
    Placement placement;
    placement.base = macho::filesetEntry(collection, identifier);
    if (!placement.base)
        snprintf(reason, reasonSize, "%s: not a collection entry", identifier);
    return placement;
}

bool kextStarted(const char *) {
    return false;
}

uint64_t placedBase(const mach_header_64 *, const char *identifier) {
    // The boot firmware names each placed kext's device-tree node by its
    // CFBundleIdentifier under /chosen/ambrosia, with base the slid mach_header.
    char path[192];
    if (snprintf(path, sizeof(path), "/chosen/ambrosia/%s", identifier) >=
        static_cast<int>(sizeof(path)))
        return 0;
    auto node = IORegistryEntry::fromPath(path, gIODTPlane);
    if (!node)
        return 0;
    uint64_t base = 0;
    auto data = OSDynamicCast(OSData, node->getProperty("base"));
    if (data && data->getLength() == sizeof(base))
        memcpy(&base, data->getBytesNoCopy(), sizeof(base));
    node->release();
    return base;
}

bool writableConsumerTarget(const RuntimeImage &consumer, uint64_t address, uint64_t size) {
    // arm64e has no write window: the store goes straight to the consumer's own
    // memory, which faults if the kernel has locked the segment read-only. Refuse
    // a target outside a writable, non-read-only data segment.
    return consumer.writableData(address, size);
}

} // namespace graft
