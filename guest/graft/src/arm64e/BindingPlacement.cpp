// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The arm64e binding placement (src/BindingPlacement.h). The boot firmware places
// every image before Graft's kmod start, so there is nothing to route and nothing
// to wait for: the binds run once. The Apple providers stay in the Boot collection
// openKernel holds, so a provider is its fileset entry there. A placed kext is not
// started until a driver matches in it or a started kext depends on it, and its
// classes register only as it starts, so Graft starts each provider through
// OSKextLoadKextWithIdentifier, as the x86-64 placement does. The kexts Graft
// serves, its consumers and clients, land in the Auxiliary collection instead, so
// each is read from the boot firmware's per-kext device-tree node, named by its
// bundle identifier.
#include "BindingPlacement.h"
#include "MachO.h"
#include <IOKit/IODeviceTreeSupport.h>
#include <IOKit/IOLib.h>
#include <IOKit/IORegistryEntry.h>
#include <libkern/OSKextLib.h>
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
    uint64_t base = macho::filesetEntry(collection, identifier);
    if (!base) {
        snprintf(reason, reasonSize, "%s: not a collection entry", identifier);
        return placement;
    }
    // Start the provider, as the kernel starts a kext's dependencies. One that
    // is already running returns success at once. There is no later attempt on
    // arm64e, so a failed start refuses the consumer.
    auto result = OSKextLoadKextWithIdentifier(identifier);
    if (result != kOSReturnSuccess) {
        snprintf(reason, reasonSize, "%s is not started (load %#x)", identifier, result);
        return placement;
    }
    placement.base = base;
    return placement;
}

bool kextStarted(const char *) {
    return false;
}

namespace {
// A placed kext's slid mach_header, its device-tree node's 8-byte base; 0 for a
// node with none, such as vtable-disc.
uint64_t nodeBase(IORegistryEntry *node) {
    uint64_t base = 0;
    auto data = OSDynamicCast(OSData, node->getProperty("base"));
    if (data && data->getLength() == sizeof(base))
        memcpy(&base, data->getBytesNoCopy(), sizeof(base));
    return base;
}
} // namespace

void eachPlacedKext(const mach_header_64 *,
                    void (*visit)(void *context, const char *identifier, uint64_t base),
                    void *context) {
    // The boot firmware names each placed kext's device-tree node by its
    // CFBundleIdentifier under /chosen/ambrosia, with base the slid mach_header.
    auto root = IORegistryEntry::fromPath("/chosen/ambrosia", gIODTPlane);
    auto children = root ? root->getChildIterator(gIODTPlane) : nullptr;
    while (auto child =
                   children ? OSDynamicCast(IORegistryEntry, children->getNextObject()) : nullptr) {
        uint64_t base = nodeBase(child);
        auto name = child->getName(gIODTPlane);
        if (base && name)
            visit(context, name, base);
    }
    OSSafeReleaseNULL(children);
    OSSafeReleaseNULL(root);
}

uint64_t placedBase(const mach_header_64 *, const char *identifier) {
    char path[192];
    if (snprintf(path, sizeof(path), "/chosen/ambrosia/%s", identifier) >=
        static_cast<int>(sizeof(path)))
        return 0;
    auto node = IORegistryEntry::fromPath(path, gIODTPlane);
    if (!node)
        return 0;
    uint64_t base = nodeBase(node);
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
