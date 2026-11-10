// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The x86-64 binding placement (src/BindingPlacement.h). A consumer and its
// providers register as kexts, so Graft finds a provider through OSKext (it must
// be registered and started) and the live kernel collection's fileset, with the
// pageable collection as a fallback, and retries the bind on each catalogue
// personality send by routing the two catalogue entry points. The dormant
// consumer is found by its fileset entry in the Boot collection (placedBase).
#include "BindingPlacement.h"
#include "Log.h"
#include "MachO.h"
#include "Route.h"
#include <IOKit/IOLib.h>
#include <libkern/OSKextLib.h>
#include <libkern/c++/OSArray.h>
#include <libkern/c++/OSBoolean.h>
#include <libkern/c++/OSDictionary.h>
#include <libkern/c++/OSString.h>
#include <mach/vm_prot.h>
#include <pexpert/pexpert.h>

#if !defined(__x86_64__)
#error Graft's x86-64 binding placement requires the x86-64 architecture implementation
#endif

namespace graft {
namespace {
using Lookup = OSObject *(*)(const char *);
using CopyInfo = OSDictionary *(*)(OSArray *, OSArray *);
using Send = void (*)(bool);
using Copy = OSArray *(*)(bool);
Lookup lookup;
CopyInfo copyInfo;
void *originalSend;
void *originalCopy;
void (*boundTrigger)();

OSDictionary *information(const char *identifier) {
    auto name = OSString::withCString(identifier);
    const OSObject *objects[] = {name};
    auto ids = name ? OSArray::withObjects(objects, 1) : nullptr;
    auto all = ids ? copyInfo(ids, nullptr) : nullptr;
    auto info = all ? OSDynamicCast(OSDictionary, all->getObject(identifier)) : nullptr;
    if (info)
        info->retain();
    OSSafeReleaseNULL(all);
    OSSafeReleaseNULL(ids);
    OSSafeReleaseNULL(name);
    return info;
}

bool started(OSDictionary *info) {
    auto flag = info ? OSDynamicCast(OSBoolean, info->getObject("OSBundleStarted")) : nullptr;
    return flag && flag->isTrue();
}

// The routed catalogue entry points: retry the bind, then call the original.
void onSend(bool match) {
    boundTrigger();
    reinterpret_cast<Send>(__atomic_load_n(&originalSend, __ATOMIC_ACQUIRE))(match);
}
OSArray *onCopy(bool filter) {
    boundTrigger();
    return reinterpret_cast<Copy>(__atomic_load_n(&originalCopy, __ATOMIC_ACQUIRE))(filter);
}
} // namespace

bool installBindingTrigger(const RuntimeImage &kernel, void (*onBind)()) {
    boundTrigger = onBind;
    lookup = reinterpret_cast<Lookup>(kernel.resolve("__ZN6OSKext24lookupKextWithIdentifierEPKc"));
    copyInfo = reinterpret_cast<CopyInfo>(
            kernel.resolve("__ZN6OSKext18copyLoadedKextInfoEP7OSArrayS1_"));
    if (!lookup || !copyInfo) {
        note("binding: refuse OSKext lookup/information: required symbols missing");
        return false;
    }
    RouteRequest requests[] = {{"__ZN6OSKext24copyAllKextPersonalitiesEb",
                                reinterpret_cast<void *>(onCopy), &originalCopy},
                               {"__ZN6OSKext33sendAllKextPersonalitiesToCatalogEb",
                                reinterpret_cast<void *>(onSend), &originalSend}};
    return routeKernel(nullptr, requests, sizeof(requests) / sizeof(requests[0]));
}

Placement locate(const mach_header_64 *collection, const char *identifier, char *reason,
                 size_t reasonSize) {
    Placement placement;
    placement.hold = lookup(identifier);
    if (!placement.hold) {
        placement.waiting = true;
        snprintf(reason, reasonSize, "%s is not registered", identifier);
        return placement;
    }
    auto result = OSKextLoadKextWithIdentifier(identifier);
    auto info = information(identifier);
    bool running = started(info);
    OSSafeReleaseNULL(info);
    if (result != kOSReturnSuccess || !running) {
        placement.waiting = true;
        snprintf(reason, reasonSize, "%s is not started (load %#x)", identifier, result);
        return placement;
    }
    // Collection entries identify the live header. User-facing kext information
    // is not a contract for a dereferenceable kernel address.
    uint64_t address = macho::filesetEntry(collection, identifier);
    if (!address) {
        auto pageable = macho::header(PE_get_kc_header(KCKindPageable));
        if (pageable && pageable->sizeofcmds > macho::MaxLoadCommands)
            note("binding: skip pageable collection: load commands exceed the 64 KiB cap");
        else if (pageable)
            address = macho::filesetEntry(pageable, identifier);
    }
    if (!address) {
        snprintf(reason, reasonSize, "%s: no registered collection entry", identifier);
        return placement;
    }
    placement.base = address;
    return placement;
}

bool kextStarted(const char *identifier) {
    auto info = information(identifier);
    bool active = started(info);
    OSSafeReleaseNULL(info);
    return active;
}

uint64_t placedBase(const mach_header_64 *collection, const char *identifier) {
    // OpenCore injects the consumer into the Boot collection, so it is a fileset
    // entry there.
    return macho::filesetEntry(collection, identifier);
}

bool writableConsumerTarget(const RuntimeImage &consumer, uint64_t address, uint64_t size) {
    // x86-64 binds branch sites in __TEXT, so a mapped, readable target is allowed;
    // the write window clears CR0.WP for the store.
    return consumer.contains(address, size, VM_PROT_READ);
}

} // namespace graft
