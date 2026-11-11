// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Graft's kmod entry point and the C interface its clients call (Graft.h).
// graft_start initializes diagnostics, the kernel image, the write and route
// facilities and the catalogue binding from the kmod load, before the Graft
// service starts. A call's identifier is accepted when it names a placed kext
// (placedBase), whether or not Graft is active, so a client can record that
// Graft refused; the functions that act on the kernel additionally require Graft
// active. The architecture-neutral core does the work.
#include "BindingPlacement.h"
#include "KernelBinding.h"
#include "KernelWrite.h"
#include "Log.h"
#include "Policy.h"
#include "Route.h"
#include <IOKit/IOLib.h>
#include <IOKit/IOService.h>
#include <graft/Graft.h>
#include <mach/mach_types.h>

// A resource personality makes the injected kext start even when no consumer
// is present. Initialization remains in graft_start, before this service's
// ordinary IOService start; it needs neither BSD nor a boot framebuffer.
class Graft final : public IOService {
    OSDeclareDefaultStructors(Graft)
};
OSDefineMetaClassAndStructors(Graft, IOService);

namespace {
graft::RuntimeImage kernel;
const mach_header_64 *bootCollection;
bool ready;

// A call's identifier is accepted when it names a kext the boot linker placed:
// on x86-64 a fileset entry in the Boot collection, on arm64e a device-tree node
// the firmware wrote (graft::placedBase). A placed identifier is what makes
// every GraftClient.<id> registry key name a real kext. graft_start keeps the
// Boot collection before anything can refuse, so this holds when Graft is not
// active too; on x86-64 it needs the collection's load commands to validate,
// while arm64e reads the device tree instead.
bool placedClient(const char *identifier) {
    auto collection = __atomic_load_n(&bootCollection, __ATOMIC_ACQUIRE);
    return identifier && graft::placedBase(collection, identifier) != 0;
}
} // namespace

extern "C" bool graft_client_ready(const char *identifier) {
    return __atomic_load_n(&ready, __ATOMIC_ACQUIRE) && placedClient(identifier);
}

extern "C" bool graft_kernel_symbol(const char *identifier, const char *symbol, uint64_t *address) {
    if (!address)
        return false;
    *address = 0;
    if (!graft_client_ready(identifier) || !symbol)
        return false;
    *address = kernel.resolve(symbol);
    if (!*address)
        graft::clientState(identifier, "refused", symbol);
    return *address != 0;
}

extern "C" bool graft_route_kernel(const char *identifier, const graft_route_request *requests,
                                   size_t count) {
    if (!graft_client_ready(identifier) || !requests || !count ||
        count > graft::policy::MaxRequestRoutes)
        return false;
    graft::RouteRequest routes[graft::policy::MaxRequestRoutes]{};
    for (size_t i = 0; i < count; ++i)
        routes[i] = {requests[i].symbol, requests[i].replacement, requests[i].original};
    return graft::routeKernel(identifier, routes, count);
}

extern "C" void graft_client_log(const char *identifier, const char *format, ...) {
    if (!placedClient(identifier) || !format)
        return;
    char line[320];
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(line, sizeof(line), format, arguments);
    va_end(arguments);
    graft::note("%s: %s", identifier, line);
}

extern "C" void graft_client_state(const char *identifier, const char *state, const char *reason) {
    if (placedClient(identifier) && state && reason)
        graft::clientState(identifier, state, reason);
}

extern "C" kern_return_t graft_start(kmod_info_t *, void *) {
    if (!graft::initializeLog()) {
        IOLog("graft: refused: cannot allocate diagnostics\n");
        return KERN_SUCCESS;
    }
    graft::setState("starting", "own kmod entry");
    const mach_header_64 *collection = nullptr;
    bool opened = graft::openKernel(kernel, collection);
    // Keep the Boot collection for placedClient, even if Graft refuses below: it
    // tells a placed kext's identifier from any other, so every served call names
    // real placed code, and a client can still publish that Graft is not active.
    __atomic_store_n(&bootCollection, collection, __ATOMIC_RELEASE);
    if (!opened) {
        graft::setState("refused", "kernel collection or calibrated symbols unavailable");
        return KERN_SUCCESS;
    }
    if (!graft::initializeWrites(kernel) || !graft::initializeRoutes(kernel)) {
        graft::setState("refused", "kernel write/routing facilities unavailable");
        return KERN_SUCCESS;
    }
    if (!graft::initializeBinding(kernel, collection)) {
        graft::setState("refused", "catalogue binding routes unavailable");
        return KERN_SUCCESS;
    }
    __atomic_store_n(&ready, true, __ATOMIC_RELEASE);
    graft::setState("active", "catalogue binding and client routing initialized");
    graft::bindConsumers();
    return KERN_SUCCESS;
}

extern "C" kern_return_t graft_stop(kmod_info_t *, void *) {
    return KERN_FAILURE;
}

extern "C" {
KMOD_EXPLICIT_DECL(ai.aneesiqbal.graft, GRAFT_VERSION, graft_start, graft_stop)
}
