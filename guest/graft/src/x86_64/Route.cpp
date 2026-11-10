// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The x86-64 route installer: it reserves executable slots, refuses duplicates
// and overlaps, gates installation on early boot (before IOBSD), and drives
// the writes. It builds each route's bytes through X86Route.h and commits
// trampolines before entries. All instruction knowledge lives below this file;
// the slot bookkeeping and the fail-safe policy live here. Routing writes into
// kernel text, which is read-only once arm64 locks down, so the mechanism is
// x86-64's; src/arm64e/Route.cpp is the sibling for that architecture.
#include "Route.h"
#include "KernelWrite.h"
#include "Log.h"
#include "MachO.h"
#include "Policy.h"
#include "X86Route.h"
#include <IOKit/IOLib.h>
#include <IOKit/IOService.h>
#include <mach/vm_prot.h>
#include <stdarg.h>

// The two catalogue routes and an owned client's MAC dispatchers (Policy.h).
constexpr size_t RouteCapacity = graft::policy::RouteCapacity;
static_assert(RouteCapacity * graft::x86::MaxTrampoline == 2048, "executable route storage size");
extern "C" const uint8_t graft_route_storage[RouteCapacity][graft::x86::MaxTrampoline];
__asm__(".section __TEXT,__graft_routes,regular,pure_instructions\n"
        ".p2align 4\n"
        "_graft_route_storage:\n"
        ".fill 2048,1,0xcc\n"
        ".text\n");

namespace graft {
namespace {
const RuntimeImage *kernel;
IOLock *routeLock;
size_t used;
bool uncertain;
uint64_t installed[RouteCapacity];
size_t installedSize[RouteCapacity];
void **originalStorage[RouteCapacity];
x86::RoutePlan plans[RouteCapacity];
uint8_t oldTrampolines[RouteCapacity][x86::MaxTrampoline];
char refusal[192];

// Under routeLock: the reason a transaction is refused, logged and kept for
// the requesting client's state.
void refuse(const char *format, ...) __attribute__((format(printf, 1, 2)));
void refuse(const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(refusal, sizeof(refusal), format, arguments);
    va_end(arguments);
    note("route: refuse %s", refusal);
}

// Whether `address` lies in an executable segment of `client`'s fileset entry in
// the Boot collection. Unknown when the client has no name (Graft's own routes)
// or no entry here, so the replacement is left unchecked rather than refused.
enum class ClientText { Unknown, Inside, Outside };

ClientText replacementInClientText(const char *client, uint64_t address) {
    if (!client)
        return ClientText::Unknown;
    auto collection = bootCollectionHeader();
    uint64_t entry = collection ? macho::filesetEntry(collection, client) : 0;
    auto header = entry ? macho::header(reinterpret_cast<const void *>(entry)) : nullptr;
    auto text = header ? macho::segment(header, "__TEXT") : nullptr;
    if (!text)
        return ClientText::Unknown; // the client is not locatable in this collection
    uint64_t slide = entry - text->vmaddr;
    bool inside =
            macho::eachCommand(header, [&](const load_command *lc) {
                if (lc->cmd != LC_SEGMENT_64 || lc->cmdsize < sizeof(segment_command_64))
                    return false;
                auto segment = reinterpret_cast<const segment_command_64 *>(lc);
                if (!(segment->initprot & VM_PROT_EXECUTE) || segment->vmaddr > UINT64_MAX - slide)
                    return false;
                uint64_t start = segment->vmaddr + slide;
                return address >= start && address - start < segment->vmsize;
            }) != nullptr;
    return inside ? ClientText::Inside : ClientText::Outside;
}
} // namespace

bool initializeRoutes(const RuntimeImage &image) {
    routeLock = IOLockAlloc();
    if (!routeLock)
        return false;
    kernel = &image;
    return true;
}

bool routeKernel(const char *client, const RouteRequest *requests, size_t count) {
    if (!kernel || !routeLock || !requests || !count)
        return false;
    IOLockLock(routeLock);
    refusal[0] = 0;
    bool complete = count <= RouteCapacity - used;
    auto resources = IOService::getResourceService();
    if (uncertain) {
        refuse("installation after an unverified entry rollback");
        complete = false;
    } else if (resources && resources->getProperty("IOBSD")) {
        refuse("live installation after BSD publication");
        complete = false;
    } else if (!complete) {
        refuse("transaction: all %zu owned route slots are reserved", RouteCapacity);
    }
    Patch trampolines[RouteCapacity]{};
    Patch entries[RouteCapacity]{};
    void *prior[RouteCapacity]{};
    for (size_t i = 0; complete && i < count; ++i) {
        auto &request = requests[i];
        if (!request.symbol || !request.replacement || !request.original ||
            reinterpret_cast<uint64_t>(request.original) % sizeof(void *)) {
            refuse("invalid request %lu", i);
            complete = false;
            break;
        }
        // When the owned client's image is locatable, require its replacement to
        // lie in one of the client's executable segments, so a client that passes
        // a data pointer refuses here rather than faulting on the first call.
        if (replacementInClientText(client, reinterpret_cast<uint64_t>(request.replacement)) ==
            ClientText::Outside) {
            refuse("%s: replacement is not in the client's executable text", request.symbol);
            complete = false;
            break;
        }
        uint64_t source = kernel->resolve(request.symbol);
        uint64_t extent = source ? kernel->symbolExtent(source) : 0;
        size_t available = extent < x86::MaxDisplaced ? extent : x86::MaxDisplaced;
        for (size_t j = 0; j < used; ++j)
            if (installed[j] == source || originalStorage[j] == request.original)
                complete = false;
        for (size_t j = 0; j < i; ++j)
            if (reinterpret_cast<uint64_t>(entries[j].address) == source ||
                requests[j].original == request.original)
                complete = false;
        if (!complete || !source || !kernel->contains(source, available, VM_PROT_EXECUTE)) {
            refuse("%s: missing, duplicate or unreadable function", request.symbol);
            complete = false;
            break;
        }
        auto trampoline = graft_route_storage[used + i];
        auto failure = x86::buildRoute(reinterpret_cast<const uint8_t *>(source), available, source,
                                       reinterpret_cast<uint64_t>(trampoline),
                                       reinterpret_cast<uint64_t>(request.replacement), plans[i]);
        if (failure.error != x86::Error::None) {
            refuse("%s at +%lu: %s", request.symbol, failure.offset, x86::describe(failure.error));
            complete = false;
            break;
        }
        for (size_t j = 0; j < used; ++j) {
            if (source <= installed[j] ? plans[i].displacedSize > installed[j] - source
                                       : installedSize[j] > source - installed[j]) {
                refuse("%s: overlaps an installed entry", request.symbol);
                complete = false;
                break;
            }
        }
        for (size_t j = 0; complete && j < i; ++j) {
            auto other = reinterpret_cast<uint64_t>(entries[j].address);
            if (source <= other ? plans[i].displacedSize > other - source
                                : plans[j].displacedSize > source - other) {
                refuse("%s: overlaps another route in this request", request.symbol);
                complete = false;
            }
        }
        if (!complete)
            break;
        memcpy(oldTrampolines[i], trampoline, plans[i].trampolineSize);
        trampolines[i] = {const_cast<uint8_t *>(trampoline), oldTrampolines[i], plans[i].trampoline,
                          plans[i].trampolineSize};
        entries[i] = {reinterpret_cast<uint8_t *>(source), plans[i].original, plans[i].entry,
                      plans[i].displacedSize};
    }
    if (complete) {
        // Trampolines are permanent before any entry can use them. An entry
        // rollback must never restore a reachable trampoline to its prior
        // trap-filled storage while a routed entry might remain installed.
        auto error = applyWrites(trampolines, count);
        complete = error == WriteError::None;
        if (!complete) {
            refuse("trampoline preparation: write error %u", static_cast<unsigned>(error));
            // An unverified rollback of our own route storage is as anomalous
            // as one of a kernel entry: wall off all later routing, as the
            // entry path does, rather than reuse storage of unknown content.
            if (error == WriteError::Rollback)
                uncertain = true;
        }
    }
    if (complete) {
        // Publish callable originals before making the entry batch live.
        for (size_t i = 0; i < count; ++i) {
            prior[i] = __atomic_load_n(requests[i].original, __ATOMIC_ACQUIRE);
            __atomic_store_n(requests[i].original,
                             const_cast<uint8_t *>(graft_route_storage[used + i]),
                             __ATOMIC_RELEASE);
        }
        auto error = applyWrites(entries, count);
        complete = error == WriteError::None;
        if (!complete && error != WriteError::Rollback) {
            for (size_t i = 0; i < count; ++i)
                __atomic_store_n(requests[i].original, prior[i], __ATOMIC_RELEASE);
            refuse("transaction: write error %u", static_cast<unsigned>(error));
        } else {
            // A failed rollback may leave a routed entry. Preserve every
            // original pointer and its verified trampoline, and reserve the
            // entire batch so later requests cannot reuse uncertain state.
            for (size_t i = 0; i < count; ++i) {
                installed[used + i] = reinterpret_cast<uint64_t>(entries[i].address);
                installedSize[used + i] = entries[i].size;
                originalStorage[used + i] = requests[i].original;
                note("route: %s %s", requests[i].symbol,
                     complete ? "installed" : "retained after unverified rollback");
            }
            used += count;
            uncertain = !complete;
            if (uncertain)
                refuse("transaction: entry rollback unverified; replacement code and original "
                       "storage stay resident");
        }
    }
    char reason[sizeof(refusal)];
    if (complete)
        snprintf(reason, sizeof(reason), "%lu kernel routes installed", count);
    else
        strlcpy(reason, refusal, sizeof(reason));
    IOLockUnlock(routeLock);
    if (client)
        clientState(client, complete ? "routed" : "refused", reason);
    return complete;
}

} // namespace graft
