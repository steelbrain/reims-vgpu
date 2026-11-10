// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The arm64e route installer (src/Route.h's interface). On x86-64 Graft routes a
// client's kernel entries through trampolines it writes into kernel text. On
// arm64 the collection's text and AMFI's policy ops table are read-only when
// Graft runs (its kmod start is after machine lockdown), so an owned client's
// admission hooks are routed by shadowing AMFI's MAC policy: Graft copies AMFI's
// mac_policy_ops and mac_policy_conf into its own writable memory, re-signs the
// routed op slots in the shadow ops for the client's replacements, points the
// shadow conf's mpc_ops at the shadow ops, and repoints the one writable
// mac_policy_list element whose mpc is AMFI's conf at the shadow conf. That
// single element store is the only live kernel write; both pointers it touches
// are stored raw, so only the op function pointers are re-signed. The plan
// arithmetic and the ABI constants are the pure, host-tested Arm64Route.h; this
// file owns resolution, the live reads and the fail-safe transaction. The x86-64
// installer is src/x86_64/Route.cpp.
#include "Route.h"
#include "Arm64Route.h"
#include "BindingPlacement.h"
#include "Fixup.h"
#include "Log.h"
#include "MachO.h"
#include "Policy.h"
#include <IOKit/IOLib.h>
#include <mach/vm_prot.h>
#include <stdarg.h>
#include <string.h>

#if !defined(__arm64__)
#error Graft's arm64e routing requires the arm64e architecture implementation
#endif

namespace graft {
namespace {
const RuntimeImage *kernel;
IOLock *routeLock;
bool used; // the owned client's shadow is installed at most once
char refusal[192];

// 8-aligned writable backing for the shadow ops and conf, in Graft's own __DATA
// (the Auxiliary collection maps Graft read-write). Because AMFI signs every ops
// slot with no address diversity, the wholesale copy keeps the untouched slots'
// signatures valid at this new address, and the shadow lives for the routes'
// lifetime.
static_assert(arm64::OpsSize % sizeof(uint64_t) == 0, "shadow ops alignment");
static_assert(arm64::ConfSize % sizeof(uint64_t) == 0, "shadow conf alignment");
uint64_t shadowOps[arm64::OpsSize / sizeof(uint64_t)];
uint64_t shadowConf[arm64::ConfSize / sizeof(uint64_t)];

void refuse(const char *format, ...) __attribute__((format(printf, 1, 2)));
void refuse(const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(refusal, sizeof(refusal), format, arguments);
    va_end(arguments);
    note("route: refuse %s", refusal);
}

// A canonical, 8-aligned kernel pointer: the gate before dereferencing a policy
// structure resolved from live memory, so a mis-resolution refuses rather than
// faults on a wild address.
bool canonicalKernel(uint64_t pointer) {
    return (pointer & 0xFFFF000000000000ULL) == 0xFFFF000000000000ULL &&
           pointer % sizeof(uint64_t) == 0;
}

uint64_t readPointer(uint64_t address) {
    uint64_t value;
    memcpy(&value, reinterpret_cast<const void *>(address), sizeof(value));
    return value;
}

// A fileset entry's runtime placement in the collection. The entry's mach_header
// sits at __TEXT's linked base, so the collection slide is the difference, and
// each segment's runtime span is its vmaddr plus the slide. ok is false when the
// identifier has no entry in this collection.
struct ImageSpan {
    const mach_header_64 *header;
    uint64_t slide;
    bool ok;
};

ImageSpan imageSpan(const mach_header_64 *collection, const char *identifier) {
    ImageSpan span{nullptr, 0, false};
    if (!collection || !identifier)
        return span;
    uint64_t entry = macho::filesetEntry(collection, identifier);
    span.header = entry ? macho::header(reinterpret_cast<const void *>(entry)) : nullptr;
    auto text = span.header ? macho::segment(span.header, "__TEXT") : nullptr;
    if (text) {
        span.slide = entry - text->vmaddr;
        span.ok = true;
    }
    return span;
}

// Whether [address, address+size) lies wholly within one of the image's
// segments. A policy structure resolved from live memory is required to pass this
// against AMFI's image before it is dereferenced, so a mis-resolution refuses
// rather than faults on a wild address.
bool spanContains(const ImageSpan &span, uint64_t address, size_t size) {
    if (!span.ok || !size || size - 1 > UINT64_MAX - address)
        return false;
    return macho::eachCommand(span.header, [&](const load_command *lc) {
               if (lc->cmd != LC_SEGMENT_64 || lc->cmdsize < sizeof(segment_command_64))
                   return false;
               auto segment = reinterpret_cast<const segment_command_64 *>(lc);
               if (segment->vmaddr > UINT64_MAX - span.slide)
                   return false;
               uint64_t start = segment->vmaddr + span.slide;
               return address >= start && address - start <= segment->vmsize &&
                      size <= segment->vmsize - (address - start);
           }) != nullptr;
}

// AMFI's executable address range in the running collection. The op pointers AMFI
// signs point here; Graft uses the range both to find AMFI's policy-list element
// and as the shadow-ops guard.
bool amfiExecRange(const mach_header_64 *collection, uint64_t &low, uint64_t &high) {
    ImageSpan amfi = imageSpan(collection, policy::AmfiIdentifier);
    if (!amfi.ok)
        return false;
    bool any = false;
    low = high = 0;
    macho::eachCommand(amfi.header, [&](const load_command *lc) {
        if (lc->cmd == LC_SEGMENT_64 && lc->cmdsize >= sizeof(segment_command_64)) {
            auto segment = reinterpret_cast<const segment_command_64 *>(lc);
            if (segment->initprot & VM_PROT_EXECUTE) {
                uint64_t start = segment->vmaddr + amfi.slide;
                uint64_t end = start + segment->vmsize;
                if (!any || start < low)
                    low = start;
                if (!any || end > high)
                    high = end;
                any = true;
            }
        }
        return false;
    });
    return any && low < high;
}

// Whether the policy-list element at `address` lies in writable kernel memory.
// The element is an 8-byte pointer in struct mac_policy_list's entries. When it
// lies in the kernel image, a store needs a segment that is writable and not
// marked read-only (SG_READ_ONLY, which the kernel has locked by the time Graft
// starts). An element the kernel allocated outside its image is reported
// NotInImage and is not checked further; the swap's compare-and-exchange guards
// the value it replaces, not the permission to store.
enum class Writability { NotInImage, Writable, ReadOnly };

Writability elementWritability(uint64_t address) {
    Writability result = Writability::NotInImage;
    macho::eachCommand(kernel->header, [&](const load_command *lc) {
        if (lc->cmd != LC_SEGMENT_64 || lc->cmdsize < sizeof(segment_command_64))
            return false;
        auto segment = reinterpret_cast<const segment_command_64 *>(lc);
        if (segment->vmaddr > UINT64_MAX - kernel->segmentBias)
            return false;
        uint64_t start = segment->vmaddr + kernel->segmentBias;
        if (address < start || address - start >= segment->vmsize)
            return false;
        result = (segment->initprot & VM_PROT_WRITE) && !(segment->flags & SG_READ_ONLY)
                         ? Writability::Writable
                         : Writability::ReadOnly;
        return true; // the containing segment decides
    });
    return result;
}

// AMFI's policy-list element, its conf and its ops, found by resolving the
// symbol-less mac_policy_list from mac_policy_register and scanning its populated
// entries for the one policy whose ops implements both library validation and
// code-signature checking into AMFI's own text -- AMFI alone does on a stock
// guest. The list global itself must lie in one of the kernel's data segments,
// and each candidate conf and ops table inside AMFI's image, before Graft reads
// it; the scan is bounded by the list's own max. The entries array is heap
// memory the kernel allocated, outside any image, so Graft has no bounds to
// check it against: it requires only a canonical, aligned kernel pointer before
// reading the elements, and a wrong pointer there faults rather than refuses.
// Refuses on zero or more than one match rather than guess. Returns the
// element's mpc-field address (the repoint target) and AMFI's conf and ops
// addresses.
bool findAmfiPolicy(const mach_header_64 *collection, uint64_t textLow, uint64_t textHigh,
                    uint64_t &elementMpc, uint64_t &conf, uint64_t &ops) {
    uint64_t reg = kernel->resolve("_mac_policy_register");
    uint64_t extent = reg ? kernel->symbolExtent(reg) : 0;
    if (!reg || !extent || !kernel->contains(reg, extent < 256 ? extent : 256, VM_PROT_EXECUTE)) {
        refuse("mac_policy_register is not resolvable");
        return false;
    }
    uint64_t listBase =
            arm64::resolvePolicyListBase(reinterpret_cast<const uint8_t *>(reg), extent, reg);
    if (!listBase) {
        refuse("mac_policy_list is not resolvable");
        return false;
    }
    // The kernel updates the list as policies register, so the global lies in a
    // segment it maps writable. Requiring that keeps a misresolved base in the
    // kernel's code or constants from being read as the list.
    if (!kernel->contains(listBase, arm64::PolicyListEntriesOffset + sizeof(uint64_t),
                          VM_PROT_READ | VM_PROT_WRITE)) {
        refuse("mac_policy_list is not in the kernel's data segments");
        return false;
    }
    uint32_t max, maxindex;
    memcpy(&max, reinterpret_cast<const void *>(listBase + arm64::PolicyListMaxOffset),
           sizeof(max));
    memcpy(&maxindex, reinterpret_cast<const void *>(listBase + arm64::PolicyListMaxIndexOffset),
           sizeof(maxindex));
    uint64_t entries = readPointer(listBase + arm64::PolicyListEntriesOffset);
    if (!canonicalKernel(entries)) {
        refuse("mac_policy_list entries pointer is invalid");
        return false;
    }
    const ImageSpan amfi = imageSpan(collection, policy::AmfiIdentifier);
    const uint32_t lvOffset = arm64::operationFor("file_check_library_validation")->offset;
    const uint32_t svOffset = arm64::operationFor("vnode_check_signature")->offset;
    size_t scan = arm64::policyScanCount(maxindex, max);
    size_t matches = 0;
    for (size_t i = 0; i < scan; ++i) {
        uint64_t element = entries + i * arm64::PolicyElementStride;
        uint64_t candidateConf = readPointer(element + arm64::PolicyElementMpcOffset);
        if (!canonicalKernel(candidateConf) || !spanContains(amfi, candidateConf, arm64::ConfSize))
            continue;
        uint64_t candidateOps = readPointer(candidateConf + arm64::ConfOpsOffset);
        if (!canonicalKernel(candidateOps) || !spanContains(amfi, candidateOps, arm64::OpsSize))
            continue;
        uint64_t validation = fixup::stripCode(readPointer(candidateOps + lvOffset));
        uint64_t signature = fixup::stripCode(readPointer(candidateOps + svOffset));
        if (validation < textLow || validation >= textHigh || signature < textLow ||
            signature >= textHigh)
            continue;
        ++matches;
        elementMpc = element + arm64::PolicyElementMpcOffset;
        conf = candidateConf;
        ops = candidateOps;
    }
    if (matches != 1) {
        refuse(matches ? "AMFI policy ambiguous in the list (%zu matches)"
                       : "AMFI policy not found in the list",
               matches);
        return false;
    }
    return true;
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
    bool ok = true;
    const char *names[arm64::OperationCount] = {};
    uint64_t targets[arm64::OperationCount] = {};
    if (used) {
        refuse("the owned client's routes are already installed");
        ok = false;
    } else if (count > arm64::OperationCount) {
        refuse("transaction: %zu requests exceed the %zu routable operations", count,
               arm64::OperationCount);
        ok = false;
    }
    // No BSD-publication gate here, unlike x86-64's kernel-text trampolines
    // (src/x86_64/Route.cpp). The arm64 route goes live by one atomic
    // compare-and-exchange of a single policy-list element, from AMFI's conf to
    // the fully built shadow conf; a concurrent MAC hook reads one valid conf or
    // the other, never a torn value, so the install is safe after BSD starts. An
    // owned client that matches as an IOKit driver necessarily starts post-BSD,
    // so gating on it would refuse every arm64 client.
    for (size_t i = 0; ok && i < count; ++i) {
        auto &request = requests[i];
        if (!request.symbol || !request.replacement || !request.original ||
            reinterpret_cast<uintptr_t>(request.original) % sizeof(void *)) {
            refuse("invalid request %zu", i);
            ok = false;
            break;
        }
        for (size_t j = 0; j < i; ++j)
            if (requests[j].original == request.original) {
                refuse("request %zu: duplicate original storage", i);
                ok = false;
                break;
            }
        names[i] = request.symbol;
        targets[i] = reinterpret_cast<uint64_t>(request.replacement);
    }

    const mach_header_64 *collection = ok ? bootCollectionHeader() : nullptr;
    uint64_t textLow = 0, textHigh = 0;
    if (ok && (!collection || !amfiExecRange(collection, textLow, textHigh))) {
        refuse("AMFI image or its executable range not found");
        ok = false;
    }
    // Require each replacement to lie in one of the owned client's executable
    // segments, so a client that passes a data pointer refuses here rather than
    // panicking when AMFI's dispatch first branches to the shadow slot. The
    // firmware places the client in the Auxiliary collection and names its image
    // by bundle identifier, as it names the consumer's (placedBase). A
    // replacement is a signed function pointer, so its stripped address is
    // compared. A client the firmware names no image for is left unchecked, as
    // on x86-64.
    uint64_t clientBase = ok && client ? placedBase(collection, client) : 0;
    RuntimeImage clientImage;
    if (clientBase && !clientImage.map(reinterpret_cast<const mach_header_64 *>(clientBase))) {
        refuse("the client's placed image cannot be read (%s)", clientImage.error());
        ok = false;
    }
    for (size_t i = 0; ok && clientBase && i < count; ++i)
        if (!clientImage.contains(fixup::stripCode(targets[i]), 1, VM_PROT_EXECUTE)) {
            refuse("request %zu: replacement is not in the client's executable text", i);
            ok = false;
        }
    uint64_t elementMpc = 0, amfiConf = 0, amfiOps = 0;
    if (ok && !findAmfiPolicy(collection, textLow, textHigh, elementMpc, amfiConf, amfiOps))
        ok = false;

    arm64::Slot slots[arm64::OperationCount] = {};
    auto *ops = reinterpret_cast<uint8_t *>(shadowOps);
    auto *conf = reinterpret_cast<uint8_t *>(shadowConf);
    if (ok) {
        memcpy(ops, reinterpret_cast<const void *>(amfiOps), arm64::OpsSize);
        auto failure = arm64::buildShadowOps(ops, arm64::OpsSize, names, targets, count, textLow,
                                             textHigh, slots);
        if (failure.error != arm64::Error::None) {
            refuse("%s: %s", names[failure.index] ? names[failure.index] : "(unnamed)",
                   arm64::describe(failure.error));
            ok = false;
        }
    }
    if (ok) {
        memcpy(conf, reinterpret_cast<const void *>(amfiConf), arm64::ConfSize);
        auto error = arm64::buildShadowConf(conf, arm64::ConfSize, reinterpret_cast<uint64_t>(ops));
        if (error != arm64::Error::None) {
            refuse("shadow conf: %s", arm64::describe(error));
            ok = false;
        }
    }
    if (ok && (elementMpc % sizeof(uint64_t) ||
               elementWritability(elementMpc) == Writability::ReadOnly)) {
        // The one live write is the policy-list element's mpc. Require it
        // 8-aligned and in writable memory, since a store to read-only memory
        // would fault rather than refuse.
        refuse("policy-list element is not in writable memory");
        ok = false;
    }
    if (ok) {
        // Publish each callable original before the swap makes the shadow live, so
        // a routed op reached through the shadow always finds its original set.
        for (size_t i = 0; i < count; ++i)
            __atomic_store_n(requests[i].original, reinterpret_cast<void *>(slots[i].original),
                             __ATOMIC_RELEASE);
        __asm__ volatile("dsb ish" ::: "memory");
        // One atomic 8-byte compare-and-exchange repoints the element from AMFI's
        // conf to the shadow conf: a concurrent MAC hook reads one whole pointer
        // or the other, never a torn value. It swaps only when the element still
        // holds AMFI's conf, so a shifted or already-swapped list refuses rather
        // than trampling an unexpected value, and a failed swap has written
        // nothing.
        auto *element = reinterpret_cast<uint64_t *>(elementMpc);
        uint64_t expected = amfiConf;
        uint64_t shadowConfAddress = reinterpret_cast<uint64_t>(conf);
        if (__atomic_compare_exchange_n(element, &expected, shadowConfAddress, false,
                                        __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
            used = true;
            for (size_t i = 0; i < count; ++i)
                note("route: %s installed (AMFI ops +0x%x via shadow)", names[i],
                     static_cast<unsigned>(slots[i].offset));
        } else {
            for (size_t i = 0; i < count; ++i)
                __atomic_store_n(requests[i].original, nullptr, __ATOMIC_RELEASE);
            refuse("policy-list element did not hold AMFI's conf");
            ok = false;
        }
    }

    char reason[sizeof(refusal)];
    if (ok)
        snprintf(reason, sizeof(reason), "%zu AMFI policy op(s) routed through the shadow table",
                 count);
    else
        strlcpy(reason, refusal, sizeof(reason));
    IOLockUnlock(routeLock);
    if (client)
        clientState(client, ok ? "routed" : "refused", reason);
    return ok;
}

} // namespace graft
