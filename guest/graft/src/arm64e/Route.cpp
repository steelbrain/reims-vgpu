// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The arm64e route installer (src/Route.h's interface). On x86-64 Graft routes a
// client's kernel entries through trampolines it writes into kernel text. On
// arm64 the collection's text and a MAC policy's ops table are read-only when
// Graft runs (its kmod start is after machine lockdown), so a client's admission
// hooks are routed by shadowing that policy: Graft copies the current
// mac_policy_ops and mac_policy_conf into fresh writable memory, re-signs the
// routed op slots in the copy for the client's replacements, points the copy's
// mpc_ops at the copy's ops, and repoints the one writable mac_policy_list
// element from the conf it currently holds to the copy. That single element
// store is the only live kernel write; both pointers it touches are stored raw,
// so only the op function pointers are re-signed. A route targets one policy: the
// client names it, AMFI by default or another such as Sandbox, and Graft keeps a
// separate shadow per policy. A policy's "current" conf and ops are its own before
// any client routes it and the newest shadow afterwards, so each request builds on
// the last and several clients route one after another; an operation on a policy
// that one client holds is refused to the next. The plan arithmetic and the ABI
// constants are the pure, host-tested Arm64Route.h; this file owns resolution, the
// live reads and the fail-safe transaction. The x86-64 installer is
// src/x86_64/Route.cpp.
#include "Route.h"
#include "Arm64Route.h"
#include "BindingPlacement.h"
#include "Fixup.h"
#include "Log.h"
#include "MachO.h"
#include "Policy.h"
#include <IOKit/IOLib.h>
#include <mach/kmod.h>
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

// A MAC policy the arm64e mechanism can shadow: the registered name a client
// routes against (and the mpc_name the by-name match confirms), and the kext
// whose image carries the policy's conf and ops. AMFI is the default, found by
// its admission ops rather than by name (findPolicy); the rest are matched by
// mpc_name within their image. Apple's own kexts, named as Apple's identifiers
// are (Policy.h); a client that names none of these is refused.
struct PolicyTarget {
    const char *name;
    const char *identifier;
};
enum { AmfiPolicy, SandboxPolicy, KnownPolicyCount };
const PolicyTarget policyTargets[KnownPolicyCount] = {
        {"AMFI", policy::AmfiIdentifier},
        {"Sandbox", policy::SandboxIdentifier},
};

// Split a route symbol into its policy and operation. On arm64e a symbol names an
// operation alone -- AMFI's, the default -- or "<policy>:<operation>", such as
// "Sandbox:file_check_mmap". Sets `index` to the policy target and `operation` to
// the operation name, a pointer into `symbol` past any separator. A separator
// never appears in an operation name, so the split is unambiguous. Returns false,
// with `operation` still set, when a named policy is not one Graft shadows.
bool parseSymbol(const char *symbol, size_t &index, const char *&operation) {
    const char *colon = __builtin_strchr(symbol, ':');
    if (!colon) {
        index = AmfiPolicy;
        operation = symbol;
        return true;
    }
    operation = colon + 1;
    size_t length = static_cast<size_t>(colon - symbol);
    for (size_t i = 0; i < KnownPolicyCount; ++i)
        if (__builtin_strlen(policyTargets[i].name) == length &&
            !__builtin_memcmp(symbol, policyTargets[i].name, length)) {
            index = i;
            return true;
        }
    return false;
}

// The per-policy shadow state, guarded by routeLock, one entry per PolicyTarget.
// A policy's list element is located once (established), and the conf and ops it
// currently points at are kept: the policy's own before any client routes it, the
// newest shadow afterwards. Each request copies the current ops and conf of its
// policy, routes its operations into the copy, and swaps that policy's element
// from the current conf to the copy, so several clients route one after another.
// routedBy names the client that holds each operation of the policy, empty while
// none does, so the next client is refused it, by that name, rather than handed a
// prior client's replacement as the policy's original. A copy, since a client's
// identifier string need not outlive its call.
struct Shadow {
    bool established;
    uint64_t policyElement;
    uint64_t currentConf;
    uint64_t currentOps;
    char routedBy[arm64::OperationCount][KMOD_MAX_NAME];
};
Shadow shadows[KnownPolicyCount];

static_assert(arm64::OpsSize % sizeof(uint64_t) == 0, "shadow ops alignment");
static_assert(arm64::ConfSize % sizeof(uint64_t) == 0, "shadow conf alignment");

// Allocate 8-aligned, zeroed, boot-lifetime backing for a shadow ops table or
// conf. Because every ops slot is signed with no address diversity, the
// wholesale copy keeps the untouched slots' signatures valid at this new
// address, so any 8-aligned block serves. A superseded shadow is never freed:
// a concurrent MAC check may still read it for the rest of the boot.
uint64_t *shadowAlloc(size_t size) {
    auto block = static_cast<uint64_t *>(IOMalloc(size));
    if (block)
        memset(block, 0, size);
    return block;
}

char refusal[192];

void refuse(const char *format, ...) __attribute__((format(printf, 1, 2)));
void refuse(const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(refusal, sizeof(refusal), format, arguments);
    va_end(arguments);
    note("route: refuse %s", refusal);
}

// A kernel address: canonical form, with the high 16 bits set. The name pointer
// in a policy conf is a string and need not be 8-aligned, so this is its gate,
// paired with a span check before the string itself is read.
bool kernelAddress(uint64_t pointer) {
    return (pointer & 0xFFFF000000000000ULL) == 0xFFFF000000000000ULL;
}

// A canonical, 8-aligned kernel pointer: the gate before dereferencing a policy
// structure resolved from live memory, so a mis-resolution refuses rather than
// faults on a wild address.
bool canonicalKernel(uint64_t pointer) {
    return kernelAddress(pointer) && pointer % sizeof(uint64_t) == 0;
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
// against the policy's image before it is dereferenced, so a mis-resolution
// refuses rather than faults on a wild address.
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

// Whether the policy conf's mpc_name equals `want`, read safely. mpc_name is a
// string pointer in the conf; it must be a kernel address lying within the
// policy's image -- where the kext's string literals live -- for the length
// compared, so a mis-resolved conf refuses rather than faults. The by-name match
// confirms a policy a client names, other than AMFI (findPolicy).
bool policyNamed(const ImageSpan &image, uint64_t conf, const char *want) {
    uint64_t name = readPointer(conf + arm64::ConfNameOffset);
    size_t length = __builtin_strlen(want) + 1; // compare the NUL terminator too
    if (!kernelAddress(name) || !spanContains(image, name, length))
        return false;
    return __builtin_memcmp(reinterpret_cast<const void *>(name), want, length) == 0;
}

// A policy image's executable address range in the running collection. The op
// pointers the policy signs point here; Graft uses the range both to confirm the
// policy's list element and as the shadow-ops guard. False when the image is not
// in the collection or has no executable span.
bool execRange(const mach_header_64 *collection, const char *identifier, uint64_t &low,
               uint64_t &high) {
    ImageSpan image = imageSpan(collection, identifier);
    if (!image.ok)
        return false;
    bool any = false;
    low = high = 0;
    macho::eachCommand(image.header, [&](const load_command *lc) {
        if (lc->cmd == LC_SEGMENT_64 && lc->cmdsize >= sizeof(segment_command_64)) {
            auto segment = reinterpret_cast<const segment_command_64 *>(lc);
            if (segment->initprot & VM_PROT_EXECUTE) {
                uint64_t start = segment->vmaddr + image.slide;
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

// Resolve the symbol-less mac_policy_list from mac_policy_register and return its
// entries pointer and the number of elements to scan. Shared by every policy
// finder. The list global itself must lie in one of the kernel's writable data
// segments before Graft reads it; the scan is bounded by the list's own max. The
// entries array is heap memory the kernel allocated, outside any image, so Graft
// has no bounds to check it against: it requires only a canonical, aligned kernel
// pointer before reading the elements, and a wrong pointer there faults rather
// than refuses. Refuses, setting the reason, on any resolution failure.
bool resolvePolicyList(uint64_t &entries, size_t &scan) {
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
    uint64_t base = readPointer(listBase + arm64::PolicyListEntriesOffset);
    if (!canonicalKernel(base)) {
        refuse("mac_policy_list entries pointer is invalid");
        return false;
    }
    entries = base;
    scan = arm64::policyScanCount(maxindex, max);
    return true;
}

// The policy-list element for target `index`, its conf and its ops, found by
// scanning the list for the one element whose conf and ops both lie in the
// policy's image and that the target confirms. AMFI is confirmed by behaviour:
// its ops implement both library validation and code-signature checking into its
// own text -- AMFI alone does on a stock guest, so no name read is needed. Every
// other target is confirmed by mpc_name, matched inside the image where the string
// lives. [textLow, textHigh) is the policy image's own executable range, which
// execRange returned. Refuses on zero or more than one match rather than guess.
// Returns the element's mpc-field address (the repoint target) and the policy's
// conf and ops addresses.
bool findPolicy(const mach_header_64 *collection, size_t index, uint64_t textLow, uint64_t textHigh,
                uint64_t &elementMpc, uint64_t &conf, uint64_t &ops) {
    uint64_t entries = 0;
    size_t scan = 0;
    if (!resolvePolicyList(entries, scan))
        return false;
    const ImageSpan image = imageSpan(collection, policyTargets[index].identifier);
    const uint32_t lvOffset = arm64::operationFor("file_check_library_validation")->offset;
    const uint32_t svOffset = arm64::operationFor("vnode_check_signature")->offset;
    size_t matches = 0;
    for (size_t i = 0; i < scan; ++i) {
        uint64_t element = entries + i * arm64::PolicyElementStride;
        uint64_t candidateConf = readPointer(element + arm64::PolicyElementMpcOffset);
        if (!canonicalKernel(candidateConf) || !spanContains(image, candidateConf, arm64::ConfSize))
            continue;
        uint64_t candidateOps = readPointer(candidateConf + arm64::ConfOpsOffset);
        if (!canonicalKernel(candidateOps) || !spanContains(image, candidateOps, arm64::OpsSize))
            continue;
        bool confirmed;
        if (index == AmfiPolicy) {
            uint64_t validation = fixup::stripCode(readPointer(candidateOps + lvOffset));
            uint64_t signature = fixup::stripCode(readPointer(candidateOps + svOffset));
            confirmed = validation >= textLow && validation < textHigh && signature >= textLow &&
                        signature < textHigh;
        } else {
            confirmed = policyNamed(image, candidateConf, policyTargets[index].name);
        }
        if (!confirmed)
            continue;
        ++matches;
        elementMpc = element + arm64::PolicyElementMpcOffset;
        conf = candidateConf;
        ops = candidateOps;
    }
    if (matches != 1) {
        refuse(matches ? "%s policy ambiguous in the list (%zu matches)"
                       : "%s policy not found in the list",
               policyTargets[index].name, matches);
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
    if (count > arm64::OperationCount) {
        refuse("transaction: %zu requests exceed the %zu routable operations", count,
               arm64::OperationCount);
        ok = false;
    }
    // No BSD-publication gate here, unlike x86-64's kernel-text trampolines
    // (src/x86_64/Route.cpp). The arm64 route goes live by one atomic
    // compare-and-exchange of a single policy-list element, from the policy's conf
    // to the fully built shadow conf; a concurrent MAC hook reads one valid conf or
    // the other, never a torn value, so the install is safe after BSD starts. An
    // owned client that matches as an IOKit driver necessarily starts post-BSD,
    // so gating on it would refuse every arm64 client.
    //
    // Every route in the call targets the same MAC policy, parsed from each
    // symbol's optional "<policy>:" prefix: the install swaps that one policy's
    // element, so a call cannot straddle two. A client routing a second policy
    // makes a second call; a bare operation name is AMFI's, the default.
    size_t policyIndex = AmfiPolicy;
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
        if (!ok)
            break;
        size_t idx;
        const char *operation;
        if (!parseSymbol(request.symbol, idx, operation)) {
            refuse("policy not shadowable: %s", request.symbol);
            ok = false;
            break;
        }
        if (i == 0)
            policyIndex = idx;
        else if (idx != policyIndex) {
            refuse("one route call targets one policy, not both %s and %s",
                   policyTargets[policyIndex].name, policyTargets[idx].name);
            ok = false;
            break;
        }
        // Refuse an operation a prior client already routed on this policy, naming
        // that client. Its slot in the current ops points at that client's
        // replacement, not the policy's own function, so routing it again would
        // hand this client a replacement in place of the policy's original. One
        // operation of a policy belongs to one client for the boot.
        const arm64::Operation *op = arm64::operationFor(operation);
        const char *heldBy =
                op ? shadows[idx].routedBy[static_cast<size_t>(op - &arm64::Operations[0])] : "";
        if (heldBy[0]) {
            refuse("%s: already routed by %s", request.symbol, heldBy);
            ok = false;
            break;
        }
        names[i] = operation;
        targets[i] = reinterpret_cast<uint64_t>(request.replacement);
    }
    Shadow &shadow = shadows[policyIndex];

    const mach_header_64 *collection = ok ? bootCollectionHeader() : nullptr;
    uint64_t textLow = 0, textHigh = 0;
    if (ok && (!collection ||
               !execRange(collection, policyTargets[policyIndex].identifier, textLow, textHigh))) {
        refuse("%s image or its executable range not found", policyTargets[policyIndex].name);
        ok = false;
    }
    // Require each replacement to lie in one of the client's own executable
    // segments, so a client that passes a data pointer refuses here rather than
    // panicking when the policy's dispatch first branches to the shadow slot. The
    // firmware places the client in the Auxiliary collection and names its image
    // by bundle identifier, as it names each kext it places (placedBase); a
    // client it names no image for is refused, not skipped, so a route's
    // replacement is always tied to the client's real placed code. A
    // replacement is a signed function pointer, so its stripped address is
    // compared. Graft's own routing never reaches here with a null client.
    uint64_t clientBase = ok && client ? placedBase(collection, client) : 0;
    if (ok && client && !clientBase) {
        refuse("the client %s names no placed image", client);
        ok = false;
    }
    RuntimeImage clientImage;
    if (ok && clientBase &&
        !clientImage.map(reinterpret_cast<const mach_header_64 *>(clientBase))) {
        refuse("the client's placed image cannot be read (%s)", clientImage.error());
        ok = false;
    }
    for (size_t i = 0; ok && clientBase && i < count; ++i)
        if (!clientImage.contains(fixup::stripCode(targets[i]), 1, VM_PROT_EXECUTE)) {
            refuse("request %zu: replacement is not in the client's executable text", i);
            ok = false;
        }

    // Locate the policy's list element the first time any client routes it. After
    // that the element holds a shadow conf, so the policy's own signature no longer
    // finds it: the remembered element and the current conf/ops are reused. The
    // element's alignment and writability are checked once, when it is adopted.
    if (ok && !shadow.established) {
        uint64_t mpc = 0, foundConf = 0, foundOps = 0;
        if (!findPolicy(collection, policyIndex, textLow, textHigh, mpc, foundConf, foundOps)) {
            ok = false;
        } else if (mpc % sizeof(uint64_t) || elementWritability(mpc) == Writability::ReadOnly) {
            refuse("policy-list element is not in writable memory");
            ok = false;
        } else {
            shadow.policyElement = mpc;
            shadow.currentConf = foundConf;
            shadow.currentOps = foundOps;
        }
    }

    // Build a fresh shadow from the current ops and conf, route into the copy,
    // and keep it for the boot. The copies are freed only on a failure that never
    // published them (below); a superseded live shadow is never freed.
    arm64::Slot slots[arm64::OperationCount] = {};
    uint64_t *ops = ok ? shadowAlloc(arm64::OpsSize) : nullptr;
    uint64_t *conf = ok ? shadowAlloc(arm64::ConfSize) : nullptr;
    if (ok && (!ops || !conf || reinterpret_cast<uintptr_t>(ops) % sizeof(uint64_t) ||
               reinterpret_cast<uintptr_t>(conf) % sizeof(uint64_t))) {
        refuse("shadow allocation unavailable or misaligned");
        ok = false;
    }
    if (ok) {
        memcpy(ops, reinterpret_cast<const void *>(shadow.currentOps), arm64::OpsSize);
        auto failure = arm64::buildShadowOps(reinterpret_cast<uint8_t *>(ops), arm64::OpsSize,
                                             names, targets, count, textLow, textHigh, slots);
        if (failure.error != arm64::Error::None) {
            refuse("%s: %s", names[failure.index] ? names[failure.index] : "(unnamed)",
                   arm64::describe(failure.error));
            ok = false;
        }
    }
    if (ok) {
        memcpy(conf, reinterpret_cast<const void *>(shadow.currentConf), arm64::ConfSize);
        auto error = arm64::buildShadowConf(reinterpret_cast<uint8_t *>(conf), arm64::ConfSize,
                                            reinterpret_cast<uint64_t>(ops));
        if (error != arm64::Error::None) {
            refuse("shadow conf: %s", arm64::describe(error));
            ok = false;
        }
    }
    if (ok) {
        // Publish each callable original before the swap makes the shadow live, so
        // a routed op reached through the shadow always finds its original set.
        for (size_t i = 0; i < count; ++i)
            __atomic_store_n(requests[i].original, reinterpret_cast<void *>(slots[i].original),
                             __ATOMIC_RELEASE);
        __asm__ volatile("dsb ish" ::: "memory");
        // One atomic 8-byte compare-and-exchange repoints the element from the
        // conf it currently holds to the new shadow conf: a concurrent MAC hook
        // reads one whole pointer or the other, never a torn value. It swaps only
        // when the element still holds the current conf, so a shifted or
        // externally changed list refuses rather than trampling an unexpected
        // value, and a failed swap has written nothing.
        auto *element = reinterpret_cast<uint64_t *>(shadow.policyElement);
        uint64_t expected = shadow.currentConf;
        uint64_t newConf = reinterpret_cast<uint64_t>(conf);
        if (__atomic_compare_exchange_n(element, &expected, newConf, false, __ATOMIC_RELEASE,
                                        __ATOMIC_RELAXED)) {
            shadow.established = true;
            shadow.currentConf = newConf;
            shadow.currentOps = reinterpret_cast<uint64_t>(ops);
            for (size_t i = 0; i < count; ++i) {
                const arm64::Operation *op = arm64::operationFor(names[i]);
                if (op)
                    strlcpy(shadow.routedBy[static_cast<size_t>(op - &arm64::Operations[0])],
                            client ? client : "Graft", KMOD_MAX_NAME);
                note("route: %s installed (%s ops +0x%x via shadow)", names[i],
                     policyTargets[policyIndex].name, static_cast<unsigned>(slots[i].offset));
            }
        } else {
            for (size_t i = 0; i < count; ++i)
                __atomic_store_n(requests[i].original, nullptr, __ATOMIC_RELEASE);
            refuse("policy-list element did not hold the current conf");
            ok = false;
        }
    }
    if (!ok) {
        // The shadow was never installed, so no MAC check can reach it; free it.
        if (conf)
            IOFree(conf, arm64::ConfSize);
        if (ops)
            IOFree(ops, arm64::OpsSize);
    }

    char reason[sizeof(refusal)];
    if (ok)
        snprintf(reason, sizeof(reason), "%zu %s policy op(s) routed through the shadow table",
                 count, policyTargets[policyIndex].name);
    else
        strlcpy(reason, refusal, sizeof(reason));
    IOLockUnlock(routeLock);
    if (client)
        clientState(client, ok ? "routed" : "refused", reason);
    return ok;
}

} // namespace graft
