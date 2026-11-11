// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The catalogue binding (docs/design.md, "Binding a consumer"). At start it finds
// the bound consumers, each placed kext that carries a manifest. It routes the
// two catalogue entries and, on each send, binds each dormant consumer against
// the live System KC by name: it opens the consumer's manifest, retains and
// verifies its providers, checks each class's live layout, stages and commits
// its import and vtable writes, then opens its activation gate so the deferred
// match starts it. Every stage fails safe and never half-binds, and each consumer
// binds or is refused on its own. The manifest format is docs/manifest.md.
#include "KernelBinding.h"
#include "BindingPlacement.h"
#include "Consumer.h"
#include "KernelWrite.h"
#include "Log.h"
#include "Memory.h"
#include "Resolve.h"
#include "Vtable.h"
#include "arm64e/Synthesize.h"
#include "arm64e/SynthesizeVtable.h"
#include <IOKit/IODeviceTreeSupport.h>
#include <IOKit/IOService.h>
#include <libkern/c++/OSBoolean.h>
#include <libkern/c++/OSData.h>
#include <libkern/c++/OSDictionary.h>
#include <libkern/c++/OSString.h>
#include <libkern/c++/OSSymbol.h>
#include <libkern/c++/OSUnserialize.h>
#include <mach/kmod.h>
#include <mach/vm_prot.h>

namespace graft {
namespace {
const RuntimeImage *kernel;
const mach_header_64 *bootCollection;
uint32_t busy;
uint32_t pending;
struct LibraryState {
    RuntimeImage image;
    OSObject *hold;
};

// One bound consumer: a placed kext that carries a manifest, found at start
// (findConsumers). Each has its own bind, state and activation gate, so one
// consumer's refusal leaves the others as they are.
struct Consumer {
    char identifier[KMOD_MAX_NAME];
    uint64_t base;
    // Bound or refused: no later trigger retries it.
    bool finished;
    // The state last published as GraftClient.<identifier>.
    char lastState[32], lastReason[320];
    // The activation gate (openGate): the key, copied so the published-nub handler
    // can run after the bind pass returns; the provider matching, kept for the
    // life of the kext, since its keys name the personalities the handler
    // reports; and whether a nub has opened the gate yet.
    char gateKey[manifest::MaxGateLength + 1];
    OSDictionary *matching;
    uint32_t gateOpened;
    // The provider holds kept for the rest of the boot once writes may be in
    // place (attemptBind).
    LibraryState *retainedLibraries;
};
Consumer *consumers;
size_t consumerCount;
// Orders report: the bind pass and the matching threads that deliver
// gatePublished both publish a consumer's state, so a record and the state it
// publishes change together, one report at a time. initializeBinding allocates it.
IOLock *reportLock;

// Publish a consumer's state as GraftClient.<identifier>, unless it is the state
// already published. With unlessGateOpen, a report that comes once a nub has
// opened the gate only goes to the log: a later failure to watch for another
// personality's providers must not hide that the consumer is starting.
void report(Consumer &consumer, const char *state, const char *reason,
            bool unlessGateOpen = false) {
    IOLockLock(reportLock);
    if (unlessGateOpen && __atomic_load_n(&consumer.gateOpened, __ATOMIC_ACQUIRE)) {
        note("%s: %s: %s", consumer.identifier, state, reason);
    } else if (strcmp(consumer.lastState, state) || strcmp(consumer.lastReason, reason)) {
        strlcpy(consumer.lastState, state, sizeof(consumer.lastState));
        strlcpy(consumer.lastReason, reason, sizeof(consumer.lastReason));
        clientState(consumer.identifier, state, reason);
    }
    IOLockUnlock(reportLock);
}

const OSMetaClass *metaClass(const char *name) {
    auto symbol = OSSymbol::withCString(name);
    auto meta = symbol ? OSMetaClass::getMetaClassWithName(symbol) : nullptr;
    OSSafeReleaseNULL(symbol);
    return meta;
}

// A consumer manifest offset's runtime address. Without a segment table the
// consumer's segments are contiguous (x86-64) and an offset is base + offset.
// With one (arm64e) the boot firmware coalesces segments by name, so an offset is
// resolved within its original segment and relocated through the placed segment
// of the same name: its placed address plus the in-segment distance. 0 where the
// span lies in no segment or the address is unreadable.
uint64_t mapConsumerOffset(const manifest::View &view, const RuntimeImage &consumer, uint64_t base,
                           uint64_t offset, uint64_t size) {
    if (!size || size - 1 > UINT64_MAX - offset)
        return 0;
    const auto &h = view.header();
    if (!h.segments.count) {
        if (offset > UINT64_MAX - base)
            return 0;
        uint64_t address = base + offset;
        return consumer.contains(address, size, VM_PROT_READ) ? address : 0;
    }
    for (uint32_t i = 0; i < h.segments.count; ++i) {
        auto segment = view.row<manifest::Segment>(h.segments, i);
        if (offset < segment.origin || size > segment.size ||
            offset - segment.origin > segment.size - size)
            continue;
        uint64_t placed = consumer.segmentAddress(view.string(segment.name));
        uint64_t within = offset - segment.origin;
        if (!placed || placed > UINT64_MAX - within)
            return 0;
        uint64_t address = placed + within;
        return consumer.contains(address, size, VM_PROT_READ) ? address : 0;
    }
    return 0;
}

bool openLibraries(const manifest::View &view, LibraryState *states, Provider *providers,
                   bool &waiting, char *reason, size_t reasonSize) {
    const auto &h = view.header();
    for (uint32_t i = 0; i < h.libraries.count; ++i) {
        auto id = view.string(view.row<manifest::Library>(h.libraries, i).identifier);
        // Find the provider's live header, per architecture (BindingPlacement.h):
        // it must be a started, registered kext on x86-64, a placed collection
        // entry on arm64. The hold, where there is one, keeps it resident.
        Placement provider = locate(bootCollection, id, reason, reasonSize);
        states[i].hold = provider.hold;
        uint64_t address = provider.base;
        if (!address) {
            waiting = provider.waiting;
            return false;
        }
        const char *className = manifest::libraryAnchor(view, i);
        auto meta = className ? metaClass(className) : nullptr;
        char anchor[160];
        int count = className ? snprintf(anchor, sizeof(anchor), "__ZN%lu%s10gMetaClassE",
                                         strlen(className), className)
                              : -1;
        if (!meta || count < 0 || static_cast<size_t>(count) >= sizeof(anchor) ||
            !states[i].image.open(reinterpret_cast<const mach_header_64 *>(address), anchor,
                                  reinterpret_cast<uint64_t>(meta))) {
            snprintf(reason, reasonSize, "%s: cannot verify %s's class anchor (%s)", id,
                     className ? className : "no class or anchor",
                     !meta ? "class not registered"
                     : (count < 0 || static_cast<size_t>(count) >= sizeof(anchor))
                             ? "anchor name too long"
                             : states[i].image.error());
            return false;
        }
        providers[i].symbols = states[i].image.symbols;
    }
    return true;
}

bool codeInProviders(const LibraryState *states, size_t count, uint64_t address) {
    if (kernel->contains(address, 1, VM_PROT_EXECUTE))
        return true;
    for (size_t i = 0; i < count; ++i)
        if (states[i].image.contains(address, 1, VM_PROT_EXECUTE))
            return true;
    return false;
}

// The live vtable `symbol` names in `image`: the address of its word 0 and its
// method-slot count, the symbol's extent past the two header words less any
// trailing zero words (the boot firmware's handover counts the same way). False
// when the symbol is missing, its extent is not whole words past the header, or
// no method word is set.
bool liveVtable(const RuntimeImage &image, const char *symbol, uint64_t &table, uint32_t &slots) {
    table = image.resolve(symbol);
    uint64_t extent = table ? image.symbolExtent(table) : 0;
    if (extent < 24 || extent % 8 || (extent - 16) / 8 > UINT32_MAX)
        return false;
    slots = static_cast<uint32_t>((extent - 16) / 8);
    auto entries = reinterpret_cast<const uint8_t *>(table + 16);
    while (slots) {
        uint64_t value;
        memcpy(&value, entries + static_cast<size_t>(slots - 1) * 8, 8);
        if (value)
            break;
        --slots;
    }
    return slots != 0;
}

bool actualClasses(const manifest::View &view, LibraryState *states, ActualClass *classes,
                   char *reason, size_t reasonSize) {
    const auto &h = view.header();
    for (uint32_t i = 0; i < h.classes.count; ++i) {
        auto declared = view.row<manifest::Class>(h.classes, i);
        auto name = view.string(declared.name);
        auto meta = metaClass(name);
        uint64_t table = 0;
        uint32_t slots = 0;
        if (!meta || !liveVtable(states[declared.library].image, view.string(declared.vtable),
                                 table, slots)) {
            snprintf(reason, reasonSize, "%s: missing or unbounded vtable", name);
            return false;
        }
        auto entries = reinterpret_cast<const uint8_t *>(table + 16);
        for (uint32_t j = 0; j < slots; ++j) {
            uint64_t target;
            memcpy(&target, entries + static_cast<size_t>(j) * 8, 8);
            // arm64e provider slots are signed; strip before the execute check
            // (x86-64 strip is identity), so a signed method is not mistaken for
            // a non-executable value and false-refused.
            if (!codeInProviders(states, h.libraries.count, fixup::stripCode(target))) {
                snprintf(reason, reasonSize, "%s: vtable slot %u is not retained executable code",
                         name, j);
                return false;
            }
        }
        classes[i] = {meta->getClassSize(), slots, entries};
    }
    return true;
}

bool validSites(const manifest::View &view, const RuntimeImage &consumer, uint64_t base,
                char *reason, size_t reasonSize) {
    const auto &h = view.header();
    for (uint32_t i = 0; i < h.sites.count; ++i) {
        auto site = view.row<manifest::Site>(h.sites, i);
        uint64_t start = site.offset - (site.width == 4 ? 1u : 0u);
        uint64_t width = site.width + (site.width == 4 ? 1u : 0u);
        if (!mapConsumerOffset(view, consumer, base, start, width) ||
            !mapConsumerOffset(view, consumer, base, site.placeholder, 1)) {
            snprintf(reason, reasonSize, "site %u: address outside consumer segments", i);
            return false;
        }
    }
    for (uint32_t i = 0; i < h.imports.count; ++i) {
        auto import = view.row<manifest::Import>(h.imports, i);
        if (import.fallback != manifest::NoFallback &&
            !mapConsumerOffset(view, consumer, base, import.fallback, 1)) {
            snprintf(reason, reasonSize, "import %u: fallback outside consumer segments", i);
            return false;
        }
    }
    return true;
}

// Map the placed kext at base and find its __DATA,__graft section: its runtime
// address, 0 when the image does not map or has no such section.
uint64_t manifestSection(uint64_t base, RuntimeImage &image, size_t &size) {
    size = 0;
    return image.map(reinterpret_cast<const mach_header_64 *>(base))
                   ? image.section("__DATA", "__graft", size)
                   : 0;
}

// Map the dormant consumer at base and open the manifest it carries in
// __DATA,__graft. A missing section and a malformed manifest both leave the view
// invalid and return open()'s failure; a missing section is a header failure.
// sectionStart is the section's runtime address, 0 without one, so the caller
// can keep writes out of it and read a refused manifest's version.
manifest::Failure openManifest(uint64_t base, RuntimeImage &consumer, manifest::View &view,
                               uint64_t &sectionStart) {
    size_t sectionSize = 0;
    sectionStart = manifestSection(base, consumer, sectionSize);
    return manifest::open(reinterpret_cast<const void *>(sectionStart), sectionSize, view);
}

// Whether the placed kext at base carries a consumer manifest: a __DATA,__graft
// section that starts with the manifest's magic. Graft itself and an owned client
// carry none. The manifest is validated whole when the consumer is bound.
bool carriesManifest(uint64_t base) {
    RuntimeImage image;
    size_t size = 0;
    uint64_t start = manifestSection(base, image, size);
    uint32_t magic = 0;
    if (start && size >= sizeof(magic))
        memcpy(&magic, reinterpret_cast<const void *>(start), sizeof(magic));
    return magic == manifest::Magic;
}

// The bound consumers' discovery: a count, then, with records, each consumer
// recorded until capacity. An identifier longer than a kext's bundle identifier
// can be is no kext's.
struct Discovery {
    Consumer *records;
    size_t capacity, count;
};
void discovered(void *context, const char *identifier, uint64_t base) {
    auto discovery = static_cast<Discovery *>(context);
    if (!identifier || strlen(identifier) >= KMOD_MAX_NAME || !carriesManifest(base))
        return;
    if (discovery->records) {
        if (discovery->count == discovery->capacity)
            return;
        Consumer &consumer = discovery->records[discovery->count];
        strlcpy(consumer.identifier, identifier, sizeof(consumer.identifier));
        consumer.base = base;
        note("binding: found bound consumer %s", identifier);
    }
    ++discovery->count;
}

// Find the bound consumers, once, at start: each placed kext that carries a
// manifest (BindingPlacement.h). Every kext Graft can bind is in place by then:
// on x86-64 the Boot collection is fixed before the kernel starts, and on arm64
// the boot firmware placed each kext and published its node. Graft holds no
// consumer's identity: a consumer declares itself by its manifest.
void findConsumers() {
    Discovery discovery{};
    eachPlacedKext(bootCollection, discovered, &discovery);
    if (!discovery.count) {
        note("binding: no bound consumer is placed");
        return;
    }
    discovery = {allocateArray<Consumer>(discovery.count), discovery.count, 0};
    if (!discovery.records) {
        note("binding: cannot allocate %zu bound consumers' records", discovery.capacity);
        return;
    }
    eachPlacedKext(bootCollection, discovered, &discovery);
    consumers = discovery.records;
    consumerCount = discovery.count;
}

// The write-plan capacity: one write per import site, each declared vtable's
// parent slots (the inherited entries Graft fills), and each synthesized class's
// vtable-slot cap (its whole live-ordered vtable, built word by word; the live
// parent's word count never exceeds the cap). False on overflow.
bool planCapacity(const manifest::View &view, size_t &capacity) {
    const auto &h = view.header();
    capacity = h.sites.count;
    for (uint32_t i = 0; i < h.vtables.count; ++i) {
        auto vtable = view.row<manifest::Vtable>(h.vtables, i);
        auto cls = view.row<manifest::Class>(h.classes, vtable.classIndex);
        if (cls.parentSlots > SIZE_MAX - capacity)
            return false;
        capacity += cls.parentSlots;
    }
    for (uint32_t i = 0; i < h.synthClasses.count; ++i) {
        auto synthClass = view.row<manifest::SynthClass>(h.synthClasses, i);
        if (synthClass.vtableSlotCap > SIZE_MAX - capacity)
            return false;
        capacity += synthClass.vtableSlotCap;
    }
    return true;
}

// The pointer the kernel's gOSKextUnresolved holds: where the kernel linker
// binds a weak import nothing defines (Resolve.h). 0 where the kernel has none.
uint64_t unresolvedSentinel() {
    return binding::unresolvedValue(
            kernel->resolve("_gOSKextUnresolved"), [](uint64_t address, uint64_t &value) {
                if (!kernel->contains(address, sizeof(value), VM_PROT_READ))
                    return false;
                memcpy(&value, reinterpret_cast<const void *>(address), sizeof(value));
                return true;
            });
}

// Resolve every import site against its providers and stage the writes. On
// refusal, name the site's first candidate and the plan and fixup errors.
bool planConsumerImports(const manifest::View &view, const RuntimeImage &consumer, uint64_t base,
                         const Provider *providers, uint64_t unresolved, binding::Site *scratch,
                         binding::Write *writes, size_t capacity, char *reason, size_t reasonSize) {
    const auto &h = view.header();
    auto map = [&](uint64_t offset, uint64_t size) {
        return mapConsumerOffset(view, consumer, base, offset, size);
    };
    // Read a site's old bytes, and a branch's leading opcode byte, from the
    // live mapping; the segment map bounds them (coalesced on arm64e).
    auto read = [&](uint64_t offset, uint8_t width, uint64_t &old, uint8_t &opcode) {
        uint64_t start = offset - (width == 4 ? 1u : 0u);
        uint64_t span = width + (width == 4 ? 1u : 0u);
        uint64_t address = mapConsumerOffset(view, consumer, base, start, span);
        if (!address)
            return false;
        if (width == 4) {
            opcode = *reinterpret_cast<const uint8_t *>(address);
            old = 0;
            memcpy(&old, reinterpret_cast<const void *>(address + 1), 4);
        } else {
            opcode = 0;
            old = 0;
            memcpy(&old, reinterpret_cast<const void *>(address), width);
        }
        return true;
    };
    auto result = planConsumerSites(view, providers, h.libraries.count, unresolved, map, read,
                                    scratch, writes, capacity);
    if (result.error == binding::Error::None)
        return true;
    if (result.site < h.sites.count) {
        auto site = view.row<manifest::Site>(h.sites, static_cast<uint32_t>(result.site));
        auto import = view.row<manifest::Import>(h.imports, site.import);
        snprintf(reason, reasonSize, "%s: site %lu refused (plan %u, %s)",
                 view.string(view.row<uint32_t>(h.candidates, import.firstCandidate)), result.site,
                 static_cast<unsigned>(result.error), fixup::describe(result.detail));
    } else {
        snprintf(reason, reasonSize, "site plan refused (%u)", static_cast<unsigned>(result.error));
    }
    return false;
}

// Check where each staged write now points against the live provider and
// consumer mappings, before anything is written. A refusal names the site.
bool verifyTargets(const manifest::View &view, const RuntimeImage &consumer, uint64_t base,
                   uint64_t unresolved, const LibraryState *states, const binding::Write *writes,
                   char *reason, size_t reasonSize) {
    const auto &h = view.header();
    auto inProvider = [&](uint32_t library, uint64_t address, bool code) {
        return states[library].image.contains(address, 1, code ? VM_PROT_EXECUTE : VM_PROT_READ);
    };
    auto inConsumer = [&](uint64_t address, bool code) {
        return consumer.contains(address, 1, code ? VM_PROT_EXECUTE : VM_PROT_READ);
    };
    auto map = [&](uint64_t offset, uint64_t size) {
        return mapConsumerOffset(view, consumer, base, offset, size);
    };
    auto result = checkTargets(view, map, unresolved, writes, inProvider, inConsumer);
    if (result.error == binding::Error::None)
        return true;
    auto site = view.row<manifest::Site>(h.sites, static_cast<uint32_t>(result.site));
    auto import = view.row<manifest::Import>(h.imports, site.import);
    snprintf(reason, reasonSize,
             result.error == binding::Error::Bounds ? "site %lu: %s: replacement address overflow"
                                                    : "site %lu: %s resolves outside its provider",
             result.site, view.string(view.row<uint32_t>(h.candidates, import.firstCandidate)));
    return false;
}

// Check each class's live layout and stage the inherited vtable writes after
// the import writes. On refusal, name the class or the vtable slot.
bool planInheritedVtables(const manifest::View &view, const RuntimeImage &consumer, uint64_t base,
                          const LibraryState *states, const ActualClass *classes,
                          binding::Write *writes, size_t &vtableWrites, size_t capacity,
                          char *reason, size_t reasonSize) {
    const auto &h = view.header();
    // A slot's runtime address, resolved through the segment map (coalesced on
    // arm64e), used both to read the slot and to re-sign it for that address.
    auto map = [&](uint64_t offset) { return mapConsumerOffset(view, consumer, base, offset, 8); };
    auto read = [&](uint64_t offset, uint64_t &value) {
        uint64_t address = map(offset);
        if (!address)
            return false;
        memcpy(&value, reinterpret_cast<const void *>(address), 8);
        return true;
    };
    auto code = [&](uint64_t address) { return consumer.contains(address, 1, VM_PROT_EXECUTE); };
    auto defines = [&](uint32_t provider, const char *name, uint64_t address) {
        (void)provider;
        if (macho::definesAt(kernel->symbols, name, address))
            return true;
        for (uint32_t i = 0; i < h.libraries.count; ++i)
            if (macho::definesAt(states[i].image.symbols, name, address))
                return true;
        return false;
    };
    auto result =
            planVtables(view, classes, h.classes.count, writes ? writes + h.sites.count : nullptr,
                        capacity - h.sites.count, vtableWrites, read, map, code, defines);
    if (result.error == VtableError::None)
        return true;
    snprintf(reason, reasonSize, "vtable %u slot %u refused (plan %u, layout %u)", result.record,
             result.slot, static_cast<unsigned>(result.error),
             static_cast<unsigned>(result.detail));
    if (result.error == VtableError::Class || result.error == VtableError::Method) {
        auto cls = view.row<manifest::Class>(h.classes, result.record);
        snprintf(reason, reasonSize, "%s slot %u refused (plan %u, layout %u)",
                 view.string(cls.name), result.slot, static_cast<unsigned>(result.error),
                 static_cast<unsigned>(result.detail));
    }
    return false;
}

// Read the donor vtable-discriminator table the boot firmware handed over for
// `symbol` from `/chosen/ambrosia/vtable-disc` (one device-tree child per vtable
// symbol, named and tagged by it). Each child carries the slid vtable-base, the
// first-slot word index and a `slots` blob of 8-byte records. The blob is copied
// into `dest` (so the registry entry can be released at once), bounded by destCap.
// False when the handover, the child or a field is missing or malformed, or the
// blob does not fit; the caller refuses the synthesis by name.
bool donorBlob(const char *symbol, uint64_t &vtableBase, uint32_t &firstSlot, uint8_t *dest,
               size_t destCap, size_t &blobLen) {
    auto node = IORegistryEntry::fromPath("/chosen/ambrosia/vtable-disc", gIODTPlane);
    if (!node)
        return false;
    bool found = false;
    if (auto entries = node->getChildIterator(gIODTPlane)) {
        while (auto entry = OSDynamicCast(IORegistryEntry, entries->getNextObject())) {
            auto tag = OSDynamicCast(OSData, entry->getProperty("symbol"));
            auto text = tag ? static_cast<const char *>(tag->getBytesNoCopy()) : nullptr;
            unsigned length = tag ? tag->getLength() : 0;
            if (!text || !length || text[length - 1] != '\0' || strcmp(text, symbol))
                continue;
            auto baseData = OSDynamicCast(OSData, entry->getProperty("vtable-base"));
            auto slotData = OSDynamicCast(OSData, entry->getProperty("first-slot"));
            auto blobData = OSDynamicCast(OSData, entry->getProperty("slots"));
            if (baseData && baseData->getLength() == sizeof(uint64_t) && slotData &&
                slotData->getLength() == sizeof(uint32_t) && blobData && blobData->getLength() &&
                blobData->getLength() <= destCap) {
                memcpy(&vtableBase, baseData->getBytesNoCopy(), sizeof(uint64_t));
                memcpy(&firstSlot, slotData->getBytesNoCopy(), sizeof(uint32_t));
                blobLen = blobData->getLength();
                memcpy(dest, blobData->getBytesNoCopy(), blobLen);
                found = true;
            }
            break;
        }
        entries->release();
    }
    node->release();
    return found;
}

// Resolve one override to the live parent word slot that holds the same method,
// forward: from each selector's mangled name compose the
// method's symbol under the parent class, resolve it in the parent's provider and
// scan the live parent vtable for the slot; if the parent leaves it pure (the
// parent symbol does not resolve), compose under the sibling, resolve it in the
// entry that holds the sibling, and scan the sibling's live vtable.
// The first selector that resolves wins. NoSlot when none does (an optional
// override the live parent lacks), AmbiguousSlot when a resolved address sits in
// two live slots. The scan copies are PAC-stripped; the resolved address is a
// bare symbol value, which the stripped slot equals.
uint32_t resolveOverride(const manifest::View &view, const manifest::SynthOverride &ov,
                         const char *parentClass, const char *siblingClass,
                         const macho::SymbolTable &provider, const uint64_t *parentStripped,
                         uint32_t parentCount, const macho::SymbolTable &siblingSymbols,
                         const uint64_t *siblingStripped, uint32_t siblingCount) {
    const auto &h = view.header();
    for (uint32_t s = 0; s < ov.selectorCount; ++s) {
        auto id = synth::methodIdentity(
                view.string(view.row<uint32_t>(h.selectors, ov.firstSelector + s)));
        if (!id.valid())
            continue;
        char symbol[256];
        if (synth::composeMemberSymbol(id, parentClass, symbol, sizeof(symbol))) {
            uint64_t address = 0;
            if (binding::unique(provider, symbol, address) == binding::Resolution::Found) {
                auto match = synth::scanVtable(parentStripped, parentCount, address);
                if (match.result == synth::SlotMatch::Found)
                    return synth::DonorFirstSlot + match.slot;
                if (match.result == synth::SlotMatch::Ambiguous)
                    return synth::AmbiguousSlot;
            }
        }
        if (siblingClass && siblingCount &&
            synth::composeMemberSymbol(id, siblingClass, symbol, sizeof(symbol))) {
            uint64_t address = 0;
            if (binding::unique(siblingSymbols, symbol, address) == binding::Resolution::Found) {
                auto match = synth::scanVtable(siblingStripped, siblingCount, address);
                if (match.result == synth::SlotMatch::Found)
                    return synth::DonorFirstSlot + match.slot;
                if (match.result == synth::SlotMatch::Ambiguous)
                    return synth::AmbiguousSlot;
            }
        }
    }
    return synth::NoSlot;
}

// Whether the discriminator fallback may place override `ov` on the live parent
// slot whose stripped target is `target` (synth::fallbackTargetFits). The names
// checked are those Graft's symbol snapshots hold at the target: the kernel's and
// each provider's. A target none of them names, such as a method its image does
// not export, cannot be told from the override's own method, so a discriminator
// that matches such a slot by chance still places the override there.
bool fallbackFits(const manifest::View &view, const manifest::SynthOverride &ov,
                  const LibraryState *states, uint32_t libraryCount, uint64_t target) {
    const auto &h = view.header();
    auto selector = [&](uint32_t s) {
        return view.string(view.row<uint32_t>(h.selectors, ov.firstSelector + s));
    };
    auto eachName = [&](auto visit) {
        auto scan = [&](const macho::SymbolTable &table) {
            for (uint32_t i = 0; i < table.count; ++i)
                if (macho::symbolAddress(table, i) == target)
                    if (auto name = macho::definedName(table, i))
                        visit(name);
        };
        scan(kernel->symbols);
        for (uint32_t i = 0; i < libraryCount; ++i)
            scan(states[i].image.symbols);
    };
    return synth::fallbackTargetFits(selector, ov.selectorCount, eachName);
}

// Scratch for one bind's synthesis, sized to the largest class and reused across
// them. The vtable-word buffers hold vtableSlotCap words (the live parent never
// exceeds it, nor does an ancestor); donorCopy holds a slots blob per chain table
// (one 8-byte record per method word), MaxDonorChain of them.
struct SynthScratch {
    uint64_t *parentStripped;             // live parent vtable, PAC-stripped, for scanning
    uint64_t *siblingStripped;            // live sibling vtable, PAC-stripped, for pure slots
    uint64_t *out;                        // the synthesized vtable, built word by word
    uint8_t *donorCopy;                   // the chain's slots blobs, copied out of the registry
    bool *required;                       // whether each override must land
    uint32_t *resolvedSlots;              // each override's resolved live slot
    uint32_t *overrideSlot;               // planSynthesis's placement result
    synth::OverridePlacement *placements; // the overrides that landed, for the builder
    uint32_t capWords;                    // the word capacity of the vtable buffers
};

// Read one chain table: the handover for `symbol`, copied into `blob` (capacity
// `cap` bytes), parsed and checked against the live vtable at `table` of `slots`
// method words: its address, its length and the discriminator each of its words
// is signed with. Names the table in `reason` when it is missing or diverges.
bool chainTable(const char *name, const char *symbol, uint64_t table, uint32_t slots, uint8_t *blob,
                size_t cap, synth::DonorTable &out, char *reason, size_t reasonSize) {
    uint64_t vtableBase = 0;
    uint32_t firstSlot = 0;
    size_t blobLen = 0;
    if (!donorBlob(symbol, vtableBase, firstSlot, blob, cap, blobLen)) {
        snprintf(reason, reasonSize, "%s: no vtable-disc handover for %s", name, symbol);
        return false;
    }
    auto parsed = synth::parseDonorTable(vtableBase, firstSlot, blob, blobLen, out);
    if (parsed.error != synth::DonorError::None) {
        snprintf(reason, reasonSize, "%s: donor table for %s invalid (error %u, slot %u)", name,
                 symbol, static_cast<unsigned>(parsed.error), parsed.slot);
        return false;
    }
    auto checked = synth::checkDonorLive(
            out, table, slots + synth::DonorFirstSlot,
            reinterpret_cast<const uint64_t *>(table + synth::DonorFirstSlot * 8));
    if (checked.error == synth::DonorError::Signature) {
        snprintf(reason, reasonSize,
                 "%s: donor table for %s does not sign the live vtable's slot %u", name, symbol,
                 checked.slot);
        return false;
    }
    if (checked.error != synth::DonorError::None) {
        snprintf(reason, reasonSize,
                 "%s: donor table for %s diverges from the live vtable (error %u)", name, symbol,
                 static_cast<unsigned>(checked.error));
        return false;
    }
    return true;
}

// Synthesize one consumer class's vtable against the live parent and the donor
// chain, and stage its words. Measures the live parent (the declared class of the
// same name, already measured) and the sibling, reads and cross-checks the chain
// (the parent's table and each ancestor's within its provider), resolves each
// override forward, builds the live-ordered vtable (each slot signed with the
// chain's discriminator: the live method where inherited, the consumer's
// implementation where overridden), and stages one write per word. Refuses by
// name on every divergence; `count` advances by the words it staged. (arm64e in
// practice; the neutral path never enters it on x86-64, where no manifest carries
// a synthesized class.)
bool synthesizeClass(const manifest::View &view, const RuntimeImage &consumer, uint64_t base,
                     const LibraryState *states, const ActualClass *classes,
                     const manifest::SynthClass &synthClass, const SynthScratch &scratch,
                     binding::Write *writes, size_t &count, size_t capacity, char *reason,
                     size_t reasonSize) {
    const auto &h = view.header();
    const char *name = view.string(synthClass.name);
    const char *parentName = view.string(synthClass.parentClass);
    const char *siblingName =
            synthClass.siblingClass ? view.string(synthClass.siblingClass) : nullptr;
    const auto &provider = states[synthClass.library].image;

    // The live parent is the declared class of the same name actualClasses measured.
    uint32_t parentIndex = UINT32_MAX;
    for (uint32_t i = 0; i < h.classes.count; ++i)
        if (!strcmp(view.string(view.row<manifest::Class>(h.classes, i).name), parentName)) {
            parentIndex = i;
            break;
        }
    if (parentIndex == UINT32_MAX) {
        snprintf(reason, reasonSize, "%s: synthesis parent %s is not a declared class", name,
                 parentName);
        return false;
    }
    auto parentDeclared = view.row<manifest::Class>(h.classes, parentIndex);
    const ActualClass &parent = classes[parentIndex];
    uint32_t parentCount = parent.slotCount;
    // Bound synthesis by the stand-in's own reserved slot count (the parent
    // class's parentSlots), not by the whole compiled vtable: the words past the
    // stand-in's reserve are the consumer's own appended virtuals, which a longer
    // live parent must not overwrite. Those words and the two header words must
    // also fit the compiled vtable's cap, which sizes the scratch buffers.
    if (!synth::slotsFit(parentCount, parentDeclared.parentSlots, synthClass.vtableSlotCap)) {
        snprintf(reason, reasonSize,
                 "%s: live parent %s has %u vtable words, over the %u the stand-in reserves or "
                 "the %u-word vtable",
                 name, parentName, parentCount, parentDeclared.parentSlots,
                 synthClass.vtableSlotCap);
        return false;
    }
    if (!synth::instanceFits(parent.instanceSize, synthClass.parentRegion)) {
        snprintf(reason, reasonSize, "%s: live parent %s instance size %u over the %u reserved",
                 name, parentName, parent.instanceSize, synthClass.parentRegion);
        return false;
    }
    uint64_t parentTable = reinterpret_cast<uint64_t>(parent.slots) - synth::DonorFirstSlot * 8;

    // The donor chain: the live parent's table, then the table of each ancestor
    // whose vtable the parent's provider defines, toward the root. A caller
    // authenticates a slot with its first declaration's discriminator, which the
    // most-base table covering the slot carries (VtableDisc.h). Ancestors outside
    // the provider (IOService and its bases, in the kernel) need none: the
    // provider's own most-base table covers their slots with their values.
    size_t blobCap = static_cast<size_t>(scratch.capWords) * 8;
    synth::DonorChain chain;
    if (!chainTable(name, view.string(parentDeclared.vtable), parentTable, parentCount,
                    scratch.donorCopy, blobCap, chain.tables[0], reason, reasonSize))
        return false;
    chain.count = 1;
    auto parentMeta = metaClass(parentName);
    for (auto ancestor = parentMeta ? parentMeta->getSuperClass() : nullptr; ancestor;
         ancestor = ancestor->getSuperClass()) {
        const char *ancestorName = ancestor->getClassName();
        char symbol[160];
        int n = snprintf(symbol, sizeof(symbol), "__ZTV%zu%s", strlen(ancestorName), ancestorName);
        if (n <= 0 || static_cast<size_t>(n) >= sizeof(symbol)) {
            snprintf(reason, reasonSize, "%s: ancestor %s's vtable name is too long", name,
                     ancestorName);
            return false;
        }
        uint64_t table = 0;
        uint32_t slots = 0;
        if (!liveVtable(provider, symbol, table, slots))
            break;
        if (chain.count == synth::MaxDonorChain) {
            snprintf(reason, reasonSize, "%s: more than %u classes of %s's chain in its provider",
                     name, synth::MaxDonorChain, parentName);
            return false;
        }
        auto &tableOut = chain.tables[chain.count];
        if (!chainTable(name, symbol, table, slots, scratch.donorCopy + chain.count * blobCap,
                        blobCap, tableOut, reason, reasonSize))
            return false;
        auto prefix = synth::checkDonorAncestor(tableOut, chain.tables[chain.count - 1]);
        if (prefix.error != synth::DonorError::None) {
            snprintf(reason, reasonSize, "%s: donor table for %s is longer than its subclass's",
                     name, symbol);
            return false;
        }
        ++chain.count;
    }

    // The scan copies: the live parent, and the sibling up to the parent's range.
    for (uint32_t i = 0; i < parentCount; ++i) {
        uint64_t value;
        memcpy(&value, parent.slots + static_cast<size_t>(i) * 8, 8);
        scratch.parentStripped[i] = fixup::stripCode(value);
    }
    // The sibling need not be in the parent's provider: on vmapple it is another
    // kext of the boot collection, placed but never started. It is read only for
    // method identity, never called, so Graft opens whichever entry defines its
    // vtable, with the kernel's own symbol bias. Where none does, the pure
    // overrides fall back to the discriminator match below.
    uint32_t siblingCount = 0;
    RuntimeImage sibling;
    if (siblingName) {
        char siblingSymbol[160];
        int n = snprintf(siblingSymbol, sizeof(siblingSymbol), "__ZTV%zu%s", strlen(siblingName),
                         siblingName);
        uint64_t table = 0;
        uint32_t slots = 0;
        if (n > 0 && static_cast<size_t>(n) < sizeof(siblingSymbol) &&
            sibling.openEntryDefining(bootCollection, siblingSymbol, kernel->symbols.bias) &&
            liveVtable(sibling, siblingSymbol, table, slots)) {
            auto entries = reinterpret_cast<const uint8_t *>(table + 16);
            uint32_t scan = slots < parentCount ? slots : parentCount;
            for (uint32_t i = 0; i < scan; ++i) {
                uint64_t value;
                memcpy(&value, entries + static_cast<size_t>(i) * 8, 8);
                scratch.siblingStripped[i] = fixup::stripCode(value);
            }
            siblingCount = scan;
        }
    }

    // Resolve each override forward, then plan the placements.
    for (uint32_t k = 0; k < synthClass.overrideCount; ++k) {
        auto ov = view.row<manifest::SynthOverride>(h.synthOverrides, synthClass.firstOverride + k);
        scratch.required[k] = ov.flags & manifest::SynthRequired;
        scratch.resolvedSlots[k] = resolveOverride(
                view, ov, parentName, siblingName, provider.symbols, scratch.parentStripped,
                parentCount, sibling.symbols, scratch.siblingStripped, siblingCount);
        // A pure override the live parent leaves as ___cxa_pure_virtual resolves by
        // the sibling's symbol -- but some releases export no symbol for that slot's
        // method, or hold a parameter form the recorded name does not match. When no
        // selector resolved (NoSlot, not the hard AmbiguousSlot), place it on the
        // unique live parent slot whose discriminator in the parent's own table
        // equals the override's recorded one. The table is cross-checked against the
        // live vtable above and the match must be unique, but a 16-bit discriminator
        // can equal an unrelated slot's when the parent lacks the method, so the
        // slot is taken only when no symbol at its method names another method.
        if (scratch.resolvedSlots[k] == synth::NoSlot) {
            uint32_t slot;
            if (synth::uniqueDiscriminatorSlot(chain.tables[0], ov.diversity, slot) &&
                fallbackFits(view, ov, states, h.libraries.count,
                             scratch.parentStripped[slot - synth::DonorFirstSlot]))
                scratch.resolvedSlots[k] = slot;
        }
    }
    auto plan = synth::planSynthesis(scratch.resolvedSlots, scratch.required,
                                     synthClass.overrideCount, scratch.overrideSlot);
    if (plan.error != synth::PlanError::None) {
        snprintf(reason, reasonSize, "%s: override %u unresolved (plan %u, slot %u)", name,
                 plan.index, static_cast<unsigned>(plan.error), plan.slot);
        return false;
    }

    // Each landed override's implementation address. Its slot signs with the
    // chain's discriminator; the consumer's build must have signed the slot under
    // the same scheme, or it is not a kernel vtable override.
    uint32_t placementCount = 0;
    for (uint32_t k = 0; k < synthClass.overrideCount; ++k) {
        if (scratch.overrideSlot[k] == synth::NoSlot)
            continue;
        auto ov = view.row<manifest::SynthOverride>(h.synthOverrides, synthClass.firstOverride + k);
        uint64_t impl = mapConsumerOffset(view, consumer, base, ov.impl, 1);
        if (!impl || !consumer.contains(impl, 1, VM_PROT_EXECUTE)) {
            snprintf(reason, reasonSize, "%s: override %u implementation is not consumer code",
                     name, k);
            return false;
        }
        if (ov.opcode != synth::DonorOpcode) {
            snprintf(reason, reasonSize, "%s: override %u is signed as %#x, not as a vtable slot",
                     name, k, ov.opcode);
            return false;
        }
        scratch.placements[placementCount++] = {scratch.overrideSlot[k], impl};
    }

    // Build the live-ordered vtable and stage each of its words.
    uint64_t destBase =
            mapConsumerOffset(view, consumer, base, synthClass.vtableOffset,
                              (static_cast<uint64_t>(parentCount) + synth::DonorFirstSlot) * 8);
    if (!destBase) {
        snprintf(reason, reasonSize, "%s: vtable offset outside consumer segments", name);
        return false;
    }
    // The offset names the consumer's __ZTV word 0: its offset-to-top and typeinfo
    // words, both zero in a kext built without RTTI. Anything else means the offset
    // is not the vtable's start, and filling from it would shift every word.
    uint64_t header[synth::DonorFirstSlot];
    memcpy(header, reinterpret_cast<const void *>(destBase), sizeof(header));
    if (header[0] || header[1]) {
        snprintf(reason, reasonSize, "%s: vtable offset is not the start of its vtable", name);
        return false;
    }
    auto built = synth::buildSynthesizedVtable(scratch.out, parentCount, destBase,
                                               reinterpret_cast<const uint64_t *>(parent.slots),
                                               chain, scratch.placements, placementCount);
    if (built.error != synth::BuildError::None) {
        snprintf(reason, reasonSize, "%s: vtable build failed (error %u, slot %u)", name,
                 static_cast<unsigned>(built.error), built.slot);
        return false;
    }
    for (uint32_t w = 0; w < parentCount + synth::DonorFirstSlot; ++w) {
        if (count >= capacity) {
            snprintf(reason, reasonSize, "%s: synthesis exceeds write capacity", name);
            return false;
        }
        uint64_t offset = synthClass.vtableOffset + static_cast<uint64_t>(w) * 8;
        uint64_t address = mapConsumerOffset(view, consumer, base, offset, 8);
        if (!address) {
            snprintf(reason, reasonSize, "%s: vtable word %u outside consumer segments", name, w);
            return false;
        }
        uint64_t before;
        memcpy(&before, reinterpret_cast<const void *>(address), 8);
        writes[count++] = {offset, before, scratch.out[w], 8};
    }
    return true;
}

// Build and stage every synthesized class's vtable after the import and inherited
// vtable writes. Allocates the per-bind synthesis scratch once, sized to the
// largest class, and refuses the whole bind by name on any class's divergence. A
// failed allocation is transient, so it sets waiting instead, as attemptBind's own
// allocations do. A manifest with no synthesized class (every x86-64 manifest, and
// an arm64e consumer that declares none) returns at once.
bool planSynthesizedVtables(const manifest::View &view, const RuntimeImage &consumer, uint64_t base,
                            const LibraryState *states, const ActualClass *classes,
                            binding::Write *writes, size_t &synthWrites, size_t capacity,
                            bool &waiting, char *reason, size_t reasonSize) {
    synthWrites = 0;
    const auto &h = view.header();
    if (!h.synthClasses.count)
        return true;
    uint32_t capWords = 0, maxOverrides = 0;
    for (uint32_t i = 0; i < h.synthClasses.count; ++i) {
        auto synthClass = view.row<manifest::SynthClass>(h.synthClasses, i);
        if (synthClass.vtableSlotCap > capWords)
            capWords = synthClass.vtableSlotCap;
        if (synthClass.overrideCount > maxOverrides)
            maxOverrides = synthClass.overrideCount;
    }
    size_t donorBytes = static_cast<size_t>(capWords) * 8 * synth::MaxDonorChain;
    SynthScratch scratch{allocateArray<uint64_t>(capWords),
                         allocateArray<uint64_t>(capWords),
                         allocateArray<uint64_t>(capWords),
                         allocateArray<uint8_t>(donorBytes),
                         allocateArray<bool>(maxOverrides),
                         allocateArray<uint32_t>(maxOverrides),
                         allocateArray<uint32_t>(maxOverrides),
                         allocateArray<synth::OverridePlacement>(maxOverrides),
                         capWords};
    bool ready = scratch.parentStripped && scratch.siblingStripped && scratch.out &&
                 scratch.donorCopy && scratch.required && scratch.resolvedSlots &&
                 scratch.overrideSlot && scratch.placements;
    bool ok = ready;
    size_t count = 0;
    if (!ready) {
        waiting = true;
        snprintf(reason, reasonSize, "cannot allocate synthesis scratch");
    }
    for (uint32_t i = 0; ok && i < h.synthClasses.count; ++i)
        ok = synthesizeClass(view, consumer, base, states, classes,
                             view.row<manifest::SynthClass>(h.synthClasses, i), scratch, writes,
                             count, capacity, reason, reasonSize);
    if (ok)
        synthWrites = count;
    freeArray(scratch.parentStripped, capWords);
    freeArray(scratch.siblingStripped, capWords);
    freeArray(scratch.out, capWords);
    freeArray(scratch.donorCopy, donorBytes);
    freeArray(scratch.required, maxOverrides);
    freeArray(scratch.resolvedSlots, maxOverrides);
    freeArray(scratch.overrideSlot, maxOverrides);
    freeArray(scratch.placements, maxOverrides);
    return ok;
}

// Turn the staged writes into patches and apply them in one transaction.
// preserveBindings stays set when the writes committed or an unverified
// rollback may have left some in place, so the caller keeps provider holds.
// manifestStart/manifestSize are the consumer's own __graft section, which Graft
// reads back after the commit (the gate key), so no write may land in it.
bool commitWrites(const manifest::View &view, const RuntimeImage &consumer, uint64_t base,
                  const binding::Write *writes, Patch *patches, size_t count,
                  uint64_t manifestStart, uint64_t manifestSize, bool &preserveBindings,
                  char *reason, size_t reasonSize) {
    for (size_t i = 0; i < count; ++i) {
        uint64_t address =
                mapConsumerOffset(view, consumer, base, writes[i].offset, writes[i].width);
        if (!address) {
            preserveBindings = false;
            snprintf(reason, reasonSize, "write %zu: offset outside consumer segments", i);
            return false;
        }
        // The target segment must accept the store on this architecture (x86-64
        // allows __TEXT; arm64e refuses a read-only or non-writable segment),
        // checked before the first write so nothing is committed on a refusal.
        if (!writableConsumerTarget(consumer, address, writes[i].width)) {
            preserveBindings = false;
            snprintf(reason, reasonSize, "write %zu: target segment is not writable", i);
            return false;
        }
        // A write into the manifest section would corrupt the data Graft reads
        // back from it after the commit; refuse it rather than write there.
        if (binding::overlaps(address, writes[i].width, manifestStart, manifestSize)) {
            preserveBindings = false;
            snprintf(reason, reasonSize, "write %zu: target overlaps the manifest section", i);
            return false;
        }
        patches[i] = {reinterpret_cast<uint8_t *>(address),
                      reinterpret_cast<const uint8_t *>(&writes[i].before),
                      reinterpret_cast<const uint8_t *>(&writes[i].after), writes[i].width};
    }
    auto result = applyWrites(patches, count);
    preserveBindings = result == WriteError::None || result == WriteError::Rollback;
    if (result == WriteError::None)
        return true;
    snprintf(reason, reasonSize, "write transaction refused (%u)", static_cast<unsigned>(result));
    return false;
}

// Log each weak import that nothing defines: the consumer runs without it,
// as it chose to (manifest::ImportWeak).
void noteAbsentWeakImports(const manifest::View &view, const RuntimeImage &consumer, uint64_t base,
                           const Provider *providers, uint64_t unresolved) {
    const auto &h = view.header();
    auto find = [&](uint32_t library, const char *name, uint64_t &address) {
        return library < h.libraries.count
                       ? binding::unique(providers[library].symbols, name, address)
                       : binding::Resolution::Invalid;
    };
    auto map = [&](uint64_t offset, uint64_t size) {
        return mapConsumerOffset(view, consumer, base, offset, size);
    };
    for (uint32_t i = 0; i < h.imports.count; ++i) {
        auto import = view.row<manifest::Import>(h.imports, i);
        uint64_t address = 0;
        if ((import.flags & manifest::ImportWeak) &&
            binding::resolveImport(view, i, find, unresolved, map, address) ==
                    binding::Resolution::Found &&
            address == unresolved)
            note("binding: %s: weak import absent, bound to gOSKextUnresolved",
                 view.string(view.row<uint32_t>(h.candidates, import.firstCandidate)));
    }
}

// The consumer's provider matching (docs/manifest.md): the manifest's XML
// dictionary from each personality's name to the matching dictionary of its
// provider. Parsed before anything is written, so a consumer whose gate could
// never open is refused while it is still untouched. Null, with reason set, when
// it does not parse, names no personality, or holds a personality that is not a
// dictionary naming an IOProviderClass.
OSDictionary *providerMatching(const manifest::View &view, char *reason, size_t reasonSize) {
    OSString *error = nullptr;
    OSObject *parsed = OSUnserializeXML(view.string(view.header().matching), &error);
    auto matching = OSDynamicCast(OSDictionary, parsed);
    if (!matching || !matching->getCount()) {
        if (!parsed)
            snprintf(reason, reasonSize, "provider matching does not parse%s%s", error ? ": " : "",
                     error ? error->getCStringNoCopy() : "");
        else
            snprintf(reason, reasonSize, "provider matching %s",
                     matching ? "names no personality" : "is not a dictionary of personalities");
        OSSafeReleaseNULL(parsed);
        OSSafeReleaseNULL(error);
        return nullptr;
    }
    OSSafeReleaseNULL(error);
    struct Check {
        char *reason;
        size_t reasonSize;
        bool invalid;
    } check{reason, reasonSize, false};
    matching->iterateObjects(&check, [](void *context, const OSSymbol *name, OSObject *object) {
        auto personality = OSDynamicCast(OSDictionary, object);
        if (personality && OSDynamicCast(OSString, personality->getObject("IOProviderClass")))
            return false;
        auto check = static_cast<Check *>(context);
        snprintf(check->reason, check->reasonSize,
                 "personality %s: provider matching names no IOProviderClass",
                 name->getCStringNoCopy());
        return check->invalid = true;
    });
    if (check.invalid)
        OSSafeReleaseNULL(matching);
    return matching;
}

// The activation gate: the consumer's gate property is a passive match term on
// its personalities, evaluated before the consumer class is ever constructed, so
// unlike an IOResourceMatch (which gates only start(), after the base
// constructors have already run through the bound slots) it holds the consumer
// dormant until Graft sets the property on its provider nub and re-registers it.
// The consumer's record holds what the published-nub handler needs after the
// bind pass returns.
//
// A nub that a personality's provider matching matches: set the gate property on
// it and re-register it, so its deferred match now starts the consumer with its
// slots and vtable bound. addMatchingNotification delivers every current match
// this way and every future one, so a nub published after the bind opens the gate
// too. registerService does asynchronous matching (it posts a job rather than
// matching inline), so calling it from within notification delivery re-enters no
// matching lock; setProperty takes only the nub's own property lock.
// Re-registering an already-published nub does not re-fire the first-publish
// notification, so each personality's handler sees each nub once. A nub that
// another of the consumer's personalities already opened is left alone: its
// re-registration matches every personality. The first nub reports the gate
// open, naming the personality (refCon); later nubs only set the property. It
// marks the gate open before it reports, so a watch failure that report orders
// after it cannot replace the open state. target is the consumer's record.
bool gatePublished(void *target, void *personality, IOService *nub, IONotifier *) {
    auto &consumer = *static_cast<Consumer *>(target);
    if (!nub || !consumer.gateKey[0] || nub->getProperty(consumer.gateKey) == kOSBooleanTrue)
        return true;
    nub->setProperty(consumer.gateKey, kOSBooleanTrue);
    nub->registerService();
    if (!__atomic_exchange_n(&consumer.gateOpened, 1u, __ATOMIC_ACQ_REL)) {
        char reason[320];
        snprintf(reason, sizeof(reason),
                 "imports and vtables committed; activation gate open on a provider of "
                 "personality %s",
                 static_cast<const OSSymbol *>(personality)->getCStringNoCopy());
        report(consumer, "bound", reason);
    }
    return true;
}

// Open the activation gate: watch, for each personality, for the nubs its
// provider matching matches, and open the gate on every one, now or later. It
// reports the bound state: gate open if a provider is already present, else
// waiting for one, so the state always begins with "bound". addMatchingNotification,
// setProperty and registerService are exported on every release, macOS 27
// included, so one binary activates across the range.
void openGate(Consumer &consumer, const char *key, OSDictionary *matching) {
    strlcpy(consumer.gateKey, key, sizeof(consumer.gateKey));
    consumer.matching = matching;
    // Report waiting first: the handler reports the gate open, so whichever nub
    // opens it, now or on a later matching thread, writes the last state.
    struct Names {
        char text[160];
    } names{};
    matching->iterateObjects(&names, [](void *context, const OSSymbol *name, OSObject *) {
        auto names = static_cast<Names *>(context);
        if (names->text[0])
            strlcat(names->text, ", ", sizeof(names->text));
        strlcat(names->text, name->getCStringNoCopy(), sizeof(names->text));
        return false;
    });
    char reason[320];
    snprintf(reason, sizeof(reason),
             "imports and vtables committed; waiting for a provider of personality %s to open "
             "the gate",
             names.text);
    report(consumer, "bound", reason);
    // addMatchingNotification retains each matching dictionary and delivers its
    // existing matches synchronously before it returns; no notifier is ever
    // removed, so later-published nubs stay covered for the life of the kext. A
    // personality it cannot watch is the consumer's state only while no nub has
    // opened the gate; after that it goes to the log.
    matching->iterateObjects(&consumer, [](void *target, const OSSymbol *name, OSObject *object) {
        if (!IOService::addMatchingNotification(gIOFirstPublishNotification,
                                                OSDynamicCast(OSDictionary, object), &gatePublished,
                                                target, const_cast<OSSymbol *>(name))) {
            char reason[320];
            snprintf(reason, sizeof(reason),
                     "imports and vtables committed; cannot watch for a provider of personality %s",
                     name->getCStringNoCopy());
            report(*static_cast<Consumer *>(target), "bound", reason, true);
        }
        return false;
    });
}

// Bind one dormant consumer: open its manifest, verify its providers and class
// layouts, stage and commit its import and vtable writes, then open its
// activation gate so the deferred match starts it. Each stage fails safe,
// setting the consumer's registry state to waiting (retry on a later send) or
// refused, and never half-binds. Nothing here touches another consumer's record.
void attemptBind(Consumer &record) {
    uint64_t base = record.base;
    RuntimeImage consumer;
    manifest::View view;
    uint64_t manifestStart = 0;
    auto failure = openManifest(base, consumer, view, manifestStart);
    if (failure.error != manifest::Error::None) {
        // A consumer built with another release's SDK is discovered by its magic
        // but carries another version; name it, since rebuilding is the fix.
        char reason[160];
        uint32_t version = manifest::unsupportedVersion(
                reinterpret_cast<const void *>(manifestStart), failure);
        if (version)
            snprintf(reason, sizeof(reason),
                     "consumer manifest version %u is not one this Graft reads; rebuild the "
                     "consumer with this release's SDK",
                     version);
        else
            strlcpy(reason, "invalid consumer manifest", sizeof(reason));
        report(record, "refused", reason);
        record.finished = true;
        return;
    }
    const auto &h = view.header();
    if (kextStarted(record.identifier)) {
        report(record, "refused", "consumer started before binding");
        record.finished = true;
        return;
    }
    size_t capacity = 0;
    if (!planCapacity(view, capacity)) {
        report(record, "refused", "write-plan size overflow");
        record.finished = true;
        return;
    }
    char reason[320];
    OSDictionary *matching = providerMatching(view, reason, sizeof(reason));
    if (!matching) {
        report(record, "refused", reason);
        record.finished = true;
        return;
    }
    auto states = allocateArray<LibraryState>(h.libraries.count);
    auto providers = allocateArray<Provider>(h.libraries.count);
    auto classes = allocateArray<ActualClass>(h.classes.count);
    auto scratch = allocateArray<binding::Site>(h.sites.count);
    auto writes = allocateArray<binding::Write>(capacity);
    auto patches = allocateArray<Patch>(capacity);
    bool complete = (!h.libraries.count || (states && providers)) &&
                    (!h.classes.count || classes) && (!h.sites.count || scratch) &&
                    (!capacity || (writes && patches));
    // A failed allocation is transient: wait and retry on a later trigger rather
    // than refusing the consumer for the whole boot. The later pipeline steps set
    // waiting themselves where a miss may still resolve.
    bool waiting = !complete;
    strlcpy(reason, "cannot allocate binding resources", sizeof(reason));
    size_t vtableWrites = 0;
    size_t synthWrites = 0;
    bool preserveBindings = false;
    // Where a weak import binds when nothing defines it (manifest::ImportWeak).
    uint64_t unresolved = unresolvedSentinel();
    if (complete)
        complete = openLibraries(view, states, providers, waiting, reason, sizeof(reason));
    if (complete)
        complete = actualClasses(view, states, classes, reason, sizeof(reason));
    if (complete)
        complete = validSites(view, consumer, base, reason, sizeof(reason));
    if (complete)
        complete = planConsumerImports(view, consumer, base, providers, unresolved, scratch, writes,
                                       capacity, reason, sizeof(reason));
    if (complete)
        complete = verifyTargets(view, consumer, base, unresolved, states, writes, reason,
                                 sizeof(reason));
    if (complete)
        complete = planInheritedVtables(view, consumer, base, states, classes, writes, vtableWrites,
                                        capacity, reason, sizeof(reason));
    if (complete)
        // Build each synthesized class's vtable (arm64e): the consumer subclasses a
        // header-less, per-release parent, so Graft fills its own oversized __ZTV in
        // live order at bind, after the inherited writes.
        complete = planSynthesizedVtables(view, consumer, base, states, classes,
                                          writes ? writes + h.sites.count + vtableWrites : nullptr,
                                          synthWrites, capacity - h.sites.count - vtableWrites,
                                          waiting, reason, sizeof(reason));
    size_t count = h.sites.count + vtableWrites + synthWrites;
    if (complete)
        complete = commitWrites(view, consumer, base, writes, patches, count, manifestStart, h.size,
                                preserveBindings, reason, sizeof(reason));
    if (complete) {
        noteAbsentWeakImports(view, consumer, base, providers, unresolved);
        // Open the activation gate: watch for the consumer's provider nubs and set
        // its gate property so the personalities' passive IOPropertyMatch, which
        // deferred the consumer's very construction, now matches and the kernel
        // starts it with its slots and vtable bound. openGate keeps the matching
        // and reports the bound state (gate open, or waiting for a provider).
        openGate(record, view.string(h.gate), matching);
        matching = nullptr;
        record.finished = true;
    }
    OSSafeReleaseNULL(matching);
    // Opening the gate can expose the consumer to a concurrent match. Never undo
    // valid addresses after that boundary or release providers used by a
    // possibly running client. An unverified write rollback likewise requires
    // permanent provider holds.
    if (preserveBindings) {
        record.retainedLibraries = states;
        states = nullptr;
    }
    if (!complete) {
        report(record, waiting ? "waiting" : "refused", reason);
        record.finished = !waiting;
    }
    if (states)
        for (uint32_t i = 0; i < h.libraries.count; ++i)
            OSSafeReleaseNULL(states[i].hold);
    freeArray(states, h.libraries.count);
    freeArray(providers, h.libraries.count);
    freeArray(classes, h.classes.count);
    freeArray(scratch, h.sites.count);
    freeArray(writes, capacity);
    freeArray(patches, capacity);
}

} // namespace

void bindConsumers() {
    // Triggers can arrive concurrently (each catalogue send on x86-64). Only one
    // thread runs the pass, which retries every consumer not yet bound or refused,
    // at a time, but a trigger that arrives while another thread holds the pass
    // must not be dropped: it may be the send whose state lets a bind complete. Record it in
    // pending, which the holder drains, and re-check pending after releasing busy so a trigger in
    // that window is not lost.
    __atomic_store_n(&pending, 1, __ATOMIC_RELEASE);
    for (;;) {
        uint32_t expected = 0;
        if (!__atomic_compare_exchange_n(&busy, &expected, 1, false, __ATOMIC_ACQ_REL,
                                         __ATOMIC_RELAXED))
            return;
        while (__atomic_exchange_n(&pending, 0, __ATOMIC_ACQ_REL)) {
            for (size_t i = 0; i < consumerCount; ++i)
                if (!consumers[i].finished)
                    attemptBind(consumers[i]);
            publishLog();
        }
        __atomic_store_n(&busy, 0, __ATOMIC_RELEASE);
        if (!__atomic_load_n(&pending, __ATOMIC_ACQUIRE))
            return;
        // A trigger landed between the last drain and releasing busy; loop to
        // re-acquire and handle it. If another thread acquires first, it will.
    }
}

bool initializeBinding(const RuntimeImage &image, const mach_header_64 *collection) {
    reportLock = IOLockAlloc();
    if (!reportLock) {
        note("binding: cannot allocate the consumer state lock");
        return false;
    }
    kernel = &image;
    bootCollection = collection;
    findConsumers();
    // The trigger is per-architecture (BindingPlacement.h): x86-64 routes the
    // catalogue entry points so each send retries the binds; arm64 needs none, as
    // every image is placed before Graft starts and the kmod-start bind is the
    // one chance. graft_start calls bindConsumers once either way.
    return installBindingTrigger(image, bindConsumers);
}

} // namespace graft
