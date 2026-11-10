// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The consumer manifest's format, which docs/manifest.md specifies: the records
// a bound consumer's build writes into its __DATA,__graft section and Graft
// reads when it binds the consumer. Every field is little-endian and records
// are packed with no padding. A table's offset counts bytes from the start of
// the section. The image offsets of sites, placeholders, fallbacks, vtables,
// segments and synthesized classes and overrides count from the consumer's
// load address. The records hold no pointers, so they need no relocations.
//
// open() validates a manifest as Graft does before it uses any of it. The kext
// and the SDK's build tool share this header, so it needs only <stddef.h>,
// <stdint.h>, and memcpy, strcmp and strlen from <string.h>.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <string.h>

namespace graft {
namespace manifest {

constexpr uint32_t Magic = 0x54465247; // "GRFT"
// Version 2 is an x86-64 consumer, version 3 an arm64e one. They share the
// header and every table, but a site's two trailing bytes differ (Site), and
// only version 3 carries segments, fills and synthesized classes.
constexpr uint32_t Version = 2;
constexpr uint32_t VersionArm64e = 3;
// An import's fallback when the consumer gives none.
constexpr uint64_t NoFallback = UINT64_MAX;

struct Table {
    uint32_t offset, count;
};

// gate is the string offset of the consumer's activation property key, the
// IOPropertyMatch its personalities carry. device is the string offset of their
// IOPCIPrimaryMatch, which names the PCI nub Graft sets the gate property on
// once it has bound the consumer.
//
// The longest gate key, in bytes without its NUL. Graft keeps a copy of the key
// to open the gate on device nubs published after the bind, so open() refuses a
// longer one rather than let the copy cut it short.
constexpr size_t MaxGateLength = 159;
struct Header {
    uint32_t magic, version, size, headerSize;
    Table strings, libraries, imports, candidates, sites, classes, slots, vtables, overrides,
            segments, fills, selectors, synthOverrides, synthClasses;
    uint32_t gate, device;
};

// anchor names a class of the library whose registered metaclass confirms the
// library's image, for a library the consumer declares no class from; 0 where
// the first class it declares from the library serves.
struct Library {
    uint32_t identifier, anchor;
};

// The consumer references the import weakly. Where no candidate or fallback
// resolves it, it binds to the pointer the kernel's gOSKextUnresolved holds,
// as the kernel's linker binds an unresolved weak reference.
constexpr uint32_t ImportWeak = 1;
struct Import {
    uint32_t library, firstCandidate, candidateCount, flags;
    uint64_t fallback;
};

// A reference to an import at an image offset. On a version-2 manifest width is
// 8 for a pointer, with opcode OpcodePointer, or 4 for a branch displacement,
// with the branch's opcode; reserved is 0. On a version-3 manifest every site is
// an 8-byte pointer, opcode is its auth byte (SiteSigned, SiteKeyMask,
// SiteAddrDiv) and reserved its 16-bit diversity; a plain pointer has both 0.
struct Site {
    uint64_t offset, placeholder;
    int64_t addend;
    uint32_t import;
    uint8_t width, opcode;
    uint16_t reserved;
};
// A version-2 site's opcode: a pointer, or the instruction before a branch's
// 32-bit displacement.
constexpr uint8_t OpcodePointer = 0;
constexpr uint8_t OpcodeCall = 0xe8;
constexpr uint8_t OpcodeJump = 0xe9;
// A version-3 site's auth byte.
constexpr uint8_t SiteSigned = 0x1;  // bit 0: the pointer is signed
constexpr uint8_t SiteKeyMask = 0x6; // bits 1-2: key, 0=IA 1=IB 2=DA 3=DB
constexpr uint8_t SiteAddrDiv = 0x8; // bit 3: blend the slot address into the diversity
constexpr uint8_t SiteAuthBits = SiteSigned | SiteKeyMask | SiteAddrDiv;
constexpr uint8_t siteAuthKey(uint8_t opcode) {
    return (opcode & SiteKeyMask) >> 1;
}

struct Class {
    uint32_t name, vtable, library, instanceSize, parentSlots, firstSlot, slotCount;
};
struct Slot {
    uint32_t index, method;
};
// offset is the image offset of the vtable's address point, 16 bytes in.
struct Vtable {
    uint64_t offset;
    uint32_t classIndex, slotCount, firstOverride, overrideCount, firstFill, fillCount;
};

// One of the consumer's own segments (version 3 only): its first byte as an
// image offset and its size. The arm64e loader places segments by name, so
// Graft maps an image offset through the placed segment that holds it.
struct Segment {
    uint64_t origin, size;
    uint32_t name, reserved;
};
// An inherited vtable slot Graft fills from the live parent (version 3 only),
// with the signing context the consumer's build gave the slot: opcode is an auth
// byte, as a site's, and diversity its constant discriminator.
struct Fill {
    uint32_t slot;
    uint16_t diversity;
    uint8_t opcode, reserved;
};

// A synthesized class (version 3 only) derives from a parent whose vtable
// changes between releases, so Graft lays out its vtable when it binds, placing
// each override on the live slot that holds the same method.
constexpr uint8_t SynthRequired = 1; // the bind must place the override
// One override: its run of selectors (the method's mangled names), the image
// offset of the consumer's implementation, and the signing context the
// consumer's build gave its slot. flags holds SynthRequired.
struct SynthOverride {
    uint64_t impl;
    uint32_t firstSelector, selectorCount;
    uint16_t diversity;
    uint8_t opcode, flags;
    uint32_t reserved;
};
struct SynthClass {
    uint64_t vtableOffset;                 // image offset of the vtable's first word
    uint32_t name;                         // the consumer class
    uint32_t parentClass;                  // the live parent, also named in classes
    uint32_t siblingClass;                 // a concrete subclass of the parent, or 0
    uint32_t library;                      // where the parent is
    uint32_t vtableSlotCap;                // the compiled vtable's words, from its first
    uint32_t parentRegion;                 // the largest live parent instance it has room for
    uint32_t firstOverride, overrideCount; // its run of synthOverrides
};

static_assert(sizeof(Header) == 136 && sizeof(Library) == 8 && sizeof(Import) == 24 &&
                      sizeof(Site) == 32 && sizeof(Class) == 28 && sizeof(Slot) == 8 &&
                      sizeof(Vtable) == 32 && sizeof(Segment) == 24 && sizeof(Fill) == 8 &&
                      sizeof(SynthOverride) == 24 && sizeof(SynthClass) == 40,
              "manifest record layout");

enum class Error {
    None,
    Header,
    Version,
    Range,
    Overlap,
    String,
    Library,
    Import,
    Site,
    Class,
    Slot,
    Vtable,
    Segment,
    Fill,
    Selector,
    SynthOverride,
    SynthClass
};
struct Failure {
    Error error = Error::None;
    uint32_t record = 0;
};

class View {
    const uint8_t *bytes_ = nullptr;
    Header header_{};

public:
    const Header &header() const { return header_; }
    bool valid() const { return bytes_ != nullptr; }
    // Only use accessors after open() succeeds; tables and indices must
    // then be bounded by the corresponding count in header().
    template <typename T>
    T row(Table table, uint32_t index) const {
        T out{};
        memcpy(&out, bytes_ + table.offset + static_cast<size_t>(index) * sizeof(T), sizeof(T));
        return out;
    }
    // The non-empty string at offset in the string table, NUL-terminated inside it.
    const char *string(uint32_t offset) const {
        if (offset >= header_.strings.count)
            return nullptr;
        auto at = reinterpret_cast<const char *>(bytes_ + header_.strings.offset + offset);
        uint32_t room = header_.strings.count - offset;
        for (uint32_t i = 1; *at && i < room; ++i) {
            if (!at[i])
                return at;
        }
        return nullptr;
    }
    friend Failure open(const void *, size_t, View &);
};

inline bool contains(uint32_t first, uint32_t count, uint32_t total) {
    return first <= total && count <= total - first;
}

inline Failure open(const void *bytes, size_t size, View &out) {
    out = View{};
    if (!bytes || size < sizeof(Header))
        return {Error::Header};
    View v;
    v.bytes_ = static_cast<const uint8_t *>(bytes);
    memcpy(&v.header_, bytes, sizeof(Header));
    auto &h = v.header_;
    if (h.magic != Magic || h.headerSize != sizeof(Header) || h.size < sizeof(Header) ||
        h.size > size)
        return {Error::Header};
    if (h.version != Version && h.version != VersionArm64e)
        return {Error::Version};
    bool authenticated = h.version == VersionArm64e;
    // The segment, fill and synthesis tables are an arm64e form: x86-64 places
    // segments contiguously, copies vtable slots verbatim and synthesizes no
    // class, so it carries none of them.
    if (!authenticated && (h.segments.count || h.fills.count || h.selectors.count ||
                           h.synthOverrides.count || h.synthClasses.count))
        return {Error::Version};
    Table tables[] = {h.strings, h.libraries, h.imports,        h.candidates,  h.sites,
                      h.classes, h.slots,     h.vtables,        h.overrides,   h.segments,
                      h.fills,   h.selectors, h.synthOverrides, h.synthClasses};
    const size_t widths[] = {1,
                             sizeof(Library),
                             sizeof(Import),
                             sizeof(uint32_t),
                             sizeof(Site),
                             sizeof(Class),
                             sizeof(Slot),
                             sizeof(Vtable),
                             sizeof(uint32_t),
                             sizeof(Segment),
                             sizeof(Fill),
                             sizeof(uint32_t),
                             sizeof(SynthOverride),
                             sizeof(SynthClass)};
    uint64_t ends[14]{};
    for (size_t i = 0; i < 14; ++i) {
        auto t = tables[i];
        uint64_t length = static_cast<uint64_t>(t.count) * widths[i];
        if (t.offset < sizeof(Header) || t.offset > h.size || length > h.size - t.offset)
            return {Error::Range, static_cast<uint32_t>(i)};
        ends[i] = t.offset + length;
        for (size_t j = 0; j < i; ++j) {
            if (t.count && tables[j].count && t.offset < ends[j] && tables[j].offset < ends[i])
                return {Error::Overlap, static_cast<uint32_t>(i)};
        }
    }
    if (!h.strings.count || v.bytes_[h.strings.offset] != 0 || !v.string(h.gate) ||
        strlen(v.string(h.gate)) > MaxGateLength || !v.string(h.device))
        return {Error::String};
    for (uint32_t i = 0; i < h.libraries.count; ++i) {
        auto r = v.row<Library>(h.libraries, i);
        if (!v.string(r.identifier) || (r.anchor && !v.string(r.anchor)))
            return {Error::Library, i};
    }
    for (uint32_t i = 0; i < h.candidates.count; ++i) {
        if (!v.string(v.row<uint32_t>(h.candidates, i)))
            return {Error::String, i};
    }
    for (uint32_t i = 0; i < h.imports.count; ++i) {
        auto r = v.row<Import>(h.imports, i);
        if (r.library >= h.libraries.count || (r.flags & ~ImportWeak) || !r.candidateCount ||
            !contains(r.firstCandidate, r.candidateCount, h.candidates.count))
            return {Error::Import, i};
    }
    uint64_t end = 0;
    for (uint32_t i = 0; i < h.sites.count; ++i) {
        auto r = v.row<Site>(h.sites, i);
        if (r.import >= h.imports.count || r.offset > UINT64_MAX - r.width)
            return {Error::Site, i};
        if (authenticated) {
            // An 8-byte pointer; opcode is its auth byte, reserved its
            // diversity. A plain (unsigned) pointer sets neither.
            if (r.width != 8 || (r.opcode & ~SiteAuthBits) ||
                (!(r.opcode & SiteSigned) && (r.opcode || r.reserved)))
                return {Error::Site, i};
        } else if (r.reserved || (r.width != 4 && r.width != 8) ||
                   (r.width == 8 ? r.opcode != 0
                                 : (r.offset == 0 || (r.opcode != 0xe8 && r.opcode != 0xe9)))) {
            return {Error::Site, i};
        }
        uint64_t start = r.offset - (r.width == 4 ? 1u : 0u);
        if (i && start < end)
            return {Error::Site, i};
        end = r.offset + r.width;
    }
    // The consumer's own segment table: sorted, bounded, disjoint original ranges.
    uint64_t segmentEnd = 0;
    for (uint32_t i = 0; i < h.segments.count; ++i) {
        auto seg = v.row<Segment>(h.segments, i);
        if (seg.reserved || !v.string(seg.name) || !seg.size ||
            seg.origin > UINT64_MAX - seg.size || (i && seg.origin < segmentEnd))
            return {Error::Segment, i};
        segmentEnd = seg.origin + seg.size;
    }
    // Each class owns a contiguous, non-overlapping range of requirements;
    // requiring an exact partition also rejects unused method declarations.
    uint32_t first = 0;
    for (uint32_t i = 0; i < h.classes.count; ++i) {
        auto r = v.row<Class>(h.classes, i);
        if (!v.string(r.name) || !v.string(r.vtable) || r.library >= h.libraries.count ||
            !r.instanceSize || !r.parentSlots || r.firstSlot != first ||
            !contains(r.firstSlot, r.slotCount, h.slots.count))
            return {Error::Class, i};
        uint32_t previous = 0;
        for (uint32_t j = 0; j < r.slotCount; ++j) {
            auto slot = v.row<Slot>(h.slots, r.firstSlot + j);
            if (!v.string(slot.method) || slot.index >= r.parentSlots ||
                (j && slot.index <= previous))
                return {Error::Slot, r.firstSlot + j};
            previous = slot.index;
        }
        first += r.slotCount;
    }
    if (first != h.slots.count)
        return {Error::Class, h.classes.count};
    first = 0;
    end = 0;
    uint32_t firstFill = 0;
    for (uint32_t i = 0; i < h.vtables.count; ++i) {
        auto r = v.row<Vtable>(h.vtables, i);
        if (r.classIndex >= h.classes.count || r.offset < 16 || r.offset % 8 || !r.slotCount ||
            r.offset > UINT64_MAX - static_cast<uint64_t>(r.slotCount) * 8 ||
            r.firstOverride != first || !contains(first, r.overrideCount, h.overrides.count) ||
            (i && r.offset - 16 < end))
            return {Error::Vtable, i};
        auto cls = v.row<Class>(h.classes, r.classIndex);
        if (r.slotCount < cls.parentSlots)
            return {Error::Vtable, i};
        // A synthesized class's parent is measured at bind, not layout-checked, so
        // its declared parent-slot count is a stand-in placeholder, not the live
        // vtable length. A Vtable record of that same class would make vtable
        // planning copy parentSlots words from a live vtable that may be shorter.
        // Refuse it here, where the parent is named by a synthesized class.
        for (uint32_t s = 0; s < h.synthClasses.count; ++s) {
            auto parent = v.string(v.row<SynthClass>(h.synthClasses, s).parentClass);
            if (parent && !strcmp(parent, v.string(cls.name)))
                return {Error::Vtable, i};
        }
        uint32_t previous = 0;
        for (uint32_t j = 0; j < r.overrideCount; ++j) {
            auto slot = v.row<uint32_t>(h.overrides, first + j);
            if (slot >= cls.parentSlots || (j && slot <= previous))
                return {Error::Vtable, i};
            previous = slot;
        }
        // The inherited slots Graft fills: none on x86-64; on arm64e exactly the
        // parent slots the consumer does not override, each with its signing
        // context, so fills and overrides partition [0, parentSlots) in order.
        if (!authenticated) {
            if (r.fillCount || r.firstFill != firstFill)
                return {Error::Fill, i};
        } else {
            if (r.firstFill != firstFill || !contains(firstFill, r.fillCount, h.fills.count) ||
                r.overrideCount > cls.parentSlots ||
                r.fillCount != cls.parentSlots - r.overrideCount)
                return {Error::Fill, i};
            uint32_t nextOverride = 0, nextFill = 0;
            for (uint32_t slot = 0; slot < cls.parentSlots; ++slot) {
                if (nextOverride < r.overrideCount &&
                    v.row<uint32_t>(h.overrides, first + nextOverride) == slot) {
                    ++nextOverride;
                    continue;
                }
                auto fill = v.row<Fill>(h.fills, firstFill + nextFill);
                if (fill.slot != slot || fill.reserved || (fill.opcode & ~SiteAuthBits) ||
                    (!(fill.opcode & SiteSigned) && (fill.opcode || fill.diversity)))
                    return {Error::Fill, i};
                ++nextFill;
            }
        }
        first += r.overrideCount;
        firstFill += r.fillCount;
        end = r.offset + static_cast<uint64_t>(r.slotCount) * 8;
        for (uint32_t j = 0; j < h.sites.count; ++j) {
            auto site = v.row<Site>(h.sites, j);
            uint64_t start = site.offset - (site.width == 4 ? 1u : 0u);
            if (start < end && r.offset - 16 < site.offset + site.width)
                return {Error::Vtable, i};
        }
    }
    if (first != h.overrides.count || firstFill != h.fills.count)
        return {Error::Vtable, h.vtables.count};
    // Synthesis (arm64e): selector strings, then overrides partitioning the selector
    // table in order with a valid signing context, then classes partitioning the
    // override table in order. The version gate above rejects any of these on x86-64.
    for (uint32_t i = 0; i < h.selectors.count; ++i)
        if (!v.string(v.row<uint32_t>(h.selectors, i)))
            return {Error::Selector, i};
    uint32_t firstSelector = 0;
    for (uint32_t i = 0; i < h.synthOverrides.count; ++i) {
        auto r = v.row<SynthOverride>(h.synthOverrides, i);
        if (!r.selectorCount || r.firstSelector != firstSelector ||
            !contains(r.firstSelector, r.selectorCount, h.selectors.count) || !r.impl ||
            r.reserved || (r.flags & ~SynthRequired) || (r.opcode & ~SiteAuthBits) ||
            (!(r.opcode & SiteSigned) && (r.opcode || r.diversity)))
            return {Error::SynthOverride, i};
        firstSelector += r.selectorCount;
    }
    if (firstSelector != h.selectors.count)
        return {Error::SynthOverride, h.synthOverrides.count};
    uint32_t firstSynthOverride = 0;
    for (uint32_t i = 0; i < h.synthClasses.count; ++i) {
        auto r = v.row<SynthClass>(h.synthClasses, i);
        if (!v.string(r.name) || !v.string(r.parentClass) ||
            (r.siblingClass && !v.string(r.siblingClass)) || r.library >= h.libraries.count ||
            r.vtableOffset < 16 || r.vtableOffset % 8 || !r.vtableSlotCap || !r.parentRegion ||
            !r.overrideCount || r.firstOverride != firstSynthOverride ||
            !contains(r.firstOverride, r.overrideCount, h.synthOverrides.count))
            return {Error::SynthClass, i};
        firstSynthOverride += r.overrideCount;
    }
    if (firstSynthOverride != h.synthOverrides.count)
        return {Error::SynthClass, h.synthClasses.count};
    out = v;
    return {};
}

} // namespace manifest
} // namespace graft
