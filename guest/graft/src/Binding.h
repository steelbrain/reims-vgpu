// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Preflight a dormant consumer's sites. A plan is usable only on success;
// this layer never writes the consumer or opens its activation gate.
#pragma once
#include "Fixup.h"
#include <stddef.h>

namespace graft {
namespace binding {

// Runtime views, not the manifest's on-disk representation. The reader must
// construct these from validated consumer metadata. A site's placeholder is its
// runtime address, and import indexes the manifest's imports, which the resolver
// reads its candidates and library from.
struct Site {
    uint64_t offset;
    uint64_t placeholder;
    int64_t addend;
    uint8_t width;
    uint8_t opcode;
    uint16_t diversity = 0; // the arm64e signature's constant diversity; 0 on x86-64
    uint32_t import = 0;
};

struct Write {
    uint64_t offset;
    uint64_t before;
    uint64_t after;
    uint8_t width;
};

// Whether [address, address + width) overlaps [start, start + size). Used to keep
// a staged write out of a forbidden region, such as the consumer's own manifest
// section, which Graft reads back after the commit. Overflow is treated as no
// overlap, since the caller has already bounded each range within the image.
inline bool overlaps(uint64_t address, uint64_t width, uint64_t start, uint64_t size) {
    if (!width || !size || width - 1 > UINT64_MAX - address || size - 1 > UINT64_MAX - start)
        return false;
    return address < start + size && start < address + width;
}

enum class Resolution { Missing, Found, Ambiguous, Invalid };
enum class Error { None, Capacity, Bounds, Order, Missing, Ambiguous, Metadata, Fixup, Target };
// A failure's site is the index of the site it refers to, or NoSite for one that
// refers to no particular site, such as a plan with too little room.
constexpr size_t NoSite = SIZE_MAX;
struct Failure {
    Error error = Error::None;
    size_t site = NoSite;
    fixup::Error detail = fixup::Error::None;
};

// Sites must be sorted by offset and disjoint. The resolver owns the image
// scope and candidate/fallback policy, and must refuse ambiguous symbols rather
// than quietly choosing a local copy. read(offset, width, old, opcode) reads the
// width old bytes at a site, and a branch's opcode byte before it, through the
// consumer's live mapping, and bounds them; it returns false when the site is
// not wholly within a readable segment. sign(offset) is the site's runtime
// address, the modifier for an arm64e signature and the base for a branch
// displacement. The two coincide in the running consumer; a coalesced image
// (arm64e) resolves both through its segment map, so neither is loadAddress +
// offset. The caller must also check segment permissions and lifetime.
template <typename Resolve, typename Read, typename Sign>
Failure plan(const Site *sites, size_t count, Write *writes, size_t capacity, Resolve resolve,
             Read read, Sign sign) {
    if (count > capacity || (count && (!sites || !writes)))
        return {Error::Capacity};
    uint64_t end = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto &s = sites[i];
        if (s.width != 4 && s.width != 8)
            return {Error::Fixup, i, fixup::Error::Width};
        // Include a branch's opcode in its range, since it too is read.
        uint64_t start = s.offset - (s.width == 4 ? 1u : 0u);
        if (i && start < end)
            return {Error::Order, i};
        end = s.offset + s.width;
        uint64_t site = sign(s.offset);
        uint64_t old = 0;
        uint8_t actualOpcode = 0;
        if (!site || !read(s.offset, s.width, old, actualOpcode))
            return {Error::Bounds, i};
        uint64_t target = 0;
        auto resolution = resolve(s, target);
        if (resolution == Resolution::Invalid)
            return {Error::Metadata, i};
        if (resolution == Resolution::Ambiguous)
            return {Error::Ambiguous, i};
        if (resolution == Resolution::Missing || !target)
            return {Error::Missing, i};
        uint64_t value = 0;
        auto result = fixup::replacement(s.width, s.opcode, actualOpcode, site, old, s.placeholder,
                                         target, s.addend, value, s.diversity);
        if (result != fixup::Error::None)
            return {Error::Fixup, i, result};
        writes[i] = {s.offset, old, value, s.width};
    }
    return {};
}

} // namespace binding
} // namespace graft
