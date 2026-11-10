// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Resolve only within the declared provider, with explicit candidate order
// and a consumer-owned fallback. Ambiguity never silently selects a copy.
#pragma once
#include "Binding.h"
#include "MachO.h"
#include "Manifest.h"

namespace graft {
namespace binding {

inline Resolution unique(const macho::SymbolTable &table, const char *name, uint64_t &address) {
    address = 0;
    for (uint32_t i = 0; i < table.count; ++i) {
        auto candidate = macho::definedName(table, i);
        if (!candidate || strcmp(candidate, name))
            continue;
        uint64_t value = macho::symbolAddress(table, i);
        if (!value)
            return Resolution::Invalid;
        if (address && address != value) {
            address = 0;
            return Resolution::Ambiguous;
        }
        address = value;
    }
    return address ? Resolution::Found : Resolution::Missing;
}

// The address the kernel's linker gives a weak import nothing defines: the
// pointer stored in the kernel's gOSKextUnresolved. OSKextSymbolIsResolved()
// compares a weak symbol's address against that value, not against the
// variable's own address. variable is gOSKextUnresolved's address, or 0;
// read(address, value) loads eight readable bytes. 0 where there is none.
template <typename Read>
uint64_t unresolvedValue(uint64_t variable, Read read) {
    uint64_t value = 0;
    return variable && read(variable, value) ? value : 0;
}

// find(libraryIndex, name, address) returns a unique definition in that
// library. The returned address is consumed only for Found. A fallback offset
// refers to code/data already mapped with this dormant consumer; map(offset,
// size) gives its runtime address, or 0 where it lies outside the consumer, so
// a coalesced image (arm64e) resolves it through its segment table rather than
// loadAddress + offset. unresolved is unresolvedValue(), or 0 where the kernel
// has none: a weak import that no candidate or fallback resolves binds there.
template <typename Find, typename Map>
Resolution resolveImport(const manifest::View &view, uint32_t index, Find find, uint64_t unresolved,
                         Map map, uint64_t &address) {
    address = 0;
    if (!view.valid() || index >= view.header().imports.count)
        return Resolution::Invalid;
    const auto &h = view.header();
    auto r = view.row<manifest::Import>(h.imports, index);
    for (uint32_t i = 0; i < r.candidateCount; ++i) {
        auto name = view.string(view.row<uint32_t>(h.candidates, r.firstCandidate + i));
        uint64_t candidate = 0;
        auto result = find(r.library, name, candidate);
        if (result == Resolution::Missing)
            continue;
        if (result != Resolution::Found)
            return result;
        if (!candidate)
            return Resolution::Invalid;
        address = candidate;
        return Resolution::Found;
    }
    if (r.fallback == manifest::NoFallback) {
        if (!(r.flags & manifest::ImportWeak) || !unresolved)
            return Resolution::Missing;
        address = unresolved;
        return Resolution::Found;
    }
    address = map(r.fallback, 1);
    return address ? Resolution::Found : Resolution::Invalid;
}

} // namespace binding
} // namespace graft
