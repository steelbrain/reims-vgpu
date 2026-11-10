// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
#pragma once
#include "Binding.h"
#include "MachO.h"
#include "Manifest.h"
#include "Resolve.h"

namespace graft {

// The kernel retains each loaded provider for the lifetime of the consumer.
// These views are constructed only from checked mappings of those images.
struct Provider {
    macho::SymbolTable symbols;
};

// Translate a validated manifest and preflight all import writes. Caller-owned
// scratch avoids allocations in the planning code. No write or IOKit
// publication happens here, even if a later site fails. map(offset, size) is a
// consumer offset's runtime address, or 0 outside the image, so a coalesced
// image (arm64e) resolves it through its segment table rather than loadAddress +
// offset. read(offset, width, old, opcode) serves the width old bytes and a
// branch's leading opcode byte through the live mapping. unresolved is the
// pointer gOSKextUnresolved holds (binding::unresolvedValue), or 0.
template <typename Map, typename Read>
binding::Failure planConsumerSites(const manifest::View &view, const Provider *providers,
                                   size_t providerCount, uint64_t unresolved, Map map, Read read,
                                   binding::Site *scratch, binding::Write *writes,
                                   size_t capacity) {
    using namespace binding;
    if (!view.valid() || providerCount != view.header().libraries.count ||
        (providerCount && !providers))
        return {Error::Metadata};
    const auto &h = view.header();
    if (h.sites.count > capacity || (h.sites.count && (!scratch || !writes)))
        return {Error::Capacity};
    for (uint32_t i = 0; i < h.sites.count; ++i) {
        auto s = view.row<manifest::Site>(h.sites, i);
        uint64_t placeholder = map(s.placeholder, 1);
        if (!placeholder)
            return {Error::Bounds, i};
        scratch[i] = {s.offset, placeholder, s.addend, s.width, s.opcode, s.reserved, s.import};
    }
    auto find = [&](uint32_t library, const char *name, uint64_t &address) {
        if (library >= providerCount || !providers[library].symbols.valid())
            return Resolution::Invalid;
        return unique(providers[library].symbols, name, address);
    };
    auto resolve = [&](const Site &site, uint64_t &address) {
        return resolveImport(view, site.import, find, unresolved, map, address);
    };
    auto sign = [&](uint64_t offset) { return map(offset, 1); };
    return plan(scratch, h.sites.count, writes, capacity, resolve, read, sign);
}

// Check where each planned site now points, before anything is written. A
// site must reach its provider's image (code, for a branch), or its import's
// own fallback in the consumer, or, for a weak import nothing defines and that
// has no fallback, exactly unresolved, the pointer gOSKextUnresolved holds:
// the address the kernel's linker gives such an import, which the consumer
// tests and never follows. map(offset, size) resolves a branch's next
// instruction and a fallback, as in planConsumerSites. inProvider(library,
// address, code) and inConsumer(address, code) report the live mappings. A
// refusal names the site: Bounds where a branch's target overflows, Target
// where it lies anywhere else.
template <typename Map, typename InProvider, typename InConsumer>
binding::Failure checkTargets(const manifest::View &view, Map map, uint64_t unresolved,
                              const binding::Write *writes, InProvider inProvider,
                              InConsumer inConsumer) {
    const auto &h = view.header();
    for (uint32_t i = 0; i < h.sites.count; ++i) {
        auto site = view.row<manifest::Site>(h.sites, i);
        auto import = view.row<manifest::Import>(h.imports, site.import);
        bool code = site.width == 4;
        // The committed value is the real address, signed on arm64e; strip the
        // signature so the live mappings locate it (x86-64 strip is identity).
        uint64_t target = fixup::stripCode(writes[i].after);
        if (code) {
            int64_t displacement = target <= INT32_MAX
                                           ? static_cast<int64_t>(target)
                                           : static_cast<int64_t>(target) - (int64_t{1} << 32);
            uint64_t next = map(site.offset, site.width);
            if (!next || next > UINT64_MAX - 4 || !fixup::add(next + 4, displacement, target))
                return {binding::Error::Bounds, i};
        }
        if (inProvider(import.library, target, code))
            continue;
        // The import's own address, before the site's addend.
        __int128 real = static_cast<__int128>(target) - site.addend;
        uint64_t fallback = import.fallback != manifest::NoFallback ? map(import.fallback, 1) : 0;
        bool valid =
                import.fallback != manifest::NoFallback
                        ? fallback && real == static_cast<__int128>(fallback) &&
                                  inConsumer(target, code)
                        : (import.flags & manifest::ImportWeak) && unresolved && real == unresolved;
        if (!valid)
            return {binding::Error::Target, i};
    }
    return {};
}

} // namespace graft
