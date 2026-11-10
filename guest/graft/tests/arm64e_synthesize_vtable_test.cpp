// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Host test of the synthesized-vtable builder (src/arm64e/SynthesizeVtable.h),
// native arm64e so it exercises the real pointer-authentication re-sign (as the
// fixup and route tests do). Synthetic inputs -- fake method addresses and
// discriminators, no Apple data: words 0,1 are zeroed; an inherited slot re-signs
// the live parent method, and an overridden slot the consumer implementation, each
// with the chain's discriminator for the slot, the most-base table's where an
// ancestor's covers it; a chain short of a slot refuses.
#include "Check.h"
#include "arm64e/SynthesizeVtable.h"
#include <stdint.h>
#include <stdio.h>
#include <vector>

using namespace graft;
using graft::synth::BuildError;
using graft::synth::DonorChain;
using graft::synth::OverridePlacement;

namespace {

void pack(std::vector<uint8_t> &blob, uint16_t slot, uint16_t div) {
    blob.push_back(slot & 0xff);
    blob.push_back(slot >> 8);
    blob.push_back(div & 0xff);
    blob.push_back(div >> 8);
    blob.push_back(0); // key IA
    blob.push_back(1); // addrDiv
    blob.push_back(0);
    blob.push_back(0);
}

void builds() {
    const uint32_t methods = 6; // live vtable words 2..7
    // The parent's table gives word s 0x1000 + (s - 2). An ancestor's covers words
    // 2..5 and carries 0x2003 on word 3 and 0x2005 on word 5, where the parent's
    // table carries its own hash: words 3 and 5 sign with the ancestor's values.
    std::vector<uint8_t> blob, baseBlob;
    for (uint32_t i = 0; i < methods; ++i)
        pack(blob, static_cast<uint16_t>(2 + i), static_cast<uint16_t>(0x1000 + i));
    for (uint32_t i = 0; i < 4; ++i)
        pack(baseBlob, static_cast<uint16_t>(2 + i),
             static_cast<uint16_t>(i == 1 || i == 3 ? 0x2000 + 2 + i : 0x1000 + i));
    DonorChain chain;
    CHECK(synth::parseDonorTable(0, 2, blob.data(), blob.size(), chain.tables[0]).error ==
          synth::DonorError::None);
    CHECK(synth::parseDonorTable(0, 2, baseBlob.data(), baseBlob.size(), chain.tables[1]).error ==
          synth::DonorError::None);
    chain.count = 2;
    auto div = [](uint32_t slot) {
        return static_cast<uint16_t>(slot == 3 || slot == 5 ? 0x2000 + slot : 0x1000 + slot - 2);
    };

    // Synthetic live parent method words (stand-ins for signed live pointers) and a
    // destination the vtable will live at.
    uint64_t liveParent[methods];
    for (uint32_t i = 0; i < methods; ++i)
        liveParent[i] = 0xffff'0000'0010'0000ull + i * 0x40;
    const uint64_t destBase = 0xffff'0000'0020'0000ull;

    // Overrides on word 3, where the ancestor's value applies, and word 6, where
    // only the parent's table reaches. Word 5 stays inherited under the ancestor's.
    OverridePlacement placements[] = {
            {3, 0xffff'0000'0030'0000ull},
            {6, 0xffff'0000'0031'0000ull},
    };

    uint64_t out[2 + methods];
    auto r =
            synth::buildSynthesizedVtable(out, methods, destBase, liveParent, chain, placements, 2);
    CHECK(r.error == BuildError::None);
    CHECK(out[0] == 0 && out[1] == 0);

    for (uint32_t i = 0; i < methods; ++i) {
        uint32_t slot = 2 + i;
        uint64_t site = destBase + static_cast<uint64_t>(slot) * 8;
        uint64_t impl = slot == 3   ? placements[0].impl
                        : slot == 6 ? placements[1].impl
                                    : liveParent[i];
        uint64_t want = 0;
        CHECK(fixup::vtableSlot(impl, 0x9, div(slot), site, want) == fixup::Error::None);
        CHECK(out[slot] == want);
    }
    // The override on word 3 is signed with the ancestor's value, not the parent
    // table's: the value its callers authenticate with.
    uint64_t parentSigned = 0;
    CHECK(fixup::vtableSlot(placements[0].impl, 0x9, 0x1001, destBase + 3 * 8, parentSigned) ==
          fixup::Error::None);
    CHECK(out[3] != parentSigned);
    puts("arm64e synthesize vtable: pass (inherited + override signed from the chain)");
}

void refusesShortDonor() {
    // A donor table covering only slots 2..4 cannot fill a 6-method vtable.
    std::vector<uint8_t> blob;
    for (uint32_t i = 0; i < 3; ++i)
        pack(blob, static_cast<uint16_t>(2 + i), static_cast<uint16_t>(0x1000 + i));
    DonorChain chain;
    CHECK(synth::parseDonorTable(0, 2, blob.data(), blob.size(), chain.tables[0]).error ==
          synth::DonorError::None);
    chain.count = 1;
    uint64_t liveParent[6] = {1, 2, 3, 4, 5, 6};
    uint64_t out[8];
    auto r = synth::buildSynthesizedVtable(out, 6, 0x1000, liveParent, chain, nullptr, 0);
    CHECK(r.error == BuildError::Donor && r.slot == 5);
    puts("arm64e synthesize vtable: pass (short donor refuses)");
}

} // namespace

int main() {
    builds();
    refusesShortDonor();
}
