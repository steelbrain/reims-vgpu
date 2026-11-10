// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Build a consumer's synthesized arm64e vtable from the live parent and the donor
// discriminator chain. The consumer subclasses a
// header-less, per-release-variable parent (IOMobileFramebufferLegacy), so its
// vtable is sized and ordered to the LIVE parent, not its compile-time stand-in:
// every method word is the live parent's method re-signed for the new slot's
// address, except the words a consumer override lands on, which take the override's
// own implementation. The arm64e re-sign is the binding's own fixup::vtableSlot.
// Every slot, inherited or overridden, signs with the discriminator its callers
// authenticate with, the chain's (VtableDisc.h chainDiv), never the consumer's
// recorded one: that one is hashed from the stand-in's declaration, which is not
// always the method's first.
//
// This is the pure transform over the measured inputs: the kernel layer measures
// the live parent vtable and instance size, resolves each override to its live slot
// (Synthesize.h planSynthesis), then calls this to fill a buffer whose words it
// writes over the consumer's own vtable, in place. Host-tested native
// arm64e (tests/arm64e_synthesize_vtable_test.cpp), as the fixup and route tests are.
#pragma once
#include "../Fixup.h"
#include "VtableDisc.h"
#include <stddef.h>
#include <stdint.h>

namespace graft {
namespace synth {

// One override's placement in the synthesized vtable: the live vtable WORD index
// planSynthesis resolved it to (>= DonorFirstSlot) and the consumer's
// implementation (its runtime code address).
struct OverridePlacement {
    uint32_t slot;
    uint64_t impl;
};

enum class BuildError { None, Inherited, Override, Donor };
struct BuildResult {
    BuildError error = BuildError::None;
    uint32_t slot = 0; // the word slot the error refers to
};

// Fill `out` -- DonorFirstSlot + methodSlots words -- with the synthesized vtable.
// destBase is the slid address `out` will live at; liveParent[i] is the live
// parent's method word i (its signed pointer, as read from the live vtable); chain
// is the validated donor chain (checkDonorLive already matched its parent table's
// count to methodSlots); placements[0,placementCount) map resolved overrides to
// their slots (planSynthesis guarantees no two share a slot). Words 0,1
// (offset-to-top, typeinfo) are 0; each method word re-signs for its own slot
// address with the chain's discriminator for that slot -- an override's
// implementation where one lands there, else the live parent's method. Output is
// usable only on BuildError::None.
inline BuildResult buildSynthesizedVtable(uint64_t *out, uint32_t methodSlots, uint64_t destBase,
                                          const uint64_t *liveParent, const DonorChain &chain,
                                          const OverridePlacement *placements,
                                          uint32_t placementCount) {
    out[0] = 0; // offset-to-top
    out[1] = 0; // typeinfo: kernel C++ carries none
    for (uint32_t i = 0; i < methodSlots; ++i) {
        uint32_t slot = DonorFirstSlot + i;
        uint64_t site = destBase + static_cast<uint64_t>(slot) * 8;
        const OverridePlacement *hit = nullptr;
        for (uint32_t k = 0; k < placementCount; ++k)
            if (placements[k].slot == slot) {
                hit = &placements[k];
                break;
            }
        uint16_t div;
        if (!chainDiv(chain, slot, div))
            return {BuildError::Donor, slot};
        uint64_t value;
        if (fixup::vtableSlot(hit ? hit->impl : liveParent[i], DonorOpcode, div, site, value) !=
            fixup::Error::None)
            return {hit ? BuildError::Override : BuildError::Inherited, slot};
        out[slot] = value;
    }
    return {};
}

} // namespace synth
} // namespace graft
