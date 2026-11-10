// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Host test of the donor vtable-discriminator parser (src/arm64e/VtableDisc.h). It
// builds SYNTHETIC records -- made-up slots and discriminators -- and a synthetic
// live vtable signed with them, so it commits no Apple discriminator data: a
// well-formed table parses and reads back its discriminators, and every divergence
// from the measured arm64e scheme (wrong first-slot, a short blob, an out-of-order
// slot, a non-IA key, no address diversity, a vtable-base or slot-count that
// misses the live class, a discriminator the live slot is not signed with)
// refuses. A chain gives each slot its most-base table's discriminator, and an
// ancestor table longer than its subclass's refuses. Built and run as native
// arm64e, so the live vtable is signed, and checked, with real pointer
// authentication.
#include "Check.h"
#include "arm64e/VtableDisc.h"
#include <initializer_list>
#include <ptrauth.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>

using namespace graft;
using graft::synth::DonorError;
using graft::synth::DonorTable;

namespace {

// Pack one 8-byte record: { u16 slot, u16 div, u8 key, u8 addrDiv, u8 pad[2] }, LE.
void pack(std::vector<uint8_t> &blob, uint16_t slot, uint16_t div, uint8_t key, uint8_t addrDiv) {
    blob.push_back(slot & 0xff);
    blob.push_back(slot >> 8);
    blob.push_back(div & 0xff);
    blob.push_back(div >> 8);
    blob.push_back(key);
    blob.push_back(addrDiv);
    blob.push_back(0);
    blob.push_back(0);
}

// A valid blob: method slots 2..2+count-1, IA / address-diversified, div = slot*7+1.
std::vector<uint8_t> validBlob(uint32_t count) {
    std::vector<uint8_t> blob;
    for (uint32_t i = 0; i < count; ++i)
        pack(blob, static_cast<uint16_t>(2 + i), static_cast<uint16_t>((2 + i) * 7 + 1), 0, 1);
    return blob;
}

void parses() {
    auto blob = validBlob(5);
    DonorTable table;
    auto r = synth::parseDonorTable(0xffff000012340000ull, 2, blob.data(), blob.size(), table);
    CHECK(r.error == DonorError::None);
    CHECK(table.count == 5 && table.firstSlot == 2 && table.vtableBase == 0xffff000012340000ull);

    // Each method slot reads back its discriminator; out-of-range refuses.
    uint16_t div = 0;
    CHECK(synth::donorDiv(table, 2, div) && div == 2 * 7 + 1);
    CHECK(synth::donorDiv(table, 6, div) && div == 6 * 7 + 1);
    CHECK(!synth::donorDiv(table, 1, div)); // before the first method slot
    CHECK(!synth::donorDiv(table, 7, div)); // past the last
    puts("arm64e vtable-disc: pass (parse + read back)");
}

// A live vtable method word as the kernel's loader signs it, independently of
// the code under test: key IA, the slot's address blended with the method's
// discriminator.
uint64_t signedSlot(uint64_t target, uint64_t vtable, uint32_t slot, uint16_t div) {
    uint64_t modifier = static_cast<uint64_t>(__builtin_ptrauth_blend_discriminator(
            reinterpret_cast<void *>(vtable + uint64_t{slot} * 8), div));
    return reinterpret_cast<uint64_t>(__builtin_ptrauth_sign_unauthenticated(
            reinterpret_cast<void *>(target), ptrauth_key_asia, modifier));
}

void crossChecksLive() {
    const uint64_t vtable = 0xfffffe0012340000ull;
    auto blob = validBlob(10);
    DonorTable table;
    CHECK(synth::parseDonorTable(vtable, 2, blob.data(), blob.size(), table).error ==
          DonorError::None);
    // The live method words, each signed for its own slot with validBlob's s*7+1.
    uint64_t live[10];
    for (uint32_t i = 0; i < 10; ++i)
        live[i] = signedSlot(0xfffffe0007000000ull + i * 0x40, vtable, 2 + i,
                             static_cast<uint16_t>((2 + i) * 7 + 1));
    // 10 method slots means a live vtable of 12 words (2 header + 10 methods).
    CHECK(synth::checkDonorLive(table, vtable, 12, live).error == DonorError::None);
    CHECK(synth::checkDonorLive(table, 0xdead00ull, 12, live).error == DonorError::VtableBase);
    CHECK(synth::checkDonorLive(table, vtable, 13, live).error == DonorError::SlotCount);
    CHECK(synth::checkDonorLive(table, vtable, 1, live).error == DonorError::SlotCount);

    // A table of the right shape whose discriminator for slot 5 is not the one
    // the live slot is signed with refuses, naming the slot.
    auto wrong = validBlob(10);
    wrong[3 * 8 + 2] ^= 1; // record 3 is slot 5
    DonorTable wrongTable;
    CHECK(synth::parseDonorTable(vtable, 2, wrong.data(), wrong.size(), wrongTable).error ==
          DonorError::None);
    auto mismatch = synth::checkDonorLive(wrongTable, vtable, 12, live);
    CHECK(mismatch.error == DonorError::Signature && mismatch.slot == 5);

    // So does a live word signed for another slot address, an unsigned word,
    // and an empty one.
    uint64_t moved[10];
    for (uint64_t word :
         {signedSlot(0xfffffe0007000100ull, vtable + 8, 9, static_cast<uint16_t>(9 * 7 + 1)),
          uint64_t{0xfffffe0007000100ull}, uint64_t{0}}) {
        memcpy(moved, live, sizeof(moved));
        moved[7] = word; // slot 9
        auto r = synth::checkDonorLive(table, vtable, 12, moved);
        CHECK(r.error == DonorError::Signature && r.slot == 9);
    }
    puts("arm64e vtable-disc: pass (live cross-check: address, length, each discriminator)");
}

void refuses() {
    DonorTable table;
    auto blob = validBlob(4);

    // A node whose first-slot is not the two-word header boundary.
    CHECK(synth::parseDonorTable(0, 3, blob.data(), blob.size(), table).error ==
          DonorError::FirstSlot);
    // A blob that is not a whole number of records, or empty.
    CHECK(synth::parseDonorTable(0, 2, blob.data(), blob.size() - 1, table).error ==
          DonorError::BlobSize);
    CHECK(synth::parseDonorTable(0, 2, blob.data(), 0, table).error == DonorError::BlobSize);

    // A record out of slot order.
    auto gap = validBlob(4);
    gap[2 * 8] = 9; // record 2's slot low byte -> 9 instead of 4
    auto res = synth::parseDonorTable(0, 2, gap.data(), gap.size(), table);
    CHECK(res.error == DonorError::Order && res.slot == 9);

    // A key that is not IA.
    auto key = validBlob(4);
    key[1 * 8 + 4] = 1; // record 1's key -> IB
    CHECK(synth::parseDonorTable(0, 2, key.data(), key.size(), table).error == DonorError::Key);

    // A slot without address diversity.
    auto ad = validBlob(4);
    ad[3 * 8 + 5] = 0; // record 3's addrDiv -> 0
    CHECK(synth::parseDonorTable(0, 2, ad.data(), ad.size(), table).error == DonorError::AddrDiv);
    puts("arm64e vtable-disc: pass (refuse-by-name on divergence)");
}

void discriminatorFallback() {
    // validBlob gives method slot s the discriminator s*7+1, each distinct, so a
    // value one slot carries resolves to that slot, a value none carries does not,
    // and 0 never matches (a real method type discriminator is nonzero).
    auto blob = validBlob(6); // method slots 2..7
    DonorTable table;
    CHECK(synth::parseDonorTable(0x1000ull, 2, blob.data(), blob.size(), table).error ==
          DonorError::None);
    uint32_t slot = 0xffffffff;
    CHECK(synth::uniqueDiscriminatorSlot(table, 4 * 7 + 1, slot) && slot == 4);
    CHECK(synth::uniqueDiscriminatorSlot(table, 7 * 7 + 1, slot) && slot == 7);
    CHECK(!synth::uniqueDiscriminatorSlot(table, 9999, slot)); // no slot carries it
    CHECK(!synth::uniqueDiscriminatorSlot(table, 0, slot));    // 0 never matches

    // Two slots sharing a discriminator is ambiguous: no unique match, so the
    // caller leaves the override unresolved rather than place it on a guess.
    auto dup = validBlob(4);         // method slots 2..5
    dup[2 * 8 + 2] = dup[0 * 8 + 2]; // record 2's div (slot 4) <- record 0's (slot 2)
    dup[2 * 8 + 3] = dup[0 * 8 + 3];
    DonorTable dtable;
    CHECK(synth::parseDonorTable(0x2000ull, 2, dup.data(), dup.size(), dtable).error ==
          DonorError::None);
    CHECK(!synth::uniqueDiscriminatorSlot(dtable, static_cast<uint16_t>(2 * 7 + 1), slot));
    puts("arm64e vtable-disc: pass (discriminator fallback)");
}

void chainTakesMostBase() {
    // The parent's table covers method slots 2..9 with s*7+1; an ancestor's covers
    // 2..5 and carries another value on slot 4, as IOMobileFramebuffer does where
    // IOMobileFramebufferLegacy redeclares a method pure. Slot 4 takes the
    // ancestor's value, the other ancestor slots their shared one, and the slots
    // past the ancestor the parent's.
    auto parentBlob = validBlob(8);
    auto baseBlob = validBlob(4);
    baseBlob[2 * 8 + 2] = 0x34; // record 2 is slot 4: div 0x1234
    baseBlob[2 * 8 + 3] = 0x12;
    synth::DonorChain chain;
    CHECK(synth::parseDonorTable(0x1000ull, 2, parentBlob.data(), parentBlob.size(),
                                 chain.tables[0])
                  .error == DonorError::None);
    CHECK(synth::parseDonorTable(0x2000ull, 2, baseBlob.data(), baseBlob.size(), chain.tables[1])
                  .error == DonorError::None);
    CHECK(synth::checkDonorAncestor(chain.tables[1], chain.tables[0]).error == DonorError::None);
    chain.count = 2;
    uint16_t div = 0;
    CHECK(synth::chainDiv(chain, 4, div) && div == 0x1234);
    CHECK(synth::chainDiv(chain, 3, div) && div == 3 * 7 + 1);
    CHECK(synth::chainDiv(chain, 9, div) && div == 9 * 7 + 1);
    CHECK(!synth::chainDiv(chain, 10, div)); // past the parent's last slot
    CHECK(!synth::chainDiv(chain, 1, div));  // before the first method slot

    // A table of one covers what the parent's covers, as before chains.
    chain.count = 1;
    CHECK(synth::chainDiv(chain, 4, div) && div == 4 * 7 + 1);

    // An ancestor's vtable is a prefix of its subclass's: a longer table refuses.
    CHECK(synth::checkDonorAncestor(chain.tables[0], chain.tables[1]).error ==
          DonorError::Ancestor);
    puts("arm64e vtable-disc: pass (chain takes the most-base table's discriminator)");
}

} // namespace

int main() {
    parses();
    crossChecksLive();
    refuses();
    discriminatorFallback();
    chainTakesMostBase();
}
