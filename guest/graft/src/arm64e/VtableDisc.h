// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Parse and validate the donor vtable-discriminator tables the boot firmware hands
// over in `/chosen/ambrosia/vtable-disc`. Synthesis fills a consumer's vtable in
// the live parent's layout and must sign each slot for the consumer slot's
// address; the arm64e signing discriminator is a per-method type value that the
// kernel consumes at load and does not leave in the runtime pointer, so the boot
// firmware measures it from the Boot KC on disk, per named vtable symbol, and
// hands Graft a table of it. Each exported symbol's sub-node
// carries vtable-base, first-slot and a `slots` blob of 8-byte little-endian
// records { u16 slot, u16 div, u8 key, u8 addrDiv, u8 pad[2] }, one per method
// slot in order.
//
// A caller authenticates a slot with the discriminator of the method's first
// declaration, in the most-base class that has the slot. A class's own table
// usually carries that value, but not always: IOMobileFramebufferLegacy's signs
// the methods it redeclares pure with its own hash, where IOMobileFramebuffer,
// which declares them first, and every caller use theirs. So Graft takes the
// parent's table and its ancestors' (a DonorChain) and gives each slot the
// discriminator of the most-base table that covers it.
//
// This header is the pure parse and the refuse-by-name checks: a future build that
// diverges from the measured scheme (not IA, not address-diversified, out of order,
// with another vtable address or slot count than the live vtable Graft resolves,
// or with a discriminator that does not sign the live slot's pointer) refuses the
// bind rather than mis-signs. The kernel layer reads each device-tree node, calls
// parseDonorTable, checkDonorLive against the live vtable it measured and, for an
// ancestor, checkDonorAncestor, then reads chainDiv(slot) for each slot it fills.
// checkDonorLive re-signs with real pointer authentication, so the host test
// (tests/arm64e_vtable_disc_test.cpp) runs native arm64e against synthetic
// records and a synthetic vtable; it commits no Apple discriminator data.
#pragma once
#include "../Fixup.h"
#include <stddef.h>
#include <stdint.h>

namespace graft {
namespace synth {

// The method slots begin after the vtable's offset-to-top and typeinfo words.
constexpr uint32_t DonorFirstSlot = 2;
// A `slots` record: { u16 slot, u16 div, u8 key, u8 addrDiv, u8 pad[2] }.
constexpr uint32_t DonorRecordSize = 8;
// The measured arm64e scheme every method slot of a handed table carries.
constexpr uint8_t DonorKeyIA = 0;   // key field: 0 = IA
constexpr uint8_t DonorAddrDiv = 1; // address diversity on
// The fixup auth byte for a synthesized slot: signed, key IA, address diversity --
// the scheme parseDonorTable enforces (Fixup.h: bit0 signed, bits1-2 key, bit3
// addrDiv). Every slot, inherited or overridden, signs with this and the slot's
// chain discriminator.
constexpr uint8_t DonorOpcode = 0x9;
// The most tables a chain holds: the parent's and its ancestors' within its
// provider (IOMobileFramebufferLegacy, IOMobileFramebufferService,
// IOMobileFramebuffer), with room for one more.
constexpr uint32_t MaxDonorChain = 4;

enum class DonorError {
    None,
    FirstSlot,  // the node's first-slot is not DonorFirstSlot
    BlobSize,   // the blob is empty or not a whole number of records
    Order,      // a record's slot is not firstSlot + its index
    Key,        // a record's key is not IA
    AddrDiv,    // a record's addrDiv is not set
    VtableBase, // the handed vtable-base is not the live vtable address
    SlotCount,  // the record count is not the live method-slot count
    Ancestor,   // an ancestor's table is longer than its subclass's
    Signature,  // a record's discriminator does not sign the live slot's pointer
};

struct DonorResult {
    DonorError error = DonorError::None;
    uint32_t slot = 0; // the record slot an Order/Key/AddrDiv/Signature error refers to
};

// A validated view over one vtable symbol's donor records.
struct DonorTable {
    const uint8_t *records = nullptr; // the `slots` blob
    uint32_t count = 0;               // number of method-slot records
    uint32_t firstSlot = 0;           // DonorFirstSlot
    uint64_t vtableBase = 0;          // slid VA of vtable word 0
};

inline uint16_t donorU16(const uint8_t *p) {
    return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8));
}

// Validate the node's fields and blob and fill `table`. Refuses when first-slot is
// not DonorFirstSlot, the blob is not a whole number of 8-byte records, a record's
// slot is out of order (so slot == firstSlot + index holds for every record), or a
// record's key/addrDiv is not the measured arm64e scheme.
inline DonorResult parseDonorTable(uint64_t vtableBase, uint32_t firstSlot, const uint8_t *blob,
                                   size_t blobLen, DonorTable &table) {
    if (firstSlot != DonorFirstSlot)
        return {DonorError::FirstSlot, firstSlot};
    if (!blob || blobLen == 0 || blobLen % DonorRecordSize)
        return {DonorError::BlobSize, 0};
    uint32_t count = static_cast<uint32_t>(blobLen / DonorRecordSize);
    for (uint32_t i = 0; i < count; ++i) {
        const uint8_t *r = blob + static_cast<size_t>(i) * DonorRecordSize;
        uint32_t slot = donorU16(r);
        if (slot != firstSlot + i)
            return {DonorError::Order, slot};
        if (r[4] != DonorKeyIA)
            return {DonorError::Key, slot};
        if (r[5] != DonorAddrDiv)
            return {DonorError::AddrDiv, slot};
    }
    table = {blob, count, firstSlot, vtableBase};
    return {};
}

// The discriminator for method word `slot`, or false if `slot` is outside the
// table's method range.
inline bool donorDiv(const DonorTable &table, uint32_t slot, uint16_t &div) {
    if (slot < table.firstSlot || slot - table.firstSlot >= table.count)
        return false;
    div = donorU16(table.records + static_cast<size_t>(slot - table.firstSlot) * DonorRecordSize +
                   2);
    return true;
}

// Cross-check a parsed table against the live vtable Graft measured: the handed
// vtable-base must equal the live vtable address Graft resolved for the symbol, and
// the record count must equal the live method-slot count (liveSlotCount minus the
// two header words). Each record's discriminator must then be the one the live
// slot is signed with: liveMethods holds the live vtable's method words, as read,
// and re-signing slot i's stripped pointer with the blend of slot i's address and
// the record's discriminator must reproduce word i. So a table measured from a
// different image, or one with the right shape but another discriminator, refuses
// rather than fills the wrong discriminators.
inline DonorResult checkDonorLive(const DonorTable &table, uint64_t liveVtableAddr,
                                  uint32_t liveSlotCount, const uint64_t *liveMethods) {
    if (table.vtableBase != liveVtableAddr)
        return {DonorError::VtableBase, 0};
    if (liveSlotCount < table.firstSlot || table.count != liveSlotCount - table.firstSlot)
        return {DonorError::SlotCount, 0};
    for (uint32_t slot = table.firstSlot; slot < table.firstSlot + table.count; ++slot) {
        uint64_t live = liveMethods[slot - table.firstSlot];
        uint16_t div = 0;
        uint64_t resigned = 0;
        if (!donorDiv(table, slot, div) ||
            fixup::vtableSlot(live, DonorOpcode, div, liveVtableAddr + uint64_t{slot} * 8,
                              resigned) != fixup::Error::None ||
            resigned != live)
            return {DonorError::Signature, slot};
    }
    return {};
}

// An ancestor's vtable is a prefix of its subclass's, so its table covers no
// more slots than the subclass's does. A longer one is not this class's ancestor
// as measured, and refuses rather than lends its discriminators.
inline DonorResult checkDonorAncestor(const DonorTable &ancestor, const DonorTable &derived) {
    if (ancestor.count > derived.count)
        return {DonorError::Ancestor, ancestor.firstSlot + ancestor.count};
    return {};
}

// The parent's table first, then each ancestor's toward the root, every one
// checked against its live vtable and the one before it.
struct DonorChain {
    DonorTable tables[MaxDonorChain];
    uint32_t count = 0;
};

// The discriminator a caller authenticates method word `slot` with: the record of
// the most-base table in `chain` that covers it. False when no table does.
inline bool chainDiv(const DonorChain &chain, uint32_t slot, uint16_t &div) {
    for (uint32_t i = chain.count; i-- > 0;)
        if (donorDiv(chain.tables[i], slot, div))
            return true;
    return false;
}

// The unique method word whose donor discriminator equals `div`, written to
// `slot`, or false when no word has it, more than one does, or `div` is 0 -- so a
// caller never guesses. `table` is the parent's own, and `div` the discriminator
// the consumer's build recorded for an override, hashed from the stand-in's
// declaration of the method in the parent class. The parent's own table carries
// that same hash on each slot whose method the parent itself declares, pure
// (___cxa_pure_virtual) or not, so such an override whose method has no
// resolvable symbol this release (an unexported sibling method, or a parameter
// form its recorded name does not match) is still placed on the right slot.
// Placement only: the slot is then signed with chainDiv's value. A real method
// type discriminator is nonzero, so a `div` of 0 never matches.
inline bool uniqueDiscriminatorSlot(const DonorTable &table, uint16_t div, uint32_t &slot) {
    if (div == 0)
        return false;
    uint32_t matches = 0, found = 0;
    for (uint32_t s = table.firstSlot; s < table.firstSlot + table.count; ++s) {
        uint16_t d;
        if (donorDiv(table, s, d) && d == div) {
            ++matches;
            found = s;
        }
    }
    if (matches != 1)
        return false;
    slot = found;
    return true;
}

} // namespace synth
} // namespace graft
