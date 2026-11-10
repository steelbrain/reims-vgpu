// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
#pragma once
#include "Binding.h"
#include "Layout.h"
#include "Manifest.h"

namespace graft {

struct ActualClass {
    uint32_t instanceSize;
    uint32_t slotCount;
    const uint8_t *slots;
};
enum class VtableError { None, Metadata, Class, Method, Capacity, Bounds, Override };
struct VtableFailure {
    VtableError error = VtableError::None;
    uint32_t record = 0;
    uint32_t slot = 0;
    layout::Error detail = layout::Error::None;
};

// Parent slot mappings are checked and retained by the caller. read(offset,
// value) bounds each consumer word; map(offset) is a consumer slot's runtime
// address, 0 outside the image, which arm64e re-signing blends into the slot's
// discriminator and a coalesced image resolves through its segment table;
// code(address) admits only executable consumer code; defines(provider, name,
// address) includes that provider's retained dependencies. A provider slot and
// a kept slot are stripped before their checks, since arm64e signs them
// (x86-64 strip is identity). Output is usable only on whole-plan success.
template <typename Read, typename Map, typename Code, typename Defines>
VtableFailure planVtables(const manifest::View &view, const ActualClass *classes, size_t classCount,
                          binding::Write *writes, size_t capacity, size_t &written, Read read,
                          Map map, Code code, Defines defines) {
    written = 0;
    if (!view.valid() || classCount != view.header().classes.count || (classCount && !classes))
        return {VtableError::Metadata};
    const auto &h = view.header();
    for (uint32_t i = 0; i < h.classes.count; ++i) {
        auto declared = view.row<manifest::Class>(h.classes, i);
        // A synthesized class's parent is measured, not layout-checked: its declared
        // stand-in size and slot count are placeholders a consumer cannot know at
        // build time, and the live layout is per-release-variable -- synthesis sizes
        // its vtable to the live parent instead of verifying a declaration.
        // actualClasses still measures it for synthesis.
        bool synthParent = false;
        for (uint32_t s = 0; s < h.synthClasses.count && !synthParent; ++s)
            synthParent = !strcmp(
                    view.string(view.row<manifest::SynthClass>(h.synthClasses, s).parentClass),
                    view.string(declared.name));
        if (synthParent)
            continue;
        auto &actual = classes[i];
        layout::Requirement required{view.string(declared.name), declared.instanceSize,
                                     declared.parentSlots};
        auto match = layout::compare(required, actual.instanceSize, actual.slotCount);
        if (match != layout::Error::None || !actual.slots)
            return {VtableError::Class, i, 0,
                    match != layout::Error::None ? match : layout::Error::Unreadable};
        for (uint32_t j = 0; j < declared.slotCount; ++j) {
            auto slot = view.row<manifest::Slot>(h.slots, declared.firstSlot + j);
            uint64_t value;
            memcpy(&value, actual.slots + static_cast<size_t>(slot.index) * 8, 8);
            if (!value ||
                !defines(declared.library, view.string(slot.method), fixup::stripCode(value)))
                return {VtableError::Method, i, slot.index, layout::Error::Method};
        }
    }
    size_t count = 0;
    for (uint32_t i = 0; i < h.vtables.count; ++i) {
        auto table = view.row<manifest::Vtable>(h.vtables, i);
        auto declared = view.row<manifest::Class>(h.classes, table.classIndex);
        auto &actual = classes[table.classIndex];
        // The live vtable must be at least as long as the declared parent slots the
        // loop copies from it. For a layout-checked class this already holds (the
        // first loop required equality); a class skipped there as a synthesized
        // parent was never measured against its declaration, so bound the copy here
        // rather than read past the live vtable. manifest::open also refuses a
        // Vtable record of a synthesized parent, so this is a second line of defence.
        if (!actual.slots || actual.slotCount < declared.parentSlots)
            return {VtableError::Bounds, i, 0};
        uint32_t nextOverride = 0, nextFill = 0;
        for (uint32_t j = 0; j < table.slotCount; ++j) {
            uint64_t offset = table.offset + static_cast<uint64_t>(j) * 8;
            uint64_t before;
            if (!read(offset, before))
                return {VtableError::Bounds, i, j};
            bool keep = j >= declared.parentSlots;
            if (nextOverride < table.overrideCount &&
                view.row<uint32_t>(h.overrides, table.firstOverride + nextOverride) == j) {
                keep = true;
                ++nextOverride;
            }
            if (keep) {
                // An override or appended slot already points to the consumer's
                // own code, signed locally on arm64e; strip before the check.
                if (!code(fixup::stripCode(before)))
                    return {VtableError::Override, i, j};
                continue;
            }
            if (count == capacity || !writes)
                return {VtableError::Capacity, i, j};
            uint64_t providerValue;
            memcpy(&providerValue, actual.slots + static_cast<size_t>(j) * 8, 8);
            uint64_t slot = map(offset);
            if (!slot)
                return {VtableError::Bounds, i, j};
            // Re-sign the inherited method for this slot's own runtime address
            // using the signing context the consumer's build recorded (arm64e);
            // on x86-64 fixup::vtableSlot copies the provider pointer verbatim
            // and ignores the fill, and x86-64 carries none (fillCount 0).
            uint8_t opcode = 0;
            uint16_t diversity = 0;
            if (table.fillCount) {
                auto fill = view.row<manifest::Fill>(h.fills, table.firstFill + nextFill);
                opcode = fill.opcode;
                diversity = fill.diversity;
            }
            ++nextFill;
            uint64_t after;
            if (fixup::vtableSlot(providerValue, opcode, diversity, slot, after) !=
                fixup::Error::None)
                return {VtableError::Override, i, j};
            writes[count++] = {offset, before, after, 8};
        }
    }
    written = count;
    return {};
}

} // namespace graft
