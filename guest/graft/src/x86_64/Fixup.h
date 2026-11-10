// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Checked x86-64 binding arithmetic. No kernel calls and no writes. The error
// set and the address-plus-addend helper are shared (../FixupShared.h).
#pragma once
#include "../FixupShared.h"

namespace graft {
namespace fixup {

// A pointer, or the four-byte displacement of E8/E9. The caller supplies the
// byte immediately before a displacement as actualOpcode. On failure, value is
// untouched. Both targets include the original relocation addend. diversity is
// unused on x86-64 (it carries an arm64e signature); the parameter keeps one
// call signature across architectures for Binding.h.
inline Error replacement(uint8_t width, uint8_t opcode, uint8_t actualOpcode, uint64_t site,
                         uint64_t old, uint64_t placeholder, uint64_t real, int64_t addend,
                         uint64_t &value, uint16_t /*diversity*/ = 0) {
    if (width != 4 && width != 8)
        return Error::Width;
    if (width == 8 ? opcode != 0 : ((opcode != 0xe8 && opcode != 0xe9) || actualOpcode != opcode))
        return Error::Opcode;
    uint64_t expected, target;
    if (!add(placeholder, addend, expected) || !add(real, addend, target))
        return Error::Address;
    if (width == 8) {
        if (old != expected)
            return Error::Placeholder;
        value = target;
        return Error::None;
    }
    uint64_t next, priorTarget;
    if (old > UINT32_MAX || !add(site, 4, next))
        return Error::Address;
    int64_t priorDisplacement = old <= INT32_MAX ? static_cast<int64_t>(old)
                                                 : static_cast<int64_t>(old) - (int64_t{1} << 32);
    if (!add(next, priorDisplacement, priorTarget))
        return Error::Address;
    if (priorTarget != expected)
        return Error::Placeholder;
    // Subtract in the wider domain: kernel addresses exceed INT64_MAX.
    __int128 displacement = static_cast<__int128>(target) - next;
    if (displacement < INT32_MIN || displacement > INT32_MAX)
        return Error::Range;
    value = static_cast<uint32_t>(displacement);
    return Error::None;
}

// A code pointer with its signature removed. x86-64 pointers are unsigned, so
// this is the identity; the arm64e sibling strips the pointer-authentication
// bits. It keeps one call signature across architectures for Vtable.h.
inline uint64_t stripCode(uint64_t pointer) {
    return pointer;
}

// The value for an inherited consumer vtable slot, taken from the live
// provider's slot. x86-64 copies it verbatim; the arm64e sibling re-signs the
// provider method for the consumer's own slot. opcode, diversity and site carry
// the arm64e signing context and the slot's runtime address, unused here.
inline Error vtableSlot(uint64_t provider, uint8_t /*opcode*/, uint16_t /*diversity*/,
                        uint64_t /*site*/, uint64_t &value) {
    value = provider;
    return Error::None;
}

} // namespace fixup
} // namespace graft
