// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Checked arm64e binding arithmetic: an authenticated 8-byte pointer slot. Every
// arm64e import site is a GOT slot the kernel filled and signed when it loaded
// the Auxiliary collection, so the slot already holds a signed pointer to the
// local placeholder. Graft re-signs the real target under
// the same key and modifier. Graft's kmod start runs under the kernel's live PAC
// keys, so a pointer signed here is one the consumer authenticates. The error set
// and the address-plus-addend helper are shared (../FixupShared.h).
#pragma once
#include "../FixupShared.h"
#include <ptrauth.h>

namespace graft {
namespace fixup {

// The version-5 auth byte (manifest::Site::opcode, ManifestFormat.h): bit 0 signed, bits 1-2
// the key (0=IA 1=IB 2=DA 3=DB), bit 3 address diversity. Literals here keep this
// arithmetic header independent of the manifest layer.
inline uint64_t signPointer(uint64_t pointer, uint8_t key, uint64_t discriminator) {
    auto p = reinterpret_cast<void *>(pointer);
    // The key must be a constant, so select it rather than pass a variable.
    switch (key & 0x3) {
    case 0:
        return reinterpret_cast<uint64_t>(
                __builtin_ptrauth_sign_unauthenticated(p, ptrauth_key_asia, discriminator));
    case 1:
        return reinterpret_cast<uint64_t>(
                __builtin_ptrauth_sign_unauthenticated(p, ptrauth_key_asib, discriminator));
    case 2:
        return reinterpret_cast<uint64_t>(
                __builtin_ptrauth_sign_unauthenticated(p, ptrauth_key_asda, discriminator));
    default:
        return reinterpret_cast<uint64_t>(
                __builtin_ptrauth_sign_unauthenticated(p, ptrauth_key_asdb, discriminator));
    }
}

// The discriminator the kernel signed this slot with, and so the one the consumer
// authenticates with: a constant diversity, the slot's full runtime address when
// the site is address-diversified with no constant (the __auth_got access is `add
// xN; braa x16, xN`, no movk, so the modifier is the whole 64-bit slot address),
// or that address blended with a nonzero constant.
inline uint64_t discriminator(uint8_t opcode, uint16_t diversity, uint64_t slot) {
    if (!(opcode & 0x8)) // no address diversity
        return diversity;
    if (!diversity)
        return slot;
    return static_cast<uint64_t>(
            __builtin_ptrauth_blend_discriminator(reinterpret_cast<void *>(slot), diversity));
}

// The replacement value for an 8-byte arm64e import slot. site is the slot's
// runtime address. A plain slot (opcode 0) holds the rebased placeholder; a signed
// slot holds that placeholder signed, which this verifies before re-signing the
// real target identically, so a wrong key or discriminator refuses rather than
// writing a pointer the consumer cannot authenticate. actualOpcode is unused:
// arm64e has no branch sites.
inline Error replacement(uint8_t width, uint8_t opcode, uint8_t /*actualOpcode*/, uint64_t site,
                         uint64_t old, uint64_t placeholder, uint64_t real, int64_t addend,
                         uint64_t &value, uint16_t diversity = 0) {
    if (width != 8)
        return Error::Width;
    if ((opcode & 0xf0) != 0) // only the four version-5 auth bits are defined
        return Error::Opcode;
    // rebindToPlaceholders (sdk/tool/consumer.cpp) rewrites each provider import to a
    // rebase at the bare placeholder, with no addend, so the kernel signed the
    // slot over `placeholder` alone; the addend lands only in the real target.
    uint64_t target;
    if (!add(real, addend, target))
        return Error::Address;
    uint64_t expected = placeholder;
    if (!(opcode & 0x1)) { // an unsigned, plain-pointer slot
        if (opcode != 0)
            return Error::Opcode;
        if (old != expected)
            return Error::Placeholder;
        value = target;
        return Error::None;
    }
    uint8_t key = (opcode >> 1) & 0x3;
    uint64_t disc = discriminator(opcode, diversity, site);
    if (old != signPointer(expected, key, disc))
        return Error::Placeholder;
    value = signPointer(target, key, disc);
    return Error::None;
}

// A code pointer with its pointer-authentication signature removed, not
// authenticated: __builtin_ptrauth_strip clears the signature bits with no
// braa/autia, so it never faults under FEAT_FPAC. Every arm64e key shares the
// bit layout, so IA strips a vtable method pointer or a local override alike.
inline uint64_t stripCode(uint64_t pointer) {
    return reinterpret_cast<uint64_t>(
            __builtin_ptrauth_strip(reinterpret_cast<void *>(pointer), ptrauth_key_asia));
}

// The value for an inherited consumer vtable slot: the live provider's slot,
// signed for the provider's own slot address, re-signed for the consumer's.
// provider is the provider's slot value; site is the consumer slot's runtime
// address; opcode and diversity are the consumer's signing context for that slot
// (manifest Fill), captured from the consumer's own vtable fixup. A plain
// (unsigned) slot takes the stripped pointer.
inline Error vtableSlot(uint64_t provider, uint8_t opcode, uint16_t diversity, uint64_t site,
                        uint64_t &value) {
    if ((opcode & 0xf0) != 0)
        return Error::Opcode;
    uint64_t real = stripCode(provider);
    if (!(opcode & 0x1)) {
        if (opcode != 0)
            return Error::Opcode;
        value = real;
        return Error::None;
    }
    value = signPointer(real, (opcode >> 1) & 0x3, discriminator(opcode, diversity, site));
    return Error::None;
}

} // namespace fixup
} // namespace graft
