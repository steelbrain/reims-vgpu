// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Host test of the arm64e binding fixup (src/arm64e/Fixup.h). Built and run as
// native arm64e, so __builtin_ptrauth_* use real pointer-authentication. The
// test signs the placeholder the fixup expects, then checks the fixup's output
// authenticates back to the real target under the same key and discriminator --
// an oracle independent of the fixup's own signing. It also tampers the slot and
// the discriminator to confirm the precheck refuses rather than mis-signing.
#include "Check.h"
#include "arm64e/Fixup.h"
#include <ptrauth.h>
#include <stdio.h>

using namespace graft;
using graft::fixup::Error;

namespace {

// A version-5 auth byte: bit 0 signed, bits 1-2 key, bit 3 address diversity.
uint8_t opcodeFor(uint8_t key, bool addrDiv) {
    return 0x1 | static_cast<uint8_t>((key & 0x3) << 1) | (addrDiv ? 0x8 : 0x0);
}

uint64_t refSign(uint64_t pointer, uint8_t key, uint64_t disc) {
    auto p = reinterpret_cast<void *>(pointer);
    switch (key) {
    case 0:
        return reinterpret_cast<uint64_t>(
                __builtin_ptrauth_sign_unauthenticated(p, ptrauth_key_asia, disc));
    case 1:
        return reinterpret_cast<uint64_t>(
                __builtin_ptrauth_sign_unauthenticated(p, ptrauth_key_asib, disc));
    case 2:
        return reinterpret_cast<uint64_t>(
                __builtin_ptrauth_sign_unauthenticated(p, ptrauth_key_asda, disc));
    default:
        return reinterpret_cast<uint64_t>(
                __builtin_ptrauth_sign_unauthenticated(p, ptrauth_key_asdb, disc));
    }
}

uint64_t refAuth(uint64_t pointer, uint8_t key, uint64_t disc) {
    auto p = reinterpret_cast<void *>(pointer);
    switch (key) {
    case 0: return reinterpret_cast<uint64_t>(__builtin_ptrauth_auth(p, ptrauth_key_asia, disc));
    case 1: return reinterpret_cast<uint64_t>(__builtin_ptrauth_auth(p, ptrauth_key_asib, disc));
    case 2: return reinterpret_cast<uint64_t>(__builtin_ptrauth_auth(p, ptrauth_key_asda, disc));
    default: return reinterpret_cast<uint64_t>(__builtin_ptrauth_auth(p, ptrauth_key_asdb, disc));
    }
}

// The discriminator the kernel used, computed here independently of the fixup.
uint64_t refDisc(bool addrDiv, uint16_t diversity, uint64_t slot) {
    if (!addrDiv)
        return diversity;
    if (!diversity)
        return slot;
    return static_cast<uint64_t>(
            __builtin_ptrauth_blend_discriminator(reinterpret_cast<void *>(slot), diversity));
}

void signedSites() {
    // Kernel-half addresses, as a slid Auxiliary-collection consumer would carry.
    const uint64_t placeholder = 0xFFFFFE0011110000ULL;
    const uint64_t real = 0xFFFFFE0022220000ULL;
    const uint64_t slot = 0xFFFFFE0033330008ULL;
    struct Case {
        uint8_t key;
        bool addrDiv;
        uint16_t diversity;
    } cases[] = {
            {0, true, 0},       // IA, address diversity, no constant: full slot address
            {0, false, 0x90e1}, // IA, a constant diversity
            {1, true, 0},       // IB, address diversity
            {2, true, 0},       // DA, address diversity
            {3, true, 0},       // DB, address diversity
            {0, true, 0x1234},  // IA, address diversity blended with a constant
    };
    for (auto c : cases) {
        uint8_t opcode = opcodeFor(c.key, c.addrDiv);
        uint64_t disc = refDisc(c.addrDiv, c.diversity, slot);
        uint64_t old = refSign(placeholder, c.key, disc);
        uint64_t value = 0;
        auto e = fixup::replacement(8, opcode, 0, slot, old, placeholder, real, 0, value,
                                    c.diversity);
        CHECK(e == Error::None);
        CHECK(value == refSign(real, c.key, disc)); // signed identically to the reference
        CHECK(refAuth(value, c.key, disc) == real); // and authenticates to the real target
        // A tampered slot (one bit flipped in the signature) refuses.
        uint64_t ignored = 0;
        CHECK(fixup::replacement(8, opcode, 0, slot, old ^ 1, placeholder, real, 0, ignored,
                                 c.diversity) == Error::Placeholder);
        // The discriminator is actually used: a slot signed with the full address
        // instead of the blended discriminator refuses.
        if (c.addrDiv && c.diversity) {
            uint64_t wrong = refSign(placeholder, c.key, slot);
            CHECK(fixup::replacement(8, opcode, 0, slot, wrong, placeholder, real, 0, ignored,
                                     c.diversity) == Error::Placeholder);
        }
    }
    // A signed site with an addend: the kernel signed the bare placeholder, and
    // the addend lands only in the signed real target.
    {
        uint8_t opcode = opcodeFor(0, true); // IA, address diversity
        uint64_t disc = refDisc(true, 0, slot);
        uint64_t old = refSign(placeholder, 0, disc);
        uint64_t value = 0;
        CHECK(fixup::replacement(8, opcode, 0, slot, old, placeholder, real, 0x40, value, 0) ==
              Error::None);
        CHECK(refAuth(value, 0, disc) == real + 0x40);
    }
    puts("arm64e fixup: pass (signed sites, four keys, address diversity, tamper)");
}

void unsignedAndErrors() {
    const uint64_t placeholder = 0xFFFFFE0011110000ULL;
    const uint64_t real = 0xFFFFFE0022220000ULL;
    const uint64_t slot = 0xFFFFFE0033330010ULL;
    uint64_t value = 0;
    // A plain (unsigned) slot holds the raw placeholder; the fixup writes the raw target.
    CHECK(fixup::replacement(8, 0, 0, slot, placeholder, placeholder, real, 0, value) ==
          Error::None);
    CHECK(value == real);
    // A wrong placeholder refuses.
    CHECK(fixup::replacement(8, 0, 0, slot, placeholder + 1, placeholder, real, 0, value) ==
          Error::Placeholder);
    // Only 8-byte sites exist on arm64e.
    CHECK(fixup::replacement(4, 0, 0, slot, placeholder, placeholder, real, 0, value) ==
          Error::Width);
    // Undefined opcode bits refuse.
    CHECK(fixup::replacement(8, 0x10, 0, slot, placeholder, placeholder, real, 0, value) ==
          Error::Opcode);
    // The addend shifts only the target; arm64e stores the bare placeholder
    // (rebind_to_placeholders adds no addend), unlike the x86-64 convention.
    uint64_t shifted = 0;
    CHECK(fixup::replacement(8, 0, 0, slot, placeholder, placeholder, real, 0x20, shifted) ==
          Error::None);
    CHECK(shifted == real + 0x20);
    // A slot holding placeholder+addend (the x86-64 form) is refused on arm64e.
    CHECK(fixup::replacement(8, 0, 0, slot, placeholder + 0x20, placeholder, real, 0x20, shifted) ==
          Error::Placeholder);
    puts("arm64e fixup: pass (unsigned slot, width, opcode, addend)");
}

} // namespace

int main() {
    signedSites();
    unsignedAndErrors();
}
