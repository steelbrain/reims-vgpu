// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Host test of the arm64e route planner (src/arm64e/Arm64Route.h). Built and
// run as native arm64e, so the re-sign uses real pointer authentication. The
// test stands up a mac_policy_ops buffer with the three admission slots signed
// the way AMFI's init signs them -- key IA, each operation's constant
// discriminator, no address diversity -- then builds the shadow and checks, with
// an independent oracle, that each shadow slot authenticates to the routed
// replacement under the same context, that the untouched bytes are unchanged,
// and that each guard (unknown operation, a slot not pointing into AMFI text, a
// slot signed with another discriminator, and a duplicate operation) refuses by
// index.
#include "Check.h"
#include "arm64e/Arm64Route.h"
#include <ptrauth.h>
#include <stdio.h>
#include <string.h>

using namespace graft;
using graft::arm64::Error;

namespace {

// AMFI signs each op slot with key IA, so the oracle needs only IA.
uint64_t refSignIA(uint64_t pointer, uint64_t disc) {
    return reinterpret_cast<uint64_t>(__builtin_ptrauth_sign_unauthenticated(
            reinterpret_cast<void *>(pointer), ptrauth_key_asia, disc));
}
uint64_t refAuthIA(uint64_t pointer, uint64_t disc) {
    return reinterpret_cast<uint64_t>(
            __builtin_ptrauth_auth(reinterpret_cast<void *>(pointer), ptrauth_key_asia, disc));
}

// A fabricated AMFI __TEXT range and the three original op pointers inside it.
constexpr uint64_t TextLo = 0xFFFFFE0007000000ULL;
constexpr uint64_t TextHi = 0xFFFFFE0007100000ULL;

// Stage a fresh ops buffer: a recognisable sentinel in every 8-byte slot, then
// each routed slot overwritten with AMFI's signed original op pointer.
void stage(uint8_t *ops, const uint64_t *originals) {
    for (size_t i = 0; i + sizeof(uint64_t) <= arm64::OpsSize; i += sizeof(uint64_t)) {
        uint64_t sentinel = 0xA11CE00000000000ULL | i;
        memcpy(ops + i, &sentinel, sizeof(sentinel));
    }
    for (size_t i = 0; i < arm64::OperationCount; ++i) {
        uint64_t signed_ = refSignIA(originals[i], arm64::Operations[i].discriminator);
        memcpy(ops + arm64::Operations[i].offset, &signed_, sizeof(signed_));
    }
}

void shadowBuilt() {
    uint64_t originals[arm64::OperationCount] = {TextLo + 0x1000, TextLo + 0x2000, TextLo + 0x3000};
    uint8_t ops[arm64::OpsSize];
    stage(ops, originals);

    // Route all three, in an order different from the table's, to a set of
    // replacement targets (raw addresses, as a client's stripped pointers).
    const char *names[] = {"vnode_check_signature", "file_check_mmap",
                           "file_check_library_validation"};
    uint64_t targets[] = {0xFFFFFE0008000000ULL, 0xFFFFFE0008001000ULL, 0xFFFFFE0008002000ULL};
    arm64::Slot slots[3]{};

    // A sentinel slot that no operation names, to confirm it is left untouched.
    uint64_t before8;
    memcpy(&before8, ops + 0x8, sizeof(before8));

    auto f = arm64::buildShadowOps(ops, sizeof(ops), names, targets, 3, TextLo, TextHi, slots);
    CHECK(f.error == Error::None);

    for (size_t i = 0; i < 3; ++i) {
        const arm64::Operation *op = arm64::operationFor(names[i]);
        CHECK(op != nullptr);
        CHECK(slots[i].offset == op->offset);
        // The captured original is the stripped live op pointer re-signed as a
        // plain C function pointer (key IA, discriminator 0), so the client calls
        // it directly; it authenticates under that schema to the original.
        size_t opIndex = static_cast<size_t>(op - &arm64::Operations[0]);
        CHECK(slots[i].original == refSignIA(originals[opIndex], 0));
        CHECK(refAuthIA(slots[i].original, 0) == originals[opIndex]);
        uint64_t value;
        memcpy(&value, ops + op->offset, sizeof(value));
        CHECK(value == slots[i].replacement);
        // The independent oracle: the shadow slot authenticates to the routed
        // target under the operation's own IA discriminator.
        CHECK(value == refSignIA(targets[i], op->discriminator));
        CHECK(refAuthIA(value, op->discriminator) == targets[i]);
        // The discriminator is actually used: signing the same target under a
        // different operation's discriminator yields a different slot value. (We
        // never authenticate with a wrong discriminator -- that traps under FPAC.)
        uint64_t other = arm64::Operations[(i + 1) % 3].discriminator;
        CHECK(value != refSignIA(targets[i], other));
    }

    uint64_t after8;
    memcpy(&after8, ops + 0x8, sizeof(after8));
    CHECK(after8 == before8);
    puts("arm64e route: pass (shadow slots re-signed, oracle, untouched bytes)");
}

void guards() {
    uint64_t originals[arm64::OperationCount] = {TextLo + 0x1000, TextLo + 0x2000, TextLo + 0x3000};
    uint8_t ops[arm64::OpsSize];
    arm64::Slot slots[3]{};
    uint64_t targets[] = {0xFFFFFE0008000000ULL, 0xFFFFFE0008001000ULL, 0xFFFFFE0008002000ULL};

    // An operation Graft does not route on arm64.
    {
        stage(ops, originals);
        const char *names[] = {"vnode_check_exec"};
        auto f = arm64::buildShadowOps(ops, sizeof(ops), names, targets, 1, TextLo, TextHi, slots);
        CHECK(f.error == Error::UnknownOperation && f.index == 0);
    }
    // A live slot whose stripped pointer lies outside AMFI __TEXT: the carried
    // offset is wrong or the table was mis-read, so refuse rather than mis-swap.
    {
        uint64_t strayed[arm64::OperationCount] = {TextLo + 0x1000, TextHi + 0x4000,
                                                   TextLo + 0x3000};
        stage(ops, strayed);
        const char *names[] = {"file_check_mmap", "file_check_library_validation"};
        auto f = arm64::buildShadowOps(ops, sizeof(ops), names, targets, 2, TextLo, TextHi, slots);
        CHECK(f.error == Error::SlotOutsideText && f.index == 1);
    }
    // A live slot signed with another discriminator than its operation's, as a
    // release that changed the operation's mpo_*_t prototype would sign it:
    // refuse, rather than route a slot the MAC framework would fail to
    // authenticate. Nothing is written for the refused slot.
    {
        stage(ops, originals);
        const auto &mmap = arm64::Operations[0];
        uint64_t resigned = refSignIA(originals[0], mmap.discriminator ^ 1);
        memcpy(ops + mmap.offset, &resigned, sizeof(resigned));
        const char *names[] = {"file_check_library_validation", "file_check_mmap"};
        auto f = arm64::buildShadowOps(ops, sizeof(ops), names, targets, 2, TextLo, TextHi, slots);
        CHECK(f.error == Error::SlotSignature && f.index == 1);
        uint64_t kept;
        memcpy(&kept, ops + mmap.offset, sizeof(kept));
        CHECK(kept == resigned);
    }
    // The same operation named twice in one request.
    {
        stage(ops, originals);
        const char *names[] = {"file_check_mmap", "file_check_mmap"};
        auto f = arm64::buildShadowOps(ops, sizeof(ops), names, targets, 2, TextLo, TextHi, slots);
        CHECK(f.error == Error::DuplicateOperation && f.index == 1);
    }
    // A short buffer, an inverted text range, and a null table are rejected.
    {
        stage(ops, originals);
        const char *names[] = {"file_check_mmap"};
        CHECK(arm64::buildShadowOps(ops, arm64::OpsSize - 1, names, targets, 1, TextLo, TextHi,
                                    slots)
                      .error == Error::Null);
        CHECK(arm64::buildShadowOps(ops, sizeof(ops), names, targets, 1, TextHi, TextLo, slots)
                      .error == Error::Null);
        CHECK(arm64::buildShadowOps(nullptr, sizeof(ops), names, targets, 1, TextLo, TextHi, slots)
                      .error == Error::Null);
    }
    puts("arm64e route: pass (unknown op, slot outside text, slot signature, duplicate, "
         "inputs)");
}

void chainedShadows() {
    // Several clients route one after another: each request copies the current
    // ops (AMFI's first, the last shadow after) and routes into the copy, so a
    // later client's shadow carries the earlier clients' routed slots unchanged.
    // The live chaining -- the swap from the current conf, the routed-operation
    // bookkeeping -- is in src/arm64e/Route.cpp and proven by booting the route
    // probe; here the ops-carrying property and the re-route guard are checked.
    uint64_t originals[arm64::OperationCount] = {TextLo + 0x1000, TextLo + 0x2000, TextLo + 0x3000};
    uint8_t ops[arm64::OpsSize];
    stage(ops, originals);

    // Client 1 routes file_check_mmap into a fresh copy of AMFI's ops.
    const char *first[] = {"file_check_mmap"};
    uint64_t firstTarget[] = {0xFFFFFE0008000000ULL};
    arm64::Slot firstSlot[1]{};
    CHECK(arm64::buildShadowOps(ops, sizeof(ops), first, firstTarget, 1, TextLo, TextHi, firstSlot)
                  .error == Error::None);
    const arm64::Operation *mmap = arm64::operationFor("file_check_mmap");
    uint64_t firstValue;
    memcpy(&firstValue, ops + mmap->offset, sizeof(firstValue));

    // Client 2 copies client 1's shadow and routes vnode_check_signature into it.
    uint8_t ops2[arm64::OpsSize];
    memcpy(ops2, ops, sizeof(ops2));
    const char *second[] = {"vnode_check_signature"};
    uint64_t secondTarget[] = {0xFFFFFE0008002000ULL};
    arm64::Slot secondSlot[1]{};
    CHECK(arm64::buildShadowOps(ops2, sizeof(ops2), second, secondTarget, 1, TextLo, TextHi,
                                secondSlot)
                  .error == Error::None);

    // Client 1's routed slot is carried into client 2's shadow unchanged.
    uint64_t carried;
    memcpy(&carried, ops2 + mmap->offset, sizeof(carried));
    CHECK(carried == firstValue);
    CHECK(refAuthIA(carried, mmap->discriminator) == firstTarget[0]);
    // Client 2's own slot is routed to its replacement.
    const arm64::Operation *sig = arm64::operationFor("vnode_check_signature");
    uint64_t sigValue;
    memcpy(&sigValue, ops2 + sig->offset, sizeof(sigValue));
    CHECK(sigValue == refSignIA(secondTarget[0], sig->discriminator));
    // Each captured original authenticates to AMFI's own function (index 0 is
    // file_check_mmap, index 2 is vnode_check_signature).
    CHECK(refAuthIA(firstSlot[0].original, 0) == originals[0]);
    CHECK(refAuthIA(secondSlot[0].original, 0) == originals[2]);

    // A third client that re-routes client 1's operation is refused: the live
    // slot now holds client 1's replacement, outside AMFI text, which is exactly
    // the guard that lets the installer hand back only AMFI's own originals. (The
    // installer also refuses it by name before reaching here; src/arm64e/Route.cpp.)
    uint8_t ops3[arm64::OpsSize];
    memcpy(ops3, ops2, sizeof(ops3));
    arm64::Slot dupSlot[1]{};
    auto dup = arm64::buildShadowOps(ops3, sizeof(ops3), first, firstTarget, 1, TextLo, TextHi,
                                     dupSlot);
    CHECK(dup.error == Error::SlotOutsideText && dup.index == 0);
    puts("arm64e route: pass (chained shadows carry prior routes; re-route of a held op refused)");
}

void confBuilt() {
    uint8_t conf[arm64::ConfSize];
    for (size_t i = 0; i < sizeof(conf); ++i)
        conf[i] = static_cast<uint8_t>(0x30 + (i & 0x3F));
    uint8_t before[arm64::ConfSize];
    memcpy(before, conf, sizeof(before));

    const uint64_t shadowOps = 0xFFFFFE0009abc000ULL;
    CHECK(arm64::buildShadowConf(conf, sizeof(conf), shadowOps) == Error::None);
    uint64_t stored;
    memcpy(&stored, conf + arm64::ConfOpsOffset, sizeof(stored));
    CHECK(stored == shadowOps); // written raw, no re-sign
    // Every byte outside the mpc_ops field is AMFI's verbatim.
    for (size_t i = 0; i < sizeof(conf); ++i)
        if (i < arm64::ConfOpsOffset || i >= arm64::ConfOpsOffset + sizeof(uint64_t))
            CHECK(conf[i] == before[i]);

    CHECK(arm64::buildShadowConf(nullptr, sizeof(conf), shadowOps) == Error::Null);
    CHECK(arm64::buildShadowConf(conf, sizeof(conf), 0) == Error::Null);
    CHECK(arm64::buildShadowConf(conf, arm64::ConfSize - 1, shadowOps) == Error::Bounds);
    puts("arm64e route: pass (shadow conf mpc_ops raw, verbatim rest, inputs)");
}

// Minimal A64 encoders for the forms resolvePolicyListBase decodes.
uint32_t adrp(uint32_t rd, uint64_t pc, uint64_t target) {
    int64_t pages = static_cast<int64_t>(target >> 12) - static_cast<int64_t>(pc >> 12);
    uint32_t immlo = static_cast<uint32_t>(pages) & 0x3;
    uint32_t immhi = static_cast<uint32_t>(pages >> 2) & 0x7FFFF;
    return 0x90000000u | (immlo << 29) | (immhi << 5) | (rd & 0x1F);
}
uint32_t addImm(uint32_t rd, uint32_t rn, uint32_t imm12) {
    return 0x91000000u | ((imm12 & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rd & 0x1F);
}
uint32_t ldr64(uint32_t rt, uint32_t rn, uint32_t byteOffset) {
    uint32_t imm12 = byteOffset / 8;
    return 0xF9400000u | ((imm12 & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F);
}
uint32_t ldr32(uint32_t rt, uint32_t rn, uint32_t byteOffset) {
    uint32_t imm12 = byteOffset / 4;
    return 0xB9400000u | ((imm12 & 0xFFF) << 10) | ((rn & 0x1F) << 5) | (rt & 0x1F);
}
constexpr uint32_t Nop = 0xD503201Fu;
constexpr uint32_t Ret = 0xD65F03C0u;

void policyListResolved() {
    const uint64_t funcAddr = 0xFFFFFE0007050000ULL;
    const uint64_t listBase = 0xFFFFFE000AC152A0ULL; // an 8-aligned nonzero lo12 (0x2a0)
    const uint64_t lo = listBase & 0xFFF;
    const uint32_t entries = arm64::PolicyListEntriesOffset;   // +0x18, 8-byte
    const uint32_t maxindex = arm64::PolicyListMaxIndexOffset; // +0x08, 4-byte

    // Shape 1 (classic): ADRP+ADD form the base, then both fields load off it --
    // LDR [base,#0x18] (entries) and LDR32 [base,#0x08] (maxindex).
    {
        uint32_t c[6];
        size_t i = 0;
        c[i] = 0xA9BF7BFDu, ++i; // stp x29,x30,[sp,#-16]! -- a decoy the scan ignores
        c[i] = adrp(24, funcAddr + i * 4, listBase), ++i;
        c[i] = addImm(24, 24, static_cast<uint32_t>(lo)), ++i;
        c[i] = ldr64(9, 24, entries), ++i;
        c[i] = ldr32(10, 24, maxindex), ++i;
        c[i] = Ret, ++i;
        CHECK(arm64::resolvePolicyListBase(reinterpret_cast<uint8_t *>(c), i * 4, funcAddr) ==
              listBase);
    }
    // Shape 2 (separate-ADRP): the base is formed in one register, but each field is
    // reached by a FRESH ADRP of the same page in another register, loading at the
    // full page-offset (lo+0x18, lo+0x08) -- not off the base register at all.
    {
        uint32_t c[8];
        size_t i = 0;
        c[i] = adrp(8, funcAddr + i * 4, listBase), ++i;
        c[i] = addImm(8, 8, static_cast<uint32_t>(lo)), ++i; // x8 = listBase (the base A)
        c[i] = Nop, ++i;
        c[i] = adrp(24, funcAddr + i * 4, listBase), ++i;
        c[i] = ldr64(9, 24, static_cast<uint32_t>(lo) + entries), ++i; // -> A+0x18
        c[i] = adrp(23, funcAddr + i * 4, listBase), ++i;
        c[i] = ldr32(10, 23, static_cast<uint32_t>(lo) + maxindex), ++i; // -> A+0x08
        c[i] = Ret, ++i;
        CHECK(arm64::resolvePolicyListBase(reinterpret_cast<uint8_t *>(c), i * 4, funcAddr) ==
              listBase);
    }
    // Shape 2 with the field loads far from the base formation (as on 13.6/14.6.1,
    // ~45 instructions out): the whole-body scan has no window, so it still resolves.
    {
        uint32_t c[64];
        size_t i = 0;
        c[i] = adrp(8, funcAddr + i * 4, listBase), ++i;
        c[i] = addImm(8, 8, static_cast<uint32_t>(lo)), ++i;
        for (int j = 0; j < 45; ++j)
            c[i] = Nop, ++i;
        c[i] = adrp(24, funcAddr + i * 4, listBase), ++i;
        c[i] = ldr64(9, 24, static_cast<uint32_t>(lo) + entries), ++i;
        c[i] = adrp(23, funcAddr + i * 4, listBase), ++i;
        c[i] = ldr32(10, 23, static_cast<uint32_t>(lo) + maxindex), ++i;
        c[i] = Ret, ++i;
        CHECK(arm64::resolvePolicyListBase(reinterpret_cast<uint8_t *>(c), i * 4, funcAddr) ==
              listBase);
    }
    // Entries at +0x18 but no maxindex at +0x08: not the list's signature.
    {
        uint32_t c[4];
        c[0] = adrp(8, funcAddr, listBase);
        c[1] = addImm(8, 8, static_cast<uint32_t>(lo));
        c[2] = ldr64(9, 8, entries);
        c[3] = Ret;
        CHECK(arm64::resolvePolicyListBase(reinterpret_cast<uint8_t *>(c), sizeof(c), funcAddr) ==
              0);
    }
    // Maxindex at +0x08 but no entries at +0x18: not the signature either.
    {
        uint32_t c[4];
        c[0] = adrp(8, funcAddr, listBase);
        c[1] = addImm(8, 8, static_cast<uint32_t>(lo));
        c[2] = ldr32(10, 8, maxindex);
        c[3] = Ret;
        CHECK(arm64::resolvePolicyListBase(reinterpret_cast<uint8_t *>(c), sizeof(c), funcAddr) ==
              0);
    }
    // Both fields at the wrong offsets, then no pattern, truncated, and null.
    {
        uint32_t c[5];
        c[0] = adrp(8, funcAddr, listBase);
        c[1] = addImm(8, 8, static_cast<uint32_t>(lo));
        c[2] = ldr64(9, 8, 0x20);
        c[3] = ldr32(10, 8, 0x10);
        c[4] = Ret;
        CHECK(arm64::resolvePolicyListBase(reinterpret_cast<uint8_t *>(c), sizeof(c), funcAddr) ==
              0);
    }
    {
        uint32_t c[3] = {Nop, Nop, Ret};
        CHECK(arm64::resolvePolicyListBase(reinterpret_cast<uint8_t *>(c), sizeof(c), funcAddr) ==
              0);
        CHECK(arm64::resolvePolicyListBase(reinterpret_cast<uint8_t *>(c), 2, funcAddr) == 0);
        CHECK(arm64::resolvePolicyListBase(nullptr, 64, funcAddr) == 0);
    }
    puts("arm64e route: pass (policy-list resolve: classic + separate-ADRP shapes, both fields)");
}

void scanCount() {
    const uint32_t wide = 0xFFFFFFFFu; // max large enough not to constrain
    CHECK(arm64::policyScanCount(0, wide) == arm64::PolicyScanFloor);
    CHECK(arm64::policyScanCount(arm64::PolicyScanFloor - 2, wide) == arm64::PolicyScanFloor);
    CHECK(arm64::policyScanCount(arm64::PolicyScanFloor, wide) == arm64::PolicyScanFloor + 1);
    CHECK(arm64::policyScanCount(100, wide) == 101);
    CHECK(arm64::policyScanCount(0xFFFFFFFFu, wide) == arm64::PolicyScanCeiling);
    // max bounds the scan below the ceiling and below the maxindex-derived count,
    // so a garbage-large maxindex never reads past the allocated entries array.
    CHECK(arm64::policyScanCount(0xFFFFFFFFu, 12) == 12);
    CHECK(arm64::policyScanCount(100, 4) == 4);
    CHECK(arm64::policyScanCount(2, 0) == 0);
    puts("arm64e route: pass (scan count floor, exact, ceiling, max bound)");
}

// The mac_policy ABI constants are derived from XNU's public security/mac_policy.h:
// the discriminator from each mpo_*_t prototype's function-pointer type
// discriminator, the offsets from field indices, the sizes from the layout. They
// are the same for any policy's table, AMFI's or another's. Pin them, so a
// toolchain change that moved any value fails here.
void constants() {
    CHECK(arm64::operationFor("file_check_mmap")->discriminator == 0x438d);
    CHECK(arm64::operationFor("file_check_library_validation")->discriminator == 0x9e37);
    CHECK(arm64::operationFor("vnode_check_signature")->discriminator == 0xcbd2);
    CHECK(arm64::operationFor("file_check_mmap")->offset == 0x120);
    CHECK(arm64::operationFor("file_check_library_validation")->offset == 0x200);
    CHECK(arm64::operationFor("vnode_check_signature")->offset == 0x980);
    CHECK(arm64::OpsSize == 0xa78);
    CHECK(arm64::ConfSize == 0x50);
    CHECK(arm64::ConfNameOffset == 0x0);
    CHECK(arm64::ConfOpsOffset == 0x20);
    puts("arm64e route: pass (mac_policy constants from the public header)");
}

} // namespace

int main() {
    constants();
    shadowBuilt();
    guards();
    chainedShadows();
    confBuilt();
    policyListResolved();
    scanCount();
}
