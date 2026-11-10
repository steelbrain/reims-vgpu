// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// arm64e route planning: the AMFI mac_policy_ops shadow. On x86-64 Graft
// inline-hooks each MAC dispatcher's entry; on arm64 the collection's text and
// AMFI's ops table are read-only when Graft runs, so a client's admission hooks
// are routed by shadowing AMFI's policy table instead. This header is pure
// logic: it derives AMFI's ABI constants from XNU's public security/mac_policy.h,
// maps a route request's operation to its mpo field, and builds the shadow ops
// table from a copy of AMFI's live table, re-signing each routed slot. The
// kernel layer (src/arm64e/Route.cpp) owns resolving AMFI's symbols, reading the
// live bytes, allocating the shadow, repointing the policy list and the write
// transaction. Re-signing uses real pointer authentication, so the host test
// (tests/arm64e_route_test.cpp) runs native arm64e, as the fixup test does.
#pragma once
#include "Fixup.h"
#include <stddef.h>
#include <stdint.h>

// The route layer signs pointers and derives the AMFI ABI from the public
// function-pointer type discriminators, both of which need an arm64e
// (pointer-authentication) target. Keep this header to the arm64e build.
#if !__has_feature(ptrauth_calls)
#error "Graft's arm64e route layer requires an arm64e pointer-authentication target"
#endif

namespace graft {
namespace arm64 {

// The three mac_policy_ops entries an owned client may route on arm64, named by
// operation (not by AMFI's local, unstable op symbols). AMFI signs each ops slot
// with key IA, the slot type's function-pointer discriminator and no address
// diversity; the signatures do not blend the storage address, which is what lets
// a relocated shadow table keep valid signatures. The discriminator for a slot
// is the type discriminator of its mpo_*_t prototype, so Graft derives it from
// that prototype rather than carrying a literal, and buildShadowOps checks it
// against the live slot's signature before routing. The prototypes are written
// with pointer parameters standing in for the kernel struct types, since the
// function-pointer type discriminator ignores pointee types; they are the
// prototypes of mpo_file_check_mmap_t, mpo_file_check_library_validation_t and
// mpo_vnode_check_signature_t in XNU's public security/mac_policy.h.
using FileCheckMmapFn = int(void *, void *, void *, int, int, uint64_t, int *);
using FileCheckLibraryValidationFn = int(void *, void *, long long, long long, unsigned long);
using VnodeCheckSignatureFn = int(void *, void *, int, void *, unsigned *, unsigned *, int,
                                  unsigned, char **, unsigned long *);

// Field indices in struct mac_policy_ops (security/mac_policy.h). Every field is
// a pointer, so a field's byte offset in the table is its index times 8.
inline constexpr uint32_t FileCheckMmapIndex = 36;
inline constexpr uint32_t FileCheckLibraryValidationIndex = 64;
inline constexpr uint32_t VnodeCheckSignatureIndex = 304;

struct Operation {
    const char *name;
    uint32_t offset;
    uint8_t key; // ptrauth key, as fixup::signPointer takes it: 0 = IA
    uint16_t discriminator;
};

inline constexpr Operation Operations[] = {
        {"file_check_mmap", FileCheckMmapIndex * 8, 0,
         __builtin_ptrauth_type_discriminator(FileCheckMmapFn)},
        {"file_check_library_validation", FileCheckLibraryValidationIndex * 8, 0,
         __builtin_ptrauth_type_discriminator(FileCheckLibraryValidationFn)},
        {"vnode_check_signature", VnodeCheckSignatureIndex * 8, 0,
         __builtin_ptrauth_type_discriminator(VnodeCheckSignatureFn)},
};
inline constexpr size_t OperationCount = sizeof(Operations) / sizeof(Operations[0]);

// struct mac_policy_ops and struct mac_policy_conf are not in the Kernel
// framework SDK, so their sizes follow from the layout of XNU's public
// security/mac_policy.h: mac_policy_ops is a table of 335 function pointers, and
// mac_policy_conf carries mpc_ops at offset 0x20. The shadow copies the whole
// structs, so only mpc_ops, the pointer repointed at the shadow ops, needs a
// named offset.
inline constexpr size_t OpsSize = 335 * 8;    // sizeof(struct mac_policy_ops)
inline constexpr size_t ConfSize = 0x50;      // sizeof(struct mac_policy_conf)
inline constexpr size_t ConfOpsOffset = 0x20; // offsetof(struct mac_policy_conf, mpc_ops)

// The MAC framework dispatches through a global list of registered policies,
// struct mac_policy_list in security/mac_internal.h. mac_policy_list has no symbol
// (Graft resolves it at runtime, below); its u_int max is at +0x04, its u_int
// maxindex at +0x08 and its entries pointer at +0x18, and each element is a bare
// pointer to a mac_policy_conf (stride 8, mpc at offset 0). Repointing AMFI's
// element at a shadow conf is the one writable store. Both pointers Graft writes
// -- the element's mpc and the shadow conf's mpc_ops -- are stored raw (no
// pointer authentication), so only the three op function pointers in the shadow
// ops are re-signed.
inline constexpr size_t PolicyListMaxOffset = 0x04;      // u_int max, the allocated entry count
inline constexpr size_t PolicyListMaxIndexOffset = 0x08; // u_int maxindex, highest populated index
inline constexpr size_t PolicyListEntriesOffset = 0x18;
inline constexpr size_t PolicyElementStride = 8;
inline constexpr size_t PolicyElementMpcOffset = 0;
// entries[0..maxindex] inclusive are the populated elements, and entries holds
// max of them. The scan is floored at one chunk (always allocated, so safe to
// read even if maxindex is small), bounded by max so it never runs past the
// allocated array, and ceilinged against an implausible count from a corrupt
// read.
inline constexpr size_t PolicyScanFloor = 8;
inline constexpr size_t PolicyScanCeiling = 256;

enum class Error {
    None,
    Null,
    Bounds,
    UnknownOperation,
    DuplicateOperation,
    SlotOutsideText,
    SlotSignature
};

inline const char *describe(Error error) {
    switch (error) {
    case Error::None: return "ok";
    case Error::Null: return "route input";
    case Error::Bounds: return "staged copy too small for the structure";
    case Error::UnknownOperation: return "operation not routable on arm64";
    case Error::DuplicateOperation: return "operation routed twice in one request";
    case Error::SlotOutsideText: return "live ops slot does not point into AMFI text";
    case Error::SlotSignature:
        return "live ops slot is not signed with the operation's discriminator";
    }
    return "unknown route error";
}

struct Failure {
    Error error = Error::None;
    size_t index = 0;
};

// The operation row for `name`, or null. Route.cpp maps each RouteRequest's
// symbol (its operation name on arm64, Route.h) through this.
inline const Operation *operationFor(const char *name) {
    if (!name)
        return nullptr;
    for (size_t i = 0; i < OperationCount; ++i)
        if (!__builtin_strcmp(name, Operations[i].name))
            return &Operations[i];
    return nullptr;
}

// One routed slot: the byte offset in the ops table it occupies, a callable
// original (AMFI's original op pointer, stripped and re-signed as a plain C
// function pointer, which the kernel layer hands back as the route's
// `original`), and the re-signed replacement the shadow carries.
struct Slot {
    uint32_t offset;
    uint64_t original;
    uint64_t replacement;
};

// Build the shadow ops table in place. `ops` is a writable copy of AMFI's live
// mac_policy_ops, `size` its length (OpsSize). For each request i, the routed
// slot's live value is read from the copy, its signature stripped (never faults
// under FPAC) and range-checked into AMFI's __TEXT [textLo, textHi) -- a guard
// that the carried offset constant truly names an AMFI op pointer, refusing by
// index rather than mis-swapping on any mismatch. The stripped pointer, signed
// again with the operation's (key, discriminator), must then reproduce the live
// value: the slot carries no address diversity, so the signature depends only
// on the pointer and the discriminator. A release whose mpo_*_t prototype, and
// so its discriminator, differs from the one Graft derives refuses here, rather
// than routing a slot that the MAC framework's first call through it would fail
// to authenticate. The replacement target is stripped of whatever signature the
// caller's pointer carries and re-signed for the slot's own (key,
// discriminator), exactly as AMFI signed the original, so the MAC framework
// authenticates the shadow slot identically. The captured
// original is that stripped live pointer re-signed as a plain C function pointer
// (key IA, discriminator 0), so the client calls it directly. `names[i]` is the
// operation name of request i; `targets[i]` its replacement. On any failure the
// output and `ops` past the already-written slots are left as the caller staged
// them, and the failing index is returned; the kernel layer discards the shadow.
inline Failure buildShadowOps(uint8_t *ops, size_t size, const char *const *names,
                              const uint64_t *targets, size_t count, uint64_t textLo,
                              uint64_t textHi, Slot *slots) {
    if (!ops || !names || !targets || (count && !slots) || size < OpsSize || textLo >= textHi)
        return {Error::Null, 0};
    for (size_t i = 0; i < count; ++i) {
        const Operation *op = operationFor(names[i]);
        if (!op)
            return {Error::UnknownOperation, i};
        if (op->offset > size - sizeof(uint64_t))
            return {Error::Bounds, i};
        for (size_t j = 0; j < i; ++j)
            if (slots[j].offset == op->offset)
                return {Error::DuplicateOperation, i};
        uint64_t live;
        __builtin_memcpy(&live, ops + op->offset, sizeof(live));
        uint64_t original = fixup::stripCode(live);
        if (original < textLo || original >= textHi)
            return {Error::SlotOutsideText, i};
        if (fixup::signPointer(original, op->key, op->discriminator) != live)
            return {Error::SlotSignature, i};
        uint64_t value =
                fixup::signPointer(fixup::stripCode(targets[i]), op->key, op->discriminator);
        __builtin_memcpy(ops + op->offset, &value, sizeof(value));
        // Re-sign the stripped original as a plain arm64e C function pointer
        // (key IA, discriminator 0), so the client's `original` is callable
        // directly rather than only under the slot's own discriminator.
        uint64_t callable = fixup::signPointer(original, 0, 0);
        slots[i] = {op->offset, callable, value};
    }
    return {};
}

// Build the shadow mac_policy_conf in place. `conf` is a writable copy of AMFI's
// live conf (ConfSize bytes); `shadowOps` is the raw runtime address of the
// shadow ops table. AMFI stores conf.mpc_ops unsigned, so the shadow's mpc_ops
// is written raw. Every other field is AMFI's verbatim, so
// the MAC framework reads the same policy name, flags, load-time handle and
// label slot through the shadow.
inline Error buildShadowConf(uint8_t *conf, size_t size, uint64_t shadowOps) {
    if (!conf || !shadowOps)
        return Error::Null;
    if (size < ConfSize)
        return Error::Bounds;
    __builtin_memcpy(conf + ConfOpsOffset, &shadowOps, sizeof(shadowOps));
    return Error::None;
}

// The number of policy-list entries to scan for AMFI's conf, from the live
// maxindex (highest populated index; entries[0..maxindex] inclusive) and max
// (the allocated entry count). Floored at one chunk, never past the allocated
// array (max), and ceilinged against an implausible count. Pure, so it is
// host-tested; Route.cpp reads max, maxindex and the entries pointer from the
// live struct and does the bounded scan for the element whose mpc is AMFI's conf.
inline size_t policyScanCount(uint32_t maxindex, uint32_t max) {
    uint64_t wanted = static_cast<uint64_t>(maxindex) + 1;
    if (wanted < PolicyScanFloor)
        wanted = PolicyScanFloor;
    if (wanted > PolicyScanCeiling)
        wanted = PolicyScanCeiling;
    if (wanted > max)
        wanted = max;
    return static_cast<size_t>(wanted);
}

// Resolve the runtime address of the symbol-less mac_policy_list global from the
// code of mac_policy_register (resolved by name, exported on every build). The
// framework forms the global's address A with an ADRP and an ADD, and accesses it
// at two fixed fields: the 8-byte entries pointer at A+0x18 (PolicyListEntriesOffset)
// and the 4-byte maxindex at A+0x08 (PolicyListMaxIndexOffset). The access SHAPE
// varies by build -- some load the field off the base register (LDR [A,#0x18]),
// others re-materialise the page with a fresh ADRP in another register and load at
// the full page-offset -- but both reach the same effective addresses A+0x18 and
// A+0x08. So this scans the whole body to RET (no fixed window, no fixed
// register): it collects every ADRP+ADD-formed
// address and every field access's effective address, then returns the formed
// address A read at BOTH A+0x18 (8-byte) and A+0x08 (4-byte) -- the structural
// signature of the list. `code`/`size` are the function's instructions and byte
// length, `funcAddr` its runtime address. Returns 0 if no base meets the signature
// (a changed shape on some future build), in which case Route.cpp refuses by name
// rather than guess; the base is adopted only after scanning its entries finds
// AMFI's own conf (Route.cpp), so a wrong match is refused downstream.
inline uint64_t resolvePolicyListBase(const uint8_t *code, size_t size, uint64_t funcAddr) {
    if (!code || size < sizeof(uint32_t))
        return 0;
    uint64_t regAddr[32] = {};
    bool regValid[32] = {};
    size_t count = size / sizeof(uint32_t);
    constexpr size_t MaxInstructions = 512; // mac_policy_register's body; scan to RET
    if (count > MaxInstructions)
        count = MaxInstructions;
    // ADRP+ADD-formed candidate bases, and the effective addresses of 8-byte and
    // 4-byte field accesses. Bounded, no allocation; overflow simply stops recording
    // (the signature is dense near the base, so the real one is captured first).
    constexpr size_t MaxBases = 64, MaxAccesses = 128;
    uint64_t bases[MaxBases];
    uint64_t load8[MaxAccesses];
    uint64_t load4[MaxAccesses];
    size_t baseCount = 0, load8Count = 0, load4Count = 0;
    auto record = [](uint64_t *array, size_t &n, size_t cap, uint64_t value) {
        if (n < cap)
            array[n++] = value;
    };
    for (size_t k = 0; k < count; ++k) {
        uint32_t insn;
        __builtin_memcpy(&insn, code + k * sizeof(uint32_t), sizeof(insn));
        if (insn == 0xD65F03C0u) // RET: the function body ends
            break;
        uint32_t rd = insn & 0x1F;
        uint32_t rn = (insn >> 5) & 0x1F;
        if ((insn & 0x9F000000u) == 0x90000000u) { // ADRP Xd, #page
            uint64_t immlo = (insn >> 29) & 0x3;
            uint64_t immhi = (insn >> 5) & 0x7FFFF;
            int64_t imm = static_cast<int64_t>((immhi << 2) | immlo);
            if (imm & (int64_t{1} << 20)) // sign-extend the 21-bit page count
                imm |= ~((int64_t{1} << 21) - 1);
            uint64_t pc = funcAddr + k * sizeof(uint32_t);
            regAddr[rd] = (pc & ~uint64_t{0xFFF}) + (static_cast<uint64_t>(imm) << 12);
            regValid[rd] = true;
        } else if ((insn & 0xFF000000u) == 0x91000000u && !((insn >> 23) & 1)) { // ADD Xd,Xn,#imm
            uint64_t imm12 = (insn >> 10) & 0xFFF;
            uint64_t imm = ((insn >> 22) & 1) ? (imm12 << 12) : imm12;
            if (regValid[rn]) {
                regAddr[rd] = regAddr[rn] + imm;
                regValid[rd] = true;
                record(bases, baseCount, MaxBases, regAddr[rd]); // a formed global address
            } else if (rd != rn) {
                regValid[rd] = false;
            }
        } else if ((insn & 0xFFC00000u) == 0xF9400000u) { // LDR Xt, [Xn, #pimm] (8-byte)
            if (regValid[rn])
                record(load8, load8Count, MaxAccesses,
                       regAddr[rn] + static_cast<uint64_t>((insn >> 10) & 0xFFF) * 8);
            regValid[rd] = false;                         // the load overwrites its destination
        } else if ((insn & 0xFFC00000u) == 0xB9400000u) { // LDR Wt, [Xn, #pimm] (4-byte)
            if (regValid[rn])
                record(load4, load4Count, MaxAccesses,
                       regAddr[rn] + static_cast<uint64_t>((insn >> 10) & 0xFFF) * 4);
            regValid[rd] = false;
        } else if ((insn & 0xFFC00000u) == 0xB9000000u) { // STR Wt, [Xn, #pimm] (4-byte)
            if (regValid[rn])
                record(load4, load4Count, MaxAccesses,
                       regAddr[rn] + static_cast<uint64_t>((insn >> 10) & 0xFFF) * 4);
        } else {
            // Any other instruction may overwrite a tracked register; drop the
            // registers it writes so a later ADD or load never forms an address
            // from a stale value. Data-processing and single loads write Rd/Rt in
            // bits [4:0], a load pair also writes Rt2 in bits [14:10], and BL
            // writes the link register. Over-dropping only misses a match (resolve
            // returns 0), never reads a wild address.
            regValid[rd] = false;
            if ((insn & 0x3C000000u) == 0x28000000u && (insn & 0x00400000u)) // LDP: Rt2
                regValid[(insn >> 10) & 0x1F] = false;
            if ((insn & 0xFC000000u) == 0x94000000u) // BL: the link register
                regValid[30] = false;
        }
    }
    for (size_t b = 0; b < baseCount; ++b) {
        bool entries = false, maxindex = false;
        for (size_t e = 0; e < load8Count && !entries; ++e)
            entries = load8[e] == bases[b] + PolicyListEntriesOffset;
        for (size_t m = 0; m < load4Count && !maxindex; ++m)
            maxindex = load4[m] == bases[b] + PolicyListMaxIndexOffset;
        if (entries && maxindex)
            return bases[b];
    }
    return 0;
}

} // namespace arm64
} // namespace graft
