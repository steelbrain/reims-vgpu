// The arm64 facts RgfxGate's admission routes carry, as named checked constants
// so a native test pins their values (tests/amfi_test.cpp).
//
// On arm64 Graft routes an owned client's admission hooks by shadowing AMFI's
// mac_policy_ops table: it places the client's replacement in the operation's
// slot and hands back AMFI's own op pointer as the route's `original`, in a
// void * (Graft.h). Two facts make the routes RgfxGate's:
//   - the operation NAME graft_route_request.symbol carries (no _mac_/mpo_
//     prefix), which Graft maps to the mpo field; and
//   - the pointer-authentication the slot is signed with: key IA, a
//     per-operation CONSTANT discriminator, NO address diversity. It is the mpo
//     type's own (function-type discrimination), so RgfxGateHooks.cpp checks
//     each replacement's type against it at compile time.
// Both are uniform across the five config-2 Boot KCs (13.6, 14.6.1, 15.6.1,
// 26.6.2, 27.0.1); Ambrosia's AMFI survey measured the discriminators and Graft
// carries the same values in its own route probe (2026-11-08). 27.2.1's AMFI
// signs the three operations' slots with the same discriminators, key IA and no
// address diversity (read from its code, 2026-11-11).
#pragma once
#include <stdint.h>

// The operations RgfxGate routes, by name (Graft's Arm64Route operation table).
#define RGFX_OP_FILE_CHECK_MMAP "file_check_mmap"
#define RGFX_OP_LIBRARY_VALIDATION "file_check_library_validation"
#define RGFX_OP_VNODE_CHECK_SIGNATURE "vnode_check_signature"

// The Sandbox policy's file_check_mmap, which RgfxGate routes beside AMFI's: a
// policy's registered name and a colon before the operation names that policy's
// (Graft.h); a bare name is AMFI's. A sandboxed process whose profile allows
// reading /Library/GPUBundles but not mapping it executable (WebKit's GPU
// process, replayd) is refused our bundle by Sandbox's op alone, which AMFI's
// route does not reach: on arm64e each policy's op is called on its own. Its slot
// holds the same mpo type as AMFI's, so the same discriminator.
#define RGFX_OP_SANDBOX_FILE_CHECK_MMAP "Sandbox:file_check_mmap"

// The ptrauth discriminator (key IA, addrDiv 0) each operation's mpo pointer is
// signed with, which each replacement's type must have.
static const uint16_t kRgfxAmfiDiscFileCheckMmap = 0x438d;
static const uint16_t kRgfxAmfiDiscLibraryValidation = 0x9e37;
static const uint16_t kRgfxAmfiDiscVnodeCheckSignature = 0xcbd2;
