// The arm64 facts RgfxGate's admission routes carry, as named checked constants
// so a native test pins their values (tests/amfi_test.cpp).
//
// On arm64 Graft routes an owned client's admission hooks by shadowing AMFI's
// mac_policy_ops table: it places the client's replacement in the operation's
// slot and hands back AMFI's own op pointer as the route's `original`. Two
// facts make that callable from RgfxGate:
//   - the operation NAME graft_route_request.symbol carries (no _mac_/mpo_
//     prefix), which Graft maps to the mpo field; and
//   - the pointer-authentication the slot is signed with, so the saved original
//     is declared as a matching __ptrauth function pointer and the call through
//     it authenticates as AMFI signed it: key IA, a per-operation CONSTANT
//     discriminator, NO address diversity.
// Both are uniform across the five config-2 Boot KCs (13.6, 14.6.1, 15.6.1,
// 26.6.2, 27.0.1); Ambrosia's AMFI survey measured the discriminators and Graft
// carries the same values in its own route probe (2026-11-08).
#pragma once
#include <stdint.h>

// The operations RgfxGate routes, by name (Graft's Arm64Route operation table).
#define RGFX_OP_FILE_CHECK_MMAP "file_check_mmap"
#define RGFX_OP_LIBRARY_VALIDATION "file_check_library_validation"
#define RGFX_OP_VNODE_CHECK_SIGNATURE "vnode_check_signature"

// The ptrauth discriminator (key IA, addrDiv 0) each operation's mpo pointer is
// signed with. The saved `original` for each route is declared
// __ptrauth(ptrauth_key_asia, 0, <this>) and called through.
static const uint16_t kRgfxAmfiDiscFileCheckMmap = 0x438d;
static const uint16_t kRgfxAmfiDiscLibraryValidation = 0x9e37;
static const uint16_t kRgfxAmfiDiscVnodeCheckSignature = 0xcbd2;
