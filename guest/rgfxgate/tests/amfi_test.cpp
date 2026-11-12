// Native test that pins the arm64 AMFI route facts (src/rgfx_amfi.h): the three
// admission operation names and the pointer-authentication discriminators AMFI
// signs their slots with. These are measured ABI constants, uniform
// across the five config-2 Boot KCs (Ambrosia 2026-11-08) and matched by Graft's
// own route layer; a stray edit to either side must fail the build, so this test
// holds the literal values and the cross-checks that tie them together. Build
// and run with `make test`.
#include "rgfx_amfi.h"
#include <cassert>
#include <cstdio>
#include <cstring>

int main() {
	// The discriminators AMFI signs each mac_policy_ops slot with (key IA, no
	// address diversity). These are the same values Graft's Arm64Route operation
	// table carries, and each replacement's type must have its slot's
	// (RgfxGateHooks.cpp's static_asserts).
	assert(kRgfxAmfiDiscFileCheckMmap == 0x438d);
	assert(kRgfxAmfiDiscLibraryValidation == 0x9e37);
	assert(kRgfxAmfiDiscVnodeCheckSignature == 0xcbd2);

	// No two operations share a discriminator (each slot is distinct).
	assert(kRgfxAmfiDiscFileCheckMmap != kRgfxAmfiDiscLibraryValidation);
	assert(kRgfxAmfiDiscFileCheckMmap != kRgfxAmfiDiscVnodeCheckSignature);
	assert(kRgfxAmfiDiscLibraryValidation != kRgfxAmfiDiscVnodeCheckSignature);

	// The operation names graft_route_request.symbol carries on arm64: the bare
	// operation, no "_mac_" or "mpo_" prefix (Graft maps each to its mpo field).
	assert(strcmp(RGFX_OP_FILE_CHECK_MMAP, "file_check_mmap") == 0);
	assert(strcmp(RGFX_OP_LIBRARY_VALIDATION, "file_check_library_validation") == 0);
	assert(strcmp(RGFX_OP_VNODE_CHECK_SIGNATURE, "vnode_check_signature") == 0);

	// The Sandbox policy's file_check_mmap: its registered name ("Sandbox", the
	// mpc_name Graft finds it by), a colon, and the bare operation AMFI's route
	// names, so the two routes name one operation on two policies.
	assert(strcmp(RGFX_OP_SANDBOX_FILE_CHECK_MMAP, "Sandbox:file_check_mmap") == 0);
	assert(strcmp(&RGFX_OP_SANDBOX_FILE_CHECK_MMAP[strlen("Sandbox:")], RGFX_OP_FILE_CHECK_MMAP) ==
	       0);

	printf("amfi_test: ok\n");
	return 0;
}
