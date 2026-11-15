// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Branch islands: how a consumer's branch reaches a provider beyond its 32-bit
// displacement. On x86-64 an import site may be a call or jump whose
// displacement reaches 2 GiB either way, and a provider's collection can be
// loaded farther than that from the Boot collection the consumer is placed in.
// Graft's own image is placed in that collection too, so x86_64/Island.cpp keeps
// a pool of islands there, each one absolute jump to a provider's function, and
// the binding writes such a branch to an island. arm64e has no branch sites, and
// its arm64e/Island.cpp gives no island.
//
// Only the bind pass calls these, one consumer at a time (bindConsumers), so
// the pool needs no lock of its own.
#pragma once
#include "WritePlan.h"
#include <stdint.h>

namespace graft {
// The address of an island that jumps to destination: one already written, or
// one reserved for the bind in progress. 0 when none is free.
uint64_t reserveIsland(uint64_t destination);
// Where the island at address jumps, written or reserved; 0 where address is no
// island.
uint64_t islandDestination(uint64_t address);
// Write the islands reserved since the last commit, in one transaction, before
// any branch is written to them. Once written, an island stays for the boot. A
// refused write drops the reservations.
WriteError commitIslands();
// Drop the islands reserved since the last commit, for a bind that is refused or
// waits before writing.
void discardIslands();
} // namespace graft
