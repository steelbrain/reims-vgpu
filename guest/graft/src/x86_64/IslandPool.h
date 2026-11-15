// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The x86-64 branch islands' bookkeeping and code (src/Island.h). An island is
// one absolute jump, so a consumer's branch whose 32-bit displacement cannot
// reach its provider reaches the island instead, and the island reaches the
// provider. This module only computes; x86_64/Island.cpp owns the storage and
// writes it.
#pragma once
#include "X86Route.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

namespace graft {
namespace x86 {

// One island: `jmp [rip+0]` and the eight-byte destination it reads, padded with
// int3 to keep each island 16-byte aligned.
constexpr size_t IslandSize = 16;
static_assert(JumpSize <= IslandSize, "an island holds one absolute jump");

inline void islandCode(uint8_t (&code)[IslandSize], uint64_t destination) {
    memset(code, 0xcc, sizeof(code));
    detail::absoluteJump(code, destination);
}

// The islands of a pool of Capacity islands, the first at base. Islands
// [0, committed) are written and stay for the boot; [committed, staged) are
// reserved by the plan in progress. An island serves one destination, shared by
// every branch that needs it, so a destination already written or reserved is
// never given a second one.
template <size_t Capacity>
struct IslandPool {
    uint64_t destinations[Capacity];
    size_t committed, staged;

    // The address of the island that jumps to destination: one written or
    // reserved already, else a new reservation. 0 when every island is taken or
    // the pool's addresses overflow.
    uint64_t reserve(uint64_t base, uint64_t destination) {
        size_t slot = 0;
        while (slot < staged && destinations[slot] != destination)
            ++slot;
        if (slot == Capacity || !destination || base > UINT64_MAX - Capacity * IslandSize)
            return 0;
        if (slot == staged)
            destinations[staged++] = destination;
        return base + slot * IslandSize;
    }

    // Where the island at address jumps, written or reserved; 0 where address is
    // not the start of one.
    uint64_t destination(uint64_t base, uint64_t address) const {
        if (address < base || (address - base) % IslandSize)
            return 0;
        uint64_t slot = (address - base) / IslandSize;
        return slot < staged ? destinations[slot] : 0;
    }

    void commit() { committed = staged; }
    void discard() { staged = committed; }
};

} // namespace x86
} // namespace graft
