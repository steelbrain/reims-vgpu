// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// The x86-64 branch islands (src/Island.h). The pool lies in Graft's own
// __TEXT, filled with int3. OpenCore injects Graft into the Boot collection
// beside the consumers, so a consumer's branch reaches every island, and each
// island's absolute jump reaches any provider. IslandPool.h keeps the
// bookkeeping and builds each island; this file owns the storage and writes it
// through the kernel-write window, as the route trampolines are written.
#include "../Island.h"
#include "../KernelWrite.h"
#include "../Policy.h"
#include "IslandPool.h"

#if !defined(__x86_64__)
#error Graft's x86-64 branch islands require the x86-64 architecture implementation
#endif

constexpr size_t IslandCapacity = graft::policy::IslandCapacity;
static_assert(IslandCapacity * graft::x86::IslandSize == 4096, "branch island storage size");
extern "C" const uint8_t graft_island_storage[IslandCapacity][graft::x86::IslandSize];
__asm__(".section __TEXT,__graft_islands,regular,pure_instructions\n"
        ".p2align 4\n"
        "_graft_island_storage:\n"
        ".fill 4096,1,0xcc\n"
        ".text\n");

namespace graft {
namespace {
x86::IslandPool<IslandCapacity> pool;
// An unwritten island's bytes, and a commit's scratch: each island's code and
// its write. Static rather than on the kernel stack, where a full pool's writes
// alone would take 8 KiB.
const uint8_t trapFill[x86::IslandSize] = {0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc,
                                           0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc, 0xcc};
uint8_t code[IslandCapacity][x86::IslandSize];
Patch writes[IslandCapacity];

uint64_t storage() {
    return reinterpret_cast<uint64_t>(graft_island_storage);
}
} // namespace

uint64_t reserveIsland(uint64_t destination) {
    return pool.reserve(storage(), destination);
}

uint64_t islandDestination(uint64_t address) {
    return pool.destination(storage(), address);
}

WriteError commitIslands() {
    size_t count = pool.staged - pool.committed;
    if (!count)
        return WriteError::None;
    for (size_t i = 0; i < count; ++i) {
        size_t slot = pool.committed + i;
        x86::islandCode(code[i], pool.destinations[slot]);
        writes[i] = {const_cast<uint8_t *>(graft_island_storage[slot]), trapFill, code[i],
                     x86::IslandSize};
    }
    // An island is written only over the int3 it was filled with, so a later
    // commit that reuses a slot an unverified rollback left changed is refused.
    auto error = applyWrites(writes, count);
    if (error == WriteError::None)
        pool.commit();
    else
        pool.discard();
    return error;
}

void discardIslands() {
    pool.discard();
}

} // namespace graft
