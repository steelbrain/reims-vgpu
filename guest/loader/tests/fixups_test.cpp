// Ours: Fixup.h, the arithmetic the loader binds each site with, tested at
// canonical high kernel addresses (make test).
#include "Fixup.h"
#include <assert.h>
#include <stdio.h>
#include <initializer_list>
using namespace Bridge;
int main() {
    uint64_t value = 0;
    constexpr uint64_t site = 0xffffff8001001000ULL;
    constexpr uint64_t stub = site - 0x2000;
    constexpr uint64_t real = site + 0x100000;
    assert(replacement(8, 0, 0, site, stub + 16, stub, real, 16, value) == OK && value == real + 16);
    assert(replacement(8, 0, 0, site, stub - 8, stub, real, -8, value) == OK && value == real - 8);
    assert(replacement(8, 0, 0, site, stub + 1, stub, real, 0, value) == Placeholder);
    uint32_t old = static_cast<uint32_t>(stub - (site + 4));
    for (uint8_t op : {kCall, kJump}) {
        assert(replacement(4, op, op, site, old, stub, real, 0, value) == OK);
        assert(site + 4 + static_cast<int32_t>(value) == real);
        assert(replacement(4, op, 0x90, site, old, stub, real, 0, value) == Opcode);
        assert(replacement(4, op, op, site + 1, old, stub, real, 0, value) == Placeholder);
        assert(replacement(4, op, op, site, old, stub + 16, real, 0, value) == Placeholder);
        assert(replacement(4, op, op, site, old, stub, site + 4 + INT32_MAX, 0, value) == OK);
        assert(replacement(4, op, op, site, old, stub, site + 5 + INT32_MAX, 0, value) == Range);
        assert(replacement(4, op, op, site, old, stub, site + 4 - (1ULL << 31), 0, value) == OK);
        assert(replacement(4, op, op, site, old, stub, site + 3 - (1ULL << 31), 0, value) == Range);
    }
    assert(replacement(2, 0, 0, site, old, stub, real, 0, value) == Width);
    assert(replacement(4, 0x90, 0x90, site, old, stub, real, 0, value) == Opcode);
    puts("fixup arithmetic: pass (pointers, addends, opcodes, stale sites, rel32 bounds)");
}
