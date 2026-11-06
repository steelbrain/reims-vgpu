// Ours: the value one bridge site gets, shared by the loader (Loader.cpp) and
// its host test (tests/fixups_test.cpp), so it has no kernel headers.
#pragma once
#include <stdint.h>
#include <limits.h>

namespace Bridge {
// The only branches a rel32 site can be the field of (bridge.py checks it).
constexpr uint8_t kCall = 0xe8, kJump = 0xe9;

enum Result { OK, Placeholder, Opcode, Range, Width };

inline const char *describe(Result result) {
    switch (result) {
    case OK: return "ok";
    case Placeholder: return "placeholder";
    case Opcode: return "opcode";
    case Range: return "rel32 range";
    case Width: return "width";
    }
    return "unknown";
}

// The value for the site at `site`, which holds `old`, to refer to `real`
// instead of the shim's `placeholder`; both plus the addend the consumer's
// file holds there. Checked first: that the site holds what OpenCore linked
// there, and for a rel32 field that its branch is the expected call or jmp,
// and that the new target is in range.
inline Result replacement(uint8_t width, uint8_t opcode, uint8_t actualOpcode,
                          uint64_t site, uint64_t old, uint64_t placeholder,
                          uint64_t real, int64_t addend, uint64_t &value) {
    uint64_t expected = placeholder + static_cast<uint64_t>(addend);
    uint64_t target = real + static_cast<uint64_t>(addend);
    if (width == 8) {
        if (old != expected)
            return Placeholder;
        value = target;
        return OK;
    }
    if (width != 4)
        return Width;
    if ((opcode != kCall && opcode != kJump) || actualOpcode != opcode)
        return Opcode;
    uint64_t next = site + 4;
    if (next + static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(old))) != expected)
        return Placeholder;
    __int128 displacement = static_cast<__int128>(target) - next;
    if (displacement < INT32_MIN || displacement > INT32_MAX)
        return Range;
    value = static_cast<uint32_t>(displacement);
    return OK;
}
} // namespace Bridge
