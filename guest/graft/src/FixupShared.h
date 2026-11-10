// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Arithmetic shared by the per-architecture binding fixups (x86_64/Fixup.h and
// arm64e/Fixup.h): the fixup error set and a checked address-plus-addend. No
// kernel calls and no writes.
#pragma once
#include <stdint.h>

namespace graft {
namespace fixup {

enum class Error { None, Width, Opcode, Address, Placeholder, Range };

inline const char *describe(Error error) {
    switch (error) {
    case Error::None: return "ok";
    case Error::Width: return "site width";
    case Error::Opcode: return "branch opcode";
    case Error::Address: return "address overflow";
    case Error::Placeholder: return "placeholder mismatch";
    case Error::Range: return "branch out of range";
    }
    return "unknown fixup error";
}

inline bool add(uint64_t address, int64_t addend, uint64_t &out) {
    // Unsigned subtraction also handles INT64_MIN without signed overflow.
    if (addend < 0) {
        uint64_t magnitude = uint64_t{0} - static_cast<uint64_t>(addend);
        if (address < magnitude)
            return false;
        out = address - magnitude;
    } else {
        if (address > UINT64_MAX - static_cast<uint64_t>(addend))
            return false;
        out = address + static_cast<uint64_t>(addend);
    }
    return true;
}

} // namespace fixup
} // namespace graft
