// Ours: the rows of the loader's generated tables (tools/gen-tables.py) and
// the comparison of a library class's layout with its row, shared by the
// loader (Loader.cpp) and its host test (tests/layouts_test.cpp), so it has
// no kernel headers.
#pragma once
#include <stddef.h>
#include <stdint.h>

namespace Bridge {
struct Library {  // a System-KC library, which one consumer links through a shim
    const char *id;
    const char *path;     // its binary in /System/Library/Extensions
    const char *shim;     // the shim's bundle ID
};
struct Site {  // a reference in a consumer to one of its libraries' symbols
    uint32_t offset;       // from the consumer's base
    uint8_t width;         // 8: a pointer; 4: the rel32 field of a call or jmp
    uint8_t library;       // index in libraries
    uint8_t opcode;        // the branch's, the byte before a rel32 field; else 0
    const char *kind;      // tools/bridge.py's: call, got, vtable, superclass, const
    const char *symbol;
    uint32_t placeholder;  // the symbol's placeholder, from the shim's base
    int64_t addend;
};
// A library class one of the consumer's classes derives from, as the table's
// KDK lays it out (tools/bridge.py, layouts).
struct Layout {
    const char *name;
    const char *vtable;          // its vtable's symbol, __ZTV<length><name>
    uint8_t library;             // index in libraries
    uint32_t size;               // its OSMetaClass constructor's size argument
    uint32_t firstSlot, slotCount;  // its vtable's slots in the consumer's slots
};
struct Slot {  // the method a KDK vtable holds in one slot
    const char *symbol;
    int32_t kernel;  // its index in kernelSymbols if the kernel defines it, else -1
};
struct Consumer {  // one of our dormant kexts
    const char *name, *id;
    const char *personality;  // its IOKitPersonalities, as XML
    size_t firstLibrary, libraryCount;
    const Site *sites;
    size_t siteCount;
    const Layout *layouts;
    size_t layoutCount;
    const Slot *slots;
    size_t slotTotal;
};

// What differs first between a class in the guest and its row.
enum class Mismatch { None, Class, Size, Symbol, Vtable, Slot, Length };

inline const char *describe(Mismatch mismatch) {
    switch (mismatch) {
    case Mismatch::None: return "none";
    case Mismatch::Class: return "no OSMetaClass";
    case Mismatch::Size: return "size";
    case Mismatch::Symbol: return "method not found";
    case Mismatch::Vtable: return "vtable unreadable";
    case Mismatch::Slot: return "vtable slot";
    case Mismatch::Length: return "vtable length";
    }
    return "unknown";
}

// Compares a class in the guest with its row: first its size, `size()`, then
// each slot of its vtable, `read(i, value)` (false if the word is not in the
// library's image), with the address of the method the row names there,
// `resolve(slot, value)`: the value itself if it is that method's (a
// library's local method can have several copies, one per unit), else the
// method's first address, or 0 if there is none. Then the word after the
// last slot, which is 0 in the KDK. On a mismatch, `at` is the slot, and
// `got` and `want` the guest's and the row's values.
template <typename Size, typename Read, typename Resolve>
Mismatch compareLayout(const Layout &layout, const Slot *slots, Size size, Read read, Resolve resolve,
                       uint32_t &at, uint64_t &got, uint64_t &want) {
    at = 0;
    want = layout.size;
    got = size();
    if (!got)
        return Mismatch::Class;
    if (got != want)
        return Mismatch::Size;
    for (at = 0; at < layout.slotCount; ++at) {
        got = want = 0;
        if (!read(at, got))
            return Mismatch::Vtable;
        want = resolve(slots[layout.firstSlot + at], got);
        if (!want)
            return Mismatch::Symbol;
        if (got != want)
            return Mismatch::Slot;
    }
    want = 0;
    if (!read(at, got))
        return Mismatch::Vtable;
    return got ? Mismatch::Length : Mismatch::None;
}
} // namespace Bridge
