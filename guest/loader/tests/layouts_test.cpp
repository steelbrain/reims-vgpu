// Ours: Rows.h's compareLayout, which the loader checks each library class
// with (Loader.cpp, layoutsMatch), tested on this Mac against the tables a
// loader build compiles in: the Makefile builds it with -I to its generated
// tables.h and runs it before it compiles the loader. A guest laid out as
// each row says passes, also with a slot holding another copy of its
// method; the table with one size changed, and with one slot changed, is
// refused, naming the class and the field; so are a longer vtable, a method
// not found, an unreadable vtable and a class without an OSMetaClass.
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include "Rows.h"
template <typename T, size_t N>
constexpr size_t arrsize(const T (&)[N]) { return N; }
using namespace Bridge;
#include "tables.h"

namespace {
// The address a guest gives a method: made from its name (FNV-1a), in the
// kernel's range, never 0.
uint64_t address(const char *symbol) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (; *symbol; ++symbol)
        hash = (hash ^ static_cast<uint8_t>(*symbol)) * 0x100000001b3ULL;
    return 0xffffff8000000000ULL | (hash & 0x7ffffffff8ULL) | 8;
}

// A guest class: its size, and its vtable's words after the address point.
struct Guest {
    uint64_t size;
    std::vector<uint64_t> words;
    const char *missing = nullptr;  // a method the guest does not have
    const char *copied = nullptr;   // a method with a second copy, at `copy`
    uint64_t copy = 0;
    bool readable = true;
};

// The guest a row describes: its size, each slot's method, then a zero word.
Guest asBuilt(const Layout &layout, const Slot *slots) {
    Guest guest {layout.size, {}};
    for (uint32_t i = 0; i < layout.slotCount; ++i)
        guest.words.push_back(address(slots[layout.firstSlot + i].symbol));
    guest.words.push_back(0);
    return guest;
}

Mismatch compare(const Layout &layout, const Slot *slots, const Guest &guest, uint32_t &at, uint64_t &got,
                 uint64_t &want) {
    auto size = [&guest] { return guest.size; };
    auto read = [&guest](uint32_t i, uint64_t &value) {
        if (!guest.readable || i >= guest.words.size())
            return false;
        value = guest.words[i];
        return true;
    };
    auto resolve = [&guest](const Slot &slot, uint64_t value) -> uint64_t {
        if (guest.missing && !strcmp(guest.missing, slot.symbol))
            return 0;
        if (guest.copied && !strcmp(guest.copied, slot.symbol) && value == guest.copy)
            return value;
        return address(slot.symbol);
    };
    return compareLayout(layout, slots, size, read, resolve, at, got, want);
}

// The slot of the layout whose method differs from the next one's.
uint32_t slotToChange(const Layout &layout, const Slot *slots) {
    for (uint32_t i = layout.slotCount / 2; i + 1 < layout.slotCount; ++i)
        if (strcmp(slots[layout.firstSlot + i].symbol, slots[layout.firstSlot + i + 1].symbol))
            return i;
    assert(!"no two adjacent slots differ");
    return 0;
}
} // namespace

int main() {
    uint32_t at;
    uint64_t got, want;
    size_t classes = 0, slots = 0;
    for (auto &consumer : consumers) {
        for (size_t i = 0; i < consumer.layoutCount; ++i) {
            auto &layout = consumer.layouts[i];
            auto guest = asBuilt(layout, consumer.slots);
            assert(compare(layout, consumer.slots, guest, at, got, want) == Mismatch::None);
            ++classes;
            slots += layout.slotCount;
        }
    }
    printf("layouts: %zu classes, %zu vtable slots pass as built\n", classes, slots);

    auto &consumer = consumers[0];
    auto &original = consumer.layouts[0];
    auto guest = asBuilt(original, consumer.slots);

    // The table with one size changed.
    auto layout = original;
    layout.size += 8;
    assert(compare(layout, consumer.slots, guest, at, got, want) == Mismatch::Size);
    assert(got == original.size && want == layout.size);
    printf("size changed: %s: REFUSED: %s's size is %llu, built against %llu\n", consumer.name, layout.name,
           static_cast<unsigned long long>(got), static_cast<unsigned long long>(want));

    // The table with one slot changed: it names the next slot's method.
    std::vector<Slot> changed(consumer.slots, consumer.slots + consumer.slotTotal);
    uint32_t slot = slotToChange(original, consumer.slots);
    changed[original.firstSlot + slot] = changed[original.firstSlot + slot + 1];
    assert(compare(original, changed.data(), guest, at, got, want) == Mismatch::Slot);
    assert(at == slot && got == address(consumer.slots[original.firstSlot + slot].symbol) &&
           want == address(changed[original.firstSlot + slot].symbol));
    printf("slot changed: %s: REFUSED: %s's vtable slot %u holds %#llx, built against %s at %#llx\n",
           consumer.name, original.name, at, static_cast<unsigned long long>(got),
           changed[original.firstSlot + slot].symbol, static_cast<unsigned long long>(want));

    // A guest whose slot holds another copy of its method, as a library's
    // local methods can have (Loader.cpp, namedAt).
    auto copied = guest;
    copied.copied = consumer.slots[original.firstSlot + slot].symbol;
    copied.copy = copied.words[slot] ^ 0x1000;
    copied.words[slot] = copied.copy;
    assert(compare(original, consumer.slots, copied, at, got, want) == Mismatch::None);
    copied.copied = nullptr;
    assert(compare(original, consumer.slots, copied, at, got, want) == Mismatch::Slot && at == slot);
    puts("layouts: passed a slot holding another copy of its method, refused it as another method");

    // A guest whose vtable is longer, which lacks a method, whose vtable
    // cannot be read, or which has no such class.
    auto longer = guest;
    longer.words.back() = address("__ZN11OSMetaClass19addedVirtualMethodEv");
    longer.words.push_back(0);
    assert(compare(original, consumer.slots, longer, at, got, want) == Mismatch::Length &&
           at == original.slotCount);
    auto missing = guest;
    missing.missing = consumer.slots[original.firstSlot + slot].symbol;
    assert(compare(original, consumer.slots, missing, at, got, want) == Mismatch::Symbol);
    auto unreadable = guest;
    unreadable.readable = false;
    assert(compare(original, consumer.slots, unreadable, at, got, want) == Mismatch::Vtable && at == 0);
    auto shorter = guest;
    shorter.words.pop_back();
    assert(compare(original, consumer.slots, shorter, at, got, want) == Mismatch::Vtable &&
           at == original.slotCount);
    auto absent = guest;
    absent.size = 0;
    assert(compare(original, consumer.slots, absent, at, got, want) == Mismatch::Class);
    puts("layouts: refused a longer vtable, a method not found, an unreadable vtable, no OSMetaClass");
}
