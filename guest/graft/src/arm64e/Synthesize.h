// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// arm64e vtable synthesis for a header-less, per-release-variable parent. The
// binding (src/Vtable.h) fills a consumer's inherited slots from a parent whose
// layout the consumer declared (manifest parentSlots, overrides by slot index)
// -- which needs a parent with an SDK header, so the consumer's compiler knows
// the layout. IOMobileFramebuffer and its Legacy subclass have none and their
// vtables grow per release, so a consumer cannot declare the layout at build
// time. Synthesis measures the live parent at bind time and places each override
// on the parent slot that holds the SAME method, so a release that renumbers
// slots still lands each override right.
//
// It finds that slot forward: from an override's own mangled name it composes
// the parent's -- or a concrete
// sibling's -- symbol for the same method, resolves that one symbol in the live
// KC, and scans the live parent vtable for the slot holding the resolved address.
// Only the overrides need a symbol; there is no reverse address->symbol table
// over every slot of the parent.
//
// This header is the pure part, host-tested against synthetic IOKit-shaped
// symbols (tests/arm64e_synthesize_test.cpp):
//   - methodIdentity extracts the part of a mangled name a method and its
//     overrides share (everything but the class component);
//   - composeMemberSymbol rebuilds that method's symbol under another class;
//   - scanVtable finds the unique slot of a vtable holding an address;
//   - fallbackTargetFits accepts a slot the discriminator fallback found only
//     when no symbol at its target names another method;
//   - planSynthesis turns the kernel's per-override resolved slots into the fill
//     plan, refusing an unmatched required (pure) slot, an ambiguous live vtable,
//     or two overrides on one slot;
//   - instanceFits is the bind-time reserve/size check.
//
// The kernel layer (the binding, behind the per-architecture boundary) measures
// the live parent (IOMobileFramebufferLegacy) and the concrete sibling
// (AppleParavirtDisplay). For each override it composes + resolves + scans each
// selector in turn -- the parent's symbol first, the sibling's for a pure slot the
// parent leaves as ___cxa_pure_virtual -- into a resolved slot (or NoSlot /
// AmbiguousSlot), calls planSynthesis, then fills the consumer vtable with the
// existing per-slot re-sign (src/arm64e/Fixup.h vtableSlot): slot n takes an
// override's implementation where planSynthesis mapped one there, else the live
// parent's slot n re-signed for the new slot's own address.
#pragma once
#include <stddef.h>
#include <stdint.h>

namespace graft {
namespace synth {

// A member function's identity: the part of an Itanium-mangled name a method and
// any subclass's override of it share -- everything except the class component.
// That is the cv/ref-qualifiers (which precede the class in a nested-name and so
// are part of the signature, distinguishing `foo()` from `foo() const`) and the
// `tail`: the member-name component, any further nested components and the
// parameter mangling (which follow the class). composeMemberSymbol rebuilds a
// full symbol by placing another class component between the two.
struct Identity {
    const char *tail = nullptr; // after the class component; null if not a method
    uint8_t quals = 0;          // bit0 const (K), bit1 volatile (V), bit2 restrict (r)
    uint8_t refqual = 0;        // 'R' (&) or 'O' (&&), else 0
    bool valid() const { return tail != nullptr; }
};

// The identity of `mangled`, or an invalid Identity for a symbol that is not a
// nested member function in the handled form (a non-method slot, an RTTI or
// metaclass pointer, a pure-virtual stub). A single asm-linkage underscore
// before the _Z ABI prefix is accepted, so both the symbol-table form (__Z...)
// and a bare _Z... are read. Only plain length-prefixed class components are
// handled; a substitution or operator-name form yields invalid, so it never
// matches an override (synthesis refuses an unmatched pure slot rather than
// guessing).
inline Identity methodIdentity(const char *mangled) {
    Identity id;
    if (!mangled)
        return id;
    const char *p = mangled;
    if (p[0] == '_' && p[1] == '_' && p[2] == 'Z')
        p += 3;
    else if (p[0] == '_' && p[1] == 'Z')
        p += 2;
    else
        return id;
    if (*p != 'N') // only nested names (member functions) carry a class
        return id;
    ++p;
    for (;; ++p) { // cv-qualifiers, in any order, part of the signature
        if (*p == 'K')
            id.quals |= 1;
        else if (*p == 'V')
            id.quals |= 2;
        else if (*p == 'r')
            id.quals |= 4;
        else
            break;
    }
    if (*p == 'R' || *p == 'O') { // a ref-qualifier, at most one
        id.refqual = static_cast<uint8_t>(*p);
        ++p;
    }
    if (*p < '1' || *p > '9') // a substitution or operator form, not a plain class
        return id;
    size_t length = 0;
    while (*p >= '0' && *p <= '9') {
        size_t digit = static_cast<size_t>(*p - '0');
        if (length > (SIZE_MAX - digit) / 10)
            return id;
        length = length * 10 + digit;
        ++p;
    }
    for (size_t i = 0; i < length; ++i)
        if (!p[i]) // the class name runs past the string: malformed
            return id;
    const char *tail = p + length;
    if (*tail) // something must follow the class component
        id.tail = tail;
    return id;
}

// Rebuild `id`'s method symbol under `className`: the Itanium nested-name
// `__ZN <cv> <ref> <len><className> <tail>` the live KC holds for that class's own
// definition of the method. Writes a NUL-terminated symbol-table-form name (the
// `__Z` the KC uses, matching methodIdentity's input) to out[0, outSize) and
// returns its length, or 0 if `id` is invalid or the result would not fit
// (bounded; never truncates into a valid-looking symbol). The cv-qualifiers are
// emitted in Itanium order (r, V, K), as the compiler mangles them, so the result
// equals the KC symbol byte for byte and resolves with no normalisation.
inline size_t composeMemberSymbol(const Identity &id, const char *className, char *out,
                                  size_t outSize) {
    if (!id.valid() || !className || outSize == 0)
        return 0;
    size_t pos = 0;
    const char prefix[] = "__ZN";
    for (size_t i = 0; i < sizeof(prefix) - 1; ++i) {
        if (pos + 1 >= outSize)
            return 0;
        out[pos++] = prefix[i];
    }
    const char order[] = {'r', 'V', 'K'}; // restrict, volatile, const
    const uint8_t bits[] = {4, 2, 1};
    for (size_t i = 0; i < sizeof(order); ++i)
        if (id.quals & bits[i]) {
            if (pos + 1 >= outSize)
                return 0;
            out[pos++] = order[i];
        }
    if (id.refqual) {
        if (pos + 1 >= outSize)
            return 0;
        out[pos++] = static_cast<char>(id.refqual);
    }
    size_t classLen = __builtin_strlen(className);
    char digits[20];
    size_t nd = 0;
    size_t value = classLen;
    do {
        digits[nd++] = static_cast<char>('0' + value % 10);
        value /= 10;
    } while (value && nd < sizeof(digits));
    while (nd) {
        if (pos + 1 >= outSize)
            return 0;
        out[pos++] = digits[--nd];
    }
    for (size_t i = 0; i < classLen; ++i) {
        if (pos + 1 >= outSize)
            return 0;
        out[pos++] = className[i];
    }
    for (const char *t = id.tail; *t; ++t) {
        if (pos + 1 >= outSize)
            return 0;
        out[pos++] = *t;
    }
    out[pos] = '\0';
    return pos;
}

// The member name of a method identity: the source name that follows the class
// component (`6enable` in `6enableEv`, as `enable`), or the two-character code
// of a constructor, destructor or operator (`D0`, `aS`). False when `id` is not a
// method or its tail is malformed.
inline bool memberName(const Identity &id, const char *&name, size_t &length) {
    if (!id.valid())
        return false;
    const char *p = id.tail;
    if (*p < '1' || *p > '9') {
        if (!p[0] || !p[1])
            return false;
        name = p;
        length = 2;
        return true;
    }
    size_t count = 0;
    while (*p >= '0' && *p <= '9') {
        size_t digit = static_cast<size_t>(*p - '0');
        if (count > (SIZE_MAX - digit) / 10)
            return false;
        count = count * 10 + digit;
        ++p;
    }
    for (size_t i = 0; i < count; ++i)
        if (!p[i])
            return false;
    name = p;
    length = count;
    return true;
}

// Whether mangled symbols `a` and `b` are member functions of one name, whatever
// their classes, qualifiers and parameters.
inline bool sameMethodName(const char *a, const char *b) {
    const char *nameA = nullptr, *nameB = nullptr;
    size_t lengthA = 0, lengthB = 0;
    return memberName(methodIdentity(a), nameA, lengthA) &&
           memberName(methodIdentity(b), nameB, lengthB) && lengthA == lengthB &&
           !__builtin_memcmp(nameA, nameB, lengthA);
}

// Whether the discriminator fallback (VtableDisc.h uniqueDiscriminatorSlot) may
// place an override on a live slot. A 16-bit discriminator can equal that of an
// unrelated slot when the parent lacks the override's method, so the slot's
// target is checked by name: eachName(visit) calls visit(name) for each symbol
// name the caller holds at the target, and selector(s), for s below
// selectorCount, is each of the override's own mangled names. The slot fits when
// no name at the target is a member function, as for ___cxa_pure_virtual or a
// method no symbol table names, or when one of them has the override's method
// name. A target named only as other methods refuses the slot.
template <typename Selector, typename EachName>
inline bool fallbackTargetFits(Selector selector, uint32_t selectorCount, EachName eachName) {
    bool same = false, other = false;
    eachName([&](const char *name) {
        if (!methodIdentity(name).valid())
            return;
        bool match = false;
        for (uint32_t s = 0; s < selectorCount && !match; ++s)
            match = sameMethodName(name, selector(s));
        if (match)
            same = true;
        else
            other = true;
    });
    return same || !other;
}

struct SlotMatch {
    enum Result { Found, None, Ambiguous } result;
    uint32_t slot;
};

// The slot of a live vtable holding `target` (a PAC-stripped code address). The
// kernel resolves a composed symbol to `target`, then scans the class's live
// vtable for it; the resulting index is where that method sits this release. A
// target held by two slots is ambiguous and refuses rather than guess; a target
// of 0 (an unresolved symbol) or no match is None.
inline SlotMatch scanVtable(const uint64_t *slots, uint32_t slotCount, uint64_t target) {
    SlotMatch match{SlotMatch::None, 0};
    if (!slots || target == 0)
        return match;
    for (uint32_t i = 0; i < slotCount; ++i) {
        if (slots[i] != target)
            continue;
        if (match.result == SlotMatch::Found)
            return {SlotMatch::Ambiguous, i};
        match = {SlotMatch::Found, i};
    }
    return match;
}

// The slot an override maps to, or NoSlot when no selector resolved (an optional
// override the live parent lacks), or AmbiguousSlot when a selector's address was
// held by two live slots (a hard refusal the kernel's scan found).
constexpr uint32_t NoSlot = UINT32_MAX;
constexpr uint32_t AmbiguousSlot = UINT32_MAX - 1;

enum class PlanError { None, UnmatchedRequired, Ambiguous, Collision };
struct PlanResult {
    PlanError error = PlanError::None;
    uint32_t index = 0; // the override the error refers to
    uint32_t slot = 0;  // the contested slot, for Collision
};

// Turn the kernel's per-override resolved slots into the fill plan. resolvedSlots[k]
// is what the kernel composed + resolved + scanned for override k: a live parent
// slot, NoSlot (no selector resolved), or AmbiguousSlot (a selector's address held
// two slots). The binding resolves each override from the manifest's selector
// table (KernelBinding.cpp, resolveOverride). required[k] marks an override the
// synthesis must place, such as one of a pure parent slot. An AmbiguousSlot
// refuses; a NoSlot refuses if the override is required, else records NoSlot
// (skipped, left inherited); two overrides on one slot refuse (Collision). On success each
// overrideSlot[k] is a parent slot or NoSlot, and synthesized slot n takes an override's
// implementation where some overrideSlot[k] == n, else the live parent's slot n re-signed for the
// new slot's own address.
inline PlanResult planSynthesis(const uint32_t *resolvedSlots, const bool *required,
                                uint32_t overrideCount, uint32_t *overrideSlot) {
    for (uint32_t k = 0; k < overrideCount; ++k) {
        uint32_t slot = resolvedSlots[k];
        if (slot == AmbiguousSlot)
            return {PlanError::Ambiguous, k, 0};
        overrideSlot[k] = slot; // a real slot, or NoSlot
        if (slot == NoSlot && required[k])
            return {PlanError::UnmatchedRequired, k, 0};
    }
    for (uint32_t a = 0; a < overrideCount; ++a) {
        if (overrideSlot[a] == NoSlot)
            continue;
        for (uint32_t b = a + 1; b < overrideCount; ++b)
            if (overrideSlot[b] == overrideSlot[a])
                return {PlanError::Collision, b, overrideSlot[a]};
    }
    return {};
}

// The bind-time reserve check: the live parent instance must fit within the
// consumer stand-in's parent region (field_reserve plus the bases below it), so
// the consumer's own fields, which follow that region, never overlap the live
// parent's. Graft refuses the bind otherwise.
inline bool instanceFits(uint64_t liveParentInstanceSize, uint64_t parentRegionSize) {
    return liveParentInstanceSize <= parentRegionSize;
}

// The bind-time slot-count check: the live parent must have no more vtable method
// words than the consumer's stand-in reserves for it. The consumer's compiled
// vtable is the stand-in's reserved slots followed by the consumer's own appended
// virtuals, so a live parent longer than the stand-in would make synthesis write
// parent methods over those appended slots. Bound synthesis by the stand-in's own
// slot count, not by the whole compiled vtable, and refuse otherwise. The words
// written, the live parent's slots after the two header words, must also fit the
// compiled vtable, writeCap words from word 0, which sizes the scratch buffers; the
// SDK's output always fits, so only a malformed manifest fails this half.
inline bool slotsFit(uint32_t liveParentSlots, uint32_t standInSlots, uint32_t writeCap) {
    return liveParentSlots <= standInSlots && writeCap >= 2 && liveParentSlots <= writeCap - 2;
}

} // namespace synth
} // namespace graft
