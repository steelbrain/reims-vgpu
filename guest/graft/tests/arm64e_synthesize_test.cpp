// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Host test of the arm64e vtable-synthesis pure layer (src/arm64e/Synthesize.h).
// It uses synthetic IOKit-shaped Itanium-mangled names, so it commits no Apple
// symbol data: methodIdentity extracts the part a method and its overrides share
// (const-ness and ref-qualifier included, a non-method symbol has none),
// composeMemberSymbol rebuilds that method's symbol byte for byte under another
// class, scanVtable finds the one slot holding an address (refusing on two),
// fallbackTargetFits refuses a slot the discriminator fallback found whose target
// is named as another method, and planSynthesis turns the kernel's resolved slots
// into a fill plan, refusing an unmatched required slot, an ambiguous live
// vtable, or a collision.
#include "Check.h"
#include "arm64e/Synthesize.h"
#include <cstring>
#include <initializer_list>
#include <stdio.h>

using namespace graft;
using graft::synth::Identity;
using graft::synth::SlotMatch;

namespace {

void identity() {
    // A method and a subclass override of it share the identity, class aside.
    Identity parent = synth::methodIdentity("__ZN19IOMobileFramebuffer6enableEv");
    Identity child = synth::methodIdentity("__ZN16ReimsFramebuffer6enableEv");
    CHECK(parent.valid() && child.valid());
    CHECK(!strcmp(parent.tail, "6enableEv") && parent.quals == 0 && parent.refqual == 0);
    CHECK(!strcmp(child.tail, "6enableEv"));

    // const-ness (the `K` before the class) is part of the identity.
    Identity constGet = synth::methodIdentity("__ZNK19IOMobileFramebuffer9getStateEv");
    CHECK(constGet.valid() && constGet.quals == 1);
    CHECK(!strcmp(constGet.tail, "9getStateEv"));

    // A ref-qualifier is captured too, after the cv-qualifiers.
    Identity refGet = synth::methodIdentity("__ZNO3Foo3barEv");
    CHECK(refGet.valid() && refGet.refqual == 'O' && !strcmp(refGet.tail, "3barEv"));

    // A single-underscore linkage prefix and the double are both accepted.
    CHECK(!strcmp(synth::methodIdentity("_ZN3Foo3barEv").tail, "3barEv"));
    CHECK(!strcmp(synth::methodIdentity("__ZN3Foo3barEv").tail, "3barEv"));

    // Non-member and malformed symbols have no identity.
    CHECK(!synth::methodIdentity(nullptr).valid());
    CHECK(!synth::methodIdentity("_gOSKextUnresolved").valid());         // not _Z
    CHECK(!synth::methodIdentity("__ZL7mac_ops").valid());               // _ZL, not nested
    CHECK(!synth::methodIdentity("__ZTV19IOMobileFramebuffer").valid()); // RTTI, not nested
    CHECK(!synth::methodIdentity("__ZN99Xy").valid()); // class length runs off the end
    puts("arm64e synthesize: pass (method identity extraction)");
}

void composing() {
    char out[256];

    // From a subclass override, rebuild the parent's symbol for the same method:
    // the class component changes, everything else is byte-identical.
    Identity enable = synth::methodIdentity("__ZN16ReimsFramebuffer6enableEv");
    CHECK(synth::composeMemberSymbol(enable, "IOMobileFramebuffer", out, sizeof(out)) > 0);
    CHECK(!strcmp(out, "__ZN19IOMobileFramebuffer6enableEv"));

    // const-ness is re-emitted (the `K` before the class).
    Identity getState = synth::methodIdentity("__ZNK16ReimsFramebuffer9getStateEv");
    CHECK(synth::composeMemberSymbol(getState, "IOMobileFramebuffer", out, sizeof(out)) > 0);
    CHECK(!strcmp(out, "__ZNK19IOMobileFramebuffer9getStateEv"));

    // A pure slot sources its symbol from the concrete sibling, not the parent:
    // compose under the sibling class.
    Identity swap = synth::methodIdentity("__ZN13ReimsGraphics10swap_beginEv");
    CHECK(synth::composeMemberSymbol(swap, "AppleParavirtDisplay", out, sizeof(out)) > 0);
    CHECK(!strcmp(out, "__ZN20AppleParavirtDisplay10swap_beginEv"));

    // The composed symbol round-trips: its identity equals the override's.
    Identity round = synth::methodIdentity(out);
    CHECK(round.valid() && round.quals == swap.quals && round.refqual == swap.refqual &&
          !strcmp(round.tail, swap.tail));

    // A ref-qualifier is re-emitted after the cv-qualifiers, in order.
    Identity refm = synth::methodIdentity("__ZNO4Base3barEv");
    CHECK(synth::composeMemberSymbol(refm, "Derived", out, sizeof(out)) > 0);
    CHECK(!strcmp(out, "__ZNO7Derived3barEv"));

    // An invalid identity (a non-method symbol) composes nothing.
    CHECK(synth::composeMemberSymbol(synth::methodIdentity("__ZTV3Foo"), "Bar", out, sizeof(out)) ==
          0);

    // A buffer too small for the result refuses rather than truncate.
    CHECK(synth::composeMemberSymbol(enable, "IOMobileFramebuffer", out, 10) == 0);
    puts("arm64e synthesize: pass (compose member symbol)");
}

void scanning() {
    // A live vtable of PAC-stripped slot addresses, with a duplicate and a null.
    const uint64_t slots[] = {0x4000, 0x5000, 0x6000, 0x5000, 0};
    const uint32_t count = sizeof(slots) / sizeof(slots[0]);

    CHECK(synth::scanVtable(slots, count, 0x4000).result == SlotMatch::Found);
    CHECK(synth::scanVtable(slots, count, 0x4000).slot == 0);
    CHECK(synth::scanVtable(slots, count, 0x6000).slot == 2);

    // An address no slot holds is None.
    CHECK(synth::scanVtable(slots, count, 0xDEAD).result == SlotMatch::None);
    // An address two slots hold is ambiguous and refuses.
    CHECK(synth::scanVtable(slots, count, 0x5000).result == SlotMatch::Ambiguous);
    // A zero target (an unresolved symbol) or a null vtable is None.
    CHECK(synth::scanVtable(slots, count, 0).result == SlotMatch::None);
    CHECK(synth::scanVtable(nullptr, 0, 0x4000).result == SlotMatch::None);
    puts("arm64e synthesize: pass (vtable scan)");
}

void fallbackNames() {
    // The member name: a source name, or a two-character special name.
    const char *name = nullptr;
    size_t length = 0;
    CHECK(synth::memberName(synth::methodIdentity("__ZN3Foo6enableEv"), name, length) &&
          length == 6 && !strncmp(name, "enable", 6));
    CHECK(synth::memberName(synth::methodIdentity("__ZN3FooD0Ev"), name, length) && length == 2 &&
          !strncmp(name, "D0", 2));
    CHECK(!synth::memberName(synth::methodIdentity("__ZN3Foo9enableEv"), name, length));
    CHECK(!synth::memberName(synth::methodIdentity("___cxa_pure_virtual"), name, length));

    // One method name across classes, qualifiers and parameter forms.
    CHECK(synth::sameMethodName("__ZN3Foo8setStateEb", "__ZNK3Bar8setStateEj"));
    CHECK(!synth::sameMethodName("__ZN3Foo8setStateEb", "__ZN3Foo8getStateEv"));
    CHECK(!synth::sameMethodName("__ZN3Foo3setEv", "__ZN3Foo8setStateEv"));
    CHECK(!synth::sameMethodName("___cxa_pure_virtual", "___cxa_pure_virtual"));

    // The override is ReimsGraphics::setState in either parameter form.
    const char *selectors[] = {"__ZN13ReimsGraphics8setStateEb", "__ZN13ReimsGraphics8setStateEj"};
    auto selector = [&](uint32_t s) { return selectors[s]; };
    auto fits = [&](std::initializer_list<const char *> names) {
        return synth::fallbackTargetFits(selector, 2, [&](auto visit) {
            for (const char *each : names)
                visit(each);
        });
    };
    CHECK(fits({}));                                                // no name at the target
    CHECK(fits({"___cxa_pure_virtual"}));                           // a pure slot
    CHECK(fits({"__ZN6Parent8setStateEj"}));                        // the method, another form
    CHECK(!fits({"__ZN6Parent5flushEv"}));                          // another method
    CHECK(!fits({"__ZN6Parent5flushEv", "_shared_stub"}));          // only another method
    CHECK(fits({"__ZN6Parent5flushEv", "__ZN6Parent8setStateEb"})); // folded with it
    puts("arm64e synthesize: pass (discriminator fallback checks the target's names)");
}

void planning() {
    using synth::AmbiguousSlot;
    using synth::NoSlot;
    using synth::PlanError;
    // planSynthesis works over the slots the kernel resolved for each override
    // and whether each is required. Five overrides: enable, swap_begin, setState,
    // getState, all required, and an optional flush.
    const bool required5[] = {true, true, true, true, false};
    // What the kernel resolved: enable->0, swap (sibling-sourced pure slot)->1, set
    // by its second per-release selector->5, const get->2, optional flush unmatched.
    const uint32_t resolved[] = {0, 1, 5, 2, NoSlot};
    uint32_t mapped[5];
    auto r = synth::planSynthesis(resolved, required5, 5, mapped);
    CHECK(r.error == PlanError::None);
    CHECK(mapped[0] == 0 && mapped[1] == 1 && mapped[2] == 5 && mapped[3] == 2 &&
          mapped[4] == NoSlot);

    // A required override no selector resolved refuses the whole plan.
    const uint32_t none[] = {NoSlot};
    const bool required[] = {true};
    uint32_t one[1];
    auto req = synth::planSynthesis(none, required, 1, one);
    CHECK(req.error == PlanError::UnmatchedRequired && req.index == 0);

    // An optional override the live parent lacks is skipped, left inherited.
    const bool optional[] = {false};
    uint32_t oneOpt[1];
    auto opt = synth::planSynthesis(none, optional, 1, oneOpt);
    CHECK(opt.error == PlanError::None && oneOpt[0] == NoSlot);

    // Two overrides resolved onto one slot collide.
    const uint32_t collideSlots[] = {0, 0};
    const bool collide[] = {true, true};
    uint32_t two[2];
    auto c = synth::planSynthesis(collideSlots, collide, 2, two);
    CHECK(c.error == PlanError::Collision && c.slot == 0 && c.index == 1);

    // A selector whose address the kernel's scan found in two live slots refuses.
    const uint32_t ambiguous[] = {AmbiguousSlot};
    const bool want[] = {true};
    uint32_t amb[1];
    auto a = synth::planSynthesis(ambiguous, want, 1, amb);
    CHECK(a.error == PlanError::Ambiguous && a.index == 0);

    // The reserve/size check: the live parent must fit the stand-in's region,
    // up to its last byte.
    CHECK(synth::instanceFits(0x1800, 0x3000));
    CHECK(synth::instanceFits(0x3000, 0x3000));
    CHECK(!synth::instanceFits(0x3001, 0x3000));

    // The slot-count check: the live parent must have no more method words than
    // the stand-in reserves, or synthesis would overwrite appended virtuals.
    CHECK(synth::slotsFit(300, 400, 410));
    CHECK(synth::slotsFit(400, 400, 402));
    CHECK(!synth::slotsFit(401, 400, 410));
    // A stand-in reserve larger than the compiled vtable, which only a malformed
    // manifest declares, must not let the parent's words overrun the vtable.
    CHECK(!synth::slotsFit(400, 400, 401));
    CHECK(!synth::slotsFit(0, 400, 1));
    puts("arm64e synthesize: pass (vtable plan + reserve + slot-count check)");
}

} // namespace

int main() {
    identity();
    composing();
    scanning();
    fallbackNames();
    planning();
}
