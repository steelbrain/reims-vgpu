// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Host test of a synthesized consumer's census and manifest (sdk/tool/consumer.h)
// on the display in tests/synthesized/, a subclass of the SDK's
// IOMobileFramebufferLegacy stand-in, which make test builds for arm64e with
// graft-kext build. The 19 declared overrides resolve by method name, the
// consumer's own undeclared methods (its destructors and getMetaClass) are
// carried as required overrides, and the stand-in's reserved slots and the
// inherited binds are left out of the synthesis records.
//
//     synth_emit_test DECLARATION DIRECTORY
#include "ToolTest.h"

#include <algorithm>
#include <set>

using namespace graft;
using namespace graft::tool;
using graft_test::Built;
using graft_test::manifestOf;
using graft_test::refuses;

namespace {

// The 9 pure methods a subclass must implement (required), and the 10 others it
// overrides but the live parent still provides (optional: an absent one stays
// inherited).
const std::set<std::string> Required = {"set_notify_tag_mask_gated",
                                        "supported_frame_info",
                                        "factory_portal",
                                        "executeFactoryConfigurations_gated",
                                        "set_frame_info_notification_state_gated",
                                        "do_swap_info_notify_gated",
                                        "notify_swap_done_gated",
                                        "notify_crc_ready_gated",
                                        "are_timings_enabled"};
const std::set<std::string> Optional = {"start_hardware",
                                        "stop_hardware_gated",
                                        "swap_begin_gated",
                                        "verify_swap",
                                        "swap_layer_map",
                                        "surface_layer_map",
                                        "surface_layer_unmap",
                                        "start",
                                        "stop",
                                        "free"};
// supported_frame_info's three-argument form, which the declaration names as a
// selector alternative under the consumer class.
const std::string SupportedFrameInfoThree = "__ZN12SynthDisplay20supported_frame_infoEjPjPA64_c";
// The consumer's own methods its declaration does not name: the destructors and
// getMetaClass OSDefineMetaClassAndStructors defines. They are carried as
// required overrides, so the bind places them or refuses, never leaving the
// parent's in their slots.
const std::set<std::string> Carried = {"__ZN12SynthDisplayD1Ev", "__ZN12SynthDisplayD0Ev",
                                       "__ZNK12SynthDisplay12getMetaClassEv"};
const size_t OverrideCount = Required.size() + Optional.size() + Carried.size();

void oneSynthesizedClass(const Built &built, const Census &found) {
    CHECK(found.synth.size() == 1);
    const SynthClassRecord &record = found.synth[0];
    CHECK(record.name == "SynthDisplay");
    CHECK(record.parentClass == "IOMobileFramebufferLegacy");
    CHECK(record.siblingClass == "AppleParavirtDisplay");
    CHECK(record.library == 0);
    // The compiled vtable runs past the stand-in's reserved slots
    // (GRAFT_IOMFB_VTABLE_RESERVE, 512), and the parent region past its reserved
    // field (GRAFT_IOMFB_LEGACY_RESERVE, 0x2000).
    CHECK(record.vtableSlotCap > 512);
    CHECK(record.parentRegion > 0x2000);
    // The cap counts the compiled vtable's words from word 0, as Graft compares
    // the live vtable's words with it: the method slots and the two leading words.
    CHECK(record.vtableSlotCap == built.layouts.synthesized.at(0).slots.size() + 2);
    puts("synthesized class: pass (its names, library, slot cap and parent region)");
}

// Graft fills the vtable from its word 0: the offset-to-top and typeinfo words,
// zero without RTTI, then the method slots. An offset at the address point would
// shift every synthesized word two slots late.
void vtableOffsetIsWordZero(const Built &built, const Census &found) {
    Image image(built.census);
    const Symbol &vtable = image.symbol(vtableName("SynthDisplay"));
    CHECK(found.synth[0].vtableOffset == vtable.value - image.base);
    CHECK(image.read(vtable.value, 16) == Bytes(16));
    puts("vtable offset: pass (the vtable's first word)");
}

void overridesResolveByMethodName(const Census &found) {
    std::map<std::string, const SynthOverrideRecord *> byMethod, carried;
    for (const SynthOverrideRecord &override : found.synth[0].overrides) {
        const std::string &primary = override.selectors.at(0);
        if (Carried.count(primary))
            carried[primary] = &override;
        else
            byMethod[memberName(primary).value()] = &override;
        // The primary selector is the consumer's own method (a const method
        // carries the __ZNK form), pointing at its own code, signed as an
        // arm64e vtable slot is.
        CHECK(primary.find("12SynthDisplay") != std::string::npos);
        CHECK(override.impl != 0);
        CHECK(override.opcode & manifest::SiteSigned);
    }
    std::set<std::string> methods, carriedNames, declared = Required;
    declared.insert(Optional.begin(), Optional.end());
    for (const auto &entry : byMethod)
        methods.insert(entry.first);
    for (const auto &entry : carried)
        carriedNames.insert(entry.first);
    CHECK(methods == declared);
    CHECK(carriedNames == Carried);
    for (const std::string &name : Required)
        CHECK(byMethod.at(name)->required);
    for (const std::string &name : Optional)
        CHECK(!byMethod.at(name)->required);
    for (const auto &entry : carried)
        CHECK(entry.second->required && entry.second->selectors.size() == 1);
    const SynthOverrideRecord &frameInfo = *byMethod.at("supported_frame_info");
    CHECK(frameInfo.selectors.size() == 2 && frameInfo.selectors[1] == SupportedFrameInfoThree);
    puts("overrides: pass (by method name, the carried methods, the selector alternative)");
}

void packPartitionsTheRecords(const Built &built, const Census &found) {
    Bytes bytes = packManifest(built.declaration, built.layouts, found, built.activation, {},
                               manifest::VersionArm64e);
    manifest::View view = manifestOf(bytes);
    const manifest::Header &header = view.header();
    // The stand-in parent is measured at bind, not checked slot by slot, so it
    // records no slots.
    CHECK(header.classes.count == 1);
    CHECK(view.row<manifest::Class>(header.classes, 0).slotCount == 0);
    CHECK(header.slots.count == 0);
    CHECK(header.synthClasses.count == 1);
    CHECK(header.synthOverrides.count == OverrideCount);
    // The one class claims every override in order from 0, and the overrides
    // partition the selector table in order.
    auto synth = view.row<manifest::SynthClass>(header.synthClasses, 0);
    CHECK(synth.firstOverride == 0 && synth.overrideCount == OverrideCount);
    CHECK(synth.vtableSlotCap == found.synth[0].vtableSlotCap);
    CHECK(synth.parentRegion > 0x2000);
    uint32_t cursor = 0;
    for (uint32_t i = 0; i < header.synthOverrides.count; ++i) {
        auto override = view.row<manifest::SynthOverride>(header.synthOverrides, i);
        CHECK(override.firstSelector == cursor && override.selectorCount >= 1);
        CHECK(override.impl != 0 && override.reserved == 0);
        CHECK(override.opcode & manifest::SiteSigned);
        cursor += override.selectorCount;
    }
    // One selector an override, and supported_frame_info's alternative.
    CHECK(cursor == header.selectors.count && header.selectors.count == OverrideCount + 1);
    puts("synthesis records: pass (one class, its overrides partitioning the selectors)");
}

// Layouts whose slots name the display's methods in other places than its
// vtable holds them, as stale layouts would, pair one method's selector with
// another's code, so census refuses them.
void staleLayoutsAreRefused(const Built &built) {
    Layouts stale = built.layouts;
    std::vector<std::string> &slots = stale.synthesized.at(0).slots;
    auto slotOf = [&](const std::string &name) {
        auto found = std::find_if(slots.begin(), slots.end(), [&](const std::string &slot) {
            return slot.find("12SynthDisplay") != std::string::npos && memberName(slot) == name;
        });
        CHECK(found != slots.end());
        return found;
    };
    std::iter_swap(slotOf("start_hardware"), slotOf("stop_hardware_gated"));
    CHECK(refuses([&] { census(Image(built.census), built.declaration, stale); },
                  "synthesized vtable slot does not hold the method its layout names"));
    puts("stale layouts: pass (a selector paired with another method's code is refused)");
}

// A synthesized class's parent records only the stand-in's slot count, so Graft
// refuses a consumer vtable derived from it, and the build refuses the manifest
// first, by the same rules.
void sharedParentIsRefused(const Built &built, const Census &found) {
    Census shared = found;
    auto parentSlots = static_cast<uint32_t>(built.layouts.classes.at(0).slots.size());
    shared.vtables.push_back({found.synth[0].vtableOffset + 16, 0, parentSlots, {}, {}});
    CHECK(refuses(
            [&] {
                packManifest(built.declaration, built.layouts, shared, built.activation, {},
                             manifest::VersionArm64e);
            },
            "Graft would refuse this manifest: invalid vtables record 0"));
    puts("shared parent: pass (a consumer vtable of a synthesized class's parent is refused)");
}

} // namespace

int main(int argc, char **argv) {
    CHECK(argc == 3);
    Built built(argv[1], argv[2], "SynthDisplay");
    Census found = census(Image(built.census), built.declaration, built.layouts);
    oneSynthesizedClass(built, found);
    vtableOffsetIsWordZero(built, found);
    overridesResolveByMethodName(found);
    packPartitionsTheRecords(built, found);
    staleLayoutsAreRefused(built);
    sharedParentIsRefused(built, found);
}
