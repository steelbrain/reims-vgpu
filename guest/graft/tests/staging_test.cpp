// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Host test of consumer staging (sdk/tool/consumer.h) on the fixture consumer in
// tests/consumer/, which make test builds with graft-kext build for x86-64 and
// for arm64e. It reads each build's own steps back: the probe's layouts, the
// census of the first link, and the staging of the second. The arm64e build
// also covers reading chained fixups (sdk/tool/macho.h).
//
//     staging_test DECLARATION X86_64_DIRECTORY ARM64E_DIRECTORY
#include "ToolTest.h"
#include "process.h"

#include <algorithm>
#include <filesystem>
#include <random>
#include <set>
#include <tuple>

using namespace graft;
using namespace graft::tool;
using graft_test::Built;
using graft_test::graftSection;
using graft_test::headers;
using graft_test::manifestOf;
using graft_test::refuses;

namespace {

const std::string Provider = "ai.aneesiqbal.graft.test.Provider";
// Fields of a section header (section_64): its size follows its two names and
// its address, then its file offset, and its flags come after the relocations.
constexpr uint64_t SectionSizeField = 40, SectionOffsetField = 48, SectionFlagsField = 64;
// Each import's placeholder in __DATA,__graft starts on this boundary.
constexpr uint64_t PlaceholderAlignment = 16;

bool contains(const std::vector<std::string> &names, const std::string &name) {
    return std::find(names.begin(), names.end(), name) != names.end();
}

void compilerLayout(const Built &x86) {
    const ClassLayout &layout = x86.layouts.classes.at(0);
    CHECK(layout.size == 32);
    CHECK(contains(layout.slots, "__ZN6Parent9inheritedEi"));
    CHECK(contains(layout.slots, "__ZN6Parent7changedEi"));
    for (const std::string &slot : layout.slots)
        CHECK(slot.find("GraftLayoutProbe") == std::string::npos);
    puts("compiler layout: pass (the parent's size and named slots, no probe class)");
}

// The staged bundle graft-kext build wrote: a version-arch manifest with a
// zero-filled tail, and no __DATA_CONST, so the import GOT and vtables stay in
// writable __DATA, where Graft writes them.
void dormantBinary(const Built &built, uint32_t version) {
    Image image(built.staged);
    Bytes section = graftSection(image);
    manifest::View view = manifestOf(section);
    CHECK(view.header().magic == manifest::Magic && view.header().version == version);
    CHECK(std::all_of(section.begin() + view.header().size, section.end(),
                      [](uint8_t byte) { return byte == 0; }));
    for (const Segment &segment : image.segments)
        CHECK(segment.name != "__DATA_CONST");
}

void stageRemapsSymbolsAndRelocations(const Built &x86) {
    Image original(x86.unbound);
    Census found = census(original, x86.declaration, x86.layouts);
    CHECK(found.imports.count("_parent_global"));
    CHECK(found.vtables.size() == 1);
    CHECK(!found.vtables[0].overrides.empty());
    CHECK(found.vtables[0].count == x86.layouts.consumers.at(0).slots.size());
    Image staged = stageImage(original, x86.declaration, x86.layouts, x86.activation);
    for (const Symbol &symbol : staged.symbols)
        CHECK(!(symbol.undefined() && found.imports.count(symbol.name)));
    // Every external relocation still names exactly its original symbol after the
    // symbol table has been regrouped.
    std::vector<Relocation> before = original.relocations(), after = staged.relocations();
    CHECK(before.size() == after.size());
    for (size_t i = 0; i < before.size(); ++i) {
        CHECK(before[i].address == after[i].address);
        if (before[i].external)
            CHECK(original.symbols.at(before[i].symbolIndex).name ==
                  staged.symbols.at(after[i].symbolIndex).name);
    }
    const Section &section = staged.section("__DATA", "__graft");
    Bytes bytes = graftSection(staged);
    manifest::View view = manifestOf(bytes);
    const manifest::Header &header = view.header();
    CHECK(header.magic == manifest::Magic && header.version == manifest::Version);
    CHECK(header.headerSize == sizeof(manifest::Header));
    CHECK(header.sites.count == found.sites.size());
    CHECK(header.vtables.count == 1);
    CHECK(header.size <= section.size);
    for (const auto &entry : found.imports) {
        const Symbol &symbol = staged.symbol(entry.first);
        CHECK(symbol.kind == N_SECT); // a private local, never a provider export
        CHECK(symbol.value % PlaceholderAlignment == 0);
        CHECK(symbol.section == section.index);
    }
    CHECK(staged.symbol("__ZN8Consumer10graftClassE").value ==
          original.symbol("__ZN8Consumer10superClassE").value);
    for (const Symbol &symbol : staged.symbols)
        CHECK(symbol.name != "__ZN8Consumer10superClassE");
    dormantBinary(x86, manifest::Version);
    puts("x86-64 staging: pass (placeholders, renumbered relocations, the renamed metaclass)");
}

// The compiler marks a weak_import reference N_WEAK_REF; the staged manifest
// carries ImportWeak for exactly those imports, so the binder binds an absent one
// to gOSKextUnresolved (tests/manifest_test.cpp).
void weakReferencesMarkTheirImports(const Built &x86) {
    Image original(x86.unbound);
    std::map<std::string, Import> imports = census(original, x86.declaration, x86.layouts).imports;
    CHECK(imports.at("_weak_present").weak && imports.at("_weak_absent").weak);
    CHECK(!imports.at("_parent_global").weak);
    Bytes bytes = graftSection(stageImage(original, x86.declaration, x86.layouts, x86.activation));
    manifest::View view = manifestOf(bytes);
    std::map<std::string, uint32_t> flags;
    for (uint32_t i = 0; i < view.header().imports.count; ++i) {
        auto row = view.row<manifest::Import>(view.header().imports, i);
        uint32_t name = view.row<uint32_t>(view.header().candidates, row.firstCandidate);
        flags[view.string(name)] = row.flags;
    }
    CHECK(flags.at("_weak_present") == manifest::ImportWeak);
    CHECK(flags.at("_weak_absent") == manifest::ImportWeak);
    CHECK(flags.at("_parent_global") == 0);
    std::set<uint32_t> values;
    for (const auto &entry : flags)
        values.insert(entry.second);
    CHECK((values == std::set<uint32_t>{0, manifest::ImportWeak}));
    puts("weak imports: pass (exactly the weak references carry ImportWeak)");
}

// A provider the consumer declares no class from is verified by its anchor,
// which the manifest carries; without one the declaration is refused.
void providerAnchor(const Built &x86, const std::string &scratch) {
    Declaration config = x86.declaration;
    config.libraries.push_back(
            {"ai.aneesiqbal.graft.test.ImportsOnly", std::string("ImportsClass"), {}});
    Census found = census(Image(x86.unbound), x86.declaration, x86.layouts);
    Bytes bytes = packManifest(config, x86.layouts, found, x86.activation, {}, manifest::Version);
    manifest::View view = manifestOf(bytes);
    CHECK(view.header().libraries.count == 2);
    auto first = view.row<manifest::Library>(view.header().libraries, 0);
    auto second = view.row<manifest::Library>(view.header().libraries, 1);
    CHECK(std::string(view.string(first.identifier)) == Provider && first.anchor == 0);
    CHECK(std::string(view.string(second.identifier)) == "ai.aneesiqbal.graft.test.ImportsOnly");
    CHECK(std::string(view.string(second.anchor)) == "ImportsClass");

    auto declaration = [&](const std::string &anchor) {
        std::string path = scratch + "/anchor.json";
        writeText(path, R"({
  "libraries": [{"id": ")" + Provider +
                                R"("}, {"id": "ai.aneesiqbal.graft.test.ImportsOnly")" + anchor +
                                R"(}],
  "classes": [{"name": "Parent", "header": "Fixture.h", "library": 0}],
  "consumers": [{"name": "Consumer", "header": "Fixture.h", "parent": 0}]
}
)");
        return path;
    };
    std::string anchorless = declaration("");
    CHECK(refuses([&] { readDeclaration(anchorless); },
                  "ImportsOnly has no declared class and no anchor"));
    std::string nested = declaration(R"(, "anchor": "Imports::Class")");
    CHECK(refuses([&] { readDeclaration(nested); }, "only top-level C++ class names"));
    puts("provider anchors: pass (carried by the manifest, required without a declared class)");
}

void stagingGatesPersonalities(const Built &x86, const std::string &scratch) {
    std::string unbound = x86.work + "/Unbound.kext";
    std::string output = scratch + "/Staged.kext";
    stageBundle(unbound, x86.declaration, x86.layouts, output);
    Plist info = parsePlist(readFile(output + "/Contents/Info.plist"), "Info.plist");
    const Plist *personalities = info.find("IOKitPersonalities");
    CHECK(personalities && !personalities->dict.empty());
    for (const auto &entry : personalities->dict) {
        const Plist *match = entry.second.find("IOPropertyMatch");
        CHECK(match && match->kind == Plist::Kind::Dict && match->dict.size() == 1);
        CHECK(match->dict[0].first == x86.activation.gate && match->dict[0].second.boolean);
        const Plist *identifier = entry.second.find("CFBundleIdentifier");
        CHECK(identifier && identifier->text == x86.info.find("CFBundleIdentifier")->text);
    }
    const Plist *libraries = info.find("OSBundleLibraries");
    CHECK(libraries && !libraries->find(Provider) && libraries->find("com.apple.kpi.libkern"));
    CHECK(readFile(unbound + "/Contents/MacOS/Fixture") == x86.unbound);
    CHECK(refuses([&] { stageBundle(unbound, x86.declaration, x86.layouts, output); },
                  "already exists"));
    // The manifest's provider matching is each personality as written, without the
    // gate it gained or the class it names.
    Bytes bytes = graftSection(Image(readFile(output + "/Contents/MacOS/Fixture")));
    manifest::View view = manifestOf(bytes);
    const char *staged = view.string(view.header().matching);
    Plist provider = *x86.info.find("IOKitPersonalities")->find("Fixture");
    provider.erase("IOClass");
    Plist matching = Plist::makeDict();
    matching.set("Fixture", provider);
    CHECK(staged && Bytes(staged, staged + strlen(staged)) == dumpPlist(matching));
    puts("staged bundle: pass (gated personalities, provider matching, no provider dependency, "
         "input untouched)");
}

// A copy of the fixture's unbound bundle in scratch as name, with its Info.plist
// edited by edit.
template <typename Edit>
std::string unboundCopy(const Built &x86, const std::string &scratch, const std::string &name,
                        Edit &&edit) {
    std::string bundle = scratch + "/" + name;
    std::filesystem::copy(x86.work + "/Unbound.kext", bundle,
                          std::filesystem::copy_options::recursive);
    Plist info = x86.info;
    edit(info);
    writeFile(bundle + "/Contents/Info.plist", dumpPlist(info));
    return bundle;
}

// Staging takes the declared libraries out of the per-architecture dependency
// lists as well. A personality's own IOPropertyMatch, which gains the gate, and
// other personalities change the manifest's provider matching and so its size,
// which the fixture's reservation does not leave room for: graft_kext_test.py
// checks them through a whole build.
void stagingLibrariesAndRefusals(const Built &x86, const std::string &scratch) {
    std::string bundle = unboundCopy(x86, scratch, "Libraries.kext", [&](Plist &info) {
        info.set("OSBundleLibraries_x86_64", *info.find("OSBundleLibraries"));
    });
    std::string output = scratch + "/Libraries-staged.kext";
    stageBundle(bundle, x86.declaration, x86.layouts, output);
    Plist info = parsePlist(readFile(output + "/Contents/Info.plist"), "Info.plist");
    const Plist *libraries = info.find("OSBundleLibraries_x86_64");
    CHECK(libraries && !libraries->find(Provider) && libraries->find("com.apple.kpi.libkern"));

    std::string unmatchable = unboundCopy(x86, scratch, "Unmatchable.kext", [](Plist &info) {
        info.find("IOKitPersonalities")
                ->find("Fixture")
                ->set("IOPropertyMatch", Plist::makeString("model"));
    });
    CHECK(refuses(
            [&] {
                stageBundle(unmatchable, x86.declaration, x86.layouts,
                            scratch + "/Unmatchable-staged.kext");
            },
            "has an IOPropertyMatch that is not a dictionary or a list of them"));
    std::string providerless = unboundCopy(x86, scratch, "Providerless.kext", [](Plist &info) {
        info.find("IOKitPersonalities")->find("Fixture")->erase("IOProviderClass");
    });
    CHECK(refuses(
            [&] {
                stageBundle(providerless, x86.declaration, x86.layouts,
                            scratch + "/Providerless-staged.kext");
            },
            "consumer personality Fixture names no IOProviderClass"));
    puts("staged Info.plist: pass (per-architecture libraries, a personality IOKit cannot "
         "match refused)");
}

// A staging that fails part way leaves neither its output nor its partial copy.
void failedStagingLeavesNoOutput(const Built &x86, const std::string &scratch) {
    namespace fs = std::filesystem;
    std::string bundle = unboundCopy(x86, scratch, "ReadOnly.kext", [](Plist &) {});
    // The copy keeps the read-only Info.plist, so staging cannot rewrite it.
    fs::permissions(bundle + "/Contents/Info.plist",
                    fs::perms::owner_read | fs::perms::group_read | fs::perms::others_read);
    std::string output = scratch + "/ReadOnly-staged.kext";
    CHECK(refuses([&] { stageBundle(bundle, x86.declaration, x86.layouts, output); },
                  "cannot write"));
    CHECK(!fs::exists(output));
    CHECK(!fs::exists(scratch + "/.ReadOnly-staged.kext.staging"));
    puts("failed staging: pass (no output and no partial copy)");
}

void refusesShortReservationAndWrongLayout(const Built &x86) {
    Image image(x86.unbound);
    Bytes shortened = x86.unbound;
    uint64_t header =
            headers(x86.unbound).sections.at(image.section("__DATA", "__graft").index - 1);
    store64(&shortened[header + SectionSizeField], 1);
    CHECK(refuses(
            [&] { stageImage(Image(shortened), x86.declaration, x86.layouts, x86.activation); },
            "reservation is too small"));
    Layouts layouts = x86.layouts;
    layouts.classes.at(0).name = "Other";
    CHECK(refuses([&] { census(image, x86.declaration, layouts); }, "another declaration"));
    puts("refusals: pass (a short reservation, layouts of another declaration)");
}

// A __DATA,__graft of a zerofill type has no bytes in the file, so staging
// refuses it by name rather than write the manifest wherever its header points,
// here at the file's end.
void refusesZerofillReservation(const Built &x86) {
    Image image(x86.unbound);
    uint64_t header =
            headers(x86.unbound).sections.at(image.section("__DATA", "__graft").index - 1);
    // S_ZEROFILL, S_GB_ZEROFILL and S_THREAD_LOCAL_ZEROFILL
    for (uint32_t type : {0x1u, 0xcu, 0x12u}) {
        Bytes data = x86.unbound;
        store32(&data[header + SectionFlagsField], type);
        store32(&data[header + SectionOffsetField], static_cast<uint32_t>(data.size()));
        CHECK(refuses(
                [&] { stageImage(Image(data), x86.declaration, x86.layouts, x86.activation); },
                "__DATA,__graft is a zerofill section"));
    }
    puts("zerofill reservation: pass (refused, nothing written past the file)");
}

// An x86-64 vtable slot may point at the consumer's own method through an
// external relocation to the method's symbol, the slot holding only the addend.
// Census reads it as the override it is, at the symbol, as it does the local
// relocation the linker writes.
void externalPointerToOwnCode(const Built &x86) {
    constexpr uint32_t External = 1u << 27; // relocation_info's r_extern
    Image image(x86.unbound);
    uint64_t slot = image.symbol(vtableName("Consumer")).value + 16 + 3 * 8; // changed(int)
    std::vector<Relocation> relocations = image.relocations();
    auto local =
            std::find_if(relocations.begin(), relocations.end(),
                         [&](const Relocation &relocation) { return relocation.address == slot; });
    CHECK(local != relocations.end() && !local->external);
    Bytes data = x86.unbound;
    uint32_t info = load32(&data[local->entryOffset + 4]);
    store32(&data[local->entryOffset + 4],
            (info & ~0xffffffu) | External | image.symbol("__ZN8Consumer7changedEi").index);
    store64(&data[image.fileOffset(slot, 8)], 0);
    Census original = census(image, x86.declaration, x86.layouts);
    Census external = census(Image(data), x86.declaration, x86.layouts);
    CHECK(external.vtables.size() == 1);
    CHECK(external.vtables[0].overrides == original.vtables[0].overrides);
    puts("external pointer: pass (an override through its symbol)");
}

// The header, command list, sections, symbol names and relocation arrays all
// have independent bounds checks.
void truncatedMachO(const Built &x86) {
    size_t size = x86.unbound.size();
    for (size_t length :
         {size_t(0), size_t(1), size_t(31), size_t(32), size_t(40), size / 2, size - 1}) {
        Bytes cut(x86.unbound.begin(), x86.unbound.begin() + static_cast<ptrdiff_t>(length));
        CHECK(refuses([&] { Image image(cut); }, ""));
    }
    puts("truncated Mach-O: pass (every cut is refused)");
}

void requiredMethodSubsetKeepsOverrides(const Built &x86) {
    Declaration config = x86.declaration;
    config.classes.at(0).required = std::vector<std::string>{"__ZN6Parent9inheritedEi"};
    Image staged = stageImage(Image(x86.unbound), config, x86.layouts, x86.activation);
    Bytes bytes = graftSection(staged);
    manifest::View view = manifestOf(bytes);
    auto parent = view.row<manifest::Class>(view.header().classes, 0);
    CHECK(parent.instanceSize == 32);
    CHECK(parent.parentSlots == x86.layouts.classes.at(0).slots.size());
    CHECK(parent.slotCount == 4); // the destructor pair, the override, the named call
    CHECK(parent.slotCount < parent.parentSlots); // the unused method is not required
    config.classes.at(0).required = std::vector<std::string>{"_not_a_method"};
    CHECK(refuses([&] { stageImage(Image(x86.unbound), config, x86.layouts, x86.activation); },
                  "no unique compiler slot"));
    puts("required methods: pass (a named subset keeps the overridden slots)");
}

// With no required list every parent slot is verified except the kernel's
// reserved placeholders, whose live slot may hold a method the header does not
// name.
void defaultRequiredSkipsReservedSlots() {
    ClassLayout layout{"Parent",
                       0,
                       {"__ZN6Parent3fooEv", "__ZN15OSMetaClassBase25_RESERVEDOSMetaClassBase3Ev",
                        "__ZN9IOService19_RESERVEDIOService0Ev", "__ZN6Parent3barEv"}};
    CHECK((requiredSlots(ClassDeclaration{}, layout, {}) == std::vector<uint32_t>{0, 3}));
    VtableRecord overriding{0, 0, 4, {1}, {}};
    CHECK((requiredSlots(ClassDeclaration{}, layout, {&overriding}) ==
           std::vector<uint32_t>{0, 1, 3}));
    puts("default required slots: pass (reserved slots skipped unless overridden)");
}

void chainedFixups(const Built &arm64e, const Built &x86) {
    Image image(arm64e.census);
    CHECK(image.arch == "arm64e");
    const std::vector<ChainedFixup> &fixups = image.fixups();
    std::set<std::string> bound, weak;
    std::set<uint64_t> addresses;
    size_t rebases = 0;
    for (const ChainedFixup &fixup : fixups) {
        if (fixup.bind) {
            bound.insert(fixup.symbol);
            if (fixup.weak)
                weak.insert(fixup.symbol);
        } else {
            ++rebases;
            // Every rebase targets the image.
            CHECK(std::any_of(image.segments.begin(), image.segments.end(), [&](const Segment &s) {
                return s.address <= fixup.target && fixup.target < s.address + s.vmsize;
            }));
        }
        if (fixup.auth)
            CHECK(fixup.auth->key < 4 && fixup.auth->diversity <= 0xffff);
        // -no_data_const folds the GOTs into __DATA, so every site is in one of
        // its sections.
        CHECK(std::any_of(image.sections.begin(), image.sections.end(), [&](const Section &s) {
            return s.segment == "__DATA" && s.address <= fixup.address &&
                   fixup.address < s.address + s.size;
        }));
        CHECK(addresses.insert(fixup.address).second);
    }
    CHECK(rebases && bound.size());
    for (const char *name : {"_parent_global", "_weak_present", "_weak_absent",
                             "__ZN6Parent9inheritedEi"}) // an inherited method, bound by name
        CHECK(bound.count(name));
    CHECK((weak == std::set<std::string>{"_weak_present", "_weak_absent"}));

    Image other(x86.census);
    CHECK(other.arch == "x86_64");
    CHECK(refuses([&] { other.fixups(); }, "chained fixups are read only from an arm64e image"));
    puts("chained fixups: pass (binds by name, weak flags, rebases inside the image)");
}

// Chained fixups a kext's link never writes, or whose tables reach outside the
// fixups data, are refused, and corrupting the data makes reading them refuse
// it, never read past it.
void malformedChainedFixups(const Built &arm64e) {
    const Bytes &good = arm64e.census;
    uint64_t blob = 0, blobSize = 0;
    for (uint64_t at = MACH_HEADER_SIZE, n = 0; n < load32(&good[16]);
         ++n, at += load32(&good[at + 4])) {
        if (load32(&good[at]) == LC_DYLD_CHAINED_FIXUPS) {
            blob = load32(&good[at + 8]);
            blobSize = load32(&good[at + 12]);
        }
    }
    CHECK(blob && blobSize);
    // The first segment with fixups, its page size and its first page's start:
    // dyld_chained_starts_in_image, then dyld_chained_starts_in_segment.
    uint64_t starts = blob + load32(&good[blob + 4]), segment = 0;
    for (uint32_t i = 0; !segment && i < load32(&good[starts]); ++i)
        segment =
                load32(&good[starts + 4 + 4 * i]) ? starts + load32(&good[starts + 4 + 4 * i]) : 0;
    CHECK(segment);
    uint16_t pageSize = load16(&good[segment + 4]);
    uint64_t firstStart = segment + 22;
    CHECK(load16(&good[firstStart]) < pageSize);
    auto refusedAs = [&](auto edit, const std::string &reason) {
        Bytes data = good;
        edit(data);
        return refuses([&] { Image(data).fixups(); }, reason);
    };
    CHECK(refusedAs([&](Bytes &data) { store32(&data[blob + 4], static_cast<uint32_t>(blobSize)); },
                    "chained fixups reach outside their data"));
    CHECK(refusedAs([&](Bytes &data) { store16(&data[firstStart], pageSize); },
                    "chained page start outside its page"));
    CHECK(refusedAs(
            [&](Bytes &data) {
                store16(&data[firstStart],
                        static_cast<uint16_t>(0x8000 | load16(&data[firstStart])));
            },
            "chained page with several starts is unsupported"));

    // An unsigned rebase in place of _parent_global's unsigned bind: target:43,
    // high8:8, then the entry's own next.
    Image original(good);
    const std::vector<ChainedFixup> &fixups = original.fixups();
    auto bind = std::find_if(fixups.begin(), fixups.end(), [](const ChainedFixup &fixup) {
        return fixup.symbol == "_parent_global";
    });
    CHECK(bind != fixups.end() && !bind->auth);
    uint64_t rebase = (load64(&good[bind->entryOffset]) & uint64_t(0x7ff) << 51) | 0x4008;
    Bytes plain = good;
    store64(&plain[bind->entryOffset], rebase);
    Image image(plain);
    CHECK(std::any_of(image.fixups().begin(), image.fixups().end(), [&](const ChainedFixup &f) {
        return f.address == bind->address && !f.bind && f.target == image.base + 0x4008;
    }));
    CHECK(refusedAs(
            [&](Bytes &data) { store64(&data[bind->entryOffset], rebase | uint64_t(0x12) << 43); },
            "chained rebase with a top byte is unsupported"));

    std::mt19937 random(11);
    size_t refused = 0;
    for (int round = 0; round < 2000; ++round) {
        Bytes data = good;
        for (uint32_t n = 1 + random() % 4; n; --n)
            data[blob + random() % blobSize] = static_cast<uint8_t>(random());
        try {
            Image(data).fixups();
        } catch (const Error &) {
            ++refused;
        }
    }
    CHECK(refused);
    puts("malformed chained fixups: pass (refused, and 2000 corruptions read or refused)");
}

// census reads the same imports and vtables from an arm64e kext's chained
// fixups as from an x86-64 kext's relocations; arm64e has one GOT slot where
// x86-64 may have several sites.
void arm64eCensusMatchesX86(const Census &arm64e, const Built &x86) {
    Census other = census(Image(x86.unbound), x86.declaration, x86.layouts);
    std::set<std::string> names, otherNames, sited, otherSited;
    for (const auto &entry : arm64e.imports)
        names.insert(entry.first);
    for (const auto &entry : other.imports)
        otherNames.insert(entry.first);
    CHECK(names == otherNames);
    auto shapes = [](const Census &found) {
        std::vector<std::tuple<uint32_t, uint32_t, std::vector<uint32_t>>> out;
        for (const VtableRecord &vtable : found.vtables)
            out.emplace_back(vtable.parent, vtable.count, vtable.overrides);
        std::sort(out.begin(), out.end());
        return out;
    };
    CHECK(shapes(arm64e) == shapes(other));
    for (const Site &site : arm64e.sites)
        sited.insert(site.symbol);
    for (const Site &site : other.sites)
        otherSited.insert(site.symbol);
    CHECK(sited == otherSited);
    puts("arm64e census: pass (the imports, vtables and sited symbols of the x86-64 build)");
}

// Staging writes a version-5 manifest whose authenticated sites carry their
// signing context, and rewrites each provider-import bind into a rebase at its
// placeholder, so the dormant consumer carries nothing the boot linker cannot
// resolve.
void arm64eStaging(const Built &arm64e, const Census &found) {
    dormantBinary(arm64e, manifest::VersionArm64e);
    Image image(arm64e.staged);
    Bytes bytes = graftSection(image);
    manifest::View view = manifestOf(bytes);
    CHECK(!found.sites.empty());
    size_t signedSites = 0;
    for (uint32_t i = 0; i < view.header().sites.count; ++i) {
        auto site = view.row<manifest::Site>(view.header().sites, i);
        CHECK(site.width == 8);
        CHECK(!(site.opcode & ~0x0f)); // only the defined signing bits
        if (site.opcode & manifest::SiteSigned)
            ++signedSites;
        else
            CHECK(site.reserved == 0); // a plain pointer carries no diversity
        CHECK(site.placeholder);       // every site points at its import's placeholder
    }
    CHECK(signedSites); // the destructor's import is a signed pointer
    for (const ChainedFixup &fixup : image.fixups())
        CHECK(!(fixup.bind && found.imports.count(fixup.symbol)));
    for (const Site &site : found.sites) {
        if (!site.auth)
            continue;
        auto slot = std::find_if(image.fixups().begin(), image.fixups().end(),
                                 [&](const ChainedFixup &fixup) {
                                     return fixup.address == image.base + site.offset;
                                 });
        CHECK(slot != image.fixups().end());
        CHECK(!slot->bind); // a rebase now, not a bind
        CHECK(slot->auth);  // still signed, for placement to re-sign
        CHECK(slot->auth->key == site.auth->key && slot->auth->diversity == site.auth->diversity &&
              slot->auth->addrDiv == site.auth->addrDiv);
    }
    puts("arm64e staging: pass (a version-5 manifest, signed sites keep their context)");
}

// Graft fills each inherited slot of an arm64e consumer's vtable with the live
// parent method, signed as the slot's bind to it says. A slot taken out of the
// fixup chains has no bind, so census refuses it rather than fill it unsigned.
void inheritedSlotWithoutBind(const Built &arm64e) {
    // <mach-o/fixup-chains.h>: an arm64e chain entry's next field, in 4-byte strides.
    constexpr unsigned NextShift = 51;
    constexpr uint64_t NextMask = 0x7ff;
    Image image(arm64e.unbound);
    uint64_t slot = image.symbol(vtableName("Consumer")).value + 16 + 2 * 8; // inherited(int)
    const std::vector<ChainedFixup> &fixups = image.fixups();
    auto bind = std::find_if(fixups.begin(), fixups.end(),
                             [&](const ChainedFixup &fixup) { return fixup.address == slot; });
    CHECK(bind != fixups.end() && bind != fixups.begin() && bind->bind);
    auto previous = std::prev(bind);
    CHECK(previous->address == slot - 8);
    // Link the previous entry past the bind.
    Bytes data = arm64e.unbound;
    uint64_t before = load64(&data[previous->entryOffset]);
    uint64_t skipped = load64(&data[bind->entryOffset]) >> NextShift & NextMask;
    uint64_t next = skipped ? (before >> NextShift & NextMask) + skipped : 0;
    store64(&data[previous->entryOffset], (before & ~(NextMask << NextShift)) | next << NextShift);
    Image unbound(data);
    CHECK(unbound.fixups().size() == fixups.size() - 1);
    CHECK(refuses([&] { census(unbound, arm64e.declaration, arm64e.layouts); },
                  "consumer vtable's inherited slot has no bind to its parent's method"));
    puts("arm64e inherited slot: pass (one with no bind is refused)");
}

} // namespace

int main(int argc, char **argv) {
    CHECK(argc == 4);
    Built x86(argv[1], argv[2], "Fixture"), arm64e(argv[1], argv[3], "Fixture");
    TemporaryDirectory scratch("graft-staging-test-");
    compilerLayout(x86);
    stageRemapsSymbolsAndRelocations(x86);
    weakReferencesMarkTheirImports(x86);
    providerAnchor(x86, scratch.path().string());
    stagingGatesPersonalities(x86, scratch.path().string());
    stagingLibrariesAndRefusals(x86, scratch.path().string());
    failedStagingLeavesNoOutput(x86, scratch.path().string());
    refusesShortReservationAndWrongLayout(x86);
    refusesZerofillReservation(x86);
    externalPointerToOwnCode(x86);
    truncatedMachO(x86);
    requiredMethodSubsetKeepsOverrides(x86);
    defaultRequiredSkipsReservedSlots();
    chainedFixups(arm64e, x86);
    malformedChainedFixups(arm64e);
    Census found = census(Image(arm64e.unbound), arm64e.declaration, arm64e.layouts);
    arm64eCensusMatchesX86(found, x86);
    arm64eStaging(arm64e, found);
    inheritedSlotWithoutBind(arm64e);
}
