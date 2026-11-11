// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Host test of the consumer manifest (src/Manifest.h) and everything planned
// from it: import resolution, the site plan and target check (src/Consumer.h),
// and vtable planning (src/Vtable.h). Its argument is the kext that
// `make kernel-check` links, whose manifest section it reads back.
#include "Check.h"
#include "Consumer.h"
#include "Manifest.h"
#include "Resolve.h"
#include "Vtable.h"
#include <fstream>
#include <iterator>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

using namespace graft;
using namespace graft::manifest;

// A contiguous x86-64 mapping at a fake load address: a consumer offset's
// runtime address, bounded by the image size, as the running consumer presents
// its segments. The site plan reads old bytes through imageReader instead, so a
// small load address keeps the branch arithmetic in range.
static auto contiguous(uint64_t base, size_t size) {
    return [base, size](uint64_t offset, uint64_t width) -> uint64_t {
        return width && offset < size && width <= size - offset && offset <= UINT64_MAX - base
                       ? base + offset
                       : 0;
    };
}

// A site's old bytes, and a branch's leading opcode byte, from a raw image buffer.
static auto imageReader(const uint8_t *image, size_t size) {
    return [image, size](uint64_t offset, uint8_t width, uint64_t &old, uint8_t &opcode) {
        if (offset > size || width > size - offset || (width == 4 && !offset))
            return false;
        old = 0;
        memcpy(&old, image + offset, width);
        opcode = width == 4 ? image[offset - 1] : 0;
        return true;
    };
}

struct Fixture {
    std::vector<uint8_t> bytes = std::vector<uint8_t>(sizeof(Header));
    Header h{Magic, Version, 0, sizeof(Header), {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {},
             {},    1,       0};
    uint32_t library = 0, method = 0, renamed = 0, cls = 0, vtable = 0;

    template <typename T>
    Table append(const std::vector<T> &items) {
        Table table{static_cast<uint32_t>(bytes.size()), static_cast<uint32_t>(items.size())};
        auto data = reinterpret_cast<const uint8_t *>(items.data());
        if (!items.empty())
            bytes.insert(bytes.end(), data, data + items.size() * sizeof(T));
        return table;
    }
    explicit Fixture(const char *gate = "ai.aneesiqbal.graft.test.Fixture.graft-bound") {
        std::vector<uint8_t> strings{0};
        auto string = [&](const char *s) {
            auto offset = static_cast<uint32_t>(strings.size());
            strings.insert(strings.end(), s, s + strlen(s) + 1);
            return offset;
        };
        h.gate = string(gate);
        h.matching = string("<dict><key>Fixture</key><dict><key>IOProviderClass</key>"
                            "<string>IOResources</string></dict></dict>");
        library = string("ai.aneesiqbal.graft.test.Fixture.library");
        method = string("_method");
        renamed = string("_renamed");
        cls = string("FixtureParent");
        vtable = string("__ZTV13FixtureParent");
        h.strings = append(strings);
        h.libraries = append<Library>({{library, 0}});
        h.imports = append<Import>({{0, 0, 2, 0, NoFallback}});
        h.candidates = append<uint32_t>({method, renamed});
        h.sites = append<Site>({{8, 32, 0, 0, 8, 0, 0}});
        h.classes = append<Class>({{cls, vtable, 0, 64, 3, 0, 3}});
        h.slots = append<Slot>({{0, method}, {1, method}, {2, method}});
        h.vtables = append<Vtable>({});
        h.overrides = append<uint32_t>({});
        h.segments = append<Segment>({});
        h.fills = append<Fill>({});
        h.selectors = append<uint32_t>({});
        h.synthOverrides = append<SynthOverride>({});
        h.synthClasses = append<SynthClass>({});
        h.size = static_cast<uint32_t>(bytes.size());
        save();
    }
    void save() { memcpy(bytes.data(), &h, sizeof(h)); }
    template <typename T>
    void set(Table table, uint32_t index, const T &row) {
        memcpy(bytes.data() + table.offset + index * sizeof(T), &row, sizeof(T));
    }
    Failure open(View &view) { return manifest::open(bytes.data(), bytes.size(), view); }
};

static void validation() {
    Fixture f;
    View v;
    CHECK(f.open(v).error == Error::None && v.valid());
    CHECK(!unsupportedVersion(f.bytes.data(), f.open(v)));
    CHECK(!strcmp(v.string(f.library), "ai.aneesiqbal.graft.test.Fixture.library"));
    CHECK(!v.string(0) && !v.string(UINT32_MAX));
    // Every byte-truncated blob must fail and clear a previously valid view. No
    // such failure is a version Graft names in its refusal.
    for (size_t length = 0; length < f.bytes.size(); ++length) {
        auto failure = manifest::open(f.bytes.data(), length, v);
        CHECK(failure.error != Error::None && !v.valid());
        CHECK(!unsupportedVersion(f.bytes.data(), failure));
    }
    // Only the current versions are read: the earlier ones, which named a device
    // rather than carrying provider matching, and any later one are refused, and
    // the refusal names the version, so its consumer's author knows to rebuild.
    for (uint32_t version : {2u, 3u, VersionArm64e + 1}) {
        f.h.version = version;
        f.save();
        auto failure = f.open(v);
        CHECK(failure.error == Error::Version);
        CHECK(unsupportedVersion(f.bytes.data(), failure) == version);
    }
    f.h.magic = Magic + 1; // another version under a wrong magic is not a manifest
    f.save();
    CHECK(!unsupportedVersion(f.bytes.data(), f.open(v)));
    f.h.magic = Magic;
    // A version-5 (arm64e) manifest: the base site is a valid plain pointer,
    // and an authenticated site carries key, addrDiv and diversity.
    f.h.version = VersionArm64e;
    f.save();
    CHECK(f.open(v).error == Error::None);
    f.set<Site>(f.h.sites, 0, {8, 32, 0, 0, 8, SiteSigned | SiteAddrDiv | (1 << 1), 0x1234});
    CHECK(f.open(v).error == Error::None);                // signed, key IB, addrDiv, diversity
    f.set<Site>(f.h.sites, 0, {8, 32, 0, 0, 8, 0x10, 0}); // auth byte bit outside the defined set
    CHECK(f.open(v).error == Error::Site);
    f.set<Site>(f.h.sites, 0, {8, 32, 0, 0, 8, 0, 0x55}); // plain pointer may carry no diversity
    CHECK(f.open(v).error == Error::Site);
    f.set<Site>(f.h.sites, 0, {8, 32, 0, 0, 4, 0, 0}); // every arm64e site is an 8-byte pointer
    CHECK(f.open(v).error == Error::Site);
    // A gate key Graft can copy whole is accepted, one byte longer is refused.
    std::string gate(MaxGateLength, 'g');
    CHECK(Fixture(gate.c_str()).open(v).error == Error::None);
    gate += 'g';
    CHECK(Fixture(gate.c_str()).open(v).error == Error::String);
    // The provider matching is a string the table holds.
    f = Fixture{};
    f.h.matching = 0;
    f.save();
    CHECK(f.open(v).error == Error::String);
    f = Fixture{};
    f.h.candidates.offset = f.h.imports.offset;
    f.save();
    CHECK(f.open(v).error == Error::Overlap);
    f = Fixture{};
    f.h.sites.count = UINT32_MAX;
    f.save();
    CHECK(f.open(v).error == Error::Range);
    f = Fixture{};
    f.set<Import>(f.h.imports, 0, {0, UINT32_MAX, 2, 0, NoFallback});
    CHECK(f.open(v).error == Error::Import);
    f = Fixture{};
    f.set<Library>(f.h.libraries, 0, {f.library, UINT32_MAX});
    CHECK(f.open(v).error == Error::Library);
    f.set<Library>(f.h.libraries, 0, {f.library, f.cls});
    CHECK(f.open(v).error == Error::None);
    f = Fixture{};
    f.set<Import>(f.h.imports, 0, {0, 0, 2, ImportWeak, NoFallback});
    CHECK(f.open(v).error == Error::None);
    f.set<Import>(f.h.imports, 0, {0, 0, 2, ImportWeak << 1, NoFallback});
    CHECK(f.open(v).error == Error::Import);
    f = Fixture{};
    f.set<Site>(f.h.sites, 0, {0, 32, 0, 0, 4, 0xe8, 0});
    CHECK(f.open(v).error == Error::Site);
    f = Fixture{};
    f.set<Slot>(f.h.slots, 0, {3, f.method});
    CHECK(f.open(v).error == Error::Slot);
    f = Fixture{};
    f.set<Class>(f.h.classes, 0, {f.cls, f.vtable, 0, 64, 3, 1, 1});
    CHECK(f.open(v).error == Error::Class);
    f = Fixture{};
    f.bytes[f.h.strings.offset + f.h.strings.count - 1] = 'x';
    CHECK(f.open(v).error == Error::Class);
    // All accessors copy records: a section need not be naturally aligned.
    f = Fixture{};
    std::vector<uint8_t> unaligned(1, 0);
    unaligned.insert(unaligned.end(), f.bytes.begin(), f.bytes.end());
    CHECK(manifest::open(unaligned.data() + 1, f.bytes.size(), v).error == Error::None);
    puts("manifest: pass (bounds, names, references, overlap, truncation, alignment)");
}

static void resolution() {
    using binding::Resolution;
    Fixture f;
    View v;
    CHECK(f.open(v).error == Error::None);
    uint32_t attempts = 0;
    auto find = [&](uint32_t library, const char *name, uint64_t &address) {
        CHECK(library == 0);
        ++attempts;
        if (!strcmp(name, "_renamed")) {
            address = 0x9000;
            return Resolution::Found;
        }
        return Resolution::Missing;
    };
    uint64_t address = 7;
    CHECK(binding::resolveImport(v, 0, find, 0, contiguous(0x4000, 64), address) ==
          Resolution::Found);
    CHECK(address == 0x9000 && attempts == 2);
    auto missing = [](uint32_t, const char *, uint64_t &) { return Resolution::Missing; };
    CHECK(binding::resolveImport(v, 0, missing, 0, contiguous(0x4000, 64), address) ==
                  Resolution::Missing &&
          !address);
    f.set<Import>(f.h.imports, 0, {0, 0, 2, 0, 48});
    CHECK(f.open(v).error == Error::None);
    CHECK(binding::resolveImport(v, 0, missing, 0, contiguous(0x4000, 64), address) ==
                  Resolution::Found &&
          address == 0x4030);
    auto ambiguous = [](uint32_t, const char *, uint64_t &) { return Resolution::Ambiguous; };
    CHECK(binding::resolveImport(v, 0, ambiguous, 0, contiguous(0x4000, 64), address) ==
                  Resolution::Ambiguous &&
          !address);
    CHECK(binding::resolveImport(v, 0, missing, 0, contiguous(UINT64_MAX - 10, 64), address) ==
          Resolution::Invalid);
    CHECK(binding::resolveImport(v, 0, missing, 0, contiguous(0x4000, 32), address) ==
          Resolution::Invalid);
    CHECK(binding::resolveImport(v, 1, find, 0, contiguous(0x4000, 64), address) ==
          Resolution::Invalid);

    // Parse consumer metadata, resolve its ordered candidates and preflight
    // an actual image site without changing the image.
    uint8_t image[64]{};
    uint64_t placeholder = 0x4020;
    memcpy(image + 8, &placeholder, 8);
    auto map = contiguous(0x4000, sizeof(image));
    auto read = imageReader(image, sizeof(image));
    auto sign = [&](uint64_t offset) { return map(offset, 1); };
    auto record = v.row<Site>(v.header().sites, 0);
    binding::Site site{record.offset, map(record.placeholder, 1), record.addend, record.width,
                       record.opcode};
    binding::Write write{};
    auto resolve = [&](const binding::Site &, uint64_t &value) {
        return binding::resolveImport(v, record.import, find, 0, map, value);
    };
    CHECK(binding::plan(&site, 1, &write, 1, resolve, read, sign).error == binding::Error::None);
    CHECK(write.before == placeholder && write.after == 0x9000);
    uint64_t unchanged;
    memcpy(&unchanged, image + 8, 8);
    CHECK(unchanged == placeholder);

    const char providerNames[] = "\0_renamed\0";
    nlist_64 definition{};
    definition.n_type = N_SECT;
    definition.n_sect = 1;
    definition.n_un.n_strx = 1;
    definition.n_value = 0x9000;
    Provider provider{{&definition, 1, providerNames, sizeof(providerNames)}};
    binding::Site scratch{};
    CHECK(planConsumerSites(v, &provider, 1, 0, map, read, &scratch, &write, 1).error ==
          binding::Error::None);
    CHECK(write.before == placeholder && write.after == 0x9000);
    // A failure that concerns no one site names none.
    auto metadata = planConsumerSites(v, nullptr, 0, 0, map, read, &scratch, &write, 1);
    CHECK(metadata.error == binding::Error::Metadata && metadata.site == binding::NoSite);
    auto capacity = planConsumerSites(v, &provider, 1, 0, map, read, &scratch, &write, 0);
    CHECK(capacity.error == binding::Error::Capacity && capacity.site == binding::NoSite);
    puts("resolution: pass (provider scope, candidates, fallback, ambiguity, site plan)");
}

// A weak import binds to the pointer the kernel's gOSKextUnresolved holds
// where nothing defines it, as the kernel's linker binds one; a defined one
// binds as any other, and a strong miss, or a weak one with no
// gOSKextUnresolved, stays an error.
static void weak() {
    using binding::Resolution;
    // A fake kernel: gOSKextUnresolved lives at 0xb000 and holds 0xa000.
    // OSKextSymbolIsResolved() compares against the value, so 0xa000 is the
    // sentinel and 0xb000, the variable's own address, is not.
    const uint64_t variable = 0xb000;
    auto read = [&](uint64_t address, uint64_t &value) {
        if (address != variable)
            return false;
        value = 0xa000;
        return true;
    };
    const uint64_t unresolved = binding::unresolvedValue(variable, read);
    CHECK(unresolved == 0xa000);
    CHECK(!binding::unresolvedValue(0, read) && !binding::unresolvedValue(variable + 8, read));
    Fixture f;
    View v;
    f.set<Import>(f.h.imports, 0, {0, 0, 2, ImportWeak, NoFallback});
    CHECK(f.open(v).error == Error::None);
    auto missing = [](uint32_t, const char *, uint64_t &) { return Resolution::Missing; };
    auto found = [](uint32_t, const char *name, uint64_t &address) {
        address = 0x9000;
        return strcmp(name, "_renamed") ? Resolution::Missing : Resolution::Found;
    };
    uint64_t address = 7;
    CHECK(binding::resolveImport(v, 0, missing, unresolved, contiguous(0x4000, 64), address) ==
                  Resolution::Found &&
          address == unresolved);
    CHECK(binding::resolveImport(v, 0, found, unresolved, contiguous(0x4000, 64), address) ==
                  Resolution::Found &&
          address == 0x9000);
    CHECK(binding::resolveImport(v, 0, missing, 0, contiguous(0x4000, 64), address) ==
                  Resolution::Missing &&
          !address);
    f.set<Import>(f.h.imports, 0, {0, 0, 2, ImportWeak, 48});
    CHECK(f.open(v).error == Error::None);
    CHECK(binding::resolveImport(v, 0, missing, unresolved, contiguous(0x4000, 64), address) ==
                  Resolution::Found &&
          address == 0x4030);
    Fixture strong;
    CHECK(strong.open(v).error == Error::None);
    CHECK(binding::resolveImport(v, 0, missing, unresolved, contiguous(0x4000, 64), address) ==
                  Resolution::Missing &&
          !address);

    // Two weak imports in one consumer: the provider defines _renamed and not
    // _method. The planner binds the first to its definition and the second
    // to gOSKextUnresolved; made strong, the second refuses the consumer.
    Fixture two;
    two.h.imports = two.append<Import>(
            {{0, 0, 1, ImportWeak, NoFallback}, {0, 1, 1, ImportWeak, NoFallback}});
    two.h.candidates = two.append<uint32_t>({two.renamed, two.method});
    two.h.sites = two.append<Site>({{8, 32, 0, 0, 8, 0, 0}, {16, 40, 0, 1, 8, 0, 0}});
    two.h.size = static_cast<uint32_t>(two.bytes.size());
    two.save();
    CHECK(two.open(v).error == Error::None);
    uint8_t image[64]{};
    uint64_t placeholders[] = {0x4020, 0x4028};
    memcpy(image + 8, placeholders, sizeof(placeholders));
    auto map = contiguous(0x4000, sizeof(image));
    auto imageRead = imageReader(image, sizeof(image));
    const char providerNames[] = "\0_renamed\0";
    nlist_64 definition{};
    definition.n_type = N_SECT;
    definition.n_sect = 1;
    definition.n_un.n_strx = 1;
    definition.n_value = 0x9000;
    Provider provider{{&definition, 1, providerNames, sizeof(providerNames)}};
    binding::Site scratch[2]{};
    binding::Write writes[2]{};
    auto result =
            planConsumerSites(v, &provider, 1, unresolved, map, imageRead, scratch, writes, 2);
    CHECK(result.error == binding::Error::None);
    CHECK(writes[0].before == 0x4020 && writes[0].after == 0x9000);
    CHECK(writes[1].before == 0x4028 && writes[1].after == unresolved);
    // The target check after the plan, as the kernel runs it: the provider
    // maps 0x9000 to 0x9100, the consumer 0x4000 to 0x4040, and
    // gOSKextUnresolved lies in neither.
    auto inProvider = [](uint32_t library, uint64_t address, bool) {
        return library == 0 && address >= 0x9000 && address < 0x9100;
    };
    auto inConsumer = [](uint64_t address, bool) { return address >= 0x4000 && address < 0x4040; };
    CHECK(checkTargets(v, map, unresolved, writes, inProvider, inConsumer).error ==
          binding::Error::None);
    // The sentinel is accepted for that import and nothing else: not with
    // no gOSKextUnresolved, not for a strong import, not for a weak one
    // with a fallback, and no other address outside the provider.
    auto refused = checkTargets(v, map, 0, writes, inProvider, inConsumer);
    CHECK(refused.error == binding::Error::Target && refused.site == 1);
    writes[1].after = variable;
    refused = checkTargets(v, map, unresolved, writes, inProvider, inConsumer);
    CHECK(refused.error == binding::Error::Target && refused.site == 1);
    writes[1].after = unresolved + 8;
    refused = checkTargets(v, map, unresolved, writes, inProvider, inConsumer);
    CHECK(refused.error == binding::Error::Target && refused.site == 1);
    writes[1].after = unresolved;
    two.set<Import>(two.h.imports, 1, {0, 1, 1, ImportWeak, 48});
    CHECK(two.open(v).error == Error::None);
    refused = checkTargets(v, map, unresolved, writes, inProvider, inConsumer);
    CHECK(refused.error == binding::Error::Target && refused.site == 1);
    result = planConsumerSites(v, &provider, 1, unresolved, map, imageRead, scratch, writes, 2);
    CHECK(result.error == binding::Error::None && writes[1].after == 0x4030);
    CHECK(checkTargets(v, map, unresolved, writes, inProvider, inConsumer).error ==
          binding::Error::None);
    two.set<Import>(two.h.imports, 1, {0, 1, 1, 0, NoFallback});
    CHECK(two.open(v).error == Error::None);
    result = planConsumerSites(v, &provider, 1, unresolved, map, imageRead, scratch, writes, 2);
    CHECK(result.error == binding::Error::Missing && result.site == 1);
    writes[1] = {16, 0x4028, unresolved, 8};
    refused = checkTargets(v, map, unresolved, writes, inProvider, inConsumer);
    CHECK(refused.error == binding::Error::Target && refused.site == 1);

    // A strong import with no site, as an inherited vtable slot's method is: the
    // vtable fill writes that slot, so Graft never resolves the import, and a
    // release that no longer defines it does not refuse the consumer.
    Fixture siteless;
    siteless.h.imports =
            siteless.append<Import>({{0, 0, 1, 0, NoFallback}, {0, 1, 1, 0, NoFallback}});
    siteless.h.candidates = siteless.append<uint32_t>({siteless.renamed, siteless.method});
    siteless.h.sites = siteless.append<Site>({{8, 32, 0, 0, 8, 0, 0}});
    siteless.h.size = static_cast<uint32_t>(siteless.bytes.size());
    siteless.save();
    CHECK(siteless.open(v).error == Error::None);
    result = planConsumerSites(v, &provider, 1, unresolved, map, imageRead, scratch, writes, 2);
    CHECK(result.error == binding::Error::None && writes[0].after == 0x9000);

    // A weak call nothing defines: the branch's displacement reaches
    // gOSKextUnresolved, and the target check accepts it as it does a pointer.
    Fixture call;
    call.set<Import>(call.h.imports, 0, {0, 0, 2, ImportWeak, NoFallback});
    call.set<Site>(call.h.sites, 0, {9, 32, 0, 0, 4, 0xe8, 0});
    CHECK(call.open(v).error == Error::None);
    uint8_t code[64]{};
    code[8] = 0xe8;
    int32_t toPlaceholder = 32 - (9 + 4);
    memcpy(code + 9, &toPlaceholder, 4);
    const char otherNames[] = "\0_other\0";
    nlist_64 other = definition;
    Provider none{{&other, 1, otherNames, sizeof(otherNames)}};
    binding::Site branch{};
    binding::Write write{};
    auto codeMap = contiguous(0x4000, sizeof(code));
    auto codeRead = imageReader(code, sizeof(code));
    result = planConsumerSites(v, &none, 1, unresolved, codeMap, codeRead, &branch, &write, 1);
    CHECK(result.error == binding::Error::None);
    CHECK(static_cast<int32_t>(write.after) == static_cast<int64_t>(unresolved) - (0x4000 + 9 + 4));
    CHECK(checkTargets(v, codeMap, unresolved, &write, inProvider, inConsumer).error ==
          binding::Error::None);
    call.set<Import>(call.h.imports, 0, {0, 0, 2, 0, NoFallback});
    CHECK(call.open(v).error == Error::None);
    refused = checkTargets(v, codeMap, unresolved, &write, inProvider, inConsumer);
    CHECK(refused.error == binding::Error::Target && refused.site == 0);
    puts("weak: pass (absent weak to gOSKextUnresolved through the plan and the target check, "
         "pointer and branch, present weak, fallback, strong miss, no sentinel)");
}

// The class that verifies a provider: its anchor where it has one, else the
// first class the consumer declares from it; none where it has neither.
static void anchors() {
    Fixture f;
    View v;
    CHECK(f.open(v).error == Error::None);
    CHECK(!strcmp(libraryAnchor(v, 0), "FixtureParent"));
    CHECK(!libraryAnchor(v, 1));
    f.set<Library>(f.h.libraries, 0, {f.library, f.renamed});
    CHECK(f.open(v).error == Error::None);
    CHECK(!strcmp(libraryAnchor(v, 0), "_renamed"));
    // A provider the consumer takes only imports from.
    Fixture imports;
    imports.h.classes = imports.append<Class>({});
    imports.h.slots = imports.append<Slot>({});
    imports.h.size = static_cast<uint32_t>(imports.bytes.size());
    imports.save();
    CHECK(imports.open(v).error == Error::None);
    CHECK(!libraryAnchor(v, 0));
    imports.set<Library>(imports.h.libraries, 0, {imports.library, imports.cls});
    CHECK(imports.open(v).error == Error::None);
    CHECK(!strcmp(libraryAnchor(v, 0), "FixtureParent"));
    puts("anchors: pass (declared class, anchor first, anchor alone, neither)");
}

static void duplicates() {
    using binding::Resolution;
    const char names[] = "\0_method\0";
    nlist_64 list[2]{};
    for (auto &symbol : list) {
        symbol.n_type = N_SECT;
        symbol.n_sect = 1;
        symbol.n_un.n_strx = 1;
        symbol.n_value = 0x1000;
    }
    macho::SymbolTable table{list, 2, names, sizeof(names)};
    uint64_t address = 0;
    CHECK(binding::unique(table, "_method", address) == Resolution::Found && address == 0x1000);
    list[1].n_value = 0x2000;
    CHECK(binding::unique(table, "_method", address) == Resolution::Ambiguous && !address);
    CHECK(binding::unique(table, "_absent", address) == Resolution::Missing && !address);
}

static void vtables() {
    Fixture f;
    f.h.vtables = f.append<Vtable>({{32, 0, 4, 0, 1, 0, 0}});
    f.h.overrides = f.append<uint32_t>({1});
    f.h.size = static_cast<uint32_t>(f.bytes.size());
    f.save();
    View view;
    CHECK(f.open(view).error == Error::None);
    uint64_t parent[] = {0x9000, 0x9010, 0x9020};
    ActualClass actual{64, 3, reinterpret_cast<const uint8_t *>(parent)};
    uint64_t consumer[] = {0, 0, 0, 0, 0x4100, 0x4110, 0x4120, 0x4130};
    binding::Write writes[3]{};
    size_t written = 999;
    auto read = [&](uint64_t offset, uint64_t &value) {
        if (offset % 8 || offset / 8 >= 8)
            return false;
        value = consumer[offset / 8];
        return true;
    };
    auto code = [](uint64_t address) { return address >= 0x4000 && address < 0x5000; };
    auto defines = [](uint32_t provider, const char *name, uint64_t address) {
        return provider == 0 && !strcmp(name, "_method") && address >= 0x9000 && address <= 0x9020;
    };
    // A slot's runtime address; x86-64 copies the provider slot verbatim and
    // ignores it, so any nonzero stand-in serves (arm64e exercises re-signing
    // in tests/arm64e_fixup_test.cpp).
    auto map = [](uint64_t offset) -> uint64_t { return offset + 0x8000; };
    auto run = [&](size_t capacity = 3) {
        return planVtables(view, &actual, 1, writes, capacity, written, read, map, code, defines);
    };
    CHECK(run().error == VtableError::None && written == 2);
    CHECK(writes[0].offset == 32 && writes[0].before == 0x4100 && writes[0].after == 0x9000);
    CHECK(writes[1].offset == 48 && writes[1].before == 0x4120 && writes[1].after == 0x9020);
    CHECK(consumer[4] == 0x4100 && consumer[5] == 0x4110 && consumer[7] == 0x4130);
    CHECK(run(1).error == VtableError::Capacity && written == 0);
    actual.instanceSize = 72;
    CHECK(run().detail == layout::Error::Size && written == 0);
    actual.instanceSize = 64;
    actual.slotCount = 4;
    CHECK(run().detail == layout::Error::Length);
    actual.slotCount = 3;
    parent[2] = 0;
    CHECK(run().error == VtableError::Method);
    parent[2] = 0x9020;
    consumer[5] = 0x9000;
    CHECK(run().error == VtableError::Override);
    consumer[5] = 0x4110;
    consumer[7] = 0;
    CHECK(run().error == VtableError::Override);
    f.set<Vtable>(f.h.vtables, 0, {8, 0, 4, 0, 1, 0, 0});
    CHECK(f.open(view).error == Error::Vtable);
    puts("vtables: pass (live inherited slots, preserved overrides/appended methods, refusal)");
}

// A version-5 (arm64e) manifest's segment table and vtable fills: the segments
// are sorted, bounded, disjoint and named; the fills partition [0, parentSlots)
// with the overrides, in slot order, each a valid signing context. x86-64
// carries neither. The real staged forms are cross-checked at build time; this
// exercises the refusals the kernel must make on a malformed one.
static void segmentsAndFills() {
    auto build = [](Fixture &f) {
        f = Fixture{};
        f.h.version = VersionArm64e;
        f.h.segments = f.append<Segment>({{0, 0x1000, f.cls, 0}, {0x1000, 0x1000, f.vtable, 0}});
        f.h.vtables = f.append<Vtable>({{32, 0, 4, 0, 1, 0, 2}});
        f.h.overrides = f.append<uint32_t>({1});
        f.h.fills = f.append<Fill>({{0, 0, 0, 0}, {2, 0x1234, SiteSigned | SiteAddrDiv, 0}});
        f.h.size = static_cast<uint32_t>(f.bytes.size());
        f.save();
    };
    Fixture f;
    View v;
    build(f);
    CHECK(f.open(v).error == Error::None);
    build(f);
    f.set<Segment>(f.h.segments, 1, {0, 0x1000, f.vtable, 0}); // not after segment 0
    CHECK(f.open(v).error == Error::Segment);
    build(f);
    f.set<Segment>(f.h.segments, 1, {0x800, 0x1000, f.vtable, 0}); // overlaps segment 0
    CHECK(f.open(v).error == Error::Segment);
    build(f);
    f.set<Segment>(f.h.segments, 0, {0, 0, f.cls, 0}); // zero size
    CHECK(f.open(v).error == Error::Segment);
    build(f);
    f.set<Segment>(f.h.segments, 0, {0, 0x1000, 0, 0}); // no name
    CHECK(f.open(v).error == Error::Segment);
    build(f);
    f.set<Segment>(f.h.segments, 0, {0, 0x1000, f.cls, 1}); // reserved set
    CHECK(f.open(v).error == Error::Segment);
    build(f);
    f.set<Fill>(f.h.fills, 0, {2, 0, 0, 0}); // first fill is not slot 0
    CHECK(f.open(v).error == Error::Fill);
    build(f);
    f.set<Vtable>(f.h.vtables, 0, {32, 0, 4, 0, 1, 0, 1}); // fillCount != parent - overrides
    CHECK(f.open(v).error == Error::Fill);
    build(f);
    f.set<Fill>(f.h.fills, 0, {0, 0x55, 0, 0}); // a plain fill carries no diversity
    CHECK(f.open(v).error == Error::Fill);
    build(f);
    f.set<Fill>(f.h.fills, 1, {2, 0, 0x10, 0}); // auth byte outside the defined set
    CHECK(f.open(v).error == Error::Fill);
    build(f);
    f.h.version = Version; // x86-64 carries no segment or fill table
    f.save();
    CHECK(f.open(v).error == Error::Version);
    CHECK(!unsupportedVersion(f.bytes.data(), f.open(v))); // a version Graft reads
    puts("segments and fills: pass (sorted/bounded segments, fill partition, version gate)");
}

// A version-5 (arm64e) manifest's synthesis tables: selector strings, overrides
// that partition the selectors in order with a valid signing context, and classes
// that partition the overrides. x86-64 carries none. The real staged form is
// cross-checked at build time; this exercises the refusals open() must make on a
// malformed one.
static void synthesis() {
    // Two overrides (the second signed + required), two selectors they partition,
    // one class over both. Strings are reused as class/selector names -- open()
    // checks they are valid offsets, not their content.
    auto build = [](Fixture &f) {
        f = Fixture{};
        f.h.version = VersionArm64e;
        f.h.selectors = f.append<uint32_t>({f.method, f.renamed});
        f.h.synthOverrides = f.append<SynthOverride>(
                {{64, 0, 1, 0, 0, 0, 0},
                 {72, 1, 1, 0x1234, SiteSigned | SiteAddrDiv, SynthRequired, 0}});
        f.h.synthClasses =
                f.append<SynthClass>({{16, f.cls, f.renamed, f.method, 0, 600, 0x3000, 0, 2}});
        f.h.size = static_cast<uint32_t>(f.bytes.size());
        f.save();
    };
    Fixture f;
    View v;
    build(f);
    CHECK(f.open(v).error == Error::None);
    build(f);
    f.set<uint32_t>(f.h.selectors, 0, UINT32_MAX); // a selector names no string
    CHECK(f.open(v).error == Error::Selector);
    build(f);
    f.set<SynthOverride>(f.h.synthOverrides, 0, {64, 0, 0, 0, 0, 0, 0}); // no selector
    CHECK(f.open(v).error == Error::SynthOverride);
    build(f);
    f.set<SynthOverride>(f.h.synthOverrides, 0, {64, 1, 1, 0, 0, 0, 0}); // firstSelector not 0
    CHECK(f.open(v).error == Error::SynthOverride);
    build(f);
    f.set<SynthOverride>(f.h.synthOverrides, 0, {0, 0, 1, 0, 0, 0, 0}); // no implementation
    CHECK(f.open(v).error == Error::SynthOverride);
    build(f);
    f.set<SynthOverride>(f.h.synthOverrides, 0, {64, 0, 1, 0, 0, 0, 1}); // reserved set
    CHECK(f.open(v).error == Error::SynthOverride);
    build(f);
    f.set<SynthOverride>(f.h.synthOverrides, 0, {64, 0, 1, 0, 0, 2, 0}); // flag outside the set
    CHECK(f.open(v).error == Error::SynthOverride);
    build(f);
    f.set<SynthOverride>(f.h.synthOverrides, 0, {64, 0, 1, 0, 0x10, 0, 0}); // auth byte outside set
    CHECK(f.open(v).error == Error::SynthOverride);
    build(f);
    f.set<SynthOverride>(f.h.synthOverrides, 0, {64, 0, 1, 0x55, 0, 0, 0}); // plain + diversity
    CHECK(f.open(v).error == Error::SynthOverride);
    build(f);
    f.set<SynthClass>(f.h.synthClasses, 0,
                      {16, UINT32_MAX, f.renamed, f.method, 0, 600, 0x3000, 0, 2});
    CHECK(f.open(v).error == Error::SynthClass); // name names no string
    build(f);
    f.set<SynthClass>(f.h.synthClasses, 0,
                      {16, f.cls, f.renamed, UINT32_MAX, 0, 600, 0x3000, 0, 2});
    CHECK(f.open(v).error == Error::SynthClass); // sibling names no string
    build(f);
    f.set<SynthClass>(f.h.synthClasses, 0, {16, f.cls, f.renamed, 0, 0, 600, 0x3000, 0, 2});
    CHECK(f.open(v).error == Error::None); // a zero sibling is allowed
    build(f);
    f.set<SynthClass>(f.h.synthClasses, 0,
                      {16, f.cls, f.renamed, f.method, UINT32_MAX, 600, 0x3000, 0, 2});
    CHECK(f.open(v).error == Error::SynthClass); // library out of range
    build(f);
    f.set<SynthClass>(f.h.synthClasses, 0, {8, f.cls, f.renamed, f.method, 0, 600, 0x3000, 0, 2});
    CHECK(f.open(v).error == Error::SynthClass); // vtable offset below the two-word header
    build(f);
    f.set<SynthClass>(f.h.synthClasses, 0, {20, f.cls, f.renamed, f.method, 0, 600, 0x3000, 0, 2});
    CHECK(f.open(v).error == Error::SynthClass); // vtable offset not 8-aligned
    build(f);
    f.set<SynthClass>(f.h.synthClasses, 0, {16, f.cls, f.renamed, f.method, 0, 0, 0x2088, 0, 2});
    CHECK(f.open(v).error == Error::SynthClass); // zero slot cap
    build(f);
    f.set<SynthClass>(f.h.synthClasses, 0, {16, f.cls, f.renamed, f.method, 0, 600, 0, 0, 2});
    CHECK(f.open(v).error == Error::SynthClass); // zero parent region
    build(f);
    f.set<SynthClass>(f.h.synthClasses, 0, {16, f.cls, f.renamed, f.method, 0, 600, 0x3000, 1, 2});
    CHECK(f.open(v).error == Error::SynthClass); // firstOverride not a partition
    build(f);
    f.set<SynthClass>(f.h.synthClasses, 0, {16, f.cls, f.renamed, f.method, 0, 600, 0x3000, 0, 1});
    CHECK(f.open(v).error == Error::SynthClass); // leaves an override unclaimed
    build(f);
    f.h.version = Version; // x86-64 carries no synthesis tables
    f.save();
    CHECK(f.open(v).error == Error::Version);
    puts("synthesis: pass (selectors, override partition + signing, class bounds + partition, "
         "version gate)");
}

// A synthesized class's parent is measured at bind, not layout-checked, so a
// regular Vtable record of that same class would make vtable planning copy the
// declared parent-slot count of words from a live vtable that may be shorter.
// open() must refuse such a manifest from the manifest alone.
static void synthParentVtable() {
    auto build = [](Fixture &f) {
        f = Fixture{};
        f.h.version = VersionArm64e;
        // A synth-parent class declares no slots; its parentSlots is a stand-in cap.
        f.h.classes = f.append<Class>({{f.cls, f.vtable, 0, 64, 4, 0, 0}});
        f.h.slots = f.append<Slot>({});
        f.h.vtables = f.append<Vtable>({{32, 0, 4, 0, 0, 0, 4}});
        f.h.fills = f.append<Fill>({{0, 0, 0, 0}, {1, 0, 0, 0}, {2, 0, 0, 0}, {3, 0, 0, 0}});
        f.h.selectors = f.append<uint32_t>({f.method});
        f.h.synthOverrides = f.append<SynthOverride>({{64, 0, 1, 0, 0, 0, 0}});
        f.h.synthClasses = f.append<SynthClass>({{16, f.renamed, f.cls, 0, 0, 600, 0x3000, 0, 1}});
        f.h.size = static_cast<uint32_t>(f.bytes.size());
        f.save();
    };
    Fixture f;
    View v;
    build(f);
    // The Vtable's class (FixtureParent) is the synthesized class's parent.
    CHECK(f.open(v).error == Error::Vtable);
    // Point the synthesized class at a different parent: the Vtable is now allowed.
    build(f);
    f.set<SynthClass>(f.h.synthClasses, 0, {16, f.renamed, f.method, 0, 0, 600, 0x3000, 0, 1});
    CHECK(f.open(v).error == Error::None);
    puts("synth-parent vtable: pass (a Vtable record of a synthesized parent is refused)");
}

static void linkedImage(const char *path) {
    std::ifstream input(path, std::ios::binary);
    CHECK(input);
    std::vector<uint8_t> image{std::istreambuf_iterator<char>(input),
                               std::istreambuf_iterator<char>()};
    CHECK(image.size() >= sizeof(mach_header_64));
    auto mh = macho::header(image.data());
    CHECK(mh && mh->sizeofcmds <= image.size() - sizeof(*mh) && macho::commandsValid(mh) &&
          mh->filetype == MH_KEXT_BUNDLE);
    auto section = macho::section(mh, "__DATA", "__graft");
    CHECK(section && section->offset <= image.size() &&
          section->size <= image.size() - section->offset);
    View view;
    CHECK(manifest::open(image.data() + section->offset, section->size, view).error == Error::None);
    CHECK(!strcmp(view.string(view.header().gate), "graft-bound"));
    CHECK(!strcmp(view.string(view.header().matching), "<dict/>"));
    puts("kernel fixture: pass (x86-64 kext link, embedded manifest section)");
}

int main(int argc, char **argv) {
    validation();
    resolution();
    weak();
    anchors();
    duplicates();
    vtables();
    segmentsAndFills();
    synthesis();
    synthParentVtable();
    CHECK(argc == 2);
    linkedImage(argv[1]);
}
