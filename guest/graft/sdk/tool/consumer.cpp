// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
#include "consumer.h"

#include "json.h"
#include "manifest.h"
#include "policy.h"
#include "process.h"

#include <algorithm>
#include <filesystem>
#include <iostream>

namespace graft::tool {

namespace fs = std::filesystem;

namespace {

// A vtable's slots start after its offset-to-top and type-information words, and
// under -fapple-kext the compiler ends it with one zero word.
constexpr uint64_t PointerSize = 8;
constexpr uint64_t VtableAddressPoint = 2 * PointerSize;

// Each import gets a 16-byte-aligned placeholder of its own, 32 bytes apart, so a
// vtable import, whose sites add the 16-byte address point, still lands inside
// its own placeholder.
constexpr uint64_t PlaceholderAlignment = 16;
constexpr uint64_t PlaceholderSize = 32;

const std::string ProbeClass = "GraftLayoutProbe";

// The suffix that forms a consumer's activation property key from its bundle
// identifier. Graft sets the key on the device nub once it has bound the
// consumer; the staged personalities carry it as an IOPropertyMatch, so none of
// them matches, or is even constructed, until then.
const std::string GateSuffix = ".graft-bound";

// --- graft.json -----------------------------------------------------------------

// name, if it is a top-level C++ class name the probe can derive from.
const std::string &identifier(const std::string &name) {
    bool valid = !name.empty() && !(name[0] >= '0' && name[0] <= '9');
    for (char c : name) {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
              c == '_'))
            valid = false;
    }
    if (!valid)
        throw Error("only top-level C++ class names are supported");
    return name;
}

// graft.json's values by type. A value that is absent, or null, reads as
// absent; one of another type is refused, named by its place in the declaration.
[[noreturn]] void malformed(const std::string &where, const char *what) {
    throw Error("declaration " + where + " must be " + what);
}

bool absent(const Json *value) {
    return !value || value->kind == Json::Kind::Null;
}

const std::vector<Json> &list(const Json *value, const std::string &where) {
    static const std::vector<Json> none;
    if (absent(value))
        return none;
    if (value->kind != Json::Kind::Array)
        malformed(where, "a list");
    return value->array;
}

const Json &object(const Json &value, const std::string &where) {
    if (value.kind != Json::Kind::Object)
        malformed(where, "an object");
    return value;
}

std::string text(const Json *value, const std::string &where) {
    if (absent(value))
        return "";
    if (value->kind != Json::Kind::String)
        malformed(where, "a string");
    return value->string;
}

std::optional<std::string> optionalText(const Json *value, const std::string &where) {
    if (absent(value))
        return std::nullopt;
    return text(value, where);
}

int64_t integer(const Json *value, const std::string &where) {
    if (!value || value->kind != Json::Kind::Integer)
        malformed(where, "an integer");
    return value->integer;
}

std::vector<std::string> names(const Json *value, const std::string &where) {
    std::vector<std::string> out;
    const std::vector<Json> &items = list(value, where);
    for (size_t i = 0; i < items.size(); ++i)
        out.push_back(text(&items[i], where + "[" + std::to_string(i) + "]"));
    return out;
}

std::optional<std::vector<std::string>> optionalNames(const Json *value, const std::string &where) {
    if (absent(value))
        return std::nullopt;
    return names(value, where);
}

Declaration parseDeclaration(const Json &config) {
    object(config, "root");
    Declaration out;
    const std::vector<Json> &libraries = list(config.find("libraries"), "libraries");
    for (size_t i = 0; i < libraries.size(); ++i) {
        std::string where = "libraries[" + std::to_string(i) + "]";
        const Json &library = object(libraries[i], where);
        out.libraries.push_back({text(library.find("id"), where + ".id"),
                                 optionalText(library.find("anchor"), where + ".anchor"),
                                 names(library.find("classes"), where + ".classes")});
    }
    const std::vector<Json> &classes = list(config.find("classes"), "classes");
    for (size_t i = 0; i < classes.size(); ++i) {
        std::string where = "classes[" + std::to_string(i) + "]";
        const Json &cls = object(classes[i], where);
        out.classes.push_back({text(cls.find("name"), where + ".name"),
                               text(cls.find("header"), where + ".header"),
                               integer(cls.find("library"), where + ".library"),
                               optionalNames(cls.find("required"), where + ".required")});
    }
    const std::vector<Json> &consumers = list(config.find("consumers"), "consumers");
    for (size_t i = 0; i < consumers.size(); ++i) {
        std::string where = "consumers[" + std::to_string(i) + "]";
        const Json &consumer = object(consumers[i], where);
        out.consumers.push_back({text(consumer.find("name"), where + ".name"),
                                 text(consumer.find("header"), where + ".header"),
                                 integer(consumer.find("parent"), where + ".parent")});
    }
    const std::vector<Json> &synthesized = list(config.find("synthesized"), "synthesized");
    for (size_t i = 0; i < synthesized.size(); ++i) {
        std::string where = "synthesized[" + std::to_string(i) + "]";
        const Json &synth = object(synthesized[i], where);
        SynthesizedDeclaration declared{text(synth.find("name"), where + ".name"),
                                        text(synth.find("header"), where + ".header"),
                                        integer(synth.find("parent"), where + ".parent"),
                                        optionalText(synth.find("sibling"), where + ".sibling"),
                                        {}};
        // Overrides that are not an object read as none, which readDeclaration refuses.
        const Json *overrides = synth.find("overrides");
        const auto &specs = overrides && overrides->kind == Json::Kind::Object
                                    ? overrides->object
                                    : std::vector<std::pair<std::string, Json>>();
        for (const auto &[method, value] : specs) {
            std::string at = where + ".overrides." + method;
            const Json &spec = object(value, at);
            const Json *required = spec.find("required");
            if (!absent(required) && required->kind != Json::Kind::Boolean)
                malformed(at + ".required", "true or false");
            declared.overrides.push_back({method, !absent(required) && required->boolean,
                                          names(spec.find("selectors"), at + ".selectors")});
        }
        out.synthesized.push_back(std::move(declared));
    }
    const Json *imports = config.find("imports");
    if (!absent(imports)) {
        object(*imports, "imports");
        for (const auto &[name, value] : imports->object) {
            std::string where = "imports." + name;
            const Json &policy = object(value, where);
            out.imports[name] = {integer(policy.find("library"), where + ".library"),
                                 optionalNames(policy.find("candidates"), where + ".candidates"),
                                 text(policy.find("fallback"), where + ".fallback")};
        }
    }
    return out;
}

bool inRange(int64_t index, size_t count) {
    return index >= 0 && static_cast<uint64_t>(index) < count;
}

bool symbolName(const std::string &name) {
    return !name.empty() && name.find('\0') == std::string::npos;
}

// --- Mangled names ----------------------------------------------------------------

bool digit(char c) {
    return c >= '0' && c <= '9';
}

// The decimal number at text[at], advancing at past its digits; at least one digit
// must be there. A number too large for any name saturates.
std::optional<size_t> number(const std::string &text, size_t &at) {
    size_t start = at, value = 0;
    while (at < text.size() && digit(text[at])) {
        value = value > text.size() ? value : value * 10 + static_cast<size_t>(text[at] - '0');
        ++at;
    }
    if (at == start)
        return std::nullopt;
    return value;
}

bool startsWith(const std::string &text, const std::string &prefix) {
    return text.compare(0, prefix.size(), prefix) == 0;
}

// The top-level class whose member, vtable or typeinfo the symbol name is.
std::optional<std::string> ownerClass(const std::string &name) {
    size_t at;
    if (startsWith(name, "__ZNK") && name.size() > 5 && digit(name[5]))
        at = 5;
    else if (startsWith(name, "__ZN"))
        at = 4;
    else if (startsWith(name, "__ZT") && name.size() > 4 &&
             (name[4] == 'V' || name[4] == 'I' || name[4] == 'S'))
        at = 5;
    else
        return std::nullopt;
    std::optional<size_t> length = number(name, at);
    if (!length)
        return std::nullopt;
    std::string rest = name.substr(at);
    rest = rest.substr(0, rest.find('\n'));
    return rest.substr(0, std::min(*length, rest.size()));
}

std::string mangledLength(const std::string &name) {
    return std::to_string(name.size()) + name;
}

// --- The layout probe ----------------------------------------------------------------

// C++ that derives probe class i from rows[i] and records each class's sizeof.
std::string probeSource(const std::vector<std::pair<std::string, std::string>> &rows) {
    std::string source;
    for (const auto &row : rows)
        source += "#include \"" + row.second + "\"\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        std::string probe = ProbeClass + std::to_string(i);
        source += "class " + probe + " : public " + identifier(rows[i].first) + " { public:\n";
        source += "~" + probe + "() override; };\n";
        source += probe + "::~" + probe + "() {}\n";
    }
    std::string sizes;
    for (const auto &row : rows)
        sizes += (sizes.empty() ? "sizeof(" : ",sizeof(") + identifier(row.first) + ")";
    source += "extern \"C\" const unsigned graft_layout_sizes[] = {" + sizes + "};\n";
    return source;
}

// The architecture the consumer builds for, from its flags' -arch pair, x86-64 when
// they have none. The probe compiles for it, because a class's inherited vtable
// differs between architectures: the Kernel.framework headers give arm64 more
// reserved slots than x86-64, in another order.
std::string targetArch(const std::vector<std::string> &flags) {
    for (size_t i = 0; i + 1 < flags.size(); ++i) {
        if (flags[i] == "-arch")
            return flags[i + 1];
    }
    return "x86_64";
}

// The consumer's flags without their -arch pairs: the probe names its own.
std::vector<std::string> withoutArch(const std::vector<std::string> &flags) {
    std::vector<std::string> kept;
    bool skip = false;
    for (const std::string &flag : flags) {
        if (skip)
            skip = false;
        else if (flag == "-arch")
            skip = true;
        else
            kept.push_back(flag);
    }
    return kept;
}

// What command, which must succeed, prints, without the whitespace around it.
std::string captured(const std::vector<std::string> &command) {
    TemporaryDirectory temp("graft-xcrun-");
    std::string output = (temp.path() / "output").string();
    if (!run(command, output))
        throw Error(shellJoin(command) + " failed");
    std::string text = readText(output);
    size_t first = text.find_first_not_of(" \t\n\r\f\v");
    size_t last = text.find_last_not_of(" \t\n\r\f\v");
    return first == std::string::npos ? "" : text.substr(first, last - first + 1);
}

// Compile source with kext flags for macOS 11, on the consumer's architecture and
// then with its own flags, in a scratch directory outside the consumer's tree.
// Returns the object and the compiler's dump of its vtable layouts.
std::pair<Image, std::string> compileProbe(const std::string &source,
                                           const std::vector<std::string> &flags) {
    std::string sdk = captured({"xcrun", "--sdk", "macosx", "--show-sdk-path"});
    TemporaryDirectory temp("graft-layout-");
    std::string sourcePath = (temp.path() / "probe.cpp").string();
    std::string objectPath = (temp.path() / "probe.o").string();
    writeText(sourcePath, source);
    std::vector<std::string> command = {"xcrun",
                                        "clang++",
                                        "-arch",
                                        targetArch(flags),
                                        "-isysroot",
                                        sdk,
                                        "-mmacosx-version-min=11.0",
                                        "-mkernel",
                                        "-nostdinc",
                                        "-isystem",
                                        sdk + "/System/Library/Frameworks/Kernel.framework/Headers",
                                        "-DKERNEL",
                                        "-std=c++17",
                                        "-fapple-kext",
                                        "-fno-rtti",
                                        "-fno-exceptions",
                                        "-Xclang",
                                        "-fdump-vtable-layouts"};
    for (const std::string &flag : withoutArch(flags))
        command.push_back(flag);
    command.insert(command.end(), {"-c", sourcePath, "-o", objectPath});
    std::string dump = (temp.path() / "layouts.txt").string();
    std::string errors = (temp.path() / "errors.txt").string();
    if (!run(command, dump, errors)) {
        std::cerr << readText(errors);
        throw Error("consumer layout compilation failed");
    }
    return {Image(readFile(objectPath)), readText(dump)};
}

// Each probe class's vtable entry count, from the compiler's dump of lines such
// as "Vtable for 'GraftLayoutProbe0' (12 entries)". A class dumped twice keeps
// its last count.
std::map<size_t, uint64_t> vtableEntries(const std::string &dump) {
    std::map<size_t, uint64_t> entries;
    const std::string marker = "Vtable for '" + ProbeClass;
    for (size_t at = dump.find(marker); at != std::string::npos; at = dump.find(marker, at + 1)) {
        size_t cursor = at + marker.size();
        const std::string between = "' (", after = " entries)";
        std::optional<size_t> index = number(dump, cursor);
        if (!index || dump.compare(cursor, between.size(), between) != 0)
            continue;
        cursor += between.size();
        std::optional<size_t> count = number(dump, cursor);
        if (!count || dump.compare(cursor, after.size(), after) != 0)
            continue;
        entries[*index] = *count;
    }
    return entries;
}

// The methods in class name's vtable slots, read from probe class index's
// vtable. entries is the compiler's count, which includes the offset-to-top and
// type-information words and the final zero word: the slots are the rest.
std::vector<std::string> probeSlots(const Image &image, size_t index, const std::string &name,
                                    std::optional<uint64_t> entries,
                                    const std::map<uint64_t, Relocation> &relocations) {
    std::string probe = ProbeClass + std::to_string(index);
    const Symbol &vtable = image.symbol(vtableName(probe));
    if (!entries || *entries < 4)
        throw Error("compiler did not report a single primary vtable");
    uint64_t end = vtable.value + (*entries - 1) * PointerSize;
    if (relocations.count(end) || image.read(end, PointerSize) != Bytes(PointerSize))
        throw Error("compiler vtable lacks its final zero sentinel");
    // An arm64e object relocates each slot as a signed pointer, an x86-64 one as a
    // plain pointer. Either way the slot names its method by the relocation's symbol.
    uint32_t pointer =
            image.arch == "arm64e" ? ARM64_RELOC_AUTHENTICATED_POINTER : X86_64_RELOC_UNSIGNED;
    std::vector<std::string> slots;
    for (uint64_t slot = 0; slot < *entries - 3; ++slot) {
        auto found = relocations.find(vtable.value + VtableAddressPoint + slot * PointerSize);
        if (found == relocations.end() || !found->second.external ||
            found->second.width != PointerSize || found->second.pcrel ||
            found->second.kind != pointer)
            throw Error("compiler vtable slot has no supported symbolic relocation");
        std::string method = image.symbols[found->second.symbolIndex].name;
        std::string prefix = "__ZN" + mangledLength(probe);
        // The probe's destructors name the probe class; they stand for the parent's.
        if (startsWith(method, prefix)) {
            std::string destructor = method.substr(prefix.size());
            if (destructor != "D0Ev" && destructor != "D1Ev")
                throw Error("layout probe introduced an unexpected method");
            method = "__ZN" + mangledLength(name) + destructor;
        }
        slots.push_back(method);
    }
    return slots;
}

// --- layouts.json ---------------------------------------------------------------------

Json layoutJson(const ClassLayout &layout) {
    std::vector<Json> slots;
    for (const std::string &slot : layout.slots)
        slots.push_back(Json::makeString(slot));
    return Json::makeObject({{"name", Json::makeString(layout.name)},
                             {"size", Json::makeInteger(layout.size)},
                             {"slots", Json::makeArray(std::move(slots))}});
}

std::vector<ClassLayout> parseLayouts(const Json *rows, const std::string &where) {
    std::vector<ClassLayout> out;
    if (!rows)
        return out;
    auto malformedLayouts = [&] { throw Error("layouts " + where + " are malformed"); };
    if (rows->kind != Json::Kind::Array)
        malformedLayouts();
    for (const Json &row : rows->array) {
        if (row.kind != Json::Kind::Object)
            malformedLayouts();
        const Json *name = row.find("name"), *size = row.find("size"), *slots = row.find("slots");
        if (!name || name->kind != Json::Kind::String || !size ||
            size->kind != Json::Kind::Integer || size->integer < 0 || size->integer > UINT32_MAX ||
            !slots || slots->kind != Json::Kind::Array)
            malformedLayouts();
        ClassLayout layout{name->string, static_cast<uint32_t>(size->integer), {}};
        for (const Json &slot : slots->array) {
            if (slot.kind != Json::Kind::String)
                malformedLayouts();
            layout.slots.push_back(slot.string);
        }
        out.push_back(std::move(layout));
    }
    return out;
}

// --- The census --------------------------------------------------------------------------

// Each declared row with its measured layout, refusing lists of different lengths.
void requirePaired(size_t declared, size_t measured) {
    if (declared != measured)
        throw Error("layouts describe another declaration");
}

std::vector<SegmentRecord> consumerSegments(const Image &image) {
    // Graft maps a manifest offset through the placed segment of the same name.
    // __LINKEDIT holds no bound offset and the loaded image leaves it out.
    std::vector<SegmentRecord> segments;
    for (const Segment &segment : image.segments) {
        if (segment.name != "__LINKEDIT" && segment.vmsize && segment.address >= image.base)
            segments.push_back({segment.name, segment.address - image.base, segment.vmsize});
    }
    std::stable_sort(segments.begin(), segments.end(),
                     [](const auto &a, const auto &b) { return a.origin < b.origin; });
    return segments;
}

// The provider index that owns each class: its declared parents and the classes
// each library lists.
std::map<std::string, int64_t> classOwners(const Declaration &config) {
    std::map<std::string, int64_t> owners;
    for (const ClassDeclaration &cls : config.classes)
        owners[cls.name] = cls.library;
    for (size_t i = 0; i < config.libraries.size(); ++i) {
        for (const std::string &name : config.libraries[i].classes) {
            auto owner = owners.find(name);
            if (owner != owners.end() && owner->second != static_cast<int64_t>(i))
                throw Error("class is claimed by two providers");
            owners[name] = static_cast<int64_t>(i);
        }
    }
    return owners;
}

// Each symbol the consumer imports and whether it is weak. An x86-64 kext's
// imports are its undefined symbols; an arm64e kext has none and records its
// imports as chained-fixup binds instead.
std::vector<std::pair<std::string, bool>> importedSymbols(const Image &image) {
    std::vector<std::pair<std::string, bool>> out;
    if (image.arch == "arm64e") {
        std::map<std::string, bool> weak;
        for (const ChainedFixup &fixup : image.fixups()) {
            if (fixup.bind)
                weak[fixup.symbol] = weak[fixup.symbol] || fixup.weak;
        }
        return std::vector<std::pair<std::string, bool>>(weak.begin(), weak.end());
    }
    for (const Symbol &symbol : image.symbols) {
        if (symbol.undefined())
            out.emplace_back(symbol.name, (symbol.descriptor & N_WEAK_REF) != 0);
    }
    return out;
}

// The consumer's imports that Graft binds, by name: those graft.json's imports
// name, and the members, vtables and typeinfo of the classes a provider owns. The
// others are kernel and Boot-collection imports, which the boot linker resolves.
std::map<std::string, Import> selectImports(const Declaration &config, const Image &image) {
    std::map<std::string, int64_t> owners = classOwners(config);
    std::map<std::string, Import> selected;
    for (const auto &[name, weak] : importedSymbols(image)) {
        auto policy = config.imports.find(name);
        std::optional<int64_t> library;
        if (policy != config.imports.end()) {
            library = policy->second.library;
        } else if (std::optional<std::string> owner = ownerClass(name)) {
            auto found = owners.find(*owner);
            if (found != owners.end())
                library = found->second;
        }
        if (!library)
            continue;
        if (!inRange(*library, config.libraries.size()))
            throw Error("import provider index outside declaration");
        std::vector<std::string> candidates = {name};
        if (policy != config.imports.end() && policy->second.candidates)
            candidates = *policy->second.candidates;
        if (candidates.empty() || !std::all_of(candidates.begin(), candidates.end(), symbolName))
            throw Error("import candidates must be nonempty symbol names");
        uint64_t fallback = manifest::NoFallback;
        if (policy != config.imports.end() && !policy->second.fallback.empty())
            fallback = image.symbol(policy->second.fallback).value - image.base;
        selected[name] = {*library, candidates, fallback, weak};
    }
    for (const auto &policy : config.imports) {
        if (!selected.count(policy.first))
            throw Error("an explicit import policy names no undefined import");
    }
    return selected;
}

bool executable(const Image &image, uint64_t address) {
    for (const Section &section : image.sections) {
        if (section.executable() && section.address <= address &&
            address - section.address < section.size)
            return true;
    }
    return false;
}

// The address in the consumer's own image that the pointer at address points
// to, by its relocation, or nothing when it points elsewhere. A local
// relocation's pointer holds the address; an external one's holds only the
// addend to its symbol, which the consumer defines when the pointer is local.
std::optional<uint64_t> localTarget(const Image &image, const Relocation *relocation,
                                    uint64_t address) {
    if (!relocation)
        return std::nullopt;
    uint64_t stored = load64(image.read(address, PointerSize).data());
    if (!relocation->external)
        return stored;
    const Symbol &symbol = image.symbols[relocation->symbolIndex];
    if (!symbol.defined())
        return std::nullopt;
    return symbol.value + stored;
}

// Each x86-64 consumer class's vtable, and the addresses of all their parent
// slots. A slot is the consumer's own when it holds a local pointer to the
// consumer's code. The other parent slots are left for Graft to fill from the
// live parent, so no parent slot is a site.
std::vector<VtableRecord> consumerVtables(const Image &image, const Declaration &config,
                                          const Layouts &layouts,
                                          const std::vector<Relocation> &relocations,
                                          std::set<uint64_t> &parentSlots) {
    std::map<uint64_t, const Relocation *> byAddress;
    for (const Relocation &relocation : relocations)
        byAddress[relocation.address] = &relocation;
    std::vector<VtableRecord> vtables;
    for (size_t i = 0; i < config.consumers.size(); ++i) {
        const ConsumerDeclaration &consumer = config.consumers[i];
        const Symbol &symbol = image.symbol(vtableName(consumer.name));
        uint64_t inherited = layouts.classes[static_cast<size_t>(consumer.parent)].slots.size();
        uint64_t count = layouts.consumers[i].slots.size();
        if (count < inherited || image.symbolExtent(symbol) < (count + 2) * PointerSize)
            throw Error("consumer vtable is shorter than its compiler layout");
        VtableRecord vtable{symbol.value + VtableAddressPoint - image.base,
                            static_cast<uint32_t>(consumer.parent),
                            static_cast<uint32_t>(count),
                            {},
                            {}};
        for (uint64_t slot = 0; slot < count; ++slot) {
            uint64_t address = symbol.value + VtableAddressPoint + slot * PointerSize;
            auto relocation = byAddress.find(address);
            std::optional<uint64_t> target = localTarget(
                    image, relocation == byAddress.end() ? nullptr : relocation->second, address);
            if (target) {
                if (!executable(image, *target))
                    throw Error("consumer override does not point to its executable code");
                if (slot < inherited)
                    vtable.overrides.push_back(static_cast<uint32_t>(slot));
            } else if (slot >= inherited) {
                throw Error("appended consumer virtual method is not local executable code");
            }
            if (slot < inherited)
                parentSlots.insert(address);
        }
        vtables.push_back(std::move(vtable));
    }
    std::stable_sort(vtables.begin(), vtables.end(),
                     [](const auto &a, const auto &b) { return a.offset < b.offset; });
    return vtables;
}

// The site of an x86-64 relocation against import name: a pointer, or the
// displacement of a call or a jump.
Site importSite(const Image &image, const Relocation &relocation, const std::string &name) {
    uint8_t opcode;
    if (relocation.pcrel) {
        if (relocation.width != 4 || relocation.kind != X86_64_RELOC_BRANCH)
            throw Error("unsupported System-KC PC-relative relocation");
        opcode = image.read(relocation.address - 1, 1)[0];
        if (opcode != manifest::OpcodeCall && opcode != manifest::OpcodeJump)
            throw Error("unsupported System-KC branch instruction");
    } else {
        if (relocation.width != PointerSize || relocation.kind != X86_64_RELOC_UNSIGNED)
            throw Error("unsupported System-KC pointer relocation");
        opcode = manifest::OpcodePointer;
    }
    Bytes stored = image.read(relocation.address, relocation.width);
    uint64_t raw = 0;
    for (size_t i = 0; i < stored.size(); ++i)
        raw |= static_cast<uint64_t>(stored[i]) << (8 * i);
    uint64_t sign = uint64_t{1} << (8 * stored.size() - 1);
    int64_t addend = static_cast<int64_t>((raw ^ sign) - sign);
    return {relocation.address - image.base, name, relocation.width, opcode, addend, std::nullopt};
}

// Every x86-64 reference to a selected import outside the vtables' parent slots.
std::vector<Site> importSites(const Image &image, const std::map<std::string, Import> &imports,
                              const std::vector<Relocation> &relocations,
                              const std::set<uint64_t> &parentSlots) {
    std::vector<Site> sites;
    std::set<std::string> referenced;
    for (const Relocation &relocation : relocations) {
        if (!relocation.external)
            continue;
        const std::string &name = image.symbols[relocation.symbolIndex].name;
        if (!imports.count(name))
            continue;
        referenced.insert(name);
        if (!parentSlots.count(relocation.address))
            sites.push_back(importSite(image, relocation, name));
    }
    if (referenced.size() != imports.size())
        throw Error("selected import has no relocation site");
    std::stable_sort(sites.begin(), sites.end(),
                     [](const auto &a, const auto &b) { return a.offset < b.offset; });
    return sites;
}

// A signing context's auth byte (signed, key, address diversity) and diversity,
// or 0 and 0 for none.
std::pair<uint8_t, uint16_t> authTrailer(const std::optional<AuthContext> &auth) {
    if (!auth)
        return {0, 0};
    return {static_cast<uint8_t>(manifest::SiteSigned | auth->key << 1 |
                                 (auth->addrDiv ? manifest::SiteAddrDiv : 0)),
            static_cast<uint16_t>(auth->diversity)};
}

using Sited = std::map<uint64_t, const ChainedFixup *>;

// Each arm64e consumer class's vtable, and the addresses of all their parent
// slots. A slot is the consumer's own when its fixup is a local rebase to the
// consumer's code. A parent slot's fixup is instead a bind to the inherited
// method, which Graft fills from the live parent, so no parent slot is a site.
std::vector<VtableRecord> consumerVtablesArm64e(const Image &image, const Declaration &config,
                                                const Layouts &layouts, const Sited &sited,
                                                std::set<uint64_t> &parentSlots) {
    std::vector<VtableRecord> vtables;
    for (size_t i = 0; i < config.consumers.size(); ++i) {
        const ConsumerDeclaration &consumer = config.consumers[i];
        const Symbol &symbol = image.symbol(vtableName(consumer.name));
        uint64_t inherited = layouts.classes[static_cast<size_t>(consumer.parent)].slots.size();
        uint64_t count = layouts.consumers[i].slots.size();
        if (count < inherited || image.symbolExtent(symbol) < (count + 2) * PointerSize)
            throw Error("consumer vtable is shorter than its compiler layout");
        VtableRecord vtable{symbol.value + VtableAddressPoint - image.base,
                            static_cast<uint32_t>(consumer.parent),
                            static_cast<uint32_t>(count),
                            {},
                            {}};
        for (uint64_t slot = 0; slot < count; ++slot) {
            uint64_t address = symbol.value + VtableAddressPoint + slot * PointerSize;
            auto found = sited.find(address);
            const ChainedFixup *fixup = found == sited.end() ? nullptr : found->second;
            bool own = fixup && !fixup->bind;
            if (own) {
                if (!executable(image, fixup->target))
                    throw Error("consumer override does not point to its executable code");
                if (slot < inherited)
                    vtable.overrides.push_back(static_cast<uint32_t>(slot));
            } else if (slot >= inherited) {
                throw Error("appended consumer virtual method is not local executable code");
            }
            if (slot < inherited) {
                parentSlots.insert(address);
                // An inherited slot Graft re-signs takes the signing context of its
                // bind to the parent's method, so Graft signs the live method for
                // this slot exactly as the consumer authenticates it. A slot with no
                // bind has no context to sign with.
                if (!own) {
                    if (!fixup)
                        throw Error("consumer vtable's inherited slot has no bind to its "
                                    "parent's method");
                    auto [opcode, diversity] = authTrailer(fixup->auth);
                    vtable.fills.push_back({static_cast<uint32_t>(slot), opcode, diversity});
                }
            }
        }
        vtables.push_back(std::move(vtable));
    }
    std::stable_sort(vtables.begin(), vtables.end(),
                     [](const auto &a, const auto &b) { return a.offset < b.offset; });
    return vtables;
}

// Each synthesized class's record, and the addresses of its inherited slots.
// Each override's first selector is its own slot's method symbol, and its
// implementation and signing context come from that slot's local rebase. Every
// parent slot holding one of the class's own methods is an override: the
// declared ones take their declaration's required flag and selectors, and the
// rest, such as the destructors and getMetaClass the metaclass macros define,
// are required, so the bind refuses rather than leave the parent's method in
// their place. A local rebase to code the stand-in defines, its reserved slots,
// is not an override. The inherited slots are binds to the parent's methods,
// which Graft fills when it lays out the vtable, so none is an import site. The
// record's vtable offset is the image offset of the vtable's first word.
std::vector<SynthClassRecord> consumerSynthesizedArm64e(const Image &image,
                                                        const Declaration &config,
                                                        const Layouts &layouts, const Sited &sited,
                                                        std::set<uint64_t> &parentSlots) {
    std::vector<SynthClassRecord> records;
    for (size_t i = 0; i < config.synthesized.size(); ++i) {
        const SynthesizedDeclaration &synth = config.synthesized[i];
        const ClassLayout &layout = layouts.synthesized[i];
        const Symbol &symbol = image.symbol(vtableName(synth.name));
        const ClassLayout &parentLayout = layouts.classes[static_cast<size_t>(synth.parent)];
        uint64_t inherited = parentLayout.slots.size();
        uint64_t count = layout.slots.size();
        if (count < inherited || image.symbolExtent(symbol) < (count + 2) * PointerSize)
            throw Error("synthesized consumer vtable is shorter than its compiler layout");
        std::map<std::string, std::pair<std::string, const ChainedFixup *>> found;
        std::vector<std::pair<std::string, const ChainedFixup *>> undeclared;
        for (uint64_t slot = 0; slot < count; ++slot) {
            uint64_t address = symbol.value + VtableAddressPoint + slot * PointerSize;
            auto entry = sited.find(address);
            const ChainedFixup *fixup = entry == sited.end() ? nullptr : entry->second;
            if (!fixup || fixup->bind) {
                if (slot < inherited)
                    parentSlots.insert(address);
                continue;
            }
            if (!executable(image, fixup->target))
                throw Error("synthesized override does not point to its executable code");
            const std::string &method = layout.slots[slot];
            if (slot >= inherited || ownerClass(method) != synth.name)
                continue;
            // The override's selector is the method the layout names in this slot,
            // and its code the slot's target, so the two must be one method. A stale
            // layout would otherwise put one method's code in another's live slot.
            if (image.symbol(method).value != fixup->target)
                throw Error("synthesized vtable slot does not hold the method its layout names");
            std::optional<std::string> name = memberName(method);
            bool declared = name && std::any_of(synth.overrides.begin(), synth.overrides.end(),
                                                [&](const auto &o) { return o.method == *name; });
            if (declared) {
                if (found.count(*name))
                    throw Error("synthesized override method is ambiguous");
                found[*name] = {method, fixup};
            } else {
                undeclared.emplace_back(method, fixup);
            }
        }
        const ClassDeclaration &parent = config.classes[static_cast<size_t>(synth.parent)];
        SynthClassRecord record{synth.name,
                                parent.name,
                                synth.sibling.value_or(""),
                                static_cast<uint32_t>(parent.library),
                                symbol.value - image.base,
                                static_cast<uint32_t>(count + VtableAddressPoint / PointerSize),
                                parentLayout.size,
                                {}};
        for (const OverrideDeclaration &spec : synth.overrides) {
            auto method = found.find(spec.method);
            if (method == found.end())
                throw Error("synthesized override " + spec.method +
                            " is not the consumer's own method");
            auto [selector, fixup] = method->second;
            auto [opcode, diversity] = authTrailer(fixup->auth);
            std::vector<std::string> selectors = {selector};
            selectors.insert(selectors.end(), spec.selectors.begin(), spec.selectors.end());
            record.overrides.push_back(
                    {selectors, fixup->target - image.base, diversity, opcode, spec.required});
        }
        for (const auto &[selector, fixup] : undeclared) {
            auto [opcode, diversity] = authTrailer(fixup->auth);
            record.overrides.push_back(
                    {{selector}, fixup->target - image.base, diversity, opcode, true});
        }
        records.push_back(std::move(record));
    }
    return records;
}

// Every arm64e chained-fixup bind to a selected import outside the vtables'
// parent slots.
std::vector<Site> importSitesArm64e(const Image &image,
                                    const std::map<std::string, Import> &imports,
                                    const std::vector<ChainedFixup> &fixups,
                                    const std::set<uint64_t> &parentSlots) {
    std::vector<Site> sites;
    std::set<std::string> referenced;
    for (const ChainedFixup &fixup : fixups) {
        if (!fixup.bind || !imports.count(fixup.symbol))
            continue;
        referenced.insert(fixup.symbol);
        if (!parentSlots.count(fixup.address))
            sites.push_back({fixup.address - image.base, fixup.symbol,
                             static_cast<uint32_t>(PointerSize), manifest::OpcodePointer,
                             fixup.addend, fixup.auth});
    }
    if (referenced.size() != imports.size())
        throw Error("selected import has no fixup site");
    std::stable_sort(sites.begin(), sites.end(),
                     [](const auto &a, const auto &b) { return a.offset < b.offset; });
    return sites;
}

// --- The manifest -----------------------------------------------------------------------

// The manifest's string table: each name stored once, NUL-terminated, at the
// offset of its first use. Offset 0 is the empty string.
class StringTable {
public:
    uint32_t add(const std::string &value) {
        if (!symbolName(value))
            throw Error("manifest strings must be nonempty and contain no NUL");
        auto found = offsets_.find(value);
        if (found != offsets_.end())
            return found->second;
        uint32_t offset = static_cast<uint32_t>(data_.size());
        offsets_[value] = offset;
        data_.insert(data_.end(), value.begin(), value.end());
        data_.push_back(0);
        return offset;
    }
    const Bytes &data() const { return data_; }

private:
    Bytes data_ = Bytes(1, 0);
    std::map<std::string, uint32_t> offsets_;
};

// A manifest table's rows, encoded, and their count.
struct Rows {
    Bytes bytes;
    uint32_t count = 0;
    template <typename Record>
    void add(const Record &record) {
        encode(bytes, record);
        ++count;
    }
};

// What a validation failure names, as docs/manifest.md names the tables.
std::string failureTable(manifest::Error error) {
    switch (error) {
    case manifest::Error::None: break;
    case manifest::Error::Header: return "header";
    case manifest::Error::Version: return "version";
    case manifest::Error::Range: return "table range";
    case manifest::Error::Overlap: return "table overlap";
    case manifest::Error::String: return "strings";
    case manifest::Error::Library: return "libraries";
    case manifest::Error::Import: return "imports";
    case manifest::Error::Site: return "sites";
    case manifest::Error::Class: return "classes";
    case manifest::Error::Slot: return "slots";
    case manifest::Error::Vtable: return "vtables";
    case manifest::Error::Segment: return "segments";
    case manifest::Error::Fill: return "fills";
    case manifest::Error::Selector: return "selectors";
    case manifest::Error::SynthOverride: return "synthOverrides";
    case manifest::Error::SynthClass: return "synthClasses";
    }
    return "";
}

uint32_t manifestVersion(const Image &image) {
    return image.arch == "arm64e" ? manifest::VersionArm64e : manifest::Version;
}

uint64_t align(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

// Rewrite each chained-fixup bind to a selected import into a rebase at its
// placeholder. A dormant arm64e consumer must carry no bind the boot linker
// cannot resolve, as its x86-64 form carries no undefined provider symbol. Each
// rewritten slot keeps its chain link and, when it is signed, its signing
// context, so the boot firmware signs the placeholder pointer and Graft later
// replaces it with the real address, signed the same way.
Image rebindToPlaceholders(const Image &image,
                           const std::map<std::string, uint64_t> &placeholders) {
    Bytes data = image.data;
    for (const ChainedFixup &fixup : image.fixups()) {
        auto placeholder = placeholders.find(fixup.symbol);
        if (!fixup.bind || placeholder == placeholders.end())
            continue;
        uint64_t target = placeholder->second;
        uint64_t next = load64(&data[fixup.entryOffset]) >> 51 & 0x7ff;
        uint64_t rebased;
        if (fixup.auth) {
            // auth 1, bind 0
            rebased = (target & 0xffffffff) | uint64_t{fixup.auth->diversity} << 32 |
                      uint64_t{fixup.auth->addrDiv} << 48 | uint64_t{fixup.auth->key} << 49 |
                      next << 51 | uint64_t{1} << 63;
        } else {
            // auth 0, bind 0
            rebased = (target & 0x7ffffffffff) | next << 51;
        }
        store64(&data[fixup.entryOffset], rebased);
    }
    return Image(std::move(data));
}

// The renames that hide each consumer class's superClass marker from the boot
// linker. A consumer class's parent is in the System collection, which the boot
// linker cannot see, so Graft links its vtable instead. The marker keeps its
// bytes and relocations, under the same-length name graftClass.
std::vector<std::pair<std::string, std::string>>
deferredSuperclassMarkers(const Image &image, const Declaration &config) {
    std::set<std::string> present;
    for (const Symbol &symbol : image.symbols)
        present.insert(symbol.name);
    std::vector<std::pair<std::string, std::string>> markers;
    for (const ConsumerDeclaration &consumer : config.consumers) {
        std::string marker = "__ZN" + mangledLength(consumer.name) + "10superClassE";
        if (present.count(marker))
            markers.emplace_back(marker, "__ZN" + mangledLength(consumer.name) + "10graftClassE");
    }
    return markers;
}

// --- The Info.plist ------------------------------------------------------------------------

const std::string &plistString(const Plist &info, const std::string &key) {
    const Plist *value = info.kind == Plist::Kind::Dict ? info.find(key) : nullptr;
    if (!value || value->kind != Plist::Kind::String)
        throw Error("Info.plist has no " + key + " string");
    return value->text;
}

// The consumer's personalities, gated on the activation key. Each keeps its own
// keys, must name the device, and gains the bundle's CFBundleIdentifier and an
// IOPropertyMatch requiring the key on the device nub it attaches to. The kernel
// registers the personalities when the consumer loads, but none matches until
// Graft sets the key on the nub after binding, so the consumer is not even
// constructed until then. A personality's own IOPropertyMatch, a dictionary or
// a list of them any one of which matches, keeps its properties and gains the
// key, in each dictionary of a list.
Plist gatedPersonalities(const Plist &info) {
    const Plist *personalities = info.find("IOKitPersonalities");
    bool none = !personalities ||
                (personalities->kind == Plist::Kind::Dict && personalities->dict.empty());
    if (none)
        throw Error("consumer has no personalities to gate");
    if (personalities->kind != Plist::Kind::Dict)
        throw Error("Info.plist IOKitPersonalities is not a dictionary");
    std::string key = gateKey(info);
    Plist gated = Plist::makeDict();
    for (const auto &[name, value] : personalities->dict) {
        if (value.kind != Plist::Kind::Dict)
            throw Error("consumer personality " + name + " is not a dictionary");
        const Plist *match = value.find("IOPCIPrimaryMatch");
        if (!match || match->kind != Plist::Kind::String || match->text != DeviceMatch)
            throw Error("consumer personality " + name + " does not match IOPCIPrimaryMatch " +
                        DeviceMatch + ", the device Graft serves");
        Plist personality = value;
        personality.set("CFBundleIdentifier",
                        Plist::makeString(plistString(info, "CFBundleIdentifier")));
        Plist *properties = personality.find("IOPropertyMatch");
        if (!properties) {
            personality.set("IOPropertyMatch", Plist::makeDict());
            properties = personality.find("IOPropertyMatch");
        }
        std::vector<Plist *> gates;
        if (properties->kind == Plist::Kind::Dict)
            gates.push_back(properties);
        else if (properties->kind == Plist::Kind::Array)
            for (Plist &each : properties->array)
                gates.push_back(&each);
        if (gates.empty() || std::any_of(gates.begin(), gates.end(), [](const Plist *each) {
                return each->kind != Plist::Kind::Dict;
            }))
            throw Error("consumer personality " + name +
                        " has an IOPropertyMatch that is not a dictionary or a list of them");
        for (Plist *each : gates)
            each->set(key, Plist::makeBoolean(true));
        gated.set(name, std::move(personality));
    }
    return gated;
}

} // namespace

// --- The steps ---------------------------------------------------------------------------

std::string vtableName(const std::string &name) {
    return "__ZTV" + mangledLength(name);
}

std::optional<std::string> memberName(const std::string &symbol) {
    // __ZN, then cv and ref qualifiers, then the class's and the method's names,
    // each prefixed by its length.
    if (!startsWith(symbol, "__ZN"))
        return std::nullopt;
    size_t at = 4;
    while (at < symbol.size() && (symbol[at] == 'K' || symbol[at] == 'V' || symbol[at] == 'r'))
        ++at;
    if (at < symbol.size() && (symbol[at] == 'R' || symbol[at] == 'O'))
        ++at;
    std::optional<size_t> classLength = number(symbol, at);
    if (!classLength || *classLength > symbol.size() - at)
        return std::nullopt;
    at += *classLength;
    std::optional<size_t> length = number(symbol, at);
    if (!length || *length > symbol.size() - at)
        return std::nullopt;
    return symbol.substr(at, *length);
}

Declaration readDeclaration(const std::string &path) {
    Declaration config = parseDeclaration(parseJson(readText(path), path));
    const auto &libraries = config.libraries;
    const auto &classes = config.classes;
    if (libraries.empty() ||
        std::any_of(libraries.begin(), libraries.end(), [](const auto &l) { return l.id.empty(); }))
        throw Error("declare at least one named provider");
    std::set<std::string> ids;
    for (const LibraryDeclaration &library : libraries)
        ids.insert(library.id);
    if (ids.size() != libraries.size())
        throw Error("duplicate provider identifier");
    for (const ClassDeclaration &cls : classes) {
        identifier(cls.name);
        if (cls.header.empty() || !inRange(cls.library, libraries.size()))
            throw Error("class has no header or provider");
    }
    // Graft confirms each provider through a class's registered metaclass: a class
    // the consumer declares from it, or its anchor where the consumer only imports
    // from it.
    for (size_t i = 0; i < libraries.size(); ++i) {
        if (libraries[i].anchor) {
            identifier(*libraries[i].anchor);
        } else if (std::none_of(classes.begin(), classes.end(), [&](const auto &cls) {
                       return cls.library == static_cast<int64_t>(i);
                   })) {
            throw Error("provider " + libraries[i].id +
                        " has no declared class and no anchor to verify it by");
        }
    }
    std::set<std::string> parents;
    for (const ClassDeclaration &cls : classes)
        parents.insert(cls.name);
    if (parents.size() != classes.size())
        throw Error("duplicate parent class");
    for (const ConsumerDeclaration &consumer : config.consumers) {
        identifier(consumer.name);
        if (consumer.header.empty() || !inRange(consumer.parent, classes.size()))
            throw Error("consumer class has no header or parent");
    }
    // A synthesized class derives from a stand-in for a parent whose vtable changes
    // between releases. Graft lays out its vtable when it binds, by method, so the
    // class declares its overrides by method name, each with its required flag and
    // other selectors, rather than by slot.
    for (const SynthesizedDeclaration &synth : config.synthesized) {
        identifier(synth.name);
        if (synth.header.empty() || !inRange(synth.parent, classes.size()))
            throw Error("synthesized class has no header or parent");
        if (synth.sibling)
            identifier(*synth.sibling);
        if (synth.overrides.empty())
            throw Error("synthesized class must declare its overrides");
        for (const OverrideDeclaration &spec : synth.overrides) {
            if (!std::all_of(spec.selectors.begin(), spec.selectors.end(), symbolName))
                throw Error("synthesized override selector must be a symbol name");
        }
    }
    // Census finds each consumer and synthesized class's vtable by the class's
    // name, so two classes of one name would share one vtable.
    std::set<std::string> own;
    for (const ConsumerDeclaration &consumer : config.consumers)
        own.insert(consumer.name);
    for (const SynthesizedDeclaration &synth : config.synthesized)
        own.insert(synth.name);
    if (own.size() != config.consumers.size() + config.synthesized.size())
        throw Error("duplicate consumer class");
    return config;
}

Layouts probeLayouts(const Declaration &config, const std::vector<std::string> &flags) {
    // Each row's class name and header: the parents, then the consumers, then the
    // synthesized classes. The probe derives a member-less subclass from each, with
    // an out-of-line destructor, so the compiler emits the subclass's vtable even
    // for an abstract parent. That vtable has exactly the parent's slots, each
    // relocated against the method in it, but for the destructors, which name the
    // subclass and map back to the parent.
    std::vector<std::pair<std::string, std::string>> rows;
    for (const auto &cls : config.classes)
        rows.emplace_back(cls.name, cls.header);
    for (const auto &consumer : config.consumers)
        rows.emplace_back(consumer.name, consumer.header);
    for (const auto &synth : config.synthesized)
        rows.emplace_back(synth.name, synth.header);
    Layouts layouts;
    if (rows.empty())
        return layouts;
    auto [image, dump] = compileProbe(probeSource(rows), flags);
    std::map<size_t, uint64_t> entries = vtableEntries(dump);
    const Symbol &sizes = image.symbol("_graft_layout_sizes");
    Bytes sizeBytes = image.read(sizes.value, 4 * rows.size());
    std::map<uint64_t, Relocation> relocations;
    for (const Relocation &relocation : image.relocations())
        relocations.insert_or_assign(relocation.address, relocation);
    for (size_t i = 0; i < rows.size(); ++i) {
        auto count = entries.find(i);
        ClassLayout layout{rows[i].first, load32(&sizeBytes[4 * i]),
                           probeSlots(image, i, rows[i].first,
                                      count == entries.end()
                                              ? std::nullopt
                                              : std::optional<uint64_t>(count->second),
                                      relocations)};
        if (i < config.classes.size())
            layouts.classes.push_back(std::move(layout));
        else if (i < config.classes.size() + config.consumers.size())
            layouts.consumers.push_back(std::move(layout));
        else
            layouts.synthesized.push_back(std::move(layout));
    }
    return layouts;
}

std::string layoutsText(const Layouts &layouts) {
    auto rows = [](const std::vector<ClassLayout> &classes) {
        std::vector<Json> out;
        for (const ClassLayout &layout : classes)
            out.push_back(layoutJson(layout));
        return Json::makeArray(std::move(out));
    };
    return dumpJson(Json::makeObject({{"classes", rows(layouts.classes)},
                                      {"consumers", rows(layouts.consumers)},
                                      {"synthesized", rows(layouts.synthesized)}})) +
           "\n";
}

Layouts readLayouts(const std::string &path) {
    Json json = parseJson(readText(path), path);
    if (json.kind != Json::Kind::Object)
        throw Error("layouts " + path + " are malformed");
    return {parseLayouts(json.find("classes"), "classes"),
            parseLayouts(json.find("consumers"), "consumers"),
            parseLayouts(json.find("synthesized"), "synthesized")};
}

Census census(const Image &image, const Declaration &config, const Layouts &layouts) {
    if (image.filetype != MH_KEXT_BUNDLE)
        throw Error("consumer must be linked as a kext");
    auto declaredNames = [](const auto &rows) {
        std::vector<std::string> out;
        for (const auto &row : rows)
            out.push_back(row.name);
        return out;
    };
    if (declaredNames(config.classes) != declaredNames(layouts.classes) ||
        declaredNames(config.consumers) != declaredNames(layouts.consumers) ||
        declaredNames(config.synthesized) != declaredNames(layouts.synthesized))
        throw Error("layouts describe another declaration");
    Census found;
    found.imports = selectImports(config, image);
    std::set<uint64_t> parentSlots;
    if (image.arch == "arm64e") {
        const std::vector<ChainedFixup> &fixups = image.fixups();
        Sited sited;
        for (const ChainedFixup &fixup : fixups)
            sited[fixup.address] = &fixup;
        found.vtables = consumerVtablesArm64e(image, config, layouts, sited, parentSlots);
        found.synth = consumerSynthesizedArm64e(image, config, layouts, sited, parentSlots);
        found.sites = importSitesArm64e(image, found.imports, fixups, parentSlots);
        found.segments = consumerSegments(image);
    } else {
        if (!config.synthesized.empty())
            throw Error("synthesis is supported only for arm64e");
        std::vector<Relocation> relocations = image.relocations();
        found.vtables = consumerVtables(image, config, layouts, relocations, parentSlots);
        found.sites = importSites(image, found.imports, relocations, parentSlots);
    }
    return found;
}

std::vector<uint32_t> requiredSlots(const ClassDeclaration &declaration, const ClassLayout &layout,
                                    const std::vector<const VtableRecord *> &vtables) {
    // A reserved slot, one of the kernel's _RESERVED<class><n> placeholders, is
    // not checked by default: the header keeps the placeholder's name where the
    // running kernel may have put a method in the slot, and no consumer calls one.
    // Graft still fills it from the live parent.
    std::set<uint32_t> slots;
    if (!declaration.required) {
        for (size_t slot = 0; slot < layout.slots.size(); ++slot) {
            if (!startsWith(memberName(layout.slots[slot]).value_or(""), "_RESERVED"))
                slots.insert(static_cast<uint32_t>(slot));
        }
    } else {
        for (const std::string &name : *declaration.required) {
            size_t matches = std::count(layout.slots.begin(), layout.slots.end(), name);
            if (matches != 1)
                throw Error("required inherited method has no unique compiler slot");
            slots.insert(static_cast<uint32_t>(
                    std::find(layout.slots.begin(), layout.slots.end(), name) -
                    layout.slots.begin()));
        }
    }
    for (const VtableRecord *vtable : vtables)
        slots.insert(vtable->overrides.begin(), vtable->overrides.end());
    return std::vector<uint32_t>(slots.begin(), slots.end());
}

Bytes packManifest(const Declaration &config, const Layouts &layouts, const Census &found,
                   const std::string &gate, const std::map<std::string, uint64_t> &placeholders,
                   uint32_t version) {
    StringTable strings;
    uint32_t gateOffset = strings.add(gate);
    uint32_t deviceOffset = strings.add(DeviceMatch);
    Rows libraries, imports, candidates, sites, classes, slots, vtables, overrides, segments, fills,
            selectors, synthOverrides, synthClasses;
    for (const LibraryDeclaration &library : config.libraries) {
        uint32_t id = strings.add(library.id);
        libraries.add(manifest::Library{id, library.anchor ? strings.add(*library.anchor) : 0});
    }
    std::map<std::string, uint32_t> importIndex;
    for (const auto &[name, record] : found.imports) {
        importIndex[name] = imports.count;
        imports.add(manifest::Import{static_cast<uint32_t>(record.library), candidates.count,
                                     static_cast<uint32_t>(record.candidates.size()),
                                     record.weak ? manifest::ImportWeak : 0, record.fallback});
        for (const std::string &candidate : record.candidates)
            candidates.add(strings.add(candidate));
    }
    for (const Site &site : found.sites) {
        auto placeholder = placeholders.find(site.symbol);
        uint8_t opcode = site.opcode;
        uint16_t reserved = 0;
        if (version == manifest::VersionArm64e)
            std::tie(opcode, reserved) = authTrailer(site.auth);
        sites.add(manifest::Site{site.offset,
                                 placeholder == placeholders.end() ? 0 : placeholder->second,
                                 site.addend, importIndex.at(site.symbol),
                                 static_cast<uint8_t>(site.width), opcode, reserved});
    }
    // A synthesized class's parent is measured when Graft binds, not checked slot
    // by slot: its probed slots are the stand-in's, so it records none.
    std::set<int64_t> synthParents;
    for (const SynthesizedDeclaration &synth : config.synthesized)
        synthParents.insert(synth.parent);
    requirePaired(config.classes.size(), layouts.classes.size());
    for (size_t i = 0; i < config.classes.size(); ++i) {
        const ClassDeclaration &declaration = config.classes[i];
        const ClassLayout &layout = layouts.classes[i];
        std::vector<uint32_t> required;
        if (!synthParents.count(static_cast<int64_t>(i))) {
            std::vector<const VtableRecord *> children;
            for (const VtableRecord &vtable : found.vtables) {
                if (vtable.parent == i)
                    children.push_back(&vtable);
            }
            required = requiredSlots(declaration, layout, children);
        }
        uint32_t first = slots.count;
        for (uint32_t slot : required)
            slots.add(manifest::Slot{slot, strings.add(layout.slots[slot])});
        uint32_t name = strings.add(declaration.name);
        uint32_t vtable = strings.add(vtableName(declaration.name));
        classes.add(manifest::Class{name, vtable, static_cast<uint32_t>(declaration.library),
                                    layout.size, static_cast<uint32_t>(layout.slots.size()), first,
                                    static_cast<uint32_t>(required.size())});
    }
    for (const VtableRecord &vtable : found.vtables) {
        vtables.add(manifest::Vtable{vtable.offset, vtable.parent, vtable.count, overrides.count,
                                     static_cast<uint32_t>(vtable.overrides.size()), fills.count,
                                     static_cast<uint32_t>(vtable.fills.size())});
        for (uint32_t slot : vtable.overrides)
            overrides.add(slot);
        for (const FillRecord &fill : vtable.fills)
            fills.add(manifest::Fill{fill.slot, fill.diversity, fill.opcode, 0});
    }
    for (const SegmentRecord &segment : found.segments)
        segments.add(manifest::Segment{segment.origin, segment.size, strings.add(segment.name), 0});
    for (const SynthClassRecord &record : found.synth) {
        uint32_t firstOverride = synthOverrides.count;
        for (const SynthOverrideRecord &override : record.overrides) {
            uint32_t firstSelector = selectors.count;
            for (const std::string &selector : override.selectors)
                selectors.add(strings.add(selector));
            synthOverrides.add(manifest::SynthOverride{
                    override.impl, firstSelector, static_cast<uint32_t>(override.selectors.size()),
                    override.diversity, override.opcode,
                    override.required ? manifest::SynthRequired : uint8_t{0}, 0});
        }
        uint32_t name = strings.add(record.name);
        uint32_t parent = strings.add(record.parentClass);
        uint32_t sibling = record.siblingClass.empty() ? 0 : strings.add(record.siblingClass);
        synthClasses.add(manifest::SynthClass{record.vtableOffset, name, parent, sibling,
                                              record.library, record.vtableSlotCap,
                                              record.parentRegion, firstOverride,
                                              static_cast<uint32_t>(record.overrides.size())});
    }

    // The header, then the string table, then each table's rows.
    const uint32_t headerSize = sizeof(manifest::Header);
    Bytes body = strings.data();
    manifest::Header header{};
    header.strings = {headerSize, static_cast<uint32_t>(strings.data().size())};
    const Rows *tables[] = {&libraries, &imports,        &candidates,  &sites,    &classes,
                            &slots,     &vtables,        &overrides,   &segments, &fills,
                            &selectors, &synthOverrides, &synthClasses};
    for (size_t i = 0; i < 13; ++i) {
        header.*ManifestTables[i + 1] = {headerSize + static_cast<uint32_t>(body.size()),
                                         tables[i]->count};
        body.insert(body.end(), tables[i]->bytes.begin(), tables[i]->bytes.end());
    }
    header.magic = manifest::Magic;
    header.version = version;
    header.size = headerSize + static_cast<uint32_t>(body.size());
    header.headerSize = headerSize;
    header.gate = gateOffset;
    header.device = deviceOffset;
    Bytes out;
    encode(out, header);
    out.insert(out.end(), body.begin(), body.end());
    // Graft validates the manifest before it uses any of it, and refuses one that
    // breaks a rule, so the build applies the same rules and refuses it first.
    manifest::View view;
    manifest::Failure failure = manifest::open(out.data(), out.size(), view);
    if (failure.error != manifest::Error::None)
        throw Error("Graft would refuse this manifest: invalid " + failureTable(failure.error) +
                    " record " + std::to_string(failure.record));
    return out;
}

std::string gateKey(const Plist &info) {
    return plistString(info, "CFBundleIdentifier") + GateSuffix;
}

uint64_t reservationSize(const Image &image, const Declaration &config, const Layouts &layouts,
                         const std::string &gate) {
    Census found = census(image, config, layouts);
    Bytes manifest = packManifest(config, layouts, found, gate, {}, manifestVersion(image));
    return manifest.size() + PlaceholderAlignment - 1 + PlaceholderSize * found.imports.size();
}

Image stageImage(const Image &image, const Declaration &config, const Layouts &layouts,
                 const std::string &gate) {
    const Section &section = image.section("__DATA", "__graft");
    // Staging writes the manifest over the reservation's bytes in the file. A
    // zerofill section has none, and its offset and size name no file range.
    if (!section.hasFileBytes())
        throw Error("__DATA,__graft is a zerofill section, with no bytes in the file to hold the "
                    "manifest");
    image.bounds(section.offset, section.size);
    uint32_t version = manifestVersion(image);
    Census found = census(image, config, layouts);
    Bytes sized = packManifest(config, layouts, found, gate, {}, version);
    uint64_t first = align(section.address + sized.size(), PlaceholderAlignment) - section.address;
    if (first + PlaceholderSize * found.imports.size() > section.size)
        throw Error("manifest reservation is too small; regenerate and relink");
    std::map<std::string, uint64_t> placeholders;
    uint64_t next = section.address - image.base + first;
    for (const auto &import : found.imports) {
        placeholders[import.first] = next;
        next += PlaceholderSize;
    }
    Bytes manifest = packManifest(config, layouts, found, gate, placeholders, version);
    Image staged = image;
    std::fill_n(staged.data.begin() + section.offset, section.size, 0);
    std::copy(manifest.begin(), manifest.end(), staged.data.begin() + section.offset);
    if (image.arch == "arm64e") {
        staged = rebindToPlaceholders(Image(std::move(staged.data)), placeholders);
    } else {
        std::map<std::string, std::pair<uint32_t, uint64_t>> definitions;
        for (const auto &[name, offset] : placeholders)
            definitions[name] = {section.index, image.base + offset};
        staged = Image(std::move(staged.data)).localized(definitions);
    }
    return staged.renamed(deferredSuperclassMarkers(staged, config));
}

void stageBundle(const std::string &kext, const Declaration &config, const Layouts &layouts,
                 const std::string &out) {
    fs::path source = kext, output = out;
    Plist info = parsePlist(readFile((source / "Contents/Info.plist").string()),
                            (source / "Contents/Info.plist").string());
    if (info.kind != Plist::Kind::Dict)
        throw Error("Info.plist is not a dictionary");
    std::string executable = plistString(info, "CFBundleExecutable");
    fs::path binary = source / "Contents/MacOS" / executable;
    info.set("IOKitPersonalities", gatedPersonalities(info));
    Image image = stageImage(Image(readFile(binary.string())), config, layouts, gateKey(info));
    Plist *dependencies = info.find("OSBundleLibraries");
    for (const LibraryDeclaration &library : config.libraries) {
        if (!dependencies || dependencies->kind != Plist::Kind::Dict ||
            !dependencies->find(library.id))
            throw Error("consumer does not declare a selected provider dependency");
        dependencies->erase(library.id);
    }
    // The kernel reads OSBundleLibraries_ and an architecture's name in place of
    // OSBundleLibraries on that architecture, so the declared libraries leave
    // those lists too.
    for (auto &[key, list] : info.dict) {
        if (key.rfind("OSBundleLibraries_", 0) != 0)
            continue;
        if (list.kind != Plist::Kind::Dict)
            throw Error("Info.plist " + key + " is not a dictionary");
        for (const LibraryDeclaration &library : config.libraries)
            list.erase(library.id);
    }
    if (fs::exists(output))
        throw Error("staging output already exists");
    // The bundle is staged beside its output and moved into place once whole, so
    // a staging that fails or is stopped leaves no output that looks current.
    fs::path partial = output.parent_path() / ("." + output.filename().string() + ".staging");
    std::error_code error;
    fs::create_directories(output.parent_path(), error);
    fs::remove_all(partial, error);
    try {
        fs::copy(source, partial, fs::copy_options::recursive, error);
        if (error)
            throw Error("cannot copy " + source.string() + " to " + partial.string() + ": " +
                        error.message());
        writeFile((partial / "Contents/Info.plist").string(), dumpPlist(info));
        writeFile((partial / "Contents/MacOS" / executable).string(), image.data);
        fs::rename(partial, output, error);
        if (error)
            throw Error("cannot move " + partial.string() + " to " + output.string() + ": " +
                        error.message());
    } catch (...) {
        fs::remove_all(partial, error);
        throw;
    }
}

} // namespace graft::tool
