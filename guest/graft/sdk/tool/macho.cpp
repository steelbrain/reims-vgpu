// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
#include "macho.h"

#include <algorithm>
#include <iterator>
#include <set>

namespace graft::tool {

namespace {

constexpr uint32_t CPU_SUBTYPE_ARM64E = 0x2;
// The capability bits above a CPU subtype.
constexpr uint32_t CPU_SUBTYPE_MASK = 0xff000000;
constexpr uint32_t LC_SYMTAB = 0x2;
constexpr uint32_t LC_DYSYMTAB = 0xb;
// A section's type is its flags' low byte; these types have no bytes in the file:
// S_ZEROFILL, S_GB_ZEROFILL and S_THREAD_LOCAL_ZEROFILL.
constexpr uint32_t SECTION_TYPE = 0xff;
constexpr uint32_t ZEROFILL_TYPES[] = {0x1, 0xc, 0x12};
// S_ATTR_PURE_INSTRUCTIONS or S_ATTR_SOME_INSTRUCTIONS: a section that holds code.
constexpr uint32_t INSTRUCTION_ATTRIBUTES = 0x80000000 | 0x400;
constexpr uint32_t NLIST_SIZE = 16;
constexpr uint8_t N_STAB = 0xe0;
constexpr uint8_t N_TYPE = 0x0e;
constexpr uint8_t N_UNDF = 0x0;
constexpr uint8_t N_EXT = 0x01;
constexpr uint32_t RELOCATION_SIZE = 8;
// Indirect-symbol entries that name no symbol.
constexpr uint32_t INDIRECT_SYMBOL_LOCAL = 0x80000000;
constexpr uint32_t INDIRECT_SYMBOL_ABS = 0x40000000;
// <mach-o/fixup-chains.h>. The tool reads only the one shape an arm64e kext link
// emits: imports as DYLD_CHAINED_IMPORT, sites as DYLD_CHAINED_PTR_ARM64E_KERNEL,
// whose stride is 4 and whose unauthenticated rebase target is a vm offset.
constexpr uint32_t DYLD_CHAINED_IMPORT = 1;
constexpr uint16_t DYLD_CHAINED_PTR_ARM64E_KERNEL = 7;
constexpr uint16_t DYLD_CHAINED_PTR_START_NONE = 0xffff;
constexpr uint16_t DYLD_CHAINED_PTR_START_MULTI = 0x8000;
constexpr uint64_t CHAINED_STRIDE = 4;

// LC_DYSYMTAB's symbol groups, in the order the symbol table must hold them.
enum Group { Local, External, Undefined };

int64_t signExtend(uint64_t value, unsigned bits) {
    uint64_t sign = uint64_t{1} << (bits - 1);
    return static_cast<int64_t>((value ^ sign) - sign);
}

} // namespace

bool Section::hasFileBytes() const {
    uint32_t type = flags & SECTION_TYPE;
    return std::find(std::begin(ZEROFILL_TYPES), std::end(ZEROFILL_TYPES), type) ==
           std::end(ZEROFILL_TYPES);
}

bool Section::executable() const {
    return flags & INSTRUCTION_ATTRIBUTES;
}

bool Symbol::defined() const {
    return !(kind & N_STAB) && (kind & N_TYPE) == N_SECT;
}

bool Symbol::undefined() const {
    return !(kind & N_STAB) && (kind & N_TYPE) == N_UNDF;
}

Image::Image(Bytes bytes) : data(std::move(bytes)) {
    bounds(0, MACH_HEADER_SIZE);
    uint32_t magic = u32(0), cpu = u32(4), subtype = u32(8);
    filetype = u32(12);
    if (cpu == CPU_TYPE_X86_64)
        arch = "x86_64";
    else if (cpu == CPU_TYPE_ARM64 && (subtype & ~CPU_SUBTYPE_MASK) == CPU_SUBTYPE_ARM64E)
        arch = "arm64e";
    bool bundle = filetype == MH_OBJECT || filetype == MH_KEXT_BUNDLE;
    if (magic != MH_MAGIC_64 || arch.empty() || !bundle)
        throw Error("expected thin x86-64 or arm64e object or kext");
    readLoadCommands(u32(16), u32(20));
    readSymbols();
    base = segments.empty() ? 0 : segments[0].address;
}

void Image::readLoadCommands(uint32_t count, uint32_t size) {
    bounds(MACH_HEADER_SIZE, size);
    uint64_t at = MACH_HEADER_SIZE, end = MACH_HEADER_SIZE + uint64_t{size};
    for (uint32_t i = 0; i < count; ++i) {
        if (end - at < 8)
            throw Error("truncated load command");
        uint32_t command = u32(at), length = u32(at + 4);
        if (length < 8 || length % 8 || length > end - at)
            throw Error("invalid load command");
        if (command == LC_SEGMENT_64) {
            readSegment(at, length);
        } else if (command == LC_SYMTAB) {
            if (length != 24 || symtab_)
                throw Error("invalid symbol-table command");
            symtab_ = SymbolTable{u32(at + 8), u32(at + 12), u32(at + 16), u32(at + 20)};
        } else if (command == LC_DYSYMTAB) {
            if (length != 80 || dysymtab_)
                throw Error("invalid dynamic symbol-table command");
            std::vector<uint32_t> fields(DynamicFields);
            for (size_t field = 0; field < fields.size(); ++field)
                fields[field] = u32(at + 8 + 4 * field);
            dysymtab_ = fields;
            dysymtabOffset_ = at;
        } else if (command == LC_DYLD_CHAINED_FIXUPS) {
            if (length != 16 || chainedFixups_)
                throw Error("invalid chained-fixups command");
            uint32_t dataoff = u32(at + 8), datasize = u32(at + 12);
            bounds(dataoff, datasize);
            chainedFixups_ = std::make_pair(dataoff, datasize);
        }
        at += length;
    }
    if (at != end || !symtab_)
        throw Error("incomplete load commands or symbol table");
}

void Image::readSegment(uint64_t at, uint32_t length) {
    if (length < SEGMENT_COMMAND_SIZE)
        throw Error("truncated segment command");
    Segment segment{fixedString(at + 8), u64(at + 24), u64(at + 32), u64(at + 40), u64(at + 48)};
    uint32_t nsects = u32(at + 64);
    if (nsects > (length - SEGMENT_COMMAND_SIZE) / SECTION_HEADER_SIZE)
        throw Error("sections exceed segment command");
    bounds(segment.offset, segment.filesize);
    segments.push_back(segment);
    for (uint32_t i = 0; i < nsects; ++i)
        readSection(at + SEGMENT_COMMAND_SIZE + uint64_t{i} * SECTION_HEADER_SIZE);
}

void Image::readSection(uint64_t at) {
    bounds(at, SECTION_HEADER_SIZE);
    Section section{
            fixedString(at + 16), fixedString(at), u64(at + 32),
            u64(at + 40),         u32(at + 48),    u32(at + 56),
            u32(at + 60),         u32(at + 64),    static_cast<uint32_t>(sections.size() + 1)};
    if (section.hasFileBytes())
        bounds(section.offset, section.size);
    bounds(section.relocationOffset, uint64_t{section.relocationCount} * RELOCATION_SIZE);
    sections.push_back(section);
}

void Image::readSymbols() {
    auto [symoff, nsyms, stroff, strsize] = *symtab_;
    bounds(symoff, uint64_t{nsyms} * NLIST_SIZE);
    bounds(stroff, strsize);
    for (uint32_t index = 0; index < nsyms; ++index) {
        uint64_t at = symoff + uint64_t{index} * NLIST_SIZE;
        uint32_t strx = u32(at);
        if (strx >= strsize)
            throw Error("symbol name outside string table");
        Symbol symbol{string(uint64_t{stroff} + strx, strsize - strx),
                      u8(at + 4),
                      u8(at + 5),
                      u16(at + 6),
                      u64(at + 8),
                      index,
                      at,
                      strx};
        if (symbol.defined() && !(1 <= symbol.section && symbol.section <= sections.size()))
            throw Error("defined symbol has invalid section");
        symbols.push_back(symbol);
    }
}

void Image::bounds(uint64_t offset, uint64_t size) const {
    if (offset > data.size() || size > data.size() - offset)
        throw Error("Mach-O range outside file");
}

uint8_t Image::u8(uint64_t offset) const {
    bounds(offset, 1);
    return data[offset];
}

uint16_t Image::u16(uint64_t offset) const {
    bounds(offset, 2);
    return load16(&data[offset]);
}

uint32_t Image::u32(uint64_t offset) const {
    bounds(offset, 4);
    return load32(&data[offset]);
}

uint64_t Image::u64(uint64_t offset) const {
    bounds(offset, 8);
    return load64(&data[offset]);
}

// A 16-byte NUL-padded ASCII name, such as a segment's or a section's.
std::string Image::fixedString(uint64_t offset) const {
    bounds(offset, 16);
    std::string name;
    for (uint64_t i = offset; i < offset + 16 && data[i]; ++i) {
        if (data[i] >= 0x80)
            throw Error("invalid Mach-O fixed string");
        name += static_cast<char>(data[i]);
    }
    return name;
}

// A NUL-terminated UTF-8 string that must end within size bytes.
std::string Image::string(uint64_t offset, uint64_t size) const {
    bounds(offset, size);
    auto first = data.begin() + static_cast<ptrdiff_t>(offset);
    auto end = std::find(first, first + static_cast<ptrdiff_t>(size), 0);
    if (end == first + static_cast<ptrdiff_t>(size))
        throw Error("unterminated Mach-O string");
    std::string text(first, end);
    if (!validUtf8(text))
        throw Error("invalid symbol string");
    return text;
}

const Section &Image::section(const std::string &segment, const std::string &name) const {
    const Section *found = nullptr;
    size_t count = 0;
    for (const Section &section : sections) {
        if (section.segment == segment && section.name == name) {
            found = &section;
            ++count;
        }
    }
    if (count != 1)
        throw Error("expected one " + segment + "," + name + " section");
    return *found;
}

uint64_t Image::fileOffset(uint64_t address, uint64_t width) const {
    for (const Section &section : sections) {
        if (!section.hasFileBytes())
            continue;
        if (section.address <= address && width <= section.size &&
            address - section.address <= section.size - width) {
            uint64_t offset = section.offset + (address - section.address);
            bounds(offset, width);
            return offset;
        }
    }
    throw Error("address is outside file-backed sections");
}

Bytes Image::read(uint64_t address, uint64_t width) const {
    uint64_t offset = fileOffset(address, width);
    return Bytes(data.begin() + static_cast<ptrdiff_t>(offset),
                 data.begin() + static_cast<ptrdiff_t>(offset + width));
}

const Symbol &Image::symbol(const std::string &name) const {
    const Symbol *found = nullptr;
    size_t count = 0;
    for (const Symbol &symbol : symbols) {
        if (symbol.name == name && symbol.defined()) {
            found = &symbol;
            ++count;
        }
    }
    if (count != 1)
        throw Error("expected one definition of " + name);
    return *found;
}

uint64_t Image::symbolExtent(const Symbol &symbol) const {
    const Section &section = sections[symbol.section - 1];
    uint64_t sectionEnd = section.address + section.size;
    uint64_t end = sectionEnd;
    bool following = false;
    for (const Symbol &other : symbols) {
        if (other.defined() && other.section == symbol.section && other.value > symbol.value &&
            (!following || other.value < end)) {
            end = other.value;
            following = true;
        }
    }
    if (symbol.value < section.address || end > sectionEnd)
        throw Error("symbol outside owning section");
    return end - symbol.value;
}

void Image::readRelocations(uint64_t offset, uint32_t count, uint64_t base,
                            std::vector<Relocation> &out) const {
    bounds(offset, uint64_t{count} * RELOCATION_SIZE);
    for (uint32_t i = 0; i < count; ++i) {
        uint64_t at = offset + uint64_t{i} * RELOCATION_SIZE;
        // r_address, whose top bit is r_scattered, then r_symbolnum:24, r_pcrel:1,
        // r_length:2, r_extern:1 and r_type:4.
        int32_t address = static_cast<int32_t>(u32(at));
        uint32_t info = u32(at + 4);
        if (address < 0)
            throw Error("scattered relocation is unsupported");
        uint32_t symbolIndex = info & 0xffffff;
        bool external = info >> 27 & 1;
        if (external && symbolIndex >= symbols.size())
            throw Error("relocation symbol index outside table");
        out.push_back(Relocation{base + static_cast<uint32_t>(address), symbolIndex,
                                 static_cast<bool>(info >> 24 & 1), 1u << (info >> 25 & 3),
                                 external, info >> 28, at});
    }
}

std::vector<Relocation> Image::relocations() const {
    std::vector<Relocation> out;
    if (filetype == MH_OBJECT) {
        for (const Section &section : sections)
            readRelocations(section.relocationOffset, section.relocationCount, section.address,
                            out);
        return out;
    }
    if (!dysymtab_)
        throw Error("kext has no dynamic symbol table");
    const std::vector<uint32_t> &table = *dysymtab_;
    readRelocations(table[extreloff], table[nextrel], base, out);
    readRelocations(table[locreloff], table[nlocrel], base, out);
    return out;
}

const std::vector<ChainedFixup> &Image::fixups() const {
    if (arch != "arm64e")
        throw Error("chained fixups are read only from an arm64e image");
    if (!chainedFixups_)
        throw Error("arm64e kext has no chained fixups");
    // The census reads the fixups several times; an edit makes a new Image, so
    // they are decoded once.
    if (fixups_)
        return *fixups_;
    // Every table the header names lies inside the fixups data, which the load
    // command's bounds already keep inside the file.
    uint64_t blob = chainedFixups_->first, end = blob + chainedFixups_->second;
    inside(blob, 28, end);
    uint32_t version = u32(blob), startsOffset = u32(blob + 4), importsOffset = u32(blob + 8);
    uint32_t symbolsOffset = u32(blob + 12), count = u32(blob + 16);
    uint32_t importsFormat = u32(blob + 20), symbolsFormat = u32(blob + 24);
    if (version != 0 || importsFormat != DYLD_CHAINED_IMPORT || symbolsFormat != 0)
        throw Error("unsupported chained-fixups header");
    inside(blob + symbolsOffset, 0, end);
    std::vector<ChainedImport> imports =
            readChainedImports(blob + importsOffset, count, blob + symbolsOffset, end);
    std::vector<ChainedFixup> fixups;
    uint64_t starts = blob + startsOffset;
    inside(starts, 4, end);
    uint32_t segmentCount = u32(starts);
    inside(starts + 4, uint64_t{segmentCount} * 4, end);
    for (uint32_t i = 0; i < segmentCount; ++i) {
        uint32_t offset = u32(starts + 4 + uint64_t{i} * 4);
        if (offset)
            walkSegment(starts + offset, end, imports, fixups);
    }
    fixups_ = std::move(fixups);
    return *fixups_;
}

// The name and weakness of each import, by ordinal, which a bind indexes.
void Image::inside(uint64_t offset, uint64_t size, uint64_t end) const {
    if (offset > end || size > end - offset)
        throw Error("chained fixups reach outside their data");
}

std::vector<Image::ChainedImport> Image::readChainedImports(uint64_t offset, uint32_t count,
                                                            uint64_t symbols, uint64_t end) const {
    inside(offset, uint64_t{count} * 4, end);
    std::vector<ChainedImport> imports;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t entry = u32(offset + uint64_t{i} * 4);
        // lib_ordinal:8, weak_import:1, name_offset:23
        bool weak = entry >> 8 & 1;
        uint64_t nameOffset = entry >> 9;
        if (symbols + nameOffset > end)
            throw Error("chained import name outside the fixups data");
        imports.push_back({string(symbols + nameOffset, end - symbols - nameOffset), weak});
    }
    return imports;
}

// Walk one segment's page chains, appending each decoded fixup. Its starts lie
// inside the fixups data, which ends at end.
void Image::walkSegment(uint64_t offset, uint64_t end, const std::vector<ChainedImport> &imports,
                        std::vector<ChainedFixup> &fixups) const {
    // dyld_chained_starts_in_segment: size, page_size, pointer_format,
    // segment_offset, max_valid_pointer, page_count, then page_start[page_count].
    inside(offset, 22, end);
    uint32_t size = u32(offset);
    uint16_t pageSize = u16(offset + 4), pointerFormat = u16(offset + 6);
    uint64_t segmentOffset = u64(offset + 8);
    uint16_t pageCount = u16(offset + 20);
    if (pointerFormat != DYLD_CHAINED_PTR_ARM64E_KERNEL)
        throw Error("unsupported chained pointer format");
    inside(offset, size, end);
    inside(offset + 22, uint64_t{pageCount} * 2, end);
    for (uint16_t page = 0; page < pageCount; ++page) {
        uint16_t start = u16(offset + 22 + uint64_t{page} * 2);
        if (start == DYLD_CHAINED_PTR_START_NONE)
            continue;
        // A page's one chain starts inside it; a list of starts is a format a kext
        // never uses.
        if (start & DYLD_CHAINED_PTR_START_MULTI)
            throw Error("chained page with several starts is unsupported");
        if (start >= pageSize)
            throw Error("chained page start outside its page");
        walkChain(base + segmentOffset + uint64_t{page} * pageSize + start, imports, fixups);
    }
}

// Walk one chain from address, decoding each DYLD_CHAINED_PTR_ARM64E_KERNEL site,
// until a site's next is 0.
void Image::walkChain(uint64_t address, const std::vector<ChainedImport> &imports,
                      std::vector<ChainedFixup> &fixups) const {
    for (;;) {
        uint64_t entryOffset = fileOffset(address, 8);
        uint64_t raw = load64(&data[entryOffset]);
        bool authenticated = raw >> 63 & 1;
        std::optional<AuthContext> context;
        if (authenticated)
            context = AuthContext{static_cast<uint32_t>(raw >> 49 & 3),
                                  static_cast<uint32_t>(raw >> 32 & 0xffff),
                                  static_cast<bool>(raw >> 48 & 1)};
        if (raw >> 62 & 1) {
            uint64_t ordinal = raw & 0xffff;
            if (ordinal >= imports.size())
                throw Error("chained bind ordinal outside the import table");
            int64_t addend = authenticated ? 0 : signExtend(raw >> 32 & 0x7ffff, 19);
            fixups.push_back({address, entryOffset, true, imports[ordinal].name,
                              imports[ordinal].weak, addend, 0, context});
        } else {
            // A rebase's target is a vm offset from the image base. An unsigned
            // one's high8, a top byte for the pointer, is never set in a kext.
            if (!authenticated && raw >> 43 & 0xff)
                throw Error("chained rebase with a top byte is unsupported");
            uint64_t target = base + (authenticated ? raw & 0xffffffff : raw & 0x7ffffffffff);
            fixups.push_back({address, entryOffset, false, "", false, 0, target, context});
        }
        uint64_t next = raw >> 51 & 0x7ff;
        if (!next)
            return;
        address += next * CHAINED_STRIDE;
    }
}

Image Image::localized(
        const std::map<std::string, std::pair<uint32_t, uint64_t>> &definitions) const {
    if (filetype != MH_KEXT_BUNDLE || !dysymtab_)
        throw Error("localization requires a linked kext");
    std::vector<uint32_t> table = *dysymtab_;
    // Module, table-of-contents and reference tables belong to dylibs, never a kext.
    if (table[ntoc] || table[nmodtab] || table[nextrefsyms])
        throw Error("unsupported dynamic symbol metadata");
    // Each nlist entry as its group, old index and bytes, in its new order.
    struct Entry {
        Group group;
        uint32_t index;
        Bytes bytes;
    };
    std::vector<Entry> entries;
    std::set<std::string> missing;
    for (const auto &definition : definitions)
        missing.insert(definition.first);
    for (const Symbol &symbol : symbols) {
        Bytes entry(data.begin() + static_cast<ptrdiff_t>(symbol.entryOffset),
                    data.begin() + static_cast<ptrdiff_t>(symbol.entryOffset + NLIST_SIZE));
        Group group;
        auto definition = definitions.find(symbol.name);
        if (definition != definitions.end()) {
            if (!symbol.undefined())
                throw Error("placeholder names an already defined symbol");
            auto [section, value] = definition->second;
            if (!(1 <= section && section <= sections.size()))
                throw Error("placeholder section outside image");
            store32(&entry[0], symbol.nameOffset);
            entry[4] = N_SECT;
            entry[5] = static_cast<uint8_t>(section);
            store16(&entry[6], 0);
            store64(&entry[8], value);
            missing.erase(symbol.name);
            group = Local;
        } else if (symbol.undefined()) {
            group = Undefined;
        } else {
            group = symbol.kind & N_EXT ? External : Local;
        }
        entries.push_back({group, symbol.index, std::move(entry)});
    }
    if (!missing.empty())
        throw Error("placeholder has no undefined symbol");
    std::sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) {
        return std::make_pair(a.group, a.index) < std::make_pair(b.group, b.index);
    });
    std::vector<uint32_t> renumbered(symbols.size());
    for (size_t index = 0; index < entries.size(); ++index)
        renumbered[entries[index].index] = static_cast<uint32_t>(index);

    Image out = *this;
    for (const Relocation &relocation : relocations()) {
        if (!relocation.external)
            continue;
        uint64_t at = relocation.entryOffset + 4;
        uint32_t info = out.u32(at);
        store32(&out.data[at], (info & 0xff000000) | renumbered[relocation.symbolIndex]);
    }
    uint64_t indirect = table[indirectsymoff], indirectCount = table[nindirectsyms];
    bounds(indirect, indirectCount * 4);
    for (uint64_t i = 0; i < indirectCount; ++i) {
        uint64_t at = indirect + i * 4;
        uint32_t index = out.u32(at);
        if (!(index & (INDIRECT_SYMBOL_LOCAL | INDIRECT_SYMBOL_ABS))) {
            if (index >= symbols.size())
                throw Error("indirect symbol index outside table");
            store32(&out.data[at], renumbered[index]);
        }
    }
    for (size_t index = 0; index < entries.size(); ++index)
        std::copy(entries[index].bytes.begin(), entries[index].bytes.end(),
                  out.data.begin() + static_cast<ptrdiff_t>(symtab_->symoff + index * NLIST_SIZE));
    uint32_t locals = 0, externals = 0;
    for (const Entry &entry : entries) {
        locals += entry.group == Local;
        externals += entry.group == External;
    }
    table[ilocalsym] = 0;
    table[nlocalsym] = locals;
    table[iextdefsym] = locals;
    table[nextdefsym] = externals;
    table[iundefsym] = locals + externals;
    table[nundefsym] = static_cast<uint32_t>(entries.size()) - locals - externals;
    for (size_t field = 0; field < table.size(); ++field)
        store32(&out.data[dysymtabOffset_ + 8 + 4 * field], table[field]);
    return Image(std::move(out.data));
}

Image Image::renamed(const std::vector<std::pair<std::string, std::string>> &names) const {
    Bytes edited = data;
    uint64_t stroff = symtab_->stroff;
    for (const auto &[old, replacement] : names) {
        if (replacement.size() != old.size() || replacement.find('\0') != std::string::npos)
            throw Error("marker rename must preserve its byte length");
        bool matched = false;
        for (const Symbol &symbol : symbols) {
            if (symbol.name != old)
                continue;
            matched = true;
            uint64_t start = stroff + symbol.nameOffset, end = start + replacement.size();
            // A name whose bytes another symbol's name shares is refused rather
            // than renaming that symbol too.
            for (const Symbol &other : symbols) {
                if (other.name == old)
                    continue;
                uint64_t otherStart = stroff + other.nameOffset;
                uint64_t otherEnd = otherStart + other.name.size();
                if (start < otherEnd && otherStart < end)
                    throw Error("marker string shares storage with another symbol");
            }
            std::copy(replacement.begin(), replacement.end(),
                      edited.begin() + static_cast<ptrdiff_t>(start));
        }
        if (!matched)
            throw Error("marker rename names no symbol");
    }
    return Image(std::move(edited));
}

} // namespace graft::tool
