// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Read and edit the thin x86-64 and arm64e Mach-O files of a consumer's build.
//
// The consumer steps read compiler objects and linked kexts through Image, and
// stage a kext by editing its symbol table (Image::localized, Image::renamed);
// the prelink check reads staged kexts. Every offset is checked against the
// file before it is read or written. Image handles what Xcode's clang and ld
// emit for a kext, not arbitrary architectures or linkers, and throws Error for
// anything else rather than guessing.
//
// An x86-64 kext records its references to other images as classic
// relocations (Image::relocations). An arm64e kext records them as chains of
// fixups (LC_DYLD_CHAINED_FIXUPS, Image::fixups), whose authenticated sites
// carry their own signing context. Image reads the chains a kext's link writes,
// one starting inside each page and no rebase with a top byte, and every table
// of them it reads lies inside the fixups data.
#pragma once
#include "base.h"

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace graft::tool {

// <mach-o/loader.h> and <mach/machine.h>
constexpr uint32_t MH_MAGIC_64 = 0xfeedfacf;
constexpr uint32_t CPU_TYPE_X86_64 = 0x01000007;
constexpr uint32_t CPU_TYPE_ARM64 = 0x0100000c;
constexpr uint32_t MH_OBJECT = 0x1;
constexpr uint32_t MH_KEXT_BUNDLE = 0xb;
constexpr uint32_t MH_FILESET = 0xc;
constexpr uint32_t MACH_HEADER_SIZE = 32;
constexpr uint32_t LC_SEGMENT_64 = 0x19;
constexpr uint32_t LC_DYLD_CHAINED_FIXUPS = 0x80000034;
constexpr uint32_t LC_FILESET_ENTRY = 0x80000035;
constexpr uint32_t SEGMENT_COMMAND_SIZE = 72;
constexpr uint32_t SECTION_HEADER_SIZE = 80;
// <mach-o/nlist.h>: an undefined symbol's n_desc bit for __attribute__((weak_import)).
constexpr uint16_t N_WEAK_REF = 0x0040;
constexpr uint8_t N_SECT = 0x0e;
// <mach-o/x86_64/reloc.h> and <mach-o/arm64/reloc.h>
constexpr uint32_t X86_64_RELOC_UNSIGNED = 0;
constexpr uint32_t X86_64_RELOC_BRANCH = 2;
// An arm64e object relocates each vtable slot as a signed pointer.
constexpr uint32_t ARM64_RELOC_AUTHENTICATED_POINTER = 11;

// An LC_SEGMENT_64 command.
struct Segment {
    std::string name;
    uint64_t address, vmsize, offset, filesize;
};

// A section header. index counts from 1, as nlist's n_sect does.
struct Section {
    std::string segment, name;
    uint64_t address, size;
    uint32_t offset, relocationOffset, relocationCount, flags;
    uint32_t index;

    bool hasFileBytes() const;
    bool executable() const;
};

// An nlist_64 entry: index is its position in the symbol table, entryOffset its
// file offset, and nameOffset its name's offset in the string table.
struct Symbol {
    std::string name;
    uint8_t kind, section;
    uint16_t descriptor;
    uint64_t value;
    uint32_t index;
    uint64_t entryOffset;
    uint32_t nameOffset;

    // Defined in a section of this image, and not a debugging entry.
    bool defined() const;
    bool undefined() const;
};

// A relocation_info entry with its address resolved to an image address: kind is
// its r_type, width the bytes it rewrites, entryOffset the entry's file offset.
struct Relocation {
    uint64_t address;
    uint32_t symbolIndex;
    bool pcrel;
    uint32_t width;
    bool external;
    uint32_t kind;
    uint64_t entryOffset;
};

// An arm64e site's pointer-authentication context: key (0=IA 1=IB 2=DA 3=DB),
// the 16-bit diversity and whether the slot's address is blended into it.
struct AuthContext {
    uint32_t key, diversity;
    bool addrDiv;
};

// One arm64e chained-fixup site: an 8-byte slot the loader resolves. A bind
// names an imported symbol, with an addend; a rebase targets an address in the
// image. address is the slot's image address, entryOffset its file offset.
struct ChainedFixup {
    uint64_t address, entryOffset;
    bool bind;
    std::string symbol;
    bool weak;
    int64_t addend;
    uint64_t target;
    std::optional<AuthContext> auth;
};

// A thin x86-64 or arm64e MH_OBJECT or MH_KEXT_BUNDLE, parsed from its bytes.
// arch is "x86_64" or "arm64e". base is the first segment's address, from which
// the manifest's image offsets count.
class Image {
public:
    explicit Image(Bytes bytes);

    Bytes data;
    std::string arch;
    uint32_t filetype = 0;
    uint64_t base = 0;
    std::vector<Segment> segments;
    std::vector<Section> sections;
    std::vector<Symbol> symbols;

    // The one section named segment,name.
    const Section &section(const std::string &segment, const std::string &name) const;
    // The file offset of width bytes at address, which one section's file bytes hold.
    uint64_t fileOffset(uint64_t address, uint64_t width) const;
    Bytes read(uint64_t address, uint64_t width) const;
    // The one defined symbol named name.
    const Symbol &symbol(const std::string &name) const;
    // The bytes from a defined symbol to the next one in its section, or to its end.
    uint64_t symbolExtent(const Symbol &symbol) const;
    // Every relocation: an object's section by section, a kext's external then local ones.
    std::vector<Relocation> relocations() const;
    // Every arm64e chained fixup, an import bind or a local rebase, in chain order.
    const std::vector<ChainedFixup> &fixups() const;

    // The image with the named undefined symbols turned into private local
    // definitions, each at its (section index, address). The symbol table is
    // regrouped as LC_DYSYMTAB requires (locals, external definitions, then
    // undefined symbols), and every external relocation and indirect-symbol entry
    // is renumbered to match.
    Image localized(const std::map<std::string, std::pair<uint32_t, uint64_t>> &definitions) const;
    // The image with symbols renamed in the string table: each new name has its
    // old name's byte length, so no offset moves.
    Image renamed(const std::vector<std::pair<std::string, std::string>> &names) const;

    // Throws Error unless size bytes at offset lie inside the file.
    void bounds(uint64_t offset, uint64_t size) const;
    uint8_t u8(uint64_t offset) const;
    uint16_t u16(uint64_t offset) const;
    uint32_t u32(uint64_t offset) const;
    uint64_t u64(uint64_t offset) const;

private:
    struct SymbolTable {
        uint32_t symoff, nsyms, stroff, strsize;
    };
    // LC_DYSYMTAB's 18 fields after cmd and cmdsize, by <mach-o/loader.h>'s names.
    enum DynamicField {
        ilocalsym,
        nlocalsym,
        iextdefsym,
        nextdefsym,
        iundefsym,
        nundefsym,
        tocoff,
        ntoc,
        modtaboff,
        nmodtab,
        extrefsymoff,
        nextrefsyms,
        indirectsymoff,
        nindirectsyms,
        extreloff,
        nextrel,
        locreloff,
        nlocrel,
        DynamicFields
    };
    struct ChainedImport {
        std::string name;
        bool weak;
    };

    void readLoadCommands(uint32_t count, uint32_t size);
    void readSegment(uint64_t at, uint32_t length);
    void readSection(uint64_t at);
    void readSymbols();
    std::string fixedString(uint64_t offset) const;
    std::string string(uint64_t offset, uint64_t size) const;
    void readRelocations(uint64_t offset, uint32_t count, uint64_t base,
                         std::vector<Relocation> &out) const;
    std::vector<ChainedImport> readChainedImports(uint64_t offset, uint32_t count, uint64_t symbols,
                                                  uint64_t end) const;
    // Throws Error unless size bytes at offset lie inside the chained-fixups
    // data, which ends at end.
    void inside(uint64_t offset, uint64_t size, uint64_t end) const;
    void walkSegment(uint64_t offset, uint64_t end, const std::vector<ChainedImport> &imports,
                     std::vector<ChainedFixup> &fixups) const;
    void walkChain(uint64_t address, const std::vector<ChainedImport> &imports,
                   std::vector<ChainedFixup> &fixups) const;

    std::optional<SymbolTable> symtab_;
    std::optional<std::vector<uint32_t>> dysymtab_;
    uint64_t dysymtabOffset_ = 0;
    std::optional<std::pair<uint32_t, uint32_t>> chainedFixups_;
    mutable std::optional<std::vector<ChainedFixup>> fixups_;
};

} // namespace graft::tool
