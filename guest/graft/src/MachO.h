// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Graft's reader of a mapped Mach-O image: its load commands, fileset entries
// and symbol table (docs/design.md, "Reading kernel images"). It is pure logic
// with no kernel calls, so the kext and the host tests (tests/macho_test.cpp)
// share it.
//
// A library can hold several local copies of one name, one per unit that
// compiled it. Lookup is duplicate-aware: definesAt tests a name at a given
// address, and the binding's lookup by name (Resolve.h) refuses a name defined
// at two addresses rather than pick the first copy silently.
//
// The Mach-O structures come from <mach-o/loader.h> and <mach-o/nlist.h>,
// which Xcode's Kernel.framework and the host SDK both provide.
#pragma once
#include <mach-o/loader.h>
#include <mach-o/nlist.h>
#include <mach/vm_prot.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

namespace graft {
namespace macho {

// The most load-command bytes Graft walks in one image. The walk is bounded by
// the header's own sizeofcmds (the image carries no length of its own in
// memory), so a header claiming more than this is refused rather than walked. No
// kernel collection or kext Graft reads approaches it; a larger one is rejected.
constexpr uint32_t MaxLoadCommands = 65536;

// The header at `base` if it is a 64-bit Mach-O, else null. `base` is where
// the image is mapped: a file on the host, or an image in the running kernel.
inline const mach_header_64 *header(const void *base) {
    auto mh = reinterpret_cast<const mach_header_64 *>(base);
    return (mh && mh->magic == MH_MAGIC_64) ? mh : nullptr;
}

// Validate the entire command list before using a table discovered in it.
// A matching command followed by a malformed command must not be accepted.
inline bool commandsValid(const mach_header_64 *mh) {
    if (!mh || mh->magic != MH_MAGIC_64)
        return false;
    size_t offset = 0;
    auto bytes = reinterpret_cast<const uint8_t *>(mh + 1);
    for (uint32_t i = 0; i < mh->ncmds; ++i) {
        if (mh->sizeofcmds - offset < sizeof(load_command))
            return false;
        load_command lc;
        memcpy(&lc, bytes + offset, sizeof(lc));
        if (lc.cmdsize < sizeof(lc) || lc.cmdsize % 8 || lc.cmdsize > mh->sizeofcmds - offset)
            return false;
        offset += lc.cmdsize;
    }
    return offset == mh->sizeofcmds;
}

// The first load command for which `match` returns true, else null; also null
// at the first command that runs past `sizeofcmds` or is too small to be one.
// Bounds the walk by the header's own `sizeofcmds`, as the image carries no
// length of its own in memory.
template <typename Match>
inline const load_command *eachCommand(const mach_header_64 *mh, Match match) {
    auto at = reinterpret_cast<const uint8_t *>(mh + 1);
    auto end = at + mh->sizeofcmds;
    for (uint32_t i = 0; i < mh->ncmds; ++i) {
        auto lc = reinterpret_cast<const load_command *>(at);
        auto left = static_cast<size_t>(end - at);
        if (left < sizeof(load_command) || lc->cmdsize < sizeof(load_command) || lc->cmdsize > left)
            return nullptr;
        if (match(lc))
            return lc;
        at += lc->cmdsize;
    }
    return nullptr;
}

// The LC_SEGMENT_64 named `name` (at most 16 chars, e.g. SEG_LINKEDIT), or null.
inline const segment_command_64 *segment(const mach_header_64 *mh, const char *name) {
    if (!commandsValid(mh))
        return nullptr;
    auto lc = eachCommand(mh, [name](const load_command *lc) {
        return lc->cmd == LC_SEGMENT_64 && lc->cmdsize >= sizeof(segment_command_64) &&
               !strncmp(reinterpret_cast<const segment_command_64 *>(lc)->segname, name, 16);
    });
    return reinterpret_cast<const segment_command_64 *>(lc);
}

inline const section_64 *section(const mach_header_64 *mh, const char *segname, const char *name) {
    auto seg = segment(mh, segname);
    if (!seg || seg->nsects > (seg->cmdsize - sizeof(*seg)) / sizeof(section_64))
        return nullptr;
    auto sections = reinterpret_cast<const section_64 *>(seg + 1);
    for (uint32_t i = 0; i < seg->nsects; ++i) {
        if (!strncmp(sections[i].sectname, name, 16) && !strncmp(sections[i].segname, segname, 16))
            return sections + i;
    }
    return nullptr;
}

// Whether [address, address+size) lies wholly in one segment of `mh`, slid by
// `bias`, that accepts a store: its protection grants write, it is not locked
// read-only (SG_READ_ONLY), and it is not a __TEXT* segment. A store to a
// read-only segment (such as __DATA_CONST) faults once the kernel locks it, so a
// write target is checked against this first. Pure logic over the header; the
// caller owns the mapping and has already bounded the span.
inline bool writableSegment(const mach_header_64 *mh, uint64_t bias, uint64_t address,
                            uint64_t size) {
    if (!mh || !size || size - 1 > UINT64_MAX - address)
        return false;
    return eachCommand(mh, [&](const load_command *lc) {
               if (lc->cmd != LC_SEGMENT_64 || lc->cmdsize < sizeof(segment_command_64))
                   return false;
               auto seg = reinterpret_cast<const segment_command_64 *>(lc);
               if (!(seg->initprot & VM_PROT_WRITE) || (seg->flags & SG_READ_ONLY) ||
                   !strncmp(seg->segname, SEG_TEXT, sizeof(SEG_TEXT) - 1) ||
                   seg->vmaddr > UINT64_MAX - bias)
                   return false;
               uint64_t start = seg->vmaddr + bias;
               return address >= start && address - start <= seg->vmsize &&
                      size <= seg->vmsize - (address - start);
           }) != nullptr;
}

// A mapped symbol table: the nlist array and the string table, already at the
// addresses where they are read. Built by openSymbols.
struct SymbolTable {
    const struct nlist_64 *list = nullptr;
    uint32_t count = 0;
    const char *strings = nullptr;
    uint32_t stringsSize = 0;
    uint64_t bias = 0;
    bool valid() const { return list && strings; }
};

inline uint64_t symbolAddress(const SymbolTable &table, uint32_t index) {
    uint64_t value = table.list[index].n_value;
    return value <= UINT64_MAX - table.bias ? value + table.bias : 0;
}

// Fill `out` from `mh`'s LC_SYMTAB and __LINKEDIT, given `linkeditBase`: the
// address where the bytes at __LINKEDIT's file offset are mapped. On the host
// that is the file's base plus __LINKEDIT's fileoff; in the running kernel it
// is __LINKEDIT's vmaddr. Returns false, leaving
// `out` empty, unless both commands are present and the tables lie within
// __LINKEDIT's file-backed bytes. Those bytes are mapped only up to vmsize, so
// a segment whose filesize exceeds its vmsize is refused. The header cannot
// show residency: the caller must hold whatever keeps the mapping alive (the
// running kernel's own collection, or a retained kext that keeps its linkedit).
inline bool openSymbols(const mach_header_64 *mh, const void *linkeditBase, SymbolTable &out) {
    out = SymbolTable{};
    if (!commandsValid(mh) || !linkeditBase)
        return false;
    auto le = segment(mh, SEG_LINKEDIT);
    const symtab_command *st = nullptr;
    bool invalid = false;
    eachCommand(mh, [&st, &invalid](const load_command *lc) {
        if (lc->cmd == LC_SYMTAB) {
            if (st || lc->cmdsize != sizeof(symtab_command)) {
                invalid = true;
                return false;
            }
            st = reinterpret_cast<const symtab_command *>(lc);
        }
        return false;
    });
    if (invalid || !le || !st || le->filesize > le->vmsize || st->symoff < le->fileoff ||
        st->stroff < le->fileoff)
        return false;
    uint64_t symRel = st->symoff - le->fileoff;
    uint64_t strRel = st->stroff - le->fileoff;
    if (symRel + static_cast<uint64_t>(st->nsyms) * sizeof(nlist_64) > le->filesize ||
        strRel + st->strsize > le->filesize)
        return false;
    auto base = reinterpret_cast<const uint8_t *>(linkeditBase);
    if (reinterpret_cast<uintptr_t>(base) % alignof(nlist_64) || symRel % alignof(nlist_64))
        return false;
    out.list = reinterpret_cast<const nlist_64 *>(base + symRel);
    out.count = st->nsyms;
    out.strings = reinterpret_cast<const char *>(base + strRel);
    out.stringsSize = st->strsize;
    return true;
}

// The name of symbol `i` if it is a defined section symbol with a name wholly
// inside the string table, else null. Skips debug (N_STAB) and undefined
// symbols, so lookup sees only real definitions.
inline const char *definedName(const SymbolTable &t, uint32_t i) {
    if (!t.valid() || i >= t.count)
        return nullptr;
    const nlist_64 &n = t.list[i];
    if ((n.n_type & N_STAB) || (n.n_type & N_TYPE) != N_SECT)
        return nullptr;
    uint32_t at = n.n_un.n_strx;
    // The name must be NUL-terminated within the string table: strnlen returns
    // the remaining length only when it found no terminator.
    if (at >= t.stringsSize || strnlen(t.strings + at, t.stringsSize - at) == t.stringsSize - at)
        return nullptr;
    return t.strings + at;
}

// Space for an independent copy that answers every lookup as `table` does: each
// entry, including duplicate local definitions, and the name of each entry that
// lookup sees (definedName), copied once per entry. A collection's entries all
// name one string table holding every symbol in the collection, so copying only
// these names keeps an entry's copy to its own. Zero means invalid.
inline size_t symbolSnapshotSize(const SymbolTable &table) {
    if (!table.valid())
        return 0;
    size_t names = 1; // the empty name at offset 0
    for (uint32_t i = 0; i < table.count; ++i) {
        if (auto name = definedName(table, i)) {
            names += strlen(name) + 1; // bounded: definedName found the terminator
            if (names > UINT32_MAX)
                return 0;
        }
    }
    return static_cast<size_t>(table.count) * sizeof(nlist_64) + names;
}

// The caller owns aligned storage disjoint from the source tables, and keeps
// it alive while using out. Only the source read needs linkedit residency. An
// entry lookup does not see keeps a string offset past the copied names, so the
// copy does not see it either.
inline bool snapshotSymbols(const SymbolTable &table, void *storage, size_t size,
                            SymbolTable &out) {
    size_t required = symbolSnapshotSize(table);
    if (!required || !storage || size < required ||
        reinterpret_cast<uintptr_t>(storage) % alignof(nlist_64))
        return false;
    size_t listSize = static_cast<size_t>(table.count) * sizeof(nlist_64);
    auto list = static_cast<nlist_64 *>(storage);
    auto strings = static_cast<char *>(storage) + listSize;
    auto stringsSize = static_cast<uint32_t>(required - listSize);
    memcpy(list, table.list, listSize);
    strings[0] = '\0';
    uint32_t cursor = 1;
    for (uint32_t i = 0; i < table.count; ++i) {
        auto name = definedName(table, i);
        if (!name) {
            list[i].n_un.n_strx = stringsSize;
            continue;
        }
        auto length = static_cast<uint32_t>(strlen(name) + 1);
        memcpy(strings + cursor, name, length);
        list[i].n_un.n_strx = cursor;
        cursor += length;
    }
    out = {list, table.count, strings, stringsSize, table.bias};
    return true;
}

// Whether `name` is defined exactly at `address`: used
// to tell which of a duplicated name's copies sits at a slot's address.
inline bool definesAt(const SymbolTable &t, const char *name, uint64_t address) {
    for (uint32_t i = 0; i < t.count; ++i) {
        if (symbolAddress(t, i) != address)
            continue;
        const char *s = definedName(t, i);
        if (s && !strcmp(s, name))
            return true;
    }
    return false;
}

// The vmaddr of fileset entry `id` in the collection `mh`, else 0. xnu slides
// the entries and their segments with the collection, so this finds an entry
// whether or not the kernel has loaded or started it, such as a dormant
// consumer.
inline uint64_t filesetEntry(const mach_header_64 *mh, const char *id) {
    if (!commandsValid(mh) || mh->filetype != MH_FILESET)
        return 0;
    auto lc = eachCommand(mh, [id](const load_command *lc) {
        if (lc->cmd != LC_FILESET_ENTRY || lc->cmdsize < sizeof(fileset_entry_command))
            return false;
        auto offset = reinterpret_cast<const fileset_entry_command *>(lc)->entry_id.offset;
        if (offset < sizeof(fileset_entry_command) || offset >= lc->cmdsize)
            return false;
        auto name = reinterpret_cast<const char *>(lc) + offset;
        uint32_t room = lc->cmdsize - offset;
        return strnlen(name, room) < room && !strcmp(name, id);
    });
    return lc ? reinterpret_cast<const fileset_entry_command *>(lc)->vmaddr : 0;
}

// Each fileset entry's vmaddr in the collection `mh`, in load-command order,
// until visit(vmaddr) returns true; returns that vmaddr, or 0 when none did. For
// finding an entry by what it holds when its bundle identifier is not known.
template <typename Visit>
inline uint64_t eachFilesetEntry(const mach_header_64 *mh, Visit visit) {
    if (!commandsValid(mh) || mh->filetype != MH_FILESET)
        return 0;
    auto lc = eachCommand(mh, [&visit](const load_command *lc) {
        return lc->cmd == LC_FILESET_ENTRY && lc->cmdsize >= sizeof(fileset_entry_command) &&
               visit(reinterpret_cast<const fileset_entry_command *>(lc)->vmaddr);
    });
    return lc ? reinterpret_cast<const fileset_entry_command *>(lc)->vmaddr : 0;
}

} // namespace macho
} // namespace graft
