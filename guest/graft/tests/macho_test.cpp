// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Host test of src/MachO.h (make test). Two parts:
//  - synthetic: a hand-built symbol table in memory, to prove duplicate-aware
//    lookup, STAB/undefined skipping and string-table bounds without any file;
//  - real: a dylib built from tests/fixture.c (our own code), mmap'd, to prove
//    the whole parse path on a genuine linked Mach-O.
#include "MachO.h"

#include "Check.h"
#include <fcntl.h>
#include <initializer_list>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace graft::macho;

// A symbol table assembled by hand: entries point into `strings` by offset.
static void synthetic() {
    // offsets: 0 "" , 1 "_dup" , 6 "_single" , 14 "_stab" , 20 "_undef"
    static const char strings[] = "\0_dup\0_single\0_stab\0_undef";
    nlist_64 list[6] = {};
    auto def = [](nlist_64 &n, uint32_t strx, uint64_t value) {
        n.n_un.n_strx = strx;
        n.n_type = N_SECT;
        n.n_sect = 1;
        n.n_value = value;
    };
    def(list[0], 1, 0x1000); // _dup at 0x1000
    def(list[1], 6, 0x2000); // _single at 0x2000
    def(list[2], 1, 0x3000); // _dup again at 0x3000 (a second copy)
    def(list[3], 14, 0x4000);
    list[3].n_type |= N_STAB; // _stab: a debug entry, ignored
    list[4].n_un.n_strx = 20;
    list[4].n_type = N_UNDF; // _undef: undefined, ignored
    def(list[5], 1, 0x5000); // _dup a third time at 0x5000

    SymbolTable t;
    t.list = list;
    t.count = 6;
    t.strings = strings;
    t.stringsSize = sizeof(strings);

    // Duplicate-aware: _dup is defined three times, and definesAt finds each
    // copy at its own address; _single once.
    CHECK(definesAt(t, "_dup", 0x1000));
    CHECK(definesAt(t, "_dup", 0x3000));
    CHECK(definesAt(t, "_dup", 0x5000));
    CHECK(definesAt(t, "_single", 0x2000));
    CHECK(!definesAt(t, "_dup", 0x2000)); // that address is _single
    CHECK(!definesAt(t, "_single", 0x1000));

    // A debug entry and an undefined one are not definitions.
    CHECK(!definedName(t, 3) && !definesAt(t, "_stab", 0x4000));
    CHECK(!definedName(t, 4) && !definesAt(t, "_undef", 0));
    CHECK(!definedName(t, t.count));

    // The bias slides every address.
    t.bias = 0x100000;
    CHECK(symbolAddress(t, 1) == 0x102000 && definesAt(t, "_single", 0x102000));
    CHECK(definesAt(t, "_dup", 0x101000) && definesAt(t, "_dup", 0x103000));
    CHECK(!definesAt(t, "_dup", 0x3000));
    t.bias = 0;

    // A short name at the very end of a table must not be compared past
    // its terminator when asked about a longer name at that address.
    char shortName[] = {'x', 0};
    nlist_64 shortSymbol{};
    def(shortSymbol, 0, 0x1234);
    SymbolTable shortTable{&shortSymbol, 1, shortName, sizeof(shortName)};
    CHECK(!definesAt(shortTable, "x_much_longer_than_the_table", 0x1234));

    puts("macho synthetic: pass (duplicates, STAB/undefined skip, bias, bounded names)");
}

static void malformed() {
    struct Image {
        mach_header_64 mh;
        segment_command_64 segment;
        symtab_command symbols;
        load_command tail;
    } image{};
    image.mh.magic = MH_MAGIC_64;
    image.mh.ncmds = 3;
    image.mh.sizeofcmds = sizeof(image) - sizeof(image.mh);
    image.segment.cmd = LC_SEGMENT_64;
    image.segment.cmdsize = sizeof(image.segment);
    memcpy(image.segment.segname, SEG_LINKEDIT, sizeof(SEG_LINKEDIT));
    image.segment.filesize = 16;
    image.symbols.cmd = LC_SYMTAB;
    image.symbols.cmdsize = sizeof(image.symbols);
    image.tail.cmdsize = sizeof(image.tail);
    CHECK(header(&image) && commandsValid(&image.mh));
    image.mh.magic = MH_MAGIC; // a 32-bit image
    CHECK(!header(&image) && !commandsValid(&image.mh));
    image.mh.magic = MH_MAGIC_64;
    image.tail.cmdsize = 0;
    CHECK(!commandsValid(&image.mh));
    CHECK(!segment(&image.mh, SEG_LINKEDIT));
    uint64_t linkedit[2]{};
    SymbolTable table;
    CHECK(!openSymbols(&image.mh, linkedit, table) && !table.valid());
    image.tail.cmdsize = sizeof(image.tail);
    image.mh.ncmds = 2;
    CHECK(!commandsValid(&image.mh)); // unaccounted bytes in sizeofcmds
    CHECK(!commandsValid(nullptr) && !header(nullptr));
    puts("macho malformed: pass (truncated and trailing commands, bounded names)");
}

// writableSegment: a write target is accepted only inside a writable, non
// read-only, non-__TEXT segment, spanning it wholly, and slid by the bias.
static void writableSegments() {
    struct Image {
        mach_header_64 mh;
        segment_command_64 text;
        segment_command_64 data;
        segment_command_64 constData;
    } image{};
    image.mh.magic = MH_MAGIC_64;
    image.mh.ncmds = 3;
    image.mh.sizeofcmds = sizeof(image) - sizeof(image.mh);
    auto fill = [](segment_command_64 &s, const char *name, uint64_t vmaddr, uint64_t vmsize,
                   vm_prot_t initprot, uint32_t flags) {
        s.cmd = LC_SEGMENT_64;
        s.cmdsize = sizeof(s);
        strncpy(s.segname, name, sizeof(s.segname));
        s.vmaddr = vmaddr;
        s.vmsize = vmsize;
        s.initprot = initprot;
        s.flags = flags;
    };
    fill(image.text, SEG_TEXT, 0x1000, 0x1000, VM_PROT_READ | VM_PROT_EXECUTE, 0);
    fill(image.data, SEG_DATA, 0x2000, 0x1000, VM_PROT_READ | VM_PROT_WRITE, 0);
    fill(image.constData, "__DATA_CONST", 0x3000, 0x1000, VM_PROT_READ | VM_PROT_WRITE,
         SG_READ_ONLY);
    CHECK(commandsValid(&image.mh));
    const uint64_t bias = 0x40000000;
    // Inside __DATA, wholly: accepted.
    CHECK(writableSegment(&image.mh, bias, 0x2000 + bias, 8));
    CHECK(writableSegment(&image.mh, bias, 0x2000 + bias, 0x1000));
    // __TEXT (executable, read-only data behind it): refused even though mapped.
    CHECK(!writableSegment(&image.mh, bias, 0x1000 + bias, 8));
    // __DATA_CONST: writable protection but locked read-only: refused.
    CHECK(!writableSegment(&image.mh, bias, 0x3000 + bias, 8));
    // A span that runs past the segment end: refused.
    CHECK(!writableSegment(&image.mh, bias, 0x2000 + bias, 0x1001));
    // Outside every segment, and a zero span: refused.
    CHECK(!writableSegment(&image.mh, bias, 0x9000 + bias, 8));
    CHECK(!writableSegment(&image.mh, bias, 0x2000 + bias, 0));
    puts("macho writable segments: pass (writable __DATA only, __TEXT and read-only refused)");
}

// A fileset collection: eachFilesetEntry visits each entry's vmaddr and
// identifier in command order until one is accepted, skipping other commands and
// an entry whose identifier is out of bounds or unterminated, and filesetEntry
// finds an entry by its bundle identifier. Graft finds the bound consumers on
// x86-64 by this walk.
static void filesetEntries() {
    struct Entry {
        fileset_entry_command command;
        char id[24];
    };
    struct Collection {
        mach_header_64 mh;
        Entry entries[4];
        load_command other;
    } image{};
    image.mh.magic = MH_MAGIC_64;
    image.mh.filetype = MH_FILESET;
    image.mh.ncmds = 5;
    image.mh.sizeofcmds = sizeof(image) - sizeof(image.mh);
    const char *ids[] = {"com.example.a", "com.example.b", "com.example.c", "com.example.d"};
    for (uint32_t i = 0; i < 4; ++i) {
        auto &entry = image.entries[i];
        entry.command.cmd = LC_FILESET_ENTRY;
        entry.command.cmdsize = sizeof(Entry);
        entry.command.vmaddr = 0x1000 * (i + 1);
        entry.command.entry_id.offset = sizeof(entry.command);
        strlcpy(entry.id, ids[i], sizeof(entry.id));
    }
    // The fourth entry's identifier fills its command with no terminator.
    memset(image.entries[3].id, 'd', sizeof(image.entries[3].id));
    image.other.cmd = LC_UUID;
    image.other.cmdsize = sizeof(image.other);
    CHECK(commandsValid(&image.mh));

    uint64_t seen[4] = {};
    const char *named[4] = {};
    size_t count = 0;
    CHECK(eachFilesetEntry(&image.mh, [&](uint64_t vmaddr, const char *id) {
              seen[count] = vmaddr;
              named[count++] = id;
              return false;
          }) == 0);
    CHECK(count == 3 && seen[0] == 0x1000 && seen[1] == 0x2000 && seen[2] == 0x3000);
    CHECK(!strcmp(named[0], "com.example.a") && !strcmp(named[2], "com.example.c"));
    count = 0;
    CHECK(eachFilesetEntry(&image.mh, [&](uint64_t vmaddr, const char *) {
              ++count;
              return vmaddr == 0x2000;
          }) == 0x2000);
    CHECK(count == 2);
    CHECK(filesetEntry(&image.mh, "com.example.c") == 0x3000);
    CHECK(filesetEntry(&image.mh, "com.example.d") == 0);
    CHECK(filesetEntry(&image.mh, "com.example.e") == 0);
    image.entries[1].command.entry_id.offset = sizeof(Entry); // past its command
    count = 0;
    eachFilesetEntry(&image.mh, [&](uint64_t, const char *) { return ++count, false; });
    CHECK(count == 2 && filesetEntry(&image.mh, "com.example.b") == 0);

    image.mh.filetype = MH_KEXT_BUNDLE; // not a collection: nothing is visited
    count = 0;
    CHECK(eachFilesetEntry(&image.mh, [&](uint64_t, const char *) { return ++count, true; }) == 0 &&
          !count);
    puts("macho fileset: pass (entries and identifiers in order, stop on accept, malformed "
         "identifiers skipped, lookup by identifier)");
}

// Symbol and string tables must lie inside __LINKEDIT's file-backed bytes,
// and those bytes must fit the segment's mapped extent (vmsize).
static void linkeditBounds() {
    struct Image {
        mach_header_64 mh;
        segment_command_64 segment;
        symtab_command symbols;
    } image{};
    image.mh.magic = MH_MAGIC_64;
    image.mh.ncmds = 2;
    image.mh.sizeofcmds = sizeof(image) - sizeof(image.mh);
    image.segment.cmd = LC_SEGMENT_64;
    image.segment.cmdsize = sizeof(image.segment);
    memcpy(image.segment.segname, SEG_LINKEDIT, sizeof(SEG_LINKEDIT));
    image.segment.fileoff = 0x100;
    image.segment.filesize = 48;
    image.segment.vmsize = 48;
    image.symbols.cmd = LC_SYMTAB;
    image.symbols.cmdsize = sizeof(image.symbols);
    image.symbols.symoff = 0x100;
    image.symbols.nsyms = 2;
    image.symbols.stroff = 0x120;
    image.symbols.strsize = 16;
    uint64_t linkedit[6]{};
    SymbolTable table;
    CHECK(openSymbols(&image.mh, linkedit, table) && table.count == 2 && table.stringsSize == 16);

    image.segment.vmsize = 47; // file bytes past the mapped extent
    CHECK(!openSymbols(&image.mh, linkedit, table) && !table.valid());
    image.segment.vmsize = 48;
    image.symbols.nsyms = 4; // nlist array past filesize
    CHECK(!openSymbols(&image.mh, linkedit, table) && !table.valid());
    image.symbols.nsyms = 2;
    image.symbols.strsize = 17; // string table past filesize
    CHECK(!openSymbols(&image.mh, linkedit, table) && !table.valid());
    image.symbols.strsize = 16;
    image.symbols.symoff = 0xf0; // nlist array before the segment
    CHECK(!openSymbols(&image.mh, linkedit, table) && !table.valid());
    image.symbols.symoff = 0x100;
    image.symbols.stroff = 0xff; // string table before the segment
    CHECK(!openSymbols(&image.mh, linkedit, table) && !table.valid());
    image.symbols.stroff = 0x120;
    CHECK(openSymbols(&image.mh, linkedit, table));
    puts("macho linkedit: pass (filesize within vmsize, tables within file bytes)");
}

// Late lookups must survive loss of the original linkedit bytes. Preserve
// duplicate local definitions, calibration bias, and bounded malformed names.
static void symbolSnapshot() {
    // The names after "_last" stand for the rest of a collection's shared string
    // table, which no entry of this table names, so the copy leaves them out.
    char strings[] = "\0_dup\0_last\0_elsewhere_in_the_collection";
    nlist_64 list[4]{};
    for (unsigned i = 0; i < 4; ++i) {
        list[i].n_type = N_SECT;
        list[i].n_sect = 1;
        list[i].n_un.n_strx = i < 2 ? 1 : 6;
        list[i].n_value = 0x1000 + i * 0x100;
    }
    list[3].n_un.n_strx = sizeof(strings); // invalid name stays bounded
    SymbolTable source{list, 4, strings, sizeof(strings), 0x100000};
    // The empty name, then each defined entry's own: "_dup" twice and "_last".
    constexpr size_t names = 1 + 5 + 5 + 6;
    constexpr size_t size = sizeof(list) + names;
    static_assert(names < sizeof(strings), "the copy leaves the unnamed strings out");
    alignas(nlist_64) uint8_t storage[size + 1];
    memset(storage, 0xa5, sizeof(storage));
    SymbolTable snapshot;
    CHECK(symbolSnapshotSize(source) == size);
    CHECK(!snapshotSymbols(source, storage, size - 1, snapshot) && !snapshot.valid());
    CHECK(!snapshotSymbols(source, storage + 1, size, snapshot) && !snapshot.valid());
    CHECK(!snapshotSymbols(source, nullptr, size, snapshot) && !snapshot.valid());
    CHECK(!snapshotSymbols({}, storage, size, snapshot) && !snapshot.valid());
    for (uint8_t byte : storage)
        CHECK(byte == 0xa5); // refusals leave storage untouched
    CHECK(snapshotSymbols(source, storage, size, snapshot));
    CHECK(storage[size] == 0xa5);
    memset(list, 0, sizeof(list));
    memset(strings, 0xff, sizeof(strings));
    CHECK(definesAt(snapshot, "_dup", 0x101000) && definesAt(snapshot, "_dup", 0x101100));
    CHECK(definesAt(snapshot, "_last", 0x101200));
    CHECK(!definedName(snapshot, 3));
    CHECK(snapshot.stringsSize == names);
    puts("macho snapshot: pass (owned bytes, own names only, duplicate locals, bias, refusal "
         "bounds)");
}

// The real dylib: parse it end to end and find our own fixture symbols.
static void realImage(const char *path) {
    int fd = open(path, O_RDONLY);
    CHECK(fd >= 0);
    struct stat sb;
    CHECK(fstat(fd, &sb) == 0);
    void *base = mmap(nullptr, (size_t)sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    CHECK(base != MAP_FAILED);
    close(fd);

    const mach_header_64 *mh = header(base);
    CHECK(mh && mh->filetype == MH_DYLIB);

    // __LINKEDIT's bytes are mapped at the file's base plus its fileoff.
    auto le = segment(mh, SEG_LINKEDIT);
    CHECK(le);
    const void *linkeditBase = reinterpret_cast<const uint8_t *>(base) + le->fileoff;

    SymbolTable t;
    CHECK(openSymbols(mh, linkeditBase, t) && t.valid() && t.count > 0);

    // Each fixture function is defined by exactly one entry, at an address
    // definesAt confirms, and a name the fixture lacks is defined by none.
    auto definitions = [&](const char *name, uint64_t &address) {
        size_t count = 0;
        for (uint32_t i = 0; i < t.count; ++i) {
            const char *defined = definedName(t, i);
            if (defined && !strcmp(defined, name)) {
                address = symbolAddress(t, i);
                ++count;
            }
        }
        return count;
    };
    for (const char *name : {"_graft_fixture_a", "_graft_fixture_b", "_graft_fixture_c"}) {
        uint64_t addr = 0;
        CHECK(definitions(name, addr) == 1 && addr != 0);
        CHECK(definesAt(t, name, addr));
        CHECK(!definesAt(t, name, addr + 1));
    }
    uint64_t missing = 0;
    CHECK(definitions("_graft_not_here", missing) == 0);

    munmap(base, (size_t)sb.st_size);
    puts("macho real dylib: pass (header, __LINKEDIT, symtab, defined symbols)");
}

int main(int argc, char **argv) {
    synthetic();
    malformed();
    writableSegments();
    filesetEntries();
    linkeditBounds();
    symbolSnapshot();
    CHECK(argc >= 2 && "usage: macho_test <fixture.dylib>");
    realImage(argv[1]);
    puts("macho reader: pass");
    return 0;
}
