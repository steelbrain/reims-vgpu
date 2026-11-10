// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// RuntimeImage's implementation: map a loaded Mach-O, copy its symbol table
// while its linkedit is resident, and answer segment-bounded reads, symbol
// lookups and section locations afterward. openKernel finds the running
// kernel's image in the primary collection and calibrates it against &IOLog.
// The reader itself (MachO.h) is pure logic; this file owns the live mappings.
#include "RuntimeImage.h"
#include "Log.h"
#include "Resolve.h"
#include <IOKit/IOLib.h>
#include <mach/vm_prot.h>

namespace graft {

RuntimeImage::~RuntimeImage() {
    reset();
}

void RuntimeImage::reset() {
    symbols = {};
    header = nullptr;
    segmentBias = 0;
    failure = nullptr;
    if (symbolStorage)
        IOFree(symbolStorage, symbolStorageSize);
    symbolStorage = nullptr;
    symbolStorageSize = 0;
}

bool RuntimeImage::refuse(const char *reason) {
    reset();
    failure = reason;
    return false;
}

bool RuntimeImage::map(const mach_header_64 *mh) {
    reset();
    // The header comes from the platform expert or a retained loaded kext.
    // This bounds command walking; the provider guarantees that its header
    // mapping and declared segment mappings stay accessible while retained.
    if (!mh || mh->magic != MH_MAGIC_64)
        return refuse("invalid image header");
    if (mh->sizeofcmds > macho::MaxLoadCommands)
        return refuse("image load commands exceed the 64 KiB cap");
    if (!macho::commandsValid(mh))
        return refuse("invalid image load commands");
    auto text = macho::segment(mh, SEG_TEXT);
    auto base = reinterpret_cast<uint64_t>(mh);
    if (!text || text->vmaddr > base)
        return refuse("missing text segment or negative segment bias");
    header = mh;
    segmentBias = base - text->vmaddr;
    return contains(base, sizeof(*mh) + mh->sizeofcmds, VM_PROT_READ) ||
           refuse("header outside readable segments");
}

// Map mh and find its symbol tables where its linkedit is mapped, unbiased.
bool RuntimeImage::mapSymbols(const mach_header_64 *mh, macho::SymbolTable &table) {
    if (!map(mh))
        return false;
    auto le = macho::segment(mh, SEG_LINKEDIT);
    if (!le)
        return refuse("missing linkedit segment");
    if (le->vmaddr > UINT64_MAX - segmentBias)
        return refuse("linkedit address overflow");
    if (!macho::openSymbols(mh, reinterpret_cast<const void *>(le->vmaddr + segmentBias), table))
        return refuse("symbol tables outside mapped linkedit bounds");
    return true;
}

bool RuntimeImage::open(const mach_header_64 *mh, const char *anchor, uint64_t actualAddress) {
    macho::SymbolTable table;
    if (!mapSymbols(mh, table))
        return false;
    uint64_t raw = 0;
    auto resolution = binding::unique(table, anchor, raw);
    if (resolution != binding::Resolution::Found)
        return refuse(resolution == binding::Resolution::Missing ? "anchor symbol missing"
                                                                 : "anchor symbol ambiguous");
    if (actualAddress < raw)
        return refuse("anchor address below symbol value");
    table.bias = actualAddress - raw;
    if (!contains(actualAddress, 1))
        return refuse("anchor outside retained image segments");
    return keepSymbols(table);
}

bool RuntimeImage::openEntryDefining(const mach_header_64 *collection, const char *symbol,
                                     uint64_t bias) {
    const mach_header_64 *match = nullptr;
    bool twice = false;
    macho::eachFilesetEntry(collection, [&](uint64_t vmaddr) {
        auto mh = macho::header(reinterpret_cast<const void *>(vmaddr));
        macho::SymbolTable table;
        if (!mh || !mapSymbols(mh, table))
            return false;
        table.bias = bias;
        uint64_t address = 0;
        if (binding::unique(table, symbol, address) != binding::Resolution::Found ||
            !contains(address, 1))
            return false;
        twice = match != nullptr;
        match = mh;
        return twice;
    });
    if (!match || twice)
        return refuse(match ? "symbol defined by two collection entries"
                            : "no collection entry defines the symbol");
    macho::SymbolTable table;
    if (!mapSymbols(match, table))
        return false;
    table.bias = bias;
    return keepSymbols(table);
}

// Copy the symbol tables into owned storage while the linkedit is still mapped.
bool RuntimeImage::keepSymbols(const macho::SymbolTable &table) {
    size_t size = macho::symbolSnapshotSize(table);
    auto storage = size ? IOMalloc(size) : nullptr;
    if (!storage)
        return refuse("cannot allocate symbol snapshot");
    if (!macho::snapshotSymbols(table, storage, size, symbols)) {
        IOFree(storage, size);
        return refuse("cannot copy symbol snapshot");
    }
    symbolStorage = storage;
    symbolStorageSize = size;
    return true;
}

bool RuntimeImage::contains(uint64_t address, size_t size, int protection, bool linkedit) const {
    if (!header || !size || size - 1 > UINT64_MAX - address)
        return false;
    return macho::eachCommand(header, [&](const load_command *lc) {
               if (lc->cmd != LC_SEGMENT_64 || lc->cmdsize < sizeof(segment_command_64))
                   return false;
               auto segment = reinterpret_cast<const segment_command_64 *>(lc);
               if ((!linkedit && !strncmp(segment->segname, SEG_LINKEDIT, 16)) ||
                   (segment->initprot & protection) != protection ||
                   segment->vmaddr > UINT64_MAX - segmentBias)
                   return false;
               uint64_t start = segment->vmaddr + segmentBias;
               return address >= start && address - start <= segment->vmsize &&
                      size <= segment->vmsize - (address - start);
           }) != nullptr;
}

bool RuntimeImage::writableData(uint64_t address, size_t size) const {
    return header && macho::writableSegment(header, segmentBias, address, size);
}

uint64_t RuntimeImage::resolve(const char *name) const {
    uint64_t address = 0;
    return binding::unique(symbols, name, address) == binding::Resolution::Found &&
                           contains(address, 1)
                   ? address
                   : 0;
}

uint64_t RuntimeImage::section(const char *segment, const char *name, size_t &size) const {
    size = 0;
    if (!header)
        return 0;
    auto section = macho::section(header, segment, name);
    if (!section || !section->size || section->addr > UINT64_MAX - segmentBias ||
        !contains(section->addr + segmentBias, section->size))
        return 0;
    size = section->size;
    return section->addr + segmentBias;
}

uint64_t RuntimeImage::segmentAddress(const char *name) const {
    if (!header || !name)
        return 0;
    uint64_t address = 0;
    macho::eachCommand(header, [&](const load_command *lc) {
        if (lc->cmd != LC_SEGMENT_64 || lc->cmdsize < sizeof(segment_command_64))
            return false;
        auto segment = reinterpret_cast<const segment_command_64 *>(lc);
        if (strncmp(segment->segname, name, 16) || segment->vmaddr > UINT64_MAX - segmentBias)
            return false;
        address = segment->vmaddr + segmentBias;
        return true;
    });
    return address;
}

uint64_t RuntimeImage::symbolExtent(uint64_t address) const {
    if (!contains(address, 1))
        return 0;
    uint64_t end = UINT64_MAX;
    for (uint32_t i = 0; i < symbols.count; ++i) {
        if (!macho::definedName(symbols, i))
            continue;
        uint64_t next = macho::symbolAddress(symbols, i);
        if (next > address && next < end)
            end = next;
    }
    // An extent is accepted only wholly within a single mapped segment.
    return end != UINT64_MAX && contains(address, end - address) ? end - address : 0;
}

bool openKernel(RuntimeImage &kernel, const mach_header_64 *&collection) {
    collection = bootCollectionHeader();
    if (!collection)
        return false;
    if (collection->sizeofcmds > macho::MaxLoadCommands) {
        note("binding: refuse kernel collection: load commands exceed the 64 KiB cap");
        return false;
    }
    if (!macho::commandsValid(collection))
        return false;
    uint64_t address = macho::filesetEntry(collection, "com.apple.kernel");
    auto mh = address ? macho::header(reinterpret_cast<const void *>(address)) : collection;
    return kernel.open(mh, "_IOLog", ioLogAddress());
}

} // namespace graft
