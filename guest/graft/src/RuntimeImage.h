// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
#pragma once
#include "MachO.h"

namespace graft {

// A view over a retained loaded image. open() receives a symbol whose actual
// address the caller independently knows, so it can calibrate symbol values
// without assuming whether the collection has already slid its nlist data.
// open() copies the symbol and string tables before returning. Linkedit must
// be readable during open(); later lookups use owned storage. The caller
// retains the image's header and non-linkedit mappings for this view's life.
struct RuntimeImage {
    RuntimeImage() = default;
    ~RuntimeImage();
    RuntimeImage(const RuntimeImage &) = delete;
    RuntimeImage &operator=(const RuntimeImage &) = delete;

    const mach_header_64 *header = nullptr;
    macho::SymbolTable symbols{};
    uint64_t segmentBias = 0;

    bool map(const mach_header_64 *);
    bool open(const mach_header_64 *, const char *anchor, uint64_t actualAddress);
    // Open the one entry of `collection` that defines `symbol`, with the symbol
    // bias another entry of the same collection measured: a collection's entries
    // share one slide, so an entry whose classes never register (a kext that is
    // placed but never started) needs no anchor of its own. Refuses when no entry,
    // or more than one, defines the symbol within its own segments.
    bool openEntryDefining(const mach_header_64 *collection, const char *symbol, uint64_t bias);
    bool contains(uint64_t address, size_t size, int protection = 0, bool linkedit = false) const;
    // True when `address`..+size lies wholly in one writable data segment: one
    // whose protection grants write, that the kernel has not locked read-only
    // (SG_READ_ONLY), and whose name does not begin with __TEXT. A store to a
    // segment the kernel maps read-only (such as __DATA_CONST) faults, so the
    // arm64e binding uses this to refuse such a target before writing.
    bool writableData(uint64_t address, size_t size) const;
    uint64_t resolve(const char *name) const;
    uint64_t section(const char *segment, const char *name, size_t &size) const;
    // The runtime start address of the named segment (its placed vmaddr plus the
    // slide), or 0 if absent. Graft relocates a manifest offset through this when
    // the consumer's segments are coalesced non-contiguously (arm64e).
    uint64_t segmentAddress(const char *name) const;
    uint64_t symbolExtent(uint64_t address) const;
    const char *error() const { return failure; }

private:
    void reset();
    bool refuse(const char *reason);
    bool mapSymbols(const mach_header_64 *, macho::SymbolTable &table);
    bool keepSymbols(const macho::SymbolTable &table);
    const char *failure = nullptr;
    void *symbolStorage = nullptr;
    size_t symbolStorageSize = 0;
};

// Open the running kernel's image from the primary (Boot) kernel collection.
// collection is that collection once its load commands validate, else null, and
// is set whether or not the kernel's image then opens, so Graft's start can keep
// it to tell placed kexts apart even when it refuses.
bool openKernel(RuntimeImage &kernel, const mach_header_64 *&collection);

// Start-path points that differ by architecture, defined in src/<arch>/Kernel.cpp:
// the primary (Boot) kernel collection's header, and IOLog's running address,
// which openKernel calibrates the kernel's symbol table against. On x86-64 the
// kernel's own accessor and the plain linked address; on arm64e the device-tree
// handover and IOLog's raw address from its unsigned data GOT slot.
const mach_header_64 *bootCollectionHeader();
uint64_t ioLogAddress();

} // namespace graft
