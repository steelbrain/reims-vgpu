// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Check that a prelinked kernel collection keeps its staged consumers bindable.
//
// graft-kext check-prelinked runs this check. The collection is an x86-64 kernel
// collection after OpenCore prelinked Graft and the kexts it serves into it, and
// each consumer a staged bundle. The check finds each consumer's fileset entry by
// its bundle identifier and requires what Graft relies on when it binds that
// entry:
//
// - the entry's header starts its __TEXT, and every staged segment keeps its
//   position relative to the header, because Graft adds manifest offsets to the
//   entry's address;
// - __DATA,__graft holds the staged manifest unchanged through the size the
//   manifest declares; the placeholder tail after it is not compared;
// - every manifest site holds its exact local placeholder plus addend, and each
//   placeholder lies inside the reserved tail. A pointer site must be an entry of
//   the collection's fixup chains, since the booter rewrites only those when it
//   slides the collection, and its decoded target must be that address. A
//   branch site's E8 or E9 displacement must reach it;
// - no fixup-chain entry overlaps a branch site or the manifest, whose bytes the
//   booter would otherwise rewrite.
//
// An x86-64 collection encodes every pointer as a chained rebase
// (DYLD_CHAINED_PTR_X86_64_KERNEL_CACHE). A region's segment_offset counts from
// the collection header, and a rebase target from the collection's lowest
// segment address. Regions are matched by segment_offset, not by load-command
// order, since OpenCore reuses a region slot for the kexts it injects, and
// writes their starts after the table's declared size. Chains may run past their
// starting page, so each region's chains are walked whole.
//
// The collection is only read, and every read is bounded by its file. This part
// of graft-kext runs no tool, so it builds and runs on any host.
#pragma once
#include "base.h"
#include "macho.h"

#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace graft::tool {

class Collection {
public:
    explicit Collection(Bytes bytes);

    // A segment of the collection or of a fileset entry: offset is
    // collection-absolute, and sections maps each section's name to its address
    // and size.
    struct Segment {
        std::string name;
        uint64_t address, vmsize, offset, filesize;
        std::map<std::string, std::pair<uint64_t, uint64_t>> sections;
    };
    // One fileset entry's kext: its header's address, and its segments by name,
    // which order lists as its load commands do.
    struct Entry {
        std::string name;
        uint64_t address;
        std::map<std::string, Segment> segments;
        std::vector<std::string> order;
    };

    // The fileset entry whose identifier is name.
    Entry entry(const std::string &name) const;
    // width bytes at address, which one of entry's file-backed segments holds.
    Bytes read(const Entry &entry, uint64_t address, uint64_t width) const;
    // The address the chained rebase at address holds once the booter slides it.
    uint64_t rebase(uint64_t address) const;
    // Whether any chain entry's eight bytes overlap the width bytes at address.
    bool overlapsFixup(uint64_t address, uint64_t width) const;

private:
    // One dyld_chained_starts_in_segment: pages from start, each with the offset
    // of its first chain entry. entries holds every entry's address once the
    // region's chains have been walked.
    struct Region {
        uint64_t start, pageSize;
        std::vector<uint16_t> starts;
        mutable std::optional<std::set<uint64_t>> entries;
        uint64_t end() const { return start + pageSize * starts.size(); }
    };

    void bounds(uint64_t offset, uint64_t size) const;
    uint32_t u32(uint64_t offset) const;
    uint64_t u64(uint64_t offset) const;
    std::string fixed(uint64_t offset) const;
    std::string string(uint64_t offset, uint64_t room) const;
    // Visit each load command after the Mach-O header at header with its cmd, its
    // offset and its cmdsize, checking each before it is visited.
    void forEachCommand(uint64_t header, uint32_t count, uint32_t size,
                        const std::function<void(uint32_t, uint64_t, uint32_t)> &visit) const;
    Segment readSegment(uint64_t at, uint32_t length) const;
    std::vector<Region> chainedStarts(uint64_t offset, uint64_t size) const;
    Region readRegion(uint64_t at, uint64_t room) const;
    Bytes readCollection(uint64_t address, uint64_t width) const;
    const std::set<uint64_t> &chainEntries(uint64_t address) const;

    Bytes data_;
    std::map<std::string, std::pair<uint64_t, uint64_t>> entries_;
    std::vector<Segment> segments_;
    uint64_t header_ = 0, base_ = 0;
    std::vector<Region> regions_;
};

// Check one staged consumer against its entry in collection, returning the
// number of sites checked; throws Error at the first defect.
size_t checkPrelinked(const Collection &collection, const std::string &identifier,
                      const Image &staged);

} // namespace graft::tool
