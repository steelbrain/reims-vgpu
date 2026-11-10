// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
#include "prelink.h"

#include "manifest.h"

#include <algorithm>
#include <cstdio>
#include <functional>

namespace graft::tool {

namespace {

constexpr uint32_t FILESET_ENTRY_COMMAND_SIZE = 32;
constexpr uint64_t CHAINED_FIXUPS_HEADER_SIZE = 28;
constexpr uint64_t CHAINED_STARTS_IN_SEGMENT_SIZE = 22;
constexpr uint16_t DYLD_CHAINED_PTR_X86_64_KERNEL_CACHE = 11;
constexpr uint16_t DYLD_CHAINED_PTR_START_NONE = 0xffff;
constexpr uint16_t DYLD_CHAINED_PTR_START_MULTI = 0x8000;
// dyld_chained_ptr_x86_64_kernel_cache_rebase is target:30 cacheLevel:2
// diversity:16 addrDiv:1 key:2 next:12 isAuth:1. The check accepts only plain
// rebases into this collection, whose every field but target and next is zero.
constexpr uint64_t TargetMask = (uint64_t{1} << 30) - 1;
constexpr unsigned NextShift = 51;
constexpr uint64_t NextMask = 0xfff;
constexpr uint64_t NotPlain = ~(TargetMask | NextMask << NextShift);

// The file offset of width bytes at address, if one segment's file bytes hold them.
std::optional<uint64_t> fileOffset(const std::vector<Collection::Segment> &segments,
                                   uint64_t address, uint64_t width) {
    for (const Collection::Segment &segment : segments) {
        if (segment.address <= address && width <= segment.filesize &&
            address - segment.address <= segment.filesize - width)
            return segment.offset + (address - segment.address);
    }
    return std::nullopt;
}

// A byte as Python formats {value:#04x}: 0x and two lowercase digits.
std::string hexByte(uint8_t value) {
    char text[8];
    std::snprintf(text, sizeof text, "0x%02x", static_cast<unsigned>(value));
    return text;
}

} // namespace

Collection::Collection(Bytes bytes) : data_(std::move(bytes)) {
    bounds(0, MACH_HEADER_SIZE);
    if (u32(0) != MH_MAGIC_64 || u32(4) != CPU_TYPE_X86_64 || u32(12) != MH_FILESET)
        throw Error("expected an x86-64 kernel collection");
    std::vector<std::pair<uint32_t, uint32_t>> fixups;
    forEachCommand(0, u32(16), u32(20), [&](uint32_t command, uint64_t at, uint32_t length) {
        if (command == LC_SEGMENT_64) {
            segments_.push_back(readSegment(at, length));
        } else if (command == LC_DYLD_CHAINED_FIXUPS) {
            if (length != 16)
                throw Error("invalid chained-fixups command");
            fixups.emplace_back(u32(at + 8), u32(at + 12));
        } else if (command == LC_FILESET_ENTRY) {
            if (length < FILESET_ENTRY_COMMAND_SIZE)
                throw Error("truncated fileset entry command");
            uint64_t address = u64(at + 8), offset = u64(at + 16);
            uint32_t nameOffset = u32(at + 24);
            if (!(FILESET_ENTRY_COMMAND_SIZE <= nameOffset && nameOffset < length))
                throw Error("fileset entry name outside its command");
            std::string name = string(at + nameOffset, length - nameOffset);
            if (entries_.count(name))
                throw Error("duplicate fileset entry " + name);
            entries_[name] = {address, offset};
        }
    });
    std::vector<uint64_t> headers, mapped;
    for (const Segment &segment : segments_) {
        if (segment.offset == 0 && segment.filesize)
            headers.push_back(segment.address);
        if (segment.vmsize)
            mapped.push_back(segment.address);
    }
    if (headers.size() != 1 || fixups.size() != 1)
        throw Error("collection needs one header segment and one chained-fixups table");
    header_ = headers[0];
    base_ = *std::min_element(mapped.begin(), mapped.end());
    regions_ = chainedStarts(fixups[0].first, fixups[0].second);
}

void Collection::bounds(uint64_t offset, uint64_t size) const {
    if (offset > data_.size() || size > data_.size() - offset)
        throw Error("collection range outside file");
}

uint32_t Collection::u32(uint64_t offset) const {
    bounds(offset, 4);
    return load32(&data_[offset]);
}

uint64_t Collection::u64(uint64_t offset) const {
    bounds(offset, 8);
    return load64(&data_[offset]);
}

// A 16-byte NUL-padded segment or section name.
std::string Collection::fixed(uint64_t offset) const {
    bounds(offset, 16);
    std::string name;
    for (uint64_t i = offset; i < offset + 16 && data_[i]; ++i) {
        if (data_[i] >= 0x80)
            throw Error("invalid segment or section name");
        name += static_cast<char>(data_[i]);
    }
    return name;
}

// A nonempty NUL-terminated ASCII string that ends within room bytes.
std::string Collection::string(uint64_t offset, uint64_t room) const {
    bounds(offset, room);
    std::string text;
    for (uint64_t i = offset; i < offset + room; ++i) {
        if (!data_[i]) {
            if (text.empty())
                break;
            for (char c : text) {
                if (static_cast<unsigned char>(c) >= 0x80)
                    throw Error("invalid load-command string");
            }
            return text;
        }
        text += static_cast<char>(data_[i]);
    }
    throw Error("unterminated or empty load-command string");
}

void Collection::forEachCommand(
        uint64_t header, uint32_t count, uint32_t size,
        const std::function<void(uint32_t, uint64_t, uint32_t)> &visit) const {
    uint64_t at = header + MACH_HEADER_SIZE;
    bounds(at, size);
    uint64_t end = at + size;
    for (uint32_t i = 0; i < count; ++i) {
        if (end - at < 8)
            throw Error("truncated load command");
        uint32_t command = u32(at), length = u32(at + 4);
        if (length < 8 || length % 8 || length > end - at)
            throw Error("invalid load command");
        visit(command, at, length);
        at += length;
    }
    if (at != end)
        throw Error("load commands do not fill their declared size");
}

Collection::Segment Collection::readSegment(uint64_t at, uint32_t length) const {
    if (length < SEGMENT_COMMAND_SIZE)
        throw Error("truncated collection segment command");
    uint64_t address = u64(at + 24), vmsize = u64(at + 32), offset = u64(at + 40),
             filesize = u64(at + 48);
    if (filesize > vmsize || address > UINT64_MAX - vmsize)
        throw Error("invalid collection segment command");
    bounds(offset, filesize);
    return {fixed(at + 8), address, vmsize, offset, filesize, {}};
}

// The regions of the chained-fixups table at offset, sorted by address. OpenCore
// appends the starts of the region it injects after the table's declared size,
// and the booter follows their offsets, so the starts are bounded by the
// file-backed segment that holds the table rather than by size.
std::vector<Collection::Region> Collection::chainedStarts(uint64_t offset, uint64_t size) const {
    std::vector<const Segment *> holders;
    for (const Segment &segment : segments_) {
        if (segment.offset <= offset && size <= segment.filesize &&
            offset - segment.offset <= segment.filesize - size)
            holders.push_back(&segment);
    }
    if (holders.size() != 1 || size < CHAINED_FIXUPS_HEADER_SIZE)
        throw Error("chained-fixups table is not inside one collection segment");
    uint64_t room = holders[0]->offset + holders[0]->filesize - offset;
    uint32_t version = u32(offset), starts = u32(offset + 4);
    if (version != 0 || !(CHAINED_FIXUPS_HEADER_SIZE <= starts && starts <= size - 4))
        throw Error("unsupported chained-fixups header");
    uint64_t image = offset + starts;
    room -= starts;
    uint32_t count = u32(image);
    if (count > (room - 4) / 4)
        throw Error("chained starts outside their segment");
    std::vector<Region> regions;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t info = u32(image + 4 + uint64_t{i} * 4);
        if (!info)
            continue;
        if (info > room)
            throw Error("chained segment starts outside their segment");
        regions.push_back(readRegion(image + info, room - info));
    }
    std::stable_sort(regions.begin(), regions.end(),
                     [](const Region &a, const Region &b) { return a.start < b.start; });
    for (size_t i = 1; i < regions.size(); ++i) {
        if (regions[i - 1].end() > regions[i].start)
            throw Error("chained-fixup regions overlap");
    }
    return regions;
}

// The dyld_chained_starts_in_segment at at, which room bytes of the file hold.
Collection::Region Collection::readRegion(uint64_t at, uint64_t room) const {
    if (room < CHAINED_STARTS_IN_SEGMENT_SIZE)
        throw Error("chained segment starts outside their segment");
    bounds(at, CHAINED_STARTS_IN_SEGMENT_SIZE);
    uint32_t length = load32(&data_[at]);
    uint16_t pageSize = load16(&data_[at + 4]), pointerFormat = load16(&data_[at + 6]);
    uint64_t segmentOffset = load64(&data_[at + 8]);
    uint16_t pages = load16(&data_[at + 20]);
    if (pointerFormat != DYLD_CHAINED_PTR_X86_64_KERNEL_CACHE || !pageSize ||
        length < CHAINED_STARTS_IN_SEGMENT_SIZE + 2 * uint64_t{pages} || length > room)
        throw Error("unsupported chained segment starts");
    bounds(at + CHAINED_STARTS_IN_SEGMENT_SIZE, 2 * uint64_t{pages});
    Region region{header_ + segmentOffset, pageSize, {}, std::nullopt};
    for (uint16_t page = 0; page < pages; ++page) {
        uint16_t start = load16(&data_[at + CHAINED_STARTS_IN_SEGMENT_SIZE + 2 * page]);
        if (start != DYLD_CHAINED_PTR_START_NONE && (start & DYLD_CHAINED_PTR_START_MULTI))
            throw Error("unsupported chained page start");
        region.starts.push_back(start);
    }
    return region;
}

Bytes Collection::readCollection(uint64_t address, uint64_t width) const {
    std::optional<uint64_t> offset = fileOffset(segments_, address, width);
    if (!offset)
        throw Error(std::to_string(width) + " bytes at " + hex(address) +
                    " are outside the collection's file-backed segments");
    bounds(*offset, width);
    return Bytes(data_.begin() + static_cast<ptrdiff_t>(*offset),
                 data_.begin() + static_cast<ptrdiff_t>(*offset + width));
}

// Every chain entry in the region that holds address, walked whole once.
const std::set<uint64_t> &Collection::chainEntries(uint64_t address) const {
    static const std::set<uint64_t> none;
    auto region = std::find_if(regions_.begin(), regions_.end(), [&](const Region &r) {
        return r.start <= address && address < r.end();
    });
    if (region == regions_.end())
        return none;
    if (!region->entries) {
        std::set<uint64_t> entries;
        for (size_t page = 0; page < region->starts.size(); ++page) {
            if (region->starts[page] == DYLD_CHAINED_PTR_START_NONE)
                continue;
            uint64_t at = region->start + page * region->pageSize + region->starts[page];
            for (;;) {
                if (at >= region->end())
                    throw Error("fixup chain leaves its region");
                entries.insert(at);
                uint64_t step = load64(readCollection(at, 8).data()) >> NextShift & NextMask;
                if (!step)
                    break;
                at += step;
            }
        }
        region->entries = std::move(entries);
    }
    return *region->entries;
}

uint64_t Collection::rebase(uint64_t address) const {
    if (!chainEntries(address).count(address))
        throw Error("pointer at " + hex(address) + " is not in the collection's fixup chains");
    uint64_t raw = load64(readCollection(address, 8).data());
    if (raw & NotPlain)
        throw Error("pointer at " + hex(address) + " is not a plain chained rebase");
    return base_ + (raw & TargetMask);
}

bool Collection::overlapsFixup(uint64_t address, uint64_t width) const {
    for (uint64_t at : {address - 7, address + width - 1}) {
        for (uint64_t entry : chainEntries(at)) {
            if (entry < address + width && address < entry + 8)
                return true;
        }
    }
    return false;
}

Collection::Entry Collection::entry(const std::string &name) const {
    auto found = entries_.find(name);
    if (found == entries_.end())
        throw Error("collection has no fileset entry " + name);
    auto [address, offset] = found->second;
    bounds(offset, MACH_HEADER_SIZE);
    if (u32(offset) != MH_MAGIC_64 || u32(offset + 4) != CPU_TYPE_X86_64 ||
        u32(offset + 12) != MH_KEXT_BUNDLE)
        throw Error(name + ": fileset entry is not an x86-64 kext");
    Entry entry{name, address, {}, {}};
    auto addSegment = [&](uint32_t command, uint64_t at, uint32_t length) {
        if (command != LC_SEGMENT_64)
            return;
        if (length < SEGMENT_COMMAND_SIZE)
            throw Error(name + ": truncated segment command");
        std::string segmentName = fixed(at + 8);
        Segment segment{segmentName, u64(at + 24), u64(at + 32), u64(at + 40), u64(at + 48), {}};
        uint32_t nsects = u32(at + 64);
        if (nsects > (length - SEGMENT_COMMAND_SIZE) / SECTION_HEADER_SIZE ||
            segment.filesize > segment.vmsize || entry.segments.count(segmentName))
            throw Error(name + ": invalid segment command " + segmentName);
        bounds(segment.offset, segment.filesize);
        for (uint32_t i = 0; i < nsects; ++i) {
            uint64_t header = at + SEGMENT_COMMAND_SIZE + uint64_t{i} * SECTION_HEADER_SIZE;
            std::string section = fixed(header);
            if (segment.sections.count(section))
                throw Error(name + ": duplicate section " + segmentName + "," + section);
            segment.sections[section] = {u64(header + 32), u64(header + 40)};
        }
        entry.order.push_back(segmentName);
        entry.segments[segmentName] = std::move(segment);
    };
    forEachCommand(offset, u32(offset + 16), u32(offset + 20), addSegment);
    auto text = entry.segments.find("__TEXT");
    if (text == entry.segments.end() || text->second.address != address ||
        text->second.offset != offset)
        throw Error(name + ": fileset entry header does not start its __TEXT segment");
    return entry;
}

Bytes Collection::read(const Entry &entry, uint64_t address, uint64_t width) const {
    std::vector<Segment> segments;
    for (const std::string &name : entry.order)
        segments.push_back(entry.segments.at(name));
    std::optional<uint64_t> offset = fileOffset(segments, address, width);
    if (!offset)
        throw Error(entry.name + ": " + std::to_string(width) + " bytes at " + hex(address) +
                    " are outside its file-backed segments");
    bounds(*offset, width);
    return Bytes(data_.begin() + static_cast<ptrdiff_t>(*offset),
                 data_.begin() + static_cast<ptrdiff_t>(*offset + width));
}

namespace {

// The staged __TEXT address, from which the staged image's offsets count, after
// requiring every staged segment at its staged place relative to the header.
uint64_t checkSegments(const Collection::Entry &entry, const Image &staged) {
    std::set<std::string> names;
    bool text = false;
    uint64_t base = 0;
    for (const Segment &segment : staged.segments) {
        names.insert(segment.name);
        if (segment.name == "__TEXT" && !text) {
            text = true;
            base = segment.address;
        }
    }
    if (!text || names.size() != staged.segments.size())
        throw Error(entry.name + ": staged kext needs one __TEXT and unique segments");
    for (const Segment &segment : staged.segments) {
        if (segment.name == "__LINKEDIT")
            continue;
        auto linked = entry.segments.find(segment.name);
        if (linked == entry.segments.end() ||
            linked->second.address - entry.address != segment.address - base ||
            linked->second.vmsize < segment.vmsize)
            throw Error(entry.name + ": segment " + segment.name +
                        " changed its layout relative to the header");
    }
    return base;
}

// Require a site to target its placeholder, which must lie in the manifest's tail.
void checkSite(const Collection &collection, const Collection::Entry &entry,
               const manifest::Site &site, uint64_t tailStart, uint64_t tailEnd,
               const std::string &name) {
    std::string where = name + " at offset " + hex(site.offset);
    if (site.placeholder < tailStart || site.placeholder >= tailEnd)
        throw Error(where + ": placeholder is outside the reserved manifest tail");
    uint64_t address = entry.address + site.offset;
    uint64_t expected = entry.address + site.placeholder + static_cast<uint64_t>(site.addend);
    uint64_t actual;
    if (site.width == 8 && site.opcode == manifest::OpcodePointer) {
        // The site must be the consumer's own file-backed bytes.
        collection.read(entry, address, 8);
        try {
            actual = collection.rebase(address);
        } catch (const Error &error) {
            throw Error(where + ": " + error.what());
        }
    } else if (site.width == 4 &&
               (site.opcode == manifest::OpcodeCall || site.opcode == manifest::OpcodeJump)) {
        Bytes instruction = collection.read(entry, address - 1, 5);
        if (collection.overlapsFixup(address - 1, 5))
            throw Error(where + ": a fixup-chain entry overlaps the branch");
        if (instruction[0] != site.opcode)
            throw Error(where + ": branch opcode " + hexByte(instruction[0]) + " is not " +
                        hexByte(site.opcode));
        int32_t displacement = static_cast<int32_t>(load32(&instruction[1]));
        actual = address + 4 + static_cast<uint64_t>(static_cast<int64_t>(displacement));
    } else {
        throw Error(where + ": unsupported site form");
    }
    if (actual != expected)
        throw Error(where + ": targets " + hex(actual) + ", not its placeholder " + hex(expected));
}

} // namespace

size_t checkPrelinked(const Collection &collection, const std::string &identifier,
                      const Image &staged) {
    Collection::Entry entry = collection.entry(identifier);
    uint64_t base = checkSegments(entry, staged);
    const Section &section = staged.section("__DATA", "__graft");
    uint64_t sectionEnd = std::min<uint64_t>(section.offset + section.size, staged.data.size());
    Bytes raw(staged.data.begin() + std::min<uint64_t>(section.offset, sectionEnd),
              staged.data.begin() + static_cast<ptrdiff_t>(sectionEnd));
    // The manifest's declared bytes and its sites.
    const uint64_t headerSize = sizeof(manifest::Header);
    if (raw.size() < headerSize)
        throw Error("manifest section is shorter than its header");
    manifest::Header header = decodeHeader(raw.data());
    if (header.magic != manifest::Magic || header.version != manifest::Version ||
        header.headerSize != headerSize || header.size < headerSize || header.size > raw.size())
        throw Error("section holds no staged manifest");
    if (header.sites.offset < headerSize || header.sites.offset > header.size ||
        header.sites.count > (header.size - header.sites.offset) / sizeof(manifest::Site))
        throw Error("manifest site table outside its declared size");
    std::vector<manifest::Site> sites;
    for (uint32_t i = 0; i < header.sites.count; ++i)
        sites.push_back(
                decodeSite(&raw[header.sites.offset + uint64_t{i} * sizeof(manifest::Site)]));
    Bytes declared(raw.begin(), raw.begin() + header.size);

    uint64_t manifest = entry.address + section.address - base;
    auto data = entry.segments.find("__DATA");
    std::pair<uint64_t, uint64_t> placed{manifest, section.size};
    if (data == entry.segments.end() || !data->second.sections.count("__graft") ||
        data->second.sections.at("__graft") != placed)
        throw Error(identifier + ": __DATA,__graft moved or changed size");
    if (collection.overlapsFixup(manifest, declared.size()))
        throw Error(identifier + ": a fixup-chain entry overlaps the manifest");
    if (collection.read(entry, manifest, declared.size()) != declared)
        throw Error(identifier + ": manifest differs from the staged consumer");
    uint64_t tailStart = section.address - base + declared.size();
    uint64_t tailEnd = section.address - base + section.size;
    for (size_t index = 0; index < sites.size(); ++index)
        checkSite(collection, entry, sites[index], tailStart, tailEnd,
                  identifier + ": site " + std::to_string(index));
    return sites.size();
}

} // namespace graft::tool
