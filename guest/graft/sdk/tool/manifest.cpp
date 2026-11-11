// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
#include "manifest.h"

#include <cstddef>

namespace graft::tool {

using manifest::Header;

manifest::Table Header::*const ManifestTables[14] = {
        &Header::strings,        &Header::libraries,   &Header::imports, &Header::candidates,
        &Header::sites,          &Header::classes,     &Header::slots,   &Header::vtables,
        &Header::overrides,      &Header::segments,    &Header::fills,   &Header::selectors,
        &Header::synthOverrides, &Header::synthClasses};

namespace {

void put(uint8_t *record, size_t offset, uint8_t value) {
    record[offset] = value;
}
void put(uint8_t *record, size_t offset, uint16_t value) {
    store16(record + offset, value);
}
void put(uint8_t *record, size_t offset, uint32_t value) {
    store32(record + offset, value);
}
void put(uint8_t *record, size_t offset, uint64_t value) {
    store64(record + offset, value);
}
void put(uint8_t *record, size_t offset, int64_t value) {
    store64(record + offset, static_cast<uint64_t>(value));
}

// The header holds its tables side by side, from strings to synthClasses, in
// ManifestTables' order, so table i's offset in the header follows from i.
static_assert(offsetof(Header, synthClasses) ==
                              offsetof(Header, strings) + 13 * sizeof(manifest::Table) &&
                      offsetof(Header, gate) ==
                              offsetof(Header, strings) + 14 * sizeof(manifest::Table),
              "the header's tables are contiguous");
size_t tableOffset(size_t i) {
    return offsetof(Header, strings) + i * sizeof(manifest::Table);
}

// A record's bytes, built field by field at each field's offset.
template <typename Record>
class Fields {
public:
    template <typename Field>
    Fields &set(size_t offset, Field value) {
        put(bytes_, offset, value);
        return *this;
    }
    void appendTo(Bytes &out) const { out.insert(out.end(), bytes_, bytes_ + sizeof(Record)); }

private:
    uint8_t bytes_[sizeof(Record)] = {};
};

} // namespace

#define FIELD(type, record, member) set(offsetof(manifest::type, member), record.member)

void encode(Bytes &out, const Header &record) {
    Fields<Header> fields;
    fields.FIELD(Header, record, magic)
            .FIELD(Header, record, version)
            .FIELD(Header, record, size)
            .FIELD(Header, record, headerSize)
            .FIELD(Header, record, gate)
            .FIELD(Header, record, matching);
    for (size_t i = 0; i < 14; ++i) {
        size_t offset = tableOffset(i);
        fields.set(offset, (record.*ManifestTables[i]).offset)
                .set(offset + offsetof(manifest::Table, count), (record.*ManifestTables[i]).count);
    }
    fields.appendTo(out);
}

void encode(Bytes &out, const manifest::Library &record) {
    Fields<manifest::Library>()
            .FIELD(Library, record, identifier)
            .FIELD(Library, record, anchor)
            .appendTo(out);
}

void encode(Bytes &out, const manifest::Import &record) {
    Fields<manifest::Import>()
            .FIELD(Import, record, library)
            .FIELD(Import, record, firstCandidate)
            .FIELD(Import, record, candidateCount)
            .FIELD(Import, record, flags)
            .FIELD(Import, record, fallback)
            .appendTo(out);
}

void encode(Bytes &out, const manifest::Site &record) {
    Fields<manifest::Site>()
            .FIELD(Site, record, offset)
            .FIELD(Site, record, placeholder)
            .FIELD(Site, record, addend)
            .FIELD(Site, record, import)
            .FIELD(Site, record, width)
            .FIELD(Site, record, opcode)
            .FIELD(Site, record, reserved)
            .appendTo(out);
}

void encode(Bytes &out, const manifest::Class &record) {
    Fields<manifest::Class>()
            .FIELD(Class, record, name)
            .FIELD(Class, record, vtable)
            .FIELD(Class, record, library)
            .FIELD(Class, record, instanceSize)
            .FIELD(Class, record, parentSlots)
            .FIELD(Class, record, firstSlot)
            .FIELD(Class, record, slotCount)
            .appendTo(out);
}

void encode(Bytes &out, const manifest::Slot &record) {
    Fields<manifest::Slot>().FIELD(Slot, record, index).FIELD(Slot, record, method).appendTo(out);
}

void encode(Bytes &out, const manifest::Vtable &record) {
    Fields<manifest::Vtable>()
            .FIELD(Vtable, record, offset)
            .FIELD(Vtable, record, classIndex)
            .FIELD(Vtable, record, slotCount)
            .FIELD(Vtable, record, firstOverride)
            .FIELD(Vtable, record, overrideCount)
            .FIELD(Vtable, record, firstFill)
            .FIELD(Vtable, record, fillCount)
            .appendTo(out);
}

void encode(Bytes &out, const manifest::Segment &record) {
    Fields<manifest::Segment>()
            .FIELD(Segment, record, origin)
            .FIELD(Segment, record, size)
            .FIELD(Segment, record, name)
            .FIELD(Segment, record, reserved)
            .appendTo(out);
}

void encode(Bytes &out, const manifest::Fill &record) {
    Fields<manifest::Fill>()
            .FIELD(Fill, record, slot)
            .FIELD(Fill, record, diversity)
            .FIELD(Fill, record, opcode)
            .FIELD(Fill, record, reserved)
            .appendTo(out);
}

void encode(Bytes &out, const manifest::SynthOverride &record) {
    Fields<manifest::SynthOverride>()
            .FIELD(SynthOverride, record, impl)
            .FIELD(SynthOverride, record, firstSelector)
            .FIELD(SynthOverride, record, selectorCount)
            .FIELD(SynthOverride, record, diversity)
            .FIELD(SynthOverride, record, opcode)
            .FIELD(SynthOverride, record, flags)
            .FIELD(SynthOverride, record, reserved)
            .appendTo(out);
}

void encode(Bytes &out, const manifest::SynthClass &record) {
    Fields<manifest::SynthClass>()
            .FIELD(SynthClass, record, vtableOffset)
            .FIELD(SynthClass, record, name)
            .FIELD(SynthClass, record, parentClass)
            .FIELD(SynthClass, record, siblingClass)
            .FIELD(SynthClass, record, library)
            .FIELD(SynthClass, record, vtableSlotCap)
            .FIELD(SynthClass, record, parentRegion)
            .FIELD(SynthClass, record, firstOverride)
            .FIELD(SynthClass, record, overrideCount)
            .appendTo(out);
}

#undef FIELD

void encode(Bytes &out, uint32_t record) {
    uint8_t bytes[4];
    store32(bytes, record);
    out.insert(out.end(), bytes, bytes + 4);
}

Header decodeHeader(const uint8_t *at) {
    Header header{};
    header.magic = load32(at + offsetof(Header, magic));
    header.version = load32(at + offsetof(Header, version));
    header.size = load32(at + offsetof(Header, size));
    header.headerSize = load32(at + offsetof(Header, headerSize));
    header.gate = load32(at + offsetof(Header, gate));
    header.matching = load32(at + offsetof(Header, matching));
    for (size_t i = 0; i < 14; ++i) {
        size_t offset = tableOffset(i);
        (header.*ManifestTables[i]).offset = load32(at + offset);
        (header.*ManifestTables[i]).count = load32(at + offset + offsetof(manifest::Table, count));
    }
    return header;
}

manifest::Site decodeSite(const uint8_t *at) {
    using manifest::Site;
    Site site{};
    site.offset = load64(at + offsetof(Site, offset));
    site.placeholder = load64(at + offsetof(Site, placeholder));
    site.addend = static_cast<int64_t>(load64(at + offsetof(Site, addend)));
    site.import = load32(at + offsetof(Site, import));
    site.width = at[offsetof(Site, width)];
    site.opcode = at[offsetof(Site, opcode)];
    site.reserved = load16(at + offsetof(Site, reserved));
    return site;
}

} // namespace graft::tool
