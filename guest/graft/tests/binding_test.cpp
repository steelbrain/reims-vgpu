// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Host test of the import-site plan (src/Binding.h), its pointer and branch
// arithmetic (src/x86_64/Fixup.h) and class size and vtable length comparison
// (src/Layout.h).
#include "Binding.h"
#include "Check.h"
#include "Layout.h"
#include <initializer_list>
#include <stdio.h>
#include <string.h>

using namespace graft;

static void arithmetic() {
    using fixup::Error;
    uint64_t out = 7;
    CHECK(fixup::add(0x8000000000000000ULL, INT64_MIN, out) && out == 0);
    CHECK(!fixup::add(0, -1, out) && out == 0);
    CHECK(!fixup::add(UINT64_MAX, 1, out) && out == 0);
    CHECK(fixup::add(UINT64_MAX, -1, out) && out == UINT64_MAX - 1);
    out = 7;
    auto pointer = [&](uint64_t old, uint64_t placeholder, uint64_t real, int64_t addend) {
        return fixup::replacement(8, 0, 0, 0, old, placeholder, real, addend, out);
    };
    CHECK(pointer(0x1010, 0x1000, 0x9000, 16) == Error::None && out == 0x9010);
    CHECK(pointer(0xff8, 0x1000, 0x9000, -8) == Error::None && out == 0x8ff8);
    out = 7;
    CHECK(pointer(0, 0x1000, 0x9000, 0) == Error::Placeholder && out == 7);
    CHECK(pointer(0, UINT64_MAX, 0x9000, 1) == Error::Address && out == 7);
    CHECK(pointer(1, 1, 0, -1) == Error::Address && out == 7);
    CHECK(fixup::replacement(2, 0, 0, 0, 0, 0, 0, 0, out) == Error::Width);
    CHECK(fixup::replacement(8, 0xe8, 0, 0, 0, 0, 0, 0, out) == Error::Opcode);

    // Signed displacement endpoints around a high kernel address, both
    // directions. Decode the result independently to check the target.
    const uint64_t next = 0xffffff8001000004ULL;
    const uint64_t placeholder = next - 12;
    for (uint8_t opcode : {uint8_t{0xe8}, uint8_t{0xe9}}) {
        for (int64_t distance : {int64_t{INT32_MIN}, int64_t{-1}, int64_t{0}, int64_t{INT32_MAX}}) {
            uint64_t real = next + static_cast<uint64_t>(distance);
            CHECK(fixup::replacement(4, opcode, opcode, next - 4, 0xfffffff4, placeholder, real, 0,
                                     out) == Error::None);
            CHECK(static_cast<uint32_t>(out) == static_cast<uint32_t>(distance));
        }
    }
    out = 7;
    CHECK(fixup::replacement(4, 0xe8, 0xe8, next - 4, 0xfffffff4, placeholder,
                             next + uint64_t{INT32_MAX} + 1, 0, out) == Error::Range &&
          out == 7);
    CHECK(fixup::replacement(4, 0xe8, 0xe8, next - 4, 0xfffffff4, placeholder,
                             next - (uint64_t{INT32_MAX} + 2), 0, out) == Error::Range &&
          out == 7);
    CHECK(fixup::replacement(4, 0xe8, 0xe9, next - 4, 0xfffffff4, placeholder, next, 0, out) ==
          Error::Opcode);
    CHECK(fixup::replacement(4, 0xe8, 0xe8, UINT64_MAX - 2, 0, placeholder, next, 0, out) ==
          Error::Address);
    CHECK(fixup::replacement(4, 0xe8, 0xe8, 0, 0x80000000, placeholder, next, 0, out) ==
          Error::Address);
    CHECK(fixup::replacement(4, 0xe8, 0xe8, next - 4, 0x100000000, placeholder, next, 0, out) ==
          Error::Address);
    puts("fixups: pass (pointers, signed addends, branch endpoints, overflow)");
}

static void preflight() {
    using namespace binding;
    uint8_t image[64] = {};
    uint64_t placeholder = 0x1000;
    uint32_t displacement = 0x1000 - 0x401c;
    memcpy(image + 8, &placeholder, 8);
    image[23] = 0xe8;
    memcpy(image + 24, &displacement, 4);
    uint8_t before[sizeof(image)];
    memcpy(before, image, sizeof(image));
    Site sites[] = {{8, placeholder, 0, 8, 0}, {24, placeholder, 0, 4, 0xe8}};
    Write writes[2] = {};
    auto resolve = [](const Site &, uint64_t &address) {
        address = 0x9000;
        return Resolution::Found;
    };
    // read serves old bytes from the buffer; sign places each site at a low load
    // address, as the running consumer's mapping would, so a branch stays in range.
    auto read = [&](uint64_t offset, uint8_t width, uint64_t &old, uint8_t &opcode) {
        if (offset > sizeof(image) || width > sizeof(image) - offset || (width == 4 && !offset))
            return false;
        old = 0;
        memcpy(&old, image + offset, width);
        opcode = width == 4 ? image[offset - 1] : 0;
        return true;
    };
    auto run = [&](size_t capacity = 2, uint64_t base = 0x4000) {
        auto sign = [base](uint64_t offset) -> uint64_t {
            return offset > UINT64_MAX - base ? 0 : base + offset;
        };
        return plan(sites, 2, writes, capacity, resolve, read, sign);
    };
    CHECK(run().error == Error::None);
    CHECK(writes[0].offset == 8 && writes[0].before == placeholder && writes[0].after == 0x9000);
    CHECK(writes[1].offset == 24 && writes[1].before == displacement && writes[1].after == 0x4fe4);
    auto capacity = run(1);
    CHECK(capacity.error == Error::Capacity && capacity.site == NoSite);
    CHECK(!memcmp(before, image, sizeof(image)));
}

static void failures() {
    using namespace binding;
    uint8_t image[32] = {};
    uint64_t placeholder = 0x1000;
    memcpy(image + 8, &placeholder, 8);
    memcpy(image + 16, &placeholder, 8);
    Site sites[] = {{8, placeholder, 0, 8, 0}, {16, placeholder, 0, 8, 0}};
    Write writes[2] = {};
    auto resolve = [](const Site &, uint64_t &address) {
        address = 0x9000;
        return Resolution::Found;
    };
    auto read = [&](uint64_t offset, uint8_t width, uint64_t &old, uint8_t &opcode) {
        if (offset > sizeof(image) || width > sizeof(image) - offset || (width == 4 && !offset))
            return false;
        old = 0;
        memcpy(&old, image + offset, width);
        opcode = width == 4 ? image[offset - 1] : 0;
        return true;
    };
    auto signAt = [](uint64_t base) {
        return [base](uint64_t offset) -> uint64_t {
            return offset > UINT64_MAX - base ? 0 : base + offset;
        };
    };
    auto run = [&](auto r, auto sign) { return plan(sites, 2, writes, 2, r, read, sign); };
    uint8_t before[sizeof(image)];
    memcpy(before, image, sizeof(image));
    sites[1].placeholder = 0x2000;
    auto failure = run(resolve, signAt(0x4000));
    CHECK(failure.error == Error::Fixup && failure.site == 1 &&
          failure.detail == fixup::Error::Placeholder);
    CHECK(!memcmp(before, image, sizeof(image))); // even after the first site succeeded
    sites[1].placeholder = placeholder;
    sites[1].offset = 12;
    CHECK(run(resolve, signAt(0x4000)).error == Error::Order);
    sites[1].offset = 25;
    CHECK(run(resolve, signAt(0x4000)).error == Error::Bounds);
    sites[1].offset = 16;
    CHECK(run(resolve, signAt(UINT64_MAX - 10)).error == Error::Bounds);
    sites[0].width = 4;
    sites[0].offset = 0;
    CHECK(run(resolve, signAt(0x4000)).error == Error::Bounds);
    sites[0].width = 8;
    sites[0].offset = 8;
    auto missing = [](const Site &, uint64_t &) { return Resolution::Missing; };
    CHECK(run(missing, signAt(0)).error == Error::Missing);
    auto ambiguous = [](const Site &, uint64_t &) { return Resolution::Ambiguous; };
    CHECK(run(ambiguous, signAt(0)).error == Error::Ambiguous);
    auto noRead = [](uint64_t, uint8_t, uint64_t &, uint8_t &) { return false; };
    CHECK(plan(sites, 2, writes, 2, resolve, noRead, signAt(0)).error == Error::Bounds);
    CHECK(plan(nullptr, 0, nullptr, 0, resolve, read, signAt(0)).error == Error::None);
    puts("binding: pass (whole-plan validation, bounds, overlap, missing/ambiguous)");
}

static void layouts() {
    using namespace layout;
    Requirement r{"Parent", 64, 3};
    CHECK(compare(r, 64, 3) == Error::None);
    CHECK(compare(r, 0, 3) == Error::MissingClass);
    CHECK(compare(r, 72, 3) == Error::Size);
    CHECK(compare(r, 64, 4) == Error::Length);
    CHECK(compare(r, 64, 2) == Error::Length);
    CHECK(compare({nullptr, 64, 3}, 64, 3) == Error::Declaration);
    CHECK(compare({"Parent", 0, 3}, 64, 3) == Error::Declaration);
    CHECK(compare({"Parent", 64, 0}, 64, 3) == Error::Declaration);
    puts("layouts: pass (declaration, missing class, size, length)");
}

// A staged write that lands in a forbidden region (the consumer's manifest
// section) overlaps it; one wholly before or after does not, and overflow or an
// empty range never overlaps.
static void overlap() {
    using binding::overlaps;
    const uint64_t start = 0x2000, size = 0x100;      // [0x2000, 0x2100)
    CHECK(overlaps(0x2000, 8, start, size));          // at the start
    CHECK(overlaps(0x20f8, 8, start, size));          // last word inside
    CHECK(overlaps(0x1ffc, 8, start, size));          // straddles the start
    CHECK(!overlaps(0x1ff8, 8, start, size));         // the word just before
    CHECK(!overlaps(0x2100, 8, start, size));         // the word just after
    CHECK(!overlaps(0x2080, 0, start, size));         // an empty write
    CHECK(!overlaps(0x2080, 8, start, 0));            // an empty region
    CHECK(!overlaps(UINT64_MAX - 2, 8, start, size)); // address+width overflows
    puts("overlap: pass (write inside the manifest section refused, outside allowed)");
}

int main() {
    arithmetic();
    preflight();
    failures();
    layouts();
    overlap();
}
