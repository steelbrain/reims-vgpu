// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Host test of the write transaction (src/WritePlan.h) over an in-memory
// stand-in for kernel memory.
#include "Check.h"
#include "WritePlan.h"
#include <initializer_list>
#include <stdio.h>
#include <string.h>

using namespace graft;

struct Memory {
    bool active = false;
    size_t enters = 0, leaves = 0, reads = 0, stores = 0;
    size_t patchCount = 0;
    size_t falseRead = 0, corruptStore = 0, dropStore = 0;
    const uint8_t *readAddresses[32]{};

    bool matches(const uint8_t *address, const uint8_t *expected, size_t size) {
        CHECK(reads < 32);
        readAddresses[reads++] = address;
        if (reads <= patchCount)
            CHECK(!active && stores == 0);
        else
            CHECK(active);
        bool equal = !memcmp(address, expected, size);
        return reads == falseRead ? false : equal;
    }
    void enter() {
        CHECK(!active && reads == patchCount && stores == 0);
        active = true;
        ++enters;
    }
    void store(uint8_t *address, const uint8_t *bytes, size_t size) {
        CHECK(active && reads >= patchCount);
        ++stores;
        if (stores == dropStore)
            return;
        memcpy(address, bytes, size);
        if (stores == corruptStore)
            address[size - 1] ^= 0xff;
    }
    void leave() {
        CHECK(active && enters == 1);
        active = false;
        ++leaves;
    }
};

struct Fixture {
    uint8_t bytes[16]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint8_t before[16]{};
    const uint8_t after[8]{90, 91, 92, 93, 94, 95, 96, 97};
    Patch patches[2]{};
    Fixture() {
        memcpy(before, bytes, sizeof(before));
        patches[0] = {bytes, before, after, 4};
        patches[1] = {bytes + 8, before + 8, after + 4, 4};
    }
};

static void success() {
    Fixture fixture;
    Memory memory;
    memory.patchCount = 2;
    CHECK(write::apply(fixture.patches, 2, memory) == WriteError::None);
    CHECK(!memcmp(fixture.bytes, fixture.after, 4));
    CHECK(!memcmp(fixture.bytes + 8, fixture.after + 4, 4));
    CHECK(!memcmp(fixture.bytes + 4, fixture.before + 4, 4));
    CHECK(memory.enters == 1 && memory.leaves == 1 && memory.stores == 2 && memory.reads == 4);
    CHECK(memory.readAddresses[0] == fixture.bytes && memory.readAddresses[1] == fixture.bytes + 8);
    CHECK(memory.readAddresses[2] == fixture.bytes && memory.readAddresses[3] == fixture.bytes + 8);
    CHECK(!memory.active);
    puts("write transaction: pass (all old values before stores, all new values read back)");
}

static void changed() {
    Fixture fixture;
    fixture.bytes[9] ^= 1;
    uint8_t unchanged[sizeof(fixture.bytes)];
    memcpy(unchanged, fixture.bytes, sizeof(unchanged));
    Memory memory;
    memory.patchCount = 2;
    CHECK(write::apply(fixture.patches, 2, memory) == WriteError::Changed);
    CHECK(memory.reads == 2 && !memory.enters && !memory.leaves && !memory.stores);
    CHECK(!memcmp(fixture.bytes, unchanged, sizeof(unchanged)));
    puts("write refusal: pass (later old-value mismatch changes no targets)");
}

static void rollback() {
    for (bool corruptWrite : {false, true}) {
        Fixture fixture;
        Memory memory;
        memory.patchCount = 2;
        if (corruptWrite)
            memory.corruptStore = 2;
        else
            memory.falseRead = 3; // even a first-after miss still checks both
        CHECK(write::apply(fixture.patches, 2, memory) == WriteError::Verify);
        CHECK(!memcmp(fixture.bytes, fixture.before, sizeof(fixture.bytes)));
        CHECK(memory.enters == 1 && memory.leaves == 1 && memory.stores == 4 && memory.reads == 6);
        for (size_t pass = 0; pass < 3; ++pass) {
            CHECK(memory.readAddresses[pass * 2] == fixture.bytes);
            CHECK(memory.readAddresses[pass * 2 + 1] == fixture.bytes + 8);
        }
        CHECK(!memory.active);
    }
    Fixture fixture;
    Memory memory;
    memory.patchCount = 2;
    memory.corruptStore = 2;
    memory.dropStore = 3;
    CHECK(write::apply(fixture.patches, 2, memory) == WriteError::Rollback);
    CHECK(memcmp(fixture.bytes, fixture.before, 4));
    CHECK(!memcmp(fixture.bytes + 8, fixture.before + 8, 4));
    CHECK(memory.reads == 6 && memory.stores == 4 && memory.leaves == 1 && !memory.active);
    puts("write rollback: pass (restoration verified, failed restoration distinguished)");
}

static void declarations() {
    Fixture fixture;
    Memory memory;
    memory.patchCount = 2;
    auto reject = [&]() {
        CHECK(write::apply(fixture.patches, 2, memory) == WriteError::Declaration);
        CHECK(memory.reads == 0 && memory.stores == 0 && memory.enters == 0);
    };
    fixture.patches[1].address = fixture.bytes + 3;
    reject();
    fixture.patches[1].address = fixture.bytes + 8;
    fixture.patches[0].before = fixture.bytes;
    reject(); // self-alias destroys the old snapshot
    fixture.patches[0].before = fixture.before;
    fixture.patches[0].after = fixture.bytes + 8;
    reject(); // another patch would destroy the replacement snapshot
    fixture.patches[0].after = fixture.after;
    fixture.patches[1].address = reinterpret_cast<uint8_t *>(fixture.patches);
    reject(); // a target must not rewrite the pending declarations
    fixture.patches[1].address = fixture.bytes + 8;
    fixture.patches[0].size = 0;
    reject();
    fixture.patches[0].size = 4;
    fixture.patches[0].after = nullptr;
    reject();
    fixture.patches[0].after = fixture.after;
    fixture.patches[0].address = reinterpret_cast<uint8_t *>(UINTPTR_MAX - 1);
    reject();
    fixture.patches[0].address = fixture.bytes;
    fixture.patches[0].before = reinterpret_cast<const uint8_t *>(UINTPTR_MAX - 1);
    reject();
    CHECK(write::apply(nullptr, 2, memory) == WriteError::Declaration);
    CHECK(write::apply(fixture.patches, SIZE_MAX / sizeof(Patch) + 1, memory) ==
          WriteError::Declaration);
    CHECK(!memcmp(fixture.bytes, fixture.before, sizeof(fixture.bytes)));
    puts("write declarations: pass (overlap, snapshot/metadata aliasing, null, size overflow)");
}

static void emptyAndAdjacent() {
    Memory empty;
    CHECK(write::apply(nullptr, 0, empty) == WriteError::None);
    CHECK(!empty.reads && !empty.stores && !empty.enters && !empty.leaves);
    Fixture fixture;
    fixture.patches[1] = {fixture.bytes + 4, fixture.before + 4, fixture.after + 4, 4};
    Memory adjacent;
    adjacent.patchCount = 2;
    CHECK(write::apply(fixture.patches, 2, adjacent) == WriteError::None);
    CHECK(!memcmp(fixture.bytes, fixture.after, 8));
    puts("write edge cases: pass (empty transaction and adjacent non-overlapping targets)");
}

int main() {
    success();
    changed();
    rollback();
    declarations();
    emptyAndAdjacent();
}
