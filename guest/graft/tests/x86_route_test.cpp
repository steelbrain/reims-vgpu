// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Host test of x86-64 instruction decoding and route building
// (src/x86_64/X86Route.h). It runs on x86-64 code, so it is built for x86-64
// and runs under Rosetta on Apple silicon.
#include "Check.h"
#include "Policy.h"
#include "x86_64/X86Route.h"
#include <initializer_list>
#include <stdio.h>
#include <string.h>
#if defined(__x86_64__)
#include <sys/mman.h>
#endif

using namespace graft::x86;

static uint64_t pointerAt(const uint8_t *at) {
    uint64_t value;
    memcpy(&value, at, sizeof(value));
    return value;
}

static void decoding() {
    struct Case {
        uint8_t bytes[15];
        size_t size;
        Kind kind;
        bool call = false;
    };
    const Case cases[] = {
            {{0x55}, 1, Kind::Plain},                         // push rbp
            {{0x48, 0x89, 0xe5}, 3, Kind::Plain},             // mov rbp,rsp
            {{0x41, 0x57}, 2, Kind::Plain},                   // push r15
            {{0x48, 0x83, 0xec, 0x20}, 4, Kind::Plain},       // sub rsp,32
            {{0x48, 0x81, 0xec, 0, 1, 0, 0}, 7, Kind::Plain}, // sub rsp,256
            {{0x48, 0x8b, 0x05, 1, 2, 3, 4}, 7, Kind::RipRelative},
            {{0x48, 0x8d, 0x3d, 1, 2, 3, 4}, 7, Kind::RipRelative},
            {{0x65, 0x48, 0x8b, 0x04, 0x25, 1, 2, 3, 4}, 9, Kind::Plain},
            {{0x66, 0x81, 0xc0, 1, 2}, 5, Kind::Plain}, // add ax,imm16
            {{0x66, 0x48, 0x81, 0xc0, 1, 2, 3, 4}, 8, Kind::Plain},
            {{0x48, 0xb8, 1, 2, 3, 4, 5, 6, 7, 8}, 10, Kind::Plain},
            {{0xf3, 0x0f, 0x1e, 0xfa}, 4, Kind::Plain}, // endbr64
            {{0xf3, 0x90}, 2, Kind::Plain},             // pause
            {{0x0f, 0x1f, 0x84, 0, 0, 0, 0, 0}, 8, Kind::Plain},
            {{0x0f, 0x1f, 0x05, 1, 2, 3, 4}, 7, Kind::Plain},
            {{0xf7, 0xc0, 1, 2, 3, 4}, 6, Kind::Plain}, // test eax,imm32
            {{0xf7, 0xd8}, 2, Kind::Plain},             // neg eax
            {{0x0f, 0xba, 0xe0, 3}, 4, Kind::Plain},    // bt eax,3
            {{0x0f, 0x45, 0xc1}, 3, Kind::Plain},       // cmovne eax,ecx
            {{0xe8, 1, 2, 3, 4}, 5, Kind::Call, true},
            {{0x74, 0x7f}, 2, Kind::Conditional},
            {{0x0f, 0x85, 1, 2, 3, 4}, 6, Kind::Conditional},
            {{0xff, 0x15, 1, 2, 3, 4}, 6, Kind::RipRelative, true}, // call [rip+disp]
            {{0xff, 0xd0}, 2, Kind::Plain, true},                   // call rax
            {{0x41, 0xff, 0xd3}, 3, Kind::Plain, true},             // call r11
            {{0xff, 0x50, 0x10}, 3, Kind::Plain, true},             // call [rax+16]
            {{0xff, 0xe0}, 2, Kind::Plain},                         // jmp rax
            {{0xff, 0xc0}, 2, Kind::Plain},                         // inc eax
            {{0xff, 0x30}, 2, Kind::Plain},                         // push [rax]
    };
    for (const auto &c : cases) {
        Instruction decoded;
        CHECK(decode(c.bytes, c.size, decoded) == Error::None);
        CHECK(decoded.size == c.size && decoded.kind == c.kind && decoded.call == c.call);
        for (size_t length = 0; length < c.size; ++length) {
            Instruction untouched;
            untouched.size = 99;
            CHECK(decode(c.bytes, length, untouched) != Error::None && untouched.size == 99);
        }
    }
    const uint8_t bad[][4] = {{0x67, 0x8b, 0, 0},    {0xc5, 0xf8, 0x77, 0},   {0x62, 0, 0, 0},
                              {0xf0, 0x01, 0, 0},    {0xe2, 0, 0, 0},         {0xc7, 0xf8, 0, 0},
                              {0xff, 0xd8, 0, 0},    {0x8f, 0xc8, 0, 0},      {0xf6, 0xc8, 0, 0},
                              {0x48, 0x66, 0x90, 0}, {0xf3, 0x48, 0x90, 0},   {0x8d, 0xc0, 0, 0},
                              {0x0f, 0x1f, 0xc8, 0}, {0xf3, 0x0f, 0x1e, 0xfb}};
    for (const auto &bytes : bad) {
        Instruction instruction;
        CHECK(decode(bytes, sizeof(bytes), instruction) != Error::None);
    }
    // Long supported forms retain their operand and address field lengths.
    const uint8_t longMove[] = {0x66, 0x64, 0x48, 0xc7, 0x84, 0x24, 0, 0, 0, 0, 1, 2, 3, 4};
    Instruction instruction;
    CHECK(decode(longMove, sizeof(longMove), instruction) == Error::None && instruction.size == 14);
    const uint8_t longImmediate[] = {0x66, 0x64, 0x48, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0};
    CHECK(decode(longImmediate, sizeof(longImmediate), instruction) == Error::None &&
          instruction.size == 12);
    puts("x86 decode: pass (integer entries, ModRM/SIB, prefixes, truncation)");
}

static void plainAndRip() {
    uint8_t code[32];
    memset(code, 0x90, sizeof(code));
    const uint8_t prolog[] = {0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41,
                              0x55, 0x41, 0x54, 0x53, 0x48, 0x83, 0xec, 0x28};
    memcpy(code, prolog, sizeof(prolog));
    RoutePlan plan;
    const uint64_t original = 0xffffff8001000000ULL;
    const uint64_t trampoline = original + 0x100000;
    const uint64_t destination = original + 0x80000000ULL;
    CHECK(buildRoute(code, sizeof(code), original, trampoline, destination, plan).error ==
          Error::None);
    CHECK(plan.displacedSize == 17 && plan.trampolineSize == 31);
    CHECK(!memcmp(plan.original, code, 17) && !memcmp(plan.trampoline, code, 17));
    CHECK(plan.entry[0] == 0xff && plan.entry[1] == 0x25 &&
          pointerAt(plan.entry + 6) == destination);
    CHECK(plan.entry[14] == 0x90 && plan.entry[16] == 0x90);
    CHECK(pointerAt(plan.trampoline + 17 + 6) == original + 17);
    const uint8_t load[] = {0x48, 0x8b, 0x05, 0x79, 0, 0, 0}; // target original+128
    memset(code, 0x90, sizeof(code));
    memcpy(code, load, sizeof(load));
    CHECK(buildRoute(code, sizeof(code), original, trampoline, destination, plan).error ==
          Error::None);
    int32_t rebased;
    memcpy(&rebased, plan.trampoline + 3, 4);
    CHECK(static_cast<__int128>(trampoline) + 7 + rebased == static_cast<__int128>(original) + 128);
    // Range endpoints use a single displacement from a near source.
    const uint64_t target = original + 128;
    uint64_t near = target - 7 - static_cast<uint64_t>(INT32_MAX);
    CHECK(buildRoute(code, sizeof(code), original, near, destination, plan).error == Error::None);
    CHECK(buildRoute(code, sizeof(code), original, near - 1, destination, plan).error ==
          Error::RipRange);
    near = target - 7 + (uint64_t{1} << 31);
    CHECK(buildRoute(code, sizeof(code), original, near, destination, plan).error == Error::None);
    CHECK(buildRoute(code, sizeof(code), original, near + 1, destination, plan).error ==
          Error::RipRange);

    // A trailing immediate is part of the next-IP used by a RIP operand.
    const uint8_t store[] = {0xc7, 0x05, 0x76, 0, 0, 0, 1, 2, 3, 4};
    memset(code, 0x90, sizeof(code));
    memcpy(code, store, sizeof(store));
    CHECK(buildRoute(code, sizeof(code), original, trampoline, destination, plan).error ==
          Error::None);
    memcpy(&rebased, plan.trampoline + 2, 4);
    CHECK(static_cast<__int128>(trampoline) + sizeof(store) + rebased ==
          static_cast<__int128>(original) + 128);
    puts("x86 route: pass (whole instructions, absolute route, RIP displacement endpoints)");
}

static void branches() {
    uint8_t code[32];
    RoutePlan plan;
    // A displaced call whose return point falls inside the 14-byte window is
    // refused: a thread still in the callee would return into the jump.
    memset(code, 0x90, sizeof(code));
    code[0] = 0xe8;
    uint32_t delta = 0x100 - 5;
    memcpy(code + 1, &delta, 4);
    CHECK(buildRoute(code, sizeof(code), 0x1000, 0x10000000000ULL, 0x5000, plan).error ==
          Error::CallReturn);

    // A call whose return point is at the window's end (placed at offset 9) is
    // displaced and relocated: a very distant trampoline still reaches its target
    // by an inline absolute pointer, without clobbering a general register.
    memset(code, 0x90, sizeof(code));
    code[9] = 0xe8;
    delta = 0x100 - 14;
    memcpy(code + 10, &delta, 4);
    CHECK(buildRoute(code, sizeof(code), 0x1000, 0x10000000000ULL, 0x5000, plan).error ==
          Error::None);
    CHECK(plan.trampoline[9] == 0xff && plan.trampoline[10] == 0x15);
    CHECK(plan.trampoline[15] == 0xeb && plan.trampoline[16] == 8);
    CHECK(pointerAt(plan.trampoline + 17) == 0x1100 && plan.trampolineSize == 39);

    // An indirect call returns just past itself as a relative one does, so it is
    // refused alike when that point falls inside the window: through a register,
    // through memory, or through a RIP-relative pointer.
    const uint8_t indirect[][6] = {
            {0xff, 0xd0}, {0x41, 0xff, 0xd3}, {0xff, 0x50, 0x10}, {0xff, 0x15, 0x40, 0, 0, 0}};
    for (const auto &call : indirect) {
        memset(code, 0x90, sizeof(code));
        memcpy(code + 4, call, sizeof(call));
        CHECK(buildRoute(code, sizeof(code), 0x1000, 0x3000, 0x5000, plan).error ==
              Error::CallReturn);
    }
    // One that returns at the window's end is displaced and copied as it is.
    memset(code, 0x90, sizeof(code));
    code[12] = 0xff;
    code[13] = 0xd0;
    CHECK(buildRoute(code, sizeof(code), 0x1000, 0x3000, 0x5000, plan).error == Error::None);
    CHECK(plan.displacedSize == 14 && plan.trampoline[12] == 0xff && plan.trampoline[13] == 0xd0);

    memset(code, 0x90, sizeof(code));
    code[0] = 0x75;
    code[1] = 2; // jump to instruction at offset 4
    CHECK(buildRoute(code, sizeof(code), 0x1000, 0x3000, 0x5000, plan).error == Error::None);
    CHECK(plan.trampoline[0] == 0x74 && plan.trampoline[1] == JumpSize);
    CHECK(pointerAt(plan.trampoline + 8) == 0x3012); // 16-byte branch, two NOPs

    memset(code, 0x90, sizeof(code));
    code[0] = 0x0f;
    code[1] = 0x8f;
    delta = 0x200 - 6;
    memcpy(code + 2, &delta, 4);
    CHECK(buildRoute(code, sizeof(code), 0x1000, 0x10000000000ULL, 0x5000, plan).error ==
          Error::None);
    CHECK(plan.trampoline[0] == 0x7e && pointerAt(plan.trampoline + 8) == 0x1200);

    // A backward internal target maps to the relocated start, not the
    // installed route. The conditional branch begins at source offset 8.
    memset(code, 0x90, sizeof(code));
    code[8] = 0x74;
    code[9] = 0xf6;
    CHECK(buildRoute(code, sizeof(code), 0x1000, 0x3000, 0x5000, plan).error == Error::None);
    CHECK(pointerAt(plan.trampoline + 8 + 8) == 0x3000);

    // Branching into the middle of a displaced MOV cannot be reconstructed
    // as a branch to the start of that instruction.
    memset(code, 0x90, sizeof(code));
    code[0] = 0x75;
    code[1] = 1;
    code[2] = 0xb8;
    memset(code + 3, 0, 4);
    CHECK(buildRoute(code, sizeof(code), 0x1000, 0x3000, 0x5000, plan).error ==
          Error::InternalTarget);

    // An unconditional tail branch can be displaced if its end supplies
    // enough bytes for the entry patch; an earlier one bounds a thunk.
    memset(code, 0x90, sizeof(code));
    code[9] = 0xe9;
    delta = 0x100 - 14;
    memcpy(code + 10, &delta, 4);
    CHECK(buildRoute(code, sizeof(code), 0x1000, 0x3000, 0x5000, plan).error == Error::None);
    CHECK(pointerAt(plan.trampoline + 9 + 6) == 0x1100);
    code[0] = 0xeb;
    code[1] = 12;
    CHECK(buildRoute(code, sizeof(code), 0x1000, 0x3000, 0x5000, plan).error == Error::Terminal);
    puts("x86 branches: pass (calls, Jcc, internal targets, terminal thunks)");
}

static void failures() {
    uint8_t code[32];
    memset(code, 0x90, sizeof(code));
    RoutePlan plan;
    memset(&plan, 0xa5, sizeof(plan));
    RoutePlan before = plan;
    CHECK(buildRoute(nullptr, 32, 0x1000, 0x3000, 0x5000, plan).error == Error::Input);
    CHECK(buildRoute(code, 13, 0x1000, 0x3000, 0x5000, plan).error == Error::Truncated);
    CHECK(buildRoute(code, 32, UINT64_MAX - 10, 0x3000, 0x5000, plan).error == Error::Address);
    CHECK(buildRoute(code, 32, 0x1000, UINT64_MAX - 20, 0x5000, plan).error == Error::Address);
    CHECK(buildRoute(code, 32, 0x1000, 0x1008, 0x5000, plan).error == Error::Overlap);
    CHECK(buildRoute(code, 32, 0x1000, 0x3000, 0x1001, plan).error == Error::Overlap);
    CHECK(buildRoute(code, 32, 0x1000, 0x3000, 0x3001, plan).error == Error::Overlap);
    code[0] = 0xc3;
    CHECK(buildRoute(code, 32, 0x1000, 0x3000, 0x5000, plan).error == Error::Terminal);
    code[0] = 0xc5;
    CHECK(buildRoute(code, 32, 0x1000, 0x3000, 0x5000, plan).error == Error::Unsupported);
    memset(code, 0x90, sizeof(code));
    const uint8_t endbr[] = {0xf3, 0x0f, 0x1e, 0xfa};
    memcpy(code, endbr, sizeof(endbr));
    CHECK(buildRoute(code, 32, 0x1000, 0x3000, 0x5000, plan).error == Error::ControlFlowProtection);
    memset(code, 0x90, sizeof(code));
    const uint8_t selfRead[] = {0x8b, 0x05, 0xfa, 0xff, 0xff, 0xff};
    memcpy(code, selfRead, sizeof(selfRead));
    CHECK(buildRoute(code, 32, 0x1000, 0x3000, 0x5000, plan).error == Error::InternalReference);
    CHECK(!memcmp(&plan, &before, sizeof(plan)));
    puts("x86 refusal: pass (unchanged output, bounds, overlap, unsupported forms)");
}

static void boundedInputs() {
    uint32_t state = 0x12345678;
    uint8_t bytes[32];
    for (size_t trial = 0; trial < 20000; ++trial) {
        for (auto &byte : bytes) {
            state = state * 1664525 + 1013904223;
            byte = state >> 24;
        }
        size_t available = trial % (sizeof(bytes) + 1);
        Instruction instruction;
        if (decode(bytes, available, instruction) == Error::None)
            CHECK(instruction.size > 0 && instruction.size <= 15 && instruction.size <= available);
        RoutePlan plan;
        if (buildRoute(bytes, available, 0x1000, 0x3000, 0x5000, plan).error == Error::None) {
            CHECK(plan.displacedSize >= JumpSize && plan.displacedSize <= MaxDisplaced);
            CHECK(plan.displacedSize <= available && plan.trampolineSize <= MaxTrampoline);
        }
    }
    puts("x86 bounded inputs: pass (20,000 deterministic instruction streams)");
}

#if defined(__x86_64__)
static void execution() {
    auto memory = static_cast<uint8_t *>(
            mmap(nullptr, 4096, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANON, -1, 0));
    CHECK(memory != MAP_FAILED);
    auto run = [&](const uint8_t *bytes, size_t length, int argument, int expected) {
        memset(memory, 0x90, 4096);
        memcpy(memory, bytes, length);
        const uint8_t replacement[] = {0xb8, 99, 0, 0, 0, 0xc3};
        memcpy(memory + 256, replacement, sizeof(replacement));
        RoutePlan plan;
        auto original = reinterpret_cast<uint64_t>(memory);
        CHECK(buildRoute(memory, length, original, original + 512, original + 256, plan).error ==
              Error::None);
        memcpy(memory + 512, plan.trampoline, plan.trampolineSize);
        auto originalCall = reinterpret_cast<int (*)(int)>(memory);
        auto trampolineCall = reinterpret_cast<int (*)(int)>(memory + 512);
        CHECK(originalCall(argument) == expected && trampolineCall(argument) == expected);
        memcpy(memory, plan.entry, plan.displacedSize);
        CHECK(originalCall(argument) == 99 && trampolineCall(argument) == expected);
        memcpy(memory, plan.original, plan.displacedSize);
        CHECK(originalCall(argument) == expected);
    };
    // A conditional loop targets a displaced instruction boundary.
    const uint8_t loop[] = {0x31, 0xc0, 0xff, 0xc0, 0x83, 0xf8, 3,   0x75,
                            0xf9, 0x90, 0x90, 0x90, 0x90, 0x90, 0xc3};
    run(loop, sizeof(loop), 0, 3);
    uint8_t directCall[80];
    memset(directCall, 0x90, sizeof(directCall));
    // The call sits at offset 9 so its return point is byte 14, at the window's
    // end; a call earlier than that is refused (see branches()).
    directCall[9] = 0xe8;
    uint32_t delta = 64 - 14;
    memcpy(directCall + 10, &delta, 4);
    const uint8_t add[] = {0x83, 0xc0, 5};
    memcpy(directCall + 14, add, sizeof(add));
    directCall[17] = 0xc3;
    const uint8_t callee[] = {0xb8, 11, 0, 0, 0, 0xc3};
    memcpy(directCall + 64, callee, sizeof(callee));
    run(directCall, sizeof(directCall), 0, 16);
    uint8_t load[80];
    memset(load, 0x90, sizeof(load));
    load[0] = 0x8b;
    load[1] = 0x05;
    delta = 64 - 6;
    memcpy(load + 2, &delta, 4);
    load[14] = 0xc3;
    uint32_t value = 23;
    memcpy(load + 64, &value, 4);
    run(load, sizeof(load), 0, 23);
    munmap(memory, 4096);
    puts("x86 execution: pass (internal loop, direct call, RIP load, route/restore)");
}
#endif

// The route pool and the per-request cap (src/Policy.h), and the invariant that
// ties the executable storage src/x86_64/Route.cpp reserves to the pool size.
// The transaction behaviour that spends the pool -- several clients sharing it
// first come first served, and a function one client holds refused to the next
// -- lives in the kernel-only installer and is proven by booting the core probe.
static void capacity() {
    CHECK(graft::policy::RouteCapacity == 32);
    CHECK(graft::policy::MaxRequestRoutes == 16);
    // Graft's own two catalogue routes are part of the shared pool, and a single
    // request can never ask for more than the pool can hold beyond them.
    CHECK(graft::policy::MaxRequestRoutes <= graft::policy::RouteCapacity - 2);
    // src/x86_64/Route.cpp static-asserts this same product for its __TEXT fill.
    CHECK(graft::policy::RouteCapacity * MaxTrampoline == 8192);
    puts("x86 capacity: pass (32-slot shared pool, 16-route request cap, 8 KiB storage)");
}

int main() {
    decoding();
    plainAndRip();
    branches();
    failures();
    boundedInputs();
    capacity();
#if defined(__x86_64__)
    execution();
#endif
}
