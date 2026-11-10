// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Original x86-64 entry-route planning. This module only reads bytes and
// produces a plan; the kernel layer owns mapping, serialization and writes.
#pragma once
#include "Fixup.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

namespace graft {
namespace x86 {

constexpr size_t JumpSize = 14;
constexpr size_t MaxDisplaced = 32;
constexpr size_t MaxTrampoline = 256;

enum class Error {
    None,
    Input,
    Truncated,
    InstructionLength,
    Unsupported,
    Prefix,
    Terminal,
    CallReturn,
    ControlFlowProtection,
    Address,
    Capacity,
    InternalTarget,
    InternalReference,
    RipRange,
    Overlap
};

inline const char *describe(Error error) {
    switch (error) {
    case Error::None: return "ok";
    case Error::Input: return "route input";
    case Error::Truncated: return "truncated instruction";
    case Error::InstructionLength: return "instruction exceeds 15 bytes";
    case Error::Unsupported: return "unsupported instruction";
    case Error::Prefix: return "unsupported instruction prefix";
    case Error::Terminal: return "function ends before route fits";
    case Error::CallReturn: return "displaced call returns inside the route window";
    case Error::ControlFlowProtection: return "entry requires ENDBR preservation";
    case Error::Address: return "route address overflow";
    case Error::Capacity: return "route capacity";
    case Error::InternalTarget: return "branch targets displaced instruction interior";
    case Error::InternalReference: return "memory operand references displaced bytes";
    case Error::RipRange: return "RIP-relative operand out of range";
    case Error::Overlap: return "route address ranges overlap";
    }
    return "unknown route error";
}

struct Failure {
    Error error = Error::None;
    size_t offset = 0;
};

// kind says how an instruction is relocated into the trampoline: a relative
// call (E8) is rebuilt as an absolute one, while an indirect call (FF /2) is
// copied as it is, or with its RIP-relative operand adjusted. call marks both,
// since either leaves a return address just past itself.
enum class Kind { Plain, RipRelative, Call, Jump, Conditional };
struct Instruction {
    Kind kind = Kind::Plain;
    uint8_t size = 0;
    uint8_t displacement = 0;
    uint8_t displacementSize = 0;
    uint8_t condition = 0;
    bool terminal = false;
    bool call = false;
};

namespace detail {

struct Reader {
    const uint8_t *code;
    size_t available;
    size_t at = 0;
    Error error = Error::None;

    bool take(size_t count) {
        if (error != Error::None)
            return false;
        if (count > 15 - at) {
            error = Error::InstructionLength;
            return false;
        }
        if (at > available || count > available - at) {
            error = Error::Truncated;
            return false;
        }
        at += count;
        return true;
    }
    uint8_t byte() { return take(1) ? code[at - 1] : 0; }
};

inline int64_t displacement(const uint8_t *at, uint8_t size) {
    if (size == 1)
        return *at < 0x80 ? *at : static_cast<int64_t>(*at) - 0x100;
    uint32_t value;
    memcpy(&value, at, sizeof(value));
    return value <= INT32_MAX ? value : static_cast<int64_t>(value) - (int64_t{1} << 32);
}

inline void absoluteJump(uint8_t *at, uint64_t destination) {
    const uint8_t instruction[] = {0xff, 0x25, 0, 0, 0, 0};
    memcpy(at, instruction, sizeof(instruction));
    memcpy(at + sizeof(instruction), &destination, sizeof(destination));
}

inline void absoluteCall(uint8_t *at, uint64_t destination) {
    // The return address points to a short jump over the inline pointer.
    const uint8_t instruction[] = {0xff, 0x15, 2, 0, 0, 0, 0xeb, 8};
    memcpy(at, instruction, sizeof(instruction));
    memcpy(at + sizeof(instruction), &destination, sizeof(destination));
}

inline size_t relocatedSize(const Instruction &instruction) {
    switch (instruction.kind) {
    case Kind::Call:
    case Kind::Conditional: return 16;
    case Kind::Jump: return JumpSize;
    case Kind::Plain:
    case Kind::RipRelative: return instruction.size;
    }
    return 0;
}

} // namespace detail

// Deliberately bounded to ordinary integer instructions used at function
// entries. Unknown encodings, address-size overrides, lock/rep prefixes and
// vector encodings are refused; they are never guessed to have a length.
// F3 0F 1E FA (ENDBR64) and F3 90 (PAUSE) are the explicit REP exceptions.
// On failure, the output instruction is untouched.
inline Error decode(const uint8_t *code, size_t available, Instruction &out) {
    if (!code)
        return Error::Input;
    detail::Reader r{code, available};
    Instruction instruction;
    bool operand16 = false, rex = false, rexW = false, segment = false;
    bool repeat = false;
    uint8_t opcode = r.byte();
    while (r.error == Error::None) {
        if (opcode == 0x66 || opcode == 0x64 || opcode == 0x65) {
            if (rex || repeat || (opcode == 0x66 ? operand16 : segment))
                return Error::Prefix;
            if (opcode == 0x66)
                operand16 = true;
            else
                segment = true;
        } else if (opcode >= 0x40 && opcode <= 0x4f) {
            if (rex || repeat)
                return Error::Prefix;
            rex = true;
            rexW = (opcode & 8) != 0;
        } else if (opcode == 0xf3) {
            if (r.at != 1)
                return Error::Prefix;
            repeat = true;
        } else if (opcode == 0xf0 || opcode == 0xf2 || opcode == 0x67 || opcode == 0x26 ||
                   opcode == 0x2e || opcode == 0x36 || opcode == 0x3e) {
            return Error::Prefix;
        } else {
            break;
        }
        opcode = r.byte();
    }
    if (r.error != Error::None)
        return r.error;
    const bool prefixed = operand16 || rex || segment || repeat;
    const size_t operandImmediate = operand16 && !rexW ? 2 : 4;
    size_t immediate = 0;
    bool modrm = false, nop = false, lea = false;
    enum class Group { None, Pop, Mov, Unary8, Unary, Inc, Control, Bit, Nop } group = Group::None;

    if (opcode == 0x0f) {
        uint8_t second = r.byte();
        if (r.error != Error::None)
            return r.error;
        if (repeat) {
            if (second != 0x1e || r.byte() != 0xfa)
                return r.error == Error::None ? Error::Unsupported : r.error;
        } else if (second >= 0x80 && second <= 0x8f) {
            if (prefixed)
                return Error::Prefix;
            instruction.kind = Kind::Conditional;
            instruction.condition = second & 15;
            instruction.displacement = static_cast<uint8_t>(r.at);
            instruction.displacementSize = 4;
            immediate = 4;
        } else if (second == 0x1f) {
            modrm = nop = true;
            group = Group::Nop;
        } else if ((second >= 0x40 && second <= 0x4f) || (second >= 0x90 && second <= 0x9f) ||
                   second == 0xaf || second == 0xb6 || second == 0xb7 || second == 0xbe ||
                   second == 0xbf || second == 0xbc || second == 0xbd || second == 0xb0 ||
                   second == 0xb1 || second == 0xc0 || second == 0xc1 || second == 0xa3 ||
                   second == 0xab || second == 0xb3 || second == 0xbb) {
            modrm = true;
        } else if (second == 0xba) {
            modrm = true;
            immediate = 1;
            group = Group::Bit;
        } else if ((second >= 0xc8 && second <= 0xcf) || second == 0x31 || second == 0x32 ||
                   second == 0x30 || second == 0xa2) {
            // Register-only instructions, no trailing fields.
        } else {
            return Error::Unsupported;
        }
    } else if (repeat) {
        if (opcode != 0x90)
            return Error::Prefix;
    } else if (opcode >= 0x70 && opcode <= 0x7f) {
        if (prefixed)
            return Error::Prefix;
        instruction.kind = Kind::Conditional;
        instruction.condition = opcode & 15;
        instruction.displacement = static_cast<uint8_t>(r.at);
        instruction.displacementSize = 1;
        immediate = 1;
    } else if (opcode == 0xe8 || opcode == 0xe9 || opcode == 0xeb) {
        if (prefixed)
            return Error::Prefix;
        instruction.kind = opcode == 0xe8 ? Kind::Call : Kind::Jump;
        instruction.call = opcode == 0xe8;
        instruction.terminal = opcode != 0xe8;
        instruction.displacement = static_cast<uint8_t>(r.at);
        instruction.displacementSize = opcode == 0xeb ? 1 : 4;
        immediate = instruction.displacementSize;
    } else if ((opcode >= 0x50 && opcode <= 0x5f) || (opcode >= 0x90 && opcode <= 0x99) ||
               opcode == 0x9c || opcode == 0x9d || opcode == 0xc9 ||
               (opcode >= 0xf8 && opcode <= 0xfd)) {
        // No trailing fields.
    } else if (opcode == 0xc3 || opcode == 0xc2) {
        instruction.terminal = true;
        immediate = opcode == 0xc2 ? 2 : 0;
    } else if (opcode == 0x68 || opcode == 0x6a) {
        immediate = opcode == 0x6a ? 1 : operandImmediate;
    } else if (opcode >= 0xb0 && opcode <= 0xb7) {
        immediate = 1;
    } else if (opcode >= 0xb8 && opcode <= 0xbf) {
        immediate = rexW ? 8 : operandImmediate;
    } else if (opcode >= 0xa0 && opcode <= 0xa3) {
        immediate = 8; // Address-size overrides are refused above.
    } else if (opcode == 0xa8 || opcode == 0xa9) {
        immediate = opcode == 0xa8 ? 1 : operandImmediate;
    } else if (opcode <= 0x3d && (opcode & 7) <= 5) {
        if ((opcode & 7) <= 3)
            modrm = true;
        else
            immediate = (opcode & 7) == 4 ? 1 : operandImmediate;
    } else if (opcode == 0x63 || (opcode >= 0x84 && opcode <= 0x8b) || opcode == 0x8d ||
               opcode == 0x69 || opcode == 0x6b || opcode == 0x80 || opcode == 0x81 ||
               opcode == 0x83 || opcode == 0xc0 || opcode == 0xc1 ||
               (opcode >= 0xd0 && opcode <= 0xd3)) {
        modrm = true;
        lea = opcode == 0x8d;
        if (opcode == 0x69 || opcode == 0x81)
            immediate = operandImmediate;
        if (opcode == 0x6b || opcode == 0x80 || opcode == 0x83 || opcode == 0xc0 || opcode == 0xc1)
            immediate = 1;
    } else if (opcode == 0x8f || opcode == 0xc6 || opcode == 0xc7 || opcode == 0xf6 ||
               opcode == 0xf7 || opcode == 0xfe || opcode == 0xff) {
        modrm = true;
        if (opcode == 0x8f)
            group = Group::Pop;
        if (opcode == 0xc6 || opcode == 0xc7) {
            group = Group::Mov;
            immediate = opcode == 0xc6 ? 1 : operandImmediate;
        }
        if (opcode == 0xf6)
            group = Group::Unary8;
        if (opcode == 0xf7)
            group = Group::Unary;
        if (opcode == 0xfe)
            group = Group::Inc;
        if (opcode == 0xff)
            group = Group::Control;
    } else {
        return Error::Unsupported;
    }

    if (modrm) {
        uint8_t byte = r.byte();
        if (r.error != Error::None)
            return r.error;
        uint8_t mode = byte >> 6, reg = (byte >> 3) & 7, rm = byte & 7;
        if (lea && mode == 3)
            return Error::Unsupported;
        switch (group) {
        case Group::Pop:
        case Group::Mov:
        case Group::Nop:
            if (reg != 0)
                return Error::Unsupported;
            break;
        case Group::Unary8:
        case Group::Unary:
            if (reg == 1)
                return Error::Unsupported;
            if (reg == 0)
                immediate = group == Group::Unary8 ? 1 : operandImmediate;
            break;
        case Group::Inc:
            if (reg > 1)
                return Error::Unsupported;
            break;
        case Group::Control:
            // FF /3 and /5 are far calls and jumps through memory, and /7 is
            // undefined; /2 is an indirect near call, /4 an indirect near jump.
            if (reg == 3 || reg == 5 || reg == 7)
                return Error::Unsupported;
            instruction.call = reg == 2;
            instruction.terminal = reg == 4;
            break;
        case Group::Bit:
            if (reg < 4)
                return Error::Unsupported;
            break;
        case Group::None: break;
        }
        bool absoluteSib = false;
        if (mode != 3 && rm == 4) {
            uint8_t sib = r.byte();
            absoluteSib = mode == 0 && (sib & 7) == 5;
        }
        if (mode == 0 && rm == 5) {
            if (!nop) {
                instruction.kind = Kind::RipRelative;
                instruction.displacement = static_cast<uint8_t>(r.at);
                instruction.displacementSize = 4;
            }
            r.take(4);
        } else if (absoluteSib || mode == 2) {
            r.take(4);
        } else if (mode == 1) {
            r.take(1);
        }
    }
    r.take(immediate);
    if (r.error != Error::None)
        return r.error;
    instruction.size = static_cast<uint8_t>(r.at);
    out = instruction;
    return Error::None;
}

struct RoutePlan {
    uint8_t original[MaxDisplaced]{};
    uint8_t entry[MaxDisplaced]{};
    uint8_t trampoline[MaxTrampoline]{};
    size_t displacedSize = 0;
    size_t trampolineSize = 0;
};

// The caller supplies a readable source range bounded by its owning
// function/image. Destination and trampoline must remain mapped while the
// route is installed. Output remains untouched on every failure.
inline Failure buildRoute(const uint8_t *code, size_t codeSize, uint64_t originalAddress,
                          uint64_t trampolineAddress, uint64_t destination, RoutePlan &out) {
    if (!code || !originalAddress || !trampolineAddress || !destination)
        return {Error::Input};
    struct Moved {
        Instruction instruction;
        size_t source, relocated;
    } moved[JumpSize]{};
    RoutePlan plan;
    size_t count = 0;
    while (plan.displacedSize < JumpSize) {
        if (count == JumpSize || plan.displacedSize >= codeSize)
            return {Error::Truncated, plan.displacedSize};
        Instruction instruction;
        auto error = decode(code + plan.displacedSize, codeSize - plan.displacedSize, instruction);
        if (error != Error::None)
            return {error, plan.displacedSize};
        // Replacing an indirect-branch landing pad would fault under IBT.
        // Its decoding is supported, but routing such entries requires an
        // entry-preserving policy and matching wrapper instrumentation.
        const uint8_t endbr[] = {0xf3, 0x0f, 0x1e, 0xfa};
        if (plan.displacedSize == 0 && instruction.size == sizeof(endbr) &&
            !memcmp(code, endbr, sizeof(endbr)))
            return {Error::ControlFlowProtection};
        if (instruction.size > MaxDisplaced - plan.displacedSize)
            return {Error::Capacity, plan.displacedSize};
        size_t length = detail::relocatedSize(instruction);
        if (length > MaxTrampoline - JumpSize - plan.trampolineSize)
            return {Error::Capacity, plan.displacedSize};
        // A displaced call, relative or indirect, whose return point falls inside
        // the jump would send a thread still in the callee back into the
        // overwritten bytes. Refuse it; its return address is the byte after the
        // call, at this offset.
        if (instruction.call && plan.displacedSize + instruction.size < JumpSize)
            return {Error::CallReturn, plan.displacedSize};
        moved[count++] = {instruction, plan.displacedSize, plan.trampolineSize};
        plan.displacedSize += instruction.size;
        plan.trampolineSize += length;
        if (instruction.terminal && plan.displacedSize < JumpSize)
            return {Error::Terminal, moved[count - 1].source};
    }
    uint64_t originalEnd, trampolineEnd;
    if (!fixup::add(originalAddress, static_cast<int64_t>(plan.displacedSize), originalEnd) ||
        !fixup::add(trampolineAddress, static_cast<int64_t>(plan.trampolineSize + JumpSize),
                    trampolineEnd))
        return {Error::Address};
    if ((trampolineAddress < originalEnd && originalAddress < trampolineEnd) ||
        (destination >= originalAddress && destination < originalEnd) ||
        (destination >= trampolineAddress && destination < trampolineEnd))
        return {Error::Overlap};
    memcpy(plan.original, code, plan.displacedSize);
    memset(plan.entry, 0x90, plan.displacedSize);
    detail::absoluteJump(plan.entry, destination);

    for (size_t i = 0; i < count; ++i) {
        const auto &move = moved[i];
        const auto &instruction = move.instruction;
        auto source = code + move.source;
        auto relocated = plan.trampoline + move.relocated;
        if (instruction.kind == Kind::Plain) {
            memcpy(relocated, source, instruction.size);
            continue;
        }
        uint64_t next, target;
        if (!fixup::add(originalAddress, static_cast<int64_t>(move.source + instruction.size),
                        next) ||
            !fixup::add(next,
                        detail::displacement(source + instruction.displacement,
                                             instruction.displacementSize),
                        target))
            return {Error::Address, move.source};
        if (instruction.kind == Kind::RipRelative) {
            if (target >= originalAddress && target < originalEnd)
                return {Error::InternalReference, move.source};
            uint64_t relocatedNext;
            if (!fixup::add(trampolineAddress,
                            static_cast<int64_t>(move.relocated + instruction.size), relocatedNext))
                return {Error::Address, move.source};
            __int128 displacement = static_cast<__int128>(target) - relocatedNext;
            if (displacement < INT32_MIN || displacement > INT32_MAX)
                return {Error::RipRange, move.source};
            memcpy(relocated, source, instruction.size);
            uint32_t value = static_cast<uint32_t>(displacement);
            memcpy(relocated + instruction.displacement, &value, sizeof(value));
            continue;
        }
        if (target >= originalAddress && target < originalEnd) {
            bool found = false;
            for (size_t j = 0; j < count; ++j) {
                if (target - originalAddress == moved[j].source) {
                    if (!fixup::add(trampolineAddress, static_cast<int64_t>(moved[j].relocated),
                                    target))
                        return {Error::Address, move.source};
                    found = true;
                    break;
                }
            }
            if (!found)
                return {Error::InternalTarget, move.source};
        }
        switch (instruction.kind) {
        case Kind::Call: detail::absoluteCall(relocated, target); break;
        case Kind::Jump: detail::absoluteJump(relocated, target); break;
        case Kind::Conditional:
            relocated[0] = 0x70 | (instruction.condition ^ 1);
            relocated[1] = JumpSize;
            detail::absoluteJump(relocated + 2, target);
            break;
        case Kind::Plain:
        case Kind::RipRelative: break;
        }
    }
    detail::absoluteJump(plan.trampoline + plan.trampolineSize, originalEnd);
    plan.trampolineSize += JumpSize;
    out = plan;
    return {};
}

} // namespace x86
} // namespace graft
