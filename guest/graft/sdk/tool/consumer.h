// SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
// Build a consumer kext's manifest and stage the kext for Graft to bind.
//
// The declaration is the consumer's graft.json (readDeclaration). graft-kext's
// build runs these steps in order, and its probe, reserve and stage commands run
// them one at a time:
//
// probeLayouts compiles a subclass of each declared class with the consumer's
// compiler flags, and measures each class's size and the method in each of its
// vtable slots.
//
// reservationSize reads the consumer linked without a manifest and gives the
// size of a zero-filled reservation as large as the manifest and its import
// placeholders. The consumer is then linked again with the reservation as
// __DATA,__graft.
//
// stageBundle copies the bundle around that second link. It writes the manifest
// into __DATA,__graft and leaves the consumer dormant: each import from a
// declared library becomes a local placeholder in the section, the declared
// libraries leave the Info.plist, and the personalities stay but gain an
// IOPropertyMatch gate Graft opens when it binds the consumer.
#pragma once
#include "base.h"
#include "macho.h"
#include "plist.h"

#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace graft::tool {

// graft.json, as docs and sdk/README.md describe it. Indices are as written,
// and readDeclaration checks each against its list.
struct LibraryDeclaration {
    std::string id;
    std::optional<std::string> anchor;
    std::vector<std::string> classes;
};
struct ClassDeclaration {
    std::string name, header;
    int64_t library = 0;
    std::optional<std::vector<std::string>> required;
};
struct ConsumerDeclaration {
    std::string name, header;
    int64_t parent = 0;
};
struct OverrideDeclaration {
    std::string method;
    bool required = false;
    std::vector<std::string> selectors;
};
struct SynthesizedDeclaration {
    std::string name, header;
    int64_t parent = 0;
    std::optional<std::string> sibling;
    // In the declaration's order.
    std::vector<OverrideDeclaration> overrides;
};
struct ImportPolicy {
    int64_t library = 0;
    std::optional<std::vector<std::string>> candidates;
    // A symbol the consumer defines, or empty for none.
    std::string fallback;
};
struct Declaration {
    std::vector<LibraryDeclaration> libraries;
    std::vector<ClassDeclaration> classes;
    std::vector<ConsumerDeclaration> consumers;
    std::vector<SynthesizedDeclaration> synthesized;
    std::map<std::string, ImportPolicy> imports;
};

// Read graft.json, throwing Error for a declaration Graft could not bind by.
Declaration readDeclaration(const std::string &path);

// A class as the consumer's compiler lays it out: its size and the method in
// each of its vtable slots.
struct ClassLayout {
    std::string name;
    uint32_t size = 0;
    std::vector<std::string> slots;
};
// The probe's layouts, in the declaration's order.
struct Layouts {
    std::vector<ClassLayout> classes, consumers, synthesized;
};

Layouts probeLayouts(const Declaration &declaration, const std::vector<std::string> &flags);
// layouts.json's text.
std::string layoutsText(const Layouts &layouts);
Layouts readLayouts(const std::string &path);

// What the manifest records about a linked consumer.
struct Import {
    int64_t library = 0;
    std::vector<std::string> candidates;
    uint64_t fallback = 0;
    bool weak = false;
};
// A reference to an import: the pointer or branch displacement at an image
// offset. auth is an arm64e signed pointer's context; opcode is the x86-64
// branch's opcode, or OpcodePointer.
struct Site {
    uint64_t offset = 0;
    std::string symbol;
    uint32_t width = 0;
    uint8_t opcode = 0;
    int64_t addend = 0;
    std::optional<AuthContext> auth;
};
// An inherited slot Graft fills and re-signs on arm64e, with the slot's own
// signing context.
struct FillRecord {
    uint32_t slot = 0;
    uint8_t opcode = 0;
    uint16_t diversity = 0;
};
// A consumer class's vtable: its address point's image offset, the index of its
// parent class, its slot count, and the parent slots the consumer overrides.
struct VtableRecord {
    uint64_t offset = 0;
    uint32_t parent = 0;
    uint32_t count = 0;
    std::vector<uint32_t> overrides;
    std::vector<FillRecord> fills;
};
struct SynthOverrideRecord {
    std::vector<std::string> selectors;
    uint64_t impl = 0;
    uint16_t diversity = 0;
    uint8_t opcode = 0;
    bool required = false;
};
struct SynthClassRecord {
    std::string name, parentClass, siblingClass;
    uint32_t library = 0;
    uint64_t vtableOffset = 0;
    uint32_t vtableSlotCap = 0, parentRegion = 0;
    std::vector<SynthOverrideRecord> overrides;
};
struct SegmentRecord {
    std::string name;
    uint64_t origin = 0, size = 0;
};
struct Census {
    std::map<std::string, Import> imports;
    std::vector<Site> sites;
    std::vector<VtableRecord> vtables;
    std::vector<SegmentRecord> segments;
    std::vector<SynthClassRecord> synth;
};

// Read the imports, sites and vtables a linked consumer's manifest records,
// throwing Error where Graft could not bind the consumer.
Census census(const Image &image, const Declaration &declaration, const Layouts &layouts);
// The manifest's bytes. placeholders maps each import to its placeholder's
// image offset; without them every site's placeholder is 0, which gives the
// manifest its final size.
Bytes packManifest(const Declaration &declaration, const Layouts &layouts, const Census &found,
                   const std::string &gate, const std::map<std::string, uint64_t> &placeholders,
                   uint32_t version);
// The parent slots whose methods Graft must find unchanged, in order.
std::vector<uint32_t> requiredSlots(const ClassDeclaration &declaration, const ClassLayout &layout,
                                    const std::vector<const VtableRecord *> &vtables);

// The activation property key gating the consumer's personalities.
std::string gateKey(const Plist &info);
// The __DATA,__graft size the consumer linked as image needs.
uint64_t reservationSize(const Image &image, const Declaration &declaration, const Layouts &layouts,
                         const std::string &gate);
// image with its manifest written and its imports turned into placeholders.
Image stageImage(const Image &image, const Declaration &declaration, const Layouts &layouts,
                 const std::string &gate);
// Copy the bundle kext to out with its binary staged and its Info.plist gated.
void stageBundle(const std::string &kext, const Declaration &declaration, const Layouts &layouts,
                 const std::string &out);

// The Mach-O symbol of class name's vtable.
std::string vtableName(const std::string &name);
// The first member-name component of a nested-name mangled symbol, which names
// the method a vtable slot holds, or nothing for any other form.
std::optional<std::string> memberName(const std::string &symbol);

} // namespace graft::tool
