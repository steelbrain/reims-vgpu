#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
"""Build a consumer kext's Graft manifest and stage the kext for Graft to bind.

    consumer.py probe DECLARATION LAYOUTS [-- COMPILER_FLAGS...]
    consumer.py reserve BINARY DECLARATION LAYOUTS INFO_PLIST RESERVATION
    consumer.py stage KEXT DECLARATION LAYOUTS OUTPUT

DECLARATION is the consumer's graft.json. consumer.mk runs the three steps in order:

probe compiles a subclass of each declared class with the consumer's compiler flags and
writes LAYOUTS: each class's size and the method in each of its vtable slots.

reserve reads BINARY, the consumer linked without a manifest, and writes RESERVATION, a
zero-filled file as large as the manifest and its import placeholders. The consumer is
then linked again with -sectcreate __DATA __graft RESERVATION.

stage copies KEXT, the bundle around that second link, to OUTPUT. It writes the manifest
into __DATA,__graft and leaves the consumer dormant: each System-KC import becomes a
local placeholder in the section, the System-KC dependencies leave the Info.plist, and the
personalities stay but gain an IOPropertyMatch gate Graft opens when it binds the consumer.

It uses Python's standard library and Xcode's clang, and fetches nothing.
"""

from __future__ import annotations

import argparse
import json
import plistlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

# Python's -I leaves the script's own directory off the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graft_macho import (
    ARM64_RELOC_AUTHENTICATED_POINTER,
    MH_KEXT_BUNDLE,
    N_WEAK_REF,
    X86_64_RELOC_BRANCH,
    X86_64_RELOC_UNSIGNED,
    AuthContext,
    Error,
    Image,
    Relocation,
)

# graft.json and the probe's layouts, as JSON reads them.
Declaration = dict[str, Any]
Layouts = dict[str, Any]

# The manifest format: src/Manifest.h, which docs/manifest.md describes.
MAGIC = 0x54465247  # "GRFT"
# Version 2 is x86-64; version 3 is arm64e, differing only in how a site's two
# trailing bytes are read (notes/arm64-boot.md, "The v3 authenticated-site form").
VERSION_X86_64 = 2
VERSION_ARM64E = 3
NO_FALLBACK = (1 << 64) - 1
IMPORT_WEAK = 1
HEADER = struct.Struct("<28I")
HEADER_SIZE = HEADER.size
LIBRARY = struct.Struct("<2I")  # identifier, anchor
IMPORT = struct.Struct("<4IQ")  # library, first candidate, candidate count, flags, fallback
CANDIDATE = struct.Struct("<I")  # name
# offset, placeholder, addend, import, width; then two arch-specific bytes: on
# x86-64 opcode and a zero reserved; on arm64e an auth byte (bit0 signed, bits1-2
# key, bit3 addrDiv) and the 16-bit diversity.
SITE = struct.Struct("<QQqIBBH")
SITE_AUTH_SIGNED = 1
CLASS = struct.Struct("<7I")  # name, vtable, library, size, parent slots, first slot, slot count
SLOT = struct.Struct("<II")  # index, method
# offset, class, slot count, first override, override count, first fill, fill count
VTABLE = struct.Struct("<Q6I")
OVERRIDE = struct.Struct("<I")  # parent slot index
# The consumer's segments and the inherited slots Graft re-signs: arm64e only
# (src/Manifest.h, Segment and Fill). A version-2 manifest carries neither.
SEGMENT = struct.Struct("<QQII")  # origin, size, name, reserved
FILL = struct.Struct("<IHBB")  # slot, diversity, opcode, reserved

# A site's opcode: 0 for an eight-byte pointer, or the opcode before a branch's rel32.
POINTER = 0
CALL_REL32, JMP_REL32 = 0xE8, 0xE9
BRANCH_OPCODES = (CALL_REL32, JMP_REL32)

# A vtable's slots start after its offset-to-top and RTTI words; with -fapple-kext the
# compiler ends it with one zero word.
POINTER_SIZE = 8
VTABLE_ADDRESS_POINT = 2 * POINTER_SIZE

# Each import gets its own 16-byte-aligned placeholder, 32 bytes apart, so a vtable
# import, whose sites add the 16-byte address point, still lands inside its own.
PLACEHOLDER_ALIGNMENT = 16
PLACEHOLDER_SIZE = 32

PROBE_CLASS = "GraftLayoutProbe"

# Our paravirtual GPU's IOPCIPrimaryMatch (vendor 0x1234, device 0x5247). A bound
# consumer's personalities must name exactly this device; stage checks it and records it in
# the manifest so Graft finds the nub to open the gate on, with no device constant of its
# own (src/Policy.h, notes/activation-gate.md).
DEVICE_MATCH = "0x52471234"

# The suffix forming a consumer's activation property key from its bundle identifier.
# Graft sets the key on the device nub once it binds; the staged personalities carry it as
# an IOPropertyMatch so none matches, and none is even constructed, until then.
GATE_SUFFIX = ".graft-bound"


class Table(NamedTuple):
    """A manifest table's section offset and row count (bytes, for the string table)."""

    offset: int
    count: int


class Header(NamedTuple):
    """The manifest's header (src/Manifest.h, Header)."""

    magic: int
    version: int
    size: int
    header_size: int
    strings: Table
    libraries: Table
    imports: Table
    candidates: Table
    sites: Table
    classes: Table
    slots: Table
    vtables: Table
    overrides: Table
    segments: Table
    fills: Table
    gate: int
    device: int

    @classmethod
    def unpack_from(cls, data: bytes | bytearray, offset: int = 0) -> Header:
        words = HEADER.unpack_from(data, offset)
        tables = [Table(*words[i : i + 2]) for i in range(4, 26, 2)]
        return cls(*words[:4], *tables, *words[26:])

    def pack(self) -> bytes:
        tables = [word for table in self[4:15] for word in table]
        return HEADER.pack(*self[:4], *tables, *self[15:])


@dataclass
class Import:
    """A System-KC import Graft binds for the consumer.

    library indexes the declaration's providers. candidates are the names to try, in order.
    fallback is an image offset in the consumer, or NO_FALLBACK when the import is required.
    weak is set when the consumer references the import with weak_import.
    """

    library: int
    candidates: list[str]
    fallback: int
    weak: bool


@dataclass
class Site:
    """A reference to an import: the pointer or branch displacement at an image offset.

    auth is the pointer-authentication context of an arm64e signed pointer, else None (every
    x86-64 site, and an arm64e plain pointer). opcode is the x86-64 branch opcode or POINTER.
    """

    offset: int
    symbol: str
    width: int
    opcode: int
    addend: int
    auth: AuthContext | None = None


@dataclass
class Vtable:
    """A consumer class's vtable, whose parent slots Graft fills from the live parent.

    offset is its address point's image offset, parent indexes the declared classes,
    count is its slot count, and overrides are the parent slots the consumer implements.
    fills are the inherited (non-override) parent slots Graft re-signs, each a (slot,
    opcode, diversity) carrying the slot's own signing context; empty on x86-64.
    """

    offset: int
    parent: int
    count: int
    overrides: list[int]
    fills: list[tuple[int, int, int]] = field(default_factory=list)


class Census(NamedTuple):
    """What the manifest records about a linked consumer."""

    imports: dict[str, Import]
    sites: list[Site]
    vtables: list[Vtable]
    segments: list[tuple[str, int, int]]


def identifier(name: str) -> str:
    """name, if it is a C++ class name the probe can derive from."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise Error("only top-level C++ class names are supported")
    return name


def vtable_name(name: str) -> str:
    """The Mach-O symbol of class name's vtable."""
    return f"__ZTV{len(name)}{name}"


def owner_class(name: str) -> str | None:
    """The top-level class whose member, vtable or typeinfo the Mach-O symbol name is."""
    match = re.match(r"__(?:ZNK?|ZT[VIS])(\d+)(.*)", name)
    return match[2][: int(match[1])] if match else None


def read_declaration(path: Path) -> Declaration:
    """Read graft.json, raising Error for a declaration Graft could not bind by."""
    config = json.loads(Path(path).read_text())
    libraries, classes = config["libraries"], config.get("classes", [])
    if not libraries or any(not library.get("id") for library in libraries):
        raise Error("declare at least one named provider")
    if len({library["id"] for library in libraries}) != len(libraries):
        raise Error("duplicate provider identifier")
    for cls in classes:
        identifier(cls["name"])
        if not cls.get("header") or not 0 <= cls["library"] < len(libraries):
            raise Error("class has no header or provider")
    # Graft verifies each provider through a class's registered metaclass: a declared
    # class from it, or its anchor where the consumer takes only imports from it.
    for i, library in enumerate(libraries):
        if "anchor" in library:
            identifier(library["anchor"])
        elif not any(cls["library"] == i for cls in classes):
            raise Error(
                f"provider {library['id']} has no declared class and no anchor to verify it by"
            )
    if len({cls["name"] for cls in classes}) != len(classes):
        raise Error("duplicate parent class")
    for consumer in config.get("consumers", []):
        identifier(consumer["name"])
        if not consumer.get("header") or not 0 <= consumer["parent"] < len(classes):
            raise Error("consumer class has no header or parent")
    return config


def probe_layouts(config: Declaration, flags: list[str]) -> Layouts:
    """Each declared class's size and vtable slots, as the consumer's compiler builds them.

    The probe derives a subclass with no members from each class, with an out-of-line
    destructor, so the compiler emits the subclass's vtable even for an abstract parent.
    That vtable has exactly the parent's slots, each relocated against the method in it,
    except for the destructors, which name the subclass and map back to the parent.
    """
    parents = config.get("classes", [])
    rows = parents + config.get("consumers", [])
    if not rows:
        return {"classes": [], "consumers": []}
    image, dump = compile_probe(probe_source(rows), flags)
    entries = {
        int(i): int(count)
        for i, count in re.findall(rf"Vtable for '{PROBE_CLASS}(\d+)' \((\d+) entries\)", dump)
    }
    sizes = image.symbol("_graft_layout_sizes")
    size_values = struct.unpack(f"<{len(rows)}I", image.read(sizes.value, 4 * len(rows)))
    relocations = {r.address: r for r in image.relocations()}
    layouts = [
        {
            "name": row["name"],
            "size": size_values[i],
            "slots": probe_slots(image, i, row["name"], entries.get(i), relocations),
        }
        for i, row in enumerate(rows)
    ]
    return {"classes": layouts[: len(parents)], "consumers": layouts[len(parents) :]}


def probe_source(rows: list[dict[str, Any]]) -> str:
    """C++ that derives probe class i from rows[i] and records each class's sizeof."""
    source = [f'#include "{row["header"]}"' for row in rows]
    for i, row in enumerate(rows):
        probe, name = f"{PROBE_CLASS}{i}", identifier(row["name"])
        source += [
            f"class {probe} : public {name} {{ public:",
            f"~{probe}() override; }};",
            f"{probe}::~{probe}() {{}}",
        ]
    sizes = ",".join(f"sizeof({identifier(row['name'])})" for row in rows)
    source.append(f'extern "C" const unsigned graft_layout_sizes[] = {{{sizes}}};')
    return "\n".join(source) + "\n"


def target_arch(flags: list[str]) -> str:
    """The architecture the consumer builds for, from its flags' -arch pair (x86-64 default).

    The layout probe compiles for this architecture, because a class's inherited vtable is
    NOT the same on every one: the Kernel headers give OSMetaClassBase four more slots on
    arm64 (the arm64-only _RESERVEDOSMetaClassBase* under APPLE_KEXT_VTABLE_PADDING) and turn
    each OSMetaClassDeclareReservedUsedX86 slot into a reserved one, so an arm64e consumer's
    parent slots differ from x86-64's in both count and order (IOAccelerator 272 vs 266).
    """
    for i, flag in enumerate(flags):
        if flag == "-arch" and i + 1 < len(flags):
            return flags[i + 1]
    return "x86_64"


def without_arch(flags: list[str]) -> list[str]:
    """The consumer's flags with any -arch pair dropped.

    compile_probe sets the probe's -arch from target_arch(flags) and then appends these, so
    the consumer's own -arch is dropped here to avoid a second, conflicting one.
    """
    kept, skip = [], False
    for flag in flags:
        if skip:
            skip = False
        elif flag == "-arch":
            skip = True
        else:
            kept.append(flag)
    return kept


def compile_probe(source: str, flags: list[str]) -> tuple[Image, str]:
    """Compile source with kext flags for macOS 11 on the consumer's arch, then its flags.

    Returns the object and the compiler's vtable-layout dump. The source and object live in
    a temporary directory outside the consumer's tree.
    """
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
    ).strip()
    with tempfile.TemporaryDirectory(prefix="graft-layout-", dir="/tmp") as temp:
        source_path, object_path = Path(temp) / "probe.cpp", Path(temp) / "probe.o"
        source_path.write_text(source)
        command = [
            *("xcrun", "clang++", "-arch", target_arch(flags), "-isysroot", sdk),
            *("-mmacosx-version-min=11.0", "-mkernel", "-nostdinc"),
            *("-isystem", f"{sdk}/System/Library/Frameworks/Kernel.framework/Headers"),
            *("-DKERNEL", "-std=c++17", "-fapple-kext", "-fno-rtti", "-fno-exceptions"),
            *("-Xclang", "-fdump-vtable-layouts", *without_arch(flags)),
            *("-c", str(source_path), "-o", str(object_path)),
        ]
        process = subprocess.run(command, capture_output=True, text=True, check=False)
        if process.returncode:
            sys.stderr.write(process.stderr)
            raise Error("consumer layout compilation failed")
        return Image(object_path.read_bytes()), process.stdout


def probe_slots(
    image: Image,
    index: int,
    name: str,
    entries: int | None,
    relocations: dict[int, Relocation],
) -> list[str]:
    """The methods in class name's vtable slots, read from probe class index's vtable.

    entries is the compiler's reported entry count, which includes the offset-to-top and
    RTTI words and the trailing zero word: the slots are the rest.
    """
    probe = f"{PROBE_CLASS}{index}"
    vtable = image.symbol(vtable_name(probe))
    if entries is None or entries < 4:
        raise Error("compiler did not report a single primary vtable")
    end = vtable.value + (entries - 1) * POINTER_SIZE
    if end in relocations or image.read(end, POINTER_SIZE) != bytes(POINTER_SIZE):
        raise Error("compiler vtable lacks its final zero sentinel")
    # An arm64e object relocates each vtable slot as an authenticated pointer; x86-64 as a
    # plain one. Either way the slot names its method through the relocation's symbol.
    pointer = ARM64_RELOC_AUTHENTICATED_POINTER if image.arch == "arm64e" else X86_64_RELOC_UNSIGNED
    slots = []
    for slot in range(entries - 3):
        relocation = relocations.get(vtable.value + VTABLE_ADDRESS_POINT + slot * POINTER_SIZE)
        if (
            relocation is None
            or not relocation.external
            or relocation.width != POINTER_SIZE
            or relocation.pcrel
            or relocation.kind != pointer
        ):
            raise Error("compiler vtable slot has no supported symbolic relocation")
        method = image.symbols[relocation.symbol_index].name
        prefix = f"__ZN{len(probe)}{probe}"
        if method.startswith(prefix):
            destructor = method[len(prefix) :]
            if destructor not in ("D0Ev", "D1Ev"):
                raise Error("layout probe introduced an unexpected method")
            method = f"__ZN{len(name)}{name}{destructor}"
        slots.append(method)
    return slots


def census(image: Image, config: Declaration, layouts: Layouts) -> Census:
    """Read from a linked consumer the imports, sites and vtables its manifest records.

    Raises Error where Graft could not bind the consumer: an import with no reference, a
    reference other than a pointer or a direct call or jump, or a vtable whose overrides
    and appended methods are not the consumer's own code.
    """
    if image.filetype != MH_KEXT_BUNDLE:
        raise Error("consumer must be linked as a kext")
    declared = [[row["name"] for row in config.get(key, [])] for key in ("classes", "consumers")]
    measured = [[row["name"] for row in layouts[key]] for key in ("classes", "consumers")]
    if declared != measured:
        raise Error("layouts describe another declaration")
    imports = select_imports(config, image)
    if image.arch == "arm64e":
        fixups = image.fixups()
        sited = {f.address: f for f in fixups}
        vtables, parent_slots = consumer_vtables_arm64e(image, config, layouts, sited)
        sites = import_sites_arm64e(image, imports, fixups, parent_slots)
        segments = consumer_segments(image)
    else:
        relocations = image.relocations()
        vtables, parent_slots = consumer_vtables(image, config, layouts, relocations)
        sites = import_sites(image, imports, relocations, parent_slots)
        segments = []
    return Census(imports, sites, vtables, segments)


def consumer_segments(image: Image) -> list[tuple[str, int, int]]:
    """Each loadable segment's name, origin (vmaddr - base) and vmsize, sorted by origin.

    Graft relocates a manifest offset through the placed segment of the same name, because
    Ambrosia coalesces the segments by name (notes/arm64-boot.md). __LINKEDIT carries no
    bound offset and the runtime view excludes it, so it is left out.
    """
    segments = [
        (segment.name, segment.address - image.base, segment.vmsize)
        for segment in image.segments
        if segment.name != "__LINKEDIT" and segment.vmsize and segment.address >= image.base
    ]
    return sorted(segments, key=lambda segment: segment[1])


def class_owners(config: Declaration) -> dict[str, int]:
    """The provider index that owns each class: its declared parents and listed classes."""
    owners = {cls["name"]: cls["library"] for cls in config.get("classes", [])}
    for i, library in enumerate(config["libraries"]):
        for name in library.get("classes", []):
            if name in owners and owners[name] != i:
                raise Error("class is claimed by two providers")
            owners[name] = i
    return owners


def imported_symbols(image: Image) -> list[tuple[str, bool]]:
    """Each symbol the consumer imports and whether it is weak, by architecture.

    An x86-64 kext's imports are its undefined symbol-table entries; an arm64e kext is
    NOUNDEFS and records them as chained-fixup binds instead (graft_macho, Image.fixups).
    """
    if image.arch == "arm64e":
        weak: dict[str, bool] = {}
        for fixup in image.fixups():
            if fixup.bind:
                weak[fixup.symbol] = weak.get(fixup.symbol, False) or fixup.weak
        return sorted(weak.items())
    return [(s.name, bool(s.descriptor & N_WEAK_REF)) for s in image.symbols if s.undefined]


def select_imports(config: Declaration, image: Image) -> dict[str, Import]:
    """The consumer's imports that Graft binds, by name.

    An import is selected when graft.json's imports names it, or when it is a member,
    vtable or typeinfo of a class a provider owns. The others are kernel and Boot-KC
    imports, which the boot linker (on arm64, Ambrosia's placement) resolves.
    """
    owners = class_owners(config)
    policies = config.get("imports", {})
    selected = {}
    for name, weak in imported_symbols(image):
        policy = policies.get(name)
        library = policy["library"] if policy else owners.get(owner_class(name))
        if library is None:
            continue
        if not 0 <= library < len(config["libraries"]):
            raise Error("import provider index outside declaration")
        candidates = policy.get("candidates", [name]) if policy else [name]
        if not candidates or any(not isinstance(c, str) or not c or "\0" in c for c in candidates):
            raise Error("import candidates must be nonempty symbol names")
        fallback = policy.get("fallback") if policy else None
        selected[name] = Import(
            library=library,
            candidates=candidates,
            fallback=image.symbol(fallback).value - image.base if fallback else NO_FALLBACK,
            weak=weak,
        )
    if set(policies) - set(selected):
        raise Error("an explicit import policy names no undefined import")
    return selected


def consumer_vtables(
    image: Image, config: Declaration, layouts: Layouts, relocations: list[Relocation]
) -> tuple[list[Vtable], set[int]]:
    """Each consumer class's vtable, and the addresses of all their parent slots.

    A slot is the consumer's own when it holds a local pointer to the consumer's code. Its
    parent slots that are not are left for Graft to fill from the live parent, so none of
    the parent slots is a site.
    """
    by_address = {r.address: r for r in relocations}
    vtables, parent_slots = [], set()
    for consumer, layout in zip(config.get("consumers", []), layouts["consumers"], strict=True):
        symbol = image.symbol(vtable_name(consumer["name"]))
        parent = consumer["parent"]
        inherited = len(layouts["classes"][parent]["slots"])
        count = len(layout["slots"])
        if count < inherited or image.symbol_extent(symbol) < (count + 2) * POINTER_SIZE:
            raise Error("consumer vtable is shorter than its compiler layout")
        overrides = []
        for slot in range(count):
            address = symbol.value + VTABLE_ADDRESS_POINT + slot * POINTER_SIZE
            if holds_local_pointer(image, by_address.get(address)):
                value = struct.unpack("<Q", image.read(address, POINTER_SIZE))[0]
                if not executable(image, value):
                    raise Error("consumer override does not point to its executable code")
                if slot < inherited:
                    overrides.append(slot)
            elif slot >= inherited:
                raise Error("appended consumer virtual method is not local executable code")
            if slot < inherited:
                parent_slots.add(address)
        vtables.append(
            Vtable(
                offset=symbol.value + VTABLE_ADDRESS_POINT - image.base,
                parent=parent,
                count=count,
                overrides=overrides,
            )
        )
    return sorted(vtables, key=lambda vtable: vtable.offset), parent_slots


def holds_local_pointer(image: Image, relocation: Relocation | None) -> bool:
    """Whether a slot's relocation points into the consumer's own image."""
    if relocation is None:
        return False
    if relocation.external:
        return image.symbols[relocation.symbol_index].defined
    return True


def executable(image: Image, address: int) -> bool:
    return any(s.address <= address < s.address + s.size and s.executable for s in image.sections)


def import_sites(
    image: Image, imports: dict[str, Import], relocations: list[Relocation], parent_slots: set[int]
) -> list[Site]:
    """Every reference to a selected import outside the vtables' parent slots."""
    sites, referenced = [], set()
    for relocation in relocations:
        if not relocation.external:
            continue
        name = image.symbols[relocation.symbol_index].name
        if name not in imports:
            continue
        referenced.add(name)
        if relocation.address not in parent_slots:
            sites.append(import_site(image, relocation, name))
    if referenced != set(imports):
        raise Error("selected import has no relocation site")
    return sorted(sites, key=lambda site: site.offset)


def import_site(image: Image, relocation: Relocation, name: str) -> Site:
    """The site of a relocation against import name: a pointer, or a call or jump."""
    if relocation.pcrel:
        if relocation.width != 4 or relocation.kind != X86_64_RELOC_BRANCH:
            raise Error("unsupported System-KC PC-relative relocation")
        opcode = image.read(relocation.address - 1, 1)[0]
        if opcode not in BRANCH_OPCODES:
            raise Error("unsupported System-KC branch instruction")
    else:
        if relocation.width != POINTER_SIZE or relocation.kind != X86_64_RELOC_UNSIGNED:
            raise Error("unsupported System-KC pointer relocation")
        opcode = POINTER
    addend = image.read(relocation.address, relocation.width)
    return Site(
        offset=relocation.address - image.base,
        symbol=name,
        width=relocation.width,
        opcode=opcode,
        addend=int.from_bytes(addend, "little", signed=True),
    )


def consumer_vtables_arm64e(
    image: Image, config: Declaration, layouts: Layouts, sited: dict[int, Any]
) -> tuple[list[Vtable], set[int]]:
    """Each consumer class's vtable, and the addresses of all their parent slots (arm64e).

    A slot is the consumer's own when its chained fixup is a local rebase to the consumer's
    executable code; a parent slot's fixup is instead a bind to the inherited method, which
    Graft fills from the live parent, so none of the parent slots is a site.
    """
    vtables, parent_slots = [], set()
    for consumer, layout in zip(config.get("consumers", []), layouts["consumers"], strict=True):
        symbol = image.symbol(vtable_name(consumer["name"]))
        parent = consumer["parent"]
        inherited = len(layouts["classes"][parent]["slots"])
        count = len(layout["slots"])
        if count < inherited or image.symbol_extent(symbol) < (count + 2) * POINTER_SIZE:
            raise Error("consumer vtable is shorter than its compiler layout")
        overrides, fills = [], []
        for slot in range(count):
            address = symbol.value + VTABLE_ADDRESS_POINT + slot * POINTER_SIZE
            fixup = sited.get(address)
            override = fixup is not None and not fixup.bind  # a local rebase: the consumer's code
            if override:
                if not executable(image, fixup.target):
                    raise Error("consumer override does not point to its executable code")
                if slot < inherited:
                    overrides.append(slot)
            elif slot >= inherited:
                raise Error("appended consumer virtual method is not local executable code")
            if slot < inherited:
                parent_slots.add(address)
                if not override:
                    # An inherited slot Graft re-signs: record its signing context
                    # from the bind to the parent method, so Graft signs the live
                    # method for this slot exactly as the consumer authenticates it.
                    auth = fixup.auth if (fixup is not None and fixup.bind) else None
                    opcode, diversity = auth_trailer(auth)
                    fills.append((slot, opcode, diversity))
        vtables.append(
            Vtable(
                offset=symbol.value + VTABLE_ADDRESS_POINT - image.base,
                parent=parent,
                count=count,
                overrides=overrides,
                fills=fills,
            )
        )
    return sorted(vtables, key=lambda vtable: vtable.offset), parent_slots


def import_sites_arm64e(
    image: Image, imports: dict[str, Import], fixups: list[Any], parent_slots: set[int]
) -> list[Site]:
    """Every chained-fixup bind to a selected import outside the vtables' parent slots."""
    sites, referenced = [], set()
    for fixup in fixups:
        if not fixup.bind or fixup.symbol not in imports:
            continue
        referenced.add(fixup.symbol)
        if fixup.address not in parent_slots:
            sites.append(
                Site(
                    offset=fixup.address - image.base,
                    symbol=fixup.symbol,
                    width=POINTER_SIZE,
                    opcode=POINTER,
                    addend=fixup.addend,
                    auth=fixup.auth,
                )
            )
    if referenced != set(imports):
        raise Error("selected import has no fixup site")
    return sorted(sites, key=lambda site: site.offset)


class StringTable:
    """The manifest's string table: each name stored once, NUL-terminated; 0 is empty."""

    def __init__(self) -> None:
        self.data = bytearray(b"\0")
        self.offsets: dict[str, int] = {}

    def add(self, value: str) -> int:
        """The offset of value, which is stored on its first use."""
        if not value or "\0" in value:
            raise Error("manifest strings must be nonempty and contain no NUL")
        if value not in self.offsets:
            self.offsets[value] = len(self.data)
            self.data.extend(value.encode() + b"\0")
        return self.offsets[value]


def auth_trailer(auth: AuthContext | None) -> tuple[int, int]:
    """A signing context's auth byte (signed, key, addrDiv) and diversity, or (0, 0) for none."""
    if auth is None:
        return 0, 0
    return SITE_AUTH_SIGNED | auth.key << 1 | int(auth.addr_div) << 3, auth.diversity


def site_trailer(site: Site, version: int) -> tuple[int, int]:
    """A site's two arch-specific trailing fields: (opcode, reserved) for v2; for v3, the
    auth byte (signed, key, addrDiv) and the diversity."""
    if version == VERSION_X86_64:
        return site.opcode, 0
    return auth_trailer(site.auth)


def pack_manifest(
    config: Declaration,
    layouts: Layouts,
    found: Census,
    gate: str,
    placeholders: dict[str, int],
    version: int = VERSION_X86_64,
) -> bytes:
    """The manifest's bytes.

    gate is the consumer's activation property key, which Graft sets on the device nub once
    it binds so the deferred match starts the consumer. placeholders maps each import to its
    placeholder's image offset. Without them every site's placeholder is 0, which gives the
    manifest its final size. version is 2 (x86-64) or 3 (arm64e), which sets a site's two
    trailing bytes.
    """
    strings = StringTable()
    gate_offset = strings.add(gate)
    device_offset = strings.add(DEVICE_MATCH)
    libraries = [
        (strings.add(library["id"]), strings.add(library["anchor"]) if "anchor" in library else 0)
        for library in config["libraries"]
    ]
    names = sorted(found.imports)
    imports, candidates = [], []
    for name in names:
        record = found.imports[name]
        imports.append(
            (
                record.library,
                len(candidates),
                len(record.candidates),
                IMPORT_WEAK if record.weak else 0,
                record.fallback,
            )
        )
        candidates += [(strings.add(candidate),) for candidate in record.candidates]
    import_index = {name: i for i, name in enumerate(names)}
    sites = [
        (
            site.offset,
            placeholders.get(site.symbol, 0),
            site.addend,
            import_index[site.symbol],
            site.width,
            *site_trailer(site, version),
        )
        for site in found.sites
    ]
    classes, slots = [], []
    for i, (declaration, layout) in enumerate(
        zip(config.get("classes", []), layouts["classes"], strict=True)
    ):
        required = required_slots(declaration, layout, [v for v in found.vtables if v.parent == i])
        first = len(slots)
        slots += [(slot, strings.add(layout["slots"][slot])) for slot in required]
        classes.append(
            (
                strings.add(declaration["name"]),
                strings.add(vtable_name(declaration["name"])),
                declaration["library"],
                layout["size"],
                len(layout["slots"]),
                first,
                len(required),
            )
        )
    vtables, overrides, fills = [], [], []
    for vtable in found.vtables:
        vtables.append(
            (
                vtable.offset,
                vtable.parent,
                vtable.count,
                len(overrides),
                len(vtable.overrides),
                len(fills),
                len(vtable.fills),
            )
        )
        overrides += [(slot,) for slot in vtable.overrides]
        fills += [(slot, diversity, opcode, 0) for slot, opcode, diversity in vtable.fills]
    segments = [(origin, size, strings.add(name), 0) for name, origin, size in found.segments]
    rows = [
        (LIBRARY, libraries),
        (IMPORT, imports),
        (CANDIDATE, candidates),
        (SITE, sites),
        (CLASS, classes),
        (SLOT, slots),
        (VTABLE, vtables),
        (OVERRIDE, overrides),
        (SEGMENT, segments),
        (FILL, fills),
    ]
    return assemble(strings.data, rows, gate_offset, device_offset, version)


def required_slots(
    declaration: dict[str, Any], layout: dict[str, Any], vtables: list[Vtable]
) -> list[int]:
    """The parent slots whose methods Graft must find unchanged, in order.

    These are the inherited methods the declaration's required list names, or every parent
    slot when it has none, and every slot a consumer class overrides.
    """
    required = declaration.get("required")
    if required is None:
        slots = set(range(len(layout["slots"])))
    else:
        slots = set()
        for name in required:
            found = [slot for slot, method in enumerate(layout["slots"]) if method == name]
            if len(found) != 1:
                raise Error("required inherited method has no unique compiler slot")
            slots.add(found[0])
    for vtable in vtables:
        slots.update(vtable.overrides)
    return sorted(slots)


def assemble(
    strings: bytes | bytearray,
    rows: list[tuple[struct.Struct, list[tuple]]],
    gate: int,
    device: int,
    version: int,
) -> bytes:
    """The header, then the string table, then each table's packed rows."""
    body = bytearray(strings)
    tables = [Table(HEADER_SIZE, len(strings))]
    for layout, table in rows:
        tables.append(Table(HEADER_SIZE + len(body), len(table)))
        body += b"".join(layout.pack(*row) for row in table)
    size = HEADER_SIZE + len(body)
    header = Header(MAGIC, version, size, HEADER_SIZE, *tables, gate, device)
    return header.pack() + body


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def manifest_version(image: Image) -> int:
    return VERSION_ARM64E if image.arch == "arm64e" else VERSION_X86_64


def reservation_size(image: Image, config: Declaration, layouts: Layouts, gate: str) -> int:
    """The __DATA,__graft size the consumer linked as image needs."""
    found = census(image, config, layouts)
    manifest = pack_manifest(config, layouts, found, gate, {}, manifest_version(image))
    return len(manifest) + PLACEHOLDER_ALIGNMENT - 1 + PLACEHOLDER_SIZE * len(found.imports)


def stage_image(image: Image, config: Declaration, layouts: Layouts, gate: str) -> Image:
    """Write the manifest into image's __DATA,__graft and turn its imports into placeholders.

    Edits image's bytes in place and returns the staged image. An x86-64 import becomes a
    local placeholder definition in the symbol table (localize); an arm64e import's chained
    fixup becomes a rebase at its placeholder (rebind_to_placeholders), so neither leaves a
    reference the boot linker cannot resolve.
    """
    section = image.section("__DATA", "__graft")
    version = manifest_version(image)
    found = census(image, config, layouts)
    manifest = pack_manifest(config, layouts, found, gate, {}, version)
    first = align(section.address + len(manifest), PLACEHOLDER_ALIGNMENT) - section.address
    if first + PLACEHOLDER_SIZE * len(found.imports) > section.size:
        raise Error("manifest reservation is too small; regenerate and relink")
    placeholders = {
        name: section.address - image.base + first + i * PLACEHOLDER_SIZE
        for i, name in enumerate(sorted(found.imports))
    }
    manifest = pack_manifest(config, layouts, found, gate, placeholders, version)
    image.data[section.offset : section.offset + section.size] = bytes(section.size)
    image.data[section.offset : section.offset + len(manifest)] = manifest
    if image.arch == "arm64e":
        image = rebind_to_placeholders(image, set(found.imports), placeholders)
    else:
        image = image.localize(
            {name: (section.index, image.base + offset) for name, offset in placeholders.items()}
        )
    return image.rename_symbols(deferred_superclass_markers(image, config))


def rebind_to_placeholders(image: Image, selected: set[str], placeholders: dict[str, int]) -> Image:
    """Rewrite each chained-fixup bind to a selected import into a rebase at its placeholder.

    A dormant arm64e consumer must carry no bind the boot linker cannot resolve, just as its
    x86-64 form carries no undefined provider symbol. Each rewritten slot keeps its chain
    link and, when authenticated, its signing context, so Ambrosia's placement re-signs the
    placeholder pointer and Graft later overwrites it with the real, identically signed
    address. Returns the image read again from the edited bytes.
    """
    for fixup in image.fixups():
        if not fixup.bind or fixup.symbol not in selected:
            continue
        target = placeholders[fixup.symbol]
        nxt = struct.unpack_from("<Q", image.data, fixup.entry_offset)[0] >> 51 & 0x7FF
        if fixup.auth is not None:
            rebased = (
                target & 0xFFFFFFFF
                | fixup.auth.diversity << 32
                | int(fixup.auth.addr_div) << 48
                | fixup.auth.key << 49
                | nxt << 51
                | 1 << 63  # auth = 1, bind = 0
            )
        else:
            rebased = target & 0x7FFFFFFFFFF | nxt << 51  # bind = 0, auth = 0
        struct.pack_into("<Q", image.data, fixup.entry_offset, rebased)
    return Image(image.data)


def deferred_superclass_markers(image: Image, config: Declaration) -> dict[str, str]:
    """The renames that hide each consumer class's superClass marker from the boot linker.

    A consumer class's parent is in the System KC, which the boot linker cannot see, so
    Graft links its vtable instead. The marker keeps its bytes and relocations, under the
    same-length name graftClass.
    """
    names = {symbol.name for symbol in image.symbols}
    markers = {}
    for row in config.get("consumers", []):
        name = row["name"]
        marker = f"__ZN{len(name)}{name}10superClassE"
        if marker in names:
            markers[marker] = f"__ZN{len(name)}{name}10graftClassE"
    return markers


def gate_key(info: dict[str, Any]) -> str:
    """The activation property key gating the consumer's personalities.

    Graft sets this key on the device nub once it has bound the consumer, so the passive
    IOPropertyMatch the personalities defer on now matches and the kernel constructs and
    starts them with the imports and vtable ready. Keyed to the bundle identifier so one
    property opens every personality at once.
    """
    return info["CFBundleIdentifier"] + GATE_SUFFIX


def gated_personalities(info: dict[str, Any]) -> dict[str, Any]:
    """The consumer's personalities, named to its bundle and gated on the activation key.

    Each personality keeps its own keys and gains CFBundleIdentifier and an IOPropertyMatch
    requiring gate_key on the device nub it attaches to, and must name our device. The
    kernel registers the personalities on load, but the passive property match fails until
    Graft sets gate_key on the nub after binding, so the consumer is not even constructed
    until then. This gates matching, not only start() as an IOResourceMatch would, and
    needs none of the catalogue personality injection macOS 27 removed
    (notes/activation-gate.md).
    """
    personalities = info.get("IOKitPersonalities")
    if not personalities:
        raise Error("consumer has no personalities to gate")
    key = gate_key(info)
    gated = {}
    for name, value in personalities.items():
        personality = dict(value)
        if personality.get("IOPCIPrimaryMatch") != DEVICE_MATCH:
            raise Error("consumer personality does not name our device")
        personality["CFBundleIdentifier"] = info["CFBundleIdentifier"]
        personality["IOPropertyMatch"] = {key: True}
        gated[name] = personality
    return gated


def stage_bundle(kext: Path, config: Declaration, layouts: Layouts, out: Path) -> None:
    """Copy kext to out with its binary staged and its Info.plist gated.

    The personalities stay in the Info.plist, gated by an IOPropertyMatch Graft opens once
    it binds; only the System-collection dependencies leave. Every input is checked before
    out is created, and an existing out is refused.
    """
    info = plistlib.loads((kext / "Contents/Info.plist").read_bytes())
    binary = kext / "Contents/MacOS" / info["CFBundleExecutable"]
    info["IOKitPersonalities"] = gated_personalities(info)
    image = stage_image(Image(binary.read_bytes()), config, layouts, gate_key(info))
    dependencies = info.get("OSBundleLibraries", {})
    for library in config["libraries"]:
        if library["id"] not in dependencies:
            raise Error("consumer does not declare a selected provider dependency")
        del dependencies[library["id"]]
    if out.exists():
        raise Error("staging output already exists")
    shutil.copytree(kext, out)
    (out / "Contents/Info.plist").write_bytes(plistlib.dumps(info))
    (out / "Contents/MacOS" / info["CFBundleExecutable"]).write_bytes(image.data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("probe")
    probe.add_argument("declaration", type=Path)
    probe.add_argument("layouts", type=Path)
    probe.add_argument("flags", nargs=argparse.REMAINDER)
    reserve = commands.add_parser("reserve")
    for name in ("binary", "declaration", "layouts", "info_plist", "reservation"):
        reserve.add_argument(name, type=Path)
    stage = commands.add_parser("stage")
    for name in ("kext", "declaration", "layouts", "output"):
        stage.add_argument(name, type=Path)
    args = parser.parse_args()
    try:
        config = read_declaration(args.declaration)
        if args.command == "probe":
            flags = args.flags[1:] if args.flags[:1] == ["--"] else args.flags
            args.layouts.write_text(json.dumps(probe_layouts(config, flags), indent=2) + "\n")
        elif args.command == "reserve":
            layouts = json.loads(args.layouts.read_text())
            gate = gate_key(plistlib.loads(args.info_plist.read_bytes()))
            image = Image(args.binary.read_bytes())
            size = reservation_size(image, config, layouts, gate)
            args.reservation.write_bytes(bytes(size))
        else:
            stage_bundle(args.kext, config, json.loads(args.layouts.read_text()), args.output)
    except (Error, OSError, ValueError, KeyError, TypeError, struct.error) as error:
        parser.exit(1, f"consumer: {error}\n")


if __name__ == "__main__":
    main()
