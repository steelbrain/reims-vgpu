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
local placeholder in the section, and the System-KC dependencies and the personalities
leave the Info.plist, for Graft to supply when it binds the consumer.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

# Python's -I leaves the script's own directory off the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graft_macho import (
    MH_KEXT_BUNDLE,
    N_WEAK_REF,
    X86_64_RELOC_BRANCH,
    X86_64_RELOC_UNSIGNED,
    Error,
    Image,
    Relocation,
)

# graft.json and the probe's layouts, as JSON reads them.
Declaration = dict[str, Any]
Layouts = dict[str, Any]

# The manifest format: src/Manifest.h, which docs/manifest.md describes.
MAGIC = 0x54465247  # "GRFT"
VERSION = 2
NO_FALLBACK = (1 << 64) - 1
IMPORT_WEAK = 1
HEADER = struct.Struct("<24I")
HEADER_SIZE = HEADER.size
LIBRARY = struct.Struct("<2I")  # identifier, anchor
IMPORT = struct.Struct("<4IQ")  # library, first candidate, candidate count, flags, fallback
CANDIDATE = struct.Struct("<I")  # name
SITE = struct.Struct("<QQqIBBH")  # offset, placeholder, addend, import, width, opcode, reserved
CLASS = struct.Struct("<7I")  # name, vtable, library, size, parent slots, first slot, slot count
SLOT = struct.Struct("<II")  # index, method
VTABLE = struct.Struct("<Q4I")  # offset, class, slot count, first override, override count
OVERRIDE = struct.Struct("<I")  # parent slot index

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
    personalities: int
    reserved: int

    @classmethod
    def unpack_from(cls, data: bytes | bytearray, offset: int = 0) -> Header:
        words = HEADER.unpack_from(data, offset)
        tables = [Table(*words[i : i + 2]) for i in range(4, 22, 2)]
        return cls(*words[:4], *tables, *words[22:])

    def pack(self) -> bytes:
        tables = [word for table in self[4:13] for word in table]
        return HEADER.pack(*self[:4], *tables, *self[13:])


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
    """A reference to an import: the pointer or branch displacement at an image offset."""

    offset: int
    symbol: str
    width: int
    opcode: int
    addend: int


@dataclass
class Vtable:
    """A consumer class's vtable, whose parent slots Graft fills from the live parent.

    offset is its address point's image offset, parent indexes the declared classes,
    count is its slot count, and overrides are the parent slots the consumer implements.
    """

    offset: int
    parent: int
    count: int
    overrides: list[int]


class Census(NamedTuple):
    """What the manifest records about a linked consumer."""

    imports: dict[str, Import]
    sites: list[Site]
    vtables: list[Vtable]


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


def compile_probe(source: str, flags: list[str]) -> tuple[Image, str]:
    """Compile source with x86-64 kext flags for macOS 11, followed by the consumer's flags.

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
            *("xcrun", "clang++", "-arch", "x86_64", "-isysroot", sdk),
            *("-mmacosx-version-min=11.0", "-mkernel", "-nostdinc"),
            *("-isystem", f"{sdk}/System/Library/Frameworks/Kernel.framework/Headers"),
            *("-DKERNEL", "-std=c++17", "-fapple-kext", "-fno-rtti", "-fno-exceptions"),
            *("-Xclang", "-fdump-vtable-layouts", *flags),
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
    slots = []
    for slot in range(entries - 3):
        relocation = relocations.get(vtable.value + VTABLE_ADDRESS_POINT + slot * POINTER_SIZE)
        if (
            relocation is None
            or not relocation.external
            or relocation.width != POINTER_SIZE
            or relocation.pcrel
            or relocation.kind != X86_64_RELOC_UNSIGNED
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
    relocations = image.relocations()
    vtables, parent_slots = consumer_vtables(image, config, layouts, relocations)
    sites = import_sites(image, imports, relocations, parent_slots)
    return Census(imports, sites, vtables)


def class_owners(config: Declaration) -> dict[str, int]:
    """The provider index that owns each class: its declared parents and listed classes."""
    owners = {cls["name"]: cls["library"] for cls in config.get("classes", [])}
    for i, library in enumerate(config["libraries"]):
        for name in library.get("classes", []):
            if name in owners and owners[name] != i:
                raise Error("class is claimed by two providers")
            owners[name] = i
    return owners


def select_imports(config: Declaration, image: Image) -> dict[str, Import]:
    """The consumer's undefined symbols that Graft binds, by name.

    A symbol is selected when graft.json's imports names it, or when it is a member,
    vtable or typeinfo of a class a provider owns. The other undefined symbols are kernel
    and Boot-KC imports, which the boot linker resolves.
    """
    owners = class_owners(config)
    policies = config.get("imports", {})
    selected = {}
    for symbol in image.symbols:
        if not symbol.undefined:
            continue
        policy = policies.get(symbol.name)
        library = policy["library"] if policy else owners.get(owner_class(symbol.name))
        if library is None:
            continue
        if not 0 <= library < len(config["libraries"]):
            raise Error("import provider index outside declaration")
        candidates = policy.get("candidates", [symbol.name]) if policy else [symbol.name]
        if not candidates or any(
            not isinstance(name, str) or not name or "\0" in name for name in candidates
        ):
            raise Error("import candidates must be nonempty symbol names")
        fallback = policy.get("fallback") if policy else None
        selected[symbol.name] = Import(
            library=library,
            candidates=candidates,
            fallback=image.symbol(fallback).value - image.base if fallback else NO_FALLBACK,
            weak=bool(symbol.descriptor & N_WEAK_REF),
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


def pack_manifest(
    config: Declaration,
    layouts: Layouts,
    found: Census,
    personalities: str,
    placeholders: dict[str, int],
) -> bytes:
    """The manifest's bytes.

    placeholders maps each import to its placeholder's image offset. Without them every
    site's placeholder is 0, which gives the manifest its final size.
    """
    strings = StringTable()
    personality = strings.add(personalities)
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
            site.opcode,
            0,
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
    vtables, overrides = [], []
    for vtable in found.vtables:
        vtables.append(
            (vtable.offset, vtable.parent, vtable.count, len(overrides), len(vtable.overrides))
        )
        overrides += [(slot,) for slot in vtable.overrides]
    rows = [
        (LIBRARY, libraries),
        (IMPORT, imports),
        (CANDIDATE, candidates),
        (SITE, sites),
        (CLASS, classes),
        (SLOT, slots),
        (VTABLE, vtables),
        (OVERRIDE, overrides),
    ]
    return assemble(strings.data, rows, personality)


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
    strings: bytes | bytearray, rows: list[tuple[struct.Struct, list[tuple]]], personality: int
) -> bytes:
    """The header, then the string table, then each table's packed rows."""
    body = bytearray(strings)
    tables = [Table(HEADER_SIZE, len(strings))]
    for layout, table in rows:
        tables.append(Table(HEADER_SIZE + len(body), len(table)))
        body += b"".join(layout.pack(*row) for row in table)
    size = HEADER_SIZE + len(body)
    header = Header(MAGIC, VERSION, size, HEADER_SIZE, *tables, personality, 0)
    return header.pack() + body


def align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def reservation_size(
    image: Image, config: Declaration, layouts: Layouts, personalities: str
) -> int:
    """The __DATA,__graft size the consumer linked as image needs."""
    found = census(image, config, layouts)
    manifest = pack_manifest(config, layouts, found, personalities, {})
    return len(manifest) + PLACEHOLDER_ALIGNMENT - 1 + PLACEHOLDER_SIZE * len(found.imports)


def stage_image(image: Image, config: Declaration, layouts: Layouts, personalities: str) -> Image:
    """Write the manifest into image's __DATA,__graft and turn its imports into placeholders.

    Edits image's bytes in place and returns the staged image.
    """
    section = image.section("__DATA", "__graft")
    found = census(image, config, layouts)
    manifest = pack_manifest(config, layouts, found, personalities, {})
    first = align(section.address + len(manifest), PLACEHOLDER_ALIGNMENT) - section.address
    if first + PLACEHOLDER_SIZE * len(found.imports) > section.size:
        raise Error("manifest reservation is too small; regenerate and relink")
    placeholders = {
        name: section.address - image.base + first + i * PLACEHOLDER_SIZE
        for i, name in enumerate(sorted(found.imports))
    }
    manifest = pack_manifest(config, layouts, found, personalities, placeholders)
    image.data[section.offset : section.offset + section.size] = bytes(section.size)
    image.data[section.offset : section.offset + len(manifest)] = manifest
    image = image.localize(
        {name: (section.index, image.base + offset) for name, offset in placeholders.items()}
    )
    return image.rename_symbols(deferred_superclass_markers(image, config))


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


def personality_xml(info: dict[str, Any]) -> str:
    """The Info.plist's personalities as XML, each naming the consumer's bundle identifier."""
    personalities = info.get("IOKitPersonalities")
    if not personalities:
        raise Error("consumer has no personalities to publish")
    personalities = {key: dict(value) for key, value in personalities.items()}
    for personality in personalities.values():
        personality["CFBundleIdentifier"] = info["CFBundleIdentifier"]
    return plistlib.dumps(personalities).decode()


def stage_bundle(kext: Path, config: Declaration, layouts: Layouts, out: Path) -> None:
    """Copy kext to out with its binary staged and its Info.plist made dormant.

    Every input is checked before out is created, and an existing out is refused.
    """
    info = plistlib.loads((kext / "Contents/Info.plist").read_bytes())
    binary = kext / "Contents/MacOS" / info["CFBundleExecutable"]
    xml = personality_xml(info)
    del info["IOKitPersonalities"]
    image = stage_image(Image(binary.read_bytes()), config, layouts, xml)
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
            personalities = personality_xml(plistlib.loads(args.info_plist.read_bytes()))
            image = Image(args.binary.read_bytes())
            size = reservation_size(image, config, layouts, personalities)
            args.reservation.write_bytes(bytes(size))
        else:
            stage_bundle(args.kext, config, json.loads(args.layouts.read_text()), args.output)
    except (Error, OSError, ValueError, KeyError, TypeError, struct.error) as error:
        parser.exit(1, f"consumer: {error}\n")


if __name__ == "__main__":
    main()
