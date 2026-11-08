#!/usr/bin/env python3
"""Build consumer metadata with Xcode and Python's standard library only.

  consumer.py probe DECLARATIONS LAYOUTS [-- compiler flags...]
  consumer.py reserve BINARY DECLARATIONS LAYOUTS INFO_PLIST OUTPUT
  consumer.py stage KEXT DECLARATIONS LAYOUTS OUTPUT

Link once to census imports, reserve the reported section, then link again
with -sectcreate __DATA __graft OUTPUT. Stage that second bundle. Staging
keeps all offsets with the consumer and creates no bridge kexts.
"""
import argparse
import json
from pathlib import Path
import plistlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile

# Isolated Python invocation intentionally has no current-directory imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graft_macho import Image, Error

MAGIC, VERSION, HEADER_SIZE = 0x54465247, 1, 96
NO_FALLBACK = (1 << 64) - 1


def identifier(name):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise Error("only top-level C++ class names are supported")
    return name


def vtable_name(name):
    return f"__ZTV{len(name)}{name}"


def owner_class(name):
    match = re.match(r"__(?:ZNK?|ZT[VIS])(\d+)(.*)", name)
    return match[2][:int(match[1])] if match else None


def declarations(path):
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
    if len({cls["name"] for cls in classes}) != len(classes):
        raise Error("duplicate parent class")
    for consumer in config.get("consumers", []):
        identifier(consumer["name"])
        if not consumer.get("header") or not 0 <= consumer["parent"] < len(classes):
            raise Error("consumer class has no header or parent")
    return config


def probe(config, flags):
    """Use compiler-emitted vtable relocations and sizeof constants.

    A no-member subclass with an out-of-line destructor emits an abstract
    vtable even when the parent has pure methods. It adds no slots. Only
    its destructor names differ and those map directly to the parent.
    """
    parents = config.get("classes", [])
    rows = parents + config.get("consumers", [])
    if not rows:
        return {"classes": [], "consumers": []}
    source = [f'#include "{row["header"]}"' for row in rows]
    for i, row in enumerate(rows):
        name = identifier(row["name"])
        source += [f"class GraftLayoutProbe{i} : public {name} {{ public:",
            f"~GraftLayoutProbe{i}() override; }};", f"GraftLayoutProbe{i}::~GraftLayoutProbe{i}() {{}}"]
    source += ['extern "C" const unsigned graft_layout_sizes[] = {' +
        ",".join(f'sizeof({identifier(row["name"])})' for row in rows) + "};"]
    sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    with tempfile.TemporaryDirectory(prefix="graft-layout-", dir="/tmp") as temp:
        path = Path(temp)
        (path / "probe.cpp").write_text("\n".join(source) + "\n")
        command = ["xcrun", "clang++", "-arch", "x86_64", "-isysroot", sdk,
            "-mmacosx-version-min=11.0", "-mkernel", "-nostdinc", "-isystem",
            sdk + "/System/Library/Frameworks/Kernel.framework/Headers", "-DKERNEL",
            "-std=c++17", "-fapple-kext", "-fno-rtti", "-fno-exceptions",
            "-Xclang", "-fdump-vtable-layouts", *flags, "-c", str(path / "probe.cpp"),
            "-o", str(path / "probe.o")]
        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode:
            sys.stderr.write(process.stderr)
            raise Error("consumer layout compilation failed")
        image = Image((path / "probe.o").read_bytes())
    counts = {int(i): int(count) for i, count in
        re.findall(r"Vtable for 'GraftLayoutProbe(\d+)' \((\d+) entries\)", process.stdout)}
    sizes = image.symbol("_graft_layout_sizes")
    size_values = struct.unpack("<" + "I" * len(rows), image.read(sizes.value, 4 * len(rows)))
    relocations = {r.address: r for r in image.relocations()}
    result = []
    for i, row in enumerate(rows):
        probe_name = f"GraftLayoutProbe{i}"
        symbol = image.symbol(vtable_name(probe_name))
        if i not in counts or counts[i] < 4:
            raise Error("compiler did not report a single primary vtable")
        # The kext ABI appends one zero vcall-offset sentinel after the
        # method slots. It is part of the compiler's reported entry count,
        # but never a virtual method or a destination for parent copying.
        sentinel = symbol.value + (counts[i] - 1) * 8
        if sentinel in relocations or image.read(sentinel, 8) != bytes(8):
            raise Error("compiler vtable lacks its final zero sentinel")
        slots = []
        for index in range(counts[i] - 3):
            address = symbol.value + 16 + index * 8
            relocation = relocations.get(address)
            if relocation is None or not relocation.external or relocation.width != 8 or relocation.pcrel or relocation.kind:
                raise Error("compiler vtable slot has no supported symbolic relocation")
            method = image.symbols[relocation.symbol_index].name
            prefix = f"__ZN{len(probe_name)}{probe_name}"
            if method.startswith(prefix):
                suffix = method[len(prefix):]
                if suffix not in ("D0Ev", "D1Ev"):
                    raise Error("layout probe introduced an unexpected method")
                method = f'__ZN{len(row["name"])}{row["name"]}{suffix}'
            slots.append(method)
        result.append({"name": row["name"], "size": size_values[i], "slots": slots})
    return {"classes": result[:len(parents)], "consumers": result[len(parents):]}


def import_owners(config, image):
    classes = {cls["name"]: cls["library"] for cls in config.get("classes", [])}
    for i, library in enumerate(config["libraries"]):
        for name in library.get("classes", []):
            if name in classes and classes[name] != i:
                raise Error("class is claimed by two providers")
            classes[name] = i
    overrides = config.get("imports", {})
    found = {}
    for symbol in image.symbols:
        if not symbol.undefined:
            continue
        record = overrides.get(symbol.name)
        library = record["library"] if record else classes.get(owner_class(symbol.name))
        if library is None:
            continue
        if not 0 <= library < len(config["libraries"]):
            raise Error("import provider index outside declaration")
        candidates = record.get("candidates", [symbol.name]) if record else [symbol.name]
        if not candidates or any(not isinstance(name, str) or not name or "\0" in name for name in candidates):
            raise Error("import candidates must be nonempty symbol names")
        fallback = record.get("fallback") if record else None
        found[symbol.name] = {"library": library, "candidates": candidates,
            "fallback": image.symbol(fallback).value - image.base if fallback else NO_FALLBACK}
    absent = set(overrides) - set(found)
    if absent:
        raise Error("an explicit import policy names no undefined import")
    return found


def executable(image, address):
    return any(s.address <= address < s.address + s.size and s.flags & 0x80000400 for s in image.sections)


def model(image, config, layouts):
    if image.filetype != 0xb:
        raise Error("consumer must be linked as a kext")
    if [row["name"] for row in layouts["classes"]] != [row["name"] for row in config.get("classes", [])] or \
       [row["name"] for row in layouts["consumers"]] != [row["name"] for row in config.get("consumers", [])]:
        raise Error("layouts describe another declaration")
    imports = import_owners(config, image)
    relocations = image.relocations()
    by_address = {r.address: r for r in relocations}
    vtables = []
    excluded = set()
    for consumer, layout in zip(config.get("consumers", []), layouts["consumers"]):
        symbol = image.symbol(vtable_name(consumer["name"]))
        parent = consumer["parent"]
        parent_slots = len(layouts["classes"][parent]["slots"])
        count = len(layout["slots"])
        if count < parent_slots or image.symbol_extent(symbol) < (count + 2) * 8:
            raise Error("consumer vtable is shorter than its compiler layout")
        own = []
        for index in range(count):
            address = symbol.value + 16 + index * 8
            relocation = by_address.get(address)
            local = relocation is not None and not relocation.external
            if relocation is not None and relocation.external:
                local = image.symbols[relocation.symbol_index].defined
            if local:
                value = struct.unpack("<Q", image.read(address, 8))[0]
                if not executable(image, value):
                    raise Error("consumer override does not point to its executable code")
                if index < parent_slots:
                    own.append(index)
            elif index >= parent_slots:
                raise Error("appended consumer virtual method is not local executable code")
            if index < parent_slots:
                excluded.add(address)
        vtables.append({"offset": symbol.value + 16 - image.base, "class": parent,
            "count": count, "overrides": own})
    sites = []
    used = set()
    for relocation in relocations:
        if not relocation.external:
            continue
        name = image.symbols[relocation.symbol_index].name
        if name not in imports:
            continue
        used.add(name)
        if relocation.address in excluded:
            continue
        if relocation.pcrel:
            if relocation.width != 4 or relocation.kind != 2:
                raise Error("unsupported System-KC PC-relative relocation")
            opcode = image.read(relocation.address - 1, 1)[0]
            if opcode not in (0xe8, 0xe9):
                raise Error("unsupported System-KC branch instruction")
        else:
            if relocation.width != 8 or relocation.kind:
                raise Error("unsupported System-KC pointer relocation")
            opcode = 0
        addend = struct.unpack("<i" if relocation.width == 4 else "<q",
            image.read(relocation.address, relocation.width))[0]
        sites.append({"offset": relocation.address - image.base, "symbol": name,
            "width": relocation.width, "opcode": opcode, "addend": addend})
    if set(imports) != used:
        raise Error("selected import has no relocation site")
    sites.sort(key=lambda site: site["offset"])
    return imports, sites, sorted(vtables, key=lambda vtable: vtable["offset"])


def pack(config, layouts, imports, sites, vtables, personalities, placeholders):
    strings = bytearray(b"\0")
    interned = {}
    def string(value):
        if not value or "\0" in value:
            raise Error("manifest strings must be nonempty and contain no NUL")
        if value not in interned:
            interned[value] = len(strings)
            strings.extend(value.encode() + b"\0")
        return interned[value]
    personality = string(personalities)
    libraries = [(string(library["id"]),) for library in config["libraries"]]
    candidates, import_rows = [], []
    names = sorted(imports)
    for name in names:
        record = imports[name]
        first = len(candidates)
        candidates += [(string(candidate),) for candidate in record["candidates"]]
        import_rows.append((record["library"], first, len(record["candidates"]), 0, record["fallback"]))
    indices = {name: index for index, name in enumerate(names)}
    site_rows = [(site["offset"], placeholders.get(site["symbol"], 0), site["addend"],
        indices[site["symbol"]], site["width"], site["opcode"], 0) for site in sites]
    classes, slots = [], []
    for class_index, (declaration, layout) in enumerate(zip(config.get("classes", []), layouts["classes"])):
        first = len(slots)
        required = declaration.get("required")
        indices = set(range(len(layout["slots"]))) if required is None else set()
        if required is not None:
            for name in required:
                found = [index for index, method in enumerate(layout["slots"]) if method == name]
                if len(found) != 1:
                    raise Error("required inherited method has no unique compiler slot")
                indices.add(found[0])
        for vtable in vtables:
            if vtable["class"] == class_index:
                indices.update(vtable["overrides"])
        slots += [(index, string(layout["slots"][index])) for index in sorted(indices)]
        classes.append((string(declaration["name"]), string(vtable_name(declaration["name"])),
            declaration["library"], layout["size"], len(layout["slots"]), first, len(indices)))
    vtable_rows, overrides = [], []
    for vtable in vtables:
        first = len(overrides)
        overrides += [(index,) for index in vtable["overrides"]]
        vtable_rows.append((vtable["offset"], vtable["class"], vtable["count"], first, len(vtable["overrides"])))
    raw = bytearray(HEADER_SIZE)
    tables = []
    def table(data, count):
        tables.extend((len(raw), count))
        raw.extend(data)
    table(strings, len(strings))
    for rows, format in ((libraries, "<I"), (import_rows, "<4IQ"), (candidates, "<I"),
            (site_rows, "<QQqIBBH"), (classes, "<7I"), (slots, "<II"),
            (vtable_rows, "<Q4I"), (overrides, "<I")):
        table(b"".join(struct.pack(format, *row) for row in rows), len(rows))
    struct.pack_into("<24I", raw, 0, MAGIC, VERSION, len(raw), HEADER_SIZE, *tables, personality, 0)
    return bytes(raw)


def reservation(image, config, layouts, personalities):
    imports, sites, vtables = model(image, config, layouts)
    metadata = pack(config, layouts, imports, sites, vtables, personalities, {})
    # A distinct aligned placeholder for each symbol, including vtable
    # symbols whose address point has the conventional 16-byte addend.
    return len(metadata) + 15 + 32 * len(imports)


def stage_bytes(image, config, layouts, personalities):
    section = image.section("__DATA", "__graft")
    imports, sites, vtables = model(image, config, layouts)
    metadata = pack(config, layouts, imports, sites, vtables, personalities, {})
    cursor = ((section.address + len(metadata) + 15) & ~15) - section.address
    if cursor + 32 * len(imports) > section.size:
        raise Error("manifest reservation is too small; regenerate and relink")
    placeholders = {name: section.address - image.base + cursor + i * 32
        for i, name in enumerate(sorted(imports))}
    metadata = pack(config, layouts, imports, sites, vtables, personalities, placeholders)
    image.data[section.offset:section.offset + section.size] = bytes(section.size)
    image.data[section.offset:section.offset + len(metadata)] = metadata
    image = image.localize({name: (section.index, image.base + offset)
        for name, offset in placeholders.items()})
    # These classes' parents are resolved by Graft, so the boot linker
    # cannot patch their vtables against a real parent yet. Preserve the
    # marker's pointer and relocations while hiding only its linker name.
    names = {symbol.name for symbol in image.symbols}
    markers = {}
    for row in config.get("consumers", []):
        prefix = f'__ZN{len(row["name"])}{row["name"]}10'
        if prefix + "superClassE" in names:
            markers[prefix + "superClassE"] = prefix + "graftClassE"
    return image.rename_symbols(markers)


def personality_xml(info):
    personalities = info.get("IOKitPersonalities")
    if not personalities:
        raise Error("consumer has no personalities to publish")
    personalities = {key: dict(value) for key, value in personalities.items()}
    for personality in personalities.values():
        personality["CFBundleIdentifier"] = info["CFBundleIdentifier"]
    return plistlib.dumps(personalities).decode()


def stage_bundle(kext, config, layouts, out):
    info = plistlib.loads((kext / "Contents/Info.plist").read_bytes())
    binary = kext / "Contents/MacOS" / info["CFBundleExecutable"]
    xml = personality_xml(info)
    del info["IOKitPersonalities"]
    image = stage_bytes(Image(binary.read_bytes()), config, layouts, xml)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("probe")
    p.add_argument("declarations", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("flags", nargs=argparse.REMAINDER)
    r = sub.add_parser("reserve")
    for name in ("binary", "declarations", "layouts", "info", "output"):
        r.add_argument(name, type=Path)
    s = sub.add_parser("stage")
    for name in ("kext", "declarations", "layouts", "output"):
        s.add_argument(name, type=Path)
    args = parser.parse_args()
    try:
        config = declarations(args.declarations)
        if args.command == "probe":
            flags = args.flags[1:] if args.flags[:1] == ["--"] else args.flags
            args.output.write_text(json.dumps(probe(config, flags), indent=2) + "\n")
        elif args.command == "reserve":
            layouts = json.loads(args.layouts.read_text())
            personalities = personality_xml(plistlib.loads(args.info.read_bytes()))
            count = reservation(Image(args.binary.read_bytes()), config, layouts, personalities)
            args.output.write_bytes(bytes(count))
        else:
            stage_bundle(args.kext, config, json.loads(args.layouts.read_text()), args.output)
    except (Error, OSError, ValueError, KeyError, TypeError, struct.error) as error:
        parser.exit(1, f"consumer: {error}\n")


if __name__ == "__main__":
    main()
