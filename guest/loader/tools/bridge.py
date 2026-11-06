"""The loader's table for one consumer kext (README.md): every external
relocation of the consumer that names an export of one of the given System
KC libraries, with what the loader must check before it writes the
library's real address there.

A site's kind:
- call: a rel32 branch in the consumer's code (a call or a tail-call jmp);
- got: a pointer in a __got section;
- vtable: a pointer inside one of the consumer's own vtables (a `__ZTV`
  symbol's bytes): an inherited slot, compiled as the name of the library's
  method;
- superclass: the pointer of a `superClass` symbol, the consumer's class's
  metaclass link to its library superclass;
- const: any other pointer.

Each site has its offset from the consumer's base (its first segment's
vmaddr, 0 for a kext), its kind, its width (8 for a pointer, 4 for a rel32
field), the symbol, the addend the consumer's file holds there, and, for a
branch, the opcode byte before the field. Each library has its bundle ID,
version and binary name, its binary's path below the directory its
outermost kext is in (the KDK's Extensions, the guest's
/System/Library/Extensions), and its `classes`: each class of it that the
consumer derives from (a superclass site's gMetaClass), as the library lays
it out, for the loader's checks at load (Loader.cpp, layoutsMatch). A class
has its instance size, the size argument of its OSMetaClass constructor
call in the library, and its vtable, slot by slot: the method's name, and
whether the library imports it from the kernel (one of the KDK kernel's
exports, beside the Extensions directory) or defines it.

The libraries and the kernel are the KDK's binaries, untrusted and only
read (macho.py). Ours. Run with `python3 -I`.
"""

import bisect
import collections
import os
import plistlib
import struct

import macho

S_ATTR_PURE_INSTRUCTIONS = 0x80000000
S_ATTR_SOME_INSTRUCTIONS = 0x00000400
# The only relocations a bridge site can be (mach-o/x86_64/reloc.h): an
# 8-byte pointer, and the rel32 field of a call or of a tail-call jmp.
X86_64_RELOC_UNSIGNED = 0
X86_64_RELOC_BRANCH = 2
CALL_REL32 = 0xE8
JMP_REL32 = 0xE9
METACLASS = "10gMetaClassE"
# OSMetaClass's constructor, which every class's static initializer calls
# with the class's name, superclass and instance size.
OSMETACLASS_CTOR = "__ZN11OSMetaClassC2EPKcPKS_j"


def vtable_symbol(cls):
    return f"__ZTV{len(cls)}{cls}"


def exports(sl):
    """The image's external defined symbols: name -> 'function' or 'data',
    by the flags of the section each lies in."""
    sections = {s.index: s for s in sl.all_sections()}
    out = {}
    for s in sl.symbols():
        if not (s.is_defined and s.is_external):
            continue
        sec = sections.get(s.n_sect)
        code = sec is not None and bool(
            sec.flags & (S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS))
        out[s.name] = "function" if code else "data"
    return out


def bundle_info(binary_path):
    """The Info.plist of the kext a binary sits in: its Contents/Info.plist,
    or the flat bundle's Info.plist."""
    d = os.path.dirname(os.path.abspath(binary_path))
    while d != "/" and not d.endswith(".kext"):
        d = os.path.dirname(d)
    for p in (os.path.join(d, "Contents", "Info.plist"), os.path.join(d, "Info.plist")):
        if os.path.isfile(p):
            with open(p, "rb") as f:
                return plistlib.load(f)
    raise SystemExit(f"{binary_path}: no Info.plist for its kext")


def extensions_path(binary_path):
    """The binary's path below the directory its outermost kext is in."""
    parts = os.path.abspath(binary_path).split(os.sep)
    for i, part in enumerate(parts):
        if part.endswith(".kext"):
            return "/".join(parts[i:])
    raise SystemExit(f"{binary_path}: not in a kext")


class Ranges:
    """An image's defined symbols as address ranges, each up to the next
    symbol of its section (or the section's end)."""

    def __init__(self, sl):
        sections = {s.index: s for s in sl.all_sections()}
        by_sect = collections.defaultdict(list)
        for s in sl.symbols():
            if s.is_defined and s.n_sect in sections:
                by_sect[s.n_sect].append((s.value, s.name))
        self.rows = []
        for idx, syms in by_sect.items():
            sec = sections[idx]
            syms.sort()
            for i, (value, name) in enumerate(syms):
                end = syms[i + 1][0] if i + 1 < len(syms) else sec.addr + sec.size
                self.rows.append((value, end, name))
        self.rows.sort()
        self.starts = [r[0] for r in self.rows]

    def name_at(self, addr):
        """The symbol whose range holds `addr`, or None. Of several symbols at
        one address the last listed is taken; the kinds asked of it (a
        vtable, a superClass pointer) are not aliased in a kext."""
        i = bisect.bisect_right(self.starts, addr) - 1
        while i >= 0 and self.rows[i][0] == self.starts[i]:
            value, end, name = self.rows[i]
            if value <= addr < end:
                return name
            if value < addr:
                return None
            i -= 1
        return None


class Image:
    """A library's x86_64 image as its classes are read from it: its
    symbols by address, and its relocated fields."""

    def __init__(self, sl):
        self.sl = sl
        self.text = sl.section("__TEXT", "__text")
        self.names_at = collections.defaultdict(list)
        self.defined, self.undefined = {}, set()
        for s in sl.symbols():
            if s.is_defined:
                self.names_at[s.value].append(s.name)
                self.defined.setdefault(s.name, s.value)
            elif s.is_undefined and s.is_external:
                self.undefined.add(s.name)
        self.fields = {r.address: r for r in sl.relocations()}

    def name_at(self, addr):
        """The first name at `addr` that is not an assembler label."""
        return next((n for n in self.names_at.get(addr, ()) if not n.startswith(("ltmp", "L"))), None)

    def ref(self, field):
        """The name a relocated pointer, or a rel32 field, reaches: the
        symbol a bind names, else the name at the address it holds."""
        r = self.fields.get(field)
        if r is not None and r.symbol:
            return r.symbol
        if r is not None and r.length == 3 and not r.pcrel:
            return self.name_at(int.from_bytes(self.sl.read(field, 8), "little"))
        return self.name_at(field + 4 + struct.unpack("<i", self.sl.read(field, 4))[0])

    def metaclass_sizes(self):
        """{class: size} from each `mov ecx, imm32` (b9) followed by a call
        (e8) to OSMetaClass's constructor, the class named by the string
        the last `lea rsi, [rip + ...]` (48 8d 35) before it loads."""
        out = {}
        code = self.sl.read(self.text.addr, self.text.size)
        for i in range(len(code) - 10):
            if code[i] != 0xB9 or code[i + 5] != 0xE8:
                continue
            if self.ref(self.text.addr + i + 6) != OSMETACLASS_CTOR:
                continue
            j = code.rfind(b"\x48\x8d\x35", max(0, i - 40), i)
            if j < 0:
                continue
            field = self.text.addr + j + 3
            r = self.fields.get(field)
            if r is not None and r.symbol:
                continue
            disp = struct.unpack("<i", self.sl.read(field, 4))[0]
            out[self.sl.cstring(field + 4 + disp)] = struct.unpack("<I", code[i + 1:i + 5])[0]
        return out

    def vtable(self, cls):
        """[slot name] of `cls`'s vtable here, or None: the pointers after
        the two-word header up to the zero word."""
        addr = self.defined.get(vtable_symbol(cls))
        if addr is None:
            return None
        out, a = [], addr + 16
        while True:
            if a not in self.fields:
                if int.from_bytes(self.sl.read(a, 8), "little") == 0:
                    return out
                raise SystemExit(f"{cls}'s vtable slot {a:#x} has no relocation")
            out.append(self.ref(a))
            a += 8


def kind_of(sl, ranges, r):
    if r.pcrel:
        return "call"
    sec = sl.section_at(r.address)
    if sec is not None and sec.sectname == "__got":
        return "got"
    owner = ranges.name_at(r.address)
    if owner is not None and owner.startswith("__ZTV"):
        return "vtable"
    if owner is not None and owner.endswith("10superClassE"):
        return "superclass"
    return "const"


class Provider:
    """One System KC library the consumer links against, and its sites."""

    def __init__(self, path, sl, info):
        self.path, self.sl = path, sl
        self.name = os.path.basename(path)
        self.library = info["CFBundleIdentifier"]
        self.version = info["CFBundleVersion"]
        self.exports = exports(sl)
        self.sites = []

    def symbols(self, kind):
        return {s["symbol"] for s in self.sites if self.exports[s["symbol"]] == kind}

    def superclasses(self):
        return {s["symbol"] for s in self.sites if s["kind"] == "superclass"}


def analyze(consumer, libraries):
    """For each library path, a Provider with the consumer's sites that name
    its exports, in address order."""
    sl = macho.x86_64(consumer)
    ranges = Ranges(sl)
    base = sl.segments[0].vmaddr if sl.segments else 0
    relocs = sorted((r for r in sl.relocations() if r.symbol), key=lambda r: r.address)
    out = []
    for path in libraries:
        p = Provider(path, macho.x86_64(path), bundle_info(path))
        for r in relocs:
            if r.symbol not in p.exports:
                continue
            width = 1 << r.length
            if (r.pcrel and (width, r.type) != (4, X86_64_RELOC_BRANCH)) or \
                    (not r.pcrel and (width, r.type) != (8, X86_64_RELOC_UNSIGNED)):
                raise SystemExit(f"unsupported bridge relocation at {r.address:#x}")
            addend = struct.unpack("<q" if width == 8 else "<i", sl.read(r.address, width))[0]
            site = {"offset": r.address - base, "kind": kind_of(sl, ranges, r),
                    "width": width, "symbol": r.symbol, "addend": addend}
            if r.pcrel:
                site["opcode"] = sl.read(r.address - 1, 1)[0]
                if site["opcode"] not in (CALL_REL32, JMP_REL32):
                    raise SystemExit(f"unsupported branch opcode at {r.address:#x}")
            p.sites.append(site)
        out.append(p)
    return out


def class_of(metaclass):
    """The class a `__ZN<length><class>10gMetaClassE` symbol names."""
    digits = metaclass[4:].split(METACLASS)[0]
    n = len(digits) - len(digits.lstrip("0123456789"))
    name = digits[n:]
    if not metaclass.startswith("__ZN") or not metaclass.endswith(METACLASS) or \
            not n or int(digits[:n]) != len(name):
        raise SystemExit(f"{metaclass}: not a top-level class's gMetaClass")
    return name


def kdk_kernel(library_path):
    """The KDK kernel beside the Extensions directory the library is in."""
    d = os.path.dirname(os.path.abspath(library_path))
    while d != "/" and os.path.basename(d) != "Extensions":
        d = os.path.dirname(d)
    path = os.path.join(os.path.dirname(d), "Kernels", "kernel")
    if not os.path.isfile(path):
        raise SystemExit(f"{library_path}: no KDK kernel at {path}")
    return path


def layouts(provider):
    """The library's classes the consumer derives from, as the library lays
    them out: [{name, vtable, size, slots}], each slot {symbol, kernel}.
    Refuses a class without both, and a slot the library imports from
    anywhere but the kernel."""
    image = Image(provider.sl)
    sizes = image.metaclass_sizes()
    kernel = exports(macho.x86_64(kdk_kernel(provider.path)))
    out = []
    for metaclass in sorted(provider.superclasses()):
        name = class_of(metaclass)
        slots = image.vtable(name)
        if not sizes.get(name) or not slots:
            raise SystemExit(f"{provider.name}: {name} has no OSMetaClass size or vtable")
        rows = []
        for symbol in slots:
            imported = symbol in image.undefined
            if imported and symbol not in kernel:
                raise SystemExit(f"{provider.name}: {name}'s slot {symbol} is not the kernel's")
            if not imported and symbol not in image.defined:
                raise SystemExit(f"{provider.name}: {name}'s slot {symbol} is not defined here")
            rows.append({"symbol": symbol, "kernel": imported})
        out.append({"name": name, "vtable": vtable_symbol(name), "size": sizes[name], "slots": rows})
    return out


def table(consumer, providers, placeholders, binaries):
    """The table as JSON. `placeholders` maps each library to its shim's
    (bundle ID, {symbol: offset in the shim's binary}), as gen-shims.py
    builds them, so each site names the offset of the placeholder it must
    hold once OpenCore has linked it (the addend aside). `binaries` maps
    each library to its shim's base name (<binary>Bridge.kext)."""
    out = {"consumer": os.path.basename(consumer), "providers": []}
    for p in providers:
        shim, offsets = placeholders[p.library]
        row = {"library": p.library, "binary": binaries[p.library],
               "version": p.version, "path": extensions_path(p.path), "shim": shim,
               "sites": [dict(s, placeholder=offsets[s["symbol"]]) for s in p.sites],
               "classes": layouts(p)}
        out["providers"].append(row)
    return out
