#!/usr/bin/env python3
"""Check that a guest release's kernel collection defines every name a kext set imports.

    python3 -I -B guest/tools/kc-imports.py KC SET

KC is a guest's kernel collection, only read: an x86-64 guest's
/System/Library/KernelCollections/BootKernelExtensions.kc, which OpenCore links the
x86-64 set into, or an arm64 guest's kernelcache, which Ambrosia links the arm64e set
against (the Preboot volume's boot/<hash>/System/Library/Caches/com.apple.kernelcaches/
kernelcache, an IMG4 whose LZFSE payload this decodes with macOS's compression_tool).
SET is a set directory with its set.toml (guest/prebuilt or guest/prebuilt-arm64e).

Each kext of the set, in set.toml's order, must find each name it imports defined by the
collection (any of its fileset entries) or exported by a kext before it in the set, as the
boot linker binds it. A name it lacks keeps the kext out of that release: Ambrosia refuses
to boot ("it imports X, which neither the kernel collection nor the other kexts define"),
and OpenCore leaves the name unbound. An x86-64 kext imports each external undefined name in
its symbol table; an arm64e kext each name its chained fixups bind, so an import that
Graft binds at run time (a consumer's, which staging turns into a rebase) is not one.

Prints the collection's kernel version and, for each kext, how many names it imports, or
the names the collection lacks; exits 1 when any is missing. Nothing it reads is written.
"""

import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile

MH_MAGIC_64 = 0xFEEDFACF
FAT_MAGIC = 0xCAFEBABE
MH_FILESET = 12
LC_SYMTAB = 0x2
LC_DYLD_CHAINED_FIXUPS = 0x80000034
LC_FILESET_ENTRY = 0x80000035
CPU = {0x01000007: "x86_64", 0x0100000C: "arm64e"}
DYLD_CHAINED_IMPORT = 1
DYLD_CHAINED_PTR_ARM64E_KERNEL = 7


def fail(message):
    raise SystemExit(f"kc-imports: {message}")


def der(data, at):
    """A DER element at `at`: its tag, where its contents start, and their length."""
    tag, length, at = data[at], data[at + 1], at + 2
    if length & 0x80:
        n = length & 0x7F
        length, at = int.from_bytes(data[at:at + n], "big"), at + n
    return tag, at, length


def der_children(data, start, length):
    out, at = [], start
    while at < start + length:
        tag, body, n = der(data, at)
        out.append((tag, body, n))
        at = body + n
    return out


def im4p_payload(data, path):
    """The Mach-O inside an IMG4 or IM4P: its payload, LZFSE-decoded where it is."""
    _, body, length = der(data, 0)
    kids = der_children(data, body, length)
    if data[kids[0][1]:kids[0][1] + kids[0][2]] == b"IMG4":
        _, body, length = kids[1]
        kids = der_children(data, body, length)
    if data[kids[0][1]:kids[0][1] + kids[0][2]] != b"IM4P" or len(kids) < 4:
        fail(f"{path} is neither a Mach-O nor an IMG4/IM4P")
    _, body, length = kids[3]
    payload = data[body:body + length]
    if payload[:4] != b"bvx2":
        fail(f"{path}: its payload is not LZFSE ({payload[:8]!r})")
    if not shutil.which("compression_tool"):
        fail(f"{path} is LZFSE-compressed: decode it on a Mac (compression_tool)")
    with tempfile.TemporaryDirectory(prefix="kc-imports.") as tmp:
        src, dst = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        Path(src).write_bytes(payload)
        subprocess.run(["compression_tool", "-decode", "-a", "lzfse", "-i", src, "-o", dst], check=True)
        return Path(dst).read_bytes()


def load_commands(data, base=0):
    ncmds = struct.unpack_from("<I", data, base + 16)[0]
    at = base + 32
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, at)
        yield cmd, at
        at += size


def cstr(data, at):
    return data[at:data.index(b"\0", at)].decode("utf-8", "replace")


def collection(path):
    """The kernel version string and every name the collection's fileset entries define
    (no debugging entry, a nonzero address: what Ambrosia binds against)."""
    data = Path(path).read_bytes()
    if data[:1] == b"\x30":
        data = im4p_payload(data, path)
    magic, cputype, _, filetype = struct.unpack_from("<IiiI", data, 0)
    if magic != MH_MAGIC_64 or filetype != MH_FILESET:
        fail(f"{path} is not a kernel collection (MH_FILESET)")
    names = set()
    for cmd, at in load_commands(data):
        if cmd != LC_FILESET_ENTRY:
            continue
        fileoff = struct.unpack_from("<Q", data, at + 16)[0]
        for sub, sat in load_commands(data, fileoff):
            if sub != LC_SYMTAB:
                continue
            symoff, nsyms, stroff = struct.unpack_from("<III", data, sat + 8)
            for i in range(nsyms):
                strx, n_type, _, _, value = struct.unpack_from("<IBBHQ", data, symoff + 16 * i)
                if not n_type & 0xE0 and value:
                    names.add(cstr(data, stroff + strx))
    version = re.search(rb"Darwin Kernel Version [^:\0]*", data)
    return CPU.get(cputype & 0xFFFFFFFF, hex(cputype)), (version[0].decode() if version else "?"), names


def thin(data, arch, path):
    """The offset of `arch`'s Mach-O in `data`, a thin or fat file."""
    if int.from_bytes(data[:4], "big") == FAT_MAGIC:
        for i in range(int.from_bytes(data[4:8], "big")):
            cputype, _, offset = struct.unpack_from(">iiI", data, 8 + 20 * i)
            if CPU.get(cputype & 0xFFFFFFFF) == arch:
                return offset
        fail(f"{path} has no {arch} slice")
    if struct.unpack_from("<I", data, 0)[0] != MH_MAGIC_64:
        fail(f"{path} is not a 64-bit Mach-O")
    return 0


def chained_binds(data, base, at, path):
    """The names an arm64e kext's chained fixups bind (DYLD_CHAINED_PTR_ARM64E_KERNEL with
    DYLD_CHAINED_IMPORT imports, as kexts are linked; any other form is refused)."""
    off = base + struct.unpack_from("<I", data, at + 8)[0]
    starts, imports_off, symbols_off, count, fmt, symfmt = struct.unpack_from("<IIIIII", data, off + 4)
    if count and (fmt != DYLD_CHAINED_IMPORT or symfmt != 0):
        fail(f"{path}: chained imports of format {fmt}, names of format {symfmt}")
    imports = [cstr(data, off + symbols_off + (struct.unpack_from("<I", data, off + imports_off + 4 * i)[0] >> 9))
               for i in range(count)]
    starts += off
    bound = set()
    for s in range(struct.unpack_from("<I", data, starts)[0]):
        seg = struct.unpack_from("<I", data, starts + 4 + 4 * s)[0]
        if not seg:
            continue
        seg += starts
        page_size, pointer_format, seg_offset = struct.unpack_from("<HHQ", data, seg + 4)
        if pointer_format != DYLD_CHAINED_PTR_ARM64E_KERNEL:
            fail(f"{path}: chained pointers of format {pointer_format}")
        page_count = struct.unpack_from("<H", data, seg + 20)[0]
        for p in range(page_count):
            start = struct.unpack_from("<H", data, seg + 22 + 2 * p)[0]
            if start == 0xFFFF:
                continue
            loc = base + seg_offset + p * page_size + start
            while True:
                raw = struct.unpack_from("<Q", data, loc)[0]
                if raw >> 62 & 1:
                    bound.add(imports[raw & 0xFFFF])
                step = raw >> 51 & 0x7FF
                if not step:
                    break
                loc += 4 * step
    return bound


def kext_symbols(path, arch):
    """The names the kext's boot linker binds, and the names it exports."""
    data = Path(path).read_bytes()
    base = thin(data, arch, path)
    undefined, defined, binds = set(), set(), None
    for cmd, at in load_commands(data, base):
        if cmd == LC_SYMTAB:
            symoff, nsyms, stroff = struct.unpack_from("<III", data, at + 8)
            for i in range(nsyms):
                strx, n_type = struct.unpack_from("<IB", data, base + symoff + 16 * i)
                if n_type & 0xE0 or not n_type & 0x01:  # a debugging entry, or not external
                    continue
                name = cstr(data, base + stroff + strx)
                if n_type & 0x0E == 0x00:
                    undefined.add(name)
                elif n_type & 0x0E == 0x0E:
                    defined.add(name)
        elif cmd == LC_DYLD_CHAINED_FIXUPS:
            binds = chained_binds(data, base, at, path)
    return (binds if arch == "arm64e" else undefined), defined


def kexts(top):
    """(bundle, executable) for each [[kext]] of SET's set.toml, in order."""
    manifest = top / "set.toml"
    if not manifest.is_file():
        fail(f"no set.toml in {top}")
    out, bundle = [], None
    for line in manifest.read_text().splitlines():
        if m := re.match(r'BundlePath = "(.*)"$', line.strip()):
            bundle = m[1]
        elif (m := re.match(r'ExecutablePath = "(.*)"$', line.strip())) and bundle:
            out.append((bundle, top / bundle / m[1]))
            bundle = None
    return out


def demangle(names):
    try:
        out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True, check=True)
        return [f"{n}  ({d})" if d != n else n for n, d in zip(names, out.stdout.splitlines())]
    except (OSError, subprocess.CalledProcessError):
        return names


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    arch, version, defined = collection(argv[1])
    print(f"kc-imports: {Path(argv[1]).name}: {arch}, {version}, {len(defined)} names")
    status = 0
    for bundle, exe in kexts(Path(argv[2])):
        imports, exports = kext_symbols(exe, arch)
        missing = sorted(imports - defined)
        if missing:
            status = 1
            print(f"kc-imports: {bundle}: {len(missing)} of its {len(imports)} imports neither the collection "
                  "nor a kext before it defines:", file=sys.stderr)
            for line in demangle(missing):
                print(f"  {line}", file=sys.stderr)
        else:
            print(f"kc-imports: {bundle}: all {len(imports)} imports resolve")
        defined |= exports
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv))
