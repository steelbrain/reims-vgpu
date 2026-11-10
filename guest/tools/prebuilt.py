#!/usr/bin/env python3
"""Stage/check the SDK-only Graft set and its source/output hashes.

    python3 -I -B guest/tools/prebuilt.py stage OUT WORK
    python3 -I -B guest/tools/prebuilt.py stage-arm64e OUT WORK
    python3 -I -B guest/tools/prebuilt.py record OUT
    python3 -I -B guest/tools/prebuilt.py check [DIR]

The x86-64 set (stage, guest/prebuilt) includes the vendored Graft build, dormant
ReimsGraphics and RgfxGate. The arm64e set (stage-arm64e, guest/prebuilt-arm64e) is
the vendored arm64e Graft build, dormant ReimsGraphics and RgfxGate for the arm64
guests Ambrosia boots.
No kernel-range gate, KDK, loader, shim or downloaded build input is used.
check also verifies guest/graft against the record Graft's tools/vendor.py
wrote beside it. stage and check both fail a set where a kext leaves one of our
own symbols undefined that no kext before it in the set defines.
"""

import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
PREBUILT = ROOT / "guest/prebuilt"
RECORD = "sources.toml"
GRAFT = ROOT / "guest/graft"
GRAFT_RECORD = "source.toml"
MANIFEST = "set.toml"
FORMAT = 1
# The Darwin releases OpenCore injects the set on: those whose row of the x86-64 matrix
# passes in full, from macOS 11 (Darwin 20). Graft decides from what the running system has,
# never from a version; this bound keeps the set off releases where the display or Metal is
# not ready, and a release joins in its own commit once its row passes.
MIN_KERNEL, MAX_KERNEL = "20.0.0", ""
SET = (
    ("Graft.kext", "Binds and routes the owned guest kexts"),
    ("kext/ReimsGraphics.kext", "The reims-graphics display, accelerator and device control"),
    ("rgfxgate/RgfxGate.kext", "Installs the reims-graphics Metal bundle and admits it"),
)
# The arm64e set, for the arm64 guests Ambrosia boots: from macOS 14 (Darwin 23). arm64 macOS 11
# to 13 are not supported for now: 13.6 offers the device no MSI, and the kexts have no INTx
# path. Ambrosia's config bundle places these kexts.
MIN_KERNEL_ARM64E = "23.0.0"
SET_ARM64E = (
    ("Graft-arm64e.kext", "Binds and routes the owned guest kexts"),
    ("kext/ReimsGraphics.kext", "The reims-graphics display and accelerator"),
    ("rgfxgate/RgfxGate.kext", "Installs the reims-graphics Metal bundle and admits it"),
)
HEADER = (
    "# The kexts an x86-64 macOS guest's OpenCore image injects for the reims-graphics",
    "# device: one [[kext]] per Kernel > Add entry, in order, each the bundle",
)
HEADER_ARM64E = (
    "# The kexts an arm64 macOS guest's Ambrosia config places for the reims-graphics",
    "# device: one [[kext]] per Kernel > Add entry, in order, each the bundle",
)
# What the set is built from, below the repository's root: what each part's
# build reads (guest/kext, guest/bundle, guest/rgfxgate, guest/graft and the
# wire header), and the files that build and stage it.
INPUTS = (
    "crates/reims-vgpu-graphics/include",
    "guest/Makefile",
    "guest/tools/prebuilt.py",
    "guest/kext/Makefile", "guest/kext/src",
    "guest/bundle/Makefile", "guest/bundle/Info.plist", "guest/bundle/src",
    "guest/rgfxgate/Makefile", "guest/rgfxgate/Info.plist", "guest/rgfxgate/src",
    "guest/rgfxgate/tools",
    "guest/graft",
)
IGNORED = (".DS_Store", "__pycache__")
# Our own names: a symbol with one of these in it is the set's own code, which a kext of the set
# defines, never the kernel or Apple's kexts. A boot linker (OpenCore's for an x86-64 guest,
# Ambrosia's for an arm64 one) resolves a kext's undefined symbols against the kernel and the
# kexts linked before it, so one of ours that no kext before it defines leaves the kext
# unlinkable and the guest unbootable: an arm64e ReimsGraphics that called a member compiled for
# x86-64 alone stopped Ambrosia before the kernel ("boot failed: NoSymbol").
OWN_NAMES = re.compile(r"ReimsGraphics|Rgfx|rgfx|RGFX|Graft|graft")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plist(kext):
    return plistlib.loads((kext / "Contents/Info.plist").read_bytes())


def files_under(top):
    """Every file below `top`, relative to it, sorted; Finder's and Python's
    litter left out."""
    out = []
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames[:] = sorted(d for d in dirnames if d not in IGNORED)
        for name in filenames:
            if name not in IGNORED:
                out.append(Path(dirpath, name).relative_to(top).as_posix())
    return sorted(out)


def sources():
    """{path: sha256} of INPUTS' files that git tracks or would add (their
    contents in the work tree); every file below them where git cannot list
    them."""
    try:
        listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others",
                                 "--exclude-standard", "--", *INPUTS],
                                capture_output=True, check=True).stdout.decode().split("\0")
        paths = sorted(p for p in listed if p)
    except (OSError, subprocess.CalledProcessError):
        paths = []
        for top in INPUTS:
            if (ROOT / top).is_file():
                paths.append(top)
            else:
                paths += [f"{top}/{p}" for p in files_under(ROOT / top)]
        paths.sort()
    # A tracked file deleted in the work tree is gone.
    return {p: sha256(ROOT / p) for p in paths if (ROOT / p).is_file()}


def quote(text):
    if not re.fullmatch(r"[ -~]*", text):
        raise SystemExit(f"prebuilt: {text!r} is not printable ASCII")
    return json.dumps(text)


def stage(out, work, arch="x86_64"):
    arm64e = arch == "arm64e"
    members, header = (SET_ARM64E, HEADER_ARM64E) if arm64e else (SET, HEADER)
    min_kernel = MIN_KERNEL_ARM64E if arm64e else MIN_KERNEL
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    kexts = [(work / src, comment) for src, comment in members]
    infos = {}
    for kext, _ in kexts:
        info = plist(kext)
        exe = kext / "Contents/MacOS" / info["CFBundleExecutable"]
        if not exe.is_file():
            raise SystemExit(f"prebuilt: {kext} has no executable {exe.name}")
        infos[kext.name] = info
    # Graft's staging keeps the personalities, each gated on the device nub by the property
    # Graft sets once it has bound the kext, so none is even constructed before that.
    personalities = infos["ReimsGraphics.kext"].get("IOKitPersonalities", {})
    gate = {"ai.aneesiqbal.ReimsGraphics.graft-bound": True}
    if not personalities or any(p.get("IOPropertyMatch") != gate for p in personalities.values()):
        raise SystemExit("prebuilt: dormant ReimsGraphics has a personality without Graft's gate")
    dependencies = infos["ReimsGraphics.kext"].get("OSBundleLibraries", {})
    if {"com.apple.iokit.IOGraphicsFamily", "com.apple.AppleGraphicsDeviceControl"} & set(dependencies):
        raise SystemExit("prebuilt: dormant ReimsGraphics still names System-KC dependencies")
    graft = "Graft-arm64e.kext" if arm64e else "Graft.kext"
    if sha256(work / graft / "Contents/MacOS/Graft") != sha256(GRAFT / graft / "Contents/MacOS/Graft"):
        raise SystemExit(f"prebuilt: {graft} differs from the vendored build")
    for kext, _ in kexts:
        exe = kext / "Contents/MacOS" / infos[kext.name]["CFBundleExecutable"]
        if arch not in tool("lipo", "-archs", str(exe)).split():
            raise SystemExit(f"prebuilt: {kext.name} has no {arch} slice")
    unresolved = unresolved_own(
        [(k.name, k / "Contents/MacOS" / infos[k.name]["CFBundleExecutable"]) for k, _ in kexts], arch)
    if unresolved:
        raise SystemExit("\n".join(f"prebuilt: {e}" for e in unresolved))
    # Each kext after those of the set it names.
    ids = [infos[k.name]["CFBundleIdentifier"] for k, _ in kexts]
    for i, (kext, _) in enumerate(kexts):
        later = set(infos[kext.name].get("OSBundleLibraries", {})) & set(ids[i:])
        if later:
            raise SystemExit(f"prebuilt: {kext.name} names {sorted(later)}, which come at or after it")
    target = "prebuilt-arm64e" if arm64e else "prebuilt"
    lines = [
        *header,
        f"# beside this file named by BundlePath. Written by `make -C guest {target}`; do not edit.",
        f"# The keys are OpenCore's, but sha256, the SHA-256 of the entry's executable.",
        f"format = {FORMAT}",
        f"MinKernel = {quote(min_kernel)}",
        f"MaxKernel = {quote(MAX_KERNEL)}",
    ]
    for kext, comment in kexts:
        exe = infos[kext.name]["CFBundleExecutable"]
        shutil.copytree(kext, out / kext.name, ignore=shutil.ignore_patterns(*IGNORED))
        lines += [
            "",
            "[[kext]]",
            f"BundlePath = {quote(kext.name)}",
            f"ExecutablePath = {quote('Contents/MacOS/' + exe)}",
            f"PlistPath = {quote('Contents/Info.plist')}",
            f"Arch = {quote(arch)}",
            f"MinKernel = {quote(min_kernel)}",
            f"MaxKernel = {quote(MAX_KERNEL)}",
            f"Comment = {quote(comment)}",
            f"sha256 = {quote(sha256(kext / 'Contents/MacOS' / exe))}",
        ]
    (out / MANIFEST).write_text("\n".join(lines) + "\n")
    print(f"prebuilt: {out}: {', '.join(k.name for k, _ in kexts)}")


def macho_symbols(exe, arch):
    """The external symbols the Mach-O at `exe` (its `arch` slice, where it is fat) leaves
    undefined, and those it defines: two sets of names, from its LC_SYMTAB."""
    data = Path(exe).read_bytes()
    base = 0
    if int.from_bytes(data[:4], "big") == 0xCAFEBABE:
        cputype = {"x86_64": 0x01000007, "arm64e": 0x0100000C}[arch]
        for i in range(int.from_bytes(data[4:8], "big")):
            entry = data[8 + 20 * i: 28 + 20 * i]
            if int.from_bytes(entry[:4], "big") == cputype:
                base = int.from_bytes(entry[8:12], "big")
                break
        else:
            raise SystemExit(f"prebuilt: {exe} has no {arch} slice")
    if int.from_bytes(data[base:base + 4], "little") != 0xFEEDFACF:
        raise SystemExit(f"prebuilt: {exe} is not a 64-bit Mach-O")
    ncmds = int.from_bytes(data[base + 16:base + 20], "little")
    at, undefined, defined = base + 32, set(), set()
    for _ in range(ncmds):
        cmd, size = (int.from_bytes(data[at + k:at + k + 4], "little") for k in (0, 4))
        if cmd == 0x2:  # LC_SYMTAB
            symoff, nsyms, stroff = (int.from_bytes(data[at + k:at + k + 4], "little")
                                     for k in (8, 12, 16))
            for i in range(nsyms):
                entry = base + symoff + 16 * i
                strx = int.from_bytes(data[entry:entry + 4], "little")
                n_type = data[entry + 4]
                start = base + stroff + strx
                name = data[start:data.index(b"\0", start)].decode()
                if n_type & 0xE0 or not n_type & 0x01:  # a debugging entry, or not external
                    continue
                if n_type & 0x0E == 0x00:  # N_UNDF
                    undefined.add(name)
                elif n_type & 0x0E == 0x0E:  # N_SECT
                    defined.add(name)
        at += size
    return undefined, defined


def unresolved_own(kexts, arch):
    """Errors for each of our own symbols (OWN_NAMES) a kext of `kexts`, (name, executable)
    pairs in the set's order, leaves undefined where no kext before it defines it."""
    errors, defined = [], set()
    for name, exe in kexts:
        undefined, exported = macho_symbols(exe, arch)
        errors += [f"{name} leaves {symbol} undefined, and no kext before it in the set defines it"
                   for symbol in sorted(undefined) if OWN_NAMES.search(symbol) and symbol not in defined]
        defined |= exported
    return errors


def tool(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def record(out):
    lines = [
        "# What the kexts beside this file were built from: written by `make -C guest prebuilt`,",
        "# and checked by `make -C guest prebuilt-check` and `cargo xtask check`, which fail",
        "# when a source or an output differs from its SHA-256 here (guest/tools/prebuilt.py).",
        "# [built] records the local toolchain; every input is in the tree.",
        "[built]",
        f"xcode = {quote(' '.join(tool('xcodebuild', '-version').split()))}",
        f"sdk = {quote(Path(tool('xcrun', '--sdk', 'macosx', '--show-sdk-path')).name)}",
        f"clang = {quote(tool('xcrun', 'clang', '--version').splitlines()[0])}",
        "",
        "[sources]",
        *(f"{quote(p)} = {quote(h)}" for p, h in sources().items()),
        "",
        "[outputs]",
        *(f"{quote(p)} = {quote(sha256(out / p))}" for p in files_under(out) if p != RECORD),
    ]
    (out / RECORD).write_text("\n".join(lines) + "\n")
    print(f"prebuilt: {out / RECORD}")


def read_record(path, top_level=False):
    """A record's tables: {section: {key: value}}. Its own subset of TOML:
    `[section]` headers, and `key = "value"` lines whose key is bare or a
    string. With top_level, lines before the first header are table ""."""
    tables, section = {}, ({} if top_level else None)
    if top_level:
        tables[""] = section
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if m := re.fullmatch(r"\[(\w+)\]", line):
            section = tables.setdefault(m[1], {})
        elif (m := re.fullmatch(r'(\w+|"(?:[^"\\]|\\.)*") = ("(?:[^"\\]|\\.)*")', line)) and section is not None:
            section[json.loads(m[1]) if m[1].startswith('"') else m[1]] = json.loads(m[2])
        else:
            raise SystemExit(f"prebuilt: {path}:{n}: not a line of the record")
    return tables


def check_graft():
    """Errors where guest/graft differs from the record of the Graft commit
    it was copied from."""
    path = GRAFT / GRAFT_RECORD
    if not path.is_file():
        return [f"no guest/graft/{GRAFT_RECORD}: copy Graft with its tools/vendor.py"]
    tables = read_record(path, top_level=True)
    if not re.fullmatch(r"[0-9a-f]{40}", tables.get("", {}).get("commit", "")):
        return [f"guest/graft/{GRAFT_RECORD}: records no Graft commit"]
    want, present = tables.get("files", {}), set(files_under(GRAFT)) - {GRAFT_RECORD}
    errors = [f"guest/graft/{p}: missing" for p in sorted(want.keys() - present)]
    errors += [f"guest/graft/{p}: not the file Graft's record names" for p in sorted(want.keys() & present)
               if sha256(GRAFT / p) != want[p]]
    errors += [f"guest/graft/{p}: not in Graft's record" for p in sorted(present - want.keys())]
    if not want:
        errors.append(f"guest/graft/{GRAFT_RECORD}: records no files")
    return errors


def check(top):
    path = top / RECORD
    rel = top.relative_to(ROOT) if top.is_relative_to(ROOT) else top
    if not path.is_file():
        print(f"prebuilt: no {rel / RECORD}", file=sys.stderr)
        return 1
    tables = read_record(path)
    errors = check_graft()
    want, have = tables.get("sources", {}), sources()
    errors += [f"source changed since the set was built: {p}" for p in sorted(want.keys() & have.keys())
               if want[p] != have[p]]
    errors += [f"source added since the set was built: {p}" for p in sorted(have.keys() - want.keys())]
    errors += [f"source gone since the set was built: {p}" for p in sorted(want.keys() - have.keys())]
    outputs, present = tables.get("outputs", {}), set(files_under(top)) - {RECORD}
    for p in sorted(outputs):
        if p not in present:
            errors.append(f"{rel}/{p}: missing")
        elif sha256(top / p) != outputs[p]:
            errors.append(f"{rel}/{p}: not the file the set was built as")
    errors += [f"{rel}/{p}: not part of the built set" for p in sorted(present - outputs.keys())]
    arm64e = top.name.endswith("arm64e")
    kexts = []
    for src, _ in SET_ARM64E if arm64e else SET:
        kext = top / Path(src).name
        if (kext / "Contents/Info.plist").is_file():
            exe = kext / "Contents/MacOS" / plist(kext)["CFBundleExecutable"]
            if exe.is_file():
                kexts.append((kext.name, exe))
    errors += unresolved_own(kexts, "arm64e" if arm64e else "x86_64")
    if not outputs:
        errors.append(f"{rel / RECORD}: records no outputs")
    for e in errors:
        print(f"prebuilt: {e}", file=sys.stderr)
    if errors:
        print(f"prebuilt: {rel} is stale: on a Mac, `make -C guest prebuilt` builds it again; commit {rel} with the change",
              file=sys.stderr)
        return 1
    print(f"prebuilt: {rel}: {len(outputs)} files, built from the tree's {len(have)} sources")
    return 0


def main(argv):
    args = argv[1:]
    if len(args) == 3 and args[0] == "stage":
        stage(Path(args[1]), Path(args[2]))
    elif len(args) == 3 and args[0] == "stage-arm64e":
        stage(Path(args[1]), Path(args[2]), "arm64e")
    elif len(args) == 2 and args[0] == "record":
        record(Path(args[1]))
    elif args == ["check"]:
        return check(PREBUILT)
    elif len(args) == 2 and args[0] == "check":
        return check(Path(args[1]).resolve())
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
