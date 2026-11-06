#!/usr/bin/env python3
"""The kext set an x86-64 macOS guest's OpenCore image injects for the
reims-graphics device, and the record of what a copy of it was built from
(guest/Makefile, `make set` and `make prebuilt`).

    python3 -I -B guest/tools/prebuilt.py stage OUT WORK
    python3 -I -B guest/tools/prebuilt.py record OUT KDK
    python3 -I -B guest/tools/prebuilt.py check [DIR]

stage: copies the set's bundles from WORK, where guest/Makefile built each
part, into OUT, which it starts empty, in OpenCore's Kernel > Add order, and
writes OUT/set.toml, the entries in that order. It refuses a set whose loader
was made from another ReimsGraphics, whose ReimsGraphics still carries its
personalities or names a library the loader does not bind, or whose kexts
name one of the set's own kexts that comes after them.

record: writes OUT/sources.toml: the SHA-256 of each source the set is built
from ([sources], INPUTS below, as git tracks them), of each file in OUT
([outputs]), and, for the reader, the KDK's files the loader's tables were
read from and the toolchain.

check: hashes only, and builds nothing, so it runs on any host: fails unless
the tree's sources and DIR's files (default guest/prebuilt) are the ones
DIR/sources.toml records. `make -C guest prebuilt-check` and `cargo xtask
check` run it. Ours.
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
MANIFEST = "set.toml"
FORMAT = 1
# The Darwin releases the set is injected on: macOS 26.4 to 26.x. The loader
# also refuses a release whose library classes are laid out otherwise than
# the KDK it was built from (guest/loader/README.md).
MIN_KERNEL, MAX_KERNEL = "25.4.0", "25.99.99"
# The set, in Kernel > Add order (after Lilu, which the image carries first):
# each bundle's place in WORK, and its entry's comment.
SET = (
    ("loader/ReimsGraphicsLoader.kext",
     "Binds ReimsGraphics to IOGraphicsFamily and AppleGraphicsDeviceControl"),
    ("loader/staged/RgfxIOGraphicsFamilyBridge.kext",
     "IOGraphicsFamily placeholders ReimsGraphics links against"),
    ("loader/staged/RgfxAppleGraphicsDeviceControlBridge.kext",
     "AppleGraphicsDeviceControl placeholders ReimsGraphics links against"),
    ("loader/staged/ReimsGraphics.kext",
     "The reims-graphics display, accelerator and device control"),
    ("rgfxgate/RgfxGate.kext",
     "Installs the reims-graphics Metal bundle and lets macOS load it"),
)
# What the set is built from, below the repository's root: what each part's
# build reads (guest/kext, guest/bundle, guest/rgfxgate, guest/loader and the
# wire header), and the files that build and stage it.
INPUTS = (
    "crates/reims-graphics/include",
    "guest/Makefile",
    "guest/tools/prebuilt.py",
    "guest/kext/Makefile", "guest/kext/src",
    "guest/bundle/Makefile", "guest/bundle/Info.plist", "guest/bundle/src",
    "guest/rgfxgate/Makefile", "guest/rgfxgate/Info.plist", "guest/rgfxgate/src",
    "guest/rgfxgate/tools",
    "guest/loader/Makefile", "guest/loader/src", "guest/loader/tools",
)
# The KDK's files the loader's tables and shims are made from, below KDK.
KDK_FILES = (
    "IOGraphicsFamily.kext/IOGraphicsFamily",
    "AppleGraphicsControl.kext/Contents/PlugIns/AppleGraphicsDeviceControl.kext/Contents/MacOS/AppleGraphicsDeviceControl",
    "../Kernels/kernel",
)
IGNORED = (".DS_Store", "__pycache__")


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
    """{path: sha256} of INPUTS' files, as git tracks them (their contents in
    the work tree); every file below them where git cannot list them."""
    try:
        listed = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--", *INPUTS],
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


def stage(out, work):
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    kexts = [(work / src, comment) for src, comment in SET]
    infos = {}
    for kext, _ in kexts:
        info = plist(kext)
        exe = kext / "Contents/MacOS" / info["CFBundleExecutable"]
        if not exe.is_file():
            raise SystemExit(f"prebuilt: {kext} has no executable {exe.name}")
        infos[kext.name] = info
    # The loader binds exactly this ReimsGraphics, through exactly these shims.
    staged = work / "loader/staged"
    table = json.loads((staged / "ReimsGraphics.bridge/fixups.json").read_text())
    graphics = staged / "ReimsGraphics.kext"
    if table["consumer_sha256"] != sha256(graphics / "Contents/MacOS/ReimsGraphics"):
        raise SystemExit("prebuilt: the loader's table was made from another ReimsGraphics")
    if infos["ReimsGraphics.kext"].get("IOKitPersonalities"):
        raise SystemExit("prebuilt: the staged ReimsGraphics carries personalities")
    shims = {p["shim"] for p in table["providers"]}
    bound = {infos[k.name]["CFBundleIdentifier"] for k, _ in kexts if k.name.endswith("Bridge.kext")}
    if shims != bound:
        raise SystemExit(f"prebuilt: the loader binds {sorted(shims)}, the set carries {sorted(bound)}")
    # Each kext after those of the set it names.
    ids = [infos[k.name]["CFBundleIdentifier"] for k, _ in kexts]
    for i, (kext, _) in enumerate(kexts):
        later = set(infos[kext.name].get("OSBundleLibraries", {})) & set(ids[i:])
        if later:
            raise SystemExit(f"prebuilt: {kext.name} names {sorted(later)}, which come at or after it")
    lines = [
        f"# The kexts an x86-64 macOS guest's OpenCore image injects for the reims-graphics",
        f"# device: one [[kext]] per Kernel > Add entry, in order, after Lilu's, each the bundle",
        f"# beside this file named by BundlePath. Written by `make -C guest prebuilt`; do not edit.",
        f"# The keys are OpenCore's, but sha256, the SHA-256 of the entry's executable.",
        f"format = {FORMAT}",
        f"MinKernel = {quote(MIN_KERNEL)}",
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
            f"Arch = {quote('x86_64')}",
            f"MinKernel = {quote(MIN_KERNEL)}",
            f"MaxKernel = {quote(MAX_KERNEL)}",
            f"Comment = {quote(comment)}",
            f"sha256 = {quote(sha256(kext / 'Contents/MacOS' / exe))}",
        ]
    (out / MANIFEST).write_text("\n".join(lines) + "\n")
    print(f"prebuilt: {out}: {', '.join(k.name for k, _ in kexts)}")


def tool(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def record(out, kdk):
    kdk = Path(kdk)
    version = kdk.parent / "SystemVersion.plist"
    release = "unknown"
    if version.is_file():
        v = plistlib.loads(version.read_bytes())
        release = f"macOS {v.get('ProductVersion')} ({v.get('ProductBuildVersion')})"
    lines = [
        "# What the kexts beside this file were built from: written by `make -C guest prebuilt`,",
        "# and checked by `make -C guest prebuilt-check` and `cargo xtask check`, which fail",
        "# when a source or an output differs from its SHA-256 here (guest/tools/prebuilt.py).",
        "# [built] and [kdk] are for the reader: the KDK and the toolchain are not in the tree.",
        "[built]",
        f"kdk = {quote(release)}",
        f"xcode = {quote(' '.join(tool('xcodebuild', '-version').split()))}",
        f"sdk = {quote(Path(tool('xcrun', '--sdk', 'macosx', '--show-sdk-path')).name)}",
        f"clang = {quote(tool('xcrun', 'clang', '--version').splitlines()[0])}",
        "",
        "[kdk]",
        *(f"{quote(f)} = {quote(sha256(kdk / f))}" for f in KDK_FILES),
        "",
        "[sources]",
        *(f"{quote(p)} = {quote(h)}" for p, h in sources().items()),
        "",
        "[outputs]",
        *(f"{quote(p)} = {quote(sha256(out / p))}" for p in files_under(out) if p != RECORD),
    ]
    (out / RECORD).write_text("\n".join(lines) + "\n")
    print(f"prebuilt: {out / RECORD}")


def read_record(path):
    """sources.toml's tables: {section: {key: value}}. Its own subset of
    TOML: `[section]` headers, and `key = "value"` lines whose key is bare
    or a string."""
    tables, section = {}, None
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


def check(top):
    path = top / RECORD
    rel = top.relative_to(ROOT) if top.is_relative_to(ROOT) else top
    if not path.is_file():
        print(f"prebuilt: no {rel / RECORD}", file=sys.stderr)
        return 1
    tables = read_record(path)
    errors = []
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
    if not outputs:
        errors.append(f"{rel / RECORD}: records no outputs")
    for e in errors:
        print(f"prebuilt: {e}", file=sys.stderr)
    if errors:
        print(f"prebuilt: {rel} is stale: on a Mac, `make -C guest prebuilt KDK=<a macOS 26 KDK's "
              f"Extensions directory>` builds it again; commit {rel} with the change",
              file=sys.stderr)
        return 1
    print(f"prebuilt: {rel}: {len(outputs)} files, built from the tree's {len(have)} sources")
    return 0


def main(argv):
    args = argv[1:]
    if len(args) == 3 and args[0] == "stage":
        stage(Path(args[1]), Path(args[2]))
    elif len(args) == 3 and args[0] == "record":
        record(Path(args[1]), args[2])
    elif args == ["check"]:
        return check(PREBUILT)
    elif len(args) == 2 and args[0] == "check":
        return check(Path(args[1]).resolve())
    else:
        raise SystemExit(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
