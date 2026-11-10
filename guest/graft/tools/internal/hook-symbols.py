#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
"""Report which kernel symbols Graft and its owned client use each macOS release defines.

  python3 hook-symbols.py <release-dir> [<release-dir> ...]
  python3 hook-symbols.py --all CACHE        # every release directory under CACHE
  python3 hook-symbols.py --all CACHE --json # the same as JSON

A release is a directory holding an x86-64 `kernel`, `BootKernelExtensions.kc` and
`SystemKernelExtensions.kc`. Under a cache, each release's directory is named
`<version>-<build>`, such as `26.7.2-25J55`.

The symbols are in hook-symbols.txt next to this file, the first word of each line; the
rest of the line says, for a reader, how the symbol is reached and why. For each release
and each symbol, this prints whether the symbol is a defined global ("def"), an undefined
reference ("und") or absent ("--") in the kernel, the Boot KC and the System KC. It reads
Mach-O symbol tables directly and keeps nothing from the binaries but the names it was
asked about and a yes or no: no Apple bytes are copied, printed or written.

Exit status is 1 for incomplete or unreadable inputs, 2 when a wanted symbol is a defined
global in no image of a complete release, else 0.
"""

from __future__ import annotations

import argparse
import json
import mmap
import re
import struct
import sys
from pathlib import Path

IMAGES = ("kernel", "BootKernelExtensions.kc", "SystemKernelExtensions.kc")
DEFINED, UNDEFINED, ABSENT = "def", "und", "--"

# <mach-o/loader.h> and <mach-o/nlist.h>
MH_MAGIC_64 = 0xFEEDFACF
CPU_TYPE_X86_64 = 0x01000007
MH_FILESET = 0xC
LC_SYMTAB = 0x2
LC_FILESET_ENTRY = 0x80000035
N_STAB, N_TYPE, N_EXT = 0xE0, 0x0E, 0x01
N_UNDF, N_ABS, N_SECT = 0x0, 0x2, 0x0E


def read_wanted() -> list[str]:
    """The symbols hook-symbols.txt lists, in order: the first word of each line that is
    neither blank nor a comment."""
    path = Path(__file__).with_name("hook-symbols.txt")
    lines = (line.strip() for line in path.read_text().splitlines())
    return [line.split()[0] for line in lines if line and not line.startswith("#")]


def classify_bytes(data: bytes | mmap.mmap, names: list[str]) -> dict[str, str]:
    """Each name's state in a thin x86-64 image or kernel collection: DEFINED, UNDEFINED
    or ABSENT.

    A collection's fileset entries are read too; their symbol-table offsets count from
    the collection file. Every read is bounded by the input, and no other load command is
    interpreted. No bytes from the input leave this function: it returns only the states
    of the requested names.
    """
    result = dict.fromkeys(names, ABSENT)
    wanted = {name.encode(): name for name in names}

    def bounds(offset: int, size: int) -> None:
        if offset < 0 or size < 0 or offset > len(data) or size > len(data) - offset:
            raise ValueError("Mach-O range outside input")

    pending, seen = [0], set()
    while pending:
        base = pending.pop()
        if base in seen:
            raise ValueError("duplicate fileset entry")
        seen.add(base)
        bounds(base, 32)
        magic, cpu, _, filetype, count, size, _, _ = struct.unpack_from("<8I", data, base)
        if magic != MH_MAGIC_64 or cpu != CPU_TYPE_X86_64:
            raise ValueError("expected thin x86-64 Mach-O")
        bounds(base + 32, size)
        at, end = base + 32, base + 32 + size
        table = None
        for _ in range(count):
            if end - at < 8:
                raise ValueError("truncated Mach-O load command")
            command, length = struct.unpack_from("<II", data, at)
            if length < 8 or length % 8 or length > end - at:
                raise ValueError("invalid Mach-O load command")
            if command == LC_SYMTAB:
                if length != 24 or table is not None:
                    raise ValueError("invalid symbol-table command")
                table = struct.unpack_from("<4I", data, at + 8)
            elif command == LC_FILESET_ENTRY:
                if filetype != MH_FILESET or length < 32:
                    raise ValueError("invalid fileset entry")
                _, offset, name, _ = struct.unpack_from("<QQII", data, at + 8)
                if name < 32 or name >= length or data.find(b"\0", at + name, at + length) < 0:
                    raise ValueError("invalid fileset identifier")
                pending.append(offset)
            at += length
        if at != end:
            raise ValueError("load-command size mismatch")
        if table is None:
            if filetype != MH_FILESET:
                raise ValueError("image has no symbol table")
            continue
        symoff, nsyms, stroff, strsize = table
        bounds(symoff, nsyms * 16)
        bounds(stroff, strsize)
        for i in range(nsyms):
            strx, kind, _, _, _ = struct.unpack_from("<IBBHQ", data, symoff + i * 16)
            if kind & N_STAB or not kind & N_EXT:  # skip debugging and local symbols
                continue
            if strx >= strsize:
                raise ValueError("symbol name outside string table")
            stop = data.find(b"\0", stroff + strx, stroff + strsize)
            if stop < 0:
                raise ValueError("unterminated symbol name")
            name = wanted.get(data[stroff + strx : stop])
            if name is None:
                continue
            if kind & N_TYPE in (N_SECT, N_ABS):
                result[name] = DEFINED
            elif kind & N_TYPE == N_UNDF and result[name] == ABSENT:
                result[name] = UNDEFINED
    return result


def classify(path: Path, names: list[str]) -> dict[str, str]:
    with path.open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
        return classify_bytes(data, names)


def classify_release(release: Path, names: list[str]) -> dict[str, dict[str, str]]:
    """Each image's classify result for a release directory, which must hold all three."""
    found = {}
    for image in IMAGES:
        path = release / image
        if not path.is_file():
            raise ValueError(f"incomplete inputs: {image} is absent")
        found[image] = classify(path, names)
    return found


def cached_releases(cache: Path) -> list[Path]:
    """The release directories under cache, named <version>-<build>.

    Other directories in the cache, such as work trees or KDKs, are not releases.
    """
    return sorted(
        p for p in cache.iterdir() if p.is_dir() and re.fullmatch(r"\d+(\.\d+)+-\w+", p.name)
    )


def defined_anywhere(found: dict[str, dict[str, str]], symbol: str) -> bool:
    return any(found[image].get(symbol) == DEFINED for image in IMAGES)


def print_report(report: dict[str, dict[str, dict[str, str]]], names: list[str]) -> None:
    """One table per release: each symbol's state in the kernel, Boot KC and System KC."""
    for release, found in report.items():
        print(f"== {release}")
        print(f"   {'symbol':<48} {'kernel':<7} {'boot':<6} {'system':<7}")
        for symbol in names:
            kernel, boot, system = (found[image].get(symbol, ABSENT) for image in IMAGES)
            flag = "" if defined_anywhere(found, symbol) else "   <- resolved nowhere"
            print(f"   {symbol:<48} {kernel:<7} {boot:<6} {system:<7}{flag}")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("releases", nargs="*", type=Path)
    parser.add_argument(
        "--all", type=Path, metavar="CACHE", help="every release directory under CACHE"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    releases = list(args.releases)
    if args.all:
        releases += cached_releases(args.all)
    if not releases:
        parser.error("name at least one release directory, or pass --all CACHE")

    names = read_wanted()
    report = {}
    missing = []
    incomplete = False
    for release in releases:
        try:
            found = classify_release(release, names)
        except (ValueError, OSError) as error:
            sys.stderr.write(f"{release.name}: {error}\n")
            incomplete = True
            continue
        report[release.name] = found
        missing += [(release.name, s) for s in names if not defined_anywhere(found, s)]

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_report(report, names)

    if missing:
        sys.stderr.write("resolved nowhere:\n")
        for release, symbol in missing:
            sys.stderr.write(f"  {release}: {symbol}\n")
    return 1 if incomplete else (2 if missing else 0)


if __name__ == "__main__":
    sys.exit(main())
