#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
"""Check that a prelinked kernel collection keeps its staged consumers bindable.

    prelink.py COLLECTION KEXT...

COLLECTION is an x86-64 kernel collection after OpenCore prelinked our kexts into it.
Each KEXT is a staged consumer bundle (consumer.py stage). The check finds each
consumer's fileset entry by its bundle identifier and requires what Graft's runtime
relies on when it binds that entry:

- the entry's header starts its __TEXT, and every staged segment keeps its position
  relative to the header, because the runtime adds manifest offsets to the entry address;
- __DATA,__graft holds the staged manifest unchanged through the size the manifest
  declares; the placeholder tail after it is not compared;
- every manifest site holds its exact local placeholder plus addend, with each
  placeholder inside the reserved tail. A pointer site must be an entry of the
  collection's fixup chains, since the booter rewrites only those when it slides the
  collection, and its decoded target must be that address. A branch site's E8/E9
  displacement must reach it;
- no fixup-chain entry overlaps a branch site or the manifest, whose bytes the booter
  would otherwise rewrite.

x86-64 collections encode every pointer as a chained rebase
(DYLD_CHAINED_PTR_X86_64_KERNEL_CACHE). A region's segment_offset counts from the
collection header, and a rebase target counts from the collection's lowest segment
address. Regions are matched by segment_offset, not by load-command order, since
OpenCore reuses a slot for the region it injects. Chains may run past their starting
page, so a region's chains are walked whole.

The collection is only read, and every read is bounded by its file.
"""

from __future__ import annotations

import argparse
import plistlib
import struct
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import NamedTuple

# Python's -I leaves the script's own directory off the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from consumer import BRANCH_OPCODES, HEADER_SIZE, MAGIC, POINTER, SITE, VERSION, Header
from graft_macho import (
    CPU_TYPE_X86_64,
    LC_SEGMENT_64,
    MACH_HEADER_SIZE,
    MH_KEXT_BUNDLE,
    MH_MAGIC_64,
    SECTION_HEADER_SIZE,
    SEGMENT_COMMAND_SIZE,
    Error,
    Image,
)

# <mach-o/loader.h> and <mach-o/fixup-chains.h>
MH_FILESET = 0xC
LC_DYLD_CHAINED_FIXUPS = 0x80000034
LC_FILESET_ENTRY = 0x80000035
FILESET_ENTRY_COMMAND_SIZE = 32
CHAINED_FIXUPS_HEADER_SIZE = 28
CHAINED_STARTS_IN_SEGMENT_SIZE = 22
DYLD_CHAINED_PTR_X86_64_KERNEL_CACHE = 11
DYLD_CHAINED_PTR_START_NONE = 0xFFFF
DYLD_CHAINED_PTR_START_MULTI = 0x8000
# dyld_chained_ptr_x86_64_kernel_cache_rebase is target:30 cacheLevel:2 diversity:16
# addrDiv:1 key:2 next:12 isAuth:1. The check accepts only plain rebases into this
# collection: every field but target and next must be zero.
TARGET_MASK = (1 << 30) - 1
NEXT_SHIFT, NEXT_MASK = 51, 0xFFF
NOT_PLAIN = ~(TARGET_MASK | NEXT_MASK << NEXT_SHIFT) & ((1 << 64) - 1)


class SiteRecord(NamedTuple):
    """A manifest site row (src/Manifest.h, Site)."""

    offset: int
    placeholder: int
    addend: int
    import_index: int
    width: int
    opcode: int
    reserved: int


@dataclass
class Segment:
    """A segment of the collection or of a fileset entry. offset is collection-absolute,
    and sections maps each section's name to its (address, size)."""

    name: str
    address: int
    vmsize: int
    offset: int
    filesize: int
    sections: dict[str, tuple[int, int]]


def file_offset(segments: Iterable[Segment], address: int, width: int) -> int | None:
    """The file offset of width bytes at address, if one segment's file bytes hold them."""
    for segment in segments:
        if (
            segment.address <= address
            and width <= segment.filesize
            and address - segment.address <= segment.filesize - width
        ):
            return segment.offset + address - segment.address
    return None


@dataclass
class Entry:
    """One fileset entry's kext: its header's address and file offset, and its segments."""

    collection: Collection
    name: str
    address: int
    offset: int
    segments: dict[str, Segment]

    def read(self, address: int, width: int) -> bytes:
        offset = file_offset(self.segments.values(), address, width)
        if offset is None:
            raise Error(
                f"{self.name}: {width} bytes at {address:#x} are outside its file-backed segments"
            )
        return self.collection.bytes(offset, width)


@dataclass
class Region:
    """One dyld_chained_starts_in_segment: pages from start, each with its first entry.

    entries caches every chain entry's address once the region's chains are walked.
    """

    start: int
    page_size: int
    starts: tuple[int, ...]
    entries: set[int] | None = None

    @property
    def end(self) -> int:
        return self.start + self.page_size * len(self.starts)


class Collection:
    """An x86-64 kernel collection (MH_FILESET): its segments, entries and fixup chains."""

    def __init__(self, data: bytes) -> None:
        self.data = memoryview(data)
        magic, cpu, _, filetype, ncmds, sizeofcmds, _, _ = self.unpack("<8I", 0)
        if (magic, cpu, filetype) != (MH_MAGIC_64, CPU_TYPE_X86_64, MH_FILESET):
            raise Error("expected an x86-64 kernel collection")
        self.entries: dict[str, tuple[int, int]] = {}
        self.segments: list[Segment] = []
        fixups = []
        for command, at, length in self.commands(0, ncmds, sizeofcmds):
            if command == LC_SEGMENT_64:
                self.segments.append(self._read_segment(at, length))
            elif command == LC_DYLD_CHAINED_FIXUPS:
                if length != 16:
                    raise Error("invalid chained-fixups command")
                fixups.append(self.unpack("<II", at + 8))
            elif command == LC_FILESET_ENTRY:
                name, address, offset = self._read_fileset_entry(at, length)
                if name in self.entries:
                    raise Error(f"duplicate fileset entry {name}")
                self.entries[name] = (address, offset)
        headers = [s.address for s in self.segments if s.offset == 0 and s.filesize]
        mapped = [s.address for s in self.segments if s.vmsize]
        if len(headers) != 1 or len(fixups) != 1:
            raise Error("collection needs one header segment and one chained-fixups table")
        self.header, self.base = headers[0], min(mapped)
        self.regions = self.chained_starts(*fixups[0])

    def _read_segment(self, at: int, length: int) -> Segment:
        if length < SEGMENT_COMMAND_SIZE:
            raise Error("truncated collection segment command")
        address, vmsize, offset, filesize = self.unpack("<QQQQ", at + 24)
        if filesize > vmsize or address > (1 << 64) - 1 - vmsize:
            raise Error("invalid collection segment command")
        self.bounds(offset, filesize)
        return Segment(self.fixed(at + 8), address, vmsize, offset, filesize, {})

    def _read_fileset_entry(self, at: int, length: int) -> tuple[str, int, int]:
        """A fileset_entry_command's entry identifier, address and file offset."""
        if length < FILESET_ENTRY_COMMAND_SIZE:
            raise Error("truncated fileset entry command")
        address, offset, name_offset, _ = self.unpack("<QQII", at + 8)
        if not FILESET_ENTRY_COMMAND_SIZE <= name_offset < length:
            raise Error("fileset entry name outside its command")
        return self.string(at + name_offset, length - name_offset), address, offset

    def chained_starts(self, offset: int, size: int) -> list[Region]:
        """The regions of the chained-fixups table at offset, sorted by address.

        OpenCore appends the starts of the region it injects after the table's declared
        size, and the booter follows their offsets, so the starts are bounded by the
        file-backed segment that holds the table rather than by size.
        """
        holders = [
            s
            for s in self.segments
            if s.offset <= offset and size <= s.filesize and offset - s.offset <= s.filesize - size
        ]
        if len(holders) != 1 or size < CHAINED_FIXUPS_HEADER_SIZE:
            raise Error("chained-fixups table is not inside one collection segment")
        room = holders[0].offset + holders[0].filesize - offset
        version, starts = self.unpack("<II", offset)
        if version != 0 or not CHAINED_FIXUPS_HEADER_SIZE <= starts <= size - 4:
            raise Error("unsupported chained-fixups header")
        image, room = offset + starts, room - starts
        count = self.unpack("<I", image)[0]
        if count > (room - 4) // 4:
            raise Error("chained starts outside their segment")
        regions = sorted(
            (
                self._read_region(image + info, room - info)
                for info in self.unpack(f"<{count}I", image + 4)
                if info
            ),
            key=lambda region: region.start,
        )
        if any(a.end > b.start for a, b in pairwise(regions)):
            raise Error("chained-fixup regions overlap")
        return regions

    def _read_region(self, at: int, room: int) -> Region:
        """The dyld_chained_starts_in_segment at at, which room bytes of the file hold."""
        if room < CHAINED_STARTS_IN_SEGMENT_SIZE:
            raise Error("chained segment starts outside their segment")
        length, page_size, pointer_format, segment_offset, _, pages = self.unpack("<IHHQIH", at)
        if (
            pointer_format != DYLD_CHAINED_PTR_X86_64_KERNEL_CACHE
            or not page_size
            or length < CHAINED_STARTS_IN_SEGMENT_SIZE + 2 * pages
            or length > room
        ):
            raise Error("unsupported chained segment starts")
        starts = self.unpack(f"<{pages}H", at + CHAINED_STARTS_IN_SEGMENT_SIZE)
        if any(
            start != DYLD_CHAINED_PTR_START_NONE and start & DYLD_CHAINED_PTR_START_MULTI
            for start in starts
        ):
            raise Error("unsupported chained page start")
        return Region(self.header + segment_offset, page_size, starts)

    def read(self, address: int, width: int) -> bytes:
        offset = file_offset(self.segments, address, width)
        if offset is None:
            raise Error(
                f"{width} bytes at {address:#x} are outside the collection's file-backed segments"
            )
        return self.bytes(offset, width)

    def chain_entries(self, address: int) -> set[int]:
        """Every chain entry in the region holding address, walked whole once."""
        region = next((r for r in self.regions if r.start <= address < r.end), None)
        if region is None:
            return set()
        if region.entries is None:
            region.entries = set(self._walk_chains(region))
        return region.entries

    def _walk_chains(self, region: Region) -> Iterator[int]:
        for page, start in enumerate(region.starts):
            if start == DYLD_CHAINED_PTR_START_NONE:
                continue
            at = region.start + page * region.page_size + start
            while True:
                if at >= region.end:
                    raise Error("fixup chain leaves its region")
                yield at
                step = struct.unpack("<Q", self.read(at, 8))[0] >> NEXT_SHIFT & NEXT_MASK
                if not step:
                    break
                at += step

    def rebase(self, address: int) -> int:
        """The address a chained rebase at address holds once the booter slides it."""
        if address not in self.chain_entries(address):
            raise Error(f"pointer at {address:#x} is not in the collection's fixup chains")
        raw = struct.unpack("<Q", self.read(address, 8))[0]
        if raw & NOT_PLAIN:
            raise Error(f"pointer at {address:#x} is not a plain chained rebase")
        return self.base + (raw & TARGET_MASK)

    def overlaps_fixup(self, address: int, width: int) -> bool:
        """Whether any chain entry's eight bytes overlap the width bytes at address."""
        for at in (address - 7, address + width - 1):
            for entry in self.chain_entries(at):
                if entry < address + width and address < entry + 8:
                    return True
        return False

    def bounds(self, offset: int, size: int) -> None:
        """Raise Error unless size bytes at offset lie inside the file."""
        if offset < 0 or size < 0 or offset > len(self.data) or size > len(self.data) - offset:
            raise Error("collection range outside file")

    def bytes(self, offset: int, size: int) -> bytes:
        self.bounds(offset, size)
        return bytes(self.data[offset : offset + size])

    def unpack(self, fmt: str, offset: int) -> tuple:
        return struct.unpack(fmt, self.bytes(offset, struct.calcsize(fmt)))

    def string(self, offset: int, room: int) -> str:
        """A nonempty NUL-terminated ASCII string that ends within room bytes."""
        raw = self.bytes(offset, room)
        end = raw.find(0)
        if end <= 0:
            raise Error("unterminated or empty load-command string")
        try:
            return raw[:end].decode("ascii")
        except UnicodeDecodeError as error:
            raise Error("invalid load-command string") from error

    def fixed(self, offset: int) -> str:
        """A 16-byte NUL-padded segment or section name."""
        try:
            return self.bytes(offset, 16).split(b"\0", 1)[0].decode("ascii")
        except UnicodeDecodeError as error:
            raise Error("invalid segment or section name") from error

    def commands(self, header: int, count: int, size: int) -> Iterator[tuple[int, int, int]]:
        """Each load command after the Mach-O header at header, as (cmd, offset, cmdsize)."""
        at = header + MACH_HEADER_SIZE
        self.bounds(at, size)
        end = at + size
        for _ in range(count):
            if end - at < 8:
                raise Error("truncated load command")
            command, length = self.unpack("<II", at)
            if length < 8 or length % 8 or length > end - at:
                raise Error("invalid load command")
            yield command, at, length
            at += length
        if at != end:
            raise Error("load commands do not fill their declared size")

    def entry(self, name: str) -> Entry:
        """The fileset entry whose identifier is name, with its segments and sections."""
        if name not in self.entries:
            raise Error(f"collection has no fileset entry {name}")
        address, offset = self.entries[name]
        magic, cpu, _, filetype, ncmds, sizeofcmds, _, _ = self.unpack("<8I", offset)
        if (magic, cpu, filetype) != (MH_MAGIC_64, CPU_TYPE_X86_64, MH_KEXT_BUNDLE):
            raise Error(f"{name}: fileset entry is not an x86-64 kext")
        segments = {}
        for command, at, length in self.commands(offset, ncmds, sizeofcmds):
            if command != LC_SEGMENT_64:
                continue
            if length < SEGMENT_COMMAND_SIZE:
                raise Error(f"{name}: truncated segment command")
            segment = self.fixed(at + 8)
            seg_address, vmsize, seg_offset, filesize, _, _, nsects, _ = self.unpack(
                "<QQQQIIII", at + 24
            )
            if (
                nsects > (length - SEGMENT_COMMAND_SIZE) // SECTION_HEADER_SIZE
                or filesize > vmsize
                or segment in segments
            ):
                raise Error(f"{name}: invalid segment command {segment}")
            self.bounds(seg_offset, filesize)
            sections = {}
            for i in range(nsects):
                header = at + SEGMENT_COMMAND_SIZE + i * SECTION_HEADER_SIZE
                section = self.fixed(header)
                if section in sections:
                    raise Error(f"{name}: duplicate section {segment},{section}")
                sections[section] = self.unpack("<QQ", header + 32)
            segments[segment] = Segment(
                segment, seg_address, vmsize, seg_offset, filesize, sections
            )
        text = segments.get("__TEXT")
        if text is None or text.address != address or text.offset != offset:
            raise Error(f"{name}: fileset entry header does not start its __TEXT segment")
        return Entry(self, name, address, offset, segments)


def read_manifest(raw: bytes) -> tuple[bytes, list[SiteRecord]]:
    """The manifest's declared bytes and its sites, from a __DATA,__graft section."""
    if len(raw) < HEADER_SIZE:
        raise Error("manifest section is shorter than its header")
    header = Header.unpack_from(raw)
    if (header.magic, header.version, header.header_size) != (
        MAGIC,
        VERSION,
        HEADER_SIZE,
    ) or not HEADER_SIZE <= header.size <= len(raw):
        raise Error("section holds no staged manifest")
    offset, count = header.sites
    if offset < HEADER_SIZE or offset > header.size or count > (header.size - offset) // SITE.size:
        raise Error("manifest site table outside its declared size")
    sites = [SiteRecord(*SITE.unpack_from(raw, offset + i * SITE.size)) for i in range(count)]
    return bytes(raw[: header.size]), sites


def check(collection: Collection, identifier: str, staged: Image) -> int:
    """Check one staged consumer against its collection entry.

    Returns the number of sites checked, and raises Error on the first defect.
    """
    entry = collection.entry(identifier)
    base = check_segments(entry, staged)
    section = staged.section("__DATA", "__graft")
    raw = bytes(staged.data[section.offset : section.offset + section.size])
    declared, sites = read_manifest(raw)
    manifest = entry.address + section.address - base
    if entry.segments["__DATA"].sections.get("__graft") != (manifest, section.size):
        raise Error(f"{identifier}: __DATA,__graft moved or changed size")
    if collection.overlaps_fixup(manifest, len(declared)):
        raise Error(f"{identifier}: a fixup-chain entry overlaps the manifest")
    if entry.read(manifest, len(declared)) != declared:
        raise Error(f"{identifier}: manifest differs from the staged consumer")
    tail = range(section.address - base + len(declared), section.address - base + section.size)
    for index, site in enumerate(sites):
        check_site(collection, entry, site, tail, f"{identifier}: site {index}")
    return len(sites)


def check_segments(entry: Entry, staged: Image) -> int:
    """Require every staged segment at its staged position relative to the header.

    Returns the staged __TEXT address, from which the staged image's offsets count.
    """
    names = [segment.name for segment in staged.segments]
    if "__TEXT" not in names or len(set(names)) != len(names):
        raise Error(f"{entry.name}: staged kext needs one __TEXT and unique segments")
    base = next(segment.address for segment in staged.segments if segment.name == "__TEXT")
    for segment in staged.segments:
        if segment.name == "__LINKEDIT":
            continue
        linked = entry.segments.get(segment.name)
        if (
            linked is None
            or linked.address - entry.address != segment.address - base
            or linked.vmsize < segment.vmsize
        ):
            raise Error(
                f"{entry.name}: segment {segment.name} changed its layout relative to the header"
            )
    return base


def check_site(
    collection: Collection, entry: Entry, site: SiteRecord, tail: range, name: str
) -> None:
    """Require a site to target its placeholder, which must lie in the manifest's tail."""
    where = f"{name} at offset {site.offset:#x}"
    if site.placeholder not in tail:
        raise Error(f"{where}: placeholder is outside the reserved manifest tail")
    address = entry.address + site.offset
    expected = entry.address + site.placeholder + site.addend
    if site.width == 8 and site.opcode == POINTER:
        entry.read(address, 8)  # the site must be the consumer's own file-backed bytes
        try:
            actual = collection.rebase(address)
        except Error as error:
            raise Error(f"{where}: {error}") from error
    elif site.width == 4 and site.opcode in BRANCH_OPCODES:
        instruction = entry.read(address - 1, 5)
        if collection.overlaps_fixup(address - 1, 5):
            raise Error(f"{where}: a fixup-chain entry overlaps the branch")
        if instruction[0] != site.opcode:
            raise Error(f"{where}: branch opcode {instruction[0]:#04x} is not {site.opcode:#04x}")
        actual = address + 4 + struct.unpack("<i", instruction[1:])[0]
    else:
        raise Error(f"{where}: unsupported site form")
    if actual != expected:
        raise Error(f"{where}: targets {actual:#x}, not its placeholder {expected:#x}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("collection", type=Path)
    parser.add_argument("kexts", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        collection = Collection(args.collection.read_bytes())
        for kext in args.kexts:
            info = plistlib.loads((kext / "Contents/Info.plist").read_bytes())
            staged = Image((kext / "Contents/MacOS" / info["CFBundleExecutable"]).read_bytes())
            count = check(collection, info["CFBundleIdentifier"], staged)
            print(
                f"prelink: {info['CFBundleIdentifier']}: manifest intact, "
                f"{count} sites at their placeholders"
            )
    except (Error, OSError, ValueError, KeyError, TypeError, struct.error) as error:
        parser.exit(1, f"prelink: {error}\n")
    print(f"prelink: OK: every listed consumer is bindable in {args.collection.name}")


if __name__ == "__main__":
    main()
