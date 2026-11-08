#!/usr/bin/env python3
"""Check that a prelinked kernel collection keeps staged consumers bindable.

  prelink.py COLLECTION KEXT...

COLLECTION is an x86-64 kernel collection after OpenCore prelinked our kexts
into it. Each KEXT is a staged consumer bundle (consumer.py stage). Each
consumer's fileset entry is found by its bundle identifier, and the check
requires what Graft's runtime relies on when it binds that entry:

- the entry's header starts its __TEXT, and every staged segment keeps its
  position relative to the header, because the runtime adds manifest
  offsets to the entry address;
- __DATA,__graft holds the staged manifest unchanged through the size the
  manifest declares; the placeholder tail after it is not compared;
- every manifest site holds its exact local placeholder plus addend, with
  each placeholder inside the reserved tail. A pointer site must be an
  entry of the collection's fixup chains, since the booter rewrites only
  those when it slides the collection, and its decoded target must be that
  address. A branch site's E8/E9 displacement must reach it;
- no fixup-chain entry overlaps a branch site or the manifest, whose bytes
  the booter would otherwise rewrite.

x86-64 collections encode every pointer as a chained rebase
(DYLD_CHAINED_PTR_X86_64_KERNEL_CACHE). A slot's segment_offset counts from
the collection header, and a target counts from the collection's lowest
segment address. Slots are matched by segment_offset, not by load-command
order, since OpenCore reuses a slot for the region it injects. Chains may
run past their starting page, so a region's chains are walked whole.

The collection is only read, and every read is bounded by its file.
"""
import argparse
from pathlib import Path
import plistlib
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from consumer import HEADER_SIZE, MAGIC, VERSION
from graft_macho import Error, Image

MH_MAGIC_64, CPU_X86_64, MH_KEXT_BUNDLE, MH_FILESET = 0xfeedfacf, 0x01000007, 0xb, 0xc
LC_SEGMENT_64, LC_DYLD_CHAINED_FIXUPS, LC_FILESET_ENTRY = 0x19, 0x80000034, 0x80000035
CHAINED_PTR_X86_64_KERNEL_CACHE, PAGE_NONE, PAGE_MULTI = 11, 0xffff, 0x8000
SITE = struct.Struct("<QQqIBBH")
# target:30 cacheLevel:2 diversity:16 addrDiv:1 key:2 next:12 isAuth:1. The
# check accepts only plain rebases into this collection: every field but
# target and next must be zero.
TARGET, NEXT_SHIFT, NEXT_MASK = (1 << 30) - 1, 51, 0xfff
PLAIN = ~(TARGET | NEXT_MASK << NEXT_SHIFT) & ((1 << 64) - 1)


class Segment:
    def __init__(self, name, address, vmsize, offset, filesize, sections):
        self.name, self.address, self.vmsize = name, address, vmsize
        self.offset, self.filesize, self.sections = offset, filesize, sections


class Entry:
    """One fileset entry's kext. Its file offsets are collection-absolute."""

    def __init__(self, collection, name, address, offset, segments):
        self.collection, self.name, self.address = collection, name, address
        self.offset, self.segments = offset, segments

    def read(self, address, width):
        for segment in self.segments.values():
            if segment.address <= address and width <= segment.filesize and \
               address - segment.address <= segment.filesize - width:
                return self.collection.bytes(segment.offset + address - segment.address, width)
        raise Error(f"{self.name}: {width} bytes at {address:#x} are outside its file-backed segments")


class Region:
    """One chained-starts slot: the pages from start, each with a chain start."""

    def __init__(self, start, page_size, starts):
        self.start, self.page_size, self.starts = start, page_size, starts
        self.end = start + page_size * len(starts)
        self.entries = None


class Collection:
    def __init__(self, data):
        self.data = memoryview(data)
        magic, cpu, _, filetype, count, size, _, _ = self.unpack("<8I", 0)
        if (magic, cpu, filetype) != (MH_MAGIC_64, CPU_X86_64, MH_FILESET):
            raise Error("expected an x86-64 kernel collection")
        self.entries, self.segments, fixups = {}, [], []
        for kind, at, length in self.commands(0, count, size):
            if kind == LC_SEGMENT_64:
                if length < 72:
                    raise Error("truncated collection segment command")
                address, vmsize, offset, filesize = self.unpack("<QQQQ", at + 24)
                if filesize > vmsize or address > (1 << 64) - 1 - vmsize:
                    raise Error("invalid collection segment command")
                self.bounds(offset, filesize)
                self.segments.append(Segment(self.fixed(at + 8), address, vmsize, offset, filesize, {}))
            elif kind == LC_DYLD_CHAINED_FIXUPS:
                if length != 16:
                    raise Error("invalid chained-fixups command")
                fixups.append(self.unpack("<II", at + 8))
            elif kind == LC_FILESET_ENTRY:
                if length < 32:
                    raise Error("truncated fileset entry command")
                address, offset, name_offset, _ = self.unpack("<QQII", at + 8)
                if not 32 <= name_offset < length:
                    raise Error("fileset entry name outside its command")
                name = self.string(at + name_offset, length - name_offset)
                if name in self.entries:
                    raise Error(f"duplicate fileset entry {name}")
                self.entries[name] = (address, offset)
        headers = [s.address for s in self.segments if s.offset == 0 and s.filesize]
        mapped = [s.address for s in self.segments if s.vmsize]
        if len(headers) != 1 or len(fixups) != 1:
            raise Error("collection needs one header segment and one chained-fixups table")
        self.header, self.base = headers[0], min(mapped)
        self.regions = self.chained_starts(*fixups[0])

    def chained_starts(self, offset, size):
        # OpenCore appends the starts of the region it injects after the
        # table's declared size, and the booter follows their offsets. So the
        # starts are bounded by the file-backed segment that holds the table.
        holders = [s for s in self.segments if s.offset <= offset and size <= s.filesize and
                   offset - s.offset <= s.filesize - size]
        if len(holders) != 1 or size < 28:
            raise Error("chained-fixups table is not inside one collection segment")
        room = holders[0].offset + holders[0].filesize - offset
        version, starts = self.unpack("<II", offset)
        if version != 0 or not 28 <= starts <= size - 4:
            raise Error("unsupported chained-fixups header")
        image, room = offset + starts, room - starts
        count = self.unpack("<I", image)[0]
        if count > (room - 4) // 4:
            raise Error("chained starts outside their segment")
        regions = []
        for info in self.unpack(f"<{count}I", image + 4):
            if not info:
                continue
            if info > room - 22:
                raise Error("chained segment starts outside their segment")
            at = image + info
            length, page_size, pointer_format, segment_offset, _, pages = self.unpack("<IHHQIH", at)
            if pointer_format != CHAINED_PTR_X86_64_KERNEL_CACHE or not page_size or \
               length < 22 + 2 * pages or length > room - info:
                raise Error("unsupported chained segment starts")
            region = Region(self.header + segment_offset, page_size, self.unpack(f"<{pages}H", at + 22))
            if any(start != PAGE_NONE and start & PAGE_MULTI for start in region.starts):
                raise Error("unsupported chained page start")
            regions.append(region)
        regions.sort(key=lambda region: region.start)
        if any(a.end > b.start for a, b in zip(regions, regions[1:])):
            raise Error("chained-fixup regions overlap")
        return regions

    def read(self, address, width):
        for segment in self.segments:
            if segment.address <= address and width <= segment.filesize and \
               address - segment.address <= segment.filesize - width:
                return self.bytes(segment.offset + address - segment.address, width)
        raise Error(f"{width} bytes at {address:#x} are outside the collection's file-backed segments")

    def chain_entries(self, address):
        """Every chain entry in the region holding address, walked whole once."""
        region = next((r for r in self.regions if r.start <= address < r.end), None)
        if region is None:
            return set()
        if region.entries is None:
            entries = set()
            for page, start in enumerate(region.starts):
                if start == PAGE_NONE:
                    continue
                at = region.start + page * region.page_size + start
                while True:
                    if at >= region.end:
                        raise Error("fixup chain leaves its region")
                    entries.add(at)
                    step = struct.unpack("<Q", self.read(at, 8))[0] >> NEXT_SHIFT & NEXT_MASK
                    if not step:
                        break
                    at += step
            region.entries = entries
        return region.entries

    def rebase(self, address):
        """The address a chained rebase at address holds once the booter slides it."""
        if address not in self.chain_entries(address):
            raise Error(f"pointer at {address:#x} is not in the collection's fixup chains")
        raw = struct.unpack("<Q", self.read(address, 8))[0]
        if raw & PLAIN:
            raise Error(f"pointer at {address:#x} is not a plain chained rebase")
        return self.base + (raw & TARGET)

    def unfixed(self, address, width):
        """Refuse if any chained rebase overlaps address..address+width."""
        for at in (address - 7, address + width - 1):
            for entry in self.chain_entries(at):
                if entry < address + width and address < entry + 8:
                    return False
        return True

    def bounds(self, offset, size):
        if offset < 0 or size < 0 or offset > len(self.data) or size > len(self.data) - offset:
            raise Error("collection range outside file")

    def bytes(self, offset, size):
        self.bounds(offset, size)
        return bytes(self.data[offset:offset + size])

    def unpack(self, format, offset):
        return struct.unpack(format, self.bytes(offset, struct.calcsize(format)))

    def string(self, offset, room):
        raw = self.bytes(offset, room)
        end = raw.find(0)
        if end <= 0:
            raise Error("unterminated or empty load-command string")
        try:
            return raw[:end].decode("ascii")
        except UnicodeDecodeError as error:
            raise Error("invalid load-command string") from error

    def fixed(self, offset):
        try:
            return self.bytes(offset, 16).split(b"\0", 1)[0].decode("ascii")
        except UnicodeDecodeError as error:
            raise Error("invalid segment or section name") from error

    def commands(self, header, count, size):
        at = header + 32
        self.bounds(at, size)
        end = at + size
        for _ in range(count):
            if end - at < 8:
                raise Error("truncated load command")
            kind, length = self.unpack("<II", at)
            if length < 8 or length % 8 or length > end - at:
                raise Error("invalid load command")
            yield kind, at, length
            at += length
        if at != end:
            raise Error("load commands do not fill their declared size")

    def entry(self, name):
        if name not in self.entries:
            raise Error(f"collection has no fileset entry {name}")
        address, offset = self.entries[name]
        magic, cpu, _, filetype, count, size, _, _ = self.unpack("<8I", offset)
        if (magic, cpu, filetype) != (MH_MAGIC_64, CPU_X86_64, MH_KEXT_BUNDLE):
            raise Error(f"{name}: fileset entry is not an x86-64 kext")
        segments = {}
        for kind, at, length in self.commands(offset, count, size):
            if kind != LC_SEGMENT_64:
                continue
            if length < 72:
                raise Error(f"{name}: truncated segment command")
            segment = self.fixed(at + 8)
            seg_address, vmsize, seg_offset, filesize, _, _, nsects, _ = self.unpack("<QQQQIIII", at + 24)
            if nsects > (length - 72) // 80 or filesize > vmsize or segment in segments:
                raise Error(f"{name}: invalid segment command {segment}")
            self.bounds(seg_offset, filesize)
            sections = {}
            for i in range(nsects):
                pos = at + 72 + i * 80
                section = self.fixed(pos)
                if section in sections:
                    raise Error(f"{name}: duplicate section {segment},{section}")
                sections[section] = self.unpack("<QQ", pos + 32)
            segments[segment] = Segment(segment, seg_address, vmsize, seg_offset, filesize, sections)
        text = segments.get("__TEXT")
        if text is None or text.address != address or text.offset != offset:
            raise Error(f"{name}: fileset entry header does not start its __TEXT segment")
        return Entry(self, name, address, offset, segments)


def manifest(raw):
    """Return the declared manifest bytes and its site rows, bounded by raw."""
    if len(raw) < HEADER_SIZE:
        raise Error("manifest section is shorter than its header")
    fields = struct.unpack_from("<24I", raw)
    magic, version, size, header_size = fields[:4]
    if (magic, version, header_size) != (MAGIC, VERSION, HEADER_SIZE) or not HEADER_SIZE <= size <= len(raw):
        raise Error("section holds no staged manifest")
    offset, count = fields[12:14]
    if offset < HEADER_SIZE or offset > size or count > (size - offset) // SITE.size:
        raise Error("manifest site table outside its declared size")
    return bytes(raw[:size]), [SITE.unpack_from(raw, offset + i * SITE.size) for i in range(count)]


def check(collection, identifier, staged):
    """Check one staged consumer Image against its collection entry.

    Return the number of sites verified; raise Error on the first defect.
    """
    entry = collection.entry(identifier)
    names = [segment[0] for segment in staged.segments]
    if "__TEXT" not in names or len(set(names)) != len(names):
        raise Error(f"{identifier}: staged kext needs one __TEXT and unique segments")
    base = next(segment[1] for segment in staged.segments if segment[0] == "__TEXT")
    for name, address, vmsize, _, _, _ in staged.segments:
        if name == "__LINKEDIT":
            continue
        found = entry.segments.get(name)
        if found is None or found.address - entry.address != address - base or found.vmsize < vmsize:
            raise Error(f"{identifier}: segment {name} changed its layout relative to the header")
    section = staged.section("__DATA", "__graft")
    raw = bytes(staged.data[section.offset:section.offset + section.size])
    declared, sites = manifest(raw)
    linked = entry.segments["__DATA"].sections.get("__graft")
    if linked != (entry.address + section.address - base, section.size):
        raise Error(f"{identifier}: __DATA,__graft moved or changed size")
    graft = entry.address + section.address - base
    if not collection.unfixed(graft, len(declared)):
        raise Error(f"{identifier}: a fixup-chain entry overlaps the manifest")
    if entry.read(graft, len(declared)) != declared:
        raise Error(f"{identifier}: manifest differs from the staged consumer")
    tail_start, tail_end = section.address - base + len(declared), section.address - base + section.size
    for index, (offset, placeholder, addend, _, width, opcode, _) in enumerate(sites):
        where = f"{identifier}: site {index} at offset {offset:#x}"
        if not tail_start <= placeholder < tail_end:
            raise Error(f"{where}: placeholder is outside the reserved manifest tail")
        site, expected = entry.address + offset, entry.address + placeholder + addend
        if width == 8 and opcode == 0:
            entry.read(site, 8)  # the site must be the consumer's own file-backed bytes
            try:
                actual = collection.rebase(site)
            except Error as error:
                raise Error(f"{where}: {error}") from error
        elif width == 4 and opcode in (0xe8, 0xe9):
            prefix = entry.read(site - 1, 5)
            if not collection.unfixed(site - 1, 5):
                raise Error(f"{where}: a fixup-chain entry overlaps the branch")
            if prefix[0] != opcode:
                raise Error(f"{where}: branch opcode {prefix[0]:#04x} is not {opcode:#04x}")
            actual = site + 4 + struct.unpack("<i", prefix[1:])[0]
        else:
            raise Error(f"{where}: unsupported site form")
        if actual != expected:
            raise Error(f"{where}: targets {actual:#x}, not its placeholder {expected:#x}")
    return len(sites)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("collection", type=Path)
    parser.add_argument("kexts", type=Path, nargs="+")
    args = parser.parse_args()
    try:
        collection = Collection(args.collection.read_bytes())
        for kext in args.kexts:
            info = plistlib.loads((kext / "Contents/Info.plist").read_bytes())
            staged = Image((kext / "Contents/MacOS" / info["CFBundleExecutable"]).read_bytes())
            count = check(collection, info["CFBundleIdentifier"], staged)
            print(f"prelink: {info['CFBundleIdentifier']}: manifest intact, {count} sites at their placeholders")
    except (Error, OSError, ValueError, KeyError, TypeError, struct.error) as error:
        parser.exit(1, f"prelink: {error}\n")
    print(f"prelink: OK: every listed consumer is bindable in {args.collection.name}")


if __name__ == "__main__":
    main()
