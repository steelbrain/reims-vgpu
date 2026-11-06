"""The parts of a 64-bit little-endian Mach-O file the loader's tools read,
with the standard library alone: fat slices, segments and sections, the
symbol table and LC_DYSYMTAB's external and local relocations, as ld writes
them for an x86_64 MH_KEXT_BUNDLE. Ours, written for these tools.

Inputs include the KDK's binaries, untrusted: every read is bounds-checked,
and nothing is run. Run the tools that import it with `python3 -I`.
"""

import struct
from dataclasses import dataclass, field

MH_MAGIC_64 = 0xFEEDFACF
FAT_MAGIC, FAT_MAGIC_64 = 0xCAFEBABE, 0xCAFEBABF
CPU_TYPE_X86_64 = 0x01000007
LC_SEGMENT_64, LC_SYMTAB, LC_DYSYMTAB = 0x19, 0x2, 0xB
N_STAB, N_TYPE, N_EXT = 0xE0, 0x0E, 0x01
N_UNDF, N_SECT = 0x0, 0xE


class MachOError(ValueError):
    pass


@dataclass
class Section:
    segname: str
    sectname: str
    addr: int
    size: int
    flags: int
    index: int  # 1-based, as nlist's n_sect numbers it

    def contains(self, addr):
        return self.addr <= addr < self.addr + self.size


@dataclass
class Segment:
    name: str
    vmaddr: int
    vmsize: int
    fileoff: int
    filesize: int
    sections: list = field(default_factory=list)


@dataclass(frozen=True)
class Symbol:
    name: str
    n_type: int
    n_sect: int
    value: int

    @property
    def is_external(self):
        return bool(self.n_type & N_EXT)

    @property
    def is_defined(self):
        """Defined in a section of this image."""
        return not self.n_type & N_STAB and (self.n_type & N_TYPE) == N_SECT

    @property
    def is_undefined(self):
        return not self.n_type & N_STAB and (self.n_type & N_TYPE) == N_UNDF


@dataclass(frozen=True)
class Relocation:
    """A relocation entry of LC_DYSYMTAB: `address` is the field's address in
    the image (r_address plus the first segment's vmaddr, as ld writes a
    kext's), `symbol` the name an external entry binds, None for a local
    one (the target is in the field)."""
    address: int
    type: int
    length: int  # log2 of the field's width
    pcrel: bool
    symbol: str = None


def _cstr(data, off, limit):
    end = data.find(b"\0", off, off + limit)
    return data[off:end if end >= 0 else off + limit].decode("utf-8", "replace")


class Slice:
    """One architecture's image, at `base` in the file's bytes."""

    def __init__(self, data, base, size):
        if base + 32 > len(data) or base + size > len(data):
            raise MachOError("truncated Mach-O")
        magic, self.cputype, _, self.filetype, ncmds, sizeofcmds, _, _ = \
            struct.unpack_from("<IIIIIIII", data, base)
        if magic != MH_MAGIC_64:
            raise MachOError(f"not a 64-bit little-endian Mach-O (magic {magic:#x})")
        self.data, self.base, self.size = data, base, size
        self.segments, self.symtab, self.dysymtab = [], None, None
        pos, end, index = base + 32, base + 32 + sizeofcmds, 0
        if end > base + size:
            raise MachOError("load commands run past the image")
        for _ in range(ncmds):
            if pos + 8 > end:
                raise MachOError("load commands run past their size")
            cmd, cmdsize = struct.unpack_from("<II", data, pos)
            if cmdsize < 8 or pos + cmdsize > end:
                raise MachOError(f"bad load command size {cmdsize} at {pos:#x}")
            if cmd == LC_SEGMENT_64:
                vmaddr, vmsize, fileoff, filesize = struct.unpack_from("<QQQQ", data, pos + 24)
                nsects = struct.unpack_from("<I", data, pos + 64)[0]
                if 72 + 80 * nsects > cmdsize:
                    raise MachOError("segment's sections run past its command")
                seg = Segment(_cstr(data, pos + 8, 16), vmaddr, vmsize, fileoff, filesize)
                for s in range(nsects):
                    at = pos + 72 + 80 * s
                    addr, sect_size = struct.unpack_from("<QQ", data, at + 32)
                    flags = struct.unpack_from("<I", data, at + 64)[0]
                    index += 1
                    seg.sections.append(Section(_cstr(data, at + 16, 16), _cstr(data, at, 16),
                                                addr, sect_size, flags, index))
                self.segments.append(seg)
            elif cmd == LC_SYMTAB:
                self.symtab = struct.unpack_from("<IIII", data, pos + 8)
            elif cmd == LC_DYSYMTAB:
                v = struct.unpack_from("<18I", data, pos + 8)
                self.dysymtab = {"extreloff": v[14], "nextrel": v[15],
                                 "locreloff": v[16], "nlocrel": v[17]}
            pos += cmdsize
        self._symbols = None

    @property
    def arch(self):
        return "x86_64" if self.cputype == CPU_TYPE_X86_64 else f"cputype {self.cputype:#x}"

    def all_sections(self):
        return [s for seg in self.segments for s in seg.sections]

    def section(self, segname, sectname):
        return next((s for s in self.all_sections()
                     if s.segname == segname and s.sectname == sectname), None)

    def section_at(self, addr):
        return next((s for s in self.all_sections() if s.contains(addr)), None)

    def _offset(self, addr):
        for seg in self.segments:
            if seg.filesize and seg.vmaddr <= addr < seg.vmaddr + seg.filesize:
                return self.base + seg.fileoff + (addr - seg.vmaddr)
        return None

    def read(self, addr, size):
        off = self._offset(addr)
        if off is None or off + size > self.base + self.size:
            raise MachOError(f"{addr:#x}+{size:#x} is not in the file")
        return bytes(self.data[off:off + size])

    def cstring(self, addr, limit=4096):
        off = self._offset(addr)
        if off is None:
            return None
        end = self.data.find(b"\0", off, min(off + limit, self.base + self.size))
        return None if end < 0 else self.data[off:end].decode("utf-8", "replace")

    def symbols(self):
        """Every nlist_64 entry, in table order."""
        if self._symbols is None:
            out = []
            if self.symtab:
                symoff, nsyms, stroff, strsize = self.symtab
                limit = self.base + self.size
                if self.base + symoff + 16 * nsyms > limit or self.base + stroff + strsize > limit:
                    raise MachOError("symbol table runs past the image")
                for i in range(nsyms):
                    strx, n_type, n_sect, _, value = struct.unpack_from(
                        "<IBBHQ", self.data, self.base + symoff + 16 * i)
                    if strx >= strsize:
                        raise MachOError(f"symbol {i}'s name is past the string table")
                    name = _cstr(self.data, self.base + stroff + strx, strsize - strx)
                    out.append(Symbol(name, n_type, n_sect, value))
            self._symbols = out
        return self._symbols

    def relocations(self):
        """LC_DYSYMTAB's external relocations, then its local ones."""
        if not self.dysymtab:
            return []
        syms = self.symbols()
        first = self.segments[0].vmaddr if self.segments else 0
        out = []
        for off_key, n_key in (("extreloff", "nextrel"), ("locreloff", "nlocrel")):
            off, n = self.dysymtab[off_key], self.dysymtab[n_key]
            if n and self.base + off + 8 * n > self.base + self.size:
                raise MachOError("relocations run past the image")
            for i in range(n):
                r_address, packed = struct.unpack_from("<iI", self.data, self.base + off + 8 * i)
                if r_address < 0:
                    raise MachOError("scattered relocation in a 64-bit image")
                extern = bool(packed >> 27 & 1)
                symbolnum = packed & 0xFFFFFF
                if extern and symbolnum >= len(syms):
                    raise MachOError(f"relocation {i} names symbol {symbolnum}, past the table")
                out.append(Relocation(first + r_address, packed >> 28 & 0xF, packed >> 25 & 3,
                                      bool(packed >> 24 & 1),
                                      syms[symbolnum].name if extern else None))
        return out


def slices(data):
    """Every architecture's slice of a thin or fat Mach-O."""
    if len(data) < 8:
        raise MachOError("file too short")
    magic = struct.unpack_from(">I", data, 0)[0]
    if magic not in (FAT_MAGIC, FAT_MAGIC_64):
        return [Slice(data, 0, len(data))]
    nfat = struct.unpack_from(">I", data, 4)[0]
    out, pos = [], 8
    for _ in range(nfat):
        if magic == FAT_MAGIC:
            _, _, offset, size, _ = struct.unpack_from(">iiIII", data, pos)
            pos += 20
        else:
            _, _, offset, size, _, _ = struct.unpack_from(">iiQQII", data, pos)
            pos += 32
        out.append(Slice(data, offset, size))
    return out


def x86_64(path):
    """The x86_64 slice of the file at `path`; exactly one, or refused."""
    with open(path, "rb") as f:
        found = [s for s in slices(f.read()) if s.cputype == CPU_TYPE_X86_64]
    if len(found) != 1:
        raise SystemExit(f"{path}: {len(found)} x86_64 slices, not one")
    return found[0]
