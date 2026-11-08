"""The thin x86-64 Mach-O surface emitted by our consumer build.

All offsets are checked before reading or mutating. This intentionally
handles our objects and kexts, not arbitrary architectures or linkers.
"""
import struct
from dataclasses import dataclass


class Error(ValueError):
    pass


@dataclass
class Section:
    segment: str
    name: str
    address: int
    size: int
    offset: int
    relocation_offset: int
    relocation_count: int
    flags: int
    index: int
    command_offset: int


@dataclass
class Symbol:
    name: str
    kind: int
    section: int
    descriptor: int
    value: int
    index: int
    offset: int

    @property
    def defined(self):
        return not self.kind & 0xe0 and self.kind & 0x0e == 0x0e

    @property
    def undefined(self):
        return not self.kind & 0xe0 and self.kind & 0x0e == 0


@dataclass
class Relocation:
    address: int
    symbol_index: int
    pcrel: bool
    width: int
    external: bool
    kind: int
    offset: int


class Image:
    def __init__(self, data):
        self.data = bytearray(data)
        self.bounds(0, 32)
        magic, cpu, _, self.filetype, count, size, _, _ = self.unpack("<8I", 0)
        if magic != 0xfeedfacf or cpu != 0x01000007 or self.filetype not in (1, 0xb):
            raise Error("expected thin x86-64 object or kext")
        self.bounds(32, size)
        self.sections, self.segments = [], []
        self.symtab = self.dysymtab = None
        at, end = 32, 32 + size
        for _ in range(count):
            if end - at < 8:
                raise Error("truncated load command")
            kind, length = self.unpack("<II", at)
            if length < 8 or length % 8 or length > end - at:
                raise Error("invalid load command")
            if kind == 0x19:
                if length < 72:
                    raise Error("truncated segment command")
                name = self.fixed_string(at + 8, 16)
                address, vmsize, fileoff, filesize, _, _, nsects, _ = self.unpack("<QQQQIIII", at + 24)
                if nsects > (length - 72) // 80:
                    raise Error("sections exceed segment command")
                self.bounds(fileoff, filesize)
                self.segments.append((name, address, vmsize, fileoff, filesize, at))
                for i in range(nsects):
                    pos = at + 72 + i * 80
                    addr, section_size, offset, _, reloff, nreloc, flags, _, _, _ = self.unpack("<QQ8I", pos + 32)
                    section = Section(self.fixed_string(pos + 16, 16), self.fixed_string(pos, 16),
                        addr, section_size, offset, reloff, nreloc, flags, len(self.sections) + 1, pos)
                    if flags & 0xff not in (1, 0xc, 0x12):
                        self.bounds(offset, section_size)
                    self.bounds(reloff, nreloc * 8)
                    self.sections.append(section)
            elif kind == 2:
                if length != 24 or self.symtab is not None:
                    raise Error("invalid symbol-table command")
                self.symtab = (*self.unpack("<4I", at + 8), at)
            elif kind == 0xb:
                if length != 80 or self.dysymtab is not None:
                    raise Error("invalid dynamic symbol-table command")
                self.dysymtab = (list(self.unpack("<18I", at + 8)), at)
            at += length
        if at != end or self.symtab is None:
            raise Error("incomplete load commands or symbol table")
        symoff, nsyms, stroff, strsize, _ = self.symtab
        self.bounds(symoff, nsyms * 16)
        self.bounds(stroff, strsize)
        self.symbols = []
        for i in range(nsyms):
            pos = symoff + i * 16
            strx, kind, section, descriptor, value = self.unpack("<IBBHQ", pos)
            if strx >= strsize:
                raise Error("symbol name outside string table")
            name = self.string(stroff + strx, strsize - strx)
            if kind & 0x0e == 0x0e and not kind & 0xe0 and not 1 <= section <= len(self.sections):
                raise Error("defined symbol has invalid section")
            self.symbols.append(Symbol(name, kind, section, descriptor, value, i, pos))
        self.base = self.segments[0][1] if self.segments else 0

    def bounds(self, offset, size):
        if offset < 0 or size < 0 or offset > len(self.data) or size > len(self.data) - offset:
            raise Error("Mach-O range outside file")

    def unpack(self, format, offset):
        self.bounds(offset, struct.calcsize(format))
        return struct.unpack_from(format, self.data, offset)

    def fixed_string(self, offset, size):
        self.bounds(offset, size)
        try:
            return bytes(self.data[offset:offset + size]).split(b"\0", 1)[0].decode("ascii")
        except UnicodeDecodeError as error:
            raise Error("invalid Mach-O fixed string") from error

    def string(self, offset, size):
        self.bounds(offset, size)
        end = self.data.find(0, offset, offset + size)
        if end < 0:
            raise Error("unterminated Mach-O string")
        try:
            return self.data[offset:end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise Error("invalid symbol string") from error

    def section(self, segment, name):
        found = [s for s in self.sections if (s.segment, s.name) == (segment, name)]
        if len(found) != 1:
            raise Error(f"expected one {segment},{name} section")
        return found[0]

    def file_offset(self, address, width):
        for section in self.sections:
            if section.flags & 0xff in (1, 0xc, 0x12):
                continue
            if section.address <= address and width <= section.size and address - section.address <= section.size - width:
                offset = section.offset + address - section.address
                self.bounds(offset, width)
                return offset
        raise Error("address is outside file-backed sections")

    def read(self, address, width):
        offset = self.file_offset(address, width)
        return bytes(self.data[offset:offset + width])

    def symbol(self, name):
        found = [s for s in self.symbols if s.name == name and s.defined]
        if len(found) != 1:
            raise Error(f"expected one definition of {name}")
        return found[0]

    def symbol_extent(self, symbol):
        section = self.sections[symbol.section - 1]
        end = min((s.value for s in self.symbols if s.defined and s.section == symbol.section and s.value > symbol.value),
            default=section.address + section.size)
        if symbol.value < section.address or end > section.address + section.size:
            raise Error("symbol outside owning section")
        return end - symbol.value

    def relocation_range(self, offset, count, address_base):
        self.bounds(offset, count * 8)
        for i in range(count):
            pos = offset + i * 8
            address, packed = self.unpack("<iI", pos)
            if address < 0:
                raise Error("scattered relocation is unsupported")
            symbol = packed & 0xffffff
            external = bool(packed >> 27 & 1)
            if external and symbol >= len(self.symbols):
                raise Error("relocation symbol index outside table")
            yield Relocation(address_base + address, symbol, bool(packed >> 24 & 1),
                1 << (packed >> 25 & 3), external, packed >> 28, pos)

    def relocations(self):
        if self.filetype == 1:
            return [r for section in self.sections for r in
                self.relocation_range(section.relocation_offset, section.relocation_count, section.address)]
        if self.dysymtab is None:
            raise Error("kext has no dynamic symbol table")
        fields, _ = self.dysymtab
        return list(self.relocation_range(fields[14], fields[15], self.base)) + \
            list(self.relocation_range(fields[16], fields[17], self.base))

    def localize(self, definitions):
        """Make selected undefined symbols private local section definitions.

        definitions maps symbol name to (section index, image address).
        Reorder symbol groups and repair every supported symbol index.
        """
        if self.filetype != 0xb or self.dysymtab is None:
            raise Error("localization requires a linked kext")
        fields, command = self.dysymtab
        # Module/TOC/reference tables belong to dylibs, never our kexts.
        if any(fields[i] for i in (7, 9, 11)):
            raise Error("unsupported dynamic symbol metadata")
        missing = set(definitions)
        records = []
        for symbol in self.symbols:
            row = bytes(self.data[symbol.offset:symbol.offset + 16])
            if symbol.name in definitions:
                if not symbol.undefined:
                    raise Error("placeholder names an already defined symbol")
                section, value = definitions[symbol.name]
                if not 1 <= section <= len(self.sections):
                    raise Error("placeholder section outside image")
                strx = struct.unpack_from("<I", row)[0]
                row = struct.pack("<IBBHQ", strx, 0x0e, section, 0, value)
                missing.discard(symbol.name)
                group = 0
            else:
                group = 2 if symbol.undefined else (1 if symbol.kind & 1 else 0)
            records.append((group, symbol.index, row))
        if missing:
            raise Error("placeholder has no undefined symbol")
        records.sort(key=lambda row: (row[0], row[1]))
        mapping = {old: new for new, (_, old, _) in enumerate(records)}
        for relocation in self.relocations():
            if relocation.external:
                packed = self.unpack("<I", relocation.offset + 4)[0]
                struct.pack_into("<I", self.data, relocation.offset + 4,
                    (packed & 0xff000000) | mapping[relocation.symbol_index])
        indirect, count = fields[12:14]
        self.bounds(indirect, count * 4)
        for i in range(count):
            index = self.unpack("<I", indirect + i * 4)[0]
            if not index & 0xc0000000:
                if index >= len(self.symbols):
                    raise Error("indirect symbol index outside table")
                struct.pack_into("<I", self.data, indirect + i * 4, mapping[index])
        symoff = self.symtab[0]
        for i, (_, _, row) in enumerate(records):
            self.data[symoff + i * 16:symoff + (i + 1) * 16] = row
        local = sum(group == 0 for group, _, _ in records)
        external = sum(group == 1 for group, _, _ in records)
        undefined = len(records) - local - external
        fields[:6] = [0, local, local, external, local + external, undefined]
        struct.pack_into("<18I", self.data, command + 8, *fields)
        return Image(self.data)

    def rename_symbols(self, names):
        """Rename private compiler markers without changing layout.

        Reject shared string storage rather than changing an unrelated
        symbol. Every replacement has exactly the original byte length.
        """
        stroff = self.symtab[2]
        for old, new in names.items():
            encoded = new.encode()
            if len(encoded) != len(old.encode()) or b"\0" in encoded:
                raise Error("marker rename must preserve its byte length")
            matches = [s for s in self.symbols if s.name == old]
            if not matches:
                raise Error("marker rename names no symbol")
            for symbol in matches:
                start = stroff + self.unpack("<I", symbol.offset)[0]
                end = start + len(encoded)
                for other in self.symbols:
                    if other.name == old:
                        continue
                    other_start = stroff + self.unpack("<I", other.offset)[0]
                    other_end = other_start + len(other.name.encode())
                    if start < other_end and other_start < end:
                        raise Error("marker string shares storage with another symbol")
                self.data[start:end] = encoded
        return Image(self.data)
