# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
"""Read and edit the thin x86-64 Mach-O files of a consumer's build.

consumer.py reads compiler objects and linked kexts through Image, and stages a kext by
editing its symbol table in place (Image.localize, Image.rename_symbols). prelink.py reads
staged kexts. Every offset is checked against the file before it is read or written.
The module handles what Xcode's clang and ld emit for our kexts, not arbitrary
architectures or linkers, and raises Error for anything else rather than guessing.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple

# <mach-o/loader.h>
MH_MAGIC_64 = 0xFEEDFACF
CPU_TYPE_X86_64 = 0x01000007
MH_OBJECT = 0x1
MH_KEXT_BUNDLE = 0xB
MACH_HEADER_SIZE = 32
LC_SYMTAB = 0x2
LC_DYSYMTAB = 0xB
LC_SEGMENT_64 = 0x19
SEGMENT_COMMAND_SIZE = 72
SECTION_HEADER_SIZE = 80
# A section's type is its flags' low byte. These types have no bytes in the file.
SECTION_TYPE = 0xFF
ZEROFILL_TYPES = (0x1, 0xC, 0x12)  # S_ZEROFILL, S_GB_ZEROFILL, S_THREAD_LOCAL_ZEROFILL
# S_ATTR_PURE_INSTRUCTIONS or S_ATTR_SOME_INSTRUCTIONS: a section holding code.
INSTRUCTION_ATTRIBUTES = 0x80000000 | 0x400

# <mach-o/nlist.h>
NLIST_SIZE = 16
N_STAB = 0xE0
N_TYPE = 0x0E
N_UNDF = 0x0
N_SECT = 0x0E
N_EXT = 0x01
# An undefined symbol's n_desc bit for a weak reference (__attribute__((weak_import))).
N_WEAK_REF = 0x0040

# <mach-o/reloc.h> and <mach-o/x86_64/reloc.h>
RELOCATION_SIZE = 8
X86_64_RELOC_UNSIGNED = 0
X86_64_RELOC_SIGNED = 1
X86_64_RELOC_BRANCH = 2
# Indirect-symbol entries that name no symbol.
INDIRECT_SYMBOL_LOCAL = 0x80000000
INDIRECT_SYMBOL_ABS = 0x40000000

# LC_DYSYMTAB's symbol groups, in the order the symbol table must hold them.
LOCAL, EXTERNAL, UNDEFINED = 0, 1, 2


class Error(ValueError):
    """An input the consumer tools refuse: a Mach-O file, declaration or collection."""


class Segment(NamedTuple):
    """An LC_SEGMENT_64 command. command_offset is where the command sits in the file."""

    name: str
    address: int
    vmsize: int
    offset: int
    filesize: int
    command_offset: int


@dataclass
class Section:
    """A section header. index counts from 1, as nlist's n_sect does."""

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

    @property
    def has_file_bytes(self) -> bool:
        return self.flags & SECTION_TYPE not in ZEROFILL_TYPES

    @property
    def executable(self) -> bool:
        return bool(self.flags & INSTRUCTION_ATTRIBUTES)


@dataclass
class Symbol:
    """An nlist_64 entry. index is its position in the symbol table, entry_offset its file
    offset, and name_offset its name's offset in the string table (n_strx)."""

    name: str
    kind: int
    section: int
    descriptor: int
    value: int
    index: int
    entry_offset: int
    name_offset: int

    @property
    def defined(self) -> bool:
        """Defined in a section of this image (N_SECT), and not a debugging entry."""
        return not self.kind & N_STAB and self.kind & N_TYPE == N_SECT

    @property
    def undefined(self) -> bool:
        return not self.kind & N_STAB and self.kind & N_TYPE == N_UNDF


@dataclass
class Relocation:
    """A relocation_info entry, with its address resolved to an image address.

    kind is the x86-64 relocation type (X86_64_RELOC_*), width the bytes it rewrites, and
    entry_offset the entry's own file offset.
    """

    address: int
    symbol_index: int
    pcrel: bool
    width: int
    external: bool
    kind: int
    entry_offset: int


class SymbolTable(NamedTuple):
    """LC_SYMTAB's fields after cmd and cmdsize."""

    symoff: int
    nsyms: int
    stroff: int
    strsize: int


class DynamicSymbolTable(NamedTuple):
    """LC_DYSYMTAB's fields after cmd and cmdsize (<mach-o/loader.h>, dysymtab_command)."""

    ilocalsym: int
    nlocalsym: int
    iextdefsym: int
    nextdefsym: int
    iundefsym: int
    nundefsym: int
    tocoff: int
    ntoc: int
    modtaboff: int
    nmodtab: int
    extrefsymoff: int
    nextrefsyms: int
    indirectsymoff: int
    nindirectsyms: int
    extreloff: int
    nextrel: int
    locreloff: int
    nlocrel: int


class Image:
    """A thin x86-64 MH_OBJECT or MH_KEXT_BUNDLE, parsed from a copy of its bytes.

    base is the first segment's address, from which the manifest's image offsets count.
    """

    def __init__(self, data: bytes | bytearray) -> None:
        self.data = bytearray(data)
        self.bounds(0, MACH_HEADER_SIZE)
        magic, cpu, _, self.filetype, ncmds, sizeofcmds, _, _ = self.unpack("<8I", 0)
        if (
            magic != MH_MAGIC_64
            or cpu != CPU_TYPE_X86_64
            or self.filetype not in (MH_OBJECT, MH_KEXT_BUNDLE)
        ):
            raise Error("expected thin x86-64 object or kext")
        self.segments: list[Segment] = []
        self.sections: list[Section] = []
        self.symtab: SymbolTable | None = None
        self.dysymtab: DynamicSymbolTable | None = None
        self.dysymtab_offset = 0
        self._read_load_commands(ncmds, sizeofcmds)
        self.symbols = self._read_symbols()
        self.base = self.segments[0].address if self.segments else 0

    def _read_load_commands(self, count: int, size: int) -> None:
        self.bounds(MACH_HEADER_SIZE, size)
        at, end = MACH_HEADER_SIZE, MACH_HEADER_SIZE + size
        for _ in range(count):
            if end - at < 8:
                raise Error("truncated load command")
            command, length = self.unpack("<II", at)
            if length < 8 or length % 8 or length > end - at:
                raise Error("invalid load command")
            if command == LC_SEGMENT_64:
                self._read_segment(at, length)
            elif command == LC_SYMTAB:
                if length != 24 or self.symtab is not None:
                    raise Error("invalid symbol-table command")
                self.symtab = SymbolTable(*self.unpack("<4I", at + 8))
            elif command == LC_DYSYMTAB:
                if length != 80 or self.dysymtab is not None:
                    raise Error("invalid dynamic symbol-table command")
                self.dysymtab = DynamicSymbolTable(*self.unpack("<18I", at + 8))
                self.dysymtab_offset = at
            at += length
        if at != end or self.symtab is None:
            raise Error("incomplete load commands or symbol table")

    def _read_segment(self, at: int, length: int) -> None:
        if length < SEGMENT_COMMAND_SIZE:
            raise Error("truncated segment command")
        name = self.fixed_string(at + 8, 16)
        address, vmsize, offset, filesize, _, _, nsects, _ = self.unpack("<QQQQIIII", at + 24)
        if nsects > (length - SEGMENT_COMMAND_SIZE) // SECTION_HEADER_SIZE:
            raise Error("sections exceed segment command")
        self.bounds(offset, filesize)
        self.segments.append(Segment(name, address, vmsize, offset, filesize, at))
        for i in range(nsects):
            self._read_section(at + SEGMENT_COMMAND_SIZE + i * SECTION_HEADER_SIZE)

    def _read_section(self, at: int) -> None:
        address, size, offset, _, reloff, nreloc, flags, _, _, _ = self.unpack("<QQ8I", at + 32)
        section = Section(
            segment=self.fixed_string(at + 16, 16),
            name=self.fixed_string(at, 16),
            address=address,
            size=size,
            offset=offset,
            relocation_offset=reloff,
            relocation_count=nreloc,
            flags=flags,
            index=len(self.sections) + 1,
            command_offset=at,
        )
        if section.has_file_bytes:
            self.bounds(offset, size)
        self.bounds(reloff, nreloc * RELOCATION_SIZE)
        self.sections.append(section)

    def _read_symbols(self) -> list[Symbol]:
        symoff, nsyms, stroff, strsize = self.symtab
        self.bounds(symoff, nsyms * NLIST_SIZE)
        self.bounds(stroff, strsize)
        symbols = []
        for index in range(nsyms):
            at = symoff + index * NLIST_SIZE
            strx, kind, section, descriptor, value = self.unpack("<IBBHQ", at)
            if strx >= strsize:
                raise Error("symbol name outside string table")
            name = self.string(stroff + strx, strsize - strx)
            symbol = Symbol(name, kind, section, descriptor, value, index, at, strx)
            if symbol.defined and not 1 <= section <= len(self.sections):
                raise Error("defined symbol has invalid section")
            symbols.append(symbol)
        return symbols

    def bounds(self, offset: int, size: int) -> None:
        """Raise Error unless size bytes at offset lie inside the file."""
        if offset < 0 or size < 0 or offset > len(self.data) or size > len(self.data) - offset:
            raise Error("Mach-O range outside file")

    def unpack(self, fmt: str, offset: int) -> tuple:
        self.bounds(offset, struct.calcsize(fmt))
        return struct.unpack_from(fmt, self.data, offset)

    def fixed_string(self, offset: int, size: int) -> str:
        """A NUL-padded ASCII name of size bytes, such as a segment or section name."""
        self.bounds(offset, size)
        try:
            return bytes(self.data[offset : offset + size]).split(b"\0", 1)[0].decode("ascii")
        except UnicodeDecodeError as error:
            raise Error("invalid Mach-O fixed string") from error

    def string(self, offset: int, size: int) -> str:
        """A NUL-terminated UTF-8 string that must end within size bytes."""
        self.bounds(offset, size)
        end = self.data.find(0, offset, offset + size)
        if end < 0:
            raise Error("unterminated Mach-O string")
        try:
            return self.data[offset:end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise Error("invalid symbol string") from error

    def section(self, segment: str, name: str) -> Section:
        """The one section named segment,name."""
        found = [s for s in self.sections if (s.segment, s.name) == (segment, name)]
        if len(found) != 1:
            raise Error(f"expected one {segment},{name} section")
        return found[0]

    def file_offset(self, address: int, width: int) -> int:
        """The file offset of width bytes at address, which one section's file bytes hold."""
        for section in self.sections:
            if not section.has_file_bytes:
                continue
            if (
                section.address <= address
                and width <= section.size
                and address - section.address <= section.size - width
            ):
                offset = section.offset + address - section.address
                self.bounds(offset, width)
                return offset
        raise Error("address is outside file-backed sections")

    def read(self, address: int, width: int) -> bytes:
        offset = self.file_offset(address, width)
        return bytes(self.data[offset : offset + width])

    def symbol(self, name: str) -> Symbol:
        """The one defined symbol named name."""
        found = [s for s in self.symbols if s.name == name and s.defined]
        if len(found) != 1:
            raise Error(f"expected one definition of {name}")
        return found[0]

    def symbol_extent(self, symbol: Symbol) -> int:
        """The bytes from a defined symbol to the next one in its section, or to its end."""
        section = self.sections[symbol.section - 1]
        end = min(
            (
                s.value
                for s in self.symbols
                if s.defined and s.section == symbol.section and s.value > symbol.value
            ),
            default=section.address + section.size,
        )
        if symbol.value < section.address or end > section.address + section.size:
            raise Error("symbol outside owning section")
        return end - symbol.value

    def _read_relocations(self, offset: int, count: int, base: int) -> Iterator[Relocation]:
        """The count relocation entries at offset, whose addresses count from base."""
        self.bounds(offset, count * RELOCATION_SIZE)
        for i in range(count):
            at = offset + i * RELOCATION_SIZE
            # r_address, whose top bit is r_scattered, then r_symbolnum:24, r_pcrel:1,
            # r_length:2, r_extern:1 and r_type:4.
            address, info = self.unpack("<iI", at)
            if address < 0:
                raise Error("scattered relocation is unsupported")
            symbol_index = info & 0xFFFFFF
            external = bool(info >> 27 & 1)
            if external and symbol_index >= len(self.symbols):
                raise Error("relocation symbol index outside table")
            yield Relocation(
                address=base + address,
                symbol_index=symbol_index,
                pcrel=bool(info >> 24 & 1),
                width=1 << (info >> 25 & 3),
                external=external,
                kind=info >> 28,
                entry_offset=at,
            )

    def relocations(self) -> list[Relocation]:
        """Every relocation: an object's per section, or a kext's external then local ones."""
        if self.filetype == MH_OBJECT:
            return [
                relocation
                for section in self.sections
                for relocation in self._read_relocations(
                    section.relocation_offset, section.relocation_count, section.address
                )
            ]
        if self.dysymtab is None:
            raise Error("kext has no dynamic symbol table")
        table = self.dysymtab
        return [
            *self._read_relocations(table.extreloff, table.nextrel, self.base),
            *self._read_relocations(table.locreloff, table.nlocrel, self.base),
        ]

    def localize(self, definitions: dict[str, tuple[int, int]]) -> Image:
        """Turn undefined symbols into private local definitions, editing the bytes in place.

        definitions maps each symbol's name to the (section index, address) that defines it.
        The symbol table is regrouped as LC_DYSYMTAB requires (locals, external definitions,
        then undefined symbols), and every external relocation and indirect-symbol entry is
        renumbered to match. Returns the image read again from the edited bytes.
        """
        if self.filetype != MH_KEXT_BUNDLE or self.dysymtab is None:
            raise Error("localization requires a linked kext")
        table = self.dysymtab
        # Module, table-of-contents and reference tables belong to dylibs, never our kexts.
        if table.ntoc or table.nmodtab or table.nextrefsyms:
            raise Error("unsupported dynamic symbol metadata")
        entries = self._regrouped_symbol_entries(definitions)
        renumbered = {old: new for new, (_, old, _) in enumerate(entries)}
        self._renumber_relocations(renumbered)
        self._renumber_indirect_symbols(renumbered)
        for new, (_, _, entry) in enumerate(entries):
            at = self.symtab.symoff + new * NLIST_SIZE
            self.data[at : at + NLIST_SIZE] = entry
        local = sum(group == LOCAL for group, _, _ in entries)
        external = sum(group == EXTERNAL for group, _, _ in entries)
        table = table._replace(
            ilocalsym=0,
            nlocalsym=local,
            iextdefsym=local,
            nextdefsym=external,
            iundefsym=local + external,
            nundefsym=len(entries) - local - external,
        )
        struct.pack_into("<18I", self.data, self.dysymtab_offset + 8, *table)
        return Image(self.data)

    def _regrouped_symbol_entries(
        self, definitions: dict[str, tuple[int, int]]
    ) -> list[tuple[int, int, bytes]]:
        """Each nlist entry as (group, old index, bytes), in its new order."""
        entries = []
        missing = set(definitions)
        for symbol in self.symbols:
            entry = bytes(self.data[symbol.entry_offset : symbol.entry_offset + NLIST_SIZE])
            if symbol.name in definitions:
                if not symbol.undefined:
                    raise Error("placeholder names an already defined symbol")
                section, value = definitions[symbol.name]
                if not 1 <= section <= len(self.sections):
                    raise Error("placeholder section outside image")
                entry = struct.pack("<IBBHQ", symbol.name_offset, N_SECT, section, 0, value)
                missing.discard(symbol.name)
                group = LOCAL
            elif symbol.undefined:
                group = UNDEFINED
            else:
                group = EXTERNAL if symbol.kind & N_EXT else LOCAL
            entries.append((group, symbol.index, entry))
        if missing:
            raise Error("placeholder has no undefined symbol")
        return sorted(entries, key=lambda entry: entry[:2])

    def _renumber_relocations(self, renumbered: dict[int, int]) -> None:
        for relocation in self.relocations():
            if relocation.external:
                at = relocation.entry_offset + 4
                info = self.unpack("<I", at)[0]
                new = renumbered[relocation.symbol_index]
                struct.pack_into("<I", self.data, at, (info & 0xFF000000) | new)

    def _renumber_indirect_symbols(self, renumbered: dict[int, int]) -> None:
        offset, count = self.dysymtab.indirectsymoff, self.dysymtab.nindirectsyms
        self.bounds(offset, count * 4)
        for i in range(count):
            at = offset + i * 4
            index = self.unpack("<I", at)[0]
            if not index & (INDIRECT_SYMBOL_LOCAL | INDIRECT_SYMBOL_ABS):
                if index >= len(self.symbols):
                    raise Error("indirect symbol index outside table")
                struct.pack_into("<I", self.data, at, renumbered[index])

    def rename_symbols(self, names: dict[str, str]) -> Image:
        """Rename symbols in the string table, editing the bytes in place.

        names maps each old name to a new name of the same byte length, so no offset moves.
        A name whose bytes another symbol's name shares is refused rather than changing that
        other symbol too. Returns the image read again from the edited bytes.
        """
        stroff = self.symtab.stroff
        for old, new in names.items():
            encoded = new.encode()
            if len(encoded) != len(old.encode()) or b"\0" in encoded:
                raise Error("marker rename must preserve its byte length")
            matches = [s for s in self.symbols if s.name == old]
            if not matches:
                raise Error("marker rename names no symbol")
            for symbol in matches:
                start = stroff + symbol.name_offset
                end = start + len(encoded)
                for other in self.symbols:
                    if other.name == old:
                        continue
                    other_start = stroff + other.name_offset
                    other_end = other_start + len(other.name.encode())
                    if start < other_end and other_start < end:
                        raise Error("marker string shares storage with another symbol")
                self.data[start:end] = encoded
        return Image(self.data)
