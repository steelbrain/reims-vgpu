# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
"""Measurement input errors must never become compatibility conclusions."""

import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "hook_symbols", Path(__file__).parents[1] / "tools/internal/hook-symbols.py"
)
symbols = importlib.util.module_from_spec(spec)
spec.loader.exec_module(symbols)

MACH_HEADER = struct.Struct("<8I")
SYMTAB_COMMAND = struct.Struct("<6I")
NLIST = struct.Struct("<IBBHQ")


def image(base=0):
    """A kext with one LC_SYMTAB, whose tables follow it at base plus 56.

    Its symbols: _local (local), _debug (a debugging entry), and _method three times, as
    an undefined external reference, a defined global, and an undefined reference again.
    """
    strings = b"\0_local\0_debug\0_method\0"
    rows = [(1, 0x0E), (8, 0xE1), (15, 0x01), (15, 0x0F), (15, 0x01)]
    header = MACH_HEADER.pack(0xFEEDFACF, 0x01000007, 3, 0xB, 1, SYMTAB_COMMAND.size, 0, 0)
    symoff = base + MACH_HEADER.size + SYMTAB_COMMAND.size
    stroff = symoff + NLIST.size * len(rows)
    table = SYMTAB_COMMAND.pack(2, SYMTAB_COMMAND.size, symoff, len(rows), stroff, len(strings))
    entries = b"".join(NLIST.pack(name, kind, 1, 0, 32) for name, kind in rows)
    return header + table + entries + strings


class SymbolsTest(unittest.TestCase):
    def test_only_defined_globals(self):
        self.assertEqual(
            symbols.classify_bytes(image(), ["_local", "_debug", "_method", "_absent"]),
            {"_local": "--", "_debug": "--", "_method": "def", "_absent": "--"},
        )

    def test_fileset(self):
        header = MACH_HEADER.pack(0xFEEDFACF, 0x01000007, 3, 0xC, 1, 40, 0, 0)
        entry = struct.pack("<IIQQII8s", 0x80000035, 40, 0, 72, 32, 0, b"fixture\0")
        self.assertEqual(
            symbols.classify_bytes(header + entry + image(72), ["_method"]), {"_method": "def"}
        )

    def test_truncation(self):
        data = image()
        for length in range(len(data)):
            with self.assertRaises(ValueError):
                symbols.classify_bytes(data[:length], ["_method"])

    def test_bad_command(self):
        data = bytearray(image())
        struct.pack_into("<I", data, 36, 0)  # LC_SYMTAB's cmdsize
        with self.assertRaises(ValueError):
            symbols.classify_bytes(data, ["_method"])

    def test_missing_input(self):
        with (
            patch.object(Path, "is_file", return_value=False),
            self.assertRaisesRegex(ValueError, "incomplete inputs"),
        ):
            symbols.classify_release(Path("fixture"), ["_method"])

    def test_wanted_symbols_are_the_first_column(self):
        names = symbols.read_wanted()
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(name.startswith("_") for name in names), names)
        self.assertIn("_IOLog", names)
        self.assertIn("__ZN6OSKext24lookupKextWithIdentifierEPKc", names)

    def test_cached_releases_skip_work_and_kdk(self):
        with tempfile.TemporaryDirectory(prefix="graft-hook-cache-") as cache:
            root = Path(cache)
            for name in ("11.7.10-20G1427", "26.7.2-25G332", "kdk-25G241", "work"):
                (root / name).mkdir()
            (root / "12.7.4-21H1123").write_text("")  # a file, not a release
            self.assertEqual(
                [p.name for p in symbols.cached_releases(root)],
                ["11.7.10-20G1427", "26.7.2-25G332"],
            )


if __name__ == "__main__":
    unittest.main()
