#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
"""Vendoring takes a release zip only when every file matches its SHA256SUMS, records it,
and never replaces a foreign directory."""

from __future__ import annotations

import hashlib
import json
import stat
import sys
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from release_fixture import (
    EXECUTABLE,
    RELEASE_FILES,
    VERSION,
    git,
    make_bundles,
    make_release,
    make_repository,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import vendor

TOP = f"graft-{VERSION}/"
# What DEST receives besides its record: the release's files but COMMIT and SHA256SUMS.
TAKEN = [name for name in RELEASE_FILES if name not in (vendor.COMMIT, vendor.SUMS)]
FILE = stat.S_IFREG | 0o644


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def with_sums(members: list[tuple[str, bytes, int]]) -> list[tuple[str, bytes, int]]:
    """members, with SHA256SUMS rewritten to list every other file as it is."""
    others = [member for member in members if member[0] != TOP + vendor.SUMS]
    sums = "".join(f"{digest(data)}  {name[len(TOP) :]}\n" for name, data, _mode in others)
    return [*others, (TOP + vendor.SUMS, sums.encode(), FILE)]


class VendorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        scratch = Path(self.temp.name)
        self.root = scratch / "graft"
        self.root.mkdir()
        make_repository(self.root)
        self.zip = make_release(self.root, make_bundles(scratch / "rebuilt"))
        self.dest = scratch / "consumer/graft"

    def tearDown(self):
        self.temp.cleanup()

    def members(self) -> list[tuple[str, bytes, int]]:
        """The release zip's members, each as its name, its bytes and its mode."""
        with zipfile.ZipFile(self.zip) as opened:
            return [
                (info.filename, opened.read(info), info.external_attr >> 16)
                for info in opened.infolist()
            ]

    def changed(self, members: list[tuple[str, bytes, int]]) -> Path:
        """A zip of members, written as the release zip is."""
        path = Path(self.temp.name) / "changed.zip"
        with warnings.catch_warnings(), zipfile.ZipFile(path, "w") as out:
            warnings.simplefilter("ignore")  # zipfile warns of a name written twice.
            for name, data, mode in members:
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = mode << 16
                out.writestr(info, data)
        return path

    def assert_refused(self, archive: Path, pattern: str) -> None:
        with self.assertRaisesRegex(vendor.VendorError, pattern):
            vendor.vendor(archive, self.dest)
        self.assertFalse(self.dest.exists())

    def test_takes_the_release_and_records_it(self):
        commit = git(self.root, "rev-parse", "HEAD")
        self.assertEqual(
            vendor.vendor(self.zip, self.dest),
            f"vendor: {self.dest}: Graft {VERSION}, {len(TAKEN)} files of {commit[:12]}",
        )
        copied = sorted(
            p.relative_to(self.dest).as_posix() for p in self.dest.rglob("*") if p.is_file()
        )
        self.assertEqual(copied, sorted([*TAKEN, vendor.RECORD]))
        self.assertEqual((self.dest / "sdk/consumer.mk").read_text(), "# rules\n")
        # The SDK's documentation links into docs/, which the copy holds too.
        self.assertTrue((self.dest / "sdk/../docs/consumers.md").is_file())
        record = (self.dest / vendor.RECORD).read_text()
        self.assertIn(f'version = "{VERSION}"', record)
        self.assertIn(f'commit = "{commit}"', record)
        for name in TAKEN:
            path = self.dest / name
            self.assertIn(f"{json.dumps(name)} = {json.dumps(digest(path.read_bytes()))}", record)
            self.assertEqual(
                stat.S_IMODE(path.stat().st_mode), 0o755 if name in EXECUTABLE else 0o644
            )

    def test_replaces_an_earlier_copy_whole(self):
        vendor.vendor(self.zip, self.dest)
        (self.dest / "stale.txt").write_text("left by an older copy\n")
        vendor.vendor(self.zip, self.dest)
        self.assertFalse((self.dest / "stale.txt").exists())

    def test_a_failed_copy_leaves_the_earlier_copy(self):
        vendor.vendor(self.zip, self.dest)
        before = (self.dest / vendor.RECORD).read_text()
        failing = mock.patch.object(vendor.shutil, "copytree", side_effect=OSError("disk full"))
        with failing, self.assertRaisesRegex(OSError, "disk full"):
            vendor.vendor(self.zip, self.dest)
        self.assertEqual((self.dest / vendor.RECORD).read_text(), before)
        self.assertEqual(sorted(p.name for p in self.dest.parent.iterdir()), ["graft"])

    def test_refuses_a_directory_without_a_record(self):
        self.dest.mkdir(parents=True)
        (self.dest / "unrelated.txt").write_text("keep me\n")
        with self.assertRaisesRegex(vendor.VendorError, "refusing to replace it"):
            vendor.vendor(self.zip, self.dest)
        self.assertTrue((self.dest / "unrelated.txt").is_file())

    def test_refuses_a_file_that_differs_from_its_sum(self):
        members = [
            (name, b"# changed\n" if name == TOP + "sdk/consumer.mk" else data, mode)
            for name, data, mode in self.members()
        ]
        self.assert_refused(self.changed(members), "differ from SHA256SUMS: sdk/consumer.mk")

    def test_refuses_a_missing_or_unlisted_file(self):
        members = [member for member in self.members() if member[0] != TOP + "LICENSE"]
        self.assert_refused(self.changed(members), "in SHA256SUMS but not in the release: LICENSE")
        members = [*self.members(), (TOP + "extra.txt", b"extra\n", FILE)]
        self.assert_refused(
            self.changed(members), "in the release but not in SHA256SUMS: extra.txt"
        )

    def test_refuses_a_link_a_directory_or_a_name_outside_the_folder(self):
        for name, mode in (
            (TOP + "sdk/link", stat.S_IFLNK | 0o777),
            (TOP + "sdk/more/", stat.S_IFDIR | 0o755),
            ("/" + TOP + "sdk/absolute", FILE),
            (TOP + "../outside", FILE),
            (TOP + "sdk/./dot", FILE),
            ("loose", FILE),
        ):
            with self.subTest(name=name):
                members = with_sums([*self.members(), (name, b"consumer.mk", mode)])
                self.assert_refused(self.changed(members), "refusing zip member")

    def test_refuses_anything_but_one_graft_folder(self):
        members = with_sums([*self.members(), ("other/file", b"other\n", FILE)])
        self.assert_refused(self.changed(members), "one folder, not 2")
        members = [
            ("release/" + name[len(TOP) :], data, mode) for name, data, mode in self.members()
        ]
        self.assert_refused(self.changed(members), "folder is not graft-<version>")
        members = [*self.members(), self.members()[0]]
        self.assert_refused(self.changed(members), "appears twice")

    def test_refuses_a_release_that_holds_its_own_record(self):
        members = with_sums([*self.members(), (TOP + vendor.RECORD, b"version = 0\n", FILE)])
        self.assert_refused(self.changed(members), "holds a source.toml, which the record")

    def test_refuses_a_malformed_record(self):
        sums = TOP + vendor.SUMS
        members = [
            (name, data + b"not a sum\n" if name == sums else data, mode)
            for name, data, mode in self.members()
        ]
        self.assert_refused(self.changed(members), "SHA256SUMS: unexpected line 'not a sum'")
        members = [
            (name, b"not a commit\n" if name == TOP + vendor.COMMIT else data, mode)
            for name, data, mode in self.members()
        ]
        self.assert_refused(self.changed(with_sums(members)), "COMMIT names no commit")
        members = [member for member in self.members() if member[0] != TOP + vendor.COMMIT]
        self.assert_refused(self.changed(with_sums(members)), "holds no COMMIT")

    def test_main_refuses_a_file_that_is_not_a_zip(self):
        bad = Path(self.temp.name) / "graft-1.2.3.zip"
        bad.write_bytes(b"not a zip\n")
        with self.assertRaisesRegex(SystemExit, "vendor: File is not a zip file"):
            vendor.main(["vendor.py", str(bad), str(self.dest)])
        self.assertFalse(self.dest.exists())


if __name__ == "__main__":
    unittest.main()
