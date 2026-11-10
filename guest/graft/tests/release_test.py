#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
"""make release writes a release's folder and zip, the tracked tree and both kexts, the same
bytes from the same commit and Xcode, and refuses a tree that differs from HEAD and any other
Xcode. Every link in a release's Markdown resolves inside the release."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "internal"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lint
from release_fixture import (
    EXECUTABLE,
    RELEASE_FILES,
    TRACKED,
    VERSION,
    XCODE,
    git,
    make_bundle,
    make_bundles,
    make_repository,
    release,
    write,
)

# The real compile and Xcode query, kept before setUp replaces them with stand-ins.
COMPILE = release.compile_bundles
SELECTED_XCODE = release.selected_xcode


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "graft"
        self.root.mkdir()
        make_repository(self.root)
        # Stand-ins for Xcode: the version it reports, and a build that yields a
        # bundle per architecture.
        self.xcode = XCODE
        self.rebuilt = make_bundles(Path(self.temp.name) / "rebuilt")
        patches = [
            mock.patch.object(release, "selected_xcode", lambda: self.xcode),
            mock.patch.object(release, "compile_bundles", self.compile),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def tearDown(self):
        self.temp.cleanup()

    def compile(self, source: Path) -> dict[str, Path]:
        self.compiled = sorted(
            p.relative_to(source).as_posix() for p in source.rglob("*") if p.is_file()
        )
        return dict(self.rebuilt)

    def released(self, root: Path | None = None) -> tuple[Path, Path]:
        """The folder and the zip of the release make_repository's commit builds."""
        dist = (root or self.root) / "dist"
        return dist / f"graft-{VERSION}", dist / f"graft-{VERSION}.zip"

    def test_release_writes_the_folder_and_the_zip(self):
        summary = release.release(self.root)
        folder, archive = self.released()
        commit = git(self.root, "rev-parse", "HEAD")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        self.assertEqual(
            summary,
            f"dist/graft-{VERSION}.zip: Graft {VERSION} (x86_64, arm64e)"
            f" of {commit[:12]}, SHA-256 {digest}",
        )
        self.assertEqual(release.folder_files(folder), RELEASE_FILES)
        self.assertEqual((folder / "COMMIT").read_text(), commit + "\n")
        for name in TRACKED:
            self.assertEqual((folder / name).read_bytes(), (self.root / name).read_bytes())
        for arch in release.ARCHITECTURES:
            for name in release.BUNDLE_FILES:
                self.assertEqual(
                    (folder / arch["bundle"] / name).read_bytes(),
                    (self.rebuilt[arch["name"]] / name).read_bytes(),
                )
        sums = [
            f"{hashlib.sha256((folder / name).read_bytes()).hexdigest()}  {name}"
            for name in RELEASE_FILES
            if name != "SHA256SUMS"
        ]
        self.assertEqual((folder / "SHA256SUMS").read_text().splitlines(), sums)
        for name in RELEASE_FILES:
            self.assertEqual(
                stat.S_IMODE((folder / name).stat().st_mode),
                0o755 if name in EXECUTABLE else 0o644,
            )

    def test_the_zip_holds_the_folder_stored_with_fixed_modes_and_time(self):
        release.release(self.root)
        folder, archive = self.released()
        date = time.gmtime(int(git(self.root, "log", "-1", "--format=%ct")))[:6]
        with zipfile.ZipFile(archive) as opened:
            members = opened.infolist()
            self.assertEqual(
                [member.filename for member in members],
                [f"graft-{VERSION}/{name}" for name in RELEASE_FILES],
            )
            for member, name in zip(members, RELEASE_FILES):
                self.assertEqual(member.compress_type, zipfile.ZIP_STORED)
                # A zip keeps the seconds of a time to an even number.
                self.assertEqual(member.date_time, (*date[:5], date[5] // 2 * 2))
                self.assertEqual(
                    member.external_attr >> 16,
                    stat.S_IFREG | (0o755 if name in EXECUTABLE else 0o644),
                )
                self.assertEqual(opened.read(member), (folder / name).read_bytes())

    def test_two_releases_of_a_commit_are_the_same_bytes(self):
        release.release(self.root)
        first = self.released()[1].read_bytes()
        # Another checkout, other file times and modes, and another umask reach no byte.
        clone = Path(self.temp.name) / "clone"
        git(Path(self.temp.name), "clone", "-q", str(self.root), str(clone))
        for bundle in self.rebuilt.values():
            for name in release.BUNDLE_FILES:
                (bundle / name).chmod(0o600)
                os.utime(bundle / name, (0, 0))
        umask = os.umask(0o077)
        try:
            release.release(clone)
        finally:
            os.umask(umask)
        self.assertEqual(self.released(clone)[1].read_bytes(), first)

    def test_release_builds_and_carries_the_tracked_files_only(self):
        # Ignored files are allowed in the tree, and neither built from nor carried.
        for name in ("src/.DS_Store", "build/Start.o", "sdk/.DS_Store"):
            write(self.root, name, "ignored\n")
        release.release(self.root)
        self.assertEqual(self.compiled, TRACKED)
        self.assertEqual(release.folder_files(self.released()[0]), RELEASE_FILES)

    def test_release_replaces_an_earlier_one(self):
        release.release(self.root)
        folder = self.released()[0]
        write(folder, "stale.txt", "left by an earlier release\n")
        release.release(self.root)
        self.assertEqual(release.folder_files(folder), RELEASE_FILES)

    def test_release_refuses_a_tree_that_differs_from_head(self):
        # An edit to any tracked file, or an untracked file git does not ignore, anywhere.
        cases = (
            ("sdk/consumer.mk", "# an uncommitted edit\n"),
            ("tools/script.py", "# an uncommitted edit\n"),
            (".xcode-version", "26.6\n"),
            ("notes.txt", "untracked\n"),
            ("LICENSES/extra.txt", "untracked\n"),
        )
        for name, text in cases:
            with self.subTest(name=name):
                path = self.root / name
                before = path.read_bytes() if path.exists() else None
                write(self.root, name, text)
                with self.assertRaisesRegex(
                    release.ReleaseError, f"commit or remove: {re.escape(name)}"
                ):
                    release.release(self.root)
                if before is None:
                    path.unlink()
                else:
                    path.write_bytes(before)
        write(self.root, "staged.txt", "added, not committed\n")
        git(self.root, "add", "staged.txt")
        with self.assertRaisesRegex(release.ReleaseError, "commit or remove: staged.txt"):
            release.release(self.root)
        self.assertFalse((self.root / "dist").exists())

    def test_release_refuses_a_tracked_name_it_adds(self):
        # A tracked file at a name the release adds at its top would be replaced by it, and
        # one that differs only in case would be on a case-insensitive file system.
        for name in ("COMMIT", "sha256sums", "Graft.kext/Contents/Info.plist", "graft-arm64e.kext"):
            with self.subTest(name=name):
                write(self.root, name, "tracked\n")
                git(self.root, "add", name)
                git(self.root, "commit", "-q", "-m", "add")
                with self.assertRaisesRegex(
                    release.ReleaseError, f"tracked {re.escape(name)} collides with the release's"
                ):
                    release.release(self.root)
                git(self.root, "rm", "-q", name)
                git(self.root, "commit", "-q", "-m", "remove")
        self.assertFalse((self.root / "dist").exists())

    def test_release_refuses_another_xcode(self):
        # Another version, or another build of the pinned version, such as its beta.
        for selected in ("26.6 (26F80)", "27.2 (27B5040e)"):
            with self.subTest(selected=selected):
                self.xcode = selected
                with self.assertRaisesRegex(
                    release.ReleaseError,
                    rf"built with Xcode 27\.2 \(27B52\), which \.xcode-version pins, not Xcode"
                    rf" {re.escape(selected)}: select Xcode 27\.2 \(27B52\) with xcode-select",
                ):
                    release.release(self.root)
                self.assertFalse((self.root / "dist").exists())

    def test_selected_xcode_is_xcodebuilds(self):
        def xcode(command, **_options):
            self.assertEqual(command, ["xcodebuild", "-version"])
            return subprocess.CompletedProcess(command, 0, "Xcode 27.2\nBuild version 27B52\n")

        def tools_only(command, **_options):
            text = "xcode-select: error: tool 'xcodebuild' requires Xcode\n"
            return subprocess.CompletedProcess(command, 1, text)

        def no_build(command, **_options):
            return subprocess.CompletedProcess(command, 0, "Xcode 27.2\n")

        with mock.patch.object(release.subprocess, "run", xcode):
            self.assertEqual(SELECTED_XCODE(), "27.2 (27B52)")
        failing = mock.patch.object(release.subprocess, "run", tools_only)
        with failing, self.assertRaisesRegex(release.ReleaseError, "names no Xcode: xcode-select"):
            SELECTED_XCODE()
        failing = mock.patch.object(release.subprocess, "run", no_build)
        with failing, self.assertRaisesRegex(release.ReleaseError, "names no Xcode: Xcode 27.2$"):
            SELECTED_XCODE()

    def test_release_refuses_bundles_of_two_versions(self):
        arch = release.ARCHITECTURES[1]
        make_bundle(self.rebuilt[arch["name"]], arch["header"], "1.2.4")
        with self.assertRaisesRegex(
            release.ReleaseError, f"{arch['bundle']} names a different version"
        ):
            release.release(self.root)

    def test_compile_fixes_the_build_directory_and_the_environment(self):
        source = Path(self.temp.name) / "snapshot"

        def make(command, cwd, env, **_options):
            self.assertEqual(command, ["make", "BUILD=build", "all", "graft-arm64e"])
            self.assertEqual(cwd, source)
            for name in release.BUILD_ENVIRONMENT:
                self.assertNotIn(name, env)
            self.assertEqual(env["KEPT"], "kept")

        environment = dict.fromkeys(release.BUILD_ENVIRONMENT, "elsewhere")
        environment["KEPT"] = "kept"
        run = mock.patch.object(release.subprocess, "run", make)
        with mock.patch.dict(os.environ, environment), run:
            bundles = COMPILE(source)
        self.assertEqual(
            bundles,
            {
                "x86_64": source / "build/Graft.kext",
                "arm64e": source / "build/Graft-arm64e.kext",
            },
        )

    def test_bundle_shape(self):
        arch = release.ARCHITECTURES[0]
        bundle = self.rebuilt[arch["name"]]
        (bundle / "extra").touch()
        with self.assertRaisesRegex(release.ReleaseError, "exactly"):
            release.release(self.root)
        (bundle / "extra").unlink()
        (bundle / "extra").symlink_to("Contents/Info.plist")
        with self.assertRaisesRegex(release.ReleaseError, "symlink"):
            release.release(self.root)
        (bundle / "extra").unlink()
        (bundle / "Contents/MacOS/Graft").write_bytes(bytes(32))
        with self.assertRaisesRegex(release.ReleaseError, "not the expected Mach-O kext"):
            release.release(self.root)
        self.assertFalse((self.root / "dist").exists())

    def test_release_refuses_a_tracked_link(self):
        (self.root / "src/Alias.cpp").symlink_to("Start.cpp")
        git(self.root, "add", "src/Alias.cpp")
        git(self.root, "commit", "-q", "-m", "link")
        with self.assertRaisesRegex(
            release.ReleaseError, "tracked src/Alias.cpp is not a regular file"
        ):
            release.release(self.root)
        self.assertFalse((self.root / "dist").exists())


class ReleaseLinksTest(unittest.TestCase):
    def test_every_link_in_a_release_resolves_inside_it(self):
        # A consumer reads Graft's docs in its vendored copy of a release, so every relative
        # link in the Markdown of the release this repository makes, and every heading it
        # names, must be in that release.
        root = release.ROOT
        tracked = release.tracked_files(root)
        files = {*tracked, *release.release_modes(tracked)}
        problems = []
        for name in tracked:
            if name.endswith(".md"):
                problems += lint.check_links(root, name, (root / name).read_text(), files)
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
