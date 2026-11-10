#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
"""Build a Graft release into dist/.

    python3 -I -B tools/internal/release.py

A release is the source tree at a commit and both kexts built from it. It
refuses a tree that is not its commit's, builds both kexts from a copy of every
file git tracks, with the Xcode whose version and build .xcode-version names,
and writes dist/graft-<version>/ and dist/graft-<version>.zip, where <version>
is the Makefile's VERSION. The folder holds every tracked file as it is in the
repository, and at its top Graft.kext (x86-64), Graft-arm64e.kext (arm64e),
COMMIT (the commit the release is built from) and SHA256SUMS (the SHA-256 of
every other file). The zip holds the folder. Its entries are stored, in a fixed
order, with fixed modes and the commit's time, so the same commit and Xcode give
the same zip.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import plistlib
import shutil
import stat
import struct
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_FILES = ("Contents/Info.plist", "Contents/MacOS/Graft")
# Each architecture Graft releases: the make target that builds its kext, the
# bundle's name under both build/ and the release's folder, and the (magic, CPU
# type, CPU subtype, file type) its Mach-O binary must have. One binary per
# architecture covers every macOS release.
ARCHITECTURES = (
    {
        "name": "x86_64",
        "target": "all",
        "bundle": "Graft.kext",
        "header": (0xFEEDFACF, 0x01000007, 0x00000003, 11),
    },
    {
        "name": "arm64e",
        "target": "graft-arm64e",
        "bundle": "Graft-arm64e.kext",
        "header": (0xFEEDFACF, 0x0100000C, 0xC0000002, 11),
    },
)
XCODE_PIN = ".xcode-version"
DIST = "dist"
COMMIT = "COMMIT"
SUMS = "SHA256SUMS"
# The names a release adds at the top of its folder, beside the tracked tree.
ADDED = (*(arch["bundle"] for arch in ARCHITECTURES), COMMIT, SUMS)
# Each kext's binary, which a release holds with the execute bits set, as it
# does each file git tracks as executable.
EXECUTABLES = tuple(f"{arch['bundle']}/Contents/MacOS/Graft" for arch in ARCHITECTURES)
# git's modes for a tracked regular file, and for one with the execute bits set.
GIT_FILE, GIT_EXECUTABLE = "100644", "100755"
# Variables that would let the caller's environment change the build: make's
# own, the compiler's include paths and extra options, and xcrun's SDK and
# toolchain.
BUILD_ENVIRONMENT = (
    "MAKEFLAGS",
    "MAKEOVERRIDES",
    "MFLAGS",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "C_INCLUDE_PATH",
    "CCC_OVERRIDE_OPTIONS",
    "SDKROOT",
    "TOOLCHAINS",
    "MACOSX_DEPLOYMENT_TARGET",
)


class ReleaseError(Exception):
    """A release that cannot be built as asked."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tracked_files(root: Path) -> dict[str, bool]:
    """Every file git tracks in root's repository, by its name there, with whether git
    records it as executable.

    The names come from git ls-files, not from walking the tree, so no ignored or untracked
    file is among them. A tracked link or submodule is refused, as is a name that would
    collide with one the release adds at its top, on a case-insensitive file system too.
    """
    files = {}
    added = {name.lower() for name in ADDED}
    for entry in git(root, "ls-files", "-z", "--stage").split(b"\0"):
        if not entry:
            continue
        status, name = entry.decode().split("\t", 1)
        mode = status.split()[0]
        if mode not in (GIT_FILE, GIT_EXECUTABLE):
            raise ReleaseError(f"tracked {name} is not a regular file")
        if name.split("/")[0].lower() in added:
            raise ReleaseError(
                f"tracked {name} collides with the release's own " + ", ".join(ADDED)
            )
        files[name] = mode == GIT_EXECUTABLE
    return files


def check_bundle(bundle: Path, header: tuple[int, int, int, int]) -> None:
    """Refuse a bundle that is not exactly an Info.plist and a kext binary for header."""
    if not bundle.is_dir() or bundle.is_symlink():
        raise ReleaseError(f"no Graft bundle at {bundle}")
    present = []
    for path in bundle.rglob("*"):
        if path.is_symlink():
            raise ReleaseError(f"symlink in the Graft bundle: {path.relative_to(bundle)}")
        if path.is_file():
            present.append(path.relative_to(bundle).as_posix())
    if sorted(present) != sorted(BUNDLE_FILES):
        raise ReleaseError("the Graft bundle must hold exactly " + " and ".join(BUNDLE_FILES))
    binary = (bundle / "Contents/MacOS/Graft").read_bytes()
    if len(binary) < 16 or struct.unpack_from("<IIII", binary) != header:
        raise ReleaseError(f"{bundle.name}'s binary is not the expected Mach-O kext")


def bundle_version(bundle: Path) -> str:
    info = plistlib.loads((bundle / "Contents/Info.plist").read_bytes())
    version = info.get("CFBundleVersion")
    if not isinstance(version, str) or not version:
        raise ReleaseError("the Graft bundle names no CFBundleVersion")
    return version


def git(root: Path, *args: str) -> bytes:
    """Run git on root's repository, whatever GIT_* variables the caller set."""
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    result = subprocess.run(
        ["git", "-C", str(root), *args], env=environment, check=True, capture_output=True
    )
    return result.stdout


def compile_bundles(source: Path) -> dict[str, Path]:
    """Build every architecture in source, a snapshot of its sources, and return
    each one's bundle path by architecture name.

    BUILD is named on the command line so the environment cannot move the output.
    """
    environment = {k: v for k, v in os.environ.items() if k not in BUILD_ENVIRONMENT}
    targets = [arch["target"] for arch in ARCHITECTURES]
    subprocess.run(["make", "BUILD=build", *targets], cwd=source, env=environment, check=True)
    return {arch["name"]: source / "build" / arch["bundle"] for arch in ARCHITECTURES}


def require_committed(root: Path) -> None:
    """Refuse a tree that differs from HEAD: a tracked file changed, or an untracked file git
    does not ignore.

    A release is a release of a commit, so an edit would be in the release but not in its
    commit, and an untracked file is a sign the tree is not the commit's. An ignored file,
    such as build/ or .DS_Store, is never carried, so it is allowed.
    """
    status = git(root, "status", "--porcelain", "--untracked-files=all")
    if status:
        names = [line[3:] for line in status.decode().splitlines()]
        raise ReleaseError("sources differ from HEAD; commit or remove: " + ", ".join(names))


def selected_xcode() -> str:
    """The Xcode that xcrun and xcodebuild use, as its version and build, such as
    27.2 (27B52)."""
    result = subprocess.run(
        ["xcodebuild", "-version"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    lines = result.stdout.splitlines()
    if (
        result.returncode
        or len(lines) < 2
        or not lines[0].startswith("Xcode ")
        or not lines[1].startswith("Build version ")
    ):
        raise ReleaseError("xcodebuild -version names no Xcode: " + result.stdout.strip())
    return f"{lines[0][len('Xcode ') :]} ({lines[1][len('Build version ') :]})"


def require_xcode(root: Path) -> None:
    """Refuse to build with any Xcode but the one .xcode-version names, by version and build.

    Another Xcode builds other bytes, even a beta of the same version, so the pin is what
    lets a release built anywhere hold the kexts a release built in CI holds.
    """
    pinned = (root / XCODE_PIN).read_text().strip()
    selected = selected_xcode()
    if selected != pinned:
        raise ReleaseError(
            f"a release is built with Xcode {pinned}, which {XCODE_PIN} pins, not Xcode"
            f" {selected}: select Xcode {pinned} with xcode-select"
        )


def release_modes(files: dict[str, bool]) -> dict[str, int]:
    """The mode of each file in the release of the tracked files files, by its name in the
    release's folder: 755 for each kext's binary and each file git tracks as executable, 644
    for the rest."""
    modes = {name: 0o755 if executable else 0o644 for name, executable in files.items()}
    for arch in ARCHITECTURES:
        for name in BUNDLE_FILES:
            path = f"{arch['bundle']}/{name}"
            modes[path] = 0o755 if path in EXECUTABLES else 0o644
    modes[COMMIT] = modes[SUMS] = 0o644
    return modes


def folder_files(folder: Path) -> list[str]:
    """Every file under folder, by its name there, in a stable order."""
    return sorted(
        path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()
    )


def write_folder(
    folder: Path,
    source: Path,
    files: dict[str, bool],
    bundles: dict[str, Path],
    commit: str,
    modes: dict[str, int],
) -> None:
    """Write the release's folder: the tracked files, from source, both kexts, COMMIT and
    SHA256SUMS, each file with its mode in modes."""
    copies = [
        (bundles[arch["name"]] / name, f"{arch['bundle']}/{name}")
        for arch in ARCHITECTURES
        for name in BUNDLE_FILES
    ]
    copies += [(source / name, name) for name in files]
    for path, name in copies:
        (folder / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, folder / name)
    (folder / COMMIT).write_text(commit + "\n")
    sums = "".join(
        f"{digest((folder / name).read_bytes())}  {name}\n" for name in folder_files(folder)
    )
    (folder / SUMS).write_text(sums)
    for name in folder_files(folder):
        (folder / name).chmod(modes[name])


def write_zip(folder: Path, path: Path, when: int, modes: dict[str, int]) -> None:
    """Write folder into the zip at path, under its own name, as the same bytes every time:
    stored, in name order, each file with its mode in modes and the time when (UTC)."""
    date = time.gmtime(when)[:6]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for name in folder_files(folder):
            info = zipfile.ZipInfo(f"{folder.name}/{name}", date_time=date)
            info.create_system = 3  # Unix, so that the mode below applies.
            info.external_attr = (stat.S_IFREG | modes[name]) << 16
            archive.writestr(info, (folder / name).read_bytes())


def release(root: Path) -> str:
    """Build the release into dist/; return a one-line summary."""
    require_committed(root)
    files = tracked_files(root)
    require_xcode(root)
    commit, when = git(root, "log", "-1", "--format=%H %ct", "HEAD").decode().split()
    with tempfile.TemporaryDirectory(prefix="graft-release-") as work:
        # The build reads only the tracked files, copied as the release carries them.
        snapshot = Path(work) / "source"
        for name in files:
            target = snapshot / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, target)
        bundles = compile_bundles(snapshot)
        for arch in ARCHITECTURES:
            check_bundle(bundles[arch["name"]], arch["header"])
        # The Makefile writes its VERSION into each bundle's Info.plist.
        version = bundle_version(bundles[ARCHITECTURES[0]["name"]])
        for arch in ARCHITECTURES:
            if bundle_version(bundles[arch["name"]]) != version:
                raise ReleaseError(f"{arch['bundle']} names a different version")
        top = f"graft-{version}"
        folder = Path(work) / top
        modes = release_modes(files)
        write_folder(folder, snapshot, files, bundles, commit, modes)
        write_zip(folder, Path(work) / f"{top}.zip", int(when), modes)
        dist = root / DIST
        dist.mkdir(exist_ok=True)
        shutil.rmtree(dist / top, ignore_errors=True)
        shutil.move(str(folder), str(dist / top))
        shutil.move(str(Path(work) / f"{top}.zip"), str(dist / f"{top}.zip"))
    names = ", ".join(arch["name"] for arch in ARCHITECTURES)
    zip_digest = digest((dist / f"{top}.zip").read_bytes())
    return f"{DIST}/{top}.zip: Graft {version} ({names}) of {commit[:12]}, SHA-256 {zip_digest}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.parse_args()
    try:
        print(release(ROOT))
    except (OSError, ValueError, ReleaseError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"release: {error}\n")


if __name__ == "__main__":
    main()
