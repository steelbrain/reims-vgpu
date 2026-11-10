#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
"""Copy a Graft release into a consumer's tree, with a record of where it came from.

    python3 -I -B tools/vendor.py graft-<version>.zip DEST

The zip is the one Graft's release on GitHub attaches. Every file in it must
match the zip's SHA256SUMS, and it is refused whole for a link, a name outside
its one folder, or a file missing from SHA256SUMS or from the zip. DEST
receives the release's folder as it is, but for COMMIT and SHA256SUMS: Graft's
source tree at the release's commit, with the SDK in sdk/, and Graft.kext and
Graft-arm64e.kext. DEST/source.toml records the version, the commit the
release was built from and each file's SHA-256, in place of the zip's COMMIT
and SHA256SUMS. A consumer commits DEST as it is and checks its files against
the record. DEST is replaced whole, so it must be absent or hold an earlier
copy's record.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path

RECORD = "source.toml"
COMMIT = "COMMIT"
SUMS = "SHA256SUMS"
FOLDER = re.compile(r"graft-([0-9][0-9A-Za-z.+-]*)")
SUM = re.compile(r"([0-9a-f]{64})  (.+)")


class VendorError(Exception):
    """A release that cannot be taken as it is."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_files(archive: Path) -> tuple[str, dict[str, tuple[bytes, int]]]:
    """The zip's version, and each file's bytes and mode by its name in the zip's folder.

    Every member must be a regular file inside one folder named graft-<version>: a link, a
    directory, an absolute name, an empty, "." or ".." part, a second folder, or a name that
    appears twice is refused.
    """
    folders = set()
    files: dict[str, tuple[bytes, int]] = {}
    with zipfile.ZipFile(archive) as opened:
        for member in opened.infolist():
            parts = member.filename.split("/")
            mode = member.external_attr >> 16
            if (
                len(parts) < 2
                or any(part in ("", ".", "..") for part in parts)
                or stat.S_IFMT(mode) != stat.S_IFREG
            ):
                raise VendorError(f"{archive.name}: refusing zip member {member.filename}")
            folders.add(parts[0])
            name = "/".join(parts[1:])
            if name in files:
                raise VendorError(f"{archive.name}: {member.filename} appears twice")
            files[name] = (opened.read(member), stat.S_IMODE(mode))
    if len(folders) != 1:
        raise VendorError(f"{archive.name}: the release must be one folder, not {len(folders)}")
    match = FOLDER.fullmatch(folders.pop())
    if not match:
        raise VendorError(f"{archive.name}: its folder is not graft-<version>")
    return match.group(1), files


def check_sums(files: dict[str, tuple[bytes, int]]) -> str:
    """Check every file against SHA256SUMS and return the commit COMMIT names."""
    for name in (SUMS, COMMIT):
        if name not in files:
            raise VendorError(f"the release holds no {name}")
    sums = {}
    for line in files[SUMS][0].decode().splitlines():
        match = SUM.fullmatch(line)
        if not match or match.group(2) in sums:
            raise VendorError(f"{SUMS}: unexpected line {line!r}")
        sums[match.group(2)] = match.group(1)
    listed = set(files) - {SUMS}
    missing = sorted(set(sums) - listed)
    if missing:
        raise VendorError(f"files in {SUMS} but not in the release: " + ", ".join(missing))
    unlisted = sorted(listed - set(sums))
    if unlisted:
        raise VendorError(f"files in the release but not in {SUMS}: " + ", ".join(unlisted))
    differ = sorted(name for name in sums if digest(files[name][0]) != sums[name])
    if differ:
        raise VendorError(f"files that differ from {SUMS}: " + ", ".join(differ))
    commit = files[COMMIT][0].decode()
    if not re.fullmatch(r"[0-9a-f]{40}\n", commit):
        raise VendorError(f"{COMMIT} names no commit")
    return commit.strip()


def record_text(version: str, commit: str, files: dict[str, str]) -> str:
    """source.toml's text: the version, the commit and each file's SHA-256."""
    lines = [
        f"# Graft {version}, copied from its release zip by Graft's tools/vendor.py.",
        "# Replace the directory and this record together; do not edit either.",
        f"version = {json.dumps(version)}",
        f"commit = {json.dumps(commit)}",
        "",
        "[files]",
    ]
    lines += [f"{json.dumps(name)} = {json.dumps(files[name])}" for name in sorted(files)]
    return "\n".join(lines) + "\n"


def vendor(archive: Path, dest: Path) -> str:
    """Replace dest with the release in the zip at archive; return a one-line summary."""
    version, files = read_files(archive)
    commit = check_sums(files)
    if RECORD in files:
        raise VendorError(f"the release holds a {RECORD}, which the record would replace")
    if dest.exists() and not (dest / RECORD).is_file():
        raise VendorError(f"{dest} exists and holds no {RECORD}; refusing to replace it")
    taken = {name: files[name] for name in files if name not in (SUMS, COMMIT)}
    with tempfile.TemporaryDirectory(prefix="graft-vendor-") as work:
        copy = Path(work) / "graft"
        for name, (data, mode) in taken.items():
            target = copy.joinpath(*name.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            target.chmod(0o755 if mode & 0o100 else 0o644)
        record = {name: digest(data) for name, (data, _mode) in taken.items()}
        (copy / RECORD).write_text(record_text(version, commit, record))
        # Copy beside dest first, so a failed copy leaves dest as it was, then swap the two.
        staging = dest.with_name(f".{dest.name}.vendor")
        previous = dest.with_name(f".{dest.name}.previous")
        for leftover in (staging, previous):
            if leftover.exists():
                shutil.rmtree(leftover)
        shutil.copytree(copy, staging)
        if dest.exists():
            dest.rename(previous)
        staging.rename(dest)
        if previous.exists():
            shutil.rmtree(previous)
    return f"vendor: {dest}: Graft {version}, {len(taken)} files of {commit[:12]}"


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit(__doc__)
    try:
        print(vendor(Path(argv[1]), Path(argv[2]).resolve()))
    except (OSError, ValueError, VendorError, zipfile.BadZipFile) as error:
        raise SystemExit(f"vendor: {error}") from error
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
