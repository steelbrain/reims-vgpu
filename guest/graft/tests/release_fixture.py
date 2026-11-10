# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
"""A throwaway Graft repository with one commit, and its release, for the release tests."""

from __future__ import annotations

import os
import plistlib
import struct
import subprocess
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "internal"))
import release

VERSION = "1.2.3"
XCODE = "27.2 (27B52)"
# The files make_repository commits, by name, with their text.
SOURCES = {
    ".gitignore": "build/\ndist/\n.DS_Store\n",
    release.XCODE_PIN: XCODE + "\n",
    "Makefile": "all:\n",
    "src/Start.cpp": "int start;\n",
    "sdk/include/graft/Graft.h": "#pragma once\n",
    "sdk/consumer.mk": "# rules\n",
    "sdk/README.md": "[docs/consumers.md](../docs/consumers.md)\n",
    "docs/consumers.md": "# Consumers\n",
    "tools/script.py": "#!/usr/bin/env python3\n",
    "LICENSE": "license\n",
    "LICENSES/GPL-2.0-or-later.txt": "text\n",
}
TRACKED = sorted(SOURCES)
# The files of the release make_repository's commit builds, by their names in its folder,
# and those it holds as executable: each kext's binary, and the tracked file git records as
# executable.
RELEASE_FILES = sorted(
    [
        *TRACKED,
        *(
            f"{arch['bundle']}/{name}"
            for arch in release.ARCHITECTURES
            for name in release.BUNDLE_FILES
        ),
        release.COMMIT,
        release.SUMS,
    ]
)
EXECUTABLE = {*release.EXECUTABLES, "tools/script.py"}


def git(root: Path, *args: str) -> str:
    """Run git in root with no user or system configuration."""
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    environment.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    command = ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", *args]
    result = subprocess.run(
        command, cwd=root, env=environment, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def make_bundle(bundle: Path, header, version: str = VERSION) -> Path:
    """A stand-in Graft bundle: an Info.plist naming version and a kext header."""
    info = bundle / "Contents/Info.plist"
    binary = bundle / "Contents/MacOS/Graft"
    binary.parent.mkdir(parents=True, exist_ok=True)
    info.write_bytes(plistlib.dumps({"CFBundleVersion": version}))
    binary.write_bytes(struct.pack("<IIII", *header) + bytes(16))
    binary.chmod(0o755)
    return bundle


def make_bundles(parent: Path, version: str = VERSION) -> dict[str, Path]:
    """A stand-in bundle per architecture under parent, by architecture name."""
    return {
        arch["name"]: make_bundle(parent / arch["bundle"], arch["header"], version)
        for arch in release.ARCHITECTURES
    }


def write(root: Path, name: str, text: str) -> None:
    """Write text to root/name, making its directories."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def make_repository(root: Path) -> None:
    """A repository with one commit holding SOURCES, tools/script.py executable."""
    for name, text in SOURCES.items():
        write(root, name, text)
    (root / "tools/script.py").chmod(0o755)
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "release")


def make_release(root: Path, bundles: dict[str, Path]) -> Path:
    """Build root's release with stand-ins for Xcode, whose build yields bundles; return
    the release's zip."""
    selected = mock.patch.object(release, "selected_xcode", lambda: XCODE)
    compile_bundles = mock.patch.object(release, "compile_bundles", lambda _source: dict(bundles))
    with selected, compile_bundles:
        release.release(root)
    return root / release.DIST / f"graft-{VERSION}.zip"
