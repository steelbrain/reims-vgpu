#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
"""Check what clang-format and ruff leave alone: license headers, plain-text
hygiene and the links between the docs.

    python3 -I -B tools/internal/lint.py

- Every C, C++, Objective-C, Python, make and shell file, and every script
  that starts with #!, carries the SPDX license line near its top.
- Every text file ends in exactly one newline and has no trailing whitespace,
  carriage returns or tab indentation (make files may indent with tabs).
- Every relative link in a Markdown file names a file or directory in the
  repository, and every #fragment names a heading in its target.

It checks the files git tracks or would track, so a new file is checked
before it is committed.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPDX = "SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later"
CODE = {".c", ".cpp", ".h", ".m", ".py", ".mk", ".sh"}
# The license texts are kept exactly as the FSF publishes them.
VERBATIM = ("LICENSES/",)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
FENCE = re.compile(r"^(```|~~~)")


def tracked(root: Path) -> list[str]:
    """The files git tracks or would track under root, as sorted relative names."""
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    names = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        env=environment,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    return sorted({name.decode() for name in names if name and (root / name.decode()).is_file()})


def is_code(name: str) -> bool:
    path = Path(name)
    return path.suffix in CODE or path.name == "Makefile"


def text_of(path: Path) -> str | None:
    """The file's text, or None for a binary or non-UTF-8 file."""
    data = path.read_bytes()
    if b"\0" in data:
        return None
    try:
        return data.decode()
    except UnicodeDecodeError:
        return None


def check_text(name: str, text: str) -> list[str]:
    """The license-line and plain-text problems in one file."""
    problems = []
    # A #! line makes a file a script, whatever its name says.
    code = is_code(name) or text.startswith("#!")
    if code and SPDX not in "\n".join(text.splitlines()[:3]):
        problems.append(f"{name}: no SPDX license line in its first three lines")
    if name.startswith(VERBATIM):
        return problems
    if (text and not text.endswith("\n")) or text.endswith("\n\n"):
        problems.append(f"{name}: must end in exactly one newline")
    tabs_allowed = Path(name).name == "Makefile" or name.endswith(".mk")
    for number, line in enumerate(text.split("\n"), 1):
        if line.endswith((" ", "\t")):
            problems.append(f"{name}:{number}: trailing whitespace")
        if "\r" in line:
            problems.append(f"{name}:{number}: carriage return")
        if not tabs_allowed and line.startswith("\t"):
            problems.append(f"{name}:{number}: tab indentation")
    return problems


def prose(text: str) -> list[tuple[int, str]]:
    """Each line outside fenced code blocks, with inline code removed."""
    lines, fenced = [], False
    for number, line in enumerate(text.split("\n"), 1):
        if FENCE.match(line.lstrip()):
            fenced = not fenced
            continue
        if not fenced:
            lines.append((number, re.sub(r"`[^`]*`", "", line)))
    return lines


def slug(heading: str) -> str:
    """The fragment GitHub gives a heading."""
    # Code spans and emphasis marks go; an underscore inside a word, as in mac_policy_ops,
    # stays.
    text = re.sub(
        r"[`*]|(?<![^\W_])_+|_+(?![^\W_])|\[([^\]]*)\]\([^)]*\)", r"\1", heading.strip().lower()
    )
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def anchors(text: str) -> set[str]:
    """The fragments of every heading in a Markdown text, numbered as GitHub numbers repeats."""
    seen: dict[str, int] = {}
    result = set()
    for _, line in prose(text):
        match = re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line)
        if not match:
            continue
        base = slug(match[1])
        count = seen.get(base, 0)
        seen[base] = count + 1
        result.add(base if count == 0 else f"{base}-{count}")
    return result


def check_links(root: Path, name: str, text: str, files: set[str]) -> list[str]:
    """The relative links in Markdown file name whose file or heading does not exist."""
    problems = []
    directories = {str(Path(f).parent) for f in files} | {
        str(parent) for f in files for parent in Path(f).parents
    }
    for number, line in prose(text):
        for target in LINK.findall(line):
            if re.match(r"[a-z]+:", target):
                continue
            path, _, fragment = target.partition("#")
            resolved = name if not path else os.path.normpath(Path(name).parent / path)
            if resolved not in files and resolved.rstrip("/") not in directories:
                problems.append(f"{name}:{number}: link to a missing file: {target}")
            elif (
                fragment
                and resolved.endswith(".md")
                and fragment not in anchors((root / resolved).read_text())
            ):
                problems.append(f"{name}:{number}: link to a missing heading: {target}")
    return problems


def lint(root: Path) -> list[str]:
    """Every problem in the tree at root, one line each."""
    files = tracked(root)
    problems = []
    for name in files:
        text = text_of(root / name)
        if text is None:
            continue
        problems += check_text(name, text)
        if name.endswith(".md") and not name.startswith(VERBATIM):
            problems += check_links(root, name, text, set(files))
    return problems


def main() -> int:
    try:
        problems = lint(ROOT)
    except subprocess.CalledProcessError as error:
        raise SystemExit("lint: run in a git checkout; git ls-files failed") from error
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
