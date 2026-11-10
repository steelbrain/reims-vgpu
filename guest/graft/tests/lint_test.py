#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
"""The lint finds missing license lines, untidy text and broken doc links."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "internal"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import lint
from release_fixture import git

HEADER = "// " + lint.SPDX + "\n"


class LintTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        git(self.root, "init", "-q")

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name: str, text: str) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_clean_tree(self):
        self.write("src/A.cpp", HEADER + "int a;\n")
        self.write("Makefile", "# " + lint.SPDX + "\nall:\n\ttrue\n")
        self.write("README.md", "# Title\n\n[The code](src/A.cpp), [below](#usage).\n\n## Usage\n")
        self.assertEqual(lint.lint(self.root), [])

    def test_license_line(self):
        self.write("src/A.cpp", "int a;\n")
        self.write("tool.py", "#!/usr/bin/env python3\n# " + lint.SPDX + "\n")
        self.write("bin/tool", "#!/bin/sh\necho\n")
        self.write("bin/notes", "echo\n")
        missing = "no SPDX license line in its first three lines"
        self.assertEqual(lint.lint(self.root), [f"bin/tool: {missing}", f"src/A.cpp: {missing}"])

    def test_text_hygiene(self):
        self.write("a.md", "trailing \n")
        self.write("b.md", "no newline")
        self.write("c.md", "two newlines\n\n")
        self.write("d.md", "\tindented\n")
        self.write("e.md", "carriage\r\n")
        self.assertEqual(
            lint.lint(self.root),
            [
                "a.md:1: trailing whitespace",
                "b.md: must end in exactly one newline",
                "c.md: must end in exactly one newline",
                "d.md:1: tab indentation",
                "e.md:1: carriage return",
            ],
        )

    def test_ignored_files_are_not_checked(self):
        self.write(".gitignore", "build/\n")
        self.write("build/out.md", "trailing \n")
        self.assertEqual(lint.lint(self.root), [])

    def test_links(self):
        self.write("docs/guide.md", "# Guide\n\n## Two words\n\n## Two words\n")
        self.write(
            "README.md",
            "[ok](docs/guide.md#two-words) [dup](docs/guide.md#two-words-1) [dir](docs/)\n"
            "[web](https://example.invalid/x) `[code](missing.md)`\n"
            "[gone](docs/missing.md) [heading](docs/guide.md#nowhere)\n"
            "```\n[fenced](missing.md)\n```\n",
        )
        self.assertEqual(
            lint.lint(self.root),
            [
                "README.md:3: link to a missing file: docs/missing.md",
                "README.md:3: link to a missing heading: docs/guide.md#nowhere",
            ],
        )

    def test_slugs(self):
        self.assertEqual(lint.slug("`make lint`, explained!"), "make-lint-explained")
        self.assertEqual(lint.slug("3. Graft's x86-64 core"), "3-grafts-x86-64-core")
        self.assertEqual(lint.slug("The `mac_policy_ops` table"), "the-mac_policy_ops-table")
        self.assertEqual(lint.slug("_Emphasis_ and __strong__"), "emphasis-and-strong")


if __name__ == "__main__":
    unittest.main()
