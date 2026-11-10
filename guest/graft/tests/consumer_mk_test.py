# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
"""sdk/consumer.mk's rule as make sees it: what it depends on, under any makefile name.

A consumer whose makefile is not named Makefile includes a copy of the SDK's consumer.mk
and headers, with GRAFT_KEXT naming a stand-in that only writes the staged binary, or fails.
make's question mode then says whether the staged bundle is current: after a build, after
an SDK header changes, and after a build that failed.
"""

import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

SDK = Path(__file__).resolve().parents[1] / "sdk"
CONSUMER = """NAME = Fake
VERSION = 1.0.0
ARCH = -arch x86_64
BUILD = build
KEXT = build/Fake.kext
BINARY = $(KEXT)/Contents/MacOS/$(NAME)
OBJECTS = build/Fake.o
HEADERS = src/Fake.h
CXXFLAGS =

all: $(BINARY)

include ../sdk/consumer.mk
"""
# graft-kext's stand-in: it writes the staged binary build names, unless FAIL is set.
TOOL = """#!/bin/sh
for argument; do
    case $argument in
    --name=*) name=${argument#--name=} ;;
    --output=*) output=${argument#--output=} ;;
    esac
done
[ -z "$FAIL" ] || exit 1
mkdir -p "$output/Contents/MacOS" && : >"$output/Contents/MacOS/$name"
"""


class ConsumerMkTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="graft-consumer-mk-")
        root = Path(self.temp.name)
        (root / "sdk").mkdir()
        shutil.copy(SDK / "consumer.mk", root / "sdk")
        shutil.copytree(SDK / "include", root / "sdk/include")
        self.consumer = root / "consumer"
        for name in ("src/graft.json", "src/Info.plist", "src/Fake.h", "build/Fake.o"):
            (self.consumer / name).parent.mkdir(parents=True, exist_ok=True)
            (self.consumer / name).touch()
        (self.consumer / "consumer.make").write_text(CONSUMER)
        self.tool = root / "graft-kext"
        self.tool.write_text(TOOL)
        self.tool.chmod(0o755)
        # Every input is older than anything a build writes.
        for path in [*root.rglob("*")]:
            os.utime(path, (1, 1))

    def tearDown(self):
        self.temp.cleanup()

    def make(self, *arguments, fail=False):
        environment = {k: v for k, v in os.environ.items() if not k.startswith("MAKE")}
        if fail:
            environment["FAIL"] = "1"
        return subprocess.run(
            ["make", "-f", "consumer.make", f"GRAFT_KEXT={self.tool}", *arguments],
            cwd=self.consumer,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def touch(self, path):
        """Make path newer than anything the build wrote, past make's time resolution."""
        later = time.time() + 10
        os.utime(path, (later, later))

    def current(self):
        """Whether make takes the staged bundle as current."""
        return self.make("-q").returncode == 0

    def test_built_bundle_is_current(self):
        built = self.make()
        self.assertEqual(built.returncode, 0, built.stderr)
        self.assertTrue((self.consumer / "build/Fake.kext/Contents/MacOS/Fake").exists())
        self.assertTrue(self.current())

    def test_an_sdk_header_change_builds_again(self):
        self.assertEqual(self.make().returncode, 0)
        self.touch(Path(self.temp.name) / "sdk/include/graft/IOMobileFramebufferLegacy.h")
        self.assertFalse(self.current())

    def test_a_failed_build_is_not_current(self):
        self.assertEqual(self.make().returncode, 0)
        self.touch(self.consumer / "src/graft.json")
        self.assertNotEqual(self.make(fail=True).returncode, 0)
        self.assertFalse(self.current())


if __name__ == "__main__":
    unittest.main()
