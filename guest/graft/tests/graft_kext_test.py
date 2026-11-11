# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
"""graft-kext as a consumer's build runs it: the files it writes, its output and exit status.

These tests see only what a consumer's build sees. They run the graft-kext GRAFT_KEXT
names, which make test sets to the one it builds, and build the fixture consumer in
tests/consumer/ from its source, for x86-64 and for arm64e. They check that:

- build stages a dormant bundle with a gated Info.plist, and leaves each step's output in
  its work directory;
- the staged manifest carries each personality's provider matching, as its author wrote
  it, for any provider, without the gate key or the keys that name the driver;
- build writes what probe, the two links consumer.mk documents, reserve and stage write when
  run one at a time;
- each refusal exits 1 with its reason, and a malformed command line exits 2.

What the staged binary holds is checked by staging_test.cpp, on the same fixture.
"""

import copy
import json
import os
import plistlib
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/consumer"
TOOL = os.environ.get("GRAFT_KEXT", str(ROOT / "build/graft-kext"))
VERSION = "1.0.0"
BINARY = "Contents/MacOS/Fixture"
IDENTIFIER = "ai.aneesiqbal.graft.test.ConsumerFixture"
PROVIDER = "ai.aneesiqbal.graft.test.Provider"
# What the probe measures of the fixture's classes, the same for both architectures.
LAYOUTS = {
    "classes": [
        {
            "name": "Parent",
            "size": 32,
            "slots": [
                "__ZN6ParentD1Ev",
                "__ZN6ParentD0Ev",
                "__ZN6Parent9inheritedEi",
                "__ZN6Parent7changedEi",
                "__ZN6Parent6unusedEi",
            ],
        }
    ],
    "consumers": [
        {
            "name": "Consumer",
            "size": 32,
            "slots": [
                "__ZN8ConsumerD1Ev",
                "__ZN8ConsumerD0Ev",
                "__ZN6Parent9inheritedEi",
                "__ZN8Consumer7changedEi",
                "__ZN6Parent6unusedEi",
                "__ZN8Consumer8appendedEi",
            ],
        }
    ],
    "synthesized": [],
}


def run(*arguments, environment=None):
    """graft-kext run with arguments, its output captured as text."""
    return subprocess.run(
        [TOOL, *map(str, arguments)], capture_output=True, text=True, check=False, env=environment
    )


def files(root):
    """Every file under root, by its name relative to root, with its bytes."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def compile_fixture(arch, output):
    """Compile the fixture's source as a macOS 11 kext object for arch."""
    sdk = subprocess.check_output(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True
    ).strip()
    command = [
        *("xcrun", "clang++", "-arch", arch, "-isysroot", sdk),
        *("-mmacosx-version-min=11.0", "-mkernel", "-nostdinc"),
        *("-isystem", f"{sdk}/System/Library/Frameworks/Kernel.framework/Headers"),
        *("-DKERNEL", "-std=c++17", "-fapple-kext", "-fno-rtti", "-fno-exceptions", "-O2"),
        *("-c", str(FIXTURE / "Fixture.cpp"), "-o", str(output)),
    ]
    subprocess.run(command, check=True, capture_output=True)


def link(arch, objects, output, reservation=None):
    """The link consumer.mk documents: clang++ as a kext, -no_data_const on arm64e."""
    command = ["xcrun", "clang++", "-arch", arch, "-nostdlib", "-Xlinker", "-kext"]
    if arch == "arm64e":
        command += ["-Xlinker", "-no_data_const"]
    command += [*map(str, objects), "-lcc_kext"]
    if reservation is not None:
        command += ["-Xlinker", "-sectcreate", "-Xlinker", "__DATA", "-Xlinker", "__graft"]
        command += ["-Xlinker", str(reservation)]
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([*command, "-o", str(output)], check=True, capture_output=True)


def fixture_info():
    """The fixture's Info.plist, with @VERSION@ for its version."""
    return plistlib.loads((FIXTURE / "Info.plist").read_bytes())


def fixture_declaration():
    """The fixture's graft.json."""
    return json.loads((FIXTURE / "graft.json").read_text())


def provider_matching(binary):
    """The provider matching the manifest in a staged binary carries, parsed.

    The manifest starts with its magic, "GRFT"; its header is 34 little-endian u32s, of
    which the fifth is the string table's offset and the last the matching string's offset
    in it (docs/manifest.md).
    """
    data = binary.read_bytes()
    start = data.index(b"GRFT")
    header = struct.unpack_from("<34I", data, start)
    assert header[3] == 136, "not a manifest header"
    at = start + header[4] + header[33]
    return plistlib.loads(data[at : data.index(b"\0", at)])


class Inputs:
    """The fixture consumer's object, declaration and Info.plist for arch, under root."""

    def __init__(self, root, arch):
        root.mkdir(parents=True)
        self.root, self.arch = root, arch
        self.object = root / "Fixture.o"
        compile_fixture(arch, self.object)
        self.declaration = FIXTURE / "graft.json"
        self.info_plist = FIXTURE / "Info.plist"
        self.flags = ["-arch", arch, f"-I{FIXTURE}"]

    def write(self, name, data):
        path = self.root / name
        path.write_bytes(data)
        return path

    def build(self, name, *, declaration=None, info_plist=None, objects=None, flags=None):
        """build into root/name, as consumer.mk runs it: the work directory is name/graft."""
        options = {
            "name": "Fixture",
            "version": VERSION,
            "arch": f"-arch {self.arch}",
            "declaration": declaration or self.declaration,
            "info-plist": info_plist or self.info_plist,
            "work": self.root / name / "graft",
            "output": self.root / name / "Fixture.kext",
        }
        return run(
            "build",
            *(f"--{key}={value}" for key, value in options.items()),
            *(objects or [self.object]),
            "--",
            *(self.flags if flags is None else flags),
        )


class BuildChecks:
    """The checks of a build for ARCH, which each subclass sets."""

    ARCH = ""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="graft-kext-test-")
        cls.inputs = Inputs(Path(cls.temp.name) / "inputs", cls.ARCH)
        cls.process = cls.inputs.build("built")
        cls.work = cls.inputs.root / "built/graft"
        cls.output = cls.inputs.root / "built/Fixture.kext"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.assertEqual((self.process.returncode, self.process.stdout), (0, ""))

    def test_build_stages_a_dormant_bundle(self):
        self.assertEqual(sorted(files(self.output)), ["Contents/Info.plist", BINARY])
        expected = fixture_info()
        expected["CFBundleVersion"] = VERSION
        del expected["OSBundleLibraries"][PROVIDER]
        expected["IOKitPersonalities"]["Fixture"].update(
            CFBundleIdentifier=IDENTIFIER,
            IOPropertyMatch={f"{IDENTIFIER}.graft-bound": True},
        )
        staged = plistlib.loads((self.output / "Contents/Info.plist").read_bytes())
        self.assertEqual(staged, expected)

    def test_build_writes_the_provider_matching(self):
        personality = fixture_info()["IOKitPersonalities"]["Fixture"]
        del personality["IOClass"]
        self.assertEqual(provider_matching(self.output / BINARY), {"Fixture": personality})

    def test_build_leaves_each_steps_output(self):
        self.assertEqual(
            (self.work / "layouts.json").read_text(), json.dumps(LAYOUTS, indent=2) + "\n"
        )
        unbound = self.work / "Unbound.kext"
        info = self.inputs.info_plist.read_bytes().replace(b"@VERSION@", VERSION.encode())
        self.assertEqual(sorted(files(unbound)), ["Contents/Info.plist", BINARY])
        self.assertEqual((unbound / "Contents/Info.plist").read_bytes(), info)
        self.assertTrue((self.work / "census").is_file())
        reservation = (self.work / "manifest-reserve.bin").read_bytes()
        self.assertTrue(reservation)
        self.assertEqual(reservation, bytes(len(reservation)))

    def test_build_is_its_steps_run_one_at_a_time(self):
        steps = self.inputs.root / "steps"
        steps.mkdir()
        unbound = steps / "Unbound.kext"
        declaration, layouts = self.inputs.declaration, steps / "layouts.json"
        probed = run("probe", declaration, layouts, "--", *self.inputs.flags)
        self.assertEqual((probed.returncode, probed.stdout, probed.stderr), (0, "", ""))
        link(self.ARCH, [self.inputs.object], steps / "census")
        info = self.inputs.info_plist.read_bytes().replace(b"@VERSION@", VERSION.encode())
        (unbound / "Contents/MacOS").mkdir(parents=True)
        (unbound / "Contents/Info.plist").write_bytes(info)
        reserved = run(
            "reserve",
            steps / "census",
            declaration,
            layouts,
            unbound / "Contents/Info.plist",
            steps / "manifest-reserve.bin",
        )
        self.assertEqual((reserved.returncode, reserved.stdout, reserved.stderr), (0, "", ""))
        link(self.ARCH, [self.inputs.object], unbound / BINARY, steps / "manifest-reserve.bin")
        staged = run("stage", unbound, declaration, layouts, steps / "Fixture.kext")
        self.assertEqual((staged.returncode, staged.stdout, staged.stderr), (0, "", ""))
        self.assertEqual(files(steps / "Fixture.kext"), files(self.output))
        work = {k: v for k, v in files(steps).items() if not k.startswith("Fixture.kext/")}
        self.assertEqual(work, files(self.work))


class BuildX86Test(BuildChecks, unittest.TestCase):
    ARCH = "x86_64"


class BuildArm64eTest(BuildChecks, unittest.TestCase):
    ARCH = "arm64e"

    def test_the_probe_targets_the_links_architecture(self):
        # Compiler flags that name no architecture are probed for ARCH's, as the
        # links are: this copy of the fixture's header compiles only for arm64e.
        include = self.inputs.root / "arm64e-only"
        include.mkdir()
        header = (FIXTURE / "Fixture.h").read_text()
        (include / "Fixture.h").write_text(f"#ifndef __arm64e__\n#error x86-64\n#endif\n{header}")
        process = self.inputs.build("no-arch-flag", flags=[f"-I{include}"])
        self.assertEqual((process.returncode, process.stderr), (0, ""))
        built = self.inputs.root / "no-arch-flag/Fixture.kext"
        self.assertEqual(files(built), files(self.output))


class ProviderMatchingTest(unittest.TestCase):
    """A consumer's personalities may match any provider. The manifest carries each one's
    matching as written, its own IOPropertyMatch included, without the gate key, IOClass,
    CFBundleIdentifier, IOProbeScore or IOMatchCategory."""

    def test_any_provider(self):
        with tempfile.TemporaryDirectory(prefix="graft-kext-matching-") as temp:
            inputs = Inputs(Path(temp) / "inputs", "x86_64")
            info = fixture_info()
            device = {
                "IOClass": "Consumer",
                "IOProviderClass": "IOPCIDevice",
                "IOPCIClassMatch": "0x03000000&0xff000000",
                "IOPropertyMatch": [{"model": "one"}, {"model": "two"}],
                "IOProbeScore": 5000,
                "IOMatchCategory": "FixtureCategory",
            }
            resource = {
                "IOClass": "Consumer",
                "IOProviderClass": "IOResources",
                "IOResourceMatch": "IOKit",
                "IOPropertyMatch": {"model": "three"},
                "CFBundleIdentifier": IDENTIFIER,
            }
            info["IOKitPersonalities"] = {"Device": device, "Resource": resource}
            path = inputs.write("any-provider.plist", plistlib.dumps(info))
            process = inputs.build("any-provider", info_plist=path)
            self.assertEqual((process.returncode, process.stdout, process.stderr), (0, "", ""))
            output = inputs.root / "any-provider/Fixture.kext"
            driver = ("IOClass", "CFBundleIdentifier", "IOProbeScore", "IOMatchCategory")
            self.assertEqual(
                provider_matching(output / BINARY),
                {
                    name: {k: v for k, v in personality.items() if k not in driver}
                    for name, personality in info["IOKitPersonalities"].items()
                },
            )
            staged = plistlib.loads((output / "Contents/Info.plist").read_bytes())
            gate = {f"{IDENTIFIER}.graft-bound": True}
            self.assertEqual(
                staged["IOKitPersonalities"]["Device"]["IOPropertyMatch"],
                [{"model": "one", **gate}, {"model": "two", **gate}],
            )
            self.assertEqual(
                staged["IOKitPersonalities"]["Resource"]["IOPropertyMatch"],
                {"model": "three", **gate},
            )


class RefusalTest(unittest.TestCase):
    """Each input graft-kext refuses: exit status 1 and the reason, or 2 and the usage. With
    --help it prints the usage and exits 0."""

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="graft-kext-refusal-")
        cls.root = Path(cls.temp.name)
        cls.inputs = Inputs(cls.root / "inputs", "x86_64")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def assertRefused(self, process, reason):  # noqa: N802 (unittest's naming)
        self.assertEqual((process.returncode, process.stdout), (1, ""))
        self.assertTrue(process.stderr.endswith(f"consumer: {reason}\n"), process.stderr)

    def test_a_provider_the_info_plist_does_not_name(self):
        info = fixture_info()
        del info["OSBundleLibraries"][PROVIDER]
        path = self.inputs.write("no-provider.plist", plistlib.dumps(info))
        process = self.inputs.build("no-provider", info_plist=path)
        self.assertRefused(process, "consumer does not declare a selected provider dependency")
        self.assertFalse((self.inputs.root / "no-provider/Fixture.kext").exists())

    def test_a_declaration_graft_could_not_bind_by(self):
        config = fixture_declaration()
        config["libraries"].append(dict(config["libraries"][0]))
        path = self.inputs.write("twice.json", json.dumps(config).encode())
        layouts = self.root / "twice-layouts.json"
        self.assertRefused(run("probe", path, layouts), "duplicate provider identifier")
        self.assertFalse(layouts.exists())
        self.assertRefused(
            self.inputs.build("twice", declaration=path), "duplicate provider identifier"
        )

    def test_a_declaration_of_the_wrong_shape(self):
        # A value of the wrong type is refused, named by its place in the declaration.
        parent = {"name": "Parent", "header": "Fixture.h", "library": 0}
        display = {"name": "Display", "header": "Fixture.h", "parent": 0}
        imports = {"_parent_global": {"library": 0, "candidates": "_parent_global"}}
        cases = [
            ({"libraries": [PROVIDER]}, "libraries[0] must be an object"),
            ({"consumers": {}}, "consumers must be a list"),
            ({"imports": []}, "imports must be an object"),
            ({"classes": [{**parent, "library": "0"}]}, "classes[0].library must be an integer"),
            ({"classes": [{**parent, "required": "_f"}]}, "classes[0].required must be a list"),
            ({"consumers": [{**display, "name": 7}]}, "consumers[0].name must be a string"),
            ({"imports": imports}, "imports._parent_global.candidates must be a list"),
            (
                {"synthesized": [{**display, "overrides": {"start": {"selectors": "_f"}}}]},
                "synthesized[0].overrides.start.selectors must be a list",
            ),
            (
                {"synthesized": [{**display, "overrides": {"start": {"required": "yes"}}}]},
                "synthesized[0].overrides.start.required must be true or false",
            ),
        ]
        path, layouts = self.root / "shape.json", self.root / "shape-layouts.json"
        for members, reason in cases:
            path.write_text(json.dumps({**fixture_declaration(), **members}))
            self.assertRefused(run("probe", path, layouts), f"declaration {reason}")
        path.write_text("[]")
        self.assertRefused(run("probe", path, layouts), "declaration root must be an object")

    def test_a_declaration_nested_too_deep(self):
        # 64 levels parse, and the declaration is then refused for its shape; a 65th,
        # or a depth that would exhaust the stack, is refused by name.
        path, layouts = self.root / "nested.json", self.root / "nested-layouts.json"
        for depth in (64, 65, 100_000):
            path.write_text('{"libraries": ' + "[" * (depth - 1) + "]" * (depth - 1) + "}")
            process = run("probe", path, layouts)
            if depth == 64:
                self.assertRefused(process, "declaration libraries[0] must be an object")
            else:
                self.assertEqual((process.returncode, process.stdout), (1, ""))
                self.assertIn(
                    f"consumer: {path}: arrays and objects nested deeper than 64 levels at line 1",
                    process.stderr,
                )

    def test_an_info_plist_nested_too_deep(self):
        # Inside the Info.plist's root dictionary, 63 more levels parse, and a 64th,
        # or a depth that would exhaust the stack, is refused by name.
        self.assertEqual(self.inputs.build("nested").returncode, 0)
        work = self.inputs.root / "nested/graft"
        info = (work / "Unbound.kext/Contents/Info.plist").read_text()
        path = self.root / "nested.plist"
        for depth in (63, 64, 100_000):
            nested = "<key>Nested</key>" + "<array>" * depth + "</array>" * depth
            path.write_text(info.replace("<dict>", "<dict>" + nested, 1))
            process = run(
                "reserve",
                work / "census",
                self.inputs.declaration,
                work / "layouts.json",
                path,
                self.root / "nested-reserve.bin",
            )
            if depth == 63:
                self.assertEqual((process.returncode, process.stderr), (0, ""))
            else:
                self.assertEqual((process.returncode, process.stdout), (1, ""))
                self.assertIn(
                    f"consumer: {path}: dictionaries and arrays nested deeper than 64 levels",
                    process.stderr,
                )

    def test_two_consumer_classes_of_one_name(self):
        # Census finds each class's vtable by its name, so two of one name would share it.
        config = fixture_declaration()
        twice = copy.deepcopy(config)
        twice["consumers"].append(dict(config["consumers"][0]))
        synthesized = copy.deepcopy(config)
        synthesized["synthesized"] = [{**config["consumers"][0], "overrides": {"changed": {}}}]
        for name, declaration in (("twice", twice), ("synthesized", synthesized)):
            path = self.inputs.write(f"{name}.json", json.dumps(declaration).encode())
            process = run("probe", path, self.root / f"{name}-layouts.json")
            self.assertRefused(process, "duplicate consumer class")

    def test_a_personality_that_names_no_provider_class(self):
        info = fixture_info()
        del info["IOKitPersonalities"]["Fixture"]["IOProviderClass"]
        path = self.inputs.write("no-provider-class.plist", plistlib.dumps(info))
        process = self.inputs.build("no-provider-class", info_plist=path)
        self.assertRefused(process, "consumer personality Fixture names no IOProviderClass")
        self.assertFalse((self.inputs.root / "no-provider-class/Fixture.kext").exists())

    def test_a_personality_that_already_carries_the_gate(self):
        # As a staged Info.plist's do: its provider matching would then match only nubs
        # whose gate is already open, so Graft would never open one.
        key = f"{IDENTIFIER}.graft-bound"
        reason = (
            f"consumer personality Fixture already carries the activation key {key} in its "
            "IOPropertyMatch; give the Info.plist as written, not a staged one"
        )
        cases = (("gated", {key: True}), ("gated-list", [{"model": "one"}, {key: False}]))
        for name, match in cases:
            info = fixture_info()
            info["IOKitPersonalities"]["Fixture"]["IOPropertyMatch"] = match
            path = self.inputs.write(f"{name}.plist", plistlib.dumps(info))
            process = self.inputs.build(name, info_plist=path)
            self.assertRefused(process, reason)
            self.assertFalse((self.inputs.root / f"{name}/Fixture.kext").exists())

    def test_compiler_flags_for_another_architecture(self):
        process = self.inputs.build("other-arch", flags=["-arch", "arm64e", f"-I{FIXTURE}"])
        self.assertRefused(process, "the compiler flags target arm64e, not ARCH's x86_64")
        self.assertFalse((self.inputs.root / "other-arch").exists())

    def test_a_class_the_compiler_cannot_lay_out(self):
        config = fixture_declaration()
        for row in config["classes"] + config["consumers"]:
            row["header"] = "Missing.h"
        path = self.inputs.write("missing-header.json", json.dumps(config).encode())
        process = self.inputs.build("missing-header", declaration=path)
        self.assertRefused(process, "consumer layout compilation failed")

    def test_the_probe_compiles_under_tmpdir(self):
        scratch = self.root / "scratch"
        scratch.mkdir()
        config = fixture_declaration()
        for row in config["classes"] + config["consumers"]:
            row["header"] = "Missing.h"
        path = self.inputs.write("tmpdir.json", json.dumps(config).encode())
        environment = {**os.environ, "TMPDIR": str(scratch)}
        layouts = self.root / "tmpdir-layouts.json"
        process = run("probe", path, layouts, "--", *self.inputs.flags, environment=environment)
        # The compiler's error names the probe's source, in its directory under TMPDIR,
        # which is gone once the probe is.
        self.assertRefused(process, "consumer layout compilation failed")
        self.assertIn(f"{scratch}/graft-layout-", process.stderr)
        self.assertEqual(list(scratch.iterdir()), [])
        environment["TMPDIR"] = str(self.root / "missing")
        process = run("probe", path, layouts, "--", *self.inputs.flags, environment=environment)
        self.assertEqual((process.returncode, process.stdout), (1, ""))
        self.assertTrue(
            process.stderr.startswith("consumer: cannot use the temporary directory: "),
            process.stderr,
        )

    def test_a_link_that_fails(self):
        missing = self.inputs.root / "missing.o"
        process = self.inputs.build("missing-object", objects=[missing])
        census = self.inputs.root / "missing-object/graft/census"
        command = f"xcrun clang++ -arch x86_64 -nostdlib -Xlinker -kext {missing} -lcc_kext"
        self.assertRefused(process, f"{command} -o {census} failed")

    def test_an_info_plist_that_does_not_lint(self):
        path = self.inputs.write("broken.plist", b"<plist><dict><key>a</key></dict></plist>\n")
        process = self.inputs.build("broken", info_plist=path)
        versioned = self.inputs.root / "broken/graft/Unbound.kext/Contents/Info.plist"
        self.assertRefused(process, f"plutil -lint {versioned} failed")

    def test_layouts_of_another_declaration(self):
        self.assertEqual(self.inputs.build("other").returncode, 0)
        work = self.inputs.root / "other/graft"
        layouts = json.loads((work / "layouts.json").read_text())
        layouts["classes"][0]["name"] = "Other"
        other = self.inputs.write("other-layouts.json", json.dumps(layouts).encode())
        process = run(
            "reserve",
            work / "census",
            self.inputs.declaration,
            other,
            work / "Unbound.kext/Contents/Info.plist",
            self.root / "other-reserve.bin",
        )
        self.assertRefused(process, "layouts describe another declaration")

    def test_an_existing_staging_output(self):
        self.assertEqual(self.inputs.build("again").returncode, 0)
        work = self.inputs.root / "again/graft"
        process = run(
            "stage",
            work / "Unbound.kext",
            self.inputs.declaration,
            work / "layouts.json",
            self.inputs.root / "again/Fixture.kext",
        )
        self.assertRefused(process, "staging output already exists")

    def test_a_malformed_command_line(self):
        work = self.root / "usage"
        for arguments in (
            [],
            ["link"],
            ["build", "--name=Fixture"],
            [
                "build",
                *(f"--{key}={work}" for key in ("name", "version", "arch")),
                *(f"--{key}={work}" for key in ("declaration", "info-plist", "work", "output")),
            ],
            ["stage", work],
            ["check-prelinked", work],
        ):
            process = run(*arguments)
            self.assertEqual((process.returncode, process.stdout), (2, ""), arguments)
            self.assertTrue(process.stderr.startswith("usage: graft-kext"), process.stderr)
        self.assertFalse(work.exists())

    def test_help(self):
        process = run("--help")
        self.assertEqual((process.returncode, process.stderr), (0, ""))
        self.assertTrue(process.stdout.startswith("usage: graft-kext build"), process.stdout)
        self.assertIn(
            "check-prelinked checks that COLLECTION, an x86-64 kernel collection OpenCore\n"
            "prelinked Graft and the kexts it serves into, keeps each staged consumer KEXT\n"
            "bindable.\n",
            process.stdout,
        )


if __name__ == "__main__":
    unittest.main()
