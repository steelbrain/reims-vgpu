#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later
"""Cold-boot Graft as this checkout builds it, and optional test fixtures, on a Linux clone.

    python3 -I -B tools/internal/boot-core.py BASE --env DIR --gpu NAME --account FILE
                                              [--probe KEXT] [--hello-consumer KEXT]

Run on a Linux host with reims-vgpu's VM tool, reims-ctl, and its QEMU installed under DIR
(DIR/cargo/bin/reims-ctl and DIR/data/reims-ctl). BASE is an installed x86-64 base
directory (disk.qcow2, opencore.qcow2 and its OVMF files), never booted in place. FILE
holds the base's admin account: its name on the first line, its password on the second.
The script:
1. refuses a source another process can write, and copies it with reims-ctl create;
2. replaces the copy's injected kexts with build/Graft.kext and the optional fixtures,
   clears its boot-args and csr-active-config, and checks that the injected Graft is
   the one built here;
3. boots it on reims-vgpu's device, headless, logs in if the base stops at loginwindow,
   and waits for the desktop;
4. requires Graft active, each fixture passed, SIP enabled, csr-active-config 0, no
   boot-args, and no `reims-vgpu: fatal` line on the host.
The copy is stopped and deleted even on a failed check. Results, screenshots and logs
stay in build/boot-core; the final screenshot needs inspecting.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import plistlib
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "build/boot-core"
GRAFT = ROOT / "build/Graft.kext"
# Apple's NVRAM variable GUID, which holds boot-args and csr-active-config.
APPLE_NVRAM_GUID = "7C436110-AB2A-4BBB-A880-FE41995C9F82"
SSH_UP_TIMEOUT = 480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "base", type=Path, help="installed x86-64 base directory; never booted in place"
    )
    parser.add_argument(
        "--probe", type=Path, help="test-only GraftCoreProbe.kext, built on the Mac"
    )
    parser.add_argument(
        "--hello-consumer", type=Path, help="test-only dormant GraftHelloConsumer.kext"
    )
    parser.add_argument(
        "--env", type=Path, required=True, help="directory holding reims-ctl and its QEMU"
    )
    parser.add_argument(
        "--gpu", required=True, help="reims-ctl's name for the host GPU the device renders on"
    )
    parser.add_argument(
        "--account", type=Path, required=True, help="file: account name, then its password"
    )
    return parser.parse_args()


class Account(NamedTuple):
    """The base's admin account, which the script logs in as and runs sudo with."""

    user: str
    password: str


def read_account(path: Path) -> Account:
    """The account in path. The password is typed at loginwindow over QMP one key per
    character, which this script does only for lowercase letters and digits."""
    lines = path.read_text().splitlines()
    if len(lines) < 2 or not lines[0] or not re.fullmatch(r"[a-z0-9]+", lines[1]):
        raise SystemExit("the account file needs a name, then a lowercase password")
    return Account(lines[0], lines[1])


def fixture(path: Path | None, identifier: str, name: str, refusal: str) -> Path | None:
    """The resolved fixture bundle, which must be name and carry identifier, if given."""
    if path is None:
        return None
    path = path.resolve()
    info = plistlib.loads((path / "Contents/Info.plist").read_bytes())
    if info.get("CFBundleIdentifier") != identifier or path.name != name:
        raise SystemExit(refusal)
    return path


def refuse_writers(sources: set[Path]) -> None:
    """Refuse a source that any process holds open for writing.

    A read-only opener, such as a VM booted with snapshot=on, cannot change the bytes the
    copy reads, so it does not block the copy.
    """
    for process in Path("/proc").glob("[0-9]*"):
        try:
            for fd in (process / "fd").iterdir():
                try:
                    if fd.resolve() not in sources:
                        continue
                    info = (process / "fdinfo" / fd.name).read_text()
                except OSError:
                    continue
                flags = next(
                    int(line.split()[1], 8)
                    for line in info.splitlines()
                    if line.startswith("flags:")
                )
                if flags & os.O_ACCMODE != os.O_RDONLY:
                    raise SystemExit(
                        f"{fd.resolve().name} is open for writing; refusing to copy it"
                    )
        except OSError:
            pass


def first_partition_offset(disk: Path) -> int:
    """The byte offset of a raw GPT disk's first partition.

    The GPT header, at LBA 1, holds the partition array's LBA at byte 72; the array's first
    entry holds its partition's starting LBA at byte 32.
    """
    with disk.open("rb") as f:
        f.seek(512 + 72)
        table = struct.unpack("<Q", f.read(8))[0] * 512
        f.seek(table + 32)
        return struct.unpack("<Q", f.read(8))[0] * 512


def kernel_add_entry(bundle: str, executable: str, comment: str) -> dict[str, Any]:
    """An OpenCore Kernel > Add entry for an x86-64 kext, on every kernel."""
    return {
        "Arch": "x86_64",
        "BundlePath": bundle,
        "Comment": comment,
        "Enabled": True,
        "ExecutablePath": executable,
        "MaxKernel": "",
        "MinKernel": "",
        "PlistPath": "Contents/Info.plist",
    }


class Clone:
    """The base's copy as a reims-ctl VM, and the ways the script reaches it."""

    def __init__(self, name: str, install: Path, account: Account) -> None:
        self.name = name
        self.vm = install / "vms" / name
        self.port = 0
        self.account = account
        self.reims_ctl = install / "cargo/bin/reims-ctl"
        self.qemu_img = install / "data/reims-ctl/qemu/current/bin/qemu-img"
        self.env = os.environ.copy()
        self.env.update(
            REIMS_CTL_CONFIG_DIR=str(OUT / "config"),
            REIMS_CTL_DATA_DIR=str(install / "data/reims-ctl"),
        )
        # The device remains headless; QMP provides the screenshot.
        self.env.pop("DISPLAY", None)
        self.env.pop("WAYLAND_DISPLAY", None)
        self.env["PATH"] = str(self.qemu_img.parent) + ":" + self.env["PATH"]

    def run(self, command: list[Any], **kwargs: Any) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(x) for x in command], env=self.env, check=True, text=True, **kwargs
        )

    def ctl(self, *arguments: Any, **kwargs: Any) -> subprocess.CompletedProcess:
        return self.run([self.reims_ctl, *arguments], **kwargs)

    def qmp(self, command: str, arguments: dict[str, Any] | None = None) -> Any:
        """Send one QMP command to the VM and return its result."""
        with socket.socket(socket.AF_UNIX) as s:
            s.settimeout(15)
            s.connect(str(self.vm / "qmp.sock"))
            f = s.makefile("rwb", buffering=0)
            f.readline()

            def request(name: str, request_arguments: dict[str, Any] | None) -> Any:
                message = {"execute": name}
                if request_arguments is not None:
                    message["arguments"] = request_arguments
                f.write(json.dumps(message).encode() + b"\n")
                while True:
                    line = f.readline()
                    if not line:
                        raise RuntimeError("QMP closed")
                    reply = json.loads(line)
                    if "error" in reply:
                        raise RuntimeError(str(reply["error"]))
                    if "return" in reply:
                        return reply["return"]

            request("qmp_capabilities", None)
            return request(command, arguments)

    def screendump(self, name: str) -> None:
        self.qmp("screendump", {"filename": str(OUT / name)})

    def ssh(self, command: str) -> subprocess.CompletedProcess:
        """Run command in the guest as the test user; never raises for its exit status."""
        return subprocess.run(
            [
                *("ssh", "-n", "-p", str(self.port)),
                *("-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"),
                *("-o", "ConnectTimeout=5", "-o", "LogLevel=ERROR"),
                f"{self.account.user}@127.0.0.1",
                command,
            ],
            env={
                **self.env,
                "SSH_ASKPASS": str(OUT / "askpass"),
                "SSH_ASKPASS_REQUIRE": "force",
                "DISPLAY": "none",
            },
            start_new_session=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=90,
            check=False,
        )

    def registry_root(self) -> dict[str, Any]:
        """The guest's IORegistry root properties."""
        registry = self.ssh("ioreg -a -l -d 1")
        if registry.returncode:
            raise RuntimeError("cannot read registry root")
        properties = plistlib.loads(registry.stdout.encode())
        if isinstance(properties, list) and len(properties) == 1:
            properties = properties[0]
        if not isinstance(properties, dict):
            raise RuntimeError("registry archive does not contain one root dictionary")
        return properties

    def graft_properties(self) -> dict[str, Any]:
        return {
            key: value for key, value in self.registry_root().items() if key.startswith("Graft")
        }

    def create(self, disk: Path, opencore: Path, gpu: str) -> None:
        """Copy the base into a new reims-ctl VM on our device."""
        self.vm.parent.mkdir(parents=True, exist_ok=True)
        self.ctl(
            *("create", self.name, "--dir", self.vm),
            *("--disk", disk, "--disk-mode", "copy", "--opencore", opencore),
            *("--memory", "4G", "--cpus", "6", "--graphics", "reims-vgpu", "--gpu", gpu),
            *("--ssh-port", self.port, "--logs", "true", "--audio", "false"),
        )

    def inject(self, probe: Path | None, consumer: Path | None) -> None:
        """Make Graft and the given fixtures the copy's only injected kexts, with SIP's
        NVRAM defaults, and check that the injected Graft is build/Graft.kext."""
        raw = OUT / "oc.raw"
        self.run([self.qemu_img, "convert", "-O", "raw", self.vm / "opencore.qcow2", raw])
        image = str(raw) + "@@" + str(first_partition_offset(raw))
        config_path = OUT / "config.plist"
        self.run(["mcopy", "-n", "-i", image, "::/EFI/OC/config.plist", config_path])
        config = plistlib.loads(config_path.read_bytes())
        config["Kernel"]["Add"] = [
            kernel_add_entry("Graft.kext", "Contents/MacOS/Graft", "Graft core boot check")
        ]
        if probe:
            config["Kernel"]["Add"].append(
                kernel_add_entry(probe.name, "Contents/MacOS/GraftCoreProbe", "Graft core probe")
            )
        if consumer:
            config["Kernel"]["Add"].append(
                kernel_add_entry(
                    consumer.name, "Contents/MacOS/GraftHelloConsumer", "Graft hello consumer"
                )
            )
        nvram = config["NVRAM"]["Add"].setdefault(APPLE_NVRAM_GUID, {})
        nvram.update({"boot-args": "", "csr-active-config": bytes(4)})
        delete = config["NVRAM"]["Delete"].setdefault(APPLE_NVRAM_GUID, [])
        for key in ("boot-args", "csr-active-config"):
            if key not in delete:
                delete.append(key)
        config_path.write_bytes(plistlib.dumps(config))
        self.run(["mcopy", "-o", "-n", "-i", image, config_path, "::/EFI/OC/config.plist"])
        self.run(["mdeltree", "-i", image, "::/EFI/OC/Kexts"])
        self.run(["mmd", "-i", image, "::/EFI/OC/Kexts"])
        self.run(["mcopy", "-s", "-i", image, GRAFT, "::/EFI/OC/Kexts/"])
        if probe:
            self.run(["mcopy", "-s", "-i", image, probe, "::/EFI/OC/Kexts/"])
        if consumer:
            self.run(["mcopy", "-s", "-i", image, consumer, "::/EFI/OC/Kexts/"])
        # Check the exact bytes that will be injected before booting.
        readback = OUT / "Graft-readback"
        executable = "::/EFI/OC/Kexts/Graft.kext/Contents/MacOS/Graft"
        self.run(["mcopy", "-n", "-i", image, executable, readback])
        if readback.read_bytes() != (GRAFT / "Contents/MacOS/Graft").read_bytes():
            raise RuntimeError("the injected Graft differs from build/Graft.kext")
        print("Graft sha256:", hashlib.sha256(readback.read_bytes()).hexdigest(), flush=True)
        self.run([self.qemu_img, "convert", "-O", "qcow2", raw, self.vm / "opencore.qcow2"])

    def wait_for_ssh(self) -> None:
        """Wait for the guest's SSH, printing QMP's view of the VM meanwhile."""
        deadline = time.monotonic() + SSH_UP_TIMEOUT
        while time.monotonic() < deadline:
            if self.ssh("true").returncode == 0:
                return
            try:
                print("QMP:", self.qmp("query-status"), flush=True)
                self.screendump("boot.ppm")
            except (OSError, RuntimeError) as error:
                print("QMP:", error, flush=True)
            time.sleep(10)
        raise RuntimeError(f"guest SSH did not become available within {SSH_UP_TIMEOUT}s")

    def type_password(self) -> None:
        """Type the test user's password at loginwindow over QMP, and press return.

        SSH can start before the password field accepts input, so Command-A first selects
        any partial entry to type over (Control-A only moves the cursor in a password field).
        """
        for keys in [["meta_l", "a"], *[[key] for key in self.account.password], ["ret"]]:
            self.qmp(
                "send-key",
                {"keys": [{"type": "qcode", "data": key} for key in keys], "hold-time": 50},
            )
            time.sleep(0.1)

    def log_in(self) -> tuple[subprocess.CompletedProcess, subprocess.CompletedProcess, dict]:
        """Wait for the test user's desktop, logging in where the base stops at loginwindow.

        Returns the console owner's, desktop processes' and console session's evidence.
        """
        deadline = time.monotonic() + 90
        next_login = 0
        while time.monotonic() < deadline:
            console = self.ssh("stat -f %Su /dev/console")
            owner = console.stdout.strip() if console.returncode == 0 else None
            user = self.account.user
            if owner == user:
                desktop = self.ssh(f"pgrep -u {user} -x Finder && pgrep -u {user} -x Dock")
                sessions = self.registry_root().get("IOConsoleUsers", [])
                session = next(
                    (
                        s
                        for s in sessions
                        if s.get("kCGSSessionUserNameKey") == user
                        and s.get("kCGSSessionOnConsoleKey") is True
                        and s.get("kCGSessionLoginDoneKey") is True
                    ),
                    None,
                )
                if desktop.returncode == 0 and session:
                    return console, desktop, session
            elif (
                owner == "root"
                and time.monotonic() >= next_login
                and self.ssh("pgrep -x loginwindow").returncode == 0
            ):
                self.type_password()
                next_login = time.monotonic() + 10
            time.sleep(2)
        raise RuntimeError("guest has not logged in to the desktop")

    def wait_for_fixtures(self, probe: Path | None, consumer: Path | None) -> dict[str, Any]:
        """Graft's registry state, once each given fixture has passed or been refused."""
        state = self.graft_properties()
        if not (probe or consumer):
            return state
        deadline = time.monotonic() + 145
        while time.monotonic() < deadline and state.get("GraftState") != "refused":
            routes_done = not probe or state.get("GraftProbeState", "").startswith(
                ("passed", "refused")
            )
            consumer_done = (
                not consumer
                or state.get("GraftHelloConsumerState", "").startswith(("passed", "refused"))
                or state.get("GraftClient.ai.aneesiqbal.ReimsGraphics", "").startswith("refused")
            )
            if routes_done and consumer_done:
                break
            time.sleep(2)
            state = self.graft_properties()
        return state

    def tear_down(self) -> None:
        """Stop the copy, keep its logs and delete it."""
        with contextlib.suppress(OSError, RuntimeError):
            self.screendump("last.ppm")
        try:
            self.ctl("stop", self.name, "--force")
        finally:
            if (self.vm / "logs").exists():
                shutil.copytree(self.vm / "logs", OUT / "logs")
            self.ctl("delete", self.name, "--yes")


def interrupted(signum: int, _frame: object) -> None:
    raise SystemExit(128 + signum)


def main() -> None:
    args = parse_args()
    base = args.base.resolve()
    probe = fixture(
        args.probe,
        "ai.aneesiqbal.graft.CoreProbe",
        "GraftCoreProbe.kext",
        "the optional probe must be GraftCoreProbe.kext",
    )
    consumer = fixture(
        args.hello_consumer,
        "ai.aneesiqbal.ReimsGraphics",
        "GraftHelloConsumer.kext",
        "the hello consumer must be GraftHelloConsumer.kext",
    )
    if sys.platform != "linux":
        raise SystemExit("VM boots run on Linux")
    account = read_account(args.account)
    clone = Clone(f"graft-core-{os.getpid()}", args.env.resolve(), account)
    if OUT.exists():
        raise SystemExit("build/boot-core already exists; use a fresh workspace")
    disk = (base / "disk.qcow2").resolve()
    sources = {disk, (base / "opencore.qcow2").resolve()}
    if not all(source.is_file() for source in sources):
        raise SystemExit("base disk or OpenCore image is missing")
    refuse_writers(sources)
    OUT.mkdir(parents=True)
    askpass = OUT / "askpass"
    askpass.write_text(f"#!/bin/sh\nprintf '%s\\n' {account.password}\n")
    askpass.chmod(0o700)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        clone.port = s.getsockname()[1]
    created = False
    signal.signal(signal.SIGTERM, interrupted)
    try:
        clone.create(disk, base / "opencore.qcow2", args.gpu)
        created = True
        clone.ctl("set", clone.name, "--env", "REIMS_VGPU_WINDOW=0")
        clone.inject(probe, consumer)
        with (OUT / "command.sh").open("w") as command:
            clone.ctl("command", clone.name, stdout=command)
        clone.ctl("start", clone.name)
        clone.wait_for_ssh()
        guest = clone.ssh(
            "sw_vers; uptime; csrutil status; nvram csr-active-config; nvram boot-args; "
            f"printf '{account.password}\\n' | sudo -S -p '' kmutil showloaded --list-only 2>&1 "
            "| grep -Ei 'graft|lilu|reims|rgfx'; "
            "log show --last boot --style compact --predicate "
            '\'process == "kernel" AND eventMessage CONTAINS[c] "ai.aneesiqbal.graft"\' '
            "2>/dev/null | tail -30; stat -f %Su /dev/console"
        )
        (OUT / "probe.txt").write_text(guest.stdout)
        print(guest.stdout, flush=True)
        # Some test bases deliberately stop at loginwindow; no guest installation step is used.
        console, desktop, session = clone.log_in()
        # WindowServer can still show loginwindow after the session reports done. Allow
        # presentation to catch up; the saved screenshot needs inspection.
        time.sleep(20)
        clone.screendump("desktop.ppm")
        (OUT / "console.txt").write_text(console.stdout)
        (OUT / "desktop-processes.txt").write_text(desktop.stdout)
        (OUT / "desktop-session.json").write_text(json.dumps(session, indent=2) + "\n")
        print("Graft desktop console:", console.stdout.strip(), flush=True)
        state = clone.wait_for_fixtures(probe, consumer)
        (OUT / "registry.json").write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps(state, indent=2), flush=True)
        clone.screendump("desktop.ppm")
        if state.get("GraftState") != "active":
            raise RuntimeError("Graft did not report active")
        if probe and state.get("GraftProbeState") != "passed":
            raise RuntimeError("core probe did not pass")
        if consumer and (
            state.get("GraftHelloConsumerState") != "passed"
            or not state.get("GraftClient.ai.aneesiqbal.ReimsGraphics", "").startswith("bound:")
        ):
            raise RuntimeError("hello consumer did not pass")
        if "System Integrity Protection status: enabled." not in guest.stdout:
            raise RuntimeError("guest did not confirm SIP enabled")
        qemu_log = (clone.vm / "logs/qemu.log").read_text(errors="replace")
        if "reims-vgpu: fatal" in qemu_log:
            raise RuntimeError("host device reported a fatal error")
        bootargs = clone.ssh("sysctl -n kern.bootargs")
        if bootargs.returncode or bootargs.stdout.strip():
            raise RuntimeError("guest has boot-args")
        if "csr-active-config\t%00%00%00%00" not in guest.stdout:
            raise RuntimeError("guest csr-active-config is not zero")
        print("GRAFT_CORE_BOOT_OK", flush=True)
    finally:
        if created:
            clone.tear_down()


if __name__ == "__main__":
    main()
