#!/usr/bin/env python3
"""vgpu-dashboard.py — a local control panel for the reims-vgpu VM pathways.

Launch a boot, pick USB devices to pass through, choose the network mode, and
watch the device's own censuses while it runs — in one page, instead of six
terminals and a remembered set of env knobs.

    scripts/vgpu-dashboard/vgpu-dashboard.py            # then open the printed URL
    scripts/vgpu-dashboard/vgpu-dashboard.py --port 8765
    scripts/vgpu-dashboard/vgpu-dashboard.py --selftest # API smoke test, no browser

WHAT THIS IS AND IS NOT. It is an *instrument*: it observes host state and drives
the existing scripts. It is deliberately not a second implementation of any rule
this repo already owns. Rails come from `boot-x86.sh --list-rails`, USB specs
from `boot-x86.sh --list-usb` (which calls vm/lib/usb-passthrough.sh), the guest
address from `vm/guest-ip.sh`, screenshots from the host helper. When one of
those changes, this dashboard changes with it and cannot disagree with it. The
one thing it does parse itself is the fail log, because nothing else does.

READING THE FAIL LOG IS WHERE A DASHBOARD GOES WRONG, so the four traps that
`AGENTS.md` records are implemented here rather than left to the reader:

  * The channel comes first. `OFF ` records carry `reason=` too, for ordering
    and control-flow events that are not losses, so ranking `reason=` without
    splitting the channel inverts the queue. `_fail_channel_lines` does the
    split; the UI shows the two counts separately and never sums them.
  * Per-window and cumulative counters are different objects. `store_routes`
    resets every census interval, so a boot total is the SUM of its samples and
    `tail -1` reads three to four times low. `registry_pressure` is the
    opposite — the device labels it "(levels, not per-interval)" and the last
    sample IS the answer. Summing that one is the error. Each series is tagged
    with which it is and reduced accordingly.
  * `present_hz` alone says nothing. It is a reading of the presenter AND of the
    device's publish rate, so it is always reported beside `offered_hz`, with
    `busy_fence`/`busy_acquire` next to them: both up is a win, offered up alone
    means the presenter has become a ceiling.
  * A log may hold several boots of several builds. `vk_caps` appears once per
    device creation, so counting it is the boot count, and more than one means
    every ranking below mixes builds. The UI says so loudly, because
    `first_sight` latches per process and a stale log inflates in a way that
    reads as a finding rather than as noise.

Boots are also scored the way the repo scores them: a guest kernel panic is
grepped out of the boot's own stdout and OUTRANKS a probe's exit status, and the
60 Hz / free-running population a boot latched is surfaced from `present_hz`,
because per-draw numbers are not comparable across the two.

SAFETY. This server execs QEMU and the repo's scripts, so it binds 127.0.0.1
only and every request must carry the random token printed at startup. Nothing
is ever passed to a shell: every child is an argv list, and a USB spec is
re-validated against the same three forms the library accepts before it is
allowed near a command line. Requests that would launch something are POST-only
so a stray browser prefetch cannot start a VM.

Stdlib only, on purpose — this repo has no Python dependency and this is not the
place to introduce one.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import re
import secrets
import shlex
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent

FAIL_LOG = Path(os.environ.get("REIMS_VGPU_FAIL_LOG", "/tmp/reims-vgpu-fail.log"))

# How much of the tail to parse for live metrics. The log is append-only and
# routinely reaches a gigabyte on a long boot, so it is never read whole: the
# censuses are 1 Hz, which makes a few megabytes several minutes of history and
# a full read a stall. Boot counting is the one question that needs the whole
# file, and it is cached (see `_boot_count`).
LOG_TAIL_BYTES = int(os.environ.get("REIMS_VGPU_DASH_TAIL", 6 << 20))

# The census tags this dashboard understands, and — the load-bearing half — how
# each series must be reduced. "window" resets every interval, so a boot total is
# the sum and the last sample is one window. "levels" is a cumulative high-water
# where the last sample IS the answer and summing is meaningless. Getting this
# backwards is the single easiest way to publish a wrong number, so it is data
# here rather than a habit at each call site.
CENSUS_REDUCE = {
    "host_window_cadence": "window",
    "drain_duty": "window",
    "gpu_span": "window",
    "window_publish": "window",
    "store_routes": "window",
    "display_vbl": "levels",
    "registry_pressure": "levels",
    "vk_caps": "levels",
}

# A macOS guest that latched a 16 666 666 ns frame period paces at exactly 60 Hz
# for its whole life; one that latched 0 free-runs and is work-limited at
# 95-117 Hz. The gap between them is empty on every boot on record, so a single
# threshold classifies a boot — and per-draw numbers are not comparable across
# the two populations, which is why the UI labels every reading with it.
PACED_HZ_CEILING = 75.0


def _run(argv, timeout=30, cwd=None, env=None):
    """Run an argv list and capture it. Never a shell — nothing here interpolates."""
    merged = dict(os.environ)
    if env:
        merged.update({k: str(v) for k, v in env.items()})
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd or REPO_ROOT),
            env=merged,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            "argv": argv,
            "rc": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "argv": argv,
            "rc": None,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + f"\n[timed out after {timeout}s]",
            "timeout": True,
        }
    except FileNotFoundError as exc:
        return {"argv": argv, "rc": None, "stdout": "", "stderr": str(exc), "timeout": False}


# --------------------------------------------------------------------- host state

def host_pathway():
    """Which boot script this host can actually run.

    Not a preference: boot-x86.sh needs KVM and boot-arm64.sh needs HVF, so the
    host decides and the UI must not offer the other one as if it would work.
    """
    system = os.uname().sysname
    if system == "Darwin":
        return "arm64"
    return "x86"


def kernel_module_state():
    """Whether a module can load at all, which gates bridge networking.

    `tun` is a module on every distro kernel that has it, so it loads on demand —
    and CANNOT when the running kernel's module tree is absent, which is what a
    kernel upgrade without a reboot leaves behind. That state breaks every
    bridged netdev while lsmod/modinfo/modprobe all report something that reads
    like a missing package, so the diagnosis is worth surfacing by name.
    """
    release = os.uname().release
    tree = Path("/lib/modules") / release
    installed = sorted(p.name for p in Path("/lib/modules").glob("*")) if Path("/lib/modules").is_dir() else []
    return {
        "release": release,
        "module_tree_present": tree.is_dir(),
        "installed_trees": installed,
        # Computed here rather than in the page, so the UI renders one fact
        # instead of re-deriving a packaging rule in JavaScript.
        "cached_package": cached_kernel_package(release),
    }


def cached_kernel_package(release):
    """The package-cache archive holding the RUNNING kernel's own modules, if kept.

    A kernel replaced on disk usually leaves its package behind in the cache, and
    that archive carries the module tree for the kernel still running. `tun`
    declares no dependencies, so one `.ko` out of it loads with `insmod` — no
    depmod, no restored tree, no reboot. Returns None unless the archive is
    actually there: a hint naming a file nobody has is worse than no hint.

    Arch spells the release `7.1.8-arch1-3` as `7.1.8.arch1-3` in the package
    name. On packaging this does not recognise, it simply finds nothing.
    """
    cache = Path("/var/cache/pacman/pkg")
    if not cache.is_dir():
        return None
    matches = sorted(cache.glob(f"linux-{release.replace('-arch', '.arch')}-*.pkg.tar.zst"))
    if not matches:
        return None
    module = f"usr/lib/modules/{release}/kernel/drivers/net/tun.ko.zst"
    return {
        "package": str(matches[0]),
        "module_path_in_package": module,
        "commands": [
            'd=$(mktemp -d)',
            f'bsdtar -C "$d" -xf {matches[0]} {module}',
            f'unzstd "$d/{module}" -o "$d/tun.ko"',
            'sudo insmod "$d/tun.ko"',
        ],
        "undo": "sudo rmmod tun",
    }


def tun_available():
    """Open /dev/net/tun rather than stat it.

    The node existing is not the check: a missing tun driver leaves the node in
    place and fails the OPEN with ENODEV, which is exactly the state that makes
    `-netdev bridge` die inside the helper with no explanation.
    """
    try:
        fd = os.open("/dev/net/tun", os.O_RDWR)
    except OSError as exc:
        return {"ok": False, "error": f"{exc.strerror} ({exc.errno})"}
    os.close(fd)
    return {"ok": True, "error": None}


def bridges():
    """Host bridges, with whether each is a plausible target.

    A bridge with no guest attached reports DOWN/NO-CARRIER; that is normal and
    not a failure, so it is reported as a fact rather than as a warning.
    """
    out = _run(["ip", "-j", "link", "show", "type", "bridge"], timeout=5)
    found = []
    if out["rc"] == 0 and out["stdout"].strip():
        try:
            for link in json.loads(out["stdout"]):
                found.append({
                    "name": link.get("ifname"),
                    "operstate": link.get("operstate"),
                    "flags": link.get("flags", []),
                })
        except json.JSONDecodeError:
            pass
    addrs = {}
    out4 = _run(["ip", "-j", "-4", "addr", "show"], timeout=5)
    if out4["rc"] == 0 and out4["stdout"].strip():
        try:
            for link in json.loads(out4["stdout"]):
                for info in link.get("addr_info", []):
                    if info.get("family") == "inet":
                        addrs.setdefault(link.get("ifname"), []).append(
                            f"{info.get('local')}/{info.get('prefixlen')}")
        except json.JSONDecodeError:
            pass
    for br in found:
        br["addresses"] = addrs.get(br["name"], [])
    return found


def bridge_helper():
    """Locate a qemu-bridge-helper that can actually enslave a tap.

    Mirrors what boot-x86.sh resolves, and for the same reason: a helper that
    cannot get CAP_NET_ADMIN fails inside QEMU with an exit status and no
    explanation, so the UI should say which file was rejected and why before a
    boot is attempted.
    """
    candidates = [
        REPO_ROOT / "vendor/qemu/build/qemu-bridge-helper",
        Path("/usr/lib/qemu/qemu-bridge-helper"),
        Path("/usr/libexec/qemu-bridge-helper"),
        Path("/usr/local/libexec/qemu-bridge-helper"),
        Path("/usr/lib/x86_64-linux-gnu/qemu/qemu-bridge-helper"),
    ]
    unprivileged = None
    for path in candidates:
        if not (path.is_file() and os.access(path, os.X_OK)):
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        if (st.st_mode & 0o4000) and st.st_uid == 0:
            return {"path": str(path), "privileged": True, "how": "setuid root", "rejected": None}
        getcap = shutil.which("getcap")
        if getcap:
            cap = _run([getcap, str(path)], timeout=5)
            if "cap_net_admin" in (cap["stdout"] or ""):
                return {"path": str(path), "privileged": True, "how": "cap_net_admin", "rejected": None}
        unprivileged = unprivileged or str(path)
    return {"path": None, "privileged": False, "how": None, "rejected": unprivileged}


def bridge_acl(bridge):
    """What /etc/qemu/bridge.conf says about this bridge.

    The helper is the authority — it honours allow/deny/all and follows
    `include` lines — so this reads the obvious file and reports, rather than
    becoming a second copy of the rule that can disagree with the first.
    """
    conf = Path("/etc/qemu/bridge.conf")
    if not conf.is_file():
        return {"file": str(conf), "readable": False, "allows": None, "lines": []}
    try:
        lines = [ln.strip() for ln in conf.read_text().splitlines() if ln.strip()]
    except OSError:
        return {"file": str(conf), "readable": False, "allows": None, "lines": []}
    allows = any(
        re.fullmatch(rf"allow\s+({re.escape(bridge)}|all)", ln) for ln in lines
    )
    return {"file": str(conf), "readable": True, "allows": allows, "lines": lines}


def load_average():
    """Load, because every `us=` number this device reports is wall clock.

    A driven boot taken while a build or a second VM runs measures the harness as
    much as the device — it has been measured to halve throughput and invert the
    ranking between the two largest costs — and the log looks perfectly healthy
    either way. Counts survive contention; timings do not.
    """
    try:
        one, five, fifteen = os.getloadavg()
    except OSError:
        return None
    return {"1m": round(one, 2), "5m": round(five, 2), "15m": round(fifteen, 2),
            "cpus": os.cpu_count()}


# ------------------------------------------------------------------ the fail log

# `key=N` and `key=N.N`. Some census fields are a pair — `resample_peak_ms=60007/2000`,
# `slab_mib=1025/1264` — where the first number is the reading and the second is
# the cap it is measured against; both are kept, the second as `<key>_cap`.
_KV = re.compile(r"([a-z_][a-z_0-9]*)=(-?[0-9]+(?:\.[0-9]+)?)(?:/(-?[0-9]+(?:\.[0-9]+)?))?")
_TAG = re.compile(r"^(OFF )?([a-z_][a-z_0-9]*)\b")

# The full-scan cache. `vk_caps` is emitted once per device creation, at the
# START of each boot, so a tail window of any practical size does not contain it
# on a long-running boot — the structural facts about the host GPU have to come
# from the same pass that counts boots, or they come back empty.
_scan_cache = {"mtime": None, "size": None, "count": None, "vk_caps_line": None}


def _tail_text(path, nbytes):
    """Last nbytes of the log, starting at a line boundary."""
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > nbytes:
                fh.seek(size - nbytes)
                fh.readline()          # discard the partial first line
            data = fh.read()
    except OSError:
        return "", 0
    return data.decode("utf-8", "replace"), size


def _split_channel(line):
    """(channel, tag) for one record.

    A fail-channel record begins with its own event name; an off-channel one
    begins with the literal `OFF `. This is the split that has to happen before
    anything ranks `reason=`, because OFF records carry `reason=` too — for
    ordering and control-flow events that are not losses — and ranking without
    it inverts the queue.
    """
    match = _TAG.match(line)
    if not match:
        return None, None
    return ("off" if match.group(1) else "fail"), match.group(2)


def _scan_log(path):
    """One full pass for the two questions a tail cannot answer.

    BOOT COUNT, from `vk_caps` — one line per device creation. More than one
    means every ranking below mixes builds, and because `first_sight` latches per
    process a stale log inflates in a way that reads as a finding rather than as
    noise: a stale log ranked naively once named two documented healthy-zero
    decoders as firing ~96 000 times between them, where a clean driven boot put
    both at zero.

    THE NEWEST `vk_caps` LINE, because it is the host's own report of what it
    supports — GPU, memory topology, whether the host-pointer import is live —
    and it is written at boot start, far outside any tail window.

    Cached on (mtime, size): a gigabyte scan is ~1 s and the answer only changes
    when the file does.
    """
    try:
        st = path.stat()
    except OSError:
        return {"count": None, "vk_caps_line": None}
    if (_scan_cache["mtime"], _scan_cache["size"]) == (st.st_mtime, st.st_size):
        return {"count": _scan_cache["count"], "vk_caps_line": _scan_cache["vk_caps_line"]}
    count = 0
    newest = None
    try:
        with path.open("rb") as fh:
            tail = b""
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                buf = tail + chunk
                count += buf.count(b"vk_caps") - tail.count(b"vk_caps")
                idx = buf.rfind(b"vk_caps")
                if idx != -1:
                    start = buf.rfind(b"\n", 0, idx) + 1
                    end = buf.find(b"\n", idx)
                    if end != -1:
                        newest = buf[start:end].decode("utf-8", "replace")
                # Carry the last partial line so a tag split across a chunk
                # boundary is neither missed nor double-counted.
                cut = buf.rfind(b"\n") + 1
                tail = buf[cut:]
    except OSError:
        return {"count": None, "vk_caps_line": None}
    _scan_cache.update(mtime=st.st_mtime, size=st.st_size, count=count, vk_caps_line=newest)
    return {"count": count, "vk_caps_line": newest}


def _parse_vk_caps(line):
    """The host capability line, keeping its string fields.

    `_KV` sees only numbers, and the interesting half here is words — the GPU
    name is quoted, and `memory=discrete` / `host_pointer_import=supported` are
    the two that decide which memory rails a boot even exercises.
    """
    if not line:
        return None
    caps = {"raw": line[:600]}
    caps.update({k: v for k, v, _cap in _KV.findall(line)})
    for key in ("memory", "memory_signal", "host_pointer_import", "type", "api", "baseline"):
        match = re.search(rf"\b{key}=([^\s]+)", line)
        if match:
            caps[key] = match.group(1)
    match = re.search(r'\bname="([^"]*)"', line)
    if match:
        caps["name"] = match.group(1)
    return caps


def _sum_window(samples):
    """Sum a per-window series. `t` is a clock, not a quantity, so it is skipped
    and the band is reported separately."""
    totals = {}
    for sample in samples:
        for key, value in sample.items():
            if key == "t":
                continue
            totals[key] = totals.get(key, 0) + value
    return totals


def _band(entry, t_lo, t_hi):
    """The samples of one per-window series that fall inside a shared `t` band.

    This is the join that has to be by timestamp rather than by line ordinal.
    `drain_duty`, `gpu_span`, `window_publish` and `store_routes` each skip
    different windows, so pairing them by position drifts and pulls idle-desktop
    samples into a driven band — a harness that did it read a driven boot at
    ~31 fps where banding by `t` reads 47-52. Every rate quoted from a bad join
    is wrong in the direction that looks like a device problem.
    """
    if entry is None:
        return None
    inside = [
        smp for smp in entry["_samples"]
        if smp.get("t") is not None and t_lo <= smp["t"] <= t_hi
    ]
    if not inside:
        return None
    return {"samples": len(inside), "sum": _sum_window(inside), "last": inside[-1]}


def parse_log(tail_bytes=None):
    """Parse the tail into per-tag series, reduced the way each tag demands."""
    tail_bytes = tail_bytes or LOG_TAIL_BYTES
    text, total = _tail_text(FAIL_LOG, tail_bytes)
    scan = _scan_log(FAIL_LOG)
    if not text:
        return {
            "present": FAIL_LOG.exists(),
            "path": str(FAIL_LOG),
            "size": total,
            "boots": None,
            "series": {},
            "fail_reasons": [],
            "off_reasons": [],
            "fail_recent": [],
            "counts": {"fail": 0, "off": 0},
            "vk_caps": None,
            "boot_boundary_in_tail": False,
        }

    series = {}
    fail_reasons = {}
    off_reasons = {}
    fail_recent = []
    counts = {"fail": 0, "off": 0}

    for line in text.splitlines():
        channel, tag = _split_channel(line)
        if channel is None:
            continue
        counts[channel] += 1

        reason = None
        rmatch = re.search(r"\breason=([a-z_0-9]+)", line)
        if rmatch:
            reason = rmatch.group(1)
        if channel == "fail":
            if reason:
                fail_reasons[reason] = fail_reasons.get(reason, 0) + 1
            fail_recent.append(line[:400])
        elif reason:
            off_reasons[reason] = off_reasons.get(reason, 0) + 1

        if tag in CENSUS_REDUCE:
            fields = {}
            for key, value, cap in _KV.findall(line):
                fields[key] = float(value) if "." in value else int(value)
                if cap:
                    fields[key + "_cap"] = float(cap) if "." in cap else int(cap)
            series.setdefault(tag, []).append(fields)

    reduced = {}
    boot_boundary_in_tail = False
    for tag, samples in series.items():
        mode = CENSUS_REDUCE[tag]
        # `t` is milliseconds since THIS device's creation, so it restarts at ~0
        # on every boot. A log holding several boots can therefore put two boots
        # inside one tail window, and banding across that boundary mixes builds
        # while every number stays plausible. Keep only the final run in which
        # `t` never goes backwards — that is the newest boot.
        cut = 0
        for i in range(1, len(samples)):
            prev, cur = samples[i - 1].get("t"), samples[i].get("t")
            if prev is not None and cur is not None and cur < prev:
                cut = i
        if cut:
            boot_boundary_in_tail = True
            samples = samples[cut:]
        entry = {
            "mode": mode,
            "samples": len(samples),
            "last": samples[-1],
            "t_lo": samples[0].get("t"),
            "t_hi": samples[-1].get("t"),
            # Kept so metrics() can re-band them against the other tags. Every
            # census carries `t=` and each one SKIPS DIFFERENT WINDOWS, so a
            # cross-tag ratio taken over each tag's own full range silently
            # compares different spans of time.
            "_samples": samples,
        }
        entry["sum"] = _sum_window(samples) if mode == "window" else None
        reduced[tag] = entry

    return {
        "present": True,
        "path": str(FAIL_LOG),
        "size": total,
        "tail_bytes": min(tail_bytes, total),
        "boots": scan["count"],
        "series": reduced,
        "fail_reasons": sorted(fail_reasons.items(), key=lambda kv: -kv[1])[:15],
        "off_reasons": sorted(off_reasons.items(), key=lambda kv: -kv[1])[:15],
        "fail_recent": fail_recent[-40:],
        "counts": counts,
        "vk_caps": _parse_vk_caps(scan["vk_caps_line"]),
        # True when the parsed tail straddled a device restart and older samples
        # were dropped. Surfaced so a reading is never silently a two-boot blend.
        "boot_boundary_in_tail": boot_boundary_in_tail,
    }


def metrics(parsed):
    """The handful of readings this repo actually ranks a change on.

    Everything per-draw is computed over ONE `t` band shared by every per-window
    census involved, never over each tag's own range — see `_band`. The band is
    reported alongside the numbers so a narrow overlap is visible rather than
    implied.
    """
    out = {}
    series = parsed.get("series", {})
    windowed = {
        tag: entry for tag, entry in series.items()
        if entry["mode"] == "window" and entry.get("t_lo") is not None
    }

    # The shared band is the intersection of the per-window censuses: the latest
    # start and the earliest end. Anything outside it is a span some census did
    # not observe.
    if windowed:
        t_lo = max(entry["t_lo"] for entry in windowed.values())
        t_hi = min(entry["t_hi"] for entry in windowed.values())
        if t_hi < t_lo:
            t_lo = t_hi = None
    else:
        t_lo = t_hi = None
    out["band"] = {
        "t_lo": t_lo,
        "t_hi": t_hi,
        "span_ms": (t_hi - t_lo) if (t_lo is not None and t_hi is not None) else None,
        "tags": sorted(windowed),
    }
    joined = t_lo is not None and t_hi is not None and t_hi > t_lo

    def banded(tag):
        if not joined:
            return None
        return _band(series.get(tag), t_lo, t_hi)

    cadence = series.get("host_window_cadence")
    if cadence:
        last = cadence["last"]
        present_hz = last.get("present_hz")
        offered_hz = last.get("offered_hz")
        band = banded("host_window_cadence")
        out["present"] = {
            "present_hz": present_hz,
            "offered_hz": offered_hz,
            "busy_fence": last.get("busy_fence"),
            "busy_acquire": last.get("busy_acquire"),
            # `present_hz` is NEVER a reading on its own: the presenter passes
            # everything at the shipping depth, so it equals `offered_hz`, which
            # is the device's own publish rate. Both up is the win; offered up
            # alone means the presenter has become a ceiling again (check the two
            # busy counters, which are 0 at the shipping depth); neither moving
            # means the change bought no frames whatever else it bought.
            "presenter_is_ceiling": (
                present_hz is not None and offered_hz is not None
                and offered_hz > 0 and (offered_hz - present_hz) / offered_hz > 0.05
            ),
            # A macOS guest latches its frame period once, for its life: exactly
            # 60 Hz if the kernel handed it 16 666 666 ns, or work-limited
            # 95-117 Hz if it latched 0 and free-runs. The gap between the two is
            # empty on every boot on record, and NOTHING per-draw is comparable
            # across them — so the population is a label on every reading here.
            "population": (
                None if present_hz is None
                else ("paced-60" if present_hz <= PACED_HZ_CEILING else "free-running")
            ),
            "presents_banded": (band or {}).get("sum", {}).get("presents"),
        }

    drain = banded("drain_duty")
    if drain:
        draws = drain["sum"].get("draws", 0)
        out["cpu"] = {
            "duty": drain["last"].get("duty"),
            "draws": draws,
            "busy_us": drain["sum"].get("busy_us"),
            "us_per_draw": round(drain["sum"].get("busy_us", 0) / draws, 2) if draws else None,
            "tranches": drain["sum"].get("tranches"),
            "skipped": drain["sum"].get("skipped"),
            "samples": drain["samples"],
        }

    gpu = banded("gpu_span")
    if gpu:
        draw_n = gpu["sum"].get("draw_n", 0)
        out["gpu"] = {
            "busy_us": gpu["sum"].get("busy_us"),
            "draw_n": draw_n,
            # Half the score, and on an integrated-GPU pathway the LARGER half:
            # frames track the sum of CPU and GPU per draw roughly linearly, so
            # ranking a change on `us/draw` alone scores half of it. This half
            # also carries a wider boot-to-boot spread (~12 % against ~4 %), so
            # a GPU arm needs more boots than a CPU one.
            "us_per_draw": round(gpu["sum"].get("busy_us", 0) / draw_n, 2) if draw_n else None,
            "samples": gpu["samples"],
        }

    if out.get("cpu") and out.get("gpu"):
        cpu_us = out["cpu"].get("us_per_draw")
        gpu_us = out["gpu"].get("us_per_draw")
        if cpu_us is not None and gpu_us is not None:
            out["per_draw_total_us"] = round(cpu_us + gpu_us, 2)

    publish = banded("window_publish")
    if publish:
        out["publish"] = {
            "fresh": publish["sum"].get("fresh"),
            "same_key": publish["sum"].get("same_key"),
            "no_window": publish["sum"].get("no_window"),
            "no_frame": publish["sum"].get("no_frame"),
            "samples": publish["samples"],
        }

    # --- levels series: the LAST sample is the answer, summing is the error ----
    vbl = series.get("display_vbl")
    if vbl:
        last = vbl["last"]
        out["vbl"] = {
            "grid_hz": last.get("grid_hz"),
            "window_hz": last.get("window_hz"),
            "delivered": last.get("delivered"),
            "not_enabled": last.get("not_enabled"),
        }

    pressure = series.get("registry_pressure")
    if pressure:
        last = pressure["last"]
        out["pressure"] = {
            "peak": last.get("peak"),
            "peak_mib": last.get("peak_mib"),
            "resample_peak_ms": last.get("resample_peak_ms"),
            "resample_window_ms": last.get("resample_peak_ms_cap"),
            "slab_mib": last.get("slab_mib"),
            "slab_mib_cap": last.get("slab_mib_cap"),
        }

    # An undriven boot measures an idle device, and its counters are
    # self-consistent and well-formed while saying nothing about throughput. The
    # drain worker's duty is the discriminator: ~0.00 on a bursty interaction
    # probe, ~0.91 on a sustained one. Only the sustained regime can turn a
    # per-draw CPU saving into frames, so a reading taken at low duty must not be
    # ranked against one taken high.
    duty = (out.get("cpu") or {}).get("duty")
    if duty is not None:
        out["workload"] = {
            "duty": duty,
            "regime": (
                "idle" if duty < 0.10
                else "bursty" if duty < 0.50
                else "sustained"
            ),
            "rankable": duty >= 0.50,
        }

    # `store_routes` splits must add up, and that identity is the cheapest way to
    # catch a bad read of the series. Reported when the three parts are present.
    routes = banded("store_routes")
    if routes:
        total = routes["sum"]
        parts = ("no_state", "texture", "linear")
        if all(key in total for key in parts) and "unknown_object" in total:
            summed = sum(total[key] for key in parts)
            out["store_routes_identity"] = {
                "no_state+texture+linear": summed,
                "unknown_object": total["unknown_object"],
                "holds": summed == total["unknown_object"],
            }
    return out


# ---------------------------------------------------------------- repo interfaces
# Everything below asks an existing script rather than reimplementing it, so a
# change to a rule lands here for free and cannot be contradicted by this file.

BOOT_SCRIPT = {"x86": "vm/boot-x86.sh", "arm64": "vm/boot-arm64.sh"}

# The QEMU command line every sweep in this repo matches. Bracketing one
# character is load-bearing: `pgrep -f` matches whole command lines, and the
# shell running the pattern has the pattern in ITS command line, so an
# unbracketed pattern matches the process issuing it. Here the pattern goes to
# pgrep/pkill as an argv element rather than through a shell, but it is kept
# bracketed anyway so a copy of it into a terminal is safe.
QEMU_PATTERN = {
    "x86": r"qemu-system-x86_6[4].*reims-vgpu",
    "arm64": r"qemu-system-aarch6[4].*reims-vgpu",
}


# Both list headers read `... (current -> LABEL):`, and an unset `current`
# prints the literal `(unset)`. A greedy `(\S+)` capture takes the trailing
# `):` with it and turns "unset" into a label named `(unset)):` — which then
# shows up in a UI as a selectable snapshot. Anchor on the closing `):`.
_CURRENT = re.compile(r"current -> (.*?)\):\s*$")


def _parse_current(line):
    match = _CURRENT.search(line)
    if not match:
        return None
    label = match.group(1).strip()
    return None if label in ("(unset)", "") else label


def _parse_labels(stdout):
    """Indented single-token lines are the labels; the header is not indented."""
    labels, default = [], None
    for line in (stdout or "").splitlines():
        found = _parse_current(line)
        if found:
            default = found
            continue
        match = re.match(r"^\s{2,}(\S+)\s*$", line)
        if match:
            labels.append(match.group(1))
    return labels, default


def list_rails(pathway):
    """Rails and the default, from the boot script's own --list-rails."""
    script = BOOT_SCRIPT[pathway]
    out = _run([str(REPO_ROOT / script), "--list-rails"], timeout=20)
    rails, default = _parse_labels(out["stdout"])
    return {"rails": rails, "default": default, "rc": out["rc"], "stderr": out["stderr"][-400:]}


def list_snapshots(pathway, rail):
    """Snapshots within one rail. A rail is a plain label, never a path — the
    boot script enforces that too, and passing it as one argv element means a
    label containing a slash cannot become a second argument."""
    script = BOOT_SCRIPT[pathway]
    out = _run([str(REPO_ROOT / script), "--rail", rail, "--list-snapshots"], timeout=20)
    labels, default = _parse_labels(out["stdout"])
    return {"snapshots": labels, "default": default, "rc": out["rc"]}


# The three spec forms vm/lib/usb-passthrough.sh accepts. Re-validated here
# before a spec is allowed onto a command line: the library is the authority on
# what a spec MEANS, but this process is the one handing bytes to exec, and it
# must not pass through anything it has not itself recognised.
USB_SPEC_FORMS = (
    re.compile(r"^[0-9a-fA-F]{4}:[0-9a-fA-F]{4}$"),
    re.compile(r"^[0-9]+\.[0-9]+$"),
    re.compile(r"^[0-9]+-[0-9]+(\.[0-9]+)*$"),
)


def valid_usb_spec(spec):
    return isinstance(spec, str) and any(form.match(spec) for form in USB_SPEC_FORMS)


def list_usb(pathway):
    """Host USB devices, parsed out of the boot script's --list-usb table.

    The table is produced by vm/lib/usb-passthrough.sh, so the spec forms shown
    here are by construction the ones a boot will accept. Writability of the
    /dev/bus/usb node is added per row because it is the difference between a
    device that passes through and one that fails inside the guest's driver
    probe, and it is the single most common reason a first attempt does nothing.
    """
    script = BOOT_SCRIPT[pathway]
    out = _run([str(REPO_ROOT / script), "--list-usb"], timeout=20)
    devices = []
    for line in (out["stdout"] or "").splitlines():
        parts = line.split(None, 3)
        if len(parts) < 4 or parts[0] == "VID:PID" or ":" not in parts[0]:
            continue
        vidpid, busaddr, busport, desc = parts[0], parts[1], parts[2], parts[3]
        if not valid_usb_spec(vidpid):
            continue
        node, writable = None, None
        m = re.match(r"^(\d+)\.(\d+)$", busaddr)
        if m:
            node = "/dev/bus/usb/%03d/%03d" % (int(m.group(1)), int(m.group(2)))
            writable = os.access(node, os.R_OK | os.W_OK)
        devices.append({
            "vidpid": vidpid,
            "busaddr": busaddr,
            "busport": busport,
            "description": desc.strip(),
            "node": node,
            "writable": writable,
            # A hub is listed because the table lists it, but passing one is
            # almost never what someone means, so it is flagged rather than
            # hidden — hiding it would be this file inventing a rule.
            "looks_like_hub": "hub" in desc.lower(),
        })
    return {"devices": devices, "rc": out["rc"], "stderr": out["stderr"][-400:]}


def qemu_processes(pathway):
    """Live QEMU processes for this pathway, with start time and elapsed.

    A boot measured next to another VM measures the contention, and two guests on
    one bridge collide on the pinned MAC in silence, so knowing what is already
    running is a precondition for both launching and believing a number.
    """
    out = _run(["pgrep", "-af", QEMU_PATTERN[pathway]], timeout=5)
    procs = []
    for line in (out["stdout"] or "").splitlines():
        pid, _, cmd = line.partition(" ")
        if not pid.isdigit():
            continue
        info = _run(["ps", "-o", "lstart=,etime=", "-p", pid], timeout=5)
        started, elapsed = "", ""
        if info["rc"] == 0 and info["stdout"].strip():
            fields = info["stdout"].strip().rsplit(None, 1)
            if len(fields) == 2:
                started, elapsed = fields[0].strip(), fields[1].strip()
        # The netdev in the command line is the ground truth for how this guest
        # is reachable, and it is more reliable than remembering what was asked
        # for: a boot that fell back, or an older binary, says so here.
        net = "unknown"
        if "-netdev bridge" in cmd:
            net = "bridge"
        elif "-netdev user" in cmd:
            net = "user"
        elif "-nic none" in cmd:
            net = "none"
        procs.append({
            "pid": int(pid),
            "started": started,
            "elapsed": elapsed,
            "net": net,
            "usb_passthrough": cmd.count("usb-host,"),
            "cmd": cmd[:2000],
        })
    return procs


# --------------------------------------------------------------------- launching

# Env knobs this UI is allowed to set. An allowlist rather than a passthrough,
# because an override may only NARROW what the device does: a switch can turn off
# a rail the host could have run, but it can never turn on one the host reported
# it cannot — binding an extension a host does not advertise fails
# vkCreateDevice, and importing a handle type it declines is undefined behavior
# inside the driver. Every entry here is a documented narrowing or a
# verification aid, and the UI cannot invent a new one.
ENV_KNOBS = {
    "REIMS_VGPU_GUEST_IMPORT": {
        "values": ["", "off"],
        "label": "Guest host-pointer import",
        "note": "off takes a capable host to the disabled_by_env rung, which is how the "
                "copying rails get exercised without hunting for hardware that lacks the "
                "extension. Compare the two boots on their PIXELS, not only their counters.",
    },
    "REIMS_VGPU_GATHER_AUDIT_ALL": {
        "values": ["", "on"],
        "label": "Audit every vouched bind",
        "note": "Judges every zero-copy bind instead of 1 in 64. Never quote a timing from "
                "such a boot — the fold re-reads the very windows the cache exists to avoid.",
    },
    "REIMS_VGPU_PRESENT_DEPTH": {
        "values": ["", "off"],
        "label": "Presents in flight",
        "note": "off drops to one present in flight, the old ~41 fps clamp. An ablation arm.",
    },
    "TRACE": {
        "values": ["", "1"],
        "label": "QEMU trace log",
        "note": "Device trace events to the run dir. Costs throughput; not for a timed boot.",
    },
}

BOOT_CLASSES = ("testing", "interactive", "capture")
X86_DEVICES = ("reims-vgpu-pci", "vmware-svga")
ARM64_DEVICES = ("reims-vgpu-mmio", "apple-gfx-mmio")
NET_MODES = ("bridge", "user", "none")

_boot_state = {
    "proc": None,
    "argv": None,
    "env": None,
    "log_path": None,
    "started": None,
    "returncode": None,
    "config": None,
}
_boot_lock = threading.Lock()


def build_boot_argv(cfg):
    """Turn a UI config into an argv list and an env dict, or raise ValueError.

    Validation is total: every field is checked against a closed set, and a USB
    spec must match one of the three forms before it can reach a command line.
    Nothing is interpolated into a string anywhere on this path.
    """
    pathway = cfg.get("pathway") or host_pathway()
    if pathway not in BOOT_SCRIPT:
        raise ValueError(f"unknown pathway {pathway!r}")

    argv = [str(REPO_ROOT / BOOT_SCRIPT[pathway])]

    boot_class = cfg.get("boot_class", "testing")
    if boot_class not in BOOT_CLASSES:
        raise ValueError(f"unknown boot class {boot_class!r}")
    # `capture` mutates the rail by writing a new snapshot on clean shutdown, so
    # it is never reachable by default from a web form: the caller has to say so.
    if boot_class == "capture" and not cfg.get("confirm_capture"):
        raise ValueError("boot class 'capture' writes a new snapshot into the rail; "
                         "it needs confirm_capture=true")
    argv.append(f"--{boot_class}")

    device = cfg.get("device")
    allowed = X86_DEVICES if pathway == "x86" else ARM64_DEVICES
    if device:
        if device not in allowed:
            raise ValueError(f"device {device!r} is not one of {allowed}")
        argv += ["--device", device]

    rail = cfg.get("rail")
    if rail:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", rail):
            raise ValueError(f"rail {rail!r} is not a plain label")
        argv += ["--rail", rail]

    snapshot = cfg.get("snapshot")
    if snapshot:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", snapshot):
            raise ValueError(f"snapshot {snapshot!r} is not a plain label")
        argv += ["--snapshot", snapshot]

    for spec in cfg.get("usb", []):
        if not valid_usb_spec(spec):
            raise ValueError(f"USB spec {spec!r} is not VID:PID, BUS.ADDR or BUS-PORT")
        argv += ["--usb", spec]

    env = {}
    net = cfg.get("net")
    if net:
        if net not in NET_MODES:
            raise ValueError(f"unknown net mode {net!r}")
        if net == "bridge" and pathway == "arm64":
            raise ValueError("the arm64 pathway has no bridge mode: virbr0 and "
                             "qemu-bridge-helper do not exist on an Apple host")
        env["NET"] = net
    bridge = cfg.get("bridge")
    if bridge:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,14}", bridge):
            raise ValueError(f"bridge {bridge!r} is not an interface name")
        env["BRIDGE"] = bridge

    for key, value in (cfg.get("env") or {}).items():
        if key not in ENV_KNOBS:
            raise ValueError(f"{key} is not a settable knob here")
        if value not in ENV_KNOBS[key]["values"]:
            raise ValueError(f"{key}={value!r} is not one of {ENV_KNOBS[key]['values']}")
        if value:
            env[key] = value

    for key, pattern in (("RAM", r"^[0-9]{1,3}[MG]$"),
                         ("CPU_CORES", r"^[0-9]{1,2}$"),
                         ("CPU_THREADS", r"^[0-9]{1,2}$"),
                         ("GUEST_MAC", r"^([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$")):
        value = cfg.get(key.lower())
        if value:
            if not re.match(pattern, str(value)):
                raise ValueError(f"{key}={value!r} is malformed")
            env[key] = str(value)

    return pathway, argv, env


def start_boot(cfg):
    """Launch a boot in the background, capturing its stdout to a file.

    THE STDOUT FILE IS NOT A CONVENIENCE. A guest kernel panic can land after a
    probe has finished and reported success, so `probe exit=0` and a live device
    both read green on a boot whose guest died. The boot script prints
    `capture-then-revert (guest kernel panic)` on its own stdout, and grepping
    that is the verdict that outranks the probe's.
    """
    with _boot_lock:
        proc = _boot_state["proc"]
        if proc is not None and proc.poll() is None:
            raise RuntimeError(f"a boot is already running under this dashboard (pid {proc.pid})")

        pathway, argv, env = build_boot_argv(cfg)

        # Two guests at once is not an error we can rule out from here — the user
        # may want it — but it is never what someone clicking Boot means, and on a
        # bridge the second one collides on the pinned MAC in total silence.
        running = qemu_processes(pathway)
        if running and not cfg.get("allow_concurrent"):
            pids = ", ".join(str(p["pid"]) for p in running)
            raise RuntimeError(
                f"QEMU is already running for the {pathway} pathway (pid {pids}). "
                "On NET=bridge a second guest collides on the pinned GUEST_MAC with no "
                "diagnostic, and either way the measurement is contended. Stop it first, "
                "or pass allow_concurrent with its own GUEST_MAC.")

        # A boot's readings are only its own if the log is only its own: the
        # device appends and never truncates, and a stale log inflates in a way
        # that reads as a finding. Removing it is what makes the boot count 1.
        if cfg.get("fresh_log", True):
            try:
                FAIL_LOG.unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise RuntimeError(f"could not remove {FAIL_LOG}: {exc}") from exc
            _scan_cache.update(mtime=None, size=None, count=None, vk_caps_line=None)

        run_dir = REPO_ROOT / ("vm/disks/run" if pathway == "x86" else "vm/guest/run")
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / f"dashboard-boot-{time.strftime('%Y%m%d-%H%M%S')}.log"
        handle = log_path.open("wb")

        merged = dict(os.environ)
        merged.update(env)
        child = subprocess.Popen(
            argv,
            cwd=str(REPO_ROOT),
            env=merged,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            # Its own process group, so stopping the boot can signal the whole
            # tree rather than only the script that spawned QEMU.
            start_new_session=True,
        )
        handle.close()
        _boot_state.update(
            proc=child, argv=argv, env=env, log_path=str(log_path),
            started=time.time(), returncode=None, config=cfg,
        )
        return {"pid": child.pid, "argv": argv, "env": env, "log": str(log_path)}


PANIC_MARKERS = ("guest kernel panic", "Debugger called: <panic>")


def boot_status():
    """What the launched boot is doing, including the panic verdict.

    `probe exit=0` is not a clean boot and neither is a live device — both read
    green on a boot whose guest panicked, which on some rails is roughly one
    driven boot in three. The panic grep outranks them.
    """
    with _boot_lock:
        proc = _boot_state["proc"]
        log_path = _boot_state["log_path"]
        state = {
            "running": proc is not None and proc.poll() is None,
            "pid": proc.pid if proc else None,
            "returncode": proc.poll() if proc else None,
            "argv": _boot_state["argv"],
            "env": _boot_state["env"],
            "log": log_path,
            "started": _boot_state["started"],
            "elapsed": round(time.time() - _boot_state["started"], 1) if _boot_state["started"] else None,
            "config": _boot_state["config"],
        }
    tail, panic = "", False
    if log_path and Path(log_path).is_file():
        tail, _ = _tail_text(Path(log_path), 64 << 10)
        low = tail.lower()
        panic = any(marker.lower() in low for marker in PANIC_MARKERS)
    state["stdout_tail"] = tail[-8000:]
    state["panic"] = panic
    return state


def stop_boot(pathway):
    """Stop the dashboard's boot, then sweep any QEMU left for this pathway.

    Both halves are needed: the boot script outlives its QEMU in some paths and
    QEMU outlives its script in others, and a survivor is what makes the next
    boot measure the previous build.
    """
    results = []
    with _boot_lock:
        proc = _boot_state["proc"]
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            results.append(f"SIGTERM to process group {os.getpgid(proc.pid)}")
        except (ProcessLookupError, PermissionError) as exc:
            results.append(f"could not signal the boot group: {exc}")
        for _ in range(50):
            if proc.poll() is not None:
                break
            time.sleep(0.2)
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                results.append("SIGKILL to process group")
            except (ProcessLookupError, PermissionError):
                pass
    # `pkill` with the pattern as an argv element: no shell means the bracket is
    # belt-and-braces rather than the thing preventing self-slaughter.
    sweep = _run(["pkill", "-f", QEMU_PATTERN[pathway]], timeout=10)
    results.append(f"pkill -f {QEMU_PATTERN[pathway]} rc={sweep['rc']}")
    time.sleep(0.5)
    return {"actions": results, "still_running": qemu_processes(pathway)}


# ----------------------------------------------------------------- guest actions

QMP_PY = REPO_ROOT / "scripts/qmp/qmp.py"

# QMP verbs this UI exposes, with how many numeric arguments each takes. An
# allowlist: `qmp.py cmd` can issue arbitrary QMP, which is not something a web
# form should reach.
QMP_ACTIONS = {
    "size": 0, "click": 2, "move": 2, "key": None, "type": None, "wheel": None,
}


def qmp_action(action, args):
    if action not in QMP_ACTIONS:
        raise ValueError(f"{action!r} is not an exposed QMP action")
    argv = [sys.executable, str(QMP_PY), action]
    arity = QMP_ACTIONS[action]
    if arity == 0:
        if args:
            raise ValueError(f"{action} takes no arguments")
    elif arity is None:
        # Free-form verbs (key names, typed text, wheel counts). Passed as
        # separate argv elements, so nothing here can become a second command.
        for arg in args:
            if not isinstance(arg, str) or len(arg) > 200:
                raise ValueError("argument must be a string under 200 chars")
            argv.append(arg)
    else:
        if len(args) != arity:
            raise ValueError(f"{action} takes {arity} arguments")
        for arg in args:
            if not re.fullmatch(r"-?[0-9]{1,6}", str(arg)):
                raise ValueError(f"{arg!r} is not a coordinate")
            argv.append(str(arg))
    return _run(argv, timeout=30)


def screenshot(pathway, out_path):
    """Host-side PNG of the live QEMU window.

    QMP `shot`/`screendump` is NOT the route and is disabled on purpose: with the
    host-owned window and QEMU at `-display none` the frame never crosses into
    QEMU's address space, so a screendump shows something other than what the
    window shows.
    """
    helper = (
        "scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh"
        if pathway == "x86" else
        "scripts/screenshot-when-macos-host/screenshot-when-macos-host.sh"
    )
    path = REPO_ROOT / helper
    if not path.is_file():
        return {"rc": None, "stdout": "", "stderr": f"no screenshot helper at {helper}"}
    if pathway == "x86":
        return _run([str(path), "-o", str(out_path)], timeout=60)
    return _run([str(path), str(out_path)], timeout=60)


# Probes the dashboard can start, with the flags each actually takes. Listed
# explicitly rather than globbed, because a probe's argv is part of its contract
# and a wrong flag reads as a device result.
PROBES = {
    "guest-authorize": {
        "argv": ["vm/guest-authorize.sh"],
        "label": "Authorize guest (key + ssh alias)",
        "note": "Resolves the boot's endpoint, installs the ssh key into the COW clone, and "
                "points `macos-vm` at it. Every other probe needs this first.",
        "timeout": 480,
    },
    "guest-ip": {
        "argv": ["vm/guest-ip.sh", "--wait", "0"],
        "label": "Guest IP (bridge lease)",
        "note": "Reads the DHCP lease for the pinned MAC. Exit 3 means no lease yet.",
        "timeout": 30,
    },
    "window-drag": {
        "argv": ["scripts/window-drag-probe/window-drag-probe.sh", "--seconds", "25", "--app", "Safari"],
        "label": "Window drag (interaction)",
        "note": "Real window-server compositing. Bursty: most of its wall clock is guest "
                "animation, so it measures the gaps between its bursts.",
        "timeout": 300,
    },
    "sustained-animation": {
        "argv": ["scripts/sustained-animation-probe/sustained-animation-probe.sh"],
        "label": "Sustained animation (throughput)",
        "note": "The only regime whose drain duty is high enough to turn a per-draw CPU "
                "saving into frames. Use this to rank a throughput change.",
        "timeout": 300,
    },
    "wait-for-desktop": {
        "argv": ["scripts/app-sweep-probe/wait-for-desktop.sh"],
        "label": "Wait for desktop",
        "note": "sshd answers well before the desktop composites. Also collects crash "
                "reports and REFUSES to log in when WindowServer has aborted.",
        "timeout": 480,
    },
}

_probe_state = {}
_probe_lock = threading.Lock()


def start_probe(name, extra_env=None):
    spec = PROBES.get(name)
    if spec is None:
        raise ValueError(f"{name!r} is not an exposed probe")
    with _probe_lock:
        prior = _probe_state.get(name)
        if prior and prior["proc"].poll() is None:
            raise RuntimeError(f"probe {name} is already running (pid {prior['proc'].pid})")
        argv = [str(REPO_ROOT / spec["argv"][0])] + spec["argv"][1:]
        run_dir = REPO_ROOT / "vm/disks/run"
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / f"dashboard-probe-{name}-{time.strftime('%Y%m%d-%H%M%S')}.log"
        handle = log_path.open("wb")
        merged = dict(os.environ)
        for key, value in (extra_env or {}).items():
            if key not in ("NET", "BRIDGE"):
                raise ValueError(f"{key} may not be set for a probe here")
            merged[key] = str(value)
        child = subprocess.Popen(
            argv, cwd=str(REPO_ROOT), env=merged,
            stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        handle.close()
        _probe_state[name] = {"proc": child, "log": str(log_path), "started": time.time()}
        return {"pid": child.pid, "log": str(log_path), "argv": argv}


def probe_status():
    out = {}
    with _probe_lock:
        items = list(_probe_state.items())
    for name, state in items:
        tail = ""
        if Path(state["log"]).is_file():
            tail, _ = _tail_text(Path(state["log"]), 32 << 10)
        out[name] = {
            "running": state["proc"].poll() is None,
            "pid": state["proc"].pid,
            "returncode": state["proc"].poll(),
            "elapsed": round(time.time() - state["started"], 1),
            "log": state["log"],
            "tail": tail[-4000:],
        }
    return out


# ------------------------------------------------------------------------ the API

def api_state(bridge_name="virbr0"):
    """One call that answers everything the page needs to render."""
    pathway = host_pathway()
    kernel = kernel_module_state()
    tun = tun_available()
    helper = bridge_helper()
    acl = bridge_acl(bridge_name)
    parsed = parse_log()

    # Whether a bridged boot can work AT ALL, assembled once here so the UI shows
    # one verdict with its reasons rather than four independent badges the reader
    # has to combine.
    blockers = []
    if pathway != "x86":
        blockers.append("the arm64 pathway has no bridge mode (no virbr0, no bridge helper on macOS)")
    if not tun["ok"]:
        detail = f"/dev/net/tun will not open: {tun['error']}"
        if not kernel["module_tree_present"]:
            detail += (f" — the running kernel {kernel['release']} has no module tree, so no module "
                       f"can load. Installed: {', '.join(kernel['installed_trees'])}.")
            detail += (" The running kernel's own tun module is still in the package cache, so this "
                       "does not need a reboot — see Host & device."
                       if kernel["cached_package"] else " A reboot is the fix.")
        blockers.append(detail)
    if not any(br["name"] == bridge_name for br in bridges()):
        blockers.append(f"no bridge named {bridge_name}: sudo virsh --connect qemu:///system net-start default")
    if not helper["privileged"]:
        blockers.append(
            f"no privileged qemu-bridge-helper"
            + (f" ({helper['rejected']} is present but unprivileged)" if helper["rejected"] else ""))
    if acl["readable"] and acl["allows"] is False:
        blockers.append(f"{acl['file']} has no 'allow {bridge_name}' line")

    return {
        "pathway": pathway,
        "repo_root": str(REPO_ROOT),
        "host": {
            "kernel": kernel,
            "tun": tun,
            "load": load_average(),
            "uname": " ".join(os.uname()),
        },
        "net": {
            "bridges": bridges(),
            "helper": helper,
            "acl": acl,
            "bridge_ready": not blockers,
            "blockers": blockers,
            "modes": list(NET_MODES),
        },
        "rails": list_rails(pathway),
        "usb": list_usb(pathway),
        "devices": list(X86_DEVICES if pathway == "x86" else ARM64_DEVICES),
        "boot_classes": list(BOOT_CLASSES),
        "env_knobs": ENV_KNOBS,
        "probes": {name: {"label": spec["label"], "note": spec["note"]}
                   for name, spec in PROBES.items()},
        "qemu": qemu_processes(pathway),
        "boot": boot_status(),
        "probe_status": probe_status(),
        "log": {
            "path": parsed["path"],
            "present": parsed["present"],
            "size": parsed["size"],
            "boots": parsed["boots"],
            "boot_boundary_in_tail": parsed.get("boot_boundary_in_tail"),
            "counts": parsed["counts"],
            "fail_reasons": parsed["fail_reasons"],
            "off_reasons": parsed["off_reasons"],
            "fail_recent": parsed["fail_recent"],
            "vk_caps": parsed["vk_caps"],
            "series": {tag: {k: v for k, v in entry.items() if k != "_samples"}
                       for tag, entry in parsed["series"].items()},
        },
        "metrics": metrics(parsed),
        "now": time.time(),
    }


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "vgpu-dashboard"
    token = None
    index_html = None

    def log_message(self, fmt, *args):   # quieter than the default access log
        if os.environ.get("REIMS_VGPU_DASH_VERBOSE"):
            sys.stderr.write("dash: " + (fmt % args) + "\n")

    # ---- plumbing ----------------------------------------------------------
    def _authorized(self, query):
        supplied = query.get("token", [None])[0] or self.headers.get("X-Dash-Token")
        return supplied is not None and secrets.compare_digest(supplied, self.token)

    def _send(self, code, body, content_type="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, default=str).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        # This page is a control surface for a local process; nothing about it
        # should be cached or embedded anywhere.
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def _body_json(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > (1 << 20):
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"body is not JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("body must be a JSON object")
        return parsed

    # ---- routes ------------------------------------------------------------
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(url.query)
        path = url.path

        if path == "/" or path == "/index.html":
            if not self._authorized(query):
                return self._send(403, "Forbidden: open the URL this server printed, token included.",
                                  "text/plain; charset=utf-8")
            return self._send(200, self.index_html, "text/html; charset=utf-8")

        if not self._authorized(query):
            return self._send(403, {"error": "bad or missing token"})

        try:
            if path == "/api/state":
                bridge = query.get("bridge", ["virbr0"])[0]
                return self._send(200, api_state(bridge))
            if path == "/api/snapshots":
                rail = query.get("rail", [""])[0]
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", rail or ""):
                    return self._send(400, {"error": "rail must be a plain label"})
                return self._send(200, list_snapshots(host_pathway(), rail))
            if path == "/api/boot-argv":
                # Dry run: show exactly what would be executed, so a launch is
                # never a surprise and a config can be copied to a terminal.
                cfg = json.loads(query.get("cfg", ["{}"])[0])
                pathway, argv, env = build_boot_argv(cfg)
                return self._send(200, {
                    "pathway": pathway, "argv": argv, "env": env,
                    "shell": " ".join(
                        [f"{k}={shlex.quote(v)}" for k, v in sorted(env.items())]
                        + [shlex.quote(a) for a in argv]),
                })
            if path == "/api/screenshot":
                out = REPO_ROOT / "vm/disks/run/dashboard-screenshot.png"
                result = screenshot(host_pathway(), out)
                if result["rc"] != 0 or not out.is_file():
                    return self._send(503, {"error": "capture failed", **result})
                return self._send(200, out.read_bytes(), "image/png")
            return self._send(404, {"error": f"no route {path}"})
        except ValueError as exc:
            return self._send(400, {"error": str(exc)})
        except Exception as exc:                       # noqa: BLE001 - report, never 500 silently
            return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        query = urllib.parse.parse_qs(url.query)
        if not self._authorized(query):
            return self._send(403, {"error": "bad or missing token"})
        try:
            body = self._body_json()
            path = url.path
            if path == "/api/boot":
                return self._send(200, start_boot(body))
            if path == "/api/stop":
                return self._send(200, stop_boot(host_pathway()))
            if path == "/api/qmp":
                return self._send(200, qmp_action(body.get("action", ""), body.get("args", [])))
            if path == "/api/probe":
                return self._send(200, start_probe(body.get("name", ""), body.get("env")))
            return self._send(404, {"error": f"no route {path}"})
        except ValueError as exc:
            return self._send(400, {"error": str(exc)})
        except RuntimeError as exc:
            return self._send(409, {"error": str(exc)})
        except Exception as exc:                       # noqa: BLE001
            return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


class Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ------------------------------------------------------------------------- driver

def selftest():
    """Exercise every read-only path against the real host and report.

    Deliberately never launches a boot: the point is that a checkout can verify
    the dashboard's own plumbing — parsing, host probes, argv construction,
    validation — without a VM, an Apple host or a reboot. What it cannot cover is
    a live guest, and it says so rather than passing quietly.
    """
    failures = []

    def check(name, fn):
        try:
            value = fn()
            print(f"  ok    {name}")
            return value
        except Exception as exc:                       # noqa: BLE001
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            failures.append(name)
            return None

    print("host probes")
    check("host_pathway", host_pathway)
    check("kernel_module_state", kernel_module_state)
    check("tun_available", tun_available)
    check("bridges", bridges)
    check("bridge_helper", bridge_helper)
    check("bridge_acl", lambda: bridge_acl("virbr0"))
    check("load_average", load_average)

    print("repo interfaces")
    pathway = host_pathway()
    rails = check("list_rails", lambda: list_rails(pathway))
    if rails and rails["rails"]:
        check("list_snapshots", lambda: list_snapshots(pathway, rails["rails"][0]))
    usb = check("list_usb", lambda: list_usb(pathway))
    check("qemu_processes", lambda: qemu_processes(pathway))

    print("log parsing")
    parsed = check("parse_log", parse_log)
    if parsed:
        check("metrics", lambda: metrics(parsed))
        # The two reduce modes must not be confused: a per-window series carries
        # a sum, a levels series must not.
        def reduce_modes():
            for tag, entry in parsed["series"].items():
                if entry["mode"] == "window" and entry["sum"] is None:
                    raise AssertionError(f"{tag} is per-window but has no sum")
                if entry["mode"] == "levels" and entry["sum"] is not None:
                    raise AssertionError(f"{tag} is a levels series but was summed")
            return True
        check("reduce modes are honoured", reduce_modes)

    print("argv construction")
    def good():
        _, argv, env = build_boot_argv({
            "pathway": "x86", "boot_class": "testing", "device": "reims-vgpu-pci",
            "rail": "macos-13", "net": "bridge", "usb": ["5-1.2", "046d:c099", "1.4"],
            "env": {"REIMS_VGPU_GUEST_IMPORT": "off"}, "ram": "16G",
        })
        assert "--usb" in argv and argv.count("--usb") == 3, argv
        assert env["NET"] == "bridge" and env["REIMS_VGPU_GUEST_IMPORT"] == "off", env
        return True
    check("valid config builds", good)

    rejections = [
        ("shell metacharacter in rail", {"rail": "a; rm -rf /"}),
        ("path in rail", {"rail": "../../etc"}),
        ("bad usb spec", {"usb": ["not-a-spec"]}),
        ("usb spec with a semicolon", {"usb": ["046d:c099; id"]}),
        ("unknown device", {"device": "nvidia-please"}),
        ("unknown net mode", {"net": "wifi"}),
        ("unlisted env knob", {"env": {"PATH": "/tmp"}}),
        ("bad value for a known knob", {"env": {"REIMS_VGPU_GUEST_IMPORT": "yes-please"}}),
        ("capture without confirmation", {"boot_class": "capture"}),
        ("malformed MAC", {"guest_mac": "zz:zz"}),
        ("malformed RAM", {"ram": "16 gigs"}),
        ("bridge on arm64", {"pathway": "arm64", "net": "bridge"}),
    ]
    for label, cfg in rejections:
        cfg = {"pathway": cfg.get("pathway", "x86"), "boot_class": cfg.get("boot_class", "testing"), **cfg}
        try:
            build_boot_argv(cfg)
            print(f"  FAIL  rejects {label}: it was ACCEPTED")
            failures.append(f"rejects {label}")
        except ValueError:
            print(f"  ok    rejects {label}")

    print("usb spec validator")
    for spec, want in (("046d:c099", True), ("046D:C099", True), ("5.3", True),
                       ("5-1.2", True), ("5-1", True), ("nonsense", False),
                       ("", False), ("046d:c099 ; ls", False), ("../x", False)):
        got = valid_usb_spec(spec)
        if got == want:
            print(f"  ok    {spec!r} -> {got}")
        else:
            print(f"  FAIL  {spec!r} -> {got}, wanted {want}")
            failures.append(f"spec {spec!r}")

    print("qmp action validator")
    for action, args, ok in (("size", [], True), ("click", ["10", "20"], True),
                             ("click", ["10"], False), ("cmd", ["quit"], False),
                             ("click", ["x", "y"], False)):
        try:
            argv = [sys.executable, str(QMP_PY), action]
            if action not in QMP_ACTIONS:
                raise ValueError("not exposed")
            arity = QMP_ACTIONS[action]
            if arity == 0 and args:
                raise ValueError("takes none")
            if isinstance(arity, int) and arity > 0:
                if len(args) != arity:
                    raise ValueError("arity")
                for arg in args:
                    if not re.fullmatch(r"-?[0-9]{1,6}", str(arg)):
                        raise ValueError("not a coordinate")
            accepted = True
        except ValueError:
            accepted = False
        if accepted == ok:
            print(f"  ok    {action}{args} -> {'accept' if accepted else 'reject'}")
        else:
            print(f"  FAIL  {action}{args} -> {'accept' if accepted else 'reject'}")
            failures.append(f"qmp {action}")

    print("full state assembly")
    state = check("api_state", lambda: api_state("virbr0"))
    if state:
        check("state is JSON-serialisable", lambda: json.dumps(state, default=str))

    print()
    print("NOT COVERED by this selftest, and not claimed:")
    print("  - a live guest (boot, DHCP lease, ssh, probes, screenshot)")
    print("  - anything on the arm64 pathway, which needs an Apple host")
    if usb and not any(d["writable"] for d in usb["devices"]):
        print("  - a passthrough-eligible USB device: no /dev/bus/usb node here is writable")
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("all read-only checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser(description="reims-vgpu control dashboard")
    ap.add_argument("--port", type=int, default=int(os.environ.get("REIMS_VGPU_DASH_PORT", 8787)))
    # Bound to loopback and not configurable to anything else from the command
    # line. This server execs QEMU and the repo's scripts; a token on a LAN
    # socket is a much weaker thing than a token on a loopback socket, and there
    # is no use case here that needs the weaker one.
    ap.add_argument("--selftest", action="store_true", help="run read-only checks and exit")
    ap.add_argument("--no-browser", action="store_true", help="do not try to open a browser")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    index = HERE / "index.html"
    if not index.is_file():
        print(f"vgpu-dashboard: missing {index}", file=sys.stderr)
        return 1

    Handler.token = secrets.token_urlsafe(24)
    Handler.index_html = index.read_text()

    try:
        httpd = Server(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        print(f"vgpu-dashboard: cannot bind 127.0.0.1:{args.port}: {exc}", file=sys.stderr)
        return 1

    url = f"http://127.0.0.1:{args.port}/?token={Handler.token}"
    # The token is only ever printed here, so this banner must reach the terminal
    # even when stdout is a pipe or a log file — a buffered URL is a server
    # nobody can open.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass
    print("reims-vgpu dashboard")
    print(f"  pathway   {host_pathway()}")
    print(f"  repo      {REPO_ROOT}")
    print(f"  fail log  {FAIL_LOG}")
    print()
    print(f"  {url}")
    print()
    print("  Loopback only, and every request needs that token. Ctrl-C to stop.", flush=True)
    if not args.no_browser and os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        opener = shutil.which("xdg-open") or shutil.which("open")
        if opener and not args.no_browser:
            subprocess.Popen([opener, url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nvgpu-dashboard: stopping (the VM, if any, keeps running)")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
