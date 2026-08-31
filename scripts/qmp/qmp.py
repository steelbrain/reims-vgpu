#!/usr/bin/env python3
"""qmp.py — QMP client for the vmapple VM (raw commands + GUI helpers).

Both boot scripts expose a per-boot QMP unix socket plus a stable `qmp.sock`
symlink to the live one, in the run directory belonging to their pathway:
vm/guest/run/ for boot-arm64.sh, vm/disks/run/ for boot-x86.sh. This resolves
whichever of the two is currently live, so a driver script works on either
pathway without being told which. Override with QMP_SOCK=/path/to.sock.

Raw QMP:
  scripts/qmp/qmp.py cmd query-status
  scripts/qmp/qmp.py cmd human-monitor-command '{"command-line":"info usb"}'

GUI helpers (usb-kbd + usb-tablet on the vmapple machine; cocoa is observability only):
  scripts/qmp/qmp.py click X Y [--double]     # guest-pixel coords
  scripts/qmp/qmp.py rclick X Y               # right button (dock context menus)
  scripts/qmp/qmp.py move X Y
  scripts/qmp/qmp.py drag X1 Y1 X2 Y2 [X3 Y3 ...]  # left-button rubber-band drag through points
      QMP_DRAG_STEPS=N    sub-moves interpolated per segment (default 8)
      QMP_DRAG_HOLD_S=F   seconds between sub-moves (default 0.02; 0 = socket rate)
      prints moves_s, the pointer rate the guest actually saw -- a ceiling on
      the frames any pointer-tracking pan can contain. See drag().
  scripts/qmp/qmp.py size                     # "WIDTH HEIGHT" of the guest display
  scripts/qmp/qmp.py key NAME[+NAME...] ...   # e.g. key ret, key meta_l+q
  scripts/qmp/qmp.py wheel [up|down] [N] [dt] # N wheel ticks, dt seconds apart (one connection)
  scripts/qmp/qmp.py type TEXT                # ASCII (shift combos handled)
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# One per pathway. A boot removes its own symlink on the way out, so at most one
# of these exists during a boot and "whichever is there" is unambiguous. Ordered
# only to make the both-present case deterministic rather than to prefer a
# pathway: that case means a stale symlink outlived its boot, and connecting to a
# dead socket raises here instead of driving the wrong VM.
_SOCK_CANDIDATES = (
    os.path.join(_REPO_ROOT, "vm", "disks", "run", "qmp.sock"),
    os.path.join(_REPO_ROOT, "vm", "guest", "run", "qmp.sock"),
)


def _default_sock() -> str:
    for path in _SOCK_CANDIDATES:
        if os.path.exists(path):
            return path
    # Nothing live: return the first so the error names a real path to look at.
    return _SOCK_CANDIDATES[0]


SOCK = os.environ.get("QMP_SOCK") or _default_sock()
TABLET_MAX = 32767

# ASCII → qcode (unshifted), per qapi/ui.json QKeyCode.
PLAIN = {
    " ": "spc",
    "-": "minus",
    "=": "equal",
    "[": "bracket_left",
    "]": "bracket_right",
    ";": "semicolon",
    "'": "apostrophe",
    "`": "grave_accent",
    "\\": "backslash",
    ",": "comma",
    ".": "dot",
    "/": "slash",
    "\n": "ret",
    "\t": "tab",
}
SHIFTED = {
    "!": "1",
    "@": "2",
    "#": "3",
    "$": "4",
    "%": "5",
    "^": "6",
    "&": "7",
    "*": "8",
    "(": "9",
    ")": "0",
    "_": "minus",
    "+": "equal",
    "{": "bracket_left",
    "}": "bracket_right",
    ":": "semicolon",
    '"': "apostrophe",
    "~": "grave_accent",
    "|": "backslash",
    "<": "comma",
    ">": "dot",
    "?": "slash",
}


class Qmp:
    def __init__(self, path: str = SOCK):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(15)
        self.sock.connect(path)
        self.f = self.sock.makefile("rwb", buffering=0)
        self.f.readline()  # greeting
        self.execute("qmp_capabilities")

    def execute(self, name: str, args: dict | None = None) -> dict:
        """Send a QMP command; return the raw reply object (has return/error)."""
        req: dict = {"execute": name}
        if args is not None:
            req["arguments"] = args
        self.f.write((json.dumps(req) + "\n").encode())
        while True:
            line = self.f.readline()
            if not line:
                return {"error": "connection closed"}
            msg = json.loads(line)
            if "return" in msg or "error" in msg:
                return msg
            # async event — skip

    def cmd(self, name: str, args: dict | None = None):
        """Like execute, but raises on error and returns the `return` payload."""
        msg = self.execute(name, args)
        if "error" in msg:
            raise RuntimeError(f"QMP {name}: {msg['error']}")
        return msg["return"]


def screendump_ppm(qmp: Qmp, filename: str, device: str | None = None, head: int | None = None):
    args: dict = {"filename": filename}
    if device is not None:
        args["device"] = device
    if head is not None:
        args["head"] = head
    qmp.cmd("screendump", args)


def ppm_to_png(ppm: str, out_png: str) -> None:
    """Convert QEMU screendump PPM → PNG. Prefer ImageMagick; fall back to macOS sips."""
    magick = shutil.which("magick")
    if magick:
        subprocess.run(
            [magick, ppm, out_png],
            check=True,
            capture_output=True,
        )
        return
    # ImageMagick 6 ships `convert` only.
    convert = shutil.which("convert")
    if convert:
        subprocess.run(
            [convert, ppm, out_png],
            check=True,
            capture_output=True,
        )
        return
    sips = shutil.which("sips")
    if sips:
        subprocess.run(
            [sips, "-s", "format", "png", ppm, "--out", out_png],
            check=True,
            capture_output=True,
        )
        return
    raise RuntimeError(
        "PPM→PNG needs ImageMagick (`magick` or `convert`) or macOS `sips`"
    )


def screendump_png(qmp: Qmp, out_png: str) -> tuple[int, int]:
    with tempfile.NamedTemporaryFile(suffix=".ppm", delete=False) as tmp:
        ppm = tmp.name
    try:
        screendump_ppm(qmp, ppm)
        with open(ppm, "rb") as fh:
            header = fh.read(64).split()
        width, height = int(header[1]), int(header[2])
        ppm_to_png(ppm, out_png)
        return width, height
    finally:
        os.unlink(ppm)


def display_size(qmp: Qmp) -> tuple[int, int]:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        png = tmp.name
    try:
        return screendump_png(qmp, png)
    finally:
        os.unlink(png)


def send_pointer(qmp: Qmp, x: int, y: int, width: int, height: int, buttons=(),
                 button: str = "left"):
    ax = int(x * TABLET_MAX / max(width - 1, 1))
    ay = int(y * TABLET_MAX / max(height - 1, 1))
    move = [
        {"type": "abs", "data": {"axis": "x", "value": ax}},
        {"type": "abs", "data": {"axis": "y", "value": ay}},
    ]
    qmp.cmd("input-send-event", {"events": move})
    for down in buttons:
        time.sleep(0.05)
        send_button(qmp, down, button)


def send_button(qmp: Qmp, down: bool, button: str = "left"):
    """Press or release one pointer button.

    `button` is the QMP `InputButton` name. The right one is not a convenience:
    a dock context menu is only reachable with it, and the menu's close-in
    animation is a reported defect, so a driver that speaks only `left` cannot
    drive that interaction at all.
    """
    qmp.cmd(
        "input-send-event",
        {"events": [{"type": "btn", "data": {"down": down, "button": button}}]},
    )


def drag(qmp: Qmp, points, width: int, height: int, steps: int = 8, hold_s: float = 0.02):
    """Press left button at points[0], glide through the rest (button held),
    release at the last. Interpolates `steps` sub-moves per segment so the guest
    sees a continuous drag (rubber-band selection), not two teleports.

    Returns `(moves, seconds)` for the held glide alone -- the press, its settle
    and the release sit outside the timed window.

    That pair is not diagnostics, it is half of any frame rate measured across a
    drag. An app that pans by tracking the pointer redraws per motion event, so
    it cannot produce more distinct views than it was sent: `moves` is a ceiling
    on the frames such a phase can contain, whatever the device could have
    drawn. A caller that reports frames without reporting this is quoting
    whichever of the two is smaller and cannot say which.

    `hold_s` does not set the rate on its own. Every sub-move is a synchronous
    QMP round-trip -- `Qmp.execute` blocks on the reply, and QEMU answers from
    the main loop it also runs the device on -- so the achieved rate is
    `1 / (hold_s + rtt)`. Read that rate; do not assume it in either direction.
    Measured x86/PCI, macos-15, bare desktop: `rtt` is **0.7-0.9 ms**, so the
    default `hold_s` dominates and this call achieves ~77/s in-glide. A session
    that instead assumed a ~20 ms round-trip talked itself into believing a
    28 fps app was bounded by this loop, which the first measurement refuted.

    **The returned rate is in-glide, and a caller looping this process does not
    achieve it.** The timed window covers the sub-moves only; spawning
    `qmp.py` costs ~0.14 s, so a caller invoking one circuit per iteration pays
    that between every circuit. A 20 s arm of 24-point circuits at the default
    `hold_s` moved 1080 points -- 77/s in-glide but **54/s sustained**, and at
    `hold_s=0` the gap is 1490 against 151. Pass the whole path as one
    invocation, and divide `moves` by your own wall clock for the sustained
    figure.
    """
    if len(points) < 2:
        raise ValueError("drag needs at least two points")
    send_pointer(qmp, points[0][0], points[0][1], width, height)
    send_button(qmp, True)
    time.sleep(0.05)
    moves = 0
    started = time.monotonic()
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        for s in range(1, steps + 1):
            x = x0 + (x1 - x0) * s // steps
            y = y0 + (y1 - y0) * s // steps
            send_pointer(qmp, x, y, width, height)
            moves += 1
            if hold_s:
                time.sleep(hold_s)
    elapsed = time.monotonic() - started
    send_button(qmp, False)
    return moves, elapsed


def char_keys(ch: str):
    """Return (qcodes, shifted) for one ASCII char."""
    if ch.isalpha():
        return [ch.lower()], ch.isupper()
    if ch.isdigit():
        return [ch], False
    if ch in PLAIN:
        return [PLAIN[ch]], False
    if ch in SHIFTED:
        return [SHIFTED[ch]], True
    raise ValueError(f"no qcode mapping for {ch!r}")


def send_keys(qmp: Qmp, qcodes):
    keys = [{"type": "qcode", "data": q} for q in qcodes]
    qmp.cmd("send-key", {"keys": keys})


# `shot` and `screendump` are deliberately removed, not broken. QMP screendump
# reads QEMU's DisplaySurface, so it can only ever see a frame that was copied
# into QEMU's address space. Under the host-owned window the frame stays a
# VkImage on the Rust side and goes straight to the compositor, so a screendump
# there shows a stale CPU-capture surface or nothing at all -- a screenshot that
# silently does not match what is on screen. Capture the actual window instead.
CAPTURE_DISABLED = """\
qmp.py {mode} is disabled -- use the host screenshot helper instead:

  scripts/screenshot/screenshot.sh -o out.png

Why: screendump reads QEMU's DisplaySurface and can only see frames copied into
QEMU's address space. With the host-owned window (REIMS_VGPU_WINDOW=1, QEMU at
-display none) the frame never crosses that boundary, so a screendump does not
show what the window shows. Use the host helper to grab the compositor window.

Note: click/move/drag still size the guest display through QMP internally.
They are input helpers, not screenshot helpers."""


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2

    mode, args = argv[1], argv[2:]

    # Rejected before connecting: the guidance must not be masked by a socket
    # error when no VM is running.
    if mode in ("shot", "screendump"):
        print(CAPTURE_DISABLED.format(mode=mode), file=sys.stderr)
        return 2

    if mode == "cmd" and args and args[0] == "screendump":
        print(CAPTURE_DISABLED.format(mode="cmd screendump"), file=sys.stderr)
        return 2

    qmp = Qmp()

    if mode == "cmd":
        if not args:
            print(__doc__)
            return 2
        req_args = json.loads(args[1]) if len(args) > 1 else None
        reply = qmp.execute(args[0], req_args)
        print(json.dumps(reply, indent=2))
        return 0 if "return" in reply else 1

    # Sizing is not capture, which is why this is allowed where `shot` is not.
    # A screendump of the host-owned window shows the wrong *pixels*, but the
    # DisplaySurface it reads still carries the right dimensions — and that is
    # all `click`/`move`/`drag` have ever used it for, as the note in
    # CAPTURE_DISABLED says. Exposing it means a host-side probe can place a
    # pointer in guest coordinates without asking the guest anything.
    #
    # The guest-side alternatives are all unavailable on at least one rail:
    # `screencapture` fails outright on macOS 26 ("could not create image from
    # display"), and the `osascript` desktop-bounds routes need Apple Events
    # consent that a fresh ssh session does not have.
    if mode == "size":
        w, h = display_size(qmp)
        print(f"{w} {h}")
        return 0

    if mode in ("click", "move", "rclick"):
        if len(args) < 2:
            print(__doc__)
            return 2
        x, y = int(args[0]), int(args[1])
        w, h = display_size(qmp)
        if mode == "move":
            send_pointer(qmp, x, y, w, h)
        elif mode == "rclick":
            send_pointer(qmp, x, y, w, h, buttons=(True, False), button="right")
        else:
            clicks = 2 if "--double" in args else 1
            for i in range(clicks):
                send_pointer(qmp, x, y, w, h, buttons=(True, False))
                if i + 1 < clicks:
                    time.sleep(0.12)
        print(f"{mode} {x},{y} ok")
        return 0

    if mode == "drag":
        if len(args) < 4 or len(args) % 2 != 0:
            print("drag X1 Y1 X2 Y2 [X3 Y3 ...]  (even count, >=2 points)")
            return 2
        coords = [int(a) for a in args]
        points = list(zip(coords[0::2], coords[1::2]))
        w, h = display_size(qmp)
        # Gesture duration and pointer-event count are separately controllable so
        # a caller can hold one fixed while varying the other. A drag's residue
        # scales with one or the other, and with duration welded to event count
        # (steps * hold_s per segment) the two cannot be told apart.
        steps = int(os.environ.get("QMP_DRAG_STEPS", "8"))
        hold_s = float(os.environ.get("QMP_DRAG_HOLD_S", "0.02"))
        moves, secs = drag(qmp, points, w, h, steps=steps, hold_s=hold_s)
        rate = moves / secs if secs > 0 else 0.0
        print(
            f"drag {points} steps={steps} hold_s={hold_s} "
            f"moves={moves} secs={secs:.2f} moves_s={rate:.1f} ok"
        )
        return 0

    if mode == "wheel":
        # wheel [up|down] [COUNT] [INTERVAL_S] — COUNT wheel ticks in ONE QMP
        # connection (per-invocation connect overhead would otherwise dwarf the
        # tick cadence; sustained-scroll load needs back-to-back ticks).
        direction = args[0] if args else "down"
        if direction not in ("up", "down"):
            print("wheel [up|down] [COUNT] [INTERVAL_S]")
            return 2
        count = int(args[1]) if len(args) > 1 else 1
        interval = float(args[2]) if len(args) > 2 else 0.05
        button = f"wheel-{direction}"
        for i in range(count):
            for down in (True, False):
                qmp.cmd(
                    "input-send-event",
                    {"events": [{"type": "btn", "data": {"down": down, "button": button}}]},
                )
            if i + 1 < count:
                time.sleep(interval)
        print(f"wheel {direction} x{count} ok")
        return 0

    if mode == "key":
        if not args:
            print(__doc__)
            return 2
        for chord in args:
            send_keys(qmp, chord.split("+"))
            time.sleep(0.08)
        print("key ok")
        return 0

    if mode == "type":
        if not args:
            print(__doc__)
            return 2
        text = " ".join(args) if len(args) > 1 else args[0]
        for ch in text:
            qcodes, shifted = char_keys(ch)
            send_keys(qmp, (["shift"] if shifted else []) + qcodes)
            time.sleep(0.06)
        print(f"typed {len(text)} chars")
        return 0

    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
