#!/usr/bin/env python3
"""drag-load.py — the load phase of `vibrancy-latch-probe`, with a real drag.

The user's report is "run testufo at count=8 **and drag the window around**".
The probe's original load phase could not do the second half: it teleports the
window with `set position` through System Events, and a window the AX API moves
does not take the same path through the window server as a pointer held on a
title bar. `window-drag-probe` explains why a guest-side `CGEventPost` cannot
be used — the posting process is not trusted for Accessibility, and TCC.db is
unwritable under SIP — and then settles for repositioning for the same reason.

Neither of them needed to. QEMU's own usb-tablet is a HID device: an
`input-send-event` pointer stream arrives at the window server as a real mouse,
with no trust to arrange, and `scripts/qmp/qmp.py drag` already speaks it. That
is what this script uses, so the motion here is a drag session rather than a
sequence of teleports.

Both stressors run, because they are different ones. The drag is the reported
trigger. The resize and the second app opening and closing are what a previous
session measured as the thing that moves `type4_pages_stale` off zero — surface
*reallocation*, which a window that only moves never does.

Usage:
  drag-load.py --seconds N [--guest macos-vm] [--qmp PATH] [--app Safari]

Prints one summary line and exits 2 if the window never actually moved, so a
run that produced no motion cannot be read as a load the device shrugged off.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QMP = os.path.join(REPO, "scripts", "qmp", "qmp.py")

# The guest's own display, which is what qmp.py's pointer helpers take.
GUEST_W, GUEST_H = 1920, 1080
# A title bar is grabbed a little below the window's top edge. macOS puts the
# window's AX origin at the top-left of the frame including the title bar, and
# the bar is ~28 pt tall, so the middle of it is a safe grab point that is not
# on the traffic lights (which are at the left).
TITLE_GRAB_DY = 14
# Far enough right of the traffic lights that a press lands on draggable chrome
# rather than on a button.
TITLE_GRAB_DX = 240


def sh(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, **kw)


class Guest:
    def __init__(self, host: str, app: str):
        self.host = host
        self.app = app

    def osa(self, script: str) -> str:
        return sh(["ssh", "-o", "BatchMode=yes", self.host, "osascript -e " + shquote(script)]).stdout.strip()

    def proc(self, body: str) -> str:
        return self.osa(f'tell application "System Events" to tell process "{self.app}" to {body}')

    def position(self):
        out = self.proc("get position of window 1")
        nums = [int(n) for n in re.findall(r"-?\d+", out)]
        return (nums[0], nums[1]) if len(nums) >= 2 else None

    def set_frame(self, x, y, w, h):
        self.proc(f"set position of window 1 to {{{x}, {y}}}")
        self.proc(f"set size of window 1 to {{{w}, {h}}}")

    def raise_front(self):
        self.proc("set frontmost to true")

    def open_app(self, name):
        sh(["ssh", "-o", "BatchMode=yes", self.host, f"open -a {shquote(name)}"])

    def quit_app(self, name):
        self.osa(f'tell application "{name}" to quit')


def shquote(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def drag(qmp_sock: str, points) -> bool:
    """One real left-button drag through `points`, as guest display pixels."""
    args = [QMP, "drag"]
    for x, y in points:
        args += [str(max(0, min(GUEST_W - 1, x))), str(max(0, min(GUEST_H - 1, y)))]
    env = dict(os.environ, QMP_SOCK=qmp_sock)
    return sh(args, env=env).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--guest", default=os.environ.get("GUEST", "macos-vm"))
    ap.add_argument("--app", default="Safari")
    ap.add_argument(
        "--qmp", default=os.path.join(REPO, "vm", "disks", "run", "qmp.sock")
    )
    ap.add_argument("--churn-apps", default="TextEdit,Calculator,Preview")
    args = ap.parse_args()

    if not os.path.exists(args.qmp):
        print(f"drag-load: no QMP socket at {args.qmp}", file=sys.stderr)
        return 2

    guest = Guest(args.guest, args.app)
    churn = [a for a in args.churn_apps.split(",") if a]

    # A known frame to start from, so the first grab point is known rather than
    # read back. Everything after tracks the drag's own destination.
    x, y, w, h = 300, 120, 1100, 700
    guest.set_frame(x, y, w, h)
    time.sleep(1.5)
    start = guest.position()

    # Destinations the drag walks between. Each is a window origin; the pointer
    # path is derived from it, so the window really is carried there.
    stops = [(760, 150), (240, 320), (700, 420), (180, 140), (620, 260)]
    sizes = [(1400, 900), (700, 500), (1200, 780), (1600, 1000), (900, 620)]

    t0 = time.time()
    n_drag = n_resize = n_app = 0
    moved_mid = None
    i = 0
    while time.time() - t0 < args.seconds:
        dx, dy = stops[i % len(stops)]
        # Pointer path: grab the title bar at the current origin, then move to
        # where the same grab point would be at the destination, through two
        # intermediate points so the window server sees a real drag session and
        # not a single jump.
        gx, gy = x + TITLE_GRAB_DX, y + TITLE_GRAB_DY
        tx, ty = dx + TITLE_GRAB_DX, dy + TITLE_GRAB_DY
        path = [
            (gx, gy),
            (gx + (tx - gx) // 3, gy + (ty - gy) // 3),
            (gx + 2 * (tx - gx) // 3, gy + 2 * (ty - gy) // 3),
            (tx, ty),
        ]
        if drag(args.qmp, path):
            n_drag += 1
            x, y = dx, dy
        if i % 2 == 1:
            w, h = sizes[i % len(sizes)]
            guest.set_frame(x, y, w, h)
            n_resize += 1
        if churn:
            if i % 6 == 3:
                guest.open_app(churn[i % len(churn)])
                n_app += 1
                # The launched app comes to the front, and the next grab point is
                # computed from *this* window's origin — so without raising it
                # back the drag would press on the other app's chrome and the
                # tracked position would stop describing anything.
                guest.raise_front()
            elif i % 6 == 0 and i:
                guest.quit_app(churn[i % len(churn)])
                guest.raise_front()
        if i == 2:
            moved_mid = guest.position()
        i += 1

    for a in churn:
        guest.quit_app(a)

    end = guest.position()
    print(
        f"drag-load: drags={n_drag} resizes={n_resize} app_launches={n_app} "
        f"start={start} mid={moved_mid} end={end}"
    )
    # The mid-run sample is the one that matters: the destinations cycle, so
    # where the window stops is whichever one the last iteration reached, and
    # comparing that to the start reports "never moved" for a run that moved
    # many times and came back.
    if n_drag == 0:
        print("drag-load: no drag was delivered", file=sys.stderr)
        return 2
    if moved_mid is not None and start is not None and moved_mid == start:
        print(
            "drag-load: the pointer stream never moved the window — the load "
            "measured an idle guest",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
