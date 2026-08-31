# qmp.py

One-shot QMP client for the VM's control socket. `vm/boot-arm64.sh` opens a
per-boot socket in `vm/guest/run/` plus a stable `qmp.sock` symlink — the
default here; override with `QMP_SOCK=/path/to.sock`.

The x86 guest rail (`vm/boot-x86.sh`) puts its per-boot socket + `qmp.sock` symlink
under `vm/disks/run/`, so the arm default does **not** apply there — pass
`QMP_SOCK=vm/disks/run/qmp.sock` (the boot script prints the exact path).

Covers both **raw QMP** commands and GUI input helpers over the machine's
built-in usb-kbd + usb-tablet.

## Raw QMP

```sh
scripts/qmp/qmp.py cmd query-status
scripts/qmp/qmp.py cmd quit
scripts/qmp/qmp.py cmd human-monitor-command '{"command-line":"info usb"}'
```

## GUI helpers

```sh
scripts/qmp/qmp.py click 640 480
scripts/qmp/qmp.py click 640 480 --double
scripts/qmp/qmp.py move 100 100
scripts/qmp/qmp.py drag 900 500 400 500 1400 700   # left-button rubber-band drag through points
scripts/qmp/qmp.py key ret                  # qcodes; chords via +
scripts/qmp/qmp.py key meta_l+q
scripts/qmp/qmp.py type 'hello World_123'
```

Click coordinates are guest display pixels.

## Screenshots

Screenshot capture is intentionally handled by the host helpers, not by QMP.
`qmp.py shot`, `qmp.py screendump`, and `qmp.py cmd screendump` are blocked so a
stale QEMU display surface is not mistaken for the host-owned window.

```sh
scripts/screenshot/screenshot.sh out.png
```

**`click` / `move` / `drag` still size the guest display through QMP
internally.** They are input helpers, not screenshot helpers.

## Why QMP Screenshots Are Blocked

QMP display capture reads QEMU's `DisplaySurface`. The host-owned window presents through the Rust
backend and the compositor, so QMP is the wrong observation surface for screenshots. Use the
cross-platform helper under `scripts/screenshot/`.

**Input delivery is unaffected.** `click` / `move` / `drag` / `key` / `type` ride
`input-send-event` to the guest's usb-kbd + usb-tablet, which is display-
independent; they keep working under `-display none` and the host window. The
pointer helpers still use QMP to learn the display geometry, but input delivery
itself does not depend on QEMU owning the visible window.
