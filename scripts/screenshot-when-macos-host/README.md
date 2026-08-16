# Reims vGPU screenshot (macOS)

`screenshot-when-macos-host.sh` captures the guest's display window on a macOS
host. It resolves the window through ScreenCaptureKit and uses a
desktop-independent window filter, so the capture includes compositor-owned
Metal content while the window is inactive or covered and after the guest
changes display resolution. It requires macOS 14 or later.

The script prefers a process whose name contains `qemu-system`, avoiding
terminal or editor windows that happen to mention the display title. Override
the process hint with `REIMS_PROCESS_HINT` when necessary.

## It captures the reference device too, and that is the point

`vm/boot-arm64.sh --device` decides which host window a boot produces:

| device | display | window title |
|---|---|---|
| `reims-vgpu-mmio` | `reims-host-window` | `Reims vGPU` |
| `apple-gfx-mmio` | `cocoa` | `QEMU` |

Both titles are tried in that order and the matched one is printed to stderr, so
a harness that captures without asking still photographs the device under test
when both are up.

This used to name only the first title, and the omission was expensive out of
proportion to its size. `apple-gfx-mmio` is Apple's own
ParavirtualizedGraphics.framework running against the same disk, which makes it
the only thing in this tree that can answer **"is this pixel supposed to look
like that"** — and with no capture route it may as well not have existed. Two
visual defects were argued from our own output alone until one reference capture
settled both: it showed the window corner is meant to be a single ~0.86 blend of
the backdrop rather than opaque black, and that the menu bar renders once rather
than twice.

Pin an exact title with `--window NAME` when a host has two guests up, or when
some other application owns a window called `QEMU`. Check the matched title on
stderr before trusting a capture on a busy host.

## Usage

```sh
scripts/screenshot-when-macos-host/screenshot-when-macos-host.sh /tmp/reims.png
scripts/screenshot-when-macos-host/screenshot-when-macos-host.sh /tmp/ref.png --window QEMU
```

With no output argument, the script writes a timestamped PNG under `/tmp`.

The reference boot picks its own display mode — 1440x1080 against our 1920x1080
on the macos-13 rail — so compare **ratios against the local backdrop**, never
absolute pixel values, and never match a feature by absolute coordinate.

The invoking terminal or application needs Screen & System Audio Recording
permission:

`System Settings → Privacy & Security → Screen & System Audio Recording`

## A sleeping host display fails this the same way a missing permission does

`Failed to start stream due to audio/video capture failure` / `window capture
failed (exit 65)` is what ScreenCaptureKit says when the **host's own display is
asleep**, and it is indistinguishable from the permission failure above. An
unattended agent reads it as "I was never granted Screen Recording", goes
looking for a settings pane it cannot open, and concludes this pathway has no
pixel verification at all — which is how one session ran a driven macos-13 boot
and reported the visual result as unobtainable while the capture was one command
away.

Take a whole-screen `screencapture -x` to tell the two apart. All black at the
full display resolution is a sleeping display; a permission problem does not
produce a well-formed black frame. `pmset -g assertions` confirms it:
`PreventUserIdleDisplaySleep 0` with the system held awake by something else
(a caffeinate-style keep-awake holds `PreventUserIdleSystemSleep` and lets the
*display* sleep regardless) is exactly the state that fails here.

Wake it for the length of the capture and the same command succeeds:

```sh
nohup caffeinate -u -t 90 >/dev/null 2>&1 & disown
scripts/screenshot-when-macos-host/screenshot-when-macos-host.sh /tmp/reims.png
```

`-u` is the part that matters — it declares user activity, which is what wakes
the display; `caffeinate` without it only keeps the *system* awake and changes
nothing here. `nohup … & disown` is not decoration: a `caffeinate` backgrounded
inside a one-shot shell dies with that shell, so the next capture in a sequence
fails again and the whole thing reads as intermittent.

This tool is for macOS hosts only. Linux hosts use
`scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh`.
