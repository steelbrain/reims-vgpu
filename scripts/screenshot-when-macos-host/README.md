# Reims vGPU screenshot (macOS)

`screenshot-when-macos-host.sh` captures the exact host window titled `Reims vGPU`
on a macOS host. It resolves the window through ScreenCaptureKit and uses a
desktop-independent window filter, so the capture includes compositor-owned
Metal content while the window is inactive or covered and after the guest
changes display resolution. It requires macOS 14 or later.

The script prefers a process whose name contains `qemu-system`, avoiding
terminal or editor windows that happen to mention the display title. Override
the process hint with `REIMS_PROCESS_HINT` when necessary.

## Usage

```sh
scripts/screenshot-when-macos-host/screenshot-when-macos-host.sh /tmp/reims.png
```

With no output argument, the script writes a timestamped PNG under `/tmp`.

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
