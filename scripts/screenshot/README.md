# Reims vGPU host screenshot

`screenshot.sh` captures the exact host-owned window titled `Reims vGPU` on
either KDE Plasma 6 Wayland or macOS 14 and later. It uses `capture-app`, so an
occluded window or a window on another desktop is captured without restoring,
focusing, moving, or otherwise disturbing it.

The helper always asks for native rendering resolution first. It validates the
PNG and rejects an almost-entirely-black frame before downsampling it to fit a
1280x720 box. Pass `--1440p` to fit it within 2560x1440 instead, or set
`REIMS_SHOT_NATIVE=1` for a measurement where resampling is itself a confound.
Aspect ratio is preserved and smaller captures are never enlarged.

## Requirements

- [`capture-app`](https://github.com/steelbrain/capture-app)
- Python 3
- ImageMagick (`magick` or `convert`)
- Screen capture permission on macOS, or a reachable Plasma 6 Wayland session

## Usage

```sh
# Randomly named PNG under /tmp; final path printed on stdout
scripts/screenshot/screenshot.sh

# Explicit output path, capped at 720p
scripts/screenshot/screenshot.sh -o /tmp/reims.png

# Higher-detail review image
scripts/screenshot/screenshot.sh --1440p -o /tmp/reims-1440p.png

# Capture one of several identically titled VM windows
scripts/screenshot/screenshot.sh --window WINDOW_ID -o /tmp/reims.png

# Include compositor-owned window decorations
scripts/screenshot/screenshot.sh --decorations -o /tmp/reims.png
```

Run `capture-app --json list` to obtain a window ID. Diagnostics go to stderr;
stdout contains only the final PNG path.

The invoking terminal or application needs Screen & System Audio Recording
permission on macOS:

`System Settings -> Privacy & Security -> Screen & System Audio Recording`
