#!/usr/bin/env bash
# Capture a screenshot of the guest's display window on macOS.
#
# Usage:
#   ./screenshot-when-macos-host.sh [output-path] [--window NAME]
#
# If no output path is given, saves to /tmp as:
#   /tmp/Reims-vGPU-YYYYMMDD-HHMMSS.png
#
# WHICH WINDOW. `vm/boot-arm64.sh --device` decides which of two host windows a
# boot produces, and they are titled differently:
#
#   reims-vgpu-mmio   host-owned window,   title "Reims vGPU"  (display=reims-host-window)
#   apple-gfx-mmio    QEMU's cocoa window, title "QEMU"        (display=cocoa)
#
# So both are tried, in that order, and the one that matched is printed. This
# script used to name only the first, which left the *reference* device — Apple's
# own ParavirtualizedGraphics, and the only thing in this tree that can say
# whether a pixel is supposed to look the way it does — with no capture route at
# all. Two visual defects were being argued from our own output alone for want of
# the one command that photographs Apple's.
#
# `--window NAME` pins an exact title when a host has both up at once.
#
# Requires Screen & System Audio Recording permission for Terminal (or the app
# running this script).

set -euo pipefail

# Tab-separated because a window title may contain spaces; the Swift below splits
# on the tab. Order is the search order.
WINDOW_NAMES="${REIMS_WINDOW_NAME:-$(printf 'Reims vGPU\tQEMU')}"
# Optional process name hint (QEMU guest display). Exact title still wins.
PROCESS_HINT="${REIMS_PROCESS_HINT:-qemu-system}"
OUTPUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --window)
      [[ $# -ge 2 ]] || { echo "error: --window needs a title" >&2; exit 64; }
      WINDOW_NAMES="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      OUTPUT="$1"
      shift
      ;;
  esac
done

if [[ -z "$OUTPUT" ]]; then
  timestamp="$(date +%Y%m%d-%H%M%S)"
  OUTPUT="/tmp/Reims-vGPU-${timestamp}.png"
fi

# Resolve absolute path for clearer messaging.
outdir="$(dirname "$OUTPUT")"
outfile="$(basename "$OUTPUT")"
if [[ -d "$outdir" ]]; then
  OUTPUT="$(cd "$outdir" && pwd)/${outfile}"
fi

# ScreenCaptureKit's desktop-independent filter captures the selected window's
# compositor content, including Metal layers, while it is inactive or covered.
#
# `set -e` must not eat the exit status: each failure below needs its own
# advice, and collapsing them into one message is how a permission denial spent
# a session being read as "the VM never came up".
set +e
window_id="$(
  /usr/bin/swift - "$WINDOW_NAMES" "$PROCESS_HINT" "$OUTPUT" <<'SWIFT'
import Foundation
import AppKit
import CoreGraphics
import ImageIO
import ScreenCaptureKit
import UniformTypeIdentifiers

let args = Array(CommandLine.arguments.dropFirst())
guard args.count == 3 else {
    FileHandle.standardError.write(Data("invalid capture arguments\n".utf8))
    exit(64)
}

// Search order, not a set: the product window wins when both are up, so a
// harness that captures without asking still photographs the device under test.
let targetNames = args[0].components(separatedBy: "\t").filter { !$0.isEmpty }
let processHint = args[1]
let outputURL = URL(fileURLWithPath: args[2])

func writePNG(_ image: CGImage, to url: URL) throws {
    guard let destination = CGImageDestinationCreateWithURL(
        url as CFURL,
        UTType.png.identifier as CFString,
        1,
        nil
    ) else {
        throw NSError(
            domain: "screenshot-when-macos-host",
            code: 2,
            userInfo: [NSLocalizedDescriptionKey: "could not create PNG destination"]
        )
    }
    CGImageDestinationAddImage(destination, image, nil)
    guard CGImageDestinationFinalize(destination) else {
        throw NSError(
            domain: "screenshot-when-macos-host",
            code: 3,
            userInfo: [NSLocalizedDescriptionKey: "could not finalize PNG"]
        )
    }
}

@available(macOS 14.0, *)
func capture() async throws -> CGWindowID {
    let content = try await SCShareableContent.excludingDesktopWindows(
        false,
        onScreenWindowsOnly: false
    )
    // First title with a match wins outright — a later title is not consulted,
    // so "both are up" resolves by the order the caller gave rather than by
    // whichever window the system happened to list first.
    var found: SCWindow?
    var foundName = ""
    for name in targetNames {
        let matches = content.windows.filter {
            $0.title == name && $0.windowLayer == 0
        }
        if let window = matches.first(where: {
            $0.owningApplication?.applicationName.range(
                of: processHint,
                options: .caseInsensitive
            ) != nil
        }) ?? matches.first {
            found = window
            foundName = name
            break
        }
    }
    guard let window = found else {
        throw NSError(
            domain: "screenshot-when-macos-host",
            code: 1,
            userInfo: [
                NSLocalizedDescriptionKey:
                    "window not found: \(targetNames.joined(separator: ", "))"
            ]
        )
    }
    FileHandle.standardError.write(Data("matched window: \(foundName)\n".utf8))

    let filter = SCContentFilter(desktopIndependentWindow: window)

    // Size the capture in PIXELS, not points. `SCWindow.frame` is points, and a
    // configuration sized from it captures a 1920x1080 guest as 960x540 on a 2x
    // display — every second guest pixel is gone, so no capture can be compared
    // against the guest's own declared framebuffer. The filter answers both
    // terms itself: `contentRect` is the region in points and `pointPixelScale`
    // is that display's pixels per point, so their product is exactly the pixel
    // count ScreenCaptureKit can deliver without resampling.
    let scale = CGFloat(filter.pointPixelScale)
    let configuration = SCStreamConfiguration()
    configuration.width = max(1, Int((filter.contentRect.width * scale).rounded(.up)))
    configuration.height = max(1, Int((filter.contentRect.height * scale).rounded(.up)))
    configuration.showsCursor = false
    configuration.ignoreShadowsSingleWindow = true
    configuration.captureResolution = .best

    let image = try await SCScreenshotManager.captureImage(
        contentFilter: filter,
        configuration: configuration
    )
    try writePNG(image, to: outputURL)
    return window.windowID
}

guard #available(macOS 14.0, *) else {
    FileHandle.standardError.write(
        Data("ScreenCaptureKit window capture requires macOS 14 or later\n".utf8)
    )
    exit(69)
}

// Ask the system whether this process may capture, rather than inferring it
// from a failure downstream. Without this guard a permission denial surfaces
// only as the SCStream message "Failed to start stream due to audio/video
// capture failure", which names neither the permission nor the process that
// lacks it -- and the process that needs it is whatever invoked this script,
// not QEMU.
//
// No apostrophes in this heredoc. It sits inside a $( ) substitution, and bash
// tracks quotes through the body while it scans for the closing paren, so one
// unpaired quote here is a syntax error at the end of the file.
guard CGPreflightScreenCaptureAccess() else {
    FileHandle.standardError.write(
        Data("screen recording permission denied for this process\n".utf8)
    )
    exit(77)
}

// The desktop-independent filter expects a process with an
// initialized WindowServer connection. A command-line Swift process does not
// create one until AppKit is touched.
let application = NSApplication.shared
application.setActivationPolicy(.prohibited)

Task {
    do {
        print(try await capture())
        exit(0)
    } catch {
        FileHandle.standardError.write(Data("\(error.localizedDescription)\n".utf8))
        // The two failures need different advice, so they get different codes.
        // A missing window means the boot is not up; anything else got past the
        // window lookup and failed inside the capture itself.
        let error = error as NSError
        exit(error.domain == "screenshot-when-macos-host" && error.code == 1 ? 66 : 65)
    }
}
dispatchMain()
SWIFT
)"
status=$?
set -e

case "$status" in
  0) ;;
  66)
    echo "error: no window titled $(printf '%s' "$WINDOW_NAMES" | tr '\t' '/') is open" >&2
    echo "hint: the boot has not reached the host window yet, or QEMU is not running" >&2
    echo "hint: reims-vgpu-mmio titles its window \"Reims vGPU\"; apple-gfx-mmio is cocoa, titled \"QEMU\"" >&2
    exit 1
    ;;
  77)
    echo "error: Screen Recording is not granted to the process running this script" >&2
    echo "hint: System Settings > Privacy & Security > Screen & System Audio Recording." >&2
    echo "hint: grant it to the terminal, IDE or agent harness that invoked this — not to QEMU." >&2
    echo "hint: the permission is per-app and a newly-granted app must be restarted." >&2
    exit 77
    ;;
  69)
    exit 69
    ;;
  *)
    echo "error: window capture failed (exit ${status}); the line above is the system's own reason" >&2
    exit 1
    ;;
esac

if [[ ! "$window_id" =~ ^[0-9]+$ ]]; then
  echo "error: failed to resolve a numeric window id: ${window_id}" >&2
  exit 1
fi

echo "Saved screenshot (window id ${window_id}); the matched title is on stderr above"
echo "  → ${OUTPUT}"
