#!/usr/bin/env bash
# Capture a screenshot of the "Reims vGPU" window on macOS.
#
# Usage:
#   ./screenshot-when-macos-host.sh [output-path]
#
# If no output path is given, saves to /tmp as:
#   /tmp/Reims-vGPU-YYYYMMDD-HHMMSS.png
#
# Requires Screen & System Audio Recording permission for Terminal (or the app
# running this script).

set -euo pipefail

WINDOW_NAME="Reims vGPU"
# Optional process name hint (QEMU guest display). Exact title still wins.
PROCESS_HINT="${REIMS_PROCESS_HINT:-qemu-system}"
OUTPUT="${1:-}"

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
window_id="$(
  /usr/bin/swift - "$WINDOW_NAME" "$PROCESS_HINT" "$OUTPUT" <<'SWIFT'
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

let targetName = args[0]
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
    let matches = content.windows.filter {
        $0.title == targetName && $0.windowLayer == 0
    }
    guard let window = matches.first(where: {
        $0.owningApplication?.applicationName.range(
            of: processHint,
            options: .caseInsensitive
        ) != nil
    }) ?? matches.first else {
        throw NSError(
            domain: "screenshot-when-macos-host",
            code: 1,
            userInfo: [NSLocalizedDescriptionKey: "window not found: \(targetName)"]
        )
    }

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
        exit(1)
    }
}
dispatchMain()
SWIFT
)" || {
  echo "error: could not find a window named \"${WINDOW_NAME}\"" >&2
  echo "hint: make sure the window is open and Screen Recording is allowed" >&2
  exit 1
}

if [[ ! "$window_id" =~ ^[0-9]+$ ]]; then
  echo "error: failed to resolve a numeric window id: ${window_id}" >&2
  exit 1
fi

echo "Saved screenshot of \"${WINDOW_NAME}\" (window id ${window_id})"
echo "  → ${OUTPUT}"
