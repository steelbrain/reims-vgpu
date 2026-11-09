#!/bin/bash
# The manual walk of a display's window on macOS: every entry of
# crates/reims-vgpu-window/features.md up to "On Linux", plus the refresh
# rate on a ProMotion screen, on vmapple boots, at the Mac, in an unlocked
# session. About 15 minutes. It walks SDL3's window by default; IMPL=appkit
# walks AppKit's, to compare.
#
#   scripts/internal/walk_window_macos.sh [VM_DIR]
#
# VM_DIR is a vmapple guest that logs in by itself (default
# ~/Projects/macos-gpu-re/guest-26). Each boot runs on a fresh APFS clone of
# it under OUT, so the guest itself is never written; each clone is deleted
# when its boot ends, and OUT (the logs) at the end unless KEEP=1.
#
# Each step says what to do and what should happen; answer with Enter or p
# (as described), f (not as described: a note is asked for), or s (skipped:
# a reason is asked for). Steps a unit test already holds say so, and only
# need a look. The script measures what it can itself: the window's and the
# guest's frame rates (REIMS_VGPU_STATS), the screen's refresh rate,
# screendump, and QEMU's exit. The results go to RESULTS, printed at the
# end; send that file.
#
# Environment:
#   QEMU    qemu-system-aarch64 linked with this checkout's library
#           (default build/qemu's: make qemu)
#   IMPL    sdl (default) or appkit: REIMS_VGPU_WINDOW_IMPL
#   OUT     the clones, logs and results (default /tmp/rvg-walk-<time>);
#           on the same APFS volume as VM_DIR, for cp -c
#   KEEP=1  keep OUT (the clones are deleted either way)
#   RESULTS the results file (default /tmp/walk-window-macos-<time>.txt)
#
# Nothing else may use the GPU meanwhile: the rates are measured.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VM_DIR="${1:-$HOME/Projects/macos-gpu-re/guest-26}"
IMPL="${IMPL:-sdl}"
OUT="${OUT:-/tmp/rvg-walk-$(date +%Y%m%d-%H%M%S)}"
QEMU="${QEMU:-$ROOT/build/qemu/qemu-system-aarch64}"
RESULTS="${RESULTS:-/tmp/walk-window-macos-$(date +%Y%m%d-%H%M%S).txt}"

die() { echo "error: $*" >&2; exit 1; }
[[ -f "$VM_DIR/disk.img" && -f "$VM_DIR/aux.img" ]] || die "no guest in $VM_DIR"
[[ -x "$QEMU" ]] || die "$QEMU not found: run make qemu, or set QEMU"
[[ "$IMPL" == sdl || "$IMPL" == appkit ]] || die "IMPL is sdl or appkit"
mkdir -p "$OUT"

QEMU_PID=""
CLONE=""
cleanup() {
  [[ -n "$QEMU_PID" ]] && kill "$QEMU_PID" 2>/dev/null
  [[ -n "$CLONE" ]] && rm -rf "$CLONE"
  [[ "${KEEP:-0}" == 1 ]] || rm -rf "$OUT"
}
# bash 3.2, macOS's, exits 0 from a set -u error where an EXIT trap is set, with $? 0 in the
# trap too, so an aborted run would read as a pass (docs/lessons.md): the trap fails a run
# that exits 0 short of its end, which sets `ended`.
ended=0
trap 'rc=$?; cleanup; [[ $rc != 0 || $ended == 1 ]] || rc=1; exit $rc' EXIT

# Results: one line per step.
record() { printf '%-6s %-5s %s\n' "$1" "$2" "${3:-}" >>"$RESULTS"; }
note() { printf '       %s\n' "$*" >>"$RESULTS"; }

# step ID TEXT: shows TEXT, asks how it went, records the answer.
step() {
  local id=$1 text=$2 answer why
  printf '\n\033[1m[%s]\033[0m %s\n' "$id" "$text"
  while true; do
    read -r -p "  Enter/p as described, f not, s skip: " answer </dev/tty
    case "$answer" in
      "" | p) record "$id" PASS; return ;;
      f) read -r -p "  what happened? " why </dev/tty; record "$id" FAIL "$why"; return ;;
      s) read -r -p "  why skipped? " why </dev/tty; record "$id" SKIP "$why"; return ;;
    esac
  done
}

# wait_for TEXT: shows TEXT and waits for Enter.
wait_for() { read -r -p "$(printf '\n>> %s, then Enter: ' "$1")" _ </dev/tty; }

# The screen's highest refresh rate, from AppKit.
refresh_hz() {
  osascript -l JavaScript -e \
    'ObjC.import("AppKit"); $.NSScreen.mainScreen.maximumFramesPerSecond' 2>/dev/null
}

# rates SECONDS: the mean guest and window frame rates REIMS_VGPU_STATS logs
# over the next SECONDS, as "guest G/s, window W/s".
rates() {
  local from
  from=$(wc -l <"$OUT/qemu.log")
  sleep "$1"
  tail -n +"$((from + 1))" "$OUT/qemu.log" | awk '
    /reims-vgpu stats: port 0:/ {
      for (i = 1; i <= NF; i++) {
        if ($i == "guest") { g += $(i + 1); n++ }
        if ($i == "window") { w += $(i + 1) }
      }
    }
    END { if (n) printf "guest %.1f/s, window %.1f/s over %d s", g / n, w / n, n;
          else printf "no stats lines" }'
}

# boot NAME [VAR=VALUE...]: a fresh clone, QEMU with the window on, until the
# person sees the guest's desktop.
boot() {
  local name=$1; shift
  CLONE="$OUT/clone-$name"
  rm -rf "$CLONE"; mkdir -p "$CLONE"
  cp -c "$VM_DIR/disk.img" "$VM_DIR/aux.img" "$CLONE/" || die "cp -c: OUT must be on VM_DIR's volume"
  : >"$OUT/qemu.log"
  env WINDOW=1 OUT="$OUT" QEMU="$QEMU" MEM=8G CPUS=4 \
    REIMS_VGPU_WINDOW_IMPL="$IMPL" REIMS_VGPU_STATS=1 \
    REIMS_VGPU_LOG="warn,reims_vgpu_window=info" "$@" \
    "$ROOT/scripts/internal/run_vmapple.sh" "$VM_DIR" "$CLONE" >"$OUT/run-$name.log" 2>&1 &
  QEMU_PID=$!
  printf '\n=== boot %s (%s window%s) ===\n' "$name" "$IMPL" "${*:+, $*}"
  record boot "$name" "$IMPL $*"
  wait_for "Wait for the guest's desktop in its window (about a minute)"
  kill -0 "$QEMU_PID" 2>/dev/null || {
    record boot FAIL "QEMU exited: $(tail -3 "$OUT/run-$name.log" "$OUT/qemu.log" | tr '\n' ' ')"
    return 1
  }
}

# ended NAME: waits up to a minute for QEMU to exit after the window closed,
# and records its exit status.
ended() {
  local name=$1 status t=0
  while kill -0 "$QEMU_PID" 2>/dev/null && ((t < 60)); do sleep 1; t=$((t + 1)); done
  if kill -0 "$QEMU_PID" 2>/dev/null; then
    record "L1-$name" FAIL "QEMU still running 60 s after the close"
    kill "$QEMU_PID" 2>/dev/null
  fi
  wait "$QEMU_PID"; status=$?
  QEMU_PID=""
  # 139 at QEMU's own exit is QEMU's, not the library's (out of scope).
  note "QEMU exit status $status after the close ($name)"
  rm -rf "$CLONE"; CLONE=""
}

: >"$RESULTS"
{
  echo "window walk, $(date '+%Y-%m-%d %H:%M'), IMPL=$IMPL"
  echo "checkout $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null), QEMU $QEMU"
  echo "host $(sw_vers -productVersion), $(sysctl -n machdep.cpu.brand_string), screen max $(refresh_hz) Hz"
} >>"$RESULTS"

cat <<EOF
The walk of the $IMPL window: three boots of $VM_DIR, about 15 minutes.
Results go to $RESULTS. Keep the Mac on its built-in screen unless a step
asks for another. Have the guest's Terminal handy: some steps run a command
in it. While input is captured the pointer stays over the guest's image:
press ⌃⌥⌘ to release it whenever a step needs it elsewhere (the bars, the
title bar, an edge to drag, another app), and before you answer here.
EOF
wait_for "Close other GPU-heavy apps"

# ---------------------------------------------------------------- boot 1
boot windowed || exit 1

step O1 "One window opened by itself (QEMU runs with -display none), titled \"Reims vGPU — click to capture input\" (tests: handle::a_window_s_title_is_reims_vgpu_and_its_port, capture::the_title_says_whether_input_is_captured)."
step O2 "The VM is an app: a Dock icon, a ⌘Tab entry, and a menu bar with Quit, Hide and Hide Others, and a Window menu with Minimize, Zoom, Full Screen and Release Input."
step O3 "The window shows the 1920x1080 1x guest a guest point per host point, shrunk to fit the screen's visible area (not at half size), the image scaled up on the GPU and sharp (tests: geometry::a_new_window_*)."
step O4 "Display i opens on screen i: with only the built-in screen, the window is on it. With a second screen, skip unless REIMS_VGPU_WINDOW_SCREENS was set (tests: screens::*)."
step O5 "Two windows on one screen cascade: vmapple has one display, so skip, citing the tests (native::macos::shell::windows_on_one_screen_cascade and two more)."
step M1 "In the guest's System Settings > Displays: 1920x1080 first, with the standard sizes within it, at the screen's highest rate ($(refresh_hz) Hz here) (tests: policy::a_chosen_mode_*)."
step P1 "Colours look as on the host (sRGB, not stretched over a P3 screen's gamut): the guest's wallpaper and System Settings' accent colours are no more saturated than the host's own."
step P2 "One GPU pass with the display's gamma and colour matrix: nothing looks washed out or too dark, and the image has no tearing or stale bands."

wait_for "Open the guest's Terminal (its Finder: Applications > Utilities), click into it and run: while :; do date; done   (the text scrolls without pause); keep the window frontmost and uncovered"
r=$(rates 8)
echo "  measured: $r (screen max $(refresh_hz) Hz)"
record P3 MEAS "$r, screen max $(refresh_hz) Hz"
step P3 "Each guest frame is drawn as it arrives, up to the screen's rate: the window rate above keeps up with the guest's (about $(refresh_hz)/s on ProMotion while the guest presents that fast)."

wait_for "Cover the guest window completely with another app's window (Terminal maximized on the host), leaving the guest's loop running"
r=$(rates 5)
echo "  measured while covered: $r"
record P4 MEAS "covered: $r"
wait_for "Uncover it"
r=$(rates 3)
echo "  measured after: $r"
record P4 MEAS "uncovered: $r"
step P4 "Nothing is drawn while the window is fully hidden (window near 0/s while covered), the guest's frames go on (guest rate as before), and it redraws when shown (tests: native::macos::render::a_hidden_window_is_drawn_when_it_shows)."

wait_for "Minimize the window (⌘M while uncaptured, or the yellow button) and leave it 5 s"
r=$(rates 5)
echo "  measured while minimized: $r"
record L2 MEAS "minimized: $r"
wait_for "Bring it back from the Dock, then stop the loop with ^C in the guest's Terminal"
step L2 "The VM keeps full speed with its window minimized: the guest rate while minimized is the one before (no App Nap)."

step P5 "Drag an edge: the window resizes freely, and the image keeps the guest's ratio, centred on whole pixels between black bars (tests: geometry::keeps_the_guest_ratio, centres_the_image, the_image_is_on_whole_pixels)."
step P6 "In the guest's Displays, pick another mode (e.g. 1680x1050): since you resized the window, the image re-fits at once and the window keeps its size (tests: geometry::a_mode_change_refits_the_image)."
step P7 "Native full screen (green button or ⌃⌘F) keeps the ratio, and leaving it restores the window."
step P8 "Sharp: one drawable pixel per screen pixel (guest text is crisp at 1:1 when the window is sized to half the mode). With a second screen of another scale, move the window there: still sharp. Otherwise check the built-in screen only."

step I1 "Pointer: absolute, where the host's is over the image, held at the nearest edge outside it (tests: geometry::the_pointer_*)."
step I2 "Left, right, middle and (if your mouse has them) side buttons click and drag; a drag in a guest window stops at the image's edge and drops in the guest where its button comes up (tests: native::macos::input::a_click_in_the_image_is_sent_with_its_button, a_drag_keeps_going_outside_the_window)."
step I3 "Press ⌃⌥⌘, click another host app, then click into the guest window: that click only brings the window forward, the guest does not get it (test: native::macos::input::the_click_that_makes_the_window_key_is_not_sent)."
step C1 "Click into the image: the title says input is captured and how to release it; ⌘Tab, ⌘Space, ⌘W and ⌘Q now go to the guest (⌘Q quits the guest's frontmost app, not the VM) (tests: capture::*, native::macos::input::captured_command_keys_reach_the_guest_not_the_menu)."
step C2 "Captured, the pointer stays over the image: push it at each edge and into each corner, fast and slow; it stops at the image, never over the black bars, the title bar, another window, the menu bar or the Dock, and the guest's pointer reaches the guest's own edges (its Apple menu at the top left, its Dock where it shows, a hot corner if one is set) (test: capture::captured_input_holds_the_pointer_over_the_image)."
step C2e "Captured, at each edge and into each corner of the image where it meets the window's edge (where black bars are, that edge is not one): no resize arrows show, a click there and a press dragged inward are the guest's, and the window neither resizes nor moves (tests: capture::a_captured_window_cannot_be_resized, native::macos::shell::a_windowed_window_is_not_resizable_while_captured)."
step C2m "With a trackpad, captured: swipe up with three fingers (Mission Control), then back down; the title says input is captured only while the pointer is still held over the image. No trackpad: skip."
step C2f "Press ⌃⌥⌘, enter native full screen (the green button), click into the image and push the pointer at the screen's top and bottom edges: the hold is as above, but where the image reaches an edge the host's menu bar or Dock may slide in there, as features.md says. Still captured, where the title bar slides in at the top, leave full screen with its green button: the window comes back at its size, still captured, and its edges do not resize it, as in C2e; then press ⌃⌥⌘. Where no title bar slides in, press ⌃⌥⌘ and leave full screen."
step K1 "Captured, in the guest's TextEdit or Terminal: keys arrive by position (the guest's layout applies); left and right Shift, Control, Option and Command are told apart (try right Shift+A)."
step K2 "On an ISO keyboard, the key left of 1 and the key right of left Shift arrive as on a real ISO keyboard. Not ISO: skip (test: keymap::iso_swaps_section_and_grave)."
step K3 "Caps Lock toggles as on a Mac keyboard (one press on, one off)."
step K4 "A held key repeats at the guest's rate, with no doubled characters from the host's repeat (tests: keys::host_repeats_are_dropped, native::macos::input::host_repeats_are_dropped_at_the_view)."
step K5 "Key-ups under Command reach the guest: hold ⌘, tap A then release: the guest selects all once and no key stays down (test: native::macos::input::key_ups_under_command_reach_the_guest)."
step C3 "Press ⌃⌥⌘ together: capture is released, the title says \"click to capture input\" again, the pointer leaves the window freely, and no key stays down in the guest (tests: capture::control_option_command_releases, native::macos::input::the_release_chord_lifts_every_key_and_releases)."
step C3r "Released, drag an edge and a corner of the window: it resizes again, as in P5; native full screen (the green button) enters and leaves as in P7, and the window resizes after it too."
wait_for "Read this first: 10 s after Enter, Finder comes forward by itself. Press Enter, then click into the guest's TextEdit until the title says input is captured, and hold a letter key until Finder comes"
sleep 10
open -a Finder
step C4 "Finder came forward while you held the key: capture was released, the pointer moves freely again, the host's ⌘Tab and Spotlight work again, and the letter stopped repeating in the guest (tests: native::macos::input::losing_focus_*, native::macos::hotkeys::the_hot_keys_come_back_when_the_last_holder_lets_go)."
step C5 "Uncaptured, ⌘ shortcuts drive the host's menu (⌘H hides the VM), and other keys go to the guest (tests: capture::uncaptured_command_goes_to_the_host)."

step S1 "Trackpad or Magic Mouse in a long guest page (Safari or System Settings): the guest scrolls as far as the host and coasts with momentum, as on the host (tests: native::macos::input::smooth_scrolling_sends_the_gesture_to_its_end, appkit_phases_are_the_guests)."
step S2 "Horizontal scrolling reaches the guest (a wide page or Finder's column view)."
step S3 "A gesture begun over the image scrolls the guest to its end, momentum included, even when the pointer leaves the image; one begun over the black bars never scrolls the guest (tests: scroll::a_gesture_begun_*)."
step S4 "A mouse wheel, where you have one, scrolls the guest the lines the host gives, accelerated as on the host; the guest gets the physical direction, its own natural-scrolling setting decides (tests: scroll::*)."

step U1 "Over the image the cursor is the guest's (shape, hot spot, size, scaling with the window); over the bars and the title bar it is the host's arrow (test: native::macos::cursor::the_cursor_is_the_glyph_at_the_images_scale)."
step U2 "Typing in the guest's TextEdit hides the guest's cursor, and the window hides the host's over the image with it (test: native::macos::cursor::a_hidden_guest_cursor_is_invisible)."

shot="$OUT/screendump.png"
echo "screendump $shot -f png" | nc -U "$OUT/monitor.sock" >/dev/null 2>&1
sleep 2
if [[ -s "$shot" ]]; then
  record L3 PASS "screendump wrote $(wc -c <"$shot" | tr -d ' ') bytes with the window open"
else
  record L3 FAIL "screendump wrote nothing with the window open"
fi
echo "  screendump: $(tail -1 "$RESULTS")"

wait_for "Release capture, then close the window with its red button"
step L1 "Closing the window powers the VM off, and QEMU exits."
ended windowed

# ---------------------------------------------------------------- boot 2
boot borderless REIMS_VGPU_WINDOW_BORDERLESS=1 RESOLUTION=host || exit 1
step B1 "Borderless: the window covers the whole screen, with no title bar; on a screen with a camera housing the image fits below it (test: geometry::the_image_fits_below_a_camera_housing)."
step B2 "The menu bar and the Dock hide while the VM is frontmost, and come back when another app is (⌘Tab away and back)."
step B3 "The window takes keys and clicks as a titled one does: click to capture, type in the guest, the pointer stays on the image (below a camera housing too), ⌃⌥⌘ releases."
step M2 "resolution=host: the guest's Displays offers modes that fill the screen, the largest 16:10, 16:9 and 4:3 first (16:10 first of all), then smaller standard sizes, at $(refresh_hz) Hz; on the built-in Retina screen, a HiDPI (2x) desktop (tests: policy::offered_modes_*)."
step B4 "Change the host's screen: System Settings > Displays, pick another \"Larger Text\"/\"More Space\" size: the window re-fills its screen. (With a second screen, unplugging it moves its window to the main screen.) Set it back afterwards."
wait_for "Uncaptured, press ⌘Q"
step L4 "⌘Q while uncaptured powers the VM off, and QEMU exits."
ended borderless

# ---------------------------------------------------------------- boot 3
boot chosen RESOLUTION=2560x1600@200 || exit 1
step M3 "resolution=2560x1600@200: the guest runs a HiDPI desktop that looks like 1280x800, and the windowed window on this Retina screen shows it one guest pixel per screen pixel where half the mode fits (tests: policy::a_chosen_mode_at_a_chosen_ppi, a_chosen_density_tells_the_scale_the_guest_runs)."
step M4 "In the guest's Displays, switch to another offered mode: the window, never resized by you, follows it, top left corner kept, inside the visible area, and the pointer, still captured, is held over the new image at each edge (tests: geometry::a_resize_keeps_the_top_left_corner, a_resize_stays_in_the_visible_area)."
step M5 "Modes the host sets with QEMU's display-modes replace the offered ones: vmapple's built-in apple-gfx-mmio takes the property only through -global, whose string form holds no list; skip unless you have run it otherwise."
wait_for "Close the window"
ended chosen

ended=1
echo
echo "=== results ($RESULTS) ==="
cat "$RESULTS"
fails=$(grep -c ' FAIL ' "$RESULTS")
echo
echo "$fails step(s) failed."
