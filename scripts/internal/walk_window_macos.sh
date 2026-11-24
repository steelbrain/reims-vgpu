#!/bin/bash
# The manual walk of a display's window on macOS: every entry of
# crates/reims-vgpu-window/features.md up to "On Linux", plus the refresh
# rate on a ProMotion screen, on boots of an arm64 macOS guest with the
# reims-graphics device on Ambrosia under HVF (docs/qemu.md, "arm64 guests"),
# at the Mac, in an unlocked session. About 18 minutes. It walks SDL3's window
# by default; IMPL=appkit walks AppKit's, to compare.
#
#   scripts/internal/walk_window_macos.sh VM_DIR
#
# VM_DIR is an arm64 guest that logs in by itself, as
# firmware/ambrosia/check-boot.sh takes one: disk.img, aux.img.trimmed (or
# aux.img, which a trimmed copy is made of) and vm.toml, whose ecid is the
# guest's ECID. Each boot runs on fresh APFS clones of its disks under OUT, so
# the guest itself is never written; each clone is deleted when its boot
# ends, and OUT (the logs) at the end unless KEEP=1. The guest boots as
# check-boot.sh boots it, with 8 GiB and 4 CPUs, Ambrosia, our configuration
# bundle and reims-graphics-pci, and with the library's window open: QEMU runs
# under -display none, and the device's display opens its window.
#
# Each step says what to do and what should happen; answer with Enter or p
# (as described), f (not as described: a note is asked for), or s (skipped:
# a reason is asked for). Steps a unit test already holds say so, and only
# need a look. The script measures what it can itself: the guest's present
# rate (the display's flips to the surfaces the guest presents, which the
# library logs at debug as reims_vgpu::graphics::shown), the screen's refresh
# rate, screendump, and QEMU's exit. The results go to RESULTS, printed at the
# end; send that file.
#
# Environment:
#   AMBROSIA  Ambrosia's image (no default): its target/ambrosia.bin, or a
#           release's ambrosia.bin, one that boots the guest (docs/qemu.md,
#           "arm64 guests: the boot check of our configuration")
#   BUNDLE  the configuration bundle with our arm64e kexts (default
#           firmware/ambrosia/build/Config.cpio), as docs/qemu.md makes it:
#             make -C firmware/ambrosia config-bundle \
#               GRAFT=$PWD/guest/prebuilt-arm64e/Graft-arm64e.kext \
#               REIMS_GRAPHICS=$PWD/guest/prebuilt-arm64e/ReimsGraphics.kext \
#               RGFXGATE=$PWD/guest/prebuilt-arm64e/RgfxGate.kext
#   QEMU    qemu-system-aarch64 with reims-graphics-pci on this checkout's
#           library (default build/qemu-aarch64's, configured as
#           docs/qemu.md's "arm64 guests: the device on Ambrosia" says)
#   IMPL    sdl (default) or appkit: REIMS_VGPU_WINDOW_IMPL
#   OUT     the clones, logs and results (default /tmp/rvg-walk-<time>);
#           on the same APFS volume as VM_DIR, for cp -c
#   KEEP=1  keep OUT (the clones are deleted either way)
#   RESULTS the results file (default /tmp/walk-window-macos-<time>.txt)
#
# Nothing else may use the GPU meanwhile: the rates are measured.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VM_DIR="${1:-}"
IMPL="${IMPL:-sdl}"
OUT="${OUT:-/tmp/rvg-walk-$(date +%Y%m%d-%H%M%S)}"
QEMU="${QEMU:-$ROOT/build/qemu-aarch64/qemu-system-aarch64}"
AMBROSIA="${AMBROSIA:-}"
BUNDLE="${BUNDLE:-$ROOT/firmware/ambrosia/build/Config.cpio}"
RESULTS="${RESULTS:-/tmp/walk-window-macos-$(date +%Y%m%d-%H%M%S).txt}"

die() { echo "error: $*" >&2; exit 1; }
[[ -n "$VM_DIR" ]] || die "usage: scripts/internal/walk_window_macos.sh VM_DIR"
[[ -f "$VM_DIR/disk.img" ]] || die "no disk.img in $VM_DIR"
[[ -f "$VM_DIR/aux.img.trimmed" || -f "$VM_DIR/aux.img" ]] || die "no aux.img.trimmed or aux.img in $VM_DIR"
ECID=$(sed -n 's/^ecid *= *"\{0,1\}\([0-9a-fA-Fx]*\)"\{0,1\} *$/\1/p' "$VM_DIR/vm.toml" 2>/dev/null | head -1)
[[ -n "$ECID" ]] || die "no ecid in $VM_DIR/vm.toml"
[[ -n "$AMBROSIA" && -f "$AMBROSIA" ]] || die "no Ambrosia image: set AMBROSIA to its ambrosia.bin"
[[ -f "$BUNDLE" ]] || die "no configuration bundle $BUNDLE: make -C firmware/ambrosia config-bundle, or set BUNDLE"
[[ -x "$QEMU" ]] || die "$QEMU not found: build it as docs/qemu.md's \"arm64 guests\" says, or set QEMU"
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
# shellcheck disable=SC2154 # rc is assigned in the trap itself, as it runs
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

# rates SECONDS: the guest's present rate over the next SECONDS, as the
# display's flips to a surface the guest presented that QEMU's log gains
# meanwhile: "guest G/s over S s". WindowServer presents through three
# surfaces in turn, so each of its presents is a flip.
rates() {
  local from n
  from=$(wc -l <"$OUT/qemu.log")
  sleep "$1"
  n=$(tail -n +"$((from + 1))" "$OUT/qemu.log" | grep -c -E 'the display shows IOSurface [1-9]')
  awk -v n="${n:-0}" -v s="$1" 'BEGIN { printf "guest %.1f/s over %d s", n / s, s }'
}

# boot NAME [RESOLUTION=WxH] [VAR=VALUE...]: fresh clones, QEMU with the
# window on, until the person sees the guest's desktop. RESOLUTION is the
# device's resolution property, the size the window opens for; the other
# VAR=VALUEs are QEMU's environment (the library's REIMS_VGPU_WINDOW_*).
boot() {
  local name=$1 a device=reims-graphics-pci; shift
  local envs
  envs=()
  for a in "$@"; do
    case "$a" in
      RESOLUTION=*) device="reims-graphics-pci,resolution=${a#RESOLUTION=}" ;;
      *) envs+=("$a") ;;
    esac
  done
  CLONE="$OUT/clone-$name"
  rm -rf "$CLONE"; mkdir -p "$CLONE"
  cp -c "$VM_DIR/disk.img" "$CLONE/" || die "cp -c: OUT must be on VM_DIR's volume"
  if [[ -f "$VM_DIR/aux.img.trimmed" ]]; then
    cp -c "$VM_DIR/aux.img.trimmed" "$CLONE/" || die "cp -c: OUT must be on VM_DIR's volume"
  else
    dd if="$VM_DIR/aux.img" of="$CLONE/aux.img.trimmed" bs=16384 skip=1 2>/dev/null ||
      die "can't trim $VM_DIR/aux.img"
  fi
  : >"$OUT/qemu.log"
  rm -f "$OUT/monitor.sock"
  env REIMS_VGPU_WINDOW_IMPL="$IMPL" \
    REIMS_VGPU_LOG="warn,reims_vgpu_window=info,reims_vgpu::graphics::shown=debug" \
    ${envs[@]+"${envs[@]}"} \
    "$QEMU" -accel hvf,kernel-irqchip=on -M "vmapple,uuid=$ECID" -m 8G -smp 4 -bios "$AMBROSIA" \
    -drive "file=$CLONE/aux.img.trimmed,if=pflash,format=raw" \
    -drive "file=$CLONE/disk.img,if=pflash,format=raw" \
    -drive "file=$CLONE/aux.img.trimmed,if=none,id=aux,format=raw" \
    -drive "file=$CLONE/disk.img,if=none,id=root,format=raw" \
    -device vmapple-virtio-blk-pci,variant=aux,drive=aux \
    -device vmapple-virtio-blk-pci,variant=root,drive=root \
    -netdev user,id=net0,ipv6=off -device virtio-net-pci,netdev=net0,romfile= \
    -device "$device" \
    -device "loader,file=$BUNDLE,addr=0x120000000,force-raw=on" \
    -display none -serial "file:$OUT/serial.log" \
    -monitor "unix:$OUT/monitor.sock,server=on,wait=off" \
    2>"$OUT/qemu.log" &
  QEMU_PID=$!
  printf '\n=== boot %s (%s window%s) ===\n' "$name" "$IMPL" "${*:+, $*}"
  record boot "$name" "$IMPL $*"
  wait_for "Wait for the guest's desktop in its window (about a minute)"
  kill -0 "$QEMU_PID" 2>/dev/null || {
    record boot FAIL "QEMU exited: $(tail -3 "$OUT/serial.log" "$OUT/qemu.log" | tr '\n' ' ')"
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
  echo "Ambrosia $AMBROSIA, bundle $BUNDLE, guest $VM_DIR"
  echo "host $(sw_vers -productVersion), $(sysctl -n machdep.cpu.brand_string), screen max $(refresh_hz) Hz"
} >>"$RESULTS"

cat <<EOF
The walk of the $IMPL window: four boots of $VM_DIR, about 18 minutes.
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
step O3 "The window opened at 1920x1080 points (the device's default resolution), a guest pixel per host point, shrunk to fit the screen's visible area where it does not fit (not at half size), and the guest's desktop fills it at its pixels: HiDPI on a Retina screen, so it looks like the window's size in points, and sharp (tests: geometry::a_new_window_*, geometry::the_layout_carries_the_screen_s_scale)."
step P6 "Through the boot the window kept the size it opened at: Ambrosia's console and the kernel's progress showed in it, fitted, and the desktop came up filling it, with no resize of the window (tests: geometry::a_mode_change_refits_the_image)."
step O4 "Display i opens on screen i: with only the built-in screen, the window is on it. With a second screen, skip unless REIMS_VGPU_WINDOW_SCREENS was set (tests: screens::*)."
step O5 "Two windows on one screen cascade: the reims-graphics device has one display, so skip, citing the tests (native::macos::shell::windows_on_one_screen_cascade and two more)."
step M1 "In the guest's System Settings > Displays: one mode, the window's size in pixels with its width a multiple of 16 and its height even, HiDPI on a Retina screen, at the rate the host screen is set to (its highest is $(refresh_hz) Hz here) (tests: geometry::a_drawable_offers_a_mode_a_multiple_of_16_wide_drawn_unscaled, screens::a_screen_runs_at_its_current_mode_s_rate)."
step P1 "Colours look as on the host (sRGB, not stretched over a P3 screen's gamut): the guest's wallpaper and System Settings' accent colours are no more saturated than the host's own."
step P2 "One GPU pass with the display's gamma and colour matrix: nothing looks washed out or too dark, and the image has no tearing or stale bands."

wait_for "Open the guest's Terminal (its Finder: Applications > Utilities), click into it and run: while :; do date; done   (the text scrolls without pause); keep the window frontmost and uncovered"
r=$(rates 8)
echo "  measured: $r (screen max $(refresh_hz) Hz)"
record P3 MEAS "$r, screen max $(refresh_hz) Hz"
step P3 "Each guest frame is drawn as it arrives, up to the screen's rate: the guest's rate above is about the screen's ($(refresh_hz)/s on ProMotion while the guest presents that fast), and the scrolling in the window is as smooth as that, with no frame held or skipped."

wait_for "Cover the guest window completely with another app's window (Terminal maximized on the host), leaving the guest's loop running"
r=$(rates 5)
echo "  measured while covered: $r"
record P4 MEAS "covered: $r"
wait_for "Uncover it"
r=$(rates 3)
echo "  measured after: $r"
record P4 MEAS "uncovered: $r"
step P4 "Nothing is drawn while the window is fully hidden, the guest's frames go on (its rate while covered as before), and it redraws when shown, with no stale image (tests: native::macos::render::a_hidden_window_is_drawn_when_it_shows)."

wait_for "Minimize the window (⌘M while uncaptured, or the yellow button) and leave it 5 s"
r=$(rates 5)
echo "  measured while minimized: $r"
record L2 MEAS "minimized: $r"
wait_for "Bring it back from the Dock, then stop the loop with ^C in the guest's Terminal"
step L2 "The VM keeps full speed with its window minimized: the guest rate while minimized is the one before (no App Nap)."

step P5 "Drag an edge: the window resizes by 16 pixels wide and 2 high at a time, and the guest takes each new size as its mode: its desktop re-lays out to fill the window, with no black beside it; a zoom or tiling, which take no steps, leaves at most 15 columns and a row of black at the right and bottom (tests: native::macos::shell::a_windowed_window_resizes_by_a_mode_s_steps, geometry::a_drawable_offers_a_mode_a_multiple_of_16_wide_drawn_unscaled)."
step P7 "Native full screen (green button or ⌃⌘F): the guest takes the full-screen window's size as its mode and fills it; leaving it restores the window, and the guest takes its size again."
step P8 "Sharp: one guest pixel per screen pixel, guest text crisp at every window size. With a second screen of another scale, move the window there: still sharp, the guest told the new screen's scale (tests: geometry::the_layout_carries_the_screen_s_scale). Otherwise check the built-in screen only."

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
boot borderless REIMS_VGPU_WINDOW_BORDERLESS=1 || exit 1
step B1 "Borderless: the window covers the whole screen, with no title bar; on a screen with a camera housing the image fits below it (test: geometry::the_image_fits_below_a_camera_housing)."
step B2 "The menu bar and the Dock hide while the VM is frontmost, and come back when another app is (⌘Tab away and back)."
step B3 "The window takes keys and clicks as a titled one does: click to capture, type in the guest, the pointer stays on the image (below a camera housing too), ⌃⌥⌘ releases."
step M2 "In the guest's System Settings > Displays: one mode, the borderless window's pixels with its width a multiple of 16 and its height even, HiDPI on a Retina screen, at the rate the host screen is set to (its highest is $(refresh_hz) Hz here) (tests: geometry::a_drawable_offers_a_mode_a_multiple_of_16_wide_drawn_unscaled, screens::a_screen_runs_at_its_current_mode_s_rate)."
step B4 "Change the host's screen: System Settings > Displays, pick another \"Larger Text\"/\"More Space\" size: the window re-fills its screen, and the guest takes its new size. (With a second screen, unplugging it moves its window to the main screen.) Set it back afterwards."
wait_for "Uncaptured, press ⌘Q"
step L4 "⌘Q while uncaptured powers the VM off, and QEMU exits."
ended borderless

# ---------------------------------------------------------------- boot 3
boot chosen RESOLUTION=2560x1600 || exit 1
step M3 "resolution=2560x1600: the window opened for a 2560x1600 guest, a guest pixel per host point, shrunk to fit the screen's visible area where it does not fit, and the guest's desktop fills it at its pixels, HiDPI on a Retina screen (tests: geometry::a_new_window_shows_a_guest_point_per_host_point, geometry::a_new_window_shrinks_to_the_visible_area)."
wait_for "Close the window"
ended chosen

# ---------------------------------------------------------------- boot 4
boot nocapture REIMS_VGPU_WINDOW_CAPTURE=0 || exit 1
step N1 "Capture off: the title ends \"— input capture off\", and names no click and no keys to release (tests: capture::capture_off_never_captures)."
step N2 "Click into the image and move around: the guest's pointer follows over the image, and the host's pointer leaves the window at every edge for the title bar, the menu bar, the Dock and other windows; nothing holds it. A drag begun in a guest window goes on outside the image, stops at its edge in the guest, and drops there when the button comes up (tests: capture::capture_off_sends_the_pointer_outside_the_image_only_in_a_drag, native::macos::input::capture_off_sends_keys_and_the_pointer_over_the_image_and_holds_nothing)."
step N3 "With the window in front, keys reach the guest's TextEdit or Terminal, ⌘ shortcuts too (⌘A selects all in the guest, the VM's menu does not flash); ⌘Tab and ⌘Space stay the Mac's; ⌃⌥⌘ releases nothing and the title stays as it was."
step N4 "After clicks into the image, drag an edge and a corner of the window: it resizes as in P5."
wait_for "Quit the VM from its menu bar (the app menu's Quit), since ⌘Q is the guest's"
step N5 "The menu's Quit powers the VM off, and QEMU exits."
ended nocapture

ended=1
echo
echo "=== results ($RESULTS) ==="
cat "$RESULTS"
fails=$(grep -c ' FAIL ' "$RESULTS")
echo
echo "$fails step(s) failed."
