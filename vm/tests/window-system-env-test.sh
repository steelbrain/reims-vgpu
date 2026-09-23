#!/usr/bin/env bash
#
# vm/tests/window-system-env-test.sh — contract test for the host window's
# window-system selector (vm/window-system-env.sh). No VM and no display are
# needed: the selector only reads and writes environment.
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER="$HERE/../window-system-env.sh"
[ -r "$HELPER" ] || { echo "window-system-env-test: missing $HELPER" >&2; exit 1; }

fail() { echo "window-system-env-test: $*" >&2; exit 1; }

# run <mode> <extra env...> -- prints the child env after resolution and exits
# with the selector's status. env -i keeps the host environment out of the test.
run() {
  env -i REIMS_VGPU_WINDOW_SYSTEM="$1" "${@:2}" bash --noprofile --norc -c '
    source "'"$HELPER"'"
    reims_resolve_window_system || exit $?
    printf "DISPLAY=%s\n" "${DISPLAY-<unset>}"
    printf "WAYLAND_DISPLAY=%s\n" "${WAYLAND_DISPLAY-<unset>}"
    printf "WAYLAND_SOCKET=%s\n" "${WAYLAND_SOCKET-<unset>}"
  '
}

# --- x11 mode: DISPLAY wins, Wayland names are removed -----------------------
out="$(run x11 DISPLAY=:0 WAYLAND_DISPLAY=wayland-1 WAYLAND_SOCKET=/run/user/1000/wayland-1)" \
  || fail "x11 with DISPLAY should succeed"
grep -qx 'DISPLAY=:0' <<<"$out" || fail "x11 did not keep DISPLAY: $out"
grep -qx 'WAYLAND_DISPLAY=<unset>' <<<"$out" || fail "x11 left WAYLAND_DISPLAY: $out"
grep -qx 'WAYLAND_SOCKET=<unset>' <<<"$out" || fail "x11 left WAYLAND_SOCKET: $out"
echo "T009_VGPU_X11_SELECTOR=PASS"

# --- x11 mode without DISPLAY is a refusal, never a Wayland fallback ---------
out="$(run x11 WAYLAND_DISPLAY=wayland-1 2>&1)" && fail "x11 without DISPLAY must fail"
grep -q 'requires DISPLAY' <<<"$out" || fail "x11 refusal message missing: $out"
out="$(run x11 DISPLAY= 2>&1)" && fail "x11 with empty DISPLAY must fail"
echo "T009_VGPU_X11_REQUIRES_DISPLAY=PASS"

# --- auto keeps the historical development behavior --------------------------
out="$(run auto WAYLAND_DISPLAY=wayland-1 DISPLAY=:42 WAYLAND_SOCKET=/tmp/w)" \
  || fail "auto should always succeed"
grep -qx 'WAYLAND_DISPLAY=wayland-1' <<<"$out" || fail "auto must not touch WAYLAND_DISPLAY: $out"
grep -qx 'DISPLAY=:42' <<<"$out" || fail "auto must not touch DISPLAY: $out"
grep -qx 'WAYLAND_SOCKET=/tmp/w' <<<"$out" || fail "auto must not touch WAYLAND_SOCKET: $out"
out="$(run auto)" || fail "auto with no display at all must still succeed"
grep -qx 'DISPLAY=<unset>' <<<"$out" || fail "auto must not invent DISPLAY: $out"
echo "T009_VGPU_AUTO_COMPAT=PASS"

# --- wayland mode requires a real Wayland name -------------------------------
out="$(run wayland WAYLAND_DISPLAY=wayland-0)" || fail "wayland with WAYLAND_DISPLAY should succeed"
out="$(run wayland 2>&1)" && fail "wayland without a socket must fail"
grep -q 'requires WAYLAND_DISPLAY or WAYLAND_SOCKET' <<<"$out" || fail "wayland refusal message missing: $out"
echo "T009_VGPU_WAYLAND_SELECTOR=PASS"

# --- an unknown mode is refused, not treated as auto ------------------------
out="$(run banana DISPLAY=:0 2>&1)" && fail "invalid mode must fail"
grep -q 'invalid REIMS_VGPU_WINDOW_SYSTEM' <<<"$out" || fail "invalid-mode message missing: $out"
echo "T009_VGPU_INVALID_MODE_REFUSED=PASS"

echo "T009_VGPU_WINDOW_SYSTEM_TEST_PASS"
