#!/usr/bin/env bash
#
# vm/window-system-env.sh — resolve which window system the host-owned Reims
# window opens on, before QEMU/reims-vgpu is launched.
#
# The window is a winit 0.30 window. That version has no `WINIT_UNIX_BACKEND`;
# it chooses X11 or Wayland from the environment and prefers Wayland whenever
# both a Wayland socket and an X11 `DISPLAY` are present:
#
#   WAYLAND_DISPLAY / WAYLAND_SOCKET  ->  Wayland
#   DISPLAY                           ->  X11
#
# A caller that wants the dedicated X11 session cannot express that by setting
# `DISPLAY` alone, because launching from a Wayland desktop leaves
# `WAYLAND_DISPLAY` in the environment and winit silently follows it. The
# selector below is the single place that decides, so the product has one
# contract instead of an env sniff at each call site.
#
#   REIMS_VGPU_WINDOW_SYSTEM=auto     (default) leave the environment alone;
#                                     the historical development behaviour,
#                                     which works on both a Wayland desktop and
#                                     an X11 session.
#   REIMS_VGPU_WINDOW_SYSTEM=x11      require a non-empty `DISPLAY` and remove
#                                     the Wayland variables so winit cannot
#                                     prefer Wayland. Missing `DISPLAY` is a
#                                     refusal, never a fallback.
#   REIMS_VGPU_WINDOW_SYSTEM=wayland  require an existing Wayland socket and
#                                     leave the standard Wayland variables.
#
# Refusals are explicit: this function returns 64 (EX_USAGE) and writes one
# line to stderr. Callers must propagate it.
reims_resolve_window_system() {
  local mode="${REIMS_VGPU_WINDOW_SYSTEM:-auto}"
  case "$mode" in
    auto)
      return 0
      ;;
    x11)
      if [ -z "${DISPLAY:-}" ]; then
        echo "reims-vgpu: REIMS_VGPU_WINDOW_SYSTEM=x11 requires DISPLAY to be set and non-empty" >&2
        return 64
      fi
      # winit would prefer a stray Wayland socket over this X11 display, so
      # remove both names rather than trusting the caller to have done it.
      unset WAYLAND_DISPLAY
      unset WAYLAND_SOCKET
      export DISPLAY
      return 0
      ;;
    wayland)
      if [ -z "${WAYLAND_DISPLAY:-}" ] && [ -z "${WAYLAND_SOCKET:-}" ]; then
        echo "reims-vgpu: REIMS_VGPU_WINDOW_SYSTEM=wayland requires WAYLAND_DISPLAY or WAYLAND_SOCKET" >&2
        return 64
      fi
      return 0
      ;;
    *)
      echo "reims-vgpu: invalid REIMS_VGPU_WINDOW_SYSTEM '$mode' (auto | x11 | wayland)" >&2
      return 64
      ;;
  esac
}
