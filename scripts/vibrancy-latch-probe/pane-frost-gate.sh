#!/usr/bin/env bash
# pane-frost-gate.sh — did the pane's vibrancy change across the load?
#
# The symptom this whole probe exists for is visual, and up to now the verdict
# was a person looking at two PNGs. That is not a gate: it does not survive a
# session boundary, it cannot be run in a loop, and "the sidebar looks less
# frosted" is not a number anyone can disagree with.
#
# What makes it measurable is that `before` and `after` photograph the *same*
# pane, in the same place, over the same wallpaper. Vibrancy is the only thing
# between them, so a per-region difference is the finding and no model of what
# frost should look like is needed. Two regions are compared:
#
#   --pane    the pane's own area. A pane whose backdrop stopped being frosted
#             shows the wallpaper through it at full saturation, which moves
#             every pixel of that region.
#   --control desktop wallpaper with no window over it, which is static in both
#             shots. It is the noise floor: screenshot scaling, the menu-bar
#             clock and the Dock's own animation all land somewhere, and a pane
#             delta only means something against a control that stayed near zero.
#
# # The threshold, and where it comes from
#
# Three measured pairs on the x86 / Vulkan rig, same pane at its restored
# position, region RMSE in [0,1]:
#
#   healthy vs healthy, *different boots*      0.000000
#   healthy vs healthy, same boot either
#     side of a 260 s drag load                0.000191
#   the confirmed degraded capture
#     (`journal/vib4-evidence`) vs healthy     0.150112
#
# The first is the load-bearing one: this pane composites **byte-identically**
# across boots, so the healthy floor is not "small", it is zero, and the 0.000191
# is the menu-bar clock clipping the region's corner. Against that, the degraded
# reading is three orders of magnitude out.
#
# `FROST_RMSE_MAX` is set at 0.01 — 50x above the observed noise and 15x below
# the one degraded reading — so it is a threshold with headroom on both sides
# rather than a line drawn just past the last good number. Exit 1 when the pane
# region crosses it.
#
# Usage:
#   pane-frost-gate.sh --before A.png --after B.png \
#     [--pane WxH+X+Y] [--control WxH+X+Y] [--max RMSE]
#
# Geometry is in screenshot pixels. The screenshot helper caps at 720p, so a
# 1920x1080 guest lands at 1280x719 and guest coordinates scale by 2/3.
set -euo pipefail
export LC_ALL=C

BEFORE=""
AFTER=""
# Defaults describe the pane the probe opens at its restored position, and the
# left third of the desktop, which no phase puts a window over.
PANE="360x300+200+90"
CONTROL="180x300+10+90"
FROST_RMSE_MAX="${FROST_RMSE_MAX:-0.01}"

while [ $# -gt 0 ]; do
  case "$1" in
    --before) BEFORE="$2"; shift 2 ;;
    --after) AFTER="$2"; shift 2 ;;
    --pane) PANE="$2"; shift 2 ;;
    --control) CONTROL="$2"; shift 2 ;;
    --max) FROST_RMSE_MAX="$2"; shift 2 ;;
    -h|--help) sed -n '2,48p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "pane-frost-gate: unknown argument $1" >&2; exit 2 ;;
  esac
done

[ -s "$BEFORE" ] && [ -s "$AFTER" ] || {
  echo "pane-frost-gate: need --before and --after PNGs that exist" >&2; exit 2; }
command -v magick >/dev/null || {
  echo "pane-frost-gate: ImageMagick (magick) is required" >&2; exit 2; }

# `compare -metric RMSE` writes "absolute (normalised)" to stderr and exits
# nonzero whenever the images differ at all, which is the normal case here.
rmse() {
  local geom="$1" a b out
  a=$(mktemp -t frost-a-XXXXXX.png); b=$(mktemp -t frost-b-XXXXXX.png)
  magick "$BEFORE" -crop "$geom" +repage "$a"
  magick "$AFTER" -crop "$geom" +repage "$b"
  out=$(magick compare -metric RMSE "$a" "$b" null: 2>&1 || true)
  rm -f "$a" "$b"
  # Keep the normalised figure, which is comparable across region sizes.
  echo "$out" | sed -n 's/.*(\([0-9.e-]*\)).*/\1/p'
}

PANE_RMSE=$(rmse "$PANE")
CTRL_RMSE=$(rmse "$CONTROL")

echo "pane-frost-gate: pane=$PANE rmse=${PANE_RMSE:-0} max=$FROST_RMSE_MAX"
echo "pane-frost-gate: control=$CONTROL rmse=${CTRL_RMSE:-0}"

# The control is a sanity check on the pair, not part of the verdict: it must be
# near zero for the pane number to mean anything, because a control that moved
# says the two shots are not of the same scene and the pane delta could be the
# wallpaper, the resolution or the display mode rather than the backdrop.
if awk -v c="${CTRL_RMSE:-0}" -v m="$FROST_RMSE_MAX" 'BEGIN { exit !(c > m) }'; then
  echo "pane-frost-gate: the control region moved too — these two shots are not \
of the same scene, so the pane reading is not about vibrancy" >&2
  exit 2
fi

if awk -v p="${PANE_RMSE:-0}" -v m="$FROST_RMSE_MAX" 'BEGIN { exit !(p > m) }'; then
  echo "pane-frost-gate: DEGRADED — the pane did not composite the same way \
twice, and the desktop behind it did"
  exit 1
fi
echo "pane-frost-gate: unchanged — the pane composited the same way either side \
of the load"
