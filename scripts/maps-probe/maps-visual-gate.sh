#!/usr/bin/env bash
# maps-visual-gate.sh — did Maps draw a geographic canvas at all?
#
# This is a validity check on the measured workload, in the same class as the
# probe's `scored_draws` check: it asks whether there is a scene to measure, not
# whether the scene is correct. A run whose map interior is one flat fill drew
# no tiles, so its frame rate is the rate of an unstarted workload and must not
# be reported as this device's.
#
# It quantizes the map interior to 16 colours and refuses a frame in which one
# colour owns 80 % or more of it. Quantizing makes this a large-scale canvas
# test rather than an antialiasing test: in a controlled empty-grid capture the
# fill owns 0.8721 of the interior, and in the widest valid driven capture it
# owns 0.5915, so the boundary sits between two measured populations with room
# on both sides.
#
# **It does not judge whether the frame is right, and nothing here may.** A
# capture is scored by opening it and looking at it — `AGENTS.md`'s
# "Never score a frame by OCR. Look at it." is the rule, and this file used to
# break it: it ran Tesseract over the interior and failed any frame with fewer
# than twelve confident words. That verdict was wrong in both directions on this
# workload. It read road casings and antialiasing as type on a scene carrying no
# labels at all, and it scored a torn frame — hundreds of full-width stripes of
# correct colour — exactly as it scored a clean one. Worse, it is a *blocking*
# gate: with Maps' label layer genuinely absent on this rail it refused every
# frame and no frame-rate window was ever measured, so an open correctness
# defect silently suppressed the performance measurement that was valid beside
# it.
#
# So the label question is answered by a human reading the captures the probe
# keeps, and this gate answers only the question a machine can: is there a map.
set -euo pipefail
export LC_ALL=C

BEFORE=""
AFTER=""
SETTLED=""

while [ $# -gt 0 ]; do
  case "$1" in
    --before) BEFORE="$2"; shift 2 ;;
    --after) AFTER="$2"; shift 2 ;;
    --settled) SETTLED="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "maps-visual-gate: unknown argument $1" >&2; exit 2 ;;
  esac
done

for image in "$BEFORE" "$AFTER" "$SETTLED"; do
  [ -s "$image" ] || { echo "maps-visual-gate: missing captured frame" >&2; exit 2; }
done
command -v magick >/dev/null || {
  echo "maps-visual-gate: ImageMagick is required" >&2; exit 2; }

WORK=$(mktemp -d)
trap 'find "$WORK" -type f -delete; rmdir "$WORK"' EXIT
failed=0
index=0

for image in "$BEFORE" "$AFTER" "$SETTLED"; do
  index=$((index + 1))
  read -r width height <<<"$(magick identify -format '%w %h' "$image")"

  # Keep only the central map viewport, so the sidebar, toolbar, scale bar,
  # compass and zoom buttons cannot supply the variation the fill test looks
  # for.
  x=$((width * 20 / 100))
  y=$((height * 10 / 100))
  crop_width=$((width * 75 / 100))
  crop_height=$((height * 80 / 100))
  crop="$WORK/frame-$index.png"
  magick "$image" -crop "${crop_width}x${crop_height}+${x}+${y}" +repage "$crop"

  pixels=$((crop_width * crop_height))
  dominant=$(magick "$crop" -colors 16 -format %c histogram:info:- |
    awk -F: 'BEGIN { max=0 } { gsub(/ /, "", $1); if ($1 + 0 > max) max=$1 + 0 } END { print max }')
  dominant_fraction=$(awk -v count="$dominant" -v total="$pixels" \
    'BEGIN { printf "%.4f", count / total }')

  printf 'maps-visual-gate: %s dominant=%s\n' \
    "$(basename "$image")" "$dominant_fraction"

  if awk -v fraction="$dominant_fraction" 'BEGIN { exit !(fraction >= 0.80) }'; then
    echo "maps-visual-gate: INVALID — geographic layers do not cover the map interior"
    failed=1
  fi
done

[ "$failed" -eq 0 ] || exit 1
echo "maps-visual-gate: a geographic canvas is present in every frame"
