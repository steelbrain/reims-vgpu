#!/usr/bin/env bash
# Exercise `maps-visual-gate.sh` against the two populations it separates: a
# flat fill, which is what an unstarted or tileless workload looks like, and a
# canvas carrying geographic layers. Neither case involves type, because the
# gate no longer reads any -- whether the label layer rendered is a question a
# human answers by looking at the probe's captures.
set -euo pipefail

DIR=$(mktemp -d)
trap 'find "$DIR" -type f -delete; rmdir "$DIR"' EXIT
GATE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/maps-visual-gate.sh"

magick -size 1280x720 canvas:'#86cee8' "$DIR/empty.png"
if "$GATE" --before "$DIR/empty.png" --after "$DIR/empty.png" \
    --settled "$DIR/empty.png" >/dev/null 2>&1; then
  echo "self-test: empty canvas was accepted" >&2
  exit 1
fi

# Land, water and road-shaped fills spread across the whole frame, so the
# interior the gate crops carries them wherever it lands.
magick -size 1280x720 canvas:'#f2efe9' \
  -fill '#8bc6e8' -draw 'polygon 0,0 620,0 760,300 560,720 0,720' \
  -fill '#cfe3c0' -draw 'rectangle 700,60 1240,380' \
  -fill '#e8d9a0' -draw 'rectangle 660,420 1240,690' \
  -fill '#ffffff' -stroke '#c8c2b4' -strokewidth 9 \
  -draw 'line 0,240 1280,300' -draw 'line 0,540 1280,470' \
  -draw 'line 340,0 420,720' -draw 'line 900,0 980,720' \
  "$DIR/map.png"
"$GATE" --before "$DIR/map.png" --after "$DIR/map.png" \
  --settled "$DIR/map.png"

echo "self-test: pass"
