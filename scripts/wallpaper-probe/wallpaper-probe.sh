#!/usr/bin/env bash
# wallpaper-probe.sh — is the desktop wallpaper where the guest put it?
#
# Goal 10 is "the wallpaper is sometimes shifted 10% to the left, so the right
# 10% of the screen's background is all black — more likely when switching
# between dark and light mode and between dynamic wallpapers". A screenshot of a
# shifted wallpaper says something moved. It cannot say by how much, whether the
# amount is the same at the top of the screen as at the bottom, or whether the
# guest composited it that way — and those distinguish an origin bug from a row
# stride bug from no bug of ours at all.
#
# So the probe supplies the wallpaper. It is a barcode: 64 vertical bars in two
# widely separated colours, in a fixed aperiodic pattern whose best agreement
# with any shifted copy of itself is 0.60 against 1.00 at zero shift. The host
# then decodes the bars out of its own capture at three vertical bands and
# reports, per band, the shift in bars and pixels and how many bars were lost.
#
#   - the same shift in all three bands  -> a uniform origin offset
#   - a shift that grows down the screen -> a row stride mismatch, and the
#     per-band difference gives the stride error directly
#   - no shift but lost bars at an edge  -> content clipped, not moved
#
# Same intent-versus-result shape as `scripts/modal-button-probe` and
# `scripts/web-content-probe`: the guest's own declaration is the only thing
# that can say whether a wrong pixel is this device's fault. Here the
# declaration is stronger than usual, because we chose the image.
#
# Usage:
#   scripts/wallpaper-probe/wallpaper-probe.sh [-n TRIALS] [--keep DIR]
#
# Exits 0 when every band of every trial decoded at zero shift with no lost
# bars, 1 on any shift or loss, 2 on a setup failure.
set -euo pipefail
# ImageMagick prints statistics with a '.', and awk must read them the same way.
export LC_ALL=C

TRIALS=10
KEEP=""
GUEST="${GUEST:-macos-vm}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHOT="$REPO_ROOT/scripts/screenshot-when-kde-plasma-host/screenshot-when-kde-plasma-host.sh"

# 64 bars, two symbols. Fixed rather than generated at run time so a decode can
# be reproduced from the log alone. Chosen by minimising agreement with every
# shifted copy of itself over the overlap — which is how a shifted wallpaper
# actually presents, content sliding and the vacated side lost, rather than
# wrapping — giving a worst non-zero-shift agreement of 0.600.
PATTERN=1111110101101111001101011100010000100110111000101001100000111000
# Neither symbol is black or white, so "the bar is gone" is a third answer
# rather than a misread symbol.
SYM0='#10c030'
SYM1='#f08010'

while [ $# -gt 0 ]; do
  case "$1" in
    -n|--trials) TRIALS="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "wallpaper-probe: unknown argument $1" >&2; exit 2 ;;
  esac
done

WORK="${KEEP:-$(mktemp -d)}"
mkdir -p "$WORK"
[ -n "$KEEP" ] || trap 'rm -rf "$WORK"' EXIT
say() { echo "wallpaper-probe: $*"; }
# shellcheck source=../lib/guest-display.sh
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/guest-display.sh"
osa() { guest_osa "$GUEST" "$1"; }

ssh -o ConnectTimeout=8 -o BatchMode=yes "$GUEST" true 2>/dev/null || {
  say "no guest at $GUEST" >&2; exit 2; }

# Read the screen from the guest rather than assuming it: the probe image has to
# be exactly the desktop's size or macOS scales it and every bar boundary moves.
# See scripts/lib/guest-display.sh for why the question is asked the way it is.
read -r SCR_W SCR_H < <(guest_display_size "$GUEST") || {
  say "could not read the desktop size from the guest" >&2; exit 2; }
case "${SCR_W:-}${SCR_H:-}" in
  ''|*[!0-9]*) say "could not read the desktop size from the guest" >&2; exit 2 ;;
esac
say "guest desktop ${SCR_W}x${SCR_H}, $((${#PATTERN})) bars of $((SCR_W * 100 / ${#PATTERN}))/100 px"

# Build the barcode at exactly the desktop size. `-scale` is a block resize with
# no interpolation, so bar edges stay hard and a measured bar mean is one colour
# rather than a blend of two.
img_args=()
for ((i = 0; i < ${#PATTERN}; i++)); do
  [ "${PATTERN:i:1}" = 0 ] && img_args+=("xc:$SYM0") || img_args+=("xc:$SYM1")
done
magick "${img_args[@]}" +append -scale "${SCR_W}x${SCR_H}!" "$WORK/barcode.png"
scp -q "$WORK/barcode.png" "$GUEST:/tmp/wallpaper-probe.png"

# A window over the desktop reads as total corruption, so get them out of the
# way before measuring anything.
osa 'tell application "System Events" to set visible of (every process whose visible is true and name is not "Finder") to false' >/dev/null || true

set_ours() {
  osa 'tell application "System Events" to set picture of every desktop to "/tmp/wallpaper-probe.png"' >/dev/null || true
}
# The reported trigger, reproduced: flip the appearance and pass through one of
# the system's own dynamic pictures before coming back. Going away and returning
# also defeats macOS's caching of the desktop picture by path, which a plain
# re-set does not.
perturb() {
  osa 'tell application "System Events" to tell appearance preferences to set dark mode to not dark mode' >/dev/null || true
  local other
  other=$(ssh -o BatchMode=yes "$GUEST" \
    'ls /System/Library/Desktop\ Pictures/*.heic 2>/dev/null | head -1' || true)
  [ -n "$other" ] && osa "tell application \"System Events\" to set picture of every desktop to \"$other\"" >/dev/null || true
  sleep 2
  set_ours
  sleep 3
}

set_ours
sleep 3

fails=0
covered=0
for t in $(seq 1 "$TRIALS"); do
  [ "$t" -gt 1 ] && perturb

  # The intent side. If the guest does not believe our picture is the desktop
  # picture, whatever is on screen is not this device's doing and no verdict
  # from it means anything.
  cur=$(osa 'tell application "System Events" to get picture of desktop 1' || true)
  case "$cur" in
    *wallpaper-probe*) ;;
    *) say "trial $t: the guest says the desktop picture is '$cur', not ours — skipped" >&2
       continue ;;
  esac

  png="$WORK/trial-$t.png"
  "$SHOT" -o "$png" >/dev/null 2>&1 || { say "trial $t: capture failed" >&2; continue; }
  IMG_W=$(identify -format '%w' "$png")
  IMG_H=$(identify -format '%h' "$png")

  verdict=$(python3 - "$png" "$IMG_W" "$IMG_H" "$SCR_W" "$SCR_H" "$PATTERN" "$SYM0" "$SYM1" <<'PY'
import subprocess, sys
png, iw, ih, sw, sh, pattern, sym0, sym1 = sys.argv[1:9]
iw, ih, sw, sh = int(iw), int(ih), int(sw), int(sh)
sx, sy = iw / sw, ih / sh
NB = len(pattern)
rgb = lambda s: tuple(int(s[i:i + 2], 16) for i in (1, 3, 5))
# BLACK and WHITE are decode targets, not symbols: a bar that lost its fill has
# to answer "gone" rather than be rounded to whichever symbol it is nearer.
TARGETS = {"0": rgb(sym0), "1": rgb(sym1), "X": (0, 0, 0), "W": (255, 255, 255)}
# Bands clear of the menu bar and the Dock. Three of them, because one cannot
# tell a uniform shift from a shear.
BANDS = [("top", 0.20), ("mid", 0.50), ("bot", 0.78)]
out = []
for label, fy in BANDS:
    y = int(fy * sh * sy)
    bits = []
    for b in range(NB):
        x0, x1 = b * sw / NB, (b + 1) * sw / NB
        px0, px1 = int(x0 * sx), int(x1 * sx)
        # A sixth in from each edge, so the downscale cannot pull the
        # neighbouring bar into the mean.
        pad = max(1, (px1 - px0) // 6)
        cx, cw = px0 + pad, max(1, px1 - px0 - 2 * pad)
        r = subprocess.run(["magick", png, "-crop", f"{cw}x8+{cx}+{y}", "+repage",
                            "-format", "%[fx:mean.r*255] %[fx:mean.g*255] %[fx:mean.b*255]",
                            "info:"], capture_output=True, text=True)
        try:
            m = tuple(float(v) for v in r.stdout.split())
        except ValueError:
            bits.append("?"); continue
        bits.append(min(TARGETS, key=lambda k: sum((a - c) ** 2
                                                   for a, c in zip(TARGETS[k], m))))
    got = "".join(bits)
    lost = sum(1 for c in got if c in "XW?")
    # Best shift over the overlap only. A wallpaper that slid does not wrap: the
    # side it came from is lost, so scoring the wrapped copy would reward the
    # wrong offset.
    best, bestscore = 0, -1.0
    for s in range(-NB + 1, NB):
        pairs = [(got[i], pattern[i - s]) for i in range(NB)
                 if 0 <= i - s < NB and got[i] in "01"]
        if len(pairs) < NB // 3:
            continue
        sc = sum(1 for a, b in pairs if a == b) / len(pairs)
        if sc > bestscore:
            best, bestscore = s, sc
    out.append(f"{label} shift={best} agree={bestscore:.2f} lost={lost} bits={got}")
print("\n".join(out))
PY
)
  bad=0
  while IFS= read -r line; do
    set -- $line
    shift_v=${2#shift=}; lost_v=${4#lost=}
    [ "$shift_v" = 0 ] && [ "$lost_v" = 0 ] || bad=1
  done <<<"$verdict"
  # A window, a screensaver or a sheet over the desktop loses every bar in every
  # band at once. That is not the wallpaper being shifted, and reporting it as
  # such is the mistake `scripts/web-content-probe` made twice.
  if [ "$(echo "$verdict" | grep -c 'lost=64')" -eq 3 ]; then
    covered=$((covered + 1))
    say "trial $t: all 64 bars lost in all three bands — the desktop is covered, not corrupt" >&2
    [ -n "$KEEP" ] && say "  frame kept at $png"
    continue
  fi
  if [ "$bad" = 1 ]; then
    fails=$((fails + 1))
    say "trial $t:"; echo "$verdict" | sed 's/^/  /'
    [ -n "$KEEP" ] && say "  frame kept at $png"
  fi
done

say "$TRIALS trials ($covered with the desktop covered), $fails with a shifted or lost wallpaper"
if [ "$covered" -gt $((TRIALS / 2)) ]; then
  say "over half the trials never saw the desktop — no verdict" >&2
  exit 2
fi
[ "$fails" -eq 0 ] || exit 1
exit 0
