#!/usr/bin/env bash
# web-content-probe.sh — does the guest's web content reach the screen intact?
#
# Goal 8 is "web content is occasionally corrupted in Firefox and Safari —
# background disappearing, subtle bugs, not all-white". A screenshot of a real
# page cannot be checked against anything, because nothing declares what it
# should have looked like. So this serves a page that does declare it
# (`content-probe.html`: a fixed palette of widely separated opaque colours,
# posting back each region's screen rectangle and expected colour), then samples
# the host capture at exactly those rectangles, repeatedly.
#
# Same intent-versus-result shape as `scripts/modal-button-probe`, and for the
# same reason: the guest's own declaration is the only thing that can say
# whether a wrong pixel is this device's fault.
#
# Usage:
#   scripts/web-content-probe/web-content-probe.sh [-n CAPTURES]
#     [--browser safari|chrome|firefox] [--churn 0|1] [--keep DIR]
#
# Exits 0 when every declared region measured its declared colour in every
# capture, 1 on any mismatch, 2 on a setup failure — which includes the page's
# repaint beat not advancing, because a static page reports clean and means
# nothing.
set -euo pipefail
# ImageMagick prints statistics with a '.', and awk must read them the same way.
export LC_ALL=C

CAPTURES=20
BROWSER=safari
CHURN=1
KEEP=""
# Where the probe page is served from.
#
# `guest` runs `probe_server.py` inside the guest over ssh, which is how this
# probe has always worked and is kept as the default so rails it already
# measures are not changed underneath them.
#
# `host` serves the same file from this machine and points the browser at the
# slirp gateway. It exists because the guest arm has a dependency the guest need
# not satisfy: macOS ships `/usr/bin/python3` as a **stub** that answers
# `command -v` but refuses to run until the Command Line Tools are installed,
# and refuses by opening a GUI dialog. On a rail image without them the server
# never starts, the page is never served, and the only symptom is "the page
# never declared a layout" -- which reads as a rendering failure and is a
# missing interpreter. Two boots of rail macos-15 were spent on that message.
#
# Serving from the host removes the guest dependency entirely and puts the
# layout JSON on this side, so a postmortem no longer needs an ssh into a guest
# that is about to be reverted.
SERVE=guest
# The slirp gateway: what 127.0.0.1 on the host looks like from inside a guest
# on QEMU user networking. Not configurable, because it is a property of the
# network backend rather than a preference.
HOST_GATEWAY=10.0.2.2
GUEST="${GUEST:-macos-vm}"
PORT="${PROBE_PORT:-8997}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SHOT="$REPO_ROOT/scripts/screenshot/screenshot.sh"
SERVER="$REPO_ROOT/scripts/browser-probe/probe_server.py"

while [ $# -gt 0 ]; do
  case "$1" in
    -n|--captures) CAPTURES="$2"; shift 2 ;;
    --browser) BROWSER="$2"; shift 2 ;;
    --churn) CHURN="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    --serve) SERVE="$2"; shift 2 ;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "web-content-probe: unknown argument $1" >&2; exit 2 ;;
  esac
done

case "$BROWSER" in
  safari)  APP="Safari" ;;
  chrome)  APP="Google Chrome" ;;
  firefox) APP="Firefox" ;;
  *) echo "web-content-probe: unknown browser $BROWSER" >&2; exit 2 ;;
esac
case "$CHURN" in
  0|1) ;;
  *) echo "web-content-probe: --churn takes 0 or 1" >&2; exit 2 ;;
esac
case "$SERVE" in
  guest) URL="http://127.0.0.1:$PORT/?churn=$CHURN" ;;
  host)  URL="http://$HOST_GATEWAY:$PORT/?churn=$CHURN" ;;
  *) echo "web-content-probe: --serve takes guest or host" >&2; exit 2 ;;
esac

WORK="${KEEP:-$(mktemp -d)}"
mkdir -p "$WORK"
[ -n "$KEEP" ] || trap 'rm -rf "$WORK"' EXIT
say() { echo "web-content-probe: $*"; }

ssh -o ConnectTimeout=8 -o BatchMode=yes "$GUEST" true 2>/dev/null || {
  say "no guest at $GUEST" >&2; exit 2; }

# Firefox asks to be made the default browser on first launch, and macOS draws
# that question as a sheet, which dims the window behind it — every declared
# colour then measures at half and the run reports the page as corrupt. Turning
# the question off is setup, not a workaround for a device bug.
[ "$BROWSER" = firefox ] && ssh -o BatchMode=yes "$GUEST" '
for p in "$HOME/Library/Application Support/Firefox/Profiles/"*/; do
  [ -d "$p" ] || continue
  printf %s\\n "user_pref(\"browser.shell.checkDefaultBrowser\", false);" >"$p/user.js"
done' >/dev/null 2>&1 || true

HOST_JSON="$WORK/content-probe.json"
HOST_SERVER_LOG="$WORK/content-probe-server.log"
if [ "$SERVE" = host ]; then
  # Bound to the whole work directory's lifetime, and killed on exit whichever
  # way this script leaves -- a server surviving a failed run would be picked up
  # by the next one and report a layout for a page nobody loaded.
  pkill -f "probe_server.py $PORT " >/dev/null 2>&1 || true
  : > "$HOST_JSON"
  nohup python3 "$SERVER" "$PORT" "$SCRIPT_DIR/content-probe.html" "$HOST_JSON" \
    > "$HOST_SERVER_LOG" 2>&1 &
  HOST_SERVER_PID=$!
  trap 'kill "$HOST_SERVER_PID" 2>/dev/null || true; [ -n "$KEEP" ] || rm -rf "$WORK"' EXIT
  sleep 2
  kill -0 "$HOST_SERVER_PID" 2>/dev/null || {
    say "the host probe server exited immediately; see $HOST_SERVER_LOG" >&2
    exit 2; }
else
  scp -q "$SERVER" "$SCRIPT_DIR/content-probe.html" "$GUEST:/tmp/"
fi
ssh -o BatchMode=yes "$GUEST" "pkill -f probe_server.py >/dev/null 2>&1 || true
pkill -f '$APP' >/dev/null 2>&1 || true
sleep 2
if [ '$SERVE' = guest ]; then
  nohup python3 /tmp/probe_server.py $PORT /tmp/content-probe.html /tmp/content-probe.json \
    >/tmp/content-probe-server.log 2>&1 &
  sleep 2
fi
open -a '$APP' '$URL'
sleep 6
# Dismiss whatever sheet the browser opened on us before reaching for the
# keyboard, because a sheet swallows the fullscreen chord too — the Firefox run
# that found this was still windowed for all twenty captures.
osascript -e 'tell application \"System Events\" to key code 53' >/dev/null 2>&1 || true
sleep 1
osascript -e 'tell application \"System Events\" to key code 53' >/dev/null 2>&1 || true
sleep 1
# Fullscreen so the viewport is the display and the page's screen-space
# rectangles need no chrome model. Done after load: entering fullscreen fires a
# resize, and the page re-declares its layout on resize.
osascript -e 'tell application \"System Events\" to key code 3 using {command down, control down}' >/dev/null 2>&1 || true
sleep 5" >/dev/null

LAYOUT="$WORK/layout.json"

# The page republishes its layout every second, and the newest record is the one
# that describes the window the next capture will contain. Re-read per capture
# rather than once: a window that is moved, resized or fullscreened mid-run would
# otherwise be measured against rectangles for a window that no longer exists,
# and every region would report a mismatch that is the probe's fault.
refresh_layout() {
  if [ "$SERVE" = host ]; then
    cp "$HOST_JSON" "$LAYOUT" 2>/dev/null || true
  else
    ssh -o BatchMode=yes "$GUEST" "cat /tmp/content-probe.json 2>/dev/null" >"$LAYOUT" || true
  fi
  grep -q '"kind":' "$LAYOUT" || return 1
  python3 - "$LAYOUT" "$WORK/regions.txt" <<'PY'
import json, sys
last = None
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    try:
        d = json.loads(line)
    except ValueError:
        continue
    if d.get("kind") == "layout":
        last = d
if last is None:
    sys.exit("no layout record")
with open(sys.argv[2], "w") as f:
    f.write(f"SCREEN {last['screen']['w']} {last['screen']['h']}\n")
    # The page's beat counter, carried so the host can tell a stalled page from
    # a dropped frame. Absent from records written by an older copy of the page.
    f.write(f"BEAT {last.get('beat', -1)}\n")
    for r in last["regions"]:
        e = r["expect"]
        f.write(f"R {r['name']} {r['x']} {r['y']} {r['w']} {r['h']} {e[0]} {e[1]} {e[2]}\n")
PY
}

beat_now() { awk '/^BEAT /{print $2; exit}' "$WORK/regions.txt"; }

refresh_layout || {
  if [ "$SERVE" = host ]; then
    say "the page never declared a layout — see $HOST_SERVER_LOG on the host" >&2
  else
    say "the page never declared a layout — see /tmp/content-probe-server.log in the guest (note: macOS ships python3 as a stub that needs the Command Line Tools; try --serve host)" >&2
  fi
  ssh -o BatchMode=yes "$GUEST" "pkill -f probe_server.py; pkill -f '$APP'" >/dev/null 2>&1 || true
  exit 2; }

read -r _ SCR_W SCR_H < <(grep -m1 '^SCREEN ' "$WORK/regions.txt")
say "guest screen ${SCR_W}x${SCR_H}, $(grep -c '^R ' "$WORK/regions.txt") declared regions"

# The page repaints on a beat, and everything this probe is looking for happens
# because of that repaint. A run whose beat never ran is a run of a static page,
# and it reports itself clean — that is exactly what the first churning run did:
# six captures, zero mismatches, and retained frames with no churn and no beat
# counter in them.
#
# So establish that the beat is advancing before believing any verdict, and call
# a stalled page a setup failure rather than a result. `CHURN_WITNESS` fails the
# same way a dropped patch does, and without this gate a wedged page would be
# indistinguishable from this device losing a layer.
b0=$(beat_now); sleep 3; refresh_layout || true; b1=$(beat_now)
if [ "$b0" = "-1" ]; then
  say "the page did not report a beat — guest is running an older content-probe.html" >&2
  ssh -o BatchMode=yes "$GUEST" "pkill -f probe_server.py; pkill -f '$APP'" >/dev/null 2>&1 || true
  exit 2
fi
if [ "$b1" -le "$b0" ]; then
  say "the page's beat is stalled at $b0 after 3s — nothing is repainting, so no verdict is meaningful" >&2
  ssh -o BatchMode=yes "$GUEST" "pkill -f probe_server.py; pkill -f '$APP'" >/dev/null 2>&1 || true
  exit 2
fi
if [ "$CHURN" = 1 ] && ! grep -q '^R CHURN_WITNESS ' "$WORK/regions.txt"; then
  say "churn is on but the page declared no CHURN_WITNESS — the churn container never built" >&2
  ssh -o BatchMode=yes "$GUEST" "pkill -f probe_server.py; pkill -f '$APP'" >/dev/null 2>&1 || true
  exit 2
fi
say "beat advancing ($b0 -> $b1), churn=$CHURN"

fails=0
stalls=0
dimmed=0
prev_beat=$b1
for i in $(seq 1 "$CAPTURES"); do
  refresh_layout || { say "capture $i: no fresh layout" >&2; continue; }
  # A capture taken while the page is not repainting says nothing about this
  # device, so it is neither a pass nor a failure.
  this_beat=$(beat_now)
  if [ "$this_beat" -le "$prev_beat" ]; then
    stalls=$((stalls + 1)); prev_beat=$this_beat
    say "capture $i: page beat stalled at $this_beat, not counted"
    sleep 1; continue
  fi
  prev_beat=$this_beat
  png="$WORK/cap-$i.png"
  "$SHOT" -o "$png" >/dev/null 2>&1 || { say "capture $i failed" >&2; continue; }
  IMG_W=$(identify -format '%w' "$png")
  IMG_H=$(identify -format '%h' "$png")

  bad=$(python3 - "$WORK/regions.txt" "$png" "$IMG_W" "$IMG_H" "$SCR_W" "$SCR_H" <<'PY'
import subprocess, sys
regions_path, png, iw, ih, sw, sh = sys.argv[1:7]
sx, sy = int(iw) / int(sw), int(ih) / int(sh)
# The palette every measured mean is classified against. Nearest-entry rather
# than a tolerance: the colours are far apart by construction, so "which of
# these is it" is answerable without naming a distance that counts as close,
# and a region that lost its fill reports as WHITE or BLACK by name instead of
# as an unexplained miss.
PALETTE = {
    "BG": (0x20, 0x20, 0x80), "RED": (0xe0, 0x10, 0x10), "GREEN": (0x10, 0xc0, 0x30),
    "YELLOW": (0xf0, 0xe0, 0x10), "MAGENTA": (0xd0, 0x10, 0xd0), "CYAN": (0x10, 0xd0, 0xe0),
    "ORANGE": (0xf0, 0x80, 0x10), "VIOLET": (0x70, 0x10, 0xe0),
    "WHITE": (0xff, 0xff, 0xff), "BLACK": (0x00, 0x00, 0x00),
}
specs = []
for line in open(regions_path):
    p = line.split()
    if p and p[0] == "R":
        specs.append((p[1], *(int(v) for v in p[2:9])))
# Inset each rectangle before measuring so a one-pixel rounding error in the
# downscale cannot pull a neighbouring colour into the mean.
args = []
for name, x, y, w, h, *_ in specs:
    px, py = int(x * sx), int(y * sy)
    pw, ph = max(1, int(w * sx)), max(1, int(h * sy))
    ix, iy = px + max(1, pw // 6), py + max(1, ph // 6)
    iw2, ih2 = max(1, pw - 2 * max(1, pw // 6)), max(1, ph - 2 * max(1, ph // 6))
    args.append((name, ix, iy, iw2, ih2))
bad = []
measured = []
for (name, x, y, w, h), spec in zip(args, specs):
    r = subprocess.run(["magick", png, "-crop", f"{w}x{h}+{x}+{y}", "+repage",
                        "-format", "%[fx:mean.r*255] %[fx:mean.g*255] %[fx:mean.b*255]",
                        "info:"], capture_output=True, text=True)
    try:
        mr, mg, mb = (float(v) for v in r.stdout.split())
    except ValueError:
        bad.append(f"{name}=UNREADABLE")
        continue
    # Kept alongside the verdict so the uniform-scale fit below can be taken
    # over every region, including the ones that classified correctly.
    got = min(PALETTE, key=lambda k: sum((a - b) ** 2 for a, b in zip(PALETTE[k], (mr, mg, mb))))
    want = min(PALETTE, key=lambda k: sum((a - b) ** 2 for a, b in zip(PALETTE[k], spec[5:8])))
    measured.append((name, (mr, mg, mb), spec[5:8]))
    if got != want:
        bad.append(f"{name}: declared {want} measured {got} rgb=({mr:.0f},{mg:.0f},{mb:.0f})")

# Is the whole frame a uniform multiple of what was declared?
#
# The README's standing lesson is that a disagreement affecting every region at
# once is the oracle's fault, not the device's, because a real compositing loss
# is local. This is that lesson as code, and it earned its place immediately: a
# Firefox run reported seven regions corrupted in nineteen consecutive captures,
# and the frame showed a "make Firefox your default browser" sheet, which macOS
# draws by dimming the window behind it. Every measured colour was its declared
# colour times 0.5, to the last bit — RED 224->111, YELLOW 240->120, the near
# black 16->8 alike.
#
# The two regions that did *not* report were the coincidence that makes this
# worth detecting rather than eyeballing: halved BG and halved GREEN are exactly
# equidistant from their own entry and from BLACK, so the tie-break passed them.
# A uniform dim can therefore report as a partial, plausible-looking corruption.
#
# So: fit one scale k across every region by least squares, and if the fit is
# tight the frame is globally attenuated. That is a state of the guest's screen,
# not a loss in this device, and no verdict taken through it means anything.
num = sum(m[i] * s[i] for _, m, s in measured for i in range(3))
den = sum(s[i] * s[i] for _, _, s in measured for i in range(3))
k = num / den if den else 1.0
resid = max((abs(m[i] - k * s[i]) for _, m, s in measured for i in range(3)), default=0.0)
# 8 of 255 is under one part in thirty, well inside the 96-unit palette
# separation, so this cannot swallow a real per-region loss: a region that went
# black or white is nowhere near k times its declaration for the k the rest of
# the frame fits.
if bad and resid <= 8.0 and k < 0.9:
    print(f"ATTENUATED {k:.3f} {resid:.1f}")
else:
    print("; ".join(bad))
PY
)
  case "$bad" in
    ATTENUATED*)
      set -- $bad
      say "capture $i: the whole frame measures ${2} times what the page declared \
(worst residual ${3}/255) — the guest's screen is dimmed, most likely a modal \
sheet over the window. Not a device loss; no verdict from this frame." >&2
      [ -n "$KEEP" ] && say "  frame kept at $png"
      dimmed=$((dimmed + 1))
      sleep 1; continue ;;
  esac
  if [ -n "$bad" ]; then
    fails=$((fails + 1))
    say "capture $i: $bad"
    [ -n "$KEEP" ] && say "  frame kept at $png"
  fi
  sleep 1
done

ssh -o BatchMode=yes "$GUEST" "pkill -f probe_server.py; pkill -f '$APP'" >/dev/null 2>&1 || true
say "$CAPTURES captures ($stalls not counted, page not repainting; \
$dimmed not counted, screen dimmed), \
$fails with a region that did not measure its declared colour"
# Mostly frozen or mostly dimmed means the run measured something other than
# what it was pointed at. Report that as a setup failure: a clean verdict from
# it would be the same lie the beat gate exists to stop.
if [ $((stalls + dimmed)) -gt $((CAPTURES / 2)) ]; then
  say "over half the captures were unusable ($stalls frozen, $dimmed dimmed) — no verdict" >&2
  exit 2
fi
[ "$fails" -eq 0 ] || exit 1
exit 0
