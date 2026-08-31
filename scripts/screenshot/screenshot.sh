#!/usr/bin/env bash
# Cross-platform host-window capture for KDE Plasma 6 Wayland and macOS.
set -euo pipefail

SCRIPT_NAME="screenshot"
WINDOW_TITLE="Reims vGPU"
OUT=""
WINDOW_FILTER=""
ALLOW_BLACK=0
INCLUDE_DECORATIONS=0
CAP_WIDTH=1280
CAP_HEIGHT=720

usage() {
  cat <<'EOF'
usage: scripts/screenshot/screenshot.sh [options] [output-path]

Capture the exact host-owned Reims vGPU window without focusing or moving it.

Options:
  -o, --output PATH   Write the PNG here (default: random PNG under /tmp)
  --1440p             Cap output at 2560x1440 instead of 1280x720
  --window ID         Capture this exact capture-app window ID
  --allow-black       Accept an almost-entirely-black capture
  --decorations       Include window decorations
  -h, --help          Show this help

The capture is always requested at native rendering resolution before it is
validated and downsampled. REIMS_SHOT_NATIVE=1 keeps that native-resolution PNG.
The final PNG path is printed on stdout; diagnostics go to stderr.
EOF
}

die() {
  printf '%s: %s\n' "$SCRIPT_NAME" "$*" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o | --output | --out-file)
      [[ $# -ge 2 ]] || die "$1 requires a path"
      OUT="$2"
      shift 2
      ;;
    --output=* | --out-file=*)
      OUT="${1#*=}"
      shift
      ;;
    --1440p)
      CAP_WIDTH=2560
      CAP_HEIGHT=1440
      shift
      ;;
    --window)
      [[ $# -ge 2 ]] || die "--window requires an ID"
      WINDOW_FILTER="$2"
      shift 2
      ;;
    --window=*)
      WINDOW_FILTER="${1#--window=}"
      shift
      ;;
    --allow-black)
      ALLOW_BLACK=1
      shift
      ;;
    --decorations | --decoration)
      INCLUDE_DECORATIONS=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --*)
      usage >&2
      die "unknown argument: $1"
      ;;
    *)
      [[ -z "$OUT" ]] || die "only one output path may be specified"
      OUT="$1"
      shift
      ;;
  esac
done

command -v capture-app >/dev/null 2>&1 || die \
  "capture-app is required; install it from https://github.com/steelbrain/capture-app"
command -v python3 >/dev/null 2>&1 || die "python3 is required to read capture-app's window list"

MAGICK=""
for candidate in magick convert; do
  if command -v "$candidate" >/dev/null 2>&1; then
    MAGICK="$candidate"
    break
  fi
done
[[ -n "$MAGICK" ]] || die "ImageMagick (magick or convert) is required for validation and downsampling"

WINDOWS_JSON="$(capture-app --json list)" || die "capture-app could not list capturable windows"

SELECTOR_ERROR="$(mktemp "${TMPDIR:-/tmp}/reims-screenshot-select.XXXXXX")"
cleanup() {
  unlink "$SELECTOR_ERROR" 2>/dev/null || true
}
trap cleanup EXIT

if ! WINDOW_ID="$(
  printf '%s' "$WINDOWS_JSON" | python3 -c '
import json
import sys

title, wanted_id = sys.argv[1:]
windows = json.load(sys.stdin)
windows = [w for w in windows if w.get("capture_candidate", True)]

selector = ""
if wanted_id:
    selector = f"window ID {wanted_id!r}"
    matches = [w for w in windows if str(w.get("id")) == wanted_id]
else:
    selector = f"title exactly {title!r}"
    matches = [w for w in windows if w.get("title") == title]

if not matches:
    if not wanted_id:
        print(f"found no window titled exactly {title!r}", file=sys.stderr)
    else:
        print(f"found no window matching {selector}", file=sys.stderr)
    sys.exit(1)

if not wanted_id and selector.startswith("title exactly") and len(matches) != 1:
    print(f"found {len(matches)} windows titled exactly {title!r}; pass --window ID", file=sys.stderr)
    sys.exit(1)

match = matches[0]
print(match["id"])
' "$WINDOW_TITLE" "$WINDOW_FILTER" \
    2>"$SELECTOR_ERROR"
)"; then
  while IFS= read -r line; do
    printf '%s: %s\n' "$SCRIPT_NAME" "$line" >&2
  done <"$SELECTOR_ERROR"
  exit 1
fi

CAPTURE_ARGS=(capture "$WINDOW_ID" --native-resolution)
if [[ -n "$OUT" ]]; then
  mkdir -p "$(dirname -- "$OUT")"
  CAPTURE_ARGS+=(-o "$OUT")
else
  CAPTURE_ARGS+=(--out-dir "${TMPDIR:-/tmp}")
fi
CAPTURE_ARGS+=(--json)
[[ "$INCLUDE_DECORATIONS" -eq 1 ]] && CAPTURE_ARGS+=(--decoration)

if ! CAPTURE_JSON="$(capture-app "${CAPTURE_ARGS[@]}")"; then
  die "capture-app failed: ${CAPTURE_JSON:-no diagnostic}"
fi

CAPTURED_PATH="$(
  printf '%s' "$CAPTURE_JSON" | python3 -c '
import json
import sys
result = json.load(sys.stdin)
path = result.get("output")
if not path:
    print(result.get("message") or "capture result did not contain an output path", file=sys.stderr)
    sys.exit(1)
print(path)
'
)" || die "capture-app returned an invalid result"

[[ -s "$CAPTURED_PATH" ]] || die "capture produced an empty or missing file: $CAPTURED_PATH"
MAGIC="$(head -c 8 "$CAPTURED_PATH" | od -An -tx1 | tr -d ' \n')"
[[ "$MAGIC" == "89504e470d0a1a0a" ]] || die "output is not a PNG: $CAPTURED_PATH"

STATS="$("$MAGICK" "$CAPTURED_PATH" -alpha off \
  -format 'max=%[fx:maxima*255] mean=%[fx:mean*255] colors=%k' info: 2>/dev/null)" \
  || die "ImageMagick could not inspect capture: $CAPTURED_PATH"
printf '%s: %s\n' "$SCRIPT_NAME" "$STATS" >&2

MAXV="${STATS#max=}"
MAXV="${MAXV%% *}"
MAXV="${MAXV%%.*}"
MEANV="${STATS#*mean=}"
MEANV="${MEANV%% *}"
BLACK_WHY=""
if [[ "$MAXV" =~ ^[0-9]+$ ]] && [[ "$MAXV" -le 8 ]]; then
  BLACK_WHY="max_rgb=${MAXV}, nothing brighter than near-black anywhere"
elif awk -v mean="$MEANV" 'BEGIN { exit !(mean + 0 < 0.5) }'; then
  BLACK_WHY="mean_rgb=${MEANV} with max_rgb=${MAXV}, almost nothing lit"
fi

if [[ -n "$BLACK_WHY" ]]; then
  if [[ "$ALLOW_BLACK" -eq 1 ]]; then
    printf '%s: WARNING: capture is black (%s); --allow-black set\n' \
      "$SCRIPT_NAME" "$BLACK_WHY" >&2
  else
    die "capture is black (${BLACK_WHY}); treating it as a failed capture, not evidence that the guest rendered black. Wrote: $CAPTURED_PATH"
  fi
fi

DIM_BEFORE="$("$MAGICK" "$CAPTURED_PATH" -format '%wx%h' info: 2>/dev/null)" \
  || die "ImageMagick could not read capture dimensions: $CAPTURED_PATH"
if [[ -n "${REIMS_SHOT_NATIVE:-}" ]]; then
  printf '%s: REIMS_SHOT_NATIVE set; keeping native resolution %s\n' \
    "$SCRIPT_NAME" "$DIM_BEFORE" >&2
else
  "$MAGICK" "$CAPTURED_PATH" -resize "${CAP_WIDTH}x${CAP_HEIGHT}>" "$CAPTURED_PATH" \
    || die "ImageMagick could not downsample capture: $CAPTURED_PATH"
  DIM_AFTER="$("$MAGICK" "$CAPTURED_PATH" -format '%wx%h' info: 2>/dev/null)" \
    || die "ImageMagick could not read downsampled dimensions: $CAPTURED_PATH"
  printf '%s: size=%s -> %s (%sp cap)\n' \
    "$SCRIPT_NAME" "$DIM_BEFORE" "$DIM_AFTER" "$CAP_HEIGHT" >&2
fi

printf '%s\n' "$CAPTURED_PATH"
