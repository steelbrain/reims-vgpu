#!/usr/bin/env bash
# audio-crackle-probe.sh — does the guest's audio reach the host unbroken?
#
# "Crackling" is not a thing a screenshot can show and not a thing a counter in
# this device reports, because the audio path does not run through this device at
# all: the guest drives QEMU's class-compliant `usb-audio` over the emulated
# xHCI, and QEMU's audio backend hands the result to the host. What this probe
# does is make the symptom a number.
#
# The guest plays a continuous 440 Hz sine. A sine has no silence in it, so any
# silence in the host capture is a gap the path introduced — an underrun that
# emitted zeroes, which is exactly what a click sounds like. The verdict is the
# count of those gaps and the fraction of the run they cover.
#
# It is deliberately blind to *why*. Run it against two `-audiodev` choices and
# the difference between the two numbers is the finding; a single run only says
# whether this host, this backend and this build drop audio at all.
#
# Usage:
#   scripts/audio-crackle-probe/audio-crackle-probe.sh [--seconds N] [--keep DIR]
#
# Requires: a `--testing` boot already up (ssh alias `macos-vm`), PipeWire or
# PulseAudio on the host (`parec`), and `ffmpeg` for tone generation and
# analysis.
#
# Exits 0 when no gap was found, 1 when gaps were found, 2 on a setup failure —
# including a capture that contains no tone at all, because a silent capture
# would otherwise report as one enormous dropout and read as the worst possible
# result when it actually means nothing played.
set -uo pipefail
export LC_ALL=C

SECONDS_TO_PLAY=20
KEEP=""
GUEST="${GUEST:-macos-vm}"
# A gap shorter than one xHCI frame is not something a host can be asked to
# avoid; one frame is 1 ms and the shortest audible click is around 2 ms of
# zeroes, so that is the floor the detector runs at.
GAP_SECONDS=0.002
# A sine at full scale sits far above this. The threshold is for telling
# "zeroes" from "quiet", not for judging level.
GAP_DBFS=-50

while [ $# -gt 0 ]; do
  case "$1" in
    --seconds) SECONDS_TO_PLAY="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "audio-crackle-probe: unknown argument $1" >&2; exit 2 ;;
  esac
done

for tool in ffmpeg parec ssh; do
  command -v "$tool" >/dev/null || { echo "audio-crackle-probe: $tool not found" >&2; exit 2; }
done

WORK="${KEEP:-$(mktemp -d)}"
mkdir -p "$WORK"
TONE="$WORK/tone.wav"
CAPTURE="$WORK/capture.wav"

# Long enough to cover the three seconds before the capture starts, the capture
# itself, and the ssh round trip that gets `afplay` running — so the tone never
# runs out underneath the capture and an end-of-file silence is never counted as
# a dropout.
ffmpeg -hide_banner -loglevel error -f lavfi \
  -i "sine=frequency=440:sample_rate=44100:duration=$((SECONDS_TO_PLAY + 12))" \
  -c:a pcm_s16le "$TONE" -y || { echo "audio-crackle-probe: tone generation failed" >&2; exit 2; }

timeout 30 ssh -o BatchMode=yes "$GUEST" true \
  || { echo "audio-crackle-probe: guest not reachable over ssh" >&2; exit 2; }
timeout 60 scp -o BatchMode=yes "$TONE" "$GUEST:/tmp/reims-tone.wav" >/dev/null \
  || { echo "audio-crackle-probe: could not copy the tone into the guest" >&2; exit 2; }

# The monitor of whatever the host is playing to. Nothing else may be playing;
# the probe cannot tell this device's stream from a notification sound.
MONITOR="$(LC_ALL=C pactl get-default-sink 2>/dev/null).monitor"
[ "$MONITOR" = ".monitor" ] && { echo "audio-crackle-probe: no default sink" >&2; exit 2; }

# Play first, capture second, and stop the capture while the tone still has
# seconds left to run.
#
# The tone has silence on either side of it, and a capture containing either
# edge reports one enormous "dropout" that is really the probe photographing its
# own start. The first version of this did exactly that and scored a run as
# `gaps=1 total_gap_s=0.354` — one gap, a third of a second, entirely the
# lead-in. Measuring only the steady state is what makes the count mean clicks.
#
# The player is bounded host-side: an unattended harness cannot tell a wedged
# `afplay` from a wedged guest.
timeout $((SECONDS_TO_PLAY + 20)) ssh -o BatchMode=yes "$GUEST" \
  "afplay /tmp/reims-tone.wav" >/dev/null 2>&1 &
PLAY_PID=$!
sleep 3
parec --device="$MONITOR" --format=s16le --rate=44100 --channels=2 \
  --file-format=wav "$CAPTURE" &
CAPTURE_PID=$!
sleep "$SECONDS_TO_PLAY"
kill "$CAPTURE_PID" 2>/dev/null
wait "$CAPTURE_PID" 2>/dev/null
kill "$PLAY_PID" 2>/dev/null
timeout 20 ssh -o BatchMode=yes "$GUEST" 'pkill afplay' >/dev/null 2>&1

[ -s "$CAPTURE" ] || { echo "audio-crackle-probe: capture is empty" >&2; exit 2; }

# Everything below is measured from `ANALYSIS_SKIP` seconds in, because `parec`
# opens the monitor and reaches steady state a few milliseconds after it is
# asked to. Both backends measured here reported exactly `gaps=1
# total_gap_s=0.005` starting at t=0 — the same 5.26 ms, to the sample, on two
# unrelated audio paths. That is the recorder's own first buffer, not a
# dropout, and a detector that counts it reports one click on every run and
# makes a clean host indistinguishable from a slightly bad one.
ANALYSIS_SKIP=0.5

# Mean volume over the analysed window. A capture with no tone in it is a setup
# failure, not a perfect score and not a total dropout.
MEAN_DB="$(ffmpeg -hide_banner -nostats -ss "$ANALYSIS_SKIP" -i "$CAPTURE" \
  -af volumedetect -f null - 2>&1 \
  | sed -n 's/.*mean_volume: \(-\?[0-9.]*\) dB.*/\1/p' | tail -1)"
MEAN_DB="${MEAN_DB:--100}"
if awk -v m="$MEAN_DB" 'BEGIN { exit !(m < -60) }'; then
  echo "audio-crackle-probe: capture mean_volume=${MEAN_DB} dBFS — nothing played, no verdict" >&2
  echo "  (check the guest's output device and volume, and that nothing else holds the sink)" >&2
  exit 2
fi

DETECT="$(ffmpeg -hide_banner -nostats -ss "$ANALYSIS_SKIP" -i "$CAPTURE" \
  -af "silencedetect=noise=${GAP_DBFS}dB:d=${GAP_SECONDS}" -f null - 2>&1)"
GAPS="$(printf '%s\n' "$DETECT" | grep -c 'silence_start')"
GAP_SECONDS_TOTAL="$(printf '%s\n' "$DETECT" \
  | sed -n 's/.*silence_duration: \([0-9.]*\).*/\1/p' \
  | awk '{ s += $1 } END { printf "%.3f", s + 0 }')"

echo "audio-crackle-probe: played ${SECONDS_TO_PLAY}s, capture mean_volume=${MEAN_DB} dBFS"
echo "audio-crackle-probe: gaps=${GAPS} total_gap_s=${GAP_SECONDS_TOTAL} (>= ${GAP_SECONDS}s below ${GAP_DBFS} dBFS)"
[ -n "$KEEP" ] && echo "audio-crackle-probe: capture kept at $CAPTURE"
[ "$GAPS" -eq 0 ] && { echo "audio-crackle-probe: PASS"; exit 0; }
echo "audio-crackle-probe: FAIL — the tone reached the host with holes in it"
exit 1
