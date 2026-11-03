#!/bin/bash
# The lane gate's probe batch: other agents' guest probes, run on the lane's published tip (the
# image and host build lane-gate.sh --publish put on the lane host) in one exclusive boot a round,
# so that a probe needs no boot of its own.
#
# To drop a probe, on the lane host, make ~/.cache/rgk-run/probe-queue/NAME/ holding:
#   run.sh   what runs in the guest, as the autologin user, over SSH: `bash run.sh` in the guest's
#            copy of the directory (~/probes/NAME), with PROBE_NAME and PROBE_PASS (the user's
#            password, for sudo) in its environment. What it prints is its result; what it leaves in
#            ./out/ is copied back.
#   budget   the most seconds it may run, 1 to 180: the guest kills its process group there.
#   tip      the commit it is valid for, full or short: a probe for another tip than the published
#            one is skipped, and its results say so.
#   shots    (optional, any content) run.sh may take the guest's own screenshots: `bash
#            ~/probes/.shot.sh LABEL` leaves out/shot-LABEL.png (Command-Shift-3, sent by the host;
#            exit 1 where none came in 20 s), as screencapture over SSH cannot
#   (any other files run.sh reads, in the same directory)
# Write the directory elsewhere and move it in, so a round never takes it half written. Its results
# land in ~/.cache/rgk-run/probe-results/NAME/ (msi-probes.sh lists the files: stdout, stderr, rc,
# wall_s, host.log, gpu-clocks.txt, out/, status), and the probe leaves the queue once it ran or
# was skipped. One not reached in a round's hold stays queued for the next.
#
#   probe-batch.sh [--force] [--list] [--dry-run]
# A round runs where 3 or more probes wait, or the oldest has waited 45 minutes, or with --force
# (after each gate). It takes the probes oldest first while their budgets fit the hold, boots the
# published tip once through msi-boot.sh (an exclusive request: the GPU alone, in the host's
# turnstile, the host quiet; the gate's one request on the host while it runs), waits for the
# host's load to settle under 4, runs them back to back and powers the guest off. --list shows the
# queue and what a round would take. --dry-run runs a round on this machine: the lane host's
# directories under a scratch directory and the guest a directory of it (msi-probes.sh FAKE_GUEST),
# with three example probes, one of them for another tip.
set -euo pipefail
L=$(cd "$(dirname "$0")" && pwd)
GH=${LANE_GATE_HOME:-$HOME/.cache/rgk-lane-gate}
force=0 list=0 dry=0
for a in "$@"; do
  case $a in
    --force) force=1 ;;
    --list) list=1 ;;
    --dry-run) dry=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) echo "probe-batch: unknown argument $a" >&2; exit 2 ;;
  esac
done
. "$GH/config.sh"
PUB=$(cat "$GH/published-last")
SHORT=${PUB:0:9}
WAIT_N=3 WAIT_S=2700
# The host's log filter in a probe boot: info, and the display's presents ("PRESENT_SURFACE of
# transaction", reims_graphics at trace, about 23 lines a second at 120 Hz), so a probe's host.log
# counts composites. reims_graphics has no other trace lines; at debug it logs SUBMIT_ROOM posts
# (reims_graphics::room). Gate and bench boots log at info.
PROBE_HOST_LOG=info,reims_graphics=trace
# The hold the probes get: the boot's own time to the desktop and the device check (about 90 s),
# the longest load settle (180 s) and a margin come first.
FIT_S=$(( HOLD_S - 90 - 180 - 60 ))

if ((dry)); then
  D=$(mktemp -d /tmp/probe-dry.XXXXXX)
  MSI_RGK_RUN=$D/rgk-run MSI_GATE=$D/lane-gate-msi
  mkdir -p "$MSI_RGK_RUN/probe-queue" "$D/guest"
  # Three probes: two for the published tip (one over its budget), one for another commit.
  mk() {
    local n=$1 b=$2 t=$3 body=$4
    mkdir -p "$D/drop/$n"
    printf '%s\n' "$body" > "$D/drop/$n/run.sh"
    echo "$b" > "$D/drop/$n/budget"
    echo "$t" > "$D/drop/$n/tip"
    mv "$D/drop/$n" "$MSI_RGK_RUN/probe-queue/$n"
  }
  mk a-hello 10 "$SHORT" 'echo "hello from $PROBE_NAME"; mkdir -p out; date > out/when; echo to-stderr >&2
bash ~/probes/.shot.sh first && echo "shot first: $(cat out/shot-first.png)"'
  : > "$MSI_RGK_RUN/probe-queue/a-hello/shots"
  mk b-slow 2 "$PUB" 'echo start; sleep 30; echo never'
  mk c-other 10 0123456789ab 'echo should not run'
  msi() { bash -c "$1"; }
  put() { cp -R "$1" "$2"; }
  echo "probe-batch: dry run in $D"
else
  msi() { ssh -o BatchMode=yes "$MSI" "bash -c $(printf '%q' "$1")"; }
  put() { scp -q -r "$1" "$MSI:$2"; }
fi
Q=$MSI_RGK_RUN/probe-queue
RES=$MSI_RGK_RUN/probe-results

# The queue: "AGE_S BUDGET TIP NAME" a probe, oldest first.
queue=$(msi "mkdir -p '$Q' '$RES'; cd '$Q' && now=\$(date +%s) && for d in */; do d=\${d%/}; [ -d \"\$d\" ] || continue
  m=\$(stat -c %Y \"\$d\" 2> /dev/null || stat -f %m \"\$d\"); echo \"\$((now - m)) \$(tr -dc 0-9 < \"\$d/budget\" 2> /dev/null || true) \$(tr -dc 0-9a-f < \"\$d/tip\" 2> /dev/null || true) \$d\"
  done | sort -rn")
n=$(printf '%s' "$queue" | grep -c . || true)
oldest=$(printf '%s\n' "$queue" | awk 'NR == 1 { print $1 + 0 }')
# What a round takes: the probes for the published tip, oldest first, while their budgets fit;
# the others are skipped (their results say why).
take=() fit=0
while read -r age budget tip name; do
  [[ -n ${name:-} ]] || continue
  if [[ -z ${tip:-} || ( $PUB != "$tip"* && $tip != "$PUB"* ) ]]; then take+=("$name"); continue; fi
  (( fit + ${budget:-180} + 15 <= FIT_S )) || continue
  fit=$(( fit + ${budget:-180} + 15 ))
  take+=("$name")
done <<< "$queue"
echo "probe-batch: $n waiting (oldest $(( ${oldest:-0} / 60 )) min); published tip $SHORT; a round takes: ${take[*]:-none} ($fit s of $FIT_S)"
((list)) && { printf '%s\n' "$queue" | sed '/^$/d; s/^/  /'; exit 0; }
if (( !dry && !force )) && (( n < WAIT_N )) && (( ${oldest:-0} < WAIT_S )); then
  echo "probe-batch: no round (a round runs at $WAIT_N waiting or the oldest at $(( WAIT_S / 60 )) min, or with --force)"
  exit 0
fi
((${#take[@]})) || { echo "probe-batch: nothing to run"; exit 0; }

# The round: its probes leave the queue for its directory, and those it does not reach go back.
ROUND=$(date +%Y%m%d-%H%M%S)
RD=$MSI_GATE/probes/$ROUND
msi "mkdir -p '$RD/probes' '$RD/payload' '$RD/bin' && cd '$Q' && for p in ${take[*]}; do mv \"\$p\" '$RD/probes/'; done"
put "$L/msi-probes.sh" "$RD/bin/"
put "$L/msi-xid.sh" "$RD/bin/"
if ((dry)); then
  msi "PROBES='$RD/probes' PROBE_RESULTS='$RES' PROBE_TIP='$PUB' DEADLINE=$(( $(date +%s) + HOLD_S )) FAKE_GUEST='$D/guest' \
    R='$RD' bash '$RD/bin/msi-probes.sh'"
else
  # The gate's scripts the boot runs, and the tip's bench for its check of our Metal device.
  run=$(ls -1dt "$GH/runs/${PUB:0:12}-"* 2> /dev/null | head -1)
  [[ -x $run/artifacts/payload/bench ]] || { echo "probe-batch: no gate run of $SHORT with its bench" >&2; exit 1; }
  put "$run/artifacts/payload/bench" "$RD/payload/"
  put "$L/msi-boot.sh" "$RD/bin/"
  msi "cat > '$RD/boot.env'" << EOF
QEMU=$MSI_RGK_RUN/tip-host/$SHORT/qemu-system-x86_64
ROM=$MSI_RGK_RUN/tip-host/$SHORT/ReimsGraphics.rom
OC=$MSI_RGK_RUN/loader-fix/oc-sipon-$SHORT.qcow2
DISK=$MSI_DISK
TAG=probes-$ROUND
PAYLOAD=$RD/payload
CORPUS=
GUEST_USER=$GUEST_USER
GUEST_PASS=$GUEST_PASS
RGK_RUN=$MSI_RGK_RUN
EXCLUSIVE=1
HOLD_S=$HOLD_S
LOCK_WAIT_S=${LOCK_WAIT_S:-10800}
DRAG=0
HOST_LOG=$PROBE_HOST_LOG
PROBES=$RD/probes
PROBE_RESULTS=$RES
PROBE_TIP=$PUB
EOF
  msi "bash '$RD/bin/msi-boot.sh' '$RD'" || echo "probe-batch: the boot ended $? (its log: $RD/boot.log)"
  # The round's GPU errors (msi-boot.sh's xid.txt), the boot's own QEMU's marked "own".
  msi "cat '$RD/xid.txt' 2> /dev/null" | sed 's/^/  Xid: /'
fi

# Each probe's status; the ones not reached (or never started: a failed boot) go back to the queue.
for p in "${take[@]}"; do
  st=$(msi "cat '$RES/$p/status' 2> /dev/null || echo 'not reached: the boot did not run it'")
  echo "  $p: $st"
  case $st in
    ran|skipped*) msi "rm -rf '$RD/probes/$p'" ;;
    *) msi "mv '$RD/probes/$p' '$Q/' && rm -rf '$RES/$p'" ;;
  esac
done
((dry)) && { echo "probe-batch: the dry run's results:"; for p in "$RES"/*/; do
  echo "== ${p%/}"; for f in status rc wall_s stdout stderr; do [[ -f $p$f ]] && echo "  $f: $(tr '\n' '|' < "$p$f")"; done
  [[ -d ${p}out ]] && echo "  out/: $(ls "${p}out" | tr '\n' ' ')"; done
  echo "  left in the queue: $(ls "$Q" | tr '\n' ' ')"; }
exit 0
