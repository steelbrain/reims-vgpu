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
#   budget   the most seconds it may run, 1 to 300: the guest kills its process group there.
#   tip      the commit it is valid for, full or short (9 characters or more): a round boots one
#            tip, the newest published one or an earlier published tip (its image and host still on
#            the lane host, tip-host/SHORT and loader-fix/oc-sipon-SHORT.qcow2), and a probe for a
#            tip with no published build is skipped, its results saying so.
#   shots    (optional, any content) run.sh may take the guest's own screenshots: `bash
#            ~/probes/.shot.sh LABEL` leaves out/shot-LABEL.png (Command-Shift-3, sent by the host;
#            exit 1 where none came in 20 s), as screencapture over SSH cannot
#   (any other files run.sh reads, in the same directory)
# Write the directory elsewhere and move it in, so a round never takes it half written. Its results
# land in ~/.cache/rgk-run/probe-results/NAME/ (msi-probes.sh lists the files: stdout, stderr, rc,
# wall_s, host.log, gpu-clocks.txt, out/, status), and the probe leaves the queue once it ran or
# was skipped. One not reached in a round's hold stays queued for the next.
#
#   probe-batch.sh [--force] [--list] [--tip SHA] [--capture] [--first NAME,...] [--only NAME,...]
#                  [--exclude NAME,...] [--hold S] [--catch-from-boot] [--host-tip SHA] [--image-tip SHA]
#                  [--cold-cache KEY] [--qemu-hook CMD] [--pre-qemu CMD] [--log-add FILTER]
#                  [--host-log FILTER] [--stage NAME=DIR[@TIP]]... [--ram GIB] [--after CMD]
#                  [--image PATH] [--rom PATH] [--gfx-b64 B64] [--qemu-env K=V]... [--no-metal-check]
#                  [--stage-local NAME=DIR[@TIP]]... [--appload] [--kring-s S]
#
# The round runs as a potluck job on the lane host (lane-gate/msi-round.sh), holding the 5080 and the
# whole host (--exclusive; potluck's lease is the lock), dispatched from a throwaway commit of the
# round's own (lane-gate/ship-commit.sh) that carries the gate's host scripts, the tip's bench and the
# round's boot.env. This script only reads the lane host's state (its queue and published tips, over
# ssh); the job makes every change there. --stage NAME=DIR[@TIP] adds the probe in DIR on the lane host
# to the round as NAME, copied in by the job (its tip file TIP where given), rather than from the
# queue; --stage-local NAME=DIR[@TIP] does the same for a probe in DIR on this machine, which the
# round's commit carries (lg/probes/NAME; a binary of our own built for the round goes in DIR; a
# file DIR/.base naming a probe directory on the lane host lays DIR's files over that one's);
# --kring-s S takes the kext's lines from the guest's kernel ring every S seconds (default 5) in a
# captured round; --appload runs the gate's app load (msi-appload.sh, every app and 7 Safari pages) once the
# desktop is up, before the probes, which then run on that busy desktop (appload.txt in the
# round's directory); --ram GIB is the job's memory (default 64: the guest's 16 GiB and the host's own, and page
# cache a probe's host side fills counts too); --after CMD runs CMD on the lane host after the boot,
# inside the job (a hook's own wait for its post-processing, say). --dry-run does not run as a job.
# A stock control boots other devices than ours: --image PATH (an OpenCore image on the lane host,
# only read: the boot writes an overlay), --rom PATH, --gfx-b64 B64 (base64 of QEMU's graphics
# arguments, a shell fragment in place of reims-graphics-pci), --qemu-env K=V (exported for QEMU,
# {DIR} the round's directory), --no-metal-check (no check that the Metal device is ours).
# A round runs where 3 or more probes wait, or the oldest has waited 45 minutes, or with --force
# (after each gate). Its tip is the newest published one (published-last) where a probe waits for
# it, else the oldest probe's that is still published. It takes the probes for that tip oldest
# first while their budgets fit the hold (those for other published tips wait for their own round;
# --tip SHA picks the round's tip, a published one), boots the tip once through msi-boot.sh (an
# exclusive request: the GPU alone, in the
# host's turnstile, the host quiet; the gate's one request on the host while it runs), waits for
# the host's load to settle under 4, runs them back to back and powers the guest off. --capture
# also captures the boot as the gate's --capture-boot does (msi-boot.sh's CAPTURE: its scanout,
# the guest's screenshots and the kext's log of the boot, in the round's capture/), the host's log
# at info,reims_graphics=trace,reims_vgpu::graphics=debug; a round with --capture runs with no
# probe too. --first NAME,... has the round consider those probes first, in that order, before the
# others oldest first, and --only NAME,... takes those alone, in that order; --exclude NAME,...
# leaves those in the queue (a probe that must come last in its own round); --catch-from-boot
# starts the hang catcher with QEMU (msi-boot.sh's CATCH_FROM_BOOT), for a restart or a silence
# before the desktop, and runs with no probe too; --host-tip and --image-tip boot another
# published tip's host build (QEMU and ROM) or image instead of the round's (a one-variable
# control); --cold-cache KEY boots with the host's pipeline cache entry KEY
# (~/.cache/reims-vgpu-vulkan/v1/KEY) set aside, put back when QEMU has stopped, both under the
# boot's GPU lock (msi-boot.sh's COLD_CACHE); --qemu-hook CMD runs CMD on the lane host when QEMU
# starts, {PID} its pid and {DIR} the round's directory (QEMU_HOOK); --pre-qemu CMD runs CMD on
# the lane host under the GPU lock just before QEMU starts, its output in the round's
# pre-qemu.txt (PRE_QEMU); --log-add FILTER adds FILTER's directives to the round's host log filter
# (as reims_vgpu_vulkan::icb_writes=debug, a target that is off by default); --host-log FILTER
# replaces the round's filter (as info, for a round timed against info-level bench boots: the
# presents' trace lines cost each op, and its probes' host.log then counts no composites); --hold S sets the
# round's hold (config.sh's HOLD_S) for this round. --list shows the
# queue and what a round would take. --dry-run runs a round on this machine: the lane host's
# directories under a scratch directory and the guest a directory of it (msi-probes.sh FAKE_GUEST),
# with three example probes, one of them for another tip.
set -euo pipefail
L=$(cd "$(dirname "$0")" && pwd)
GH=${LANE_GATE_HOME:-$HOME/.cache/rgk-lane-gate}
force=0 list=0 dry=0 capture=0 want_tip= first= hold= only=0 exclude= from_boot=0 host_tip= image_tip=
cold_cache= qemu_hook= pre_qemu= log_add= host_log= stages=() lstages=() ram=64 after=
oc_path= rom_path= gfx_b64= qemu_env=() metal_check=1 appload=0 kring_s=
while (($#)); do
  a=$1
  shift
  case $a in
    --force) force=1 ;;
    --capture) capture=1 force=1 ;;
    --tip) want_tip=${1:?--tip SHA}; shift ;;
    --first) first=${1:?--first NAME,...}; shift ;;
    --only) first=${1:?--only NAME,...} only=1; shift ;;
    --exclude) exclude=${1?--exclude NAME,...}; shift ;;
    --catch-from-boot) from_boot=1 force=1 ;;
    --host-tip) host_tip=${1:?--host-tip SHA}; shift ;;
    --image-tip) image_tip=${1:?--image-tip SHA}; shift ;;
    --cold-cache) cold_cache=$(tr -dc 0-9a-f <<< "${1:?--cold-cache KEY}"); shift ;;
    --qemu-hook) qemu_hook=${1:?--qemu-hook CMD}; shift ;;
    --pre-qemu) pre_qemu=${1:?--pre-qemu CMD}; shift ;;
    --log-add) log_add=${1:?--log-add FILTER}; shift ;;
    --host-log) host_log=${1:?--host-log FILTER}; shift ;;
    --hold) hold=${1:?--hold S}; shift ;;
    --stage) stages+=("${1:?--stage NAME=DIR[@TIP]}"); shift ;;
    --stage-local) lstages+=("${1:?--stage-local NAME=DIR[@TIP]}"); shift ;;
    --appload) appload=1 ;;
    --kring-s) kring_s=${1:?--kring-s S}; shift ;;
    --ram) ram=${1:?--ram GIB}; shift ;;
    --after) after=${1:?--after CMD}; shift ;;
    --image) oc_path=${1:?--image PATH}; shift ;;
    --rom) rom_path=${1:?--rom PATH}; shift ;;
    --gfx-b64) gfx_b64=${1:?--gfx-b64 B64}; shift ;;
    --qemu-env) qemu_env+=("${1:?--qemu-env K=V}"); shift ;;
    --no-metal-check) metal_check=0 ;;
    --list) list=1 ;;
    --dry-run) dry=1 ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) echo "probe-batch: unknown argument $a" >&2; exit 2 ;;
  esac
done
. "$GH/config.sh"
[[ -n $hold ]] && HOLD_S=$hold
PUB=$(cat "$GH/published-last")
SHORT=${PUB:0:9}
WAIT_N=3 WAIT_S=2700
# The host's log filter in a probe boot: info, and the display's presents ("PRESENT_SURFACE of
# transaction", reims_graphics at trace, about 23 lines a second at 120 Hz), so a probe's host.log
# counts composites. reims_graphics has no other trace lines; at debug it logs SUBMIT_ROOM posts
# (reims_graphics::room). Gate and bench boots log at info.
PROBE_HOST_LOG=${host_log:-info,reims_graphics=trace}
# The hold the probes get: the boot's own time to the desktop and the device check (about 90 s),
# the longest load settle (180 s) and a margin come first.
FIT_S=$(( HOLD_S - 90 - 180 - 60 ))
# A capture's wait for the Dock to settle and its fetch (about 90 s) come off it too (--capture).

((dry)) && { echo "probe-batch: --dry-run does not run as a potluck job yet" >&2; exit 2; }
# Reads of the lane host's state only (its queue, results and published tips): the round's job
# makes every change there.
msi() { ssh -o BatchMode=yes "$MSI" "bash -c $(printf '%q' "$1")"; }
Q=$MSI_RGK_RUN/probe-queue
RES=$MSI_RGK_RUN/probe-results

# The queue: "AGE_S BUDGET TIP NAME" a probe, oldest first.
queue=$(msi "[ -d '$Q' ] || exit 0; cd '$Q' && now=\$(date +%s) && for d in */; do d=\${d%/}; [ -d \"\$d\" ] || continue
  m=\$(stat -c %Y \"\$d\" 2> /dev/null || stat -f %m \"\$d\"); echo \"\$((now - m)) \$(tr -dc 0-9 < \"\$d/budget\" 2> /dev/null || true) \$(tr -dc 0-9a-f < \"\$d/tip\" 2> /dev/null || true) \$d\"
  done | sort -rn")
# The staged probes (--stage), as if just queued: their budget and tip as the job will see them.
stage_specs=()
for st_ in ${stages[@]+"${stages[@]}"}; do
  sn=${st_%%=*} sr=${st_#*=} stip=
  sdir=${sr%@*}
  [[ $sr == *@* ]] && stip=${sr##*@}
  info=$(msi "[ -f '$sdir/run.sh' ] && echo \"\$(tr -dc 0-9 < '$sdir/budget' 2> /dev/null) \$(tr -dc 0-9a-f < '$sdir/tip' 2> /dev/null)\"") ||
    { echo "probe-batch: no probe in $sdir on the lane host" >&2; exit 2; }
  read -r sb st0 <<< "$info"
  stip=${stip:-$st0}
  queue=$(printf '%s\n%s\n' "$queue" "0 ${sb:-180} $stip $sn" | sed '/^$/d')
  stage_specs+=("$sn=$sdir@$stip")
done
# The probes on this machine (--stage-local), carried in the round's commit (lg/probes/NAME).
for st_ in ${lstages[@]+"${lstages[@]}"}; do
  sn=${st_%%=*} sr=${st_#*=} stip=
  sdir=${sr%@*}
  [[ $sr == *@* ]] && stip=${sr##*@}
  [[ -f $sdir/run.sh ]] || { echo "probe-batch: no probe in $sdir" >&2; exit 2; }
  sb=$(tr -dc 0-9 < "$sdir/budget" 2> /dev/null || true)
  stip=${stip:-$(tr -dc 0-9a-f < "$sdir/tip" 2> /dev/null || true)}
  queue=$(printf '%s\n%s\n' "$queue" "0 ${sb:-180} $stip $sn" | sed '/^$/d')
  stage_specs+=("$sn=@WS/lg/probes/$sn@$stip")
done
queue_all=$queue
[[ -n $exclude ]] &&  # --exclude: those stay queued
  queue=$(awk -v l="$exclude" 'BEGIN { n = split(l, a, ","); for (i = 1; i <= n; i++) s[a[i]] = 1 } !($4 in s)' <<< "$queue")
if [[ -n $first ]]; then  # --first: those probes first, in the order given
  IFS=, read -r -a firsts <<< "$first"
  head_=$(for f in "${firsts[@]}"; do awk -v f="$f" '$4 == f' <<< "$queue"; done)
  rest_=$(awk -v l="$first" 'BEGIN { n = split(l, a, ","); for (i = 1; i <= n; i++) s[a[i]] = 1 } !($4 in s)' <<< "$queue")
  ((only)) && rest_=
  queue=$(printf '%s\n%s\n' "$head_" "$rest_" | sed '/^$/d')
fi
n=$(printf '%s' "$queue_all" | grep -c . || true)
oldest=$(printf '%s\n' "$queue_all" | awk '$1 + 0 > m { m = $1 + 0 } END { print m + 0 }')
# The published tips: the newest (published-last), and those whose image and host the lane host
# still has. A probe's tip resolves to one of them by its first 9 characters.
pubs=$(msi "cd '$MSI_RGK_RUN' && for d in tip-host/*/; do d=\${d%/}; s=\${d#tip-host/}
  [ -x \"\$d/qemu-system-x86_64\" ] && [ -f \"loader-fix/oc-sipon-\$s.qcow2\" ] && echo \"\$s\"; done" 2> /dev/null || true)
((dry)) && pubs=$SHORT
published() {  # TIP: its published short name, or nothing
  local t=$1
  (( ${#t} >= 9 )) || { [[ -n $t && $PUB == "$t"* ]] && echo "$SHORT"; return; }
  grep -x -F "${t:0:9}" <<< "$pubs" || { [[ ${t:0:9} == "$SHORT" ]] && echo "$SHORT"; } || true
}
# The round's tip: the newest published one where a probe waits for it, else the oldest probe's
# that is still published.
RT=
while read -r age budget tip name; do
  [[ -n ${name:-} && $(published "${tip:-}") == "$SHORT" ]] && { RT=$SHORT; break; }
done <<< "$queue"
if [[ -z $RT ]]; then
  while read -r age budget tip name; do
    [[ -n ${name:-} ]] || continue
    RT=$(published "${tip:-}")
    [[ -n $RT ]] && break
  done <<< "$queue"
fi
RT=${RT:-$SHORT}
if [[ -n $want_tip ]]; then
  RT=$(published "$want_tip")
  [[ -n $RT ]] || { echo "probe-batch: $want_tip has no published image and host on the lane host" >&2; exit 2; }
fi
# A control's host build or image: another published tip's.
HOST_SHORT=$RT IMAGE_SHORT=$RT
if [[ -n $host_tip ]]; then
  HOST_SHORT=$(published "$host_tip")
  [[ -n $HOST_SHORT ]] || { echo "probe-batch: --host-tip $host_tip is not published on the lane host" >&2; exit 2; }
fi
if [[ -n $image_tip ]]; then
  IMAGE_SHORT=$(published "$image_tip")
  [[ -n $IMAGE_SHORT ]] || { echo "probe-batch: --image-tip $image_tip is not published on the lane host" >&2; exit 2; }
fi
if [[ $RT != "$SHORT" ]]; then
  PUB=$(git -C "$L" rev-parse --verify -q "$RT^{commit}" 2> /dev/null || echo "$RT")
  SHORT=$RT
fi
# What a round takes: the probes for its tip, oldest first, while their budgets fit; those for a
# tip with no published build are skipped (their results say why); the others wait.
((capture)) && FIT_S=$(( FIT_S - 90 ))
((appload)) && FIT_S=$(( FIT_S - 480 ))  # the app load's time (msi-boot.sh's APPLOAD_NEED_S)
take=() fit=0
while read -r age budget tip name; do
  [[ -n ${name:-} ]] || continue
  t=$(published "${tip:-}")
  [[ -z $t ]] && { take+=("$name"); continue; }
  [[ $t == "$SHORT" ]] || continue
  (( fit + ${budget:-180} + 15 <= FIT_S )) || continue
  fit=$(( fit + ${budget:-180} + 15 ))
  take+=("$name")
done <<< "$queue"
echo "probe-batch: $n waiting (oldest $(( ${oldest:-0} / 60 )) min); the round's tip $SHORT$([[ $HOST_SHORT$IMAGE_SHORT != "$SHORT$SHORT" ]] && echo " (host $HOST_SHORT, image $IMAGE_SHORT)"); a round takes: ${take[*]:-none} ($fit s of $FIT_S)"
((list)) && { printf '%s\n' "$queue" | sed '/^$/d; s/^/  /'; exit 0; }
if (( !dry && !force )) && (( n < WAIT_N )) && (( ${oldest:-0} < WAIT_S )); then
  echo "probe-batch: no round (a round runs at $WAIT_N waiting or the oldest at $(( WAIT_S / 60 )) min, or with --force)"
  exit 0
fi
((${#take[@]} || capture || from_boot)) || { echo "probe-batch: nothing to run"; exit 0; }

# The round: a job on the lane host holding the 5080 and the host alone, dispatched from the round's
# own commit (its parent the harness's HEAD), which carries what the job needs.
ROUND=$(date +%Y%m%d-%H%M%S)
RD=$MSI_GATE/probes/$ROUND
# The tip's bench (the boot's check of our Metal device), else the image tip's (the guest it runs
# on), else the newest gate run's.
# (|| true: a tip no gate built, such as a lane's probe-only host, has no run; set -e must not end us.)
run=$(ls -1dt "$GH/runs/${PUB:0:12}-"* 2> /dev/null | head -1 || true)
[[ -x $run/artifacts/payload/bench ]] || run=$(ls -1dt "$GH/runs/$IMAGE_SHORT"* 2> /dev/null | head -1 || true)
[[ -x $run/artifacts/payload/bench ]] || run=$(ls -1dt "$GH/runs/"*/ 2> /dev/null | while read -r d; do
  [[ -x ${d}artifacts/payload/bench ]] && { echo "${d%/}"; break; }; done || true)
[[ -x $run/artifacts/payload/bench ]] || { echo "probe-batch: no gate run with its bench" >&2; exit 1; }
S=$(mktemp -d /tmp/lg-round.XXXXXX)
mkdir -p "$S/lg/bin" "$S/lg/ship/payload"
find "$L" -maxdepth 1 -type f -exec cp -p {} "$S/lg/bin/" \;
cp -p "$run/artifacts/payload/bench" "$S/lg/ship/payload/"
for st_ in ${lstages[@]+"${lstages[@]}"}; do
  sn=${st_%%=*} sr=${st_#*=}
  mkdir -p "$S/lg/probes"
  cp -pR "${sr%@*}" "$S/lg/probes/$sn"
done
cat > "$S/lg/boot.env" << EOF
QEMU=$MSI_RGK_RUN/tip-host/$HOST_SHORT/qemu-system-x86_64
ROM=${rom_path:-$MSI_RGK_RUN/tip-host/$HOST_SHORT/ReimsGraphics.rom}
OC=${oc_path:-$MSI_RGK_RUN/loader-fix/oc-sipon-$IMAGE_SHORT.qcow2}
DISK=$MSI_DISK
TAG=probes-$ROUND
GFX_B64=$gfx_b64
QEMU_ENV="${qemu_env[*]:-}"
METAL_CHECK=$metal_check
PAYLOAD=$RD/payload
CORPUS=
GUEST_USER=$GUEST_USER
GUEST_PASS=$GUEST_PASS
RGK_RUN=$MSI_RGK_RUN
EXCLUSIVE=1
HOLD_S=$HOLD_S
DRAG=0
APPLOAD=$appload
KRING_S=${kring_s:-5}
HOST_LOG=$( ((capture)) && echo "$PROBE_HOST_LOG,reims_vgpu::graphics=debug" || echo "$PROBE_HOST_LOG")${log_add:+,$log_add}
CAPTURE=$capture
CATCH_FROM_BOOT=$from_boot
COLD_CACHE=$cold_cache
QEMU_HOOK_B64=$(printf '%s' "$qemu_hook" | base64 | tr -d '\n')
PRE_QEMU_B64=$(printf '%s' "$pre_qemu" | base64 | tr -d '\n')
PROBES=$RD/probes
PROBE_RESULTS=$RES
PROBE_TIP=$PUB
EOF
printf '%s=%q\n' ROUND "$ROUND" RD "$RD" Q "$Q" RES "$RES" TAKE "${take[*]:-}" STAGES "${stage_specs[*]:-}" \
  AFTER_B64 "$(printf '%s' "$after" | base64 | tr -d '\n')" > "$S/lg/round.env"
branch=steelbrain/lane-gate-round-$ROUND wt=$HOME/.cache/lane-gate-ship/round-$ROUND
commit=$(cd "$L" && bash "$L/ship-commit.sh" "$S" "$(git -C "$L" rev-parse HEAD)" "$branch" "$wt" \
  ":construction: WIP lane-gate probe round $ROUND (throwaway)")
rm -rf "$S"
echo "probe-batch: round $ROUND from $commit on $branch"
# The job's limit, near its own time (potluck reserves an exclusive job's start at the latest limit
# among the jobs running on its machine): the hold, the boot's own time and its waits for quiet and
# the cool-down (12 minutes), and 40 minutes more for --after.
tmo=$(( (HOLD_S + 720) / 60 + ${after:+40} + 0 ))
out=$(cd "$wt" && potluck dispatch --json --timeout "${tmo}m" --ram "$ram" --exclusive --gpu "${POTLUCK_MSI_GPU:-5080}" \
  --on "node=${POTLUCK_MSI_NODE:-msi-laptop}" --label lane-gate=round -- bash lg/bin/msi-round.sh) ||
  { echo "probe-batch: potluck dispatch: $out" >&2; exit 1; }
job=$(python3 -I -c 'import json, sys; print(json.load(sys.stdin)["job"])' <<< "$out")
echo "probe-batch: potluck job $job"
(cd "$wt" && potluck wait "$job") || echo "probe-batch: job $job ended $?"
ws=$(potluck status --json | python3 -I -c '
import json, sys
d = json.load(sys.stdin)
for k in ("running", "recent", "queued"):
    for j in d.get(k) or []:
        if j.get("id") == sys.argv[1] and j.get("workspace"):
            print(j["workspace"]); sys.exit(0)' "$job" || true)
[[ -z $ws ]] || potluck release "$ws" > /dev/null 2>&1 || true
git -C "$L" worktree remove --force "$wt" 2> /dev/null || true
git -C "$L" branch -q -D "$branch" 2> /dev/null || true
((dry)) && { echo "probe-batch: the dry run's results:"; for p in "$RES"/*/; do
  echo "== ${p%/}"; for f in status rc wall_s stdout stderr; do [[ -f $p$f ]] && echo "  $f: $(tr '\n' '|' < "$p$f")"; done
  [[ -d ${p}out ]] && echo "  out/: $(ls "${p}out" | tr '\n' ' ')"; done
  echo "  left in the queue: $(ls "$Q" | tr '\n' ' ')"; }
exit 0
