#!/bin/bash
# The graphics lane's landing gate: checks one commit end to end and compares it with the last
# good baseline. Run it on the Mac, from any checkout of the repository that has this script:
#
#   scripts/internal/lane-gate.sh [--record | --landed] [--baseline SHA] [--resume [RUNID]] [--image PATH]
#                                 [--host DIR] [--load DUTY] [--skip mac|boot] [--phases "PHASE..."]
#                                 [--bench-phases "PHASE..."] [--boot-tag TAG] [--capture-boot]
#                                 [--no-appload] [--no-bench | --wait-bench] [--qemu-hook CMD]
#                                 [--bench-host-log FILTER]
#                                 [--publish | --probe-tip] [--keep-workspace] [--boot-only] [--priority N]
#                                 [--shared-boot] [REV]
#
#   REV           the commit to check (default HEAD); the gate checks commits, never a dirty tree
#   --record      store the run as a sample of REV's baseline (in baselines/SHA/runs), measure the
#                 noise band again from every baseline with two samples or more, and make SHA the
#                 last good baseline (baselines/LAST) when the run passed
#   --landed      REV is a car that landed on the lane: --record, and once its bench leg has run,
#                 REV becomes baselines/LAST whatever the verdict (a car gated without the Mac's
#                 checks is incomplete), so the next car's bench is judged against this one's. Its
#                 own baseline is LAST as it was, the car that landed before it
#   --baseline    compare with SHA's baseline (default: baselines/LAST)
#   --resume      go on with REV's newest run (or RUNID), skipping the steps it finished
#   --image PATH  boot this OpenCore image on the lane host instead of the one the gate builds
#   --host DIR    boot the host build in DIR on the lane host (a published tip-host/SHORT, its QEMU
#                 and option ROM) instead of the one the gate builds: a control that changes only
#                 the host against REV's guest; never --record it
#   --load DUTY   run a fixed GPU load beside the bench boot's measurements (lane-gate/gpuload.c:
#                 the GPU busy DUTY of every 10 ms, 0 to 1), in place of other work's unknown load;
#                 never --record it
#   --skip PART   leave out the Mac's checks (mac) or the correctness boot (boot); the run is then
#                 incomplete
#   --phases      the correctness boot's guest-run.sh phases (default: info sweep decode icb
#                 webgl present final)
#   --bench-phases  the bench boot's (default: info webgl present idle bench decodefps final)
#   --boot-tag TAG  the correctness boot's tag on the lane host (default gate-RUNID-boot): a tag the
#                 host's gpu-urgent file names passes its turnstile ahead of the exclusive waiters
#   --host-log FILTER  the correctness boot's host log filter (REIMS_VGPU_LOG, default info), as
#                 info,reims_vgpu_graphics::room=debug to count a path's debug records
#   --bench-host-log FILTER  the bench boot's host log filter (default info), as
#                 info,reims_vgpu::work=debug for the runtime's work counts each second, which
#                 report.py rates the drags and WebGL by; their logger costs the numbers it rates
#                 (car 9: draws.10k 2.54M a second without it, 0.80M with it), so another filter
#                 makes a control: never --record it
#   --final-cmd CMD  run CMD in the guest (bash, as the user, 120 s at most) at the end of the
#                 correctness boot's final phase; its output is guest.log's "GATE final-cmd:" lines
#   --capture-boot  capture what the correctness boot's display showed as it booted (msi-boot.sh's
#                 CAPTURE, lane-gate/msi-capture.py): its scanout every 100 ms from QEMU's memory,
#                 the guest's screenshots once a second, the kext's log lines; kept on the lane host
#                 in the boot's capture/ (capture.txt, fetched, says where)
#   --no-appload  leave out the correctness boot's app load (lane-gate/msi-appload.sh: every app
#                 and 7 Safari pages after the measurements, on a copy of rgk-ctx-slots' appload
#                 tools from the lane host's APPLOAD_TOOLS, default ~/.cache/rgk-ctx-slots-msi/appload/tools)
#   --appload-info WHY  report a miss of the app load's bar as INFO, not FAIL, saying why (a
#                 baseline taken before the bar holds); kept in runs/RUNID/appload-info
#   --qemu-hook CMD  run CMD on the lane host as the bench boot's QEMU starts, as probe-batch.sh's
#                 --qemu-hook ({PID} QEMU's pid, {DIR} the boot's directory, whose guest.log has
#                 guest-run.sh's lines as they come, "# start CASE" before each bench case); a
#                 control: never --record it
#   --post-probes DIR  run the probes in DIR (one directory each, as probe-batch.sh's: run.sh,
#                 budget, tip; DIR/.order their order) in the correctness boot after everything
#                 the gate measures and checks there, the app load's desktop still up
#                 (lane-gate/msi-probes.sh, into the boot's post-probes/; report.py lists each as
#                 INFO, and their fatal lines count in the boot's); the boot's hold grows by their
#                 budgets. Given to the build, it ships DIR; given to a --boot-only pass alone, the
#                 pass adds DIR to the run's ship commit (a child commit) and the boot moves the
#                 kept workspace to it (potluck --advance)
#   --no-bench    leave out the bench leg; --wait-bench waits for it before exiting, where by
#                 default it runs on after the verdict, detached (its output in runs/RUNID/bench.out)
#   --publish     REV is the lane's tip: publish its image and host build on the lane host, read-only,
#                 as loader-fix/oc-sipon-SHORT.qcow2 and tip-host/SHORT/ under the host's RGK_RUN
#                 (lane-gate/msi-publish.sh), unless nothing they carry changed since the last
#                 commit the gate published (published-last)
#   --probe-tip   publish REV's image and host build as --publish does, always, for probes tipped at
#                 it (a fix's proof), without making it the lane's tip: published-last stays, so a
#                 probe round's tip stays the lane's (probe-batch.sh --tip SHA runs a round on REV)
#   --keep-workspace  keep the run's potluck workspace, worktree and branch at its end (a run built
#                 with --skip boot, to boot later with --resume --boot-only)
#   --boot-only   the correctness boot of a resumed run alone, in its kept workspace, taking no lock
#                 (as a bench leg), so two runs' boots can be queued back to back
#   --priority N  the potluck jobs' priority (default 3, a --landed car's 4)
#   --shared-boot  the correctness boot's job takes the 5080 alone, not the whole host (no --exclusive):
#                 it times nothing, so other jobs' builds may run beside it once its 180 s wait for
#                 them ends; the bench boot keeps the whole host
#
# What it runs: the Mac builds the guest's side, and the lane host runs the rest as potluck jobs
# (potluck's lease is the lock), dispatched from a throwaway commit of the run's own (do_ship):
#   Mac:  the guest kexts, bundle and tools (lane-gate/guest: guest-run.sh's payload), and the
#         kext set: guest/graft's Graft and the candidate's ReimsGraphics and RgfxGate. The Mac's
#         checks (`make -C guest/kext check`, the kext's, bundle's and RgfxGate's tests, `cargo fmt
#         --all --check` and the cargo tests the change reaches) do not run as potluck jobs yet:
#         a gate runs with --skip mac;
#   host: one job with no GPU (lane-gate/msi-job.sh build): the candidate's library dist, option
#         ROMs and QEMU (lane-gate/msi-build.sh), and its SIP-on OpenCore image (the set's kexts,
#         RgfxGate carrying its bundle, lane-gate/msi-image.sh), and publishing;
#   boot: the correctness boot, which decides: one cold boot on overlays (lane-gate/msi-boot.sh,
#         through lane-gate/msi-launch.sh), a job holding the 5080 and the whole host
#         (--exclusive), after running builds end and the CPU package cools below 80 C, running
#         the guest's checks (lane-gate/guest/guest-run.sh): the lane's correctness sweep, the
#         H.264 and HEVC decode sets against their MD5s, Safari's WebGL readPixels, present_probe's
#         answers full-screen and windowed, and the desktop's state (SIP on, the kexts loaded and
#         bound, the bundle installed, the Metal device ours, WindowServer up throughout, no new
#         crash report, and the guest's capture of its screen); and the host log's fatal lines and
#         display updates.
#   bench: the bench leg, which reports and never decides: a second cold boot, a job holding the
#         5080 and the host alone the same way, for at most HOLD_S seconds, running the GPU
#         benchmark, the on-screen benchmark, present_probe and the timed decodes, with QEMU's CPU
#         and the host's display updates, each number against the baseline's with its noise band.
# Then lane-gate/report.py writes the run's report.txt and results.json: PASS, FAIL or SKIP for
# every item (KNOWN for a failure the known-failures list, KNOWN_FAILS, matches exactly, with its
# hits across runs in known-hits.tsv), and each number against the baseline. Exit 0 when the run passed, 1 when a check
# failed, 2 when a part did not run; the bench leg's failures and regressions show in the report
# and never change the exit.
#
# Its state is under LANE_GATE_HOME (default ~/.cache/rgk-lane-gate): config.sh (this host's
# paths: the KDK, the lane host's; config.sh's comments say what each is),
# tree/ (a worktree the gate checks commits out into, so builds there are incremental),
# target/, runs/RUNID/ (steps/, artifacts/, results/, report.txt), baselines/SHA/runs/*.json
# and LAST, and noise.json. One gate runs at a time on the Mac.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
L=$here/lane-gate
SRC=$(cd "$here/../.." && pwd)
GH=${LANE_GATE_HOME:-$HOME/.cache/rgk-lane-gate}

die() { echo "lane-gate: $*" >&2; exit 2; }
# One gate at a time: lockf holds $GH/lock for the run, and refuses at once when another holds it.
# A bench leg (--bench-only) takes no lock: it waits its turn on the lane host beside later gates.
mkdir -p "$GH/runs" "$GH/baselines"
if [[ -z ${LANE_GATE_LOCKED:-} && " $* " != *" --bench-only "* && " $* " != *" --boot-only "* ]]; then
  export LANE_GATE_LOCKED=1
  exec lockf -t 0 "$GH/lock" "$0" "$@"
fi
[[ -f $GH/config.sh ]] || die "no $GH/config.sh: it names this host's KDK and lane host"
. "$GH/config.sh"

shared_boot=0 record=0 landed=0 qemu_hook= bench_host_log= base_sha= resume= image= host_dir= load= skip= phases= bench_phases= boot_tag= host_log= capture=0 appload=1 appload_info= rev=HEAD bench=detach publish=0 probe_tip=0 keep_ws=0 boot_only=0 prio= post_probes= advance=
while (($#)); do
  case $1 in
    --record) record=1 ;;
    --landed) record=1 landed=1 ;;
    --qemu-hook) qemu_hook=$2; shift ;;
    --bench-host-log) bench_host_log=$2; shift ;;
    --baseline) base_sha=$2; shift ;;
    --resume) if [[ ${2:-} =~ ^[0-9a-f]{12}- ]]; then resume=$2; shift; else resume=newest; fi ;;
    --image) image=$2; shift ;;
    --host) host_dir=$2; shift ;;
    --load) load=$2; shift ;;
    --bench-phases) bench_phases=$2; shift ;;
    --skip) skip="$skip $2"; shift ;;
    --phases) phases=$2; shift ;;
    --boot-tag) boot_tag=$2; shift ;;
    --host-log) host_log=$2; shift ;;
    --capture-boot) capture=1 ;;
    --final-cmd) final_cmd=$2; shift ;;
    --no-appload) appload=0 ;;
    --appload-info) appload_info=$2; shift ;;
    --post-probes) post_probes=$(cd "$2" && pwd) || die "no directory $2"; shift ;;
    --no-bench) bench=no ;;
    --wait-bench) bench=wait ;;
    --bench-only) bench=only ;;  # the bench leg of a resumed run alone (what the detached leg runs)
    --publish) publish=1 ;;
    --probe-tip) publish=1 probe_tip=1 ;;
    --keep-workspace) keep_ws=1 ;;
    --boot-only) boot_only=1 ;;
    --shared-boot) shared_boot=1 ;;
    --priority) prio=$2; shift ;;
    -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    -*) die "unknown option $1" ;;
    *) rev=$1 ;;
  esac
  shift
done

sha=$(git -C "$SRC" rev-parse --verify "$rev^{commit}") || die "no commit $rev"
subject=$(git -C "$SRC" log -1 --format=%s "$sha")
if [[ -n $resume ]]; then
  if [[ $resume == newest ]]; then
    resume=$(ls -1d "$GH/runs/${sha:0:12}-"* 2> /dev/null | tail -1 | xargs -n1 basename 2> /dev/null || true)
    [[ -n $resume ]] || die "no run of ${sha:0:12} to resume"
  fi
  runid=$resume
else
  runid=${sha:0:12}-$(date +%Y%m%d-%H%M%S)
fi
RUN=$GH/runs/$runid
ART=$RUN/artifacts
RES=$RUN/results
mkdir -p "$RUN/steps" "$ART" "$RES"
[[ -z $appload_info ]] || echo "$appload_info" > "$RUN/appload-info"
MR=$MSI_GATE/runs/$runid
TREE=$GH/tree
export CARGO_TARGET_DIR=$GH/target
T0=$(date +%s)
[[ -f $RUN/started ]] || date +%s > "$RUN/started"

# A resumed run keeps the baseline it started with: LAST may since have become the run's own commit.
if [[ -z $base_sha && -n $resume && -f $RUN/meta.json ]]; then
  base_sha=$(python3 -I -c 'import json, sys; print(json.load(open(sys.argv[1])).get("baseline") or "")' "$RUN/meta.json")
fi
if [[ -z $base_sha && -e $GH/baselines/LAST ]]; then base_sha=$(basename "$(readlink "$GH/baselines/LAST")"); fi
[[ -z $base_sha ]] || base_sha=$(git -C "$SRC" rev-parse --verify "$base_sha^{commit}")
[[ -z $host_dir ]] || ((!record)) || die "--host makes a control: it is never a sample of REV's baseline (--record)"
[[ -z $load ]] || ((!record)) || die "--load measures beside a fixed load: it is never a sample of REV's baseline (--record)"
[[ -z $qemu_hook ]] || ((!record)) || die "--qemu-hook makes a control: it is never a sample of REV's baseline (--record)"
[[ -z $bench_host_log ]] || ((!record)) || die "--bench-host-log makes a control: it is never a sample of REV's baseline (--record)"
BASE=${base_sha:+$GH/baselines/$base_sha}
[[ -z $BASE || -d $BASE/runs ]] || die "no baseline of $base_sha in $GH/baselines"

say() { echo "$(date +%T) $*"; }
# step NAME FUNCTION...: runs FUNCTION with errexit, its output into steps/NAME.log, and leaves
# steps/NAME.ok or .fail holding its seconds. A step that finished before is not run again.
step() {
  local name=$1 t rc
  shift
  [[ -f $RUN/steps/$name.ok ]] && { say "  $name: done in an earlier pass"; return 0; }
  rm -f "$RUN/steps/$name.fail"
  t=$(date +%s)
  set +e
  ( set -e; "$@" ) > "$RUN/steps/$name.log" 2>&1
  rc=$?
  set -e
  if ((rc == 0)); then
    echo $(( $(date +%s) - t )) > "$RUN/steps/$name.ok"
    say "  $name: ok in $(cat "$RUN/steps/$name.ok") s"
  else
    echo $(( $(date +%s) - t )) > "$RUN/steps/$name.fail"
    say "  $name: FAILED (rc $rc) after $(cat "$RUN/steps/$name.fail") s; log $RUN/steps/$name.log"
    tail -6 "$RUN/steps/$name.log" | sed 's/^/      /'
  fi
  return $rc
}
skipped() { [[ " $skip " == *" $1 "* ]]; }

# --- the tree ---------------------------------------------------------------------------------

do_tree() {
  if [[ ! -e $TREE/.git ]]; then
    (cd "$SRC" && scripts/internal/new-worktree.sh "$TREE" steelbrain/lane-gate-tree-start "$sha")
    git -C "$TREE" checkout -q --detach
    git -C "$SRC" branch -q -D steelbrain/lane-gate-tree-start
  fi
  git -C "$TREE" checkout -q -f --detach "$sha"
  [[ $(git -C "$TREE" rev-parse HEAD) == "$sha" ]]
  git -C "$TREE" status --porcelain --untracked-files=no | { ! grep -q .; }
}

# --- the Mac: the guest's build ---------------------------------------------------------------

tool_src() {  # the candidate's tool source, or the gate's where the candidate has none
  if [[ -f $TREE/$1 ]]; then echo "$TREE/$1"; else echo "$SRC/$1"; fi
}
do_guest_build() {
  cd "$TREE"
  make -C guest/kext
  make -C guest/bundle package sweep bench
  local cc=(xcrun clang -arch x86_64 -mmacosx-version-min=14.0 -fobjc-arc -O2 -Wall -Wextra -Werror -Wno-unused-parameter)
  local P=$ART/payload
  rm -rf "$P" "$ART/ReimsGraphics.kext" "$ART/RgfxGate.kext"
  mkdir -p "$P"
  "${cc[@]}" -framework AppKit -framework Metal -framework QuartzCore -framework CoreVideo \
    "$(tool_src guest/bundle/tools/present_probe.m)" -o "$P/present_probe"
  "${cc[@]}" -framework Foundation -framework CoreMedia -framework CoreVideo -framework VideoToolbox \
    "$(tool_src guest/bundle/tools/decode_md5.m)" -o "$P/decode_md5"
  "${cc[@]}" -framework Foundation -framework CoreGraphics \
    "$(tool_src guest/bundle/tools/winmon.m)" -o "$P/winmon"
  cp guest/bundle/build/sweep-x86_64 "$P/sweep"
  cp guest/bundle/build/bench-x86_64 "$P/bench"
  # oracle_icb (the ICB probe, with its metallib beside it), as its build.sh builds it, where the
  # tree has it: the icb phase's checks of CPU- and GPU-encoded indirect command buffers.
  local o=guest/bundle/tools/oracle
  if [[ -f $o/oracle_icb.m && -f $o/oracle_icb.metal ]]; then
    xcrun clang -arch x86_64 -mmacosx-version-min=26.0 -fobjc-arc -Wall -framework Metal -framework IOSurface \
      -framework CoreVideo -framework QuartzCore -framework VideoToolbox -framework CoreGraphics \
      -framework Foundation "$o/oracle_icb.m" -o "$P/oracle_icb"
    xcrun -sdk macosx metal -std=metal3.0 -mmacosx-version-min=26.0 -c "$o/oracle_icb.metal" -o "$ART/oracle_icb.air"
    xcrun -sdk macosx metallib "$ART/oracle_icb.air" -o "$P/oracle_icb.metallib"
    rm -f "$ART/oracle_icb.air"
  fi
  cp "$L/guest/"* "$P/"
  # RgfxGate's payload is the bundle it carries; its Makefile does not see the bundle change.
  make -C guest/rgfxgate clean
  make -C guest/rgfxgate BUNDLE_DIR="$TREE/guest/bundle/build/package/reims-graphics.bundle"
  cp -R guest/kext/build/ReimsGraphics.kext guest/rgfxgate/build/RgfxGate.kext "$ART/"
  cp guest/bundle/build/reims-graphics.bundle.tgz "$ART/"
  # The codecs the candidate's device offers the guest where the host decodes them: H.264 where
  # its bundle has the H.264 driver, HEVC where its kext states IOGVAHEVCDecode.
  { grep -rqs RGFX_VIDEO_CODEC_H264 guest/bundle/src && echo h264
    grep -rqs IOGVAHEVCDecode guest/kext/src && echo hevc
    true; } > "$ART/tip-codecs.txt"
  { for f in guest/bundle/tools/sweep/*.[mh] guest/bundle/tools/bench/bench.m; do shasum "$f"; done | shasum | echo "sweep+bench $(cut -c1-16)"
    echo "bench $(shasum < guest/bundle/tools/bench/bench.m | cut -c1-16)"  # bench.m builds alone
    echo "present_probe $(shasum < "$(tool_src guest/bundle/tools/present_probe.m)" | cut -c1-16)"
    echo "decode_md5 $(shasum < "$(tool_src guest/bundle/tools/decode_md5.m)" | cut -c1-16)"
    [[ ! -f $o/oracle_icb.m ]] || echo "oracle_icb $(cat "$o/oracle_icb.m" "$o/oracle_icb.metal" "$o/oracle_common.h" | shasum | cut -c1-16)"
    echo "drag $(cat "$(tool_src guest/bundle/tools/winmon.m)" "$L/msi-drag.py" | shasum | cut -c1-16)"
    echo "bundle $(shasum < guest/bundle/build/reims-graphics.bundle | cut -c1-16)"
    echo "kext $(shasum < guest/kext/build/ReimsGraphics.kext/Contents/MacOS/ReimsGraphics | cut -c1-16)"
  } > "$ART/tools.txt"
  cat "$ART/tools.txt"
}

# --- the Mac's checks -------------------------------------------------------------------------

# The kext's imports and layout against the KDK, then the set linked into BOOT_KC with
# OpenCore's linker, ReimsGraphics still bindable by Graft (guest/'s set-check).
do_kext_check() {
  [[ -n $KDK && -n $KEXTINJECT && -n $BOOT_KC ]]
  make -C "$TREE/guest/kext" check KDK="$KDK" | tee "$RES/kext-check.txt"
  make -C "$TREE/guest" set OUT="$ART/set"
  make -C "$TREE/guest" set-check SET="$ART/set" KEXTINJECT="$KEXTINJECT" BOOT_KC="$BOOT_KC" | tee -a "$RES/kext-check.txt"
  ! grep -q SKIPPED "$RES/kext-check.txt"
}
do_kext_test() { make -C "$TREE/guest/kext" test; }
do_bundle_test() { make -C "$TREE/guest/bundle" test; }
do_rgfxgate_test() { make -C "$TREE/guest/rgfxgate" test; }
# That the packaged bundle, the one RgfxGate installs, writes no file of its own (the tree's
# bundle_files_check.sh, where it has one).
do_bundle_files() {
  local c=$TREE/guest/bundle/tests/bundle_files_check.sh
  if [[ ! -f $c ]]; then echo "no guest/bundle/tests/bundle_files_check.sh in this tree"; return 0; fi
  bash "$c" "$TREE/guest/bundle/build/package/reims-graphics.bundle"
}
do_fmt() { cd "$TREE" && . scripts/internal/dev-env.sh && cargo fmt --all --check; }

# The packages the change reaches: those with a file changed since the baseline, and every
# workspace package that depends on one of them.
reached() {
  [[ -n $base_sha ]] || return 0
  local changed
  changed=$(git -C "$SRC" diff --name-only "$base_sha" "$sha" -- crates xtask | awk -F/ '$1 == "xtask" {print "xtask"; next} {print $2}' | sort -u)
  # The workspace's manifest, lock and toolchain (a pin move among them) reach every package.
  if ! git -C "$SRC" diff --quiet "$base_sha" "$sha" -- Cargo.toml Cargo.lock rust-toolchain.toml .cargo; then
    (cd "$TREE" && cargo metadata --no-deps --format-version 1) |
      python3 -I -c 'import json, sys; print(" ".join(sorted(p["name"] for p in json.load(sys.stdin)["packages"])))'
    return 0
  fi
  [[ -n $changed ]] || return 0
  (cd "$TREE" && cargo metadata --no-deps --format-version 1) | python3 -I -c '
import json, sys
meta = json.load(sys.stdin)
changed = set(sys.argv[1].split())
names = {p["name"] for p in meta["packages"]}
deps = {p["name"]: {d["name"] for d in p["dependencies"] if d["name"] in names} for p in meta["packages"]}
out = set(n for n in changed if n in names)
grew = True
while grew:
    grew = False
    for n, ds in deps.items():
        if n not in out and ds & out:
            out.add(n); grew = True
print(" ".join(sorted(out)))' "$changed"
}
do_cargo() {
  cd "$TREE"
  . scripts/internal/dev-env.sh
  local more expr pkgs=(-p reims-vgpu-graphics -p reims-vgpu)
  more=$(reached)
  expr="package(reims-vgpu-graphics) | (package(reims-vgpu) & test(/graphics/))"
  for p in $more; do
    [[ $p == reims-ctl ]] && continue
    pkgs+=(-p "$p")
    expr="$expr | package($p)"
  done
  echo "packages: ${pkgs[*]}; filter: $expr"
  local feats=reims-vgpu-vulkan/testkit,reims-vgpu/testkit,reims-vgpu/vulkan
  printf '[profile.default.junit]\npath = "junit.xml"\n' > "$GH/nextest-tool.toml"
  cargo nextest run --no-run "${pkgs[@]}" --features "$feats"
  # nextest keeps its store under the workspace's target/, whatever CARGO_TARGET_DIR says.
  local junit=$TREE/target/nextest/default/junit.xml
  rm -f "$junit"
  # The GPU's tests under the Mac's GPU lock, built before it is taken.
  set +e
  "$SRC/scripts/internal/gpu-locks.sh" -- cargo nextest run "${pkgs[@]}" --features "$feats" --no-fail-fast \
    --tool-config-file "lane-gate:$GH/nextest-tool.toml" -E "$expr" 2>&1 | tail -40
  set -e
  # A run that ended before its report (stopped, or killed with its lock's ticket) wrote no junit:
  # the step fails (errexit does not reach inside a step's function).
  [[ -s $junit ]] || { echo "cargo: nextest wrote no $junit: the run did not finish"; return 1; }
  python3 -I - "$junit" "$RES" << 'EOF' || return 1
import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
fails, total = [], 0
for suite in root.iter("testsuite"):
    for case in suite.iter("testcase"):
        total += 1
        if case.find("failure") is not None or case.find("error") is not None:
            fails.append(f"{suite.get('name')} {case.get('name')}")
open(sys.argv[2] + "/cargo-fails.txt", "w").write("".join(f + "\n" for f in sorted(fails)))
open(sys.argv[2] + "/cargo-summary.txt", "w").write(f"{total} tests run, {len(fails)} failed\n")
print(f"{total} tests run, {len(fails)} failed")
EOF
  # A GPU test that fails under the check's load runs once more alone (AGENTS.md, Checking a
  # change): passing alone, it is not a failure, and the report names it as load-sensitive.
  # A known failure (the baseline's, or MAC_KNOWN_FAILS) is not run again: it fails as known.
  : > "$RES/cargo-alone.txt"
  local bin name rest=() known=$GH/baselines/LAST/runs
  local knownlist
  knownlist=$( { cat "${MAC_KNOWN_FAILS:-/dev/null}" 2> /dev/null
    python3 -I -c 'import json, os, sys
d = sys.argv[1]
for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
    for t in json.load(open(os.path.join(d, f))).get("cargo_fails") or []:
        print(t)' "$known" 2> /dev/null; } | awk '{print $NF}' | sort -u)
  while read -r bin name; do
    [[ -n $name ]] || continue
    if grep -q -x -F "$name" <<< "$knownlist"; then
      rest+=("$bin $name")
      continue
    fi
    if "$SRC/scripts/internal/gpu-locks.sh" -- cargo nextest run "${pkgs[@]}" --features "$feats" \
      -E "binary_id($bin) & test(=$name)" > "$RES/cargo-alone.log" 2>&1; then
      echo "$bin $name" >> "$RES/cargo-alone.txt"
      echo "passes alone: $bin $name"
    else
      rest+=("$bin $name")
      echo "fails alone too: $bin $name"
    fi
  done < "$RES/cargo-fails.txt"
  printf '%s\n' ${rest[@]+"${rest[@]}"} | sed '/^$/d' > "$RES/cargo-fails.txt"
}

mac_checks() {
  local ok=0
  step kext-check do_kext_check || ok=1
  step kext-test do_kext_test || ok=1
  step bundle-test do_bundle_test || ok=1
  step bundle-files do_bundle_files || ok=1
  step rgfxgate-test do_rgfxgate_test || ok=1
  step fmt do_fmt || ok=1
  step cargo do_cargo || ok=1
  return $ok
}

# --- the lane host: potluck jobs ---------------------------------------------------------------
#
# Everything on the lane host runs as potluck jobs (potluck's lease is the lock), dispatched from
# a throwaway commit the gate makes for the run (do_ship): its parent is the candidate, and its
# tree is lg/ alone, the gate's host scripts and what the Mac built (msi-job.sh says what). The
# commit is on its own branch, steelbrain/lane-gate-ship-RUNID, checked out in its own worktree,
# ~/.cache/lane-gate-ship/RUNID, which the dispatches run from. The host build is one job with no
# GPU (msi-job.sh build), whose workspace is held for the boots; each boot is a job in that
# workspace with the 5080 and the whole host (--exclusive). Both go after the run: the workspace
# released, the worktree and the branch removed (pl_cleanup).

SHIP_WT=$HOME/.cache/lane-gate-ship/$runid
SHIP_BRANCH=steelbrain/lane-gate-ship-$runid
PL_NODE=${POTLUCK_MSI_NODE:-msi-laptop}
PL_GPU=${POTLUCK_MSI_GPU:-5080}
# A landed car's checks go ahead of other waiting jobs (a car's checks go first on msi).
PL_PRIO=${prio:-$( ((landed)) && echo 4 || echo 3)}

# pl_run NAME DISPATCH-OPTION...: dispatches a job from the run's worktree, waits for it with its
# output, and leaves the job's id in RUN/job-NAME; its status is the job's exit.
pl_run() {
  local name=$1 out job
  shift
  out=$(cd "$SHIP_WT" && potluck dispatch --json "$@") || { echo "potluck dispatch: $out"; return 2; }
  job=$(python3 -I -c 'import json, sys; print(json.load(sys.stdin)["job"])' <<< "$out")
  echo "$job" > "$RUN/job-$name"
  echo "potluck job $job"
  (cd "$SHIP_WT" && potluck wait "$job")
}
# pl_kept NAME PATH DEST [WS]: what job NAME brought back of PATH with its --keep (potluck's
# jobs/JOB/keep/, the workspace's layout), copied to DEST; else, where WS is given, PATH from that
# held workspace (potluck cp), for a job that ended without its keep; fails where neither has it.
pl_kept() {
  local k=${POTLUCK_STATE:-$HOME/.local/state/potluck}/jobs/$(cat "$RUN/job-$1")/keep/$2
  if [[ -e $k ]]; then
    cp -R "$k" "$3" && echo "$2: from the job's keep"
  else
    [[ -n ${4:-} ]] && potluck cp "$4:$2" "$3" && echo "$2: no keep, from the workspace (potluck cp)"
  fi
}
# pl_workspace JOB: the job's workspace.
pl_workspace() {
  potluck status --json | python3 -I -c '
import json, sys
d = json.load(sys.stdin)
for k in ("running", "recent", "queued"):
    for j in d.get(k) or []:
        if j.get("id") == sys.argv[1] and j.get("workspace"):
            print(j["workspace"]); sys.exit(0)
sys.exit(1)' "$1"
}
pl_cleanup() {
  local ws
  ws=$(cat "$RUN/workspace" 2> /dev/null || true)
  [[ -z $ws ]] || potluck release "$ws" > /dev/null 2>&1 || true
  [[ ! -e $SHIP_WT ]] || git -C "$SRC" worktree remove --force "$SHIP_WT" || true
  git -C "$SRC" branch -q -D "$SHIP_BRANCH" 2> /dev/null || true
  say "  potluck: released ${ws:-no workspace}; removed $SHIP_WT and $SHIP_BRANCH"
}

# The set's kexts but RgfxGate, which ships on its own: Graft, the committed build guest/graft
# carries, and the candidate's ReimsGraphics, which its build stages for Graft to bind, as
# `make -C guest set` stages them.
do_set() {
  rm -rf "$ART/kexts"
  mkdir -p "$ART/kexts"
  cp -R "$TREE/guest/graft/dist/Graft.kext" "$ART/ReimsGraphics.kext" "$ART/kexts/"
}

# The paths whose changes reach the image or the host build.
PUBLISHED_PATHS=(guest/kext guest/bundle guest/rgfxgate crates firmware vendor/qemu xtask Cargo.toml Cargo.lock)
# Whether the run publishes the tip (--publish): not where nothing the image or the host carries
# changed since the last commit the gate published, unless --probe-tip.
publish_wanted() {
  ((publish)) && [[ -z $image ]] || return 1
  local last
  last=$(cat "$GH/published-last" 2> /dev/null || true)
  if ((!probe_tip)) && [[ -n $last ]] && git -C "$SRC" cat-file -e "$last^{commit}" 2> /dev/null &&
    git -C "$SRC" diff --quiet "$last" "$sha" -- "${PUBLISHED_PATHS[@]}"; then
    echo "NOT_PUBLISHED: nothing the image or the host carries changed since ${last:0:9}, whose they are" > "$RUN/not-published"
    return 1
  fi
}

# The run's commit: lg/ holds the gate's host scripts (bin/), what the Mac built (ship/), and the
# run's settings (run.env); its parent is the candidate. Checked out in SHIP_WT on SHIP_BRANCH.
do_ship() {
  local S=$ART/ship qc m2v cdhash pub=0 doimg=1 gd idx tree commit
  rm -rf "$S" && mkdir -p "$S/lg/bin" "$S/lg/ship"
  find "$L" -maxdepth 1 -type f -exec cp -p {} "$S/lg/bin/" \;
  cp -p "$SRC/scripts/internal/opencore-add-kexts.sh" "$S/lg/bin/"
  # The guest runner and its files as the gate has them now, and the sweep's case order from its
  # last full run (guest-run.sh runs the cases a crash cut off one by one in that order), and the
  # streams to check again (decode-repeat.txt, "SET STREAM N" a line), where the gate has any.
  cp -pR "$ART/payload" "$S/lg/ship/payload"
  cp -p "$L/guest/"* "$S/lg/ship/payload/"
  [[ -z $post_probes ]] || cp -pR "$post_probes" "$S/lg/post-probes"  # --post-probes
  [[ ! -s $GH/sweep-cases.txt ]] || cp "$GH/sweep-cases.txt" "$S/lg/ship/payload/"
  rm -f "$S/lg/ship/payload/decode-repeat.txt"
  [[ ! -s $GH/decode-repeat.txt ]] || cp "$GH/decode-repeat.txt" "$S/lg/ship/payload/"
  cp -p "$ART/reims-graphics.bundle.tgz" "$S/lg/ship/"
  if [[ -z $image ]]; then
    cp -pR "$ART/kexts" "$S/lg/ship/kexts"
    cp -pR "$ART/RgfxGate.kext" "$S/lg/ship/RgfxGate.kext"
  else
    doimg=0
  fi
  # vendor/qemu's commits the lane host's seed repository lacks, as a git bundle (a few KiB).
  qc=$(git -C "$TREE" ls-tree "$sha" vendor/qemu | awk '{print $3}')
  [[ -n $qc ]]
  echo "$qc" > "$ART/qemu-commit"
  local qg qb
  qg=$(git -C "$QEMU_REPO" rev-parse --absolute-git-dir)
  qb=$(mktemp -d /tmp/lg-qemu-bundle.XXXXXX)
  git init -q --bare "$qb"
  echo "$qg/objects" > "$qb/objects/info/alternates"
  git -C "$qb" update-ref refs/heads/lane-gate "$qc" || { echo "vendor/qemu's $qc is not in $QEMU_REPO"; rm -rf "$qb"; return 1; }
  local nots=()
  for c in ${MSI_QEMU_HAVE:-}; do git -C "$qb" cat-file -e "$c^{commit}" 2> /dev/null && nots+=("^$c") || true; done
  if [[ -n $(git -C "$qb" rev-list -1 refs/heads/lane-gate ${nots[@]+"${nots[@]}"}) ]]; then
    git -C "$qb" bundle create -q "$S/lg/ship/qemu.bundle" refs/heads/lane-gate ${nots[@]+"${nots[@]}"}
  fi
  rm -rf "$qb"
  m2v=$(grep -m1 -o -E '^metal2vulkan-runtime = .*rev = "[0-9a-f]+"' "$TREE/Cargo.toml" | grep -o -E '[0-9a-f]{40}' || echo unknown)
  echo "$m2v" > "$RUN/m2v"
  # The signed bundle RgfxGate carries, so that a fixer can build an RgfxGate of exactly the tip's.
  cdhash=$(codesign -dvvv "$TREE/guest/bundle/build/package/reims-graphics.bundle" 2>&1 | sed -n 's/^CDHash=//p')
  echo "$cdhash" > "$ART/cdhash"
  rm -f "$RUN/not-published"
  publish_wanted && pub=1
  {
    printf '%s=%q\n' RUNID "$runid" SHA "$sha" QC "$qc" SHORT "${sha:0:9}" M2V "$m2v" CDHASH "$cdhash" \
      PUBLISH "$pub" IMAGE "$doimg" MSI_GATE "$MSI_GATE" MSI_RGK_RUN "$MSI_RGK_RUN" MSI_BASE_OC "$MSI_BASE_OC" \
      MSI_CORPUS_SRC "$MSI_CORPUS_SRC" MSI_CORPUS_ODD "${MSI_CORPUS_ODD:-}" MSI_QEMU_SUBPROJECTS "$MSI_QEMU_SUBPROJECTS" \
      MSI_QEMU_SEED "$MSI_QEMU_SEED"
  } > "$S/lg/run.env"
  # Build output only: the gate's own scripts, our kexts, bundle and guest tools (no Apple binary,
  # nothing from a guest), committed on a throwaway branch that never merges.
  gd=$(git -C "$SRC" rev-parse --path-format=absolute --git-common-dir)
  idx=$RUN/ship.index
  rm -f "$idx"
  if [[ -n $(cd "$S" && find . -type l -print -quit) ]]; then echo "ship: symbolic links in $S"; return 1; fi
  (cd "$S" && find . -type f | sed 's#^\./##' | LC_ALL=C sort) > "$RUN/ship.files"
  sed "s#^#$S/#" "$RUN/ship.files" | git --git-dir="$gd" hash-object -w --stdin-paths > "$RUN/ship.blobs"
  while IFS= read -r f; do if [[ -x $S/$f ]]; then echo 100755; else echo 100644; fi; done < "$RUN/ship.files" > "$RUN/ship.modes"
  paste -d' ' "$RUN/ship.modes" "$RUN/ship.blobs" | paste -d'\t' - "$RUN/ship.files" |
    GIT_INDEX_FILE=$idx git --git-dir="$gd" update-index --add --index-info
  tree=$(GIT_INDEX_FILE=$idx git --git-dir="$gd" write-tree)
  commit=$(git --git-dir="$gd" commit-tree "$tree" -p "$sha" -m ":construction: WIP lane-gate ship $runid (throwaway)

The gate's host scripts and the Mac's build of ${sha:0:12} for its potluck jobs on
the lane host. Never merged; removed after the run.")
  rm -f "$idx"
  [[ ! -e $SHIP_WT ]] || git -C "$SRC" worktree remove --force "$SHIP_WT"
  git -C "$SRC" branch -f -q "$SHIP_BRANCH" "$commit"
  mkdir -p "$(dirname "$SHIP_WT")"
  git -C "$SRC" worktree add -q "$SHIP_WT" "$SHIP_BRANCH"
  echo "shipped $commit on $SHIP_BRANCH: $(wc -l < "$RUN/ship.files") files, $(du -sh "$S" | cut -f1)"
  [[ ! -s $RUN/not-published ]] || cat "$RUN/not-published"
}

# host_step NAME: the host job's step NAME as one of this run's steps (steps/NAME.ok or .fail and
# its log), where the job ran it.
host_step() {
  local n=$1 d=$RUN/host-steps
  [[ -f $d/$n.rc ]] || return 0
  cp "$d/$n.log" "$RUN/steps/$n.log"
  rm -f "$RUN/steps/$n.ok" "$RUN/steps/$n.fail"
  if [[ $(cat "$d/$n.rc") == 0 ]]; then
    cp "$d/$n.s" "$RUN/steps/$n.ok"; say "  $n: ok in $(cat "$d/$n.s") s (on $PL_NODE)"
  else
    cp "$d/$n.s" "$RUN/steps/$n.fail"
    say "  $n: FAILED (rc $(cat "$d/$n.rc")) after $(cat "$d/$n.s") s on $PL_NODE; log $RUN/steps/$n.log"
    tail -6 "$RUN/steps/$n.log" | sed 's/^/      /'
  fi
}
# The host build, the image and publishing: one job on the lane host with no GPU, its workspace
# held for the boots.
do_host() {
  local rc=0 ws carry=() m2v
  # The metal2vulkan pin, carried to the lane host's mirror (M2V_REPO, the local checkout, where
  # it has the commit): a pin the public repository has no tag of yet builds there too.
  m2v=$(cat "$RUN/m2v" 2> /dev/null || true)
  local mr=${M2V_REPO:-$HOME/Projects/steelbrain/metal2vulkan}
  [[ $m2v =~ ^[0-9a-f]{40}$ ]] && git -C "$mr" cat-file -e "$m2v^{commit}" 2> /dev/null && carry=(--carry "$mr@$m2v")
  pl_run host --timeout "${HOST_TIMEOUT:-30m}" --ram 40 --priority "$PL_PRIO" --on "node=$PL_NODE" --hold 8h --label lane-gate=build ${carry[@]+"${carry[@]}"} \
    --keep lg-out/steps --keep lg-out/msi-build.txt -- bash lg/bin/msi-job.sh build || rc=$?
  ws=$(pl_workspace "$(cat "$RUN/job-host")") || { echo "no workspace for job $(cat "$RUN/job-host")"; return 1; }
  echo "$ws" > "$RUN/workspace"
  rm -rf "$RUN/host-steps"
  pl_kept host lg-out/steps "$RUN/host-steps" "$ws" || return 1
  pl_kept host lg-out/msi-build.txt "$RES/msi-build.txt" "$ws" 2> /dev/null || true
  return $rc
}

# --- the boot ---------------------------------------------------------------------------------

# ship_post: --post-probes on a --boot-only pass, whose build shipped none: a child of the ship
# commit with DIR in lg/post-probes, checked out in SHIP_WT, for the boot's --advance.
ship_post() {
  local gd idx parent tree commit
  [[ -n $post_probes && ! -d $SHIP_WT/lg/post-probes ]] || return 0
  gd=$(git -C "$SRC" rev-parse --path-format=absolute --git-common-dir)
  idx=$RUN/ship-post.index
  rm -f "$idx"
  parent=$(git -C "$SHIP_WT" rev-parse HEAD)
  GIT_INDEX_FILE=$idx git --git-dir="$gd" read-tree "$parent"
  (cd "$post_probes" && find . -type f | sed 's#^\./##' | LC_ALL=C sort) > "$RUN/post.files"
  [[ -s $RUN/post.files ]] || { echo "ship_post: no files in $post_probes"; return 1; }
  sed "s#^#$post_probes/#" "$RUN/post.files" | git --git-dir="$gd" hash-object -w --stdin-paths > "$RUN/post.blobs"
  while IFS= read -r f; do if [[ -x $post_probes/$f ]]; then echo 100755; else echo 100644; fi; done < "$RUN/post.files" > "$RUN/post.modes"
  sed 's#^#lg/post-probes/#' "$RUN/post.files" | paste -d' ' "$RUN/post.modes" "$RUN/post.blobs" - |
    awk '{ printf "%s %s\t%s\n", $1, $2, $3 }' | GIT_INDEX_FILE=$idx git --git-dir="$gd" update-index --add --index-info
  tree=$(GIT_INDEX_FILE=$idx git --git-dir="$gd" write-tree)
  commit=$(git --git-dir="$gd" commit-tree "$tree" -p "$parent" -m ":construction: WIP lane-gate ship $runid: post probes (throwaway)")
  rm -f "$idx"
  git -C "$SHIP_WT" reset -q --hard "$commit"
  echo "post probes: $(wc -l < "$RUN/post.files") files from $post_probes in $commit"
  advance=1
}

# do_boot boot|bench: the correctness boot or the bench boot, each a job in the host build's
# workspace holding the 5080 and the whole host (--exclusive). msi-launch.sh picks each boot's SSH
# port.
do_boot() {
  local mode=$1 oc=${image:-$MR/oc.qcow2} out=$MR/out-$1 excl=0 hold=${HOLD_SHARED_S:-1200}
  # Each job's limit near its own time (about 1.5 times): potluck reserves an exclusive job's start
  # at the latest limit among the jobs running on its machine.
  local ph=${phases:-info sweep decode icb webgl present final} pres=5 ws tmo=40m
  if [[ $mode == bench ]]; then
    # WebGL's frame rate and present_probe's answers first: the quickest read of a tip's frame pace.
    excl=1 hold=$HOLD_S ph=${bench_phases:-info webgl present idle bench decodefps final} pres=10 tmo=30m
  fi
  # --post-probes: their budgets, each with its start's settle (60 s at most) and a margin.
  local post_s=0
  if [[ $mode == boot && -n $post_probes ]]; then
    post_s=$(cat "$post_probes"/*/budget 2> /dev/null | awk '{s += $1 + 90} END {print s + 120}')
    hold=$(( hold + post_s ))
    tmo=$(( 40 + post_s / 60 ))m
  fi
  ws=$(cat "$RUN/workspace")
  cat > "$RUN/boot-$mode.env" << EOF
QEMU=${host_dir:-$MR}/qemu-system-x86_64
ROM=${host_dir:-$MR}/ReimsGraphics.rom
OC=$oc
DISK=$MSI_DISK
TAG=$([[ $mode == boot && -n $boot_tag ]] && echo "$boot_tag" || echo "gate-$runid-$mode")
PAYLOAD=$MR/payload
CORPUS=$MSI_GATE/corpus
GUEST_USER=$GUEST_USER
GUEST_PASS=$GUEST_PASS
RGK_RUN=$MSI_RGK_RUN
EXCLUSIVE=$excl
HOLD_S=$hold
LOCK_WAIT_S=${LOCK_WAIT_S:-3600}
PHASES="$ph"
PRESENT_S=$pres
LOAD=$([[ $mode == bench ]] && echo "$load")
# The bench's host log is info: the work counts' logger (reims_vgpu::work at debug) slows what it
# rates, and runs only where --bench-host-log asks for it.
HOST_LOG=$([[ $mode == boot ]] && echo "${host_log:-info}" || echo "${bench_host_log:-info}")
CAPTURE=$([[ $mode == boot ]] && echo "$capture" || echo 0)
APPLOAD=$([[ $mode == boot ]] && echo "$appload" || echo 0)
FINAL_CMD_B64=$([[ $mode == boot && -n ${final_cmd:-} ]] && printf '%s' "$final_cmd" | base64 | tr -d '\n')
QEMU_HOOK_B64=$([[ $mode == bench && -n $qemu_hook ]] && printf '%s' "$qemu_hook" | base64 | tr -d '\n')
POST_PROBES=$([[ $mode == boot && -n $post_probes ]] && echo "$MR/post-probes")
POST_PROBE_TIP=$sha
EOF
  set +e
  local xa=--exclusive
  [[ $mode == boot ]] && ((shared_boot)) && xa=
  pl_run "$mode" --timeout "$tmo" --ram 64 --priority "$PL_PRIO" ${xa:+"$xa"} --gpu "$PL_GPU" --workspace "$ws" ${advance:+--advance} \
    --label "lane-gate=$mode" --keep "lg-out/$mode" -- bash lg/bin/msi-job.sh boot "$mode" "$(base64 < "$RUN/boot-$mode.env" | tr -d '\n')"
  local rc=$?
  set -e
  rm -rf "${RES:?}/$mode"
  pl_kept "$mode" "lg-out/$mode" "$RES/$mode" "$ws" || mkdir -p "$RES/$mode"
  [[ ! -s $RES/$mode/rc ]] || rc=$(cat "$RES/$mode/rc")
  rm -f "$RES/$mode/rc"
  grep -E 'BOOT_FAIL|watchdog' "$RES/$mode/boot.log" || true
  # A sweep that ran whole leaves its case order for the next boot's.
  local order
  order=$(sed -n 's/^[0-9.]* GATE sweep-order //p' "$RES/$mode/guest.log" 2> /dev/null | head -1)
  [[ -z $order ]] || printf '%s\n' $order > "$GH/sweep-cases.txt"
  return $rc
}

# --- the run ----------------------------------------------------------------------------------

# meta.json for report.py: what the run is, and which parts ran in any of its passes.
write_meta() {
  python3 -I - "$RUN/meta.json" "$sha" "$subject" "$runid" "${base_sha:-}" "$(sort -u "$RUN/ran" 2> /dev/null | tr '\n' ' ')" \
    "$image|$host_dir|$load" "$phases" "$(( $(date +%s) - $(cat "$RUN/started") ))" "$ART/tools.txt" "${MAC_KNOWN_FAILS:-}" \
    "${KNOWN_FAILS:-}" "$(cat "$ART/tip-codecs.txt" 2> /dev/null || echo unknown)" "$MSI" << 'EOF'
import json, sys
out, sha, subject, runid, base, ran, image, phases, wall, tools_txt, known_txt, known_items, codecs, host = sys.argv[1:]
def lines(p):
    try:
        return [l.strip() for l in open(p) if l.strip()]
    except OSError:
        return []
tools = dict(l.split(None, 1) for l in lines(tools_txt))
try:
    known_fails = json.load(open(known_items)) if known_items else []
except (OSError, ValueError) as e:
    sys.exit(f"lane-gate: the known-failures list {known_items}: {e}")
json.dump({"sha": sha, "subject": subject, "runid": runid, "baseline": base, "ran": ran.split(),
           "image": image.split("|")[0], "host_build": image.split("|")[1], "load": image.split("|")[2],
           "phases": phases, "wall_s": int(wall), "tools": tools,
           "known_cargo_fails": lines(known_txt),
           "known_fails": known_fails, "lane_host": host,
           "tip_codecs": None if codecs == "unknown" else codecs.split()}, open(out, "w"), indent=1)
EOF
}
# report: the verdict as its status (0 pass, 1 fail, 2 incomplete). It leaves errexit as it was:
# its callers take the status with errexit off.
report() {
  write_meta
  local v=0
  python3 -I "$L/report.py" "$RUN" ${BASE:+--baseline "$BASE"} --noise "$GH/noise.json" \
    --history "$GH/known-hits.tsv" || v=$?
  if ((record)); then
    python3 -I "$L/report.py" --record "$RUN" "$GH/baselines"
    python3 -I "$L/report.py" --noise-from "$GH/baselines" "$GH/noise.json"
  fi
  return $v
}

# The bench leg alone: what a detached leg runs, after its run's verdict, in the run's workspace.
if [[ $bench == only ]]; then
  say "lane-gate bench leg: ${sha:0:12}, run $runid"
  [[ -s $RUN/workspace && -d $SHIP_WT ]] || die "run $runid has no workspace or no $SHIP_WT: the bench leg cannot run"
  rm -f "$RUN/steps/bench.ok" "$RUN/steps/bench.fail"
  echo bench >> "$RUN/ran"
  step bench do_boot bench || true
  set +e
  report > "$RUN/report.out"
  set -e
  head -1 "$RUN/report.txt"
  if ((landed)); then
    if [[ -f $RUN/steps/bench.ok ]]; then
      ln -sfn "$sha" "$GH/baselines/LAST"; say "baselines/LAST is now ${sha:0:12}, the car that landed"
    else
      say "baselines/LAST stays ${base_sha:0:12}: the landed car's bench boot failed"
    fi
  fi
  pl_cleanup
  say "lane-gate bench leg: report $RUN/report.txt; $(( $(date +%s) - T0 )) s"
  exit 0
fi

# The correctness boot alone, of a run built before with --keep-workspace (--resume --boot-only).
if ((boot_only)); then
  say "lane-gate boot leg: ${sha:0:12}, run $runid"
  [[ -s $RUN/workspace && -d $SHIP_WT ]] || die "run $runid has no workspace or no $SHIP_WT: its boot cannot run"
  [[ -f $RUN/steps/msi-build.ok && ( -n $image || -f $RUN/steps/image.ok ) ]] || die "run $runid has no host build or image"
  rm -f "$RUN/steps/boot.ok" "$RUN/steps/boot.fail"
  grep -q -x boot "$RUN/ran" 2> /dev/null || echo boot >> "$RUN/ran"
  ship_post || die "the post probes could not be added to the ship commit"
  step boot do_boot boot || true
  set +e
  report
  verdict=$?
  set -e
  ((keep_ws)) || pl_cleanup
  say "lane-gate boot leg: report $RUN/report.txt; $(( $(date +%s) - T0 )) s"
  exit $verdict
fi

say "lane-gate: ${sha:0:12} ($subject), run $runid${BASE:+, baseline ${base_sha:0:12}}"
# The Mac's checks take the Mac's GPU lock, which potluck's lease replaced: until they run as
# potluck jobs, a gate leaves them out.
skipped mac || die "the Mac's checks do not run as potluck jobs yet: run the gate with --skip mac"
step tree do_tree || die "the tree could not be checked out"

gb=0 set_ok=0
step guest-build do_guest_build || gb=1
if ((gb == 0)) && [[ -z $image ]]; then step set do_set && set_ok=1 || true; fi
msi_ok=0 img_ok=0
[[ -z $image ]] || img_ok=1
if ((gb == 0)) && { [[ -n $image ]] || ((set_ok)); }; then
  if step ship do_ship; then
    step host do_host || true
    for n in corpus stage msi-build image publish; do host_step "$n"; done
    [[ -f $RUN/steps/stage.ok && -f $RUN/steps/msi-build.ok ]] && msi_ok=1
    [[ -n $image || -f $RUN/steps/image.ok ]] && img_ok=1 || img_ok=0
  fi
else
  say "  ship and host: not run (the guest build or the set failed)"
fi
if ((publish)); then
  if [[ -f $RUN/steps/publish.ok ]]; then
    grep -E '^(PUBLISHED|NOT_PUBLISHED)' "$RUN/steps/publish.log" | sed 's/^/    /' || true
    ((probe_tip)) || echo "$sha" > "$GH/published-last"
  elif [[ -s $RUN/not-published ]]; then
    sed 's/^/    /' "$RUN/not-published"
    echo 0 > "$RUN/steps/publish.ok"; cp "$RUN/not-published" "$RUN/steps/publish.log"
  else
    say "  publish: not run (the host build, the image or the guest build failed, or --image)"
  fi
fi
if ! skipped boot; then
  if ((msi_ok && img_ok && gb == 0)); then
    echo boot >> "$RUN/ran"
    if ! step boot do_boot boot && grep -q -E 'BOOT_QUEUE_TIMEOUT|BOOT_FAIL missing' "$RES/boot/boot.log" 2> /dev/null; then
      # The candidate did not run (a file the boot needs is missing): the boot is not run, and a
      # --resume tries it again.
      say "  boot: not run ($(grep -m1 -o -E 'BOOT_QUEUE_TIMEOUT|BOOT_FAIL missing [^ ]*' "$RES/boot/boot.log"); --resume $runid)"
      rm -f "$RUN/steps/boot.fail"
      grep -v -x boot "$RUN/ran" > "$RUN/ran.new" || true
      mv "$RUN/ran.new" "$RUN/ran"
    fi
  else
    say "  boot: not run (the host build, the image or the guest build failed)"
  fi
fi

set +e
report
verdict=$?
set -e
if ((record && !landed && verdict == 0)); then ln -sfn "$sha" "$GH/baselines/LAST"; say "baselines/LAST is now ${sha:0:12}"; fi
if ((landed)) && { [[ $bench == no || $msi_ok != 1 || $img_ok != 1 || $gb != 0 ]] || skipped bench; }; then
  say "baselines/LAST stays ${base_sha:0:12}: the landed car has no bench leg to judge the next one against"
fi
say "lane-gate: report $RUN/report.txt; $(( $(date +%s) - T0 )) s this pass"

# The bench leg: after the verdict, the boot that holds the GPU and the host alone, in the run's
# workspace. Detached, it holds none of this gate's descriptors (lockf's among them), so the next
# gate can start while it waits its turn in potluck's queue.
if [[ $bench != no && $msi_ok == 1 && $img_ok == 1 && $gb == 0 ]] && ! skipped bench; then
  again=(--resume "$runid" --bench-only)
  if ((landed)); then again+=(--landed); elif ((record)); then again+=(--record); fi
  [[ -z $qemu_hook ]] || again+=(--qemu-hook "$qemu_hook")
  [[ -z $bench_host_log ]] || again+=(--bench-host-log "$bench_host_log")
  [[ -z $image ]] || again+=(--image "$image")
  [[ -z $host_dir ]] || again+=(--host "$host_dir")
  [[ -z $load ]] || again+=(--load "$load")
  [[ -z $bench_phases ]] || again+=(--bench-phases "$bench_phases")
  [[ -z $base_sha ]] || again+=(--baseline "$base_sha")
  if [[ $bench == wait ]]; then
    "$0" "${again[@]}" "$sha" | tee "$RUN/bench.out"
  else
    ( for f in /dev/fd/*; do n=${f##*/}; ((n > 2)) && eval "exec $n>&-" 2> /dev/null; done
      exec nohup "$0" "${again[@]}" "$sha" > "$RUN/bench.out" 2>&1 < /dev/null ) &
    say "lane-gate: the bench leg runs on detached; its output: $RUN/bench.out"
  fi
elif ((!keep_ws)); then
  pl_cleanup
fi
exit $verdict
