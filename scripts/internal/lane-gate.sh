#!/bin/bash
# The graphics lane's landing gate: checks one commit end to end and compares it with the last
# good baseline. Run it on the Mac, from any checkout of the repository that has this script:
#
#   scripts/internal/lane-gate.sh [--record] [--baseline SHA] [--resume [RUNID]] [--image PATH]
#                                 [--host DIR] [--load DUTY] [--skip mac|boot] [--phases "PHASE..."]
#                                 [--bench-phases "PHASE..."] [--no-bench | --wait-bench] [--publish] [REV]
#
#   REV           the commit to check (default HEAD); the gate checks commits, never a dirty tree
#   --record      store the run as a sample of REV's baseline (in baselines/SHA/runs), measure the
#                 noise band again from every baseline with two samples or more, and make SHA the
#                 last good baseline (baselines/LAST) when the run passed
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
#   --phases      the correctness boot's guest-run.sh phases (default: info sweep decode webgl
#                 present final)
#   --bench-phases  the bench boot's (default: info webgl present idle bench decodefps final)
#   --no-bench    leave out the bench leg; --wait-bench waits for it before exiting, where by
#                 default it runs on after the verdict, detached (its output in runs/RUNID/bench.out)
#   --publish     REV is the lane's tip: publish its image and host build on the lane host, read-only,
#                 as loader-fix/oc-sipon-SHORT.qcow2 and tip-host/SHORT/ under the host's RGK_RUN
#                 (lane-gate/msi-publish.sh), unless nothing they carry changed since the last
#                 commit the gate published (published-last)
#
# What it runs, the Mac's leg beside the lane host's build:
#   Mac:  the guest kexts, bundle and tools (lane-gate/guest: guest-run.sh's payload); then
#         `make -C guest/kext check` against the guest release's KDK and OpenCore's linker,
#         the kext's, bundle's and RgfxGate's tests, `cargo fmt --all --check`, and the cargo
#         tests the change reaches (the lane's always, plus every workspace package with a file
#         changed since the baseline and the packages that depend on it), the GPU's under the
#         Mac's GPU lock, compared with the baseline's known failures;
#   host: the candidate's library dist, option ROMs and QEMU (lane-gate/msi-build.sh), and its
#         SIP-on OpenCore image (the loader built for its ReimsGraphics, its kexts, and RgfxGate
#         carrying its bundle, lane-gate/msi-image.sh);
#   boot: the correctness boot, which decides: one cold boot on overlays (lane-gate/msi-boot.sh,
#         through the host's rgk-boot2.sh) sharing the GPU with other VMs, running the guest's
#         checks (lane-gate/guest/guest-run.sh): the lane's correctness sweep, the H.264 and HEVC
#         decode sets against their MD5s, Safari's WebGL readPixels, present_probe's answers
#         full-screen and windowed, and the desktop's state (SIP on, the kexts loaded and bound,
#         the bundle installed, the Metal device ours, WindowServer up throughout, no new crash
#         report, and the guest's capture of its screen); and the host log's fatal lines and
#         display updates.
#   bench: the bench leg, which reports and never decides: a second cold boot holding the GPU
#         alone (RGK_EXCLUSIVE) for at most HOLD_S seconds, running the GPU benchmark, the
#         on-screen benchmark, present_probe and the timed decodes, with QEMU's CPU and the host's
#         display updates, each number against the baseline's with its noise band.
# Then lane-gate/report.py writes the run's report.txt and results.json: PASS, FAIL or SKIP for
# every item (KNOWN for a failure the known-failures list, KNOWN_FAILS, matches exactly, with its
# hits across runs in known-hits.tsv), and each number against the baseline. Exit 0 when the run passed, 1 when a check
# failed, 2 when a part did not run; the bench leg's failures and regressions show in the report
# and never change the exit.
#
# Its state is under LANE_GATE_HOME (default ~/.cache/rgk-lane-gate): config.sh (this host's
# paths: the KDK, the loader builder, the lane host's; config.sh's comments say what each is),
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
if [[ -z ${LANE_GATE_LOCKED:-} && " $* " != *" --bench-only "* ]]; then
  export LANE_GATE_LOCKED=1
  exec lockf -t 0 "$GH/lock" "$0" "$@"
fi
[[ -f $GH/config.sh ]] || die "no $GH/config.sh: it names this host's KDK, loader builder and lane host"
. "$GH/config.sh"

record=0 base_sha= resume= image= host_dir= load= skip= phases= bench_phases= rev=HEAD bench=detach publish=0
while (($#)); do
  case $1 in
    --record) record=1 ;;
    --baseline) base_sha=$2; shift ;;
    --resume) if [[ ${2:-} =~ ^[0-9a-f]{12}- ]]; then resume=$2; shift; else resume=newest; fi ;;
    --image) image=$2; shift ;;
    --host) host_dir=$2; shift ;;
    --load) load=$2; shift ;;
    --bench-phases) bench_phases=$2; shift ;;
    --skip) skip="$skip $2"; shift ;;
    --phases) phases=$2; shift ;;
    --no-bench) bench=no ;;
    --wait-bench) bench=wait ;;
    --bench-only) bench=only ;;  # the bench leg of a resumed run alone (what the detached leg runs)
    --publish) publish=1 ;;
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
msi() { ssh -o BatchMode=yes "$MSI" "$@"; }
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
  cp "$L/guest/"* "$P/"
  # RgfxGate's payload is the bundle it carries; its Makefile does not see the bundle change.
  make -C guest/rgfxgate clean
  make -C guest/rgfxgate DEPS="$RGFXGATE_DEPS" BUNDLE_DIR="$TREE/guest/bundle/build/package/reims-graphics.bundle"
  cp -R guest/kext/build/ReimsGraphics.kext guest/rgfxgate/build/RgfxGate.kext "$ART/"
  cp guest/bundle/build/reims-graphics.bundle.tgz "$ART/"
  # The codecs the candidate's device offers the guest where the host decodes them: H.264 where
  # its bundle has the H.264 driver, HEVC where its kext states IOGVAHEVCDecode.
  { grep -rqs RGFX_VIDEO_CODEC_H264 guest/bundle/src && echo h264
    grep -rqs IOGVAHEVCDecode guest/kext/src && echo hevc
    true; } > "$ART/tip-codecs.txt"
  { for f in guest/bundle/tools/sweep/*.[mh] guest/bundle/tools/bench/bench.m; do shasum "$f"; done | shasum | echo "sweep+bench $(cut -c1-16)"
    echo "present_probe $(shasum < "$(tool_src guest/bundle/tools/present_probe.m)" | cut -c1-16)"
    echo "decode_md5 $(shasum < "$(tool_src guest/bundle/tools/decode_md5.m)" | cut -c1-16)"
    echo "drag $(cat "$(tool_src guest/bundle/tools/winmon.m)" "$L/msi-drag.py" | shasum | cut -c1-16)"
    echo "bundle $(shasum < guest/bundle/build/reims-graphics.bundle | cut -c1-16)"
    echo "kext $(shasum < guest/kext/build/ReimsGraphics.kext/Contents/MacOS/ReimsGraphics | cut -c1-16)"
  } > "$ART/tools.txt"
  cat "$ART/tools.txt"
}

# --- the Mac's checks -------------------------------------------------------------------------

do_kext_check() {
  [[ -n $KDK && -n $KEXTINJECT && -n $BOOT_KC ]]
  make -C "$TREE/guest/kext" check KDK="$KDK" KEXTINJECT="$KEXTINJECT" BOOT_KC="$BOOT_KC" | tee "$RES/kext-check.txt"
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
  local more expr pkgs=(-p reims-graphics -p reims-vgpu)
  more=$(reached)
  expr="package(reims-graphics) | (package(reims-vgpu) & test(/graphics/))"
  for p in $more; do
    [[ $p == reims-ctl ]] && continue
    pkgs+=(-p "$p")
    expr="$expr | package($p)"
  done
  echo "packages: ${pkgs[*]}; filter: $expr"
  local feats=reims-vgpu-core/testkit,reims-vgpu/testkit,reims-vgpu/vulkan
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
  python3 -I - "$junit" "$RES" << 'EOF'
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

# --- the lane host ----------------------------------------------------------------------------

do_ship() {  # the gate's own scripts, which the host's steps run
  # The host keeps the newest five runs' files (each holds a QEMU binary and an image), and any
  # run a boot used in the last two hours (a detached bench leg may still wait its turn).
  msi "mkdir -p '$MR/bin' && ls -1dt '$MSI_GATE/runs'/*/ | tail -n +6 | while read -r d; do
    [ -n \"\$(find \"\$d\" -mmin -120 -print -quit)\" ] || rm -rf \"\$d\"; done"
  # Each script replaces the old one by a rename, never in place: a detached bench leg may be
  # running the old one, and bash reads a script as it runs it.
  msi "rm -rf '$MR/bin/.incoming' && mkdir -p '$MR/bin/.incoming'"
  scp -q "$L/msi-build.sh" "$L/msi-image.sh" "$L/msi-boot.sh" "$L/msi-publish.sh" "$L/msi-drag.py" "$L/gpuload.c" "$L/gpuload.comp" \
    "$SRC/scripts/internal/opencore-add-kexts.sh" "$MSI:$MR/bin/.incoming/"
  msi "cd '$MR/bin/.incoming' && for f in *; do mv -f \"\$f\" ../; done && cd .. && rmdir .incoming"
}
do_corpus() {  # the decode streams and references, copied again wherever their source changed
  local names odd
  names=$(cat "$L/guest/h264.list" "$L/guest/hevc.list" | awk '!/^#/ && NF {print $1}')
  odd=$(cat "$L/guest/h264-odd.list" "$L/guest/hevc-odd.list" | awk '!/^#/ && NF {print "odd/" $1}')
  msi bash -s -- "$MSI_GATE/corpus" "$MSI_CORPUS_SRC" "${MSI_CORPUS_ODD:-}" $names $odd << 'EOF'
set -euo pipefail
dst=$1 src=$2 oddsrc=$3; shift 3
mkdir -p "$dst/odd"
for s in "$@"; do
  from=$src/$s
  [[ $s == odd/* ]] && from=$oddsrc/${s#odd/}
  for f in "$s" "${s%.*}.md5"; do
    g=$(dirname "$from")/$(basename "$f")
    cmp -s "$g" "$dst/$f" || { cp --reflink=auto "$g" "$dst/$f" && echo "copied $f"; }
  done
done
cd "$dst" && find . -type f ! -name MANIFEST | sort | xargs sha256sum > MANIFEST &&
  echo "corpus $(sha256sum < MANIFEST | cut -c1-16), $(find . -type f | wc -l) files"
EOF
}
do_stage() {
  msi "rm -rf '$MSI_GATE/stage' && mkdir -p '$MSI_GATE/stage'"
  git -C "$TREE" archive --format=tar "$sha" -- . ':(exclude)tests/suite/expected' |
    msi "tar -C '$MSI_GATE/stage' -xmf -"
  local qc
  qc=$(git -C "$TREE" ls-tree "$sha" vendor/qemu | awk '{print $3}')
  [[ -n $qc ]]
  echo "$qc" > "$ART/qemu-commit"
  if ! msi "test -f '$MSI_GATE/qemu/$qc/src/configure'"; then
    git -C "$QEMU_REPO" cat-file -e "$qc^{commit}" || { echo "vendor/qemu's $qc is not in $QEMU_REPO"; exit 1; }
    git -C "$QEMU_REPO" archive --format=tar "$qc" | msi "mkdir -p '$MSI_GATE/qemu/$qc/src' && tar -C '$MSI_GATE/qemu/$qc/src' -xmf -"
  fi
}
do_msi_build() {
  local qc
  qc=$(cat "$ART/qemu-commit")
  # Every build on the lane host waits for host quiet (build-quiet.sh): never beside a measured boot.
  msi "bash '$MSI_RGK_RUN/build-quiet.sh' -- bash '$MR/bin/msi-build.sh' '$MSI_GATE' '$MSI_GATE/tree' '$MSI_GATE/stage' '$qc' '$MSI_QEMU_SUBPROJECTS'" | tee "$RES/msi-build.txt"
  grep -q MSI_BUILD_OK "$RES/msi-build.txt"
  local q r
  q=$(awk '$1 == "BUILT" && $2 == "qemu" {print $3}' "$RES/msi-build.txt")
  r=$(awk '$1 == "BUILT" && $2 == "rom" {print $3}' "$RES/msi-build.txt")
  # QEMU finds its firmware and option ROMs (kvmvapic.bin, efi-virtio.rom) in qemu-bundle/ beside
  # its binary: the build's bundle holds links into the commit's source tree, which stays.
  msi "cp --reflink=auto '$q' '$MR/qemu-system-x86_64' && cp '$r' '$MR/ReimsGraphics.rom' &&
    rm -rf '$MR/qemu-bundle' && mkdir -p '$MR/qemu-bundle/usr/local/share' &&
    cp -a \"\$(dirname '$q')/qemu-bundle/usr/local/share/qemu\" '$MR/qemu-bundle/usr/local/share/' &&
    test -e '$MR/qemu-bundle/usr/local/share/qemu/efi-virtio.rom'"
}
do_loader() {
  local tag=gate-$runid
  msi "rm -rf '$MR/kexts' '$MSI_RGK_RUN/loader-fix/kexts-$tag' '$MSI_RGK_RUN/loader-fix/oc-$tag.qcow2'"
  rm -rf "$HOME/.cache/rgk-run-mac/loader-$tag"
  HOST=$MSI OCADD=${MR#/home/*/}/bin/opencore-add-kexts.sh bash "$LOADER_FOR" "$ART/ReimsGraphics.kext" "$tag"
  # The builder's own image (its non-SIP base) is not booted; its kexts go into the run.
  msi "mv '$MSI_RGK_RUN/loader-fix/kexts-$tag' '$MR/kexts' && rm -f '$MSI_RGK_RUN/loader-fix/oc-$tag.qcow2'"
  rm -rf "$HOME/.cache/rgk-run-mac/loader-$tag"
}
do_image() {
  msi "rm -rf '$MR/RgfxGate.kext'"
  scp -q -r "$ART/RgfxGate.kext" "$MSI:$MR/"
  msi "bash '$MR/bin/msi-image.sh' '$MR/bin/opencore-add-kexts.sh' '$MSI_BASE_OC' '$MR/oc.qcow2' '$MR/kexts' '$MR/RgfxGate.kext'"
}
msi_leg() {
  step stage do_stage && step msi-build do_msi_build
}
image_leg() {
  [[ -z $image ]] || return 0
  step loader do_loader && step image do_image
}

# The paths whose changes reach the image or the host build.
PUBLISHED_PATHS=(guest/kext guest/bundle guest/rgfxgate crates firmware vendor/qemu xtask Cargo.toml Cargo.lock)
do_publish() {
  local last short qc m2v
  last=$(cat "$GH/published-last" 2> /dev/null || true)
  if [[ -n $last ]] && git -C "$SRC" cat-file -e "$last^{commit}" 2> /dev/null &&
    git -C "$SRC" diff --quiet "$last" "$sha" -- "${PUBLISHED_PATHS[@]}"; then
    echo "NOT_PUBLISHED: nothing the image or the host carries changed since ${last:0:9}, whose they are"
    return 0
  fi
  short=${sha:0:9}
  qc=$(cat "$ART/qemu-commit")
  m2v=$(grep -m1 -o -E '^metal2vulkan-runtime = .*rev = "[0-9a-f]+"' "$TREE/Cargo.toml" | grep -o -E '[0-9a-f]{40}' || echo unknown)
  # The signed bundle RgfxGate carries, so that a fixer can build an RgfxGate of exactly the tip's.
  local cdhash
  cdhash=$(codesign -dvvv "$TREE/guest/bundle/build/package/reims-graphics.bundle" 2>&1 | sed -n 's/^CDHash=//p')
  scp -q "$ART/reims-graphics.bundle.tgz" "$MSI:$MR/reims-graphics.bundle.tgz"
  msi "bash '$MR/bin/msi-publish.sh' '$MR' '$MSI_GATE/qemu/$qc/build' '$MSI_GATE/target/dist-vulkan' '$short' '$qc' '$m2v' '$MSI_RGK_RUN' '$cdhash'"
  echo "$sha" > "$GH/published-last"
}

# --- the boot ---------------------------------------------------------------------------------

# do_boot boot|bench: the correctness boot, shared, or the bench boot, the GPU alone. rgk-boot2.sh
# picks each boot's SSH port.
do_boot() {
  local mode=$1 oc=${image:-$MR/oc.qcow2} out=$MR/out-$1 excl=0 hold=${HOLD_SHARED_S:-1200}
  local ph=${phases:-info sweep decode webgl present final} pres=5
  if [[ $mode == bench ]]; then
    # WebGL's frame rate and present_probe's answers first: the quickest read of a tip's frame pace.
    excl=1 hold=$HOLD_S ph=${bench_phases:-info webgl present idle bench decodefps final} pres=10
  fi
  # The guest runner and its files as the gate has them now (a resumed run keeps its tools).
  cp "$L/guest/"* "$ART/payload/"
  msi "rm -rf '$out' && mkdir -p '$out' && { [ -d '$MR/payload' ] || exit 0; rm -rf '$MR/payload'; }"
  scp -q -r "$ART/payload" "$MSI:$MR/payload"
  msi "cat > '$out/boot.env'" << EOF
QEMU=${host_dir:-$MR}/qemu-system-x86_64
ROM=${host_dir:-$MR}/ReimsGraphics.rom
OC=$oc
DISK=$MSI_DISK
TAG=gate-$runid-$mode
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
EOF
  set +e
  msi "bash '$MR/bin/msi-boot.sh' '$out'"
  local rc=$?
  set -e
  rm -rf "${RES:?}/$mode"
  mkdir -p "$RES/$mode"
  for f in boot.log guest.log cpu.txt host-events.log host.txt host-video.tsv gpuload.log serial.log desktop.png drag.txt drag.log; do
    scp -q "$MSI:$out/$f" "$RES/$mode/" 2> /dev/null || true
  done
  grep -E 'BOOT_FAIL|watchdog' "$RES/$mode/boot.log" || true
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

# The bench leg alone: what a detached leg runs, after its run's verdict.
if [[ $bench == only ]]; then
  say "lane-gate bench leg: ${sha:0:12}, run $runid"
  rm -f "$RUN/steps/bench.ok" "$RUN/steps/bench.fail"
  echo bench >> "$RUN/ran"
  step bench do_boot bench || true
  set +e
  report > "$RUN/report.out"
  set -e
  head -1 "$RUN/report.txt"
  say "lane-gate bench leg: report $RUN/report.txt; $(( $(date +%s) - T0 )) s"
  exit 0
fi

say "lane-gate: ${sha:0:12} ($subject), run $runid${BASE:+, baseline ${base_sha:0:12}}"
step tree do_tree || die "the tree could not be checked out"
rm -f "$RUN/steps/ship.ok"  # every pass ships the gate's scripts as they are now
step ship do_ship || die "the lane host's scripts could not be shipped"
step corpus do_corpus || true

msi_leg > "$RUN/msi-leg.out" 2>&1 &
p_msi=$!
gb=0
step guest-build do_guest_build || gb=1
p_img= p_mac=
if ((gb == 0)); then
  image_leg > "$RUN/image-leg.out" 2>&1 &
  p_img=$!
fi
if ! skipped mac; then
  echo mac >> "$RUN/ran"
  mac_checks > "$RUN/mac-leg.out" 2>&1 &
  p_mac=$!
fi
wait "$p_msi" && msi_ok=1 || msi_ok=0
cat "$RUN/msi-leg.out"
img_ok=0
[[ -z $image ]] || img_ok=1
if [[ -n $p_img ]]; then wait "$p_img" && img_ok=1 || img_ok=0; cat "$RUN/image-leg.out"; fi
if ((publish)); then
  if ((msi_ok && img_ok && gb == 0)) && [[ -z $image ]]; then
    step publish do_publish && grep -E '^(PUBLISHED|NOT_PUBLISHED)' "$RUN/steps/publish.log" | sed 's/^/    /' || true
  else
    say "  publish: not run (the host build, the image or the guest build failed, or --image)"
  fi
fi
if ! skipped boot; then
  if ((msi_ok && img_ok && gb == 0)); then
    echo boot >> "$RUN/ran"
    if ! step boot do_boot boot && grep -q -E 'BOOT_QUEUE_TIMEOUT|BOOT_FAIL missing' "$RES/boot/boot.log" 2> /dev/null; then
      # The candidate did not run (the host's queue did not reach the boot in time, or a file the
      # boot needs is missing): the boot is not run, and a --resume tries it again.
      say "  boot: not run ($(grep -m1 -o -E 'BOOT_QUEUE_TIMEOUT|BOOT_FAIL missing [^ ]*' "$RES/boot/boot.log"); --resume $runid)"
      rm -f "$RUN/steps/boot.fail"
      grep -v -x boot "$RUN/ran" > "$RUN/ran.new" || true
      mv "$RUN/ran.new" "$RUN/ran"
    fi
  else
    say "  boot: not run (the host build, the image or the guest build failed)"
  fi
fi
if [[ -n $p_mac ]]; then wait "$p_mac" || true; cat "$RUN/mac-leg.out"; fi

set +e
report
verdict=$?
set -e
if ((record && verdict == 0)); then ln -sfn "$sha" "$GH/baselines/LAST"; say "baselines/LAST is now ${sha:0:12}"; fi
say "lane-gate: report $RUN/report.txt; $(( $(date +%s) - T0 )) s this pass"

# The bench leg: after the verdict, the boot that holds the GPU alone. Detached, it holds none of
# this gate's descriptors (lockf's among them), so the next gate can start while it waits its turn.
if [[ $bench != no && $msi_ok == 1 && $img_ok == 1 && $gb == 0 ]] && ! skipped bench; then
  again=(--resume "$runid" --bench-only)
  ((record)) && again+=(--record)
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
fi
exit $verdict
