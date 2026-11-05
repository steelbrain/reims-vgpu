#!/bin/bash
# The lane gate's potluck jobs on the lane host. lane-gate.sh dispatches them from its shipped
# commit (lane-gate.sh's do_ship), whose lg/ holds:
#   lg/run.env    the run: RUNID, SHA (the candidate), QC (its vendor/qemu commit), SHORT, M2V,
#                 CDHASH, PUBLISH (1: publish the tip), IMAGE (1: build the image; 0 under --image),
#                 and the lane host's paths (MSI_GATE, MSI_RGK_RUN, MSI_BASE_OC, MSI_CORPUS_SRC,
#                 MSI_CORPUS_ODD, MSI_QEMU_SUBPROJECTS, MSI_QEMU_SEED)
#   lg/bin/       the gate's host scripts (msi-*.sh and the rest), as the gate has them
#   lg/post-probes/  the probes the correctness boot runs last (lane-gate.sh --post-probes), if any
#   lg/ship/      what the Mac built: payload/ (guest-run.sh and its tools), kexts/ (the loader's
#                 kexts), RgfxGate.kext, reims-graphics.bundle.tgz, and qemu.bundle (vendor/qemu's
#                 commits that MSI_QEMU_SEED lacks, a git bundle of refs/heads/lane-gate)
# The candidate's own tree is not in the commit: its parent is the candidate, which the job takes
# with `git archive` from the workspace's repository.
#
#   msi-job.sh build        the host build (msi-build.sh), the image (msi-image.sh) and, with
#                           PUBLISH, publishing (msi-publish.sh); no GPU. Each step's log, exit
#                           and seconds go into lg-out/steps/NAME.{log,rc,s}: ship, corpus, stage,
#                           msi-build, image, publish
#   msi-job.sh boot MODE ENV_B64   one boot (MODE boot or bench) by msi-boot.sh, its boot.env the
#                           base64 ENV_B64; run with --exclusive --gpu 5080. Its result files go
#                           into lg-out/MODE/, msi-boot.sh's exit into lg-out/MODE/rc
# The run's files on the host stay in MSI_GATE/runs/RUNID as before (MR), outside the workspace.
set -uo pipefail
WS=$(pwd)
[[ -f $WS/lg/run.env ]] || { echo "msi-job: no lg/run.env in $WS" >&2; exit 2; }
. "$WS/lg/run.env"
MR=$MSI_GATE/runs/$RUNID
OUT=$WS/lg-out
mkdir -p "$OUT/steps" "$MR"

# st NAME FUNCTION: runs FUNCTION with errexit, its output into lg-out/steps/NAME.log.
st() {
  local name=$1 t rc
  shift
  t=$(date +%s)
  ( set -e; "$@" ) > "$OUT/steps/$name.log" 2>&1
  rc=$?
  echo "$rc" > "$OUT/steps/$name.rc"
  echo $(( $(date +%s) - t )) > "$OUT/steps/$name.s"
  echo "$(date +%T) $name: exit $rc in $(cat "$OUT/steps/$name.s") s"
  return $rc
}

do_ship() {  # the gate's scripts and the Mac's build into the run's directory
  # The host keeps the newest five runs' files (each holds a QEMU binary and an image), and any
  # run a boot used in the last two hours.
  ls -1dt "$MSI_GATE/runs"/*/ | tail -n +6 | while read -r d; do
    [[ $d == "$MR/" || -n $(find "$d" -mmin -120 -print -quit) ]] || rm -rf "$d"; done
  rm -rf "$MR/bin" "$MR/payload" "$MR/kexts" "$MR/RgfxGate.kext" "$MR/post-probes"
  cp -a "$WS/lg/bin" "$MR/bin"
  [[ ! -d $WS/lg/post-probes ]] || cp -a "$WS/lg/post-probes" "$MR/post-probes"  # lane-gate.sh --post-probes
  cp -a "$WS/lg/ship/payload" "$MR/payload"
  [[ ! -d $WS/lg/ship/kexts ]] || cp -a "$WS/lg/ship/kexts" "$MR/kexts"
  [[ ! -d $WS/lg/ship/RgfxGate.kext ]] || cp -a "$WS/lg/ship/RgfxGate.kext" "$MR/RgfxGate.kext"
  cp -f "$WS/lg/ship/reims-graphics.bundle.tgz" "$MR/"
  # The app load's tools as rgk-ctx-slots has them now, copied (msi-appload.sh), and their hash.
  local a=${APPLOAD_TOOLS:-$HOME/.cache/rgk-ctx-slots-msi/appload/tools}
  if [[ -f $a/analyze-load.py ]]; then
    cp -r --reflink=auto "$a" "$MR/bin/appload-tools"
    (cd "$MR/bin/appload-tools" && find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | cut -c1-16) > "$MR/bin/appload-tools.hash"
  fi
  ls -la "$MR"
}
do_corpus() {  # the decode streams and references, copied again wherever their source changed
  local dst=$MSI_GATE/corpus src=$MSI_CORPUS_SRC oddsrc=${MSI_CORPUS_ODD:-} s from f g
  local names
  names=$(cat "$MR/payload/h264.list" "$MR/payload/hevc.list" | awk '!/^#/ && NF {print $1}'
    cat "$MR/payload/h264-odd.list" "$MR/payload/hevc-odd.list" | awk '!/^#/ && NF {print "odd/" $1}')
  mkdir -p "$dst/odd"
  for s in $names; do
    from=$src/$s
    [[ $s == odd/* ]] && from=$oddsrc/${s#odd/}
    for f in "$s" "${s%.*}.md5"; do
      g=$(dirname "$from")/$(basename "$f")
      cmp -s "$g" "$dst/$f" || { cp --reflink=auto "$g" "$dst/$f" && echo "copied $f"; }
    done
  done
  cd "$dst" && find . -type f ! -name MANIFEST | sort | xargs sha256sum > MANIFEST &&
    echo "corpus $(sha256sum < MANIFEST | cut -c1-16), $(find . -type f | wc -l) files"
}
do_stage() {  # the candidate's tree, and vendor/qemu's source where the host has none yet
  rm -rf "$MSI_GATE/stage" && mkdir -p "$MSI_GATE/stage"
  git -C "$WS" archive --format=tar "$SHA" -- . ':(exclude)tests/suite/expected' | tar -C "$MSI_GATE/stage" -xmf -
  local Q=$MSI_GATE/qemu/$QC G=$MSI_GATE/qemu-git
  if [[ ! -f $Q/src/configure ]]; then
    # A bare clone sharing MSI_QEMU_SEED's objects, with the commits it lacks from the bundle.
    [[ -d $G ]] || git clone -q --bare --shared "$MSI_QEMU_SEED" "$G"
    [[ ! -f $WS/lg/ship/qemu.bundle ]] || git -C "$G" fetch -q "$WS/lg/ship/qemu.bundle" "+refs/heads/lane-gate:refs/lane-gate/$QC"
    git -C "$G" cat-file -e "$QC^{commit}" || { echo "vendor/qemu's $QC is not in $G"; return 1; }
    rm -rf "$Q/src" && mkdir -p "$Q/src"
    git -C "$G" archive --format=tar "$QC" | tar -C "$Q/src" -xmf -
    echo "QEMU source for $QC from $G"
  fi
  test -f "$Q/src/configure"
}
do_msi_build() {
  # The metal2vulkan pin from potluck's mirror (lane-gate.sh carries it) into cargo's git
  # databases, so cargo finds the commit without the public repository. The mirror is only read.
  local db
  if [[ -n ${POTLUCK_MIRROR_metal2vulkan:-} && $M2V =~ ^[0-9a-f]{40}$ ]]; then
    for db in "$HOME"/.cargo/git/db/metal2vulkan-*; do
      [[ -d $db ]] || continue
      git -C "$db" fetch -q "$POTLUCK_MIRROR_metal2vulkan" "$M2V:refs/lane-gate/m2v" &&
        git -C "$db" cat-file -e "$M2V^{commit}" && echo "metal2vulkan $M2V from the mirror into $db"
    done
  fi
  bash "$MR/bin/msi-build.sh" "$MSI_GATE" "$MSI_GATE/tree" "$MSI_GATE/stage" "$QC" "$MSI_QEMU_SUBPROJECTS" | tee "$OUT/msi-build.txt"
  grep -q MSI_BUILD_OK "$OUT/msi-build.txt"
  local q r
  q=$(awk '$1 == "BUILT" && $2 == "qemu" {print $3}' "$OUT/msi-build.txt")
  r=$(awk '$1 == "BUILT" && $2 == "rom" {print $3}' "$OUT/msi-build.txt")
  # QEMU finds its firmware and option ROMs (kvmvapic.bin, efi-virtio.rom) in qemu-bundle/ beside
  # its binary: the build's bundle holds links into the commit's source tree, which stays.
  cp --reflink=auto "$q" "$MR/qemu-system-x86_64" && cp "$r" "$MR/ReimsGraphics.rom" &&
    rm -rf "$MR/qemu-bundle" && mkdir -p "$MR/qemu-bundle/usr/local/share" &&
    cp -a "$(dirname "$q")/qemu-bundle/usr/local/share/qemu" "$MR/qemu-bundle/usr/local/share/" &&
    test -e "$MR/qemu-bundle/usr/local/share/qemu/efi-virtio.rom"
}
do_image() {
  bash "$MR/bin/msi-image.sh" "$MR/bin/opencore-add-kexts.sh" "$MSI_BASE_OC" "$MR/oc.qcow2" "$MR/kexts" "$MR/RgfxGate.kext"
}
do_publish() {
  bash "$MR/bin/msi-publish.sh" "$MR" "$MSI_GATE/qemu/$QC/build" "$MSI_GATE/target/dist-vulkan" "$SHORT" "$QC" "$M2V" "$MSI_RGK_RUN" "$CDHASH"
}

case ${1:-} in
build)
  rc=0
  st ship do_ship || exit 1
  st corpus do_corpus || true
  st stage do_stage && st msi-build do_msi_build || rc=1
  if [[ ${IMAGE:-1} == 1 ]]; then st image do_image || rc=1; fi
  if ((rc == 0)) && [[ ${PUBLISH:-0} == 1 ]]; then st publish do_publish || rc=1; fi
  exit $rc ;;
boot)
  mode=${2:?usage: msi-job.sh boot MODE ENV_B64} env=${3:?usage: msi-job.sh boot MODE ENV_B64}
  out=$MR/out-$mode
  rm -rf "$out" "$OUT/$mode" && mkdir -p "$out" "$OUT/$mode"
  printf '%s' "$env" | base64 -d > "$out/boot.env"
  # Post probes a boot-only pass added (lane-gate.sh --post-probes, the workspace advanced to them).
  [[ ! -d $WS/lg/post-probes ]] || { rm -rf "$MR/post-probes"; cp -a "$WS/lg/post-probes" "$MR/post-probes"; }
  bash "$MR/bin/msi-boot.sh" "$out"
  rc=$?
  for f in boot.log guest.log cpu.txt thermal.txt host-events.log host.txt host-video.tsv gpuload.log gpu-clocks.txt serial.log desktop.png drag.txt drag.log xid.txt \
    capture.txt kext-stalls.txt appload.txt appload.log kwin-reaper.txt \
    kernel-rgfx.log drag-resize-track.txt launch.log.gz post-probes; do
    [[ ! -e $out/$f ]] || cp -a "$out/$f" "$OUT/$mode/"
  done
  echo "$rc" > "$OUT/$mode/rc"
  echo "msi-boot.sh $mode: exit $rc; the boot's files in $out"
  exit $rc ;;
*) echo "usage: msi-job.sh build | boot MODE ENV_B64" >&2; exit 2 ;;
esac
