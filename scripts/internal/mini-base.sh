#!/bin/bash
# The Windows test machine's base workspace (homelab-mac-mini-2018): a kept build of development
# there, which a job's new workspace copies before it builds, so that the job builds only what its
# commit changed. In MSYS2's bash, in the job's workspace:
#
#   scripts/internal/mini-base.sh seed [--nv]   before the job's build
#   scripts/internal/mini-base.sh publish       in the base's refresh job, once its builds passed
#   scripts/internal/mini-base.sh show          the base a job would copy
#   scripts/internal/mini-base.sh failed        after a build that failed: says it was seeded
#
# The base is a potluck workspace of this repository that a refresh job built at development,
# held, and named in P:\rvg-minibase\current (`publish`, which also copies this script to
# P:\rvg-minibase\bin for trees without it). Each refresh builds in a new workspace and publishes
# it last, and nothing builds in a published base again: the refresh job is the base's one writer,
# a job only reads the base while it copies it and builds in its own copy, and potluck's leases are
# the only lock. The harness that dispatches the refreshes keeps the current and the previous base
# held.
#
# seed copies the base's target/, and with --nv the trees the harness's install-ctl.sh jobs keep
# under nv/ (cache/reims-ctl/ctl-target, cargo, qemu-clone), with their files' timestamps. Cargo
# reuses a unit by its fingerprint, which names no workspace member by an absolute path, and by
# mtimes, so seed then makes the copy right for this commit:
# - every tracked file and directory unchanged since the base's commit gets one old mtime, and
#   every changed, added or removed path (and the directories above it) now: cargo rebuilds what
#   changed and nothing else;
# - a package with a unit whose fingerprint names the base's workspace loses all its units in that
#   tree (reims-ctl's build script names guest/prebuilt by its absolute path: kept, it would read
#   the base's kexts), as does each package of this tree whose source bakes in a path cargo sets
#   for rustc and does not track (env! of CARGO_MANIFEST_DIR, CARGO_MANIFEST_PATH, CARGO_BIN_EXE_*,
#   CARGO_TARGET_TMPDIR, CARGO_RUSTC_CURRENT_DIR, or OUT_DIR outside an include: xtask's
#   repo_root() would name the base's tree);
# - a kept build script's output that names the base's workspace (sdl3-sys's link path) names
#   this one instead, its mtime kept, as cargo itself rewrites the base's OUT_DIR there: the build
#   reads only its own copy, and keeps building once the base is gone.
# With --nv, where nothing QEMU's build reads changed since the base (install-ctl.sh's kept_qemu
# paths), seed also copies the base's installed QEMU into nv/data/reims-ctl/qemu and writes
# nv/minibase.skip-qemu, for the job to run install-ctl.sh --skip-qemu (MINIBASE_QEMU=0: never).
# QEMU's build directory is not copied: its build.ninja names the source directory by its absolute
# path on every compile (vendor/qemu/meson.build's -iquote), so at another path ninja rebuilds it
# all.
#
# seed does nothing where MINIBASE=0, where target/ already holds files (an advanced workspace
# builds in place), where no base is published, where this tree does not hold the base's commit,
# or where the base built through another compiler wrapper (kache) than this job, and says so; it
# prints what it copied, dropped and rewrote, and writes it to target/minibase.seeded. failed, which a job runs after its build or tests failed, prints where
# the workspace's target/ came from (nothing, where it was not seeded) and that MINIBASE=0 builds
# it again without the seed, so a lane can tell a seed's fault from a compile error.
set -u

dir=${MINIBASE_DIR:-/p/rvg-minibase}
old=200001010000 # the mtime of what did not change, older than any build

say() { echo "mini-base: $*"; }
field() { sed -n "s/^$1=//p" "$2" 2>/dev/null | head -1; }
wrapper() {
  local w=${RUSTC_WRAPPER:-}
  w=${w##*[\\/]}
  w=${w%.exe}
  echo "${w:-none}"
}

# copy SRC DST: SRC's tree into DST, with its files' timestamps.
copy() {
  mkdir -p "$2" || return 1
  if [ "${MINIBASE_COPY:-robocopy}" = cp ]; then
    cp -a "$1/." "$2/"
    return
  fi
  # robocopy's exit codes below 8 are successes. Its switches are its own, not paths.
  MSYS2_ARG_CONV_EXCL='*' robocopy "$(cygpath -w "$1")" "$(cygpath -w "$2")" \
    /E /COPY:DAT /DCOPY:T /XJ /MT:16 /R:2 /W:1 /NFL /NDL /NJH /NJS /NP > /dev/null
  [ $? -lt 8 ]
}

# The package directory names of this tree whose sources bake in a path cargo does not track.
baking_packages() {
  local f d
  git grep -lE '(option_)?env!\(\s*"(CARGO_MANIFEST_DIR|CARGO_MANIFEST_PATH|CARGO_TARGET_TMPDIR|CARGO_RUSTC_CURRENT_DIR|OUT_DIR|CARGO_BIN_EXE_[^"]*)"' -- '*.rs' 2>/dev/null |
    while read -r f; do
      grep -E 'env!\(\s*"(CARGO_|OUT_DIR)' "$f" | grep -vqE 'include(_str|_bytes)?!\(concat!\(env!\("OUT_DIR"\)' || continue
      d=$(dirname "$f")
      while [ "$d" != . ] && ! grep -q '^\[package\]' "$d/Cargo.toml" 2>/dev/null; do d=$(dirname "$d"); done
      [ -f "$d/Cargo.toml" ] || continue
      awk '/^\[package\]/ { p = 1; next } /^\[/ { p = 0 } p && /^name *=/ { gsub(/^name *= *"|".*$/, ""); print; exit }' "$d/Cargo.toml"
    done | sort -u
}

seed() {
  local nv=0
  [ "${1:-}" = --nv ] && nv=1
  [ "${MINIBASE:-1}" = 0 ] && { say "MINIBASE=0: no seed, a fresh build"; return 0; }
  if [ -n "$(ls -A target 2>/dev/null)" ]; then
    say "target/ holds a build: it builds in place"
    return 0
  fi
  local cur=$dir/current base bname bc me start=$SECONDS
  [ -f "$cur" ] || { say "no base is published ($cur): a fresh build"; return 0; }
  base=$(field path "$cur") bname=$(field ws "$cur") bc=$(field commit "$cur") me=$(basename "$PWD")
  [ -n "$bname" ] && [ -f "$base/target/minibase.env" ] ||
    { say "the base ${bname:-?} is gone or not published: a fresh build"; return 0; }
  [ "$me" != "$bname" ] || { say "this workspace is the base"; return 0; }
  [ "$(field wrapper "$cur")" = "$(wrapper)" ] ||
    { say "the base built through $(field wrapper "$cur"), this job through $(wrapper): a fresh build"; return 0; }
  git cat-file -e "$bc^{commit}" 2>/dev/null ||
    { say "this tree does not hold the base's commit ${bc:0:10}: a fresh build"; return 0; }

  copy "$base/target" target || { say "copying the base's target/ failed: a fresh build"; rm -rf target; return 0; }
  rm -f target/minibase.env target/minibase.seeded
  local trees=(target) copied=target
  if [ $nv = 1 ]; then
    if [ -d "$base/nv/cache/reims-ctl/ctl-target" ] && copy "$base/nv/cache/reims-ctl/ctl-target" nv/cache/reims-ctl/ctl-target; then
      trees+=(nv/cache/reims-ctl/ctl-target) copied="$copied ctl-target"
    fi
    [ -d "$base/nv/cargo" ] && copy "$base/nv/cargo" nv/cargo && copied="$copied cargo"
    local q
    q=$(git ls-tree HEAD vendor/qemu | awk '{print $3}')
    if [ ! -f vendor/qemu/configure ] && [ ! -d nv/qemu-clone ] && [ -n "$q" ] &&
      git -C "$base/nv/qemu-clone" cat-file -e "$q^{commit}" 2>/dev/null; then
      copy "$base/nv/qemu-clone" nv/qemu-clone && copied="$copied qemu-clone"
    fi
  fi
  local copy_s=$((SECONDS - start))

  # Sources' mtimes: unchanged old, changed now.
  local changed
  git ls-files -z | xargs -0 touch -c -m -t $old
  find . \( -path ./.git -o -path ./target -o -path ./nv -o -path ./vendor/qemu \) -prune -o -type d -print0 |
    xargs -0 touch -c -m -t $old
  changed=$({ git diff --no-renames --name-only "$bc" HEAD; git diff --no-renames --name-only HEAD; } | sort -u)
  [ -n "$changed" ] && echo "$changed" | awk '{ print; while (sub(/\/[^\/]*$/, "")) print }' | sort -u |
    tr '\n' '\0' | xargs -0 touch -c -m

  # Units that name the base, or bake in a path: their packages go.
  local t drop=() f p
  for t in "${trees[@]}"; do
    while read -r p; do [ -n "$p" ] && drop+=("$p"); done < <(
      find "$t" -path '*/build/*/*/fingerprint/*' -type f -print0 | xargs -0 -r grep -laF "$bname" |
        sed -E 's#/[^/]+/fingerprint/[^/]+$##' | sort -u)
  done
  local baking
  baking=$(baking_packages)
  for p in $baking; do
    for t in "${trees[@]}"; do
      for f in "$t"/*/build/"$p" "$t"/*/*/build/"$p"; do [ -d "$f" ] && drop+=("$f"); done
    done
  done
  for f in ${drop[@]+"${drop[@]}"}; do
    case $f in target/* | nv/cache/reims-ctl/ctl-target/*) rm -rf "$f" ;; esac
  done

  # Kept build scripts' output: the base's workspace becomes this one.
  local rewritten=0
  for t in "${trees[@]}"; do
    while read -r f; do
      [ -n "$f" ] || continue
      cp -p "$f" "$f.minibase" && sed -i -e "s#${bname}\\\\#${me}\\\\#g" -e "s#${bname}/#${me}/#g" "$f" &&
        touch -r "$f.minibase" "$f" && rm -f "$f.minibase" && rewritten=$((rewritten + 1))
    done < <(find "$t" -path '*/build/*/*/run/*' -type f \( -name stdout -o -name root-output \) -print0 |
      xargs -0 -r grep -laF "$bname")
  done

  # QEMU, where nothing it is built from changed.
  local qemu=none
  if [ $nv = 1 ] && [ "${MINIBASE_QEMU:-1}" = 1 ] && [ -e "$base/nv/data/reims-ctl/qemu/current" ]; then
    local paths=(vendor/qemu crates firmware xtask Cargo.toml Cargo.lock rust-toolchain.toml .cargo ':!crates/reims-ctl')
    if git diff --quiet "$bc" HEAD -- "${paths[@]}" && git diff --quiet HEAD -- "${paths[@]}"; then
      local qd id
      qd=$(cd "$base/nv/data/reims-ctl/qemu/current" && pwd -P) id=$(basename "$qd")
      if copy "$qd" "nv/data/reims-ctl/qemu/$id"; then
        cmd //c mklink //J "$(cygpath -w nv/data/reims-ctl/qemu/current)" "$(cygpath -w "$PWD/nv/data/reims-ctl/qemu/$id")" > /dev/null &&
          echo "$id" > nv/minibase.skip-qemu && qemu=$id
      fi
    fi
  fi
  git update-index -q --refresh > /dev/null 2>&1

  local depth
  depth=$(($(field depth "$cur") + 1))
  {
    echo "from=$bname"
    echo "commit=$bc"
    echo "depth=$depth"
    echo "copied=$copied"
    echo "changed=$(echo -n "$changed" | grep -c .)"
    echo "dropped=$(for f in ${drop[@]+"${drop[@]}"}; do echo "${f#*/build/}"; done | sort -u | tr '\n' ' ')"
    echo "rewritten=$rewritten"
    echo "qemu=$qemu"
    echo "seconds=$((SECONDS - start)) (copy $copy_s)"
  } > target/minibase.seeded
  say "seeded from $bname at ${bc:0:10}: $(tr '\n' ';' < target/minibase.seeded)"
}

publish() {
  [ -d target ] || { say "no target/ to publish"; return 1; }
  local me depth=0
  me=$(basename "$PWD")
  [ -f target/minibase.seeded ] && depth=$(field depth target/minibase.seeded)
  {
    echo "ws=$me"
    echo "path=$PWD"
    echo "commit=$(git rev-parse HEAD)"
    echo "qemu=$(git ls-tree HEAD vendor/qemu | awk '{print $3}')"
    echo "m2v=$(grep -E '^metal2vulkan(-runtime)? *=' Cargo.toml | grep -oE 'rev = "[0-9a-f]{40}"' | head -1 | cut -d'"' -f2)"
    echo "wrapper=$(wrapper)"
    echo "depth=${depth:-0}"
    echo "nv=$([ -d nv/cargo ] && echo 1 || echo 0)"
    echo "made=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > target/minibase.env || return 1
  mkdir -p "$dir/bin" || return 1
  cp -f "${BASH_SOURCE[0]}" "$dir/bin/mini-base.sh.new" && mv -f "$dir/bin/mini-base.sh.new" "$dir/bin/mini-base.sh" ||
    say "could not copy this script to $dir/bin"
  if [ -f "$dir/current" ]; then
    cp -f "$dir/current" "$dir/previous.new" && mv -f "$dir/previous.new" "$dir/previous"
  fi
  cp -f target/minibase.env "$dir/current.new" && mv -f "$dir/current.new" "$dir/current" || return 1
  say "published $me: $(tr '\n' ' ' < target/minibase.env)"
}

failed() {
  local f=target/minibase.seeded
  [ -f "$f" ] || return 0
  say "this workspace's target/ was seeded from the base $(field from "$f") at $(field commit "$f" | cut -c1-10) ($f); MINIBASE=0 builds it again without the seed, to tell a seed's fault from a compile error"
}

main() {
  case ${1:-} in
    seed) shift; seed "$@" ;;
    publish) publish ;;
    show) cat "$dir/current" 2> /dev/null || say "no base is published" ;;
    failed) failed ;;
    *) echo "usage: mini-base.sh seed [--nv] | publish | show | failed" >&2; return 2 ;;
  esac
}
main "$@"
exit $?
