#!/bin/bash
# The library's tests on Windows, on the Windows test machine (homelab-mac-mini-2018: Windows 11,
# Intel UHD 630), as two potluck jobs dispatched from a clean checkout's HEAD: the tests' build,
# holding no GPU, then the tests in its workspace, holding the GPU (AGENTS.md, "Checking a change"
# and "GPU and VM discipline"). Run it from the Mac or another member, in the background: it
# returns when the tests end, with their exit status, and waits on the queue until then. Both jobs
# run the checkout's HEAD, and the tests' is queued once the build starts: keep the checkout clean
# until the script says the tests are queued (potluck refuses to dispatch from a dirty tree).
#
#   scripts/internal/windows-tests.sh [-p PACKAGE]... [-- NEXTEST-ARGS...]
#
#   -p PACKAGE     a crate whose tests run (default: the library's crates, reims-vgpu-graphics,
#                  reims-vgpu-vulkan, reims-vgpu-window, reims-vgpu and reims-vgpu-c), with the
#                  features of `cargo xtask check`'s tests that are that crate's
#   NEXTEST-ARGS   more arguments for the tests' `cargo nextest run`, as a filter (`-E EXPR`)
#
# Both jobs build for x86_64-pc-windows-gnullvm in MSYS2 CLANG64, whose clang the gnullvm target
# links with and SDL3's CMake build compiles with. They carry metal2vulkan at the commit
# Cargo.toml pins, from a checkout beside the main checkout (M2V_DIR names another), which the job
# fetches into cargo's git database.
#
# The build starts from the machine's base workspace, a build of development the harness keeps
# there: its new workspace first copies the base's target/ (scripts/internal/mini-base.sh seed,
# which says what it copied and dropped), and cargo then builds only what this commit changed
# since the base's. Without a base, or with MINIBASE=0, it builds from nothing. A seeded build or
# test job that fails says so in its last lines, naming the base, for a run again with MINIBASE=0
# to tell a seed's fault from a compile error.
#
# Environment: PRIORITY (the jobs' potluck priority, default 4), LANE (their `lane` label, default
# windows-tests), BUILD_RAM and TEST_RAM (GiB each job reserves, its commit charge's limit on
# Windows: default 12 and 26), OUT (where the jobs' logs go, default target/windows-tests),
# MINIBASE (0: the build does not start from the base). The workspace stays held for an hour after
# the tests, for a run again with `potluck dispatch --workspace`; the script prints it.
#
# TEST_RAM's default is 1.25 times the largest commit charge the tests' job was measured at: runs
# peaked at 14.7 to 18.1 GiB, then one at 20.7 GiB (potluck job m3-max-56017), and a job that
# reserved 20 was ended over memory at 20.1 GiB (m3-max-61085).
#
# In a job, on Windows, it runs as `windows-tests.sh --job build|test [-p PACKAGE]... [-- ARGS]`.
set -u

node=homelab-mac-mini-2018
gpu=8086:3e9b
target=x86_64-pc-windows-gnullvm
# `cargo xtask check`'s TESTED_FEATURES (xtask/src/main.rs).
tested_features="reims-vgpu-vulkan/testkit reims-vgpu/testkit reims-vgpu/vulkan reims-vgpu-c/vulkan"
default_packages="reims-vgpu-graphics reims-vgpu-vulkan reims-vgpu-window reims-vgpu reims-vgpu-c"

job=
if [ "${1:-}" = --job ]; then
  job=${2:?--job build|test}
  shift 2
fi
packages=()
while [ $# -gt 0 ]; do
  case $1 in
    -p) packages+=("${2:?-p PACKAGE}"); shift 2 ;;
    --) shift; break ;;
    *) echo "usage: scripts/internal/windows-tests.sh [-p PACKAGE]... [-- NEXTEST-ARGS...]" >&2; exit 2 ;;
  esac
done
[ ${#packages[@]} -gt 0 ] || read -r -a packages <<< "$default_packages"

# ---------------------------------------------------------------------------------------------
# In the job, on Windows.
if [ -n "$job" ]; then
  export PATH="/c/msys64/clang64/bin:$PATH" CMAKE_GENERATOR=Ninja
  . scripts/internal/dev-env.sh
  echo "== windows-tests $job on $(hostname) at $(git rev-parse --short HEAD), workspace $(basename "$PWD"), gpus ${POTLUCK_GPUS:-none}"
  rev=$(grep -E '^metal2vulkan(-runtime)? *=' Cargo.toml | grep -oE 'rev = "[0-9a-f]{40}"' | head -1 | cut -d'"' -f2)
  if [ -n "${POTLUCK_MIRROR_metal2vulkan:-}" ]; then
    for db in "$HOME"/.cargo/git/db/metal2vulkan-*; do
      [ -d "$db" ] && { git -C "$db" fetch -q "$POTLUCK_MIRROR_metal2vulkan" "$rev:refs/windows-tests/m2v" || true; }
    done
  fi
  args=()
  for p in "${packages[@]}"; do args+=(-p "$p"); done
  features=
  for f in $tested_features; do
    for p in "${packages[@]}"; do
      [ "${f%%/*}" = "$p" ] && features=${features:+$features,}$f
    done
  done
  [ -n "$features" ] && args+=(--features "$features")
  # The build's mark, which the tests' job needs: it holds the GPU, and builds nothing.
  built=target/windows-tests.built
  start=$(date +%s)
  case $job in
    build)
      bash scripts/internal/mini-base.sh seed
      rm -f "$built"
      cargo nextest run --target "$target" "${args[@]}" --no-run && git rev-parse HEAD > "$built"
      ;;
    test)
      [ -n "${POTLUCK_GPUS:-}" ] || { echo "windows-tests: the tests' job holds no GPU" >&2; exit 1; }
      [ "$(cat "$built" 2>/dev/null)" = "$(git rev-parse HEAD)" ] ||
        { echo "windows-tests: no build of this commit passed in this workspace" >&2; exit 1; }
      cargo nextest run --target "$target" "${args[@]}" --no-fail-fast "$@"
      ;;
    *) echo "windows-tests: --job build|test" >&2; exit 2 ;;
  esac
  rc=$?
  git checkout -- Cargo.lock
  [ $rc = 0 ] || bash scripts/internal/mini-base.sh failed
  echo "== windows-tests $job exit $rc in $(( $(date +%s) - start )) s"
  exit $rc
fi

# ---------------------------------------------------------------------------------------------
# The dispatch, from a clean checkout.
root=$(git rev-parse --show-toplevel) || exit 1
cd "$root" || exit 1
if [ -n "$(git status --porcelain)" ]; then
  echo "windows-tests: the jobs run HEAD, and potluck refuses a dirty tree (restore Cargo.lock: git checkout -- Cargo.lock):" >&2
  git status --short >&2
  exit 1
fi
head=$(git rev-parse --short HEAD)
out=${OUT:-$root/target/windows-tests}
mkdir -p "$out" || exit 1
rev=$(grep -E '^metal2vulkan(-runtime)? *=' Cargo.toml | grep -oE 'rev = "[0-9a-f]{40}"' | head -1 | cut -d'"' -f2)
main=$(git worktree list --porcelain | sed -n '1s/^worktree //p')
m2v=${M2V_DIR:-$(dirname "$main")/metal2vulkan}
common=(--on windows --on "node=$node" --cpu 12 --priority "${PRIORITY:-4}" --kill-on-stall
  --without-submodule vendor/qemu --label "lane=${LANE:-windows-tests}")
[ -n "${MINIBASE:-}" ] && common+=(--env "MINIBASE=$MINIBASE")
if [ -n "$rev" ] && [ -d "$m2v" ]; then
  common+=(--carry "$m2v@$rev")
else
  echo "windows-tests: no metal2vulkan checkout at $m2v (M2V_DIR): the jobs fetch it themselves"
fi
job_args=()
for p in "${packages[@]}"; do job_args+=(-p "$p"); done

# dispatch NAME OPTIONS... -- JOB-ARGS...: queues a job and prints its id. potluck's dispatch
# returns once the job's workspace starts, so it runs in the background while the id is read.
dispatch() {
  local name=$1; shift
  local opts=()
  while [ "$1" != -- ]; do opts+=("$1"); shift; done
  shift
  potluck dispatch --json "${common[@]}" --label "leg=$name" "${opts[@]}" \
    -- bash scripts/internal/windows-tests.sh "$@" > "$out/$head-$name.dispatch" 2>&1 &
  local id=
  for _ in $(seq 1 120); do
    id=$(grep -oE '"job":"[^"]+"' "$out/$head-$name.dispatch" | head -1 | cut -d'"' -f4)
    [ -n "$id" ] && break
    grep -qE '"error"|^error' "$out/$head-$name.dispatch" && break
    sleep 1
  done
  [ -n "$id" ] || { echo "windows-tests: $name was not queued:" >&2; cat "$out/$head-$name.dispatch" >&2; return 1; }
  echo "$id"
}

# wait_job ID LOG: the job's output into LOG once it ends; returns its exit code.
wait_job() {
  local rc
  while true; do
    potluck wait --max-wait 10m "$1" > "$2" 2>&1
    rc=$?
    [ "$rc" = 3 ] || return "$rc"
  done
}

build_failed() {
  echo "windows-tests: the build $build failed:" >&2
  grep -aE '^(error|warning)' -A8 "$out/$head-build.log" | head -60 >&2
  tail -5 "$out/$head-build.log" >&2
  exit 1
}

build=$(dispatch build --ram "${BUILD_RAM:-12}" --timeout 45m --hold 2h -- --job build "${job_args[@]}") || exit 1
echo "windows-tests: build $build at $head (log $out/$head-build.log); keep this checkout clean until the tests are queued"
# The tests are queued in the build's workspace as soon as it starts, where they wait for it: a
# dispatch needs a clean tree, and the checkout may change while the build runs.
workspace=
while [ -z "$workspace" ]; do
  why=$(potluck why "$build" 2>&1)
  workspace=$(echo "$why" | grep -oE 'in workspace [A-Za-z0-9-]+' | head -1 | cut -d' ' -f3)
  [ -n "$workspace" ] && break
  if echo "$why" | grep -qE "^job $build (succeeded|failed|was cancelled|timed out)"; then
    wait_job "$build" "$out/$head-build.log" || build_failed
    workspace=$(grep -aoE '^== windows-tests build on .*, workspace [^,]+' "$out/$head-build.log" | sed 's/.*workspace //' | head -1)
    [ -n "$workspace" ] || build_failed
    break
  fi
  sleep 15
done
test=$(dispatch test --workspace "$workspace" --gpu "$gpu" --ram "${TEST_RAM:-26}" --timeout 20m --hold 1h \
  -- --job test "${job_args[@]}" -- "$@") || exit 1
echo "windows-tests: tests $test queued in $workspace (log $out/$head-test.log)"
if ! wait_job "$build" "$out/$head-build.log"; then
  potluck cancel "$test" > /dev/null 2>&1
  build_failed
fi
wait_job "$test" "$out/$head-test.log"
rc=$?
grep -aE '^ +(FAIL|SIGSEGV|SIGABRT|TIMEOUT) \[|^ +Summary|^mini-base: this|^== windows-tests' "$out/$head-test.log" | tail -40
potluck why "$test" 2>&1 | grep -E 'peak memory'
echo "windows-tests: exit $rc; workspace $workspace held for an hour (potluck release $workspace ends it)"
exit "$rc"
