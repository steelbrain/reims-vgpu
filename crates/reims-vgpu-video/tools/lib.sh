# shellcheck shell=bash
# What the VA-API checks share. Each sources it after `set -euo pipefail`:
#
#   source "$(dirname "$0")/lib.sh"
#
# It sets `root` (the checkout), `CARGO_TARGET_DIR` (the checkout's target
# unless set) and cargo's PATH, and gives the checks their verdicts:
# `check` and `run` count a failure in `failed`, which the script exits with.

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$root/target}
export PATH=$HOME/.cargo/bin:$PATH
failed=0

# build PACKAGE EXAMPLE...: builds the package's examples, release.
build() {
  local package=$1 example args=()
  shift
  for example in "$@"; do args+=(--example "$example"); done
  (cd "$root" && cargo build --quiet --release -p "$package" "${args[@]}")
}

# hashes OUT INPUT-OPTIONS...: the last field of each line of `ffmpeg
# -f framemd5` of the input, its frames' MD5s, into OUT; 1 where ffmpeg
# fails. OUT is written whole or not at all, so that a run cut short leaves
# no reference that a later run would take. It returns its failure itself,
# since a function called as a condition runs with no `set -e`.
hashes() {
  local out=$1
  shift
  if ! ffmpeg -nostdin -hide_banner -loglevel error "$@" -f framemd5 - |
    awk -F', *' '!/^#/ { print $NF }' > "$out.part"; then
    rm -f "$out.part"
    return 1
  fi
  mv "$out.part" "$out"
}

# wanted NAME: whether the script's stream arguments, `only`, name NAME, or
# name none.
wanted() {
  [[ $only == "  " || $only == *" $1 "* ]]
}

# check LABEL REFERENCE GOT: PASS where GOT's lines are REFERENCE's, which
# has at least one; FAIL otherwise, counted in `failed`.
check() {
  local label=$1 ref=$2 got=$3
  local n_ref n_got
  n_ref=$(wc -l < "$ref")
  n_got=$(wc -l < "$got")
  if ((n_ref > 0)) && cmp -s "$ref" "$got"; then
    echo "PASS $label: $n_got frames match"
  else
    echo "FAIL $label: $n_got frames, reference $n_ref, first difference at line" \
      "$(cmp "$ref" "$got" 2>&1 | awk '{ print $NF }' || true)"
    failed=1
  fi
}

# run INPUT LABEL [OPTION]...: `$tool NODE INPUT OPTION...` into
# `$out/LABEL.frames` and `$out/LABEL.log`, and each frame's MD5, its
# line's last field, into `$out/LABEL.md5`. A run that fails is a FAIL,
# counted in `failed`, and returns 1.
run() {
  local input=$1 label=$2
  shift 2
  local log=$out/$label.log
  if "$tool" "$node" "$input" "$@" > "$out/$label.frames" 2> "$log"; then
    awk '{ print $NF }' "$out/$label.frames" > "$out/$label.md5"
    echo "     $(tail -n1 "$log")"
    return 0
  fi
  echo "FAIL $label: $(tail -n1 "$log")"
  failed=1
  return 1
}

# refused LABEL INPUT PATTERN: PASS where `$tool NODE INPUT` fails with a
# line matching PATTERN (grep -E) in its log, INPUT.log; FAIL where it
# decodes or fails otherwise, counted in `failed`.
refused() {
  local label=$1 input=$2 pattern=$3
  local log=$input.log
  if "$tool" "$node" "$input" > /dev/null 2> "$log"; then
    echo "FAIL $label: decoded, should be refused"
    failed=1
  elif grep -Eq -- "$pattern" "$log"; then
    echo "PASS $label: refused: $(tail -n1 "$log")"
  else
    echo "FAIL $label: $(tail -n1 "$log")"
    failed=1
  fi
}
