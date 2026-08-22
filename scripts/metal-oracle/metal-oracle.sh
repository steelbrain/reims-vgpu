#!/usr/bin/env bash
# metal-oracle.sh — run a Metal contract probe on a real macOS host.
#
#   metal-oracle.sh [probe.swift ...]
#
# With no arguments it runs every `*.swift` beside it. `METAL_ORACLE_HOST` names
# the ssh destination and defaults to `scaleway-m4-macos15`.
#
# Why this exists. This device implements a decoded Apple API contract, and
# AGENTS.md is explicit that a branch must be justified by a contract term
# rather than by a boot that happened to look right. For most terms the wire
# format answers the question. For the ones that are *promises about behavior* —
# when a CPU store becomes visible, what an allocation's footprint is, which
# texture types a linear layout admits — the wire says nothing, and the only
# non-guessing source is a native Metal host.
#
# So these probes are the same instrument as `wire-oracle.sh`, pointed at
# semantics instead of layout. Each prints `RESULT key=value` lines and nothing
# else, so a run is diffable and an answer can be quoted with its terms.
#
# The probes are ordinary Metal programs written for this repository, using
# public API. They are checked in; nothing they produce is derived from Apple
# binaries, and no output of theirs carries third-party bytes.
set -uo pipefail
export LC_ALL=C

HOST="${METAL_ORACLE_HOST:-scaleway-m4-macos15}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE="metal-oracle"

if [ "$#" -gt 0 ]; then PROBES=("$@"); else PROBES=("$HERE"/*.swift); fi

timeout 60 ssh -o BatchMode=yes "$HOST" "mkdir -p ~/$REMOTE" || {
  echo "metal-oracle: cannot reach $HOST over key auth" >&2
  echo "metal-oracle: set METAL_ORACLE_HOST, or install a key on that host" >&2
  exit 2
}

status=0
for probe in "${PROBES[@]}"; do
  [ -f "$probe" ] || { echo "metal-oracle: no such probe: $probe" >&2; status=2; continue; }
  name="$(basename "$probe" .swift)"
  echo "=== $name ==="
  # Copy rather than pipe, so a compile error names a real file and line.
  if ! timeout 60 scp -q -o BatchMode=yes "$probe" "$HOST:$REMOTE/$name.swift"; then
    echo "metal-oracle: could not copy $name" >&2; status=2; continue
  fi
  # `swiftc` and the run are one command: a probe that builds and then fails to
  # find a Metal device is a different answer from one that does not build, and
  # both are answers rather than harness errors.
  timeout 300 ssh -o BatchMode=yes "$HOST" \
    "cd ~/$REMOTE && swiftc -O $name.swift -o $name && ./$name" || {
      echo "metal-oracle: $name did not complete" >&2; status=1; }
done
exit "$status"
