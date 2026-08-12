#!/usr/bin/env bash
#
# scripts/vmapple-snapshot/vmapple-snapshot.sh
#
# Manage the vmapple guest's IMMUTABLE snapshot history. Snapshots are per-rail
# — a rail is one guest OS line — and live under
# vm/guest/rails/<rail>/snapshots/<label>/{disk.img,aux.img.trimmed} with a
# `current` symlink inside each rail. Snapshots are APFS clones (instant, COW)
# and read-only; they are NEVER overwritten. vm/boot-arm64.sh reverts to the
# selected rail's `current` on every boot and captures new snapshots via
# `--capture`; this tool covers the rest.
#
# Every command operates on ONE rail: --rail NAME, else $RAIL, else whatever
# vm/guest/rails/current names. Nothing here repoints `rails/current` — the
# default rail is a deliberate choice, not something a snapshot edit should move.
#
#   rails                list all rails (marks the default)
#   list                 list the rail's snapshots (marks current)
#   current              print the rail's current snapshot label
#   rollback <label>     repoint the rail's `current` (no data touched)
#   create [label]       clone the at-rest guest bundle into a NEW snapshot in
#                        the rail and make it current (guest must be shut down)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
GUEST_DIR="${GUEST_DIR:-$REPO_ROOT/vm/guest}"
RAILS_DIR="${RAILS_DIR:-$GUEST_DIR/rails}"
RAIL_LABEL="${RAIL:-}"

die() { echo "vmapple-snapshot: $*" >&2; exit 1; }

# `--rail NAME` may appear anywhere; strip it before the subcommand is read.
ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --rail) shift; RAIL_LABEL="${1:-}"; [ -n "$RAIL_LABEL" ] || die "--rail needs a name"; shift ;;
    --rail=*) RAIL_LABEL="${1#--rail=}"; shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
set -- ${ARGS[@]+"${ARGS[@]}"}

list_rail_labels() {
  find "$RAILS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
    | while IFS= read -r d; do basename "$d"; done | sort
}

cmd_rails() {
  local d; d="$(readlink "$RAILS_DIR/current" 2>/dev/null || echo "")"
  d="$(basename "${d:-}" 2>/dev/null || echo "")"
  list_rail_labels | while IFS= read -r name; do
    local mark="  "; [ "$name" = "$d" ] && mark="* "
    printf '%s%s\n' "$mark" "$name"
  done
}
# `rails` is answerable without resolving one, so it runs before the resolve below.
if [ "${1:-list}" = "rails" ]; then cmd_rails; exit 0; fi

# --- Resolve the rail ------------------------------------------------------------
if [ -z "$RAIL_LABEL" ]; then
  RAIL_LABEL="$(readlink "$RAILS_DIR/current" 2>/dev/null || true)"
  [ -n "$RAIL_LABEL" ] || die \
    "no default rail: $RAILS_DIR/current is unset (pass --rail NAME, or: ln -sfn <rail> $RAILS_DIR/current)
available: $(list_rail_labels | tr '\n' ' ')"
  RAIL_LABEL="$(basename "$RAIL_LABEL")"
fi
case "$RAIL_LABEL" in
  */*|.|..) die "--rail takes a plain label, not a path: '$RAIL_LABEL'" ;;
esac
RAIL_DIR="$RAILS_DIR/$RAIL_LABEL"
[ -d "$RAIL_DIR" ] || die "no rail '$RAIL_LABEL' at $RAIL_DIR
available: $(list_rail_labels | tr '\n' ' ')"
SNAPSHOTS_DIR="$RAIL_DIR/snapshots"
CURRENT="$SNAPSHOTS_DIR/current"

cur_label() { readlink "$CURRENT" 2>/dev/null || echo ""; }

cmd_list() {
  [ -d "$SNAPSHOTS_DIR" ] || die "rail '$RAIL_LABEL' has no snapshots dir at $SNAPSHOTS_DIR"
  local c; c="$(cur_label)"
  for d in "$SNAPSHOTS_DIR"/*/; do
    [ -d "$d" ] || continue
    local name; name="$(basename "$d")"
    [ "$name" = "current" ] && continue          # skip the `current` symlink
    local mark="  "; [ "$name" = "$c" ] && mark="* "
    printf '%s%s\n' "$mark" "$name"
  done
}

cmd_current() { local c; c="$(cur_label)"; [ -n "$c" ] && echo "$c" || die "rail '$RAIL_LABEL' has no current snapshot"; }

cmd_rollback() {
  local label="${1:-}"; [ -n "$label" ] || die "usage: rollback <label>"
  [ -d "$SNAPSHOTS_DIR/$label" ] || die "rail '$RAIL_LABEL' has no such snapshot: $label (see: list)"
  ln -sfn "$label" "$CURRENT"
  echo "vmapple-snapshot: rail '$RAIL_LABEL' current -> $label"
}

cmd_create() {
  [ -f "$GUEST_DIR/disk.img" ] && [ -f "$GUEST_DIR/aux.img.trimmed" ] \
    || die "no at-rest bundle at $GUEST_DIR (disk.img + aux.img.trimmed)"
  if pgrep -f 'qemu-system-aarch6[4].*vmapple' >/dev/null 2>&1; then
    die "guest is running — shut it down first (scripts/vmapple-shutdown) for a clean snapshot"
  fi
  local label="${1:-$(date +%Y-%m-%d-%H%M%S)-manual}"
  local dir="$SNAPSHOTS_DIR/$label"
  [ -e "$dir" ] && die "snapshot already exists in rail '$RAIL_LABEL': $label"
  mkdir -p "$dir"
  cp -c "$GUEST_DIR/disk.img" "$dir/disk.img" 2>/dev/null || cp "$GUEST_DIR/disk.img" "$dir/disk.img"
  cp -c "$GUEST_DIR/aux.img.trimmed" "$dir/aux.img.trimmed" 2>/dev/null || cp "$GUEST_DIR/aux.img.trimmed" "$dir/aux.img.trimmed"
  chmod 444 "$dir/disk.img" "$dir/aux.img.trimmed"
  ln -sfn "$label" "$CURRENT"
  echo "vmapple-snapshot: rail '$RAIL_LABEL' created + current -> $label"
}

case "${1:-list}" in
  list)     cmd_list ;;
  current)  cmd_current ;;
  rollback) shift; cmd_rollback "$@" ;;
  create)   shift; cmd_create "$@" ;;
  -h|--help) sed -n '2,24p' "$0" ;;
  *) die "unknown command: ${1:-} (rails | list | current | rollback <label> | create [label])" ;;
esac
