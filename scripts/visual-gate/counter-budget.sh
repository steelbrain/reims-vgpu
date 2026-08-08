#!/usr/bin/env bash
# counter-budget.sh — read the eight silent-loss classes out of one window of the
# fail log and hold each to its budget.
#
# Split out of `visual-gate.sh` so it can be tested against synthetic log text
# without a live boot. A parser that silently matches nothing reads exactly like
# a clean run, which is the failure mode this whole gate exists to remove.
#
#   scripts/visual-gate/counter-budget.sh WINDOW_LOG [BASELINE_TSV]
#
# Prints one `name<TAB>count<TAB>budget` line per class, in a fixed order. Exits
# 0 when every class is inside its budget, 1 when any is over, 2 when a file
# cannot be read.
#
# Five classes are line families, keyed on the prefix their emitter writes at
# the start of a line. Three are `note_store_route` counts, which arrive as
# `key=value` fields on a `store_routes` line — that line is emitted once per
# per-second window, so a class has to be summed across every such line in the
# window rather than read off the last one.
#
# The budgets live in `baseline.tsv` rather than here, because a non-zero one is
# an admission about the device and needs its measurement written beside it.
set -euo pipefail
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ $# -ge 1 ] && [ $# -le 2 ] || {
  echo "counter-budget: usage: counter-budget.sh WINDOW_LOG [BASELINE_TSV]" >&2; exit 2; }
WINDOW="$1"
BASELINE="${2:-$SCRIPT_DIR/baseline.tsv}"
[ -r "$WINDOW" ] || { echo "counter-budget: cannot read $WINDOW" >&2; exit 2; }
[ -r "$BASELINE" ] || { echo "counter-budget: cannot read $BASELINE" >&2; exit 2; }

# `deferred_flush_lost` — a guest render this device dropped.
# `mapping_page_drift`  — the page list changed under an armed window.
# `THRASH present_action_starvation` — one class spelled as two words.
# `render_unimplemented` — a render opcode the decoder accepted and nothing
#   executed. This is the class a green run hides best: `render::decode`
#   returning `Ok` is not a decode, because `Kind::OtherAccepted` is the
#   catch-all for "no arm claimed this", so the draw completes, the guest is
#   stamped, and the only trace is this line. It is deduped to one per distinct
#   opcode, so the count is a count of *kinds* lost, not of records.
# `device_info truncated` — the device-info reply could not carry every
#   capability key the guest's own parse ceiling admits. The guest asks once per
#   boot and never re-asks, so a firing is a capability gone for that boot's
#   life. Spelled as two words for the same reason the starvation class is.
LINE_CLASSES=(
  'deferred_flush_lost|deferred_flush_lost '
  'mapping_page_drift|mapping_page_drift '
  'present_action_starvation|THRASH present_action_starvation '
  'render_unimplemented|render_unimplemented '
  'device_info_truncated|device_info truncated '
)

# `gw_audit_unsound` — the gather witness refuted itself; a stale image is being
#   served.
# `render_flush_over_guest_write` — documented as expected-never; if it fires the
#   writeback ordering repair has broken.
# `tdc_overflow` — the census target map overflowed and re-seeded.
ROUTE_CLASSES=(gw_audit_unsound render_flush_over_guest_write tdc_overflow)

# A class with no row in the baseline is budgeted zero. A new class therefore
# starts strict rather than starting unwatched, which is the direction a
# forgotten entry should fail in.
budget_for() {
  awk -F'\t' -v key="$1" '
    /^#/ || /^[[:space:]]*$/ { next }
    $1 == key { print $2 + 0; found = 1; exit }
    END { if (!found) print 0 }' "$BASELINE"
}

over=0
report() { # report NAME COUNT
  local budget
  budget=$(budget_for "$1")
  printf '%s\t%s\t%s\n' "$1" "$2" "$budget"
  [ "$2" -le "$budget" ] || over=1
}

for entry in "${LINE_CLASSES[@]}"; do
  name=${entry%%|*}
  prefix=${entry#*|}
  report "$name" "$(grep -c -- "^$prefix" "$WINDOW" || true)"
done

for name in "${ROUTE_CLASSES[@]}"; do
  report "$name" "$(awk -v key="$name" '
    /^store_routes /{
      for (i = 2; i <= NF; i++) {
        split($i, kv, "=")
        if (kv[1] == key) total += kv[2]
      }
    }
    END { print total + 0 }' "$WINDOW")"
done

exit "$over"
