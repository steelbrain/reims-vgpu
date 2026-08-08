#!/usr/bin/env bash
# self-test.sh — does the counter budget actually read the log?
#
# The gate's whole value is that a silent loss stops being silent. A parser that
# matches nothing prints the same eight zeros a clean boot does, so "the gate
# passed" would mean nothing and there would be no way to tell from the output.
# These cases fail without the parser and pass with it.
#
# The log text below is quoted from the emitters, not invented:
# `runtime::storage_flush` writes `deferred_flush_lost kind=...`,
# `runtime::mapper` writes `mapping_page_drift mid=...`, `runtime::drain` writes
# `THRASH present_action_starvation reason=...`, and `note_store_route` counts
# arrive as `key=value` fields on the `store_routes` line that same module
# formats once per per-second window.
#
#   scripts/visual-gate/self-test.sh
#
# Exits 0 when every case holds, 1 on the first that does not. No guest, no
# QEMU, no GPU.
set -uo pipefail
export LC_ALL=C

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUDGET="$SCRIPT_DIR/counter-budget.sh"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Most cases run against an all-zero baseline, so they say what the parser found
# rather than what the shipped budgets happen to allow. The shipped file gets
# cases of its own at the end.
STRICT="$WORK/strict.tsv"
printf '# every class strict\n' >"$STRICT"

fails=0
check() { # check NAME EXPECTED_EXIT EXPECTED_GREP <<< log text
  local name="$1" want_rc="$2" want="$3" log="$WORK/case.log"
  cat >"$log"
  local out rc
  out=$("$BUDGET" "$log" "$STRICT")
  rc=$?
  if [ "$rc" != "$want_rc" ]; then
    echo "self-test: FAIL $name — exit $rc, wanted $want_rc" >&2
    echo "$out" | sed 's/^/self-test:   /' >&2
    fails=$((fails + 1))
    return
  fi
  if ! echo "$out" | grep -q -- "$want"; then
    echo "self-test: FAIL $name — output has no '$want'" >&2
    echo "$out" | sed 's/^/self-test:   /' >&2
    fails=$((fails + 1))
    return
  fi
  echo "self-test: ok $name"
}

# A window with nothing wrong in it is the working state, and it has to be
# distinguishable from a window the parser could not read.
check 'a clean window passes' 0 'deferred_flush_lost	0	' <<'EOF'
drain_duty duty=0.001 draws=0 flushes=0
store_routes mapw_fence_flush=288 gvaw_fence_flush=432 gw_vouched=40
window_publish fresh=34 same_key=9
EOF

check 'a dropped render fails' 1 'deferred_flush_lost	1	' <<'EOF'
drain_duty duty=0.97 draws=2270 flushes=523
deferred_flush_lost kind=gva reason=no_backend gva=0x7f0000 1920x1080 trigger=fence
EOF

check 'page drift under an armed window fails' 1 'mapping_page_drift	2	' <<'EOF'
mapping_page_drift mid=11 task=3 reason=task_inactive pages=2048
mapping_page_drift mid=12 task=3 page=7/2048 gva=0x1000 reason=moved
EOF

check 'present starvation fails' 1 'present_action_starvation	1	' <<'EOF'
THRASH present_action_starvation reason=pending_frames_cap ch=0 head=4 tail=9 unpainted=8 episode=1
EOF

# `render::decode` returning `Ok` is not a decode: `Kind::OtherAccepted` is the
# catch-all for "no arm claimed this", so the draw completes and the guest is
# stamped for work nothing executed. This line is the only trace, and it is
# deduped to one per distinct opcode.
check 'a render opcode with no executor fails' 1 'render_unimplemented	1	' <<'EOF'
render_unimplemented reason=accepted_without_executor task=6 opcode=0x8a len=4 target_refs=[24] pipeline=9 vbufs=1 fbufs=1 ftex=2 hex=8a00040000000000
EOF

# The guest issues the device-info command once, frees the reply buffer and
# answers every later reader from what it parsed, so a key this reply could not
# carry is a capability gone for the boot's life.
check 'a truncated device-info reply fails' 1 'device_info_truncated	1	' <<'EOF'
device_info truncated reason=reply_pairs_exhausted key_table_len=18 count=5 max_pairs=512 wrote=5 have=17 dropped=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
EOF

# Both new classes are `fail()` lines, and the off channel prefixes its own with
# the literal `OFF `. A prefix match that ignored the channel would count a
# census line as a loss — which is the inversion AGENTS.md warns about when
# ranking `reason=` slugs.
check 'an off-channel lookalike is not a loss' 0 'render_unimplemented	0	' <<'EOF'
OFF render_unimplemented this is the off channel and carries no loss
OFF device_info truncated nor does this
EOF

# The route classes are the ones a naive parser gets wrong: they are fields on a
# shared line, not lines of their own.
check 'an unsound witness fails' 1 'gw_audit_unsound	3	' <<'EOF'
store_routes gw_audit_unsound=3 gw_vouched=40 mapw_fence_flush=288
EOF

check 'route counts sum across windows' 1 'tdc_overflow	5	' <<'EOF'
store_routes tdc_overflow=2 tdc_frames=1200
store_routes tdc_overflow=3 tdc_frames=1811
EOF

check 'the writeback ordering repair breaking fails' 1 'render_flush_over_guest_write	1	' <<'EOF'
store_routes mapw_fence_flush=288 render_flush_over_guest_write=1
EOF

# A route name that is a prefix of another must not be counted for it, or a
# class could read hot because an unrelated counter shares its opening letters.
check 'a longer field name is not this class' 0 'tdc_overflow	0	' <<'EOF'
store_routes tdc_overflow_reseeds=7 tdc_frames=1200
EOF

# The same shape on the line side: `deferred_flush_lost_probe` is not
# `deferred_flush_lost`, and a substring match anywhere in a line is not either.
check 'a line family matches its own prefix only' 0 'deferred_flush_lost	0	' <<'EOF'
deferred_flush_lost_probe kind=gva gva=0x1000
readback_split note=deferred_flush_lost was considered
EOF

# The shipped baseline is what a real run is held to, and its non-zero budgets
# have to behave in both directions. The boundary is read out of the file rather
# than written here: re-measuring the standing loss rate should move one number
# in one place, and a test that hardcoded it would fail for having been right
# yesterday.
SHIPPED="$SCRIPT_DIR/baseline.tsv"
shipped_budget() { awk -F'\t' -v k="$1" '$1 == k {print $2 + 0; exit}' "$SHIPPED"; }
LOST_BUDGET=$(shipped_budget deferred_flush_lost)
# The emitted text is the real pair, quoted from the boot that measured it: a
# drift and the render it costs arrive together.
lost_lines() { # lost_lines N
  local i=0
  while [ "$i" -lt "$1" ]; do
    echo "mapping_page_drift mid=$i task=0 page=0/45 gva=0x8283000 reason=translation_moved"
    echo "deferred_flush_lost kind=render mapping=$i 1877x24 fmt=0x50 reason=mapping_page_drift"
    i=$((i + 1))
  done
}
budget_case() { # budget_case NAME EXPECTED_EXIT EXPECTED_GREP <<< log text
  local name="$1" want_rc="$2" want="$3" log="$WORK/case.log"
  cat >"$log"
  local out rc
  out=$("$BUDGET" "$log" "$SHIPPED")
  rc=$?
  if [ "$rc" != "$want_rc" ] || ! echo "$out" | grep -q -- "$want"; then
    echo "self-test: FAIL $name — exit $rc (wanted $want_rc), no '$want'" >&2
    echo "$out" | sed 's/^/self-test:   /' >&2
    fails=$((fails + 1))
    return
  fi
  echo "self-test: ok $name"
}

budget_case 'the standing loss rate is inside its budget' 0 \
  "deferred_flush_lost	$LOST_BUDGET	$LOST_BUDGET" < <(lost_lines "$LOST_BUDGET")

budget_case 'one over the budget still fails' 1 \
  "deferred_flush_lost	$((LOST_BUDGET + 1))	$LOST_BUDGET" \
  < <(lost_lines "$((LOST_BUDGET + 1))")

budget_case 'a standing alarm is budgeted zero in the shipped file' 1 'gw_audit_unsound	1	0' <<'EOF'
store_routes gw_audit_unsound=1
EOF

# A class the baseline never mentions is budgeted zero, so forgetting a row
# fails strict rather than leaving the class unwatched — which is the direction
# an omission should fail in.
PARTIAL="$WORK/partial.tsv"
printf 'deferred_flush_lost\t9\tone row, and nothing about the others\n' >"$PARTIAL"
out=$("$BUDGET" /dev/null "$PARTIAL")
if echo "$out" | grep -q 'mapping_page_drift	0	0' && echo "$out" | grep -q 'deferred_flush_lost	0	9'; then
  echo "self-test: ok a class with no baseline row is budgeted zero"
else
  echo "self-test: FAIL a class with no baseline row is budgeted zero" >&2
  echo "$out" | sed 's/^/self-test:   /' >&2
  fails=$((fails + 1))
fi


# Every class is reported on every run, so a reader can tell "this class read
# zero" from "this class is no longer parsed".
n=$("$BUDGET" /dev/null "$STRICT" | wc -l)
if [ "$n" = 8 ]; then
  echo "self-test: ok all eight classes are reported"
else
  echo "self-test: FAIL only $n classes reported, wanted 8" >&2
  fails=$((fails + 1))
fi

if [ "$fails" = 0 ]; then
  echo "self-test: PASS"
  exit 0
fi
echo "self-test: FAIL — $fails case(s)" >&2
exit 1
