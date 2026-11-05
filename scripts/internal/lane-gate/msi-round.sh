#!/bin/bash
# One probe round on the lane host, as a potluck job holding the 5080 and the whole host
# (--exclusive). probe-batch.sh picks the round's probes and dispatches it from a throwaway commit
# whose lg/ holds:
#   lg/round.env  ROUND, RD (the round's directory, MSI_GATE/probes/ROUND), Q (the probe queue), RES
#                 (the probes' results), TAKE (the probes, in order), STAGES (NAME=SRC@TIP each: a
#                 probe copied from SRC on this host, its tip file TIP, rather than taken from the
#                 queue), AFTER_B64 (a command run after the boot, base64: a hook's own end, say)
#   lg/boot.env   msi-boot.sh's boot.env for the round
#   lg/bin/       the gate's host scripts
#   lg/ship/payload/bench   the tip's bench, for the boot's check of our Metal device; winmon
#                           beside it where the gate run has one, for msi-drag.py
# It moves the queued probes into the round, copies the staged ones in, boots (msi-boot.sh, which
# runs msi-probes.sh), prints each probe's status, and puts the probes it did not reach back in
# the queue (staged ones are dropped: staging them again makes them anew).
set -uo pipefail
WS=$(pwd)
. "$WS/lg/round.env"
mkdir -p "$RD/probes" "$RD/payload" "$RD/bin" "$Q" "$RES"
staged=" "
for s in ${STAGES:-}; do
  n=${s%%=*} r=${s#*=}
  src=${r%@*} tip=${r##*@}
  [[ $src == @WS/* ]] && src=$WS/${src#@WS/}  # a probe the round's commit carries (--stage-local)
  rm -rf "$RD/probes/.$n.tmp" "$RD/probes/$n"
  cp -a "$src" "$RD/probes/.$n.tmp" || { echo "msi-round: no probe at $src" >&2; exit 2; }
  # A local probe over one on this host (--stage-local DIR whose .base names it): that one, with
  # DIR's files over it.
  if [[ -f $RD/probes/.$n.tmp/.base ]]; then
    b=$(cat "$RD/probes/.$n.tmp/.base")
    rm -rf "$RD/probes/.$n.tmp"
    cp -a "$b" "$RD/probes/.$n.tmp" || { echo "msi-round: no base probe at $b" >&2; exit 2; }
    cp -a "$src/." "$RD/probes/.$n.tmp/" && rm -f "$RD/probes/.$n.tmp/.base"
  fi
  # More files from this host (DIR/.extra, "SOURCE RELATIVE-PATH" a line).
  if [[ -f $RD/probes/.$n.tmp/.extra ]]; then
    while read -r xs xd; do
      [[ -n $xs && -n $xd ]] || continue
      mkdir -p "$(dirname "$RD/probes/.$n.tmp/$xd")"
      cp -a --reflink=auto "$xs" "$RD/probes/.$n.tmp/$xd" || { echo "msi-round: no $xs for $n" >&2; exit 2; }
    done < "$RD/probes/.$n.tmp/.extra"
    rm -f "$RD/probes/.$n.tmp/.extra"
  fi
  [[ -z $tip ]] || echo "$tip" > "$RD/probes/.$n.tmp/tip"
  mv "$RD/probes/.$n.tmp" "$RD/probes/$n"
  staged+="$n "
  echo "staged $n from $src (tip $(cat "$RD/probes/$n/tip" 2> /dev/null))"
done
for p in $TAKE; do
  [[ $staged == *" $p "* ]] && continue
  mv "$Q/$p" "$RD/probes/" || echo "msi-round: $p is not in the queue"
done
# The round's probes' results from an earlier round go now, not at each probe's start
# (msi-probes.sh): a QEMU hook starts with QEMU and reads PROBE_RESULTS/NAME/stdout for its probe's
# MARK, and the last round's stdout of a probe of the same name holds it already (round
# 20261105-111520: the hook ran before the guest was up, and the probe's start removed what it
# wrote). Each earlier round keeps its own copy in R/probe-results.
for p in $TAKE; do rm -rf "${RES:?}/$p"; done
printf '%s\n' $TAKE > "$RD/probes/.order"
# The gate's host scripts, the hooks among them ({DIR}/bin/NAME in a --qemu-hook).
cp -p "$WS"/lg/bin/* "$RD/bin/"
# The app load's tools (boot.env APPLOAD=1), as rgk-ctx-slots has them now on the lane host.
if grep -q '^APPLOAD=1' "$WS/lg/boot.env"; then
  a=${APPLOAD_TOOLS:-$HOME/.cache/rgk-ctx-slots-msi/appload/tools}
  rm -rf "$RD/bin/appload-tools"
  [[ -f $a/analyze-load.py ]] && cp -r --reflink=auto "$a" "$RD/bin/appload-tools" || echo "msi-round: no app load tools in $a"
fi
cp -p "$WS/lg/ship/payload/"* "$RD/payload/"
cp "$WS/lg/boot.env" "$RD/boot.env"
bash "$RD/bin/msi-boot.sh" "$RD" || echo "probe-batch: the boot ended $? (its log: $RD/boot.log)"
# The round's GPU errors (msi-boot.sh's xid.txt), the boot's own QEMU's marked "own".
sed 's/^/  Xid: /' "$RD/xid.txt" 2> /dev/null
grep -q '^CAPTURE=1' "$RD/boot.env" && sed 's/^/  capture: /' "$RD/capture.txt" 2> /dev/null
# The kext's block-cache stalls over the whole boot (a capture round's ring reader), "S" each
# one's seconds after the desktop came up.
grep -q '^CAPTURE=1' "$RD/boot.env" && { grep -E '^(kring_lines|kring_first_s|block_stalls|block_stalls_early) ' "$RD/host.txt"
  cat "$RD/kext-stalls.txt" 2> /dev/null; } | sed 's/^/  stalls: /'
grep -E '^(host_quiet|cool_wait_s|quiet_wait_s|pkg_temp_start_c|pkg_throttle_ms_during|fatal_lines|xid_own) ' "$RD/host.txt" 2> /dev/null | sed 's/^/  host: /'
# Each probe's status; the ones not reached (or never started: a failed boot) go back to the queue.
for p in $TAKE; do
  st=$(cat "$RES/$p/status" 2> /dev/null || echo 'not reached: the boot did not run it')
  echo "  $p: $st"
  case $st in
    ran|skipped*) rm -rf "$RD/probes/$p" ;;
    *) if [[ $staged == *" $p "* ]]; then rm -rf "$RD/probes/$p"; else mv "$RD/probes/$p" "$Q/" && rm -rf "$RES/$p"; fi ;;
  esac
done
if [[ -n ${AFTER_B64:-} ]]; then
  after=$(printf '%s' "$AFTER_B64" | base64 -d)
  after=${after//\{DIR\}/$RD}  # {DIR}: the round's directory, as in a --qemu-hook
  echo "after the boot: $after"
  ( eval "$after" ) > "$RD/after.txt" 2>&1
  echo "after the boot: exit $? (its output: $RD/after.txt)"
fi
echo "probe-batch: round $ROUND in $RD; the boot's launch.log.gz there"
exit 0
