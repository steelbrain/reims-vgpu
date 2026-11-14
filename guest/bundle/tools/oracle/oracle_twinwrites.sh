#!/bin/sh
# Runs oracle_twinwrites (beside this script, or $PROBE) on the device DEVICE names for 503, 523,
# 507 and 527 and each write, one process each, printing each process's report, its exit status
# and the tail of its standard error.
#   guest/bundle/tools/oracle/oracle_twinwrites.sh DEVICE
set -u
device=${1:?usage: oracle_twinwrites.sh DEVICE}
here=$(cd "$(dirname "$0")" && pwd)
probe=${PROBE:-$here/oracle_twinwrites}
err=$(mktemp "${TMPDIR:-/tmp}/oracle-twinwrites.XXXXXX")
trap 'rm -f "$err"' EXIT
for format in ${FORMATS:-503 523 507 527}; do
  for op in kernel fragment clear; do
    echo "=== $format $op"
    "$probe" "$device" "$format" "$op" 2>"$err"
    echo "exit=$?"
    tail -n 3 "$err" | sed 's/^/stderr: /'
  done
done
echo "=== all done"
