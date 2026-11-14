#!/bin/sh
# Runs oracle_surfresolve (beside this script, or $PROBE) on the device DEVICE names over every
# case, one process each, printing each process's report, its exit status and the tail of its
# standard error; OVERS, STORAGES and FROMS pick cases.
#   guest/bundle/tools/oracle/oracle_surfresolve.sh DEVICE
set -u
device=${1:?usage: oracle_surfresolve.sh DEVICE}
here=$(cd "$(dirname "$0")" && pwd)
probe=${PROBE:-$here/oracle_surfresolve}
err=$(mktemp "${TMPDIR:-/tmp}/oracle-surfresolve.XXXXXX")
trap 'rm -f "$err"' EXIT
for over in ${OVERS:-own flat plane plane1}; do
  for storage in ${STORAGES:-shared managed private}; do
    for from in ${FROMS:-80 81}; do
      for into in 115 554 555 "$from"; do
        for action in resolve smsr; do
          echo "=== $over $storage $from $into $action"
          "$probe" "$device" "$over" "$storage" "$from" "$into" "$action" 2>"$err"
          echo "exit=$?"
          tail -n 3 "$err" | sed 's/^/stderr: /'
        done
      done
    done
  done
done
echo "=== all done"
