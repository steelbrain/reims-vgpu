#!/bin/sh
# Runs oracle_failways (beside this script, or $PROBE) on the device DEVICE names over every shape,
# both backings and the protection options 2 and 0x10, one process each, RUNS runs each (default
# 2), printing each process's report, its exit status and the tail of its standard error; SHAPES
# picks shapes.
#   guest/bundle/tools/oracle/oracle_failways.sh DEVICE
set -u
device=${1:?usage: oracle_failways.sh DEVICE}
here=$(cd "$(dirname "$0")" && pwd)
probe=${PROBE:-$here/oracle_failways}
err=$(mktemp "${TMPDIR:-/tmp}/oracle-failways.XXXXXX")
trap 'rm -f "$err"' EXIT
runs=${RUNS:-2}
for shape in ${SHAPES:-store prot store-prot prot-store store-late late}; do
  for backing in bit24 plain; do
    for options in 2 0x10; do
      # The store shape sets no options: run it once.
      [ "$shape" = store ] && [ "$options" != 2 ] && continue
      i=0
      while [ "$i" -lt "$runs" ]; do
        echo "=== $shape $backing $options run $i"
        "$probe" "$device" "$backing" "$shape" "$options" 2>"$err"
        echo "exit=$?"
        tail -n 3 "$err" | sed 's/^/stderr: /'
        i=$((i + 1))
      done
    done
  done
done
echo "=== all done"
