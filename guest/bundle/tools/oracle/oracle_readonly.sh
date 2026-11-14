#!/bin/sh
# Runs oracle_readonly (beside this script, or $PROBE) on the device DEVICE names over every shape
# and both backings, one process each, RUNS runs each (default 2), printing each process's report,
# its exit status and the tail of its standard error; SHAPES picks shapes.
#   guest/bundle/tools/oracle/oracle_readonly.sh DEVICE
set -u
device=${1:?usage: oracle_readonly.sh DEVICE}
here=$(cd "$(dirname "$0")" && pwd)
probe=${PROBE:-$here/oracle_readonly}
err=$(mktemp "${TMPDIR:-/tmp}/oracle-readonly.XXXXXX")
trap 'rm -f "$err"' EXIT
runs=${RUNS:-2}
for shape in ${SHAPES:-store resolve-into resolve-into-late smsr-into smsr-into-late \
  late-resolve-texture-known late-resolve-texture-late untouched-into untouched-smsr \
  depth-store depth-dontcare stencil-store stencil-dontcare custom-depth custom-depth-late \
  custom-color custom-color-late custom-stencil custom-stencil-late \
  depth-resolve depth-smsr depth-resolve-late depth-late-texture \
  stencil-resolve stencil-smsr stencil-resolve-late \
  frag-write vert-write resident-write kernel-write \
  blit-copy-into fill-color fill-bytes mipmaps mipmaps-shared invalidate \
  replace replace-heap ro-pages-store ro-pages-resolve-into ro-pages-smsr-into \
  ro-pages-frag-write ro-pages-vert-write ro-pages-kernel-write ro-pages-blit-copy-into \
  ro-pages-fill-color ro-pages-fill-bytes ro-pages-replace icb-cpu icb-gpu icb-gpu-over-cpu}; do
  for backing in bit24 plain; do
    i=0
    while [ "$i" -lt "$runs" ]; do
      echo "=== $shape $backing run $i"
      "$probe" "$device" "$backing" "$shape" 2>"$err"
      echo "exit=$?"
      tail -n 3 "$err" | sed 's/^/stderr: /'
      i=$((i + 1))
    done
  done
done
echo "=== all done"
