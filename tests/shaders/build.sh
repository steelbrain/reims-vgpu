#!/bin/bash
# Compiles each .metal here into its own metallib, the test shaders the
# platforms' and the graphics device's tests load as a guest's libraries, and
# guest/bundle's tests read. The metallibs are committed, so a test's input
# bytes do not drift with the toolchain; run this on a Mac only to change a
# shader. two.metal holds two kernels on purpose, fc.metal an unspecialized
# function constant, and kernel_trunc.metallib is kernel.metallib cut to
# 1000 bytes.
set -euo pipefail
cd "$(dirname "$0")"
for m in *.metal; do
  n="${m%.metal}"
  xcrun -sdk macosx metal -c "$m" -o "$n.air"
  xcrun -sdk macosx metallib "$n.air" -o "$n.metallib"
  rm "$n.air"
done
head -c 1000 kernel.metallib > kernel_trunc.metallib
