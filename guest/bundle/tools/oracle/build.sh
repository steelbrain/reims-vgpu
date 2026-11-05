#!/bin/bash
# Builds the oracle probes and oracle.metallib into OUT (default /tmp/rgk-oracle/x86), for ARCH
# (x86_64 by default; arm64 for an Apple-silicon Mac). Build on a Mac with Xcode's toolchain,
# never on the oracle itself.
#   guest/bundle/tools/oracle/build.sh [OUT [ARCH]]
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
out=${1:-/tmp/rgk-oracle/x86}
arch=${2:-x86_64}
mkdir -p "$out"
for p in oracle_iosurface oracle_storage oracle_passes oracle_cmdbuf oracle_caps oracle_errors oracle_source oracle_ycbcr oracle_ca_ycbcr oracle_sampler oracle_icb oracle_order oracle_names oracle_surftex; do
	xcrun clang -arch "$arch" -mmacosx-version-min=26.0 -fobjc-arc -Wall \
		-framework Metal -framework IOSurface -framework CoreVideo -framework QuartzCore \
		-framework VideoToolbox -framework CoreGraphics -framework Foundation \
		"$here/$p.m" -o "$out/$p"
done
air=$(mktemp -t oracle-air)
xcrun -sdk macosx metal -std=metal3.0 -mmacosx-version-min=26.0 -c "$here/oracle.metal" -o "$air"
xcrun -sdk macosx metallib "$air" -o "$out/oracle.metallib"
xcrun -sdk macosx metal -std=metal3.0 -mmacosx-version-min=26.0 -c "$here/oracle_icb.metal" -o "$air"
xcrun -sdk macosx metallib "$air" -o "$out/oracle_icb.metallib"
rm -f "$air"
# oracle_bundle and its libraries; a library this toolchain does not build is left out, and its
# case says so.
xcrun clang -arch "$arch" -mmacosx-version-min=26.0 -fobjc-arc -Wall -framework Metal \
	-framework Foundation "$here/oracle_bundle.m" -o "$out/oracle_bundle"
for m in oracle_bundle oracle_bundle_mesh oracle_bundle_rt; do
	xcrun -sdk macosx metal -std=metal3.0 -mmacosx-version-min=26.0 "$here/$m.metal" \
		-o "$out/$m.metallib" || echo "oracle_bundle: $m.metallib not built"
done
xcrun -sdk macosx metal -std=metal3.0 -mmacosx-version-min=26.0 -dynamiclib \
	-install_name @executable_path/liboracle_bundle_dyn.metallib "$here/oracle_bundle_dyn.metal" \
	-o "$out/oracle_bundle_dyn.metallib" || echo "oracle_bundle: oracle_bundle_dyn.metallib not built"
echo "built into $out"
