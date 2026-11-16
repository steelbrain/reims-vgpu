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
for p in oracle_iosurface oracle_storage oracle_passes oracle_cmdbuf oracle_caps oracle_errors oracle_source oracle_ycbcr oracle_ca_ycbcr oracle_sampler oracle_icb oracle_order oracle_names oracle_surftex oracle_texdesc oracle_nilbind oracle_reflection oracle_mesh oracle_spi_members oracle_complete oracle_urls oracle_rows oracle_ratemap oracle_fnptr oracle_residency oracle_threads oracle_overlap oracle_profile oracle_waits oracle_archtype oracle_pastlen; do
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
xcrun -sdk macosx metal -std=metal3.0 -mmacosx-version-min=26.0 -c "$here/oracle_reflection.metal" -o "$air"
xcrun -sdk macosx metallib "$air" -o "$out/oracle_reflection.metallib"
xcrun -sdk macosx metal -std=metal3.0 -mmacosx-version-min=26.0 -c "$here/oracle_nilbind.metal" -o "$air"
xcrun -sdk macosx metallib "$air" -o "$out/oracle_nilbind.metallib"
xcrun -sdk macosx metal -std=metal3.0 -mmacosx-version-min=26.0 -c "$here/oracle_ratemap.metal" -o "$air"
xcrun -sdk macosx metallib "$air" -o "$out/oracle_ratemap.metallib"
xcrun -sdk macosx metal -std=metal3.0 -mmacosx-version-min=26.0 -c "$here/oracle_fnptr.metal" -o "$air"
xcrun -sdk macosx metallib "$air" -o "$out/oracle_fnptr.metallib"
xcrun -sdk macosx metal -std=metal3.0 -mmacosx-version-min=26.0 -c "$here/oracle_overlap.metal" -o "$air"
xcrun -sdk macosx metallib "$air" -o "$out/oracle_overlap.metallib"
rm -f "$air"
# oracle_fnptr's dynamic library, and its kernel linked against it.
xcrun -sdk macosx metal -std=metal3.0 -mmacosx-version-min=26.0 -dynamiclib \
	-install_name @executable_path/liboracle_fnptr_dyn.metallib "$here/oracle_fnptr_dyn.metal" \
	-o "$out/liboracle_fnptr_dyn.metallib"
xcrun -sdk macosx metal -std=metal3.0 -mmacosx-version-min=26.0 "$here/oracle_fnptr_link.metal" \
	-L "$out" -loracle_fnptr_dyn -o "$out/oracle_fnptr_link.metallib"
# The same kernel linked against a library of an install name that names no file.
gone=$(mktemp -d)
xcrun -sdk macosx metal -std=metal3.0 -mmacosx-version-min=26.0 -dynamiclib \
	-install_name @executable_path/liboracle_fnptr_gone.metallib "$here/oracle_fnptr_dyn.metal" \
	-o "$gone/liboracle_fnptr_gone.metallib"
xcrun -sdk macosx metal -std=metal3.0 -mmacosx-version-min=26.0 "$here/oracle_fnptr_link.metal" \
	-L "$gone" -loracle_fnptr_gone -o "$out/oracle_fnptr_link_missing.metallib"
rm -r "$gone"
# oracle_bundle and its libraries; a library this toolchain does not build is left out, and its
# case says so.
xcrun clang -arch "$arch" -mmacosx-version-min=26.0 -fobjc-arc -Wall -framework Metal \
	-framework Foundation -framework IOSurface "$here/oracle_bundle.m" -o "$out/oracle_bundle"
xcrun clang -arch "$arch" -mmacosx-version-min=26.0 -fobjc-arc -Wall -framework Metal \
	-framework Foundation -framework IOKit "$here/oracle_events.m" -o "$out/oracle_events"
xcrun clang -arch "$arch" -mmacosx-version-min=26.0 -fobjc-arc -Wall -framework Metal \
	-framework Foundation "$here/oracle_readonly.m" -o "$out/oracle_readonly"
cp "$here/oracle_readonly.sh" "$out/"
xcrun clang -arch "$arch" -mmacosx-version-min=26.0 -fobjc-arc -Wall -framework Metal \
	-framework Foundation "$here/oracle_failways.m" -o "$out/oracle_failways"
cp "$here/oracle_failways.sh" "$out/"
xcrun clang -arch "$arch" -mmacosx-version-min=26.0 -fobjc-arc -Wall -framework Metal \
	-framework IOSurface -framework Foundation "$here/oracle_twinwrites.m" -o "$out/oracle_twinwrites"
cp "$here/oracle_twinwrites.sh" "$out/"
xcrun clang -arch "$arch" -mmacosx-version-min=26.0 -fobjc-arc -Wall -framework Metal \
	-framework IOSurface -framework CoreVideo -framework Foundation "$here/oracle_surfresolve.m" \
	-o "$out/oracle_surfresolve"
cp "$here/oracle_surfresolve.sh" "$out/"
xcrun clang -arch "$arch" -mmacosx-version-min=26.0 -fobjc-arc -Wall -framework Metal \
	-framework IOSurface -framework Foundation "$here/oracle_allocsize.m" -o "$out/oracle_allocsize"
# oracle_fnreflect runs on every release the guests run (macOS 11 on), and its library is built for
# macOS 26 and for macOS 11.
xcrun clang -arch "$arch" -mmacosx-version-min=11.0 -fobjc-arc -Wall -framework Metal \
	-framework Foundation "$here/oracle_fnreflect.m" -o "$out/oracle_fnreflect"
xcrun -sdk macosx metal -std=metal3.0 -mmacosx-version-min=26.0 "$here/oracle_fnreflect.metal" \
	-o "$out/oracle_fnreflect.metallib"
xcrun -sdk macosx metal -std=macos-metal2.3 -mmacosx-version-min=11.0 "$here/oracle_fnreflect.metal" \
	-o "$out/oracle_fnreflect_11.metallib"
for m in oracle_bundle oracle_bundle_mesh oracle_bundle_rt oracle_bundle_args; do
	xcrun -sdk macosx metal -std=metal3.0 -mmacosx-version-min=26.0 "$here/$m.metal" \
		-o "$out/$m.metallib" || echo "oracle_bundle: $m.metallib not built"
done
# A depth/stencil state in an argument buffer is a newer language's: the first standard that takes it.
for std in metal3.2 metal4.0; do
	xcrun -sdk macosx metal -std=$std -mmacosx-version-min=26.0 "$here/oracle_bundle_ds.metal" \
		-o "$out/oracle_bundle_ds.metallib" 2>/dev/null && break
done || true
[ -f "$out/oracle_bundle_ds.metallib" ] || echo "oracle_bundle: oracle_bundle_ds.metallib not built"
xcrun -sdk macosx metal -std=metal3.0 -mmacosx-version-min=26.0 -dynamiclib \
	-install_name @executable_path/liboracle_bundle_dyn.metallib "$here/oracle_bundle_dyn.metal" \
	-o "$out/oracle_bundle_dyn.metallib" || echo "oracle_bundle: oracle_bundle_dyn.metallib not built"
echo "built into $out"
