#!/bin/bash
# Links the kext into the guest's Boot KC with OpenCore's own linker, in user
# space, as OpenCore's Kernel > Add does at boot: OpenCorePkg's
# Utilities/TestKextInject, which runs OcAppleKernelLib's injection on a KC
# file and writes the KC it made to out.bin.
#
#   tools/oc-link-check.sh KEXTINJECT BOOT_KC KDK KEXT OUT
#
# OpenCore links an injected kext against the Boot KC alone, and
# IOGraphicsFamily and AppleGraphicsDeviceControl live in the System KC. So the
# kext loads beside a library of placeholders, as the loader's bridges carry
# them for it: every name of those two the kext imports, as a ud2 stub for a
# function and zeroes for a class's vtable or metaclass (a zero vtable has no
# entries, so OpenCore leaves each inherited slot to be resolved by its name).
# The kext is injected with OSBundleLibraries naming that library in their
# place. This builds the placeholders from the KDK's two (only their export
# lists are read), injects both kexts, and fails unless both are injected
# with every name resolved.
#
# Skipped, exit 0 and a line saying so, without KEXTINJECT or BOOT_KC.
# Everything lands in OUT; out.bin is read, never booted.

set -euo pipefail

ki=${1:-} kc=${2:-} kdk=${3:-} kext=${4:-} out=${5:-}
if [ -z "$ki" ] || [ -z "$kc" ]; then
	echo "oc-link-check: SKIPPED: give KEXTINJECT and BOOT_KC to link with OpenCore's linker"
	exit 0
fi
[ -x "$ki" ] || { echo "oc-link-check: $ki is not an executable" >&2; exit 2; }
[ -f "$kc" ] || { echo "oc-link-check: no Boot KC at $kc" >&2; exit 2; }
[ -n "$out" ] && [ -f "$kext/Contents/Info.plist" ] || {
	echo "usage: tools/oc-link-check.sh KEXTINJECT BOOT_KC KDK KEXT OUT" >&2
	exit 2
}
# The System KC's libraries the kext names, by bundle ID, and their binaries in the KDK.
families=(com.apple.iokit.IOGraphicsFamily com.apple.AppleGraphicsDeviceControl)
binaries=("$kdk/IOGraphicsFamily.kext/IOGraphicsFamily"
	"$kdk/AppleGraphicsControl.kext/Contents/PlugIns/AppleGraphicsDeviceControl.kext/Contents/MacOS/AppleGraphicsDeviceControl")
for family in "${binaries[@]}"; do
	[ -f "$family" ] || { echo "oc-link-check: the KDK has no $family" >&2; exit 2; }
done

abs() { (cd "$(dirname "$1")" && echo "$PWD/$(basename "$1")"); }
ki=$(abs "$ki") kc=$(abs "$kc") kext=$(abs "$kext")
rm -rf "$out"
mkdir -p "$out/obj" "$out/run"
out=$(abs "$out")

SHIM_ID=com.steelbrain.ReimsGraphics.LinkCheckIOGraphicsFamily
SHIM=LinkCheckIOGraphicsFamily
exe=$(plutil -extract CFBundleExecutable raw "$kext/Contents/Info.plist")

# The placeholders: the kext's imports that those libraries export.
nm -arch x86_64 -u "$kext/Contents/MacOS/$exe" | sort -u > "$out/obj/imports"
for family in "${binaries[@]}"; do
	nm -arch x86_64 -gU "$family" | awk '{ print $3 }'
done | sort -u > "$out/obj/family"
comm -12 "$out/obj/imports" "$out/obj/family" > "$out/obj/names"
{
	echo '.text'
	grep -v -e '^__ZTV' -e '10gMetaClassE$' -e '9metaClassE$' "$out/obj/names" |
		while read -r n; do printf '.globl %s\n.p2align 4\n%s:\n\tud2\n' "$n" "$n"; done
	echo '.section __DATA,__data'
	grep -e '^__ZTV' -e '10gMetaClassE$' -e '9metaClassE$' "$out/obj/names" |
		while read -r n; do printf '.globl %s\n.p2align 4\n%s:\n\t.space 64\n' "$n" "$n"; done
} > "$out/obj/$SHIM.s"
cat > "$out/obj/kmod_info.c" << EOF
#include <mach/mach_types.h>
KMOD_EXPLICIT_DECL($SHIM_ID, "1.0.0", 0, 0)
EOF
sdk=$(xcrun --sdk macosx --show-sdk-path)
arch=(-arch x86_64 -isysroot "$sdk" -mmacosx-version-min=26.0)
bundle=$out/kexts/$SHIM.kext
mkdir -p "$bundle/Contents/MacOS"
xcrun clang "${arch[@]}" -c "$out/obj/$SHIM.s" -o "$out/obj/$SHIM.o"
xcrun clang "${arch[@]}" -mkernel -nostdinc -DKERNEL \
	-isystem "$sdk/System/Library/Frameworks/Kernel.framework/Headers" \
	-c "$out/obj/kmod_info.c" -o "$out/obj/kmod_info.o"
xcrun clang "${arch[@]}" -nostdlib -Xlinker -kext "$out/obj/$SHIM.o" "$out/obj/kmod_info.o" \
	-o "$bundle/Contents/MacOS/$SHIM"
python3 -I - "$bundle/Contents/Info.plist" "$SHIM_ID" "$SHIM" << 'EOF'
import plistlib, sys
path, ident, name = sys.argv[1:]
plistlib.dump({
    "CFBundleExecutable": name, "CFBundleIdentifier": ident, "CFBundleName": name,
    "CFBundleInfoDictionaryVersion": "6.0", "CFBundlePackageType": "KEXT",
    "CFBundleVersion": "1.0.0", "CFBundleShortVersionString": "1.0.0",
    "OSBundleCompatibleVersion": "1.0.0",
    "OSBundleLibraries": {"com.apple.kpi.libkern": "20.0.0"},
}, open(path, "wb"))
EOF

# The kext, naming the placeholders in those libraries' place.
staged=$out/kexts/$(basename "$kext")
cp -R "$kext" "$staged"
python3 -I - "$staged/Contents/Info.plist" "$SHIM_ID" "${families[@]}" << 'EOF'
import plistlib, sys
path, ident, *families = sys.argv[1:]
info = plistlib.load(open(path, "rb"))
libraries = info["OSBundleLibraries"]
for family in families:
    libraries.pop(family, None)
libraries[ident] = "1.0.0"
plistlib.dump(info, open(path, "wb"))
EOF

# KextInject exits nonzero whenever any of its own steps failed, its test
# patches of the kernel included, some of which never apply to this kernel;
# the log decides.
(cd "$out/run" && "$ki" "$kc" \
	"$bundle/Contents/MacOS/$SHIM" "$bundle/Contents/Info.plist" \
	"$staged/Contents/MacOS/$exe" "$staged/Contents/Info.plist" > kextinject.log 2>&1) || true
log=$out/run/kextinject.log
[ -s "$out/run/out.bin" ] || { echo "oc-link-check: KextInject wrote no out.bin; see $log" >&2; exit 1; }

status=0
id=$(plutil -extract CFBundleIdentifier raw "$kext/Contents/Info.plist")
for injected in "$bundle/Contents/MacOS/$SHIM" "$staged/Contents/MacOS/$exe"; do
	if grep -q -F "[OK] $injected injected - Success" "$log"; then
		echo "oc-link-check: $(basename "$injected") injected"
	else
		echo "oc-link-check: $(basename "$injected") not injected; see $log" >&2
		status=1
	fi
done
grep -q -F '[OK] Prelink inject complete success' "$log" ||
	{ echo "oc-link-check: the injection did not complete; see $log" >&2; status=1; }
# A name only a vtable slot uses resolves to 0 when no library exports it,
# and the slot stays empty: that is a failure too.
if grep -E "was resolved for kext $id to 0\$|has 0-value" "$log" >&2; then
	echo "oc-link-check: the names above resolved to 0; see $log" >&2
	status=1
fi
resolved=$(grep -c "was resolved for kext $id to" "$log" || true)
imports=$(wc -l < "$out/obj/imports" | tr -d ' ')
if [ "$resolved" != "$imports" ]; then
	echo "oc-link-check: $resolved of the kext's $imports names resolved; see $log" >&2
	status=1
fi
[ "$status" = 0 ] || exit 1
echo "oc-link-check: all $imports names resolved, $(wc -l < "$out/obj/names" | tr -d ' ') of them to the System KC's placeholders"
