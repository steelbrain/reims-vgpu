#!/bin/bash
# Links Lilu and the kext set into the guest's Boot KC with OpenCore's own
# linker, in user space, as OpenCore's Kernel > Add does at boot: OpenCorePkg's
# Utilities/TestKextInject, which runs OcAppleKernelLib's injection on a KC file
# and writes the KC it made to out.bin. Each kext is linked against the Boot KC
# and those injected before it, in set.toml's order after Lilu, as the image's
# entries are.
#
#   guest/tools/oc-link-set.sh KEXTINJECT BOOT_KC LILU SET TABLE OUT
#
# LILU is the RELEASE Lilu.kext the image carries, SET a directory with
# set.toml and its bundles (guest/prebuilt, or make set's OUT), TABLE the
# loader's fixup table for SET's ReimsGraphics (make set writes it to
# build/set-work/loader/staged/ReimsGraphics.bridge/fixups.json). Fails unless
# every kext is injected, every name each one imports resolves, none to 0,
# and every site of ReimsGraphics holds its shim's placeholder
# (guest/loader/tools/oc-link-check.py), which is what the loader checks before
# it binds. So the set needs nothing in the image but Lilu and the Boot KC.
#
# Skipped, exit 0 and a line saying so, without KEXTINJECT or BOOT_KC.
# Everything lands in OUT; out.bin is read, never booted.
set -euo pipefail

ki=${1:-} kc=${2:-} lilu=${3:-} set=${4:-} table=${5:-} out=${6:-}
if [ -z "$ki" ] || [ -z "$kc" ]; then
	echo "oc-link-set: SKIPPED: give KEXTINJECT and BOOT_KC to link with OpenCore's linker"
	exit 0
fi
[ -x "$ki" ] || { echo "oc-link-set: $ki is not an executable" >&2; exit 2; }
[ -f "$kc" ] || { echo "oc-link-set: no Boot KC at $kc" >&2; exit 2; }
[ -f "$lilu/Contents/MacOS/Lilu" ] || { echo "oc-link-set: no Lilu.kext at '$lilu' (LILU=)" >&2; exit 2; }
[ -f "$set/set.toml" ] || { echo "oc-link-set: no set.toml in '$set'" >&2; exit 2; }
[ -f "$table" ] || { echo "oc-link-set: no fixup table at '$table' (make set writes it)" >&2; exit 2; }
[ -n "$out" ] || { echo "usage: guest/tools/oc-link-set.sh KEXTINJECT BOOT_KC LILU SET TABLE OUT" >&2; exit 2; }
tools=$(cd "$(dirname "$0")/../loader/tools" && pwd)
abs() { (cd "$(dirname "$1")" && echo "$PWD/$(basename "$1")"); }
ki=$(abs "$ki") kc=$(abs "$kc") lilu=$(abs "$lilu") set=$(abs "$set") table=$(abs "$table")
rm -rf "$out"
mkdir -p "$out/run"
out=$(abs "$out")

# The table must describe the set's own ReimsGraphics.
python3 -I - "$table" "$set/ReimsGraphics.kext/Contents/MacOS/ReimsGraphics" << 'EOF'
import hashlib, json, sys
table, binary = sys.argv[1:]
want = json.load(open(table))["consumer_sha256"]
if hashlib.sha256(open(binary, "rb").read()).hexdigest() != want:
    sys.exit(f"oc-link-set: {table} was made from another ReimsGraphics than {binary}")
EOF

# The bundles in set.toml's order, each "BundlePath ExecutablePath".
entries=$(python3 -I - "$set/set.toml" << 'EOF'
import json, re, sys
bundle = None
for line in open(sys.argv[1]):
    if m := re.match(r'BundlePath = (".*")$', line.strip()):
        bundle = json.loads(m[1])
    elif (m := re.match(r'ExecutablePath = (".*")$', line.strip())) and bundle:
        print(bundle, json.loads(m[1]))
        bundle = None
EOF
)
args=("$lilu/Contents/MacOS/Lilu" "$lilu/Contents/Info.plist")
binaries=("$lilu/Contents/MacOS/Lilu")
ids=(as.vit9696.Lilu)
while read -r bundle exe; do
	args+=("$set/$bundle/$exe" "$set/$bundle/Contents/Info.plist")
	binaries+=("$set/$bundle/$exe")
	ids+=("$(plutil -extract CFBundleIdentifier raw "$set/$bundle/Contents/Info.plist")")
done <<< "$entries"

# KextInject exits nonzero whenever any of its own steps failed, its test
# patches of the kernel included, some of which never apply to this kernel;
# the log decides.
(cd "$out/run" && "$ki" "$kc" "${args[@]}" > kextinject.log 2>&1) || true
log=$out/run/kextinject.log
[ -s "$out/run/out.bin" ] || { echo "oc-link-set: KextInject wrote no out.bin; see $log" >&2; exit 1; }

status=0
for i in "${!binaries[@]}"; do
	b=${binaries[$i]} id=${ids[$i]}
	if grep -q -F "[OK] $b injected - Success" "$log"; then
		imports=$(nm -arch x86_64 -u "$b" | sort -u | wc -l | tr -d ' ')
		resolved=$(grep -c "was resolved for kext $id to" "$log" || true)
		echo "oc-link-set: $(basename "$b") injected; $resolved of its $imports imports resolved"
		[ "$resolved" = "$imports" ] || status=1
	else
		echo "oc-link-set: $(basename "$b") not injected; see $log" >&2
		status=1
	fi
done
grep -q -F '[OK] Prelink inject complete success' "$log" ||
	{ echo "oc-link-set: the injection did not complete; see $log" >&2; status=1; }
# A name only a vtable slot uses resolves to 0 when no library exports it,
# and the slot stays empty: a failure for the set's kexts. Lilu, which every
# image carries as it is, resolves two reserved IOService slots to 0.
for id in "${ids[@]:1}"; do
	if grep -E "was resolved for kext $id to 0\$|was unresolved for kext $id\$" "$log" >&2; then
		echo "oc-link-set: the names above did not resolve for $id; see $log" >&2
		status=1
	fi
done
python3 -I -B "$tools/oc-link-check.py" "$log" "$out/run/out.bin" "$table" com.steelbrain.ReimsGraphics || status=1
[ "$status" = 0 ] || { echo "oc-link-set: FAILED" >&2; exit 1; }
echo "oc-link-set: OK: Lilu and the set's $((${#ids[@]} - 1)) kexts linked into $(basename "$kc")"
