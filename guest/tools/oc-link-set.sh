#!/bin/bash
# Prelink the committed Graft set with OpenCore's linker, in user space, in
# Kernel > Add order; check every kext injects and every name resolves, then
# that each dormant consumer (a kext carrying a Graft manifest) stays
# bindable in the prelinked collection (graft-kext check-prelinked, built
# from guest/graft/sdk/tool).
# Outputs remain local.
# Usage: oc-link-set.sh KEXTINJECT BOOT_KC SET OUT
set -euo pipefail

GRAFT_SDK=$(cd "$(dirname "$0")/../graft/sdk" && pwd)

ki=${1:-} kc=${2:-} set=${3:-} out=${4:-}
if [ -z "$ki" ] || [ -z "$kc" ]; then
	echo "oc-link-set: SKIPPED: give KEXTINJECT and BOOT_KC to link with OpenCore's linker"
	exit 0
fi
[ -x "$ki" ] || { echo "oc-link-set: $ki is not an executable" >&2; exit 2; }
[ -f "$kc" ] || { echo "oc-link-set: no Boot KC at $kc" >&2; exit 2; }
[ -f "$set/set.toml" ] || { echo "oc-link-set: no set.toml in '$set'" >&2; exit 2; }
[ -n "$out" ] || { echo "usage: guest/tools/oc-link-set.sh KEXTINJECT BOOT_KC SET OUT" >&2; exit 2; }

abs() { (cd "$(dirname "$1")" && echo "$PWD/$(basename "$1")"); }
ki=$(abs "$ki") kc=$(abs "$kc") set=$(abs "$set")
rm -rf "$out"
mkdir -p "$out/run"
out=$(abs "$out")

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
args=()
binaries=()
consumers=()
ids=()
while read -r bundle exe; do
	args+=("$set/$bundle/$exe" "$set/$bundle/Contents/Info.plist")
	binaries+=("$set/$bundle/$exe")
	if otool -l "$set/$bundle/$exe" | grep -q 'sectname __graft$'; then
		consumers+=("$set/$bundle")
	fi
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
# Every imported name must resolve to a nonzero address, and no kext may import one weakly:
# where the kernel lacks it OpenCore's linker binds it to 0, which the kernel then rebases by the
# collection's slide, so the kext reads a nonzero word it cannot tell from a definition and calls
# it (docs/lessons.md). A name some release lacks is bound through Graft (a consumer's
# `imports`), never weakly.
for i in "${!binaries[@]}"; do
	b=${binaries[$i]} id=${ids[$i]}
	weak=$(nm -arch x86_64 -m "$b" | sed -n 's/^ *(undefined) weak external \([^ ]*\).*/\1/p')
	for w in $weak; do
		echo "oc-link-set: $(basename "$b") imports $w weakly, which OpenCore cannot bind absent" >&2
		status=1
	done
	unresolved=$(grep -E "was resolved for kext $id to 0\$|was unresolved for kext $id\$" "$log" |
		sed -n 's/^OCAK: Symbol \([^ ]*\) was .*/\1/p' || true)
	if [ -n "$unresolved" ]; then
		echo "$unresolved" >&2
		echo "oc-link-set: the names above did not resolve for $id; see $log" >&2
		status=1
	fi
done

if [ "${#consumers[@]}" = 0 ]; then
	echo "oc-link-set: no kext in the set carries a Graft manifest" >&2
	status=1
else
	c++ -std=c++17 -O2 -I"$GRAFT_SDK/include" "$GRAFT_SDK"/tool/*.cpp -o "$out/graft-kext"
	"$out/graft-kext" check-prelinked "$out/run/out.bin" "${consumers[@]}" || status=1
fi

[ "$status" = 0 ] || { echo "oc-link-set: FAILED" >&2; exit 1; }
echo "oc-link-set: OK: the set's ${#ids[@]} kexts linked into $(basename "$kc")"
