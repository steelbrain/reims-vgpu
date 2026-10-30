#!/bin/bash
# Checks a kext's imports against a macOS 26 KDK, as the kernel's linker
# resolves them: every name the executable imports must be exported by a
# library its OSBundleLibraries names, and it may name no private KPI
# (com.apple.kpi.private and com.apple.kpi.unsupported are open to
# com.apple.* kexts alone). Prints how many names each library gives.
#
#   tools/check-imports.sh KDK KEXT
#
# KDK is a KDK's Extensions directory (System.kext/PlugIns holds the KPIs'
# symbol sets, the families sit beside it), only read.

set -euo pipefail

kdk=${1:-}
kext=${2:-}
if [ -z "$kdk" ]; then
	echo "check-imports: no KDK given (make check KDK=<a macOS 26 KDK's Extensions directory>)" >&2
	exit 2
fi
[ -d "$kdk/System.kext/PlugIns" ] || { echo "check-imports: $kdk is not a KDK's Extensions directory" >&2; exit 2; }
[ -f "$kext/Contents/Info.plist" ] || { echo "check-imports: no kext at $kext" >&2; exit 2; }

plist=$kext/Contents/Info.plist
exe=$kext/Contents/MacOS/$(plutil -extract CFBundleExecutable raw "$plist")
tmp=$(mktemp -d /tmp/check-imports.XXXXXX)
trap 'rm -rf "$tmp"' EXIT

# The KDK's bundles by identifier: a KPI's symbol set, or a family.
find "$kdk" -name Info.plist -path '*.kext/*' -not -path '*.dSYM/*' | while read -r p; do
	id=$(plutil -extract CFBundleIdentifier raw "$p" 2> /dev/null) || continue
	name=$(plutil -extract CFBundleExecutable raw "$p" 2> /dev/null) || continue
	dir=$(dirname "$p")
	[ "$(basename "$dir")" = Contents ] && bin=$dir/MacOS/$name || bin=$dir/$name
	[ -f "$bin" ] && printf '%s\t%s\n' "$id" "$bin"
done > "$tmp/bundles"

# What a library exports: a symbol set lists its names as undefined or
# indirect entries; a family exports its defined external symbols.
exports() {  # binary
	case $1 in
	*/System.kext/PlugIns/*) nm -arch x86_64 "$1" | awk '$1 == "U" || $1 == "I" { print $2 }' ;;
	*) nm -arch x86_64 -gU "$1" | awk '{ print $3 }' ;;
	esac | sort -u
}

libraries=$(plutil -convert json -o - "$plist" | python3 -I -c \
	'import json, sys; print("\n".join(sorted(json.load(sys.stdin).get("OSBundleLibraries", {}))))')
status=0
for lib in $libraries; do
	case $lib in
	com.apple.kpi.private | com.apple.kpi.unsupported)
		echo "check-imports: $lib is named: only com.apple.* kexts may link it" >&2
		status=1
		continue ;;
	esac
	bin=$(awk -F'\t' -v id="$lib" '$1 == id { print $2; exit }' "$tmp/bundles")
	[ -n "$bin" ] || { echo "check-imports: the KDK has no $lib" >&2; status=1; continue; }
	exports "$bin" > "$tmp/lib.$lib"
done
[ "$status" = 0 ] || exit 1

nm -arch x86_64 -u "$exe" | sort -u > "$tmp/imports"
cp "$tmp/imports" "$tmp/left"
for lib in $libraries; do
	n=$(comm -12 "$tmp/left" "$tmp/lib.$lib" | wc -l | tr -d ' ')
	comm -23 "$tmp/left" "$tmp/lib.$lib" > "$tmp/next"
	mv "$tmp/next" "$tmp/left"
	printf '  %-36s %4s names\n' "$lib" "$n"
done
if [ -s "$tmp/left" ]; then
	echo "check-imports: $(wc -l < "$tmp/left" | tr -d ' ') names no named library exports:" >&2
	c++filt < "$tmp/left" | sed 's/^/  /' >&2
	exit 1
fi
echo "check-imports: all $(wc -l < "$tmp/imports" | tr -d ' ') imports resolve, no private KPI"
