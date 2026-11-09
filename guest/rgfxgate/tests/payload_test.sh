#!/bin/bash
# payload_test.sh SCRATCH: when the Makefile makes build/gen/RgfxGatePayload.h again. It packs two
# ad hoc signed bundles of our own (a one-line C program, compiled here) through the Makefile's
# payload target, in a build tree under SCRATCH, and checks the payload's cdhash after each make:
#  - a make with the same BUNDLE_DIR, unchanged, leaves the payload as it is;
#  - a bundle rebuilt in place (a new executable, signed again) is packed again;
#  - another BUNDLE_DIR is packed again, though its files are older than the payload;
#  - a make with no BUNDLE_DIR keeps the payload made last;
#  - an arm64 guest's payload, from a fat arm64e+arm64 bundle, admits both slices' cdhashes: the
#    admission decision (rgfx_cdhash_allowed over kExecCDHashes, the arm64 hooks' allowed set)
#    takes the arm64 slice's hash, which third-party processes present, and the arm64e slice's,
#    which is kExecCDHash and comes first.
# SCRATCH is removed first and at the end. Nothing is fetched.
set -euo pipefail
cd "$(dirname "$0")/.."
S=$1
rm -rf "$S"
mkdir -p "$S"
# bash 3.2, macOS's, exits 0 from a set -u error where an EXIT trap is set, with $? 0 in the
# trap too, so an aborted run would read as a pass (docs/lessons.md): the trap fails a run
# that exits 0 short of its end, which sets `ended`.
ended=0
trap 'rc=$?; rm -rf "$S"; [[ $rc != 0 || $ended == 1 ]] || rc=1; exit $rc' EXIT
fails=0

# bundle DIR CODE [ARCH...]: a signed reims-graphics.bundle in DIR whose executable returns
# CODE, with a slice for each ARCH (x86_64 alone where none is given).
bundle() {
	local dir=$1 b=$1/reims-graphics.bundle code=$2
	shift 2
	local archs=()
	for a in "${@:-x86_64}"; do archs+=(-arch "$a"); done
	mkdir -p "$b/Contents/MacOS" "$b/Contents/Resources"
	printf 'int main(void) { return %d; }\n' "$code" > "$dir/main.c"
	xcrun --sdk macosx clang "${archs[@]}" "$dir/main.c" -o "$b/Contents/MacOS/reims-graphics"
	touch "$b/Contents/Resources/wire-version-2"
	cat > "$b/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleExecutable</key><string>reims-graphics</string>
<key>CFBundleIdentifier</key><string>com.steelbrain.rgfxgate.payload-test</string>
</dict></plist>
PLIST
	codesign -s - -f "$b" 2> /dev/null
}

# cdhash DIR: the bundle's cdhash as the payload writes it, decimal bytes.
cdhash() {
	python3 -I -c 'import sys; print(",".join(str(b) for b in bytes.fromhex(sys.argv[1])[:20]))' \
		"$(codesign -dvvv "$1/reims-graphics.bundle" 2>&1 | sed -n 's/^CDHash=//p')"
}

payload() {
	make -s BUILD="$S/build" "$@" "$S/build/gen/RgfxGatePayload.h" > /dev/null
	sed -n 's/^static const uint8_t kExecCDHash\[20\] = {\(.*\)};$/\1/p' "$S/build/gen/RgfxGatePayload.h"
}

expect() {
	if [ "$2" = "$3" ]; then
		echo "payload_test: ok: $1"
	else
		echo "payload_test: FAIL: $1: payload $2, expected $3"
		fails=$((fails + 1))
	fi
}

bundle "$S/a" 1
A1=$(cdhash "$S/a")
expect "first make packs the bundle" "$(payload BUNDLE_DIR="$S/a/reims-graphics.bundle")" "$A1"
before=$(stat -f %Fm "$S/build/gen/RgfxGatePayload.h")
payload BUNDLE_DIR="$S/a/reims-graphics.bundle" > /dev/null
after=$(stat -f %Fm "$S/build/gen/RgfxGatePayload.h")
expect "an unchanged bundle is not packed again" "$after" "$before"

# No wait: each step may fall in the payload's second, which make's file times cannot tell.
bundle "$S/a" 2
A2=$(cdhash "$S/a")
[ "$A2" != "$A1" ] || { echo "payload_test: FAIL: the rebuilt bundle kept its cdhash"; exit 1; }
expect "a bundle rebuilt in place is packed again" "$(payload BUNDLE_DIR="$S/a/reims-graphics.bundle")" "$A2"

bundle "$S/b" 3
find "$S/b" -exec touch -t 202001010000 {} +
B=$(cdhash "$S/b")
expect "another, older BUNDLE_DIR is packed" "$(payload BUNDLE_DIR="$S/b/reims-graphics.bundle")" "$B"
expect "no BUNDLE_DIR keeps the last payload" "$(payload)" "$B"

# An arm64 guest's payload: every slice admitted, the guest's (arm64e) first.
bundle "$S/f" 4 arm64e arm64
slice() { codesign --arch "$2" -dvvv "$1/reims-graphics.bundle" 2>&1 | sed -n 's/^CDHash=//p'; }
FE=$(slice "$S/f" arm64e)
FP=$(slice "$S/f" arm64)
[ -n "$FE" ] && [ -n "$FP" ] && [ "$FE" != "$FP" ] ||
	{ echo "payload_test: FAIL: the fat bundle's two slices have no two cdhashes"; exit 1; }
make -s BUILD="$S/build64" ARCH_NAME=arm64e BUNDLE_DIR="$S/f/reims-graphics.bundle" \
	"$S/build64/gen/RgfxGatePayload.h" > /dev/null
cat > "$S/admit.cpp" <<'CPP'
// Prints, for each cdhash given in hex, whether the payload's allowed set admits it, then
// whether kExecCDHash (the guest's slice) is the set's first entry.
#include "RgfxGatePayload.h"
#include "rgfx_match.h"
#include <cstdio>
#include <cstring>
int main(int argc, char **argv) {
	for (int i = 1; i < argc; i++) {
		uint8_t h[kRgfxCDHashLen] = {};
		for (int j = 0; j < kRgfxCDHashLen && argv[i][2 * j] && argv[i][2 * j + 1]; j++)
			sscanf(argv[i] + 2 * j, "%2hhx", &h[j]);
		printf("%d ", rgfx_cdhash_allowed(h, kRgfxCDHashLen, rgfx_payload::kExecCDHashes,
		                                  rgfx_payload::kExecCDHashCount) ? 1 : 0);
	}
	printf("%d\n", memcmp(rgfx_payload::kExecCDHashes[0], rgfx_payload::kExecCDHash,
	                      kRgfxCDHashLen) == 0 ? 1 : 0);
	return 0;
}
CPP
OTHER=0000000000000000000000000000000000000000
if xcrun --sdk macosx clang++ -std=c++17 -Isrc -I"$S/build64/gen" "$S/admit.cpp" -o "$S/admit" 2> "$S/admit.err"; then
	got=$("$S/admit" "$FE" "$FP" "$OTHER")
else
	got="no allowed set: $(head -1 "$S/admit.err")"
fi
expect "an arm64 guest's payload admits the arm64e and arm64 slices, and no other hash" "$got" "1 1 0 1"
GE=$(python3 -I -c 'import sys; print(",".join(str(b) for b in bytes.fromhex(sys.argv[1])[:20]))' "$FE")
expect "an arm64 guest's kExecCDHash is the arm64e slice's" \
	"$(sed -n 's/^static const uint8_t kExecCDHash\[20\] = {\(.*\)};$/\1/p' "$S/build64/gen/RgfxGatePayload.h")" "$GE"

ended=1
[ $fails = 0 ] || exit 1
echo "payload_test: all checks passed"
