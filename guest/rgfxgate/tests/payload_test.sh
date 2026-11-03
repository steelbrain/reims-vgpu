#!/bin/bash
# payload_test.sh SCRATCH: when the Makefile makes build/gen/RgfxGatePayload.h again. It packs two
# ad hoc signed bundles of our own (a one-line C program, compiled here) through the Makefile's
# payload target, in a build tree under SCRATCH, and checks the payload's cdhash after each make:
#  - a make with the same BUNDLE_DIR, unchanged, leaves the payload as it is;
#  - a bundle rebuilt in place (a new executable, signed again) is packed again;
#  - another BUNDLE_DIR is packed again, though its files are older than the payload;
#  - a make with no BUNDLE_DIR keeps the payload made last.
# SCRATCH is removed first and at the end. DEPS is a stand-in, so nothing is fetched.
set -euo pipefail
cd "$(dirname "$0")/.."
S=$1
rm -rf "$S"
mkdir -p "$S/deps/Lilu.kext" "$S/deps/MacKernelSDK"
trap 'rm -rf "$S"' EXIT
fails=0

# bundle DIR CODE: a signed reims-graphics.bundle in DIR whose executable returns CODE.
bundle() {
	local b=$1/reims-graphics.bundle
	mkdir -p "$b/Contents/MacOS" "$b/Contents/Resources"
	printf 'int main(void) { return %d; }\n' "$2" > "$1/main.c"
	xcrun --sdk macosx clang -arch x86_64 "$1/main.c" -o "$b/Contents/MacOS/reims-graphics"
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
	make -s BUILD="$S/build" DEPS="$S/deps" "$@" "$S/build/gen/RgfxGatePayload.h" > /dev/null
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

[ $fails = 0 ] || exit 1
echo "payload_test: all checks passed"
