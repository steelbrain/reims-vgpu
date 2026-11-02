#!/bin/bash
# Fetch RgfxGate's pinned build dependencies into build/deps (gitignored, never
# committed): a DEBUG Lilu (its Contents/Resources carries plugin_start.cpp and
# the plugin headers the kext links against) and acidanthera's MacKernelSDK
# kernel headers. Both are acidanthera's, pinned by version/commit and verified
# by SHA-256. The RELEASE Lilu of the same version is what loads at runtime.
#
#   tools/fetch-lilu.sh [DEPSDIR]     (default build/deps)
#
# Downloads are untrusted: verified by SHA-256, only unzipped (never run); the
# kext is compiled by the Makefile with these paths passed in.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPS="${1:-$ROOT/build/deps}"
mkdir -p "$DEPS"

LILU_URL="https://github.com/acidanthera/Lilu/releases/download/1.7.2/Lilu-1.7.2-DEBUG.zip"
LILU_SHA="e95df95d82e8b151047abf359ecd9c853755c59293e19dfcc3907da8fbd8cf89"
MKSDK_COMMIT="3f750085caa17ec3a7880f11c11bf4f48cd6a164"
MKSDK_URL="https://github.com/acidanthera/MacKernelSDK/archive/$MKSDK_COMMIT.zip"
MKSDK_SHA="f028466e8a18cae2e95724152f915a44afc6352a8a80ae3d85567c55b6c1d60d"

verify() {
	local got
	got=$(shasum -a 256 "$1" | cut -d' ' -f1)
	[ "$got" = "$2" ] || { echo "SHA mismatch for $1: $got != $2" >&2; exit 1; }
}
fetch() {
	[ -f "$3" ] && verify "$3" "$2" 2>/dev/null && return 0
	echo "fetching $1"
	curl -fL --max-time 120 -o "$3" "$1"
	verify "$3" "$2"
}

fetch "$LILU_URL" "$LILU_SHA" "$DEPS/Lilu-DEBUG.zip"
fetch "$MKSDK_URL" "$MKSDK_SHA" "$DEPS/MacKernelSDK.zip"

rm -rf "$DEPS/Lilu.kext" "$DEPS/MacKernelSDK"
( cd "$DEPS" && unzip -oq Lilu-DEBUG.zip 'Lilu.kext/Contents/Resources/*' )
( cd "$DEPS" && unzip -oq MacKernelSDK.zip && mv "MacKernelSDK-$MKSDK_COMMIT" MacKernelSDK )

RES="$DEPS/Lilu.kext/Contents/Resources"
test -f "$RES/Library/plugin_start.cpp" || { echo "no plugin_start.cpp in Lilu DEBUG" >&2; exit 1; }
test -d "$RES/Headers" || { echo "no plugin Headers in Lilu DEBUG" >&2; exit 1; }
test -d "$DEPS/MacKernelSDK/Headers" || { echo "no MacKernelSDK/Headers" >&2; exit 1; }
echo "deps ready: $DEPS (Lilu 1.7.2 DEBUG, MacKernelSDK $MKSDK_COMMIT)"
