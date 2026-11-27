#!/bin/bash
# bundle_sync_check.sh BUNDLE SRC [OBJECT...]: that the bundle takes no lock on an object an app can
# reach. @synchronized(x) locks x's own lock in the Objective-C runtime (objc_sync_enter), which an
# app's @synchronized(x) takes too: an app that holds its command buffer's across the buffer's commit
# and waitUntilCompleted, as a 2D game on a common engine does, waits there for good if the bundle's
# completion takes the same lock. So the bundle keeps locks of its own (std::mutex), never an
# object's (tests/queue_tests.mm, AN APP'S LOCKS). Every route to the runtime's lock imports
# objc_sync_enter or objc_sync_exit: neither may be imported by the x86-64 bundle, BUNDLE, nor by
# the OBJECTs, which `make test` passes as the native build of the bundle's sources but the device
# shell's, so the arm64 device's code is looked at too. The grep of SRC finds a @synchronized in
# what neither build compiles (the device shell's arm64 code). Part of `make test`.
set -euo pipefail
exe=$1
src=$2
shift 2
[ -f "$exe" ] || exe=$exe/Contents/MacOS/reims-graphics
fails=0
for bin in "$exe" "$@"; do
	calls=$(nm -u "$bin" | grep -E '^_objc_sync_(enter|exit)$' || true)
	if [ -n "$calls" ]; then
		echo "bundle_sync_check: FAIL: $bin imports" $calls "(a @synchronized or a call of the runtime's object lock)"
		fails=1
	fi
done
# Code lines only: a line that starts a // comment, or a /* */ comment's * line, is not code.
lines=$(grep -rnE '@synchronized|objc_sync_(enter|exit)' "$src" | grep -vE '^[^:]+:[0-9]+:[[:space:]]*(//|\*|/\*)' || true)
if [ -n "$lines" ]; then
	echo "bundle_sync_check: FAIL: the runtime's object lock in $src:"
	echo "$lines"
	fails=1
fi
[ $fails = 0 ] || exit 1
echo "bundle_sync_check: no objc_sync_enter or objc_sync_exit in $exe and $# objects, and no @synchronized in $src"
