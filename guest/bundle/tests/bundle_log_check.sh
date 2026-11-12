#!/bin/bash
# bundle_log_check.sh BUNDLE PLIST HEADER: that the bundle's os_log lines go to its own subsystem.
# A line on OS_LOG_DEFAULT has no subsystem, and no mode of `log config` persists its debug lines
# (docs/lessons.md), so the bundle's executable imports no _os_log_default, the object
# OS_LOG_DEFAULT names; it links its log object (src/RgfxLog.cpp) rather than importing it, which
# its -undefined dynamic_lookup would let through to fail in the guest; and the subsystem HEADER
# names is PLIST's CFBundleIdentifier, the name `log config --subsystem` takes. Part of
# `make test`.
set -euo pipefail
exe=$1
[ -f "$exe" ] || exe=$1/Contents/MacOS/reims-graphics
fails=0
imports=$(nm -u "$exe")
if grep -qx '__os_log_default' <<<"$imports"; then
	echo "bundle_log_check: FAIL: $exe logs on OS_LOG_DEFAULT (it imports _os_log_default)"
	fails=1
fi
if grep -q 'rgfxLog' <<<"$imports"; then
	echo "bundle_log_check: FAIL: $exe imports its log object instead of linking it:" \
		$(grep 'rgfxLog' <<<"$imports")
	fails=1
fi
id=$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$2")
subsystem=$(sed -n 's/^#define RGFX_LOG_SUBSYSTEM "\(.*\)"$/\1/p' "$3")
if [ "$subsystem" != "$id" ]; then
	echo "bundle_log_check: FAIL: the log subsystem \"$subsystem\" ($3) is not the bundle's" \
		"identifier \"$id\" ($2)"
	fails=1
fi
[ $fails = 0 ] || exit 1
echo "bundle_log_check: $exe logs on subsystem $id, not on OS_LOG_DEFAULT"
