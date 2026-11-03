#!/bin/bash
# bundle_files_check.sh BUNDLE: that the bundle writes no file of its own. Every Metal process
# loads it, sandboxed ones too, where a write outside the process's own paths is refused and
# reported to the kernel's log: so the bundle's executable names no temporary directory and
# imports none of open(2)'s family. What it writes is the app's own, through Foundation (a binary
# archive at the app's URL); what it reports goes to os_log. Part of `make test`.
set -euo pipefail
exe=$1
[ -f "$exe" ] || exe=$1/Contents/MacOS/reims-graphics
fails=0
paths=$(strings -a "$exe" | grep -E '(^|[^[:alnum:]])/(private/)?(var/)?tmp(/|$)' || true)
if [ -n "$paths" ]; then
	echo "bundle_files_check: FAIL: a temporary path in $exe: $paths"
	fails=1
fi
calls=$(nm -u "$exe" | grep -E '^_(open|openat|open_dprotected_np|fopen|freopen|creat|mkstemp|mkstemps|mkdtemp|tmpfile)(\$.*)?$' || true)
if [ -n "$calls" ]; then
	echo "bundle_files_check: FAIL: $exe imports" $calls
	fails=1
fi
[ $fails = 0 ] || exit 1
echo "bundle_files_check: no temporary path, no open(2) family in $exe"
