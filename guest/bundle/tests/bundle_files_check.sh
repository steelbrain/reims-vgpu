#!/bin/bash
# bundle_files_check.sh BUNDLE: that the bundle writes no file outside the app's own paths, and
# writes its files with no call an app's sandbox may kill it at. Every Metal process loads it,
# sandboxed ones too. A write outside the process's own paths is refused and reported to the
# kernel's log, so the bundle's executable names no temporary directory. A sandbox may also list
# the system calls the process may make, and WebKit's WebContent profile on macOS 11 and 12 kills
# the process at any other (src/RgfxFiles.hpp). So the bundle writes its files through its own
# plain POSIX calls (src/RgfxFiles.cpp), never through Foundation's, whose system calls are its
# release's to pick: the executable names no Foundation selector that writes, moves or removes a
# file, and imports none of the file calls below, open(2)'s family but open itself among them. What it
# writes is the app's own: a binary archive or a dynamic library at the app's URL, and the GL
# shader variants it lowered, in the app's own cache directory beside Metal's
# (src/RgfxGLStore.hpp); what it reports goes to os_log. Part of `make test`.
set -euo pipefail
exe=$1
[ -f "$exe" ] || exe=$1/Contents/MacOS/reims-graphics
fails=0
paths=$(strings -a "$exe" | grep -E '(^|[^[:alnum:]])/(private/)?(var/)?tmp(/|$)' || true)
if [ -n "$paths" ]; then
	echo "bundle_files_check: FAIL: a temporary path in $exe: $paths"
	fails=1
fi
# open(2)'s family but open itself, the calls that set a file's times or attributes, and the
# calls that rename, remove, make, clone or copy at a directory's descriptor or by other routes:
# each is left out of some WebContent list (tests/files_tests.mm), or is a library call that makes
# one.
calls=$(nm -u "$exe" | grep -E '^_(openat|open_dprotected_np|openat_dprotected_np|guarded_open_np|guarded_open_dprotected_np|fopen|freopen|creat|mkstemp|mkstemps|mkostemp|mkostemps|mkdtemp|tmpfile|utimes|futimes|lutimes|utimensat|futimens|setattrlist|setattrlistat|fsetattrlist|renameat|renamex_np|renameatx_np|unlinkat|mkdirat|clonefile|clonefileat|fclonefileat|copyfile|fcopyfile|exchangedata|removefile)(\$.*)?$' || true)
if [ -n "$calls" ]; then
	echo "bundle_files_check: FAIL: $exe imports" $calls
	fails=1
fi
selectors=$(strings -a "$exe" | grep -E '^(writeToFile:|writeToURL:|createDirectoryAtPath:|createDirectoryAtURL:|createFileAtPath:|removeItemAtPath:|removeItemAtURL:|trashItemAtURL:|moveItemAtPath:|moveItemAtURL:|copyItemAtPath:|copyItemAtURL:|linkItemAtPath:|linkItemAtURL:|replaceItemAtURL:|setAttributes:ofItemAtPath:|setResourceValue:|setResourceValues:|fileHandleForWritingAtPath:|fileHandleForWritingToURL:|fileHandleForUpdatingAtPath:|fileHandleForUpdatingURL:)' | sort -u || true)
if [ -n "$selectors" ]; then
	echo "bundle_files_check: FAIL: $exe names Foundation's file writes:" $selectors
	fails=1
fi
[ $fails = 0 ] || exit 1
echo "bundle_files_check: no temporary path, no Foundation file write, no file call a WebContent list leaves out in $exe"
