#!/usr/bin/env bash
#
# runtime-dead.sh — which reims-vgpu functions never executed on a real boot.
#
# `scripts/dead-state` answers "what does nothing reference". This answers the
# other question: what compiles, links, is reachable, and the guest protocol
# still never takes. Those are different sets, and only the second one needs a
# guest to measure.
#
# Method: build the staticlib with -C instrument-coverage, link the LLVM profile
# runtime into QEMU (see hw/display/meson.build), boot the x86 guest, drive it
# so the measurement is of a busy device rather than an idle one, then stop QEMU
# with SIGTERM so the atexit writer runs. SIGKILL loses everything — continuous
# mode (%c) would survive it but needs runtime counter relocation, which this
# toolchain does not build.
#
# READ THE README BEFORE DELETING ANYTHING. A zero here is not a verdict.
#
# Usage: scripts/runtime-dead/runtime-dead.sh [--seconds N] [--app NAME] [--import-off]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DRIVE_SECONDS=25
DRIVE_APP=Safari
IMPORT_OFF=0

while [ $# -gt 0 ]; do
    case "$1" in
        --seconds) DRIVE_SECONDS="$2"; shift 2 ;;
        --app) DRIVE_APP="$2"; shift 2 ;;
        --import-off) IMPORT_OFF=1; shift ;;
        *) echo "runtime-dead: unknown argument '$1'" >&2; exit 2 ;;
    esac
done

# Where a host can import guest RAM, every guest window takes the import and the
# copying rails run zero times — so the report above says nothing about them,
# and on a host without the extension they are the only rails there are. Reason
# 7 in the README is that whole class: a rail cold here because this host is too
# capable to need it reads exactly like a dead one.
#
# `REIMS_VGPU_GUEST_IMPORT=off` is what AGENTS.md provides to exercise them, so
# this run is the other half of the measurement rather than a variant of it.
# Separate output directory on purpose: the two reports are meant to be diffed,
# and a run that overwrote the other would destroy the comparison it exists for.
if [ "$IMPORT_OFF" -eq 1 ]; then
    OUT_DIR="${OUT_DIR:-/tmp/reims-vgpu-runtime-dead-import-off}"
    export REIMS_VGPU_GUEST_IMPORT=off
else
    OUT_DIR="${OUT_DIR:-/tmp/reims-vgpu-runtime-dead}"
fi

# The profile runtime is compiler-rt's, not rustup's: rustup ships
# profiler_builtins as an rlib for linking into Rust artifacts, and what QEMU
# needs is the plain archive. Its LLVM major must match rustc's or the .profraw
# it writes will not parse.
rustc_llvm="$(rustc --version --verbose | sed -n 's/^LLVM version: \([0-9]*\).*/\1/p')"
PROFILE_RT=""
for cand in /usr/lib/clang/"$rustc_llvm"/lib/linux/libclang_rt.profile-x86_64.a \
            /usr/lib/clang/*/lib/linux/libclang_rt.profile-x86_64.a; do
    [ -f "$cand" ] || continue
    PROFILE_RT="$cand"
    break
done
if [ -z "$PROFILE_RT" ]; then
    echo "runtime-dead: no libclang_rt.profile-x86_64.a found (want LLVM $rustc_llvm)" >&2
    echo "runtime-dead: install compiler-rt for a clang whose major matches rustc's" >&2
    exit 1
fi
rt_llvm="$(printf %s "$PROFILE_RT" | sed -n 's|.*/clang/\([0-9]*\)/.*|\1|p')"
if [ -n "$rt_llvm" ] && [ "$rt_llvm" != "$rustc_llvm" ]; then
    echo "runtime-dead: WARNING profile runtime is LLVM $rt_llvm, rustc is $rustc_llvm" >&2
fi

command -v llvm-profdata >/dev/null || { echo "runtime-dead: need llvm-profdata" >&2; exit 1; }
command -v llvm-cov >/dev/null || { echo "runtime-dead: need llvm-cov" >&2; exit 1; }

# The pids of processes that really are a qemu-system-x86_64, read from
# /proc/<pid>/exe rather than matched against a command line.
#
# `ps -eo pid,args | grep '[q]emu-system-x86_64'` matches any argv holding that
# string, and two things in this script's own critical path do: the boot script
# re-runs qemu-build, whose ninja link step spawns `cc ... -o
# .../qemu-system-x86_64 ...` for a 119 MB binary, and qemu-build then makes four
# short-lived `qemu-system-x86_64 -device help` probes. The pid loop polls every
# two seconds and both windows fall inside it. Latching the linker gives a pid
# that never writes a profile at all, so the guard below reported the boot's own
# profile "missing" for a boot whose raw was on disk, complete, with a total
# count of 4.3 billion. An argv match cannot tell a compiler writing that name
# from a process running it; the exe link can.
qemu_pids() {
    local proc pid exe
    for proc in /proc/[0-9]*; do
        pid="${proc#/proc/}"
        exe="$(readlink "$proc/exe" 2>/dev/null)" || continue
        case "$exe" in
            */qemu-system-x86_64|*/qemu-system-x86_64' (deleted)') echo "$pid" ;;
        esac
    done
}

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
QEMU_BIN="$REPO_ROOT/vendor/qemu/build/qemu-system-x86_64"

# Scoped to the host triple on purpose. A bare RUSTFLAGS also reaches the
# x86_64-unknown-uefi option ROM, which has no profiler_builtins for its target
# and fails the whole boot before QEMU starts.
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C instrument-coverage"
export REIMS_VGPU_COVERAGE="$PROFILE_RT"

# `%p` per process, and every raw file merged at the end.
#
# A single fixed path is what made this instrument report 0 % for the whole
# crate on a boot that demonstrably drew: `-C instrument-coverage` also
# instruments the crate's build scripts, they run during the build below, and
# each one writes to `LLVM_PROFILE_FILE` on exit. The last writer before QEMU's
# own atexit won, so the merge saw a build script's counters — all zero for
# every function in the device — and QEMU's were never read. The two are told
# apart by process id, and the sweep after the build removes the build's.
export LLVM_PROFILE_FILE="$OUT_DIR/reims-%p.profraw"

echo "runtime-dead: profile runtime $PROFILE_RT"

# A QEMU still holding :2222 makes the new boot fail its hostfwd and exit
# immediately, and every later step then measures — or SIGTERMs — the OLD,
# uninstrumented VM. Refuse rather than guess which one is ours.
#
# This used to be a `ps | grep | awk` pipeline, and the healthy case — grep
# finding nothing — is exit 1, which under `set -o pipefail` is the whole
# pipeline's status. `set -e` therefore aborted the run here whenever no VM was
# up: the script got past this guard only when the condition it refuses on was
# true, and exited after its one banner line with status 0 every other time. A
# function that emits nothing and returns 0 has no such edge.
stale="$(qemu_pids)"
if [ -n "$stale" ]; then
    echo "runtime-dead: a qemu-system-x86_64 is already running (pid $(echo "$stale" | tr '\n' ' '))." >&2
    echo "runtime-dead: it holds localhost:2222; this boot would fail and the run" >&2
    echo "runtime-dead: would measure that VM instead. Stop it first (kill by PID)." >&2
    exit 1
fi

# Built here rather than left to `boot-x86.sh`, so the raw files the build
# writes can be swept before the one that matters is produced. The boot's own
# build call then finds everything current and adds nothing.
echo "runtime-dead: building (instrumented) ..."
"$REPO_ROOT/scripts/qemu-build/qemu-build.sh" --target x86_64 --backend vulkan \
    > "$OUT_DIR/build.log" 2>&1 || {
    echo "runtime-dead: instrumented build failed; see $OUT_DIR/build.log" >&2
    exit 1
}
rm -f "$OUT_DIR"/reims-*.profraw

# The device appends to this and never truncates it, so a run that leaves the
# previous boot's records in place produces a log holding several boots of
# several builds. Nothing in a merged log says where one ends: `first_sight`
# latches per process, so the same refusal reappears once per boot and reads as
# a decoder failing thousands of times rather than as one line seen N times.
# A reader who ranks `reason=` on that file is ranking history.
rm -f /tmp/reims-vgpu-fail.log

echo "runtime-dead: booting (instrumented) ..."
"$REPO_ROOT/vm/boot-x86.sh" --device reims-vgpu-pci --testing > "$OUT_DIR/boot.log" 2>&1 &

# Provisional: enough to notice the boot never started, and to have something to
# kill on the refusal paths below. The pid the *measurement* is named after is
# re-sampled once the guest answers, because the boot script re-runs qemu-build
# and its four `-device help` probes are real qemu-system-x86_64 processes that
# exit before the device runs.
qemu_pid=""
for _ in $(seq 1 180); do
    qemu_pid="$(qemu_pids | head -1)"
    [ -n "$qemu_pid" ] && break
    sleep 2
done
if [ -z "$qemu_pid" ]; then
    echo "runtime-dead: QEMU never started; see $OUT_DIR/boot.log" >&2
    exit 1
fi

echo "runtime-dead: waiting for the guest ..."
guest_up=0
for _ in $(seq 1 120); do
    if ssh -o ConnectTimeout=4 -o BatchMode=yes macos-vm true 2>/dev/null; then
        guest_up=1
        break
    fi
    # A refused connection returns at once — `ConnectTimeout` only bounds a
    # connection that is being *accepted slowly*, and QEMU's hostfwd refuses
    # instantly until the guest is listening. Without this the whole loop ran in
    # under a second and reported a guest that had another twenty to go.
    sleep 2
done
if [ "$guest_up" -eq 0 ]; then
    echo "runtime-dead: guest never answered on macos-vm; see $OUT_DIR/boot.log" >&2
    # No profile is written for a boot that never ran the device, so kill rather
    # than pretend this run measured anything.
    kill -TERM "$qemu_pid" 2>/dev/null || true
    exit 1
fi

# A guest that answers SSH is a guest whose QEMU is the only one alive: the build
# finished long before, and qemu-build's `-device help` probes are gone. So this
# is the moment the boot's pid is unambiguous, and the name the profile is
# checked under below comes from here rather than from the race above.
mapfile -t live < <(qemu_pids)
if [ "${#live[@]}" -ne 1 ]; then
    echo "runtime-dead: expected exactly one qemu-system-x86_64 with the guest up," >&2
    echo "runtime-dead: found ${#live[@]} (${live[*]:-none}). Which of them wrote the" >&2
    echo "runtime-dead: measurement is then a guess, and guessing it wrong is how this" >&2
    echo "runtime-dead: reported a complete profile as missing. Refusing the run." >&2
    kill -TERM "$qemu_pid" 2>/dev/null || true
    exit 1
fi
qemu_pid="${live[0]}"
echo "runtime-dead: measuring QEMU pid $qemu_pid"

# SSH answering is not the app being ready. The boot reverts to a snapshot, so
# sshd is listening within seconds while the window server is still restoring
# sessions and $DRIVE_APP has no window yet — and the probe's first act is to
# read that window's frame. It then exits in about a second with "could not read
# ... window frame (pos '' size '')", which is not the "window never moved"
# refusal below but a failure to start at all.
#
# That cost a whole ten-minute run: the probe failed, `|| true` swallowed it,
# the boot was stopped seconds after reaching the desktop, and the only thing
# that reported a problem was the all-zero guard at the very end. So wait for
# the window to exist before driving, and make its absence fatal here.
echo "runtime-dead: waiting for $DRIVE_APP to present a window ..."
app_ready=0
for _ in $(seq 1 60); do
    if ssh -o ConnectTimeout=4 -o BatchMode=yes macos-vm \
        "osascript -e 'tell application \"$DRIVE_APP\" to activate' \
                   -e 'delay 1' \
                   -e 'tell application \"System Events\" to tell process \"$DRIVE_APP\" to get position of window 1'" \
        >/dev/null 2>&1; then
        app_ready=1
        break
    fi
    sleep 2
done
if [ "$app_ready" -eq 0 ]; then
    echo "runtime-dead: $DRIVE_APP never presented a window, so nothing would have" >&2
    echo "runtime-dead: driven the device and this run cannot measure it." >&2
    kill -TERM "$qemu_pid" 2>/dev/null || true
    exit 1
fi

# An undriven boot reaches the desktop and sits there, and its zeros are the
# idle device's. The probe refuses a verdict if the window never moved, so a run
# that produced no compositing cannot be mistaken for one that did.
echo "runtime-dead: driving the guest (${DRIVE_SECONDS}s, $DRIVE_APP) ..."
drive_ok=1
"$REPO_ROOT/scripts/window-drag-probe/window-drag-probe.sh" \
    --seconds "$DRIVE_SECONDS" --app "$DRIVE_APP" > "$OUT_DIR/drive.log" 2>&1 || drive_ok=0
tail -1 "$OUT_DIR/drive.log"
if [ "$drive_ok" -eq 0 ]; then
    echo "runtime-dead: the drag probe refused a verdict — see $OUT_DIR/drive.log." >&2
    echo "runtime-dead: coverage from an undriven boot is the idle device's, and its" >&2
    echo "runtime-dead: zeros read exactly like a kill list. Refusing the run." >&2
    kill -TERM "$qemu_pid" 2>/dev/null || true
    exit 1
fi

# An override that did not take is the worst outcome this flag has, because the
# run still succeeds: a full report, self-consistent counters, a directory named
# `-import-off`, and coverage of the import rails it was supposed to exclude.
# Diffed against the other run it then shows no copying rails at all, which
# reads as "the copying rails are dead" — the exact conclusion the flag exists
# to prevent. AGENTS.md gives the confirmation, so require it: `vk_caps` must
# report the env rung by name.
#
# The device appends to the fail log and the script truncated it before this
# boot, so a match here is this boot's.
if [ "$IMPORT_OFF" -eq 1 ]; then
    if grep -q 'host_pointer_import=disabled_by_env' /tmp/reims-vgpu-fail.log 2>/dev/null; then
        echo "runtime-dead: confirmed host_pointer_import=disabled_by_env"
    else
        echo "runtime-dead: --import-off was asked for and the device did not take it." >&2
        echo "runtime-dead: vk_caps never reported host_pointer_import=disabled_by_env, so" >&2
        echo "runtime-dead: this boot used the import and its coverage is the ordinary run's." >&2
        echo "runtime-dead: Reporting it as the copying rails' would say they are dead." >&2
        grep -o 'host_pointer_import=[a-z_]*' /tmp/reims-vgpu-fail.log 2>/dev/null |
            sort -u | sed 's/^/runtime-dead:   saw /' >&2 || true
        kill -TERM "$qemu_pid" 2>/dev/null || true
        exit 1
    fi
fi

# QMP `quit` first, signal only as a fallback.
#
# The profile is written by an `atexit` hook, so how QEMU leaves decides whether
# there is a measurement at all. `quit` sets the shutdown cause from inside the
# monitor and lets `main` return normally; a signal reaches the same shutdown
# through `qemu_system_killed`, and two driven runs that were SIGTERMed left a
# zero-byte profile while short-lived probe QEMUs in the same directory wrote
# complete ones. That is the difference the guard below kept catching.
qmp_sock="$(sed -n 's/.*qmp → \([^ ]*\.sock\).*/\1/p' "$OUT_DIR/boot.log" | tail -1)"
stopped=0
if [ -n "$qmp_sock" ] && [ -S "$qmp_sock" ]; then
    echo "runtime-dead: stopping QEMU (QMP quit — the profile is written at exit) ..."
    if python3 - "$qmp_sock" <<'PY'
import json, socket, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(10)
s.connect(sys.argv[1])
f = s.makefile("rwb")
f.readline()                                  # greeting
for cmd in ({"execute": "qmp_capabilities"}, {"execute": "quit"}):
    f.write((json.dumps(cmd) + "\r\n").encode())
    f.flush()
    f.readline()
PY
    then
        stopped=1
    else
        echo "runtime-dead: QMP quit did not go through; falling back to SIGTERM." >&2
    fi
fi
if [ "$stopped" -eq 0 ]; then
    echo "runtime-dead: stopping QEMU (SIGTERM) ..."
    kill -TERM "$qemu_pid" 2>/dev/null || true
fi
for _ in $(seq 1 60); do
    ps -p "$qemu_pid" >/dev/null 2>&1 || break
    sleep 2
done

mapfile -t raws < <(find "$OUT_DIR" -name 'reims-*.profraw' -size +0c)
if [ "${#raws[@]}" -eq 0 ]; then
    echo "runtime-dead: no profile data — QEMU did not exit cleanly" >&2
    exit 1
fi

# The boot's own QEMU is the only process in this directory whose counters are
# the measurement. Everything else that writes here — the build scripts, and the
# short-lived `qemu-system-x86_64` invocations the boot script makes to query
# devices and machines — exits before the device runs, so its records are
# present and its counts are all zero.
#
# That is why "no profile data" above cannot be the only guard, and why the
# README's claim that a bad toolchain leaves "a silently 0-byte file, which is
# how you will notice" was wrong. The 0-byte file *was* written, by the one
# process that mattered; `-size +0c` dropped it, four 4.3 MB all-zero probe
# dumps stayed, `merge -sparse` discarded their zero records, and what reached
# the report was six functions from a build script. The run then declared the
# entire crate — 3360 functions, TOTAL 0.00 % — never executed, on a boot that
# had drawn a desktop and driven Safari for 25 seconds. A list like that is
# indistinguishable from a kill list, so refuse by name.
#
# Size is the wrong test for it, though, and was only ever a proxy: a probe's
# dump is 4 310 352 bytes of records that are all misses, exactly as large as the
# measurement. `llvm-profdata show` prints `Maximum function count:`, which is 0
# for every one of them and 225 055 584 for a driven boot — so ask for the
# property that matters rather than for a file that exists.
boot_raw="$OUT_DIR/reims-$qemu_pid.profraw"
boot_max=0
[ -s "$boot_raw" ] && boot_max="$(llvm-profdata show "$boot_raw" 2>/dev/null |
    sed -n 's/^Maximum function count: *\([0-9]*\)$/\1/p' | head -1)"
if [ -z "$boot_max" ] || [ "$boot_max" -eq 0 ]; then
    echo "runtime-dead: the boot's own profile is missing, empty, or all zero ($boot_raw)." >&2
    echo "runtime-dead: QEMU pid $qemu_pid did not complete the atexit profile write," >&2
    echo "runtime-dead: so nothing here measured the device. The other raw files are" >&2
    echo "runtime-dead: build scripts and the boot script's own QEMU probes; their" >&2
    echo "runtime-dead: counters are all zero and merging them reports the whole crate" >&2
    echo "runtime-dead: as never-ran. Refusing to write a report." >&2
    exit 1
fi

echo "runtime-dead: merging ${#raws[@]} raw profile(s) ..."
llvm-profdata merge -sparse "${raws[@]}" -o "$OUT_DIR/merged.profdata"

# The coverage mapping lives in the linked QEMU binary, not the archive.
mapfile -t sources < <(find "$REPO_ROOT/crates/reims-vgpu/src" -name '*.rs')
llvm-cov report --instr-profile="$OUT_DIR/merged.profdata" "$QEMU_BIN" \
    "${sources[@]}" > "$OUT_DIR/by-file.txt" 2>/dev/null
llvm-cov export --instr-profile="$OUT_DIR/merged.profdata" "$QEMU_BIN" \
    --format=text "${sources[@]}" > "$OUT_DIR/export.json" 2>/dev/null

# Every function whose counter stayed at zero, with the file it lives in.
#
# The source list passed to `llvm-cov export` restricts the per-file summaries
# and NOT the function list — a bare walk of data[].functions reports every
# monomorphization in every dependency, which on this binary is 54 210 rather
# than the ~1 000 that are ours. Filter on the filenames each function actually
# spans.
python3 - "$OUT_DIR/export.json" "$OUT_DIR/never-ran.txt" <<'PY'
import json, sys

MARK = "/crates/reims-vgpu/src/"
data = json.load(open(sys.argv[1]))
rows, ours = [], 0
for export in data["data"]:
    for fn in export.get("functions", []):
        spans = [f for f in fn["filenames"] if MARK in f]
        if not spans:
            continue
        ours += 1
        if fn["count"]:
            continue
        files = ", ".join(sorted({f.split(MARK)[-1] for f in spans}))
        rows.append((files, fn["name"]))
rows.sort()

# A boot that answered on SSH and survived the drag probe's did-the-window-move
# verdict executed this device. So "every function in the crate is cold" is not
# a measurement of the guest, it is a measurement of a broken profile, and the
# difference is invisible once the list is on disk. Write nothing rather than
# hand the next reader a kill list with the whole crate on it.
if ours and not rows:
    sys.exit("runtime-dead: the export has no functions of ours at all — "
             "the coverage mapping does not match this binary")
if ours and len(rows) == ours:
    sys.exit(f"runtime-dead: all {ours} of our functions read zero, on a driven "
             "boot that reached the desktop. The profile is wrong, not the "
             "crate. Refusing to write a never-ran list.")

with open(sys.argv[2], "w") as out:
    for path, name in rows:
        out.write(f"{path}\t{name}\n")
print(f"runtime-dead: {len(rows)} of {ours} functions never ran "
      f"({ours - len(rows)} ran)")
PY

echo
echo "runtime-dead: per-file coverage  $OUT_DIR/by-file.txt"
echo "runtime-dead: never-ran list     $OUT_DIR/never-ran.txt"
echo "runtime-dead: driven-boot log    $OUT_DIR/drive.log"
echo
echo "A zero is a question, not a verdict. See $SCRIPT_DIR/README.md."
