#!/bin/bash
# The lane gate's guest side (scripts/internal/lane-gate.sh): run in the guest over SSH, as the
# autologin user, from the directory the gate copied it into with its tools:
#   GATE_PASS=<the user's password> bash guest-run.sh DIR [PHASE...]
# DIR holds sweep, bench, present_probe and decode_md5 (x86_64 builds of guest/bundle/tools),
# webgl.html, webgl-serve.pl, h264.list and hevc.list (the decode sets: their streams, in corpus/
# beside each one's reference .md5, with the timed passes after a stream whose rate is measured).
#
# It prints what it finds as lines the gate's report reads ("GATE item NAME PASS|FAIL|SKIP
# detail", "GATE metric NAME VALUE"), each phase between "GATE phase NAME start" and "... end",
# and the tools' own result lines ("sweep: ...", "bench: ...", "present_probe: ...",
# "decode_md5: ...") as they come. The phases run in the order given (default: info idle bench
# present sweep decode webgl final); the gate's correctness boot runs info sweep decode webgl
# present final, its bench boot info idle bench present decodefps final. decode checks every
# stream's MD5s, decodefps also times the streams the lists time (decode_md5 -l), and present
# runs present_probe for GATE_PRESENT_S seconds (default 10) each way. Each tool runs under an
# alarm, so a hang ends as a FAIL line.
# sudo (for launchctl asuser, which runs the on-screen tools in the logged-in user's window
# server session) takes GATE_PASS.
set -u
D=${1:?usage: guest-run.sh DIR [PHASE...]}
shift
cd "$D" || exit 2
PHASES=${*:-info idle bench present sweep decode webgl final}
: "${GATE_PASS:?GATE_PASS must hold the password of the user}"
ME=$(id -un) UIDN=$(id -u)

item() { echo "GATE item $*"; }
metric() { echo "GATE metric $1 $2"; }
phase() { echo "GATE phase $1 $2 $(date +%s)"; }
# CMD under an alarm of SECONDS: a hang is killed and shows as exit 142.
limit() { perl -e 'alarm shift @ARGV; exec @ARGV or exit 127' "$@"; }
S() { echo "$GATE_PASS" | sudo -S -p '' "$@"; }
# CMD in the logged-in user's GUI session, as that user.
gui() { S launchctl asuser "$UIDN" sudo -u "$ME" "$@"; }
reports() { ls /Library/Logs/DiagnosticReports "$HOME/Library/Logs/DiagnosticReports" 2>/dev/null | grep -E '\.(ips|panic)$' | sort -u; }
wspid() { pgrep -x WindowServer | head -1; }

run_info() {
	reports > "$D/reports-before.txt"
	wspid > "$D/ws-pid.txt"
	echo "GATE info os $(sw_vers -productVersion) $(sw_vers -buildVersion)"
	local csr; csr=$(csrutil status 2>&1)
	case $csr in
	*"enabled."*) item sip PASS "$csr" ;;
	*) item sip FAIL "$csr" ;;
	esac
	local kexts; kexts=$(kmutil showloaded 2>/dev/null | grep -o -E 'com\.steelbrain\.[A-Za-z.]+' | sort -u | tr '\n' ' ')
	case $kexts in
	*com.steelbrain.ReimsGraphics\ *) item kexts PASS "$kexts" ;;
	*) item kexts FAIL "ReimsGraphics is not loaded: $kexts" ;;
	esac
	# The loader's log (ApvLoadLog) says whether every site of ReimsGraphics resolved.
	local ld bind
	ld=$(ioreg -l -w0 -c AppleParavirtGPULoader 2>/dev/null | grep -o -E 'ReimsGraphics:[^"]{0,160}')
	bind=$(echo "$ld" | grep -o -E 'ReimsGraphics: all [0-9]+ sites resolved.{0,40}' | head -1)
	if [ -n "$bind" ]; then item loader-bind PASS "$bind"; else
		item loader-bind FAIL "$(echo "$ld" | head -2 | tr '\n' ' ')"
	fi
	local b=/Library/GPUBundles/reims-graphics.bundle
	if codesign -v --strict "$b" 2>/dev/null; then
		item bundle PASS "$(shasum "$b/Contents/MacOS/reims-graphics" | cut -c1-16) $(codesign -dvvv "$b" 2>&1 | grep -E '^CDHash=' | head -1)"
		echo "GATE info bundle-sha $(shasum "$b/Contents/MacOS/reims-graphics" | cut -d' ' -f1)"
	else
		item bundle FAIL "$(ls -la /Library/GPUBundles 2>&1 | tr '\n' ' ')"
	fi
	local dev; dev=$(limit 30 ./bench -l 2>&1 | grep -v '^[0-9-]* [0-9:.]* .*RGFX' | head -5 | tr '\n' ' ')
	case $dev in
	*Apple\ Paravirtual*) item metal-device FAIL "$dev" ;;
	*bench:\ device*) item metal-device PASS "$dev" ;;
	*) item metal-device FAIL "no Metal device: $dev" ;;
	esac
	local desk; desk="$(wspid) $(pgrep -x Dock | head -1) $(pgrep -x Finder | head -1)"
	set -- $desk
	if [ $# -eq 3 ]; then item desktop PASS "WindowServer $1, Dock $2, Finder $3"; else item desktop FAIL "WindowServer/Dock/Finder: $desk"; fi
	run_memory
	echo "GATE info uptime $(uptime | tr -s ' ')"
}

# The GPU's memory as the guest's frameworks state it: system_profiler's "VRAM" (the kext's
# VRAM,totalMB) and the Metal device's recommendedMaxWorkingSetSize, hasUnifiedMemory and
# dedicatedMemorySize. A Mac's GPU states VRAM as its working set, and dedicatedMemorySize as
# the working set where it has memory of its own, else 0 (guest/kext/src/RgfxMemorySize.hpp).
run_memory() {
	local vram mem ws uni ded gb
	vram=$(system_profiler SPDisplaysDataType 2>/dev/null | grep -m1 -E '^ *VRAM' | sed 's/^ *//')
	mem=$(limit 30 osascript -l JavaScript -e 'ObjC.import("Metal"); var d = $.MTLCreateSystemDefaultDevice();
		[d.recommendedMaxWorkingSetSize, d.hasUnifiedMemory ? 1 : 0,
		 d.respondsToSelector("dedicatedMemorySize") ? d.dedicatedMemorySize : -1].join(" ")' 2>/dev/null)
	echo "GATE info memory vram=\"$vram\" metal=\"$mem\""
	set -- $mem
	ws=${1:-0} uni=${2:-} ded=${3:--1}
	gb=$(echo "$vram" | sed -n 's/.*: *\([0-9][0-9]*\) GB.*/\1/p')
	if [ -z "$vram" ] || [ "$ws" = 0 ] || [ "$ded" = -1 ]; then
		item memory FAIL "VRAM \"$vram\", Metal \"$mem\" (working set, unified, dedicatedMemorySize)"
	elif [ "$uni" = 1 ] && [ "$ded" != 0 ]; then
		item memory FAIL "unified memory with dedicatedMemorySize $ded"
	elif [ "$uni" = 0 ] && [ "$ded" != "$ws" ]; then
		item memory FAIL "dedicatedMemorySize $ded is not the working set $ws"
	elif [ -n "$gb" ] && [ "$(perl -e 'printf "%.0f", $ARGV[0] / 2**30' "$ws")" != "$gb" ]; then
		item memory FAIL "VRAM $gb GB is not the working set $ws"
	else
		item memory PASS "$vram; recommendedMaxWorkingSetSize $ws, hasUnifiedMemory $uni, dedicatedMemorySize $ded"
	fi
}

run_idle() { sleep 10; }

run_bench() {
	# The 5080's clocks ramp only after up to 8 s of full load: warm the GPU first, so the cases
	# run at the clocks they reach.
	limit 60 ./bench -r 1 -m 8 frag.alu.2160p > "$D/bench-warm.txt" 2>&1
	limit 300 ./bench -m 3 2>&1 | grep -E '^bench: |^# start'
	echo "rc bench ${PIPESTATUS[0]}"
	gui perl -e 'alarm shift @ARGV; exec @ARGV' 60 "$D/bench" -t 10 onscreen 2>&1 | grep -E '^bench: |^# start'
}

# present_probe full-screen, then in a 1280 x 800 window, each its own sub-phase, so the host's
# display updates can be counted in each.
run_present() {
	phase present.fullscreen start
	gui perl -e 'alarm shift @ARGV; exec @ARGV' 40 "$D/present_probe" -t "${GATE_PRESENT_S:-10}" 2>&1 | grep '^present_probe:' | sed 's/^present_probe:/present_probe: fullscreen/'
	phase present.fullscreen end
	phase present.windowed start
	gui perl -e 'alarm shift @ARGV; exec @ARGV' 40 "$D/present_probe" -t "${GATE_PRESENT_S:-10}" -w 1280x800 2>&1 | grep '^present_probe:' | sed 's/^present_probe:/present_probe: windowed/'
	phase present.windowed end
}

run_sweep() {
	limit 420 ./sweep 2>&1 | grep '^sweep: '
	echo "rc sweep ${PIPESTATUS[0]}"
}

# decode: every stream of both sets, MD5s only; decodefps: the timed streams, MD5s and timed passes.
run_decode() {
	local set s loops
	for set in h264 hevc; do
		[ -f "$set.list" ] || { item "decode-$set" SKIP "no $set.list"; continue; }
		while read -r s loops; do
			case $s in ''|'#'*) continue ;; esac
			if [ -n "${TIMED:-}" ]; then [ -n "$loops" ] || continue; else loops=0; fi
			limit 90 ./decode_md5 -l "$loops" "corpus/$s" 2>&1 | grep '^decode_md5: ' | sed "s/^decode_md5: /decode_md5: $set /"
		done < "$set.list"
	done
}
run_decodefps() { TIMED=1 run_decode; }

run_webgl() {
	local port=8765 res="$D/webgl-result.json"
	rm -f "$res"
	perl "$D/webgl-serve.pl" "$port" "$D" "$res" 60 &
	local srv=$!
	sleep 1
	gui open -a Safari "http://127.0.0.1:$port/webgl.html"
	wait "$srv"
	if [ -s "$res" ]; then
		echo "GATE webgl $(cat "$res")"
	else
		echo 'GATE webgl {"ok": false, "error": "no result from the page in 60 s"}'
	fi
	gui killall Safari > /dev/null 2>&1
	sleep 2
}

run_final() {
	local before now new
	before=$(cat "$D/ws-pid.txt" 2>/dev/null) now=$(wspid)
	if [ -n "$now" ] && [ "$now" = "$before" ]; then item windowserver PASS "pid $now throughout"; else item windowserver FAIL "pid $before at start, ${now:-none} now"; fi
	local desk; desk="$(wspid) $(pgrep -x Dock | head -1) $(pgrep -x Finder | head -1)"
	set -- $desk
	if [ $# -eq 3 ]; then item desktop-end PASS "WindowServer $1, Dock $2, Finder $3"; else item desktop-end FAIL "WindowServer/Dock/Finder: $desk"; fi
	new=$(reports | comm -13 "$D/reports-before.txt" - | tr '\n' ' ')
	echo "GATE crashes ${new:-none}"
}

for p in $PHASES; do
	phase "$p" start
	"run_$p"
	phase "$p" end
done
