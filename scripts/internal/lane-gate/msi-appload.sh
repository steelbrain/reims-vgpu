#!/bin/bash
# The lane gate's app load, on the lane host, in the correctness boot after its measurements
# (msi-boot.sh, boot.env APPLOAD=1): rgk-ctx-slots' busy desktop (its appload.sh, HOG=0) on the
# gate's own boot. Every app in the guest's /System/Applications and /Applications (42 on the
# macOS 26 base disk, the skip list leaving out those that open no window or ask for a login), one
# at a time, then 7 WebGL and WebGPU pages in Safari from the guest's own localhost, with the
# guest's default route deleted first (no page leaves the guest).
#   msi-appload.sh OUT
# Env: PORT GUEST_USER GUEST_PASS (the guest's SSH), B (the boot's directory: launch.log), TOOLS
# (a copy of rgk-ctx-slots' appload tools: analyze-load.py, and guest/ with ctx.sh, streams.sh,
# serve.pl, mtlenum and www/), DUMPS (default 1: post ai.aneesiqbal.rgfx.vastats at the end, which
# a bundle with the measurement-only counters answers with each process's region blocks).
# Writes into OUT what appload.sh does (boot.log's step lines, bundle.log, kern.log, dmesg-rgfx.raw
# and dmesg-*.txt, ctx-*.txt, reports-*.txt, ips/, launch.log), analyze-load.py's summary.txt,
# and appload.txt for the gate's report, "KEY VALUE" lines:
#   apps N, procs_max N (the most processes holding a context), kext_region_refusals N,
#   context_refusals N, bundle_region_refusals N, crash_reports_apps N (new in the apps phase) and
#   crash_names, ws_blocks N, ws_kib N, ws_max_kib N, ws_reserve_mib N (the kext's last
#   "WindowServer's blocks" line), fatal_lines N
set -uo pipefail
O=${1:?usage: msi-appload.sh OUT}
DUMPS=${DUMPS:-1}
mkdir -p "$O"
: > "$O/boot.log"
log() { echo "$(date +%T) $*" | tee -a "$O/boot.log"; }
SOPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o PreferredAuthentications=password
  -o PubkeyAuthentication=no -o ConnectTimeout=5 -o LogLevel=ERROR)
G() { timeout 120 sshpass -p "$GUEST_PASS" ssh -n -p "$PORT" "${SOPTS[@]}" "$GUEST_USER@127.0.0.1" "$@"; }
SU="echo '$GUEST_PASS' | sudo -S -p ''"
GUI() { G "$SU launchctl asuser \$(id -u) sudo -u $GUEST_USER $*"; }

G "rm -rf ~/ctx; mkdir -p ~/ctx"
sshpass -p "$GUEST_PASS" scp -q -r -P "$PORT" "${SOPTS[@]}" "$TOOLS"/guest/* "$GUEST_USER@127.0.0.1:ctx/" || log "tools copy failed"
G "$SU route -n delete default" >> "$O/boot.log" 2>&1
G "$SU sh ~/ctx/streams.sh"
G "ls /Library/Logs/DiagnosticReports ~/Library/Logs/DiagnosticReports 2>/dev/null" > "$O/reports-before.txt"
G "cd ~/ctx && (nohup perl serve.pl 18480 ~/ctx/www ~/ctx/results.txt > serve.log 2>&1 &); sleep 1; cat serve.log"
ctxcount() { G "bash ~/ctx/ctx.sh | head -1; echo -n 'context refusals '; $SU dmesg | grep -c 'every context slot is held'; echo -n 'id chunks '; $SU dmesg | grep -o '[0-9]* of 255 held' | sort -n | tail -1" 2>/dev/null | tr '\n' ' '; }
# The kernel's ring is 128 KiB, which this load's sandbox lines fill in a minute or two: the kext's
# Rgfx and RGFX lines are taken from it after every step (analyze-load.py sorts them).
kring() {
  G "$SU dmesg" 2>/dev/null | perl -ne 'while (/\[ *(\d+\.\d+)\]: (R[Gg][Ff][Xx].*?)(?=\[ *\d+\.\d+\]: |Sandbox: |$)/g) { print "[$1]: $2\n" }' >> "$O/dmesg-rgfx.raw"
}
log "contexts idle: $(ctxcount)"
G "bash ~/ctx/ctx.sh" > "$O/ctx-idle.txt"
apps=$(G "ls /System/Applications /Applications 2>/dev/null | grep '\.app$'")
skip=" Mission Control.app Apps.app Launchpad.app Siri.app Time Machine.app iPhone Mirroring.app Passwords.app System Settings.app Safari.app "
n=0
while IFS= read -r a; do
  [[ -z $a || $skip == *" $a "* ]] && continue
  if G "test -d '/System/Applications/$a'"; then path="/System/Applications/$a"; else path="/Applications/$a"; fi
  GUI "open -a '$path'" > /dev/null 2>&1
  n=$((n + 1)); sleep 2
  log "opened $a: $(ctxcount)"
  kring
done <<< "$apps"
log "opened $n apps"
for u in webgl.html webgpu.html wgs/sample/computeBoids/index.html wgs/sample/deferredRendering/index.html \
         wgs/sample/shadowMapping/index.html wgs/sample/particles/index.html wgs/sample/texturedCube/index.html; do
  GUI "open -a Safari 'http://localhost:18480/$u'" > /dev/null 2>&1; sleep 8
  log "safari $u: $(ctxcount)"
  kring
done
sleep 10
log "contexts with every app: $(ctxcount)"
G "bash ~/ctx/ctx.sh" > "$O/ctx-all.txt"
[[ $DUMPS == 1 ]] && G "notifyutil -p ai.aneesiqbal.rgfx.vastats; sleep 4; $SU notifyutil -p ai.aneesiqbal.rgfx.vastats; sleep 4"
G "$SU dmesg" > "$O/dmesg-apps.txt"
kring
G "$SU cat /tmp/bundle.log" > "$O/bundle-apps.log"
G "ls /Library/Logs/DiagnosticReports ~/Library/Logs/DiagnosticReports 2>/dev/null" > "$O/reports-apps.txt"
napps=$(comm -13 <(sort "$O/reports-before.txt") <(sort "$O/reports-apps.txt") | grep -c '\.ips$')
log "apps phase: $(grep -c 'RGFX-REGION refused' "$O/bundle-apps.log") bundle region refusals, $(grep -c 'RGFX-NIL' "$O/bundle-apps.log") nil answers logged, $napps new crash reports"
G "$SU cat /tmp/bundle.log" > "$O/bundle.log"
G "$SU cat /tmp/kern.log" > "$O/kern.log"
G "$SU dmesg" > "$O/dmesg-end.txt"
kring
G "ls /Library/Logs/DiagnosticReports ~/Library/Logs/DiagnosticReports 2>/dev/null" > "$O/reports-after.txt"
mkdir -p "$O/ips"
comm -13 <(sort "$O/reports-before.txt") <(sort "$O/reports-after.txt") | grep -E '\.ips$' | while IFS= read -r f; do
  q=$(printf '%q' "$f")
  G "cat /Library/Logs/DiagnosticReports/$q 2>/dev/null || cat ~/Library/Logs/DiagnosticReports/$q" > "$O/ips/$f" 2> /dev/null
  [[ -s $O/ips/$f ]] || rm -f "$O/ips/$f"
done
log "new crash reports: $(ls "$O/ips" | tr '\n' ',')"
cp -f "$B/launch.log" "$O/launch.log"
log "host fatal lines: $(grep -c 'reims-vgpu: fatal' "$O/launch.log")"
python3 -I "$TOOLS/analyze-load.py" "$O" > "$O/summary.txt" 2>&1

# The gate's numbers, from the summary and the step lines.
{
  echo "apps $n"
  echo "procs_max $(sed -n 's/.*most processes with a context \([0-9]*\).*/\1/p' "$O/summary.txt" | head -1)"
  echo "kext_region_refusals $(sed -n 's/^ *region refusals: \([0-9]*\).*/\1/p' "$O/summary.txt" | head -1)"
  echo "context_refusals $(sed -n 's/^ *every context slot is held: \([0-9]*\).*/\1/p' "$O/summary.txt" | head -1)"
  echo "bundle_region_refusals $(grep -c 'RGFX-REGION refused' "$O/bundle-apps.log")"
  echo "crash_reports_apps $napps"
  echo "crash_names $(comm -13 <(sort "$O/reports-before.txt") <(sort "$O/reports-apps.txt") | grep '\.ips$' |
    sed 's/-[0-9]\{4\}-[0-9][0-9]-[0-9][0-9]-.*//' | sort | uniq -c |
    awk '{n = $1; sub(/^ *[0-9]+ /, ""); printf "%s%s x%s", (NR > 1 ? "," : ""), $0, n}')"
  # "RgfxAccelerator: WindowServer's blocks: 1397 (138924 KiB), at most 1397 (138924 KiB); its
  # reserve: 1024 entries, 160 MiB; ..."
  sed -n "s/.*WindowServer's blocks: \([0-9]*\) (\([0-9]*\) KiB), at most \([0-9]*\) (\([0-9]*\) KiB); its reserve: \([0-9]*\) entries, \([0-9]*\) MiB.*/ws_blocks \1\nws_kib \2\nws_max_blocks \3\nws_max_kib \4\nws_reserve_entries \5\nws_reserve_mib \6/p" \
    "$O/summary.txt" | tail -6
  echo "fatal_lines $(grep -c 'reims-vgpu: fatal' "$O/launch.log")"
} > "$O/appload.txt"
log "app load: $(tr '\n' ' ' < "$O/appload.txt")"
