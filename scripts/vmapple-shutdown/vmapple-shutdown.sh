#!/usr/bin/env bash
#
# scripts/vmapple-shutdown/vmapple-shutdown.sh
#
# Cleanly stop the running vmapple guest.
#
# Trigger a clean HALT from inside the guest (`shutdown -h now` over SSH) so macOS
# syncs and powers off, THEN force QEMU to exit. QMP `system_powerdown` does NOT
# work on vmapple: it is an ACPI power-button event and Apple's platform has no
# ACPI, so the macOS guest never sees it (verified: 90s no-op).
#
# On vmapple, macOS halting does NOT reliably make QEMU exit on its own — it
# sometimes does, sometimes leaves QEMU wedged (CPU halted, process alive). So
# after the in-guest halt we wait a short GRACE for a self-exit and otherwise
# QMP `quit` to terminate it. Because macOS has already synced/halted by then,
# that quit still lands a clean disk (and `vm/boot-arm64.sh --capture` captures on the
# rc=0 exit). Every boot reverts to a snapshot anyway, so a slightly-early quit is
# harmless.
#
# Escalation ladder:
#   1. SSH `sudo shutdown -h now` (macOS syncs + halts).
#   2. Wait up to GRACE for QEMU to self-exit; if it stays up (the common case) or
#      SSH was unreachable, QMP `quit`.
#   3. SIGKILL.
#
# SSH creds default to the guest convention (user=password=macosvm); the QMP
# fallback uses vm/<guest>/run/qmp.sock. Override via the env vars below.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
QMP="$REPO_ROOT/scripts/qmp/qmp.py"

GUEST_DIR="${GUEST_DIR:-$REPO_ROOT/vm/guest}"
RUN_DIR="${RUN_DIR:-$GUEST_DIR/run}"
GUEST_USER="${GUEST_USER:-macosvm}"
GUEST_PW="${GUEST_PW:-macosvm}"
SSH_PORT="${SSH_PORT:-2222}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/vmapple_guest}"
GRACE="${GRACE:-30}"          # seconds to wait for the clean in-guest halt before QMP quit

# Match the qemu ARGUMENT (`-M vmapple`) rather than a loose `.*vmapple` substring,
# so a launcher/ps/pgrep command line that merely mentions the pattern is not
# mistaken for a running guest. Only the real boot carries `-M vmapple` in its argv.
qemu_pids() { pgrep -f 'qemu-system-aarch64.*-M vmapple' 2>/dev/null; }

if [ -z "$(qemu_pids)" ]; then
  echo "vmapple-shutdown: no running vmapple QEMU — nothing to do."
  exit 0
fi

# --- 1. In-guest clean shutdown (PSCI SYSTEM_OFF → QEMU exits) -------------------
echo "vmapple-shutdown: in-guest 'shutdown -h now' over SSH (clean, up to ${GRACE}s) ..."
# The connection drops as macOS halts, so its exit status is meaningless — ignore
# it and judge success by whether QEMU actually exits. Password on stdin for sudo;
# the installed key handles SSH auth (falls through to the QMP fallback if not).
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ConnectTimeout=8 -o BatchMode=yes -p "$SSH_PORT" "$GUEST_USER@localhost" \
    "echo '$GUEST_PW' | sudo -S shutdown -h now" >/dev/null 2>&1 &
ssh_pid=$!

waited=0
while kill -0 "$ssh_pid" 2>/dev/null && [ "$waited" -lt "$GRACE" ]; do
  [ -z "$(qemu_pids)" ] && break
  sleep 1; waited=$((waited + 1))
done
if kill -0 "$ssh_pid" 2>/dev/null; then
  kill "$ssh_pid" 2>/dev/null || true
fi
wait "$ssh_pid" 2>/dev/null || true

while [ -n "$(qemu_pids)" ] && [ "$waited" -lt "$GRACE" ]; do
  sleep 3; waited=$((waited + 3))
done
if [ -z "$(qemu_pids)" ]; then
  echo "vmapple-shutdown: guest halted; QEMU exited cleanly (${waited}s)."
  exit 0
fi

# --- 2. Fallback: QMP quit ------------------------------------------------------
QMP_SOCK="${QMP_SOCK:-$RUN_DIR/qmp.sock}"
[ -S "$QMP_SOCK" ] || QMP_SOCK="$(ls -t "$RUN_DIR"/qmp-*.sock 2>/dev/null | head -1)"
if [ -n "${QMP_SOCK:-}" ] && [ -S "$QMP_SOCK" ]; then
  echo "vmapple-shutdown: QEMU still up — QMP quit via $QMP_SOCK ..."
  QMP_SOCK="$QMP_SOCK" python3 "$QMP" cmd quit >/dev/null 2>&1 || true
  sleep 3
fi

# --- 3. SIGKILL (last resort) ---------------------------------------------------
if [ -n "$(qemu_pids)" ]; then
  echo "vmapple-shutdown: SIGKILL ..."; qemu_pids | xargs kill -9 2>/dev/null; sleep 1
fi
[ -z "$(qemu_pids)" ] && echo "vmapple-shutdown: stopped." \
  || { echo "vmapple-shutdown: FAILED to stop QEMU" >&2; exit 1; }
