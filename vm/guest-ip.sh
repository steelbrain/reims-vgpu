#!/usr/bin/env bash
#
# vm/guest-ip.sh — print the bridged guest's IP address.
#
# WHY THIS EXISTS. With NET=bridge (boot-x86.sh's default) the guest is a peer on
# the bridge's subnet and there is no hostfwd, so `localhost:2222` reaches
# nothing. The guest's address is assigned by the bridge network's DHCP server
# and is therefore not knowable from this repo — but the guest MAC IS pinned
# (GUEST_MAC in boot-x86.sh, load-bearing for exactly this reason), so the
# address is a lookup rather than a guess.
#
# THREE SOURCES, tried in order, because they fail in different places and the
# first one that answers is the cheapest:
#
#   1. libvirt's dnsmasq lease file, /var/lib/libvirt/dnsmasq/<bridge>.status.
#      World-readable JSON, needs no privilege and no libvirt connection, and
#      holds the lease the moment dnsmasq grants it. This is the normal answer.
#   2. `virsh net-dhcp-leases`. Same data through libvirt, which is worth having
#      as a second opinion when the status file is unreadable (a distro that
#      tightens its mode) or when the bridge belongs to a network whose lease
#      file is named differently.
#   3. The host's own neighbour table for the bridge. This is the only source
#      that works for a guest with a STATIC address, and the only one that keeps
#      working on a bridge with no DHCP server at all. It answers only after the
#      guest has put a packet on the wire, which for macOS means after it has
#      finished its own network setup.
#
# None of them is a side channel: each is a record of an address the guest itself
# claimed, keyed by the MAC this repo pinned. Nothing here infers an address from
# timing, ordering, or a scan.
#
#   vm/guest-ip.sh                  # print the IP, or fail after --wait seconds
#   vm/guest-ip.sh --wait 180       # bound the wait differently (default 120)
#   vm/guest-ip.sh --wait 0         # answer now or fail now, for a status check
#   vm/guest-ip.sh --ssh-target     # print user@ip, ready to hand to ssh
#
# Environment: BRIDGE (default virbr0), GUEST_MAC, REIMS_GUEST_USER, LIBVIRT_NET,
# and LEASE_STATUS to name a dnsmasq status file directly — which is what a
# bridge served by a dnsmasq that libvirt does not own needs.
#
# Exits 0 having printed one address, or 3 with nothing on stdout. Exit 3 rather
# than 1 so a harness can tell "no lease yet" from "this script is broken".
set -euo pipefail

BRIDGE="${BRIDGE:-virbr0}"
GUEST_MAC="${GUEST_MAC:-52:54:00:c9:18:27}"
GUEST_USER="${REIMS_GUEST_USER:-aneesiqbal}"
LIBVIRT_NET="${LIBVIRT_NET:-default}"
WAIT_SECONDS=120
WANT_SSH_TARGET=0

die() { echo "guest-ip: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --wait) shift; WAIT_SECONDS="${1:-}"; [ -n "$WAIT_SECONDS" ] || die "--wait needs seconds"; shift ;;
    --wait=*) WAIT_SECONDS="${1#--wait=}"; shift ;;
    --ssh-target) WANT_SSH_TARGET=1; shift ;;
    --bridge) shift; BRIDGE="${1:-}"; [ -n "$BRIDGE" ] || die "--bridge needs a name"; shift ;;
    --bridge=*) BRIDGE="${1#--bridge=}"; shift ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

case "$WAIT_SECONDS" in
  ''|*[!0-9]*) die "--wait takes whole seconds, got '$WAIT_SECONDS'" ;;
esac

# dnsmasq writes the MAC lowercase; a MAC given in either case must match.
MAC_LC="$(printf '%s' "$GUEST_MAC" | tr 'A-F' 'a-f')"

# --- Source 1: the lease file ------------------------------------------------
# The file is a JSON array of objects carrying "mac-address" and "ip-address".
# Parsed with python3 rather than a regex because the two fields are not
# adjacent and their order is not promised; a regex that assumes either would
# return the wrong guest's address on a bridge with more than one VM, which is
# the failure mode that reads as working.
from_lease_file() {
  local status="${LEASE_STATUS:-/var/lib/libvirt/dnsmasq/$BRIDGE.status}"
  [ -r "$status" ] || return 1
  command -v python3 >/dev/null 2>&1 || return 1
  python3 - "$status" "$MAC_LC" <<'PY' 2>/dev/null
import json, sys
try:
    with open(sys.argv[1]) as f:
        leases = json.load(f)
except Exception:
    sys.exit(1)
want = sys.argv[2].lower()
for lease in leases if isinstance(leases, list) else []:
    if str(lease.get("mac-address", "")).lower() == want:
        ip = lease.get("ip-address")
        if ip:
            print(ip)
            sys.exit(0)
sys.exit(1)
PY
}

# --- Source 2: libvirt ---------------------------------------------------------
from_virsh() {
  command -v virsh >/dev/null 2>&1 || return 1
  # Columns: Expiry | MAC | Protocol | IP/prefix | Hostname | ClientID.
  # The address carries a prefix length that ssh must not see.
  virsh --connect qemu:///system net-dhcp-leases "$LIBVIRT_NET" 2>/dev/null \
    | awk -v mac="$MAC_LC" 'tolower($2) == mac { split($5, a, "/"); if (a[1] != "") { print a[1]; exit } }' \
    | grep . || return 1
}

# --- Source 3: the neighbour table ---------------------------------------------
# Works for a static guest and on a DHCP-less bridge, but only once the guest has
# sent something. `ip neigh` prints "<ip> dev <br> lladdr <mac> <state>"; a
# FAILED or INCOMPLETE entry is a MAC we asked about and got no answer for, so
# those states are excluded rather than reported as an address.
from_neighbours() {
  ip neigh show dev "$BRIDGE" 2>/dev/null \
    | awk -v mac="$MAC_LC" '
        tolower($3) == "lladdr" && tolower($4) == mac {
          state = $NF
          if (state != "FAILED" && state != "INCOMPLETE") { print $1; exit }
        }' \
    | grep . || return 1
}

resolve_once() {
  from_lease_file || from_virsh || from_neighbours
}

IP=""
deadline=$(( $(date +%s) + WAIT_SECONDS ))
while :; do
  if IP="$(resolve_once)" && [ -n "$IP" ]; then break; fi
  IP=""
  [ "$(date +%s)" -lt "$deadline" ] || break
  sleep 3
done

if [ -z "$IP" ]; then
  {
    echo "guest-ip: no address for $GUEST_MAC on $BRIDGE after ${WAIT_SECONDS}s."
    if ! ip link show "$BRIDGE" >/dev/null 2>&1; then
      echo "guest-ip: there is no bridge '$BRIDGE' — start libvirt's network:"
      echo "guest-ip:   sudo virsh --connect qemu:///system net-start $LIBVIRT_NET"
    else
      echo "guest-ip: the bridge exists, so the guest has not taken a lease yet."
      echo "guest-ip: a macOS guest DHCPs late in boot — wait for the desktop, then retry."
      echo "guest-ip: if it never does, confirm the boot ran with NET=bridge (NET=user has no lease;"
      echo "guest-ip:   its guest is reachable at localhost:\${SSH_PORT:-2222} instead)."
    fi
  } >&2
  exit 3
fi

if [ "$WANT_SSH_TARGET" -eq 1 ]; then
  printf '%s@%s\n' "$GUEST_USER" "$IP"
else
  printf '%s\n' "$IP"
fi
