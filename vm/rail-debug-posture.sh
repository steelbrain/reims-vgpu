#!/usr/bin/env bash
#
# vm/rail-debug-posture.sh — give a rail's OpenCore the debug posture that makes
# its guest kernel talk, and shorten the picker so a reboot is not 45 s of
# waiting.
#
# WHY THIS EXISTS. Every reading taken on a rail whose OpenCore has
# `Misc/Debug/Target = 3` and `Misc/Serial/Init = false` is taken blind: the
# serial log stops at `HANDOFF TO XNU` and XNU emits nothing for the rest of the
# boot. A guest that panics and reboots is then indistinguishable from one that
# shut down cleanly, because `-action reboot=shutdown` turns a panic into an
# ordinary QEMU exit. That is not a cosmetic gap — it is the difference between
# a diagnosis and a guess, and it cost one rail several sessions of them.
#
# Doing it by hand is a six-step dance (qcow2 -> raw, carve the ESP out, mcopy
# the plist off, edit, mcopy back, dd the partition back, raw -> qcow2) with a
# non-obvious trap in the middle, and it has to be done once per rail. So it is
# a script.
#
#   vm/rail-debug-posture.sh --rail macos-12
#   vm/rail-debug-posture.sh --rail macos-14 --label 2026-08-09-serial-debug
#   vm/rail-debug-posture.sh --rail macos-15 --dry-run
#
# It never edits a snapshot in place. Snapshot directories are read-only by
# contract — `vm/boot-x86.sh` reflink-clones whichever one `current` names and
# throws the clone away — so this reflinks the source snapshot into a new one,
# edits only `OpenCore.qcow2` there, and repoints `current` at it. The original
# stays byte-identical and is one `ln -sfn` away.
#
# WHAT IT SETS, and why each one.
#
#   Misc/Debug/Target       75   0x4B = enable | console | serial | file. Without
#                                the serial bit the log stops at the handoff.
#   Misc/Debug/AppleDebug   true forward XNU's own debug output to the log.
#   Misc/Debug/ApplePanic   true keep the panic log where it can be read back.
#   Misc/Debug/DisableWatchDog true a stalled boot must stay stalled and
#                                observable rather than being reset out from
#                                under the person watching it.
#   Misc/Serial/Init        true actually bring the serial port up. `Target`
#                                asking for serial output does nothing on its own.
#   Misc/Boot/Timeout       5    the picker, down from 45.
#   boot-args += debug=0x10A     DB_PRT | DB_KPRT | DB_LOG_PI_SCRN.
#   boot-args += keepsyms=1      symbolicate panics.
#
# **`debug` is deliberately 0x10A and not 0x10B.** The extra `0x01` is `DB_HALT`,
# and macOS honours it: the boot stops at `Waiting for remote debugger
# connection.` and never reaches the desktop. A rail was lost to exactly that
# once, by copying a working rail's boot-args wholesale. A bitmask is not one
# value, and the rails do not have to agree bit for bit. If an existing
# `boot-args` already carries a `debug=`, this script rewrites that one key and
# leaves every other argument alone rather than replacing the string.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAILS_DIR="$REPO_ROOT/vm/disks/rails"

RAIL=""
FROM=""
LABEL=""
DRY_RUN=0

die() { echo "rail-debug-posture: $*" >&2; exit 1; }
say() { echo "rail-debug-posture: $*"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --rail) shift; RAIL="${1:-}"; [ -n "$RAIL" ] || die "--rail needs a name"; shift ;;
    --rail=*) RAIL="${1#--rail=}"; shift ;;
    --from) shift; FROM="${1:-}"; [ -n "$FROM" ] || die "--from needs a snapshot label"; shift ;;
    --from=*) FROM="${1#--from=}"; shift ;;
    --label) shift; LABEL="${1:-}"; [ -n "$LABEL" ] || die "--label needs a name"; shift ;;
    --label=*) LABEL="${1#--label=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) sed -n '2,60p' "$0"; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done

[ -n "$RAIL" ] || die "--rail is required (one of: $(ls "$RAILS_DIR" 2>/dev/null | grep -v '^current$' | tr '\n' ' '))"
RAIL_DIR="$RAILS_DIR/$RAIL"
[ -d "$RAIL_DIR/snapshots" ] || die "no such rail: $RAIL_DIR/snapshots"

# Source snapshot: --from, else whatever `current` names.
if [ -n "$FROM" ]; then
  SRC="$RAIL_DIR/snapshots/$FROM"
else
  SRC="$(readlink -f "$RAIL_DIR/snapshots/current")"
fi
[ -d "$SRC" ] || die "source snapshot not found: $SRC"
[ -f "$SRC/OpenCore.qcow2" ] || die "no OpenCore.qcow2 in $SRC"

LABEL="${LABEL:-$(date +%Y-%m-%d)-serial-debug}"
DST="$RAIL_DIR/snapshots/$LABEL"

say "rail=$RAIL"
say "source=$SRC"
say "target=$DST"

if [ "$DRY_RUN" -eq 1 ]; then
  say "--dry-run: reading the existing posture and stopping"
fi

WORK="$(mktemp -d -t rail-debug-XXXXXX)"
# A failure partway through must not leave a half-written snapshot that `current`
# might later be pointed at, so the target is built under a temporary name and
# only moved into place once the image round-trips.
STAGE=""
cleanup() {
  rm -rf "$WORK"
  if [ -n "$STAGE" ] && [ -d "$STAGE" ]; then rm -rf "$STAGE"; fi
}
trap cleanup EXIT

# --- Carve the EFI system partition out of the OpenCore image ---------------
say "converting OpenCore.qcow2 -> raw ..."
qemu-img convert -O raw "$SRC/OpenCore.qcow2" "$WORK/oc.raw"

# The ESP is the first partition. Read where it starts and how long it is from
# the partition table rather than assuming the 1 MiB offset every OpenCore image
# happens to use — an assumption that is right until it is not, and then it
# corrupts an image instead of failing.
read -r PART_START PART_SECTORS < <(
  sfdisk -J "$WORK/oc.raw" 2>/dev/null \
    | python3 -c 'import json,sys; p=json.load(sys.stdin)["partitiontable"]["partitions"][0]; print(p["start"], p["size"])'
) || die "could not read the partition table out of OpenCore.qcow2"
SECTOR=512
PART_OFF=$((PART_START * SECTOR))
PART_LEN=$((PART_SECTORS * SECTOR))
say "ESP at offset $PART_OFF, length $PART_LEN"

dd if="$WORK/oc.raw" of="$WORK/esp.img" bs=1M skip=$((PART_OFF / 1048576)) \
   count=$(( (PART_LEN + 1048575) / 1048576 )) status=none
[ "$((PART_OFF % 1048576))" -eq 0 ] || die "ESP offset $PART_OFF is not 1 MiB aligned; refusing to guess"

# mtools refuses an image whose geometry it cannot verify, and this is a carved
# partition rather than a whole disk, so the check is the wrong one to apply.
export MTOOLS_SKIP_CHECK=1

say "reading EFI/OC/config.plist ..."
mcopy -i "$WORK/esp.img" ::/EFI/OC/config.plist "$WORK/config.plist" \
  || die "no EFI/OC/config.plist in this image"

# --- Edit the plist ---------------------------------------------------------
python3 - "$WORK/config.plist" "$WORK/config.new.plist" "$DRY_RUN" <<'PY'
import plistlib, sys

src, dst, dry = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
with open(src, "rb") as f:
    cfg = plistlib.load(f)

BOOT_ARGS_GUID = "7C436110-AB2A-4BBB-A880-FE41995C9F82"
changes = []

def setpath(container, path, value):
    """Set container[path[0]][path[1]]... = value, reporting the old value.

    Missing intermediate dictionaries are an error rather than something to
    create: an OpenCore config that has no `Misc/Debug` is not a config this
    script understands, and inventing the key would produce an image that boots
    and silently ignores it."""
    node = container
    for key in path[:-1]:
        if key not in node or not isinstance(node[key], dict):
            changes.append(("SKIP", "/".join(path), "missing section %r" % key))
            return
        node = node[key]
    leaf = path[-1]
    old = node.get(leaf, "<unset>")
    if old == value:
        changes.append(("same", "/".join(path), repr(value)))
        return
    changes.append(("set", "/".join(path), "%r -> %r" % (old, value)))
    if not dry:
        node[leaf] = value

setpath(cfg, ["Misc", "Debug", "Target"], 75)
setpath(cfg, ["Misc", "Debug", "AppleDebug"], True)
setpath(cfg, ["Misc", "Debug", "ApplePanic"], True)
setpath(cfg, ["Misc", "Debug", "DisableWatchDog"], True)
setpath(cfg, ["Misc", "Serial", "Init"], True)
setpath(cfg, ["Misc", "Boot", "Timeout"], 5)

# boot-args is a single space-separated string, and it belongs to the guest as
# much as to us — it can carry csr-active-config-adjacent tuning, vti=, tlbto_us=
# and whatever else a rail was provisioned with. Rewrite only the keys named
# here and leave every other token in place and in order.
WANT = {"debug": "0x10A", "keepsyms": "1"}
try:
    add = cfg["NVRAM"]["Add"][BOOT_ARGS_GUID]
except (KeyError, TypeError):
    add = None
if add is None:
    changes.append(("SKIP", "NVRAM/boot-args", "no %s section" % BOOT_ARGS_GUID))
else:
    raw = add.get("boot-args", "")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    toks, seen = [], set()
    for tok in raw.split():
        key = tok.split("=", 1)[0]
        if key in WANT:
            seen.add(key)
            new = "%s=%s" % (key, WANT[key])
            if tok != new:
                changes.append(("set", "boot-args/" + key, "%s -> %s" % (tok, new)))
            else:
                changes.append(("same", "boot-args/" + key, tok))
            toks.append(new)
        else:
            toks.append(tok)
    for key, val in WANT.items():
        if key not in seen:
            changes.append(("set", "boot-args/" + key, "<absent> -> %s=%s" % (key, val)))
            toks.append("%s=%s" % (key, val))
    joined = " ".join(toks)
    if not dry:
        add["boot-args"] = joined
    changes.append(("note", "boot-args", joined))

width = max(len(p) for _, p, _ in changes)
for kind, path, detail in changes:
    print("  %-4s %-*s  %s" % (kind, width, path, detail))

if any(k == "SKIP" for k, _, _ in changes):
    sys.exit("rail-debug-posture: refusing to write — a section this script needs is absent")

if not dry:
    with open(dst, "wb") as f:
        plistlib.dump(cfg, f)
PY

if [ "$DRY_RUN" -eq 1 ]; then
  say "--dry-run: nothing written"
  exit 0
fi

# --- Put it back ------------------------------------------------------------
say "writing config.plist back into the ESP ..."
mcopy -o -i "$WORK/esp.img" "$WORK/config.new.plist" ::/EFI/OC/config.plist

say "writing the ESP back into the raw image ..."
dd if="$WORK/esp.img" of="$WORK/oc.raw" bs=1M seek=$((PART_OFF / 1048576)) \
   conv=notrunc status=none

# --- Build the new snapshot -------------------------------------------------
STAGE="$DST.partial.$$"
rm -rf "$STAGE"
mkdir -p "$STAGE"
say "reflinking $SRC -> $STAGE ..."
for f in "$SRC"/*; do
  base="$(basename "$f")"
  [ "$base" = "OpenCore.qcow2" ] && continue
  cp --reflink=auto -p "$f" "$STAGE/$base"
done
qemu-img convert -O qcow2 "$WORK/oc.raw" "$STAGE/OpenCore.qcow2"
chmod 444 "$STAGE/OpenCore.qcow2"

# Verify the round-trip before anything points at it: the plist has to read back
# out of the image we just built, with the value we just set. A silent mcopy
# failure would otherwise produce a snapshot that boots the old posture and
# looks like the tool did nothing.
qemu-img convert -O raw "$STAGE/OpenCore.qcow2" "$WORK/verify.raw"
dd if="$WORK/verify.raw" of="$WORK/verify-esp.img" bs=1M skip=$((PART_OFF / 1048576)) \
   count=$(( (PART_LEN + 1048575) / 1048576 )) status=none
mcopy -i "$WORK/verify-esp.img" ::/EFI/OC/config.plist "$WORK/verify.plist" \
  || die "the rebuilt image has no EFI/OC/config.plist — not repointing current"
python3 - "$WORK/verify.plist" <<'PY'
import plistlib, sys
with open(sys.argv[1], "rb") as f:
    cfg = plistlib.load(f)
target = cfg["Misc"]["Debug"]["Target"]
serial = cfg["Misc"]["Serial"]["Init"]
timeout = cfg["Misc"]["Boot"]["Timeout"]
if target != 75 or serial is not True or timeout != 5:
    sys.exit("rail-debug-posture: read-back mismatch "
             "(Target=%r Serial/Init=%r Timeout=%r)" % (target, serial, timeout))
print("  verified Target=75 Serial/Init=True Timeout=5")
PY

mv "$STAGE" "$DST"
STAGE=""
ln -sfn "$LABEL" "$RAIL_DIR/snapshots/current"

say "done. $RAIL/snapshots/current -> $LABEL"
say "revert with: ln -sfn $(basename "$SRC") $RAIL_DIR/snapshots/current"
