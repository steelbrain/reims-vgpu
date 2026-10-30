#!/bin/bash
# A copy of a VM's OpenCore image with kexts added: each bundle in
# EFI/OC/Kexts, with a Kernel > Add entry made as reims-util makes its own
# (from the image's first entry: x86_64, its executable and Info.plist, and the
# Darwin bounds of the loader bridge's set, macOS 26.4 to 26.x). For iterating
# on the guest display kext (guest/kext) and the loader that binds it, without
# installing anything inside the guest.
#
#   scripts/internal/opencore-add-kexts.sh SRC OUT KEXT...
#
# SRC is the VM's image, qcow2 or raw, and is only read. OUT is the copy, qcow2
# for a .qcow2 name and raw otherwise; it must not exist, so SRC is never
# written. The KEXTs go after the image's own entries, each after the given
# kexts its OSBundleLibraries names. A kext the image already carries (by
# bundle name) is replaced, files and entry, in its entry's place, or moves to
# the end where a given kext it links against comes later. Prints the
# Kernel > Add entries of OUT.
#
# Needs qemu-img, mtools and python3. It edits a raw copy in a temporary
# directory, so nothing is mounted.
set -euo pipefail

usage() { echo "usage: scripts/internal/opencore-add-kexts.sh SRC OUT KEXT..." >&2; exit 2; }
[[ $# -ge 3 ]] || usage
src=$1 out=$2
shift 2
for tool in qemu-img mcopy mdir mdeltree python3; do
  command -v "$tool" > /dev/null || { echo "opencore-add-kexts: needs $tool (qemu-img; mtools)" >&2; exit 2; }
done
[[ -f $src ]] || { echo "opencore-add-kexts: no image $src" >&2; exit 2; }
[[ ! -e $out ]] || { echo "opencore-add-kexts: $out exists; give a new path for the copy" >&2; exit 2; }
for k in "$@"; do
  [[ -f $k/Contents/Info.plist ]] || { echo "opencore-add-kexts: $k is not a kext bundle" >&2; exit 2; }
done

tmp=$(mktemp -d "${TMPDIR:-/tmp}/opencore-add-kexts.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
qemu-img convert -O raw "$src" "$tmp/disk.raw"

# The ESP: the GPT's first partition, in 512-byte sectors as reims-util
# writes them.
offset=$(python3 -I - "$tmp/disk.raw" << 'EOF'
import struct, sys
with open(sys.argv[1], "rb") as f:
    f.seek(512)
    header = f.read(92)
    if header[:8] != b"EFI PART":
        sys.exit("opencore-add-kexts: the image has no GPT")
    entries_lba, = struct.unpack_from("<Q", header, 72)
    f.seek(entries_lba * 512)
    first_lba, = struct.unpack_from("<Q", f.read(128), 32)
print(first_lba * 512)
EOF
)
export MTOOLS_SKIP_CHECK=1
img="$tmp/disk.raw@@$offset"
mcopy -i "$img" ::/EFI/OC/config.plist "$tmp/config.plist"

# The bundles, without Finder's files.
mkdir "$tmp/kexts"
for k in "$@"; do
  name=$(basename "$k")
  cp -R "$k" "$tmp/kexts/$name"
  find "$tmp/kexts/$name" -name '.*' -exec rm -rf {} +
  if mdir -i "$img" "::/EFI/OC/Kexts/$name" > /dev/null 2>&1; then
    mdeltree -i "$img" "::/EFI/OC/Kexts/$name"
  fi
  mcopy -s -i "$img" "$tmp/kexts/$name" ::/EFI/OC/Kexts/
done

python3 -I - "$tmp/config.plist" "$tmp/kexts" "$@" << 'EOF'
import copy, os, plistlib, sys

config_path, staged = sys.argv[1], sys.argv[2]
with open(config_path, "rb") as f:
    config = plistlib.load(f)
add = config["Kernel"]["Add"]
if not add:
    sys.exit("opencore-add-kexts: the image's Kernel > Add is empty: no entry to take keys from")
template = add[0]

kexts = []
for path in sys.argv[3:]:
    name = os.path.basename(path.rstrip("/"))
    with open(os.path.join(staged, name, "Contents/Info.plist"), "rb") as f:
        info = plistlib.load(f)
    kexts.append((name, info))

# Each after the given kexts it links against, otherwise in the order given.
ids = {info["CFBundleIdentifier"] for _, info in kexts}
placed, ordered, left = set(), [], list(kexts)
while left:
    for i, (name, info) in enumerate(left):
        libs = set(info.get("OSBundleLibraries", {})) & ids
        if libs <= placed:
            break
    else:
        sys.exit("opencore-add-kexts: the kexts link against each other in a cycle")
    name, info = left.pop(i)
    placed.add(info["CFBundleIdentifier"])
    ordered.append((name, info))

for name, info in ordered:
    entry = copy.deepcopy(template)
    executable = info.get("CFBundleExecutable")
    entry.update({
        "Arch": "x86_64",
        "BundlePath": name,
        "Comment": f"{name[:-5] if name.endswith('.kext') else name}, added for development",
        "Enabled": True,
        "ExecutablePath": f"Contents/MacOS/{executable}" if executable else "",
        "MaxKernel": "25.99.99",
        "MinKernel": "25.4.0",
        "PlistPath": "Contents/Info.plist",
    })
    at = next((i for i, e in enumerate(add) if e.get("BundlePath") == name), None)
    # A kext replaced in its entry's place must still come after the given
    # kexts it links against, which may be new and so appended.
    libs = set(info.get("OSBundleLibraries", {})) & ids
    after = [i for i, e in enumerate(add) if e.get("BundlePath") in
             {n for n, i2 in ordered if i2["CFBundleIdentifier"] in libs}]
    if at is not None and after and max(after) > at:
        del add[at]
        at = None
    if at is None:
        add.append(entry)
    else:
        add[at] = entry

with open(config_path, "wb") as f:
    plistlib.dump(config, f)
for e in add:
    print(f"  {e['BundlePath']:<40} {e['ExecutablePath'] or '(no code)':<44} "
          f"{e['MinKernel'] or '-'}..{e['MaxKernel'] or '-'}{'' if e['Enabled'] else ' (disabled)'}")
EOF
mcopy -o -i "$img" "$tmp/config.plist" ::/EFI/OC/config.plist

case $out in
  *.qcow2) format=qcow2 ;;
  *) format=raw ;;
esac
qemu-img convert -O "$format" "$tmp/disk.raw" "$tmp/out"
mv "$tmp/out" "$out"
echo "opencore-add-kexts: wrote $out ($format)"
