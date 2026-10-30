#!/usr/bin/env python3
"""Checks that the SDK's IOFramebuffer is the guest's.

    python3 -I tools/check-layout.py KDK_DSYM OBJECT

The kext is built against a newer SDK's headers than the macOS it runs on, and
inherits IOFramebuffer from the guest's IOGraphicsFamily. That holds only if
both agree on the class's size, where the subclass's fields start, and on the
vtable slot of every virtual method IOFramebuffer declares, which is the slot
the guest calls for it.

KDK_DSYM is the guest release's IOGraphicsFamily.kext.dSYM (a KDK's), OBJECT
the kext's source compiled with -g -fstandalone-debug. Both are read through
dwarfdump: IOFramebuffer's byte size, and each of its virtual methods' linkage
name and vtable slot. Exits 1 on any difference, naming it.
"""

import re
import subprocess
import sys

CLASS = "IOFramebuffer"


def layout(path):
    """The class's size and its virtual methods' slots, from the first full
    definition of it in path's DWARF."""
    out = subprocess.run(
        ["dwarfdump", "--arch", "x86_64", "--name", CLASS, "--show-children", path],
        check=True, capture_output=True, text=True).stdout
    # Each match starts at column 0; take the first that is a definition.
    for match in re.split(r"\n(?=0x[0-9a-f]+: DW_TAG)", out):
        head = match.split("\n", 3)
        if "DW_TAG_class_type" not in head[0]:
            continue
        size = re.search(r"^\s*DW_AT_byte_size\s*\((0x[0-9a-f]+)\)", match, re.M)
        if not size:
            continue
        slots = {}
        for sub in match.split("DW_TAG_subprogram")[1:]:
            name = re.search(r'DW_AT_linkage_name\s*\("([^"]+)"\)', sub)
            slot = re.search(r"DW_AT_vtable_elem_location\s*\(DW_OP_constu (0x[0-9a-f]+)\)", sub)
            if name and slot:
                slots[name.group(1)] = int(slot.group(1), 16)
        return int(size.group(1), 16), slots
    raise SystemExit(f"check-layout: no definition of {CLASS} in {path}")


def main(argv):
    if len(argv) != 3:
        raise SystemExit(__doc__)
    guest_size, guest = layout(argv[1])
    sdk_size, sdk = layout(argv[2])
    errors = []
    if guest_size != sdk_size:
        errors.append(f"size: the guest's is {guest_size:#x}, the SDK's {sdk_size:#x}")
    for name in sorted(set(guest) | set(sdk), key=lambda n: guest.get(n, sdk.get(n))):
        g, s = guest.get(name), sdk.get(name)
        if g != s:
            errors.append(f"{name}: guest slot {g}, SDK slot {s}")
    for e in errors:
        print(f"check-layout: {CLASS} {e}", file=sys.stderr)
    if errors:
        return 1
    print(f"check-layout: {CLASS} matches the guest's: {guest_size:#x} bytes, "
          f"{len(guest)} virtual methods in their slots")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
