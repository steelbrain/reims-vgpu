#!/usr/bin/env python3
"""Stage our configuration's bundle for Ambrosia, which the Makefile's config-bundle target packs.

    stage-config.py OUT CONFIG MODULE GRAFT REIMS_GRAPHICS RGFXGATE
    stage-config.py --identifiers

OUT is written afresh: CONFIG, the config.plist that names the module, with a Kernel > Add entry
for each kext and the display's node (Graphics > DisplayNode) as OUT/config.plist; MODULE as
OUT/Drivers/ReimsGraphics.elf; and each kext as OUT/Kexts/<its name>.kext. The entries list the
kexts in KEXTS' order, Graft before the two kexts that run on it.

Graphics > DisplayNode has Ambrosia add /arm-io/<DISPLAY_NODE> to the device tree, a display-subsystem
node, where the module drives a display: IOMobileFramebuffer.framework counts such nodes once in a
process, WindowServer before the kexts load, and waits for as many framebuffer services as it
counted. The node is given only with the kexts, which publish display 0's service and use the node
as it stands; a bundle of the module alone gives none, so no process waits for a service that never
comes.

Each kext must be a bundle whose Contents/Info.plist names the identifier KEXTS gives its role,
and whose executable has an arm64e slice, the architecture of an arm64 guest's kernel. Every kext
that is not given or not usable is named, with why, and nothing is written.

--identifiers prints the kexts' identifiers, one a line, for check-boot.sh's kexts check.
"""
import os
import plistlib
import shutil
import struct
import sys

# Each kext of our configuration, in Kernel > Add order: the make variable that gives its path,
# its bundle's name in the archive, and its CFBundleIdentifier.
KEXTS = [
    ("GRAFT", "Graft.kext", "ai.aneesiqbal.graft"),
    ("REIMS_GRAPHICS", "ReimsGraphics.kext", "ai.aneesiqbal.ReimsGraphics"),
    ("RGFXGATE", "RgfxGate.kext", "ai.aneesiqbal.RgfxGate"),
]

# The name of display 0's node under /arm-io, the one ReimsGraphics looks for
# (ReimsGraphicsMobile.cpp, attachMobileNode).
DISPLAY_NODE = "reims-display0"

CPU_TYPE_X86_64 = 0x01000007
CPU_TYPE_ARM64 = 0x0100000C
CPU_SUBTYPE_ARM64E = 2
CPU_SUBTYPE_MASK = 0xFF000000


def arch_name(cputype, subtype):
    if cputype == CPU_TYPE_ARM64:
        return "arm64e" if subtype & ~CPU_SUBTYPE_MASK == CPU_SUBTYPE_ARM64E else "arm64"
    if cputype == CPU_TYPE_X86_64:
        return "x86_64"
    return f"cputype {cputype:#x}"


def archs(path):
    """The architectures of the Mach-O file at path, thin or fat, or None where it is neither."""
    with open(path, "rb") as f:
        head = f.read(12)
    if len(head) < 12:
        return None
    if head[:4] == b"\xcf\xfa\xed\xfe":  # MH_MAGIC_64, little-endian: cputype, cpusubtype
        return [arch_name(*struct.unpack("<II", head[4:12]))]
    magic, count = struct.unpack(">II", head[:8])
    if magic not in (0xCAFEBABE, 0xCAFEBABF) or count > 64:  # FAT_MAGIC, FAT_MAGIC_64
        return None
    size = 20 if magic == 0xCAFEBABE else 32  # fat_arch, fat_arch_64: cputype, cpusubtype first
    with open(path, "rb") as f:
        f.seek(8)
        table = f.read(size * count)
    if len(table) < size * count:
        return None
    return [arch_name(*struct.unpack(">II", table[i * size:i * size + 8])) for i in range(count)]


def check(var, path, name, identifier):
    """Why the kext given as var=path can't be used, or None."""
    if not path:
        return f"{var} is not set: give its arm64e build, {var}=<dir>/{name}"
    where = f"{var}={path}"
    if not os.path.isdir(path):
        return f"{where}: no such directory"
    info_path = os.path.join(path, "Contents", "Info.plist")
    try:
        with open(info_path, "rb") as f:
            info = plistlib.load(f)
    except FileNotFoundError:
        return f"{where}: no Contents/Info.plist"
    except Exception as e:  # plistlib raises several types for a bad file
        return f"{where}: Contents/Info.plist is not a property list: {e}"
    if not isinstance(info, dict) or info.get("CFBundleIdentifier") != identifier:
        found = info.get("CFBundleIdentifier") if isinstance(info, dict) else None
        return f"{where}: its CFBundleIdentifier is {found!r}, not {identifier!r}"
    exe = info.get("CFBundleExecutable")
    if not isinstance(exe, str) or not exe:
        return f"{where}: its Info.plist names no CFBundleExecutable"
    exe_path = os.path.join(path, "Contents", "MacOS", exe)
    if not os.path.isfile(exe_path):
        return f"{where}: no Contents/MacOS/{exe}"
    found = archs(exe_path)
    if found is None:
        return f"{where}: Contents/MacOS/{exe} is not a Mach-O file"
    if "arm64e" not in found:
        return f"{where}: Contents/MacOS/{exe} has no arm64e slice, only {', '.join(found)}"
    return None


def main(argv):
    if argv[1:] == ["--identifiers"]:
        for _, _, identifier in KEXTS:
            print(identifier)
        return 0
    if len(argv) != 4 + len(KEXTS):
        print("usage: " + __doc__.strip().splitlines()[2].strip(), file=sys.stderr)
        return 2
    out, config, module = argv[1:4]
    paths = argv[4:]
    errors = [e for e in (check(var, path, name, identifier)
                          for (var, name, identifier), path in zip(KEXTS, paths)) if e]
    if errors:
        for e in errors:
            print(f"stage-config: {e}", file=sys.stderr)
        print(f"stage-config: no bundle written: {len(errors)} of {len(KEXTS)} kexts are missing "
              "or not usable", file=sys.stderr)
        return 1
    with open(config, "rb") as f:
        plist = plistlib.load(f)
    plist.setdefault("Kernel", {})["Add"] = [{"BundlePath": f"Kexts/{name}"} for _, name, _ in KEXTS]
    plist.setdefault("Graphics", {})["DisplayNode"] = DISPLAY_NODE
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(os.path.join(out, "Drivers"))
    with open(os.path.join(out, "config.plist"), "wb") as f:
        plistlib.dump(plist, f, fmt=plistlib.FMT_XML)
    shutil.copy(module, os.path.join(out, "Drivers", "ReimsGraphics.elf"))
    for (_, name, _), path in zip(KEXTS, paths):
        shutil.copytree(path, os.path.join(out, "Kexts", name))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
