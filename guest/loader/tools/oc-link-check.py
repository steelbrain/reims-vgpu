#!/usr/bin/env python3
"""What OpenCore's linker made of a consumer and its shims
(guest/tools/oc-link-set.sh): KextInject's log, and the kernel collection
it wrote, against gen-shims.py's fixup table.

    python3 -I -B tools/oc-link-check.py KEXTINJECT.LOG OUT.BIN FIXUPS.JSON CONSUMER_ID

From the log: each injected kext's result, every "Symbol ... was resolved"
and "... was unresolved" line, vtable patch failures and RIP-relative
overflows. From out.bin: each injected kext's LC_FILESET_ENTRY (its link
address and file offset) and its segments, then every site of the table:
- that the log resolved the site's symbol, for the consumer, to the
  placeholder gen-shims.py laid out for it (the shim's link address plus
  the placeholder's offset in it);
- that the site holds that address plus the addend: a rel32 branch's
  target, or a pointer, which OpenCore writes into a KC as a chained fixup
  (MACH_DYLD_CHAINED_PTR_64_KERNEL_CACHE_REBASE: a 30-bit target, the
  address less KERNEL_FIXUP_OFFSET, 1 MB; KernelCollection.c
  InternalKcConvertRelocToFixup in OpenCorePkg 1.0.8).

Exits non-zero if any kext failed, a symbol was unresolved, or a site does
not hold its placeholder. out.bin holds the guest's Boot KC, untrusted: this
only reads it. Ours. Run with `python3 -I`.
"""

import json
import re
import struct
import sys
from collections import Counter, defaultdict

LC_SEGMENT_64 = 0x19
LC_FILESET_ENTRY = 0x80000035
KERNEL_FIXUP_OFFSET = 1 << 20
TARGET_MASK = (1 << 30) - 1

RESOLVED = re.compile(r"^OCAK: Symbol (\S+) was resolved for kext (\S+) to ([0-9A-Fa-f]+)$")
UNRESOLVED = re.compile(r"^OCAK: Symbol (\S+) was unresolved for kext (\S+)$")
INJECTED = re.compile(r"^\[(OK|FAIL)\] (\S+) injected - (.*)$")


def read_log(path):
    resolved = defaultdict(dict)
    unresolved, injected, other = [], [], []
    for line in open(path, errors="replace"):
        line = line.rstrip("\n")
        if m := RESOLVED.match(line):
            resolved[m[2]][m[1]] = int(m[3], 16)
        elif m := UNRESOLVED.match(line):
            unresolved.append((m[2], m[1]))
        elif m := INJECTED.match(line):
            injected.append((m[1], m[2], m[3]))
        elif ("Vtable patching failed" in line or "RIP-relative overflow" in line
              or "Failed to patch" in line or "Prelink inject complete" in line):
            other.append(line)
    return resolved, unresolved, injected, other


def fileset(data):
    """The KC's fileset entries: bundle ID -> (vmaddr, fileoff)."""
    ncmds = struct.unpack_from("<I", data, 16)[0]
    off, out = 32, {}
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, off)
        if cmd == LC_FILESET_ENTRY:
            vmaddr, fileoff, name_off = struct.unpack_from("<QQI", data, off + 8)
            name = data[off + name_off:off + size].split(b"\0", 1)[0].decode()
            out[name] = (vmaddr, fileoff)
        off += size
    return out


def segments(data, fileoff):
    """An injected kext's segments, read from its header in the KC:
    (name, vmaddr, vmsize, fileoff, filesize)."""
    ncmds = struct.unpack_from("<I", data, fileoff + 16)[0]
    off, out = fileoff + 32, []
    for _ in range(ncmds):
        cmd, size = struct.unpack_from("<II", data, off)
        if cmd == LC_SEGMENT_64:
            name = data[off + 8:off + 24].rstrip(b"\0").decode()
            out.append((name, *struct.unpack_from("<QQQQ", data, off + 24)))
        off += size
    return out


def owner(data, entries):
    """A function from an address to the fileset entry whose segments hold
    it (the KC's own and the injected), or None."""
    spans = []
    for name, (_, fileoff) in entries.items():
        for seg, va, vs, _, _ in segments(data, fileoff):
            if vs and seg != "__LINKEDIT":
                spans.append((va, va + vs, name))
    spans.sort()

    def find(addr):
        for start, end, name in spans:
            if start <= addr < end:
                return name
        return None
    return find


def file_offset(segs, vmaddr):
    for _, va, vs, fo, fs in segs:
        if fs and va <= vmaddr < va + fs:
            return fo + (vmaddr - va)
    return None


def main(argv):
    if len(argv) != 5:
        raise SystemExit(__doc__.split("\n\n")[1])
    log, out_bin, fixups, consumer = argv[1:]
    resolved, unresolved, injected, other = read_log(log)
    data = open(out_bin, "rb").read()
    table = json.load(open(fixups))
    entries = fileset(data)
    bad = False

    print("== injected ==")
    for status, path, rest in injected:
        print(f"  [{status}] {path.split('/')[-1]} - {rest}")
        bad |= status != "OK"
    for line in other:
        print(f"  {line}")
        bad |= "complete success" not in line
    print(f"== unresolved: {len(unresolved)} ==")
    for kext, sym in unresolved:
        print(f"  {kext}: {sym}")
    bad |= bool(unresolved)

    if consumer not in entries:
        print(f"{consumer}: no fileset entry in {out_bin}")
        return 1
    base, foff = entries[consumer]
    segs = segments(data, foff)
    got = resolved.get(consumer, {})
    print(f"== {consumer}: at {base:#x}, file offset {foff:#x}; "
          f"{len(got)} symbols resolved, into ==")
    find = owner(data, entries)
    for name, n in Counter(find(v) for v in got.values()).most_common():
        print(f"  {n:5} {name}")

    for prov in table["providers"]:
        shim = prov.get("shim")
        if shim not in entries:
            print(f"  {prov['library']}: shim {shim} not in the KC")
            bad = True
            continue
        shim_base = entries[shim][0]
        held, wrong = Counter(), []
        names = set()
        for s in prov["sites"]:
            want = shim_base + s["placeholder"]
            names.add(s["symbol"])
            if got.get(s["symbol"]) != want:
                have = got.get(s["symbol"])
                wrong.append((s, f"resolved to {'nothing' if have is None else hex(have)}, "
                                 f"placeholder {want:#x}"))
                continue
            va = base + s["offset"]
            fo = file_offset(segs, va)
            if fo is None:
                wrong.append((s, f"{va:#x} is in no segment"))
                continue
            if s["width"] == 4:
                if data[fo - 1] != s["opcode"]:
                    wrong.append((s, f"opcode {data[fo - 1]:#04x}"))
                    continue
                target = va + 4 + struct.unpack_from("<i", data, fo)[0]
                ok = target == want + s["addend"]
                seen = f"target {target:#x}"
            else:
                raw = struct.unpack_from("<Q", data, fo)[0]
                target = raw & TARGET_MASK
                ok = (target == (want + s["addend"] - KERNEL_FIXUP_OFFSET) & TARGET_MASK
                      and raw >> 63 == 0 and (raw >> 30) & 3 == 0)
                seen = f"fixup {raw:#018x}"
            if ok:
                held[s["kind"]] += 1
            else:
                wrong.append((s, f"{seen}, placeholder {want:#x}"))
        total = Counter(s["kind"] for s in prov["sites"])
        print(f"  {prov['library']} -> {shim} at {shim_base:#x}: "
              f"{len(names)} symbols, {sum(held.values())} of {len(prov['sites'])} sites "
              f"hold their placeholder: "
              + ", ".join(f"{k} {held[k]}/{total[k]}" for k in sorted(total)))
        for s, why in wrong[:10]:
            print(f"    {s['kind']} +{s['offset']:#x} {s['symbol']}: {why}")
        if len(wrong) > 10:
            print(f"    ... {len(wrong) - 10} more")
        bad |= bool(wrong)

    vt = sum(1 for p in table["providers"] for s in p["sites"] if s["kind"] == "vtable")
    print(f"== vtable slots in the table: {vt} ==")
    print("FAILED" if bad else "OK: every site holds its shim's placeholder")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
