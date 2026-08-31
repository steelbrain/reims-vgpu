#!/usr/bin/env python3
"""Score a guest run against the native one, and against what is already known.

The battery prints one line per case and a reader compares two runs by eye. That
works until the counts stop matching, and then it does not work at all: a case
that stopped being invoked and a case that started failing look the same in a
summary line, and the interesting one is the one nobody notices.

The rules are the README's table, mechanised. Read them in the order they fire,
because the first two say a reading is *not* about this device:

  native FAIL          the expectation in the suite is wrong. Not a finding, and
                       every guest reading of that case is meaningless until it
                       is fixed.
  native missing       the case ran in the guest and never natively. There is no
                       oracle for it, so its guest result says nothing.
  guest missing        a case the oracle ran and the guest did not report at
                       all. Either it was never invoked or the run died mid-way.
  guest FAIL, classified
                       a defect already written down as translator- or
                       driver-owned. Report its owner without failing.
  guest FAIL, new      a regression, and the reason this tool exits non-zero.
  guest PASS, classified
                       the inventory is stale. Also non-zero: an inventory
                       nobody prunes stops being a list of defects and becomes
                       a list of cases nobody looks at.
  guest SKIP           the device's own reported limits make the case
                       inexpressible. Never a failure.

And one more, which needs the device's own fail log rather than the two runs.
A case may print `ROUTE <case> <counter>` to claim the device rail it was
written to move. If the counter never moves in the whole run, the case is
NOT-COVERED however green it is -- three attempts at the regression case for
the unordered host read passed on the broken build for exactly that reason.
The check is per run, not per case: nothing carries a case name into the
device. A narrowing `REIMS_VGPU_*` switch can close the rail a claim names
legitimately, so `--claims-advisory` reports those without failing the run.
"""

import argparse
import re
import sys

CASE = re.compile(r"^CASE (\S+) (PASS|FAIL|SKIP) ?(.*)$")
ROUTE = re.compile(r"^ROUTE (\S+) (\S+)$")
DEVICE = re.compile(r"^DEVICE (.*)$")


def parse(path):
    """Case name -> (verdict, detail), the DEVICE line, duplicates, route claims.

    A case name repeats when a battery is edited badly; keep the first and say
    so rather than letting the later one overwrite it silently.
    """
    cases, device, dupes, routes = {}, None, [], {}
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = ROUTE.match(line)
            if m:
                routes.setdefault(m.group(1), []).append(m.group(2))
                continue
            m = CASE.match(line)
            if m:
                if m.group(1) in cases:
                    dupes.append(m.group(1))
                else:
                    cases[m.group(1)] = (m.group(2), m.group(3))
                continue
            m = DEVICE.match(line)
            if m:
                device = m.group(1)
    return cases, device, dupes, routes


def expectations(path):
    """Known guest failures: one case name a line, `#` starts a comment."""
    known = {}
    if not path:
        return known
    try:
        fh = open(path)
    except FileNotFoundError:
        return known
    with fh:
        for line in fh:
            body, _, why = line.partition("#")
            name = body.strip()
            if name:
                known[name] = why.strip()
    return known


def counters_that_moved(path):
    """Every `name=<non-zero>` this device reported anywhere in its fail log.

    Deliberately not a sum. A counter is per-census-window for some records and
    a cumulative high-water for others, and the question here is only whether
    the rail was reached at all, which both spellings answer the same way.
    """
    moved = set()
    if not path:
        return None
    try:
        fh = open(path, errors="replace")
    except FileNotFoundError:
        return None
    with fh:
        for line in fh:
            for name, value in re.findall(r"([a-z][a-z_0-9]*)=([0-9]+)", line):
                if value != "0":
                    moved.add(name)
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True, help="the oracle run, or a baseline")
    ap.add_argument("--guest", required=True)
    ap.add_argument("--translation-errors", required=True,
                    help="translator-owned failures, each backed by a bugs/ package")
    ap.add_argument("--driver-errors", required=True,
                    help="non-translation device failures")
    ap.add_argument("--device", help="the device's own fail log, for route claims")
    ap.add_argument("--claims-advisory", action="store_true",
                    help="report an unmoved route claim without failing the run; "
                         "what to pass when a narrowing REIMS_VGPU_* switch is set")
    ap.add_argument("--quiet", action="store_true", help="only the lines that matter")
    args = ap.parse_args()

    native, native_dev, native_dupes, _ = parse(args.native)
    guest, guest_dev, guest_dupes, claims = parse(args.guest)
    translation = expectations(args.translation_errors)
    driver = expectations(args.driver_errors)
    moved = counters_that_moved(args.device)

    rows, bad = [], 0
    active_translation = 0
    active_driver = 0
    for name in sorted(set(native) | set(guest)):
        n = native.get(name)
        g = guest.get(name)
        detail = (g or n or ("", ""))[1]
        if n and n[0] == "FAIL":
            verdict, note = "SUITE-BUG", "fails on the oracle: the expectation is wrong"
        elif not n:
            verdict, note = "NO-ORACLE", "never ran natively, so the guest result says nothing"
        elif not g:
            verdict, note = "NOT-RUN", "the oracle ran it and the guest did not report it"
        elif g[0] == "SKIP":
            verdict, note = "skip", detail
        elif g[0] == "FAIL" and name in translation:
            verdict, note = "translation", translation[name] or detail
            active_translation += 1
        elif g[0] == "FAIL" and name in driver:
            verdict, note = "driver", driver[name] or detail
            active_driver += 1
        elif g[0] == "FAIL":
            verdict, note = "REGRESSION", detail
        elif name in translation:
            verdict = "FIXED-TRANSLATION"
            note = "passes now; remove it from the translation inventory"
        elif name in driver:
            verdict = "FIXED-DRIVER"
            note = "passes now; remove it from the driver inventory"
        elif moved is not None and name in claims and not all(
                c in moved for c in claims[name]):
            missing = [c for c in claims[name] if c not in moved]
            verdict = "not-covered" if args.claims_advisory else "NOT-COVERED"
            note = "passed without moving " + ", ".join(missing)
        else:
            verdict, note = "ok", ""
        if verdict.isupper() or verdict in ("translation", "driver", "not-covered"):
            bad += verdict.isupper()
            rows.append((verdict, name, note))
        elif not args.quiet:
            rows.append((verdict, name, note))

    for name in sorted(set(native_dupes) | set(guest_dupes)):
        rows.append(("DUPLICATE", name, "reported twice in one run"))
        bad += 1

    for name in sorted(set(translation) & set(driver)):
        rows.append(("DUPLICATE-CLASSIFICATION", name,
                     "listed as both translator- and driver-owned"))
        bad += 1

    width = max((len(r[0]) for r in rows), default=0)
    for verdict, name, note in rows:
        print(f"{verdict:<{width}}  {name}" + (f"  -- {note}" if note else ""))

    print()
    print(f"native {len(native)} cases from {native_dev or 'an unnamed device'}")
    print(f"guest  {len(guest)} cases from {guest_dev or 'an unnamed device'}")
    claimed = len(claims)
    if moved is None:
        print(f"{claimed} cases claim a device rail; no fail log given to check them")
    else:
        print(f"{claimed} cases claim a device rail")
    print(f"active translation failures {active_translation}/{len(translation)}, "
          f"active driver failures {active_driver}/{len(driver)}, unexplained {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
