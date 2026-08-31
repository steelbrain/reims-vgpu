#!/usr/bin/env python3
"""Score a candidate guest run against an identified control guest run.

`verdict.py` answers a different question: it asks the native oracle whether a
guest result is *right*. This asks the control whether a candidate moved
*backwards*, which is the question a change has to answer before it may be
committed, and it is not the same question.

The difference matters most where the two disagree. A rail whose debt is not yet
classified reads as 77 unexplained failures to the oracle scorer and will do so
until every one of them has an established owner -- which is honest, and which
also means that scorer is red for every candidate and cannot distinguish a
candidate that broke something from one that broke nothing. The control can. It
already contains that debt, so a candidate reproducing it exactly has preserved
the device's behavior whether or not anyone has yet written down whose fault it
is.

The transitions are the workflow's table, mechanised:

  control PASS -> candidate PASS            preserved
  control PASS -> anything else             REGRESSION
  control FAIL -> candidate FAIL, same      unchanged debt
  control FAIL -> candidate PASS            improvement; prune the ledger
  control FAIL -> candidate SKIP/not-run    REGRESSION
  control SKIP -> candidate SKIP            unchanged applicability
  control SKIP -> candidate PASS            improvement
  control SKIP -> candidate FAIL/not-run    REGRESSION
  name in one run only                      REGRESSION (coverage moved)
  name twice in either run                  REGRESSION (the totals lie)

"Same failure" is the same case, result class and typed reason; changing
diagnostic payload alone is not a regression. Nothing in a `CASE` line separates
reason from payload, so a changed detail on a still-failing case is reported as
CHANGED-DETAIL and does not by itself fail the run. It is the one reading this
tool hands back rather than decides: a payload that moved may be the same defect
described differently, or a different defect wearing the same name, and only a
reader who knows the case can say which.

The device's own log is compared too. A candidate that keeps every pixel and
doubles the fence timeouts has moved backwards, and no case comparison can see
it. That comparison needs more than one control, because the log is not
reproducible the way the cases are: two back-to-back macos-13 controls of the
same build agreed on all 290 case results exactly and still disagreed on their
typed-reason counts, `stamp_wait_timeout` among them. So `--control` may be
given more than once, and a repeated control establishes an envelope rather than
a number. A candidate inside the envelope has not moved; outside it, it has.

One control is accepted and says so in the totals. It cannot separate a
regression from run-to-run variance in the log, and a reader should not have to
infer that from the absence of a second `--control`.
"""

import argparse
import collections
import re
import sys

CASE = re.compile(r"^CASE (\S+) (PASS|FAIL|SKIP) ?(.*)$")
DEVICE = re.compile(r"^DEVICE (.*)$")
# A typed record on the always-on fail channel opens with its reason slug.
REASON = re.compile(r"^([a-z][a-z0-9_]*) ")
# `OFF` records are observations, not failures, and ranking them as failures is
# a documented way to read this log wrong.
NOT_A_FAILURE = re.compile(r"\bOFF\b")
# The catastrophic classes the workflow names, as they appear in the log.
CATASTROPHIC = re.compile(r"stamp_wait_timeout|device_lost|VK_ERROR_DEVICE_LOST")


def parse_cases(path):
    """Case name -> (result, detail), plus the DEVICE line and repeated names.

    A repeated name is kept at its first sighting and reported, because the
    alternative -- letting the second overwrite the first -- is how a battery
    that reports a case twice reads as a battery that reports it once.
    """
    cases, device, dupes = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = DEVICE.match(line)
            if m:
                device = m.group(1)
                continue
            m = CASE.match(line)
            if not m:
                continue
            name, result, detail = m.group(1), m.group(2), m.group(3).strip()
            if name in cases:
                dupes.append(name)
                continue
            cases[name] = (result, detail)
    return cases, device, dupes


def parse_device(path):
    """Typed reason -> count, over one device log.

    Counted, not deduplicated: the emitters deduplicate and the counters do not,
    so this is the population a reader may compare between two runs only because
    both sides are counted the same way here.
    """
    counts = collections.Counter()
    if not path:
        return counts
    try:
        fh = open(path)
    except OSError:
        return counts
    with fh:
        for line in fh:
            if NOT_A_FAILURE.search(line):
                continue
            m = REASON.match(line)
            if m:
                counts[m.group(1)] += 1
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--control",
        required=True,
        action="append",
        help="identified control run's conformance.txt; repeat for an envelope",
    )
    ap.add_argument("--candidate", required=True, help="candidate run's conformance.txt")
    ap.add_argument(
        "--control-device",
        action="append",
        default=[],
        help="a control's device.log, in the same order as its --control",
    )
    ap.add_argument("--candidate-device", help="the candidate's device.log")
    ap.add_argument(
        "--quiet",
        action="store_true",
        help="print only the transitions that are not 'preserved' and the totals",
    )
    args = ap.parse_args()

    parsed = [parse_cases(path) for path in args.control]
    control, control_dev, control_dupes = parsed[0]
    candidate, candidate_dev, candidate_dupes = parse_cases(args.candidate)

    regressions, improvements, changed, lines = [], [], [], []
    tally = collections.Counter()

    # A case the controls themselves disagree about has no control reading to
    # score a candidate against, and scoring it anyway would charge the
    # candidate for the rail's own variance. Drop it from the comparison and say
    # so, because a silently unscored case is the failure this whole tool is
    # about.
    unstable = set()
    for other, _, _ in parsed[1:]:
        for name in set(control) | set(other):
            if control.get(name) != other.get(name):
                unstable.add(name)
    for name in sorted(unstable):
        readings = " | ".join(str(p[0].get(name)) for p in parsed)
        changed.append(f"CONTROL-UNSTABLE {name} -- not scored; controls said {readings}")
        control.pop(name, None)
        candidate.pop(name, None)

    for name in control_dupes:
        regressions.append(f"DUPLICATE-IN-CONTROL {name}")
    for name in candidate_dupes:
        regressions.append(f"DUPLICATE-IN-CANDIDATE {name}")

    for name in sorted(set(control) | set(candidate)):
        before = control.get(name)
        after = candidate.get(name)
        if before is None:
            # A name the candidate reports and the control never did cannot be
            # scored as a transition. It is not a pass to be credited: there is
            # no control reading for it.
            regressions.append(f"NOT-IN-CONTROL {name} -- candidate says {after[0]}")
            continue
        if after is None:
            regressions.append(f"NOT-RUN {name} -- control said {before[0]}")
            continue
        b, a = before[0], after[0]
        if b == "PASS" and a == "PASS":
            tally["preserved"] += 1
        elif b == "PASS":
            regressions.append(f"REGRESSION {name} -- PASS -> {a}: {after[1]}")
        elif b == "FAIL" and a == "FAIL":
            tally["unchanged debt"] += 1
            if before[1] != after[1]:
                changed.append(
                    f"CHANGED-DETAIL {name}\n    control:   {before[1]}\n    candidate: {after[1]}"
                )
        elif b == "FAIL" and a == "PASS":
            improvements.append(f"IMPROVEMENT {name} -- FAIL -> PASS; prune the ledger")
        elif b == "FAIL":
            regressions.append(f"REGRESSION {name} -- FAIL -> {a}: a failure that stopped running")
        elif b == "SKIP" and a == "SKIP":
            tally["unchanged applicability"] += 1
        elif b == "SKIP" and a == "PASS":
            improvements.append(f"IMPROVEMENT {name} -- SKIP -> PASS; the case became expressible")
        else:
            regressions.append(f"REGRESSION {name} -- SKIP -> {a}: {after[1]}")

    if control_dev != candidate_dev:
        regressions.append(
            f"DEVICE-CHANGED -- control {control_dev!r} candidate {candidate_dev!r}"
        )

    # The envelope every control run put this reason inside. A candidate within
    # it has not moved; the width is the rail's own variance, measured rather
    # than assumed.
    before_logs = [parse_device(path) for path in args.control_device]
    after_log = parse_device(args.candidate_device)
    reasons = set(after_log)
    for log in before_logs:
        reasons |= set(log)
    for reason in sorted(reasons):
        seen = [log[reason] for log in before_logs] or [0]
        lo, hi = min(seen), max(seen)
        a = after_log[reason]
        if lo <= a <= hi:
            continue
        envelope = str(lo) if lo == hi else f"{lo}..{hi}"
        line = f"DEVICE-REASON {reason} -- control {envelope} candidate {a}"
        if a > hi and CATASTROPHIC.search(reason):
            regressions.append(line + " (a catastrophic class got worse)")
        elif a > hi:
            changed.append(line)
        else:
            improvements.append(line + " (fewer)")

    for line in regressions:
        lines.append(line)
    for line in changed:
        lines.append(line)
    for line in improvements:
        lines.append(line)
    if not args.quiet:
        lines.append("")
    print("\n".join(lines))
    print(
        f"control   {len(control)} cases from {control_dev}"
        f" ({len(parsed)} control run{'s' if len(parsed) != 1 else ''})\n"
        f"candidate {len(candidate)} cases from {candidate_dev}"
    )
    if len(before_logs) < 2:
        print(
            "note: fewer than two control device logs, so a moved typed-reason "
            "count cannot be told from run-to-run variance"
        )
    print(
        f"preserved {tally['preserved']}, "
        f"unchanged debt {tally['unchanged debt']}, "
        f"unchanged applicability {tally['unchanged applicability']}, "
        f"improvements {len(improvements)}, "
        f"changed detail {len(changed)}, "
        f"control-unstable {len(unstable)}, "
        f"regressions {len(regressions)}"
    )
    # An improvement is not a failure, but it is not nothing either: the ledger
    # it makes stale is the reader's to prune. Only a backwards move exits red.
    return 1 if regressions else 0


if __name__ == "__main__":
    sys.exit(main())
