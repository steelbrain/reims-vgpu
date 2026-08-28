# vgpu-dashboard

One page to launch a boot, pick USB devices to pass through, choose the network mode, and read the
device's own censuses while it runs.

```sh
scripts/vgpu-dashboard/vgpu-dashboard.py            # prints a URL with a token; opens a browser
scripts/vgpu-dashboard/vgpu-dashboard.py --port 8765 --no-browser
scripts/vgpu-dashboard/vgpu-dashboard.py --selftest # read-only checks, no VM needed
```

Python stdlib only — this repo has no Python dependency and this is not the place to add one.

## What it is

An **instrument**. It observes host state and drives the scripts that already exist; it is
deliberately not a second implementation of anything this repo owns:

| Question | Who answers it |
|---|---|
| Which rails and snapshots exist? | `boot-x86.sh --list-rails` / `--list-snapshots` |
| What USB devices are here, and what spec names each? | `boot-x86.sh --list-usb` → `vm/lib/usb-passthrough.sh` |
| Where is the guest reachable? | `vm/guest-ip.sh`, `vm/guest-authorize.sh` |
| What does the guest look like? | the host screenshot helper |
| What did the device do? | `/tmp/reims-vgpu-fail.log` — the one thing parsed here, because nothing else parses it |

So a rule that changes lands in the dashboard for free, and the dashboard cannot disagree with the
boot script about what a spec means or which snapshot is current.

## Reading the fail log is where a dashboard goes wrong

Four traps from `AGENTS.md` are implemented rather than left to the reader, because each one fails in
the direction that looks like a finding:

- **The channel is split before anything is ranked.** `OFF` records carry `reason=` too — for
  ordering and control-flow events that are not losses — so ranking `reason=` without splitting
  inverts the queue. The UI shows fail-channel and off-channel counts separately and never sums them.
- **Per-window and cumulative counters are different objects.** `store_routes`, `drain_duty`,
  `gpu_span`, `window_publish` and `host_window_cadence` reset each interval, so a total is the
  **sum** and `tail -1` reads three to four times low. `registry_pressure` and `display_vbl` are
  high-waters — the device labels the former "(levels, not per-interval)" — where the **last** sample
  is the answer and summing is the error. `CENSUS_REDUCE` carries which is which.
- **Everything per-draw is joined by `t=`, never by line order.** Each census skips different
  windows, so pairing by position drifts and pulls idle-desktop samples into a driven band; a harness
  that did this read a driven boot at ~31 fps where banding by `t` reads 47-52. The shared band is
  the intersection of the per-window censuses and is shown with the numbers.
- **`present_hz` is never displayed alone.** It is a reading of the presenter *and* of the device's
  publish rate, so it travels with `offered_hz` and with `busy_fence`/`busy_acquire`: both up is the
  win, offered up alone means the presenter has become a ceiling, neither moving means the change
  bought no frames.

Two more it surfaces because a reading is worthless without them:

- **Boot count**, from `vk_caps` (one per device creation). More than one means every ranking mixes
  builds — `first_sight` latches per process, so a stale log inflates in a way that reads as a
  finding. A device restart *inside* the parsed tail is detected too (`t=` going backwards) and the
  older samples are dropped rather than blended.
- **The workload regime**, from drain duty. ~0.00 on a bursty interaction probe, ~0.91 on a sustained
  one, and only the sustained regime can turn a per-draw CPU saving into frames. A reading taken at
  low duty is labelled *not rankable*.

It also reports the **60 Hz / free-running population** a boot latched, because nothing per-draw is
comparable across the two, and host **load average**, because every `us=` is wall clock.

## The panic verdict outranks everything

Each launched boot's stdout is captured to `vm/disks/run/dashboard-boot-<stamp>.log` and grepped for
a guest kernel panic. `probe exit=0` is not a clean boot and neither is a live device — both read
green on a boot whose guest died, which on some rails is roughly one driven boot in three.

## Safety

This server execs QEMU and the repo's scripts, so:

- **Loopback only**, and not configurable otherwise. Every request needs the random token printed at
  startup.
- **Nothing reaches a shell.** Every child is an argv list. Rails, snapshots, bridges, MACs and RAM
  sizes are matched against closed patterns; a USB spec is re-validated against the same three forms
  the library accepts before it can go near a command line.
- **Env knobs are an allowlist**, because an override may only *narrow* what the device does — it can
  turn off a rail the host could have run, never turn on one the host reported it cannot.
- **QMP is an allowlist** of input verbs. `qmp.py cmd` issues arbitrary QMP and is not exposed.
- **Launches are POST-only**, so a browser prefetch cannot start a VM.
- `capture` needs explicit confirmation: it writes a new snapshot into the rail.
- Booting while another QEMU runs is **refused** by default. On `NET=bridge` a second guest collides
  on the pinned `GUEST_MAC` with no diagnostic at all, and either way the measurement is contended.

`--selftest` covers the host probes, both list parsers, log parsing and reduction, argv construction,
and twelve rejection cases (shell metacharacters, path traversal, unknown devices, unlisted knobs,
bad specs, `capture` without confirmation, bridge-on-arm64). It does **not** cover a live guest or
anything on the arm64 pathway, and says so rather than passing quietly.

## Bridged networking needs the host to be ready

The dashboard assembles one verdict with its reasons instead of four badges to combine: bridge
present, `/dev/net/tun` openable, a privileged `qemu-bridge-helper`, and an `allow` line in
`/etc/qemu/bridge.conf`. If the running kernel has no module tree — a kernel upgrade without a
reboot — it says so by name, because that state breaks every bridged netdev while `lsmod`, `modinfo`
and `modprobe` all report something that reads like a missing package.

After a bridged boot reaches the desktop, run **Authorize guest**: there is no `localhost:2222` on a
bridge, and that step resolves the lease and points the `macos-vm` alias every probe types at this
boot. See `vm/README.md`.
