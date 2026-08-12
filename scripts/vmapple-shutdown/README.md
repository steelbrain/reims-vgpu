# vmapple-shutdown.sh

Cleanly stop the running vmapple guest.

The clean path is an **in-guest `shutdown -h now` over SSH**: macOS halts, issues an ARM PSCI
SYSTEM_OFF, and QEMU exits on its own (rc=0) — which is what lets `vm/boot-arm64.sh --capture` capture.
QMP `system_powerdown` does **not** work on vmapple (it is an ACPI power-button event and Apple's
platform has no ACPI, so the macOS guest never sees it — verified: 90s no-op vs. `shutdown -h now`
exiting QEMU in ~5s).

Escalation ladder:

1. **SSH `sudo shutdown -h now`** → wait up to `GRACE` (default 60s) for QEMU to exit cleanly.
2. If SSH is unreachable or QEMU stays up, **QMP `quit`** (immediate; the disk is clean only if
   macOS had already halted — fine here since every boot reverts to a snapshot anyway).
3. **SIGKILL**.

SSH creds default to the guest convention (user=password=`macosvm`, key `~/.ssh/vmapple_guest`);
the QMP fallback uses `vm/<guest>/run/qmp.sock`. Override via `GUEST_DIR GUEST_USER GUEST_PW
SSH_PORT SSH_KEY QMP_SOCK GRACE`.

## Run

```sh
scripts/vmapple-shutdown/vmapple-shutdown.sh
GUEST_DIR="$PWD/vm/guest-13" scripts/vmapple-shutdown/vmapple-shutdown.sh   # non-default bundle
```

Exits 0 when QEMU is stopped (or was already stopped).
