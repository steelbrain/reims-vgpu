# vmapple-snapshot.sh

Manage the vmapple guest's **immutable snapshot history**. Snapshots are **per-rail** — a rail is one
guest OS line — and live under `vm/guest/rails/<rail>/snapshots/<label>/{disk.img,aux.img.trimmed}`,
with a `current` symlink inside each rail naming that rail's active snapshot. Snapshots are APFS
clones (instant, COW) and read-only — they are **never overwritten**. `vm/boot-arm64.sh` reverts to
the selected rail's `current` on every boot and captures new snapshots via `--capture`; this tool
covers the rest.

Every command below operates on **one rail**: `--rail NAME`, else `$RAIL`, else whatever
`vm/guest/rails/current` names. Nothing here repoints `rails/current` — the default rail is a
deliberate choice, not something a snapshot edit should move.

```sh
scripts/vmapple-snapshot/vmapple-snapshot.sh rails             # list rails (* = default)
scripts/vmapple-snapshot/vmapple-snapshot.sh list              # the rail's snapshots (* = current)
scripts/vmapple-snapshot/vmapple-snapshot.sh current           # print the rail's current label
scripts/vmapple-snapshot/vmapple-snapshot.sh rollback <label>  # repoint the rail's current (no data touched)
scripts/vmapple-snapshot/vmapple-snapshot.sh create [label]    # clone the at-rest bundle → new snapshot, make current
scripts/vmapple-snapshot/vmapple-snapshot.sh --rail macos-15 list
```

`create` requires the guest to be shut down (clean snapshot). `rollback` just moves the rail's
`current` pointer, so you can jump between that rail's snapshots freely — the whole history stays on
disk.

Change the default rail with `ln -sfn <rail> vm/guest/rails/current`.

Env: `GUEST_DIR`, `RAILS_DIR`, `RAIL`.
