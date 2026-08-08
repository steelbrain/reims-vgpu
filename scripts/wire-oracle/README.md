# wire-oracle

Regenerates ground-truth fixtures for `crates/reims-vgpu-wire` by driving
Apple's own paravirtualized GPU serializer on the host.

```sh
scripts/wire-oracle/wire-oracle.sh --fixtures    # wire bytes + expectations
scripts/wire-oracle/wire-oracle.sh --inventory   # every selector Apple ships
scripts/wire-oracle/wire-oracle.sh --all
```

Output lands in `crates/reims-vgpu-wire/fixtures/`, which is gitignored — the
bytes are Apple's serializer output and are regenerated rather than committed.
Override with `REIMS_WIRE_FIXTURES_DIR`.

## Requirements

- **An Apple host.** The serializer only exists on macOS. The script refuses
  elsewhere rather than producing an empty result.
- **Rosetta.** `AppleParavirtGPUMetal.bundle` ships x86_64 and arm64e slices
  with no plain-arm64 one, and third-party arm64e needs a preview-ABI boot-arg,
  so the harness builds `-arch x86_64` and runs under `arch -x86_64`. Install
  with `softwareupdate --install-rosetta`.

No paravirtualized GPU is needed. `PGSerializer` takes an `id<MTLDevice>` and an
object-ref allocator, neither of which touches IOKit, so it runs on bare metal.
`AppleParavirtDevice`, which does need an `AppleParavirtGPU` IOService, is never
instantiated.

## Consuming the output

```sh
REIMS_WIRE_FIXTURES_REQUIRED=1 cargo test -p reims-vgpu-wire -- --test-threads=1
```

Without that variable a missing fixture set makes the oracle tests report
`ignored`, one line each, naming the command that regenerates them. With it, a
missing set fails the build — set it on any Apple host, where there is no excuse
for not having them.

`inventory.json` gates separately from `fixtures.json`, so a run that has one
and not the other stands down only the tests that read the missing half.

The procedure for turning a new capture into a checked view is in
`crates/reims-vgpu-wire/AGENTS.md`.
