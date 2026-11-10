# Contributing to Graft

This file covers building Graft, the checks every change passes, the rules for
code and data that may enter the repository, the commit style, and how a
release is made. [docs/](docs/README.md) describes how Graft works.

## What you need

- A Mac with Apple silicon and Xcode. Graft builds against the Kernel.framework
  headers in Xcode's macOS SDK. Four host tests run as native arm64e processes,
  and three run as x86-64 under Rosetta 2.
- For the arm64e tests, a macOS that runs arm64e code built outside Apple.
  macOS 27 runs them with no boot-args; an older macOS may need the
  `-arm64e_preview_abi` boot-arg on that Mac ([tests/](tests/README.md)).
- Python 3.9 or later with only its standard library, for the tests and the
  tools. macOS's own `/usr/bin/python3` will do.
- clang-format and ruff, for `make lint` and `make format`. CI pins
  clang-format 21.1 and ruff 0.16.6.

Nothing is fetched at build time, and no third-party code goes into the kext.

## Building

| target | does |
|---|---|
| `make` | builds `build/Graft.kext`, the x86-64 kext |
| `make graft-arm64e` | builds `build/Graft-arm64e.kext`, the arm64e kext |
| `make test` | builds and runs every host test ([tests/](tests/README.md)) |
| `make kernel-check` | compiles a test kext that carries a manifest (`tests/kernel_fixture.cpp`) with kext flags against Kernel.framework, and links it |
| `make kernel-check-arm64e` | compiles the architecture-neutral core and the arm64e layer for arm64e |
| `make examples` | builds both [examples](examples/README.md) for x86-64 and arm64e, through their own Makefiles |
| `make core-probe`, `make route-probe` | build the test-only owned clients ([tests/kexts/](tests/kexts/README.md)) |
| `make lint` | checks formatting, SPDX license lines, text hygiene and Markdown links |
| `make format` | applies clang-format and ruff |
| `make release` | builds both kexts from the committed sources, with the Xcode `.xcode-version` names, into `dist/graft-<version>/` and `dist/graft-<version>.zip` |
| `make clean` | removes `build/` |

## Checks

Every commit builds and passes:

```sh
make test kernel-check kernel-check-arm64e graft-arm64e core-probe route-probe examples lint
```

CI (`.github/workflows/ci.yml`) selects the Xcode that `.xcode-version` pins,
builds the kext and runs `make test`, the kernel-flag builds, the arm64e
compile check, the test kexts ([tests/kexts/](tests/kexts/README.md)), the
examples, `make lint` and `make release` on a macOS runner.
It keeps `dist/` as the run's artifact, and for a pushed tag `v<version>` it
publishes the release ([docs/guides/releasing.md](docs/guides/releasing.md)).

A failing check is fixed, never loosened. Fix the code or the expectation. If a
check must be disabled, say why next to it.

The host tests cover everything that needs no kernel. What only a running
guest shows, such as a route being entered or a real consumer binding and
starting, is checked by booting: [tools/](tools/README.md) describes
`boot-core.py`, which boots Graft and the test kexts in a disposable x86-64
guest.

## Code

- C, C++ and Objective-C follow `.clang-format`, and Python follows
  `ruff.toml`. Both use 100 columns. `make format` applies them.
- Every code file starts with the line
  `SPDX-License-Identifier: GPL-2.0-or-later OR LGPL-3.0-or-later`.
- The kext is C++17 with no exceptions, no RTTI and no standard library.
- The SDK's `graft-kext` (`sdk/tool/`) is C++17 with only the standard library,
  because every consumer's build compiles it. It runs no tool but Xcode's, and
  its `check-prelinked` runs none, so that command builds on any host.
  `make test` compiles it with warnings as errors; a consumer's build keeps
  them warnings, so that a newer compiler does not stop it.
- The manifest's records, constants and validator are defined once, in
  `sdk/include/graft/ManifestFormat.h`, which the kext and `graft-kext` both
  compile. It needs only `<stddef.h>`, `<stdint.h>` and `<string.h>`.
- What differs between x86-64 and arm64e lives in `src/x86_64/` and
  `src/arm64e/`, behind the interfaces in `src/`
  ([src/README.md](src/README.md#the-architecture-boundary)).
- Graft decides from what the running system has. Add no macOS version check,
  no per-release table and no boot-arg.
- Graft runs with SIP on. A change may not need SIP off, an AMFI-relaxing
  boot-arg, or any boot-arg of Graft's own.
- Keep it simple. Handle what the SDK and the supported boot paths produce, and
  add no option, switch or environment variable without a concrete use.
- Leave no dead code, no commented-out code and no stopgap.

## What may enter the repository

- **Apple's code and data.** Never commit Apple's code or binaries, or anything
  taken from them or from a guest: disassembly, dumps, excerpts of kernel
  collections, per-release offsets, sizes, slot counts or discriminators, or
  logs that carry any of these. You may read Apple's binaries to diagnose a
  problem. What you learn goes in as code that finds the fact in the running
  system, or as a value computed from Apple's published headers.
- **Test inputs.** Tests use synthetic data, or binaries built from this
  repository's own source, such as `tests/fixture.c` and the consumers in
  `tests/consumer/` and `tests/synthesized/`.
- **Third-party code.** Graft has none, at build time or at boot, and takes
  none.

## Documentation

`README.md`, `docs/` and each directory's README describe Graft as it is, in
plain sentences a reader can check against the code. Change them in the same
commit as the code they describe. History belongs in git and in the
changelog, not in the docs.

## Commits

- Changes land on `development` as a straight line of small commits. Rebase
  onto it; do not merge.
- A commit message starts with an emoji code naming its kind (`:new:`,
  `:bug:`, `:art:`, `:fire:`, `:memo:`, `:white_check_mark:`), then a subject
  in the imperative of at most 72 characters. The body is wrapped at 80
  columns and says why.
- Commit each step once it works, rather than one large change.

## Releases

A release is a commit with a version bump and a changelog entry, tagged `v`
plus the version. CI builds the release from the tag, with the Xcode that
`.xcode-version` pins, and publishes it as a GitHub release; nothing built is
committed. [docs/guides/releasing.md](docs/guides/releasing.md) gives the
steps.

## License

Graft is free software under the GNU General Public License, version 2 or
later, or the GNU Lesser General Public License, version 3 or later, at your
option ([LICENSE](LICENSE)).
