# Developing reims-vgpu

The repository's layout, its tests, and the Rust library's crates, commands, build profiles and
features. [README.md](../README.md) lists what working on the code needs and the `make` targets,
and [AGENTS.md](../AGENTS.md) has the rules for changing it.

## Repo layout

```text
AGENTS.md                   the rules for working on the repository
Cargo.toml                  the Cargo workspace, with Cargo.lock,
                            rust-toolchain.toml, clippy.toml and
                            .cargo/config.toml
crates/                     its six crates (below), and reims-ctl, a
                            workspace of its own that links none of them
xtask/                      cargo xtask (below)
firmware/                   the option ROM: the PCI device's UEFI driver,
                            in C
tests/
  suite/                    the behaviour-lock suite (its README)
scripts/
  install-ctl.sh            builds and installs reims-ctl and its QEMU (and
                            reims-util, on Linux)
  run_vmapple.sh            boots a macOS VM on make qemu's QEMU
  vm_drive.py, vm_ocr.swift drive a VM through a fixed tour
  walk_window_macos.sh      the manual walk of a display's window on macOS
docs/
  api.md                    the C API and the Rust API it wraps, with the
                            settings and environment variables
  architecture.md           how the code is built
  conventions.md            the conventions the code follows
  development.md            this file: the layout, the tests, the crates
                            and their commands
  testing.md                the suite's specification, and the guest ABI
  qemu.md                   QEMU on the library: its build, VM boots and
                            what they measured
  lessons.md                traps the suite and reviews found
  accuracy-ledger.md        what the Vulkan platform's defaults trade
                            against Metal's answers, and the switches
vendor/qemu/                QEMU with apple-gfx on reims-vgpu (submodule)
assets/readme/              the README's screenshots
LICENSE                     the license, and LICENSES/ its texts (README.md, License)
```

## Testing

- **`cargo xtask check`** runs the lints, rustfmt, clippy and every unit test. Run it once before a
  branch that changes code merges.
- **`cargo xtask check-ctl`** runs rustfmt, clippy and the tests for `reims-ctl`, whose separate
  workspace is intentionally outside `cargo xtask check`. `make check-ctl` is the equivalent
  Makefile target.
- **`make smoke`** (`cargo xtask smoke`) links a guest program with the library the way QEMU links
  it and runs an isolated guest task.
- **The behaviour-lock suite** (`tests/suite`, `make test`) plays the hypervisor and the guest, and
  compares what the guest and the host observe with its expected files.
  [testing.md](testing.md) specifies the suite and the guest ABI it is written against, and
  [tests/suite/README.md](../tests/suite/README.md) is its user's guide.
- **The pins**: `crates/reims-vgpu-c/tests/pins`, a C host against the C API, which
  `cargo xtask check` runs with the other tests.

Expected files and pin files are recorded from the checkout's own build. A change that means to
alter behaviour records them again in the same commit (`make bless TEST=<patterns> ARGS=--update`,
`RVG_PINS_RECORD=1 cargo test -p reims-vgpu-c --test pins`), so the diff shows exactly what changed
for the guest and the host.

## The Rust library

[architecture.md](architecture.md) describes the crates, threads and locks, the raise and death
model, the platform boundary, isolated tasks, the APIs, the option ROM and QEMU. Each module's `//!`
doc holds its area's ordering contracts and the measured platform facts it relies on.

### Crates

| crate | path | role |
|---|---|---|
| `reims-vgpu-core` | `crates/reims-vgpu-core` | all behaviour, generic over `P: Platform`; links no Apple framework |
| `reims-vgpu-metal` | `crates/reims-vgpu-metal` | the `Platform` over Metal, IOSurface, Mach VM and threads, and the library's shaders (`src/gpu/shaders.metal`) |
| `reims-vgpu-vulkan` | `crates/reims-vgpu-vulkan` | the `Platform` over metal2vulkan's runtime: the library on Linux, and on macOS behind the `vulkan` features, on MoltenVK |
| `reims-vgpu-window` | `crates/reims-vgpu-window` | a display's own window: AppKit's or SDL3's on macOS, SDL3's on Linux, its input and the host's screens; knows nothing of devices or displays. `features.md` lists what the window does, each entry deliberate |
| `reims-vgpu` | `crates/reims-vgpu` | the public Rust API; the process's platform; a display's window over the window crate |
| `reims-vgpu-c` | `crates/reims-vgpu-c` | the C API (`rvg_*`, `include/reims_vgpu.h`), which QEMU links; `cdylib`, `staticlib` and `rlib` |
| `xtask` | `xtask` | `cargo xtask`, below |

### Build and test

Run from the repository root. `cargo xtask` is an alias for `cargo run --package xtask --`
(`.cargo/config.toml`), and with no command it lists the commands.

```sh
cargo build --workspace
cargo nextest run --workspace
cargo xtask check          # the lints, rustfmt, clippy and every test; run it once before a merge
cargo xtask check-ctl      # rustfmt, clippy and tests for the separate reims-ctl workspace
cargo xtask dist           # target/dist: the library, its header, reims-vgpu.pc and
                           # the option ROM, for a host that links it through pkg-config
cargo xtask smoke          # a guest linked with the library as QEMU links it (make smoke)
```

| command | what it does |
|---|---|
| `testkit` | core's `testkit` module is gated, and only dev-dependencies turn its feature on (D50) |
| `header [--check]` | `include/reims_vgpu.h` from cbindgen; `--check` fails when the committed header drifts from cbindgen's output |
| `dist` | the library as a plain library, below |
| `smoke` | the arm64 smoke of the library, below |
| `check` | `testkit`, `header --check`, rustfmt, clippy and the tests ([architecture.md](architecture.md) 13.1); `--no-cargo` skips the cargo steps |

`cargo xtask check` tests with every feature but `reims-vgpu-c/vulkan`, which puts the C API on the
Vulkan platform, and lints with every feature too. On a Mac it also clippy-checks the Linux target,
and, where Rosetta is installed, runs core's tests as `x86_64-apple-darwin`.

#### Dist

```sh
cargo xtask dist                   # release, into target/dist
cargo xtask dist --profile checked --out /tmp/reims-vgpu
cargo xtask dist --platform vulkan # the Vulkan platform, into target/dist-vulkan
```

`cargo xtask dist` builds `reims-vgpu-c` and writes the library for a host that links it the way it
links its other dependencies, through pkg-config:

```text
target/dist/
  include/reims_vgpu.h                   the committed header
  lib/libreims_vgpu.a                    the staticlib
  lib/libreims_vgpu.dylib                the cdylib, @rpath/libreims_vgpu.dylib, signed ad hoc
  lib/pkgconfig/reims-vgpu.pc            the package
  share/reims-vgpu/ReimsVgpu.rom         the EFI option ROM, built from firmware/
  share/doc/reims-vgpu/                  the license: LICENSE and LICENSES/, as in the repository
```

`reims-vgpu.pc` takes its prefix from its own path, so the tree can move. Its `Libs.private` lists
the system frameworks and libraries a static link needs, and its `optionrom` variable names the
ROM, which QEMU installs as `apple-gfx-pci`'s default `romfile`. QEMU and the smoke link
`libreims_vgpu.a`, so the library is part of their binaries; the suite's runner links
`libreims_vgpu.dylib` with an `LC_RPATH` at the tree's `lib`.

`--platform vulkan` builds the C API on the Vulkan platform into `target/dist-vulkan`, laid out the
same; the smoke (`--dist`) and the suite (`pgtest run --dist`) take either tree. On Linux `dist`
builds that platform alone, with `libreims_vgpu.so` in place of the dylib (architecture 11.3).
`CARGO_TARGET_DIR` moves every output, the dist tree included, out of `target`.

#### Smoke

```sh
cargo xtask dist
cargo xtask smoke           # --dist DIR for another tree
```

It runs arm64 only: a guest program (`xtask/src/smoke_guest.m`), linked statically with the dist
tree's library, that brings up a device and runs an isolated task, whose facts must equal the
library's expected values (`GUEST_EXPECTED` in `xtask/src/smoke.rs`). `make smoke` builds the tree
and runs it.

#### The behaviour suite

The suite (`tests/suite`) runs against the library, by default the checkout's dist tree
(`target/dist`):

```sh
make test                   # the dist tree, then the suite
cd tests/suite && python3 -m pgtest run -j 32   # the same, with pgtest's options
make bless TEST=<patterns>  # record new or changed scenarios from the checkout's build
make bless TEST=<patterns> ARGS=--update   # record them again after a deliberate change
```

[testing.md](testing.md) 5.7 and [tests/suite/README.md](../tests/suite/README.md) have the rest.

### Profiles

Set in `Cargo.toml` (architecture 2.3):

| profile | settings | why |
|---|---|---|
| `dev` | `panic = "unwind"`, overflow checks on | an unannotated arithmetic slip dies in tests |
| `release` | `panic = "unwind"`, overflow checks off, `debug = "line-tables-only"`, thin LTO | the product. Wrapping is explicit where the guest's values wrap (D19), so the checks add nothing |
| `checked` | `release` plus overflow checks and debug assertions | a second pass of the suite: a death here is an arithmetic site review missed |

- **`panic = "unwind"` everywhere** (D9). objc2's `exception::catch` needs it. It is not a way to
  survive a panic: `reims-vgpu`'s singleton installs a hook that aborts before anything unwinds, so
  an unintended panic dies at its op with no side effects.
- **objc2's debug assertions are on in `dev`** and off in `checked`
  (`[profile.checked.package.objc2]`, D36). They panic on a message to nil, a missing method and an
  encoding the runtime disagrees with, which the library never sends, so the tests that send to
  real Metal check each send. Declarations are also checked with `AnyClass::verify_sel` in
  `reims-vgpu-metal/tests/spi_encodings.rs`.
- objc2's `exception` feature is always on, and `catch-all` never.

### Features

- `reims-vgpu/vulkan` and `reims-vgpu-c/vulkan`: the Vulkan platform as the API's, and the C API's,
  platform. Off by default on macOS; on Linux the library builds only with them.
- `macos-sdl` (`reims-vgpu-window`, forwarded by `reims-vgpu` and `reims-vgpu-c`), on by default:
  SDL3's window on macOS, which `REIMS_VGPU_WINDOW_IMPL=sdl` opens in place of AppKit's. Without it
  the macOS library has no SDL.
- `reims-vgpu-window/sdl-no-link`: SDL left unbuilt, for the Linux target's clippy on a Mac.
- `reims-vgpu-core/testkit`: `core::testkit` for other crates' tests.
- `reims-vgpu/testkit`: the same, through the public crate, and each handle's core object.

[api.md](api.md) section 2 lists the device's settings and the environment variables the library
reads.
