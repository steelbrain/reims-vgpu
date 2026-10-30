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
crates/                     its twelve crates (below), and reims-ctl, a
                            workspace of its own that links none of them
xtask/                      cargo xtask (below)
firmware/                   the option ROM: the PCI device's UEFI driver,
                            in C
tests/
  suite/                    the behaviour-lock suite (its README)
scripts/                    what a person installing reims-ctl runs
  install-ctl.sh            builds and installs reims-ctl and its QEMU (and
                            reims-util, on Linux)
scripts/internal/           what developing the project runs
  gpu-locks.sh              the GPU locks a run takes on a host whose GPUs
                            other projects' runs share
  dev-env.sh                the environment a developer's builds run in,
                            through kache where it is installed (below)
  new-worktree.sh           a new worktree, its tests/suite cloned from the
                            main checkout's
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
  video/                    the video devices: their contract, core, the
                            backends, hardening and the open bugs
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
| `reims-vgpu-video-core` | `crates/reims-vgpu-video-core` | the VideoToolbox and M2 scaler devices' behaviour as the guest sees it, over the backends' traits; independent of host, driver and QEMU |
| `reims-vgpu-video-codec` | `crates/reims-vgpu-video-codec` | the codecs' syntax and the decoding and encoding processes above the hardware, for every backend, in plain Rust |
| `reims-vgpu-video-vaapi` | `crates/reims-vgpu-video-vaapi` | the VA-API backend (Linux), through libva alone |
| `reims-vgpu-video-nvidia` | `crates/reims-vgpu-video-nvidia` | the NVIDIA backend (Linux): NVDEC and NVENC over the CUDA driver API, loaded at run time |
| `reims-vgpu-video-videotoolbox` | `crates/reims-vgpu-video-videotoolbox` | the VideoToolbox backend (macOS) |
| `reims-vgpu-video-conformance` | `crates/reims-vgpu-video-conformance` | the conformance suite every backend runs: a stand-in guest drives core's devices on it |
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
and, where Rosetta is installed, runs core's tests as `x86_64-apple-darwin`. Its cargo steps run
with `CARGO_INCREMENTAL=0`: each builds once in its tree, where incremental caches would only take
disk.

#### Building through kache

[kache](https://github.com/kunobi-ninja/kache) is a compiler cache: run in front of rustc, it stores
each unit's outputs on the host, keyed by everything the unit is built from, and restores them as
clones of the stored files (APFS clones on a Mac, reflinks on btrfs), so a worktree builds only the
units no tree on the host built before. It is a tool on the PATH (`cargo install --locked kache`),
never a dependency and never Cargo's configured wrapper: a host without it builds as before. A
developer's builds take it up through `scripts/internal/dev-env.sh`, sourced from the worktree's
root before each cargo command:

```sh
. scripts/internal/dev-env.sh && cargo nextest run -p reims-vgpu-core
```

which sets `CARGO_INCREMENTAL=0` and, where `kache` is on the PATH, `RUSTC_WRAPPER=kache`,
`KACHE_LOCAL_ONLY=1`, `KACHE_ADAPTIVE_INCREMENTAL=0` and `KACHE_BUILD_SCRIPT_HERMETIC=1`. `cargo
xtask check` and `make` run their cargo steps in the environment they are started in.

A unit is shared between worktrees when nothing it is built from names the worktree's path, and a
build script's run when nothing it writes does. So the tests find the tree at run time: they read
`CARGO_MANIFEST_DIR` and `OUT_DIR` with `std::env::var_os`, which cargo and nextest set for a
package's test processes (`OUT_DIR` where the package has a build script), and
`CARGO_TARGET_TMPDIR`, which neither sets at run time, the same way, with the system's temporary
directory in its place (measured with cargo 1.98.1 and nextest 0.9.140). `env!` of any of them
compiles the worktree's path into the test. `include_bytes!` and `include_str!` compile in a file's
bytes, which kache keys by content, so a test includes a fixture unless the fixture's bytes name the
worktree: the Vulkan tests read their import metallibs, whose install names are paths in `OUT_DIR`,
at run time. These are built again in each worktree:
- `xtask` and its tests: the binary finds the repository from where it was built
  (`env!("CARGO_MANIFEST_DIR")`), so a run of another tree's binary acts on the tree that built it;
- on a Mac, the Vulkan crate's build script, whose import metallibs name its `OUT_DIR`: a
  workspace package's hermetic run (below) is keyed by its manifest directory.

A build script runs hermetically (`KACHE_BUILD_SCRIPT_HERMETIC=1`): kache runs it with its `OUT_DIR`
in the store, under `out-dirs/v2/`, in a directory named by everything the script sees but its
`OUT_DIR`, and makes the target directory's `OUT_DIR` a link to it. A registry package's run is
then shared between worktrees even where what it writes names its `OUT_DIR`, as SDL3's library
does: its objects name their build directory there (`sdl3-sys`), so otherwise each worktree builds
SDL again (22 s on the Ryzen 7 7840HS, 31 to 67 s on the M3 Max beside other work) and links a
library of other bytes into `sdl3-sys`, the C API's library and the tests of the window,
`reims-vgpu` and the C API. A run kache cannot share (one that fails, writes outside its `OUT_DIR`
or declares other inputs) runs in the target directory, as without it. What ships is the same: a
host without kache builds SDL in its target directory, and through kache only SDL's objects' debug
build directory names the store instead. kache's gc (`kache gc`, and the daemon's) removes a run
no target directory has linked to for a week; removing the store leaves the links dangling, so
remove the target directories built through it with it.

Measured on 2026-10-26: a second worktree's build from an empty target directory, after a first
worktree had built the same tree through kache (`cargo build --workspace`, then `cargo nextest run
--workspace --no-run`, with `cargo xtask check`'s features), and the test targets of the 31 it
built that came from the store:

| | Ryzen 7 7840HS, Linux | M3 Max, macOS, beside other work |
|---|---|---|
| tests compiling the tree's paths in | 170 s (26 s + 144 s), 23 | 216 s (36 s + 180 s), 21 |
| tests reading them at run time | 30 s (26 s + 5 s), 24 | 87 s (37 s + 50 s), 24 |
| and build scripts run hermetically | 1.5 s (0.7 s + 0.8 s), 30 | 9.5 s (6.1 s + 3.4 s), 30 |

Each dependency and workspace library came from the store but `sdl3-sys` (all of them with hermetic
runs), and in the last row only `xtask` and its tests compiled. The M3 Max ran them beside
another project's tests (load averages 11 to 34), so its times are upper bounds.

kache does not restore what a build script writes outside its `OUT_DIR`. The one such write here,
`cargo xtask metallibs`' export, always runs: the export rewrites the record it declares as an
input, and kache records no run whose declared input changed while it ran (measured with kache
1.2.2).

Each host keeps one configuration, `~/.config/kache/config.toml`, read by every build through
kache on it:

```toml
[cache]
local_only = true                   # no remote storage, no planner: nothing leaves the host
local_max_size = "60GiB"            # the store's ceiling, kept by kache (40GiB on Linux)
daemon_idle_timeout_secs = 1800     # a daemon some command started exits when idle
auto_clean_orphaned_targets = false # kache never removes or rewrites a target directory
auto_share_target_files = false
auto_clean_unused_units_days = 0
auto_clean_idle_targets_days = 0
auto_recover_targets = false
seed_new_targets = false            # nor copies one tree's build units into another's
adaptive_incremental = false        # no incremental state in target directories
```

Never run `kache init`, which writes the wrapper into `~/.cargo/config.toml` for every project on
the host, links compiler shims into the shell's PATH and installs a login service, nor `kache
login`. kache's store is under `~/Library/Caches/kache` on a Mac and `~/.cache/kache` on Linux:
`kache stats` shows its size and limit, `kache explain CRATE` why a unit missed, and removing the
directory (with no build running) empties it. Builds through kache start no daemon, but `kache stats`
does, with no way around it, and the daemon exits after `daemon_idle_timeout_secs` with nothing to
do (`kache daemon stop` ends it at once); `kache doctor` and `kache explain` start none. The
ceiling counts what the store alone holds: evicting a file still cloned into a target directory
frees nothing, so kache keeps it until that target directory is gone.

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
| `dev` | `panic = "unwind"`, overflow checks on, `debug = "line-tables-only"`, dependencies without debuginfo | an unannotated arithmetic slip dies in tests. Line tables keep a backtrace's files and lines, at a fraction of full debuginfo's build time and disk |
| `debugging` | `dev` with full debuginfo, dependencies' too | a debugger session's build (`--profile debugging`), whose variables and types lldb shows |
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
