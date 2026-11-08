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
crates/                     its eight crates (below), and reims-ctl, a
                            workspace of its own that links none of them
xtask/                      cargo xtask (below)
firmware/                   the reims-graphics device's option ROM: its UEFI
                            driver, in C, and the prebuilt ROM reims-ctl
                            carries; in firmware/ambrosia, the device's
                            driver module for Ambrosia, arm64 guests'
                            firmware
guest/                      the guest's kexts, loader and Metal bundle, and
                            the prebuilt kext set reims-ctl carries
tests/
  shaders/                  the tests' metallibs and their sources
  suite/fixtures/shaders    a link to tests/shaders, which guest/bundle's
                            build and tools name
scripts/                    what a person installing reims-ctl runs
  install-ctl.sh            builds and installs reims-ctl and its QEMU (and
                            reims-util, on Linux)
scripts/internal/           what developing the project runs
  gpu-locks.sh              the GPU locks a run takes on a host whose GPUs
                            other projects' runs share
  dev-env.sh                the environment a developer's builds run in,
                            through kache where it is installed (below)
  new-worktree.sh           a new worktree on a new branch
  lane-gate.sh, lane-gate/  the graphics lane's landing gate
  opencore-add-kexts.sh     a copy of a VM's OpenCore image with kexts added
  guest-minibuddy-off.sh    keeps Setup Assistant's panes from a guest's
                            next login
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
  testing.md                how the code is tested, and the tags a claim
                            about Metal carries
  qemu.md                   QEMU on the library: its build, VM boots and
                            what they measured
  lessons.md                traps that tests, reviews and boots found
  accuracy-ledger.md        what the Vulkan platform's defaults trade
                            against Metal's answers, and the switches
  graphics-kext/            the reims-graphics device: its design, the dGPU
                            oracle and the benchmark
  video/                    the device's host video: what it answers, its
                            arms, the video crate and the open bugs
vendor/qemu/                QEMU with the reims-graphics device on
                            reims-vgpu (submodule)
assets/readme/              the README's screenshots
LICENSE                     the license, and LICENSES/ its texts (README.md, License)
```

## Testing

- **`cargo xtask check`** runs the lints, rustfmt, clippy and every unit test. Run it once before a
  branch that changes code merges.
- **Unit tests** sit beside the code. A test of the Metal or the Vulkan platform runs on the host's
  GPU, and on a Mac the Vulkan platform's tests compare its answers with Metal's
  ([testing.md](testing.md)).
- **VM boots** are the only check of what a guest's frameworks choose and of QEMU's side of the C
  API ([qemu.md](qemu.md), "Booting a guest to check a change").

A change that means to alter behaviour changes the tests that hold it in the same commit, so the
diff shows exactly what changed for the guest and the host.

## The Rust library

[architecture.md](architecture.md) describes the crates, threads and locks, the raise and death
model, the platform boundary, the APIs, the option ROM and QEMU. Each module's `//!`
doc holds its area's ordering contracts and the measured platform facts it relies on.

### Crates

| crate | path | role |
|---|---|---|
| `reims-vgpu-core` | `crates/reims-vgpu-core` | what the platforms and the device share, generic over `P: Platform`: the platform traits, the runtime, raises and deaths, the testkit; links no Apple framework |
| `reims-vgpu-vulkan` | `crates/reims-vgpu-vulkan` | the `Platform` over metal2vulkan's runtime, the library's only one: on Linux, and on macOS over MoltenVK, a build for development and tests; its OS services (on macOS IOSurface, Mach VM, threads and the Objective-C message layer the window's AppKit side sends through), and the library's shaders (`src/gpu/shaders.metal`) |
| `reims-vgpu-window` | `crates/reims-vgpu-window` | the device's own window: AppKit's or SDL3's on macOS, SDL3's on Linux, its input and the host's screens; knows nothing of devices or displays. `features.md` lists what the window does, each entry deliberate |
| `reims-vgpu` | `crates/reims-vgpu` | the public Rust API: the reims-graphics device over the platform, its Metal-call decoder and its window; the process's platform |
| `reims-vgpu-c` | `crates/reims-vgpu-c` | the C API (`rvg_*`, `include/reims_vgpu.h`), which QEMU links; `cdylib`, `staticlib` and `rlib` |
| `reims-vgpu-video` | `crates/reims-vgpu-video` | the reims-graphics device's video on the host, independent of the library and QEMU: `core` (what the video arms share), `codec` (the codecs above the hardware, in plain Rust), and on Linux `vaapi` (VA-API through libva) and `nvidia` (NVDEC and NVENC over the CUDA driver API), their libraries loaded at run time |
| `reims-vgpu-video-conformance` | `crates/reims-vgpu-video/conformance` | the test streams' readers and writers the video examples use: MP4 in and out, and frames hashed as ffmpeg's `framemd5` does |
| `reims-graphics` | `crates/reims-graphics` | the reims-graphics device (1234:5247, `docs/graphics-kext/design.md`): its wire format, whose C header (`include/reims_graphics_wire.h`) the guest's kext and option ROM include, and its host side |
| `xtask` | `xtask` | `cargo xtask`, below |

### Build and test

Run from the repository root. `cargo xtask` is an alias for `cargo run --package xtask --`
(`.cargo/config.toml`), and with no command it lists the commands.

```sh
cargo build --workspace
cargo nextest run --workspace
cargo xtask check          # the lints, rustfmt, clippy and every test; run it once before a merge
cargo xtask dist           # target/dist-vulkan: the library, its header and reims-vgpu.pc,
                           # for a host that links it through pkg-config
cargo xtask metallibs      # on a Mac: the Vulkan platform's metallibs and their record
```

| command | what it does |
|---|---|
| `testkit` | core's `testkit` module is gated, and only dev-dependencies turn its feature on (D50) |
| `header [--check]` | `include/reims_vgpu.h` from cbindgen; `--check` fails when the committed header drifts from cbindgen's output |
| `dist` | the library as a plain library, below |
| `metallibs` | the Vulkan platform's metallibs and their record, `metallibs.toml`, compiled on a Mac into `crates/reims-vgpu-vulkan/metallibs`, which a build elsewhere takes them from |
| `check` | `testkit`, `header --check`, the records of the prebuilt option ROM and kext set, rustfmt, clippy and the tests ([architecture.md](architecture.md) 13.1); `--no-cargo` skips the cargo steps |

`cargo xtask check` tests with every feature but `reims-vgpu-c/vulkan`, which puts the C API on the
Vulkan platform, and lints with every feature too. On a Mac it also clippy-checks the Linux target,
and, where Rosetta is installed, runs core's tests as `x86_64-apple-darwin`. Its cargo steps run
in the environment it was started in, incremental as cargo's dev and test profiles are, so a check
run again after a fix rebuilds what the fix changed.

#### Building through kache

[kache](https://github.com/kunobi-ninja/kache) is a compiler cache: run in front of rustc, it stores
each unit's outputs on the host, keyed by everything the unit is built from, and restores them as
clones of the stored files (APFS clones on a Mac, reflinks on btrfs), so a worktree builds only the
units no tree on the host built before. It is a tool on the PATH, version 2.0.0 or later (`cargo
install --locked kache --version 2.0.0`, which installs `kache` and `kache-engine` together; run
`kache daemon restart` after an upgrade), never a dependency and never Cargo's configured wrapper:
a host without it builds as before. A
developer's builds take it up through `scripts/internal/dev-env.sh`, sourced from the worktree's
root before each cargo command:

```sh
. scripts/internal/dev-env.sh && cargo nextest run -p reims-vgpu-core
```

which, where `kache` is on the PATH, sets `RUSTC_WRAPPER=kache`, `KACHE_LOCAL_ONLY=1` and
`KACHE_BUILD_SCRIPT_HERMETIC=1`, and leaves the build incremental (it unsets `CARGO_INCREMENTAL`).
`cargo xtask check` and `make` run their cargo steps in the environment they are started in.

Builds are incremental, and kache keeps their incremental state (`adaptive_incremental` and
`seed_incremental` in the host's configuration, below). A unit whose inputs match an entry in the
store is restored from it. A workspace crate or path dependency whose sources or dependencies
changed since its last build in the same target directory (an edit, a pull, a branch switch, a
rebase) compiles incrementally instead, with incremental state of its own in the target directory,
and kache neither stores nor restores it for up to eight builds, each within 15 minutes of the
last. A compile error keeps the state. A worktree's first incremental compile of a unit imports
compatible finished incremental state from another worktree's build of it on the host, as clones,
so a new worktree's edits start warm (rustc 1.98 or later; read from kache 2.0.0's documentation).

kache seeds only between target directories inside their worktrees, cargo's default `target/` with
no `CARGO_TARGET_DIR`. It maps the compiler's working directory, the worktree's root, to one path
(`-Zremap-cwd-prefix=/src`), and a target directory elsewhere puts a path of its own in the
compiler's arguments, so each target's state is compatible with no other's. Measured on the M3 Max
on 2026-11-03, the first edit of `reims-vgpu-core` after a build of the tree: with targets inside
the worktrees, a second worktree's compile seeded from the first's state took 4.3 s, against 7.0 s
for the first worktree's, which had no donor; with each target in a directory of its own under
`~/.cache`, the two units' seed identities differed and the second took 5.7 s, against 6.0 s.

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
local_max_size = "100GiB"           # the store's ceiling, kept by kache (40GiB on Linux)
daemon_idle_timeout_secs = 1800     # the daemon exits when idle
auto_clean_orphaned_targets = false # kache never removes or rewrites a target directory
auto_share_target_files = false
auto_clean_unused_units_days = 0
auto_clean_idle_targets_days = 0
auto_recover_targets = false
seed_new_targets = false            # nor copies one tree's build units into another's
adaptive_incremental = true         # a unit being edited compiles incrementally
seed_incremental = true             # its incremental state seeded from another worktree's
```

Never run `kache init`, which writes the wrapper into `~/.cargo/config.toml` for every project on
the host, links compiler shims into the shell's PATH and installs a login service, nor `kache
login`. kache's store is under `~/Library/Caches/kache` on a Mac and `~/.cache/kache` on Linux:
`kache stats` shows its size and limit, `kache explain CRATE` why a unit missed, and removing the
directory (with no build running) empties it. kache 2 compiles through its daemon (`kache-engine`),
which a build through kache starts where none runs, as `kache stats` does, and which exits after
`daemon_idle_timeout_secs` with nothing to do (`kache daemon stop` ends it at once). The
ceiling counts what the store alone holds: evicting a file still cloned into a target directory
frees nothing, so kache keeps it until that target directory is gone.

#### Dist

```sh
cargo xtask dist                   # release, into target/dist-vulkan
cargo xtask dist --profile checked --out /tmp/reims-vgpu
```

`cargo xtask dist` builds `reims-vgpu-c` and writes the library for a host that links it the way it
links its other dependencies, through pkg-config:

```text
target/dist-vulkan/
  include/reims_vgpu.h                   the committed header
  lib/libreims_vgpu.a                    the staticlib
  lib/libreims_vgpu.so                   the cdylib (on a Mac libreims_vgpu.dylib,
                                         @rpath/libreims_vgpu.dylib, signed ad hoc)
  lib/pkgconfig/reims-vgpu.pc            the package
  share/doc/reims-vgpu/                  the license: LICENSE and LICENSES/, as in the repository
```

`reims-vgpu.pc` takes its prefix from its own path, so the tree can move. Its `Libs.private` lists
the system frameworks and libraries a static link needs. QEMU links `libreims_vgpu.a`, so the
library is part of its binary; the C API's tests link the shared library.

The library is built on the Vulkan platform, its only one (architecture 11.3); `--platform vulkan`
names it for the builds that pass it, and any other platform is refused. `CARGO_TARGET_DIR` moves
every output, the dist tree included, out of `target`.

### Profiles

Set in `Cargo.toml` (architecture 2.3):

| profile | settings | why |
|---|---|---|
| `dev` | `panic = "unwind"`, overflow checks on, `debug = "line-tables-only"`, dependencies without debuginfo | an unannotated arithmetic slip dies in tests. Line tables keep a backtrace's files and lines, at a fraction of full debuginfo's build time and disk |
| `debugging` | `dev` with full debuginfo, dependencies' too | a debugger session's build (`--profile debugging`), whose variables and types lldb shows |
| `release` | `panic = "unwind"`, overflow checks off, `debug = "line-tables-only"`, thin LTO | the product. Wrapping is explicit where the guest's values wrap (D19), so the checks add nothing |
| `checked` | `release` plus overflow checks and debug assertions | a second pass of the tests: a death here is an arithmetic site review missed |

- **`panic = "unwind"` everywhere** (D9). objc2's `exception::catch` needs it. It is not a way to
  survive a panic: `reims-vgpu`'s singleton installs a hook that aborts before anything unwinds, so
  an unintended panic dies at its op with no side effects.
- **objc2's debug assertions are on in `dev`** and off in `checked`
  (`[profile.checked.package.objc2]`, D36). They panic on a message to nil, a missing method and an
  encoding the runtime disagrees with, which the library never sends, so the tests that send to
  real Metal check each send.
- objc2's `exception` feature is always on, and `catch-all` never.

### Features

- `reims-vgpu/vulkan` and `reims-vgpu-c/vulkan`: they change nothing, as the Vulkan platform is
  the library's only one, and stay for the builds that name them.
- `macos-sdl` (`reims-vgpu-window`, forwarded by `reims-vgpu` and `reims-vgpu-c`), on by default:
  SDL3's window on macOS, which `REIMS_VGPU_WINDOW_IMPL=sdl` opens in place of AppKit's. Without it
  the macOS library has no SDL.
- `reims-vgpu-window/sdl-no-link`: SDL left unbuilt, for the Linux target's clippy on a Mac.
- `reims-vgpu-core/testkit`: `core::testkit` for other crates' tests.
- `reims-vgpu/testkit`: the same, through the public crate, and each handle's core object.

[api.md](api.md) section 2 lists the device's settings and the environment variables the library
reads.
