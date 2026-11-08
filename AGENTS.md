# Working on this repository

This repository builds `libreims_vgpu`, a Rust library that is the host
side of the reims-graphics device (PCI 1234:5247), the GPU our own guest
drivers drive in x86-64 macOS guests:
- the library and its C API, `reims_vgpu.h`, which `cargo xtask dist`
  writes into its dist tree with a pkg-config package;
- the guest's side (`guest/`): our kexts, their loader and the Metal bundle,
  which reims-ctl adds to an x86-64 VM's OpenCore image as a prebuilt set.
  They are to work on every x86-64 macOS release from 11.0 to 26's last;
- the option ROM (`firmware/`), the device's UEFI driver in C, which an
  x86-64 guest's firmware runs;
- QEMU's `reims-graphics-pci` (`vendor/qemu`), in C on that API, which finds
  the library through pkg-config and links it statically;
- `reims-ctl` (`crates/reims-ctl`), the VM manager users run.

The guest's Metal bundle serialises its Metal calls onto the device's rings,
and the library carries them out on the host's GPU: on Metal on macOS, and
on Linux on the Vulkan platform (`crates/reims-vgpu-vulkan`), over
metal2vulkan's Metal-shaped runtime, which a Mac can also build on MoltenVK.
The device's video decodes run on the host's VA-API or NVDEC.

[README.md](README.md) says what the project is, and
[docs/development.md](docs/development.md) how it is laid out. This file
covers what holds the code, how to check a change, and the rules for
changing it. [docs/architecture.md](docs/architecture.md) describes the
code, [docs/graphics-kext/design.md](docs/graphics-kext/design.md) the
device and its protocol, and [docs/conventions.md](docs/conventions.md) the
conventions the code follows.

**The codebase is its own reference.** Its unit tests record what it does.
Every merge ends green: the merged tree passes what the tree before it
passed. A change meant to alter behaviour changes the tests that hold the
old behaviour in the same commit, and the diff is reviewed with the change.

## Start here

```sh
make                # the library in its dist tree (cargo xtask dist): target/dist on a Mac,
                    # target/dist-vulkan on Linux: libreims_vgpu.a and the shared library,
                    # reims_vgpu.h and reims-vgpu.pc
make qemu           # then QEMU's qemu-system-x86_64 with reims-graphics-pci, in build/qemu
cargo xtask check   # lints, rustfmt, clippy and every test: run it once before a branch merges
```

The Makefile's header lists every target. README.md lists the requirements.

## Where things are

| what | where |
|---|---|
| the Cargo workspace: eleven crates and `xtask` | `Cargo.toml`, `crates/`, `xtask/` |
| `reims-ctl`, the VM manager users run, and its installer | `crates/reims-ctl` (a workspace of its own, linking nothing of the library), `scripts/install-ctl.sh`; reims-util, which it runs to make an x86-64 VM's OpenCore image and download its installer, is pinned in `crates/reims-ctl/src/reims_util.rs` |
| the C API | `crates/reims-vgpu-c/include/reims_vgpu.h`, specified in [docs/api.md](docs/api.md) |
| the device: its wire, rings and transport | `crates/reims-graphics`, whose `include/reims_graphics_wire.h` the guest's kexts and the option ROM compile; its design and protocol are [docs/graphics-kext](docs/graphics-kext) |
| the device over the platform: the Metal-call decoder, the display and its window, the host's read-back | `crates/reims-vgpu/src/graphics.rs` and `graphics/` |
| the display's shaders | `crates/reims-vgpu-metal/src/gpu/shaders.metal` |
| the Vulkan platform | `crates/reims-vgpu-vulkan`, over metal2vulkan's runtime, at the commit of its public repository `Cargo.toml` names; what its defaults trade against Metal's answers is [docs/accuracy-ledger.md](docs/accuracy-ledger.md) |
| the Vulkan platform's metallibs, which a build off a Mac copies | `crates/reims-vgpu-vulkan/metallibs`, written on a Mac by `cargo xtask metallibs` (README.md, Building) |
| a display's own window, and the register of what it does | `crates/reims-vgpu-window`, its `features.md` and its manual walk on macOS, `scripts/internal/walk_window_macos.sh`; the wiring is `crates/reims-vgpu/src/window.rs` |
| the reims-graphics device's video on the host, on NVDEC or VA-API | `crates/reims-vgpu-video-*`: `core` (what the arms share: their runtime, locks, a guest surface's memory and bounded waits), `codec` (the codecs above the hardware), `vaapi` and `nvidia` (the hardware's), and `conformance` (the test streams' readers and writers); the arms are reims-vgpu's `crates/reims-vgpu/src/graphics/{nvidia,vaapi,video_choice,video_host}.rs`, on the runtime in `crates/reims-vgpu/src/video/runtime.rs` |
| the video docs | [docs/video](docs/video): `devices.md` is what the device answers and what a decode does, `backends.md` the arms, their crates and the conformance suite (`crates/reims-vgpu-video-vaapi/tools/h264-check.sh` and `hevc-check.sh`), `core.md`, `vaapi.md` and `nvidia.md` the crates, `bugs.md` what is wrong or put off |
| the guest's kexts, loader and Metal bundle | `guest/`, and the prebuilt set reims-ctl carries, `guest/prebuilt` |
| the dist tree QEMU links | `xtask/src/dist.rs` (`cargo xtask dist`) |
| the option ROM reims-ctl gives the device | `firmware/`, and the prebuilt ROM, `firmware/prebuilt/ReimsGraphics.rom`, described in [docs/architecture.md](docs/architecture.md) section 11.4 |
| `cargo xtask` | `xtask/` |
| the tests' metallibs and their sources | `tests/shaders` (`tests/suite/fixtures/shaders` links to it, for `guest/bundle`'s build and tools) |
| QEMU and VM boots | `vendor/qemu`, `vendor/qemu/hw/display/reims-graphics-pci.c`, [docs/qemu.md](docs/qemu.md) |
| the architecture | [docs/architecture.md](docs/architecture.md) |
| traps that tests, reviews and boots caught | [docs/lessons.md](docs/lessons.md) |
| measured platform facts | the docs of the modules that rely on them |

## The contract

The guest's kexts and Metal bundle speak the device's protocol
([docs/graphics-kext/design.md](docs/graphics-kext/design.md)), and the
host drives the library through its C API, as QEMU's `reims-graphics-pci`
does. What the guest and the host observe is the contract, in its order:
- C API results: every status, returned value and refusal;
- host callbacks, with their arguments and their order, on the threads the
  header names for each ([docs/api.md](docs/api.md) section 5);
- guest memory and MMIO reads, and what the device writes into its region:
  its rings' messages and events;
- interrupts, and their order relative to guest-visible writes;
- the display's frames, pixels, modes and the cursor;
- GPU results: what the guest's Metal calls produce, Metal's answer the
  reference;
- every value that reaches the guest: object IDs, GPU addresses, resource
  indices and numbering;
- where a run dies or hangs.

An effect that shows at the API only under a race, a mid-operation release
or exhaustion is still part of it.

**What is free**: which library thread runs what where the header names no
thread, internal locking, the retention machinery (autorelease pools,
retains) where the order of releases the host sees is unchanged, Metal's
internal call order where no result or guest-visible value changes, and
diagnostics (log lines and their targets, thread and queue names, Metal
labels, crash messages, signals).

**The C API is ours, and so is its host.** QEMU's `reims-graphics-pci`, in
this repository, is the API's one host: a change to the API lands with
QEMU's side of it, and a VM boot checks the two together. `cargo xtask
header --check` keeps the committed header equal to cbindgen's output.

## Checking a change

Run the checks a change can reach, and no others. Measured on the M3 Max on
2026-11-07, the workspace's tests take 29 s of GPU time after their build
(`cargo nextest run --workspace` with the tested features, 2415 tests).
The test that faults a Mac's GPU on purpose and the one that holds the
most of its memory are opt-in (`#[ignore]`): they fail the work of every
process on the Mac, the desktop's included, so they run only for a change
to what they test (`--run-ignored only`), and then first and alone
(`.config/nextest.toml`, [docs/lessons.md](docs/lessons.md)). A GPU test
can still fail beside GPU work or heavy load elsewhere on the host, and
passes run again alone.

| the change touches | check it with |
|---|---|
| the library (`crates/`), the C API or its header | `cargo xtask check` |
| what the Vulkan platform ends the process on, or what device info, GPU families or feature queries answer | the library's checks, and a cold boot of each guest the change can reach before it merges ("A boot is the only check", below) |
| a `.metal` source of the Vulkan crate, or the metal crate's `shaders.metal` | `cargo xtask metallibs` on a Mac, committing the metallibs it writes with the change; then the library's checks |
| the video crates (`crates/reims-vgpu-video-*`), or their arms in `reims-vgpu` | `cargo xtask check` on a Mac; the Linux tests on a VA-API host and on an NVIDIA host; and, where the VA-API crate's decoders change, the conformance suite on a VA-API host ("The video crates' checks", below) |
| `xtask/` | `cargo test -p xtask`, rustfmt and clippy on it |
| `crates/reims-ctl` | `cargo test`, `cargo fmt --check` and `cargo clippy --all-targets` in it (`cargo xtask check` does not reach it); a change to the QEMU command it builds, or to `scripts/install-ctl.sh`, also needs an install and a VM boot on each host it touches |
| `guest/` (kext, bundle, rgfxgate, loader) | `make -C guest test` on a Mac; a change to a source the set is built from also needs `make -C guest prebuilt KDK=<KDK Extensions dir>`, committing `guest/prebuilt` with it (`make -C guest prebuilt-check` fails otherwise) |
| `firmware/` | build it (`make -C firmware`); a change not meant to alter what the guest's firmware sees keeps the ROM's bytes; a change to what `ReimsGraphics.rom` is built from (its sources, the Makefile, `crates/reims-graphics/include/reims_graphics_wire.h`) also runs `make -C firmware prebuilt` on a Mac, committing the ROM and record it writes |
| `vendor/qemu` | build it (`make qemu`) and realize the device paused (`-S`); a change to `hw/display/reims-graphics-pci.c`, or to what QEMU calls, also needs a VM boot ([docs/lessons.md](docs/lessons.md), [docs/qemu.md](docs/qemu.md)) |
| docs and comments | nothing |

`cargo xtask check` builds and runs neither QEMU nor the option ROM, and
boots no guest. When a rule here seems to ask for a check the change cannot
affect, say why it is skipped.

The video crates' checks:
- **The Linux tests**, on a host with libva (radeonsi) and on one with an
  NVIDIA GPU, from a copy of the tree, holding the host's GPU locks
  (`scripts/internal/gpu-locks.sh`) and with no core dumps, since the death
  tests crash on purpose and systemd-coredump keeps each crash:
  ```sh
  ulimit -c 0
  scripts/internal/gpu-locks.sh --all -- cargo nextest run --workspace --no-fail-fast \
    --features reims-vgpu-core/testkit,reims-vgpu/testkit,reims-vgpu/vulkan,reims-vgpu-c/vulkan
  scripts/internal/gpu-locks.sh --all -- cargo nextest run -p reims-vgpu-video-vaapi --run-ignored only   # the VA-API host
  scripts/internal/gpu-locks.sh --all -- cargo nextest run -p reims-vgpu-video-nvidia --run-ignored only  # the NVIDIA host
  ```
  The features are `cargo xtask check`'s on Linux. The ignored tests check
  the bindings against libva's and NVIDIA's headers, and the hardware's own
  answers on the GPU.
- **The conformance suite**, where the VA-API crate's decoders change:
  `crates/reims-vgpu-video-vaapi/tools/h264-check.sh NODE` and
  `hevc-check.sh NODE` on each render node
  ([docs/video/backends.md](docs/video/backends.md), "The conformance
  suite"). Each script takes its GPU's locks itself.


Three tiers:

1. **The loop**, while editing: build, then run only the affected tests
   (`cargo nextest run -p <crate> -E 'test(/<pattern>/)'`). The loop never
   runs clippy, rustfmt or rustdoc.
2. **The task**, once: one commit per task, staging only the task's own
   paths (`git add <new files>`, then `git commit -m ... -- <paths>`), never
   `git add -A` or `git commit -a`. A commit runs no full check.
3. **The merge**, once per branch, by whoever merges it: the checks the
   branch's changes reach (the table), once, on its tree before it merges
   into the integration branch. A failure gets a fix commit on the branch
   before it merges. A GPU test that fails under the check's load runs once
   more alone; passing alone, it is not a failure. Nothing runs beyond that
   for its own sake: no sweep, review or confirming run.

- **Wall-clock first, nothing skipped.** Checks run in the background,
  cheapest to fail first (the affected tests, then `cargo xtask check`), and
  stop at the first failure. While a slow run or a boot is in flight, work
  the runs have already shown goes on. Work may be reordered freely; a check
  the change reaches is never skipped, a result never faked and a comparison
  never loosened.
- **A boot is the only check of what guests choose.** The unit tests run
  the library's own shaders, and guest shaders are never committed, so they
  do not see which code a guest's frameworks pick from the device's answers,
  nor a death every boot reaches ([docs/lessons.md](docs/lessons.md)). For
  the table's row: first look up each code the change would end the process
  on in the last good boot's logged degradation counts, which needs no GPU;
  then boot each guest cold and fail on any `reims-vgpu: fatal` line. An
  answer stays as it was until the guests boot with the new one, and each
  exact fix of a death lands with a test of its pattern in a shader of our
  own.
- **A change meant to alter what the guest or host observes** is its own
  commit. It changes the tests that held the old behaviour, and its message
  says what changed for the guest or the host. Never loosen a comparison to
  make a change pass.

## Working in parallel

- **Each agent works in a worktree of its own**, from its first edit, docs
  included, on a branch of its own. The main checkout is the owner's: agents
  do not edit it. Finished work is merged on an integration worktree, and
  the owner's branch is fast-forwarded to it once the merge's checks pass.
  - Make it with `scripts/internal/new-worktree.sh PATH BRANCH [START]`, not
    `git worktree add`: it adds the worktree from the main checkout, whose
    sparse-checkout settings a new worktree takes.
- **Each worktree builds in its own `target/`**, cargo's default, with no
  `CARGO_TARGET_DIR`; the worktree itself lives under `~/.cache`. A target
  shared between trees runs another tree's `xtask`
  ([docs/lessons.md](docs/lessons.md)), and kache seeds a worktree's
  incremental state only from targets inside their worktrees
  ([docs/development.md](docs/development.md)). Caches never go under
  `/tmp`.
  - Build there in the environment of `scripts/internal/dev-env.sh`, sourced
    from the worktree's root before each cargo command
    (`. scripts/internal/dev-env.sh && cargo ...`): incremental, as the
    check builds, and, where `kache` (2.0.0 or later) is installed, rustc
    and build scripts through kache, a compiler cache local to the host,
    which keeps an edited unit's incremental state and seeds a new
    worktree's from another's. A new worktree's first build then compiles
    only what no tree on the host built before (measured: `cargo xtask
    check`'s tests built in a second worktree in 1.5 s on a Ryzen 7 7840HS
    and 9.5 s on the M3 Max, after a first had built them).
  - Start the target directory empty. Never build through kache in a target
    copied from a build without it: every workspace crate misses
    ([docs/lessons.md](docs/lessons.md)).
  - kache is a tool on the PATH, never Cargo's configured wrapper, so a host
    without it builds as before. Its host configuration is in
    [docs/development.md](docs/development.md), "Building through kache".
- **Two full checks at once can exceed RAM** (one test holds 16 GiB of a
  Mac's GPU memory), and their GPU tests compete for the GPU: run one at a
  time on the host.
- **Set work aside with a WIP commit or a patch**, never `git stash`, whose
  list every worktree shares.
- **Never initialise `vendor/qemu` inside a worktree** that shares the main
  checkout's module repository: it rewrites the module's `core.worktree`
  ([docs/lessons.md](docs/lessons.md)). A worktree that builds QEMU clones
  the submodule for itself.
- **Keep agents few**: as many as the critical path needs, each stopped when
  its work is done.
- **Whoever merges resolves the merge.** Merging finished work and
  fast-forwarding the owner's branch are not questions for the owner.
  Uncommitted work is never called merged, or clean to merge, before a
  hunk-level comparison shows it.
- **Clean up after merged work**: remove its worktree (saving any dirty
  diff first) and its target directory. When a VM round ends, remove its
  disk clones and keep its logs. Never remove a worktree or cache someone
  else made.
- **The metal2vulkan pin in `Cargo.toml` moves in batches**: to a newer
  runtime commit once a set of changes has landed, or one the work waits
  on, with one merge's checks for the move and what rides with it, never a
  check per runtime commit.
- **Decide what the rules decide.** A routine trade-off (a residual, an
  interim difference, where a death goes, the order of work) is decided from
  these rules and the docs, and reported with its reason. The owner decides
  large spends of time or money, the project's direction and rules, and
  anything irreversible or outward-facing. A question for the owner never
  blocks the rest of the work.

## Landing work

- When the merge pipeline is idle and any finished lane waits, start a run
  with every finished lane at once; lanes still running form the next batch.
  Add no lane once a batch's checks start. The moment its last check passes,
  merge it into its target, or report it ready where the owner holds that
  branch's merges.
- When a batch fails, take the failing lane out and run the rest as the next
  batch; the lane rejoins once fixed.
- Log each lane's finish and merge times in the checklist file. If a finished
  lane waits more than one pipeline run for its own run to start, fix the
  pipeline before starting more lanes.
- Report progress as merged commit hashes and each finished lane's wait. A
  plan is not progress.
- Never let an agent block on a command that returns only when a queued job
  starts: submit and poll, or time out near the expected wait. Every 15
  minutes, check each agent and remote job, and act on any with no new
  output.
- On a first-in-first-out queue without backfill, submit a whole-machine job
  only when that machine has nothing running or queued; otherwise it holds
  the head of the queue while the machine drains. Submit short jobs before
  long ones.
- Shared test machines serve the merge pipeline first: queue measurements
  only while no pipeline job waits, since queue wait grows steeply as a
  machine nears full use. Run a harness on its shortest case before a long
  run, and stop a series at a harness fault until it is fixed.

## Rules for changing code

- **Behaviour changes are deliberate.** Some behaviours look odd and are
  kept because the guest's frameworks rely on Metal behaving that way; each
  carries a doc comment and a test named for it (docs/architecture.md 17.2).
  Changing one is a behaviour change: make it on purpose, in its own commit,
  with its tests changed and the reason stated.
- **The window's features are a register.** What a display's window does
  for the person at the host is listed in
  `crates/reims-vgpu-window/features.md`, each entry with its guard: a named
  test, which `features_md_guards_exist` keeps, or a manual check on a VM
  boot. Dropping or changing an entry is a deliberate change, made in its
  own commit that updates the file and says why.
- **The Vulkan platform's rules** (guest memory in place, where stalls are
  fixed, what is the runtime's, dependencies as released) are in
  [docs/conventions.md](docs/conventions.md).
- **Never guess a name.** A field of a guest record or of a copy of a Metal
  struct whose meaning nothing states is `field_xx`, `xx` its byte offset in
  lowercase hex. A wrong name is worse than an unknown one: it attaches the
  wrong meaning to the right bytes, and nothing flags it.
- **Say what the code does.** Comments and docs describe the code and what
  the guest or host observes. They do not cite the earlier ports, which
  stay at their tags (`byte-exact-parity`, `rust-port-parity`,
  `native-parity`).
- **arm64 on the Mac.** Every Rust binary built for macOS is arm64
  (`aarch64-apple-darwin`). rustc's arm64e target has no pointer
  authentication, and its binaries crash before `main`
  (`rust-toolchain.toml`). On Linux the library ships as
  `x86_64-unknown-linux-gnu`, built there.
  `cargo xtask check` on a Mac also clippy-checks that target, and, where
  Rosetta is installed, runs core's tests as `x86_64-apple-darwin`, for
  the x86-64 guest's 4 KiB pages and instructions; it skips them, saying
  so, where Rosetta is not. No x86-64 macOS binary ships.
- **Run `cargo xtask check` once before a branch that changes the library
  merges**, not before each commit (Checking a change).
- **Stage explicit paths, and never commit build output.** Before each
  commit, `git diff --cached --name-only | grep -E
  '(^|/)target[^/]*/|\.(bin|o|obj|rlib|rmeta|dylib|a|efi|rom)$' | grep -vxF
  firmware/prebuilt/ReimsGraphics.rom` must print nothing.
  - **The exceptions are the outputs below**, each committed with the
    sources it is built from.
  - **Metallibs** compiled from this repository's own `.metal` sources,
    which only Apple's compiler makes: the Vulkan platform's, in
    `crates/reims-vgpu-vulkan/metallibs`, which a build off a Mac copies, and
    the tests', in `tests/shaders`. The Vulkan
    platform's are rewritten by `cargo xtask metallibs` on a Mac, and a
    stale one fails the Mac's `cargo xtask check` and the Linux build.
  - **The reims-graphics device's option ROM**,
    `firmware/prebuilt/ReimsGraphics.rom`, which reims-ctl embeds and names
    as the device's `romfile`. `make -C firmware prebuilt` rewrites it on a
    Mac with its record, `firmware/prebuilt/ReimsGraphics.toml`, and a stale
    one fails `cargo xtask check` and reims-ctl's `cargo test`
    (`make -C firmware prebuilt-check`).
  - **The guest's OpenCore kext set for that device**, in `guest/prebuilt`:
    `set.toml`, `sources.toml`, and for each of `ReimsGraphicsLoader`,
    `RgfxIOGraphicsFamilyBridge`, `RgfxAppleGraphicsDeviceControlBridge`,
    `ReimsGraphics` and `RgfxGate`, `<name>.kext/Contents/Info.plist` and
    `<name>.kext/Contents/MacOS/<name>`. reims-ctl embeds the bundles
    `set.toml` lists and adds them to the OpenCore image of an x86-64 VM on
    reims-vgpu. `make -C guest prebuilt` rewrites them from `guest/` on a Mac
    with a KDK, with the record of their sources, `sources.toml`, and a stale
    set fails `make -C guest prebuilt-check`, which `cargo xtask check` runs.
- **Never commit Apple binaries.** The build needs none: the option ROM is
  built from `firmware/`, and the kexts from `guest/`. The committed
  metallibs hold only our shaders and constant samplers, with no Apple code
  linked in. Nothing taken from a guest or from Apple's binaries is
  committed or pushed either: a guest's shader caches and metallibs, what is
  translated or dumped from them, logs that carry its serialized
  descriptors, excerpts of Apple's code. A report to an upstream project
  carries inputs written from scratch.

## Fixing bugs

- **Fix what runs show.** Boot guests and use their apps, and fix the
  deaths, glitches, stalls and slow frames they show, biggest first, end to
  end. A gap no run reaches waits, unless it is cheap.
- **Fix a case by its class.** A boot shows only its first death, and users
  would meet the rest: when a run finds one unhandled case, find every case
  of its kind (every pixel format, every death site, every refusal of the
  runtime) and fix them together.
- **Reproduce under the conditions the bug was seen in**, not under a
  theory's trigger. Size a proof window from the measured failure rate, and
  ship a cheap, low-risk fix as soon as the evidence points to it, with its
  confirmation run behind it.
- **Check a claim before relaying it.** A claim about Metal is tagged
  measured or read from code ([docs/testing.md](docs/testing.md), "Tags").
  A claim that the hardware cannot do something is checked against the
  driver (its format features and extensions) first.

## GPU and VM discipline

Some tests fault the GPU and crash processes on purpose, and the GPU
watchdog corrupts concurrent work.

- **A VM boot never overlaps GPU work.** No test boots a VM. Before a boot,
  whoever may run GPU work holds it ([docs/qemu.md](docs/qemu.md), "Booting
  a guest to check a change", which also says how a boot checks its screen
  and which crashes are out of scope).
- **A boot writes only a copy** of a guest's disk (a clone, or a
  `snapshot=on` overlay), and a step that changes the guest's state runs
  only there. A damaged disk is copied again from its source.
- **Measure what users get.** Boots and timings run on hosts set up as
  users' are: no option that dodges a host policy users have (a renamed
  process, a raised priority, a stopped daemon). Undo only what the harness
  itself adds. Contention under such a policy is the code's to fix.
- **Run a GPU test that failed under load again alone** before counting a
  failure.
- **A GPU shared with other projects' runs is locked.** A run on it holds
  the locks `scripts/internal/gpu-locks.sh` names for its GPUs for as long
  as it runs: `scripts/internal/gpu-locks.sh [--node PATH] -- CMD` runs CMD
  holding them, under metal2vulkan's `maclock` on a Mac.
- **Stop only the processes you started**, by their pids.

## Conventions

- Branches are prefixed `steelbrain/`.
- Commit messages use an emoji prefix (`:new:`, `:bug:`, `:art:`,
  `:racehorse:`, `:fire:`, `:memo:`, `:white_check_mark:`), imperative mood,
  a subject of at most 72 characters, and a body wrapped at 80 that explains
  why. Docs-only commits carry `[ci skip]`.
- Never `git push` unless asked.
- Merged commits are not rewritten: a fix to merged work, a review's
  included, is a new commit on top. Amend only a commit the owner names; a
  flaw only in a merged commit's message stays.
- Scratch goes under `/tmp`, and is removed when the work is done. Caches
  (a target directory, a build kept for reuse) go under `~/.cache/<name>`,
  and so does what must outlive a reboot, which empties `/tmp`: a worktree,
  the state a long run resumes from.
- Instruments' recordings are scratch. `xctrace record` writes its raw trace
  to `$TMPDIR/instruments*.ktrace`, gigabytes a run, and can leave it behind:
  189 left there by two days of profiling held 254 GiB. A profiling run
  removes its `.trace` and the `.ktrace` files it made when it is done.
- The code's conventions are [docs/conventions.md](docs/conventions.md).

## Recording what was learned

- A trap about what decides behaviour, found by a test, a review or a boot,
  goes in [docs/lessons.md](docs/lessons.md): what went wrong, why, and how
  to spot the next one.
- A measured platform fact goes in the `//!` doc of the module that relies on
  it.
- QEMU and VM measurements go in [docs/qemu.md](docs/qemu.md).
