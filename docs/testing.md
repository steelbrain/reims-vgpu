# Testing

How the code is tested, and how a claim about Metal or the hardware says where it comes from.
[AGENTS.md](../AGENTS.md), "Checking a change", says which of these checks a change runs.

## The tests

- **Unit tests** sit beside the code, in `mod tests`, named by the behaviour they hold.
  `cargo xtask check` runs every one with cargo-nextest, and the doc-tests.
- **Core's testkit** (`core::testkit`, the Vulkan crate's `testkit` feature) is a fake `Platform`
  that models Metal's objects and logs the calls whose order Metal observes
  ([architecture.md](architecture.md) 13.2). Core's tests, and the tests of other crates that need
  no GPU, run on it.
- **The Vulkan platform's tests** run on an rt device of the host's GPU, strict
  ([architecture.md](architecture.md) 13.3). Metal's answer is the reference: where this
  platform answers otherwise on purpose, the test says so and
  [accuracy-ledger.md](accuracy-ledger.md) has the row. No test reads answers recorded from
  Metal. The tests that need an rt device and are ignored on macOS say why in their ignore
  reason.
- **The C API's tests** (`crates/reims-vgpu-c/tests`): its exports, and a host in C built against
  the committed header and linked with the shared library.
- **The wire's header** (`crates/reims-vgpu-graphics/tests/header.rs`) compiles as the guest's kext and
  option ROM compile it, as C11 and as C++17 with no warning, its records laid out as `wire`'s.
- **The host video backends** have hardware tests and conformance runs of their own
  ([video/](video/)).
- **The guest's kexts and Metal bundle** have their tests (`make -C guest test`, on a Mac).
  `cargo xtask check` also runs the records of the prebuilt kext sets, x86-64 and arm64e (`make -C guest
  prebuilt-check`) and of the option ROM (`make -C firmware prebuilt-check`), which fail when the
  committed binaries are stale.
- **VM boots** are the only check of what a guest's frameworks choose from the device's answers,
  of a death every boot reaches, and of QEMU's side of the C API
  ([qemu.md](qemu.md), "Booting a guest to check a change").

A test of an odd behaviour, one that guests may rely on, names the behaviour. Changing what it
holds is a change of behaviour, made on purpose in a commit of its own that says why.

## Tests that disturb the host

Every process on a Mac shares its GPU, and a test that faults the GPU fails the work of every
other process there, the desktop's included ([lessons.md](lessons.md)). The tree holds no test
that faults the GPU or reserves most of its memory on purpose; one added would be opt-in
(`#[ignore]`), run only for a change to what it tests, with `--run-ignored only`, first and alone
(`.config/nextest.toml`). A GPU test can still fail beside GPU work or heavy load elsewhere on the
host; one that passes run again alone is not a failure.

## Tags

A claim about Metal, a driver or the hardware is tagged:
- **measured**: run, and the text says how, on which host and OS;
- **read**: from the code, ours or a driver's;
- **cited**: from an external source, named where it is used;
- **unknown**: not established.

An honest unknown is worth more than a plausible guess, for the reason AGENTS.md gives for names:
a wrong claim attaches the wrong meaning to the right code, and nothing flags it. A claim that the
hardware cannot do something is checked against the driver (its format features and extensions)
before it is relayed.
