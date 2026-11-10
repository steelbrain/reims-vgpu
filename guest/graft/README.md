# Graft

Graft is a macOS kernel extension that binds and routes for a fixed set of
kexts inside macOS virtual machines, with System Integrity Protection on. It
builds for x86-64 and arm64e guests, and does two jobs:

- It binds a **bound consumer**: a kext injected dormant, which derives from
  classes or calls functions in Apple's System kernel collection. Its build
  embeds a manifest in `__DATA,__graft`. At boot Graft checks each parent class
  against the running system, resolves each import by symbol name, writes
  everything in one transaction, and lets the kernel start the kext.
- It routes kernel functions for an **owned client**, a kext that calls
  Graft's C interface, [sdk/include/graft/Graft.h](sdk/include/graft/Graft.h),
  and keeps its own policy in its replacements.

Graft checks no macOS version. It decides from what the running system has:
symbols, class sizes, vtable slots and IOKit's answers. When something a kext
needs is missing, Graft refuses that kext by name, publishes the reason in the
IORegistry, and the guest keeps booting.

## Who Graft serves

Graft serves two kexts, named in [src/Policy.h](src/Policy.h): the bound
consumer `ai.aneesiqbal.ReimsGraphics` and the owned client
`ai.aneesiqbal.RgfxGate`. The SDK also requires the consumer's personalities
to match the PCI device `1234:5247`. These are the guest drivers of
reims-vgpu, a paravirtual GPU for macOS virtual machines: ReimsGraphics is its
display and Metal driver, and RgfxGate installs its Metal bundle. Graft
replaces Lilu for them.

Graft has no plugin interface. Serving another kext means changing
`src/Policy.h` and rebuilding Graft, and serving another device also means
changing the device match in the SDK's build tool. The
[examples](examples/README.md) carry the two pinned identities for that
reason.

## Status

This is release 0.1.0. Every boot below ran with SIP on (`csr-active-config`
0) and no boot-arg for Graft or the kexts it serves.

**x86-64, on macOS 11.7.11 and 26.7.2.**

- Graft alone, with the test kexts `tests/kexts/core-probe` and
  `examples/hello-consumer` and no boot-args at all, booted to the desktop.
  Graft bound the hello consumer and opened its gate, and routed six kernel
  functions, each of which the system then called.
- With reims-vgpu's guest kexts on its paravirtual GPU, Graft bound the display
  driver and routed for its installer, the kexts installed themselves on that
  boot, and WindowServer used Metal on the device.

**arm64e, on macOS 14.6.1, 15.6.1, 26.6.2 and 27.0.1**, under Ambrosia 0.1.1,
the arm64 boot firmware Graft was developed with. With reims-vgpu's guest
kexts, Graft bound the display driver, building its vtable at bind in the
running release's layout, and routed three AMFI operations for its installer.
The installer wrote its bundle, and a binary that AMFI must refuse was still
refused. These boots checked Graft's part; the display reaching a desktop
composited with Metal is reims-vgpu's open work on arm64.

Development builds before 0.1.0 also passed the x86-64 checks with
reims-vgpu's kexts on macOS 12.7.6, 13.7.8, 14.8.9, 15.7.9, 15.8.2 and 26.6.1,
and bound the display driver on arm64e macOS 13.6.

## What you need

- **In the guest.** On x86-64, OpenCore injects Graft and the kexts it serves.
  On arm64e, the boot firmware places them and hands Graft what
  [src/README.md](src/README.md#what-the-arm64-boot-firmware-provides) lists.
  SIP stays on (`csr-active-config` 0), and Graft needs no boot-args.
- **To build a kext for Graft.** A Mac with Xcode or its Command Line Tools.
  The SDK needs nothing else and fetches nothing.
- **To build and test Graft.** A Mac with Apple silicon and Xcode; see
  [CONTRIBUTING.md](CONTRIBUTING.md).

## Quick start

```sh
make                # build/Graft.kext, for x86-64
make graft-arm64e   # build/Graft-arm64e.kext, for arm64e
make test           # the host tests
make examples       # the two examples, for both architectures
```

To use a release instead of building one, download `graft-<version>.zip`
from the release on GitHub and copy it into your tree with `tools/vendor.py`
([docs/guides/taking-a-release.md](docs/guides/taking-a-release.md)). CI
builds each release from its tag, with the Xcode that `.xcode-version` pins.

## How it works

The kernel starts Graft from its own kmod start routine. Graft finds the
running kernel's image, copies and calibrates its symbol table, and prepares
its kernel writes and routes. On x86-64 it then routes the two kernel
functions that hand kext personalities to IOKit's catalogue, so that every
catalogue send gives it a chance to bind. On arm64e the firmware has placed
every image already, so Graft binds at once.

To bind, Graft reads the consumer's manifest and finds the libraries it imports
from, loading them if they are not loaded yet. It confirms each library's image,
measures each parent class, plans every import site and inherited vtable slot,
and checks every target. Then it writes them all in one transaction, and sets
the property that the consumer's personalities wait on, so the kernel constructs
and starts it. If any check fails, Graft writes nothing and publishes the
reason.

On arm64e, kernel text is read-only when Graft runs. Graft routes the owned
client's functions by shadowing AMFI's MAC policy table. For a parent class
that has no header in Apple's SDK, Graft builds the consumer's vtable at bind
time, in the running system's order.

[docs/design.md](docs/design.md) describes all of this in full.

## Documentation

- [docs/](docs/README.md): how Graft works, the manifest format, the how-to
  guides and the lessons learned building it.
- [CONTRIBUTING.md](CONTRIBUTING.md): building, testing, the rules for changes,
  and releases.
- [CHANGELOG.md](CHANGELOG.md): what each release changed.

## Repository layout

| directory | contents |
|---|---|
| [src/](src/README.md) | the kext |
| [sdk/](sdk/README.md) | what a kext builds with to work with Graft: the header, the start glue, and the build rules and tool for a bound consumer |
| [examples/](examples/README.md) | a minimal bound consumer and a minimal owned client |
| [docs/](docs/README.md) | design, manifest format, guides and lessons learned |
| [tests/](tests/README.md) | the host tests, and the test-only kexts that only a guest can check |
| [tools/](tools/README.md) | `vendor.py`, which copies a release into another repository, and the maintainers' tools |

## License

Graft is free software. You may use it under the GNU General Public License,
version 2 or later, or under the GNU Lesser General Public License, version 3
or later, at your option. See [LICENSE](LICENSE).
