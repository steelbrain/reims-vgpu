# Graft

Graft is a macOS kernel extension that binds and routes for other kexts
inside macOS virtual machines, with System Integrity Protection on. Any kext
can use it. It builds for x86-64 and arm64e guests, and does two jobs:

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

Graft serves any kext placed in the boot image beside it, as a bound consumer
or as an owned client. It names none of the kexts it serves, and serving a new
one needs no change to Graft.

- **A bound consumer** builds with the SDK's `consumer.mk`
  ([guide](docs/guides/bound-consumer.md)). At start, Graft looks through every
  kext placed at boot: the Boot collection's fileset entries on x86-64, and the
  boot firmware's device-tree nodes on arm64e. It binds each one whose
  `__DATA,__graft` section starts with the manifest's magic, any number of
  them. The consumer's personalities match their providers as any driver's do,
  and Graft opens the consumer's gate on every nub they match.
- **An owned client** calls the functions in
  [Graft.h](sdk/include/graft/Graft.h), passing its own bundle identifier
  ([guide](docs/guides/owned-client.md)). Graft serves a call when that
  identifier names a kext placed at boot, and several clients can route in one
  boot.

The [examples](examples/README.md) are one of each, under identifiers of their
own.

## Status

This is release 0.2.0. It passes the host tests and the build checks
([CONTRIBUTING.md](CONTRIBUTING.md#checks)), and the boots below ran its
kexts, with SIP on (`csr-active-config` 0) and no boot-arg for Graft or the
kexts it serves.

Several boots used the guest kexts of reims-vgpu, a paravirtual GPU for macOS
virtual machines: its display and Metal driver is a bound consumer, and the
kext that installs its Metal bundle is an owned client.

**x86-64, on macOS 11.7.11 and 26.7.2.**

- Graft alone, with this repository's two consumers and two clients
  (`examples/hello-consumer`, `tests/kexts/gate-probe`,
  `tests/kexts/core-probe` and `examples/hello-client`) and no boot-args at
  all, booted to the desktop. Graft bound both consumers, and opened each
  one's gate on the provider its personalities match: `IOResources`, and the
  PCI host bridge by its class. It routed six kernel functions for one client,
  each of which the system then called, and resolved a kernel symbol for the
  other.
- On 26.7.2, with gate-probe built to declare a class the kernel does not
  register, Graft refused that consumer, naming the class, and it never
  loaded. The other consumer was still bound and the client still routed.
- With reims-vgpu's guest kexts on its paravirtual GPU, Graft bound the display
  driver and routed six kernel functions for its installer. The installer
  wrote its Metal bundle, WindowServer mapped it, and Metal ran on the
  device.

**arm64e, on macOS 14.6.1, 15.6.1, 26.6.2 and 27.0.1**, under Ambrosia 0.1.1,
the arm64 boot firmware Graft was developed with.

- Graft alone, with the same two consumers, `examples/hello-client`, and
  `tests/kexts/route-probe`, a client that routes three AMFI operations: Graft
  bound both consumers and opened their gates, and routed the three
  operations, each of which the system then called. A copy of route-probe
  under an identifier of its own, asking for the same operations later in the
  boot, was refused, naming route-probe as the client that holds them.
- With reims-vgpu's guest kexts, Graft bound the display driver, building its
  vtable at bind in the running release's layout, and routed three AMFI
  operations for its installer. The installer wrote its bundle, WindowServer
  mapped it and composited with Metal on the device, and a binary that AMFI
  must refuse was still refused. The same checks passed on macOS 27.2.1 under
  a development build of Ambrosia, since 0.1.1 does not boot 27.2.1.

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
