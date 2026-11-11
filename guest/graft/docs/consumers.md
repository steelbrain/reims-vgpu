# Building a kext for Graft

Graft serves two kinds of kexts, and a kext is built differently for each. Any
kext placed in the boot image beside Graft can be either.

- An **owned client** calls Graft's interface to look up kernel symbols and
  route kernel functions.
- A **bound consumer** derives from classes, or calls functions, in libraries
  of the System kernel collection, and Graft binds it at boot.

This page says what sets the two apart and how each goes into a boot image.
The guides take you through the steps, and [design.md](design.md) explains
what Graft does with each kext at boot.

Graft finds each kind without being told its name:

- **A bound consumer** by its manifest. At start Graft looks through every
  kext placed at boot, the Boot collection's fileset entries on x86-64 and the
  boot firmware's device-tree nodes on arm64e, and keeps each one whose
  `__DATA,__graft` section starts with the manifest's magic. Its personalities
  match their providers as any driver's do, and once it is bound, Graft opens
  its gate on every nub they match.
- **An owned client** by the bundle identifier each call passes. Graft serves
  the call when that identifier names a kext placed at boot, so a client
  passes its own.

So serving a new kext of either kind needs no change to Graft.

Everything a kext builds with is in [sdk/](../sdk/README.md). Another
repository takes a release, the SDK included, with `tools/vendor.py`
([guides/taking-a-release.md](guides/taking-a-release.md)).

## An owned client

An owned client is an ordinary kext with four additions:

1. Graft's header, `<graft/Graft.h>`, from the SDK's `include/`.
2. The SDK's start glue, `sdk/Start.cpp`, which declares the kmod and calls
   the client's `graft_client_initialize()` from its start routine.
3. A dependency on Graft: `ai.aneesiqbal.graft` at version `0.1.0` in
   `OSBundleLibraries`. Release 0.2.0 declares `OSBundleCompatibleVersion`
   `0.1.0`, so a client that lists `0.1.0` loads with it.
4. An `IOService` subclass and one personality that matches `IOResources`,
   because an injected kext is not started for `OSBundleRequired` Root alone.

A route request has a different form on each architecture:

- On x86-64, it names a function in the kernel's own symbol table, and the
  replacement and original have that function's prototype. Routes must be
  installed before BSD starts, so install them from `graft_client_initialize()`.
- On arm64e, it names one of three MAC policy operations, and the replacement
  and original have the operation's `mpo_*_t` prototype from XNU's
  `security/mac_policy.h`. The routes go live with one atomic swap of AMFI's
  policy, so they are safe to install after BSD has started.

On both, the client keeps the `original` in a `void *` and converts it to the
prototype where it calls it, which arm64e's pointer authentication needs.
[design.md](design.md#routing) gives the rules and limits, and
[Graft.h](../sdk/include/graft/Graft.h) documents each call.
[guides/owned-client.md](guides/owned-client.md) builds one, with a replacement
for each architecture.

## A bound consumer

A bound consumer is written like any kext against the Kernel.framework headers
in the macOS SDK. Its build differs at the final link, which `sdk/consumer.mk`
replaces with the SDK's `graft-kext` command. That command measures the
classes the consumer derives from, links it, writes the manifest that tells
Graft what to bind into `__DATA,__graft`, and leaves a dormant bundle: its
personalities are gated by an `IOPropertyMatch`, and its System-collection
libraries are removed from `OSBundleLibraries`. A bound consumer calls nothing
in Graft, so it needs no dependency on it.

The consumer declares what Graft binds in `src/graft.json`: the libraries it
imports from, the classes it derives from, and its own classes.
[sdk/README.md](../sdk/README.md#graftjson) documents every key.

- [guides/bound-consumer.md](guides/bound-consumer.md) builds one, from
  `examples/hello-consumer/`.
- [guides/synthesized-class.md](guides/synthesized-class.md) covers an arm64e
  consumer whose parent class has no Apple header.

## Putting it in the image

On x86-64, OpenCore injects Graft and the kexts it serves through
`Kernel > Add`, each with its bundle's executable and Info.plist path. List
`Graft.kext` first, then the kexts it serves, since OpenCore needs a kext's
dependencies above it.

On arm64e, the boot firmware places `Graft-arm64e.kext` and the kexts it serves
in the Auxiliary kernel collection before the kernel starts, binds their
undefined symbols to kernel exports, and publishes in the device tree the
addresses Graft reads: the Boot collection's, each placed kext's, and the
vtable discriminator tables a synthesized class needs.
[src/README.md](../src/README.md#what-the-arm64-boot-firmware-provides) gives
the format.

On both, SIP stays on (`csr-active-config` 0), and Graft needs no boot-args.

## Seeing what happened

Graft publishes its state on the IORegistry root. In the guest,

```sh
ioreg -l -d 1 | grep Graft
```

shows `GraftState`, `GraftReason`, and a `GraftClient.<bundle id>` entry for
each kext Graft serves, as `<state>: <reason>`. It shows only the start of
`GraftLog`, Graft's recent steps, one a line. This prints all of it:

```sh
ioreg -l -d 1 | sed -n '/"GraftLog" = /,/^"$/p'
```

On arm64e the steps are also in the unified log, prefixed `graft: `; on x86-64
they are not. [guides/debugging-refusals.md](guides/debugging-refusals.md)
explains each state and reason, and how to read the log.
