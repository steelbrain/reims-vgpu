# hello-consumer

The smallest bound consumer. A bound consumer links against a class in Apple's
System kernel collection, which the Boot collection does not hold, so OpenCore's
linker cannot resolve it at inject time. Graft resolves it by symbol name at
boot. [docs/design.md](../../docs/design.md) explains what Graft does;
[docs/consumers.md](../../docs/consumers.md) and [sdk/README.md](../../sdk/README.md)
explain how a consumer is built.

## What it shows

`src/Hello.h` declares one class, `GraftHelloConsumer`, derived from
`IOAccelerator`. `src/graft.json` declares the provider library
(`com.apple.iokit.IOGraphicsFamily`), the System-collection class the consumer
derives from (`IOAccelerator`), and the consumer's own class. The build through
`sdk/consumer.mk` records those in a `__DATA,__graft` manifest and stages a
dormant bundle: its personality is gated by an `IOPropertyMatch` and the
System-collection library is dropped from `OSBundleLibraries`.

At boot Graft measures `IOAccelerator`'s live size and vtable, fills the
consumer's inherited vtable slots, writes each of its import sites into
IOGraphicsFamily, and opens the gate so the kernel starts the kext. Besides the
sites the compiler emits for the class's constructors, destructor and metaclass,
`src/Hello.cpp` holds one site of each form, a data pointer and a direct call,
and its start routine calls through both and records the result.

Graft binds exactly one consumer, the one named in `src/Policy.h`
(`ai.aneesiqbal.ReimsGraphics`). This example carries that bundle identifier so
the shipped Graft binds it; its class and executable are named Hello, but the
identifier must stay `ai.aneesiqbal.ReimsGraphics` to be bound.

## Build

The build needs a Mac with Xcode or its Command Line Tools, and fetches
nothing. Graft's SDK is two directories up, at `../../sdk`.

```sh
make                      # ../../build/GraftHelloConsumer.kext (x86-64)
make ARCH='-arch arm64e'  # ../../build/GraftHelloConsumer-arm64e.kext (arm64e)
```

`ARCH` picks the target; x86-64 is the default. Both write into the
repository's `build/`, two directories up. The arm64e variant is suffixed,
so both can be built without overwriting each other.

## Install and expect

On x86-64, add `GraftHelloConsumer.kext` to the OpenCore image after
`Graft.kext`: this example lists Graft in its `OSBundleLibraries`, and OpenCore
needs a dependency listed first. On arm64e, the boot firmware places
`GraftHelloConsumer-arm64e.kext` beside `Graft-arm64e.kext`. Boot with SIP on
and no boot-args. In the guest:

```sh
ioreg -l -d 1 | grep Graft
```

Once Graft has bound the consumer you see `GraftState = active`,
`GraftClient.ai.aneesiqbal.ReimsGraphics` beginning `bound:`, and this example's
own `GraftHelloConsumerState = passed`. If a provider is not yet registered,
the consumer's line reads `waiting`. If a layout check fails, it reads
`refused` with the reason
([debugging-refusals.md](../../docs/guides/debugging-refusals.md)).
