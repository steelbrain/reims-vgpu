# Write a bound consumer

A bound consumer is a kext that derives from classes, or calls functions, in
Apple's System kernel collection. The boot loader cannot link an injected kext
against that collection, so the consumer is injected dormant, and Graft binds
it by symbol name at boot. This guide builds one, using
[examples/hello-consumer](../../examples/hello-consumer/README.md) as the
running example.

Any kext can be a bound consumer. Graft binds every kext placed at boot that
carries the manifest its build writes, whatever its bundle identifier, and
opens its gate on whatever provider its personalities match. The example's
identifier is its own, `ai.aneesiqbal.graft.HelloConsumer`.

## What you need

- A Mac with Xcode or its Command Line Tools. The build needs nothing else and
  fetches nothing.
- Graft's SDK: the `sdk/` directory of a Graft checkout, or a copy of a
  release taken with `tools/vendor.py` ([taking-a-release.md](taking-a-release.md)).

## 1. Write the class

Write the kext as you would any other, against the Kernel.framework headers
in the macOS SDK. The example's `src/Hello.h` declares `GraftHelloConsumer`,
derived from `IOAccelerator`, which IOGraphicsFamily defines in the System
collection.

## 2. Declare what Graft binds

`src/graft.json` names the System-collection libraries the kext imports from,
the classes it derives from, and its own classes. The example's:

```json
{
  "libraries": [{"id": "com.apple.iokit.IOGraphicsFamily"}],
  "classes": [{
    "name": "IOAccelerator",
    "header": "IOKit/graphics/IOAccelerator.h",
    "library": 0,
    "required": []
  }],
  "consumers": [{
    "name": "GraftHelloConsumer",
    "header": "src/Hello.h",
    "parent": 0
  }]
}
```

- `libraries` lists each provider by bundle identifier. A provider you only
  import from, without deriving from any of its classes, also needs an
  `anchor`: a class it defines, which Graft uses to confirm it found the
  provider's image.
- `classes` lists the System-collection classes you derive from, with the
  header that declares each. `required` lists, by mangled name, the inherited
  methods whose slots your code calls. Graft checks that each one is in the
  live class's vtable where your build expects it. Leave `required` out to
  check every slot except the headers' `_RESERVED` placeholders. An empty
  list checks only the slots your class overrides.
- `consumers` lists your own classes, each with the index of its parent in
  `classes`.
- `imports`, optional, gives an import candidate names to try in order, or a
  fallback in your own code, for a symbol only some releases define.

[sdk/README.md](../../sdk/README.md#graftjson) documents every key. An import
you declare `__attribute__((weak_import))` in your source is weak: when the
running release does not define it, Graft binds it so that
`OSKextSymbolIsResolved()` returns false, and the bind still succeeds.

## 3. Write the Info.plist

Write `src/Info.plist` as an ordinary kext's:

- `CFBundleIdentifier` is the kext's own, under which Graft reports its bind;
- each personality names its provider's class in `IOProviderClass` and
  matches the provider as any personality does. A driver matches its own
  device, such as an `IOPCIDevice` by `IOPCIPrimaryMatch`. The example matches
  `IOResources`, with an `IOMatchCategory` of its own, so that it runs in any
  guest;
- `OSBundleLibraries` lists the System-collection libraries from
  `graft.json`, beside your other dependencies;
- `CFBundleVersion` may be `@VERSION@`, which the build replaces.

The build checks that each personality names an `IOProviderClass`, records
each personality's provider matching in the manifest, removes the
System-collection libraries from `OSBundleLibraries` (and from any
per-architecture list, such as `OSBundleLibraries_x86_64`), and gates every
personality with a passive `IOPropertyMatch`, added to the one a personality
already has. The kernel then registers the kext but does not construct it
until Graft opens the gate on a nub the personality's provider matching
matches. A bound
consumer calls nothing in Graft, so it needs no dependency on it. The build
refuses a personality whose `IOPropertyMatch` already holds the gate's key, as
a staged Info.plist's do, since Graft would then watch only for nubs whose gate
is already open. Give it the Info.plist you wrote.

## 4. Link through consumer.mk

Your Makefile compiles the objects with its own rules, defines the variables
`consumer.mk` reads, and includes it after its default target. A Makefile in
the shape of the example's, for a project that copied Graft's release to
`graft/`:

```make
NAME = GraftHelloConsumer
VERSION = 1.0.0
SDK = $(shell xcrun --sdk macosx --show-sdk-path)
KERNEL = $(SDK)/System/Library/Frameworks/Kernel.framework/Headers
ARCH ?= -arch x86_64
BUILD = build/obj
KEXT = build/GraftHelloConsumer.kext
BINARY = $(KEXT)/Contents/MacOS/$(NAME)
OBJECTS = $(BUILD)/Hello.o
HEADERS = src/Hello.h
CXXFLAGS = $(ARCH) -isysroot $(SDK) -mmacosx-version-min=11.0 -mkernel -nostdinc \
    -isystem $(KERNEL) -DKERNEL -std=c++17 -fapple-kext -fno-rtti -fno-exceptions \
    -Os -Wall -Werror -I.

all: $(BINARY)

$(BUILD)/Hello.o: src/Hello.cpp $(HEADERS) Makefile ; mkdir -p $(@D) && xcrun clang++ $(CXXFLAGS) -c $< -o $@

include graft/sdk/consumer.mk
```

The compile rule uses make's one-line form, `target: prerequisites ; recipe`,
which means the same as a recipe on its own tab-indented line.

`consumer.mk` replaces your final link with one command, `graft-kext build`.
It measures each declared class with your compiler flags, links the kext,
writes the manifest into its `__DATA,__graft` section, and stages the dormant
bundle at `$(KEXT)`. For arm64e, set `ARCH` to `-arch arm64e`; the link then
also takes `-no_data_const`, which keeps the import slots and vtables in
writable memory, where Graft writes them.
[sdk/README.md](../../sdk/README.md#consumermk) lists every variable.

## 5. Build

Run `make`, or `make ARCH='-arch arm64e'`. The Makefile above writes both
architectures to the same `BUILD` and `KEXT`, and its objects do not depend on
`ARCH`, so remove `build/` between them, or give each architecture its own
`BUILD` and `KEXT` as the example's Makefile does. In this repository, the
example builds with:

```sh
make -C examples/hello-consumer                      # x86-64
make -C examples/hello-consumer ARCH='-arch arm64e'  # arm64e
```

Its output is `build/GraftHelloConsumer.kext`, or
`build/GraftHelloConsumer-arm64e.kext`. A build that cannot produce a
consumer Graft could bind stops with `consumer:` and the reason, such as a
provider with no class or anchor, or a personality that names no
`IOProviderClass`.

## 6. Put it in the boot image

On x86-64, add Graft and the consumer to OpenCore's `Kernel > Add`, Graft
first. Keep SIP on (`csr-active-config` 0). Graft needs no boot-args.

On arm64e, the boot firmware places Graft and the consumer in the Auxiliary
kernel collection and publishes the consumer's address in the device tree
([src/README.md](../../src/README.md#what-the-arm64-boot-firmware-provides)).
Keep SIP on there too.

## 7. Check that it bound

In the guest:

```sh
ioreg -l -d 1 | grep Graft
```

A bound consumer whose provider is present reads:

```
"GraftClient.ai.aneesiqbal.graft.HelloConsumer" = "bound: imports and vtables committed; activation gate open on a provider of personality GraftHelloConsumer"
```

The kernel then constructs and starts it. Until a nub that one of the
consumer's personalities matches is published, the reason ends
`waiting for a provider of personality GraftHelloConsumer to open the gate`
instead. The example also publishes its own `GraftHelloConsumerState`,
`passed` when both of its bound calls into IOGraphicsFamily returned success.

The consumer's state may also read:

- `waiting`, with the reason: on x86-64, a provider is not registered or not
  started yet. Graft tries again at each catalogue send. This is normal early
  in boot.
- `refused`, with the reason: Graft will not bind this consumer on this
  boot. [debugging-refusals.md](debugging-refusals.md) explains each reason.

A consumer with no `GraftClient` entry at all was not found at start: Graft's
log names each consumer it found
([debugging-refusals.md](debugging-refusals.md#a-bound-consumer)).
