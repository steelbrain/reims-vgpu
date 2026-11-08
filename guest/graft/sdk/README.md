# sdk

The files our kexts build with to work with Graft. An owned client compiles
against the header and the start glue; a bound consumer replaces its final
link with `consumer.mk`. [docs/consumers.md](../docs/consumers.md) explains
the two kinds and how to build each; this file is the reference for the
files here.

The tools need Xcode's clang and ld, and Python 3.12 or later with only its
standard library. Nothing is fetched. Another repository takes the SDK as
part of a release, with `tools/vendor.py` ([tools/](../tools/README.md)).

| file | what it is |
|---|---|
| `include/graft/Graft.h` | the calls an owned client makes: `graft_client_ready`, `graft_kernel_symbol`, `graft_route_kernel`, `graft_client_log` and `graft_client_state` |
| `Start.cpp` | an owned client's start glue: declares its kmod from `GRAFT_CLIENT_ID` and `GRAFT_CLIENT_VERSION`, calls `graft_client_initialize()` at start, and refuses to stop |
| `consumer.mk` | the make rules that link and stage a bound consumer |
| `consumer.py` | the three steps `consumer.mk` runs: `probe`, `reserve` and `stage` |
| `graft_macho.py` | the Mach-O reader and editor that `consumer.py` and `prelink.py` share |
| `prelink.py` | checks that a kernel collection OpenCore prelinked still has each staged consumer bindable |

## Building a bound consumer

A bound consumer compiles its objects with its own rules, then includes
`consumer.mk`, which:

1. **probes** each declared class's layout: it compiles a member-less
   subclass of every class in `src/graft.json` with the consumer's flags, and
   reads each class's size and the method in each vtable slot from the
   object;
2. **links once** without a manifest, and **reserves** room in a
   zero-filled file for the manifest and one 32-byte placeholder per import;
3. **links again**, with that file as `__DATA,__graft`;
4. **stages** the second link into `$(KEXT)`: it writes the manifest into
   `__DATA,__graft` and leaves the kext dormant for Graft to bind.

[docs/manifest.md](../docs/manifest.md) specifies the manifest, and
[docs/design.md](../docs/design.md) how Graft binds a consumer at boot.

### A worked example

A consumer with one class, `OurAccelerator`, derived from the System
collection's `IOAccelerator`, with Graft vendored at `graft/`:

```make
NAME     = OurAccelerator
VERSION  = 1.0.0
SDK      = $(shell xcrun --sdk macosx --show-sdk-path)
ARCH     = -arch x86_64
BUILD    = build
KEXT     = $(BUILD)/$(NAME).kext
BINARY   = $(KEXT)/Contents/MacOS/$(NAME)
OBJECTS  = $(BUILD)/obj/OurAccelerator.o
HEADERS  = src/OurAccelerator.hpp
CXXFLAGS = $(ARCH) -isysroot $(SDK) -mmacosx-version-min=11.0 -mkernel -nostdinc \
    -isystem $(SDK)/System/Library/Frameworks/Kernel.framework/Headers -DKERNEL \
    -std=c++17 -fapple-kext -fno-rtti -fno-exceptions -Os -Wall -Werror

all: $(BINARY)

$(BUILD)/obj/%.o: src/%.cpp $(HEADERS) Makefile ; mkdir -p $(@D) && xcrun clang++ $(CXXFLAGS) -c $< -o $@

include graft/sdk/consumer.mk
```

The compile rule uses make's one-line form, `target: prerequisites ; recipe`,
which means the same as a recipe on its own tab-indented line. The
consumer's `src/graft.json`:

```json
{
  "libraries": [{"id": "com.apple.iokit.IOGraphicsFamily"}],
  "classes": [
    {"name": "IOAccelerator", "header": "IOKit/graphics/IOAccelerator.h", "library": 0}
  ],
  "consumers": [{"name": "OurAccelerator", "header": "OurAccelerator.hpp", "parent": 0}]
}
```

Its `src/Info.plist` is an ordinary kext's, with `CFBundleExecutable`
`OurAccelerator`, `CFBundleVersion` `@VERSION@`, its `IOKitPersonalities`,
and `com.apple.iokit.IOGraphicsFamily` among its `OSBundleLibraries`. `make`
writes `build/OurAccelerator.kext`, whose Info.plist keeps the other
libraries but has no personalities and no IOGraphicsFamily.
`measure/binding-probe/` has the same shape, and imports through both a
pointer and a direct call.

### consumer.mk

Include `consumer.mk` after the Makefile's default target, since its first
rule would otherwise become the default. It reads these variables:

| variable | meaning |
|---|---|
| `NAME` | the kext's executable name, which must match the Info.plist's `CFBundleExecutable` |
| `VERSION` | replaces `@VERSION@` in `src/Info.plist` |
| `ARCH` | the target flags for both links, such as `-arch x86_64` |
| `BUILD` | a build directory; the rules write only under `$(BUILD)/graft` |
| `KEXT` | the staged bundle to write |
| `BINARY` | `$(KEXT)/Contents/MacOS/$(NAME)`, the target that writes `$(KEXT)` |
| `OBJECTS` | the kext's objects, which the Makefile's own rules compile |
| `HEADERS` | the consumer's headers; a change to one probes again |
| `CXXFLAGS` | the flags `OBJECTS` are compiled with; the probe compiles with them too |

The consumer supplies `src/graft.json` and `src/Info.plist`. `consumer.mk`
reads no `SDK` variable: the probe asks `xcrun` for the macOS SDK. Both
links run `xcrun clang++` with `$(ARCH) -nostdlib -Xlinker -kext`, the
objects and `-lcc_kext`; the second adds `-sectcreate __DATA __graft`.

Under `$(BUILD)/graft` the rules leave `layouts.json` (the probe's output),
`census` (the first link), `manifest-reserve.bin` (the reservation) and
`Unbound.kext` (the second link, with the versioned Info.plist). Staging
refuses an existing bundle, so the rule for `$(BINARY)` removes `$(KEXT)`
first. The layouts are probed again when `src/graft.json`, a header in
`HEADERS`, the Makefile, `consumer.py` or `graft_macho.py` changes.

## graft.json

The declaration names the libraries the consumer imports from, the
System-collection classes it derives from, and its own classes derived from
them. Indices count from 0. In this example, the second library and the
import illustrate the fields:

```json
{
  "libraries": [
    {"id": "com.apple.iokit.IOGraphicsFamily"},
    {"id": "com.example.OptionalFamily", "anchor": "ExampleService"}
  ],
  "classes": [
    {"name": "IOAccelerator", "header": "IOKit/graphics/IOAccelerator.h", "library": 0,
     "required": ["__ZN9IOService5startEPS_"]}
  ],
  "consumers": [{"name": "OurAccelerator", "header": "OurAccelerator.hpp", "parent": 0}],
  "imports": {
    "_example_function": {
      "library": 1,
      "candidates": ["_example_function", "_example_function_old"],
      "fallback": "_our_example_fallback"
    }
  }
}
```

**`libraries`**, required and not empty: each provider, by bundle
identifier.

| key | meaning |
|---|---|
| `id` | the provider's bundle identifier; required and unique |
| `anchor` | for a provider the consumer declares no class from: a class it defines, by whose metaclass Graft confirms it found the loaded provider; it is not probed, so it needs no header |
| `classes` | other class names the provider owns, so that their members are imported from it |

Every library needs a declared class or an anchor; the build fails without
one. Each must also be in the Info.plist's `OSBundleLibraries`, from which
staging removes it.

**`classes`**: the System-collection classes the consumer derives from.

| key | meaning |
|---|---|
| `name` | a top-level C++ class name, unique among the classes |
| `header` | the header that declares it, found through `CXXFLAGS` or `src/` |
| `library` | the index of its provider |
| `required` | the inherited methods whose slots the consumer calls, by mangled name |

The probe records each class's size and parent slot count, and Graft
requires the live class to match both. Of the slots, Graft checks those
`required` names, plus every slot a consumer class overrides. Each name in
`required` must be in exactly one slot of the probed layout
(`$(BUILD)/graft/layouts.json` lists them). Without `required`, every parent
slot is checked. A shorter list is only right when it comes from a complete
census of the virtual calls the consumer makes.

**`consumers`**: the consumer's own classes that derive from those.

| key | meaning |
|---|---|
| `name` | a top-level C++ class name |
| `header` | the header that declares it |
| `parent` | the index of the class in `classes` it derives from |

Graft fills each consumer vtable's inherited slots from the live parent, and
leaves the slots the consumer overrides. Every overriding and every appended
slot must point to the consumer's own code.

**`imports`**: policies for imports, keyed by Mach-O symbol name.

| key | meaning |
|---|---|
| `library` | the index of the provider; required |
| `candidates` | the names to try, in order; by default the symbol's own name alone |
| `fallback` | a symbol the consumer defines, bound when no candidate is defined |

Graft binds each undefined symbol that is named here, or that is a member,
vtable or typeinfo of a class a provider owns: a class in `classes` or in a
library's `classes`. A policy must name a symbol the consumer imports and
references. An import that no candidate resolves and that has no fallback
refuses the consumer, unless it is weak. The boot linker resolves the
remaining kernel and Boot-collection imports, as for any injected kext.

An import the consumer declares `__attribute__((weak_import))` is weak in
the manifest. Where nothing defines it and it has no fallback, Graft binds
it to the pointer the kernel's `gOSKextUnresolved` holds, so the consumer
can ask `OSKextSymbolIsResolved()` whether it is there. List its library as
for any import.

## What staging does

Staging checks its inputs before it creates the output bundle.

- **The manifest** goes at the start of `__DATA,__graft`. After it, each
  import gets a placeholder, 16-byte aligned and 32 bytes apart, so that a
  vtable import's address point, 16 bytes in, stays inside its own.
- **Imports become placeholders.** Each selected undefined symbol becomes a
  private local symbol defined at its placeholder. The symbol table is
  regrouped as the dynamic symbol table requires, and every external
  relocation and indirect-symbol entry is renumbered to match. The boot
  linker then relocates each site against its placeholder, and Graft
  rewrites it when it binds.
- **Consumer vtables are left for Graft.** Each consumer class's
  `superClass` marker is renamed `graftClass`, with its bytes and
  relocations kept, so that the boot linker does not patch its vtable
  against a parent it cannot see. The manifest records each vtable and its
  overridden slots; the inherited slots are not sites.
- **The Info.plist goes dormant.** Its personalities move into the
  manifest, each naming the consumer's bundle identifier, and the declared
  libraries leave `OSBundleLibraries`. Its other dependencies stay.

## Commands

`consumer.mk` runs the three steps of `consumer.py`, and each can also be
run by hand:

```
python3 -I -B sdk/consumer.py probe DECLARATION LAYOUTS [-- COMPILER_FLAGS...]
python3 -I -B sdk/consumer.py reserve BINARY DECLARATION LAYOUTS INFO_PLIST RESERVATION
python3 -I -B sdk/consumer.py stage KEXT DECLARATION LAYOUTS OUTPUT
```

- `probe` writes LAYOUTS, the JSON layout of every declared class. It
  compiles in a temporary directory under `/tmp`.
- `reserve` reads BINARY, the consumer linked without a manifest, and the
  versioned INFO_PLIST, and writes RESERVATION.
- `stage` copies the bundle KEXT, linked with the reservation, to OUTPUT.

Each step refuses an input it cannot use, or a consumer Graft could not
bind: it prints `consumer:` and the reason, and exits 1.

### prelink.py

```
python3 -I -B sdk/prelink.py COLLECTION KEXT...
```

COLLECTION is an x86-64 kernel collection after OpenCore prelinked our kexts
into it, and each KEXT a staged consumer. For each consumer, `prelink.py`
finds the fileset entry named by its bundle identifier and checks what Graft
relies on when it binds that entry:

- the entry's header starts its `__TEXT`, and each staged segment keeps its
  address relative to the header, since Graft adds manifest offsets to the
  entry's address;
- `__DATA,__graft` holds the staged manifest unchanged through the size the
  manifest declares; the placeholder tail after it is not compared;
- each site holds its exact placeholder plus addend, and each placeholder
  lies in the tail;
- no fixup-chain entry overlaps a branch site or the manifest.

An x86-64 collection stores every pointer as a chained rebase
(`DYLD_CHAINED_PTR_X86_64_KERNEL_CACHE`), which the booter turns into an
address when it slides the collection. So a pointer site must be a chain
entry, a plain rebase into the collection, whose target is its placeholder.
A region's `segment_offset` counts from the collection header, and a
rebase target from the lowest segment address. OpenCore reuses a region
slot for the kexts it injects, and writes those starts after the table's
declared size, so regions are matched by `segment_offset` and bounded by
the segment holding the table. Chains can run past their starting page, so
each region's chains are walked whole.

It prints a line for each consumer and exits 0 when all pass. On the first
defect it prints `prelink:` and the reason, and exits 1. It only reads the
collection.

## Tests

`make test` runs the SDK's tests with the others:

- `tests/tooling_test.py` builds an original fixture consumer with Xcode
  (`tests/consumer_fixture.py`) and checks probing, the import census,
  weak imports, anchors, required methods, placeholders, the renumbered
  symbol table, the renamed markers, a short reservation, truncated Mach-O
  files and the dormant bundle.
- `tests/prelink_test.py` links the staged fixture into synthetic
  collections the way OpenCore does, then corrupts them, and checks that
  `prelink.py` accepts the faithful one and refuses each defect.
