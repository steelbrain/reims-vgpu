# sdk

The files a kext builds with to work with Graft. An owned client compiles
against the header and the start glue; a bound consumer replaces its final
link with `consumer.mk`. [docs/consumers.md](../docs/consumers.md) explains
the two kinds and how to build each; this file is the reference for the
files here.

A consumer's build needs Xcode: its clang and ld, and `plutil`. `consumer.mk`
compiles the SDK's tool, `graft-kext`, from its C++ sources in `tool/` before it
first runs it. Nothing is fetched. Another repository takes the SDK as part of
a release, with `tools/vendor.py`
([docs/guides/taking-a-release.md](../docs/guides/taking-a-release.md)). A
release carries Graft's whole source tree around the SDK, so these links
resolve in that copy.

| file | what it is |
|---|---|
| `include/graft/Graft.h` | the calls an owned client makes: `graft_client_ready`, `graft_kernel_symbol`, `graft_route_kernel`, `graft_client_log` and `graft_client_state` |
| `include/graft/IOMobileFramebufferLegacy.h` | the stand-in parent an arm64e synthesized class derives from ([Synthesized classes](#synthesized-classes)) |
| `Start.cpp` | an owned client's start glue: declares its kmod from `GRAFT_CLIENT_ID` and `GRAFT_CLIENT_VERSION`, calls `graft_client_initialize()` at start, and refuses to stop |
| `consumer.mk` | the make rules that build `graft-kext`, then link and stage a bound consumer with it |
| `tool/` | the C++ sources of `graft-kext`, the command `consumer.mk` runs, described under [Commands](#commands) |
| `include/graft/ManifestFormat.h` | the manifest's records and constants, which `graft-kext` writes and Graft reads, and the validator both apply ([docs/manifest.md](../docs/manifest.md)) |

## Building a bound consumer

A bound consumer compiles its objects with its own rules, then includes
`consumer.mk`, whose one rule runs `graft-kext build` ([Commands](#commands)),
which:

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
    -std=c++17 -fapple-kext -fno-rtti -fno-exceptions -Os -Wall -Werror \
    -Igraft/sdk/include

all: $(BINARY)

$(BUILD)/obj/%.o: src/%.cpp $(HEADERS) Makefile ; mkdir -p $(@D) && xcrun clang++ $(CXXFLAGS) -c $< -o $@

include graft/sdk/consumer.mk
```

`CXXFLAGS` adds the SDK's `include/` (`-Igraft/sdk/include` above, for the
path the consumer vendored Graft at), so a source or header can
`#include <graft/Graft.h>` and the stand-in parents under `graft/`; the
probe compiles with `CXXFLAGS` too, so it resolves the same headers.

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
libraries and its personalities, now gated, but no IOGraphicsFamily.
`examples/hello-consumer/` has the same shape, and imports through both a
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
| `GRAFT_DECL` | optional: the declaration, `src/graft.json` unless set before the include |
| `GRAFT_INFO_PLIST` | optional: the Info.plist, `src/Info.plist` unless set before the include |
| `GRAFT_KEXT` | optional: the `graft-kext` the rules run, `$(BUILD)/graft/graft-kext` unless set before the include or in the environment; when it is set, `consumer.mk` builds no tool of its own |

The consumer supplies the declaration and the Info.plist. A Makefile that
builds more than one variant from one source, or a per-architecture
Info.plist, sets `GRAFT_DECL` or `GRAFT_INFO_PLIST` before including
`consumer.mk`. `consumer.mk` reads no `SDK` variable: the probe asks `xcrun`
for the macOS SDK. The probe compiles with `$(CXXFLAGS) -Isrc`, for the one
architecture `ARCH` names; `CXXFLAGS` may name none, but no other. Both links
run `xcrun clang++` with `$(ARCH) -nostdlib -Xlinker -kext`, the objects and
`-lcc_kext`; the second adds `-sectcreate __DATA __graft`. When `ARCH` names
arm64e, both links also take `-Xlinker -no_data_const`, which keeps the
consumer's import GOT and vtables in writable `__DATA`, where Graft writes
them.

Unless `GRAFT_KEXT` names another command, `consumer.mk` first compiles
`graft-kext` into `$(BUILD)/graft/graft-kext`, with `xcrun clang++ -std=c++17
-O2` and the sources in `tool/`, and compiles it again when one of them, or
`include/graft/ManifestFormat.h`, changes. Its warnings are not errors, so a
newer compiler does not stop the build.

The rule for `$(BINARY)` runs every step again when an object, the
declaration, the Info.plist, a header in `HEADERS` or in the SDK's
`include/graft/`, the makefile that includes `consumer.mk`, or `graft-kext`
changes. It leaves each step's output under `$(BUILD)/graft`:
`layouts.json` (the probe's), `census` (the first link),
`manifest-reserve.bin` (the reservation) and `Unbound.kext` (the second link,
with the versioned Info.plist). It replaces `$(KEXT)`.

## graft.json

The declaration names the libraries the consumer imports from, the
System-collection classes it derives from, and its own classes derived from
them. An arm64e consumer can also declare `synthesized` classes, described in
[Synthesized classes](#synthesized-classes). Indices count from 0. In this
example, the second library and the import illustrate the fields:

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
slot is checked except the headers' `_RESERVED` placeholders, whose names say
nothing about the method a running kernel keeps there. A shorter list is only
right when it comes from a complete census of the virtual calls the consumer
makes.

**`consumers`**: the consumer's own classes that derive from those.

| key | meaning |
|---|---|
| `name` | a top-level C++ class name, unique among `consumers` and `synthesized` |
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

## Synthesized classes

An arm64e consumer can derive from a parent that no Apple SDK header declares
and whose vtable changes between releases. It compiles against a stand-in
declaration of the parent, and Graft lays out the class's vtable when it
binds, placing each override on the live slot that holds the same method
([docs/manifest.md](../docs/manifest.md#synthesized-classes)). The SDK has one
stand-in, `include/graft/IOMobileFramebufferLegacy.h`, for
IOMobileGraphicsFamily's `IOMobileFramebufferLegacy`. On x86-64 the build
refuses a synthesized class.

The stand-in goes in `classes` like any parent, and the consumer's class in
`synthesized`:

```json
{
  "libraries": [{"id": "com.apple.iokit.IOMobileGraphicsFamily"}],
  "classes": [
    {"name": "IOMobileFramebufferLegacy", "header": "graft/IOMobileFramebufferLegacy.h",
     "library": 0}
  ],
  "synthesized": [
    {"name": "OurDisplay", "header": "OurDisplay.hpp", "parent": 0,
     "sibling": "AppleParavirtDisplay",
     "overrides": {
       "are_timings_enabled": {"required": true},
       "supported_frame_info": {"required": true,
                                "selectors": ["__ZN10OurDisplay20supported_frame_infoEjPjPA64_c"]},
       "start_hardware": {}
     }}
  ]
}
```

| key | meaning |
|---|---|
| `name` | a top-level C++ class name, unique among `consumers` and `synthesized` |
| `header` | the header that declares it |
| `parent` | the index in `classes` of the stand-in it derives from |
| `sibling` | optional: a concrete subclass of the parent, which implements the parent's pure methods, through which Graft finds a pure method's slot. Graft reads it from whichever kext of the boot collection defines it, loaded or not |
| `overrides` | required and not empty: the methods the class overrides, keyed by method name, such as `start_hardware` |

Each name in `overrides` must be one method the class itself defines in one
of the stand-in's slots. Its value takes two optional keys:

| key | meaning |
|---|---|
| `required` | `true`: Graft refuses the bind when it finds no live slot for the method. Otherwise Graft leaves the override out when it finds no slot for it |
| `selectors` | other mangled symbols for the method, tried after the class's own, for a method whose parameters differ between releases |

Mark the parent's pure methods `required`. Graft reads only the method from a
selector, not its class, so a selector can be written under the consumer's
own class, as above.

From the declaration and the linked consumer, staging records for each
class its overrides, its compiled vtable's length and the stand-in's size:

- Each declared override takes the class's own method symbol as its first
  selector, followed by its `selectors`, and its implementation and signing
  context from the class's vtable.
- Every other method the class defines in a stand-in slot is an override
  too, and a required one, with its own symbol as its only selector. These
  include the destructors and `getMetaClass` that
  `OSDefineMetaClassAndStructors` defines. So the bind refuses rather than
  leave the parent's method in their place.
- The stand-in's reserved slots are not overrides. They only lengthen the
  vtable, and nothing calls them.
- The inherited slots are not import sites.

Graft measures the stand-in's live class but does not hold it to the probed
size and slots. It refuses the bind when the live class is larger than the
stand-in, or when its vtable is longer than the class's compiled vtable.

Graft signs each slot with the discriminator the running system's callers
use, which the boot firmware reads from the boot collection and hands over
per vtable. It needs the parent's vtable and that of each of the parent's
ancestors its library defines, and refuses the bind when one is missing.
For `IOMobileFramebufferLegacy`, Ambrosia's configuration lists these under
`Kernel > ExportVtableDiscriminators`:

```xml
<array>
  <string>__ZTV25IOMobileFramebufferLegacy</string>
  <string>__ZTV26IOMobileFramebufferService</string>
  <string>__ZTV19IOMobileFramebuffer</string>
</array>
```

### IOMobileFramebufferLegacy.h

The stand-in declares the class's pure methods, which a consumer implements;
the concrete methods a consumer overrides; `start`, `stop` and `free`; and
`note_vbl_ISR`, a non-virtual method a consumer calls, which Graft binds like
any import. Methods with no known return type return `graft_iomfb_word`, a
64-bit word. `GRAFT_IOMFB_VTABLE_RESERVE` reserved virtual slots make the
compiled vtable at least as long as the live class's, and a
`GRAFT_IOMFB_LEGACY_RESERVE`-byte field holds the live class's fields, so the
consumer's own fields follow them.

A display calls none of the virtual methods the stand-in declares except
`start`, `stop` and `free`: not virtually, since Graft fills its vtable in the
live order, and not as a qualified call such as
`IOMobileFramebufferLegacy::start_hardware()`, which under `-fapple-kext` loads
the method from the class's vtable at the stand-in's compiled slot, a slot the
live class gives to another method. `start`, `stop` and `free` lie in
IOService's part of the vtable, which keeps the SDK's order, so a qualified
call to them is right:

```cpp
bool OurDisplay::start(IOService *provider) { return IOMobileFramebufferLegacy::start(provider); }
```

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
- **The Info.plist goes dormant.** Its personalities stay but gain an
  `IOPropertyMatch` gate on the provider nub that Graft opens when it binds,
  each naming the consumer's bundle identifier. A personality may match any
  provider, and must name one in `IOProviderClass`. A personality that has its
  own `IOPropertyMatch`, a dictionary or a list of them, keeps it, and the
  gate's key joins the dictionary, or each one of the list. A personality
  whose `IOPropertyMatch` already holds the gate's key, as a staged
  Info.plist's do, is refused, since Graft would then watch only for nubs
  whose gate is already open. The declared libraries leave
  `OSBundleLibraries`, and any per-architecture list such as
  `OSBundleLibraries_x86_64`. Its other dependencies stay.
- **The manifest carries the provider matching.** It records each personality
  as the author wrote it, without the gate and without `IOClass`,
  `CFBundleIdentifier`, `IOProbeScore` and `IOMatchCategory`, so Graft can
  find the nubs to open the gate on
  ([docs/manifest.md](../docs/manifest.md#provider-matching)). The
  personalities size the manifest, so `reserve` reads the same Info.plist the
  bundle `stage` copies holds.

## Commands

`graft-kext` is a C++17 program that uses only the standard library. `probe`
and `build` run Xcode's `clang++` through `xcrun`, and `build` also runs
`plutil`, through the shell, so those two need a Mac with Xcode. `reserve`,
`stage` and `check-prelinked` run no tool, and `check-prelinked` builds and
runs on any host with a C++17 compiler. `consumer.mk` builds it
([consumer.mk](#consumermk)) and runs `build`:

```
graft-kext build --name=NAME --version=VERSION --arch=ARCH \
    --declaration=DECLARATION --info-plist=INFO_PLIST --work=WORK --output=KEXT \
    OBJECT... [-- COMPILER_FLAGS...]
```

`consumer.mk` passes `NAME`, `VERSION`, `ARCH`, `GRAFT_DECL`,
`GRAFT_INFO_PLIST`, `$(BUILD)/graft` as WORK, `KEXT`, then `OBJECTS` and, after
`--`, `$(CXXFLAGS) -Isrc`. ARCH is one argument holding every target flag, such
as `--arch="-arch x86_64"`. `build` runs the steps of
[Building a bound consumer](#building-a-bound-consumer) in order, the probe
with COMPILER_FLAGS for ARCH's architecture, and writes what
[consumer.mk](#consumermk) says it leaves. It prints nothing of its own unless it refuses; the compiler, linker and
`plutil` print their own errors.

Its steps can also be run one at a time, with the links between them run by
hand as `consumer.mk` documents:

```
graft-kext probe DECLARATION LAYOUTS [-- COMPILER_FLAGS...]
graft-kext reserve BINARY DECLARATION LAYOUTS INFO_PLIST RESERVATION
graft-kext stage KEXT DECLARATION LAYOUTS OUTPUT
```

- `probe` writes LAYOUTS, the JSON layout of every declared class. It
  compiles in a directory of its own under `TMPDIR`, or `/tmp` when that is
  unset, and removes it.
- `reserve` reads BINARY, the consumer linked without a manifest, and the
  versioned INFO_PLIST, whose personalities the manifest records, and writes
  RESERVATION.
- `stage` copies the bundle KEXT, linked with the reservation, to OUTPUT. It
  stages the copy beside OUTPUT and moves it into place once it is whole, so a
  staging that fails leaves no OUTPUT.

Each command refuses an input it cannot use, or a consumer Graft could not
bind: it prints `consumer:` and the reason, and exits 1. A link or a
`plutil -lint` that fails is refused with its command line. The declaration
and the layouts are read as JSON and the Info.plist as an XML property list,
and each is refused if it nests more than 64 levels deep. A malformed command
line prints the usage and exits 2.

### check-prelinked

```
graft-kext check-prelinked COLLECTION KEXT...
```

COLLECTION is an x86-64 kernel collection after OpenCore prelinked Graft and the
kexts it serves into it, and each KEXT a staged consumer. For each consumer, the
check finds the fileset entry named by its bundle identifier and checks what
Graft relies on when it binds that entry:

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
defect it prints `prelink:` and the reason, and exits 1. A malformed command
line prints the usage and exits 2. It only reads the collection.

Outside a consumer's build, such as where the collection is made, compile the
tool from the SDK with any C++17 compiler, then run it. With the SDK at
`$GRAFT_SDK`:

```sh
c++ -std=c++17 -O2 -I"$GRAFT_SDK/include" "$GRAFT_SDK"/tool/*.cpp -o graft-kext
./graft-kext check-prelinked COLLECTION KEXT...
```

## Tests

`make test` runs the SDK's tests with the others. It first builds two
consumers with `graft-kext build`: the fixture in `tests/consumer/`, for
x86-64 and for arm64e, and the synthesized display in `tests/synthesized/`,
for arm64e. The C++ tests link every source in `tool/` but its command line,
`cli.cpp`, and read what those builds wrote.

- `tests/staging_test.cpp` checks, on the fixture, probing, the import census,
  weak imports, anchors, required methods, placeholders, the renumbered symbol
  table, the renamed markers, a short reservation, truncated Mach-O files and
  the dormant bundle with its provider matching. On the arm64e build it checks
  the chained fixups the tool reads (each import bind by name, the weak ones
  flagged, every rebase inside the image) and the version-5 manifest: signed
  sites carry their signing context, and each provider import becomes a
  rebase at its placeholder.
- `tests/synth_emit_test.cpp` checks the synthesized-class and override
  records staging writes for the display.
- `tests/prelink_test.cpp` links the staged x86-64 fixture into synthetic
  collections the way OpenCore does, then corrupts them, and checks that the
  prelink check accepts the faithful one and refuses each defect, and what
  `graft-kext check-prelinked` prints and exits with.
- `tests/graft_kext_test.py` runs `graft-kext` as a consumer's build does, on
  the fixture for x86-64 and arm64e, and checks only what that build sees:
  the staged bundle, the provider matching its manifest carries for
  personalities of any provider, the files under the work directory, that
  `build` writes what its steps run one at a time write, and each refusal's
  exit status and text.
- `tests/consumer_mk_test.py` runs `consumer.mk` under make, from a makefile
  of another name, and checks that the staged bundle is current after a
  build, and not after an SDK header changes or a build fails.
- `tests/sdk_standin_test.py` compiles a subclass of the
  `IOMobileFramebufferLegacy` stand-in, and checks its reserved slots and the
  symbols its overrides produce.
