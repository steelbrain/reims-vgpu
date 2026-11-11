# Graft's source

This directory is the Graft kext. The same sources build one kext for x86-64
and one for arm64e: the architecture-neutral core in this directory, plus
`x86_64/` or `arm64e/`.

This README maps the files and follows a boot through them.
[docs/design.md](../docs/design.md) explains why Graft works the way it does,
and [docs/manifest.md](../docs/manifest.md) specifies what a consumer embeds.

The code is C++17 for the kernel: no exceptions, no RTTI, no standard library.
Types are PascalCase, functions and variables camelCase. The constants of the
generic mechanism are in one place, [Policy.h](Policy.h): AMFI's identifier and
the route limits. No kext Graft serves is named anywhere in the source: a
client is served because the identifier it passes names a placed kext, and a
consumer is bound because it carries a manifest.

## The boot, end to end

The kernel calls `graft_start` ([Start.cpp](Start.cpp)) from Graft's own kmod
load, before Graft's own `Graft` service starts. It waits for no console,
framebuffer or BSD. `graft_start` runs these steps in order. If one fails, it
records why, stops, and returns success so the guest keeps booting.

1. **Log.** [Log.cpp](Log.cpp) allocates the diagnostics, so every later step
   can report.
2. **Kernel image.** `openKernel` ([RuntimeImage.cpp](RuntimeImage.cpp)) finds
   the running kernel in the Boot kernel collection and copies its symbol
   table, calibrated against the address Graft's own link gave `IOLog`. The
   Mach-O reader is [MachO.h](MachO.h).
3. **Write path and routes.** The architecture's `KernelWrite.cpp` prepares
   kernel writes and its `Route.cpp` prepares routing.
4. **Consumers and binding trigger.** `initializeBinding`
   ([KernelBinding.cpp](KernelBinding.cpp)) finds the bound consumers, each
   placed kext that carries a manifest, and installs the architecture's
   trigger ([BindingPlacement.h](BindingPlacement.h)). On x86-64 it routes the
   kernel's two catalogue entry points, so that each later catalogue send calls
   Graft back. On arm64e there is no trigger: the boot firmware placed every
   image before Graft started.
5. **Active.** Graft marks itself active, so a client may call it, and makes a
   first `bindConsumers` attempt.

After that:

- **On x86-64, each catalogue send** calls `bindConsumers`, which tries each
  dormant consumer's bind again until it succeeds or is refused. On arm64e the
  first attempt binds against the images already in place.
- **Each owned client** calls the C interface in
  [Graft.h](../sdk/include/graft/Graft.h), implemented in [Start.cpp](Start.cpp),
  to look up kernel symbols and install its routes.

## The files

### Entry and interface

- [Start.cpp](Start.cpp): the kmod entry (`graft_start`, `graft_stop`), the
  `Graft` IOService, and the `extern "C"` functions clients call. Every call's
  identifier must name a placed kext (`placedBase`); the three that act on the
  kernel also require Graft to be active.
- [Policy.h](Policy.h): AMFI's bundle identifier, whose policy the arm64e routes
  shadow, and the route limits: at most sixteen routes in one call, and a shared
  pool of thirty-two route slots on x86-64.

### Diagnostics

- [Log.h](Log.h), [Log.cpp](Log.cpp): the log and the registry state
  (`GraftState`, `GraftReason`, `GraftLog`, `GraftClient.<id>`).

### Reading images

- [MachO.h](MachO.h): a reader of a mapped Mach-O's load commands, fileset
  entries and symbol table, which tells which of a name's local copies sits at
  an address. It makes no kernel calls, so the host tests use it as it is.
- [RuntimeImage.h](RuntimeImage.h), [RuntimeImage.cpp](RuntimeImage.cpp): a
  view over a live image that answers bounded reads, symbol lookups and section
  locations, plus `openKernel`. `openEntryDefining` opens the one collection
  entry that defines a symbol, using the slide another entry measured, for a
  kext that is placed but never started.

### The manifest and the binding plan

- [Manifest.h](Manifest.h): the manifest format, version 4 for x86-64 and
  version 5 for arm64e, and its validator, which it takes from the SDK's
  [ManifestFormat.h](../sdk/include/graft/ManifestFormat.h), shared with the
  SDK's build tool; and each provider's anchor class
  ([docs/manifest.md](../docs/manifest.md)).
- [Resolve.h](Resolve.h): resolve a symbol or an import within its declared
  provider, with candidates in order and a fallback in the consumer.
- [Binding.h](Binding.h): plan a consumer's import sites as writes, without
  touching the consumer.
- [Consumer.h](Consumer.h): turn a validated manifest into import-site
  writes, map each consumer offset to its runtime address, and check where
  each planned write points before anything is written.
- [Layout.h](Layout.h): compare the class a consumer was compiled against with
  the running class's size and vtable length.
- [Vtable.h](Vtable.h): plan the inherited vtable slots Graft fills. A class
  that a synthesized class names as its parent is measured, not checked.
- [Memory.h](Memory.h): allocate and free a zeroed array through the kernel
  allocator.

### Kernel effects

- [WritePlan.h](WritePlan.h): the write transaction. It checks every old value,
  writes, reads back, and rolls back on failure, over a memory interface the
  caller supplies.
- [KernelWrite.h](KernelWrite.h): one kernel-write transaction. Each
  architecture supplies the memory operations.
- [Route.h](Route.h): the route installer's interface. Each architecture
  implements it.
- [KernelBinding.h](KernelBinding.h), [KernelBinding.cpp](KernelBinding.cpp):
  the binding, architecture-neutral
  ([docs/design.md](../docs/design.md#binding-a-consumer)). `findConsumers`
  keeps a record for each placed kext whose `__DATA,__graft` starts with the
  manifest's magic, and each record holds that consumer's state and gate.
  `attemptBind` runs named stages for one consumer: open the manifest, verify the providers and class layouts,
  parse its provider matching, plan the import and vtable writes, build any
  synthesized vtable, check each write's target and commit the writes, and
  open the consumer's activation gate on every nub a personality's provider
  matching matches, present or published later. A refusal at any stage leaves
  the consumer unbound.
- [BindingPlacement.h](BindingPlacement.h): how the binding finds the images it
  works on: install the trigger, locate a provider by bundle identifier, visit
  every placed kext so the binding can find the consumers among them, and ask
  whether a kext has started. It also says where in the consumer a write may
  land. Each architecture implements it, and arm64e also finds one placed kext
  by bundle identifier, for its route.

### The x86-64 layer

- [x86_64/Kernel.cpp](x86_64/Kernel.cpp): the Boot collection's header, from
  the kernel's own accessor, and `IOLog`'s linked address.
- [x86_64/Fixup.h](x86_64/Fixup.h): an import site's new value, a pointer or a
  branch displacement, with checked arithmetic.
- [x86_64/X86Route.h](x86_64/X86Route.h): instruction decoding and building a
  route's entry patch and trampoline.
- [x86_64/Route.cpp](x86_64/Route.cpp): the route installer. It reserves slots
  from the shared executable pool, refuses a function another client already
  holds (naming that client) and overlaps, refuses once BSD is published, and
  commits trampolines before entries.
- [x86_64/KernelWrite.cpp](x86_64/KernelWrite.cpp): the write window, which
  clears `CR0.WP` with the other CPUs stopped and serializes instruction
  fetch.
- [x86_64/BindingPlacement.cpp](x86_64/BindingPlacement.cpp): routes the two
  catalogue entry points, finds a provider through `OSKext` (registered and
  started) and the live collection's fileset, with the pageable collection as
  a fallback, and visits the Boot collection's fileset entries for the
  consumers. A write may land anywhere the consumer maps readable, its
  `__TEXT` included, since the write window lifts write protection.

### The arm64e layer

- [arm64e/Kernel.cpp](arm64e/Kernel.cpp): the Boot collection's header, from
  the device tree, and `IOLog`'s address, read as a data symbol so that taking
  it is not authenticated.
- [arm64e/KernelWrite.cpp](arm64e/KernelWrite.cpp): stores into the consumer's
  `__DATA`, which its collection maps writable. Graft runs after the kernel's
  lockdown and the consumer is dormant, so there is no window to open.
- [arm64e/Arm64Route.h](arm64e/Arm64Route.h), [arm64e/Route.cpp](arm64e/Route.cpp):
  routing by shadowing AMFI's MAC policy. Kernel text and AMFI's
  `mac_policy_ops` are read-only when Graft runs, so each client's request
  copies the current `mac_policy_ops` and `mac_policy_conf` (AMFI's own the
  first time, the newest shadow after) into fresh memory, re-signs the routed
  operations in the copy for the client's replacements, and repoints the
  writable `mac_policy_list` element at the copy. The repoint is one atomic
  compare-and-exchange, which refuses if the element no longer holds the conf it
  currently expects. Several clients route one after another, and an operation
  one client holds is refused to the next. The client's image, which the
  replacements must lie in, is found as the consumer's is (`placedBase`) and is
  required. Arm64Route.h is the plan, which the host tests run; Route.cpp does
  the live reads and the swap.
- [arm64e/BindingPlacement.cpp](arm64e/BindingPlacement.cpp): no trigger to
  install. A provider is its fileset entry in the Boot collection, which Graft
  starts with `OSKextLoadKextWithIdentifier` before binding against it, and the
  consumers are found among the device-tree nodes of the placed kexts. Nothing
  waits for a kext to start. A write may land only in a writable data segment
  that the kernel has not locked read-only.
- [arm64e/Fixup.h](arm64e/Fixup.h): an import site's new value. Each site holds
  a pointer the loader signed for the local placeholder. Graft authenticates it
  and signs the real target under the same key and discriminator.
- [arm64e/Synthesize.h](arm64e/Synthesize.h): the plan for a synthesized class:
  a method's identity from its mangled name, its symbol rebuilt under another
  class, the live slot that holds an address, whether a slot found by its
  discriminator alone holds another named method, and where each override
  goes. It refuses a required override with no slot, an ambiguous slot, and
  two overrides on one slot.
- [arm64e/VtableDisc.h](arm64e/VtableDisc.h): parse the discriminator tables
  the firmware hands over, check each against the live vtable's address, length
  and signed words, give each slot the discriminator of the most-base table
  that covers it, and find the one slot that carries a given discriminator.
- [arm64e/SynthesizeVtable.h](arm64e/SynthesizeVtable.h): build the
  synthesized vtable word by word. An override's slot gets the consumer's
  implementation and every other slot the live parent's method, each signed
  for its own address with its discriminator. `KernelBinding.cpp` reads the
  tables, opens the sibling class, resolves each override and stages the words
  ([docs/manifest.md](../docs/manifest.md#synthesized-classes)).

## The architecture boundary

What differs between x86-64 and arm64e has a file in each of
[x86_64/](x86_64/) and [arm64e/](arm64e/), behind an interface in this
directory:

- how the Boot collection and the calibration address are found
  (`Kernel.cpp`);
- how kernel memory is written (`KernelWrite.cpp`, behind
  [KernelWrite.h](KernelWrite.h));
- how a client's kernel functions are routed (`Route.cpp`, behind
  [Route.h](Route.h));
- how the binding finds its images: the trigger, the providers, the placed
  kexts the consumers are found among and the started check, and where a bound
  write may land
  (`BindingPlacement.cpp`, behind [BindingPlacement.h](BindingPlacement.h));
- how an import site's new value is built (`Fixup.h`, behind
  [Fixup.h](Fixup.h), with the shared arithmetic in
  [FixupShared.h](FixupShared.h)).

The manifest carries both site forms ([Manifest.h](Manifest.h)): a version-4
x86-64 pointer or `0xe8`/`0xe9` branch, and a version-5 arm64e pointer whose
two trailing bytes hold its signing context. The SDK's `graft-kext` writes the
right one from the consumer's relocations or chained fixups. Version 5 also
carries the segment, fill and synthesized-class tables, which only arm64e
uses.

## What the arm64 boot firmware provides

On x86-64, OpenCore injects Graft and its kexts into the Boot kernel
collection, and Graft finds everything else itself. On arm64e, the firmware
that boots macOS does more of that work, and Graft reads the result from the
device tree. Graft was developed with Ambrosia, its authors' own arm64 boot
firmware, which is why the node is named `ambrosia`. Graft needs the
following from the firmware:

- **Placement.** Graft and the kexts it serves are placed in the Auxiliary
  kernel collection before the kernel starts, with their undefined symbols
  bound to kernel exports, and their `__DATA` mapped writable.
- **`/chosen/ambrosia`, property `boot-kc-base`.** 8 bytes: the slid address
  of the Boot kernel collection's Mach-O header.
- **`/chosen/ambrosia/<bundle id>`, property `base`.** One node per placed
  kext, named by its bundle identifier. 8 bytes: the slid address of the
  kext's Mach-O header. Graft reads each one at start and binds, as a
  consumer, each kext whose image carries a manifest. It reads a client's to
  check that its route replacements lie in its code. The other children of
  `/chosen/ambrosia`, such as `vtable-disc`, have no `base`.
- **`/chosen/ambrosia/vtable-disc`.** One child node per vtable a synthesized
  class needs ([docs/guides/synthesized-class.md](../docs/guides/synthesized-class.md)),
  with these properties:

  | property | value |
  |---|---|
  | `symbol` | the vtable's mangled symbol, NUL-terminated |
  | `vtable-base` | 8 bytes: the slid address of the vtable's first word |
  | `first-slot` | 4 bytes: the index of the first method word, 2 |
  | `slots` | one 8-byte record per method word, in order: `u16 slot`, `u16 discriminator`, `u8 key`, `u8 address diversity`, 2 bytes of padding |

  Every record has key 0 (IA) and address diversity 1, its slot is
  `first-slot` plus its index, and its discriminator is the one the live
  vtable's word at that slot is signed with.
  [arm64e/VtableDisc.h](arm64e/VtableDisc.h) refuses any other table.

All values are little-endian. The firmware boots with SIP on and passes no
boot-arg for Graft.
