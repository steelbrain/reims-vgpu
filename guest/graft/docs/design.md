# How Graft works

This document describes Graft as it is built: where it runs, what it does from
the moment the kernel starts it, and how it decides whether a kext it serves
can run. [manifest.md](manifest.md) specifies what a bound consumer embeds,
[consumers.md](consumers.md) explains how each kind of kext is built, and
[src/README.md](../src/README.md) maps the source.

## Where it runs

Graft is one kext, `ai.aneesiqbal.graft`, built from one source for two
architectures: `Graft.kext` for x86-64 macOS guests and `Graft-arm64e.kext`
for arm64e macOS guests. On x86-64, OpenCore injects it at boot beside the
kexts it serves. On arm64e, the boot firmware places it. Either way SIP stays
on, and Graft reads no boot-args.

A few terms recur below:

- **Kernel collections.** Since macOS 11 the kernel and its kexts ship as
  prelinked kernel collections. The **Boot collection** holds the kernel and
  the kexts needed to boot. The **System collection** holds most other Apple
  kexts, IOGraphicsFamily among them, and kernelmanagerd registers it later in
  boot. On x86-64, OpenCore injects kexts into the Boot collection and links
  them against what it holds, so it cannot link them against the System
  collection. On arm64e, the boot firmware places Graft and the kexts it serves
  in the **Auxiliary collection**.
- **Fileset entry.** One kext's Mach-O image inside a collection.
- **Catalogue send.** The kernel hands kext personalities to the IOKit
  catalogue through `OSKext::sendAllKextPersonalitiesToCatalog`, which then
  matches them against devices. It does so early in boot and again as more
  kexts are registered, such as when the System collection is.

Graft's Info.plist marks it `OSBundleRequired` Root and gives it one IOKit
personality, a `Graft` service matching `IOResources`. That personality makes
the kernel start the kext on every boot, even one with no consumer present.

Graft serves two kinds of kexts, and any placed kext can be either. Graft names
none of them:

- An **owned client** calls Graft's interface, declared in
  [sdk/include/graft/Graft.h](../sdk/include/graft/Graft.h), passing its own
  bundle identifier. Graft serves the call when that identifier names a placed
  kext ([Owned clients](#owned-clients)).
- A **bound consumer** derives from classes, or calls functions, in libraries
  of the System collection, and builds with the SDK, which writes a manifest
  into its `__DATA,__graft` section. Graft binds every placed kext that carries
  one, any number of them, by symbol name, and opens each one's gate on the
  providers its own personalities match
  ([Binding a consumer](#binding-a-consumer)).

## Start

The kernel calls Graft's own kmod start routine, `graft_start`. Graft does all
of its setup there, without waiting for a boot console, a framebuffer, BSD or
any other driver. In order, it:

1. allocates its log;
2. opens the running kernel's image, the `com.apple.kernel` entry of the Boot
   collection, and copies its symbol table. To place that table in memory, it
   compares the table's value for `_IOLog` with the address the kernel's
   linker gave Graft's own `IOLog` import;
3. prepares its kernel writes and its routes;
4. finds the bound consumers: each kext placed at boot whose
   `__DATA,__graft` section starts with the manifest's magic. On x86-64 it
   looks through the Boot collection's fileset entries, and on arm64e through
   the device-tree nodes the firmware publishes for the kexts it places. It
   logs each consumer it finds, or that there is none;
5. installs its binding trigger. On x86-64 it routes the kernel's two
   catalogue entry points, `OSKext::sendAllKextPersonalitiesToCatalog` and
   `OSKext::copyAllKextPersonalities`, so that every later catalogue send gives
   it a chance to bind. On arm64e there is no trigger;
6. marks itself active, so a client may call it, and makes a first attempt to
   bind each consumer.

If a step fails, Graft records why in its state, stops there and returns
success to the kernel, so the boot goes on without it.

## Reading kernel images

Graft reads Mach-O images in memory: the kernel, the consumers, and each
System-collection library a consumer imports from. It walks their load
commands, finds their fileset entries, and copies their symbol tables before
using them. It refuses an image whose load commands exceed 64 KiB.

A library can hold several local definitions of one name. Graft's lookup
treats repeated entries at one address as one definition, and distinct
addresses as ambiguous. Ambiguity refuses the lookup; Graft never picks one
copy silently.

## Owned clients

Each call passes a bundle identifier, and Graft serves it when that identifier
names a kext the boot linker placed: a fileset entry in the Boot collection on
x86-64, a device-tree node the firmware wrote on arm64e (`placedBase`). Graft
names no client, so any placed kext can be one, and every `GraftClient.<id>`
registry key names a real kext. It does not otherwise check which kext makes
the call.

- `graft_client_ready` returns whether Graft is active and the identifier names
  a placed kext. The other calls that act on the kernel check the same.
- `graft_kernel_symbol` looks a symbol up by name in the running kernel's own
  symbol table, not in other kexts. A miss sets the client's state to
  `refused`, naming the symbol.
- `graft_route_kernel` sends calls to kernel functions to the client's
  replacements, and gives each replacement a callable original. See
  [Routing](#routing).
- `graft_client_log` adds a line to Graft's log, and `graft_client_state` sets
  the client's state and reason in the registry. These two also require the
  identifier to name a placed kext, so neither can publish state under a name no
  kext owns. They do not require Graft active, so a client can record that
  Graft refused at start. On x86-64 they need only the Boot collection's load
  commands to be valid, and on arm64e only the device tree.

### Routing

A call to `graft_route_kernel` installs up to sixteen routes, all of them or
none, and the routes stay for the rest of the boot. Each request names what to
route, the replacement that receives its calls, and where Graft stores the
callable original. Graft passes each call through without reading its
arguments, so the replacement and the original must have the routed
function's prototype.

Graft writes the original as an untyped `void *`. On arm64e, kernel code
authenticates a call through a typed function pointer with a discriminator
derived from its prototype, and Graft signs the original as a `void *`, with
no discriminator. The client keeps the original in a `void *` and converts it
to the prototype at the call, which re-signs it. The same code works on
x86-64, where the conversion costs nothing.

What a request names, and how Graft installs it, differs by architecture.

**On x86-64**, each route names a function in the kernel's own symbol table,
such as `_mac_vnode_check_open`, and the replacement and original have that
function's prototype. Graft has room for 32 routes for the whole boot, shared
first come first served: its own two for binding, and up to 30 more across every
client. Installs are serialized, so clients that route take their slots in
order. It refuses to route once BSD has started.

Graft replaces the function's first instructions, at least 14 bytes, with a
jump to the replacement. It moves the instructions it displaces, at most 32
bytes, into a trampoline, which the `original` pointer calls. It refuses a
route when:

- the function starts with `ENDBR64`, or ends within its first 14 bytes;
- an instruction it would displace is one it cannot decode or relocate, such
  as a branch into the middle of another displaced instruction, or a memory
  operand that refers to the displaced bytes;
- a displaced `call`, relative or indirect, returns inside the first 14 bytes,
  where the jump now is;
- the function is already routed by another client or by Graft's own binding;
  the refusal names which client holds it;
- the `original` pointer is already used by a route, or is not pointer-aligned;
- the replacement lies outside the client's own executable segments, or Graft
  cannot find the client's placed image. This check is required, not skipped, so
  a route's replacement is always the client's real placed code.

Graft decodes only the bytes it displaces, not the whole function, because it
cannot find a function's end reliably: the next symbol bounds a symbol, not a
function, and a function body may embed data. So it cannot rule out a branch
from elsewhere in the function into bytes 1 to 13 of the jump. It also writes
the jump while the other CPUs are stopped, but cannot rule out a thread that
was preempted inside those bytes. Routes are installed early in boot, before
such threads exist, and the functions a client routes are checked by booting
the core probe ([tests/kexts/](../tests/kexts/README.md)).

**On arm64e**, kernel text and AMFI's MAC policy table, `mac_policy_ops`, are
read-only by the time Graft starts. Graft routes by shadowing AMFI's policy
instead of patching code. Each route names a MAC policy operation, such as
`vnode_check_signature`, and the replacement and original have that
operation's `mpo_*_t` prototype from XNU's `security/mac_policy.h`. Three
operations are routable: `file_check_mmap`, `file_check_library_validation`
and `vnode_check_signature`. To install the routes, Graft:

1. finds `mac_policy_list` through `mac_policy_register`, and the one element
   of it that holds AMFI's policy. It does this the first time any client
   routes; after that the element holds a shadow, so Graft reuses the element it
   remembered and the conf and ops it currently points at;
2. copies the current `mac_policy_ops` and `mac_policy_conf` into fresh memory,
   signs each routed operation's slot in the copy for the replacement, and
   points the copied conf at the copied ops. The current policy is AMFI's own
   before any client routes, and the newest shadow afterwards, so each client's
   copy carries the operations earlier clients already routed;
3. stores each callable original, then swaps the list element from the conf it
   currently holds to the copy with one atomic compare-and-exchange, which
   succeeds only if the element still holds that conf.

A MAC check running at the same moment sees one whole policy or the other, so
the arm64e routes are safe to install after BSD has started. Several clients
route one after another, each building on the last; a superseded copy stays in
memory for the boot, since a MAC check may still be reading it. Requests are
serialized. Graft refuses the routes when:

- an operation is not one of the three, is named twice, or is already routed by
  an earlier client. One operation belongs to one client for the boot, so every
  callable original Graft hands back is AMFI's own function;
- the slot in the current table does not point into AMFI's code, or is not
  signed with the discriminator Graft derives from the operation's prototype, as
  on a release that changed that prototype;
- a replacement lies outside the client's own executable segments, or the
  firmware names no placed image for the client. This check is required, not
  skipped. Graft finds the image as it finds the consumer's, through the
  device-tree node the boot firmware publishes for the bundle identifier the
  call passes;
- the list element is not writable, or no longer holds the current conf.

Graft reads `mac_policy_list` only once it finds the global in one of the
kernel's writable data segments, and reads each conf and ops table it follows
only inside AMFI's image. The list's array of entries is heap memory the kernel
allocated, which no image bounds, so Graft checks only that the pointer to it
is a canonical, 8-byte aligned kernel address. A wrong pointer there would
fault rather than refuse.

When Graft refuses, the client's state reads `refused`, naming the route and
the reason. A call Graft rejects before trying any route, for having no
requests, more than sixteen, or arriving while Graft is not active, returns
false and records nothing.

## Binding a consumer

A bound consumer is injected dormant. Its Info.plist keeps its personalities,
each gated by an `IOPropertyMatch`, and has no dependency on the
System-collection libraries, so the kernel registers the consumer but does not
construct it until Graft opens the gate. Its build, through
[sdk/](../sdk/README.md), records what it needs in a `__DATA,__graft` section,
the manifest.

Graft binds each consumer it found at start on its own. On x86-64, it tries to
bind each one at each catalogue send until it succeeds or refuses. On arm64e,
it tries once, at start. One consumer waiting or refused leaves the others as
they are. Each attempt runs these steps:

1. **Check the consumer is dormant.** A consumer that is already running is
   refused, since Graft binds only a dormant one.
2. **Read the manifest** and validate every table in it. Then parse its
   provider matching with `OSUnserializeXML`: one dictionary per personality,
   which must name an `IOProviderClass`, that matches the personality's
   provider ([manifest.md](manifest.md#provider-matching)). The consumer's
   personalities stay in its Info.plist, gated at build time.
3. **Open the providers.** On x86-64, each library the consumer imports from
   must be registered with the kernel; until then Graft waits. Graft then asks
   the kernel to load and start it (`OSKextLoadKextWithIdentifier`), and on
   x86-64 waits until it reports started. Graft finds the library's fileset
   entry, places the entry's symbol table from a class's live metaclass and
   that class's `gMetaClass` symbol, and requires the metaclass to lie inside
   the entry.
4. **Measure the classes** the consumer derives from: each class's size from
   its `OSMetaClass`, and its vtable's length and slots from the loaded
   library. Each must have exactly the size and vtable length the consumer was
   compiled against, and each method the consumer relies on must be in the
   slot it expects, compared by name and address.
5. **Plan every import site.** Each import's candidate names are tried in
   order in its provider. A miss falls back to the consumer's own fallback, if
   it declared one, and a weak import that nothing defines binds to the
   pointer the kernel's `gOSKextUnresolved` holds, as the kernel's linker
   would. Each site's new value is computed and checked against the bytes the
   build left there.
6. **Check every target.** Each site must now point into its provider (into
   code, for a call), at its own fallback, or, for an absent weak import, at
   exactly the kernel's unresolved-symbol pointer.
7. **Plan the vtables.** Each inherited slot is filled from the parent's live
   vtable. The consumer's own overrides stay as they are. On arm64e, a class
   whose parent has no Apple header has its whole vtable built here instead
   ([manifest.md](manifest.md#synthesized-classes)).
8. **Write everything in one transaction.** Before the first store, Graft
   checks that every target lies in the consumer, outside its manifest
   section, and, on arm64e, in a writable segment. Every site and slot is then
   written and read back together. If anything fails, Graft restores what it
   wrote.
9. **Open the activation gate.** Graft watches for each personality's
   providers by its provider matching, with one `IOFirstPublish` matching
   notification per personality. On every matching nub, present now or
   published later, it sets the gate property and re-registers the nub
   (`setProperty`, `registerService`), unless another of the consumer's
   personalities already did. The personalities' passive `IOPropertyMatch`
   then matches, and the kernel constructs and starts the consumer. These calls
   exist on every release, unlike the catalogue personality injection macOS 27
   removed. A passive property holds the consumer unconstructed until the
   bind, which an `IOResourceMatch` could not: IOKit consults that only before
   `start()`, after the constructors have run.

Steps 1 to 7 write nothing to the consumer or the kernel, though step 3 may
load the consumer's libraries. A refusal at any of them leaves the consumer
dormant and names the reason: the provider matching that is unusable, the
missing symbol, the provider that is not there, or the class whose layout
differs. A manifest of a version Graft does not read, as a consumer built with
another release's SDK carries, is reported with its version and a request to
rebuild the consumer with this release's SDK. Any other manifest that fails
validation is reported only as `invalid consumer manifest`, without the rule
it broke.
Graft never binds part of a consumer, and it never panics the guest over one.

Once the gate is open, the consumer can start at any moment. Graft therefore
keeps the providers loaded and its writes in place for the rest of the boot.

## Start and binding on arm64e

On arm64e the boot firmware does part of what OpenCore and the kernel do on
x86-64. Before the kernel starts, it places Graft and the kexts Graft serves
in the Auxiliary collection, binds their undefined symbols to kernel exports,
and publishes in the device tree the addresses Graft needs
([src/README.md](../src/README.md#what-the-arm64-boot-firmware-provides)).
Graft was developed with Ambrosia, its authors' own arm64 boot firmware.

Graft's start differs in three places:

- It reads the Boot collection's address from the device tree.
- It reads `_IOLog` as a data symbol, so that taking its address involves no
  pointer authentication.
- It installs no binding trigger. Every image is in place before Graft starts,
  so the first attempt at start is the only one, and a consumer left
  `waiting` stays so for the boot.

Binding follows the same steps, with these differences:

- A provider is its fileset entry in the Boot collection. Graft asks the
  kernel to start it, as on x86-64, but does not wait: the Boot collection's
  kexts are registered before Graft starts, and a start that fails refuses
  the consumer. A placed kext is not started until a driver matches in it or
  a started kext depends on it, and its classes register only as it starts, so
  this start is what registers the classes of a family nothing else on the
  machine uses.
- Graft finds the consumers among the children of `/chosen/ambrosia`, the
  device-tree nodes the firmware publishes for the kexts it places.
- The firmware places each segment of the consumer separately, by name, so
  Graft maps every offset in the manifest through the consumer's segment
  table.
- Each import site holds a pointer the loader signed for its placeholder.
  Graft authenticates it and signs the real target with the same key and
  discriminator. Each inherited vtable slot is signed again for its own
  address, with the signing context the consumer's build recorded.
- Every target must lie in a writable segment the kernel has not locked
  read-only, since Graft writes after the kernel's lockdown with no window to
  open. The SDK links an arm64e consumer with `-no_data_const` for this.

## Deciding from what is present

No part of Graft checks a macOS version. What decides is whether a symbol
exists, a class's size, the methods in a vtable, and the answers IOKit gives.
A consumer that needs something only some releases have says so in its
manifest: a list of candidate names, a fallback in its own code, or a weak
import it tests with `OSKextSymbolIsResolved()`.

## What it reports

Graft writes each step with `IOLog`, prefixed `graft: `, and publishes its
state on the IORegistry root, where `ioreg -l` shows it. The unified log keeps
the `IOLog` lines on arm64e only: on x86-64 the kernel keeps every line a kext
OpenCore injected writes out of it
([guides/debugging-refusals.md](guides/debugging-refusals.md#where-to-look)),
so `GraftLog` is the record of Graft's steps on both:

| property | holds |
|---|---|
| `GraftState` | `starting`, `active` or `refused` |
| `GraftReason` | why Graft is in that state |
| `GraftLog` | the most recent 8 KiB of Graft's log |
| `GraftClient.<bundle id>` | one per served kext, each bound consumer under its own bundle identifier, as the string `<state>: <reason>` |

A bound consumer's state is `waiting`, `bound` or `refused`. A bound
consumer's reason also says whether the gate is open yet, naming the
personality whose provider opened it, or still waiting for a provider.
An owned client's state is `routed` or `refused` after routing,
`refused` after a failed symbol lookup, or whatever it last set with
`graft_client_state`. A placed consumer Graft did not find at start has no
entry; the log names each one it found.
[guides/debugging-refusals.md](guides/debugging-refusals.md) lists every
reason.

## The architecture boundary

Most of the code is architecture-neutral. What differs has a file in each of
`src/x86_64/` and `src/arm64e/`: how the Boot collection is found, how kernel
memory is written and which targets may be written, how a kernel function is
routed, how the binding finds its images, and how a site's new value is built
([src/README.md](../src/README.md#the-architecture-boundary)).

## What is tested where

The host tests, run with `make test` on a Mac with Apple silicon, cover the
logic that needs no kernel: Mach-O reading, symbol lookup, the manifest
format, import resolution, site and target checks, vtable planning, write
transactions, route building for both architectures, and the arm64e synthesis
plan and discriminator tables. The arm64e signing tests run as native arm64e
processes. `make kernel-check` compiles a test kext carrying a manifest
(`tests/kernel_fixture.cpp`) with kext flags against Kernel.framework, and
links it. `make kernel-check-arm64e` compiles the neutral core and the arm64e
layer for arm64e. [tests/README.md](../tests/README.md) lists the tests.

Only a guest boot confirms the rest: that the routes install early enough,
that the writes land, and that a real consumer binds and runs.
`tools/internal/boot-core.py` boots Graft and the test kexts in a disposable
x86-64 guest ([tools/README.md](../tools/README.md)).
