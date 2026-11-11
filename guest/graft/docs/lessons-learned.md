# Lessons learned

These are the mistakes and surprises met while Graft was built, and what Graft
does because of them. Most of them apply to any kext that binds against or
patches the running kernel, and to any tool that does the job Lilu does for
OpenCore setups.

Each lesson says what happened, why it happened, and what to do instead. Where
Graft's code applies the lesson, the lesson names the file.

## Binding against the running kernel

### Decide from what is present, never from a version number

The framework Graft replaces gated each plugin by a minimum and maximum Darwin
version, and the boot loader's configuration added kernel ranges of its own.
The loader that bound a display driver before Graft also checked each parent
class against a layout table taken from one release. Every new macOS release
was refused until someone widened a range or added a table.

A version number says nothing about the symbol, class or method a kext
depends on. The check passes or fails for reasons unrelated to what the kext
needs.

Check the thing itself. Graft checks no macOS version anywhere. It decides
from whether a symbol exists, a class's size from its `OSMetaClass`, the
method in a vtable slot, and the answers IOKit gives. A consumer copes with a
symbol that only some releases have through candidate names tried in order, a
fallback in its own code, or a weak import it tests with
`OSKextSymbolIsResolved()` ([manifest.md](manifest.md#resolving-an-import)).
One binary per architecture then covers every release, and a new release
needs a change only when something the consumer uses has changed.

### Keep a consumer's offsets in the consumer

That loader carried a table of every import site in one build of the display
driver. Any change to the driver meant rebuilding the loader too, and a loader
built against one build of the driver could not bind the next.

The binder held facts that belong to the consumer's build.

Have the consumer's build write what the binder needs into the consumer. A
bound consumer carries its manifest in its own `__DATA,__graft` section
([manifest.md](manifest.md)), and Graft holds no consumer's offsets, so
changing a consumer rebuilds only the consumer.

### Refuse an ambiguous name instead of taking the first match

A library's symbol table can hold several local definitions of one name, such
as copies of one inline method. A lookup that returns the first match picks one
copy without saying so, and it may not be the copy a vtable points at.

Local symbols need not be unique within an image. The linker can keep more
than one copy of a method under one name.

Treat repeated entries at one address as one definition and entries at
different addresses as ambiguous, and refuse an ambiguous name. Graft's lookup
returns every address a name has (`src/MachO.h`), and resolution refuses a
name with more than one (`src/Resolve.h`).

### Reserved slot names in the headers prove nothing

When a consumer named no `required` methods, the SDK verified every parent
slot by name. That refused arm64e consumers of IOKit classes.

The Kernel.framework headers keep `_RESERVED<class><n>` placeholder names in
slots that a running kernel may have given a real method. On arm64 one of
`OSMetaClassBase`'s reserved slots holds a real method, whose name never
matches the placeholder.

Verify only the slots whose names mean something, and verify those by address.
For each method a consumer relies on, Graft looks the method up by name in its
provider and requires the live slot to hold that address (`src/Layout.h`). The
SDK leaves the reserved placeholders out of that check by default. Graft
still fills those slots from the live parent. A consumer can also list in
`required` only the inherited methods it calls.

### Measure a layout for the architecture that uses it

The SDK's layout probe compiled for x86-64 whatever the consumer's target was.
An arm64e consumer then recorded the x86-64 parent slot count, and Graft
refused its bind because the live class had more slots.

The Kernel.framework headers give `OSMetaClassBase` extra reserved slots on
arm64 (under `APPLE_KEXT_VTABLE_PADDING`), and slots declared with
`OSMetaClassDeclareReservedUsedX86` mean different things on the two
architectures. One class has a different vtable on each.

Measure with the target and flags of the code that relies on the layout. The
SDK's probe compiles for the consumer's own architecture, with the consumer's
own compiler flags, and reads each slot through that architecture's
relocations or chained fixups.

### Bind an absent weak import to the value `gOSKextUnresolved` holds

libkern declares `gOSKextUnresolved` as a pointer, and
`OSKextSymbolIsResolved()` compares a weak symbol's address with the value that
pointer holds. An early version of Graft bound an absent weak import to the
variable's own address instead. The consumer's own check then read the import as
present, took the wrong code path, and WindowServer crashed in a loop.

The kernel's linker binds an unresolved weak reference to the stored value,
not to the variable's address, and `OSKextSymbolIsResolved()` is written for
that.

Do what the kernel's linker does. Graft reads the stored pointer and binds an
absent weak import to that value (`src/Resolve.h`, `unresolvedValue`). A
strong import that nothing defines still refuses the consumer.

### A provider you only import from needs an anchor class

Graft confirms a provider's image before trusting its symbols: it places the
image's symbol table from a class's live metaclass, and requires the
metaclass to lie inside the image. A consumer that took only a weak import
from a family, without subclassing any of its classes, gave Graft no class to
check, and Graft refused the provider.

The check needs a class the provider defines, and a consumer's parent classes
were the only source of one.

Let the consumer name an anchor. A library record can name a class the
provider defines, which Graft uses to confirm the image with no layout probe
and so no header ([manifest.md](manifest.md#tables)). The SDK fails the build
for a provider with neither a declared class nor an anchor, rather than leave
Graft to refuse it at boot.

## Starting and activating

### Start from your own kmod start routine

The framework Graft replaces started its plugins from the boot console, on the
first screen-enable event. A guest that boots on a paravirtual GPU with no boot
framebuffer never sends that event, so no plugin started. The display
driver's loader worked around this by sending the event itself once BSD was
up.

The start depended on an event that only some boots produce.

Take control in your own kmod start routine and do all of the setup there.
Graft starts in `graft_start` (`src/Start.cpp`) and waits for no boot console,
framebuffer or BSD.

### Gate a dormant consumer with a passive match, not a resource

Graft first gated a dormant consumer's personalities with an
`IOResourceMatch`, published after the bind. The consumer crashed before the
gate was ever consulted.

IOKit consults an `IOResourceMatch` when it decides whether to start a
driver. By then `IOService::probeCandidates` has already allocated the class
and run its constructors, which called through import slots Graft had not yet
bound.

Hold the consumer back at a stage before it is constructed. A bound
consumer's personalities carry a passive `IOPropertyMatch` on the provider nub,
which IOKit evaluates before it allocates anything. After the bind, Graft
sets the property on each nub a personality matches and re-registers it, so
matching runs only once the imports and vtables are in place. It watches for
the nubs with a matching notification rather than looking them up once, so a
nub published after the bind opens the gate too (`src/KernelBinding.cpp`,
`openGate`).

### Do not rely on IOCatalogue personality injection

The first activation design held a consumer's personalities in its manifest
and published them with `IOCatalogue::addDrivers` after the bind. Graft's
first load on macOS 27 stopped at the missing symbol.

macOS 27 removed the IOCatalogue calls that add personalities, `addDrivers`
among them, though macOS 26 still has them.

Activate through calls every release has. The passive gate above needs only
`serviceMatching`, `addMatchingNotification`, `setProperty` and
`registerService`, so one binary loads and activates on every release it was
tested on.

### A provider family may be placed but never started

On the arm64 paravirtual machine, the display consumer's provider family is in
the kernel collection but is never started: nothing there matches it or
depends on it. Its classes' metaclasses were never constructed, so the class
lookup returned nothing, and Graft refused the consumer at its anchor check.

A kext that is placed in a collection is not started until a driver matches in
it or a started kext depends on it.

Start the provider before binding against it. Graft asks the kernel to start
each library a consumer's manifest names (`OSKextLoadKextWithIdentifier`), on
arm64e as on x86-64, so Graft itself depends on no consumer's family
(`src/arm64e/BindingPlacement.cpp`, `locate`).

## Reading kernel collections

### Collection symbol tables share one string table and hold unslid values

A prelinked kernel collection does not give each kext its own symbol table.
Each fileset entry's `LC_SYMTAB` is a slice of one shared `nlist` array that
points into one string table holding the names of every symbol in the
collection. Each entry's symbol values are unslid collection addresses.

This bit twice. A snapshot that copied an image's whole string table copied
every name in the collection, once per image, and kept each copy for the
boot. And an unslid value cannot be used as an address.

Copy only the names of the image's own defined symbols, the only names a
lookup can find (`src/MachO.h`, `symbolSnapshotSize`). Calibrate each image
against a symbol whose real address you know another way: Graft calibrates the
kernel against its own `IOLog` import (`openKernel`), and a collection entry
that registers no class against the slide another entry of the same
collection already measured (`RuntimeImage::openEntryDefining`).

## Pointer authentication on arm64e

### Segments are placed by name, and signatures are bound to addresses

On x86-64 an injected kext's segments are contiguous, so an offset in the
image is its load address plus the offset, and Graft copies inherited vtable
slots verbatim. The first arm64e bind of a real consumer read its sites at the
wrong addresses and left inherited slots the consumer could not authenticate.

On arm64e the boot firmware coalesces each kext's segments by name into shared
collection segments, so a segment is not at its origin plus one slide. And a
provider's vtable slots are signed for the provider's slot addresses, so a
copy is invalid anywhere else.

Map every offset through the placed segment of the same name, and re-sign
every copied pointer for its new address. An arm64e manifest carries a segment
table, and Graft maps each consumer offset through it
(`src/KernelBinding.cpp`, `mapConsumerOffset`). Each inherited slot is
stripped and re-signed for the consumer slot's own address, from the signing
context the consumer's build recorded ([manifest.md](manifest.md#tables),
`src/arm64e/Fixup.h`).

### Read a function's address as data when you do not want it authenticated

Graft calibrates the kernel's symbol table by comparing the table's value for
`_IOLog` with the address the kernel's linker gave Graft's own `IOLog`
import. Taking `&IOLog` as a function pointer panicked the guest in Graft's
own start.

The compiler materialized `&IOLog` as a signed function pointer and
authenticated it. The loader had filled that slot with IOLog's plain address,
so the authentication failed, and on a CPU with FEAT_FPAC a failed
authentication faults.

Read the symbol as data. `src/arm64e/Kernel.cpp` declares `_IOLog` as a data
symbol, so taking its address is a plain load with no authentication, and it
yields the raw address the calibration needs.

### A read-only policy table can be shadowed instead of patched

On x86-64 Graft routes a function by patching its entry in kernel text. The
first arm64 plan did the same in data: replace a function pointer in AMFI's
`mac_policy_ops` in place.

AMFI's ops table is in `__DATA_CONST`, which the kernel makes read-only at
lockdown, before Graft starts. A store into it fails like a store into
kernel text.

Copy the table into writable memory, change the copy, and repoint the one
writable reference to it. Graft copies AMFI's `mac_policy_ops` and
`mac_policy_conf`, re-signs the routed ops in the copy for the client's
replacements, points the copied conf at the copied ops, and repoints the
`mac_policy_list` element that held AMFI's conf at the copy
(`src/arm64e/Route.cpp`, `src/arm64e/Arm64Route.h`). The one live write is a
compare-and-exchange of one aligned pointer, which refuses if the element no
longer holds AMFI's conf. AMFI signs its op pointers without address
diversity, so the slots Graft leaves alone stay valid at the copy's address.

### Install x86-64 routes before BSD; an atomic swap needs no such window

An x86-64 route overwrites a function's first instructions with a jump. A CPU
could enter the function while its entry is half written, so the x86-64
installer refuses once BSD is published, and writes with the other CPUs
stopped (`src/x86_64/Route.cpp`). The same rule, carried over to arm64,
refused every arm64 client.

An owned client that matches as an IOKit driver starts after BSD. On arm64 the
route goes live by one atomic compare-and-exchange between two complete, valid
policy configurations, so a concurrent caller sees one or the other, never a
torn value.

Gate on the hazard, not on the habit. The arm64e installer has no BSD gate,
and `graft/Graft.h` documents the window as an x86-64 rule.

### Sign a vtable slot with its method's first declaration's discriminator

On arm64e Graft builds a synthesized class's vtable and signs each slot with a
per-method discriminator. Graft first took each slot's discriminator from the
parent class's own vtable. Once WindowServer called one of those methods, the
kernel took an instruction-fetch abort.

A caller authenticates a slot with the discriminator of the method's first
declaration, in the most-base class that has the slot. A class that redeclares
an inherited pure method signs that slot with its own hash, which no caller
uses.

Take each slot's discriminator from the most-base class that declares it.
Graft reads the tables of the parent and of each ancestor its library
defines, checks each against its live vtable, and gives every slot the
discriminator of the most-base table that covers it (`src/arm64e/VtableDisc.h`,
`chainDiv`).

### A qualified virtual call under `-fapple-kext` uses the compiled slot

A consumer that subclasses a class with no Apple header compiles against a
stand-in declaration of it. Forwarding to the parent's method with a qualified
call such as `IOMobileFramebufferLegacy::start_hardware()` faulted.

Under `-fapple-kext` a qualified virtual call loads the method from the named
class's vtable, at the slot the compiler assigned from the stand-in. The live
class orders its vtable differently, so the call reached another method.

Call none of the stand-in's virtual methods except those in a part of the
vtable that every release keeps in the SDK's order. The stand-in's comment
lists them: `start`, `stop` and `free`, which lie in `IOService`'s part
(`sdk/include/graft/IOMobileFramebufferLegacy.h`).

### A vtable's start is not its address point

A C++ vtable begins with an offset-to-top word and a typeinfo word, and an
object's vtable pointer points past them, at the address point. The SDK
recorded a synthesized class's vtable offset as the address point, while the
manifest format and Graft take it as the vtable's first word. Every
synthesized word landed two slots late, and the first virtual call on a new
object went to the wrong method.

The two conventions both appear in the format: an ordinary vtable record names
the address point, and a synthesized class names the first word.

Check the convention where the data is used. Graft refuses the bind by name when
a synthesized vtable's first two words are not the zero offset-to-top and
typeinfo it expects (`src/KernelBinding.cpp`, `synthesizeClass`), so a wrong
offset cannot reach a constructed object.

## Releasing

### Build the release from its commit, with a pinned Xcode

A binary and the sources it claims to come from can drift apart without anyone
noticing.

A rebuild with another Xcode, another environment or other files in the tree
gives different bytes, and nothing ties the binary to a commit.

Build each release from a commit, and pin what changes its bytes.
`make release` refuses a tree that differs from `HEAD` and any Xcode but the
one `.xcode-version` names. It builds from a scratch copy of the tracked files
with build-affecting environment variables removed, and writes the commit and
every file's SHA-256 beside the kexts and their source. Its zip is stored, in name order, with fixed
modes and the commit's time, so the same commit and Xcode give the same bytes.
CI publishes the release it builds from the tag, and anyone with that Xcode can
rebuild it from the tag and compare
([guides/releasing.md](guides/releasing.md)).
