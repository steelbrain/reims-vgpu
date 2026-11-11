# The consumer manifest

A bound consumer tells Graft what it needs in a manifest: a Mach-O section,
`__DATA,__graft`, that its build writes with the SDK's `graft-kext` command.
Graft holds no consumer's offsets or personalities, so changing a consumer
only rebuilds the consumer.

This document specifies format version 4, for an x86-64 consumer, and version 5,
for an arm64e consumer. The two share the header and every table. They differ in
a site's two trailing bytes: an x86-64 site is a pointer or a branch
displacement, an arm64e site a signed pointer that carries its signing context.
Only version 5 fills the segment, fill and synthesis tables; version 4 leaves
them empty. Graft reads no other version, and refuses a consumer whose manifest
has another by naming that version, since its fix is a rebuild with the SDK of
the Graft release it boots with. The definitions, and the validator below, are in
[sdk/include/graft/ManifestFormat.h](../sdk/include/graft/ManifestFormat.h),
which Graft and the SDK's `graft-kext` share.

## Encoding

The manifest is a header followed by tables of fixed-width records. All
fields are little-endian, and there are no pointers and no relocations. A
table is an `offset` and a `count`, where the offset counts bytes from the
start of the section. Strings are offsets into the string table. Image
offsets, used by sites, placeholders, fallbacks, vtables and synthesized
classes and overrides, count bytes from the consumer's load address.

Graft copies each record out before reading it, so the section need not be
aligned.

The section holds more than the manifest. After its `size` bytes, the SDK gives
each import a placeholder: a 32-byte slot of its own in the section. The build
points every site at its import's placeholder, so the linker that places the
consumer at boot, OpenCore's on x86-64, resolves the site within the consumer
and leaves a known value there. Before writing a site, Graft checks that it
still points at its placeholder plus its addend.

## Header

136 bytes.

| field | type | value |
|---|---|---|
| `magic` | u32 | `0x54465247`, "GRFT" |
| `version` | u32 | 4 for an x86-64 consumer, 5 for arm64e |
| `size` | u32 | the manifest's length in bytes |
| `headerSize` | u32 | 136 |
| `strings`, `libraries`, `imports`, `candidates`, `sites`, `classes`, `slots`, `vtables`, `overrides`, `segments`, `fills`, `selectors`, `synthOverrides`, `synthClasses` | table each | the tables below, in this order |
| `gate` | u32 | string: the consumer's activation property key, an `IOPropertyMatch` its personalities carry, which Graft sets on each provider nub once it binds |
| `matching` | u32 | string: the personalities' provider matching, an XML property list ([Provider matching](#provider-matching)), by which Graft finds the nubs to open the gate on |

## Tables

**strings.** Its count is a length in bytes. The first byte is 0, so
offset 0 means "no string". Every string ends with a NUL inside the table.

**libraries**, 8 bytes each. One per System-collection library the consumer
imports from.

| field | type | meaning |
|---|---|---|
| `identifier` | u32 | string: the library's bundle identifier |
| `anchor` | u32 | string: a class Graft uses to confirm the library's image, or 0 |

Graft confirms each library's image before using its symbols. It places
the image's symbol table from a class's live metaclass and that class's
`gMetaClass` symbol, and requires the metaclass to lie inside the image. The
class is the anchor if there is one, and otherwise the first class the
consumer declares from that library. A library with neither is
refused, and the SDK fails the build for one. An anchor serves a library the
consumer only imports from, such as one it imports weakly.

**imports**, 24 bytes each. One per symbol the consumer imports.

| field | type | meaning |
|---|---|---|
| `library` | u32 | index into libraries |
| `firstCandidate` | u32 | index of its first name in candidates |
| `candidateCount` | u32 | how many names, at least 1 |
| `flags` | u32 | bit 0 set: the consumer references it weakly; other bits are 0 |
| `fallback` | u64 | image offset of the consumer's own replacement, or `0xffffffffffffffff` for none |

**candidates**, 4 bytes each: string offsets naming an import's symbols, in
the order to try them.

**sites**, 32 bytes each. One per place in the consumer that refers to an
import.

| field | type | meaning |
|---|---|---|
| `offset` | u64 | image offset of the bytes to write |
| `placeholder` | u64 | image offset of the local placeholder the build pointed the site at |
| `addend` | i64 | the addend of the original reference |
| `import` | u32 | index into imports |
| `width` | u8 | 8 for a pointer, 4 for a call's displacement |
| `opcode` | u8 | version 4: 0 for a pointer, `0xe8` (call) or `0xe9` (jump) for a call. version 5: the auth byte |
| `reserved` | u16 | version 4: 0. version 5: the diversity |

The last two bytes differ by version. A version-4 (x86-64) site is a pointer
(`width` 8, `opcode` 0) or a branch displacement (`width` 4, `opcode` `0xe8` or
`0xe9`), and `reserved` is 0. A version-5 (arm64e) site is always a pointer
(`width` 8). Its `opcode` is an auth byte, and its `reserved` is the 16-bit
diversity. In the auth byte, bit 0 is set when the pointer is signed, bits 1-2
are the key (0=IA, 1=IB, 2=DA, 3=DB), and bit 3 blends the slot's address into
the discriminator. A plain (unsigned) arm64e pointer has both bytes 0. Graft
signs its replacement pointer with this context before writing the slot.

**classes**, 28 bytes each. One per System-collection class the consumer
derives from.

| field | type | meaning |
|---|---|---|
| `name` | u32 | string: the class name |
| `vtable` | u32 | string: the class's vtable symbol |
| `library` | u32 | index into libraries |
| `instanceSize` | u32 | the class size the consumer was compiled against |
| `parentSlots` | u32 | the number of vtable slots the consumer was compiled against |
| `firstSlot` | u32 | index of its first entry in slots |
| `slotCount` | u32 | how many slot entries |

**slots**, 8 bytes each: a vtable slot `index` (u32) and the string naming
the `method` (u32) that must be there.

**vtables**, 32 bytes each. One per consumer class whose vtable inherits
from a System-collection class. A C++ vtable starts with 16 bytes of offset
and type information, and its address point, where an object's vtable
pointer points, is the first slot after them.

| field | type | meaning |
|---|---|---|
| `offset` | u64 | image offset of the vtable's address point |
| `classIndex` | u32 | index into classes: the parent |
| `slotCount` | u32 | the vtable's full slot count |
| `firstOverride` | u32 | index of its first entry in overrides |
| `overrideCount` | u32 | how many overrides |
| `firstFill` | u32 | index of its first entry in fills |
| `fillCount` | u32 | how many fills |

**overrides**, 4 bytes each: parent slot indices where the consumer's own
method stays.

**segments**, 24 bytes each. The consumer's own segments, present only on a
version-5 (arm64e) manifest. Graft relocates each image offset through the
placed segment of the same name, because the arm64e loader coalesces segments
by name and a segment is not placed at its origin plus a single slide. A
version-4 manifest has none: its segments are contiguous, so an offset is the
load address plus the offset.

| field | type | meaning |
|---|---|---|
| `origin` | u64 | the segment's first byte as an image offset (its link vmaddr minus the image base) |
| `size` | u64 | the segment's vmsize |
| `name` | u32 | string: the segment name |
| `reserved` | u32 | 0 |

**fills**, 8 bytes each. The inherited vtable slots Graft re-signs, present
only on a version-5 (arm64e) manifest, one per parent slot a vtable does not
override, in slot order. Each carries the slot's own signing context, so Graft
signs the live parent method for this slot exactly as the consumer
authenticates it. A version-4 manifest has none: it copies each inherited slot
verbatim.

| field | type | meaning |
|---|---|---|
| `slot` | u32 | the parent slot index this fills |
| `diversity` | u16 | the slot's constant discriminator |
| `opcode` | u8 | the auth byte, as a site's (bit 0 signed, bits 1-2 key, bit 3 address diversity) |
| `reserved` | u8 | 0 |

The last three tables describe synthesized classes, and are present only on a
version-5 (arm64e) manifest. A synthesized class derives from a parent that no
Apple SDK header declares and whose vtable changes between releases, so the
consumer cannot record the parent's slots. Its overrides name methods instead of
slot indices, and Graft lays out the vtable when it binds
([Synthesized classes](#synthesized-classes) below).

**selectors**, 4 bytes each: string offsets naming the methods of the
synthesized overrides, as candidates name an import's symbols. Each override
owns a run of them, in order. The first is the consumer's own mangled symbol
for the method, and any others are alternatives, for a method whose parameters
differ between releases. Graft takes from a selector only the method it names,
the mangled name without its class name, so the class a selector is written
under does not matter. It skips a selector that is not a member function's
nested name with a plain class name.

**synthOverrides**, 24 bytes each. One per method a synthesized class
overrides, grouped by class in class order.

| field | type | meaning |
|---|---|---|
| `impl` | u64 | image offset of the consumer's implementation |
| `firstSelector` | u32 | index of its first entry in selectors |
| `selectorCount` | u32 | how many selectors, at least 1 |
| `diversity` | u16 | the constant discriminator the consumer's build signed this method's slot with |
| `opcode` | u8 | the auth byte, as a site's |
| `flags` | u8 | bit 0 set: the override is required; other bits are 0 |
| `reserved` | u32 | 0 |

The signing context is the one the consumer's build gave the method. Graft uses
`diversity` only to place an override that no selector places, and refuses the
consumer if `opcode` is not that of a vtable slot: signed, key IA, with the
slot's address blended in. It signs the slot itself with the discriminator the
running system's callers use (below). A required override must land, or Graft
refuses the consumer. One that is not required is left out when Graft finds no
live slot for it.

**synthClasses**, 40 bytes each. One per synthesized consumer class.

| field | type | meaning |
|---|---|---|
| `vtableOffset` | u64 | image offset of the consumer vtable's first word |
| `name` | u32 | string: the consumer class's name, which Graft's refusals name |
| `parentClass` | u32 | string: the parent class, which must also be in classes |
| `siblingClass` | u32 | string: a concrete subclass of the parent, which implements the parent's pure methods, or 0 |
| `library` | u32 | index into libraries: where the parent is |
| `vtableSlotCap` | u32 | the consumer vtable's length in words as compiled, counting its two leading words |
| `parentRegion` | u32 | the largest live parent instance the consumer has room for |
| `firstOverride` | u32 | index of its first entry in synthOverrides |
| `overrideCount` | u32 | how many overrides, at least 1 |

`vtableOffset` is not an address point. A vtables entry's `offset` is the
address point, 16 bytes into the vtable, but `vtableOffset` is the vtable's
start, its offset-to-top word, because Graft writes the whole vtable from there.

The SDK sets `vtableSlotCap` to the length of the consumer class's compiled
vtable, which the stand-in parent's reserved slots lengthen, and
`parentRegion` to the stand-in's size, which reserves room for the live
parent's fields.

## Validation

Graft validates the whole manifest before using any of it, and refuses the
consumer on the first violation. `graft-kext` applies the same rules to each
manifest it builds, and refuses a build whose manifest breaks one:

- The header has the right magic, header size and version, and its size
  fits the section.
- Every table lies inside the manifest, after the header. No two non-empty
  tables overlap.
- The string table is not empty and starts with a 0 byte. Every string
  offset names a non-empty, NUL-terminated string inside it, and the
  `gate` and `matching` strings exist. The `gate` key is at most 159 bytes
  long, so that Graft keeps it whole for the nubs published after the bind.
- Every index fits its table, and every range of candidates, slots,
  overrides, fills, selectors or synthesized overrides fits too.
- An import has at least one candidate and no flag but the weak bit.
- A version-4 site's `reserved` field is 0, and its width is 8 with opcode 0
  or 4 with opcode `0xe8` or `0xe9` and an offset above 0. A version-5 site's
  width is 8, its opcode sets no bit outside the auth byte, and an unsigned one
  carries no auth bits and no diversity.
- Sites are sorted by offset and do not overlap. For a call, the overlap
  check includes the opcode byte before its displacement.
- A class has a nonzero size and parent slot count. Its slot entries are
  sorted, lie within its parent slot count, and follow the previous class's
  entries with no gaps. Together the classes use every slot entry.
- Vtables are sorted by address point. An address point is 8-byte aligned
  and at least 16 bytes in, and its slot count is nonzero and covers its
  parent's, and its parent is not a class that a synthesized class names as
  its parent. Its overrides are sorted and within the parent's slots. A
  vtable, from 16 bytes before its address point, overlaps no site and not
  the vtable before it. Together the vtables use every override.
- The segment and fill tables are empty on a version-4 manifest. On a
  version-5 one, segments are sorted by origin, each named, nonzero in size,
  bounded and disjoint. A vtable's fills are the parent slots it does not
  override: their count is the parent slot count minus the overrides, and the
  fills and overrides together name each parent slot once, in slot order. A
  plain (unsigned) fill carries no auth bits and no diversity. Together the
  vtables use every fill.
- The selector, synthesized override and synthesized class tables are empty
  on a version-4 manifest. On a version-5 one, every selector names a string.
  A synthesized override has a nonzero `impl`, at least one selector, no flag
  but the required bit and a `reserved` of 0. Its opcode sets no bit outside
  the auth byte, and an unsigned one carries no auth bits and no diversity.
  Its selectors follow the previous override's with no gaps, and together the
  overrides use every selector.
- A synthesized class's `name` and `parentClass` name strings, as does its
  `siblingClass` unless it is 0, and its library fits. Its `vtableOffset` is
  8-byte aligned and at least 16, and its `vtableSlotCap` and `parentRegion`
  are nonzero. It has at least one override, following the previous class's
  with no gaps, and together the classes use every synthesized override.

Graft reports a manifest that breaks any of these rules only as `invalid
consumer manifest`, without naming the rule. It then checks the provider
matching ([below](#provider-matching)), and names what is wrong with it.

A valid manifest does not prove its image offsets are mapped. Graft checks
each site, placeholder, fallback, vtable and synthesized class against the
consumer's loaded segments separately. Before its first store, it also checks
every write it has staged, and refuses the consumer if one would land in the
manifest's own section, or, on arm64e, outside a writable data segment. A
segment such as `__DATA_CONST` is locked read-only before Graft starts, so the
SDK links an arm64e consumer with `-no_data_const`, which keeps its bound
pointers in `__DATA`.

## Provider matching

The `matching` string is an XML property list, as an Info.plist is written: a
dictionary with one entry per personality of the consumer, keyed by the
personality's name. Each value is the dictionary that matches the
personality's provider: the personality as its author wrote it, its own
`IOPropertyMatch` included, less the keys that name the driver rather than the
provider, `IOClass`, `CFBundleIdentifier`, `IOProbeScore` and
`IOMatchCategory`. It never holds the gate key, which only the staged
personalities carry: `graft-kext` refuses a personality whose own
`IOPropertyMatch` already holds it. For a personality written

```xml
<key>Display</key>
<dict>
    <key>IOClass</key>
    <string>ExampleDisplay</string>
    <key>IOProviderClass</key>
    <string>IOPCIDevice</string>
    <key>IOPCIPrimaryMatch</key>
    <string>0x11111234</string>
</dict>
```

the matching is

```xml
<dict>
    <key>Display</key>
    <dict>
        <key>IOPCIPrimaryMatch</key>
        <string>0x11111234</string>
        <key>IOProviderClass</key>
        <string>IOPCIDevice</string>
    </dict>
</dict>
```

inside the property list's `<plist>` element. Graft parses it with
`OSUnserializeXML` before it writes anything, and refuses the consumer when it
does not parse, names no personality, or holds a personality that is not a
dictionary naming an `IOProviderClass` string. Once it has bound the consumer,
it watches for the nubs each dictionary matches, with one `IOFirstPublish`
matching notification per personality, and sets the gate key on each.

## Resolving an import

Graft tries an import's candidates in order, in its library only. The first
name defined there wins. Several entries for one name at one address count
as one definition. Entries at different addresses are ambiguous, and an
ambiguous name refuses the consumer rather than moving on to the next.

If no candidate is defined:

- with a fallback, the import binds to that offset in the consumer;
- without one, a weak import binds to the pointer the kernel's
  `gOSKextUnresolved` holds, which is what `OSKextSymbolIsResolved()`
  compares against;
- otherwise the consumer is refused, with the import's names in the reason.

Candidates, fallbacks and weak imports are how a consumer copes with a
symbol that only some releases have, without a version check.

Graft resolves an import only to write its sites. An import with no site is
never resolved, so a release that does not define it refuses nothing. The SDK
records such imports for the methods a consumer's vtable inherits: the build
needs them to replace each inherited slot's reference with a placeholder, and
Graft fills those slots from the live parent instead.

## Classes and vtables

The compiler fixes a consumer's member offsets and virtual-call slots against
the Kernel.framework headers it was built with, and reading the live class
cannot repair them. So each class records the size and slot count the consumer
was compiled against, and both must match the live class exactly. Each slot
entry must then hold the named method in the live vtable, compared by name and
address. A class that a synthesized class names as its parent is the exception:
its size and slots are a stand-in's, so Graft measures the live class for the
synthesis and checks neither, and the SDK records no slots for it. Its
`parentSlots` is the number of vtable slots the stand-in reserves.

When the bind runs, each inherited slot of a consumer vtable is filled from
the parent's live vtable, except the slots listed as overrides, which keep
the consumer's own methods.

## Synthesized classes

For each synthesized class, Graft measures the live parent, the class in
classes that `parentClass` names, and refuses the consumer if there is none.
It also refuses it if the live parent's instance is larger than
`parentRegion`, or if the live parent's vtable has more slots than the
stand-in reserves, the `parentSlots` of the parent's entry in classes. The
consumer's own virtual methods follow those slots in its vtable, and a longer
parent would overwrite them.

Graft then finds each override's live slot. For each selector in turn, it builds
the method's symbol under the parent class, looks it up in the parent's library,
and finds the one slot of the live parent's vtable that holds it. When that
finds no slot and the class names a sibling, it does the same under the sibling
class, in the sibling's vtable. The sibling need not be in the parent's library:
Graft reads it from whichever entry of the boot collection defines its vtable,
which need not be loaded, and only to find slots, never to call it. When no
single entry defines that vtable, Graft goes on without the sibling. The first
selector that finds a slot wins, and a method found in two slots refuses the
consumer. An override that no selector places goes to the one live slot whose
discriminator in the parent's own vtable equals its `diversity`, if that is not
0, exactly one slot has it, and no symbol at that slot's method, in the
kernel's or a provider's symbol table, names another method than the
override's. A 16-bit discriminator can match an unrelated slot by chance, and
the name check catches that only when the unrelated method has a symbol: an
override can still land on the slot of a method no symbol table names. After
that, an override with no slot refuses the consumer if it is required, and two
overrides on one slot refuse it.

The discriminators come from the arm64 boot firmware, which reads them from the
boot collection on disk and hands them over in the device tree, one table per
vtable it exports
([src/README.md](../src/README.md#what-the-arm64-boot-firmware-provides)). Graft
needs the parent's table and the table of each of the parent's ancestors whose
vtable the parent's library defines, found by walking the live parent's
superclasses; for `IOMobileFramebufferLegacy` those are
`IOMobileFramebufferService` and `IOMobileFramebuffer`. It refuses the consumer
if one is missing, does not match its live vtable, or is longer than its
subclass's.

A caller authenticates a slot with the discriminator of the method's first
declaration, which the most-base of those tables that covers the slot
carries. The parent's own table does not always: `IOMobileFramebufferLegacy`
redeclares some of `IOMobileFramebuffer`'s pure methods and signs their slots
with its own discriminators, which no caller uses. So each slot takes the
discriminator of the most-base table that covers it.

Last, Graft rewrites the consumer's vtable in place, from `vtableOffset`, in
the live parent's order and length. Its two leading words must be 0 before,
and are 0 after. A slot an override landed on gets the consumer's
implementation, and every other slot the live parent's method, each signed for
its own address with that slot's discriminator. The `impl` of each override
that landed must lie in the consumer's executable code.
