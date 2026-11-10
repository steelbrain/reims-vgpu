# Derive from a class with no header (arm64e)

Some classes a driver must derive from have no header in Apple's SDK, and
their vtables differ in length and order between releases. A consumer cannot
record such a parent's layout at build time. Instead it compiles against a
stand-in declaration of the parent, and Graft builds the consumer class's
vtable when it binds, in the running system's order, placing each of the
consumer's methods on the live slot that holds the same method. Graft calls
this a synthesized class.

Synthesis runs on arm64e only. On x86-64 the SDK refuses a synthesized class.
The SDK ships one stand-in, `graft/IOMobileFramebufferLegacy.h`, for
IOMobileGraphicsFamily's `IOMobileFramebufferLegacy`, the display class an
arm64 WindowServer drives.

## What you need

- Everything a bound consumer needs ([bound-consumer.md](bound-consumer.md)).
  A synthesized class is part of a bound consumer and builds through the same
  `consumer.mk`, with `ARCH='-arch arm64e'`.
- The arm64e Graft, `Graft-arm64e.kext`.
- Boot firmware that hands Graft the vtable discriminator tables described
  below.

## 1. Derive from the stand-in

```cpp
#include <graft/IOMobileFramebufferLegacy.h>

class OurDisplay : public IOMobileFramebufferLegacy {
    OSDeclareDefaultStructors(OurDisplay)
    // the methods you override
};
```

Implement the parent's pure methods your display needs, and the concrete
methods whose behaviour you replace. Do not call any of the virtual methods
the stand-in declares, except `start`, `stop` and `free`:

- not virtually, because Graft fills your vtable in the live class's order,
  not the stand-in's;
- not as a qualified call such as `IOMobileFramebufferLegacy::start_hardware()`.
  Under `-fapple-kext`, a qualified virtual call loads the method from the
  class's vtable at the slot the compiler assigned from the stand-in, and the
  live class keeps another method there.

`start`, `stop` and `free` lie in `IOService`'s part of the vtable, which every
release keeps in the SDK's order, so a qualified call to them is right:

```cpp
bool OurDisplay::start(IOService *provider) {
    return IOMobileFramebufferLegacy::start(provider);
}
```

The stand-in reserves virtual slots past its declared methods
(`GRAFT_IOMFB_VTABLE_RESERVE`), so its part of your vtable is at least as long
as the live class's vtable. Graft fills that part in place, in the live order,
and leaves any virtual methods your class adds after it. The stand-in also
reserves room for the live class's fields (`GRAFT_IOMFB_LEGACY_RESERVE`), so
your own fields come after them.

## 2. Declare it in graft.json

The stand-in goes in `classes` like any parent. Your class goes in
`synthesized`, with the methods it overrides:

```json
{
  "libraries": [{"id": "com.apple.iokit.IOMobileGraphicsFamily"}],
  "classes": [
    {"name": "IOMobileFramebufferLegacy",
     "header": "graft/IOMobileFramebufferLegacy.h", "library": 0}
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
| `name` | your class, which Graft's refusals name |
| `header` | the header that declares it |
| `parent` | the index in `classes` of the stand-in it derives from |
| `sibling` | optional: a concrete subclass of the parent, through which Graft finds the slot of a method the parent leaves pure |
| `overrides` | the methods you override, by method name; required and not empty |

Each override takes two optional keys:

| key | meaning |
|---|---|
| `required` | `true` makes Graft refuse the bind when it finds no live slot for the method. Mark the parent's pure methods required |
| `selectors` | other mangled names for the method, tried after your class's own, for a method whose parameters differ between releases |

The build also records your destructors and `getMetaClass`, which
`OSDefineMetaClassAndStructors` defines, as required overrides. Graft then
refuses the bind rather than leave the parent's versions of those in your
class. [sdk/README.md](../../sdk/README.md#synthesized-classes) has the full
rules.

## 3. Have the firmware hand over the discriminators

On arm64e each vtable slot holds a signed pointer. A caller authenticates a slot
with a constant discriminator the compiler gives each method. The kernel's
loader uses that value when it signs a vtable, and leaves it nowhere Graft can
read at run time. So the boot firmware reads it from the kernel collection on
disk and publishes it in the device tree, one table per vtable symbol, under
`/chosen/ambrosia/vtable-disc`.

A caller uses the discriminator of the method's first declaration, which may
belong to an ancestor of the parent. Graft therefore needs the table of the
parent and of each ancestor that the parent's library defines. For
`IOMobileFramebufferLegacy` the firmware must export these vtables:

```
__ZTV25IOMobileFramebufferLegacy
__ZTV26IOMobileFramebufferService
__ZTV19IOMobileFramebuffer
```

[src/README.md](../../src/README.md#what-the-arm64-boot-firmware-provides)
describes the device-tree format. Graft checks each table against the live
vtable it measures, its address, its length and the discriminator each slot is
signed with, and refuses the bind by name if any is missing or disagrees.

## 4. Build and check

Build the consumer for arm64e as in [bound-consumer.md](bound-consumer.md).
The manifest then carries the synthesis tables
([manifest.md](../manifest.md#synthesized-classes)).

In a guest booted with SIP on, `ioreg -l -d 1 | grep Graft` shows the
consumer's `GraftClient` entry reading `bound: ...`, and the kernel constructs
and starts your class.

## When the bind is refused

Each reason starts with your class's name.

| reason | what to do |
|---|---|
| `override N unresolved` | No live slot holds that method. Add the release's mangled name to the override's `selectors`, or, for a method the parent leaves pure, name a concrete `sibling` |
| `no vtable-disc handover for <symbol>` | The firmware did not export that vtable's table. Add the symbol to its export list |
| `donor table for <symbol> invalid`, `diverges from the live vtable` or `does not sign the live vtable's slot S` | The table is malformed, or its address, length or a discriminator does not match the vtable in the running system |
| `live parent <class> has N vtable words, over the M the stand-in reserves or the C-word vtable` | This release's class has more vtable slots than the stand-in declares, or the manifest declares a reserve larger than the compiled vtable |
| `live parent <class> instance size N over the M reserved` | This release's class is larger than the stand-in reserves room for |
| `vtable offset is not the start of its vtable` | The consumer was built with an SDK that recorded the wrong offset. Rebuild it with the current SDK |
| `synthesis parent <class> is not a declared class` | The stand-in is missing from `classes` |

[debugging-refusals.md](debugging-refusals.md) lists the rest.
