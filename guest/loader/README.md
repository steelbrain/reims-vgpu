# ReimsGraphicsLoader

OpenCore injects kexts into the Boot kernel collection and links them only
against what that collection holds. ReimsGraphics (`guest/kext`) subclasses
IOFramebuffer and IOAccelerator from IOGraphicsFamily and uses
AppleGraphicsDeviceControl, and both of those are in the System collection.
So ReimsGraphics loads through this loader, a Lilu plugin
([docs/graphics-kext/design.md](../../docs/graphics-kext/design.md), item 13).

## How it binds

1. `tools/gen-shims.py` reads which symbols of the two libraries
   ReimsGraphics names (its external relocations), and builds two shims:
   `RgfxIOGraphicsFamilyBridge.kext` and
   `RgfxAppleGraphicsDeviceControlBridge.kext`. A shim is a data-only kext
   that exports a placeholder for each of those symbols: a `ud2` stub for a
   function, and zeroes for a vtable or a metaclass. Its `fixups.json` lists
   every site in ReimsGraphics that refers to one of the libraries, with the
   placeholder OpenCore will link there.
2. `tools/stage-consumer.py` stages ReimsGraphics as OpenCore injects it:
   its `OSBundleLibraries` name the shims in place of the two libraries, and
   its `IOKitPersonalities` are taken out, so nothing matches it until it is
   bound.
3. `tools/gen-tables.py` writes the loader's `tables.h` from that table: the
   sites, the personalities, and the size and vtable of every library class
   ReimsGraphics derives from, read from the KDK (`tools/bridge.py`).
4. At boot, on each catalogue send, the loader (`src/Loader.cpp`) loads and
   retains the two libraries. It checks that each library class has the
   size and vtable slots the table expects, and that each site still holds
   its placeholder. Then it writes the real address into every site and
   publishes the personalities, and the send's matching starts ReimsGraphics
   on its PCI function (1234:5247). If any check fails, no site is written
   and ReimsGraphics stays dormant. The first sends run before the System
   collection is registered, so the loader retries on each send until the
   bind succeeds.

The loader depends on no other driver, and on no kext of the image other
than Lilu.

## What a change needs

The loader holds the offset of every site in one ReimsGraphics build, and
that build's personalities. So any change to ReimsGraphics needs a new
loader and new shims. `make -C guest set` builds them together, and
`make -C guest prebuilt` rebuilds the copy the repository carries in
`guest/prebuilt` (guest/Makefile).

Each part needs:
- the KDK of the release the tables are read from: its Extensions directory,
  as `KDK=`. It is only read for the two libraries' exports and class
  layouts, and for the kernel's exports. Nothing from it is committed;
- a DEBUG Lilu's plugin headers and MacKernelSDK, pinned and fetched by
  `guest/rgfxgate/tools/fetch-lilu.sh`.

## Releases

The set's OpenCore entries carry MinKernel 25.4.0 and MaxKernel 25.99.99,
so OpenCore injects it on macOS 26.4 to 26.x. Lilu starts the plugin only on
Darwin 25 (Tahoe). The tables are read from one KDK, and the loader refuses
any release whose library classes are laid out differently from it. That
refusal names the class in the log, and ReimsGraphics stays dormant.

## In the guest

The loader logs each step with the tag `rgfxload:`. The log goes to IOLog,
to Lilu's always-on log (`log show`) and, all of it, to the `LoaderLog`
property of its service, class `ReimsGraphicsLoader`
(`ioreg -l -w0 -c ReimsGraphicsLoader`). `-rgfxloadoff` turns it off, and
`-rgfxloaddbg` turns on Lilu's debug log for it. A bound ReimsGraphics shows
these lines:
- `ReimsGraphics: 3 library classes laid out as built`;
- `ReimsGraphics: all 114 sites resolved and placeholders checked`;
- `publish: ReimsGraphics PUBLISHED`.

The class and site counts are those of the current build.

## Provenance

All of it is our own code. Nothing in it is Apple's code or bytes taken from
Apple's binaries. The shims and tables it generates hold only symbol names,
sizes and offsets read from the KDK.

- `src/Loader.cpp`, `src/Fixup.h`, `src/Rows.h`, `src/Log.h`,
  `src/Info.plist`, `tools/bridge.py`, `tools/gen-shims.py`,
  `tools/stage-consumer.py`, `tools/gen-tables.py`, `tools/oc-link-check.py`
  and `tests/` come from the owner's paravirt-port-x86-guest repository,
  branch `steelbrain/loader-no-gop` at 50fed4bcdb83
  (`src/loader/AppleParavirtGPULoader`, `tools/loader`). There it was
  `AppleParavirtGPULoader`, and `gen-tables.py` was `gen-stage1.py`. It was
  trimmed to ReimsGraphics as its only consumer:
  - gone is the route that bound the x86-64 port's GPU and display kexts
    behind Apple's driver (its accelerator, mapper service and the
    withholding of Apple's personalities);
  - gone is the port's patch of its Metal bundle;
  - gone is the legacy one-library route.
  The bundle, its IDs and the boot arguments were renamed, and so were the
  shims' IDs, from `com.steelbrain.paravirt.` to `com.steelbrain.`.
- `tools/macho.py` is new. It replaces the Mach-O reader the original tools
  imported from another of the owner's repositories. For ReimsGraphics
  ef90881e, it gives the same `fixups.json`, byte for byte apart from the
  shims' IDs, and the same tables as the original tools did.
