# Consumer build tooling

`consumer.py` uses Python's standard library and Xcode's compiler. It reads
only our objects and linked consumer images. It fetches no headers, KDKs,
libraries or tooling, and it creates no bridge kexts.

The build has two links. The first supplies the exact import/site census
and compiler class metadata needed to size a manifest reservation. The
second adds that reservation as `__DATA,__graft`; staging then embeds the
manifest using that final image's offsets.

`consumer.mk` supplies those steps for a kext Makefile that defines `NAME`,
`VERSION`, `SDK`, `ARCH`, `BUILD`, `KEXT`, `BINARY`, `OBJECTS`, `HEADERS` and
`CXXFLAGS`. Include it in place of the ordinary final-link rule. The
consumer supplies `src/graft.json` and `src/Info.plist`; ordinary source
compilation stays in its existing build rules.

A declaration identifies providers, the superclass declarations compiled
against, and the consumer classes whose vtables Graft fills:

```json
{
  "libraries": [{"id": "com.example.OurProvider"}],
  "classes": [{"name": "Parent", "header": "Parent.hpp", "library": 0}],
  "consumers": [{"name": "OurDriver", "header": "OurDriver.hpp", "parent": 0}],
  "imports": {
    "_optionalFunction": {
      "library": 0,
      "candidates": ["_preferredFunction", "_olderFunction"],
      "fallback": "_ourFallback"
    }
  }
}
```

Compiler probes obtain `sizeof(Parent)` and the complete vtable layout
from these headers using the consumer's compiler flags. No-member probe
subclasses emit inherited method relocations, whose names identify every
slot. The kext ABI's trailing zero sentinel is excluded from method counts.
The probe source and object live in a disposable directory outside the
repository.

Undefined C++ symbols belonging to a declared superclass are assigned to
that class's provider. A provider's optional `classes` array can name other
classes it owns. `imports` supplies explicit policies for C symbols,
candidate names and fallbacks; omitted fallback means required. Other
kernel and Boot-KC imports are left for the boot linker.

A class's optional `required` array names inherited methods whose virtual
slot indices the consumer compiles into its calls. The tool maps them to
compiler slots and also requires every overridden parent slot. A missing
or ambiguous declared method is a build error. If `required` is omitted,
the declaration conservatively requires every parent method. An explicit
list must come from a complete call census; supplying a shorter list does
not by itself establish cross-release compatibility.

Staging turns the selected undefined imports into private local symbols
at distinct placeholders inside the reservation. It regroups the symbol
table and repairs external-relocation and indirect-symbol indices, leaving
the boot linker to relocate each reference normally. Only the three kinds
of imported compiler data used here need placeholders: superclass metadata,
vtable addresses and methods. The placeholders are never meant to execute
or serve as actual metadata; the consumer stays dormant until binding.

For declared System-KC-derived consumer classes, staging renames the
compiler's `superClass` linker marker to `graftClass`, preserving its bytes
and relocations. That defers their parent-vtable linking to Graft. Other
classes keep their normal markers and boot-linker handling. The manifest
records vtable destinations and overridden slot indices; inherited slots
are excluded from ordinary symbol sites so the runtime can copy them from
the actual parent without arbitrary duplicate-symbol selection.

The staged Info.plist retains its kernel/Boot-KC dependencies and loses its
selected System-KC dependency entries and personalities. Those
personalities are embedded as bounded XML with the correct consumer bundle
identifier. Staging validates its inputs before creating a new output
bundle; it refuses an existing output.

Run the compiler-produced fixture tests on a Mac with Xcode:

```
python3 -I -B tests/tooling_test.py
```

These cover class sizes and slots, local placeholders, symbol-index repair,
superclass markers, required methods, section capacity and dormant staging.
`prelink.py` checks prelinking; guest boots remain separate checks.

## Prelink check

`prelink.py` reads a kernel collection after OpenCore has prelinked our
kexts into it, together with the staged consumer bundles:

```
python3 -I -B tools/prelink.py COLLECTION Staged.kext...
```

For each consumer it finds the fileset entry named by the bundle
identifier and checks what Graft's runtime relies on when it binds that
entry:

- the entry's header starts its `__TEXT`, and each staged segment keeps
  its address relative to the header, since the runtime adds manifest
  offsets to the entry address;
- `__DATA,__graft` holds the staged manifest unchanged through its
  declared size; the placeholder tail after it is not compared;
- each site holds its exact local placeholder plus addend, and each
  placeholder lies in the reserved tail;
- no fixup-chain entry overlaps a branch site or the manifest.

An x86-64 collection stores every pointer as a chained rebase
(`DYLD_CHAINED_PTR_X86_64_KERNEL_CACHE`), which the booter turns into an
address when it slides the collection. A pointer site must therefore be a
chain entry, a plain rebase into this collection, whose decoded target is
its placeholder. A slot's `segment_offset` counts from the collection
header, and a rebase target counts from the lowest segment address. On the
11.7.10 and 26.7.2 Boot KCs, every Apple rebase target lands in a mapped
segment under that base. OpenCore reuses slot 0, which `__TEXT` leaves
empty, for the region it injects, and writes those starts after the
table's declared size. The check matches slots by `segment_offset` and
bounds them by the segment holding the table. Chains can run past their
starting page, so each region's chains are walked whole.

The synthetic tests link our compiler-produced fixture into collections
with the same encoding, then corrupt them:

```
python3 -I -B tests/prelink_test.py
```

## Vendoring into a consumer repository

A consumer repository builds against a copy of Graft's consumer files:

```
python3 -I -B tools/vendor.py DEST
```

DEST receives `dist/` (the committed bundle and its build record), the
public header, `consumer/Start.cpp` and this directory's consumer tooling
(`consumer.mk`, `consumer.py`, `graft_macho.py`, `prelink.py`, this file).
`DEST/source.toml` records the Graft commit they were copied at and each
file's SHA-256, in the `key = "value"` subset of TOML. The tool refuses when
any copied file differs from HEAD, so the recorded commit is the files'
source. It replaces DEST whole, and refuses a DEST that holds no earlier
record. The consumer commits DEST and its record together and checks the
files against the record.

## Committed Graft build

`dist/Graft.kext` is the original Graft-only output of the production build.
It has not been prelinked into a kernel collection. `dist/build.json` records
SHA-256 hashes of the production sources, public headers, consumer glue,
build tooling and Makefile; the selected Xcode, SDK, compiler and linker;
and each bundle file's hash, size and permission mode.

After a source change, update the bundle and record together:

```
python3 -I -B tools/build_record.py write
```

Verify the committed build before consuming it:

```
python3 -I -B tools/build_record.py check
```

Both commands copy the recorded sources into a disposable directory and run
`make all` there using the selected Xcode installation. They fetch nothing.
The check rejects source or toolchain drift, then compares the fresh bundle
with both the record and the committed bundle. Comparison is byte-for-byte;
no UUIDs, load commands, signatures or other binary data are normalized away.
Source or toolchain changes during a build are rejected. The generated
record has no timestamps or checkout-specific paths.

The record is reproducibility metadata, not a signature. Review and commit
source, bundle and record changes together. Tests for detecting source,
toolchain and artifact drift run with:

```
python3 -I -B tests/build_record_test.py
```
