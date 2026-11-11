# tests

The host tests check everything about Graft that needs no kernel. `make test`
builds and runs every one of them, and stops at the first failure. What only a
booted guest can show is checked with the test kexts in
[kexts/](kexts/README.md).

The C++ tests compile Graft's own headers from `src/`, or the SDK tool's sources
from `sdk/tool/`, for the host. They report a failed check with its file, line
and expression (`Check.h`). The Python tests use `unittest`. Every input is
synthetic or built here from this repository's own source; no test reads an
Apple binary.

## What the host must be

`make test` needs a Mac with Apple silicon, Xcode, Rosetta 2, and Python 3.9 or
later with only its standard library.

- `make test` runs every Python test under `python3` and again under
  `SYSTEM_PYTHON`, which is `/usr/bin/python3` unless set. macOS's own Python
  may be 3.9, and a consumer runs `tools/vendor.py` with whatever `python3`
  its Mac has, so the tools and the tests keep to 3.9.
- Before the SDK tool's tests run, `make test` builds `graft-kext` and, with
  its `build`, the consumers in `consumer/` and `synthesized/`. The tool's C++
  tests read what those builds wrote
  ([sdk/README.md](../sdk/README.md#tests)).

- Four tests run as native arm64e processes, because they sign and
  authenticate pointers with the CPU's pointer authentication:
  `arm64e_fixup_test`, `arm64e_route_test`, `arm64e_vtable_disc_test` and
  `arm64e_synthesize_vtable_test`. They need a macOS that runs arm64e code built
  outside Apple. macOS 27 runs them with no boot-args; an older macOS may need
  the `-arm64e_preview_abi` boot-arg on the Mac that runs the tests.
- Three tests are built for x86-64 and run under Rosetta 2: `binding_test` and
  `manifest_test`, which plan x86-64 import sites, and `x86_route_test`, which
  runs the x86-64 routes it builds.

## The kext's logic

| test | covers |
|---|---|
| `macho_test.cpp` | the Mach-O reader: duplicate-aware symbol lookup, malformed and truncated load commands, fileset entries and their identifiers, which Graft finds the x86-64 consumers by, `__LINKEDIT` bounds, symbol-table snapshots, and a real dylib built from `fixture.c` |
| `binding_test.cpp` | pointer and branch fixup arithmetic, the whole-plan checks on import sites (bounds, order, missing and ambiguous symbols), and the class size and vtable length comparison |
| `manifest_test.cpp` | manifest validation, including the versions it reads, the version it names when it refuses another, and the provider matching string, import resolution with candidates and fallbacks, weak imports through the plan and the target check, provider anchors, vtable planning, and a manifest section read back from a linked kext |
| `x86_route_test.cpp` | x86-64 instruction decoding and route building, with 20,000 generated instruction streams, refusal of unsupported forms, and routes run on the host |
| `write_plan_test.cpp` | the write transaction: every old value checked before any store, read-back, rollback, and refusal of overlapping or malformed declarations |
| `arm64e_fixup_test.cpp` | the arm64e import fixup: a signed placeholder is re-signed for the real target under the same key and discriminator, checked against an independent signer, and a tampered slot or discriminator is refused |
| `arm64e_route_test.cpp` | the arm64e route plan: a shadow of a policy ops table whose routed slots authenticate to the replacements, untouched bytes unchanged, and refusal of an unknown operation, a slot outside the policy's code, a slot signed with another discriminator than its operation's, and a duplicate operation |
| `arm64e_synthesize_test.cpp` | the synthesis plan: a method's identity from its mangled name, its symbol rebuilt under another class, the one slot that holds an address, the names that keep the discriminator fallback off another method's slot, and refusal of an unplaced required override, an ambiguous slot or a collision |
| `arm64e_vtable_disc_test.cpp` | the parser of the firmware's discriminator tables: a well-formed table reads back, every malformed one is refused, one whose address, length or a discriminator misses a signed synthetic vtable is refused, and a chain gives each slot its most-base table's discriminator |
| `arm64e_synthesize_vtable_test.cpp` | building a synthesized vtable: the two leading words zeroed, inherited and overridden slots each signed for their own address with the chain's discriminator, and a chain short of a slot refused |

## The SDK and the tools

| test | covers |
|---|---|
| `staging_test.cpp` | staging the consumer fixture: class probes, the import census, the staged image, weak imports, provider anchors, the dormant bundle and its provider matching, and refusing a short or zerofill reservation; for arm64e, its chained fixups and a version-5 manifest whose signed sites carry their signing context |
| `sdk_standin_test.py` | a subclass of the SDK's `IOMobileFramebufferLegacy` stand-in, compiled for both architectures: its reserved slots and the symbols its overrides produce |
| `synth_emit_test.cpp` | staging an arm64e synthesized consumer: the synthesized-class and override records |
| `prelink_test.cpp` | the prelinked-collection check, on synthetic x86-64 collections built from the staged consumer, intact and corrupted, and what `graft-kext check-prelinked` prints and exits with |
| `graft_kext_test.py` | `graft-kext` as a consumer's build runs it, for x86-64 and arm64e: what `build` stages and leaves in its work directory, the provider matching it records for personalities of any provider, `build` against its steps run one at a time, and each refusal's exit status and text |
| `consumer_mk_test.py` | `sdk/consumer.mk` under make, included from a makefile of another name: the staged bundle current after a build, and not after an SDK header changes or a build fails |
| `hook_symbols_test.py` | `tools/internal/hook-symbols.py`: an unreadable input or symbol list is an error, never a missing symbol, and the symbols it reads from `hook-symbols.txt` or a client's list |
| `release_test.py` | `tools/internal/release.py`: the release's folder and zip, the tracked tree and both kexts, the same zip bytes from another checkout, umask and file modes, the tracked files it builds from and carries, refusing a tree that differs from `HEAD`, a tracked link, a tracked name the release adds, another Xcode, bundles of two versions and a malformed bundle; and every link in this repository's release resolving inside it |
| `vendor_test.py` | `tools/vendor.py`: taking a zip that `release.py` wrote, with its record; refusing a file that differs from `SHA256SUMS`, a file missing from either, a link, a directory entry, a name outside the zip's one folder, a name twice, a malformed `SHA256SUMS` or `COMMIT`, and a release that holds a `source.toml`; keeping the earlier copy when a copy fails; and never replacing a directory it did not write |
| `lint_test.py` | `tools/internal/lint.py`: license lines, text hygiene, Markdown links, and the bundle identifiers `src/` and `sdk/` may name, in a C file's comments and literals and anywhere in another file |

## Fixtures

| file | what it is |
|---|---|
| `Check.h` | the check macro the C++ tests use |
| `fixture.c` | the source of the small dylib `macho_test` reads |
| `kernel_fixture.cpp` | a test-only kext carrying a manifest, which `make kernel-check` links and `manifest_test` reads back; it is never booted |
| `ToolTest.h` | what the SDK tool's C++ tests share: a consumer as `graft-kext build` left it, a refusal check, a manifest read back as Graft reads it, and the file offsets of a Mach-O file's segment and section headers |
| `consumer/` | the consumer fixture: a class standing for a provider's, a consumer class that overrides and appends methods, pointer, call and weak imports, and its declaration and Info.plist |
| `synthesized/` | a synthesized display consumer, a subclass of the SDK's `IOMobileFramebufferLegacy` stand-in, with its declaration and Info.plist |
| `release_fixture.py` | a throwaway repository holding one tagged release, for the release, vendor and lint tests |

The tests that make temporary git repositories run git with no user or system
configuration and ignore the caller's `GIT_*` variables.
