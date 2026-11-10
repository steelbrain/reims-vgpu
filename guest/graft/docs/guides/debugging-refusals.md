# Find out why Graft refused

When Graft cannot serve a kext, it refuses that kext by name, records why, and
lets the guest go on booting. It never binds part of a consumer. This guide
shows where Graft reports, and what each reason means.

## Where to look

Graft publishes its state on the IORegistry root, and writes each step to the
kernel log with the prefix `graft:`. In the guest:

```sh
ioreg -l -d 1 | grep Graft
log show --last boot --predicate 'process == "kernel"' | grep graft:
```

| property | holds |
|---|---|
| `GraftState` | `starting`, `active` or `refused` |
| `GraftReason` | why Graft is in that state |
| `GraftLog` | the last 8 KiB of Graft's log |
| `GraftClient.<bundle id>` | one per kext Graft serves, as `<state>: <reason>` |

Start with `GraftState`. Graft serves nothing unless it is `active`. Then read
the `GraftClient` entry of the kext that did not work. When a reason is not
enough, `GraftLog` and the kernel log hold the steps before it.

## Graft's own start

`GraftState` reads `active` once Graft has started. Otherwise its reason is
one of these:

| reason | meaning |
|---|---|
| `kernel collection or calibrated symbols unavailable` | Graft could not find the running kernel's image or place its symbol table. When the log also says `binding: refuse kernel collection: load commands exceed the 64 KiB cap`, the Boot collection's load commands are larger than Graft reads. On arm64e, check that the firmware published `boot-kc-base` ([src/README.md](../../src/README.md#what-the-arm64-boot-firmware-provides)) |
| `kernel write/routing facilities unavailable` | Graft could not prepare its kernel writes or its routes, such as when a kernel symbol they need is missing |
| `catalogue binding routes unavailable` | Graft could not install the trigger it binds from; on x86-64 that is a route of the kernel's two catalogue entry points |

When the kernel log has `graft: refused: cannot allocate diagnostics` and the
registry has no `GraftState`, Graft could not allocate its log and stopped
before anything else.

## A bound consumer

A consumer's state is `waiting`, `bound` or `refused`.

`waiting` is not a failure on x86-64: Graft tries again at each catalogue
send, and the kernel sends again as more kexts register, such as when the
System collection does. On arm64e Graft binds once, at start, so a consumer
still `waiting` there stays so for the boot.

| waiting reason | meaning |
|---|---|
| `no consumer injected` | Graft found no consumer kext. Check that the boot image carries it, with the identifier `src/Policy.h` names. On arm64e, check that the firmware published the consumer's device-tree node |
| `<library> is not registered` | x86-64: a provider library is not registered with the kernel yet |
| `<library> is not started (load <code>)` | x86-64: a provider is registered, but loading it has not succeeded yet |
| `cannot allocate binding resources` | Graft could not allocate the memory a bind needs, and tries again at the next catalogue send |
| `cannot allocate synthesis scratch` | arm64e: Graft could not allocate the memory a synthesized class needs. Graft binds only at start there, so the consumer stays `waiting` for the boot |

A refusal is final for the boot. The reasons, in the order Graft checks:

| refused reason | meaning and what to do |
|---|---|
| `invalid consumer manifest` | the manifest in `__DATA,__graft` broke a rule of [manifest.md](../manifest.md#validation). Graft does not say which rule. Rebuild the consumer with the SDK of the Graft release you boot |
| `consumer started before binding` | the consumer was already running, so it was not injected dormant. Build it through `consumer.mk` |
| `write-plan size overflow` | the manifest declares more writes than Graft can count. Rebuild the consumer with the SDK of the Graft release you boot |
| `<library>: no registered collection entry` (x86-64) or `<library>: not a collection entry` (arm64e) | the provider is not in any kernel collection Graft reads |
| `<library>: cannot verify <class>'s class anchor (<why>)` | Graft could not confirm the provider's image from the class's live metaclass. `class not registered` means the provider's classes are not registered; for a provider the consumer only imports from, declare an `anchor`. `image load commands exceed the 64 KiB cap` means the provider's load commands are larger than Graft reads |
| `<class>: missing or unbounded vtable` | the parent class is not registered, or its vtable symbol is missing from its provider or holds no method |
| `<class>: vtable slot S is not retained executable code` | a slot of the live parent's vtable does not point into the code of the kernel or of a provider, so Graft does not trust that vtable |
| `site N: address outside consumer segments`, `import N: fallback outside consumer segments` | the manifest names a place outside the consumer's loaded image. Rebuild the consumer |
| `<symbol>: site N refused (plan P, <detail>)` or `site plan refused (P)` | an import could not be resolved or written. `<symbol>` is its first candidate name, and `<detail>` names the fixup error |
| `site N: <symbol> resolves outside its provider`, `site N: <symbol>: replacement address overflow` | the import resolved to an address outside the library it was declared from, or its new value overflows |
| `<class> slot S refused (plan P, layout L)` or `vtable V slot S refused (plan P, layout L)` | the parent class in this release differs from the one the consumer was compiled against, or consumer vtable `V` could not be planned. See [the codes](#reading-the-codes) |
| `write N: offset outside consumer segments` | a planned write lies outside the consumer's loaded image. Rebuild the consumer |
| `write N: target segment is not writable` | arm64e: a write would land in a segment the kernel maps read-only, such as `__DATA_CONST`. Build the consumer through `consumer.mk`, which links it with `-no_data_const` |
| `write N: target overlaps the manifest section` | a write would land in the consumer's own `__DATA,__graft`. Rebuild the consumer with the current SDK |
| `write transaction refused (E)` | a value changed under Graft before it wrote, or a write did not read back. Graft restores what it wrote |

An arm64e consumer with a synthesized class can also be refused for the
reasons in [A synthesized class](#a-synthesized-class-arm64e), after the
vtable checks and before the writes.

Graft checks every `write N` target before the first store, so those
refusals leave the consumer unwritten.

Once the writes are committed, the state is `bound`, and its reason says
whether the gate is open:

| bound reason | meaning |
|---|---|
| `imports and vtables committed; waiting for device <match> to open the gate` | no PCI nub with `IOPCIPrimaryMatch` `<match>` is published yet. Graft opens the gate on every matching nub, whenever it appears |
| `imports and vtables committed; activation gate open on the device nub` | Graft set the gate property on a matching nub, and the kernel now constructs and starts the consumer |
| `imports and vtables committed; cannot build the device match` | Graft could not build the matching dictionary for the device, so the gate stays shut |
| `imports and vtables committed; cannot watch for the device nub` | Graft could not register for the device's publication, so the gate stays shut |

A consumer that stays at `waiting for device` has no device in the guest that
its personalities match: check the PCI device the virtual machine presents.

## A synthesized class (arm64e)

Each reason starts with the consumer class's name. In the order Graft
checks:

| reason | meaning and what to do |
|---|---|
| `synthesis parent <class> is not a declared class` | the stand-in parent is missing from `classes` in `graft.json` |
| `live parent <class> has N vtable words, over the M the stand-in reserves or the C-word vtable` | this release's class has more vtable slots than the stand-in declares, or the manifest declares a reserve larger than the compiled vtable |
| `live parent <class> instance size N over the M reserved` | this release's class is larger than the stand-in reserves room for |
| `no vtable-disc handover for <symbol>` | the firmware exported no discriminator table for that vtable. Add it to the firmware's export list |
| `donor table for <symbol> invalid (error E, slot S)` | the handed table is malformed |
| `donor table for <symbol> diverges from the live vtable (error E)` | the table's vtable address or length does not match the running system |
| `donor table for <symbol> does not sign the live vtable's slot S` | the table's discriminator for slot `S` is not the one the running system signed that slot's method with, so the firmware's table is not this release's |
| `ancestor <class>'s vtable name is too long` | an ancestor's vtable symbol does not fit Graft's buffer |
| `more than 4 classes of <class>'s chain in its provider` | the parent's library defines the vtables of more of its ancestors than Graft holds tables for |
| `donor table for <symbol> is longer than its subclass's` | an ancestor's table has more slots than the table of the class below it |
| `override N unresolved (plan P, slot S)` | no live slot holds that method, or two do, or two overrides landed on one slot. The one slot with the override's discriminator does not count when a symbol names its method as another. Add another mangled name to its `selectors`, or name a `sibling` |
| `override N implementation is not consumer code` | the override's implementation is outside the consumer's code |
| `override N is signed as <byte>, not as a vtable slot` | the consumer's build signed the method's slot with another scheme than a vtable slot's. Rebuild it with the current SDK |
| `vtable offset outside consumer segments` | the consumer's vtable, at the live parent's length, does not lie in the consumer's image |
| `vtable offset is not the start of its vtable` | the consumer was built with an SDK that recorded the wrong offset. Rebuild it |
| `vtable build failed (error E, slot S)` | Graft could not build the slot `S` of the vtable |
| `synthesis exceeds write capacity`, `vtable word N outside consumer segments` | the vtable Graft built does not fit the writes the manifest allows, or the consumer's image |

[synthesized-class.md](synthesized-class.md) explains the parts these name.

## The owned client

The client's state is whatever it last set with `graft_client_state`, or what
Graft set after a call:

- `refused: <symbol>` after `graft_kernel_symbol` found no such symbol in the
  running kernel.
- `routed: ...` after `graft_route_kernel` installed every route.
- `refused: ...` naming the route and the reason when it installed none.

A call that passes another identifier than the owned client's, or arrives
while Graft is not active, or asks for no routes or more than six, returns
false and records nothing.

x86-64 route refusals:

| reason | meaning |
|---|---|
| `live installation after BSD publication` | routes were requested after BSD started. Install them from `graft_client_initialize()` |
| `invalid request N` | request `N` has no symbol, replacement or original, or its original is not pointer-aligned |
| `<symbol>: replacement is not in the client's executable text` | the replacement does not point into the client's code, such as a data pointer passed by mistake |
| `<symbol>: missing, duplicate or unreadable function` | the kernel does not define that name exactly once in readable code |
| `<symbol> at +N: <why>` | the instruction at byte `N` of the function cannot be moved into the trampoline; `<why>` names the reason, below |
| `<symbol>: overlaps an installed entry` or `overlaps another route in this request` | the function is already routed, or two requests overlap |
| `transaction: all 8 owned route slots are reserved` | the boot's routes are used up |
| `installation after an unverified entry rollback` | an earlier route failed and Graft could not confirm it restored the function, so it routes nothing more this boot |
| `trampoline preparation: write error E`, `transaction: write error E` | a kernel write failed; Graft restored what it wrote. `E` is `WriteError` in `src/WritePlan.h` |
| `transaction: entry rollback unverified; replacement code and original storage stay resident` | a write failed and Graft could not confirm it restored the functions. The routes may be live, and Graft routes nothing more this boot |

The `<why>` of an `at +N` refusal is one of:

| why | meaning |
|---|---|
| `entry requires ENDBR preservation` | the function starts with `ENDBR64` |
| `function ends before route fits` | a return or unconditional jump comes before byte 14, so the function may end inside the jump |
| `route capacity` | the instructions to move take more than 32 bytes, or do not fit the trampoline once relocated |
| `displaced call returns inside the route window` | a `call` among the moved instructions, relative or indirect, would return into the bytes the jump replaces |
| `branch targets displaced instruction interior` | a branch among the moved instructions lands inside another moved instruction |
| `memory operand references displaced bytes` | an instruction reads or writes the bytes the jump replaces |
| `RIP-relative operand out of range` | an operand cannot reach its target from the trampoline |
| `unsupported instruction`, `unsupported instruction prefix`, `truncated instruction`, `instruction exceeds 15 bytes` | Graft's decoder cannot move the instruction |

arm64e route refusals, in the order Graft checks:

| reason | meaning |
|---|---|
| `the owned client's routes are already installed` | the client already routed once this boot |
| `transaction: N requests exceed the 3 routable operations` | the call asks for more routes than there are operations |
| `invalid request N` | request `N` has no operation, replacement or original, or its original is not pointer-aligned |
| `request N: duplicate original storage` | two requests store their original in the same place |
| `AMFI image or its executable range not found` | Graft could not find AMFI's code in the Boot collection |
| `the client's placed image cannot be read (<why>)` | the firmware names an image for the client's bundle identifier, but its load commands are malformed or its header lies outside its segments |
| `request N: replacement is not in the client's executable text` | the replacement does not point into the client's code. Graft checks this when the firmware's device tree names an image for the bundle identifier the call passes |
| `mac_policy_register is not resolvable`, `mac_policy_list is not resolvable`, `mac_policy_list is not in the kernel's data segments`, `mac_policy_list entries pointer is invalid` | Graft could not find the kernel's list of MAC policies. The list must lie in a segment the kernel maps writable, and its entries pointer must be a canonical, 8-byte aligned kernel address |
| `AMFI policy not found in the list`, `AMFI policy ambiguous in the list (N matches)` | Graft could not find AMFI's one registered policy |
| `<name>: route input` | Graft could not build its copy of AMFI's ops table |
| `<name>: operation not routable on arm64` | the request names an operation other than `file_check_mmap`, `file_check_library_validation` or `vnode_check_signature` |
| `<name>: staged copy too small for the structure` | the operation's slot lies past the end of Graft's copy of AMFI's ops table |
| `<name>: operation routed twice in one request` | two requests name the same operation |
| `<name>: live ops slot does not point into AMFI text` | the slot Graft would replace in AMFI's table does not hold a pointer into AMFI's code, so Graft does not trust the table |
| `<name>: live ops slot is not signed with the operation's discriminator` | the slot's pointer is not signed with the discriminator Graft derives from the operation's `mpo_*_t` prototype in XNU's `security/mac_policy.h`, as when a release changed that prototype. Graft routes nothing, rather than a slot the kernel would fail to authenticate |
| `shadow conf: <why>` | Graft could not build its copy of AMFI's conf |
| `policy-list element is not in writable memory` | the list entry Graft would swap cannot be written |
| `policy-list element did not hold AMFI's conf` | the list entry changed before Graft swapped it. Nothing was written |

## Reading the codes

A binding refusal can end in numeric codes, such as `(plan 2, layout 4)`. The
text before them names what failed. The codes are the values of enums in
`src/`:

- `plan` in a class refusal is `VtableError` in `src/Vtable.h`, and in a site
  refusal `binding::Error` in `src/Binding.h`;
- `layout` is `layout::Error` in `src/Layout.h`;
- `E` in a write refusal is `WriteError` in `src/WritePlan.h`;
- `error` in a donor table refusal is `synth::DonorError` in
  `src/arm64e/VtableDisc.h`, `plan` in an override refusal is
  `synth::PlanError` in `src/arm64e/Synthesize.h`, and `error` in a vtable
  build refusal is `synth::BuildError` in `src/arm64e/SynthesizeVtable.h`.

The layout codes are the ones met most:

| layout | meaning |
|---|---|
| 2 | the class is not registered |
| 3 | its instance size differs from the consumer's build |
| 4 | its vtable length differs from the consumer's build |
| 5 | a slot of its vtable could not be read |
| 6 | a slot does not hold the method the consumer expects |

A size or length difference means the consumer was compiled against headers
that do not match this release's class. Rebuild against headers that do, or
handle the release in the consumer.

## Refusals at build time

The SDK refuses, at build time, a consumer that Graft could not bind. Its
command prints `consumer:` and the reason, and exits 1. Fix the declaration,
the Info.plist or the source it names, and build again.
