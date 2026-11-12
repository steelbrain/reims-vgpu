# test kexts

Three kexts that only a booted guest can check. The host tests in
[tests/](../README.md) cover Graft's logic without a kernel; these exercise what
needs a running system: that routes install and fire, and that a bound
consumer's activation gate opens. They are test-only and are not part of a
release. For a kext to copy, read [examples/](../../examples/README.md) instead.

Each is built on a Mac with Xcode and fetches nothing.

## core-probe

`GraftCoreProbe`, an owned client for the x86-64 route layer. It routes six MAC
dispatchers to assembly wrappers that count each call and tail-branch the
original, so it needs no guessed argument types. After BSD starts it checks that
a lazy `graft_kernel_symbol` lookup matches a known import and that every routed
function was entered. It verifies that Graft routes kernel text early enough and
that its symbol lookups stay correct late in boot.

It is built by the top Makefile:

```sh
make core-probe   # build/GraftCoreProbe.kext
```

[tools/internal/boot-core.py](../../tools/internal/boot-core.py) injects it as a
client, `--client build/GraftCoreProbe.kext routed`, and requires
`GraftProbeState` to reach `passed`.

## route-probe

`GraftRouteProbe`, the arm64e counterpart of core-probe for the route layer that
shadows a MAC policy. It routes the three AMFI admission operations to assembly
wrappers that count the call and tail-branch the authenticated original. It
verifies that Graft resolves AMFI, builds the shadow policy and repoints the list
entry, and that the shadow sits in the live dispatch path, so every routed
operation is entered. A wrong original faults under pointer authentication, so
the branch also checks the handed-back pointer. It then routes the Sandbox
policy's own `file_check_mmap` by naming the policy (`Sandbox:file_check_mmap`),
to exercise a second policy's shadow in one boot and hold its original apart from
AMFI's; that leg reports under its own `GraftRouteProbeSandbox*` registry keys.

It is built by the top Makefile for arm64e:

```sh
make route-probe   # build/GraftRouteProbe.kext
```

It runs on an arm64e boot, not under boot-core.py, which boots x86-64.

## gate-probe

`GraftGateProbe`, a bound consumer that exercises the activation gate across
several `IOMatchCategory` values on one device nub. Graft sets one gate property
on the nub, and every gated personality matches together; each started
personality also exercises a bound import, so a started category proves the bind
landed, not just that the personality matched. Its bundle identifier is its own,
`ai.aneesiqbal.graft.GateProbe`: Graft binds every placed kext that carries a
manifest.

Its three personalities match the PCI host bridge by its class code,
`IOPCIClassMatch` `0x06000000&0xffff0000`, each with its own `IOMatchCategory`.
QEMU's q35 machine, the x86-64 test guest, and vmapple, the arm64e one, both
present a host bridge of class `0x060000` with no driver attached, so the probe
needs no device of its own on either architecture. Each started category
declines to attach, so the bridge stays as it was.

It is a consumer, so it is built through the SDK with its own Makefile:

```sh
make -C tests/kexts/gate-probe                      # build/GraftGateProbe.kext (x86-64)
make -C tests/kexts/gate-probe ARCH='-arch arm64e'  # the arm64e variant
make -C tests/kexts/gate-probe REFUSE=1             # the forced-refuse variant
```

The `REFUSE=1` variant declares an absent provider anchor
(`src/graft-refuse.json`), so Graft refuses the bind and never opens the gate;
it shows that a refused consumer stays dormant with no category started.

boot-core.py injects it as a consumer, `--consumer build/GraftGateProbe.kext
bound`, and requires the key each started personality sets on the registry
root, `GraftGateProbe.<category>` named by its `GraftGateCategory`, to read
`passed`. The `REFUSE=1`
variant goes in as `--consumer build/GraftGateProbe-refuse.kext refused`, or
with more of the refusal as its state, beside other kexts to show that one
consumer's refusal leaves the others bound.
