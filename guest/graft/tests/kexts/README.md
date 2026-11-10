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

[tools/internal/boot-core.py](../../tools/internal/boot-core.py) injects it with
`--probe` and requires `GraftProbeState` to reach `passed`.

## route-probe

`GraftRouteProbe`, the arm64e counterpart of core-probe for the route layer that
shadows AMFI's MAC policy. It routes the three AMFI admission operations to
assembly wrappers that count the call and tail-branch the authenticated
original. It verifies that Graft resolves AMFI, builds the shadow policy and
repoints the list entry, and that the shadow sits in the live dispatch path, so
every routed operation is entered. A wrong original faults under pointer
authentication, so the branch also checks the handed-back pointer.

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
landed, not just that the personality matched. Its bundle identifier is
`ai.aneesiqbal.ReimsGraphics`, the one Graft binds (`src/Policy.h`).

It is a consumer, so it is built through the SDK with its own Makefile:

```sh
make -C tests/kexts/gate-probe                      # build/GraftGateProbe.kext (x86-64)
make -C tests/kexts/gate-probe ARCH='-arch arm64e'  # the arm64e variant
make -C tests/kexts/gate-probe REFUSE=1             # the forced-refuse variant
```

The `REFUSE=1` variant declares an absent provider anchor
(`src/graft-refuse.json`), so Graft refuses the bind and never opens the gate;
it shows that a refused consumer stays dormant with no category started.
boot-core.py does not inject gate-probe.
