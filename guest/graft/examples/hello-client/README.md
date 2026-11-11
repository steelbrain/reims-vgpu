# hello-client

The smallest owned client. An owned client is an ordinary kext that links
against Graft's exported C interface (`graft/Graft.h`) and uses it at boot.
[docs/consumers.md](../../docs/consumers.md) explains how an owned client is
built, and [sdk/include/graft/Graft.h](../../sdk/include/graft/Graft.h)
documents each call.

## What it shows

`src/HelloClient.cpp` uses the start glue `sdk/Start.cpp`, which declares the
kmod from `GRAFT_CLIENT_ID` and calls `graft_client_initialize()` once from the
kext's start routine. That function:

- checks `graft_client_ready`, which is true only when Graft is active and the
  identifier names a placed kext;
- looks one kernel symbol up with `graft_kernel_symbol` (`_IOLog`, exported on
  every release) and logs its address with `graft_client_log`;
- publishes its state with `graft_client_state`.

Graft serves a call when its identifier names a placed kext, so this client
passes its own bundle identifier, `ai.aneesiqbal.graft.HelloClient`. Graft
names no client, so serving this one needs no change to Graft.

This client installs no routes. Routing a kernel function, with the trampoline
and the callable original, is shown by the test client
[tests/kexts/core-probe](../../tests/kexts/core-probe). On x86-64 routes must be
installed before BSD starts, so a routing client installs from its start
routine; `graft_kernel_symbol` and the log and state calls have no such window.

## Build

The build needs a Mac with Xcode or its Command Line Tools, and fetches
nothing. Graft's header and start glue are two directories up, under
`../../sdk`.

```sh
make                      # ../../build/GraftHelloClient.kext (x86-64)
make ARCH='-arch arm64e'  # ../../build/GraftHelloClient-arm64e.kext (arm64e)
```

`ARCH` picks the target; x86-64 is the default. Both write into the
repository's `build/`, two directories up. The arm64e variant is suffixed,
so both can be built without overwriting each other.

## Install and expect

On x86-64, add `GraftHelloClient.kext` to the OpenCore image after
`Graft.kext`, which it depends on. On arm64e, the boot firmware places
`GraftHelloClient-arm64e.kext` beside `Graft-arm64e.kext`. Boot with SIP on and
no boot-args. In the guest:

```sh
ioreg -l -d 1 | grep Graft
```

shows `GraftClient.ai.aneesiqbal.graft.HelloClient` set to `ready: resolved
_IOLog`. The same line, and the logged address, are in Graft's log:

```sh
ioreg -l -d 1 | sed -n '/"GraftLog" = /,/^"$/p'
```

On arm64e they are also in the unified log, prefixed `graft: `; on x86-64 they
are not
([debugging-refusals.md](../../docs/guides/debugging-refusals.md#where-to-look)).
