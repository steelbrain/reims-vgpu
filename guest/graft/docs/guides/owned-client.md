# Write an owned client

An owned client is an ordinary kext that calls Graft's C interface to look up
kernel symbols by name and to route kernel functions to its own code. Graft
installs a route and hands back a callable original. What each replacement
does is the client's own policy. This guide builds one, using
[examples/hello-client](../../examples/hello-client/README.md) as the running
example.

Any kext placed at boot can be an owned client. Every call passes a bundle
identifier, and Graft serves it when that identifier names a kext the boot
linker placed: a fileset entry in the Boot collection on x86-64, a device-tree
node the firmware wrote on arm64e. So a client passes its own bundle
identifier; the example passes `ai.aneesiqbal.graft.HelloClient`, which is also
its `CFBundleIdentifier`. Graft names no client, so serving a new one needs no
change to Graft.

## What you need

- A Mac with Xcode or its Command Line Tools. The build needs nothing else and
  fetches nothing.
- Graft's SDK: the `sdk/` directory of a Graft checkout, or a copy of a
  release taken with `tools/vendor.py` ([taking-a-release.md](taking-a-release.md)).

## 1. Include Graft's header

Add the SDK's `include/` to your include path and include `<graft/Graft.h>`.
It declares five calls, each documented in
[Graft.h](../../sdk/include/graft/Graft.h):

| call | does |
|---|---|
| `graft_client_ready(id)` | returns whether Graft is active and `id` names a placed kext |
| `graft_kernel_symbol(id, symbol, &address)` | looks `symbol` up by name in the running kernel's own symbol table; a miss sets the client's state to `refused`, naming the symbol |
| `graft_route_kernel(id, requests, count)` | installs up to sixteen routes, all of them or none |
| `graft_client_log(id, format, ...)` | adds a line to Graft's log |
| `graft_client_state(id, state, reason)` | sets the client's state in the IORegistry |

## 2. Compile the start glue

Compile `sdk/Start.cpp` into the kext with two definitions:

- `GRAFT_CLIENT_ID`, your bundle identifier as a bare token, such as
  `-DGRAFT_CLIENT_ID=ai.aneesiqbal.graft.HelloClient`;
- `GRAFT_CLIENT_VERSION`, your version as a string, such as
  `-DGRAFT_CLIENT_VERSION='"1.0.0"'`.

The glue declares the kext's kmod. Its start routine calls your
`graft_client_initialize()` once. Its stop routine always fails, because a
client's routes stay in place for the rest of the boot.

## 3. Depend on Graft

List `ai.aneesiqbal.graft` in `OSBundleLibraries` at version `0.1.0`. The
kernel then links your calls to Graft's exports and starts Graft before your
kext.

## 4. Give the kext a personality

An injected kext marked `OSBundleRequired` `Root` is not started for that
alone. Give it an `IOService` subclass and one personality that matches
`IOResources` with `IOResourceMatch` `IOKit`. The example's
[Info.plist](../../examples/hello-client/src/Info.plist) shows the shape.

## 5. Write graft_client_initialize

The example checks that Graft is active, looks one symbol up, and publishes
its state:

```cpp
#include <graft/Graft.h>

constexpr const char *Client = "ai.aneesiqbal.graft.HelloClient";

void graft_client_initialize() {
    if (!graft_client_ready(Client)) {
        graft_client_state(Client, "refused", "Graft not active");
        return;
    }
    uint64_t address = 0;
    if (!graft_kernel_symbol(Client, "_IOLog", &address))
        return;
    graft_client_log(Client, "_IOLog resolves to 0x%llx", address);
    graft_client_state(Client, "ready", "resolved _IOLog");
}
```

A client that routes passes an array of requests. Each names what to route,
the replacement that receives its calls, and where Graft stores the original,
which the replacement calls to pass a call on. What a request names, and the
prototype the replacement and original have, depend on the architecture.

Keep the original in a `void *`, and convert it to the prototype where you
call it, as both examples below do. On arm64e, kexts are compiled with
function-pointer type discrimination: the compiler authenticates a typed
function pointer with a discriminator derived from its prototype. Graft hands
back the original signed the way a `void *` holds a function pointer, and the
conversion re-signs it for the prototype. A typed pointer that Graft wrote
directly would fail authentication when called. On x86-64 the conversion
costs nothing.

### On x86-64

A route names a function in the kernel's own symbol table, such as
`_mac_vnode_check_open`, and the replacement and original have that function's
prototype. Graft replaces the function's first instructions with a jump to the
replacement, and moves them into a trampoline, which the original calls:

```cpp
using ExampleFunction = int(int); // the routed function's exact prototype

static void *originalExample;

static int replacementExample(int value) {
    // Act on your own case, and pass every other call through.
    return reinterpret_cast<ExampleFunction *>(originalExample)(value);
}

graft_route_request routes[] = {
    {"_example_kernel_function", reinterpret_cast<void *>(replacementExample),
     &originalExample},
};
if (!graft_route_kernel(Client, routes, 1))
    return; // GraftClient.ai.aneesiqbal.graft.HelloClient says why
```

[tests/kexts/core-probe](../../tests/kexts/README.md#core-probe) routes six
kernel functions this way.

### On arm64e

Kernel text and a MAC policy's table are read-only when Graft starts, so
Graft routes by shadowing the policy. A route names a MAC policy operation,
one of `file_check_mmap`, `file_check_library_validation` and
`vnode_check_signature`. The replacement and original have the operation's
`mpo_*_t` prototype from XNU's `security/mac_policy.h`, and the original is
the policy's handler for the operation. The operation is AMFI's by default.
The macOS SDK does not ship that header, so declare the prototype yourself:

```cpp
#include <sys/kauth.h>

struct fileglob;
struct label;

// mpo_file_check_mmap_t from XNU's security/mac_policy.h.
using FileCheckMmap = int(kauth_cred_t cred, struct fileglob *fg, struct label *label,
                          int prot, int flags, uint64_t file_pos, int *maxprot);

static void *originalMmap;

static int replacementMmap(kauth_cred_t cred, struct fileglob *fg, struct label *label,
                           int prot, int flags, uint64_t file_pos, int *maxprot) {
    // Decide your own case here, and pass every other call on to AMFI.
    auto original = reinterpret_cast<FileCheckMmap *>(originalMmap);
    return original(cred, fg, label, prot, flags, file_pos, maxprot);
}

graft_route_request routes[] = {
    {"file_check_mmap", reinterpret_cast<void *>(replacementMmap), &originalMmap},
};
if (!graft_route_kernel(Client, routes, 1))
    return; // GraftClient.ai.aneesiqbal.graft.HelloClient says why
```

To route another policy's operation, prefix the operation with the policy's
registered name and a colon. Graft shadows AMFI, the default, and the Sandbox
policy, `Sandbox`. The route is otherwise the same -- the same `mpo_*_t`
prototype, its own `original`, and that policy's own handler for every other
call:

```cpp
static void *sandboxOriginalMmap;

static int sandboxReplacementMmap(kauth_cred_t cred, struct fileglob *fg, struct label *label,
                                  int prot, int flags, uint64_t file_pos, int *maxprot) {
    // Decide your own case, then pass every other call on to the Sandbox policy.
    auto original = reinterpret_cast<FileCheckMmap *>(sandboxOriginalMmap);
    return original(cred, fg, label, prot, flags, file_pos, maxprot);
}

graft_route_request sandboxRoutes[] = {
    {"Sandbox:file_check_mmap", reinterpret_cast<void *>(sandboxReplacementMmap),
     &sandboxOriginalMmap},
};
if (!graft_route_kernel(Client, sandboxRoutes, 1))
    return;
```

A client that routes the same operation on two policies, as here, holds a
separate original for each. Every route in one `graft_route_kernel` call names
one policy; route a second policy in a second call.

[tests/kexts/route-probe](../../tests/kexts/README.md#route-probe) routes all
three AMFI operations, and the Sandbox policy's `file_check_mmap`.

## Limits

- One call installs at most sixteen routes, all of them or none, and names one
  policy. On x86-64 Graft has room for 32 routes per boot, shared first come
  first served: two of its own, and up to 30 more across every client. On arm64e
  each client routes at most one of each of the three operations per policy, and
  several clients route one after another; an operation on a policy that one
  client holds is refused to the next.
- On x86-64, routes must be installed before BSD starts, so install them from
  `graft_client_initialize()`. The arm64e routes go live with one atomic swap
  of the policy's list element, and are safe to install after BSD has started.
- On x86-64, Graft refuses a function that starts with `ENDBR64`, one that
  ends within the 14 bytes the jump needs, one whose first instructions it
  cannot decode or relocate, one with a displaced call that would return into
  the jump, and one that is already routed.
- On both architectures, Graft refuses a replacement outside the client's own
  executable code. It finds the client's image by the bundle identifier the
  call passes: its Boot collection entry on x86-64, or the firmware's
  device-tree node for it on arm64e. When it finds no image, it refuses the
  route, so a route's replacement is always the client's real placed code.
  [design.md](../design.md#routing) lists the rules.
- Graft passes each call through without reading its arguments, so the
  replacement's prototype must match the routed function's.
- Keep the replacement code and the original pointers in memory for the whole
  boot, even when routing fails: a rollback that cannot be verified leaves the
  route in place.

## 6. Build

```sh
make -C examples/hello-client                      # x86-64
make -C examples/hello-client ARCH='-arch arm64e'  # arm64e
```

The output is `build/GraftHelloClient.kext`, or
`build/GraftHelloClient-arm64e.kext`.

## 7. Put it in the boot image

On x86-64, add the client to OpenCore's `Kernel > Add` after Graft, since it
depends on Graft. Keep SIP on (`csr-active-config` 0). Graft needs no
boot-args. On arm64e, the boot firmware places it beside Graft.

## 8. Check that it worked

In the guest:

```sh
ioreg -l -d 1 | grep Graft
```

The example reads:

```
"GraftClient.ai.aneesiqbal.graft.HelloClient" = "ready: resolved _IOLog"
```

A client that routes reads `routed:` and the number of routes, or `refused:`
and the route and reason. Graft adds each state, and each line the client
logs with `graft_client_log`, to `GraftLog`:

```sh
ioreg -l -d 1 | sed -n '/"GraftLog" = /,/^"$/p'
```

On arm64e the same lines are in the unified log, prefixed `graft: `
([debugging-refusals.md](debugging-refusals.md#where-to-look) gives the
command); on x86-64 they are not. A line the client writes with `IOLog` itself
is missing from the unified log on both architectures, so `graft_client_log` is
the way to leave a line that can be read after boot.

`graft_route_kernel` returns false and records nothing when Graft is not
active or the request has no routes or more than sixteen.
[debugging-refusals.md](debugging-refusals.md) explains each refusal.
