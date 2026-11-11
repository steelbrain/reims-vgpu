# Changelog

Each release of Graft, newest first. A version is the kext's
`CFBundleVersion`. Graft declares `OSBundleCompatibleVersion` 0.1.0, so a kext
that lists `ai.aneesiqbal.graft` 0.1.0 in its `OSBundleLibraries` loads with
each later release that keeps it.

## 0.2.0

Graft serves any kext, and names none of the kexts it serves. Release 0.1.0
bound one consumer and routed for one client, both named in `src/Policy.h`,
and opened the consumer's gate only on one PCI device, which the SDK named.

**Upgrading**

- Rebuild every bound consumer with this release's SDK. The manifest format is
  now version 4 for x86-64 and version 5 for arm64e, and Graft reads no other.
  It refuses a consumer built with an earlier SDK with the reason
  `consumer manifest version V is not one this Graft reads; rebuild the
  consumer with this release's SDK`.
- An owned client needs no rebuild. `Graft.h` is unchanged, and Graft still
  declares `OSBundleCompatibleVersion` 0.1.0, so a client that lists 0.1.0
  loads with this release.
- The arm64 boot firmware hands over nothing new.

**The kext**

- Binds any number of bound consumers, found by their manifest. At start
  Graft looks through every placed kext, the Boot collection's fileset entries
  on x86-64 and the firmware's `/chosen/ambrosia` nodes on arm64e, and keeps
  each one whose `__DATA,__graft` section starts with the manifest's magic.
  Each consumer binds, waits or is refused on its own, and reports under its
  own `GraftClient.<bundle id>`.
- Opens a consumer's activation gate on any provider its personalities match.
  The manifest carries each personality's provider matching, and Graft opens
  the gate on every nub it matches, present at the bind or published later. A
  consumer whose provider matching does not parse, or names no
  `IOProviderClass`, is refused before anything is written.
- Serves any placed kext as an owned client: a call is served when the bundle
  identifier it passes names a kext placed at boot. On both architectures a
  route's replacement must lie in the client's own placed code, and a client
  whose image Graft cannot find is refused.
- Route limits: up to 16 routes in one request, up from 6. On x86-64, 32 route
  slots are shared first come first served by every client and Graft's own two
  routes, up from 8. A function another client holds is refused, naming that
  client.
- On arm64e, several clients can route. Each request copies the current AMFI
  policy, AMFI's own or the newest shadow, and swaps the policy list's element
  to the copy with one compare-and-exchange. An operation an earlier client
  routed is refused to the next, naming that client.
- On arm64e, starts each library a consumer's manifest declares, with
  `OSKextLoadKextWithIdentifier`, before binding against it, and refuses the
  consumer if that start fails. `Graft-arm64e.kext` no longer lists a display
  family in its `OSBundleLibraries`.

**The SDK**

- `graft-kext` records each personality's provider matching in the manifest:
  the personality less `IOClass`, `CFBundleIdentifier`, `IOProbeScore` and
  `IOMatchCategory`. It requires each personality to name an
  `IOProviderClass`, and no longer requires one PCI device. It refuses a
  personality whose `IOPropertyMatch` already holds the gate's key, as a
  staged Info.plist's do, or whose `IOPropertyExistsMatch` names it.

**Examples and tests**

- The examples and the test kexts carry bundle identifiers of their own.
  `examples/hello-consumer` matches `IOResources`, in an `IOMatchCategory` of
  its own, and `tests/kexts/gate-probe` matches the PCI host bridge by its
  class, so both run in a guest without a particular device. The host test
  fixtures match device classes.

**Tools**

- `make lint` checks that no file under `src/` or `sdk/` names a bundle
  identifier other than Graft's own and Apple's.
- `hook-symbols.txt` lists only the kernel symbols Graft itself uses, and
  `hook-symbols.py --symbols FILE` checks a client's own list.
- `boot-core.py` boots any number of consumers and clients, given as
  `--consumer KEXT STATE` and `--client KEXT STATE`, and reads the guest's
  kernel log with `/usr/bin/log`. `tests/kexts/gate-probe` records its results
  as `GraftGateProbe.<category>`.

**Documentation**

- The guides read Graft's steps from `GraftLog` in the IORegistry. On x86-64
  the unified log keeps none of the lines an OpenCore-injected kext writes with
  `IOLog`, Graft's included. On arm64e it keeps Graft's lines but not a
  client's own `IOLog` lines, so a client logs through `graft_client_log`.

## 0.1.0

The first release.

**The kext**

- One binary per architecture, x86-64 and arm64e, for every macOS release it
  supports. Graft checks no macOS version: it decides from the running
  system's symbols, class sizes, vtable slots and IOKit answers, and refuses
  by name in its log and in the IORegistry when something is missing.
- Starts from its own kmod start routine, with SIP on and no boot-args.
- Binds one bound consumer against the System kernel collection by symbol
  name, from the manifest its build embeds: imports by candidate name or
  fallback, inherited vtable slots from the live parent, every write in one
  transaction. It then opens the consumer's activation gate on every matching
  device nub, including one published after the bind.
- On arm64e, builds a consumer's vtable at bind time for a parent class that
  has no header, in the running release's slot order, signing each slot for
  its address.
- Routes kernel functions for one owned client: kernel-text trampolines on
  x86-64, before BSD starts, and a shadow of AMFI's MAC policy table on
  arm64e, installed by one atomic swap. The client keeps each original in a
  `void *` and converts it at the call on both architectures.
- Publishes `GraftState`, `GraftReason`, `GraftLog` and
  `GraftClient.<bundle id>` on the IORegistry root.

**The SDK**

- `Graft.h` and `Start.cpp` for an owned client.
- `consumer.mk` and `graft-kext`, a C++17 command built from source by the
  consumer's build, which probes class layouts, writes and validates the
  manifest, stages the dormant consumer, and checks a staged consumer in a
  collection OpenCore prelinked. A consumer's build needs Xcode and nothing
  else.

**Releases**

- A release is the tagged source tree, every file the repository tracks at
  the tag, plus both kexts built from it. CI builds it on macOS with the Xcode
  `.xcode-version` names and publishes it as `graft-<version>.zip` on a pushed
  tag. `tools/vendor.py` copies a release zip into another repository after
  checking it against its `SHA256SUMS`.
