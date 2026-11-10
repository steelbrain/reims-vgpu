# Changelog

Each release of Graft, newest first. A version is the kext's
`CFBundleVersion`. Graft declares `OSBundleCompatibleVersion` 0.1.0, so a kext
that lists `ai.aneesiqbal.graft` 0.1.0 in its `OSBundleLibraries` loads with
each later release that keeps it.

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
