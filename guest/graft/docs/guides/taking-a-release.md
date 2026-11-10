# Take a Graft release into your project

This guide is for a project that builds kexts against Graft and puts Graft in
its boot image. It copies a release into your tree with a record you can
check, and puts the release in the image.

## What a release is

Each release is a GitHub release of Graft's repository, titled `Graft` and the
version, for the tag `v` and the version. A release is the tagged source tree
and both kexts built from it. CI builds it from the tag's commit
([releasing.md](releasing.md)) and attaches one file, `graft-<version>.zip`.
The zip holds one folder, `graft-<version>/`, with:

- every file Graft's repository tracks at the tag, as it is in the
  repository: among them `sdk/`, what your kexts build with, `docs/`, the
  kexts' source in `src/`, `tools/vendor.py`, and the license in `LICENSE`
  and `LICENSES/`;
- `Graft.kext`, the x86-64 kext, for macOS 11 and later;
- `Graft-arm64e.kext`, the arm64e kext;
- `COMMIT`, the commit the release was built from;
- `SHA256SUMS`, the SHA-256 of every other file in the folder.

So your copy holds the exact source of the kexts you ship, and every link in
its documentation resolves inside it.

The release's notes are its section of [CHANGELOG.md](../../CHANGELOG.md),
which says what each release changed.

## What you need

- `tools/vendor.py`, from any Graft checkout, from the zip, or from a copy
  `vendor.py` made earlier (`DEST/tools/vendor.py`). The script reads only the
  zip, so any copy will do.
- Python 3.9 or later with only its standard library. macOS's own
  `/usr/bin/python3` will do.

## 1. Download the release

Download `graft-<version>.zip` from the release's page on GitHub.

## 2. Check it and copy it into your tree

From the Graft checkout:

```sh
python3 -I -B tools/vendor.py path/to/graft-<version>.zip DEST
```

`vendor.py` reads the whole zip and checks it before it writes anything. Every
file must match its line in the zip's `SHA256SUMS`, and `SHA256SUMS` must list
every other file and nothing more. It refuses a zip that holds a link, a
directory entry, or a name outside its one `graft-<version>/` folder. It
fetches nothing.

It writes the folder into `DEST` as it is in the zip, but for `COMMIT` and
`SHA256SUMS`: the source tree and both kexts. In their place it writes
`DEST/source.toml`, which records the version, the commit and each file's
SHA-256. It prints one line naming `DEST`, the version, the number of files and
the commit.

`vendor.py` replaces `DEST` whole, and swaps the new copy in only once it is
complete, so a failed copy leaves the earlier one in place. It refuses a
`DEST` that exists but holds no `source.toml`, so it never overwrites a
directory it did not write. To move to a later release, run it again with the
same `DEST`.

To check the zip by hand instead, unzip it and run `shasum -a 256 -c
SHA256SUMS` inside `graft-<version>/`.

Commit `DEST` as it is. Your build can check each file against `source.toml`
to catch a copy that drifted from its record.

## 3. Depend on the right version

Graft's bundle identifier is `ai.aneesiqbal.graft`, and release 0.1.0 declares
`OSBundleCompatibleVersion` `0.1.0`. An owned client lists
`ai.aneesiqbal.graft` at `0.1.0` in its `OSBundleLibraries`, and then loads
with this release and with any later one that keeps that compatible version. A
bound consumer calls nothing in Graft and needs no such entry.

## 4. Build your kexts against the SDK

An owned client compiles against `DEST/sdk/include` and `DEST/sdk/Start.cpp`
([owned-client.md](owned-client.md)). A bound consumer includes
`DEST/sdk/consumer.mk` ([bound-consumer.md](bound-consumer.md)). Rebuild them
when you take a release whose changelog says the SDK or the manifest format
changed.

## 5. Put Graft in the boot image

On x86-64, add `Graft.kext` to OpenCore's `Kernel > Add`, before the kexts it
serves: OpenCore needs a kext's dependencies listed above it. Keep SIP on
(`csr-active-config` 0). Graft needs no boot-args, and no Lilu.

On arm64e, have the boot firmware place `Graft-arm64e.kext` with your kexts,
and publish what Graft reads from the device tree
([src/README.md](../../src/README.md#what-the-arm64-boot-firmware-provides)).

If your build produces the x86-64 kernel collection with your kexts linked into
it as OpenCore links them, the SDK's command
`graft-kext check-prelinked COLLECTION KEXT...` checks that each staged
consumer is still bindable there. It prints a line per consumer and exits 0
when all pass, or prints `prelink:` and the first defect and exits 1.
[sdk/README.md](../../sdk/README.md#commands) describes it.

## 6. Check that Graft started

In a booted guest, `ioreg -l -d 1 | grep Graft` shows `GraftState` reading
`active`, and one `GraftClient` entry for each kext Graft serves.
[debugging-refusals.md](debugging-refusals.md) explains any other state.

## Checking a release's binaries

`SHA256SUMS` shows that the files arrived as CI wrote them. To check that the
kexts are what the tag's sources build, build the release yourself and compare
the two `SHA256SUMS`. In a clean Graft checkout of the tag, with the Xcode its
`.xcode-version` names selected:

```sh
make release
unzip -p path/to/graft-<version>.zip graft-<version>/SHA256SUMS |
    diff - dist/graft-<version>/SHA256SUMS
```

`make release` refuses any other Xcode, since another Xcode builds other bytes
([releasing.md](releasing.md)). With that Xcode it builds the kexts CI built, so
`diff` prints nothing.
