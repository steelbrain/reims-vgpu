# built

This directory holds Graft's latest release, built and ready to install:

- `Graft.kext`: the x86-64 kext, for macOS 11 and later.
- `Graft-arm64e.kext`: the arm64e kext.
- `build.json`: what went into them. That covers the version, the SHA-256 of
  every source file the build reads (`Makefile`, `src/`, `sdk/include/`), the
  Xcode toolchain, and each bundle file's SHA-256, size and mode.

It changes only when a release is made, so between releases the sources
move on and this directory stays put.

## Checking a release

```sh
make check-built
```

The check takes the sources at the release's tag (`v` and the version, such as
`v1.0.0`) into a scratch directory, builds them with Xcode, and requires each
bundle it builds to match the one here byte for byte. It needs the toolchain
`build.json` names, because another Xcode produces different bytes. It
fetches nothing.

## Making a release

1. Set `VERSION` in the `Makefile` and add the release to `CHANGELOG.md`.
2. Run `make built` from a clean checkout. It builds from a snapshot of the
   source files on disk, committed or not, and rewrites this directory. It
   refuses a version that already has a tag.
3. Commit the version, the changelog and this directory together, and tag
   that commit `v` plus the version.
4. Run `make check-built` to confirm the tag rebuilds what you committed.

## Using a release in another repository

`tools/vendor.py DEST` copies the release from its tag into a consumer's
tree: this directory, `sdk/` and the license, plus a `source.toml` recording
the version, the commit and every file's SHA-256. See `sdk/README.md`.
