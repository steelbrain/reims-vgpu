# Cut a release

This guide is for Graft's maintainers. A release is a commit that sets the
version and adds its changelog entry, tagged `v` plus the version. CI builds
the release from the tag and publishes it as a GitHub release. Nothing built
is committed.

## What a release holds

A release is the source tree at its commit and both kexts built from it.
`make release` writes `dist/graft-<version>/`, where `<version>` is the
Makefile's `VERSION`:

- every file `git ls-files` lists, as it is in the repository, so a consumer
  holds the exact source of the kexts it ships and every link in the docs
  resolves in its copy;
- at the top, `Graft.kext`, the x86-64 kext, and `Graft-arm64e.kext`, the
  arm64e kext;
- `COMMIT`, the commit the release was built from;
- `SHA256SUMS`, the SHA-256 of every other file, which
  `shasum -a 256 -c SHA256SUMS` checks.

A tracked file at one of those four names, in any case, is refused, since the
release would replace it.

It also writes `dist/graft-<version>.zip`, which holds that folder and is what
the GitHub release attaches. The zip is the same bytes every time from the
same commit and Xcode: its entries are stored rather than compressed, in name
order, with the commit's time, and with mode 755 for each kext's binary and
each file git tracks as executable and 644 for every other file. `dist/` is
not committed.

## The pinned Xcode

`.xcode-version`, at the top of the tree, names the Xcode a release is built
with by its version and build, as Xcode's About window shows them, such as
`27.2 (27B52)`. `make release` compares it with the version and build that
`xcodebuild -version` prints, and refuses any other Xcode, since another
Xcode builds other bytes. A beta and the release of one version share the
version and differ in the build. Building is deterministic, so a release CI
publishes holds the same kexts as a `make release` on your Mac with that Xcode.

CI selects `/Applications/Xcode_<version>.app` on its runner, and fails,
naming the version, on a runner that lacks it; `make release` then fails if
that Xcode is another build. To move to another Xcode, change
`.xcode-version`, and change `runs-on` in `.github/workflows/ci.yml` to a
GitHub-hosted image that carries that Xcode.

## Steps

1. Set `VERSION` in the `Makefile`.

2. Decide whether existing clients can load with the release. If it changes
   `graft/Graft.h` in a way a client built against the previous release cannot
   use, set `OSBundleCompatibleVersion` in `src/Info.plist` to the new version.
   Clients that list an older version then refuse to load with it, rather than
   call an interface that changed.

3. Add the release to [CHANGELOG.md](../../CHANGELOG.md), under a heading of
   `## ` and the version, alone or followed by a space and a title or date:
   what changed, and why for a change in behaviour. Say whether consumers must
   be rebuilt, and whether the arm64 boot firmware must hand over anything new.
   The section, up to the next `## ` heading, becomes the release's notes.

4. Commit those changes, and wait for CI to pass on the commit.

5. Tag the commit `v` plus the version, such as `v0.1.0`, and push the tag:

   ```sh
   git tag v0.1.0
   git push origin v0.1.0
   ```

CI runs on the tag as on every push: the checks, then `make release`. Its
release job then:

- refuses a tag that is not `v` and the Makefile's `VERSION`, and a release
  whose `COMMIT` is not the tag's commit;
- takes the changelog's section for the version as the notes, and refuses a
  version with no section;
- creates the GitHub release `Graft <version>` for the tag, with
  `graft-<version>.zip` attached. If the release exists already, it downloads
  the release's zip and compares it with the one it built: it leaves the
  release as it is when the two are the same bytes, and fails, replacing
  nothing, when they differ.

## Build a release on your Mac

From a clean checkout, with the Xcode that `.xcode-version` names selected
(`xcode-select -s`):

```sh
make release
```

It refuses a tree that differs from `HEAD`: an edit to any tracked file, or
an untracked file that git does not ignore. Ignored files, such as `build/`,
`dist/` or a `.DS_Store`, are allowed, and never carried. It copies the
tracked files into a scratch directory and runs `make all graft-arm64e`
there, with variables that would change the build, such as `MAKEFLAGS`,
`SDKROOT` and `TOOLCHAINS`, removed from the environment. It prints the zip's
name and SHA-256.

Run in a checkout of a release's tag, it builds that release's kexts again, so
a consumer or a maintainer can check a published release against its sources
([taking-a-release.md](taking-a-release.md#checking-a-releases-binaries)).

## When it fails

- `make release` says the sources differ from `HEAD`: commit or remove the
  files it names, then run it again.
- `make release` says a tracked file collides with the release's own names:
  rename the file it names.
- `make release` names another Xcode: select the one `.xcode-version` names,
  that version and build, with `xcode-select -s`.
- `make release` says `xcodebuild -version` names no Xcode: only the Command
  Line Tools are selected. Install Xcode and select it.
- CI's "Select the pinned Xcode" step fails: the runner's image does not carry
  that Xcode. Move `runs-on` to an image that does.
- CI's "Release" step names another Xcode of the pinned version: the runner's
  image carries another build of it. Move `runs-on` to an image that carries
  the pinned build.
- CI's release job refuses the tag, or finds no changelog section: no release
  was created. Delete the tag, here and on GitHub, fix the cause in a new
  commit, and tag that commit.
- CI's release job says the release holds another zip: a run for the tag has
  published a zip whose bytes differ from this run's, such as one built with
  another Xcode build. Compare the two zips' `SHA256SUMS` to find the files
  that differ. To publish this run's zip instead, delete the release on GitHub
  and run the job again.

## A release that changes only the SDK

A release can change only what consumers build with, such as
`sdk/consumer.mk` or a stand-in header the kext does not include. Both kexts
then differ from the previous release's only in the version they carry, and a
client that lists the previous compatible version loads with them. The steps
are the same, and the changelog says that the kexts did not change.
