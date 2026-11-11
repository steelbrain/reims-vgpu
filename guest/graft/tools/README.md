# tools

Scripts for working with Graft from a checkout. `vendor.py` is for a project
that takes Graft's releases. The scripts in `internal/` are for Graft's
maintainers. What a kext builds with is in [sdk/](../sdk/README.md) instead.

Each script needs Python 3.9 or later with only its standard library, such as
macOS's own `/usr/bin/python3`, and is run with `python3 -I -B`.

| script | does | run it with |
|---|---|---|
| `vendor.py` | checks a release zip against its `SHA256SUMS` and copies the release, Graft's source tree and both kexts, into another repository, with a record of every file | `python3 -I -B tools/vendor.py graft-<version>.zip DEST` |
| `internal/release.py` | builds a release folder and zip into `dist/` | `make release` |
| `internal/lint.py` | checks SPDX license lines, text hygiene, Markdown links, and that `src/` and `sdk/` name no kext but Graft and Apple's | `make lint` |
| `internal/hook-symbols.py` | reports which kernel symbols Graft uses, or those another list names, are defined in a macOS release | `python3 -I -B tools/internal/hook-symbols.py RELEASE...` |
| `internal/boot-core.py` | boots Graft and the test kexts in a disposable x86-64 guest, and checks the result | `python3 -I -B tools/internal/boot-core.py BASE --env DIR --gpu NAME --account FILE` |

## vendor.py

`vendor.py ZIP DEST` takes a release zip, `graft-<version>.zip`, as Graft's
GitHub release attaches it. It reads the whole zip before it writes anything,
and refuses it unless:

- every member is a regular file inside one folder named `graft-<version>`,
  with no link, directory entry, absolute name, or empty, `.` or `..` part;
- every file matches its SHA-256 in `SHA256SUMS`, which lists every other file
  and nothing more;
- `COMMIT` names a commit;
- the release holds no `source.toml` of its own, which the record would
  replace.

It writes the release's folder into `DEST` as it is, but for `COMMIT` and
`SHA256SUMS`: Graft's source tree at the release's commit and both kexts. A
file the zip marks executable gets mode 755, and every other file 644. It
writes `DEST/source.toml` with the version, the commit and each file's
SHA-256, in the `key = "value"` subset of TOML. It copies beside `DEST` and
swaps the copy in only once it is complete, so a failed copy leaves the
earlier one in place. It refuses a `DEST` that exists without an earlier
record. It fetches nothing.
[docs/guides/taking-a-release.md](../docs/guides/taking-a-release.md) shows how
a project uses it.

## internal/release.py

`release.py` makes a release: the source tree at a commit and both kexts built
from it. It refuses a tree that differs from `HEAD`, by an edit to a tracked
file or an untracked file git does not ignore, and any Xcode but the one
`.xcode-version` names, by the version and build `xcodebuild -version` prints.
It takes the files `git ls-files` lists, never walking the tree, so no ignored
file rides along, and refuses a tracked link or submodule. It copies them into
a scratch directory and runs `make all graft-arm64e` there, with variables
that would change the build removed from the environment: make's flags, the
compiler's include paths and `CCC_OVERRIDE_OPTIONS`, and `SDKROOT`,
`TOOLCHAINS` and `MACOSX_DEPLOYMENT_TARGET`. It writes
`dist/graft-<version>/`, where `<version>` is the Makefile's `VERSION`:

- every tracked file, as it is in the repository;
- `Graft.kext` (x86-64) and `Graft-arm64e.kext` (arm64e), at the top;
- `COMMIT`, the commit it is built from;
- `SHA256SUMS`, the SHA-256 of every other file, in the form
  `shasum -a 256 -c` reads.

It refuses a tracked file whose path starts with one of those four names, in
any case. Each kext's binary and each file git tracks as executable has mode
755, and every other file 644. It then writes
`dist/graft-<version>.zip`, which holds that folder. The zip is the same bytes
every time from the same commit and Xcode: its entries are stored rather than
compressed, in name order, with those modes and the commit's time. It prints
the zip's name and SHA-256.

[docs/guides/releasing.md](../docs/guides/releasing.md) walks through a
release.

## internal/lint.py

`lint.py` checks what clang-format and ruff do not, in the files git tracks
plus new files git would track:

- every C, C++, Objective-C, Python, make and shell file, and every script
  that starts with `#!`, has the SPDX line in its first three lines;
- every text file ends in exactly one newline, with no trailing whitespace,
  carriage returns or tab indentation (make files may use tabs);
- every relative Markdown link names a file or directory that exists, and
  every `#fragment` a heading in its target;
- no file under `src/` or `sdk/` names a bundle identifier other than Graft's
  own, `ai.aneesiqbal.graft`, Apple's, `com.apple.*`, or one under a domain
  reserved for examples, such as `com.example.*`. An identifier is a top-level
  domain followed by two or more labels, so a file name, a version or other
  dotted text is not one. In a C, C++ or Objective-C file the check reads only
  comments and string and character literals, since a member access in code can
  look like an identifier; it reads any other file whole.

The license texts in `LICENSES/` are exempt from the text checks, since they
are kept exactly as published.

## internal/hook-symbols.py

`hook-symbols.txt`, next to the script, lists every kernel symbol Graft uses,
how each is reached, and why. `hook-symbols.py` checks a macOS release for each
of them:

```sh
python3 -I -B tools/internal/hook-symbols.py RELEASE...
python3 -I -B tools/internal/hook-symbols.py --all CACHE [--json]
```

A client project checks the kernel symbols it uses with `--symbols FILE`, where
`FILE` lists them in the same form: each symbol is the first word of its line,
and a line that starts with `#` is a comment.

A release is a directory that holds that release's x86-64 `kernel`,
`BootKernelExtensions.kc` and `SystemKernelExtensions.kc`. With `--all`, it
checks every directory under `CACHE` named `<version>-<build>`. For each
symbol it prints whether the kernel and each collection defines it, references
it, or lacks it. It reads the symbol tables in place and keeps nothing from
them but the names it was asked about and that answer.

It exits 1 when an input is incomplete or unreadable or the list names no
symbol, 2 when a release defines a wanted symbol in none of its images, and 0
otherwise.

## internal/boot-core.py

`boot-core.py` checks on a real boot what the host tests cannot: that Graft
starts, routes and binds in an x86-64 guest with SIP on. It runs on a Linux
host with reims-vgpu's VM tool and its QEMU installed under `DIR`, and boots on
reims-vgpu's paravirtual GPU, which renders on the host GPU `NAME`.

```sh
python3 -I -B tools/internal/boot-core.py BASE --env DIR --gpu NAME --account FILE \
    [--consumer KEXT STATE]... [--client KEXT STATE]...
```

`BASE` is an installed x86-64 macOS guest, with its OpenCore disk, which the
script never boots in place. `FILE` holds the guest's admin account name on its
first line and its password on the second.

Each `--consumer` names a bound consumer to inject and each `--client` a
client, with any number of each. `STATE` is what Graft's report on that kext,
`GraftClient.<bundle id>`, must begin with: a state such as `bound`, `refused`,
`routed` or `ready`, or more of the report, such as a refusal's reason. The
script reads each kext's bundle identifier and executable from its
`Info.plist`, and refuses two kexts with one folder name or one bundle
identifier. One boot of both consumers and both clients:

```sh
python3 -I -B tools/internal/boot-core.py BASE --env DIR --gpu NAME --account FILE \
    --consumer build/GraftHelloConsumer.kext bound \
    --consumer build/GraftGateProbe.kext bound \
    --client build/GraftCoreProbe.kext routed \
    --client build/GraftHelloClient.kext ready
```

The script:

1. copies `BASE`, refusing a source another process can write;
2. replaces the copy's injected kexts with `build/Graft.kext` and the kexts
   given, clears its boot-args and `csr-active-config`, and checks that the
   injected Graft is the one built here;
3. boots the copy headless, logs in if it stops at the login window, and waits
   for the desktop;
4. reads the registry until each kext's report is as expected or refused, for
   up to 145 seconds, since a consumer's report changes as its providers load
   and its gate opens;
5. requires `GraftState` `active`, each kext's report to begin with its
   `STATE`, a consumer given `bound` to have its activation gate open, each
   test kext's own state `passed` unless its `STATE` is a refusal, SIP on,
   `csr-active-config` 0, no boot-args, and no `reims-vgpu: fatal` line in
   QEMU's log on the host.

The test kexts' own states are core-probe's `GraftProbeState`,
hello-consumer's `GraftHelloConsumerState`, and the key each of gate-probe's
personalities sets, named by its `GraftGateCategory`.

It stops and deletes the copy even when a check fails. Results, screenshots and
logs stay in `build/boot-core`. Graft's steps are in `GraftLog`, in
`registry.json`: the unified log of an x86-64 guest keeps no line an injected
kext writes ([debugging-refusals.md](../docs/guides/debugging-refusals.md#where-to-look)).
`kernel-log.txt` keeps the kernel's own lines that name Graft or a given kext,
such as a failed load. The script reads them with `/usr/bin/log`, since the
guest's login shell, zsh, has a `log` builtin of its own, and fails the run when
it cannot read them.

[tests/kexts/](../tests/kexts/README.md) describes the test kexts.
