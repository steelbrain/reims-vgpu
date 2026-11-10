# Agents working on Graft

Graft is a macOS kext that binds and routes for a fixed set of kexts inside
macOS virtual machines, with System Integrity Protection on. Read
[README.md](README.md) for what it does, [CONTRIBUTING.md](CONTRIBUTING.md)
for how to build, test and change it, and [docs/](docs/README.md) for how it
works. Everything in CONTRIBUTING.md applies to agents; this file adds what an
agent most often gets wrong.

## Before you commit

Run every check, from a clean build when a change touches the build:

```sh
make test kernel-check kernel-check-arm64e graft-arm64e core-probe route-probe examples lint
```

A commit builds and passes these. Never loosen a check to make it pass: fix
the code, or fix the expectation and say why in the commit.

## Rules

- **Guest policy.** Graft runs with SIP on and needs no boot-arg. It decides
  from what the running system has (symbols, class sizes, vtable slots, IOKit
  answers) and never from a macOS version. When something is missing, it
  refuses by name and the guest keeps booting.
- **Provenance.** Nothing taken from Apple's binaries or from a guest enters
  the repository: no disassembly, dumps, excerpts, logs, or per-release
  offsets, sizes, slot counts or discriminators. A constant Graft needs is
  found in the running system or computed from Apple's published headers.
- **One project name stays out.** Never write the name of the NVIDIA macOS
  driver project the maintainer has ruled out, in code, comments, docs or file
  names.
- **Keep it simple.** Handle what the SDK and the supported boot paths
  produce. Add no option, switch or environment variable without a concrete
  use, and leave no dead or commented-out code.
- **Keep the docs true.** Change README.md, docs/ and the module READMEs in
  the same commit as the code they describe. History goes in git and
  CHANGELOG.md, not in the docs.
- **Fix what you find.** A failure you meet is yours to fix, or to report with
  its cause if it belongs to another repository.

## Git

- Work on a branch named `steelbrain/<topic>` off `development`. Never commit
  to, merge into or reset `master`.
- Land by rebasing onto `development` and fast-forwarding it. Never create a
  merge commit.
- Stage explicit paths. Do not use `git stash`. Do not push unless the
  maintainer asks.
- Commit messages follow CONTRIBUTING.md: an emoji code, an imperative
  subject of at most 72 characters, and a body wrapped at 80 that says why.

## Releases

[docs/guides/releasing.md](docs/guides/releasing.md) gives the steps. A
release is cut by tagging; CI builds and publishes it.
