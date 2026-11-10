# docs

## How Graft works

| document | covers |
|---|---|
| [design.md](design.md) | where Graft runs, its start, owned clients and routing, binding a consumer, and what it reports |
| [consumers.md](consumers.md) | the two kinds of kext Graft serves, how each is built, and how it goes into the boot image |
| [manifest.md](manifest.md) | the manifest a bound consumer embeds, field by field, and how Graft validates and uses it |
| [lessons-learned.md](lessons-learned.md) | what went wrong while Graft was built, why, and what to do instead |

## Guides

| guide | for |
|---|---|
| [Write a bound consumer](guides/bound-consumer.md) | a kext that derives from or calls into the System kernel collection |
| [Write an owned client](guides/owned-client.md) | a kext that looks up kernel symbols and routes kernel functions through Graft |
| [Derive from a class with no header](guides/synthesized-class.md) | an arm64e consumer whose parent has no Apple header and a vtable that changes between releases |
| [Find out why Graft refused](guides/debugging-refusals.md) | reading Graft's state and every refusal reason |
| [Take a Graft release into your project](guides/taking-a-release.md) | copying a release with its record and putting it in a boot image |
| [Cut a release](guides/releasing.md) | Graft's maintainers |

## Reference next to the code

- [src/README.md](../src/README.md) maps the kext's source and lists what the
  arm64 boot firmware must provide.
- [sdk/README.md](../sdk/README.md) documents the SDK's files, `graft.json`
  and the build rules.
- [tests/README.md](../tests/README.md) lists the host tests and
  [tests/kexts/README.md](../tests/kexts/README.md) the test kexts.
- [tools/README.md](../tools/README.md) describes `vendor.py` and the
  maintainers' tools.
- [CONTRIBUTING.md](../CONTRIBUTING.md) covers building, the checks and the
  rules for changes.
