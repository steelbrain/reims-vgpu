# examples

Two minimal kexts, one of each kind Graft serves. The guides walk through
them: [Write a bound consumer](../docs/guides/bound-consumer.md) and
[Write an owned client](../docs/guides/owned-client.md).
[docs/consumers.md](../docs/consumers.md) explains the two kinds.

| example | kind | what it shows |
|---|---|---|
| [hello-consumer](hello-consumer/README.md) | bound consumer | a class derived from a System-collection class, bound by Graft at boot from its manifest |
| [hello-client](hello-client/README.md) | owned client | a kext that calls Graft's C interface to look up a kernel symbol and publish its state |

Each has a bundle identifier of its own, and Graft names neither. Graft binds
the consumer, `ai.aneesiqbal.graft.HelloConsumer`, because it carries a
manifest, and serves the client, `ai.aneesiqbal.graft.HelloClient`, because
the identifier its calls pass names a placed kext. A kext of yours uses its own
identifier in the same places. Both match `IOResources`, so they run in any
guest, without a device of their own.

Each has its own Makefile and builds for x86-64 by default, or for arm64e with
`make ARCH='-arch arm64e'`. From the repository root, `make examples` builds
both for both architectures.
