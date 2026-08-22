# Reims vGPU architecture

Reims has one semantic device model and one Vulkan executor across both supported pathways. The
PCI/MMIO attach mechanism, guest page geometry, mapper availability, host-pointer import, and memory
topology differ; guest-visible resource identity, content authority, synchronization, and command
meaning do not.

## The seam

```text
guest bytes
  -> checked wire view
  -> semantic command and typed guest identities
  -> task namespace and generational resource graph
  -> immutable resolved submission
  -> Vulkan placement/transfer plan and execution
  -> typed completion fact
  -> semantic lifecycle/content/display transition
  -> guest-visible completion
```

No layer may skip across this chain to reconstruct an answer owned by another layer. In particular,
a memory optimization does not define lifetime or coherency, and a QEMU shim does not combine
semantic queries into a product rule.

## Ownership by crate

| Crate | Owns | Must not own |
|---|---|---|
| `reims-vgpu-wire` | Borrowed, checked wire views and record framing | Semantic defaults, allocation, lifecycle, or execution |
| `reims-vgpu-protocol` | Decoded enums/descriptors, typed guest identities, protocol refusals | Device state, Vulkan types, or host policy |
| `reims-vgpu-paging` | Page-table interpretation, span walks, GPA-run and window planning | Host mappings or resource lifetime |
| `reims-vgpu-memory` | Bounded guest-memory runs, slices, destinations, and transfer-plan vocabulary | Vulkan allocation policy or content authority |
| `reims-vgpu-core` | Task namespaces, generational resource graph, lifecycle/content authority, immutable command envelopes, executor ports, synchronization, and presentation semantics | QEMU, Vulkan handles/formats, environment policy, or native shader payloads |
| `reims-vgpu-vulkan` | Capability discovery, structural topology classification, placement/batching policy, translated/native shaders, GPU objects, submissions, residents, and per-device sessions | Guest object lifetime or an alternative content model |
| `reims-vgpu` | Decode orchestration, composition-owned adapters, device scheduling, QEMU ABI, and failure projection | A second semantic vocabulary or direct engine path beside the executor |
| `reims-vgpu-observe` | Shared typed emission and measurement support | Behavior-selecting state |
| `reims-vgpu-config` | Operator configuration names and parsing | Host capability or guest semantics |
| `vendor/qemu` | QOM attach, PCI/MMIO, IRQ, console/input, and host-memory plumbing | Protocol, resource, topology, or presentation policy |

The shipping artifact remains the `reims-vgpu` static library linked into QEMU. The crate is a
composition root, not the owner of every concern it links.

## Identity and lifetime

A numeric value can occur in several independent namespaces. These are never interchangeable:

- task-local `ObjectTableRef` and serializer references;
- generational `ResourceId`;
- storage/backing and view identities;
- `SurfaceId` and page-table `MappingId`;
- 64-bit `MapperSurfaceRef` and mapper-resolved surface identity;
- content, backing, mapping, and resident generations.

Task/object references are reusable wire names. A durable execution, residency, Store/gather
witness, or content-authority entry must resolve them to the canonical `ResourceId` first. Raw
names remain legitimate at the byte decoder, task namespace, pre-construction currency, and a
documented compatibility adapter which immediately projects them from or into the graph. They are
not executor identity.

Deleting a view does not imply deleting its storage. Replacing physical backing advances backing
state without manufacturing a new object lifetime. Task deletion, object deletion, mapping release,
backing retirement, host-materialization release, and display retirement are distinct typed
effects, even when one guest packet triggers several of them.

Mapper-backed arm resources share the semantic view/backing distinction with registered x86
resources, but not their construction, coherency, paging, discard, or teardown implementation.
Task death with live mapper views, live reset, interrupted queued teardown, and display-versus-
ordinary retirement remain arm-specific validation questions; shared code must not guess them.

A texture view is constructed over its immediate base view. Nested mip offsets and swizzles compose;
pixel format inherits only when the outer view leaves it unspecified. A sampled source carries the
complete allocation mip chain separately from its view range. The same representation covers 1D,
1D-array, 2D, 2D-array, and 3D base textures and multi-level views. Array storage remains
slice-major, using the declared full-chain slice pitch between equal mips; volume depth remains
inside each mip with its own depth pitch. Direct import checks every mip. If it declines, the
copy-backed image is built with the view's complete mip count and copies each declared mip and
array layer or depth plane from the same allocation description. Cube ranges remain outside this
rail until their face/slice view contract is represented; ranges are never replaced with mip or
slice zero, and empty ranges are never widened to one.

## Commands, execution, and completion

Decoded operations are normalized into immutable core commands. A resolved command contains
generational resources, typed surfaces/mappings, semantic descriptors, byte windows, and declared
access—not raw object tags, unresolved task-local references, Vulkan handles, or SPIR-V/native
payloads.

`metal2vulkan` owns shader ABI construction. Reims selects the complete per-stage descriptor layout,
fragment raster-sample count, runtime pixel-coordinate sampler state, and runtime storage-image
format/capabilities before translation, then consumes the effective layout and specialization
results, resource access and byte footprints, and vertex stage-in declarations from the returned
reflection. It does not renumber descriptors, repair aggregate layout, retype storage images, or
inject capabilities into the emitted SPIR-V. The remaining SPIR-V reads answer executable-module
questions reflection does not claim to answer, such as static descriptor use, required Vulkan
capabilities, validator acceptance, and the exact storage-image operations that determine
read/write access.

Null texture and sampler entries remain explicit semantic resources. A capable Vulkan executor
binds null descriptors; an executor without that capability refuses by type. Neither decode nor
execution may replace a null entry with fabricated image dimensions, texels, or sampler state.

`ResolvedSubmission` preserves command-buffer segmentation and resource participation. The
executor returns typed completion values; it does not mutate request fields to publish success.
Only a successful completion fact may advance content versions, Store authority, synchronization,
or presentation state.

The product `Executor` composes narrow capability, translation, residency, transfer, execution,
maintenance, presentation, and session services. `VulkanExecutor` is its implementation. Legacy
host APIs may require a task-local reference; recover it from the resource graph at that final
adapter and fail by type if the resource has retired.

## Content authority

The canonical content model records which version exists in guest pages, host replicas, and GPU
residents. Addresses, page-set hashes, cache hits, upload counters, and native allocation IDs are
evidence used to execute a plan; none substitutes for `(ResourceId, ContentVersion)`.

Guest writes, GPU Stores, synchronization, discard, readback, and delayed completion are explicit
transitions. A delayed completion cannot overwrite a newer guest write. Synchronization pays only
the named resource/subresource obligation. State whose eviction would lose the only copy is tied to
the guest lifetime and has no invented capacity; bounded caches may retain only recomputable data.

## Topology and sessions

`reims-vgpu-vulkan::memory` classifies unified versus discrete memory from structural heap/type
properties. `reims-vgpu-vulkan::policy` may select memory requests and batching defaults. Host-
pointer import is an orthogonal measured capability. All four combinations must preserve the same
semantic trace:

| | Import available | No import |
|---|---|---|
| Unified | Directly bind eligible guest memory | Copy through unified-host staging |
| Discrete | Import as backing and copy GPU-side into working memory | Stage every crossing |

Topology may change placement, transfer scheduling, and metrics. It must not change resource
lifetime, correctness, refusal meaning, or guest-visible output. Vendor and driver names are not
policy inputs.

Direct image binding is a layout contract, not a topology shortcut. The semantic request carries
the canonical guest allocation, every mip's resource/plane offset, row pitch and typed extent, the
separate view range, array layers or volume depth, inter-subresource pitch, and physical footprint.
The Vulkan executor admits it only when the imported parent allocation, memory type, image
requirements, and every subresource offset, row pitch, and—where applicable—array or depth pitch
agree exactly. This permits exact 1D,
1D-array, 2D, 2D-array, and 3D imports without treating a layer and a depth slice as the same
thing. A mismatch keeps the typed GPU buffer-to-image transfer path; it never invents a pitch or
falls back to a CPU layout repair. The imported image is a child of that allocation and delays
parent retirement until its submission fence and image lifetime have both ended. Image windows
remain parent-import-relative while physical footprints are resource-relative;
`GuestTargetMemory` owns that coordinate conversion. Attachments publish the written footprint.
Before any imported buffer or image read, the executor records one global guest-memory dependency
covering host, transfer, shader, colour, and depth/stencil writes. This scope is allocation-wide by
design: a RAMBlock import, a packed mapping import, and child images may be different Vulkan objects
over the same physical pages, so a resource-specific barrier cannot name every producer.
The dependency is re-armed at every decoded guest-operation boundary even when several operations
share one Vulkan command buffer: guest CPU writes occur outside the backend and therefore cannot be
memoized as visible across operations. Direct aliases remain reusable because they still name the
live allocation. Gathered buffers and uploaded images are snapshots and expire at that same
boundary; immutable owned byte buffers retain their command-buffer lifetime.
Encoder texture state and the shader interface remain separate inputs: a guest may leave an object
bound in a slot the current shader does not use. Such a slot has no reflected descriptor and creates
no Vulkan image or transfer. A reflected descriptor must carry its real dimension; absence is never
defaulted to 2D.
When Vulkan requires a larger binding extent than the guest allocation, the executor reports that
exact extent before materialization and the host view appends anonymous padding. Those bytes belong
only to the host binding: they never enter the guest footprint, content identity, or writeback.
A mapped sampled image retains its imported image/view with the serialized resource lifetime and
keeps an exact typed buffer-transfer representation beside it. Stable backend admission results—
either a binding requirement or a typed layout refusal—are cached by the complete image-layout
request under that same resource lifetime; different formats, dimensions, levels, and pitches
cannot reuse one answer, while transient driver failures are never retained. A refused direct
layout dispatches straight to the transfer representation instead of probing an incompatible
backend object on every bind. Page runs remain mapping/transport state, not content identity. Any
exact-layout admission disagreement selects the same transfer representation without changing the
resource, coherence, or lifecycle result.

Compute residency preserves the same texture identity across storage and sampled bindings. A
resident image is created with both Vulkan usages and is bound directly when the guest later names
it as sampled input; resource binding does not invent a snapshot copy or a second texture lifetime.

Presentation prepares the current present's resident carrier before deciding whether capture needs
CPU pixels. A prepared engine resident goes directly to the host presenter; CPU readback exists only
for a present whose capability/lifetime check cannot supply that carrier.

Direct GPU writes into imported guest pages are owned by the submission that records them. Their
canonical page set moves into that ring entry at seal and retires when its fence signals; visibility
tracking has no independent capacity, eviction, overflow merge, or global lifetime approximation.

Guest RAM host-pointer imports carry the external-memory handle selected by the physical device's
importability query. Mappings created outside Vulkan prefer the mapped-foreign handle when it is
reported importable; hosts exposing only host-allocation import retain that reported handle. The
same selected value is used for the pointer query, buffer/image creation, and memory import, and is
published in `vk_caps`; no host or driver name chooses it.

Guest-derived GPU state is owned by a device/session handle: pools, residents, imports, submissions,
presenter, counters, and completion signals. Only the physical context and immutable content-keyed
caches may be shared. Reset or deletion of one vGPU must not invalidate another.

## Regression gates

Architecture changes need behavioral tests at the owner boundary. Preserve at least these classes:

- deleting and recreating one object-table slot cannot inherit debt, witnesses, replicas, or
  residency from the retired `ResourceId`;
- two devices isolate reset, deletion, leases, counters, presentation, and guest-write debt;
- unified/discrete × import/no-import produces equivalent content and guest effects;
- resolved commands contain semantic endpoints and separate completion outputs;
- mapper fixtures preserve high reference bits and independent plane/rotation fields;
- PCI and MMIO adapters consume the same semantic presentation result.

Run the serial workspace suite and the feature matrix for cross-target changes:

```sh
cargo test --workspace --all-targets --features host-window -- --test-threads=1
scripts/feature-matrix/feature-matrix.sh
cargo clippy --workspace --all-targets --features host-window -- -D warnings
cargo clippy -p reims-vgpu --target aarch64-apple-darwin \
  --all-targets --features host-window -- -D warnings
```

Use types, constructors, dependency direction, and behavioral fixtures to preserve the seam. Do not
add tests that parse repository source to police architecture by spelling.

### The Metal conformance battery

`conformance/` is a Swift battery that runs the **same source** on a native macOS host and inside
the guest. Each case computes a value the CPU can predict exactly, asks the GPU for it, and
compares, so a failure names the case and the bytes rather than "the screenshot looks wrong". The
comparison between the two hosts is what makes a result usable: a case that fails in the guest and
passes natively is a named device defect, and one that fails on both is a wrong expectation in the
suite. `conformance/README.md` owns the details.
