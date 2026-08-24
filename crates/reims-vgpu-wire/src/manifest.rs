//! Coverage manifest — what "exhaustive" means here, and how far off we are.
//!
//! The serializer's API surface is not a list anyone curates. The Objective-C
//! runtime hands it over exactly: `class_copyMethodList` on the serializer and
//! its encoder classes enumerates every selector Apple ships. [`INVENTORY`]
//! records those counts, [`MANIFEST`] records what this crate has done about
//! each selector, and [`counts`] reports the gap.
//!
//! That makes exhaustiveness a measurement rather than a claim, and it survives
//! OS updates: when Apple adds a selector, the inventory the oracle regenerates
//! stops matching the manifest and the test says which one appeared.
//!
//! Every selector must end in one of three states. "Not looked at yet" is
//! [`Coverage::Unimplemented`], which is honest; silence is not an option,
//! because a selector missing from the manifest entirely is indistinguishable
//! from one that does not exist.
//!
//! # Absent from here does not mean absent from the class
//!
//! `class_copyMethodList` returns the methods a class **declares itself**. It
//! does not walk superclasses, and the encoder classes have one: they all derive
//! from a shared `PGSerializerCommandEncoder` that [`INVENTORY`] has no row for.
//! So a selector defined only on that base is invisible here while being
//! callable on every encoder, and the rule above — missing means non-existent —
//! does not hold for it.
//!
//! This is not hypothetical; it has already produced two wrong conclusions in
//! this workspace. Residency is declared on the base class in two forms, an
//! unqualified `useHeaps:count:` / `useResources:count:usage:` pair and their
//! singular siblings, with only the `stages:`-qualified overrides declared on the
//! render encoder. Reading "residency is not among the compute encoder's
//! selectors" as "a compute encoder cannot receive a residency call" is exactly
//! the inference this hole invites, and `runtime::decode::compute` drew it —
//! concluding that its own opcodes had no producer when the base class inherits
//! them one.
//!
//! Until the base class has an [`INVENTORY`] row and rows here, treat a selector
//! that is absent as *untriaged with respect to inheritance*, and check the base
//! class before concluding a call cannot reach an encoder.

/// What this crate has done about a selector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Coverage {
    /// A view exists and an oracle fixture pins it.
    Covered {
        /// Module path of the view, e.g. `ops::texture`.
        module: &'static str,
    },
    /// A view exists and a fixture pins it, but no opcode describes it.
    ///
    /// Two different ways a selector reaches this state, and they share the
    /// consequence: [`Entry::opcodes`] must stay empty, and that emptiness is
    /// the finding rather than a gap.
    ///
    /// * **The opcode is guest data.** `optimize:withCommand:` and its two
    ///   siblings write their `command:` argument into the record header's
    ///   opcode field, so the opcode a capture observes is whichever one the
    ///   case passed. Recording it would read as a property of the selector.
    /// * **The record has no opcode field.**
    ///   `beginSegment:protectionOptions:` writes the segment header, which is
    ///   framing around records rather than a record — see
    ///   [`crate::ops::segment`].
    ///
    /// The record's *shape* is covered like any other, by a module and a
    /// fixture.
    CoveredNoFixedOpcode {
        /// Module path of the view that reads the record's body.
        module: &'static str,
    },
    /// Known to emit a wire operation; no view yet.
    Unimplemented,
    /// Emits no wire operation, so there is nothing to view.
    ///
    /// The reason is required and must say *why* — "accessor", "returns a
    /// host-side object", "queries the device". A wrong exclusion is how a
    /// command goes missing without anything reporting it.
    Excluded { reason: &'static str },
}

/// One selector and its state.
#[derive(Clone, Copy, Debug)]
pub struct Entry {
    pub class: &'static str,
    pub selector: &'static str,
    /// Opcodes observed on the wire, where the oracle has run the selector.
    ///
    /// A list rather than one value, because a selector is not one opcode. Each
    /// draw picks between a compact encoding and a wide one by the magnitude of
    /// its arguments (see [`crate::ops::render`]), so recording only the form a
    /// first capture happened to produce would leave the other looking
    /// unimplemented while a view for it sat right beside it.
    pub opcodes: &'static [u32],
    pub coverage: Coverage,
}

/// Selector counts per class, read from the Objective-C runtime.
///
/// Regenerate with `scripts/wire-oracle/wire-oracle.sh --inventory`. These are
/// instance-method counts, which include accessors and other selectors that
/// emit nothing — so this is the surface to triage, not the number of
/// operations that exist.
#[derive(Clone, Copy, Debug)]
pub struct ClassInventory {
    pub class: &'static str,
    pub instance_methods: usize,
}

/// Measured on AppleParavirtGPUMetal 64.4.7 (macOS 26.5).
pub const INVENTORY: &[ClassInventory] = &[
    ClassInventory {
        class: "PGSerializer",
        instance_methods: 95,
    },
    ClassInventory {
        class: "PGSerializerRenderCommandEncoder",
        instance_methods: 152,
    },
    ClassInventory {
        class: "PGSerializerComputeCommandEncoder",
        instance_methods: 58,
    },
    ClassInventory {
        class: "PGSerializerBlitCommandEncoder",
        instance_methods: 38,
    },
    ClassInventory {
        class: "PGSerializerInfoCommandEncoder",
        instance_methods: 21,
    },
];

/// Exclusion reason for a selector Apple's serializer will not serialize.
///
/// These do not return an empty operation — they fail an assertion inside the
/// encoder and abort. The oracle catches that, records the selector under
/// `unsupported`, and carries on, so the exclusion is evidence rather than an
/// assumption and is re-checked on every capture.
///
/// # It is the weakest of the three reasons, for the same cause twice
///
/// An assertion is a refusal by the *capability state the capture ran in*, not
/// by Apple. `runtime::decode::compute` records the first instance: `0xe3`/`0xe6`
/// sat here as refused, and both selectors emit the moment their gate is on.
///
/// The render encoder carries a second, larger instance, and this doc is where
/// it is written down because nothing executable catches it yet. **Fifteen**
/// selectors on it are structured as *"if the serializer supports OpenGL, emit;
/// otherwise assert"* — the assertion the oracle observed is the else-branch of
/// a capability test — and behind the gate each has a fixed opcode and a fixed
/// body:
///
/// | opcode | selector | body |
/// |---|---|---|
/// | `0x8a` | `setAlphaTestReferenceValue:` | 4 |
/// | `0x8b` | `setPointSize:` | 4 |
/// | `0x8c` | `setClipPlane:p2:p3:p4:atIndex:` | 20 |
/// | `0x8d` | `setVertexSamplerState:lodMinClamp:lodMaxClamp:lodBias:atIndex:` | 20 |
/// | `0x8e` | `setFragmentSamplerState:lodMinClamp:lodMaxClamp:lodBias:atIndex:` | 20 |
/// | `0x8f` | `setViewportTransformEnabled:` | 4 |
/// | `0x90` | `setProvokingVertexMode:` | 4 |
/// | `0x91` | `setPrimitiveRestartEnabled:index:` | 8 |
/// | `0x92` | `setTriangleFrontFillMode:backFillMode:` | 4 |
/// | `0x93` | `setTransformFeedbackState:` | 4 |
/// | `0x94` | `setDepthCleared` | 0 |
/// | `0x95` | `setStencilCleared` | 0 |
/// | `0x96` | `setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex:` | 16 |
/// | `0x97` | `setDepthResolveTexture:slice:depthPlane:level:yInvert:` | 12 |
/// | `0x98` | `setStencilResolveTexture:slice:depthPlane:level:yInvert:` | 12 |
///
/// A prior version of this paragraph listed eight. The seven it missed are the
/// ones whose row is absent from `INVENTORY` for the reason this module's own
/// doc now states — the manifest cannot see a selector, and its rule is that
/// absent means non-existent.
///
/// They stay `Excluded` here rather than being promoted on that reading: this
/// crate's rule is that a row's opcode comes from a capture, and no capture has
/// run with that gate on. What the reading does establish is that the *reason*
/// is wrong — these are gated, not refused — so
/// `every_capability_gated_selector_names_the_flag_that_unlocks_it` cannot see
/// them.
///
/// **The gate is on.** The flag is the serializer's feature version, which the
/// device publishes as `reims_vgpu::model::DEVICE_INFO_KEY_SERIALIZER_VERSION`
/// and which unlocks OpenGL at rung 6; this device sends 8. So these fifteen are
/// records a guest on an OpenGL-compatibility path *will* send and this device
/// does not decode. Four of them change what a draw produces — primitive
/// restart, two-sided fill mode, and the two sampler binds that carry an LOD
/// bias — and three more name multisample resolve targets. They reach
/// `runtime::exec`'s unimplemented-opcode report rather than vanish, which is
/// the one part of this that is already right.
pub const REFUSED_BY_SERIALIZER: &str =
    "the serializer fails an assertion instead of emitting an operation";

/// Exclusion reason for a selector that runs and writes no record.
///
/// The third outcome a capture can have, and the quietest: the call returns
/// normally, asks its arguments for nothing, and allocates no operation. The
/// blit encoder's `getType` is one — a guest can issue it and there is no wire
/// operation for this device to decode. The oracle records these under
/// `silent` every run, so the exclusion is measured rather than assumed, and
/// `every_excluded_row_that_claims_silence_still_gets_it` is the check.
///
/// **A silence is only as true as the capability state it was measured in.**
/// This doc named `fillBuffer:range:pattern4:` as its example for a while, and
/// that selector emits `0x13f` the moment `-setSupportsBlitEncoderSPI:` is on.
/// All sixteen flags default off, so a row here would be a false claim about
/// Apple for any family gated on one. Two further checks close that:
/// `every_silent_selector_is_silent_under_every_capability` measures the whole
/// list again with every flag forced, and
/// `every_capability_gated_selector_names_the_flag_that_unlocks_it` says which
/// flag it was.
pub const EMITS_NO_OPERATION: &str = "the serializer returns without emitting an operation";

/// Exclusion reason for a selector that is not a command at all.
///
/// Object lifecycle: the compiler-generated `.cxx_construct` / `.cxx_destruct`
/// ivar hooks, `-dealloc`, the designated initializer, and `-endEncoding`.
/// These are not driven by the oracle — driving `.cxx_destruct` on a live
/// object would tear it down mid-capture — so unlike the other two exclusion
/// reasons this one rests on the Objective-C ABI rather than on a measurement,
/// which is why it is a separate string.
pub const NOT_A_COMMAND: &str = "an object-lifecycle hook rather than an encoder command";

/// Per-selector state.
///
/// Seeded with the operations the oracle has actually driven. The rest of the
/// surface in [`INVENTORY`] is not yet triaged into rows — [`untriaged`]
/// reports that gap so it cannot be mistaken for coverage.
pub const MANIFEST: &[Entry] = &[
    // Two opcodes, and which one the guest gets is a capability rather than an
    // argument: `-setSupportsSwizzledTextures:` switches this selector from the
    // 32-byte descriptor at `1` to the 40-byte one at `0x34`. Not
    // `TextureDescriptor2`, despite the name — that flag leaves this record
    // alone and moves the other four.
    Entry {
        class: "PGSerializer",
        selector: "newTextureWithDescriptor:allocator:",
        opcodes: &[1, crate::ops::texture::OPCODE_NEW_TEXTURE_WIDE],
        coverage: Coverage::Covered {
            module: "ops::texture",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newSamplerStateWithDescriptor:allocator:",
        opcodes: &[3],
        coverage: Coverage::Covered {
            module: "ops::sampler",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newDepthStencilStateWithDescriptor:allocator:",
        opcodes: &[4],
        coverage: Coverage::Covered {
            module: "ops::depth_stencil",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newFenceWithAllocator:",
        opcodes: &[13],
        coverage: Coverage::Covered {
            module: "ops::fence",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newTextureViewWithPixelFormat:baseTexture:allocator:",
        opcodes: &[7],
        coverage: Coverage::Covered {
            module: "ops::texture_view",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newTextureWithBuffer:descriptor:offset:bytesPerRow:allocator:",
        opcodes: &[
            9,
            crate::ops::backed_texture::OPCODE_BUFFER_TEXTURE_WIDE,
        ],
        coverage: Coverage::Covered {
            module: "ops::backed_texture",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newIOSurfaceTextureWithDescriptor:plane:allocator:",
        opcodes: &[
            crate::ops::backed_texture::OPCODE_IOSURFACE_TEXTURE,
            crate::ops::backed_texture::OPCODE_IOSURFACE_TEXTURE_ROTATED,
            crate::ops::backed_texture::OPCODE_IOSURFACE_TEXTURE_WIDE,
        ],
        coverage: Coverage::Covered {
            module: "ops::backed_texture",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newTextureViewWithPixelFormat:textureType:levels:slices:\
                   baseTexture:allocator:",
        opcodes: &[8],
        coverage: Coverage::Covered {
            module: "ops::texture_view",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newTextureViewWithPixelFormat:textureType:levels:slices:\
                   swizzle:baseTexture:allocator:",
        opcodes: &[0x1b],
        coverage: Coverage::Covered {
            module: "ops::texture_view",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newTextureWithDescriptor:heap:offset:useOffset:allocator:",
        opcodes: &[
            0x15,
            crate::ops::heap_texture::OPCODE_NEW_HEAP_TEXTURE_WIDE,
        ],
        coverage: Coverage::Covered {
            module: "ops::heap_texture",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "drawPrimitives:vertexStart:vertexCount:",
        opcodes: &[0x01, 0x00],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "drawPrimitives:vertexStart:vertexCount:instanceCount:baseInstance:",
        opcodes: &[0x05, 0x04],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "drawPrimitives:vertexStart:vertexCount:instanceCount:",
        opcodes: &[0x03, 0x02],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:",
        opcodes: &[0x07, 0x06],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:\
                   instanceCount:",
        opcodes: &[0x09, 0x08],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "drawIndexedPrimitives:indexCount:indexType:indexBuffer:indexBufferOffset:\
                   instanceCount:baseVertex:baseInstance:",
        opcodes: &[0x0b, 0x0a],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setScissorRect:",
        opcodes: &[0x75],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setViewport:",
        opcodes: &[0x82],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setCullMode:",
        opcodes: &[0x6b],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setFrontFacingWinding:",
        opcodes: &[0x73],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setBlendColorRed:green:blue:alpha:",
        opcodes: &[0x65],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setDepthClipMode:",
        opcodes: &[0x6d],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTriangleFillMode:",
        opcodes: &[0x7c],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setLineWidth:",
        opcodes: &[0x88],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTessellationFactorScale:",
        opcodes: &[0x7b],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setStencilReferenceValue:",
        // Shares 0x77 with the front/back form below, writing its one value
        // into both fields. See `ops::render::StencilReference`.
        opcodes: &[0x77],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setStencilFrontReferenceValue:backReferenceValue:",
        opcodes: &[0x77],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setDepthBias:slopeScale:clamp:",
        opcodes: &[0x6c],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVisibilityResultMode:offset:",
        opcodes: &[0x84],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setRenderPipelineState:",
        opcodes: &[0x74],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setDepthStencilState:",
        opcodes: &[0x68],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexBuffer:offset:atIndex:",
        opcodes: &[0x7d],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    // The four attribute-stride vertex binds. `0xa5` is the bind (three
    // selectors reach it, including the bytes form, which stages through a
    // buffer) and `0xa6` is the offset re-point. They are different opcodes
    // from `0x7d`/`0x7e` rather than longer forms of them, the same way the
    // sampler LOD binds are.
    //
    // All four emit nothing unless `-supportsDynamicAttributeStride` is on, and
    // the serializer answers false by default -- so the oracle drives them
    // through `withCapability`. Captured at the default they land on `silent`,
    // which would have become an `EMITS_NO_OPERATION` row claiming Apple emits
    // nothing here.
    // Vertex amplification, gated on `-supportsVertexAmplification` and driven
    // through `withCapability` for the same reason as the four above.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexAmplificationMode:value:",
        opcodes: &[0x99],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexAmplificationCount:viewMappings:",
        opcodes: &[0x9a],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexBuffer:offset:attributeStride:atIndex:",
        opcodes: &[0xa5],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexBuffers:offsets:attributeStrides:withRange:",
        opcodes: &[0xa5],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexBytes:length:attributeStride:atIndex:",
        opcodes: &[0xa5],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexBufferOffset:attributeStride:atIndex:",
        opcodes: &[0xa6],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setFragmentBuffer:offset:atIndex:",
        opcodes: &[0x6e],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setFragmentBuffers:offsets:withRange:",
        opcodes: &[0x6e],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexBufferOffset:atIndex:",
        opcodes: &[0x7e],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setFragmentBufferOffset:atIndex:",
        opcodes: &[0x6f],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexTexture:atIndex:",
        opcodes: &[0x81],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexTextures:withRange:",
        opcodes: &[0x81],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setFragmentTexture:atIndex:",
        opcodes: &[0x72],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexSamplerState:atIndex:",
        opcodes: &[0x7f],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setFragmentSamplerState:atIndex:",
        opcodes: &[0x70],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "updateFence:afterStages:",
        opcodes: &[0x18],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "waitForFence:beforeStages:",
        opcodes: &[0x19],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "useResource:usage:stages:",
        opcodes: &[0x89],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "useResources:count:usage:stages:",
        opcodes: &[0x89],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setColorStoreAction:atIndex:",
        opcodes: &[0x66],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setDepthStoreAction:",
        opcodes: &[0x69],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setStencilStoreAction:",
        opcodes: &[0x78],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "useHeap:stages:",
        opcodes: &[0x1b],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    // The serializer refuses these seven outright: each fails an assertion
    // inside Apple's own encoder rather than emitting an operation, so there is
    // no record for a guest to send and nothing for this crate to view. The
    // oracle drives all seven every run and lists them under `unsupported`, so
    // these rows are re-measured rather than remembered —
    // `every_excluded_row_that_claims_a_refusal_still_gets_one` is the check.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTriangleFrontFillMode:backFillMode:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setProvokingVertexMode:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setDepthTestMinBound:maxBound:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setAlphaTestReferenceValue:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setPointSize:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setPrimitiveRestartEnabled:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setViewportTransformEnabled:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    // --- PGSerializerInfoCommandEncoder ------------------------------------
    //
    // All 21 driven, layouts in [`crate::ops::info`] and every one pinned by
    // `every_info_fixture_reads_back_what_metal_was_asked_for`.
    //
    // The whole class is questions rather than commands: each record ends with
    // the `(reply_buffer_ref, reply_offset)` pair the command stream handed
    // back, and the fixture asserts that pair against what the stream returned
    // rather than merely against zero.
    //
    // `heapTextureDescriptorSizeAndAlign:sizeAndAlign:` is the one exception,
    // and for a reason that is about this harness rather than about Apple: the
    // serializer asks its heap for `descriptorPrivate` and follows the answer,
    // the stub cannot supply one, and the capture records the selector under
    // `crashed`. A fault measures nothing, so the row stays Unimplemented --
    // `a_selector_that_faulted_the_harness_claims_nothing` enforces that.
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "bufferHostResourceInfo:info:",
        opcodes: &[0x1cd],
        coverage: Coverage::Covered {
            module: "ops::info",
        },
    },
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "computePipelineHostResourceInfo:info:",
        opcodes: &[0x1d3],
        coverage: Coverage::Covered {
            module: "ops::info",
        },
    },
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "computePipelineStateImageBlockMemoryLength:\
                   imageblockDimensions:info:",
        opcodes: &[0x1cb],
        coverage: Coverage::Covered {
            module: "ops::info",
        },
    },
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "computePipelineStateInfo:info:",
        opcodes: &[0x1c2],
        coverage: Coverage::Covered {
            module: "ops::info",
        },
    },
    // The one command on the class. Its record is byte-identical to a query's
    // and means something else; see [`crate::ops::info::CopyRateParameterBuffer`].
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "copyRasterizationRateParameterBuffer:buffer:bufferOffset:",
        opcodes: &[0x1c5],
        coverage: Coverage::Covered {
            module: "ops::info",
        },
    },
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "depthStencilHostResourceInfo:info:",
        opcodes: &[0x1d4],
        coverage: Coverage::Covered {
            module: "ops::info",
        },
    },
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "getRasterizationRateMapInfo:layerCount:info:",
        opcodes: &[0x1c4],
        coverage: Coverage::Covered {
            module: "ops::info",
        },
    },
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "heapHostResourceInfo:info:",
        opcodes: &[0x1cf],
        coverage: Coverage::Covered {
            module: "ops::info",
        },
    },
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "icbHostResourceInfo:info:",
        opcodes: &[0x1d1],
        coverage: Coverage::Covered {
            module: "ops::info",
        },
    },
    // The generic mapper the two fixed-opcode ones wrap: its `command:`
    // argument is written into the opcode field, so recording an opcode here
    // would record the case's argument as a property of the selector.
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "mapCoordinateInternal:fromCoordinate:forLayer:\
                   toCoordinate:command:",
        opcodes: &[],
        coverage: Coverage::CoveredNoFixedOpcode {
            module: "ops::info",
        },
    },
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "mapPhysicalToScreenCoordinates:forPhysicalCoordinate:\
                   forLayer:toScreenCoordinate:",
        opcodes: &[0x1c7],
        coverage: Coverage::Covered {
            module: "ops::info",
        },
    },
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "mapScreenToPhysicalCoordinates:forScreenCoordinate:\
                   forLayer:toPhysicalCoordinate:",
        opcodes: &[0x1c6],
        coverage: Coverage::Covered {
            module: "ops::info",
        },
    },
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "renderPipelineHostResourceInfo:info:",
        opcodes: &[0x1d2],
        coverage: Coverage::Covered {
            module: "ops::info",
        },
    },
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "renderPipelineStateImageBlockMemoryLength:\
                   imageblockDimensions:info:",
        opcodes: &[0x1ca],
        coverage: Coverage::Covered {
            module: "ops::info",
        },
    },
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "renderPipelineStateInfo:info:",
        opcodes: &[0x1c9],
        coverage: Coverage::Covered {
            module: "ops::info",
        },
    },
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "samplerStateHostResourceInfo:info:",
        opcodes: &[0x1d0],
        coverage: Coverage::Covered {
            module: "ops::info",
        },
    },
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "textureHostResourceInfo:info:",
        opcodes: &[0x1ce],
        coverage: Coverage::Covered {
            module: "ops::info",
        },
    },
    // The fourth class driven through `beginSegment:`, and the one that shows
    // the byte at `+4` is not a running index: it writes 4 where the class
    // before it writes 2.
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "beginSegment:protectionOptions:",
        opcodes: &[],
        coverage: Coverage::CoveredNoFixedOpcode {
            module: "ops::segment",
        },
    },
    // The one selector on this class whose first argument is not a resource:
    // it takes an `MTLTextureDescriptor`, and the oracle had been handing it a
    // heap stub on the strength of the name. That faulted, so the row said
    // `Unimplemented` while nothing had measured anything. Driven with the
    // descriptor, it writes the texture body verbatim and a reply pair.
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "heapTextureDescriptorSizeAndAlign:sizeAndAlign:",
        opcodes: &[
            0x1c3,
            crate::ops::info::OPCODE_HEAP_TEXTURE_DESCRIPTOR_SIZE_AND_ALIGN_WIDE,
        ],
        coverage: Coverage::Covered {
            module: "ops::info",
        },
    },
    // Asserts inside Apple's encoder; its singular siblings do not.
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "mapPhysicalToScreenCoordinateMultiple:\
                   forPhysicalCoordinates:forLayer:toScreenCoordinates:\
                   count:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerInfoCommandEncoder",
        selector: "initWithCommandBuffer:serializer:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: NOT_A_COMMAND,
        },
    },
    // --- PGSerializerComputeCommandEncoder ---------------------------------
    //
    // All 58, driven end to end. Layouts in [`crate::ops::compute`].
    //
    // This block used to say the negative results were its substance: that the
    // seven control-flow selectors, both attribute-stride bind forms and
    // setImageblockWidth:height: all returned without writing a record. Every
    // one of those claims was false, and all of them for the same reason --
    // they were measured with all sixteen capability flags at their default,
    // which is off.
    //
    // The control-flow seven emit `0xdc`-`0xe2` under
    // `-setSupportsCommandBufferJump:`, a contiguous run with the pairs
    // adjacent. Three of the seven carry a predicate and four are the header
    // alone; see [`crate::ops::compute::ControlFlowPredicate`]. This is the one
    // place in the protocol where the command stream is data-dependent.
    //
    // Two selectors here are still `Unimplemented` and still gated:
    // `maybeEmitSerialBarrier` and `writeDescriptor` under
    // `-setSupportsComputePassDescriptorDispatchType:`, and
    // `setImageblockWidth:height:` under `-setSupportsImageBlocks:`. The flag
    // for each is measured, not guessed --
    // `every_capability_gated_selector_names_the_flag_that_unlocks_it` prints
    // the table.
    // The six rows below each name **two** opcodes, and the second is the same
    // one on all six. Under `-setSupportsComputePassDescriptorDispatchType:` a
    // serial compute pass emits an 0xd7 memory barrier after every dispatch and
    // every ICB execution, measured on all six with a `_serial` fixture. It is
    // the pass's dispatch type that decides — the concurrent arm emits one
    // record — so these rows describe a selector whose record *count* depends on
    // state, which is a thing this manifest cannot otherwise express.
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "dispatchThreadgroups:threadsPerThreadgroup:",
        opcodes: &[0xc8, 0xd7],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "dispatchThreadgroupsWithIndirectBuffer:\
                   indirectBufferOffset:threadsPerThreadgroup:",
        opcodes: &[0xc9, 0xd7],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "dispatchThreads:threadsPerThreadgroup:",
        opcodes: &[0xca, 0xd7],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "executeCommandsInBuffer:indirectBuffer:\
                   indirectBufferOffset:",
        opcodes: &[0xe5, 0xd7],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "executeCommandsInBuffer:withRange:",
        opcodes: &[0xe4, 0xd7],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "memoryBarrierWithResources:count:",
        opcodes: &[0xd6],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "memoryBarrierWithScope:",
        opcodes: &[0xd7],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setBuffer:offset:atIndex:",
        opcodes: &[0xcb],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setBufferOffset:atIndex:",
        opcodes: &[0xcf],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setBuffers:offsets:withRange:",
        opcodes: &[0xcb],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setBytes:length:atIndex:",
        opcodes: &[0xcb],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setComputePipelineState:",
        opcodes: &[0xd0],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setSamplerState:atIndex:",
        opcodes: &[0xcc],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setSamplerState:lodMinClamp:lodMaxClamp:atIndex:",
        opcodes: &[0xcd],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
        opcodes: &[0xcd],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setSamplerStates:withRange:",
        opcodes: &[0xcc],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setStageInRegion:",
        opcodes: &[0xd1],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setStageInRegionWithIndirectBuffer:indirectBufferOffset:",
        opcodes: &[0xd2],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setTexture:atIndex:",
        opcodes: &[0xce],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setTextures:withRange:",
        opcodes: &[0xce],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setThreadgroupMemoryLength:atIndex:",
        opcodes: &[0xd3],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "updateFence:",
        opcodes: &[0xd4],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "waitForFence:",
        opcodes: &[0xd5],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    // Driven, returned, wrote nothing. Measured every capture.
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "dispatchType",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "encodeEndDoWhile:offset:comparison:referenceValue:",
        opcodes: &[0xdd],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "encodeEndIf",
        opcodes: &[0xe2],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "encodeEndWhile",
        opcodes: &[0xdf],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "encodeStartDoWhile",
        opcodes: &[0xdc],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "encodeStartElse",
        opcodes: &[0xe1],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "encodeStartIf:offset:comparison:referenceValue:",
        opcodes: &[0xe0],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "encodeStartWhile:offset:comparison:referenceValue:",
        opcodes: &[0xde],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "flushWrites",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "getType",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "handleSplits",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "maybeEmitSerialBarrier",
        opcodes: &[0xd7],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "newKernelDebugInfo",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "reattachToCommandStream:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setBuffer:offset:attributeStride:atIndex:",
        opcodes: &[0xd9],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setBufferOffset:attributeStride:atIndex:",
        opcodes: &[0xda],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setBuffers:offsets:attributeStrides:withRange:",
        opcodes: &[0xd9],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setCurrentDispatchType:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setImageblockWidth:height:",
        opcodes: &[0xd8],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "shouldAllowReattach",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "writeDescriptor",
        opcodes: &[0xdb],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    // These two asserted inside Apple's encoder and carried
    // REFUSED_BY_SERIALIZER, which was a claim about Apple and was wrong. Each
    // is gated on its own capability flag, and with the flag forced on the
    // serializer emits rather than asserting. The capability sweep beside them
    // could not find this: it diffs the two passes' `silent` lists, and an
    // assertion is filed on `unsupported`.
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "dispatchThreadsWithIndirectBuffer:indirectBufferOffset:",
        opcodes: &[
            crate::ops::compute::OPCODE_DISPATCH_THREADS_INDIRECT,
            crate::ops::compute::OPCODE_MEMORY_BARRIER_SCOPE,
        ],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "insertCompressedTextureReinterpretationFlush",
        opcodes: &[crate::ops::compute::OPCODE_INSERT_COMPRESSED_TEXTURE_FLUSH],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    // Asserted inside Apple's encoder. Note the whole function-table and
    // acceleration-structure surface is here: this serializer serializes none
    // of it.
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "sampleCountersInBuffer:atSampleIndex:withBarrier:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setAccelerationStructure:atBufferIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setIntersectionFunctionTable:atBufferIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setIntersectionFunctionTables:withBufferRange:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setVisibleFunctionTable:atBufferIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setVisibleFunctionTables:withBufferRange:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    // Framing and lifecycle.
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "beginSegment:protectionOptions:",
        opcodes: &[],
        coverage: Coverage::CoveredNoFixedOpcode {
            module: "ops::segment",
        },
    },
    // The fourth attribute-stride form, and the one its siblings did not
    // settle: it stages its bytes through the command stream and emits the
    // *buffer* bind naming the staging pair, exactly as the non-stride
    // `setBytes:length:atIndex:` does. So its opcode is `0xd9` and its entry is
    // [`crate::ops::compute::BufferStrideBind`] -- measured, not read off the
    // sibling, because both *other* stride selectors on this class write
    // nothing at all.
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "setBytes:length:attributeStride:atIndex:",
        opcodes: &[0xd9],
        coverage: Coverage::Covered {
            module: "ops::compute",
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: ".cxx_construct",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: NOT_A_COMMAND,
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: ".cxx_destruct",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: NOT_A_COMMAND,
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "endEncoding",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: NOT_A_COMMAND,
        },
    },
    Entry {
        class: "PGSerializerComputeCommandEncoder",
        selector: "initWithCommandBuffer:descriptor:serializer:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: NOT_A_COMMAND,
        },
    },
    // The barrier, indirect-execution and plural-state cluster, plus the two
    // inline-constant setters. `setVertexBytes:length:atIndex:` and its
    // fragment sibling emit the *buffer bind* opcode: the serializer stages the
    // bytes through the command stream and records the staging buffer's ref and
    // offset, so there is no separate inline-data record.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "drawPrimitives:indirectBuffer:indirectBufferOffset:",
        opcodes: &[0x10],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "drawIndexedPrimitives:indexType:indexBuffer:\
                   indexBufferOffset:indirectBuffer:indirectBufferOffset:",
        opcodes: &[0x11],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "executeCommandsInBuffer:indirectBuffer:\
                   indirectBufferOffset:",
        opcodes: &[0x14],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "executeCommandsInBuffer:withRange:",
        opcodes: &[0x15],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "memoryBarrierWithResources:count:afterStages:beforeStages:",
        opcodes: &[0x16],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "memoryBarrierWithScope:afterStages:beforeStages:",
        opcodes: &[0x17],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "useHeaps:count:stages:",
        opcodes: &[0x1b],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "textureBarrier",
        opcodes: &[0x85],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setScissorRects:count:",
        opcodes: &[0x76],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setViewports:count:",
        opcodes: &[0x83],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexBytes:length:atIndex:",
        opcodes: &[0x7d],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setFragmentBytes:length:atIndex:",
        opcodes: &[0x6e],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    // --- PGSerializerBlitCommandEncoder ------------------------------------
    //
    // All 38, driven end to end. The record layouts are in [`crate::ops::blit`]
    // and every row below rests on a fixture, a refusal or a measured silence
    // from the same capture; none is a reading of the selector's name.
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "copyFromBuffer:sourceOffset:sourceBytesPerRow:\
                   sourceBytesPerImage:sourceSize:toTexture:\
                   destinationSlice:destinationLevel:destinationOrigin:",
        opcodes: &[0x12c],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "copyFromBuffer:sourceOffset:sourceBytesPerRow:\
                   sourceBytesPerImage:sourceSize:toTexture:\
                   destinationSlice:destinationLevel:destinationOrigin:\
                   options:",
        opcodes: &[0x12c],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:\
                   size:",
        opcodes: &[0x12d],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:\
                   sourceSize:toBuffer:destinationOffset:\
                   destinationBytesPerRow:destinationBytesPerImage:",
        opcodes: &[0x12e],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:\
                   sourceSize:toBuffer:destinationOffset:\
                   destinationBytesPerRow:destinationBytesPerImage:options:",
        opcodes: &[0x12e],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:\
                   sourceSize:toTexture:destinationSlice:destinationLevel:\
                   destinationOrigin:",
        opcodes: &[0x12f],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "copyFromTexture:sourceSlice:sourceLevel:sourceOrigin:\
                   sourceSize:toTexture:destinationSlice:destinationLevel:\
                   destinationOrigin:options:",
        opcodes: &[0x130],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "copyFromTexture:sourceSlice:sourceLevel:toTexture:\
                   destinationSlice:destinationLevel:sliceCount:levelCount:",
        opcodes: &[0x13e],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "copyFromTexture:toTexture:",
        opcodes: &[0x13e],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "copyIndirectCommandBuffer:sourceRange:destination:\
                   destinationIndex:",
        opcodes: &[0x131],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "fillBuffer:range:value:",
        opcodes: &[0x132],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "generateMipmapsForTexture:",
        opcodes: &[0x133],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "optimizeContentsForCPUAccess:",
        opcodes: &[0x134],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "optimizeContentsForCPUAccess:slice:level:",
        opcodes: &[0x136],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "optimizeContentsForGPUAccess:",
        opcodes: &[0x135],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "optimizeContentsForGPUAccess:slice:level:",
        opcodes: &[0x137],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "optimizeIndirectCommandBuffer:withRange:",
        opcodes: &[0x138],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "resetCommandsInBuffer:withRange:",
        opcodes: &[0x139],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "synchronizeResource:",
        opcodes: &[0x13a],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "synchronizeTexture:slice:level:",
        opcodes: &[0x13b],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "updateFence:",
        opcodes: &[0x13c],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "waitForFence:",
        opcodes: &[0x13d],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    // The opcode is the `command:` argument, so these record none. Their bodies
    // are the same three shapes their fixed-opcode siblings above use.
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "optimize:withCommand:",
        opcodes: &[],
        coverage: Coverage::CoveredNoFixedOpcode {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "optimize:slice:level:withCommand:",
        opcodes: &[],
        coverage: Coverage::CoveredNoFixedOpcode {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "optimizeReset:withRange:withCommand:",
        opcodes: &[],
        coverage: Coverage::CoveredNoFixedOpcode {
            module: "ops::blit",
        },
    },
    // The six selectors `-setSupportsBlitEncoderSPI:` gates. Every one carried
    // EMITS_NO_OPERATION until the capture drove it with that flag forced on;
    // the flag is named by the per-flag attribution passes rather than guessed.
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "fillBuffer:range:pattern4:",
        opcodes: &[0x13f],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "fillTexture:level:slice:region:bytes:length:",
        opcodes: &[0x140],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    // Two selectors, one record. The `pixelFormat:` form differs only in where
    // the format word comes from -- the argument rather than the texture.
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "fillTexture:level:slice:region:color:",
        opcodes: &[0x141],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "fillTexture:level:slice:region:color:pixelFormat:",
        opcodes: &[0x141],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "getType",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "invalidateCompressedTexture:",
        opcodes: &[0x142],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "invalidateCompressedTexture:slice:level:",
        opcodes: &[0x143],
        coverage: Coverage::Covered {
            module: "ops::blit",
        },
    },
    // The four counter selectors: every one asserts inside Apple's encoder.
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "getTextureAccessCounters:region:mipLevel:slice:\
                   resetCounters:countersBuffer:countersBufferOffset:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "resetTextureAccessCounters:region:mipLevel:slice:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "resolveCounters:inRange:destinationBuffer:\
                   destinationOffset:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "sampleCountersInBuffer:atSampleIndex:withBarrier:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    // The encoder's own framing. `beginSegment:protectionOptions:` writes the
    // segment header rather than a command, so it has no opcode at all and is
    // covered by `ops::segment`; the row is repeated for the render encoder
    // because driving the same call on two classes is what shows the byte at
    // `+4` to be a type. The designated initializer writes the blit-pass
    // record and has no view yet.
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "beginSegment:protectionOptions:",
        opcodes: &[],
        coverage: Coverage::CoveredNoFixedOpcode {
            module: "ops::segment",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "beginSegment:protectionOptions:",
        opcodes: &[],
        coverage: Coverage::CoveredNoFixedOpcode {
            module: "ops::segment",
        },
    },
    // The blit encoder's designated initializer. Its three siblings on the
    // other encoder classes carry [`NOT_A_COMMAND`] and this one had been left
    // behind; it is the same thing, an object-lifecycle hook rather than a
    // record, and the oracle calls it to stand an encoder up for every blit
    // case in the capture.
    Entry {
        class: "PGSerializerBlitCommandEncoder",
        selector: "initWithCommandBuffer:descriptor:serializer:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: NOT_A_COMMAND,
        },
    },
    // --- The plural bind forms, and the sampler binds that carry LOD --------
    //
    // The plural forms share their singular sibling's opcode exactly -- a
    // singular bind is the plural one at `count == 1` -- so these rows add no
    // opcode. What they add is the fixture that shows the leading word really
    // is a count, which a singular record alone cannot: any constant would fit.
    //
    // The LOD-bearing sampler binds are the opposite: `0x80` and `0x71` are
    // *new* opcodes, not longer forms of `0x7f` and `0x70`, so a decoder that
    // knows only the plain pair does not lose the clamps -- it never sees the
    // bind. See [`crate::ops::render::SamplerLodBind`].
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setFragmentTextures:withRange:",
        opcodes: &[0x72],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexSamplerStates:withRange:",
        opcodes: &[0x7f],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setFragmentSamplerStates:withRange:",
        opcodes: &[0x70],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexBuffers:offsets:withRange:",
        opcodes: &[0x7d],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexSamplerState:lodMinClamp:lodMaxClamp:atIndex:",
        opcodes: &[0x80],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setFragmentSamplerState:lodMinClamp:lodMaxClamp:atIndex:",
        opcodes: &[0x71],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
        opcodes: &[0x80],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setFragmentSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
        opcodes: &[0x71],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    // The `lodBias:` sibling of both: Apple's encoder asserts rather than
    // emitting, so there is no four-float entry form.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexSamplerState:lodMinClamp:lodMaxClamp:lodBias:atIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setFragmentSamplerState:lodMinClamp:lodMaxClamp:lodBias:atIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    // One selector, two records: a convenience over the singular texture and
    // sampler binds rather than a combined record. `oracle.m` drives it as a
    // split case, so both halves are pinned; a case claiming one record would
    // have recorded the texture bind and lost the sampler bind in silence.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setFragmentTexture:atTextureIndex:samplerState:atSamplerIndex:",
        opcodes: &[0x72, 0x70],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    // --- Object-lifecycle hooks on the two classes that still lacked them ---
    //
    // The compute encoder's four have been here since it was triaged; these are
    // the same four selectors on the render encoder and the three on
    // `PGSerializer`. They rest on the Objective-C ABI rather than on a capture,
    // which is what [`NOT_A_COMMAND`] exists to say -- the oracle deliberately
    // does not drive them, because driving `.cxx_destruct` on a live object
    // tears it down mid-capture and `-dealloc` would take the serializer with
    // it. A row that had to be *measured* silent could not be written this way.
    //
    // `-endEncoding` is absent here on purpose: the render encoder's row for it
    // is already above, and it is the selector that finalizes the segment
    // header rather than a pure lifecycle hook.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: ".cxx_construct",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: NOT_A_COMMAND,
        },
    },
    // The heap sizing query: opcode 0x01's record without the ref. Its payload
    // is a bare TextureDescriptorBody, byte for byte, which the doc on that
    // struct had already named as a third reader before it had a fixture.
    Entry {
        class: "PGSerializer",
        selector: "heapTextureSizeAndAlignWithDescriptor:allocator:",
        opcodes: &[0x16],
        coverage: Coverage::Covered {
            module: "ops::texture",
        },
    },
    // Both rate-map selectors write ONE opcode -- `reset` is `new` with the ref
    // supplied by the caller instead of allocated -- and the record is variable
    // length, growing with the layer count and each layer's sample counts.
    // Sixteen of its declared bytes are never written at any size; see
    // [`crate::ops::rate_map`], which returns the written extent separately
    // from the record length for exactly that reason.
    Entry {
        class: "PGSerializer",
        selector: "newRasterizationRateMapWithDescriptor:allocator:",
        opcodes: &[0x32],
        coverage: Coverage::Covered {
            module: "ops::rate_map",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "resetRasterizationRateMapWithDescriptor:existingID:allocator:",
        opcodes: &[0x32],
        coverage: Coverage::Covered {
            module: "ops::rate_map",
        },
    },
    // The `layout:` argument reaches the wire verbatim: the 52 bytes the type
    // encoding `^{?=SSSSIIIIIIIIIII}` describes appear in the record with all
    // fifteen distinct values intact, at payload +20. The descriptor half has
    // no such source and was derived one property at a time -- including the
    // finding that this is the one creation record that does **not** carry the
    // new object's ref. See [`crate::ops::icb`].
    Entry {
        class: "PGSerializer",
        selector: "newIndirectCommandBufferWithDescriptor:layout:maxCommandCount:options:allocator:",
        opcodes: &[0x36],
        coverage: Coverage::Covered {
            module: "ops::icb",
        },
    },
    // Fill a caller's struct rather than the stream, and emit nothing at all --
    // which the type encoding predicted for the first half and could not for
    // the second. See `getTileDimensions:`, which has the same `^{...}` shape
    // and does emit.
    Entry {
        class: "PGSerializer",
        selector: "serializeTextureDescriptor:textureDescriptor:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "serializeTextureDescriptor2:textureDescriptor:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    // A pure computation and a host-side image copy: neither names an object
    // and neither reaches the wire.
    Entry {
        class: "PGSerializer",
        selector: "dataSizeForRegion:pixelFormat:bytesPerRow:bytesPerImage:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "copyImageBytesFromSource:toDestination:dataSize:region:bytesPerRow:bytesPerImage:mipmapLevel:slice:texture:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    // Shared textures emit nothing. A shared texture is an IOSurface reached
    // through a mach port, so the host learns of it by another route.
    Entry {
        class: "PGSerializer",
        selector: "newSharedTextureWithDescriptor:newTextureRef:allocator:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newSharedTextureWithHandle:allocator:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "fixStoreActions:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "addSplitHandler:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    // Tessellation. All four patch draws emit, and so do both pieces of state
    // they need -- this protocol carries tessellation in full. The two wide
    // forms share opcode 0x0c and are told apart by length (56 plain, 68
    // indexed), which is the only place in this family where a wide form does
    // not get its own opcode.
    //
    // These four were claimed as REFUSED_BY_SERIALIZER for one commit, on the
    // strength of the ray-tracing binds beside them having been refused. They
    // had not been driven. See this crate's AGENTS.md.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "drawPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:instanceCount:baseInstance:",
        opcodes: &[0x0d, 0x0c],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "drawIndexedPatches:patchStart:patchCount:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:instanceCount:baseInstance:",
        opcodes: &[0x0f, 0x0c],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "drawPatches:patchIndexBuffer:patchIndexBufferOffset:indirectBuffer:indirectBufferOffset:",
        opcodes: &[0x12],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "drawIndexedPatches:patchIndexBuffer:patchIndexBufferOffset:controlPointIndexBuffer:controlPointIndexBufferOffset:indirectBuffer:indirectBufferOffset:",
        opcodes: &[0x13],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    // The serializer's own encoder bookkeeping: splitting a long encoder across
    // command buffers, and the render pass's cleared/loaded state. None is a
    // Metal command a guest issues. Driven rather than assumed silent, because
    // `getTileDimensions:` looked like this too and emitted a record.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "forceLoadActions",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "forceStoreActionsForPosition:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setEncoderPosition:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "addRenderTargetReferences",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "split",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    // Refused by the serializer, which asserts rather than emitting.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setDepthCleared",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setStencilCleared",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "isMemorylessRender",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    // The five selectors this class shares with the blit and compute encoders.
    //
    // They were driven on one of those two and left undriven here, so five rows
    // of this class rested on nothing at all -- and the outcomes are not the
    // ones the other classes gave. `sampleCountersInBuffer:atSampleIndex:
    // withBarrier:` *emits* on both other encoders and is refused on this one,
    // and `writeDescriptor` needs `ComputePassDescriptorDispatchType` forced on
    // the compute encoder and emits here at the default state.
    //
    // The generalisable form of "a family is not uniform": the same selector on
    // a sibling class is not evidence either, and a class-shaped sweep does not
    // look for it because each class's own list already has the selector.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "flushWrites",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "handleSplits",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "getType",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "sampleCountersInBuffer:atSampleIndex:withBarrier:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    // The render pass descriptor, and the reason this triage was worth running.
    //
    // `0x1a` was recorded in `reims-vgpu` as an opcode "absent from Apple's
    // render manifest" and therefore not a serializer record -- a claim that
    // held only because no case had ever driven the selector that emits it.
    // `makeRenderEncoder`'s own doc had said all along that constructing an
    // encoder emits the render-pass record; what was missing was a way to
    // capture it, and `writeDescriptor` re-emits it on demand.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "writeDescriptor",
        opcodes: &[
            crate::ops::render_pass::OPCODE_RENDER_PASS,
            crate::ops::render_pass::OPCODE_DEFAULT_RASTER_SAMPLE_COUNT,
            crate::ops::render_pass::OPCODE_SAMPLE_POSITIONS,
            crate::ops::render_pass::OPCODE_RASTERIZATION_RATE_MAP,
            crate::ops::render_pass::OPCODE_IMAGEBLOCK_SAMPLE_LENGTH,
            crate::ops::render_pass::OPCODE_THREADGROUP_MEMORY_LENGTH,
            crate::ops::render_pass::OPCODE_TILE_SIZE,
        ],
        coverage: Coverage::Covered {
            module: "ops::render_pass",
        },
    },
    // The twenty ray-tracing binds on the vertex, fragment, mesh and object
    // stages. Every one asserts inside Apple's serializer rather than emitting,
    // exactly as the five tile-stage forms do -- so all twenty-five acceleration
    // structure and function-table binds this class declares are refused, and
    // this protocol carries no ray tracing on any render stage. The compute
    // encoder's five were already recorded the same way.
    //
    // Driven rather than generalised from the tile result: a family is not
    // uniform (the tile family itself splits fourteen emitting against five
    // refused), so each stage was captured.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexAccelerationStructure:atBufferIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexVisibleFunctionTable:atBufferIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexVisibleFunctionTables:withBufferRange:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexIntersectionFunctionTable:atBufferIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setVertexIntersectionFunctionTables:withBufferRange:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setFragmentAccelerationStructure:atBufferIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setFragmentVisibleFunctionTable:atBufferIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setFragmentVisibleFunctionTables:withBufferRange:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setFragmentIntersectionFunctionTable:atBufferIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setFragmentIntersectionFunctionTables:withBufferRange:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setMeshAccelerationStructure:atBufferIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setMeshVisibleFunctionTable:atBufferIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setMeshVisibleFunctionTables:withBufferRange:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setMeshIntersectionFunctionTable:atBufferIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setMeshIntersectionFunctionTables:withBufferRange:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setObjectAccelerationStructure:atBufferIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setObjectVisibleFunctionTable:atBufferIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setObjectVisibleFunctionTables:withBufferRange:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setObjectIntersectionFunctionTable:atBufferIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setObjectIntersectionFunctionTables:withBufferRange:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    // The store-action *options*, one opcode above each store action.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setColorStoreActionOptions:atIndex:",
        opcodes: &[0x67],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setDepthStoreActionOptions:",
        opcodes: &[0x6a],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setStencilStoreActionOptions:",
        opcodes: &[0x79],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    // The factor buffer emits; every draw that would consume it is refused.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTessellationFactorBuffer:offset:instanceStride:",
        opcodes: &[0x7a],
        coverage: Coverage::Covered {
            module: "ops::render",
        },
    },
    // Tile imageblock memory. The one selector in the tile family whose name
    // does not say "tile", and the reason 0x9c looked like a hole in the run.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setThreadgroupMemoryLength:offset:atIndex:",
        opcodes: &[0x9c],
        coverage: Coverage::Covered {
            module: "ops::tile",
        },
    },
    // The three MSAA resolve targets, the clip plane, the indexed
    // primitive-restart form and the transform-feedback state: all refused by
    // the serializer.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setColorResolveTexture:slice:depthPlane:level:yInvert:atIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setDepthResolveTexture:slice:depthPlane:level:yInvert:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setStencilResolveTexture:slice:depthPlane:level:yInvert:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setClipPlane:p2:p3:p4:atIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setPrimitiveRestartEnabled:index:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTransformFeedbackState:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    // Silent at *both* MTLDepthClipMode values, so this is not a serializer
    // skipping a redundant write -- the SPI spelling emits nothing while its
    // public sibling `setDepthClipMode:` emits 0x6d at either.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setDepthClipModeSPI:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    // The tile-shader family, all twenty-one selectors.
    //
    // Every row here rests on `-supportsTileShaders` being forced on. At its
    // default of false all nineteen non-property selectors run and write
    // nothing, so a capture taken without `withCapability` would have made this
    // the largest block of `EMITS_NO_OPERATION` rows in the manifest — nineteen
    // false claims that Apple's serializer emits nothing for a tile shader.
    //
    // Fourteen emit, across nine opcodes. Five are refused by the serializer
    // itself: the ray-tracing binds assert rather than serialize, which is a
    // statement about Apple and not about this harness. Two are property reads
    // and were driven anyway, because `getTileDimensions:` looked like a pure
    // query too and turned out to emit a record.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "dispatchThreadsPerTile:",
        opcodes: &[0x9b],
        coverage: Coverage::Covered {
            module: "ops::tile",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "dispatchThreadsPerTile:inRegion:",
        opcodes: &[0xa2],
        coverage: Coverage::Covered {
            module: "ops::tile",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "dispatchThreadsPerTile:inRegion:withRenderTargetArrayIndex:",
        opcodes: &[0xa3],
        coverage: Coverage::Covered {
            module: "ops::tile",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTileBuffer:offset:atIndex:",
        opcodes: &[0x9d],
        coverage: Coverage::Covered {
            module: "ops::tile",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTileBuffers:offsets:withRange:",
        opcodes: &[0x9d],
        coverage: Coverage::Covered {
            module: "ops::tile",
        },
    },
    // No opcode of its own: the bytes are staged through the command stream and
    // the record names the staging buffer, exactly as the vertex and fragment
    // `Bytes` forms do.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTileBytes:length:atIndex:",
        opcodes: &[0x9d],
        coverage: Coverage::Covered {
            module: "ops::tile",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTileBufferOffset:atIndex:",
        opcodes: &[0x9e],
        coverage: Coverage::Covered {
            module: "ops::tile",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTileSamplerState:atIndex:",
        opcodes: &[0x9f],
        coverage: Coverage::Covered {
            module: "ops::tile",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTileSamplerStates:withRange:",
        opcodes: &[0x9f],
        coverage: Coverage::Covered {
            module: "ops::tile",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTileSamplerState:lodMinClamp:lodMaxClamp:atIndex:",
        opcodes: &[0xa0],
        coverage: Coverage::Covered {
            module: "ops::tile",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTileSamplerStates:lodMinClamps:lodMaxClamps:withRange:",
        opcodes: &[0xa0],
        coverage: Coverage::Covered {
            module: "ops::tile",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTileTexture:atIndex:",
        opcodes: &[0xa1],
        coverage: Coverage::Covered {
            module: "ops::tile",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTileTextures:withRange:",
        opcodes: &[0xa1],
        coverage: Coverage::Covered {
            module: "ops::tile",
        },
    },
    // Runs the protocol backwards: the guest names a buffer for the *host* to
    // write the tile width and height into.
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "getTileDimensions:",
        opcodes: &[0xa4],
        coverage: Coverage::Covered {
            module: "ops::tile",
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTileAccelerationStructure:atBufferIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTileVisibleFunctionTable:atBufferIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTileVisibleFunctionTables:withBufferRange:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTileIntersectionFunctionTable:atBufferIndex:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "setTileIntersectionFunctionTables:withBufferRange:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: REFUSED_BY_SERIALIZER,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "tileWidth",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "tileHeight",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: ".cxx_destruct",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: NOT_A_COMMAND,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "dealloc",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: NOT_A_COMMAND,
        },
    },
    Entry {
        class: "PGSerializerRenderCommandEncoder",
        selector: "initWithCommandBuffer:descriptor:serializer:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: NOT_A_COMMAND,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "dealloc",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: NOT_A_COMMAND,
        },
    },
    // The two designated initializers. `PGSerializer` has both because the
    // second takes a deserializer version -- which is the host's answer to
    // "what wire does this guest speak", settled before any record is written
    // rather than by one.
    Entry {
        class: "PGSerializer",
        selector: "initWithDevice:objectRefAllocator:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: NOT_A_COMMAND,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "initWithDevice:objectRefAllocator:deserializerVersion:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: NOT_A_COMMAND,
        },
    },
    // --- PGSerializer object lifecycle -------------------------------------
    //
    // Three families of eleven, one per object kind. Only `-deleteXRef:` writes
    // a record; `-newXRef` allocates a ref host-side and `-releaseXRef:` is
    // host-side accounting, and both are measured silent every capture rather
    // than assumed to be. See [`crate::ops::destroy`], which also carries what
    // the opcode span does *not* say.
    Entry {
        class: "PGSerializer",
        selector: "deleteBufferRef:allocator:",
        opcodes: &[0x3e8],
        coverage: Coverage::Covered {
            module: "ops::destroy",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "deleteComputePipelineStateRef:allocator:",
        opcodes: &[0x3ee],
        coverage: Coverage::Covered {
            module: "ops::destroy",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "deleteDepthStencilStateRef:allocator:",
        opcodes: &[0x3ea],
        coverage: Coverage::Covered {
            module: "ops::destroy",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "deleteFenceRef:allocator:",
        opcodes: &[0x3f1],
        coverage: Coverage::Covered {
            module: "ops::destroy",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "deleteFunctionRef:allocator:",
        opcodes: &[0x3ed],
        coverage: Coverage::Covered {
            module: "ops::destroy",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "deleteHeapRef:allocator:",
        opcodes: &[0x3f4],
        coverage: Coverage::Covered {
            module: "ops::destroy",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "deleteIndirectCommandBufferRef:allocator:",
        opcodes: &[0x3f7],
        coverage: Coverage::Covered {
            module: "ops::destroy",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "deleteRasterizationRateMapRef:allocator:",
        opcodes: &[0x3f6],
        coverage: Coverage::Covered {
            module: "ops::destroy",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "deleteRenderPipelineStateRef:allocator:",
        opcodes: &[0x3ef],
        coverage: Coverage::Covered {
            module: "ops::destroy",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "deleteSamplerStateRef:allocator:",
        opcodes: &[0x3eb],
        coverage: Coverage::Covered {
            module: "ops::destroy",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "deleteTextureRef:allocator:",
        opcodes: &[0x3e9],
        coverage: Coverage::Covered {
            module: "ops::destroy",
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newBufferRef",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newComputePipelineStateRef",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newDepthStencilStateRef",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newFenceRef",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newFunctionRef",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newHeapRef",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newIndirectCommandBufferRef",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newRasterizationRateMapRef",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newRenderPipelineStateRef",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newSamplerStateRef",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "newTextureRef",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "releaseBufferRef:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "releaseComputePipelineStateRef:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "releaseDepthStencilStateRef:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "releaseFenceRef:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "releaseFunctionRef:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "releaseHeapRef:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "releaseIndirectCommandBufferRef:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "releaseRasterizationRateMapRef:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "releaseRenderPipelineStateRef:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "releaseSamplerStateRef:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "releaseTextureRef:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    // --- PGSerializer capability flags -------------------------------------
    //
    // Sixteen getter/setter pairs and three read-only flags, all driven and all
    // silent. The setters are the ones worth having driven: they are inverted
    // and restored by `capabilityCases`, so "emits nothing" is measured against
    // a real state change rather than against a write of the current value,
    // which a setter that serialized only on a change would have passed.
    Entry {
        class: "PGSerializer",
        selector: "supportsBlitEncoderSPI",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "setSupportsBlitEncoderSPI:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "supportsCommandBufferJump",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "setSupportsCommandBufferJump:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "supportsComputePassDescriptorDispatchType",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "setSupportsComputePassDescriptorDispatchType:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "supportsDefaultRasterSampleCount",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "setSupportsDefaultRasterSampleCount:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "supportsDispatchThreadsIndirect",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "setSupportsDispatchThreadsIndirect:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "supportsDynamicAttributeStride",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "setSupportsDynamicAttributeStride:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "supportsImageBlocks",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "setSupportsImageBlocks:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "supportsInfoIndirect",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "setSupportsInfoIndirect:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "supportsInsertCompressedTextureReinterpretationFlush",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "setSupportsInsertCompressedTextureReinterpretationFlush:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "supportsProgrammableSamplePositions",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "setSupportsProgrammableSamplePositions:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "supportsProtectionOptionsEnvelope",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "setSupportsProtectionOptionsEnvelope:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "supportsRasterizationRateMap",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "setSupportsRasterizationRateMap:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "supportsSwizzledTextures",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "setSupportsSwizzledTextures:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "supportsTextureDescriptor2",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "setSupportsTextureDescriptor2:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "supportsTileShaders",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "setSupportsTileShaders:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "supportsVertexAmplification",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "setSupportsVertexAmplification:",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "supportsCorrectBaseVertex",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "supportsOpenGL",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
    Entry {
        class: "PGSerializer",
        selector: "supportsSharedTextures",
        opcodes: &[],
        coverage: Coverage::Excluded {
            reason: EMITS_NO_OPERATION,
        },
    },
];

/// Tally of [`MANIFEST`] rows by state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    pub covered: usize,
    pub unimplemented: usize,
    pub excluded: usize,
}

impl Counts {
    #[inline]
    pub fn rows(&self) -> usize {
        self.covered + self.unimplemented + self.excluded
    }
}

/// Count the manifest by state.
pub fn counts() -> Counts {
    let mut c = Counts::default();
    for e in MANIFEST {
        match e.coverage {
            Coverage::Covered { .. } | Coverage::CoveredNoFixedOpcode { .. } => c.covered += 1,
            Coverage::Unimplemented => c.unimplemented += 1,
            Coverage::Excluded { .. } => c.excluded += 1,
        }
    }
    c
}

/// Selectors Apple ships that the manifest has no row for.
///
/// This is the real distance to exhaustive. It is deliberately a large number
/// today; the point is that it is a *number* rather than an impression.
pub fn untriaged() -> usize {
    INVENTORY
        .iter()
        .map(|c| c.instance_methods)
        .sum::<usize>()
        .saturating_sub(counts().rows())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_selector_appears_twice_under_the_same_class() {
        for (i, a) in MANIFEST.iter().enumerate() {
            for b in &MANIFEST[i + 1..] {
                assert!(
                    !(a.class == b.class && a.selector == b.selector),
                    "duplicate manifest row: -[{} {}]",
                    a.class,
                    a.selector
                );
            }
        }
    }

    #[test]
    fn every_class_named_by_a_row_is_one_the_inventory_knows() {
        // A typo in a class name would otherwise quietly park a row outside the
        // surface the inventory measures, inflating coverage.
        for e in MANIFEST {
            assert!(
                INVENTORY.iter().any(|c| c.class == e.class),
                "row -[{} {}] names a class absent from INVENTORY",
                e.class,
                e.selector
            );
        }
    }

    #[test]
    fn a_covered_row_carries_an_opcode_and_an_excluded_row_carries_none() {
        for e in MANIFEST {
            match e.coverage {
                Coverage::Covered { module } => {
                    assert!(!module.is_empty(), "{} has an empty module", e.selector);
                    assert!(
                        !e.opcodes.is_empty(),
                        "-[{} {}] is covered but records no opcode",
                        e.class,
                        e.selector
                    );
                }
                Coverage::Excluded { reason } => {
                    assert!(
                        !reason.is_empty(),
                        "{} excluded without a reason",
                        e.selector
                    );
                    assert!(
                        e.opcodes.is_empty(),
                        "-[{} {}] is excluded as emitting nothing, yet records opcodes {:?}",
                        e.class,
                        e.selector,
                        e.opcodes
                    );
                }
                Coverage::CoveredNoFixedOpcode { module } => {
                    assert!(!module.is_empty(), "{} has an empty module", e.selector);
                    // The inverse of the arm above, and the point of the state.
                    // A row here whose opcode list filled up would be claiming
                    // the selector has an opcode of its own, which is exactly
                    // the thing this state says it does not.
                    assert!(
                        e.opcodes.is_empty(),
                        "-[{} {}] has no opcode of its own, yet records {:?} \
                         as if it did",
                        e.class,
                        e.selector,
                        e.opcodes
                    );
                }
                Coverage::Unimplemented => {}
            }
        }
    }

    /// Two selectors may share an opcode, but only into the same view.
    ///
    /// Sharing is real: `setStencilReferenceValue:` and
    /// `setStencilFrontReferenceValue:backReferenceValue:` both emit `0x77`,
    /// the one-argument form writing its value into both fields. So uniqueness
    /// is the wrong invariant. What must hold is that one opcode has one
    /// reading — two rows pointing the same record at different modules would
    /// mean a decoder has to know which selector the guest called, and the wire
    /// does not say.
    ///
    /// This is also the check that catches the copy-paste the draw family
    /// invites: twelve opcodes across six rows, where giving two rows the same
    /// wide sibling leaves one record read by the wrong view while both rows
    /// still say `Covered`.
    #[test]
    fn an_opcode_shared_by_two_selectors_is_read_by_one_module() {
        for (i, a) in MANIFEST.iter().enumerate() {
            for b in &MANIFEST[i + 1..] {
                if a.class != b.class {
                    continue;
                }
                for op in a.opcodes {
                    if !b.opcodes.contains(op) {
                        continue;
                    }
                    assert_eq!(
                        a.coverage, b.coverage,
                        "opcode {op:#x} is claimed by -[{} {}] and -[{} {}] with \
                         different coverage; one record cannot have two readings",
                        a.class, a.selector, b.class, b.selector
                    );
                }
            }
        }
    }

    /// A row may not name the same opcode twice.
    #[test]
    fn a_rows_opcode_list_has_no_repeats() {
        for e in MANIFEST {
            for (i, op) in e.opcodes.iter().enumerate() {
                assert!(
                    !e.opcodes[i + 1..].contains(op),
                    "-[{} {}] lists opcode {op:#x} twice",
                    e.class,
                    e.selector
                );
            }
        }
    }

    #[test]
    fn the_distance_to_exhaustive_is_reported_rather_than_rounded_away() {
        let c = counts();
        let gap = untriaged();
        // Not an assertion about the value — an assertion that the arithmetic
        // stays honest as rows are added.
        assert_eq!(
            c.rows() + gap,
            INVENTORY.iter().map(|c| c.instance_methods).sum::<usize>()
        );
        assert!(c.covered >= 1, "the worked example must stay covered");
    }
}
