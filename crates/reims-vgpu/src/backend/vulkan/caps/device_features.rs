//! The one place a Vulkan device feature or format capability is both
//! **queried** and **enabled**.
//!
//! # The bug this retires
//!
//! `translate::sampler::address_mode` maps `MTLSamplerAddressModeMirrorClampToEdge`
//! to `vk::SamplerAddressMode::MIRROR_CLAMP_TO_EDGE`. That address mode requires
//! either `VkPhysicalDeviceVulkan12Features::samplerMirrorClampToEdge` or
//! `VK_KHR_sampler_mirror_clamp_to_edge`, and neither was ever requested — a
//! repo-wide search for any spelling of it returned zero hits outside the
//! translation table itself. The sampler was created with a mode the device had
//! not been asked for, which is undefined behaviour that a validation layer
//! catches on someone else's GPU and a shipping driver may simply honour.
//!
//! [`super`] already owns memory topology, zero-copy rails, driver quirks and
//! device selection, and its gate keeps those decisions in one place. It did
//! **not** own sampler or format features, so `context.rs` queried those inline —
//! correctly, in every case but one. Because there was no home, the one that got
//! missed got missed silently. This module is the home.
//!
//! # Two rules
//!
//! 1. **Query and enable together.** A feature that is asked about here and not
//!    enabled here is the same bug in a new place, so the enable list is built
//!    from this struct and nothing else.
//! 2. **Enable only what the backend binds.** `multi_viewport` was once enabled
//!    while `engine::exec` declined every draw with more than one viewport, and
//!    was later disabled to match. It is enabled again now, and this time
//!    because the backend reaches it: a pipeline's `viewportCount` is the
//!    guest's own. The rule is what caught both states — a list that is not
//!    derived from what the backend binds cannot be checked.

use ash::vk;

use crate::contract::pixel_format::TexelLayout;

/// How this device can satisfy `MTLSamplerAddressModeMirrorClampToEdge`.
///
/// Three rungs rather than a bool, because *how* it is available decides what
/// must be enabled at device creation, and "not available" has to be an
/// answer the sampler path can act on rather than a silent bind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MirrorClampToEdge {
    /// `VkPhysicalDeviceVulkan12Features::samplerMirrorClampToEdge`. Preferred:
    /// core on the 1.2 baseline every matrix cell meets, so no extension string.
    Core12,
    /// `VK_KHR_sampler_mirror_clamp_to_edge`. The pre-1.2 spelling, still the
    /// only one some drivers advertise.
    KhrExtension,
    /// Neither. The address mode must be **declined by name** at the sampler
    /// binding site — never bound ungated. This is the default so a
    /// `DeviceFeatures` built without a query never claims support it has not
    /// checked for.
    #[default]
    Unsupported,
}

impl MirrorClampToEdge {
    pub fn is_available(self) -> bool {
        !matches!(self, Self::Unsupported)
    }
}

/// How this device can satisfy Metal's guarantee that an out-of-bounds texture
/// read is **defined**.
///
/// # Why this is not an optimization
///
/// Every shader this device runs was compiled by Apple for Metal, and the Metal
/// Shading Language specifies out-of-range `texture.read`/`sample` coordinates:
/// reads return zero and writes are dropped. Apple's shaders are written against
/// that guarantee and use it — a blur or a convolution samples its neighbours at
/// fixed texel offsets and lets the taps that fall outside the image return
/// zero, rather than branching per tap.
///
/// Vulkan makes the same access **undefined** unless `robustImageAccess` is
/// enabled. `robustBufferAccess`, which this device does enable and which the
/// spec requires every implementation to support, covers buffers *only*; it says
/// nothing about image accesses. So without this feature every one of those taps
/// is undefined behaviour, and what a driver does with it is a driver's choice:
/// returning zero anyway, returning garbage, or faulting.
///
/// This is the same class as the `shaderInt64` gap — a capability the guest's
/// own modules depend on and this device was not asking for — and it is the one
/// remaining semantic difference between Metal's execution model and this
/// backend's that a validation layer cannot see, because undefined behaviour is
/// not invalid usage.
///
/// Three rungs, for the same reason `MirrorClampToEdge` has three: *how* it is
/// available decides what must be chained at device creation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ImageRobustness {
    /// `VkPhysicalDeviceVulkan13Features::robustImageAccess`. Preferred where the
    /// host is 1.3, because it needs no extension string.
    Core13,
    /// `VK_EXT_image_robustness`, chained as
    /// `VkPhysicalDeviceImageRobustnessFeaturesEXT`. The 1.2-baseline spelling,
    /// and the one this project's own baseline implies is the common case.
    ExtImageRobustness,
    /// Neither. The host cannot promise Metal's guarantee, and this device runs
    /// the guest's shaders anyway — there is nothing else it can do, since the
    /// alternative is refusing every module Apple compiled. The rung is
    /// **reported** so a boot on such a host says so, which is the difference
    /// between a known gap and a silent one. Default, so a `DeviceFeatures`
    /// built without a query never claims a promise it has not checked for.
    #[default]
    Unsupported,
}

impl ImageRobustness {
    pub fn is_available(self) -> bool {
        !matches!(self, Self::Unsupported)
    }
}

/// The `maxImageDimension2D` every Vulkan 1.2 implementation must report at
/// least (spec table "Required Limits"). Used only as the floor a queried
/// value is clamped to, and as the answer when no device has been resolved.
pub const VULKAN_MIN_IMAGE_DIMENSION_2D: u32 = 4096;

/// `maxComputeWorkGroupInvocations` from the same spec table. Used as the floor
/// a queried value is clamped to, and as the answer when no device is resolved.
pub const VULKAN_MIN_COMPUTE_WORKGROUP_INVOCATIONS: u32 = 128;

/// `maxComputeWorkGroupSize` from the same spec table: 128 in x and y, 64 in z.
pub const VULKAN_MIN_COMPUTE_WORKGROUP_SIZE: [u32; 3] = [128, 128, 64];

/// `maxComputeSharedMemorySize` from the same spec table — 16 KiB, half of what
/// the device-info table promised unconditionally.
pub const VULKAN_MIN_COMPUTE_SHARED_MEMORY_BYTES: u32 = 16384;

/// Every device feature and format capability this backend depends on, resolved
/// against one physical device.
///
/// Plain bools rather than the ash feature structs: this is the *decision*, and
/// keeping it free of `p_next` chains is what lets it be built once, asserted in
/// tests without a GPU, and consumed by the two spots that need ash types.
#[derive(Clone, Copy, Debug, Default)]
pub struct DeviceFeatures {
    /// Defined bounds-clamped behaviour for out-of-range shader buffer access.
    /// The one feature the spec requires every implementation to support.
    pub robust_buffer_access: bool,
    pub sampler_anisotropy: bool,
    pub max_sampler_anisotropy: f32,
    /// `VkPhysicalDeviceLimits::maxImageDimension2D` — the largest 2D image
    /// this device can create, and therefore the largest render target a draw
    /// may name. Read from the device rather than assumed: the Vulkan 1.2
    /// floor is 4096, which every desktop GPU exceeds by 4x, so treating the
    /// floor as the cap refuses render targets the host can actually hold.
    pub max_image_dimension_2d: u32,
    /// `VkPhysicalDeviceLimits::maxComputeWorkGroupInvocations` — the guest's
    /// `maxTotalThreadsPerThreadgroup`. The Vulkan 1.2 floor is 128, so a
    /// fixed 1024 is an over-promise on any device at or near it: the guest
    /// sizes threadgroups from this answer and the host then cannot run them.
    pub max_compute_workgroup_invocations: u32,
    /// `VkPhysicalDeviceSubgroupProperties::subgroupSize` — the guest's
    /// `threadExecutionWidth`. Vendor-dependent (32 on NVIDIA and Apple, 64 on
    /// AMD's wave64 parts, 8/16/32 on Intel), which is why it is asked for
    /// rather than assumed.
    pub subgroup_size: u32,
    /// `VkPhysicalDeviceLimits::maxComputeWorkGroupSize` — the guest's
    /// `maxThreadsPerThreadgroup` width/height/depth. Separate from
    /// [`Self::max_compute_workgroup_invocations`], which bounds their product:
    /// a device can allow 1024 in x and still refuse 1024x1024x64 threads.
    pub max_compute_workgroup_size: [u32; 3],
    /// `VkPhysicalDeviceLimits::maxComputeSharedMemorySize` — the guest's
    /// `maxThreadgroupMemoryLength`. The Vulkan 1.2 floor is 16 KiB, so
    /// answering a fixed 32 KiB is an over-promise on any device at the floor:
    /// the guest builds kernels declaring that much threadgroup memory and the
    /// host then refuses every pipeline made from them.
    pub max_compute_shared_memory_bytes: u32,
    /// Highest MSAA sample count usable for both colour and depth attachments
    /// *and* for sampled images — the intersection, because the guest gets one
    /// number and uses it for all three. Answering higher than the host can
    /// render makes the guest build multisample targets this device cannot
    /// create.
    pub max_sample_count: u32,
    /// `D24_UNORM_S8_UINT` usable as a depth/stencil attachment with optimal
    /// tiling. Not spec-mandatory and genuinely absent on some desktop drivers,
    /// which is why the guest is told rather than assumed: a guest that thinks
    /// it has a packed 24/8 depth format will name one.
    pub d24_unorm_s8_attachment: bool,
    pub shader_int16: bool,
    /// Dynamically uniform indexing into sampled-image descriptor arrays.
    pub sampled_image_array_dynamic_indexing: bool,
    /// Dynamically uniform indexing into storage-image descriptor arrays.
    pub storage_image_array_dynamic_indexing: bool,
    /// Effective sampled-image descriptor capacity for one stage and set.
    pub sampled_image_descriptor_limit: u32,
    /// Effective storage-image descriptor capacity for one stage and set.
    pub storage_image_descriptor_limit: u32,
    /// `VkPhysicalDeviceFeatures::shaderInt64` — whether a SPIR-V module may
    /// declare the `Int64` capability.
    ///
    /// Not authored here: the modules this backend creates are translated from
    /// the guest's AIR, and they declare `Int64` whenever the guest's shader
    /// used a 64-bit integer — which CoreAnimation's parameter blocks do. A
    /// module declaring a capability whose feature is not enabled is undefined
    /// behaviour, not a decode error, and it is the shape that survives on one
    /// driver and breaks on another: this was live on every boot, unnoticed,
    /// until a validation run named it.
    pub shader_int64: bool,
    /// `VkPhysicalDeviceFeatures::fragmentStoresAndAtomics` — whether a
    /// fragment shader's storage buffers and images may be written.
    ///
    /// Without it, every such variable in a fragment stage must carry the
    /// `NonWritable` decoration, and a translated module carries whatever the
    /// guest's own shader implied. This backend does not rewrite decorations,
    /// so the only correct move is to ask for the feature where the host has
    /// it. Same provenance as [`Self::shader_int64`] — a validation run named
    /// it, and it had been undefined behaviour on every boot before that.
    pub fragment_stores_and_atomics: bool,
    /// `VkPhysicalDeviceFeatures::vertexPipelineStoresAndAtomics` — the same
    /// rule as [`Self::fragment_stores_and_atomics`] for the vertex,
    /// tessellation and geometry stages. Asked for separately because a device
    /// may have one and not the other.
    pub vertex_pipeline_stores_and_atomics: bool,
    pub storage_image_extended_formats: bool,
    pub storage_image_write_without_format: bool,
    /// `shaderStorageImageReadWithoutFormat`. The read half of the pair above:
    /// an `OpImageRead` from an `Unknown`-format storage image needs it, and a
    /// translated kernel may contain one whether or not this device asked for
    /// the format-less view. Requested so the SPIR-V capability can be declared
    /// when a module turns out to need it.
    pub storage_image_read_without_format: bool,
    /// `B8G8R8A8_UNORM` usable as a storage image with optimal tiling. **Not**
    /// spec-mandatory — only `R8G8B8A8_UNORM` is — so the BGRA composite path
    /// needs this *and* `storage_image_write_without_format`.
    pub bgra8_storage: bool,
    /// For each [`TexelLayout`], indexed by [`TexelLayout::index`], whether its
    /// Vulkan format is usable as a sampled image with **linear** filtering
    /// under optimal tiling.
    ///
    /// The native sampled rails bind a guest texel layout straight to an image
    /// and let the sampler read it, with interpolation, so a layout the host
    /// cannot filter is one those rails must decline. This used to be a single
    /// `bool` for `R32_SFLOAT` — the one layout then known to be optional —
    /// and every other layout was admitted on the reading that the spec's
    /// mandatory-format table covered it. That reading is an API-version
    /// assumption of exactly the kind `AGENTS.md` says to replace with a
    /// capability, and it does not even hold for the set already here:
    /// `R16_UNORM`'s linear-filter feature is optional too.
    ///
    /// Asking per layout also makes the gate impossible to forget. A new
    /// [`TexelLayout`] gets an entry because the array is sized by
    /// `TexelLayout::ALL.len()`, rather than needing someone to remember to add
    /// a second `bool`.
    pub sampled_linear_filter: [bool; TexelLayout::ALL.len()],
    /// For each [`TexelLayout`], indexed by [`TexelLayout::index`], whether its
    /// Vulkan format is usable as a **colour attachment** under optimal tiling,
    /// blending included wherever blending is a question the format can be
    /// asked.
    ///
    /// A render target's resident is created at the format the guest declared
    /// for the attachment, so a layout this host cannot render into is one that
    /// must fall back to the engine's eight-bit resident rather than be
    /// attempted.
    ///
    /// **The two bits are required together for a continuous-range layout and
    /// `COLOR_ATTACHMENT` alone for an integer one**, and that split is the
    /// contract rather than a leniency. A colour attachment that cannot blend
    /// is not a usable render target for a compositor, so admitting a unorm or
    /// float layout on `COLOR_ATTACHMENT` alone would trade a fidelity loss for
    /// a pipeline the driver refuses. But Vulkan mandates *no*
    /// `COLOR_ATTACHMENT_BLEND` for an integer format and no host advertises
    /// it, so requiring it of one asks a question whose only possible answer is
    /// `false` — which is how every `RG16Uint` render target came to be built
    /// at eight bits and lost at a later rung. An integer attachment is never
    /// blended: `TexelLayout::is_integer` is the same predicate that keeps it
    /// out of the sampled-filter vocabulary, one question over.
    ///
    /// Asked per layout for the same reason as [`Self::sampled_linear_filter`]
    /// directly above: the array is sized by `ALL.len()`, so a new
    /// [`TexelLayout`] cannot reach a render target without getting a probe.
    /// `R16G16B16A16_SFLOAT` is in Vulkan's mandatory format table for both
    /// bits, but AGENTS.md is explicit that a capability comes from the device
    /// and not from a reading of the spec's table.
    pub color_attachment: [bool; TexelLayout::ALL.len()],
    pub storage16: bool,
    pub storage8: bool,
    pub float16: bool,
    pub int8: bool,
    pub shader_output_viewport_index: bool,
    /// `VkPhysicalDeviceVulkan12Features::timelineSemaphore` — whether a
    /// submission can signal a monotonic counter that a *second* thread may
    /// wait on.
    ///
    /// This is what lets a completion be observed without owning the thing that
    /// produced it. A `VkFence` has one waiter's worth of lifetime and the ring
    /// already owns every fence it has — resetting them at retire — so a second
    /// thread waiting on a ring fence races the reset. A timeline value is
    /// monotonic, waitable from anywhere, and needs nothing back.
    ///
    /// Core in Vulkan 1.2, which is this backend's baseline, so a device that
    /// declines it is out of spec rather than merely old. Asked anyway, and
    /// gated on: the rail that uses it falls back to blocking the drain worker,
    /// which is what every host did before it existed.
    pub timeline_semaphore: bool,
    /// `descriptorBindingPartiallyBound`, used for Metal texture-handle arrays.
    /// Metal permits array slots to be nil; Vulkan descriptor arrays otherwise
    /// require every statically declared element to contain a valid descriptor.
    /// Hosts without this optional Vulkan 1.2 feature decline those shaders by
    /// name rather than allocating invented textures or leaving invalid slots.
    pub descriptor_binding_partially_bound: bool,
    pub mirror_clamp_to_edge: MirrorClampToEdge,
    /// How this host can promise that an out-of-bounds texture read is defined,
    /// which every Metal shader is entitled to assume. See [`ImageRobustness`].
    pub image_robustness: ImageRobustness,
    /// `VK_EXT_attachment_feedback_loop_layout`: whether a render attachment
    /// may also be bound as a sampled image in the extension's explicit
    /// feedback-loop layout. This is the native Vulkan spelling of Metal's
    /// attachment self-sampling contract; hosts without it keep the snapshot
    /// copy rail.
    pub attachment_feedback_loop_layout: bool,
    /// `VK_EXT_image_drm_format_modifier`, used only for the explicit linear
    /// plane layout that gives a shared guest target its declared row pitch.
    /// This is an extension capability rather than a device feature bit: when
    /// absent, imported targets keep the ordinary linear-layout probe and its
    /// copied fallback.
    pub image_drm_format_modifier: bool,
    /// `VkPhysicalDeviceFeatures::dualSrcBlend` — whether a pipeline may name
    /// the `SRC1_*` blend factors, which read the fragment shader's second
    /// colour output.
    ///
    /// Same shape as [`Self::mirror_clamp_to_edge`] and here for the same
    /// reason: `MTLBlendFactor` 15-18 are the dual-source factors, the
    /// translation table now spells them, and a pipeline naming one on a device
    /// that does not advertise this is invalid. Optional in core Vulkan — not
    /// an extension, but not mandatory either — so it is asked rather than
    /// assumed, and the pipeline path declines by name where it is absent.
    pub dual_src_blend: bool,
    /// `VkPhysicalDeviceFeatures::fillModeNonSolid` — whether a pipeline may
    /// name `VK_POLYGON_MODE_LINE` or `_POINT`.
    ///
    /// `MTLTriangleFillModeLines` has no other spelling: the polygon mode is
    /// Vulkan's only way to rasterize a triangle as its edges, and naming a
    /// non-solid one on a device that does not advertise this makes the
    /// pipeline invalid. Same shape as [`Self::dual_src_blend`] — optional
    /// core, asked rather than assumed, declined by name where absent.
    pub fill_mode_non_solid: bool,
    /// `VkPhysicalDeviceFeatures::textureCompressionBC` — whether this device
    /// can sample the BC (DXT / S3TC) block-compressed families.
    ///
    /// One bit covers BC1 through BC7, which is why
    /// `pixel_format::MTL_FORMAT_BC1_RGBA`'s doc admits the family whole: there
    /// is no per-member capability to measure. Enabling it also brings the
    /// guarantees the sampled rail relies on — Vulkan's mandatory-format table
    /// requires `SAMPLED_IMAGE`, `SAMPLED_IMAGE_FILTER_LINEAR` and `BLIT_SRC`
    /// of every BC format on a device that has this feature enabled, so no
    /// per-format query is owed either.
    ///
    /// **Asked and enabled rather than assumed**, and this one genuinely
    /// divides hosts: desktop GPUs have it and Apple GPUs do not — they carry
    /// ASTC instead — so the arm64/Metal pathways and MoltenVK refuse a BC
    /// bind by name. Creating a BC image without the feature enabled is invalid
    /// use, not a slower path, which is why the gate is at
    /// `draw::texture_view::NativeUploads` and not a `#[cfg]`.
    pub texture_compression_bc: bool,
    /// `VkPhysicalDeviceFeatures::depthClamp` — whether a pipeline may set
    /// `depthClampEnable`.
    ///
    /// This is what `MTLDepthClipModeClamp` asks for: a fragment outside the
    /// depth range is clamped to it rather than discarded. Optional core like
    /// the two above.
    pub depth_clamp: bool,
    /// `VkPhysicalDeviceFeatures::multiViewport` — whether a pipeline may
    /// declare more than one viewport/scissor slot.
    ///
    /// This is what `setViewports:count:` with a count above one asks for.
    /// Optional core like the two above, and paired with [`Self::max_viewports`]
    /// because the feature only says "more than one is allowed" and the limit
    /// says how many.
    pub multi_viewport: bool,
    /// `VkPhysicalDeviceLimits::maxViewports` — the largest slot count a
    /// pipeline may declare.
    ///
    /// At least 1 on every device, and at least 16 wherever
    /// [`Self::multi_viewport`] is set. Carried as its own number rather than
    /// assumed from the feature: the guarantee is a floor, and a guest may ask
    /// for more than the floor.
    pub max_viewports: u32,
    /// `VkPhysicalDeviceFeatures::occlusionQueryPrecise` — whether an occlusion
    /// query may be recorded with `VK_QUERY_CONTROL_PRECISE_BIT`.
    ///
    /// This is what `MTLVisibilityResultModeCounting` asks for. Without the bit
    /// a Vulkan occlusion query promises only "non-zero if any sample passed",
    /// which is `MTLVisibilityResultModeBoolean` exactly — so the query *type*
    /// needs no feature and no limit, and this gates only the counting arm.
    /// `engine::exec` refuses a counting draw where it is unset rather than
    /// recording without the bit: an imprecise count is a plausible wrong
    /// number, which is worse than a named refusal.
    pub occlusion_query_precise: bool,
}

impl DeviceFeatures {
    /// Whether this device can bind a Metal sampled-texture handle array.
    /// Both halves are required: the shader indexes an array, and Metal permits
    /// the guest to leave array elements nil.
    pub fn sampled_descriptor_arrays(&self, required_descriptors: u32) -> bool {
        self.descriptor_binding_partially_bound
            && self.sampled_image_array_dynamic_indexing
            && required_descriptors <= self.sampled_image_descriptor_limit
    }

    /// Storage-image counterpart of [`Self::sampled_descriptor_arrays`].
    pub fn storage_descriptor_arrays(&self, required_descriptors: u32) -> bool {
        self.descriptor_binding_partially_bound
            && self.storage_image_array_dynamic_indexing
            && required_descriptors <= self.storage_image_descriptor_limit
    }

    /// The BGRA-storage composite path needs the format-less write feature and
    /// BGRA8 as a usable storage image. Named once so the pair cannot drift
    /// apart at a call site.
    pub fn storage_image_write_without_format_bgra(&self) -> bool {
        self.storage_image_write_without_format && self.bgra8_storage
    }

    /// The `vk::PhysicalDeviceFeatures` to enable, derived from what is
    /// supported **and** what the backend actually binds.
    ///
    /// `multi_viewport` is bound where supported: `engine::exec` builds a
    /// pipeline whose `viewportCount` is the guest's, and a count above one is
    /// invalid without it.
    ///
    /// `occlusion_query_precise` likewise: `engine::exec` records
    /// `vkCmdBeginQuery` with `PRECISE` for a counting draw, and passing that
    /// bit is invalid without the feature enabled.
    ///
    /// `shader_int64`, `fragment_stores_and_atomics` and
    /// `vertex_pipeline_stores_and_atomics` are bound in a way the rule above
    /// does not read at a call site: they are properties of the **translated
    /// SPIR-V**, not of anything this crate spells. The backend binds them
    /// because it hands `vkCreateShaderModule` a module the guest's own shader
    /// decided the shape of, so "what the backend binds" is whatever the guest
    /// compiled — and asking for less than that is undefined behaviour rather
    /// than a narrower device.
    pub fn enabled_features(&self) -> vk::PhysicalDeviceFeatures {
        vk::PhysicalDeviceFeatures::default()
            .robust_buffer_access(self.robust_buffer_access)
            .sampler_anisotropy(self.sampler_anisotropy)
            .shader_int16(self.shader_int16)
            .shader_sampled_image_array_dynamic_indexing(self.sampled_image_array_dynamic_indexing)
            .shader_storage_image_array_dynamic_indexing(self.storage_image_array_dynamic_indexing)
            .shader_int64(self.shader_int64)
            .fragment_stores_and_atomics(self.fragment_stores_and_atomics)
            .vertex_pipeline_stores_and_atomics(self.vertex_pipeline_stores_and_atomics)
            .shader_storage_image_extended_formats(self.storage_image_extended_formats)
            .shader_storage_image_write_without_format(self.storage_image_write_without_format)
            .shader_storage_image_read_without_format(self.storage_image_read_without_format)
            .dual_src_blend(self.dual_src_blend)
            .fill_mode_non_solid(self.fill_mode_non_solid)
            .texture_compression_bc(self.texture_compression_bc)
            .depth_clamp(self.depth_clamp)
            .multi_viewport(self.multi_viewport)
            .occlusion_query_precise(self.occlusion_query_precise)
    }

    /// The Vulkan 1.2 features to enable.
    ///
    /// `sampler_mirror_clamp_to_edge` is set only on the [`MirrorClampToEdge::Core12`]
    /// rung; on [`MirrorClampToEdge::KhrExtension`] the extension string carries
    /// it instead, and on [`MirrorClampToEdge::Unsupported`] nothing is
    /// requested and the sampler path declines.
    ///
    /// # Why `storage8`, `float16` and `int8` are set *here* and not in their
    /// own structs
    ///
    /// `VkPhysicalDevice8BitStorageFeatures` and
    /// `VkPhysicalDeviceShaderFloat16Int8Features` were promoted into this
    /// struct at Vulkan 1.2, and the spec forbids chaining a promoted struct
    /// alongside the one that absorbed it
    /// (`VUID-VkDeviceCreateInfo-pNext-02830`) — precisely so that one cannot
    /// say `VK_TRUE` while the other says `VK_FALSE` for the same feature. This
    /// device used to chain all three, with the promoted spellings left
    /// `VK_FALSE` in this struct, so which value took effect was the driver's
    /// choice of traversal. Every implementation observed took the union, so
    /// nothing was lost in practice; it was still a contradiction the spec does
    /// not define an answer for, and 1.2 is this backend's baseline, so the
    /// promoted spelling is the only one needed.
    ///
    /// `storage16` is **not** here: 16-bit storage was promoted into
    /// `VkPhysicalDeviceVulkan11Features`, which this chain does not carry, so
    /// `VkPhysicalDevice16BitStorageFeatures` conflicts with nothing and stays
    /// its own struct.
    pub fn enabled_vulkan12(&self) -> vk::PhysicalDeviceVulkan12Features<'static> {
        vk::PhysicalDeviceVulkan12Features::default()
            .shader_output_viewport_index(self.shader_output_viewport_index)
            .timeline_semaphore(self.timeline_semaphore)
            .descriptor_binding_partially_bound(self.descriptor_binding_partially_bound)
            .sampler_mirror_clamp_to_edge(self.mirror_clamp_to_edge == MirrorClampToEdge::Core12)
            .storage_buffer8_bit_access(self.storage8)
            .shader_float16(self.float16)
            .shader_int8(self.int8)
    }

    /// Metal's out-of-bounds texture-read guarantee, when the host can make it.
    ///
    /// Chained by its `EXT` spelling on **both** available rungs.
    /// `VK_EXT_image_robustness` was promoted to core at Vulkan 1.3 and its
    /// feature struct is an alias of `VkPhysicalDeviceVulkan13Features`'s field,
    /// so a 1.3 driver accepts either; using one spelling means there is no
    /// second one to disagree with it, which is the trap
    /// `VUID-VkDeviceCreateInfo-pNext-02830` exists for and which this device
    /// has already been caught by once. What the rung decides is only whether an
    /// extension *string* is also named — see [`Self::required_extensions`].
    pub fn enabled_image_robustness(
        &self,
    ) -> vk::PhysicalDeviceImageRobustnessFeaturesEXT<'static> {
        vk::PhysicalDeviceImageRobustnessFeaturesEXT::default()
            .robust_image_access(self.image_robustness.is_available())
    }

    pub fn enabled_attachment_feedback_loop_layout(
        &self,
    ) -> vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT<'static> {
        vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT::default()
            .attachment_feedback_loop_layout(self.attachment_feedback_loop_layout)
    }

    /// 16-bit storage-buffer access, for shaders that pack half-precision data.
    pub fn enabled_16bit_storage(&self) -> vk::PhysicalDevice16BitStorageFeatures<'static> {
        vk::PhysicalDevice16BitStorageFeatures::default()
            .storage_buffer16_bit_access(self.storage16)
    }

    /// One line naming every feature and limit this backend resolved against the
    /// bound device, so a boot says what it turned on and what it did without.
    ///
    /// # Why this destructures instead of reading fields
    ///
    /// A report built from field accesses goes stale the moment a field is
    /// added: the new capability is queried, gates a rail, and is invisible in
    /// every log — which is the same silence `device_features` was created to
    /// end, one level up. A `let Self { .. }` pattern with no rest binding makes
    /// the compiler refuse to build until the new field is named here, so the
    /// line cannot fall behind the struct. Do not add `..` to it.
    ///
    /// The two per-layout arrays are reported as the layouts that came back
    /// **false**, because that is the actionable set and it is usually empty; a
    /// bitfield per layout would be denser and unreadable in a bug report.
    pub fn report_line(&self) -> String {
        let Self {
            robust_buffer_access,
            sampler_anisotropy,
            max_sampler_anisotropy,
            max_image_dimension_2d,
            max_compute_workgroup_invocations,
            subgroup_size,
            max_compute_workgroup_size,
            max_compute_shared_memory_bytes,
            max_sample_count,
            d24_unorm_s8_attachment,
            shader_int16,
            sampled_image_array_dynamic_indexing,
            storage_image_array_dynamic_indexing,
            sampled_image_descriptor_limit,
            storage_image_descriptor_limit,
            shader_int64,
            fragment_stores_and_atomics,
            vertex_pipeline_stores_and_atomics,
            storage_image_extended_formats,
            storage_image_write_without_format,
            storage_image_read_without_format,
            bgra8_storage,
            texture_compression_bc,
            sampled_linear_filter,
            color_attachment,
            storage16,
            storage8,
            float16,
            int8,
            shader_output_viewport_index,
            timeline_semaphore,
            descriptor_binding_partially_bound,
            mirror_clamp_to_edge,
            image_robustness,
            attachment_feedback_loop_layout,
            image_drm_format_modifier,
            dual_src_blend,
            fill_mode_non_solid,
            depth_clamp,
            multi_viewport,
            max_viewports,
            occlusion_query_precise,
        } = self;
        let missing = |probes: &[bool; TexelLayout::ALL.len()]| {
            let names: Vec<String> = TexelLayout::ALL
                .iter()
                .filter(|l| !probes[l.index()])
                .map(|l| format!("{l:?}"))
                .collect();
            if names.is_empty() {
                "none".to_owned()
            } else {
                names.join(",")
            }
        };
        format!(
            "vk_features robust_buffer_access={robust_buffer_access} \
             image_robustness={image_robustness:?} \
             attachment_feedback_loop_layout={attachment_feedback_loop_layout} \
             image_drm_format_modifier={image_drm_format_modifier} \
             sampler_anisotropy={sampler_anisotropy} max_sampler_anisotropy={max_sampler_anisotropy} \
             max_image_dimension_2d={max_image_dimension_2d} \
             max_compute_workgroup_invocations={max_compute_workgroup_invocations} \
             subgroup_size={subgroup_size} \
             max_compute_workgroup_size={max_compute_workgroup_size:?} \
             max_compute_shared_memory_bytes={max_compute_shared_memory_bytes} \
             max_sample_count={max_sample_count} d24_unorm_s8_attachment={d24_unorm_s8_attachment} \
             shader_int16={shader_int16} shader_int64={shader_int64} \
             texture_compression_bc={texture_compression_bc} \
             sampled_image_array_dynamic_indexing={sampled_image_array_dynamic_indexing} \
             storage_image_array_dynamic_indexing={storage_image_array_dynamic_indexing} \
             sampled_image_descriptor_limit={sampled_image_descriptor_limit} \
             storage_image_descriptor_limit={storage_image_descriptor_limit} \
             fragment_stores_and_atomics={fragment_stores_and_atomics} \
             vertex_pipeline_stores_and_atomics={vertex_pipeline_stores_and_atomics} \
             storage_image_extended_formats={storage_image_extended_formats} \
             storage_image_write_without_format={storage_image_write_without_format} \
             storage_image_read_without_format={storage_image_read_without_format} \
             bgra8_storage={bgra8_storage} no_linear_filter={} no_blendable_attachment={} \
             storage16={storage16} storage8={storage8} float16={float16} int8={int8} \
             shader_output_viewport_index={shader_output_viewport_index} \
             timeline_semaphore={timeline_semaphore} \
             descriptor_binding_partially_bound={descriptor_binding_partially_bound} \
             mirror_clamp_to_edge={mirror_clamp_to_edge:?} \
             dual_src_blend={dual_src_blend} fill_mode_non_solid={fill_mode_non_solid} \
             depth_clamp={depth_clamp} multi_viewport={multi_viewport} max_viewports={max_viewports} \
             occlusion_query_precise={occlusion_query_precise}",
            missing(sampled_linear_filter),
            missing(color_attachment),
        )
    }

    /// Device extension names this feature set requires, beyond the ones the
    /// interop rails ask for.
    pub fn required_extensions(&self) -> Vec<*const std::os::raw::c_char> {
        let mut out = Vec::new();
        if self.mirror_clamp_to_edge == MirrorClampToEdge::KhrExtension {
            out.push(vk::KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_NAME.as_ptr());
        }
        if self.image_robustness == ImageRobustness::ExtImageRobustness {
            out.push(vk::EXT_IMAGE_ROBUSTNESS_NAME.as_ptr());
        }
        if self.attachment_feedback_loop_layout {
            out.push(vk::EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_NAME.as_ptr());
        }
        if self.image_drm_format_modifier {
            out.push(vk::EXT_IMAGE_DRM_FORMAT_MODIFIER_NAME.as_ptr());
        }
        out
    }
}

/// Resolve every feature this backend depends on against one physical device.
///
/// `has_extension` answers whether the device advertises a given extension; the
/// caller already enumerates them for the interop rails, so it is passed in
/// rather than enumerated twice.
///
/// # Safety
///
/// `pd` must be a physical device belonging to `instance`.
pub unsafe fn query(
    instance: &ash::Instance,
    pd: vk::PhysicalDevice,
    has_extension: &dyn Fn(&std::ffi::CStr) -> bool,
) -> DeviceFeatures {
    let mut supported_16 = vk::PhysicalDevice16BitStorageFeatures::default();
    let mut supported_8 = vk::PhysicalDevice8BitStorageFeatures::default();
    let mut supported_f16i8 = vk::PhysicalDeviceShaderFloat16Int8Features::default();
    let mut supported_vulkan12 = vk::PhysicalDeviceVulkan12Features::default();
    // Chained by its `EXT` spelling rather than the 1.3 one, because 1.2 is this
    // backend's baseline and chaining a 1.3 struct to a 1.2 driver is not
    // answerable. A 1.3 host advertises the extension too — promotion keeps the
    // extension name valid — so one query covers both rungs and only the
    // *enable* side has to know which it took.
    let mut supported_image_robustness = vk::PhysicalDeviceImageRobustnessFeaturesEXT::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut supported_16)
        .push_next(&mut supported_8)
        .push_next(&mut supported_f16i8)
        .push_next(&mut supported_vulkan12)
        .push_next(&mut supported_image_robustness);
    unsafe { instance.get_physical_device_features2(pd, &mut features2) };
    let supported = features2.features;
    let props = unsafe { instance.get_physical_device_properties(pd) };
    // Subgroup size is Vulkan 1.1 core and chains onto `Properties2`; the
    // baseline is 1.2, so it is always answerable. It is what the guest's
    // `threadExecutionWidth` query is asking for.
    let mut subgroup = vk::PhysicalDeviceSubgroupProperties::default();
    let mut props2 = vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup);
    unsafe { instance.get_physical_device_properties2(pd, &mut props2) };

    // BGRA8 as a storage image is optional; ask the device rather than assume.
    let bgra8_storage = unsafe {
        instance.get_physical_device_format_properties(
            pd,
            crate::backend::vulkan::translate::pixel::SCANOUT_FORMAT,
        )
    }
    .optimal_tiling_features
    .contains(vk::FormatFeatureFlags::STORAGE_IMAGE);

    // One probe per sampled texel layout, so the native rails decline a layout
    // this host cannot filter instead of sampling it wrong. Derived from
    // `TexelLayout::ALL`, so adding a layout adds its probe.
    let mut sampled_linear_filter = [false; TexelLayout::ALL.len()];
    // The same derivation for the render-target side: a resident is created at
    // the guest's declared format, so a layout has to be renderable before one
    // may be.
    let mut color_attachment = [false; TexelLayout::ALL.len()];
    for &layout in TexelLayout::ALL {
        let format = crate::backend::vulkan::translate::pixel::vk_texel_layout(layout);
        let features = unsafe { instance.get_physical_device_format_properties(pd, format) }
            .optimal_tiling_features;
        sampled_linear_filter[layout.index()] =
            features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR);
        // Blending is demanded only of a layout that can be blended. See the
        // field's doc: for an integer format the spec mandates no
        // `COLOR_ATTACHMENT_BLEND`, so requiring it would refuse every host.
        let mut required = vk::FormatFeatureFlags::COLOR_ATTACHMENT;
        if !layout.is_integer() {
            required |= vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND;
        }
        color_attachment[layout.index()] = features.contains(required);
    }

    // The guest is handed ONE sample-count answer and uses it for colour
    // attachments, depth attachments and sampled images alike, so the honest
    // answer is the intersection of the three masks.
    let sample_mask = props.limits.framebuffer_color_sample_counts
        & props.limits.framebuffer_depth_sample_counts
        & props.limits.sampled_image_color_sample_counts;
    let max_sample_count = [
        vk::SampleCountFlags::TYPE_16,
        vk::SampleCountFlags::TYPE_8,
        vk::SampleCountFlags::TYPE_4,
        vk::SampleCountFlags::TYPE_2,
    ]
    .into_iter()
    .find(|f| sample_mask.contains(*f))
    .map(|f| f.as_raw())
    // Single-sample is the only count the spec requires, and it is the one
    // answer that cannot make the guest build a target this host refuses.
    .unwrap_or(1);

    let d24_unorm_s8_attachment = unsafe {
        instance.get_physical_device_format_properties(pd, vk::Format::D24_UNORM_S8_UINT)
    }
    .optimal_tiling_features
    .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT);

    // Prefer the 1.2 core feature over the extension: it needs no extension
    // string and it is the spelling the baseline guarantees exists to ask about.
    // Metal defines an out-of-bounds texture read and Vulkan does not, so this is
    // taken whenever the host offers it. `Core13` needs no extension string;
    // `ExtImageRobustness` names one. Both are the same guarantee.
    let image_robustness = if supported_image_robustness.robust_image_access != vk::TRUE {
        ImageRobustness::Unsupported
    } else if props.api_version >= vk::API_VERSION_1_3 {
        ImageRobustness::Core13
    } else if has_extension(vk::EXT_IMAGE_ROBUSTNESS_NAME) {
        ImageRobustness::ExtImageRobustness
    } else {
        ImageRobustness::Unsupported
    };
    let mirror_clamp_to_edge = if supported_vulkan12.sampler_mirror_clamp_to_edge == vk::TRUE {
        MirrorClampToEdge::Core12
    } else if has_extension(vk::KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_NAME) {
        MirrorClampToEdge::KhrExtension
    } else {
        MirrorClampToEdge::Unsupported
    };
    // Unlike promoted feature structs above, this one exists only with its
    // extension. Do not put it on the unconditional features2 chain: asking a
    // 1.2 device about a structure whose extension it did not advertise is not
    // a capability query that device promised to accept.
    let attachment_feedback_loop_layout =
        if has_extension(vk::EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_NAME) {
            let mut feedback = vk::PhysicalDeviceAttachmentFeedbackLoopLayoutFeaturesEXT::default();
            let mut feedback_features =
                vk::PhysicalDeviceFeatures2::default().push_next(&mut feedback);
            unsafe { instance.get_physical_device_features2(pd, &mut feedback_features) };
            feedback.attachment_feedback_loop_layout == vk::TRUE
        } else {
            false
        };
    let image_drm_format_modifier = has_extension(vk::EXT_IMAGE_DRM_FORMAT_MODIFIER_NAME);

    DeviceFeatures {
        robust_buffer_access: supported.robust_buffer_access == vk::TRUE,
        image_robustness,
        attachment_feedback_loop_layout,
        image_drm_format_modifier,
        sampler_anisotropy: supported.sampler_anisotropy == vk::TRUE,
        dual_src_blend: supported.dual_src_blend == vk::TRUE,
        fill_mode_non_solid: supported.fill_mode_non_solid == vk::TRUE,
        texture_compression_bc: supported.texture_compression_bc == vk::TRUE,
        depth_clamp: supported.depth_clamp == vk::TRUE,
        multi_viewport: supported.multi_viewport == vk::TRUE,
        occlusion_query_precise: supported.occlusion_query_precise == vk::TRUE,
        // `max(1)` because a pipeline always declares at least one slot, and a
        // device reporting 0 here would otherwise make every draw undrawable.
        max_viewports: props.limits.max_viewports.max(1),
        max_sampler_anisotropy: props.limits.max_sampler_anisotropy.max(1.0),
        max_image_dimension_2d: props
            .limits
            .max_image_dimension2_d
            .max(VULKAN_MIN_IMAGE_DIMENSION_2D),
        max_compute_workgroup_invocations: props
            .limits
            .max_compute_work_group_invocations
            .max(VULKAN_MIN_COMPUTE_WORKGROUP_INVOCATIONS),
        // A device reporting 0 is out of spec; one lane is the only answer that
        // cannot make the guest oversize a dispatch.
        subgroup_size: subgroup.subgroup_size.max(1),
        max_compute_workgroup_size: props.limits.max_compute_work_group_size,
        max_compute_shared_memory_bytes: props.limits.max_compute_shared_memory_size,
        max_sample_count,
        d24_unorm_s8_attachment,
        shader_int16: supported.shader_int16 == vk::TRUE,
        sampled_image_array_dynamic_indexing: supported.shader_sampled_image_array_dynamic_indexing
            == vk::TRUE,
        storage_image_array_dynamic_indexing: supported.shader_storage_image_array_dynamic_indexing
            == vk::TRUE,
        sampled_image_descriptor_limit: props
            .limits
            .max_per_stage_descriptor_sampled_images
            .min(props.limits.max_descriptor_set_sampled_images),
        storage_image_descriptor_limit: props
            .limits
            .max_per_stage_descriptor_storage_images
            .min(props.limits.max_descriptor_set_storage_images),
        shader_int64: supported.shader_int64 == vk::TRUE,
        fragment_stores_and_atomics: supported.fragment_stores_and_atomics == vk::TRUE,
        vertex_pipeline_stores_and_atomics: supported.vertex_pipeline_stores_and_atomics
            == vk::TRUE,
        storage_image_extended_formats: supported.shader_storage_image_extended_formats == vk::TRUE,
        storage_image_write_without_format: supported.shader_storage_image_write_without_format
            == vk::TRUE,
        storage_image_read_without_format: supported.shader_storage_image_read_without_format
            == vk::TRUE,
        bgra8_storage,
        sampled_linear_filter,
        color_attachment,
        storage16: supported_16.storage_buffer16_bit_access == vk::TRUE,
        storage8: supported_8.storage_buffer8_bit_access == vk::TRUE,
        float16: supported_f16i8.shader_float16 == vk::TRUE,
        int8: supported_f16i8.shader_int8 == vk::TRUE,
        shader_output_viewport_index: supported_vulkan12.shader_output_viewport_index == vk::TRUE,
        timeline_semaphore: supported_vulkan12.timeline_semaphore == vk::TRUE,
        descriptor_binding_partially_bound: supported_vulkan12.descriptor_binding_partially_bound
            == vk::TRUE,
        mirror_clamp_to_edge,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_supported() -> DeviceFeatures {
        DeviceFeatures {
            occlusion_query_precise: true,
            robust_buffer_access: true,
            texture_compression_bc: true,
            sampler_anisotropy: true,
            max_sampler_anisotropy: 16.0,
            max_image_dimension_2d: 16384,
            max_compute_workgroup_invocations: 1024,
            subgroup_size: 64,
            max_compute_workgroup_size: [1024, 1024, 64],
            max_compute_shared_memory_bytes: 32768,
            max_sample_count: 8,
            d24_unorm_s8_attachment: true,
            shader_int16: true,
            sampled_image_array_dynamic_indexing: true,
            storage_image_array_dynamic_indexing: true,
            sampled_image_descriptor_limit: u32::MAX,
            storage_image_descriptor_limit: u32::MAX,
            shader_int64: true,
            fragment_stores_and_atomics: true,
            vertex_pipeline_stores_and_atomics: true,
            storage_image_extended_formats: true,
            storage_image_write_without_format: true,
            storage_image_read_without_format: true,
            bgra8_storage: true,
            sampled_linear_filter: [true; TexelLayout::ALL.len()],
            color_attachment: [true; TexelLayout::ALL.len()],
            storage16: true,
            storage8: true,
            float16: true,
            int8: true,
            shader_output_viewport_index: true,
            timeline_semaphore: true,
            descriptor_binding_partially_bound: true,
            mirror_clamp_to_edge: MirrorClampToEdge::Core12,
            image_robustness: ImageRobustness::Core13,
            attachment_feedback_loop_layout: true,
            image_drm_format_modifier: true,
            dual_src_blend: true,
            fill_mode_non_solid: true,
            depth_clamp: true,
            multi_viewport: true,
            max_viewports: 16,
        }
    }

    /// `dualSrcBlend` is queried and enabled together, which is rule 1 of this
    /// module: a feature asked about here and not enabled here is the same
    /// ungated-bind bug in a new place.
    ///
    /// It is a plain optional core feature, so unlike the mirror-clamp mode it
    /// has no extension rung — the only two states are advertised and not, and
    /// the not-advertised state must leave the enable bit clear so device
    /// creation does not fail asking for it.
    #[test]
    fn dual_source_blend_is_enabled_only_where_the_device_advertises_it() {
        assert_eq!(all_supported().enabled_features().dual_src_blend, vk::TRUE);
        let without = DeviceFeatures {
            dual_src_blend: false,
            attachment_feedback_loop_layout: false,
            image_drm_format_modifier: false,
            ..all_supported()
        };
        assert_eq!(without.enabled_features().dual_src_blend, vk::FALSE);
        assert!(without.required_extensions().is_empty());
        // The default is "not supported", so a `DeviceFeatures` built without a
        // query never claims a capability it has not checked for.
        assert!(!DeviceFeatures::default().dual_src_blend);
    }

    /// The two rasterization features the guest's `setTriangleFillMode:` and
    /// `setDepthClipMode:` records need, under the same rule as
    /// `dualSrcBlend`: queried here, enabled here, and left clear where the
    /// device says no so `vkCreateDevice` does not fail asking for them.
    ///
    /// Both are optional core with no extension rung, and both are consumed by
    /// `engine::caches`, which declines the pipeline by name rather than
    /// rasterizing the other way.
    #[test]
    fn the_raster_features_are_enabled_only_where_the_device_advertises_them() {
        let all = all_supported().enabled_features();
        assert_eq!(all.fill_mode_non_solid, vk::TRUE);
        assert_eq!(all.depth_clamp, vk::TRUE);
        let without = DeviceFeatures {
            fill_mode_non_solid: false,
            depth_clamp: false,
            ..all_supported()
        }
        .enabled_features();
        assert_eq!(without.fill_mode_non_solid, vk::FALSE);
        assert_eq!(without.depth_clamp, vk::FALSE);
        // Never claimed without a query, the same way `dual_src_blend` is not.
        assert!(!DeviceFeatures::default().fill_mode_non_solid);
        assert!(!DeviceFeatures::default().depth_clamp);
    }

    /// The 1.2 rung sets the core feature bit and asks for no extension.
    #[test]
    fn the_core_rung_needs_no_extension_string() {
        let caps = DeviceFeatures {
            attachment_feedback_loop_layout: false,
            image_drm_format_modifier: false,
            ..all_supported()
        };
        assert_eq!(
            caps.enabled_vulkan12().sampler_mirror_clamp_to_edge,
            vk::TRUE
        );
        assert!(caps.required_extensions().is_empty());
    }

    /// The extension rung is the mirror image: extension string, no core bit.
    /// Setting both would request a feature the device may not expose in 1.2
    /// form, which is how a "belt and braces" enable becomes a device-creation
    /// failure on the driver that only has the extension.
    #[test]
    fn the_extension_rung_asks_for_the_extension_and_not_the_core_bit() {
        let caps = DeviceFeatures {
            mirror_clamp_to_edge: MirrorClampToEdge::KhrExtension,
            attachment_feedback_loop_layout: false,
            image_drm_format_modifier: false,
            ..all_supported()
        };
        assert_eq!(
            caps.enabled_vulkan12().sampler_mirror_clamp_to_edge,
            vk::FALSE
        );
        assert_eq!(caps.required_extensions().len(), 1);
    }

    /// Neither rung: nothing is requested. The sampler path must decline the
    /// address mode by name rather than bind it — the whole point of the enum
    /// having a third state instead of being a bool.
    #[test]
    fn without_support_nothing_is_requested() {
        let caps = DeviceFeatures {
            mirror_clamp_to_edge: MirrorClampToEdge::Unsupported,
            attachment_feedback_loop_layout: false,
            image_drm_format_modifier: false,
            ..all_supported()
        };
        assert_eq!(
            caps.enabled_vulkan12().sampler_mirror_clamp_to_edge,
            vk::FALSE
        );
        assert!(caps.required_extensions().is_empty());
        assert!(!caps.mirror_clamp_to_edge.is_available());
    }

    /// The rung decides the extension string; **both available rungs enable the
    /// feature.**
    ///
    /// Getting this backwards is the shape of the bug this module exists to
    /// retire: naming an extension without enabling its feature, or enabling a
    /// feature the device declined. The first silently does nothing, the second
    /// fails `vkCreateDevice`.
    #[test]
    fn image_robustness_is_enabled_on_both_rungs_and_named_on_one() {
        let ext_name = vk::EXT_IMAGE_ROBUSTNESS_NAME.as_ptr();

        let core13 = DeviceFeatures {
            image_robustness: ImageRobustness::Core13,
            ..Default::default()
        };
        assert_eq!(
            core13.enabled_image_robustness().robust_image_access,
            vk::TRUE
        );
        assert!(
            !core13.required_extensions().contains(&ext_name),
            "promoted to core at 1.3, so the string would be redundant"
        );

        let ext = DeviceFeatures {
            image_robustness: ImageRobustness::ExtImageRobustness,
            ..Default::default()
        };
        assert_eq!(ext.enabled_image_robustness().robust_image_access, vk::TRUE);
        assert!(
            ext.required_extensions().contains(&ext_name),
            "the 1.2-baseline rung has to name the extension it is chaining"
        );

        let none = DeviceFeatures::default();
        assert_eq!(
            none.enabled_image_robustness().robust_image_access,
            vk::FALSE,
            "asking for a feature the host declined fails vkCreateDevice"
        );
        assert!(!none.required_extensions().contains(&ext_name));
        assert!(!none.image_robustness.is_available());
    }

    /// This extension has one indivisible contract: the queried feature, the
    /// feature enabled at device creation, and its extension string must move
    /// together. Naming only the extension leaves the layout unusable; asking
    /// for the feature without the advertised extension fails device creation.
    #[test]
    fn attachment_feedback_loop_feature_and_extension_move_together() {
        let name = vk::EXT_ATTACHMENT_FEEDBACK_LOOP_LAYOUT_NAME.as_ptr();
        let on = DeviceFeatures {
            attachment_feedback_loop_layout: true,
            ..Default::default()
        };
        assert_eq!(
            on.enabled_attachment_feedback_loop_layout()
                .attachment_feedback_loop_layout,
            vk::TRUE
        );
        assert!(on.required_extensions().contains(&name));

        let off = DeviceFeatures::default();
        assert_eq!(
            off.enabled_attachment_feedback_loop_layout()
                .attachment_feedback_loop_layout,
            vk::FALSE
        );
        assert!(!off.required_extensions().contains(&name));
    }

    #[test]
    fn explicit_drm_layout_names_its_extension_only_when_available() {
        let name = vk::EXT_IMAGE_DRM_FORMAT_MODIFIER_NAME.as_ptr();
        let on = DeviceFeatures {
            image_drm_format_modifier: true,
            ..Default::default()
        };
        assert!(on.required_extensions().contains(&name));
        assert!(!DeviceFeatures::default()
            .required_extensions()
            .contains(&name));
    }

    /// A feature the device declines is never enabled — the enable list is a
    /// derivation, not a wish.
    #[test]
    fn unsupported_features_are_not_enabled() {
        let caps = DeviceFeatures::default();
        let enabled = caps.enabled_features();
        assert_eq!(enabled.robust_buffer_access, vk::FALSE);
        assert_eq!(enabled.sampler_anisotropy, vk::FALSE);
        assert_eq!(enabled.shader_int16, vk::FALSE);
        assert_eq!(enabled.shader_storage_image_extended_formats, vk::FALSE);
    }

    /// The BGRA composite path needs BOTH halves. Naming the pair once is what
    /// stops a call site checking only the feature and binding a format the
    /// device does not support as a storage image.
    #[test]
    fn the_bgra_storage_path_needs_both_halves() {
        let both = all_supported();
        assert!(both.storage_image_write_without_format_bgra());
        let no_format = DeviceFeatures {
            bgra8_storage: false,
            ..all_supported()
        };
        assert!(!no_format.storage_image_write_without_format_bgra());
        let no_feature = DeviceFeatures {
            storage_image_write_without_format: false,
            ..all_supported()
        };
        assert!(!no_feature.storage_image_write_without_format_bgra());
    }

    /// The render-target bound is the device's limit, not the spec's floor.
    ///
    /// The draw path used to refuse any target wider or taller than 4096 with
    /// `GeometryUnsupported`. 4096 is the Vulkan 1.2 *required minimum* for
    /// `maxImageDimension2D` — the smallest value a conformant implementation
    /// may report — and desktop GPUs report 16384. Using the floor as the cap
    /// therefore refused targets the host could hold: a guest on a 5K or 6K
    /// display names them, and the refusal costs the frame.
    ///
    /// So a host reporting more than the floor must be believed, and a host
    /// reporting less than it is out of spec and clamped up rather than
    /// trusted downward.
    #[test]
    fn the_render_target_bound_follows_the_device_not_the_spec_floor() {
        assert!(
            all_supported().max_image_dimension_2d > VULKAN_MIN_IMAGE_DIMENSION_2D,
            "a desktop-class device reports well past the floor, and the draw \
             path must accept a target that large"
        );
        // A 5K-wide target: refused under the old fixed 4096, accepted here.
        assert!(5120 <= all_supported().max_image_dimension_2d);
    }

    /// The compute limits the guest is told are the device's, not a fixed pair.
    ///
    /// `CmdGetComputeInfo` used to answer `maxTotalThreadsPerThreadgroup` 1024
    /// and `threadExecutionWidth` 32 on every host. The guest sizes its
    /// dispatches from the first, and the Vulkan 1.2 floor for
    /// `maxComputeWorkGroupInvocations` is 128 — so on any device at or near
    /// the floor that answer promised threadgroups eight times larger than the
    /// device can run. The second is vendor-dependent (64 on AMD wave64), and
    /// a guest told 32 there sizes every dispatch against the wrong wave.
    ///
    /// A host below the spec floor is out of spec and clamped up rather than
    /// believed downward; a host above it must be believed.
    #[test]
    fn the_compute_limits_reported_to_the_guest_come_from_the_device() {
        let f = all_supported();
        assert!(f.max_compute_workgroup_invocations >= VULKAN_MIN_COMPUTE_WORKGROUP_INVOCATIONS);
        assert_ne!(
            f.subgroup_size, 32,
            "the fixture is a wave64 part precisely so a hardcoded 32 cannot pass"
        );
        assert!(
            f.subgroup_size > 0,
            "a zero wave would divide by zero downstream"
        );
    }

    /// The enable list is derived from what the backend binds, and
    /// `multi_viewport` is the case that proves it.
    ///
    /// It has now been wrong in both directions. It was enabled wherever
    /// supported while nothing could bind a second viewport, then disabled to
    /// match that; now a draw's `viewportCount` is the guest's own, so it must
    /// be enabled again or every multi-viewport pipeline is invalid. What makes
    /// the rule checkable rather than a wish is that both halves are asserted
    /// here against one another: a request that carries two viewports, and the
    /// feature that makes two legal.
    #[test]
    fn multi_viewport_is_enabled_because_a_draw_can_bind_a_second() {
        let enabled = all_supported().enabled_features();
        assert_eq!(
            enabled.multi_viewport,
            vk::TRUE,
            "a draw can name several viewports, so the feature must be requested"
        );
        let vp = |x: f32| crate::backend::vulkan::engine::ViewportResource {
            x,
            y: 0.0,
            width: 1.0,
            height: 1.0,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let req = crate::backend::vulkan::engine::DrawRequest {
            viewports: vec![vp(0.0), vp(1.0)],
            ..Default::default()
        };
        assert_eq!(
            crate::backend::vulkan::engine::viewport_slot_count(&req),
            2,
            "the second viewport must reach the pipeline's slot count"
        );
    }

    /// The boot line reports a feature that came back **false** as false rather
    /// than omitting it.
    ///
    /// This is the whole reason the line exists beside `vk_device_select`: a rail
    /// that declines by name and a rail that was never asked for read the same in
    /// a log that only prints what was enabled, and they are different bug
    /// reports. Both directions are asserted, because a line that printed only
    /// the false ones would have the same defect mirrored.
    #[test]
    fn the_feature_line_reports_both_directions() {
        let on = all_supported().report_line();
        assert!(on.starts_with("vk_features "), "{on}");
        assert!(on.contains("depth_clamp=true"), "{on}");
        assert!(on.contains("timeline_semaphore=true"), "{on}");
        assert!(on.contains("attachment_feedback_loop_layout=true"), "{on}");
        assert!(on.contains("subgroup_size=64"), "{on}");
        // The layout probes report the *missing* set, which is empty here.
        assert!(on.contains("no_linear_filter=none"), "{on}");
        assert!(on.contains("no_blendable_attachment=none"), "{on}");

        let off = DeviceFeatures::default().report_line();
        assert!(off.contains("depth_clamp=false"), "{off}");
        assert!(off.contains("timeline_semaphore=false"), "{off}");
        assert!(
            off.contains("attachment_feedback_loop_layout=false"),
            "{off}"
        );
        assert!(
            off.contains("mirror_clamp_to_edge=Unsupported"),
            "the rung, not a bool: which spelling a device has decides what is \
             requested at create time — {off}"
        );
    }

    /// A layout the host cannot filter is named on the line, so "the sampled rail
    /// declined" is answerable from one boot's log rather than from a second run
    /// with a probe.
    #[test]
    fn the_feature_line_names_the_layouts_a_host_cannot_serve() {
        let mut f = all_supported();
        f.sampled_linear_filter[TexelLayout::R32Float.index()] = false;
        let line = f.report_line();
        assert!(line.contains("no_linear_filter=R32Float"), "{line}");
        // The other array is independent and must not be dragged along.
        assert!(line.contains("no_blendable_attachment=none"), "{line}");
    }

    /// A device that advertises no `multiViewport` reports a limit of one, and
    /// that is the number the draw path compares against.
    ///
    /// The limit and the feature are separate fields because Vulkan reports
    /// them separately, and `maxViewports` is *not* required to be 1 on a
    /// device without the feature — the spec's floor is 1, but an
    /// implementation may report 16 while still refusing to use them. Reading
    /// the limit alone would then have built an invalid pipeline.
    #[test]
    fn a_device_without_multi_viewport_offers_exactly_one_slot() {
        let mut f = all_supported();
        f.multi_viewport = false;
        f.max_viewports = 16;
        let allowed = if f.multi_viewport { f.max_viewports } else { 1 };
        assert_eq!(
            allowed, 1,
            "the feature gates the limit; a generous limit does not license a second slot"
        );
    }

    /// The three shader-shape features a **translated** module can demand, all
    /// of which this device bound for its whole life without asking for them.
    ///
    /// Every other feature in this module is one the backend spells at a call
    /// site, so the rule "enable only what the backend binds" can be checked by
    /// finding that site. These three have no such site: they are properties of
    /// SPIR-V that `metal2vulkan` produced from the guest's own AIR, and the
    /// only place they appear is inside a `pCode` blob. That is why they were
    /// missed, and why the check here is against the enable list rather than
    /// against a caller.
    ///
    /// What named them was a driven macos-11 boot under the Khronos validation
    /// layer: `VUID-VkShaderModuleCreateInfo-pCode-08740` for a module
    /// declaring `Int64` with `shaderInt64` disabled, and
    /// `VUID-RuntimeSpirv-NonWritable-06340` / `-06341` for storage buffers in
    /// fragment and vertex stages that carried no `NonWritable` decoration
    /// while the matching feature was off. All three are undefined behaviour
    /// rather than a decode error, which is the class that runs for months on
    /// one driver and resets the GPU on another.
    #[test]
    fn the_features_a_translated_module_can_demand_are_requested() {
        let enabled = all_supported().enabled_features();
        assert_eq!(
            enabled.shader_int64,
            vk::TRUE,
            "a translated module declaring Int64 needs the feature enabled"
        );
        assert_eq!(
            enabled.fragment_stores_and_atomics,
            vk::TRUE,
            "a fragment stage's storage buffer that is not NonWritable needs it"
        );
        assert_eq!(
            enabled.vertex_pipeline_stores_and_atomics,
            vk::TRUE,
            "the vertex stage's half of the same rule"
        );
        assert_eq!(
            enabled.shader_sampled_image_array_dynamic_indexing,
            vk::TRUE
        );
        assert_eq!(
            enabled.shader_storage_image_array_dynamic_indexing,
            vk::TRUE
        );
        // And a host that declines them is never asked, or `vkCreateDevice`
        // fails for every guest instead of the pipelines that need them.
        let none = DeviceFeatures::default().enabled_features();
        assert_eq!(none.shader_int64, vk::FALSE);
        assert_eq!(none.fragment_stores_and_atomics, vk::FALSE);
        assert_eq!(none.vertex_pipeline_stores_and_atomics, vk::FALSE);
        assert_eq!(none.shader_sampled_image_array_dynamic_indexing, vk::FALSE);
        assert_eq!(none.shader_storage_image_array_dynamic_indexing, vk::FALSE);
    }

    #[test]
    fn descriptor_arrays_require_indexing_and_partially_bound_support() {
        let all = all_supported();
        assert!(all.sampled_descriptor_arrays(128));
        assert!(all.storage_descriptor_arrays(128));

        assert!(!DeviceFeatures {
            descriptor_binding_partially_bound: false,
            ..all
        }
        .sampled_descriptor_arrays(128));
        assert!(!DeviceFeatures {
            sampled_image_array_dynamic_indexing: false,
            ..all
        }
        .sampled_descriptor_arrays(128));
        assert!(!DeviceFeatures {
            storage_image_array_dynamic_indexing: false,
            ..all
        }
        .storage_descriptor_arrays(128));
        assert!(!DeviceFeatures {
            sampled_image_descriptor_limit: 127,
            ..all
        }
        .sampled_descriptor_arrays(128));
        assert!(!DeviceFeatures {
            storage_image_descriptor_limit: 127,
            ..all
        }
        .storage_descriptor_arrays(128));
    }

    /// `VUID-VkDeviceCreateInfo-pNext-02830`: a promoted feature struct may not
    /// be chained beside the `VkPhysicalDeviceVulkan12Features` that absorbed
    /// it, so the promoted spelling has to carry the value.
    ///
    /// This device used to chain `VkPhysicalDevice8BitStorageFeatures` and
    /// `VkPhysicalDeviceShaderFloat16Int8Features` next to a
    /// `VkPhysicalDeviceVulkan12Features` whose matching fields were left
    /// `VK_FALSE` — two structs disagreeing about one feature, which is exactly
    /// what the VU exists to forbid. Asserting the 1.2 struct carries them is
    /// what stops the separate structs coming back: there is nothing left for
    /// them to say.
    ///
    /// 16-bit storage is deliberately absent. It was promoted into
    /// `VkPhysicalDeviceVulkan11Features`, which this chain does not carry, so
    /// its own struct conflicts with nothing.
    #[test]
    fn the_features_promoted_into_vulkan12_are_set_on_the_vulkan12_struct() {
        let v12 = all_supported().enabled_vulkan12();
        assert_eq!(v12.storage_buffer8_bit_access, vk::TRUE);
        assert_eq!(v12.shader_float16, vk::TRUE);
        assert_eq!(v12.shader_int8, vk::TRUE);
        assert_eq!(v12.descriptor_binding_partially_bound, vk::TRUE);
        // A host that declines one must leave it clear on this struct too,
        // because this struct is now the only place it is said.
        let without = DeviceFeatures {
            float16: false,
            descriptor_binding_partially_bound: false,
            ..all_supported()
        }
        .enabled_vulkan12();
        assert_eq!(without.shader_float16, vk::FALSE);
        assert_eq!(without.descriptor_binding_partially_bound, vk::FALSE);
        assert_eq!(without.shader_int8, vk::TRUE);
    }
}
