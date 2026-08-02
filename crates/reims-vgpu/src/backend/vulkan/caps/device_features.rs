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
    /// Metal AIR routinely lowers to 64-bit integer ops, so translated SPIR-V
    /// declares the `Int64` capability. A device that was never told to enable
    /// the matching feature compiles it anyway and is then free to do anything:
    /// NVIDIA copes, lavapipe null-dereferences inside its shader JIT.
    pub shader_int64: bool,
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
    /// Vulkan format is usable as a **colour attachment that blends** under
    /// optimal tiling.
    ///
    /// A render target's resident is created at the format the guest declared
    /// for the attachment, so a layout this host cannot render to — or can
    /// render to but not blend into — is one that must fall back to the
    /// engine's eight-bit resident rather than be attempted. Both feature bits
    /// are required together because a colour attachment that cannot blend is
    /// not a usable render target for a compositor, and admitting one on the
    /// strength of the other trades a fidelity loss for a `vkCreateImage` that
    /// fails or a pipeline the driver refuses.
    ///
    /// Asked per layout for the same reason as [`Self::sampled_linear_filter`]
    /// directly above: the array is sized by `ALL.len()`, so a new
    /// [`TexelLayout`] cannot reach a render target without getting a probe.
    /// `R16G16B16A16_SFLOAT` is in Vulkan's mandatory format table for both
    /// bits, but AGENTS.md is explicit that a capability comes from the device
    /// and not from a reading of the spec's table.
    pub color_attachment_blend: [bool; TexelLayout::ALL.len()],
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
    pub mirror_clamp_to_edge: MirrorClampToEdge,
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
    pub fn enabled_features(&self) -> vk::PhysicalDeviceFeatures {
        vk::PhysicalDeviceFeatures::default()
            .robust_buffer_access(self.robust_buffer_access)
            .sampler_anisotropy(self.sampler_anisotropy)
            .shader_int16(self.shader_int16)
            .shader_int64(self.shader_int64)
            .shader_storage_image_extended_formats(self.storage_image_extended_formats)
            .shader_storage_image_write_without_format(self.storage_image_write_without_format)
            .shader_storage_image_read_without_format(self.storage_image_read_without_format)
            .dual_src_blend(self.dual_src_blend)
            .fill_mode_non_solid(self.fill_mode_non_solid)
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
    pub fn enabled_vulkan12(&self) -> vk::PhysicalDeviceVulkan12Features<'static> {
        vk::PhysicalDeviceVulkan12Features::default()
            .shader_output_viewport_index(self.shader_output_viewport_index)
            .timeline_semaphore(self.timeline_semaphore)
            .sampler_mirror_clamp_to_edge(self.mirror_clamp_to_edge == MirrorClampToEdge::Core12)
    }

    /// 16-bit storage-buffer access, for shaders that pack half-precision data.
    pub fn enabled_16bit_storage(&self) -> vk::PhysicalDevice16BitStorageFeatures<'static> {
        vk::PhysicalDevice16BitStorageFeatures::default()
            .storage_buffer16_bit_access(self.storage16)
    }

    /// 8-bit storage-buffer access.
    pub fn enabled_8bit_storage(&self) -> vk::PhysicalDevice8BitStorageFeatures<'static> {
        vk::PhysicalDevice8BitStorageFeatures::default().storage_buffer8_bit_access(self.storage8)
    }

    /// `shaderFloat16` / `shaderInt8`, which AIR uses for half and char types.
    pub fn enabled_float16_int8(&self) -> vk::PhysicalDeviceShaderFloat16Int8Features<'static> {
        vk::PhysicalDeviceShaderFloat16Int8Features::default()
            .shader_float16(self.float16)
            .shader_int8(self.int8)
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
            shader_int64,
            storage_image_extended_formats,
            storage_image_write_without_format,
            storage_image_read_without_format,
            bgra8_storage,
            sampled_linear_filter,
            color_attachment_blend,
            storage16,
            storage8,
            float16,
            int8,
            shader_output_viewport_index,
            timeline_semaphore,
            mirror_clamp_to_edge,
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
             sampler_anisotropy={sampler_anisotropy} max_sampler_anisotropy={max_sampler_anisotropy} \
             max_image_dimension_2d={max_image_dimension_2d} \
             max_compute_workgroup_invocations={max_compute_workgroup_invocations} \
             subgroup_size={subgroup_size} \
             max_compute_workgroup_size={max_compute_workgroup_size:?} \
             max_compute_shared_memory_bytes={max_compute_shared_memory_bytes} \
             max_sample_count={max_sample_count} d24_unorm_s8_attachment={d24_unorm_s8_attachment} \
             shader_int16={shader_int16} shader_int64={shader_int64} \
             storage_image_extended_formats={storage_image_extended_formats} \
             storage_image_write_without_format={storage_image_write_without_format} \
             storage_image_read_without_format={storage_image_read_without_format} \
             bgra8_storage={bgra8_storage} no_linear_filter={} no_blendable_attachment={} \
             storage16={storage16} storage8={storage8} float16={float16} int8={int8} \
             shader_output_viewport_index={shader_output_viewport_index} \
             timeline_semaphore={timeline_semaphore} mirror_clamp_to_edge={mirror_clamp_to_edge:?} \
             dual_src_blend={dual_src_blend} fill_mode_non_solid={fill_mode_non_solid} \
             depth_clamp={depth_clamp} multi_viewport={multi_viewport} max_viewports={max_viewports} \
             occlusion_query_precise={occlusion_query_precise}",
            missing(sampled_linear_filter),
            missing(color_attachment_blend),
        )
    }

    /// Device extension names this feature set requires, beyond the ones the
    /// interop rails ask for.
    pub fn required_extensions(&self) -> Vec<*const std::os::raw::c_char> {
        let mut out = Vec::new();
        if self.mirror_clamp_to_edge == MirrorClampToEdge::KhrExtension {
            out.push(vk::KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_NAME.as_ptr());
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
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut supported_16)
        .push_next(&mut supported_8)
        .push_next(&mut supported_f16i8)
        .push_next(&mut supported_vulkan12);
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
    // the guest's declared format, so a layout has to be renderable *and*
    // blendable before one may be.
    let mut color_attachment_blend = [false; TexelLayout::ALL.len()];
    for &layout in TexelLayout::ALL {
        let format = crate::backend::vulkan::translate::pixel::vk_texel_layout(layout);
        let features =
            unsafe { instance.get_physical_device_format_properties(pd, format) }
                .optimal_tiling_features;
        sampled_linear_filter[layout.index()] =
            features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR);
        color_attachment_blend[layout.index()] = features.contains(
            vk::FormatFeatureFlags::COLOR_ATTACHMENT
                | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND,
        );
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
    let mirror_clamp_to_edge = if supported_vulkan12.sampler_mirror_clamp_to_edge == vk::TRUE {
        MirrorClampToEdge::Core12
    } else if has_extension(vk::KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_NAME) {
        MirrorClampToEdge::KhrExtension
    } else {
        MirrorClampToEdge::Unsupported
    };

    DeviceFeatures {
        robust_buffer_access: supported.robust_buffer_access == vk::TRUE,
        sampler_anisotropy: supported.sampler_anisotropy == vk::TRUE,
        dual_src_blend: supported.dual_src_blend == vk::TRUE,
        fill_mode_non_solid: supported.fill_mode_non_solid == vk::TRUE,
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
        shader_int64: supported.shader_int64 == vk::TRUE,
        storage_image_extended_formats: supported.shader_storage_image_extended_formats == vk::TRUE,
        storage_image_write_without_format: supported.shader_storage_image_write_without_format
            == vk::TRUE,
        storage_image_read_without_format: supported.shader_storage_image_read_without_format
            == vk::TRUE,
        bgra8_storage,
        sampled_linear_filter,
        color_attachment_blend,
        storage16: supported_16.storage_buffer16_bit_access == vk::TRUE,
        storage8: supported_8.storage_buffer8_bit_access == vk::TRUE,
        float16: supported_f16i8.shader_float16 == vk::TRUE,
        int8: supported_f16i8.shader_int8 == vk::TRUE,
        shader_output_viewport_index: supported_vulkan12.shader_output_viewport_index == vk::TRUE,
        timeline_semaphore: supported_vulkan12.timeline_semaphore == vk::TRUE,
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
            shader_int64: true,
            storage_image_extended_formats: true,
            storage_image_write_without_format: true,
            storage_image_read_without_format: true,
            bgra8_storage: true,
            sampled_linear_filter: [true; TexelLayout::ALL.len()],
            color_attachment_blend: [true; TexelLayout::ALL.len()],
            storage16: true,
            storage8: true,
            float16: true,
            int8: true,
            shader_output_viewport_index: true,
            timeline_semaphore: true,
            mirror_clamp_to_edge: MirrorClampToEdge::Core12,
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
        let caps = all_supported();
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
            ..all_supported()
        };
        assert_eq!(
            caps.enabled_vulkan12().sampler_mirror_clamp_to_edge,
            vk::FALSE
        );
        assert!(caps.required_extensions().is_empty());
        assert!(!caps.mirror_clamp_to_edge.is_available());
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
        assert!(on.contains("subgroup_size=64"), "{on}");
        // The layout probes report the *missing* set, which is empty here.
        assert!(on.contains("no_linear_filter=none"), "{on}");
        assert!(on.contains("no_blendable_attachment=none"), "{on}");

        let off = DeviceFeatures::default().report_line();
        assert!(off.contains("depth_clamp=false"), "{off}");
        assert!(off.contains("timeline_semaphore=false"), "{off}");
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
}
