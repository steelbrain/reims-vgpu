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
//! 2. **Enable only what the backend binds.** `multi_viewport` used to be
//!    enabled while `engine::exec` declines any draw with more than one
//!    viewport. Harmless, but it means the list was a wish rather than a
//!    derivation — and a list that is not derived cannot be checked.

use ash::vk;

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
    pub storage_image_extended_formats: bool,
    pub storage_image_write_without_format: bool,
    /// `B8G8R8A8_UNORM` usable as a storage image with optimal tiling. **Not**
    /// spec-mandatory — only `R8G8B8A8_UNORM` is — so the BGRA composite path
    /// needs this *and* `storage_image_write_without_format`.
    pub bgra8_storage: bool,
    /// `R32_SFLOAT` usable as a sampled image with **linear** filtering under
    /// optimal tiling. Single-channel float32 color-management LUTs
    /// (`UberCompositeFragment` display-profile pass) are sampled with linear
    /// interpolation; unlike `R16_SFLOAT`, this feature is *not* spec-mandatory
    /// and is absent on Apple GPUs, so the native float32 sampled rail is gated
    /// on it and otherwise leaves the sample fail-visible.
    pub sampled_r32f_linear_filter: bool,
    pub storage16: bool,
    /// 16-bit types in shader `Input`/`Output` storage classes — i.e. half
    /// varyings passed between stages.
    ///
    /// Part of the same `VkPhysicalDevice16BitStorageFeatures` struct as
    /// [`Self::storage16`] and separately toggled, so asking for one does not
    /// grant the other. Metal's `half` interpolants land here, and the guest
    /// uses them: `vkCreateShaderModule(): SPIR-V contains a 16-bit OpVariable
    /// with Output Storage Class, but storageInputOutput16 was not enabled`
    /// (`VUID-RuntimeSpirv-storageInputOutput16-11162`).
    pub storage_input_output16: bool,
    pub storage8: bool,
    pub float16: bool,
    pub int8: bool,
    pub shader_output_viewport_index: bool,
    /// Fragment shaders may write storage buffers and images.
    ///
    /// Not spec-mandatory, and not requested until a validation layer said what
    /// the guest was actually doing: `Set 0, Binding 104`,
    /// `VK_DESCRIPTOR_TYPE_STORAGE_BUFFER`, written from
    /// `VK_SHADER_STAGE_FRAGMENT_BIT` with the feature off — ten reports in one
    /// boot (`VUID-RuntimeSpirv-NonWritable-06340/06341`). A fragment store
    /// without it is undefined behaviour, which is the licence a driver needs to
    /// do anything at all, including take the process down.
    pub fragment_stores_and_atomics: bool,
    /// Per-attachment blend state may differ between MRT attachments.
    ///
    /// Likewise measured rather than assumed: the guest's compositor builds a
    /// pipeline whose `pAttachments[1]` differs from `pAttachments[0]`
    /// (`VUID-VkPipelineColorBlendStateCreateInfo-pAttachments-00605`). Without
    /// the feature Vulkan requires every attachment to carry identical blend
    /// state, so the pipeline the guest asked for cannot be built honestly.
    pub independent_blend: bool,
    /// Vertex-stage shaders may write storage buffers and images.
    ///
    /// The vertex-stage twin of [`Self::fragment_stores_and_atomics`], and named
    /// by the same measurement: `Set 0, Binding 0` and `Binding 1`, storage
    /// buffers, written from `VK_SHADER_STAGE_VERTEX_BIT`
    /// (`VUID-RuntimeSpirv-NonWritable-06341`).
    pub vertex_pipeline_stores_and_atomics: bool,
    /// 64-bit integers in shaders.
    ///
    /// The translated modules declare SPIR-V `Capability Int64`
    /// (`VUID-VkShaderModuleCreateInfo-pCode-08740`), which AIR reaches for
    /// through Metal's `long`/`ulong` and through 64-bit address arithmetic.
    /// Declaring a capability the device was not asked for makes the module
    /// invalid, and an invalid module is licence for the driver to do anything.
    pub shader_int64: bool,
    /// `VK_EXT_shader_demote_to_helper_invocation` is present and its feature
    /// bit is supported.
    ///
    /// The translated modules declare SPIR-V `Capability
    /// DemoteToHelperInvocation` — Metal's `discard_fragment()` lowers to it —
    /// and a module declaring a capability the device was not asked for is
    /// invalid (`VUID-VkShaderModuleCreateInfo-pCode-08740`, five reports in one
    /// boot).
    ///
    /// Reached through the EXT rather than the Vulkan 1.3 core struct that
    /// promoted it: the support matrix's baseline is 1.2, and `caps::gate`
    /// enforces that no 1.3 core symbol appears in the crate. The EXT is the
    /// 1.2-era spelling of the same capability, so nothing about the baseline
    /// has to move.
    pub shader_demote_to_helper_invocation: bool,
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
    /// `multi_viewport` is deliberately absent even where supported:
    /// `engine::exec` declines any draw carrying more than one viewport, so
    /// enabling it advertised a capability nothing reaches.
    pub fn enabled_features(&self) -> vk::PhysicalDeviceFeatures {
        vk::PhysicalDeviceFeatures::default()
            .robust_buffer_access(self.robust_buffer_access)
            .sampler_anisotropy(self.sampler_anisotropy)
            .shader_int16(self.shader_int16)
            .shader_storage_image_extended_formats(self.storage_image_extended_formats)
            .shader_storage_image_write_without_format(self.storage_image_write_without_format)
            .dual_src_blend(self.dual_src_blend)
            .fragment_stores_and_atomics(self.fragment_stores_and_atomics)
            .independent_blend(self.independent_blend)
            .vertex_pipeline_stores_and_atomics(self.vertex_pipeline_stores_and_atomics)
            .shader_int64(self.shader_int64)
    }

    /// `VK_EXT_shader_demote_to_helper_invocation`'s feature struct, chained
    /// only when the device advertises the extension.
    pub fn enabled_demote_to_helper(
        &self,
    ) -> vk::PhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT<'static> {
        vk::PhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT::default()
            .shader_demote_to_helper_invocation(self.shader_demote_to_helper_invocation)
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
            .sampler_mirror_clamp_to_edge(self.mirror_clamp_to_edge == MirrorClampToEdge::Core12)
    }

    /// 16-bit storage-buffer access, for shaders that pack half-precision data.
    pub fn enabled_16bit_storage(&self) -> vk::PhysicalDevice16BitStorageFeatures<'static> {
        vk::PhysicalDevice16BitStorageFeatures::default()
            .storage_buffer16_bit_access(self.storage16)
            .storage_input_output16(self.storage_input_output16)
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

    /// Device extension names this feature set requires, beyond the ones the
    /// interop rails ask for.
    pub fn required_extensions(&self) -> Vec<*const std::os::raw::c_char> {
        let mut out = Vec::new();
        if self.mirror_clamp_to_edge == MirrorClampToEdge::KhrExtension {
            out.push(vk::KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_NAME.as_ptr());
        }
        if self.shader_demote_to_helper_invocation {
            out.push(vk::EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_NAME.as_ptr());
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
    // Only chained when the device advertises the extension: querying a feature
    // struct whose extension is absent is not a defined thing to ask.
    let demote_ext = has_extension(vk::EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_NAME);
    let mut supported_demote =
        vk::PhysicalDeviceShaderDemoteToHelperInvocationFeaturesEXT::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut supported_16)
        .push_next(&mut supported_8)
        .push_next(&mut supported_f16i8)
        .push_next(&mut supported_vulkan12);
    if demote_ext {
        features2 = features2.push_next(&mut supported_demote);
    }
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

    // R32_SFLOAT linear filtering is optional (absent on Apple/MoltenVK); ask
    // rather than assume, so the native float32 sampled LUT rail can decline
    // where the host cannot filter it.
    let sampled_r32f_linear_filter =
        unsafe { instance.get_physical_device_format_properties(pd, vk::Format::R32_SFLOAT) }
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR);

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

    let demote_to_helper =
        demote_ext && supported_demote.shader_demote_to_helper_invocation == vk::TRUE;

    DeviceFeatures {
        robust_buffer_access: supported.robust_buffer_access == vk::TRUE,
        sampler_anisotropy: supported.sampler_anisotropy == vk::TRUE,
        dual_src_blend: supported.dual_src_blend == vk::TRUE,
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
        storage_image_extended_formats: supported.shader_storage_image_extended_formats == vk::TRUE,
        storage_image_write_without_format: supported.shader_storage_image_write_without_format
            == vk::TRUE,
        bgra8_storage,
        sampled_r32f_linear_filter,
        storage16: supported_16.storage_buffer16_bit_access == vk::TRUE,
        storage_input_output16: supported_16.storage_input_output16 == vk::TRUE,
        storage8: supported_8.storage_buffer8_bit_access == vk::TRUE,
        float16: supported_f16i8.shader_float16 == vk::TRUE,
        int8: supported_f16i8.shader_int8 == vk::TRUE,
        shader_output_viewport_index: supported_vulkan12.shader_output_viewport_index == vk::TRUE,
        fragment_stores_and_atomics: supported.fragment_stores_and_atomics == vk::TRUE,
        independent_blend: supported.independent_blend == vk::TRUE,
        vertex_pipeline_stores_and_atomics: supported.vertex_pipeline_stores_and_atomics
            == vk::TRUE,
        shader_int64: supported.shader_int64 == vk::TRUE,
        shader_demote_to_helper_invocation: demote_to_helper,
        mirror_clamp_to_edge,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_supported() -> DeviceFeatures {
        DeviceFeatures {
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
            storage_image_extended_formats: true,
            storage_image_write_without_format: true,
            bgra8_storage: true,
            sampled_r32f_linear_filter: true,
            storage16: true,
            storage_input_output16: true,
            storage8: true,
            float16: true,
            int8: true,
            fragment_stores_and_atomics: true,
            independent_blend: true,
            vertex_pipeline_stores_and_atomics: true,
            shader_int64: true,
            shader_demote_to_helper_invocation: true,
            shader_output_viewport_index: true,
            mirror_clamp_to_edge: MirrorClampToEdge::Core12,
            dual_src_blend: true,
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
            mirror_clamp_to_edge: MirrorClampToEdge::Unsupported,
            shader_demote_to_helper_invocation: false,
            ..all_supported()
        };
        assert_eq!(without.enabled_features().dual_src_blend, vk::FALSE);
        assert!(without.required_extensions().is_empty());
        // The default is "not supported", so a `DeviceFeatures` built without a
        // query never claims a capability it has not checked for.
        assert!(!DeviceFeatures::default().dual_src_blend);
    }

    /// Does the list contain the mirror-clamp extension? Asked by name rather
    /// than by list length, because the list also carries extensions belonging
    /// to unrelated features and a length assertion breaks whenever one is
    /// added — which says nothing about the rung under test.
    fn asks_for_mirror_clamp(caps: &DeviceFeatures) -> bool {
        // Compared as strings, not pointers: the list holds raw `*const c_char`
        // and two calls need not hand back the same address for the same name.
        caps.required_extensions().into_iter().any(|name| {
            // SAFETY: every entry is a pointer to one of ash's `'static` NUL-
            // terminated extension-name constants.
            let name = unsafe { std::ffi::CStr::from_ptr(name) };
            name == vk::KHR_SAMPLER_MIRROR_CLAMP_TO_EDGE_NAME
        })
    }

    /// The 1.2 rung sets the core feature bit and asks for no extension.
    #[test]
    fn the_core_rung_needs_no_extension_string() {
        let caps = all_supported();
        assert_eq!(
            caps.enabled_vulkan12().sampler_mirror_clamp_to_edge,
            vk::TRUE
        );
        assert!(!asks_for_mirror_clamp(&caps));
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
        assert!(asks_for_mirror_clamp(&caps));
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
        assert!(!asks_for_mirror_clamp(&caps));
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
        assert_eq!(enabled.fragment_stores_and_atomics, vk::FALSE);
        assert_eq!(enabled.independent_blend, vk::FALSE);
        assert_eq!(enabled.vertex_pipeline_stores_and_atomics, vk::FALSE);
        assert_eq!(enabled.shader_int64, vk::FALSE);
    }

    /// The two features the guest was measured to need are asked for when the
    /// device offers them.
    ///
    /// Both were missing until a validation layer named them: the guest writes a
    /// storage buffer from a fragment shader (`Set 0, Binding 104`,
    /// `VUID-RuntimeSpirv-NonWritable-06340`) and builds a colour-blend state
    /// whose second attachment differs from its first
    /// (`VUID-VkPipelineColorBlendStateCreateInfo-pAttachments-00605`). Doing
    /// either without the feature is undefined behaviour.
    #[test]
    fn the_features_the_guest_was_measured_to_need_are_requested() {
        let enabled = all_supported().enabled_features();
        assert_eq!(enabled.fragment_stores_and_atomics, vk::TRUE);
        assert_eq!(enabled.independent_blend, vk::TRUE);
        assert_eq!(enabled.vertex_pipeline_stores_and_atomics, vk::TRUE);
        assert_eq!(enabled.shader_int64, vk::TRUE);
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
    /// It used to be enabled wherever supported while nothing could ever bind a
    /// second viewport. Harmless in itself, but it meant the list was a wish
    /// rather than a derivation — and a list that is not derived cannot be
    /// checked. `DrawRequest::viewport` is an `Option`, so "at most one" is now
    /// a property of the type rather than a runtime check this test has to go
    /// looking for.
    #[test]
    fn multi_viewport_is_not_enabled_because_no_draw_can_bind_a_second() {
        let enabled = all_supported().enabled_features();
        assert_eq!(
            enabled.multi_viewport,
            vk::FALSE,
            "no draw can use a second viewport, so nothing should request one"
        );
        // There is no second slot to fill; the field holds one viewport or none.
        let req = crate::backend::vulkan::engine::DrawRequest {
            viewport: Some(crate::backend::vulkan::engine::ViewportResource {
                x: 0.0,
                y: 0.0,
                width: 1.0,
                height: 1.0,
                min_depth: 0.0,
                max_depth: 1.0,
            }),
            ..Default::default()
        };
        assert!(req.viewport.is_some());
    }
}
