//! Exact compatibility between one guest plane and one imported linear image.
//!
//! Device-level format support is only the outer gate. A guest-backed image is
//! correct only when the driver's actual subresource layout places its first
//! texel at the plane's declared offset, uses the same row pitch, fits inside
//! the retained packed alias, and accepts a memory type the host pointer also
//! accepts. The imported allocation is the parent resource; each image is a
//! child view alias-bound at its checked plane offset. Those facts belong
//! together because every one participates in the same `vkBindImageMemory`
//! equation.

use ash::vk;

use crate::observe::Decline;

use super::context::DeviceContext;
use super::types::GuestTargetBacking;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WindowPlan {
    bind_offset: u64,
    memory_type_index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LayoutMode {
    DriverLinear,
    ExplicitLinear,
}

// The linear DRM modifier is the API value zero. Unlike vendor modifiers, it
// describes ordinary row-major storage and therefore lets the guest's declared
// byte offset and row pitch be stated directly in the image-create contract.
const DRM_FORMAT_MOD_LINEAR: u64 = 0;

fn subresource_aspect(mode: LayoutMode) -> vk::ImageAspectFlags {
    match mode {
        LayoutMode::DriverLinear => vk::ImageAspectFlags::COLOR,
        LayoutMode::ExplicitLinear => vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum WindowRefusal {
    /// [`crate::env::SHARED_TARGET`] is `off`, so this host takes the arm a
    /// discrete one has no choice about. First, because it is a policy answer
    /// and every check below it is a measurement of the host.
    DisabledByEnv,
    UnsupportedTopology,
    HostImportUnavailable,
    ParentAllocationMismatch,
    ParentImport(super::host_ram::HostRamDecline),
    HostPointerMisaligned,
    SubresourceAfterPlane,
    BindOffsetMisaligned,
    RowPitchMismatch,
    AllocationTooShort,
    NoMemoryType,
    DedicatedBindingRequired,
    ModifierQuery(vk::Result),
    CreateImage(vk::Result),
    BindImage(vk::Result),
}

impl WindowRefusal {
    pub(super) fn slug(self) -> &'static str {
        match self {
            Self::DisabledByEnv => "disabled_by_env",
            Self::UnsupportedTopology => "discrete_topology",
            Self::HostImportUnavailable => "no_host_import",
            Self::ParentAllocationMismatch => "parent_allocation_mismatch",
            Self::ParentImport(inner) => Decline::slug(&inner),
            Self::HostPointerMisaligned => "host_pointer_misaligned",
            Self::SubresourceAfterPlane => "subresource_after_plane",
            Self::BindOffsetMisaligned => "bind_offset_misaligned",
            Self::RowPitchMismatch => "row_pitch_mismatch",
            Self::AllocationTooShort => "allocation_too_short",
            Self::NoMemoryType => "no_memory_type",
            Self::DedicatedBindingRequired => "dedicated_binding_required",
            Self::ModifierQuery(_) => "modifier_query_failed",
            Self::CreateImage(_) => "create_failed",
            Self::BindImage(_) => "bind_failed",
        }
    }

    pub(super) fn result(self) -> Option<vk::Result> {
        match self {
            Self::CreateImage(result) | Self::ModifierQuery(result) | Self::BindImage(result) => {
                Some(result)
            }
            _ => None,
        }
    }
}
impl crate::observe::Decline for WindowRefusal {
    fn slug(&self) -> &'static str {
        (*self).slug()
    }

    /// Delegated with `slug`; see
    /// [`crate::observe::slugs`].
    fn owner(&self) -> &'static str {
        match self {
            Self::ParentImport(inner) => crate::observe::Decline::owner(inner),
            _ => std::any::type_name::<Self>(),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::ParentImport(inner) => crate::observe::Decline::fields(inner),
            _ => self
                .result()
                .map(|result| vec![("result", format!("{result:?}"))])
                .unwrap_or_default(),
        }
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "the planner checks one independently reported value per term of the image-binding equation"
)]
fn plan_window(
    layout: vk::SubresourceLayout,
    requirements: vk::MemoryRequirements,
    allocation_len: u64,
    plane_offset: u64,
    guest_row_pitch: u64,
    pointer_memory_type_bits: u32,
    memory_type_index: Option<u32>,
    requires_dedicated: bool,
) -> Result<WindowPlan, WindowRefusal> {
    let bind_offset = plane_offset
        .checked_sub(layout.offset)
        .ok_or(WindowRefusal::SubresourceAfterPlane)?;
    if requirements.alignment == 0 || !bind_offset.is_multiple_of(requirements.alignment) {
        return Err(WindowRefusal::BindOffsetMisaligned);
    }
    if layout.row_pitch != guest_row_pitch {
        return Err(WindowRefusal::RowPitchMismatch);
    }
    let required_end = bind_offset
        .checked_add(requirements.size)
        .ok_or(WindowRefusal::AllocationTooShort)?;
    if required_end > allocation_len {
        return Err(WindowRefusal::AllocationTooShort);
    }
    if requirements.memory_type_bits & pointer_memory_type_bits == 0 {
        return Err(WindowRefusal::NoMemoryType);
    }
    let memory_type_index = memory_type_index.ok_or(WindowRefusal::NoMemoryType)?;
    if requires_dedicated {
        return Err(WindowRefusal::DedicatedBindingRequired);
    }
    Ok(WindowPlan {
        bind_offset,
        memory_type_index,
    })
}

#[allow(
    clippy::too_many_arguments,
    reason = "the explicit layout validates every term returned for the imported image"
)]
fn plan_explicit_window(
    layout: vk::SubresourceLayout,
    requirements: vk::MemoryRequirements,
    allocation_len: u64,
    plane_offset: u64,
    guest_row_pitch: u64,
    pointer_memory_type_bits: u32,
    memory_type_index: Option<u32>,
    requires_dedicated: bool,
) -> Result<WindowPlan, WindowRefusal> {
    if layout.offset != plane_offset {
        return Err(WindowRefusal::SubresourceAfterPlane);
    }
    if layout.row_pitch != guest_row_pitch {
        return Err(WindowRefusal::RowPitchMismatch);
    }
    if requirements.alignment == 0 || requirements.size > allocation_len {
        return Err(WindowRefusal::AllocationTooShort);
    }
    if requirements.memory_type_bits & pointer_memory_type_bits == 0 {
        return Err(WindowRefusal::NoMemoryType);
    }
    let memory_type_index = memory_type_index.ok_or(WindowRefusal::NoMemoryType)?;
    if requires_dedicated {
        return Err(WindowRefusal::DedicatedBindingRequired);
    }
    Ok(WindowPlan {
        bind_offset: 0,
        memory_type_index,
    })
}

fn required_modifier_features(usage: vk::ImageUsageFlags) -> vk::FormatFeatureFlags {
    let mut required = vk::FormatFeatureFlags::empty();
    if usage.contains(vk::ImageUsageFlags::SAMPLED) {
        required |= vk::FormatFeatureFlags::SAMPLED_IMAGE;
    }
    if usage
        .intersects(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT)
    {
        required |= vk::FormatFeatureFlags::COLOR_ATTACHMENT
            | vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND;
    }
    if usage.contains(vk::ImageUsageFlags::TRANSFER_SRC) {
        required |= vk::FormatFeatureFlags::TRANSFER_SRC;
    }
    if usage.contains(vk::ImageUsageFlags::TRANSFER_DST) {
        required |= vk::FormatFeatureFlags::TRANSFER_DST;
    }
    required
}

fn external_import_is_shareable(features: vk::ExternalMemoryFeatureFlags) -> bool {
    features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
        && !features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY)
}

fn parent_allocation_matches(
    import: &crate::runtime::guest_ram::GuestRamImport,
    backing: GuestTargetBacking,
) -> bool {
    import.host_base() == backing.allocation_host_ptr && import.len() == backing.allocation_len
}

unsafe fn explicit_linear_supported(
    ctx: &DeviceContext,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Result<bool, WindowRefusal> {
    if !ctx.features.image_drm_format_modifier {
        return Ok(false);
    }
    let key = (format.as_raw(), usage.as_raw());
    if let Some(answer) = ctx
        .explicit_linear_support
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(&key)
        .copied()
    {
        return Ok(answer);
    }

    let answer = unsafe { query_explicit_linear_support(ctx, format, usage) }?;
    ctx.explicit_linear_support
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .insert(key, answer);
    Ok(answer)
}

unsafe fn query_explicit_linear_support(
    ctx: &DeviceContext,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Result<bool, WindowRefusal> {
    let mut modifier_list = vk::DrmFormatModifierPropertiesListEXT::default();
    let mut properties = vk::FormatProperties2::default().push_next(&mut modifier_list);
    unsafe {
        ctx.instance
            .get_physical_device_format_properties2(ctx.pd, format, &mut properties)
    };
    let mut modifiers = vec![
        vk::DrmFormatModifierPropertiesEXT::default();
        modifier_list.drm_format_modifier_count as usize
    ];
    let mut modifier_list = vk::DrmFormatModifierPropertiesListEXT::default()
        .drm_format_modifier_properties(&mut modifiers);
    let mut properties = vk::FormatProperties2::default().push_next(&mut modifier_list);
    unsafe {
        ctx.instance
            .get_physical_device_format_properties2(ctx.pd, format, &mut properties)
    };
    let required = required_modifier_features(usage);
    if !modifiers.iter().any(|modifier| {
        modifier.drm_format_modifier == DRM_FORMAT_MOD_LINEAR
            && modifier.drm_format_modifier_plane_count == 1
            && modifier
                .drm_format_modifier_tiling_features
                .contains(required)
    }) {
        return Ok(false);
    }

    let handle = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
    let mut modifier = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
        .drm_format_modifier(DRM_FORMAT_MOD_LINEAR)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let mut external = vk::PhysicalDeviceExternalImageFormatInfo::default().handle_type(handle);
    let info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(format)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(usage)
        .flags(vk::ImageCreateFlags::ALIAS)
        .push_next(&mut modifier)
        .push_next(&mut external);
    let mut external_properties = vk::ExternalImageFormatProperties::default();
    let mut properties = vk::ImageFormatProperties2::default().push_next(&mut external_properties);
    unsafe {
        ctx.instance
            .get_physical_device_image_format_properties2(ctx.pd, &info, &mut properties)
    }
    .map_err(WindowRefusal::ModifierQuery)?;
    Ok(external_import_is_shareable(
        external_properties
            .external_memory_properties
            .external_memory_features,
    ))
}

pub(super) struct ImportedTarget {
    pub image: vk::Image,
}

/// Whether the primary colour attachment may be the guest's own pages.
/// **Default on**; [`crate::env::SHARED_TARGET`]`=off` is the ablation arm.
///
/// Read once. A target created under one answer outlives the draw that created
/// it and is recycled by geometry and format alone, so an answer that changed
/// mid-boot would put both kinds in one registry with nothing able to tell them
/// apart.
fn shared_target_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| shared_target_from(crate::env::read(crate::env::SHARED_TARGET).0))
}

/// The rail's answer for one parsed spelling, split out of the `OnceLock` above
/// so both arms are reachable from a test — a latched answer can be asked only
/// once per process, which is exactly one arm.
fn shared_target_from(switch: crate::env::Switch) -> bool {
    !matches!(switch, crate::env::Switch::Off)
}

/// Create a linear image whose storage is the guest surface allocation itself.
///
/// A refusal is an optional-rail answer: callers keep the ordinary optimal
/// resident. Once this returns an image, only the child image was created: its
/// memory is owned by [`super::host_ram::HostRamImports`] and must not be freed
/// when the child retires.
#[allow(
    clippy::too_many_arguments,
    reason = "the child binding validates its parent, geometry, format and complete Vulkan usage"
)]
pub(super) unsafe fn create(
    ctx: &DeviceContext,
    imports: &mut super::host_ram::HostRamImports,
    import: &crate::runtime::guest_ram::GuestRamImport,
    backing: GuestTargetBacking,
    width: u32,
    height: u32,
    format: vk::Format,
    mut usage: vk::ImageUsageFlags,
) -> Result<ImportedTarget, WindowRefusal> {
    use crate::backend::vulkan::caps::memory_topology::MemoryTopology;

    if !shared_target_enabled() {
        return Err(WindowRefusal::DisabledByEnv);
    }
    if ctx.caps.memory.topology != MemoryTopology::Unified {
        return Err(WindowRefusal::UnsupportedTopology);
    }
    if ctx.external_memory_host.is_none() {
        return Err(WindowRefusal::HostImportUnavailable);
    }
    if !parent_allocation_matches(import, backing) {
        return Err(WindowRefusal::ParentAllocationMismatch);
    }
    let allocation =
        unsafe { imports.allocation(ctx, import) }.map_err(WindowRefusal::ParentImport)?;
    let alignment = ctx.caps.host_pointer.min_alignment;
    if alignment == 0
        || !(backing.allocation_host_ptr as u64).is_multiple_of(alignment)
        || !backing.allocation_len.is_multiple_of(alignment)
    {
        return Err(WindowRefusal::HostPointerMisaligned);
    }
    if ctx.features.attachment_feedback_loop_layout
        && usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
    {
        usage |= vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
    }
    if unsafe { explicit_linear_supported(ctx, format, usage) }? {
        let explicit = unsafe {
            create_with_layout(
                ctx,
                allocation,
                backing,
                width,
                height,
                format,
                usage,
                LayoutMode::ExplicitLinear,
            )
        };
        if explicit.is_ok() {
            imports.retain_child(import);
            return explicit;
        }
        // A format/modifier combination is structural, but an individual row
        // pitch can still violate that modifier's alignment. Preserve the
        // ordinary exact-pitch route where it happens to fit.
        let ordinary = unsafe {
            create_with_layout(
                ctx,
                allocation,
                backing,
                width,
                height,
                format,
                usage,
                LayoutMode::DriverLinear,
            )
        };
        let result = ordinary.or(explicit);
        if result.is_ok() {
            imports.retain_child(import);
        }
        return result;
    }
    let result = unsafe {
        create_with_layout(
            ctx,
            allocation,
            backing,
            width,
            height,
            format,
            usage,
            LayoutMode::DriverLinear,
        )
    };
    if result.is_ok() {
        imports.retain_child(import);
    }
    result
}

#[allow(clippy::too_many_arguments)]
unsafe fn create_with_layout(
    ctx: &DeviceContext,
    allocation: super::host_ram::ImportedHostRam,
    backing: GuestTargetBacking,
    width: u32,
    height: u32,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    mode: LayoutMode,
) -> Result<ImportedTarget, WindowRefusal> {
    let handle = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
    let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(handle);
    let base = vk::ImageCreateInfo::default()
        .flags(vk::ImageCreateFlags::ALIAS | vk::ImageCreateFlags::MUTABLE_FORMAT)
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .usage(usage)
        .initial_layout(vk::ImageLayout::PREINITIALIZED);
    let image = match mode {
        LayoutMode::DriverLinear => {
            let create = base
                .tiling(vk::ImageTiling::LINEAR)
                .push_next(&mut external);
            unsafe { ctx.device.create_image(&create, None) }
        }
        LayoutMode::ExplicitLinear => {
            let plane_layout = [vk::SubresourceLayout {
                offset: backing.plane_offset,
                size: 0,
                row_pitch: backing.row_pitch,
                array_pitch: 0,
                depth_pitch: 0,
            }];
            let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
                .drm_format_modifier(DRM_FORMAT_MOD_LINEAR)
                .plane_layouts(&plane_layout);
            let create = base
                .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                .push_next(&mut external)
                .push_next(&mut explicit);
            unsafe { ctx.device.create_image(&create, None) }
        }
    }
    .map_err(WindowRefusal::CreateImage)?;

    let result = (|| {
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        let info = vk::ImageMemoryRequirementsInfo2::default().image(image);
        unsafe {
            ctx.device
                .get_image_memory_requirements2(&info, &mut requirements)
        };
        let layout = unsafe {
            ctx.device.get_image_subresource_layout(
                image,
                vk::ImageSubresource {
                    // Modifier images describe memory planes, not format
                    // aspects. A single-plane linear modifier therefore has
                    // exactly MEMORY_PLANE_0_EXT; ordinary linear images keep
                    // the colour aspect query.
                    aspect_mask: subresource_aspect(mode),
                    mip_level: 0,
                    array_layer: 0,
                },
            )
        };
        let parent_type_bits = 1_u32
            .checked_shl(allocation.memory_type_index)
            .ok_or(WindowRefusal::NoMemoryType)?;
        let plan = match mode {
            LayoutMode::DriverLinear => plan_window(
                layout,
                requirements.memory_requirements,
                backing.allocation_len,
                backing.plane_offset,
                backing.row_pitch,
                parent_type_bits,
                Some(allocation.memory_type_index),
                dedicated.requires_dedicated_allocation != 0,
            ),
            LayoutMode::ExplicitLinear => plan_explicit_window(
                layout,
                requirements.memory_requirements,
                backing.allocation_len,
                backing.plane_offset,
                backing.row_pitch,
                parent_type_bits,
                Some(allocation.memory_type_index),
                dedicated.requires_dedicated_allocation != 0,
            ),
        }?;
        if let Err(result) = unsafe {
            ctx.device
                .bind_image_memory(image, allocation.memory, plan.bind_offset)
        } {
            return Err(WindowRefusal::BindImage(result));
        }
        Ok(ImportedTarget { image })
    })();
    if result.is_err() {
        unsafe { ctx.device.destroy_image(image, None) };
    }
    result
}

/// Probe the complete binding equation for one live guest surface.
///
/// This creates no memory and changes no rendering behavior. It asks the
/// driver for the linear image's actual layout and memory requirements, then
/// checks them against the guest allocation. The behavior implementation will
/// consume this same planner rather than reconstructing the admission rule.
///
/// # Safety
///
/// `host_ptr..host_ptr + allocation_len` must remain a live host mapping while
/// the pointer-properties query runs, and `ctx` must own the logical device.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn probe_window(
    ctx: &DeviceContext,
    host_ptr: usize,
    allocation_len: u64,
    plane_offset: u64,
    guest_row_pitch: u64,
    width: u32,
    height: u32,
    format: vk::Format,
    mut usage: vk::ImageUsageFlags,
) {
    use crate::backend::vulkan::caps::memory_topology::MemoryTopology;

    if ctx.caps.memory.topology != MemoryTopology::Unified {
        crate::observe::off(format!(
            "vk_linear_target_window verdict=discrete_topology format={format:?} {width}x{height}"
        ));
        return;
    }
    let Some(ext) = ctx.external_memory_host.as_ref() else {
        crate::observe::off(format!(
            "vk_linear_target_window verdict=no_host_import format={format:?} {width}x{height}"
        ));
        return;
    };
    if ctx.features.attachment_feedback_loop_layout {
        usage |= vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
    }
    let handle = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
    let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(handle);
    let create = vk::ImageCreateInfo::default()
        .flags(vk::ImageCreateFlags::ALIAS)
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::LINEAR)
        .usage(usage)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .push_next(&mut external);
    let image = match unsafe { ctx.device.create_image(&create, None) } {
        Ok(image) => image,
        Err(result) => {
            crate::observe::off(format!(
                "vk_linear_target_window verdict=create_failed result={result:?} format={format:?} {width}x{height}"
            ));
            return;
        }
    };
    let requirements = unsafe { ctx.device.get_image_memory_requirements(image) };
    let layout = unsafe {
        ctx.device.get_image_subresource_layout(
            image,
            vk::ImageSubresource {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                array_layer: 0,
            },
        )
    };
    let mut pointer = vk::MemoryHostPointerPropertiesEXT::default();
    let pointer_result = unsafe {
        (ext.fp().get_memory_host_pointer_properties_ext)(
            ext.device(),
            handle,
            host_ptr as *const std::ffi::c_void,
            &mut pointer,
        )
    };
    let pointer_bits = if pointer_result == vk::Result::SUCCESS {
        pointer.memory_type_bits
    } else {
        0
    };
    let compatible_bits = pointer_bits & requirements.memory_type_bits;
    let picked = ctx.memory_type_with(
        compatible_bits,
        allocation_len,
        &ctx.caps
            .memory_request(crate::backend::vulkan::caps::MemoryClass::Upload),
    );
    // This is a probe and the refusal detail is the selector's own; all the
    // window plan needs is whether a type was named.
    let picked = picked.ok();
    let plan = plan_window(
        layout,
        requirements,
        allocation_len,
        plane_offset,
        guest_row_pitch,
        pointer_bits,
        picked.map(|pick| pick.index),
        false,
    );
    let verdict = match (plan, picked) {
        (Ok(_), Some(_)) => "alias_exact",
        (Ok(_), None) => WindowRefusal::NoMemoryType.slug(),
        (Err(reason), _) => reason.slug(),
    };
    let bind_offset = plan.ok().map(|p| p.bind_offset).unwrap_or(u64::MAX);
    crate::observe::off(format!(
        "vk_linear_target_window verdict={verdict} format={format:?} {width}x{height} allocation_len={allocation_len} plane_offset={plane_offset} guest_row_pitch={guest_row_pitch} layout_offset={} layout_row_pitch={} requirements_size={} requirements_align={} bind_offset={bind_offset} image_type_bits=0x{:x} pointer_type_bits=0x{pointer_bits:x} compatible_type_bits=0x{compatible_bits:x} memory_type={}",
        layout.offset,
        layout.row_pitch,
        requirements.size,
        requirements.alignment,
        requirements.memory_type_bits,
        picked.map(|p| p.index.to_string()).unwrap_or_else(|| "none".into()),
    ));
    unsafe { ctx.device.destroy_image(image, None) };
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(offset: u64, row_pitch: u64) -> vk::SubresourceLayout {
        vk::SubresourceLayout {
            offset,
            size: 0,
            row_pitch,
            array_pitch: 0,
            depth_pitch: 0,
        }
    }

    fn requirements(size: u64, alignment: u64, bits: u32) -> vk::MemoryRequirements {
        vk::MemoryRequirements {
            size,
            alignment,
            memory_type_bits: bits,
        }
    }

    #[test]
    fn a_child_can_only_name_the_parent_import_that_owns_its_allocation() {
        let import =
            crate::runtime::guest_ram::GuestRamImport::new_host_allocation(0x1000, 0x4000, 0x1000)
                .expect("aligned synthetic import");
        let backing = GuestTargetBacking {
            allocation_host_ptr: 0x1000,
            allocation_len: 0x4000,
            plane_offset: 0x1000,
            row_pitch: 256,
        };
        assert!(parent_allocation_matches(&import, backing));
        assert!(!parent_allocation_matches(
            &import,
            GuestTargetBacking {
                allocation_host_ptr: 0x5000,
                ..backing
            }
        ));
        assert!(!parent_allocation_matches(
            &import,
            GuestTargetBacking {
                allocation_len: 0x3000,
                ..backing
            }
        ));
    }

    #[test]
    fn an_exact_window_derives_the_binding_offset() {
        assert_eq!(
            plan_window(
                layout(0, 7680),
                requirements(8 << 20, 4096, 0b110),
                12 << 20,
                4096,
                7680,
                0b010,
                Some(1),
                false,
            ),
            Ok(WindowPlan {
                bind_offset: 4096,
                memory_type_index: 1,
            })
        );
    }

    #[test]
    fn every_part_of_the_binding_equation_can_refuse() {
        let req = requirements(8192, 4096, 0b010);
        assert_eq!(
            plan_window(
                layout(8192, 256),
                req,
                16384,
                4096,
                256,
                0b010,
                Some(1),
                false
            ),
            Err(WindowRefusal::SubresourceAfterPlane)
        );
        assert_eq!(
            plan_window(layout(0, 256), req, 16384, 2048, 256, 0b010, Some(1), false),
            Err(WindowRefusal::BindOffsetMisaligned)
        );
        assert_eq!(
            plan_window(layout(0, 512), req, 16384, 4096, 256, 0b010, Some(1), false),
            Err(WindowRefusal::RowPitchMismatch)
        );
        assert_eq!(
            plan_window(layout(0, 256), req, 8192, 4096, 256, 0b010, Some(1), false),
            Err(WindowRefusal::AllocationTooShort)
        );
        assert_eq!(
            plan_window(layout(0, 256), req, 16384, 4096, 256, 0b100, None, false),
            Err(WindowRefusal::NoMemoryType)
        );
        assert_eq!(
            plan_window(layout(0, 256), req, 16384, 4096, 256, 0b010, Some(1), true,),
            Err(WindowRefusal::DedicatedBindingRequired)
        );
    }

    #[test]
    fn explicit_layout_binds_the_import_at_zero() {
        assert_eq!(
            plan_explicit_window(
                layout(4096, 7040),
                requirements(8 << 20, 4096, 0b110),
                12 << 20,
                4096,
                7040,
                0b010,
                Some(1),
                false,
            ),
            Ok(WindowPlan {
                bind_offset: 0,
                memory_type_index: 1,
            })
        );
    }

    #[test]
    fn explicit_layout_must_be_returned_exactly() {
        let req = requirements(8192, 4096, 0b010);
        assert_eq!(
            plan_explicit_window(layout(0, 256), req, 16384, 4096, 256, 0b010, Some(1), false,),
            Err(WindowRefusal::SubresourceAfterPlane)
        );
        assert_eq!(
            plan_explicit_window(
                layout(4096, 512),
                req,
                16384,
                4096,
                256,
                0b010,
                Some(1),
                false,
            ),
            Err(WindowRefusal::RowPitchMismatch)
        );
    }

    #[test]
    fn modifier_features_follow_the_declared_usage() {
        let features = required_modifier_features(
            vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST,
        );
        assert!(features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE));
        assert!(features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT));
        assert!(features.contains(vk::FormatFeatureFlags::COLOR_ATTACHMENT_BLEND));
        assert!(features.contains(vk::FormatFeatureFlags::TRANSFER_SRC));
        assert!(features.contains(vk::FormatFeatureFlags::TRANSFER_DST));
    }

    #[test]
    fn explicit_layout_queries_the_memory_plane() {
        assert_eq!(
            subresource_aspect(LayoutMode::DriverLinear),
            vk::ImageAspectFlags::COLOR
        );
        assert_eq!(
            subresource_aspect(LayoutMode::ExplicitLinear),
            vk::ImageAspectFlags::MEMORY_PLANE_0_EXT
        );
    }

    #[test]
    fn explicit_import_requires_a_non_dedicated_importable_image() {
        assert!(external_import_is_shareable(
            vk::ExternalMemoryFeatureFlags::IMPORTABLE
        ));
        assert!(!external_import_is_shareable(
            vk::ExternalMemoryFeatureFlags::empty()
        ));
        assert!(!external_import_is_shareable(
            vk::ExternalMemoryFeatureFlags::IMPORTABLE
                | vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY
        ));
    }

    /// Only the negative spelling turns the rail off. `Unset` is the shipping
    /// arm, `On` cannot widen anything (the topology and extension gates below
    /// it still decide), and `Unrecognized` is a typo — reading a typo as `off`
    /// would silently move a host onto the copying rail and read as a device
    /// regression rather than as an operator mistake.
    #[test]
    fn only_off_takes_this_host_to_the_copying_rail() {
        use crate::env::Switch;
        assert!(shared_target_from(Switch::Unset));
        assert!(shared_target_from(Switch::On));
        assert!(shared_target_from(Switch::Unrecognized));
        assert!(!shared_target_from(Switch::Off));
    }

    /// Every refusal names itself. A slug copied from a sibling makes two
    /// different reasons one line in the fail log, and the one that gets read is
    /// whichever was written first — the copied-failure-line trap `AGENTS.md`
    /// records. `DisabledByEnv` is the newest and the likeliest to have been
    /// spelled as one of the host measurements beside it.
    #[test]
    fn no_two_refusals_share_a_slug() {
        let all = [
            WindowRefusal::DisabledByEnv,
            WindowRefusal::UnsupportedTopology,
            WindowRefusal::HostImportUnavailable,
            WindowRefusal::ParentAllocationMismatch,
            WindowRefusal::HostPointerMisaligned,
            WindowRefusal::SubresourceAfterPlane,
            WindowRefusal::BindOffsetMisaligned,
            WindowRefusal::RowPitchMismatch,
            WindowRefusal::AllocationTooShort,
            WindowRefusal::NoMemoryType,
            WindowRefusal::DedicatedBindingRequired,
        ];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a.slug(), b.slug(), "{a:?} and {b:?} report as one reason");
            }
        }
    }
}
