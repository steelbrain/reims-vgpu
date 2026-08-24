//! Exact admission of a guest surface allocation as a Vulkan image.
//!
//! This is a binding equation, not topology policy. The child image aliases
//! the canonical host-pointer import only when Vulkan reports the same offset,
//! row pitch, and array/depth pitch for every declared mip, the image fits the
//! parent, and the parent's selected memory type satisfies the image
//! requirements.

use ash::vk;
use reims_vgpu_memory::{GuestImageLayout, GuestRamImport, GuestTargetBacking};
use reims_vgpu_observe::Decline;

use super::{context::DeviceContext, host_ram};

const DRM_FORMAT_MOD_LINEAR: u64 = 0;

fn mutable_view_formats(format: vk::Format) -> Vec<vk::Format> {
    match format {
        vk::Format::R8G8B8A8_UNORM | vk::Format::R8G8B8A8_SRGB => {
            vec![vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_SRGB]
        }
        vk::Format::B8G8R8A8_UNORM | vk::Format::B8G8R8A8_SRGB => {
            vec![vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB]
        }
        other => vec![other],
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WindowPlan {
    bind_offset: u64,
    required_allocation_len: u64,
}

#[derive(Clone, Copy)]
struct WindowAdmission {
    layout: vk::SubresourceLayout,
    requirements: vk::MemoryRequirements,
    backing: GuestTargetBacking,
    guest_layout: GuestImageLayout,
    parent_memory_type: Option<u32>,
    requires_dedicated: bool,
    require_allocation_fit: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LayoutMode {
    DriverLinear,
    ExplicitLinear,
}

impl LayoutMode {
    fn slug(self) -> &'static str {
        match self {
            Self::DriverLinear => "driver_linear",
            Self::ExplicitLinear => "explicit_linear",
        }
    }
}

/// Which term of the sampled aliasing rail's admission rule refused a
/// declaration.
///
/// The rail builds exactly **one `TYPE_2D` view over the whole mip chain of one
/// layer**, and every clause of that sentence is a way to be refused. Each names
/// a different thing for a reader to go and fix — a `D2Array` wants a layered
/// view, a partial mip range wants the view range carried in the resident's
/// identity — so the rule lives here once and the arms that admit onto the rail
/// ask it rather than restating it.
///
/// Restating it is not hypothetical. This rule was written by hand twice, seven
/// terms in the binding arm and two in the length arm, and the shorter copy is
/// the one a reader meets first.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SampledCopyCause {
    /// The base plane is not a 2-D image, so no `TYPE_2D` view describes it.
    /// A volume, a 1-D texture and either array kind all land here, and the
    /// layout is carried because which one it is decides what the rail would
    /// have to learn to build.
    Layout(GuestImageLayout),
    /// The allocation declares levels that are not one Vulkan mip chain.
    ///
    /// Vulkan derives every level's extent from level zero, so a chain is
    /// representable only when the guest's own levels halve in step, share a
    /// dimensional family and layer domain, and each name a pitch that is a
    /// whole number of texels — which is
    /// [`reims_vgpu_memory::GuestImageAllocationLayout::is_vulkan_mip_chain`].
    /// A chain that passes that is admitted; this is the one that does not.
    IrregularMipChain { mips: usize },
    /// The bind names part of the allocation's chain rather than all of it.
    ///
    /// A `TYPE_2D` view can name any contiguous run of levels, so this is not a
    /// view-type limit: it is that the resident's identity records how many
    /// levels its image carries and not which sub-run a bind asked for, so two
    /// binds naming different sub-runs of one allocation would be served one
    /// view. Teaching the rail a partial range means carrying the range in
    /// [`super::pools::ResidentViewKey`], not relaxing this term.
    ViewMip { base: u32, count: u32 },
    /// The bind names a layer other than a whole single layer 0.
    ViewLayer { base: u32, count: u32 },
    /// The declaration is admissible and the rail still could not serve it:
    /// no resident aliasing these guest bytes was available to pin.
    ///
    /// This is the one cause that is not about shape, and it is separate
    /// precisely so it cannot be reported as one. A shape cause tells a reader
    /// to go and teach the rail a new view; this one tells them the rail
    /// already fits and the registry declined, which is a different
    /// investigation entirely.
    ResidentUnavailable,
}

impl SampledCopyCause {
    /// The allocation half of the rule: whether the rail can plan an image over
    /// this shape at all. Both arms that admit onto the rail ask this, which is
    /// what makes them agree by construction rather than by inspection.
    ///
    /// `None` means the allocation is admissible, not that the bind is — the
    /// view half is [`Self::of_view`] and only the binding arm holds its inputs.
    /// A mip count is deliberately **not** a term here. It used to be, and on a
    /// driven Maps boot it refused 949 008 binds of fifty-three textures — every
    /// mipmapped texture the guest samples. The stated reason was that no
    /// `TYPE_2D` view describes a chain, which is untrue: a view type chooses
    /// the dimensionality, and `subresourceRange` chooses the levels, so one
    /// `TYPE_2D` view over N levels is exactly what a mipmapped 2D texture
    /// wants. Whether the *guest's* levels form one Vulkan chain is a separate
    /// question and belongs to
    /// [`reims_vgpu_memory::GuestImageAllocationLayout::is_vulkan_mip_chain`],
    /// which the length arm asks and reports as
    /// [`Self::IrregularMipChain`].
    pub(crate) fn of_allocation(base: GuestImageLayout) -> Option<Self> {
        match base {
            GuestImageLayout::D2 { .. } => None,
            other => Some(Self::Layout(other)),
        }
    }

    /// The view half: which subresource of an admissible allocation the bind
    /// names.
    ///
    /// A component swizzle is deliberately **not** a term here. It used to be,
    /// and on a driven Maps boot it was 99.3 % of every refusal this rule
    /// produced — every single-channel glyph coverage texture the guest samples
    /// through a channel other than its own. A swizzle is a property of the
    /// *view*, not of the memory: the alias carries it as a
    /// `VkComponentMapping` the hardware applies at sample time, so admitting
    /// one costs nothing and refusing one bought a whole texture copy to move
    /// bytes the GPU would have moved for free.
    ///
    /// `allocation_mips` is what the bind's mip range is measured against: the
    /// rail's image carries the allocation's whole chain, so the admissible
    /// view is the one naming all of it. Comparing against a literal `1` here
    /// is what refused every mipmapped texture the guest owns.
    pub(crate) fn of_view(
        base_mip: u32,
        mip_count: u32,
        allocation_mips: u32,
        base_layer: u32,
        layer_count: u32,
    ) -> Option<Self> {
        if base_mip != 0 || mip_count != allocation_mips {
            return Some(Self::ViewMip {
                base: base_mip,
                count: mip_count,
            });
        }
        if base_layer != 0 || layer_count != 1 {
            return Some(Self::ViewLayer {
                base: base_layer,
                count: layer_count,
            });
        }
        None
    }

    fn slug(self) -> &'static str {
        match self {
            Self::Layout(GuestImageLayout::D1 { .. }) => "sampled_copy_layout_1d",
            Self::Layout(GuestImageLayout::D1Array { .. }) => "sampled_copy_layout_1d_array",
            // Reachable only through a caller that built the cause itself; the
            // constructors above never pair `D2` with this variant.
            Self::Layout(GuestImageLayout::D2 { .. }) => "sampled_copy_layout_2d",
            Self::Layout(GuestImageLayout::D2Array { .. }) => "sampled_copy_layout_2d_array",
            Self::Layout(GuestImageLayout::D3 { .. }) => "sampled_copy_layout_3d",
            Self::IrregularMipChain { .. } => "sampled_copy_irregular_mip_chain",
            Self::ViewMip { .. } => "sampled_copy_view_mip",
            Self::ViewLayer { .. } => "sampled_copy_view_layer",
            Self::ResidentUnavailable => "sampled_copy_resident_unavailable",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WindowRefusal {
    HostImportUnavailable,
    /// The operator withheld the aliasing rail with `REIMS_VGPU_SAMPLED_ALIAS=off`.
    ///
    /// A narrowing switch, so this is a refusal and not a capability: the bind
    /// falls to the imported-buffer copy rail the same way every shape refusal
    /// does. It is named so an ablation boot can be told apart from a host that
    /// genuinely cannot alias.
    SampledAliasDisabledByEnv,
    UnsupportedImageShape {
        layers: u32,
        volume: bool,
        cube: bool,
        arrayed: bool,
        one_dim: bool,
    },
    /// A sampled declaration the aliasing rail does not admit. Aliasing binds
    /// one image over one allocation shape, so anything that reinterprets the
    /// allocation goes to the imported-buffer copy rail instead.
    ///
    /// The cause is carried because the ways to be refused have different
    /// fixes and one `reason=` could not tell them apart: a boot
    /// refusing ten thousand binds said only that a copy was required, which
    /// is the thing a reader already knew.
    SampledContentRequiresCopy(SampledCopyCause),
    /// The imported memory type carries no `HOST_COHERENT`, so a guest CPU
    /// write into the aliased pages is not guaranteed visible to the device.
    ///
    /// The repair a non-coherent mapping needs is `vkFlushMappedMemoryRanges`
    /// over the writer's mapping, and the writer here is the guest, whose
    /// mapping this device does not own and cannot flush. There is nothing to
    /// arrange, so the alias is refused and the copy rail — which reads the
    /// bytes through a buffer whose own visibility this device does control —
    /// serves the bind.
    SampledAliasHostWritesNotCoherent {
        memory_type_index: u32,
    },
    /// The guest row pitch is not a whole number of texels, so no
    /// `bufferRowLength` describes it and the materialization copy cannot name
    /// the rows it has to land.
    SampledAliasRowPitchNotTexelMultiple {
        row_pitch: u64,
        bytes_per_texel: u64,
    },
    AmbiguousResidentBacking {
        matches: usize,
    },
    ParentAllocationMismatch,
    ParentImport(host_ram::HostRamDecline),
    HostPointerMisaligned,
    ResourceWindowTooShort,
    /// Vulkan puts the base subresource further into the image than the guest
    /// plane is into the allocation, so no bind offset places one on the other.
    ///
    /// The two numbers are counted from different bases — `plane_offset` from
    /// the allocation, `subresource_offset` from the image — and the refusal is
    /// that the subtraction between them underflows. That is a real
    /// impossibility rather than a mismatch: an image cannot be bound at a
    /// negative offset.
    SubresourceAfterPlane {
        plane_offset: u64,
        subresource_offset: u64,
    },
    BindOffsetMisaligned,
    RowPitchMismatch {
        /// Which level disagreed. Two sites raise this — the plane's own check
        /// and the per-level chain walk — and without the level their lines
        /// read identically, so a reader cannot tell a chain this host lays out
        /// its own way from a plane it would never have accepted.
        mip_level: u32,
        /// That level's texel width. A reader divides each pitch by it to get
        /// the bytes per texel each side believes in, which is what separates a
        /// host alignment choice from a disagreement about the format.
        width: u32,
        guest: u64,
        host: u64,
    },
    ArrayPitchMismatch,
    DepthPitchMismatch,
    MipOffsetMismatch {
        mip_level: u32,
        guest_offset: u64,
        host_offset: u64,
        /// The two bases the host offset was translated through, reported
        /// because their difference is what a mismatch is actually made of and
        /// a reader cannot recover either one from the difference alone.
        image_base: u64,
        resource_base: u64,
        /// The backing's own plane, and the layout mode whose subresource query
        /// produced the host side. The bind offset is `plane_offset` less the
        /// host subresource offset, so without both a reader cannot tell a
        /// backing that names a different plane from a driver that reports a
        /// different subresource base.
        plane_offset: u64,
        mode: LayoutMode,
    },
    BindingRangeOverflow,
    AllocationTooShort {
        required_end: u64,
        allocation_len: u64,
    },
    NoMemoryType,
    DedicatedBindingRequired,
    ModifierQuery(vk::Result),
    CreateImage {
        result: vk::Result,
        mode: LayoutMode,
        format: vk::Format,
        image_type: vk::ImageType,
        width: u32,
        height: u32,
        depth: u32,
        mip_levels: u32,
        array_layers: u32,
        plane_offset: u64,
        row_pitch: u64,
    },
    CreateView(vk::Result),
    BindImage(vk::Result),
}

impl Decline for WindowRefusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::HostImportUnavailable => "no_host_import",
            Self::SampledAliasDisabledByEnv => "sampled_alias_disabled_by_env",
            Self::UnsupportedImageShape { .. } => "unsupported_image_shape",
            Self::SampledContentRequiresCopy(cause) => cause.slug(),
            Self::SampledAliasHostWritesNotCoherent { .. } => {
                "sampled_alias_host_writes_not_coherent"
            }
            Self::SampledAliasRowPitchNotTexelMultiple { .. } => {
                "sampled_alias_row_pitch_not_texel_multiple"
            }
            Self::AmbiguousResidentBacking { .. } => "ambiguous_resident_backing",
            Self::ParentAllocationMismatch => "parent_allocation_mismatch",
            Self::ParentImport(inner) => inner.slug(),
            Self::HostPointerMisaligned => "host_pointer_misaligned",
            Self::ResourceWindowTooShort => "resource_window_too_short",
            Self::SubresourceAfterPlane { .. } => "subresource_after_plane",
            Self::BindOffsetMisaligned => "bind_offset_misaligned",
            Self::RowPitchMismatch { .. } => "row_pitch_mismatch",
            Self::ArrayPitchMismatch => "array_pitch_mismatch",
            Self::DepthPitchMismatch => "depth_pitch_mismatch",
            Self::MipOffsetMismatch { .. } => "mip_offset_mismatch",
            Self::BindingRangeOverflow => "binding_range_overflow",
            Self::AllocationTooShort { .. } => "allocation_too_short",
            Self::NoMemoryType => "no_memory_type",
            Self::DedicatedBindingRequired => "dedicated_binding_required",
            Self::ModifierQuery(_) => "modifier_query_failed",
            Self::CreateImage { .. } => "create_failed",
            Self::CreateView(_) => "view_create_failed",
            Self::BindImage(_) => "bind_failed",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::UnsupportedImageShape {
                layers,
                volume,
                cube,
                arrayed,
                one_dim,
            } => vec![
                ("layers", layers.to_string()),
                ("volume", u8::from(*volume).to_string()),
                ("cube", u8::from(*cube).to_string()),
                ("arrayed", u8::from(*arrayed).to_string()),
                ("one_dim", u8::from(*one_dim).to_string()),
            ],
            Self::AllocationTooShort {
                required_end,
                allocation_len,
            } => vec![
                ("required_end", required_end.to_string()),
                ("allocation_len", allocation_len.to_string()),
            ],
            Self::MipOffsetMismatch {
                mip_level,
                guest_offset,
                host_offset,
                image_base,
                resource_base,
                plane_offset,
                mode,
            } => vec![
                ("mip", mip_level.to_string()),
                ("guest_offset", guest_offset.to_string()),
                ("host_offset", host_offset.to_string()),
                ("image_base", image_base.to_string()),
                ("resource_base", resource_base.to_string()),
                ("plane_offset", plane_offset.to_string()),
                ("mode", mode.slug().to_string()),
            ],
            Self::SubresourceAfterPlane {
                plane_offset,
                subresource_offset,
            } => vec![
                ("plane_offset", plane_offset.to_string()),
                ("subresource_offset", subresource_offset.to_string()),
            ],
            Self::RowPitchMismatch {
                mip_level,
                width,
                guest,
                host,
            } => vec![
                ("mip", mip_level.to_string()),
                ("width", width.to_string()),
                ("guest_pitch", guest.to_string()),
                ("host_pitch", host.to_string()),
            ],
            Self::AmbiguousResidentBacking { matches } => {
                vec![("matches", matches.to_string())]
            }
            Self::SampledAliasHostWritesNotCoherent { memory_type_index } => {
                vec![("memory_type", memory_type_index.to_string())]
            }
            Self::SampledAliasRowPitchNotTexelMultiple {
                row_pitch,
                bytes_per_texel,
            } => vec![
                ("row_pitch", row_pitch.to_string()),
                ("bytes_per_texel", bytes_per_texel.to_string()),
            ],
            Self::ParentImport(inner) => inner.fields(),
            Self::CreateImage {
                result,
                mode,
                format,
                image_type,
                width,
                height,
                depth,
                mip_levels,
                array_layers,
                plane_offset,
                row_pitch,
            } => vec![
                ("result", format!("{result:?}")),
                ("mode", mode.slug().to_string()),
                ("format", format.as_raw().to_string()),
                ("image_type", image_type.as_raw().to_string()),
                ("width", width.to_string()),
                ("height", height.to_string()),
                ("depth", depth.to_string()),
                ("mips", mip_levels.to_string()),
                ("layers", array_layers.to_string()),
                ("plane_offset", plane_offset.to_string()),
                ("row_pitch", row_pitch.to_string()),
            ],
            Self::ModifierQuery(result) | Self::CreateView(result) | Self::BindImage(result) => {
                vec![("result", format!("{result:?}"))]
            }
            _ => Vec::new(),
        }
    }
}

reims_vgpu_observe::decline_display!(WindowRefusal);

/// Place an aliasing image so that its base subresource lands on the guest plane.
///
/// Both linear modes use this. They differ in *who* chooses the base
/// subresource's image-relative offset — the driver picks one under
/// `VK_IMAGE_TILING_LINEAR`, and the explicit mode declares the guest's own —
/// but the placement question is the same either way and the bind offset is the
/// answer to it. An arm that instead required the reported offset to equal the
/// allocation-relative plane offset was comparing two different bases and could
/// only admit a plane sitting at byte zero of guest RAM.
fn plan_linear_window(
    layout: vk::SubresourceLayout,
    requirements: vk::MemoryRequirements,
    backing: GuestTargetBacking,
    guest_layout: GuestImageLayout,
    parent_memory_type: Option<u32>,
    requires_dedicated: bool,
    require_allocation_fit: bool,
) -> Result<WindowPlan, WindowRefusal> {
    let bind_offset = backing.plane_offset.checked_sub(layout.offset).ok_or(
        WindowRefusal::SubresourceAfterPlane {
            plane_offset: backing.plane_offset,
            subresource_offset: layout.offset,
        },
    )?;
    if requirements.alignment == 0 || !bind_offset.is_multiple_of(requirements.alignment) {
        return Err(WindowRefusal::BindOffsetMisaligned);
    }
    validate_common(
        WindowAdmission {
            layout,
            requirements,
            backing,
            guest_layout,
            parent_memory_type,
            requires_dedicated,
            require_allocation_fit,
        },
        bind_offset,
    )
}

fn validate_common(
    admission: WindowAdmission,
    bind_offset: u64,
) -> Result<WindowPlan, WindowRefusal> {
    let WindowAdmission {
        layout,
        requirements,
        backing,
        guest_layout,
        parent_memory_type,
        requires_dedicated,
        require_allocation_fit,
    } = admission;
    if layout.row_pitch != backing.row_pitch {
        return Err(WindowRefusal::RowPitchMismatch {
            // The plane is the chain's own level zero: this check runs against
            // the layout Vulkan reported for mip zero, so naming any other
            // level here would be a second, wrong spelling of the same thing.
            mip_level: 0,
            width: guest_layout.width(),
            guest: backing.row_pitch,
            host: layout.row_pitch,
        });
    }
    match guest_layout {
        GuestImageLayout::D1Array {
            layers,
            array_pitch,
            ..
        }
        | GuestImageLayout::D2Array {
            layers,
            array_pitch,
            ..
        } if layers > 1 && layout.array_pitch != array_pitch => {
            return Err(WindowRefusal::ArrayPitchMismatch);
        }
        GuestImageLayout::D3 {
            depth, depth_pitch, ..
        } if depth > 1 && layout.depth_pitch != depth_pitch => {
            return Err(WindowRefusal::DepthPitchMismatch);
        }
        _ => {}
    }
    let required_end = bind_offset
        .checked_add(requirements.size)
        .ok_or(WindowRefusal::BindingRangeOverflow)?;
    if requirements.alignment == 0
        || (require_allocation_fit && required_end > backing.allocation_len)
    {
        return Err(WindowRefusal::AllocationTooShort {
            required_end,
            allocation_len: backing.allocation_len,
        });
    }
    if let Some(parent_memory_type) = parent_memory_type {
        let parent_bit = 1_u32
            .checked_shl(parent_memory_type)
            .ok_or(WindowRefusal::NoMemoryType)?;
        if requirements.memory_type_bits & parent_bit == 0 {
            return Err(WindowRefusal::NoMemoryType);
        }
    }
    if requires_dedicated {
        return Err(WindowRefusal::DedicatedBindingRequired);
    }
    Ok(WindowPlan {
        bind_offset,
        required_allocation_len: required_end,
    })
}

/// Where an image's subresource offsets and a guest resource's mip offsets meet.
///
/// Three bases converge on this comparison and no two of them are counted from
/// the same place. A `VkSubresourceLayout::offset` is image-relative. The bind
/// offset places that image inside the imported allocation, which for this
/// device is a whole RAMBlock. A [`reims_vgpu_memory::GuestImageMipLayout`]
/// offset is relative to the guest *resource*, whose own start in the allocation
/// is `GuestTargetBacking::resource_offset` — the producer builds a base mip's
/// offset as `plane_offset - resource_offset` and says so.
///
/// Comparing the first two against the third without the resource base is a
/// subtraction short a term, and it does not read as one: both sides are byte
/// offsets of the right order for a small texture and only disagree once the
/// resource sits deep in guest RAM, which is every resource. The translation
/// therefore lives here rather than at each comparison.
#[derive(Clone, Copy)]
struct MipPlacement {
    /// Allocation-relative byte at which the image's memory begins.
    image_base: u64,
    /// Allocation-relative byte at which the guest resource begins.
    resource_base: u64,
    /// The backing's own plane, carried for the refusal rather than the
    /// comparison: `image_base` is already this less the host subresource
    /// offset, and a reader given only the difference cannot separate the two.
    plane_offset: u64,
    /// Which layout mode's subresource query produced the host side.
    mode: LayoutMode,
}

impl MipPlacement {
    /// The guest-resource-relative offset a host subresource layout lands on.
    fn resource_relative(self, host_offset: u64) -> Option<u64> {
        self.image_base
            .checked_add(host_offset)?
            .checked_sub(self.resource_base)
    }
}

fn validate_mip_subresource(
    mip_level: u32,
    host: vk::SubresourceLayout,
    guest: reims_vgpu_memory::GuestImageMipLayout,
    placement: MipPlacement,
) -> Result<(), WindowRefusal> {
    let host_offset = placement
        .resource_relative(host.offset)
        .ok_or(WindowRefusal::BindingRangeOverflow)?;
    if host_offset != guest.resource_relative_offset {
        return Err(WindowRefusal::MipOffsetMismatch {
            mip_level,
            guest_offset: guest.resource_relative_offset,
            host_offset,
            image_base: placement.image_base,
            resource_base: placement.resource_base,
            plane_offset: placement.plane_offset,
            mode: placement.mode,
        });
    }
    if host.row_pitch != guest.row_pitch {
        return Err(WindowRefusal::RowPitchMismatch {
            mip_level,
            width: guest.layout.width(),
            guest: guest.row_pitch,
            host: host.row_pitch,
        });
    }
    match guest.layout {
        GuestImageLayout::D1Array {
            layers,
            array_pitch,
            ..
        }
        | GuestImageLayout::D2Array {
            layers,
            array_pitch,
            ..
        } if layers > 1 && host.array_pitch != array_pitch => {
            Err(WindowRefusal::ArrayPitchMismatch)
        }
        GuestImageLayout::D3 {
            depth, depth_pitch, ..
        } if depth > 1 && host.depth_pitch != depth_pitch => Err(WindowRefusal::DepthPitchMismatch),
        _ => Ok(()),
    }
}

fn required_features(usage: vk::ImageUsageFlags) -> vk::FormatFeatureFlags {
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

unsafe fn explicit_linear_supported(
    ctx: &DeviceContext,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    image_type: vk::ImageType,
) -> Result<bool, WindowRefusal> {
    if !ctx.features.image_drm_format_modifier {
        return Ok(false);
    }
    let key = (format.as_raw(), usage.as_raw(), image_type.as_raw());
    if let Some(answer) = ctx
        .explicit_linear_support
        .lock()
        .unwrap_or_else(|poison| poison.into_inner())
        .get(&key)
        .copied()
    {
        return Ok(answer);
    }
    let answer = unsafe { query_explicit_linear_support(ctx, format, usage, image_type) }?;
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
    image_type: vk::ImageType,
) -> Result<bool, WindowRefusal> {
    let mut list = vk::DrmFormatModifierPropertiesListEXT::default();
    let mut properties = vk::FormatProperties2::default().push_next(&mut list);
    unsafe {
        ctx.instance
            .get_physical_device_format_properties2(ctx.pd, format, &mut properties)
    };
    let mut modifiers = vec![
        vk::DrmFormatModifierPropertiesEXT::default();
        list.drm_format_modifier_count as usize
    ];
    let mut list = vk::DrmFormatModifierPropertiesListEXT::default()
        .drm_format_modifier_properties(&mut modifiers);
    let mut properties = vk::FormatProperties2::default().push_next(&mut list);
    unsafe {
        ctx.instance
            .get_physical_device_format_properties2(ctx.pd, format, &mut properties)
    };
    let required = required_features(usage);
    if !modifiers.iter().any(|modifier| {
        modifier.drm_format_modifier == DRM_FORMAT_MOD_LINEAR
            && modifier.drm_format_modifier_plane_count == 1
            && modifier
                .drm_format_modifier_tiling_features
                .contains(required)
    }) {
        return Ok(false);
    }

    let handle = ctx.caps.host_pointer.handle_type;
    let mut modifier = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
        .drm_format_modifier(DRM_FORMAT_MOD_LINEAR)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let mut external = vk::PhysicalDeviceExternalImageFormatInfo::default().handle_type(handle);
    let view_formats = mutable_view_formats(format);
    let mut format_list = vk::ImageFormatListCreateInfo::default().view_formats(&view_formats);
    let info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(format)
        .ty(image_type)
        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
        .usage(usage)
        .flags(vk::ImageCreateFlags::ALIAS | vk::ImageCreateFlags::MUTABLE_FORMAT)
        .push_next(&mut modifier)
        .push_next(&mut format_list)
        .push_next(&mut external);
    let mut external_properties = vk::ExternalImageFormatProperties::default();
    let mut properties = vk::ImageFormatProperties2::default().push_next(&mut external_properties);
    let query = unsafe {
        ctx.instance
            .get_physical_device_image_format_properties2(ctx.pd, &info, &mut properties)
    };
    if query == Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) {
        return Ok(false);
    }
    query.map_err(WindowRefusal::ModifierQuery)?;
    let features = external_properties
        .external_memory_properties
        .external_memory_features;
    Ok(
        features.contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE)
            && !features.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY),
    )
}

pub(crate) struct ImportedImage {
    pub image: vk::Image,
}

#[derive(Clone, Copy)]
struct ImageGeometry {
    image_type: vk::ImageType,
    extent: vk::Extent3D,
    array_layers: u32,
}

fn image_geometry(layout: GuestImageLayout) -> ImageGeometry {
    ImageGeometry {
        image_type: match layout {
            GuestImageLayout::D1 { .. } | GuestImageLayout::D1Array { .. } => {
                vk::ImageType::TYPE_1D
            }
            GuestImageLayout::D2 { .. } | GuestImageLayout::D2Array { .. } => {
                vk::ImageType::TYPE_2D
            }
            GuestImageLayout::D3 { .. } => vk::ImageType::TYPE_3D,
        },
        extent: vk::Extent3D {
            width: layout.width(),
            height: layout.height(),
            depth: layout.depth(),
        },
        array_layers: layout.array_layers(),
    }
}

fn note_explicit_linear_declined(
    decline: &WindowRefusal,
    backing: GuestTargetBacking,
    allocation_layout: &reims_vgpu_memory::GuestImageAllocationLayout,
    format: vk::Format,
    image_type: vk::ImageType,
    usage: vk::ImageUsageFlags,
) {
    let key = explicit_linear_decline_key(
        decline,
        backing,
        allocation_layout,
        format,
        image_type,
        usage,
    );
    if reims_vgpu_observe::first_sight("sampled_guest_image_explicit_declined", key) {
        reims_vgpu_observe::Emit::decline("sampled_guest_image_explicit_declined", decline).off();
    }
}

fn explicit_linear_decline_key(
    decline: &WindowRefusal,
    backing: GuestTargetBacking,
    allocation_layout: &reims_vgpu_memory::GuestImageAllocationLayout,
    format: vk::Format,
    image_type: vk::ImageType,
    usage: vk::ImageUsageFlags,
) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut key = std::collections::hash_map::DefaultHasher::new();
    (
        decline.slug(),
        backing.resource_offset,
        backing.resource_len,
        backing.plane_offset,
        backing.row_pitch,
        allocation_layout,
        format.as_raw(),
        image_type.as_raw(),
        usage.as_raw(),
    )
        .hash(&mut key);
    key.finish()
}

fn note_explicit_linear_unavailable(
    format: vk::Format,
    image_type: vk::ImageType,
    usage: vk::ImageUsageFlags,
) {
    use std::hash::{Hash, Hasher};
    let mut key = std::collections::hash_map::DefaultHasher::new();
    (format.as_raw(), image_type.as_raw(), usage.as_raw()).hash(&mut key);
    let key = key.finish();
    if reims_vgpu_observe::first_sight("sampled_guest_image_explicit_unavailable", key) {
        reims_vgpu_observe::off(format!(
            "sampled_guest_image_explicit_unavailable format={} image_type={} usage={:#x}",
            format.as_raw(),
            image_type.as_raw(),
            usage.as_raw(),
        ));
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn create(
    ctx: &DeviceContext,
    imports: &mut host_ram::HostRamImports,
    import: &GuestRamImport,
    backing: GuestTargetBacking,
    layout: GuestImageLayout,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Result<ImportedImage, WindowRefusal> {
    // The chain is resource-relative and `plane_offset` is allocation-relative,
    // so the sole mip's offset is the distance between the plane and the
    // resource that holds it -- not the plane itself. Handing the plane over
    // directly declared every image's base subresource to be hundreds of
    // megabytes into its own memory.
    let allocation_layout = reims_vgpu_memory::GuestImageAllocationLayout::single(
        backing
            .plane_offset
            .checked_sub(backing.resource_offset)
            .ok_or(WindowRefusal::ResourceWindowTooShort)?,
        backing.row_pitch,
        layout,
    );
    unsafe {
        create_allocation(
            ctx,
            imports,
            import,
            backing,
            &allocation_layout,
            format,
            usage,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn create_allocation(
    ctx: &DeviceContext,
    imports: &mut host_ram::HostRamImports,
    import: &GuestRamImport,
    backing: GuestTargetBacking,
    allocation_layout: &reims_vgpu_memory::GuestImageAllocationLayout,
    format: vk::Format,
    mut usage: vk::ImageUsageFlags,
) -> Result<ImportedImage, WindowRefusal> {
    if ctx.external_memory_host.is_none() {
        return Err(WindowRefusal::HostImportUnavailable);
    }
    if import.host_base() != backing.allocation_host_ptr || import.len() != backing.allocation_len {
        return Err(WindowRefusal::ParentAllocationMismatch);
    }
    let alignment = ctx.caps.host_pointer.min_alignment;
    if alignment == 0
        || !(backing.allocation_host_ptr as u64).is_multiple_of(alignment)
        || !backing.allocation_len.is_multiple_of(alignment)
    {
        return Err(WindowRefusal::HostPointerMisaligned);
    }
    let bytes_per_texel = crate::translate::pixel::texel_layout_of(format)
        .map(|layout| u64::from(layout.bytes_per_texel()))
        .ok_or(WindowRefusal::ResourceWindowTooShort)?;
    if !allocation_layout.is_vulkan_mip_chain(bytes_per_texel) {
        return Err(WindowRefusal::ResourceWindowTooShort);
    }
    for mip in allocation_layout.mips.iter() {
        let mip_backing = mip
            .plane_in(backing)
            .ok_or(WindowRefusal::ResourceWindowTooShort)?;
        if mip_backing
            .visible_image_window(mip.layout, bytes_per_texel)
            .is_none()
        {
            return Err(WindowRefusal::ResourceWindowTooShort);
        }
    }
    let allocation =
        unsafe { imports.allocation(ctx, import) }.map_err(WindowRefusal::ParentImport)?;
    if ctx.features.attachment_feedback_loop_layout
        && usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
    {
        usage |= vk::ImageUsageFlags::ATTACHMENT_FEEDBACK_LOOP_EXT;
    }

    let geometry = image_geometry(
        allocation_layout
            .base()
            .ok_or(WindowRefusal::ResourceWindowTooShort)?
            .layout,
    );
    let explicit = unsafe { explicit_linear_supported(ctx, format, usage, geometry.image_type) }?;
    let result = if explicit {
        let first = unsafe {
            create_with_layout(
                ctx,
                allocation,
                backing,
                allocation_layout,
                format,
                usage,
                LayoutMode::ExplicitLinear,
            )
        };
        match first {
            Ok(image) => Ok(image),
            Err(decline) => {
                note_explicit_linear_declined(
                    &decline,
                    backing,
                    allocation_layout,
                    format,
                    geometry.image_type,
                    usage,
                );
                unsafe {
                    create_with_layout(
                        ctx,
                        allocation,
                        backing,
                        allocation_layout,
                        format,
                        usage,
                        LayoutMode::DriverLinear,
                    )
                }
            }
        }
    } else {
        note_explicit_linear_unavailable(format, geometry.image_type, usage);
        unsafe {
            create_with_layout(
                ctx,
                allocation,
                backing,
                allocation_layout,
                format,
                usage,
                LayoutMode::DriverLinear,
            )
        }
    };
    if result.is_ok() {
        imports.retain_child(import);
    }
    result
}

#[allow(clippy::too_many_arguments)]
unsafe fn plan_image_with_layout(
    ctx: &DeviceContext,
    backing: GuestTargetBacking,
    allocation_layout: &reims_vgpu_memory::GuestImageAllocationLayout,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    mode: LayoutMode,
    parent_memory_type: Option<u32>,
    require_allocation_fit: bool,
) -> Result<(vk::Image, WindowPlan), WindowRefusal> {
    let handle = ctx.caps.host_pointer.handle_type;
    let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(handle);
    let base_mip = allocation_layout
        .base()
        .ok_or(WindowRefusal::ResourceWindowTooShort)?;
    // The explicit plane layout below declares where plane zero sits inside the
    // image's *memory*, and that memory is the whole parent allocation -- so the
    // number it wants is allocation-relative, which the mip chain is not.
    let base_plane = base_mip
        .plane_in(backing)
        .ok_or(WindowRefusal::ResourceWindowTooShort)?
        .plane_offset;
    let mip_levels = allocation_layout
        .mip_level_count()
        .ok_or(WindowRefusal::ResourceWindowTooShort)?;
    let guest_layout = base_mip.layout;
    let geometry = image_geometry(guest_layout);
    let base = vk::ImageCreateInfo::default()
        .flags(vk::ImageCreateFlags::ALIAS | vk::ImageCreateFlags::MUTABLE_FORMAT)
        .image_type(geometry.image_type)
        .format(format)
        .extent(geometry.extent)
        .mip_levels(mip_levels)
        .array_layers(geometry.array_layers)
        .samples(vk::SampleCountFlags::TYPE_1)
        .usage(usage)
        // External-memory images are born without Vulkan-visible contents.
        // Guest bytes are materialized through the buffer transfer rail before
        // the first LOAD; later attachment writes keep this image authoritative.
        //
        // `UNDEFINED` is forced by the specification here, not chosen, and
        // `PREINITIALIZED` is not an available alternative for carrying the
        // guest's existing bytes across image creation. Two valid-usage
        // statements compose to close that door:
        //
        // - VUID-vkBindImageMemory-memory-02989 requires memory created by
        //   *any* import operation other than a non-NULL
        //   `VkImportAndroidHardwareBufferInfoANDROID` to be bound to an image
        //   whose `VkExternalMemoryImageCreateInfo::handleTypes` names that
        //   handle type. Host-pointer import is not in that exclusion list, so
        //   the `external` chain pushed below is mandatory rather than an
        //   optimization.
        // - VUID-VkImageCreateInfo-pNext-01443 then requires `initialLayout` to
        //   be `UNDEFINED` whenever such a chain is present with non-zero
        //   `handleTypes`.
        //
        // That is a valid-usage rule and not a capability, so no host can
        // report otherwise and there is nothing here to measure. An aliasing
        // rail needing the guest's prior bytes must materialize them after
        // creation; it cannot inherit them through the initial layout.
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let view_formats = mutable_view_formats(format);
    let mut format_list = vk::ImageFormatListCreateInfo::default().view_formats(&view_formats);
    let image = match mode {
        LayoutMode::DriverLinear => unsafe {
            ctx.device.create_image(
                &base
                    .tiling(vk::ImageTiling::LINEAR)
                    .push_next(&mut format_list)
                    .push_next(&mut external),
                None,
            )
        },
        LayoutMode::ExplicitLinear => {
            let layouts = [vk::SubresourceLayout {
                offset: base_plane,
                size: 0,
                row_pitch: base_mip.row_pitch,
                array_pitch: match guest_layout {
                    GuestImageLayout::D1Array { array_pitch, .. }
                    | GuestImageLayout::D2Array { array_pitch, .. } => array_pitch,
                    _ => 0,
                },
                depth_pitch: match guest_layout {
                    GuestImageLayout::D3 { depth_pitch, .. } => depth_pitch,
                    _ => 0,
                },
            }];
            let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
                .drm_format_modifier(DRM_FORMAT_MOD_LINEAR)
                .plane_layouts(&layouts);
            unsafe {
                ctx.device.create_image(
                    &base
                        .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
                        .push_next(&mut format_list)
                        .push_next(&mut external)
                        .push_next(&mut explicit),
                    None,
                )
            }
        }
    }
    .map_err(|result| WindowRefusal::CreateImage {
        result,
        mode,
        format,
        image_type: geometry.image_type,
        width: geometry.extent.width,
        height: geometry.extent.height,
        depth: geometry.extent.depth,
        mip_levels,
        array_layers: geometry.array_layers,
        plane_offset: base_plane,
        row_pitch: base_mip.row_pitch,
    })?;

    let result = (|| {
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        unsafe {
            ctx.device.get_image_memory_requirements2(
                &vk::ImageMemoryRequirementsInfo2::default().image(image),
                &mut requirements,
            )
        };
        let aspect_mask = match mode {
            LayoutMode::DriverLinear => vk::ImageAspectFlags::COLOR,
            LayoutMode::ExplicitLinear => vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
        };
        let layout = unsafe {
            ctx.device.get_image_subresource_layout(
                image,
                vk::ImageSubresource {
                    aspect_mask,
                    mip_level: 0,
                    array_layer: 0,
                },
            )
        };
        let plan = plan_linear_window(
            layout,
            requirements.memory_requirements,
            backing,
            guest_layout,
            parent_memory_type,
            dedicated.requires_dedicated_allocation != 0,
            require_allocation_fit,
        )?;
        let placement = MipPlacement {
            image_base: plan.bind_offset,
            resource_base: backing.resource_offset,
            plane_offset: backing.plane_offset,
            mode,
        };
        for (mip_level, guest) in allocation_layout.mips.iter().copied().enumerate() {
            let mip_level =
                u32::try_from(mip_level).map_err(|_| WindowRefusal::BindingRangeOverflow)?;
            let host = unsafe {
                ctx.device.get_image_subresource_layout(
                    image,
                    vk::ImageSubresource {
                        aspect_mask,
                        mip_level,
                        array_layer: 0,
                    },
                )
            };
            validate_mip_subresource(mip_level, host, guest, placement)?;
        }
        Ok((image, plan))
    })();
    if result.is_err() {
        unsafe { ctx.device.destroy_image(image, None) };
    }
    result
}

/// The allocation length an aliasing image over this guest layout would need,
/// or the refusal that says no such image is representable on this device.
///
/// This is the same question [`create_allocation`] answers, asked without any
/// memory to bind: it plans an image in whichever linear mode the device
/// supports, reads the length that plan requires, and destroys the probe.
/// Nothing is retained, so a caller may ask before it owns an import.
///
/// Two arguments differ from the creating path and both are forced by having no
/// allocation yet. `require_allocation_fit` is `false` because discovering the
/// length the caller must grow to is the entire point — checking the length it
/// currently has would refuse every allocation that is merely short, which is
/// all of them before the growth this answer triggers. `parent_memory_type` is
/// `None` for the same reason: the memory whose type would be compared does not
/// exist yet, so that check belongs to creation and is made there.
pub(crate) unsafe fn binding_allocation_len(
    ctx: &DeviceContext,
    backing: GuestTargetBacking,
    allocation_layout: &reims_vgpu_memory::GuestImageAllocationLayout,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
) -> Result<u64, WindowRefusal> {
    if ctx.external_memory_host.is_none() {
        return Err(WindowRefusal::HostImportUnavailable);
    }
    let bytes_per_texel = crate::translate::pixel::texel_layout_of(format)
        .map(|layout| u64::from(layout.bytes_per_texel()))
        .ok_or(WindowRefusal::ResourceWindowTooShort)?;
    if !allocation_layout.is_vulkan_mip_chain(bytes_per_texel) {
        return Err(WindowRefusal::SampledContentRequiresCopy(
            SampledCopyCause::IrregularMipChain {
                mips: allocation_layout.mips.len(),
            },
        ));
    }
    let base = allocation_layout
        .base()
        .ok_or(WindowRefusal::ResourceWindowTooShort)?;
    // Admission must be exactly the shape the aliasing rail can build, and that
    // rail builds one `TYPE_2D` view. A volume, an array or a 1D texture plans
    // as a Vulkan image perfectly well and would be admitted here on its own
    // merits, but no `TYPE_2D` view describes it — so admitting one hands the
    // guest a declaration this device would then have to reinterpret, which is
    // the copying rail's work and not this rail's. A mip chain is not on that
    // list: `subresourceRange` names the levels and the view type does not.
    //
    // This is not a capability statement: the copying rail serves every one of
    // these shapes, and is the only rail at all on a host without the import.
    if let Some(cause) = SampledCopyCause::of_allocation(base.layout) {
        return Err(WindowRefusal::SampledContentRequiresCopy(cause));
    }
    let geometry = image_geometry(base.layout);
    // Mode selection mirrors `create_allocation` because a length planned in a
    // mode creation would not choose is not the length creation will need.
    let explicit = unsafe { explicit_linear_supported(ctx, format, usage, geometry.image_type) }?;
    let plan_in = |mode: LayoutMode| unsafe {
        plan_image_with_layout(
            ctx,
            backing,
            allocation_layout,
            format,
            usage,
            mode,
            None,
            false,
        )
        .map(|(image, plan)| {
            ctx.device.destroy_image(image, None);
            plan.required_allocation_len
        })
    };
    if !explicit {
        note_explicit_linear_unavailable(format, geometry.image_type, usage);
        return plan_in(LayoutMode::DriverLinear);
    }
    // Both modes are planned because either can carry the alias, but only the
    // explicit one declares the guest's own row pitch — the driver's linear
    // tiling reports whatever pitch it prefers, and a guest that padded its rows
    // disagrees with it by construction. So the explicit mode's refusal is the
    // one that says why the aliasing rail was not taken, and the fallback's is a
    // consequence of having fallen back.
    //
    // Only one refusal can return, so the explicit mode's is reported twice
    // over: through the same decline helper the creating path uses, which names
    // it against the declaration that produced it, and as the returned refusal
    // when the fallback refuses too.
    match plan_in(LayoutMode::ExplicitLinear) {
        Ok(len) => Ok(len),
        Err(explicit_refusal) => {
            note_explicit_linear_declined(
                &explicit_refusal,
                backing,
                allocation_layout,
                format,
                geometry.image_type,
                usage,
            );
            plan_in(LayoutMode::DriverLinear).map_err(|_| explicit_refusal)
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn create_with_layout(
    ctx: &DeviceContext,
    allocation: host_ram::ImportedHostRam,
    backing: GuestTargetBacking,
    allocation_layout: &reims_vgpu_memory::GuestImageAllocationLayout,
    format: vk::Format,
    usage: vk::ImageUsageFlags,
    mode: LayoutMode,
) -> Result<ImportedImage, WindowRefusal> {
    let (image, plan) = unsafe {
        plan_image_with_layout(
            ctx,
            backing,
            allocation_layout,
            format,
            usage,
            mode,
            Some(allocation.memory_type_index),
            true,
        )
    }?;
    match unsafe {
        ctx.device
            .bind_image_memory(image, allocation.memory, plan.bind_offset)
    } {
        Ok(()) => Ok(ImportedImage { image }),
        Err(result) => {
            unsafe { ctx.device.destroy_image(image, None) };
            Err(WindowRefusal::BindImage(result))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every layout the guest can declare, so the table below is total rather
    /// than a sample of the ones a boot happened to send.
    const EVERY_LAYOUT: [GuestImageLayout; 5] = [
        GuestImageLayout::D1 { width: 8 },
        GuestImageLayout::D1Array {
            width: 8,
            layers: 2,
            array_pitch: 32,
        },
        GuestImageLayout::D2 {
            width: 8,
            height: 8,
        },
        GuestImageLayout::D2Array {
            width: 8,
            height: 8,
            layers: 2,
            array_pitch: 256,
        },
        GuestImageLayout::D3 {
            width: 8,
            height: 8,
            depth: 2,
            depth_pitch: 256,
        },
    ];

    #[test]
    fn the_alias_rail_admits_a_two_d_allocation_at_any_mip_count() {
        // A mip count is not an allocation term. It was, and refusing on it
        // cost every mipmapped texture the guest owns its zero-copy bind.
        for layout in EVERY_LAYOUT {
            let admitted = SampledCopyCause::of_allocation(layout).is_none();
            assert_eq!(
                admitted,
                matches!(layout, GuestImageLayout::D2 { .. }),
                "{layout:?} was admitted={admitted}"
            );
        }
    }

    #[test]
    fn the_view_half_admits_exactly_the_allocation_s_whole_chain() {
        for mips in [1u32, 2, 9] {
            assert_eq!(
                SampledCopyCause::of_view(0, mips, mips, 0, 1),
                None,
                "a view naming all {mips} level(s) is the rail's own shape"
            );
        }
        assert_eq!(
            SampledCopyCause::of_view(2, 1, 4, 0, 1),
            Some(SampledCopyCause::ViewMip { base: 2, count: 1 })
        );
        assert_eq!(
            SampledCopyCause::of_view(0, 3, 4, 0, 1),
            Some(SampledCopyCause::ViewMip { base: 0, count: 3 }),
            "a prefix of the chain is still not the chain"
        );
        assert_eq!(
            SampledCopyCause::of_view(0, 1, 4, 5, 1),
            Some(SampledCopyCause::ViewMip { base: 0, count: 1 }),
            "level zero of a four-level allocation is a partial view, not a whole one"
        );
        assert_eq!(
            SampledCopyCause::of_view(0, 1, 1, 5, 1),
            Some(SampledCopyCause::ViewLayer { base: 5, count: 1 })
        );
        assert_eq!(
            SampledCopyCause::of_view(0, 1, 1, 0, 6),
            Some(SampledCopyCause::ViewLayer { base: 0, count: 6 })
        );
    }

    #[test]
    fn no_two_admission_terms_share_a_reason() {
        // The whole point of carrying a cause is that a census can rank the
        // terms against each other. Two terms sharing a slug would silently
        // merge two populations, which is the reading this replaced.
        let causes = [
            SampledCopyCause::Layout(EVERY_LAYOUT[0]),
            SampledCopyCause::Layout(EVERY_LAYOUT[1]),
            SampledCopyCause::Layout(EVERY_LAYOUT[2]),
            SampledCopyCause::Layout(EVERY_LAYOUT[3]),
            SampledCopyCause::Layout(EVERY_LAYOUT[4]),
            SampledCopyCause::IrregularMipChain { mips: 2 },
            SampledCopyCause::ViewMip { base: 1, count: 1 },
            SampledCopyCause::ViewLayer { base: 1, count: 1 },
            SampledCopyCause::ResidentUnavailable,
        ];
        let mut slugs: Vec<&str> = causes.iter().map(|cause| cause.slug()).collect();
        slugs.sort_unstable();
        let before = slugs.len();
        slugs.dedup();
        assert_eq!(before, slugs.len(), "two causes share a reason: {slugs:?}");
    }

    #[test]
    fn the_refusal_reports_the_cause_it_carries() {
        // The cause has to survive the trip through `Decline`, or the census
        // still reads one undifferentiated reason.
        assert_eq!(
            WindowRefusal::SampledContentRequiresCopy(SampledCopyCause::ResidentUnavailable).slug(),
            "sampled_copy_resident_unavailable"
        );
    }

    #[test]
    fn mutable_image_format_list_contains_exactly_the_compatible_view_family() {
        assert_eq!(
            mutable_view_formats(vk::Format::B8G8R8A8_UNORM),
            vec![vk::Format::B8G8R8A8_UNORM, vk::Format::B8G8R8A8_SRGB]
        );
        assert_eq!(
            mutable_view_formats(vk::Format::R8G8B8A8_SRGB),
            vec![vk::Format::R8G8B8A8_UNORM, vk::Format::R8G8B8A8_SRGB]
        );
        assert_eq!(
            mutable_view_formats(vk::Format::R16G16B16A16_SFLOAT),
            vec![vk::Format::R16G16B16A16_SFLOAT]
        );
    }

    fn backing() -> GuestTargetBacking {
        GuestTargetBacking {
            allocation_host_ptr: 0x1000,
            allocation_len: 0x4000,
            resource_offset: 0x1000,
            resource_len: 0x3000,
            plane_offset: 0x1000,
            row_pitch: 256,
        }
    }

    #[test]
    fn explicit_decline_dedupes_equal_layout_contracts_across_allocations() {
        let first = backing();
        let second = GuestTargetBacking {
            allocation_host_ptr: 0x9000,
            allocation_len: 0x8000,
            ..first
        };
        let layout = reims_vgpu_memory::GuestImageAllocationLayout::single(
            first.plane_offset - first.resource_offset,
            first.row_pitch,
            d2(),
        );
        let key = |backing| {
            explicit_linear_decline_key(
                &WindowRefusal::RowPitchMismatch {
                    mip_level: 0,
                    width: 64,
                    guest: 256,
                    host: 512,
                },
                backing,
                &layout,
                vk::Format::R8_UNORM,
                vk::ImageType::TYPE_2D,
                vk::ImageUsageFlags::SAMPLED,
            )
        };
        assert_eq!(key(first), key(second));

        let different_pitch = GuestTargetBacking {
            row_pitch: 512,
            ..first
        };
        assert_ne!(key(first), key(different_pitch));
    }

    fn requirements() -> vk::MemoryRequirements {
        vk::MemoryRequirements {
            size: 0x2000,
            alignment: 0x1000,
            memory_type_bits: 0b10,
        }
    }

    fn d2() -> GuestImageLayout {
        GuestImageLayout::D2 {
            width: 16,
            height: 16,
        }
    }

    #[test]
    fn exact_driver_layout_derives_the_parent_binding_offset() {
        assert_eq!(
            plan_linear_window(
                vk::SubresourceLayout {
                    offset: 0,
                    row_pitch: 256,
                    ..Default::default()
                },
                requirements(),
                backing(),
                d2(),
                Some(1),
                false,
                true,
            ),
            Ok(WindowPlan {
                bind_offset: 0x1000,
                required_allocation_len: 0x3000,
            })
        );
    }

    #[test]
    fn every_binding_term_can_refuse() {
        let mut b = backing();
        let req = requirements();
        assert_eq!(
            plan_linear_window(
                vk::SubresourceLayout {
                    offset: 0x2000,
                    row_pitch: 256,
                    ..Default::default()
                },
                req,
                b,
                d2(),
                Some(1),
                false,
                true,
            ),
            Err(WindowRefusal::SubresourceAfterPlane {
                plane_offset: 0x1000,
                subresource_offset: 0x2000,
            })
        );
        b.plane_offset = 0x800;
        assert_eq!(
            plan_linear_window(
                vk::SubresourceLayout {
                    row_pitch: 256,
                    ..Default::default()
                },
                req,
                b,
                d2(),
                Some(1),
                false,
                true,
            ),
            Err(WindowRefusal::BindOffsetMisaligned)
        );
        b = backing();
        assert_eq!(
            plan_linear_window(
                vk::SubresourceLayout {
                    row_pitch: 512,
                    ..Default::default()
                },
                req,
                b,
                d2(),
                Some(1),
                false,
                true,
            ),
            Err(WindowRefusal::RowPitchMismatch {
                mip_level: 0,
                width: 16,
                guest: 256,
                host: 512,
            })
        );
        b.allocation_len = 0x2000;
        assert_eq!(
            plan_linear_window(
                vk::SubresourceLayout {
                    row_pitch: 256,
                    ..Default::default()
                },
                req,
                b,
                d2(),
                Some(1),
                false,
                true,
            ),
            Err(WindowRefusal::AllocationTooShort {
                required_end: 0x3000,
                allocation_len: 0x2000,
            })
        );
        b = backing();
        assert_eq!(
            plan_linear_window(
                vk::SubresourceLayout {
                    row_pitch: 256,
                    ..Default::default()
                },
                req,
                b,
                d2(),
                Some(2),
                false,
                true,
            ),
            Err(WindowRefusal::NoMemoryType)
        );
        assert_eq!(
            plan_linear_window(
                vk::SubresourceLayout {
                    row_pitch: 256,
                    ..Default::default()
                },
                req,
                b,
                d2(),
                Some(1),
                true,
                true,
            ),
            Err(WindowRefusal::DedicatedBindingRequired)
        );
    }

    #[test]
    fn planning_reports_the_required_tail_without_claiming_guest_bytes() {
        let mut b = backing();
        b.allocation_len = 0x2000;
        assert_eq!(
            plan_linear_window(
                vk::SubresourceLayout {
                    row_pitch: 256,
                    ..Default::default()
                },
                requirements(),
                b,
                d2(),
                None,
                false,
                false,
            ),
            Ok(WindowPlan {
                bind_offset: 0x1000,
                required_allocation_len: 0x3000,
            })
        );
        assert_eq!(
            b.allocation_len, 0x2000,
            "the guest allocation is unchanged"
        );
    }

    #[test]
    fn a_declared_base_subresource_offset_absorbs_the_whole_plane_offset() {
        assert_eq!(
            plan_linear_window(
                vk::SubresourceLayout {
                    offset: 0x1000,
                    row_pitch: 256,
                    ..Default::default()
                },
                requirements(),
                backing(),
                d2(),
                Some(1),
                false,
                true,
            ),
            Ok(WindowPlan {
                bind_offset: 0,
                required_allocation_len: 0x2000,
            })
        );
    }

    #[test]
    fn array_and_volume_pitch_are_independent_binding_terms() {
        let array = GuestImageLayout::D1Array {
            width: 16,
            layers: 3,
            array_pitch: 256,
        };
        assert_eq!(
            plan_linear_window(
                vk::SubresourceLayout {
                    row_pitch: 256,
                    array_pitch: 512,
                    ..Default::default()
                },
                requirements(),
                backing(),
                array,
                Some(1),
                false,
                true,
            ),
            Err(WindowRefusal::ArrayPitchMismatch)
        );

        let volume = GuestImageLayout::D3 {
            width: 16,
            height: 2,
            depth: 3,
            depth_pitch: 512,
        };
        assert_eq!(
            plan_linear_window(
                vk::SubresourceLayout {
                    row_pitch: 256,
                    depth_pitch: 768,
                    ..Default::default()
                },
                requirements(),
                backing(),
                volume,
                Some(1),
                false,
                true,
            ),
            Err(WindowRefusal::DepthPitchMismatch)
        );
    }

    #[test]
    fn image_geometry_preserves_one_and_three_dimensional_types() {
        let d1 = image_geometry(GuestImageLayout::D1 { width: 17 });
        assert_eq!(d1.image_type, vk::ImageType::TYPE_1D);
        assert_eq!(
            d1.extent,
            vk::Extent3D::default().width(17).height(1).depth(1)
        );
        assert_eq!(d1.array_layers, 1);

        let d3 = image_geometry(GuestImageLayout::D3 {
            width: 7,
            height: 5,
            depth: 3,
            depth_pitch: 256,
        });
        assert_eq!(d3.image_type, vk::ImageType::TYPE_3D);
        assert_eq!(
            d3.extent,
            vk::Extent3D::default().width(7).height(5).depth(3)
        );
        assert_eq!(d3.array_layers, 1);
    }

    #[test]
    fn every_mip_offset_is_checked_in_the_guest_resource_namespace() {
        let guest = reims_vgpu_memory::GuestImageMipLayout {
            resource_relative_offset: 0x240,
            row_pitch: 64,
            layout: GuestImageLayout::D1 { width: 16 },
        };
        let host = vk::SubresourceLayout {
            offset: 0x240,
            row_pitch: 64,
            ..Default::default()
        };
        let at_allocation_base = MipPlacement {
            image_base: 0,
            resource_base: 0,
            plane_offset: 0,
            mode: LayoutMode::DriverLinear,
        };
        assert_eq!(
            validate_mip_subresource(1, host, guest, at_allocation_base),
            Ok(())
        );
        assert_eq!(
            validate_mip_subresource(
                1,
                host,
                reims_vgpu_memory::GuestImageMipLayout {
                    resource_relative_offset: 0x280,
                    ..guest
                },
                at_allocation_base,
            ),
            Err(WindowRefusal::MipOffsetMismatch {
                mip_level: 1,
                guest_offset: 0x280,
                host_offset: 0x240,
                image_base: 0,
                resource_base: 0,
                plane_offset: 0,
                mode: LayoutMode::DriverLinear,
            })
        );
    }

    /// A guest resource does not begin at byte zero of the imported allocation,
    /// and its mip offsets are not counted from there.
    ///
    /// The allocation is a whole RAMBlock, so a resource's start runs to
    /// hundreds of megabytes while the mip offsets inside it stay small. An
    /// equation that reaches the guest offset by adding the bind offset to the
    /// host offset and stopping is short the resource base, and it fails in the
    /// direction that hides: on a fixture whose resource happens to start at
    /// zero the two agree, and on every real one the comparison is a small
    /// number against a large one.
    #[test]
    fn a_mip_offset_is_read_past_the_resource_base_not_the_allocation_base() {
        let resource_base = 0x2000_0000;
        let plane_offset = resource_base + 0x1000;
        let guest = reims_vgpu_memory::GuestImageMipLayout {
            // What `GuestImageSource::single_mip` builds: plane, less resource.
            resource_relative_offset: plane_offset - resource_base,
            row_pitch: 64,
            layout: GuestImageLayout::D1 { width: 16 },
        };
        // The driver places the base subresource at the start of the image, so
        // the bind offset carries the plane's whole distance into the RAMBlock.
        let host = vk::SubresourceLayout {
            offset: 0,
            row_pitch: 64,
            ..Default::default()
        };
        assert_eq!(
            validate_mip_subresource(
                0,
                host,
                guest,
                MipPlacement {
                    image_base: plane_offset,
                    resource_base,
                    plane_offset,
                    mode: LayoutMode::ExplicitLinear,
                },
            ),
            Ok(()),
            "the plane the guest named is where the image's base subresource lands"
        );
    }
}
