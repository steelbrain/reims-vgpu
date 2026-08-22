//! Semantic resource namespace entries.

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::{ObjectTableRef, ResourceObject, TaskId};

/// Number of color attachments carried by one render pass and pipeline.
///
/// Derived from the wire array width so pass decoding, pipeline decoding, and
/// backend allocation cannot acquire independent bounds.
pub const MAX_COLOR_ATTACHMENTS: usize =
    reims_vgpu_wire::ops::render_pass::RENDER_PASS_COLOR_ATTACHMENTS;

// `MTLColorWriteMask` bits, in Metal's alpha-first ordering.
pub const MTL_COLOR_WRITE_MASK_NONE: u32 = 0;
pub const MTL_COLOR_WRITE_MASK_ALPHA: u32 = 1 << 0;
pub const MTL_COLOR_WRITE_MASK_BLUE: u32 = 1 << 1;
pub const MTL_COLOR_WRITE_MASK_GREEN: u32 = 1 << 2;
pub const MTL_COLOR_WRITE_MASK_RED: u32 = 1 << 3;
pub const MTL_COLOR_WRITE_MASK_ALL: u32 = 0xf;

/// Channels written by one render-pipeline color attachment.
///
/// Default is `all`, matching Metal descriptor semantics; a derived zero
/// default would instead suppress every color write.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorWriteMask {
    bits: u32,
}

impl Default for ColorWriteMask {
    fn default() -> Self {
        Self {
            bits: MTL_COLOR_WRITE_MASK_ALL,
        }
    }
}

impl ColorWriteMask {
    pub fn new(bits: u32) -> Option<Self> {
        (bits <= MTL_COLOR_WRITE_MASK_ALL).then_some(Self { bits })
    }

    pub const fn bits(self) -> u32 {
        self.bits
    }
}

/// One semantic color-attachment entry in a render-pipeline descriptor.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineColorAttachment {
    pub slot: u32,
    pub has_pixel_format: bool,
    pub pixel_format: u32,
    pub blending_enabled: bool,
    pub src_rgb: u32,
    pub dst_rgb: u32,
    pub op_rgb: u32,
    pub src_alpha: u32,
    pub dst_alpha: u32,
    pub op_alpha: u32,
    /// Channels written independently of whether blending is enabled.
    pub write_mask: ColorWriteMask,
}

/// One vertex attribute and its buffer-layout state from a render pipeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VertexAttribute {
    pub location: u32,
    pub format: u32,
    pub offset: u32,
    pub buffer_index: u32,
    pub stride: u32,
    /// `MTLVertexStepFunction` ordinal stated by the layout, when present.
    pub declared_step_function: Option<u32>,
    /// `MTLVertexBufferLayoutDescriptor.stepRate` stated by the layout.
    pub declared_step_rate: Option<u32>,
}

impl VertexAttribute {
    /// Metal defaults an omitted step rate to one; a stated zero remains zero.
    pub fn step_rate(&self) -> u32 {
        self.declared_step_rate.unwrap_or(1)
    }

    /// Return the stated step function or the context-specific Metal default.
    pub fn step_function_ordinal(&self, when_absent: u32) -> u32 {
        self.declared_step_function.unwrap_or(when_absent)
    }
}

/// Immutable semantic declaration of a render-pipeline object.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderPipelineDescriptor {
    pub object_id: u32,
    pub serialized_payload_len: u32,
    pub vertex_func_ref: u32,
    pub fragment_func_ref: u32,
    pub object_func_ref: u32,
    pub mesh_func_ref: u32,
    pub color_attachment_offset: u32,
    pub has_color_attachment_offset: bool,
    pub vertex_descriptor_offset: u32,
    pub has_vertex_descriptor_offset: bool,
    pub raster_sample_count: u32,
    pub max_tessellation_factor: u32,
    pub tessellation_factor_step_function: u32,
    pub tessellation_output_winding_order: u32,
    pub vertex_attributes: Vec<VertexAttribute>,
    pub color0: PipelineColorAttachment,
    pub color_attachments: Vec<PipelineColorAttachment>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComputeStageInputAttribute {
    pub raw_bits: u32,
    pub location: u32,
    pub format: u32,
    pub offset: u32,
    pub buffer_index: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComputeStageInputLayout {
    pub raw_bits: u32,
    pub buffer_index: u32,
    pub step_function: u32,
    pub step_rate: u32,
    pub stride: u64,
}

/// Optional stage-input declaration retained by a compute pipeline.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ComputeStageInputDescriptor {
    pub word0: u32,
    pub header0: u32,
    pub header1: u32,
    pub index_type: u32,
    pub index_buffer_index: u32,
    pub attributes: Vec<ComputeStageInputAttribute>,
    pub layouts: Vec<ComputeStageInputLayout>,
    /// Entries the boundary could not retain. A nonzero value refuses creation.
    pub dropped_attributes: u32,
    /// Layout entries the boundary could not retain. A nonzero value refuses creation.
    pub dropped_layouts: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ComputePipelineDescriptor {
    pub kernel_func_ref: u32,
    pub stage_input: Option<ComputeStageInputDescriptor>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FunctionDescriptor {
    pub blob_gva: u64,
    pub blob_size: u32,
    pub function_id: u32,
}

/// Immutable semantic sampler declaration.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SamplerDescriptor {
    pub min_filter: u32,
    pub mag_filter: u32,
    pub mip_filter: u32,
    pub s_address: u32,
    pub t_address: u32,
    pub r_address: u32,
    pub max_anisotropy: u32,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub compare_function: u32,
    pub border_color: u32,
    pub normalized_coordinates: bool,
    pub support_argument_buffers: bool,
    pub lod_average: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DepthStencilFace {
    pub compare_function: u32,
    pub stencil_failure_operation: u32,
    pub depth_failure_operation: u32,
    pub depth_stencil_pass_operation: u32,
    pub read_mask: u32,
    pub write_mask: u32,
}

/// Immutable semantic depth/stencil declaration.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DepthStencilDescriptor {
    pub depth_stencil_id: u32,
    pub depth_compare_function: u32,
    pub depth_write_enabled: bool,
    pub front_stencil_present: bool,
    pub back_stencil_present: bool,
    pub front_face: DepthStencilFace,
    pub back_face: DepthStencilFace,
}

/// Semantic construction descriptor for a linear buffer resource.
///
/// The wire retains both the full handle word and the low-width page handle.
/// Address construction always takes page geometry explicitly so neither guest
/// architecture can leak a fixed shift into portable code.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BufferDescriptor {
    pub allocation_size: u64,
    pub handle64: u64,
    pub handle: u32,
}

impl BufferDescriptor {
    /// Guest VA and allocation size, or `None` when the declaration cannot
    /// identify a non-empty page-backed allocation.
    pub fn backing_gva_size(&self, page_shift: u32) -> Option<(u64, u64)> {
        if self.allocation_size == 0 || self.handle == 0 || page_shift == 0 || page_shift > 30 {
            return None;
        }
        Some(((self.handle as u64) << page_shift, self.allocation_size))
    }
}

/// Semantic form of a texture view after its serializer opcode is consumed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TextureViewForm {
    #[default]
    Simple,
    Ranged,
    Swizzled,
}

/// A task-local texture view over an existing texture resource.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TextureViewDescriptor {
    pub form: TextureViewForm,
    pub view_texture_ref: u32,
    pub base_texture_ref: u32,
    pub pixel_format: u16,
    pub texture_type: u16,
    pub level_base: u64,
    pub level_count: u64,
    pub slice_base: u64,
    pub slice_count: u64,
    pub swizzle: [u8; 4],
}

impl TextureViewDescriptor {
    pub const fn carries_range(&self) -> bool {
        matches!(
            self.form,
            TextureViewForm::Ranged | TextureViewForm::Swizzled
        )
    }

    pub const fn carries_swizzle(&self) -> bool {
        matches!(self.form, TextureViewForm::Swizzled)
    }

    pub const fn declared_pixel_format(&self) -> Option<u16> {
        if self.pixel_format == 0 {
            None
        } else {
            Some(self.pixel_format)
        }
    }
}

/// Layout of one mip level within a linear texture allocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureLevelLayout {
    pub offset: u64,
    pub size: u64,
    pub row_stride: u64,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl TextureLevelLayout {
    /// Bytes touched by a row walk, excluding padding after the final row.
    pub fn read_span(&self, tight_row: u32) -> Option<u64> {
        self.height
            .checked_sub(1)
            .map(u64::from)?
            .checked_mul(self.row_stride)?
            .checked_add(u64::from(tight_row))
    }

    /// Number of contiguous depth planes; zero is the protocol's 2D spelling.
    pub fn planes(&self) -> u32 {
        self.depth.max(1)
    }

    pub fn slice_stride(&self) -> Option<u64> {
        self.row_stride
            .checked_mul(u64::from(self.height))?
            .checked_mul(u64::from(self.planes()))
    }

    pub fn slice_read_span(&self, tight_row: u32) -> Option<u64> {
        u64::from(self.planes() - 1)
            .checked_mul(self.row_stride)?
            .checked_mul(u64::from(self.height))?
            .checked_add(self.read_span(tight_row)?)
    }
}

/// Faces in a cube texture, which is a property of the type rather than a
/// count anything reports.
///
/// A cube always has exactly six slices, one per face, and a cube array holds
/// consecutive groups of six. Both APIs meeting in this device agree on the
/// order those six sit in — `+X, -X, +Y, -Y, +Z, -Z` — so a face needs no
/// permutation and a cube needs no layout distinct from a six-slice array.
///
/// Every place that would otherwise spell `6` refers here, because a second
/// spelling is the only way the two can disagree.
pub const CUBE_FACES: u32 = 6;

/// Semantic construction descriptor for a page-backed linear texture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LinearTextureDescriptor {
    pub allocation_size: u64,
    pub handle: u32,
    pub mipmap_level_count: u32,
    /// Allocation-relative start of the first texture slice.
    pub base_offset: u64,
    /// Distance between adjacent physical slices (array layers or cube
    /// faces), spanning every mip level belonging to one slice.
    ///
    /// This is an inter-slice advance, not a mandatory size field. A
    /// single-slice client-backed texture may carry zero; its occupied span is
    /// then described entirely by the level records and allocation size.
    pub bytes_per_slice: u64,
    /// Number of declared array slices before cube faces are expanded.
    pub slice_count: u32,
    /// The dimension record says its slices are six-face cube slices.
    pub cube_faces: bool,
    /// The dimension records describe compressed blocks rather than ordinary
    /// texels.
    pub compressed_layout: bool,
    pub bytes_per_element: u8,
    pub used_size: u32,
    pub row_stride: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub declaration: Option<crate::TextureDeclaration>,
    pub levels: Vec<TextureLevelLayout>,
}

impl Default for LinearTextureDescriptor {
    fn default() -> Self {
        Self {
            allocation_size: 0,
            handle: 0,
            mipmap_level_count: 0,
            base_offset: 0,
            bytes_per_slice: 0,
            slice_count: 1,
            cube_faces: false,
            compressed_layout: false,
            bytes_per_element: 0,
            used_size: 0,
            row_stride: 0,
            width: 0,
            height: 0,
            depth: 0,
            declaration: None,
            levels: Vec::new(),
        }
    }
}

impl LinearTextureDescriptor {
    pub fn extent(&self) -> Option<(u32, u32)> {
        (self.width != 0 && self.height != 0).then_some((self.width, self.height))
    }

    pub fn declared_row_stride(&self) -> Option<u32> {
        (self.row_stride != 0).then_some(self.row_stride)
    }

    pub fn declared_pixel_format(&self) -> Option<u16> {
        self.declaration
            .and_then(|declaration| declaration.declared_pixel_format())
    }

    pub fn declared_usage(&self) -> Option<u32> {
        self.declaration.map(|declaration| declaration.usage)
    }

    /// Whether the guest is obliged to announce its CPU writes to this texture.
    ///
    /// Fails closed: a descriptor whose declaration this device never decoded
    /// carries no obligation it can rely on, so an absent declaration is
    /// [`crate::GuestWriteAnnouncement::Silent`] rather than an assumed mode.
    /// See [`crate::StorageMode`] for why only `Managed` announces.
    pub fn guest_write_announcement(&self) -> crate::GuestWriteAnnouncement {
        self.declaration
            .map_or(crate::GuestWriteAnnouncement::Silent, |declaration| {
                declaration.announcement()
            })
    }

    pub fn allocation_base_gva(&self, page_shift: u32) -> Option<u64> {
        if self.handle == 0 || page_shift == 0 || page_shift > 30 {
            return None;
        }
        Some((self.handle as u64) << page_shift)
    }

    pub fn backing_gva_size(&self, page_shift: u32) -> Option<(u64, u64)> {
        if self.allocation_size == 0 || self.handle == 0 || self.base_offset >= self.allocation_size
        {
            return None;
        }
        let base = self.allocation_base_gva(page_shift)?;
        Some((
            base.checked_add(self.base_offset)?,
            self.allocation_size.checked_sub(self.base_offset)?,
        ))
    }

    /// Number of physical slices represented by the allocation. Cube and
    /// cube-array textures store six consecutive faces per declared slice.
    pub fn physical_slice_count(&self) -> Option<u32> {
        self.slice_count
            .checked_mul(if self.cube_faces { CUBE_FACES } else { 1 })
    }

    /// Span occupied by one physical slice's mip records.
    ///
    /// Multi-slice storage requires the declared inter-slice advance. For one
    /// physical slice, zero means there is no next-slice advance; the level
    /// records still provide a complete occupied span.
    pub fn physical_slice_span(&self) -> Option<u64> {
        let physical_slices = self.physical_slice_count()?;
        if physical_slices == 0 {
            return None;
        }
        if self.bytes_per_slice != 0 {
            return Some(self.bytes_per_slice);
        }
        if physical_slices != 1 {
            return None;
        }
        let mut level_end = None;
        for level in &self.levels {
            if level.size == 0 {
                return None;
            }
            let end = level.offset.checked_add(level.size)?;
            level_end = Some(level_end.map_or(end, |current: u64| current.max(end)));
        }
        let level_end = level_end?;
        Some(level_end)
    }

    /// Allocation-relative end of the declared slice-major mip packing.
    pub fn packed_allocation_end(&self) -> Option<u64> {
        let physical_slices = self.physical_slice_count()?;
        let slice_span = self.physical_slice_span()?;
        self.base_offset
            .checked_add(slice_span.checked_mul(u64::from(physical_slices))?)
    }

    pub fn declared_packing_fits_allocation(&self) -> bool {
        self.packed_allocation_end()
            .is_some_and(|end| end <= self.allocation_size)
    }

    pub fn level_fits_slice(&self, level: &TextureLevelLayout) -> bool {
        level.size != 0
            && level
                .offset
                .checked_add(level.size)
                .zip(self.physical_slice_span())
                .is_some_and(|(end, span)| end <= span)
    }

    /// Allocation-relative start of one `(slice, mip)` subresource.
    pub fn subresource_offset(&self, slice: u32, level: u32) -> Option<u64> {
        let physical_slices = self.physical_slice_count()?;
        if slice >= physical_slices {
            return None;
        }
        let slice_stride = if physical_slices == 1 {
            0
        } else {
            (self.bytes_per_slice != 0).then_some(self.bytes_per_slice)?
        };
        u64::from(slice)
            .checked_mul(slice_stride)?
            .checked_add(self.base_offset)?
            .checked_add(self.level(level)?.offset)
    }

    pub fn level(&self, level: u32) -> Option<&TextureLevelLayout> {
        self.levels.get(level as usize)
    }

    pub fn level_gva(&self, level: u32, page_shift: u32) -> Option<(u64, &TextureLevelLayout)> {
        let base = self.allocation_base_gva(page_shift)?;
        let layout = self.level(level)?;
        if layout.width == 0 || layout.height == 0 || layout.row_stride == 0 {
            return None;
        }
        let offset = self.subresource_offset(0, level)?;
        if self.allocation_size != 0
            && (offset >= self.allocation_size || self.allocation_size - offset < layout.row_stride)
        {
            return None;
        }
        Some((base.checked_add(offset)?, layout))
    }

    pub fn subresource_gva(
        &self,
        slice: u32,
        level: u32,
        page_shift: u32,
    ) -> Option<(u64, &TextureLevelLayout)> {
        let base = self.allocation_base_gva(page_shift)?;
        let layout = self.level(level)?;
        if layout.width == 0 || layout.height == 0 || layout.row_stride == 0 {
            return None;
        }
        let offset = self.subresource_offset(slice, level)?;
        if self.allocation_size != 0
            && (offset >= self.allocation_size || self.allocation_size - offset < layout.row_stride)
        {
            return None;
        }
        Some((base.checked_add(offset)?, layout))
    }
}

/// Per-command-slot layout inside an indirect-command-buffer allocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IcbCommandLayout {
    pub command_type_offset: u16,
    pub barrier_offset: u16,
    pub kernel_dispatch_arguments_offset: u16,
    pub tessellation_factor_offset: u16,
    pub pipeline_state_offset: u32,
    pub vertex_buffer_bind_offset: u32,
    pub fragment_buffer_bind_offset: u32,
    pub object_buffer_bind_offset: u32,
    pub mesh_buffer_bind_offset: u32,
    pub kernel_buffer_bind_offset: u32,
    pub attribute_stride_offset: u32,
    pub object_threadgroup_memory_length_offset: u32,
    pub threadgroup_memory_length_offset: u32,
    pub command_arguments_offset: u32,
    pub command_size: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IndirectCommandBufferDescriptor {
    pub command_types: u32,
    pub max_vertex_buffer_bind_count: u16,
    pub max_fragment_buffer_bind_count: u16,
    pub max_kernel_buffer_bind_count: u16,
    pub max_object_buffer_bind_count: u16,
    pub max_mesh_buffer_bind_count: u16,
    pub max_kernel_threadgroup_memory_bind_count: u16,
    pub max_object_threadgroup_memory_bind_count: u16,
    pub flags: u16,
    pub max_command_count: u32,
    pub options: u16,
    pub layout: IcbCommandLayout,
}

/// Guest ICB command-memory association used for CPU-side command decoding.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IcbCommandMemory {
    pub gva: u64,
    pub byte_len: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IcbUnappliedFlag {
    SupportRayTracing,
    SupportDynamicAttributeStride,
    InheritDepthStencilState,
    InheritDepthBias,
    InheritDepthClipMode,
    InheritCullMode,
    InheritFrontFacingWinding,
    InheritTriangleFillMode,
}

impl IcbUnappliedFlag {
    pub const fn slug(self) -> &'static str {
        match self {
            Self::SupportRayTracing => "icb_flag_support_ray_tracing_dropped",
            Self::SupportDynamicAttributeStride => "icb_flag_dynamic_attribute_stride_dropped",
            Self::InheritDepthStencilState => "icb_flag_no_inherit_depth_stencil_dropped",
            Self::InheritDepthBias => "icb_flag_no_inherit_depth_bias_dropped",
            Self::InheritDepthClipMode => "icb_flag_no_inherit_depth_clip_dropped",
            Self::InheritCullMode => "icb_flag_no_inherit_cull_mode_dropped",
            Self::InheritFrontFacingWinding => "icb_flag_no_inherit_winding_dropped",
            Self::InheritTriangleFillMode => "icb_flag_no_inherit_fill_mode_dropped",
        }
    }
}

impl IndirectCommandBufferDescriptor {
    pub const fn inherit_pipeline_state(&self) -> bool {
        self.flags & (1 << 0) != 0
    }

    pub const fn inherit_buffers(&self) -> bool {
        self.flags & (1 << 1) != 0
    }

    pub const fn unidentified_flags(&self) -> u16 {
        self.flags & ((1 << 6) | (1 << 11) | (1 << 12) | (1 << 13) | (1 << 14))
    }

    pub fn unapplied_flags(&self) -> Vec<IcbUnappliedFlag> {
        let mut out = Vec::new();
        for (bit, flag) in [
            (1 << 2, IcbUnappliedFlag::SupportRayTracing),
            (1 << 3, IcbUnappliedFlag::SupportDynamicAttributeStride),
        ] {
            if self.flags & bit != 0 {
                out.push(flag);
            }
        }
        for (bit, flag) in [
            (1 << 4, IcbUnappliedFlag::InheritDepthStencilState),
            (1 << 5, IcbUnappliedFlag::InheritDepthBias),
            (1 << 7, IcbUnappliedFlag::InheritDepthClipMode),
            (1 << 8, IcbUnappliedFlag::InheritCullMode),
            (1 << 9, IcbUnappliedFlag::InheritFrontFacingWinding),
            (1 << 10, IcbUnappliedFlag::InheritTriangleFillMode),
        ] {
            if self.flags & bit == 0 {
                out.push(flag);
            }
        }
        out
    }
}

/// Semantic result of decoding one object-list construction descriptor.
#[derive(Clone, Debug, PartialEq)]
pub enum ResourceDescriptor {
    Buffer(BufferDescriptor),
    Texture(LinearTextureDescriptor),
    SurfaceBacking(SurfaceBackingDescriptor),
    Sampler(SamplerDescriptor),
    Function(FunctionDescriptor),
    RenderPipeline(RenderPipelineDescriptor),
    ComputePipeline(ComputePipelineDescriptor),
    DepthStencil(DepthStencilDescriptor),
    TextureView(TextureViewDescriptor),
    BufferTexture(BufferTextureDescriptor),
    HeapTexture(crate::HeapTextureDescriptor),
    IOSurfacePlaneView(IOSurfacePlaneViewResourceDescriptor),
    MapperIOSurfaceTextureView(crate::MapperIOSurfaceTextureView),
    IndirectCommandBuffer(IndirectCommandBufferDescriptor),
}

/// One plane declared by a registered IOSurface backing object.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SurfaceBackingPlane {
    pub offset: u32,
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
    pub bytes_per_element: u8,
}

/// Complete semantic construction descriptor for a registered IOSurface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SurfaceBackingDescriptor {
    pub length: u64,
    pub backing_pfn: u32,
    /// IOSurface pixel-format word: an OSType FourCC or a pathway-produced
    /// Metal format ordinal, interpreted only by a format adapter.
    pub pixel_format: u32,
    pub plane_count: u8,
    pub planes: [SurfaceBackingPlane; reims_vgpu_wire::device_desc::SURFACE_BACKING_PLANE_CAP],
    /// Plane-zero conveniences used by the single-plane pathway.
    pub width: u32,
    pub height: u32,
    pub bytes_per_row: u32,
}

/// Decode one complete registered-surface construction descriptor.
pub fn decode_surface_backing_descriptor(
    bytes: &[u8],
) -> Result<SurfaceBackingDescriptor, ResourceDecodeError> {
    use reims_vgpu_wire::device_desc as wire;

    let header = wire::surface_backing_header(bytes)
        .map_err(|_| ResourceDecodeError::ErrShort("res_surface_backing_header"))?;
    let length = header.length.get();
    let backing_pfn = header.backing_pfn.get();
    if length == 0 || backing_pfn == 0 {
        return Err(ResourceDecodeError::ErrUnsupported(
            "res_surface_backing_empty",
        ));
    }
    let plane_count = header.plane_count;
    if usize::from(plane_count) > wire::SURFACE_BACKING_PLANE_CAP {
        return Err(ResourceDecodeError::ErrUnsupported(
            "res_surface_backing_plane_count",
        ));
    }
    let mut planes = [SurfaceBackingPlane::default(); wire::SURFACE_BACKING_PLANE_CAP];
    for (index, plane) in planes.iter_mut().enumerate().take(usize::from(plane_count)) {
        let record = wire::surface_backing_plane(bytes, index)
            .map_err(|_| ResourceDecodeError::ErrShort("res_surface_backing_plane"))?;
        *plane = SurfaceBackingPlane {
            offset: record.offset.get(),
            width: record.width.get(),
            height: record.height.get(),
            bytes_per_row: record.bytes_per_row(),
            bytes_per_element: record.bytes_per_element(),
        };
    }
    let plane_zero = planes.first().copied().unwrap_or_default();
    let (width, height, bytes_per_row) = if plane_count == 0 {
        (0, 0, 0)
    } else {
        (
            plane_zero.width,
            plane_zero.height,
            plane_zero.bytes_per_row,
        )
    };
    Ok(SurfaceBackingDescriptor {
        length,
        backing_pfn,
        pixel_format: header.pixel_format.get(),
        plane_count,
        planes,
        width,
        height,
        bytes_per_row,
    })
}

/// Semantic texture geometry carried by a registered IOSurface plane view.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IOSurfacePlaneViewDescriptor {
    pub pixel_format: u16,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub plane_index: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IOSurfacePlaneViewRecordKind {
    Plane,
    ColorView,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IOSurfacePlaneViewDecodeState {
    Complete,
    MissingOperation,
    MissingRecord,
    UnknownRecordTag(u8),
    InvalidGeometry,
}

/// Complete semantic form of a wire tag 5 registered-surface view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IOSurfacePlaneViewResourceDescriptor {
    pub surface: ObjectTableRef<ResourceObject>,
    pub owner_task: TaskId,
    pub operation_kind: Option<u32>,
    pub operation_length: Option<u32>,
    pub own_ref: Option<ObjectTableRef<ResourceObject>>,
    pub record_kind: Option<IOSurfacePlaneViewRecordKind>,
    /// Producer-carried byte whose public meaning is not established.
    pub unidentified_record_flags: u8,
    /// Valid 2D view geometry when the nested record carries one. The surface
    /// relation remains meaningful when construction has not populated a usable
    /// texture view yet.
    pub view: Option<IOSurfacePlaneViewDescriptor>,
    pub decode_state: IOSurfacePlaneViewDecodeState,
}

/// A texture view placed over an existing buffer's storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferTextureDescriptor {
    pub new_texture_ref: u32,
    pub buffer_ref: u32,
    pub offset: u64,
    pub bytes_per_row: u64,
    pub desc: crate::TextureDeclaration,
}

/// Typed refusal produced while decoding a resource construction descriptor.
///
/// The slug is retained at the protocol boundary because it names the exact
/// failed contract check. Logging adapters may attach the coarse class without
/// owning or reclassifying the semantic error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceDecodeError {
    ErrShort(&'static str),
    ErrUnknownType(&'static str),
    ErrUnsupported(&'static str),
}

impl ResourceDecodeError {
    pub const fn slug(self) -> &'static str {
        match self {
            Self::ErrShort(slug) | Self::ErrUnknownType(slug) | Self::ErrUnsupported(slug) => slug,
        }
    }

    pub const fn class(self) -> &'static str {
        match self {
            Self::ErrShort(_) => "short",
            Self::ErrUnknownType(_) => "unknown_type",
            Self::ErrUnsupported(_) => "unsupported",
        }
    }

    pub fn fields(self) -> Vec<(&'static str, String)> {
        vec![("class", self.class().to_string())]
    }
}

/// Marker for the sampler API's independent reference namespace.
pub enum SamplerObject {}
/// Marker for the depth-stencil API's independent reference namespace.
pub enum DepthStencilObject {}
/// Marker for the render-pipeline API's independent reference namespace.
pub enum RenderPipelineObject {}
/// Marker for the compute-pipeline API's independent reference namespace.
pub enum ComputePipelineObject {}
/// Marker for the function API's independent reference namespace.
pub enum FunctionObject {}
/// Marker for the fence API's independent reference namespace.
#[derive(Debug)]
pub enum FenceObject {}
/// Marker for the event API's independent reference namespace.
#[derive(Debug)]
pub enum EventObject {}

/// Bytes in one object-list entry.
pub const OBJECT_LIST_ENTRY_LEN: usize = 12;
const OBJECT_TYPE_MASK: u32 = 0xff;
const OBJECT_DESC_LEN_SHIFT: u32 = 8;

/// Semantic class selected by an object-list tag.
///
/// Two texture wire encodings normalize to [`Self::Texture`]. The raw tag is
/// retained privately by [`ObjectListEntry`] solely for boundary diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ObjectKind {
    Buffer,
    Texture,
    SurfaceBacking,
    IOSurfacePlaneView,
    Function,
    SerializerResource,
    TextureView,
    MemorylessTexture,
    IOSurfaceTexture,
    DualPlaneTexture,
    ResourceHandle,
    HeapBuffer,
    ExternalBuffer,
}

impl ObjectKind {
    pub const fn from_wire_tag(tag: u8) -> Option<Self> {
        match tag {
            1 => Some(Self::Buffer),
            2 | 3 => Some(Self::Texture),
            4 => Some(Self::SurfaceBacking),
            5 => Some(Self::IOSurfacePlaneView),
            6 => Some(Self::Function),
            7 => Some(Self::SerializerResource),
            8 => Some(Self::TextureView),
            9 => Some(Self::MemorylessTexture),
            11 => Some(Self::IOSurfaceTexture),
            12 => Some(Self::DualPlaneTexture),
            13 => Some(Self::ResourceHandle),
            14 => Some(Self::HeapBuffer),
            15 => Some(Self::ExternalBuffer),
            _ => None,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Buffer => "buffer",
            Self::Texture => "texture",
            Self::SurfaceBacking => "surface_backing",
            Self::IOSurfacePlaneView => "iosurface_plane_view",
            Self::Function => "function",
            Self::SerializerResource => "serializer_resource",
            Self::TextureView => "texture_view",
            Self::MemorylessTexture => "memoryless_texture",
            Self::IOSurfaceTexture => "iosurface_texture",
            Self::DualPlaneTexture => "dual_plane_texture",
            Self::ResourceHandle => "resource_handle",
            Self::HeapBuffer => "heap_buffer",
            Self::ExternalBuffer => "external_buffer",
        }
    }
}

impl fmt::Display for ObjectKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One decoded task-local object namespace entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObjectListEntry {
    pub kind: ObjectKind,
    pub descriptor_length: u32,
    pub descriptor_gva: u64,
    wire_tag: u8,
}

impl ObjectListEntry {
    /// Construct a semantic entry outside the wire decoder.
    ///
    /// Production guest entries should use [`decode_object_list_entry`]. This
    /// constructor exists for already-semantic producers such as scripted
    /// executors and tests, and chooses the canonical encoding when a semantic
    /// kind has more than one wire representation.
    pub const fn new(kind: ObjectKind, descriptor_length: u32, descriptor_gva: u64) -> Self {
        let wire_tag = match kind {
            ObjectKind::Buffer => 1,
            ObjectKind::Texture => 2,
            ObjectKind::SurfaceBacking => 4,
            ObjectKind::IOSurfacePlaneView => 5,
            ObjectKind::Function => 6,
            ObjectKind::SerializerResource => 7,
            ObjectKind::TextureView => 8,
            ObjectKind::MemorylessTexture => 9,
            ObjectKind::IOSurfaceTexture => 11,
            ObjectKind::DualPlaneTexture => 12,
            ObjectKind::ResourceHandle => 13,
            ObjectKind::HeapBuffer => 14,
            ObjectKind::ExternalBuffer => 15,
        };
        Self {
            kind,
            descriptor_length,
            descriptor_gva,
            wire_tag,
        }
    }

    /// The original numeric tag, for boundary diagnostics and fixture parity.
    pub const fn wire_tag(self) -> u8 {
        self.wire_tag
    }
}

/// A typed refusal from the object-list boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectListDecodeError {
    Short { actual: usize },
    UnknownKind { wire_tag: u8 },
}

/// Parse one object-list entry and consume its numeric class tag.
pub fn decode_object_list_entry(bytes: &[u8]) -> Result<ObjectListEntry, ObjectListDecodeError> {
    if bytes.len() < OBJECT_LIST_ENTRY_LEN {
        return Err(ObjectListDecodeError::Short {
            actual: bytes.len(),
        });
    }
    let first = u32::from_le_bytes(bytes[0..4].try_into().expect("length checked"));
    let wire_tag = (first & OBJECT_TYPE_MASK) as u8;
    let kind = ObjectKind::from_wire_tag(wire_tag)
        .ok_or(ObjectListDecodeError::UnknownKind { wire_tag })?;
    Ok(ObjectListEntry {
        kind,
        descriptor_length: first >> OBJECT_DESC_LEN_SHIFT,
        descriptor_gva: u64::from_le_bytes(bytes[4..12].try_into().expect("length checked")),
        wire_tag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(tag: u8, len: u32, gva: u64) -> [u8; OBJECT_LIST_ENTRY_LEN] {
        let mut bytes = [0u8; OBJECT_LIST_ENTRY_LEN];
        bytes[0..4].copy_from_slice(&(u32::from(tag) | (len << 8)).to_le_bytes());
        bytes[4..12].copy_from_slice(&gva.to_le_bytes());
        bytes
    }

    /// A texture this device never decoded a declaration for announces nothing.
    ///
    /// The absent declaration is the case that must not be filled in. Assuming
    /// a mode there would hand the gather witness a statement the guest never
    /// made, on exactly the resources this device understands least — so the
    /// answer is `Silent`, which costs the re-read and keeps the content.
    #[test]
    fn a_texture_with_no_declaration_promises_no_announcement() {
        let mut descriptor = LinearTextureDescriptor::default();
        assert_eq!(descriptor.declaration, None);
        assert_eq!(
            descriptor.guest_write_announcement(),
            crate::GuestWriteAnnouncement::Silent
        );

        // A declared `Managed` texture is the one mode that does announce, so
        // the fail-closed answer above is the absence and not a constant.
        let mut declaration = crate::TextureDeclaration {
            texture_type: 0,
            framebuffer_only: false,
            is_drawable: false,
            write_swizzle_enabled: None,
            allow_gpu_optimized_contents: false,
            usage: 0,
            pixel_format: 0,
            width: 1,
            height: 1,
            depth: 1,
            mipmap_level_count: 1,
            sample_count: 1,
            array_length: 1,
            // `MTLResourceOptions` carries the mode ordinal shifted left by
            // four, so `Managed` (ordinal 1) is `0x0010`.
            resource_options: 0x0010,
            protection_options: 0,
            swizzle: None,
        };
        descriptor.declaration = Some(declaration);
        assert_eq!(declaration.storage_mode(), crate::StorageMode::Managed);
        assert_eq!(
            descriptor.guest_write_announcement(),
            crate::GuestWriteAnnouncement::Announced
        );

        declaration.resource_options = 0x0000;
        descriptor.declaration = Some(declaration);
        assert_eq!(declaration.storage_mode(), crate::StorageMode::Shared);
        assert_eq!(
            descriptor.guest_write_announcement(),
            crate::GuestWriteAnnouncement::Silent,
            "a Shared texture is CPU-written without ever announcing it"
        );
    }

    #[test]
    fn texture_encodings_normalize_at_the_boundary() {
        let primary = decode_object_list_entry(&entry(2, 32, 0x4000)).unwrap();
        let alternate = decode_object_list_entry(&entry(3, 32, 0x5000)).unwrap();
        assert_eq!(primary.kind, ObjectKind::Texture);
        assert_eq!(alternate.kind, ObjectKind::Texture);
        assert_eq!(primary.wire_tag(), 2);
        assert_eq!(alternate.wire_tag(), 3);
    }

    #[test]
    fn iosurface_texture_has_a_semantic_name() {
        let decoded = decode_object_list_entry(&entry(11, 0x38, 0x6000)).unwrap();
        assert_eq!(decoded.kind, ObjectKind::IOSurfaceTexture);
        assert_eq!(decoded.kind.name(), "iosurface_texture");
    }

    #[test]
    fn unknown_tags_are_refused_at_the_boundary() {
        assert_eq!(
            decode_object_list_entry(&entry(0xfe, 16, 0x7000)),
            Err(ObjectListDecodeError::UnknownKind { wire_tag: 0xfe })
        );
    }

    #[test]
    fn color_write_masks_are_total_over_the_contract_bits() {
        assert_eq!(ColorWriteMask::default().bits(), MTL_COLOR_WRITE_MASK_ALL);
        for bits in 0..=MTL_COLOR_WRITE_MASK_ALL {
            assert_eq!(ColorWriteMask::new(bits).unwrap().bits(), bits);
        }
        assert_eq!(ColorWriteMask::new(MTL_COLOR_WRITE_MASK_ALL + 1), None);
    }

    #[test]
    fn linear_buffer_addresses_require_explicit_guest_page_geometry() {
        let descriptor = BufferDescriptor {
            allocation_size: 0x3000,
            handle64: 0x1_0000_0042,
            handle: 0x42,
        };
        assert_eq!(descriptor.backing_gva_size(12), Some((0x42_000, 0x3000)));
        assert_eq!(descriptor.backing_gva_size(14), Some((0x108_000, 0x3000)));
        assert_eq!(descriptor.backing_gva_size(0), None);
    }

    #[test]
    fn registered_surface_planes_decode_as_one_semantic_descriptor() {
        let bytes =
            reims_vgpu_wire::device_desc::SurfaceBackingBuilder::new(0x8000, 0x123, 0x3432_3066, 2)
                .plane(0, 0, 640, 480, 640, 1)
                .plane(1, 0x4000, 320, 240, 640, 2);
        let decoded = decode_surface_backing_descriptor(bytes.bytes()).unwrap();
        assert_eq!((decoded.length, decoded.backing_pfn), (0x8000, 0x123));
        assert_eq!(decoded.plane_count, 2);
        assert_eq!(
            decoded.planes[1],
            SurfaceBackingPlane {
                offset: 0x4000,
                width: 320,
                height: 240,
                bytes_per_row: 640,
                bytes_per_element: 2,
            }
        );
        assert_eq!((decoded.width, decoded.height), (640, 480));
    }

    #[test]
    fn registered_surface_decode_refuses_lossy_plane_prefixes() {
        use reims_vgpu_wire::device_desc::{
            surface_backing_len_for, SurfaceBackingBuilder, SURFACE_BACKING_PLANE_CAP,
        };

        let over_cap = SurfaceBackingBuilder::new(
            0x1000,
            1,
            0x4247_5241,
            (SURFACE_BACKING_PLANE_CAP + 1) as u8,
        );
        assert_eq!(
            decode_surface_backing_descriptor(over_cap.bytes()),
            Err(ResourceDecodeError::ErrUnsupported(
                "res_surface_backing_plane_count"
            ))
        );

        let short = SurfaceBackingBuilder::new(0x1000, 1, 0x4247_5241, 2)
            .with_len(surface_backing_len_for(1));
        assert_eq!(
            decode_surface_backing_descriptor(short.bytes()),
            Err(ResourceDecodeError::ErrShort("res_surface_backing_plane"))
        );
    }

    #[test]
    fn texture_view_semantics_do_not_expose_serializer_opcodes() {
        let simple = TextureViewDescriptor::default();
        assert!(!simple.carries_range());
        assert!(!simple.carries_swizzle());

        let swizzled = TextureViewDescriptor {
            form: TextureViewForm::Swizzled,
            pixel_format: 80,
            ..Default::default()
        };
        assert!(swizzled.carries_range());
        assert!(swizzled.carries_swizzle());
        assert_eq!(swizzled.declared_pixel_format(), Some(80));
    }

    #[test]
    fn linear_texture_geometry_is_checked_in_the_semantic_descriptor() {
        let descriptor = LinearTextureDescriptor {
            allocation_size: 0x8000,
            handle: 0x20,
            base_offset: 0x100,
            bytes_per_slice: 0x1000,
            slice_count: 1,
            width: 16,
            height: 8,
            row_stride: 128,
            levels: vec![TextureLevelLayout {
                offset: 0x300,
                size: 0x1000,
                row_stride: 128,
                width: 16,
                height: 8,
                depth: 0,
            }],
            ..Default::default()
        };
        assert_eq!(descriptor.backing_gva_size(12), Some((0x20_100, 0x7f00)));
        assert_eq!(descriptor.physical_slice_count(), Some(1));
        assert_eq!(descriptor.packed_allocation_end(), Some(0x1100));
        assert!(descriptor.declared_packing_fits_allocation());
        let (gva, level) = descriptor.level_gva(0, 12).unwrap();
        assert_eq!(gva, 0x20_400);
        assert_eq!(level.read_span(64), Some(7 * 128 + 64));
        assert_eq!(level.slice_read_span(64), level.read_span(64));
        assert_eq!(descriptor.level_gva(1, 12), None);
    }

    #[test]
    fn cube_texture_packing_expands_faces_and_keeps_mip_offsets_slice_relative() {
        let descriptor = LinearTextureDescriptor {
            allocation_size: 0x40_000,
            handle: 3,
            base_offset: 0x100,
            bytes_per_slice: 0x3000,
            slice_count: 3,
            cube_faces: true,
            levels: vec![
                TextureLevelLayout {
                    offset: 0x200,
                    row_stride: 0x100,
                    width: 32,
                    height: 16,
                    depth: 1,
                    ..Default::default()
                },
                TextureLevelLayout {
                    offset: 0x2400,
                    row_stride: 0x80,
                    width: 16,
                    height: 8,
                    depth: 1,
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert_eq!(descriptor.physical_slice_count(), Some(18));
        assert_eq!(descriptor.subresource_offset(2, 1), Some(0x8500));
        assert_eq!(descriptor.packed_allocation_end(), Some(0x36_100));
        assert!(descriptor.declared_packing_fits_allocation());

        let short = LinearTextureDescriptor {
            allocation_size: 0x36_0ff,
            ..descriptor
        };
        assert!(!short.declared_packing_fits_allocation());
    }

    #[test]
    fn a_single_physical_slice_does_not_require_an_inter_slice_advance() {
        let descriptor = LinearTextureDescriptor {
            allocation_size: 0x4100,
            handle: 9,
            mipmap_level_count: 2,
            base_offset: 0x100,
            bytes_per_slice: 0,
            slice_count: 1,
            levels: vec![
                TextureLevelLayout {
                    offset: 0,
                    size: 0x4000,
                    row_stride: 0x100,
                    width: 64,
                    height: 64,
                    depth: 1,
                },
                TextureLevelLayout {
                    offset: 0,
                    size: 0x1000,
                    row_stride: 0x80,
                    width: 32,
                    height: 32,
                    depth: 1,
                },
            ],
            ..Default::default()
        };

        assert_eq!(descriptor.physical_slice_span(), Some(0x4000));
        assert_eq!(descriptor.packed_allocation_end(), Some(0x4100));
        assert!(descriptor.declared_packing_fits_allocation());
        assert_eq!(descriptor.subresource_offset(0, 1), Some(0x100));

        let short = LinearTextureDescriptor {
            allocation_size: 0x40ff,
            ..descriptor.clone()
        };
        assert!(!short.declared_packing_fits_allocation());

        let array = LinearTextureDescriptor {
            allocation_size: 0x8100,
            slice_count: 2,
            ..descriptor
        };
        assert_eq!(array.physical_slice_span(), None);
        assert_eq!(array.subresource_offset(1, 0), None);
        assert!(!array.declared_packing_fits_allocation());
    }

    #[test]
    fn linear_texture_levels_refuse_invalid_allocation_geometry() {
        let descriptor = LinearTextureDescriptor {
            allocation_size: 0x1000,
            handle: 7,
            levels: vec![TextureLevelLayout {
                offset: 0x0f80,
                row_stride: 0x100,
                width: 8,
                height: 1,
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(descriptor.level_gva(0, 12), None);
        assert_eq!(descriptor.allocation_base_gva(0), None);
    }

    #[test]
    fn indirect_command_flags_are_classified_without_losing_unknown_bits() {
        let descriptor = IndirectCommandBufferDescriptor {
            flags: (1 << 0) | (1 << 2) | (1 << 6),
            ..Default::default()
        };
        assert!(descriptor.inherit_pipeline_state());
        assert!(!descriptor.inherit_buffers());
        assert_eq!(descriptor.unidentified_flags(), 1 << 6);
        let unapplied = descriptor.unapplied_flags();
        assert!(unapplied.contains(&IcbUnappliedFlag::SupportRayTracing));
        assert!(unapplied.contains(&IcbUnappliedFlag::InheritDepthStencilState));
    }
}
