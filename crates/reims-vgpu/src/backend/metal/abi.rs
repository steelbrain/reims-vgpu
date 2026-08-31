//! `repr(C)` types for the Metal encode path.
//!
//! These began as mirrors of a C backend header that is no longer part of this
//! tree, and no C or Objective-C translation unit reads them today — the
//! `repr(C)` layout is what the encode path and its hashes are written against,
//! which is why the size and offset pins below stay.
//!
//! The layout pins below are `const` assertions rather than `#[test]`s, for the
//! reason spelled out in [`crate::backend::metal::constants`]: this module is
//! `backend-metal`-gated, so a `#[cfg(test)]` block in it is compiled out of the
//! Vulkan arm and its `--lib` suite is run on Apple hosts only — the pins were
//! live on no machine anybody edits this code from. `rustc` evaluates a `const`
//! assertion on every arm that compiles the file, including the cross-compiled
//! `--target aarch64-apple-darwin` clippy run `AGENTS.md` requires from Linux.

#![allow(non_camel_case_types)]

use core::mem::offset_of;

pub const REIMS_VGPU_OK: i32 = 0;
pub const REIMS_VGPU_ERR_ARGS: i32 = 1;
pub const REIMS_VGPU_ERR_TRANSLATE: i32 = 2;
pub const REIMS_VGPU_ERR_EXECUTE: i32 = 3;

// The two binding-band bases this backend encodes (class, index) into. They are
// the device's bands, not the archived header's: the sampler base moved up so
// the texture band could be Metal's whole 128-entry argument table. `const`
// assertions in `backend::metal::constants` pin both to
// `runtime::spirv_bind`'s, so the two arms cannot number a bind differently.
pub const REIMS_VGPU_BINDING_TEXTURE_BASE: u32 = 32;
pub const REIMS_VGPU_BINDING_SAMPLER_BASE: u32 = 160;

pub const REIMS_VGPU_MTL_PRIMITIVE_TYPE_POINT: u32 = 0;
pub const REIMS_VGPU_MTL_PRIMITIVE_TYPE_LINE: u32 = 1;
pub const REIMS_VGPU_MTL_PRIMITIVE_TYPE_LINE_STRIP: u32 = 2;
pub const REIMS_VGPU_MTL_PRIMITIVE_TYPE_TRIANGLE: u32 = 3;
pub const REIMS_VGPU_MTL_PRIMITIVE_TYPE_TRIANGLE_STRIP: u32 = 4;

// `contract::dispatch` is where the shared decode/exec path reads this pair, so
// it is the definition and these are aliases. This module is
// `backend-metal`-gated, so a value spelled only here is unreachable from the
// code that accepts the field off the wire; deriving rather than re-spelling is
// what stops the two names from parting.
pub const REIMS_VGPU_MTL_DISPATCH_TYPE_SERIAL: u32 =
    crate::contract::dispatch::MTL_DISPATCH_TYPE_SERIAL;
pub const REIMS_VGPU_MTL_DISPATCH_TYPE_CONCURRENT: u32 =
    crate::contract::dispatch::MTL_DISPATCH_TYPE_CONCURRENT;

// The dispatch kind reaches `compute::compute_core` as a `bool` and never
// becomes an ordinal on this side: it is produced as a `bool`, and widening it
// to `{0, 1}` to cross a call put it beside `dispatch_type`, which is also
// `{0, 1}`. Named here so the encode path has the spelling if it ever needs it.
pub const REIMS_VGPU_COMPUTE_DISPATCH_KIND_THREADGROUPS: u32 = 0;
pub const REIMS_VGPU_COMPUTE_DISPATCH_KIND_THREADS: u32 = 1;

// These two size the mirror arrays below; they do not decide what a guest may
// declare. The decoder's caps are the contract (`MTLStageInputOutputDescriptor`
// is a 31-slot array on both sides), and this array must be wide enough to carry
// everything the decoder admits or the handoff to Metal truncates what decode
// kept. Derived from the decoder so the array cannot be sized under it.
pub const REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES: usize =
    crate::runtime::decode::resource::MAX_COMPUTE_STAGE_INPUT_ATTRS;
pub const REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS: usize =
    crate::runtime::decode::resource::MAX_COMPUTE_STAGE_INPUT_LAYOUTS;
pub const REIMS_VGPU_COMPUTE_STAGE_INPUT_STRIDE_DYNAMIC: u64 = u64::MAX;

pub const REIMS_VGPU_MTL_PIXEL_FORMAT_DEPTH32_FLOAT: u32 = 252;
pub const REIMS_VGPU_MTL_PIXEL_FORMAT_STENCIL8: u32 = 253;
// `contract::pass_action` declares these five as the `u16` the render-pass
// attachment prefix carries them in, and the encode path converts between the
// two widths. Widening the contract's `u16` here is the conversion, so there is
// one definition and no second spelling to drift from it.
pub const REIMS_VGPU_MTL_LOAD_ACTION_DONT_CARE: u32 =
    crate::contract::pass_action::MTL_LOAD_ACTION_DONT_CARE as u32;
pub const REIMS_VGPU_MTL_LOAD_ACTION_LOAD: u32 =
    crate::contract::pass_action::MTL_LOAD_ACTION_LOAD as u32;
pub const REIMS_VGPU_MTL_LOAD_ACTION_CLEAR: u32 =
    crate::contract::pass_action::MTL_LOAD_ACTION_CLEAR as u32;
pub const REIMS_VGPU_MTL_STORE_ACTION_DONT_CARE: u32 =
    crate::contract::pass_action::MTL_STORE_ACTION_DONT_CARE as u32;
pub const REIMS_VGPU_MTL_STORE_ACTION_STORE: u32 =
    crate::contract::pass_action::MTL_STORE_ACTION_STORE as u32;

pub const REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_READ: u32 = 0;
pub const REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_READ_WRITE: u32 = 1;
pub const REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_WRITE: u32 = 2;

pub const REIMS_VGPU_TEXTURE_SWIZZLE_ZERO: u8 = 0;
pub const REIMS_VGPU_TEXTURE_SWIZZLE_ONE: u8 = 1;
pub const REIMS_VGPU_TEXTURE_SWIZZLE_RED: u8 = 2;
pub const REIMS_VGPU_TEXTURE_SWIZZLE_GREEN: u8 = 3;
pub const REIMS_VGPU_TEXTURE_SWIZZLE_BLUE: u8 = 4;
pub const REIMS_VGPU_TEXTURE_SWIZZLE_ALPHA: u8 = 5;

/// Metal's own viewport-array width, not this device's choice.
///
/// The Metal Shading Language specification declares `[[viewport_array_index]]`
/// as taking values `0` through `15`, so a render encoder rasterizes into at
/// most sixteen viewports and `setViewports:count:` with more is out of
/// contract. `setScissorRects:count:` is one rect per viewport and takes the
/// same width, which is why the two constants are equal rather than one being
/// derived from the other by coincidence.
///
/// The refusal at the comparison is what keeps a larger count from reaching
/// Metal, where it is a process-aborting exception rather than a status.
pub const REIMS_VGPU_BACKEND_MAX_VIEWPORTS: usize = 16;
pub const REIMS_VGPU_BACKEND_MAX_SCISSORS: usize = REIMS_VGPU_BACKEND_MAX_VIEWPORTS;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuComputeStageInputAttribute {
    pub raw_bits: u32,
    pub location: u32,
    pub format: u32,
    pub offset: u32,
    pub buffer_index: u32,
    pub reserved0: u32,
}

const _: () = assert!(size_of::<ReimsVgpuComputeStageInputAttribute>() == 6 * size_of::<u32>());
const _: () = assert!(align_of::<ReimsVgpuComputeStageInputAttribute>() == align_of::<u32>());

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuComputeStageInputLayout {
    pub raw_bits: u32,
    pub buffer_index: u32,
    pub step_function: u32,
    pub step_rate: u32,
    pub stride: u64,
}

// Four `u32`s then a `u64`, so `stride` is 8-aligned and the record is 24 and
// not 20. A field reordered ahead of `stride` moves it and changes every offset
// the descriptor below inherits.
const _: () = assert!(size_of::<ReimsVgpuComputeStageInputLayout>() == 24);
const _: () = assert!(offset_of!(ReimsVgpuComputeStageInputLayout, stride) == 16);

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuComputeStageInputDescriptor {
    pub word0: u32,
    pub header0: u32,
    pub header1: u32,
    pub attribute_count: u32,
    pub layout_count: u32,
    pub index_type: u32,
    pub index_buffer_index: u32,
    pub attributes:
        [ReimsVgpuComputeStageInputAttribute; REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_ATTRIBUTES],
    pub layouts: [ReimsVgpuComputeStageInputLayout; REIMS_VGPU_COMPUTE_STAGE_INPUT_MAX_LAYOUTS],
}

// The literal size and both array offsets are what the tests these replaced
// asserted, kept literal on purpose: derived from the two caps they would stay
// true across a cap change, and a cap change is exactly the edit that has to be
// walked to the sites reading the arrays.
const _: () = assert!(size_of::<ReimsVgpuComputeStageInputDescriptor>() == 1520);
const _: () = assert!(offset_of!(ReimsVgpuComputeStageInputDescriptor, attributes) == 28);
const _: () = assert!(offset_of!(ReimsVgpuComputeStageInputDescriptor, layouts) == 776);

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuBuffer {
    pub binding: u32,
    pub data: *mut u8,
    pub len: usize,
    pub attribute_stride: u64,
    pub has_attribute_stride: u32,
    pub reserved0: u32,
    pub backing_data: *mut u8,
    pub backing_len: usize,
    pub backing_offset: usize,
}

// The pointer-bearing records are 64-bit shaped: every `u32` ahead of a pointer
// or `usize` is padded up to the 8-byte slot the next field starts on. Reading
// these offsets as if they packed is what a 32-bit assumption looks like here.
const _: () = assert!(size_of::<usize>() == 8, "Metal backend ABI is 64-bit");
const _: () = assert!(size_of::<ReimsVgpuBuffer>() == 64);
const _: () = assert!(offset_of!(ReimsVgpuBuffer, data) == 8);
const _: () = assert!(offset_of!(ReimsVgpuBuffer, backing_data) == 40);
const _: () = assert!(offset_of!(ReimsVgpuBuffer, backing_offset) == 56);

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuStorageImage {
    pub binding: u32,
    /// The contract's selector, not its ordinal. `StorageImageSelector` is
    /// `#[repr(u32)]`, so this occupies the same four bytes the `u32` did and
    /// the layout assertions below are unchanged — but the consumer can no
    /// longer be handed a value it has no arm for.
    pub format: crate::contract::pixel_format::StorageImageSelector,
    pub width: u32,
    pub height: u32,
    pub data: *mut u8,
    pub len: usize,
}

const _: () = assert!(size_of::<ReimsVgpuStorageImage>() == 32);
const _: () = assert!(offset_of!(ReimsVgpuStorageImage, data) == 16);

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuComputeSampledImage {
    pub binding: u32,
    /// The contract's selector, for [`ReimsVgpuStorageImage::format`]'s reason.
    pub format: crate::contract::pixel_format::StorageImageSelector,
    pub width: u32,
    pub height: u32,
    pub data: *const u8,
    pub len: usize,
    pub has_swizzle: u32,
    /// Read by [`super::compute`] only when `has_swizzle != 0`; inert otherwise.
    /// Build through [`Self::unswizzled`] rather than filling it in by hand.
    pub swizzle: [u8; 4],
}

const _: () = assert!(size_of::<ReimsVgpuComputeSampledImage>() == 40);
const _: () = assert!(offset_of!(ReimsVgpuComputeSampledImage, swizzle) == 36);

impl ReimsVgpuComputeSampledImage {
    /// A binding whose texels are consumed in their declared channel order.
    ///
    /// `has_swizzle` is 0, so the consumer never reads `swizzle` — which is
    /// exactly why the two call sites that build these had drifted to writing
    /// different dead values into it, one `[0; 4]` and one `[2, 3, 4, 5]`
    /// carrying an "identity RGBA selectors" comment. Both were inert and the
    /// pair reads as a behavioral divergence to anyone diffing them. Naming the
    /// case removes the field from the call site, so there is nothing left to
    /// disagree about.
    pub fn unswizzled(
        binding: u32,
        format: crate::contract::pixel_format::StorageImageSelector,
        width: u32,
        height: u32,
        data: *const u8,
        len: usize,
    ) -> Self {
        Self {
            binding,
            format,
            width,
            height,
            data,
            len,
            has_swizzle: 0,
            swizzle: [0; 4],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuComputeTextureUsage {
    pub binding: u32,
    pub access: u32,
}

/// Whether a compute texture binding materializes as a storage image, from the
/// shader's own reflected usage.
///
/// A binding the reflection does not mention defaults to read-write, and so to
/// storage. That is deliberate and it is the permissive direction: this answer
/// picks the descriptor the texture is *bound* as, and a shader that writes
/// through a binding materialized as sampled-only writes nothing, silently.
/// Materializing a read-only texture as storage costs a descriptor type, not a
/// result. When the two errors are not symmetric, take the one that cannot lose
/// guest work.
///
/// One function because it was two — `compute_exec` and `compute_session` each
/// reflected the `.mtlb`, each built the same `access_for` closure over the
/// result, and each applied this rule in the same three lines. They agreed, but
/// by copy: nothing compared them, and the rule is a decision about what the
/// guest's shader does rather than a lookup.
pub fn texture_binds_as_storage(usages: &[ReimsVgpuComputeTextureUsage], binding: u32) -> bool {
    usages
        .iter()
        .find(|u| u.binding == binding)
        .map_or(REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_READ_WRITE, |u| u.access)
        != REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_READ
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuSampledImage {
    pub binding: u32,
    pub width: u32,
    pub height: u32,
    pub rgba8: *const u8,
    pub len: usize,
    pub pixel_format: u32,
    pub bytes_per_row: u32,
    pub data: *const u8,
    pub data_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuSampler {
    pub binding: u32,
    pub unnormalized: u32,
    pub min_filter: u32,
    pub mag_filter: u32,
    pub mip_filter: u32,
    pub s_address_mode: u32,
    pub t_address_mode: u32,
    pub r_address_mode: u32,
    pub border_color: u32,
    pub compare_function: u32,
    pub lod_min_bits: u32,
    pub lod_max_bits: u32,
    pub max_anisotropy: u32,
    pub lod_average: u32,
    pub support_argument_buffers: u32,
    pub has_lod_clamp: u32,
    pub clamp_lod_min_bits: u32,
    pub clamp_lod_max_bits: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuThreadgroupMemory {
    pub index: u32,
    pub length: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuComputeStageInRegion {
    pub origin_x: u64,
    pub origin_y: u64,
    pub origin_z: u64,
    pub size_x: u64,
    pub size_y: u64,
    pub size_z: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuComputeStageInRegionIndirectArguments {
    pub origin_x: u32,
    pub origin_y: u32,
    pub origin_z: u32,
    pub size_x: u32,
    pub size_y: u32,
    pub size_z: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuComputeImageblockDimensions {
    pub width: u32,
    pub height: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuBlendState {
    pub enable: u32,
    pub src_rgb: u32,
    pub dst_rgb: u32,
    pub op_rgb: u32,
    pub src_alpha: u32,
    pub dst_alpha: u32,
    pub op_alpha: u32,
    pub has_blend_color: u32,
    pub blend_color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuViewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub znear: f32,
    pub zfar: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuScissor {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// The encoder raster state this device applies.
///
/// Four of the five `MTLRenderCommandEncoder` raster setters. Each state is a
/// `has_` flag beside its raw Metal ordinal, so a word the stream never bound
/// is not read at all — the ordinals have no reserved "unset" value, and 0 is
/// a real mode in all four.
///
/// The fifth is line width, which `setLineWidth:` puts on the wire and
/// `MTLRenderCommandEncoder` has no public setter for; `runtime::exec` still
/// counts it as `render_line_width_dropped`.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuRasterState {
    pub has_cull_mode: u32,
    pub cull_mode: u32,
    pub has_front_facing_winding: u32,
    pub front_facing_winding: u32,
    pub has_fill_mode: u32,
    pub fill_mode: u32,
    pub has_depth_clip_mode: u32,
    pub depth_clip_mode: u32,
}

impl ReimsVgpuRasterState {
    /// Whether the stream bound any of these states, and so whether the record
    /// is worth encoding at all.
    ///
    /// Here rather than at the producer because the answer is a property of
    /// the struct: a field added without an arm in this `or` chain is a state
    /// the guest set and the encoder never hears about, and this way there is
    /// one place for that mistake instead of one per call site.
    pub fn any_bound(&self) -> bool {
        self.has_cull_mode != 0
            || self.has_front_facing_winding != 0
            || self.has_fill_mode != 0
            || self.has_depth_clip_mode != 0
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuDepthBiasState {
    pub depth_bias: f32,
    pub slope_scale: f32,
    pub clamp: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ReimsVgpuDepthStencilFaceState {
    pub compare_function: u32,
    pub stencil_failure_operation: u32,
    pub depth_failure_operation: u32,
    pub depth_stencil_pass_operation: u32,
    pub read_mask: u32,
    pub write_mask: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ReimsVgpuDepthStencilState {
    pub depth_compare_function: u32,
    pub depth_write_enabled: u32,
    pub front_stencil_enabled: u32,
    pub back_stencil_enabled: u32,
    pub front_face: ReimsVgpuDepthStencilFaceState,
    pub back_face: ReimsVgpuDepthStencilFaceState,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuStencilReferenceState {
    pub front: u32,
    pub back: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuDepthAttachment {
    pub pixel_format: u32,
    pub load_action: u32,
    pub store_action: u32,
    pub clear_depth: f64,
    pub data: *mut u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuStencilAttachment {
    pub pixel_format: u32,
    pub load_action: u32,
    pub store_action: u32,
    pub clear_stencil: u32,
    pub data: *mut u8,
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuPrimitiveIndirectArguments {
    pub vertex_count: u32,
    pub instance_count: u32,
    pub vertex_start: u32,
    pub base_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuPrimitiveIndirectDraw {
    pub arguments: *const u8,
    pub arguments_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuIndexedIndirectArguments {
    pub index_count: u32,
    pub instance_count: u32,
    pub index_start: u32,
    pub base_vertex: i32,
    pub base_instance: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuIndexedIndirectDraw {
    pub arguments: *const u8,
    pub arguments_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuIndexedDraw {
    pub index_type: u32,
    pub index_count: usize,
    pub base_vertex: i64,
    pub indices: *const u8,
    pub indices_len: usize,
    pub indirect: *const ReimsVgpuIndexedIndirectDraw,
}

/// One vertex attribute as the Metal encoder wants it.
///
/// `step_function` and `step_rate` are the *resolved* values rather than the
/// tagged record's optionals: the caller applies
/// [`VertexAttribute::step_function_ordinal`] and [`VertexAttribute::step_rate`]
/// once and this side sets what it is given. Carrying the presence bits across
/// instead made two functions in [`super::render`] re-derive the same defaults,
/// and put both bits in the pipeline cache key beside the values they had
/// already been folded into.
///
/// [`VertexAttribute::step_function_ordinal`]: crate::runtime::decode::resource::VertexAttribute::step_function_ordinal
/// [`VertexAttribute::step_rate`]: crate::runtime::decode::resource::VertexAttribute::step_rate
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ReimsVgpuVertexAttr {
    pub location: u32,
    pub format: u32,
    pub offset: u32,
    pub buffer_index: u32,
    pub stride: u32,
    pub data: *const u8,
    pub len: usize,
    pub step_function: u32,
    pub step_rate: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule two call sites held a copy of each, in the direction that cannot
    /// lose a shader's writes.
    #[test]
    fn an_unreflected_texture_binding_materializes_as_storage() {
        let usages = [
            ReimsVgpuComputeTextureUsage {
                binding: 4,
                access: REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_READ,
            },
            ReimsVgpuComputeTextureUsage {
                binding: 5,
                access: REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_WRITE,
            },
            ReimsVgpuComputeTextureUsage {
                binding: 6,
                access: REIMS_VGPU_COMPUTE_TEXTURE_ACCESS_READ_WRITE,
            },
        ];
        // Read-only is the only access that is not storage; both writing
        // accesses are, which is why the test names all three rather than the
        // one the default happens to equal.
        assert!(!texture_binds_as_storage(&usages, 4));
        assert!(texture_binds_as_storage(&usages, 5));
        assert!(texture_binds_as_storage(&usages, 6));

        // A binding the reflection never mentioned, and the empty reflection —
        // both take the default, and it is storage.
        assert!(texture_binds_as_storage(&usages, 7));
        assert!(texture_binds_as_storage(&[], 4));
    }
}
