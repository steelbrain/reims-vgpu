//! Resource descriptor decode (port of `host/utils/reims-vgpu-resource-decode`).

use crate::contract::endian::{ld16, ld32, ld64}; // ld64: texture-view level base/count
#[cfg(test)]
use crate::contract::endian::{st16, st32};
use crate::runtime::heap_query; // ICB layout fixture encoder only

use core::mem::{offset_of, size_of};
use reims_vgpu_wire::ops::{
    backed_texture as w_backed, depth_stencil as w_ds, heap_texture as w_heap, icb as w_icb,
    sampler as w_smp, texture_view as w_view,
};
use reims_vgpu_wire::OP_HEADER_LEN as OP_HDR;

/// A refusal from the descriptor decoder.
///
/// This decoder sits upstream of every rail — blit, compute, render, mipmap and
/// resource registration all reach a guest object through it — so a refusal
/// here means some object the guest created is unusable everywhere. The payload
/// is the registered slug naming which check refused: **29 of the 40 sites were
/// `ErrShort`**, one name for twenty-nine different reads, from a 12-byte
/// object-list entry to a vertex-attribute table offset inside a type-7 body.
///
/// Slugs carry a `res_` prefix. Six modules under `runtime/decode/` define a
/// type called `DecodeStatus` and five of them have an `ErrShort` meaning a
/// different read, so without the prefix the crate-wide uniqueness gate could
/// not tell this decoder's refusals from the stream framer's.
///
/// There is deliberately no `Ok`: every entry point returns
/// `Result<_, DecodeStatus>`. `Ok`, `ErrArgs` and `ErrOverflow` were **never
/// constructed anywhere in the crate** — they existed only as arms of a
/// `decode_status_name` helper that itself had no callers — so they are gone
/// and this is a [`crate::observe::Decline`], not a `Refusal`.
///
/// `ErrBadLength` went the same way. Every length disagreement this decoder can
/// see is a read that ran off the end, and all three sites of the compact-TLV
/// walk already say so with their own slug (`res_tlv_offset_past_end`,
/// `res_tlv_header_short`, `res_tlv_value_short`). A second class for the same
/// condition would only split one check's census across two names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeStatus {
    /// The blob is shorter than the field being read.
    ErrShort(&'static str),
    /// An object-list type tag this host has no contract for.
    ErrUnknownType(&'static str),
    /// A well-formed blob whose tag/opcode names a variant the decoder does not
    /// implement.
    ErrUnsupported(&'static str),
}

impl crate::observe::Decline for DecodeStatus {
    fn slug(&self) -> &'static str {
        match self {
            Self::ErrShort(s) | Self::ErrUnknownType(s) | Self::ErrUnsupported(s) => s,
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![(
            "class",
            match self {
                Self::ErrShort(_) => "short",
                Self::ErrUnknownType(_) => "unknown_type",
                Self::ErrUnsupported(_) => "unsupported",
            }
            .to_string(),
        )]
    }
}

/// Live object-list type tags (`reims_vgpu_resource_decode.h` / arm contract).
///
/// Type 3 carries the same geometry prefix as type 2 (WindowServer composite
/// and glyph sources); type 7 is the container for sampler, depth-stencil and
/// render/compute pipeline descriptors.
pub const OBJECT_TYPE_BUFFER: u8 = 1;
pub const OBJECT_TYPE_TEXTURE: u8 = 2;
pub const OBJECT_TYPE_TEXTURE_VARIANT: u8 = 3;
pub const OBJECT_TYPE_FUNCTION: u8 = 6;
pub const OBJECT_TYPE_TYPE7: u8 = 7;
pub const OBJECT_TYPE_TEXTURE_VIEW: u8 = 8;
pub const OBJECT_TYPE_IOSURFACE: u8 = 11;

/// Type-7 first dword subtypes.
pub const TYPE7_OBJECT_SAMPLER: u32 = w_smp::OPCODE_NEW_SAMPLER;
pub const TYPE7_OBJECT_DEPTH_STENCIL: u32 = w_ds::OPCODE_NEW_DEPTH_STENCIL;
pub const TYPE7_OBJECT_COMPUTE_PIPELINE: u32 = 0x0b;
pub const TYPE7_OBJECT_RENDER_PIPELINE: u32 = 0x0e;
/// Indirect command buffer create body from
/// `PGSerializer newIndirectCommandBufferWithDescriptor:layout:maxCommandCount:options:allocator:`.
pub const TYPE7_OBJECT_ICB: u32 = w_icb::OPCODE_NEW_INDIRECT_COMMAND_BUFFER;
/// End of the 16-byte type-7 header, which is also where its first TLV
/// starts — one boundary, so one name.
pub const TYPE7_FIRST_TLVS: usize = 16;
/// Serialized ICB descriptor length (allocateOperationBytes 0x58).
pub const ICB_DESC_LEN: usize = w_icb::NEW_INDIRECT_COMMAND_BUFFER_TOTAL_LEN as usize;
/// Per-stage max bind counts are single bytes (PGSerializer create body).
/// `newIndirectCommandBufferWithDescriptor:…` strb order:
/// +0xc vertex · +0xd fragment · +0xe kernel · +0xf object · +0x10 mesh ·
/// +0x11 kernelTG · +0x12 objectTG.
#[cfg(test)]
pub(crate) const ICB_DESC_MAX_VERTEX_BINDS: usize =
    OP_HDR + offset_of!(w_icb::NewIcbBody, max_vertex_buffer_bind_count);
#[cfg(test)]
pub(crate) const ICB_DESC_MAX_FRAGMENT_BINDS: usize =
    OP_HDR + offset_of!(w_icb::NewIcbBody, max_fragment_buffer_bind_count);
/// maxKernelBufferBindCount.
#[cfg(test)]
pub(crate) const ICB_DESC_MAX_KERNEL_BINDS: usize =
    OP_HDR + offset_of!(w_icb::NewIcbBody, max_kernel_buffer_bind_count);
/// maxObjectBufferBindCount (mesh object stage).
#[cfg(test)]
pub(crate) const ICB_DESC_MAX_OBJECT_BINDS: usize =
    OP_HDR + offset_of!(w_icb::NewIcbBody, max_object_buffer_bind_count);
/// maxMeshBufferBindCount.
#[cfg(test)]
pub(crate) const ICB_DESC_MAX_MESH_BINDS: usize =
    OP_HDR + offset_of!(w_icb::NewIcbBody, max_mesh_buffer_bind_count);
/// maxKernelThreadgroupMemoryBindCount.
#[cfg(test)]
pub(crate) const ICB_DESC_MAX_KERNEL_TG_BINDS: usize =
    OP_HDR + offset_of!(w_icb::NewIcbBody, max_kernel_threadgroup_memory_bind_count);
/// maxObjectThreadgroupMemoryBindCount.
#[cfg(test)]
pub(crate) const ICB_DESC_MAX_OBJECT_TG_BINDS: usize =
    OP_HDR + offset_of!(w_icb::NewIcbBody, max_object_threadgroup_memory_bind_count);
#[cfg(test)]
pub(crate) const ICB_DESC_FLAGS: usize = OP_HDR + offset_of!(w_icb::NewIcbBody, flags);
/// Bytes per ICB kernel-threadgroup-memory length slot (`u64` length at index).
pub const ICB_TG_MEMORY_STRIDE: usize = 8;
/// Bytes per ICB attribute-stride table entry (`u64` stride at buffer index).
/// `setKernelBuffer:offset:attributeStride:atIndex:` and
/// `setVertexBuffer:offset:attributeStride:atIndex:` store at
/// `attributeStrideOffset + index*8`.
pub const ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE: usize = 8;
/// Flags at `+0x16`, one bit per `MTLIndirectCommandBufferDescriptor` BOOL.
///
/// Every position below was derived by inverting exactly that property from the
/// value a fresh descriptor reads back and diffing the emitted record — one
/// case per property, so no bit is named from an assumption about ordering. The
/// derivation and its fixtures live in
/// [`reims_vgpu_wire::ops::icb::flag`](reims_vgpu_wire::ops::icb::flag), which
/// this device agrees with bit for bit; the two are checked against each other
/// in this module's tests.
///
/// The order is **not** the order Metal declares the properties, and the run is
/// **not contiguous**: bit 6 sits between `INHERIT_DEPTH_BIAS` and
/// `INHERIT_DEPTH_CLIP_MODE` and no property moves it. Do not extend this list
/// by counting.
pub const ICB_FLAG_INHERIT_PIPELINE_STATE: u16 = 1 << 0;
pub const ICB_FLAG_INHERIT_BUFFERS: u16 = 1 << 1;
/// `supportRayTracing`, default **off** on both Metal's descriptor and the
/// guest's, so a set bit is the guest asking for something.
pub const ICB_FLAG_SUPPORT_RAY_TRACING: u16 = 1 << 2;
/// `supportDynamicAttributeStride`, default off.
pub const ICB_FLAG_SUPPORT_DYNAMIC_ATTRIBUTE_STRIDE: u16 = 1 << 3;
/// `inheritDepthStencilState`, default **on** — so a *clear* bit is the guest
/// asking for something, which is the opposite reading from the two above.
pub const ICB_FLAG_INHERIT_DEPTH_STENCIL_STATE: u16 = 1 << 4;
/// `inheritDepthBias`, default on.
pub const ICB_FLAG_INHERIT_DEPTH_BIAS: u16 = 1 << 5;
/// `inheritDepthClipMode`, default on. Bit **7**, not bit 6.
pub const ICB_FLAG_INHERIT_DEPTH_CLIP_MODE: u16 = 1 << 7;
/// `inheritCullMode`, default on.
pub const ICB_FLAG_INHERIT_CULL_MODE: u16 = 1 << 8;
/// `inheritFrontFacingWinding`, default on.
pub const ICB_FLAG_INHERIT_FRONT_FACING_WINDING: u16 = 1 << 9;
/// `inheritTriangleFillMode`, default on.
pub const ICB_FLAG_INHERIT_TRIANGLE_FILL_MODE: u16 = 1 << 10;
/// Bits 6 and 11-14: set in every record the serializer produced and moved by
/// none of the eleven BOOLs the descriptor declares. Bit 15 is excluded because
/// the serializer never writes it, which the poison test measures rather than
/// assumes.
pub const ICB_FLAG_UNIDENTIFIED: u16 = (1 << 6) | (1 << 11) | (1 << 12) | (1 << 13) | (1 << 14);
/// Bit 15, which the serializer never writes: on a guest's ring it is whatever
/// the last record left there.
///
/// [`decode_icb_descriptor`] masks it off, so the decoded word holds only bits
/// Apple wrote. That is not fastidiousness —
/// [`IndirectCommandBufferDescriptor`] derives `PartialEq` and the host ICB
/// cache compares descriptors, so a noise bit would make one buffer look like
/// two. Storing the raw word without this mask is a bug the fixture instrument
/// caught within minutes of the word being stored at all.
pub const ICB_FLAG_NEVER_WRITTEN: u16 = 1 << 15;
/// The word Apple's serializer writes for a descriptor whose BOOLs are all at
/// their defaults: the six inherit-state flags **on**, the two `support*`
/// **off**, and the five unidentified bits on. Measured on every ICB fixture
/// the oracle captured.
///
/// Exists so a synthetic record in a test is a record Apple would actually
/// produce. A helper that writes `0` here builds a descriptor asking to inherit
/// *nothing*, which is a guest request rather than a blank, and it would trip
/// six of the counters in
/// [`IndirectCommandBufferDescriptor::unapplied_flags`] on every test that used
/// it.
#[cfg(test)]
pub(crate) const ICB_FLAGS_DEFAULT: u16 = 0x7ff0;
/// Embedded ICB command layout (52 B) at +0x1c in the create body.
#[cfg(test)]
pub(crate) const ICB_DESC_LAYOUT: usize = OP_HDR + offset_of!(w_icb::NewIcbBody, layout);
pub const ICB_LAYOUT_LEN: usize = size_of::<w_icb::IcbLayout>();
#[cfg(test)]
pub(crate) const ICB_DESC_MAX_COMMAND_COUNT: usize =
    OP_HDR + offset_of!(w_icb::NewIcbBody, max_command_count);
/// `MTLResourceOptions`, and it is a **`u16`**: the serializer narrows the `Q`
/// its selector declares, and `+0x56`/`+0x57` are never written at all.
///
/// Measured, not read. This was a `ld32` until the oracle's complementary-fill
/// passes were pointed at the record — `no_decoder_reads_a_bit_apples_serializer
/// _never_wrote` reported the same descriptor decoding `options: 0` under one
/// fill and `0xffff0000` under the other, which on a guest's ring is whatever
/// the last record left there. Same shape as the `copyFromTexture:toBuffer:`
/// `options` bug: a field read wider than the serializer writes.
pub const ICB_DESC_OPTIONS: usize = OP_HDR + offset_of!(w_icb::NewIcbBody, options);
/// The two bytes above [`ICB_DESC_OPTIONS`], which the serializer never writes.
/// Named so a future widening has to delete a constant that says why not.
#[cfg(test)]
const ICB_DESC_OPTIONS_UNWRITTEN: usize =
    OP_HDR + offset_of!(w_icb::NewIcbBody, never_written_tail);
/// Command-type values written by PGSerializerIndirect*Command fills.
pub const ICB_CMD_TYPE_DRAW: u32 = 0x1;
pub const ICB_CMD_TYPE_DRAW_INDEXED: u32 = 0x2;
/// `drawPatches` stores wire type `4`.
pub const ICB_CMD_TYPE_DRAW_PATCHES: u32 = 0x4;
/// `drawIndexedPatches` stores wire type `8`.
pub const ICB_CMD_TYPE_DRAW_INDEXED_PATCHES: u32 = 0x8;
pub const ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADGROUPS: u32 = 0x20;
pub const ICB_CMD_TYPE_CONCURRENT_DISPATCH_THREADS: u32 = 0x40;
/// Wire command type = SDK bit value (same pattern as Draw/Patches).
/// `setupCommandLayout:` uses `1<<7` / `1<<8` for mesh args size.
/// Fill IMPs are stubs; type value follows the bit-pattern convention.
pub const ICB_CMD_TYPE_DRAW_MESH_THREADGROUPS: u32 = 0x80;
pub const ICB_CMD_TYPE_DRAW_MESH_THREADS: u32 = 0x100;
/// Bytes per kernel/vertex/fragment buffer bind slot in the command layout.
pub const ICB_BUFFER_BIND_STRIDE: usize = 0x14;
/// Tessellation-factor table used size (u32 ref + 3×u64) at `tessellationFactorOffset`.
pub const ICB_TESSELLATION_FACTOR_LEN: usize = 0x1c;
/// Concurrent-dispatch args size: two `MTLSize`, grid then threadgroup, at
/// 3xu64 each — 2 * 3 * 8 = 0x30. Matches the `ConcurrentDispatch` bit's
/// allocation in host RE `setupCommandLayout:`.
pub const ICB_CONCURRENT_DISPATCH_ARGS_LEN: usize = 0x30;
/// DrawPatches args size: `setupCommandLayout` allocates 0x38, and the fill IMP
/// writes through `baseInstance` — a u64 *starting* at 0x2e, so ending at 0x36.
/// The two bytes between are the allocation's slack, exactly as
/// [`ICB_DRAW_INDEXED_PATCHES_ARGS_LEN`] documents for its own 0x4a/0x4c pair.
/// (This doc used to read "baseInstance ends at +0x2e", which reads as though
/// 0x38 were the fill extent and makes the constant look two bytes wrong.)
pub const ICB_DRAW_PATCHES_ARGS_LEN: u32 = 0x38;
/// DrawIndexedPatches args size (baseInstance u64 @0x42 → end 0x4a).
/// Note: `setupCommandLayout` allocates max `0x4c` for this bit; fill IMP uses through `0x4a`.
pub const ICB_DRAW_INDEXED_PATCHES_ARGS_LEN: u32 = 0x4a;
/// Mesh drawMeshThreadgroups / drawMeshThreads args size.
/// `setupCommandLayout:`: both mesh create bits take **0x48** —
/// three `MTLSize` (3×u64 each) matching Metal SPI
/// `MTLIndirectDrawMesh{Threadgroups,Threads}Arguments` field order:
/// grid / threadsPerGrid @0, object TG @0x18, mesh TG @0x30.
pub const ICB_DRAW_MESH_ARGS_LEN: u32 = 0x48;
/// SDK MTLIndirectCommandType bits (not metal-0.33's shifted ConcurrentDispatch).
pub const MTL_INDIRECT_CMD_DRAW: u32 = 1 << 0;
pub const MTL_INDIRECT_CMD_DRAW_INDEXED: u32 = 1 << 1;
pub const MTL_INDIRECT_CMD_DRAW_PATCHES: u32 = 1 << 2;
pub const MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES: u32 = 1 << 3;
pub const MTL_INDIRECT_CMD_CONCURRENT_DISPATCH: u32 = 1 << 5;
pub const MTL_INDIRECT_CMD_CONCURRENT_DISPATCH_THREADS: u32 = 1 << 6;
/// Mesh create bits (SDK). Wire args size from setupCommandLayout; fill IMPs stubbed.
pub const MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS: u32 = 1 << 7;
pub const MTL_INDIRECT_CMD_DRAW_MESH_THREADS: u32 = 1 << 8;

/// Compact type-7 TLV field: `[tag:u8][length:u8][value…]` after a field-count byte.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CompactTlv {
    pub tag: u8,
    pub length: u8,
    pub value_offset: usize,
    pub value_u32: u32,
    pub has_u32: bool,
}

/// Linear buffer descriptor (type-1): allocation size + guest page handle.
///
/// Contract: `reims_vgpu_resource_format.h` `REIMS_VGPU_RESOURCE_LINEAR_DESC_*`.
/// Backing GVA = `(handle as u64) << PAGE_SHIFT` (14 on arm64e guest).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BufferDescriptor {
    pub allocation_size: u64,
    pub handle64: u64,
    pub handle: u32,
}

/// Linear descriptor offsets (shared with type-2 texture prefix).
pub const LINEAR_DESC_MIN_LEN: usize = 16;
pub const LINEAR_DESC_SIZE: usize = 0;
pub const LINEAR_DESC_HANDLE: usize = 8;
/// Arm fixture alias of [`crate::contract::gva::PAGE_SHIFT_ARM64E`].
/// Prefer `PAGE_SHIFT_ARM64E` / `PAGE_SHIFT_X86` at new call sites. Product
/// paths must pass `DeviceState::page_shift`, not a fixed arch constant.
pub const RESOURCE_PAGE_SHIFT: u32 = crate::contract::gva::PAGE_SHIFT_ARM64E;

impl BufferDescriptor {
    /// Guest VA of buffer backing at the given page_shift, or None when
    /// size/handle/shift is invalid. Callers must pass arm (14) or x86 (12)
    /// explicitly — there is no default.
    pub fn backing_gva_size(&self, page_shift: u32) -> Option<(u64, u64)> {
        if self.allocation_size == 0 || self.handle == 0 || page_shift == 0 || page_shift > 30 {
            return None;
        }
        Some(((self.handle as u64) << page_shift, self.allocation_size))
    }
}

/// One mip level layout inside a type-2/3 texture allocation.
///
/// Level 0 comes from the geometry prefix; levels 1..N-1 are 36-byte records at
/// `TEXTURE_DESC_LEVEL_RECORDS` (offset/size/row_stride from allocation base).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TextureLevelLayout {
    /// Byte offset from allocation base (`handle << PAGE_SHIFT`).
    pub offset: u64,
    pub size: u64,
    pub row_stride: u64,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

impl TextureLevelLayout {
    /// Bytes from [`Self::offset`] that a row-by-row reader or writer of this
    /// level actually touches, given one tight row of its format.
    ///
    /// `row_stride * height` is the obvious answer and it overcounts by one
    /// row of trailing padding: the last row occupies `tight_row` bytes, and
    /// every reader in this crate walks `base + y * row_stride` for `tight_row`
    /// bytes. Used as a bound against `TextureDescriptor::allocation_size`,
    /// the looser form rejects allocations the guest sized correctly.
    ///
    /// Measured: a 27x27 `RG8Unorm` window-corner mask at offset 0x850 with a
    /// 384-byte stride in a 12 288-byte allocation scores 12 496 under
    /// `stride * height` and is refused, while its true extent is 12 166. That
    /// refusal dropped the WindowServer's whole full-screen composite draw, so
    /// the guest's rounded window corners and drop shadows rendered square.
    ///
    /// `None` for zero height (no rows) or on overflow. A bound this feeds must
    /// treat `None` as a refusal, never as "no limit".
    pub fn read_span(&self, tight_row: u32) -> Option<u64> {
        self.height
            .checked_sub(1)
            .map(u64::from)?
            .checked_mul(self.row_stride)?
            .checked_add(u64::from(tight_row))
    }

    /// How many contiguous depth planes one array slice of this level holds.
    ///
    /// `depth` 0 and 1 both mean one plane, matching `TextureDescriptor`'s
    /// "0 means 2D" encoding. **The one place that rule is written.** It used to
    /// be written three times — here as `depth.max(1)`, again in
    /// `blit_exec`'s caller as `if layout.depth == 0 { 1 } else { .. }`, and a
    /// third time inside the stride helper that caller then passed the already
    /// normalized value to — and only this copy carried the encoding it comes
    /// from. Two of the three read as a defensive clamp on a decoded guest field,
    /// which is the shape that fabricates a geometry the guest did not send the
    /// day one of them is reached without the others.
    pub fn planes(&self) -> u32 {
        self.depth.max(1)
    }

    /// Contiguous Metal packing for one array slice / cube face of this level:
    /// `row_stride * height * planes`.
    ///
    /// Zero height gives zero, which is the honest answer for a level with no
    /// rows rather than the one-row stride a `height.max(1)` would invent. The
    /// blit caller refuses a zero-height level by name (`tex_zero_geom`) before
    /// it gets here, so that arm is unreachable through it today; it is written
    /// this way so it stays right for a caller that does not.
    pub fn slice_stride(&self) -> Option<u64> {
        self.row_stride
            .checked_mul(u64::from(self.height))?
            .checked_mul(u64::from(self.planes()))
    }

    /// Bytes from [`Self::offset`] that a reader of one whole array slice /
    /// cube face of this level touches, with [`Self::planes`] packed
    /// contiguously inside the slice at `row_stride * height` each.
    ///
    /// Every plane below the last is walked in full; the last one ends at
    /// [`Self::read_span`], because the padding after its final row is no more
    /// read here than it is in the 2D case.
    ///
    /// Takes its plane count from `self` rather than from the caller. It used to
    /// take a `depth` parameter, and its only production caller passed
    /// `self.depth` with the encoding above already applied — a value the
    /// receiver holds, travelling as a loose argument that a second caller could
    /// get wrong with nothing to notice.
    pub fn slice_read_span(&self, tight_row: u32) -> Option<u64> {
        u64::from(self.planes() - 1)
            .checked_mul(self.row_stride)?
            .checked_mul(u64::from(self.height))?
            .checked_add(self.read_span(tight_row)?)
    }
}

/// Type-2/3 linear texture geometry (`REIMS_VGPU_RESOURCE_TEXTURE_DESC_*`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TextureDescriptor {
    pub allocation_size: u64,
    pub handle: u32,
    pub mipmap_level_count: u32,
    pub data_offset: u32,
    pub bytes_per_element: u8,
    pub used_size: u32,
    pub row_stride: u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub pixel_format: u16,
    /// Per-mip layouts (index 0 = L0). Empty if geometry incomplete.
    pub levels: Vec<TextureLevelLayout>,
}

impl TextureDescriptor {
    /// The extent this descriptor names, or `None` when it names none.
    ///
    /// The decoder reads each geometry field only if the record is long enough
    /// to carry it, so a short record and a record of zeroes both arrive here as
    /// zero. Neither is a texture, and neither is a 1x1 one: a caller that
    /// clamps the two fields up sizes a four-byte payload, finds any buffer long
    /// enough, and binds a single texel of whatever the read returned — which is
    /// indistinguishable from a real bind at every layer above.
    ///
    /// This replaced a `has_width` and a `has_height` field, each set to
    /// `field != 0` beside the field it described and each read as a zero check
    /// by three call sites. Storing a predicate next to its own input lets the
    /// two disagree; there is nothing here for a future decode arm to forget to
    /// set. Callers wanting only the handle or the format read those directly.
    pub fn extent(&self) -> Option<(u32, u32)> {
        if self.width == 0 || self.height == 0 {
            return None;
        }
        Some((self.width, self.height))
    }

    /// The row stride this descriptor names, or `None` when it names none.
    ///
    /// Same shape as [`Self::extent`] and for the same reason: this replaced a
    /// `has_row_stride` field assigned `row_stride != 0` two lines after
    /// `row_stride` itself, which three readers then spelled back out as a zero
    /// check. A stride of zero is a record that carried no stride, and the
    /// consequence — how far a row read may reach — is not something to guess.
    pub fn declared_row_stride(&self) -> Option<u32> {
        (self.row_stride != 0).then_some(self.row_stride)
    }

    /// The pixel format this descriptor names, or `None` when it names none.
    ///
    /// `MTLPixelFormatInvalid` is 0, so a zero is an absent format rather than
    /// a format, and the gates downstream of it fail closed. Named the same as
    /// [`TextureViewDescriptor::declared_pixel_format`] because it is the same
    /// question: the two decoders once disagreed about what an absent format
    /// meant, one storing `!= 0` and the other an unconditional `true`.
    pub fn declared_pixel_format(&self) -> Option<u16> {
        (self.pixel_format != 0).then_some(self.pixel_format)
    }

    /// Allocation base GVA (`handle << page_shift`), not including data_offset.
    /// `page_shift` must be chosen explicitly (12 or 14).
    pub fn allocation_base_gva(&self, page_shift: u32) -> Option<u64> {
        if self.handle == 0 || page_shift == 0 || page_shift > 30 {
            return None;
        }
        Some((self.handle as u64) << page_shift)
    }

    /// Level-0 texel base GVA and allocation size at the given page_shift.
    pub fn backing_gva_size(&self, page_shift: u32) -> Option<(u64, u64)> {
        if self.allocation_size == 0 || self.handle == 0 {
            return None;
        }
        let base = self.allocation_base_gva(page_shift)?;
        Some((base + self.data_offset as u64, self.allocation_size))
    }

    /// Layout for mip `level` (0-based).
    pub fn level(&self, level: u32) -> Option<&TextureLevelLayout> {
        self.levels.get(level as usize)
    }

    /// Guest VA of mip `level` texel base + layout at the given page_shift.
    pub fn level_gva(&self, level: u32, page_shift: u32) -> Option<(u64, &TextureLevelLayout)> {
        let base = self.allocation_base_gva(page_shift)?;
        let layout = self.level(level)?;
        if layout.width == 0 || layout.height == 0 || layout.row_stride == 0 {
            return None;
        }
        // Offset must leave room for at least one row within the allocation.
        if self.allocation_size != 0
            && (layout.offset >= self.allocation_size
                || self.allocation_size - layout.offset < layout.row_stride)
        {
            return None;
        }
        Some((base.checked_add(layout.offset)?, layout))
    }
}

/// Texture descriptor field offsets (geometry prefix + format trailer).
pub const TEXTURE_DESC_GEOMETRY_LEN: usize = 68;
pub const TEXTURE_DESC_MIPMAP_LEVEL_COUNT: usize = 12;
pub const TEXTURE_DESC_DATA_OFFSET: usize = 16;
pub const TEXTURE_DESC_BYTES_PER_ELEMENT: usize = 35;
pub const TEXTURE_DESC_USED_SIZE: usize = 44;
pub const TEXTURE_DESC_ROW_STRIDE: usize = 52;
pub const TEXTURE_DESC_WIDTH: usize = 60;
pub const TEXTURE_DESC_HEIGHT: usize = 64;
pub const TEXTURE_DESC_DEPTH: usize = 68;
pub const TEXTURE_DESC_LEVEL_RECORDS: usize = 72;
pub const TEXTURE_DESC_MIP_LEVEL_RECORD_LEN: usize = 36;
pub const TEXTURE_LEVEL_OFFSET: usize = 0;
pub const TEXTURE_LEVEL_SIZE: usize = 8;
pub const TEXTURE_LEVEL_ROW_STRIDE: usize = 16;
pub const TEXTURE_LEVEL_WIDTH: usize = 24;
pub const TEXTURE_LEVEL_HEIGHT: usize = 28;
pub const TEXTURE_LEVEL_DEPTH: usize = 32;
pub const TEXTURE_DESC_PIXEL_FORMAT: usize = 86;
#[cfg(test)]
pub(crate) const TEXTURE_DESC_BASE_LEN: usize = 116;
/// Mip level records this device will read from a texture descriptor.
///
/// A **corruption guard, not a capacity choice** — the same kind of bound as
/// `wire::device_desc::TYPE4_PLANE_CAP`, and it is written here because nothing
/// else in this file said so. A pyramid of 16 levels has a base of 2^15 = 32768
/// pixels in its largest dimension, which is above every Metal texture-size
/// limit, so a descriptor declaring more levels than this describes a texture
/// Metal cannot create. It is a malformed record rather than a larger one, and
/// no size of this constant makes it decodable.
///
/// So it does not bound guest work, and it is not the thing that bounds the
/// decode loop either: that loop stops on `bytes.len()`, so the descriptor's own
/// length already limits how many level records can be read. This sits above
/// that check and can only bite a declaration the payload would not have
/// satisfied anyway.
///
/// Both consumers fail visibly rather than quietly. The decoder emits
/// `texture_desc_levels_over_cap` and leaves `mipmap_level_count` at the
/// declared value while `levels` holds fewer, so a dropped level reads as a drop
/// and not as an absence. `runtime::mipmap` then refuses the generation as
/// `IncompleteLayout` — through `levels.len() < levels`, which subsumes its
/// explicit `levels > TEXTURE_MAX_MIP_LEVELS` test for exactly that reason.
pub const TEXTURE_MAX_MIP_LEVELS: usize = 16;

/// Vertex attribute from a type-7 render-pipeline vertex-input block.
///
/// The two step fields are `Option` rather than a value beside its own presence
/// bit. The serializer writes a layout's `stepFunction` and `stepRate` as tagged
/// entries and omits the tag the guest never set, so absence is a state of the
/// record — and a pair of fields that can spell "absent, 7" is a state the wire
/// cannot produce. Resolve them through [`VertexAttribute::step_rate`] and
/// [`VertexAttribute::step_function_ordinal`]; the raw `Option` is what the
/// record said, and the methods are what Metal does with it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VertexAttribute {
    pub location: u32,
    pub format: u32,
    pub offset: u32,
    pub buffer_index: u32,
    pub stride: u32,
    /// `MTLVertexStepFunction` ordinal the layout entry declared, if it did.
    pub declared_step_function: Option<u32>,
    /// `MTLVertexBufferLayoutDescriptor.stepRate` the layout entry declared.
    pub declared_step_rate: Option<u32>,
}

impl VertexAttribute {
    /// The step rate this attribute's buffer layout runs at.
    ///
    /// `MTLVertexBufferLayoutDescriptor.stepRate` defaults to 1, so a layout
    /// that declared none steps once per vertex or instance. A layout that
    /// declared **zero** means zero — that is what `MTLVertexStepFunctionConstant`
    /// pairs with — so nothing here clamps it up. Two call sites used to, which
    /// turned a constant-rate attribute into a per-instance one that advanced,
    /// while the third passed the zero through: one wire field, two answers.
    pub fn step_rate(&self) -> u32 {
        self.declared_step_rate.unwrap_or(1)
    }

    /// The `MTLVertexStepFunction` ordinal this attribute's layout runs at.
    ///
    /// `when_absent` is the caller's default because it genuinely differs: a
    /// plain vertex descriptor defaults to `PerVertex`, and a post-tessellation
    /// one indexes control points instead.
    pub fn step_function_ordinal(&self, when_absent: u32) -> u32 {
        self.declared_step_function.unwrap_or(when_absent)
    }
}

/// `MTLColorWriteMask` for one attachment, in Metal's own bit order.
///
/// A newtype rather than a bare `u32` for one reason: the value that means
/// "write every channel" is `0xf`, and the value a derived `Default` would
/// produce is `0` — which means *write nothing*. `PipelineColorAttachment` is
/// built with `..Default::default()` in the decoder and defaulted outright at
/// several call sites, so a bare field would make an omitted mask a black
/// attachment. Here the omission is unwritable: `Default` is `all`, which is
/// also what an entry that does not carry tag `0x09` means on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColorWriteMask {
    /// Private so the only ways to obtain one are [`ColorWriteMask::new`],
    /// which range-checks, and `Default`, which is `all`. A `pub` field would
    /// also make this a decode field needing its own coverage disposition,
    /// when the disposition that matters is `PipelineColorAttachment`'s.
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
    /// `None` for a value outside `MTLColorWriteMask`'s four bits — see
    /// [`ColorWriteMaskOutOfRange`], which is what the decoder reports for it.
    pub fn new(bits: u32) -> Option<Self> {
        (bits <= MTL_COLOR_WRITE_MASK_ALL).then_some(Self { bits })
    }

    pub fn bits(self) -> u32 {
        self.bits
    }
}

/// One pipeline color-attachment entry (format + blend) from the type-7 color section.
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
    /// Which channels this attachment writes. Independent of blending: a
    /// masked attachment with blending off still leaves the unwritten channels
    /// alone, so this cannot ride inside the blend state.
    pub write_mask: ColorWriteMask,
}

/// Decoded type-7 render pipeline (functions + optional stage-in attrs).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderPipelineDescriptor {
    pub object_id: u32,
    /// The fourth header word: the serialized property list's length **before**
    /// the four-byte padding the declared length includes.
    ///
    /// It was called `word3` and carried no doc for as long as it has been
    /// decoded, which made it indistinguishable from a field nobody had
    /// identified. It is not: the header is `objType`, declared length, the new
    /// object's ref, and this — and the declared length is exactly
    /// [`TYPE7_FIRST_TLVS`] plus this rounded up to four.
    ///
    /// That relation is checkable from the header alone, which is what
    /// [`note_type7_payload_len`] does. Nothing else reads the field: the walks
    /// below bound themselves on the *declared* length, which is the larger of
    /// the two and the one that describes the bytes actually present.
    pub serialized_payload_len: u32,
    pub vertex_func_ref: u32,
    pub fragment_func_ref: u32,
    /// Object-stage function ref (mesh SPI shape: tag `0x01`).
    ///
    /// Zero on classic pipelines. Mesh-only/dual-export fixtures may leave this
    /// zero and put the metallib in `vertex_func_ref` instead.
    pub object_func_ref: u32,
    /// Mesh-stage function ref (mesh SPI shape: tag `0x02`).
    ///
    /// Zero on classic pipelines. When zero, product mesh fill falls back to
    /// dual-export / mesh-only metallib in `vertex_func_ref`.
    pub mesh_func_ref: u32,
    /// Byte offset from end of 16-byte header to color-attachment section (tag 0x08).
    pub color_attachment_offset: u32,
    pub has_color_attachment_offset: bool,
    /// Byte offset from the end of the same header to the serialized
    /// `vertexDescriptor`, on the classic shape only. See
    /// [`PIPELINE_TAG_VERTEX_DESCRIPTOR_OFFSET`].
    pub vertex_descriptor_offset: u32,
    /// Whether the descriptor stated that offset. False means **no vertex
    /// descriptor**, not an unknown position; the mesh shape never states one
    /// and falls back to inferring the block.
    pub has_vertex_descriptor_offset: bool,
    /// The guest's `rasterSampleCount`. See
    /// [`PIPELINE_TAG_RASTER_SAMPLE_COUNT`].
    ///
    /// **Zero means the descriptor did not state the property**, which is the
    /// Metal default of [`DEFAULT_RASTER_SAMPLE_COUNT`]. Readers ask `> 1` rather
    /// than `!= 1` so that an absent property and a stated single-sample count
    /// are one answer, and so a defaulted `RenderPipelineDescriptor` does not
    /// read as a count no device can rasterize at.
    pub raster_sample_count: u32,
    pub vertex_attributes: Vec<VertexAttribute>,
    /// First color attachment (compat / color0).
    pub color0: PipelineColorAttachment,
    /// All color attachments with Metal slot indices (0..count-1 by entry order).
    pub color_attachments: Vec<PipelineColorAttachment>,
}

/// Compute stage-input attribute from type-7 compute pipeline compact block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComputeStageInputAttribute {
    pub raw_bits: u32,
    pub location: u32,
    pub format: u32,
    pub offset: u32,
    pub buffer_index: u32,
}

/// Compute stage-input layout from type-7 compute pipeline compact block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComputeStageInputLayout {
    pub raw_bits: u32,
    pub buffer_index: u32,
    pub step_function: u32,
    pub step_rate: u32,
    pub stride: u64,
}

/// Decoded compute `stageInputDescriptor` from a type-7 compute pipeline.
///
/// Layout is the MetalSerializer compact block after the first TLV record:
/// `word0`, `header0` (payload len + counts + index metadata), `header1`
/// (layout/attribute section offsets), then packed layout/attribute entries.
/// See `reims_vgpu_resource_format.h` COMPUTE_STAGE_INPUT_* and resource-surface-manifest.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ComputeStageInputDescriptor {
    pub word0: u32,
    pub header0: u32,
    pub header1: u32,
    pub index_type: u32,
    pub index_buffer_index: u32,
    pub attributes: Vec<ComputeStageInputAttribute>,
    pub layouts: Vec<ComputeStageInputLayout>,
    /// Attributes beyond [`MAX_COMPUTE_STAGE_INPUT_ATTRS`]. A healthy zero: the
    /// cap is the width of the wire's own count field, so no descriptor can
    /// state a count that reaches this. Nonzero refuses the pipeline
    /// (`stage_input_over_cap`), which is the reading that says the count field
    /// is wider than this decoder believes.
    pub dropped_attributes: u32,
    /// Layouts beyond [`MAX_COMPUTE_STAGE_INPUT_LAYOUTS`]; same healthy zero and
    /// same refusal as [`Self::dropped_attributes`].
    pub dropped_layouts: u32,
}

/// Decoded type-7 compute pipeline (kernel function + optional stage-input).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ComputePipelineDescriptor {
    pub kernel_func_ref: u32,
    pub stage_input: Option<ComputeStageInputDescriptor>,
}

/// Both counts are 5-bit fields of `header0`
/// ([`COMPUTE_STAGE_INPUT_HEADER0_COUNT_MASK`] is `0x1f`, applied at both
/// [`COMPUTE_STAGE_INPUT_HEADER0_ATTR_COUNT_SHIFT`] and
/// [`COMPUTE_STAGE_INPUT_HEADER0_LAYOUT_COUNT_SHIFT`]), so 31 is not a cap this
/// device chose — it is the largest number the wire can state. Sized to the
/// field, the two `dropped_*` counters below become healthy zeros that no guest
/// can make fire, which is the only bound shape that cannot lose a descriptor
/// the guest was entitled to.
///
/// It agrees with Metal on both sides, which is why the field is 5 bits:
/// `MTLStageInputOutputDescriptor.attributes` is the compute-stage counterpart
/// of `MTLVertexDescriptor.attributes` and the same 31-slot array (see
/// [`MAX_VERTEX_ATTRS`], which states it for the render stage), and `.layouts`
/// is indexed by the kernel buffer-table index a layout names — 31 slots, the
/// same number as [`crate::runtime::compute_exec::MAX_COMPUTE_BUFFER_SLOTS`].
///
/// These were 16, which is the width of the backend's mirror array and nothing
/// else — and that array is sized *from* here. At 16 a descriptor naming 17
/// attributes did not lose only the 17th: crossing the cap dropped the entire
/// stage-input, so a kernel that fetches per-thread `stage_in` became one that
/// declares none.
pub const MAX_COMPUTE_STAGE_INPUT_ATTRS: usize = 31;
/// The layout half of [`MAX_COMPUTE_STAGE_INPUT_ATTRS`]; same wire field width,
/// same reasoning.
pub const MAX_COMPUTE_STAGE_INPUT_LAYOUTS: usize = 31;
// Both caps *are* the count field's range. Pinned here rather than in a test
// because the two must not be able to disagree at all: widening the field
// without widening the caps silently reintroduces the drop.
const _: () =
    assert!(MAX_COMPUTE_STAGE_INPUT_ATTRS == COMPUTE_STAGE_INPUT_HEADER0_COUNT_MASK as usize);
const _: () =
    assert!(MAX_COMPUTE_STAGE_INPUT_LAYOUTS == COMPUTE_STAGE_INPUT_HEADER0_COUNT_MASK as usize);

// MetalSerializer compute stage-input compact block (offsets relative to block start).
pub const COMPUTE_STAGE_INPUT_WORD0: usize = 0;
pub const COMPUTE_STAGE_INPUT_HEADER0: usize = 4;
pub const COMPUTE_STAGE_INPUT_HEADER1: usize = 8;
pub const COMPUTE_STAGE_INPUT_MIN_LEN: usize = 12;
pub const COMPUTE_STAGE_INPUT_HEADER0_LEN_MASK: u32 = 0xffff;
pub const COMPUTE_STAGE_INPUT_HEADER0_INDEX_TYPE_SHIFT: u32 = 16;
pub const COMPUTE_STAGE_INPUT_HEADER0_INDEX_TYPE_MASK: u32 = 0x1;
pub const COMPUTE_STAGE_INPUT_HEADER0_INDEX_BUFFER_SHIFT: u32 = 17;
pub const COMPUTE_STAGE_INPUT_HEADER0_INDEX_BUFFER_MASK: u32 = 0x1f;
pub const COMPUTE_STAGE_INPUT_HEADER0_ATTR_COUNT_SHIFT: u32 = 22;
pub const COMPUTE_STAGE_INPUT_HEADER0_LAYOUT_COUNT_SHIFT: u32 = 27;
pub const COMPUTE_STAGE_INPUT_HEADER0_COUNT_MASK: u32 = 0x1f;
pub const COMPUTE_STAGE_INPUT_HEADER1_LAYOUT_OFFSET_MASK: u32 = 0xffff;
pub const COMPUTE_STAGE_INPUT_HEADER1_ATTR_OFFSET_SHIFT: u32 = 16;
/// Offsets in header1 are relative to header0 (not word0).
pub const COMPUTE_STAGE_INPUT_HEADER1_OFFSET_BASE: usize = COMPUTE_STAGE_INPUT_HEADER0;
pub const COMPUTE_STAGE_INPUT_LAYOUT_ENTRY_SIZE: usize = 16;
pub const COMPUTE_STAGE_INPUT_LAYOUT_BITS_BUFFER_MASK: u32 = 0x1f;
pub const COMPUTE_STAGE_INPUT_LAYOUT_BITS_STEP_SHIFT: u32 = 5;
pub const COMPUTE_STAGE_INPUT_LAYOUT_BITS_STEP_MASK: u32 = 0x1f;
pub const COMPUTE_STAGE_INPUT_LAYOUT_STEP_RATE: usize = 4;
pub const COMPUTE_STAGE_INPUT_LAYOUT_STRIDE: usize = 8;
pub const COMPUTE_STAGE_INPUT_ATTR_ENTRY_SIZE: usize = 8;
pub const COMPUTE_STAGE_INPUT_ATTR_BITS_LOCATION_MASK: u32 = 0x1f;
pub const COMPUTE_STAGE_INPUT_ATTR_BITS_BUFFER_SHIFT: u32 = 5;
pub const COMPUTE_STAGE_INPUT_ATTR_BITS_BUFFER_MASK: u32 = 0x1f;
pub const COMPUTE_STAGE_INPUT_ATTR_BITS_FORMAT_SHIFT: u32 = 10;
pub const COMPUTE_STAGE_INPUT_ATTR_BITS_FORMAT_MASK: u32 = 0x3f;
pub const COMPUTE_STAGE_INPUT_ATTR_OFFSET: usize = 4;

/// Which bits of each packed word above a reader consumes.
///
/// A bit-packed word is the same blind spot as a TLV entry read one named tag at
/// a time, one level down and harder to see: a field is `(word >> shift) & mask`
/// at its own site, and no site is in a position to notice that some bits are
/// named by no field at all. `note_pipeline_tlv_fields` exists because the tag
/// form had that hole; these exist because this form has it too.
///
/// The two headers **tile their word exactly** — no gap, no overlap — and that
/// is pinned below by `const` assertion rather than by a runtime line, because
/// it is a property of the constants and a field that leaves a hole should fail
/// the build rather than a boot. `const` and not a `#[test]` for the reason
/// `MAX_COMPUTE_STAGE_INPUT_ATTRS` gives above: this file compiles on the Metal
/// arm, where its tests do not run.
///
/// The two *entry* words do **not** tile. The layout entry names 10 of 32 bits
/// and the attribute entry 16, so 22 and 16 bits respectively reach this decoder
/// with no reader. That is a runtime question rather than a build one, because
/// what matters is whether a guest ever *sets* one — see [`note_unread_bits`].
const COMPUTE_STAGE_INPUT_HEADER0_READ: u32 = COMPUTE_STAGE_INPUT_HEADER0_LEN_MASK
    | (COMPUTE_STAGE_INPUT_HEADER0_INDEX_TYPE_MASK << COMPUTE_STAGE_INPUT_HEADER0_INDEX_TYPE_SHIFT)
    | (COMPUTE_STAGE_INPUT_HEADER0_INDEX_BUFFER_MASK
        << COMPUTE_STAGE_INPUT_HEADER0_INDEX_BUFFER_SHIFT)
    | (COMPUTE_STAGE_INPUT_HEADER0_COUNT_MASK << COMPUTE_STAGE_INPUT_HEADER0_ATTR_COUNT_SHIFT)
    | (COMPUTE_STAGE_INPUT_HEADER0_COUNT_MASK << COMPUTE_STAGE_INPUT_HEADER0_LAYOUT_COUNT_SHIFT);
const _: () = assert!(COMPUTE_STAGE_INPUT_HEADER0_READ == u32::MAX);
// Not covered by the line above on its own: five fields that *overlap* can still
// OR to all-ones. Summing them proves each bit is claimed by exactly one field,
// which is the half that breaks if a shift moves.
const _: () = assert!(
    (COMPUTE_STAGE_INPUT_HEADER0_LEN_MASK as u64)
        + ((COMPUTE_STAGE_INPUT_HEADER0_INDEX_TYPE_MASK
            << COMPUTE_STAGE_INPUT_HEADER0_INDEX_TYPE_SHIFT) as u64)
        + ((COMPUTE_STAGE_INPUT_HEADER0_INDEX_BUFFER_MASK
            << COMPUTE_STAGE_INPUT_HEADER0_INDEX_BUFFER_SHIFT) as u64)
        + ((COMPUTE_STAGE_INPUT_HEADER0_COUNT_MASK << COMPUTE_STAGE_INPUT_HEADER0_ATTR_COUNT_SHIFT)
            as u64)
        + ((COMPUTE_STAGE_INPUT_HEADER0_COUNT_MASK
            << COMPUTE_STAGE_INPUT_HEADER0_LAYOUT_COUNT_SHIFT) as u64)
        == u32::MAX as u64
);
// `header1` is two halves and the upper one is taken by a bare shift with no
// mask, so it claims everything above the shift by construction.
const _: () = assert!(
    COMPUTE_STAGE_INPUT_HEADER1_LAYOUT_OFFSET_MASK
        == (1u32 << COMPUTE_STAGE_INPUT_HEADER1_ATTR_OFFSET_SHIFT) - 1
);

/// Bits of a compute stage-input **layout** entry's packed word with a reader:
/// the buffer index and the step function. Twenty-two above them have none.
const COMPUTE_STAGE_INPUT_LAYOUT_BITS_READ: u32 = COMPUTE_STAGE_INPUT_LAYOUT_BITS_BUFFER_MASK
    | (COMPUTE_STAGE_INPUT_LAYOUT_BITS_STEP_MASK << COMPUTE_STAGE_INPUT_LAYOUT_BITS_STEP_SHIFT);

/// Bits of a compute stage-input **attribute** entry's packed word with a
/// reader: the location, the buffer index and the format. Sixteen above them
/// have none.
const COMPUTE_STAGE_INPUT_ATTR_BITS_READ: u32 = COMPUTE_STAGE_INPUT_ATTR_BITS_LOCATION_MASK
    | (COMPUTE_STAGE_INPUT_ATTR_BITS_BUFFER_MASK << COMPUTE_STAGE_INPUT_ATTR_BITS_BUFFER_SHIFT)
    | (COMPUTE_STAGE_INPUT_ATTR_BITS_FORMAT_MASK << COMPUTE_STAGE_INPUT_ATTR_BITS_FORMAT_SHIFT);

/// Type-8 texture view (base texture + optional format/level/slice/swizzle).
///
/// Which fields the wire carried is a property of `view_opcode` and nothing
/// else: the three forms are three distinct records, and the swizzle body
/// embeds the ranged one whole. So the question is asked of the opcode through
/// [`Self::carries_range`] and [`Self::carries_swizzle`], rather than answered
/// by booleans set beside the fields they describe. Five such booleans used to
/// live here, written at all three decode sites — fifteen assignments that
/// could disagree with the opcode, and one of them (`has_pixel_format`) that no
/// consumer ever read, because both of them spelled its rule again themselves.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TextureViewDescriptor {
    pub view_opcode: u32,
    pub view_texture_ref: u32,
    pub base_texture_ref: u32,
    pub pixel_format: u16,
    /// `MTLTextureType`, on the forms [`Self::carries_range`] names (`@18`).
    pub texture_type: u16,
    pub level_base: u64,
    pub level_count: u64,
    pub slice_base: u64,
    pub slice_count: u64,
    pub swizzle: [u8; 4],
}

impl TextureViewDescriptor {
    /// Whether this form carries `texture_type`, the level range and the slice
    /// range — the ranged body, which the swizzle form embeds whole.
    ///
    /// The three travel together because they are one record. Reading them off
    /// a simple view would take `MTLTextureType1D` (which is 0) and a
    /// zero-length level range for things the guest asked for.
    pub fn carries_range(&self) -> bool {
        matches!(
            self.view_opcode,
            TEXTURE_VIEW_OPCODE_RANGED | TEXTURE_VIEW_OPCODE_SWIZZLE
        )
    }

    /// Whether this form carries per-channel swizzles.
    pub fn carries_swizzle(&self) -> bool {
        self.view_opcode == TEXTURE_VIEW_OPCODE_SWIZZLE
    }

    /// The view's own pixel format, where it declared one.
    ///
    /// All three forms carry the field, and `MTLPixelFormatInvalid` is 0 — so a
    /// zero is a view that reinterprets nothing and keeps its base texture's
    /// format, which is why this is the one presence question the opcode does
    /// not answer.
    pub fn declared_pixel_format(&self) -> Option<u16> {
        (self.pixel_format != 0).then_some(self.pixel_format)
    }
}

// The record header every type-8 blob starts with. Named here because this
// file reads it on five different records, and derived from the wire crate's
// `OpHeader` so the two words cannot be swapped in one place and not the other.
pub const TEXTURE_VIEW_DESC_OPCODE: usize = offset_of!(reims_vgpu_wire::OpHeader, opcode);
pub const TEXTURE_VIEW_DESC_LEN: usize = offset_of!(reims_vgpu_wire::OpHeader, length);

// The three texture-view forms. Every offset is `offset_of!` on the wire
// crate's struct field, so a field it renames fails this build rather than
// leaving two readings that agree only by habit — the same treatment the heap
// and buffer-backed records below get.
//
// The `*_MIN_*` names are historical: each is the record's *total* length, not
// a floor. Apple's serializer writes exactly one length per opcode, which is
// what the wire crate's `*_TOTAL_LEN` names say.
#[cfg(test)]
pub(crate) const TEXTURE_VIEW_DESC_TEXTURE_REF: usize =
    OP_HDR + offset_of!(w_view::TextureViewBody, object_ref);
#[cfg(test)]
pub(crate) const TEXTURE_VIEW_DESC_BASE_REF: usize =
    OP_HDR + offset_of!(w_view::TextureViewBody, base_texture_ref);
#[cfg(test)]
pub(crate) const TEXTURE_VIEW_DESC_PIXEL_FORMAT: usize =
    OP_HDR + offset_of!(w_view::TextureViewBody, pixel_format);
#[cfg(test)]
pub(crate) const TEXTURE_VIEW_DESC_TEXTURE_TYPE: usize =
    OP_HDR + offset_of!(w_view::TextureViewRangedBody, texture_type);
#[cfg(test)]
pub(crate) const TEXTURE_VIEW_DESC_LEVEL_BASE: usize =
    OP_HDR + offset_of!(w_view::TextureViewRangedBody, level_base);
#[cfg(test)]
pub(crate) const TEXTURE_VIEW_DESC_LEVEL_COUNT: usize =
    OP_HDR + offset_of!(w_view::TextureViewRangedBody, level_count);
#[cfg(test)]
pub(crate) const TEXTURE_VIEW_DESC_SLICE_BASE: usize =
    OP_HDR + offset_of!(w_view::TextureViewRangedBody, slice_base);
#[cfg(test)]
pub(crate) const TEXTURE_VIEW_DESC_SLICE_COUNT: usize =
    OP_HDR + offset_of!(w_view::TextureViewRangedBody, slice_count);
#[cfg(test)]
pub(crate) const TEXTURE_VIEW_DESC_SWIZZLE: usize =
    OP_HDR + offset_of!(w_view::TextureViewSwizzleBody, swizzle);
pub const TEXTURE_VIEW_MIN_SIMPLE: usize = w_view::TEXTURE_VIEW_TOTAL_LEN as usize;
pub const TEXTURE_VIEW_MIN_RANGED: usize = w_view::TEXTURE_VIEW_RANGED_TOTAL_LEN as usize;
pub const TEXTURE_VIEW_MIN_SWIZZLE: usize = w_view::TEXTURE_VIEW_SWIZZLE_TOTAL_LEN as usize;
pub const TEXTURE_VIEW_OPCODE_SIMPLE: u32 = w_view::OPCODE_TEXTURE_VIEW;
pub const TEXTURE_VIEW_OPCODE_RANGED: u32 = w_view::OPCODE_TEXTURE_VIEW_RANGED;
pub const TEXTURE_VIEW_OPCODE_SWIZZLE: u32 = w_view::OPCODE_TEXTURE_VIEW_SWIZZLE;
// Heap-backed texture (`newTextureWithDescriptor:heap:offset:useOffset:
// allocator:`). It shares the type-8 object tag, but is a complete texture
// resource rather than a view: a heap ref, the embedded
// PGSerializedTextureDescriptor, then `useOffset` and the heap byte offset.
//
// Every offset below is `offset_of!` on the wire crate's struct rather than a
// number written again here, so a field it renames fails this build instead of
// leaving two readings that agree only by habit.
pub const HEAP_TEXTURE_OPCODE: u32 = w_heap::OPCODE_NEW_HEAP_TEXTURE;
pub const HEAP_TEXTURE_LEN: usize = w_heap::NEW_HEAP_TEXTURE_TOTAL_LEN as usize;
#[cfg(test)]
pub(crate) const HEAP_TEXTURE_HEAP_REF: usize =
    OP_HDR + offset_of!(w_heap::NewHeapTextureBody, heap_ref);
#[cfg(test)]
pub(crate) const HEAP_TEXTURE_DESCRIPTOR: usize =
    OP_HDR + offset_of!(w_heap::NewHeapTextureBody, desc);
pub const HEAP_TEXTURE_USE_OFFSET: usize =
    OP_HDR + offset_of!(w_heap::NewHeapTextureBody, use_offset_bits);
pub const HEAP_TEXTURE_OFFSET: usize = OP_HDR + offset_of!(w_heap::NewHeapTextureBody, offset);

// The same record once the guest's serializer has `TextureDescriptor2` on. It
// is a different opcode, not a longer one, and every field after the heap ref
// moves by the eight bytes the wide descriptor adds.
pub const HEAP_TEXTURE_WIDE_OPCODE: u32 = w_heap::OPCODE_NEW_HEAP_TEXTURE_WIDE;
pub const HEAP_TEXTURE_WIDE_LEN: usize = w_heap::NEW_HEAP_TEXTURE_WIDE_TOTAL_LEN as usize;
#[cfg(test)]
const HEAP_TEXTURE_WIDE_HEAP_REF: usize =
    OP_HDR + offset_of!(w_heap::NewHeapTextureWideBody, heap_ref);
#[cfg(test)]
const HEAP_TEXTURE_WIDE_DESCRIPTOR: usize =
    OP_HDR + offset_of!(w_heap::NewHeapTextureWideBody, desc);
#[cfg(test)]
const HEAP_TEXTURE_WIDE_USE_OFFSET: usize =
    OP_HDR + offset_of!(w_heap::NewHeapTextureWideBody, use_offset_bits);
#[cfg(test)]
const HEAP_TEXTURE_WIDE_OFFSET: usize = OP_HDR + offset_of!(w_heap::NewHeapTextureWideBody, offset);

// Opcode 9 is NOT a view: it is a buffer-backed texture (`newTextureWithBuffer:
// descriptor:offset:bytesPerRow:`) serialized by `-[PGSerializer newTextureWith
// Buffer:...]`.
// It shares only the type-8 object tag + 16-byte header (opcode@0, len@4,
// self-ref@8, source-ref@0xc); the source ref @0xc is a BUFFER, not a texture,
// and the body is {u64 offset, u64 bytesPerRow, embedded MTLTextureDescriptor}.
pub const TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE: u32 = w_backed::OPCODE_BUFFER_TEXTURE;
#[cfg(test)]
const BUF_TEX_DESC_BUFFER_REF: usize = OP_HDR + offset_of!(w_backed::BufferTextureBody, buffer_ref);
#[cfg(test)]
const BUF_TEX_DESC_OFFSET: usize = OP_HDR + offset_of!(w_backed::BufferTextureBody, offset);
#[cfg(test)]
const BUF_TEX_DESC_BYTES_PER_ROW: usize =
    OP_HDR + offset_of!(w_backed::BufferTextureBody, bytes_per_row);
// The embedded `PGSerializedTextureDescriptor` is not named here at all: there
// is one decoder for it and everything inside it is at that decoder's own
// offsets. The seven that used to be named here — flags, width, height, depth,
// mip count, sample count, array length — were a second copy of a layout
// `heap_query` already had, and a second copy is a second thing to get wrong.
pub const BUF_TEX_MIN_LEN: usize = w_backed::BUFFER_TEXTURE_TOTAL_LEN as usize;

// The buffer-backed record's `TextureDescriptor2` form. The three fields before
// the descriptor keep their offsets; only the descriptor widens.
pub const TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE_WIDE: u32 = w_backed::OPCODE_BUFFER_TEXTURE_WIDE;
#[cfg(test)]
const BUF_TEX_WIDE_DESC_BODY: usize = OP_HDR + offset_of!(w_backed::BufferTextureWideBody, desc);
pub const BUF_TEX_WIDE_LEN: usize = w_backed::BUFFER_TEXTURE_WIDE_TOTAL_LEN as usize;
// MTLTextureType values (Metal.framework Headers/MTLTextureType.h).
pub const TEXTURE_VIEW_MTL_TYPE_1D: u16 = 0;
pub const TEXTURE_VIEW_MTL_TYPE_1D_ARRAY: u16 = 1;
pub const TEXTURE_VIEW_MTL_TYPE_2D: u16 = 2;
pub const TEXTURE_VIEW_MTL_TYPE_2D_ARRAY: u16 = 3;
#[cfg(test)]
pub(crate) const TEXTURE_VIEW_MTL_TYPE_2D_MULTISAMPLE: u16 = 4;
pub const TEXTURE_VIEW_MTL_TYPE_CUBE: u16 = 5;
pub const TEXTURE_VIEW_MTL_TYPE_CUBE_ARRAY: u16 = 6;
pub const TEXTURE_VIEW_MTL_TYPE_3D: u16 = 7;

/// Whether a type-8 view `texture_type` is supported for product-path blit/sample.
pub fn texture_view_type_supported(texture_type: u16) -> bool {
    matches!(
        texture_type,
        TEXTURE_VIEW_MTL_TYPE_1D
            | TEXTURE_VIEW_MTL_TYPE_1D_ARRAY
            | TEXTURE_VIEW_MTL_TYPE_2D
            | TEXTURE_VIEW_MTL_TYPE_2D_ARRAY
            | TEXTURE_VIEW_MTL_TYPE_CUBE
            | TEXTURE_VIEW_MTL_TYPE_CUBE_ARRAY
            | TEXTURE_VIEW_MTL_TYPE_3D
    )
}

/// Types that use the Metal array-slice dimension (not 3D depth).
pub fn texture_view_type_uses_slices(texture_type: u16) -> bool {
    matches!(
        texture_type,
        TEXTURE_VIEW_MTL_TYPE_1D_ARRAY
            | TEXTURE_VIEW_MTL_TYPE_2D_ARRAY
            | TEXTURE_VIEW_MTL_TYPE_CUBE
            | TEXTURE_VIEW_MTL_TYPE_CUBE_ARRAY
    )
}

/// 3D volume type (uses z depth planes; array slice must be 0).
pub fn texture_view_type_is_3d(texture_type: u16) -> bool {
    texture_type == TEXTURE_VIEW_MTL_TYPE_3D
}

// Colour-attachment TLV tags. `0x01..=0x09` are `MTLRenderPipelineColorAttach\
// mentDescriptor`'s nine properties in the order `MTLRenderPipeline.h` declares
// them — pixelFormat, blendingEnabled, sourceRGBBlendFactor,
// destinationRGBBlendFactor, rgbBlendOperation, sourceAlphaBlendFactor,
// destinationAlphaBlendFactor, alphaBlendOperation, writeMask — so the tag is
// the property's one-based header index. Tag `0x00` sits before the first
// property and is not one; it rides every entry with value 0 in every workload
// measured and is reported by `note_color_entry_fields` as unconsumed.
/// Which `colorAttachments[n]` this entry configures.
///
/// Tag `0x00` is the entry's own index in all three sections this serializer
/// emits in this shape: [`VERTEX_ATTR_TAG_LOCATION`] is the attribute's
/// location and [`VERTEX_LAYOUT_TAG_BUFFER_INDEX`] is the layout's buffer
/// index, both read from the wire here. It is outside the property numbering
/// for the same reason — tags `0x01..=0x09` are the nine properties of
/// `MTLRenderPipelineColorAttachmentDescriptor` in header order, so there is no
/// property left for `0x00` to be.
pub const COLOR_ATTACHMENT_TAG_INDEX: u8 = 0x00;
pub const COLOR_ATTACHMENT_TAG_PIXEL_FORMAT: u8 = 0x01;
pub const COLOR_ATTACHMENT_TAG_BLEND_ENABLE: u8 = 0x02;
pub const COLOR_ATTACHMENT_TAG_SRC_RGB: u8 = 0x03;
pub const COLOR_ATTACHMENT_TAG_DST_RGB: u8 = 0x04;
pub const COLOR_ATTACHMENT_TAG_RGB_OP: u8 = 0x05;
pub const COLOR_ATTACHMENT_TAG_SRC_ALPHA: u8 = 0x06;
pub const COLOR_ATTACHMENT_TAG_DST_ALPHA: u8 = 0x07;
pub const COLOR_ATTACHMENT_TAG_ALPHA_OP: u8 = 0x08;
/// `MTLColorWriteMask`, the ninth and last property.
///
/// Read off a live x86/Vulkan guest on 2026-07-30: the tag appears with
/// `len=4 value=1` on a pipeline whose entry is `[00, 01, 02, 06, 09]`, and
/// `value=1` is [`MTL_COLOR_WRITE_MASK_ALPHA`] — an alpha-only attachment,
/// which is how a compositor punches a shape into a surface's alpha without
/// touching its colour. Serialized entries omit properties left at their
/// default, which is why only the one non-`all` mask in that boot appeared.
pub const COLOR_ATTACHMENT_TAG_WRITE_MASK: u8 = 0x09;
pub const BLEND_FACTOR_ZERO: u32 = 0;
pub const BLEND_FACTOR_ONE: u32 = 1;
pub const BLEND_OP_ADD: u32 = 0;

// `MTLColorWriteMask` (Metal.framework Headers/MTLRenderPipeline.h). The bits
// run alpha-first from the low end, which is the reverse of the RGBA reading
// order the name suggests — `Red` is `1 << 3`, not `1 << 0`.
//
// This is an SDK mirror, so the table is the whole enum and stays `pub` even
// where a member has no reader. `_NONE` has one on the Vulkan arm only, and
// gating it on that arm would make the mirror's completeness depend on which
// backend is compiled — which is the property a mirror exists to not have.
pub const MTL_COLOR_WRITE_MASK_NONE: u32 = 0;
pub const MTL_COLOR_WRITE_MASK_ALPHA: u32 = 1 << 0;
pub const MTL_COLOR_WRITE_MASK_BLUE: u32 = 1 << 1;
pub const MTL_COLOR_WRITE_MASK_GREEN: u32 = 1 << 2;
pub const MTL_COLOR_WRITE_MASK_RED: u32 = 1 << 3;
pub const MTL_COLOR_WRITE_MASK_ALL: u32 = 0xf;

/// Where the sampler-creation record (type-7 subtype 0x03) puts each field, for
/// the synthetic buffers the tests below assemble.
///
/// Derived from the view `decode_sampler_descriptor` actually reads. These were
/// eight literals from a ported C header, and `ops::sampler`'s module doc says
/// the two derivations agree — which was worth stating precisely because until
/// the fixtures existed nothing had compared them. Deriving them is how that
/// stays true without anyone re-checking it.
///
/// Two of them were also *anonymous*: `SAMPLER_DESC_WORD16` and `_WORD20` named
/// their own offsets because the C header did not know what was there. The
/// oracle does — `flags`, whose low nibble is the only written part, and
/// `lodMinClamp` — so they are named for the fields now.
///
/// The cfg is their one consumer's, `icb::tests::put_type7_sampler`, which
/// builds a sampler for the Metal ICB encoder and is gated the same way. Naming
/// that cfg here rather than reaching for `allow(dead_code)` is what keeps the
/// Vulkan arm able to say these are unreferenced if the consumer ever goes.
#[cfg(all(test, feature = "backend-metal", target_os = "macos"))]
pub(crate) mod sampler_desc {
    use super::{offset_of, w_smp, OP_HDR};

    pub(crate) const LEN: usize = w_smp::NEW_SAMPLER_TOTAL_LEN as usize;
    pub(crate) const TAG: usize = offset_of!(reims_vgpu_wire::OpHeader, opcode);
    pub(crate) const DECLARED_LEN: usize = offset_of!(reims_vgpu_wire::OpHeader, length);
    pub(crate) const ID: usize = OP_HDR + offset_of!(w_smp::SamplerBody, object_ref);
    pub(crate) const STATE_BITS: usize = OP_HDR + offset_of!(w_smp::SamplerBody, state);
    pub(crate) const FLAGS: usize = OP_HDR + offset_of!(w_smp::SamplerBody, flags);
    pub(crate) const LOD_MIN: usize = OP_HDR + offset_of!(w_smp::SamplerBody, lod_min_clamp);
    pub(crate) const LOD_MAX: usize = OP_HDR + offset_of!(w_smp::SamplerBody, lod_max_clamp);
}

/// Live function descriptor (reims_vgpu_resource_format.h).
pub const FUNCTION_DESC_BLOB_GVA: usize = 0;
pub const FUNCTION_DESC_BLOB_SIZE: usize = 8;
pub const FUNCTION_DESC_FUNCTION_ID: usize = 0x14;
pub const FUNCTION_DESC_MIN_LEN: usize = 12;

/// Compact first-subrecord tags (u8) on type-7 pipelines.
pub const PIPELINE_TAG_KERNEL_FUNC: u8 = 0x00;
/// Classic: vertex function. Mesh SPI: object function.
pub const PIPELINE_TAG_VERTEX_FUNC: u8 = 0x01;
/// Classic: fragment function. Mesh SPI: mesh function.
pub const PIPELINE_TAG_FRAGMENT_FUNC: u8 = 0x02;
/// Mesh SPI only: fragment function (classic tag 0x03 is a different field).
pub const PIPELINE_TAG_MESH_FRAGMENT_FUNC: u8 = 0x03;
/// Classic: where the serialized `vertexDescriptor` starts, in the same units as
/// [`PIPELINE_TAG_COLOR_ATTACH_OFFSET`] — a byte offset from the end of the
/// 16-byte type-7 header.
///
/// The same wire tag as [`PIPELINE_TAG_MESH_FRAGMENT_FUNC`], whose role it takes
/// on the mesh shape. The pair is the third instance of this file's standing
/// rule that a tag's meaning is a property of the shape it arrives in: `0x01`
/// and `0x02` are already vertex/fragment on one and object/mesh on the other.
///
/// **A classic descriptor without this tag has no vertex descriptor at all**,
/// rather than one whose offset went missing. A driven boot shows the shape
/// `[08,01,02]` producing no vertex-descriptor entry and every shape carrying
/// `0x03` producing one, which is the reading that licenses treating absence as
/// "none" rather than as "look for it".
///
/// Reading it retires a guess. Before it was identified, the block was assumed
/// to be everything between the end of the TLV block and the colour section, and
/// that assumption needs [`skip_optional_label_and_pad`] to step over a `label`
/// string of unknown length — a heuristic that misreads any `fieldCount` of
/// `0x20` or above as the first character of a label. See
/// [`note_vertex_block_inferred`], which is what still measures the mesh arm's
/// reliance on it.
pub const PIPELINE_TAG_VERTEX_DESCRIPTOR_OFFSET: u8 = PIPELINE_TAG_MESH_FRAGMENT_FUNC;
/// `MTLRenderPipelineDescriptor.rasterSampleCount` — how many samples each
/// fragment is rasterized at.
///
/// Present on both shapes: it is a property of the descriptor rather than of
/// either role map, so it is in both `*_TAGS_CONSUMED` lists.
///
/// **A property equal to a freshly-initialised descriptor's is omitted from the
/// block entirely**, which is the grammar rule behind every ragged shape this
/// decoder sees, and it makes this tag's presence meaningful on its own: the
/// Metal default is one sample, so a descriptor that states this property is
/// asking for something other than single-sampling. The value is still what is
/// read — [`DEFAULT_RASTER_SAMPLE_COUNT`] and absence are treated alike — because
/// acting on presence would make the decode depend on the encoder's omission
/// rule rather than on the number the guest sent.
pub const PIPELINE_TAG_RASTER_SAMPLE_COUNT: u8 = 0x04;
/// Offset (from header end) to color-attachment section; vertex block lives before it.
pub const PIPELINE_TAG_COLOR_ATTACH_OFFSET: u8 = 0x08;
/// `MTLRenderPipelineDescriptor.depthAttachmentPixelFormat`, an `MTLPixelFormat`.
///
/// A compile-time compatibility declaration rather than an instruction: Metal
/// requires a pipeline's declared attachment formats to match the render pass it
/// is used with, so this restates what the pass already carries. See
/// [`RENDER_PIPELINE_TAGS_BENIGN`] for why this device does not apply it.
pub const PIPELINE_TAG_DEPTH_ATTACH_FORMAT: u8 = 0x09;
/// `MTLRenderPipelineDescriptor.stencilAttachmentPixelFormat`, an
/// `MTLPixelFormat`. The stencil half of [`PIPELINE_TAG_DEPTH_ATTACH_FORMAT`],
/// and benign for the same reason.
pub const PIPELINE_TAG_STENCIL_ATTACH_FORMAT: u8 = 0x0a;
/// The `rasterSampleCount` a descriptor that omits the property is asking for,
/// and the only count this device rasterizes at.
///
/// Every render target the backends allocate is single-sampled, so this is not a
/// policy choice: a pipeline built at any other count would not be compatible
/// with the pass it is used in.
pub const DEFAULT_RASTER_SAMPLE_COUNT: u32 = 1;
/// Mesh SPI section offset (analog of classic [`PIPELINE_TAG_COLOR_ATTACH_OFFSET`]).
///
/// Live host Metal `-[_MTLDevice serializeMeshRenderPipelineDescriptor:]`
/// differentials (2026-07-12, Apple M3 Max): same compact first-subrecord
/// grammar as classic type-7 (`[fieldCount]×[tag][0x04][u32]`). Presence of
/// this tag selects the mesh role map for tags 0x01/0x02/0x03.
pub const PIPELINE_TAG_MESH_SECTION_OFFSET: u8 = 0x14;
/// Mesh object-stage function — same wire tag as classic vertex (`0x01`).
#[cfg(test)]
pub(crate) const PIPELINE_TAG_OBJECT_FUNC: u8 = PIPELINE_TAG_VERTEX_FUNC;
/// Mesh mesh-stage function — same wire tag as classic fragment (`0x02`).
#[cfg(test)]
pub(crate) const PIPELINE_TAG_MESH_FUNC: u8 = PIPELINE_TAG_FRAGMENT_FUNC;

pub const VERTEX_DESC_TAG_ATTRIBUTES: u8 = 0x00;
pub const VERTEX_DESC_TAG_LAYOUTS: u8 = 0x01;
pub const VERTEX_ATTR_TAG_LOCATION: u8 = 0x00;
pub const VERTEX_ATTR_TAG_FORMAT: u8 = 0x01;
pub const VERTEX_ATTR_TAG_OFFSET: u8 = 0x02;
pub const VERTEX_ATTR_TAG_BUFFER_INDEX: u8 = 0x03;
pub const VERTEX_LAYOUT_TAG_BUFFER_INDEX: u8 = 0x00;
pub const VERTEX_LAYOUT_TAG_STEP_FUNCTION: u8 = 0x01;
pub const VERTEX_LAYOUT_TAG_STEP_RATE: u8 = 0x02;
pub const VERTEX_LAYOUT_TAG_STRIDE: u8 = 0x03;

/// The three consumed-tag sets for the vertex-descriptor walks, each listing
/// exactly what its own reader names.
///
/// Each of these entries is read by `entry_tag_u32`, which walks the entry once
/// per tag the caller asks for. A reader written that way never forms a list of
/// what the entry actually held, so it structurally cannot notice a tag it does
/// not ask for — which is why the sets are written out here and handed to
/// [`note_entry_tlv_fields`] rather than left implicit at the call sites.
///
/// All three look complete against the Metal types they mirror:
/// `MTLVertexDescriptor` has `attributes` and `layouts`;
/// `MTLVertexAttributeDescriptor` has `format`, `offset` and `bufferIndex` past
/// its subscript; `MTLVertexBufferLayoutDescriptor` has `stride`,
/// `stepFunction` and `stepRate` past its own. "Looks complete" is exactly the
/// claim this instrument exists to replace — the pipeline block one level up
/// looked complete too, and a driven boot found four tags in it with no reader.
///
/// # The reading, x86/Vulkan, one driven boot (Safari window drag)
///
/// Six distinct entry shapes, **every one `unconsumed=0`**:
///
/// ```text
/// kind=vertex_desc    tags=[00:4,01:4]
/// kind=vertex_layout  tags=[00:4,03:4]
/// kind=vertex_attr    tags=[00:4,01:4]
/// kind=vertex_attr    tags=[00:4,01:4,02:4]
/// kind=vertex_attr    tags=[00:4,01:4,03:4]
/// kind=vertex_attr    tags=[00:4,01:4,02:4,03:4]
/// ```
///
/// So these three walks read what this guest sends, and that is now a
/// measurement rather than an inference from the Metal headers. It is the
/// answer the pipeline block did *not* give, which is the point of asking both
/// with one instrument.
///
/// Two things the shapes say past the zero. Attribute entries **omit** a tag
/// whose value is the Metal default rather than sending a zero, so
/// `entry_tag_u32`'s defaults are load-bearing on a live guest rather than a
/// fallback for malformed records. And the layout entry never carries `0x01` or
/// `0x02` at all on this workload — this guest states no step function and no
/// step rate — which is why `declared_step_function` and `declared_step_rate`
/// are `Option` and why a declared zero had to stay distinguishable from an
/// absent tag.
const VERTEX_DESC_TAGS_CONSUMED: [u8; 2] = [VERTEX_DESC_TAG_ATTRIBUTES, VERTEX_DESC_TAG_LAYOUTS];
const VERTEX_ATTR_TAGS_CONSUMED: [u8; 4] = [
    VERTEX_ATTR_TAG_LOCATION,
    VERTEX_ATTR_TAG_FORMAT,
    VERTEX_ATTR_TAG_OFFSET,
    VERTEX_ATTR_TAG_BUFFER_INDEX,
];
const VERTEX_LAYOUT_TAGS_CONSUMED: [u8; 4] = [
    VERTEX_LAYOUT_TAG_BUFFER_INDEX,
    VERTEX_LAYOUT_TAG_STEP_FUNCTION,
    VERTEX_LAYOUT_TAG_STEP_RATE,
    VERTEX_LAYOUT_TAG_STRIDE,
];

/// `MTLVertexDescriptor.attributes` is a 31-slot array, so a descriptor naming
/// more is malformed rather than something we chose not to read. The same
/// number is [`crate::backend::metal::constants::REIMS_VGPU_METAL_MAX_ATTRS`],
/// which is the Metal limit itself; this is the decode side of it, stated here
/// because the Vulkan arm decodes the same descriptor and must lose the same
/// attributes or none.
pub const MAX_VERTEX_ATTRS: usize = 31;
/// `MTLVertexDescriptor.layouts` is the matching 31-slot array, so it has no
/// subscript at or above this and a layout naming one cannot be built. See
/// [`VertexDescriptorTruncated`].
pub const MAX_VERTEX_LAYOUTS: usize = 31;

/// A vertex descriptor that named more attributes or layout buffer indices than
/// [`MAX_VERTEX_ATTRS`] / [`MAX_VERTEX_LAYOUTS`] admit — which is to say, more
/// than `MTLVertexDescriptor` has slots for.
///
/// This is the line beside the refusal, not beside a truncation. Truncating is
/// what [`parse_vertex_block`] used to do, and neither loss was recoverable or
/// even visible downstream: a dropped attribute is indistinguishable from a
/// guest that declared fewer, so the draw runs with a stage input the shader
/// expects and never receives; a dropped layout is worse, because the attributes
/// that named its buffer fell through to `stride = 0` — a well-formed pipeline
/// that fetches element zero for every vertex, geometry collapsed to a point,
/// with nothing refusing. The block now returns `Err` in both cases and the
/// pipeline load reports `desc_decode` on top of this line.
///
/// Named for the same reason [`ColorAttachTableTruncated`] is, and it is the
/// sibling this decoder was missing: the colour-attachment table 500 lines below
/// has reported its truncation since it was written, while the two loops here
/// bounded themselves with a bare `break` and an `if` and said nothing. That
/// constant's doc carries the boot both were measured on; both read zero.
struct VertexDescriptorTruncated {
    /// Which array overflowed.
    what: &'static str,
    /// What the descriptor asked for — an attribute count, or the buffer index a
    /// layout entry named.
    declared: usize,
    /// The bound that refused it.
    max: usize,
}

impl crate::observe::Decline for VertexDescriptorTruncated {
    fn slug(&self) -> &'static str {
        "vertex_descriptor_truncated"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("what", self.what.to_string()),
            ("declared", self.declared.to_string()),
            ("max", self.max.to_string()),
        ]
    }
}

/// Report a vertex descriptor that lost attributes or a layout stride.
///
/// Deduped per distinct `(what, declared)` pair — a malformed descriptor is
/// re-decoded on every pipeline bind, so an undeduped line would arrive once per
/// draw.
fn note_vertex_truncated(what: &'static str, declared: usize, max: usize) {
    let disc = ((what.len() as u64) << 56) | declared as u64;
    if !crate::observe::first_sight("vertex_descriptor_truncated", disc) {
        return;
    }
    crate::observe::Emit::decline(
        "res_vertex_block",
        &VertexDescriptorTruncated {
            what,
            declared,
            max,
        },
    )
    .fail();
}
pub const VERTEX_LABEL_MIN_ASCII: u8 = 0x20;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FunctionDescriptor {
    pub blob_gva: u64,
    pub blob_size: u32,
    pub function_id: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SamplerDescriptor {
    pub min_filter: u32,
    pub mag_filter: u32,
    pub mip_filter: u32,
    pub s_address: u32,
    pub t_address: u32,
    pub r_address: u32,
    /// `MTLSamplerDescriptor.maxAnisotropy`, already floored at 1 by
    /// [`decode_sampler_descriptor`], which is the only producer of this type.
    /// Every consumer may bind it straight through; four of them used to floor
    /// it again and a fifth did not, which is the shape that leaves one arm
    /// disagreeing about a value none of them owned.
    pub max_anisotropy: u32,
    pub lod_min_clamp: f32,
    pub lod_max_clamp: f32,
    pub compare_function: u32,
    pub border_color: u32,
    pub normalized_coordinates: bool,
    pub support_argument_buffers: bool,
    pub lod_average: bool,
}

/// Type-7 depth-stencil face (12 bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DepthStencilFace {
    pub compare_function: u32,
    pub stencil_failure_operation: u32,
    pub depth_failure_operation: u32,
    pub depth_stencil_pass_operation: u32,
    pub read_mask: u32,
    pub write_mask: u32,
}

/// Type-7 depth-stencil state (40 bytes).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DepthStencilDescriptor {
    pub depth_stencil_id: u32,
    pub depth_compare_function: u32,
    pub depth_write_enabled: bool,
    pub front_stencil_enabled: bool,
    pub back_stencil_enabled: bool,
    pub front_face: DepthStencilFace,
    pub back_face: DepthStencilFace,
}

/// Where the depth-stencil creation record puts each field, for the synthetic
/// buffers the tests below assemble.
///
/// Derived from the wire view `decode_depth_stencil_descriptor` reads, not
/// restated beside it: these were five literals ported from a C header, and a
/// literal cannot notice when the struct it transcribes is re-derived. Now a
/// rename or a reordering in `w_ds` fails this build instead of silently
/// leaving the tests assembling a record shaped like last year's.
#[cfg(test)]
const DEPTH_STENCIL_DESC_LEN: usize = w_ds::NEW_DEPTH_STENCIL_TOTAL_LEN as usize;
#[cfg(test)]
const DEPTH_STENCIL_DESC_STATE_BITS: usize =
    OP_HDR + offset_of!(w_ds::DepthStencilBody, depth_state);
#[cfg(test)]
const DEPTH_STENCIL_DESC_ID: usize = OP_HDR + offset_of!(w_ds::DepthStencilBody, object_ref);
#[cfg(test)]
const DEPTH_STENCIL_DESC_FRONT_FACE: usize = OP_HDR + offset_of!(w_ds::DepthStencilBody, front);
#[cfg(test)]
pub(crate) const DEPTH_STENCIL_DEPTH_WRITE: u32 = 1 << 3;
pub const DEPTH_STENCIL_FRONT_STENCIL_ENABLED: u32 = 1 << 4;
pub const DEPTH_STENCIL_BACK_STENCIL_ENABLED: u32 = 1 << 5;

/// Per-command-slot layout offsets inside the ICB backing buffer.
///
/// Type encoding from `AppleParavirtIndirectCommandBuffer._commandLayout`,
/// embedded at create-body `+0x1c` (52 bytes).
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

/// Decoded type-7 ICB create descriptor (88-byte MetalSerializer body).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct IndirectCommandBufferDescriptor {
    pub command_types: u32,
    pub max_vertex_buffer_bind_count: u16,
    pub max_fragment_buffer_bind_count: u16,
    pub max_kernel_buffer_bind_count: u16,
    /// `maxObjectBufferBindCount` create body `+0x0f`.
    pub max_object_buffer_bind_count: u16,
    /// `maxMeshBufferBindCount` create body `+0x10`.
    pub max_mesh_buffer_bind_count: u16,
    /// `maxKernelThreadgroupMemoryBindCount` create body `+0x11`.
    pub max_kernel_threadgroup_memory_bind_count: u16,
    /// `maxObjectThreadgroupMemoryBindCount` create body `+0x12`.
    pub max_object_threadgroup_memory_bind_count: u16,
    /// The whole flag word at `+0x16`, ten of whose bits are attributed.
    ///
    /// Stored as the word rather than as a bool per bit, and read through the
    /// accessors below, because this device previously carried two of them as
    /// fields and dropped the other eight on the floor. One declaration is what
    /// keeps a decoded bit and its accessor from drifting apart, and it means a
    /// bit that gains a consumer needs no decoder change.
    pub flags: u16,
    pub max_command_count: u32,
    /// `MTLResourceOptions`, **sixteen bits wide on the wire** despite the
    /// selector declaring the argument `Q`. See [`ICB_DESC_OPTIONS`].
    pub options: u16,
    /// Layout for decoding filled command slots in the ICB backing buffer.
    pub layout: IcbCommandLayout,
}

/// One decoded flag the guest asked for and this device does not carry into the
/// host indirect command buffer.
///
/// Eight of the ten attributed bits reach no host setter. Six of them default
/// **on** in the descriptor Metal builds and in the one the guest built, so a
/// guest that leaves them alone loses nothing; the loss is the guest turning one
/// *off*, or turning one of the two `support*` flags *on*. That asymmetry is why
/// this is a list of named losses rather than a mask comparison — see
/// [`IndirectCommandBufferDescriptor::unapplied_flags`].
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
    /// One slug per flag rather than one for the set. A count that said only
    /// "some flag was dropped" would not say whether the guest wanted ray
    /// tracing or wanted its cull mode not inherited, and those cost entirely
    /// different things to implement.
    pub fn slug(self) -> &'static str {
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
    /// `inheritPipelineState`, bit 0 of [`Self::flags`].
    pub fn inherit_pipeline_state(&self) -> bool {
        self.flags & ICB_FLAG_INHERIT_PIPELINE_STATE != 0
    }

    /// `inheritBuffers`, bit 1.
    pub fn inherit_buffers(&self) -> bool {
        self.flags & ICB_FLAG_INHERIT_BUFFERS != 0
    }

    // `supportRayTracing` (bit 2) and `supportDynamicAttributeStride` (bit 3)
    // have no accessor, and that is the rule rather than an omission: an
    // accessor here means the device acts on the bit, and those two it does not.
    // Both are read by `unapplied_flags` instead, which names each one the guest
    // asked for so the request is measured rather than dropped in silence. An
    // accessor beside that would be a second reader of the same bit whose only
    // effect is to make the two kinds of flag look alike.

    /// The bits no `MTLIndirectCommandBufferDescriptor` property moves.
    ///
    /// Exposed so a reading other than the `0x7840` every record Apple produced
    /// carries is visible: one of them would then be a real field this contract
    /// does not know about.
    pub fn unidentified_flags(&self) -> u16 {
        self.flags & ICB_FLAG_UNIDENTIFIED
    }

    /// Every flag the guest asked for that this device does not apply.
    ///
    /// Empty on a descriptor whose flags are at their defaults, which is what
    /// makes each of these a healthy zero: a non-zero count is the measured
    /// argument for building the host setter that flag needs.
    pub fn unapplied_flags(&self) -> Vec<IcbUnappliedFlag> {
        let mut out = Vec::new();
        // Default off: asking is setting the bit.
        for (bit, flag) in [
            (
                ICB_FLAG_SUPPORT_RAY_TRACING,
                IcbUnappliedFlag::SupportRayTracing,
            ),
            (
                ICB_FLAG_SUPPORT_DYNAMIC_ATTRIBUTE_STRIDE,
                IcbUnappliedFlag::SupportDynamicAttributeStride,
            ),
        ] {
            if self.flags & bit != 0 {
                out.push(flag);
            }
        }
        // Default on: asking is clearing the bit. This device sets none of
        // these on the host descriptor, so it inherits whatever Metal defaults
        // to — which is the same "on" the guest started from. A guest that
        // cleared one gets a host ICB that still inherits that state.
        for (bit, flag) in [
            (
                ICB_FLAG_INHERIT_DEPTH_STENCIL_STATE,
                IcbUnappliedFlag::InheritDepthStencilState,
            ),
            (
                ICB_FLAG_INHERIT_DEPTH_BIAS,
                IcbUnappliedFlag::InheritDepthBias,
            ),
            (
                ICB_FLAG_INHERIT_DEPTH_CLIP_MODE,
                IcbUnappliedFlag::InheritDepthClipMode,
            ),
            (
                ICB_FLAG_INHERIT_CULL_MODE,
                IcbUnappliedFlag::InheritCullMode,
            ),
            (
                ICB_FLAG_INHERIT_FRONT_FACING_WINDING,
                IcbUnappliedFlag::InheritFrontFacingWinding,
            ),
            (
                ICB_FLAG_INHERIT_TRIANGLE_FILL_MODE,
                IcbUnappliedFlag::InheritTriangleFillMode,
            ),
        ] {
            if self.flags & bit == 0 {
                out.push(flag);
            }
        }
        out
    }
}

/// One decoded object-list descriptor.
///
/// There is deliberately no `Unknown`: an object type this host has no contract
/// for is [`DecodeStatus::ErrUnknownType`], and a type-7 subtype it does not
/// implement is [`DecodeStatus::ErrUnsupported`]. Both name the check that
/// refused. An `Unknown` variant would let the same condition arrive as a
/// successful decode carrying nothing, which every consumer would then have to
/// re-refuse without knowing why.
#[derive(Clone, Debug, PartialEq)]
pub enum Descriptor {
    Buffer(BufferDescriptor),
    Texture(TextureDescriptor),
    Sampler(SamplerDescriptor),
    Function(FunctionDescriptor),
    RenderPipeline(RenderPipelineDescriptor),
    ComputePipeline(ComputePipelineDescriptor),
    DepthStencil(DepthStencilDescriptor),
    TextureView(TextureViewDescriptor),
    IOSurfaceTexture {
        mapping_id: u32,
        object_ref: u32,
        pixel_format: u16,
        width: u32,
        height: u32,
    },
    IndirectCommandBuffer(IndirectCommandBufferDescriptor),
}

/// Live Reims VGPU object-list entry size (kb + reims-vgpu-resource-format).
pub const OBJECT_LIST_ENTRY_LEN: usize = 12;
pub const OBJECT_LIST_ENTRY_HEADER: usize = 0;
pub const OBJECT_LIST_ENTRY_DESC_GVA: usize = 4;
pub const OBJECT_TYPE_MASK: u32 = 0xff;
pub const OBJECT_DESC_LEN_SHIFT: u32 = 8;

/// Wire object-list entry: `[type:u8 | desc_len:u24]<< packed u32` + `desc_gva:u64`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ListObjectEntry {
    pub object_type: u8,
    pub descriptor_length: u32,
    pub descriptor_gva: u64,
}

/// Decode one 12-byte object-list entry (live arm Reims VGPU contract).
pub fn decode_list_object_entry(bytes: &[u8]) -> Result<ListObjectEntry, DecodeStatus> {
    if bytes.len() < OBJECT_LIST_ENTRY_LEN {
        return Err(DecodeStatus::ErrShort("res_list_entry_short"));
    }
    let first = ld32(&bytes[OBJECT_LIST_ENTRY_HEADER..]);
    Ok(ListObjectEntry {
        object_type: (first & OBJECT_TYPE_MASK) as u8,
        descriptor_length: first >> OBJECT_DESC_LEN_SHIFT,
        descriptor_gva: ld64(&bytes[OBJECT_LIST_ENTRY_DESC_GVA..]),
    })
}

/// Byte offset of object-list slot `ref_` (0-based index; ref_ < entry_count).
pub fn list_object_entry_offset(ref_: u32, entry_count: u32) -> Option<u64> {
    if ref_ >= entry_count {
        return None;
    }
    (ref_ as u64).checked_mul(OBJECT_LIST_ENTRY_LEN as u64)
}

pub fn decode_buffer_descriptor(bytes: &[u8]) -> Result<BufferDescriptor, DecodeStatus> {
    if bytes.len() < LINEAR_DESC_MIN_LEN {
        return Err(DecodeStatus::ErrShort("res_buffer_desc_short"));
    }
    let handle64 = ld64(&bytes[LINEAR_DESC_HANDLE..]);
    Ok(BufferDescriptor {
        allocation_size: ld64(&bytes[LINEAR_DESC_SIZE..]),
        handle64,
        handle: handle64 as u32,
    })
}

pub fn decode_texture_descriptor(bytes: &[u8]) -> Result<TextureDescriptor, DecodeStatus> {
    if bytes.len() < TEXTURE_DESC_GEOMETRY_LEN {
        return Err(DecodeStatus::ErrShort("res_texture_desc_short"));
    }
    let mut out = TextureDescriptor {
        allocation_size: ld64(&bytes[LINEAR_DESC_SIZE..]),
        handle: ld32(&bytes[LINEAR_DESC_HANDLE..]),
        ..Default::default()
    };
    if bytes.len() >= TEXTURE_DESC_MIPMAP_LEVEL_COUNT + 2 {
        out.mipmap_level_count = ld16(&bytes[TEXTURE_DESC_MIPMAP_LEVEL_COUNT..]) as u32;
    }
    if bytes.len() >= TEXTURE_DESC_DATA_OFFSET + 4 {
        out.data_offset = ld32(&bytes[TEXTURE_DESC_DATA_OFFSET..]);
    }
    if bytes.len() > TEXTURE_DESC_BYTES_PER_ELEMENT {
        out.bytes_per_element = bytes[TEXTURE_DESC_BYTES_PER_ELEMENT];
    }
    if bytes.len() >= TEXTURE_DESC_USED_SIZE + 4 {
        out.used_size = ld32(&bytes[TEXTURE_DESC_USED_SIZE..]);
    }
    if bytes.len() >= TEXTURE_DESC_ROW_STRIDE + 4 {
        out.row_stride = ld32(&bytes[TEXTURE_DESC_ROW_STRIDE..]);
    }
    if bytes.len() >= TEXTURE_DESC_WIDTH + 4 {
        out.width = ld32(&bytes[TEXTURE_DESC_WIDTH..]);
    }
    if bytes.len() >= TEXTURE_DESC_HEIGHT + 4 {
        out.height = ld32(&bytes[TEXTURE_DESC_HEIGHT..]);
    }
    if bytes.len() >= TEXTURE_DESC_DEPTH + 4 {
        out.depth = ld32(&bytes[TEXTURE_DESC_DEPTH..]);
        if out.depth == 0 {
            out.depth = 1;
        }
    } else {
        out.depth = 1;
    }

    // Level layouts: L0 from geometry prefix; L1.. from records at +72.
    let declared_levels = if out.mipmap_level_count > 0 {
        out.mipmap_level_count
    } else {
        1
    };
    if out.extent().is_some() {
        // `size` is the level's *allocated* span, not the bytes a reader
        // touches. The two differ by the padding after the final row, and the
        // difference is load-bearing in both directions: `blit_exec` compares
        // this field for equality against `row_stride * height * depth` to tell
        // a single-slice allocation from an array one, so the padded form is the
        // one that can match — while the same function charges a *read* through
        // `TextureLevelLayout::slice_read_span`, whose doc records that using
        // the padded form as a bound refuses allocations the guest sized
        // correctly. Do not "fix" this to `read_span`; levels 1.. take `size`
        // from the wire at `TEXTURE_LEVEL_SIZE` and mean the same padded span.
        let l0_size = if out.used_size != 0 {
            out.used_size as u64
        } else if out.declared_row_stride().is_some() && out.height > 0 {
            (out.row_stride as u64).saturating_mul(out.height as u64)
        } else {
            0
        };
        out.levels.push(TextureLevelLayout {
            offset: out.data_offset as u64,
            size: l0_size,
            row_stride: out.row_stride as u64,
            width: out.width,
            height: out.height,
            depth: if out.depth == 0 { 1 } else { out.depth },
        });
        if declared_levels > 1 {
            let mut rec_off = TEXTURE_DESC_LEVEL_RECORDS;
            let max_extra = (declared_levels as usize - 1).min(TEXTURE_MAX_MIP_LEVELS - 1);
            // Both truncations below leave `mipmap_level_count` at what the
            // guest declared while `levels` holds fewer, so `level(n)` answers
            // `None` for a level the descriptor named. That is a level of a
            // texture this device will not sample or blit, and it has to be
            // legible as a drop rather than as an absence.
            if declared_levels as usize - 1 > max_extra
                && crate::observe::first_sight(
                    "texture_desc_levels_over_cap",
                    u64::from(declared_levels),
                )
            {
                crate::observe::fail(format!(
                    "texture_desc_levels_over_cap declared={declared_levels} \
                     cap={TEXTURE_MAX_MIP_LEVELS}"
                ));
            }
            for _ in 0..max_extra {
                if rec_off + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN > bytes.len() {
                    if crate::observe::first_sight(
                        "texture_desc_level_record_short",
                        u64::from(declared_levels),
                    ) {
                        crate::observe::fail(format!(
                            "texture_desc_level_record_short declared={declared_levels} \
                             decoded={} rec_off={rec_off} len={} \
                             (body ends before a level record the descriptor named)",
                            out.levels.len(),
                            bytes.len()
                        ));
                    }
                    break;
                }
                let rec = &bytes[rec_off..rec_off + TEXTURE_DESC_MIP_LEVEL_RECORD_LEN];
                let mut depth = ld32(&rec[TEXTURE_LEVEL_DEPTH..]);
                if depth == 0 {
                    depth = 1;
                }
                out.levels.push(TextureLevelLayout {
                    offset: ld64(&rec[TEXTURE_LEVEL_OFFSET..]),
                    size: ld64(&rec[TEXTURE_LEVEL_SIZE..]),
                    row_stride: ld64(&rec[TEXTURE_LEVEL_ROW_STRIDE..]),
                    width: ld32(&rec[TEXTURE_LEVEL_WIDTH..]),
                    height: ld32(&rec[TEXTURE_LEVEL_HEIGHT..]),
                    depth,
                });
                rec_off += TEXTURE_DESC_MIP_LEVEL_RECORD_LEN;
            }
        }
    }

    // Format trailer: shift by (levels-1)*36 for multi-mip bodies.
    let levels = declared_levels;
    let format_shift = if levels > 1 {
        (levels as usize - 1).saturating_mul(TEXTURE_DESC_MIP_LEVEL_RECORD_LEN)
    } else {
        0
    };
    let pf_off = TEXTURE_DESC_PIXEL_FORMAT + format_shift;
    if bytes.len() >= pf_off + 2 {
        out.pixel_format = ld16(&bytes[pf_off..]);
    } else if crate::observe::first_sight("texture_desc_format_unreachable", levels as u64) {
        // No fallback to the unshifted offset. The fallback's own length test
        // was `TEXTURE_DESC_PIXEL_FORMAT + 2`, so for a single-mip body it
        // guarded the same two bytes the branch above already read. It
        // was reachable only when `format_shift > 0` — and there offset 86 is
        // not the format at all: the level records start at 72 and run 36 bytes
        // each, so 86..88 is inside level record 1. The fallback therefore
        // produced a format only in the case where it was guaranteed to be
        // reading something else, and then set `has_pixel_format`, which is
        // what seven downstream gates fail closed on. Better to have no format.
        crate::observe::fail(format!(
            "texture_desc_format_unreachable levels={levels} pf_off={pf_off} len={} \
             (multi-mip body too short for its shifted format trailer)",
            bytes.len()
        ));
    }
    Ok(out)
}

pub fn decode_function_descriptor(bytes: &[u8]) -> Result<FunctionDescriptor, DecodeStatus> {
    if bytes.len() < FUNCTION_DESC_MIN_LEN {
        return Err(DecodeStatus::ErrShort("res_function_desc_short"));
    }
    let blob_gva = ld64(&bytes[FUNCTION_DESC_BLOB_GVA..]);
    let blob_size = ld32(&bytes[FUNCTION_DESC_BLOB_SIZE..]);
    let function_id = if bytes.len() >= FUNCTION_DESC_FUNCTION_ID + 4 {
        ld32(&bytes[FUNCTION_DESC_FUNCTION_ID..])
    } else {
        0
    };
    Ok(FunctionDescriptor {
        blob_gva,
        blob_size,
        function_id,
    })
}

/// Compact type-7 first sub-record: `[fieldCount:u8]` × `[tag:u8][len:u8][value…]`.
pub fn decode_compact_tlv_record(
    bytes: &[u8],
    offset: usize,
) -> Result<(Vec<CompactTlv>, usize), DecodeStatus> {
    if offset >= bytes.len() {
        return Err(DecodeStatus::ErrShort("res_tlv_offset_past_end"));
    }
    let field_count = bytes[offset] as usize;
    let mut p = offset + 1;
    let mut out = Vec::with_capacity(field_count);
    for _ in 0..field_count {
        if p + 2 > bytes.len() {
            return Err(DecodeStatus::ErrShort("res_tlv_header_short"));
        }
        let tag = bytes[p];
        let field_len = bytes[p + 1] as usize;
        if p + 2 + field_len > bytes.len() {
            return Err(DecodeStatus::ErrShort("res_tlv_value_short"));
        }
        let value_offset = p + 2;
        let (has_u32, value_u32) = if field_len >= 4 {
            (true, ld32(&bytes[value_offset..]))
        } else {
            (false, 0)
        };
        out.push(CompactTlv {
            tag,
            length: field_len as u8,
            value_offset,
            value_u32,
            has_u32,
        });
        p += 2 + field_len;
    }
    Ok((out, p - offset))
}

fn compact_tlv_u32(fields: &[CompactTlv], tag: u8) -> Option<u32> {
    fields
        .iter()
        .find(|f| f.tag == tag && f.has_u32)
        .map(|f| f.value_u32)
}

/// [`entry_tag_u32_present`] with a value for "the entry does not carry `tag`".
///
/// The two used to be written out separately, with identical control flow and
/// five identical bounds checks, differing only in what they returned when the
/// walk fell through. A walk written twice is a walk that can be fixed once.
fn entry_tag_u32(bytes: &[u8], entry_off: usize, tag: u8, default: u32) -> u32 {
    entry_tag_u32_present(bytes, entry_off, tag).unwrap_or(default)
}

fn entry_tag_u32_present(bytes: &[u8], entry_off: usize, tag: u8) -> Option<u32> {
    if entry_off >= bytes.len() {
        return None;
    }
    let field_count = bytes[entry_off] as usize;
    let mut p = entry_off + 1;
    for _ in 0..field_count {
        if p + 2 > bytes.len() {
            return None;
        }
        let t = bytes[p];
        let field_len = bytes[p + 1] as usize;
        if p + 2 + field_len > bytes.len() {
            return None;
        }
        if t == tag && field_len >= 4 {
            return Some(ld32(&bytes[p + 2..]));
        }
        p += 2 + field_len;
    }
    None
}

fn skip_optional_label_and_pad(bytes: &[u8], mut off: usize) -> usize {
    if off < bytes.len() && bytes[off] >= VERTEX_LABEL_MIN_ASCII {
        while off < bytes.len() && bytes[off] != 0 {
            off += 1;
        }
    }
    while off < bytes.len() && bytes[off] == 0 {
        off += 1;
    }
    off
}

/// Parse vertex-input block between first TLVs and color-attachment section.
pub fn parse_vertex_block(
    bytes: &[u8],
    block_start: usize,
    block_end: usize,
) -> Result<Vec<VertexAttribute>, DecodeStatus> {
    if block_start >= block_end {
        return Ok(Vec::new());
    }
    // The block's declared end becomes the slice, so `block_end` stops being a
    // second number the body has to remember to compare against.
    let Some(bytes) = bytes.get(..block_end) else {
        return Ok(Vec::new());
    };
    let bo = skip_optional_label_and_pad(bytes, block_start);
    if bo >= bytes.len() {
        return Ok(Vec::new());
    }
    note_entry_tlv_fields("vertex_desc", bytes, bo, &VERTEX_DESC_TAGS_CONSUMED);
    let attr_off = entry_tag_u32(bytes, bo, VERTEX_DESC_TAG_ATTRIBUTES, u32::MAX);
    let layout_off = entry_tag_u32(bytes, bo, VERTEX_DESC_TAG_LAYOUTS, u32::MAX);
    if attr_off == u32::MAX || layout_off == u32::MAX {
        return Ok(Vec::new());
    }

    let mut strides = [0u32; MAX_VERTEX_LAYOUTS];
    let mut have_stride = [false; MAX_VERTEX_LAYOUTS];
    // Buffer index, and the step function / step rate that buffer's layout
    // entry declared — `None` where the entry carried no such tag.
    let mut layout_steps: Vec<(u32, Option<u32>, Option<u32>)> = Vec::new();

    let layout_section = bo.saturating_add(layout_off as usize);
    if layout_section + 4 > bytes.len() {
        return Err(DecodeStatus::ErrShort("res_vertex_layout_count_oob"));
    }
    let layout_count = ld32(&bytes[layout_section..]) as usize;
    for i in 0..layout_count {
        let offloc = layout_section + 4 + i * 4;
        if offloc + 4 > bytes.len() {
            return Err(DecodeStatus::ErrShort("res_vertex_layout_offset_oob"));
        }
        let entry = layout_section + ld32(&bytes[offloc..]) as usize;
        if entry >= bytes.len() {
            return Err(DecodeStatus::ErrShort("res_vertex_layout_entry_oob"));
        }
        note_entry_tlv_fields("vertex_layout", bytes, entry, &VERTEX_LAYOUT_TAGS_CONSUMED);
        let buffer_index = entry_tag_u32(bytes, entry, VERTEX_LAYOUT_TAG_BUFFER_INDEX, 0);
        let stride = entry_tag_u32(bytes, entry, VERTEX_LAYOUT_TAG_STRIDE, 0);
        let declared_step_function =
            entry_tag_u32_present(bytes, entry, VERTEX_LAYOUT_TAG_STEP_FUNCTION);
        let declared_step_rate = entry_tag_u32_present(bytes, entry, VERTEX_LAYOUT_TAG_STEP_RATE);
        if (buffer_index as usize) >= MAX_VERTEX_LAYOUTS {
            // `MTLVertexDescriptor.layouts` has no such subscript, so this is a
            // descriptor no Apple driver produces and one this decoder cannot
            // implement. Refusing the pipeline is the only outcome that does not
            // guess: keeping the block dropped the stride, every attribute naming
            // this buffer fell through to 0, and a zero stride is a *valid*
            // pipeline that fetches element zero for every vertex — geometry
            // collapsed to a point, indistinguishable downstream from a guest
            // that asked for it.
            note_vertex_truncated(
                "layout_buffer_index",
                buffer_index as usize,
                MAX_VERTEX_LAYOUTS,
            );
            return Err(DecodeStatus::ErrUnsupported("res_vertex_layout_buffer_oob"));
        } else if stride != 0 {
            strides[buffer_index as usize] = stride;
            have_stride[buffer_index as usize] = true;
        }
        layout_steps.push((buffer_index, declared_step_function, declared_step_rate));
    }

    let attr_section = bo.saturating_add(attr_off as usize);
    if attr_section + 4 > bytes.len() {
        return Err(DecodeStatus::ErrShort("res_vertex_attr_count_oob"));
    }
    let attr_count = ld32(&bytes[attr_section..]) as usize;
    let mut attrs = Vec::new();
    if attr_count > MAX_VERTEX_ATTRS {
        // Same refusal as the layout bound above, for the same reason: keeping
        // the first 31 leaves the shader with stage inputs it declares and never
        // receives, and a draw missing an attribute is indistinguishable from a
        // guest that declared fewer.
        note_vertex_truncated("attribute_count", attr_count, MAX_VERTEX_ATTRS);
        return Err(DecodeStatus::ErrUnsupported("res_vertex_attr_count_over"));
    }
    for i in 0..attr_count {
        let offloc = attr_section + 4 + i * 4;
        if offloc + 4 > bytes.len() {
            return Err(DecodeStatus::ErrShort("res_vertex_attr_offset_oob"));
        }
        let entry = attr_section + ld32(&bytes[offloc..]) as usize;
        if entry >= bytes.len() {
            return Err(DecodeStatus::ErrShort("res_vertex_attr_entry_oob"));
        }
        note_entry_tlv_fields("vertex_attr", bytes, entry, &VERTEX_ATTR_TAGS_CONSUMED);
        let location = entry_tag_u32(bytes, entry, VERTEX_ATTR_TAG_LOCATION, i as u32);
        let format = entry_tag_u32(bytes, entry, VERTEX_ATTR_TAG_FORMAT, 0);
        let offset = entry_tag_u32(bytes, entry, VERTEX_ATTR_TAG_OFFSET, 0);
        let buffer_index = entry_tag_u32(bytes, entry, VERTEX_ATTR_TAG_BUFFER_INDEX, 0);
        let stride =
            if (buffer_index as usize) < MAX_VERTEX_LAYOUTS && have_stride[buffer_index as usize] {
                strides[buffer_index as usize]
            } else {
                0
            };
        let (sf, sr) = layout_steps
            .iter()
            .find(|(bi, ..)| *bi == buffer_index)
            .map(|(_, sf, sr)| (*sf, *sr))
            .unwrap_or((None, None));
        attrs.push(VertexAttribute {
            location,
            format,
            offset,
            buffer_index,
            stride,
            declared_step_function: sf,
            declared_step_rate: sr,
        });
    }
    Ok(attrs)
}

fn face_from_wire(face: &w_ds::StencilFace) -> DepthStencilFace {
    DepthStencilFace {
        compare_function: face.compare_function() as u32,
        stencil_failure_operation: face.stencil_failure_operation() as u32,
        depth_failure_operation: face.depth_failure_operation() as u32,
        depth_stencil_pass_operation: face.depth_stencil_pass_operation() as u32,
        read_mask: face.read_mask.get(),
        write_mask: face.write_mask.get(),
    }
}

pub fn decode_depth_stencil_descriptor(
    bytes: &[u8],
) -> Result<DepthStencilDescriptor, DecodeStatus> {
    let op = reims_vgpu_wire::op(bytes, 0)
        .map_err(|_| DecodeStatus::ErrShort("res_depth_stencil_short"))?;
    if op.opcode() != TYPE7_OBJECT_DEPTH_STENCIL || op.length() as usize != bytes.len() {
        return Err(DecodeStatus::ErrUnsupported("res_depth_stencil_tag"));
    }
    let body = w_ds::new_depth_stencil(&op)
        .map_err(|_| DecodeStatus::ErrShort("res_depth_stencil_short"))?;
    // Product still names bits [5:4] as face-enabled; wire records them as
    // unidentified (Metal substitutes default faces before serialize). Keep the
    // product field names; source the bit from the same byte the view exposes.
    let state = body.depth_state;
    Ok(DepthStencilDescriptor {
        depth_stencil_id: body.object_ref.get(),
        depth_compare_function: body.depth_compare_function() as u32,
        depth_write_enabled: body.depth_write_enabled(),
        front_stencil_enabled: (state & DEPTH_STENCIL_FRONT_STENCIL_ENABLED as u8) != 0,
        back_stencil_enabled: (state & DEPTH_STENCIL_BACK_STENCIL_ENABLED as u8) != 0,
        front_face: face_from_wire(&body.front),
        back_face: face_from_wire(&body.back),
    })
}

/// `MTLSamplerDescriptor.maxAnisotropy` from the state word's five-bit field.
///
/// Metal documents the range as 1 through 16, which is exactly what five bits
/// hold, and the wire crate's oracle fixture for a sampler with nothing set
/// reports 1 — Apple's serializer writes the API default rather than leaving
/// the field clear. So zero is not a value this field carries, and a zero here
/// is a malformed record rather than a guest asking for something.
///
/// Floored rather than refused, because the record is otherwise complete and a
/// refusal would drop a whole sampler over one out-of-range field; counted so
/// the choice is not silent. This is the single site — the four `.max(1)`s
/// downstream of it are gone, and with them the one consumer that had none.
///
/// `sampler_max_anisotropy_zero` **never fires** on a driven x86/Vulkan boot
/// (25 s Safari window drag, 2 758 posted events, ~1 500 draws), which is the
/// healthy reading and matches what the oracle says: every sampler this
/// workload creates declares a value in range. A firing is the signal, and it
/// would mean the field can be zero after all — at which point the question is
/// whether the record deserves a refusal rather than a floor.
fn wire_max_anisotropy(value: u8) -> u32 {
    if value != 0 {
        return u32::from(value);
    }
    crate::runtime::drain::note_store_route("sampler_max_anisotropy_zero");
    1
}

pub fn decode_sampler_descriptor(bytes: &[u8]) -> Result<SamplerDescriptor, DecodeStatus> {
    let op =
        reims_vgpu_wire::op(bytes, 0).map_err(|_| DecodeStatus::ErrShort("res_sampler_short"))?;
    if op.opcode() != TYPE7_OBJECT_SAMPLER || op.length() as usize != bytes.len() {
        return Err(DecodeStatus::ErrUnsupported("res_sampler_tag"));
    }
    let body = w_smp::new_sampler(&op).map_err(|_| DecodeStatus::ErrShort("res_sampler_short"))?;
    Ok(SamplerDescriptor {
        min_filter: body.min_filter() as u32,
        mag_filter: body.mag_filter() as u32,
        mip_filter: body.mip_filter() as u32,
        s_address: body.s_address_mode() as u32,
        t_address: body.t_address_mode() as u32,
        r_address: body.r_address_mode() as u32,
        max_anisotropy: wire_max_anisotropy(body.max_anisotropy()),
        lod_min_clamp: body.lod_min_clamp.get(),
        lod_max_clamp: body.lod_max_clamp.get(),
        compare_function: body.compare_function() as u32,
        border_color: body.border_color() as u32,
        normalized_coordinates: body.normalized_coordinates(),
        support_argument_buffers: body.support_argument_buffers(),
        lod_average: body.lod_average(),
    })
}

/// Tags [`decode_render_pipeline_descriptor`] reads on the **classic** shape —
/// the branch a descriptor without [`PIPELINE_TAG_MESH_SECTION_OFFSET`] takes.
///
/// Three, out of the function refs and the colour-attachment section offset.
/// `0x03` is deliberately **not** here: the decoder reads it into `tag03` and
/// then uses that variable only in the mesh branch, so on a classic descriptor
/// the field is loaded and discarded. A first draft of this instrument listed
/// every tag either branch reads and reported `unconsumed=0` for a driven
/// boot's classic pipelines that carry `0x03` — an instrument built to find
/// unread fields hiding one behind a tag its *other* branch reads. Which tags
/// count as consumed is a property of the branch taken, so it is chosen there.
const CLASSIC_PIPELINE_TAGS_CONSUMED: [u8; 5] = [
    PIPELINE_TAG_VERTEX_FUNC,
    PIPELINE_TAG_FRAGMENT_FUNC,
    PIPELINE_TAG_VERTEX_DESCRIPTOR_OFFSET,
    PIPELINE_TAG_COLOR_ATTACH_OFFSET,
    PIPELINE_TAG_RASTER_SAMPLE_COUNT,
];

/// Tags [`decode_render_pipeline_descriptor`] reads on the **mesh** shape.
///
/// Four: the section offset that selected this branch and the three function
/// refs whose roles it re-maps. See [`CLASSIC_PIPELINE_TAGS_CONSUMED`] for why
/// the two sets are listed apart.
const MESH_PIPELINE_TAGS_CONSUMED: [u8; 5] = [
    PIPELINE_TAG_MESH_SECTION_OFFSET,
    PIPELINE_TAG_VERTEX_FUNC,
    PIPELINE_TAG_FRAGMENT_FUNC,
    PIPELINE_TAG_MESH_FRAGMENT_FUNC,
    PIPELINE_TAG_RASTER_SAMPLE_COUNT,
];

/// Tags [`decode_compute_pipeline_descriptor`] reads out of a compute
/// pipeline's own compact-TLV block.
///
/// One. Listed apart from its render sibling rather than merged into a union,
/// because a union would report a render tag as *consumed* on a compute
/// pipeline that has no reader for it — an instrument built to find unread
/// fields must not hide one behind a tag its other caller reads.
const COMPUTE_PIPELINE_TAGS_CONSUMED: [u8; 1] = [PIPELINE_TAG_KERNEL_FUNC];

/// `MTLRenderPipelineDescriptor.label`, a four-byte reference into the record's
/// string area.
///
/// Numerically the same tag as [`PIPELINE_TAG_KERNEL_FUNC`], and that is not a
/// collision: the tag is a property index within *this* descriptor's property
/// list, so tag 0 on a render descriptor and tag 0 on a compute one name
/// different properties. Declared separately for that reason — sharing the
/// constant would say the two are one property.
const RENDER_PIPELINE_TAG_LABEL: u8 = 0x00;
/// `MTLComputePipelineDescriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth`,
/// a `BOOL` widened to four bytes.
const COMPUTE_PIPELINE_TAG_THREADGROUP_MULTIPLE: u8 = 0x01;
/// `MTLComputePipelineDescriptor.label`. Same property as
/// [`RENDER_PIPELINE_TAG_LABEL`] at a different index, for the reason that
/// constant's doc gives.
const COMPUTE_PIPELINE_TAG_LABEL: u8 = 0x02;

/// Tags this decoder deliberately does not read, and which cost the guest
/// nothing.
///
/// The distinction from `*_TAGS_CONSUMED` is the whole of what makes refusing an
/// unknown tag safe. A consumed tag is one the decoder reads; a *benign* one is
/// one it has read the meaning of, decided not to apply, and written down why.
/// Everything in neither list is a property of `MTLRenderPipelineDescriptor`
/// nobody here has identified, and applying Metal's default in its place is the
/// silent modification [`PipelineFieldDropped`] exists to stop.
///
/// A tag may only join this list with the argument for it beside it:
///
/// * [`RENDER_PIPELINE_TAG_LABEL`] and [`COMPUTE_PIPELINE_TAG_LABEL`] are a
///   debug name. Nothing this device renders depends on one, and nothing about
///   the frame changes if it is dropped.
/// * [`COMPUTE_PIPELINE_TAG_THREADGROUP_MULTIPLE`] is a hint to Metal's own
///   shader compiler about how it may size a threadgroup. This device compiles
///   the shader itself and takes the threadgroup size from the dispatch record,
///   so there is nothing to apply — and the property's own default is the
///   conservative arm, so not applying it cannot be the unsafe direction.
/// * [`PIPELINE_TAG_DEPTH_ATTACH_FORMAT`] and
///   [`PIPELINE_TAG_STENCIL_ATTACH_FORMAT`] declare the attachment formats the
///   pipeline is compiled against. They do not *select* an attachment: the
///   render pass descriptor carries the depth and stencil textures a draw
///   renders into, and this device takes them from there. Both backends then
///   pin the host format — Vulkan to `TRANSIENT_DEPTH_FORMAT` or the device's
///   queried combined format, the Metal arm to `DEPTH32_FLOAT` in
///   `validate_depth_attachment` — so there is no arm on which the guest's
///   declared format could change the attachment it gets. Dropping them cannot
///   change what the guest observes; the attachment it observes is the one its
///   own pass named.
///
/// **`rasterizationEnabled` and `alphaToCoverageEnabled` are deliberately not
/// here.** They are two of the three the old doc named as silently defaulted,
/// and neither has appeared in this block on a driven boot. If one arrives it
/// must refuse rather than be waved through. The third,
/// [`PIPELINE_TAG_RASTER_SAMPLE_COUNT`], is now read — it is consumed on both
/// shapes and a count this device cannot rasterize at is a named degradation
/// rather than a silent default. See `note_pipeline_raster_sample_count`.
const RENDER_PIPELINE_TAGS_BENIGN: [u8; 3] = [
    RENDER_PIPELINE_TAG_LABEL,
    PIPELINE_TAG_DEPTH_ATTACH_FORMAT,
    PIPELINE_TAG_STENCIL_ATTACH_FORMAT,
];
/// The compute half of [`RENDER_PIPELINE_TAGS_BENIGN`]; same rule, and listed
/// apart for the same reason `COMPUTE_PIPELINE_TAGS_CONSUMED` is.
const COMPUTE_PIPELINE_TAGS_BENIGN: [u8; 2] = [
    COMPUTE_PIPELINE_TAG_THREADGROUP_MULTIPLE,
    COMPUTE_PIPELINE_TAG_LABEL,
];

/// A pipeline-descriptor field this decoder does not read **and has not
/// identified**. The pipeline is refused.
///
/// The colour-attachment walk beside this one has had both halves of this
/// instrument — a shape line and a per-tag decline — for as long as it has
/// refused an unknown tag. The pipeline's *own* block had neither, so a property
/// the guest set on the descriptor and this device never read was not merely
/// unimplemented, it was unmeasured: nothing said how many arrive, which ones,
/// or how often.
///
/// It is now a refusal on that same sibling's licence, and the licence is what
/// took the work: the sibling refuses because `type7_color_attach_shape`
/// measured its zero first. This block's `unconsumed` count is *not* zero and
/// never will be — two labels arrive on every boot — so the zero that had to be
/// measured was a different one. Splitting the unread tags into
/// [`RENDER_PIPELINE_TAGS_BENIGN`] (identified, argued, deliberately dropped)
/// and everything else made the second count exist, and *that* one reads zero
/// across the twelve shapes below. The shape line now carries both: `*` for
/// unread, `!` for unread and unidentified, and `unconsumed=` beside
/// `unknown=`.
///
/// # The reading, x86/Vulkan, one driven boot (Safari window drag)
///
/// Six distinct shapes, and the block is **small** — three to five fields, not
/// the twenty-odd `MTLRenderPipelineDescriptor` declares. So most of that
/// descriptor is not serialized into this block at all, and looking for
/// `rasterSampleCount` or `rasterizationEnabled` among these tags is looking in
/// the wrong place. That is the first thing this instrument settled, and it
/// settled it against the expectation that motivated writing it.
///
/// ```text
/// kind=render      tags=[08:4,01:4,02:4]            unconsumed=0  unknown=0
/// kind=render      tags=[03:4,08:4,01:4,02:4]       unconsumed=0  unknown=0
/// kind=render      tags=[00:4*,03:4,08:4,01:4,02:4] unconsumed=1  unknown=0
/// kind=render      tags=[00:4*,08:4,01:4,02:4]      unconsumed=1  unknown=0
/// kind=compute     tags=[02:4*,01:4*,00:4]          unconsumed=2  unknown=0
/// kind=compute     tags=[00:4]                      unconsumed=0  unknown=0
/// ```
///
/// Taken after `0x03` became a read tag and after the benign split, so the two
/// columns are what a reader should compare: **`unconsumed` is never zero and
/// `unknown` always is.** The same boot raised this decline zero times where it
/// raised it three times before the split — not because anything stopped being
/// dropped, but because what is dropped now has an argument behind it and the
/// fail channel is for the losses.
///
/// # What the four were
///
/// The block is `MTLRenderPipelineDescriptor`'s property list, one tag per
/// property, and a property left at its Metal default is **omitted** rather than
/// sent as a zero — which is why only three to five arrive. Same grammar as the
/// colour-attachment and vertex entries this file already decodes.
///
/// * **render `0x03` = `vertexDescriptor`**, a byte offset to the serialized
///   sub-object in the same units as `0x08` beside it. **Load-bearing, and now
///   read** — see [`PIPELINE_TAG_VERTEX_DESCRIPTOR_OFFSET`]. It is the one of
///   the four that cost anything: the block was being located by guesswork while
///   the descriptor stated its position.
/// * **render `0x00` = `label`** and **compute `0x02` = `label`**, each a
///   four-byte reference into the record's string area. A debug name; nothing
///   this device renders depends on it.
/// * **compute `0x01` =
///   `threadGroupSizeIsMultipleOfThreadExecutionWidth`**, a `BOOL` widened to
///   four bytes. A hint to Metal's own compiler about how it may size a
///   threadgroup; this device compiles the shader itself and takes the
///   threadgroup size from the dispatch record, so there is nothing here to
///   apply.
///
/// The last three are now named in [`RENDER_PIPELINE_TAGS_BENIGN`] and its
/// compute sibling, which is what turned "unread" into two different answers
/// and let the refusal below exist.
///
/// An earlier draft of this doc guessed that the compute pair might include
/// `maxTotalThreadsPerThreadgroup` and warned that dropping it could let this
/// device dispatch past a cap the guest set. **That alarm was wrong.** That
/// property is a different tag and a `u16`, and it appears in none of the shapes
/// above. The guess is recorded here rather than deleted because the shape line
/// is what disproved it, which is the argument for having one.
///
/// # Refused, and what it costs to be wrong in each direction
///
/// The alternative — the behaviour this replaced — is that an unidentified
/// property gets Metal's default: `rasterizationEnabled` becomes yes,
/// `alphaToCoverageEnabled` becomes no, `rasterSampleCount` becomes one. The
/// guest asked for something, the device built a pipeline that does something
/// else, and the frame is wrong with nothing downstream able to name it. That is
/// the class this file exists to stop, and refusing is what a GPU does with a
/// request it cannot represent.
///
/// The cost of refusing is a lost pipeline. The zero above does not bound it,
/// and reading it as a bound was the mistake: those shapes are one compositing
/// workload's, and the population they measure is "pipelines this rig's desktop
/// builds", not "pipelines a guest builds". A map view built three tags this
/// list had never seen — `0x04`, `0x09`, `0x0a` — and every pipeline carrying one
/// was refused, which put its whole canvas through the clear-only fallback while
/// the window chrome around it composited normally. A refusal is the right
/// answer to a property that cannot be represented; it is the wrong answer to
/// one nobody had got around to identifying, and only the second kind is
/// bounded by what a past boot happened to send.
///
/// So a firing is a guest doing something the *measured* workload does not, and
/// the answer is to identify the tag — into the consumed list if it is
/// load-bearing, into the benign list with an argument if it is not. Widening
/// the benign list to silence a refusal without that argument is the one move
/// this instrument is built to prevent. All three of the above are now
/// identified: see [`PIPELINE_TAG_RASTER_SAMPLE_COUNT`],
/// [`PIPELINE_TAG_DEPTH_ATTACH_FORMAT`] and
/// [`PIPELINE_TAG_STENCIL_ATTACH_FORMAT`].
///
/// Deduped per distinct `(tag, len)` rather than per value: what a reader needs
/// first is which properties arrive, and a per-value latch on a field like a
/// sample count would emit once per distinct count for no extra information.
/// The refusal itself is outside that latch — the line names a tag once, the
/// pipeline is refused every time.
struct PipelineFieldDropped {
    kind: &'static str,
    tag: u8,
    len: u8,
}

impl crate::observe::Decline for PipelineFieldDropped {
    fn slug(&self) -> &'static str {
        "pipeline_descriptor_field_dropped"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("kind", self.kind.to_string()),
            ("tag", format!("0x{:02x}", self.tag)),
            ("len", self.len.to_string()),
        ]
    }
}

/// A pipeline asking to be rasterized at a sample count this device does not
/// have attachments for. The pipeline is built at
/// [`DEFAULT_RASTER_SAMPLE_COUNT`] instead.
///
/// A **degradation, not a loss**: the draw runs and the guest keeps its
/// geometry, with edges that alias where the guest asked for them not to. That
/// is why this reports rather than refusing — a refusal here costs the whole
/// draw to buy nothing, because every render target both backends allocate is
/// single-sampled and so a pipeline built at any other count would not be
/// compatible with the pass it is used in. Widening it is an attachment-path
/// change, and this line is what would measure the demand for one.
///
/// On the fail channel because the reliance has to stay measurable, per this
/// crate's rule that a repair which succeeded is still worth naming. Deduped on
/// the requested count, which is the whole of what varies between two guests
/// asking for this.
struct PipelineRasterSampleCountDegraded {
    count: u32,
}

impl crate::observe::Decline for PipelineRasterSampleCountDegraded {
    fn slug(&self) -> &'static str {
        "pipeline_raster_sample_count_degraded"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            ("count", self.count.to_string()),
            ("built_at", DEFAULT_RASTER_SAMPLE_COUNT.to_string()),
        ]
    }
}

/// Name a `rasterSampleCount` this device cannot honour. See
/// [`PipelineRasterSampleCountDegraded`].
///
/// Zero and one are the same answer — see
/// [`RenderPipelineDescriptor::raster_sample_count`] — so only `> 1` reports.
fn note_pipeline_raster_sample_count(count: u32) {
    if count <= DEFAULT_RASTER_SAMPLE_COUNT {
        return;
    }
    let drop = PipelineRasterSampleCountDegraded { count };
    crate::observe::Emit::decline("type7_pipeline", &drop).fail_once(u64::from(count));
}

/// Report the shape of a type-7 pipeline's own compact-TLV block and every
/// field in it this decoder does not consume.
///
/// Two lines with the two jobs [`note_color_entry_fields`] splits for the same
/// reason: a silent census cannot tell "the guest sends only the tags we read"
/// from "this walk never ran on a live guest".
///
/// * `type7_pipeline_shape` is the *branch*, on the off channel, deduped per
///   distinct `(kind, tag, len)` sequence and starring the unread tags. A boot
///   with pipeline shapes and no drop line is a positive reading.
/// * `pipeline_descriptor_field_dropped` is the *loss*, one typed decline per
///   distinct dropped `(tag, len)`. See [`PipelineFieldDropped`].
///
/// Pipeline descriptors are decoded once per distinct pipeline and cached, so
/// neither line is on a per-draw path.
fn note_pipeline_tlv_fields(
    kind: &'static str,
    consumed_tags: &[u8],
    benign_tags: &[u8],
    fields: &[CompactTlv],
) -> Result<(), DecodeStatus> {
    let unknown = report_tlv_shape(
        kind,
        fields.len(),
        fields.iter().map(|f| (f.tag, f.length)),
        consumed_tags,
        benign_tags,
    );
    // Outside the emit above, so the refusal does not inherit `fail_once`'s
    // latch: the line names a tag once, the pipeline is refused every time.
    if unknown > 0 {
        return Err(DecodeStatus::ErrUnsupported("res_pipeline_field_unread"));
    }
    Ok(())
}

/// [`note_pipeline_tlv_fields`] for an entry that is walked **in place** rather
/// than enumerated first.
///
/// `entry_tag_u32` and its sibling walk a `[fieldCount][tag][len][value…]*`
/// entry once per tag the caller names, so a caller reading four tags never
/// forms a list of what the entry held and structurally cannot see a fifth. The
/// vertex descriptor, its layout entries and its attribute entries are all read
/// that way. This walks the same bytes once and reports the whole shape.
///
/// Silent on a malformed entry rather than reporting a truncated shape as if it
/// were the guest's: the callers have their own out-of-bounds refusals over
/// these bytes (`res_vertex_layout_entry_oob` and its siblings), and a second
/// opinion from an instrument would be a second answer to a question already
/// answered by a refusal.
///
/// **Reports and does not refuse**, unlike its pipeline sibling, and the count
/// it discards is the difference. Every tag these entries carry is consumed —
/// six shapes across a driven boot, all `unconsumed=0` — so it has no benign
/// list to declare and nothing to refuse *yet*; promoting it needs its own
/// commit and its own reading, not a share of the pipeline block's.
fn note_entry_tlv_fields(kind: &'static str, bytes: &[u8], entry: usize, consumed_tags: &[u8]) {
    let Some(&field_count) = bytes.get(entry) else {
        return;
    };
    let mut seen: Vec<(u8, u8)> = Vec::with_capacity(field_count as usize);
    let mut p = entry + 1;
    for _ in 0..field_count {
        if p + 2 > bytes.len() {
            return;
        }
        let (tag, len) = (bytes[p], bytes[p + 1]);
        if p + 2 + len as usize > bytes.len() {
            return;
        }
        seen.push((tag, len));
        p += 2 + len as usize;
    }
    // No benign list: see this function's doc for why it refuses nothing.
    let _unknown = report_tlv_shape(
        kind,
        field_count as usize,
        seen.into_iter(),
        consumed_tags,
        &[],
    );
}

/// The half [`note_pipeline_tlv_fields`] and [`note_entry_tlv_fields`] share:
/// the shape line, the star, and one decline per unread tag.
///
/// `kind` is folded into both latches over its whole length rather than its
/// first byte. `render` and `render_mesh` share a first byte, so a first-byte
/// key would have let the two branches' drops silence each other — the same
/// class of collision as two checks sharing a decline slug, one level down.
fn report_tlv_shape(
    kind: &'static str,
    field_count: usize,
    fields: impl Iterator<Item = (u8, u8)>,
    consumed_tags: &[u8],
    benign_tags: &[u8],
) -> usize {
    let kind_key = kind
        .bytes()
        .fold(0u64, |acc, b| acc.rotate_left(7) ^ u64::from(b));
    let mut shape = String::new();
    let mut shape_key = kind_key;
    let mut dropped: Vec<(u8, u8)> = Vec::new();
    let mut unknown: Vec<(u8, u8)> = Vec::new();
    for (tag, len) in fields {
        let consumed = consumed_tags.contains(&tag);
        let sep = if shape.is_empty() { "" } else { "," };
        // `*` keeps its old meaning — this decoder does not read the tag — so
        // shapes recorded in earlier readings still say what they said. `!` is
        // the new half: unread *and* unidentified, which is the arm that
        // refuses.
        let star = if consumed { "" } else { "*" };
        let bang = if consumed || benign_tags.contains(&tag) {
            ""
        } else {
            "!"
        };
        let _ = std::fmt::Write::write_fmt(
            &mut shape,
            format_args!("{sep}{tag:02x}:{len}{star}{bang}"),
        );
        // Order-sensitive, so a reordered block reads as a different shape. The
        // tag and the length are what a reader of this block depends on; the
        // value is not, and mixing it in would make every distinct sample count
        // a distinct shape.
        shape_key = shape_key.rotate_left(9) ^ (u64::from(tag) << 8) ^ u64::from(len);
        if !consumed {
            dropped.push((tag, len));
            if !benign_tags.contains(&tag) {
                unknown.push((tag, len));
            }
        }
    }
    if crate::observe::first_sight("type7_pipeline_shape", shape_key) {
        crate::observe::off(format!(
            "type7_pipeline_shape kind={kind} nfields={field_count} tags=[{shape}] \
             unconsumed={} unknown={}",
            dropped.len(),
            unknown.len()
        ));
    }
    // Only the unidentified ones reach the fail channel. A benign drop is
    // expected control flow with a written argument behind it, and `AGENTS.md`
    // asks that expected control flow stay quiet; the shape line above is where
    // it stays visible.
    for &(tag, len) in &unknown {
        crate::observe::Emit::decline("type7_pipeline", &PipelineFieldDropped { kind, tag, len })
            .fail_once(kind_key.rotate_left(16) ^ (u64::from(tag) << 8) ^ u64::from(len));
    }
    unknown.len()
}

pub fn decode_render_pipeline_descriptor(
    bytes: &[u8],
) -> Result<RenderPipelineDescriptor, DecodeStatus> {
    if bytes.len() < TYPE7_MIN_LEN {
        return Err(DecodeStatus::ErrShort("res_render_pipeline_short"));
    }
    let obj_type = ld32(&bytes[0..]);
    let declared = ld32(&bytes[4..]) as usize;
    if obj_type != TYPE7_OBJECT_RENDER_PIPELINE {
        return Err(DecodeStatus::ErrUnsupported("res_render_pipeline_tag"));
    }
    if declared != bytes.len() || declared < TYPE7_MIN_LEN {
        return Err(DecodeStatus::ErrShort("res_render_pipeline_declared_len"));
    }
    let mut out = RenderPipelineDescriptor {
        object_id: ld32(&bytes[8..]),
        serialized_payload_len: ld32(&bytes[12..]),
        ..Default::default()
    };
    note_type7_payload_len("render", out.serialized_payload_len, declared);
    let (fields, consumed) = decode_compact_tlv_record(bytes, TYPE7_FIRST_TLVS)?;
    let tag01 = compact_tlv_u32(&fields, PIPELINE_TAG_VERTEX_FUNC).unwrap_or(0);
    let tag02 = compact_tlv_u32(&fields, PIPELINE_TAG_FRAGMENT_FUNC).unwrap_or(0);
    let tag03 = compact_tlv_u32(&fields, PIPELINE_TAG_MESH_FRAGMENT_FUNC).unwrap_or(0);
    // A property of the descriptor rather than of either role map, so it is read
    // before the shape is chosen and both branches carry it. Zero is the absent
    // case and means single-sampled; see the field's doc.
    out.raster_sample_count =
        compact_tlv_u32(&fields, PIPELINE_TAG_RASTER_SAMPLE_COUNT).unwrap_or(0);
    note_pipeline_raster_sample_count(out.raster_sample_count);
    // Mesh SPI shape: tag 0x14 section offset (host serializeMeshRenderPipelineDescriptor).
    // Classic type-7 uses tag 0x08. Roles for 0x01/0x02/0x03 differ by shape.
    if let Some(off) = compact_tlv_u32(&fields, PIPELINE_TAG_MESH_SECTION_OFFSET) {
        note_pipeline_tlv_fields(
            "render_mesh",
            &MESH_PIPELINE_TAGS_CONSUMED,
            &RENDER_PIPELINE_TAGS_BENIGN,
            &fields,
        )?;
        out.object_func_ref = tag01;
        out.mesh_func_ref = tag02;
        out.fragment_func_ref = tag03;
        out.vertex_func_ref = 0;
        out.color_attachment_offset = off;
        out.has_color_attachment_offset = true;
    } else {
        note_pipeline_tlv_fields(
            "render",
            &CLASSIC_PIPELINE_TAGS_CONSUMED,
            &RENDER_PIPELINE_TAGS_BENIGN,
            &fields,
        )?;
        out.vertex_func_ref = tag01;
        out.fragment_func_ref = tag02;
        out.object_func_ref = 0;
        out.mesh_func_ref = 0;
        // The classic role of `0x03`: where the serialized `vertexDescriptor`
        // starts, in the same units as the colour-attachment offset beside it.
        // A descriptor without this tag has no vertex descriptor at all — it is
        // not one whose offset went missing.
        if let Some(off) = compact_tlv_u32(&fields, PIPELINE_TAG_VERTEX_DESCRIPTOR_OFFSET) {
            out.vertex_descriptor_offset = off;
            out.has_vertex_descriptor_offset = true;
        }
        if let Some(off) = compact_tlv_u32(&fields, PIPELINE_TAG_COLOR_ATTACH_OFFSET) {
            out.color_attachment_offset = off;
            out.has_color_attachment_offset = true;
        }
    }
    let first_tlv_end = TYPE7_FIRST_TLVS + consumed;
    if out.has_color_attachment_offset {
        let color_abs = TYPE7_FIRST_TLVS + out.color_attachment_offset as usize;
        // The vertex block runs from where the descriptor says it starts to
        // where the colour section begins. Only the classic shape states that
        // start; the mesh shape does not, so it keeps the old inference —
        // everything between the end of the TLV block and the colour section —
        // and `note_vertex_block_inferred` is what keeps the reliance on that
        // guess measurable rather than assumed.
        let inferred = !out.has_vertex_descriptor_offset;
        let start = if inferred {
            first_tlv_end
        } else {
            TYPE7_FIRST_TLVS + out.vertex_descriptor_offset as usize
        };
        if color_abs <= declared && start < color_abs {
            out.vertex_attributes = parse_vertex_block(bytes, start, color_abs)?;
            if inferred && !out.vertex_attributes.is_empty() {
                note_vertex_block_inferred(start, color_abs);
            }
        }
        if color_abs < declared {
            out.color_attachments = parse_color_attachments(bytes, declared, color_abs)?;
            if let Some(c0) = out.color_attachments.first().copied() {
                out.color0 = c0;
            }
        }
    }
    Ok(out)
}

/// The two lengths in a type-7 header disagree about the same payload.
///
/// The header states the payload's length twice: the declared length at `+4`
/// covers the header and the payload padded to four bytes, and
/// [`RenderPipelineDescriptor::serialized_payload_len`] at `+0xc` is the same
/// payload unpadded. So `declared == TYPE7_FIRST_TLVS + round_up_4(payload)`
/// always, and a record where it does not hold is one whose two halves were
/// written by different ideas of how long it is.
///
/// Reported rather than refused, and not because the check is soft. This is a
/// *newly readable* relation — the field carried no name until it was
/// identified, so nothing has ever compared the two on live traffic — and this
/// file's own rule is that a refusal needs its zero measured first. The
/// colour-attachment walk is the worked example: it reported for as long as it
/// took to read `unconsumed=0` off driven boots, and only then refused.
///
/// Latched on the pair, so a repeating malformed shape names itself once.
///
/// **It reads zero on a driven x86/Vulkan boot**, across the twelve distinct
/// pipeline shapes that boot produces. That is the confirmation the field is
/// what this doc says: a relation derived from the header holding on every real
/// record is a stronger reading than any single value would have been.
///
/// Promoting it to a refusal needs one more step than a boot, and this is the
/// trap: most synthetic fixtures in this file leave the word zero, so they trip
/// the relation and a refusal would fail them wholesale. The fixtures have to
/// state a payload length first. A reader who measures only the boot will
/// conclude the promotion is free, and it is not.
///
/// Both pipeline subtypes reach this, and **only** those two: the other type-7
/// subtypes — sampler, depth-stencil, ICB — are fixed-layout wire structs rather
/// than property-list containers, so their fourth header word is a declared
/// field of the struct and not a payload length. The relation would be false
/// there, and asking it would be reading one format's rule into another.
///
/// **Both halves have now been booted, which is worth stating because a
/// `kind=compute` zero could have meant the arm never ran.** The compute arm was
/// added a commit after the render one and had no boot behind it; a later driven
/// x86 boot reported `type7_pipeline_shape kind=compute` twice against
/// `kind=render` four times, so the compute arm decoded two real descriptors and
/// agreed on both. The denominator to quote for this instrument is that shape
/// line, not this one, because this one is silent on success by construction.
fn note_type7_payload_len(kind: &'static str, payload: u32, declared: usize) {
    let padded = (payload as usize).next_multiple_of(4);
    if TYPE7_FIRST_TLVS.checked_add(padded) == Some(declared) {
        return;
    }
    if crate::observe::first_sight(
        "type7_payload_len_disagrees",
        ((payload as u64) << 32) | declared as u64,
    ) {
        crate::observe::fail(format!(
            "type7_payload_len kind={kind} reason=type7_payload_len_disagrees \
             payload={payload} padded={padded} declared={declared} \
             expected={} (the header states its payload length twice and the \
             two do not agree; nothing is refused, the walks below bound \
             themselves on the declared length)",
            TYPE7_FIRST_TLVS + padded
        ));
    }
}

/// A bit in a packed decoded word that no field of this decoder names, set by
/// the guest.
///
/// The tag instruments above answer "which properties arrived that nobody
/// reads"; this answers the same question for a word where the properties are
/// bit ranges. It is the harder of the two to see by reading: a tag with no arm
/// is at least a token somebody could grep for, while a bit with no field is an
/// absence in a set of shifts spread across a struct literal.
///
/// Zero is the expected reading and the only comfortable one. A set bit here is
/// guest state this device drops with no name for what it was — the loss class
/// with the least to go on, because unlike a tag it cannot even be reported by
/// number in a way that identifies the property.
///
/// **Its zero is not yet a measurement, and the denominator says so.** Both
/// callers sit inside [`parse_compute_stage_input_block`], and on a driven x86
/// boot — Safari composited over a Ventura desktop, 25 s of window drag —
/// `type7_pipeline_shape` reported two compute pipelines while
/// `compute_stage_input_decoded` reported **none**. Neither compute pipeline
/// carried a stage-input block at all, so no entry word was ever offered to this
/// function and its silence is an empty walk rather than a clean one. That is
/// exactly the confusion the denominator was added to break, and it broke it on
/// the first boot that had both halves compiled in. A workload that builds a
/// stage-input compute pipeline is what would turn this zero into a reading.
///
/// Latched per `(kind, unread bits)` rather than per word, so a field that
/// varies within the read bits does not re-report, and a *new* unread bit does.
fn note_unread_bits(kind: &'static str, word: u32, read_mask: u32) {
    let unread = word & !read_mask;
    if unread == 0 {
        return;
    }
    let kind_key = kind
        .bytes()
        .fold(0u64, |acc, b| acc.rotate_left(7) ^ u64::from(b));
    if crate::observe::first_sight(
        "packed_word_unread_bits",
        kind_key.rotate_left(32) ^ u64::from(unread),
    ) {
        crate::observe::fail(format!(
            "packed_word_unread_bits reason=packed_word_unread_bits kind={kind} \
             word={word:#010x} read={read_mask:#010x} unread={unread:#010x} \
             (the guest set bits this decoder has no field for; the state they \
              carry is dropped and there is no name for what it was)"
        ));
    }
}

/// A vertex descriptor found without the wire having said where it starts.
///
/// The classic shape carries [`PIPELINE_TAG_VERTEX_DESCRIPTOR_OFFSET`]; the mesh
/// shape does not, so the mesh arm still has to guess that the descriptor is
/// whatever sits between the end of the TLV block and the colour section. That
/// guess is what *both* arms made until the classic tag was identified, and it
/// needs [`skip_optional_label_and_pad`] to step over a `label` string of
/// unknown length first — a byte-sniffing heuristic that reads a `fieldCount`
/// of `0x20` or above as the first character of a label. The explicit offset
/// makes it unnecessary on the arm that has one, and this line is what says how
/// much traffic still depends on it.
///
/// Not a loss and not a refusal: the guess is taken and the attributes are
/// decoded. Off channel, deduped on the pair of offsets, because what a reader
/// wants is whether the fallback runs at all rather than how often. A boot that
/// never emits this is a boot on which the heuristic could be deleted.
///
/// **It reads zero on a driven x86/Vulkan boot**, on which every vertex block
/// the guest sends is located from the stated offset and the same six
/// vertex-entry shapes decode as they did from the inferred one. That is not yet
/// a licence to delete [`skip_optional_label_and_pad`]: the mesh shape is a real
/// Apple one this workload never builds, and a zero from a workload that does
/// not enter the branch says nothing about the branch. What the zero does
/// establish is that the classic arm no longer depends on byte-sniffing, which
/// is the whole of what the offset was read for.
fn note_vertex_block_inferred(start: usize, color_abs: usize) {
    if crate::observe::first_sight(
        "type7_vertex_block_inferred",
        ((start as u64) << 32) | color_abs as u64,
    ) {
        crate::observe::off(format!(
            "type7_vertex_block_inferred start={start} color={color_abs} \
             (the descriptor stated no vertexDescriptor offset, so the block was \
              located by stepping over the label)"
        ));
    }
}

/// A colour-attachment TLV field this decoder does not read, and the pipeline
/// it refuses.
///
/// The entry is `[field_count][tag][len][value…]*` and the consumed tags are
/// exactly `0x00..=0x09`: the entry's own index (`COLOR_ATTACHMENT_TAG_INDEX`)
/// and the nine properties of `MTLRenderPipelineColorAttachmentDescriptor`, in
/// the order `MTLRenderPipeline.h` declares them. So the consumed set is not a
/// subset this device chose — it is the whole descriptor, and an eleventh tag is
/// a property the decoder has no name for.
///
/// # Why this refuses rather than building the pipeline anyway
///
/// It used to emit this line and continue, so the pipeline was built with
/// Metal's default wherever the guest had set something else. That is the same
/// wrong-content-over-refusal trade this file already rejected twice in
/// `parse_one_color_entry` alone — for an attachment index past the table, and
/// for a write mask wider than four bits — and both rejections give the reason:
/// there is no second-best. A property is either the one the guest serialized or
/// it is a guess, and a guess reaches the frame with nothing to say it was one.
///
/// The refusal is not deduped even though the line is. `first_sight` latches the
/// emission so a repeating unknown tag names itself once; every pipeline
/// carrying one is still refused, because a refusal that fired once and then let
/// the same descriptor through would be worse than never refusing.
///
/// # What licenses it
///
/// A bare zero would not. `type7_color_attach_shape` is the sibling that fires:
/// it reports every entry's tag sequence and stars the unread ones, and across
/// every driven boot in the record it appears 4–13 times per boot, each one
/// `unconsumed=0`, over the tag set `00,01,02,04,07`. So the walk runs on a live
/// guest, reads the tags, and this guest sends none outside the descriptor. The
/// zero is measured rather than unreached, which is what makes refusing safe.
struct ColorAttachDropped {
    tag: u8,
}

impl crate::observe::Decline for ColorAttachDropped {
    fn slug(&self) -> &'static str {
        "color_attachment_field_dropped"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![("tag", format!("0x{:02x}", self.tag))]
    }
}

/// A colour-attachment entry whose `field_count` promised more fields than the
/// descriptor holds.
///
/// Distinct from [`ColorAttachDropped`] beside it, and refused for a different
/// reason. That one is a tag this decoder cannot name; this one is a tag it
/// cannot *read*, because the record ends first. The consequence was the same
/// and quieter: the walk stopped, `entry_tag_u32` found none of the tags past
/// the cut, and each returned the absent-field default — opaque `ONE`/`ZERO`
/// blending, no pixel format, a write mask of `all`. An attachment assembled
/// from defaults the guest never sent.
///
/// The section level of this same fault refuses three ways already —
/// `res_color_section_oob`, `res_color_entry_oob` and
/// `res_color_reach_past_record`, all reported through
/// `note_color_table_truncated`. This is the entry level of it.
///
/// `read` and `declared` are both carried because they separate the two shapes:
/// `0/4` is an entry that is entirely absent, `3/4` is one cut mid-field, and
/// only the second says the walk was reading real fields when it ran out.
struct ColorAttachEntryShort {
    read: usize,
    declared: usize,
}

impl crate::observe::Decline for ColorAttachEntryShort {
    fn slug(&self) -> &'static str {
        "color_attachment_entry_short"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![("fields", format!("{}/{}", self.read, self.declared))]
    }
}

const COLOR_ATTACHMENT_TAGS_CONSUMED: [u8; 10] = [
    COLOR_ATTACHMENT_TAG_INDEX,
    COLOR_ATTACHMENT_TAG_PIXEL_FORMAT,
    COLOR_ATTACHMENT_TAG_BLEND_ENABLE,
    COLOR_ATTACHMENT_TAG_SRC_RGB,
    COLOR_ATTACHMENT_TAG_DST_RGB,
    COLOR_ATTACHMENT_TAG_RGB_OP,
    COLOR_ATTACHMENT_TAG_SRC_ALPHA,
    COLOR_ATTACHMENT_TAG_DST_ALPHA,
    COLOR_ATTACHMENT_TAG_ALPHA_OP,
    COLOR_ATTACHMENT_TAG_WRITE_MASK,
];

/// A colour-attachment `writeMask` outside `MTLColorWriteMask`'s four bits.
///
/// This is the standing check on the tag identification itself. Tag `0x09` is
/// `writeMask` because it is the ninth property in `MTLRenderPipeline.h` and
/// tags `0x01..=0x08` are the first eight in order — an argument from the
/// header, not from the one observed value. If the tag is something else, it
/// will eventually carry a value no four-bit mask can hold, and that value
/// arrives here by name instead of quietly masking channels off.
/// A colour-attachment entry naming a slot above [`MAX_COLOR_ATTACHMENTS`].
struct ColorAttachIndexOutOfRange {
    declared: u32,
}

impl crate::observe::Decline for ColorAttachIndexOutOfRange {
    fn slug(&self) -> &'static str {
        "color_attachment_index_out_of_range"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![("declared", self.declared.to_string())]
    }
}

struct ColorWriteMaskOutOfRange;

impl crate::observe::Decline for ColorWriteMaskOutOfRange {
    fn slug(&self) -> &'static str {
        "color_write_mask_out_of_range"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        Vec::new()
    }
}

/// The largest field value that rides in the dedup key.
///
/// A dropped field's *value distribution* is the identifying signal — an
/// `MTLColorWriteMask` only ever takes 0..=15, which names the field from the
/// wire alone without knowing its tag in advance. So small values each get
/// their own line. A field carrying wide values would otherwise make this
/// unbounded, so above the cap the key collapses to the tag and only the first
/// value seen is printed.
const COLOR_ATTACH_DROP_VALUE_CAP: u32 = 64;

/// Report the shape of one colour-attachment entry and every field in it this
/// decoder does not consume.
///
/// Two lines with different jobs, because a silent census cannot distinguish
/// "the guest sends nothing but the eight tags we read" from "this walk never
/// ran on a live guest":
///
/// * `type7_color_attach_shape` is the *branch*, deduped per distinct `(tag,
///   len)` sequence. A boot with entries but no drop line is then a positive
///   reading — the entries were seen and carried only consumed tags — rather
///   than an absence.
/// * `color_attachment_field_dropped` is the *loss*, one typed decline per
///   dropped `(tag, len, value)`. A serialized field we never read is guest
///   intent discarded, which the ground rules say must not be silent.
///
/// Pipeline descriptors are decoded once per distinct pipeline and cached, so
/// this walk is not on a per-draw path.
fn note_color_entry_fields(bytes: &[u8], entry: usize, slot: u32) -> Result<(), DecodeStatus> {
    if entry >= bytes.len() {
        return Ok(());
    }
    let field_count = bytes[entry] as usize;
    let mut p = entry + 1;
    let mut shape = String::new();
    let mut shape_key: u64 = 0;
    let mut dropped: Vec<(u8, u8, u32)> = Vec::new();
    let mut short = None;
    for read in 0..field_count {
        // A `[tag][len]` header, or a value, that runs past the descriptor. The
        // entry's own `field_count` promised a field the record does not
        // contain, so this is a malformed entry rather than a short one — and
        // the fields *after* the break are unreadable, not absent.
        //
        // Both used to `break`, which left the walk quiet and let the parse
        // below apply `entry_tag_u32`'s defaults for every tag past the cut: an
        // attachment with opaque `ONE`/`ZERO` blending, no pixel format and a
        // write mask of `all`, none of which the guest asked for. The section
        // level of this same fault already refuses three ways
        // (`res_color_section_oob`, `res_color_entry_oob`,
        // `res_color_reach_past_record`); the entry level did not.
        if p + 2 > bytes.len() {
            short = Some((read, p));
            break;
        }
        let tag = bytes[p];
        let field_len = bytes[p + 1] as usize;
        if p + 2 + field_len > bytes.len() {
            short = Some((read, p));
            break;
        }
        let value = if field_len >= 4 {
            ld32(&bytes[p + 2..])
        } else {
            0
        };
        let consumed = COLOR_ATTACHMENT_TAGS_CONSUMED.contains(&tag);
        let sep = if shape.is_empty() { "" } else { "," };
        let star = if consumed { "" } else { "*" };
        let _ = std::fmt::Write::write_fmt(
            &mut shape,
            format_args!("{sep}{tag:02x}:{field_len}{star}"),
        );
        // Order-sensitive so a reordered entry reads as a different shape; the
        // tag and length are what the walk depends on, the value is not.
        shape_key = shape_key.rotate_left(9) ^ (u64::from(tag) << 8) ^ (field_len as u64);
        if !consumed {
            dropped.push((tag, field_len as u8, value));
        }
        p += 2 + field_len;
    }
    if crate::observe::first_sight("type7_color_attach_shape", shape_key) {
        crate::observe::off(format!(
            "type7_color_attach_shape slot={slot} nfields={field_count} \
             tags=[{shape}] unconsumed={}",
            dropped.len()
        ));
    }
    // Before the dropped-tag refusal, because a truncated entry is why a tag
    // might be missing rather than a second, independent fault.
    if let Some((read, at)) = short {
        if crate::observe::first_sight(
            "color_attachment_entry_short",
            ((slot as u64) << 40) | ((read as u64) << 16) | (field_count as u64),
        ) {
            crate::observe::Emit::decline(
                "type7_color_attach",
                &ColorAttachEntryShort {
                    read,
                    declared: field_count,
                },
            )
            .field("slot", slot)
            .field("at", at)
            .field("reach", bytes.len())
            .fail();
        }
        return Err(DecodeStatus::ErrShort("res_color_entry_fields_short"));
    }
    let unread = !dropped.is_empty();
    for (tag, field_len, value) in dropped {
        let keyed_value = if value <= COLOR_ATTACH_DROP_VALUE_CAP {
            u64::from(value)
        } else {
            u64::from(COLOR_ATTACH_DROP_VALUE_CAP) + 1
        };
        let disc = (u64::from(tag) << 40) | (u64::from(field_len) << 32) | keyed_value;
        if !crate::observe::first_sight("color_attachment_field_dropped", disc) {
            continue;
        }
        crate::observe::Emit::decline("type7_color_attach", &ColorAttachDropped { tag })
            .field("slot", slot)
            .field("len", field_len)
            .field("value", value)
            .fail();
    }
    // Outside the loop, so the refusal does not inherit `first_sight`'s latch:
    // the line names a tag once, the pipeline is refused every time.
    if unread {
        return Err(DecodeStatus::ErrUnsupported("res_color_field_unread"));
    }
    Ok(())
}

/// `position` is the entry's index in the section's offset table, used only
/// when the entry does not carry [`COLOR_ATTACHMENT_TAG_INDEX`]. Defaulting an
/// absent index to 0 the way the vertex-layout sibling does would collapse every
/// attachment onto slot 0, which is worse than the position it replaces.
fn parse_one_color_entry(
    bytes: &[u8],
    entry: usize,
    position: u32,
) -> Result<PipelineColorAttachment, DecodeStatus> {
    let slot = match entry_tag_u32_present(bytes, entry, COLOR_ATTACHMENT_TAG_INDEX) {
        Some(declared) if (declared as usize) < MAX_COLOR_ATTACHMENTS => {
            if declared != position {
                // The case this decoder could not previously see: the guest's
                // attachments are not a dense in-order prefix, so every consumer
                // that matches `a.slot == c.slot` was reading another slot's
                // blend state, write mask and pixel format.
                crate::runtime::drain::note_store_route("type7_color_slot_off_position");
            }
            declared
        }
        Some(declared) => {
            // A slot this device cannot represent, and there is no second-best.
            // Falling back to `position` is exactly the aliasing to avoid — a
            // position is itself a slot 0..7, so this entry's blend state, write
            // mask and pixel format would be bound to a real attachment the guest
            // named nothing about. Refuse the pipeline instead.
            if crate::observe::first_sight(
                "color_attachment_index_out_of_range",
                u64::from(declared),
            ) {
                crate::observe::Emit::decline(
                    "type7_color_attach",
                    &ColorAttachIndexOutOfRange { declared },
                )
                .field("position", position)
                .field("max", MAX_COLOR_ATTACHMENTS)
                .fail();
            }
            return Err(DecodeStatus::ErrUnsupported("res_color_slot_over"));
        }
        None => position,
    };
    note_color_entry_fields(bytes, entry, slot)?;
    let mut out = PipelineColorAttachment {
        slot,
        src_rgb: BLEND_FACTOR_ONE,
        dst_rgb: BLEND_FACTOR_ZERO,
        op_rgb: BLEND_OP_ADD,
        src_alpha: BLEND_FACTOR_ONE,
        dst_alpha: BLEND_FACTOR_ZERO,
        op_alpha: BLEND_OP_ADD,
        ..Default::default()
    };
    let pf = entry_tag_u32(bytes, entry, COLOR_ATTACHMENT_TAG_PIXEL_FORMAT, u32::MAX);
    if pf != u32::MAX {
        out.has_pixel_format = true;
        out.pixel_format = pf;
    }
    out.blending_enabled = entry_tag_u32(bytes, entry, COLOR_ATTACHMENT_TAG_BLEND_ENABLE, 0) != 0;
    out.src_rgb = entry_tag_u32(bytes, entry, COLOR_ATTACHMENT_TAG_SRC_RGB, BLEND_FACTOR_ONE);
    out.dst_rgb = entry_tag_u32(
        bytes,
        entry,
        COLOR_ATTACHMENT_TAG_DST_RGB,
        BLEND_FACTOR_ZERO,
    );
    out.op_rgb = entry_tag_u32(bytes, entry, COLOR_ATTACHMENT_TAG_RGB_OP, BLEND_OP_ADD);
    out.src_alpha = entry_tag_u32(
        bytes,
        entry,
        COLOR_ATTACHMENT_TAG_SRC_ALPHA,
        BLEND_FACTOR_ONE,
    );
    out.dst_alpha = entry_tag_u32(
        bytes,
        entry,
        COLOR_ATTACHMENT_TAG_DST_ALPHA,
        BLEND_FACTOR_ZERO,
    );
    out.op_alpha = entry_tag_u32(bytes, entry, COLOR_ATTACHMENT_TAG_ALPHA_OP, BLEND_OP_ADD);
    // An entry that omits the tag left the property at its default, which for
    // `MTLColorWriteMask` is `all` — the same thing `ColorWriteMask::default()`
    // says, so the absent case needs no branch.
    if let Some(mask) = entry_tag_u32_present(bytes, entry, COLOR_ATTACHMENT_TAG_WRITE_MASK) {
        let Some(decoded) = ColorWriteMask::new(mask) else {
            // `MTLColorWriteMask` is four bits, so a wider value is not a mask
            // Apple's serializer can emit — the same class of malformed record
            // as an attachment index past the table, and refused the same way.
            //
            // Keeping the default here instead was a wrong pixel rather than a
            // refusal: the default is `all`, the *widest* mask there is, so a
            // guest that masked a channel off got it written. There is no
            // second-best to fall back to for the same reason the slot above
            // has none — every representable mask is a mask the guest might
            // have meant, and picking one is guessing which channels it wanted.
            if crate::observe::first_sight("color_write_mask_out_of_range", u64::from(mask)) {
                crate::observe::Emit::decline("type7_color_attach", &ColorWriteMaskOutOfRange)
                    .field("slot", slot)
                    .field("value", mask)
                    .fail();
            }
            return Err(DecodeStatus::ErrUnsupported("res_color_write_mask_over"));
        };
        out.write_mask = decoded;
    }
    Ok(out)
}
/// A color-attachment section that named more entries than it delivered.
///
/// The section is `[count:u32][entry_offset:u32 × count]`, each offset relative
/// to the section start. `count` above [`MAX_COLOR_ATTACHMENTS`], an offset word
/// running past the descriptor, or an entry offset resolving outside it, all
/// mean the same thing: the pixel format and blend state the guest serialized
/// for a slot never reaches the pipeline, and that slot silently takes
/// `parse_one_color_entry`'s defaults — opaque `ONE`/`ZERO`, blending off.
///
/// Named because the alternative is indistinguishable downstream from a guest
/// that declared fewer attachments, which is the shape a wrong blend or a
/// missing render target would arrive in. Each of the three now refuses the
/// descriptor after emitting this, so the line stands beside a refusal rather
/// than beside a pipeline that was built anyway.
///
/// A healthy zero on this workload. A driven x86/Vulkan boot — Safari window
/// drag, 25 s, ~500 draws/s, desktop and window compositing on screen — read
/// this, [`ColorAttachIndexOutOfRange`] and [`VertexDescriptorTruncated`] all
/// unfired, with no `desc_decode` refusal from any exit. The refusals cost that
/// boot nothing; a firing is the bug.
struct ColorAttachTableTruncated {
    /// `None` when the section header itself did not fit, so the count was never
    /// readable and an unknown number of attachments were lost.
    declared: Option<usize>,
    decoded: usize,
}

impl crate::observe::Decline for ColorAttachTableTruncated {
    fn slug(&self) -> &'static str {
        "color_attachment_table_truncated"
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        vec![
            (
                "declared",
                self.declared
                    .map_or_else(|| "unreadable".to_string(), |d| d.to_string()),
            ),
            ("decoded", self.decoded.to_string()),
        ]
    }
}

/// Report a color-attachment table that lost entries. Deduped per distinct
/// (declared, decoded) pair — a malformed descriptor replayed every frame would
/// otherwise flood the log with one line per draw.
fn note_color_table_truncated(
    declared: Option<usize>,
    decoded: usize,
    section_off: usize,
    len: usize,
) {
    let disc = ((declared.unwrap_or(usize::MAX) as u64) << 32) | decoded as u64;
    if !crate::observe::first_sight("color_attachment_table_truncated", disc) {
        return;
    }
    crate::observe::Emit::decline(
        "type7_color_attach",
        &ColorAttachTableTruncated { declared, decoded },
    )
    .field("section_off", section_off)
    .field("desc_len", len)
    .fail();
}

/// `MTLRenderPipelineDescriptor.colorAttachments` is an eight-slot array, so a
/// section naming more than eight is malformed rather than something we chose
/// not to read. Same bound as `render::PASS_MAX_COLOR_ATTACHMENTS`, stated here
/// because this is the pipeline-descriptor side of the same Metal limit.
const MAX_COLOR_ATTACHMENTS: usize = 8;
// The doc above says "same bound" and then writes the number again, which is the
// one thing that cannot keep it true. The pass side derives its copy from the
// wire record's own array width (`wire::RENDER_PASS_COLOR_ATTACHMENTS`), so this
// pin makes the claim checkable: if Apple's record ever carries more slots, the
// two sides part at compile time instead of this one refusing pipeline
// descriptors the pass decoder happily accepts.
const _: () =
    assert!(MAX_COLOR_ATTACHMENTS == crate::runtime::decode::render::PASS_MAX_COLOR_ATTACHMENTS);

/// Parse all color-attachment entries.
///
/// The slot is the index the entry declares in [`COLOR_ATTACHMENT_TAG_INDEX`],
/// not its position in this offset table. The two agree whenever the guest
/// serializes a dense in-order prefix, which is why the position stood in for
/// the index for so long without a visible symptom; they part as soon as it
/// does not, and every consumer of the result selects by `slot`.
///
/// `section_off == 0` is the descriptor saying it has no color section at all —
/// expected control flow, and quiet, and the only way to get an empty table.
/// Every other way of ending up with fewer attachments than the count promised
/// is a loss: it says so through [`ColorAttachTableTruncated`] and then refuses,
/// because a pipeline built from a short table is a *valid* pipeline whose
/// missing attachments read as a guest that declared fewer — the wrong blend or
/// the absent render target arrives with nothing naming it.
///
/// `len` is a **reach within `bytes`**, not a second name for its length: a
/// section may be bounded short of the record's end, and the caller is the only
/// thing that knows where.
///
/// It is applied by narrowing — `bytes.get(..len)` — rather than by being
/// carried alongside the slice, and that is the whole of the fix here. This
/// family used to thread `len` down through four helpers, each of which
/// compared an offset against `len` and then indexed `bytes`; the two are the
/// same thing only while `len <= bytes.len()`, and nothing checked it.
/// [`parse_vertex_block`] takes the same shape of bound and had always carried
/// the equivalent clause, so a `len` past the end passed every guard here and
/// panicked on the first `ld32`. Nothing guest-side could produce it — the one
/// caller has already required `declared == bytes.len()` — but a `pub fn` whose
/// totality rests on an invariant stated nowhere is one caller away from it.
///
/// After the narrowing there is exactly one length in scope and it is the
/// slice's own, so the class cannot come back by someone adding a check against
/// the wrong number.
pub fn parse_color_attachments(
    bytes: &[u8],
    len: usize,
    section_off: usize,
) -> Result<Vec<PipelineColorAttachment>, DecodeStatus> {
    let mut out = Vec::new();
    if section_off == 0 {
        return Ok(out);
    }
    // Narrow to the reach *once*, here, and every check below is against a slice
    // Rust bounds for us. Threading `len` down instead is what let this family
    // check one number and index another.
    let Some(bytes) = bytes.get(..len) else {
        return Err(DecodeStatus::ErrShort("res_color_reach_past_record"));
    };
    // The header is the count plus the first entry's offset word. A section the
    // descriptor cannot contain loses an unreadable number of attachments, which
    // the count mismatch below cannot see, so it is reported here.
    if section_off + 8 > bytes.len() {
        note_color_table_truncated(None, 0, section_off, bytes.len());
        return Err(DecodeStatus::ErrShort("res_color_section_oob"));
    }
    let declared = ld32(&bytes[section_off..]) as usize;
    // `MTLRenderPipelineDescriptor.colorAttachments` has eight subscripts, so a
    // ninth is a descriptor Metal cannot hold and this decoder cannot place.
    if declared > MAX_COLOR_ATTACHMENTS {
        note_color_table_truncated(
            Some(declared),
            MAX_COLOR_ATTACHMENTS,
            section_off,
            bytes.len(),
        );
        return Err(DecodeStatus::ErrUnsupported("res_color_count_over"));
    }
    for i in 0..declared {
        // Two ways the table can stop short of what the count promised: its own
        // offset word runs past the descriptor, or the entry that word points at
        // does. Both used to `break`, which left a shorter table and one line
        // saying so; both now carry that line and refuse.
        let offloc = section_off + 4 + i * 4;
        let entry = if offloc + 4 > bytes.len() {
            None
        } else {
            section_off
                .checked_add(ld32(&bytes[offloc..]) as usize)
                .filter(|e| *e < bytes.len())
        };
        let Some(entry) = entry else {
            note_color_table_truncated(Some(declared), out.len(), section_off, bytes.len());
            return Err(DecodeStatus::ErrShort("res_color_entry_oob"));
        };
        out.push(parse_one_color_entry(bytes, entry, i as u32)?);
    }
    Ok(out)
}

/// A heap-placed texture record, opcode [`HEAP_TEXTURE_OPCODE`].
///
/// It shares the type-8 object tag with the texture views, so it arrives at the
/// same peek, but it is a complete texture resource: a heap ref, the same
/// 32-byte `PGSerializedTextureDescriptor` a plain creation carries, and where
/// in the heap to put it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapTextureRecord<'a> {
    /// Ref of the heap the texture is placed in. Never 0 for a well-formed
    /// record; the caller decides what a 0 means for its own path.
    pub heap_ref: u32,
    /// Whether [`HeapTextureRecord::offset`] is the guest's request or is to be
    /// ignored. The serializer writes the offset either way.
    pub use_offset: bool,
    /// Byte offset into the heap.
    pub offset: u64,
    /// The embedded descriptor, for
    /// [`crate::runtime::heap_query::decode_serialized_texture_descriptor`] —
    /// or its wide sibling when [`HeapTextureRecord::wide`] is set.
    pub descriptor: &'a [u8],
    /// Which of the two descriptor bodies [`HeapTextureRecord::descriptor`]
    /// holds, taken from the record's **opcode**.
    ///
    /// Carried rather than inferred from the slice length on purpose: the two
    /// bodies are 32 and 40 bytes, and a reader that picks by length is one
    /// record-length change away from decoding the wrong layout in silence.
    pub wide: bool,
}

/// Decode a heap-placed texture record.
///
/// The layout is pinned by `reims_vgpu_wire::ops::heap_texture` against bytes
/// Apple's serializer produced. Split out of `compute_exec`, where it was open
/// coded, so it can be tested at all: the interesting part is not the offsets
/// but `use_offset`, which is **one bit** of its byte rather than a word — the
/// seven bits above it and the three bytes to [`HEAP_TEXTURE_OFFSET`] are
/// whatever the guest's ring last contained, so a 32-bit load there reads noise
/// into 31 of its bits. The open-coded read got that wrong and had no test.
/// `NewHeapTextureBody::use_offset` applies the mask, and this decodes through
/// it rather than restating it.
pub fn decode_heap_texture(bytes: &[u8]) -> Result<HeapTextureRecord<'_>, DecodeStatus> {
    let op = reims_vgpu_wire::op(bytes, 0)
        .map_err(|_| DecodeStatus::ErrShort("res_heap_texture_len"))?;
    // Dispatch on the opcode, then require the length that opcode implies. The
    // wide form is a different opcode rather than a longer record.
    match op.opcode() {
        HEAP_TEXTURE_OPCODE => {
            if bytes.len() != HEAP_TEXTURE_LEN {
                return Err(DecodeStatus::ErrShort("res_heap_texture_len"));
            }
            let b = w_heap::new_heap_texture(&op)
                .map_err(|_| DecodeStatus::ErrShort("res_heap_texture_len"))?;
            let desc_at = OP_HDR + offset_of!(w_heap::NewHeapTextureBody, desc);
            let use_offset_at = OP_HDR + offset_of!(w_heap::NewHeapTextureBody, use_offset_bits);
            Ok(HeapTextureRecord {
                heap_ref: b.heap_ref.get(),
                use_offset: b.use_offset(),
                offset: b.offset.get(),
                descriptor: &bytes[desc_at..use_offset_at],
                wide: false,
            })
        }
        HEAP_TEXTURE_WIDE_OPCODE => {
            if bytes.len() != HEAP_TEXTURE_WIDE_LEN {
                return Err(DecodeStatus::ErrShort("res_heap_texture_len"));
            }
            let b = w_heap::new_heap_texture_wide(&op)
                .map_err(|_| DecodeStatus::ErrShort("res_heap_texture_len"))?;
            let desc_at = OP_HDR + offset_of!(w_heap::NewHeapTextureWideBody, desc);
            let use_offset_at =
                OP_HDR + offset_of!(w_heap::NewHeapTextureWideBody, use_offset_bits);
            Ok(HeapTextureRecord {
                heap_ref: b.heap_ref.get(),
                use_offset: b.use_offset(),
                offset: b.offset.get(),
                descriptor: &bytes[desc_at..use_offset_at],
                wide: true,
            })
        }
        _ => Err(DecodeStatus::ErrUnsupported("res_heap_texture_opcode")),
    }
}

pub fn decode_texture_view_descriptor(bytes: &[u8]) -> Result<TextureViewDescriptor, DecodeStatus> {
    let op = reims_vgpu_wire::op(bytes, 0)
        .map_err(|_| DecodeStatus::ErrShort("res_texture_view_short"))?;
    let view_opcode = op.opcode();
    let declared = op.length() as usize;
    let min_len = match view_opcode {
        TEXTURE_VIEW_OPCODE_SIMPLE => TEXTURE_VIEW_MIN_SIMPLE,
        TEXTURE_VIEW_OPCODE_RANGED => TEXTURE_VIEW_MIN_RANGED,
        TEXTURE_VIEW_OPCODE_SWIZZLE => TEXTURE_VIEW_MIN_SWIZZLE,
        _ => return Err(DecodeStatus::ErrUnsupported("res_texture_view_opcode")),
    };
    if declared < min_len || declared != bytes.len() {
        return Err(DecodeStatus::ErrShort("res_texture_view_declared_len"));
    }

    match view_opcode {
        TEXTURE_VIEW_OPCODE_SIMPLE => {
            let b = w_view::texture_view(&op)
                .map_err(|_| DecodeStatus::ErrShort("res_texture_view_short"))?;
            let pixel_format = b.pixel_format.get();
            Ok(TextureViewDescriptor {
                view_opcode,
                view_texture_ref: b.object_ref.get(),
                base_texture_ref: b.base_texture_ref.get(),
                pixel_format,
                ..Default::default()
            })
        }
        TEXTURE_VIEW_OPCODE_RANGED => {
            let b = w_view::texture_view_ranged(&op)
                .map_err(|_| DecodeStatus::ErrShort("res_texture_view_short"))?;
            let pixel_format = b.pixel_format.get();
            Ok(TextureViewDescriptor {
                view_opcode,
                view_texture_ref: b.object_ref.get(),
                base_texture_ref: b.base_texture_ref.get(),
                pixel_format,
                texture_type: b.texture_type.get(),
                level_base: b.level_base.get(),
                level_count: b.level_count.get(),
                slice_base: b.slice_base.get(),
                slice_count: b.slice_count.get(),
                ..Default::default()
            })
        }
        TEXTURE_VIEW_OPCODE_SWIZZLE => {
            let b = w_view::texture_view_swizzle(&op)
                .map_err(|_| DecodeStatus::ErrShort("res_texture_view_short"))?;
            let r = &b.ranged;
            let pixel_format = r.pixel_format.get();
            Ok(TextureViewDescriptor {
                view_opcode,
                view_texture_ref: r.object_ref.get(),
                base_texture_ref: r.base_texture_ref.get(),
                pixel_format,
                texture_type: r.texture_type.get(),
                level_base: r.level_base.get(),
                level_count: r.level_count.get(),
                slice_base: r.slice_base.get(),
                slice_count: r.slice_count.get(),
                swizzle: [
                    b.swizzle.red,
                    b.swizzle.green,
                    b.swizzle.blue,
                    b.swizzle.alpha,
                ],
            })
        }
        _ => Err(DecodeStatus::ErrUnsupported("res_texture_view_opcode")),
    }
}

/// A texture aliased over an MTLBuffer's storage — object type 8, view_opcode 9
/// (`newTextureWithDescriptor:offset:bytesPerRow:`). Distinct from a texture view:
/// the source ref is a BUFFER and the sampled bytes come straight from that
/// buffer's guest storage at `offset`, `bytes_per_row` stride.
///
/// The trailing 32 bytes are the same `PGSerializedTextureDescriptor` a plain
/// texture creation carries, so they are carried as one rather than flattened —
/// which is also the shape `reims_vgpu_wire::ops::backed_texture` derived from
/// Apple's bytes (`BufferTextureBody { object_ref, buffer_ref, offset,
/// bytes_per_row, desc }`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferTextureDescriptor {
    pub new_texture_ref: u32,
    pub buffer_ref: u32,
    pub offset: u64,
    pub bytes_per_row: u64,
    /// The embedded texture descriptor, read by
    /// [`crate::runtime::heap_query::decode_serialized_texture_descriptor`] —
    /// the single reader that function's own doc asks every path to use.
    pub desc: crate::runtime::heap_query::TextureDescriptor,
}

/// Decode the opcode-9 (buffer-backed texture) type-8 descriptor — the
/// serialized form of
/// `newTextureWithBuffer:descriptor:offset:bytesPerRow:allocator:`.
///
/// The embedded descriptor is handed to the shared decoder rather than read
/// here. It used to be re-derived inline, and the two readings agreed on every
/// offset they shared — but this one stopped after `texture_type`,
/// `pixel_format` and the geometry, so `usage`, `resource_options`,
/// `protection_options` and the three descriptor flag bits were decoded by the
/// serializer and dropped by this device. That is the divergence shape this
/// repository keeps finding: two consumers of one wire form, one of which
/// contradicts a rule the other one states in a comment. The rule is on
/// `decode_serialized_texture_descriptor` ("keeping one decoder prevents the
/// query and resource paths from drifting"), and there is now one decoder.
pub fn decode_buffer_texture_descriptor(
    bytes: &[u8],
) -> Result<BufferTextureDescriptor, DecodeStatus> {
    let op = reims_vgpu_wire::op(bytes, 0)
        .map_err(|_| DecodeStatus::ErrShort("res_buffer_texture_short"))?;
    // Exactly the length this opcode implies — see `decode_heap_texture`.
    match op.opcode() {
        TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE => {
            if bytes.len() != BUF_TEX_MIN_LEN {
                return Err(DecodeStatus::ErrShort("res_buffer_texture_short"));
            }
            if op.length() as usize != BUF_TEX_MIN_LEN {
                return Err(DecodeStatus::ErrShort("res_buffer_texture_declared_len"));
            }
            let b = w_backed::buffer_texture(&op)
                .map_err(|_| DecodeStatus::ErrShort("res_buffer_texture_short"))?;
            let body_at = OP_HDR + offset_of!(w_backed::BufferTextureBody, desc);
            let desc = heap_query::decode_serialized_texture_descriptor(
                &bytes[body_at..body_at + heap_query::TEXTURE_BODY_LEN],
            )
            .map_err(|_| DecodeStatus::ErrShort("res_buffer_texture_body"))?;
            Ok(BufferTextureDescriptor {
                new_texture_ref: b.object_ref.get(),
                buffer_ref: b.buffer_ref.get(),
                offset: b.offset.get(),
                bytes_per_row: b.bytes_per_row.get(),
                desc,
            })
        }
        TEXTURE_VIEW_OPCODE_BUFFER_TEXTURE_WIDE => {
            if bytes.len() != BUF_TEX_WIDE_LEN {
                return Err(DecodeStatus::ErrShort("res_buffer_texture_short"));
            }
            if op.length() as usize != BUF_TEX_WIDE_LEN {
                return Err(DecodeStatus::ErrShort("res_buffer_texture_declared_len"));
            }
            let b = w_backed::buffer_texture_wide(&op)
                .map_err(|_| DecodeStatus::ErrShort("res_buffer_texture_short"))?;
            let body_at = OP_HDR + offset_of!(w_backed::BufferTextureWideBody, desc);
            let desc = heap_query::decode_wide_serialized_texture_descriptor(
                &bytes[body_at..body_at + heap_query::WIDE_TEXTURE_BODY_LEN],
            )
            .map_err(|_| DecodeStatus::ErrShort("res_buffer_texture_body"))?;
            Ok(BufferTextureDescriptor {
                new_texture_ref: b.object_ref.get(),
                buffer_ref: b.buffer_ref.get(),
                offset: b.offset.get(),
                bytes_per_row: b.bytes_per_row.get(),
                desc,
            })
        }
        _ => Err(DecodeStatus::ErrUnsupported("res_buffer_texture_opcode")),
    }
}

/// Peek the raw `(view_opcode, declared length)` header of a type-8 descriptor
/// (opcode 9 = buffer-backed texture, 7/8/0x1b = texture view). `None` only for
/// a blob too short to hold the header.
///
/// The bound is [`OP_HDR`] — the bytes this actually reads — and not any one
/// variant's total length. The type-8 forms do not share a length (20 / 36 / 44
/// / 64 / 72 …), so guarding a header peek with one of those totals hides every
/// shorter variant behind `None` before its own decoder ever sees the opcode.
/// That is the same mistake `compute_stage_tex` had to unpick at its call site,
/// where checking the narrow heap-texture length would have rejected every wide
/// record.
///
/// Neither word is validated against the blob, deliberately: the callers that
/// want both use them for the length-mismatch and unknown-opcode census, which
/// needs the guest's declared value precisely when it disagrees with what
/// arrived. [`decode_texture_view_descriptor`] is the checked reader.
pub fn texture_type8_header(bytes: &[u8]) -> Option<(u32, u32)> {
    if bytes.len() < OP_HDR {
        return None;
    }
    Some((
        ld32(&bytes[TEXTURE_VIEW_DESC_OPCODE..]),
        ld32(&bytes[TEXTURE_VIEW_DESC_LEN..]),
    ))
}

/// The opcode half of [`texture_type8_header`], for callers routing on the
/// variant alone.
pub fn texture_type8_opcode(bytes: &[u8]) -> Option<u32> {
    texture_type8_header(bytes).map(|(opcode, _)| opcode)
}

const TYPE7_MIN_LEN: usize = 17;

/// **Unused on the x86 PCI pathway.** A probe placed at the top of this function
/// — before the length check, so a short record would also report — emitted
/// nothing across a full interactive session. Type-11 geometry on that pathway
/// is latched from the **type-4** surface backing descriptor instead
/// (`runtime/objects`, `decode_type4_surface` -> `set_mapping_geom`). Do not
/// reason about what the guest tells us at surface-create time from this
/// decoder without re-confirming it runs; measure `decode_type4_surface`.
/// Offsets in the type-11 IOSurface-texture descriptor. Named rather than
/// written as hex at the read sites, which is the convention every other
/// decoder in this file already follows — an offset that appears only as a
/// literal cannot be found by a reader checking whether the layout still holds.
pub const IOSURFACE_TEX_MAPPING_ID: usize = 0x00;
pub const IOSURFACE_TEX_OBJECT_REF: usize = 0x10;
pub const IOSURFACE_TEX_PIXEL_FORMAT: usize = 0x16;
pub const IOSURFACE_TEX_WIDTH: usize = 0x18;
pub const IOSURFACE_TEX_HEIGHT: usize = 0x1c;
/// Through `height`. Live blobs run longer (0x38 / 0x58); the tail past this is
/// not decoded, and the comment on the decoder says why.
///
/// Whether that tail carries guest intent is **open, and an x86 boot cannot
/// settle it.** A probe emitting every distinct (length, tail) shape from this
/// decoder was run against a driven x86 PCI boot and emitted nothing at all,
/// which confirms rather than contradicts
/// [`crate::runtime::objects::undecoded_type4_surface_bytes`] — that doc already
/// records that this decoder does not run on that pathway. The 0x38/0x58 blobs
/// are an arm64 phenomenon.
///
/// Settling it needs the same Apple host as the other arm64-only questions:
/// re-add a dedup-by-(len, tail) probe here, boot `reims-vgpu-mmio`, and read
/// the shapes out of the fail log. Do not re-run it on x86 — that measurement
/// has been made and its answer is "not observable here".
pub const IOSURFACE_TEX_MIN_LEN: usize = 0x20;

pub fn decode_iosurface_texture_descriptor(bytes: &[u8]) -> Result<Descriptor, DecodeStatus> {
    // Matches reims-vgpu-iosurface-pages texture descriptor min layout (mappingID,
    // object self-ref, format, width, height). Live type-11 blobs are longer
    // (0x38/0x58); multi-mip level records are **not** part of this object type
    // — Metal forbids mipmapped IOSurface textures
    // (`newTextureWithDescriptor:iosurface:` rejects mipmapLevelCount > 1),
    // and product resolve fail-closes non-zero levels rather than inventing
    // a pyramid packing in the mapping (see blit_exec::Type11Texture).
    if bytes.len() < IOSURFACE_TEX_MIN_LEN {
        return Err(DecodeStatus::ErrShort("res_iosurface_short"));
    }
    Ok(Descriptor::IOSurfaceTexture {
        mapping_id: ld32(&bytes[IOSURFACE_TEX_MAPPING_ID..]),
        object_ref: ld32(&bytes[IOSURFACE_TEX_OBJECT_REF..]),
        pixel_format: ld16(&bytes[IOSURFACE_TEX_PIXEL_FORMAT..]),
        width: ld32(&bytes[IOSURFACE_TEX_WIDTH..]),
        height: ld32(&bytes[IOSURFACE_TEX_HEIGHT..]),
    })
}

fn section_range_fits(len: usize, start: u64, count: u32, entry_size: usize) -> bool {
    if count == 0 {
        return true;
    }
    let bytes = (count as u64).saturating_mul(entry_size as u64);
    start
        .checked_add(bytes)
        .map(|end| end <= len as u64)
        .unwrap_or(false)
}

fn section_ranges_overlap(a_start: u64, a_len: u64, b_start: u64, b_len: u64) -> bool {
    if a_len == 0 || b_len == 0 {
        return false;
    }
    let a_end = a_start.saturating_add(a_len);
    let b_end = b_start.saturating_add(b_len);
    a_start < b_end && b_start < a_end
}

/// Parse optional MetalSerializer compute stage-input block after first TLVs.
///
/// Returns `Ok(None)` when no valid block is present (short / length mismatch).
/// Returns `Err` only when the block header claims a valid payload but entry
/// ranges are out of bounds or overlapping (fail-closed structural error).
pub fn parse_compute_stage_input_block(
    bytes: &[u8],
    block_start: usize,
) -> Result<Option<ComputeStageInputDescriptor>, DecodeStatus> {
    if block_start >= bytes.len() {
        return Ok(None);
    }
    let bo = skip_optional_label_and_pad(bytes, block_start);
    if bo >= bytes.len() {
        return Ok(None);
    }
    let block_len = bytes.len() - bo;
    if block_len < COMPUTE_STAGE_INPUT_MIN_LEN {
        return Ok(None);
    }
    let word0 = ld32(&bytes[bo + COMPUTE_STAGE_INPUT_WORD0..]);
    let header0 = ld32(&bytes[bo + COMPUTE_STAGE_INPUT_HEADER0..]);
    let header1 = ld32(&bytes[bo + COMPUTE_STAGE_INPUT_HEADER1..]);
    let declared_payload = header0 & COMPUTE_STAGE_INPUT_HEADER0_LEN_MASK;
    // header0 low16 is payload length after word0; total block = word0 + payload.
    if (declared_payload as u64).saturating_add(4) != block_len as u64 {
        return Ok(None);
    }

    let attr_count = (header0 >> COMPUTE_STAGE_INPUT_HEADER0_ATTR_COUNT_SHIFT)
        & COMPUTE_STAGE_INPUT_HEADER0_COUNT_MASK;
    let layout_count = (header0 >> COMPUTE_STAGE_INPUT_HEADER0_LAYOUT_COUNT_SHIFT)
        & COMPUTE_STAGE_INPUT_HEADER0_COUNT_MASK;
    let index_type = (header0 >> COMPUTE_STAGE_INPUT_HEADER0_INDEX_TYPE_SHIFT)
        & COMPUTE_STAGE_INPUT_HEADER0_INDEX_TYPE_MASK;
    let index_buffer_index = (header0 >> COMPUTE_STAGE_INPUT_HEADER0_INDEX_BUFFER_SHIFT)
        & COMPUTE_STAGE_INPUT_HEADER0_INDEX_BUFFER_MASK;
    let layout_rel = header1 & COMPUTE_STAGE_INPUT_HEADER1_LAYOUT_OFFSET_MASK;
    let attr_rel = header1 >> COMPUTE_STAGE_INPUT_HEADER1_ATTR_OFFSET_SHIFT;

    let offset_base = (bo + COMPUTE_STAGE_INPUT_HEADER1_OFFSET_BASE) as u64;
    let min_entries = (bo + COMPUTE_STAGE_INPUT_MIN_LEN) as u64;
    let layout_section = offset_base.saturating_add(layout_rel as u64);
    let attr_section = offset_base.saturating_add(attr_rel as u64);
    let layout_bytes = (layout_count as u64) * (COMPUTE_STAGE_INPUT_LAYOUT_ENTRY_SIZE as u64);
    let attr_bytes = (attr_count as u64) * (COMPUTE_STAGE_INPUT_ATTR_ENTRY_SIZE as u64);
    if (layout_count != 0 && layout_section < min_entries)
        || (attr_count != 0 && attr_section < min_entries)
        || !section_range_fits(
            bytes.len(),
            layout_section,
            layout_count,
            COMPUTE_STAGE_INPUT_LAYOUT_ENTRY_SIZE,
        )
        || !section_range_fits(
            bytes.len(),
            attr_section,
            attr_count,
            COMPUTE_STAGE_INPUT_ATTR_ENTRY_SIZE,
        )
        || section_ranges_overlap(layout_section, layout_bytes, attr_section, attr_bytes)
    {
        return Err(DecodeStatus::ErrShort("res_stage_input_section_oob"));
    }

    let mut out = ComputeStageInputDescriptor {
        word0,
        header0,
        header1,
        index_type,
        index_buffer_index,
        attributes: Vec::new(),
        layouts: Vec::new(),
        dropped_attributes: 0,
        dropped_layouts: 0,
    };

    for i in 0..layout_count {
        let entry = layout_section + (i as u64) * (COMPUTE_STAGE_INPUT_LAYOUT_ENTRY_SIZE as u64);
        let entry = entry as usize;
        let raw_bits = ld32(&bytes[entry..]);
        note_unread_bits(
            "compute_stage_input_layout",
            raw_bits,
            COMPUTE_STAGE_INPUT_LAYOUT_BITS_READ,
        );
        if out.layouts.len() < MAX_COMPUTE_STAGE_INPUT_LAYOUTS {
            out.layouts.push(ComputeStageInputLayout {
                raw_bits,
                buffer_index: raw_bits & COMPUTE_STAGE_INPUT_LAYOUT_BITS_BUFFER_MASK,
                step_function: (raw_bits >> COMPUTE_STAGE_INPUT_LAYOUT_BITS_STEP_SHIFT)
                    & COMPUTE_STAGE_INPUT_LAYOUT_BITS_STEP_MASK,
                step_rate: ld32(&bytes[entry + COMPUTE_STAGE_INPUT_LAYOUT_STEP_RATE..]),
                stride: ld64(&bytes[entry + COMPUTE_STAGE_INPUT_LAYOUT_STRIDE..]),
            });
        } else {
            out.dropped_layouts += 1;
        }
    }
    for i in 0..attr_count {
        let entry = attr_section + (i as u64) * (COMPUTE_STAGE_INPUT_ATTR_ENTRY_SIZE as u64);
        let entry = entry as usize;
        let raw_bits = ld32(&bytes[entry..]);
        note_unread_bits(
            "compute_stage_input_attr",
            raw_bits,
            COMPUTE_STAGE_INPUT_ATTR_BITS_READ,
        );
        if out.attributes.len() < MAX_COMPUTE_STAGE_INPUT_ATTRS {
            out.attributes.push(ComputeStageInputAttribute {
                raw_bits,
                location: raw_bits & COMPUTE_STAGE_INPUT_ATTR_BITS_LOCATION_MASK,
                buffer_index: (raw_bits >> COMPUTE_STAGE_INPUT_ATTR_BITS_BUFFER_SHIFT)
                    & COMPUTE_STAGE_INPUT_ATTR_BITS_BUFFER_MASK,
                format: (raw_bits >> COMPUTE_STAGE_INPUT_ATTR_BITS_FORMAT_SHIFT)
                    & COMPUTE_STAGE_INPUT_ATTR_BITS_FORMAT_MASK,
                offset: ld32(&bytes[entry + COMPUTE_STAGE_INPUT_ATTR_OFFSET..]),
            });
        } else {
            out.dropped_attributes += 1;
        }
    }
    // The denominator for `packed_word_unread_bits` over the two entry words
    // above. Without it, a boot where no compute pipeline carries a stage-input
    // block and a boot where every entry's bits are all named read identically
    // at zero — and only the second is a measurement. This is the split
    // `note_color_entry_fields` states for the tag form; the bit form needs it
    // more, because a stage-input block is optional and this walk answers `None`
    // from six earlier returns.
    //
    // On the first driven x86 boot that carried both halves it read **absent**,
    // against two decoded compute pipelines — so this guest's compute pipelines
    // have no stage-input block and the walk above never runs. The split earned
    // its keep immediately: without it the same boot would have been quoted as
    // "no unread bits in a stage-input entry".
    if crate::observe::first_sight(
        "compute_stage_input_decoded",
        (u64::from(layout_count) << 32) | u64::from(attr_count),
    ) {
        crate::observe::off(format!(
            "compute_stage_input_decoded layouts={layout_count} attrs={attr_count} \
             index_type={index_type} index_buffer={index_buffer_index} \
             (the denominator for packed_word_unread_bits: this many entry words \
              were read and had every set bit named)"
        ));
    }
    Ok(Some(out))
}

/// Decode type-7 compute pipeline (`objType=0x0b`): kernel TLV + optional stage-input.
pub fn decode_compute_pipeline_descriptor(
    bytes: &[u8],
) -> Result<ComputePipelineDescriptor, DecodeStatus> {
    if bytes.len() < TYPE7_MIN_LEN {
        return Err(DecodeStatus::ErrShort("res_compute_pipeline_short"));
    }
    if ld32(&bytes[0..]) != TYPE7_OBJECT_COMPUTE_PIPELINE {
        return Err(DecodeStatus::ErrUnsupported("res_compute_pipeline_tag"));
    }
    let declared = ld32(&bytes[4..]) as usize;
    if declared != bytes.len() || declared < TYPE7_MIN_LEN {
        return Err(DecodeStatus::ErrShort("res_compute_pipeline_declared_len"));
    }
    // The same four-word header its render sibling carries, so the same relation
    // between its two lengths holds. Checked rather than stored: this descriptor
    // has no consumer for the value, and the walks below bound themselves on the
    // declared length.
    note_type7_payload_len("compute", ld32(&bytes[12..]), declared);
    let (fields, consumed) = decode_compact_tlv_record(bytes, TYPE7_FIRST_TLVS)?;
    note_pipeline_tlv_fields(
        "compute",
        &COMPUTE_PIPELINE_TAGS_CONSUMED,
        &COMPUTE_PIPELINE_TAGS_BENIGN,
        &fields,
    )?;
    let first_tlv_end = TYPE7_FIRST_TLVS + consumed;
    let stage_input = parse_compute_stage_input_block(bytes, first_tlv_end)?;
    Ok(ComputePipelineDescriptor {
        kernel_func_ref: compact_tlv_u32(&fields, PIPELINE_TAG_KERNEL_FUNC).unwrap_or(0),
        stage_input,
    })
}

/// Decode the 52-byte ICB command layout (create body `+0x1c` or live object).
pub fn decode_icb_command_layout(bytes: &[u8]) -> Result<IcbCommandLayout, DecodeStatus> {
    if bytes.len() < ICB_LAYOUT_LEN {
        return Err(DecodeStatus::ErrShort("res_icb_layout_short"));
    }
    Ok(IcbCommandLayout {
        command_type_offset: ld16(&bytes[0..]),
        barrier_offset: ld16(&bytes[2..]),
        kernel_dispatch_arguments_offset: ld16(&bytes[4..]),
        tessellation_factor_offset: ld16(&bytes[6..]),
        pipeline_state_offset: ld32(&bytes[8..]),
        vertex_buffer_bind_offset: ld32(&bytes[0xc..]),
        fragment_buffer_bind_offset: ld32(&bytes[0x10..]),
        object_buffer_bind_offset: ld32(&bytes[0x14..]),
        mesh_buffer_bind_offset: ld32(&bytes[0x18..]),
        kernel_buffer_bind_offset: ld32(&bytes[0x1c..]),
        attribute_stride_offset: ld32(&bytes[0x20..]),
        object_threadgroup_memory_length_offset: ld32(&bytes[0x24..]),
        threadgroup_memory_length_offset: ld32(&bytes[0x28..]),
        command_arguments_offset: ld32(&bytes[0x2c..]),
        command_size: ld32(&bytes[0x30..]),
    })
}

/// Encode layout into 52 bytes (tests / fixtures).
#[cfg(test)]
pub fn encode_icb_command_layout(layout: &IcbCommandLayout) -> [u8; ICB_LAYOUT_LEN] {
    let mut b = [0u8; ICB_LAYOUT_LEN];
    st16(&mut b[0..], layout.command_type_offset);
    st16(&mut b[2..], layout.barrier_offset);
    st16(&mut b[4..], layout.kernel_dispatch_arguments_offset);
    st16(&mut b[6..], layout.tessellation_factor_offset);
    st32(&mut b[8..], layout.pipeline_state_offset);
    st32(&mut b[0xc..], layout.vertex_buffer_bind_offset);
    st32(&mut b[0x10..], layout.fragment_buffer_bind_offset);
    st32(&mut b[0x14..], layout.object_buffer_bind_offset);
    st32(&mut b[0x18..], layout.mesh_buffer_bind_offset);
    st32(&mut b[0x1c..], layout.kernel_buffer_bind_offset);
    st32(&mut b[0x20..], layout.attribute_stride_offset);
    st32(
        &mut b[0x24..],
        layout.object_threadgroup_memory_length_offset,
    );
    st32(&mut b[0x28..], layout.threadgroup_memory_length_offset);
    st32(&mut b[0x2c..], layout.command_arguments_offset);
    st32(&mut b[0x30..], layout.command_size);
    b
}

/// Render-only layout for Draw / DrawIndexed / patches / mesh, no inherit.
///
/// `setupCommandLayout:` (pipeline `0x60`): bind tables in order
/// vertex → fragment → **object → mesh** → kernel, each `count × 0x14`, then
/// attribute-stride table (`maxVertex × 8` when dynamic stride), object-TG
/// lengths, kernel-TG lengths, then args.
#[cfg(test)]
pub fn render_icb_layout(
    max_vertex: u16,
    max_fragment: u16,
    command_types: u32,
) -> IcbCommandLayout {
    render_icb_layout_ex(max_vertex, max_fragment, 0, 0, 0, command_types)
}

/// Like [`render_icb_layout`] with object/mesh bind tables and object-TG lengths.
#[cfg(test)]
pub fn render_icb_layout_ex(
    max_vertex: u16,
    max_fragment: u16,
    max_object: u16,
    max_mesh: u16,
    max_object_tg: u16,
    command_types: u32,
) -> IcbCommandLayout {
    let pipeline = 0x60u32;
    let vertex_bind = 0x64u32;
    let after_vertex = vertex_bind + (max_vertex as u32) * (ICB_BUFFER_BIND_STRIDE as u32);
    let fragment_bind = after_vertex;
    let after_fragment = fragment_bind + (max_fragment as u32) * (ICB_BUFFER_BIND_STRIDE as u32);
    let object_bind = after_fragment;
    let after_object = object_bind + (max_object as u32) * (ICB_BUFFER_BIND_STRIDE as u32);
    let mesh_bind = after_object;
    let after_mesh = mesh_bind + (max_mesh as u32) * (ICB_BUFFER_BIND_STRIDE as u32);
    // No kernel binds on pure render ICBs (maxKernel=0).
    let free_after_binds = after_mesh;
    // RE setupCommandLayout: attribute-stride table is maxVertex × 8 after binds
    // when supportDynamicAttributeStride (product always reserves for max_vertex).
    let stride_off = free_after_binds;
    let after_stride = stride_off + (max_vertex as u32) * (ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE as u32);
    // Object TG length table then kernel TG (kernel 0 for pure render).
    let object_tg_off = after_stride;
    let after_object_tg = object_tg_off + (max_object_tg as u32) * (ICB_TG_MEMORY_STRIDE as u32);
    let kernel_tg_off = after_object_tg;
    let after_tg = after_object_tg;
    // setupCommandLayout (host RE): max of per-type argument region sizes.
    // Draw=0x24, DrawIndexed/DrawPatches=0x38, DrawIndexedPatches fill=0x4a
    // (layout alloc may use 0x4c), Mesh=0x48, ConcurrentDispatch=0x30.
    let mut args_size = 0u32;
    if command_types & MTL_INDIRECT_CMD_DRAW != 0 {
        args_size = 0x24;
    }
    if command_types & MTL_INDIRECT_CMD_DRAW_INDEXED != 0 {
        args_size = args_size.max(0x38);
    }
    if command_types & MTL_INDIRECT_CMD_DRAW_PATCHES != 0 {
        args_size = args_size.max(ICB_DRAW_PATCHES_ARGS_LEN);
    }
    if command_types & MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES != 0 {
        args_size = args_size.max(ICB_DRAW_INDEXED_PATCHES_ARGS_LEN);
    }
    if command_types
        & (MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS | MTL_INDIRECT_CMD_DRAW_MESH_THREADS)
        != 0
    {
        args_size = args_size.max(ICB_DRAW_MESH_ARGS_LEN);
    }
    if args_size == 0 {
        // Default Draw if no bits (tests); keep deterministic.
        args_size = 0x24;
    }
    IcbCommandLayout {
        command_type_offset: 0,
        barrier_offset: 4,
        kernel_dispatch_arguments_offset: 8,
        tessellation_factor_offset: 0x40,
        pipeline_state_offset: pipeline,
        vertex_buffer_bind_offset: vertex_bind,
        fragment_buffer_bind_offset: fragment_bind,
        object_buffer_bind_offset: object_bind,
        mesh_buffer_bind_offset: mesh_bind,
        kernel_buffer_bind_offset: free_after_binds,
        attribute_stride_offset: stride_off,
        object_threadgroup_memory_length_offset: object_tg_off,
        threadgroup_memory_length_offset: kernel_tg_off,
        command_arguments_offset: after_tg,
        command_size: after_tg + args_size,
    }
}

/// Draw-only convenience (commandTypes Draw).
#[cfg(test)]
pub fn render_only_icb_layout(max_vertex: u16) -> IcbCommandLayout {
    render_icb_layout(max_vertex, 0, MTL_INDIRECT_CMD_DRAW)
}

/// DrawIndexed-only convenience.
#[cfg(test)]
pub fn render_draw_indexed_icb_layout(max_vertex: u16) -> IcbCommandLayout {
    render_icb_layout(max_vertex, 0, MTL_INDIRECT_CMD_DRAW_INDEXED)
}

/// DrawPatches-only convenience (args 0x38 + tessellation factor table at 0x40).
#[cfg(test)]
pub fn render_draw_patches_icb_layout(max_vertex: u16) -> IcbCommandLayout {
    render_icb_layout(max_vertex, 0, MTL_INDIRECT_CMD_DRAW_PATCHES)
}

/// DrawIndexedPatches-only convenience (args 0x4a).
#[cfg(test)]
pub fn render_draw_indexed_patches_icb_layout(max_vertex: u16) -> IcbCommandLayout {
    render_icb_layout(max_vertex, 0, MTL_INDIRECT_CMD_DRAW_INDEXED_PATCHES)
}

/// DrawMeshThreads-only convenience (args 0x48, optional mesh bind slots).
#[cfg(test)]
pub fn render_draw_mesh_threads_icb_layout(max_mesh: u16) -> IcbCommandLayout {
    render_icb_layout_ex(0, 0, 0, max_mesh, 0, MTL_INDIRECT_CMD_DRAW_MESH_THREADS)
}

/// DrawMeshThreadgroups-only convenience (args 0x48).
#[cfg(test)]
pub fn render_draw_mesh_threadgroups_icb_layout() -> IcbCommandLayout {
    render_icb_layout(0, 0, MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS)
}

/// Mesh draw with object + mesh bind tables.
#[cfg(test)]
pub fn render_draw_mesh_threads_icb_layout_with_binds(
    max_object: u16,
    max_mesh: u16,
) -> IcbCommandLayout {
    render_icb_layout_ex(
        0,
        0,
        max_object,
        max_mesh,
        0,
        MTL_INDIRECT_CMD_DRAW_MESH_THREADS,
    )
}

/// Object+mesh drawMeshThreadgroups with optional object TG memory slots.
#[cfg(test)]
pub fn render_draw_mesh_threadgroups_icb_layout_ex(
    max_object: u16,
    max_mesh: u16,
    max_object_tg: u16,
) -> IcbCommandLayout {
    render_icb_layout_ex(
        0,
        0,
        max_object,
        max_mesh,
        max_object_tg,
        MTL_INDIRECT_CMD_DRAW_MESH_THREADGROUPS,
    )
}

/// Compute-only layout for ConcurrentDispatch with `max_kernel` binds, no inherit.
///
/// Matches `AppleParavirtIndirectCommandBuffer setupCommandLayout:` for the
/// common product case (commandTypes=`1<<5`, inheritBuffers=false,
/// inheritPipelineState=false). Threadgroup-memory table size 0 (no TG binds).
#[cfg(test)]
pub fn compute_only_icb_layout(max_kernel: u16) -> IcbCommandLayout {
    compute_icb_layout(max_kernel, 0)
}

/// Compute ICB layout with optional TG-memory table and attribute-stride table.
///
/// `setupCommandLayout:` order after kernel binds:
/// 1. `max_kernel × 8` attribute-stride u64s at `attributeStrideOffset`
/// 2. `max_kernel_tg × 8` TG-memory length u64s at `threadgroupMemoryLengthOffset`
/// 3. dispatch args. Barrier is u32 at `barrierOffset` (typically 4).
#[cfg(test)]
pub fn compute_icb_layout(max_kernel: u16, max_kernel_tg: u16) -> IcbCommandLayout {
    let pipeline = 0x60u32;
    let kernel_bind = 0x64u32;
    let free_after_binds = kernel_bind + (max_kernel as u32) * (ICB_BUFFER_BIND_STRIDE as u32);
    let stride_off = free_after_binds;
    let after_stride = stride_off + (max_kernel as u32) * (ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE as u32);
    let tg_off = after_stride;
    let after_tg = tg_off + (max_kernel_tg as u32) * (ICB_TG_MEMORY_STRIDE as u32);
    // ConcurrentDispatch-only args size = 0x30 (3×u64 grid + 3×u64 tptg).
    let args_size = 0x30u32;
    IcbCommandLayout {
        command_type_offset: 0,
        barrier_offset: 4,
        kernel_dispatch_arguments_offset: 8,
        tessellation_factor_offset: 0x40,
        pipeline_state_offset: pipeline,
        vertex_buffer_bind_offset: kernel_bind,
        fragment_buffer_bind_offset: kernel_bind,
        object_buffer_bind_offset: kernel_bind,
        mesh_buffer_bind_offset: kernel_bind,
        kernel_buffer_bind_offset: kernel_bind,
        attribute_stride_offset: stride_off,
        object_threadgroup_memory_length_offset: after_tg,
        threadgroup_memory_length_offset: tg_off,
        command_arguments_offset: after_tg,
        command_size: after_tg + args_size,
    }
}

/// How many `stride`-sized entries the layout region `[start, end)` holds.
///
/// The four ICB tables — vertex/fragment/object/mesh/kernel binds, object-TG
/// lengths, kernel-TG lengths, attribute strides — each pick their own two
/// endpoints out of [`IcbCommandLayout`], and then all four ask this same
/// question of them. It used to be written out at each of the four, which is
/// four chances for one of them to answer differently.
///
/// **The answer is `u32` because the endpoints are.** The layout's offsets are
/// the 32-bit words of the guest's own layout blob, so a region can name more
/// than `u16::MAX` entries and the count has to be able to say so. All four
/// used to narrow the quotient to `u16` on the way out, which wraps rather than
/// saturates: a table of 65 537 entries answered 1, and the caller then read
/// one entry and dropped the rest, or — at
/// [`crate::runtime::icb`]'s attribute-stride readers, which bounds-check the
/// guest's index against this count — refused a perfectly good index for being
/// past a table it had just been told was one entry long.
///
/// A well-formed descriptor cannot reach that: the create body declares each
/// bind count in a single byte (see `reims_vgpu_wire::ops::icb::NewIcbBody`),
/// so a guest whose layout agrees with its own create body stays under 256. The
/// layout blob is a separate copy from the create body and nothing here checks
/// the two against each other, so the width is not left resting on that
/// agreement.
pub(crate) fn icb_layout_table_len(start: u32, end: u32, stride: usize) -> u32 {
    if end <= start {
        return 0;
    }
    (end - start) / stride as u32
}

/// Number of kernel-threadgroup-memory length slots implied by layout offsets.
pub fn icb_layout_kernel_tg_slot_count(layout: &IcbCommandLayout) -> u32 {
    icb_layout_table_len(
        layout.threadgroup_memory_length_offset,
        layout.command_arguments_offset,
        ICB_TG_MEMORY_STRIDE,
    )
}

/// Number of attribute-stride table entries implied by layout offsets.
///
/// Table is `[attribute_stride_offset, next_region)` in u64 slots, where
/// `next_region` is the earliest of object-TG / kernel-TG / command-args that
/// lies strictly after the stride table start.
pub fn icb_layout_attribute_stride_slot_count(layout: &IcbCommandLayout) -> u32 {
    let start = layout.attribute_stride_offset;
    if start == 0 {
        return 0;
    }
    let end = [
        layout.object_threadgroup_memory_length_offset,
        layout.threadgroup_memory_length_offset,
        layout.command_arguments_offset,
    ]
    .into_iter()
    .filter(|&e| e > start)
    .min()
    .unwrap_or(start);
    icb_layout_table_len(start, end, ICB_ATTRIBUTE_STRIDE_ENTRY_SIZE)
}

/// Decode type-7 ICB create descriptor (tag 0x36, length 0x58).
///
/// Field map from PGSerializer
/// `newIndirectCommandBufferWithDescriptor:layout:maxCommandCount:options:allocator:`
/// local emission + layout memcpy at `+0x1c` (2026-07-11 arm64 host RE).
pub fn decode_icb_descriptor(
    bytes: &[u8],
) -> Result<IndirectCommandBufferDescriptor, DecodeStatus> {
    let op =
        reims_vgpu_wire::op(bytes, 0).map_err(|_| DecodeStatus::ErrShort("res_icb_desc_short"))?;
    if op.opcode() != TYPE7_OBJECT_ICB
        || op.length() as usize != ICB_DESC_LEN
        || bytes.len() != ICB_DESC_LEN
    {
        return Err(DecodeStatus::ErrUnsupported("res_icb_desc_tag"));
    }
    let body = w_icb::new_indirect_command_buffer(&op)
        .map_err(|_| DecodeStatus::ErrShort("res_icb_desc_short"))?;
    // Bit 15 is never written by the serializer; see [`ICB_FLAG_NEVER_WRITTEN`].
    let flags = body.flags.get() & !ICB_FLAG_NEVER_WRITTEN;
    // Layout remains a nested decode of the embedded layout block (same bytes).
    let layout_at = OP_HDR + offset_of!(w_icb::NewIcbBody, layout);
    let layout = decode_icb_command_layout(&bytes[layout_at..layout_at + ICB_LAYOUT_LEN])?;
    Ok(IndirectCommandBufferDescriptor {
        command_types: body.command_types.get(),
        max_vertex_buffer_bind_count: body.max_vertex_buffer_bind_count as u16,
        max_fragment_buffer_bind_count: body.max_fragment_buffer_bind_count as u16,
        max_kernel_buffer_bind_count: body.max_kernel_buffer_bind_count as u16,
        max_object_buffer_bind_count: body.max_object_buffer_bind_count as u16,
        max_mesh_buffer_bind_count: body.max_mesh_buffer_bind_count as u16,
        max_kernel_threadgroup_memory_bind_count: body.max_kernel_threadgroup_memory_bind_count
            as u16,
        max_object_threadgroup_memory_bind_count: body.max_object_threadgroup_memory_bind_count
            as u16,
        flags,
        max_command_count: body.max_command_count.get(),
        options: body.options.get(),
        layout,
    })
}

/// Decode type-7 container (sampler / depth-stencil / pipelines / ICB).
pub fn decode_type7_descriptor(bytes: &[u8]) -> Result<Descriptor, DecodeStatus> {
    if bytes.len() < 4 {
        return Err(DecodeStatus::ErrShort("res_type7_short"));
    }
    let first = ld32(&bytes[0..]);
    match first {
        TYPE7_OBJECT_SAMPLER => Ok(Descriptor::Sampler(decode_sampler_descriptor(bytes)?)),
        TYPE7_OBJECT_DEPTH_STENCIL => Ok(Descriptor::DepthStencil(
            decode_depth_stencil_descriptor(bytes)?,
        )),
        TYPE7_OBJECT_RENDER_PIPELINE => Ok(Descriptor::RenderPipeline(
            decode_render_pipeline_descriptor(bytes)?,
        )),
        TYPE7_OBJECT_COMPUTE_PIPELINE => Ok(Descriptor::ComputePipeline(
            decode_compute_pipeline_descriptor(bytes)?,
        )),
        TYPE7_OBJECT_ICB => Ok(Descriptor::IndirectCommandBuffer(decode_icb_descriptor(
            bytes,
        )?)),
        _ => Err(DecodeStatus::ErrUnsupported("res_type7_subtype_unknown")),
    }
}

pub fn decode_descriptor(object_type: u8, bytes: &[u8]) -> Result<Descriptor, DecodeStatus> {
    match object_type {
        OBJECT_TYPE_BUFFER => Ok(Descriptor::Buffer(decode_buffer_descriptor(bytes)?)),
        OBJECT_TYPE_TEXTURE | OBJECT_TYPE_TEXTURE_VARIANT => {
            Ok(Descriptor::Texture(decode_texture_descriptor(bytes)?))
        }
        OBJECT_TYPE_FUNCTION => Ok(Descriptor::Function(decode_function_descriptor(bytes)?)),
        OBJECT_TYPE_TYPE7 => decode_type7_descriptor(bytes),
        OBJECT_TYPE_TEXTURE_VIEW => Ok(Descriptor::TextureView(decode_texture_view_descriptor(
            bytes,
        )?)),
        OBJECT_TYPE_IOSURFACE => decode_iosurface_texture_descriptor(bytes),
        _ => Err(DecodeStatus::ErrUnknownType("res_object_type_unknown")),
    }
}

#[cfg(test)]
mod tests;
