//! Opcode 1 — create texture.
//!
//! This is the worked example the rest of the operations follow. Read it, then
//! read `AGENTS.md` in this crate for the procedure to add the next one.
//!
//! # Layout
//!
//! Total 44 bytes: the 8-byte [`crate::op::OpHeader`] then a 36-byte payload.
//!
//! The 32 bytes from `+004` are [`TextureDescriptorBody`], which
//! [`crate::ops::heap_texture`] embeds verbatim.
//!
//! ```text
//! payload +000  u32  object_ref  the ref the guest's allocator assigned
//! payload +004  u32  packed      texture_type[3:0] framebuffer_only[4]
//!                                is_drawable[5] gpu_opt[6]
//!                                usage[15:8] pixel_format[31:16]; bit 7 unwritten
//! payload +008  u32  width
//! payload +012  u32  height
//! payload +016  u32  depth
//! payload +020  u16  mipmap_level_count
//! payload +022  u16  sample_count
//! payload +024  u16  array_length
//! payload +026  u16  resource_options
//! payload +028  u64  protection_options
//! ```
//!
//! `object_ref` is a payload field, not a header one. Object-creation records
//! carry it first and encoder records do not carry it at all, so a header that
//! claimed it would eat the first four bytes of every render command's payload.
//! See [`crate::op`] for the derivation.
//!
//! # How the layout was derived
//!
//! Two independent sources agree on it, which is why it is stated rather than
//! proposed.
//!
//! The Objective-C type encoding of
//! `-[PGSerializer serializeTextureDescriptor:textureDescriptor:]` declares the
//! out-parameter as `^{?=b4b1b1b1b1b8b16IIISSSSQ}` — a 4-bit field, four 1-bit
//! fields, an 8-bit field, a 16-bit field, three `u32`, four `u16`, one `u64`.
//! That fixes the widths and their order without any guessing.
//!
//! The values were then pinned by perturbation: build a baseline descriptor,
//! change exactly one property, serialize, diff the bytes. Whatever moved is
//! that property's home. The oracle drives this; every claim below cites the
//! observations behind it, and anything not perturbed is marked unidentified
//! rather than named.
//!
//! # `packed` bit 7 is not a field
//!
//! The oracle captures every case twice under complementary arena fills and
//! reports the bits that agreed, so "the serializer wrote here" is measured
//! rather than inferred from a byte that happened to read `0xAA`. Bit 7 of
//! `packed` disagrees between the fills in **every** texture fixture: the
//! serializer never writes it, and on a real wire it holds whatever the guest's
//! ring last contained.
//!
//! This is not an academic distinction. Before that measurement this module
//! named `packed[7:4]` as one four-bit unidentified group and documented it as
//! reading `0b1100` on every descriptor — a value whose top bit was the fill,
//! not the contract. Anything reading that accessor got one bit of noise.
//!
//! # What the serializer does not carry
//!
//! `MTLTextureDescriptor.compressionType` reaches the wire nowhere.
//! `texture_compression_lossy` differs from `texture_baseline` in the object
//! ref and in nothing else, across all 44 bytes. So a guest that asks for a
//! lossy-compressed texture emits a record that does not say so, and no decoder
//! can recover the request.

use crate::le::{U16le, U32le, U64le};
use crate::op::Op;
use crate::view::{view, Wire, WireError};

/// Opcode for texture creation, observed on
/// `-[PGSerializer newTextureWithDescriptor:allocator:]`.
pub const OPCODE_NEW_TEXTURE: u32 = 1;

/// Total wire length of a texture-creation operation, header included.
pub const NEW_TEXTURE_TOTAL_LEN: u32 = 44;

/// Bytes of [`TextureDescriptorBody`], the half two records share.
pub const TEXTURE_DESCRIPTOR_LEN: usize = 32;

/// `MTLStorageMode`, `resource_options[7:4]`, from the raw options word.
///
/// The narrow and wide descriptor bodies both carry that word, and a semantic
/// consumer holding only the word needs the same answer they do — so the shift
/// is written once here and the two accessors call it. See
/// [`TextureDescriptorBody::storage_mode`] for what the field means and for the
/// misreading it invites.
#[inline]
#[must_use]
pub const fn storage_mode_nibble(resource_options: u16) -> u8 {
    ((resource_options >> 4) & 0xf) as u8
}

/// The 32-byte texture descriptor, without the new object's ref.
///
/// Declared apart from [`NewTextureBody`] because it is not only that record's
/// tail: [`crate::ops::heap_texture`] embeds the same 32 bytes after a heap ref,
/// and `reims-vgpu` reaches a third copy through its heap size-and-align query.
/// One declaration is what stops those drifting — the fixtures for both records
/// run through the accessors below, so a layout change fails twice rather than
/// leaving one reader right and the other wrong.
#[repr(C)]
#[derive(Debug)]
pub struct TextureDescriptorBody {
    /// Texture type, framebuffer/drawable flags, GPU-optimized contents, usage
    /// and pixel format, packed. Prefer the accessors below over reading this
    /// directly; in particular bit 7 is **never written by the serializer**.
    pub packed: U32le,
    pub width: U32le,
    pub height: U32le,
    pub depth: U32le,
    pub mipmap_level_count: U16le,
    pub sample_count: U16le,
    pub array_length: U16le,
    /// `MTLResourceOptions`. See [`TextureDescriptorBody::storage_mode`],
    /// [`TextureDescriptorBody::cpu_cache_mode`] and
    /// [`TextureDescriptorBody::hazard_tracking_mode`].
    pub resource_options: U16le,
    /// Private texture-descriptor `protectionOptions` value.
    ///
    /// An independent `setProtectionOptions:` perturbation moved exactly these
    /// eight bytes in both narrow and wide records. Resource index, rotation,
    /// and sparse-surface-default perturbations left them unchanged.
    pub protection_options: U64le,
}

// SAFETY: every field is an align-1 all-bytes-valid `le` scalar, so the struct
// is align-1 and all 32-byte patterns are valid.
unsafe impl Wire for TextureDescriptorBody {}

/// Payload of a texture-creation record: the new object's ref, then the
/// descriptor.
#[repr(C)]
#[derive(Debug)]
pub struct NewTextureBody {
    /// Ref the guest's object-ref allocator assigned to the new texture.
    /// Subsequent records name the texture by this value.
    pub object_ref: U32le,
    /// The descriptor Metal was handed.
    pub desc: TextureDescriptorBody,
}

// SAFETY: a `U32le` and a `Wire` struct, both align-1 and all-bytes-valid, so
// the whole 36 bytes is too.
unsafe impl Wire for NewTextureBody {}

impl TextureDescriptorBody {
    /// `MTLTextureType` ordinal, `packed[3:0]`.
    ///
    /// Observed: 2D→2, 2DArray→3, 2DMultisample→4, Cube→5, 3D→7 — the
    /// `MTLTextureType` ordinals unchanged. 1D→0, 1DArray→1 and CubeArray→6
    /// follow from that enum but have not been observed here.
    #[inline]
    pub fn texture_type(&self) -> u8 {
        (self.packed.get() & 0xf) as u8
    }

    /// `packed[6]` — whether the guest allowed GPU-optimized contents.
    ///
    /// Observed: the baseline descriptor leaves
    /// `MTLTextureDescriptor.allowGPUOptimizedContents` at its default of `YES`
    /// and `packed` reads `0x005005c2`; setting it to `NO` and changing nothing
    /// else reads `0x00500582`. One bit moved, and it is this one.
    #[inline]
    pub fn allow_gpu_optimized_contents(&self) -> bool {
        self.packed.get() & (1 << 6) != 0
    }

    /// `packed[4]` — the descriptor's private `framebufferOnly` property.
    ///
    /// Setting only that property moved the packed byte from `0xc2` to `0xd2`.
    #[inline]
    pub fn framebuffer_only(&self) -> bool {
        self.packed.get() & (1 << 4) != 0
    }

    /// `packed[5]` — the descriptor's private `isDrawable` property.
    ///
    /// Setting only that property moved the packed byte from `0xc2` to `0xe2`.
    #[inline]
    pub fn is_drawable(&self) -> bool {
        self.packed.get() & (1 << 5) != 0
    }

    /// `MTLTextureUsage` mask, `packed[15:8]`.
    ///
    /// Observed: ShaderRead→1, ShaderWrite→2, RenderTarget→4, and
    /// ShaderRead|RenderTarget→5, so it is the Metal mask carried straight
    /// through rather than a re-encoding.
    #[inline]
    pub fn usage(&self) -> u8 {
        ((self.packed.get() >> 8) & 0xff) as u8
    }

    /// `MTLPixelFormat` ordinal, `packed[31:16]`.
    ///
    /// Observed: BGRA8Unorm→80, RGBA8Unorm→70, R8Unorm→10 — the `MTLPixelFormat`
    /// ordinals unchanged.
    ///
    /// Note this is a *16-bit ordinal*, which is not the only format encoding in
    /// this protocol: `reims-vgpu`'s device-descriptor path carries either an
    /// MTL ordinal or an OSType FourCC, separated by width. Nothing here is a
    /// FourCC — the field is too narrow to hold one.
    #[inline]
    pub fn pixel_format(&self) -> u16 {
        (self.packed.get() >> 16) as u16
    }

    /// `MTLStorageMode`, `resource_options[7:4]`.
    ///
    /// Observed: Shared→`0x0000`, Managed→`0x0010`, Private→`0x0020`, i.e. the
    /// mode ordinal shifted left by 4. That matches `MTLResourceOptions`'
    /// documented storage-mode shift, so the field is a `MTLResourceOptions`
    /// word rather than a bare mode.
    ///
    /// # This is an announcement contract, not an access contract
    ///
    /// It reads like a licence to skip coherence work for `Private` — the
    /// device would not have to keep guest pages current for a resource the
    /// guest has declared GPU-only. It is not one. Reading the emitting
    /// serializer says why:
    ///
    /// - Backing is **mode-blind**. A `Private` texture still gets page-rounded
    ///   guest backing allocated unconditionally, exactly as `Shared` does.
    /// - The guest still **CPU-touches** it. Create-with-contents,
    ///   region-replace and get-bytes each memcpy through that mapping with no
    ///   storage-mode check on the path.
    /// - What the mode actually gates is the **announcement**: the
    ///   modified-range notification is emitted only for `Managed`.
    ///
    /// So `Private` means "I will not tell you when I write this", not "I will
    /// not write this". Treating it as the latter converts silence into a
    /// guarantee of inaction, which is the one reading the field does not
    /// support, and the resulting stale-page bug would be invisible at every
    /// counter because its failure mode is content.
    ///
    /// The experiment that would settle it, if the question is reopened: read
    /// the *host* deserializer for a mode-dependent branch on the backing or
    /// the coherence path. Absence of one on the emitting side is what is
    /// established above; the receiving side has not been read.
    ///
    /// # The one consumer, and why it runs the other way
    ///
    /// This field is read exactly once outside this crate, by
    /// `reims_vgpu_protocol::StorageMode::from_nibble`, and every decision
    /// downstream of it *withholds* a claim rather than skipping work. The
    /// gather witness may call a resource's content quiet only where the guest
    /// is obliged to announce a write to it, which is `Managed` alone; the
    /// silent modes lose the memoization and re-read their bytes. That is the
    /// safe direction, and it is the exact inverse of the misreading this doc
    /// warns about above — a bug in it costs throughput, not content. Adding a
    /// consumer that reads the mode to *avoid* work reopens the hazard.
    #[inline]
    pub fn storage_mode(&self) -> u8 {
        storage_mode_nibble(self.resource_options.get())
    }

    /// `MTLCPUCacheMode`, `resource_options[3:0]`.
    ///
    /// Observed: `DefaultCache`→`0x0020` (with `StorageModePrivate`),
    /// `WriteCombined` with `StorageModeShared`→`0x0001`. The storage nibble
    /// went to 0 with the mode change, and the cache mode landed at `[3:0]`
    /// carrying its ordinal, which is `MTLResourceOptions`' documented layout.
    ///
    /// The two moved together because Metal will not keep a write-combined
    /// cache mode on a private texture, so the case perturbs both; the
    /// attribution is still clean, since the storage nibble's own three values
    /// are pinned separately by the three storage cases.
    #[inline]
    pub fn cpu_cache_mode(&self) -> u8 {
        (self.resource_options.get() & 0xf) as u8
    }

    /// `MTLHazardTrackingMode`, `resource_options[9:8]`.
    ///
    /// Observed: Default→`0x0020`, Untracked→`0x0120`, Tracked→`0x0220`, i.e.
    /// the mode ordinal shifted left by 8 with the storage nibble untouched.
    #[inline]
    pub fn hazard_tracking_mode(&self) -> u8 {
        ((self.resource_options.get() >> 8) & 0x3) as u8
    }

    /// The whole `MTLResourceOptions` word.
    ///
    /// Every field this crate names is an accessor above; the raw word is here
    /// for bits none of them claim, which is `[7:4]`'s high half and everything
    /// from bit 10 up. All of those read 0 in every fixture.
    #[inline]
    pub fn resource_options_raw(&self) -> u16 {
        self.resource_options.get()
    }
}

/// View the payload of a texture-creation record.
///
/// Refuses a record whose opcode is not [`OPCODE_NEW_TEXTURE`]; the caller
/// is expected to have dispatched on opcode already, and a mismatch here means
/// a dispatch bug rather than a malformed guest.
pub fn new_texture<'a>(op: &Op<'a>) -> Result<&'a NewTextureBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_NEW_TEXTURE);
    view::<NewTextureBody>(op.payload)
}

// --- 0x34 newTextureWithDescriptor:allocator:, wide form -------------------

/// The same selector's opcode with `-setSupportsSwizzledTextures:` on.
pub const OPCODE_NEW_TEXTURE_WIDE: u32 = 0x34;

/// Total wire length of the wide texture-creation operation, header included.
///
/// Fifty-two: eight header, four ref, forty body — of which the serializer
/// writes **thirty-nine**. See [`WideTextureDescriptorBody::declared_tail`].
pub const NEW_TEXTURE_WIDE_TOTAL_LEN: u32 = 52;

/// Bytes of [`WideTextureDescriptorBody`], the half five records share.
pub const WIDE_TEXTURE_DESCRIPTOR_LEN: usize = 40;

/// The 40-byte texture descriptor: the 32-byte one widened, plus a swizzle.
///
/// This is the wire form of `-serializeTextureDescriptor2:textureDescriptor:`'s
/// `b4b1b1b1b1b16IIIISSSSQCCCC`, against `b4b1b1b1b1b8b16IIISSSSQ` for
/// [`TextureDescriptorBody`]. Two changes, both visible in the encoding: `usage`
/// leaves the packed word for a `u32` of its own, and four `C` bytes trail —
/// the per-channel swizzle.
///
/// # Which flag selects it is not the one its name suggests
///
/// `newTextureWithDescriptor:allocator:` switches to [`OPCODE_NEW_TEXTURE_WIDE`]
/// under **`SwizzledTextures`**, and `TextureDescriptor2` leaves it alone —
/// `texture_baseline_descriptor2` is byte-identical to `texture_baseline`, which
/// is what `the_second_texture_descriptor_layout_does_not_reach_the_wire`
/// measures and all it measures. The other four records that carry this body
/// switch under `TextureDescriptor2` instead. So the family is split across two
/// capabilities, and a negative result about one selector said nothing about the
/// other four.
///
/// # It is a different opcode, not a longer record
///
/// The narrow form stays at [`OPCODE_NEW_TEXTURE`]. A reader that dispatches on
/// opcode is therefore safe by construction, and one that dispatches on selector
/// and assumes a length is not. That is why this is declared as its own view
/// rather than as an optional tail on the narrow one.
///
/// # Nothing here is aligned
///
/// `pixel_format` sits at `+1` and every `u32` after it at an odd offset. The
/// serializer packs the bitfields to three bytes and then writes the `I`s
/// immediately, with no padding to a four-byte boundary — so this struct is only
/// readable because every `le` type in this crate is align-1.
#[repr(C)]
#[derive(Debug)]
pub struct WideTextureDescriptorBody {
    /// Texture type, framebuffer/drawable flags, GPU-optimized contents and
    /// write-swizzle state. The narrow form packs `usage` and `pixel_format`
    /// into the same word; this one does not.
    ///
    /// **Bit 7 is written here and is not written in the narrow form.** The
    /// narrow `texture_baseline` comes back with a written mask of `0x7f` on
    /// this byte and this one with `0xff`, so the fourth `b1` flag exists in
    /// both encodings and only one of them assigns it. Reading it as a flag that
    /// is off is right here and wrong there.
    pub type_and_flags: u8,
    /// `MTLPixelFormat` ordinal. Observed BGRA8Unorm→80, RGBA8Unorm→70,
    /// R8Unorm→10, exactly as [`TextureDescriptorBody::pixel_format`].
    pub pixel_format: U16le,
    /// `MTLTextureUsage` mask, promoted from eight bits to thirty-two.
    ///
    /// Observed ShaderRead→1, ShaderWrite→2, RenderTarget→4 and
    /// ShaderRead|RenderTarget→5. The upper three bytes read zero on every
    /// fixture; the width is the encoding's fourth `I`, not an inference from
    /// them.
    pub usage: U32le,
    pub width: U32le,
    pub height: U32le,
    pub depth: U32le,
    pub mipmap_level_count: U16le,
    pub sample_count: U16le,
    pub array_length: U16le,
    /// `MTLResourceOptions`, the same word [`TextureDescriptorBody`] carries.
    pub resource_options: U16le,
    /// The same private `protectionOptions` value as
    /// [`TextureDescriptorBody::protection_options`].
    pub protection_options: U64le,
    /// `MTLTextureSwizzleChannels`, one `MTLTextureSwizzle` ordinal per channel
    /// in red, green, blue, alpha order.
    ///
    /// The baseline reads `02 03 04 05`, which is
    /// `MTLTextureSwizzleChannelsDefault` — and against one fixture that cannot
    /// be told from a constant the serializer writes. `texture_swizzled_permuted`
    /// sets (Alpha, Zero, One, Red) and reads `05 00 01 02`, which pins both the
    /// order and that the values are the guest's.
    pub swizzle_red: u8,
    pub swizzle_green: u8,
    pub swizzle_blue: u8,
    pub swizzle_alpha: u8,
    /// The fortieth byte, which the serializer never writes.
    ///
    /// The body's fields end at `+39`; the record still declares forty. So this
    /// holds whatever the guest's ring last contained, and a reader must not
    /// look at it. Measured: the written mask reads `0x00` here on both
    /// `texture_swizzled` fixtures while every other byte reads `0xff`.
    ///
    /// This is the third instance of "a record's declared length is not its
    /// written extent" in this protocol, after the rate map's sixteen bytes and
    /// the compute scope barrier's two. It is not a rare shape.
    pub declared_tail: u8,
}

// SAFETY: `u8`s and align-1 all-bytes-valid `le` scalars, so the struct is
// align-1 and all 40-byte patterns are valid.
unsafe impl Wire for WideTextureDescriptorBody {}

/// Payload of a wide texture-creation record: the ref, then the wide descriptor.
#[repr(C)]
#[derive(Debug)]
pub struct NewTextureWideBody {
    /// Ref the guest's object-ref allocator assigned to the new texture.
    pub object_ref: U32le,
    pub desc: WideTextureDescriptorBody,
}

// SAFETY: a `U32le` and a `Wire` struct, both align-1 and all-bytes-valid.
unsafe impl Wire for NewTextureWideBody {}

impl WideTextureDescriptorBody {
    /// `MTLTextureType` ordinal, `type_and_flags[3:0]`.
    ///
    /// Same ordinals as [`TextureDescriptorBody::texture_type`], measured over
    /// the same perturbation set: 2D→2, 2DArray→3, 2DMultisample→4, Cube→5,
    /// 3D→7.
    #[inline]
    pub fn texture_type(&self) -> u8 {
        self.type_and_flags & 0xf
    }

    /// `type_and_flags[6]` — whether the guest allowed GPU-optimized contents.
    ///
    /// The same bit as [`TextureDescriptorBody::allow_gpu_optimized_contents`],
    /// and moved by the same case: `texture_no_gpu_optimized_contents` takes
    /// this byte from `0x42` to `0x02`.
    #[inline]
    pub fn allow_gpu_optimized_contents(&self) -> bool {
        self.type_and_flags & (1 << 6) != 0
    }

    /// `type_and_flags[4]` — the descriptor's private `framebufferOnly` property.
    #[inline]
    pub fn framebuffer_only(&self) -> bool {
        self.type_and_flags & (1 << 4) != 0
    }

    /// `type_and_flags[5]` — the descriptor's private `isDrawable` property.
    #[inline]
    pub fn is_drawable(&self) -> bool {
        self.type_and_flags & (1 << 5) != 0
    }

    /// `type_and_flags[7]` — private `writeSwizzleEnabled`.
    ///
    /// This bit is encoded only by the wide serializer. The narrow serializer
    /// leaves the corresponding bit unwritten.
    #[inline]
    pub fn write_swizzle_enabled(&self) -> bool {
        self.type_and_flags & (1 << 7) != 0
    }

    /// `MTLStorageMode`, `resource_options[7:4]`.
    ///
    /// See [`TextureDescriptorBody::storage_mode`] for the derivation and for
    /// why this field is an announcement contract rather than an access one —
    /// it is not a licence to skip coherence work for `Private`.
    #[inline]
    pub fn storage_mode(&self) -> u8 {
        storage_mode_nibble(self.resource_options.get())
    }

    /// `MTLCPUCacheMode`, `resource_options[3:0]`.
    #[inline]
    pub fn cpu_cache_mode(&self) -> u8 {
        (self.resource_options.get() & 0xf) as u8
    }

    /// `MTLHazardTrackingMode`, `resource_options[9:8]`.
    #[inline]
    pub fn hazard_tracking_mode(&self) -> u8 {
        ((self.resource_options.get() >> 8) & 0x3) as u8
    }
}

/// View the payload of a wide texture-creation record.
pub fn new_texture_wide<'a>(op: &Op<'a>) -> Result<&'a NewTextureWideBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_NEW_TEXTURE_WIDE);
    view::<NewTextureWideBody>(op.payload)
}

// --- 0x16 heapTextureSizeAndAlignWithDescriptor:allocator: -----------------

pub const OPCODE_HEAP_TEXTURE_SIZE_AND_ALIGN: u32 = 0x16;
pub const HEAP_TEXTURE_SIZE_AND_ALIGN_TOTAL_LEN: u32 = 40;

/// The heap sizing query: [`OPCODE_NEW_TEXTURE`]'s record without the ref.
///
/// `-heapTextureSizeAndAlignWithDescriptor:allocator:` asks the host how large
/// a heap allocation this descriptor would need. It creates no object, so there
/// is no ref to lead with, and the payload is a bare [`TextureDescriptorBody`] —
/// **byte for byte** the same 32 bytes `0x01` writes after its ref, down to the
/// unwritten `packed` bit 7. Fixture `serializer_heap_texture_size_and_align`
/// against `texture_baseline`; the two buffers differ only in opcode, length,
/// and the four ref bytes `0x01` has and this does not.
///
/// The doc on [`TextureDescriptorBody`] named this as a third reader of that
/// struct before it had a fixture. It has one now, so the three cannot drift.
///
/// Where the *answer* goes is not in this record: unlike
/// [`crate::ops::tile::GetTileDimensions`], which names a guest buffer for the
/// host to write into, nothing here points anywhere. Whatever channel carries
/// the size and alignment back is not the command stream, and this capture
/// cannot see it.
pub fn heap_texture_size_and_align<'a>(
    op: &Op<'a>,
) -> Result<&'a TextureDescriptorBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_HEAP_TEXTURE_SIZE_AND_ALIGN);
    view::<TextureDescriptorBody>(op.payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{op, OP_HEADER_LEN};
    use core::mem::size_of;

    /// Assemble a texture operation from this module's own layout constants.
    ///
    /// This proves the view reads what the layout says, which is a different
    /// and weaker claim than "the layout matches Apple". The second claim is
    /// the oracle's job — see `tests/oracle_fixtures.rs`.
    #[allow(clippy::too_many_arguments)]
    fn synth(
        packed: u32,
        width: u32,
        height: u32,
        depth: u32,
        mips: u16,
        samples: u16,
        array: u16,
        options: u16,
    ) -> [u8; NEW_TEXTURE_TOTAL_LEN as usize] {
        let mut b = [0u8; NEW_TEXTURE_TOTAL_LEN as usize];
        b[0..4].copy_from_slice(&OPCODE_NEW_TEXTURE.to_le_bytes());
        b[4..8].copy_from_slice(&NEW_TEXTURE_TOTAL_LEN.to_le_bytes());
        b[8..12].copy_from_slice(&1u32.to_le_bytes()); // object_ref, first payload word
        b[12..16].copy_from_slice(&packed.to_le_bytes());
        b[16..20].copy_from_slice(&width.to_le_bytes());
        b[20..24].copy_from_slice(&height.to_le_bytes());
        b[24..28].copy_from_slice(&depth.to_le_bytes());
        b[28..30].copy_from_slice(&mips.to_le_bytes());
        b[30..32].copy_from_slice(&samples.to_le_bytes());
        b[32..34].copy_from_slice(&array.to_le_bytes());
        b[34..36].copy_from_slice(&options.to_le_bytes());
        b
    }

    #[test]
    fn the_payload_is_exactly_the_record_minus_its_header() {
        assert_eq!(
            size_of::<NewTextureBody>() + OP_HEADER_LEN,
            NEW_TEXTURE_TOTAL_LEN as usize
        );
        assert_eq!(core::mem::align_of::<NewTextureBody>(), 1);
        // The shared half, which `ops::heap_texture` embeds at its own offset.
        // A change here that left the total right would move that record's
        // `use_offset` without moving anything in this one.
        assert_eq!(size_of::<TextureDescriptorBody>(), TEXTURE_DESCRIPTOR_LEN);
        assert_eq!(core::mem::align_of::<TextureDescriptorBody>(), 1);

        // The wide form. Align-1 is not a nicety here: every `u32` in it sits at
        // an odd offset, so a field type that wanted four-byte alignment would
        // silently insert padding and every offset past `pixel_format` would be
        // wrong by the amount inserted.
        assert_eq!(
            size_of::<NewTextureWideBody>() + OP_HEADER_LEN,
            NEW_TEXTURE_WIDE_TOTAL_LEN as usize
        );
        assert_eq!(core::mem::align_of::<NewTextureWideBody>(), 1);
        assert_eq!(
            size_of::<WideTextureDescriptorBody>(),
            WIDE_TEXTURE_DESCRIPTOR_LEN
        );
        assert_eq!(core::mem::align_of::<WideTextureDescriptorBody>(), 1);
        // Eight bytes wider, and the eight are one promoted `usage` plus four
        // swizzle channels plus the byte nobody writes. Stated as arithmetic so
        // a change to either constant that keeps the difference has to be
        // deliberate.
        assert_eq!(
            WIDE_TEXTURE_DESCRIPTOR_LEN - TEXTURE_DESCRIPTOR_LEN,
            3 + 4 + 1
        );
    }

    /// The wide descriptor's fields land where the perturbation set says.
    ///
    /// Assembled from this module's own layout, so like `synth` above it proves
    /// the view agrees with the declaration and not that either matches Apple —
    /// `every_texture_fixture_reads_back_what_metal_was_asked_for` is the second
    /// claim. What it does catch on its own is a field reordering, because every
    /// value here is distinct and none is a plausible neighbour of another.
    #[test]
    fn the_wide_descriptor_reads_its_unaligned_fields() {
        let mut b = [0u8; NEW_TEXTURE_WIDE_TOTAL_LEN as usize];
        b[0..4].copy_from_slice(&OPCODE_NEW_TEXTURE_WIDE.to_le_bytes());
        b[4..8].copy_from_slice(&NEW_TEXTURE_WIDE_TOTAL_LEN.to_le_bytes());
        b[8..12].copy_from_slice(&3u32.to_le_bytes()); // object_ref
        let d = 12; // payload + object_ref
        b[d] = 0x42; // type 2, GPU-optimized contents, bit 7 clear
        b[d + 1..d + 3].copy_from_slice(&80u16.to_le_bytes());
        b[d + 3..d + 7].copy_from_slice(&5u32.to_le_bytes());
        b[d + 7..d + 11].copy_from_slice(&0x1111u32.to_le_bytes());
        b[d + 11..d + 15].copy_from_slice(&0x2222u32.to_le_bytes());
        b[d + 15..d + 19].copy_from_slice(&1u32.to_le_bytes());
        b[d + 19..d + 21].copy_from_slice(&1u16.to_le_bytes());
        b[d + 21..d + 23].copy_from_slice(&1u16.to_le_bytes());
        b[d + 23..d + 25].copy_from_slice(&1u16.to_le_bytes());
        b[d + 25..d + 27].copy_from_slice(&0x0020u16.to_le_bytes());
        b[d + 35] = 2;
        b[d + 36] = 3;
        b[d + 37] = 4;
        b[d + 38] = 5;
        b[d + 39] = 0xAA; // the byte the serializer never writes

        let o = op(&b, 0).expect("well formed");
        let t = new_texture_wide(&o).expect("fits");
        assert_eq!(t.object_ref.get(), 3);
        assert_eq!(t.desc.texture_type(), 2);
        assert!(!t.desc.framebuffer_only());
        assert!(!t.desc.is_drawable());
        assert!(!t.desc.write_swizzle_enabled());
        assert!(t.desc.allow_gpu_optimized_contents());
        assert_eq!(t.desc.pixel_format.get(), 80);
        assert_eq!(t.desc.usage.get(), 5);
        assert_eq!(t.desc.width.get(), 0x1111);
        assert_eq!(t.desc.height.get(), 0x2222);
        assert_eq!(t.desc.depth.get(), 1);
        assert_eq!(t.desc.mipmap_level_count.get(), 1);
        assert_eq!(t.desc.sample_count.get(), 1);
        assert_eq!(t.desc.array_length.get(), 1);
        assert_eq!(t.desc.storage_mode(), 2);
        assert_eq!(t.desc.cpu_cache_mode(), 0);
        assert_eq!(t.desc.hazard_tracking_mode(), 0);
        assert_eq!(
            (
                t.desc.swizzle_red,
                t.desc.swizzle_green,
                t.desc.swizzle_blue,
                t.desc.swizzle_alpha
            ),
            (2, 3, 4, 5)
        );
        // The unwritten byte is reachable and is not any of the fields above.
        assert_eq!(t.desc.declared_tail, 0xAA);
    }

    /// Bit 7 is a flag in both encodings and only the wide one assigns it.
    ///
    /// The narrow form's `texture_baseline` comes back with that bit unwritten,
    /// so it holds the guest's stale ring; the wide form writes it. A reader
    /// that folded the two into one accessor would report a flag that is
    /// sometimes noise.
    #[test]
    fn the_wide_form_alone_exposes_write_swizzle_enabled() {
        let narrow = synth(0x0050_05c2, 0x1111, 0x2222, 1, 1, 1, 1, 0x0020);
        let o = op(&narrow, 0).expect("well formed");
        let n = new_texture(&o).expect("fits");
        // 0xc2's bit 7 is set and the narrow accessor must not carry it into
        // any field it names.
        assert_eq!(n.desc.texture_type(), 2);
        assert!(!n.desc.framebuffer_only());
        assert!(!n.desc.is_drawable());

        // The wide accessor may read bit 7 because that form writes it.
        for raw in [0x42u8, 0xc2] {
            let mut b = [0u8; NEW_TEXTURE_WIDE_TOTAL_LEN as usize];
            b[0..4].copy_from_slice(&OPCODE_NEW_TEXTURE_WIDE.to_le_bytes());
            b[4..8].copy_from_slice(&NEW_TEXTURE_WIDE_TOTAL_LEN.to_le_bytes());
            b[12] = raw;
            let o = op(&b, 0).expect("well formed");
            let w = new_texture_wide(&o).expect("fits");
            assert_eq!(w.desc.texture_type(), 2);
            assert!(w.desc.allow_gpu_optimized_contents());
            assert_eq!(w.desc.write_swizzle_enabled(), raw == 0xc2);
            assert!(!w.desc.framebuffer_only());
            assert!(!w.desc.is_drawable());
        }
    }

    #[test]
    fn the_packed_word_splits_into_type_flags_usage_and_format() {
        // 0x005005c2: type 2, GPU-optimized contents, flags 0b00, usage 5,
        // format 80 — the shape the oracle's baseline produced. Bit 7 is set
        // here because `0xc2` is what a `0xAA`-filled arena produced; the view
        // must read the same fields whatever that bit holds, which is what the
        // second assertion pass below checks.
        let buf = synth(0x0050_05c2, 0x1111, 0x2222, 1, 1, 1, 1, 0x0020);
        let o = op(&buf, 0).expect("well formed");
        let t = new_texture(&o).expect("fits");

        assert_eq!(t.object_ref.get(), 1);
        assert_eq!(t.desc.texture_type(), 2);
        assert!(!t.desc.framebuffer_only());
        assert!(!t.desc.is_drawable());
        assert!(t.desc.allow_gpu_optimized_contents());
        assert_eq!(t.desc.usage(), 5);
        assert_eq!(t.desc.pixel_format(), 80);
        assert_eq!(t.desc.width.get(), 0x1111);
        assert_eq!(t.desc.height.get(), 0x2222);
        assert_eq!(t.desc.depth.get(), 1);
        assert_eq!(t.desc.mipmap_level_count.get(), 1);
        assert_eq!(t.desc.sample_count.get(), 1);
        assert_eq!(t.desc.array_length.get(), 1);
        assert_eq!(t.desc.storage_mode(), 2);
        assert_eq!(t.desc.cpu_cache_mode(), 0);
        assert_eq!(t.desc.hazard_tracking_mode(), 0);

        // The same record with `packed` bit 7 cleared. The serializer never
        // writes that bit, so on a real wire it is the guest's stale ring, and
        // no accessor may change its answer because of it.
        let flipped = synth(0x0050_0542, 0x1111, 0x2222, 1, 1, 1, 1, 0x0020);
        let o = op(&flipped, 0).expect("well formed");
        let f = new_texture(&o).expect("fits");
        assert_eq!(f.desc.texture_type(), t.desc.texture_type());
        assert_eq!(f.desc.framebuffer_only(), t.desc.framebuffer_only());
        assert_eq!(f.desc.is_drawable(), t.desc.is_drawable());
        assert_eq!(
            f.desc.allow_gpu_optimized_contents(),
            t.desc.allow_gpu_optimized_contents()
        );
        assert_eq!(f.desc.usage(), t.desc.usage());
        assert_eq!(f.desc.pixel_format(), t.desc.pixel_format());
    }

    #[test]
    fn the_resource_options_word_splits_into_cache_storage_and_hazard() {
        // Each triple is the whole word the oracle measured, so a shift error
        // in one accessor shows up as another accessor moving.
        for (options, cache, storage, hazard) in [
            (0x0020u16, 0u8, 2u8, 0u8), // baseline: private, default cache
            (0x0000, 0, 0, 0),          // shared
            (0x0010, 0, 1, 0),          // managed
            (0x0001, 1, 0, 0),          // shared + write-combined
            (0x0120, 0, 2, 1),          // private + untracked
            (0x0220, 0, 2, 2),          // private + tracked
        ] {
            let buf = synth(0x0050_05c2, 8, 8, 1, 1, 1, 1, options);
            let o = op(&buf, 0).expect("well formed");
            let t = new_texture(&o).expect("fits");
            assert_eq!(
                t.desc.cpu_cache_mode(),
                cache,
                "{options:#06x}: cpu cache mode"
            );
            assert_eq!(
                t.desc.storage_mode(),
                storage,
                "{options:#06x}: storage mode"
            );
            assert_eq!(
                t.desc.hazard_tracking_mode(),
                hazard,
                "{options:#06x}: hazard tracking mode"
            );
        }
    }

    #[test]
    fn each_packed_subfield_moves_only_its_own_bits() {
        // The perturbation discipline, applied to the accessors themselves: a
        // shift or mask error would show as a neighbouring field moving too.
        let base = synth(0x0050_05c2, 8, 8, 1, 1, 1, 1, 0);
        let o = op(&base, 0).expect("well formed");
        let b = new_texture(&o).expect("fits");
        let (ty, framebuffer, drawable, us, fmt, gpu) = (
            b.desc.texture_type(),
            b.desc.framebuffer_only(),
            b.desc.is_drawable(),
            b.desc.usage(),
            b.desc.pixel_format(),
            b.desc.allow_gpu_optimized_contents(),
        );

        for (packed, label) in [
            (0x0050_05c5u32, "type"),
            (0x0050_01c2, "usage"),
            (0x0046_05c2, "format"),
            (0x0050_0582, "gpu_optimized_contents"),
            (0x0050_05d2, "framebuffer_only"),
            (0x0050_05e2, "is_drawable"),
        ] {
            let buf = synth(packed, 8, 8, 1, 1, 1, 1, 0);
            let o = op(&buf, 0).expect("well formed");
            let t = new_texture(&o).expect("fits");
            let moved = [
                t.desc.texture_type() != ty,
                t.desc.usage() != us,
                t.desc.pixel_format() != fmt,
                t.desc.allow_gpu_optimized_contents() != gpu,
                t.desc.framebuffer_only() != framebuffer,
                t.desc.is_drawable() != drawable,
            ];
            assert_eq!(
                moved.iter().filter(|m| **m).count(),
                1,
                "{label} perturbation moved more than its own field"
            );
        }
    }

    #[test]
    fn storage_mode_reads_the_shift_the_oracle_measured() {
        for (options, mode) in [(0x0000u16, 0u8), (0x0010, 1), (0x0020, 2)] {
            let buf = synth(0x0050_05c2, 8, 8, 1, 1, 1, 1, options);
            let o = op(&buf, 0).expect("well formed");
            assert_eq!(new_texture(&o).expect("fits").desc.storage_mode(), mode);
        }
    }

    #[test]
    fn a_truncated_texture_operation_is_refused_rather_than_read_short() {
        let buf = synth(0, 0, 0, 0, 0, 0, 0, 0);
        // Claim the full length but present fewer bytes than the body needs.
        let o = op(&buf, 0).expect("well formed");
        let short = Op {
            header: o.header,
            payload: &o.payload[..8],
            offset: 0,
        };
        assert!(matches!(
            new_texture(&short),
            Err(WireError::Short { need: 36, have: 8 })
        ));
    }
}
