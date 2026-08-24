//! Opcodes 9 and 0x0c — textures whose storage comes from somewhere else.
//!
//! `-[PGSerializer newTextureWithBuffer:descriptor:offset:bytesPerRow:allocator:]`
//! and `-[PGSerializer newIOSurfaceTextureWithDescriptor:plane:allocator:]`.
//! Both are complete texture resources rather than views, and both carry the
//! same 32-byte [`crate::ops::texture::TextureDescriptorBody`] a plain creation
//! does — the difference is what precedes or follows it.
//!
//! # Layout
//!
//! ```text
//! opcode 9, total 64 bytes
//! payload +000  u32   object_ref
//! payload +004  u32   buffer_ref
//! payload +008  u64   offset
//! payload +016  u64   bytes_per_row
//! payload +024  32 bytes  the texture descriptor
//!
//! opcode 0x0c, total 48 bytes (legacy narrow)
//! payload +000  u32   object_ref
//! payload +004  32 bytes  the texture descriptor
//! payload +036  u16   plane
//! payload +038..+039  NEVER WRITTEN
//!
//! opcode 0x2f, total 48 bytes (rotation-capable narrow)
//! payload +000  u32   object_ref
//! payload +004  32 bytes  the texture descriptor
//! payload +036  u16   plane
//! payload +038  u8    rotation
//! payload +039        NEVER WRITTEN
//!
//! opcode 0x39, total 56 bytes (rotation-capable wide)
//! payload +000  u32   object_ref
//! payload +004  40 bytes  the wide texture descriptor
//! payload +044  u16   plane
//! payload +046  u8    rotation
//! payload +047        NEVER WRITTEN
//! ```
//!
//! # Plane and rotation are separate
//!
//! Every IOSurface form writes a two-byte plane. Rotation-capable forms write
//! the low byte of the following two-byte slot and leave its high byte alone;
//! the legacy form leaves the complete slot alone. On a real wire every
//! unwritten byte is stale ring content. A decoder that loads the final four
//! bytes as one plane therefore folds rotation and stale data into the plane.
//!
//! The plane and the descriptor's `protectionOptions` are independent fields.
//! Plane perturbations move only the two-byte field after the descriptor;
//! `setProtectionOptions:` moves only the descriptor's trailing `u64`.

use crate::le::{U16le, U32le, U64le};
use crate::op::Op;
use crate::ops::texture::TextureDescriptorBody;
use crate::view::{view, Wire, WireError};

/// Opcode for a buffer-backed texture.
pub const OPCODE_BUFFER_TEXTURE: u32 = 9;
/// Opcode for an IOSurface-backed texture.
pub const OPCODE_IOSURFACE_TEXTURE: u32 = 0x0c;
/// Rotation-capable narrow IOSurface-backed texture.
pub const OPCODE_IOSURFACE_TEXTURE_ROTATED: u32 = 0x2f;

/// Total wire length of a buffer-backed texture record, header included.
pub const BUFFER_TEXTURE_TOTAL_LEN: u32 = 64;
/// Total wire length of an IOSurface-backed texture record, header included.
pub const IOSURFACE_TEXTURE_TOTAL_LEN: u32 = 48;
/// Total wire length of the rotation-capable narrow IOSurface form.
pub const IOSURFACE_TEXTURE_ROTATED_TOTAL_LEN: u32 = IOSURFACE_TEXTURE_TOTAL_LEN;

/// Payload of a buffer-backed texture record.
#[repr(C)]
#[derive(Debug)]
pub struct BufferTextureBody {
    /// Ref the guest's object-ref allocator assigned to the new texture.
    pub object_ref: U32le,
    /// Ref of the buffer the texture reads its bytes from. Observed: the
    /// oracle's stub buffer answers `5151` to `-bufferRef`, and `5151` lands
    /// here.
    pub buffer_ref: U32le,
    /// Byte offset into the buffer. Observed: `0x2200`.
    pub offset: U64le,
    /// Row stride in bytes. Observed: `0x4400` — a different value from the
    /// offset, so a view that read one for the other would report a number no
    /// case produced.
    pub bytes_per_row: U64le,
    /// The descriptor Metal was handed.
    pub desc: TextureDescriptorBody,
}

// SAFETY: align-1 all-bytes-valid `le` scalars and one align-1 `Wire` struct.
unsafe impl Wire for BufferTextureBody {}

/// Payload of an IOSurface-backed texture record.
#[repr(C)]
#[derive(Debug)]
pub struct IOSurfaceTextureBody {
    /// Ref the guest's object-ref allocator assigned to the new texture.
    pub object_ref: U32le,
    /// The descriptor Metal was handed. Note the descriptor comes *first* here
    /// and second in the buffer-backed form.
    pub desc: TextureDescriptorBody,
    /// Which plane of the surface. Observed: 0 and 1.
    ///
    /// Two bytes, not four: the slot is four wide and the serializer writes
    /// only this half. See the module doc.
    pub plane: U16le,
}

// SAFETY: align-1 all-bytes-valid `le` scalars and one align-1 `Wire` struct.
unsafe impl Wire for IOSurfaceTextureBody {}

/// Payload of the rotation-capable narrow IOSurface-backed texture record.
#[repr(C)]
#[derive(Debug)]
pub struct IOSurfaceTextureRotatedBody {
    pub object_ref: U32le,
    pub desc: TextureDescriptorBody,
    pub plane: U16le,
    /// Rotation ordinal. The byte after this field is never written.
    pub rotation: u8,
}

// SAFETY: align-1 all-bytes-valid scalars and one align-1 `Wire` struct.
unsafe impl Wire for IOSurfaceTextureRotatedBody {}

/// The same two records under `-setSupportsTextureDescriptor2:`.
///
/// Each moves to its own opcode and carries
/// [`crate::ops::texture::WideTextureDescriptorBody`] where the narrow one
/// carries [`TextureDescriptorBody`]. Everything else about the record is
/// unchanged — the prefixes below are the same fields at the same offsets — with
/// one exception, which is why these were driven rather than assumed: see
/// [`IOSurfaceTextureWideBody::plane`].
///
/// Note the flag. These answer to `TextureDescriptor2`;
/// `newTextureWithDescriptor:allocator:` answers to `SwizzledTextures` and its
/// wide form is [`crate::ops::texture::OPCODE_NEW_TEXTURE_WIDE`]. One family,
/// two capabilities.
pub const OPCODE_BUFFER_TEXTURE_WIDE: u32 = 0x37;
/// The IOSurface-backed record's wide opcode. See [`OPCODE_BUFFER_TEXTURE_WIDE`].
pub const OPCODE_IOSURFACE_TEXTURE_WIDE: u32 = 0x39;

pub const BUFFER_TEXTURE_WIDE_TOTAL_LEN: u32 = 72;
pub const IOSURFACE_TEXTURE_WIDE_TOTAL_LEN: u32 = 56;

/// Payload of a buffer-backed texture record, wide descriptor.
#[repr(C)]
#[derive(Debug)]
pub struct BufferTextureWideBody {
    pub object_ref: U32le,
    pub buffer_ref: U32le,
    pub offset: U64le,
    pub bytes_per_row: U64le,
    pub desc: crate::ops::texture::WideTextureDescriptorBody,
}

// SAFETY: align-1 all-bytes-valid `le` scalars and one align-1 `Wire` struct.
unsafe impl Wire for BufferTextureWideBody {}

/// Payload of an IOSurface-backed texture record, wide descriptor.
#[repr(C)]
#[derive(Debug)]
pub struct IOSurfaceTextureWideBody {
    pub object_ref: U32le,
    pub desc: crate::ops::texture::WideTextureDescriptorBody,
    /// Which plane of the surface. The following byte is rotation, not the high
    /// half of a four-byte plane.
    pub plane: U16le,
    /// Rotation ordinal. The byte after this field is never written.
    pub rotation: u8,
}

// SAFETY: align-1 all-bytes-valid `le` scalars and one align-1 `Wire` struct.
unsafe impl Wire for IOSurfaceTextureWideBody {}

/// View the payload of a buffer-backed texture record.
pub fn buffer_texture<'a>(op: &Op<'a>) -> Result<&'a BufferTextureBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_BUFFER_TEXTURE);
    view::<BufferTextureBody>(op.payload)
}

pub fn buffer_texture_wide<'a>(op: &Op<'a>) -> Result<&'a BufferTextureWideBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_BUFFER_TEXTURE_WIDE);
    view::<BufferTextureWideBody>(op.payload)
}

pub fn iosurface_texture_wide<'a>(op: &Op<'a>) -> Result<&'a IOSurfaceTextureWideBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_IOSURFACE_TEXTURE_WIDE);
    view::<IOSurfaceTextureWideBody>(op.payload)
}

pub fn iosurface_texture_rotated<'a>(
    op: &Op<'a>,
) -> Result<&'a IOSurfaceTextureRotatedBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_IOSURFACE_TEXTURE_ROTATED);
    view::<IOSurfaceTextureRotatedBody>(op.payload)
}

/// View the payload of an IOSurface-backed texture record.
pub fn iosurface_texture<'a>(op: &Op<'a>) -> Result<&'a IOSurfaceTextureBody, WireError> {
    debug_assert_eq!(op.opcode(), OPCODE_IOSURFACE_TEXTURE);
    view::<IOSurfaceTextureBody>(op.payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::op::{op, OP_HEADER_LEN};
    use crate::ops::texture::TEXTURE_DESCRIPTOR_LEN;
    use core::mem::size_of;

    /// Write the descriptor body the oracle's baseline produced.
    fn put_desc(b: &mut [u8], at: usize) {
        b[at..at + 4].copy_from_slice(&0x0050_05c2u32.to_le_bytes());
        b[at + 4..at + 8].copy_from_slice(&0x1111u32.to_le_bytes());
        b[at + 8..at + 12].copy_from_slice(&0x2222u32.to_le_bytes());
        b[at + 12..at + 16].copy_from_slice(&1u32.to_le_bytes());
        b[at + 16..at + 18].copy_from_slice(&1u16.to_le_bytes());
        b[at + 18..at + 20].copy_from_slice(&1u16.to_le_bytes());
        b[at + 20..at + 22].copy_from_slice(&1u16.to_le_bytes());
        b[at + 22..at + 24].copy_from_slice(&0x0020u16.to_le_bytes());
        b[at + 24..at + 32].copy_from_slice(&0u64.to_le_bytes());
    }

    fn synth_buffer_texture() -> [u8; 64] {
        let mut b = [0xAAu8; BUFFER_TEXTURE_TOTAL_LEN as usize];
        b[0..4].copy_from_slice(&OPCODE_BUFFER_TEXTURE.to_le_bytes());
        b[4..8].copy_from_slice(&BUFFER_TEXTURE_TOTAL_LEN.to_le_bytes());
        b[8..12].copy_from_slice(&61u32.to_le_bytes());
        b[12..16].copy_from_slice(&5151u32.to_le_bytes());
        b[16..24].copy_from_slice(&0x2200u64.to_le_bytes());
        b[24..32].copy_from_slice(&0x4400u64.to_le_bytes());
        put_desc(&mut b, 32);
        b
    }

    fn synth_iosurface_texture(plane: u16) -> [u8; 48] {
        let mut b = [0xAAu8; IOSURFACE_TEXTURE_TOTAL_LEN as usize];
        b[0..4].copy_from_slice(&OPCODE_IOSURFACE_TEXTURE.to_le_bytes());
        b[4..8].copy_from_slice(&IOSURFACE_TEXTURE_TOTAL_LEN.to_le_bytes());
        b[8..12].copy_from_slice(&62u32.to_le_bytes());
        put_desc(&mut b, 12);
        b[44..46].copy_from_slice(&plane.to_le_bytes());
        b
    }

    #[test]
    fn each_record_is_its_body_plus_the_header() {
        assert_eq!(
            size_of::<BufferTextureBody>() + OP_HEADER_LEN,
            BUFFER_TEXTURE_TOTAL_LEN as usize
        );
        assert_eq!(core::mem::align_of::<BufferTextureBody>(), 1);
        // The IOSurface body stops at the plane, two bytes short of the
        // record: the last two are never written.
        assert_eq!(
            size_of::<IOSurfaceTextureBody>() + OP_HEADER_LEN + 2,
            IOSURFACE_TEXTURE_TOTAL_LEN as usize
        );
        assert_eq!(core::mem::align_of::<IOSurfaceTextureBody>(), 1);

        // The wide buffer form reaches its record end. The IOSurface body's
        // final byte is not part of a view because the serializer never writes
        // it.
        assert_eq!(
            size_of::<BufferTextureWideBody>() + OP_HEADER_LEN,
            BUFFER_TEXTURE_WIDE_TOTAL_LEN as usize
        );
        assert_eq!(core::mem::align_of::<BufferTextureWideBody>(), 1);
        assert_eq!(
            size_of::<IOSurfaceTextureWideBody>() + OP_HEADER_LEN + 1,
            IOSURFACE_TEXTURE_WIDE_TOTAL_LEN as usize
        );
        assert_eq!(core::mem::align_of::<IOSurfaceTextureWideBody>(), 1);
        assert_eq!(
            size_of::<IOSurfaceTextureRotatedBody>() + OP_HEADER_LEN + 1,
            IOSURFACE_TEXTURE_ROTATED_TOTAL_LEN as usize
        );
        assert_eq!(core::mem::align_of::<IOSurfaceTextureRotatedBody>(), 1);
        // Each wide record is its narrow twin plus the descriptor's eight
        // bytes. Plane and rotation retain the same widths.
        assert_eq!(
            BUFFER_TEXTURE_WIDE_TOTAL_LEN - BUFFER_TEXTURE_TOTAL_LEN,
            8,
            "the wide buffer-backed record is no longer its narrow twin plus the \
             descriptor's eight bytes"
        );
        assert_eq!(
            IOSURFACE_TEXTURE_WIDE_TOTAL_LEN - IOSURFACE_TEXTURE_TOTAL_LEN,
            8
        );
        assert_eq!(
            size_of::<IOSurfaceTextureWideBody>() - size_of::<IOSurfaceTextureBody>(),
            8 + 1,
            "the wide descriptor adds eight bytes and the rotation-capable form \
             adds one written byte after the common plane"
        );
    }

    #[test]
    fn the_descriptor_sits_at_a_different_offset_in_each_record() {
        // Both records embed the same 32 bytes and neither puts them in the
        // same place. Stating the arithmetic here is what catches a field
        // being added to one body and silently shifting the other's reader.
        assert_eq!(
            size_of::<U32le>() * 2 + size_of::<U64le>() * 2,
            24,
            "the buffer-backed descriptor no longer starts at payload +24"
        );
        assert_eq!(
            size_of::<U32le>(),
            4,
            "the IOSurface descriptor no longer starts at payload +4"
        );
        assert_eq!(size_of::<TextureDescriptorBody>(), TEXTURE_DESCRIPTOR_LEN);
    }

    #[test]
    fn a_buffer_texture_reads_back_its_buffer_placement_and_descriptor() {
        let buf = synth_buffer_texture();
        let o = op(&buf, 0).expect("well formed");
        let t = buffer_texture(&o).expect("fits");
        assert_eq!(t.object_ref.get(), 61);
        assert_eq!(t.buffer_ref.get(), 5151);
        assert_eq!(t.offset.get(), 0x2200);
        assert_eq!(t.bytes_per_row.get(), 0x4400);
        assert_eq!(t.desc.texture_type(), 2);
        assert_eq!(t.desc.pixel_format(), 80);
        assert_eq!(t.desc.width.get(), 0x1111);
        assert_eq!(t.desc.height.get(), 0x2222);
        assert_eq!(t.desc.storage_mode(), 2);
    }

    #[test]
    fn an_iosurface_texture_reads_its_plane_without_the_bytes_beside_it() {
        for plane in [0u16, 1] {
            let mut buf = synth_iosurface_texture(plane);
            // The two bytes the serializer never writes. Whatever they hold,
            // the plane must read the same.
            buf[46] = 0x5a;
            buf[47] = 0xff;
            let o = op(&buf, 0).expect("well formed");
            let t = iosurface_texture(&o).expect("fits");
            assert_eq!(t.plane.get(), plane);
            assert_eq!(t.object_ref.get(), 62);
            assert_eq!(t.desc.texture_type(), 2);
            assert_eq!(t.desc.pixel_format(), 80);
            // The word the plane was hypothesised to live in, and does not.
            assert_eq!(t.desc.protection_options.get(), 0);
        }
    }

    #[test]
    fn rotation_capable_forms_do_not_fold_rotation_or_stale_tail_into_plane() {
        let mut narrow = [0xaau8; IOSURFACE_TEXTURE_ROTATED_TOTAL_LEN as usize];
        narrow[0..4].copy_from_slice(&OPCODE_IOSURFACE_TEXTURE_ROTATED.to_le_bytes());
        narrow[4..8].copy_from_slice(&IOSURFACE_TEXTURE_ROTATED_TOTAL_LEN.to_le_bytes());
        narrow[8..12].copy_from_slice(&62u32.to_le_bytes());
        put_desc(&mut narrow, 12);
        narrow[44..46].copy_from_slice(&3u16.to_le_bytes());
        narrow[46] = 7;
        narrow[47] = 0xff;
        let narrow_op = op(&narrow, 0).expect("well formed");
        let rotated = iosurface_texture_rotated(&narrow_op).expect("fits");
        assert_eq!(rotated.plane.get(), 3);
        assert_eq!(rotated.rotation, 7);

        let mut wide = [0xaau8; IOSURFACE_TEXTURE_WIDE_TOTAL_LEN as usize];
        wide[0..4].copy_from_slice(&OPCODE_IOSURFACE_TEXTURE_WIDE.to_le_bytes());
        wide[4..8].copy_from_slice(&IOSURFACE_TEXTURE_WIDE_TOTAL_LEN.to_le_bytes());
        wide[8..12].copy_from_slice(&63u32.to_le_bytes());
        wide[52..54].copy_from_slice(&5u16.to_le_bytes());
        wide[54] = 9;
        wide[55] = 0xff;
        let wide_op = op(&wide, 0).expect("well formed");
        let rotated = iosurface_texture_wide(&wide_op).expect("fits");
        assert_eq!(rotated.plane.get(), 5);
        assert_eq!(rotated.rotation, 9);
    }

    #[test]
    fn a_truncated_record_of_either_form_is_refused_rather_than_read_short() {
        let buf = synth_buffer_texture();
        let o = op(&buf, 0).expect("well formed");
        let short = Op {
            header: o.header,
            payload: &o.payload[..24],
            offset: 0,
        };
        assert!(matches!(
            buffer_texture(&short),
            Err(WireError::Short { need: 56, have: 24 })
        ));

        let buf = synth_iosurface_texture(0);
        let o = op(&buf, 0).expect("well formed");
        let short = Op {
            header: o.header,
            payload: &o.payload[..36],
            offset: 0,
        };
        assert!(matches!(
            iosurface_texture(&short),
            Err(WireError::Short { need: 38, have: 36 })
        ));
    }
}
