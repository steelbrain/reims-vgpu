//! `CmdHeapTextureSizeAndAlign` request decode and host requirement query.
//!
//! The command carries a `PGSerializedTextureDescriptor` record. The recovered
//! Apple host routine reconstructs an `MTLTextureDescriptor` and returns the
//! device's `MTLSizeAndAlign` as two little-endian `u64`s.

use reims_vgpu_core::endian::{ld32, ld64, st64};
use reims_vgpu_wire::ops::texture as wire;

pub use reims_vgpu_protocol::TextureDeclaration as TextureDescriptor;

pub const REQUEST_HEADER_LEN: usize = 24;
pub const REPLY_LEN: usize = 16;
/// The embedded record is `heapTextureSizeAndAlignWithDescriptor:`, so its tag
/// is that selector's opcode and its length is that record's length. Both come
/// from the crate that derived them rather than being written again here.
pub const SERIALIZED_TEXTURE_TAG: u32 = wire::OPCODE_HEAP_TEXTURE_SIZE_AND_ALIGN;
pub const SERIALIZED_TEXTURE_LEN: usize = wire::HEAP_TEXTURE_SIZE_AND_ALIGN_TOTAL_LEN as usize;
pub const TEXTURE_BODY_LEN: usize = wire::TEXTURE_DESCRIPTOR_LEN;
/// The same descriptor once the guest's serializer has a texture-descriptor
/// capability on: eight bytes wider, with `usage` promoted out of the packed
/// word into a `u32` and a four-channel swizzle appended.
///
/// Two different flags select it depending on the record — `SwizzledTextures`
/// for the plain creation, `TextureDescriptor2` for the four that embed it —
/// and neither is a length the caller may infer. Every record carrying this
/// body has an opcode of its own; see
/// [`crate::runtime::decode::resource::decode_heap_texture`].
pub const WIDE_TEXTURE_BODY_LEN: usize = wire::WIDE_TEXTURE_DESCRIPTOR_LEN;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    pub task_id: u32,
    pub reply_gva: u64,
    pub reply_len: u64,
    pub descriptor: TextureDescriptor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizeAndAlign {
    pub size: u64,
    pub align: u64,
}

impl SizeAndAlign {
    pub fn encode(self) -> [u8; REPLY_LEN] {
        let mut out = [0u8; REPLY_LEN];
        st64(&mut out[0..8], self.size);
        st64(&mut out[8..16], self.align);
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueryError {
    ShortPayload,
    BadReplyLength,
    BadSerializerLength,
    UnknownSerializerTag,
    BadDescriptorLength,
    UnknownTextureType,
    UnknownPixelFormat,
    UnknownUsage,
    UnknownResourceOptions,
    HostRequirementsUnavailable,
    ZeroRequirement,
    /// The request names a task id that resolves to no active task, so there is
    /// nowhere to write the reply. Checked by the caller in `runtime/drain/mod.rs`
    /// rather than here — the vocabulary still owns the reason, because the
    /// alternative is one untyped `reason=bad_task` sitting inside an event
    /// family where every other reason is typed.
    BadTask,
}

impl crate::observe::Decline for QueryError {
    /// Every slug carries the `heap_query_` prefix.
    ///
    /// Not decoration: these names are generic enough (`short_payload`,
    /// `unknown_pixel_format`, `unknown_usage`) that they describe checks half
    /// the crate also makes. `unknown_pixel_format` in fact **collided** with
    /// `TranslateReason`'s, and no per-enum uniqueness test could see it — both
    /// enums were internally consistent, and a grep of the fail log for that
    /// slug would have returned a mix of two unrelated subsystems' refusals.
    /// The prefix is what makes the check nameable at crate scope.
    fn slug(&self) -> &'static str {
        match self {
            Self::ShortPayload => "heap_query_short_payload",
            Self::BadReplyLength => "heap_query_bad_reply_length",
            Self::BadSerializerLength => "heap_query_bad_serializer_length",
            Self::UnknownSerializerTag => "heap_query_unknown_serializer_tag",
            Self::BadDescriptorLength => "heap_query_bad_descriptor_length",
            Self::UnknownTextureType => "heap_query_unknown_texture_type",
            Self::UnknownPixelFormat => "heap_query_unknown_pixel_format",
            Self::UnknownUsage => "heap_query_unknown_usage",
            Self::UnknownResourceOptions => "heap_query_unknown_resource_options",
            Self::HostRequirementsUnavailable => "heap_query_host_requirements_unavailable",
            Self::ZeroRequirement => "heap_query_zero_requirement",
            Self::BadTask => "heap_query_bad_task",
        }
    }
}

pub fn decode_request(payload: &[u8]) -> Result<Request, QueryError> {
    if payload.len() < REQUEST_HEADER_LEN {
        return Err(QueryError::ShortPayload);
    }
    let task_id = ld32(&payload[0..]);
    let reply_gva = ld64(&payload[4..]);
    let reply_len = ld64(&payload[12..]);
    if reply_gva == 0 || reply_len < REPLY_LEN as u64 {
        return Err(QueryError::BadReplyLength);
    }
    let serializer_len = ld32(&payload[20..]) as usize;
    if serializer_len != SERIALIZED_TEXTURE_LEN
        || payload.len() != REQUEST_HEADER_LEN + serializer_len
    {
        return Err(QueryError::BadSerializerLength);
    }
    let serialized = &payload[REQUEST_HEADER_LEN..];
    if ld32(serialized) != SERIALIZED_TEXTURE_TAG {
        return Err(QueryError::UnknownSerializerTag);
    }
    if ld32(&serialized[4..]) as usize != SERIALIZED_TEXTURE_LEN
        || serialized.len() != SERIALIZED_TEXTURE_LEN
    {
        return Err(QueryError::BadDescriptorLength);
    }
    let body = &serialized[8..];
    let descriptor = decode_serialized_texture_descriptor(body)?;
    Ok(Request {
        task_id,
        reply_gva,
        reply_len,
        descriptor,
    })
}

/// Decode the shared 32-byte `PGSerializedTextureDescriptor` body.
///
/// The same body is embedded in heap-texture resource opcode `0x15` and in the
/// buffer-backed opcode 9; keeping one decoder prevents the query and resource
/// paths from drifting.
///
/// Read through `reims_vgpu_wire`'s view rather than at offsets restated here,
/// so a field this device names is the field Apple's bytes derived. The two
/// `framebufferOnly`, `isDrawable`, and `protectionOptions` were independently
/// attributed by controlled serializer perturbations.
pub fn decode_serialized_texture_descriptor(body: &[u8]) -> Result<TextureDescriptor, QueryError> {
    if body.len() != TEXTURE_BODY_LEN {
        return Err(QueryError::BadDescriptorLength);
    }
    let d: &wire::TextureDescriptorBody =
        reims_vgpu_wire::view(body).map_err(|_| QueryError::BadDescriptorLength)?;
    Ok(reims_vgpu_protocol::texture_declaration_from_narrow(d))
}

/// Decode the 40-byte wide `PGSerializedTextureDescriptor` body.
///
/// The wide form is not a longer narrow one: `usage` leaves the packed word for
/// a `u32` of its own and four swizzle ordinals trail, so every field after the
/// first byte sits at a different offset. Which of the two a record carries is
/// a property of its **opcode**, never of its length — see
/// [`crate::runtime::decode::resource::decode_heap_texture`].
///
/// The fortieth byte is declared and never written, so nothing reads it.
pub fn decode_wide_serialized_texture_descriptor(
    body: &[u8],
) -> Result<TextureDescriptor, QueryError> {
    if body.len() != WIDE_TEXTURE_BODY_LEN {
        return Err(QueryError::BadDescriptorLength);
    }
    let d: &wire::WideTextureDescriptorBody =
        reims_vgpu_wire::view(body).map_err(|_| QueryError::BadDescriptorLength)?;
    Ok(reims_vgpu_protocol::texture_declaration_from_wide(d))
}

pub fn query_size_and_align(_desc: &TextureDescriptor) -> Result<SizeAndAlign, QueryError> {
    // Vulkan image requirements are not yet proven equivalent to the guest heap
    // placement contract, so refuse rather than return a guessed layout.
    Err(QueryError::HostRequirementsUnavailable)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn live_request_fixture() -> Vec<u8> {
        let words = [
            0x1u32, 0x162200, 0x0, 0x10, 0x0, 0x28, 0x16, 0x28, 0x7d0342, 0xb4, 0x87, 0x1, 0x10001,
            0x200001, 0x0, 0x0,
        ];
        words.into_iter().flat_map(u32::to_le_bytes).collect()
    }

    #[test]
    fn decodes_live_heap_texture_query() {
        let request = decode_request(&live_request_fixture()).unwrap();
        assert_eq!(request.task_id, 1);
        assert_eq!(request.reply_gva, 0x162200);
        assert_eq!(request.reply_len, 16);
        assert_eq!(
            request.descriptor,
            TextureDescriptor {
                texture_type: 2,
                framebuffer_only: false,
                is_drawable: false,
                write_swizzle_enabled: None,
                allow_gpu_optimized_contents: true,
                usage: 3,
                pixel_format: 125,
                width: 180,
                height: 135,
                depth: 1,
                mipmap_level_count: 1,
                sample_count: 1,
                array_length: 1,
                resource_options: 0x20,
                protection_options: 0,
                // The narrow body has no swizzle field at all. `None` rather
                // than the identity: see [`TextureDescriptor::swizzle`].
                swizzle: None,
            }
        );
    }

    #[test]
    fn rejects_unknown_serializer_version() {
        let mut payload = live_request_fixture();
        payload[24..28].copy_from_slice(&0x99u32.to_le_bytes());
        assert_eq!(
            decode_request(&payload),
            Err(QueryError::UnknownSerializerTag)
        );
    }

    #[test]
    fn encodes_two_u64_reply_fields() {
        let reply = SizeAndAlign {
            size: 0x78000,
            align: 0x80,
        }
        .encode();
        assert_eq!(ld64(&reply[0..]), 0x78000);
        assert_eq!(ld64(&reply[8..]), 0x80);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn native_query_returns_nonzero_requirement() {
        let request = decode_request(&live_request_fixture()).unwrap();
        let result = query_size_and_align(&request.descriptor).unwrap();
        assert!(result.size >= 180 * 135 * 16);
        assert!(result.align.is_power_of_two());
    }
}
