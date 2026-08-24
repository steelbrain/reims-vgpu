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

const TEXTURE_USAGE_SHADER_READ: u32 = 1 << 0;
const TEXTURE_USAGE_SHADER_WRITE: u32 = 1 << 1;
const SUPPORTED_TEXTURE_USAGE: u32 = TEXTURE_USAGE_SHADER_READ | TEXTURE_USAGE_SHADER_WRITE;
// `MTLResourceStorageModePrivate`: the storage-mode ordinal occupies
// `resource_options[7:4]`; the wire crate owns and tests that field projection.
const PRIVATE_DEFAULT_CACHE_RESOURCE_OPTIONS: u16 = 2 << 4;

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
    UnsupportedTextureShape,
    UnsupportedUsage,
    UnsupportedResourceOptions,
    UnsupportedSwizzle,
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
            Self::UnsupportedTextureShape => "heap_query_unsupported_texture_shape",
            Self::UnsupportedUsage => "heap_query_unsupported_usage",
            Self::UnsupportedResourceOptions => "heap_query_unsupported_resource_options",
            Self::UnsupportedSwizzle => "heap_query_unsupported_swizzle",
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

/// Resolve the descriptor into the one backend-neutral image definition used
/// both for this query and for later heap placement.
///
/// This first admitted cell is deliberately narrow: the execution path can
/// currently represent private, shader-readable and shader-writable 2D images
/// with one level, layer, and sample. Other declarations refuse by a typed
/// reason until their native creation contract is implemented.
pub fn image_plan(
    desc: &TextureDescriptor,
) -> Result<reims_vgpu_core::HeapTextureImagePlan, QueryError> {
    if desc.texture_type != 2
        || desc.width == 0
        || desc.height == 0
        || desc.depth != 1
        || desc.mipmap_level_count != 1
        || desc.sample_count != 1
        || desc.array_length != 1
        || desc.framebuffer_only
        || desc.is_drawable
    {
        return Err(QueryError::UnsupportedTextureShape);
    }
    // MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite. Render-target,
    // pixel-format-view, and atomic usage require additional native image
    // features and are not silently widened into this plan.
    if desc.usage != SUPPORTED_TEXTURE_USAGE {
        return Err(QueryError::UnsupportedUsage);
    }
    // MTLStorageModePrivate. Heap storage has no guest-addressable bytes, and
    // the current execution rail implements only private GPU content.
    if desc.resource_options != PRIVATE_DEFAULT_CACHE_RESOURCE_OPTIONS
        || desc.storage_mode() != reims_vgpu_protocol::StorageMode::Private
        || desc.protection_options != 0
    {
        return Err(QueryError::UnsupportedResourceOptions);
    }
    if desc.write_swizzle_enabled == Some(true)
        || desc.swizzle.is_some_and(|raw| {
            reims_vgpu_protocol::swizzle_plan(&raw)
                .is_none_or(|plan| !reims_vgpu_protocol::swizzle_is_identity(&plan))
        })
    {
        return Err(QueryError::UnsupportedSwizzle);
    }
    let format = reims_vgpu_core::pixel_format::compute_sampled_image_format(desc.pixel_format)
        .ok_or(QueryError::UnknownPixelFormat)?;
    if reims_vgpu_core::pixel_format::storage_image_format(desc.pixel_format).is_none() {
        return Err(QueryError::UnsupportedUsage);
    }
    Ok(reims_vgpu_core::HeapTextureImagePlan {
        format,
        extent: [desc.width, desc.height, desc.depth],
        mip_levels: u32::from(desc.mipmap_level_count),
        array_layers: u32::from(desc.array_length),
        sample_count: u32::from(desc.sample_count),
        sampled: true,
        storage: true,
    })
}

pub fn query_size_and_align(
    desc: &TextureDescriptor,
    query: impl FnOnce(
        reims_vgpu_core::HeapTextureImagePlan,
    ) -> Option<reims_vgpu_core::HeapTextureRequirements>,
) -> Result<SizeAndAlign, QueryError> {
    let plan = image_plan(desc)?;
    let requirement = query(plan).ok_or(QueryError::HostRequirementsUnavailable)?;
    if requirement.size == 0 || requirement.alignment == 0 {
        return Err(QueryError::ZeroRequirement);
    }
    Ok(SizeAndAlign {
        size: requirement.size,
        align: requirement.alignment,
    })
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

    #[test]
    fn delegates_the_resolved_image_plan_and_returns_backend_requirements() {
        let request = decode_request(&live_request_fixture()).unwrap();
        let result = query_size_and_align(&request.descriptor, |plan| {
            assert_eq!(plan.extent, [180, 135, 1]);
            assert!(plan.sampled);
            assert!(plan.storage);
            Some(reims_vgpu_core::HeapTextureRequirements {
                size: 0x78000,
                alignment: 0x80,
            })
        })
        .unwrap();
        assert_eq!(
            result,
            SizeAndAlign {
                size: 0x78000,
                align: 0x80
            }
        );
    }

    #[test]
    fn refuses_zero_backend_requirements() {
        let request = decode_request(&live_request_fixture()).unwrap();
        assert_eq!(
            query_size_and_align(&request.descriptor, |_| {
                Some(reims_vgpu_core::HeapTextureRequirements {
                    size: 0,
                    alignment: 0x80,
                })
            }),
            Err(QueryError::ZeroRequirement)
        );
    }

    #[test]
    fn refuses_before_calling_the_backend_for_an_unimplemented_shape() {
        let mut request = decode_request(&live_request_fixture()).unwrap();
        request.descriptor.mipmap_level_count = 2;
        let mut called = false;
        assert_eq!(
            query_size_and_align(&request.descriptor, |_| {
                called = true;
                None
            }),
            Err(QueryError::UnsupportedTextureShape)
        );
        assert!(!called);
    }
}
