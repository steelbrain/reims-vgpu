//! The video-encoder ops (`tag >> 12 == `[`RGFX_OP_GROUP_VIDEO`](super::RGFX_OP_GROUP_VIDEO)):
//! the hardware video work of the guest's video driver, which runs in command buffers of its
//! own on a queue of its session's process, so that each picture completes, names the
//! surfaces it writes and holds their CPU locks as any command buffer does
//! (docs/graphics-kext/design.md, "Hardware video").
//!
//! A video encoder is opened by the framing `ENCODER_BEGIN_VIDEO` op, whose body
//! ([`RgfxOpVideoPass`](super::RgfxOpVideoPass)) names the session
//! ([`RGFX_CMD_VIDEO_SESSION_CREATE`](crate::wire::RGFX_CMD_VIDEO_SESSION_CREATE)), and closed by
//! `ENCODER_END`. A body struct is the op's bytes **after** its
//! [`RgfxOpHeader`](super::RgfxOpHeader) (see the [module doc](super)). The tags stay in the
//! `0x4xxx` range.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// Op `DECODE_H264` ([`RgfxVideoOpDecodeH264`], then its parameter sets): decode one H.264
/// access unit into an IOSurface of two planes, NV12's (`'420v'` or `'420f'`).
///
/// The access unit is its slices' NAL units, in Annex-B form (each after a start code), the
/// `bitstream_size` bytes at `bitstream_offset` in the buffer `bitstream_buffer_obj_id`, which
/// the host reads in place. The SPS and PPS NAL units the slices use, in Annex-B form too,
/// follow the body (`parameter_sets_size` bytes, padded with zeros to a multiple of 8) where
/// they changed since the session's last decode, its first included; the host keeps those it
/// was given before, by their ids, as a decoder of a stream does.
///
/// The picture lands in the surface whose planes `luma_texture_obj_id` (`R8Unorm`, plane 0) and
/// `chroma_texture_obj_id` (`RG8Unorm`, plane 1) lie over (`TEXTURE_FROM_IOSURFACE`), at their
/// origin, in its coded size cropped as the SPS says. The command buffer writes the surface: a
/// CPU lock of it waits for the buffer's completion, as for any GPU write. Which pictures the
/// stream references, and when they leave the decoded picture buffer, is the host's, from the
/// stream: the surface holds the output alone.
pub const RGFX_OP_VIDEO_DECODE_H264: u16 = 0x4000;

/// The body of [`RGFX_OP_VIDEO_DECODE_H264`], followed by its parameter sets.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVideoOpDecodeH264 {
    /// The buffer that holds the access unit.
    pub bitstream_buffer_obj_id: u64,
    /// The access unit's first byte in the buffer.
    pub bitstream_offset: u64,
    /// The access unit's bytes.
    pub bitstream_size: u64,
    /// The texture over the output surface's plane 0.
    pub luma_texture_obj_id: u64,
    /// The texture over the output surface's plane 1.
    pub chroma_texture_obj_id: u64,
    /// The bytes of SPS and PPS NAL units after this body, before its padding; 0 for none.
    pub parameter_sets_size: u32,
    /// Zero.
    pub reserved_2c: u32,
}

/// Where a `DECODE_H264` op's parameter sets start: right after its body.
pub const RGFX_VIDEO_DECODE_H264_PARAMETER_SETS_OFFSET: u32 = 48;

/// Op `DECODE_H265` ([`RgfxVideoOpDecodeH265`], then its parameter sets): decode one HEVC
/// access unit into an IOSurface of two planes: NV12's (`'420v'` or `'420f'`) for a stream of 8
/// bits, P010's (`'x420'` or `'xf20'`) for one of 10.
///
/// As [`RGFX_OP_VIDEO_DECODE_H264`], on a session of HEVC: the access unit is its slice
/// segments' NAL units in Annex-B form, the `bitstream_size` bytes at `bitstream_offset` in the
/// buffer `bitstream_buffer_obj_id`, which the host reads in place, and the SPS and PPS NAL units
/// they use follow the body, in Annex-B form (`parameter_sets_size` bytes, padded with zeros to a
/// multiple of 8), where they changed since the session's last decode, its first included. A VPS
/// there is ignored: the host's decoder makes its own from the SPS.
///
/// The picture lands in the surface whose planes `luma_texture_obj_id` (plane 0: `R8Unorm`, or
/// `R16Unorm` at 10 bits) and `chroma_texture_obj_id` (plane 1: `RG8Unorm`, or `RG16Unorm`) lie
/// over, at their origin, in its coded size cropped as the SPS says; a sample of 10 bits is the
/// high bits of its 16. An access unit whose picture the stream's random access skips (a RASL
/// picture after the CRA picture the session started at) lands nothing, and its command buffer
/// completes as any other's.
pub const RGFX_OP_VIDEO_DECODE_H265: u16 = 0x4001;

/// The body of [`RGFX_OP_VIDEO_DECODE_H265`], followed by its parameter sets.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVideoOpDecodeH265 {
    /// The buffer that holds the access unit.
    pub bitstream_buffer_obj_id: u64,
    /// The access unit's first byte in the buffer.
    pub bitstream_offset: u64,
    /// The access unit's bytes.
    pub bitstream_size: u64,
    /// The texture over the output surface's plane 0.
    pub luma_texture_obj_id: u64,
    /// The texture over the output surface's plane 1.
    pub chroma_texture_obj_id: u64,
    /// The bytes of the SPS and PPS NAL units after this body, before its padding; 0 for none.
    pub parameter_sets_size: u32,
    /// Zero.
    pub reserved_2c: u32,
}

/// Where a `DECODE_H265` op's parameter sets start: right after its body.
pub const RGFX_VIDEO_DECODE_H265_PARAMETER_SETS_OFFSET: u32 = 48;

// The video ops' layout.
const _: () = {
    use core::mem::size_of;
    assert!(size_of::<RgfxVideoOpDecodeH264>() == 48);
    assert!(
        RGFX_VIDEO_DECODE_H264_PARAMETER_SETS_OFFSET as usize == size_of::<RgfxVideoOpDecodeH264>()
    );
    assert!(super::op_group(RGFX_OP_VIDEO_DECODE_H264) == super::RGFX_OP_GROUP_VIDEO);
    assert!(size_of::<RgfxVideoOpDecodeH265>() == 48);
    assert!(
        RGFX_VIDEO_DECODE_H265_PARAMETER_SETS_OFFSET as usize == size_of::<RgfxVideoOpDecodeH265>()
    );
    assert!(super::op_group(RGFX_OP_VIDEO_DECODE_H265) == super::RGFX_OP_GROUP_VIDEO);
};
