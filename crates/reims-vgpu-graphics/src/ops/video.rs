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
    /// The luma rows the output surface holds from the luma texture's first, whole rows of its
    /// bytes per row (the texture's height, then the rows its surface extends it by below,
    /// before plane 1), and half as many chroma rows from the chroma texture's: where the
    /// codec's [`RgfxVideoOutputLayout`](crate::wire::RgfxVideoOutputLayout) is not zero, and the
    /// textures are their planes' whole, the host may write the picture's whole coded rows
    /// there. 0 where the textures are the picture's crop alone (a tile of a larger surface, or a
    /// guest older than the field).
    pub output_rows: u32,
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
    /// As [`RgfxVideoOpDecodeH264::output_rows`].
    pub output_rows: u32,
}

/// Where a `DECODE_H265` op's parameter sets start: right after its body.
pub const RGFX_VIDEO_DECODE_H265_PARAMETER_SETS_OFFSET: u32 = 48;

/// Op `ENCODE_H264` ([`RgfxVideoOpEncode`], then an [`RgfxVideoEncodeSettings`] where the
/// settings changed): encode one frame as one H.264 access unit, on a session made in the
/// encode direction
/// ([`RGFX_VIDEO_ENCODE`](crate::wire::RGFX_VIDEO_ENCODE)).
///
/// The frame is the IOSurface of two planes `source_luma_texture_obj_id` (plane 0: `R8Unorm`, or
/// `R16Unorm` at 10 bits) and `source_chroma_texture_obj_id` (plane 1: `RG8Unorm`, or
/// `RG16Unorm`) lie over (`TEXTURE_FROM_IOSURFACE`), at their origin, the settings' width by
/// height. The host reads it from the surface's pages, or from its device cache where its block
/// has one, as a decode's picture lands; the command buffer names the surface it reads with
/// `SURFACE_USE`, as any GPU read does.
///
/// The host writes the access unit, in Annex-B form (each NAL unit after a start code), at
/// `bitstream_offset` in the Shared buffer `bitstream_buffer_obj_id`, at most
/// `bitstream_capacity` bytes, and then the frame's [`RgfxVideoEncodeResult`] at
/// `result_offset` in the Shared buffer `result_buffer_obj_id`, which may be the same buffer;
/// both before the command buffer completes. A keyframe's access unit carries the parameter sets
/// before its slices (SPS and PPS), and its result says so
/// ([`RGFX_VIDEO_ENCODED_PARAMETER_SETS`]).
///
/// An access unit past the capacity is not written: the result's status is
/// [`RGFX_VIDEO_ENCODE_OVERFLOW`], with the bytes it needed, and the session's next access unit is
/// a keyframe, since the frame is already among the encoder's references. No access unit is
/// longer than `width * height * bytes_per_sample * 3 / 2 + 1 MiB`, so a capacity of that never
/// overflows. A frame the hardware fails gets [`RGFX_VIDEO_ENCODE_FAILED`]. Both complete the
/// command buffer as any other. A reference to no such object, a source of another size than the
/// settings', an odd width or height, a size past the caps, a capacity of 0, a result not at a
/// multiple of 8 or past its buffer, or a first encode with no settings faults the command buffer.
pub const RGFX_OP_VIDEO_ENCODE_H264: u16 = 0x4002;

/// Op `ENCODE_H265` ([`RgfxVideoOpEncode`], then an [`RgfxVideoEncodeSettings`] where the
/// settings changed): encode one frame as one HEVC access unit, as [`RGFX_OP_VIDEO_ENCODE_H264`]
/// does, its keyframes' parameter sets the VPS, SPS and PPS; a 10-bit profile's source is P010's
/// planes.
pub const RGFX_OP_VIDEO_ENCODE_H265: u16 = 0x4003;

/// [`RgfxVideoOpEncode::flags`]: the frame is a keyframe (an IDR picture, or HEVC's IRAP).
pub const RGFX_VIDEO_ENCODE_FORCE_KEYFRAME: u32 = 1 << 0;

/// The body of [`RGFX_OP_VIDEO_ENCODE_H264`] and [`RGFX_OP_VIDEO_ENCODE_H265`], followed by
/// `settings_size` bytes of [`RgfxVideoEncodeSettings`] where the settings changed since the
/// session's last encode, its first included.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVideoOpEncode {
    /// The texture over the source surface's plane 0.
    pub source_luma_texture_obj_id: u64,
    /// The texture over the source surface's plane 1.
    pub source_chroma_texture_obj_id: u64,
    /// The Shared buffer the host writes the access unit into.
    pub bitstream_buffer_obj_id: u64,
    /// Where the access unit's first byte goes in it.
    pub bitstream_offset: u64,
    /// The most bytes the host may write there.
    pub bitstream_capacity: u64,
    /// The Shared buffer the host writes the [`RgfxVideoEncodeResult`] into; it may be the
    /// bitstream's.
    pub result_buffer_obj_id: u64,
    /// Where the result goes in it: a multiple of 8.
    pub result_offset: u64,
    /// The frame's place in display order, from 0; the result says which frame its access unit
    /// holds.
    pub frame_index: u64,
    /// `RGFX_VIDEO_ENCODE_*`: [`RGFX_VIDEO_ENCODE_FORCE_KEYFRAME`].
    pub flags: u32,
    /// The bytes of [`RgfxVideoEncodeSettings`] after this body; 0 where they are unchanged.
    pub settings_size: u32,
}

/// [`RgfxVideoEncodeSettings::rate_control`]: a constant QP a picture type (`qp_i`, `qp_p`,
/// `qp_b`).
pub const RGFX_VIDEO_RATE_CQP: u32 = 1;
/// [`RgfxVideoEncodeSettings::rate_control`]: a constant bitrate, `average_bitrate`.
pub const RGFX_VIDEO_RATE_CBR: u32 = 2;
/// [`RgfxVideoEncodeSettings::rate_control`]: a variable bitrate around `average_bitrate`,
/// under `max_bitrate`.
pub const RGFX_VIDEO_RATE_VBR: u32 = 3;

/// [`RgfxVideoEncodeSettings::flags`]: `max_bitrate` over `max_bitrate_window_ms` is a hard
/// limit (VideoToolbox's `DataRateLimits`), not the rate control's peak alone.
pub const RGFX_VIDEO_ENCODE_HARD_LIMIT: u32 = 1 << 0;

/// [`RgfxVideoEncodeSettings::entropy`]: H.264's CAVLC.
pub const RGFX_VIDEO_ENTROPY_CAVLC: u32 = 1;
/// [`RgfxVideoEncodeSettings::entropy`]: H.264's CABAC, which Constrained Baseline does not have.
pub const RGFX_VIDEO_ENTROPY_CABAC: u32 = 2;

/// An encode session's settings, after an [`RgfxVideoOpEncode`] whose `settings_size` says so.
/// A change of the size, format, profile, level, entropy, B-frames, keyframe interval or colour
/// starts a new sequence: the next access unit is a keyframe with its parameter sets. The
/// bitrates, QPs, quality and frame rate apply from the next frame, with no keyframe.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVideoEncodeSettings {
    /// The coded picture's width, the sources' size: even.
    pub width: u32,
    /// The coded picture's height: even.
    pub height: u32,
    /// The sources' format: [`RGFX_VIDEO_FORMAT_NV12`](crate::wire::RGFX_VIDEO_FORMAT_NV12) or
    /// [`RGFX_VIDEO_FORMAT_P010`](crate::wire::RGFX_VIDEO_FORMAT_P010).
    pub format: u32,
    /// One `RGFX_VIDEO_<codec>_*` profile bit.
    pub profile: u32,
    /// The codec's level as it numbers it (H.264's `level_idc`, HEVC's `general_level_idc`); 0
    /// for the host's pick, the lowest that fits.
    pub level: u32,
    /// `RGFX_VIDEO_RATE_*`.
    pub rate_control: u32,
    /// Bits a second, for CBR and VBR; the host saturates it to its API's range.
    pub average_bitrate: u64,
    /// VBR's peak, in bits a second over `max_bitrate_window_ms`; 0 for none.
    pub max_bitrate: u64,
    /// The window of `max_bitrate`, and the HRD buffer's length, in milliseconds; 0 for a
    /// second.
    pub max_bitrate_window_ms: u32,
    /// The expected frame rate's numerator; 0, with a 0 denominator, where it is unknown.
    pub frame_rate_num: u32,
    /// The expected frame rate's denominator.
    pub frame_rate_den: u32,
    /// The most frames from one keyframe to the next; 0 for the host's pick.
    pub key_frame_interval: u32,
    /// CQP's QP of I pictures.
    pub qp_i: u32,
    /// CQP's QP of P pictures.
    pub qp_p: u32,
    /// CQP's QP of B pictures.
    pub qp_b: u32,
    /// The least QP rate control picks; 0 for none.
    pub min_qp: u32,
    /// The greatest QP rate control picks; 0 for none.
    pub max_qp: u32,
    /// Consecutive B pictures, at most the caps' `max_b_frames`.
    pub b_frames: u32,
    /// H.264's entropy coder: 0 for the profile's (CABAC for Main and High, CAVLC for
    /// Baseline), or `RGFX_VIDEO_ENTROPY_*`; 0 for HEVC.
    pub entropy: u32,
    /// The speed and quality trade, 0 the fastest to 100 the best, which the host maps onto its
    /// API's presets.
    pub quality: u32,
    /// `RGFX_VIDEO_ENCODE_*` settings bits: [`RGFX_VIDEO_ENCODE_HARD_LIMIT`].
    pub flags: u32,
    /// The VUI's `colour_primaries` (ITU-T H.273), in the SPS the host packs.
    pub color_primaries: u8,
    /// The VUI's `transfer_characteristics`.
    pub transfer: u8,
    /// The VUI's `matrix_coefficients`.
    pub matrix: u8,
    /// The VUI's `video_full_range_flag`: 1 for full range.
    pub full_range: u8,
}

/// [`RgfxVideoEncodeResult::status`]: the access unit was written.
pub const RGFX_VIDEO_ENCODE_OK: u32 = 0;
/// [`RgfxVideoEncodeResult::status`]: the access unit did not fit the capacity, and was not
/// written; `bitstream_size` is the bytes it needed.
pub const RGFX_VIDEO_ENCODE_OVERFLOW: u32 = 1;
/// [`RgfxVideoEncodeResult::status`]: the hardware did not encode the frame.
pub const RGFX_VIDEO_ENCODE_FAILED: u32 = 2;

/// [`RgfxVideoEncodeResult::flags`]: the access unit is a keyframe (an IDR picture, or HEVC's
/// IRAP).
pub const RGFX_VIDEO_ENCODED_KEYFRAME: u32 = 1 << 0;
/// [`RgfxVideoEncodeResult::flags`]: the access unit carries the parameter sets before its
/// slices.
pub const RGFX_VIDEO_ENCODED_PARAMETER_SETS: u32 = 1 << 1;

/// [`RgfxVideoEncodeResult::picture_type`]: an I picture.
pub const RGFX_VIDEO_PICTURE_I: u32 = 1;
/// [`RgfxVideoEncodeResult::picture_type`]: a P picture.
pub const RGFX_VIDEO_PICTURE_P: u32 = 2;
/// [`RgfxVideoEncodeResult::picture_type`]: a B picture.
pub const RGFX_VIDEO_PICTURE_B: u32 = 3;

/// What became of one encode, which the host writes at the op's `result_offset` before its
/// command buffer completes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVideoEncodeResult {
    /// `RGFX_VIDEO_ENCODE_OK`, `_OVERFLOW` or `_FAILED`.
    pub status: u32,
    /// `RGFX_VIDEO_ENCODED_*`.
    pub flags: u32,
    /// The bytes written at `bitstream_offset`; for an overflow, the bytes the access unit
    /// needed.
    pub bitstream_size: u64,
    /// The frame the access unit holds, by its op's `frame_index`: the op's own while the
    /// session has no B-frames.
    pub frame_index: u64,
    /// `RGFX_VIDEO_PICTURE_*`; 0 for a frame not encoded.
    pub picture_type: u32,
    /// Zero.
    pub reserved_1c: u32,
}

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
    assert!(size_of::<RgfxVideoOpEncode>() == 72);
    assert!(core::mem::offset_of!(RgfxVideoOpEncode, flags) == 0x40);
    assert!(super::op_group(RGFX_OP_VIDEO_ENCODE_H264) == super::RGFX_OP_GROUP_VIDEO);
    assert!(super::op_group(RGFX_OP_VIDEO_ENCODE_H265) == super::RGFX_OP_GROUP_VIDEO);
    assert!(size_of::<RgfxVideoEncodeSettings>() == 96);
    assert!(core::mem::offset_of!(RgfxVideoEncodeSettings, average_bitrate) == 0x18);
    assert!(core::mem::offset_of!(RgfxVideoEncodeSettings, max_bitrate_window_ms) == 0x28);
    assert!(core::mem::offset_of!(RgfxVideoEncodeSettings, color_primaries) == 0x5c);
    assert!(size_of::<RgfxVideoEncodeResult>() == 32);
    assert!(core::mem::offset_of!(RgfxVideoEncodeResult, picture_type) == 0x18);
};
