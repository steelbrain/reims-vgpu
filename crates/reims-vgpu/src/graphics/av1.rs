//! AV1 for an arm64 guest's VideoToolbox: what the arms share of it.
//!
//! An arm decodes AV1 only for VideoToolbox's paravirtual decoders, where its own capability query
//! lists AV1's Main profile (VA-API's `AV1Profile0` decoder, NVDEC's `cudaVideoCodec_AV1`): the
//! decoder list names `av01` with profile 0 at the arm's largest frame. The device's video caps
//! list no AV1, as an x86-64 guest's video driver, AppleGVA, decodes none, so AV1's number among
//! the arm's codecs ([`HOST_VIDEO_CODEC_AV1`]) is the host's own and reaches no guest.
//!
//! Profile 0 alone is listed. A Professional (profile 2) stream is 4:2:0 only at 12 bits, which
//! the guest's output formats (two planes, 10 bits at most) would cut, and High (profile 1) is
//! 4:4:4, which no arm's AV1 decoder lands; either is refused at its session's start.

use reims_vgpu_video::paravirt::{self, Stream};

/// AV1 among the arm's codec numbers (`RGFX_VIDEO_CODEC_*`), as the host's own code numbers it:
/// far from the device's own numbers, and never written to the guest.
pub(crate) const HOST_VIDEO_CODEC_AV1: u32 = 0x8000_0001;

/// AV1's top level, 6.3 (`seq_level_idx` 19), which the decoder list answers, as Apple's own
/// AV1 decoder lists it. **Measured** on an M3 Max, macOS 27.0.1 (26A434), 2026-11-27:
/// `VTCopyAV1DecoderCapabilitiesDictionary` answers `VTMaxDecodeLevel` 19 for profiles 0, 1 and 2.
pub(crate) const AV1_TOP_LEVEL: u32 = 19;

/// What an arm decodes of AV1 for VideoToolbox's paravirtual decoders.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuestAv1 {
    /// The widest frame, as the arm's capability query lists it.
    pub max_width: u32,
    /// The tallest frame.
    pub max_height: u32,
    /// Whether it decodes 10-bit streams, besides 8-bit ones.
    pub ten_bit: bool,
    /// The narrowest frame, as the query lists it: 0 where it lists none.
    pub min_width: u32,
    /// The shortest frame.
    pub min_height: u32,
}

impl GuestAv1 {
    /// The codec module's limits of the arm's AV1 decoder.
    pub(crate) fn limits(self) -> reims_vgpu_video::codec::av1::Limits {
        reims_vgpu_video::codec::av1::Limits {
            max_width: self.max_width,
            max_height: self.max_height,
            min_width: self.min_width,
            min_height: self.min_height,
        }
    }
}

/// The decoder list's entry of AV1: `av01`, profile 0, at [`AV1_TOP_LEVEL`].
pub(crate) fn vt_decoder() -> paravirt::Decoder {
    paravirt::Decoder {
        codec: paravirt::Codec::Av1,
        profiles: vec![0],
        max_level: AV1_TOP_LEVEL,
    }
}

/// Why `name`'s arm, decoding `av1` of AV1, decodes none of `stream`'s frames: a stream the codec
/// takes (Main, 4:2:0) of a depth or a size past the arm's; `None` where it decodes them. A
/// frame the hardware fails, such as NVDEC's of a superres sequence, fails to the guest where
/// it fails (`graphics::nvidia`).
pub(crate) fn declines(name: &str, av1: Option<GuestAv1>, stream: &Stream) -> Option<String> {
    let Some(av1) = av1 else {
        return Some(format!("{name} decodes no AV1"));
    };
    if stream.chroma_format_idc != 1 {
        return Some(format!("{name}'s AV1 decoder lands 4:2:0 frames alone"));
    }
    match stream.luma_bits {
        8 => {}
        10 if av1.ten_bit => {}
        bits => {
            let depths = if av1.ten_bit {
                "8 and 10 bits"
            } else {
                "8 bits"
            };
            return Some(format!("{name} decodes AV1 at {depths} here, not {bits}"));
        }
    }
    if stream.width > av1.max_width || stream.height > av1.max_height {
        return Some(format!(
            "{name} decodes AV1 frames up to {}x{}",
            av1.max_width, av1.max_height
        ));
    }
    if stream.width < av1.min_width || stream.height < av1.min_height {
        return Some(format!(
            "{name} decodes AV1 frames from {}x{}",
            av1.min_width, av1.min_height
        ));
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream(bits: u8, chroma: u8, width: u32, height: u32) -> Stream {
        Stream {
            codec: paravirt::Codec::Av1,
            profile: 0,
            chroma_format_idc: chroma,
            luma_bits: bits,
            chroma_bits: bits,
            width,
            height,
            range_extensions: false,
        }
    }

    const ARM: GuestAv1 = GuestAv1 {
        max_width: 8192,
        max_height: 4352,
        ten_bit: true,
        min_width: 128,
        min_height: 128,
    };

    #[test]
    fn the_list_names_av01_main_at_the_top_level() {
        let entry = vt_decoder();
        assert_eq!(entry.codec.fourcc().to_be_bytes(), *b"av01");
        assert_eq!((entry.profiles, entry.max_level), (vec![0], 19));
    }

    #[test]
    fn a_stream_past_the_arms_depth_size_or_subsampling_is_declined() {
        assert_eq!(declines("A", Some(ARM), &stream(8, 1, 1920, 1080)), None);
        assert_eq!(declines("A", Some(ARM), &stream(10, 1, 8192, 4352)), None);
        assert_eq!(declines("A", Some(ARM), &stream(8, 1, 128, 128)), None);
        let eight_bit = GuestAv1 {
            ten_bit: false,
            ..ARM
        };
        for (av1, stream, why) in [
            (None, stream(8, 1, 64, 64), "A decodes no AV1"),
            (
                Some(eight_bit),
                stream(10, 1, 64, 64),
                "A decodes AV1 at 8 bits here, not 10",
            ),
            (
                Some(ARM),
                stream(12, 1, 64, 64),
                "A decodes AV1 at 8 and 10 bits here, not 12",
            ),
            (
                Some(ARM),
                stream(8, 0, 64, 64),
                "A's AV1 decoder lands 4:2:0 frames alone",
            ),
            (
                Some(ARM),
                stream(8, 1, 8193, 64),
                "A decodes AV1 frames up to 8192x4352",
            ),
            (
                Some(ARM),
                stream(8, 1, 64, 4353),
                "A decodes AV1 frames up to 8192x4352",
            ),
            (
                Some(ARM),
                stream(8, 1, 98, 62),
                "A decodes AV1 frames from 128x128",
            ),
        ] {
            assert_eq!(declines("A", av1, &stream).as_deref(), Some(why));
        }
    }
}
