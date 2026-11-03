//! Encode profiles of H.264 and H.265, and what the device answers for
//! each (`vkGetPhysicalDeviceVideoCapabilitiesKHR`, and each quality
//! level's preferences): the guest's encoder capability answers come from
//! [`h264_encode_capabilities`] and [`h265_encode_capabilities`].

use ash::vk;

use crate::device::VideoDevice;
use crate::error::{Error, Named};
use crate::profile::{P010_FORMAT, PICTURE_FORMAT, h265_level_idc, level_idc};

/// How the driver is to tune its encoding (`VkVideoEncodeTuningModeKHR`),
/// which the profile carries: NVIDIA's driver maps it onto NVENC's tuning.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Tuning {
    /// The driver's own (no `VkVideoEncodeUsageInfoKHR` in the profile).
    #[default]
    Default,
    /// For quality: offline encoding, a file's.
    HighQuality,
    /// For latency: a call, a stream, screen sharing.
    LowLatency,
    /// For the least latency.
    UltraLowLatency,
}

impl Tuning {
    fn mode(self) -> vk::VideoEncodeTuningModeKHR {
        match self {
            Self::Default => vk::VideoEncodeTuningModeKHR::DEFAULT,
            Self::HighQuality => vk::VideoEncodeTuningModeKHR::HIGH_QUALITY,
            Self::LowLatency => vk::VideoEncodeTuningModeKHR::LOW_LATENCY,
            Self::UltraLowLatency => vk::VideoEncodeTuningModeKHR::ULTRA_LOW_LATENCY,
        }
    }
}

/// The codec an encode profile is of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum EncodeCodec {
    /// H.264.
    H264,
    /// H.265.
    H265,
}

/// An encode profile: one codec profile, 4:2:0 at its bit depth, with a
/// tuning. Sessions, DPB pictures, feedback queries and the sources and
/// bitstream buffers made for one profile ([`crate::SourceSpec`],
/// [`crate::BitstreamSpec`]) name it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct EncodeProfile {
    /// The codec.
    pub codec: EncodeCodec,
    /// H.264's `profile_idc` (66 Constrained Baseline, 77 Main, 100 High),
    /// or H.265's `general_profile_idc` (1 Main, 2 Main 10).
    pub profile_idc: u8,
    /// The tuning.
    pub tuning: Tuning,
}

impl EncodeProfile {
    /// H.264 Constrained Baseline: `profile_idc` 66 with
    /// `constraint_set0_flag` and `constraint_set1_flag`, CAVLC, no B
    /// slices, so that Baseline and Main decoders take it.
    pub const H264_BASELINE: Self = Self::h264(66);
    /// H.264 Main.
    pub const H264_MAIN: Self = Self::h264(77);
    /// H.264 High, with the 8x8 transform where the device writes it.
    pub const H264_HIGH: Self = Self::h264(100);
    /// H.265 Main: 8-bit 4:2:0 from NV12.
    pub const H265_MAIN: Self = Self::h265(1);
    /// H.265 Main 10: 10-bit 4:2:0 from P010.
    pub const H265_MAIN_10: Self = Self::h265(2);

    const fn h264(profile_idc: u8) -> Self {
        Self {
            codec: EncodeCodec::H264,
            profile_idc,
            tuning: Tuning::Default,
        }
    }

    const fn h265(profile_idc: u8) -> Self {
        Self {
            codec: EncodeCodec::H265,
            profile_idc,
            tuning: Tuning::Default,
        }
    }

    /// The same profile with `tuning`.
    pub fn with_tuning(self, tuning: Tuning) -> Self {
        Self { tuning, ..self }
    }

    /// Bits a sample: 10 for H.265 Main 10, else 8.
    pub fn bit_depth(&self) -> u8 {
        if self.codec == EncodeCodec::H265 && self.profile_idc == 2 {
            10
        } else {
            8
        }
    }

    /// The format sources and DPB pictures have: NV12, or P010 at 10 bits.
    pub fn format(&self) -> vk::Format {
        if self.bit_depth() > 8 {
            P010_FORMAT
        } else {
            PICTURE_FORMAT
        }
    }

    /// The bytes of one sample: 1 in NV12, 2 in P010.
    pub fn sample_bytes(&self) -> u32 {
        if self.bit_depth() > 8 { 2 } else { 1 }
    }

    /// Whether the profile has B slices: H.264 Constrained Baseline has
    /// none.
    pub fn has_b_slices(&self) -> bool {
        !(self.codec == EncodeCodec::H264 && self.profile_idc == 66)
    }

    /// The codec operation.
    pub fn operation(&self) -> vk::VideoCodecOperationFlagsKHR {
        match self.codec {
            EncodeCodec::H264 => vk::VideoCodecOperationFlagsKHR::ENCODE_H264,
            EncodeCodec::H265 => vk::VideoCodecOperationFlagsKHR::ENCODE_H265,
        }
    }

    /// Calls `f` with the profile's `VkVideoProfileInfoKHR`, its codec's
    /// part chained, and its usage where the tuning is not the default.
    pub fn with_info<R>(&self, f: impl FnOnce(&vk::VideoProfileInfoKHR<'_>) -> R) -> R {
        let depth = if self.bit_depth() > 8 {
            vk::VideoComponentBitDepthFlagsKHR::TYPE_10
        } else {
            vk::VideoComponentBitDepthFlagsKHR::TYPE_8
        };
        let info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(self.operation())
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(depth)
            .chroma_bit_depth(depth);
        let mut usage = vk::VideoEncodeUsageInfoKHR::default().tuning_mode(self.tuning.mode());
        let info = if self.tuning == Tuning::Default {
            info
        } else {
            info.push_next(&mut usage)
        };
        match self.codec {
            EncodeCodec::H264 => {
                let mut h264 = vk::VideoEncodeH264ProfileInfoKHR::default()
                    .std_profile_idc(u32::from(self.profile_idc));
                f(&info.push_next(&mut h264))
            }
            EncodeCodec::H265 => {
                let mut h265 = vk::VideoEncodeH265ProfileInfoKHR::default()
                    .std_profile_idc(u32::from(self.profile_idc));
                f(&info.push_next(&mut h265))
            }
        }
    }

    /// Calls `f` with a `VkVideoProfileListInfoKHR` of the profile alone,
    /// to chain into an image's or a buffer's create info.
    pub fn with_list<R>(&self, f: impl FnOnce(&mut vk::VideoProfileListInfoKHR<'_>) -> R) -> R {
        self.with_info(|info| {
            let profiles = [*info];
            let mut list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
            f(&mut list)
        })
    }
}

/// What the device prefers at one quality level
/// (`vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR`).
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct QualityPreference {
    /// The rate control mode it prefers.
    pub rate_control_mode: vk::VideoEncodeRateControlModeFlagsKHR,
    /// Frames from one I picture to the next.
    pub gop_frame_count: u32,
    /// Frames from one IDR picture to the next.
    pub idr_period: u32,
    /// Consecutive B pictures.
    pub consecutive_b_frames: u32,
    /// The constant QPs of I, P and B pictures.
    pub constant_qp: [i32; 3],
    /// References a P picture predicts from.
    pub max_l0: u32,
    /// References list 1 of a B picture holds.
    pub max_l1: u32,
}

/// What the device answers for one encode profile.
#[derive(Clone)]
pub struct EncodeCapabilities {
    /// The profile.
    pub profile: EncodeProfile,
    /// The highest level it encodes, as a stream codes it: H.264's
    /// `level_idc` (52 for 5.2), H.265's `general_level_idc` (186 for
    /// 6.2).
    pub max_level_idc: u8,
    /// The smallest coded size a session takes.
    pub min_coded_extent: vk::Extent2D,
    /// The largest.
    pub max_coded_extent: vk::Extent2D,
    /// `pictureAccessGranularity`: DPB pictures are made at the coded
    /// size rounded up to it.
    pub picture_access_granularity: vk::Extent2D,
    /// `encodeInputPictureGranularity`: sources are read at the coded
    /// size rounded up to it.
    pub input_granularity: vk::Extent2D,
    /// DPB slots a session may have.
    pub max_dpb_slots: u32,
    /// References a picture may have.
    pub max_active_reference_pictures: u32,
    /// `minBitstreamBufferOffsetAlignment`.
    pub min_bitstream_buffer_offset_alignment: u64,
    /// `minBitstreamBufferSizeAlignment`.
    pub min_bitstream_buffer_size_alignment: u64,
    /// The codec's encode std header version the driver takes.
    pub std_header_version: vk::ExtensionProperties,
    /// `VkVideoEncodeCapabilitiesKHR::flags`.
    pub flags: vk::VideoEncodeCapabilityFlagsKHR,
    /// The rate control modes it takes (`DISABLED` is a constant QP).
    pub rate_control_modes: vk::VideoEncodeRateControlModeFlagsKHR,
    /// Rate control layers.
    pub max_rate_control_layers: u32,
    /// The highest bitrate, bits a second.
    pub max_bitrate: u64,
    /// Quality levels: 0 to this, less one.
    pub max_quality_levels: u32,
    /// The feedback a query gives.
    pub feedback_flags: vk::VideoEncodeFeedbackFlagsKHR,
    /// Slices (H.265: slice segments) a picture may have.
    pub max_slices: u32,
    /// References list 0 of a P picture holds.
    pub max_p_l0: u32,
    /// References list 0 of a B picture holds.
    pub max_b_l0: u32,
    /// References list 1 holds: 0 where it encodes no B picture.
    pub max_l1: u32,
    /// The lowest QP.
    pub min_qp: i32,
    /// The highest QP.
    pub max_qp: i32,
    /// Whether each encode must say how many pictures of each type are
    /// left in the GOP.
    pub requires_gop_remaining_frames: bool,
    /// The codec's capability flags (`VkVideoEncodeH264CapabilityFlagsKHR`
    /// or `VkVideoEncodeH265CapabilityFlagsKHR`), raw.
    pub codec_flags: u32,
    /// The std syntax it writes as asked (`VkVideoEncodeH264StdFlagsKHR`
    /// or `VkVideoEncodeH265StdFlagsKHR`), raw.
    pub std_syntax_flags: u32,
    /// H.265's CTB sizes (`VkVideoEncodeH265CtbSizeFlagsKHR`); 0 for H.264.
    pub ctb_sizes: u32,
    /// H.265's transform block sizes
    /// (`VkVideoEncodeH265TransformBlockSizeFlagsKHR`); 0 for H.264.
    pub transform_block_sizes: u32,
    /// What it prefers at each quality level, from 0.
    pub quality_levels: Vec<QualityPreference>,
}

impl EncodeCapabilities {
    /// Whether it encodes B pictures: references in list 1, and the
    /// profile has B slices.
    pub fn encodes_b_frames(&self) -> bool {
        self.max_l1 > 0 && self.max_b_l0 > 0 && self.profile.has_b_slices()
    }

    /// Whether it takes `mode`.
    pub fn takes(&self, mode: vk::VideoEncodeRateControlModeFlagsKHR) -> bool {
        self.rate_control_modes.contains(mode)
    }

    /// Whether the std syntax flag `flag` is among those it writes as
    /// asked.
    pub fn writes(&self, flag: u32) -> bool {
        self.std_syntax_flags & flag == flag
    }

    /// Whether a session of these takes pictures of `width` by `height`,
    /// or why not.
    pub(crate) fn check_extent(&self, width: u32, height: u32) -> Result<(), Error> {
        let (min, max) = (self.min_coded_extent, self.max_coded_extent);
        if width > max.width || height > max.height || width < min.width || height < min.height {
            return Err(Error::Unsupported(format!(
                "{:?} encode at {width}x{height}: the device encodes {}x{} to {}x{}",
                self.profile.codec, min.width, min.height, max.width, max.height
            )));
        }
        Ok(())
    }
}

/// What the device answers for each of Constrained Baseline, Main and
/// High, in that order, for encodes of NV12 sources: a profile it does not
/// encode, or whose query fails, is left out. Empty where the device has
/// no encode queue of H.264 ([`VideoDevice::encode_codecs`]).
pub fn h264_encode_capabilities(device: &VideoDevice) -> Vec<EncodeCapabilities> {
    of_profiles(
        device,
        &[
            EncodeProfile::H264_BASELINE,
            EncodeProfile::H264_MAIN,
            EncodeProfile::H264_HIGH,
        ],
    )
}

/// What the device answers for Main and Main 10, in that order, for
/// encodes of NV12 and P010 sources: as [`h264_encode_capabilities`].
pub fn h265_encode_capabilities(device: &VideoDevice) -> Vec<EncodeCapabilities> {
    of_profiles(
        device,
        &[EncodeProfile::H265_MAIN, EncodeProfile::H265_MAIN_10],
    )
}

fn of_profiles(device: &VideoDevice, profiles: &[EncodeProfile]) -> Vec<EncodeCapabilities> {
    profiles
        .iter()
        .filter(|p| device.encode_codecs().contains(p.operation()))
        .filter_map(|&profile| match encode_capabilities(device, profile) {
            Ok(caps) => caps,
            Err(error) => {
                log::debug!("{profile:?}: {error}");
                None
            }
        })
        .collect()
}

/// `profile`'s capabilities, `None` where the device encodes no source of
/// its format, or keeps no DPB picture of it.
pub(crate) fn encode_capabilities(
    device: &VideoDevice,
    profile: EncodeProfile,
) -> Result<Option<EncodeCapabilities>, Error> {
    let mut h264 = vk::VideoEncodeH264CapabilitiesKHR::default();
    let mut h265 = vk::VideoEncodeH265CapabilitiesKHR::default();
    let mut encode = vk::VideoEncodeCapabilitiesKHR::default();
    let caps = vk::VideoCapabilitiesKHR::default().push_next(&mut encode);
    let mut caps = match profile.codec {
        EncodeCodec::H264 => caps.push_next(&mut h264),
        EncodeCodec::H265 => caps.push_next(&mut h265),
    };
    let result = profile.with_info(|info| {
        // SAFETY: the physical device is the instance's; the profile and
        // the output chain live through the call.
        unsafe {
            (device
                .video_instance()
                .fp()
                .get_physical_device_video_capabilities_khr)(
                device.physical_device(),
                info,
                &mut caps,
            )
        }
    });
    if result == vk::Result::ERROR_VIDEO_PROFILE_OPERATION_NOT_SUPPORTED_KHR
        || result == vk::Result::ERROR_VIDEO_PROFILE_FORMAT_NOT_SUPPORTED_KHR
        || result == vk::Result::ERROR_VIDEO_PICTURE_LAYOUT_NOT_SUPPORTED_KHR
        || result == vk::Result::ERROR_VIDEO_PROFILE_CODEC_NOT_SUPPORTED_KHR
    {
        return Ok(None);
    }
    result.named("vkGetPhysicalDeviceVideoCapabilitiesKHR")?;
    let (min_coded_extent, max_coded_extent) = (caps.min_coded_extent, caps.max_coded_extent);
    let picture_access_granularity = caps.picture_access_granularity;
    let (max_dpb_slots, max_active_reference_pictures) =
        (caps.max_dpb_slots, caps.max_active_reference_pictures);
    let offset_alignment = caps.min_bitstream_buffer_offset_alignment;
    let size_alignment = caps.min_bitstream_buffer_size_alignment;
    let std_header_version = caps.std_header_version;
    let mut out = EncodeCapabilities {
        profile,
        max_level_idc: 0,
        min_coded_extent,
        max_coded_extent,
        picture_access_granularity,
        input_granularity: encode.encode_input_picture_granularity,
        max_dpb_slots,
        max_active_reference_pictures,
        min_bitstream_buffer_offset_alignment: offset_alignment,
        min_bitstream_buffer_size_alignment: size_alignment,
        std_header_version,
        flags: encode.flags,
        rate_control_modes: encode.rate_control_modes,
        max_rate_control_layers: encode.max_rate_control_layers,
        max_bitrate: encode.max_bitrate,
        max_quality_levels: encode.max_quality_levels,
        feedback_flags: encode.supported_encode_feedback_flags,
        max_slices: 0,
        max_p_l0: 0,
        max_b_l0: 0,
        max_l1: 0,
        min_qp: 0,
        max_qp: 0,
        requires_gop_remaining_frames: false,
        codec_flags: 0,
        std_syntax_flags: 0,
        ctb_sizes: 0,
        transform_block_sizes: 0,
        quality_levels: Vec::new(),
    };
    match profile.codec {
        EncodeCodec::H264 => {
            out.max_level_idc = level_idc(h264.max_level_idc);
            out.max_slices = h264.max_slice_count;
            out.max_p_l0 = h264.max_p_picture_l0_reference_count;
            out.max_b_l0 = h264.max_b_picture_l0_reference_count;
            out.max_l1 = h264.max_l1_reference_count;
            out.min_qp = h264.min_qp;
            out.max_qp = h264.max_qp;
            out.requires_gop_remaining_frames = h264.requires_gop_remaining_frames == vk::TRUE;
            out.codec_flags = h264.flags.as_raw();
            out.std_syntax_flags = h264.std_syntax_flags.as_raw();
        }
        EncodeCodec::H265 => {
            out.max_level_idc = h265_level_idc(h265.max_level_idc);
            out.max_slices = h265.max_slice_segment_count;
            out.max_p_l0 = h265.max_p_picture_l0_reference_count;
            out.max_b_l0 = h265.max_b_picture_l0_reference_count;
            out.max_l1 = h265.max_l1_reference_count;
            out.min_qp = h265.min_qp;
            out.max_qp = h265.max_qp;
            out.requires_gop_remaining_frames = h265.requires_gop_remaining_frames == vk::TRUE;
            out.codec_flags = h265.flags.as_raw();
            out.std_syntax_flags = h265.std_syntax_flags.as_raw();
            out.ctb_sizes = h265.ctb_sizes.as_raw();
            out.transform_block_sizes = h265.transform_block_sizes.as_raw();
        }
    }
    for level in 0..out.max_quality_levels {
        out.quality_levels
            .push(quality_preference(device, profile, level)?);
    }
    let format = profile.format();
    let offered = |usage| {
        Ok::<_, Error>(
            profile
                .with_list(|list| crate::profile::video_formats_in(device, list, usage))?
                .iter()
                .any(|f| {
                    f.format == format
                        && f.image_tiling == vk::ImageTiling::OPTIMAL
                        && f.image_usage_flags.contains(usage)
                }),
        )
    };
    if !offered(vk::ImageUsageFlags::VIDEO_ENCODE_SRC_KHR)?
        || !offered(vk::ImageUsageFlags::VIDEO_ENCODE_DPB_KHR)?
    {
        return Ok(None);
    }
    Ok(Some(out))
}

/// What the device prefers for `profile` at quality level `level`.
fn quality_preference(
    device: &VideoDevice,
    profile: EncodeProfile,
    level: u32,
) -> Result<QualityPreference, Error> {
    let mut h264 = vk::VideoEncodeH264QualityLevelPropertiesKHR::default();
    let mut h265 = vk::VideoEncodeH265QualityLevelPropertiesKHR::default();
    let properties = vk::VideoEncodeQualityLevelPropertiesKHR::default();
    let mut properties = match profile.codec {
        EncodeCodec::H264 => properties.push_next(&mut h264),
        EncodeCodec::H265 => properties.push_next(&mut h265),
    };
    profile.with_info(|info| {
        let query = vk::PhysicalDeviceVideoEncodeQualityLevelInfoKHR::default()
            .video_profile(info)
            .quality_level(level);
        device.encode_quality_level(&query, &mut properties)
    })?;
    let mode = properties.preferred_rate_control_mode;
    Ok(match profile.codec {
        EncodeCodec::H264 => QualityPreference {
            rate_control_mode: mode,
            gop_frame_count: h264.preferred_gop_frame_count,
            idr_period: h264.preferred_idr_period,
            consecutive_b_frames: h264.preferred_consecutive_b_frame_count,
            constant_qp: [
                h264.preferred_constant_qp.qp_i,
                h264.preferred_constant_qp.qp_p,
                h264.preferred_constant_qp.qp_b,
            ],
            max_l0: h264.preferred_max_l0_reference_count,
            max_l1: h264.preferred_max_l1_reference_count,
        },
        EncodeCodec::H265 => QualityPreference {
            rate_control_mode: mode,
            gop_frame_count: h265.preferred_gop_frame_count,
            idr_period: h265.preferred_idr_period,
            consecutive_b_frames: h265.preferred_consecutive_b_frame_count,
            constant_qp: [
                h265.preferred_constant_qp.qp_i,
                h265.preferred_constant_qp.qp_p,
                h265.preferred_constant_qp.qp_b,
            ],
            max_l0: h265.preferred_max_l0_reference_count,
            max_l1: h265.preferred_max_l1_reference_count,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_name_their_codec_depth_and_format() {
        assert_eq!(EncodeProfile::H265_MAIN_10.bit_depth(), 10);
        assert_eq!(
            EncodeProfile::H265_MAIN_10.format().as_raw(),
            P010_FORMAT.as_raw()
        );
        assert_eq!(EncodeProfile::H264_HIGH.sample_bytes(), 1);
        assert!(!EncodeProfile::H264_BASELINE.has_b_slices());
        assert!(EncodeProfile::H264_MAIN.has_b_slices());
        assert!(EncodeProfile::H265_MAIN.has_b_slices());
    }

    #[test]
    fn a_profile_chains_its_usage_only_when_tuned() {
        EncodeProfile::H264_MAIN.with_info(|info| {
            // SAFETY: the codec part `with_info` chained, alive in the call.
            let h264 = unsafe { &*info.p_next.cast::<vk::VideoEncodeH264ProfileInfoKHR<'_>>() };
            assert_eq!(
                h264.s_type.as_raw(),
                vk::StructureType::VIDEO_ENCODE_H264_PROFILE_INFO_KHR.as_raw()
            );
            assert_eq!(h264.std_profile_idc, 77);
            assert!(h264.p_next.is_null());
        });
        let tuned = EncodeProfile::H265_MAIN_10.with_tuning(Tuning::LowLatency);
        tuned.with_info(|info| {
            assert_eq!(
                info.luma_bit_depth.as_raw(),
                vk::VideoComponentBitDepthFlagsKHR::TYPE_10.as_raw()
            );
            // SAFETY: as above: the codec part, then the usage.
            let h265 = unsafe { &*info.p_next.cast::<vk::VideoEncodeH265ProfileInfoKHR<'_>>() };
            assert_eq!(h265.std_profile_idc, 2);
            // SAFETY: as above.
            let usage = unsafe { &*h265.p_next.cast::<vk::VideoEncodeUsageInfoKHR<'_>>() };
            assert_eq!(
                usage.tuning_mode.as_raw(),
                vk::VideoEncodeTuningModeKHR::LOW_LATENCY.as_raw()
            );
        });
    }
}
