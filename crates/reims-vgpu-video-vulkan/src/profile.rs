//! H.264 decode profiles, and what the device answers for each
//! (`vkGetPhysicalDeviceVideoCapabilitiesKHR`): the guest's capability
//! answers come from [`h264_capabilities`].

use ash::vk;
use ash::vk::native::StdVideoH264LevelIdc;

use crate::device::VideoDevice;
use crate::error::{Error, Named};

/// The format of every picture: NV12, 8-bit 4:2:0 in two planes.
pub const PICTURE_FORMAT: vk::Format = vk::Format::G8_B8R8_2PLANE_420_UNORM;

/// The usage every output has: Metal samples it per plane through views of
/// other formats, a picture decoded into a DPB picture is copied into it,
/// and tests copy it out. An output a decode writes has
/// `VIDEO_DECODE_DST` too ([`crate::SurfaceSpec`]).
pub const OUTPUT_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::SAMPLED.as_raw()
        | vk::ImageUsageFlags::TRANSFER_SRC.as_raw()
        | vk::ImageUsageFlags::TRANSFER_DST.as_raw(),
);

/// The flags every output is made with: its planes are viewed as R8 and
/// RG8, with usages NV12 itself may lack.
pub const OUTPUT_FLAGS: vk::ImageCreateFlags = vk::ImageCreateFlags::from_raw(
    vk::ImageCreateFlags::MUTABLE_FORMAT.as_raw() | vk::ImageCreateFlags::EXTENDED_USAGE.as_raw(),
);

/// An H.264 decode profile: 4:2:0 at 8 bits, progressive frames, of one
/// `profile_idc`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct H264Profile {
    /// `profile_idc`: 66 (Baseline), 77 (Main) or 100 (High).
    pub profile_idc: u8,
}

impl H264Profile {
    /// Baseline.
    pub const BASELINE: Self = Self { profile_idc: 66 };
    /// Main.
    pub const MAIN: Self = Self { profile_idc: 77 };
    /// High.
    pub const HIGH: Self = Self { profile_idc: 100 };

    /// Calls `f` with the profile's `VkVideoProfileInfoKHR`, its H.264
    /// part chained.
    pub fn with_info<R>(&self, f: impl FnOnce(&vk::VideoProfileInfoKHR<'_>) -> R) -> R {
        let mut h264 = vk::VideoDecodeH264ProfileInfoKHR::default()
            .std_profile_idc(u32::from(self.profile_idc))
            .picture_layout(vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE);
        let info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_H264)
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
            .push_next(&mut h264);
        f(&info)
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

    /// The profiles that decode a stream of `profile_idc`, most
    /// constrained first. Baseline streams without FMO (which the codec
    /// crate refuses), ASO or redundant pictures (which it drops) are a
    /// subset of Main, and Main of High.
    pub fn candidates(profile_idc: u8) -> &'static [Self] {
        match profile_idc {
            66 => &[Self::BASELINE, Self::MAIN, Self::HIGH],
            77 => &[Self::MAIN, Self::HIGH],
            _ => &[Self::HIGH],
        }
    }
}

/// What the device answers for one H.264 decode profile.
#[derive(Clone, Copy)]
pub struct H264Capabilities {
    /// The profile.
    pub profile: H264Profile,
    /// The highest `level_idc` it decodes (`maxLevelIdc`), as a stream
    /// codes it: 52 for level 5.2.
    pub max_level_idc: u8,
    /// The smallest coded size a session takes.
    pub min_coded_extent: vk::Extent2D,
    /// The largest.
    pub max_coded_extent: vk::Extent2D,
    /// `pictureAccessGranularity`.
    pub picture_access_granularity: vk::Extent2D,
    /// DPB slots a session may have.
    pub max_dpb_slots: u32,
    /// References a picture may have.
    pub max_active_reference_pictures: u32,
    /// Whether a picture's output may be its DPB picture, in one image.
    pub dpb_and_output_coincide: bool,
    /// Whether it may be another image.
    pub dpb_and_output_distinct: bool,
    /// Whether the decoder decodes each picture straight into its output,
    /// beside its DPB picture (`DPB_AND_OUTPUT_DISTINCT`, where NV12
    /// outputs may be decode targets); else into its DPB picture, then
    /// copies that into the output (`DPB_AND_OUTPUT_COINCIDE`).
    pub decodes_into_output: bool,
    /// `minBitstreamBufferOffsetAlignment`.
    pub min_bitstream_buffer_offset_alignment: u64,
    /// `minBitstreamBufferSizeAlignment`.
    pub min_bitstream_buffer_size_alignment: u64,
    /// The H.264 decode std header version the driver takes.
    pub std_header_version: vk::ExtensionProperties,
}

/// What the device answers for each of Baseline, Main and High, in that
/// order, for decodes of NV12 pictures into NV12 outputs: a profile the
/// device does not decode so, or whose query fails, is left out. Empty
/// where it decodes no H.264.
pub fn h264_capabilities(device: &VideoDevice) -> Vec<H264Capabilities> {
    [H264Profile::BASELINE, H264Profile::MAIN, H264Profile::HIGH]
        .into_iter()
        .filter_map(|profile| match profile_capabilities(device, profile) {
            Ok(caps) => caps,
            Err(error) => {
                log::debug!("H.264 profile {}: {error}", profile.profile_idc);
                None
            }
        })
        .collect()
}

/// `profile`'s capabilities, `None` where it decodes no NV12 picture into
/// an NV12 output, straight or through a copy.
fn profile_capabilities(
    device: &VideoDevice,
    profile: H264Profile,
) -> Result<Option<H264Capabilities>, Error> {
    let mut h264 = vk::VideoDecodeH264CapabilitiesKHR::default();
    let mut decode = vk::VideoDecodeCapabilitiesKHR::default();
    let mut caps = vk::VideoCapabilitiesKHR::default()
        .push_next(&mut decode)
        .push_next(&mut h264);
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
    let caps = H264Capabilities {
        profile,
        max_level_idc: 0,
        min_coded_extent: caps.min_coded_extent,
        max_coded_extent: caps.max_coded_extent,
        picture_access_granularity: caps.picture_access_granularity,
        max_dpb_slots: caps.max_dpb_slots,
        max_active_reference_pictures: caps.max_active_reference_pictures,
        dpb_and_output_coincide: false,
        dpb_and_output_distinct: false,
        decodes_into_output: false,
        min_bitstream_buffer_offset_alignment: caps.min_bitstream_buffer_offset_alignment,
        min_bitstream_buffer_size_alignment: caps.min_bitstream_buffer_size_alignment,
        std_header_version: caps.std_header_version,
    };
    let mut caps = H264Capabilities {
        max_level_idc: level_idc(h264.max_level_idc),
        dpb_and_output_coincide: decode
            .flags
            .contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE),
        dpb_and_output_distinct: decode
            .flags
            .contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_DISTINCT),
        ..caps
    };
    let nv12 = |usage, wanted_usage, wanted_flags| {
        Ok::<_, Error>(video_formats(device, profile, usage)?.iter().any(|f| {
            f.format == PICTURE_FORMAT
                && f.image_tiling == vk::ImageTiling::OPTIMAL
                && f.image_usage_flags.contains(wanted_usage)
                && f.image_create_flags.contains(wanted_flags)
        }))
    };
    let dst = vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR;
    let dpb = vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR;
    if caps.dpb_and_output_distinct
        && nv12(dpb, dpb, vk::ImageCreateFlags::empty())?
        && nv12(dst, dst | OUTPUT_USAGE, OUTPUT_FLAGS)?
    {
        caps.decodes_into_output = true;
        return Ok(Some(caps));
    }
    if caps.dpb_and_output_coincide
        && nv12(
            dst | dpb,
            dst | dpb | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageCreateFlags::empty(),
        )?
        && plain_outputs(device)
    {
        return Ok(Some(caps));
    }
    Ok(None)
}

/// The formats `profile` decodes into, or predicts from, for `usage`.
fn video_formats(
    device: &VideoDevice,
    profile: H264Profile,
    usage: vk::ImageUsageFlags,
) -> Result<Vec<vk::VideoFormatPropertiesKHR<'static>>, Error> {
    let fp = device.video_instance().fp();
    profile.with_list(|list| {
        let info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
            .image_usage(usage)
            .push_next(list);
        let mut count = 0;
        // SAFETY: as in `profile_capabilities`; a null output asks for the
        // count alone.
        let result = unsafe {
            (fp.get_physical_device_video_format_properties_khr)(
                device.physical_device(),
                &info,
                &mut count,
                std::ptr::null_mut(),
            )
        };
        if result == vk::Result::ERROR_IMAGE_USAGE_NOT_SUPPORTED_KHR
            || result == vk::Result::ERROR_FORMAT_NOT_SUPPORTED
        {
            return Ok(Vec::new());
        }
        result.named("vkGetPhysicalDeviceVideoFormatPropertiesKHR")?;
        let mut formats =
            vec![vk::VideoFormatPropertiesKHR::default(); usize::try_from(count).unwrap_or(0)];
        // SAFETY: as above, with an output of `count` elements.
        unsafe {
            (fp.get_physical_device_video_format_properties_khr)(
                device.physical_device(),
                &info,
                &mut count,
                formats.as_mut_ptr(),
            )
        }
        .named("vkGetPhysicalDeviceVideoFormatPropertiesKHR")?;
        formats.truncate(usize::try_from(count).unwrap_or(0));
        Ok(formats)
    })
}

/// Whether NV12 images that no decode writes, copied into, sampled and
/// copied out, with [`OUTPUT_FLAGS`], are the device's.
fn plain_outputs(device: &VideoDevice) -> bool {
    let info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(PICTURE_FORMAT)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(OUTPUT_USAGE)
        .flags(OUTPUT_FLAGS);
    let mut properties = vk::ImageFormatProperties2::default();
    // SAFETY: the physical device is the instance's; `info` and the output
    // live through the call.
    unsafe {
        device
            .instance()
            .get_physical_device_image_format_properties2(
                device.physical_device(),
                &info,
                &mut properties,
            )
    }
    .is_ok()
}

/// `level_idc` as a stream codes it, of a std level: 52 for
/// `STD_VIDEO_H264_LEVEL_IDC_5_2`.
pub(crate) fn level_idc(level: StdVideoH264LevelIdc) -> u8 {
    usize::try_from(level)
        .ok()
        .and_then(|i| LEVELS.get(i))
        .copied()
        .unwrap_or(0)
}

/// The std level of a stream's `level_idc`: level 1b, `level_idc` 9 or 11
/// with `constraint_set3_flag` in Baseline, Main and Extended, which has no
/// std level, is taken as 1.1, the least above it; an unknown
/// `level_idc` as the least std level at or above it, or the highest.
pub(crate) fn std_level(level_idc: u8) -> StdVideoH264LevelIdc {
    let at = LEVELS
        .iter()
        .position(|&l| l >= level_idc)
        .unwrap_or(LEVELS.len() - 1);
    let at = if level_idc == 9 { 1 } else { at };
    StdVideoH264LevelIdc::try_from(at).unwrap_or(0)
}

/// `level_idc` of each std level, by its value
/// (`STD_VIDEO_H264_LEVEL_IDC_1_0` is 0).
const LEVELS: [u8; 19] = [
    10, 11, 12, 13, 20, 21, 22, 30, 31, 32, 40, 41, 42, 50, 51, 52, 60, 61, 62,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_map_both_ways() {
        assert_eq!(
            level_idc(ash::vk::native::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_2),
            52
        );
        assert_eq!(
            level_idc(ash::vk::native::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_0),
            10
        );
        assert_eq!(
            level_idc(ash::vk::native::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_2),
            62
        );
        for &level in &LEVELS {
            assert_eq!(level_idc(std_level(level)), level);
        }
        // Level 1b, and a level between two.
        assert_eq!(level_idc(std_level(9)), 11);
        assert_eq!(level_idc(std_level(33)), 40);
        assert_eq!(level_idc(std_level(255)), 62);
    }

    #[test]
    fn baseline_streams_try_the_wider_profiles_after_their_own() {
        assert_eq!(
            H264Profile::candidates(66),
            [H264Profile::BASELINE, H264Profile::MAIN, H264Profile::HIGH]
        );
        assert_eq!(
            H264Profile::candidates(77),
            [H264Profile::MAIN, H264Profile::HIGH]
        );
        assert_eq!(H264Profile::candidates(100), [H264Profile::HIGH]);
    }

    #[test]
    fn a_profile_list_names_the_profile_once() {
        H264Profile::MAIN.with_list(|list| {
            assert_eq!(list.profile_count, 1);
            // SAFETY: the list points at one profile, alive in the call.
            let profile = unsafe { &*list.p_profiles };
            assert_eq!(
                profile.video_codec_operation.as_raw(),
                vk::VideoCodecOperationFlagsKHR::DECODE_H264.as_raw()
            );
            // SAFETY: its H.264 part, chained by `with_info`.
            let h264 = unsafe {
                &*profile
                    .p_next
                    .cast::<vk::VideoDecodeH264ProfileInfoKHR<'_>>()
            };
            assert_eq!(h264.std_profile_idc, 77);
        });
    }
}
