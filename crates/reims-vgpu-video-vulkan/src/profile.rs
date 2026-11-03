//! Decode profiles of H.264 and H.265, and what the device answers for
//! each (`vkGetPhysicalDeviceVideoCapabilitiesKHR`): the guest's capability
//! answers come from [`h264_capabilities`] and [`h265_capabilities`].

use ash::vk;
use ash::vk::native::{StdVideoH264LevelIdc, StdVideoH265LevelIdc};

use crate::device::VideoDevice;
use crate::error::{Error, Named};

/// The format of every 8-bit picture: NV12, 4:2:0 in two planes.
pub const PICTURE_FORMAT: vk::Format = vk::Format::G8_B8R8_2PLANE_420_UNORM;

/// The format of every 10-bit picture: P010, 4:2:0 in two planes of 16-bit
/// samples, each sample's 10 bits the high ones.
pub const P010_FORMAT: vk::Format = vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16;

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
/// RG8 (R16 and RG16 for P010), with usages the two-plane format itself
/// may lack.
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
        VideoProfile::H264(*self).with_info(f)
    }

    /// Calls `f` with a `VkVideoProfileListInfoKHR` of the profile alone,
    /// to chain into an image's or a buffer's create info.
    pub fn with_list<R>(&self, f: impl FnOnce(&mut vk::VideoProfileListInfoKHR<'_>) -> R) -> R {
        VideoProfile::H264(*self).with_list(f)
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

/// An H.265 decode profile: 4:2:0, Main at 8 bits or Main 10 at 10.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct H265Profile {
    /// `general_profile_idc`: 1 (Main) or 2 (Main 10).
    pub profile_idc: u8,
}

impl H265Profile {
    /// Main: 8 bits, into NV12.
    pub const MAIN: Self = Self { profile_idc: 1 };
    /// Main 10: 10 bits, into P010.
    pub const MAIN_10: Self = Self { profile_idc: 2 };

    /// The bit depth of its luma and chroma samples.
    pub fn bit_depth(&self) -> u8 {
        if *self == Self::MAIN_10 { 10 } else { 8 }
    }

    /// The profile that decodes a stream of `bit_depth` bits (the codec
    /// crate's decoder takes 8 and 10, luma and chroma alike): Main for 8,
    /// whatever profile the stream names, since Main and Main Still
    /// Picture streams are Main's, and a Main 10 stream of 8-bit samples
    /// decodes as one; Main 10 for 10.
    pub fn for_bit_depth(bit_depth: u8) -> Option<Self> {
        match bit_depth {
            8 => Some(Self::MAIN),
            10 => Some(Self::MAIN_10),
            _ => None,
        }
    }

    /// Calls `f` with the profile's `VkVideoProfileInfoKHR`, its H.265
    /// part chained.
    pub fn with_info<R>(&self, f: impl FnOnce(&vk::VideoProfileInfoKHR<'_>) -> R) -> R {
        VideoProfile::H265(*self).with_info(f)
    }

    /// Calls `f` with a `VkVideoProfileListInfoKHR` of the profile alone.
    pub fn with_list<R>(&self, f: impl FnOnce(&mut vk::VideoProfileListInfoKHR<'_>) -> R) -> R {
        VideoProfile::H265(*self).with_list(f)
    }
}

/// A decode profile of either codec, as images, buffers and sessions are
/// made for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoProfile {
    /// An H.264 profile.
    H264(H264Profile),
    /// An H.265 profile.
    H265(H265Profile),
}

impl From<H264Profile> for VideoProfile {
    fn from(profile: H264Profile) -> Self {
        Self::H264(profile)
    }
}

impl From<H265Profile> for VideoProfile {
    fn from(profile: H265Profile) -> Self {
        Self::H265(profile)
    }
}

impl VideoProfile {
    /// The bit depth of its samples.
    pub fn bit_depth(&self) -> u8 {
        match self {
            Self::H264(_) => 8,
            Self::H265(profile) => profile.bit_depth(),
        }
    }

    /// The format its pictures are decoded into, and its outputs have:
    /// [`PICTURE_FORMAT`] (NV12) at 8 bits, [`P010_FORMAT`] at 10.
    pub fn format(&self) -> vk::Format {
        if self.bit_depth() > 8 {
            P010_FORMAT
        } else {
            PICTURE_FORMAT
        }
    }

    /// The bytes of one sample in its format: 1 in NV12, 2 in P010.
    pub fn sample_bytes(&self) -> u32 {
        if self.bit_depth() > 8 { 2 } else { 1 }
    }

    /// Calls `f` with the profile's `VkVideoProfileInfoKHR`, its codec's
    /// part chained.
    pub fn with_info<R>(&self, f: impl FnOnce(&vk::VideoProfileInfoKHR<'_>) -> R) -> R {
        let depth = if self.bit_depth() > 8 {
            vk::VideoComponentBitDepthFlagsKHR::TYPE_10
        } else {
            vk::VideoComponentBitDepthFlagsKHR::TYPE_8
        };
        let info = vk::VideoProfileInfoKHR::default()
            .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
            .luma_bit_depth(depth)
            .chroma_bit_depth(depth);
        match self {
            Self::H264(profile) => {
                let mut h264 = vk::VideoDecodeH264ProfileInfoKHR::default()
                    .std_profile_idc(u32::from(profile.profile_idc))
                    .picture_layout(vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE);
                f(&info
                    .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_H264)
                    .push_next(&mut h264))
            }
            Self::H265(profile) => {
                let mut h265 = vk::VideoDecodeH265ProfileInfoKHR::default()
                    .std_profile_idc(u32::from(profile.profile_idc));
                f(&info
                    .video_codec_operation(vk::VideoCodecOperationFlagsKHR::DECODE_H265)
                    .push_next(&mut h265))
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

/// What the device answers for one decode profile, `P` being an
/// [`H264Profile`] or an [`H265Profile`] ([`H264Capabilities`],
/// [`H265Capabilities`]).
#[derive(Clone, Copy)]
pub struct Capabilities<P> {
    /// The profile.
    pub profile: P,
    /// The highest level it decodes (`maxLevelIdc`), as a stream codes it:
    /// H.264's `level_idc` (52 for level 5.2), H.265's
    /// `general_level_idc` (153 for level 5.1).
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
    /// beside its DPB picture (`DPB_AND_OUTPUT_DISTINCT`, where outputs
    /// of the profile's format may be decode targets); else into its DPB
    /// picture, then copies that into the output
    /// (`DPB_AND_OUTPUT_COINCIDE`).
    pub decodes_into_output: bool,
    /// `minBitstreamBufferOffsetAlignment`.
    pub min_bitstream_buffer_offset_alignment: u64,
    /// `minBitstreamBufferSizeAlignment`.
    pub min_bitstream_buffer_size_alignment: u64,
    /// The codec's decode std header version the driver takes.
    pub std_header_version: vk::ExtensionProperties,
}

/// What the device answers for one H.264 decode profile.
pub type H264Capabilities = Capabilities<H264Profile>;

/// What the device answers for one H.265 decode profile.
pub type H265Capabilities = Capabilities<H265Profile>;

impl<P: Copy> Capabilities<P> {
    /// The same answers, of `profile`.
    fn of<Q>(&self, profile: Q) -> Capabilities<Q> {
        Capabilities {
            profile,
            max_level_idc: self.max_level_idc,
            min_coded_extent: self.min_coded_extent,
            max_coded_extent: self.max_coded_extent,
            picture_access_granularity: self.picture_access_granularity,
            max_dpb_slots: self.max_dpb_slots,
            max_active_reference_pictures: self.max_active_reference_pictures,
            dpb_and_output_coincide: self.dpb_and_output_coincide,
            dpb_and_output_distinct: self.dpb_and_output_distinct,
            decodes_into_output: self.decodes_into_output,
            min_bitstream_buffer_offset_alignment: self.min_bitstream_buffer_offset_alignment,
            min_bitstream_buffer_size_alignment: self.min_bitstream_buffer_size_alignment,
            std_header_version: self.std_header_version,
        }
    }
}

impl<P: Copy + Into<VideoProfile>> Capabilities<P> {
    /// The same answers, of the profile as either codec's.
    pub(crate) fn video(&self) -> Capabilities<VideoProfile> {
        self.of(self.profile.into())
    }

    /// Whether a session of these takes pictures of `width` by `height`,
    /// or why not, the codec named `codec`.
    pub(crate) fn check_extent(&self, codec: &str, width: u32, height: u32) -> Result<(), Error> {
        let (min, max) = (self.min_coded_extent, self.max_coded_extent);
        if width > max.width || height > max.height || width < min.width || height < min.height {
            return Err(Error::Unsupported(format!(
                "{codec} at {width}x{height}: the device decodes {}x{} to {}x{}",
                min.width, min.height, max.width, max.height
            )));
        }
        Ok(())
    }
}

/// What the device answers for each of Baseline, Main and High, in that
/// order, for decodes of NV12 pictures into NV12 outputs: a profile the
/// device does not decode so, or whose query fails, is left out. Empty
/// where it decodes no H.264.
pub fn h264_capabilities(device: &VideoDevice) -> Vec<H264Capabilities> {
    [H264Profile::BASELINE, H264Profile::MAIN, H264Profile::HIGH]
        .into_iter()
        .filter_map(
            |profile| match profile_capabilities(device, profile.into()) {
                Ok(caps) => caps.map(|caps| caps.of(profile)),
                Err(error) => {
                    log::debug!("H.264 profile {}: {error}", profile.profile_idc);
                    None
                }
            },
        )
        .collect()
}

/// What the device answers for Main and Main 10, in that order, for
/// decodes of NV12 pictures into NV12 outputs and of P010 into P010: a
/// profile the device does not decode so, or whose query fails, is left
/// out. Empty where it decodes no H.265, or the device was not made to
/// ([`VideoDevice::codecs`]).
pub fn h265_capabilities(device: &VideoDevice) -> Vec<H265Capabilities> {
    if !device
        .codecs()
        .contains(vk::VideoCodecOperationFlagsKHR::DECODE_H265)
    {
        return Vec::new();
    }
    [H265Profile::MAIN, H265Profile::MAIN_10]
        .into_iter()
        .filter_map(
            |profile| match profile_capabilities(device, profile.into()) {
                Ok(caps) => caps.map(|caps| caps.of(profile)),
                Err(error) => {
                    log::debug!("H.265 profile {}: {error}", profile.profile_idc);
                    None
                }
            },
        )
        .collect()
}

/// `profile`'s capabilities, `None` where it decodes no picture of its
/// format into an output of it, straight or through a copy.
fn profile_capabilities(
    device: &VideoDevice,
    profile: VideoProfile,
) -> Result<Option<Capabilities<VideoProfile>>, Error> {
    let mut h264 = vk::VideoDecodeH264CapabilitiesKHR::default();
    let mut h265 = vk::VideoDecodeH265CapabilitiesKHR::default();
    let mut decode = vk::VideoDecodeCapabilitiesKHR::default();
    let caps = vk::VideoCapabilitiesKHR::default().push_next(&mut decode);
    let mut caps = match profile {
        VideoProfile::H264(_) => caps.push_next(&mut h264),
        VideoProfile::H265(_) => caps.push_next(&mut h265),
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
    let caps = Capabilities {
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
    let mut caps = Capabilities {
        max_level_idc: match profile {
            VideoProfile::H264(_) => level_idc(h264.max_level_idc),
            VideoProfile::H265(_) => h265_level_idc(h265.max_level_idc),
        },
        dpb_and_output_coincide: decode
            .flags
            .contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE),
        dpb_and_output_distinct: decode
            .flags
            .contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_DISTINCT),
        ..caps
    };
    let format = profile.format();
    let offered = |usage, wanted_usage, wanted_flags| {
        Ok::<_, Error>(video_formats(device, profile, usage)?.iter().any(|f| {
            f.format == format
                && f.image_tiling == vk::ImageTiling::OPTIMAL
                && f.image_usage_flags.contains(wanted_usage)
                && f.image_create_flags.contains(wanted_flags)
        }))
    };
    let dst = vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR;
    let dpb = vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR;
    if caps.dpb_and_output_distinct
        && offered(dpb, dpb, vk::ImageCreateFlags::empty())?
        && offered(dst, dst | OUTPUT_USAGE, OUTPUT_FLAGS)?
    {
        caps.decodes_into_output = true;
        return Ok(Some(caps));
    }
    if caps.dpb_and_output_coincide
        && offered(
            dst | dpb,
            dst | dpb | vk::ImageUsageFlags::TRANSFER_SRC,
            vk::ImageCreateFlags::empty(),
        )?
        && plain_outputs(device, format)
    {
        return Ok(Some(caps));
    }
    Ok(None)
}

/// The formats `profile` decodes into, or predicts from, for `usage`.
fn video_formats(
    device: &VideoDevice,
    profile: VideoProfile,
    usage: vk::ImageUsageFlags,
) -> Result<Vec<vk::VideoFormatPropertiesKHR<'static>>, Error> {
    profile.with_list(|list| video_formats_in(device, list, usage))
}

/// The formats the profiles of `list` take for `usage`: decode into or
/// predict from, encode from.
pub(crate) fn video_formats_in(
    device: &VideoDevice,
    list: &mut vk::VideoProfileListInfoKHR<'_>,
    usage: vk::ImageUsageFlags,
) -> Result<Vec<vk::VideoFormatPropertiesKHR<'static>>, Error> {
    let fp = device.video_instance().fp();
    let info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
        .image_usage(usage)
        .push_next(list);
    let mut count = 0;
    // SAFETY: the physical device is the instance's; `info` and its chain
    // live through the call; a null output asks for the count alone.
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
}

/// Whether images of `format` that no decode writes, copied into, sampled
/// and copied out, with [`OUTPUT_FLAGS`], are the device's.
fn plain_outputs(device: &VideoDevice, format: vk::Format) -> bool {
    let info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(format)
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

/// `general_level_idc` as a stream codes it, of an H.265 std level: 153
/// for `STD_VIDEO_H265_LEVEL_IDC_5_1`.
pub(crate) fn h265_level_idc(level: StdVideoH265LevelIdc) -> u8 {
    usize::try_from(level)
        .ok()
        .and_then(|i| H265_LEVELS.get(i))
        .copied()
        .unwrap_or(0)
}

/// The H.265 std level of a stream's `general_level_idc`: the least std
/// level at or above it, or the highest.
pub(crate) fn h265_std_level(general_level_idc: u8) -> StdVideoH265LevelIdc {
    let at = H265_LEVELS
        .iter()
        .position(|&l| l >= general_level_idc)
        .unwrap_or(H265_LEVELS.len() - 1);
    StdVideoH265LevelIdc::try_from(at).unwrap_or(0)
}

/// `general_level_idc` of each H.265 std level, by its value
/// (`STD_VIDEO_H265_LEVEL_IDC_1_0` is 0): 30 times the level.
const H265_LEVELS: [u8; 13] = [30, 60, 63, 90, 93, 120, 123, 150, 153, 156, 180, 183, 186];

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
    fn h265_levels_map_both_ways() {
        assert_eq!(
            h265_level_idc(ash::vk::native::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_5_1),
            153
        );
        assert_eq!(
            h265_level_idc(ash::vk::native::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_1_0),
            30
        );
        assert_eq!(
            h265_level_idc(ash::vk::native::StdVideoH265LevelIdc_STD_VIDEO_H265_LEVEL_IDC_6_2),
            186
        );
        for &level in &H265_LEVELS {
            assert_eq!(h265_level_idc(h265_std_level(level)), level);
        }
        // A level between two, and one past the last.
        assert_eq!(h265_level_idc(h265_std_level(100)), 120);
        assert_eq!(h265_level_idc(h265_std_level(255)), 186);
        assert_eq!(h265_level_idc(99), 0);
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
    fn h265_profiles_follow_the_bit_depth() {
        assert_eq!(H265Profile::for_bit_depth(8), Some(H265Profile::MAIN));
        assert_eq!(H265Profile::for_bit_depth(10), Some(H265Profile::MAIN_10));
        assert_eq!(H265Profile::for_bit_depth(12), None);
        let main10 = VideoProfile::from(H265Profile::MAIN_10);
        assert_eq!(main10.format().as_raw(), P010_FORMAT.as_raw());
        assert_eq!(main10.sample_bytes(), 2);
        let main = VideoProfile::from(H265Profile::MAIN);
        assert_eq!(main.format().as_raw(), PICTURE_FORMAT.as_raw());
        assert_eq!(VideoProfile::from(H264Profile::HIGH).sample_bytes(), 1);
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

    #[test]
    fn an_h265_profile_names_its_bit_depth_and_profile() {
        H265Profile::MAIN_10.with_list(|list| {
            assert_eq!(list.profile_count, 1);
            // SAFETY: the list points at one profile, alive in the call.
            let profile = unsafe { &*list.p_profiles };
            assert_eq!(
                profile.video_codec_operation.as_raw(),
                vk::VideoCodecOperationFlagsKHR::DECODE_H265.as_raw()
            );
            assert_eq!(
                profile.luma_bit_depth.as_raw(),
                vk::VideoComponentBitDepthFlagsKHR::TYPE_10.as_raw()
            );
            assert_eq!(
                profile.chroma_bit_depth.as_raw(),
                vk::VideoComponentBitDepthFlagsKHR::TYPE_10.as_raw()
            );
            // SAFETY: its H.265 part, chained by `with_info`.
            let h265 = unsafe {
                &*profile
                    .p_next
                    .cast::<vk::VideoDecodeH265ProfileInfoKHR<'_>>()
            };
            assert_eq!(h265.std_profile_idc, 2);
        });
    }
}
