//! The encoders' queue on a [`VideoDevice`]: a queue of a video-encode
//! family, attached to a device after it is made
//! ([`VideoDevice::attach_encode`]), so that every way of making one, a
//! device of another's or of our own, takes it alike.

use std::ffi::CStr;

use ash::vk;
use reims_vgpu_video_core::sync::Mutex;

use crate::device::VideoDevice;
use crate::error::{Error, Named};

/// The device extensions an encoder needs besides `VK_KHR_video_queue`:
/// the encode queue, then each codec's.
pub const ENCODE_EXTENSIONS: [&CStr; 3] = [
    ash::khr::video_encode_queue::NAME,
    ash::khr::video_encode_h264::NAME,
    ash::khr::video_encode_h265::NAME,
];

/// Another's encode queue, as [`VideoDevice::attach_encode`] takes it.
#[derive(Clone, Copy)]
pub struct RawEncodeQueue {
    /// A queue family with `VK_QUEUE_VIDEO_ENCODE_BIT_KHR` whose video
    /// codec operations include each of `codecs`.
    pub family: u32,
    /// A queue of `family`, which only the encoders on the device submit
    /// to.
    pub queue: vk::Queue,
    /// The codecs encoders on it encode (`ENCODE_H264`, `ENCODE_H265`),
    /// each of whose encode extensions the device was made with
    /// (`VK_KHR_video_encode_h264`, `VK_KHR_video_encode_h265`), with
    /// `VK_KHR_video_encode_queue`.
    pub codecs: vk::VideoCodecOperationFlagsKHR,
    /// Whether the device was made with `VK_KHR_video_maintenance1`'s
    /// `videoMaintenance1` feature enabled: sources and bitstream buffers
    /// are then made for any profile
    /// (`VK_IMAGE_CREATE_VIDEO_PROFILE_INDEPENDENT_BIT_KHR`,
    /// `VK_BUFFER_CREATE_VIDEO_PROFILE_INDEPENDENT_BIT_KHR`).
    pub video_maintenance1: bool,
}

/// The encoders' queue of a device, and the commands they record.
pub(crate) struct EncodeQueue {
    pub(crate) family: u32,
    queue: Mutex<vk::Queue>,
    pub(crate) codecs: vk::VideoCodecOperationFlagsKHR,
    pub(crate) maintenance1: bool,
    /// Whether the family copies (`TRANSFER`), so that a source in a
    /// buffer is copied into its picture, and in-band parameter sets
    /// written, in the encode's own command buffer.
    pub(crate) copies: bool,
    /// `timestampValidBits` of the family: 0 where it writes no
    /// timestamps.
    pub(crate) timestamp_bits: u32,
    pub(crate) encode: ash::khr::video_encode_queue::Device,
    pub(crate) encode_instance: ash::khr::video_encode_queue::Instance,
}

/// What [`VideoDevice::create`] found of a video-encode family.
#[derive(Clone, Copy)]
pub(crate) struct EncodeChoice {
    pub(crate) family: u32,
    /// The queue's index in its family: 0 in a family of its own, 1 in the
    /// decode family's.
    pub(crate) index: u32,
    pub(crate) codecs: vk::VideoCodecOperationFlagsKHR,
}

impl EncodeChoice {
    /// The first family that encodes H.264 or H.265, where the device has
    /// the extensions and a queue of it free: a family of its own, or the
    /// decode family (not the graphics family's too) with two queues. The
    /// families are `flags` (each family's queue flags and queue count)
    /// and `video` (each one's codec operations); `has` says whether the
    /// device has an extension.
    pub(crate) fn find(
        flags: &[(vk::QueueFlags, u32)],
        video: &[vk::QueueFamilyVideoPropertiesKHR<'_>],
        has: &dyn Fn(&CStr) -> bool,
        decode_family: u32,
        graphics_family: u32,
        decode_family_queues: u32,
    ) -> Option<Self> {
        let [queue, h264, h265] = ENCODE_EXTENSIONS;
        if !has(queue) {
            return None;
        }
        (0u32..)
            .zip(flags.iter().zip(video))
            .find_map(|(family, ((f, _), v))| {
                if !f.contains(vk::QueueFlags::VIDEO_ENCODE_KHR) {
                    return None;
                }
                let ops = v.video_codec_operations;
                let mut codecs = vk::VideoCodecOperationFlagsKHR::empty();
                if has(h264) && ops.contains(vk::VideoCodecOperationFlagsKHR::ENCODE_H264) {
                    codecs |= vk::VideoCodecOperationFlagsKHR::ENCODE_H264;
                }
                if has(h265) && ops.contains(vk::VideoCodecOperationFlagsKHR::ENCODE_H265) {
                    codecs |= vk::VideoCodecOperationFlagsKHR::ENCODE_H265;
                }
                if codecs.is_empty() {
                    return None;
                }
                let index = if family != decode_family && family != graphics_family {
                    0
                } else if family == decode_family
                    && family != graphics_family
                    && decode_family_queues >= 2
                {
                    1
                } else {
                    return None;
                };
                Some(Self {
                    family,
                    index,
                    codecs,
                })
            })
    }

    /// The extensions its codecs need.
    pub(crate) fn extensions(&self) -> Vec<&'static CStr> {
        let [queue, h264, h265] = ENCODE_EXTENSIONS;
        let mut out = vec![queue];
        if self
            .codecs
            .contains(vk::VideoCodecOperationFlagsKHR::ENCODE_H264)
        {
            out.push(h264);
        }
        if self
            .codecs
            .contains(vk::VideoCodecOperationFlagsKHR::ENCODE_H265)
        {
            out.push(h265);
        }
        out
    }
}

impl VideoDevice {
    /// Attaches `raw`, the queue encoders on this device submit to:
    /// [`crate::H264Encoder`] and [`crate::H265Encoder`] start only on a
    /// device with one, of their codec. A device [`Self::create`] made has
    /// one where it has a video-encode family with a queue free. Refused
    /// where one is attached already, or the family is not the device's.
    ///
    /// # Safety
    ///
    /// `raw.queue` is a queue of `raw.family` of the device, which nothing
    /// but the encoders made on this device submits to for as long as it
    /// lives, and the device was made as `raw`'s fields say.
    pub unsafe fn attach_encode(&self, raw: RawEncodeQueue) -> Result<(), Error> {
        // SAFETY: the physical device is the instance's.
        let families = unsafe {
            self.instance()
                .get_physical_device_queue_family_properties(self.physical_device())
        };
        let family = usize::try_from(raw.family)
            .ok()
            .and_then(|f| families.get(f))
            .filter(|f| f.queue_flags.contains(vk::QueueFlags::VIDEO_ENCODE_KHR))
            .ok_or_else(|| {
                Error::Unsupported(format!("queue family {} does not encode video", raw.family))
            })?;
        let queue = EncodeQueue {
            family: raw.family,
            queue: Mutex::new(raw.queue),
            codecs: raw.codecs
                & (vk::VideoCodecOperationFlagsKHR::ENCODE_H264
                    | vk::VideoCodecOperationFlagsKHR::ENCODE_H265),
            maintenance1: raw.video_maintenance1,
            copies: family.queue_flags.intersects(
                vk::QueueFlags::TRANSFER | vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE,
            ),
            timestamp_bits: family.timestamp_valid_bits,
            encode: ash::khr::video_encode_queue::Device::new(self.instance(), self.device()),
            encode_instance: ash::khr::video_encode_queue::Instance::new(
                self.entry(),
                self.instance(),
            ),
        };
        self.encode
            .set(queue)
            .map_err(|_| Error::Unsupported("an encode queue is attached already".into()))
    }

    /// The encoders' queue family, once one is attached.
    pub fn encode_family(&self) -> Option<u32> {
        self.encode.get().map(|e| e.family)
    }

    /// The encoders' queue, once one is attached: for a tool that hands
    /// the device to [`Self::from_raw`] and attaches it again.
    pub fn encode_queue(&self) -> Option<vk::Queue> {
        self.encode.get().map(|e| *e.queue.lock())
    }

    /// The codecs encoders on it encode: none before an encode queue is
    /// attached.
    pub fn encode_codecs(&self) -> vk::VideoCodecOperationFlagsKHR {
        self.encode
            .get()
            .map_or(vk::VideoCodecOperationFlagsKHR::empty(), |e| e.codecs)
    }

    /// Whether `videoMaintenance1` is enabled, as the attached encode
    /// queue says: sources and bitstream buffers are then made for any
    /// profile.
    pub fn encode_profile_independent(&self) -> bool {
        self.encode.get().is_some_and(|e| e.maintenance1)
    }

    /// The encode queue, or why there is none.
    pub(crate) fn encoding(&self) -> Result<&EncodeQueue, Error> {
        self.encode
            .get()
            .ok_or_else(|| Error::NoVideoEncode(format!("{}: no encode queue", self.name())))
    }

    /// The families an encoder's sources and bitstream buffers are shared
    /// between: the graphics family, whose work writes the sources and
    /// reads the bitstream, and the encode family.
    pub fn encode_families(&self) -> Vec<u32> {
        let mut out = vec![self.graphics_family()];
        if let Some(family) = self.encode_family()
            && family != self.graphics_family()
        {
            out.push(family);
        }
        out
    }

    /// Submits to the encode queue.
    pub(crate) fn submit_encode(&self, submits: &[vk::SubmitInfo2<'_>]) -> Result<(), Error> {
        let encode = self.encoding()?;
        let queue = encode.queue.lock();
        // SAFETY: the queue is the device's and is held for the call; the
        // submissions' command buffers are recorded and their semaphores
        // the device's.
        unsafe { self.queue_submit2(*queue, submits, vk::Fence::null()) }
    }

    /// Submits to the encode queue a batch of no work that waits for
    /// `waits` and then signals `signals`: the caller's own semaphores,
    /// ordered after an encoder's work by a wait for its timeline
    /// semaphore's value ([`crate::H264Encoder::submitted`]).
    pub fn signal_on_encode_queue(
        &self,
        waits: &[vk::SemaphoreSubmitInfo<'_>],
        signals: &[vk::SemaphoreSubmitInfo<'_>],
    ) -> Result<(), Error> {
        let submits = [vk::SubmitInfo2::default()
            .wait_semaphore_infos(waits)
            .signal_semaphore_infos(signals)];
        self.submit_encode(&submits)
    }

    /// `vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR` of
    /// `info` into `out`.
    pub(crate) fn encode_quality_level(
        &self,
        info: &vk::PhysicalDeviceVideoEncodeQualityLevelInfoKHR<'_>,
        out: &mut vk::VideoEncodeQualityLevelPropertiesKHR<'_>,
    ) -> Result<(), Error> {
        let encode = self.encoding()?;
        // SAFETY: the physical device is the instance's; `info`, `out` and
        // their chains live through the call.
        unsafe {
            (encode
                .encode_instance
                .fp()
                .get_physical_device_video_encode_quality_level_properties_khr)(
                self.physical_device(),
                info,
                out,
            )
        }
        .named("vkGetPhysicalDeviceVideoEncodeQualityLevelPropertiesKHR")
    }
}
