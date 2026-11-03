//! The reims-graphics device's hardware video on the Vulkan platform ([`graphics_video`]):
//! Vulkan Video on the runtime's own device, through the video crate's decoders
//! (`reims-vgpu-video-vulkan`), on the video-decode queue the device made for it
//! (`reims_vgpu_vulkan::platform::VIDEO_DECODE_EXTENSIONS` says what it enables for it): H.264,
//! and HEVC, Main and Main 10, where the device has `VK_KHR_video_decode_h265` and the queue's
//! family decodes it (`VIDEO_DECODE_H265_EXTENSION`).
//!
//! - **One video a device.** The decoders of every session of a runtime device share one
//!   [`VideoDevice`] over its handles, which alone submits to the device's video-decode queue.
//!   It holds the runtime's device, whose handles it uses, for as long as any session lives.
//! - **An access unit is read in place**, in the guest's buffer, through a buffer of the
//!   decoder's over the runtime's import of its pages (`Buffer::vk_host_memory`), where the
//!   device takes it there: the guest's video driver pads each access unit with zeros to the
//!   device's bitstream alignment, which the caps answer ([`RgfxVideoCodecCaps`]'s
//!   `bitstream_alignment`). Elsewhere the decoder copies it.
//! - **A picture lands in the surface's planes**, in the buffer the planes' textures lie in: one
//!   GPU copy out of the decoder's DPB picture writes them, cropped. The guest's video driver
//!   recycles an output surface once it is shown, whether or not its picture is still a
//!   reference, so the decoder keeps its references in DPB pictures of its own, and the copy out
//!   is the one GPU copy a picture costs where the device's DPB pictures and outputs coincide.
//!   - On a GPU of its own memory, the surface's block has a device cache, which the guest's
//!     video driver asks for (`RGFX_VIDEO_CAPS_OUTPUT_CACHED`): the picture lands there, in the
//!     runtime's buffer itself, and crosses the bus only at a CPU lock, as the cache's download.
//!     The runtime shares its buffers with its own queues' families alone, so the copy runs on a
//!     queue of those families: the one the runtime reserves for the decoders
//!     (`DeviceHandles::reserved_queues`, which the platform's options ask for), where no runtime
//!     work runs, so each copy is submitted with its decode and waits for it on the GPU. On a
//!     device with no queue to spare for it, the copy runs on the `VkQueue` of a runtime command
//!     queue of the decoders' own, which the runtime lends for each submission
//!     ([`RuntimeQueue`]). Logical queues share `VkQueue`s, so the decoders submit each copy
//!     there once the host sees its decode done, with no wait on the GPU that would hold a
//!     guest's work behind the decode. Measured on an RTX 5080 Laptop GPU: a 4K
//!     picture lands there in 100 us of the GPU (1080p in 27), where the decode queue copies it
//!     within the GPU's memory in 180 to 215, and into the pages across PCIe in 3.5 ms, as the
//!     link idles down under a load of decodes alone; a CPU lock of the surface then downloads it
//!     in 1 to 5 ms (12.4 MB).
//!   - Elsewhere it lands in the surface's pages, through a buffer of the decoder's over the
//!     buffer's import ([`ImportedPlanes`]), copied in the decode's own submission where the GPU
//!     shares the CPU's memory. The write is bracketed for the dirty log ([`host_write`]) while
//!     the planes are held.
//! - **Order.** A decode starts as the host takes the command buffer that encodes it, after the
//!   session's earlier decodes on the same queue. The session's shared event, which the command
//!   buffer waits for, is then signalled by a submission of no work that waits for the decode
//!   and its copy out ([`VideoDevice::signal_after_decode`]), its signal reserved with the
//!   runtime (`SharedEvent::reserve_vk_signal`); where that submission fails, the host signals
//!   it, so the command buffer goes on.
//! - **Decodes beside copies.** Where the device has `VK_KHR_unified_image_layouts`'s
//!   `unifiedImageLayoutsVideo`, which the platform enables where it is there
//!   (`VIDEO_DECODE_UNIFIED_LAYOUTS_EXTENSION`), the decoders keep their pictures in `GENERAL`;
//!   with the copies on the reserved queue, a decode then waits only for the copy out of the DPB
//!   picture it sets up again, and the event's signal goes on the reserved queue behind the
//!   copy, so the decode queue runs the next decode while a picture is copied out
//!   ([`VideoDevice::overlaps_copies`]). Elsewhere each decode waits for the copy before it, and
//!   the signal goes on the decode queue.

use std::sync::{Arc, LazyLock, Weak};
use std::time::Duration;

use ash::vk;
use metal2vulkan_runtime as rt;
use reims_graphics::wire::{
    RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_DECODE, RGFX_VIDEO_FORMAT_NV12,
    RGFX_VIDEO_FORMAT_P010, RGFX_VIDEO_H264_BASELINE, RGFX_VIDEO_H264_HIGH, RGFX_VIDEO_H264_MAIN,
    RGFX_VIDEO_HEVC_MAIN, RGFX_VIDEO_HEVC_MAIN_10, RgfxVideoCodecCaps,
};
use reims_vgpu_core::memory::dirty::host_write;
use reims_vgpu_core::platform::Handle as _;
use reims_vgpu_core::platform::device::GpuDevice as _;
use reims_vgpu_core::platform::resource::Buffer as _;
use reims_vgpu_core::raise::{Raise, RaiseResult};
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_video_vulkan::{
    BufferPlanes, Capabilities, Error as VideoError, H264Capabilities, H264Decoder, H264Profile,
    H265Capabilities, H265Decoder, H265Profile, HostImport, ImportedPlanes, Landing, LentQueue,
    RawDevice, SampleMemory, Surface, SurfaceSource, SurfaceSpec, VideoDevice, h264_capabilities,
    h265_capabilities, hevc,
};
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::gpu::device::VulkanDevice;
use reims_vgpu_vulkan::gpu::event::VulkanSharedEvent;
use reims_vgpu_vulkan::platform::{
    VIDEO_DECODE_EXTENSIONS, VIDEO_DECODE_H265_EXTENSION, VIDEO_DECODE_UNIFIED_LAYOUTS_EXTENSION,
};

use crate::graphics::decode::{
    GraphicsVideo, GraphicsVideoSession, VideoDecode, VideoPlane, op_fault,
};

/// How long a session's end waits for its last signal to land.
const TEARDOWN_WAIT: Duration = Duration::from_secs(10);

/// The video of each runtime device, by its registry ID, while one lives.
type MadeVideos = Vec<(u64, Weak<DeviceVideo>)>;

/// The hardware video of `device`: the one its sessions share, made on first use; `None` where
/// the device has no video-decode queue, lacks the extensions its decodes use, or decodes no
/// H.264. HEVC rides with it where the device decodes it.
pub(crate) fn graphics_video(
    device: &VulkanDevice,
) -> Option<Arc<dyn GraphicsVideo<VulkanPlatform>>> {
    static MADE: LazyLock<Mutex<MadeVideos>> = LazyLock::new(|| Mutex::new(Vec::new()));
    let mut made = MADE.lock();
    made.retain(|(_, video)| video.strong_count() > 0);
    let kept = (made.iter())
        .find(|(id, _)| *id == device.registry_id)
        .and_then(|(_, video)| video.upgrade());
    let video = match kept {
        Some(video) => video,
        None => {
            let video = Arc::new(DeviceVideo::new(device)?);
            made.push((device.registry_id, Arc::downgrade(&video)));
            video
        }
    };
    Some(Arc::new(VulkanVideo(video)))
}

/// A runtime device's hardware video, as the graphics device sees it.
struct VulkanVideo(Arc<DeviceVideo>);

/// A runtime device's hardware video: its decoders' device, and what it answers for H.264 and
/// HEVC.
struct DeviceVideo {
    /// The decoders' device, over the runtime's handles; dropped before the runtime's device.
    video: Arc<VideoDevice>,
    /// What it answers for each H.264 profile it decodes.
    h264: Vec<H264Capabilities>,
    /// What it answers for each HEVC profile it decodes; none where it decodes no HEVC.
    h265: Vec<H265Capabilities>,
    /// Whether the decoders copy pictures out on a queue of the runtime's families (the
    /// reserved queue, or a lent one: [`RuntimeQueue`]), which writes the runtime's buffers, an
    /// IOSurface's device cache among them.
    caches: bool,
    /// The runtime's device, whose handles `video` uses.
    device: VulkanDevice,
}

/// The `VkQueue` of a runtime command queue of the decoders' own, which the decoders copy
/// pictures out on where the device reserved no queue for them, lent by the runtime for each
/// submission (`CommandQueue::with_vk_queue`): of a family the runtime's buffers are shared
/// with. No command buffer is made on this queue, but
/// logical queues share `VkQueue`s, so the runtime's work for the guest may run there too: each
/// copy is submitted once the host sees its decode done, and waits for nothing on the GPU.
struct RuntimeQueue {
    queue: rt::CommandQueue,
    family: u32,
}

impl LentQueue for RuntimeQueue {
    fn family(&self) -> u32 {
        self.family
    }

    fn with_queue(&self, submit: &mut dyn FnMut(vk::Queue)) {
        // SAFETY: `submit` submits once, keeps nothing and waits for nothing on the host (the
        // trait's word): it makes no call into the runtime, and its submission waits for
        // nothing.
        unsafe { self.queue.with_vk_queue(|queue, _| submit(queue)) }
    }
}

impl DeviceVideo {
    fn new(device: &VulkanDevice) -> Option<Self> {
        let raw: &rt::Device = &device.raw;
        // SAFETY: the handles are read here and used only through `video`, which this object
        // drops before its own reference to the device; the video-decode queue is the caller's,
        // which the runtime never submits to, and `video` alone submits to it.
        let handles = unsafe { raw.raw() };
        let queue = (handles.extra_queues.iter())
            .find(|q| q.flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR))?;
        let api = handles.instance_api_version.packed();
        // VK_KHR_synchronization2's commands, where the instance's version lacks the core ones.
        let needed = (VIDEO_DECODE_EXTENSIONS.iter())
            .filter(|name| api < vk::API_VERSION_1_3 || **name != c"VK_KHR_synchronization2");
        if let Some(missing) = needed.into_iter().find(|name| !raw.enables_extension(name)) {
            log::info!("hardware video: the device does not enable {missing:?}");
            return None;
        }
        // On a GPU of its own memory, a queue of the runtime's families for the copies out,
        // which reach the runtime's buffers: the reserved one, else a lent one. A lent one stays
        // shared: the copies are bulk, and a queue that lent its `VkQueue` never moves.
        let unified = device
            .identity_caps()
            .flags
            .has_unified_memory
            .unwrap_or(false);
        let reserved = (!unified)
            .then(|| handles.reserved_queues.first())
            .flatten()
            .filter(|reserved| handles.queue_families.contains(&reserved.family));
        let lent = (!unified && reserved.is_none())
            .then(|| match raw.new_command_queue() {
                Ok(queue) => {
                    // SAFETY: the closure only reads the family; it submits nothing.
                    let family = unsafe { queue.with_vk_queue(|_, family| family) };
                    Some(RuntimeQueue { queue, family })
                }
                Err(error) => {
                    log::info!("hardware video: no runtime queue for its copies ({error})");
                    None
                }
            })
            .flatten()
            .filter(|lent| handles.queue_families.contains(&lent.family));
        // SAFETY: the physical device is the instance's.
        let families = unsafe {
            handles
                .instance
                .get_physical_device_queue_family_properties(handles.physical_device)
        };
        let codecs = decoded_codecs(&handles, queue.family, raw);
        let copies = vk::QueueFlags::TRANSFER | vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
        let decode_copies = usize::try_from(queue.family)
            .ok()
            .and_then(|family| families.get(family))
            .is_some_and(|family| family.queue_flags.intersects(copies));
        if reserved.is_none() && lent.is_none() && !decode_copies {
            log::info!(
                "hardware video: the video-decode family {} copies nothing, which a picture's \
                 landing in a surface's planes takes",
                queue.family
            );
            return None;
        }
        let device_raw = RawDevice {
            entry: handles.entry.clone(),
            instance: handles.instance.clone(),
            instance_api_version: api,
            physical_device: handles.physical_device,
            device: handles.device.clone(),
            decode_family: queue.family,
            decode_queue: queue.queue,
            copy_family: reserved.map_or(queue.family, |reserved| reserved.family),
            copy_queue: reserved.map_or(queue.queue, |reserved| reserved.queue),
            graphics_family: handles.queue_family,
            // The platform enables the extension, and its video feature where the device has it,
            // wherever it is there.
            unified_image_layouts_video: raw
                .enables_extension(VIDEO_DECODE_UNIFIED_LAYOUTS_EXTENSION)
                && VideoDevice::reports_unified_image_layouts_video(
                    handles.instance,
                    handles.physical_device,
                ),
        };
        let caches = reserved.is_some() || lent.is_some();
        let copy_queue = match (reserved, &lent) {
            (Some(_), _) => "the reserved queue",
            (None, Some(_)) => "a lent queue",
            (None, None) => "the decode queue",
        };
        let copy_family = (reserved.map(|reserved| reserved.family))
            .or(lent.as_ref().map(|lent| lent.family))
            .unwrap_or(queue.family);
        // SAFETY: the handles are the runtime's device's, made as `RawDevice` says (the video
        // extensions and synchronization2 checked above, each codec's extension and the decode
        // family's codecs by `decoded_codecs`, timeline semaphores the runtime's own events
        // use), and live while `device` holds the device, which outlives `video`; the decode
        // queue and the reserved queue are submitted to by `video`'s decoders alone (the runtime
        // never submits to either, and this device's one video makes the only decoders on
        // them), and a lent queue is the device's, of its family.
        let video = unsafe {
            match lent {
                Some(lent) => VideoDevice::from_raw_copying_on(device_raw, codecs, Box::new(lent)),
                None => Ok(VideoDevice::from_raw_with_codecs(device_raw, codecs)),
            }
        };
        let video = match video {
            Ok(video) => video,
            Err(error) => {
                log::warn!("hardware video: {error}");
                return None;
            }
        };
        let h264 = h264_capabilities(&video);
        if h264.is_empty() {
            log::info!(
                "hardware video: {} decodes no H.264 into NV12",
                video.name()
            );
            return None;
        }
        let h265 = h265_capabilities(&video);
        log::info!(
            "hardware video: H.264{} decode on {}, queue family {}, its pictures copied out on \
             {copy_queue}, family {copy_family}{}",
            if h265.is_empty() { "" } else { " and HEVC" },
            video.name(),
            queue.family,
            if video.overlaps_copies() {
                ", beside the decodes (unified image layouts)"
            } else if video.unified_image_layouts() {
                " (unified image layouts)"
            } else {
                ""
            }
        );
        Some(Self {
            video,
            h264,
            h265,
            caches,
            device: device.clone_ref(),
        })
    }
}

impl GraphicsVideo<VulkanPlatform> for VulkanVideo {
    /// H.264 decode, and HEVC decode where the device has it: every profile some device profile
    /// decodes, at the least of their limits.
    fn caps(&self) -> Vec<RgfxVideoCodecCaps> {
        (h264_caps(&self.0.h264).into_iter())
            .chain(h265_caps(&self.0.h265))
            .collect()
    }

    fn session(
        &self,
        codec: u32,
        direction: u32,
    ) -> Option<Arc<dyn GraphicsVideoSession<VulkanPlatform>>> {
        if direction != RGFX_VIDEO_DECODE {
            return None;
        }
        let decoder = match codec {
            RGFX_VIDEO_CODEC_H264 => CodecDecoder::h264(&self.0)?,
            RGFX_VIDEO_CODEC_HEVC if !self.0.h265.is_empty() => CodecDecoder::h265(&self.0)?,
            _ => return None,
        };
        Some(Arc::new(DecodeSession::new(&self.0, decoder)?))
    }

    /// Where the decoders copy pictures out on a runtime queue, which writes a device cache.
    fn lands_in_device_caches(&self) -> bool {
        self.0.caches
    }
}

/// The caps of `codec`'s decode at the least of the limits the device answers for its
/// `profiles`, `level` each one's highest level; `None` for no profile. Its profiles and formats
/// are the caller's.
fn decode_caps<P: Copy>(
    codec: u32,
    profiles: &[Capabilities<P>],
    level: impl Fn(&Capabilities<P>) -> u32,
) -> Option<RgfxVideoCodecCaps> {
    let first = profiles.first()?;
    let mut caps = RgfxVideoCodecCaps {
        codec,
        direction: RGFX_VIDEO_DECODE,
        max_level: level(first),
        min_size: [first.min_coded_extent.width, first.min_coded_extent.height],
        max_size: [first.max_coded_extent.width, first.max_coded_extent.height],
        max_dpb_pictures: first.max_dpb_slots,
        max_references: first.max_active_reference_pictures,
        ..RgfxVideoCodecCaps::default()
    };
    let mut alignment = 1u64;
    for p in profiles {
        caps.max_level = caps.max_level.min(level(p));
        caps.min_size[0] = caps.min_size[0].max(p.min_coded_extent.width);
        caps.min_size[1] = caps.min_size[1].max(p.min_coded_extent.height);
        caps.max_size[0] = caps.max_size[0].min(p.max_coded_extent.width);
        caps.max_size[1] = caps.max_size[1].min(p.max_coded_extent.height);
        caps.max_dpb_pictures = caps.max_dpb_pictures.min(p.max_dpb_slots);
        caps.max_references = caps.max_references.min(p.max_active_reference_pictures);
        alignment = alignment
            .max(p.min_bitstream_buffer_offset_alignment)
            .max(p.min_bitstream_buffer_size_alignment);
    }
    caps.bitstream_alignment = u32::try_from(alignment).unwrap_or(u32::MAX);
    Some(caps)
}

/// The caps of H.264 decode, from what the device answers for each profile; `None` for none.
fn h264_caps(profiles: &[H264Capabilities]) -> Option<RgfxVideoCodecCaps> {
    let mut caps = decode_caps(RGFX_VIDEO_CODEC_H264, profiles, |p| {
        u32::from(p.max_level_idc)
    })?;
    caps.formats = RGFX_VIDEO_FORMAT_NV12;
    // A stream of a profile decodes on the first of its candidates the device has
    // (`H264Profile::candidates`): Baseline on Main or High, Main on High.
    let has = |profile: H264Profile| profiles.iter().any(|p| p.profile == profile);
    for (bit, idc) in [
        (RGFX_VIDEO_H264_BASELINE, 66),
        (RGFX_VIDEO_H264_MAIN, 77),
        (RGFX_VIDEO_H264_HIGH, 100),
    ] {
        if H264Profile::candidates(idc).iter().any(|&p| has(p)) {
            caps.profiles |= bit;
        }
    }
    Some(caps)
}

/// The caps of HEVC decode, from what the device answers for Main and Main 10, each into its own
/// format (NV12, P010); `None` for neither.
fn h265_caps(profiles: &[H265Capabilities]) -> Option<RgfxVideoCodecCaps> {
    let mut caps = decode_caps(RGFX_VIDEO_CODEC_HEVC, profiles, |p| {
        u32::from(p.max_level_idc)
    })?;
    for p in profiles {
        let (bit, format) = if p.profile == H265Profile::MAIN_10 {
            (RGFX_VIDEO_HEVC_MAIN_10, RGFX_VIDEO_FORMAT_P010)
        } else {
            (RGFX_VIDEO_HEVC_MAIN, RGFX_VIDEO_FORMAT_NV12)
        };
        caps.profiles |= bit;
        caps.formats |= format;
    }
    Some(caps)
}

/// The codecs the device's video-decode family decodes with the extension of each enabled:
/// H.264's, which the decoders' device takes for granted, and H.265's where the family has it.
fn decoded_codecs(
    handles: &rt::DeviceHandles<'_>,
    family: u32,
    raw: &rt::Device,
) -> vk::VideoCodecOperationFlagsKHR {
    let instance = &handles.instance;
    // SAFETY: the physical device is the instance's.
    let count = unsafe {
        instance.get_physical_device_queue_family_properties2_len(handles.physical_device)
    };
    let mut video = vec![vk::QueueFamilyVideoPropertiesKHR::default(); count];
    let mut properties: Vec<vk::QueueFamilyProperties2<'_>> = (video.iter_mut())
        .map(|v| vk::QueueFamilyProperties2::default().push_next(v))
        .collect();
    // SAFETY: as above, `properties` holds `count` structures, each chained to one of `video`.
    unsafe {
        instance
            .get_physical_device_queue_family_properties2(handles.physical_device, &mut properties);
    }
    drop(properties);
    let decodes = usize::try_from(family)
        .ok()
        .and_then(|family| video.get(family))
        .map_or(vk::VideoCodecOperationFlagsKHR::empty(), |v| {
            v.video_codec_operations
        });
    let mut codecs = vk::VideoCodecOperationFlagsKHR::DECODE_H264;
    let h265 = vk::VideoCodecOperationFlagsKHR::DECODE_H265;
    if decodes.contains(h265) && raw.enables_extension(VIDEO_DECODE_H265_EXTENSION) {
        codecs |= h265;
    }
    codecs
}

/// Where a decode's picture lands: its output surface's planes.
enum Output {
    /// In the surface's pages, through the decoder's buffer over their import.
    Pages(ImportedPlanes),
    /// In the surface's device cache, the runtime's buffer, which `keep` holds.
    Cache {
        planes: BufferPlanes,
        _keep: Arc<dyn Send + Sync>,
    },
}

impl Surface for Output {
    fn landing(&self) -> Landing {
        match self {
            Self::Pages(planes) => planes.landing(),
            Self::Cache { planes, .. } => Landing::Buffer(*planes),
        }
    }
}

/// Where a decoder's next picture lands: the output of the decode being taken, which the session
/// puts here before it decodes the access unit.
struct GuestOutputs(Arc<Mutex<Option<Output>>>);

impl SurfaceSource for GuestOutputs {
    type Surface = Output;

    fn surface(&mut self, _spec: &SurfaceSpec) -> Result<Output, VideoError> {
        self.0.lock().take().ok_or_else(|| {
            VideoError::Unsupported(
                "a second picture in one access unit, which has one output".into(),
            )
        })
    }
}

/// A session's decoder, of its codec.
enum CodecDecoder {
    H264(H264Decoder<GuestOutputs>),
    H265(H265Decoder<GuestOutputs>),
}

impl CodecDecoder {
    /// An H.264 decoder of Annex-B access units, landing its pictures where the session puts them.
    fn h264(video: &Arc<DeviceVideo>) -> Option<(Self, Arc<Mutex<Option<Output>>>)> {
        let next = Arc::new(Mutex::new(None));
        match H264Decoder::new_annex_b(&video.video, GuestOutputs(Arc::clone(&next))) {
            Ok(decoder) => Some((Self::H264(decoder), next)),
            Err(error) => {
                log::warn!("hardware video: no H.264 decoder: {error}");
                None
            }
        }
    }

    /// An HEVC decoder of Annex-B access units, as [`Self::h264`].
    fn h265(video: &Arc<DeviceVideo>) -> Option<(Self, Arc<Mutex<Option<Output>>>)> {
        let next = Arc::new(Mutex::new(None));
        match H265Decoder::new_annex_b(&video.video, GuestOutputs(Arc::clone(&next))) {
            Ok(decoder) => Some((Self::H265(decoder), next)),
            Err(error) => {
                log::warn!("hardware video: no HEVC decoder: {error}");
                None
            }
        }
    }

    /// The session's codec (`RGFX_VIDEO_CODEC_*`).
    fn codec(&self) -> u32 {
        match self {
            Self::H264(_) => RGFX_VIDEO_CODEC_H264,
            Self::H265(_) => RGFX_VIDEO_CODEC_HEVC,
        }
    }

    /// Decodes `sample`, tagged `id`; the frames it gives back go, their outputs written once
    /// the decoder's timeline passes their work. A picture HEVC's random access skips is no
    /// failure.
    fn decode(&mut self, sample: &[u8], id: u64) -> Result<(), VideoError> {
        match self {
            Self::H264(decoder) => decoder.decode(sample, id).map(drop),
            Self::H265(decoder) => skipped_is_none(decoder.decode(sample, id)),
        }
    }

    /// [`Self::decode`] of a sample whose bytes lie in `memory` too.
    ///
    /// # Safety
    ///
    /// As for the decoders' `decode_in_place`.
    unsafe fn decode_in_place(
        &mut self,
        sample: &[u8],
        id: u64,
        memory: SampleMemory,
    ) -> Result<(), VideoError> {
        // SAFETY: as the caller promises.
        unsafe {
            match self {
                Self::H264(decoder) => decoder.decode_in_place(sample, id, memory).map(drop),
                Self::H265(decoder) => skipped_is_none(decoder.decode_in_place(sample, id, memory)),
            }
        }
    }

    /// The value the decoder's timeline semaphore reaches once its work submitted so far is done.
    fn submitted(&self) -> u64 {
        match self {
            Self::H264(decoder) => decoder.submitted(),
            Self::H265(decoder) => decoder.submitted(),
        }
    }

    /// The decoder's timeline semaphore.
    fn semaphore(&self) -> vk::Semaphore {
        match self {
            Self::H264(decoder) => decoder.semaphore(),
            Self::H265(decoder) => decoder.semaphore(),
        }
    }
}

/// An HEVC decode's answer with a RASL picture the decoder skips (8.1.3) taken as no picture:
/// the guest's output gets nothing, and its command buffer completes.
fn skipped_is_none<T>(decoded: Result<T, VideoError>) -> Result<(), VideoError> {
    match decoded {
        Ok(_) | Err(VideoError::H265Stream(hevc::Error::SkippedLeadingPicture)) => Ok(()),
        Err(error) => Err(error),
    }
}

/// A decode session, of one codec.
struct DecodeSession {
    /// The decoder, and the counts it is driven by; dropped first.
    state: Mutex<DecodeState>,
    /// The output of the decode being taken ([`GuestOutputs`]).
    next: Arc<Mutex<Option<Output>>>,
    /// The shared event each decode signals.
    event: VulkanSharedEvent,
    /// The device, held while the decoder lives.
    video: Arc<DeviceVideo>,
}

/// A session's decoder, and the counts it is driven by.
struct DecodeState {
    decoder: CodecDecoder,
    /// The access units decoded, which tag each.
    samples: u64,
    /// The last value of the session's event a decode signals.
    signalled: u64,
}

impl DecodeSession {
    fn new(
        video: &Arc<DeviceVideo>,
        (decoder, next): (CodecDecoder, Arc<Mutex<Option<Output>>>),
    ) -> Option<Self> {
        let event = video.device.new_shared_event()?;
        Some(Self {
            state: Mutex::new(DecodeState {
                decoder,
                samples: 0,
                signalled: 0,
            }),
            next,
            event,
            video: Arc::clone(video),
        })
    }

    /// Signals the event's `value` once the decoder's work submitted so far is done.
    fn signal(&self, decoder: &CodecDecoder, value: u64) -> RaiseResult<()> {
        // SAFETY: the reservation is signalled below, once, by a submission that first waits
        // for `after`, or from the host where that submission fails.
        let reserved = unsafe { self.event.reserve_vk_signal(value) }
            .map_err(|error| decode_fault(format_args!("the event's signal: {error}")))?;
        let rt::VkEventSignal::Counter { counter, after } = reserved else {
            // Dropped: the event is past the value already.
            return Ok(());
        };
        // SAFETY: the semaphore is waited for and signalled only as the reservation says, while
        // the session holds the event.
        let semaphore = unsafe { self.event.vk_semaphore() };
        let wait = |semaphore, value| {
            vk::SemaphoreSubmitInfo::default()
                .semaphore(semaphore)
                .value(value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        };
        let decoded = decoder.submitted();
        let mut waits = Vec::with_capacity(2);
        if decoded > 0 {
            waits.push(wait(decoder.semaphore(), decoded));
        }
        if let Some(after) = after {
            waits.push(wait(semaphore, after));
        }
        let signals = [wait(semaphore, counter)];
        let Err(error) = self.video.video.signal_after_decode(&waits, &signals) else {
            return Ok(());
        };
        log::error!("hardware video: the event's signal was not submitted ({error})");
        // The reservation is signalled from the host, once the signal before it landed, so
        // the command buffers waiting for the event go on.
        let device = self.video.video.device();
        if let Some(after) = after {
            let semaphores = [semaphore];
            let values = [after];
            let info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            let nanos = u64::try_from(TEARDOWN_WAIT.as_nanos()).unwrap_or(u64::MAX);
            // SAFETY: the semaphore is the device's, a timeline semaphore.
            if let Err(error) = unsafe { device.wait_semaphores(&info, nanos) } {
                log::error!("hardware video: the event's earlier signal: {error}");
            }
        }
        let info = vk::SemaphoreSignalInfo::default()
            .semaphore(semaphore)
            .value(counter);
        // SAFETY: as above; the reservation names `counter`, past every signal before it.
        if let Err(error) = unsafe { device.signal_semaphore(&info) } {
            log::error!("hardware video: the event's signal from the host: {error}");
        }
        Err(decode_fault(format_args!("the event's signal: {error}")))
    }
}

impl GraphicsVideoSession<VulkanPlatform> for DecodeSession {
    fn event(&self) -> &VulkanSharedEvent {
        &self.event
    }

    fn codec(&self) -> u32 {
        self.state.lock().decoder.codec()
    }

    fn decode(&self, decode: &VideoDecode<'_, VulkanPlatform>) -> RaiseResult<u64> {
        let mut state = self.state.lock();
        let state = &mut *state;
        let output = output(&self.video, decode)?;
        *self.next.lock() = Some(output);
        let decoded = decode_access_unit(state, decode);
        // An output no picture took goes now.
        drop(self.next.lock().take());
        decoded?;
        state.signalled += 1;
        let value = state.signalled;
        self.signal(&state.decoder, value)?;
        Ok(value)
    }
}

impl Drop for DecodeSession {
    fn drop(&mut self) {
        // The decoder's semaphore goes with it: the last signal, which waits for it, lands first.
        let last = self.state.lock().signalled;
        if last > 0 && !self.event.wait_until_signaled_value(last, TEARDOWN_WAIT) {
            log::error!("hardware video: a session's last signal did not land");
        }
    }
}

/// Decodes `decode`'s parameter sets, then its access unit, in place where its buffer is host
/// pages imported in place.
fn decode_access_unit(
    state: &mut DecodeState,
    decode: &VideoDecode<'_, VulkanPlatform>,
) -> RaiseResult<()> {
    if !decode.parameter_sets.is_empty() {
        state.samples += 1;
        (state.decoder)
            .decode(decode.parameter_sets, state.samples)
            .map_err(|error| decode_fault(format_args!("its parameter sets: {error}")))?;
    }
    let raw = &decode.bitstream.raw;
    let sample = raw
        .contents()
        .and_then(|bytes| {
            let start = usize::try_from(decode.offset).ok()?;
            let len = usize::try_from(decode.size).ok()?;
            // SAFETY: the buffer's bytes, `length` of them from `contents`, hold
            // `offset..offset + size` (the op's check), and stay mapped while the buffer lives,
            // which the decode holds.
            Some(unsafe { std::slice::from_raw_parts(bytes.as_ptr().add(start), len) })
        })
        .ok_or_else(|| decode_fault("an access unit in a buffer with no host bytes"))?;
    state.samples += 1;
    let id = state.samples;
    // SAFETY: as for `plane_import`: buffers of the decoder's over the import are destroyed,
    // their work done, before `keep` lets the buffer go.
    let import = unsafe { raw.vk_host_memory() };
    let decoded = match import {
        Some(import) => {
            let memory = SampleMemory {
                import: HostImport {
                    memory: import.memory,
                    offset: import.offset + decode.offset,
                    size: import.size - decode.offset,
                    type_index: import.type_index,
                    handle_type: import.handle_type,
                },
                keep: Arc::new(decode.bitstream.clone_ref()),
            };
            // SAFETY: the import holds the buffer's bytes from its offset, so the sample's at
            // `decode.offset` past it, valid while `keep` holds the buffer; the guest writes the
            // access unit's bytes again only once the command buffer that decodes it completed,
            // after the decode.
            unsafe { state.decoder.decode_in_place(sample, id, memory) }
        }
        None => state.decoder.decode(sample, id),
    };
    decoded.map_err(decode_fault)?;
    Ok(())
}

/// A command-buffer fault for a decode the session could not take.
fn decode_fault(what: impl std::fmt::Display) -> Raise {
    op_fault(format_args!("a video decode: {what}"))
}

/// Where `plane`'s bytes lie in the runtime's import of its buffer, and the host address of its
/// first byte; `None` where the buffer is not imported in place.
fn plane_import(plane: &VideoPlane<'_, VulkanPlatform>) -> Option<(rt::VkHostMemory, u64)> {
    // SAFETY: the answer is used for buffers of the decoder's, which are destroyed, their work
    // done, before the planes' buffer goes (the decode's output holds it).
    let import = unsafe { plane.buffer.raw.vk_host_memory() }?;
    let host = plane.buffer.raw.contents()?.as_ptr().addr();
    Some((import, u64::try_from(host).ok()?.checked_add(plane.offset)?))
}

/// The end of a plane's bytes from `at`, its first byte: its last row's last byte, `row_bytes`
/// past that row's start, past which a texture inside a plane whose rows end at its buffer's end
/// has nothing; `None` for no rows, or past 2^64.
fn plane_end(at: u64, bytes_per_row: u64, height: u64, row_bytes: u64) -> Option<u64> {
    let before_last = bytes_per_row.checked_mul(height.checked_sub(1)?)?;
    at.checked_add(before_last)?.checked_add(row_bytes)
}

/// [`plane_end`] of `plane` from `at`.
fn end_of(at: u64, plane: &VideoPlane<'_, VulkanPlatform>) -> Option<u64> {
    plane_end(at, plane.bytes_per_row, plane.height, plane.row_bytes)
}

/// The output `decode`'s planes are, in the buffers their textures lie in, which it holds.
fn output(video: &DeviceVideo, decode: &VideoDecode<'_, VulkanPlatform>) -> RaiseResult<Output> {
    let (luma, chroma) = (&decode.luma, &decode.chroma);
    let in_buffer = |p: &VideoPlane<'_, VulkanPlatform>| {
        end_of(p.offset, p).is_some_and(|end| end <= p.buffer.length())
    };
    if !in_buffer(luma) || !in_buffer(chroma) {
        return Err(decode_fault("an output plane past its buffer"));
    }
    if luma.cached || chroma.cached {
        return cache_output(video, luma, chroma);
    }
    let video = &video.video;
    let (Some((import, luma_host)), Some((chroma_import, chroma_host))) =
        (plane_import(luma), plane_import(chroma))
    else {
        return Err(decode_fault(
            "an output surface whose planes are no host pages in place",
        ));
    };
    // Each plane's first byte from the luma buffer's byte 0 in the import.
    let chroma_at = (chroma_import.memory == import.memory)
        .then(|| chroma_import.offset.checked_sub(import.offset))
        .flatten()
        .and_then(|apart| apart.checked_add(chroma.offset));
    let Some(chroma_at) = chroma_at else {
        return Err(decode_fault("an output surface whose planes lie apart"));
    };
    let shape = |p: &VideoPlane<'_, VulkanPlatform>, at: u64| {
        Some((
            at,
            u32::try_from(p.bytes_per_row).ok()?,
            u32::try_from(p.height).ok()?,
            u32::try_from(p.row_bytes).ok()?,
        ))
    };
    let (Some(luma_shape), Some(chroma_shape)) =
        (shape(luma, luma.offset), shape(chroma, chroma_at))
    else {
        return Err(decode_fault("an output plane past 4 GiB"));
    };
    // The planes' bytes, written while the output lives, for the dirty log (each in its buffer,
    // so its end is too).
    let end = |host: u64, p: &VideoPlane<'_, VulkanPlatform>| end_of(host, p).unwrap_or(host);
    let first = luma_host.min(chroma_host);
    let last = end(luma_host, luma).max(end(chroma_host, chroma));
    let keep: Arc<dyn Send + Sync> = Arc::new((
        luma.buffer.clone_ref(),
        chroma.buffer.clone_ref(),
        host_write(first, last - first),
    ));
    let host = HostImport {
        memory: import.memory,
        offset: import.offset,
        size: import.size,
        type_index: import.type_index,
        handle_type: import.handle_type,
    };
    // SAFETY: the import is the runtime device's, which `video` is over, holding the buffers'
    // bytes from its offset while `keep` holds them; the planes are the decode's to write until
    // the command buffer waiting for it completes.
    let planes = unsafe { ImportedPlanes::new(video, &host, luma_shape, chroma_shape, keep) }
        .map_err(decode_fault)?;
    (planes.map(Output::Pages))
        .ok_or_else(|| decode_fault("an output the device takes no buffer over"))
}

/// The output of planes in their surface's device cache: the runtime's buffer itself, which the
/// decoders' copy out writes on a runtime queue ([`RuntimeQueue`]), with no host write to log.
fn cache_output(
    video: &DeviceVideo,
    luma: &VideoPlane<'_, VulkanPlatform>,
    chroma: &VideoPlane<'_, VulkanPlatform>,
) -> RaiseResult<Output> {
    if !video.caches {
        return Err(decode_fault(
            "an output surface in a device cache, which the device's copies do not reach",
        ));
    }
    // SAFETY: the buffers are written only by the decode's copy, ordered after the cache's fill
    // (the op's wait) and before the work of the command buffer that waits for the decode, while
    // the output holds them.
    let ((buffer, luma_base), (chroma_buffer, chroma_base)) =
        unsafe { (luma.buffer.raw.vk_buffer(), chroma.buffer.raw.vk_buffer()) };
    if !(luma.cached && chroma.cached) || chroma_buffer != buffer {
        return Err(decode_fault("an output surface whose planes lie apart"));
    }
    let plane = |p: &VideoPlane<'_, VulkanPlatform>, base: u64| {
        Some((
            base.checked_add(p.offset)?,
            u32::try_from(p.bytes_per_row).ok()?,
        ))
    };
    let size = (luma_base.checked_add(luma.buffer.length()))
        .zip(chroma_base.checked_add(chroma.buffer.length()))
        .map(|(a, b)| a.max(b));
    let (Some(luma_plane), Some(chroma_plane), Some(size)) =
        (plane(luma, luma_base), plane(chroma, chroma_base), size)
    else {
        return Err(decode_fault("an output plane past 4 GiB"));
    };
    let planes = BufferPlanes {
        buffer,
        luma: luma_plane,
        chroma: chroma_plane,
        size,
    };
    let keep: Arc<dyn Send + Sync> = Arc::new((luma.buffer.clone_ref(), chroma.buffer.clone_ref()));
    Ok(Output::Cache {
        planes,
        _keep: keep,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A plane ends at its last row's last byte: a tile's chroma at (256, 512) of a 2048-byte
    /// pitch plane at 3145728 in a buffer of 4718592 bytes, 256 rows of 512 bytes, ends 1024
    /// bytes before the buffer does, where 256 whole rows past its start would end 512 bytes past
    /// it (the case of an HEIF grid's last row of tiles but its first); no rows is no plane.
    #[test]
    fn a_plane_ends_at_its_last_row_s_last_byte() {
        let at = 3_145_728 + 512 * 2048 + 256 * 2;
        assert_eq!(plane_end(at, 2048, 256, 512), Some(at + 255 * 2048 + 512));
        assert_eq!(plane_end(at, 2048, 256, 512), Some(4_718_592 - 1024));
        assert_eq!(at + 256 * 2048, 4_718_592 + 512);
        assert_eq!(plane_end(0, 2048, 1, 2048), Some(2048));
        assert_eq!(plane_end(0, 2048, 0, 2048), None);
        assert_eq!(plane_end(u64::MAX, 2048, 2, 1), None);
    }

    /// A device's answer for `profile`: `level`, coded sizes from 64x64 to `max`, `slots` DPB
    /// slots and references, and a bitstream alignment of `alignment`.
    fn answer<P>(
        profile: P,
        level: u8,
        max: (u32, u32),
        slots: u32,
        alignment: u64,
    ) -> Capabilities<P> {
        Capabilities {
            profile,
            max_level_idc: level,
            min_coded_extent: vk::Extent2D::default().width(64).height(64),
            max_coded_extent: vk::Extent2D::default().width(max.0).height(max.1),
            picture_access_granularity: vk::Extent2D::default().width(16).height(16),
            max_dpb_slots: slots,
            max_active_reference_pictures: slots,
            dpb_and_output_coincide: true,
            dpb_and_output_distinct: false,
            decodes_into_output: false,
            min_bitstream_buffer_offset_alignment: alignment,
            min_bitstream_buffer_size_alignment: alignment,
            std_header_version: vk::ExtensionProperties::default(),
        }
    }

    /// HEVC's caps list Main and Main 10, each with its format, at the least of their limits,
    /// and the larger of their alignments; a device of Main alone answers NV12 alone, and one of
    /// neither answers nothing.
    #[test]
    fn hevc_caps_answer_each_profile_the_device_decodes() {
        let main = answer(H265Profile::MAIN, 186, (8192, 8192), 17, 256);
        let main10 = answer(H265Profile::MAIN_10, 153, (4096, 8192), 16, 64);
        let caps = h265_caps(&[main, main10]).unwrap();
        assert_eq!(
            caps,
            RgfxVideoCodecCaps {
                codec: RGFX_VIDEO_CODEC_HEVC,
                direction: RGFX_VIDEO_DECODE,
                profiles: RGFX_VIDEO_HEVC_MAIN | RGFX_VIDEO_HEVC_MAIN_10,
                max_level: 153,
                min_size: [64, 64],
                max_size: [4096, 8192],
                max_dpb_pictures: 16,
                max_references: 16,
                formats: RGFX_VIDEO_FORMAT_NV12 | RGFX_VIDEO_FORMAT_P010,
                bitstream_alignment: 256,
            }
        );
        let caps = h265_caps(&[main]).unwrap();
        assert_eq!(caps.profiles, RGFX_VIDEO_HEVC_MAIN);
        assert_eq!(caps.formats, RGFX_VIDEO_FORMAT_NV12);
        assert_eq!(caps.max_level, 186);
        assert!(h265_caps(&[]).is_none());
    }

    /// H.264's caps list each profile some device profile decodes a stream of: a device of High
    /// alone decodes Baseline and Main streams too.
    #[test]
    fn h264_caps_answer_the_streams_the_profiles_decode() {
        let high = answer(H264Profile::HIGH, 52, (4096, 4096), 17, 256);
        let caps = h264_caps(&[high]).unwrap();
        assert_eq!(
            caps.profiles,
            RGFX_VIDEO_H264_BASELINE | RGFX_VIDEO_H264_MAIN | RGFX_VIDEO_H264_HIGH
        );
        assert_eq!(
            (caps.codec, caps.formats),
            (RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_FORMAT_NV12)
        );
        assert_eq!((caps.max_level, caps.bitstream_alignment), (52, 256));
    }
}
