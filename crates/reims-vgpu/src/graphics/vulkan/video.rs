//! The reims-graphics device's hardware video on the Vulkan platform ([`graphics_video`]):
//! Vulkan Video on the runtime's own device, through the video crate's decoder
//! (`reims-vgpu-video-vulkan`), on the video-decode queue the device made for it
//! (`reims_vgpu_vulkan::platform::VIDEO_DECODE_EXTENSIONS` says what it enables for it).
//!
//! - **One video a device.** The decoders of every session of a runtime device share one
//!   [`VideoDevice`] over its handles, which alone submits to the device's video-decode queue.
//!   It holds the runtime's device, whose handles it uses, for as long as any session lives.
//! - **An access unit is read in place**, in the guest's buffer, through a buffer of the
//!   decoder's over the runtime's import of its pages (`Buffer::vk_host_memory`), where the
//!   device takes it there: the guest's video driver pads each access unit with zeros to the
//!   device's bitstream alignment, which the caps answer ([`RgfxVideoCodecCaps`]'s
//!   `bitstream_alignment`). Elsewhere the decoder copies it.
//! - **A picture lands in the surface's planes**, in the buffer the planes' textures lie in,
//!   through a buffer of the decoder's over that buffer's import ([`ImportedPlanes`]): one GPU
//!   copy out of the decoder's DPB picture writes them, cropped. The guest's video driver
//!   recycles an output surface once it is shown, whether or not its picture is still a
//!   reference, so the decoder keeps its references in DPB pictures of its own, and the copy out
//!   is the one GPU copy a picture costs where the device's DPB pictures and outputs coincide.
//!   The write is bracketed for the dirty log ([`host_write`]) while the planes are held.
//! - **Order.** A decode starts as the host takes the command buffer that encodes it, after the
//!   session's earlier decodes on the same queue. The session's shared event, which the command
//!   buffer waits for, is then signalled by a submission of no work on the decode queue that
//!   waits for the decode ([`VideoDevice::signal_on_decode_queue`]), its signal reserved with the
//!   runtime (`SharedEvent::reserve_vk_signal`); where that submission fails, the host signals
//!   it, so the command buffer goes on.

use std::sync::{Arc, LazyLock, Weak};
use std::time::Duration;

use ash::vk;
use metal2vulkan_runtime as rt;
use reims_graphics::wire::{
    RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_DECODE, RGFX_VIDEO_FORMAT_NV12, RGFX_VIDEO_H264_BASELINE,
    RGFX_VIDEO_H264_HIGH, RGFX_VIDEO_H264_MAIN, RgfxVideoCodecCaps,
};
use reims_vgpu_core::memory::dirty::host_write;
use reims_vgpu_core::platform::Handle as _;
use reims_vgpu_core::platform::device::GpuDevice as _;
use reims_vgpu_core::platform::resource::Buffer as _;
use reims_vgpu_core::raise::{Raise, RaiseResult};
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_video_vulkan::{
    Error as VideoError, H264Capabilities, H264Decoder, H264Profile, HostImport, ImportedPlanes,
    RawDevice, SampleMemory, SurfaceSource, SurfaceSpec, VideoDevice, h264_capabilities,
};
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::gpu::device::VulkanDevice;
use reims_vgpu_vulkan::gpu::event::VulkanSharedEvent;
use reims_vgpu_vulkan::platform::VIDEO_DECODE_EXTENSIONS;

use crate::graphics::decode::{
    GraphicsVideo, GraphicsVideoSession, H264Decode, VideoPlane, op_fault,
};

/// How long a session's end waits for its last signal to land.
const TEARDOWN_WAIT: Duration = Duration::from_secs(10);

/// The hardware video of `device`: the one its sessions share, made on first use; `None` where
/// the device has no video-decode queue, lacks the extensions its decodes use, or decodes no
/// H.264.
pub(crate) fn graphics_video(
    device: &VulkanDevice,
) -> Option<Arc<dyn GraphicsVideo<VulkanPlatform>>> {
    // The video of each runtime device, by its registry ID, while one lives.
    static MADE: LazyLock<Mutex<Vec<(u64, Weak<DeviceVideo>)>>> =
        LazyLock::new(|| Mutex::new(Vec::new()));
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

/// A runtime device's hardware video: its decoders' device, and what it answers for H.264.
struct DeviceVideo {
    /// The decoders' device, over the runtime's handles; dropped before the runtime's device.
    video: Arc<VideoDevice>,
    /// What it answers for each H.264 profile it decodes.
    h264: Vec<H264Capabilities>,
    /// The runtime's device, whose handles `video` uses.
    device: VulkanDevice,
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
        // SAFETY: the physical device is the instance's.
        let families = unsafe {
            handles
                .instance
                .get_physical_device_queue_family_properties(handles.physical_device)
        };
        let copies = vk::QueueFlags::TRANSFER | vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;
        let decode_copies = usize::try_from(queue.family)
            .ok()
            .and_then(|family| families.get(family))
            .is_some_and(|family| family.queue_flags.intersects(copies));
        if !decode_copies {
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
            copy_family: queue.family,
            copy_queue: queue.queue,
            graphics_family: handles.queue_family,
            unified_image_layouts_video: false,
        };
        // SAFETY: the handles are the runtime's device's, made as `RawDevice` says (the video
        // extensions and synchronization2 checked above, timeline semaphores the runtime's own
        // events use), and live while `device` holds the device, which outlives `video`; the
        // queue is submitted to by `video`'s decoders alone.
        let video = unsafe { VideoDevice::from_raw(device_raw) };
        let h264 = h264_capabilities(&video);
        if h264.is_empty() {
            log::info!(
                "hardware video: {} decodes no H.264 into NV12",
                video.name()
            );
            return None;
        }
        log::info!(
            "hardware video: H.264 decode on {}, queue family {}",
            video.name(),
            queue.family
        );
        Some(Self {
            video,
            h264,
            device: device.clone_ref(),
        })
    }
}

impl GraphicsVideo<VulkanPlatform> for VulkanVideo {
    /// H.264 decode: every profile some device profile decodes, at the least of their limits.
    fn caps(&self) -> Vec<RgfxVideoCodecCaps> {
        h264_caps(&self.0.h264).into_iter().collect()
    }

    fn session(
        &self,
        codec: u32,
        direction: u32,
    ) -> Option<Arc<dyn GraphicsVideoSession<VulkanPlatform>>> {
        if codec != RGFX_VIDEO_CODEC_H264 || direction != RGFX_VIDEO_DECODE {
            return None;
        }
        Some(Arc::new(H264Session::new(&self.0)?))
    }
}

/// The caps of H.264 decode, from what the device answers for each profile; `None` for none.
fn h264_caps(profiles: &[H264Capabilities]) -> Option<RgfxVideoCodecCaps> {
    let first = profiles.first()?;
    let mut caps = RgfxVideoCodecCaps {
        codec: RGFX_VIDEO_CODEC_H264,
        direction: RGFX_VIDEO_DECODE,
        max_level: u32::from(first.max_level_idc),
        min_size: [first.min_coded_extent.width, first.min_coded_extent.height],
        max_size: [first.max_coded_extent.width, first.max_coded_extent.height],
        max_dpb_pictures: first.max_dpb_slots,
        max_references: first.max_active_reference_pictures,
        formats: RGFX_VIDEO_FORMAT_NV12,
        ..RgfxVideoCodecCaps::default()
    };
    let mut alignment = 1u64;
    for p in profiles {
        caps.max_level = caps.max_level.min(u32::from(p.max_level_idc));
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
    caps.bitstream_alignment = u32::try_from(alignment).unwrap_or(u32::MAX);
    Some(caps)
}

/// Where a decoder's next picture lands: the output of the decode being taken, which the session
/// puts here before it decodes the access unit.
struct GuestOutputs(Arc<Mutex<Option<ImportedPlanes>>>);

impl SurfaceSource for GuestOutputs {
    type Surface = ImportedPlanes;

    fn surface(&mut self, _spec: &SurfaceSpec) -> Result<ImportedPlanes, VideoError> {
        self.0.lock().take().ok_or_else(|| {
            VideoError::Unsupported(
                "a second picture in one access unit, which has one output".into(),
            )
        })
    }
}

/// An H.264 decode session.
struct H264Session {
    /// The decoder, and the counts it is driven by; dropped first.
    state: Mutex<DecodeState>,
    /// The output of the decode being taken ([`GuestOutputs`]).
    next: Arc<Mutex<Option<ImportedPlanes>>>,
    /// The shared event each decode signals.
    event: VulkanSharedEvent,
    /// The device, held while the decoder lives.
    video: Arc<DeviceVideo>,
}

/// A session's decoder, and the counts it is driven by.
struct DecodeState {
    decoder: H264Decoder<GuestOutputs>,
    /// The access units decoded, which tag each.
    samples: u64,
    /// The last value of the session's event a decode signals.
    signalled: u64,
}

impl H264Session {
    fn new(video: &Arc<DeviceVideo>) -> Option<Self> {
        let next = Arc::new(Mutex::new(None));
        let decoder = match H264Decoder::new_annex_b(&video.video, GuestOutputs(Arc::clone(&next)))
        {
            Ok(decoder) => decoder,
            Err(error) => {
                log::warn!("hardware video: no H.264 decoder: {error}");
                return None;
            }
        };
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
    fn signal(&self, decoder: &H264Decoder<GuestOutputs>, value: u64) -> RaiseResult<()> {
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
        let Err(error) = self.video.video.signal_on_decode_queue(&waits, &signals) else {
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

impl GraphicsVideoSession<VulkanPlatform> for H264Session {
    fn event(&self) -> &VulkanSharedEvent {
        &self.event
    }

    fn decode_h264(&self, decode: &H264Decode<'_, VulkanPlatform>) -> RaiseResult<u64> {
        let mut state = self.state.lock();
        let state = &mut *state;
        let output = output(&self.video.video, decode)?;
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

impl Drop for H264Session {
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
    decode: &H264Decode<'_, VulkanPlatform>,
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
    decoded.map_err(|error| decode_fault(error))?;
    Ok(())
}

/// A command-buffer fault for a decode the session could not take.
fn decode_fault(what: impl std::fmt::Display) -> Raise {
    op_fault(format_args!("an H.264 decode: {what}"))
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

/// The output `decode`'s planes are, in the buffers their textures lie in, which it holds.
fn output(
    video: &Arc<VideoDevice>,
    decode: &H264Decode<'_, VulkanPlatform>,
) -> RaiseResult<ImportedPlanes> {
    let (luma, chroma) = (&decode.luma, &decode.chroma);
    let in_buffer = |p: &VideoPlane<'_, VulkanPlatform>| {
        p.bytes_per_row
            .checked_mul(p.height)
            .and_then(|bytes| bytes.checked_add(p.offset))
            .is_some_and(|end| end <= p.buffer.length())
    };
    if !in_buffer(luma) || !in_buffer(chroma) {
        return Err(decode_fault("an output plane past its buffer"));
    }
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
        ))
    };
    let (Some(luma_shape), Some(chroma_shape)) =
        (shape(luma, luma.offset), shape(chroma, chroma_at))
    else {
        return Err(decode_fault("an output plane past 4 GiB"));
    };
    // The planes' bytes, written while the output lives, for the dirty log.
    let end = |host: u64, p: &VideoPlane<'_, VulkanPlatform>| host + p.bytes_per_row * p.height;
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
    planes.ok_or_else(|| decode_fault("an output the device takes no buffer over"))
}
