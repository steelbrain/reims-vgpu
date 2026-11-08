//! The reims-graphics device's hardware video on an NVIDIA GPU ([`open`]): NVDEC, through the
//! NVIDIA crate's decoder of whole access units (`reims_vgpu_video_nvidia::AnnexBDecoder`), in
//! the host's video sessions (`super::video_host`), whose event the host signals once a picture
//! has landed.
//!
//! - **Which GPU.** The runtime device's own, by its PCI address, where the device is NVIDIA's
//!   and CUDA opens the GPU there. Anywhere else the arm declines, with one line at info, and the
//!   device's video goes to the next arm (`super::video_choice`). One video a runtime device,
//!   shared by its sessions while one lives.
//! - **The caps** are NVDEC's own answers (`cuvidGetDecoderCaps`): H.264's Baseline, Main and
//!   High where NVDEC decodes H.264 at 8 bits, and HEVC's Main where it decodes HEVC at 8 bits
//!   and Main 10 where it does at 10, each into its format (NV12, P010), at the least of their
//!   sizes. NVDEC answers no level, so the codec's top one: H.264's 6.2 and HEVC's 6.2.
//!   The DPB is the codecs' own bound, 16 references and the current picture for H.264, 15 and
//!   the current picture for HEVC.
//! - **The access unit is read in place** at any place and length (`bitstream_alignment` 1):
//!   the codec crate parses it where it lies in the guest's buffer, and NVDEC is handed the run of
//!   it that holds the picture's slices where they are one run, and copies it within
//!   `cuvidDecodePicture`; otherwise the slices are gathered first. Each session logs, at its
//!   end, how many of its pictures went each way.
//! - **A picture lands by one GPU copy** where both of the output's planes lie in pages a device
//!   reaches in place ([`PlanePages`](super::PlanePages)): their view registered with CUDA, once
//!   a mapping, and one `cuMemcpy2D` a plane from NVDEC's mapped picture into it, the crop's rows
//!   alone at the plane's pitch, waited for before the session signals. So no layout is asked of
//!   the guest. Elsewhere, and where CUDA refuses the view (logged once a session with why), the
//!   picture is read back into the decoder's staging memory and copied by the CPU: a plane in a
//!   device cache, which the guest asks for only where a video lands in caches, and this one
//!   does not.
//! - **A picture the GPU does not finish** by its deadline fails its landing, and every later
//!   picture of the decoder; its copy keeps the planes' mappings and CUDA's registrations of them
//!   until the GPU is done, but not the guest's buffers, so a copy that ends that late may write
//!   pages the guest has used again since.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Weak};

use reims_graphics::wire::{
    RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_DECODE, RGFX_VIDEO_FORMAT_NV12,
    RGFX_VIDEO_FORMAT_P010, RGFX_VIDEO_H264_BASELINE, RGFX_VIDEO_H264_HIGH, RGFX_VIDEO_H264_MAIN,
    RGFX_VIDEO_HEVC_MAIN, RGFX_VIDEO_HEVC_MAIN_10, RgfxVideoCodecCaps,
};
use reims_vgpu_core::platform::Handle as _;
use reims_vgpu_core::sync::Mutex;
use reims_vgpu_video_core::runtime::{Runtime, StdRuntime};
use reims_vgpu_video_nvidia::{
    AnnexBCodec, AnnexBDecoder, AnnexBPicture, ChromaFormat, DecodeCapabilities, Gpu, GpuSelector,
    MemoryPlane, NvdecCodec, ReadPlane, decode_capabilities,
};
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::gpu::device::VulkanDevice;

use super::decode::{GraphicsVideo, GraphicsVideoSession, PlanePages};
use super::video_host::{
    BackendError, HostDecoder, HostPicture, HostVideoSession, Landing, OutputPlane, SourcePlane,
};

/// The PCI vendor ID of NVIDIA's GPUs.
const NVIDIA_VENDOR: u32 = 0x10de;

/// What the log calls the arm.
const NAME: &str = "NVDEC";

/// H.264's top level, 6.2 (`level_idc`).
const H264_TOP_LEVEL: u32 = 62;

/// HEVC's top level, 6.2 (`general_level_idc`, 30 times the level).
const HEVC_TOP_LEVEL: u32 = 186;

/// The video of each runtime device, by its registry ID, while one lives.
type MadeVideos = Vec<(u64, Weak<NvidiaVideo>)>;

/// NVDEC's video for `device`'s sessions: the one they share, made on first use; `None`, with a
/// line at info, where the device is not an NVIDIA GPU CUDA opens, or NVDEC decodes neither H.264
/// nor HEVC there.
pub(crate) fn open(device: &VulkanDevice) -> Option<Arc<dyn GraphicsVideo<VulkanPlatform>>> {
    static MADE: LazyLock<Mutex<MadeVideos>> = LazyLock::new(|| Mutex::new(Vec::new()));
    let mut made = MADE.lock();
    made.retain(|(_, video)| video.strong_count() > 0);
    let kept = (made.iter())
        .find(|(id, _)| *id == device.registry_id)
        .and_then(|(_, video)| video.upgrade());
    let video = match kept {
        Some(video) => video,
        None => {
            let video = Arc::new(NvidiaVideo::new(device)?);
            made.push((device.registry_id, Arc::downgrade(&video)));
            video
        }
    };
    Some(Arc::new(Shared(video)))
}

/// A runtime device's NVDEC, as the graphics device sees it.
struct Shared(Arc<NvidiaVideo>);

/// NVDEC on a runtime device's GPU, and what it answers.
struct NvidiaVideo {
    gpu: Arc<Gpu>,
    /// The runtime device, whose queues copy staged planes into device caches.
    device: VulkanDevice,
    /// The caps of H.264's decode; `None` where NVDEC has none.
    h264: Option<RgfxVideoCodecCaps>,
    /// The caps of HEVC's decode; `None` where NVDEC has none.
    hevc: Option<RgfxVideoCodecCaps>,
    /// The video crates' runtime: the threads and the clock of the NVIDIA crate's waits.
    runtime: Arc<dyn Runtime>,
}

impl NvidiaVideo {
    fn new(device: &VulkanDevice) -> Option<Self> {
        let properties = device.raw.properties();
        if properties.vendor_id != NVIDIA_VENDOR {
            log::info!(
                "hardware video: not on {NAME}: the GPU ({:#06x}) is not NVIDIA's",
                properties.vendor_id
            );
            return None;
        }
        let Some((domain, bus, slot, function)) = properties.pci_bus else {
            log::info!("hardware video: not on {NAME}: the GPU lists no PCI address");
            return None;
        };
        let address = format!("{domain:04x}:{bus:02x}:{slot:02x}.{function:x}");
        let gpu = match Gpu::open(&GpuSelector::PciBusId(address.clone())) {
            Ok(gpu) => Arc::new(gpu),
            Err(error) => {
                log::info!(
                    "hardware video: not on {NAME}: CUDA opens no GPU at {address} ({error})"
                );
                return None;
            }
        };
        let asked = |codec, bit_depth| supported(&gpu, codec, bit_depth);
        let h264 = asked(NvdecCodec::H264, 8).map(|caps| RgfxVideoCodecCaps {
            profiles: RGFX_VIDEO_H264_BASELINE | RGFX_VIDEO_H264_MAIN | RGFX_VIDEO_H264_HIGH,
            formats: RGFX_VIDEO_FORMAT_NV12,
            max_level: H264_TOP_LEVEL,
            max_dpb_pictures: 17,
            max_references: 16,
            ..decode_caps(RGFX_VIDEO_CODEC_H264, &[caps])
        });
        let hevc = {
            let profiles = [
                (
                    asked(NvdecCodec::Hevc, 8),
                    RGFX_VIDEO_HEVC_MAIN,
                    RGFX_VIDEO_FORMAT_NV12,
                ),
                (
                    asked(NvdecCodec::Hevc, 10),
                    RGFX_VIDEO_HEVC_MAIN_10,
                    RGFX_VIDEO_FORMAT_P010,
                ),
            ];
            let answered: Vec<DecodeCapabilities> = (profiles.iter())
                .filter_map(|(caps, ..)| caps.clone())
                .collect();
            (!answered.is_empty()).then(|| {
                let (mut profile_bits, mut formats) = (0, 0);
                for (_, profile, format) in profiles.iter().filter(|(caps, ..)| caps.is_some()) {
                    profile_bits |= profile;
                    formats |= format;
                }
                RgfxVideoCodecCaps {
                    profiles: profile_bits,
                    formats,
                    max_level: HEVC_TOP_LEVEL,
                    max_dpb_pictures: 16,
                    max_references: 15,
                    ..decode_caps(RGFX_VIDEO_CODEC_HEVC, &answered)
                }
            })
        };
        if h264.is_none() && hevc.is_none() {
            log::info!(
                "hardware video: not on {NAME}: it decodes neither H.264 nor HEVC at {address}"
            );
            return None;
        }
        log::info!(
            "hardware video: {NAME} on {} at {address}: H.264 {}, HEVC {}",
            gpu.info().name,
            if h264.is_some() { "yes" } else { "no" },
            match &hevc {
                Some(caps) if caps.profiles & RGFX_VIDEO_HEVC_MAIN_10 != 0 => "Main and Main 10",
                Some(_) => "Main",
                None => "no",
            },
        );
        Some(Self {
            gpu,
            device: device.clone_ref(),
            h264,
            hevc,
            runtime: StdRuntime::shared(),
        })
    }
}

/// What NVDEC answers for `codec` at 4:2:0 and `bit_depth`, where it decodes it; a refused
/// query is logged.
fn supported(gpu: &Gpu, codec: NvdecCodec, bit_depth: u32) -> Option<DecodeCapabilities> {
    match decode_capabilities(gpu, codec, ChromaFormat::Yuv420, bit_depth) {
        Ok(caps) if caps.supported => Some(caps),
        Ok(_) => None,
        Err(error) => {
            log::warn!("hardware video: {NAME}'s caps of {codec:?} at {bit_depth} bits: {error}");
            None
        }
    }
}

/// The caps of `codec`'s decode at the least of the sizes NVDEC answers in `answered`, at least
/// one; its profiles, formats, level and DPB are the caller's.
fn decode_caps(codec: u32, answered: &[DecodeCapabilities]) -> RgfxVideoCodecCaps {
    let least = |of: fn(&DecodeCapabilities) -> u32| answered.iter().map(of).min().unwrap_or(0);
    let most = |of: fn(&DecodeCapabilities) -> u32| answered.iter().map(of).max().unwrap_or(0);
    RgfxVideoCodecCaps {
        codec,
        direction: RGFX_VIDEO_DECODE,
        min_size: [most(|c| c.min_width), most(|c| c.min_height)],
        max_size: [least(|c| c.max_width), least(|c| c.max_height)],
        bitstream_alignment: 1,
        ..RgfxVideoCodecCaps::default()
    }
}

impl GraphicsVideo<VulkanPlatform> for Shared {
    fn caps(&self) -> Vec<RgfxVideoCodecCaps> {
        self.0.h264.into_iter().chain(self.0.hevc).collect()
    }

    fn session(
        &self,
        codec: u32,
        direction: u32,
    ) -> Option<Arc<dyn GraphicsVideoSession<VulkanPlatform>>> {
        let video = &self.0;
        let annex_b = match codec {
            _ if direction != RGFX_VIDEO_DECODE => return None,
            RGFX_VIDEO_CODEC_H264 if video.h264.is_some() => AnnexBCodec::H264,
            RGFX_VIDEO_CODEC_HEVC if video.hevc.is_some() => AnnexBCodec::Hevc,
            _ => return None,
        };
        let decoder = NvdecDecoder {
            decoder: AnnexBDecoder::new(&video.gpu, annex_b, &video.runtime),
            declined: Arc::new(AtomicBool::new(false)),
        };
        let session = HostVideoSession::new(&video.device, codec, NAME, Box::new(decoder))?;
        Some(Arc::new(session))
    }
}

/// One session's decoder.
struct NvdecDecoder {
    decoder: AnnexBDecoder,
    /// Whether a picture of the session found its pages out of CUDA's reach, which is logged
    /// once.
    declined: Arc<AtomicBool>,
}

impl HostDecoder<VulkanPlatform> for NvdecDecoder {
    fn parameter_sets(&mut self, sets: &[u8]) -> Result<(), BackendError> {
        Ok(self.decoder.parameter_sets(sets)?)
    }

    fn decode(
        &mut self,
        au: &[u8],
    ) -> Result<Option<Box<dyn HostPicture<VulkanPlatform>>>, BackendError> {
        let picture = self.decoder.decode(au)?;
        Ok(picture.map(|picture| {
            Box::new(NvdecPicture {
                picture,
                declined: Arc::clone(&self.declined),
            }) as Box<dyn HostPicture<VulkanPlatform>>
        }))
    }

    fn most_unlanded(&self) -> usize {
        AnnexBDecoder::SPARE_PICTURES
    }
}

impl Drop for NvdecDecoder {
    fn drop(&mut self) {
        let sent = self.decoder.bitstreams();
        log::info!(
            "hardware video: {NAME}: a session's access units: {} sent in place, {} gathered",
            sent.in_place,
            sent.gathered
        );
    }
}

/// A picture NVDEC decoded, as the session lands it.
struct NvdecPicture {
    picture: AnnexBPicture,
    declined: Arc<AtomicBool>,
}

impl NvdecPicture {
    /// Writes the picture into `luma` and `chroma`'s pages by the GPU, where both lie in pages a
    /// device reaches: `Ok(true)` once there, `Ok(false)` where the pages are not to be had so.
    fn write_in_place(
        &self,
        size: (u32, u32),
        luma: &OutputPlane<VulkanPlatform>,
        chroma: &OutputPlane<VulkanPlatform>,
    ) -> Result<bool, BackendError> {
        let (Some(luma_pages), Some(chroma_pages)) = (&luma.pages, &chroma.pages) else {
            return Ok(false);
        };
        let (Some(luma_plane), Some(chroma_plane)) = (
            memory_plane(luma, luma_pages),
            memory_plane(chroma, chroma_pages),
        ) else {
            return Err("a plane past the host's address space".into());
        };
        // Late copies keep the planes' mappings with CUDA's registrations of them.
        let hold = Box::new((luma_pages.clone(), chroma_pages.clone()));
        // SAFETY: the planes are the decode's output, which the guest writes no other way until
        // the command buffer that waits for the session's signal completes, and which lie apart
        // (the decode's check); the memories' views stay mapped while `hold` lives.
        let written = unsafe { (self.picture).write_into(size, luma_plane, chroma_plane, hold) };
        match written? {
            Ok(()) => Ok(true),
            Err(why) => {
                if !self.declined.swap(true, Ordering::Relaxed) {
                    log::info!(
                        "hardware video: {NAME}: a session's pictures are copied by the CPU: {why}"
                    );
                }
                Ok(false)
            }
        }
    }
}

/// `plane`, its crop's rows, where it lies in `pages`; `None` past the host's address space.
fn memory_plane<'a>(
    plane: &OutputPlane<VulkanPlatform>,
    pages: &'a PlanePages,
) -> Option<MemoryPlane<'a>> {
    let offset = (pages.buffer_offset.checked_add(plane.offset))
        .and_then(|offset| usize::try_from(offset).ok())?;
    Some(MemoryPlane {
        memory: &*pages.memory,
        offset,
        bytes_per_row: usize::try_from(plane.bytes_per_row).ok()?,
        row_bytes: usize::try_from(plane.row_bytes).ok()?,
        rows: u32::try_from(plane.height).ok()?,
    })
}

impl HostPicture<VulkanPlatform> for NvdecPicture {
    fn land(self: Box<Self>, to: &mut Landing<'_, VulkanPlatform>) -> Result<(), BackendError> {
        let (luma, chroma) = (to.luma, to.chroma);
        let sample = u64::try_from(self.picture.bytes_per_sample()).unwrap_or(u64::MAX);
        if luma.width.checked_mul(sample) != Some(luma.row_bytes) {
            return Err(format!(
                "a picture of {sample}-byte samples for a luma plane of {} pixels in {} bytes",
                luma.width, luma.row_bytes
            )
            .into());
        }
        let (Ok(width), Ok(height)) = (u32::try_from(luma.width), u32::try_from(luma.height))
        else {
            return Err("a plane past 2^32 pixels".into());
        };
        if self.write_in_place((width, height), luma, chroma)? {
            return Ok(());
        }
        let read = self.picture.read_back((width, height))?;
        let source = |plane: &ReadPlane<'_>| SourcePlane {
            at: plane.bytes.as_ptr(),
            bytes_per_row: plane.bytes_per_row,
            row_bytes: plane.bytes_per_row,
            rows: usize::try_from(plane.rows).unwrap_or(0),
        };
        let [luma, chroma] = read.planes();
        // SAFETY: each plane's rows lie in the decoder's staging memory, packed, which `read`
        // holds through the copy and which lies apart from the output's planes.
        unsafe { to.copy_rows(source(&luma), source(&chroma)) }
    }
}
