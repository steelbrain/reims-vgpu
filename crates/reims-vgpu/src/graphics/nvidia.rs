//! The reims-graphics device's hardware video on an NVIDIA GPU ([`open`]): NVDEC, through the
//! NVIDIA crate's decoder of whole access units (`reims_vgpu_video::nvidia::AnnexBDecoder`), in
//! the host's video sessions (`super::video_host`), whose event the host signals once a picture
//! has landed; and NVENC, in the host's encode sessions (`encode`'s module doc).
//!
//! - **Which GPU.** The runtime device's own, by its PCI address, where the device is NVIDIA's
//!   and CUDA opens the GPU there. Anywhere else the arm declines, with one line at info, and the
//!   device's video goes to the next arm (`super::video_choice`). One video a runtime device,
//!   shared by its sessions while one lives.
//! - **The decode caps** are NVDEC's own answers (`cuvidGetDecoderCaps`): H.264's Baseline, Main and
//!   High where NVDEC decodes H.264 at 8 bits, and HEVC's Main where it decodes HEVC at 8 bits
//!   and Main 10 where it does at 10, each into its format (NV12, P010), at the least of their
//!   sizes. NVDEC answers no level, so the codec's top one: H.264's 6.2 and HEVC's 6.2.
//!   The DPB is the codecs' own bound, 16 references and the current picture for H.264, 15 and
//!   the current picture for HEVC. A picture references as many as H.264's lists name, 16
//!   frames; an HEVC one 8 in its current reference sets together, the most with which every
//!   picture decodes, as `CUVIDHEVCPICPARAMS` holds 8 in each of them (the `nvidia` module's
//!   "NVDEC takes 8"), which the guest's video driver checks each picture against before it
//!   schedules it.
//! - **The access unit is read in place** at any place and length (`bitstream_alignment` 1):
//!   the codec crate parses it where it lies in the guest's buffer, and NVDEC is handed the run of
//!   it that holds the picture's slices where they are one run, and copies it within
//!   `cuvidDecodePicture`; otherwise the slices are gathered first. Each session logs, at its
//!   end, how many of its pictures went each way.
//! - **A picture lands by one GPU copy**, the crop's rows alone at the plane's pitch, one
//!   `cuMemcpy2D` a plane from NVDEC's mapped picture, waited for before the session signals. So
//!   no layout is asked of the guest.
//!   - **In the output's device caches**, where CUDA imports the device's exportable buffers:
//!     learned once as the video opens, by a buffer of a 4K picture's bytes made exportable and
//!     imported, logged at info with the make's time ([`imports_caches`]). The video then lands
//!     in device caches (`RGFX_VIDEO_CAPS_OUTPUT_CACHED`) and has the device's laid-out caches of
//!     two planes, the outputs' NV12 and P010, made exportable as they are made, so the guest
//!     asks its outputs cached, and a picture is copied within the GPU into the cache's memory,
//!     imported once a cache and let go of as the cache goes
//!     ([`VideoPlane::cache`](super::decode::VideoPlane::cache)). Once the copy is done, each
//!     cache it wrote is said written by another API (`Buffer::external_write_done`), for the
//!     device's later reads of it to see the picture.
//!   - **In the output's pages**, where both planes lie in pages a device reaches in place
//!     ([`PlanePages`]): their view registered with CUDA, once a mapping, and
//!     written by DMA.
//!
//!   Elsewhere, and where CUDA refuses the import or the view (logged once a session with why),
//!   the picture is read back into the decoder's staging memory and copied by the CPU.
//! - **An encode's frame** is copied by the GPU into an exportable buffer CUDA imported, which
//!   NVENC reads, where CUDA imports the device's exportable buffers (`encode`'s module doc).
//! - **A picture the GPU does not finish** by its deadline fails its landing, and every later
//!   picture of the decoder; its copy keeps the planes' mappings and CUDA's registrations of them
//!   until the GPU is done, but not the guest's buffers, so a copy that ends that late may write
//!   pages the guest has used again since.
//! - **VideoToolbox's paravirtual decoders** (`NvGuestDecoder`, `super::vt`): an arm64 guest's
//!   decode, on a decoder of whole access units of its own, lands in the IOSurface its kext lent,
//!   laid out as the kext read the surface and checked against its memory
//!   ([`GuestTarget`]): written there by the
//!   GPU, one `cuMemcpy2D` a plane into the view of its pages registered with CUDA, as into an
//!   output's pages above, else read back and copied by the CPU, each reason logged once a
//!   decoder. So every decode codec of the caps has one, and the device offers `VT_MESSAGES`
//!   where NVDEC decodes H.264 or HEVC.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Weak};

use reims_vgpu_graphics::wire::{
    RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_DECODE, RGFX_VIDEO_ENCODE,
    RGFX_VIDEO_FORMAT_NV12, RGFX_VIDEO_FORMAT_P010, RGFX_VIDEO_H264_BASELINE, RGFX_VIDEO_H264_HIGH,
    RGFX_VIDEO_H264_MAIN, RGFX_VIDEO_HEVC_MAIN, RGFX_VIDEO_HEVC_MAIN_10, RgfxVideoCodecCaps,
    RgfxVideoEncodeCaps,
};
use reims_vgpu_video::core::backend::BackendError as VideoError;
use reims_vgpu_video::core::device_memory::{
    DeviceImports, GuestTarget, MemoryIdentity, TargetPlane,
};
use reims_vgpu_video::core::runtime::Runtime;
use reims_vgpu_video::nvidia::{
    AnnexBCodec, AnnexBDecoder, AnnexBPicture, ChromaFormat, DecodeCapabilities, DevicePlane,
    EncodeProfileSupport, Gpu, GpuSelector, ImportedBuffer, MemoryPlane, NvdecCodec, ReadPlane,
    decode_capabilities, encode_profiles,
};
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::core::platform::Handle as _;
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::resource::Buffer as _;
use reims_vgpu_vulkan::core::raise::OnFault;
use reims_vgpu_vulkan::core::sync::Mutex;
use reims_vgpu_vulkan::gpu::device::VulkanDevice;
use reims_vgpu_vulkan::gpu::resource::VulkanBuffer;

use super::decode::{GraphicsVideo, GraphicsVideoSession, GuestDecoder, GuestOutput, PlanePages};
use super::video_host::{
    BackendError, DecodeOutput, HostDecoder, HostEncodeSession, HostPicture, HostVideoSession,
    Landing, OutputPlane, SourcePlane,
};

mod encode;

/// The PCI vendor ID of NVIDIA's GPUs.
const NVIDIA_VENDOR: u32 = 0x10de;

/// What the log calls the arm.
const NAME: &str = "NVDEC";

/// What the log calls its encode.
const ENCODE_NAME: &str = "NVENC";

/// H.264's top level, 6.2 (`level_idc`).
const H264_TOP_LEVEL: u32 = 62;

/// HEVC's top level, 6.2 (`general_level_idc`, 30 times the level).
const HEVC_TOP_LEVEL: u32 = 186;

/// `MTLResourceStorageModePrivate`, in `MTLResourceOptions`: a device cache's storage.
const STORAGE_PRIVATE: u64 = 2 << 4;

/// The bytes of the buffer the open makes exportable and imports, to learn whether caches can
/// be so: a 4K NV12 picture's, 3840 by 2160 luma samples and half as many chroma.
const PROBE_BYTES: u64 = 3840 * 2160 * 3 / 2;

/// The most imports of device caches kept at once: a few decoders' output pools.
const MOST_IMPORTS: usize = 64;

/// The video of each runtime device, by its registry ID, while one lives.
type MadeVideos = Vec<(u64, Weak<NvidiaVideo>)>;

/// NVDEC's video for `device`'s sessions: the one they share, made on first use, the NVIDIA
/// crate's threads and waits `runtime`'s, every session kind's, unless a video opened before
/// handed it another; `None`, with a line at info, where the device is not an NVIDIA GPU CUDA
/// opens, or NVDEC decodes and NVENC encodes neither H.264 nor HEVC there.
pub(crate) fn open(
    device: &VulkanDevice,
    runtime: &Arc<dyn Runtime>,
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
            let video = Arc::new(NvidiaVideo::new(device, runtime)?);
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
    /// The profiles NVENC encodes, and the encode entries they give (`encode`'s module doc).
    encoders: Vec<EncodeProfileSupport>,
    encode: Vec<RgfxVideoCodecCaps>,
    /// The video crates' runtime: the threads and the clock of the NVIDIA crate's waits.
    runtime: Arc<dyn Runtime>,
    /// CUDA's imports of the device caches pictures land in, by each cache's identity; `None`
    /// where CUDA imports no exportable buffer of the device ([`imports_caches`]), and pictures
    /// land in the surfaces' pages.
    imports: Option<DeviceImports<ImportedBuffer>>,
}

impl NvidiaVideo {
    fn new(device: &VulkanDevice, runtime: &Arc<dyn Runtime>) -> Option<Self> {
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
        // Every session kind's threads and waits in the NVIDIA crate run on `runtime`, an
        // encoder's made before any decoder's too, and the crate arms the runtime's exit gate as
        // CUDA loads, as a VA display takes its runtime as it opens.
        reims_vgpu_video::nvidia::use_runtime(runtime);
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
                    max_references: 8,
                    ..decode_caps(RGFX_VIDEO_CODEC_HEVC, &answered)
                }
            })
        };
        let encoders = match encode_profiles(Arc::clone(&gpu)) {
            Ok(encoders) => encoders,
            Err(error) => {
                log::info!("hardware video: no {ENCODE_NAME} at {address}: {error}");
                Vec::new()
            }
        };
        let encode = encode::encode_caps_of(&encoders);
        if h264.is_none() && hevc.is_none() && encode.is_empty() {
            log::info!(
                "hardware video: not on {NAME}: it decodes and {ENCODE_NAME} encodes neither \
                 H.264 nor HEVC at {address}"
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
        let encodes = |codec| {
            (encode.iter()).find(|c| c.codec == codec).map_or_else(
                || "no".to_owned(),
                |c| {
                    format!(
                        "profiles {:#x} up to {}x{}",
                        c.profiles, c.max_size[0], c.max_size[1]
                    )
                },
            )
        };
        log::info!(
            "hardware video: {ENCODE_NAME} at {address}: H.264 {}, HEVC {}",
            encodes(RGFX_VIDEO_CODEC_H264),
            encodes(RGFX_VIDEO_CODEC_HEVC),
        );
        let imports = imports_caches(device, &gpu, runtime);
        Some(Self {
            gpu,
            device: device.clone_ref(),
            h264,
            hevc,
            encoders,
            encode,
            runtime: Arc::clone(runtime),
            imports,
        })
    }
}

/// CUDA's imports of `device`'s device caches, where it imports one: a buffer of a 4K picture's
/// bytes made exportable, its memory exported and imported on `gpu`, once, as the video opens.
/// The line at info says which way pictures land, with the make's and the import's times, the
/// make's against a buffer made as any other; `None` where any step is refused.
fn imports_caches(
    device: &VulkanDevice,
    gpu: &Arc<Gpu>,
    runtime: &Arc<dyn Runtime>,
) -> Option<DeviceImports<ImportedBuffer>> {
    let declined = |why: &str| {
        log::info!(
            "hardware video: {NAME}: pictures land in surfaces' pages, not in device caches: {why}"
        );
        None
    };
    let start = runtime.now();
    let made =
        device.new_exportable_buffer_of_length(OnFault::Return, PROBE_BYTES, STORAGE_PRIVATE, None);
    let make = runtime.elapsed_since(start);
    let buffer = match made {
        Ok(Some(buffer)) => buffer,
        Ok(None) => return declined("the device makes no exportable buffer"),
        Err(raise) => return declined(&format!("an exportable buffer's make: {raise}")),
    };
    let start = runtime.now();
    let plain = device.new_buffer_of_length(OnFault::Return, PROBE_BYTES, STORAGE_PRIVATE);
    let plain_make = runtime.elapsed_since(start);
    drop(plain);
    let start = runtime.now();
    let imported = import(&buffer, gpu);
    let import_time = runtime.elapsed_since(start);
    if let Err(why) = imported {
        return declined(&why);
    }
    log::info!(
        "hardware video: {NAME}: pictures land in device caches, by a copy within the GPU: an \
         exportable cache of {} MiB made in {} µs ({} µs as any other buffer), imported in {} µs",
        PROBE_BYTES >> 20,
        make.as_micros(),
        plain_make.as_micros(),
        import_time.as_micros()
    );
    Some(DeviceImports::new(
        runtime,
        "NVDEC's device caches",
        MOST_IMPORTS,
    ))
}

/// `buffer`'s memory imported on `gpu`; `Err` says why not.
fn import(buffer: &VulkanBuffer, gpu: &Arc<Gpu>) -> Result<ImportedBuffer, String> {
    let exported = buffer.export_memory()?;
    ImportedBuffer::import(
        gpu,
        exported.fd,
        exported.allocation_size,
        exported.offset,
        exported.length,
        exported.dedicated,
    )
    .map_err(|error| format!("CUDA's import of an exportable buffer: {error}"))
}

/// What NVDEC answers for `codec` at 4:2:0 and `bit_depth`, where it decodes it; a refused
/// query is logged.
fn supported(gpu: &Gpu, codec: NvdecCodec, bit_depth: u32) -> Option<DecodeCapabilities> {
    match decode_capabilities(gpu, codec, ChromaFormat::Yuv420, bit_depth) {
        Ok(caps) if caps.supported => Some(caps),
        Ok(_) => None,
        Err(error) => {
            log::warn!(
                kind:% = error;
                "hardware video: {NAME}'s caps of {codec:?} at {bit_depth} bits: {error}"
            );
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
        let decode = self.0.h264.into_iter().chain(self.0.hevc);
        decode.chain(self.0.encode.iter().copied()).collect()
    }

    fn encode_caps(&self, codec: &RgfxVideoCodecCaps) -> RgfxVideoEncodeCaps {
        encode::encode_caps(codec, &self.0.encoders)
    }

    fn session(
        &self,
        codec: u32,
        direction: u32,
    ) -> Option<Arc<dyn GraphicsVideoSession<VulkanPlatform>>> {
        let video = &self.0;
        if direction == RGFX_VIDEO_ENCODE {
            let entry = (video.encode.iter()).find(|c| c.codec == codec)?;
            let caps = encode::encode_caps(entry, &video.encoders);
            let filler = match video.imports {
                Some(_) => Ok(encode::Filler::new(
                    &video.device,
                    &video.gpu,
                    &video.runtime,
                )),
                None => Err("CUDA imports no exportable buffer of the device".to_owned()),
            };
            let encoder = encode::NvencHostEncoder::new(
                &video.gpu,
                codec,
                &video.encoders,
                entry,
                &caps,
                filler,
            );
            let session =
                HostEncodeSession::new(&video.device, codec, ENCODE_NAME, Box::new(encoder))?;
            return Some(Arc::new(session));
        }
        let annex_b = match codec {
            _ if direction != RGFX_VIDEO_DECODE => return None,
            RGFX_VIDEO_CODEC_H264 if video.h264.is_some() => AnnexBCodec::H264,
            RGFX_VIDEO_CODEC_HEVC if video.hevc.is_some() => AnnexBCodec::Hevc,
            _ => return None,
        };
        let decoder = NvdecDecoder {
            decoder: AnnexBDecoder::new(&video.gpu, annex_b, &video.runtime),
            video: Arc::clone(video),
            declined: Arc::new(AtomicBool::new(false)),
        };
        let session = HostVideoSession::new(&video.device, codec, NAME, Box::new(decoder))?;
        Some(Arc::new(session))
    }

    fn lands_in_device_caches(&self) -> bool {
        self.0.imports.is_some()
    }

    fn imports_device_caches(&self) -> bool {
        self.0.imports.is_some()
    }

    /// NVDEC's decoders land in any memory: by the GPU, into its view registered with CUDA, or
    /// by the CPU.
    fn guest_decodes(&self) -> bool {
        true
    }

    fn guest_decoder(&self, codec: u32) -> Option<Box<dyn GuestDecoder>> {
        let video = &self.0;
        let annex_b = match codec {
            RGFX_VIDEO_CODEC_H264 if video.h264.is_some() => AnnexBCodec::H264,
            RGFX_VIDEO_CODEC_HEVC if video.hevc.is_some() => AnnexBCodec::Hevc,
            _ => return None,
        };
        Some(Box::new(NvGuestDecoder {
            decoder: AnnexBDecoder::new(&video.gpu, annex_b, &video.runtime),
            said: Vec::new(),
        }))
    }
}

/// One session's decoder.
struct NvdecDecoder {
    decoder: AnnexBDecoder,
    /// The video it decodes on, whose imports of device caches its pictures land in.
    video: Arc<NvidiaVideo>,
    /// Whether a picture of the session found its pages or its caches out of CUDA's reach, which
    /// is logged once.
    declined: Arc<AtomicBool>,
}

impl HostDecoder<VulkanPlatform> for NvdecDecoder {
    fn parameter_sets(&mut self, sets: &[u8]) -> Result<(), BackendError> {
        Ok(self.decoder.parameter_sets(sets)?)
    }

    fn decode(
        &mut self,
        au: &[u8],
        _output: &DecodeOutput<VulkanPlatform>,
    ) -> Result<Option<Box<dyn HostPicture<VulkanPlatform>>>, BackendError> {
        let picture = self.decoder.decode(au)?;
        Ok(picture.map(|picture| {
            Box::new(NvdecPicture {
                picture,
                video: Arc::clone(&self.video),
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
    video: Arc<NvidiaVideo>,
    declined: Arc<AtomicBool>,
}

impl NvdecPicture {
    /// Logs, once a session, and at info once per process (`kind`), why its pictures are copied
    /// by the CPU.
    fn decline(&self, why: &str) {
        if !self.declined.swap(true, Ordering::Relaxed) {
            log::info!(
                kind = why;
                "hardware video: {NAME}: a session's pictures are copied by the CPU: {why}"
            );
        }
    }

    /// Writes the picture into `luma` and `chroma`'s device caches by a copy within the GPU,
    /// where both lie in caches CUDA imports: `Ok(true)` once there, `Ok(false)` where the caches
    /// are not to be had so.
    fn write_in_cache(
        &self,
        size: (u32, u32),
        luma: &OutputPlane<VulkanPlatform>,
        chroma: &OutputPlane<VulkanPlatform>,
    ) -> Result<bool, BackendError> {
        let Some(imports) = &self.video.imports else {
            return Ok(false);
        };
        let (Some(luma_cache), Some(chroma_cache)) = (&luma.cache, &chroma.cache) else {
            return Ok(false);
        };
        let gpu = &self.video.gpu;
        let imported = |plane: &OutputPlane<VulkanPlatform>, cache: &MemoryIdentity| {
            imports.get_or_make(
                cache,
                0,
                || import(&plane.buffer, gpu),
                || "the device cache went".to_owned(),
            )
        };
        let luma_import = match imported(luma, luma_cache) {
            Ok(import) => import,
            Err(why) => {
                self.decline(&why);
                return Ok(false);
            }
        };
        let chroma_import = if chroma_cache.key() == luma_cache.key() {
            Arc::clone(&luma_import)
        } else {
            match imported(chroma, chroma_cache) {
                Ok(import) => import,
                Err(why) => {
                    self.decline(&why);
                    return Ok(false);
                }
            }
        };
        let (Some(luma_plane), Some(chroma_plane)) = (
            device_plane(luma, &luma_import),
            device_plane(chroma, &chroma_import),
        ) else {
            return Err("a plane past the host's address space".into());
        };
        // SAFETY: the planes are the decode's output, which the guest writes no other way until
        // the command buffer that waits for the session's signal completes, and which lie apart
        // (the decode's check); the imports, which a late copy keeps, hold the caches' memory.
        let written = unsafe {
            (self.picture).write_into_device(size, luma_plane, chroma_plane, Box::new(()))
        };
        match written? {
            // The copies are done (the write waited for them): the device's reads of each cache
            // see them once the platform acquired what CUDA wrote, said before the session's
            // signal lets a command buffer read the picture.
            Ok(()) => {
                luma.buffer.external_write_done()?;
                if chroma_cache.key() != luma_cache.key() {
                    chroma.buffer.external_write_done()?;
                }
                Ok(true)
            }
            Err(why) => {
                self.decline(&why);
                Ok(false)
            }
        }
    }

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
                self.decline(&why);
                Ok(false)
            }
        }
    }
}

/// `plane`, its crop's rows, where it lies in its device cache's import; `None` past the host's
/// address space.
fn device_plane<'a>(
    plane: &OutputPlane<VulkanPlatform>,
    import: &'a Arc<ImportedBuffer>,
) -> Option<DevicePlane<'a>> {
    Some(DevicePlane {
        buffer: import,
        offset: usize::try_from(plane.offset).ok()?,
        bytes_per_row: usize::try_from(plane.bytes_per_row).ok()?,
        row_bytes: usize::try_from(plane.row_bytes).ok()?,
        rows: u32::try_from(plane.height).ok()?,
    })
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
        if self.write_in_cache((width, height), luma, chroma)? {
            return Ok(());
        }
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

/// A decoder of VideoToolbox's paravirtual path ([`GuestDecoder`]), whose pictures land in the
/// surface the guest lent (module doc): written there by the GPU where CUDA registers the view of
/// its pages, else read back and copied by the CPU.
struct NvGuestDecoder {
    decoder: AnnexBDecoder,
    /// The reasons its pictures were copied by the CPU, each logged once.
    said: Vec<String>,
}

impl NvGuestDecoder {
    /// Logs, once a decoder, why a picture was copied by the CPU.
    fn copied_because(&mut self, why: String) {
        if !self.said.contains(&why) {
            log::info!(
                kind:% = why;
                "hardware video: {NAME}: a VideoToolbox picture copied by the CPU: {why}"
            );
            self.said.push(why);
        }
    }
}

impl GuestDecoder for NvGuestDecoder {
    fn decode(
        &mut self,
        parameter_sets: &[u8],
        au: &[u8],
        output: &GuestOutput<'_>,
    ) -> Result<bool, VideoError> {
        if !parameter_sets.is_empty() {
            (self.decoder.parameter_sets(parameter_sets)).map_err(video_error)?;
        }
        let Some(picture) = self.decoder.decode(au).map_err(video_error)? else {
            return Ok(false);
        };
        let target =
            GuestTarget::new(&output.layout, output.memory).map_err(VideoError::Unsupported)?;
        let sample = picture.bytes_per_sample();
        if target.pixel_format().bytes_per_sample() != sample {
            return Err(VideoError::Unsupported(format!(
                "a picture of {sample}-byte samples for a '{}' surface",
                output.layout.pixel_format
            )));
        }
        let size = (target.width(), target.height());
        let planes = target.planes();
        let [luma, chroma] = planes.map(|plane| MemoryPlane {
            memory: target.memory(),
            offset: plane.offset,
            bytes_per_row: plane.bytes_per_row,
            row_bytes: plane.row_bytes,
            rows: plane.rows,
        });
        // SAFETY: the planes are the surface's (`GuestTarget`'s check: apart, in order, each
        // inside its memory), which the guest's kext holds lent, its pages wired, until the
        // host's `rels`, sent once this returns, and which nothing of the host's writes
        // meanwhile; the registration the write keeps holds the view of its pages mapped.
        let written = unsafe { picture.write_into(size, luma, chroma, Box::new(())) };
        match written.map_err(video_error)? {
            Ok(()) => return Ok(true),
            Err(why) => self.copied_because(why),
        }
        let read = picture.read_back(size).map_err(video_error)?;
        let view = (output.memory.host_view())
            .map_err(|error| VideoError::Malfunction(error.to_string()))?;
        let length = output.memory.length();
        if (view.offset.checked_add(length)).is_none_or(|end| end > view.size) {
            return Err(VideoError::Malfunction(
                "the host's view is shorter than the surface".to_owned(),
            ));
        }
        // SAFETY: the view's `size` bytes from `first` are the host's mapping of the surface's
        // pages, which `view.hold` keeps mapped until it drops after the copy, and the surface's
        // `length` bytes from `offset` lie inside them (checked above); the guest's kext holds
        // the surface lent, its pages wired, until the host's `rels`, sent once this returns.
        let surface =
            unsafe { std::slice::from_raw_parts_mut(view.first.as_ptr().add(view.offset), length) };
        copy_planes(&read.planes(), &planes, surface).map_err(VideoError::Malfunction)?;
        drop(view);
        Ok(true)
    }
}

impl Drop for NvGuestDecoder {
    fn drop(&mut self) {
        let sent = self.decoder.bitstreams();
        log::info!(
            "hardware video: {NAME}: a VideoToolbox session's access units: {} sent in place, {} \
             gathered",
            sent.in_place,
            sent.gathered
        );
    }
}

/// Copies the rows `read` holds, luma then chroma, into `planes` of `surface`, the surface's
/// bytes from its first: each row as far as the plane's rows hold it, as many rows as both have.
/// `Err` says where a plane's rows lie past the surface; nothing is written then.
fn copy_planes(
    read: &[ReadPlane<'_>; 2],
    planes: &[TargetPlane; 2],
    surface: &mut [u8],
) -> Result<(), String> {
    let span = |read: &ReadPlane<'_>, plane: &TargetPlane| {
        let rows = usize::try_from(read.rows.min(plane.rows)).unwrap_or(usize::MAX);
        let row_bytes = read.bytes_per_row.min(plane.row_bytes);
        let end = match rows.checked_sub(1) {
            None => Some(plane.offset),
            Some(past) => (past.checked_mul(plane.bytes_per_row))
                .and_then(|bytes| bytes.checked_add(plane.offset))
                .and_then(|bytes| bytes.checked_add(row_bytes)),
        };
        let sourced = rows
            .checked_mul(read.bytes_per_row)
            .is_some_and(|bytes| bytes <= read.bytes.len());
        match end {
            Some(end) if end <= surface.len() && sourced => Ok((rows, row_bytes)),
            _ => Err(format!(
                "a plane at {} of {rows} rows {} bytes apart past the surface's {} bytes",
                plane.offset,
                plane.bytes_per_row,
                surface.len()
            )),
        }
    };
    let spans = [span(&read[0], &planes[0])?, span(&read[1], &planes[1])?];
    for ((read, plane), (rows, row_bytes)) in read.iter().zip(planes).zip(spans) {
        for row in 0..rows {
            let from = &read.bytes[row * read.bytes_per_row..][..row_bytes];
            surface[plane.offset + row * plane.bytes_per_row..][..row_bytes].copy_from_slice(from);
        }
    }
    Ok(())
}

/// What a VideoToolbox decoder sees of `error`: a stream that does not decode, or whose picture
/// NVDEC failed, as bad data; one NVDEC cannot decode as unsupported; a leading picture the
/// stream's random access skips as skipped; and the rest as the decoder's malfunction, a
/// picture smaller than NVDEC decodes among them, as Apple's decoders answer a size they do not
/// decode.
fn video_error(error: reims_vgpu_video::nvidia::Error) -> VideoError {
    use reims_vgpu_video::nvidia::Error;
    match error {
        Error::Bitstream(_) | Error::DecodeFailed(_) => VideoError::BadData(error.to_string()),
        Error::Unsupported(why) => VideoError::Unsupported(why),
        Error::SkippedLeadingPicture(why) => VideoError::SkippedLeadingPicture(why),
        error => VideoError::Malfunction(error.to_string()),
    }
}

#[cfg(test)]
mod tests;
