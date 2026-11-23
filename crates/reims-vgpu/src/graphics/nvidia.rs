//! The reims-graphics device's hardware video on an NVIDIA GPU ([`open`]): NVDEC, through the
//! NVIDIA crate's decoder of whole access units (`reims_vgpu_video::nvidia::AnnexBDecoder`), in
//! the host's video sessions (`super::video_host`), whose event the host signals once a picture
//! has landed; and NVENC, in the host's encode sessions (`encode`'s module doc).
//!
//! - **Which GPU.** The platform device's own, by its PCI address ([`NvidiaGpu`]), where the
//!   device is NVIDIA's and CUDA opens the GPU there. Anywhere else the arm declines, with one
//!   line at info, and the device's video goes to the next arm (`super::video_choice`). One video
//!   a platform device, shared by its sessions while one lives, the rt platform's: NVENC's input
//!   is handed to CUDA by the platform's release after each frame's copy (`encode::fill`).
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
//!   - **HEVC's 4:4:4 at 8 bits**, the range extensions' streams the codec crate decodes, has a
//!     paravirtual session where NVDEC decodes HEVC at 4:4:4 and that depth into its three
//!     planes, at the stream's size; elsewhere the session is refused at its start, the reason
//!     named (`GraphicsVideo::guest_declines`), and the guest decodes in software. The picture
//!     lands in the surface's 4:4:4 formats ('444v', '444f'), its luma as it lies and its Cb and
//!     Cr interleaved into the surface's second plane by the CPU, from the picture read back:
//!     the write of two planes refuses a picture of three, and that reason is logged.
//! - **VideoToolbox's paravirtual encoders** (`encode`'s `guest`): an arm64 guest's encode, of
//!   each codec NVENC encodes a profile of, on an NVENC encoder of its own, each source frame read
//!   from the IOSurface its kext lent: copied by the GPU into memory NVENC reads, as an x86
//!   guest's frame is, else read by the CPU.

use std::any::Any;
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
use reims_vgpu_video::core::device_wait::KeptMark;
use reims_vgpu_video::core::runtime::Runtime;
use reims_vgpu_video::nvidia::{
    AnnexBCodec, AnnexBDecoder, AnnexBPicture, ChromaFormat, DecodeCapabilities, DevicePlane,
    EncodeProfileSupport, Gpu, GpuSelector, ImportedBuffer, MemoryPlane, NvdecCodec, ReadPlane,
    SurfaceFormat, decode_capabilities, encode_profiles,
};
use reims_vgpu_video::paravirt::{Encoder, EncoderConfig, Stream};
use reims_vgpu_vulkan::core::platform::Handle as _;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
use reims_vgpu_vulkan::core::platform::resource::{Buffer as _, ExportedMemory};
use reims_vgpu_vulkan::core::raise::OnFault;
use reims_vgpu_vulkan::core::sync::Mutex;

use super::decode::{
    GraphicsVideo, GraphicsVideoSession, GuestDecoder, GuestEncoder, GuestOutput, PlanePages,
};
use super::video_host::{
    BackendError, DecodeOutput, Extent, Failed, HostDecoder, HostEncodeSession, HostPicture,
    HostVideoSession, Landing, OutputPlane, SourcePlane, picture_error,
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

/// The video of each platform device, by its registry ID, while one lives: an `NvidiaVideo` of
/// the device's platform.
type MadeVideos = Vec<(u64, Weak<dyn Any + Send + Sync>)>;

/// What the arm reads of a platform device's GPU ([`open`]).
#[derive(Debug, Clone, Copy)]
pub(crate) struct NvidiaGpu {
    /// The device's registry ID, which keys its one video.
    pub(crate) registry_id: u64,
    /// The GPU's PCI vendor ID.
    pub(crate) vendor_id: u32,
    /// The GPU's PCI domain, bus, device and function; `None` where the device lists none.
    pub(crate) pci: Option<(u32, u32, u32, u32)>,
}

/// NVDEC's video for `device`'s sessions, whose GPU is `gpu`: the one they share, made on first
/// use, the NVIDIA crate's threads and waits `runtime`'s, every session kind's, unless a video
/// opened before handed it another; NVENC's encodes with it. `None`, with a line at info, where
/// the device is not an NVIDIA GPU CUDA opens, or NVDEC decodes and NVENC encodes neither H.264
/// nor HEVC there.
pub(crate) fn open<P: Platform>(
    device: &P::Device,
    gpu: NvidiaGpu,
    runtime: &Arc<dyn Runtime>,
) -> Option<Arc<dyn GraphicsVideo<P>>> {
    static MADE: LazyLock<Mutex<MadeVideos>> = LazyLock::new(|| Mutex::new(Vec::new()));
    let mut made = MADE.lock();
    made.retain(|(_, video)| video.strong_count() > 0);
    let kept = (made.iter())
        .filter(|(id, _)| *id == gpu.registry_id)
        .find_map(|(_, video)| video.upgrade()?.downcast::<NvidiaVideo<P>>().ok());
    let video = match kept {
        Some(video) => video,
        None => {
            let video = Arc::new(NvidiaVideo::<P>::new(device, gpu, runtime)?);
            let any: Arc<dyn Any + Send + Sync> = Arc::<NvidiaVideo<P>>::clone(&video);
            made.push((gpu.registry_id, Arc::downgrade(&any)));
            video
        }
    };
    Some(Arc::new(Shared(video)))
}

/// A platform device's NVDEC, as the graphics device sees it.
struct Shared<P: Platform>(Arc<NvidiaVideo<P>>);

/// NVDEC on a platform device's GPU, and what it answers.
struct NvidiaVideo<P: Platform> {
    gpu: Arc<Gpu>,
    /// The platform device, whose queues copy staged planes into device caches.
    device: P::Device,
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

impl<P: Platform> NvidiaVideo<P> {
    fn new(device: &P::Device, facts: NvidiaGpu, runtime: &Arc<dyn Runtime>) -> Option<Self> {
        if facts.vendor_id != NVIDIA_VENDOR {
            log::info!(
                "hardware video: not on {NAME}: the GPU ({:#06x}) is not NVIDIA's",
                facts.vendor_id
            );
            return None;
        }
        let Some((domain, bus, slot, function)) = facts.pci else {
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
        let imports = imports_caches::<P>(device, &gpu, runtime);
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

impl<P: Platform> NvidiaVideo<P> {
    /// A new encode session's GPU filler, where CUDA imports the device's exportable buffers
    /// (`encode`'s module doc), or why it has none.
    fn filler(&self) -> Result<encode::Filler<P>, String> {
        match self.imports {
            Some(_) => Ok(encode::Filler::new(&self.device, &self.gpu, &self.runtime)),
            None => Err("CUDA imports no exportable buffer of the device".to_owned()),
        }
    }
}

/// CUDA's imports of `device`'s device caches, where it imports one: a buffer of a 4K picture's
/// bytes made exportable, its memory exported and imported on `gpu`, once, as the video opens.
/// The line at info says which way pictures land, with the make's and the import's times, the
/// make's against a buffer made as any other; `None` where any step is refused.
fn imports_caches<P: Platform>(
    device: &P::Device,
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
    let imported = import::<P>(&buffer, gpu);
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
fn import<P: Platform>(buffer: &P::Buffer, gpu: &Arc<Gpu>) -> Result<ImportedBuffer, String> {
    import_exported(buffer.export_memory()?, gpu)
}

/// The memory `exported` names imported on `gpu`; `Err` says why not.
fn import_exported(exported: ExportedMemory, gpu: &Arc<Gpu>) -> Result<ImportedBuffer, String> {
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

/// Why NVDEC's decoders take none of `stream`'s pictures, an HEVC stream at 4:4:4: NVDEC's caps
/// of HEVC at 4:4:4 and the stream's depth, which are to say it decodes it, into its three
/// planes (`SurfaceFormat::Yuv444` at 8 bits, `Yuv444_16Bit` past), at the stream's size; `None`
/// where they do.
fn declines_4_4_4(gpu: &Gpu, stream: &Stream) -> Option<String> {
    let bits = u32::from(stream.luma_bits);
    let caps = match decode_capabilities(gpu, NvdecCodec::Hevc, ChromaFormat::Yuv444, bits) {
        Ok(caps) => caps,
        Err(error) => {
            return Some(format!(
                "{NAME}'s caps of HEVC 4:4:4 at {bits} bits: {error}"
            ));
        }
    };
    let output = if bits > 8 {
        SurfaceFormat::Yuv444_16Bit
    } else {
        SurfaceFormat::Yuv444
    };
    let (width, height) = (stream.width, stream.height);
    if !caps.supported {
        Some(format!("{NAME} does not decode HEVC 4:4:4 at {bits} bits"))
    } else if !caps.output_formats.contains(&output) {
        Some(format!(
            "{NAME} writes HEVC 4:4:4 at {bits} bits as {:?}, not {output:?}",
            caps.output_formats
        ))
    } else if width < caps.min_width || height < caps.min_height {
        Some(format!(
            "{NAME} decodes HEVC 4:4:4 from {}x{}, not {width}x{height}",
            caps.min_width, caps.min_height
        ))
    } else if width > caps.max_width
        || height > caps.max_height
        || u64::from(width.div_ceil(16)) * u64::from(height.div_ceil(16))
            > u64::from(caps.max_macroblocks)
    {
        Some(format!(
            "{NAME} decodes HEVC 4:4:4 up to {}x{} ({} macroblocks), not {width}x{height}",
            caps.max_width, caps.max_height, caps.max_macroblocks
        ))
    } else {
        None
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

impl<P: Platform> GraphicsVideo<P> for Shared<P> {
    fn caps(&self) -> Vec<RgfxVideoCodecCaps> {
        let decode = self.0.h264.into_iter().chain(self.0.hevc);
        decode.chain(self.0.encode.iter().copied()).collect()
    }

    fn encode_caps(&self, codec: &RgfxVideoCodecCaps) -> RgfxVideoEncodeCaps {
        encode::encode_caps(codec, &self.0.encoders)
    }

    fn session(&self, codec: u32, direction: u32) -> Option<Arc<dyn GraphicsVideoSession<P>>> {
        let video = &self.0;
        if direction == RGFX_VIDEO_ENCODE {
            let entry = (video.encode.iter()).find(|c| c.codec == codec)?;
            let caps = encode::encode_caps(entry, &video.encoders);
            let encoder = encode::NvencHostEncoder::new(
                &video.gpu,
                codec,
                &video.encoders,
                entry,
                &caps,
                video.filler(),
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
        let mut annex_b = AnnexBDecoder::new(&video.gpu, annex_b, &video.runtime);
        // A picture NVDEC failed lands as NVDEC left it, and the pictures after it decode from
        // it, as a Mac's hardware decoders hand them on, H.264's and HEVC's: rather than as zeros
        // where AppleGVA's H.264 path reports no failure (docs/video/bugs.md, D25), or as a
        // failed command buffer, which AppleGVA's HEVC path answers for the rest of the session.
        annex_b.copy_failed_pictures();
        let decoder = NvdecDecoder {
            decoder: annex_b,
            video: Arc::clone(video),
            declined: Arc::new(AtomicBool::new(false)),
        };
        let clock = Arc::clone(&video.runtime);
        let session = HostVideoSession::new(&video.device, clock, codec, NAME, Box::new(decoder))?;
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

    /// One for each codec NVENC encodes a profile of (`encode::guest_encoders`).
    fn guest_encoders(&self) -> Vec<Encoder> {
        encode::guest_encoders(&self.0.encoders)
    }

    fn guest_encoder(
        &self,
        config: &EncoderConfig,
    ) -> Result<Box<dyn GuestEncoder<P>>, VideoError> {
        let video = &self.0;
        let encoder = encode::NvGuestEncoder::new(&video.gpu, config, video.filler())?;
        Ok(Box::new(encoder))
    }

    fn guest_decoder(&self, codec: u32) -> Option<Box<dyn GuestDecoder>> {
        let video = &self.0;
        let annex_b = match codec {
            RGFX_VIDEO_CODEC_H264 if video.h264.is_some() => AnnexBCodec::H264,
            RGFX_VIDEO_CODEC_HEVC if video.hevc.is_some() => AnnexBCodec::Hevc,
            _ => return None,
        };
        let mut decoder = AnnexBDecoder::new(&video.gpu, annex_b, &video.runtime);
        decoder.lands_4_4_4();
        Some(Box::new(NvGuestDecoder {
            decoder,
            said: Vec::new(),
        }))
    }

    /// 4:2:0 streams, and HEVC's at 4:4:4 where NVDEC decodes them ([`declines_4_4_4`]).
    fn guest_declines(&self, codec: u32, stream: &Stream) -> Option<String> {
        match stream.chroma_format_idc {
            1 => None,
            3 if codec == RGFX_VIDEO_CODEC_HEVC => declines_4_4_4(&self.0.gpu, stream),
            _ => Some(format!(
                "{NAME}'s decoders land 4:2:0 pictures, and HEVC's 4:4:4, alone"
            )),
        }
    }
}

/// One session's decoder.
struct NvdecDecoder<P: Platform> {
    decoder: AnnexBDecoder,
    /// The video it decodes on, whose imports of device caches its pictures land in.
    video: Arc<NvidiaVideo<P>>,
    /// Whether a picture of the session found its pages or its caches out of CUDA's reach, which
    /// is logged once.
    declined: Arc<AtomicBool>,
}

impl<P: Platform> HostDecoder<P> for NvdecDecoder<P> {
    fn parameter_sets(&mut self, sets: &[u8]) -> Result<(), BackendError> {
        Ok(self.decoder.parameter_sets(sets)?)
    }

    fn decode(
        &mut self,
        au: &[u8],
        _output: &DecodeOutput<P>,
    ) -> Result<Option<Box<dyn HostPicture<P>>>, BackendError> {
        let picture = match self.decoder.decode(au) {
            Ok(picture) => picture,
            Err(error) if error.is_picture_error() => return Ok(picture_error(NAME, error)),
            Err(error) => return Err(error.into()),
        };
        Ok(picture.map(|picture| {
            Box::new(NvdecPicture {
                picture,
                video: Arc::clone(&self.video),
                declined: Arc::clone(&self.declined),
            }) as Box<dyn HostPicture<P>>
        }))
    }

    fn most_unlanded(&self) -> usize {
        AnnexBDecoder::SPARE_PICTURES
    }
}

impl<P: Platform> Drop for NvdecDecoder<P> {
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
struct NvdecPicture<P: Platform> {
    picture: AnnexBPicture,
    video: Arc<NvidiaVideo<P>>,
    declined: Arc<AtomicBool>,
}

impl<P: Platform> NvdecPicture<P> {
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

    /// Writes the picture into the output's planes: in their device caches, in their pages by
    /// the GPU, or read back and copied by the CPU.
    fn land_planes(&self, to: &mut Landing<'_, P>) -> Result<(), BackendError> {
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
        // Each way writes the crop's `width` by `height` at most: a crop short of the planes
        // lands with zeros past it (`video_host`'s module docs, "A short picture").
        let (_, _, crop_width, crop_height) = self.picture.crop();
        to.writes_only(Extent::of_420(
            u64::from(crop_width.min(width)),
            u64::from(crop_height.min(height)),
            sample,
        ))?;
        if self.write_in_cache((width, height), to)? {
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
        let [luma, chroma] = &read.planes()[..] else {
            return Err("a 4:4:4 picture for a 4:2:0 output".into());
        };
        // SAFETY: each plane's rows lie in the decoder's staging memory, packed, which `read`
        // holds through the copy and which lies apart from the output's planes.
        unsafe { to.copy_rows(source(luma), source(chroma)) }
    }

    /// Writes the picture into `to`'s planes' device caches by a copy within the GPU, where both
    /// lie in caches CUDA imports: `Ok(true)` once there, `Ok(false)` where the caches are not to
    /// be had so.
    fn write_in_cache(&self, size: (u32, u32), to: &Landing<'_, P>) -> Result<bool, BackendError> {
        let (luma, chroma) = (to.luma, to.chroma);
        let Some(imports) = &self.video.imports else {
            return Ok(false);
        };
        let (Some(luma_cache), Some(chroma_cache)) = (&luma.cache, &chroma.cache) else {
            return Ok(false);
        };
        let gpu = &self.video.gpu;
        let imported = |plane: &OutputPlane<P>, cache: &MemoryIdentity| {
            imports.get_or_make(
                cache,
                0,
                || import_exported(plane.export(false)?, gpu),
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
        let since = KeptMark::here();
        // SAFETY: the planes are the decode's output, which the guest writes no other way until
        // the command buffer that waits for the session's signal completes, and which lie apart
        // (the decode's check); the imports, which a late copy keeps, hold the caches' memory.
        let written = unsafe {
            (self.picture).write_into_device(size, luma_plane, chroma_plane, Box::new(()))
        };
        // A decode NVDEC failed, or work past its deadline, which may still write: the caches
        // are said written before the session's signal, a stall's once the work it kept, with
        // the imports, is let go.
        let written = match written {
            Ok(written) => written,
            Err(error) => {
                let failed = match &error {
                    reims_vgpu_video::nvidia::Error::Stalled(_) => Failed::Stalled(since),
                    _ => Failed::Stopped,
                };
                let planes: &[&OutputPlane<P>] = if chroma_cache.key() == luma_cache.key() {
                    &[luma]
                } else {
                    &[luma, chroma]
                };
                return Err(to.failed_in_caches(planes, failed, error.into()));
            }
        };
        match written {
            // The copies are done (the write waited for them): the device's reads of each cache,
            // or of its surface where the planes land there, see them once the platform acquired
            // what CUDA wrote, said before the session's signal lets a command buffer read the
            // picture.
            Ok(()) => {
                luma.external_write_done()?;
                if chroma_cache.key() != luma_cache.key() {
                    chroma.external_write_done()?;
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
        luma: &OutputPlane<P>,
        chroma: &OutputPlane<P>,
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
fn device_plane<'a, P: Platform>(
    plane: &OutputPlane<P>,
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
fn memory_plane<'a, P: Platform>(
    plane: &OutputPlane<P>,
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

impl<P: Platform> HostPicture<P> for NvdecPicture<P> {
    fn land(self: Box<Self>, to: &mut Landing<'_, P>) -> Result<(), BackendError> {
        let landed = self.land_planes(to);
        if landed.is_ok()
            && let Some(status) = self.picture.copied_failed()
        {
            to.hardware_failed(format!(
                "NVDEC reports the picture's decode failed (status {status})"
            ));
        }
        landed
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
        self.land(parameter_sets, au, output, None)
    }

    /// The picture's crop narrows to the region, which then lands as a whole picture does, by
    /// the GPU's copy into the registered pages or the CPU's.
    fn decodes_regions(&self) -> bool {
        true
    }

    fn decode_region(
        &mut self,
        parameter_sets: &[u8],
        au: &[u8],
        output: &GuestOutput<'_>,
        region: [u32; 4],
    ) -> Result<bool, VideoError> {
        self.land(parameter_sets, au, output, Some(region))
    }
}

impl NvGuestDecoder {
    /// Decodes `au` after `parameter_sets`, and lands its picture, or `region` of it (`x`, `y`,
    /// `width`, `height`), in `output`.
    fn land(
        &mut self,
        parameter_sets: &[u8],
        au: &[u8],
        output: &GuestOutput<'_>,
        region: Option<[u32; 4]>,
    ) -> Result<bool, VideoError> {
        if !parameter_sets.is_empty() {
            (self.decoder.parameter_sets(parameter_sets)).map_err(video_error)?;
        }
        let Some(mut picture) = self.decoder.decode(au).map_err(video_error)? else {
            return Ok(false);
        };
        if let Some([x, y, width, height]) = region {
            picture
                .crop_within(x, y, width, height)
                .map_err(VideoError::Unsupported)?;
        }
        let target =
            GuestTarget::new(&output.layout, output.memory).map_err(VideoError::Unsupported)?;
        let sample = picture.bytes_per_sample();
        if target.pixel_format().bytes_per_sample() != sample {
            return Err(VideoError::Unsupported(format!(
                "a picture of {sample}-byte samples for a '{}' surface",
                output.layout.pixel_format
            )));
        }
        let four_four_four = picture.four_four_four();
        if target.pixel_format().is_4_4_4() != four_four_four {
            let chroma = if four_four_four { "4:4:4" } else { "4:2:0" };
            return Err(VideoError::Unsupported(format!(
                "a {chroma} picture for a '{}' surface",
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
        let read = read.planes();
        if four_four_four {
            interleave_planes(&read, &planes, surface, sample)
        } else {
            copy_planes(&read, &planes, surface)
        }
        .map_err(VideoError::Malfunction)?;
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
/// `Err` says where a plane's rows lie past the surface, or that `read` is not two planes;
/// nothing is written then.
fn copy_planes(
    read: &[ReadPlane<'_>],
    planes: &[TargetPlane; 2],
    surface: &mut [u8],
) -> Result<(), String> {
    let [luma, chroma] = read else {
        return Err(format!("{} planes read back for 2", read.len()));
    };
    let spans = [
        plane_span(luma, &planes[0], surface.len())?,
        plane_span(chroma, &planes[1], surface.len())?,
    ];
    for ((read, plane), (rows, row_bytes)) in [luma, chroma].into_iter().zip(planes).zip(spans) {
        copy_rows(read, plane, surface, rows, row_bytes);
    }
    Ok(())
}

/// Copies the rows a 4:4:4 picture's `read` holds, luma, Cb and Cr, into `planes` of `surface`
/// as [`copy_planes`] does: the luma as it lies, and Cb and Cr interleaved into the chroma plane,
/// a Cb and a Cr sample of `sample` bytes for each luma sample, Cb first, as the surface's 4:4:4
/// formats lay it ('444v', '444f'). `Err` says where a plane's rows lie past the surface or past
/// what was read back, or that `read` is not three planes; nothing is written then.
fn interleave_planes(
    read: &[ReadPlane<'_>],
    planes: &[TargetPlane; 2],
    surface: &mut [u8],
    sample: usize,
) -> Result<(), String> {
    let [luma, cb, cr] = read else {
        return Err(format!("{} planes read back for 3", read.len()));
    };
    let (pair, chroma) = (2 * sample.max(1), &planes[1]);
    let luma_span = plane_span(luma, &planes[0], surface.len())?;
    let samples = cb.bytes_per_row.min(cr.bytes_per_row) / sample.max(1);
    let (rows, row_bytes) = span(
        cb.rows.min(cr.rows),
        samples.saturating_mul(pair),
        chroma,
        surface.len(),
    )?;
    if !holds(cb, rows) || !holds(cr, rows) {
        return Err(format!("{rows} rows of Cb and Cr past what was read back"));
    }
    copy_rows(luma, &planes[0], surface, luma_span.0, luma_span.1);
    let pairs = row_bytes / pair;
    for row in 0..rows {
        let to = &mut surface[chroma.offset + row * chroma.bytes_per_row..][..pairs * pair];
        let cb = cb.bytes[row * cb.bytes_per_row..].chunks_exact(pair / 2);
        let cr = cr.bytes[row * cr.bytes_per_row..].chunks_exact(pair / 2);
        for ((to, cb), cr) in to.chunks_exact_mut(pair).zip(cb).zip(cr) {
            let (to_cb, to_cr) = to.split_at_mut(pair / 2);
            to_cb.copy_from_slice(cb);
            to_cr.copy_from_slice(cr);
        }
    }
    Ok(())
}

/// The rows and each one's bytes that a copy of `read` writes into `plane` ([`span`]), where
/// `read` holds them; `Err` where they lie past the surface's `surface` bytes.
fn plane_span(
    read: &ReadPlane<'_>,
    plane: &TargetPlane,
    surface: usize,
) -> Result<(usize, usize), String> {
    let (rows, row_bytes) = span(read.rows, read.bytes_per_row, plane, surface)?;
    if !holds(read, rows) {
        return Err(format!(
            "a plane at {} of {rows} rows {} bytes apart past the surface's {surface} bytes",
            plane.offset, plane.bytes_per_row
        ));
    }
    Ok((rows, row_bytes))
}

/// The rows of `plane`, of a surface of `surface` bytes, that a copy of `rows` rows of
/// `row_bytes` each writes, as many as both have, and each one's bytes, as far as the plane's
/// rows hold it; `Err` where they lie past the surface.
fn span(
    rows: u32,
    row_bytes: usize,
    plane: &TargetPlane,
    surface: usize,
) -> Result<(usize, usize), String> {
    let rows = usize::try_from(rows.min(plane.rows)).unwrap_or(usize::MAX);
    let row_bytes = row_bytes.min(plane.row_bytes);
    let end = match rows.checked_sub(1) {
        None => Some(plane.offset),
        Some(past) => (past.checked_mul(plane.bytes_per_row))
            .and_then(|bytes| bytes.checked_add(plane.offset))
            .and_then(|bytes| bytes.checked_add(row_bytes)),
    };
    match end {
        Some(end) if end <= surface => Ok((rows, row_bytes)),
        _ => Err(format!(
            "a plane at {} of {rows} rows {} bytes apart past the surface's {surface} bytes",
            plane.offset, plane.bytes_per_row
        )),
    }
}

/// Whether `read`'s bytes hold `rows` of its rows.
fn holds(read: &ReadPlane<'_>, rows: usize) -> bool {
    (rows.checked_mul(read.bytes_per_row)).is_some_and(|bytes| bytes <= read.bytes.len())
}

/// Copies `rows` of `read`'s rows, `row_bytes` of each, into `plane` of `surface`, a span
/// [`plane_span`] answered.
fn copy_rows(
    read: &ReadPlane<'_>,
    plane: &TargetPlane,
    surface: &mut [u8],
    rows: usize,
    row_bytes: usize,
) {
    for row in 0..rows {
        let from = &read.bytes[row * read.bytes_per_row..][..row_bytes];
        surface[plane.offset + row * plane.bytes_per_row..][..row_bytes].copy_from_slice(from);
    }
}

/// What a VideoToolbox decoder sees of `error`: a stream that does not decode, or whose picture
/// NVDEC failed, as bad data; one NVDEC cannot decode as unsupported; a leading picture the
/// stream's random access skips as skipped; and the rest as the decoder's malfunction, a
/// picture smaller than NVDEC decodes among them, as Apple's decoders answer a size they do not
/// decode.
fn video_error(error: reims_vgpu_video::nvidia::Error) -> VideoError {
    use reims_vgpu_video::nvidia::Error;
    match error {
        Error::Bitstream(_) | Error::Picture(_) | Error::DecodeFailed(_) => {
            VideoError::BadData(error.to_string())
        }
        Error::Unsupported(why) => VideoError::Unsupported(why),
        Error::SkippedLeadingPicture(why) => VideoError::SkippedLeadingPicture(why),
        error => VideoError::Malfunction(error.to_string()),
    }
}

#[cfg(test)]
mod tests;
