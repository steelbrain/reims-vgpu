//! The reims-graphics device's hardware video on VA-API ([`open`]): H.264 and HEVC decode on the
//! render node of the runtime device's GPU, whatever its vendor, through the VA-API crate's own
//! decoders, in a [`HostVideoSession`] whose event the host signals; and H.264 and HEVC encode
//! there, in a [`HostEncodeSession`] (`encode`'s module doc).
//!
//! - **What it answers** ([`caps_of`]) is the driver's: the VLD profiles it lists, with the
//!   formats its configs' `VAConfigAttribRTFormat` names, and the sizes its configs and their
//!   surfaces give. VA knows no level limit and no DPB bound of its own, so the levels are each
//!   codec's top and the DPB its picture structures' bounds (H.264 `ReferenceFrames[16]`, HEVC
//!   `ReferenceFrames[15]`). A picture references as many as H.264's lists name, 16 frames, and
//!   for HEVC the 8 its current reference sets hold together at most (va.h's `NumPocTotalCurr`,
//!   the VA-API crate's `MAX_CURRENT_REFERENCES`), which the guest's video driver checks each
//!   picture against before it schedules it. The access unit is copied into VA's slice buffers by
//!   the decoder, as every VA client's is, so it is read at any place and length
//!   (`bitstream_alignment` 1).
//! - **A decode** hands the access unit to the decoder (`H264Decoder`, `HevcDecoder` on Annex B,
//!   in decode order), which starts the hardware's work and answers the access unit's frame: one,
//!   or none for parameter sets and for a picture the stream's random access skips.
//! - **The device's caches are made dma-bufs** ([`CacheTargets`], `writes_into_device_caches`)
//!   where the platform makes caches of dma-bufs and either of the driver's checks passed anything
//!   (`cache_writes`, on Unix): the check of decodes into targets (the VA crate's `TargetChecks`,
//!   kept under `device-checks/va-targets` in the platform's cache directory) a codec and format,
//!   or the import check (the VA crate's `import_check`) NV12 or P010. iHD fails the first for
//!   every kind, its decodes writing their render target tiled whatever the import says
//!   (`docs/video/bugs.md`, U30), and passes the second, so on iHD every picture is written into
//!   its cache by the driver's video processing (the first way below).
//! - **A picture is decoded into its output's device cache** where it can be: where both planes
//!   lie in one cache made a dma-buf, the driver's check passed the codec and format, the planes'
//!   rows and starts lie on multiples of 256 bytes of the memory, and they hold the coded
//!   picture's rows. The cache's memory is exported once, and imported as the
//!   decode's render target once a cache, a new import made while the decoder still holds the
//!   last (a reference's). The picture is then there once its decode is done and the platform
//!   acquired what the driver wrote (`Buffer::external_write_done`, which the device's next
//!   submission records before its commands), and nothing is copied. Each reason a session's
//!   pictures are not, of a cache made so, is logged once. Where the platform's exported memory
//!   is not the driver's to read once it is acquired ([`CachedReferences::Apart`], the rt
//!   platform's), only a picture no later picture predicts from is decoded into its cache: a
//!   reference picture is decoded into a surface of the decoder's own, which the decoder reads it
//!   from as a reference, and lands in its cache by the first way below. Each session counts the
//!   pictures of each role it asked a cache for, and the references it kept apart, and logs them
//!   at its end.
//! - **Otherwise a picture lands** in the output's planes one of three ways, each the fallback at
//!   the last's refusal, the session counting each:
//!   1. **In the surface's pages or its device cache, by the driver's video processing** (the VA
//!      crate's `Frame::write_to_guest`): a run from the decode surface into a linear VA surface
//!      imported over a dma-buf of the planes' pages, or of the cache's memory, at their own
//!      pitches. It is taken where both planes lie in one memory a dma-buf is made of
//!      ([`PlanePages`]), or in one device cache made a dma-buf for the video
//!      ([`CacheTargets`]), whose pictures the driver's check refused to decode there or whose
//!      planes do not lie as a target's must, and where the driver's import check found it writes
//!      imports of the frame's format right, at the alignment the planes' rows and starts are a
//!      multiple of, and in whole rows where it writes them so. A picture written into a cache is
//!      acquired as a decode into it is (`Buffer::external_write_done`, or the cache's surface's
//!      `SurfaceMemory::external_writes_done`). The run writes the planes' whole even width and
//!      height within the coded picture, which [`layout`] tells the guest to make room for.
//!      CoreVideo's planes, 32 bytes into their pages, are written so by no driver: radeonsi
//!      imports planes 256 bytes apart alone, iHD writes its luma plane only at its dma-buf's
//!      first byte and its chroma plane only whole rows after it (`docs/video/bugs.md`, U26). A
//!      cache's surface, where the guest's textures show its planes whole (the decode's, as the
//!      guest's video driver makes them), lays its luma plane at its memory's first byte and its
//!      chroma plane whole rows of a pitch on 256 bytes after it, which iHD writes; a cache's
//!      buffer keeps CoreVideo's layout, its luma plane off its dma-buf's first byte, which iHD's
//!      run declines.
//!   2. **By the device's GPU** ([`GpuCopies`]): each plane copied into the plane's buffer at its
//!      own rows and offset, the crop alone as the CPU's copy writes it, where the open made a
//!      copier. On Linux the rt platform's copier (`gpu_copy`) takes planes in device caches
//!      outside their surfaces: the driver's video processing writes the picture into a scratch
//!      of the session's that lies as its imports must, and the GPU copies the crop from it into
//!      the cache, so that a cache whose layout the driver's import refuses (CoreVideo's planes
//!      32 bytes into their memory, rows narrower than the driver's pitch) is still written on
//!      the GPU; it declines a plane in the guest's pages. On Windows the rt platform's copier
//!      declines each picture, counted, so that each lands by the CPU (`rt_windows`).
//!   3. **By the CPU**: the frame read back (staged, or `vaGetImage`, the VA crate's way back),
//!      and copied into the planes' host bytes, or into the session's staging for a plane in a
//!      device cache.
//!
//!   Each reason a session's pictures take the third way is logged once, the GPU's copy's
//!   refusal with the in-place write's: the copier's own where the session has none.
//! - **VideoToolbox's paravirtual decoders** (`VaGuestDecoder`, `super::vt`): an arm64 guest's
//!   decode lands in the IOSurface its kext lent, written there by the driver's video processing
//!   (`Frame::write_to_guest`, the first way above) where the import check takes the surface's
//!   layout and memory, else read back and copied by the CPU into the surface's pages.
//! - **VideoToolbox's paravirtual encoders** (`encode::VaGuestEncoder`, `super::vt`): an arm64
//!   guest's encoder reads each source frame from the IOSurface its kext lent: written by the
//!   runtime device's GPU from a buffer over its pages into an input of the encoder's, as an x86
//!   guest's frames are (`encode`'s module doc), else through the host's view of its pages, which
//!   the VA crate's encoder uploads.
//! - **The layout** ([`layout`]) is granularity [2, 2], so that each plane holds the crop's
//!   even rows and columns, and the alignment the import check found for the codec's formats;
//!   all zero where it found none.

#[cfg(unix)]
use std::os::fd::{AsFd as _, OwnedFd};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Weak};

use reims_vgpu_graphics::wire::{
    RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_DECODE, RGFX_VIDEO_ENCODE,
    RGFX_VIDEO_FORMAT_NV12, RGFX_VIDEO_FORMAT_P010, RGFX_VIDEO_H264_BASELINE, RGFX_VIDEO_H264_HIGH,
    RGFX_VIDEO_H264_MAIN, RGFX_VIDEO_HEVC_MAIN, RGFX_VIDEO_HEVC_MAIN_10, RgfxVideoCodecCaps,
    RgfxVideoEncodeCaps, RgfxVideoOutputLayout,
};
#[cfg(unix)]
use reims_vgpu_video::codec::PictureRole;
use reims_vgpu_video::core::backend::{BackendError as VideoError, DeviceWrite};
#[cfg(unix)]
use reims_vgpu_video::core::device_memory::{
    DeviceImports, DmaBufError, MemoryIdentity, SurfaceDmaBuf,
};
use reims_vgpu_video::core::device_memory::{DeviceMemory, GuestTarget};
#[cfg(unix)]
use reims_vgpu_video::core::device_wait::KeptMark;
use reims_vgpu_video::core::fourcc::FourCc;
use reims_vgpu_video::core::runtime::Runtime;
use reims_vgpu_video::core::surfaces::{PlaneLayout, SurfaceLayout};
use reims_vgpu_video::vaapi::{
    Capabilities, ConfigCapabilities, Display, EncoderSupport, Entrypoint, Frame, GuestWrites,
    H264_MAX_MACROBLOCKS, H264Decoder, HevcDecoder, ImportWrites, PlaneRows, Profile, Target,
    TargetCodec, TargetFor, VideoEncoder, encoder_support,
};
#[cfg(unix)]
use reims_vgpu_video::vaapi::{TARGET_ALIGNMENT, TargetChecks, TargetPlane, TargetShape};
#[cfg(unix)]
use reims_vgpu_vulkan::core::num::ToU64 as _;
use reims_vgpu_vulkan::core::num::ToUsize as _;
use reims_vgpu_vulkan::core::platform::Handle as _;
use reims_vgpu_vulkan::core::platform::Platform;
#[cfg(unix)]
use reims_vgpu_vulkan::core::platform::device::GpuDevice as _;
#[cfg(unix)]
use reims_vgpu_vulkan::core::platform::resource::{Buffer as _, ExportedMemory, SurfaceMemory};
#[cfg(unix)]
use reims_vgpu_vulkan::core::raise::OnFault;
use reims_vgpu_vulkan::core::sync::Mutex;

use super::decode::{
    GraphicsVideo, GraphicsVideoSession, GuestDecoder, GuestEncoder, GuestOutput, PlanePages,
};
use reims_vgpu_video::paravirt::{Encoder, EncoderConfig};

#[cfg(unix)]
use super::video_host::Failed;
use super::video_host::{
    BackendError, DecodeOutput, Extent, HostDecoder, HostEncodeSession, HostPicture, HostPlane,
    HostVideoSession, Landing, OutputPlane, picture_error,
};

mod encode;
#[cfg(unix)]
mod fill;
#[cfg(unix)]
mod gpu_copy;
#[cfg(windows)]
mod rt_windows;

/// The backend's name in the session's log lines.
const NAME: &str = "VA-API";

/// `VA_RT_FORMAT_YUV420` and `VA_RT_FORMAT_YUV420_10` (va.h): 8-bit and 10-bit 4:2:0.
const RT_YUV420: u32 = 0x0000_0001;
const RT_YUV420_10: u32 = 0x0000_0100;

/// The smallest picture a decode answers where its configs' surfaces give no minimum.
const MIN_SIZE: u32 = 16;

/// H.264's top level, 6.2 (`level_idc` 62), and HEVC's, 6.2 (`general_level_idc` 186).
const H264_TOP_LEVEL: u32 = 62;

/// The widest H.264 picture answered where the driver's largest size holds more than
/// `H264_MAX_MACROBLOCKS`: the widest either driver decodes. radeonsi answers 4096; iHD answers
/// 8192 in `VAConfigAttribMaxPictureWidth`, but its surfaces' bound is 4096x4096 and it refuses
/// every wider picture, from 4112x2160 to 8192x1024 (measured, Iris Xe of an i9-13900H, iHD
/// 26.3.5, 2026-11-08).
const H264_WIDEST: u32 = 4096;
const HEVC_TOP_LEVEL: u32 = 186;

/// What the platform's device lets the VA-API arm do ([`open`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlatformRules {
    /// Whether a picture a later one predicts from may be decoded into its device cache.
    pub(crate) references: CachedReferences,
    /// Whether the arm answers the driver's encoders: where the platform reads their sources for
    /// them (`encode`'s module doc).
    pub(crate) encodes: bool,
}

/// Whether a reference picture is decoded into its output's device cache (module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CachedReferences {
    /// Only a picture no later picture predicts from: the platform's exported memory is not the
    /// driver's to touch between its acquire and the driver's next write of a whole picture
    /// (the rt platform's `rt_platform::export`, "The acquire").
    Apart,
}

/// Where a VA display is opened ([`open`]): a render node, or on Windows a DXGI adapter by its
/// LUID.
pub(crate) enum DisplayAt<'a> {
    /// A DRM render node (`/dev/dri/renderD128`, say).
    #[cfg(unix)]
    Node(&'a Path),
    /// The DXGI adapter of this LUID, as Vulkan's `deviceLUID` holds it.
    #[cfg(windows)]
    Adapter([u8; 8], std::marker::PhantomData<&'a ()>),
}

impl DisplayAt<'_> {
    /// A display opened there, its work on `runtime`.
    fn open(&self, runtime: Arc<dyn Runtime>) -> Result<Display, reims_vgpu_video::vaapi::Error> {
        match self {
            #[cfg(unix)]
            DisplayAt::Node(node) => Display::open(node, runtime),
            #[cfg(windows)]
            DisplayAt::Adapter(luid, _) => Display::open_adapter(Some(*luid), runtime),
        }
    }
}

impl std::fmt::Display for DisplayAt<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            #[cfg(unix)]
            DisplayAt::Node(node) => write!(f, "{}", node.display()),
            #[cfg(windows)]
            DisplayAt::Adapter(luid, _) => {
                write!(
                    f,
                    "the DXGI adapter of LUID {:#x}",
                    u64::from_le_bytes(*luid)
                )
            }
        }
    }
}

/// The VA-API video of the rt platform's `device`, on its GPU's render node
/// (`host_gpus::render_node_at`), as [`open`] opens it; `None`, logged, where its GPU has none.
/// Its pictures land by the driver's video processing, then by the GPU's copy from a scratch the
/// driver wrote, for a plane in a device cache (`gpu_copy`, the copier of [`fill::DeviceFills`]),
/// then by the CPU's copy; a reference picture is not decoded into its cache
/// ([`CachedReferences::Apart`]); and the driver's encoders are answered, their sessions' frames
/// written into the encoders' inputs by the GPU (`fill`), else uploaded by the CPU.
#[cfg(unix)]
pub(crate) fn open_on_rt(
    device: &RtDevice,
    runtime: Arc<dyn Runtime>,
) -> Option<Arc<dyn GraphicsVideo<reims_vgpu_vulkan::RtPlatform>>> {
    let gpu = device.host_gpu();
    let Some(node) = reims_vgpu_vulkan::host_gpus::render_node_at(gpu.render, gpu.pci) else {
        log::info!("hardware video: no VA-API: the device's GPU has no render node");
        return None;
    };
    let checks = reims_vgpu_vulkan::os::cache_directory()
        .map(|dir| dir.join("device-checks").join("va-targets"));
    let rules = PlatformRules {
        references: CachedReferences::Apart,
        encodes: true,
    };
    let copies: Arc<dyn GpuCopies<reims_vgpu_vulkan::RtPlatform>> =
        Arc::new(fill::DeviceFills::new(device, &runtime));
    open::<reims_vgpu_vulkan::RtPlatform>(
        device,
        device.facts().registry_id,
        DisplayAt::Node(&node),
        runtime,
        Some(copies),
        checks,
        rules,
    )
}

/// The VA-API video of the rt platform's `device` on Windows: on the DXGI adapter of its GPU, by
/// the LUID Vulkan answers for it (`rt_platform::HostGpu`), over Mesa's VA-on-D3D12 driver, as
/// [`open`] opens it; `None`, logged, where its GPU names none. Its pictures land by the CPU's
/// copy, at the GPU's copy's refusal of each, counted (`rt_windows`); a reference picture is not
/// decoded into its cache ([`CachedReferences::Apart`]), and no encoder is answered: rt imports
/// no shared Direct3D 12 texture, which an encoder's input would be written through.
#[cfg(windows)]
pub(crate) fn open_on_rt(
    device: &RtDevice,
    runtime: Arc<dyn Runtime>,
) -> Option<Arc<dyn GraphicsVideo<reims_vgpu_vulkan::RtPlatform>>> {
    let Some(luid) = device.host_gpu().luid else {
        log::info!("hardware video: no VA-API: the device's GPU names no DXGI adapter");
        return None;
    };
    let copies: Arc<dyn GpuCopies<reims_vgpu_vulkan::RtPlatform>> =
        Arc::new(rt_windows::RtCopies::new(luid));
    let rules = PlatformRules {
        references: CachedReferences::Apart,
        encodes: false,
    };
    open::<reims_vgpu_vulkan::RtPlatform>(
        device,
        device.facts().registry_id,
        DisplayAt::Adapter(luid, std::marker::PhantomData),
        runtime,
        Some(copies),
        None,
        rules,
    )
}

/// The rt platform's device.
type RtDevice = reims_vgpu_vulkan::rt_platform::RtObj<reims_vgpu_vulkan::rt_platform::RtDevice>;

/// The VA-API video of `device`'s GPU, on its render node `at` (on Windows its adapter), its decoders' threads and waits
/// on `runtime`, its pictures copied by the GPU through `copies` where given, the driver's check
/// of decodes into device caches kept in `checks` where given, as the platform's `rules` let it;
/// one per device while it lives, by `registry_id`. `None`, logged at `info` with why, where
/// libva does not load, the node opens no display, or the driver decodes neither H.264 nor HEVC
/// (and, where the rules let it, encodes neither).
pub(crate) fn open<P: Platform>(
    device: &P::Device,
    registry_id: u64,
    at: DisplayAt<'_>,
    runtime: Arc<dyn Runtime>,
    copies: Option<Arc<dyn GpuCopies<P>>>,
    checks: Option<PathBuf>,
    rules: PlatformRules,
) -> Option<Arc<dyn GraphicsVideo<P>>> {
    type Made = Vec<(u64, Weak<dyn std::any::Any + Send + Sync>)>;
    static MADE: LazyLock<Mutex<Made>> = LazyLock::new(|| Mutex::new(Vec::new()));
    let mut made = MADE.lock();
    made.retain(|(_, video)| video.strong_count() > 0);
    let kept = (made.iter())
        .filter(|(id, _)| *id == registry_id)
        .find_map(|(_, video)| video.upgrade()?.downcast::<VaapiVideo<P>>().ok());
    if let Some(video) = kept {
        return Some(video);
    }
    let opened = VaapiVideo::<P>::new(device, &at, runtime, copies, checks.as_deref(), rules);
    let video = match opened {
        Ok(video) => Arc::new(video),
        Err(why) => {
            log::info!("hardware video: no VA-API at {at}: {why}");
            return None;
        }
    };
    let any: Arc<dyn std::any::Any + Send + Sync> = video.clone();
    made.push((registry_id, Arc::downgrade(&any)));
    Some(video)
}

/// A GPU's VA-API video, as the graphics device sees it.
struct VaapiVideo<P: Platform> {
    display: Display,
    /// The device the sessions' events are made on.
    device: P::Device,
    caps: Vec<RgfxVideoCodecCaps>,
    /// The profiles the driver encodes, which the encode entries of `caps` list.
    encoders: Arc<[EncoderSupport]>,
    /// What the driver's video processing writes into imports, as the display's check found it.
    writes_imports: ImportWrites,
    /// The imports of the outputs' memory, which the sessions share.
    imports: Arc<GuestWrites>,
    /// What makes each session's GPU copies of its pictures (the module doc's second way).
    copies: Option<Arc<dyn GpuCopies<P>>>,
    /// What writes pictures into device caches through their memory, where the platform makes
    /// caches of dma-bufs and the driver's checks passed anything (`cache_writes`, on Unix).
    targets: Option<Arc<CacheTargets<P>>>,
}

impl<P: Platform> VaapiVideo<P> {
    fn new(
        device: &P::Device,
        at: &DisplayAt<'_>,
        runtime: Arc<dyn Runtime>,
        copies: Option<Arc<dyn GpuCopies<P>>>,
        checks: Option<&Path>,
        rules: PlatformRules,
    ) -> Result<Self, String> {
        let started = runtime.now();
        let display = at.open(Arc::clone(&runtime)).map_err(|e| e.to_string())?;
        let capabilities = display.capabilities().map_err(|e| e.to_string())?;
        let encoders: Arc<[EncoderSupport]> = if rules.encodes {
            encoder_support(&display).into()
        } else {
            log::info!(
                "hardware video: {NAME} at {at}: no encoder answered: the platform reads no \
                 encoder's sources for it"
            );
            Arc::new([])
        };
        let mut caps = caps_of(&capabilities);
        caps.extend(encode::encode_caps_of(&encoders, &capabilities));
        if caps.is_empty() {
            return Err("the driver decodes and encodes neither H.264 nor HEVC".to_owned());
        }
        let writes_imports = display.import_writes();
        log::info!(
            "hardware video: {NAME} at {at}: {}, opened in {:.2} ms on thread {:?}; in place: NV12 \
             {:?}, P010 {:?}",
            display.vendor(),
            runtime.elapsed_since(started).as_secs_f64() * 1e3,
            std::thread::current().name().unwrap_or("unnamed"),
            writes_imports.nv12(),
            writes_imports.p010(),
        );
        let imports = Arc::new(GuestWrites::new(&display));
        let targets = CacheTargets::new(
            device,
            &display,
            &runtime,
            checks,
            &writes_imports,
            rules.references,
        )
        .map(Arc::new);
        Ok(Self {
            display,
            device: device.clone_ref(),
            caps,
            encoders,
            writes_imports,
            imports,
            copies,
            targets,
        })
    }

    /// A session's GPU copier; `Err` with why where the video makes none or the device refuses
    /// it, which the session's CPU copies log.
    fn copier(&self) -> SessionCopier<P> {
        let copies = (self.copies.as_ref()).ok_or("the platform makes no GPU copies")?;
        let copier = copies.copier()?;
        Ok(Arc::new(Mutex::new(copier)))
    }
}

impl<P: Platform> GraphicsVideo<P> for VaapiVideo<P> {
    fn caps(&self) -> Vec<RgfxVideoCodecCaps> {
        self.caps.clone()
    }

    fn layout(&self, codec: &RgfxVideoCodecCaps) -> RgfxVideoOutputLayout {
        layout(codec, &self.writes_imports)
    }

    fn encode_caps(&self, codec: &RgfxVideoCodecCaps) -> RgfxVideoEncodeCaps {
        encode::encode_caps(codec, &self.encoders)
    }

    /// Every host's: a picture lands in an output's device cache by the GPU's copy, or by the
    /// CPU's into the session's staging and the staging's copy into the cache (module docs).
    fn lands_in_device_caches(&self) -> bool {
        true
    }

    /// Where the platform makes caches of dma-bufs and the driver's checks passed anything
    /// (`cache_writes`, on Unix): a picture is decoded into its cache, or written there by the
    /// driver's video processing (module doc).
    fn writes_into_device_caches(&self) -> bool {
        self.targets.is_some()
    }

    /// The draws the platform's GPU copies make (`GpuCopies::plane_draws`).
    #[cfg(windows)]
    fn plane_draws(&self) -> Option<Arc<reims_vgpu_video::vaapi::PlaneDraw>> {
        self.copies.as_ref()?.plane_draws()
    }

    fn session(&self, codec: u32, direction: u32) -> Option<Arc<dyn GraphicsVideoSession<P>>> {
        let entry = (self.caps.iter()).find(|c| c.codec == codec && c.direction == direction)?;
        if direction == RGFX_VIDEO_ENCODE {
            let caps = encode::encode_caps(entry, &self.encoders);
            let filler = match self.copies.as_ref() {
                Some(copies) => copies.filler(),
                None => Err("the platform makes no GPU copies".to_owned()),
            };
            let encoder = encode::VaEncoder::new(
                &self.display,
                codec,
                Arc::clone(&self.encoders),
                entry,
                &caps,
                filler,
            );
            let session = HostEncodeSession::new(&self.device, codec, NAME, Box::new(encoder))?;
            return Some(Arc::new(session));
        }
        if direction != RGFX_VIDEO_DECODE {
            return None;
        }
        let decoder = match codec {
            RGFX_VIDEO_CODEC_H264 => Codec::H264(H264Decoder::new_annex_b(&self.display)),
            RGFX_VIDEO_CODEC_HEVC => Codec::Hevc(HevcDecoder::new_annex_b(&self.display)),
            _ => return None,
        };
        let decoder = VaDecoder {
            codec: decoder,
            next_id: 0,
            imports: Arc::clone(&self.imports),
            copier: self.copier(),
            said: Arc::new(Mutex::new(Vec::new())),
            targets: self.targets.clone(),
            roles: Arc::new(RoleCounts::default()),
        };
        let clock = Arc::clone(self.display.runtime());
        let session = HostVideoSession::new(&self.device, clock, codec, NAME, Box::new(decoder))?;
        Some(Arc::new(session))
    }

    /// VA-API's decoders land in any memory: by the driver's video processing, or the CPU.
    fn guest_decodes(&self) -> bool {
        true
    }

    /// One for each codec the driver encodes a profile of (`encode::guest_encoders`).
    fn guest_encoders(&self) -> Vec<Encoder> {
        encode::guest_encoders(&self.encoders)
    }

    fn guest_encoder(
        &self,
        config: &EncoderConfig,
    ) -> Result<Box<dyn GuestEncoder<P>>, VideoError> {
        let filler = match self.copies.as_ref() {
            Some(copies) => copies.filler(),
            None => Err("the platform makes no GPU copies".to_owned()),
        };
        let encoder = encode::VaGuestEncoder::new(&self.display, &self.encoders, config, filler)?;
        Ok(Box::new(encoder))
    }

    fn guest_decoder(&self, codec: u32) -> Option<Box<dyn GuestDecoder>> {
        (self.caps.iter()).find(|c| c.codec == codec && c.direction == RGFX_VIDEO_DECODE)?;
        let codec = match codec {
            RGFX_VIDEO_CODEC_H264 => Codec::H264(H264Decoder::new_annex_b(&self.display)),
            RGFX_VIDEO_CODEC_HEVC => Codec::Hevc(HevcDecoder::new_annex_b(&self.display)),
            _ => return None,
        };
        Some(Box::new(VaGuestDecoder {
            codec,
            next_id: 0,
            imports: Arc::clone(&self.imports),
            said: Vec::new(),
        }))
    }
}

/// The caps of H.264 and HEVC decode the driver's answers give (module doc); an entry for each
/// codec of which it lists a profile it decodes.
pub(crate) fn caps_of(caps: &Capabilities) -> Vec<RgfxVideoCodecCaps> {
    h264_caps(caps).into_iter().chain(hevc_caps(caps)).collect()
}

/// `profile`'s VLD config where the driver lists one that decodes `rt_format` (or names no
/// formats at all).
fn vld(caps: &Capabilities, profile: Profile, rt_format: u32) -> Option<&ConfigCapabilities> {
    (caps.find(profile, Entrypoint::VLD)).filter(|config| {
        config
            .rt_formats
            .is_none_or(|formats| formats & rt_format != 0)
    })
}

/// An entry of `codec` decode at the least of `configs`' limits: the largest picture each config,
/// else its surfaces, gives, and the smallest its surfaces give ([`MIN_SIZE`] where they give
/// none); `None` where no config gives a largest picture.
fn decode_caps(codec: u32, configs: &[&ConfigCapabilities]) -> Option<RgfxVideoCodecCaps> {
    let mut caps = RgfxVideoCodecCaps {
        codec,
        direction: RGFX_VIDEO_DECODE,
        min_size: [MIN_SIZE; 2],
        max_size: [u32::MAX; 2],
        bitstream_alignment: 1,
        ..RgfxVideoCodecCaps::default()
    };
    let mut sized = false;
    for config in configs {
        let surfaces = config.surfaces.as_ref().ok();
        let max_width = config.max_width.or(surfaces.and_then(|s| s.max_width));
        let max_height = config.max_height.or(surfaces.and_then(|s| s.max_height));
        if let (Some(width), Some(height)) = (max_width, max_height) {
            caps.max_size = [caps.max_size[0].min(width), caps.max_size[1].min(height)];
            sized = true;
        }
        let min = |of: Option<u32>| of.unwrap_or(MIN_SIZE).max(1);
        caps.min_size[0] = caps.min_size[0].max(min(surfaces.and_then(|s| s.min_width)));
        caps.min_size[1] = caps.min_size[1].max(min(surfaces.and_then(|s| s.min_height)));
    }
    sized.then_some(caps)
}

/// H.264 decode: Baseline where a config of ConstrainedBaseline, Main or High decodes 4:2:0, as
/// the decoder runs a Baseline stream on the first of those the driver has; Main on Main or High;
/// High on High. NV12 out. The largest size is the driver's, but where it holds a picture of more
/// than `H264_MAX_MACROBLOCKS`, which the decoder refuses, it is at most `H264_WIDEST` wide and as
/// tall as the bound leaves at that width: radeonsi's 4096x4096 and iHD's 8192x8192 are answered
/// as 4096x4080, which holds 3840x2160 and 4096x2160.
fn h264_caps(caps: &Capabilities) -> Option<RgfxVideoCodecCaps> {
    let of = |profile| vld(caps, profile, RT_YUV420);
    let (baseline, main, high) = (
        of(Profile::H264_CONSTRAINED_BASELINE),
        of(Profile::H264_MAIN),
        of(Profile::H264_HIGH),
    );
    let configs: Vec<&ConfigCapabilities> = [baseline, main, high].into_iter().flatten().collect();
    let mut entry = decode_caps(RGFX_VIDEO_CODEC_H264, &configs)?;
    let [width, height] = entry.max_size;
    let macroblocks = u64::from(width.div_ceil(16)) * u64::from(height.div_ceil(16));
    if macroblocks > u64::from(H264_MAX_MACROBLOCKS) {
        let width = width.min(H264_WIDEST);
        let height = height.min(H264_MAX_MACROBLOCKS / width.div_ceil(16) * 16);
        entry.max_size = [width, height];
    }
    for (bit, on) in [
        (RGFX_VIDEO_H264_BASELINE, !configs.is_empty()),
        (RGFX_VIDEO_H264_MAIN, main.is_some() || high.is_some()),
        (RGFX_VIDEO_H264_HIGH, high.is_some()),
    ] {
        if on {
            entry.profiles |= bit;
        }
    }
    entry.formats = RGFX_VIDEO_FORMAT_NV12;
    entry.max_level = H264_TOP_LEVEL;
    entry.max_dpb_pictures = 17;
    entry.max_references = 16;
    Some(entry)
}

/// HEVC decode: Main where its config decodes 8-bit 4:2:0, into NV12, and Main 10 where its config
/// decodes 10-bit 4:2:0, into P010.
fn hevc_caps(caps: &Capabilities) -> Option<RgfxVideoCodecCaps> {
    let main = vld(caps, Profile::HEVC_MAIN, RT_YUV420);
    let main10 = vld(caps, Profile::HEVC_MAIN10, RT_YUV420_10);
    let configs: Vec<&ConfigCapabilities> = [main, main10].into_iter().flatten().collect();
    let mut entry = decode_caps(RGFX_VIDEO_CODEC_HEVC, &configs)?;
    if main.is_some() {
        entry.profiles |= RGFX_VIDEO_HEVC_MAIN;
        entry.formats |= RGFX_VIDEO_FORMAT_NV12;
    }
    if main10.is_some() {
        entry.profiles |= RGFX_VIDEO_HEVC_MAIN_10;
        entry.formats |= RGFX_VIDEO_FORMAT_P010;
    }
    entry.max_level = HEVC_TOP_LEVEL;
    entry.max_dpb_pictures = 16;
    // The current reference sets' pictures together (the module doc): a picture of more is
    // refused, and radeonsi decodes one wrongly.
    entry.max_references = 8;
    Some(entry)
}

/// How `codec`'s decode lands in place (module doc): granularity [2, 2], at the largest of the
/// alignments the import check found for the codec's formats; all zero where it found none, and
/// for an encode, whose sources the CPU reads.
pub(crate) fn layout(codec: &RgfxVideoCodecCaps, writes: &ImportWrites) -> RgfxVideoOutputLayout {
    if codec.direction != RGFX_VIDEO_DECODE {
        return RgfxVideoOutputLayout::default();
    }
    let alignments = [
        (RGFX_VIDEO_FORMAT_NV12, writes.nv12()),
        (RGFX_VIDEO_FORMAT_P010, writes.p010()),
    ];
    let alignment = (alignments.into_iter())
        .filter(|(format, _)| codec.formats & format != 0)
        .filter_map(|(_, write)| Some(write.ok()?.alignment))
        .max();
    match alignment.and_then(|a| u32::try_from(a).ok()) {
        Some(alignment) => RgfxVideoOutputLayout {
            granularity: [2, 2],
            alignment,
            reserved_0c: 0,
        },
        None => RgfxVideoOutputLayout::default(),
    }
}

/// The bytes of the cache the open makes to learn whether the platform makes caches of dma-bufs:
/// a 4K NV12 picture's.
#[cfg(unix)]
const PROBE_BYTES: u64 = 3840 * 2160 * 3 / 2;

/// `MTLStorageModePrivate`, a device cache's storage.
#[cfg(unix)]
const STORAGE_PRIVATE: u64 = 2 << 4;

/// The most device caches whose memory is kept exported at once: a few decoders' output pools.
#[cfg(unix)]
const MOST_CACHES: usize = 64;

/// What writes the video's pictures into their outputs' device caches through the caches' memory
/// (module doc), by decodes into them or the driver's video processing: the driver's check's
/// answers, and each cache's memory as a dma-buf, by the cache's identity, while the cache lives.
#[cfg(unix)]
struct CacheTargets<P: Platform> {
    display: Display,
    checks: TargetChecks,
    memories: DeviceImports<CacheMemory<P>>,
    /// Whether a reference picture is decoded into its cache.
    references: CachedReferences,
}

/// A device cache's memory, exported as a dma-buf, and the last target imported over it.
#[cfg(unix)]
struct CacheMemory<P: Platform> {
    dma_buf: OwnedFd,
    /// The allocation's bytes, and the cache's first byte in it.
    size: u64,
    offset: u64,
    /// The cache's buffer, or its surface's memory where its planes land there, which every
    /// import of its memory holds.
    cache: CacheOf<P>,
    /// The last target imported over it, at its planes, handed again while no decoder holds it.
    last: Mutex<Option<(Target, [TargetPlane; 2])>>,
}

/// The memory an output's planes lie in, in a device cache made a dma-buf for the video: the
/// cache's buffer, or the cache's surface where they land there ([`OutputPlane::surface`]).
#[cfg(unix)]
enum CacheOf<P: Platform> {
    Buffer(P::Buffer),
    Surface(Arc<dyn SurfaceMemory>),
}

#[cfg(unix)]
impl<P: Platform> CacheOf<P> {
    /// The memory `plane` lies in.
    fn of(plane: &OutputPlane<P>) -> Self {
        match &plane.surface {
            Some(target) => Self::Surface(Arc::clone(&target.memory)),
            None => Self::Buffer(plane.buffer.clone_ref()),
        }
    }

    /// Another hold of the same memory.
    fn clone_ref(&self) -> Self {
        match self {
            Self::Buffer(buffer) => Self::Buffer(buffer.clone_ref()),
            Self::Surface(memory) => Self::Surface(Arc::clone(memory)),
        }
    }

    /// The memory as a new dma-buf; `Err` says why not.
    fn export(&self) -> Result<ExportedMemory, String> {
        match self {
            Self::Buffer(buffer) => buffer.export_dma_buf(),
            Self::Surface(memory) => memory.export(),
        }
    }
}

/// How the arm writes pictures into device caches made dma-bufs, as the driver's checks let it
/// ([`cache_writes`]).
#[cfg(unix)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CacheWrites {
    /// A picture of a codec and format the driver's decode check passed is decoded into its
    /// cache, and the others are written there by its video processing where its import check
    /// passed their format.
    Decodes,
    /// The driver's video processing alone writes pictures into caches: its import check passed
    /// a format, and its decode check none, as on a driver whose decodes write their render
    /// target tiled whatever the import says (iHD, `docs/video/bugs.md`, U30).
    VideoProcessing,
}

/// What the driver's checks let the arm write into device caches made dma-bufs: `checks`, the
/// answers of its check of decodes into targets, and `writes`, its import check's. `Err` with
/// both checks' answers where neither passed anything: no cache is then made a dma-buf for the
/// video.
#[cfg(unix)]
fn cache_writes(checks: &TargetChecks, writes: &ImportWrites) -> Result<CacheWrites, String> {
    if [TargetCodec::H264, TargetCodec::Hevc]
        .into_iter()
        .any(|codec| checks.passed_any(codec))
    {
        return Ok(CacheWrites::Decodes);
    }
    match (writes.nv12(), writes.p010()) {
        (Err(nv12), Err(p010)) => Err(format!(
            "the driver's check: {}; and its video processing writes no import: NV12: {nv12}; \
             P010: {p010}",
            checks.summary()
        )),
        _ => Ok(CacheWrites::VideoProcessing),
    }
}

#[cfg(unix)]
impl<P: Platform> CacheTargets<P> {
    /// Where `device`'s platform makes caches of dma-bufs, as a cache of a 4K picture's bytes made
    /// so and exported shows, the answers of the driver's check (kept in `checks` where given),
    /// where it or the import check (`writes`) passed anything ([`cache_writes`]); `None`
    /// otherwise, logged at `info` with why.
    fn new(
        device: &P::Device,
        display: &Display,
        runtime: &Arc<dyn Runtime>,
        checks: Option<&Path>,
        writes: &ImportWrites,
        references: CachedReferences,
    ) -> Option<Self> {
        let declined = |why: &str| {
            log::info!(
                "hardware video: {NAME}: no device cache is made a dma-buf for the video: {why}"
            );
            None
        };
        let made = device.new_dma_buf_buffer_of_length_padding_unobserved(
            OnFault::Return,
            PROBE_BYTES,
            STORAGE_PRIVATE,
            None,
        );
        let probe = match made {
            Ok(Some(buffer)) => buffer,
            Ok(None) => return declined("the platform makes no cache of a dma-buf"),
            Err(raise) => return declined(&format!("a cache of a dma-buf's make: {raise}")),
        };
        if let Err(why) = probe.export_dma_buf() {
            return declined(&format!("a cache's dma-buf: {why}"));
        }
        drop(probe);
        let checks = TargetChecks::of(display, checks);
        match cache_writes(&checks, writes) {
            Err(why) => return declined(&why),
            Ok(CacheWrites::Decodes) => {
                let which = match references {
                    CachedReferences::Apart => "pictures no later picture predicts from are",
                };
                log::info!(
                    "hardware video: {NAME}: {which} decoded into device caches where the \
                     driver's check passed: {}",
                    checks.summary()
                );
            }
            Ok(CacheWrites::VideoProcessing) => log::info!(
                "hardware video: {NAME}: pictures are written into device caches by the driver's \
                 video processing, none decoded there: the driver's check: {}",
                checks.summary()
            ),
        }
        Some(Self {
            display: display.clone_ref(),
            checks,
            memories: DeviceImports::new(runtime, "VA-API's device caches", MOST_CACHES),
            references,
        })
    }

    /// What gives a decode of `codec` into `output` its picture's target: `None` where the
    /// driver's check passed none of the codec's formats, where the output's planes do not both
    /// lie in one device cache made for the host's video ([`OutputPlane::cache`]), and for a
    /// reference picture where references are kept apart (module doc). Each reason a picture's
    /// target is not given then is logged once a session, in `said`, but the check's, which the
    /// open logged, and the picture's role is counted in `roles`.
    fn target_for(
        self: &Arc<Self>,
        codec: TargetCodec,
        output: &DecodeOutput<P>,
        said: &Arc<Mutex<Vec<String>>>,
        roles: &Arc<RoleCounts>,
    ) -> Option<TargetFor> {
        if !self.checks.passed_any(codec) {
            return None;
        }
        let (luma, chroma) = (&output.luma, &output.chroma);
        let (Some(cache), Some(other)) = (&luma.cache, &chroma.cache) else {
            return None;
        };
        if cache.key() != other.key() {
            say(
                said,
                "the output's planes lie in two device caches".to_owned(),
            );
            return None;
        }
        let planes = [
            (Room::of(luma), luma.offset, luma.rows_held),
            (Room::of(chroma), chroma.offset, chroma.rows_held),
        ];
        let (targets, cache, buffer, said, roles) = (
            Arc::clone(self),
            cache.clone(),
            CacheOf::of(luma),
            Arc::clone(said),
            Arc::clone(roles),
        );
        Some(Box::new(move |shape, role| {
            if !roles.count(role, targets.references) {
                return None;
            }
            let made = targets.target(codec, shape, &cache, &buffer, planes);
            made.map_err(|why| say(&said, why)).ok()
        }))
    }

    /// The target of a picture of `shape` and `codec` in the cache `cache` (its memory `buffer`,
    /// the cache's buffer or its surface's), whose planes lie at `planes` (each its room, first
    /// byte and rows held); `Err` says why there is none.
    fn target(
        &self,
        codec: TargetCodec,
        shape: TargetShape,
        cache: &MemoryIdentity,
        buffer: &CacheOf<P>,
        planes: [(Room, u64, u64); 2],
    ) -> Result<Target, String> {
        (self.checks.passed(codec, shape.format)).map_err(|why| {
            format!(
                "the driver's check of {codec:?} into {:?}: {why}",
                shape.format
            )
        })?;
        takes_coded(
            planes.map(|(room, _, rows_held)| (room, rows_held)),
            shape.coded,
            shape.format.bytes_per_sample(),
        )?;
        let memory = self.memory(cache, buffer)?;
        let laid = planes.map(|(room, offset, _)| TargetPlane {
            offset: memory.offset + offset,
            bytes_per_row: room.bytes_per_row,
        });
        if let Some(off) = (laid.iter())
            .find(|p| p.offset % TARGET_ALIGNMENT != 0 || p.bytes_per_row % TARGET_ALIGNMENT != 0)
        {
            return Err(format!(
                "a plane at byte {} of the cache's memory, rows of {}, not on {TARGET_ALIGNMENT}",
                off.offset, off.bytes_per_row
            ));
        }
        let mut last = memory.last.lock();
        if let Some((target, at)) = &*last
            && !target.is_held()
            && target.shape() == shape
            && *at == laid
        {
            return Ok(target.clone());
        }
        let target = Target::import(
            &self.display,
            memory.dma_buf.as_fd(),
            memory.size,
            shape,
            laid,
            Box::new(memory.cache.clone_ref()),
        )
        .map_err(|e| format!("the import of the cache's memory: {e}"))?;
        *last = Some((target.clone(), laid));
        Ok(target)
    }

    /// The memory of the cache `cache` (`buffer`, the cache's buffer or its surface's), exported
    /// as a dma-buf once while the cache lives; `Err` says why not.
    fn memory(
        &self,
        cache: &MemoryIdentity,
        buffer: &CacheOf<P>,
    ) -> Result<Arc<CacheMemory<P>>, String> {
        self.memories.get_or_make(
            cache,
            0,
            || {
                let exported = buffer.export()?;
                Ok(CacheMemory {
                    dma_buf: exported.fd,
                    size: exported.allocation_size,
                    offset: exported.offset,
                    cache: buffer.clone_ref(),
                    last: Mutex::new(None),
                })
            },
            || "the cache went".to_owned(),
        )
    }

    /// Where the driver's video processing writes a picture of `bit_depth` in its output's device
    /// cache (the module doc's first way a picture lands): the layout of the output's planes in the
    /// cache, and the cache's memory as a dma-buf, where both planes lie in one cache made a
    /// dma-buf for the video; `Err` with why not.
    fn in_cache(
        &self,
        luma: &OutputPlane<P>,
        chroma: &OutputPlane<P>,
        bit_depth: u32,
    ) -> Result<(SurfaceLayout, CacheBytes<P>), String> {
        let (Some(cache), Some(other)) = (&luma.cache, &chroma.cache) else {
            return Err("an output in a device cache not made a dma-buf".to_owned());
        };
        if cache.key() != other.key() {
            return Err("an output whose planes lie in two device caches".to_owned());
        }
        let layout = in_place_layout(Place::of(luma), Place::of(chroma), bit_depth)?;
        let memory = self.memory(cache, &CacheOf::of(luma))?;
        Ok((
            layout,
            CacheBytes {
                memory,
                identity: cache.clone(),
            },
        ))
    }
}

/// A device cache's bytes as the driver's video processing writes them
/// ([`CacheTargets::in_cache`]): its memory's dma-buf, from the cache's first byte, by the cache's
/// identity, so that an import of it goes as the cache does.
#[cfg(unix)]
struct CacheBytes<P: Platform> {
    memory: Arc<CacheMemory<P>>,
    identity: MemoryIdentity,
}

#[cfg(unix)]
impl<P: Platform> DeviceMemory for CacheBytes<P> {
    fn identity(&self) -> MemoryIdentity {
        self.identity.clone()
    }

    fn length(&self) -> usize {
        let bytes = self.memory.size.saturating_sub(self.memory.offset);
        usize::try_from(bytes).unwrap_or(usize::MAX)
    }

    /// The cache's whole memory, every range at its own byte of it: one dma-buf holds the cache.
    fn dma_buf_of(&self, ranges: &[(usize, usize)]) -> Result<SurfaceDmaBuf, DmaBufError> {
        let fd = (self.memory.dma_buf.try_clone())
            .map_err(|e| DmaBufError::Refused(format!("the cache's dma-buf: {e}")))?;
        let offsets = (ranges.iter())
            .map(|&(start, _)| self.memory.offset.saturating_add(start.to_u64()))
            .collect();
        Ok(SurfaceDmaBuf {
            fd,
            size: self.memory.size,
            offsets,
        })
    }
}

/// On Windows, which makes no dma-buf, nothing decodes into a device cache or writes one in place:
/// none is made ([`CacheTargets::new`] answers `None`), and `target_for` is never called.
#[cfg(windows)]
struct CacheTargets<P: Platform> {
    never: std::convert::Infallible,
    _platform: std::marker::PhantomData<fn() -> P>,
}

#[cfg(windows)]
impl<P: Platform> CacheTargets<P> {
    /// `None`: Windows makes no dma-buf of a device cache.
    fn new(
        _device: &P::Device,
        _display: &Display,
        _runtime: &Arc<dyn Runtime>,
        _checks: Option<&Path>,
        _writes: &ImportWrites,
        _references: CachedReferences,
    ) -> Option<Self> {
        log::info!(
            "hardware video: {NAME}: no device cache is made a dma-buf for the video: this host \
             makes no dma-buf"
        );
        None
    }

    fn target_for(
        self: &Arc<Self>,
        _codec: TargetCodec,
        _output: &DecodeOutput<P>,
        _said: &Arc<Mutex<Vec<String>>>,
        _roles: &Arc<RoleCounts>,
    ) -> Option<TargetFor> {
        match self.never {}
    }
}

/// A decode session's counts of the pictures it asked a device cache for, by role, and of the
/// references it kept apart from their caches (module doc), logged at its end where it asked any.
#[derive(Debug, Default)]
struct RoleCounts {
    references: AtomicU64,
    non_references: AtomicU64,
    kept_apart: AtomicU64,
}

impl RoleCounts {
    /// Counts a picture of `role` asked a cache for; whether it may be decoded into its cache,
    /// as `references` says of a reference picture.
    #[cfg(unix)]
    fn count(&self, role: PictureRole, references: CachedReferences) -> bool {
        let (counter, taken) = match (role, references) {
            (PictureRole::NonReference, _) => (&self.non_references, true),
            (PictureRole::Reference, CachedReferences::Apart) => {
                self.kept_apart.fetch_add(1, Ordering::Relaxed);
                (&self.references, false)
            }
        };
        counter.fetch_add(1, Ordering::Relaxed);
        taken
    }

    /// The line a session logs at its end, where it asked a cache for any picture.
    fn line(&self) -> Option<String> {
        let references = self.references.load(Ordering::Relaxed);
        let non_references = self.non_references.load(Ordering::Relaxed);
        let kept_apart = self.kept_apart.load(Ordering::Relaxed);
        (references > 0 || non_references > 0).then(|| {
            format!(
                "hardware video: {NAME}: a session's pictures for device caches: {references} \
                 references ({kept_apart} kept apart, decoded in the decoder's own surfaces), \
                 {non_references} non-references"
            )
        })
    }
}

/// Logs `why` once in `said`'s session, and at info once per process (`kind`): why a picture was
/// not decoded into its cache.
#[cfg(unix)]
fn say(said: &Mutex<Vec<String>>, why: String) {
    let mut said = said.lock();
    if !said.contains(&why) {
        log::info!(
            kind:% = why;
            "hardware video: {NAME}: a picture not decoded into its device cache: {why}"
        );
        said.push(why);
    }
}

/// A session's decoder.
enum Codec {
    H264(H264Decoder),
    Hevc(HevcDecoder),
}

/// A session's decoder, and what its pictures land through.
struct VaDecoder<P: Platform> {
    codec: Codec,
    /// The id the next access unit is tagged with.
    next_id: u64,
    imports: Arc<GuestWrites>,
    /// The session's GPU copies of its pictures, or why it has none.
    copier: SessionCopier<P>,
    /// The reasons the session's pictures were copied by the CPU, or not decoded into their
    /// caches, each logged once.
    said: Arc<Mutex<Vec<String>>>,
    /// What decodes its pictures straight into their caches, where the video does.
    targets: Option<Arc<CacheTargets<P>>>,
    /// The roles of the pictures it asked a cache for.
    roles: Arc<RoleCounts>,
}

impl<P: Platform> Drop for VaDecoder<P> {
    fn drop(&mut self) {
        if let Some(line) = self.roles.line() {
            log::info!("{line}");
        }
    }
}

impl<P: Platform> VaDecoder<P> {
    /// The frames of `au`, its picture decoded into a target `target_for` gives where given.
    fn frames(
        &mut self,
        au: &[u8],
        target_for: Option<TargetFor>,
    ) -> Result<Vec<Frame>, reims_vgpu_video::vaapi::Error> {
        let id = self.next_id;
        self.next_id += 1;
        match &mut self.codec {
            Codec::H264(decoder) => decoder.decode_into(au, id, target_for),
            Codec::Hevc(decoder) => decoder.decode_into(au, id, target_for),
        }
    }
}

impl<P: Platform> HostDecoder<P> for VaDecoder<P> {
    fn parameter_sets(&mut self, sets: &[u8]) -> Result<(), BackendError> {
        match self.frames(sets, None)?.len() {
            0 => Ok(()),
            n => Err(format!("parameter sets answered {n} pictures").into()),
        }
    }

    fn decode(
        &mut self,
        au: &[u8],
        output: &DecodeOutput<P>,
    ) -> Result<Option<Box<dyn HostPicture<P>>>, BackendError> {
        let codec = match self.codec {
            Codec::H264(_) => TargetCodec::H264,
            Codec::Hevc(_) => TargetCodec::Hevc,
        };
        let target_for = (self.targets.as_ref())
            .and_then(|targets| targets.target_for(codec, output, &self.said, &self.roles));
        let mut frames = match self.frames(au, target_for) {
            Ok(frames) => frames,
            Err(error) if error.is_picture_error() => return Ok(picture_error(NAME, error)),
            Err(error) => return Err(error.into()),
        };
        let frame = match (frames.pop(), frames.len()) {
            (None, _) => return Ok(None),
            (Some(frame), 0) => frame,
            (Some(_), n) => {
                return Err(format!("an access unit answered {} pictures", n + 1).into());
            }
        };
        Ok(Some(Box::new(VaPicture {
            frame,
            imports: Arc::clone(&self.imports),
            copier: self.copier.clone(),
            said: Arc::clone(&self.said),
            targets: self.targets.clone(),
        })))
    }

    /// A picture holds its decode surface until it lands, and the decoder's pool keeps its spare
    /// surfaces for the frames its caller holds.
    fn most_unlanded(&self) -> usize {
        match self.codec {
            Codec::H264(_) => H264Decoder::SPARE_FRAMES,
            Codec::Hevc(_) => HevcDecoder::SPARE_FRAMES,
        }
    }
}

/// A decoder of VideoToolbox's paravirtual path ([`GuestDecoder`]), whose pictures land in the
/// surface the guest lent: written there by the driver's video processing where its import check
/// takes the surface's layout and memory (the module doc's first way, `Frame::write_to_guest`),
/// else read back and copied by the CPU into the surface's pages.
struct VaGuestDecoder {
    codec: Codec,
    /// The id the next access unit is tagged with.
    next_id: u64,
    imports: Arc<GuestWrites>,
    /// The reasons its pictures were copied by the CPU, each logged once.
    said: Vec<String>,
}

impl VaGuestDecoder {
    /// The frames of `bytes`, in decode order.
    fn frames(&mut self, bytes: &[u8]) -> Result<Vec<Frame>, VideoError> {
        let id = self.next_id;
        self.next_id += 1;
        let frames = match &mut self.codec {
            Codec::H264(decoder) => decoder.decode_into(bytes, id, None),
            Codec::Hevc(decoder) => decoder.decode_into(bytes, id, None),
        };
        frames.map_err(video_error)
    }

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

impl GuestDecoder for VaGuestDecoder {
    fn decode(
        &mut self,
        parameter_sets: &[u8],
        au: &[u8],
        output: &GuestOutput<'_>,
    ) -> Result<bool, VideoError> {
        self.land(parameter_sets, au, output, None)
    }

    /// The crop narrows to the region, which then lands as a whole picture does, by the driver's
    /// video processing in place or the CPU's copy.
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

impl VaGuestDecoder {
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
            let n = self.frames(parameter_sets)?.len();
            if n != 0 {
                return Err(VideoError::Malfunction(format!(
                    "parameter sets answered {n} pictures"
                )));
            }
        }
        let mut frames = self.frames(au)?;
        let mut frame = match (frames.pop(), frames.len()) {
            (None, _) => return Ok(false),
            (Some(frame), 0) => frame,
            (Some(_), n) => {
                return Err(VideoError::Malfunction(format!(
                    "an access unit answered {} pictures",
                    n + 1
                )));
            }
        };
        if let Some([x, y, width, height]) = region {
            frame
                .crop_within(x, y, width, height)
                .map_err(video_error)?;
        }
        let declined = match GuestTarget::new(&output.layout, output.memory) {
            Ok(target) => match frame.write_to_guest(&target, &self.imports) {
                Ok(DeviceWrite::Written) => return Ok(true),
                Ok(DeviceWrite::Declined(why)) => why,
                Err(error) => return Err(video_error(error)),
            },
            Err(why) => why,
        };
        self.copied_because(declined);
        let back = frame.read_back().map_err(video_error)?;
        let view = (output.memory.host_view())
            .map_err(|error| VideoError::Malfunction(error.to_string()))?;
        let sample = match frame.bit_depth() {
            8 => 1,
            _ => 2,
        };
        let [luma, chroma] = output.layout.planes[..] else {
            return Err(VideoError::Unsupported(format!(
                "a surface of {} planes",
                output.layout.planes.len()
            )));
        };
        // Each plane's bytes from its first row's first to its last row's last: luma's samples,
        // chroma's pairs of them.
        let extent = |plane: &PlaneLayout, per_column: usize| -> Option<(usize, usize, usize)> {
            let row_bytes = usize::try_from(plane.width).ok()?.checked_mul(per_column)?;
            let rows = usize::try_from(plane.height).ok()?;
            let len =
                (plane.bytes_per_row.checked_mul(rows.checked_sub(1)?))?.checked_add(row_bytes)?;
            Some((view.offset.checked_add(plane.offset)?, len, row_bytes))
        };
        let (Some((y_at, y_len, y_row)), Some((uv_at, uv_len, uv_row))) =
            (extent(&luma, sample), extent(&chroma, 2 * sample))
        else {
            return Err(VideoError::Unsupported("a plane of no rows".to_owned()));
        };
        let in_order = (y_at.checked_add(y_len)).is_some_and(|end| end <= uv_at)
            && (uv_at.checked_add(uv_len)).is_some_and(|end| end <= view.size);
        if !in_order {
            return Err(VideoError::Unsupported(
                "planes out of order or past the surface".to_owned(),
            ));
        }
        // SAFETY: the view's `size` bytes from `first` are the host's mapping of the surface's
        // pages, which `view.hold` keeps mapped until it drops after the copy; the guest's kext
        // holds the surface lent, its pages wired, until the host's `rels`, sent once this
        // returns. Luma's bytes end before chroma's begin, both inside the view (checked above).
        let all = unsafe { std::slice::from_raw_parts_mut(view.first.as_ptr(), view.size) };
        let (before, rest) = all.split_at_mut(uv_at);
        let y = &mut before[y_at..y_at + y_len];
        let uv = &mut rest[..uv_len];
        back.copy_into(
            PlaneRows {
                bytes: y,
                bytes_per_row: luma.bytes_per_row,
                rows: luma.height.to_usize(),
                row_bytes: y_row,
            },
            PlaneRows {
                bytes: uv,
                bytes_per_row: chroma.bytes_per_row,
                rows: chroma.height.to_usize(),
                row_bytes: uv_row,
            },
        )
        .map_err(video_error)?;
        drop(view);
        Ok(true)
    }
}

/// How the driver's work that `error` failed ended: stopped, for every error but a stall, whose
/// work past its deadline may still write (the VA crate's `wait`), what the landing's thread kept
/// of it from `since` on.
#[cfg(unix)]
fn work_ended(error: &reims_vgpu_video::vaapi::Error, since: KeptMark) -> Failed {
    match error {
        reims_vgpu_video::vaapi::Error::Stalled(_) => Failed::Stalled(since),
        _ => Failed::Stopped,
    }
}

/// What a VideoToolbox decoder sees of `error`: a stream that does not decode as bad data, one the
/// driver cannot decode as unsupported, and the rest as the decoder's malfunction.
fn video_error(error: reims_vgpu_video::vaapi::Error) -> VideoError {
    use reims_vgpu_video::vaapi::Error;
    match error {
        Error::Bitstream(_) | Error::HevcBitstream(_) => VideoError::BadData(error.to_string()),
        Error::Unsupported(why) => VideoError::Unsupported(why),
        error => VideoError::Malfunction(error.to_string()),
    }
}

/// A decoded frame, landed in its output by the driver's video processing, the runtime device's
/// GPU or the CPU.
struct VaPicture<P: Platform> {
    frame: Frame,
    imports: Arc<GuestWrites>,
    copier: SessionCopier<P>,
    said: Arc<Mutex<Vec<String>>>,
    /// The caches the video decodes into, where it does: the driver's video processing writes a
    /// picture decoded elsewhere into its output's cache through their memory.
    targets: Option<Arc<CacheTargets<P>>>,
}

/// A session's GPU copier, its pictures' to share, or why it has none.
type SessionCopier<P> = Result<Arc<Mutex<Box<dyn GpuCopier<P>>>>, String>;

impl<P: Platform> VaPicture<P> {
    /// Writes the picture by the driver's video processing into `memory`, laid out as `layout`
    /// (the VA crate's `Frame::write_to_guest_telling`), calling `decoded` once the decode is
    /// done and the runs are issued behind it: `None` once it is there, or why it was not
    /// written, nothing written. `Err` where the decode or a run failed or ran past its deadline
    /// ([`work_ended`] says which): no copy may race the GPU's write, so the picture fails.
    fn write_in_place(
        &self,
        layout: &SurfaceLayout,
        memory: &dyn DeviceMemory,
        decoded: &mut dyn FnMut(),
    ) -> Result<Option<String>, reims_vgpu_video::vaapi::Error> {
        let target = match GuestTarget::new(layout, memory) {
            Ok(target) => target,
            Err(why) => return Ok(Some(why)),
        };
        match (self.frame).write_to_guest_telling(&target, &self.imports, decoded)? {
            DeviceWrite::Written => Ok(None),
            DeviceWrite::Declined(why) => Ok(Some(why)),
        }
    }

    /// Logs, once a session, and at info once per process (`kind`), why a picture was copied by
    /// the CPU.
    fn copied_because(&self, why: String) {
        let mut said = self.said.lock();
        if !said.contains(&why) {
            log::info!(kind:% = why; "hardware video: {NAME}: a picture copied by the CPU: {why}");
            said.push(why);
        }
    }
}

impl<P: Platform> HostPicture<P> for VaPicture<P> {
    fn land(self: Box<Self>, to: &mut Landing<'_, P>) -> Result<(), BackendError> {
        // Decoded into its output's cache: there once its decode is done, and the device's
        // reads of the cache see it once the platform acquired what the driver wrote. A decode
        // that failed is said written all the same; one that stalled once the work the stall
        // kept is let go, the frame's target among it, which the frame keeps as it drops.
        #[cfg(unix)]
        if self.frame.is_in_target() {
            let since = KeptMark::here();
            if let Err(error) = self.frame.sync() {
                let failed = work_ended(&error, since);
                drop(self);
                return Err(to.failed_in_caches(&[to.luma], failed, error.into()));
            }
            to.luma.external_write_done()?;
            to.decoded_there();
            return Ok(());
        }
        // Into its cache's surface over Direct3D 11 textures: drawn there from the shared
        // texture it was decoded into once its decode is done, which is told first, so that what
        // is left is the draw, which the output's readers wait for; and the platform's images see
        // it once the draw's writes are said done (`graphics::drawn`). A picture not drawn is
        // copied there by the CPU.
        #[cfg(windows)]
        let undrawn = match (&to.luma.drawn, &to.chroma.drawn) {
            (Some(luma), Some(_)) => {
                match luma.planes.draw(&self.frame, &mut || to.decode_done()) {
                    Ok(()) => {
                        luma.memory.external_writes_done()?;
                        return Ok(());
                    }
                    Err(why) => Some(format!("not drawn into its cache's surface: {why}")),
                }
            }
            _ => None,
        };
        let bit_depth = self.frame.bit_depth();
        let declined = match &self.targets {
            // In its device cache, through the cache's memory as a dma-buf, where the video
            // decodes into caches: the driver's check refused the picture's codec and format, or
            // its planes do not lie as a decode's target must. Once its decode is done, what is
            // left is the runs' GPU work into the caches, which the output's readers wait for.
            #[cfg(unix)]
            Some(targets) if to.luma.cached || to.chroma.cached => {
                match targets.in_cache(to.luma, to.chroma, bit_depth) {
                    Ok((layout, bytes)) => {
                        to.writes_only(processed(&self.frame, &layout, bit_depth))?;
                        let since = KeptMark::here();
                        match self.write_in_place(&layout, &bytes, &mut || to.decode_done()) {
                            Ok(None) => {
                                to.luma.external_write_done()?;
                                return Ok(());
                            }
                            Ok(Some(why)) => format!("in its device cache: {why}"),
                            Err(error) => {
                                let failed = work_ended(&error, since);
                                return Err(to.failed_in_caches(&[to.luma], failed, error.into()));
                            }
                        }
                    }
                    Err(why) => why,
                }
            }
            _ => match in_place(to.luma, to.chroma, bit_depth) {
                Ok((layout, pages)) => {
                    to.writes_only(processed(&self.frame, &layout, bit_depth))?;
                    match self.write_in_place(&layout, &*pages.memory, &mut || {})? {
                        None => return Ok(()),
                        Some(why) => why,
                    }
                }
                Err(why) => why,
            },
        };
        #[cfg(windows)]
        let declined = match undrawn {
            Some(why) => format!("{why}; {declined}"),
            None => declined,
        };
        let declined = match &self.copier {
            // The GPU's copy writes the planes' buffer, which a plane in its cache's surface does
            // not lie in: the CPU's copy lands it there.
            _ if to.luma.in_surface() || to.chroma.in_surface() => {
                format!("{declined}; and no GPU copy into a cache's surface")
            }
            Ok(copier) => {
                if short_of(&self.frame, to.luma) {
                    to.writes_only([Extent::NONE; 2])?;
                }
                let copied = (copier.lock()).copy(&self.frame, &self.imports, to.luma, to.chroma);
                match copied? {
                    DeviceWrite::Written => {
                        to.copied_by_gpu();
                        return Ok(());
                    }
                    DeviceWrite::Declined(why) => format!("{declined}; and the GPU's copy: {why}"),
                }
            }
            Err(why) => format!("{declined}; and no GPU copy: {why}"),
        };
        self.copied_because(declined);
        let back = self.frame.read_back()?;
        let planes = to.host_for_cpu()?;
        // The copy writes the crop alone: a crop short of the planes lands over their zeros.
        if short_of(&self.frame, to.luma) {
            to.writes_only([Extent::NONE; 2])?;
        }
        let rows = |plane: &HostPlane<P>| -> Result<PlaneRows<'_>, BackendError> {
            let len = usize::try_from(plane.len()).map_err(|_| "a plane past the address space")?;
            let (Ok(bytes_per_row), Ok(rows), Ok(row_bytes)) = (
                usize::try_from(plane.bytes_per_row),
                usize::try_from(plane.rows),
                usize::try_from(plane.row_bytes),
            ) else {
                return Err("a plane past the address space".into());
            };
            // SAFETY: the plane's `len` bytes from `at` lie in its buffer's host bytes, which the
            // landing holds until it returns: the surface's pages, held by the decode, or the
            // session's staging buffer (`Landing::host`).
            let bytes = unsafe { std::slice::from_raw_parts_mut(plane.at.as_ptr::<u8>(), len) };
            Ok(PlaneRows {
                bytes,
                bytes_per_row,
                rows,
                row_bytes,
            })
        };
        back.copy_into(rows(&planes.luma)?, rows(&planes.chroma)?)?;
        Ok(())
    }
}

/// What makes the GPU copies of a video's sessions (the module doc's second way): one copier a
/// decode session, and one filler an encode session (`encode`'s module doc).
pub(crate) trait GpuCopies<P: Platform>: Send + Sync {
    /// A session's copier; `Err` with why where the device makes none.
    fn copier(&self) -> Result<Box<dyn GpuCopier<P>>, String>;

    /// An encode session's filler; `Err` with why where the device makes none.
    fn filler(&self) -> Result<Box<dyn GpuFiller<P>>, String>;

    /// The draws of the video's pictures into a cache's planes over Direct3D 11 textures
    /// (`graphics::drawn`), where the device makes them; `None` by default.
    #[cfg(windows)]
    fn plane_draws(&self) -> Option<Arc<reims_vgpu_video::vaapi::PlaneDraw>> {
        None
    }
}

/// One encode session's GPU writes of its frames into its encoder's inputs.
pub(crate) trait GpuFiller<P: Platform>: Send {
    /// Writes the frame `encoder` codes, of its settings' size and its input's samples, from
    /// `luma` and `chroma` where they lie, into an input of `encoder`'s shape, the margin past the
    /// frame filled as the VA crate's upload fills it, and returns once the GPU wrote it;
    /// [`Filled::Declined`] with why, nothing written, where it cannot. `Err` where the write
    /// failed once on the GPU, which may write the input still.
    fn fill(
        &mut self,
        encoder: &VideoEncoder,
        luma: &OutputPlane<P>,
        chroma: &OutputPlane<P>,
    ) -> Result<Filled, BackendError>;
}

/// What came of an encode session's fill of a frame ([`GpuFiller::fill`]). On Windows no filler
/// makes one: rt imports no shared Direct3D 12 texture there (`rt_windows`).
#[cfg_attr(windows, allow(dead_code))]
pub(crate) enum Filled {
    /// The frame lies in this input of the encoder's, written and waited for.
    Written(Target),
    /// Nothing was written, for the reason given: the frame is uploaded.
    Declined(String),
}

/// One session's GPU copies of its pictures into their outputs.
pub(crate) trait GpuCopier<P: Platform>: Send {
    /// Writes `frame` into `luma` and `chroma` where they lie, its crop as the CPU's copy writes
    /// it, once its decode is done, and returns once the GPU wrote it, the driver's writes of the
    /// video's memory through `imports`; [`DeviceWrite::Declined`] with why, nothing written into
    /// the planes, where it cannot. `Err` where the decode failed, or the copy failed once on the
    /// GPU, which may write the planes still.
    fn copy(
        &mut self,
        frame: &Frame,
        imports: &GuestWrites,
        luma: &OutputPlane<P>,
        chroma: &OutputPlane<P>,
    ) -> Result<DeviceWrite, BackendError>;
}

/// What one output plane holds of a picture: the bytes from one row to the next, and its crop,
/// the rows and the bytes of each that a picture decoded into it is to cover.
#[cfg(unix)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Room {
    pub(crate) bytes_per_row: u64,
    pub(crate) row_bytes: u64,
    pub(crate) height: u64,
}

#[cfg(unix)]
impl Room {
    /// What `plane` holds: its rows' pitch, and its crop.
    pub(crate) fn of<P: Platform>(plane: &OutputPlane<P>) -> Room {
        Room {
            bytes_per_row: plane.bytes_per_row,
            row_bytes: plane.row_bytes,
            height: plane.height,
        }
    }
}

/// Whether a coded picture of `width` by `height` luma samples of `sample` bytes decodes straight
/// into planes of `planes` (each its room and the rows it holds), the luma's then the chroma's:
/// each holds the coded picture's rows, and the picture covers each one's crop, so that no texel
/// of the crop keeps what it held (`video_host`'s module docs, "A short picture"): a picture
/// short of its planes decodes elsewhere, and lands with zeros past it. `Err` says why not.
#[cfg(unix)]
fn takes_coded(
    planes: [(Room, u64); 2],
    (width, height): (u32, u32),
    sample: u64,
) -> Result<(), String> {
    let row = u64::from(width) * sample;
    let rows = [u64::from(height), u64::from(height.div_ceil(2))];
    for (n, ((room, rows_held), needed)) in planes.iter().zip(rows).enumerate() {
        if *rows_held < needed || room.bytes_per_row < row {
            return Err(format!(
                "plane {n} holds {rows_held} rows of {} bytes, not the coded {width}x{height} \
                 picture's {needed} of {row}",
                room.bytes_per_row
            ));
        }
        if needed < room.height || row < room.row_bytes {
            return Err(format!(
                "the coded {width}x{height} picture's {needed} rows of {row} bytes, short of plane \
                 {n}'s {} of {}",
                room.height, room.row_bytes
            ));
        }
    }
    Ok(())
}

/// What the driver's video processing writes of `frame`, of `bit_depth`, into planes laid out as
/// `layout` (the VA crate's `Frame::write_to_guest_telling`): from the crop's first sample, the
/// layout's width and height as far as the coded picture holds them.
fn processed(frame: &Frame, layout: &SurfaceLayout, bit_depth: u32) -> [Extent; 2] {
    let (coded_width, coded_height) = frame.coded_size();
    let (x, y) = frame.crop_origin();
    let width = layout.width.min(coded_width.saturating_sub(x));
    let height = layout.height.min(coded_height.saturating_sub(y));
    let sample = if bit_depth > 8 { 2 } else { 1 };
    Extent::of_420(u64::from(width), u64::from(height), sample)
}

/// Whether `frame`'s crop is short of the output's luma plane `luma`, of its rows or its columns:
/// the CPU's copy, and a GPU copy, write the crop alone, so the planes take zeros first
/// ([`Landing::writes_only`]).
fn short_of<P: Platform>(frame: &Frame, luma: &OutputPlane<P>) -> bool {
    u64::from(frame.width()) < luma.width || u64::from(frame.height()) < luma.height
}

/// The layout of a frame of `bit_depth` written in place in the planes `luma` and `chroma`, and
/// the memory they lie in: a buffer of the planes' formats, each plane its rows and starts in the
/// memory, its even width and height within the rows it holds; `Err` with why it is not written
/// in place.
fn in_place<P: Platform>(
    luma: &OutputPlane<P>,
    chroma: &OutputPlane<P>,
    bit_depth: u32,
) -> Result<(SurfaceLayout, PlanePages), String> {
    let (Some(luma_pages), Some(chroma_pages)) = (&luma.pages, &chroma.pages) else {
        let why = if luma.cached || chroma.cached {
            "an output in a device cache"
        } else {
            "an output in memory no device is lent"
        };
        return Err(why.to_owned());
    };
    if luma_pages.memory.identity().key() != chroma_pages.memory.identity().key() {
        return Err("an output whose planes lie in two memories".to_owned());
    }
    let place = |plane: &OutputPlane<P>, pages: &PlanePages| -> Option<Place> {
        Some(Place {
            offset: pages.buffer_offset.checked_add(plane.offset)?,
            bytes_per_row: plane.bytes_per_row,
            row_bytes: plane.row_bytes,
            width: plane.width,
            height: plane.height,
            rows_held: plane.rows_held,
        })
    };
    let (Some(luma), Some(chroma)) = (place(luma, luma_pages), place(chroma, chroma_pages)) else {
        return Err("an output past 2^64 bytes".to_owned());
    };
    let layout = in_place_layout(luma, chroma, bit_depth)?;
    Ok((layout, luma_pages.clone()))
}

/// Where one plane of an output lies ([`in_place`]): its first byte in its memory, its rows, its
/// size in samples (chroma's pairs counted as one), and the rows it holds.
#[derive(Clone, Copy, Debug)]
struct Place {
    offset: u64,
    bytes_per_row: u64,
    row_bytes: u64,
    width: u64,
    height: u64,
    rows_held: u64,
}

#[cfg(unix)]
impl Place {
    /// Where `plane` lies in its buffer, from the buffer's first byte: a device cache's plane as
    /// the cache's memory holds it ([`CacheTargets::in_cache`]).
    fn of<P: Platform>(plane: &OutputPlane<P>) -> Self {
        Place {
            offset: plane.offset,
            bytes_per_row: plane.bytes_per_row,
            row_bytes: plane.row_bytes,
            width: plane.width,
            height: plane.height,
            rows_held: plane.rows_held,
        }
    }
}

/// The layout of a frame of `bit_depth` written in place at `luma` and `chroma` ([`in_place`]):
/// `420v` for 8 bits, `x420` for 10; the picture's width and height each rounded up to even, as
/// far as the planes hold them, since the run writes whole chroma rows and columns.
fn in_place_layout(luma: Place, chroma: Place, bit_depth: u32) -> Result<SurfaceLayout, String> {
    let (fourcc, sample) = match bit_depth {
        8 => (FourCc::new(*b"420v"), 1),
        10 => (FourCc::new(*b"x420"), 2),
        depth => return Err(format!("a {depth}-bit frame")),
    };
    let even = |n: u64| n.next_multiple_of(2);
    // Luma columns the planes hold: each luma row's bytes, and each chroma row's, a pair of
    // samples for every two columns.
    let columns = (luma.row_bytes / sample).min(chroma.row_bytes / sample);
    let width = even(luma.width).min(columns & !1);
    let height = even(luma.height)
        .min(luma.rows_held)
        .min(chroma.rows_held * 2);
    if width == 0 || height == 0 {
        return Err("an empty output".to_owned());
    }
    let u32_of = |n: u64| u32::try_from(n).map_err(|_| "an output past 4 GiB".to_owned());
    let usize_of = |n: u64| usize::try_from(n).map_err(|_| "an output past 4 GiB".to_owned());
    let plane = |p: Place, width: u64, height: u64| -> Result<PlaneLayout, String> {
        Ok(PlaneLayout {
            offset: usize_of(p.offset)?,
            bytes_per_row: usize_of(p.bytes_per_row)?,
            width: u32_of(width)?,
            height: u32_of(height)?,
        })
    };
    Ok(SurfaceLayout {
        pixel_format: fourcc,
        width: u32_of(width)?,
        height: u32_of(height)?,
        planes: vec![
            plane(luma, width, height)?,
            plane(chroma, width / 2, height / 2)?,
        ],
        bytes_per_row: 0,
    })
}

#[cfg(test)]
#[cfg(unix)]
mod tests;
#[cfg(test)]
#[cfg(windows)]
mod windows_tests;
