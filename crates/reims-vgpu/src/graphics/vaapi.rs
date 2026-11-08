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
//! - **A picture lands** in the output's planes one of three ways, each the fallback at the last's
//!   refusal, the session counting each:
//!   1. **In the surface's pages, by the driver's video processing** (the VA crate's
//!      `Frame::write_to_guest`): a run from the decode surface into a linear VA surface
//!      imported over a dma-buf of the planes' pages, at their own pitches. It is taken where
//!      both planes lie in one memory a dma-buf is made of ([`PlanePages`]), and where the
//!      driver's import check (the VA crate's `import_check`) found it writes imports of the
//!      frame's format right, at the alignment the planes' rows and starts are a multiple of,
//!      and in whole rows where it writes them so. The run writes the planes' whole even width
//!      and height within the coded picture, which [`layout`] tells the guest to make room for.
//!      CoreVideo's planes, 32 bytes into their pages, are written so by no driver: radeonsi
//!      imports planes 256 bytes apart alone, iHD writes a plane only at its dma-buf's first byte
//!      (`docs/video/bugs.md`, U26).
//!   2. **By the runtime device's GPU** (`gpu_copy`): the decode surface exported as dma-bufs and
//!      each plane copied into the plane's buffer, the guest's pages or the surface's device
//!      cache, at its own rows and offset, the crop alone as the CPU's copy writes it
//!      ([`crop_extent`]). It is taken wherever the open made a copier on the runtime device
//!      ([`open_on`]), the device imports the driver's surfaces as images, and the planes'
//!      buffer is their memory.
//!   3. **By the CPU**: the frame read back (staged, or `vaGetImage`, the VA crate's way back),
//!      and copied into the planes' host bytes, or into the session's staging for a plane in a
//!      device cache.
//!
//!   Each reason a session's pictures take the third way is logged once, the GPU's copy's
//!   refusal with the in-place write's: the copier's own where the session has none.
//! - **The layout** ([`layout`]) is granularity [2, 2], so that each plane holds the crop's
//!   even rows and columns, and the alignment the import check found for the codec's formats;
//!   all zero where it found none.

use std::path::Path;
use std::sync::{Arc, LazyLock, Weak};

use reims_vgpu_graphics::wire::{
    RGFX_VIDEO_CODEC_H264, RGFX_VIDEO_CODEC_HEVC, RGFX_VIDEO_DECODE, RGFX_VIDEO_ENCODE,
    RGFX_VIDEO_FORMAT_NV12, RGFX_VIDEO_FORMAT_P010, RGFX_VIDEO_H264_BASELINE, RGFX_VIDEO_H264_HIGH,
    RGFX_VIDEO_H264_MAIN, RGFX_VIDEO_HEVC_MAIN, RGFX_VIDEO_HEVC_MAIN_10, RgfxVideoCodecCaps,
    RgfxVideoEncodeCaps, RgfxVideoOutputLayout,
};
use reims_vgpu_video::core::backend::DeviceWrite;
use reims_vgpu_video::core::device_memory::GuestTarget;
use reims_vgpu_video::core::fourcc::FourCc;
use reims_vgpu_video::core::runtime::Runtime;
use reims_vgpu_video::core::surfaces::{PlaneLayout, SurfaceLayout};
use reims_vgpu_video::vaapi::{
    Capabilities, ConfigCapabilities, Display, EncoderSupport, Entrypoint, Frame, GuestWrites,
    H264Decoder, HevcDecoder, ImportWrites, PlaneRows, Profile, encoder_support,
};
use reims_vgpu_vulkan::core::platform::Handle as _;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::sync::Mutex;

use super::decode::{GraphicsVideo, GraphicsVideoSession, PlanePages};
use reims_vgpu_vulkan::VulkanPlatform;
use reims_vgpu_vulkan::gpu::device::VulkanDevice;

use super::video_host::{
    BackendError, HostDecoder, HostEncodeSession, HostPicture, HostPlane, HostVideoSession,
    Landing, OutputPlane,
};

mod encode;
mod gpu_copy;

/// The backend's name in the session's log lines.
const NAME: &str = "VA-API";

/// `VA_RT_FORMAT_YUV420` and `VA_RT_FORMAT_YUV420_10` (va.h): 8-bit and 10-bit 4:2:0.
const RT_YUV420: u32 = 0x0000_0001;
const RT_YUV420_10: u32 = 0x0000_0100;

/// The smallest picture a decode answers where its configs' surfaces give no minimum.
const MIN_SIZE: u32 = 16;

/// H.264's top level, 6.2 (`level_idc` 62), and HEVC's, 6.2 (`general_level_idc` 186).
const H264_TOP_LEVEL: u32 = 62;
const HEVC_TOP_LEVEL: u32 = 186;

/// The VA-API video of the runtime device `device`: on the render node of its GPU
/// (`host_gpus::render_node_of`), as [`open`] opens it; `None`, logged, where its GPU has none.
pub(crate) fn open_on(
    device: &VulkanDevice,
    runtime: Arc<dyn Runtime>,
) -> Option<Arc<dyn GraphicsVideo<VulkanPlatform>>> {
    let Some(node) = reims_vgpu_vulkan::host_gpus::render_node_of(&device.raw) else {
        log::info!("hardware video: no VA-API: the device's GPU has no render node");
        return None;
    };
    let copies: Arc<dyn GpuCopies<VulkanPlatform>> = Arc::new(gpu_copy::RuntimeCopies::new(device));
    open::<VulkanPlatform>(device, device.registry_id, &node, runtime, Some(copies))
}

/// The VA-API video of `device`'s GPU, on its render node `node`, its decoders' threads and waits
/// on `runtime`, its pictures copied by the GPU through `copies` where given; one per device while
/// it lives, by `registry_id`. `None`, logged at `info` with why, where libva does not load, the
/// node opens no display, or the driver decodes neither H.264 nor HEVC.
pub(crate) fn open<P: Platform>(
    device: &P::Device,
    registry_id: u64,
    node: &Path,
    runtime: Arc<dyn Runtime>,
    copies: Option<Arc<dyn GpuCopies<P>>>,
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
    let video = match VaapiVideo::<P>::new(device, node, runtime, copies) {
        Ok(video) => Arc::new(video),
        Err(why) => {
            log::info!("hardware video: no VA-API at {}: {why}", node.display());
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
}

impl<P: Platform> VaapiVideo<P> {
    fn new(
        device: &P::Device,
        node: &Path,
        runtime: Arc<dyn Runtime>,
        copies: Option<Arc<dyn GpuCopies<P>>>,
    ) -> Result<Self, String> {
        let started = runtime.now();
        let display = Display::open(node, Arc::clone(&runtime)).map_err(|e| e.to_string())?;
        let capabilities = display.capabilities().map_err(|e| e.to_string())?;
        let encoders: Arc<[EncoderSupport]> = encoder_support(&display).into();
        let mut caps = caps_of(&capabilities);
        caps.extend(encode::encode_caps_of(&encoders, &capabilities));
        if caps.is_empty() {
            return Err("the driver decodes and encodes neither H.264 nor HEVC".to_owned());
        }
        let writes_imports = display.import_writes();
        log::info!(
            "hardware video: {NAME} at {}: {}, opened in {:.2} ms on thread {:?}; in place: NV12 \
             {:?}, P010 {:?}",
            node.display(),
            display.vendor(),
            runtime.elapsed_since(started).as_secs_f64() * 1e3,
            std::thread::current().name().unwrap_or("unnamed"),
            writes_imports.nv12(),
            writes_imports.p010(),
        );
        let imports = Arc::new(GuestWrites::new(&display));
        Ok(Self {
            display,
            device: device.clone_ref(),
            caps,
            encoders,
            writes_imports,
            imports,
            copies,
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

    fn session(&self, codec: u32, direction: u32) -> Option<Arc<dyn GraphicsVideoSession<P>>> {
        let entry = (self.caps.iter()).find(|c| c.codec == codec && c.direction == direction)?;
        if direction == RGFX_VIDEO_ENCODE {
            let caps = encode::encode_caps(entry, &self.encoders);
            let encoder = encode::VaEncoder::new(
                &self.display,
                codec,
                Arc::clone(&self.encoders),
                entry,
                &caps,
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
        };
        let session = HostVideoSession::new(&self.device, codec, NAME, Box::new(decoder))?;
        Some(Arc::new(session))
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
/// High on High. NV12 out.
fn h264_caps(caps: &Capabilities) -> Option<RgfxVideoCodecCaps> {
    let of = |profile| vld(caps, profile, RT_YUV420);
    let (baseline, main, high) = (
        of(Profile::H264_CONSTRAINED_BASELINE),
        of(Profile::H264_MAIN),
        of(Profile::H264_HIGH),
    );
    let configs: Vec<&ConfigCapabilities> = [baseline, main, high].into_iter().flatten().collect();
    let mut entry = decode_caps(RGFX_VIDEO_CODEC_H264, &configs)?;
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
    /// The reasons the session's pictures were copied by the CPU, each logged once.
    said: Arc<Mutex<Vec<String>>>,
}

impl<P: Platform> VaDecoder<P> {
    fn frames(&mut self, au: &[u8]) -> Result<Vec<Frame>, BackendError> {
        let id = self.next_id;
        self.next_id += 1;
        let frames = match &mut self.codec {
            Codec::H264(decoder) => decoder.decode(au, id),
            Codec::Hevc(decoder) => decoder.decode(au, id),
        };
        Ok(frames?)
    }
}

impl<P: Platform> HostDecoder<P> for VaDecoder<P> {
    fn parameter_sets(&mut self, sets: &[u8]) -> Result<(), BackendError> {
        match self.frames(sets)?.len() {
            0 => Ok(()),
            n => Err(format!("parameter sets answered {n} pictures").into()),
        }
    }

    fn decode(&mut self, au: &[u8]) -> Result<Option<Box<dyn HostPicture<P>>>, BackendError> {
        let mut frames = self.frames(au)?;
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

/// A decoded frame, landed in its output by the driver's video processing, the runtime device's
/// GPU or the CPU.
struct VaPicture<P: Platform> {
    frame: Frame,
    imports: Arc<GuestWrites>,
    copier: SessionCopier<P>,
    said: Arc<Mutex<Vec<String>>>,
}

/// A session's GPU copier, its pictures' to share, or why it has none.
type SessionCopier<P> = Result<Arc<Mutex<Box<dyn GpuCopier<P>>>>, String>;

impl<P: Platform> VaPicture<P> {
    /// Logs, once a session, why a picture was copied by the CPU.
    fn copied_because(&self, why: String) {
        let mut said = self.said.lock();
        if !said.contains(&why) {
            log::info!("hardware video: {NAME}: a picture copied by the CPU: {why}");
            said.push(why);
        }
    }
}

impl<P: Platform> HostPicture<P> for VaPicture<P> {
    fn land(self: Box<Self>, to: &mut Landing<'_, P>) -> Result<(), BackendError> {
        let declined = match in_place(to.luma, to.chroma, self.frame.bit_depth()) {
            Ok((layout, pages)) => {
                let target = GuestTarget::new(&layout, &*pages.memory);
                match target.map(|target| self.frame.write_to_guest(&target, &self.imports)) {
                    Ok(Ok(DeviceWrite::Written)) => return Ok(()),
                    Ok(Ok(DeviceWrite::Declined(why))) => why,
                    // The decode or a run failed or ran past its deadline: no copy may race the
                    // GPU's write, so the picture fails.
                    Ok(Err(error)) => return Err(error.into()),
                    Err(why) => why,
                }
            }
            Err(why) => why,
        };
        let declined = match &self.copier {
            Ok(copier) => match copier.lock().copy(&self.frame, to.luma, to.chroma)? {
                DeviceWrite::Written => return Ok(()),
                DeviceWrite::Declined(why) => format!("{declined}; and the GPU's copy: {why}"),
            },
            Err(why) => format!("{declined}; and no GPU copy: {why}"),
        };
        self.copied_because(declined);
        let back = self.frame.read_back()?;
        let planes = to.host_for_cpu()?;
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
/// session.
pub(crate) trait GpuCopies<P: Platform>: Send + Sync {
    /// A session's copier; `Err` with why where the device makes none.
    fn copier(&self) -> Result<Box<dyn GpuCopier<P>>, String>;
}

/// One session's GPU copies of its pictures into their outputs.
pub(crate) trait GpuCopier<P: Platform>: Send {
    /// Writes `frame` into `luma` and `chroma` where they lie, its crop as the CPU's copy writes
    /// it ([`crop_extent`]), once its decode is done, and returns once the GPU wrote it;
    /// [`DeviceWrite::Declined`] with why, nothing written, where it cannot. `Err` where the
    /// decode failed, or the copy failed once on the GPU, which may write the planes still.
    fn copy(
        &mut self,
        frame: &Frame,
        luma: &OutputPlane<P>,
        chroma: &OutputPlane<P>,
    ) -> Result<DeviceWrite, BackendError>;
}

/// What one output plane holds of a picture, as the CPU's copy fits a picture to it: the bytes of
/// its rows, the bytes from one row to the next, and its rows.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Room {
    pub(crate) row_bytes: u64,
    pub(crate) bytes_per_row: u64,
    pub(crate) rows: u64,
}

impl Room {
    /// What `plane` holds: its row's bytes and pitch, and its crop's rows.
    pub(crate) fn of<P: Platform>(plane: &OutputPlane<P>) -> Room {
        Room {
            row_bytes: plane.row_bytes,
            bytes_per_row: plane.bytes_per_row,
            rows: plane.height,
        }
    }
}

/// The columns and rows of a picture of `size`, of `sample`-byte samples, that the CPU's copy
/// writes into planes of `luma` and `chroma` room (the VA crate's `ReadBack::copy_into`): as many
/// as the picture, each plane's rows and its rows' bytes hold, the chroma plane's columns in whole
/// pairs; `None` for none. The chroma plane takes half of each, rounded up, from half the crop's
/// origin.
pub(crate) fn crop_extent(
    size: (u32, u32),
    sample: u64,
    luma: Room,
    chroma: Room,
) -> Option<(u32, u32)> {
    let fit = |bytes: u64| u32::try_from(bytes / sample.max(1)).unwrap_or(u32::MAX);
    let rows = |rows: u64| u32::try_from(rows).unwrap_or(u32::MAX);
    let chroma_columns = fit(chroma.row_bytes.min(chroma.bytes_per_row)) & !1;
    let width = (size.0)
        .min(fit(luma.row_bytes.min(luma.bytes_per_row)))
        .min(chroma_columns.max(1));
    let height = (size.1)
        .min(rows(luma.rows))
        .min(rows(chroma.rows).saturating_mul(2));
    (width > 0 && height > 0).then_some((width, height))
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
mod tests;
