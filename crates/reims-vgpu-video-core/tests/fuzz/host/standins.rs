//! What the host session runs on under the fuzzer: decoders, encoders and
//! a JPEG decoder of our own that decode nothing and reach no GPU, driver
//! or VideoToolbox, each checking what the session promises it; and guest
//! surfaces laid out every way a guest can lay them out, which count their
//! holds. Each stand-in's answers follow from a hash of what it is given,
//! not from the order the session's threads ask in, so that an input
//! replays as it ran.

use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use reims_vgpu_video_core::backend::{
    BackendDecoder, BackendError, Codec, CodecCapabilities, CompressedFrame, DecodedFrame,
    DecoderBackend, DecoderConfig, DeviceWrite, FrameDecoded, FrameReadBack, PixelFormat,
    ProfileSupport, copy_biplanar, read_back_by,
};
use reims_vgpu_video_core::device_memory::{
    DeviceMemory, GuestTarget, HostView, MemoryIdentity, MemoryLife,
};
use reims_vgpu_video_core::encoder_backend::{
    BackendEncoder, EncodedFrame, EncoderBackend, EncoderCapabilities, EncoderConfig,
    EncoderSettings, SourceFrame,
};
use reims_vgpu_video_core::fourcc::FourCc;
use reims_vgpu_video_core::jpeg::{
    DecodedSamples, JpegCapabilities, JpegDecode, JpegDecoder, JpegDecodes, JpegError,
    JpegOutputFormat, JpegSubsampling,
    SampleView, write_bgra, write_samples,
};
use reims_vgpu_video_core::plist;
use reims_vgpu_video_core::status::OsStatus;
use reims_vgpu_video_core::surfaces::{
    GuestSurface, MappingId, PlaneDestination, PlaneLayout, SurfaceError, SurfaceLayout, Surfaces,
};
use reims_vgpu_video_core::sync::Mutex;

use crate::support::{Rng, mix};

/// The codecs a backend can list.
pub const CODECS: [Codec; 4] = [Codec::H264, Codec::Hevc, Codec::Av1, Codec::Vp9];

/// The formats a decoder writes and an encoder reads.
pub const FORMATS: [PixelFormat; 4] = [
    PixelFormat::Nv12VideoRange,
    PixelFormat::Nv12FullRange,
    PixelFormat::P010VideoRange,
    PixelFormat::P010FullRange,
];

/// The pixel formats a guest surface may have: the decoders', the JPEG
/// engine's, and others no stand-in takes.
pub const SURFACE_FORMATS: [[u8; 4]; 10] = [
    *b"420v", *b"420f", *b"x420", *b"xf20", *b"422v", *b"444f", *b"yuvs", *b"2vuy", *b"BGRA",
    *b"L008",
];

/// How many surfaces a guest has lent the host, with mapping IDs from 1.
pub const SURFACES: u64 = 8;

/// The most memory one surface has: a guest's layout may claim more.
const MOST_SURFACE_BYTES: u64 = 2 << 20;

/// The largest side of a decoded frame, so that copies stay small.
const LARGEST_FRAME_SIDE: u32 = 256;

/// A hash of `bytes` under `seed`: what a stand-in's answers follow from.
pub fn hash(seed: u64, bytes: &[u8]) -> u64 {
    mix(bytes.iter().fold(seed ^ 0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    }))
}

/// A backend failure, one input in `one_in` by `h`, of each kind.
fn failure(h: u64, one_in: u64) -> Option<BackendError> {
    if !h.is_multiple_of(one_in) {
        return None;
    }
    let message = "the stand-in fails".to_owned();
    Some(match (h >> 20) % 4 {
        0 => BackendError::Unsupported(message),
        1 => BackendError::BadData(message),
        2 => BackendError::Malfunction(message),
        _ => BackendError::Native {
            status: [-12909, -12911, -1, 0, 1, i32::MIN][((h >> 24) % 6) as usize],
            message,
        },
    })
}

/// A size of the host's to the stand-ins' frames: at most
/// [`LARGEST_FRAME_SIDE`] a side, and at least 1.
fn frame_size(width: u32, height: u32) -> (u32, u32) {
    let side = |side: u32| {
        if side > LARGEST_FRAME_SIDE {
            side % LARGEST_FRAME_SIDE + 1
        } else {
            side.max(1)
        }
    };
    (side(width), side(height))
}

fn to_usize(value: u64) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

/// One guest surface: its layout as the guest made it, and its memory,
/// which need not be as long as the layout says.
pub struct Surface {
    /// Its layout.
    pub layout: SurfaceLayout,
    memory: Mutex<Vec<u8>>,
    /// Whether the mapper can hold it.
    holdable: bool,
    /// Whether its memory can be reached.
    reachable: bool,
    /// Whether the host lends its memory to devices, as reims-vgpu lends
    /// a reachable surface's; never one it cannot reach.
    lent_to_devices: bool,
    /// Its mapping's life, as long as the pool's.
    life: MemoryLife,
}

impl Surface {
    /// A surface of any layout, most often one a guest's CoreVideo makes.
    fn new(rng: &mut Rng) -> Self {
        let format = if rng.one_in(16) {
            FourCc(rng.next_u32())
        } else {
            FourCc::new(*rng.pick(&SURFACE_FORMATS))
        };
        let side = |rng: &mut Rng| {
            if rng.one_in(10) {
                *rng.pick(&[0, 1, 2, 3, 15, 17, 63, 64, 65, 255, 1 << 15, u32::MAX])
            } else {
                rng.range_u32(1, 200)
            }
        };
        let (width, height) = (side(rng), side(rng));
        let (w, h) = (u64::from(width), u64::from(height));
        let pairs = w.div_ceil(2);
        let shapes: Vec<(u64, u64, u32)> = match &format.0.to_be_bytes() {
            b"420v" | b"420f" => vec![(w, h, width), (pairs * 2, h.div_ceil(2), width.div_ceil(2))],
            b"x420" | b"xf20" => vec![
                (w * 2, h, width),
                (pairs * 4, h.div_ceil(2), width.div_ceil(2)),
            ],
            b"422v" => vec![(w, h, width), (pairs * 2, h, width.div_ceil(2))],
            b"444f" => vec![(w, h, width), (w * 2, h, width)],
            b"yuvs" | b"2vuy" => vec![(pairs * 4, h, width)],
            b"BGRA" => vec![(w * 4, h, width)],
            _ => vec![(w, h, width)],
        };
        let align = *rng.pick(&[1, 2, 16, 64, 256]);
        let mut planes = Vec::new();
        let mut offset = if rng.one_in(8) { rng.range(1, 64) } else { 0 };
        for (row, rows, samples) in shapes {
            let mut bytes_per_row = row.div_ceil(align).saturating_mul(align);
            if rng.one_in(4) {
                bytes_per_row = bytes_per_row.saturating_add(rng.range(0, 64));
            }
            let mut rows = rows;
            if rng.one_in(20) {
                bytes_per_row = bytes_per_row.saturating_sub(rng.range(1, 8));
            }
            if rng.one_in(20) {
                rows = rows.saturating_sub(1);
            }
            if rng.one_in(40) && !planes.is_empty() {
                // Overlapping the plane before.
                offset = offset.saturating_sub(rng.range(1, 64));
            }
            planes.push(PlaneLayout {
                offset: to_usize(offset),
                bytes_per_row: to_usize(bytes_per_row),
                width: samples,
                height: u32::try_from(rows).unwrap_or(u32::MAX),
            });
            offset = offset.saturating_add(bytes_per_row.saturating_mul(rows));
            if rng.one_in(3) {
                offset = offset.div_ceil(256).saturating_mul(256);
            }
        }
        if rng.one_in(24) {
            planes.pop();
        } else if rng.one_in(24) {
            planes.push(PlaneLayout {
                offset: to_usize(offset),
                bytes_per_row: 16,
                width: 16,
                height: 1,
            });
            offset = offset.saturating_add(16);
        }
        let mut length = offset;
        if rng.one_in(16) {
            length = length.saturating_sub(rng.range(1, 100));
        }
        if length > MOST_SURFACE_BYTES {
            length = rng.range(0, 4096);
        }
        let (planes, bytes_per_row) = match planes.as_slice() {
            [only] if rng.bool() => (Vec::new(), only.bytes_per_row),
            _ => (planes, 0),
        };
        let mut memory = vec![rng.below(256) as u8; to_usize(length)];
        let prefix = rng.some_bytes(64);
        for (byte, value) in memory.iter_mut().zip(prefix) {
            *byte = value;
        }
        Self {
            layout: SurfaceLayout {
                pixel_format: format,
                width,
                height,
                planes,
                bytes_per_row,
            },
            memory: Mutex::new(memory),
            holdable: !rng.one_in(24),
            reachable: !rng.one_in(24),
            lent_to_devices: !rng.one_in(4),
            life: MemoryLife::new(),
        }
    }

    /// Writes `bytes` from the start of its memory, as far as they fit:
    /// the stream a JPEG decode reads.
    pub fn fill(&self, bytes: &[u8]) {
        let mut memory = self.memory.lock();
        let length = memory.len().min(bytes.len());
        memory[..length].copy_from_slice(&bytes[..length]);
    }
}

/// The guest surfaces the host can hold, by mapping ID, and how many holds
/// are live.
pub struct Pool {
    surfaces: HashMap<u64, Arc<Surface>>,
    holds: Arc<AtomicUsize>,
}

impl Pool {
    /// Surfaces 1 to [`SURFACES`], of any layout.
    pub fn new(rng: &mut Rng) -> Self {
        Self {
            surfaces: (1..=SURFACES)
                .map(|id| (id, Arc::new(Surface::new(rng))))
                .collect(),
            holds: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// The surface with mapping ID `id`.
    pub fn get(&self, id: u64) -> Option<&Arc<Surface>> {
        self.surfaces.get(&id)
    }

    /// Whether the surface with mapping ID `id` can be held.
    pub fn holdable(&self, id: u64) -> bool {
        self.surfaces
            .get(&id)
            .is_some_and(|surface| surface.holdable)
    }

    /// The holds the host has now.
    pub fn holds(&self) -> usize {
        self.holds.load(Ordering::SeqCst)
    }
}

thread_local! {
    /// The surfaces this thread has lent to it now, in a read or a write.
    static LENDS: Cell<usize> = const { Cell::new(0) };
}

/// Runs `lend` as a lend of a surface, counted on this thread while it
/// runs: a lend inside another, which reims-vgpu's mapper, lending one
/// surface at a time in the process, would make wait on itself for ever,
/// an encoder asked to encode inside one, or a JPEG decode, is the
/// session's fault.
fn lent<T>(lend: impl FnOnce() -> T) -> T {
    LENDS.with(|lends| {
        assert_eq!(lends.get(), 0, "a surface's lend inside another's");
        lends.set(lends.get() + 1);
    });
    let result = lend();
    LENDS.with(|lends| lends.set(lends.get() - 1));
    result
}

/// A hold on one surface, counted until it is dropped.
struct Held {
    surface: Arc<Surface>,
    holds: Arc<AtomicUsize>,
    device: Device,
}

/// A held surface's memory as the stand-in host lends it to a device: a
/// view of exactly its bytes, which the hold keeps.
struct Device {
    identity: MemoryIdentity,
    surface: Arc<Surface>,
}

impl DeviceMemory for Device {
    fn identity(&self) -> MemoryIdentity {
        self.identity.clone()
    }

    fn length(&self) -> usize {
        self.surface.memory.lock().len()
    }

    fn host_view(&self) -> Result<HostView, SurfaceError> {
        let mut memory = self.surface.memory.lock();
        Ok(HostView {
            first: std::ptr::NonNull::new(memory.as_mut_ptr())
                .ok_or_else(|| SurfaceError::Unreachable("an empty surface".into()))?,
            size: memory.len(),
            offset: 0,
            hold: Box::new(Arc::clone(&self.surface)),
        })
    }

    /// The stand-in's IOSurface is its surface's address, which the
    /// stand-in device checks and never dereferences.
    #[cfg(target_os = "macos")]
    fn io_surface(
        &self,
    ) -> Result<reims_vgpu_video_core::device_memory::LentIoSurface<'_>, SurfaceError> {
        let surface = std::ptr::NonNull::from(&*self.surface).cast();
        // SAFETY: the surface lives as long as this, which the hold keeps.
        Ok(unsafe {
            reims_vgpu_video_core::device_memory::LentIoSurface::new(
                surface,
                self.surface.reachable,
            )
        })
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        self.holds.fetch_sub(1, Ordering::SeqCst);
    }
}

impl GuestSurface for Held {
    fn layout(&self) -> SurfaceLayout {
        self.surface.layout.clone()
    }

    fn write(&self, write: &mut dyn FnMut(&mut [u8])) -> Result<(), SurfaceError> {
        if !self.surface.reachable {
            return Err(SurfaceError::Unreachable("the stand-in's memory".into()));
        }
        lent(|| write(&mut self.surface.memory.lock()));
        Ok(())
    }

    fn read(&self, read: &mut dyn FnMut(&[u8])) -> Result<(), SurfaceError> {
        if !self.surface.reachable {
            return Err(SurfaceError::Unreachable("the stand-in's memory".into()));
        }
        lent(|| read(&self.surface.memory.lock()));
        Ok(())
    }

    fn device_memory(&self) -> Option<&dyn DeviceMemory> {
        (self.surface.reachable && self.surface.lent_to_devices)
            .then_some(&self.device as &dyn DeviceMemory)
    }
}

impl Surfaces for Pool {
    fn retain(&self, mapping_id: MappingId) -> Result<Box<dyn GuestSurface>, SurfaceError> {
        let surface = self
            .surfaces
            .get(&mapping_id.0)
            .ok_or(SurfaceError::Unknown(mapping_id.0))?;
        if !surface.holdable {
            return Err(SurfaceError::Unreachable("the stand-in's mapper".into()));
        }
        self.holds.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Held {
            surface: Arc::clone(surface),
            holds: Arc::clone(&self.holds),
            device: Device {
                identity: surface.life.identity(),
                surface: Arc::clone(surface),
            },
        }))
    }
}

/// A side for a backend's largest size.
fn largest(rng: &mut Rng) -> u32 {
    if rng.one_in(10) {
        *rng.pick(&[0, 63, 64, 640])
    } else {
        *rng.pick(&[1920, 4096, 8192, 16384, u32::MAX])
    }
}

/// Stand-in decoders of the codecs they list.
pub struct Decoders {
    seed: u64,
    capabilities: Vec<CodecCapabilities>,
}

impl Decoders {
    /// Decoders of most codecs, with any profiles and largest sizes.
    pub fn new(rng: &mut Rng) -> Self {
        let mut capabilities = Vec::new();
        for codec in CODECS {
            if rng.one_in(5) {
                continue;
            }
            let mut profiles = Vec::new();
            for _ in 0..rng.count(3) {
                let profile = if rng.one_in(8) {
                    rng.interesting_u32()
                } else {
                    *rng.pick(&[0, 1, 2, 66, 77, 100, 110])
                };
                let max_level = match rng.below(3) {
                    0 => None,
                    1 => Some(rng.range_u32(0, 200)),
                    _ => Some(rng.interesting_u32()),
                };
                profiles.push(ProfileSupport { profile, max_level });
            }
            capabilities.push(CodecCapabilities {
                codec,
                profiles,
                max_width: largest(rng),
                max_height: largest(rng),
            });
        }
        Self {
            seed: rng.next_u64(),
            capabilities,
        }
    }
}

impl DecoderBackend for Decoders {
    fn capabilities(&self) -> Vec<CodecCapabilities> {
        self.capabilities.clone()
    }

    fn open(&self, config: &DecoderConfig) -> Result<Box<dyn BackendDecoder>, BackendError> {
        assert!(
            self.capabilities.iter().any(|c| c.codec == config.codec),
            "a {:?} decoder opened, which the backend does not list",
            config.codec
        );
        let h = hash(
            self.seed ^ (u64::from(config.width) << 32 | u64::from(config.height)),
            &config.configuration,
        );
        if let Some(error) = failure(h, 8) {
            return Err(error);
        }
        Ok(Box::new(Decoder {
            seed: h,
            codec: config.codec,
            size: frame_size(config.width, config.height),
            format: FORMATS[((h >> 8) % 4) as usize],
            held: VecDeque::new(),
            reorder: [0, 0, 1, 2][((h >> 12) % 4) as usize],
            av1: Av1Steps::default(),
        }))
    }
}

/// One stand-in decoder: frames come out of it in the configuration's
/// size, now and then another, held back up to its reorder depth. An AV1
/// one says, now and then, what each decode decoded and showed again.
struct Decoder {
    seed: u64,
    codec: Codec,
    size: (u32, u32),
    format: PixelFormat,
    held: VecDeque<Frame>,
    reorder: usize,
    av1: Av1Steps,
}

/// The AV1 frames the stand-in decoders said they decoded or showed again,
/// which says how deep the inputs get.
pub static AV1_STEPS: AtomicU64 = AtomicU64::new(0);

/// What an AV1 decoder's temporal units decoded and showed, as the codec
/// crate's decoder says ([`BackendDecoder::frames_decoded`]): hidden
/// frames, frames shown again from a reference slot, and its eight slots,
/// each frame refreshing any of them.
#[derive(Default)]
struct Av1Steps {
    /// The next frame's number.
    next_picture: u64,
    /// The frame each slot holds.
    slots: [Option<u64>; 8],
    /// The last decode's steps, until the session asks for them.
    last: Vec<FrameDecoded>,
}

impl Av1Steps {
    /// The steps of a decode, by `h`, that output `outputs` frames of
    /// `size`: as many hidden frames as `h` says among them, and each
    /// output decoded or shown again. None, one time in four, as a decoder
    /// that does not say.
    fn of(&mut self, h: u64, outputs: usize, size: (u32, u32), largest: (u32, u32)) {
        self.last.clear();
        if h.is_multiple_of(4) {
            return;
        }
        let mut h = h;
        let mut draw = |n: u64| {
            h = mix(h);
            h % n
        };
        let hidden = [0, 0, 0, 1, 2, 3, 20][draw(7) as usize];
        let mut left = outputs;
        let mut hidden_left = hidden;
        while left > 0 || hidden_left > 0 {
            let output = hidden_left == 0 || (left > 0 && draw(2) == 0);
            if output {
                left -= 1;
            } else {
                hidden_left -= 1;
            }
            let held: Vec<u64> = self.slots.iter().flatten().copied().collect();
            let again = output && !held.is_empty() && draw(5) == 0;
            let picture = if again {
                held[draw(held.len() as u64) as usize]
            } else {
                let picture = self.next_picture;
                self.next_picture += 1;
                // A key frame refreshes every slot; others, any of them.
                let refresh = if draw(6) == 0 { 0xff } else { draw(256) };
                for (slot, held) in self.slots.iter_mut().enumerate() {
                    if refresh & (1 << slot) != 0 {
                        *held = Some(picture);
                    }
                }
                picture
            };
            let primary_reference = (!again && !held.is_empty() && draw(2) == 0)
                .then(|| held[draw(held.len() as u64) as usize]);
            self.last.push(FrameDecoded {
                picture,
                shown_again: again,
                refused: false,
                width: size.0,
                height: size.1,
                references: self.slots,
                primary_reference,
                output,
                largest,
            });
        }
    }
}

impl BackendDecoder for Decoder {
    fn output_format(&self) -> PixelFormat {
        self.format
    }

    fn decode(
        &mut self,
        frame: &CompressedFrame<'_>,
    ) -> Result<Vec<Box<dyn DecodedFrame>>, BackendError> {
        let h = hash(self.seed ^ frame.frame_id.cast_unsigned(), frame.data);
        if let Some(config) = frame.new_config {
            if config.codec != self.codec {
                return Err(BackendError::BadData("another codec".into()));
            }
            if h.is_multiple_of(7) {
                return Err(BackendError::Unsupported("this configuration".into()));
            }
            self.size = frame_size(config.width, config.height);
        }
        if let Some(error) = failure(h, 20) {
            return Err(error);
        }
        if (h >> 8).is_multiple_of(16) {
            // A sample with no picture: of an AV1 decoder, now and then
            // hidden frames alone.
            if self.codec == Codec::Av1 {
                self.av1.of(h >> 32, 0, self.size, self.size);
            }
            return Ok(Vec::new());
        }
        let (width, height) = if (h >> 16).is_multiple_of(13) {
            frame_size(self.size.0 + 16, self.size.1 + 2)
        } else {
            self.size
        };
        let format = if (h >> 24).is_multiple_of(17) {
            FORMATS[((h >> 28) % 4) as usize]
        } else {
            self.format
        };
        self.held.push_back(Frame {
            frame_id: frame.frame_id,
            format,
            width,
            height,
            seed: h,
        });
        let mut out: Vec<Box<dyn DecodedFrame>> = Vec::new();
        while self.held.len() > self.reorder {
            if let Some(frame) = self.held.pop_front() {
                out.push(Box::new(frame));
            }
        }
        if self.codec == Codec::Av1 {
            self.av1.of(h >> 32, out.len(), (width, height), self.size);
        }
        Ok(out)
    }

    fn frames_decoded(&mut self) -> Vec<FrameDecoded> {
        let steps = std::mem::take(&mut self.av1.last);
        AV1_STEPS.fetch_add(steps.len() as u64, Ordering::Relaxed);
        steps
    }

    fn flush(&mut self) -> Result<Vec<Box<dyn DecodedFrame>>, BackendError> {
        if let Some(error) = failure(self.seed >> 3, 16) {
            return Err(error);
        }
        Ok(self
            .held
            .drain(..)
            .map(|frame| Box::new(frame) as Box<dyn DecodedFrame>)
            .collect())
    }

    fn can_accept(&self, config: &DecoderConfig) -> bool {
        config.codec == self.codec && hash(self.seed, &config.configuration).is_multiple_of(2)
    }
}

/// A stand-in decoded frame, whose samples are made when it is copied.
struct Frame {
    frame_id: i64,
    format: PixelFormat,
    width: u32,
    height: u32,
    seed: u64,
}

impl DecodedFrame for Frame {
    fn frame_id(&self) -> i64 {
        self.frame_id
    }

    fn pixel_format(&self) -> PixelFormat {
        self.format
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn read_back(&self) -> Result<Box<dyn FrameReadBack + '_>, BackendError> {
        if self.seed.is_multiple_of(31) {
            return Err(BackendError::Malfunction("the read back fails".into()));
        }
        Ok(read_back_by(|planes| self.copy_to(planes)))
    }

    /// A device's write: never inside a lend, now and then declined or
    /// failed, and otherwise each plane's rows of what fits, each byte
    /// checked to lie inside the surface's memory, which the target's
    /// checks promise.
    fn write_to_guest(&self, target: &GuestTarget<'_>) -> Result<DeviceWrite, BackendError> {
        assert_eq!(
            LENDS.with(Cell::get),
            0,
            "a device's write inside a surface's lend"
        );
        if (self.seed >> 4).is_multiple_of(5) {
            return Ok(DeviceWrite::Declined("the stand-in's device".into()));
        }
        if (self.seed >> 4).is_multiple_of(37) {
            return Err(BackendError::Malfunction("the device's write fails".into()));
        }
        assert_eq!(
            target.pixel_format().bytes_per_sample(),
            self.format.bytes_per_sample(),
            "a buffer of other samples"
        );
        // On macOS the host's IOSurface of the buffer comes with it, and
        // says the host would write the buffer: no other is ever lent.
        #[cfg(target_os = "macos")]
        {
            let lent = (target.memory().io_surface())
                .map_err(|error| BackendError::Malfunction(error.to_string()))?;
            assert!(lent.writable(), "a buffer lent to a device is read-only");
        }
        let view = target
            .memory()
            .host_view()
            .map_err(|error| BackendError::Malfunction(error.to_string()))?;
        let sample = self.format.bytes_per_sample();
        let (width, height) = target.written_size(self.width, self.height);
        let width = width as usize;
        let rows = [
            (height, width * sample, self.seed as u8),
            (height.div_ceil(2), width.div_ceil(2) * 2 * sample, 0x80),
        ];
        for (plane, (rows, row_bytes, value)) in target.planes().iter().zip(rows) {
            let end = plane.end_of(rows, row_bytes).unwrap_or(usize::MAX);
            assert!(
                view.offset.saturating_add(end) <= view.size,
                "a device told to write past the surface's {} bytes, to {end}: {target:?}",
                view.size
            );
            for row in 0..rows as usize {
                let at = view.offset + plane.offset + row * plane.bytes_per_row;
                // SAFETY: inside the view, checked above, which its hold
                // keeps.
                unsafe { view.first.as_ptr().add(at).write_bytes(value, row_bytes) };
            }
        }
        Ok(DeviceWrite::Written)
    }
}

impl Frame {
    fn copy_to(&self, planes: &mut [PlaneDestination<'_>]) -> Result<(), BackendError> {
        if self.seed.is_multiple_of(29) {
            return Err(BackendError::Malfunction("the copy fails".into()));
        }
        let sample = self.format.bytes_per_sample();
        let (width, height) = (self.width as usize, self.height as usize);
        let luma_stride = width * sample;
        let chroma_stride = width.div_ceil(2) * 2 * sample;
        let luma = vec![self.seed as u8; luma_stride * height];
        let chroma = vec![0x80; chroma_stride * height.div_ceil(2)];
        copy_biplanar(
            self.format,
            self.width,
            self.height,
            [(&luma, luma_stride), (&chroma, chroma_stride)],
            planes,
        );
        Ok(())
    }
}

/// What the stand-in encoders made, by frame ID: a hash of each sample's
/// bytes, which the guest checks each emitted frame against.
pub type Produced = Arc<Mutex<HashMap<i64, Vec<u64>>>>;

/// Stand-in encoders of the codecs they list.
pub struct Encoders {
    seed: u64,
    capabilities: Vec<EncoderCapabilities>,
    produced: Produced,
}

impl Encoders {
    /// Encoders of H.264 and HEVC most often, with any profiles, sizes and
    /// source formats.
    pub fn new(rng: &mut Rng, produced: Produced) -> Self {
        let mut capabilities = Vec::new();
        for codec in CODECS {
            let offered = match codec {
                Codec::H264 | Codec::Hevc => !rng.one_in(5),
                Codec::Av1 | Codec::Vp9 => rng.one_in(4),
            };
            if !offered {
                continue;
            }
            let mut profiles = Vec::new();
            for _ in 0..rng.count(3) {
                profiles.push(if rng.one_in(8) {
                    rng.interesting_u32()
                } else {
                    *rng.pick(&[1, 2, 66, 77, 100])
                });
            }
            let source_formats = FORMATS
                .iter()
                .copied()
                .filter(|_| rng.below(3) != 0)
                .collect();
            capabilities.push(EncoderCapabilities {
                codec,
                profiles,
                max_width: largest(rng),
                max_height: largest(rng),
                source_formats,
                frame_reordering: rng.bool(),
            });
        }
        Self {
            seed: rng.next_u64(),
            capabilities,
            produced,
        }
    }
}

impl EncoderBackend for Encoders {
    fn encoders(&self) -> Vec<EncoderCapabilities> {
        self.capabilities.clone()
    }

    fn open(&self, config: &EncoderConfig) -> Result<Box<dyn BackendEncoder>, BackendError> {
        let offered = self.capabilities.iter().find(|c| c.codec == config.codec);
        assert!(
            offered.is_some(),
            "a {:?} encoder opened, which the backend does not list",
            config.codec
        );
        let Some(offered) = offered else {
            return Err(BackendError::Unsupported("not listed".into()));
        };
        assert!(
            (1..=offered.max_width).contains(&config.width)
                && (1..=offered.max_height).contains(&config.height),
            "an encoder opened at {}x{}, past its {}x{}",
            config.width,
            config.height,
            offered.max_width,
            offered.max_height
        );
        let h = hash(
            self.seed ^ (u64::from(config.width) << 32 | u64::from(config.height)),
            format!("{:?}", config.settings).as_bytes(),
        );
        if let Some(error) = failure(h, 10) {
            return Err(error);
        }
        let reorder = if offered.frame_reordering && config.settings.allow_frame_reordering {
            ((h >> 8) % 3) as usize
        } else {
            0
        };
        Ok(Box::new(Encoder {
            seed: h,
            formats: offered.source_formats.clone(),
            size: (config.width, config.height),
            held: VecDeque::new(),
            reorder,
            configuration: record(h),
            frames: 0,
            produced: Arc::clone(&self.produced),
        }))
    }

    fn check_property(
        &self,
        codec: Codec,
        key: &str,
        value: &plist::Value,
    ) -> Result<(), OsStatus> {
        let h = hash(self.seed ^ codec.fourcc().0 as u64, key.as_bytes());
        let h = hash(h, format!("{value:?}").as_bytes());
        if h.is_multiple_of(9) {
            Err(-12900)
        } else {
            Ok(())
        }
    }
}

/// A configuration record: none, or bytes of any length.
fn record(h: u64) -> Vec<u8> {
    if h.is_multiple_of(5) {
        return Vec::new();
    }
    (0..(h >> 8) % 60)
        .map(|i| (h >> (i % 8 * 8)) as u8)
        .collect()
}

/// One stand-in encoder.
struct Encoder {
    seed: u64,
    formats: Vec<PixelFormat>,
    size: (u32, u32),
    held: VecDeque<EncodedFrame>,
    reorder: usize,
    configuration: Vec<u8>,
    frames: u64,
    produced: Produced,
}

impl Encoder {
    /// The frames out of the reordering window, oldest first.
    fn ready(&mut self) -> Vec<EncodedFrame> {
        let mut out = Vec::new();
        while self.held.len() > self.reorder {
            out.extend(self.held.pop_front());
        }
        out
    }
}

impl BackendEncoder for Encoder {
    fn configuration(&self) -> Vec<u8> {
        self.configuration.clone()
    }

    fn encode(&mut self, frame: &SourceFrame<'_>) -> Result<Vec<EncodedFrame>, BackendError> {
        assert_eq!(
            LENDS.with(Cell::get),
            0,
            "an encode inside a lend of a guest surface"
        );
        assert!(
            self.formats.contains(&frame.pixel_format),
            "a source frame in {:?}, which the encoder does not take",
            frame.pixel_format
        );
        assert_eq!(
            (frame.width, frame.height),
            self.size,
            "a source frame of another size than the session's"
        );
        // The planes hold the picture, as the session promises: every row
        // of each is read.
        let sample = frame.pixel_format.bytes_per_sample();
        let (width, height) = (frame.width as usize, frame.height as usize);
        assert_eq!(
            frame.planes.len(),
            2,
            "a source frame of {} planes",
            frame.planes.len()
        );
        let [luma, chroma] = frame.planes else {
            return Err(BackendError::Malfunction("not two planes".into()));
        };
        let mut sum = 0u64;
        for (plane, row_bytes, rows) in [
            (luma, width * sample, height),
            (chroma, width.div_ceil(2) * 2 * sample, height.div_ceil(2)),
        ] {
            assert!(
                plane.rows as usize >= rows && plane.bytes_per_row >= row_bytes,
                "a source plane of {} rows of {} bytes for {rows} of {row_bytes}",
                plane.rows,
                plane.bytes_per_row
            );
            for row in 0..rows {
                let start = row * plane.bytes_per_row;
                let bytes = &plane.bytes[start..start + row_bytes];
                sum = sum.wrapping_add(u64::from(bytes.first().copied().unwrap_or(0)));
                sum = sum.wrapping_add(u64::from(bytes.last().copied().unwrap_or(0)));
            }
        }
        let h = mix(self.seed ^ frame.frame_id.cast_unsigned() ^ sum);
        self.frames += 1;
        if let Some(error) = failure(h, 14) {
            return Err(error);
        }
        if let Some(error) = failure(h.rotate_left(29), 13) {
            // Taken, and failed with its own error, in its place among the
            // frames held with it.
            self.held
                .push_back(EncodedFrame::failed(frame.frame_id, error));
            return Ok(self.ready());
        }
        if h.is_multiple_of(23) {
            // A new stream format, mid-stream.
            self.configuration = record(h >> 1);
        }
        let length = match (h >> 8) % 16 {
            0 => 0,
            1 => 1,
            2 => 0x3fc0,
            3 => 0x4000,
            4 => 0x4040,
            5 => 40_000,
            6 if (h >> 16).is_multiple_of(8) => 300_000,
            _ => ((h >> 32) % 2000) as usize,
        };
        let data = vec![(h >> 40) as u8; length];
        self.produced
            .lock()
            .entry(frame.frame_id)
            .or_default()
            .push(hash(0, &data));
        let sync = frame.force_key_frame || self.frames == 1 || h.is_multiple_of(5);
        self.held.push_back(EncodedFrame {
            frame_id: frame.frame_id,
            data,
            presentation_time: frame.presentation_time,
            decode_time: frame.presentation_time,
            duration: frame.duration,
            sync,
            depends_on_others: !sync,
            depended_on_by_others: (h >> 4).is_multiple_of(2),
            error: None,
        });
        Ok(self.ready())
    }

    fn flush(&mut self) -> Result<Vec<EncodedFrame>, BackendError> {
        if let Some(error) = failure(self.seed >> 5, 16) {
            return Err(error);
        }
        Ok(self.held.drain(..).collect())
    }

    fn reconfigure(&mut self, settings: &EncoderSettings) -> Result<(), BackendError> {
        if hash(self.seed, format!("{settings:?}").as_bytes()).is_multiple_of(4) {
            Err(BackendError::Unsupported(
                "this setting while running".into(),
            ))
        } else {
            Ok(())
        }
    }
}

/// The largest JPEG the stand-in makes samples for: a decode of a larger
/// one is refused as bad data.
const LARGEST_JPEG_PIXELS: u64 = 1 << 20;

/// A stand-in JPEG decoder. Its samples are made of the stream's hash and
/// written by core's own writer, as a backend writes them.
pub struct Jpeg {
    capabilities: JpegCapabilities,
}

impl Jpeg {
    /// An engine of any capabilities.
    pub fn new(rng: &mut Rng) -> Self {
        Self {
            capabilities: JpegCapabilities {
                cores: *rng.pick(&[0, 1, 2, u32::MAX]),
                sixteen_bit_quantization_tables: rng.bool(),
                missing_eoi: rng.bool(),
                restart_logging: rng.bool(),
                interchange_formats: rng.bool(),
            },
        }
    }
}

/// The rows and bytes a row each plane of `format` needs for a `width`
/// by `height` output, as `JpegDecoder::decode` promises them.
fn jpeg_planes(format: JpegOutputFormat, width: usize, height: usize) -> Vec<(usize, usize)> {
    let pairs = width.div_ceil(2);
    match format {
        JpegOutputFormat::Biplanar420 => vec![(width, height), (pairs * 2, height.div_ceil(2))],
        JpegOutputFormat::Biplanar422 => vec![(width, height), (pairs * 2, height)],
        JpegOutputFormat::Biplanar444 => vec![(width, height), (width * 2, height)],
        JpegOutputFormat::Yuyv | JpegOutputFormat::Uyvy => vec![(pairs * 4, height)],
        JpegOutputFormat::Bgra => vec![(width * 4, height)],
    }
}

impl JpegDecoder for Jpeg {
    fn decodes(&self) -> JpegDecodes {
        JpegDecodes {
            subsamplings: vec![
                JpegSubsampling::Yuv420,
                JpegSubsampling::Yuv422,
                JpegSubsampling::Yuv444,
                JpegSubsampling::Grey,
            ],
            max_size: (u32::from(u16::MAX), u32::from(u16::MAX)),
        }
    }

    fn capabilities(&self) -> JpegCapabilities {
        self.capabilities
    }

    fn decode(
        &self,
        request: &JpegDecode<'_>,
        planes: &mut [PlaneDestination<'_>],
    ) -> Result<(), JpegError> {
        assert_eq!(
            LENDS.with(Cell::get),
            0,
            "a JPEG decode inside a lend of a guest surface"
        );
        let (width, height) = request.output_size();
        let wanted = jpeg_planes(request.format, width as usize, height as usize);
        assert_eq!(
            planes.len(),
            wanted.len(),
            "{} planes for {:?}",
            planes.len(),
            request.format
        );
        for (plane, &(row_bytes, rows)) in planes.iter().zip(&wanted) {
            assert!(
                plane.rows as usize >= rows
                    && plane.bytes_per_row >= row_bytes
                    && (rows == 0
                        || plane.bytes.len() >= (rows - 1) * plane.bytes_per_row + row_bytes),
                "a plane of {} bytes, {} rows of {}, for {rows} rows of {row_bytes}",
                plane.bytes.len(),
                plane.rows,
                plane.bytes_per_row
            );
        }
        let region = request.region;
        assert!(
            region.width > 0
                && region.height > 0
                && u64::from(region.x) + u64::from(region.width) <= u64::from(request.width)
                && u64::from(region.y) + u64::from(region.height) <= u64::from(request.height),
            "a region {region:?} outside the {}x{} image",
            request.width,
            request.height
        );
        let h = hash(u64::from(request.scale), request.data);
        match h % 16 {
            0 => return Err(JpegError::BadData("the stand-in".into())),
            1 => return Err(JpegError::UnsupportedStream("the stand-in".into())),
            2 => {
                return Err(JpegError::SizeMismatch {
                    width: request.width ^ 1,
                    height: request.height,
                });
            }
            3 => return Err(JpegError::SubsamplingMismatch),
            4 => return Err(JpegError::Malfunction("the stand-in".into())),
            _ => {}
        }
        if u64::from(request.width) * u64::from(request.height) > LARGEST_JPEG_PIXELS {
            return Err(JpegError::BadData(
                "larger than the stand-in decodes".into(),
            ));
        }
        // Planes as a driver's image holds them: rows padded past their
        // samples, after an offset, and chroma in planes of its own or
        // interleaved, as NV12's.
        let pad = ((h >> 40) % 3) as usize * 16;
        let offset = ((h >> 44) % 2) as usize * 64;
        if request.format == JpegOutputFormat::Bgra {
            // Converted by the backend, as VA-API's processing does.
            let (width, height) = (width as usize, height as usize);
            let pitch = width * 4 + pad;
            let pixels = vec![(h >> 8) as u8; offset + pitch * height];
            let view = SampleView {
                bytes: &pixels,
                offset,
                pitch,
                step: 4,
            };
            return write_bgra(&view, request.output_size(), planes);
        }
        let (image_width, image_height) = (request.width as usize, request.height as usize);
        let pitch = image_width + pad;
        let luma = vec![(h >> 8) as u8; offset + pitch * image_height];
        let interleaved = (h >> 48).is_multiple_of(2);
        let chroma = request.subsampling.chroma_divisors().map(|(across, down)| {
            let (w, h) = (
                image_width.div_ceil(across as usize),
                image_height.div_ceil(down as usize),
            );
            let pitch = if interleaved { 2 * w } else { w } + pad;
            (vec![0x80u8; offset + pitch * h], pitch)
        });
        let samples = DecodedSamples {
            subsampling: request.subsampling,
            luma: view(&luma, offset, pitch, 1),
            chroma: chroma.as_ref().map(|(bytes, pitch)| {
                if interleaved {
                    [
                        view(bytes, offset, *pitch, 2),
                        view(bytes, offset + 1, *pitch, 2),
                    ]
                } else {
                    [
                        view(bytes, offset, *pitch, 1),
                        view(bytes, offset, *pitch, 1),
                    ]
                }
            }),
        };
        write_samples(&samples, request, planes)
    }
}

/// A component's samples from `offset`, `pitch` bytes a row and `step` a
/// sample.
fn view(bytes: &[u8], offset: usize, pitch: usize, step: usize) -> SampleView<'_> {
    SampleView {
        bytes,
        offset,
        pitch,
        step,
    }
}
