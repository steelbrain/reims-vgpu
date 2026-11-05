//! Decoder sessions end to end: a guest decoder that sends what the
//! guest's does, a device thread that delivers queue 0 in order as the
//! device does, and a backend whose frames are a fill pattern.
//!
//! The guest sends what its decoder sends: `+dec`, `dsta`, `dfrm` cut into
//! fragments at 0x200000 bytes, `dfin`, `-dec`. It answers the host's
//! `Dbuf` with a buffer laid out as `Dpba` asked and `Demt` by taking the
//! buffer back, and drops what a `Dfub` names.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Sender};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::backend::{
    BackendDecoder, BackendError, Codec, CodecCapabilities, CompressedFrame, DecodedFrame,
    DecoderBackend, DecoderConfig, DeviceWrite, FrameDecoded, FrameReadBack, PixelFormat,
    ProfileSupport, copy_biplanar, read_back_by,
};
use crate::device_memory::{DeviceMemory, GuestTarget, HostView, MemoryIdentity, MemoryLife};
use crate::device_wait::GuestHold;
use crate::fourcc::FourCc;
use crate::host::test_queues::{CountingSink, Progress};
use crate::host::{HostSession, names};
use crate::runtime::{Runtime, StdRuntime};
use crate::status::{NO_ERROR, OsStatus, Status};
use crate::surfaces::{
    GuestSurface, MappingId, PlaneDestination, PlaneLayout, SurfaceError, SurfaceLayout, Surfaces,
};
use crate::sync::Mutex;
use crate::wire::WireError;
use crate::wire::image_description::{
    AVC_CONFIGURATION, HEVC_CONFIGURATION, ImageDescription, VP9_CONFIGURATION,
};
use crate::wire::message::{GuestUuid, Message, MessageFlags, MessageWriter, read_header};
use crate::wire::property_list::as_i64;
use crate::wire::sample_buffer::{
    MEDIA_TYPE_VIDEO, SampleBuffer, SerializedFormatDescription, SerializedValue,
};
use crate::wire::surface_boxes::read_surface_release;

const WIDTH: u32 = 64;
const HEIGHT: u32 = 48;
/// The guest's buffers' rows: wider than the frame, so that a copy at the
/// wrong stride shows.
const BYTES_PER_ROW: usize = 80;
const CHROMA_OFFSET: usize = 4096;
const AVCC: &[u8] = &[1, 77, 0, 30, 0xff, 0xe0, 0];
const OTHER_AVCC: &[u8] = &[1, 100, 0, 40, 0xff, 0xe0, 0];
/// The guest decoder's fragment limit.
const FRAGMENT_LIMIT: usize = 0x20_0000;
const WAIT: Duration = Duration::from_secs(5);

// The device's glue moves the session between threads.
const _: fn() = || {
    fn send_and_sync<T: Send + Sync>() {}
    send_and_sync::<HostSession>();
};

/// What the backend saw of each frame.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Submitted {
    frame_id: i64,
    length: usize,
    reconfigured: Option<Vec<u8>>,
}

/// A backend that decodes H.264, HEVC as Main 10 would, and VP9, into
/// frames filled with the sample's first two bytes, luma then chroma; an
/// empty sample has no picture, and one starting 0xff does not decode. A
/// VP9 or AV1 sample starting 0xfe is a frame of half the description's
/// size.
#[derive(Default)]
struct PatternBackend {
    opened: Mutex<Vec<DecoderConfig>>,
    submitted: Arc<Mutex<Vec<Submitted>>>,
    gate: Arc<Gate>,
    /// Whether the decoders it opens from now on decode while copying,
    /// with each frame's copy on its own thread.
    decodes_while_copying: AtomicBool,
    /// The decoders' frames' work, held while closed, and what they did.
    frames: Arc<FrameWork>,
    /// The frames each AV1 or VP9 sample's decode takes, in order, for a
    /// test that gives them ([`BackendDecoder::frames_decoded`]): each frame
    /// output is filled with its number as luma, and a sample with a frame
    /// refused does not decode.
    units: Arc<Mutex<VecDeque<Vec<FrameDecoded>>>>,
}

/// What a test holds of the frames' work: the wait for the picture and the
/// copy each pass a gate of their own, and the copy takes `copy_time`, a
/// device's write as a read back does.
#[derive(Default)]
struct FrameWork {
    wait: Gate,
    copy: Gate,
    copy_time: Mutex<Duration>,
    /// Whether the frames' device writes the guest's buffers itself where
    /// the host lends them; else it declines.
    device_writes: AtomicBool,
    /// Whether the device's writes fail.
    device_fails: AtomicBool,
    /// Whether the device's writes miss their deadline, the device still
    /// writing: the write fails as a malfunction and keeps the buffer's
    /// hold in `still_writing`, as a backend keeps it with its work.
    device_stalls: AtomicBool,
    /// The holds of buffers a stalled device still writes.
    still_writing: Mutex<Vec<GuestHold>>,
    /// `decode N`, `waited N`, `copied N` (by the CPU) and `written N` (by
    /// the device), in the order they happened.
    events: Mutex<Vec<String>>,
}

impl FrameWork {
    fn happened(&self, event: String) {
        self.events.lock().push(event);
    }

    fn events(&self) -> Vec<String> {
        self.events.lock().clone()
    }

    /// Where `event` happened among the events, or `usize::MAX`.
    fn at(&self, event: &str) -> usize {
        self.events
            .lock()
            .iter()
            .position(|seen| seen == event)
            .unwrap_or(usize::MAX)
    }
}

/// Holds the backend's decodes while closed, so a test can queue messages
/// behind a frame in hand. Open by default.
#[derive(Default)]
struct Gate {
    closed: Mutex<bool>,
    changed: crate::sync::Condvar,
}

impl Gate {
    fn close(&self) {
        *self.closed.lock() = true;
    }

    fn open(&self) {
        *self.closed.lock() = false;
        self.changed.notify_all();
    }

    /// Waits while the gate is closed, for the test's wait at most.
    fn pass(&self) {
        let started = StdRuntime.now();
        let mut closed = self.closed.lock();
        while *closed && StdRuntime.elapsed_since(started) < WAIT {
            self.changed
                .wait_for(&mut closed, Duration::from_millis(10));
        }
    }
}

impl DecoderBackend for PatternBackend {
    fn capabilities(&self) -> Vec<CodecCapabilities> {
        let listed = |codec, profiles: &[u32]| CodecCapabilities {
            codec,
            profiles: profiles
                .iter()
                .map(|&profile| ProfileSupport {
                    profile,
                    max_level: None,
                })
                .collect(),
            max_width: 4096,
            max_height: 4096,
        };
        vec![
            listed(Codec::H264, &[66, 77, 100]),
            listed(Codec::Hevc, &[1, 2]),
            listed(Codec::Av1, &[0]),
            listed(Codec::Vp9, &[0, 2]),
        ]
    }

    fn open(&self, config: &DecoderConfig) -> Result<Box<dyn BackendDecoder>, BackendError> {
        if config.configuration.is_empty() {
            return Err(BackendError::Unsupported("no avcC".into()));
        }
        self.opened.lock().push(config.clone());
        Ok(Box::new(PatternDecoder {
            codec: config.codec,
            width: config.width,
            height: config.height,
            submitted: self.submitted.clone(),
            gate: self.gate.clone(),
            decodes_while_copying: self.decodes_while_copying.load(Ordering::SeqCst),
            frames: self.frames.clone(),
            units: self.units.clone(),
            frames_decoded: Vec::new(),
        }))
    }
}

struct PatternDecoder {
    codec: Codec,
    width: u32,
    height: u32,
    submitted: Arc<Mutex<Vec<Submitted>>>,
    gate: Arc<Gate>,
    decodes_while_copying: bool,
    frames: Arc<FrameWork>,
    units: Arc<Mutex<VecDeque<Vec<FrameDecoded>>>>,
    /// The last decode's frames, from `units`.
    frames_decoded: Vec<FrameDecoded>,
}

impl PatternDecoder {
    fn frame(
        &self,
        frame_id: i64,
        (width, height): (u32, u32),
        luma: u8,
        chroma: u8,
    ) -> PatternFrame {
        PatternFrame {
            frame_id,
            width,
            height,
            luma,
            chroma,
            work: self.frames.clone(),
        }
    }
}

impl BackendDecoder for PatternDecoder {
    fn output_format(&self) -> PixelFormat {
        if self.codec == Codec::Hevc {
            PixelFormat::P010VideoRange
        } else {
            PixelFormat::Nv12VideoRange
        }
    }

    fn decode(
        &mut self,
        frame: &CompressedFrame<'_>,
    ) -> Result<Vec<Box<dyn DecodedFrame>>, BackendError> {
        self.submitted.lock().push(Submitted {
            frame_id: frame.frame_id,
            length: frame.data.len(),
            reconfigured: frame.new_config.map(|config| config.configuration.clone()),
        });
        self.frames.happened(format!("decode {}", frame.frame_id));
        self.gate.pass();
        if matches!(self.codec, Codec::Av1 | Codec::Vp9)
            && let Some(unit) = self.units.lock().pop_front()
        {
            if unit.iter().any(|step| step.refused) {
                self.frames_decoded = unit;
                return Err(BackendError::BadData("a frame refused".into()));
            }
            let shown = unit
                .iter()
                .filter(|step| step.output)
                .map(|step| {
                    Box::new(self.frame(
                        frame.frame_id,
                        (step.width, step.height),
                        u8::try_from(step.picture).unwrap_or(u8::MAX),
                        0x80,
                    )) as Box<dyn DecodedFrame>
                })
                .collect();
            self.frames_decoded = unit;
            return Ok(shown);
        }
        match frame.data {
            [] => Ok(Vec::new()),
            [0xff, ..] => Err(BackendError::BadData("a broken slice".into())),
            [0xfd, ..] => Err(BackendError::SkippedLeadingPicture("a RASL picture".into())),
            [0xfe, rest @ ..] if matches!(self.codec, Codec::Vp9 | Codec::Av1) => {
                Ok(vec![Box::new(self.frame(
                    frame.frame_id,
                    (self.width / 2, self.height / 2),
                    0xfe,
                    rest.first().copied().unwrap_or(128),
                ))])
            }
            [luma, rest @ ..] => Ok(vec![Box::new(self.frame(
                frame.frame_id,
                (self.width, self.height),
                *luma,
                rest.first().copied().unwrap_or(128),
            ))]),
        }
    }

    fn flush(&mut self) -> Result<Vec<Box<dyn DecodedFrame>>, BackendError> {
        Ok(Vec::new())
    }

    fn can_accept(&self, config: &DecoderConfig) -> bool {
        config.codec == self.codec
    }

    fn decodes_while_copying(&self) -> bool {
        self.decodes_while_copying
    }

    fn frames_decoded(&mut self) -> Vec<FrameDecoded> {
        std::mem::take(&mut self.frames_decoded)
    }
}

struct PatternFrame {
    frame_id: i64,
    width: u32,
    height: u32,
    luma: u8,
    chroma: u8,
    work: Arc<FrameWork>,
}

impl DecodedFrame for PatternFrame {
    fn frame_id(&self) -> i64 {
        self.frame_id
    }

    fn pixel_format(&self) -> PixelFormat {
        PixelFormat::Nv12VideoRange
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    /// The copy's gate and its time come before the lend, as a GPU's read
    /// back does, and the CPU's copy in it.
    fn read_back(&self) -> Result<Box<dyn FrameReadBack + '_>, BackendError> {
        self.work.copy.pass();
        std::thread::sleep(*self.work.copy_time.lock());
        Ok(read_back_by(|planes| self.copy_to(planes)))
    }

    fn wait(&self) -> Result<(), BackendError> {
        self.work.wait.pass();
        self.work.happened(format!("waited {}", self.frame_id));
        Ok(())
    }

    /// The stand-in device's write: through the host's view of the
    /// buffer, each plane's rows of the frame's size or what fits, after the
    /// copy's gate and time, as a GPU's write would take them.
    fn write_to_guest(&self, target: &GuestTarget<'_>) -> Result<DeviceWrite, BackendError> {
        if !self.work.device_writes.load(Ordering::SeqCst) {
            return Ok(DeviceWrite::Declined(
                "the stand-in's device writes nothing".into(),
            ));
        }
        self.work.copy.pass();
        std::thread::sleep(*self.work.copy_time.lock());
        if self.work.device_fails.load(Ordering::SeqCst) {
            return Err(BackendError::Malfunction(
                "the stand-in's device failed".into(),
            ));
        }
        if self.work.device_stalls.load(Ordering::SeqCst) {
            self.work.still_writing.lock().push(target.hold());
            self.work.happened(format!("stalled {}", self.frame_id));
            return Err(BackendError::Malfunction(
                "the stand-in's device did not finish in time".into(),
            ));
        }
        // A device writes no buffer the host would not write itself: the
        // IOSurface a macOS host lends says so, and none such is lent.
        #[cfg(target_os = "macos")]
        {
            let lent = (target.memory().io_surface())
                .map_err(|error| BackendError::Malfunction(error.to_string()))?;
            assert!(lent.writable(), "a buffer lent to a device is read-only");
        }
        let view = match target.memory().host_view() {
            Ok(view) => view,
            Err(error) => return Ok(DeviceWrite::Declined(error.to_string())),
        };
        let (width, height) = target.written_size(self.width, self.height);
        let width = usize::try_from(width).unwrap_or(0);
        let planes = [
            (height, self.luma, width),
            (height.div_ceil(2), self.chroma, width.div_ceil(2) * 2),
        ];
        for (plane, (rows, value, row_bytes)) in target.planes().iter().zip(planes) {
            for row in 0..usize::try_from(rows).unwrap_or(0) {
                let at = view.offset + plane.offset + row * plane.bytes_per_row;
                assert!(at + row_bytes <= view.size, "a write past the view");
                // SAFETY: inside the view, which its hold keeps, as the
                // target's planes are inside the surface's memory.
                unsafe { view.first.as_ptr().add(at).write_bytes(value, row_bytes) };
            }
        }
        self.work.happened(format!("written {}", self.frame_id));
        Ok(DeviceWrite::Written)
    }
}

impl PatternFrame {
    fn copy_to(&self, planes: &mut [PlaneDestination<'_>]) -> Result<(), BackendError> {
        let stride = usize::try_from(self.width).unwrap_or(0);
        let rows = usize::try_from(self.height).unwrap_or(0);
        let luma = vec![self.luma; stride * rows];
        let chroma = vec![self.chroma; stride * rows / 2];
        copy_biplanar(
            PixelFormat::Nv12VideoRange,
            self.width,
            self.height,
            [(&luma, stride), (&chroma, stride)],
            planes,
        );
        self.work.happened(format!("copied {}", self.frame_id));
        Ok(())
    }
}

/// What runs as the host's hold on a surface ends, before it has.
type OnRelease = Box<dyn FnOnce() + Send>;

/// The guest's IOSurfaces, as the host's mapper reaches them: memory by
/// mapping ID, and the holds the host has on each.
#[derive(Default)]
struct GuestMemory {
    surfaces: Mutex<HashMap<u64, Arc<SurfaceMemory>>>,
    holds: Arc<Mutex<HashMap<u64, usize>>>,
    /// What runs, once, as the host lets go of its hold on a surface, by
    /// mapping ID, on the thread that lets it go: the hold has not ended
    /// yet, as a host's release of it takes time.
    on_release: Arc<Mutex<HashMap<u64, OnRelease>>>,
}

struct SurfaceMemory {
    layout: SurfaceLayout,
    bytes: Mutex<Vec<u8>>,
    /// Made read-only for its clients by the guest: the glue's platforms
    /// refuse to lend it for writing, or to a device.
    read_only: bool,
    /// Whether the host lends devices its memory.
    lent_to_devices: bool,
    /// Its mapping's life: it goes with the surface.
    life: MemoryLife,
    /// How many times a device's write of it was told of
    /// (`GuestSurface::device_wrote`).
    wrote: AtomicUsize,
}

struct HeldSurface {
    mapping_id: u64,
    memory: Arc<SurfaceMemory>,
    holds: Arc<Mutex<HashMap<u64, usize>>>,
    on_release: Arc<Mutex<HashMap<u64, OnRelease>>>,
    device: StandInDevice,
}

/// A held surface's memory as the stand-in host lends it to a device: a
/// view of its bytes, held by the hold on the surface, and nothing past
/// them.
struct StandInDevice {
    identity: MemoryIdentity,
    memory: Arc<SurfaceMemory>,
}

impl DeviceMemory for StandInDevice {
    fn identity(&self) -> MemoryIdentity {
        self.identity.clone()
    }

    fn length(&self) -> usize {
        self.memory.bytes.lock().len()
    }

    fn host_view(&self) -> Result<HostView, crate::surfaces::SurfaceError> {
        let mut bytes = self.memory.bytes.lock();
        Ok(HostView {
            first: std::ptr::NonNull::new(bytes.as_mut_ptr()).ok_or_else(|| {
                crate::surfaces::SurfaceError::Unreachable("an empty surface".into())
            })?,
            size: bytes.len(),
            offset: 0,
            hold: Box::new(self.memory.clone()),
        })
    }

    /// The stand-in's IOSurface is its memory's address, which no stand-in
    /// backend dereferences, lent as writable where the host would write
    /// the buffer.
    #[cfg(target_os = "macos")]
    fn io_surface(&self) -> Result<crate::device_memory::LentIoSurface<'_>, SurfaceError> {
        let surface = std::ptr::NonNull::from(&*self.memory).cast();
        // SAFETY: the memory lives as long as this, which the hold keeps.
        Ok(unsafe { crate::device_memory::LentIoSurface::new(surface, !self.memory.read_only) })
    }
}

impl GuestSurface for HeldSurface {
    fn layout(&self) -> SurfaceLayout {
        self.memory.layout.clone()
    }

    fn write(&self, write: &mut dyn FnMut(&mut [u8])) -> Result<(), SurfaceError> {
        if self.memory.read_only {
            return Err(SurfaceError::Unreachable(
                "read-only for its clients".into(),
            ));
        }
        write(&mut self.memory.bytes.lock());
        Ok(())
    }

    fn device_memory(&self) -> Option<&dyn DeviceMemory> {
        (self.memory.lent_to_devices && !self.memory.read_only)
            .then_some(&self.device as &dyn DeviceMemory)
    }

    fn device_wrote(&self) {
        self.memory.wrote.fetch_add(1, Ordering::SeqCst);
    }
}

impl Drop for HeldSurface {
    fn drop(&mut self) {
        let on_release = self.on_release.lock().remove(&self.mapping_id);
        if let Some(on_release) = on_release {
            on_release();
        }
        if let Some(count) = self.holds.lock().get_mut(&self.mapping_id) {
            *count -= 1;
        }
    }
}

impl Surfaces for GuestMemory {
    fn retain(&self, mapping_id: MappingId) -> Result<Box<dyn GuestSurface>, SurfaceError> {
        let memory = self
            .surfaces
            .lock()
            .get(&mapping_id.0)
            .cloned()
            .ok_or(SurfaceError::Unknown(mapping_id.0))?;
        *self.holds.lock().entry(mapping_id.0).or_default() += 1;
        Ok(Box::new(HeldSurface {
            mapping_id: mapping_id.0,
            device: StandInDevice {
                identity: memory.life.identity(),
                memory: memory.clone(),
            },
            memory,
            holds: self.holds.clone(),
            on_release: self.on_release.clone(),
        }))
    }
}

impl GuestMemory {
    fn held(&self) -> usize {
        self.holds.lock().values().sum()
    }

    /// How many times a device's write of surface `mapping_id` was told of.
    fn wrote(&self, mapping_id: u64) -> usize {
        (self.surfaces.lock().get(&mapping_id)).map_or(0, |s| s.wrote.load(Ordering::SeqCst))
    }
}

/// An emitted frame as the guest's session got it.
#[derive(Clone, Debug)]
struct Emitted {
    frame_id: i64,
    status: i32,
    info: u32,
    /// The buffer's luma at the first and last pixel, the byte past the
    /// first row's last pixel, and the chroma at the first sample.
    samples: Option<[u8; 4]>,
}

/// What the guest's side keeps.
#[derive(Default)]
struct GuestState {
    /// The host's messages in arrival order, as `Dbuf 3`, `Demt 3`,
    /// `rels`, `reply dfrm`.
    seen: Mutex<Vec<String>>,
    /// The same with the buffers each names, by mapping ID: `Dpba 64x48`,
    /// `Dbuf 3 -> 4` (frame 3, buffer 4 lent), `Demt 3 4` (`-` for none),
    /// `Dfub [4]`, `rels [4]`, `reply dfrm`.
    buffers: Mutex<Vec<String>>,
    attributes: Mutex<Option<plist::Dictionary>>,
    /// The pixel buffers lent to the host, by UUID: their mapping IDs.
    lent: Mutex<HashMap<GuestUuid, u64>>,
    emitted: Mutex<Vec<Emitted>>,
    /// Whether the guest answers `Dbuf`; a stalled guest does not.
    stalled: AtomicBool,
    /// Whether the buffers it lends are read-only for their clients.
    read_only_buffers: AtomicBool,
    /// Whether the host lends devices no memory of its buffers.
    buffers_kept_from_devices: AtomicBool,
    /// Whether its buffers' memory ends a byte before their layout's last
    /// row.
    short_buffers: AtomicBool,
    next_surface: AtomicU64,
    waiting: Mutex<HashMap<u64, Sender<Vec<u8>>>>,
}

/// The guest decoder, its handler thread, and the device's queue-0 thread.
struct Guest {
    uuid: GuestUuid,
    state: Arc<GuestState>,
    memory: Arc<GuestMemory>,
    backend: Arc<PatternBackend>,
    session: Arc<HostSession>,
    queue0: Option<Sender<Vec<u8>>>,
    next_reply_id: AtomicU64,
    /// How many messages the device thread has delivered, sweeps and all.
    delivered: Arc<AtomicU64>,
    /// How far the queues have got.
    progress: Arc<Progress>,
    threads: Vec<JoinHandle<()>>,
}

fn header_of(bytes: &[u8]) -> Option<crate::wire::message::MessageHeader> {
    read_header(bytes, 0).ok()
}

impl Guest {
    fn new() -> Self {
        let memory = Arc::new(GuestMemory::default());
        let backend = Arc::new(PatternBackend::default());
        let (to_guest, from_host) = mpsc::channel::<Vec<u8>>();
        let progress = Arc::new(Progress::default());
        let session = Arc::new(HostSession::new(
            StdRuntime::shared(),
            Arc::new(CountingSink::new(to_guest, progress.clone())),
            memory.clone(),
            backend.clone(),
        ));
        let (queue0, device_queue) = mpsc::channel::<Vec<u8>>();
        let delivered = Arc::new(AtomicU64::new(0));
        let device = {
            let session = session.clone();
            let delivered = delivered.clone();
            let progress = progress.clone();
            StdRuntime::thread("test-device", move || {
                while let Ok(message) = device_queue.recv() {
                    session.deliver(&message);
                    progress.delivered(&message);
                    delivered.fetch_add(1, Ordering::SeqCst);
                }
            })
            .expect("a thread")
        };
        let state = Arc::new(GuestState::default());
        state.next_surface.store(1, Ordering::SeqCst);
        let handler = {
            let state = state.clone();
            let memory = memory.clone();
            let queue0 = queue0.clone();
            let progress = progress.clone();
            StdRuntime::thread("test-guest", move || {
                while let Ok(message) = from_host.recv() {
                    handle_host_message(&state, &memory, &queue0, &message);
                    progress.handled_one();
                }
            })
            .expect("a thread")
        };
        Self {
            uuid: GuestUuid([0x7c; 16]),
            state,
            memory,
            backend,
            session,
            queue0: Some(queue0),
            next_reply_id: AtomicU64::new(100),
            delivered,
            progress,
            threads: vec![device, handler],
        }
    }

    fn seen(&self) -> Vec<String> {
        std::mem::take(&mut *self.state.seen.lock())
    }

    /// The host's messages once there are `count` of them, or what came
    /// within the wait.
    fn seen_once(&self, count: usize) -> Vec<String> {
        let started = StdRuntime.now();
        while self.state.seen.lock().len() < count && StdRuntime.elapsed_since(started) < WAIT {
            std::thread::sleep(Duration::from_millis(2));
        }
        self.seen()
    }

    fn emitted(&self) -> Vec<Emitted> {
        std::mem::take(&mut *self.state.emitted.lock())
    }

    /// Sends on queue 0 without waiting.
    fn send(&self, message: Vec<u8>) {
        if let Some(queue0) = &self.queue0 {
            let _ = queue0.send(message);
        }
    }

    /// `VTParavirtualizationGuestSendMessageToHostAndCopyReplySync`: the
    /// message gets the next reply ID, flag 0x1, and the reply is waited
    /// for.
    fn call(
        &self,
        message_type: &[u8; 4],
        timeout: Duration,
        parameters: impl FnOnce(&mut MessageWriter),
    ) -> Option<Vec<u8>> {
        let reply_id = self.next_reply_id.fetch_add(1, Ordering::SeqCst);
        let mut message = MessageWriter::new(
            FourCc::new(*message_type),
            MessageFlags::GUEST_EXPECTS_REPLY,
            Some(self.uuid),
            reply_id,
        );
        parameters(&mut message);
        self.call_with(reply_id, message.finish(), timeout)
    }

    fn call_with(&self, reply_id: u64, message: Vec<u8>, timeout: Duration) -> Option<Vec<u8>> {
        let (arrived, reply) = mpsc::channel();
        self.state.waiting.lock().insert(reply_id, arrived);
        self.send(message);
        let answer = reply.recv_timeout(timeout).ok();
        self.state.waiting.lock().remove(&reply_id);
        answer
    }

    fn open(&self) -> Result<(), WireError> {
        self.open_codec(b"avc1", "reims-vgpu-video.videodecoder.avc")
    }

    /// `+dec` for `codec`, by the decoder ID the host listed for it.
    fn open_codec(&self, codec: &[u8; 4], id: &str) -> Result<(), WireError> {
        let reply = self
            .call(b"+dec", WAIT, |message| {
                message.s32(names::CODEC, i32::from_be_bytes(*codec));
                message.u32(names::VERSION, 6);
                let _ =
                    message.property_list(names::CODEC_ID, Some(&plist::Value::String(id.into())));
                let _ = message.property_list(names::GUEST_INFO, None);
            })
            .ok_or(WireError::Malformed("no +dec reply"))?;
        let reply = Message::parse(&reply)?;
        assert_eq!(reply.s32(names::ERROR)?, Some(NO_ERROR));
        assert_eq!(reply.u32(names::VERSION)?, Some(6));
        Ok(())
    }

    fn start(&self) -> Result<i32, WireError> {
        self.start_with(ImageDescription {
            codec: FourCc::new(*b"avc1"),
            width: 64,
            height: 48,
            depth: 24,
            compressor_name: Vec::new(),
            extensions: vec![(AVC_CONFIGURATION, AVCC.to_vec())],
        })
    }

    /// `dsta` with `description` as its format description.
    fn start_with(&self, description: ImageDescription) -> Result<i32, WireError> {
        let description = description.to_bytes();
        let reply = self
            .call(b"dsta", WAIT, |message| {
                message.image_description(names::FORMAT_DESCRIPTION, &description);
                message.s32(names::GUEST_POOL, 0);
            })
            .ok_or(WireError::Malformed("no dsta reply"))?;
        Ok(Message::parse(&reply)?.s32(names::ERROR)?.unwrap_or(-1))
    }

    /// `DecodeFrameWithOptions`: the sample buffer cut into fragments at
    /// the limit, every one but the last flagged 0x10 with 0x1 cleared and
    /// sent at once, the last waited for, 10 s or 1.1 s when asynchronous.
    fn decode(
        &self,
        frame_id: i64,
        flags: u32,
        sample: &SampleBuffer,
    ) -> Result<Option<(i32, u32)>, WireError> {
        let serialization = sample.serialize();
        let first_flags = MessageFlags::GUEST_EXPECTS_REPLY;
        let mut message =
            MessageWriter::new(FourCc::new(*b"dfrm"), first_flags, Some(self.uuid), 0);
        message.s64(names::FRAME, frame_id);
        message.u32(names::DECODE_FLAGS, flags);
        let mut rest =
            message.sample_buffer_with_limit(names::SAMPLE_BUFFER, FRAGMENT_LIMIT, &serialization);
        while !rest.is_empty() {
            message.change_flags(
                MessageFlags::MORE_FRAGMENTS,
                MessageFlags(MessageFlags::MORE_FRAGMENTS.0 | MessageFlags::GUEST_EXPECTS_REPLY.0),
            );
            self.send(message.finish());
            message = MessageWriter::new(FourCc::new(*b"dfrm"), first_flags, Some(self.uuid), 0);
            rest = message.sample_buffer_with_limit(names::SAMPLE_BUFFER, FRAGMENT_LIMIT, rest);
        }
        let reply_id = self.next_reply_id.fetch_add(1, Ordering::SeqCst);
        let mut last = message.finish();
        if let Some(field) = last.get_mut(32..40) {
            field.copy_from_slice(&reply_id.to_be_bytes());
        }
        let timeout = if flags & 1 != 0 {
            Duration::from_millis(1100)
        } else {
            Duration::from_secs(10)
        };
        let Some(reply) = self.call_with(reply_id, last, timeout) else {
            return Ok(None);
        };
        let reply = Message::parse(&reply)?;
        Ok(Some((
            reply.s32(names::ERROR)?.unwrap_or(-1),
            reply.u32(names::INFO_FLAGS)?.unwrap_or(u32::MAX),
        )))
    }

    /// The decoder's `Finalize`: `-dec`, sent without waiting.
    fn close(&self) {
        self.send(
            MessageWriter::new(
                FourCc::new(*b"-dec"),
                MessageFlags::NONE,
                Some(self.uuid),
                0,
            )
            .finish(),
        );
    }

    /// Waits until the device thread has delivered everything sent so far
    /// and the guest has handled what came back: a `decs` round trip, and
    /// the end of its delivery, whose sweep sends its `rels` after the
    /// reply.
    fn settle(&self) {
        let reply_id = self.next_reply_id.fetch_add(1, Ordering::SeqCst);
        let mut decs = MessageWriter::new(
            FourCc::new(*b"decs"),
            MessageFlags::GUEST_EXPECTS_REPLY,
            None,
            reply_id,
        );
        decs.u32(names::VERSION, 6);
        let _ = self.call_with(reply_id, decs.finish(), WAIT);
        let _ = self.progress.settled(reply_id, WAIT);
        self.state.seen.lock().retain(|seen| seen != "reply decs");
    }
}

impl Drop for Guest {
    fn drop(&mut self) {
        // Invalidation lets the sink go, which ends the handler thread and
        // its sender on queue 0; then queue 0 closes, and the device
        // thread ends.
        self.session.invalidate();
        self.session.complete_invalidate();
        self.queue0.take();
        for thread in self.threads.drain(..) {
            let _ = thread.join();
        }
    }
}

/// The guest's handler for the host's messages: the host session's
/// replies go to their waiters, and the decoder's requests are answered.
fn handle_host_message(
    state: &GuestState,
    memory: &GuestMemory,
    queue0: &Sender<Vec<u8>>,
    message: &[u8],
) {
    if let Some((ids, _)) = read_surface_release(message) {
        let rels = format!("rels {:?}", ids.iter().map(|id| id.0).collect::<Vec<_>>());
        state.buffers.lock().push(rels.clone());
        state.seen.lock().push(rels);
        return;
    }
    let Some(header) = header_of(message) else {
        state.seen.lock().push("unreadable".into());
        return;
    };
    let kind = header.message_type.to_string();
    if header.flags.contains(MessageFlags::REPLY_TO_GUEST) {
        state.buffers.lock().push(format!("reply {kind}"));
        state.seen.lock().push(format!("reply {kind}"));
        if let Some(waiter) = state.waiting.lock().remove(&header.reply_id) {
            let _ = waiter.send(message.to_vec());
        }
        return;
    }
    let Ok(parsed) = Message::parse(message) else {
        return;
    };
    let frame = parsed.s64(names::FRAME).ok().flatten();
    let answer = |status: i32, extra: &dyn Fn(&mut MessageWriter), surface: Option<u64>| {
        let mut reply = MessageWriter::reply_to(&header, MessageFlags::REPLY_TO_HOST);
        reply.s32(names::ERROR, status);
        extra(&mut reply);
        let bytes = reply.finish();
        let _ = queue0.send(match surface {
            Some(id) => with_surface(id, &bytes),
            None => bytes,
        });
    };
    match &header.message_type.to_be_bytes() {
        b"Dpba" => {
            state.seen.lock().push(kind);
            let attributes = parsed
                .dictionary(names::PIXEL_BUFFER_ATTRIBUTES)
                .ok()
                .flatten();
            let read = |key| {
                attributes
                    .as_ref()
                    .and_then(|attributes| attributes.get(key))
                    .and_then(plist::Value::as_unsigned_integer)
                    .unwrap_or(0)
            };
            let size = format!("Dpba {}x{}", read("Width"), read("Height"));
            state.buffers.lock().push(size);
            *state.attributes.lock() = attributes;
        }
        b"Dbuf" => {
            state
                .seen
                .lock()
                .push(format!("{kind} {}", frame.unwrap_or(-1)));
            if state.stalled.load(Ordering::SeqCst) {
                return;
            }
            let mapping_id = state.next_surface.fetch_add(1, Ordering::SeqCst);
            let mut uuid = [0; 16];
            uuid[..8].copy_from_slice(&mapping_id.to_be_bytes());
            let uuid = GuestUuid(uuid);
            let surface = nv12_surface();
            if state.short_buffers.load(Ordering::SeqCst) {
                surface.bytes.lock().pop();
            }
            let surface = SurfaceMemory {
                read_only: state.read_only_buffers.load(Ordering::SeqCst),
                lent_to_devices: !state.buffers_kept_from_devices.load(Ordering::SeqCst),
                ..surface
            };
            memory.surfaces.lock().insert(mapping_id, Arc::new(surface));
            state.lent.lock().insert(uuid, mapping_id);
            state
                .buffers
                .lock()
                .push(format!("Dbuf {} -> {mapping_id}", frame.unwrap_or(-1)));
            answer(
                NO_ERROR,
                &|reply| {
                    reply.uuid(names::PIXEL_BUFFER_UUID, uuid);
                },
                Some(mapping_id),
            );
        }
        b"Demt" => {
            state
                .seen
                .lock()
                .push(format!("{kind} {}", frame.unwrap_or(-1)));
            let uuid = parsed.uuid(names::PIXEL_BUFFER_UUID).ok().flatten();
            state.buffers.lock().push(format!(
                "Demt {} {}",
                frame.unwrap_or(-1),
                uuid.map_or("-".into(), |uuid| mapping_id_of(uuid).to_string())
            ));
            let samples = uuid.and_then(|uuid| {
                let mapping_id = state.lent.lock().remove(&uuid)?;
                let surface = memory.surfaces.lock().get(&mapping_id).cloned()?;
                let bytes = surface.bytes.lock();
                let last = (usize::try_from(HEIGHT).unwrap_or(0) - 1) * BYTES_PER_ROW
                    + usize::try_from(WIDTH).unwrap_or(0)
                    - 1;
                Some([
                    bytes[0],
                    bytes[last],
                    bytes[usize::try_from(WIDTH).unwrap_or(0)],
                    bytes[CHROMA_OFFSET],
                ])
            });
            state.emitted.lock().push(Emitted {
                frame_id: frame.unwrap_or(-1),
                status: parsed.s32(names::ERROR).ok().flatten().unwrap_or(-1),
                info: parsed
                    .u32(names::INFO_FLAGS)
                    .ok()
                    .flatten()
                    .unwrap_or(u32::MAX),
                samples,
            });
            answer(NO_ERROR, &|_| {}, None);
        }
        b"Dfub" => {
            state.seen.lock().push(kind);
            let uuids = parsed
                .uuid_array(names::PIXEL_BUFFER_UUIDS)
                .ok()
                .flatten()
                .unwrap_or_default();
            let ids: Vec<u64> = uuids.iter().copied().map(mapping_id_of).collect();
            state.buffers.lock().push(format!("Dfub {ids:?}"));
            for uuid in uuids {
                state.lent.lock().remove(&uuid);
            }
        }
        _ => state.seen.lock().push(kind),
    }
}

/// The mapping ID the stand-in guest put in a buffer's UUID.
fn mapping_id_of(uuid: GuestUuid) -> u64 {
    let mut id = [0; 8];
    id.copy_from_slice(&uuid.0[..8]);
    u64::from_be_bytes(id)
}

/// A 64x48 `420v` buffer, both planes 80 bytes a row, chroma at 4096.
fn nv12_surface() -> SurfaceMemory {
    SurfaceMemory {
        layout: SurfaceLayout {
            pixel_format: FourCc::new(*b"420v"),
            width: WIDTH,
            height: HEIGHT,
            planes: vec![
                PlaneLayout {
                    offset: 0,
                    bytes_per_row: BYTES_PER_ROW,
                    width: WIDTH,
                    height: HEIGHT,
                },
                PlaneLayout {
                    offset: CHROMA_OFFSET,
                    bytes_per_row: BYTES_PER_ROW,
                    width: WIDTH / 2,
                    height: HEIGHT / 2,
                },
            ],
            bytes_per_row: 0,
        },
        bytes: Mutex::new(vec![0; CHROMA_OFFSET + BYTES_PER_ROW * 24]),
        read_only: false,
        lent_to_devices: true,
        life: MemoryLife::new(),
        wrote: AtomicUsize::new(0),
    }
}

/// The guest kext's `subs` box for one surface, then the message.
fn with_surface(mapping_id: u64, message: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0, 0, 0, 0x30];
    bytes.extend_from_slice(b"subs");
    bytes.extend_from_slice(&mapping_id.to_be_bytes());
    bytes.extend_from_slice(&[0; 32]);
    bytes.extend_from_slice(message);
    bytes
}

fn format(avcc: &[u8]) -> SerializedFormatDescription {
    SerializedFormatDescription {
        media_type: MEDIA_TYPE_VIDEO,
        codec: FourCc::new(*b"avc1"),
        width: 64,
        height: 48,
        extensions: Some(vec![(
            "SampleDescriptionExtensionAtoms".into(),
            SerializedValue::Dictionary(vec![(
                "avcC".into(),
                SerializedValue::PropertyList(plist::Value::Data(avcc.to_vec())),
            )]),
        )]),
    }
}

/// A one-sample buffer of `data`, with the format description when
/// `format` is given, as the guest's writer adds it only when it changed.
fn sample(data: Vec<u8>, format: Option<SerializedFormatDescription>) -> SampleBuffer {
    SampleBuffer {
        sample_count: 1,
        sizes: vec![u32::try_from(data.len()).unwrap_or(0)],
        data: Some(data),
        format_description: format,
        ..SampleBuffer::default()
    }
}

fn opened() -> Result<Guest, WireError> {
    let guest = Guest::new();
    guest.open()?;
    assert_eq!(guest.start()?, NO_ERROR);
    Ok(guest)
}

/// [`opened`], with a decoder that decodes while it copies.
fn opened_decoding_ahead() -> Result<Guest, WireError> {
    let guest = Guest::new();
    guest
        .backend
        .decodes_while_copying
        .store(true, Ordering::SeqCst);
    guest.open()?;
    assert_eq!(guest.start()?, NO_ERROR);
    Ok(guest)
}

#[test]
fn starting_asks_for_buffers_in_the_backends_format_before_replying() -> Result<(), WireError> {
    let guest = opened()?;
    assert_eq!(guest.seen(), ["reply +dec", "Dpba", "reply dsta"]);
    let attributes = guest.state.attributes.lock().clone().unwrap_or_default();
    assert_eq!(
        attributes.get("PixelFormatType"),
        Some(&plist::Value::Array(vec![plist::Value::Integer(
            i64::from(u32::from_be_bytes(*b"420v")).into()
        )]))
    );
    assert_eq!(attributes.get("Width").and_then(as_i64), Some(64));
    assert_eq!(attributes.get("Height").and_then(as_i64), Some(48));
    assert_eq!(
        attributes.get("IOSurfaceProperties"),
        Some(&plist::Value::Dictionary(plist::Dictionary::new()))
    );
    // AVD's alignment and cache-mode keys.
    assert_eq!(
        attributes.get("BytesPerRowAlignment").and_then(as_i64),
        Some(64)
    );
    assert_eq!(attributes.get("PlaneAlignment").and_then(as_i64), Some(256));
    assert_eq!(
        attributes.get("PreferRealTimeCacheModeIfEveryoneDoes"),
        Some(&plist::Value::Boolean(true))
    );
    assert_eq!(
        guest
            .backend
            .opened
            .lock()
            .first()
            .map(|config| config.configuration.clone()),
        Some(AVCC.to_vec())
    );
    Ok(())
}

#[test]
fn a_synchronous_frame_is_emitted_and_released_before_its_reply() -> Result<(), WireError> {
    let guest = opened()?;
    guest.seen();
    let reply = guest.decode(1, 0, &sample(vec![0x40, 0x90], Some(format(AVCC))))?;
    assert_eq!(reply, Some((NO_ERROR, 0)));
    assert_eq!(guest.seen(), ["Dbuf 1", "Demt 1", "rels [1]", "reply dfrm"]);
    let emitted = guest.emitted();
    assert_eq!(emitted.len(), 1);
    assert_eq!(
        (emitted[0].frame_id, emitted[0].status, emitted[0].info),
        (1, 0, 0)
    );
    // Luma at both corners, the row's padding untouched, chroma.
    assert_eq!(emitted[0].samples, Some([0x40, 0x40, 0, 0x90]));
    assert_eq!(guest.memory.held(), 0);
    assert_eq!(
        *guest.backend.submitted.lock(),
        [Submitted {
            frame_id: 1,
            length: 2,
            reconfigured: None,
        }]
    );
    Ok(())
}

#[test]
fn an_asynchronous_frame_is_answered_once_its_buffer_is_lent() -> Result<(), WireError> {
    let guest = opened()?;
    guest.seen();
    let reply = guest.decode(2, 1, &sample(vec![0x50], Some(format(AVCC))))?;
    assert_eq!(reply, Some((NO_ERROR, 1)));
    assert_eq!(
        guest.seen_once(4),
        ["Dbuf 2", "reply dfrm", "Demt 2", "rels [1]"]
    );
    let emitted = guest.emitted();
    assert_eq!((emitted[0].status, emitted[0].info), (0, 1));
    Ok(())
}

/// The `dfrm` reply's info flags are those Apple's decoder answers its
/// decode call with: kVTDecodeInfo_Asynchronous for a frame sent with the
/// asynchronous flag, whether it decodes, has no picture or fails, and
/// none for one sent without; none for a RASL picture dropped, either way,
/// and none from a decoder that never started. Measured with AVD's H.264,
/// HEVC, VP9 and AV1 decoders from 64x64 to 1920x1080 (macOS 27.0.1,
/// 26A434, 2026-10-24).
#[test]
fn the_frame_reply_says_asynchronous_where_apples_decoder_does() -> Result<(), WireError> {
    let guest = opened()?;
    let cases: [(&str, Vec<u8>, [u32; 2]); 4] = [
        ("decoded", vec![0x10, 0x90], [0, 1]),
        ("no picture", Vec::new(), [0, 1]),
        ("failed", vec![0xff], [0, 1]),
        ("a RASL picture dropped", vec![0xfd], [0, 0]),
    ];
    let mut frame_id = 0;
    for (what, data, replies) in cases {
        for (flags, info) in [0, 1].into_iter().zip(replies) {
            frame_id += 1;
            let format = (frame_id == 1).then(|| format(AVCC));
            assert_eq!(
                guest.decode(frame_id, flags, &sample(data.clone(), format))?,
                Some((NO_ERROR, info)),
                "{what}, decode flags {flags:#x}"
            );
        }
    }
    let unstarted = Guest::new();
    unstarted.open()?;
    assert_eq!(
        unstarted.decode(1, 1, &sample(vec![0x10], Some(format(AVCC))))?,
        Some((Status::DecoderNotStarted.code(), 0))
    );
    Ok(())
}

/// A whole one-fragment `dfrm` for `frame_id`, sent without waiting for its
/// reply: what an x86_64 guest's synchronous decode does, since its wait
/// for the reply ends at once.
fn send_frame(guest: &Guest, frame_id: i64, flags: u32, data: Vec<u8>) {
    let reply_id = guest.next_reply_id.fetch_add(1, Ordering::SeqCst);
    let mut message = MessageWriter::new(
        FourCc::new(*b"dfrm"),
        MessageFlags::GUEST_EXPECTS_REPLY,
        Some(guest.uuid),
        reply_id,
    );
    message.s64(names::FRAME, frame_id);
    message.u32(names::DECODE_FLAGS, flags);
    let serialization = sample(data, None).serialize();
    message.sample_buffer_with_limit(names::SAMPLE_BUFFER, FRAGMENT_LIMIT, &serialization);
    guest.send(message.finish());
}

/// Waits until the backend has been handed `frame_id`.
fn wait_for_submitted(guest: &Guest, frame_id: i64) -> bool {
    let started = StdRuntime.now();
    while StdRuntime.elapsed_since(started) < WAIT {
        if guest
            .backend
            .submitted
            .lock()
            .iter()
            .any(|submitted| submitted.frame_id == frame_id)
        {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    false
}

/// A frame the guest has stopped waiting for, because its next `dfrm` is
/// already queued, is decoded but asks for no buffer and emits nothing; the
/// next one goes through as before. Checked for a synchronous decode, the
/// wallpaper's, and an asynchronous one.
#[test]
fn a_frame_the_guest_gave_up_on_is_decoded_but_not_emitted() -> Result<(), WireError> {
    for flags in [0, 1] {
        let guest = opened()?;
        guest.seen();
        // The first frame carries the format description.
        assert_eq!(
            guest.decode(1, flags, &sample(vec![0x10], Some(format(AVCC))))?,
            Some((NO_ERROR, flags))
        );
        assert_eq!(guest.seen_once(4).len(), 4, "decode flags {flags:#x}");
        guest.emitted();
        // Frame 2 is in the backend's hands when frame 3's dfrm comes.
        guest.backend.gate.close();
        send_frame(&guest, 2, flags, vec![0x20, 0x90]);
        assert!(wait_for_submitted(&guest, 2));
        send_frame(&guest, 3, flags, vec![0x30, 0x90]);
        // The decs round trip is answered on the device's thread after frame
        // 3's delivery, so frame 3 is queued once it returns.
        guest.settle();
        guest.backend.gate.open();
        let seen = guest.seen_once(5);
        assert_eq!(
            seen,
            if flags == 0 {
                ["reply dfrm", "Dbuf 3", "Demt 3", "rels [2]", "reply dfrm"]
            } else {
                ["reply dfrm", "Dbuf 3", "reply dfrm", "Demt 3", "rels [2]"]
            },
            "decode flags {flags:#x}"
        );
        let emitted = guest.emitted();
        assert_eq!(
            emitted
                .iter()
                .map(|frame| frame.frame_id)
                .collect::<Vec<_>>(),
            [3],
            "decode flags {flags:#x}"
        );
        // Both were decoded, so frame 3 had its references.
        let submitted: Vec<i64> = guest
            .backend
            .submitted
            .lock()
            .iter()
            .map(|frame| frame.frame_id)
            .collect();
        assert_eq!(submitted, [1, 2, 3], "decode flags {flags:#x}");
        assert_eq!(guest.memory.held(), 0);
    }
    Ok(())
}

/// A frame whose `Dbuf` the guest leaves unanswered, and that the guest
/// gives up on meanwhile, is not emitted: the host waits out the `Dbuf`'s
/// 1.1 s and not a `Demt`'s 10 s after it (where the wallpaper's guest
/// stopped answering `Dbuf` at its seventh frame).
#[test]
fn a_frame_given_up_on_while_its_buffer_is_awaited_is_not_emitted() -> Result<(), WireError> {
    let guest = opened()?;
    guest.seen();
    assert_eq!(
        guest.decode(1, 0, &sample(vec![0x10], Some(format(AVCC))))?,
        Some((NO_ERROR, 0))
    );
    assert_eq!(guest.seen_once(4).len(), 4);
    guest.emitted();
    guest.state.stalled.store(true, Ordering::SeqCst);
    send_frame(&guest, 2, 0, vec![0x20, 0x90]);
    let started = StdRuntime.now();
    while !guest.state.seen.lock().iter().any(|seen| seen == "Dbuf 2") {
        assert!(StdRuntime.elapsed_since(started) < WAIT, "no Dbuf");
        std::thread::sleep(Duration::from_millis(2));
    }
    // Frame 3 comes while frame 2's Dbuf waits.
    send_frame(&guest, 3, 0, vec![0x30, 0x90]);
    // Frame 2's Dbuf, then frame 3's, and the Demt that is not given up on.
    let seen = guest.seen_once(5);
    let waited = StdRuntime.elapsed_since(started);
    assert_eq!(
        seen,
        ["Dbuf 2", "reply dfrm", "Dbuf 3", "Demt 3", "reply dfrm"]
    );
    // Two Dbufs' 1.1 s, and no Demt's 10 s.
    assert!(waited < Duration::from_secs(5), "waited {waited:?}");
    let emitted = guest.emitted();
    assert_eq!(
        emitted
            .iter()
            .map(|frame| (frame.frame_id, frame.status))
            .collect::<Vec<_>>(),
        [(3, Status::ReplyTimedOut.code())]
    );
    Ok(())
}

/// Waits until the guest has seen `what` from the host, leaving it seen.
fn wait_for_seen(guest: &Guest, what: &str) -> bool {
    let started = StdRuntime.now();
    while StdRuntime.elapsed_since(started) < WAIT {
        if guest.state.seen.lock().iter().any(|seen| seen == what) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    false
}

/// Waits until the guest's record of buffers has `what`: what the
/// session's thread sends after a reply, which [`Guest::settle`], answered
/// on the device's thread, does not wait for.
fn wait_for_buffers(guest: &Guest, what: &str) -> bool {
    let started = StdRuntime.now();
    while StdRuntime.elapsed_since(started) < WAIT {
        if guest.state.buffers.lock().iter().any(|line| line == what) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    false
}

/// A `dcop` of `RealTime`, sent without waiting for its reply.
fn send_copy_property(guest: &Guest) {
    let reply_id = guest.next_reply_id.fetch_add(1, Ordering::SeqCst);
    let mut message = MessageWriter::new(
        FourCc::new(*b"dcop"),
        MessageFlags::GUEST_EXPECTS_REPLY,
        Some(guest.uuid),
        reply_id,
    );
    let _ = message.property_list(
        names::PROPERTY_KEY,
        Some(&plist::Value::String("RealTime".into())),
    );
    guest.send(message.finish());
}

/// An asynchronous frame of a decoder that decodes while it copies is
/// copied on a thread of its own, and the next `dfrm` is decoded once the
/// frame is whole, before its copy is over; the guest sees what it saw with
/// one frame at a time.
#[test]
fn an_asynchronous_frame_is_copied_while_the_next_one_decodes() -> Result<(), WireError> {
    let guest = opened_decoding_ahead()?;
    guest.seen();
    assert_eq!(
        guest.decode(1, 1, &sample(vec![0x10], Some(format(AVCC))))?,
        Some((NO_ERROR, 1))
    );
    assert_eq!(guest.seen_once(4).len(), 4);
    let work = guest.backend.frames.clone();
    work.wait.close();
    work.copy.close();
    // Frame 2's reply comes once its buffer is lent, before its copy.
    assert_eq!(
        guest.decode(2, 1, &sample(vec![0x20, 0x90], None))?,
        Some((NO_ERROR, 1))
    );
    send_frame(&guest, 3, 1, vec![0x30, 0x91]);
    guest.settle();
    // Frame 3 waits while frame 2 is not whole.
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(work.at("decode 3"), usize::MAX, "{:?}", work.events());
    work.wait.open();
    assert!(wait_for_submitted(&guest, 3), "{:?}", work.events());
    assert_eq!(work.at("copied 2"), usize::MAX, "{:?}", work.events());
    work.copy.open();
    assert_eq!(
        guest.seen_once(8),
        [
            "Dbuf 2",
            "reply dfrm",
            "Demt 2",
            "rels [2]",
            "Dbuf 3",
            "reply dfrm",
            "Demt 3",
            "rels [3]"
        ]
    );
    assert!(work.at("waited 2") < work.at("decode 3"));
    assert!(work.at("decode 3") < work.at("copied 2"));
    let emitted = guest.emitted();
    assert_eq!(
        emitted
            .iter()
            .map(|frame| (frame.frame_id, frame.status, frame.info, frame.samples))
            .collect::<Vec<_>>(),
        [
            (1, 0, 1, Some([0x10, 0x10, 0, 128])),
            (2, 0, 1, Some([0x20, 0x20, 0, 0x90])),
            (3, 0, 1, Some([0x30, 0x30, 0, 0x91]))
        ]
    );
    guest.settle();
    assert_eq!(guest.memory.held(), 0);
    Ok(())
}

/// A synchronous frame is copied on the session's thread, and a `dfrm`
/// behind it waits for it whatever the decoder.
#[test]
fn a_synchronous_frame_is_copied_before_the_next_one_decodes() -> Result<(), WireError> {
    let guest = opened_decoding_ahead()?;
    guest.seen();
    assert_eq!(
        guest.decode(1, 0, &sample(vec![0x10], Some(format(AVCC))))?,
        Some((NO_ERROR, 0))
    );
    assert_eq!(guest.seen_once(4).len(), 4);
    let work = guest.backend.frames.clone();
    work.copy.close();
    send_frame(&guest, 2, 0, vec![0x20, 0x90]);
    assert!(wait_for_seen(&guest, "Dbuf 2"));
    send_frame(&guest, 3, 0, vec![0x30, 0x91]);
    guest.settle();
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(work.at("decode 3"), usize::MAX, "{:?}", work.events());
    work.copy.open();
    assert_eq!(
        guest.seen_once(8),
        [
            "Dbuf 2",
            "Demt 2",
            "rels [2]",
            "reply dfrm",
            "Dbuf 3",
            "Demt 3",
            "rels [3]",
            "reply dfrm"
        ]
    );
    assert!(work.at("copied 2") < work.at("decode 3"));
    assert_eq!(work.at("waited 2"), usize::MAX, "copied without the wait");
    Ok(())
}

/// A message that comes before the next `dfrm` while a frame is copied
/// keeps the order: it is answered after the frame's emission, and the
/// `dfrm` behind it is decoded in its turn, not ahead.
#[test]
fn a_message_before_the_next_frame_keeps_its_turn() -> Result<(), WireError> {
    let guest = opened_decoding_ahead()?;
    guest.seen();
    assert_eq!(
        guest.decode(1, 1, &sample(vec![0x10], Some(format(AVCC))))?,
        Some((NO_ERROR, 1))
    );
    assert_eq!(guest.seen_once(4).len(), 4);
    let work = guest.backend.frames.clone();
    work.copy.close();
    assert_eq!(
        guest.decode(2, 1, &sample(vec![0x20, 0x90], None))?,
        Some((NO_ERROR, 1))
    );
    send_copy_property(&guest);
    send_frame(&guest, 3, 1, vec![0x30, 0x91]);
    guest.settle();
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(work.at("decode 3"), usize::MAX, "{:?}", work.events());
    work.copy.open();
    assert_eq!(
        guest.seen_once(9),
        [
            "Dbuf 2",
            "reply dfrm",
            "Demt 2",
            "rels [2]",
            "reply dcop",
            "Dbuf 3",
            "reply dfrm",
            "Demt 3",
            "rels [3]"
        ]
    );
    assert!(work.at("copied 2") < work.at("decode 3"));
    Ok(())
}

/// Closing a decoder while a frame is copied and the next is decoded ahead
/// ends the session once the copy is over: the frame copied is not
/// emitted, the one decoded ahead asks for no buffer, and nothing stays
/// held.
#[test]
fn closing_a_decoder_while_it_decodes_ahead_ends_it() -> Result<(), WireError> {
    let guest = opened_decoding_ahead()?;
    guest.seen();
    assert_eq!(
        guest.decode(1, 1, &sample(vec![0x10], Some(format(AVCC))))?,
        Some((NO_ERROR, 1))
    );
    assert_eq!(guest.seen_once(4).len(), 4);
    let work = guest.backend.frames.clone();
    work.copy.close();
    assert_eq!(
        guest.decode(2, 1, &sample(vec![0x20, 0x90], None))?,
        Some((NO_ERROR, 1))
    );
    send_frame(&guest, 3, 1, vec![0x30, 0x91]);
    assert!(wait_for_submitted(&guest, 3), "{:?}", work.events());
    guest.close();
    std::thread::sleep(Duration::from_millis(50));
    work.copy.open();
    guest.settle();
    // The host session's sweep still lets frame 2's buffer go.
    assert_eq!(guest.seen(), ["Dbuf 2", "reply dfrm", "rels [2]"]);
    assert_eq!(guest.memory.held(), 0);
    assert_eq!(
        guest.call(b"dfin", Duration::from_millis(200), |_| {}),
        None
    );
    Ok(())
}

/// What a run of frames gave: the host's messages as the guest saw them,
/// each emission's frame ID, status, info flags and samples, and the
/// backend's events.
struct RunOfFrames {
    seen: Vec<String>,
    emitted: Vec<(i64, i32, u32, Option<[u8; 4]>)>,
    events: Vec<String>,
}

/// A run of frames with a decoder that decodes while it copies or not:
/// every path a frame takes, each copy long enough for the next `dfrm` to
/// come during it.
fn run_of_frames(decodes_while_copying: bool) -> Result<RunOfFrames, WireError> {
    run_of_frames_written(decodes_while_copying, false)
}

/// [`run_of_frames`], the frames written by the backend's device where
/// `device_writes`, else copied by the CPU.
fn run_of_frames_written(
    decodes_while_copying: bool,
    device_writes: bool,
) -> Result<RunOfFrames, WireError> {
    let guest = Guest::new();
    guest
        .backend
        .decodes_while_copying
        .store(decodes_while_copying, Ordering::SeqCst);
    guest
        .backend
        .frames
        .device_writes
        .store(device_writes, Ordering::SeqCst);
    guest.open()?;
    assert_eq!(guest.start()?, NO_ERROR);
    guest.seen();
    *guest.backend.frames.copy_time.lock() = Duration::from_millis(20);
    let frames: [(u32, Vec<u8>); 9] = [
        (1, vec![0x10, 0x80]),
        (1, vec![0x20, 0x81]),
        // No picture, then a frame that does not decode.
        (1, Vec::new()),
        (1, vec![0xff]),
        (1, vec![0x30, 0x82]),
        // Not to be output.
        (1 | 2, vec![0x40, 0x83]),
        (1, vec![0x50, 0x84]),
        // Synchronous, then asynchronous again.
        (0, vec![0x60, 0x85]),
        (1, vec![0x70, 0x86]),
    ];
    for (frame_id, (flags, data)) in (1..).zip(frames) {
        let format = (frame_id == 1).then(|| format(AVCC));
        assert_eq!(
            guest.decode(frame_id, flags, &sample(data, format))?,
            Some((NO_ERROR, flags & 1)),
            "frame {frame_id}"
        );
    }
    // dfin is answered after every frame is done.
    guest
        .call(b"dfin", WAIT, |_| {})
        .ok_or(WireError::Malformed("no dfin reply"))?;
    let seen = guest.seen();
    let emitted = guest
        .emitted()
        .iter()
        .map(|frame| (frame.frame_id, frame.status, frame.info, frame.samples))
        .collect();
    guest.settle();
    assert_eq!(guest.memory.held(), 0);
    Ok(RunOfFrames {
        seen,
        emitted,
        events: guest.backend.frames.events(),
    })
}

/// Whether the backend decoded a frame while the one before it was still
/// being copied.
fn decoded_during_a_copy(events: &[String]) -> bool {
    let at = |event: String| events.iter().position(|seen| *seen == event);
    (1..9).any(|frame| {
        let copied = at(format!("copied {frame}")).or_else(|| at(format!("written {frame}")));
        match (at(format!("decode {}", frame + 1)), copied) {
            (Some(decoded), Some(copied)) => decoded < copied,
            _ => false,
        }
    })
}

/// Decoding ahead leaves every message, reply and emission the guest sees
/// as it was, and their order: the run of frames gives the same with and
/// without it, and decodes ahead only with it.
#[test]
fn decoding_ahead_leaves_what_the_guest_sees_as_it_was() -> Result<(), WireError> {
    let one_at_a_time = run_of_frames(false)?;
    assert!(
        !decoded_during_a_copy(&one_at_a_time.events),
        "{:?}",
        one_at_a_time.events
    );
    let ahead = run_of_frames(true)?;
    assert!(decoded_during_a_copy(&ahead.events), "{:?}", ahead.events);
    assert_eq!(ahead.seen, one_at_a_time.seen);
    assert_eq!(ahead.emitted, one_at_a_time.emitted);
    assert_eq!(one_at_a_time.emitted.len(), 9);
    Ok(())
}

#[test]
fn a_sample_larger_than_a_message_arrives_whole_from_its_fragments() -> Result<(), WireError> {
    let guest = opened()?;
    let mut data = vec![0x30; 2 * FRAGMENT_LIMIT + 1000];
    data[1] = 0x70;
    let reply = guest.decode(3, 0, &sample(data, Some(format(AVCC))))?;
    assert_eq!(reply, Some((NO_ERROR, 0)));
    assert_eq!(
        guest
            .backend
            .submitted
            .lock()
            .last()
            .map(|frame| frame.length),
        Some(2 * FRAGMENT_LIMIT + 1000)
    );
    assert_eq!(guest.emitted()[0].samples, Some([0x30, 0x30, 0, 0x70]));
    Ok(())
}

/// A sample with no picture is emitted dropped, with no buffer. A frame
/// sent with kVTDecodeFrame_DoNotOutputFrame is emitted as any other, with
/// its image, as Apple's AVD emits it (measured on macOS 27.0.1, 26A434,
/// 2026-10-24).
#[test]
fn a_frame_without_a_picture_is_dropped_and_one_not_to_be_output_is_not() -> Result<(), WireError> {
    let guest = opened()?;
    guest.seen();
    assert_eq!(
        guest.decode(4, 0, &sample(Vec::new(), Some(format(AVCC))))?,
        Some((NO_ERROR, 0))
    );
    assert_eq!(
        guest.decode(5, 2, &sample(vec![0x50, 0x90], None))?,
        Some((NO_ERROR, 0))
    );
    assert_eq!(
        guest.seen(),
        [
            "Demt 4",
            "reply dfrm",
            "Dbuf 5",
            "Demt 5",
            "rels [1]",
            "reply dfrm"
        ]
    );
    let emitted = guest.emitted();
    assert_eq!(
        emitted
            .iter()
            .map(|frame| (frame.frame_id, frame.status, frame.info, frame.samples))
            .collect::<Vec<_>>(),
        [(4, 0, 2, None), (5, 0, 0, Some([0x50, 0x50, 0, 0x90]))]
    );
    Ok(())
}

/// A buffer the guest made read-only for its clients is never written: its
/// frame is emitted with kVTVideoDecoderMalfunctionErr and no buffer, and
/// the buffer goes back to the guest as it was. Apple's host hands such a
/// buffer to its hardware.
#[test]
fn a_frame_for_a_read_only_buffer_is_a_malfunction_and_leaves_it() -> Result<(), WireError> {
    let guest = opened()?;
    guest.seen();
    guest.state.read_only_buffers.store(true, Ordering::SeqCst);
    let reply = guest.decode(1, 0, &sample(vec![0x40, 0x90], Some(format(AVCC))))?;
    assert_eq!(reply, Some((NO_ERROR, 0)));
    assert_eq!(
        guest.seen(),
        ["Dbuf 1", "Demt 1", "Dfub", "rels [1]", "reply dfrm"],
        "the unwritten buffer goes back to the guest's pool"
    );
    let emitted = guest.emitted();
    assert_eq!(
        (emitted[0].frame_id, emitted[0].status),
        (1, Status::VideoDecoderMalfunction.code())
    );
    assert_eq!(emitted[0].samples, None, "emitted with no buffer");
    let surfaces = guest.memory.surfaces.lock();
    assert_eq!(surfaces.len(), 1);
    assert!(
        surfaces
            .values()
            .all(|s| s.bytes.lock().iter().all(|&b| b == 0)),
        "the buffer as it was"
    );
    drop(surfaces);
    assert_eq!(guest.memory.held(), 0);
    Ok(())
}

#[test]
fn a_frame_that_does_not_decode_is_emitted_with_the_backends_status() -> Result<(), WireError> {
    let guest = opened()?;
    guest.seen();
    assert_eq!(
        guest.decode(6, 0, &sample(vec![0xff], Some(format(AVCC))))?,
        Some((NO_ERROR, 0))
    );
    // Apple's AVD asks a buffer for a frame that fails, emits the frame
    // without it, and gives it back.
    assert_eq!(
        guest.seen(),
        ["Dbuf 6", "Demt 6", "Dfub", "rels [1]", "reply dfrm"]
    );
    let emitted = guest.emitted();
    assert_eq!(
        (emitted[0].status, emitted[0].samples),
        (Status::VideoDecoderBadData.code(), None)
    );
    Ok(())
}

#[test]
fn a_new_format_description_reaches_the_backend_once() -> Result<(), WireError> {
    let guest = opened()?;
    guest.decode(7, 0, &sample(vec![1], Some(format(AVCC))))?;
    guest.decode(8, 0, &sample(vec![1], Some(format(OTHER_AVCC))))?;
    guest.decode(9, 0, &sample(vec![1], None))?;
    assert_eq!(
        guest
            .backend
            .submitted
            .lock()
            .iter()
            .map(|frame| frame.reconfigured.clone())
            .collect::<Vec<_>>(),
        [None, Some(OTHER_AVCC.to_vec()), None]
    );
    Ok(())
}

/// Measured on macOS 27.0.1 (26A434): AVD's H.264, HEVC, VP9 and AV1
/// decoders answer a `dfrm` before any `dsta` with 311 and no info flags,
/// either decode flag, and emit nothing (2026-10-24).
#[test]
fn decoding_before_starting_is_refused_with_nothing_emitted() -> Result<(), WireError> {
    let guest = Guest::new();
    guest.open()?;
    guest.seen();
    for (frame_id, flags) in [(1, 0), (2, 1)] {
        assert_eq!(
            guest.decode(frame_id, flags, &sample(vec![1], Some(format(AVCC))))?,
            Some((Status::DecoderNotStarted.code(), 0)),
            "decode flags {flags:#x}"
        );
    }
    assert_eq!(guest.seen(), ["reply dfrm", "reply dfrm"]);
    assert!(guest.backend.submitted.lock().is_empty());
    Ok(())
}

/// Measured on Apple's host, macOS 27.0.1 (26A434), with interlaced H.264
/// and H.264, HEVC, VP9 and AV1 under 64 a side: after a `dsta` AVD
/// refused, each `dfrm` answers 311 with no info flags and emits nothing,
/// either decode flag; a second `dsta`, of a stream AVD decodes, answers
/// -12909 with no `Dpba`, and the frames after it answer 311 still
/// (2026-10-24).
#[test]
fn after_a_refused_start_frames_and_starts_are_refused_as_apples_are() -> Result<(), WireError> {
    let guest = Guest::new();
    guest.open()?;
    let interlaced = avcc_with(&main_sps(false), 4);
    assert_eq!(
        guest.start_with(avc_description(interlaced, 64, 64))?,
        Status::VideoDecoderMalfunction.code()
    );
    assert_eq!(guest.seen(), ["reply +dec", "reply dsta"]);
    let refused = (Status::DecoderNotStarted.code(), 0);
    for (frame_id, flags) in [(1, 0), (2, 1)] {
        assert_eq!(
            guest.decode(frame_id, flags, &sample(vec![1], Some(format(AVCC))))?,
            Some(refused),
            "decode flags {flags:#x}"
        );
    }
    let progressive = avcc_with(&main_sps(true), 4);
    assert_eq!(
        guest.start_with(avc_description(progressive, 64, 64))?,
        Status::VideoDecoderBadData.code()
    );
    for (frame_id, flags) in [(3, 0), (4, 1)] {
        assert_eq!(
            guest.decode(frame_id, flags, &sample(vec![1], None))?,
            Some(refused),
            "decode flags {flags:#x}"
        );
    }
    assert_eq!(
        guest.seen(),
        [
            "reply dfrm",
            "reply dfrm",
            "reply dsta",
            "reply dfrm",
            "reply dfrm"
        ]
    );
    assert!(guest.backend.opened.lock().is_empty());
    assert!(guest.backend.submitted.lock().is_empty());
    Ok(())
}

/// Measured on macOS 27.0.1 (26A434): after a `dsta` AVD took, a second
/// answers -12909 with no `Dpba`, of the same stream or of one AVD
/// refuses, and the session decodes on as the first started it
/// (2026-10-24).
#[test]
fn a_second_start_is_refused_and_the_first_session_decodes_on() -> Result<(), WireError> {
    let guest = opened()?;
    guest.decode(1, 0, &sample(vec![1], Some(format(AVCC))))?;
    guest.seen();
    for record in [AVCC.to_vec(), avcc_with(&main_sps(false), 4)] {
        assert_eq!(
            guest.start_with(avc_description(record, 64, 48))?,
            Status::VideoDecoderBadData.code()
        );
        assert_eq!(guest.seen(), ["reply dsta"]);
    }
    assert_eq!(guest.backend.opened.lock().len(), 1);
    assert_eq!(
        guest.decode(2, 0, &sample(vec![1], None))?,
        Some((NO_ERROR, 0))
    );
    assert_eq!(
        guest.emitted().last().map(|emitted| emitted.status),
        Some(NO_ERROR)
    );
    assert_eq!(guest.backend.submitted.lock().len(), 2);
    Ok(())
}

/// Measured on macOS 27.0.1 (26A434): AVD's H.264, HEVC, VP9 and AV1
/// decoders answer `dfin` with -12782, started or not, and `dinv` with 0
/// (2026-10-24).
#[test]
fn finishing_and_invalidating_reply_as_apples_decoders_do() -> Result<(), WireError> {
    let unstarted = Guest::new();
    unstarted.open()?;
    let guest = opened()?;
    for (guest, message_type, status) in [
        (&unstarted, b"dfin", Status::UnimplementedFunction.code()),
        (&guest, b"dfin", Status::UnimplementedFunction.code()),
        (&guest, b"dinv", NO_ERROR),
        (&guest, b"dfin", Status::UnimplementedFunction.code()),
    ] {
        let reply = guest
            .call(message_type, WAIT, |_| {})
            .ok_or(WireError::Malformed("no reply"))?;
        assert_eq!(Message::parse(&reply)?.s32(names::ERROR)?, Some(status));
    }
    // An invalidated decoder decodes nothing, and answers as Apple's AVD
    // decoders answer after `dinv` (macOS 27.0.1, 2026-10-24).
    assert_eq!(
        guest.decode(1, 0, &sample(vec![1], Some(format(AVCC))))?,
        Some((Status::DecoderNotStarted.code(), 0))
    );
    Ok(())
}

#[test]
fn closing_a_decoder_waiting_for_a_buffer_ends_its_wait_at_once() -> Result<(), WireError> {
    let guest = opened()?;
    guest.state.stalled.store(true, Ordering::SeqCst);
    // The frame waits for a Dbuf reply that never comes.
    guest.send({
        let mut message = MessageWriter::new(
            FourCc::new(*b"dfrm"),
            MessageFlags::GUEST_EXPECTS_REPLY,
            Some(guest.uuid),
            900,
        );
        message.s64(names::FRAME, 1);
        let serialization = sample(vec![1], Some(format(AVCC))).serialize();
        message.sample_buffer_with_limit(names::SAMPLE_BUFFER, FRAGMENT_LIMIT, &serialization);
        message.finish()
    });
    let started = StdRuntime.now();
    while !guest.seen().iter().any(|seen| seen == "Dbuf 1") {
        assert!(StdRuntime.elapsed_since(started) < WAIT, "no Dbuf");
        std::thread::sleep(Duration::from_millis(5));
    }
    let closing = StdRuntime.now();
    guest.close();
    guest.settle();
    // Apple's would wait out the Dbuf's 1.1 s here.
    assert!(StdRuntime.elapsed_since(closing) < Duration::from_millis(1000));
    // The session is gone, and nothing more came from it.
    let reply = guest
        .call(b"dfin", Duration::from_millis(200), |_| {})
        .map(|reply| reply.len());
    assert_eq!(reply, None);
    assert!(guest.seen().iter().all(|seen| !seen.starts_with("Demt")));
    Ok(())
}

#[test]
fn a_frame_the_guest_answers_late_is_emitted_with_the_timeout() -> Result<(), WireError> {
    let guest = opened()?;
    guest.state.stalled.store(true, Ordering::SeqCst);
    guest.seen();
    let started = StdRuntime.now();
    let reply = guest.decode(1, 0, &sample(vec![1], Some(format(AVCC))))?;
    let waited = StdRuntime.elapsed_since(started);
    assert_eq!(reply, Some((NO_ERROR, 0)));
    // The Dbuf's 1.1 s.
    assert!(waited >= Duration::from_millis(1000) && waited < Duration::from_secs(5));
    assert_eq!(guest.seen(), ["Dbuf 1", "Demt 1", "reply dfrm"]);
    assert_eq!(guest.emitted()[0].status, Status::ReplyTimedOut.code());
    Ok(())
}

#[test]
fn the_decoder_answers_for_its_properties() -> Result<(), WireError> {
    let guest = opened()?;
    let string = |key: &str| plist::Value::String(key.into());
    let reply = guest.call(b"dsup", WAIT, |_| {}).unwrap_or_default();
    let supported = Message::parse(&reply)?
        .dictionary(names::SUPPORTED_PROPERTIES)?
        .unwrap_or_default();
    assert!(supported.contains_key("UsingHardwareAcceleratedVideoDecoder"));
    assert!(supported.contains_key("RealTime"));

    let copy = |key: &str| -> Result<(Option<i32>, Option<plist::Value>), WireError> {
        let reply = guest
            .call(b"dcop", WAIT, |message| {
                let _ = message.property_list(names::PROPERTY_KEY, Some(&string(key)));
            })
            .unwrap_or_default();
        let reply = Message::parse(&reply)?;
        Ok((
            reply.s32(names::ERROR)?,
            reply.property_list(names::PROPERTY_VALUE)?,
        ))
    };
    assert_eq!(
        copy("UsingHardwareAcceleratedVideoDecoder")?,
        (Some(NO_ERROR), Some(plist::Value::Boolean(true)))
    );
    assert_eq!(
        copy("NoSuchProperty")?,
        (Some(Status::PropertyNotSupported.code()), None)
    );

    let set = |key: &str, timeout: Duration| -> Option<OsStatus> {
        let reply = guest.call(b"dset", timeout, |message| {
            let _ = message.property_list(names::PROPERTY_KEY, Some(&string(key)));
            let _ =
                message.property_list(names::PROPERTY_VALUE, Some(&plist::Value::Boolean(true)));
        })?;
        Message::parse(&reply).ok()?.s32(names::ERROR).ok()?
    };
    assert_eq!(set("RealTime", WAIT), Some(NO_ERROR));
    assert_eq!(copy("RealTime")?.1, Some(plist::Value::Boolean(true)));
    assert_eq!(
        set("NoSuchProperty", WAIT),
        Some(Status::PropertyNotSupported.code())
    );
    // The host's own property gets no answer at all.
    assert_eq!(set("Paravirtualized", Duration::from_millis(200)), None);

    let reply = guest
        .call(b"dmul", WAIT, |message| {
            let mut properties = plist::Dictionary::new();
            properties.insert("RealTime".into(), plist::Value::Boolean(false));
            let _ = message.property_list(
                names::PROPERTIES,
                Some(&plist::Value::Dictionary(properties)),
            );
        })
        .unwrap_or_default();
    assert_eq!(
        Message::parse(&reply)?.s32(names::ERROR)?,
        Some(Status::UnimplementedFunction.code())
    );
    Ok(())
}

#[test]
fn an_av1_decoder_applies_film_grain_where_it_decodes() -> Result<(), WireError> {
    let string = |key: &str| plist::Value::String(key.into());
    let copy = |guest: &Guest| -> Result<(Option<i32>, Option<plist::Value>), WireError> {
        let reply = guest
            .call(b"dcop", WAIT, |message| {
                let _ =
                    message.property_list(names::PROPERTY_KEY, Some(&string("AV1FilmGrainMode")));
            })
            .unwrap_or_default();
        let reply = Message::parse(&reply)?;
        Ok((
            reply.s32(names::ERROR)?,
            reply.property_list(names::PROPERTY_VALUE)?,
        ))
    };
    let set = |guest: &Guest, mode: &str| -> Option<OsStatus> {
        let reply = guest.call(b"dset", WAIT, |message| {
            let _ = message.property_list(names::PROPERTY_KEY, Some(&string("AV1FilmGrainMode")));
            let _ = message.property_list(names::PROPERTY_VALUE, Some(&string(mode)));
        })?;
        Message::parse(&reply).ok()?.s32(names::ERROR).ok()?
    };
    let supported = |guest: &Guest| -> Result<plist::Dictionary, WireError> {
        let reply = guest.call(b"dsup", WAIT, |_| {}).unwrap_or_default();
        Ok(Message::parse(&reply)?
            .dictionary(names::SUPPORTED_PROPERTIES)?
            .unwrap_or_default())
    };

    let av1 = Guest::new();
    av1.open_codec(b"av01", "reims-vgpu-video.videodecoder.av1")?;
    assert!(supported(&av1)?.contains_key("AV1FilmGrainMode"));
    assert_eq!(copy(&av1)?, (Some(NO_ERROR), Some(string("Immediate"))));
    // The guest sends Immediate alone, Defer rewritten to it.
    assert_eq!(set(&av1, "Immediate"), Some(NO_ERROR));
    assert_eq!(
        set(&av1, "Defer"),
        Some(Status::PropertyNotSupported.code())
    );
    assert_eq!(copy(&av1)?.1, Some(string("Immediate")));

    // Apple's AVC decoder lists no such property, and neither does ours.
    let avc = Guest::new();
    avc.open()?;
    assert!(!supported(&avc)?.contains_key("AV1FilmGrainMode"));
    assert_eq!(
        copy(&avc)?,
        (Some(Status::PropertyNotSupported.code()), None)
    );
    assert_eq!(
        set(&avc, "Immediate"),
        Some(Status::PropertyNotSupported.code())
    );
    Ok(())
}

#[test]
fn a_vp9_decoder_answers_for_its_properties_as_apples_does() -> Result<(), WireError> {
    // Measured on macOS 27.0.1 (26A434), 2026-10-25: Apple's VP9 decoder
    // lists UsingHardwareAcceleratedVideoDecoder,
    // SuggestedQualityOfServiceTiers and MaxDecoderPixelRate, before dsta
    // and after it, reads OnlyTheseFrames as AllFrames, and takes no
    // property: RealTime, MaximizePowerEfficiency and OnlyTheseFrames
    // -12900, its read-only ones -12901. Ours lists the first two.
    let string = |key: &str| plist::Value::String(key.into());
    let guest = Guest::new();
    guest.open_codec(b"vp09", "reims-vgpu-video.videodecoder.vp9")?;
    let supported = |guest: &Guest| -> Result<Vec<String>, WireError> {
        let reply = guest.call(b"dsup", WAIT, |_| {}).unwrap_or_default();
        let reply = Message::parse(&reply)?;
        assert_eq!(reply.s32(names::ERROR)?, Some(NO_ERROR));
        let supported = reply
            .dictionary(names::SUPPORTED_PROPERTIES)?
            .unwrap_or_default();
        assert!(supported.values().all(|description| {
            description
                .as_dictionary()
                .is_some_and(plist::Dictionary::is_empty)
        }));
        let mut keys: Vec<String> = supported.keys().cloned().collect();
        keys.sort();
        Ok(keys)
    };
    let listed = [
        "SuggestedQualityOfServiceTiers",
        "UsingHardwareAcceleratedVideoDecoder",
    ];
    assert_eq!(supported(&guest)?, listed);
    let copy = |key: &str| -> Result<(Option<i32>, Option<plist::Value>), WireError> {
        let reply = guest
            .call(b"dcop", WAIT, |message| {
                let _ = message.property_list(names::PROPERTY_KEY, Some(&string(key)));
            })
            .unwrap_or_default();
        let reply = Message::parse(&reply)?;
        Ok((
            reply.s32(names::ERROR)?,
            reply.property_list(names::PROPERTY_VALUE)?,
        ))
    };
    let set = |key: &str, value: plist::Value| -> Option<OsStatus> {
        let reply = guest.call(b"dset", WAIT, |message| {
            let _ = message.property_list(names::PROPERTY_KEY, Some(&string(key)));
            let _ = message.property_list(names::PROPERTY_VALUE, Some(&value));
        })?;
        Message::parse(&reply).ok()?.s32(names::ERROR).ok()?
    };
    let tiers = plist::Value::Array(
        ["AllFrames", "NonDroppableFrames", "IFrames", "KeyFrames"]
            .into_iter()
            .map(|tier| {
                let mut entry = plist::Dictionary::new();
                entry.insert("OnlyTheseFrames".into(), string(tier));
                plist::Value::Dictionary(entry)
            })
            .collect(),
    );
    let not_supported = (Some(Status::PropertyNotSupported.code()), None);
    for _ in 0..2 {
        assert_eq!(
            copy("UsingHardwareAcceleratedVideoDecoder")?,
            (Some(NO_ERROR), Some(plist::Value::Boolean(true)))
        );
        assert_eq!(
            copy("SuggestedQualityOfServiceTiers")?,
            (Some(NO_ERROR), Some(tiers.clone()))
        );
        assert_eq!(
            copy("OnlyTheseFrames")?,
            (Some(NO_ERROR), Some(string("AllFrames")))
        );
        for key in [
            "RealTime",
            "MaximizePowerEfficiency",
            "AV1FilmGrainMode",
            "MaxDecoderPixelRate",
            "NoSuchProperty",
        ] {
            assert_eq!(copy(key)?, not_supported, "{key}");
        }
        let true_ = plist::Value::Boolean(true);
        for (key, value, status) in [
            ("RealTime", true_.clone(), Status::PropertyNotSupported),
            (
                "MaximizePowerEfficiency",
                true_.clone(),
                Status::PropertyNotSupported,
            ),
            (
                "OnlyTheseFrames",
                string("KeyFrames"),
                Status::PropertyNotSupported,
            ),
            (
                "MaxDecoderPixelRate",
                plist::Value::Integer(1000.into()),
                Status::PropertyNotSupported,
            ),
            (
                "UsingHardwareAcceleratedVideoDecoder",
                plist::Value::Boolean(false),
                Status::PropertyReadOnly,
            ),
            (
                "SuggestedQualityOfServiceTiers",
                string("none"),
                Status::PropertyReadOnly,
            ),
        ] {
            assert_eq!(set(key, value), Some(status.code()), "{key}");
        }
        assert_eq!(copy("RealTime")?, not_supported);
        assert_eq!(
            copy("OnlyTheseFrames")?,
            (Some(NO_ERROR), Some(string("AllFrames")))
        );
        // The same once the decoder has started.
        let vpcc = vec![1, 0, 0, 0, 0, 31, 0x82, 2, 2, 2, 0, 0];
        let _ = guest.start_with(ImageDescription {
            codec: FourCc::new(*b"vp09"),
            width: 128,
            height: 96,
            depth: 24,
            compressor_name: Vec::new(),
            extensions: vec![(VP9_CONFIGURATION, vpcc)],
        })?;
        assert_eq!(supported(&guest)?, listed);
    }
    Ok(())
}

#[test]
fn the_decoder_list_offers_what_the_backend_decodes() -> Result<(), WireError> {
    let guest = Guest::new();
    let reply_id = 500;
    let mut decs = MessageWriter::new(
        FourCc::new(*b"decs"),
        MessageFlags::GUEST_EXPECTS_REPLY,
        None,
        reply_id,
    );
    decs.u32(names::VERSION, 6);
    let reply = guest
        .call_with(reply_id, decs.finish(), WAIT)
        .unwrap_or_default();
    let list = Message::parse(&reply)?.property_list(names::DECODERS)?;
    let Some(plist::Value::Array(entries)) = list else {
        panic!("no decoder list: {list:?}");
    };
    let entry = entries
        .first()
        .and_then(plist::Value::as_dictionary)
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        entry.get("CMClassImplementationID"),
        Some(&plist::Value::String(
            "reims-vgpu-video.videodecoder.avc".into()
        ))
    );
    let level = entry
        .get("VTDecoderCapabilities")
        .and_then(plist::Value::as_dictionary)
        .and_then(|capabilities| capabilities.get("VTPerProfileDetails"))
        .and_then(plist::Value::as_dictionary)
        .and_then(|details| details.get("100"))
        .and_then(plist::Value::as_dictionary)
        .and_then(|profile| profile.get("VTMaxDecodeLevel"))
        .and_then(as_i64);
    assert_eq!(level, Some(80));

    // Apple's host allows HDR for H.264 too (measured on macOS 27, 26A434).
    let reply = guest
        .call(b"dcap", WAIT, |message| {
            message.u32(names::VERSION, 6);
            message.s32(names::CODEC, i32::from_be_bytes(*b"avc1"));
        })
        .unwrap_or_default();
    let capabilities = Message::parse(&reply)?
        .property_list(names::DECODER_CAPABILITIES)?
        .and_then(|value| value.into_dictionary())
        .unwrap_or_default();
    assert_eq!(
        capabilities.get("VTIsHDRAllowedOnDevice"),
        Some(&plist::Value::Boolean(true))
    );
    Ok(())
}

#[test]
fn hevc_is_listed_with_main_and_main_10_at_apples_level() -> Result<(), WireError> {
    let guest = Guest::new();
    let mut decs = MessageWriter::new(
        FourCc::new(*b"decs"),
        MessageFlags::GUEST_EXPECTS_REPLY,
        None,
        501,
    );
    decs.u32(names::VERSION, 6);
    let reply = guest
        .call_with(501, decs.finish(), WAIT)
        .unwrap_or_default();
    let list = Message::parse(&reply)?.property_list(names::DECODERS)?;
    let Some(plist::Value::Array(entries)) = list else {
        panic!("no decoder list: {list:?}");
    };
    let hevc = entries
        .iter()
        .filter_map(plist::Value::as_dictionary)
        .find(|entry| {
            entry.get("CMClassImplementationID")
                == Some(&plist::Value::String(
                    "reims-vgpu-video.videodecoder.hevc".into(),
                ))
        })
        .cloned()
        .unwrap_or_default();
    assert_eq!(
        hevc.get("VTCodecType"),
        Some(&plist::Value::Array(vec![plist::Value::String(
            "hvc1".into()
        )]))
    );
    let details = hevc
        .get("VTDecoderCapabilities")
        .and_then(plist::Value::as_dictionary)
        .and_then(|capabilities| capabilities.get("VTPerProfileDetails"))
        .and_then(plist::Value::as_dictionary)
        .cloned()
        .unwrap_or_default();
    let level = |profile: &str| {
        details
            .get(profile)
            .and_then(plist::Value::as_dictionary)
            .and_then(|profile| profile.get("VTMaxDecodeLevel"))
            .and_then(as_i64)
    };
    assert_eq!(details.len(), 2);
    assert_eq!((level("1"), level("2")), (Some(186), Some(186)));

    let reply = guest
        .call(b"dcap", WAIT, |message| {
            message.u32(names::VERSION, 6);
            message.s32(names::CODEC, i32::from_be_bytes(*b"hvc1"));
        })
        .unwrap_or_default();
    let capabilities = Message::parse(&reply)?
        .property_list(names::DECODER_CAPABILITIES)?
        .and_then(|value| value.into_dictionary())
        .unwrap_or_default();
    // Apple's host allows HDR for HEVC (measured on macOS 27, 26A434).
    assert_eq!(
        capabilities.get("VTIsHDRAllowedOnDevice"),
        Some(&plist::Value::Boolean(true))
    );
    assert_eq!(
        capabilities.get("VTSupportedProfiles"),
        Some(&plist::Value::Array(vec![
            plist::Value::Integer(1.into()),
            plist::Value::Integer(2.into())
        ]))
    );
    let per_profile = capabilities
        .get("VTPerProfileSupport")
        .and_then(plist::Value::as_dictionary)
        .cloned()
        .unwrap_or_default();
    for profile in ["1", "2"] {
        let support = per_profile
            .get(profile)
            .and_then(plist::Value::as_dictionary)
            .cloned()
            .unwrap_or_default();
        assert_eq!(support.get("VTMaxDecodeLevel").and_then(as_i64), Some(186));
        assert_eq!(
            support.get("VTMaxPlaybackLevel").and_then(as_i64),
            Some(186)
        );
    }
    Ok(())
}

#[test]
fn a_10_bit_hevc_session_asks_for_x420_buffers() -> Result<(), WireError> {
    let guest = Guest::new();
    guest.open_codec(b"hvc1", "reims-vgpu-video.videodecoder.hevc")?;
    // The record's contents are the backend's to read; this one's are
    // only carried through.
    let hvcc = vec![
        1, 2, 0x20, 0, 0, 0, 0x90, 0, 0, 0, 0, 0, 93, 0xf0, 0, 0xfc, 0xfd, 0xfa, 0xfa, 0, 0, 0x0f,
        0,
    ];
    let status = guest.start_with(ImageDescription {
        codec: FourCc::new(*b"hvc1"),
        width: 64,
        height: 48,
        depth: 24,
        compressor_name: Vec::new(),
        extensions: vec![(HEVC_CONFIGURATION, hvcc.clone())],
    })?;
    assert_eq!(status, NO_ERROR);
    assert_eq!(guest.seen(), ["reply +dec", "Dpba", "reply dsta"]);
    let attributes = guest.state.attributes.lock().clone().unwrap_or_default();
    assert_eq!(
        attributes.get("PixelFormatType"),
        Some(&plist::Value::Array(vec![plist::Value::Integer(
            i64::from(u32::from_be_bytes(*b"x420")).into()
        )]))
    );
    let opened = guest.backend.opened.lock().first().cloned();
    assert_eq!(
        opened.as_ref().map(|config| config.codec),
        Some(Codec::Hevc)
    );
    assert_eq!(opened.map(|config| config.configuration), Some(hvcc));
    Ok(())
}

#[test]
fn a_decoder_for_another_decoders_id_is_refused() -> Result<(), WireError> {
    let guest = Guest::new();
    let reply = guest
        .call(b"+dec", WAIT, |message| {
            message.s32(names::CODEC, i32::from_be_bytes(*b"avc1"));
            message.u32(names::VERSION, 6);
            let _ = message.property_list(
                names::CODEC_ID,
                Some(&plist::Value::String(
                    "com.apple.videotoolbox.videodecoder.avd.avc".into(),
                )),
            );
        })
        .unwrap_or_default();
    assert_eq!(
        Message::parse(&reply)?.s32(names::ERROR)?,
        Some(Status::CouldNotFindVideoDecoder.code())
    );
    Ok(())
}

#[test]
fn invalidating_the_host_session_ends_a_decoders_wait_at_once() -> Result<(), WireError> {
    let guest = opened()?;
    guest.state.stalled.store(true, Ordering::SeqCst);
    guest.send({
        let mut message = MessageWriter::new(
            FourCc::new(*b"dfrm"),
            MessageFlags::GUEST_EXPECTS_REPLY,
            Some(guest.uuid),
            901,
        );
        message.s64(names::FRAME, 1);
        let serialization = sample(vec![1], Some(format(AVCC))).serialize();
        message.sample_buffer_with_limit(names::SAMPLE_BUFFER, FRAGMENT_LIMIT, &serialization);
        message.finish()
    });
    let started = StdRuntime.now();
    while !guest.seen().iter().any(|seen| seen == "Dbuf 1") {
        assert!(StdRuntime.elapsed_since(started) < WAIT, "no Dbuf");
        std::thread::sleep(Duration::from_millis(5));
    }
    let invalidating = StdRuntime.now();
    guest.session.invalidate();
    guest.session.complete_invalidate();
    assert!(StdRuntime.elapsed_since(invalidating) < Duration::from_millis(1000));
    assert_eq!(guest.memory.held(), 0);
    Ok(())
}

#[test]
fn closing_a_decoder_ends_its_session_and_its_holds() -> Result<(), WireError> {
    let guest = opened()?;
    guest.decode(1, 0, &sample(vec![1], Some(format(AVCC))))?;
    guest.close();
    guest.settle();
    assert_eq!(guest.memory.held(), 0);
    // Nothing answers for the UUID now: the host session finds no session.
    assert_eq!(
        guest.call(b"dfin", Duration::from_millis(200), |_| {}),
        None
    );
    Ok(())
}

#[test]
fn a_vp9_frame_of_another_size_is_lent_a_buffer_of_its_own_size() -> Result<(), WireError> {
    // Measured on macOS 27 (26A434): Apple's VP9 decoder emits -12909 and
    // no image for each frame whose coded size is not the description's.
    // A backend here that decodes such a frame shows it, with a Dpba at its
    // size before its Dbuf, as AVD's AV1 decoder asks.
    let guest = Guest::new();
    guest.open_codec(b"vp09", "reims-vgpu-video.videodecoder.vp9")?;
    let vpcc = vec![1, 0, 0, 0, 0, 31, 0x82, 2, 2, 2, 0, 0];
    let status = guest.start_with(ImageDescription {
        codec: FourCc::new(*b"vp09"),
        width: 128,
        height: 96,
        depth: 24,
        compressor_name: Vec::new(),
        extensions: vec![(VP9_CONFIGURATION, vpcc)],
    })?;
    assert_eq!(status, NO_ERROR);
    guest.seen();
    let size = |guest: &Guest| {
        let attributes = guest.state.attributes.lock().clone().unwrap_or_default();
        let read = |key| {
            attributes
                .get(key)
                .and_then(plist::Value::as_unsigned_integer)
        };
        (read("Width"), read("Height"))
    };
    let mut sizes = Vec::new();
    for (frame_id, first) in [(1, 0x40), (2, 0xfe), (3, 0xfe), (4, 0x41)] {
        assert_eq!(
            guest.decode(frame_id, 0, &sample(vec![first, 0x90], None))?,
            Some((NO_ERROR, 0))
        );
        sizes.push(size(&guest));
    }
    assert_eq!(
        guest.seen(),
        [
            "Dbuf 1",
            "Demt 1",
            "rels [1]",
            "reply dfrm",
            "Dpba",
            "Dbuf 2",
            "Demt 2",
            "rels [2]",
            "reply dfrm",
            "Dbuf 3",
            "Demt 3",
            "rels [3]",
            "reply dfrm",
            "Dpba",
            "Dbuf 4",
            "Demt 4",
            "rels [4]",
            "reply dfrm"
        ]
    );
    assert_eq!(
        sizes,
        [
            (Some(128), Some(96)),
            (Some(64), Some(48)),
            (Some(64), Some(48)),
            (Some(128), Some(96))
        ]
    );
    assert!(
        guest
            .emitted()
            .iter()
            .all(|frame| frame.status == NO_ERROR && frame.samples.is_some())
    );
    Ok(())
}

#[test]
fn vp9_sessions_of_12_bits_are_refused_as_apples_decoder_refuses_them() -> Result<(), WireError> {
    // Measured on macOS 27 (26A434): -12911 for 12 bits. No pixel format
    // here holds them.
    let guest = Guest::new();
    guest.open_codec(b"vp09", "reims-vgpu-video.videodecoder.vp9")?;
    let vpcc = vec![1, 0, 0, 0, 2, 21, 0xc2, 2, 2, 2, 0, 0];
    let status = guest.start_with(ImageDescription {
        codec: FourCc::new(*b"vp09"),
        width: 640,
        height: 360,
        depth: 24,
        compressor_name: Vec::new(),
        extensions: vec![(VP9_CONFIGURATION, vpcc)],
    })?;
    assert_eq!(status, Status::VideoDecoderMalfunction.code());
    assert_eq!(guest.seen(), ["reply +dec", "reply dsta"]);
    assert!(guest.backend.opened.lock().is_empty());
    Ok(())
}

#[test]
fn a_vp09_description_without_vpcc_is_refused_as_apples_decoder_refuses_it() -> Result<(), WireError>
{
    // Measured on macOS 27.0.1 (26A434), 2026-10-25: Apple's VP9 decoder
    // answers dsta -8971 with no Dpba, and each frame after it 311.
    let guest = Guest::new();
    guest.open_codec(b"vp09", "reims-vgpu-video.videodecoder.vp9")?;
    let status = guest.start_with(ImageDescription {
        codec: FourCc::new(*b"vp09"),
        width: 1280,
        height: 720,
        depth: 24,
        compressor_name: Vec::new(),
        extensions: Vec::new(),
    })?;
    assert_eq!(status, -8971);
    assert_eq!(status, Status::ConfigurationRecordMissing.code());
    assert_eq!(guest.seen(), ["reply +dec", "reply dsta"]);
    assert!(guest.backend.opened.lock().is_empty());
    assert_eq!(
        guest.decode(1, 0, &sample(vec![0x82, 0x49], None))?,
        Some((Status::DecoderNotStarted.code(), 0))
    );
    assert_eq!(guest.seen(), ["reply dfrm"]);
    Ok(())
}

#[test]
fn sessions_of_any_size_go_to_the_backend() -> Result<(), WireError> {
    // Apple's decoders refuse a side under 64 at dsta, H.264's and HEVC's
    // by the SPS's coded size (measured on macOS 27.0.1, 26A434). The sizes
    // a session takes are the backend's, so core asks the backend for each,
    // and a backend that does not decode one refuses it with its status.
    let av1 = crate::wire::image_description::AV1_CONFIGURATION;
    let av1c = vec![0x81, 0x00, 0x0c, 0x00];
    let vpcc = vec![1, 0, 0, 0, 0, 21, 0x82, 2, 2, 2, 0, 0];
    let cases = [
        (
            b"avc1",
            "avc",
            16,
            16,
            AVC_CONFIGURATION,
            avcc_with(&main_sps_of(true, 1, 1), 4),
        ),
        (
            b"avc1",
            "avc",
            48,
            64,
            AVC_CONFIGURATION,
            avcc_with(&main_sps_of(true, 3, 4), 4),
        ),
        (b"hvc1", "hevc", 16, 16, HEVC_CONFIGURATION, vec![1, 2, 3]),
        (b"vp09", "vp9", 2, 2, VP9_CONFIGURATION, vpcc.clone()),
        (b"vp09", "vp9", 63, 64, VP9_CONFIGURATION, vpcc),
        (b"av01", "av1", 1, 1, av1, av1c.clone()),
        (b"av01", "av1", 1280, 63, av1, av1c),
    ];
    for (codec, name, width, height, key, record) in cases {
        let guest = Guest::new();
        guest.open_codec(codec, &format!("reims-vgpu-video.videodecoder.{name}"))?;
        let status = guest.start_with(ImageDescription {
            codec: FourCc::new(*codec),
            width,
            height,
            depth: 24,
            compressor_name: Vec::new(),
            extensions: vec![(key, record)],
        })?;
        assert_eq!(status, NO_ERROR, "{name} at {width}x{height}");
        assert_eq!(guest.seen(), ["reply +dec", "Dpba", "reply dsta"]);
        assert_eq!(guest.backend.opened.lock().len(), 1);
    }
    Ok(())
}

#[test]
fn a_vp09_description_configures_vp9_with_its_vpcc_record() {
    // A vpcC box's payload for 720p Profile 2 at 10 bits, level 3.1, as
    // CoreMedia keeps it: version 1 and flags, then the record.
    let vpcc = vec![1, 0, 0, 0, 2, 31, 0xa2, 2, 2, 2, 0, 0];
    let description = ImageDescription {
        codec: FourCc::new(*b"vp09"),
        width: 1280,
        height: 720,
        depth: 24,
        compressor_name: Vec::new(),
        extensions: vec![(VP9_CONFIGURATION, vpcc.clone())],
    };
    let expected = DecoderConfig {
        codec: Codec::Vp9,
        width: 1280,
        height: 720,
        configuration: vpcc.clone(),
    };
    assert_eq!(
        super::config_from_description(&description),
        Some(expected.clone())
    );
    let format = SerializedFormatDescription {
        media_type: MEDIA_TYPE_VIDEO,
        codec: FourCc::new(*b"vp09"),
        width: 1280,
        height: 720,
        extensions: Some(vec![(
            "SampleDescriptionExtensionAtoms".into(),
            SerializedValue::Dictionary(vec![(
                "vpcC".into(),
                SerializedValue::PropertyList(plist::Value::Data(vpcc)),
            )]),
        )]),
    };
    assert_eq!(super::config_from_format(&format), Some(expected));
}

#[test]
fn an_av01_description_configures_av1_with_its_av1c_record() {
    // An av1C record's first four bytes, for 720p Main at 10 bits.
    let av1c = vec![0x81, 0x05, 0x4c, 0x00];
    let description = ImageDescription {
        codec: FourCc::new(*b"av01"),
        width: 1280,
        height: 720,
        depth: 24,
        compressor_name: Vec::new(),
        extensions: vec![(
            crate::wire::image_description::AV1_CONFIGURATION,
            av1c.clone(),
        )],
    };
    let expected = DecoderConfig {
        codec: Codec::Av1,
        width: 1280,
        height: 720,
        configuration: av1c.clone(),
    };
    assert_eq!(
        super::config_from_description(&description),
        Some(expected.clone())
    );
    let format = SerializedFormatDescription {
        media_type: MEDIA_TYPE_VIDEO,
        codec: FourCc::new(*b"av01"),
        width: 1280,
        height: 720,
        extensions: Some(vec![(
            "SampleDescriptionExtensionAtoms".into(),
            SerializedValue::Dictionary(vec![(
                "av1C".into(),
                SerializedValue::PropertyList(plist::Value::Data(av1c)),
            )]),
        )]),
    };
    assert_eq!(super::config_from_format(&format), Some(expected));
}

/// Apple's HEVC decoder answers a RASL picture it skips with
/// kVTVideoDecoderReferenceMissingErr and kVTDecodeInfo_FrameDropped and
/// SkippedLeadingFrameDropped, in a failed frame's order, and replies to
/// the `dfrm` last either way.
#[test]
fn a_skipped_leading_picture_is_emitted_as_apples_hevc_decoder_drops_one() -> Result<(), WireError>
{
    let guest = opened()?;
    guest.seen();
    for (frame, flags, buffer) in [(6, 0, 1), (7, 1, 2)] {
        assert_eq!(
            guest.decode(frame, flags, &sample(vec![0xfd], Some(format(AVCC))))?,
            Some((NO_ERROR, 0))
        );
        assert_eq!(
            guest.seen_once(5),
            [
                format!("Dbuf {frame}"),
                format!("Demt {frame}"),
                "Dfub".into(),
                format!("rels [{buffer}]"),
                "reply dfrm".into()
            ]
        );
    }
    let emitted = guest.emitted();
    assert_eq!(
        emitted
            .iter()
            .map(|frame| (frame.frame_id, frame.status, frame.info, frame.samples))
            .collect::<Vec<_>>(),
        [(6, -17694, 0xa, None), (7, -17694, 0xa, None)]
    );
    Ok(())
}

#[test]
fn a_failed_frame_decoded_asynchronously_is_answered_once_its_buffer_is_lent()
-> Result<(), WireError> {
    let guest = opened()?;
    guest.seen();
    assert_eq!(
        guest.decode(6, 1, &sample(vec![0xff], Some(format(AVCC))))?,
        Some((NO_ERROR, 1))
    );
    assert_eq!(
        guest.seen_once(5),
        ["Dbuf 6", "reply dfrm", "Demt 6", "Dfub", "rels [1]"]
    );
    let emitted = guest.emitted();
    assert_eq!(
        (emitted[0].status, emitted[0].info, emitted[0].samples),
        (Status::VideoDecoderBadData.code(), 1, None)
    );
    Ok(())
}

/// A Main SPS written from scratch (7.3.2.1.1), `width_in_mbs` macroblocks
/// by `height_in_map_units`: SPS 0, frame num and order count lsb of 4 bits
/// each, order count type 0, one reference, no gaps, `frame_mbs_only` as
/// given and no MBAFF, then `direct_8x8_inference_flag`, no cropping or
/// VUI.
fn main_sps_of(frame_mbs_only: bool, width_in_mbs: u32, height_in_map_units: u32) -> Vec<u8> {
    use super::rbsp::writer::{Field, escape, rbsp};
    let mut fields = vec![
        Field::Ue(0),
        Field::Ue(0),
        Field::Ue(0),
        Field::Ue(0),
        Field::Ue(1),
        Field::U(1, 0),
        Field::Ue(width_in_mbs - 1),
        Field::Ue(height_in_map_units - 1),
        Field::U(1, u32::from(frame_mbs_only)),
    ];
    if !frame_mbs_only {
        fields.push(Field::U(1, 0)); // mb_adaptive_frame_field_flag
    }
    fields.extend([Field::U(1, 1), Field::U(1, 0), Field::U(1, 0)]);
    let mut sps = vec![0x67, 77, 0, 30];
    sps.extend(escape(&rbsp(&fields)));
    sps
}

/// A Main SPS of 64x64, or 64x128 of field pairs.
fn main_sps(frame_mbs_only: bool) -> Vec<u8> {
    main_sps_of(frame_mbs_only, 4, 4)
}

/// An `avcC` record of one SPS and no PPS, with `length_size` bytes of
/// NAL unit length.
fn avcc_with(sps: &[u8], length_size: u8) -> Vec<u8> {
    let mut record = vec![1, sps[1], sps[2], sps[3], 0xfc | (length_size - 1), 0xe1, 0];
    record.push(u8::try_from(sps.len()).unwrap_or(0));
    record.extend_from_slice(sps);
    record.push(0);
    record
}

fn avc_description(record: Vec<u8>, width: u16, height: u16) -> ImageDescription {
    ImageDescription {
        codec: FourCc::new(*b"avc1"),
        width,
        height,
        depth: 24,
        compressor_name: Vec::new(),
        extensions: vec![(AVC_CONFIGURATION, record)],
    }
}

#[test]
fn avc_sessions_apples_decoder_refuses_are_refused_as_it_refuses_them() -> Result<(), WireError> {
    // Measured on macOS 27.0.1 (26A434): -12910 for 3-byte NAL unit
    // lengths, -12911 for any SPS with frame_mbs_only_flag 0, neither with
    // a Dpba first; 1, 2 and 4 bytes of a progressive SPS decode.
    for (record, expected) in [
        (
            avcc_with(&main_sps(true), 3),
            Status::VideoDecoderUnsupportedDataFormat.code(),
        ),
        (
            avcc_with(&main_sps(false), 4),
            Status::VideoDecoderMalfunction.code(),
        ),
        (
            avcc_with(&main_sps(false), 2),
            Status::VideoDecoderMalfunction.code(),
        ),
    ] {
        let guest = Guest::new();
        guest.open()?;
        assert_eq!(guest.start_with(avc_description(record, 64, 64))?, expected);
        assert_eq!(guest.seen(), ["reply +dec", "reply dsta"]);
        assert!(guest.backend.opened.lock().is_empty());
    }
    for length_size in [1, 2, 4] {
        let guest = Guest::new();
        guest.open()?;
        let record = avcc_with(&main_sps(true), length_size);
        assert_eq!(guest.start_with(avc_description(record, 64, 64))?, NO_ERROR);
        assert_eq!(guest.seen(), ["reply +dec", "Dpba", "reply dsta"]);
    }
    Ok(())
}

impl Guest {
    /// `dcan` with `description`: `,can`.
    fn can_accept(&self, description: ImageDescription) -> Result<i32, WireError> {
        let description = description.to_bytes();
        let reply = self
            .call(b"dcan", WAIT, |message| {
                message.image_description(names::FORMAT_DESCRIPTION, &description);
            })
            .ok_or(WireError::Malformed("no dcan reply"))?;
        Ok(Message::parse(&reply)?
            .s32(names::CAN_ACCEPT)?
            .unwrap_or(-1))
    }
}

#[test]
fn an_avc_session_accepts_only_what_apples_decoder_accepts() -> Result<(), WireError> {
    // Measured on macOS 27.0.1 (26A434): from a 4-byte session, AVD's dcan
    // answers 1 for the same length size and size, another profile among
    // them, and 0 for 3- or 2-byte lengths, an interlaced SPS, or another
    // size.
    let guest = Guest::new();
    guest.open()?;
    let progressive = avcc_with(&main_sps(true), 4);
    assert_eq!(
        guest.start_with(avc_description(progressive.clone(), 64, 64))?,
        NO_ERROR
    );
    let mut high = progressive.clone();
    high[1] = 100;
    for (record, width, can) in [
        (progressive.clone(), 64, 1),
        (high, 64, 1),
        (avcc_with(&main_sps(true), 3), 64, 0),
        (avcc_with(&main_sps(true), 2), 64, 0),
        (avcc_with(&main_sps(false), 4), 64, 0),
        (progressive, 128, 0),
    ] {
        assert_eq!(guest.can_accept(avc_description(record, width, 64))?, can);
    }
    Ok(())
}

#[test]
fn an_av1_frame_of_another_size_is_lent_a_buffer_of_its_own_size() -> Result<(), WireError> {
    // Measured on macOS 27.0.1 (26A434) with libaom's 640x360 key frames
    // and 569x320 frames between: AVD's AV1 decoder sends Dpba at a frame's
    // size before its Dbuf where the size changes, and not for the first
    // frame.
    let guest = Guest::new();
    guest.open_codec(b"av01", "reims-vgpu-video.videodecoder.av1")?;
    let status = guest.start_with(ImageDescription {
        codec: FourCc::new(*b"av01"),
        width: 128,
        height: 96,
        depth: 24,
        compressor_name: Vec::new(),
        extensions: vec![(
            crate::wire::image_description::AV1_CONFIGURATION,
            vec![0x81, 0x00, 0x0c, 0x00],
        )],
    })?;
    assert_eq!(status, NO_ERROR);
    guest.seen();
    let size = |guest: &Guest| {
        let attributes = guest.state.attributes.lock().clone().unwrap_or_default();
        let read = |key| {
            attributes
                .get(key)
                .and_then(plist::Value::as_unsigned_integer)
        };
        (read("Width"), read("Height"))
    };
    assert_eq!(size(&guest), (Some(128), Some(96)));
    let mut sizes = Vec::new();
    for (frame_id, first) in [(1, 0x40), (2, 0xfe), (3, 0xfe), (4, 0x41)] {
        assert_eq!(
            guest.decode(frame_id, 0, &sample(vec![first, 0x90], None))?,
            Some((NO_ERROR, 0))
        );
        sizes.push(size(&guest));
    }
    let asked: Vec<String> = guest
        .seen()
        .into_iter()
        .filter(|seen| seen == "Dpba" || seen.starts_with("Dbuf"))
        .collect();
    assert_eq!(
        asked,
        ["Dbuf 1", "Dpba", "Dbuf 2", "Dbuf 3", "Dpba", "Dbuf 4"]
    );
    assert_eq!(
        sizes,
        [
            (Some(128), Some(96)),
            (Some(64), Some(48)),
            (Some(64), Some(48)),
            (Some(128), Some(96))
        ]
    );
    // Every frame is emitted with its image.
    let emitted = guest.emitted();
    assert_eq!(emitted.len(), 4);
    assert!(
        emitted
            .iter()
            .all(|frame| frame.status == NO_ERROR && frame.samples.is_some())
    );
    Ok(())
}

/// One frame of a test AV1 temporal unit or VP9 sample.
#[derive(Clone, Copy)]
enum UnitFrame {
    /// A frame decoded at `size`, into the slots `refresh` names, loading
    /// its state from slot `primary`, and output or not.
    Decoded {
        refresh: u8,
        primary: Option<usize>,
        output: bool,
        size: (u32, u32),
    },
    /// The frame in a slot shown again.
    Again(usize),
    /// A VP9 frame refused: its sample does not decode, and no slot holds
    /// a frame after it.
    Refused,
}

/// An inter frame of 64x48, output or hidden.
fn decoded(refresh: u8, primary: Option<usize>, output: bool) -> UnitFrame {
    UnitFrame::Decoded {
        refresh,
        primary,
        output,
        size: (64, 48),
    }
}

/// The slots of a test AV1 or VP9 stream, which numbers its frames from 1.
#[derive(Default)]
struct UnitStream {
    slots: [Option<u64>; 8],
    pictures: u64,
}

impl UnitStream {
    /// What a backend's parser reports of a unit of `frames`: of one with a
    /// frame refused, no frame output, as the sample does not decode.
    fn unit(&mut self, frames: &[UnitFrame]) -> Vec<FrameDecoded> {
        let fails = frames
            .iter()
            .any(|frame| matches!(frame, UnitFrame::Refused));
        let mut unit: Vec<FrameDecoded> = frames
            .iter()
            .map(|frame| match *frame {
                UnitFrame::Decoded {
                    refresh,
                    primary,
                    output,
                    size: (width, height),
                } => {
                    let primary_reference = primary.and_then(|slot| self.slots[slot]);
                    self.pictures += 1;
                    for (i, slot) in self.slots.iter_mut().enumerate() {
                        if refresh & (1 << i) != 0 {
                            *slot = Some(self.pictures);
                        }
                    }
                    FrameDecoded {
                        picture: self.pictures,
                        shown_again: false,
                        refused: false,
                        width,
                        height,
                        references: self.slots,
                        primary_reference,
                        output,
                        largest: (64, 48),
                    }
                }
                UnitFrame::Again(slot) => FrameDecoded {
                    picture: self.slots[slot].unwrap_or(0),
                    shown_again: true,
                    refused: false,
                    width: 64,
                    height: 48,
                    references: self.slots,
                    primary_reference: None,
                    output: true,
                    largest: (64, 48),
                },
                UnitFrame::Refused => {
                    self.pictures += 1;
                    self.slots = [None; 8];
                    FrameDecoded {
                        picture: self.pictures,
                        shown_again: false,
                        refused: true,
                        width: 0,
                        height: 0,
                        references: self.slots,
                        primary_reference: None,
                        output: false,
                        largest: (0, 0),
                    }
                }
            })
            .collect();
        if fails {
            for step in &mut unit {
                step.output = false;
            }
        }
        unit
    }
}

/// What the guest saw of the buffers ([`GuestState::buffers`]), one line
/// for each `dfrm` up to its reply, and what came after the last reply.
fn by_frame(guest: &Guest) -> Vec<String> {
    let mut frames = Vec::new();
    let mut frame = Vec::new();
    for seen in guest.state.buffers.lock().iter() {
        if seen == "reply dfrm" {
            frames.push(std::mem::take(&mut frame).join(", "));
        } else {
            frame.push(seen.clone());
        }
    }
    if !frame.is_empty() {
        frames.push(frame.join(", "));
    }
    frames
}

/// A guest with an AV1 session started, whose backend decodes `units`, and
/// nothing seen yet.
fn av1_session(units: &[&[UnitFrame]]) -> Result<Guest, WireError> {
    let guest = Guest::new();
    let mut stream = UnitStream::default();
    guest
        .backend
        .units
        .lock()
        .extend(units.iter().map(|unit| stream.unit(unit)));
    guest.open_codec(b"av01", "reims-vgpu-video.videodecoder.av1")?;
    // Apple's minimum is 64x64; the frames are smaller, as the stand-in
    // guest's buffers are.
    let status = guest.start_with(ImageDescription {
        codec: FourCc::new(*b"av01"),
        width: 128,
        height: 96,
        depth: 24,
        compressor_name: Vec::new(),
        extensions: vec![(
            crate::wire::image_description::AV1_CONFIGURATION,
            vec![0x81, 0x00, 0x0c, 0x00],
        )],
    })?;
    assert_eq!(status, NO_ERROR);
    guest.seen();
    guest.state.buffers.lock().clear();
    Ok(guest)
}

/// The first 13 temporal units of libaom's 1280x720 stream with forward
/// key frames (`aom-720p-fwdkf`, read with `av1-info --frames`): a key
/// frame, then units of hidden frames each shown again later from its
/// slot.
fn forward_key_frame_units() -> Vec<Vec<UnitFrame>> {
    let key = UnitFrame::Decoded {
        refresh: 0xff,
        primary: None,
        output: true,
        size: (64, 48),
    };
    vec![
        vec![key],
        vec![
            decoded(0x02, None, false),
            decoded(0x04, None, false),
            decoded(0x08, Some(2), false),
            decoded(0x10, Some(3), false),
            decoded(0x20, None, true),
        ],
        vec![UnitFrame::Again(4)],
        vec![decoded(0x40, Some(5), true)],
        vec![UnitFrame::Again(3)],
        vec![decoded(0x80, Some(4), false), decoded(0x20, Some(6), true)],
        vec![UnitFrame::Again(7)],
        vec![decoded(0x10, Some(5), true)],
        vec![UnitFrame::Again(2)],
        vec![
            decoded(0x40, Some(7), false),
            decoded(0x08, Some(6), false),
            decoded(0x20, Some(4), true),
        ],
        vec![UnitFrame::Again(3)],
        vec![decoded(0x80, Some(5), true)],
        vec![UnitFrame::Again(6)],
    ]
}

/// Apple's AV1 decoder lends a buffer for each frame a temporal unit
/// carries, emits a frame shown again in the buffer it was decoded into,
/// and lets each buffer go a step after no slot holds its frame, two after
/// its decode, and two after the last frame that loaded its state. Its
/// messages for these units, synchronously, as recorded on Apple's host
/// (macOS 27.0.1, 26A434, 2026-10-24), with its buffer IDs, which the
/// stand-in guest's mapping IDs follow.
#[test]
fn av1_frames_are_lent_and_held_as_apples_decoder_lends_and_holds_them() -> Result<(), WireError> {
    let units = forward_key_frame_units();
    let units: Vec<&[UnitFrame]> = units.iter().map(Vec::as_slice).collect();
    let guest = av1_session(&units)?;
    for frame_id in 0..13 {
        assert_eq!(
            guest.decode(frame_id, 0, &sample(vec![0x10], None))?,
            Some((NO_ERROR, 0))
        );
    }
    let apples = [
        "Dbuf 0 -> 1, Demt 0 1",
        "Dbuf 1 -> 2, Dbuf 1 -> 3, Dbuf 1 -> 4, Dbuf 1 -> 5, Dbuf 1 -> 6, Demt 1 6",
        "Dbuf 2 -> 7, Demt 2 5",
        "rels [7], Dbuf 3 -> 8, Demt 3 8, Dfub [7]",
        "Dbuf 4 -> 9, Demt 4 4",
        "rels [9], Dbuf 5 -> 10, Dfub [9], Dbuf 5 -> 11, Demt 5 11",
        "rels [6], Dbuf 6 -> 12, Demt 6 10",
        "rels [12], Dbuf 7 -> 13, Demt 7 13, Dfub [12]",
        "rels [5], Dbuf 8 -> 14, Demt 8 3",
        "rels [14], Dbuf 9 -> 15, Dfub [14], Dbuf 9 -> 16, rels [8], Dbuf 9 -> 17, rels [4], \
         Demt 9 17",
        "rels [11], Dbuf 10 -> 18, Demt 10 16",
        "rels [18], Dbuf 11 -> 19, Demt 11 19, Dfub [18]",
        "rels [10], Dbuf 12 -> 20, Demt 12 15",
    ];
    assert_eq!(by_frame(&guest), apples);
    // Each frame shown is its own picture, in the buffer it was decoded
    // into: frames are numbered in decode order, and each is filled with
    // its number.
    let lumas: Vec<u8> = guest
        .emitted()
        .iter()
        .map(|frame| frame.samples.map_or(0, |samples| samples[0]))
        .collect();
    assert_eq!(lumas, [1, 6, 5, 7, 4, 9, 8, 10, 3, 13, 12, 14, 11]);
    // Nothing is written into a buffer whose frame is not shown: frame 2's
    // (buffer 2), and those lent to show a frame again (7, 9, 12, ...).
    let surfaces = guest.memory.surfaces.lock();
    for unwritten in [2, 7, 9, 12, 14, 18, 20] {
        let bytes = surfaces
            .get(&unwritten)
            .map(|memory| memory.bytes.lock().clone());
        assert!(
            bytes.is_some_and(|bytes| bytes.iter().all(|&byte| byte == 0)),
            "buffer {unwritten} was written"
        );
    }
    Ok(())
}

/// Has `guest`'s device thread deliver a message for no session (`-all`,
/// which nothing answers), and waits for its delivery to end, its sweep
/// included, `bound` at most: whether it ended.
fn deliver_meanwhile(queue0: &Sender<Vec<u8>>, delivered: &AtomicU64, bound: Duration) -> bool {
    let before = delivered.load(Ordering::SeqCst);
    let message = MessageWriter::new(FourCc::new(*b"-all"), MessageFlags::NONE, None, 0);
    if queue0.send(message.finish()).is_err() {
        return false;
    }
    let started = StdRuntime.now();
    while delivered.load(Ordering::SeqCst) == before {
        if StdRuntime.elapsed_since(started) >= bound {
            return false;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    true
}

/// Two buffers let go at one step reach the guest in one `rels`, once both
/// holds have ended, whatever another thread's sweep does meanwhile: here
/// the device thread delivers a message, and sweeps, as the session lets go
/// of each, before that hold has ended. Buffers 1 and 2 are due at step 5,
/// once step 4 has refreshed every slot that held their frames, and are let
/// go as the guest's next `dfrm` comes, as Apple's are; a delivery's sweep
/// that names the first and not the second splits their `rels` in two, and
/// one that names a hold not ended tells the guest it is free while the
/// host still holds it.
#[test]
fn buffers_let_go_together_are_named_in_one_rels_once_let_go() -> Result<(), WireError> {
    let key = UnitFrame::Decoded {
        refresh: 0xff,
        primary: None,
        output: true,
        size: (64, 48),
    };
    let units: [&[UnitFrame]; 5] = [
        &[key],
        &[decoded(0x01, None, true)],
        &[UnitFrame::Again(0)],
        &[decoded(0xff, None, true)],
        &[decoded(0x01, None, true)],
    ];
    let guest = av1_session(&units)?;
    for frame_id in 0..4 {
        assert_eq!(
            guest.decode(frame_id, 0, &sample(vec![0x10], None))?,
            Some((NO_ERROR, 0))
        );
    }
    guest.settle();
    let named_early = Arc::new(Mutex::new(Vec::new()));
    for buffer in [1, 2] {
        let queue0 = guest
            .queue0
            .clone()
            .ok_or(WireError::Malformed("no queue 0"))?;
        let delivered = guest.delivered.clone();
        let progress = guest.progress.clone();
        let state = guest.state.clone();
        let named_early = named_early.clone();
        let on_release: OnRelease = Box::new(move || {
            // What that delivery's sweep sent reaches the guest before its
            // record is read; a guest that never got it is u64::MAX.
            if deliver_meanwhile(&queue0, &delivered, Duration::from_millis(200))
                && !progress.caught_up(WAIT)
            {
                named_early.lock().push(u64::MAX);
            }
            let named = state.buffers.lock().iter().any(|seen| {
                seen.strip_prefix("rels [")
                    .and_then(|ids| ids.strip_suffix(']'))
                    .is_some_and(|ids| ids.split(", ").any(|id| id == buffer.to_string()))
            });
            if named {
                named_early.lock().push(buffer);
            }
        });
        guest.memory.on_release.lock().insert(buffer, on_release);
    }
    assert_eq!(
        guest.decode(4, 0, &sample(vec![0x10], None))?,
        Some((NO_ERROR, 0))
    );
    guest.settle();
    guest
        .state
        .buffers
        .lock()
        .retain(|seen| seen != "reply decs");
    assert_eq!(
        by_frame(&guest),
        [
            "Dbuf 0 -> 1, Demt 0 1",
            "Dbuf 1 -> 2, Demt 1 2",
            "Dbuf 2 -> 3, Demt 2 2",
            "rels [3], Dbuf 3 -> 4, Demt 3 4, Dfub [3]",
            "rels [1, 2], Dbuf 4 -> 5, Demt 4 5",
        ]
    );
    assert_eq!(*named_early.lock(), Vec::<u64>::new());
    assert!(guest.memory.on_release.lock().is_empty());
    Ok(())
}

/// Apple's AV1 decoder lends a buffer for each spatial layer's frame,
/// asking for the guest's buffers at each frame's size where it changes,
/// from the sequence's largest frame on, and gives back the lower layer's,
/// which are never emitted. Its messages for the first five units of
/// libvpx's two-layer stream (`svc-720p-l2t1`, layers of 640x360 and
/// 1280x720, here 32x24 and 64x48), as recorded as above.
#[test]
fn both_spatial_layers_are_lent_buffers_at_their_own_sizes() -> Result<(), WireError> {
    let lower = |refresh, primary| UnitFrame::Decoded {
        refresh,
        primary,
        output: false,
        size: (32, 24),
    };
    let first: &[UnitFrame] = &[lower(0xff, None), decoded(0x02, None, true)];
    let next: &[UnitFrame] = &[lower(0x01, Some(0)), decoded(0x02, Some(1), true)];
    let guest = av1_session(&[first, next, next, next, next])?;
    for frame_id in 0..5 {
        assert_eq!(
            guest.decode(frame_id, 0, &sample(vec![0x10], None))?,
            Some((NO_ERROR, 0))
        );
    }
    let apples = [
        "Dpba 32x24, Dbuf 0 -> 1, Dpba 64x48, Dbuf 0 -> 2, Demt 0 2",
        "Dpba 32x24, Dbuf 1 -> 3, Dpba 64x48, Dbuf 1 -> 4, Demt 1 4",
        "Dpba 32x24, Dbuf 2 -> 5, Dpba 64x48, Dbuf 2 -> 6, rels [2], Demt 2 6",
        "rels [3], Dpba 32x24, Dbuf 3 -> 7, Dfub [3], Dpba 64x48, Dbuf 3 -> 8, rels [4], Demt 3 8",
        "rels [5], Dpba 32x24, Dbuf 4 -> 9, Dfub [5], Dpba 64x48, Dbuf 4 -> 10, rels [6], \
         Demt 4 10",
    ];
    assert_eq!(by_frame(&guest), apples);
    Ok(())
}

/// Asynchronously, Apple's AV1 decoder asks for all of a unit's buffers,
/// answers, then decodes the unit's frames: each step's `Dfub`, the
/// `Demt` and the `rels` of what each step lets go come after the reply.
/// Its messages for the units of the synchronous test above, sent a frame
/// each 33 ms after its reply, as a player does (macOS 27.0.1, 26A434,
/// 2026-10-24). In unit 9 Apple's `Dfub [14]`, of the unit's first step,
/// came before its last `Dbuf`, as AVD's hardware finished that step
/// while the unit's buffers were still being asked for; here it comes
/// after the reply, as in units 5 and 13 of the same run.
#[test]
fn an_asynchronous_av1_unit_asks_for_its_buffers_before_it_decodes() -> Result<(), WireError> {
    let units = forward_key_frame_units();
    let units: Vec<&[UnitFrame]> = units.iter().map(Vec::as_slice).collect();
    let guest = av1_session(&units)?;
    for frame_id in 0..13 {
        assert_eq!(
            guest.decode(frame_id, 1, &sample(vec![0x10], None))?,
            Some((NO_ERROR, 1))
        );
    }
    guest.settle();
    // The last unit's frame is emitted after its reply, once its steps are
    // seen through on the session's thread, which the settle does not
    // wait for.
    assert!(
        wait_for_buffers(&guest, "Demt 12 15"),
        "{:?}",
        by_frame(&guest)
    );
    guest
        .state
        .buffers
        .lock()
        .retain(|seen| seen != "reply decs");
    // Each frame's messages up to its reply: its buffers, and what the
    // frame before it did after its own reply.
    let apples = [
        "Dbuf 0 -> 1",
        "Demt 0 1, Dbuf 1 -> 2, Dbuf 1 -> 3, Dbuf 1 -> 4, Dbuf 1 -> 5, Dbuf 1 -> 6",
        "Demt 1 6, Dbuf 2 -> 7",
        "Demt 2 5, rels [7], Dbuf 3 -> 8",
        "Demt 3 8, Dfub [7], Dbuf 4 -> 9",
        "Demt 4 4, rels [9], Dbuf 5 -> 10, Dbuf 5 -> 11",
        "Dfub [9], Demt 5 11, rels [6], Dbuf 6 -> 12",
        "Demt 6 10, rels [12], Dbuf 7 -> 13",
        "Demt 7 13, Dfub [12], rels [5], Dbuf 8 -> 14",
        "Demt 8 3, rels [14], Dbuf 9 -> 15, Dbuf 9 -> 16, Dbuf 9 -> 17",
        "Dfub [14], rels [8], Demt 9 17, rels [4], rels [11], Dbuf 10 -> 18",
        "Demt 10 16, rels [18], Dbuf 11 -> 19",
        "Demt 11 19, Dfub [18], rels [10], Dbuf 12 -> 20",
        "Demt 12 15",
    ];
    assert_eq!(by_frame(&guest), apples);
    let lumas: Vec<u8> = guest
        .emitted()
        .iter()
        .map(|frame| frame.samples.map_or(0, |samples| samples[0]))
        .collect();
    assert_eq!(lumas, [1, 6, 5, 7, 4, 9, 8, 10, 3, 13, 12, 14, 11]);
    Ok(())
}

/// An asynchronous unit of more frames than a session holds buffers for
/// sees its first steps through before it asks for more, so that it never
/// holds more than the bound (ours, for host safety).
#[test]
fn a_long_asynchronous_unit_holds_no_more_than_the_bound() -> Result<(), WireError> {
    let key = UnitFrame::Decoded {
        refresh: 0xff,
        primary: None,
        output: true,
        size: (64, 48),
    };
    // Twenty frames into slot 0, each out of the slots at the next.
    let mut long: Vec<UnitFrame> = (0..19).map(|_| decoded(0x01, None, false)).collect();
    long.push(decoded(0x01, None, true));
    let guest = av1_session(&[&[key], &long])?;
    for frame_id in 0..2 {
        assert_eq!(
            guest.decode(frame_id, 1, &sample(vec![0x10], None))?,
            Some((NO_ERROR, 1))
        );
    }
    // The unit's frame is emitted, in the last buffer lent, after its
    // steps are seen through on the session's thread.
    let emitted = wait_for_buffers(&guest, "Demt 1 21");
    let seen = guest.state.buffers.lock().clone();
    assert!(emitted, "{seen:?}");
    let (mut held, mut most) = (0usize, 0usize);
    for line in &seen {
        if line.starts_with("Dbuf") {
            held += 1;
            most = most.max(held);
        } else if let Some(ids) = line.strip_prefix("rels ") {
            held -= ids.split(',').count();
        }
    }
    assert_eq!(
        seen.iter().filter(|line| line.starts_with("Dbuf")).count(),
        21
    );
    assert_eq!(most, 16);
    Ok(())
}

/// A frame shown again whose buffer the session does not hold, as after
/// a decode the guest gave up on, is emitted in the buffer lent to show it.
/// A guest with a VP9 session started, whose backend decodes `units`, and
/// nothing seen yet.
fn vp9_session(units: &[&[UnitFrame]]) -> Result<Guest, WireError> {
    let guest = Guest::new();
    let mut stream = UnitStream::default();
    guest
        .backend
        .units
        .lock()
        .extend(units.iter().map(|unit| stream.unit(unit)));
    guest.open_codec(b"vp09", "reims-vgpu-video.videodecoder.vp9")?;
    let status = guest.start_with(ImageDescription {
        codec: FourCc::new(*b"vp09"),
        width: 128,
        height: 96,
        depth: 24,
        compressor_name: Vec::new(),
        extensions: vec![(
            VP9_CONFIGURATION,
            vec![1, 0, 0, 0, 0, 31, 0x82, 2, 2, 2, 0, 0],
        )],
    })?;
    assert_eq!(status, NO_ERROR);
    guest.seen();
    guest.state.buffers.lock().clear();
    Ok(guest)
}

/// A VP9 key frame, which every slot holds.
const VP9_KEY: UnitFrame = UnitFrame::Decoded {
    refresh: 0xff,
    primary: None,
    output: true,
    size: (64, 48),
};

/// Apple's VP9 decoder lends a buffer for each frame a sample carries, a
/// superframe's hidden alternate references and each frame shown again,
/// emits a frame shown again in the buffer it was decoded into, and lets
/// each buffer go a step after no slot holds its frame and two after its
/// decode. Its messages for the first 13 samples of libvpx's 1280x720
/// stream with six alternate references (`p0-720p-arf6`, read with
/// `vp9-info --frames`), synchronously, as recorded on Apple's host session
/// on the M3 Max (macOS 27.0.1, 2026-10-25), with its buffer IDs, which the
/// stand-in guest's mapping IDs follow.
#[test]
fn vp9_frames_are_lent_and_held_as_apples_decoder_lends_and_holds_them() -> Result<(), WireError> {
    let hidden = |refresh| decoded(refresh, None, false);
    let shown = |refresh| decoded(refresh, None, true);
    let units: [&[UnitFrame]; 13] = [
        &[VP9_KEY],
        &[
            hidden(0x08),
            hidden(0x10),
            hidden(0x20),
            hidden(0x40),
            shown(0x01),
        ],
        &[UnitFrame::Again(6)],
        &[shown(0x40)],
        &[UnitFrame::Again(5)],
        &[hidden(0x01), shown(0x20)],
        &[UnitFrame::Again(0)],
        &[shown(0x01)],
        &[UnitFrame::Again(4)],
        &[hidden(0x01), hidden(0x20), shown(0x10)],
        &[UnitFrame::Again(5)],
        &[shown(0x20)],
        &[UnitFrame::Again(0)],
    ];
    let guest = vp9_session(&units)?;
    for frame_id in 0..13 {
        assert_eq!(
            guest.decode(frame_id, 0, &sample(vec![0x82], None))?,
            Some((NO_ERROR, 0))
        );
    }
    let apples = [
        "Dbuf 0 -> 1, Demt 0 1",
        "Dbuf 1 -> 2, Dbuf 1 -> 3, Dbuf 1 -> 4, Dbuf 1 -> 5, Dbuf 1 -> 6, Demt 1 6",
        "Dbuf 2 -> 7, Demt 2 5",
        "rels [7], Dbuf 3 -> 8, Demt 3 8, Dfub [7]",
        "rels [5], Dbuf 4 -> 9, Demt 4 4",
        "rels [9], Dbuf 5 -> 10, Dfub [9], Dbuf 5 -> 11, rels [6], Demt 5 11",
        "rels [4], Dbuf 6 -> 12, Demt 6 10",
        "rels [12], Dbuf 7 -> 13, Demt 7 13, Dfub [12]",
        "rels [10], Dbuf 8 -> 14, Demt 8 3",
        "rels [14], Dbuf 9 -> 15, Dfub [14], Dbuf 9 -> 16, rels [13], Dbuf 9 -> 17, rels [11], \
         Demt 9 17",
        "rels [3], Dbuf 10 -> 18, Demt 10 16",
        "rels [18], Dbuf 11 -> 19, Demt 11 19, Dfub [18]",
        "rels [16], Dbuf 12 -> 20, Demt 12 15",
    ];
    assert_eq!(by_frame(&guest), apples);
    // Each frame shown in the buffer it was decoded into, filled with its
    // number: frames are numbered in decode order.
    let lumas: Vec<u8> = guest
        .emitted()
        .iter()
        .map(|frame| frame.samples.map_or(0, |samples| samples[0]))
        .collect();
    assert_eq!(lumas, [1, 6, 5, 7, 4, 9, 8, 10, 3, 13, 12, 14, 11]);
    Ok(())
}

/// Apple's VP9 decoder refuses a frame of another size than the
/// description's, and each after it until a key frame, and lends a buffer
/// for each all the same: the refused frame's `Dbuf`, then its `Demt` with
/// -12909 and no image, then the reply, last asynchronously too. The
/// buffer and those the slots held are let go at the next frame, their
/// `rels` with the next `dfrm`, and the refused frame's comes back in the
/// `Dfub` after that frame's `Demt`. Its messages for the first 5 samples
/// of libvpx's dynamic resize from its second frame (`p0-720p-resize-early`),
/// as recorded on Apple's host session on the M3 Max (macOS 27.0.1,
/// 2026-10-25), then those it sent at the 1280x720 key frame of
/// `p0-720p-resize` with `p0-720p` after it, after 180 refused frames:
/// the key frame's `Dfub` gives back the last refused frame's buffer.
#[test]
fn a_refused_vp9_frame_is_lent_a_buffer_let_go_with_the_next_dfrm() -> Result<(), WireError> {
    let refused = [UnitFrame::Refused];
    let units: [&[UnitFrame]; 9] = [
        &[VP9_KEY],
        &refused,
        &refused,
        &refused,
        &refused,
        &[VP9_KEY],
        &[decoded(0x04, None, false), decoded(0x01, None, true)],
        &[decoded(0x01, None, true)],
        &[decoded(0x01, None, true)],
    ];
    let bad_data = Status::VideoDecoderBadData.code();
    for flags in [0, 1] {
        let guest = vp9_session(&units)?;
        for frame_id in 0..9 {
            assert_eq!(
                guest.decode(frame_id, flags, &sample(vec![0x82], None))?,
                Some((NO_ERROR, flags))
            );
        }
        guest.settle();
        assert!(
            wait_for_buffers(&guest, "Demt 8 10"),
            "{:?}",
            by_frame(&guest)
        );
        guest
            .state
            .buffers
            .lock()
            .retain(|seen| seen != "reply decs");
        let mut apples = if flags == 0 {
            vec!["Dbuf 0 -> 1, Demt 0 1", "Dbuf 1 -> 2, Demt 1 -"]
        } else {
            // The first frame's Demt comes after its reply; the refused
            // frame's before it.
            vec!["Dbuf 0 -> 1", "Demt 0 1, Dbuf 1 -> 2, Demt 1 -"]
        };
        apples.extend([
            "rels [1, 2], Dbuf 2 -> 3, Demt 2 -, Dfub [2]",
            "rels [3], Dbuf 3 -> 4, Demt 3 -, Dfub [3]",
            "rels [4], Dbuf 4 -> 5, Demt 4 -, Dfub [4]",
        ]);
        if flags == 0 {
            apples.extend([
                "rels [5], Dbuf 5 -> 6, Demt 5 6, Dfub [5]",
                "Dbuf 6 -> 7, Dbuf 6 -> 8, Demt 6 8",
                // Apple's run let buffer 8 go before this reply, a race,
                // where its rels most often comes with the next dfrm.
                "Dbuf 7 -> 9, Demt 7 9",
                "rels [8], Dbuf 8 -> 10, Demt 8 10",
            ]);
        } else {
            apples.extend([
                "rels [5], Dbuf 5 -> 6",
                "Demt 5 6, Dfub [5], Dbuf 6 -> 7, Dbuf 6 -> 8",
                "Demt 6 8, Dbuf 7 -> 9",
                "Demt 7 9, rels [8], Dbuf 8 -> 10",
                "Demt 8 10",
            ]);
        }
        assert_eq!(by_frame(&guest), apples, "flags {flags}");
        let statuses: Vec<(i64, i32, u32, bool)> = guest
            .emitted()
            .iter()
            .map(|frame| {
                (
                    frame.frame_id,
                    frame.status,
                    frame.info,
                    frame.samples.is_some(),
                )
            })
            .collect();
        let mut expected = vec![(0, NO_ERROR, flags, true)];
        expected.extend((1..5).map(|frame_id| (frame_id, bad_data, flags, false)));
        expected.extend((5..9).map(|frame_id| (frame_id, NO_ERROR, flags, true)));
        assert_eq!(statuses, expected, "flags {flags}");
    }
    Ok(())
}

/// A sample whose frames Apple's VP9 decoder takes, then refuses one of,
/// is lent a buffer for each: libvpx's three spatial layers in one
/// superframe (`vp9-svc 3 1 0 0`) under a description of the lowest
/// layer's 320x180, whose 640x360 frame AVD refuses, and each frame of the
/// samples after it. The key frame decoded before it goes with the slots
/// at the refused frame, and nothing is emitted but the sample's -12909.
/// Its messages for the first 3 samples, as recorded on Apple's host
/// session on the M3 Max (macOS 27.0.1, 2026-10-25), the same synchronously
/// and not.
#[test]
fn every_frame_of_a_refused_vp9_superframe_is_lent_a_buffer() -> Result<(), WireError> {
    let refused = UnitFrame::Refused;
    let units: [&[UnitFrame]; 3] = [
        &[VP9_KEY, refused, refused],
        &[refused, refused, refused],
        &[refused, refused, refused],
    ];
    for flags in [0, 1] {
        let guest = vp9_session(&units)?;
        for frame_id in 0..3 {
            assert_eq!(
                guest.decode(frame_id, flags, &sample(vec![0x82], None))?,
                Some((NO_ERROR, flags))
            );
        }
        let apples = [
            "Dbuf 0 -> 1, Dbuf 0 -> 2, Dbuf 0 -> 3, rels [1, 2], Demt 0 -, Dfub [1, 2]",
            "rels [3], Dbuf 1 -> 4, Dfub [3], Dbuf 1 -> 5, rels [4], Dfub [4], Dbuf 1 -> 6, \
             rels [5], Demt 1 -, Dfub [5]",
            "rels [6], Dbuf 2 -> 7, Dfub [6], Dbuf 2 -> 8, rels [7], Dfub [7], Dbuf 2 -> 9, \
             rels [8], Demt 2 -, Dfub [8]",
        ];
        assert_eq!(by_frame(&guest), apples, "flags {flags}");
        assert!(
            guest.emitted().iter().all(|frame| frame.samples.is_none()
                && frame.status == Status::VideoDecoderBadData.code())
        );
        // Nothing is written into the buffers lent for frames refused.
        let surfaces = guest.memory.surfaces.lock();
        assert!(
            (1..=9).all(|id| surfaces.get(&id).is_some_and(|memory| memory
                .bytes
                .lock()
                .iter()
                .all(|&byte| byte == 0))),
            "a buffer was written"
        );
    }
    Ok(())
}

#[test]
fn a_frame_shown_again_without_its_buffer_is_emitted_in_its_own() -> Result<(), WireError> {
    let guest = Guest::new();
    let key = FrameDecoded {
        picture: 3,
        shown_again: true,
        refused: false,
        width: 64,
        height: 48,
        references: [Some(3); 8],
        primary_reference: None,
        output: true,
        largest: (64, 48),
    };
    guest.backend.units.lock().push_back(vec![key]);
    guest.open_codec(b"av01", "reims-vgpu-video.videodecoder.av1")?;
    // Apple's minimum is 64x64; the frames are smaller, as the stand-in
    // guest's buffers are.
    let status = guest.start_with(ImageDescription {
        codec: FourCc::new(*b"av01"),
        width: 128,
        height: 96,
        depth: 24,
        compressor_name: Vec::new(),
        extensions: vec![(
            crate::wire::image_description::AV1_CONFIGURATION,
            vec![0x81, 0x00, 0x0c, 0x00],
        )],
    })?;
    assert_eq!(status, NO_ERROR);
    guest.state.buffers.lock().clear();
    assert_eq!(
        guest.decode(0, 0, &sample(vec![0x10], None))?,
        Some((NO_ERROR, 0))
    );
    assert_eq!(by_frame(&guest), ["Dbuf 0 -> 1, Demt 0 1"]);
    assert_eq!(
        guest.emitted().first().and_then(|frame| frame.samples),
        Some([3, 3, 0, 0x80])
    );
    Ok(())
}

/// Where the host lends a buffer's memory to devices and the backend's
/// device writes it, the frame goes there with no read back and no copy by
/// the CPU, and the guest sees what it saw with the CPU's copy.
#[test]
fn a_device_writes_frames_into_the_buffers_the_host_lends_it() -> Result<(), WireError> {
    let guest = opened()?;
    guest.seen();
    let frames = &guest.backend.frames;
    frames.device_writes.store(true, Ordering::SeqCst);
    let reply = guest.decode(1, 0, &sample(vec![0x40, 0x90], Some(format(AVCC))))?;
    assert_eq!(reply, Some((NO_ERROR, 0)));
    assert_eq!(guest.seen(), ["Dbuf 1", "Demt 1", "rels [1]", "reply dfrm"]);
    let emitted = guest.emitted();
    assert_eq!(emitted[0].samples, Some([0x40, 0x40, 0, 0x90]));
    let events = frames.events();
    assert!(events.contains(&"written 1".to_owned()), "{events:?}");
    assert!(!events.contains(&"copied 1".to_owned()), "{events:?}");
    Ok(())
}

/// A frame the device declines, a buffer the host lends no device, and a
/// session with zero copy off are copied by the CPU, as before.
#[test]
fn what_no_device_writes_the_cpu_copies() -> Result<(), WireError> {
    // Declined by the device.
    let guest = opened()?;
    guest.decode(1, 0, &sample(vec![0x41, 0x91], Some(format(AVCC))))?;
    assert_eq!(guest.emitted()[0].samples, Some([0x41, 0x41, 0, 0x91]));
    assert!(
        guest
            .backend
            .frames
            .events()
            .contains(&"copied 1".to_owned())
    );

    // A buffer lent to no device.
    let guest = opened()?;
    guest
        .backend
        .frames
        .device_writes
        .store(true, Ordering::SeqCst);
    guest
        .state
        .buffers_kept_from_devices
        .store(true, Ordering::SeqCst);
    guest.decode(1, 0, &sample(vec![0x42, 0x92], Some(format(AVCC))))?;
    assert_eq!(guest.emitted()[0].samples, Some([0x42, 0x42, 0, 0x92]));
    let events = guest.backend.frames.events();
    assert!(events.contains(&"copied 1".to_owned()), "{events:?}");

    // Zero copy off.
    let guest = Guest::new();
    guest.session.set_zero_copy(false);
    assert!(!guest.session.zero_copy());
    guest
        .backend
        .frames
        .device_writes
        .store(true, Ordering::SeqCst);
    guest.open()?;
    assert_eq!(guest.start()?, NO_ERROR);
    guest.decode(1, 0, &sample(vec![0x43, 0x93], Some(format(AVCC))))?;
    assert_eq!(guest.emitted()[0].samples, Some([0x43, 0x43, 0, 0x93]));
    let events = guest.backend.frames.events();
    assert!(events.contains(&"copied 1".to_owned()), "{events:?}");
    assert!(
        !events.iter().any(|e| e.starts_with("written")),
        "{events:?}"
    );
    Ok(())
}

/// A buffer the guest made read-only for its clients is lent to no device
/// either: its frame is a malfunction, as with the CPU's copy, and nothing
/// writes it.
#[test]
fn a_read_only_buffer_is_never_lent_to_a_device() -> Result<(), WireError> {
    let guest = opened()?;
    guest.seen();
    guest
        .backend
        .frames
        .device_writes
        .store(true, Ordering::SeqCst);
    guest.state.read_only_buffers.store(true, Ordering::SeqCst);
    guest.decode(1, 0, &sample(vec![0x40, 0x90], Some(format(AVCC))))?;
    assert_eq!(
        guest.seen(),
        ["Dbuf 1", "Demt 1", "Dfub", "rels [1]", "reply dfrm"]
    );
    let emitted = guest.emitted();
    assert_eq!(
        (emitted[0].status, emitted[0].samples),
        (Status::VideoDecoderMalfunction.code(), None)
    );
    let events = guest.backend.frames.events();
    assert!(
        !events.iter().any(|e| e.starts_with("written")),
        "{events:?}"
    );
    Ok(())
}

/// A buffer whose memory ends before its layout's last row is given to no
/// device: core checks the layout against the memory first, and the frame
/// fails as the CPU's copy fails on it.
#[test]
fn a_layout_past_its_memory_is_never_given_to_a_device() -> Result<(), WireError> {
    let guest = opened()?;
    guest.seen();
    guest
        .backend
        .frames
        .device_writes
        .store(true, Ordering::SeqCst);
    guest.state.short_buffers.store(true, Ordering::SeqCst);
    guest.decode(1, 0, &sample(vec![0x40, 0x90], Some(format(AVCC))))?;
    let emitted = guest.emitted();
    assert_eq!(emitted[0].status, Status::VideoDecoderMalfunction.code());
    let events = guest.backend.frames.events();
    assert!(
        !events.iter().any(|e| e.starts_with("written")),
        "{events:?}"
    );
    Ok(())
}

/// A device write that fails fails its frame, with the backend's status, as
/// a read back's failure does.
#[test]
fn a_device_write_that_fails_fails_its_frame() -> Result<(), WireError> {
    let guest = opened()?;
    guest.seen();
    let frames = &guest.backend.frames;
    frames.device_writes.store(true, Ordering::SeqCst);
    frames.device_fails.store(true, Ordering::SeqCst);
    guest.decode(1, 0, &sample(vec![0x40, 0x90], Some(format(AVCC))))?;
    assert_eq!(
        guest.seen(),
        ["Dbuf 1", "Demt 1", "Dfub", "rels [1]", "reply dfrm"]
    );
    let emitted = guest.emitted();
    assert_eq!(
        (emitted[0].status, emitted[0].samples),
        (Status::VideoDecoderMalfunction.code(), None)
    );
    Ok(())
}

/// A device write past its deadline fails its frame as a malfunction,
/// with no image, and its buffer stays held, with no `rels`, while the
/// device may still write it: the guest's pages are neither reused nor
/// unmapped. Once the backend lets the hold go, the next sweep sends the
/// buffer's `rels` (`crate::device_wait`).
#[test]
fn a_device_write_past_its_deadline_keeps_its_buffer_until_the_device_is_done()
-> Result<(), WireError> {
    for ahead in [false, true] {
        let guest = if ahead {
            opened_decoding_ahead()?
        } else {
            opened()?
        };
        guest.seen();
        let frames = &guest.backend.frames;
        frames.device_writes.store(true, Ordering::SeqCst);
        frames.device_stalls.store(true, Ordering::SeqCst);
        let flags = u32::from(ahead);
        guest.decode(1, flags, &sample(vec![0x40, 0x90], Some(format(AVCC))))?;
        let seen = guest.seen_once(4);
        if ahead {
            assert_eq!(seen, ["Dbuf 1", "reply dfrm", "Demt 1", "Dfub"]);
        } else {
            assert_eq!(seen, ["Dbuf 1", "Demt 1", "Dfub", "reply dfrm"]);
        }
        let emitted = guest.emitted();
        assert_eq!(
            (emitted[0].status, emitted[0].samples),
            (Status::VideoDecoderMalfunction.code(), None)
        );
        assert_eq!(
            guest.memory.held(),
            1,
            "the buffer the device writes is let go"
        );
        assert_eq!(frames.still_writing.lock().len(), 1);
        assert_eq!(
            guest.memory.wrote(1),
            0,
            "a write told of before the device is done"
        );
        // The device is done: the backend lets the hold go, which tells the
        // host of its write (`GuestHold`), and the next delivery's sweep
        // sends the buffer's `rels`. Here that is the settle's. Were it the
        // next frame's, it would race that frame's `Dbuf`, as on Apple's
        // host: the delivery sweeps on its own thread as the session's queue
        // handles the frame (devices.md, "Messages", its surfaces, and "The
        // decoder session").
        frames.still_writing.lock().clear();
        assert_eq!(guest.memory.wrote(1), 1, "the late write is told of once");
        frames.device_stalls.store(false, Ordering::SeqCst);
        guest.settle();
        assert_eq!(guest.seen(), ["rels [1]"]);
        assert_eq!(guest.memory.held(), 0);
        guest.decode(2, 0, &sample(vec![0x41, 0x90], None))?;
        assert_eq!(
            guest.seen_once(4),
            ["Dbuf 2", "Demt 2", "rels [2]", "reply dfrm"]
        );
        assert_eq!(guest.memory.held(), 0);
    }
    Ok(())
}

/// The run of frames over every path, one at a time and decoding ahead,
/// gives the guest the same messages, emissions and pixels whether the
/// device writes each frame or the CPU copies it.
#[test]
fn zero_copy_leaves_what_the_guest_sees_as_it_was() -> Result<(), WireError> {
    for ahead in [false, true] {
        let copied = run_of_frames_written(ahead, false)?;
        let written = run_of_frames_written(ahead, true)?;
        assert!(
            written.events.iter().any(|e| e.starts_with("written")),
            "{:?}",
            written.events
        );
        assert!(
            !written.events.iter().any(|e| e.starts_with("copied")),
            "{:?}",
            written.events
        );
        assert_eq!(written.seen, copied.seen, "decoding ahead: {ahead}");
        assert_eq!(written.emitted, copied.emitted, "decoding ahead: {ahead}");
        if ahead {
            assert!(
                decoded_during_a_copy(&written.events),
                "{:?}",
                written.events
            );
        }
    }
    Ok(())
}
