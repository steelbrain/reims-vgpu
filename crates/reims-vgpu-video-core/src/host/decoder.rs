//! Decoder sessions: what the host does for each `+dec` the guest's
//! VideoToolbox opens, following Apple's host decoder session
//! with a [`BackendDecoder`] where Apple has a `VTVideoDecoder`.
//!
//! Each session has a thread of its own, its serial queue: the device's
//! thread hands it each message and returns at once, and the session's
//! thread handles them in order. Apple's decoder makes its output buffers
//! in the guest: for each frame it asks for a pixel buffer (`Dbuf`) and
//! waits for the guest's reply, which comes back on the device's thread
//! through the session's [`ReplyClerk`]; then it decodes into the buffer's
//! surface and hands the frame back (`Demt`). Here the backend decodes into
//! its own surfaces and the session copies each frame into the guest's
//! buffer between the two.
//!
//! For an asynchronous frame of a backend that decodes on a device, the
//! copy runs on a thread of its own, and the session's thread decodes the
//! next `dfrm` meanwhile, if the guest has sent it; what the guest sees
//! stays as it was (ours).

mod avc_record;
mod frame_times;
mod lends;
mod rbsp;

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};

use crate::backend::{
    BackendDecoder, BackendError, Codec, CompressedFrame, DecodedFrame, DecoderBackend,
    DecoderConfig, DeviceWrite, FrameDecoded, PixelFormat,
};
use crate::device_memory::GuestTarget;
use crate::device_wait::GuestHold;
use crate::fourcc::FourCc;
use crate::host::HostShared;
use crate::host::codec_lists::decoder_id;
use crate::host::guest_bounds::{Held, MAX_FRAME_MESSAGE_BYTES};
use crate::host::mapped_surfaces::{SurfaceHold, ending_together};
use crate::host::names::{self, DecoderMessage};
use crate::host::refused::{RefusedKind, RefusedSession};
use crate::host::reply_clerk::{CODEC_OPERATION_REPLY_TIMEOUT, DEFAULT_REPLY_TIMEOUT, Reply};
use crate::host::session::{
    self, Created, SessionEntry, SessionKind, SessionLink, SessionQueue, create_reply,
    property_key, reply,
};
use crate::runtime::{self, Runtime};
use crate::status::{NO_ERROR, OsStatus, Status};
use crate::surfaces::SurfaceLayout;
use crate::sync::Mutex;
use crate::wire::WireError;
use crate::wire::image_description::{
    AV1_CONFIGURATION, AVC_CONFIGURATION, HEVC_CONFIGURATION, ImageDescription, VP9_CONFIGURATION,
};
use crate::wire::message::{
    GuestUuid, MediaTime, Message, MessageFlags, MessageHeader, MessageWriter, read_header,
};
use crate::wire::property_list::{dictionary, integer};
use crate::wire::sample_buffer::{SampleBuffer, SerializedFormatDescription};
use frame_times::{CopyTimes, FrameTimes};
use lends::Lends;

/// A decoder session's thread (ours) and log name.
const KIND: SessionKind = SessionKind {
    thread: "reims-vgpu-video.decoder",
    name: "a decoder session",
};
/// The name of the thread a frame is copied on while the session decodes
/// ahead (ours).
const COPIER_THREAD_NAME: &str = "reims-vgpu-video.decoder-copy";

/// `kVTDecodeFrame_EnableAsynchronousDecompression` (`VTErrors.h`).
const DECODE_ASYNCHRONOUSLY: u32 = 1 << 0;
/// `kVTDecodeInfo_Asynchronous`.
const INFO_ASYNCHRONOUS: u32 = 1 << 0;
/// `kVTDecodeInfo_FrameDropped`.
const INFO_FRAME_DROPPED: u32 = 1 << 1;
/// `kVTDecodeInfo_SkippedLeadingFrameDropped`.
const INFO_SKIPPED_LEADING_FRAME_DROPPED: u32 = 1 << 3;

/// `kVTDecompressionPropertyKey_Paravirtualized`: the host's own, which a
/// guest may not set.
const PARAVIRTUALIZED: &str = "Paravirtualized";
/// `kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder`.
const USING_HARDWARE: &str = "UsingHardwareAcceleratedVideoDecoder";
/// `kVTDecompressionPropertyKey_RealTime`.
const REAL_TIME: &str = "RealTime";
/// `kVTDecompressionPropertyKey_MaximizePowerEfficiency`.
const MAXIMIZE_POWER_EFFICIENCY: &str = "MaximizePowerEfficiency";
/// `kVTDecompressionPropertyKey_AV1FilmGrainMode`, which Apple's AV1
/// decoder lists, and which the guest sets only to
/// [`FILM_GRAIN_IMMEDIATE`].
const AV1_FILM_GRAIN_MODE: &str = "AV1FilmGrainMode";
/// `kVTAV1FilmGrainMode_Immediate`: film grain applied to the frame the
/// decoder outputs, as every backend applies it.
const FILM_GRAIN_IMMEDIATE: &str = "Immediate";
/// `kVTDecompressionPropertyKey_SuggestedQualityOfServiceTiers`, which
/// Apple's VP9 decoder lists, read-only.
const SUGGESTED_QUALITY_OF_SERVICE_TIERS: &str = "SuggestedQualityOfServiceTiers";
/// `kVTDecompressionPropertyKey_OnlyTheseFrames`, which Apple's VP9
/// decoder reads as [`ALL_FRAMES`] and does not take.
const ONLY_THESE_FRAMES: &str = "OnlyTheseFrames";
/// `kVTDecompressionProperty_OnlyTheseFrames_AllFrames`.
const ALL_FRAMES: &str = "AllFrames";
/// The values of [`ONLY_THESE_FRAMES`] Apple's VP9 decoder suggests, one
/// tier each, in its order: `kVTDecompressionProperty_OnlyTheseFrames_`
/// `AllFrames`, `NonDroppableFrames`, `IFrames`, `KeyFrames`.
const QUALITY_OF_SERVICE_TIERS: [&str; 4] =
    [ALL_FRAMES, "NonDroppableFrames", "IFrames", "KeyFrames"];

/// Makes the session `+dec` asks for and its create reply. Its `,cod`
/// must be nonzero, and every parameter must read, or there is no
/// session and no reply. A codec the backend does not list, or a `,cid`
/// naming another decoder than this host's for it, makes a session
/// whose reply says -12906, as Apple's selection finds no decoder.
pub(crate) fn create(message: &Message<'_>, host: &Arc<HostShared>) -> Created<CopyEvent> {
    let codec_type = message
        .s32(names::CODEC)
        .map_err(|error| error.status())?
        .unwrap_or(0);
    if codec_type == 0 {
        return Err(Status::MalformedMessage);
    }
    message
        .u32(names::VERSION)
        .map_err(|error| error.status())?;
    let decoder_id_asked = message
        .property_list(names::CODEC_ID)
        .map_err(|error| error.status())?;
    message
        .dictionary(names::GUEST_INFO)
        .map_err(|error| error.status())?;
    let codec = Codec::from_fourcc(FourCc(codec_type.cast_unsigned())).filter(|&codec| {
        host.backend
            .capabilities()
            .iter()
            .any(|offered| offered.codec == codec)
            && decoder_id_asked
                .as_ref()
                .is_none_or(|asked| asked.as_string() == Some(decoder_id(codec)))
    });
    let Some(codec) = codec else {
        let (session, reply) = RefusedSession::create(message, RefusedKind::Decoder)
            .map_err(|error| error.status())?;
        return Ok((SessionEntry::Refused(session), reply));
    };
    let session = start(message.header().guest_uuid, codec, host)?;
    let reply = create_reply(message, NO_ERROR).map_err(|error| error.status())?;
    Ok((SessionEntry::Running(session), reply))
}

/// A copy thread's word to its session's thread
/// ([`Worker::copy_looking_ahead`]).
pub(crate) enum CopyEvent {
    /// The frame being copied on its own thread is whole on its device
    /// ([`DecodedFrame::wait`]): the session may decode the next.
    Waited,
    /// That frame's copy is over, or its thread ended.
    Copied,
}

/// A decoder session's job.
type Job = session::Job<CopyEvent>;

/// Starts a decoding session for `codec`, its thread waiting for its first
/// message.
fn start(
    guest_uuid: GuestUuid,
    codec: Codec,
    host: &Arc<HostShared>,
) -> Result<SessionQueue<CopyEvent>, Status> {
    let link = SessionLink::new(guest_uuid, host);
    let frames_queued = Arc::new(AtomicUsize::new(0));
    let (jobs, queue) = mpsc::channel();
    let worker = Worker {
        queue,
        wake: Some(jobs.clone()),
        deferred: VecDeque::new(),
        ahead: None,
        link: link.clone(),
        frames_queued: frames_queued.clone(),
        backend: host.backend.clone(),
        codec,
        decoder: None,
        start_answered: false,
        config: None,
        last_format: None,
        asked_size: None,
        lends: Lends::default(),
        releasing: Vec::new(),
        fragments: Vec::new(),
        fragments_held: Vec::new(),
        discarding: false,
        unemitted: Vec::new(),
        properties: HashMap::new(),
        abandoned: 0,
        times: FrameTimes::default(),
        zero_copy: Arc::new(ZeroCopy::new(host.zero_copy.load(Ordering::Relaxed))),
        runtime: host.runtime.clone(),
    };
    let counted = move |header: &MessageHeader| {
        // A whole `dfrm`, its last fragment.
        if DecoderMessage::from_fourcc(header.message_type) == Some(DecoderMessage::DecodeFrame)
            && !header.flags.contains(MessageFlags::MORE_FRAGMENTS)
        {
            frames_queued.fetch_add(1, Ordering::AcqRel);
        }
    };
    Ok(SessionQueue::start(link, host, &KIND, jobs, move || worker.run())?.telling(counted))
}

/// The reply a `dfrm` is owed, and when it goes.
struct FrameReply {
    request: MessageHeader,
    sent: bool,
    /// Whether it goes as soon as the first buffer is in hand (an
    /// asynchronous decode) rather than at the end.
    when_buffer_is_in_hand: bool,
    /// Its `,inf`: the info flags Apple's decoder answers its decode call
    /// with, which Apple's host replies with.
    info: u32,
}

/// A guest pixel buffer the session holds, from `Dbuf`'s reply until the
/// frame is emitted into it or it is let go.
struct GuestBuffer {
    uuid: GuestUuid,
    surface: Arc<SurfaceHold>,
}

/// What a whole `dfrm` carries.
struct FrameMessage {
    frame_id: i64,
    sample: SampleBuffer,
    flags: u32,
}

/// Reads a whole `dfrm`: the parameters it must have, and those that must
/// read where they are.
fn read_frame(whole: &[u8]) -> Result<FrameMessage, WireError> {
    let message = Message::parse(whole)?;
    let frame_id = message.s64(names::FRAME)?.unwrap_or(0);
    let serialization = message
        .sample_buffer(names::SAMPLE_BUFFER)?
        .ok_or(WireError::Malformed("no sample buffer"))?;
    let sample = SampleBuffer::parse(&serialization)?;
    let flags = message.u32(names::DECODE_FLAGS)?.unwrap_or(0);
    message.dictionary(names::FRAME_OPTIONS)?;
    log::debug!(
        "dfrm: frame {frame_id:#x}, decode flags {flags:#x}, {} bytes",
        whole.len()
    );
    Ok(FrameMessage {
        frame_id,
        sample,
        flags,
    })
}

/// A `dfrm` read whole, and its decode done or handed to the backend's
/// device.
struct Submitted {
    request: MessageHeader,
    frame_id: i64,
    flags: u32,
    /// The configuration its sample buffer changed to, the session's once
    /// the frame is finished.
    new_config: Option<DecoderConfig>,
    /// What the backend's decode gave, or `None` with no decoder.
    decoded: Option<Result<Vec<Box<dyn DecodedFrame>>, BackendError>>,
    /// Every frame the decode took ([`BackendDecoder::frames_decoded`]).
    frames_decoded: Vec<FrameDecoded>,
}

/// A `dfrm` taken from the queue while the frame before it was copied, and
/// decoded then; the rest waits for its turn ([`Worker::look_ahead`]).
struct Ahead {
    frame: Submitted,
    /// Held as the queued message holds them, until it is handled.
    surfaces: Vec<Arc<SurfaceHold>>,
    held: Held,
}

/// Wakes the session's thread when a frame's copy ends, however it ends.
struct WakeWhenCopied(Sender<Job>);

impl Drop for WakeWhenCopied {
    fn drop(&mut self) {
        // An Err is a session's thread that has ended.
        let _ = self.0.send(Job::Own(CopyEvent::Copied));
    }
}

/// The session's thread and what only it touches.
struct Worker {
    queue: Receiver<Job>,
    /// The session's own queue, for a frame's copy to say it is done; gone
    /// once the session is invalidated, so that the queue can close.
    wake: Option<Sender<Job>>,
    /// Messages taken from the queue while a frame was copied, in order,
    /// and not handled yet.
    deferred: VecDeque<Job>,
    /// The `dfrm` decoded while the frame before it was copied.
    ahead: Option<Ahead>,
    link: Arc<SessionLink>,
    /// The whole `dfrm`s (last fragments) the device has queued for the
    /// session's thread and it has not yet taken up. Ours: a guest waits
    /// for each `dfrm`'s reply before it sends the next, so one queued
    /// behind the frame in hand means the guest has stopped waiting for
    /// it ([`Worker::finish`]). A `dfrm` decoded ahead counts as queued
    /// until its turn.
    frames_queued: Arc<AtomicUsize>,
    backend: Arc<dyn DecoderBackend>,
    codec: Codec,
    decoder: Option<Box<dyn BackendDecoder>>,
    /// Whether a `dsta` has been answered, whatever its answer: Apple's
    /// AVD decoders start once, and answer every later `dsta` with -12909.
    start_answered: bool,
    /// What the decoder was opened with, or last reconfigured to.
    config: Option<DecoderConfig>,
    /// The last format description a sample buffer carried: the guest
    /// leaves it out of the next ones while it stays the same.
    last_format: Option<SerializedFormatDescription>,
    /// The size the guest's buffers were last asked for, by an AV1 or VP9
    /// frame ([`Worker::ask_for_the_frames_size`],
    /// [`Worker::ask_for_the_steps_size`]).
    asked_size: Option<(u32, u32)>,
    /// The buffers lent for an AV1 or VP9 decode's frames, held as Apple's
    /// AV1 and VP9 decoders hold them ([`Worker::emit_frames_decoded`]).
    lends: Lends<GuestBuffer>,
    /// The lent buffers let go, held until the guest's next message
    /// ([`Worker::let_go_of_released`]).
    releasing: Vec<GuestBuffer>,
    /// The fragments of a `dfrm` so far, and their shares of what the host
    /// session holds.
    fragments: Vec<u8>,
    fragments_held: Vec<Held>,
    /// Whether the `dfrm` in hand passed [`MAX_FRAME_MESSAGE_BYTES`]: its
    /// fragments are dropped up to its last.
    discarding: bool,
    /// The buffers lent and let go without a frame, for the next `Dfub`.
    unemitted: Vec<GuestUuid>,
    /// The properties the guest set.
    properties: HashMap<String, plist::Value>,
    /// How many frames the guest had stopped waiting for by the time they
    /// decoded, for the log ([`Worker::finish`]).
    abandoned: u64,
    /// Where the frames' time went, for the log at the end.
    times: FrameTimes,
    /// Whether frames are written into the guest's buffers by the backend's
    /// device, and why not where they are not.
    zero_copy: Arc<ZeroCopy>,
    /// Where the copy's thread comes from, and the clock the times are on.
    runtime: Arc<dyn Runtime>,
}

/// What a handler does when its message does not read: Apple's handlers
/// send no reply, and the guest's wait times out.
type Handled = Result<(), WireError>;

impl Worker {
    /// Handles the session's messages in order: a frame decoded ahead,
    /// then those taken while it was, then the queue's.
    fn run(mut self) {
        loop {
            if let Some(ahead) = self.ahead.take() {
                self.finish_ahead(ahead);
                continue;
            }
            let job = match self.deferred.pop_front() {
                Some(job) => job,
                None => {
                    let waiting = self.runtime.now();
                    let Ok(job) = self.queue.recv() else { break };
                    self.times
                        .add(|times| &mut times.idle, self.runtime.elapsed_since(waiting));
                    job
                }
            };
            match job {
                Job::Message {
                    bytes,
                    surfaces,
                    held,
                } => {
                    if !self.link.is_invalidated() {
                        self.handle(&bytes, &surfaces, held);
                    }
                    self.let_go_of_lent(surfaces);
                }
                Job::Invalidate(done) => {
                    if let Some(summary) = self.times.summary() {
                        log::info!("decoder session ends: {summary}");
                    }
                    self.times = FrameTimes::default();
                    if self.abandoned > 0 {
                        log::info!(
                            "decoder session ends: the guest had given up on {} of its frames",
                            self.abandoned
                        );
                    }
                    self.decoder = None;
                    // In one `rels`, where a session destroyed with buffers
                    // lent lets them go.
                    let held = (self.lends.clear(), std::mem::take(&mut self.releasing));
                    self.link.let_go(held);
                    // Nothing more is copied ahead, and the queue can close.
                    self.wake = None;
                    done.give(());
                }
                // A copy's own, taken while it ran.
                Job::Own(_) => {}
            }
        }
    }

    /// Lets go of the surfaces a message lent once it is handled, with a
    /// sweep, as the encoder and motion-estimation sessions do: their
    /// `rels` comes then, not with whichever thread's sweep comes next.
    fn let_go_of_lent(&self, surfaces: Vec<Arc<SurfaceHold>>) {
        if !surfaces.is_empty() {
            self.link.let_go(surfaces);
        }
    }

    fn handle(&mut self, bytes: &[u8], _surfaces: &[Arc<SurfaceHold>], held: Held) {
        self.let_go_of_released();
        let header = match read_header(bytes, 0) {
            Ok(header) => header,
            Err(error) => {
                log::warn!("decoder session message does not read: {error}");
                return;
            }
        };
        let Some(kind) = DecoderMessage::from_fourcc(header.message_type) else {
            log::warn!("unrecognized messageType '{}'", header.message_type);
            return;
        };
        let handled = match kind {
            DecoderMessage::DecodeFrame => self.decode_frame(bytes, header, held),
            _ => match Message::parse(bytes) {
                Ok(message) => self.handle_message(kind, &message),
                Err(error) => Err(error),
            },
        };
        if let Err(error) = handled {
            log::warn!("'{}' got no reply: {error}", header.message_type);
        }
    }

    fn handle_message(&mut self, kind: DecoderMessage, message: &Message<'_>) -> Handled {
        match kind {
            DecoderMessage::CopyProperty => self.copy_property(message),
            DecoderMessage::SetProperty => self.set_property(message),
            DecoderMessage::StartSession => self.start_session(message),
            DecoderMessage::SupportedProperties => self.supported_properties(message),
            DecoderMessage::StartTileSession => {
                // Tiles are image decoding's; this decoder has none, as
                // VideoToolbox's frame decoders answer.
                message
                    .image_description(names::FORMAT_DESCRIPTION)?
                    .ok_or(WireError::Malformed("no format description"))?;
                self.reply_status(message, Status::UnimplementedFunction.code())
            }
            DecoderMessage::DecodeTile => {
                let mut reply = reply(message);
                reply.s32(names::ERROR, Status::UnimplementedFunction.code());
                reply.u32(names::INFO_FLAGS, 0);
                self.send(reply.finish());
                Ok(())
            }
            DecoderMessage::FinishDelayedTiles | DecoderMessage::SerializableProperties => {
                self.reply_status(message, Status::UnimplementedFunction.code())
            }
            DecoderMessage::FinishDelayedFrames => self.finish_delayed_frames(message),
            DecoderMessage::CanAcceptFormatDescription => self.can_accept(message),
            DecoderMessage::Invalidate => {
                self.decoder = None;
                // The buffers held go once the reply has, with a sweep, so
                // that their `rels` follows it whichever thread's sweep
                // comes first, as the next delivery's would send it.
                let held = (self.lends.clear(), std::mem::take(&mut self.releasing));
                let replied = self.reply_status(message, NO_ERROR);
                self.link.let_go(held);
                replied
            }
            DecoderMessage::SetProperties => self.set_properties(message),
            // Read whole, fragments and all, by the caller.
            DecoderMessage::DecodeFrame => Ok(()),
        }
    }

    fn send(&self, message: Vec<u8>) {
        if let Err(status) = self.link.send(message) {
            log::debug!("decoder session send dropped: {status:?}");
        }
    }

    fn reply_status(&self, message: &Message<'_>, status: OsStatus) -> Handled {
        let mut reply = reply(message);
        reply.s32(names::ERROR, status);
        self.send(reply.finish());
        Ok(())
    }

    /// `dsta`: opens the backend decoder on the format description, asks
    /// the guest for buffers in its output format (`Dpba`) and replies
    /// with the status. The guest's pool flag, `,blu`, is read; the guest
    /// sends 0, so buffers come one by one through `Dbuf`.
    ///
    /// A decoder starts once. Apple's AVD decoders answer every `dsta` after
    /// the first with -12909 and no `Dpba`, whether the first was taken or
    /// refused and whatever the second describes, and a session the first
    /// started decodes on as before (AVD's H.264, HEVC, VP9 and AV1
    /// decoders, macOS 27.0.1, 2026-10-24). So does this one.
    fn start_session(&mut self, message: &Message<'_>) -> Handled {
        let description = message.image_description(names::FORMAT_DESCRIPTION)?;
        message.s32(names::GUEST_POOL)?;
        let description = ImageDescription::parse(
            description.ok_or(WireError::Malformed("no format description"))?,
        )?;
        if self.start_answered {
            log::warn!("a second dsta: refused, as Apple's decoders refuse it");
            return self.reply_status(message, Status::VideoDecoderBadData.code());
        }
        self.start_answered = true;
        let config = config_from_description(&description);
        let status = match config {
            Some(config) if Codec::from_fourcc(description.codec) == Some(self.codec) => {
                match session_refusal(&config).map_or_else(|| self.backend.open(&config), Err) {
                    Ok(decoder) => {
                        self.set_pixel_buffer_attributes(
                            (config.width, config.height),
                            decoder.output_format(),
                        )?;
                        self.decoder = Some(decoder);
                        self.config = Some(config);
                        self.last_format = None;
                        NO_ERROR
                    }
                    Err(error) => {
                        log::warn!("cannot open a decoder: {error}");
                        error.status()
                    }
                }
            }
            _ => Status::VideoDecoderUnsupportedDataFormat.code(),
        };
        self.reply_status(message, status)
    }

    /// `Dpba`: the attributes the guest's buffers are to have: the format
    /// the backend decodes into, the stream's size (or an AV1 frame's own,
    /// [`Worker::ask_for_the_frames_size`]), IOSurface backing,
    /// which the host needs to reach them, and the alignment and cache-mode
    /// keys Apple's AVD asks for, so the guest's pixel-buffer pool is laid
    /// out and cached as it is under Apple's host
    /// (the keys measured on macOS 27). The guest passes the dictionary
    /// straight to its decompression session, so the host copies each frame
    /// into whatever stride and plane offsets the guest then chooses, read
    /// back from the surface's own layout. `ExtendedPixels*`, which AVD
    /// adds only for small frames, is left out: the rule is not known, and
    /// AVD omitted it for the 1280x720 and larger streams measured.
    fn set_pixel_buffer_attributes(
        &self,
        (width, height): (u32, u32),
        format: PixelFormat,
    ) -> Handled {
        let attributes = dictionary([
            (
                "PixelFormatType",
                plist::Value::Array(vec![integer(i64::from(format.fourcc().0))]),
            ),
            ("Width", integer(i64::from(width))),
            ("Height", integer(i64::from(height))),
            (
                "IOSurfaceProperties",
                plist::Value::Dictionary(plist::Dictionary::new()),
            ),
            // AVD's constant alignment and cache-mode keys (macOS 27).
            ("BytesPerRowAlignment", integer(64)),
            ("PlaneAlignment", integer(256)),
            (
                "PreferRealTimeCacheModeIfEveryoneDoes",
                plist::Value::Boolean(true),
            ),
        ]);
        let mut message = MessageWriter::new(
            names::SET_PIXEL_BUFFER_ATTRIBUTES,
            MessageFlags::NONE,
            Some(self.link.guest_uuid),
            0,
        );
        message.property_list(
            names::PIXEL_BUFFER_ATTRIBUTES,
            Some(&plist::Value::Dictionary(attributes)),
        )?;
        self.send(message.finish());
        Ok(())
    }

    /// `dfrm`: a fragment flagged with more to come is kept; the last one
    /// completes the message, whose reply follows the last fragment's
    /// header. A message that passes [`MAX_FRAME_MESSAGE_BYTES`] is
    /// dropped, its fragments up to the last too, with no reply, as for one
    /// that does not read.
    fn decode_frame(&mut self, bytes: &[u8], last: MessageHeader, held: Held) -> Handled {
        let more = last.flags.contains(MessageFlags::MORE_FRAGMENTS);
        if !more {
            self.take_up_frame();
        }
        if !self.discarding
            && self.fragments.len().saturating_add(bytes.len()) > MAX_FRAME_MESSAGE_BYTES
        {
            self.fragments = Vec::new();
            self.fragments_held.clear();
            self.discarding = true;
        }
        if self.discarding {
            self.discarding = more;
            return Err(WireError::Malformed(
                "a frame message past the host's bound",
            ));
        }
        if more {
            self.fragments.extend_from_slice(bytes);
            self.fragments_held.push(held);
            return Ok(());
        }
        let whole = if self.fragments.is_empty() {
            bytes.to_vec()
        } else {
            let mut whole = std::mem::take(&mut self.fragments);
            whole.extend_from_slice(bytes);
            whole
        };
        // What was held of the fragments goes with them, once read.
        let _held = (held, std::mem::take(&mut self.fragments_held));
        let submitted = self.read_and_submit(&whole, last)?;
        self.finish(submitted);
        Ok(())
    }

    /// A whole `dfrm` taken up: it no longer counts as queued.
    fn take_up_frame(&self) {
        let _ = self
            .frames_queued
            .try_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
                queued.checked_sub(1)
            });
    }

    /// Reads a whole `dfrm` and hands its sample to the backend.
    fn read_and_submit(
        &mut self,
        whole: &[u8],
        request: MessageHeader,
    ) -> Result<Submitted, WireError> {
        let frame = read_frame(whole)?;
        Ok(self.submit(request, &frame))
    }

    /// Hands one frame to the backend's decoder, which decodes it or
    /// starts it on its device; nothing goes to the guest yet.
    fn submit(&mut self, request: MessageHeader, frame: &FrameMessage) -> Submitted {
        let FrameMessage {
            frame_id,
            ref sample,
            flags,
        } = *frame;
        let Some(mut decoder) = self.decoder.take() else {
            return Submitted {
                request,
                frame_id,
                flags,
                new_config: None,
                decoded: None,
                frames_decoded: Vec::new(),
            };
        };
        let new_config = self.reconfiguration(sample);
        let timing = sample.timings.first();
        let frame = CompressedFrame {
            frame_id,
            data: sample.data.as_deref().unwrap_or_default(),
            sample_sizes: &sample.sizes,
            presentation_time: timing.map_or(MediaTime::INVALID, |timing| timing.presentation),
            decode_time: timing.map_or(MediaTime::INVALID, |timing| timing.decode),
            duration: timing.map_or(MediaTime::INVALID, |timing| timing.duration),
            new_config: new_config.as_ref(),
        };
        let decoding = self.runtime.now();
        let decoded = decoder.decode(&frame);
        // Asked after every decode, so that each decode's are read once;
        // after one that failed, kept for VP9 alone, whose refused frames
        // Apple's decoder lends buffers for.
        let frames_decoded = decoder.frames_decoded();
        let frames_decoded = if decoded.is_ok() || self.codec == Codec::Vp9 {
            frames_decoded
        } else {
            Vec::new()
        };
        self.times.add(
            |times| &mut times.decode,
            self.runtime.elapsed_since(decoding),
        );
        self.times.frames += 1;
        self.decoder = Some(decoder);
        Submitted {
            request,
            frame_id,
            flags,
            new_config,
            decoded: Some(decoded),
            frames_decoded,
        }
    }

    /// Emits what a frame's decode gave, and replies. Without the
    /// asynchronous flag the reply comes after the frame is emitted and its
    /// surface let go; with it, as soon as the guest has lent the buffer,
    /// and the reply and each emission carry `kVTDecodeInfo_Asynchronous`.
    /// Apple's AVD answers so for every codec, size and frame, decoded or
    /// failed, but a RASL picture it drops, whose reply carries 0 (measured
    /// on macOS 27.0.1, 26A434, 2026-10-24).
    fn finish(&mut self, submitted: Submitted) {
        let Submitted {
            request,
            frame_id,
            flags,
            new_config,
            decoded,
            frames_decoded,
        } = submitted;
        let asynchronous = flags & DECODE_ASYNCHRONOUSLY != 0;
        let info = if asynchronous { INFO_ASYNCHRONOUS } else { 0 };
        let mut reply = FrameReply {
            request,
            sent: false,
            when_buffer_is_in_hand: asynchronous,
            info,
        };
        let Some(decoded) = decoded else {
            // No decode session: none started, the start refused, or the
            // decoder invalidated. Apple's AVD decoders answer each such
            // frame with their own 311 and no info flags, either decode
            // flag, and emit nothing.
            reply.info = 0;
            self.send_frame_reply(&mut reply, Status::DecoderNotStarted.code());
            return;
        };
        if new_config.is_some() && decoded.is_ok() {
            self.config = new_config;
        }
        if self.guest_gave_up(frame_id) {
            // The frame was still decoded, so that the frames after it have
            // their references; no buffer is asked for, and those held go
            // as they would have.
            drop(decoded);
            self.follow_frames_decoded(&frames_decoded);
            self.send_frame_reply(&mut reply, NO_ERROR);
            self.clean_up_after_decode();
            return;
        }
        match decoded {
            Ok(frames) if self.lends_for(&frames_decoded, &frames) => {
                self.emit_frames_decoded(frame_id, frames_decoded, frames, info, &mut reply, None);
            }
            Ok(frames) => {
                let mut emitted_this_one = false;
                for decoded in frames {
                    let this_one = decoded.frame_id() == frame_id;
                    emitted_this_one |= this_one;
                    // A frame sent with kVTDecodeFrame_DoNotOutputFrame is
                    // emitted too: Apple's AVD emits it as any other, with
                    // its image (measured on macOS 27.0.1, 26A434,
                    // 2026-10-24).
                    self.emit(decoded, info, &mut reply);
                }
                if !emitted_this_one {
                    // Frames come out in decode order: a sample with no
                    // picture is a frame dropped.
                    self.buffer_in_hand(&mut reply);
                    self.emit_without_image(frame_id, NO_ERROR, info | INFO_FRAME_DROPPED);
                }
            }
            Err(error @ BackendError::SkippedLeadingPicture(_)) => {
                // Apple's HEVC decoder drops a RASL picture it skips with
                // its own status and these flags, in a failed frame's
                // order, and with no asynchronous flag: the reply comes
                // last, as for a synchronous decode, with no info flags.
                log::debug!("frame {frame_id}: {error}");
                reply.when_buffer_is_in_hand = false;
                reply.info = 0;
                let info = INFO_FRAME_DROPPED | INFO_SKIPPED_LEADING_FRAME_DROPPED;
                self.emit_failed(frame_id, error.status(), info, &mut reply);
            }
            Err(error) if !frames_decoded.is_empty() => {
                // A VP9 sample refused: Apple's VP9 decoder lends a buffer
                // for each of its frames, emits the sample with the status
                // and no image at its last, and replies after, with the
                // asynchronous flag where it was asked for (measured on
                // Apple's host session on the M3 Max, macOS 27.0.1,
                // 2026-10-25, with libvpx's dynamic resize and its spatial
                // layers, synchronously and not).
                log::warn!("frame {frame_id} does not decode: {error}");
                reply.when_buffer_is_in_hand = false;
                let status = error.status();
                self.emit_frames_decoded(
                    frame_id,
                    frames_decoded,
                    Vec::new(),
                    info,
                    &mut reply,
                    Some(status),
                );
            }
            Err(error) => {
                log::warn!("frame {frame_id} does not decode: {error}");
                if matches!(self.codec, Codec::H264 | Codec::Hevc) {
                    self.emit_failed(frame_id, error.status(), info, &mut reply);
                } else {
                    self.buffer_in_hand(&mut reply);
                    self.emit_without_image(frame_id, error.status(), info);
                }
            }
        }
        self.send_frame_reply(&mut reply, NO_ERROR);
    }

    /// A `dfrm` decoded ahead, in its turn: taken up now, as it would have
    /// been from the queue, and finished. Its message's surfaces and its
    /// share of what the host session holds go once it is handled, as a
    /// queued message's do.
    fn finish_ahead(&mut self, ahead: Ahead) {
        let Ahead {
            frame,
            surfaces,
            held,
        } = ahead;
        if !self.link.is_invalidated() {
            self.let_go_of_released();
            self.take_up_frame();
            self.finish(frame);
        }
        drop(held);
        self.let_go_of_lent(surfaces);
    }

    /// The `dfrm` reply, once: `,err` and `,inf`.
    fn send_frame_reply(&self, reply: &mut FrameReply, status: OsStatus) {
        if reply.sent {
            return;
        }
        reply.sent = true;
        let mut message = MessageWriter::reply_to(&reply.request, MessageFlags::REPLY_TO_GUEST);
        message.s32(names::ERROR, status);
        message.u32(names::INFO_FLAGS, reply.info);
        self.send(message.finish());
    }

    /// An asynchronous decode's reply goes once its buffer is in hand.
    fn buffer_in_hand(&self, reply: &mut FrameReply) {
        if reply.when_buffer_is_in_hand {
            self.send_frame_reply(reply, NO_ERROR);
        }
    }

    /// The configuration a sample buffer's format description gives when
    /// it is not the one the decoder has: a new `avcC`, a new size. One
    /// the sample buffer leaves out is the last one it carried.
    fn reconfiguration(&mut self, sample: &SampleBuffer) -> Option<DecoderConfig> {
        if let Some(format) = &sample.format_description {
            self.last_format = Some(format.clone());
        }
        let config = config_from_format(self.last_format.as_ref()?)?;
        (Some(&config) != self.config.as_ref()).then_some(config)
    }

    /// `dfin`: emits every frame the backend still holds, then replies
    /// -12782, kCMBaseObjectError_UnimplementedFunction, as Apple's host
    /// answers for AVD's decoders, started or not: their class has no
    /// FinishDelayedFrames, as they emit each frame for its own `dfrm`
    /// (AVD's H.264, HEVC, VP9 and AV1 decoders, macOS 27.0.1, 2026-10-24).
    /// The guest hands the status to VideoToolbox, so it sees what an arm64
    /// guest sees on Apple's host. A backend's frames still go out before
    /// the reply, as before.
    fn finish_delayed_frames(&mut self, message: &Message<'_>) -> Handled {
        match self.decoder.as_mut().map(|decoder| decoder.flush()) {
            None => {}
            Some(Ok(frames)) => {
                // The reply below is this message's own.
                let mut no_frame_reply = FrameReply {
                    request: message.header(),
                    sent: true,
                    when_buffer_is_in_hand: false,
                    info: 0,
                };
                for frame in frames {
                    self.emit(frame, 0, &mut no_frame_reply);
                }
            }
            Some(Err(error)) => log::warn!("the decoder's held frames are lost: {error}"),
        }
        self.reply_status(message, Status::UnimplementedFunction.code())
    }

    /// Emits one decoded frame: asks the guest for a buffer, copies the
    /// frame into its surface, hands it back with `Demt`, and cleans up.
    /// An asynchronous frame of a decoder that decodes while copying is
    /// copied on a thread of its own, while this one decodes ahead.
    fn emit(&mut self, frame: Box<dyn DecodedFrame>, info: u32, reply: &mut FrameReply) {
        let frame_id = frame.frame_id();
        self.ask_for_the_frames_size(frame.as_ref());
        let buffer = match self.create_pixel_buffer(frame_id) {
            Ok(buffer) => buffer,
            Err(status) => {
                log::warn!("no guest buffer for frame {frame_id}: {status}");
                self.buffer_in_hand(reply);
                if self.guest_gave_up(frame_id) {
                    // Given up on while the `Dbuf` waited: no `Demt` either.
                    self.clean_up_after_decode();
                } else {
                    self.emit_without_image(frame_id, status, info);
                }
                return;
            }
        };
        self.buffer_in_hand(reply);
        let ahead = reply.when_buffer_is_in_hand;
        let (frame, copied) = self.copy_to_guest(frame, &buffer.surface, ahead);
        match copied {
            Ok(()) => {
                self.emit_decoded_frame(frame_id, NO_ERROR, info, Some(buffer.uuid));
            }
            Err(status) => {
                log::warn!("frame {frame_id} cannot be copied to the guest: {status}");
                self.unemitted.push(buffer.uuid);
                self.emit_decoded_frame(frame_id, status, info, None);
            }
        }
        self.clean_up_releasing(Some(buffer));
        // The frame goes last, as it did before it was copied apart.
        drop(frame);
    }

    /// Copies `frame` into the guest's `surface`, timed: on a thread of its
    /// own while this one decodes ahead, where the frame is `ahead` (an
    /// asynchronous frame whose reply has gone) and the decoder decodes
    /// while copying ([`Worker::copy_looking_ahead`]), and here otherwise.
    /// The frame comes back, to go where the caller lets it go.
    fn copy_to_guest(
        &mut self,
        frame: Box<dyn DecodedFrame>,
        surface: &Arc<SurfaceHold>,
        ahead: bool,
    ) -> (Option<Box<dyn DecodedFrame>>, Result<(), OsStatus>) {
        let copying = self.runtime.now();
        let ahead = ahead
            && self
                .decoder
                .as_ref()
                .is_some_and(|decoder| decoder.decodes_while_copying());
        let (frame, copied) = if ahead {
            self.copy_looking_ahead(frame, surface)
        } else {
            let mut parts = CopyTimes::default();
            let copied = copy_frame(
                frame.as_ref(),
                surface,
                &mut parts,
                &self.zero_copy,
                &*self.runtime,
            );
            self.times.copies.add(parts);
            (Some(frame), copied)
        };
        self.times
            .add(|times| &mut times.copy, self.runtime.elapsed_since(copying));
        (frame, copied)
    }

    /// Copies `frame` into `surface` on a thread of its own and, once the
    /// frame is whole on its device, takes the next `dfrm` from the queue
    /// and decodes it, if it comes before the copy is over
    /// ([`Worker::look_ahead`]). Without a thread, the copy runs here.
    ///
    /// The copy's thread is the runtime's. It borrows the frame through a
    /// slot, so that the frame stays here if the thread does not start. It
    /// lets go of the frame and of the guest's surface before it answers,
    /// and answers before it wakes the session's thread, so that once the
    /// wake comes the answer is there and the copy holds nothing. A copy
    /// that ends with no answer, which only a runtime that lets a panic
    /// unwind allows, is done again here.
    ///
    /// The wait for the copy has no bound of its own: the copy is the
    /// backend's wait for its device, bounded by its deadline
    /// ([`crate::device_wait`]), and its read back and the CPU's copy, which
    /// the session's thread would make itself one frame at a time. A
    /// backend call that never returns holds this thread as it would there,
    /// and the session's end leaves it running, detached
    /// ([`crate::host::session_thread`]).
    fn copy_looking_ahead(
        &mut self,
        frame: Box<dyn DecodedFrame>,
        surface: &Arc<SurfaceHold>,
    ) -> (Option<Box<dyn DecodedFrame>>, Result<(), OsStatus>) {
        let slot = Arc::new(Mutex::new(Some(frame)));
        let copied = self.wake.clone().and_then(|wake| {
            let (answer, answered) = runtime::answer();
            let target = surface.clone();
            let zero_copy = self.zero_copy.clone();
            let clock = self.runtime.clone();
            let lent = slot.clone();
            let started = self.runtime.spawn(
                COPIER_THREAD_NAME,
                Box::new(move || {
                    // In this order, so that an unwind drops what the copy
                    // held, then the answer, then wakes the session's
                    // thread, as the end below does.
                    let wake = WakeWhenCopied(wake);
                    let answer = answer;
                    let lent = lent;
                    let target = target;
                    let copy = {
                        let frame = lent.lock();
                        match frame.as_deref() {
                            Some(frame) => copy_ahead(frame, &target, &wake, &zero_copy, &*clock),
                            // Never so: the slot is the session's until the
                            // answer.
                            None => (
                                Err(Status::VideoDecoderMalfunction.code()),
                                CopyTimes::default(),
                            ),
                        }
                    };
                    // What the copy held of the frame and of the guest's
                    // surface goes before the answer: once the session's
                    // thread has the answer, the copy holds none of it.
                    drop(lent);
                    drop(target);
                    answer.give(copy);
                    drop(wake);
                }),
            );
            if let Err(error) = started {
                log::warn!("cannot start a decoder session's copy thread: {error}");
                return None;
            }
            self.look_ahead_until_copied();
            match answered.wait_until(&*self.runtime, runtime::MonoTime::END) {
                Ok((copied, parts)) => {
                    self.times.copies.add(parts);
                    Some(copied)
                }
                Err(_) => {
                    log::error!("a decoder session's copy thread ended without its copy");
                    None
                }
            }
        });
        let frame = slot.lock().take();
        let copied = match (copied, frame.as_deref()) {
            (Some(copied), _) => copied,
            (None, Some(frame)) => {
                let mut parts = CopyTimes::default();
                let copied =
                    copy_frame(frame, surface, &mut parts, &self.zero_copy, &*self.runtime);
                self.times.copies.add(parts);
                copied
            }
            (None, None) => Err(Status::VideoDecoderMalfunction.code()),
        };
        (frame, copied)
    }

    /// Takes the session's messages while a frame is copied: each is kept
    /// for its turn, and once the frame is whole on its device, a whole
    /// `dfrm` that comes first is decoded at once ([`Worker::look_ahead`]).
    fn look_ahead_until_copied(&mut self) {
        let mut waited = false;
        while let Ok(job) = self.queue.recv() {
            match job {
                Job::Own(CopyEvent::Copied) => return,
                Job::Own(CopyEvent::Waited) => {
                    waited = true;
                    if let Some(job) = self.deferred.pop_front() {
                        // Only what came first may go ahead.
                        let rest = std::mem::take(&mut self.deferred);
                        self.look_ahead(job);
                        self.deferred.extend(rest);
                    }
                }
                job if waited => self.look_ahead(job),
                job => self.deferred.push_back(job),
            }
        }
    }

    /// One message taken while a frame is copied. A whole `dfrm` in one
    /// message that reads, whose sample buffer leaves the decoder's
    /// configuration as it is, with nothing before it still to handle, is
    /// handed to the backend now; the rest of its handling, and everything
    /// else, waits for its turn. Its reply, buffer and emission go in that
    /// turn, so the guest sees what it saw without this. A reconfiguration
    /// waits, as the backend may reopen its session for it.
    fn look_ahead(&mut self, job: Job) {
        let frame = match &job {
            Job::Message { bytes, .. }
                if self.ahead.is_none()
                    && self.deferred.is_empty()
                    && self.fragments.is_empty()
                    && !self.discarding
                    && bytes.len() <= MAX_FRAME_MESSAGE_BYTES
                    && !self.link.is_invalidated() =>
            {
                self.whole_frame_keeping_its_configuration(bytes)
            }
            _ => None,
        };
        match (frame, job) {
            (Some((request, frame)), Job::Message { surfaces, held, .. }) => {
                let frame = self.submit(request, &frame);
                self.times.ahead += 1;
                self.ahead = Some(Ahead {
                    frame,
                    surfaces,
                    held,
                });
            }
            (_, job) => self.deferred.push_back(job),
        }
    }

    /// `bytes` read, if they are a whole `dfrm` in one message that reads
    /// and whose sample buffer leaves the decoder's configuration as it is.
    fn whole_frame_keeping_its_configuration(
        &self,
        bytes: &[u8],
    ) -> Option<(MessageHeader, FrameMessage)> {
        let header = read_header(bytes, 0).ok()?;
        if DecoderMessage::from_fourcc(header.message_type) != Some(DecoderMessage::DecodeFrame)
            || header.flags.contains(MessageFlags::MORE_FRAGMENTS)
        {
            return None;
        }
        let frame = read_frame(bytes).ok()?;
        // As [`Worker::reconfiguration`] finds a new configuration.
        let format = (frame.sample.format_description.as_ref()).or(self.last_format.as_ref());
        let keeps = format
            .and_then(config_from_format)
            .is_none_or(|config| Some(&config) == self.config.as_ref());
        keeps.then_some((header, frame))
    }

    /// Whether the guest has stopped waiting for `frame_id`, the frame in
    /// hand: a later `dfrm` is already queued behind it
    /// ([`Worker::frames_queued`]). Ours: the guest's VideoToolbox
    /// has emitted such a frame as failed already, and discards a `Demt`
    /// for it ("Unrecognised VTVideoDecoderFrame token"), so the host asks
    /// no buffer for it and emits nothing, rather than wait 1.1 s and 10 s
    /// on a guest that has moved on. Apple's host never meets the guest
    /// this was made for: an x86_64 one without VtWaitPatch, whose
    /// synchronous `dfrm` waits end at once. With the patch the guest waits
    /// 10 s, and this is left for a frame slower than that.
    fn guest_gave_up(&mut self, frame_id: i64) -> bool {
        if self.frames_queued.load(Ordering::Acquire) == 0 {
            return false;
        }
        self.abandoned += 1;
        if self.abandoned == 1 {
            log::info!(
                "the guest stopped waiting for frame {frame_id:#x} before it was emitted: \
                 frames it has given up on are decoded and not emitted"
            );
        }
        log::debug!("frame {frame_id:#x}: given up on by the guest, not emitted");
        true
    }

    /// Emits a frame that did not decode as Apple's AVD emits one for H.264
    /// and HEVC: it asks for a buffer (`Dbuf`) all the same, emits the frame
    /// with the status and no buffer (`Demt`), then gives the buffer back
    /// (`Dfub`) and sweeps, so that a `rels` follows (measured on macOS
    /// 27.0.1, 26A434, 2026-10-24, on truncated, garbage and interlaced
    /// frames). A buffer the guest does not give is answered as
    /// [`Worker::emit`] answers it.
    fn emit_failed(&mut self, frame_id: i64, status: OsStatus, info: u32, reply: &mut FrameReply) {
        let buffer = match self.create_pixel_buffer(frame_id) {
            Ok(buffer) => buffer,
            Err(buffer_status) => {
                log::warn!("no guest buffer for frame {frame_id}: {buffer_status}");
                self.buffer_in_hand(reply);
                if self.guest_gave_up(frame_id) {
                    self.clean_up_after_decode();
                } else {
                    self.emit_without_image(frame_id, status, info);
                }
                return;
            }
        };
        self.buffer_in_hand(reply);
        self.emit_decoded_frame(frame_id, status, info, None);
        self.unemitted.push(buffer.uuid);
        self.clean_up_releasing(Some(buffer));
    }

    fn emit_without_image(&mut self, frame_id: i64, status: OsStatus, info: u32) {
        self.emit_decoded_frame(frame_id, status, info, None);
        self.clean_up_after_decode();
    }

    /// `Dpba` again, at an AV1 or VP9 frame's own size, before its `Dbuf`,
    /// where the size is not the last asked for: for a backend that says
    /// nothing of the frames it decoded ([`BackendDecoder::frames_decoded`]),
    /// as each frame is emitted. The first frame asks nothing. Apple's AVD
    /// asks so for AV1, as it decodes ([`Worker::ask_for_the_steps_size`]).
    /// Its VP9 decoder emits a frame whose size is not the description's
    /// with -12909 and no image (measured on macOS 27, 26A434,
    /// 2026-10-24); a backend here that decodes such a frame right shows
    /// it, at its size, by AV1's way, and one that does not refuses it
    /// itself.
    fn ask_for_the_frames_size(&mut self, frame: &dyn DecodedFrame) {
        if !matches!(self.codec, Codec::Av1 | Codec::Vp9) {
            return;
        }
        let size = (frame.width(), frame.height());
        let last = self.asked_size.replace(size);
        if last.is_none_or(|last| last == size) {
            return;
        }
        let Some(format) = self.decoder.as_ref().map(|decoder| decoder.output_format()) else {
            return;
        };
        if let Err(error) = self.set_pixel_buffer_attributes(size, format) {
            log::warn!("no Dpba for a frame of {}x{}: {error}", size.0, size.1);
        }
    }

    /// Whether a decode's frames are emitted step by step
    /// ([`Worker::emit_frames_decoded`]): where the backend said which
    /// frames it took, and they show the frames it returned.
    fn lends_for(&self, steps: &[FrameDecoded], frames: &[Box<dyn DecodedFrame>]) -> bool {
        if steps.is_empty() {
            return false;
        }
        let shown = steps.iter().filter(|step| step.output).count();
        if shown != frames.len() {
            log::warn!(
                "the backend decoded {} frames, {shown} of them shown, and returned {}",
                steps.len(),
                frames.len()
            );
            return false;
        }
        true
    }

    /// Emits an AV1 or VP9 decode's frames as Apple's AV1 and VP9 decoders
    /// do, one step for each frame the temporal unit or the sample carries,
    /// in order (measured on Apple's host, macOS 27.0.1, 26A434, 2026-10-24,
    /// and for VP9 2026-10-25):
    /// - `Dpba` where the frame's size is not the last asked for, then a
    ///   `Dbuf`, for every frame decoded, hidden ones and lower spatial
    ///   layers' too, and for every frame shown again;
    /// - for the frame shown, its `Demt`: in the buffer it was decoded
    ///   into, which a frame shown again is too, and which holds a hidden
    ///   frame's picture only from then on, as nothing of it reaches the
    ///   guest before;
    /// - the clean-up after each step, its `Dfub` and sweep, then the
    ///   buffers due at the next step's start let go ([`lends`]).
    ///
    /// A VP9 sample `refusal` refused has a step for each of its frames,
    /// none of them shown, and its `Demt` with that status and no image at
    /// its last step, before that step's clean-up; no `Dpba` is asked for a
    /// frame refused, as Apple's VP9 decoder asks for none.
    ///
    /// Synchronously each step is over before the next is asked for.
    /// Asynchronously, as AVD does, every step's `Dpba` and `Dbuf` come
    /// first, then the reply, then the rest of each step in turn: AVD asks
    /// for a unit's buffers before its hardware decodes the unit's frames
    /// (measured with frames sent 33 ms apart). Where the session would
    /// then hold more than [`lends::MAX_LENDS`], the steps asked for are
    /// seen through first.
    fn emit_frames_decoded(
        &mut self,
        frame_id: i64,
        steps: Vec<FrameDecoded>,
        frames: Vec<Box<dyn DecodedFrame>>,
        info: u32,
        reply: &mut FrameReply,
        refusal: Option<OsStatus>,
    ) {
        let ahead = reply.when_buffer_is_in_hand;
        let last = steps.len().saturating_sub(1);
        let mut frames = frames.into_iter();
        let mut shown = false;
        // The steps asked for and not over: each step, its index and
        // number, and its buffer's status.
        let mut asked = VecDeque::new();
        let mut see_through =
            |worker: &mut Self,
             reply: &FrameReply,
             (step, index, at, status): (&FrameDecoded, usize, u64, OsStatus)| {
                let frame = step.output.then(|| frames.next()).flatten();
                shown |= frame.is_some();
                let refused = refusal.filter(|_| index == last);
                worker.finish_step(frame_id, at, step, status, frame, info, reply, refused);
            };
        for (index, step) in steps.iter().enumerate() {
            // What the session holds, those let go but not yet swept too.
            while ahead && self.lends.len() + self.releasing.len() >= lends::MAX_LENDS {
                let Some(oldest) = asked.pop_front() else {
                    break;
                };
                see_through(self, reply, oldest);
            }
            let at = self.lends.begin();
            if !step.refused {
                self.ask_for_the_steps_size(step);
            }
            let lent = self.create_pixel_buffer(frame_id);
            if index == last {
                self.buffer_in_hand(reply);
            }
            let buffer_status = match lent {
                Ok(buffer) => {
                    if let Some(early) = self.lends.lend(at, step, buffer) {
                        log::warn!("a session holds too many buffers: one is let go early");
                        if !early.emitted {
                            self.unemitted.push(early.buffer.uuid);
                        }
                        self.releasing.push(early.buffer);
                    }
                    NO_ERROR
                }
                Err(status) => {
                    log::warn!("no guest buffer for frame {frame_id}: {status}");
                    status
                }
            };
            asked.push_back((step, index, at, buffer_status));
            if !ahead && let Some(step) = asked.pop_front() {
                see_through(self, reply, step);
            }
        }
        for step in asked {
            see_through(self, reply, step);
        }
        if !shown && refusal.is_none() {
            // A unit that shows nothing is a frame dropped.
            self.buffer_in_hand(reply);
            self.emit_without_image(frame_id, NO_ERROR, info | INFO_FRAME_DROPPED);
        }
    }

    /// The rest of step `at` once its buffer is asked for: the frames its
    /// slots keep followed, its `frame`'s `Demt` where it shows one, or the
    /// sample's `Demt` with the status it was `refused` with and no image,
    /// its clean-up ([`Worker::clean_up_after_step`]), and the buffers due
    /// at the next step let go.
    #[allow(clippy::too_many_arguments)]
    fn finish_step(
        &mut self,
        frame_id: i64,
        at: u64,
        step: &FrameDecoded,
        buffer_status: OsStatus,
        frame: Option<Box<dyn DecodedFrame>>,
        info: u32,
        reply: &FrameReply,
        refused: Option<OsStatus>,
    ) {
        self.lends.follow(at, step);
        let frame =
            frame.and_then(|frame| self.emit_step(frame, at, step, buffer_status, info, reply));
        if let Some(status) = refused
            && !self.guest_gave_up(frame_id)
        {
            self.emit_decoded_frame(frame_id, status, info, None);
        }
        self.clean_up_after_step();
        // The frame goes once its step is over, as in [`Worker::emit`].
        drop(frame);
        self.release_lends(at);
    }

    /// The `Demt` of step `at`'s frame, copied first into its buffer
    /// ([`Lends::target`]) unless it is there already. With no buffer, the
    /// frame is emitted with `buffer_status` and no image, as
    /// [`Worker::emit`] emits one the guest gave no buffer for. The frame
    /// comes back, to go after the step's clean-up, unless its copy's
    /// thread kept it.
    fn emit_step(
        &mut self,
        frame: Box<dyn DecodedFrame>,
        at: u64,
        step: &FrameDecoded,
        buffer_status: OsStatus,
        info: u32,
        reply: &FrameReply,
    ) -> Option<Box<dyn DecodedFrame>> {
        let frame_id = frame.frame_id();
        let target = self
            .lends
            .target(at, step)
            .map(|lend| (lend.buffer.uuid, lend.buffer.surface.clone(), lend.written));
        let Some((uuid, surface, written)) = target else {
            if !self.guest_gave_up(frame_id) {
                let status = if buffer_status == NO_ERROR {
                    Status::VideoDecoderMalfunction.code()
                } else {
                    buffer_status
                };
                self.emit_decoded_frame(frame_id, status, info, None);
            }
            return Some(frame);
        };
        let mut frame = Some(frame);
        if !written && let Some(unwritten) = frame.take() {
            let ahead = reply.sent && reply.when_buffer_is_in_hand;
            let (back, copied) = self.copy_to_guest(unwritten, &surface, ahead);
            frame = back;
            if let Err(status) = copied {
                log::warn!("frame {frame_id} cannot be copied to the guest: {status}");
                self.emit_decoded_frame(frame_id, status, info, None);
                return frame;
            }
        }
        if let Some(lend) = self.lends.target(at, step) {
            lend.written = true;
            lend.emitted = true;
        }
        self.emit_decoded_frame(frame_id, NO_ERROR, info, Some(uuid));
        frame
    }

    /// Lets go, after step `at`, of the buffers due at the next step's
    /// start: their `rels` comes with the next message's sweep, and those
    /// never emitted go back in the next `Dfub`, after the next step.
    fn release_lends(&mut self, at: u64) {
        for lend in self.lends.release(at) {
            if !lend.emitted {
                self.unemitted.push(lend.buffer.uuid);
            }
            self.releasing.push(lend.buffer);
        }
    }

    /// Lets go of the lent buffers released ([`Worker::release_lends`]) as
    /// the guest's next message arrives, a `Dbuf`'s reply or a message for
    /// the session, and sweeps: Apple's `rels` for them comes with that
    /// message's sweep, before the session's next message (measured as
    /// [`Worker::emit_frames_decoded`]). Held until then, no sweep of the
    /// device's thread after an earlier message sends it sooner; and let go
    /// as one step for that thread's sweep of the message that came
    /// ([`HostShared::let_go`]), they are named in one `rels`, never split
    /// by a sweep between their ends.
    fn let_go_of_released(&mut self) {
        if self.releasing.is_empty() {
            return;
        }
        let releasing = std::mem::take(&mut self.releasing);
        self.link.let_go(releasing);
    }

    /// Follows a decode's frames with no buffer asked for, for a frame the
    /// guest gave up on: the buffers held go as they would have.
    fn follow_frames_decoded(&mut self, steps: &[FrameDecoded]) {
        for step in steps {
            let at = self.lends.begin();
            self.lends.follow(at, step);
            self.release_lends(at);
        }
    }

    /// `Dpba` before a step's `Dbuf`, at the frame's size, where it is not
    /// the size last asked for, starting from the largest frame the
    /// sequence codes: AVD asked for a 640x360 base layer's buffers before
    /// the first `Dbuf` of a 1280x720 sequence, then for 1280x720 before
    /// the upper layer's, and nothing for a 1280x720 stream under a
    /// 1920x1080 description (libaom's and libvpx's streams, macOS 27.0.1,
    /// 2026-10-24).
    fn ask_for_the_steps_size(&mut self, step: &FrameDecoded) {
        let size = (step.width, step.height);
        let last = self.asked_size.replace(size).unwrap_or(step.largest);
        if last == size {
            return;
        }
        let Some(format) = self.decoder.as_ref().map(|decoder| decoder.output_format()) else {
            return;
        };
        if let Err(error) = self.set_pixel_buffer_attributes(size, format) {
            log::warn!("no Dpba for a frame of {}x{}: {error}", size.0, size.1);
        }
    }

    /// `Dbuf`: a pixel buffer for `frame_id`, its UUID and its surface,
    /// waited for 1.1 s; the error is the guest's status, or ours. The
    /// buffers released go as the reply comes ([`Worker::let_go_of_released`]),
    /// and with them, in the same sweep, a surface the reply lent with no
    /// buffer.
    fn create_pixel_buffer(&mut self, frame_id: i64) -> Result<GuestBuffer, OsStatus> {
        let asked = self.runtime.now();
        let reply = self
            .link
            .request(
                names::CREATE_PIXEL_BUFFER,
                DEFAULT_REPLY_TIMEOUT,
                |message| {
                    message.s64(names::FRAME, frame_id);
                },
            )
            .map_err(Status::code);
        self.times
            .add(|times| &mut times.buffer, self.runtime.elapsed_since(asked));
        let reply = reply?;
        let mut lent = Err(Status::Parameter.code());
        let refused = ending_together(|| lent = buffer_of(reply));
        let releasing = std::mem::take(&mut self.releasing);
        if !releasing.is_empty() || !refused.is_empty() {
            self.link.let_go((releasing, refused));
        }
        lent
    }

    /// `Demt`: the frame's status and info flags, and the buffer it was
    /// decoded into, waited for 10 s; the guest's answer is not read.
    fn emit_decoded_frame(
        &mut self,
        frame_id: i64,
        status: OsStatus,
        info: u32,
        buffer: Option<GuestUuid>,
    ) {
        let emitting = self.runtime.now();
        let sent = self.link.request(
            names::EMIT_DECODED_FRAME,
            CODEC_OPERATION_REPLY_TIMEOUT,
            |message| {
                message.s64(names::FRAME, frame_id);
                message.s32(names::ERROR, status);
                message.u32(names::INFO_FLAGS, info);
                if let Some(uuid) = buffer {
                    message.uuid(names::PIXEL_BUFFER_UUID, uuid);
                }
            },
        );
        self.times.add(
            |times| &mut times.emit,
            self.runtime.elapsed_since(emitting),
        );
        if let Err(error) = sent {
            log::warn!("emitting frame {frame_id} failed: {error:?}");
        }
    }

    /// After each frame: the buffers let go unemitted go back to the guest
    /// in a `Dfub`, then the host session sweeps its surfaces.
    fn clean_up_after_decode(&mut self) {
        self.clean_up_releasing(None);
    }

    /// [`Worker::clean_up_after_decode`], holding `buffer`'s surface until
    /// the `Dfub` is sent, so that no sweep sends its `rels` first, and
    /// letting it go in the sweep ([`HostShared::let_go`]).
    fn clean_up_releasing(&mut self, buffer: Option<GuestBuffer>) {
        self.give_back_unemitted();
        self.link.let_go(buffer);
    }

    /// The clean-up after an AV1 or VP9 step: its `Dfub`, then a sweep, which
    /// lets go too of the buffers the step before let go, where no message
    /// of the guest's has come since to let them go. An asynchronous unit's
    /// steps have none between them, and AVD's `rels` for a buffer due at a
    /// step came after that step's `Dfub`, or its `Demt` where it shows a
    /// frame (frames sent 33 ms apart).
    fn clean_up_after_step(&mut self) {
        self.give_back_unemitted();
        let releasing = std::mem::take(&mut self.releasing);
        self.link.let_go(releasing);
    }

    /// `Dfub` of the buffers let go without a `Demt`, where there are any.
    fn give_back_unemitted(&mut self) {
        if self.unemitted.is_empty() {
            return;
        }
        let mut message = MessageWriter::new(
            names::FREE_UNUSED_PIXEL_BUFFERS,
            MessageFlags::NONE,
            Some(self.link.guest_uuid),
            0,
        );
        message.uuid_array(names::PIXEL_BUFFER_UUIDS, &self.unemitted);
        self.unemitted.clear();
        self.send(message.finish());
    }

    /// `dcan`: whether the decoder can go on with a new format
    /// description; 0 before a session is started, as Apple's decoder
    /// answered.
    fn can_accept(&self, message: &Message<'_>) -> Handled {
        let description = message.image_description(names::FORMAT_DESCRIPTION)?;
        let config = description
            .map(ImageDescription::parse)
            .transpose()?
            .as_ref()
            .and_then(config_from_description);
        let can = match (&self.decoder, config) {
            (Some(decoder), Some(config)) => self
                .config
                .as_ref()
                .and_then(|current| apple_accepts(current, &config))
                .unwrap_or_else(|| decoder.can_accept(&config)),
            _ => false,
        };
        let mut reply = reply(message);
        reply.s32(names::CAN_ACCEPT, i32::from(can));
        self.send(reply.finish());
        Ok(())
    }

    /// `dsup`: the properties this decoder has ([`supported_properties`]),
    /// each with an empty description, as Apple's decoders list most of
    /// their own.
    fn supported_properties(&self, message: &Message<'_>) -> Handled {
        let supported: plist::Dictionary = supported_properties(self.codec)
            .iter()
            .map(|&key| {
                (
                    key.to_owned(),
                    plist::Value::Dictionary(plist::Dictionary::new()),
                )
            })
            .collect();
        let mut reply = reply(message);
        reply.s32(names::ERROR, NO_ERROR);
        reply.property_list(
            names::SUPPORTED_PROPERTIES,
            Some(&plist::Value::Dictionary(supported)),
        )?;
        self.send(reply.finish());
        Ok(())
    }

    /// `dcop`: a property's value. The key must be a string, or there is
    /// no reply. A VP9 decoder answers as Apple's VP9 decoder does: its
    /// quality of service tiers and `OnlyTheseFrames` read, `RealTime` and
    /// `MaximizePowerEfficiency` not (measured on macOS 27.0.1, 26A434,
    /// 2026-10-25).
    fn copy_property(&self, message: &Message<'_>) -> Handled {
        let key = property_key(message)?;
        let vp9 = self.codec == Codec::Vp9;
        let (status, value) = match key.as_str() {
            USING_HARDWARE => (NO_ERROR, Some(plist::Value::Boolean(true))),
            REAL_TIME | MAXIMIZE_POWER_EFFICIENCY if !vp9 => (
                NO_ERROR,
                Some(
                    self.properties
                        .get(&key)
                        .cloned()
                        .unwrap_or(plist::Value::Boolean(false)),
                ),
            ),
            AV1_FILM_GRAIN_MODE if self.codec == Codec::Av1 => (
                NO_ERROR,
                Some(plist::Value::String(FILM_GRAIN_IMMEDIATE.into())),
            ),
            SUGGESTED_QUALITY_OF_SERVICE_TIERS if vp9 => (
                NO_ERROR,
                Some(plist::Value::Array(
                    QUALITY_OF_SERVICE_TIERS
                        .into_iter()
                        .map(|tier| {
                            plist::Value::Dictionary(dictionary([(
                                ONLY_THESE_FRAMES,
                                plist::Value::String(tier.into()),
                            )]))
                        })
                        .collect(),
                )),
            ),
            ONLY_THESE_FRAMES if vp9 => (NO_ERROR, Some(plist::Value::String(ALL_FRAMES.into()))),
            _ => (Status::PropertyNotSupported.code(), None),
        };
        let mut reply = reply(message);
        reply.s32(names::ERROR, status);
        if let Some(value) = value {
            reply.property_list(names::PROPERTY_VALUE, Some(&value))?;
        }
        self.send(reply.finish());
        Ok(())
    }

    /// `dset`: sets a property. The key must be a string and not
    /// `Paravirtualized`, or there is no reply.
    fn set_property(&mut self, message: &Message<'_>) -> Handled {
        let key = property_key(message)?;
        if key == PARAVIRTUALIZED {
            return Err(WireError::Malformed("a property the host handles"));
        }
        let value = message.property_list(names::PROPERTY_VALUE)?;
        let status = self.set(&key, value);
        self.reply_status(message, status)
    }

    /// A property set: a VP9 decoder takes none, as Apple's VP9 decoder
    /// refuses `RealTime`, `MaximizePowerEfficiency` and `OnlyTheseFrames`
    /// with -12900, and its read-only properties with -12901 (measured on
    /// macOS 27.0.1, 26A434, 2026-10-25).
    fn set(&mut self, key: &str, value: Option<plist::Value>) -> OsStatus {
        let vp9 = self.codec == Codec::Vp9;
        match key {
            REAL_TIME | MAXIMIZE_POWER_EFFICIENCY if !vp9 => {
                match value {
                    Some(value) => self.properties.insert(key.to_owned(), value),
                    None => self.properties.remove(key),
                };
                NO_ERROR
            }
            USING_HARDWARE => Status::PropertyReadOnly.code(),
            SUGGESTED_QUALITY_OF_SERVICE_TIERS if vp9 => Status::PropertyReadOnly.code(),
            // Grain is applied where the frame decodes: `Defer`, which
            // the guest never sends, would ask for it left to the guest.
            AV1_FILM_GRAIN_MODE if self.codec == Codec::Av1 => {
                if value.as_ref().and_then(plist::Value::as_string) == Some(FILM_GRAIN_IMMEDIATE) {
                    NO_ERROR
                } else {
                    Status::PropertyNotSupported.code()
                }
            }
            _ => Status::PropertyNotSupported.code(),
        }
    }

    /// `dmul`: `,pro` without `Paravirtualized` must not be empty, or there
    /// is no reply. Apple's AVC decoder sets none of them together and
    /// answers -12782, and the guest then sets them one by one; so does
    /// this one.
    fn set_properties(&self, message: &Message<'_>) -> Handled {
        let properties = message.dictionary(names::PROPERTIES)?.unwrap_or_default();
        if !properties.keys().any(|key| key != PARAVIRTUALIZED) {
            return Err(WireError::Malformed("no properties for the decoder"));
        }
        self.reply_status(message, Status::UnimplementedFunction.code())
    }
}

/// The properties a decoder of `codec` lists in `dsup`.
/// - Every decoder: `UsingHardwareAcceleratedVideoDecoder`, which every
///   Apple decoder measured lists.
/// - H.264, HEVC and AV1: `RealTime` and `MaximizePowerEfficiency` too,
///   which Apple's AVC decoder lists among its own, and for AV1
///   `AV1FilmGrainMode`, as Apple's AV1 decoder lists it.
/// - VP9: `SuggestedQualityOfServiceTiers` too, which with the first is
///   what Apple's VP9 decoder lists but for `MaxDecoderPixelRate`, before
///   and after `dsta` (measured on macOS 27.0.1, 26A434, 2026-10-25). That
///   one's value is the decoder hardware's decode rate (7365427200 on an
///   M3 Max, whatever the stream), which no backend here reports, so it is
///   not listed.
fn supported_properties(codec: Codec) -> &'static [&'static str] {
    match codec {
        Codec::Vp9 => &[USING_HARDWARE, SUGGESTED_QUALITY_OF_SERVICE_TIERS],
        Codec::Av1 => &[
            USING_HARDWARE,
            REAL_TIME,
            MAXIMIZE_POWER_EFFICIENCY,
            AV1_FILM_GRAIN_MODE,
        ],
        Codec::H264 | Codec::Hevc => &[USING_HARDWARE, REAL_TIME, MAXIMIZE_POWER_EFFICIENCY],
    }
}

/// The buffer a `Dbuf` reply lends, its UUID and its surface, or the
/// guest's status, or ours; a reply that lends none lets go of its surface.
fn buffer_of(reply: Reply) -> Result<GuestBuffer, OsStatus> {
    let code = |error: WireError| error.status().code();
    let message = Message::parse(&reply.bytes).map_err(code)?;
    let status = message.s32(names::ERROR).map_err(code)?.unwrap_or(0);
    if status != NO_ERROR {
        return Err(status);
    }
    let uuid = message.uuid(names::PIXEL_BUFFER_UUID).map_err(code)?;
    match (uuid, reply.surface) {
        (Some(uuid), Some(surface)) => Ok(GuestBuffer { uuid, surface }),
        // Apple's makes its pixel buffer of the surface, and fails
        // without one.
        _ => Err(Status::Parameter.code()),
    }
}

/// The decoder configuration of a `=fds` format description: its codec,
/// size and configuration record.
fn config_from_description(description: &ImageDescription) -> Option<DecoderConfig> {
    let codec = Codec::from_fourcc(description.codec)?;
    let record = match codec {
        Codec::H264 => AVC_CONFIGURATION,
        Codec::Hevc => HEVC_CONFIGURATION,
        Codec::Av1 => AV1_CONFIGURATION,
        Codec::Vp9 => VP9_CONFIGURATION,
    };
    Some(DecoderConfig {
        codec,
        width: u32::from(description.width),
        height: u32::from(description.height),
        configuration: description.extension(record).unwrap_or_default().to_vec(),
    })
}

/// The decoder configuration of a sample buffer's format description.
fn config_from_format(format: &SerializedFormatDescription) -> Option<DecoderConfig> {
    let codec = Codec::from_fourcc(format.codec)?;
    let record = match codec {
        Codec::H264 => "avcC",
        Codec::Hevc => "hvcC",
        Codec::Av1 => "av1C",
        Codec::Vp9 => "vpcC",
    };
    Some(DecoderConfig {
        codec,
        width: u32::try_from(format.width).ok()?,
        height: u32::try_from(format.height).ok()?,
        configuration: format
            .sample_description_atom(record)
            .unwrap_or_default()
            .to_vec(),
    })
}

/// Why no backend here starts a session for `config`, which core can tell
/// from the record before any backend is asked. Each is answered with
/// Apple's decoders' status for it (measured on macOS 27, 26A434,
/// 2026-10-24):
/// - 3-byte NAL unit lengths, which ISO/IEC 14496-15 does not allow
///   (`lengthSizeMinusOne` is 0, 1 or 3, 5.3.3.1.2): Apple's AVC decoder's
///   kVTVideoDecoderUnsupportedDataFormatErr;
/// - an SPS that lets pictures be fields (`frame_mbs_only_flag` 0), MBAFF
///   or not: the codec crate decodes no field and no MBAFF frame, and
///   VideoToolbox's hardware decoder refuses such a session itself, so no
///   backend takes one. Apple's AVC decoder's kVTVideoDecoderMalfunctionErr,
///   with no `Dpba`;
/// - VP9 at 12 bits: core's pixel formats hold 8 and 10 bits alone
///   ([`PixelFormat`]). Apple's VP9 decoder's kVTVideoDecoderMalfunctionErr.
/// - A `vp09` description with no `vpcC` record, which tells no backend
///   the stream's depth and range, as the guest's buffers must have them
///   before the first sample: Apple's VP9 decoder's own -8971
///   ([`Status::ConfigurationRecordMissing`], measured on macOS 27.0.1,
///   2026-10-25), as a [`BackendError::Native`] status, which reaches the
///   guest as it is.
///
/// The sizes a session takes are the backend's. Apple's decoders refuse a
/// side under 64 at `dsta`, H.264 and HEVC by the SPS's coded size; here
/// a backend refuses only what it does not decode right.
fn session_refusal(config: &DecoderConfig) -> Option<BackendError> {
    match config.codec {
        Codec::H264 => avc_refusal(&config.configuration),
        Codec::Vp9 if config.configuration.is_empty() => Some(BackendError::Native {
            status: Status::ConfigurationRecordMissing.code(),
            message: "a vp09 description without its vpcC record".into(),
        }),
        Codec::Vp9 => {
            // vpcC: the box's version and flags, the profile, the level,
            // then the bit depth in the top four bits.
            let bit_depth = config.configuration.get(6).map(|byte| byte >> 4);
            (bit_depth > Some(10)).then(|| {
                BackendError::Malfunction("VP9 at 12 bits: no pixel format here holds them".into())
            })
        }
        Codec::Hevc | Codec::Av1 => None,
    }
}

/// Why no backend starts a session on the `avcC` `record`
/// ([`session_refusal`]).
fn avc_refusal(record: &[u8]) -> Option<BackendError> {
    if avc_record::length_size(record) == Some(3) {
        return Some(BackendError::Unsupported(
            "3-byte NAL unit lengths, which ISO/IEC 14496-15 does not allow".into(),
        ));
    }
    avc_record::fields_allowed(record)?.then(|| {
        BackendError::Malfunction(
            "an SPS that lets pictures be fields (frame_mbs_only_flag 0): \
             no backend here decodes fields"
                .into(),
        )
    })
}

/// Whether Apple's decoder would go on from the session's `current`
/// configuration to `next` without a new session, where it decides
/// before the backend: `None` leaves it to the backend. Apple's AVC
/// decoder answers no to a description it would refuse a session on, and
/// to one whose NAL unit length size or dimensions differ from the
/// session's; another profile at the same size it takes (measured on macOS
/// 27.0.1, 26A434, 2026-10-24).
fn apple_accepts(current: &DecoderConfig, next: &DecoderConfig) -> Option<bool> {
    if current.codec != Codec::H264 || next.codec != Codec::H264 {
        return None;
    }
    let refused = avc_refusal(&next.configuration).is_some()
        || avc_record::length_size(&current.configuration)
            != avc_record::length_size(&next.configuration)
        || (current.width, current.height) != (next.width, next.height);
    refused.then_some(false)
}

/// The most reasons a session logs for frames its backend's device did not
/// write: each the first time it comes.
const MOST_DECLINES_LOGGED: usize = 8;

/// Whether a session's frames are written into the guest's buffers by the
/// backend's device, and the reasons it logged for those that were not,
/// each once.
struct ZeroCopy {
    on: bool,
    logged: Mutex<HashSet<String>>,
    /// Whether one reason went unlogged, past [`MOST_DECLINES_LOGGED`].
    unlogged: AtomicBool,
}

impl ZeroCopy {
    fn new(on: bool) -> Self {
        if !on {
            log::info!("zero copy is off: the CPU copies every frame into the guest's buffers");
        }
        Self {
            on,
            logged: Mutex::new(HashSet::new()),
            unlogged: AtomicBool::new(false),
        }
    }

    /// `frame` written into `surface`, laid out as `layout`, by its
    /// backend's device; `None` where the CPU is to copy it, the reason
    /// logged the first time it comes. The time it took goes to `parts`.
    fn write(
        &self,
        frame: &dyn DecodedFrame,
        held: &Arc<SurfaceHold>,
        layout: &SurfaceLayout,
        parts: &mut CopyTimes,
        clock: &dyn Runtime,
    ) -> Option<Result<(), OsStatus>> {
        if !self.on {
            return None;
        }
        let Some(memory) = held.surface.device_memory() else {
            self.declined(frame, "the host lends devices no memory of this buffer");
            return None;
        };
        // A device that does not finish by its deadline keeps the buffer
        // held with this until it does (`crate::device_wait`).
        let kept: Arc<dyn std::any::Any + Send + Sync> = held.clone();
        let hold = GuestHold::new(kept);
        let target = match GuestTarget::new(layout, memory) {
            Ok(target) => target.with_hold(hold),
            Err(why) => {
                self.declined(frame, &format!("the buffer's layout: {why}"));
                return None;
            }
        };
        let writing = clock.now();
        let written = frame.write_to_guest(&target);
        parts.device += clock.elapsed_since(writing);
        match written {
            Ok(DeviceWrite::Written) => {
                held.surface.device_wrote();
                parts.written += 1;
                Some(Ok(()))
            }
            Ok(DeviceWrite::Declined(why)) => {
                parts.declined += 1;
                self.declined(frame, &why);
                None
            }
            Err(error) => {
                log::warn!(
                    "frame {} cannot be written into the guest's buffer: {error}",
                    frame.frame_id()
                );
                // A device that missed its deadline may still write the buffer: told of
                // once it is done (`GuestHold`).
                let wrote = Arc::clone(held);
                target
                    .hold()
                    .report_late_write(Box::new(move || wrote.surface.device_wrote()));
                Some(Err(error.status()))
            }
        }
    }

    /// Logs `why` the CPU copies `frame`, the first time it comes.
    fn declined(&self, frame: &dyn DecodedFrame, why: &str) {
        let mut logged = self.logged.lock();
        if logged.contains(why) {
            return;
        }
        if logged.len() >= MOST_DECLINES_LOGGED {
            if !self.unlogged.swap(true, Ordering::Relaxed) {
                log::info!("zero copy: more reasons than are logged; the CPU copies those frames");
            }
            return;
        }
        logged.insert(why.to_owned());
        log::info!(
            "zero copy: the CPU copies frame {:#x} into the guest's buffer, and the session's \
             like it: {why}",
            frame.frame_id()
        );
    }
}

/// The copy a decoder session's copy thread makes of `frame` into
/// `target` ([`Worker::copy_looking_ahead`]): it waits for the frame to be
/// whole on its device, tells the session's thread through `wake` that it
/// may decode the next, then copies, and gives the copy's status and how
/// long its parts took.
fn copy_ahead(
    frame: &dyn DecodedFrame,
    target: &Arc<SurfaceHold>,
    wake: &WakeWhenCopied,
    zero_copy: &ZeroCopy,
    clock: &dyn Runtime,
) -> (Result<(), OsStatus>, CopyTimes) {
    let waiting = clock.now();
    let waited = frame.wait();
    let mut parts = CopyTimes {
        wait: clock.elapsed_since(waiting),
        ..CopyTimes::default()
    };
    // An Err is a session's thread that has ended.
    let _ = wake.0.send(Job::Own(CopyEvent::Waited));
    let copied = match waited {
        Ok(()) => copy_frame(frame, target, &mut parts, zero_copy, clock),
        Err(error) => {
            log::warn!("frame {} does not finish: {error}", frame.frame_id());
            Err(error.status())
        }
    };
    (copied, parts)
}

/// Copies `frame` into a guest surface laid out for the same sample size,
/// adding the time of its parts to `parts`: the backend's device writes it
/// there itself where it can and zero copy is on ([`ZeroCopy::write`]),
/// with no lend; else the backend reads the frame back first, and only its
/// copy on the CPU runs while the surface is lent.
fn copy_frame(
    frame: &dyn DecodedFrame,
    held: &Arc<SurfaceHold>,
    parts: &mut CopyTimes,
    zero_copy: &ZeroCopy,
    clock: &dyn Runtime,
) -> Result<(), OsStatus> {
    let surface = held.surface.as_ref();
    let layout = surface.layout();
    let same_samples = PixelFormat::from_fourcc(layout.pixel_format)
        .is_some_and(|format| format.bytes_per_sample() == frame.pixel_format().bytes_per_sample());
    if !same_samples {
        log::warn!(
            "guest buffer is '{}', frames are '{}'",
            layout.pixel_format,
            frame.pixel_format().fourcc()
        );
        return Err(Status::VideoDecoderMalfunction.code());
    }
    if let Some(written) = zero_copy.write(frame, held, &layout, parts, clock) {
        return written;
    }
    parts.copied += 1;
    let reading = clock.now();
    let read = frame.read_back();
    parts.read_back += clock.elapsed_since(reading);
    let read = read.map_err(|error| error.status())?;
    let mut copied = Err(Status::VideoDecoderMalfunction.code());
    let lending = clock.now();
    let lent = surface.write(&mut |memory| {
        let filling = clock.now();
        copied = match layout.split(memory) {
            Ok(mut planes) => read.copy_to(&mut planes).map_err(|error| error.status()),
            Err(error) => {
                log::warn!("guest buffer: {error}");
                Err(Status::VideoDecoderMalfunction.code())
            }
        };
        parts.fill += clock.elapsed_since(filling);
    });
    parts.lend += clock.elapsed_since(lending);
    lent.map_err(|error| {
        log::warn!("guest buffer: {error}");
        Status::VideoDecoderMalfunction.code()
    })?;
    copied
}

#[cfg(test)]
mod tests;
