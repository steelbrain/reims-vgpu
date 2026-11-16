//! A decoder session (`+dec`): the messages of one guest decoder, on a thread of the session's
//! own, over the host's video arm ([`Backend`]).
//!
//! The guest's decoder sends its session's messages to the host, which answers each on the
//! session's thread, in order: a reply (flag `REPLY_TO_GUEST`), or for a message it cannot read,
//! nothing, and the guest's wait for the reply times out. The host's own requests (flag
//! `HOST_EXPECTS_REPLY`), `Dbuf` and `Demt`, wait for the guest's reply (flag `REPLY_TO_HOST`),
//! which goes straight to the request that waits for it ([`Session::deliver`]), not through the
//! session's queue.
//!
//! - **`dsta`** starts the session once: its format description (`,fds`) names the codec, the
//!   size and the configuration record (`avcC`, `hvcC`). A record of 3-byte NAL unit lengths is
//!   refused with `kVTVideoDecoderUnsupportedDataFormatErr`, an H.264 SPS that lets pictures be
//!   fields with `kVTVideoDecoderMalfunctionErr`, and the backend refuses what it does not decode,
//!   each with no `Dpba`. A started session sends `Dpba`, the attributes the guest's output
//!   buffers are to have, before its reply. A second `dsta` is answered
//!   `kVTVideoDecoderBadDataErr`, and the session goes on as the first left it.
//! - **`dfrm`** decodes one sample: the host asks the guest for an output buffer (`Dbuf`, waiting
//!   [`BUFFER_REPLY_WAIT`]), which lends the buffer's surface; the backend decodes the sample's
//!   access unit into it and returns once the picture is there; the host emits the frame
//!   (`Demt`, waiting [`EMIT_REPLY_WAIT`]) and lets go of the surface (`rels`). A frame that does
//!   not decode is emitted with the backend's status and no buffer, and its buffer named unused
//!   (`Dfub`). Frames come out in decode order, one a `dfrm`. The `dfrm` reply says noErr, with
//!   `kVTDecodeInfo_Asynchronous` for a frame sent with the asynchronous flag, whose reply goes
//!   as soon as its buffer is in hand. With no decoder (before `dsta`, after one refused, after
//!   `dinv`) it answers [`NO_DECODER`] and sends nothing else.
//! - A sample buffer whose format description carries another configuration record reopens the
//!   decoder from that sample.
//! - **The rest**: `dsup` lists `UsingHardwareAcceleratedVideoDecoder` (true, read-only),
//!   `RealTime` and `MaximizePowerEfficiency` (taken); `dcop` and `dset` answer them;
//!   `Paravirtualized` is the guest's own, and a `dset` of it gets no reply; `dcan` answers
//!   whether the decoder goes on with another format description; `dinv` drops the decoder;
//!   `dfin`, `dmul`, `dsav`, `dtfi`, `dsts` and `dtil` answer `kVTUnimplementedErr`.
//!
//! A message for a surface the session does not hold, and a surface lent with a message that
//! takes none, is let go of at once (`rels`): the guest's kext holds each surface it lent until
//! the host lets go of it.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::Duration;

use crate::codec::h264::avcc;
use crate::codec::h264::nal::LengthPrefixed;
use crate::codec::h264::sps::Sps as H264Sps;
use crate::codec::hevc::hvcc;
use crate::codec::hevc::nal::nal_type;
use crate::codec::hevc::sps::Sps as HevcSps;
use crate::core::fourcc::FourCc;
use crate::core::runtime::{self, Answerer, Runtime};
use crate::core::sync::Mutex;
use crate::paravirt::WireError;
use crate::paravirt::host::Codec;
use crate::paravirt::image_description::{AVC_CONFIGURATION, HEVC_CONFIGURATION, ImageDescription};
use crate::paravirt::message::{GuestUuid, Message, MessageFlags, MessageHeader, MessageWriter};
use crate::paravirt::names;
use crate::paravirt::plist::Value;
use crate::paravirt::sample_buffer::{SampleBuffer, SerializedFormatDescription};
use crate::paravirt::status::{self, OsStatus};

/// How long a `Dbuf` waits for the guest's buffer.
pub const BUFFER_REPLY_WAIT: Duration = Duration::from_millis(1100);
/// How long a `Demt` waits for the guest's answer.
pub const EMIT_REPLY_WAIT: Duration = Duration::from_secs(10);

/// What a `dfrm` with no decoder answers, as the guest's decoders see it from a host with none.
pub const NO_DECODER: OsStatus = 311;

/// The most bytes of one `dfrm`'s fragments the session keeps: ours, past any sample the guest's
/// 10 MiB message buffer holds.
pub const MAX_FRAME_MESSAGE_BYTES: usize = 64 << 20;

/// `kVTDecodeFrame_EnableAsynchronousDecompression`.
const DECODE_ASYNCHRONOUSLY: u32 = 1 << 0;
/// `kVTDecodeInfo_Asynchronous`.
const INFO_ASYNCHRONOUS: u32 = 1 << 0;
/// `kVTDecodeInfo_FrameDropped`.
const INFO_FRAME_DROPPED: u32 = 1 << 1;

/// The property the guest's VideoToolbox keeps for itself.
const PARAVIRTUALIZED: &str = "Paravirtualized";
/// `kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder`.
const USING_HARDWARE: &str = "UsingHardwareAcceleratedVideoDecoder";
/// `kVTDecompressionPropertyKey_RealTime`.
const REAL_TIME: &str = "RealTime";
/// `kVTDecompressionPropertyKey_MaximizePowerEfficiency`.
const MAXIMIZE_POWER_EFFICIENCY: &str = "MaximizePowerEfficiency";

/// The thread a session's messages run on.
const THREAD_NAME: &str = "reims-vgpu-video.vt-decoder";

/// The four-byte start code an Annex-B NAL unit follows.
const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// A guest surface a message lent the host, as the device resolved it: held by the host until it
/// lets go of it with a `rels` of [`LentSurface::id`].
pub trait LentSurface: Send + Sync + std::fmt::Debug {
    /// The name the guest's kext gives the surface, which a `rels` names it by.
    fn id(&self) -> u64;
    /// The device's own type, for its backend to read.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// The way the host's messages reach the guest's VideoToolbox, from any thread.
pub trait Link: Send + Sync {
    /// Sends `message`, a `vtpv` message or a `rels` box; false where it was not sent.
    fn send(&self, message: &[u8]) -> bool;
}

/// A decoder's configuration, from a `dsta`'s format description or a sample's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecoderConfig {
    /// The codec.
    pub codec: Codec,
    /// The width the description gives.
    pub width: u32,
    /// The height the description gives.
    pub height: u32,
    /// The configuration record (`avcC`, `hvcC`).
    pub record: Vec<u8>,
}

/// What a backend's decode of one access unit came to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decoded {
    /// The picture is in the output surface.
    Landed,
    /// The stream's random access skips the picture, and the output got nothing.
    Skipped,
}

/// The host's video arm, as the decoder sessions take it.
pub trait Backend: Send + Sync {
    /// A decoder of `config`, on the session's thread at its `dsta`, or the status the guest sees
    /// for a configuration it does not decode.
    fn open(&self, config: &DecoderConfig) -> Result<Box<dyn BackendDecoder>, OsStatus>;
}

/// One open decoder, called on its session's thread, one call at a time.
pub trait BackendDecoder: Send {
    /// Decodes `access_unit`, NAL units in Annex-B form, after `parameter_sets` (the record's,
    /// Annex-B, at the first decode of a configuration; else empty), into `output`, and returns
    /// once the picture is there, or the status the guest sees for a sample that does not decode.
    fn decode(
        &mut self,
        parameter_sets: &[u8],
        access_unit: &[u8],
        output: &dyn LentSurface,
    ) -> Result<Decoded, OsStatus>;
}

/// The guest's reply to a host request, with the surfaces it lent.
pub(crate) struct GuestReply {
    bytes: Vec<u8>,
    surfaces: Vec<Arc<dyn LentSurface>>,
}

/// What a session's handle and its thread share.
struct Shared {
    uuid: GuestUuid,
    /// The last link a message came with.
    link: Mutex<Option<Arc<dyn Link>>>,
    /// The host's requests waiting for the guest's reply, by reply identifier.
    waiting: Mutex<HashMap<u64, Answerer<GuestReply>>>,
}

impl Shared {
    fn send(&self, message: &[u8]) -> bool {
        let link = self.link.lock().clone();
        link.is_some_and(|link| link.send(message))
    }

    /// Lets go of `surfaces`, where there are any: a `rels` of their IDs.
    fn let_go(&self, surfaces: &[Arc<dyn LentSurface>]) {
        if surfaces.is_empty() {
            return;
        }
        let ids: Vec<u64> = surfaces.iter().map(|surface| surface.id()).collect();
        if !self.send(&surface_release(&ids)) {
            log::warn!(
                "hardware video: a release of {} lent surfaces was not sent",
                ids.len()
            );
        }
    }
}

/// The `rels` box that tells the guest the host holds these surfaces no longer: a big-endian size
/// of 8 plus 8 an ID, the type, then each ID big-endian. It travels alone, not inside a `vtpv`
/// message.
pub fn surface_release(ids: &[u64]) -> Vec<u8> {
    let size = u32::try_from(8 + 8 * ids.len()).unwrap_or(u32::MAX);
    let mut bytes = Vec::with_capacity(8 + 8 * ids.len());
    bytes.extend_from_slice(&size.to_be_bytes());
    bytes.extend_from_slice(b"rels");
    for id in ids {
        bytes.extend_from_slice(&id.to_be_bytes());
    }
    bytes
}

/// A message for the session's thread.
struct Job {
    bytes: Vec<u8>,
    surfaces: Vec<Arc<dyn LentSurface>>,
}

/// A decoder session's handle, which the host keeps by its UUID. Dropping it ends the session's
/// thread once its current message is done.
pub(crate) struct Session {
    queue: Sender<Job>,
    shared: Arc<Shared>,
}

impl Session {
    /// A session of `codec` for `uuid`, its thread started on `runtime`.
    pub(crate) fn start(
        uuid: GuestUuid,
        codec: Codec,
        backend: Arc<dyn Backend>,
        runtime: Arc<dyn Runtime>,
        link: Option<Arc<dyn Link>>,
    ) -> std::io::Result<Self> {
        let shared = Arc::new(Shared {
            uuid,
            link: Mutex::new(link),
            waiting: Mutex::new(HashMap::new()),
        });
        let (queue, jobs) = channel();
        let worker = Worker {
            shared: Arc::clone(&shared),
            jobs,
            runtime: Arc::clone(&runtime),
            backend,
            codec,
            decoder: None,
            config: None,
            length_size: 4,
            parameter_sets: Vec::new(),
            started: false,
            properties: Vec::new(),
            fragments: Vec::new(),
            discarding: false,
            next_request: 1,
        };
        runtime.spawn(THREAD_NAME, Box::new(move || worker.run()))?;
        Ok(Self { queue, shared })
    }

    /// Takes the guest's message `bytes` for the session, with the surfaces it lent and the link
    /// it came on: a reply to a host request goes to the request, any other message to the
    /// session's queue. noErr, or [`status::SESSION_NOT_FOUND`] where the session's thread has
    /// ended.
    pub(crate) fn deliver(
        &self,
        header: &MessageHeader,
        bytes: &[u8],
        surfaces: Vec<Arc<dyn LentSurface>>,
        link: Option<Arc<dyn Link>>,
    ) -> OsStatus {
        if let Some(link) = link {
            *self.shared.link.lock() = Some(link);
        }
        if header.flags.contains(MessageFlags::REPLY_TO_HOST) {
            let waiting = self.shared.waiting.lock().remove(&header.reply_id);
            match waiting {
                Some(answerer) => answerer.give(GuestReply {
                    bytes: bytes.to_vec(),
                    surfaces,
                }),
                // A reply past its request's wait: what it lent goes back.
                None => self.shared.let_go(&surfaces),
            }
            return status::NO_ERROR;
        }
        let job = Job {
            bytes: bytes.to_vec(),
            surfaces,
        };
        match self.queue.send(job) {
            Ok(()) => status::NO_ERROR,
            Err(std::sync::mpsc::SendError(job)) => {
                self.shared.let_go(&job.surfaces);
                status::SESSION_NOT_FOUND
            }
        }
    }
}

/// A frame's `dfrm` reply, sent once.
struct FrameReply {
    request: MessageHeader,
    sent: bool,
    /// Its `,inf`.
    info: u32,
}

/// An output buffer the guest lent for a frame.
struct GuestBuffer {
    uuid: GuestUuid,
    surface: Arc<dyn LentSurface>,
}

type Handled = Result<(), WireError>;

/// The session's thread and what only it touches.
struct Worker {
    shared: Arc<Shared>,
    jobs: Receiver<Job>,
    runtime: Arc<dyn Runtime>,
    backend: Arc<dyn Backend>,
    codec: Codec,
    decoder: Option<Box<dyn BackendDecoder>>,
    config: Option<DecoderConfig>,
    /// The bytes before each NAL unit of the configuration's samples.
    length_size: u8,
    /// The configuration's parameter sets in Annex-B form, for its first decode; empty after it.
    parameter_sets: Vec<u8>,
    /// Whether a `dsta` was answered.
    started: bool,
    /// The properties set (`RealTime`, `MaximizePowerEfficiency`).
    properties: Vec<(String, Value)>,
    /// A `dfrm`'s fragments so far.
    fragments: Vec<u8>,
    /// Whether the fragments of a `dfrm` past [`MAX_FRAME_MESSAGE_BYTES`] are being dropped.
    discarding: bool,
    /// The next host request's reply identifier.
    next_request: u64,
}

impl Worker {
    fn run(mut self) {
        while let Ok(job) = self.jobs.recv() {
            if let Err(error) = self.handle(&job.bytes) {
                log::warn!("hardware video: a VideoToolbox message the host cannot read: {error}");
            }
            // A session's own messages lend nothing it keeps.
            self.shared.let_go(&job.surfaces);
        }
    }

    fn send(&self, message: &[u8]) {
        if !self.shared.send(message) {
            log::warn!(
                "hardware video: a {}-byte VideoToolbox message was not sent",
                message.len()
            );
        }
    }

    fn reply(&self, message: &Message<'_>) -> MessageWriter {
        MessageWriter::reply_to(&message.header(), MessageFlags::REPLY_TO_GUEST)
    }

    fn reply_status(&self, message: &Message<'_>, status: OsStatus) -> Handled {
        let mut reply = self.reply(message);
        reply.s32(names::ERROR, status);
        self.send(&reply.finish());
        Ok(())
    }

    fn handle(&mut self, bytes: &[u8]) -> Handled {
        let message = Message::parse(bytes)?;
        let kind = message.header().message_type;
        match &kind.to_be_bytes() {
            b"dfrm" => self.decode_frame(bytes, message.header()),
            b"dsta" => self.start_session(&message),
            b"dsup" => self.supported_properties(&message),
            b"dcop" => self.copy_property(&message),
            b"dset" => self.set_property(&message),
            b"dmul" => self.set_properties(&message),
            b"dcan" => self.can_accept(&message),
            b"dinv" => {
                self.decoder = None;
                self.reply_status(&message, status::NO_ERROR)
            }
            b"dfin" | b"dsav" | b"dtfi" | b"dsts" | b"dtil" => {
                self.reply_status(&message, status::UNIMPLEMENTED)
            }
            _ => {
                log::warn!("hardware video: an unknown decoder message {kind}");
                self.reply_status(&message, status::UNIMPLEMENTED)
            }
        }
    }

    /// `dsta`: starts the session once (the module doc).
    fn start_session(&mut self, message: &Message<'_>) -> Handled {
        let description = message
            .image_description(names::FORMAT_DESCRIPTION)?
            .ok_or(WireError::Malformed("no format description"))?;
        message.s32(names::GUEST_POOL)?;
        let description = ImageDescription::parse(description)?;
        if self.started {
            log::warn!("hardware video: a second dsta, refused");
            return self.reply_status(message, status::VIDEO_DECODER_BAD_DATA);
        }
        self.started = true;
        let config = config_from_description(&description).filter(|c| c.codec == self.codec);
        let status = match config {
            Some(config) => match self.open(&config) {
                Ok(()) => NO_ERROR,
                Err(status) => status,
            },
            None => status::VIDEO_DECODER_UNSUPPORTED_DATA_FORMAT,
        };
        self.reply_status(message, status)
    }

    /// Opens a decoder of `config`, sending `Dpba`, or the status it is refused with.
    fn open(&mut self, config: &DecoderConfig) -> Result<(), OsStatus> {
        let record = read_record(config)?;
        let decoder = self.backend.open(config)?;
        self.send_buffer_attributes(config, &record);
        self.decoder = Some(decoder);
        self.config = Some(config.clone());
        self.length_size = record.length_size;
        self.parameter_sets = record.parameter_sets;
        Ok(())
    }

    /// `Dpba`: the attributes the guest's output buffers are to have: the format the picture
    /// lands in, the description's size, IOSurface backing, which the host needs to reach them,
    /// and the row and plane alignments and cache mode a hardware decoder's buffers have.
    fn send_buffer_attributes(&self, config: &DecoderConfig, record: &Record) {
        let format = match (record.ten_bit, record.full_range) {
            (false, false) => *b"420v",
            (false, true) => *b"420f",
            (true, false) => *b"x420",
            (true, true) => *b"xf20",
        };
        let attributes = Value::dictionary([
            (
                "PixelFormatType",
                Value::Array(vec![Value::Integer(i64::from(u32::from_be_bytes(format)))]),
            ),
            ("Width", Value::Integer(i64::from(config.width))),
            ("Height", Value::Integer(i64::from(config.height))),
            ("IOSurfaceProperties", Value::Dictionary(Vec::new())),
            ("BytesPerRowAlignment", Value::Integer(64)),
            ("PlaneAlignment", Value::Integer(256)),
            (
                "PreferRealTimeCacheModeIfEveryoneDoes",
                Value::Boolean(true),
            ),
        ]);
        let mut message = MessageWriter::new(
            names::SET_PIXEL_BUFFER_ATTRIBUTES,
            MessageFlags::NONE,
            Some(self.shared.uuid),
            0,
        );
        message.property_list(names::PIXEL_BUFFER_ATTRIBUTES, Some(&attributes));
        self.send(&message.finish());
    }

    /// `dfrm`: a fragment flagged with more to come is kept; the last one completes the message,
    /// whose reply goes to the last fragment's header. A message past
    /// [`MAX_FRAME_MESSAGE_BYTES`] is dropped, its fragments up to the last too, with no reply.
    fn decode_frame(&mut self, bytes: &[u8], last: MessageHeader) -> Handled {
        let more = last.flags.contains(MessageFlags::MORE_FRAGMENTS);
        if !self.discarding
            && self.fragments.len().saturating_add(bytes.len()) > MAX_FRAME_MESSAGE_BYTES
        {
            self.fragments = Vec::new();
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
            return Ok(());
        }
        let whole = if self.fragments.is_empty() {
            bytes.to_vec()
        } else {
            let mut whole = std::mem::take(&mut self.fragments);
            whole.extend_from_slice(bytes);
            whole
        };
        let message = Message::parse(&whole)?;
        let frame = message.s64(names::FRAME)?.unwrap_or(0);
        let serialization = message
            .sample_buffer(names::SAMPLE_BUFFER)?
            .ok_or(WireError::Malformed("no sample buffer"))?;
        let sample = SampleBuffer::parse(&serialization)?;
        let flags = message.u32(names::DECODE_FLAGS)?.unwrap_or(0);
        let info = if flags & DECODE_ASYNCHRONOUSLY != 0 {
            INFO_ASYNCHRONOUS
        } else {
            0
        };
        let mut reply = FrameReply {
            request: last,
            sent: false,
            info,
        };
        if self.decoder.is_none() {
            reply.info = 0;
            self.send_frame_reply(&mut reply, NO_DECODER);
            return Ok(());
        }
        self.decode_sample(frame, &sample, &mut reply);
        self.send_frame_reply(&mut reply, status::NO_ERROR);
        Ok(())
    }

    /// Decodes `sample`, frame `frame`, into a buffer the guest lends, and emits it.
    fn decode_sample(&mut self, frame: i64, sample: &SampleBuffer, reply: &mut FrameReply) {
        let reopened = sample
            .format_description
            .as_ref()
            .and_then(config_from_format)
            .filter(|config| Some(config) != self.config.as_ref())
            .map(|config| self.open(&config));
        let access_unit = match reopened {
            Some(Err(status)) => {
                self.decoder = None;
                Err(status)
            }
            _ => sample
                .data
                .as_deref()
                .ok_or(status::VIDEO_DECODER_BAD_DATA)
                .and_then(|data| annex_b(data, self.length_size)),
        };
        let buffer = match self.create_pixel_buffer(frame) {
            Ok(buffer) => buffer,
            Err(status) => {
                self.emit(frame, status, reply.info, None);
                return;
            }
        };
        if reply.info & INFO_ASYNCHRONOUS != 0 {
            self.send_frame_reply(reply, status::NO_ERROR);
        }
        let parameter_sets = std::mem::take(&mut self.parameter_sets);
        let decoded = access_unit.and_then(|au| match self.decoder.as_mut() {
            Some(decoder) => decoder.decode(&parameter_sets, &au, &*buffer.surface),
            None => Err(status::VIDEO_DECODER_MALFUNCTION),
        });
        match decoded {
            Ok(Decoded::Landed) => {
                self.emit(frame, status::NO_ERROR, reply.info, Some(buffer.uuid))
            }
            Ok(Decoded::Skipped) => {
                self.emit(
                    frame,
                    status::NO_ERROR,
                    reply.info | INFO_FRAME_DROPPED,
                    None,
                );
                self.free_unused(buffer.uuid);
            }
            Err(status) => {
                self.emit(frame, status, reply.info, None);
                self.free_unused(buffer.uuid);
            }
        }
        self.shared.let_go(&[buffer.surface]);
    }

    fn send_frame_reply(&self, reply: &mut FrameReply, status: OsStatus) {
        if reply.sent {
            return;
        }
        reply.sent = true;
        let mut message = MessageWriter::reply_to(&reply.request, MessageFlags::REPLY_TO_GUEST);
        message.s32(names::ERROR, status);
        message.u32(names::INFO_FLAGS, reply.info);
        self.send(&message.finish());
    }

    /// A host request of type `kind`, filled by `fill`, and the guest's reply, waited for `wait`;
    /// `kVTParameterErr` where it was not sent or not answered in time.
    fn request(
        &mut self,
        kind: FourCc,
        wait: Duration,
        fill: impl FnOnce(&mut MessageWriter),
    ) -> Result<GuestReply, OsStatus> {
        let id = self.next_request;
        self.next_request = self.next_request.wrapping_add(1).max(1);
        let (answerer, answer) = runtime::answer();
        self.shared.waiting.lock().insert(id, answerer);
        let mut message = MessageWriter::new(
            kind,
            MessageFlags::HOST_EXPECTS_REPLY,
            Some(self.shared.uuid),
            id,
        );
        fill(&mut message);
        if !self.shared.send(&message.finish()) {
            self.shared.waiting.lock().remove(&id);
            return Err(status::PARAMETER);
        }
        let deadline = self.runtime.deadline_in(wait);
        answer
            .wait_until(&*self.runtime, deadline)
            .map_err(|unanswered| {
                self.shared.waiting.lock().remove(&id);
                log::warn!("hardware video: no answer to {kind} in {wait:?}: {unanswered:?}");
                status::PARAMETER
            })
    }

    /// `Dbuf`: the buffer the guest lends for frame `frame`, or the guest's status, or ours.
    fn create_pixel_buffer(&mut self, frame: i64) -> Result<GuestBuffer, OsStatus> {
        let reply = self.request(names::CREATE_PIXEL_BUFFER, BUFFER_REPLY_WAIT, |message| {
            message.s64(names::FRAME, frame);
        })?;
        let mut surfaces = reply.surfaces.into_iter();
        let surface = surfaces.next();
        // A buffer has one surface; more go back.
        self.shared.let_go(&surfaces.collect::<Vec<_>>());
        let read = |error: WireError| error.status();
        let lent = Message::parse(&reply.bytes)
            .map_err(read)
            .and_then(|message| {
                let status = message.s32(names::ERROR).map_err(read)?.unwrap_or(0);
                if status != status::NO_ERROR {
                    return Err(status);
                }
                message
                    .uuid(names::PIXEL_BUFFER_UUID)
                    .map_err(read)?
                    .ok_or(status::PARAMETER)
            });
        match (lent, surface) {
            (Ok(uuid), Some(surface)) => Ok(GuestBuffer { uuid, surface }),
            (lent, surface) => {
                if let Some(surface) = surface {
                    self.shared.let_go(&[surface]);
                }
                // A buffer is made of its surface, and fails without one.
                Err(lent.err().unwrap_or(status::PARAMETER))
            }
        }
    }

    /// `Demt`: the frame's status and info flags, and the buffer it was decoded into; the guest's
    /// answer is waited for and not read.
    fn emit(&mut self, frame: i64, status: OsStatus, info: u32, buffer: Option<GuestUuid>) {
        let sent = self.request(names::EMIT_DECODED_FRAME, EMIT_REPLY_WAIT, |message| {
            message.s64(names::FRAME, frame);
            message.s32(names::ERROR, status);
            message.u32(names::INFO_FLAGS, info);
            if let Some(uuid) = buffer {
                message.uuid(names::PIXEL_BUFFER_UUID, uuid);
            }
        });
        if let Ok(reply) = sent {
            self.shared.let_go(&reply.surfaces);
        }
    }

    /// `Dfub`: the buffer `uuid` was let go of unemitted.
    fn free_unused(&self, uuid: GuestUuid) {
        let mut message = MessageWriter::new(
            names::FREE_UNUSED_PIXEL_BUFFERS,
            MessageFlags::NONE,
            Some(self.shared.uuid),
            0,
        );
        message.uuid_array(names::PIXEL_BUFFER_UUIDS, &[uuid]);
        self.send(&message.finish());
    }

    /// `dcan`: whether the decoder goes on with another format description: the same codec,
    /// NAL unit length and size, and a record it would start a session on; 0 with no decoder.
    fn can_accept(&self, message: &Message<'_>) -> Handled {
        let description = message.image_description(names::FORMAT_DESCRIPTION)?;
        let next = description
            .map(ImageDescription::parse)
            .transpose()?
            .as_ref()
            .and_then(config_from_description);
        let can = match (&self.decoder, &self.config, next) {
            (Some(_), Some(current), Some(next)) => {
                next.codec == current.codec
                    && (next.width, next.height) == (current.width, current.height)
                    && read_record(&next).is_ok_and(|r| r.length_size == self.length_size)
            }
            _ => false,
        };
        let mut reply = self.reply(message);
        reply.s32(names::CAN_ACCEPT, i32::from(can));
        self.send(&reply.finish());
        Ok(())
    }

    /// `dsup`: the properties this decoder has, each with an empty description.
    fn supported_properties(&self, message: &Message<'_>) -> Handled {
        let supported = [USING_HARDWARE, REAL_TIME, MAXIMIZE_POWER_EFFICIENCY]
            .iter()
            .map(|&key| (key.to_owned(), Value::Dictionary(Vec::new())))
            .collect();
        let mut reply = self.reply(message);
        reply.s32(names::ERROR, status::NO_ERROR);
        reply.property_list(
            names::SUPPORTED_PROPERTIES,
            Some(&Value::Dictionary(supported)),
        );
        self.send(&reply.finish());
        Ok(())
    }

    /// `dcop`: a property's value. The key must be a string, or there is no reply.
    fn copy_property(&self, message: &Message<'_>) -> Handled {
        let key = property_key(message)?;
        let (status, value) = match key.as_str() {
            USING_HARDWARE => (status::NO_ERROR, Some(Value::Boolean(true))),
            REAL_TIME | MAXIMIZE_POWER_EFFICIENCY => (
                status::NO_ERROR,
                Some(
                    self.properties
                        .iter()
                        .find(|(found, _)| *found == key)
                        .map_or(Value::Boolean(false), |(_, value)| value.clone()),
                ),
            ),
            _ => (status::PROPERTY_NOT_SUPPORTED, None),
        };
        let mut reply = self.reply(message);
        reply.s32(names::ERROR, status);
        if let Some(value) = value {
            reply.property_list(names::PROPERTY_VALUE, Some(&value));
        }
        self.send(&reply.finish());
        Ok(())
    }

    /// `dset`: sets a property. The key must be a string and not `Paravirtualized`, or there is
    /// no reply.
    fn set_property(&mut self, message: &Message<'_>) -> Handled {
        let key = property_key(message)?;
        if key == PARAVIRTUALIZED {
            return Err(WireError::Malformed("a property the guest keeps"));
        }
        let value = match message.property_list_bytes(names::PROPERTY_VALUE)? {
            Some(bytes) if !bytes.is_empty() => Some(crate::paravirt::plist::decode(bytes)?),
            _ => None,
        };
        let status = match key.as_str() {
            REAL_TIME | MAXIMIZE_POWER_EFFICIENCY => {
                self.properties.retain(|(found, _)| *found != key);
                if let Some(value) = value {
                    self.properties.push((key, value));
                }
                status::NO_ERROR
            }
            USING_HARDWARE => status::PROPERTY_READ_ONLY,
            _ => status::PROPERTY_NOT_SUPPORTED,
        };
        self.reply_status(message, status)
    }

    /// `dmul`: `,pro` without `Paravirtualized` must not be empty, or there is no reply; the
    /// properties are not set together, and the guest then sets them one by one.
    fn set_properties(&self, message: &Message<'_>) -> Handled {
        let properties = match message.property_list_bytes(names::PROPERTIES)? {
            Some(bytes) if !bytes.is_empty() => crate::paravirt::plist::decode(bytes)?,
            _ => Value::Dictionary(Vec::new()),
        };
        let Value::Dictionary(entries) = properties else {
            return Err(WireError::Malformed("properties that are no dictionary"));
        };
        if !entries.iter().any(|(key, _)| key != PARAVIRTUALIZED) {
            return Err(WireError::Malformed("no properties for the decoder"));
        }
        self.reply_status(message, status::UNIMPLEMENTED)
    }
}

const NO_ERROR: OsStatus = status::NO_ERROR;

/// A message's `,key`, which must be a string.
fn property_key(message: &Message<'_>) -> Result<String, WireError> {
    match message.property_list_bytes(names::PROPERTY_KEY)? {
        Some(bytes) if !bytes.is_empty() => match crate::paravirt::plist::decode(bytes)? {
            Value::String(key) => Ok(key),
            _ => Err(WireError::Malformed("a property key that is no string")),
        },
        _ => Err(WireError::Malformed("no property key")),
    }
}

/// The decoder configuration of a `=fds` format description: its codec, size and configuration
/// record.
fn config_from_description(description: &ImageDescription) -> Option<DecoderConfig> {
    let codec = Codec::from_fourcc(description.codec)?;
    let record = match codec {
        Codec::H264 => AVC_CONFIGURATION,
        Codec::Hevc => HEVC_CONFIGURATION,
    };
    Some(DecoderConfig {
        codec,
        width: u32::from(description.width),
        height: u32::from(description.height),
        record: description.extension(record).unwrap_or_default().to_vec(),
    })
}

/// The decoder configuration of a sample buffer's format description.
fn config_from_format(format: &SerializedFormatDescription) -> Option<DecoderConfig> {
    let codec = Codec::from_fourcc(format.codec)?;
    let record = match codec {
        Codec::H264 => "avcC",
        Codec::Hevc => "hvcC",
    };
    Some(DecoderConfig {
        codec,
        width: u32::try_from(format.width).ok()?,
        height: u32::try_from(format.height).ok()?,
        record: format
            .sample_description_atom(record)
            .unwrap_or_default()
            .to_vec(),
    })
}

/// What the host reads of a configuration record.
struct Record {
    length_size: u8,
    /// Its parameter sets, Annex B.
    parameter_sets: Vec<u8>,
    /// Whether the luma samples have more than 8 bits.
    ten_bit: bool,
    /// Whether the samples use the full range.
    full_range: bool,
}

/// Reads `config`'s record, or the status a session on it is refused with: a record that does
/// not read, 3-byte NAL unit lengths, which ISO/IEC 14496-15 does not allow, and samples of more
/// than 10 bits, `kVTVideoDecoderUnsupportedDataFormatErr`; an H.264 SPS that lets pictures be
/// fields (`frame_mbs_only_flag` 0), `kVTVideoDecoderMalfunctionErr`, as no backend here decodes a
/// field.
fn read_record(config: &DecoderConfig) -> Result<Record, OsStatus> {
    let unsupported = status::VIDEO_DECODER_UNSUPPORTED_DATA_FORMAT;
    let (length_size, units, ten_bit, full_range) = match config.codec {
        Codec::H264 => {
            let record =
                avcc::DecoderConfiguration::parse(&config.record).map_err(|_| unsupported)?;
            let sps = record
                .sequence_parameter_sets
                .first()
                .ok_or(unsupported)
                .and_then(|unit| H264Sps::parse(unit).map_err(|_| unsupported))?;
            if !sps.frame_mbs_only_flag {
                return Err(status::VIDEO_DECODER_MALFUNCTION);
            }
            let units: Vec<&[u8]> = record
                .sequence_parameter_sets
                .iter()
                .chain(&record.picture_parameter_sets)
                .copied()
                .collect();
            let depth = sps.bit_depth_luma_minus8;
            (record.length_size, units, depth > 0, sps.is_full_range())
        }
        Codec::Hevc => {
            let record =
                hvcc::DecoderConfiguration::parse(&config.record).map_err(|_| unsupported)?;
            let sps = record
                .nal_units
                .iter()
                .find(|unit| {
                    unit.first()
                        .is_some_and(|b| (b >> 1) & 0x3f == nal_type::SPS)
                })
                .ok_or(unsupported)
                .and_then(|unit| HevcSps::parse(unit).map_err(|_| unsupported))?;
            let full = sps
                .vui
                .as_ref()
                .is_some_and(|vui| vui.video_full_range_flag);
            (
                record.length_size,
                record.nal_units,
                sps.bit_depth_luma_minus8 > 0,
                full,
            )
        }
    };
    if length_size == 3 || !matches!(length_size, 1 | 2 | 4) {
        return Err(unsupported);
    }
    if ten_bit && config.codec == Codec::H264 {
        return Err(unsupported);
    }
    let mut parameter_sets = Vec::new();
    for unit in units {
        parameter_sets.extend_from_slice(&START_CODE);
        parameter_sets.extend_from_slice(unit);
    }
    Ok(Record {
        length_size,
        parameter_sets,
        ten_bit,
        full_range,
    })
}

/// A sample's NAL units, each after a `length_size`-byte length, in Annex-B form; a sample whose
/// lengths run past it is `kVTVideoDecoderBadDataErr`.
fn annex_b(data: &[u8], length_size: u8) -> Result<Vec<u8>, OsStatus> {
    let mut out = Vec::with_capacity(data.len() + 16);
    for unit in LengthPrefixed::new(data, length_size) {
        let unit = unit.map_err(|_| status::VIDEO_DECODER_BAD_DATA)?;
        out.extend_from_slice(&START_CODE);
        out.extend_from_slice(unit);
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
