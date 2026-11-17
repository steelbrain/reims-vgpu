//! An encoder session (`+enc`): the messages of one guest encoder, on a thread of the session's
//! own ([`crate::paravirt::session`]), over the host's video arm ([`EncoderBackend`]).
//!
//! Each message is answered on the session's thread, in order: a reply (flag `REPLY_TO_GUEST`),
//! or for a message whose parameters do not read, nothing, and the guest's wait times out.
//!
//! - **The properties** (`esup`, `ecop`, `eset`, `emul`) are [`properties`]'. A set the running
//!   encoder cannot take ([`BackendEncoder::reconfigure`]) is refused with the backend's status,
//!   and the properties stay as they were. Past the first frame, the host's arms take a set that
//!   would make a new sequence (a profile, an entropy coder, frame reordering) and leave it out,
//!   the stream going on as it was while the property reads as set, as an M3's encoder does
//!   (macOS 27.0.1, measured). `eset` of `Paravirtualized`, a `,key` that is no
//!   string and an `emul` with nothing but `Paravirtualized` get no reply; `emul` sets each in
//!   turn and answers the first refusal.
//! - **`esta`** opens the backend's encoder at the source frames' size, `,vdi`. A side under 1
//!   is `kVTPVMalformedMessageErr`, a size past the encoder's
//!   [`status::ENCODER_SIZE_UNSUPPORTED`],
//!   each with no `Epba`. An odd side is taken: the stream is the frames' top left, each side
//!   rounded down to even ([`encoded_size`]), and a side of 1 is `kVTParameterErr`. An open
//!   encoder asks for its source buffers before the reply (`Epba`, which a guest of protocol 3
//!   or later answers within [`ATTRIBUTES_REPLY_WAIT`]): the video-range format of the profile
//!   it writes (`420v`, or `x420` for 10 bits), the frames' size, IOSurface backing, and rows and planes aligned to 64 bytes. A second
//!   `esta` first completes the frames the encoder before it holds.
//! - **`efrm`** encodes one frame: its one lent surface (with any other number there is no
//!   reply), its times and its frame properties (`EncoderForceKeyframe`). The frame waits until
//!   it is both encoded and emitted, then its surface goes (`rels`); a frame that does not
//!   encode goes at once. The reply carries the encode's status and info flags 0. With no
//!   encoder it answers `kVTInvalidSessionErr`.
//! - **Emission**: each sample the encoder gives goes to the guest in `Eemt` (waited for
//!   [`EMIT_REPLY_WAIT`]) with the frame's UUID, as a sample buffer of one sample: its data, the
//!   frame's own times and the decode time, the sync attachments, and the stream's format
//!   description on the first sample after it changes. A new configuration record first asks
//!   the guest to make the description (`Efds`, waited for [`FORMAT_REPLY_WAIT`]), which the
//!   samples carry as the guest made it. A sample buffer past the guest's
//!   [`HOST_MESSAGE_MAX`]-byte buffer goes in fragments: each but the last flagged
//!   `MORE_FRAGMENTS` with reply identifier 0, the last waited for. A frame the encoder dropped
//!   goes with `kVTEncodeInfo_FrameDropped` and no sample, one that failed with its status and
//!   no sample.
//! - **`ecfr`** emits every frame the encoder holds, then each frame it took and gave nothing
//!   back for with `kVTVideoEncoderMalfunctionErr`, then answers. **`einv`** drops the encoder
//!   and lets go of every frame's surface. `esav`, `emif`, the multi-pass and the tile messages
//!   answer `kVTUnimplementedErr`.
//!
//! The backend's clock runs at [`TICKS_PER_SECOND`]: each frame's times are converted onto it,
//! and each sample's are the guest's own again, as the frame came or, for a decode time, as the
//! frame whose presentation time it is came.

mod properties;

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Duration;

use crate::codec::encode::{
    Coded, EncodedSample, EncoderSettings, FailedSample, InputFormat, Profile,
};
use crate::core::fourcc::FourCc;
use crate::core::runtime::Runtime;
use crate::paravirt::host::Codec;
use crate::paravirt::message::{
    GuestUuid, MediaTime, Message, MessageFlags, MessageWriter, ValueType,
};
use crate::paravirt::plist::{self, Value};
use crate::paravirt::sample_buffer::{
    MEDIA_TYPE_VIDEO, SampleBuffer, SampleTiming, SerializedFormatDescription, SerializedValue,
};
use crate::paravirt::session::{Job, LentSurface, Link, Session, Shared};
use crate::paravirt::status::{self, OsStatus};
use crate::paravirt::{WireError, names};

use self::properties::{PARAVIRTUALIZED, Properties};

/// How long an `Epba` waits for the guest's answer.
pub const ATTRIBUTES_REPLY_WAIT: Duration = Duration::from_millis(1100);
/// How long an `Efds` waits for the guest's format description.
pub const FORMAT_REPLY_WAIT: Duration = Duration::from_millis(1100);
/// How long an `Eemt` waits for the guest's answer.
pub const EMIT_REPLY_WAIT: Duration = Duration::from_secs(10);

/// The longest message the guest's VideoToolbox takes from its host: the 16 KiB it reads each
/// one into.
pub const HOST_MESSAGE_MAX: usize = 16 << 10;

/// Ticks a second of the times a backend is given: ours, the 90 kHz clock of MPEG streams.
pub const TICKS_PER_SECOND: u32 = 90_000;

/// The first guest protocol that answers `Epba`.
const EPBA_REPLY_PROTOCOL: u32 = 3;

/// The alignment of the rows and planes of the source buffers `Epba` asks for.
const SOURCE_ALIGNMENT: i64 = 64;

/// `kVTEncodeFrameOptionKey_ForceKeyFrame`'s value.
const FORCE_KEY_FRAME: &str = "EncoderForceKeyframe";

/// `kVTEncodeInfo_FrameDropped`.
const INFO_FRAME_DROPPED: u32 = 1 << 1;

/// The presentation times the session remembers, to give a sample's decode time back as the
/// guest's own: ours, past the frames any encoder holds back.
const TIMES_KEPT: usize = 64;

/// The thread a session's messages run on.
const THREAD_NAME: &str = "reims-vgpu-video.vt-encoder";

/// An encoder the host's video arm runs, as the encoder list names it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Encoder {
    /// Its codec.
    pub codec: Codec,
    /// The profiles it writes, of that codec.
    pub profiles: Vec<Profile>,
    /// The widest source frame it takes.
    pub max_width: u32,
    /// The tallest source frame it takes.
    pub max_height: u32,
}

/// What an encoder is opened and reconfigured with.
#[derive(Clone, Debug, PartialEq)]
pub struct EncoderConfig {
    /// The settings the session's properties make, at the stream's size, on a clock of
    /// [`TICKS_PER_SECOND`].
    pub settings: EncoderSettings,
    /// `Quality`, 0 to 1, where the guest set it, for the backend to map onto its driver's
    /// quality levels.
    pub quality: Option<f32>,
}

/// One source frame, in display order, as a backend takes it with its surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceFrame {
    /// The frame's tag, which its sample gives back.
    pub id: u64,
    /// Its presentation time, in ticks of [`TICKS_PER_SECOND`], later than the last frame's.
    pub pts: i64,
    /// Its duration in ticks, 0 where unknown.
    pub duration: i64,
    /// Whether the frame must be a key frame.
    pub force_key_frame: bool,
}

/// The host's video arm, as the encoder sessions take it.
pub trait EncoderBackend: Send + Sync {
    /// An encoder of `config`, on the session's thread at its `esta`, or the status the guest
    /// sees.
    fn open(&self, config: &EncoderConfig) -> Result<Box<dyn BackendEncoder>, OsStatus>;
}

/// One open encoder, called on its session's thread, one call at a time.
pub trait BackendEncoder: Send {
    /// Encodes `frame` from `source`, a surface of the stream's size or larger, whose top left
    /// is the picture, and returns the samples that came out, in decode order, or the status the
    /// guest sees for a frame that does not encode.
    fn encode(
        &mut self,
        source: &dyn LentSurface,
        frame: &SourceFrame,
    ) -> Result<Vec<Coded<OsStatus>>, OsStatus>;
    /// Encodes every frame it holds and returns their samples.
    fn flush(&mut self) -> Result<Vec<Coded<OsStatus>>, OsStatus>;
    /// Takes `config` from the next frame on, or the status a property set that asked for it is
    /// refused with.
    fn reconfigure(&mut self, config: &EncoderConfig) -> Result<(), OsStatus>;
}

/// The size a stream of `width` by `height` source frames is encoded at: each side rounded down
/// to even, the frames' last column or row left out, as Apple's hardware encoder does (measured
/// on macOS 27, 2026-10-24: 641x361 frames made a 640x360 stream of their top left). A 4:2:0
/// stream crops in pairs of luma samples.
pub fn encoded_size(width: u32, height: u32) -> (u32, u32) {
    (width & !1, height & !1)
}

/// Starts an encoder session for `uuid` on `encoder`, its thread on `runtime`, encoding on
/// `backend`; `guest_version` is the guest's protocol, from its `+enc`.
pub(crate) fn start(
    uuid: GuestUuid,
    encoder: Encoder,
    guest_version: u32,
    backend: Arc<dyn EncoderBackend>,
    runtime: Arc<dyn Runtime>,
    link: Option<Arc<dyn Link>>,
) -> std::io::Result<Session> {
    let thread_runtime = Arc::clone(&runtime);
    Session::start(uuid, THREAD_NAME, &*runtime, link, move |shared, jobs| {
        Worker {
            shared,
            jobs,
            runtime: thread_runtime,
            backend,
            properties: Properties::new(encoder.codec, encoder.profiles.clone()),
            encoder,
            guest_version,
            running: None,
            dimensions: None,
            pending: BTreeMap::new(),
            times: VecDeque::new(),
            last_pts: None,
            format: None,
        }
        .run();
    })
}

/// A frame the guest handed the host, until it has been both encoded and emitted.
struct Pending {
    uuid: Option<GuestUuid>,
    surface: Arc<dyn LentSurface>,
    pts: MediaTime,
    duration: MediaTime,
    submitted: bool,
    emitted: bool,
}

/// The format description the samples carry, and the configuration record it was made of.
struct StreamFormat {
    record: Arc<[u8]>,
    description: SerializedFormatDescription,
    /// Whether a sample carried it yet.
    written: bool,
}

type Handled = Result<(), WireError>;

/// The session's thread and what only it touches.
struct Worker {
    shared: Arc<Shared>,
    jobs: Receiver<Job>,
    runtime: Arc<dyn Runtime>,
    backend: Arc<dyn EncoderBackend>,
    encoder: Encoder,
    guest_version: u32,
    properties: Properties,
    running: Option<Box<dyn BackendEncoder>>,
    /// `esta`'s size: the source frames'.
    dimensions: Option<(u32, u32)>,
    pending: BTreeMap<i64, Pending>,
    /// The last [`TIMES_KEPT`] frames' presentation times, in ticks and as the guest gave them.
    times: VecDeque<(i64, MediaTime)>,
    /// The last frame's presentation time in ticks.
    last_pts: Option<i64>,
    format: Option<StreamFormat>,
}

impl Worker {
    fn run(mut self) {
        while let Ok(job) = self.jobs.recv() {
            let mut surfaces = job.surfaces;
            if let Err(error) = self.handle(&job.bytes, &mut surfaces) {
                log::warn!("hardware video: a VideoToolbox message the host cannot read: {error}");
            }
            // What a message lent and the session did not keep goes back.
            self.shared.let_go(&surfaces);
        }
        self.let_go_of_pending();
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

    fn handle(&mut self, bytes: &[u8], surfaces: &mut Vec<Arc<dyn LentSurface>>) -> Handled {
        let message = Message::parse(bytes)?;
        let kind = message.header().message_type;
        match &kind.to_be_bytes() {
            b"esup" => {
                let mut reply = self.reply(&message);
                reply.s32(names::ERROR, status::NO_ERROR);
                reply.property_list(
                    names::SUPPORTED_PROPERTIES,
                    Some(&self.properties.supported()),
                );
                self.send(&reply.finish());
                Ok(())
            }
            b"ecop" => self.copy_property(&message),
            b"eset" => self.set_property(&message),
            b"emul" => self.set_properties(&message),
            b"esta" => self.start_session(&message),
            b"epfr" => self.reply_status(&message, status::NO_ERROR),
            b"efrm" => self.encode_frame(&message, surfaces),
            b"ecfr" => {
                message.time(names::PRESENTATION_TIME)?;
                let status = self.complete_held_frames();
                self.reply_status(&message, status)
            }
            b"einv" => {
                self.running = None;
                self.let_go_of_pending();
                self.reply_status(&message, status::NO_ERROR)
            }
            b"emif" => {
                let mut reply = self.reply(&message);
                reply.s32(names::ERROR, status::UNIMPLEMENTED);
                reply.u32(names::INFO_FLAGS, 0);
                self.send(&reply.finish());
                Ok(())
            }
            b"eeps" => {
                let mut reply = self.reply(&message);
                reply.s32(names::FRAME_PROPERTIES, 0);
                reply.s32(names::ERROR, status::UNIMPLEMENTED);
                self.send(&reply.finish());
                Ok(())
            }
            b"esav" | b"ebps" | b"ests" | b"epti" | b"etil" | b"ecti" => {
                self.reply_status(&message, status::UNIMPLEMENTED)
            }
            _ => {
                log::warn!("hardware video: an unknown encoder message {kind}");
                self.reply_status(&message, status::UNIMPLEMENTED)
            }
        }
    }

    /// `ecop`: a property's value.
    fn copy_property(&self, message: &Message<'_>) -> Handled {
        let key = property_key(message)?;
        let (status, value) = self.properties.copy(&key);
        let mut reply = self.reply(message);
        reply.s32(names::ERROR, status);
        if let Some(value) = value {
            reply.property_list(names::PROPERTY_VALUE, Some(&value));
        }
        self.send(&reply.finish());
        Ok(())
    }

    /// `eset`: sets a property.
    fn set_property(&mut self, message: &Message<'_>) -> Handled {
        let key = property_key(message)?;
        if key == PARAVIRTUALIZED {
            return Err(WireError::Malformed("a property the guest keeps"));
        }
        let value = property_list(message, names::PROPERTY_VALUE)?;
        let status = self.set(&key, value);
        self.reply_status(message, status)
    }

    /// `emul`: sets each property in turn.
    fn set_properties(&mut self, message: &Message<'_>) -> Handled {
        let properties = match property_list(message, names::PROPERTIES)? {
            Some(Value::Dictionary(entries)) => entries,
            Some(_) => return Err(WireError::Malformed("properties that are no dictionary")),
            None => Vec::new(),
        };
        let properties: Vec<(String, Value)> = (properties.into_iter())
            .filter(|(key, _)| key != PARAVIRTUALIZED)
            .collect();
        if properties.is_empty() {
            return Err(WireError::Malformed("no properties for the encoder"));
        }
        let mut status = status::NO_ERROR;
        for (key, value) in properties {
            let set = self.set(&key, Some(value));
            if status == status::NO_ERROR {
                status = set;
            }
        }
        self.reply_status(message, status)
    }

    /// Sets one property, and hands a running encoder the configuration it makes.
    fn set(&mut self, key: &str, value: Option<Value>) -> OsStatus {
        let earlier = self.properties.snapshot();
        let status = self.properties.set(key, value);
        if status != status::NO_ERROR {
            return status;
        }
        let (Some(encoder), Some((width, height))) = (self.running.as_mut(), self.dimensions)
        else {
            return status::NO_ERROR;
        };
        let (width, height) = encoded_size(width, height);
        let refused = match self.properties.config(width, height) {
            Some(config) => encoder.reconfigure(&config).err(),
            None => Some(status::PROPERTY_NOT_SUPPORTED),
        };
        match refused {
            None => status::NO_ERROR,
            Some(status) => {
                log::warn!("hardware video: the encoder cannot take {key} now ({status})");
                self.properties.restore(earlier);
                status
            }
        }
    }

    /// `esta`: opens the encoder (the module doc).
    fn start_session(&mut self, message: &Message<'_>) -> Handled {
        let (width, height) =
            (message.i32_pair(names::DIMENSIONS, ValueType::Dimensions)?).unwrap_or((0, 0));
        let (Ok(width @ 1..), Ok(height @ 1..)) = (u32::try_from(width), u32::try_from(height))
        else {
            return self.reply_status(message, status::MALFORMED_MESSAGE);
        };
        let status = if width > self.encoder.max_width || height > self.encoder.max_height {
            status::ENCODER_SIZE_UNSUPPORTED
        } else {
            match encoded_size(width, height) {
                (0, _) | (_, 0) => status::PARAMETER,
                (w, h) => match self.properties.config(w, h) {
                    Some(config) => self.open(&config, width, height),
                    None => status::PARAMETER,
                },
            }
        };
        self.reply_status(message, status)
    }

    /// Opens an encoder of `config` for `width` by `height` frames, completing the frames the
    /// one before it holds, and asks for its source buffers.
    fn open(&mut self, config: &EncoderConfig, width: u32, height: u32) -> OsStatus {
        if self.running.is_some() {
            let completed = self.complete_held_frames();
            if completed != status::NO_ERROR {
                log::warn!(
                    "hardware video: the encoder before this start completed with {completed}"
                );
            }
        }
        match self.backend.open(config) {
            Ok(encoder) => {
                self.running = Some(encoder);
                self.dimensions = Some((width, height));
                self.format = None;
                self.send_source_attributes(config.settings.input_format(), width, height)
            }
            Err(status) => {
                self.running = None;
                status
            }
        }
    }

    /// `Epba`: the attributes the guest's source buffers are to have.
    fn send_source_attributes(&self, format: InputFormat, width: u32, height: u32) -> OsStatus {
        // Video range alone: the guest's session converts a full-range source to it, and the
        // stream's VUI says video range.
        let format = match format {
            InputFormat::Nv12 => *b"420v",
            InputFormat::P010 => *b"x420",
        };
        let formats = vec![Value::Integer(i64::from(u32::from_be_bytes(format)))];
        let attributes = Value::dictionary([
            ("PixelFormatType", Value::Array(formats)),
            ("Width", Value::Integer(i64::from(width))),
            ("Height", Value::Integer(i64::from(height))),
            ("IOSurfaceProperties", Value::Dictionary(Vec::new())),
            ("BytesPerRowAlignment", Value::Integer(SOURCE_ALIGNMENT)),
            ("PlaneAlignment", Value::Integer(SOURCE_ALIGNMENT)),
        ]);
        let fill = |message: &mut MessageWriter| {
            message.property_list(names::PIXEL_BUFFER_ATTRIBUTES, Some(&attributes));
        };
        if self.guest_version < EPBA_REPLY_PROTOCOL {
            let mut message = MessageWriter::new(
                names::SET_SOURCE_PIXEL_BUFFER_ATTRIBUTES,
                MessageFlags::NONE,
                Some(self.shared.uuid),
                0,
            );
            fill(&mut message);
            self.send(&message.finish());
            return status::NO_ERROR;
        }
        let (id, message) = self
            .shared
            .request_message(names::SET_SOURCE_PIXEL_BUFFER_ATTRIBUTES, fill);
        let reply = (self.shared).request(
            &*self.runtime,
            id,
            &[],
            &message.finish(),
            ATTRIBUTES_REPLY_WAIT,
        );
        match reply {
            Ok(reply) => {
                self.shared.let_go(&reply.surfaces);
                reply_error(&reply.bytes)
            }
            Err(status) => status,
        }
    }

    /// `efrm`: one source frame (the module doc).
    fn encode_frame(
        &mut self,
        message: &Message<'_>,
        surfaces: &mut Vec<Arc<dyn LentSurface>>,
    ) -> Handled {
        let frame = message.s64(names::FRAME)?.unwrap_or(0);
        let pts = message
            .time(names::PRESENTATION_TIME)?
            .unwrap_or(MediaTime::INVALID);
        let duration = message.time(names::DURATION)?.unwrap_or(MediaTime::INVALID);
        let frame_properties = property_list(message, names::FRAME_PROPERTIES)?;
        let uuid = message.uuid(names::PIXEL_BUFFER_UUID)?;
        message.property_list_bytes(names::PIXEL_BUFFER_ATTACHMENTS)?;
        let [_] = surfaces.as_slice() else {
            return Err(WireError::Malformed("a frame without its one surface"));
        };
        let Some(surface) = surfaces.pop() else {
            return Err(WireError::Malformed("a frame without its one surface"));
        };
        let force_key_frame = (frame_properties.as_ref())
            .and_then(|properties| properties.get(FORCE_KEY_FRAME))
            .and_then(Value::as_boolean)
            .unwrap_or(false);
        if self.running.is_none() {
            self.shared.let_go(&[surface]);
            let mut reply = self.reply(message);
            reply.s32(names::ERROR, status::INVALID_SESSION);
            reply.u32(names::INFO_FLAGS, 0);
            self.send(&reply.finish());
            return Ok(());
        }
        // Each frame later than the last on the backend's clock, as its encoder takes them.
        let ticks = to_ticks(pts).unwrap_or(0);
        let ticks = match self.last_pts {
            Some(last) if ticks <= last => last.saturating_add(1),
            _ => ticks,
        };
        self.last_pts = Some(ticks);
        if self.times.len() == TIMES_KEPT {
            self.times.pop_front();
        }
        self.times.push_back((ticks, pts));
        let source = SourceFrame {
            id: frame.cast_unsigned(),
            pts: ticks,
            duration: to_ticks(duration).unwrap_or(0).max(0),
            force_key_frame,
        };
        let encoded = match self.running.as_mut() {
            Some(encoder) => encoder.encode(&*surface, &source),
            None => Err(status::INVALID_SESSION),
        };
        self.pending.insert(
            frame,
            Pending {
                uuid,
                surface,
                pts,
                duration,
                submitted: false,
                emitted: false,
            },
        );
        let status = match encoded {
            Ok(coded) => {
                for coded in coded {
                    self.emit(coded);
                }
                status::NO_ERROR
            }
            Err(status) => status,
        };
        let mut reply = self.reply(message);
        reply.s32(names::ERROR, status);
        reply.u32(names::INFO_FLAGS, 0);
        self.send(&reply.finish());
        self.frame_submitted(frame, status != status::NO_ERROR);
        Ok(())
    }

    /// The encode of `frame` returned: the frame goes where it was emitted too, or at once
    /// where the encode `failed`.
    fn frame_submitted(&mut self, frame: i64, failed: bool) {
        if let Some(pending) = self.pending.get_mut(&frame) {
            pending.submitted = true;
            if (pending.emitted || failed)
                && let Some(done) = self.pending.remove(&frame)
            {
                self.shared.let_go(&[done.surface]);
            }
        }
    }

    /// `frame` is being emitted: its pending entry, marked, and taken where its encode
    /// returned, for the caller to let go of once the emission is answered.
    fn frame_emitted(&mut self, frame: i64) -> (Option<GuestUuid>, Option<Pending>) {
        let Some(pending) = self.pending.get_mut(&frame) else {
            return (None, None);
        };
        pending.emitted = true;
        let uuid = pending.uuid;
        let taken = if pending.submitted {
            self.pending.remove(&frame)
        } else {
            None
        };
        (uuid, taken)
    }

    /// Lets go of every pending frame's surface, in one `rels`.
    fn let_go_of_pending(&mut self) {
        let surfaces: Vec<_> = (std::mem::take(&mut self.pending).into_values())
            .map(|pending| pending.surface)
            .collect();
        self.shared.let_go(&surfaces);
    }

    /// Emits every frame the encoder holds, then each one it took and gave nothing back for
    /// (the module doc); the flush's status.
    fn complete_held_frames(&mut self) -> OsStatus {
        let flushed = self.running.as_mut().map(|encoder| encoder.flush());
        let status = match flushed {
            None => status::NO_ERROR,
            Some(Ok(coded)) => {
                for coded in coded {
                    self.emit(coded);
                }
                status::NO_ERROR
            }
            Some(Err(status)) => status,
        };
        let lost: Vec<i64> = (self.pending.iter())
            .filter(|(_, pending)| pending.submitted && !pending.emitted)
            .map(|(&frame, _)| frame)
            .collect();
        let mut surfaces = Vec::with_capacity(lost.len());
        for frame in lost {
            log::warn!("hardware video: frame {frame} was lost in the encoder");
            let (uuid, taken) = self.frame_emitted(frame);
            self.emit_encoded_frame(frame, status::VIDEO_ENCODER_MALFUNCTION, 0, uuid, None);
            surfaces.extend(taken.map(|pending| pending.surface));
        }
        self.shared.let_go(&surfaces);
        status
    }

    /// Emits one of the encoder's results (the module doc).
    fn emit(&mut self, coded: Coded<OsStatus>) {
        let sample = match coded {
            Ok(sample) if !sample.data.is_empty() => sample,
            Ok(sample) => {
                return self.emit_unsampled(sample.id, status::NO_ERROR, INFO_FRAME_DROPPED);
            }
            Err(FailedSample { id, error }) => {
                let status = error.unwrap_or(status::VIDEO_ENCODER_MALFUNCTION);
                return self.emit_unsampled(id, status, 0);
            }
        };
        let frame = sample.id.cast_signed();
        let serialization = self.sample_buffer(&sample).map(|buffer| buffer.serialize());
        let (uuid, taken) = self.frame_emitted(frame);
        match serialization {
            Ok(bytes) => self.emit_encoded_frame(frame, status::NO_ERROR, 0, uuid, Some(&bytes)),
            Err(status) => self.emit_encoded_frame(frame, status, 0, uuid, None),
        }
        self.shared.let_go(
            &taken
                .map(|pending| pending.surface)
                .into_iter()
                .collect::<Vec<_>>(),
        );
    }

    /// Emits frame `id` with `status`, `info` and no sample.
    fn emit_unsampled(&mut self, id: u64, status: OsStatus, info: u32) {
        let frame = id.cast_signed();
        let (uuid, taken) = self.frame_emitted(frame);
        self.emit_encoded_frame(frame, status, info, uuid, None);
        self.shared.let_go(
            &taken
                .map(|pending| pending.surface)
                .into_iter()
                .collect::<Vec<_>>(),
        );
    }

    /// `sample` as a sample buffer of one sample, in the guest's times, with the stream's format
    /// description where it is new; the status where the stream has none.
    fn sample_buffer(&mut self, sample: &EncodedSample) -> Result<SampleBuffer, OsStatus> {
        if let Some(record) = &sample.configuration
            && self
                .format
                .as_ref()
                .is_none_or(|format| format.record != *record)
        {
            let description = self.create_video_format_description(record)?;
            self.format = Some(StreamFormat {
                record: Arc::clone(record),
                description,
                written: false,
            });
        }
        let format = self
            .format
            .as_mut()
            .ok_or(status::VIDEO_ENCODER_MALFUNCTION)?;
        let description = (!format.written).then(|| format.description.clone());
        format.written = true;
        let (pts, duration) = match self.pending.get(&sample.id.cast_signed()) {
            Some(pending) => (pending.pts, pending.duration),
            None => (MediaTime::INVALID, MediaTime::INVALID),
        };
        let decode = (self.times.iter())
            .find(|(ticks, _)| *ticks == sample.dts)
            .map_or_else(|| from_ticks(sample.dts, pts.timescale), |(_, time)| *time);
        let flag = |value| SerializedValue::PropertyList(Value::Boolean(value));
        let mut attachments = Vec::with_capacity(3);
        if !sample.is_sync {
            attachments.push(("NotSync".to_owned(), flag(true)));
        }
        attachments.push(("DependsOnOthers".to_owned(), flag(sample.depends_on_others)));
        attachments.push(("EarlierDisplayTimesAllowed".to_owned(), flag(false)));
        Ok(SampleBuffer {
            sample_count: 1,
            timings: vec![SampleTiming {
                duration,
                presentation: pts,
                decode,
            }],
            sizes: vec![u32::try_from(sample.data.len()).unwrap_or(u32::MAX)],
            data: Some(sample.data.clone()),
            output_presentation_time: Some(pts),
            format_description: description,
            propagated_attachments: None,
            other_attachments: None,
            sample_attachments: Some(vec![SerializedValue::Dictionary(attachments)]),
        })
    }

    /// `Efds`: asks the guest to make the stream's format description of configuration record
    /// `record`, and reads it into the description the samples carry.
    fn create_video_format_description(
        &self,
        record: &[u8],
    ) -> Result<SerializedFormatDescription, OsStatus> {
        let (width, height) = (self.dimensions)
            .map(|(w, h)| encoded_size(w, h))
            .ok_or(status::INVALID_SESSION)?;
        let codec = self.encoder.codec;
        let atom = configuration_atom(codec);
        let mut extensions = vec![
            ("FormatName".to_owned(), Value::string(codec_name(codec))),
            (
                "SampleDescriptionExtensionAtoms".to_owned(),
                Value::dictionary([(atom, Value::Data(record.to_vec()))]),
            ),
        ];
        for (key, value) in self.properties.colour_extensions() {
            extensions.push((key.to_owned(), Value::String(value)));
        }
        let extensions = Value::Dictionary(extensions);
        let (id, message) =
            self.shared
                .request_message(names::CREATE_VIDEO_FORMAT_DESCRIPTION, |m| {
                    m.s32(names::CODEC, codec.fourcc().0.cast_signed());
                    m.s32(names::WIDTH, i32::try_from(width).unwrap_or(i32::MAX));
                    m.s32(names::HEIGHT, i32::try_from(height).unwrap_or(i32::MAX));
                    m.property_list(names::EXTENSIONS, Some(&extensions));
                });
        let reply = (self.shared).request(
            &*self.runtime,
            id,
            &[],
            &message.finish(),
            FORMAT_REPLY_WAIT,
        )?;
        self.shared.let_go(&reply.surfaces);
        let read = |error: WireError| error.status();
        let message = Message::parse(&reply.bytes).map_err(read)?;
        let status = message.s32(names::ERROR).map_err(read)?.unwrap_or(0);
        if status != status::NO_ERROR {
            return Err(status);
        }
        let image = (message.image_description(names::CREATED_FORMAT_DESCRIPTION))
            .map_err(read)?
            .ok_or(status::VIDEO_ENCODER_MALFUNCTION)?;
        Ok(serialized_description(
            codec,
            width,
            height,
            record,
            image,
            &extensions,
        ))
    }

    /// `Eemt`: frame `frame`'s status, info flags, UUID and sample buffer, in fragments where
    /// it is long (the module doc); the guest's answer is waited for and not read.
    fn emit_encoded_frame(
        &self,
        frame: i64,
        status: OsStatus,
        info: u32,
        uuid: Option<GuestUuid>,
        serialization: Option<&[u8]>,
    ) {
        let (id, mut message) = self.shared.request_message(names::EMIT_ENCODED_FRAME, |m| {
            m.s64(names::FRAME, frame);
            m.s32(names::ERROR, status);
            m.u32(names::ENCODE_INFO_FLAGS, info);
            if let Some(uuid) = uuid {
                m.uuid(names::PIXEL_BUFFER_UUID, uuid);
            }
        });
        let mut before = Vec::new();
        if let Some(mut rest) = serialization {
            rest = message.sample_buffer_with_limit(names::SAMPLE_BUFFER, HOST_MESSAGE_MAX, rest);
            while !rest.is_empty() {
                let more = MessageFlags::MORE_FRAGMENTS;
                let mask = MessageFlags(more.0 | MessageFlags::HOST_EXPECTS_REPLY.0);
                message.change_flags(more, mask).set_reply_id(0);
                before.push(message.finish());
                message = MessageWriter::new(
                    names::EMIT_ENCODED_FRAME,
                    MessageFlags::HOST_EXPECTS_REPLY,
                    Some(self.shared.uuid),
                    id,
                );
                let left = rest.len();
                rest =
                    message.sample_buffer_with_limit(names::SAMPLE_BUFFER, HOST_MESSAGE_MAX, rest);
                if rest.len() == left {
                    log::warn!("hardware video: frame {frame}'s sample does not fit a message");
                    break;
                }
            }
        }
        let sent = (self.shared).request(
            &*self.runtime,
            id,
            &before,
            &message.finish(),
            EMIT_REPLY_WAIT,
        );
        if let Ok(reply) = sent {
            self.shared.let_go(&reply.surfaces);
        }
    }
}

/// A reply's `,err`: what a reply without one, or one that does not read, answers too.
fn reply_error(bytes: &[u8]) -> OsStatus {
    match Message::parse(bytes).and_then(|message| message.s32(names::ERROR)) {
        Ok(status) => status.unwrap_or(status::NO_ERROR),
        Err(error) => error.status(),
    }
}

/// A message's property list `key`: none where it is absent or empty (NULL).
fn property_list(message: &Message<'_>, key: FourCc) -> Result<Option<Value>, WireError> {
    match message.property_list_bytes(key)? {
        Some(bytes) if !bytes.is_empty() => Ok(Some(plist::decode(bytes)?)),
        _ => Ok(None),
    }
}

/// A message's `,key`, which must be a string.
fn property_key(message: &Message<'_>) -> Result<String, WireError> {
    match property_list(message, names::PROPERTY_KEY)? {
        Some(Value::String(key)) => Ok(key),
        Some(_) => Err(WireError::Malformed("a property key that is no string")),
        None => Err(WireError::Malformed("no property key")),
    }
}

/// `time` in ticks of [`TICKS_PER_SECOND`], to the nearest; `None` for an invalid time.
fn to_ticks(time: MediaTime) -> Option<i64> {
    if !time.is_valid() || time.timescale <= 0 {
        return None;
    }
    let scaled = i128::from(time.value) * i128::from(TICKS_PER_SECOND);
    let timescale = i128::from(time.timescale);
    let rounded = (scaled + scaled.signum() * timescale / 2) / timescale;
    i64::try_from(rounded).ok()
}

/// `ticks` of [`TICKS_PER_SECOND`] as a time of `timescale`, where it is positive; else on the
/// ticks' own clock.
fn from_ticks(ticks: i64, timescale: i32) -> MediaTime {
    let (value, timescale) = if timescale > 0 {
        let scaled = i128::from(ticks) * i128::from(timescale) / i128::from(TICKS_PER_SECOND);
        (i64::try_from(scaled).unwrap_or(ticks), timescale)
    } else {
        (ticks, i32::try_from(TICKS_PER_SECOND).unwrap_or(i32::MAX))
    };
    MediaTime {
        value,
        timescale,
        flags: MediaTime::FLAG_VALID,
        epoch: 0,
    }
}

/// The extension atom a codec's configuration record goes in.
fn configuration_atom(codec: Codec) -> &'static str {
    match codec {
        Codec::H264 => "avcC",
        Codec::Hevc => "hvcC",
    }
}

/// The codec's `FormatName`.
fn codec_name(codec: Codec) -> &'static str {
    match codec {
        Codec::H264 => "H.264",
        Codec::Hevc => "HEVC",
    }
}

/// The format description the samples carry: what CoreMedia makes of the guest's
/// ImageDescription `image`, the extensions sent in `Efds` and what the description's header
/// says (`Version`, `RevisionLevel`, the qualities, `Depth`, `VerbatimSampleDescription`), a
/// progressive frame of square pixels.
fn serialized_description(
    codec: Codec,
    width: u32,
    height: u32,
    record: &[u8],
    image: &[u8],
    sent: &Value,
) -> SerializedFormatDescription {
    let leaf = SerializedValue::PropertyList;
    let field16 = |offset: usize| {
        (image.get(offset..))
            .and_then(<[u8]>::first_chunk)
            .map_or(0, |bytes| i64::from(u16::from_be_bytes(*bytes)))
    };
    let field32 = |offset: usize| {
        (image.get(offset..))
            .and_then(<[u8]>::first_chunk)
            .map_or(0, |bytes| i64::from(u32::from_be_bytes(*bytes)))
    };
    let mut extensions = vec![
        (
            "FormatName".to_owned(),
            leaf(Value::string(codec_name(codec))),
        ),
        ("Version".to_owned(), leaf(Value::Integer(field16(16)))),
        (
            "RevisionLevel".to_owned(),
            leaf(Value::Integer(field16(18))),
        ),
        (
            "TemporalQuality".to_owned(),
            leaf(Value::Integer(field32(24))),
        ),
        (
            "SpatialQuality".to_owned(),
            leaf(Value::Integer(field32(28))),
        ),
        ("Depth".to_owned(), leaf(Value::Integer(field16(82)))),
        (
            "VerbatimSampleDescription".to_owned(),
            leaf(Value::Data(image.to_vec())),
        ),
        (
            "CVPixelAspectRatio".to_owned(),
            SerializedValue::Dictionary(vec![
                ("HorizontalSpacing".to_owned(), leaf(Value::Integer(1))),
                ("VerticalSpacing".to_owned(), leaf(Value::Integer(1))),
            ]),
        ),
        ("CVFieldCount".to_owned(), leaf(Value::Integer(1))),
        (
            "SampleDescriptionExtensionAtoms".to_owned(),
            SerializedValue::Dictionary(vec![(
                configuration_atom(codec).to_owned(),
                leaf(Value::Data(record.to_vec())),
            )]),
        ),
    ];
    if let Value::Dictionary(sent) = sent {
        for (key, value) in sent {
            if key.starts_with("CVImageBuffer") {
                extensions.push((key.clone(), leaf(value.clone())));
            }
        }
    }
    SerializedFormatDescription {
        media_type: MEDIA_TYPE_VIDEO,
        codec: codec.fourcc(),
        width: i32::try_from(width).unwrap_or(i32::MAX),
        height: i32::try_from(height).unwrap_or(i32::MAX),
        extensions: Some(extensions),
    }
}

#[cfg(test)]
mod tests;
