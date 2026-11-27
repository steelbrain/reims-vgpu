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
//!   size and the configuration record (`avcC`, `hvcC`). A record of 3-byte NAL unit lengths, and
//!   a stream whose pictures the host decodes none of, are refused with
//!   `kVTVideoDecoderUnsupportedDataFormatErr`, an H.264 SPS that lets pictures be fields with
//!   `kVTVideoDecoderMalfunctionErr`, and the backend refuses what it does not decode, each with
//!   no `Dpba`. The host decodes 4:2:0 pictures at 8 bits, HEVC's at 10 too, and HEVC's range
//!   extensions pictures (4:2:0, 4:2:2 and 4:4:4 at 8, 10 and 12 bits) where the arm does, in
//!   the profiles and tools the codec's `check_supported` takes, which every arm's decode of a
//!   picture checks: a stream of monochrome pictures (a HEIC image's alpha), of deeper samples,
//!   or of a chroma format and depth the arm declines ([`Backend::declines`], from its own
//!   query of the host's decoder; a HEIC image's 4:4:4 thumbnail where the arm decodes no 4:4:4)
//!   is refused at its start, not at each frame, with a warning that names why, and the stream's
//!   profile, chroma format, bit depths and size ([`Undecoded`]). A started session sends `Dpba`,
//!   the attributes the guest's output buffers are to have: the two-plane CoreVideo format of its
//!   chroma, depth and range that the guest's own software decoder lands the stream in
//!   ([`PixelFormat::of`]: `420v`, `x420`, `tv20`, `422v`, `x422`, `tv22`, `444v`, `x444`, `tv44`
//!   and their full-range twins), before its reply. A second `dsta` is answered
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
//! - **Tiles** (a `VTTileDecompressionSession`'s, such as CMPhoto's for a HEIC image's grid):
//!   `dsts` starts a tile session on its format description (`,fds`) where the backend decodes
//!   tiles into regions ([`BackendDecoder::tile_limits`]), sending `TDrs` first: the attributes
//!   the guest's output buffers are to have (`,cba`) and the tiles it decodes (`,trd`); else it
//!   answers `kVTUnimplementedErr`, and the guest's VideoToolbox decodes the image in software.
//!   It answers the same for a stream whose pictures the host decodes none of (as `dsta` refuses
//!   it), so a session's tiles land only in the format `TDrs` names, laid out as the guest lent
//!   each buffer.
//!   `dtil` decodes one tile's sample (`,sbf`, in fragments as `dfrm`'s) into the one surface it
//!   lends: the `,tcd` columns and rows of the tile's picture from `,tco` land at `,obp` in it
//!   ([`TileRegion`]). The host emits the tile (`TDem`: its `,til`, status, info flags and the
//!   buffer's `,uiu`, waiting [`EMIT_REPLY_WAIT`]) before the reply, which says noErr, and lets
//!   go of the surface. Tiles decode in order, one a `dtil`, so `dtfi` has none left to finish.
//! - **The rest**: `dsup` lists `UsingHardwareAcceleratedVideoDecoder` (true, read-only),
//!   `RealTime` and `MaximizePowerEfficiency` (taken); `dcop` and `dset` answer them;
//!   `Paravirtualized` is the guest's own, and a `dset` of it gets no reply; `dcan` answers
//!   whether the decoder goes on with another format description; `dinv` drops the decoder;
//!   `dfin`, `dmul` and `dsav` answer `kVTUnimplementedErr`.
//!
//! A message for a surface the session does not hold, and a surface lent with a message that
//! takes none, is let go of at once (`rels`): the guest's kext holds each surface it lent until
//! the host lets go of it.

use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Duration;

use crate::codec::h264::nal::LengthPrefixed;
use crate::codec::h264::sps::Sps as H264Sps;
use crate::codec::h264::{self, avcc};
use crate::codec::hevc::nal::nal_type;
use crate::codec::hevc::sps::Sps as HevcSps;
use crate::codec::hevc::{self, hvcc};
use crate::core::backend::{Chroma, PixelFormat};
use crate::core::fourcc::FourCc;
use crate::core::runtime::Runtime;
use crate::paravirt::WireError;
use crate::paravirt::host::Codec;
use crate::paravirt::image_description::{AVC_CONFIGURATION, HEVC_CONFIGURATION, ImageDescription};
use crate::paravirt::message::{
    GuestUuid, Message, MessageFlags, MessageHeader, MessageWriter, ValueType,
};
use crate::paravirt::names;
use crate::paravirt::plist::Value;
use crate::paravirt::sample_buffer::{SampleBuffer, SerializedFormatDescription};
use crate::paravirt::session::{GuestReply, Job, LentSurface, Link, Session, Shared};
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

/// The alignment a tile session asks of its tiles (`TileWidthPixelAlignment`,
/// `TileHeightPixelAlignment`): Apple's hardware decoder's, measured on an M3 Max on macOS 27.0.1,
/// which the guest's CMPhoto meets. A tile lands where its place in its output buffer is a
/// multiple of it, which keeps every region on 4:2:0's even rows and columns.
pub const TILE_PIXEL_ALIGNMENT: u32 = 8;
/// A tile's first byte in its output buffer is a multiple of this: Apple's hardware decoder's
/// `TileByteAlignment`, measured as [`TILE_PIXEL_ALIGNMENT`] was.
pub const TILE_BYTE_ALIGNMENT: u32 = 64;
/// The bytes of a 10-bit sample, which bound an output buffer's row by the widest tile.
const WIDEST_SAMPLE_BYTES: u32 = 2;

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

/// The part of a tile's picture a `dtil` lands, and where in its output buffer, in luma samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileRegion {
    /// The picture's first column and row that land (`,tco`).
    pub source_x: u32,
    /// See `source_x`.
    pub source_y: u32,
    /// The columns that land (`,tcd`).
    pub width: u32,
    /// The rows that land (`,tcd`).
    pub height: u32,
    /// The output buffer's column they land at (`,obp`).
    pub x: u32,
    /// The output buffer's row they land at (`,obp`).
    pub y: u32,
}

/// The tiles a backend decoder decodes into regions: the largest and smallest picture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileLimits {
    /// The widest.
    pub max_width: u32,
    /// The tallest.
    pub max_height: u32,
    /// The narrowest.
    pub min_width: u32,
    /// The shortest.
    pub min_height: u32,
}

/// The host's video arm, as the decoder sessions take it.
pub trait Backend: Send + Sync {
    /// A decoder of `config`, on the session's thread at its `dsta`, or the status the guest sees
    /// for a configuration it does not decode.
    fn open(&self, config: &DecoderConfig) -> Result<Box<dyn BackendDecoder>, OsStatus>;

    /// Why the arm decodes none of the pictures of a stream of `stream`, one the codec takes, so
    /// that its session is refused at its start as one the codec refuses is, not at each picture
    /// or tile; `None` where it decodes them. The default takes 4:2:0 streams of Main and Main 10
    /// alone.
    fn declines(&self, stream: &Stream) -> Option<String> {
        if stream.chroma_format_idc != 1 {
            Some("the arm decodes 4:2:0 pictures alone".to_owned())
        } else {
            (stream.range_extensions)
                .then(|| "the arm decodes no range extensions pictures".to_owned())
        }
    }
}

/// What a stream's SPS says it is: what a refusal of it names ([`Backend::declines`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stream {
    /// The stream's codec.
    pub codec: Codec,
    /// HEVC's `general_profile_idc`, H.264's `profile_idc`.
    pub profile: u8,
    /// `chroma_format_idc`.
    pub chroma_format_idc: u8,
    /// `BitDepthY`.
    pub luma_bits: u8,
    /// `BitDepthC`.
    pub chroma_bits: u8,
    /// The coded picture's width, in luma samples.
    pub width: u32,
    /// The coded picture's height, in luma samples.
    pub height: u32,
    /// Whether HEVC's stream decodes as the range extensions' (`codec::hevc::is_range_extensions`),
    /// on their profiles; false for H.264's.
    pub range_extensions: bool,
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

    /// The tiles it decodes into regions of an output ([`BackendDecoder::decode_tile`]); `None`,
    /// the default, where it decodes none, and a tile session is refused.
    fn tile_limits(&self) -> Option<TileLimits> {
        None
    }

    /// Decodes `access_unit` as [`BackendDecoder::decode`] does, and lands `region` of its
    /// picture in `output`, the rest of `output` left as it was; `kVTUnimplementedErr`, the
    /// default, where it decodes no tiles.
    fn decode_tile(
        &mut self,
        parameter_sets: &[u8],
        access_unit: &[u8],
        output: &dyn LentSurface,
        region: TileRegion,
    ) -> Result<Decoded, OsStatus> {
        let _ = (parameter_sets, access_unit, output, region);
        Err(status::UNIMPLEMENTED)
    }
}

/// Starts a decoder session of `codec` for `uuid`, its thread on `runtime`, decoding on
/// `backend`.
pub(crate) fn start(
    uuid: GuestUuid,
    codec: Codec,
    backend: Arc<dyn Backend>,
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
            codec,
            decoder: None,
            config: None,
            length_size: 4,
            parameter_sets: Vec::new(),
            started: false,
            properties: Vec::new(),
            fragments: Vec::new(),
            discarding: false,
            tiles: None,
        }
        .run();
    })
}

/// A frame's `dfrm` reply, sent once.
struct FrameReply {
    request: MessageHeader,
    sent: bool,
    /// Its `,inf`.
    info: u32,
}

/// A tile session's decoder, and what it reads its samples with.
struct TileSession {
    decoder: Box<dyn BackendDecoder>,
    /// The bytes before each NAL unit of the tiles' samples.
    length_size: u8,
    /// The configuration's parameter sets in Annex-B form, for its first decode; empty after it.
    parameter_sets: Vec<u8>,
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
    /// The tile session a `dsts` started.
    tiles: Option<TileSession>,
}

impl Worker {
    fn run(mut self) {
        while let Ok(job) = self.jobs.recv() {
            if let Err(error) = self.handle(&job.bytes, &job.surfaces) {
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

    fn handle(&mut self, bytes: &[u8], surfaces: &[Arc<dyn LentSurface>]) -> Handled {
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
            b"dsts" => self.start_tiles(&message),
            b"dtil" => self.decode_tile(bytes, message.header(), surfaces),
            b"dtfi" => self.reply_status(&message, status::NO_ERROR),
            b"dfin" | b"dsav" => self.reply_status(&message, status::UNIMPLEMENTED),
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

    /// Opens a decoder of `config`, sending `Dpba`, or the status it is refused with:
    /// `kVTVideoDecoderUnsupportedDataFormatErr` for a stream whose pictures the host decodes
    /// none of.
    fn open(&mut self, config: &DecoderConfig) -> Result<(), OsStatus> {
        let record = read_record(config)?;
        if let Some(undecoded) = undecoded(&*self.backend, &record) {
            log::warn!(
                kind:% = undecoded.kind();
                "hardware video: a decoder session of a stream the host does not decode, \
                 refused: {undecoded}"
            );
            return Err(status::VIDEO_DECODER_UNSUPPORTED_DATA_FORMAT);
        }
        let format = record
            .format
            .ok_or(status::VIDEO_DECODER_UNSUPPORTED_DATA_FORMAT)?;
        let decoder = self.backend.open(config)?;
        self.send_buffer_attributes(config, format);
        self.decoder = Some(decoder);
        self.config = Some(config.clone());
        self.length_size = record.length_size;
        self.parameter_sets = record.parameter_sets;
        Ok(())
    }

    /// `Dpba`: the attributes the guest's output buffers are to have: the format the picture
    /// lands in, the description's size, IOSurface backing, which the host needs to reach them,
    /// and the row and plane alignments and cache mode a hardware decoder's buffers have.
    fn send_buffer_attributes(&self, config: &DecoderConfig, format: PixelFormat) {
        let attributes = Value::dictionary([
            (
                "PixelFormatType",
                Value::Array(vec![Value::Integer(i64::from(format.fourcc().0))]),
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
        let Some(whole) = self.gather(bytes, last)? else {
            return Ok(());
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

    /// A `dfrm`'s or `dtil`'s fragment: kept where flagged with more to come (`None`), else the
    /// whole message, this fragment last. A message past [`MAX_FRAME_MESSAGE_BYTES`] is dropped,
    /// its fragments up to the last too.
    fn gather(&mut self, bytes: &[u8], last: MessageHeader) -> Result<Option<Vec<u8>>, WireError> {
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
            return Ok(None);
        }
        Ok(Some(if self.fragments.is_empty() {
            bytes.to_vec()
        } else {
            let mut whole = std::mem::take(&mut self.fragments);
            whole.extend_from_slice(bytes);
            whole
        }))
    }

    /// `dsts`: starts a tile session on the format description, or answers why not (the module
    /// doc).
    fn start_tiles(&mut self, message: &Message<'_>) -> Handled {
        let description = message
            .image_description(names::FORMAT_DESCRIPTION)?
            .ok_or(WireError::Malformed("no format description"))?;
        let description = ImageDescription::parse(description)?;
        let config = config_from_description(&description).filter(|c| c.codec == self.codec);
        let status = match config {
            Some(config) => match self.open_tiles(&config) {
                Ok(()) => NO_ERROR,
                Err(status) => status,
            },
            None => status::VIDEO_DECODER_UNSUPPORTED_DATA_FORMAT,
        };
        self.reply_status(message, status)
    }

    /// Opens a tile session's decoder of `config`, sending `TDrs`, or the status it is refused
    /// with: `kVTUnimplementedErr` where the backend decodes no tiles, and for a stream whose
    /// pictures the host decodes none of, whose tiles the guest then decodes in software, as
    /// where the backend decodes none.
    fn open_tiles(&mut self, config: &DecoderConfig) -> Result<(), OsStatus> {
        self.tiles = None;
        let record = read_record(config)?;
        if let Some(undecoded) = undecoded(&*self.backend, &record) {
            log::warn!(
                kind:% = undecoded.kind();
                "hardware video: a tile session of a stream the host does not decode, \
                 refused: {undecoded}"
            );
            return Err(status::UNIMPLEMENTED);
        }
        let format = record.format.ok_or(status::UNIMPLEMENTED)?;
        let decoder = self.backend.open(config)?;
        let limits = decoder.tile_limits().ok_or(status::UNIMPLEMENTED)?;
        self.send_tile_requirements(format, limits);
        self.tiles = Some(TileSession {
            decoder,
            length_size: record.length_size,
            parameter_sets: record.parameter_sets,
        });
        Ok(())
    }

    /// `TDrs`: the attributes the guest's output buffers are to have (the format the tiles land
    /// in, IOSurface backing, and a hardware decoder's row and plane alignments), and the tiles
    /// the session decodes: the backend's largest and smallest picture, [`TILE_PIXEL_ALIGNMENT`],
    /// [`TILE_BYTE_ALIGNMENT`], and rows of an output buffer up to the widest tile's.
    fn send_tile_requirements(&self, format: PixelFormat, limits: TileLimits) {
        let attributes = Value::dictionary([
            (
                "PixelFormatType",
                Value::Array(vec![Value::Integer(i64::from(format.fourcc().0))]),
            ),
            ("IOSurfaceProperties", Value::Dictionary(Vec::new())),
            ("BytesPerRowAlignment", Value::Integer(64)),
            ("PlaneAlignment", Value::Integer(256)),
        ]);
        let widest_row = limits
            .max_width
            .saturating_mul(WIDEST_SAMPLE_BYTES)
            .next_multiple_of(TILE_BYTE_ALIGNMENT);
        let requirements = Value::dictionary([
            (
                "MaximumCanvasRowBytes",
                Value::Integer(i64::from(widest_row)),
            ),
            (
                "MaximumTileHeight",
                Value::Integer(i64::from(limits.max_height)),
            ),
            (
                "MaximumTileWidth",
                Value::Integer(i64::from(limits.max_width)),
            ),
            (
                "MinimumDecodableHeight",
                Value::Integer(i64::from(limits.min_height)),
            ),
            (
                "MinimumDecodableWidth",
                Value::Integer(i64::from(limits.min_width)),
            ),
            (
                "MinimumTileHeight",
                Value::Integer(i64::from(limits.min_height)),
            ),
            (
                "MinimumTileWidth",
                Value::Integer(i64::from(limits.min_width)),
            ),
            (
                "TileByteAlignment",
                Value::Integer(i64::from(TILE_BYTE_ALIGNMENT)),
            ),
            (
                "TileHeightPixelAlignment",
                Value::Integer(i64::from(TILE_PIXEL_ALIGNMENT)),
            ),
            (
                "TileWidthPixelAlignment",
                Value::Integer(i64::from(TILE_PIXEL_ALIGNMENT)),
            ),
        ]);
        let mut message = MessageWriter::new(
            names::SET_TILE_DECODE_REQUIREMENTS,
            MessageFlags::NONE,
            Some(self.shared.uuid),
            0,
        );
        message.property_list(names::TILE_BUFFER_ATTRIBUTES, Some(&attributes));
        message.property_list(names::TILE_REQUIREMENTS, Some(&requirements));
        self.send(&message.finish());
    }

    /// `dtil`: decodes one tile into the surface it lends, emits it, then replies (the module
    /// doc). Fragments are kept as `dfrm`'s.
    fn decode_tile(
        &mut self,
        bytes: &[u8],
        last: MessageHeader,
        surfaces: &[Arc<dyn LentSurface>],
    ) -> Handled {
        let Some(whole) = self.gather(bytes, last)? else {
            return Ok(());
        };
        let message = Message::parse(&whole)?;
        let tile = message.s64(names::TILE)?.unwrap_or(0);
        let origin =
            (message.i32_pair(names::TILE_SOURCE_ORIGIN, ValueType::Point)?).unwrap_or((0, 0));
        let size = (message.i32_pair(names::TILE_SOURCE_SIZE, ValueType::Size)?)
            .ok_or(WireError::Malformed("no tile size"))?;
        let at = (message.i32_pair(names::TILE_OUTPUT_POINT, ValueType::Point)?).unwrap_or((0, 0));
        let buffer = message.uuid(names::TILE_BUFFER_UUID)?;
        let serialization = message
            .sample_buffer(names::SAMPLE_BUFFER)?
            .ok_or(WireError::Malformed("no sample buffer"))?;
        let sample = SampleBuffer::parse(&serialization)?;
        let status = match tile_region(origin, size, at) {
            Some(region) => self.land_tile(&sample, region, surfaces.first()),
            None => status::PARAMETER,
        };
        self.emit_tile(tile, status, buffer);
        let mut reply = MessageWriter::reply_to(&last, MessageFlags::REPLY_TO_GUEST);
        reply.s32(names::ERROR, status::NO_ERROR);
        reply.u32(names::INFO_FLAGS, 0);
        self.send(&reply.finish());
        Ok(())
    }

    /// Decodes `sample`, a tile, and lands `region` of it in `output`: the status its `TDem`
    /// carries.
    fn land_tile(
        &mut self,
        sample: &SampleBuffer,
        region: TileRegion,
        output: Option<&Arc<dyn LentSurface>>,
    ) -> OsStatus {
        let Some(tiles) = self.tiles.as_mut() else {
            return NO_DECODER;
        };
        let Some(output) = output else {
            return status::PARAMETER;
        };
        let access_unit = (sample.data.as_deref())
            .ok_or(status::VIDEO_DECODER_BAD_DATA)
            .and_then(|data| annex_b(data, tiles.length_size));
        let parameter_sets = std::mem::take(&mut tiles.parameter_sets);
        let decoded = access_unit
            .and_then(|au| (tiles.decoder).decode_tile(&parameter_sets, &au, &**output, region));
        match decoded {
            Ok(Decoded::Landed) => NO_ERROR,
            // A tile is a picture of its own: one its decode skips landed nothing.
            Ok(Decoded::Skipped) => status::VIDEO_DECODER_BAD_DATA,
            Err(status) => {
                // The next tile gets the parameter sets this one did not land with.
                tiles.parameter_sets = parameter_sets;
                status
            }
        }
    }

    /// `TDem`: the tile's status, and its output buffer; the guest's answer is waited for and not
    /// read.
    fn emit_tile(&mut self, tile: i64, status: OsStatus, buffer: Option<GuestUuid>) {
        let sent = self.request(names::EMIT_DECODED_TILE, EMIT_REPLY_WAIT, |message| {
            message.s64(names::TILE, tile);
            message.s32(names::ERROR, status);
            message.u32(names::INFO_FLAGS, 0);
            if let Some(uuid) = buffer {
                message.uuid(names::TILE_BUFFER_UUID, uuid);
            }
        });
        if let Ok(reply) = sent {
            self.shared.let_go(&reply.surfaces);
        }
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
        &self,
        kind: FourCc,
        wait: Duration,
        fill: impl FnOnce(&mut MessageWriter),
    ) -> Result<GuestReply, OsStatus> {
        let (id, message) = self.shared.request_message(kind, fill);
        (self.shared).request(&*self.runtime, id, &[], &message.finish(), wait)
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

/// The CoreVideo format a stream's pictures land in: the two-plane format of its chroma format,
/// depth and range ([`PixelFormat::of`]); `None` for one of none (monochrome, or past 12 bits).
fn pixel_format(stream: &Stream, full_range: bool) -> Option<PixelFormat> {
    PixelFormat::of(
        Chroma::of_idc(stream.chroma_format_idc)?,
        stream.luma_bits,
        full_range,
    )
}

/// A `dtil`'s region from its `,tco`, `,tcd` and `,obp`: `None` for a negative or empty one, one
/// from an odd column or row of the picture, or one off [`TILE_PIXEL_ALIGNMENT`] in its output. Its
/// width and height may be odd, as an image's last column and row of tiles' are.
fn tile_region(origin: (i32, i32), size: (i32, i32), at: (i32, i32)) -> Option<TileRegion> {
    let value = |v: i32| u32::try_from(v).ok();
    let region = TileRegion {
        source_x: value(origin.0)?,
        source_y: value(origin.1)?,
        width: value(size.0)?,
        height: value(size.1)?,
        x: value(at.0)?,
        y: value(at.1)?,
    };
    let even = (region.source_x | region.source_y) & 1 == 0;
    let aligned = (region.x | region.y).is_multiple_of(TILE_PIXEL_ALIGNMENT);
    (even && aligned && region.width != 0 && region.height != 0).then_some(region)
}

/// What the host reads of a configuration record.
struct Record {
    length_size: u8,
    /// Its parameter sets, Annex B.
    parameter_sets: Vec<u8>,
    /// The CoreVideo format its pictures land in ([`pixel_format`]), where it has one.
    format: Option<PixelFormat>,
    /// The stream its SPS describes.
    stream: Stream,
    /// Why the codec takes none of the stream's pictures, where it takes none.
    undecoded: Option<Undecoded>,
}

/// A stream whose pictures the host's decoders take none of: what the codec's `check_supported`
/// refuses of its SPS (a profile, chroma format, bit depths or tools outside 4:2:0 at 8 bits,
/// HEVC's at 8 or 10, and HEVC's range extensions at 8, 10 and 12), which every arm's decode of
/// each picture checks, or what the arm declines of it ([`Backend::declines`]), and what the SPS
/// says the stream is, which the refusal's line names.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Undecoded {
    /// What `check_supported` refuses, or the arm declines.
    why: String,
    stream: Stream,
}

impl Stream {
    /// HEVC's: `sps`'s.
    fn of_hevc(sps: &HevcSps) -> Self {
        Self {
            codec: Codec::Hevc,
            profile: sps.profile_tier_level.general_profile_idc,
            chroma_format_idc: sps.chroma_format_idc,
            luma_bits: sps.bit_depth_luma_minus8.saturating_add(8),
            chroma_bits: sps.bit_depth_chroma_minus8.saturating_add(8),
            width: sps.pic_width_in_luma_samples,
            height: sps.pic_height_in_luma_samples,
            range_extensions: hevc::is_range_extensions(sps),
        }
    }

    /// H.264's: `sps`'s.
    fn of_h264(sps: &H264Sps) -> Self {
        Self {
            codec: Codec::H264,
            profile: sps.profile_idc,
            chroma_format_idc: sps.chroma_format_idc,
            luma_bits: sps.bit_depth_luma_minus8.saturating_add(8),
            chroma_bits: sps.bit_depth_chroma_minus8.saturating_add(8),
            width: sps.coded_width(),
            height: sps.coded_height(),
            range_extensions: false,
        }
    }
}

impl Undecoded {
    /// The kind of its refusal's line (docs/conventions.md, "Each degradation is logged once per
    /// boot, then counted"): the stream's codec, profile, chroma format and bit depths, so that
    /// the first stream of each is named, and the rest counted. They are spelled out, as the
    /// logger reads a kind up to its first digit; the size is left out, so that images of one
    /// kind at many sizes make one.
    fn kind(&self) -> String {
        let codec = match self.stream.codec {
            Codec::H264 => "AVC",
            Codec::Hevc => "HEVC",
        };
        format!(
            "{codec} profile {} chroma format {}, luma {} bits, chroma {} bits",
            spelled(self.stream.profile.into()),
            spelled(self.stream.chroma_format_idc.into()),
            spelled(self.stream.luma_bits.into()),
            spelled(self.stream.chroma_bits.into()),
        )
    }
}

impl std::fmt::Display for Undecoded {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stream = &self.stream;
        let (codec, profile) = match stream.codec {
            Codec::H264 => ("H.264", "profile_idc"),
            Codec::Hevc => ("HEVC", "general_profile_idc"),
        };
        write!(
            f,
            "{} ({codec} {profile} {}, chroma_format_idc {}, luma {} bits, chroma {} bits, {}x{})",
            self.why,
            stream.profile,
            stream.chroma_format_idc,
            stream.luma_bits,
            stream.chroma_bits,
            stream.width,
            stream.height
        )
    }
}

/// `n`'s digits as words, joined by hyphens: `244` is `two-four-four`.
fn spelled(n: u32) -> String {
    const DIGITS: [&str; 10] = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ];
    let words: Vec<&str> = (n.to_string().chars())
        .filter_map(|c| DIGITS.get(usize::try_from(c.to_digit(10)?).ok()?).copied())
        .collect();
    words.join("-")
}

/// Why the host decodes none of `record`'s stream's pictures on `backend`, where it decodes none:
/// what the codec refuses of it, else what the arm declines.
fn undecoded(backend: &dyn Backend, record: &Record) -> Option<Undecoded> {
    (record.undecoded.clone()).or_else(|| {
        let why = backend.declines(&record.stream)?;
        Some(Undecoded {
            why,
            stream: record.stream,
        })
    })
}

/// Reads `config`'s record, or the status a session on it is refused with: a record that does
/// not read, and 3-byte NAL unit lengths, which ISO/IEC 14496-15 does not allow,
/// `kVTVideoDecoderUnsupportedDataFormatErr`; an H.264 SPS that lets pictures be fields
/// (`frame_mbs_only_flag` 0), `kVTVideoDecoderMalfunctionErr`, as no backend here decodes a field.
/// A stream whose pictures the host decodes none of reads, with why and what its SPS says it is
/// ([`Record::undecoded`]).
fn read_record(config: &DecoderConfig) -> Result<Record, OsStatus> {
    let unsupported = status::VIDEO_DECODER_UNSUPPORTED_DATA_FORMAT;
    let (length_size, units, full_range, stream, undecoded) = match config.codec {
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
            let stream = Stream::of_h264(&sps);
            let undecoded = (h264::check_supported(&sps).err()).map(|error| Undecoded {
                why: error.to_string(),
                stream,
            });
            let full_range = sps.is_full_range();
            (record.length_size, units, full_range, stream, undecoded)
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
            let stream = Stream::of_hevc(&sps);
            let undecoded = (hevc::check_supported(&sps).err()).map(|error| Undecoded {
                why: error.to_string(),
                stream,
            });
            (
                record.length_size,
                record.nal_units,
                full,
                stream,
                undecoded,
            )
        }
    };
    if length_size == 3 || !matches!(length_size, 1 | 2 | 4) {
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
        format: pixel_format(&stream, full_range),
        stream,
        undecoded,
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
