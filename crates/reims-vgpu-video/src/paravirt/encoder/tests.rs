//! An encoder session driven as the guest's VideoToolbox drives it, on a backend that records
//! what it is given and gives back samples of its own. Every byte here is written by hand: the
//! samples are a few bytes of an IDR slice, the configuration record an `avcC` header with no
//! parameter sets, and the guest's format description the QuickTime layout written field by
//! field ([`hand_written`]).

use std::sync::mpsc;

use super::*;
use crate::codec::encode::FrameType;
use crate::core::runtime::StdRuntime;
use crate::core::sync::Mutex;
use crate::paravirt::image_description::tests::hand_written;
use crate::paravirt::message::{MessageHeader, read_header};
use crate::paravirt::{Answer, Host};

/// An `avcC` header of High profile, level 1.1, 4-byte lengths, and no parameter sets.
const RECORD: [u8; 7] = [1, 0x64, 0, 0x0b, 0xff, 0xe0, 0];
/// An IDR slice's first bytes, behind its 4-byte length.
const SLICE: [u8; 6] = [0, 0, 0, 2, 0x65, 0x88];

const OURS: &str = "reims-vgpu-video.videoencoder.avc";

struct FakeLink(Mutex<mpsc::Sender<Vec<u8>>>);

impl Link for FakeLink {
    fn send(&self, message: &[u8]) -> bool {
        self.0.lock().send(message.to_vec()).is_ok()
    }
}

#[derive(Debug)]
struct Surface(u64);

impl LentSurface for Surface {
    fn id(&self) -> u64 {
        self.0
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// How the fake encoder answers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// A sample of `n` bytes for each frame, at once.
    AtOnce(usize),
    /// Nothing until a flush, which gives a sample for every frame but the last.
    Hold,
}

/// What the fake backend was given.
#[derive(Default)]
struct Calls {
    opened: Vec<EncoderConfig>,
    frames: Vec<(SourceFrame, u64)>,
    reconfigured: Vec<EncoderConfig>,
}

type Recorded = Arc<Mutex<Calls>>;

struct FakeBackend {
    calls: Recorded,
    mode: Mode,
    /// What a reconfigure answers.
    reconfigure: Result<(), OsStatus>,
}

struct FakeEncoder {
    calls: Recorded,
    mode: Mode,
    reconfigure: Result<(), OsStatus>,
    held: Vec<SourceFrame>,
    /// Whether a sample carried the record yet.
    configured: bool,
}

impl EncoderBackend for FakeBackend {
    fn open(&self, config: &EncoderConfig) -> Result<Box<dyn BackendEncoder>, OsStatus> {
        self.calls.lock().opened.push(config.clone());
        Ok(Box::new(FakeEncoder {
            calls: Arc::clone(&self.calls),
            mode: self.mode,
            reconfigure: self.reconfigure,
            held: Vec::new(),
            configured: false,
        }))
    }
}

impl FakeEncoder {
    fn sample(&mut self, frame: &SourceFrame, bytes: usize) -> Coded<OsStatus> {
        let mut data = SLICE.to_vec();
        data.resize(bytes.max(SLICE.len()), 0x5a);
        let configuration = (!self.configured).then(|| Arc::from(&RECORD[..]));
        self.configured = true;
        Ok(EncodedSample {
            id: frame.id,
            data,
            pts: frame.pts,
            dts: frame.pts,
            duration: frame.duration,
            frame_type: FrameType::Idr,
            is_sync: frame.id == 0,
            depends_on_others: frame.id != 0,
            is_depended_on: true,
            configuration,
        })
    }
}

impl BackendEncoder for FakeEncoder {
    fn encode(
        &mut self,
        source: &dyn LentSurface,
        frame: &SourceFrame,
    ) -> Result<Vec<Coded<OsStatus>>, OsStatus> {
        self.calls.lock().frames.push((*frame, source.id()));
        match self.mode {
            Mode::AtOnce(bytes) => Ok(vec![self.sample(frame, bytes)]),
            Mode::Hold => {
                self.held.push(*frame);
                Ok(Vec::new())
            }
        }
    }

    fn flush(&mut self) -> Result<Vec<Coded<OsStatus>>, OsStatus> {
        let mut held = std::mem::take(&mut self.held);
        held.pop();
        Ok(held.iter().map(|frame| self.sample(frame, 0)).collect())
    }

    fn reconfigure(&mut self, config: &EncoderConfig) -> Result<(), OsStatus> {
        self.calls.lock().reconfigured.push(config.clone());
        self.reconfigure
    }
}

fn encoders() -> Vec<Encoder> {
    vec![Encoder {
        codec: Codec::H264,
        profiles: vec![Profile::H264Main, Profile::H264High],
        max_width: 4096,
        max_height: 2304,
    }]
}

fn time(value: i64, timescale: i32) -> MediaTime {
    MediaTime {
        value,
        timescale,
        flags: MediaTime::FLAG_VALID,
        epoch: 0,
    }
}

/// The guest's side of one session.
struct Guest {
    host: Host,
    link: Arc<dyn Link>,
    sent: mpsc::Receiver<Vec<u8>>,
    uuid: GuestUuid,
    calls: Recorded,
}

impl Guest {
    fn new(mode: Mode, reconfigure: Result<(), OsStatus>) -> Self {
        let calls = Recorded::default();
        let backend = Arc::new(FakeBackend {
            calls: Arc::clone(&calls),
            mode,
            reconfigure,
        });
        let (sender, sent) = mpsc::channel();
        let guest = Self {
            host: Host::default().with_encoders(encoders(), backend, StdRuntime::shared()),
            link: Arc::new(FakeLink(Mutex::new(sender))),
            sent,
            uuid: GuestUuid([0x5e; 16]),
            calls,
        };
        assert_eq!(guest.create(Some(OURS)), 0, "the create is taken");
        guest
    }

    /// Sends `+enc` of `avc1` naming `encoder`, and the reply's `,err`.
    fn create(&self, encoder: Option<&str>) -> OsStatus {
        let answer = self.request(b"+enc", 1, Vec::new(), |m| {
            m.s32(names::CODEC, i32::from_be_bytes(*b"avc1"));
            m.u32(names::VERSION, 6);
            if let Some(id) = encoder {
                m.property_list(names::CODEC_ID, Some(&Value::string(id)));
            }
        });
        status_of(&answer.replies[0]).unwrap()
    }

    fn send(
        &self,
        kind: &[u8; 4],
        flags: MessageFlags,
        reply_id: u64,
        surfaces: Vec<Arc<dyn LentSurface>>,
        fill: impl FnOnce(&mut MessageWriter),
    ) -> Answer {
        let mut message = MessageWriter::new(FourCc::new(*kind), flags, Some(self.uuid), reply_id);
        fill(&mut message);
        self.host
            .deliver_lending(&message.finish(), surfaces, Some(Arc::clone(&self.link)))
    }

    /// Sends a message of `kind` that expects a reply.
    fn request(
        &self,
        kind: &[u8; 4],
        reply_id: u64,
        surfaces: Vec<Arc<dyn LentSurface>>,
        fill: impl FnOnce(&mut MessageWriter),
    ) -> Answer {
        self.send(
            kind,
            MessageFlags::GUEST_EXPECTS_REPLY,
            reply_id,
            surfaces,
            fill,
        )
    }

    fn next(&self) -> Vec<u8> {
        self.sent
            .recv_timeout(Duration::from_secs(5))
            .expect("a message from the host")
    }

    /// The next message, which must be a `vtpv` message of `kind`.
    fn next_of(&self, kind: &[u8; 4]) -> (MessageHeader, Vec<u8>) {
        let bytes = self.next();
        let header = read_header(&bytes, 0).expect("a vtpv message");
        assert_eq!(header.message_type, FourCc::new(*kind), "{bytes:02x?}");
        (header, bytes)
    }

    /// The next message, which must be a `rels`, and the surfaces it names.
    fn next_release(&self) -> Vec<u64> {
        let bytes = self.next();
        assert_eq!(bytes.get(4..8), Some(&b"rels"[..]), "{bytes:02x?}");
        let (ids, _) = bytes[8..].as_chunks::<8>();
        ids.iter().map(|id| u64::from_be_bytes(*id)).collect()
    }

    fn quiet(&self) -> bool {
        self.sent.recv_timeout(Duration::from_millis(200)).is_err()
    }

    /// Answers the host's request `request` with `,err` `status`, filled by `fill`.
    fn answer(
        &self,
        request: &MessageHeader,
        status: OsStatus,
        fill: impl FnOnce(&mut MessageWriter),
    ) {
        let kind = request.message_type.to_be_bytes();
        let answer = self.send(
            &kind,
            MessageFlags::REPLY_TO_HOST,
            request.reply_id,
            Vec::new(),
            |m| {
                m.s32(names::ERROR, status);
                fill(m);
            },
        );
        assert_eq!(answer.status, 0);
    }

    /// Sends `esta` at `width` by `height`.
    fn start(&self, width: i32, height: i32) {
        self.request(b"esta", 2, Vec::new(), |m| {
            m.i32_pair(names::DIMENSIONS, ValueType::Dimensions, (width, height));
        });
    }

    /// Starts the session at `width` by `height`, answering its `Epba`: the attributes it asked
    /// for.
    fn started(&self, width: i32, height: i32) -> Value {
        self.start(width, height);
        let (epba, request) = self.next_of(b"Epba");
        assert!(epba.flags.contains(MessageFlags::HOST_EXPECTS_REPLY));
        self.answer(&epba, 0, |_| {});
        let (_, reply) = self.next_of(b"esta");
        assert_eq!(status_of(&reply), Some(0));
        plist_of(&request, names::PIXEL_BUFFER_ATTRIBUTES)
    }

    /// Sends frame `frame` at `pts` thirtieths of a second, on surface `surface`.
    fn frame(&self, frame: i64, surface: u64, properties: Option<&Value>) {
        self.request(
            b"efrm",
            100 + frame as u64,
            vec![Arc::new(Surface(surface))],
            |m| {
                m.s64(names::FRAME, frame);
                m.time(names::PRESENTATION_TIME, time(frame, 30));
                m.time(names::DURATION, time(1, 30));
                m.property_list(names::FRAME_PROPERTIES, properties);
                m.uuid(names::PIXEL_BUFFER_UUID, GuestUuid([frame as u8; 16]));
            },
        );
    }

    /// Answers the `Efds` that comes next with a description the guest made of its extensions.
    fn describe(&self) -> Value {
        let (efds, request) = self.next_of(b"Efds");
        let message = Message::parse(&request).unwrap();
        assert_eq!(
            message.s32(names::CODEC).unwrap(),
            Some(i32::from_be_bytes(*b"avc1"))
        );
        let size = (
            message.s32(names::WIDTH).unwrap(),
            message.s32(names::HEIGHT).unwrap(),
        );
        let (Some(width), Some(height)) = size else {
            panic!("an Efds with a size");
        };
        let description = hand_written(b"avc1", width as u16, height as u16, &RECORD);
        self.answer(&efds, 0, |m| {
            m.image_description(names::CREATED_FORMAT_DESCRIPTION, &description);
        });
        plist_of(&request, names::EXTENSIONS)
    }

    /// The `Eemt` that comes next, gathered from its fragments as the guest gathers them, one
    /// after another, and answered: the last fragment's message, and the sample buffer, where it
    /// has one.
    fn emitted(&self) -> (Vec<u8>, Option<SampleBuffer>) {
        let mut whole = Vec::new();
        loop {
            let (header, bytes) = self.next_of(b"Eemt");
            assert!(bytes.len() <= HOST_MESSAGE_MAX);
            whole.extend_from_slice(&bytes);
            if header.flags.contains(MessageFlags::MORE_FRAGMENTS) {
                assert_eq!(header.reply_id, 0);
                assert!(!header.flags.contains(MessageFlags::HOST_EXPECTS_REPLY));
                continue;
            }
            assert!(header.flags.contains(MessageFlags::HOST_EXPECTS_REPLY));
            self.answer(&header, 0, |_| {});
            let serialization = (Message::parse(&whole).unwrap())
                .sample_buffer(names::SAMPLE_BUFFER)
                .unwrap();
            let sample = serialization.map(|bytes| SampleBuffer::parse(&bytes).unwrap());
            return (bytes, sample);
        }
    }
}

fn status_of(bytes: &[u8]) -> Option<i32> {
    Message::parse(bytes).unwrap().s32(names::ERROR).unwrap()
}

fn plist_of(bytes: &[u8], key: FourCc) -> Value {
    let message = Message::parse(bytes).unwrap();
    plist::decode(message.property_list_bytes(key).unwrap().unwrap()).unwrap()
}

fn yes_no(value: bool) -> SerializedValue {
    SerializedValue::PropertyList(Value::Boolean(value))
}

#[test]
fn a_start_asks_for_source_buffers_of_the_profiles_format_at_the_frames_size() {
    let guest = Guest::new(Mode::AtOnce(0), Ok(()));
    let attributes = guest.started(641, 361);
    assert_eq!(
        attributes,
        Value::dictionary([
            (
                "PixelFormatType",
                Value::Array(vec![Value::Integer(i64::from(u32::from_be_bytes(
                    *b"420v"
                )))]),
            ),
            ("Width", Value::Integer(641)),
            ("Height", Value::Integer(361)),
            ("IOSurfaceProperties", Value::Dictionary(Vec::new())),
            ("BytesPerRowAlignment", Value::Integer(64)),
            ("PlaneAlignment", Value::Integer(64)),
        ])
    );
    let opened = &guest.calls.lock().opened;
    let [config] = opened.as_slice() else {
        panic!("one encoder opened");
    };
    assert_eq!((config.settings.width, config.settings.height), (640, 360));
    assert_eq!(config.settings.profile, Profile::H264High);
}

#[test]
fn a_size_the_encoder_does_not_take_is_refused_without_a_request() {
    let guest = Guest::new(Mode::AtOnce(0), Ok(()));
    for ((width, height), refused) in [
        ((4097, 64), status::ENCODER_SIZE_UNSUPPORTED),
        ((1, 64), status::PARAMETER),
        ((0, 64), status::MALFORMED_MESSAGE),
    ] {
        guest.start(width, height);
        let (_, reply) = guest.next_of(b"esta");
        assert_eq!(status_of(&reply), Some(refused), "{width}x{height}");
    }
    assert!(guest.calls.lock().opened.is_empty());
}

#[test]
fn a_frame_is_encoded_from_its_surface_and_emitted_with_the_streams_description() {
    let guest = Guest::new(Mode::AtOnce(0), Ok(()));
    guest.started(64, 48);
    let key = Value::dictionary([(FORCE_KEY_FRAME, Value::Boolean(true))]);
    guest.frame(0, 7, Some(&key));
    let extensions = guest.describe();
    assert_eq!(extensions.get("FormatName"), Some(&Value::string("H.264")));
    assert_eq!(
        extensions.get("SampleDescriptionExtensionAtoms"),
        Some(&Value::dictionary([("avcC", Value::Data(RECORD.to_vec()))]))
    );
    let (emitted, sample) = guest.emitted();
    let message = Message::parse(&emitted).unwrap();
    assert_eq!(message.s64(names::FRAME).unwrap(), Some(0));
    assert_eq!(status_of(&emitted), Some(0));
    assert_eq!(message.u32(names::ENCODE_INFO_FLAGS).unwrap(), Some(0));
    assert_eq!(
        message.uuid(names::PIXEL_BUFFER_UUID).unwrap(),
        Some(GuestUuid([0; 16]))
    );
    let sample = sample.expect("a sample");
    assert_eq!(sample.data.as_deref(), Some(&SLICE[..]));
    assert_eq!(sample.sizes, vec![6]);
    assert_eq!(
        sample.timings,
        vec![SampleTiming {
            duration: time(1, 30),
            presentation: time(0, 30),
            decode: time(0, 30),
        }]
    );
    assert_eq!(sample.output_presentation_time, Some(time(0, 30)));
    let description = sample
        .format_description
        .expect("the first sample's description");
    assert_eq!(
        (description.codec, description.width, description.height),
        (FourCc::new(*b"avc1"), 64, 48)
    );
    assert_eq!(
        description.sample_description_atom("avcC"),
        Some(&RECORD[..])
    );
    assert_eq!(
        sample.sample_attachments,
        Some(vec![SerializedValue::Dictionary(vec![
            ("DependsOnOthers".to_owned(), yes_no(false)),
            ("EarlierDisplayTimesAllowed".to_owned(), yes_no(false)),
        ])])
    );
    let (_, reply) = guest.next_of(b"efrm");
    assert_eq!(status_of(&reply), Some(0));
    assert_eq!(
        guest.next_release(),
        vec![7],
        "the frame's surface, once emitted"
    );
    let frames = &guest.calls.lock().frames;
    assert_eq!(
        frames.as_slice(),
        [(
            SourceFrame {
                id: 0,
                pts: 0,
                duration: 3000,
                force_key_frame: true,
            },
            7
        )]
    );
}

#[test]
fn later_samples_carry_no_description_and_are_marked_not_sync() {
    let guest = Guest::new(Mode::AtOnce(0), Ok(()));
    guest.started(64, 48);
    guest.frame(0, 7, None);
    guest.describe();
    guest.emitted();
    guest.next_of(b"efrm");
    guest.next_release();
    guest.frame(1, 8, None);
    let (_, sample) = guest.emitted();
    let sample = sample.unwrap();
    assert_eq!(sample.format_description, None);
    assert_eq!(
        sample.sample_attachments,
        Some(vec![SerializedValue::Dictionary(vec![
            ("NotSync".to_owned(), yes_no(true)),
            ("DependsOnOthers".to_owned(), yes_no(true)),
            ("EarlierDisplayTimesAllowed".to_owned(), yes_no(false)),
        ])])
    );
    guest.next_of(b"efrm");
    assert_eq!(guest.next_release(), vec![8]);
}

#[test]
fn a_long_sample_goes_in_fragments_that_read_back_whole() {
    let guest = Guest::new(Mode::AtOnce(40_000), Ok(()));
    guest.started(64, 48);
    guest.frame(0, 7, None);
    guest.describe();
    let (_, sample) = guest.emitted();
    let data = sample.unwrap().data.unwrap();
    assert_eq!(data.len(), 40_000);
    assert_eq!(&data[..6], &SLICE);
    guest.next_of(b"efrm");
    assert_eq!(guest.next_release(), vec![7]);
}

#[test]
fn frames_the_encoder_holds_are_emitted_at_complete_and_a_lost_one_with_an_error() {
    let guest = Guest::new(Mode::Hold, Ok(()));
    guest.started(64, 48);
    for frame in 0..3 {
        guest.frame(frame, 10 + frame as u64, None);
        let (_, reply) = guest.next_of(b"efrm");
        assert_eq!(status_of(&reply), Some(0));
    }
    guest.request(b"ecfr", 9, Vec::new(), |m| {
        m.time(names::PRESENTATION_TIME, MediaTime::INVALID);
    });
    guest.describe();
    for frame in 0..2 {
        let (emitted, sample) = guest.emitted();
        assert_eq!(
            Message::parse(&emitted).unwrap().s64(names::FRAME).unwrap(),
            Some(frame)
        );
        assert!(sample.is_some());
        assert_eq!(guest.next_release(), vec![10 + frame as u64]);
    }
    let (lost, sample) = guest.emitted();
    assert_eq!(
        Message::parse(&lost).unwrap().s64(names::FRAME).unwrap(),
        Some(2)
    );
    assert_eq!(status_of(&lost), Some(status::VIDEO_ENCODER_MALFUNCTION));
    assert!(sample.is_none());
    assert_eq!(guest.next_release(), vec![12]);
    let (_, reply) = guest.next_of(b"ecfr");
    assert_eq!(status_of(&reply), Some(0));
}

#[test]
fn a_frame_before_the_start_is_an_invalid_session_and_its_surface_goes() {
    let guest = Guest::new(Mode::AtOnce(0), Ok(()));
    guest.frame(0, 7, None);
    assert_eq!(guest.next_release(), vec![7]);
    let (_, reply) = guest.next_of(b"efrm");
    assert_eq!(status_of(&reply), Some(status::INVALID_SESSION));
    assert!(guest.quiet());
}

#[test]
fn an_invalidate_lets_go_of_the_frames_the_encoder_holds() {
    let guest = Guest::new(Mode::Hold, Ok(()));
    guest.started(64, 48);
    for frame in 0..2 {
        guest.frame(frame, 20 + frame as u64, None);
        guest.next_of(b"efrm");
    }
    guest.request(b"einv", 9, Vec::new(), |_| {});
    assert_eq!(guest.next_release(), vec![20, 21]);
    let (_, reply) = guest.next_of(b"einv");
    assert_eq!(status_of(&reply), Some(0));
}

#[test]
fn a_set_reaches_the_running_encoder_and_a_refused_one_is_undone() {
    let guest = Guest::new(Mode::AtOnce(0), Err(status::PARAMETER));
    let set = |key: &str, value: Value| {
        guest.request(b"eset", 3, Vec::new(), |m| {
            m.property_list(names::PROPERTY_KEY, Some(&Value::string(key)));
            m.property_list(names::PROPERTY_VALUE, Some(&value));
        });
        status_of(&guest.next_of(b"eset").1)
    };
    let copy = |key: &str| {
        guest.request(b"ecop", 4, Vec::new(), |m| {
            m.property_list(names::PROPERTY_KEY, Some(&Value::string(key)));
        });
        plist_of(&guest.next_of(b"ecop").1, names::PROPERTY_VALUE)
    };
    assert_eq!(
        set("AverageBitRate", Value::Integer(1_000_000)),
        Some(0),
        "before the start"
    );
    guest.started(64, 48);
    assert_eq!(
        guest.calls.lock().opened[0].settings.rate_control,
        crate::codec::encode::RateControl::VariableBitRate {
            average_bits_per_second: 1_000_000,
            peak_bits_per_second: 1_500_000,
        }
    );
    assert_eq!(
        set("AverageBitRate", Value::Integer(2_000_000)),
        Some(status::PARAMETER)
    );
    assert_eq!(guest.calls.lock().reconfigured.len(), 1);
    assert_eq!(
        copy("AverageBitRate"),
        Value::Integer(1_000_000),
        "the set was undone"
    );
    assert_eq!(
        copy("UsingHardwareAcceleratedVideoEncoder"),
        Value::Boolean(true)
    );
    guest.request(b"eset", 5, Vec::new(), |m| {
        m.property_list(names::PROPERTY_KEY, Some(&Value::string(PARAVIRTUALIZED)));
        m.property_list(names::PROPERTY_VALUE, Some(&Value::Boolean(false)));
    });
    assert!(guest.quiet(), "the guest's own property gets no reply");
}

#[test]
fn a_create_takes_this_hosts_encoder_apples_or_none_and_refuses_another() {
    let guest = Guest::new(Mode::AtOnce(0), Ok(()));
    assert_eq!(
        guest.create(Some("com.apple.videotoolbox.videoencoder.ave.avc")),
        0
    );
    assert_eq!(guest.create(None), 0);
    for other in [
        "com.apple.videotoolbox.videoencoder.ave.hevc",
        "another.encoder",
    ] {
        assert_eq!(
            guest.create(Some(other)),
            status::COULD_NOT_FIND_VIDEO_ENCODER,
            "{other}"
        );
    }
}

#[test]
fn the_encoder_list_names_each_encoder_and_none_without_a_backend() {
    let guest = Guest::new(Mode::AtOnce(0), Ok(()));
    let answer = guest.request(b"encs", 1, Vec::new(), |m| {
        m.u32(names::VERSION, 6);
    });
    assert_eq!(
        plist_of(&answer.replies[0], names::ENCODERS),
        Value::Array(vec![Value::dictionary([
            (
                "VTCodecType",
                Value::Integer(i64::from(u32::from_be_bytes(*b"avc1")))
            ),
            ("VTRating", Value::Integer(400)),
            ("VTCodecName", Value::string("H.264")),
            (
                "CMClassImplementationName",
                Value::string("reims-vgpu-video Encoder - AVC")
            ),
            ("CMClassImplementationID", Value::string(OURS)),
        ])])
    );
    assert!(Host::new(Vec::new()).encoders().is_empty());
}

#[test]
fn times_go_onto_the_backends_clock_and_back() {
    assert_eq!(to_ticks(time(1, 30)), Some(3000));
    assert_eq!(to_ticks(time(1001, 30_000)), Some(3003));
    assert_eq!(to_ticks(time(1, 1_000_000_000)), Some(0));
    assert_eq!(to_ticks(MediaTime::INVALID), None);
    assert_eq!(from_ticks(3003, 30_000), time(1001, 30_000));
    assert_eq!(from_ticks(3000, 0), time(3000, 90_000));
}
