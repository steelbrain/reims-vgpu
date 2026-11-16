//! A decoder session driven as the guest's VideoToolbox drives it, on a backend that records what
//! it is given. Every stream here is written by hand: an H.264 SPS and PPS of a 64x48 baseline
//! stream, their `avcC`, and slices of a few bytes, which the fake backend never decodes.

use std::sync::mpsc;

use super::*;
use crate::core::runtime::StdRuntime;
use crate::paravirt::host::Decoder;
use crate::paravirt::image_description::tests::hand_written;
use crate::paravirt::message::read_header;
use crate::paravirt::sample_buffer::tests::one_sample;
use crate::paravirt::{Answer, Host, plist};

/// A 64x48 baseline SPS (`profile_idc` 66, level 3.0): `frame_mbs_only_flag` 1, no VUI.
const SPS: [u8; 7] = [0x67, 66, 0, 30, 0xf4, 0x23, 0xc8];
/// The same with `frame_mbs_only_flag` 0 and `mb_adaptive_frame_field_flag` 0.
const INTERLACED_SPS: [u8; 7] = [0x67, 66, 0, 30, 0xf4, 0x23, 0x24];
const PPS: [u8; 4] = [0x68, 0xce, 0x38, 0x80];
/// An IDR slice's first bytes, after its 4-byte length.
const SAMPLE: [u8; 6] = [0, 0, 0, 2, 0x65, 0x88];

/// An `avcC` of `sps` and [`PPS`] with NAL unit lengths of `length_size` bytes.
fn avcc(sps: &[u8], length_size: u8) -> Vec<u8> {
    let mut record = vec![1, 66, 0, 30, 0xfc | (length_size - 1), 0xe1];
    record.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    record.extend_from_slice(sps);
    record.push(1);
    record.extend_from_slice(&(PPS.len() as u16).to_be_bytes());
    record.extend_from_slice(&PPS);
    record
}

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

/// What the fake backend's decoders were given: the parameter sets, the access unit and the
/// output's ID, a decode each.
type Calls = Arc<Mutex<Vec<(Vec<u8>, Vec<u8>, u64)>>>;

struct FakeBackend {
    calls: Calls,
    result: Result<Decoded, OsStatus>,
}

struct FakeDecoder {
    calls: Calls,
    result: Result<Decoded, OsStatus>,
}

impl Backend for FakeBackend {
    fn open(&self, _config: &DecoderConfig) -> Result<Box<dyn BackendDecoder>, OsStatus> {
        Ok(Box::new(FakeDecoder {
            calls: Arc::clone(&self.calls),
            result: self.result,
        }))
    }
}

impl BackendDecoder for FakeDecoder {
    fn decode(
        &mut self,
        parameter_sets: &[u8],
        access_unit: &[u8],
        output: &dyn LentSurface,
    ) -> Result<Decoded, OsStatus> {
        self.calls
            .lock()
            .push((parameter_sets.to_vec(), access_unit.to_vec(), output.id()));
        self.result
    }
}

/// The guest's side of one session: the host, the messages it sent, and the session's UUID.
struct Guest {
    host: Host,
    link: Arc<dyn Link>,
    sent: mpsc::Receiver<Vec<u8>>,
    uuid: GuestUuid,
    calls: Calls,
}

impl Guest {
    fn new(result: Result<Decoded, OsStatus>) -> Self {
        let calls = Calls::default();
        let backend = Arc::new(FakeBackend {
            calls: Arc::clone(&calls),
            result,
        });
        let decoders = vec![Decoder {
            codec: Codec::H264,
            profiles: vec![66, 77, 100],
            max_level: 52,
        }];
        let (sender, sent) = mpsc::channel();
        let guest = Self {
            host: Host::with_backend(decoders, backend, StdRuntime::shared()),
            link: Arc::new(FakeLink(Mutex::new(sender))),
            sent,
            uuid: GuestUuid([0x5a; 16]),
            calls,
        };
        let answer = guest.send(
            b"+dec",
            MessageFlags::GUEST_EXPECTS_REPLY,
            1,
            Vec::new(),
            |m| {
                m.s32(names::CODEC, i32::from_be_bytes(*b"avc1"));
                m.u32(names::VERSION, 6);
            },
        );
        let reply = Message::parse(&answer.replies[0])
            .unwrap()
            .s32(names::ERROR)
            .unwrap();
        assert_eq!(reply, Some(0), "the create is taken");
        guest
    }

    /// Sends the session a message of `kind`.
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

    /// The next message the host sent.
    fn next(&self) -> Vec<u8> {
        self.sent
            .recv_timeout(Duration::from_secs(5))
            .expect("a message from the host")
    }

    /// The next message, which must be a `vtpv` message of `kind`, and its header.
    fn next_of(&self, kind: &[u8; 4]) -> (MessageHeader, Vec<u8>) {
        let bytes = self.next();
        let header = read_header(&bytes, 0).expect("a vtpv message");
        assert_eq!(header.message_type, FourCc::new(*kind), "{bytes:02x?}");
        (header, bytes)
    }

    /// Whether the host sent nothing more within a short wait.
    fn quiet(&self) -> bool {
        self.sent.recv_timeout(Duration::from_millis(200)).is_err()
    }

    /// Answers a host request with `,err` `status` and, for a `Dbuf`, a buffer of `uuid` on
    /// `surface`.
    fn answer(&self, request: &MessageHeader, status: OsStatus, buffer: Option<(GuestUuid, u64)>) {
        let surfaces: Vec<Arc<dyn LentSurface>> = buffer
            .map(|(_, id)| Arc::new(Surface(id)) as Arc<dyn LentSurface>)
            .into_iter()
            .collect();
        let kind = request.message_type.to_be_bytes();
        let answer = self.send(
            &kind,
            MessageFlags::REPLY_TO_HOST,
            request.reply_id,
            surfaces,
            |m| {
                m.s32(names::ERROR, status);
                if let Some((uuid, _)) = buffer {
                    m.uuid(names::PIXEL_BUFFER_UUID, uuid);
                }
            },
        );
        assert_eq!(answer.status, 0);
    }

    /// Starts the session on a 64x48 description of `record`.
    fn start(&self, record: &[u8]) {
        let description = hand_written(b"avc1", 64, 48, record);
        self.send(
            b"dsta",
            MessageFlags::GUEST_EXPECTS_REPLY,
            2,
            Vec::new(),
            |m| {
                m.image_description(names::FORMAT_DESCRIPTION, &description);
                m.s32(names::GUEST_POOL, 0);
            },
        );
    }

    /// Sends frame `frame` with `flags`, its sample `sample`.
    fn frame(&self, frame: i64, flags: u32, sample: &[u8]) {
        self.send(
            b"dfrm",
            MessageFlags::GUEST_EXPECTS_REPLY,
            100 + frame as u64,
            Vec::new(),
            |m| {
                m.s64(names::FRAME, frame);
                m.sample_buffer_with_limit(names::SAMPLE_BUFFER, usize::MAX, sample);
                m.u32(names::DECODE_FLAGS, flags);
            },
        );
    }
}

fn status_of(bytes: &[u8]) -> Option<i32> {
    Message::parse(bytes).unwrap().s32(names::ERROR).unwrap()
}

fn info_of(bytes: &[u8]) -> Option<u32> {
    Message::parse(bytes)
        .unwrap()
        .u32(names::INFO_FLAGS)
        .unwrap()
}

fn plist_of(bytes: &[u8], key: FourCc) -> Value {
    let message = Message::parse(bytes).unwrap();
    plist::decode(message.property_list_bytes(key).unwrap().unwrap()).unwrap()
}

fn annex_b_sets() -> Vec<u8> {
    let mut sets = START_CODE.to_vec();
    sets.extend_from_slice(&SPS);
    sets.extend_from_slice(&START_CODE);
    sets.extend_from_slice(&PPS);
    sets
}

#[test]
fn the_hand_written_spss_read_as_they_say() {
    assert!(H264Sps::parse(&SPS).unwrap().frame_mbs_only_flag);
    assert!(!H264Sps::parse(&INTERLACED_SPS).unwrap().frame_mbs_only_flag);
}

#[test]
fn a_start_sends_the_buffer_attributes_then_its_reply() {
    let guest = Guest::new(Ok(Decoded::Landed));
    guest.start(&avcc(&SPS, 4));
    let (_, dpba) = guest.next_of(b"Dpba");
    let attributes = plist_of(&dpba, names::PIXEL_BUFFER_ATTRIBUTES);
    assert_eq!(
        attributes.get("PixelFormatType"),
        Some(&Value::Array(vec![Value::Integer(0x3432_3076)])),
        "420v"
    );
    assert_eq!(attributes.get("Width"), Some(&Value::Integer(64)));
    assert_eq!(attributes.get("Height"), Some(&Value::Integer(48)));
    assert_eq!(
        attributes.get("IOSurfaceProperties"),
        Some(&Value::Dictionary(Vec::new()))
    );
    let (header, reply) = guest.next_of(b"dsta");
    assert!(header.flags.contains(MessageFlags::REPLY_TO_GUEST));
    assert_eq!(status_of(&reply), Some(0));

    guest.start(&avcc(&SPS, 4));
    let (_, again) = guest.next_of(b"dsta");
    assert_eq!(
        status_of(&again),
        Some(status::VIDEO_DECODER_BAD_DATA),
        "a second start"
    );
}

#[test]
fn a_start_on_a_record_no_decoder_here_takes_is_refused_without_attributes() {
    for (record, refused) in [
        (avcc(&SPS, 3), status::VIDEO_DECODER_UNSUPPORTED_DATA_FORMAT),
        (avcc(&INTERLACED_SPS, 4), status::VIDEO_DECODER_MALFUNCTION),
        (vec![1, 2], status::VIDEO_DECODER_UNSUPPORTED_DATA_FORMAT),
    ] {
        let guest = Guest::new(Ok(Decoded::Landed));
        guest.start(&record);
        let (_, reply) = guest.next_of(b"dsta");
        assert_eq!(status_of(&reply), Some(refused));
        guest.frame(0, 0, &one_sample(&SAMPLE, None, false));
        let (_, frame) = guest.next_of(b"dfrm");
        assert_eq!(
            status_of(&frame),
            Some(NO_DECODER),
            "no decoder after a refusal"
        );
        assert!(guest.quiet());
    }
}

#[test]
fn a_frame_lands_in_the_buffer_the_guest_lends_then_is_emitted_and_let_go() {
    let guest = Guest::new(Ok(Decoded::Landed));
    guest.start(&avcc(&SPS, 4));
    guest.next_of(b"Dpba");
    guest.next_of(b"dsta");
    let buffer = GuestUuid([0xb0; 16]);
    for frame in 0..2 {
        guest.frame(frame, 0, &one_sample(&SAMPLE, None, false));
        let (dbuf, request) = guest.next_of(b"Dbuf");
        assert!(dbuf.flags.contains(MessageFlags::HOST_EXPECTS_REPLY));
        assert_eq!(
            Message::parse(&request).unwrap().s64(names::FRAME).unwrap(),
            Some(frame)
        );
        guest.answer(&dbuf, 0, Some((buffer, 7)));
        let (demt, emitted) = guest.next_of(b"Demt");
        assert_eq!(status_of(&emitted), Some(0));
        assert_eq!(info_of(&emitted), Some(0));
        let message = Message::parse(&emitted).unwrap();
        assert_eq!(
            message.uuid(names::PIXEL_BUFFER_UUID).unwrap(),
            Some(buffer)
        );
        guest.answer(&demt, 0, None);
        assert_eq!(guest.next(), surface_release(&[7]), "the surface goes back");
        let (header, reply) = guest.next_of(b"dfrm");
        assert_eq!(header.reply_id, 100 + frame as u64);
        assert_eq!((status_of(&reply), info_of(&reply)), (Some(0), Some(0)));
    }
    let calls = guest.calls.lock();
    let mut au = START_CODE.to_vec();
    au.extend_from_slice(&[0x65, 0x88]);
    assert_eq!(
        calls[0],
        (annex_b_sets(), au.clone(), 7),
        "the record's sets go first"
    );
    assert_eq!(calls[1], (Vec::new(), au, 7), "once");
}

#[test]
fn an_asynchronous_frame_is_answered_once_its_buffer_is_in_hand() {
    let guest = Guest::new(Ok(Decoded::Landed));
    guest.start(&avcc(&SPS, 4));
    guest.next_of(b"Dpba");
    guest.next_of(b"dsta");
    guest.frame(0, DECODE_ASYNCHRONOUSLY, &one_sample(&SAMPLE, None, false));
    let (dbuf, _) = guest.next_of(b"Dbuf");
    guest.answer(&dbuf, 0, Some((GuestUuid([1; 16]), 9)));
    let (_, reply) = guest.next_of(b"dfrm");
    assert_eq!(
        (status_of(&reply), info_of(&reply)),
        (Some(0), Some(INFO_ASYNCHRONOUS))
    );
    let (demt, emitted) = guest.next_of(b"Demt");
    assert_eq!(info_of(&emitted), Some(INFO_ASYNCHRONOUS));
    guest.answer(&demt, 0, None);
    assert_eq!(guest.next(), surface_release(&[9]));
    assert!(guest.quiet());
}

#[test]
fn a_frame_that_does_not_decode_is_emitted_without_its_buffer() {
    let guest = Guest::new(Err(status::VIDEO_DECODER_BAD_DATA));
    guest.start(&avcc(&SPS, 4));
    guest.next_of(b"Dpba");
    guest.next_of(b"dsta");
    let buffer = GuestUuid([2; 16]);
    guest.frame(0, 0, &one_sample(&SAMPLE, None, false));
    let (dbuf, _) = guest.next_of(b"Dbuf");
    guest.answer(&dbuf, 0, Some((buffer, 11)));
    let (demt, emitted) = guest.next_of(b"Demt");
    assert_eq!(status_of(&emitted), Some(status::VIDEO_DECODER_BAD_DATA));
    assert_eq!(
        Message::parse(&emitted)
            .unwrap()
            .uuid(names::PIXEL_BUFFER_UUID)
            .unwrap(),
        None
    );
    guest.answer(&demt, 0, None);
    let (_, unused) = guest.next_of(b"Dfub");
    assert_eq!(
        Message::parse(&unused)
            .unwrap()
            .uuid_array(names::PIXEL_BUFFER_UUIDS)
            .unwrap(),
        Some(vec![buffer])
    );
    assert_eq!(guest.next(), surface_release(&[11]));
    let (_, reply) = guest.next_of(b"dfrm");
    assert_eq!(
        status_of(&reply),
        Some(0),
        "the dfrm itself is answered noErr"
    );
}

#[test]
fn a_skipped_picture_is_emitted_dropped_and_its_buffer_unused() {
    let guest = Guest::new(Ok(Decoded::Skipped));
    guest.start(&avcc(&SPS, 4));
    guest.next_of(b"Dpba");
    guest.next_of(b"dsta");
    guest.frame(0, 0, &one_sample(&SAMPLE, None, false));
    let (dbuf, _) = guest.next_of(b"Dbuf");
    guest.answer(&dbuf, 0, Some((GuestUuid([3; 16]), 12)));
    let (demt, emitted) = guest.next_of(b"Demt");
    assert_eq!(
        (status_of(&emitted), info_of(&emitted)),
        (Some(0), Some(INFO_FRAME_DROPPED))
    );
    guest.answer(&demt, 0, None);
    guest.next_of(b"Dfub");
    assert_eq!(guest.next(), surface_release(&[12]));
    guest.next_of(b"dfrm");
}

#[test]
fn a_buffer_the_guest_refuses_emits_the_frame_with_its_status() {
    let guest = Guest::new(Ok(Decoded::Landed));
    guest.start(&avcc(&SPS, 4));
    guest.next_of(b"Dpba");
    guest.next_of(b"dsta");
    guest.frame(0, 0, &one_sample(&SAMPLE, None, false));
    let (dbuf, _) = guest.next_of(b"Dbuf");
    guest.answer(&dbuf, -6662, None);
    let (demt, emitted) = guest.next_of(b"Demt");
    assert_eq!(status_of(&emitted), Some(-6662));
    guest.answer(&demt, 0, None);
    let (_, reply) = guest.next_of(b"dfrm");
    assert_eq!(status_of(&reply), Some(0));
    assert!(guest.calls.lock().is_empty(), "nothing is decoded");
}

#[test]
fn a_sample_of_another_record_reopens_the_decoder_with_its_sets() {
    let guest = Guest::new(Ok(Decoded::Landed));
    guest.start(&avcc(&SPS, 4));
    guest.next_of(b"Dpba");
    guest.next_of(b"dsta");
    let record = avcc(&SPS, 2);
    guest.frame(0, 0, &one_sample(&[0, 2, 0x65, 0x88], Some(&record), false));
    guest.next_of(b"Dpba");
    let (dbuf, _) = guest.next_of(b"Dbuf");
    guest.answer(&dbuf, 0, Some((GuestUuid([4; 16]), 13)));
    let (demt, _) = guest.next_of(b"Demt");
    guest.answer(&demt, 0, None);
    let mut au = START_CODE.to_vec();
    au.extend_from_slice(&[0x65, 0x88]);
    assert_eq!(
        guest.calls.lock()[0],
        (annex_b_sets(), au, 13),
        "2-byte lengths read"
    );
}

#[test]
fn the_properties_are_listed_read_and_set() {
    let guest = Guest::new(Ok(Decoded::Landed));
    let key = |m: &mut MessageWriter, key: &str| {
        m.property_list(names::PROPERTY_KEY, Some(&Value::string(key)));
    };
    guest.send(
        b"dsup",
        MessageFlags::GUEST_EXPECTS_REPLY,
        3,
        Vec::new(),
        |_| {},
    );
    let (_, supported) = guest.next_of(b"dsup");
    let Value::Dictionary(entries) = plist_of(&supported, names::SUPPORTED_PROPERTIES) else {
        panic!("a dictionary");
    };
    let keys: Vec<&str> = entries.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, [USING_HARDWARE, REAL_TIME, MAXIMIZE_POWER_EFFICIENCY]);

    guest.send(
        b"dcop",
        MessageFlags::GUEST_EXPECTS_REPLY,
        4,
        Vec::new(),
        |m| key(m, USING_HARDWARE),
    );
    let (_, copied) = guest.next_of(b"dcop");
    assert_eq!(
        plist_of(&copied, names::PROPERTY_VALUE),
        Value::Boolean(true)
    );

    guest.send(
        b"dset",
        MessageFlags::GUEST_EXPECTS_REPLY,
        5,
        Vec::new(),
        |m| {
            key(m, REAL_TIME);
            m.property_list(names::PROPERTY_VALUE, Some(&Value::Boolean(true)));
        },
    );
    let (_, set) = guest.next_of(b"dset");
    assert_eq!(status_of(&set), Some(0));
    guest.send(
        b"dcop",
        MessageFlags::GUEST_EXPECTS_REPLY,
        6,
        Vec::new(),
        |m| key(m, REAL_TIME),
    );
    let (_, copied) = guest.next_of(b"dcop");
    assert_eq!(
        plist_of(&copied, names::PROPERTY_VALUE),
        Value::Boolean(true)
    );

    guest.send(
        b"dset",
        MessageFlags::GUEST_EXPECTS_REPLY,
        7,
        Vec::new(),
        |m| key(m, USING_HARDWARE),
    );
    let (_, read_only) = guest.next_of(b"dset");
    assert_eq!(status_of(&read_only), Some(status::PROPERTY_READ_ONLY));
    guest.send(
        b"dcop",
        MessageFlags::GUEST_EXPECTS_REPLY,
        8,
        Vec::new(),
        |m| key(m, "Nope"),
    );
    let (_, unknown) = guest.next_of(b"dcop");
    assert_eq!(status_of(&unknown), Some(status::PROPERTY_NOT_SUPPORTED));

    // Paravirtualized is the guest's: no reply.
    guest.send(
        b"dset",
        MessageFlags::GUEST_EXPECTS_REPLY,
        9,
        Vec::new(),
        |m| key(m, PARAVIRTUALIZED),
    );
    assert!(guest.quiet());
}

#[test]
fn can_accept_answers_whether_the_decoder_goes_on() {
    let guest = Guest::new(Ok(Decoded::Landed));
    let can = |record: &[u8], width: u16| {
        let description = hand_written(b"avc1", width, 48, record);
        guest.send(
            b"dcan",
            MessageFlags::GUEST_EXPECTS_REPLY,
            10,
            Vec::new(),
            |m| {
                m.image_description(names::FORMAT_DESCRIPTION, &description);
            },
        );
        let (_, reply) = guest.next_of(b"dcan");
        Message::parse(&reply)
            .unwrap()
            .s32(names::CAN_ACCEPT)
            .unwrap()
    };
    assert_eq!(can(&avcc(&SPS, 4), 64), Some(0), "no decoder yet");
    guest.start(&avcc(&SPS, 4));
    guest.next_of(b"Dpba");
    guest.next_of(b"dsta");
    assert_eq!(can(&avcc(&SPS, 4), 64), Some(1));
    assert_eq!(can(&avcc(&SPS, 2), 64), Some(0), "another length size");
    assert_eq!(can(&avcc(&SPS, 4), 80), Some(0), "another size");
    assert_eq!(
        can(&avcc(&INTERLACED_SPS, 4), 64),
        Some(0),
        "a record it would refuse"
    );
}

#[test]
fn an_invalidated_decoder_answers_frames_with_no_decoder() {
    let guest = Guest::new(Ok(Decoded::Landed));
    guest.start(&avcc(&SPS, 4));
    guest.next_of(b"Dpba");
    guest.next_of(b"dsta");
    guest.send(
        b"dinv",
        MessageFlags::GUEST_EXPECTS_REPLY,
        11,
        Vec::new(),
        |_| {},
    );
    let (_, invalidated) = guest.next_of(b"dinv");
    assert_eq!(status_of(&invalidated), Some(0));
    guest.frame(0, DECODE_ASYNCHRONOUSLY, &one_sample(&SAMPLE, None, false));
    let (_, reply) = guest.next_of(b"dfrm");
    assert_eq!(
        (status_of(&reply), info_of(&reply)),
        (Some(NO_DECODER), Some(0))
    );
    guest.send(
        b"dfin",
        MessageFlags::GUEST_EXPECTS_REPLY,
        12,
        Vec::new(),
        |_| {},
    );
    let (_, finished) = guest.next_of(b"dfin");
    assert_eq!(status_of(&finished), Some(status::UNIMPLEMENTED));
}

#[test]
fn a_late_reply_gives_back_what_it_lent_and_a_close_ends_the_session() {
    let guest = Guest::new(Ok(Decoded::Landed));
    let late = MessageHeader {
        message_type: FourCc::new(*b"Dbuf"),
        flags: MessageFlags::NONE,
        guest_uuid: guest.uuid,
        reply_id: 999,
    };
    guest.answer(&late, 0, Some((GuestUuid([5; 16]), 21)));
    assert_eq!(guest.next(), surface_release(&[21]));

    let answer = guest.send(b"-dec", MessageFlags::NONE, 0, Vec::new(), |_| {});
    assert_eq!(answer.status, 0);
    let answer = guest.send(
        b"dsup",
        MessageFlags::GUEST_EXPECTS_REPLY,
        13,
        Vec::new(),
        |_| {},
    );
    assert_eq!(answer.status, status::SESSION_NOT_FOUND);
}

#[test]
fn a_create_of_a_codec_the_list_lacks_is_refused() {
    let guest = Guest::new(Ok(Decoded::Landed));
    let mut message = MessageWriter::new(
        FourCc::new(*b"+dec"),
        MessageFlags::GUEST_EXPECTS_REPLY,
        Some(GuestUuid([6; 16])),
        1,
    );
    message.s32(names::CODEC, i32::from_be_bytes(*b"hvc1"));
    message.u32(names::VERSION, 6);
    let answer = guest
        .host
        .deliver_lending(&message.finish(), Vec::new(), None);
    assert_eq!(
        status_of(&answer.replies[0]),
        Some(status::COULD_NOT_FIND_VIDEO_DECODER)
    );
}
