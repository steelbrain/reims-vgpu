//! A decoder session driven as the guest's VideoToolbox drives it, on a backend that records what
//! it is given. Every stream here is written by hand: an H.264 SPS and PPS of a 64x48 baseline
//! stream, their `avcC`, and slices of a few bytes, which the fake backend never decodes.

use std::sync::mpsc;

use super::*;
use crate::codec::hevc::tests::BitWriter;
use crate::core::runtime::StdRuntime;
use crate::core::sync::Mutex;
use crate::paravirt::host::Decoder;
use crate::paravirt::image_description::tests::hand_written;
use crate::paravirt::message::read_header;
use crate::paravirt::sample_buffer::tests::one_sample;
use crate::paravirt::session::surface_release;
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

/// The regions the fake backend's decoders landed tiles in, a tile each.
type Regions = Arc<Mutex<Vec<TileRegion>>>;

struct FakeBackend {
    calls: Calls,
    regions: Regions,
    result: Result<Decoded, OsStatus>,
    tiles: Option<TileLimits>,
}

struct FakeDecoder {
    calls: Calls,
    regions: Regions,
    result: Result<Decoded, OsStatus>,
    tiles: Option<TileLimits>,
}

impl FakeBackend {
    /// One whose decoders land every picture, and no tiles, recording what no test reads.
    fn landing() -> Self {
        Self {
            calls: Calls::default(),
            regions: Regions::default(),
            result: Ok(Decoded::Landed),
            tiles: None,
        }
    }
}

impl Backend for FakeBackend {
    fn open(&self, _config: &DecoderConfig) -> Result<Box<dyn BackendDecoder>, OsStatus> {
        Ok(Box::new(FakeDecoder {
            calls: Arc::clone(&self.calls),
            regions: Arc::clone(&self.regions),
            result: self.result,
            tiles: self.tiles,
        }))
    }
}

/// A fake backend whose arm decodes 4:4:4 streams as well as 4:2:0 ones.
struct FourFourFour(FakeBackend);

impl Backend for FourFourFour {
    fn open(&self, config: &DecoderConfig) -> Result<Box<dyn BackendDecoder>, OsStatus> {
        self.0.open(config)
    }

    fn declines(&self, stream: &Stream) -> Option<String> {
        (!matches!(stream.chroma_format_idc, 1 | 3))
            .then(|| "the arm decodes 4:2:0 and 4:4:4 pictures alone".to_owned())
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

    fn tile_limits(&self) -> Option<TileLimits> {
        self.tiles
    }

    fn decode_tile(
        &mut self,
        parameter_sets: &[u8],
        access_unit: &[u8],
        output: &dyn LentSurface,
        region: TileRegion,
    ) -> Result<Decoded, OsStatus> {
        if self.tiles.is_none() {
            return Err(status::UNIMPLEMENTED);
        }
        self.regions.lock().push(region);
        self.decode(parameter_sets, access_unit, output)
    }
}

/// The guest's side of one session: the host, the messages it sent, and the session's UUID.
struct Guest {
    host: Host,
    link: Arc<dyn Link>,
    sent: mpsc::Receiver<Vec<u8>>,
    uuid: GuestUuid,
    calls: Calls,
    regions: Regions,
    /// The session's codec, as its descriptions name it.
    codec: [u8; 4],
}

impl Guest {
    fn new(result: Result<Decoded, OsStatus>) -> Self {
        Self::with_tiles(result, None)
    }

    /// A session on a backend whose decoders decode `tiles`.
    fn with_tiles(result: Result<Decoded, OsStatus>, tiles: Option<TileLimits>) -> Self {
        Self::of(Codec::H264, result, tiles)
    }

    /// A session of `codec` on a backend whose decoders decode `tiles`.
    fn of(codec: Codec, result: Result<Decoded, OsStatus>, tiles: Option<TileLimits>) -> Self {
        Self::on(codec, result, tiles, false)
    }

    /// [`Guest::of`], on an arm that decodes 4:4:4 streams too where `four_four_four`.
    fn on(
        codec: Codec,
        result: Result<Decoded, OsStatus>,
        tiles: Option<TileLimits>,
        four_four_four: bool,
    ) -> Self {
        let calls = Calls::default();
        let regions = Regions::default();
        let fake = FakeBackend {
            calls: Arc::clone(&calls),
            regions: Arc::clone(&regions),
            result,
            tiles,
        };
        let backend: Arc<dyn Backend> = if four_four_four {
            Arc::new(FourFourFour(fake))
        } else {
            Arc::new(fake)
        };
        let (fourcc, profiles, max_level) = match codec {
            Codec::H264 => (*b"avc1", vec![66, 77, 100], 52),
            Codec::Hevc => (*b"hvc1", vec![1, 2, 3], 156),
        };
        let decoders = vec![Decoder {
            codec,
            profiles,
            max_level,
        }];
        let (sender, sent) = mpsc::channel();
        let guest = Self {
            host: Host::with_backend(decoders, backend, StdRuntime::shared()),
            link: Arc::new(FakeLink(Mutex::new(sender))),
            sent,
            uuid: GuestUuid([0x5a; 16]),
            calls,
            regions,
            codec: fourcc,
        };
        let answer = guest.send(
            b"+dec",
            MessageFlags::GUEST_EXPECTS_REPLY,
            1,
            Vec::new(),
            |m| {
                m.s32(names::CODEC, i32::from_be_bytes(fourcc));
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
        let description = hand_written(&self.codec, 64, 48, record);
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

    /// Starts a tile session on a 64x48 description of `record`.
    fn start_tiles(&self, record: &[u8]) {
        let description = hand_written(&self.codec, 64, 48, record);
        self.send(
            b"dsts",
            MessageFlags::GUEST_EXPECTS_REPLY,
            3,
            Vec::new(),
            |m| {
                m.image_description(names::FORMAT_DESCRIPTION, &description);
            },
        );
    }

    /// Sends tile `tile`, its sample `sample`, landing the `size` columns and rows from `origin`
    /// at `at` of the surface `surface` it lends, the buffer `buffer`.
    fn tile(
        &self,
        tile: i64,
        (origin, size, at): ((i32, i32), (i32, i32), (i32, i32)),
        buffer: GuestUuid,
        surface: u64,
        sample: &[u8],
    ) {
        self.send(
            b"dtil",
            MessageFlags::GUEST_EXPECTS_REPLY,
            200 + tile as u64,
            vec![Arc::new(Surface(surface))],
            |m| {
                m.s64(names::TILE, tile);
                m.i32_pair(names::TILE_SOURCE_ORIGIN, ValueType::Point, origin);
                m.i32_pair(names::TILE_SOURCE_SIZE, ValueType::Size, size);
                m.uuid(names::TILE_BUFFER_UUID, buffer);
                m.i32_pair(names::TILE_OUTPUT_POINT, ValueType::Point, at);
                m.u32(names::DECODE_FLAGS, 1);
                m.sample_buffer_with_limit(names::SAMPLE_BUFFER, usize::MAX, sample);
            },
        );
    }

    /// Checks the host's `TDem` of tile `tile` with `status` and `buffer`, answers it, and checks
    /// that `surface` goes back and the `dtil` is answered noErr.
    fn tile_emitted(&self, tile: i64, status: OsStatus, buffer: GuestUuid, surface: u64) {
        let (tdem, emitted) = self.next_of(b"TDem");
        assert!(tdem.flags.contains(MessageFlags::HOST_EXPECTS_REPLY));
        let message = Message::parse(&emitted).unwrap();
        assert_eq!(message.s64(names::TILE).unwrap(), Some(tile));
        assert_eq!(status_of(&emitted), Some(status));
        assert_eq!(info_of(&emitted), Some(0));
        assert_eq!(message.uuid(names::TILE_BUFFER_UUID).unwrap(), Some(buffer));
        self.answer(&tdem, 0, None);
        let (header, reply) = self.next_of(b"dtil");
        assert_eq!(header.reply_id, 200 + tile as u64);
        assert_eq!((status_of(&reply), info_of(&reply)), (Some(0), Some(0)));
        assert_eq!(
            self.next(),
            surface_release(&[surface]),
            "the surface goes back"
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
    assert!(
        guest.quiet(),
        "a message that lends nothing is let go of with nothing"
    );

    let answer = guest.send(
        b"Dbuf",
        MessageFlags::REPLY_TO_HOST,
        14,
        vec![Arc::new(Surface(22))],
        |_| {},
    );
    assert_eq!(answer.status, status::SESSION_NOT_FOUND);
    assert_eq!(
        guest.next(),
        surface_release(&[22]),
        "a surface lent to no session goes back"
    );
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

/// The tiles the fake backend's decoders decode, where a test lets them.
const TILES: TileLimits = TileLimits {
    max_width: 8192,
    max_height: 4352,
    min_width: 64,
    min_height: 64,
};

#[test]
fn a_tile_session_sends_its_requirements_then_its_reply() {
    let guest = Guest::with_tiles(Ok(Decoded::Landed), Some(TILES));
    guest.start_tiles(&avcc(&SPS, 4));
    let (tdrs, requirements) = guest.next_of(b"TDrs");
    assert!(!tdrs.flags.contains(MessageFlags::HOST_EXPECTS_REPLY));
    let attributes = plist_of(&requirements, names::TILE_BUFFER_ATTRIBUTES);
    assert_eq!(
        attributes.get("PixelFormatType"),
        Some(&Value::Array(vec![Value::Integer(0x3432_3076)])),
        "420v"
    );
    assert_eq!(attributes.get("Width"), None, "the guest sizes its buffers");
    let tiles = plist_of(&requirements, names::TILE_REQUIREMENTS);
    for (key, value) in [
        ("MaximumTileWidth", 8192),
        ("MaximumTileHeight", 4352),
        ("MinimumTileWidth", 64),
        ("MinimumTileHeight", 64),
        ("MinimumDecodableWidth", 64),
        ("MinimumDecodableHeight", 64),
        ("TileWidthPixelAlignment", 8),
        ("TileHeightPixelAlignment", 8),
        ("TileByteAlignment", 64),
        ("MaximumCanvasRowBytes", 16384),
    ] {
        assert_eq!(tiles.get(key), Some(&Value::Integer(value)), "{key}");
    }
    let (header, reply) = guest.next_of(b"dsts");
    assert!(header.flags.contains(MessageFlags::REPLY_TO_GUEST));
    assert_eq!(status_of(&reply), Some(0));
}

#[test]
fn a_tile_session_on_a_backend_that_decodes_no_tiles_is_refused_unimplemented() {
    let guest = Guest::new(Ok(Decoded::Landed));
    guest.start_tiles(&avcc(&SPS, 4));
    let (_, reply) = guest.next_of(b"dsts");
    assert_eq!(
        status_of(&reply),
        Some(status::UNIMPLEMENTED),
        "no TDrs first"
    );
    let buffer = GuestUuid([0xc0; 16]);
    guest.tile(
        0,
        ((0, 0), (64, 48), (0, 0)),
        buffer,
        30,
        &one_sample(&SAMPLE, None, false),
    );
    guest.tile_emitted(0, NO_DECODER, buffer, 30);
    assert!(guest.calls.lock().is_empty());
    assert!(guest.quiet());
}

/// A 64x48 SPS of the High profile `profile_idc`, of `chroma_format_idc` and `bit_depth_minus8`
/// (luma's and chroma's), progressive, with no VUI.
fn high_sps(profile_idc: u8, chroma_format_idc: u32, bit_depth_minus8: u32) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.u(8, u32::from(profile_idc)).u(8, 0).u(8, 30).ue(0);
    w.ue(chroma_format_idc);
    if chroma_format_idc == 3 {
        w.bit(false); // separate_colour_plane_flag
    }
    // The depths, then no transform bypass and no scaling matrix.
    w.ue(bit_depth_minus8)
        .ue(bit_depth_minus8)
        .bit(false)
        .bit(false);
    // log2_max_frame_num_minus4, pic_order_cnt_type 0 and its lsb's size; one reference, no gaps.
    w.ue(0).ue(0).ue(0).ue(1).bit(false);
    // Four by three macroblocks, frames alone, direct 8x8 inference, no cropping, no VUI.
    w.ue(3).ue(2).bit(true).bit(true).bit(false).bit(false);
    let mut unit = vec![0x67];
    unit.extend(w.finish());
    unit
}

/// An `hvcC` of `sps`, with a VPS and a PPS the host passes on unread.
fn hvcc_of(sps: &[u8]) -> Vec<u8> {
    let vps = [0x40, 0x01, 0x0c];
    let pps = [0x44, 0x01, 0xc0];
    crate::codec::encode::hevc::hvcc(&vps, sps, &pps).expect("an hvcC")
}

/// Starts a decoder session and a tile session of `codec` on `record`, and checks that each
/// names `format` for its buffers, or, where it is `None`, that each is refused at its start:
/// the decoder session `kVTVideoDecoderUnsupportedDataFormatErr` without `Dpba`, the tile
/// session `kVTUnimplementedErr` without `TDrs`, and neither decodes after.
fn starts_in(
    codec: Codec,
    record: &[u8],
    four_four_four: bool,
    format: Option<&[u8; 4]>,
    what: &str,
) {
    let named =
        format.map(|f| Value::Array(vec![Value::Integer(i64::from(u32::from_be_bytes(*f)))]));
    let guest = Guest::on(codec, Ok(Decoded::Landed), Some(TILES), four_four_four);
    guest.start(record);
    if let Some(named) = &named {
        let (_, dpba) = guest.next_of(b"Dpba");
        let attributes = plist_of(&dpba, names::PIXEL_BUFFER_ATTRIBUTES);
        assert_eq!(attributes.get("PixelFormatType"), Some(named), "{what}");
    }
    let (_, reply) = guest.next_of(b"dsta");
    let refused = status::VIDEO_DECODER_UNSUPPORTED_DATA_FORMAT;
    assert_eq!(
        status_of(&reply),
        Some(if named.is_some() { 0 } else { refused }),
        "{what}"
    );
    if named.is_none() {
        guest.frame(0, 0, &one_sample(&SAMPLE, None, false));
        let (_, frame) = guest.next_of(b"dfrm");
        assert_eq!(status_of(&frame), Some(NO_DECODER), "{what}");
    }
    assert!(guest.quiet(), "{what}");

    let guest = Guest::on(codec, Ok(Decoded::Landed), Some(TILES), four_four_four);
    guest.start_tiles(record);
    if let Some(named) = &named {
        let (_, tdrs) = guest.next_of(b"TDrs");
        let attributes = plist_of(&tdrs, names::TILE_BUFFER_ATTRIBUTES);
        assert_eq!(attributes.get("PixelFormatType"), Some(named), "{what}");
    }
    let (_, reply) = guest.next_of(b"dsts");
    let refused = status::UNIMPLEMENTED;
    assert_eq!(
        status_of(&reply),
        Some(if named.is_some() { 0 } else { refused }),
        "{what}"
    );
    if named.is_none() {
        let buffer = GuestUuid([0xc1; 16]);
        guest.tile(
            0,
            ((0, 0), (64, 48), (0, 0)),
            buffer,
            31,
            &one_sample(&SAMPLE, None, false),
        );
        guest.tile_emitted(0, NO_DECODER, buffer, 31);
        assert!(guest.calls.lock().is_empty(), "{what}");
    }
    assert!(guest.quiet(), "{what}");
}

#[test]
fn each_hevc_stream_lands_in_its_format_or_is_refused_at_its_start() {
    // general_profile_idc, chroma_format_idc, the depths past 8, the range extensions, the
    // VUI's range, and the format the pictures land in, on an arm of 4:2:0 alone and on one of
    // 4:4:4 too. Apple's HEIC images hold 4:2:0 items of Main Still Picture and Main, and 4:4:4
    // thumbnails and monochrome alpha items of the range extensions' profile (read with ffprobe
    // on an M3 Max's own, macOS 27.0.1): the thumbnails land where the arm decodes 4:4:4, and
    // the rest of the range extensions' streams nowhere, each refused at its start, not at each
    // tile.
    for (profile, chroma, deeper, range_extension, full_range, format, with_444) in [
        (1, 1, 0, false, None, Some(b"420v"), Some(b"420v")),
        (1, 1, 0, false, Some(true), Some(b"420f"), Some(b"420f")),
        (3, 1, 0, false, Some(true), Some(b"420f"), Some(b"420f")),
        (2, 1, 2, false, Some(false), Some(b"x420"), Some(b"x420")),
        (2, 1, 2, false, Some(true), Some(b"xf20"), Some(b"xf20")),
        (4, 3, 0, true, Some(true), None, Some(b"444f")),
        (4, 3, 0, true, None, None, Some(b"444v")),
        (4, 3, 2, true, None, None, None),
        (4, 2, 0, true, Some(true), None, None),
        (4, 2, 2, true, None, None, None),
        (4, 0, 0, true, Some(true), None, None),
        (1, 3, 0, false, Some(true), None, None),
        (2, 2, 2, false, None, None, None),
        (1, 0, 0, false, None, None, None),
        (2, 1, 4, false, None, None, None),
        (4, 1, 0, true, None, None, None),
    ] {
        let sps =
            crate::codec::hevc::tests::sps_of(profile, chroma, deeper, range_extension, full_range);
        let what = format!("profile {profile}, chroma {chroma}, 8+{deeper} bits, {full_range:?}");
        starts_in(Codec::Hevc, &hvcc_of(&sps), false, format, &what);
        let what = format!("{what}, on an arm of 4:4:4");
        starts_in(Codec::Hevc, &hvcc_of(&sps), true, with_444, &what);
    }
}

/// What a session on `backend` refuses of a record's stream, the codec's refusal or the arm's:
/// `None` where both take it.
fn undecoded_on(backend: &dyn Backend, codec: Codec, record: Vec<u8>) -> Option<Undecoded> {
    let config = DecoderConfig {
        codec,
        width: 64,
        height: 48,
        record,
    };
    read_record(&config)
        .ok()
        .and_then(|record| undecoded(backend, &record))
}

#[test]
fn a_refused_stream_is_named_by_its_profile_chroma_format_depths_and_size() {
    // Apple's HEIC images' 4:4:4 thumbnails and monochrome alpha items are of the range
    // extensions' profile, as the HEVC test of the formats above notes: the codec takes the
    // thumbnails, and an arm of 4:2:0 alone declines them.
    let arm = FakeBackend::landing();
    let thumbnail = crate::codec::hevc::tests::sps_of(4, 3, 0, true, Some(true));
    let refused = undecoded_on(&arm, Codec::Hevc, hvcc_of(&thumbnail)).expect("refused");
    assert_eq!(
        refused.to_string(),
        "the arm decodes 4:2:0 pictures alone (HEVC general_profile_idc 4, chroma_format_idc 3, \
         luma 8 bits, chroma 8 bits, 64x48)"
    );
    assert_eq!(
        refused.kind(),
        "HEVC profile four chroma format three, luma eight bits, chroma eight bits"
    );
    let four_four_four = FourFourFour(FakeBackend::landing());
    assert_eq!(
        undecoded_on(&four_four_four, Codec::Hevc, hvcc_of(&thumbnail)),
        None
    );
    let alpha = crate::codec::hevc::tests::sps_of(4, 0, 2, true, None);
    let refused = undecoded_on(&four_four_four, Codec::Hevc, hvcc_of(&alpha)).expect("refused");
    assert_eq!(
        refused.to_string(),
        "unsupported: range extensions streams other than 4:4:4 at 8 bits (HEVC \
         general_profile_idc 4, chroma_format_idc 0, luma 10 bits, chroma 10 bits, 64x48)"
    );
    let high =
        undecoded_on(&four_four_four, Codec::H264, avcc(&high_sps(244, 3, 2), 4)).expect("refused");
    assert!(
        high.to_string().ends_with(
            "(H.264 profile_idc 244, chroma_format_idc 3, luma 10 bits, chroma 10 bits, 64x48)"
        ),
        "{high}"
    );
    assert_eq!(
        high.kind(),
        "AVC profile two-four-four chroma format three, luma one-zero bits, chroma one-zero bits"
    );
    // A stream the host decodes names nothing.
    let main = crate::codec::hevc::tests::sps_of(1, 1, 0, false, None);
    assert_eq!(undecoded_on(&arm, Codec::Hevc, hvcc_of(&main)), None);
}

#[test]
fn each_kind_of_refused_stream_has_a_kind_of_its_own_whatever_its_size() {
    // The logger reads a kind up to its first digit, quote, backquote or `;`
    // (`reims_vgpu_vulkan::core::logging::class`): none is in a kind, so the whole of it tells
    // the streams apart.
    let arm = FakeBackend::landing();
    let kind_of = |profile, chroma, deeper| {
        let sps = crate::codec::hevc::tests::sps_of(profile, chroma, deeper, true, None);
        let kind = undecoded_on(&arm, Codec::Hevc, hvcc_of(&sps))
            .expect("refused")
            .kind();
        assert!(
            !kind.contains(|c: char| c.is_ascii_digit() || "\"`;".contains(c)),
            "{kind}"
        );
        kind
    };
    let kinds = [
        kind_of(4, 3, 0),
        kind_of(4, 0, 0),
        kind_of(4, 2, 2),
        kind_of(4, 3, 4),
        kind_of(4, 1, 4),
    ];
    for (i, kind) in kinds.iter().enumerate() {
        assert!(!kinds[..i].contains(kind), "{kind}");
    }
    let sps = crate::codec::hevc::tests::sps_of(4, 3, 0, true, None);
    let small = undecoded_on(&arm, Codec::Hevc, hvcc_of(&sps)).expect("refused");
    let large = Undecoded {
        stream: Stream {
            width: 4032,
            height: 3024,
            ..small.stream
        },
        ..small.clone()
    };
    assert_eq!(small.kind(), large.kind());
    assert_ne!(small.to_string(), large.to_string());
}

#[test]
fn each_h264_stream_lands_in_its_4_2_0_format_or_is_refused_at_its_start() {
    // profile_idc, chroma_format_idc, the depths past 8, and the format the pictures land in:
    // 8-bit 4:2:0 alone decodes here.
    for (profile, chroma, deeper, format) in [
        (100, 1, 0, Some(b"420v")),
        (100, 0, 0, None),
        (110, 1, 2, None),
        (122, 2, 0, None),
        (122, 2, 2, None),
        (244, 3, 0, None),
        (244, 3, 2, None),
    ] {
        let what = format!("profile {profile}, chroma {chroma}, 8+{deeper} bits");
        // On an arm of 4:4:4 too: the codec refuses H.264's.
        starts_in(
            Codec::H264,
            &avcc(&high_sps(profile, chroma, deeper), 4),
            true,
            format,
            &what,
        );
    }
}

#[test]
fn a_tile_lands_its_region_in_the_surface_it_lends_then_is_emitted_and_let_go() {
    let guest = Guest::with_tiles(Ok(Decoded::Landed), Some(TILES));
    guest.start_tiles(&avcc(&SPS, 4));
    guest.next_of(b"TDrs");
    guest.next_of(b"dsts");
    let buffer = GuestUuid([0xc1; 16]);
    guest.tile(
        0,
        ((0, 0), (64, 48), (0, 0)),
        buffer,
        31,
        &one_sample(&SAMPLE, None, false),
    );
    guest.tile_emitted(0, 0, buffer, 31);
    guest.tile(
        1,
        ((16, 8), (47, 39), (64, 48)),
        buffer,
        32,
        &one_sample(&SAMPLE, None, false),
    );
    guest.tile_emitted(1, 0, buffer, 32);
    let mut au = START_CODE.to_vec();
    au.extend_from_slice(&[0x65, 0x88]);
    assert_eq!(
        *guest.calls.lock(),
        vec![(annex_b_sets(), au.clone(), 31), (Vec::new(), au, 32)],
        "the record's sets go first, once"
    );
    assert_eq!(
        *guest.regions.lock(),
        vec![
            TileRegion {
                source_x: 0,
                source_y: 0,
                width: 64,
                height: 48,
                x: 0,
                y: 0,
            },
            TileRegion {
                source_x: 16,
                source_y: 8,
                width: 47,
                height: 39,
                x: 64,
                y: 48,
            },
        ],
        "an odd width and height at the image's edge"
    );
    guest.send(
        b"dtfi",
        MessageFlags::GUEST_EXPECTS_REPLY,
        4,
        Vec::new(),
        |_| {},
    );
    let (_, finished) = guest.next_of(b"dtfi");
    assert_eq!(status_of(&finished), Some(0), "no tile left to finish");
    assert!(guest.quiet());
}

#[test]
fn a_tile_the_host_cannot_place_is_emitted_with_a_parameter_error_undecoded() {
    let guest = Guest::with_tiles(Ok(Decoded::Landed), Some(TILES));
    guest.start_tiles(&avcc(&SPS, 4));
    guest.next_of(b"TDrs");
    guest.next_of(b"dsts");
    let buffer = GuestUuid([0xc2; 16]);
    for (tile, region) in [
        (0, ((0, 0), (64, 48), (4, 0))),
        (1, ((1, 0), (62, 48), (0, 0))),
        (2, ((0, 0), (0, 48), (0, 0))),
        (3, ((0, -2), (64, 48), (0, 0))),
    ] {
        guest.tile(tile, region, buffer, 33, &one_sample(&SAMPLE, None, false));
        guest.tile_emitted(tile, status::PARAMETER, buffer, 33);
    }
    assert!(guest.calls.lock().is_empty());
}

#[test]
fn a_tile_that_does_not_decode_is_emitted_with_its_status_and_the_next_gets_the_sets() {
    let guest = Guest::with_tiles(Err(status::VIDEO_DECODER_BAD_DATA), Some(TILES));
    guest.start_tiles(&avcc(&SPS, 4));
    guest.next_of(b"TDrs");
    guest.next_of(b"dsts");
    let buffer = GuestUuid([0xc3; 16]);
    for tile in 0..2 {
        guest.tile(
            tile,
            ((0, 0), (64, 48), (0, 0)),
            buffer,
            34,
            &one_sample(&SAMPLE, None, false),
        );
        guest.tile_emitted(tile, status::VIDEO_DECODER_BAD_DATA, buffer, 34);
    }
    let calls = guest.calls.lock();
    assert_eq!(calls.len(), 2);
    assert!(calls.iter().all(|(sets, _, _)| *sets == annex_b_sets()));
}
