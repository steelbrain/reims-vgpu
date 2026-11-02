use std::sync::Arc;

use reims_vgpu_core::testkit::FakePlatform;
use reims_vgpu_video_core::backend::{
    BackendDecoder, BackendError, Codec, CodecCapabilities, CompressedFrame, DecodedFrame,
    DecoderConfig, FrameDecoded, PixelFormat,
};
use reims_vgpu_video_core::encoder_backend::{
    BackendEncoder, EncodedFrame, EncoderBackend, EncoderCapabilities, EncoderConfig,
    EncoderSettings, SourceFrame,
};
use reims_vgpu_video_core::wire::surface_boxes::read_surface_release;

use super::*;

/// A watcher of the host's devices' writes that keeps what it is told.
struct Seen(Mutex<Vec<std::ops::Range<u64>>>);

impl crate::device_writes::Watcher for Seen {
    fn wrote(&self, runs: &[std::ops::Range<u64>]) {
        self.0.lock().extend_from_slice(runs);
    }
}

/// A device's write of a held surface's memory reports the surface's pages to the host's
/// devices' watchers; a lend for writing that fails reports nothing
#[test]
fn a_held_surfaces_writes_report_its_pages() {
    let p = Arc::new(FakePlatform::new());
    let surface = p.fake_surfaces().make_simple(4, 4, 0, 4);
    // Addresses no other test writes, as the watchers are the process's.
    let at = 0x5ee0_0000_0000;
    let held = MappedSurface {
        mapping_id: 7,
        surface: Ours::new(surface),
        platform: Arc::clone(&p),
        device: OnceLock::new(),
        pages: crate::device_writes::page_runs(&[at, at + 0x4000, at + 0x10000], 0x4000),
    };
    let seen = Arc::new(Seen(Mutex::new(Vec::new())));
    let watcher: Arc<dyn crate::device_writes::Watcher> = Arc::clone(&seen) as _;
    crate::device_writes::watch(&watcher);
    assert!(
        held.write(&mut |_| {}).is_err(),
        "the fake platform lends nothing"
    );
    assert!(seen.0.lock().is_empty());
    held.device_wrote();
    assert_eq!(
        seen.0.lock().as_slice(),
        [at..at + 0x8000, at + 0x10000..at + 0x14000]
    );
}

/// A `subs` atom naming `ids`, the rest 0.
fn subs(ids: &[u64]) -> Vec<u8> {
    let mut m = Vec::new();
    m.extend_from_slice(&0x30u32.to_be_bytes());
    m.extend_from_slice(b"subs");
    for i in 0..5 {
        m.extend_from_slice(&ids.get(i).copied().unwrap_or(0).to_be_bytes());
    }
    m
}

/// A `vtpv` request of type `kind` the guest expects a reply to, of the zero UUID and reply
/// identifier `reply`, with `,ver` as `=u32` 6: the guest's `decs` and `encs`, written by hand.
fn request(kind: &[u8; 4], reply: u64) -> Vec<u8> {
    let mut m = Vec::new();
    m.extend_from_slice(&0x30u32.to_be_bytes());
    m.extend_from_slice(b"vtpv");
    m.extend_from_slice(kind);
    m.extend_from_slice(&1u32.to_be_bytes());
    m.extend_from_slice(&[0; 16]);
    m.extend_from_slice(&reply.to_be_bytes());
    m.extend_from_slice(&[0; 8]);
    m.extend_from_slice(&20u32.to_be_bytes());
    m.extend_from_slice(b"para,ver=u32");
    m.extend_from_slice(&6u32.to_be_bytes());
    m
}

type Sent = Arc<Mutex<Vec<Vec<u8>>>>;

/// Why the tests' devices open no backend of the host's (`VideoDevice::without_backend`).
const NO_BACKEND: &str = "a test's, on the testkit platform's virtual clock";

/// A device over `sent`, which opens none of the host's backends: what these tests check is the
/// device's, the same whatever the host decodes on.
fn device(sent: &Sent) -> VideoDevice<FakePlatform> {
    let sent = Arc::clone(sent);
    let send: SendToGuest = Arc::new(move |message| sent.lock().push(message.to_vec()));
    VideoDevice::without_backend(
        Arc::new(FakePlatform::new()),
        VideoDeviceConfig {
            iosurface_device: None,
            send_to_guest: Some(send),
        },
        NO_BACKEND,
    )
}

/// The header fields of a message the host sent: its type, flags and reply identifier.
fn header_of(message: &[u8]) -> ([u8; 4], u32, u64) {
    let h = read_header(message, 0).unwrap();
    (h.message_type.to_be_bytes(), h.flags.0, h.reply_id)
}

/// No session before the start and after a reset: a message answers INVALID_SESSION and nothing
/// is sent; between them the host session answers `decs` with its reply, through send_to_guest
#[test]
fn a_session_runs_from_start_to_reset() {
    let sent: Sent = Arc::default();
    let d = device(&sent);
    let decs = request(b"decs", 41);
    assert_eq!(d.deliver(&decs), INVALID_SESSION);
    assert!(sent.lock().is_empty());
    d.start();
    assert_eq!(d.deliver(&decs), 0);
    let replies = std::mem::take(&mut *sent.lock());
    assert_eq!(replies.len(), 1);
    assert_eq!(header_of(&replies[0]), (*b"decs", 4, 41));
    d.reset();
    assert_eq!(d.deliver(&decs), INVALID_SESSION);
    d.start();
    d.start();
    assert_eq!(d.deliver(&request(b"encs", 42)), 0);
    let replies = std::mem::take(&mut *sent.lock());
    assert_eq!(replies.len(), 1);
    assert_eq!(header_of(&replies[0]), (*b"encs", 4, 42));
}

/// A message longer than MAX_MESSAGE_SIZE is answered ALLOCATION_FAILED, session or none, and
/// the session sees nothing of it: no reply, and no hold of the surfaces its `subs` names, so
/// no `rels`. One of MAX_MESSAGE_SIZE bytes is the session's
#[test]
fn a_message_past_the_bound_is_answered_unread() {
    let sent: Sent = Arc::default();
    let d = device(&sent);
    let mut m = subs(&[3]);
    m.extend_from_slice(&request(b"decs", 9));
    m.resize(MAX_MESSAGE_SIZE + 1, 0);
    assert_eq!(d.deliver(&m), ALLOCATION_FAILED);
    d.start();
    assert_eq!(d.deliver(&m), ALLOCATION_FAILED);
    assert!(sent.lock().is_empty());
    m.truncate(MAX_MESSAGE_SIZE);
    assert_ne!(d.deliver(&m), ALLOCATION_FAILED);
    assert!(!sent.lock().is_empty(), "the session's rels, at least");
}

/// After the first phase of a reset the session sends nothing more
#[test]
fn an_invalidated_session_sends_nothing() {
    let sent: Sent = Arc::default();
    let d = device(&sent);
    d.start();
    d.invalidate();
    d.deliver(&request(b"decs", 1));
    assert!(sent.lock().is_empty());
    d.reset();
}

/// With no IOSurface device in the process, a named surface is unreachable; the message it came
/// with is still answered, with none of its surfaces, and the `rels` after it names the failed
/// hold's ID 0, as Apple's host's does (core's `MappedSurfaces::hold`)
#[test]
fn surfaces_without_a_mapper_are_unreachable() {
    let sent: Sent = Arc::default();
    let d = device(&sent);
    d.start();
    assert!(matches!(
        d.surfaces.retain(3),
        Err(SurfaceError::Unreachable(_))
    ));
    let mut m = subs(&[3, 4]);
    m.extend_from_slice(&request(b"decs", 7));
    assert_eq!(d.deliver(&m), 0);
    let sent = std::mem::take(&mut *sent.lock());
    assert_eq!(sent.len(), 2);
    assert_eq!(header_of(&sent[0]), (*b"decs", 4, 7));
    let (released, _) = read_surface_release(&sent[1]).unwrap();
    assert_eq!(released, [MappingId(0)]);
}

/// Without a send_to_guest the session's messages are dropped
#[test]
fn without_send_to_guest_messages_are_dropped() {
    let d = VideoDevice::without_backend(
        Arc::new(FakePlatform::new()),
        VideoDeviceConfig::default(),
        NO_BACKEND,
    );
    d.start();
    assert_eq!(d.deliver(&request(b"decs", 1)), 0);
}

/// Four-character codes print as text where they are, as hex where they are not
#[test]
fn four_character_codes_print() {
    assert_eq!(format!("{:?}", FourCc(*b"420v")), "'420v'");
    assert_eq!(format!("{:?}", FourCc([0, 1, 2, 3])), "0x00010203");
}

/// A backend whose decoders decode nothing, decode while copying as `0` says, and tell the
/// frames of an AV1 temporal unit of two: one decoded hidden, then one shown.
struct DecodesAhead(bool);

impl DecoderBackend for DecodesAhead {
    fn capabilities(&self) -> Vec<CodecCapabilities> {
        Vec::new()
    }

    fn open(&self, _: &DecoderConfig) -> Result<Box<dyn BackendDecoder>, BackendError> {
        Ok(Box::new(DecodesAheadDecoder(self.0)))
    }
}

struct DecodesAheadDecoder(bool);

/// The frames [`DecodesAheadDecoder`] tells it decoded.
fn two_frames() -> Vec<FrameDecoded> {
    let frame = |picture, output| FrameDecoded {
        picture,
        shown_again: false,
        refused: false,
        width: 64,
        height: 64,
        references: [Some(picture); 8],
        primary_reference: None,
        output,
        largest: (64, 64),
    };
    vec![frame(1, false), frame(2, true)]
}

impl BackendDecoder for DecodesAheadDecoder {
    fn output_format(&self) -> PixelFormat {
        PixelFormat::Nv12VideoRange
    }

    fn decode(
        &mut self,
        _: &CompressedFrame<'_>,
    ) -> Result<Vec<Box<dyn DecodedFrame>>, BackendError> {
        Ok(Vec::new())
    }

    fn flush(&mut self) -> Result<Vec<Box<dyn DecodedFrame>>, BackendError> {
        Ok(Vec::new())
    }

    fn can_accept(&self, _: &DecoderConfig) -> bool {
        true
    }

    fn decodes_while_copying(&self) -> bool {
        self.0
    }

    fn frames_decoded(&mut self) -> Vec<FrameDecoded> {
        two_frames()
    }
}

/// A logged decoder of the stand-in backend that decodes while copying as `ahead` says.
fn logged_decoder(ahead: bool) -> Box<dyn BackendDecoder> {
    let logged = logged::LoggedBackend {
        inner: Arc::new(DecodesAhead(ahead)),
        name: "stand-in",
        platform: Arc::new(FakePlatform::new()),
    };
    let config = DecoderConfig {
        codec: Codec::Av1,
        width: 64,
        height: 64,
        configuration: Vec::new(),
    };
    logged.open(&config).unwrap()
}

/// The logged decoder decodes while copying as the backend's own does: the session decodes the
/// next frame during a copy back only where the backend says it may.
#[test]
fn a_logged_decoder_decodes_while_copying_as_the_backends_does() {
    for ahead in [false, true] {
        assert_eq!(logged_decoder(ahead).decodes_while_copying(), ahead);
    }
}

/// The logged decoder tells the frames each decode took as the backend's own does, so that the
/// session lends and holds an AV1 unit's buffers as Apple's AV1 decoder does, not one buffer a
/// frame output.
#[test]
fn a_logged_decoder_tells_the_frames_its_backends_decoded() {
    assert_eq!(logged_decoder(false).frames_decoded(), two_frames());
}

/// A backend whose encoders encode nothing and count their prepares, and whose `+enc` answers
/// -12912, as VideoToolbox's does while every hardware encoder session is taken.
struct CountsPrepares(Arc<std::sync::atomic::AtomicUsize>);

/// What [`CountsPrepares`] answers a `+enc`.
const ENCODERS_TAKEN: i32 = -12912;

impl EncoderBackend for CountsPrepares {
    fn encoders(&self) -> Vec<EncoderCapabilities> {
        Vec::new()
    }

    fn session_capabilities(&self, _: Codec) -> Result<Option<EncoderCapabilities>, i32> {
        Err(ENCODERS_TAKEN)
    }

    fn open(&self, _: &EncoderConfig) -> Result<Box<dyn BackendEncoder>, BackendError> {
        Ok(Box::new(CountsPreparesEncoder(self.0.clone())))
    }
}

struct CountsPreparesEncoder(Arc<std::sync::atomic::AtomicUsize>);

impl BackendEncoder for CountsPreparesEncoder {
    fn configuration(&self) -> Vec<u8> {
        Vec::new()
    }

    fn encode(&mut self, _: &SourceFrame<'_>) -> Result<Vec<EncodedFrame>, BackendError> {
        Ok(Vec::new())
    }

    fn flush(&mut self) -> Result<Vec<EncodedFrame>, BackendError> {
        Ok(Vec::new())
    }

    fn reconfigure(&mut self, _: &EncoderSettings) -> Result<(), BackendError> {
        Ok(())
    }

    fn prepare(&mut self) -> Result<(), BackendError> {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

/// The logged encoder passes the guest's `epfr` on to the backend's, where the VideoToolbox
/// backend prepares its session as Apple's host prepares its encoder.
#[test]
fn a_logged_encoder_prepares_as_the_backends_does() {
    let prepares = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let logged = logged::LoggedEncoders {
        inner: Arc::new(CountsPrepares(prepares.clone())),
        name: "stand-in",
        platform: Arc::new(FakePlatform::new()),
    };
    let config = EncoderConfig {
        codec: Codec::H264,
        width: 64,
        height: 64,
        settings: EncoderSettings::default(),
    };
    let mut encoder = logged.open(&config).unwrap();
    encoder.prepare().unwrap();
    assert_eq!(prepares.load(std::sync::atomic::Ordering::SeqCst), 1);
}

/// The logged encoders answer a `+enc` as the backend's own do: VideoToolbox's -12912 while
/// every hardware encoder session is taken, as Apple's host answers it, not the listed entry.
#[test]
fn logged_encoders_answer_a_session_as_the_backends_do() {
    let logged = logged::LoggedEncoders {
        inner: Arc::new(CountsPrepares(Arc::default())),
        name: "stand-in",
        platform: Arc::new(FakePlatform::new()),
    };
    assert_eq!(
        logged.session_capabilities(Codec::H264),
        Err(ENCODERS_TAKEN)
    );
}
