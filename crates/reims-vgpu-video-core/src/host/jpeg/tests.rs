//! The JPEG engine against hand-built `jcap`, `jdec` and `jenc` messages, a
//! stand-in decoder that records what it is asked, and surfaces of the
//! formats and sizes each test names.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread::ThreadId;
use std::time::Duration;

use crate::backend::Backend;
use crate::fourcc::FourCc;
use crate::host::session_thread::DetachedThreads;
use crate::host::tests::RecordingSink;
use crate::host::{HostSession, names};
use crate::jpeg::{
    JpegCapabilities, JpegDecode, JpegDecoder, JpegDecodes, JpegError, JpegOutputFormat,
    JpegSubsampling, PixelRect,
};
use crate::runtime::{Runtime, StdRuntime};
use crate::status::NO_ERROR;
use crate::surfaces::{
    GuestSurface, MappingId, PlaneDestination, PlaneLayout, SurfaceError, SurfaceLayout, Surfaces,
};
use crate::sync::Mutex;
use crate::wire::WireError;
use crate::wire::message::{GuestUuid, Message, MessageFlags, MessageWriter};
use crate::wire::property_list::{dictionary, integer};

/// What the stand-in decoder was asked, without the data's borrow.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Asked {
    data: Vec<u8>,
    width: u32,
    height: u32,
    subsampling: JpegSubsampling,
    region: PixelRect,
    scale: u32,
    format: JpegOutputFormat,
    plane_rows: Vec<u32>,
}

/// A decoder that records each request, fills its planes with 0x55 but for
/// 0x11 in the first plane's first byte and 0x22 in the last's, and answers
/// what it is told to.
#[derive(Default)]
struct StandIn {
    asked: Mutex<Vec<Asked>>,
    answer: Mutex<Option<JpegError>>,
    /// The guest surfaces lent now, by any thread: the surfaces' count.
    lends: Arc<AtomicUsize>,
    /// The decodes asked while a surface was lent.
    inside_a_lend: AtomicUsize,
    /// The thread each decode ran on.
    threads: Mutex<Vec<ThreadId>>,
    /// When there is one, each decode waits for it to send or close
    /// before it answers: a backend call that does not return until let go.
    gate: Mutex<Option<mpsc::Receiver<()>>>,
    /// Whether a decode panics.
    panics: Mutex<bool>,
}

impl JpegDecoder for StandIn {
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
        JpegCapabilities {
            cores: 1,
            sixteen_bit_quantization_tables: false,
            missing_eoi: false,
            restart_logging: false,
            interchange_formats: false,
        }
    }

    fn decode(
        &self,
        request: &JpegDecode<'_>,
        planes: &mut [PlaneDestination<'_>],
    ) -> Result<(), JpegError> {
        if self.lends.load(Ordering::SeqCst) > 0 {
            self.inside_a_lend.fetch_add(1, Ordering::SeqCst);
        }
        self.threads.lock().push(std::thread::current().id());
        self.asked.lock().push(Asked {
            data: request.data.to_vec(),
            width: request.width,
            height: request.height,
            subsampling: request.subsampling,
            region: request.region,
            scale: request.scale,
            format: request.format,
            plane_rows: planes.iter().map(|plane| plane.rows).collect(),
        });
        let gate = self.gate.lock().take();
        if let Some(gate) = gate {
            let _ = gate.recv();
        }
        assert!(!*self.panics.lock(), "the stand-in's decode panics");
        if let Some(error) = self.answer.lock().clone() {
            return Err(error);
        }
        for plane in planes.iter_mut() {
            plane.bytes.fill(0x55);
        }
        if let Some(first) = planes.first_mut() {
            first.bytes[0] = 0x11;
        }
        if let Some(last) = planes.last_mut() {
            last.bytes[0] = 0x22;
        }
        Ok(())
    }
}

/// Each destination's memory after a write, by surface ID.
type Written = Arc<Mutex<Vec<(u64, Vec<u8>)>>>;

/// The JPEG's bytes, as surface 1 holds them.
const JPEG: &[u8] = b"\xff\xd8 a stand-in stream \xff\xd9";

/// What a destination holds before the engine writes it: what the guest
/// left there.
const LEFT: u8 = 0x77;

/// Surface 1 is the source, holding [`JPEG`]; each other ID is a
/// destination of the format and size the test gave it.
struct TestSurfaces {
    destinations: Vec<(u64, SurfaceLayout)>,
    written: Written,
    /// The IDs the guest made read-only for their clients.
    read_only: Vec<u64>,
    /// The surfaces lent now.
    lends: Arc<AtomicUsize>,
}

struct TestSurface {
    id: u64,
    layout: SurfaceLayout,
    written: Written,
    /// Lent for reading only, as the glue's platforms lend such a surface.
    read_only: bool,
    lends: Arc<AtomicUsize>,
}

/// Counts a lend while it runs.
struct Lent<'a>(&'a AtomicUsize);

impl<'a> Lent<'a> {
    fn new(lends: &'a AtomicUsize) -> Self {
        lends.fetch_add(1, Ordering::SeqCst);
        Self(lends)
    }
}

impl Drop for Lent<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

impl GuestSurface for TestSurface {
    fn layout(&self) -> SurfaceLayout {
        self.layout.clone()
    }

    fn write(&self, write: &mut dyn FnMut(&mut [u8])) -> Result<(), SurfaceError> {
        if self.read_only {
            return Err(SurfaceError::Unreachable(
                "read-only for its clients".into(),
            ));
        }
        let mut memory = vec![LEFT; size(&self.layout)];
        let lent = Lent::new(&self.lends);
        write(&mut memory);
        drop(lent);
        self.written.lock().push((self.id, memory));
        Ok(())
    }

    fn read(&self, read: &mut dyn FnMut(&[u8])) -> Result<(), SurfaceError> {
        // A page of the stream and zeros, as an IOSurface pads it.
        let mut memory = JPEG.to_vec();
        memory.resize(4096, 0);
        let _lent = Lent::new(&self.lends);
        read(&memory);
        Ok(())
    }
}

/// The bytes a layout's planes span.
fn size(layout: &SurfaceLayout) -> usize {
    let height = |rows: u32| usize::try_from(rows).unwrap_or(0);
    match layout.planes.last() {
        Some(plane) => plane.offset + plane.bytes_per_row * height(plane.height),
        None => layout.bytes_per_row * height(layout.height),
    }
}

impl Surfaces for TestSurfaces {
    fn retain(&self, mapping_id: MappingId) -> Result<Box<dyn GuestSurface>, SurfaceError> {
        let layout = if mapping_id.0 == 1 {
            SurfaceLayout {
                pixel_format: FourCc(0),
                width: 4096,
                height: 1,
                planes: Vec::new(),
                bytes_per_row: 4096,
            }
        } else {
            (self.destinations.iter())
                .find(|(id, _)| *id == mapping_id.0)
                .map(|(_, layout)| layout.clone())
                .ok_or(SurfaceError::Unknown(mapping_id.0))?
        };
        Ok(Box::new(TestSurface {
            id: mapping_id.0,
            layout,
            written: Arc::clone(&self.written),
            read_only: self.read_only.contains(&mapping_id.0),
            lends: Arc::clone(&self.lends),
        }))
    }
}

/// A biplanar 4:2:0 surface, `format` (`420f`, say), of `width` by
/// `height`.
fn biplanar(format: &[u8; 4], width: u32, height: u32) -> SurfaceLayout {
    let row = usize::try_from(width.div_ceil(2) * 2).unwrap_or(0);
    let luma_rows = usize::try_from(height).unwrap_or(0);
    SurfaceLayout {
        pixel_format: FourCc::new(*format),
        width,
        height,
        planes: vec![
            PlaneLayout {
                offset: 0,
                bytes_per_row: row,
                width,
                height,
            },
            PlaneLayout {
                offset: row * luma_rows,
                bytes_per_row: row,
                width: width.div_ceil(2),
                height: height.div_ceil(2),
            },
        ],
        bytes_per_row: 0,
    }
}

/// A one-plane surface, `format`, of `width` by `height` at
/// `bytes_per_pixel`.
fn packed(format: &[u8; 4], width: u32, height: u32, bytes_per_pixel: usize) -> SurfaceLayout {
    SurfaceLayout {
        pixel_format: FourCc::new(*format),
        width,
        height,
        planes: Vec::new(),
        bytes_per_row: usize::try_from(width).unwrap_or(0) * bytes_per_pixel,
    }
}

struct Engine {
    sink: Arc<RecordingSink>,
    decoder: Arc<StandIn>,
    written: Written,
    session: HostSession,
}

/// A session whose backend has the stand-in, and whose destinations are
/// surface 2, a 640x480 `420f`, and those given.
fn engine(more: Vec<(u64, SurfaceLayout)>) -> Engine {
    engine_with_read_only(more, Vec::new())
}

/// [`engine`], with the surfaces `read_only` names read-only for their
/// clients.
fn engine_with_read_only(more: Vec<(u64, SurfaceLayout)>, read_only: Vec<u64>) -> Engine {
    engine_with(more, read_only, None)
}

/// [`engine`], with a decode waited `wait` at most and its thread counted
/// in `detached` when left running.
fn engine_with_bound(wait: Duration, detached: &Arc<DetachedThreads>) -> Engine {
    engine_with(Vec::new(), Vec::new(), Some((wait, Arc::clone(detached))))
}

fn engine_with(
    more: Vec<(u64, SurfaceLayout)>,
    read_only: Vec<u64>,
    bound: Option<(Duration, Arc<DetachedThreads>)>,
) -> Engine {
    let sink = Arc::new(RecordingSink::default());
    let lends = Arc::new(AtomicUsize::new(0));
    let decoder = Arc::new(StandIn {
        lends: Arc::clone(&lends),
        ..StandIn::default()
    });
    let written = Arc::new(Mutex::new(Vec::new()));
    let mut destinations = vec![(2, biplanar(b"420f", 640, 480))];
    destinations.extend(more);
    // A backend with the stand-in for its JPEG decoder, and nothing else.
    let backend = Backend {
        jpeg: Some(Arc::clone(&decoder) as Arc<dyn JpegDecoder>),
        ..Backend::none("JPEG alone")
    };
    let session = HostSession::with_backend(
        StdRuntime::shared(),
        sink.clone(),
        Arc::new(TestSurfaces {
            destinations,
            written: Arc::clone(&written),
            read_only,
            lends,
        }),
        &backend,
    );
    let session = match bound {
        Some((wait, detached)) => session.with_teardown(wait, detached),
        None => session,
    };
    Engine {
        sink,
        decoder,
        written,
        session,
    }
}

/// The options CMPhoto's guest side builds for a 640x480 4:2:0 stream at
/// full size.
fn options() -> plist::Dictionary {
    dictionary([
        ("InputSize", integer(i64::try_from(JPEG.len()).unwrap_or(0))),
        ("Width", integer(640)),
        ("Height", integer(480)),
        ("Subsampling", integer(0)),
        ("MCUsHor", integer(40)),
        ("MCUsVer", integer(30)),
        ("Flags", integer(0)),
        ("DecodeWidth", integer(640)),
        ("DecodeHeight", integer(480)),
    ])
}

/// A CGRect's dictionary, as `CGRectCreateDictionaryRepresentation` writes
/// one.
fn rect(x: f64, y: f64, width: f64, height: f64) -> plist::Value {
    plist::Value::Dictionary(dictionary([
        ("X", plist::Value::Real(x)),
        ("Y", plist::Value::Real(y)),
        ("Width", plist::Value::Real(width)),
        ("Height", plist::Value::Real(height)),
    ]))
}

/// A message of `message_type` naming `surfaces`, with `options` as `,joo`
/// when there are some.
fn message(
    message_type: &[u8; 4],
    surfaces: &[u64],
    options: Option<&plist::Dictionary>,
) -> Result<Vec<u8>, WireError> {
    let mut writer = MessageWriter::new(
        FourCc::new(*message_type),
        MessageFlags::GUEST_EXPECTS_REPLY,
        Some(GuestUuid([0x3c; 16])),
        100,
    );
    if let Some(options) = options {
        writer.property_list(
            names::JPEG_OPTIONS,
            Some(&plist::Value::Dictionary(options.clone())),
        )?;
    }
    let body = writer.finish();
    if surfaces.is_empty() {
        return Ok(body);
    }
    let mut bytes = vec![0, 0, 0, 0x30];
    bytes.extend_from_slice(b"subs");
    for slot in 0..5 {
        bytes.extend_from_slice(&surfaces.get(slot).copied().unwrap_or(0).to_be_bytes());
    }
    bytes.extend_from_slice(&body);
    Ok(bytes)
}

impl Engine {
    /// Delivers a `jdec` of surface 1 into `destination` with `options`,
    /// and returns the reply's `,err`.
    fn decode(
        &self,
        destination: u64,
        options: Option<&plist::Dictionary>,
    ) -> Result<Option<i32>, WireError> {
        let delivered = self
            .session
            .deliver(&message(b"jdec", &[1, destination], options)?);
        assert_eq!(delivered, NO_ERROR);
        let sent = self.sink.take();
        let reply = sent.first().ok_or(WireError::Malformed("no reply"))?;
        let reply = Message::parse(reply)?;
        assert_eq!(reply.header().message_type, FourCc::new(*b"jdec"));
        assert_eq!(reply.header().flags, MessageFlags::REPLY_TO_GUEST);
        assert_eq!(reply.header().reply_id, 100);
        reply.s32(names::ERROR)
    }

    fn asked(&self) -> Vec<Asked> {
        std::mem::take(&mut *self.decoder.asked.lock())
    }

    /// What each write lend left in each destination, by surface ID.
    fn written(&self) -> Vec<(u64, Vec<u8>)> {
        std::mem::take(&mut *self.written.lock())
    }
}

const BAD_MEDIA: i32 = 0xe000_02d1_u32.cast_signed();
const BAD_ARGUMENT: i32 = 0xe000_02c2_u32.cast_signed();
const ERROR: i32 = 0xe000_02bc_u32.cast_signed();
const DEVICE_ERROR: i32 = 0xe000_02e9_u32.cast_signed();
const UNSUPPORTED: i32 = 0xe000_02c7_u32.cast_signed();
const TIMEOUT: i32 = 0xe000_02d6_u32.cast_signed();
const BUSY: i32 = 0xe000_02d5_u32.cast_signed();

#[test]
fn capabilities_are_cmphotos_keys_with_the_backends_answers() -> Result<(), WireError> {
    let e = engine(Vec::new());
    assert_eq!(e.session.deliver(&message(b"jcap", &[], None)?), NO_ERROR);
    let sent = e.sink.take();
    assert_eq!(sent.len(), 1);
    let reply = Message::parse(&sent[0])?;
    assert_eq!(reply.s32(names::ERROR)?, Some(0));
    let expected = dictionary([
        ("Version", integer(1)),
        ("NumberOfCores", integer(1)),
        ("SupportsInterchangeFormats", plist::Value::Boolean(false)),
        ("SupportsMissingEOI", plist::Value::Boolean(false)),
        ("SupportsRSTLogging", plist::Value::Boolean(false)),
        (
            "Supports16BitQuantizationTables",
            plist::Value::Boolean(false),
        ),
    ]);
    assert_eq!(reply.dictionary(names::JPEG_CAPABILITIES)?, Some(expected));
    Ok(())
}

#[test]
fn capabilities_with_surfaces_are_a_parameter_error() -> Result<(), WireError> {
    let e = engine(Vec::new());
    e.session.deliver(&message(b"jcap", &[2], None)?);
    let sent = e.sink.take();
    let reply = Message::parse(&sent[0])?;
    assert_eq!(reply.s32(names::ERROR)?, Some(-12902));
    assert_eq!(reply.dictionary(names::JPEG_CAPABILITIES)?, None);
    Ok(())
}

#[test]
fn a_decode_writes_the_whole_image_into_the_destination() -> Result<(), WireError> {
    let e = engine(Vec::new());
    assert_eq!(e.decode(2, Some(&options()))?, Some(0));
    let mut source = JPEG.to_vec();
    source.resize(4096, 0);
    assert_eq!(
        e.asked(),
        [Asked {
            data: source,
            width: 640,
            height: 480,
            subsampling: JpegSubsampling::Yuv420,
            region: PixelRect {
                x: 0,
                y: 0,
                width: 640,
                height: 480
            },
            scale: 1,
            format: JpegOutputFormat::Biplanar420,
            plane_rows: vec![480, 240],
        }]
    );
    // The check before the decode, which writes nothing, and the copy.
    let written = e.written();
    assert_eq!(written.len(), 2);
    assert!(written.iter().all(|(id, _)| *id == 2));
    assert!(written[0].1.iter().all(|&byte| byte == LEFT));
    let memory = &written[1].1;
    assert_eq!((memory[0], memory[640 * 480]), (0x11, 0x22));
    Ok(())
}

/// The output goes into the destination's top-left corner, and the rest
/// keeps what the guest left there, as on Apple's host (a 700x500 `420f`).
#[test]
fn a_decode_writes_its_corner_and_leaves_the_rest() -> Result<(), WireError> {
    let e = engine(vec![(3, biplanar(b"420f", 700, 500))]);
    assert_eq!(e.decode(3, Some(&options()))?, Some(0));
    assert_eq!(e.asked()[0].plane_rows, vec![480, 240], "the output's rows");
    let written = e.written();
    let memory = &written.last().ok_or(WireError::Malformed("no write"))?.1;
    let chroma = 700 * 500;
    for (offset, rows, row, name) in [(0, 500, 480, "luma"), (chroma, 250, 240, "chroma")] {
        for y in 0..rows {
            let line = &memory[offset + y * 700..offset + (y + 1) * 700];
            let (inside, outside) = line.split_at(if y < row { 640 } else { 0 });
            let first = if y == 0 { 1 } else { 0 };
            assert!(
                inside[first..].iter().all(|&byte| byte == 0x55),
                "{name} row {y} inside"
            );
            assert!(
                outside.iter().all(|&byte| byte == LEFT),
                "{name} row {y} outside"
            );
        }
    }
    assert_eq!((memory[0], memory[chroma]), (0x11, 0x22));
    Ok(())
}

/// The backend decodes with no guest surface lent, on a thread of its own,
/// so that a call that never returns holds no lend.
#[test]
fn no_decode_runs_inside_a_lend() -> Result<(), WireError> {
    let e = engine(vec![
        (3, packed(b"BGRA", 640, 480, 4)),
        (4, packed(b"2vuy", 640, 480, 2)),
    ]);
    for id in [2, 3, 4] {
        assert_eq!(e.decode(id, Some(&options()))?, Some(0));
    }
    assert_eq!(e.asked().len(), 3);
    assert_eq!(e.decoder.inside_a_lend.load(Ordering::SeqCst), 0);
    let threads = e.decoder.threads.lock().clone();
    assert!(
        threads
            .iter()
            .all(|thread| *thread != std::thread::current().id()),
        "decoded on the delivering thread"
    );
    Ok(())
}

/// A decode that does not return in the bound is answered kIOReturnTimeout,
/// with nothing written; its thread is left running, counted, and holds no
/// surface. While the process's detached threads are full, a decode is
/// kIOReturnBusy with the backend not asked. Once the call returns, the
/// thread ends and decodes go on.
#[test]
fn a_decode_that_does_not_return_is_answered_at_the_bound() -> Result<(), WireError> {
    const BOUND: Duration = Duration::from_millis(200);
    let detached = Arc::new(DetachedThreads::new(1));
    let e = engine_with_bound(BOUND, &detached);
    let (release, gate) = mpsc::channel();
    *e.decoder.gate.lock() = Some(gate);
    let started = StdRuntime.now();
    let delivered = e
        .session
        .deliver(&message(b"jdec", &[1, 2], Some(&options()))?);
    let waited = StdRuntime.elapsed_since(started);
    assert_eq!(delivered, NO_ERROR);
    assert!(waited >= BOUND && waited < BOUND * 10, "waited {waited:?}");
    // The reply, then the release of both surfaces: the thread left running
    // holds neither.
    let sent = e.sink.take();
    assert_eq!(sent.len(), 2);
    assert_eq!(Message::parse(&sent[0])?.s32(names::ERROR)?, Some(TIMEOUT));
    assert_eq!(
        crate::wire::surface_boxes::read_surface_release(&sent[1]),
        Some((vec![MappingId(1), MappingId(2)], &[][..]))
    );
    assert_eq!(e.asked().len(), 1);
    assert_eq!(detached.count(), 1);
    // The check before the decode, and nothing after.
    let written = e.written();
    assert_eq!(written.len(), 1);
    assert!(written[0].1.iter().all(|&byte| byte == LEFT));
    assert_eq!(e.decode(2, Some(&options()))?, Some(BUSY));
    assert!(e.asked().is_empty());
    assert!(e.written().is_empty());
    drop(release);
    let started = StdRuntime.now();
    while detached.count() > 0 {
        assert!(
            StdRuntime.elapsed_since(started) < Duration::from_secs(5),
            "never ended"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(e.decode(2, Some(&options()))?, Some(0));
    Ok(())
}

/// A backend call that panics is a malfunction, kIOReturnDeviceError, and
/// the engine decodes on.
#[test]
fn a_decode_that_panics_is_a_device_error() -> Result<(), WireError> {
    let e = engine(Vec::new());
    *e.decoder.panics.lock() = true;
    assert_eq!(e.decode(2, Some(&options()))?, Some(DEVICE_ERROR));
    *e.decoder.panics.lock() = false;
    assert_eq!(e.decode(2, Some(&options()))?, Some(0));
    Ok(())
}

/// A source the guest made read-only for its clients is read; a destination
/// is never written: CMPhoto's bad options, -16990, before the backend is
/// asked. Apple's host hands such a destination to its hardware.
#[test]
fn a_read_only_destination_is_never_written() -> Result<(), WireError> {
    let e = engine_with_read_only(Vec::new(), vec![1]);
    assert_eq!(
        e.decode(2, Some(&options()))?,
        Some(0),
        "a read-only source"
    );
    assert_eq!(e.asked().len(), 1);
    let e = engine_with_read_only(Vec::new(), vec![2]);
    assert_eq!(e.decode(2, Some(&options()))?, Some(-16990));
    assert!(e.asked().is_empty());
    assert!(e.written.lock().is_empty());
    Ok(())
}

#[test]
fn each_required_option_absent_is_cmphotos_bad_options() -> Result<(), WireError> {
    let e = engine(Vec::new());
    for key in [
        "Width",
        "Height",
        "Subsampling",
        "MCUsHor",
        "MCUsVer",
        "Flags",
        "DecodeWidth",
        "DecodeHeight",
    ] {
        let mut options = options();
        options.remove(key);
        assert_eq!(e.decode(2, Some(&options))?, Some(-16990), "{key}");
        options.insert(key.to_owned(), plist::Value::String("6".into()));
        assert_eq!(
            e.decode(2, Some(&options))?,
            Some(-16990),
            "{key} as a string"
        );
    }
    assert_eq!(e.decode(2, None)?, Some(-16990), "no options");
    let mut options = options();
    options.remove("InputSize");
    assert_eq!(e.decode(2, Some(&options))?, Some(0), "InputSize absent");
    options.insert("CropRect".to_owned(), integer(3));
    assert_eq!(
        e.decode(2, Some(&options))?,
        Some(-16990),
        "CropRect a number"
    );
    assert_eq!(e.asked().len(), 1, "only the decode without InputSize");
    Ok(())
}

#[test]
fn a_destination_format_the_engine_has_no_layout_for_is_kioreturnerror() -> Result<(), WireError> {
    let e = engine(vec![
        (3, packed(b"L008", 640, 480, 1)),
        // `420f` in one plane: not a layout of the format.
        (4, packed(b"420f", 640, 480, 1)),
    ]);
    assert_eq!(e.decode(3, Some(&options()))?, Some(ERROR));
    assert_eq!(e.decode(4, Some(&options()))?, Some(ERROR));
    assert!(e.asked().is_empty());
    Ok(())
}

#[test]
fn a_destination_smaller_than_the_decode_is_a_bad_argument() -> Result<(), WireError> {
    let e = engine(vec![(3, biplanar(b"420v", 320, 240))]);
    assert_eq!(e.decode(3, Some(&options()))?, Some(BAD_ARGUMENT));
    assert!(e.asked().is_empty());
    // At half size it fits.
    let mut half = options();
    half.insert("DecodeWidth".to_owned(), integer(320));
    half.insert("DecodeHeight".to_owned(), integer(240));
    assert_eq!(e.decode(3, Some(&half))?, Some(0));
    assert_eq!(e.asked()[0].scale, 2);
    Ok(())
}

#[test]
fn a_destination_whose_planes_do_not_hold_the_decode_is_a_bad_argument() -> Result<(), WireError> {
    // 640x480 by the layout, with planes of one byte, or rows short of
    // the picture.
    let mut tiny = biplanar(b"420f", 640, 480);
    for plane in &mut tiny.planes {
        plane.bytes_per_row = 1;
        plane.height = 1;
    }
    let mut narrow = biplanar(b"420f", 640, 480);
    narrow.planes[1].bytes_per_row = 320;
    let short = packed(b"BGRA", 640, 480, 2);
    let e = engine(vec![(3, tiny), (4, narrow), (5, short)]);
    for destination in [3, 4, 5] {
        assert_eq!(
            e.decode(destination, Some(&options()))?,
            Some(BAD_ARGUMENT),
            "{destination}"
        );
    }
    assert!(e.asked().is_empty());
    Ok(())
}

#[test]
fn the_scale_is_the_hardwares_choice_from_the_decode_size() -> Result<(), WireError> {
    // As measured on Apple's host.
    let e = engine(Vec::new());
    for ((width, height), scale) in [
        ((640, 480), Some(1)),
        ((320, 240), Some(2)),
        ((500, 375), Some(2)),
        ((320, 480), Some(1)),
        ((160, 120), Some(4)),
        ((80, 60), Some(8)),
        ((40, 30), None),
        ((1000, 1000), Some(1)),
    ] {
        let mut options = options();
        options.insert("DecodeWidth".to_owned(), integer(width));
        options.insert("DecodeHeight".to_owned(), integer(height));
        let status = e.decode(2, Some(&options))?;
        let asked = e.asked();
        match scale {
            Some(scale) => {
                assert_eq!(status, Some(0), "{width}x{height}");
                assert_eq!(asked[0].scale, scale, "{width}x{height}");
            }
            None => {
                assert_eq!(status, Some(BAD_ARGUMENT), "{width}x{height}");
                assert!(asked.is_empty());
            }
        }
    }
    Ok(())
}

#[test]
fn an_odd_image_scales_as_measured() -> Result<(), WireError> {
    let e = engine(Vec::new());
    for (width, height) in [(167, 106), (166, 105)] {
        let mut options = options();
        for (key, value) in [
            ("Width", 333),
            ("Height", 211),
            ("MCUsHor", 21),
            ("MCUsVer", 14),
            ("DecodeWidth", width),
            ("DecodeHeight", height),
        ] {
            options.insert(key.to_owned(), integer(value));
        }
        assert_eq!(e.decode(2, Some(&options))?, Some(0));
        let asked = e.asked();
        assert_eq!(asked[0].scale, 2, "{width}x{height}");
    }
    Ok(())
}

#[test]
fn a_crop_is_whole_mcus_cut_at_the_images_edge() -> Result<(), WireError> {
    let e = engine(Vec::new());
    let region = |crop: plist::Value| -> Result<(Option<i32>, Option<PixelRect>), WireError> {
        let mut options = options();
        options.insert("CropRect".to_owned(), crop);
        let status = e.decode(2, Some(&options))?;
        Ok((status, e.asked().first().map(|asked| asked.region)))
    };
    let at = |x, y, width, height| PixelRect {
        x,
        y,
        width,
        height,
    };
    // Measured: {0, 0, 20, 15} decodes the top-left 320x240.
    assert_eq!(
        region(rect(0.0, 0.0, 20.0, 15.0))?,
        (Some(0), Some(at(0, 0, 320, 240)))
    );
    assert_eq!(
        region(rect(36.0, 29.0, 4.0, 1.0))?,
        (Some(0), Some(at(576, 464, 64, 16)))
    );
    // Every MCU is no crop.
    assert_eq!(
        region(rect(0.0, 0.0, 40.0, 30.0))?,
        (Some(0), Some(at(0, 0, 640, 480)))
    );
    // Values round half away from zero.
    assert_eq!(
        region(rect(0.4, 0.5, 3.5, 1.0))?,
        (Some(0), Some(at(0, 16, 64, 16)))
    );
    // Measured: a crop in pixels runs past the MCUs.
    assert_eq!(
        region(rect(0.0, 0.0, 320.0, 240.0))?,
        (Some(BAD_ARGUMENT), None)
    );
    assert_eq!(
        region(rect(0.0, 0.0, 0.0, 15.0))?,
        (Some(BAD_ARGUMENT), None)
    );
    Ok(())
}

#[test]
fn a_crop_of_an_odd_image_stops_at_its_edge() -> Result<(), WireError> {
    let e = engine(Vec::new());
    let mut options = options();
    for (key, value) in [
        ("Width", 333),
        ("Height", 211),
        ("MCUsHor", 21),
        ("MCUsVer", 14),
    ] {
        options.insert(key.to_owned(), integer(value));
    }
    options.insert("CropRect".to_owned(), rect(17.0, 12.0, 4.0, 2.0));
    assert_eq!(e.decode(2, Some(&options))?, Some(0));
    assert_eq!(
        e.asked()[0].region,
        PixelRect {
            x: 272,
            y: 192,
            width: 61,
            height: 19
        }
    );
    Ok(())
}

#[test]
fn a_decode_of_any_area_reaches_the_backend() -> Result<(), WireError> {
    // Apple's hardware refuses a decode under 24 blocks of area, each
    // false below as measured on Apple's host, with kIOReturnBadArgument;
    // the backend's decoder takes any, so each reaches it. Surface 3 holds
    // the tallest and the widest.
    let e = engine(vec![(3, biplanar(b"420f", 4096, 4096))]);
    let size = |subsampling: JpegSubsampling, width: u32, height: u32| {
        let (mcu_width, mcu_height) = subsampling.mcu_size();
        let mut options = options();
        for (key, value) in [
            ("Subsampling", subsampling.code()),
            ("Width", width.into()),
            ("Height", height.into()),
            ("MCUsHor", width.div_ceil(mcu_width).into()),
            ("MCUsVer", height.div_ceil(mcu_height).into()),
            ("DecodeWidth", width.into()),
            ("DecodeHeight", height.into()),
        ] {
            options.insert(key.to_owned(), integer(value));
        }
        options
    };
    for (subsampling, width, height, apple_decodes) in [
        (JpegSubsampling::Yuv420, 16, 16, false),
        (JpegSubsampling::Yuv420, 48, 16, false),
        (JpegSubsampling::Yuv420, 17, 9, false),
        (JpegSubsampling::Yuv420, 64, 16, true),
        (JpegSubsampling::Yuv420, 4095, 7, true),
        (JpegSubsampling::Yuv422, 80, 8, false),
        (JpegSubsampling::Yuv422, 96, 8, true),
        (JpegSubsampling::Yuv444, 15, 15, false),
        (JpegSubsampling::Yuv444, 56, 8, false),
        (JpegSubsampling::Yuv444, 64, 8, true),
        (JpegSubsampling::Grey, 1, 1, false),
        (JpegSubsampling::Grey, 184, 8, false),
        (JpegSubsampling::Grey, 192, 8, true),
        (JpegSubsampling::Grey, 7, 4095, true),
        (JpegSubsampling::Yuv411, 96, 8, false),
        (JpegSubsampling::Yuv411, 128, 8, true),
    ] {
        let status = e.decode(3, Some(&size(subsampling, width, height)))?;
        let asked = e.asked();
        let case = format!("{width}x{height} {subsampling:?}, Apple's decodes: {apple_decodes}");
        assert_eq!(status, Some(0), "{case}");
        assert_eq!(asked.len(), 1, "{case}");
    }
    // A crop asks for its own region, an MCU the image's edge cuts for the
    // pixels it has.
    let odd_420 = size(JpegSubsampling::Yuv420, 333, 211);
    let odd_422 = size(JpegSubsampling::Yuv422, 65, 33);
    for (image, (x, y, across, down), (width, height), apple_decodes) in [
        (&odd_420, (0, 0, 3, 1), (48, 16), false),
        (&odd_420, (0, 0, 4, 1), (64, 16), true),
        (&odd_420, (17, 0, 4, 1), (61, 16), false),
        (&odd_420, (17, 12, 4, 2), (61, 19), true),
        (&odd_420, (0, 13, 5, 1), (80, 3), false),
        (&odd_420, (20, 9, 1, 5), (13, 67), false),
        (&odd_420, (20, 8, 1, 5), (13, 80), true),
        (&odd_420, (19, 11, 2, 3), (29, 35), false),
        (&odd_420, (18, 12, 3, 2), (45, 19), false),
        (&odd_422, (2, 0, 3, 2), (33, 16), false),
        (&odd_422, (2, 0, 3, 3), (33, 24), true),
        (&odd_422, (3, 0, 2, 5), (17, 33), false),
    ] {
        let mut options = image.clone();
        options.insert(
            "CropRect".to_owned(),
            rect(x.into(), y.into(), across.into(), down.into()),
        );
        let status = e.decode(3, Some(&options))?;
        let asked = e.asked();
        let case = format!("{width}x{height} at MCU ({x}, {y}), Apple's decodes: {apple_decodes}");
        assert_eq!(status, Some(0), "{case}");
        assert_eq!(
            (asked[0].region.width, asked[0].region.height),
            (width, height),
            "{case}"
        );
    }
    Ok(())
}

/// Options of the largest sides a `jdec` takes, whose area times an MCU's
/// blocks would pass 64 bits in a plain area rule, are refused as too
/// large to fit and never reach the decoder.
#[test]
fn an_image_of_no_width_or_height_is_a_bad_argument() -> Result<(), WireError> {
    // Nothing to decode, which Apple's hardware refuses as under its least
    // area; the backend is never asked (the host session's fuzzer, seed
    // 0x1b63d0927cbaaf89, input 171: a 41x0 image).
    let e = engine(vec![(3, biplanar(b"420f", 64, 64))]);
    for (width, height) in [(41, 0), (0, 9), (0, 0)] {
        let mut options = options();
        for (key, value) in [
            ("Width", width),
            ("Height", height),
            ("MCUsHor", 3),
            ("MCUsVer", 1),
            ("DecodeWidth", width),
            ("DecodeHeight", height),
        ] {
            options.insert(key.to_owned(), integer(value));
        }
        let case = format!("{width}x{height}");
        assert_eq!(e.decode(3, Some(&options))?, Some(BAD_ARGUMENT), "{case}");
        assert!(e.asked().is_empty(), "{case}");
    }
    Ok(())
}

#[test]
fn options_of_the_largest_sides_are_refused_without_overflow() -> Result<(), WireError> {
    let e = engine(vec![(3, biplanar(b"420f", 4096, 4096))]);
    let side = i64::from(u32::MAX);
    for subsampling in [JpegSubsampling::Yuv420, JpegSubsampling::Yuv444] {
        let mut options = options();
        for (key, value) in [
            ("Subsampling", subsampling.code()),
            ("Width", side),
            ("Height", side),
            ("MCUsHor", side),
            ("MCUsVer", side),
        ] {
            options.insert(key.to_owned(), integer(value));
        }
        for decode in [640, side] {
            options.insert("DecodeWidth".to_owned(), integer(decode));
            options.insert("DecodeHeight".to_owned(), integer(decode));
            let case = format!("{subsampling:?} decoded at {decode}");
            assert_eq!(e.decode(3, Some(&options))?, Some(BAD_ARGUMENT), "{case}");
            assert!(e.asked().is_empty(), "{case}");
        }
    }
    Ok(())
}

#[test]
fn backend_errors_are_the_hardwares_statuses() -> Result<(), WireError> {
    let e = engine(Vec::new());
    for (error, status) in [
        (JpegError::BadData("cut".into()), BAD_MEDIA),
        (
            JpegError::UnsupportedStream("progressive".into()),
            BAD_MEDIA,
        ),
        (
            JpegError::SizeMismatch {
                width: 320,
                height: 480,
            },
            BAD_MEDIA,
        ),
        (JpegError::SubsamplingMismatch, DEVICE_ERROR),
        (JpegError::UnsupportedFormat("BGRA".into()), ERROR),
        (JpegError::UnsupportedScale(2), UNSUPPORTED),
        (JpegError::Malfunction("ring".into()), DEVICE_ERROR),
    ] {
        *e.decoder.answer.lock() = Some(error.clone());
        assert_eq!(e.decode(2, Some(&options()))?, Some(status), "{error}");
    }
    Ok(())
}

#[test]
fn a_subsampling_cmphoto_never_sends_is_a_device_error() -> Result<(), WireError> {
    let e = engine(Vec::new());
    let mut options = options();
    options.insert("Subsampling".to_owned(), integer(9));
    assert_eq!(e.decode(2, Some(&options))?, Some(DEVICE_ERROR));
    assert!(e.asked().is_empty());
    Ok(())
}

#[test]
fn packed_and_other_biplanar_layouts_reach_the_backend() -> Result<(), WireError> {
    let e = engine(vec![
        (3, packed(b"2vuy", 640, 480, 2)),
        (4, packed(b"yuvs", 640, 480, 2)),
        (5, packed(b"BGRA", 640, 480, 4)),
    ]);
    for (id, format) in [
        (3, JpegOutputFormat::Uyvy),
        (4, JpegOutputFormat::Yuyv),
        (5, JpegOutputFormat::Bgra),
    ] {
        assert_eq!(e.decode(id, Some(&options()))?, Some(0));
        let asked = e.asked();
        assert_eq!(asked[0].format, format);
        assert_eq!(asked[0].plane_rows, vec![480]);
    }
    Ok(())
}

#[test]
fn an_encode_is_answered_as_by_a_host_without_jpeg_hardware() -> Result<(), WireError> {
    let e = engine(Vec::new());
    e.session
        .deliver(&message(b"jenc", &[2, 1], Some(&options()))?);
    let sent = e.sink.take();
    let reply = Message::parse(&sent[0])?;
    assert_eq!(reply.s32(names::ERROR)?, Some(-16994));
    assert_eq!(reply.u32(FourCc::new(*b",jes"))?, None);
    e.session
        .deliver(&message(b"jenc", &[2], Some(&options()))?);
    let sent = e.sink.take();
    assert_eq!(Message::parse(&sent[0])?.s32(names::ERROR)?, Some(-12902));
    assert!(e.asked().is_empty());
    Ok(())
}

#[test]
fn a_decode_lets_its_surfaces_go_after_its_reply() -> Result<(), WireError> {
    let e = engine(Vec::new());
    e.session
        .deliver(&message(b"jdec", &[1, 2], Some(&options()))?);
    let sent = e.sink.take();
    assert_eq!(sent.len(), 2, "the reply, then the release");
    assert_eq!(
        crate::wire::surface_boxes::read_surface_release(&sent[1]),
        Some((vec![MappingId(1), MappingId(2)], &[][..]))
    );
    Ok(())
}
