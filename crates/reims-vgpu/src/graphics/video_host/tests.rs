use std::collections::VecDeque;
use std::time::Duration;

use reims_graphics::wire::RGFX_VIDEO_CODEC_H264;
use reims_vgpu_core::testkit::FakePlatform;
use reims_vgpu_core::testkit::calllog::Arg;
use reims_vgpu_core::testkit::objects::CompleteOn;

use super::*;

type P = FakePlatform;
type FakeBuffer = <P as Platform>::Buffer;

/// `MTLResourceStorageModePrivate`: a device cache's storage, with no host bytes.
const PRIVATE: u64 = 2 << 4;
/// What the output's bytes hold before a picture lands.
const UNTOUCHED: u8 = 0xee;
const COPY: &str = "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:";

fn buffer(p: &FakePlatform, len: u64, options: u64) -> FakeBuffer {
    let b = (p
        .device()
        .new_buffer_of_length(OnFault::Return, len, options))
    .unwrap()
    .unwrap();
    if options == STORAGE_SHARED {
        // SAFETY: the fake's Shared buffer holds `len` host bytes.
        unsafe { std::ptr::write_bytes(b.contents().as_ptr::<u8>(), UNTOUCHED, len as usize) };
    }
    b
}

/// The bytes of `b`, a Shared buffer of the fake's.
fn bytes(b: &FakeBuffer) -> Vec<u8> {
    // SAFETY: the fake's Shared buffer holds `length` host bytes.
    unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u8>(), b.length() as usize) }.to_vec()
}

/// What the decoder and its pictures saw, in order.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Seen {
    Sets(Vec<u8>),
    /// An access unit: its host address and length.
    Unit(usize, usize),
    /// A picture's landing: whether each plane's host bytes were staged.
    Landed {
        luma: bool,
        chroma: bool,
    },
}

type Log = Arc<Mutex<Vec<Seen>>>;

/// How a picture of the test's lands.
enum Picture {
    /// Each plane's crop rows of `fill`, written through `Landing::host`.
    Fill(u8, Log),
    /// The same by `Landing::copy_rows`.
    Cpu(u8, Log),
    /// As `Fill`, once `go` is sent.
    Gated(u8, Log, mpsc::Receiver<()>),
    /// A failure to land.
    Fail,
}

impl HostPicture<P> for Picture {
    fn land(self: Box<Self>, to: &mut Landing<'_, P>) -> Result<(), BackendError> {
        let fill = |to: &mut Landing<'_, P>, fill: u8, log: &Log| {
            let planes = to.host()?;
            for plane in [&planes.luma, &planes.chroma] {
                for row in 0..plane.rows {
                    let at = plane.at.wrapping_add(row * plane.bytes_per_row);
                    // SAFETY: the plane's rows lie in its host bytes (`Landing::host`'s word).
                    unsafe {
                        std::ptr::write_bytes(at.as_ptr::<u8>(), fill, plane.row_bytes as usize);
                    }
                }
            }
            let landed = Seen::Landed {
                luma: planes.luma.staged,
                chroma: planes.chroma.staged,
            };
            log.lock().push(landed);
            Ok::<(), BackendError>(())
        };
        match *self {
            Picture::Fill(n, log) => fill(to, n, &log),
            Picture::Gated(n, log, go) => {
                go.recv().unwrap();
                fill(to, n, &log)
            }
            Picture::Cpu(n, log) => {
                let rows = vec![n; 4096 * 128];
                let source = |p: &OutputPlane<P>| SourcePlane {
                    at: rows.as_ptr(),
                    bytes_per_row: 4096,
                    row_bytes: p.row_bytes as usize,
                    rows: p.height as usize,
                };
                let (luma, chroma) = (source(to.luma), source(to.chroma));
                // SAFETY: `rows` holds 128 rows of 4096 bytes, apart from the output.
                unsafe { to.copy_rows(luma, chroma) }?;
                log.lock().push(Seen::Landed {
                    luma: to.luma.cached,
                    chroma: to.chroma.cached,
                });
                Ok(())
            }
            Picture::Fail => Err("the hardware failed".into()),
        }
    }
}

/// A decoder that logs what it is handed and answers `answers` in turn: a picture of
/// `Fill(n)` for decode `n` once they run out.
struct Decoder {
    log: Log,
    answers: VecDeque<Result<Option<Picture>, &'static str>>,
    decoded: u8,
}

impl HostDecoder<P> for Decoder {
    fn parameter_sets(&mut self, sets: &[u8]) -> Result<(), BackendError> {
        self.log.lock().push(Seen::Sets(sets.to_vec()));
        Ok(())
    }

    fn decode(&mut self, au: &[u8]) -> Result<Option<Box<dyn HostPicture<P>>>, BackendError> {
        (self.log.lock()).push(Seen::Unit(au.as_ptr().addr(), au.len()));
        self.decoded += 1;
        let answer = (self.answers.pop_front())
            .unwrap_or_else(|| Ok(Some(Picture::Fill(self.decoded, Arc::clone(&self.log)))));
        let picture = answer?;
        Ok(picture.map(|p| Box::new(p) as Box<dyn HostPicture<P>>))
    }
}

/// A session over a decoder answering `answers`, and the log it keeps.
fn session(
    p: &FakePlatform,
    answers: Vec<Result<Option<Picture>, &'static str>>,
    log: &Log,
) -> HostVideoSession<P> {
    let decoder = Decoder {
        log: Arc::clone(log),
        answers: answers.into(),
        decoded: 0,
    };
    HostVideoSession::new(
        &p.device(),
        RGFX_VIDEO_CODEC_H264,
        "test",
        Box::new(decoder),
    )
    .unwrap()
}

/// The planes of an output in `luma` and `chroma`: a 96x64 luma plane of 96-byte rows, 128
/// apart, at 0, and its 48x32 chroma plane at 8192 (each at 4096 rows' pitch where `pitch`
/// says so).
fn planes<'a>(
    luma: &'a FakeBuffer,
    chroma: &'a FakeBuffer,
    row_bytes: u64,
) -> (VideoPlane<'a, P>, VideoPlane<'a, P>) {
    let plane = |buffer: &'a FakeBuffer, offset, width, height| VideoPlane {
        buffer,
        offset,
        bytes_per_row: 128,
        row_bytes,
        width,
        height,
        cached: buffer.contents().is_null(),
        pages: None,
    };
    (plane(luma, 0, 96, 64), plane(chroma, 8192, 48, 32))
}

fn decode<'a>(
    bitstream: &'a FakeBuffer,
    sets: &'a [u8],
    (luma, chroma): (VideoPlane<'a, P>, VideoPlane<'a, P>),
) -> VideoDecode<'a, P> {
    VideoDecode {
        bitstream,
        offset: 256,
        size: 100,
        parameter_sets: sets,
        luma,
        chroma,
        output_rows: 0,
    }
}

fn event_of(s: &HostVideoSession<P>) -> &<P as Platform>::SharedEvent {
    match s.event() {
        EventRef::Shared(event) => event,
        EventRef::Event(_) => panic!("a plain event"),
    }
}

/// Whether `b`'s rows of a plane at `offset`, `rows` of them 128 apart, hold `fill` in their
/// first `row_bytes` and nothing else.
fn holds(b: &[u8], offset: usize, rows: usize, row_bytes: usize, fill: u8) -> bool {
    (0..rows).all(|row| {
        let at = offset + row * 128;
        b[at..at + row_bytes].iter().all(|&x| x == fill)
            && b[at + row_bytes..at + 128].iter().all(|&x| x == UNTOUCHED)
    })
}

/// A decode hands the decoder its parameter sets, then its access unit in place in the
/// bitstream's buffer; one with no parameter sets hands the access unit alone. Each answers the
/// next value, signalled once its picture landed.
#[test]
fn a_decode_hands_its_parameter_sets_then_its_access_unit_in_place() {
    let p = FakePlatform::new();
    let log = Log::default();
    let s = session(&p, vec![], &log);
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let pages = buffer(&p, 16384, STORAGE_SHARED);
    let sets = [0, 0, 0, 1, 0x67, 0x42];
    let at = bitstream.contents().addr() + 256;
    assert_eq!(
        s.decode(&decode(&bitstream, &sets, planes(&pages, &pages, 96)))
            .unwrap(),
        1
    );
    assert_eq!(
        s.decode(&decode(&bitstream, &[], planes(&pages, &pages, 96)))
            .unwrap(),
        2
    );
    assert!(event_of(&s).wait_until_signaled_value(2, 10_000));
    // The landings run on the landing thread, beside the decodes: the decoder's calls are in
    // order among themselves.
    let (landed, handed): (Vec<Seen>, Vec<Seen>) =
        (log.lock().iter().cloned()).partition(|seen| matches!(seen, Seen::Landed { .. }));
    assert_eq!(
        handed,
        [
            Seen::Sets(sets.to_vec()),
            Seen::Unit(at, 100),
            Seen::Unit(at, 100)
        ]
    );
    let in_pages = Seen::Landed {
        luma: false,
        chroma: false,
    };
    assert_eq!(landed, [in_pages.clone(), in_pages]);
}

/// A picture lands in the surface's pages, its crop's rows alone, the bytes past each row and
/// past its rows as they were: by the hardware through the host bytes, and by a CPU copy.
#[test]
fn a_picture_lands_in_the_pages_its_rows_alone() {
    let p = FakePlatform::new();
    let log = Log::default();
    let answers = vec![Ok(Some(Picture::Cpu(9, Arc::clone(&log))))];
    let s = session(&p, answers, &log);
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let (first, second) = (
        buffer(&p, 16384, STORAGE_SHARED),
        buffer(&p, 16384, STORAGE_SHARED),
    );
    assert_eq!(
        s.decode(&decode(&bitstream, &[], planes(&first, &first, 96)))
            .unwrap(),
        1
    );
    assert_eq!(
        s.decode(&decode(&bitstream, &[], planes(&second, &second, 96)))
            .unwrap(),
        2
    );
    drop(s);
    for (b, fill) in [(bytes(&first), 9), (bytes(&second), 2)] {
        assert!(holds(&b, 0, 64, 96, fill));
        assert!(holds(&b, 8192, 32, 96, fill));
        assert!(b[64 * 128..8192].iter().all(|&x| x == UNTOUCHED));
        assert!(b[8192 + 32 * 128..].iter().all(|&x| x == UNTOUCHED));
    }
}

/// A plane in a device cache lands through the staging buffer and a copy into the cache on the
/// session's queue, before the decode's signal: one copy of its rows where they fill their
/// pitch, else one a row, leaving another texture's bytes between them; a plane in pages beside
/// it lands in place.
#[test]
fn a_plane_in_a_device_cache_lands_staged_then_copied() {
    let p = FakePlatform::new();
    p.world().set_auto_complete(Some(CompleteOn::Caller));
    let log = Log::default();
    let s = session(&p, vec![], &log);
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let cache = buffer(&p, 16384, PRIVATE);
    assert_eq!(
        s.decode(&decode(&bitstream, &[], planes(&cache, &cache, 128)))
            .unwrap(),
        1
    );
    assert!(event_of(&s).wait_until_signaled_value(1, 10_000));
    let copies = p.call_log().calls_to(COPY);
    assert_eq!(copies.len(), 2);
    let staging = copies[0].args[0].clone();
    let at = |args: &[Arg]| {
        (
            args[1].clone(),
            args[2].clone(),
            args[3].clone(),
            args[4].clone(),
        )
    };
    let to = Arg::obj(&cache);
    assert_eq!(
        at(&copies[0].args),
        (Arg::U64(0), to.clone(), Arg::U64(0), Arg::U64(8192))
    );
    assert_eq!(
        at(&copies[1].args),
        (Arg::U64(8192), to.clone(), Arg::U64(8192), Arg::U64(4096))
    );
    assert_eq!(copies[1].args[0], staging);

    // Rows narrower than their pitch: one copy a row. The luma plane in pages lands in place.
    let pages = buffer(&p, 16384, STORAGE_SHARED);
    assert_eq!(
        s.decode(&decode(&bitstream, &[], planes(&pages, &cache, 96)))
            .unwrap(),
        2
    );
    drop(s);
    let copies = &p.call_log().calls_to(COPY)[2..];
    assert_eq!(copies.len(), 32);
    assert_eq!(
        at(&copies[0].args),
        (Arg::U64(0), to.clone(), Arg::U64(8192), Arg::U64(96))
    );
    let last = (
        Arg::U64(31 * 128),
        to,
        Arg::U64(8192 + 31 * 128),
        Arg::U64(96),
    );
    assert_eq!(at(&copies[31].args), last);
    let b = bytes(&pages);
    assert!(holds(&b, 0, 64, 96, 2));
    assert!(b[64 * 128..].iter().all(|&x| x == UNTOUCHED));
    let landed = (log.lock().iter())
        .filter(|s| matches!(s, Seen::Landed { .. }))
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        landed,
        [
            Seen::Landed {
                luma: true,
                chroma: true
            },
            Seen::Landed {
                luma: false,
                chroma: true
            }
        ]
    );
}

/// Values are signalled in decode order, each once its picture landed: a decode that answers no
/// picture is signalled in its turn, behind a picture still landing.
#[test]
fn values_are_signalled_in_decode_order_once_landed() {
    let p = FakePlatform::new();
    let log = Log::default();
    let (go, gate) = mpsc::channel();
    let answers = vec![
        Ok(Some(Picture::Gated(1, Arc::clone(&log), gate))),
        Ok(None),
    ];
    let s = session(&p, answers, &log);
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let pages = buffer(&p, 16384, STORAGE_SHARED);
    for value in 1..=3 {
        let d = decode(&bitstream, &[], planes(&pages, &pages, 96));
        assert_eq!(s.decode(&d).unwrap(), value);
    }
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(event_of(&s).signaled_value(), 0);
    go.send(()).unwrap();
    assert!(event_of(&s).wait_until_signaled_value(3, 10_000));
    assert_eq!(s.counts.lock().skipped, 1);
    assert_eq!(s.counts.lock().landed, [2, 0, 0, 0, 0]);
}

/// A picture that fails to land is counted, its output left as it was, and its value signalled
/// all the same; a refused access unit faults its command buffer and spends no value.
#[test]
fn a_failed_landing_is_signalled_and_a_refusal_spends_no_value() {
    let p = FakePlatform::new();
    let log = Log::default();
    let s = session(&p, vec![Ok(Some(Picture::Fail)), Err("a bad slice")], &log);
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let pages = buffer(&p, 16384, STORAGE_SHARED);
    let d = || decode(&bitstream, &[], planes(&pages, &pages, 96));
    assert_eq!(s.decode(&d()).unwrap(), 1);
    assert!(event_of(&s).wait_until_signaled_value(1, 10_000));
    assert!(bytes(&pages).iter().all(|&x| x == UNTOUCHED));
    assert_eq!(s.counts.lock().failed, 1);
    assert!(s.decode(&d()).is_err());
    assert_eq!(s.decode(&d()).unwrap(), 2);
    assert!(event_of(&s).wait_until_signaled_value(2, 10_000));
}

/// A session's end waits for its last picture to land.
#[test]
fn an_end_waits_for_the_last_picture() {
    let p = FakePlatform::new();
    let log = Log::default();
    let (go, gate) = mpsc::channel();
    let s = session(
        &p,
        vec![Ok(Some(Picture::Gated(5, Arc::clone(&log), gate)))],
        &log,
    );
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let pages = buffer(&p, 16384, STORAGE_SHARED);
    assert_eq!(
        s.decode(&decode(&bitstream, &[], planes(&pages, &pages, 96)))
            .unwrap(),
        1
    );
    let opener = std::thread::Builder::new()
        .spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            go.send(()).unwrap();
        })
        .unwrap();
    drop(s);
    assert!(holds(&bytes(&pages), 0, 64, 96, 5));
    opener.join().unwrap();
}

/// An output plane past its buffer, or in pages with no host bytes, faults the decode before
/// the decoder sees it.
#[test]
fn an_output_past_its_buffer_faults_before_the_decoder() {
    let p = FakePlatform::new();
    let log = Log::default();
    let s = session(&p, vec![], &log);
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let short = buffer(&p, 8192, STORAGE_SHARED);
    assert!(
        s.decode(&decode(&bitstream, &[1], planes(&short, &short, 96)))
            .is_err()
    );
    let cache = buffer(&p, 16384, PRIVATE);
    let (luma, chroma) = planes(&cache, &cache, 96);
    let uncached = VideoPlane {
        cached: false,
        ..luma
    };
    assert!(
        s.decode(&decode(&bitstream, &[1], (uncached, chroma)))
            .is_err()
    );
    assert!(log.lock().is_empty());
}

/// A plane holds the op's `output_rows` where they lie in its buffer, half as many chroma rows,
/// and its crop's rows where they do not.
#[test]
fn a_plane_holds_the_output_rows_that_lie_in_its_buffer() {
    let p = FakePlatform::new();
    let pages = buffer(&p, 16384, STORAGE_SHARED);
    let (luma, chroma) = planes(&pages, &pages, 96);
    assert_eq!(output_plane(&luma, 68).unwrap().rows_held, 68);
    assert_eq!(output_plane(&chroma, 34).unwrap().rows_held, 34);
    assert_eq!(output_plane(&luma, 0).unwrap().rows_held, 64);
    // 66 chroma rows from 8192 end past the buffer's 16384 bytes.
    assert_eq!(output_plane(&chroma, 66).unwrap().rows_held, 32);
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let mut d = decode(&bitstream, &[], planes(&pages, &pages, 96));
    d.output_rows = 68;
    let output = Output::of(&d).unwrap();
    assert_eq!((output.luma.rows_held, output.chroma.rows_held), (68, 34));
}

/// A staged plane is copied at once where its rows fill their pitch, else a row at a time; a
/// plane of no rows copies nothing.
#[test]
fn a_staged_plane_copies_whole_or_by_rows() {
    let p = FakePlatform::new();
    let cache = buffer(&p, 16384, PRIVATE);
    let plane = |row_bytes, height| OutputPlane::<P> {
        buffer: cache.clone_ref(),
        offset: 4096,
        bytes_per_row: 128,
        row_bytes,
        width: 0,
        height,
        rows_held: height,
        cached: true,
        host: None,
        pages: None,
    };
    assert_eq!(plane_copies(256, &plane(128, 4)), [(256, 4096, 512)]);
    assert_eq!(
        plane_copies(256, &plane(100, 2)),
        [(256, 4096, 100), (384, 4224, 100)]
    );
    assert!(plane_copies(256, &plane(128, 0)).is_empty());
}
