use std::collections::VecDeque;
#[cfg(unix)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use reims_vgpu_graphics::wire::RGFX_VIDEO_CODEC_H264;
use reims_vgpu_video::core::device_memory::MemoryLife;
#[cfg(unix)]
use reims_vgpu_video::core::device_wait::Unfinished;
use reims_vgpu_video::core::runtime::StdRuntime;
use reims_vgpu_vulkan::core::testkit::FakePlatform;
use reims_vgpu_vulkan::core::testkit::calllog::Arg;
use reims_vgpu_vulkan::core::testkit::objects::CompleteOn;

use super::*;

type P = FakePlatform;
type FakeBuffer = <P as Platform>::Buffer;

/// `MTLResourceStorageModePrivate`: a device cache's storage, with no host bytes.
const PRIVATE: u64 = 2 << 4;
/// What the output's bytes hold before a picture lands.
const UNTOUCHED: u8 = 0xee;
const COPY: &str = "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:";
const FILL: &str = "fillBuffer:range:value:";
const RENDER: &str = "renderCommandEncoderWithDescriptor:";
/// What a short picture of the tests' covers of the planes of [`planes`]: its luma's 64 bytes of
/// its first 40 rows, and its chroma's of its first 20.
const SHORT: [Extent; 2] = [
    Extent {
        row_bytes: 64,
        rows: 40,
    },
    Extent {
        row_bytes: 64,
        rows: 20,
    },
];

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
    /// A failure to land once `go` is sent, after its decode was said done.
    FailDone(mpsc::Receiver<()>),
    /// As `Fill`, once `go` is sent, after its decode was said done and `told` sent: the rest of
    /// a landing behind its decode, as the VA-API arm's video processing and its draw on Windows.
    DoneGated(u8, Log, mpsc::Sender<()>, mpsc::Receiver<()>),
    /// As `Fill`, of a decode the hardware answered failed.
    FailedFill(u8, Log),
    /// `fill` by `Landing::copy_rows` from sources of what `only` covers of each plane.
    CpuOnly(u8, [Extent; 2]),
    /// `fill` in what `only` covers of each plane's host bytes, said once they are asked
    /// (`Landing::writes_only`), as the VA-API arm's CPU copy says it.
    HostOnly(u8, [Extent; 2]),
    /// As `CpuOnly`, said first, before the planes' host bytes are asked, as the NVDEC arm says
    /// it.
    SaidCpuOnly(u8, [Extent; 2]),
    /// What the hardware writes into the planes where they lie, saying first that it covers
    /// `only` of them; the planes' bytes here are left alone.
    InPlaceOnly([Extent; 2]),
    /// Landed by the GPU's copy into the output's device caches from a scratch the hardware
    /// wrote, as the VA-API arm's GPU copier lands it.
    GpuCopied,
    /// A failure to land in the output's device caches, which the hardware wrote, its work
    /// stopped ([`Landing::failed_in_caches`]).
    #[cfg(unix)]
    FailInCaches,
    /// A failure to land in the output's device caches whose work ran past its deadline: the
    /// work it keeps on the landing's thread, where it keeps one, which its device is done with
    /// once the flag is set ([`Stuck`]).
    #[cfg(unix)]
    StallInCaches(Option<Arc<AtomicBool>>),
}

/// A stalled picture's work, kept on the process's list until its device is done with it, once
/// the flag is set.
#[cfg(unix)]
struct Stuck(Arc<AtomicBool>);

#[cfg(unix)]
impl Unfinished for Stuck {
    fn finished(&mut self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
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
            Picture::FailDone(go) => {
                go.recv().unwrap();
                to.decode_done();
                Err("the GPU failed the copy".into())
            }
            Picture::DoneGated(n, log, told, go) => {
                to.decode_done();
                told.send(()).unwrap();
                go.recv().unwrap();
                fill(to, n, &log)
            }
            Picture::FailedFill(n, log) => {
                to.hardware_failed("status 8".to_owned());
                fill(to, n, &log)
            }
            Picture::CpuOnly(n, only) => {
                let rows = vec![n; 4096 * 128];
                let source = |extent: Extent| SourcePlane {
                    at: rows.as_ptr(),
                    bytes_per_row: 4096,
                    row_bytes: extent.row_bytes as usize,
                    rows: extent.rows as usize,
                };
                // SAFETY: `rows` holds 128 rows of 4096 bytes, apart from the output.
                unsafe { to.copy_rows(source(only[0]), source(only[1])) }
            }
            Picture::HostOnly(n, only) => {
                let planes = to.host()?;
                to.writes_only(only)?;
                for (plane, extent) in [(&planes.luma, only[0]), (&planes.chroma, only[1])] {
                    for row in 0..extent.rows {
                        let at = plane.at.wrapping_add(row * plane.bytes_per_row);
                        // SAFETY: the extent lies in the plane's rows, in its host bytes
                        // (`Landing::host`'s word).
                        unsafe {
                            let len = extent.row_bytes as usize;
                            std::ptr::write_bytes(at.as_ptr::<u8>(), n, len);
                        }
                    }
                }
                Ok(())
            }
            Picture::SaidCpuOnly(n, only) => {
                to.writes_only(only)?;
                Box::new(Picture::CpuOnly(n, only)).land(to)
            }
            Picture::InPlaceOnly(only) => to.writes_only(only),
            Picture::GpuCopied => {
                to.copied_by_gpu();
                Ok(())
            }
            #[cfg(unix)]
            Picture::FailInCaches => {
                let failed = "the hardware failed the decode".into();
                Err(to.failed_in_caches(&[to.luma], Failed::Stopped, failed))
            }
            #[cfg(unix)]
            Picture::StallInCaches(kept) => {
                let since = KeptMark::here();
                if let Some(done) = kept {
                    let runtime = StdRuntime::shared();
                    Lingering::process(&runtime)
                        .keep("a test's stalled write", Box::new(Stuck(done)));
                }
                let failed = "the hardware's write ran past its deadline".into();
                Err(to.failed_in_caches(&[to.luma], Failed::Stalled(since), failed))
            }
        }
    }
}

/// A decoder that logs what it is handed and answers `answers` in turn: a picture of
/// `Fill(n)` for decode `n` once they run out.
struct Decoder {
    log: Log,
    answers: VecDeque<Result<Option<Picture>, &'static str>>,
    decoded: u8,
    most_unlanded: usize,
}

impl HostDecoder<P> for Decoder {
    fn parameter_sets(&mut self, sets: &[u8]) -> Result<(), BackendError> {
        self.log.lock().push(Seen::Sets(sets.to_vec()));
        Ok(())
    }

    fn decode(
        &mut self,
        au: &[u8],
        _output: &DecodeOutput<P>,
    ) -> Result<Option<Box<dyn HostPicture<P>>>, BackendError> {
        (self.log.lock()).push(Seen::Unit(au.as_ptr().addr(), au.len()));
        self.decoded += 1;
        let answer = (self.answers.pop_front())
            .unwrap_or_else(|| Ok(Some(Picture::Fill(self.decoded, Arc::clone(&self.log)))));
        let picture = answer?;
        Ok(picture.map(|p| Box::new(p) as Box<dyn HostPicture<P>>))
    }

    fn most_unlanded(&self) -> usize {
        self.most_unlanded
    }
}

/// A session over a decoder answering `answers`, and the log it keeps, that spares 8 pictures.
fn session(
    p: &FakePlatform,
    answers: Vec<Result<Option<Picture>, &'static str>>,
    log: &Log,
) -> HostVideoSession<P> {
    session_sparing(p, answers, log, 8)
}

/// [`session`]'s, its decoder sparing `most_unlanded` pictures.
fn session_sparing(
    p: &FakePlatform,
    answers: Vec<Result<Option<Picture>, &'static str>>,
    log: &Log,
    most_unlanded: usize,
) -> HostVideoSession<P> {
    let decoder = Decoder {
        log: Arc::clone(log),
        answers: answers.into(),
        decoded: 0,
        most_unlanded,
    };
    HostVideoSession::new(
        &p.device(),
        StdRuntime::shared(),
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
        cache: None,
        rows_room: None,
        #[cfg(unix)]
        surface: None,
        #[cfg(windows)]
        drawn: None,
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

/// Whether `b`'s rows of a plane at `offset`, `rows` of them 128 apart, each of `row_bytes`, hold
/// `fill` in what `kept` covers of them, zeros in the rest of their `row_bytes`, and nothing past
/// those in each row.
fn holds_over_zeros(
    b: &[u8],
    offset: usize,
    (rows, row_bytes): (usize, usize),
    fill: u8,
    kept: Extent,
) -> bool {
    let (kept_rows, kept_bytes) = (kept.rows as usize, kept.row_bytes as usize);
    (0..rows).all(|row| {
        let at = offset + row * 128;
        let filled = if row < kept_rows { kept_bytes } else { 0 };
        b[at..at + filled].iter().all(|&x| x == fill)
            && b[at + filled..at + row_bytes].iter().all(|&x| x == 0)
            && b[at + row_bytes..at + 128].iter().all(|&x| x == UNTOUCHED)
    })
}

/// Lands `picture` in `output` on `stage` as the landing thread does, its command buffers
/// completed by their waits.
fn land_in(
    picture: Picture,
    output: &DecodeOutput<P>,
    stage: &mut Stage<P>,
) -> Result<Way, BackendError> {
    let clock = StdRuntime::shared();
    let landed = Arc::new(Landed::default());
    let (way, _, _) = land(Box::new(picture), output, stage, &*clock, &landed)?;
    Ok(way)
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
            .unwrap()
            .value,
        1
    );
    assert_eq!(
        s.decode(&decode(&bitstream, &[], planes(&pages, &pages, 96)))
            .unwrap()
            .value,
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
            .unwrap()
            .value,
        1
    );
    assert_eq!(
        s.decode(&decode(&bitstream, &[], planes(&second, &second, 96)))
            .unwrap()
            .value,
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
            .unwrap()
            .value,
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
            .unwrap()
            .value,
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

/// A picture the GPU copied into its output's device caches from a scratch the hardware wrote
/// is counted as landed staged by the hardware and copied, not as written there by the hardware,
/// and the session copies nothing of it.
#[test]
fn a_picture_the_gpu_copied_from_a_scratch_counts_as_staged_by_the_hardware() {
    let p = FakePlatform::new();
    let log = Log::default();
    let s = session(&p, vec![Ok(Some(Picture::GpuCopied))], &log);
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let cache = buffer(&p, 16384, PRIVATE);
    let decoded = s
        .decode(&decode(&bitstream, &[], planes(&cache, &cache, 128)))
        .unwrap();
    assert!(event_of(&s).wait_until_signaled_value(1, 10_000));
    assert!(decoded.landed.as_deref().is_some_and(|l| !l.failed()));
    assert!(
        p.call_log().calls_to(COPY).is_empty(),
        "no copy of the session's"
    );
    let at = |way: Way| Way::ALL.iter().position(|w| *w == way).unwrap();
    let counts = s.counts.lock();
    assert_eq!(
        (
            counts.landed[at(Way::Staged)],
            counts.landed[at(Way::Cache)]
        ),
        (1, 0)
    );
}

/// A cache's surface memory that keeps the session's signalled value at each acquire said of it.
#[cfg(unix)]
struct Acquires {
    event: <P as Platform>::SharedEvent,
    seen: Mutex<Vec<u64>>,
}

#[cfg(unix)]
impl reims_vgpu_vulkan::core::platform::resource::SurfaceMemory for Acquires {
    fn export(&self) -> Result<ExportedMemory, String> {
        Err("the test's memory exports nothing".to_owned())
    }

    fn external_writes_done(&self) -> Result<(), String> {
        self.seen.lock().push(self.event.signaled_value());
        Ok(())
    }

    fn handle(&self) -> reims_vgpu_vulkan::core::platform::resource::SurfaceHandle {
        reims_vgpu_vulkan::core::platform::resource::SurfaceHandle::DmaBuf
    }

    fn in_place(&self) -> bool {
        false
    }

    fn layout(&self) -> &[reims_vgpu_vulkan::core::platform::resource::SurfacePlaneLayout] {
        &[]
    }
}

/// A session whose decodes land in one output whose luma plane lies in a device cache over
/// [`Acquires`], which keeps the session's signalled value at each acquire: the session, the
/// memory, and a decode into the output.
#[cfg(unix)]
struct InCache {
    s: HostVideoSession<P>,
    memory: Arc<Acquires>,
    bitstream: FakeBuffer,
    cache: FakeBuffer,
    life: MemoryLife,
    texture: <P as Platform>::Texture,
}

#[cfg(unix)]
impl InCache {
    fn new(p: &FakePlatform, answers: Vec<Result<Option<Picture>, &'static str>>) -> Self {
        use reims_vgpu_vulkan::core::platform::values::TextureDesc;
        let s = session(p, answers, &Log::default());
        let memory = Arc::new(Acquires {
            event: event_of(&s).clone_ref(),
            seen: Mutex::new(Vec::new()),
        });
        let desc = p.texture_descriptor(&TextureDesc {
            pixel_format: 10, // MTLPixelFormatR8Unorm
            width: 96,
            height: 64,
            ..TextureDesc::default()
        });
        let texture = (p.device().new_texture(OnFault::Return, &desc))
            .unwrap()
            .unwrap();
        InCache {
            s,
            memory,
            bitstream: buffer(p, 4096, STORAGE_SHARED),
            cache: buffer(p, 16384, PRIVATE),
            life: MemoryLife::new(),
            texture,
        }
    }

    fn decode(&self) -> Decoded {
        let (mut luma, chroma) = planes(&self.cache, &self.cache, 96);
        let memory: Arc<dyn reims_vgpu_vulkan::core::platform::resource::SurfaceMemory> =
            Arc::clone(&self.memory) as _;
        luma.surface = Some(SurfaceTarget {
            memory,
            identity: self.life.identity(),
            texture: self.texture.clone_ref(),
            offset: 0,
            bytes_per_row: 128,
            rows: 64,
            clear: None,
            zeros: false,
        });
        (self.s)
            .decode(&decode(&self.bitstream, &[], (luma, chroma)))
            .unwrap()
    }
}

/// A picture that fails to land in its output's device cache, which the hardware wrote, has the
/// cache said written before the session signals its value, as the VA-API and NVDEC arms' failed
/// decodes and writes into caches do ([`Landing::failed_in_caches`]): the device's reads see
/// what the hardware left there. It is counted failed and its command buffer completes failed,
/// as any failure to land. One whose work ran past its deadline (a stall) and kept nothing is
/// said written at once too.
#[cfg(unix)]
#[test]
fn a_failed_landing_in_a_cache_says_it_written_before_its_signal() {
    let p = FakePlatform::new();
    let answers = vec![
        Ok(Some(Picture::FailInCaches)),
        Ok(Some(Picture::StallInCaches(None))),
    ];
    let c = InCache::new(&p, answers);

    let stopped = c.decode();
    assert_eq!(stopped.value, 1);
    assert!(event_of(&c.s).wait_until_signaled_value(1, 10_000));
    assert_eq!(
        *c.memory.seen.lock(),
        [0],
        "said written once, before the signal"
    );
    assert!(stopped.landed.as_deref().is_some_and(Landed::failed));

    let stalled = c.decode();
    assert_eq!(stalled.value, 2);
    assert!(event_of(&c.s).wait_until_signaled_value(2, 10_000));
    assert_eq!(
        *c.memory.seen.lock(),
        [0, 1],
        "a stall that kept nothing, said written before its signal"
    );
    assert!(stalled.landed.as_deref().is_some_and(Landed::failed));
    assert_eq!(c.s.counts.lock().failed, 2);
}

/// Whether `s`'s event reaches `value` within `within` of real time, the landing's clock (the
/// fake event's own waits are on the fake world's virtual clock), asked every 10 ms.
#[cfg(unix)]
fn signalled_within(s: &HostVideoSession<P>, value: u64, within: Duration) -> bool {
    let pause = Duration::from_millis(10);
    for _ in 0..=within.as_millis() / pause.as_millis() {
        if event_of(s).signaled_value() >= value {
            return true;
        }
        std::thread::sleep(pause);
    }
    event_of(s).signaled_value() >= value
}

/// A picture whose landing in its output's device cache ran past the hardware's deadline (a
/// stall), with work kept that may still write the cache, has the cache said written only once
/// that work is let go, its device done with it, and then before the session signals its value
/// ([`Landing::failed_in_caches`]): no read of the cache comes before its acquire. Until then the
/// session signals nothing, the later pictures' landings behind it; the next picture lands after
/// it as any.
#[cfg(unix)]
#[test]
fn a_stalled_landing_in_a_cache_says_it_written_once_its_kept_work_is_let_go() {
    let p = FakePlatform::new();
    let done = Arc::new(AtomicBool::new(false));
    let answers = vec![
        Ok(Some(Picture::StallInCaches(Some(Arc::clone(&done))))),
        Ok(Some(Picture::FailInCaches)),
    ];
    let c = InCache::new(&p, answers);
    let stalled = c.decode();
    let next = c.decode();
    assert_eq!((stalled.value, next.value), (1, 2));
    assert!(
        !signalled_within(&c.s, 1, Duration::from_millis(500)),
        "no signal while the work is kept"
    );
    assert!(c.memory.seen.lock().is_empty(), "nothing said while kept");
    done.store(true, Ordering::SeqCst);
    assert!(signalled_within(&c.s, 2, Duration::from_secs(10)));
    assert_eq!(
        *c.memory.seen.lock(),
        [0, 1],
        "said written once let go, before its signal, then the next picture's"
    );
    assert!(stalled.landed.as_deref().is_some_and(Landed::failed));
    assert_eq!(c.s.counts.lock().failed, 2);
}

/// A stalled landing whose kept work is not let go within its bound ends the process, at a
/// fault of the decode ([`hold_while_kept`]): the cache is never read unacquired, and nothing
/// waits for it unbounded. Work kept before the mark, or on another thread, holds nothing.
#[cfg(unix)]
#[test]
fn a_stalled_landing_whose_kept_work_is_not_let_go_ends_the_process() {
    use reims_vgpu_vulkan::core::testkit::capture::catch_death;
    let runtime = StdRuntime::shared();
    let lingering = Lingering::process(&runtime);
    let error: BackendError = "the hardware's write ran past its deadline".into();
    let before = Arc::new(AtomicBool::new(false));
    lingering.keep(
        "a test's work kept before",
        Box::new(Stuck(Arc::clone(&before))),
    );
    let since = KeptMark::here();
    let elsewhere = Arc::new(AtomicBool::new(false));
    std::thread::scope(|scope| {
        let kept = Box::new(Stuck(Arc::clone(&elsewhere)));
        scope.spawn(|| lingering.keep("a test's work kept elsewhere", kept));
    });
    let within = Duration::from_millis(300);
    assert_eq!(
        catch_death(|| hold_while_kept(&*runtime, since, within, &error)),
        Ok(()),
        "nothing this thread kept since the mark"
    );
    let stuck = Arc::new(AtomicBool::new(false));
    lingering.keep(
        "a test's stalled write",
        Box::new(Stuck(Arc::clone(&stuck))),
    );
    let started = runtime.now();
    assert_eq!(
        catch_death(|| hold_while_kept(&*runtime, since, within, &error)),
        Err(decode_fault("").site())
    );
    assert!(
        runtime.elapsed_since(started) >= within,
        "held to its bound"
    );
    for done in [&before, &elsewhere, &stuck] {
        done.store(true, Ordering::SeqCst);
    }
}

/// A copy into the output's caches that the platform makes no blit encoder for is skipped, as
/// the decoder skips whole a pass the platform makes no encoder of: the caches keep their bytes,
/// and the picture lands with them, its value signalled and its command buffer completing as any
/// landed picture's, not counted as a failure to land; the next picture's copy runs as any.
#[test]
fn a_copy_into_caches_the_platform_makes_no_blit_encoder_for_is_skipped() {
    use reims_vgpu_vulkan::core::testkit::Step;
    let p = FakePlatform::new();
    p.world().set_auto_complete(Some(CompleteOn::Caller));
    let log = Log::default();
    let s = session(&p, vec![], &log);
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let cache = buffer(&p, 16384, PRIVATE);
    p.script().push("blitCommandEncoder", Step::Nil);
    let skipped = (s.decode(&decode(&bitstream, &[], planes(&cache, &cache, 128)))).unwrap();
    assert_eq!(skipped.value, 1);
    assert!(event_of(&s).wait_until_signaled_value(1, 10_000));
    assert!(
        p.call_log().calls_to(COPY).is_empty(),
        "no copy into the caches"
    );
    assert!(
        skipped.landed.as_deref().is_some_and(|l| !l.failed()),
        "its command buffer completes as any landed picture's"
    );
    assert_eq!(s.counts.lock().failed, 0);

    let next = (s.decode(&decode(&bitstream, &[], planes(&cache, &cache, 128)))).unwrap();
    assert_eq!(next.value, 2);
    assert!(event_of(&s).wait_until_signaled_value(2, 10_000));
    assert_eq!(
        p.call_log().calls_to(COPY).len(),
        2,
        "the next picture's copy"
    );
    assert!(next.landed.as_deref().is_some_and(|l| !l.failed()));
    assert_eq!(s.counts.lock().failed, 0);
    drop(s);
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
        assert_eq!(s.decode(&d).unwrap().value, value);
    }
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(event_of(&s).signaled_value(), 0);
    go.send(()).unwrap();
    assert!(event_of(&s).wait_until_signaled_value(3, 10_000));
    assert_eq!(s.counts.lock().skipped, 1);
    assert_eq!(s.counts.lock().landed, [2, 0, 0, 0, 0, 0]);
}

/// A picture of the stream that does not decode ([`picture_error`]) lands nothing: its value is
/// signalled in its turn, its command buffer completes as any, and its output keeps its bytes.
#[test]
fn a_picture_of_the_stream_that_does_not_decode_lands_nothing_and_completes() {
    assert!(picture_error::<P>("test", "a reference frame is missing").is_none());
    let p = FakePlatform::new();
    let log = Log::default();
    let s = session(&p, vec![Ok(None)], &log);
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let (first, second) = (
        buffer(&p, 16384, STORAGE_SHARED),
        buffer(&p, 16384, STORAGE_SHARED),
    );
    let none = s
        .decode(&decode(&bitstream, &[], planes(&first, &first, 96)))
        .unwrap();
    assert_eq!(none.value, 1);
    let next = s
        .decode(&decode(&bitstream, &[], planes(&second, &second, 96)))
        .unwrap();
    assert_eq!(next.value, 2);
    assert!(event_of(&s).wait_until_signaled_value(2, 10_000));
    assert!(!none.landed.as_deref().is_some_and(Landed::failed));
    assert!(bytes(&first).iter().all(|&x| x == UNTOUCHED));
    assert!(holds(&bytes(&second), 0, 64, 96, 2));
    assert_eq!(s.counts.lock().skipped, 1);
}

/// A decode that would have the session hold more pictures unlanded than its decoder spares
/// waits, once decoded, for the oldest to land.
#[test]
fn a_decode_past_the_decoder_s_spare_pictures_waits_for_the_oldest_to_land() {
    let p = FakePlatform::new();
    let log = Log::default();
    let (go, gate) = mpsc::channel();
    let answers = vec![Ok(Some(Picture::Gated(1, Arc::clone(&log), gate)))];
    let s = session_sparing(&p, answers, &log, 2);
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let pages = buffer(&p, 16384, STORAGE_SHARED);
    let d = || decode(&bitstream, &[], planes(&pages, &pages, 96));
    // The first lands behind its gate, and the second waits for the landing thread.
    assert_eq!(s.decode(&d()).unwrap().value, 1);
    assert_eq!(s.decode(&d()).unwrap().value, 2);
    let (done, decoded) = mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(|| done.send(s.decode(&d()).unwrap().value).unwrap());
        assert!(decoded.recv_timeout(Duration::from_millis(100)).is_err());
        go.send(()).unwrap();
        assert_eq!(decoded.recv_timeout(Duration::from_secs(10)).unwrap(), 3);
    });
    assert!(event_of(&s).wait_until_signaled_value(3, 10_000));
}

/// A picture that fails to land is counted, its output left as it was, and its value signalled
/// all the same, its decode answering that it did not land, which fails the command buffer that
/// waits for it; a refused access unit faults its command buffer and spends no value.
#[test]
fn a_failed_landing_is_signalled_and_a_refusal_spends_no_value() {
    let p = FakePlatform::new();
    let log = Log::default();
    let s = session(&p, vec![Ok(Some(Picture::Fail)), Err("a bad slice")], &log);
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let pages = buffer(&p, 16384, STORAGE_SHARED);
    let d = || decode(&bitstream, &[], planes(&pages, &pages, 96));
    let failed = s.decode(&d()).unwrap();
    assert_eq!(failed.value, 1);
    assert!(event_of(&s).wait_until_signaled_value(1, 10_000));
    assert!(bytes(&pages).iter().all(|&x| x == UNTOUCHED));
    assert_eq!(s.counts.lock().failed, 1);
    assert!(
        failed.landed.as_deref().is_some_and(Landed::failed),
        "its command buffer completes failed"
    );
    assert!(s.decode(&d()).is_err());
    let next = s.decode(&d()).unwrap();
    assert_eq!(next.value, 2);
    assert!(event_of(&s).wait_until_signaled_value(2, 10_000));
    assert!(next.landed.as_deref().is_some_and(|l| !l.failed()));
}

/// A picture that does not land after its command buffer completed at its decode, its copy
/// into the output's caches failed, fails the session's next decode in its stead, counted
/// ([`CARRIED_CODE`]); the decode after that one lands as any.
#[test]
fn a_failure_after_a_completion_at_the_decode_fails_the_next_decode_alone() {
    let counted = || {
        (reims_vgpu_vulkan::answer::counted_kinds().into_iter())
            .find(|&(code, _)| code == CARRIED_CODE)
            .map_or(0, |(_, n)| n)
    };
    let before = counted();
    let p = FakePlatform::new();
    let log = Log::default();
    let (go, gate) = mpsc::channel();
    let s = session(&p, vec![Ok(Some(Picture::FailDone(gate)))], &log);
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let cache = buffer(&p, 16384, PRIVATE);
    let pages = buffer(&p, 16384, STORAGE_SHARED);
    let first = (s.decode(&decode(&bitstream, &[], planes(&cache, &cache, 128))))
        .unwrap()
        .landed
        .unwrap();
    // Its command buffer's commit arms its completion's post at the decode.
    first.arm_answering();
    let d = || decode(&bitstream, &[], planes(&pages, &pages, 96));
    let (next, after) = (s.decode(&d()).unwrap(), s.decode(&d()).unwrap());
    go.send(()).unwrap();
    assert!(event_of(&s).wait_until_signaled_value(3, 10_000));
    assert!(first.failed());
    assert_eq!(
        counted(),
        before + 1,
        "its completion was posted at its decode"
    );
    assert!(
        next.landed.as_deref().is_some_and(Landed::failed),
        "the next decode's command buffer completes failed"
    );
    assert!(
        after.landed.as_deref().is_some_and(|l| !l.failed()),
        "the decode after it is not affected"
    );
    assert!(holds(&bytes(&pages), 0, 64, 96, 3));
    assert_eq!(s.counts.lock().failed, 1);
}

/// A picture whose landing tells of its decode done before the rest of it (the VA-API arm's video
/// processing, and on Windows its draw into a cache's surface) has its command buffer's
/// completion posted then, while the session's event, which the buffer's end and so every reader
/// of the output waits for, reaches the decode's value only once the rest has landed: its planes
/// written and copied into their caches (`super::super::decode`'s video module docs, "Completed
/// at the decode").
#[test]
fn a_picture_landed_behind_its_decode_done_is_posted_then_and_signalled_once_landed() {
    let p = FakePlatform::new();
    p.world().set_auto_complete(Some(CompleteOn::Caller));
    let log = Log::default();
    let (told, heard) = mpsc::channel();
    let (go, gate) = mpsc::channel();
    let picture = Picture::DoneGated(5, Arc::clone(&log), told, gate);
    let s = session(&p, vec![Ok(Some(picture))], &log);
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let cache = buffer(&p, 16384, PRIVATE);
    let landed = (s.decode(&decode(&bitstream, &[], planes(&cache, &cache, 128))))
        .unwrap()
        .landed
        .unwrap();
    // Its command buffer's commit arms its completion's post at the decode.
    landed.arm_answering();
    heard.recv_timeout(Duration::from_secs(10)).unwrap();
    assert!(landed.posted(), "its completion goes at its decode");
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        event_of(&s).signaled_value(),
        0,
        "signalled before it landed"
    );
    assert!(p.call_log().calls_to(COPY).is_empty());
    assert!(!log.lock().iter().any(|s| matches!(s, Seen::Landed { .. })));

    go.send(()).unwrap();
    assert!(event_of(&s).wait_until_signaled_value(1, 10_000));
    assert!(log.lock().iter().any(|s| matches!(s, Seen::Landed { .. })));
    assert_eq!(
        p.call_log().calls_to(COPY).len(),
        2,
        "its planes copied into their caches before the signal"
    );
    assert!(!landed.failed());
    assert_eq!(s.counts.lock().failed, 0);
}

/// A picture the hardware answered failed that its backend lands all the same lands as any, its
/// value signalled and its command buffer completing as any landed picture's, and is counted
/// apart from the pictures that failed to land.
#[test]
fn a_picture_the_hardware_failed_lands_as_any_and_is_counted() {
    let p = FakePlatform::new();
    let log = Log::default();
    let s = session(
        &p,
        vec![Ok(Some(Picture::FailedFill(7, Arc::clone(&log))))],
        &log,
    );
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let pages = buffer(&p, 16384, STORAGE_SHARED);
    let decoded = s
        .decode(&decode(&bitstream, &[], planes(&pages, &pages, 96)))
        .unwrap();
    assert!(event_of(&s).wait_until_signaled_value(1, 10_000));
    let b = bytes(&pages);
    assert!(holds(&b, 0, 64, 96, 7) && holds(&b, 8192, 32, 96, 7));
    assert!(decoded.landed.as_deref().is_some_and(|l| !l.failed()));
    let counts = s.counts.lock();
    assert_eq!((counts.failed, counts.landed_failed), (0, 1));
}

/// A session counts the outputs its decodes landed in by their memory's identity, once each,
/// and the most whose memory lived at once at a decode: an output whose memory went is no longer
/// live. A decode into memory of no identity is counted apart.
#[test]
fn a_session_counts_its_outputs_by_their_memory_and_the_most_live_at_once() {
    let p = FakePlatform::new();
    let log = Log::default();
    let s = session(&p, Vec::new(), &log);
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let pages = buffer(&p, 16384, STORAGE_SHARED);
    let (a, b, c) = (MemoryLife::new(), MemoryLife::new(), MemoryLife::new());
    let into = |life: Option<&MemoryLife>| {
        let (mut luma, chroma) = planes(&pages, &pages, 96);
        luma.cache = life.map(MemoryLife::identity);
        s.decode(&decode(&bitstream, &[], (luma, chroma))).unwrap();
    };
    into(Some(&a));
    into(Some(&b));
    into(Some(&a));
    drop(a);
    into(Some(&c));
    into(Some(&b));
    into(None);
    let state = s.state.lock();
    let outputs = &state.outputs;
    assert_eq!(
        (outputs.distinct, outputs.most_live, outputs.live.len()),
        (3, 2, 2)
    );
    assert_eq!(outputs.unknown, 1);
}

/// A picture the CPU copied is counted with its copy's time; one the hardware wrote is not.
#[test]
fn a_session_times_the_cpu_s_copies() {
    let p = FakePlatform::new();
    let log = Log::default();
    let answers = vec![
        Ok(Some(Picture::Cpu(9, Arc::clone(&log)))),
        Ok(Some(Picture::Fill(4, Arc::clone(&log)))),
    ];
    let s = session(&p, answers, &log);
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let pages = buffer(&p, 16384, STORAGE_SHARED);
    for _ in 0..2 {
        s.decode(&decode(&bitstream, &[], planes(&pages, &pages, 96)))
            .unwrap();
    }
    assert!(event_of(&s).wait_until_signaled_value(2, 10_000));
    let counts = s.counts.lock();
    assert_eq!(counts.cpu_copies, 1);
    assert!(counts.cpu_most <= counts.cpu_total);
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
            .unwrap()
            .value,
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
/// and its crop's rows where they do not; and no more than the rows the host's layout of the
/// plane holds where it laid the plane out, so that a picture's rows past the plane's never run
/// into the next plane's (a 1080p luma plane's 1088 coded rows, where chroma lies right past its
/// 1080 in a cache laid out at the GPU's pitch).
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
    // Where the host laid the plane out, the rows it holds there bound it: no more than 66 rows
    // of luma, whatever the buffer's bytes past them.
    let laid = VideoPlane {
        rows_room: Some(66),
        ..planes(&pages, &pages, 96).0
    };
    assert_eq!(output_plane(&laid, 68).unwrap().rows_held, 66);
    let laid = VideoPlane {
        rows_room: Some(64),
        ..planes(&pages, &pages, 96).0
    };
    assert_eq!(output_plane(&laid, 68).unwrap().rows_held, 64);
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let mut d = decode(&bitstream, &[], planes(&pages, &pages, 96));
    d.output_rows = 68;
    let output = DecodeOutput::of(&d).unwrap();
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
        cache: None,
        #[cfg(unix)]
        surface: None,
        #[cfg(windows)]
        drawn: None,
    };
    assert_eq!(plane_copies(256, &plane(128, 4)), [(256, 4096, 512)]);
    assert_eq!(
        plane_copies(256, &plane(100, 2)),
        [(256, 4096, 100), (384, 4224, 100)]
    );
    assert!(plane_copies(256, &plane(128, 0)).is_empty());
}

/// A picture short of its planes lands with zeros in the rest of each plane's crop, in the
/// surface's pages, by the CPU's copy and by the hardware's write, which says first what it
/// covers: no texel past it keeps the plane's earlier bytes. The bytes past each row's crop, and
/// past the plane's rows, stay as they were.
#[test]
fn a_short_picture_lands_over_zeros_in_the_pages() {
    let p = FakePlatform::new();
    let log = Log::default();
    let answers = vec![
        Ok(Some(Picture::CpuOnly(9, SHORT))),
        Ok(Some(Picture::HostOnly(7, SHORT))),
        Ok(Some(Picture::InPlaceOnly(SHORT))),
    ];
    let s = session(&p, answers, &log);
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let outputs = [
        buffer(&p, 16384, STORAGE_SHARED),
        buffer(&p, 16384, STORAGE_SHARED),
        buffer(&p, 16384, STORAGE_SHARED),
    ];
    for output in &outputs {
        s.decode(&decode(&bitstream, &[], planes(output, output, 96)))
            .unwrap();
    }
    drop(s);
    // The hardware's own write leaves its extent as it was, here the untouched bytes.
    for (output, fill) in outputs.iter().zip([9, 7, UNTOUCHED]) {
        let b = bytes(output);
        assert!(holds_over_zeros(&b, 0, (64, 96), fill, SHORT[0]));
        assert!(holds_over_zeros(&b, 8192, (32, 96), fill, SHORT[1]));
        assert!(b[64 * 128..8192].iter().all(|&x| x == UNTOUCHED));
        assert!(b[8192 + 32 * 128..].iter().all(|&x| x == UNTOUCHED));
    }
    assert_eq!(
        p.call_log().calls_to(FILL).len(),
        0,
        "no GPU clear of pages"
    );
}

/// A picture short of its planes in a device cache, staged in the session's staging buffer, is
/// staged with zeros in the rest of each plane's crop, by the CPU's copy and by the hardware's
/// write that says what it covers, after a whole picture left the buffer full of its bytes: the
/// copy into the cache then writes the zeros. A picture said to cover its planes is staged as it
/// writes them.
#[test]
fn a_short_picture_staged_for_a_cache_lands_over_zeros() {
    let p = FakePlatform::new();
    p.world().set_auto_complete(Some(CompleteOn::Caller));
    let mut stage = Stage {
        device: p.device(),
        queue: None,
        buffer: None,
    };
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let cache = buffer(&p, 16384, PRIVATE);
    let output = DecodeOutput::of(&decode(&bitstream, &[], planes(&cache, &cache, 128))).unwrap();
    let log = Log::default();
    let cases = [
        (9, SHORT, Picture::CpuOnly as fn(u8, [Extent; 2]) -> Picture),
        (7, SHORT, Picture::HostOnly),
        (5, [Extent::NONE, SHORT[1]], Picture::HostOnly),
    ];
    for (fill, only, picture) in cases {
        assert_eq!(
            land_in(Picture::Fill(1, Arc::clone(&log)), &output, &mut stage).unwrap(),
            Way::Staged
        );
        land_in(picture(fill, only), &output, &mut stage).unwrap();
        let staged = bytes(stage.buffer.as_ref().unwrap());
        // The staged planes' rows fill their pitch: nothing lies past their row bytes.
        let whole = |b: &[u8], at: usize, rows: usize, extent: Extent| {
            (0..rows).all(|row| {
                let at = at + row * 128;
                let filled = if row < extent.rows as usize {
                    extent.row_bytes as usize
                } else {
                    0
                };
                b[at..at + filled].iter().all(|&x| x == fill)
                    && b[at + filled..at + 128].iter().all(|&x| x == 0)
            })
        };
        assert!(whole(&staged, 0, 64, only[0]), "the luma plane, {fill}");
        assert!(
            whole(&staged, 8192, 32, only[1]),
            "the chroma plane, {fill}"
        );
    }
    // Covering its planes, a picture's said extent zeros nothing.
    let covers = [
        Extent {
            row_bytes: 128,
            rows: 64,
        },
        Extent {
            row_bytes: 128,
            rows: 32,
        },
    ];
    land_in(Picture::HostOnly(3, covers), &output, &mut stage).unwrap();
    let staged = bytes(stage.buffer.as_ref().unwrap());
    assert!(staged[..8192 + 4096].iter().all(|&x| x == 3));
    assert_eq!(
        p.call_log().calls_to(FILL).len(),
        0,
        "no GPU clear of staged planes"
    );

    // Said before the planes are staged, the short picture has the cache's planes cleared, and
    // then staged with zeros past it, which the copy writes over the clear.
    land_in(Picture::SaidCpuOnly(4, SHORT), &output, &mut stage).unwrap();
    assert_eq!(p.call_log().calls_to(FILL).len(), 2);
    let staged = bytes(stage.buffer.as_ref().unwrap());
    let at = |row: usize| row * 128;
    assert!(staged[at(0)..at(0) + 64].iter().all(|&x| x == 4));
    assert!(staged[at(0) + 64..at(1)].iter().all(|&x| x == 0));
    assert!(staged[at(40)..8192].iter().all(|&x| x == 0));
    assert!(staged[8192 + at(20)..8192 + 4096].iter().all(|&x| x == 0));
}

/// A picture short of its planes in a device cache that the hardware writes where they lie has
/// the session clear each plane whole first, on its queue, waited for: a fill of the plane's
/// bytes in the cache's buffer, from its first row's first byte to its last row's last, before
/// the hardware's write. A picture that covers its planes, or a plane it covers, takes none.
#[test]
fn a_short_picture_the_hardware_writes_in_a_cache_has_the_plane_cleared_first() {
    let p = FakePlatform::new();
    p.world().set_auto_complete(Some(CompleteOn::Caller));
    let mut stage = Stage {
        device: p.device(),
        queue: None,
        buffer: None,
    };
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let cache = buffer(&p, 16384, PRIVATE);
    let output = DecodeOutput::of(&decode(&bitstream, &[], planes(&cache, &cache, 96))).unwrap();
    assert_eq!(
        land_in(Picture::InPlaceOnly(SHORT), &output, &mut stage).unwrap(),
        Way::Cache
    );
    let fills = p.call_log().calls_to(FILL);
    let fill = |at, len| {
        vec![
            Arg::obj(&cache),
            Arg::val(&NsRange::new(at, len)),
            Arg::U8(0),
        ]
    };
    assert_eq!(fills.len(), 2);
    assert_eq!(fills[0].args, fill(0, 63 * 128 + 96));
    assert_eq!(fills[1].args, fill(8192, 31 * 128 + 96));
    assert!(stage.buffer.is_none(), "nothing staged");

    // The luma covered, the chroma short: the chroma's plane alone.
    let luma = Extent {
        row_bytes: 96,
        rows: 64,
    };
    land_in(Picture::InPlaceOnly([luma, SHORT[1]]), &output, &mut stage).unwrap();
    let fills = &p.call_log().calls_to(FILL)[2..];
    assert_eq!(fills.len(), 1);
    assert_eq!(fills[0].args, fill(8192, 31 * 128 + 96));

    // Both covered: no clear.
    let chroma = Extent {
        row_bytes: 96,
        rows: 32,
    };
    land_in(Picture::InPlaceOnly([luma, chroma]), &output, &mut stage).unwrap();
    assert_eq!(p.call_log().calls_to(FILL).len(), 3);
}

/// A picture short of its planes in their cache's surface that the hardware writes there has
/// each plane's image cleared first by its clear pass, a render pass of the image alone, on the
/// session's queue, waited for; the cache's buffer takes no fill.
#[cfg(unix)]
#[test]
fn a_short_picture_in_a_cache_s_surface_has_its_images_cleared_first() {
    use reims_vgpu_vulkan::core::platform::resource::{
        SurfaceDesc, SurfaceHandle, SurfacePlaneDesc,
    };
    let p = FakePlatform::new();
    p.world().set_auto_complete(Some(CompleteOn::Caller));
    p.world().give_exported_surfaces(true);
    let plane = |pixel_format, width, height| SurfacePlaneDesc {
        pixel_format,
        width,
        height,
        usage: 0x15,
        rows_below: 0,
    };
    let descs = [plane(10, 96, 64), plane(30, 48, 32)];
    let desc = SurfaceDesc {
        handle: SurfaceHandle::DmaBuf,
        pitch_multiple: 128,
        start_multiple: 256,
        planes: &descs,
    };
    let made = (p.device().new_exported_surface(OnFault::Return, &desc))
        .unwrap()
        .unwrap();
    let laid = made.memory.layout().to_vec();
    let life = MemoryLife::new();
    let clears = [p.render_pass_descriptor(), p.render_pass_descriptor()];
    let target = |n: usize| SurfaceTarget {
        memory: Arc::clone(&made.memory),
        identity: life.identity(),
        texture: made.planes[n].clone_ref(),
        offset: laid[n].offset,
        bytes_per_row: laid[n].row_pitch,
        rows: descs[n].height,
        clear: Some(clears[n].clone_ref()),
        zeros: false,
    };
    let mut stage = Stage {
        device: p.device(),
        queue: None,
        buffer: None,
    };
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let cache = buffer(&p, 16384, PRIVATE);
    let (luma, chroma) = planes(&cache, &cache, 96);
    let (luma, chroma) = (
        VideoPlane {
            surface: Some(target(0)),
            ..luma
        },
        VideoPlane {
            surface: Some(target(1)),
            ..chroma
        },
    );
    let output = DecodeOutput::of(&decode(&bitstream, &[], (luma, chroma))).unwrap();
    land_in(Picture::InPlaceOnly(SHORT), &output, &mut stage).unwrap();
    let passes = p.call_log().calls_to(RENDER);
    assert_eq!(passes.len(), 2);
    assert_eq!(passes[0].args, [Arg::obj(&clears[0])]);
    assert_eq!(passes[1].args, [Arg::obj(&clears[1])]);
    assert!(p.call_log().calls_to(FILL).is_empty());
}

/// A picture that fails to land in its planes in their cache's surface, where its decode took
/// their fills as met (`SurfaceTarget::zeros`), has each plane's image cleared by its clear pass
/// on the session's queue before its value is signalled, as the fill would have left them; where
/// the decode took none, its planes are left as they were.
#[cfg(unix)]
#[test]
fn a_picture_that_fails_to_land_in_planes_owed_zeros_has_them_cleared() {
    use reims_vgpu_vulkan::core::platform::resource::{
        SurfaceDesc, SurfaceHandle, SurfacePlaneDesc,
    };
    let p = FakePlatform::new();
    p.world().set_auto_complete(Some(CompleteOn::Caller));
    p.world().give_exported_surfaces(true);
    let plane = |pixel_format, width, height| SurfacePlaneDesc {
        pixel_format,
        width,
        height,
        usage: 0x15,
        rows_below: 0,
    };
    let descs = [plane(10, 96, 64), plane(30, 48, 32)];
    let desc = SurfaceDesc {
        handle: SurfaceHandle::DmaBuf,
        pitch_multiple: 128,
        start_multiple: 256,
        planes: &descs,
    };
    let made = (p.device().new_exported_surface(OnFault::Return, &desc))
        .unwrap()
        .unwrap();
    let laid = made.memory.layout().to_vec();
    let life = MemoryLife::new();
    let clears = [p.render_pass_descriptor(), p.render_pass_descriptor()];
    let target = |n: usize, zeros: bool| SurfaceTarget {
        memory: Arc::clone(&made.memory),
        identity: life.identity(),
        texture: made.planes[n].clone_ref(),
        offset: laid[n].offset,
        bytes_per_row: laid[n].row_pitch,
        rows: descs[n].height,
        clear: Some(clears[n].clone_ref()),
        zeros,
    };
    let log = Log::default();
    let answers = vec![Ok(Some(Picture::Fail)), Ok(Some(Picture::Fail))];
    let s = session(&p, answers, &log);
    let bitstream = buffer(&p, 4096, STORAGE_SHARED);
    let cache = buffer(&p, 16384, PRIVATE);
    for (value, zeros) in [(1, true), (2, false)] {
        let (luma, chroma) = planes(&cache, &cache, 96);
        let (luma, chroma) = (
            VideoPlane {
                surface: Some(target(0, zeros)),
                ..luma
            },
            VideoPlane {
                surface: Some(target(1, zeros)),
                ..chroma
            },
        );
        let mark = p.call_log().mark();
        let failed = s.decode(&decode(&bitstream, &[], (luma, chroma))).unwrap();
        assert!(event_of(&s).wait_until_signaled_value(value, 10_000));
        assert!(failed.landed.as_deref().is_some_and(Landed::failed));
        let calls = p.call_log().since(mark);
        let passes: Vec<_> = (calls.iter())
            .filter(|c| c.sel == RENDER)
            .map(|c| c.args.clone())
            .collect();
        let want = match zeros {
            true => vec![vec![Arg::obj(&clears[0])], vec![Arg::obj(&clears[1])]],
            false => Vec::new(),
        };
        assert_eq!(passes, want, "zeros {zeros}");
        assert!(
            !calls.iter().any(|c| c.sel == FILL),
            "the cache's buffer takes no fill"
        );
    }
    assert_eq!(s.counts.lock().failed, 2);
}

/// A 4:2:0 picture's extent: the luma's samples and rows, the chroma's pairs and rows, each half
/// as many rounded up.
#[test]
fn a_4_2_0_picture_covers_half_its_rows_and_columns_of_chroma_rounded_up() {
    assert_eq!(
        Extent::of_420(1279, 719, 2),
        [
            Extent {
                row_bytes: 2558,
                rows: 719
            },
            Extent {
                row_bytes: 2560,
                rows: 360
            },
        ]
    );
    assert!(Extent::of_420(96, 64, 1)[1].covers(96, 32));
    assert!(!Extent::NONE.covers(1, 1));
}
