//! The crate's waits past their deadlines on a GPU (`super`): a decode, a
//! staged copy back's run and an encode, each asked after with no time to
//! run at all, so that each is still running at its deadline. Run on the VA-API host, on the render node
//! `RVG_VAAPI_NODE` names (`/dev/dri/renderD128` where it is unset).

use std::sync::atomic::Ordering;
use std::time::Duration;

use reims_vgpu_video_core::runtime::{Runtime, StdRuntime};

use super::tests::{KEPT_EVER, WAIT, within_5_s};
use super::{Work, current, kept, wait_for};
use crate::encode::{EncoderSettings, InputFrame, Plane, Profile};
use crate::va::caps::{self, Config, Context, Entrypoint};
use crate::va::display::{Display, Error};
use crate::va::encoder::VideoEncoder;
use crate::va::frame::Frame;
use crate::va::h264::H264Decoder;
use crate::va::readback::copy;
use crate::va::surface::{Surface, SurfaceFormat, SurfacePool};

/// The size the tests decode and encode at: a 4K frame takes the GPU
/// milliseconds.
pub(super) const SIZE: (u32, u32) = (3840, 2160);

/// How many tries a test makes for work still running when it is asked
/// after: it may be done before.
const TRIES: usize = 20;

/// The wait an encode's frame takes: long enough for its upload's map of
/// an idle surface, made on the display's call thread (1.3 ms on the RX
/// 7900 XTX, `crate::va::calls`), short of a 4K encode.
const ENCODE_WAIT: Duration = Duration::from_millis(4);

pub(super) fn display() -> Display {
    let node = std::env::var("RVG_VAAPI_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
    Display::open(node, StdRuntime::shared()).expect("a VA display")
}

/// Frame `n` of noise at [`SIZE`], NV12.
pub(super) fn noise(n: u64) -> Vec<u8> {
    let (width, height) = SIZE;
    let len = usize::try_from(u64::from(width) * u64::from(height) * 3 / 2).unwrap_or(0);
    let mut state = 0x9e37_79b9_7f4a_7c15_u64 ^ n.wrapping_mul(0x2545_f491_4f6c_dd1d);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state.to_le_bytes()[0]
        })
        .collect()
}

/// `bytes`, frame `n`, as the encoder takes it.
pub(super) fn input(bytes: &[u8], n: u64) -> InputFrame<'_> {
    let row = SIZE.0 as usize;
    let (luma, chroma) = bytes.split_at(row * SIZE.1 as usize);
    InputFrame {
        planes: [
            Plane {
                data: luma,
                bytes_per_row: row,
            },
            Plane {
                data: chroma,
                bytes_per_row: row,
            },
        ],
        pts: i64::try_from(n).unwrap_or(0) * 3000,
        duration: 3000,
        force_key_frame: false,
        id: n,
    }
}

/// An H.264 encoder at [`SIZE`] with no B pictures.
pub(super) fn encoder(display: &Display) -> VideoEncoder {
    let mut settings = EncoderSettings::new(Profile::H264High, SIZE.0, SIZE.1);
    settings.allow_frame_reordering = false;
    VideoEncoder::new(display, settings).expect("an H.264 encoder")
}

/// `count` frames of noise, H.264 by the driver's encoder: its avcC, and
/// its samples.
pub(super) fn h264(display: &Display, count: u64) -> (Vec<u8>, Vec<Vec<u8>>) {
    let mut encoder = encoder(display);
    let mut samples = Vec::new();
    for n in 0..count {
        let bytes = noise(n);
        let coded = encoder.encode(&input(&bytes, n)).expect("a frame taken");
        samples.extend(coded.into_iter().map(|c| c.expect("a sample").data));
    }
    samples.extend(
        encoder
            .flush()
            .into_iter()
            .map(|c| c.expect("a sample").data),
    );
    (encoder.configuration().to_vec(), samples)
}

/// Decodes `samples`' first, an IDR picture, and waits for it. A
/// decoder's first `vaEndPicture` waits for its answer on the display's
/// call thread, which starts the frame's deadline (`crate::va::buffer`), so
/// a test that shortens the wait once its frames are decoded decodes them
/// after this.
pub(super) fn begin(decoder: &mut H264Decoder, samples: &[Vec<u8>]) {
    let mut frames = decoder
        .decode(&samples[0], 1 << 40)
        .expect("a first decode");
    frames.extend(decoder.flush().expect("a flush"));
    for frame in &frames {
        frame.sync().expect("a first decode, the GPU free");
    }
}

/// Every frame `samples` decode into, flushed at the end.
pub(super) fn decoded(decoder: &mut H264Decoder, samples: &[Vec<u8>]) -> Vec<Frame> {
    let mut frames = Vec::new();
    for (id, sample) in (0u64..).zip(samples) {
        frames.extend(decoder.decode(sample, id).expect("a decode"));
    }
    frames.extend(decoder.flush().expect("a flush"));
    frames
}

pub(super) fn stalled<T>(result: &Result<T, Error>) -> bool {
    matches!(result, Err(Error::Stalled(_)))
}

/// A video-processing run of the whole of one NV12 surface of the
/// driver's at [`SIZE`] into another, as the staged copy back's and zero
/// copy's are ([`copy`]), waited for by its target ([`wait_for`]).
pub(super) struct Run {
    display: Display,
    /// Destroyed before the surfaces it renders into: the fields drop in
    /// their order.
    context: Context,
    from: Surface,
    to: Surface,
}

impl Run {
    pub(super) fn new(display: &Display) -> Self {
        let pool =
            SurfacePool::new(display, SurfaceFormat::Nv12, SIZE.0, SIZE.1, 2).expect("a pool");
        let from = pool.get().expect("a surface");
        let to = pool.get().expect("a surface");
        let context = Config::new(
            display,
            caps::Profile::NONE,
            Entrypoint::VIDEO_PROC,
            &mut [],
        )
        .and_then(|config| Context::new(display, config, SIZE.0, SIZE.1, &mut [to.id()]))
        .expect("a video-processing context");
        Self {
            display: display.clone_ref(),
            context,
            from,
            to,
        }
    }

    /// The run, waited for until the deadline of the frame in hand on this
    /// thread ([`current`]).
    pub(super) fn run(&self) -> Result<(), Error> {
        copy(
            &self.display,
            &self.context,
            self.from.id(),
            self.to.id(),
            None,
        )?;
        wait_for(
            &self.display,
            Work::Surface(self.to.id()),
            current(&self.display),
            "a video-processing run in time",
        )
    }
}

/// A decode still running at its deadline fails as stalled; the frame's
/// pool fails each later wait, and each surface asked of it, at once, and
/// the pool, then the decoder's context, are kept until the GPU is done.
#[test]
#[ignore = "needs a VA-API GPU: run on the VA-API host"]
fn a_decode_still_running_at_its_deadline_is_kept_until_it_is_done() {
    let display = display();
    let (avcc, samples) = h264(&display, 1);
    for _ in 0..TRIES {
        let mut decoder = H264Decoder::new(&display, &avcc).expect("a decoder");
        begin(&mut decoder, &samples);
        let kept_before = KEPT_EVER.load(Ordering::SeqCst);
        let frames = decoded(&mut decoder, &samples);
        WAIT.set(Some(Duration::ZERO));
        let missed = frames[0].sync();
        WAIT.set(None);
        if missed.is_ok() {
            continue;
        }
        assert!(stalled(&missed), "{missed:?}");
        let start = StdRuntime.now();
        let again = frames[0].sync();
        let later = decoder.decode(&samples[0], 1);
        assert!(stalled(&again), "{again:?}");
        assert!(stalled(&later), "{:?}", later.map(|f| f.len()));
        assert!(StdRuntime.elapsed_since(start) < Duration::from_millis(5));
        drop(frames);
        drop(decoder);
        let ever = KEPT_EVER.load(Ordering::SeqCst);
        assert_eq!(ever, kept_before + 2, "the pool and the context");
        assert!(within_5_s(|| kept(&display) == 0), "still kept");
        return;
    }
    panic!("every decode was done before it was asked after, in {TRIES} tries");
}

/// A staged copy back's run still running at its deadline fails as
/// stalled, with no other way back tried, and the pool, with the staging
/// memory the run writes, is kept until the GPU is done.
#[test]
#[ignore = "needs a VA-API GPU: run on the VA-API host"]
fn a_copy_back_still_running_at_its_deadline_is_kept_until_it_is_done() {
    let display = display();
    let (avcc, samples) = h264(&display, 1);
    for _ in 0..TRIES {
        let mut decoder = H264Decoder::new(&display, &avcc).expect("a decoder");
        begin(&mut decoder, &samples);
        let kept_before = KEPT_EVER.load(Ordering::SeqCst);
        let frames = decoded(&mut decoder, &samples);
        // The surface's own wait: the frame's deadline, which its waits
        // share, is left to its read back.
        frames[0].surface.sync().expect("the decode");
        WAIT.set(Some(Duration::ZERO));
        let missed = frames[0].read_back().map(|_| ());
        WAIT.set(None);
        if missed.is_ok() {
            continue;
        }
        assert!(stalled(&missed), "{missed:?}");
        assert!(stalled(&frames[0].read_back().map(|_| ())));
        drop(frames);
        drop(decoder);
        let ever = KEPT_EVER.load(Ordering::SeqCst);
        assert_eq!(ever, kept_before + 2, "the pool and the context");
        assert!(within_5_s(|| kept(&display) == 0), "still kept");
        return;
    }
    panic!("every copy back was done before it was asked after, in {TRIES} tries");
}

/// An encode still running at its deadline fails its picture as stalled;
/// each later frame fails at once, and the session is kept until the GPU
/// is done with the picture's coded buffer. The frame's deadline starts at
/// its upload's map of the input surface, made on the display's call
/// thread ([`ENCODE_WAIT`]); a try whose map is not made by then is not
/// taken, and the next try is made.
#[test]
#[ignore = "needs a VA-API GPU: run on the VA-API host"]
fn an_encode_still_running_at_its_deadline_is_kept_until_it_is_done() {
    let display = display();
    let first = noise(0);
    let second = noise(1);
    for _ in 0..TRIES {
        let mut encoder = encoder(&display);
        let kept_before = KEPT_EVER.load(Ordering::SeqCst);
        WAIT.set(Some(ENCODE_WAIT));
        let coded = encoder.encode(&input(&first, 0));
        WAIT.set(None);
        let Ok(coded) = coded else { continue };
        let missed = match &coded[..] {
            [Ok(_)] => continue,
            [Err(failed)] => failed.error.as_ref(),
            other => panic!("{} results for one frame", other.len()),
        };
        assert!(matches!(missed, Some(Error::Stalled(_))), "{missed:?}");
        let start = StdRuntime.now();
        let later = encoder.encode(&input(&second, 1));
        assert!(stalled(&later), "{:?}", later.map(|c| c.len()));
        assert!(StdRuntime.elapsed_since(start) < Duration::from_millis(5));
        let ever = KEPT_EVER.load(Ordering::SeqCst);
        assert_eq!(ever, kept_before + 1, "the session");
        drop(encoder);
        assert!(within_5_s(|| kept(&display) == 0), "still kept");
        return;
    }
    panic!("every encode was done before it was asked after, in {TRIES} tries");
}
