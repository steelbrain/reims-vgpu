//! The display's calls with the GPU held (`super`, `crate::vaapi::va::calls`):
//! each call measured to wait inside the driver for the GPU, made while a
//! spin of the tests' own holds the ring its work runs on
//! (`crate::vaapi::va::spin`), fails by its deadline, while the GPU is still held,
//! and holds no other call of the display past that call's own deadline.
//! Once the spin is let go, what was kept is let go too, and the work runs
//! again. On a GPU with no uncached memory the spin is timed
//! (`crate::vaapi::va::spin`): it holds the ring for its bound, inside which each
//! test's calls are made, and is let go by waiting for it to end. Where no
//! spin holds the GPU, as its answer says, each test says so and ends.
//!
//! A timed call counts only where the first spin was holding the GPU through it: it begins with
//! more of the hold left than its deadline and slack ([`Spin::left`]), and the spin has not ended
//! as it returns. A host under load can spend the hold's bound before the call is made, or while
//! the call's thread waits to run, and the call then measures nothing: the attempt is made again,
//! from the start, on fresh spins and a fresh display ([`on_a_held_gpu`]), up to [`ATTEMPTS`]
//! times. A call that gets through while the spin still holds the GPU fails the test. Run on
//! the VA-API host's AMD GPUs, one test at a time
//! (`--test-threads=1`: a test sets the process's way back), on the render
//! node `RVG_VAAPI_NODE` names (`/dev/dri/renderD128` where it is unset).

use std::time::Duration;

use crate::core::runtime::{Runtime, StdRuntime};

use super::gpu::{Run, SIZE, begin, decoded, display, encoder, h264, input, noise, stalled};
use super::kept;
use super::tests::{NO_SYNC_SURFACE2, WAIT, within_5_s};
use super::{Deadline, by};
use crate::vaapi::va::display::{Display, Error};
use crate::vaapi::va::h264::H264Decoder;
use crate::vaapi::va::readback::{ReadbackWay, read_back_by};
use crate::vaapi::va::spin::{self, NoSpin, Ring, Spin};
use crate::vaapi::va::surface::{SurfaceFormat, SurfacePool};

/// The wait each test's frames take in place of the device's: short, so
/// that a test holds the GPU well within a spin's bounds.
const WAIT_FOR: Duration = Duration::from_millis(50);

/// How long past [`WAIT_FOR`] a frame may take and still be said to have
/// failed by its deadline: the CPU's work before its first wait, and a
/// thread's wake-up.
const SLACK: Duration = Duration::from_millis(250);

/// How long a call may take and be said to have failed at once.
const AT_ONCE: Duration = Duration::from_millis(5);

/// How many times a test is attempted while its attempts measure nothing (module docs).
const ATTEMPTS: usize = 3;

/// How many compute queues a test fills: twice the RX 7900 XTX's compute
/// rings, so that radeonsi's processing finds none free.
const COMPUTE_SPINS: usize = 16;

fn node() -> String {
    std::env::var("RVG_VAAPI_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into())
}

/// `count` spins on `ring` of the display's GPU, the first begun; `None`
/// where no spin holds the GPU, as its answer says, which is printed.
fn held(ring: Ring, count: usize) -> Option<Vec<Spin>> {
    let node = node();
    let mut spins = Vec::with_capacity(count);
    for _ in 0..count {
        match Spin::new(&node, ring) {
            Ok(spin) => spins.push(spin),
            Err(NoSpin::Unheld(why)) => {
                eprintln!(
                    "no spin holds {node}, as its answer says, so this test does not apply: {why}"
                );
                return None;
            }
            Err(NoSpin::Failed(why)) => panic!("a spin: {why}"),
        }
    }
    for (n, spin) in spins.iter().enumerate() {
        let queued = if n == 0 { spin.hold() } else { spin.queue() };
        queued.expect("a spin queued");
    }
    Some(spins)
}

/// Lets `spins` go, a timed one by waiting for its bound, and fails the
/// test where the kernel's log said amdgpu reset a ring.
fn let_go(spins: &[Spin]) {
    spins.iter().for_each(Spin::release);
    for spin in spins {
        assert!(spin.ended(Duration::from_secs(2)), "a spin not ended");
    }
    assert!(!spin::reset_seen(), "the kernel's log said amdgpu reset");
}

/// `call`, with this thread's waits [`WAIT_FOR`]; what it gave, and how
/// long it took.
fn timed<T>(call: impl FnOnce() -> T) -> (T, Duration) {
    WAIT.set(Some(WAIT_FOR));
    let start = StdRuntime.now();
    let result = call();
    let took = StdRuntime.elapsed_since(start);
    WAIT.set(None);
    (result, took)
}

/// An attempt that measured nothing: why the first spin did not hold the GPU through a timed
/// call (module docs).
#[derive(Debug)]
struct Unheld(String);

/// [`timed`] of `call` (`what`), on the GPU `spins` hold: [`Unheld`] where the first spin had
/// less of its hold left than [`WAIT_FOR`] and [`SLACK`] as the call began, or had ended as it
/// returned.
fn timed_held<T>(
    spins: &[Spin],
    what: &str,
    call: impl FnOnce() -> T,
) -> Result<(T, Duration), Unheld> {
    let left = spins[0].left();
    if left < WAIT_FOR + SLACK {
        return Err(Unheld(format!(
            "{what}: {left:?} of the hold left as it began"
        )));
    }
    let (result, took) = timed(call);
    if spins[0].ended(Duration::ZERO) {
        return Err(Unheld(format!(
            "{what}: the spin ended before the call returned, in {took:?}"
        )));
    }
    Ok((result, took))
}

/// `attempt`, made again while it measures nothing ([`Unheld`]), [`ATTEMPTS`] times at most, the
/// spins of the one before ended first (module docs).
fn on_a_held_gpu(name: &str, attempt: impl Fn() -> Result<(), Unheld>) {
    for n in 1..=ATTEMPTS {
        match attempt() {
            Ok(()) => return,
            Err(Unheld(why)) => {
                eprintln!("{name}: attempt {n} measured nothing: {why}");
                // Its spins were let go as they dropped; a timed one ends by its bound.
                std::thread::sleep(spin::SPUN_AT_MOST);
            }
        }
    }
    panic!("{name}: no attempt of {ATTEMPTS} held the GPU through its timed calls");
}

/// Asserts that a call that failed took no longer than its deadline allows,
/// and returned while `spins` still held the GPU.
fn failed_by_its_deadline(what: &str, took: Duration, spins: &[Spin]) {
    eprintln!("{what}: failed in {took:?}");
    assert!(took < WAIT_FOR + SLACK, "{what}: failed in {took:?}");
    assert!(
        !spins[0].ended(Duration::ZERO),
        "{what}: the spin ended first"
    );
}

/// A display call, made on this thread by its own deadline while the
/// display's call thread is in a call that waits for the GPU, fails by
/// that deadline, from this thread and from another; and two calls of one
/// frame share one deadline, the second failing at once.
fn other_calls_fail_by_their_deadline(display: &Display, spins: &[Spin]) -> Result<(), Unheld> {
    let (call, took) = timed_held(spins, "a call beside it", || display.call(|_| ()))?;
    assert!(stalled(&call), "{call:?}");
    failed_by_its_deadline("a call beside it", took, spins);
    let what = "a call beside it, from another thread";
    // Timed on the other thread, whose waits [`timed`] sets.
    let (call, took) = std::thread::scope(|scope| {
        let other = scope.spawn(|| timed_held(spins, what, || display.call(|_| ())));
        other.join().expect("the other thread")
    })?;
    assert!(stalled(&call), "{call:?}");
    failed_by_its_deadline(what, took, spins);
    let ((first, second), took) = timed_held(spins, "a frame's two calls beside it", || {
        by(&Deadline::default(), || {
            let first = display.call(|_| ());
            let start = StdRuntime.now();
            let second = display.call(|_| ());
            (first, (second, StdRuntime.elapsed_since(start)))
        })
    })?;
    assert!(stalled(&first), "{first:?}");
    failed_by_its_deadline("a frame's first call beside it", took, spins);
    let (second, second_took) = second;
    assert!(stalled(&second), "{second:?}");
    assert!(
        second_took < AT_ONCE,
        "the frame's second call: {second_took:?}"
    );
    Ok(())
}

/// Puts the process's way back as it was, as a test ends.
struct Restore;

impl Drop for Restore {
    fn drop(&mut self) {
        read_back_by(None);
    }
}

/// The way back by `vaGetImage`, which waits for the graphics ring: the
/// read back fails by its deadline, and the next at once; the display's
/// other calls fail by theirs while the copy is in the driver; and the
/// decoder's pool is kept until the GPU is let go and the copy returns.
#[test]
#[ignore = "needs an AMD GPU: run on the VA-API host"]
fn a_get_image_read_back_on_a_held_gpu_fails_by_its_deadline() {
    on_a_held_gpu("a get image read back", || {
        let display = display();
        let (avcc, samples) = h264(&display, 1);
        let _restore = Restore;
        read_back_by(Some(ReadbackWay::GetImage));
        let mut decoder = H264Decoder::new(&display, &avcc).expect("a decoder");
        begin(&mut decoder, &samples);
        let frames = decoded(&mut decoder, &samples);
        // The surface's own wait: the frame's deadline is left to its read back.
        frames[0].surface.sync().expect("the decode");
        let Some(spins) = held(Ring::Graphics, 1) else {
            return Ok(());
        };
        let what = "the read back by vaGetImage";
        let (missed, took) = timed_held(&spins, what, || frames[0].read_back().map(|_| ()))?;
        assert!(stalled(&missed), "{missed:?}");
        failed_by_its_deadline(what, took, &spins);
        let (again, took) = timed(|| frames[0].read_back().map(|_| ()));
        assert!(stalled(&again), "{again:?}");
        assert!(took < AT_ONCE, "the next read back: {took:?}");
        other_calls_fail_by_their_deadline(&display, &spins)?;
        drop(frames);
        drop(decoder);
        assert!(
            kept(&display) > 0,
            "the pool let go while the copy is in the driver"
        );
        let_go(&spins);
        assert!(within_5_s(|| kept(&display) == 0), "still kept");
        display.call(|_| ()).expect("a call, the GPU let go");
        Ok(())
    });
}

/// The copy that conceals a missing reference, by `vaGetImage` then
/// `vaPutImage`: it fails by its deadline, and the next at once.
#[test]
#[ignore = "needs an AMD GPU: run on the VA-API host"]
fn a_concealment_copy_on_a_held_gpu_fails_by_its_deadline() {
    on_a_held_gpu("a concealment copy", || {
        let display = display();
        let pool =
            SurfacePool::new(&display, SurfaceFormat::Nv12, SIZE.0, SIZE.1, 2).expect("a pool");
        let from = pool.get().expect("a surface");
        let to = pool.get().expect("a surface");
        assert!(from.copy_to(&to).expect("a copy, the GPU free"), "no copy");
        let Some(spins) = held(Ring::Graphics, 1) else {
            return Ok(());
        };
        let what = "the concealment copy";
        let (copied, took) = timed_held(&spins, what, || from.copy_to(&to))?;
        assert!(stalled(&copied), "{copied:?}");
        failed_by_its_deadline(what, took, &spins);
        let (again, took) = timed(|| from.copy_to(&to));
        assert!(stalled(&again), "{again:?}");
        assert!(took < AT_ONCE, "the next copy: {took:?}");
        drop((from, to, pool));
        let_go(&spins);
        assert!(within_5_s(|| kept(&display) == 0), "still kept");
        Ok(())
    });
}

/// Puts the thread's driver back to one with `vaSyncSurface2`, as a test ends.
struct WithSyncSurface2;

impl Drop for WithSyncSurface2 {
    fn drop(&mut self) {
        NO_SYNC_SURFACE2.set(false);
    }
}

/// A driver without `vaSyncSurface2`, as a driver's other than radeonsi
/// may be: a video-processing run of NV12 into NV12, as the staged copy
/// back's and zero copy's, is waited for in `vaSyncSurface` on the
/// display's call thread, done when the GPU keeps up, and with the compute
/// rings held failing by its deadline, the GPU still held, and the
/// display's other calls by theirs while the wait is in the driver; once
/// the GPU is let go, the run is done again.
#[test]
#[ignore = "needs an AMD GPU: run on the VA-API host"]
fn a_wait_in_sync_surface_on_a_held_gpu_fails_by_its_deadline() {
    on_a_held_gpu("a wait in vaSyncSurface", || {
        let display = display();
        let run = Run::new(&display);
        let _restore = WithSyncSurface2;
        NO_SYNC_SURFACE2.set(true);
        run.run().expect("a run, the GPU free");
        let Some(spins) = held(Ring::Compute, COMPUTE_SPINS) else {
            return Ok(());
        };
        let what = "the run waited for in vaSyncSurface";
        let (ran, took) = timed_held(&spins, what, || run.run())?;
        assert!(stalled(&ran), "{ran:?}");
        failed_by_its_deadline(what, took, &spins);
        other_calls_fail_by_their_deadline(&display, &spins)?;
        let_go(&spins);
        run.run().expect("a run, the GPU let go");
        Ok(())
    });
}

/// An encode whose upload, by the input surface's derived image, waits
/// for the graphics ring: the frame fails by its deadline, and the next
/// at once.
#[test]
#[ignore = "needs an AMD GPU: run on the VA-API host"]
fn an_encode_on_a_held_gpu_fails_by_its_deadline() {
    on_a_held_gpu("an encode", || {
        let display = display();
        let mut encoder = encoder(&display);
        let first = noise(0);
        let second = noise(1);
        let coded = encoder.encode(&input(&first, 0)).expect("a frame taken");
        assert!(coded.iter().all(Result::is_ok), "an encode, the GPU free");
        let Some(spins) = held(Ring::Graphics, 1) else {
            return Ok(());
        };
        let (coded, took) =
            timed_held(&spins, "the encode", || encoder.encode(&input(&second, 1)))?;
        let missed = match &coded {
            Err(error) => matches!(error, Error::Stalled(_)),
            Ok(coded) => match &coded[..] {
                [Err(failed)] => matches!(failed.error, Some(Error::Stalled(_))),
                other => panic!("{} results for one frame, none failed", other.len()),
            },
        };
        assert!(missed, "not stalled");
        failed_by_its_deadline("the encode", took, &spins);
        let (later, took) = timed(|| encoder.encode(&input(&first, 2)));
        assert!(stalled(&later), "{:?}", later.map(|c| c.len()));
        assert!(took < AT_ONCE, "the next encode: {took:?}");
        drop(encoder);
        let_go(&spins);
        assert!(within_5_s(|| kept(&display) == 0), "still kept");
        Ok(())
    });
}

/// A new decoder's first decode, made just after another decoder was let
/// go with the graphics ring held: the unmaps of the decoder let go have
/// the kernel's next page-table update of the process's GPU VM wait for
/// the spin, radeonsi's session-create command of the new context waits
/// for that update, and its first `vaEndPicture` waits for the command
/// (docs/video/bugs.md, B92, U24). The decode fails by its deadline, while the
/// GPU is still held, and the next at once; the display's other calls fail
/// by theirs while the call is in the driver; and the new decoder's pool
/// and context are kept until the GPU is let go and the call returns,
/// after which a decoder decodes again.
#[test]
#[ignore = "needs an AMD GPU: run on the VA-API host"]
fn a_new_decoders_first_picture_after_a_free_on_a_held_gpu_fails_by_its_deadline() {
    on_a_held_gpu("a new decoder's first picture after a free", || {
        let display = display();
        let (avcc, samples) = h264(&display, 1);
        let mut before = H264Decoder::new(&display, &avcc).expect("a decoder");
        let frames = decoded(&mut before, &samples);
        frames
            .iter()
            .for_each(|frame| frame.sync().expect("a decode, the GPU free"));
        drop(frames);
        let Some(spins) = held(Ring::Graphics, 1) else {
            return Ok(());
        };
        drop(before);
        let mut decoder = H264Decoder::new(&display, &avcc).expect("a decoder");
        let what = "the new decoder's first decode";
        let (first, took) =
            timed_held(&spins, what, || decoder.decode(&samples[0], 0).map(|_| ()))?;
        assert!(stalled(&first), "{first:?} in {took:?}");
        failed_by_its_deadline(what, took, &spins);
        let (again, took) = timed(|| decoder.decode(&samples[0], 1).map(|_| ()));
        assert!(stalled(&again), "{again:?}");
        assert!(took < AT_ONCE, "the next decode: {took:?}");
        other_calls_fail_by_their_deadline(&display, &spins)?;
        drop(decoder);
        assert!(
            kept(&display) > 0,
            "the pool let go while the call is in the driver"
        );
        let_go(&spins);
        assert!(within_5_s(|| kept(&display) == 0), "still kept");
        let mut after = H264Decoder::new(&display, &avcc).expect("a decoder");
        let frames = decoded(&mut after, &samples);
        assert!(!frames.is_empty(), "no frame, the GPU let go");
        frames
            .iter()
            .for_each(|frame| frame.sync().expect("a decode, the GPU let go"));
        Ok(())
    });
}
