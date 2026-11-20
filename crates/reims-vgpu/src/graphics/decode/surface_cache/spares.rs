//! Spares made ahead, off the asker's thread: [`Spares`], which keeps as many spares of an ask as
//! the guest has decodes in flight (`exported`'s surfaces over Direct3D 11 textures, on Windows).
//!
//! An ask takes a spare kept for it, where one is, and hands over to a thread of the spares' own,
//! which nothing waits for, the makes that bring the ask's spares, kept and being made, back up to
//! the depth; an ask that finds none makes its own on its own thread, as the make is not ahead of
//! it, and hands the makes over just the same. The depth is the guest's decodes in flight at its
//! last decode, at least one ([`Spares::set_depth`]): a decode's output is asked for before the
//! decode, and a decode in flight frees its place for the next one's ask as it ends. So a stream
//! of asks finds its spares made while the asker did its other work, as long as a make takes less
//! than the time between two asks, and a run of asks as long as the decodes in flight finds one
//! each.
//!
//! An ask that finds no spare while the spares' thread is making one for it waits for that make
//! and takes what it made, rather than making its own beside it: two makes at once each took two
//! to three times one alone (**measured**, 2026-11-19, Intel UHD 630, a macOS guest's 4K H.264
//! playback with its pictures held: in a pass whose asks outran the spares' thread, 78 makes on
//! the ring took 29.2 ms each and the spares' thread's 162 took 21.0 ms, where the same pass
//! made all of its surfaces on the ring at 9.8 ms each before the spares). The wait is bounded
//! ([`TAKE_WAIT_SLICES`] slices of [`TAKE_WAIT_SLICE`]), past which the ask makes its own.
//!
//! Each make runs under the device's reset gate held shared, so none runs while the device resets,
//! and in a refusal scope: a death there makes no spare, is logged once, and ends the makes handed
//! over with it. At most [`MOST_SPARES`] spares are kept at once, of every ask, the oldest dropped
//! past them, and the depth is no more than that; the spares go with the [`Spares`], once a make
//! running then has returned. A spare goes only by an ask's take, so one kept for an ask that
//! comes no more stays until newer asks push it out, or the [`Spares`] goes.
//!
//! Counted ([`SpareCodes`]): each ask as it uses a spare or finds none, each that waited for a
//! make with its wait's nanoseconds and the longest, each spare the spares' thread makes, and
//! each spare dropped untaken.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use parking_lot::RwLock;
use reims_vgpu_vulkan::core::raise::refusals;
use reims_vgpu_vulkan::core::sync::{Condvar, Mutex};

use super::super::worker::Worker;
use super::Now;

/// The most spares a [`Spares`] keeps at once, of every ask, and the most its depth is: a bound
/// on the memory they hold, past any decodes in flight a player keeps.
pub(super) const MOST_SPARES: usize = 8;

/// A slice of an ask's wait for the make of its spare running (the module doc), woken at once
/// as the make ends.
const TAKE_WAIT_SLICE: std::time::Duration = std::time::Duration::from_millis(10);

/// The most slices an ask waits for the make of its spare running before it makes its own: a
/// bound past any one make measured, so an ask never hangs on a make.
const TAKE_WAIT_SLICES: u32 = 10;

/// The kinds a [`Spares`] counts (`reims_vgpu_vulkan::answer::count_kind`): the asks that used a
/// spare and those that found none, those that waited for a make running, the nanoseconds they
/// waited and the longest wait (a count that rises to it), the spares its thread made, and those
/// dropped untaken.
pub(super) struct SpareCodes {
    pub(super) used: &'static str,
    pub(super) missed: &'static str,
    pub(super) waited: &'static str,
    pub(super) wait_ns: &'static str,
    pub(super) wait_max_ns: &'static str,
    pub(super) made: &'static str,
    pub(super) dropped: &'static str,
}

/// Spares `T` of asks `K`, made ahead, as many of an ask as its depth (the module doc).
pub(super) struct Spares<K, T> {
    kept: Arc<Kept<K, T>>,
    /// Runs the makes, under one key.
    maker: Worker,
    /// The device's reset gate, which each make holds shared.
    gate: Arc<RwLock<()>>,
    codes: &'static SpareCodes,
    /// The spares an ask keeps, kept and being made: from 1 to [`MOST_SPARES`].
    depth: AtomicUsize,
    /// The platform's monotonic clock, which times the asks' waits.
    now: Now,
    /// The longest wait of an ask so far, in nanoseconds, which the `wait_max_ns` count rises to.
    wait_most: AtomicU64,
}

/// What a [`Spares`] and its makes share: the spares and the makes handed over, and whether the
/// [`Spares`] is gone.
struct Kept<K, T> {
    state: Mutex<KeptState<K, T>>,
    closed: AtomicBool,
    /// Woken as a make running ends, for the asks waiting for it.
    ended: Condvar,
}

struct KeptState<K, T> {
    /// The spares, each with its ask, the oldest first.
    ready: VecDeque<(K, T)>,
    /// The ask of each make handed over and not yet done.
    making: Vec<K>,
    /// The ask of the make running, from its gate's take to its spare kept.
    running: Option<K>,
}

impl<K, T> KeptState<K, T>
where
    K: PartialEq,
{
    /// The spares of `ask`, kept and being made.
    fn of(&self, ask: &K) -> usize {
        let ready = self.ready.iter().filter(|(k, _)| k == ask).count();
        ready + self.making.iter().filter(|k| *k == ask).count()
    }

    /// `n` makes of `ask` done.
    fn made(&mut self, ask: &K, n: usize) {
        for _ in 0..n {
            if let Some(at) = self.making.iter().position(|k| k == ask) {
                self.making.swap_remove(at);
            }
        }
    }
}

impl<K, T> Spares<K, T>
where
    K: PartialEq + Clone + Send + 'static,
    T: Send + 'static,
{
    /// Spares made by threads named `name-<i>`, each make under `gate` held shared, counted by
    /// `codes`, the asks' waits timed by `now`, at a depth of one until [`Self::set_depth`].
    pub(super) fn new(
        name: &'static str,
        gate: Arc<RwLock<()>>,
        codes: &'static SpareCodes,
        now: Now,
    ) -> Self {
        Spares {
            kept: Arc::new(Kept {
                state: Mutex::new(KeptState {
                    ready: VecDeque::new(),
                    making: Vec::new(),
                    running: None,
                }),
                closed: AtomicBool::new(false),
                ended: Condvar::new(),
            }),
            maker: Worker::new(name),
            gate,
            codes,
            depth: AtomicUsize::new(1),
            now,
            wait_most: AtomicU64::new(0),
        }
    }

    /// Sets the depth to `in_flight`, the guest's decodes in flight at its last decode, at least
    /// one and at most [`MOST_SPARES`]. Spares kept past a lower depth stay, for the asks to take.
    pub(super) fn set_depth(&self, in_flight: u64) {
        let depth = usize::try_from(in_flight).map_or(MOST_SPARES, |n| n.clamp(1, MOST_SPARES));
        self.depth.store(depth, Ordering::Relaxed);
    }

    /// A spare kept for `ask`, the oldest, no longer kept; where none is and the make running is
    /// `ask`'s, the spare it keeps, waited for (the module doc); `None` where none is, or the
    /// make made none or ran past the wait. Counted either way, and the wait.
    pub(super) fn take(&self, ask: &K) -> Option<T> {
        let mut state = self.kept.state.lock();
        let (mut waited, mut slices) = (None, 0);
        let taken = loop {
            if let Some(at) = state.ready.iter().position(|(k, _)| k == ask) {
                break state.ready.remove(at).map(|(_, spare)| spare);
            }
            if state.running.as_ref() != Some(ask) || slices == TAKE_WAIT_SLICES {
                break None;
            }
            waited.get_or_insert_with(|| (self.now)());
            if (self.kept.ended)
                .wait_for(&mut state, TAKE_WAIT_SLICE)
                .timed_out()
            {
                slices += 1;
            }
        };
        drop(state);
        if let Some(begun) = waited {
            let took = (self.now)().saturating_duration_since(begun);
            let took = u64::try_from(took.as_nanos()).unwrap_or(u64::MAX);
            let codes = self.codes;
            reims_vgpu_vulkan::answer::count_kind(codes.waited);
            reims_vgpu_vulkan::answer::count_kind_by(codes.wait_ns, took);
            let was = self.wait_most.fetch_max(took, Ordering::Relaxed);
            reims_vgpu_vulkan::answer::count_kind_by(codes.wait_max_ns, took.saturating_sub(was));
        }
        let code = match taken {
            Some(_) => self.codes.used,
            None => self.codes.missed,
        };
        reims_vgpu_vulkan::answer::count_kind(code);
        taken
    }

    /// Hands over to the spares' thread the makes by `make` that bring `ask`'s spares, kept and
    /// being made, up to the depth, made one at a time and each kept once made, the oldest spare
    /// dropped past [`MOST_SPARES`]. `make`'s `None` makes no spare, and ends those makes.
    /// Nothing waits for them.
    pub(super) fn want(&self, ask: K, make: impl FnMut() -> Option<T> + Send + 'static) {
        let depth = self.depth.load(Ordering::Relaxed);
        let need = {
            let mut state = self.kept.state.lock();
            let need = depth.saturating_sub(state.of(&ask));
            state.making.extend(std::iter::repeat_n(ask.clone(), need));
            need
        };
        if need == 0 {
            return;
        }
        let kept = Arc::clone(&self.kept);
        let gate = Arc::clone(&self.gate);
        let codes = self.codes;
        (self.maker).submit(
            0,
            Box::new(move || kept.make(&ask, need, make, &gate, codes)),
        );
    }

    /// Waits until no make is handed over or runs (tests only).
    #[cfg(test)]
    fn settle(&self) {
        self.maker.flush();
    }

    /// The asks of the spares kept, the oldest first (tests only).
    #[cfg(test)]
    fn kept(&self) -> Vec<K> {
        let state = self.kept.state.lock();
        state.ready.iter().map(|(k, _)| k.clone()).collect()
    }
}

impl<K, T> Drop for Spares<K, T> {
    /// Ends the makes: those handed over and not begun make nothing, and the spares go once one
    /// running has returned, each counted as dropped.
    fn drop(&mut self) {
        self.kept.closed.store(true, Ordering::Release);
        self.maker.flush();
        let ready = std::mem::take(&mut self.kept.state.lock().ready);
        for _ in &ready {
            reims_vgpu_vulkan::answer::count_kind(self.codes.dropped);
        }
        let_go_of(ready);
    }
}

impl<K: PartialEq + Clone, T> Kept<K, T> {
    /// The makes, on the spares' thread: `need` spares for `ask` by `make`, one at a time, unless
    /// the [`Spares`] is gone, each under `gate` held shared and in a refusal scope, the running
    /// one's ask kept for the asks to wait for, and each kept and counted as it is made, the asks
    /// waiting woken, the oldest spare dropped past [`MOST_SPARES`] with the lock let go of. A
    /// make that makes none ends the rest.
    fn make(
        &self,
        ask: &K,
        need: usize,
        mut make: impl FnMut() -> Option<T>,
        gate: &RwLock<()>,
        codes: &SpareCodes,
    ) {
        for left in (0..need).rev() {
            let made = {
                let _shared = gate.read();
                if self.closed.load(Ordering::Acquire) {
                    None
                } else {
                    self.state.lock().running = Some(ask.clone());
                    match refusals::refusing(&mut make) {
                        Ok(made) => made,
                        Err(refusal) => {
                            make_refused(&refusal);
                            None
                        }
                    }
                }
            };
            if made.is_some() {
                reims_vgpu_vulkan::answer::count_kind(codes.made);
            }
            let mut state = self.state.lock();
            state.running = None;
            let ended = made.is_none();
            state.made(ask, if ended { left + 1 } else { 1 });
            let gone = match made {
                Some(made) if !self.closed.load(Ordering::Acquire) => {
                    state.ready.push_back((ask.clone(), made));
                    if state.ready.len() > MOST_SPARES {
                        state.ready.pop_front()
                    } else {
                        None
                    }
                }
                made => made.map(|made| (ask.clone(), made)),
            };
            drop(state);
            self.ended.notify_all();
            if gone.is_some() {
                reims_vgpu_vulkan::answer::count_kind(codes.dropped);
            }
            let_go_of(gone);
            if ended {
                return;
            }
        }
    }
}

/// Lets go of `gone`, spares no ask took, with no lock held, a refusal logged.
fn let_go_of<G>(gone: G) {
    if let Err(refusal) = refusals::refusing(move || drop(gone)) {
        log::warn!(
            target: "reims_vgpu::graphics", kind:% = refusal;
            "a spare made ahead and not taken was let go of, and a release was refused: {refusal}"
        );
    }
}

/// A make on the spares' thread that died, `refusal`: logged the first time alone
/// (docs/conventions.md); the asks then make their own until a make succeeds.
fn make_refused(refusal: &refusals::Refusal) {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        log::warn!(
            target: "reims_vgpu::graphics", kind:% = refusal;
            "a spare made ahead was refused: {refusal}; an ask that finds no spare makes its own"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static CODES: SpareCodes = SpareCodes {
        used: "test-spares-used",
        missed: "test-spares-missed",
        waited: "test-spares-waited",
        wait_ns: "test-spares-wait-ns",
        wait_max_ns: "test-spares-wait-max-ns",
        made: "test-spares-made",
        dropped: "test-spares-dropped",
    };

    /// A spare that counts its drops in `gone`.
    struct Counted(u32, Arc<AtomicUsize>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.1.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// A clock that reads zero: the tests count waits, not their time.
    fn zero() -> Now {
        Arc::new(|| reims_vgpu_vulkan::core::platform::os::clock::MonoTime::from_nanos(0))
    }

    fn spares() -> Spares<u32, Counted> {
        Spares::new("test-spares", Arc::default(), &CODES, zero())
    }

    /// The host's monotonic clock.
    fn host() -> Now {
        use reims_vgpu_vulkan::core::platform::os::clock::Clock as _;

        let clock = reims_vgpu_vulkan::os::OsClock::new();
        Arc::new(move || clock.now_mono())
    }

    /// [`CODES`] with wait codes of the test's own, `waited` and two named after it, which it
    /// reads alone however the tests share a process.
    fn waited_as(waited: &'static str) -> &'static SpareCodes {
        let named = |end: &str| -> &'static str { format!("{waited}-{end}").leak() };
        Box::leak(Box::new(SpareCodes {
            waited,
            wait_ns: named("ns"),
            wait_max_ns: named("max-ns"),
            ..CODES
        }))
    }

    /// The count of `code`: asks that waited, or nanoseconds.
    fn waits(code: &'static str) -> u64 {
        (reims_vgpu_vulkan::answer::counted_kinds().into_iter())
            .find(|(c, _)| *c == code)
            .map_or(0, |(_, n)| n)
    }

    /// Spares counted by `codes`, the waits timed by the host's clock, whose one make, of ask 1,
    /// has begun and runs until the sender sends: the make running, as the asks see it.
    fn making(codes: &'static SpareCodes) -> (Spares<u32, u32>, std::sync::mpsc::Sender<()>) {
        let spares = Spares::new("test-spares", Arc::default(), codes, host());
        let (begun, has_begun) = std::sync::mpsc::channel();
        let (go, gone_on) = std::sync::mpsc::channel::<()>();
        spares.want(1, move || {
            begun.send(()).ok()?;
            gone_on.recv().ok()?;
            Some(7)
        });
        has_begun.recv().expect("the make begun");
        (spares, go)
    }

    /// A make of `ask`'s spares that counts its calls in `makes` and its spares' drops in `gone`.
    fn counting(
        ask: u32,
        makes: &Arc<AtomicUsize>,
        gone: &Arc<AtomicUsize>,
    ) -> impl FnMut() -> Option<Counted> + Send + 'static {
        let (makes, gone) = (Arc::clone(makes), Arc::clone(gone));
        move || {
            makes.fetch_add(1, Ordering::SeqCst);
            Some(Counted(ask, Arc::clone(&gone)))
        }
    }

    /// An ask finds no spare before its first make; the make it hands over is taken by the next
    /// ask of its kind, and none by an ask of another.
    #[test]
    fn an_ask_takes_the_spare_its_last_ask_made() {
        let (makes, gone) = (Arc::default(), Arc::new(AtomicUsize::new(0)));
        let spares = spares();
        assert!(spares.take(&1).is_none());
        spares.want(1, counting(1, &makes, &gone));
        spares.settle();
        assert!(spares.take(&2).is_none());
        assert_eq!(spares.take(&1).map(|c| c.0), Some(1));
        assert!(spares.take(&1).is_none());
        assert_eq!(gone.load(Ordering::SeqCst), 1);
    }

    /// An ask's spares, kept and being made, come up to the depth and no further: the asks
    /// that find them there hand no make over.
    #[test]
    fn an_asks_spares_come_up_to_the_depth() {
        let (makes, gone) = (Arc::new(AtomicUsize::new(0)), Arc::default());
        let spares = spares();
        spares.set_depth(3);
        for _ in 0..4 {
            spares.want(1, counting(1, &makes, &gone));
        }
        spares.settle();
        assert_eq!(makes.load(Ordering::SeqCst), 3);
        assert_eq!(spares.kept(), vec![1, 1, 1]);
        assert!(spares.take(&1).is_some());
        spares.want(1, counting(1, &makes, &gone));
        spares.settle();
        assert_eq!(makes.load(Ordering::SeqCst), 4);
        assert_eq!(spares.kept(), vec![1, 1, 1]);
    }

    /// The depth is the decodes in flight, at least one and at most [`MOST_SPARES`]; spares
    /// kept past a lower depth stay for the asks, and no make is handed over until they are
    /// taken below it.
    #[test]
    fn the_depth_is_the_decodes_in_flight_within_its_bounds() {
        let (makes, gone) = (Arc::new(AtomicUsize::new(0)), Arc::default());
        let spares = spares();
        spares.set_depth(0);
        spares.want(1, counting(1, &makes, &gone));
        spares.settle();
        assert_eq!(makes.load(Ordering::SeqCst), 1);
        spares.set_depth(u64::MAX);
        spares.want(1, counting(1, &makes, &gone));
        spares.settle();
        assert_eq!(makes.load(Ordering::SeqCst), MOST_SPARES);
        spares.set_depth(2);
        assert!(spares.take(&1).is_some());
        spares.want(1, counting(1, &makes, &gone));
        spares.settle();
        assert_eq!(makes.load(Ordering::SeqCst), MOST_SPARES);
        assert_eq!(spares.kept().len(), MOST_SPARES - 1);
    }

    /// Past [`MOST_SPARES`] spares, of every ask, the oldest is dropped; the rest go with the
    /// [`Spares`].
    #[test]
    fn the_oldest_spare_goes_past_the_most() {
        let (makes, gone) = (Arc::default(), Arc::new(AtomicUsize::new(0)));
        let spares = spares();
        let asks = u32::try_from(MOST_SPARES).unwrap() + 1;
        for ask in 0..asks {
            spares.want(ask, counting(ask, &makes, &gone));
            spares.settle();
        }
        assert_eq!(gone.load(Ordering::SeqCst), 1);
        assert_eq!(spares.kept(), (1..asks).collect::<Vec<_>>());
        drop(spares);
        assert_eq!(gone.load(Ordering::SeqCst), MOST_SPARES + 1);
    }

    /// A make that makes nothing keeps nothing and ends the makes handed over with it, and a
    /// later ask hands them over again.
    #[test]
    fn a_failed_make_ends_its_makes() {
        let (makes, gone) = (Arc::new(AtomicUsize::new(0)), Arc::default());
        let spares = spares();
        spares.set_depth(3);
        let tried = Arc::new(AtomicUsize::new(0));
        let t = Arc::clone(&tried);
        spares.want(1, move || {
            t.fetch_add(1, Ordering::SeqCst);
            None
        });
        spares.settle();
        assert_eq!(tried.load(Ordering::SeqCst), 1);
        assert!(spares.kept().is_empty());
        spares.want(1, counting(1, &makes, &gone));
        spares.settle();
        assert_eq!(makes.load(Ordering::SeqCst), 3);
        assert_eq!(spares.kept(), vec![1, 1, 1]);
    }

    /// An ask that finds no spare while the spares' thread makes its spare waits for that make
    /// and takes the spare it made, rather than making its own beside it.
    #[test]
    fn a_take_waits_for_its_asks_make_running() {
        let codes = waited_as("test-spares-waited-for-the-make");
        let (spares, go) = making(codes);
        std::thread::scope(|s| {
            let taker = s.spawn(|| spares.take(&1));
            // The make ends once the taker waits for it: a wake its wait loops past.
            while spares.kept.ended.notify_all() == 0 && !taker.is_finished() {
                std::thread::yield_now();
            }
            go.send(()).expect("the make running");
            assert_eq!(taker.join().expect("the taker"), Some(7));
        });
        assert_eq!(waits(codes.waited), 1);
        spares.settle();
        assert!(spares.kept().is_empty());
    }

    /// An ask waits only for a make of its own kind: one of another finds none at once.
    #[test]
    fn a_take_waits_for_no_other_asks_make() {
        let codes = waited_as("test-spares-waited-for-another");
        let (spares, go) = making(codes);
        assert!(spares.take(&2).is_none());
        assert_eq!(waits(codes.waited), 0);
        go.send(()).expect("the make running");
        spares.settle();
        assert_eq!(spares.kept(), vec![1]);
    }

    /// An ask waits for its make no longer than the bound, then finds none, its wait counted as
    /// the longest; the make's spare is kept once made.
    #[test]
    fn a_take_waits_no_longer_than_its_bound() {
        let codes = waited_as("test-spares-waited-past-the-bound");
        let (spares, go) = making(codes);
        assert!(spares.take(&1).is_none());
        assert_eq!(waits(codes.waited), 1);
        let bound = (TAKE_WAIT_SLICE * TAKE_WAIT_SLICES).as_nanos();
        assert!(u128::from(waits(codes.wait_ns)) >= bound);
        assert_eq!(waits(codes.wait_max_ns), waits(codes.wait_ns));
        go.send(()).expect("the make running");
        spares.settle();
        assert_eq!(spares.kept(), vec![1]);
    }

    /// No make runs while the device resets: a make handed over waits for the reset gate.
    #[test]
    fn a_make_waits_for_the_reset() {
        let gate: Arc<RwLock<()>> = Arc::default();
        let spares: Spares<u32, u32> =
            Spares::new("test-spares", Arc::clone(&gate), &CODES, zero());
        let resetting = gate.write();
        let made = Arc::new(AtomicBool::new(false));
        let m = Arc::clone(&made);
        spares.want(1, move || {
            m.store(true, Ordering::SeqCst);
            Some(1)
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(!made.load(Ordering::SeqCst));
        drop(resetting);
        spares.settle();
        assert!(made.load(Ordering::SeqCst));
        assert_eq!(spares.kept(), vec![1]);
    }
}
