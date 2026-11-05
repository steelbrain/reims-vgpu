//! Ring-0 work carried out off ring 0's threads, answered later, while the later messages that
//! name its object wait for it: a [`Worker`], a pool of threads that runs each key's jobs one at a
//! time in the order they are handed over, and the [`Unsettled`] keys (a block's offset, an
//! object's id) whose job is not yet done.
//!
//! The ring-0 thread that carries the message out (its block's or its context's lane) begins the
//! key's [`Settle`] in that lane's ring order and hands the job over; the job carries the work out,
//! posts the message's answer, then drops the settle, so a message that waited for the key is
//! answered after it. A message that names the key, on ring 0 or a queue's ring, waits for it first
//! ([`Unsettled::wait`]), holding no lock the job takes; a thread that waits for nothing (a
//! doorbell's) asks instead ([`Unsettled::try_any`]) and leaves the message to a thread that may. A
//! job takes `GraphicsControl`'s reset gate shared, recursively, as a queue that holds it may be
//! waiting for the key while the reset waits for the gate, and carries nothing out where the device
//! was reset since its begin (the caller's count of resets, kept in the settle).
//!
//! **Keys.** A worker runs the jobs handed over under one key one at a time, in the order they were
//! handed over, and those of different keys at once, on at most [`POOL_THREADS`] threads named
//! `<name>-<i>`. A thread is started when a key's job finds none waiting, and ends once it has
//! waited [`IDLE`] with nothing to do; a key's job wakes one waiting thread, never more. A key
//! whose job finds every thread busy waits for the next free one, behind the keys that were
//! waiting before it, and is counted (a cap hit, logged at debug); a thread runs a job of each
//! waiting key in turn. Where no thread can be started, the job runs where it is handed over.
//! - The surface worker's (`reims-vgpu.graphics-surfaces-<i>`) key is a block's offset: a
//!   `SURFACE_PAGES` and a `BUFFER_OVER_PAGES` handed over (a pages block's first, or one behind
//!   its block's or its context's still being made) are their block's, as are a released block's
//!   views and its device cache's let-go; an over-view buffer's drop at its destroy is its
//!   block's, or its id's where it has none. A view's mapping, let go of there wherever the
//!   view's last hold goes, is under a key of its own (`surface_pages::TEARDOWN`).
//! - The ICB worker's (`reims-vgpu.icb-create-<i>`) key is the buffer's context: a process's
//!   `ICB_CREATE`s are made in its ring order, other processes' beside them.
//! - The closes' worker's (`reims-vgpu.graphics-closes-<i>`) key is the closed context: a
//!   `CONTEXT_CLOSE` lets go of its objects there, and is answered after them, other processes'
//!   closes beside it.
//!
//! **No job waits for another key's.** A job waits only for jobs of its own key, which were handed
//! over before it and are done when it runs: a `BUFFER_OVER_PAGES` waits for its block's
//! `SURFACE_PAGES` ([`Unsettled::wait`]), and a `SURFACE_PAGES` waits for nothing
//! ([`Worker::submit_waitless`]). So jobs waiting for a thread never wait for each other, and the
//! cap cannot deadlock. A debug build checks it where a wait would block.
//!
//! What the keys keep of a single thread's order:
//! - A doorbell's thread asks as before ([`Unsettled::try_any`]): the unsettled keys and their
//!   counts are those of a single thread.
//! - A pages block's jobs go in ring order: its `SURFACE_PAGES` and its `BUFFER_OVER_PAGES` share
//!   its key, so the buffer is made after the views, and a later `SURFACE_PAGES` of the block waits
//!   for its buffers before it is handed over.
//! - A message of another context that names a shared block waits for the block on its own ring-0
//!   lane or queue ring, never on a worker's thread.
//! - Each job still takes the reset gate and checks its begin's count of resets, or its poster's
//!   epoch, itself, so where the device was reset since, it carries nothing out, on whichever
//!   thread runs it.
//!
//! Two `BUFFER_OVER_PAGES` of one context on different blocks may be made in either order: no
//! answer orders them, their objects and GPU addresses are the guest's and distinct, and every
//! later message of the context waits until both are made (the context's [`Unsettled`] count).

use std::cell::Cell;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use reims_vgpu_core::platform::os::MonoTime;
use reims_vgpu_core::sync::{Condvar, Mutex};

/// The most threads a [`Worker`] runs.
pub(crate) const POOL_THREADS: usize = 4;

/// How long a [`Worker`]'s thread waits for a job before it ends.
pub(crate) const IDLE: Duration = Duration::from_secs(1);

/// A job of a [`Worker`]'s.
pub(crate) type Job = Box<dyn FnOnce() + Send>;

thread_local! {
    /// The key of the worker job this thread runs, and whether the job may wait for it; `None`
    /// outside a job (the module doc, "No job waits for another key's").
    static TURN: Cell<Option<(u64, bool)>> = const { Cell::new(None) };
}

/// A job's turn on its thread ([`TURN`]); its drop gives the thread back the turn it ran before.
struct Turn(Option<(u64, bool)>);

impl Turn {
    fn enter(key: u64, may_wait: bool) -> Turn {
        Turn(TURN.replace(Some((key, may_wait))))
    }
}

impl Drop for Turn {
    fn drop(&mut self) {
        TURN.set(self.0);
    }
}

/// The library's threads named `<name>-<i>`, at most [`POOL_THREADS`] of them, that run the jobs
/// handed over under each key one at a time, in the order handed over, inside a library entry (the
/// module doc, "Keys"). Its drop ends the threads once the jobs handed over are done, without
/// waiting for them.
pub(crate) struct Worker {
    shared: Arc<Shared>,
}

/// What a worker's threads share.
struct Shared {
    name: &'static str,
    /// How long a thread waits for a job before it ends: [`IDLE`], or a test's.
    idle: Duration,
    state: Mutex<State>,
    /// Signalled, for one thread, as a key's job waits for one.
    ready: Condvar,
    /// The keys whose job found every thread busy.
    cap_hits: AtomicU64,
    /// A thread could not be started (logged once).
    refused: AtomicBool,
    /// Signalled as a key's last job ends ([`Worker::flush`]).
    done: Condvar,
    /// Holds the threads before each job while shut (tests only).
    #[cfg(test)]
    gate: Gate,
}

/// A worker's keys and threads.
#[derive(Default)]
struct State {
    /// Each key with a job waiting or running.
    keys: HashMap<u64, Jobs>,
    /// The keys with a job waiting and none running, in the order they take their turns.
    ready: VecDeque<u64>,
    /// Which of the threads' names are taken, a thread's for its life.
    named: [bool; POOL_THREADS],
    /// The threads waiting for a key's job, and the wakes signalled to them not yet taken.
    idle: usize,
    woken: usize,
    /// The worker is gone: its threads end once no job waits.
    closed: bool,
}

/// A key's jobs: those not yet taken, in the order handed over, and whether one runs.
#[derive(Default)]
struct Jobs {
    waiting: VecDeque<Job>,
    running: bool,
}

impl State {
    /// Takes `key`'s next job, and counts it running.
    fn take(&mut self, key: u64) -> Option<Job> {
        let jobs = self.keys.get_mut(&key)?;
        let job = jobs.waiting.pop_front()?;
        jobs.running = true;
        Some(job)
    }

    /// A job of `key` ended: the key waits for its next turn where it has a job left, and is
    /// forgotten where it has none. Whether it has one.
    fn ended(&mut self, key: u64) -> bool {
        let Some(jobs) = self.keys.get_mut(&key) else {
            return false;
        };
        jobs.running = false;
        if jobs.waiting.is_empty() {
            self.keys.remove(&key);
            return false;
        }
        true
    }
}

impl Worker {
    /// A worker whose threads are named `name-<i>`.
    pub(crate) fn new(name: &'static str) -> Self {
        Worker::with_idle(name, IDLE)
    }

    /// [`Self::new`], whose threads end after `idle` with nothing to do.
    fn with_idle(name: &'static str, idle: Duration) -> Self {
        Worker {
            shared: Arc::new(Shared {
                name,
                idle,
                state: Mutex::new(State::default()),
                ready: Condvar::new(),
                cap_hits: AtomicU64::new(0),
                refused: AtomicBool::new(false),
                done: Condvar::new(),
                #[cfg(test)]
                gate: Gate::default(),
            }),
        }
    }

    /// Hands `job` over under `key`: it runs after the jobs handed over under `key` before it.
    pub(crate) fn submit(&self, key: u64, job: Job) {
        let shared = &self.shared;
        let mut state = shared.state.lock();
        let jobs = state.keys.entry(key).or_default();
        jobs.waiting.push_back(job);
        if jobs.running || jobs.waiting.len() > 1 {
            // The key's turn is to come.
            return;
        }
        state.ready.push_back(key);
        if state.idle > state.woken {
            state.woken += 1;
            drop(state);
            shared.ready.notify_one();
            return;
        }
        let Some(index) = state.named.iter().position(|taken| !taken) else {
            drop(state);
            let hits = shared.cap_hits.fetch_add(1, Ordering::Relaxed) + 1;
            log::debug!(
                target: "reims_vgpu::graphics",
                "{}: a job of key {key:#x} waits for one of the {POOL_THREADS} threads, all busy \
                 ({hits} such waits)",
                shared.name
            );
            return;
        };
        state.named[index] = true;
        drop(state);
        let spawned = {
            let shared = Arc::clone(shared);
            std::thread::Builder::new()
                .name(format!("{}-{index}", shared.name))
                .spawn(move || work(&shared, index))
        };
        let Err(e) = spawned else {
            return;
        };
        if !shared.refused.swap(true, Ordering::Relaxed) {
            log::warn!(
                target: "reims_vgpu::graphics",
                "{}-{index} could not be started ({e}): a job no thread takes runs where it is \
                 handed over",
                shared.name
            );
        }
        let mut state = shared.state.lock();
        state.named[index] = false;
        // Unless a thread took it meanwhile.
        let Some(at) = state.ready.iter().position(|&k| k == key) else {
            return;
        };
        state.ready.remove(at);
        let mut job = state.take(key);
        drop(state);
        while let Some(next) = job {
            {
                let _turn = Turn::enter(key, true);
                crate::entry::run(|_| next());
            }
            let mut state = shared.state.lock();
            job = if state.ended(key) {
                state.take(key)
            } else {
                None
            };
            if job.is_none() {
                shared.done.notify_all();
            }
        }
    }

    /// [`Self::submit`], for a job that waits for nothing (the module doc, "No job waits for
    /// another key's"), as a debug build checks.
    pub(crate) fn submit_waitless(&self, key: u64, job: Job) {
        self.submit(
            key,
            Box::new(move || {
                let _turn = Turn::enter(key, false);
                job();
            }),
        );
    }

    /// Lets `value` go on a thread, under `key`, in a refusal scope, as object `id`'s release
    /// ([`super::drop_refused`]).
    pub(crate) fn drop_later<T: Send + 'static>(&self, key: u64, id: u64, value: T) {
        self.submit(key, Box::new(move || super::drop_refused(id, value)));
    }

    /// The jobs that found every thread busy and waited for one (tests only).
    #[cfg(test)]
    pub(crate) fn cap_hits(&self) -> u64 {
        self.shared.cap_hits.load(Ordering::Relaxed)
    }

    /// The threads running (tests only).
    #[cfg(test)]
    pub(crate) fn threads(&self) -> usize {
        self.shared
            .state
            .lock()
            .named
            .iter()
            .filter(|&&n| n)
            .count()
    }

    /// Shuts the gate: no thread starts a job until it opens (tests only).
    #[cfg(test)]
    pub(crate) fn hold(&self) {
        *self.shared.gate.shut.lock() = true;
    }

    /// Opens the gate (tests only).
    #[cfg(test)]
    pub(crate) fn release(&self) {
        *self.shared.gate.shut.lock() = false;
        self.shared.gate.opened.notify_all();
    }

    /// Waits until no job of any key waits or runs: the device's reset, for the closes' worker's
    /// jobs before it ([`super::GraphicsDecoder::closes_let_go`]).
    pub(crate) fn flush(&self) {
        let mut state = self.shared.state.lock();
        while !state.keys.is_empty() {
            self.shared.done.wait(&mut state);
        }
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        self.shared.state.lock().closed = true;
        self.shared.ready.notify_all();
    }
}

/// A worker's thread `index`: runs a job of each waiting key in turn, and ends once it has waited
/// the worker's idle time with nothing to do, or the worker is gone and no job waits.
fn work(shared: &Shared, index: usize) {
    let mut state = shared.state.lock();
    loop {
        if let Some(key) = state.ready.pop_front() {
            let Some(job) = state.take(key) else {
                continue;
            };
            drop(state);
            #[cfg(test)]
            shared.gate.pass();
            {
                let _turn = Turn::enter(key, true);
                crate::entry::run(|_| job());
            }
            state = shared.state.lock();
            if state.ended(key) {
                state.ready.push_back(key);
            } else {
                shared.done.notify_all();
            }
            continue;
        }
        if state.closed {
            break;
        }
        state.idle += 1;
        let timed_out = shared.ready.wait_for(&mut state, shared.idle).timed_out();
        state.idle -= 1;
        state.woken = state.woken.saturating_sub(1);
        if timed_out && state.ready.is_empty() {
            break;
        }
    }
    state.named[index] = false;
}

/// A worker's gate (tests only): shut, no thread starts a job.
#[cfg(test)]
#[derive(Default)]
struct Gate {
    shut: Mutex<bool>,
    opened: Condvar,
}

#[cfg(test)]
impl Gate {
    fn pass(&self) {
        let mut shut = self.shut.lock();
        while *shut {
            self.opened.wait(&mut shut);
        }
    }
}

/// The keys whose job was handed to a worker and is not yet done, each with how many of them.
#[derive(Default)]
pub(crate) struct Unsettled {
    keys: Mutex<HashMap<u64, u32>>,
    /// Signalled as each settles.
    settled: Condvar,
    /// How many there are, read without the lock where there are none.
    count: AtomicUsize,
}

impl Unsettled {
    /// `key` is unsettled until the returned [`Settle`] drops; `resets` is the caller's count of
    /// the device's resets, and `begun` the time.
    pub(crate) fn begin(self: &Arc<Self>, key: u64, resets: u64, begun: MonoTime) -> Settle {
        *self.keys.lock().entry(key).or_insert(0) += 1;
        self.count.fetch_add(1, Ordering::AcqRel);
        Settle {
            unsettled: Arc::clone(self),
            key,
            resets,
            begun,
        }
    }

    /// Waits for every job of `key` begun before now to be done. Called with no lock held that a
    /// job takes; on a worker's thread, only by a job of `key` that may wait (checked in a debug
    /// build, the module doc's "No job waits for another key's").
    pub(crate) fn wait(&self, key: u64) {
        if self.count.load(Ordering::Acquire) == 0 {
            return;
        }
        let mut keys = self.keys.lock();
        if keys.contains_key(&key)
            && let Some((own, may_wait)) = TURN.get()
        {
            debug_assert!(
                may_wait,
                "a job of key {own:#x} that waits for nothing waits for key {key:#x}"
            );
            debug_assert!(
                own == key,
                "a job of key {own:#x} waits for key {key:#x}, another key's"
            );
        }
        while keys.contains_key(&key) {
            self.settled.wait(&mut keys);
        }
    }
    /// Whether any of `keys` is unsettled.
    pub(crate) fn any(&self, mut keys: impl Iterator<Item = u64>) -> bool {
        if self.count.load(Ordering::Acquire) == 0 {
            return false;
        }
        let unsettled = self.keys.lock();
        keys.any(|k| unsettled.contains_key(&k))
    }

    /// [`Self::any`], for a thread that waits for nothing and allocates nothing (a doorbell's, the
    /// guest's vCPU): `None` where the keys are held elsewhere, their lock only tried.
    pub(crate) fn try_any(&self, mut keys: impl Iterator<Item = u64>) -> Option<bool> {
        if self.count.load(Ordering::Acquire) == 0 {
            return Some(false);
        }
        let unsettled = self.keys.try_lock()?;
        Some(keys.any(|k| unsettled.contains_key(&k)))
    }
}

/// A job begun and not yet done ([`Unsettled::begin`]); its drop settles its key.
pub(crate) struct Settle {
    unsettled: Arc<Unsettled>,
    key: u64,
    /// The caller's count of the device's resets at the begin.
    pub(crate) resets: u64,
    /// When it was begun.
    pub(crate) begun: MonoTime,
}

impl Drop for Settle {
    fn drop(&mut self) {
        let mut keys = self.unsettled.keys.lock();
        if let Some(n) = keys.get_mut(&self.key) {
            *n -= 1;
            if *n == 0 {
                keys.remove(&self.key);
            }
        }
        self.unsettled.count.fetch_sub(1, Ordering::AcqRel);
        drop(keys);
        self.unsettled.settled.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use super::*;

    /// A worker runs a key's jobs in the order they are handed over, on its own thread, and holds
    /// them while its gate is shut
    #[test]
    fn a_worker_runs_its_jobs_in_order_on_its_thread() {
        let worker = Worker::new("reims-vgpu.test-worker");
        let seen = Arc::new(Mutex::new(Vec::new()));
        worker.hold();
        for i in 0..3 {
            let seen = Arc::clone(&seen);
            worker.submit(
                7,
                Box::new(move || {
                    let name = std::thread::current().name().map(str::to_owned);
                    seen.lock().push((i, name));
                }),
            );
        }
        std::thread::sleep(Duration::from_millis(20));
        assert!(seen.lock().is_empty(), "held");
        worker.release();
        worker.flush();
        let name = Some("reims-vgpu.test-worker-0".to_owned());
        assert_eq!(
            *seen.lock(),
            [(0, name.clone()), (1, name.clone()), (2, name)]
        );
    }

    /// A job that ends when `go` is sent to, or dropped, after it tells `started` it began.
    fn blocking(started: &mpsc::Sender<u64>, key: u64) -> (Job, mpsc::Sender<()>) {
        let (go, wait) = mpsc::channel::<()>();
        let started = started.clone();
        let job = Box::new(move || {
            let _ = started.send(key);
            let _ = wait.recv();
        });
        (job, go)
    }

    /// A key whose job never ends holds its own later jobs, not another key's
    #[test]
    fn a_key_whose_job_hangs_holds_no_other_keys_jobs() {
        let worker = Worker::new("reims-vgpu.test-worker");
        let (started, begun) = mpsc::channel();
        let (job, go) = blocking(&started, 1);
        worker.submit(1, job);
        assert_eq!(begun.recv_timeout(Duration::from_secs(5)), Ok(1));
        let ran = Arc::new(Mutex::new(Vec::new()));
        for i in 0..100 {
            let ran = Arc::clone(&ran);
            worker.submit(1, Box::new(move || ran.lock().push(i)));
        }
        let (b, b_ran) = mpsc::channel();
        worker.submit(
            2,
            Box::new(move || {
                let _ = b.send(());
            }),
        );
        assert_eq!(
            b_ran.recv_timeout(Duration::from_secs(5)),
            Ok(()),
            "key 2 runs"
        );
        assert!(ran.lock().is_empty(), "key 1's jobs wait for its first");
        drop(go);
        worker.flush();
        assert_eq!(*ran.lock(), (0..100).collect::<Vec<_>>());
    }

    /// With every thread busy on a key of its own, another key's job waits for the first thread
    /// free, counted as a cap hit, then runs
    #[test]
    fn a_key_waits_for_a_free_thread_once_every_thread_is_busy() {
        let worker = Worker::new("reims-vgpu.test-worker");
        let (started, begun) = mpsc::channel();
        let mut goes = Vec::new();
        for key in 0..POOL_THREADS as u64 {
            let (job, go) = blocking(&started, key);
            worker.submit(key, job);
            goes.push(go);
        }
        for _ in 0..POOL_THREADS {
            begun.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        assert_eq!(worker.threads(), POOL_THREADS);
        assert_eq!(worker.cap_hits(), 0);
        let (fifth, fifth_ran) = mpsc::channel();
        worker.submit(
            99,
            Box::new(move || {
                let _ = fifth.send(());
            }),
        );
        assert_eq!(worker.cap_hits(), 1);
        assert_eq!(
            fifth_ran.recv_timeout(Duration::from_millis(50)),
            Err(mpsc::RecvTimeoutError::Timeout),
            "no thread is free"
        );
        assert_eq!(worker.threads(), POOL_THREADS, "no thread past the cap");
        drop(goes.remove(2));
        assert_eq!(fifth_ran.recv_timeout(Duration::from_secs(5)), Ok(()));
        drop(goes);
        worker.flush();
    }

    /// Each key's jobs run one at a time in the order handed over, while the keys' jobs run beside
    /// each other on several threads
    #[test]
    fn each_keys_jobs_run_one_at_a_time_in_handed_order() {
        const KEYS: u64 = 6;
        let worker = Worker::new("reims-vgpu.test-worker");
        let ran: Arc<Mutex<HashMap<u64, Vec<u32>>>> = Arc::default();
        let running: Arc<Vec<AtomicBool>> =
            Arc::new((0..KEYS).map(|_| AtomicBool::new(false)).collect());
        let overlapped = Arc::new(AtomicBool::new(false));
        let threads: Arc<Mutex<std::collections::HashSet<String>>> = Arc::default();
        worker.hold();
        for i in 0..40 {
            for key in 0..KEYS {
                let (ran, running, overlapped, threads) = (
                    Arc::clone(&ran),
                    Arc::clone(&running),
                    Arc::clone(&overlapped),
                    Arc::clone(&threads),
                );
                worker.submit(
                    key,
                    Box::new(move || {
                        let me = &running[key as usize];
                        if me.swap(true, Ordering::SeqCst) {
                            overlapped.store(true, Ordering::SeqCst);
                        }
                        std::thread::sleep(Duration::from_micros(200));
                        let name = std::thread::current().name().unwrap_or("").to_owned();
                        threads.lock().insert(name);
                        ran.lock().entry(key).or_default().push(i);
                        me.store(false, Ordering::SeqCst);
                    }),
                );
            }
        }
        worker.release();
        worker.flush();
        assert!(
            !overlapped.load(Ordering::SeqCst),
            "one job of a key at a time"
        );
        let ran = ran.lock();
        for key in 0..KEYS {
            assert_eq!(ran[&key], (0..40).collect::<Vec<_>>(), "key {key}");
        }
        let threads = threads.lock();
        assert!(
            threads.len() > 1 && threads.len() <= POOL_THREADS,
            "{threads:?}"
        );
        assert!(
            threads
                .iter()
                .all(|n| n.starts_with("reims-vgpu.test-worker-"))
        );
    }

    /// A thread ends once it has waited [`IDLE`] with nothing to do, and a later job starts one
    /// again
    #[test]
    fn an_idle_thread_ends_after_the_idle_time() {
        let worker = Worker::new("reims-vgpu.test-worker");
        worker.submit(1, Box::new(|| {}));
        worker.flush();
        assert_eq!(worker.threads(), 1);
        std::thread::sleep(IDLE / 2);
        assert_eq!(worker.threads(), 1, "not yet idle long enough");
        let mut waited = IDLE / 2;
        while worker.threads() > 0 {
            assert!(waited < IDLE * 5, "the idle thread did not end");
            std::thread::sleep(Duration::from_millis(20));
            waited += Duration::from_millis(20);
        }
        let (ran, wait) = mpsc::channel();
        worker.submit(
            2,
            Box::new(move || {
                let _ = ran.send(std::thread::current().name().map(str::to_owned));
            }),
        );
        assert_eq!(
            wait.recv_timeout(Duration::from_secs(5)),
            Ok(Some("reims-vgpu.test-worker-0".to_owned()))
        );
    }

    /// A job runs in its key's turn, which a waitless job's says it may not wait in
    #[test]
    fn a_job_runs_in_its_keys_turn() {
        let worker = Worker::new("reims-vgpu.test-worker");
        let (seen, turns) = mpsc::channel();
        let also = seen.clone();
        worker.submit(3, Box::new(move || seen.send(TURN.get()).unwrap()));
        worker.submit_waitless(4, Box::new(move || also.send(TURN.get()).unwrap()));
        worker.flush();
        let mut turns: Vec<_> = turns.try_iter().collect();
        turns.sort();
        assert_eq!(turns, [Some((3, true)), Some((4, false))]);
        assert_eq!(TURN.get(), None, "the test's thread runs no job");
    }

    /// A job's wait for another key, one that would block, fails the debug check
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "another key's")]
    fn a_jobs_wait_for_another_key_fails_its_check() {
        let unsettled = Arc::new(Unsettled::default());
        let _held = unsettled.begin(2, 0, MonoTime::from_nanos(0));
        let _turn = Turn::enter(1, true);
        unsettled.wait(2);
    }

    /// A waitless job's wait, even for its own key, fails the debug check
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "waits for nothing")]
    fn a_waitless_jobs_wait_fails_its_check() {
        let unsettled = Arc::new(Unsettled::default());
        let _held = unsettled.begin(1, 0, MonoTime::from_nanos(0));
        let _turn = Turn::enter(1, false);
        unsettled.wait(1);
    }

    /// A job's wait for its own key, or for a settled one, passes the check
    #[test]
    fn a_jobs_wait_for_its_own_key_passes_its_check() {
        let unsettled = Arc::new(Unsettled::default());
        let t = MonoTime::from_nanos(0);
        let held = unsettled.begin(1, 0, t);
        let _other = unsettled.begin(2, 0, t);
        let worker = Worker::new("reims-vgpu.test-worker");
        let (done, waited) = mpsc::channel();
        let waiter = Arc::clone(&unsettled);
        worker.submit(
            1,
            Box::new(move || {
                waiter.wait(3);
                waiter.wait(1);
                let _ = done.send(());
            }),
        );
        std::thread::sleep(Duration::from_millis(20));
        drop(held);
        assert_eq!(waited.recv_timeout(Duration::from_secs(5)), Ok(()));
        worker.flush();
    }

    /// A key is unsettled from its begin to its settle's drop, each begin counted: a wait for it
    /// returns once the last settles, and one for another key at once
    #[test]
    fn a_wait_for_a_key_returns_once_its_last_settle_drops() {
        let unsettled = Arc::new(Unsettled::default());
        let t = MonoTime::from_nanos(0);
        assert!(!unsettled.any([1, 2].into_iter()));
        let first = unsettled.begin(1, 0, t);
        let second = unsettled.begin(1, 0, t);
        assert!(unsettled.any([2, 1].into_iter()));
        assert!(!unsettled.any([2].into_iter()));
        unsettled.wait(2);
        let waiter = {
            let unsettled = Arc::clone(&unsettled);
            std::thread::Builder::new()
                .name("reims-vgpu.test-waiter".into())
                .spawn(move || unsettled.wait(1))
                .unwrap()
        };
        drop(first);
        std::thread::sleep(Duration::from_millis(20));
        assert!(!waiter.is_finished(), "one begin of the key is left");
        drop(second);
        waiter.join().unwrap();
        assert!(!unsettled.any([1].into_iter()));
    }

    /// Hands `job`, of `key`, to `worker`.
    fn submit_of(worker: &Worker, key: u64, job: Job) {
        worker.submit(key, job);
    }

    /// The times a ready key waited for a thread, where the worker counts them.
    fn cap_hits_of(worker: &Worker) -> Option<u64> {
        Some(worker.cap_hits())
    }

    /// Another block's job beside `K` blocks that each run 25 ms jobs back to back: the time
    /// from its submit to its start, as a mean, a median and a 99th percentile, for each `K`, with
    /// the times a ready key waited for a thread
    #[test]
    #[ignore = "a measurement, run by hand in a release build with --no-capture"]
    #[allow(clippy::disallowed_methods)] // a measurement: real time is what it measures
    fn another_blocks_job_beside_busy_blocks() {
        use std::time::Instant;

        const PROBE: u64 = 1 << 40;
        for k in [1u64, 2, 4, 6, 8] {
            let worker = Arc::new(Worker::new("reims-vgpu.bench-worker"));
            let stop = Arc::new(AtomicBool::new(false));
            let blocks: Vec<_> = (0..k)
                .map(|block| {
                    let worker = Arc::clone(&worker);
                    let stop = Arc::clone(&stop);
                    std::thread::Builder::new()
                        .name(format!("reims-vgpu.bench-block-{block}"))
                        .spawn(move || {
                            while !stop.load(Ordering::Relaxed) {
                                let (done, wait) = mpsc::channel();
                                let job = move || {
                                    std::thread::sleep(Duration::from_millis(25));
                                    let _ = done.send(());
                                };
                                submit_of(&worker, block, Box::new(job));
                                let _ = wait.recv();
                            }
                        })
                        .unwrap()
                })
                .collect();
            std::thread::sleep(Duration::from_millis(100));
            let mut waits = Vec::new();
            let from = Instant::now();
            while waits.len() < 100 || from.elapsed() < Duration::from_secs(2) {
                let (started, wait) = mpsc::channel();
                let handed = Instant::now();
                submit_of(
                    &worker,
                    PROBE,
                    Box::new(move || {
                        let _ = started.send(Instant::now());
                    }),
                );
                waits.push(wait.recv().unwrap().duration_since(handed));
                std::thread::sleep(Duration::from_millis(5));
            }
            stop.store(true, Ordering::Relaxed);
            for block in blocks {
                block.join().unwrap();
            }
            waits.sort();
            let ms = |d: Duration| d.as_secs_f64() * 1e3;
            let mean = waits.iter().copied().map(ms).sum::<f64>() / waits.len() as f64;
            let at = |q: f64| ms(waits[((waits.len() - 1) as f64 * q).round() as usize]);
            let hits = cap_hits_of(&worker).map_or("-".to_owned(), |n| n.to_string());
            println!(
                "K={k}: {} jobs, mean {mean:.3} ms, p50 {:.3} ms, p99 {:.3} ms, cap hits {hits}",
                waits.len(),
                at(0.5),
                at(0.99),
            );
        }
    }
}
