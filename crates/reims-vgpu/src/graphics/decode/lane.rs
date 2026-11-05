//! A command queue's lane: the walks and commits of the queue's command buffers, on threads of the
//! lane's, several at once where there are several to do, each committed in its turn
//! (docs/graphics-kext/design.md, "What a walk sees").
//!
//! - **Jobs.** The queue's service thread (or a doorbell's) takes and begins each command buffer,
//!   in the order the guest published them, and hands the rest of its work to the lane
//!   ([`Lane::run`]): its walk, its commit, its stand-in or the failure posted for it. The lane's
//!   threads take the jobs in the order they came. A buffer walked in its turn (one that holds a
//!   video encoder) whose turn is next ([`Turn::is_next`]), and a short buffer whose commit would
//!   wait for nothing, are carried out on the service thread instead, as each waits there for
//!   nothing (`graphics.rs`'s `carry_out`).
//! - **Threads.** A lane with work has one thread of its own, made as a job comes and none is
//!   running. A job that comes while every thread of the lane is busy may get the lane another,
//!   up to [`LANE_THREADS`] in all, where the device's budget of such threads ([`Budget`]) has one
//!   to spare; a lane never waits for the budget, so those threads add speed and nothing else, and
//!   a device of many busy queues holds one thread a busy lane and the budget's at most.
//! - **Which thread takes a job.** A job goes to the lane's thread that went idle last, so the
//!   jobs of a lane that needs one thread at a time stay on one thread, and a thread goes once it
//!   has had no job for [`IDLE`]: the threads a burst of jobs made go a second after the burst,
//!   and a burst after that grows the lane again as the first did. A thread that goes gives back
//!   the budget's thread first, so the lane's own is the last to go. Woken in turn instead, the
//!   longest idle first, every thread took a job in each round and none went: a game's queue kept
//!   the three threads it had from the game's start, one of which ran on the host's efficiency
//!   cores alone, and its walks took 13.4 ms of CPU a frame where two threads took 10.4 ms
//!   (game_6000 at 92.7 against 105.6 fps, on a Core Ultra 9 275HX host with an RTX 5080 Laptop
//!   GPU; measured).
//! - **Turns.** Each command buffer is given a turn as it is begun, in the order they are taken
//!   ([`Lane::turn`]). Its commit (or its stand-in's, or the failure posted at once for it) waits
//!   for its turn ([`Turn::wait`]), which comes once every turn before it has passed, and its turn
//!   passes as its job ends ([`Turn`]'s drop): the queue's command buffers are committed in the
//!   order the guest sent them, as one after another, whatever order their walks end in. Metal
//!   allows a queue's command buffers to be encoded at once and committed in order.
//! - **Why a thread of the lane's own.** Every wait of a job (its turn, its ticket in the commit
//!   order, a fill of a device cache) is for a command buffer begun before it, and a lane's
//!   threads take its jobs in the order they were begun, so the lane's own thread is always at
//!   or past its first unfinished job, and the device's first unfinished one always runs: the
//!   waits end. Threads only shared between queues could all be held by later jobs waiting for
//!   an earlier one not yet started.

use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use reims_vgpu_core::sync::{Condvar, Mutex};

/// The most threads a lane runs its jobs on at once: its own and the budget's.
pub(crate) const LANE_THREADS: usize = 4;

/// How long a lane's thread stays with no job before it goes.
const IDLE: Duration = Duration::from_secs(1);

/// The rest of a command buffer's work, for a lane's thread.
pub(crate) type Job = Box<dyn FnOnce() + Send>;

/// The device's budget of lanes' extra threads (the module doc): at most `limit` at once, across
/// every lane.
#[derive(Debug)]
pub(crate) struct Budget {
    limit: usize,
    taken: AtomicUsize,
}

impl Budget {
    /// A budget of twice the host's cores, as many as can run a walk each with as many waiting.
    pub(crate) fn of_host() -> Arc<Budget> {
        let cores = std::thread::available_parallelism().map_or(4, std::num::NonZero::get);
        Budget::new(2 * cores)
    }

    /// A budget of `limit` threads.
    pub(crate) fn new(limit: usize) -> Arc<Budget> {
        Arc::new(Budget {
            limit,
            taken: AtomicUsize::new(0),
        })
    }

    /// One thread of the budget, where it has one to spare; waits for nothing.
    fn take(&self) -> bool {
        (self.taken)
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < self.limit).then_some(n + 1)
            })
            .is_ok()
    }

    /// A thread of the budget went.
    fn give_back(&self) {
        self.taken.fetch_sub(1, Ordering::AcqRel);
    }

    /// The budget's threads running now (for the tests).
    #[cfg(test)]
    pub(crate) fn taken(&self) -> usize {
        self.taken.load(Ordering::Acquire)
    }
}

/// What a lane's lock guards.
#[derive(Default)]
struct State {
    /// The jobs not yet started, in the order they came.
    jobs: VecDeque<Job>,
    /// The lane has its own thread.
    own: bool,
    /// The budget's threads the lane has.
    extra: usize,
    /// The lane's threads that wait for a job, the one that went idle last at the end: a job wakes
    /// it first (the module doc).
    idle: Vec<Arc<Idler>>,
    /// The threads woken for a job that have not yet taken the lane's lock again.
    woken: usize,
    /// The queue went: its threads go once its jobs are done.
    closed: bool,
    /// The turns given so far, and every turn below `passed` has passed.
    issued: u64,
    passed: u64,
    /// The turns past `passed` that passed before it.
    early: BTreeSet<u64>,
}

/// A lane's thread waiting for a job ([`State::idle`]).
#[derive(Default)]
struct Idler {
    /// A job came for it, or the lane closed.
    wake: Condvar,
}

/// A command queue's lane (the module doc).
pub(crate) struct Lane {
    /// The queue's id, for its threads' names.
    queue: u64,
    /// The device's budget of extra threads.
    budget: Arc<Budget>,
    /// How long a thread of the lane's stays with no job before it goes: [`IDLE`].
    idle_for: Duration,
    state: Mutex<State>,
    /// A turn passed.
    turned: Condvar,
}

impl std::fmt::Debug for Lane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Lane").field("queue", &self.queue).finish()
    }
}

impl Lane {
    /// The lane of queue `queue`, with no thread yet, its extra threads from `budget`.
    pub(crate) fn new(queue: u64, budget: Arc<Budget>) -> Arc<Lane> {
        Lane::with_idle(queue, budget, IDLE)
    }

    /// The lane of queue `queue`, whose threads go after `idle_for` with no job.
    fn with_idle(queue: u64, budget: Arc<Budget>, idle_for: Duration) -> Arc<Lane> {
        Arc::new(Lane {
            queue,
            budget,
            idle_for,
            state: Mutex::default(),
            turned: Condvar::new(),
        })
    }

    /// The next turn, for a command buffer begun now; `None` where only tried and the lane's lock
    /// is busy.
    pub(crate) fn turn(self: &Arc<Self>, try_only: bool) -> Option<Turn> {
        let mut state = if try_only {
            self.state.try_lock()?
        } else {
            self.state.lock()
        };
        let number = state.issued;
        state.issued += 1;
        Some(Turn {
            lane: Arc::clone(self),
            number,
        })
    }

    /// The next turn, with what `with` makes, the two taken together, waiting for nothing: `None`,
    /// with no turn given, where the lane's lock is busy or `with` makes nothing, so a doorbell's
    /// thread lets go of nothing it took ([`super::cmdbuf::Answered`]).
    pub(crate) fn try_turn_with<R>(
        self: &Arc<Self>,
        with: impl FnOnce() -> Option<R>,
    ) -> Option<(Turn, R)> {
        let mut state = self.state.try_lock()?;
        let made = with()?;
        let number = state.issued;
        state.issued += 1;
        let turn = Turn {
            lane: Arc::clone(self),
            number,
        };
        Some((turn, made))
    }

    /// Runs `job` on one of the lane's threads, after the jobs that came before it have started:
    /// the idle one that went idle last, else the lane's own where it has none, else one of the
    /// budget's where every thread of the lane is busy and the budget has one to spare, else the
    /// next of the lane's to be free. On the calling thread where the lane has no thread and none
    /// can be made.
    pub(crate) fn run(self: &Arc<Self>, job: Job) {
        let mut state = self.state.lock();
        state.jobs.push_back(job);
        // A woken thread for each job not yet started: one woken before takes the one before.
        if state.woken >= state.jobs.len() {
            return;
        }
        if let Some(idler) = state.idle.pop() {
            state.woken += 1;
            drop(state);
            idler.wake.notify_one();
            return;
        }
        let own = !state.own;
        if own {
            state.own = true;
        } else if 1 + state.extra < LANE_THREADS && self.budget.take() {
            state.extra += 1;
        } else {
            return;
        }
        drop(state);
        let lane = Arc::clone(self);
        let spawned = std::thread::Builder::new()
            .name(format!("reims-vgpu.graphics-lane-{:#x}", self.queue))
            .spawn(move || lane.serve());
        if spawned.is_err() {
            let mut state = self.state.lock();
            if !own {
                state.extra -= 1;
                self.budget.give_back();
                return;
            }
            state.own = false;
            log::warn!(
                target: "reims_vgpu::graphics",
                "queue {:#x}: no lane thread could be started: its command buffers' work is \
                 carried out where it is taken",
                self.queue
            );
            let jobs = std::mem::take(&mut state.jobs);
            drop(state);
            for job in jobs {
                job();
            }
        }
    }

    /// A lane thread: the jobs, in the order they came, until the lane closes with none left or
    /// none comes for the lane's `idle_for`. It goes as one of the budget's where the lane has any,
    /// else as the lane's own, so the lane's own thread is the last to go.
    fn serve(&self) {
        let me = Arc::new(Idler::default());
        let mut state = self.state.lock();
        loop {
            if let Some(job) = state.jobs.pop_front() {
                drop(state);
                job();
                state = self.state.lock();
                continue;
            }
            if state.closed {
                break;
            }
            state.idle.push(Arc::clone(&me));
            let waited = me.wake.wait_for(&mut state, self.idle_for);
            // Still among the idle where nothing woke it; else a job or the close took it out.
            if let Some(at) = state.idle.iter().rposition(|idler| Arc::ptr_eq(idler, &me)) {
                state.idle.remove(at);
            } else {
                state.woken -= 1;
            }
            if waited.timed_out() && state.jobs.is_empty() {
                break;
            }
        }
        if state.extra > 0 {
            state.extra -= 1;
            self.budget.give_back();
        } else {
            state.own = false;
        }
    }

    /// The queue went: the lane's threads go once the jobs it has are done.
    pub(crate) fn close(&self) {
        let mut state = self.state.lock();
        state.closed = true;
        let idle = std::mem::take(&mut state.idle);
        state.woken += idle.len();
        drop(state);
        for idler in idle {
            idler.wake.notify_one();
        }
    }

    /// How many threads the lane has (for the tests).
    #[cfg(test)]
    pub(crate) fn threads(&self) -> usize {
        let state = self.state.lock();
        usize::from(state.own) + state.extra
    }
}

/// A command buffer's turn in its queue's lane ([`Lane::turn`]): it passes as it drops, once every
/// turn before it has, so the turns after it go on.
pub(crate) struct Turn {
    lane: Arc<Lane>,
    number: u64,
}

impl std::fmt::Debug for Turn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Turn").field(&self.number).finish()
    }
}

impl Turn {
    /// The lane it is a turn of.
    pub(crate) fn lane(&self) -> &Arc<Lane> {
        &self.lane
    }

    /// Whether every turn before this one has passed, so that it waits for none: no job of the
    /// lane's before it is running or waiting to start.
    pub(crate) fn is_next(&self) -> bool {
        self.lane.state.lock().passed == self.number
    }

    /// Waits, with no lock of the caller's held, for every turn before this one to pass.
    pub(crate) fn wait(&self) {
        let mut state = self.lane.state.lock();
        while state.passed < self.number {
            self.lane.turned.wait(&mut state);
        }
    }
}

impl Drop for Turn {
    fn drop(&mut self) {
        let mut guard = self.lane.state.lock();
        let state = &mut *guard;
        if state.passed == self.number {
            state.passed += 1;
            while state.early.remove(&state.passed) {
                state.passed += 1;
            }
        } else {
            state.early.insert(self.number);
        }
        drop(guard);
        self.lane.turned.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// A turn is next once every turn before it has passed, whichever order they pass in.
    #[test]
    fn a_turn_is_next_once_every_turn_before_it_has_passed() {
        let lane = Lane::new(1, Budget::new(0));
        let first = lane.turn(false).unwrap();
        let second = lane.turn(false).unwrap();
        let third = lane.turn(false).unwrap();
        assert!(first.is_next() && !second.is_next() && !third.is_next());
        drop(second);
        assert!(!third.is_next(), "next with the first not passed");
        drop(first);
        assert!(third.is_next());
    }

    /// A turn waits for every turn before it, whichever order they pass in.
    #[test]
    fn a_turn_waits_for_every_turn_before_it() {
        let lane = Lane::new(1, Budget::new(0));
        let first = lane.turn(false).unwrap();
        let second = lane.turn(false).unwrap();
        let third = lane.turn(false).unwrap();
        let order = Mutex::new(Vec::new());
        std::thread::scope(|scope| {
            scope.spawn(|| {
                third.wait();
                order.lock().push(3);
            });
            std::thread::sleep(Duration::from_millis(30));
            // The second passes before the first: the third still waits.
            drop(second);
            std::thread::sleep(Duration::from_millis(30));
            assert!(
                order.lock().is_empty(),
                "the third went before the first passed"
            );
            order.lock().push(1);
            drop(first);
        });
        assert_eq!(*order.lock(), [1, 3]);
        drop(third);
        // Every turn has passed: the next waits for nothing.
        lane.turn(false).unwrap().wait();
    }

    /// A gate jobs wait at until it opens.
    #[derive(Default)]
    struct Gate {
        open: Mutex<bool>,
        moved: Condvar,
    }

    impl Gate {
        fn pass(&self) {
            let mut open = self.open.lock();
            while !*open {
                self.moved.wait(&mut open);
            }
        }

        fn open(&self) {
            *self.open.lock() = true;
            self.moved.notify_all();
        }
    }

    /// Waits about 2 s for `done`; whether it came.
    fn until(done: impl Fn() -> bool) -> bool {
        for _ in 0..2000 {
            if done() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        false
    }

    /// A lane runs its jobs at once on up to [`LANE_THREADS`] threads, its own and the budget's,
    /// and its threads go once it closes.
    #[test]
    fn a_lane_runs_its_jobs_at_once_on_the_budget_s_threads() {
        let budget = Budget::new(LANE_THREADS);
        let lane = Lane::new(2, Arc::clone(&budget));
        let started = Arc::new(Mutex::new(Vec::new()));
        let running = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(Gate::default());
        for n in 0..LANE_THREADS {
            let (started, running, gate) = (
                Arc::clone(&started),
                Arc::clone(&running),
                Arc::clone(&gate),
            );
            lane.run(Box::new(move || {
                started.lock().push(n);
                running.fetch_add(1, Ordering::AcqRel);
                gate.pass();
            }));
        }
        assert!(
            until(|| running.load(Ordering::Acquire) == LANE_THREADS),
            "not all at once"
        );
        let mut order = started.lock().clone();
        order.sort_unstable();
        assert_eq!(order, (0..LANE_THREADS).collect::<Vec<_>>());
        assert_eq!(
            budget.taken(),
            LANE_THREADS - 1,
            "the lane's own thread and the budget's"
        );
        gate.open();
        lane.close();
        assert!(
            until(|| lane.threads() == 0),
            "the closed lane's threads stayed"
        );
        assert_eq!(budget.taken(), 0);
    }

    /// A lane whose budget is spent runs its jobs on its own thread alone, one after another in
    /// the order they came, and never waits for the budget.
    #[test]
    fn a_lane_with_no_budget_left_runs_its_jobs_on_its_own_thread_in_order() {
        let lane = Lane::new(3, Budget::new(0));
        let done = Arc::new(Mutex::new(Vec::new()));
        for n in 0..8 {
            let done = Arc::clone(&done);
            lane.run(Box::new(move || done.lock().push(n)));
        }
        assert!(until(|| done.lock().len() == 8));
        assert_eq!(*done.lock(), (0..8).collect::<Vec<_>>());
        assert_eq!(lane.threads(), 1);
        lane.close();
        assert!(until(|| lane.threads() == 0));
    }

    /// 2048 busy queues, the most a process may make, hold one thread each and the budget's, at
    /// most: the threads a device makes are bounded whatever its processes do.
    #[test]
    fn busy_lanes_hold_one_thread_each_and_the_budget_s() {
        const QUEUES: u64 = 2048;
        const BUDGET: usize = 16;
        let budget = Budget::new(BUDGET);
        let gate = Arc::new(Gate::default());
        let running = Arc::new(AtomicUsize::new(0));
        let lanes: Vec<Arc<Lane>> = (0..QUEUES)
            .map(|queue| Lane::new(queue, Arc::clone(&budget)))
            .collect();
        for lane in &lanes {
            for _ in 0..3 {
                let (gate, running) = (Arc::clone(&gate), Arc::clone(&running));
                lane.run(Box::new(move || {
                    running.fetch_add(1, Ordering::AcqRel);
                    gate.pass();
                }));
            }
        }
        // Every lane runs its first job on its own thread; the budget's threads run a few more.
        assert!(until(
            || running.load(Ordering::Acquire) >= QUEUES as usize + BUDGET
        ));
        let threads: usize = lanes.iter().map(|lane| lane.threads()).sum();
        assert!(
            threads <= QUEUES as usize + BUDGET,
            "{threads} threads for {QUEUES} lanes and a budget of {BUDGET}"
        );
        assert_eq!(budget.taken(), BUDGET);
        gate.open();
        for lane in &lanes {
            lane.close();
        }
        assert!(until(|| lanes.iter().all(|lane| lane.threads() == 0)));
        assert_eq!(running.load(Ordering::Acquire), 3 * QUEUES as usize);
    }

    /// How long the lanes of the tests below keep a thread with no job.
    const TEST_IDLE: Duration = Duration::from_millis(100);

    /// Runs `n` jobs on `lane` that wait at one gate, and waits until they run at once and the lane
    /// has `n` threads; the gate opened, waits until they are done.
    fn burst(lane: &Arc<Lane>, n: usize) {
        let gate = Arc::new(Gate::default());
        let running = Arc::new(AtomicUsize::new(0));
        let done = Arc::new(AtomicUsize::new(0));
        for _ in 0..n {
            let (gate, running, done) =
                (Arc::clone(&gate), Arc::clone(&running), Arc::clone(&done));
            lane.run(Box::new(move || {
                running.fetch_add(1, Ordering::AcqRel);
                gate.pass();
                done.fetch_add(1, Ordering::AcqRel);
            }));
        }
        assert!(
            until(|| running.load(Ordering::Acquire) == n),
            "the burst's {n} jobs did not run at once"
        );
        assert_eq!(lane.threads(), n, "a thread a job of the burst");
        gate.open();
        assert!(until(|| done.load(Ordering::Acquire) == n));
    }

    /// The pause after each job of [`one_at_a_time_until_one_thread`], and how many jobs it runs at
    /// most: about 4 s of them.
    const JOB_GAP: Duration = Duration::from_millis(5);
    const MOST_JOBS: u32 = 800;

    /// Runs jobs on `lane` one at a time, each after the one before has ended and its thread has
    /// had [`JOB_GAP`] to wait again, until the lane has one thread left after the jobs of more
    /// than [`TEST_IDLE`]; whether it came to that within [`MOST_JOBS`]. The threads the jobs ran
    /// on are pushed to `ran_on`.
    fn one_at_a_time_until_one_thread(
        lane: &Arc<Lane>,
        ran_on: &Mutex<Vec<std::thread::ThreadId>>,
    ) -> bool {
        let past_idle = u32::try_from(TEST_IDLE.as_millis() / JOB_GAP.as_millis()).unwrap() + 1;
        for n in 0..MOST_JOBS {
            let ended = Arc::new(AtomicUsize::new(0));
            let (job_ended, ran) = (Arc::clone(&ended), Arc::new(Mutex::new(None)));
            let job_ran = Arc::clone(&ran);
            lane.run(Box::new(move || {
                *job_ran.lock() = Some(std::thread::current().id());
                job_ended.store(1, Ordering::Release);
            }));
            assert!(until(|| ended.load(Ordering::Acquire) == 1));
            ran_on.lock().extend(ran.lock().take());
            std::thread::sleep(JOB_GAP);
            if n >= past_idle && lane.threads() == 1 {
                return true;
            }
        }
        false
    }

    /// Jobs that come one at a time go to the thread that went idle last, so the threads a burst
    /// made go after the lane's idle time while the jobs go on, the budget's first: woken the
    /// longest idle first, each of the three took a job in turn and none went.
    #[test]
    fn a_burst_s_threads_go_while_jobs_come_one_at_a_time() {
        let budget = Budget::new(LANE_THREADS);
        let lane = Lane::with_idle(4, Arc::clone(&budget), TEST_IDLE);
        burst(&lane, 3);
        let ran_on = Mutex::new(Vec::new());
        assert!(
            one_at_a_time_until_one_thread(&lane, &ran_on),
            "{} threads stayed for jobs one at a time",
            lane.threads()
        );
        assert_eq!(budget.taken(), 0, "the lane's own thread is the one left");
        // The last jobs ran on the one thread left.
        let ran_on = ran_on.lock();
        let last = ran_on.last().copied();
        assert!(ran_on.iter().rev().take(5).all(|id| Some(*id) == last));
        drop(ran_on);
        lane.close();
        assert!(until(|| lane.threads() == 0));
    }

    /// A burst that comes once a lane is down to one thread grows it to [`LANE_THREADS`] again,
    /// as the first burst did.
    #[test]
    fn a_burst_after_the_threads_went_grows_the_lane_to_its_limit_again() {
        let budget = Budget::new(LANE_THREADS);
        let lane = Lane::with_idle(5, Arc::clone(&budget), TEST_IDLE);
        burst(&lane, LANE_THREADS);
        assert!(one_at_a_time_until_one_thread(
            &lane,
            &Mutex::new(Vec::new())
        ));
        burst(&lane, LANE_THREADS);
        assert_eq!(budget.taken(), LANE_THREADS - 1);
        // With no job at all, every thread goes.
        assert!(
            until(|| lane.threads() == 0),
            "the idle lane's threads stayed"
        );
        assert_eq!(budget.taken(), 0);
    }
}
