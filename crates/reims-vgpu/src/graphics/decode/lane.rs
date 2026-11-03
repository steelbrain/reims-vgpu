//! A command queue's lane: the walks and commits of the queue's command buffers, on threads of the
//! lane's, several at once where there are several to do, each committed in its turn
//! (docs/graphics-kext/design.md, "What a walk sees").
//!
//! - **Jobs.** The queue's service thread (or a doorbell's) takes and begins each command buffer,
//!   in the order the guest published them, and hands the rest of its work to the lane
//!   ([`Lane::run`]): its walk, its commit, its stand-in or the failure posted for it. The lane's
//!   threads take the jobs in the order they came.
//! - **Threads.** A lane with work has one thread of its own, made as a job comes and none is
//!   running, which goes once it has had no job for [`IDLE`]. A job that comes while every thread
//!   of the lane is busy may get the lane another, up to [`LANE_THREADS`] in all, where the
//!   device's budget of such threads ([`Budget`]) has one to spare; a lane never waits for the
//!   budget, so those threads add speed and nothing else, and a device of many busy queues holds
//!   one thread a busy lane and the budget's at most.
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
    /// How many of the lane's threads wait for a job.
    idle: usize,
    /// The queue went: its threads go once its jobs are done.
    closed: bool,
    /// The turns given so far, and every turn below `passed` has passed.
    issued: u64,
    passed: u64,
    /// The turns past `passed` that passed before it.
    early: BTreeSet<u64>,
}

/// A command queue's lane (the module doc).
pub(crate) struct Lane {
    /// The queue's id, for its threads' names.
    queue: u64,
    /// The device's budget of extra threads.
    budget: Arc<Budget>,
    state: Mutex<State>,
    /// A job came, or the lane closed.
    work: Condvar,
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
        Arc::new(Lane {
            queue,
            budget,
            state: Mutex::default(),
            work: Condvar::new(),
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

    /// Runs `job` on one of the lane's threads, after the jobs that came before it have started:
    /// an idle one, else the lane's own where it has none, else one of the budget's where every
    /// thread of the lane is busy and the budget has one to spare, else the next of the lane's
    /// to be free. On the calling thread where the lane has no thread and none can be made.
    pub(crate) fn run(self: &Arc<Self>, job: Job) {
        let mut state = self.state.lock();
        state.jobs.push_back(job);
        // An idle thread for each job not yet started: one woken before takes the one before.
        if state.idle >= state.jobs.len() {
            drop(state);
            self.work.notify_one();
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
            .spawn(move || lane.serve(own));
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

    /// A lane thread, the lane's `own` or one of the budget's: the jobs, in the order they came,
    /// until the lane closes with none left or none comes for [`IDLE`].
    fn serve(&self, own: bool) {
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
            state.idle += 1;
            let waited = self.work.wait_for(&mut state, IDLE);
            state.idle -= 1;
            if waited.timed_out() && state.jobs.is_empty() {
                break;
            }
        }
        if own {
            state.own = false;
        } else {
            state.extra -= 1;
            self.budget.give_back();
        }
    }

    /// The queue went: the lane's threads go once the jobs it has are done.
    pub(crate) fn close(&self) {
        self.state.lock().closed = true;
        self.work.notify_all();
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
}
