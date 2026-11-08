//! `SerialQueue`: a serial queue of core's own, and the `Executor` trait.
//!
//! A queue runs its async blocks one at a time, in submission order, on a worker thread the
//! platform starts for it (`Threads::spawn_worker`); a worker idle for [`IDLE_EXIT`] ends, and
//! the next block starts another. What every block sees, whichever thread runs it:
//! - an async block starts with no handler, so a raise in it with no handler of its own dies at
//!   the raise, its site labelled as having none (6.2). Its code holds under
//!   [`Hold::NONE`](crate::rt::Hold::NONE), with no holder;
//! - a sync block runs on the caller's thread, directly, once every block submitted before it has
//!   run, and holds the queue until it returns. It too starts with no handler, whatever handler
//!   the caller is in, so a raise in it with no handler of its own dies at the raise, even under
//!   the caller's. Its code holds under `Hold::NONE` (the block is `Send` and the caller's hold
//!   is not). An Objective-C exception never leaves it either, since every call it makes into
//!   Metal, Foundation or the host catches its own at the call. Its autoreleases go to the
//!   caller's pool, and it owns the caller's queues too;
//! - a `sync` onto a queue the thread already owns runs its block at once, as a nested call: the
//!   thread holds the queue already, and the blocks queued behind the one it runs cannot run
//!   before that block returns, so waiting for them would never end
//!   (`tests::sync_onto_current_queue_runs_at_once`).
//!
//! Running the sync block on the caller's thread is what orders it against the caller's next
//! step: the caller goes on the moment the block returns, with no hand-back from another thread
//! (lessons.md, "A dispatch_sync block runs on the caller's thread": a child FIFO's first
//! interrupt otherwise overtakes the root's).
//!
//! A worker releases its last-resort holder and pops its last-resort real pool, releasing what
//! its blocks held or autoreleased outside any unit, once it has left the queue
//! (`pool::drain_idle`): the queue ran dry, or it handed the queue to a sync caller. A release
//! there may `sync` onto the same queue: the worker then runs the blocks queued ahead of it
//! itself (lessons.md, "A queue worker's idle drain").
//!
//! A worker is woken only when it has something to do, and after the queue's lock is released:
//! a submit or a sync never waits for a worker that holds the lock but has no CPU yet (a fault's
//! interrupt block on the FIFO queue would otherwise fall behind the guest's next resume).

use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use crate::platform::os::{Threads, TimerClock};
#[cfg(test)]
pub(crate) use crate::rt::block_env::current_label;
use crate::rt::block_env::{Owned, Submitted, owned_chain};
use crate::rt::pool;
use crate::sync::{Condvar, Mutex, MutexGuard};

/// A block submitted to a queue.
pub type Job = Box<dyn FnOnce() + Send + 'static>;

/// Something that runs jobs asynchronously in submission order: a `SerialQueue`, or the host's
/// queue (`host::HostQueue`). A source's target (`CoalescingSignal`, `Timer`).
pub trait Executor: Send + Sync {
    /// Runs `job` after every job submitted before it.
    fn exec_async(&self, job: Job);

    /// A source's own block on this target (a handler invocation, the cancel callout). The
    /// host's dispatch queue runs it at the queue's QoS; the default is [`Executor::exec_async`].
    fn exec_source(&self, job: Job) {
        self.exec_async(job)
    }
}

/// How long a worker waits with nothing to do before it ends.
pub const IDLE_EXIT: Duration = Duration::from_secs(5);

/// A queue worker's stack.
pub const WORKER_STACK: usize = 512 * 1024;

static NEXT_QUEUE_ID: AtomicU64 = AtomicU64::new(1);

thread_local! {
    /// The id of the queue this thread is the worker of (0: none).
    static WORKER_OF: Cell<u64> = const { Cell::new(0) };
}

/// Who holds a queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Owner {
    /// Nobody.
    Idle,
    /// The worker, running an async block.
    Worker,
    /// A sync caller, running its block on its own thread.
    Caller,
}

/// A sync caller's turn, which whoever gives the queue up hands over.
#[derive(Default)]
struct Turn {
    given: Mutex<bool>,
    cv: Condvar,
    /// The caller is the queue's own worker (its idle drain), which waits on the queue's
    /// `work` condvar rather than on `cv`.
    on_worker: bool,
}

impl Turn {
    fn give(&self) {
        *self.given.lock() = true;
        self.cv.notify_all();
    }

    fn wait(&self) {
        let mut g = self.given.lock();
        while !*g {
            self.cv.wait(&mut g);
        }
    }
}

/// An async block, and in tests the death slot of the test that submitted it (the worker's idle
/// drain releases what that block autoreleased under it).
struct Block {
    job: Job,
    #[cfg(any(test, feature = "testkit"))]
    slot: Option<Arc<crate::testkit::capture::DeathSlot>>,
}

/// What a queue holds, in submission order.
enum Item {
    Block(Block),
    /// A sync caller waiting for its turn.
    Caller(Arc<Turn>),
}

struct State {
    items: VecDeque<Item>,
    owner: Owner,
    /// A worker thread is alive for the queue.
    worker: bool,
    /// The last handle is gone: the worker ends once the queue is empty.
    released: bool,
}

struct Shared {
    id: u64,
    label: &'static str,
    threads: Arc<dyn Threads>,
    clock: Option<Arc<dyn TimerClock>>,
    state: Mutex<State>,
    /// Wakes the worker: a block queued, the queue given back, the last handle gone.
    work: Condvar,
}

/// One reference `Clone` counts; the last one's drop lets the worker end once the queue is
/// empty. The blocks already submitted still run.
struct Handle(Arc<Shared>);

impl Drop for Handle {
    fn drop(&mut self) {
        self.0.state.lock().released = true;
        self.0.work.notify_one();
    }
}

/// A serial queue. `Clone` is another reference; the blocks still queued when the last one is
/// gone still run.
#[derive(Clone)]
pub struct SerialQueue {
    handle: Arc<Handle>,
}

impl std::fmt::Debug for SerialQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SerialQueue")
            .field("label", &self.shared().label)
            .field("id", &self.shared().id)
            .finish()
    }
}

impl SerialQueue {
    /// A queue labelled `label`, with its worker started. The library's labels are
    /// `reims-vgpu.<role>` ("reims-vgpu.display" among them), and the platform names the worker's
    /// thread after its queue. Labels are diagnostics.
    pub fn new(threads: &dyn Threads, label: &'static str) -> SerialQueue {
        let shared = Arc::new(Shared {
            id: NEXT_QUEUE_ID.fetch_add(1, Ordering::Relaxed),
            label,
            threads: threads.handle(),
            clock: threads.timer_clock(),
            state: Mutex::new(State {
                items: VecDeque::new(),
                owner: Owner::Idle,
                worker: true,
                released: false,
            }),
            work: Condvar::new(),
        });
        Shared::start_worker(&shared);
        SerialQueue {
            handle: Arc::new(Handle(shared)),
        }
    }

    fn shared(&self) -> &Arc<Shared> {
        &self.handle.0
    }

    /// The label the queue was created with.
    pub fn label(&self) -> &'static str {
        self.shared().label
    }

    fn me(&self) -> Owned {
        (self.shared().id, self.shared().label)
    }

    /// The clock the queue's timers wait on (`rt::Timer`).
    pub(crate) fn timer_clock(&self) -> Option<Arc<dyn TimerClock>> {
        self.shared().clock.clone()
    }

    /// Runs `f` after every block submitted before it, on the queue's worker, with no handler (a
    /// raise in it with no handler of its own dies at the raise) and no holder
    /// ([`Hold::NONE`](crate::rt::Hold::NONE)), owning only this queue.
    pub fn exec_async(&self, f: impl FnOnce() + Send + 'static) {
        let block = Submitted::new(vec![self.me()], true);
        Shared::submit(
            self.shared(),
            Block {
                job: Box::new(move || block.run(f)),
                #[cfg(any(test, feature = "testkit"))]
                slot: crate::testkit::capture::current_slot(),
            },
        );
    }

    /// Whether the current thread owns this queue (runs one of its blocks, directly or through
    /// a chain of `sync`s).
    pub fn is_current(&self) -> bool {
        let id = self.shared().id;
        owned_chain().iter().any(|&(i, _)| i == id)
    }

    /// The queue, held by the caller from now, when it is idle with nothing queued: no worker
    /// runs a block of it and no sync caller holds or waits for it. A block or turn submitted from
    /// now on waits behind the caller's, which [`HeldTurn::run`] runs on the caller's thread as a
    /// sync block. `None`, holding nothing, when the queue is busy, and on a thread that owns it
    /// already. Nothing waits and no thread is woken to take it.
    pub fn hold_if_idle(&self) -> Option<HeldTurn<'_>> {
        if self.is_current() {
            return None;
        }
        let mut st = self.shared().state.lock();
        if st.owner != Owner::Idle || !st.items.is_empty() {
            return None;
        }
        st.owner = Owner::Caller;
        Some(HeldTurn {
            queue: self,
            _not_send: std::marker::PhantomData,
        })
    }

    /// Runs `f` on the caller's thread once every block submitted before it has run, and holds
    /// the queue until it returns. `f` starts with no handler (a raise in it with no handler of
    /// its own dies at the raise, whatever handler the caller has) and no holder
    /// ([`Hold::NONE`](crate::rt::Hold::NONE)), and what it autoreleases goes to the caller's
    /// real pool. A caller that owns the queue already runs `f` at once (module doc).
    /// `f` is `Send`, as a block of the queue's own is: it takes nothing that is bound to the
    /// caller's thread, its hold included.
    pub fn sync<R>(&self, f: impl FnOnce() -> R + Send) -> R {
        let mut owned = owned_chain();
        let nested = owned.iter().any(|&(i, _)| i == self.shared().id);
        owned.push(self.me());
        let block = Submitted::new(owned, false);
        /// Gives the queue up when the block returns, or unwinds.
        struct Release<'a>(&'a Shared);
        impl Drop for Release<'_> {
            fn drop(&mut self) {
                self.0.give_up();
            }
        }
        let release = (!nested).then(|| {
            self.shared().take_turn();
            Release(self.shared())
        });
        let out = block.run(f);
        drop(release);
        out
    }
}

/// A queue its caller holds ([`SerialQueue::hold_if_idle`]) until it runs its block or drops.
/// Thread-bound (`!Send`): the caller gives the queue up.
pub struct HeldTurn<'a> {
    queue: &'a SerialQueue,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl HeldTurn<'_> {
    /// Runs `f` as the queue's `sync` runs its block (module doc), then gives the queue up.
    pub fn run<R>(self, f: impl FnOnce() -> R + Send) -> R {
        let mut owned = owned_chain();
        owned.push(self.queue.me());
        Submitted::new(owned, false).run(f)
    }
}

impl Drop for HeldTurn<'_> {
    fn drop(&mut self) {
        self.queue.shared().give_up();
    }
}

impl Shared {
    fn start_worker(this: &Arc<Shared>) {
        let me = Arc::clone(this);
        this.threads
            .spawn_worker(this.label, Box::new(move || Shared::work(&me)));
    }

    /// Queues `block`, starting a worker if none is alive.
    ///
    /// The worker is woken after the lock is released, as every wake of it is: woken under the
    /// lock, it would queue on the lock, and the lock could be handed to it before it runs, so
    /// that a submitter or a sync caller waits for a thread that has no CPU yet.
    fn submit(this: &Arc<Shared>, block: Block) {
        let mut st = this.state.lock();
        st.items.push_back(Item::Block(block));
        let start = !st.worker;
        st.worker = true;
        drop(st);
        if start {
            Shared::start_worker(this);
        } else {
            this.work.notify_one();
        }
    }

    /// With the queue idle, runs or hands over its next item; `false` when there is none, or
    /// the queue is held.
    fn step(
        &self,
        st: &mut MutexGuard<'_, State>,
        #[cfg(any(test, feature = "testkit"))] slot: &mut Option<
            Arc<crate::testkit::capture::DeathSlot>,
        >,
    ) -> bool {
        if st.owner != Owner::Idle {
            return false;
        }
        match st.items.pop_front() {
            None => false,
            Some(Item::Block(b)) => {
                st.owner = Owner::Worker;
                #[cfg(any(test, feature = "testkit"))]
                {
                    *slot = b.slot;
                }
                MutexGuard::unlocked(st, b.job);
                st.owner = Owner::Idle;
                true
            }
            Some(Item::Caller(turn)) => {
                st.owner = Owner::Caller;
                turn.give();
                true
            }
        }
    }

    /// The worker: runs the queue's blocks in order, hands the queue to sync callers at their
    /// turn, pops its last-resort pool once it has left the queue (the queue ran dry, or a sync
    /// caller holds it), and ends once the last handle is gone or it has waited
    /// [`IDLE_EXIT`] with nothing to do.
    fn work(this: &Arc<Shared>) {
        crate::rt::thread::become_ours();
        WORKER_OF.with(|w| w.set(this.id));
        // The first last-resort pool.
        pool::drain_idle();
        #[cfg(any(test, feature = "testkit"))]
        let mut slot = None;
        let mut drained = true;
        let mut st = this.state.lock();
        loop {
            #[cfg(any(test, feature = "testkit"))]
            let stepped = this.step(&mut st, &mut slot);
            #[cfg(not(any(test, feature = "testkit")))]
            let stepped = this.step(&mut st);
            if stepped {
                drained = false;
                continue;
            }
            // The worker has left the queue.
            if !drained {
                drained = true;
                // A release in the drain may submit or sync: not under the lock.
                #[cfg(any(test, feature = "testkit"))]
                {
                    let s = slot.clone();
                    MutexGuard::unlocked(&mut st, || {
                        crate::testkit::capture::with_slot(s, pool::drain_idle)
                    });
                }
                #[cfg(not(any(test, feature = "testkit")))]
                MutexGuard::unlocked(&mut st, pool::drain_idle);
                continue;
            }
            let idle = st.owner == Owner::Idle && st.items.is_empty();
            if idle && st.released {
                break;
            }
            let timed_out = this.work.wait_for(&mut st, IDLE_EXIT).timed_out();
            if timed_out && st.owner == Owner::Idle && st.items.is_empty() {
                break;
            }
        }
        st.worker = false;
        drop(st);
        WORKER_OF.with(|w| w.set(0));
        pool::drain_thread_end();
    }

    /// Waits for the caller's turn: at once when the queue is idle and empty, else behind every
    /// item queued before it. The queue's own worker, between blocks (its idle drain), runs the
    /// blocks ahead of it itself, since no other thread would.
    fn take_turn(&self) {
        let mut st = self.state.lock();
        if st.owner == Owner::Idle && st.items.is_empty() {
            st.owner = Owner::Caller;
            return;
        }
        let on_worker = WORKER_OF.with(Cell::get) == self.id;
        let turn = Arc::new(Turn {
            on_worker,
            ..Turn::default()
        });
        st.items.push_back(Item::Caller(Arc::clone(&turn)));
        if !on_worker {
            drop(st);
            turn.wait();
            return;
        }
        #[cfg(any(test, feature = "testkit"))]
        let mut slot = None;
        loop {
            // Handed over by the caller before it.
            if *turn.given.lock() {
                return;
            }
            if st.owner == Owner::Idle
                && let Some(Item::Caller(next)) = st.items.front()
                && Arc::ptr_eq(next, &turn)
            {
                st.items.pop_front();
                st.owner = Owner::Caller;
                return;
            }
            #[cfg(any(test, feature = "testkit"))]
            let stepped = self.step(&mut st, &mut slot);
            #[cfg(not(any(test, feature = "testkit")))]
            let stepped = self.step(&mut st);
            if !stepped {
                self.work.wait(&mut st);
            }
        }
    }

    /// A sync caller's block is done: the next caller at the head takes the queue straight
    /// away; else the queue is idle. Turns are given under the queue's lock. The worker is woken
    /// only when it has something to do: a block at the head, or its own turn (in
    /// [`Shared::take_turn`]).
    fn give_up(&self) {
        let mut st = self.state.lock();
        let wake = if let Some(Item::Caller(_)) = st.items.front()
            && let Some(Item::Caller(turn)) = st.items.pop_front()
        {
            turn.give();
            turn.on_worker
        } else {
            st.owner = Owner::Idle;
            !st.items.is_empty()
        };
        drop(st);
        if wake {
            self.work.notify_one();
        }
    }
}

impl Executor for SerialQueue {
    fn exec_async(&self, job: Job) {
        SerialQueue::exec_async(self, job)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raise::OnFault;
    use crate::raise::RaiseResult;
    use crate::raise::death::DeathSite;
    use crate::rt::signal::CoalescingSignal;
    use crate::rt::thread::is_our_thread;
    use crate::rt::thread::test_threads::StdThreads;
    use crate::testkit::capture::{expect_death, raise};
    use crate::testkit::threads::FakeThreads;
    use std::sync::atomic::AtomicBool;
    use std::sync::mpsc;
    use std::time::Duration;

    const WAIT: Duration = Duration::from_secs(10);

    fn q(label: &'static str) -> SerialQueue {
        SerialQueue::new(&StdThreads, label)
    }

    /// What a block saw of the thread it ran on.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Seen {
        thread: std::thread::ThreadId,
        label: Option<&'static str>,
        ours: bool,
    }

    fn seen() -> Seen {
        Seen {
            thread: std::thread::current().id(),
            label: current_label(),
            ours: is_our_thread(),
        }
    }

    /// An idle queue is held at once: its block runs on the caller's thread with the queue
    /// current, and a block submitted meanwhile runs after it
    #[test]
    fn hold_if_idle_runs_on_the_callers_thread_ahead_of_later_blocks() {
        let queue = q("q");
        queue.sync(|| ());
        let log = Arc::new(Mutex::new(Vec::new()));
        let turn = queue.hold_if_idle().expect("an idle queue");
        let l = log.clone();
        queue.exec_async(move || l.lock().push(("later", seen())));
        std::thread::sleep(Duration::from_millis(20));
        let l = log.clone();
        turn.run(move || l.lock().push(("held", seen())));
        queue.sync(|| ());
        let log = log.lock().clone();
        assert_eq!(log.len(), 2);
        assert_eq!(log[0].0, "held");
        assert_eq!(log[0].1.thread, std::thread::current().id());
        assert_eq!(log[0].1.label, Some("q"));
        assert_eq!(log[1].0, "later");
        assert_ne!(log[1].1.thread, std::thread::current().id());
    }

    /// A busy queue, or one the thread owns, is not held, and nothing waits
    #[test]
    fn hold_if_idle_hands_a_busy_queue_back() {
        let queue = q("q");
        let (started, running) = mpsc::channel();
        let (release, held) = mpsc::channel::<()>();
        queue.exec_async(move || {
            let _ = started.send(());
            let _ = held.recv();
        });
        running.recv_timeout(WAIT).unwrap();
        assert!(queue.hold_if_idle().is_none());
        queue.exec_async(|| ());
        drop(release);
        queue.sync(|| ());
        let q2 = queue.clone();
        assert!(queue.sync(move || q2.hold_if_idle().is_none()));
        // A turn dropped unrun gives the queue up.
        drop(queue.hold_if_idle().expect("an idle queue"));
        queue.sync(|| ());
    }

    /// Async blocks run in FIFO order
    #[test]
    fn async_fifo_order() {
        let queue = q("reims-vgpu.fifo-control");
        let log = Arc::new(Mutex::new(Vec::new()));
        for i in 0..100u32 {
            let l = log.clone();
            queue.exec_async(move || l.lock().push(i));
        }
        queue.sync(|| ());
        assert_eq!(*log.lock(), (0..100).collect::<Vec<_>>());
        assert_eq!(queue.label(), "reims-vgpu.fifo-control");
        assert!(format!("{queue:?}").contains("reims-vgpu.fifo-control"));
    }

    /// A queue starts its worker through the platform, by label
    #[test]
    fn a_queue_starts_its_worker() {
        let threads = FakeThreads::default();
        let queue = SerialQueue::new(&threads, "reims-vgpu.fifo-control");
        let _r = SerialQueue::new(&threads, "reims-vgpu.reporting");
        assert_eq!(queue.sync(|| 7), 7);
        assert!(threads.spawned().is_empty());
        assert_eq!(
            threads.queues(),
            vec!["reims-vgpu.fifo-control", "reims-vgpu.reporting"]
        );
    }

    /// An async block runs on the queue's worker, not the caller, with the queue current and
    /// the thread marked ours
    #[test]
    fn async_block_runs_on_the_worker_with_the_queue_current() {
        let queue = q("reims-vgpu.fifo-control");
        let (tx, rx) = mpsc::channel();
        queue.exec_async(move || {
            let _ = tx.send(seen());
        });
        let got = rx.recv_timeout(WAIT).ok();
        let got = got.as_ref();
        assert_ne!(got.map(|s| s.thread), Some(std::thread::current().id()));
        assert_eq!(got.map(|s| s.label), Some(Some("reims-vgpu.fifo-control")));
        assert_eq!(got.map(|s| s.ours), Some(true));
    }

    /// A sync block runs on the caller's thread with the queue current, the innermost queue's
    /// through a nested sync; the caller's comes back after
    #[test]
    fn sync_block_has_the_queue_current() {
        let a = q("reims-vgpu.efi-present");
        let b = q("reims-vgpu.reporting");
        assert_eq!(current_label(), None);
        let me = std::thread::current().id();
        let got = a.sync(move || {
            let outer = (std::thread::current().id(), current_label());
            let inner = b.sync(current_label);
            (outer, inner, current_label())
        });
        assert_eq!(
            got,
            (
                (me, Some("reims-vgpu.efi-present")),
                Some("reims-vgpu.reporting"),
                Some("reims-vgpu.efi-present")
            )
        );
        assert_eq!(current_label(), None);
    }

    /// A DATA_OR source on `reims-vgpu.iosurface`: its handler runs as a block of the queue
    #[test]
    fn a_source_handler_runs_as_a_block_of_its_queue() {
        let hw_queue = q("reims-vgpu.iosurface");
        let (tx, rx) = mpsc::channel();
        let tx = Mutex::new(tx);
        let source = CoalescingSignal::new_or(Arc::new(hw_queue.clone()), move |_| {
            let _ = tx.lock().send(seen());
        });
        source.merge(1);
        let got = rx.recv_timeout(WAIT).ok();
        assert_eq!(
            got.as_ref().map(|s| (s.label, s.ours)),
            Some((Some("reims-vgpu.iosurface"), true))
        );
        source.cancel();
    }

    /// The worker ends once the last reference is gone and what was queued has run
    #[test]
    fn the_worker_ends_after_the_last_release() {
        struct OnExit(mpsc::Sender<()>);
        impl Drop for OnExit {
            fn drop(&mut self) {
                let _ = self.0.send(());
            }
        }
        let (tx, rx) = mpsc::channel();
        let (ran_tx, ran_rx) = mpsc::channel();
        {
            let queue = q("q");
            queue.exec_async(move || {
                // Released at the thread's end.
                thread_local!(static EXIT: std::cell::RefCell<Option<OnExit>> = const { std::cell::RefCell::new(None) });
                EXIT.with(|e| *e.borrow_mut() = Some(OnExit(tx)));
            });
            queue.exec_async(move || {
                let _ = ran_tx.send(());
            });
        }
        assert!(ran_rx.recv_timeout(WAIT).is_ok());
        assert!(rx.recv_timeout(WAIT).is_ok());
    }

    /// Sync returns the block's value after earlier async blocks ran
    #[test]
    fn sync_after_async() {
        let queue = q("q");
        let v = Arc::new(Mutex::new(0));
        let v2 = v.clone();
        queue.exec_async(move || {
            std::thread::sleep(Duration::from_millis(20));
            *v2.lock() = 5;
        });
        let v3 = v.clone();
        assert_eq!(queue.sync(move || *v3.lock() + 1), 6);
    }

    /// Libdispatch runs a sync block on the caller's thread, idle queue or not
    /// (`dispatch_sync(_fifoQueue, …)`)
    #[test]
    #[allow(clippy::disallowed_methods)] // test threads: no Objective-C runs here (4.5)
    fn sync_runs_on_callers_thread() {
        let queue = q("reims-vgpu.fifo-control");
        let me = std::thread::current().id();
        assert_eq!(queue.sync(|| std::thread::current().id()), me);
        let (go_tx, go_rx) = mpsc::channel::<()>();
        queue.exec_async(move || {
            let _ = go_rx.recv_timeout(Duration::from_secs(5));
        });
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            let _ = go_tx.send(());
        });
        // Busy: the caller waits its turn, then still runs the block itself.
        assert_eq!(queue.sync(|| std::thread::current().id()), me);
        assert!(t.join().is_ok());
    }

    /// Sync blocks from two threads and async blocks keep one order: none overlaps
    #[test]
    #[allow(clippy::disallowed_methods)] // test threads: no Objective-C runs here (4.5)
    fn sync_and_async_serialise() {
        for queue in [q("q"), q("q2")] {
            let inside = Arc::new(AtomicBool::new(false));
            let mut ts = Vec::new();
            for _ in 0..4 {
                let queue = queue.clone();
                let inside = inside.clone();
                ts.push(std::thread::spawn(move || {
                    for i in 0..200 {
                        let i2 = inside.clone();
                        let block = move || {
                            assert!(!i2.swap(true, Ordering::SeqCst));
                            std::thread::yield_now();
                            i2.store(false, Ordering::SeqCst);
                        };
                        if i % 2 == 0 {
                            queue.sync(block);
                        } else {
                            queue.exec_async(block);
                        }
                    }
                }));
            }
            for t in ts {
                assert!(t.join().is_ok());
            }
            queue.sync(|| ());
            assert!(!inside.load(Ordering::SeqCst));
        }
    }

    /// Sync inside an async block onto another queue (the reporting hop) works
    #[test]
    fn sync_inside_async_onto_other_queue() {
        let fifo = q("reims-vgpu.fifo-control");
        let reporting = q("reims-vgpu.reporting");
        let (tx, rx) = mpsc::channel();
        let r2 = reporting.clone();
        fifo.exec_async(move || {
            let inner = r2.sync(|| (41, current_label())).0 + 1;
            let _ = tx.send((inner, current_label()));
        });
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(5)).ok(),
            Some((42, Some("reims-vgpu.fifo-control")))
        );
    }

    /// A sync onto the current queue from its own async block runs at once, on the worker, ahead
    /// of a block queued behind the running one, which runs after
    #[test]
    fn sync_onto_current_queue_runs_at_once() {
        let queue = q("q");
        let (tx, rx) = mpsc::channel();
        let (q2, t2, t3) = (queue.clone(), tx.clone(), tx);
        let (hold, held) = mpsc::channel::<()>();
        queue.exec_async(move || {
            let _ = held.recv();
            let label = q2.sync(current_label);
            let _ = t2.send(("nested", label));
        });
        queue.exec_async(move || {
            let _ = t3.send(("behind", current_label()));
        });
        drop(hold);
        let got: Vec<_> = (0..2)
            .map(|_| rx.recv_timeout(Duration::from_secs(5)).ok())
            .collect();
        assert_eq!(
            got,
            [Some(("nested", Some("q"))), Some(("behind", Some("q")))]
        );
    }

    /// A nested sync back onto a queue in the ownership chain runs at once (A -> sync B -> sync
    /// A), and the chain comes back as it was
    #[test]
    fn sync_back_onto_owner_chain_runs_at_once() {
        let a = q("A");
        let b = q("B");
        let (a2, b2) = (a.clone(), b.clone());
        let got = a.sync(move || {
            b2.sync(move || {
                let inner = a2.sync(current_label);
                (inner, current_label())
            })
        });
        assert_eq!(got, (Some("A"), Some("B")));
        assert_eq!(current_label(), None);
        // Both queues are free again.
        assert_eq!(a.sync(|| 1) + b.sync(|| 1), 2);
    }

    /// A throw in an async block with no scope of its own dies at the throw
    #[test]
    fn throw_in_async_block_dies() {
        let queue = q("q");
        expect_death(
            DeathSite {
                kind: crate::raise::DeathKind::UnhandledRaise,
                file: "gpu_task",
                line: 0,
            },
            move || {
                queue.exec_async(|| {
                    let _ = raise(OnFault::Die, "gpu_task", "x");
                });
                loop {
                    std::thread::park();
                }
            },
        );
    }

    /// A sync block starts with no handler; its fatal raise dies at the raise, whatever handler
    /// the caller has (libdispatch, measured on macOS 26.7), and the queue is free again for the
    /// next caller
    #[test]
    fn sync_block_fatal_raise_dies() {
        let queue2 = q("q2");
        let q3 = queue2.clone();
        expect_death(
            DeathSite {
                kind: crate::raise::DeathKind::UnhandledRaise,
                file: "device",
                line: 0,
            },
            move || {
                let _: RaiseResult<()> = queue2.sync(|| Err(raise(OnFault::Die, "device", "x")));
            },
        );
        assert_eq!(q3.sync(|| 3), 3);
    }

    /// A sync block's raise that is `Err` comes back through the sync, and the queue keeps running
    #[test]
    fn sync_block_err_comes_back() {
        let queue = q("q");
        let r: RaiseResult<u32> = queue.sync(|| Err(raise(OnFault::Return, "device", "x")));
        assert!(r.is_err());
        assert_eq!(queue.sync(|| 4), 4);
    }

    /// is_current, including through a sync chain
    #[test]
    fn current_queue_checks() {
        let a = q("A");
        let b = q("B");
        assert!(!a.is_current());
        let a2 = a.clone();
        assert!(a.sync(move || a2.is_current()));
        let a3 = a.clone();
        assert!(a.sync(move || b.sync(move || a3.is_current())));
    }

    /// Through the Executor trait
    #[test]
    fn executor_trait() {
        let queue = q("q");
        let e: Arc<dyn Executor> = Arc::new(queue.clone());
        let (tx, rx) = mpsc::channel();
        e.exec_async(Box::new(move || {
            let _ = tx.send(current_label());
        }));
        assert_eq!(rx.recv().ok(), Some(Some("q")));
    }

    /// Blocks still queued when the last reference drops still run
    #[test]
    fn pending_blocks_run_after_release() {
        let threads: &dyn Threads = &StdThreads;
        let (tx, rx) = mpsc::channel();
        {
            let queue = SerialQueue::new(threads, "q");
            queue.exec_async(|| std::thread::sleep(Duration::from_millis(20)));
            queue.exec_async(move || {
                let _ = tx.send(());
            });
        }
        assert!(rx.recv_timeout(Duration::from_secs(5)).is_ok());
    }
}
