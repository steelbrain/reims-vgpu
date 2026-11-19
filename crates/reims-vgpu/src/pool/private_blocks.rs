//! The guest's contexts' Private blocks: each context (a process of the guest's) takes blocks of
//! GPU addresses for its Private objects and gives each back once it is done with it
//! (`PRIVATE_TAKE`, `PRIVATE_GIVE`, docs/graphics-kext/design.md, "Private blocks"). This is
//! their ledger: who holds which block, by its address.
//!
//! # A block
//!
//! On a source that gives Private blocks (metal2vulkan-rt's device), a take is one block of the
//! source's ([`BlockSource::take_block`] of [`MemoryClass::Private`]): device memory no one maps,
//! at the block's own GPU address, the whole of it the context's (the take's length rounded up to
//! rt's 64 KiB, carved from one of its 16 MiB allocations where it fits, else the start of an
//! allocation of its own: [`super::BLOCK_LEN`]). On a source with none (`HostBlocks`, the Vulkan
//! platform's), a block is a range of addresses the host picks ([`HostAddresses::private`],
//! `[2^44, 2^45)`) with no memory of its own, the take's length rounded up to [`TAKE_ALIGN`]: the
//! platform makes each object the guest places there as it makes one at the guest's own picks,
//! and pins it at its address. Nothing here zeroes anything: the decoder makes each Private
//! buffer zeroed at its create.
//!
//! # A take
//!
//! A take is carried out where it waits for nothing ([`PrivateBlocks::take`]): a pick, where the
//! source gives no Private blocks, is answered at once on the thread that carries the take out,
//! the context's lane. A source's block may be an allocation of the driver's, which waits for the
//! memory and, on NVIDIA's driver, holds the device's other calls while it allocates (about
//! 0.08 ms a MiB of Private memory, metal2vulkan-rt's `memory.rs`, `ALLOCATION_LEN`): on an RTX
//! 3090 a boot took 7 blocks of 256 MiB, 52.7 ms in all and 11.5 ms at most, each on the control
//! ring's lane, which held every context's next messages behind it (**measured**). So a take of
//! the source's goes to its context's worker (`reims-vgpu.graphics-private-take`), which takes
//! the block and answers: only the guest thread whose create asked for the block waits for it, as
//! Metal's create of that length allocates in its call. A context's takes are answered in its own
//! order, one worker a context with a take in flight, so a take waits only behind its own
//! context's earlier ones, never behind another's. A worker goes after [`WORKER_IDLE`] with no
//! take.
//!
//! A take ends only at its answer, its context's close, the device's reset or its release: there
//! is no timeout. A take whose context closed, or that the reset or the release ended, before its
//! answer is never answered, and the block its worker took goes back to the source.
//!
//! The device's degradation counts (docs/conventions.md) carry the takes by cause: those that
//! waited behind their context's earlier take ([`QUEUED_CODE`]), those taken from the source on
//! their worker ([`SOURCED_CODE`], and [`SOURCED_US_CODE`], the source's time for them, summed),
//! and the longest a take held the thread that carried it out ([`LANE_US_MOST_CODE`]).
//!
//! # A give, a close and a reset
//!
//! A give ([`PrivateBlocks::give`]) marks the block going, where no object of the host's lies in
//! it: the device answers it once the context's committed command buffers have finished, and
//! frees it then ([`PrivateBlocks::free`]). A context's close hands back every block it holds the
//! same way ([`PrivateBlocks::context_closed`]), freed once its objects are let go of and its
//! command buffers have finished. The device's reset frees every block at once
//! ([`PrivateBlocks::reset`]), its objects already dropped. A block is freed only by the number it
//! was taken under, so a free that comes after a reset freed it frees nothing taken since.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::{Condvar, Mutex, MutexGuard};
use reims_vgpu_vulkan::answer::{count_kind, count_kind_by};
use reims_vgpu_vulkan::exit_gate;

use super::addresses::HostAddresses;
use super::{BlockId, BlockSource, Live, MemoryClass, PoolNow, Threads};

/// A block's length is a multiple of this: 64 KiB, the wire's `RGFX_PRIVATE_BLOCK_ALIGN`.
pub const TAKE_ALIGN: u64 = 64 << 10;

/// The longest take: 1 TiB, the wire's `RGFX_PRIVATE_TAKE_MAX`.
pub const TAKE_MAX: u64 = 1 << 40;

/// How long a context's take worker waits for its next take before it goes.
const WORKER_IDLE: Duration = Duration::from_secs(2);

/// The code of a take that waited behind its context's earlier take, counted once each.
pub const QUEUED_CODE: &str = "private-take-queued";

/// The code of a take its worker took from the source, counted once each.
pub const SOURCED_CODE: &str = "private-take-sourced";

/// The code whose count is the source's time for the takes of [`SOURCED_CODE`], in
/// microseconds, summed.
pub const SOURCED_US_CODE: &str = "private-take-sourced-us";

/// The code whose count is the longest a take held the thread that carried it out, in
/// microseconds ([`PrivateBlocks::take`]'s own time, its answer's post among it where it answers
/// at once).
pub const LANE_US_MOST_CODE: &str = "private-take-lane-us-most";

/// Why a take got no block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TakeError {
    /// A length of 0 or past [`TAKE_MAX`].
    Invalid,
    /// No block or address left for it.
    Refused,
}

/// A take's answer: the block's GPU address and length, or why it got none. Called once, with no
/// lock of the ledger's held: at once on the thread that carried the take out, or from its
/// context's worker; never for a take that ended unanswered (the module doc's "A take").
pub(crate) type TakeAnswer = Box<dyn FnOnce(Result<(u64, u64), TakeError>) + Send>;

/// What came of a give ([`PrivateBlocks::give`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PrivateGive {
    /// The block is going, to be freed under this name.
    Given(HeldRef),
    /// The context holds no such block.
    NotFound,
    /// An object of the host's lies in it: it stays the context's.
    InUse,
}

/// A block as a give or a close names it until it is freed ([`PrivateBlocks::free`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct HeldRef {
    at: u64,
    serial: u64,
}

/// A context's block.
#[derive(Debug)]
struct Held {
    context: u32,
    len: u64,
    serial: u64,
    /// Given back or let go of at its context's close, freed once the device's answer is due.
    going: bool,
    /// The source's block it is, where the source gave one; `None` for the host's pick.
    block: Option<BlockId>,
}

/// A take waiting for its context's worker.
struct Job {
    /// Its context's takes' number when it was taken: the take is stale once they took another.
    life: u64,
    /// Its length, a multiple of [`TAKE_ALIGN`].
    len: u64,
    answer: TakeAnswer,
}

/// A context's takes.
struct Taker {
    /// Their number, which the context's close, the device's reset and its release change.
    life: u64,
    /// Its takes not yet taken up by its worker, in order.
    jobs: VecDeque<Job>,
    /// Whether its worker runs.
    working: bool,
    /// Whether its worker is carrying a take out.
    busy: bool,
}

/// What the ledger's lock holds.
struct State {
    /// The addresses picked for blocks where the source gives none.
    addresses: HostAddresses,
    /// The blocks held, by address.
    held: BTreeMap<u64, Held>,
    /// The number the next block, or the next life of a context's takes, is taken under.
    serial: u64,
    /// The contexts with a take in flight, by context.
    takers: HashMap<u32, Taker>,
    /// The device's release came ([`PrivateBlocks::close`]): no take is carried out from now on.
    closed: bool,
}

/// What the ledger's handles share.
struct Inner {
    /// The source of Private blocks, where it gives them.
    source: Option<Arc<dyn BlockSource>>,
    /// The platform's monotonic clock, which times the takes.
    now: PoolNow,
    state: Mutex<State>,
    /// Told when a take is handed to a worker, and when the takes end.
    changed: Condvar,
    /// The take workers' threads.
    threads: Arc<Threads>,
}

/// What a take worker's thread holds: the ledger, then its count, which a struct's fields' order
/// drops in that order, at the worker's return or unwind.
struct WorkerHold {
    inner: Arc<Inner>,
    _live: Live,
}

impl WorkerHold {
    /// Runs context `context`'s worker ([`Inner::work`]), then lets go of the ledger, then of
    /// the worker's count.
    fn run(self, context: u32) {
        self.inner.work(context);
    }
}

/// The contexts' Private blocks ([module doc](self)): a handle, which clones share.
#[derive(Clone)]
pub(crate) struct PrivateBlocks {
    inner: Arc<Inner>,
}

impl PrivateBlocks {
    /// A ledger with no block held, whose blocks are `source`'s, or the host's picks where it
    /// has none, its takes timed on `now`.
    pub(crate) fn new(source: Option<Arc<dyn BlockSource>>, now: PoolNow) -> PrivateBlocks {
        PrivateBlocks {
            inner: Arc::new(Inner {
                source,
                now,
                state: Mutex::new(State {
                    addresses: HostAddresses::private(),
                    held: BTreeMap::new(),
                    serial: 0,
                    takers: HashMap::new(),
                    closed: false,
                }),
                changed: Condvar::new(),
                threads: Arc::default(),
            }),
        }
    }

    /// Context `context`'s new block of at least `length` bytes (the module doc's "A take"),
    /// given to `answer`: its GPU address and its length, a multiple of [`TAKE_ALIGN`]: `length`
    /// rounded up to it, or the source's block's whole length rounded down to it. Answered at
    /// once for a length of 0 or past [`TAKE_MAX`] ([`TakeError::Invalid`]) and for the host's
    /// pick; else on the context's worker, after its earlier takes. A take of a valid length
    /// after the device's release is never answered.
    pub(crate) fn take(&self, context: u32, length: u64, answer: TakeAnswer) {
        let started = (self.inner.now)();
        self.carry_out(context, length, answer);
        let took = (self.inner.now)().saturating_duration_since(started);
        lane_time(took);
    }

    /// [`Self::take`]'s work, untimed.
    fn carry_out(&self, context: u32, length: u64, answer: TakeAnswer) {
        let Some(len) = length
            .checked_next_multiple_of(TAKE_ALIGN)
            .filter(|_| length != 0 && length <= TAKE_MAX)
        else {
            answer(Err(TakeError::Invalid));
            return;
        };
        let mut st = self.inner.state.lock();
        if st.closed {
            return;
        }
        if self.inner.source.is_none() {
            let picked = (st.addresses.pick(len))
                .map(|at| {
                    st.hold(at, context, len, None);
                    (at, len)
                })
                .ok_or(TakeError::Refused);
            drop(st);
            answer(picked);
            return;
        }
        let life = st.next_serial();
        let taker = st.takers.entry(context).or_insert(Taker {
            life,
            jobs: VecDeque::new(),
            working: false,
            busy: false,
        });
        if taker.busy || !taker.jobs.is_empty() {
            count_kind(QUEUED_CODE);
        }
        taker.jobs.push_back(Job {
            life: taker.life,
            len,
            answer,
        });
        if taker.working {
            drop(st);
            self.inner.changed.notify_all();
            return;
        }
        taker.working = true;
        // Counted under the lock, so that a release after the unlock waits for the thread.
        let hold = WorkerHold {
            inner: Arc::clone(&self.inner),
            _live: Live::new(&self.inner.threads),
        };
        drop(st);
        let spawned = std::thread::Builder::new()
            .name("reims-vgpu.graphics-private-take".into())
            .spawn(move || hold.run(context));
        if spawned.is_err() {
            // The hold went with the thread's closure.
            log::warn!("private blocks: no thread for context {context}'s takes: they are refused");
            let mut st = self.inner.state.lock();
            let jobs = match st.takers.remove(&context) {
                Some(taker) => taker.jobs,
                None => VecDeque::new(),
            };
            drop(st);
            self.inner.changed.notify_all();
            for job in jobs {
                (job.answer)(Err(TakeError::Refused));
            }
        }
    }

    /// Context `context` gives back its block of `length` bytes at `gpu_address`: not found
    /// where it holds none there of that length, or holds it going already; in use where
    /// `in_use(gpu_address, length)` finds an object of the host's in it, and the block stays the
    /// context's; else going, to be freed under the name returned.
    pub(crate) fn give(
        &self,
        context: u32,
        gpu_address: u64,
        length: u64,
        in_use: impl FnOnce(u64, u64) -> bool,
    ) -> PrivateGive {
        let mut state = self.inner.state.lock();
        let Some(held) = state.held.get_mut(&gpu_address) else {
            return PrivateGive::NotFound;
        };
        if held.context != context || held.len != length || held.going {
            return PrivateGive::NotFound;
        }
        if in_use(gpu_address, length) {
            return PrivateGive::InUse;
        }
        held.going = true;
        PrivateGive::Given(HeldRef {
            at: gpu_address,
            serial: held.serial,
        })
    }

    /// Context `context` closed: its takes not yet answered end unanswered, and every block it
    /// holds is going from now on, to be freed once its objects are let go of and its command
    /// buffers have finished ([`Self::free`]).
    pub(crate) fn context_closed(&self, context: u32) -> Vec<HeldRef> {
        let mut state = self.inner.state.lock();
        let ended = state.end_takes(Some(context));
        let gone = (state.held.iter_mut())
            .filter(|(_, held)| held.context == context && !held.going)
            .map(|(&at, held)| {
                held.going = true;
                HeldRef {
                    at,
                    serial: held.serial,
                }
            })
            .collect();
        drop(state);
        self.inner.changed.notify_all();
        drop(ended);
        gone
    }

    /// Frees `blocks`: each that is still held under its name goes back to the source, or its
    /// address to the picks.
    pub(crate) fn free(&self, blocks: &[HeldRef]) {
        let mut released = Vec::new();
        {
            let mut state = self.inner.state.lock();
            for name in blocks {
                if state
                    .held
                    .get(&name.at)
                    .is_none_or(|held| held.serial != name.serial)
                {
                    continue;
                }
                let held = state.held.remove(&name.at);
                match held.and_then(|held| held.block) {
                    Some(block) => released.push(block),
                    None => state.addresses.free(name.at),
                }
            }
        }
        self.inner.release(released);
    }

    /// The device's reset: every take not yet answered ends unanswered, and every block is freed
    /// at once, its objects already dropped.
    pub(crate) fn reset(&self) {
        let (released, ended) = {
            let mut state = self.inner.state.lock();
            let ended = state.end_takes(None);
            let held = std::mem::take(&mut state.held);
            let mut released = Vec::new();
            for (at, held) in held {
                match held.block {
                    Some(block) => released.push(block),
                    None => state.addresses.free(at),
                }
            }
            (released, ended)
        };
        self.inner.changed.notify_all();
        drop(ended);
        self.inner.release(released);
    }

    /// The device's release: every take not yet answered ends unanswered, and none is carried
    /// out from now on. Waits for the take workers to go, each giving back the block it was
    /// taking, so that none calls the source past the release: at most one source's take each.
    /// Each worker's thread has let go of the ledger once this returns, so the source is
    /// dropped where the ledger's last handle goes, never on a worker's thread ([`Threads`]).
    /// The blocks the contexts hold stay held, and go back with the source.
    pub(crate) fn close(&self) {
        let ended = {
            let mut state = self.inner.state.lock();
            state.closed = true;
            state.end_takes(None)
        };
        self.inner.changed.notify_all();
        drop(ended);
        let threads = &self.inner.threads;
        let mut live = threads.live.lock();
        while *live > 0 {
            threads.gone.wait(&mut live);
        }
    }

    /// The blocks held, by address: their contexts, lengths and whether each is going (for the
    /// tests).
    #[cfg(test)]
    pub(crate) fn held(&self) -> Vec<(u64, u32, u64, bool)> {
        let state = self.inner.state.lock();
        (state.held.iter())
            .map(|(&at, held)| (at, held.context, held.len, held.going))
            .collect()
    }

    /// The take workers' threads that hold the ledger (for the tests).
    #[cfg(test)]
    pub(crate) fn workers(&self) -> usize {
        *self.inner.threads.live.lock()
    }
}

impl State {
    /// A new number, for a block or a life of a context's takes.
    fn next_serial(&mut self) -> u64 {
        self.serial = self.serial.wrapping_add(1);
        self.serial
    }

    /// Context `context` holds the block of `len` bytes at `at`, the source's `block` or the
    /// host's pick.
    fn hold(&mut self, at: u64, context: u32, len: u64, block: Option<BlockId>) {
        let serial = self.next_serial();
        self.held.insert(
            at,
            Held {
                context,
                len,
                serial,
                going: false,
                block,
            },
        );
    }

    /// Whether the takes of `context` numbered `life` still stand: neither its close nor the
    /// device's reset or release came since.
    fn stands(&self, context: u32, life: u64) -> bool {
        !self.closed && self.takers.get(&context).is_some_and(|t| t.life == life)
    }

    /// Ends the takes of `context`, or of every context, not yet answered: their contexts' takes
    /// take a new number, so the take a worker carries out is stale, and those waiting for the
    /// worker are dropped here. Their answers are returned, to drop with no lock held.
    fn end_takes(&mut self, context: Option<u32>) -> Vec<TakeAnswer> {
        let contexts: Vec<u32> = match context {
            Some(context) => vec![context],
            None => self.takers.keys().copied().collect(),
        };
        let mut ended = Vec::new();
        for context in contexts {
            let life = self.next_serial();
            let Some(taker) = self.takers.get_mut(&context) else {
                continue;
            };
            taker.life = life;
            let jobs = std::mem::take(&mut taker.jobs);
            if !taker.working {
                self.takers.remove(&context);
            }
            ended.extend(jobs.into_iter().map(|job| job.answer));
        }
        ended
    }
}

impl Inner {
    /// Gives `blocks` back to the source, with no lock of the ledger's held.
    fn release(&self, blocks: Vec<BlockId>) {
        if let Some(source) = &self.source {
            for block in blocks {
                // The ledger gives each block back once, as it took it.
                let _ = source.release_block(block);
            }
        }
    }

    /// Context `context`'s take worker: takes up its takes in order, takes each one's block from
    /// the source and answers, until its takes stay none for [`WORKER_IDLE`] or the device's
    /// release comes.
    fn work(&self, context: u32) {
        let mut st = self.state.lock();
        let mut idle_since = (self.now)();
        loop {
            if st.closed {
                break;
            }
            let job = st.takers.get_mut(&context).and_then(|t| t.jobs.pop_front());
            let Some(job) = job else {
                let idle = (self.now)().saturating_duration_since(idle_since);
                match WORKER_IDLE.checked_sub(idle).filter(|left| !left.is_zero()) {
                    Some(left) => {
                        let _ = self.changed.wait_for(&mut st, left);
                        continue;
                    }
                    None => break,
                }
            };
            if let Some(taker) = st.takers.get_mut(&context) {
                taker.busy = true;
            }
            let (taken, took) = MutexGuard::unlocked(&mut st, || self.take_block(job.len));
            if let Some(taker) = st.takers.get_mut(&context) {
                taker.busy = false;
            }
            let stands = st.stands(context, job.life);
            let answer = match taken {
                Ok((block, at, whole)) if stands => {
                    st.hold(at, context, whole, Some(block));
                    Some(Ok((at, whole)))
                }
                Ok((block, ..)) => {
                    MutexGuard::unlocked(&mut st, || self.release(vec![block]));
                    None
                }
                Err(error) => stands.then_some(Err(error)),
            };
            match answer {
                Some(answer) => MutexGuard::unlocked(&mut st, || {
                    (job.answer)(answer);
                    if answer.is_ok() {
                        sourced(context, job.len, took);
                    }
                }),
                None => MutexGuard::unlocked(&mut st, || drop(job.answer)),
            }
            idle_since = (self.now)();
        }
        // Gone: a take handed to it after this starts another.
        let closed = st.closed;
        if let Some(taker) = st.takers.get_mut(&context) {
            taker.working = false;
            if closed {
                taker.jobs.clear();
            }
            if taker.jobs.is_empty() {
                st.takers.remove(&context);
            }
        }
    }

    /// The source's block of at least `len` bytes, as a step of the library's (it may call the
    /// driver), and how long the source took: its number, GPU address and whole length rounded
    /// down to [`TAKE_ALIGN`]. Refused where the source refuses, or gives a block no take could
    /// hold, which goes back.
    fn take_block(&self, len: u64) -> (Result<(BlockId, u64, u64), TakeError>, Duration) {
        let Some(_step) = exit_gate::step() else {
            // The process is exiting: the worker parks for good, its hold kept, gone for the
            // release's wait.
            self.threads.go();
            exit_gate::stay()
        };
        let Some(source) = &self.source else {
            return (Err(TakeError::Refused), Duration::ZERO);
        };
        let started = (self.now)();
        let taken = source.take_block(MemoryClass::Private, len);
        let took = (self.now)().saturating_duration_since(started);
        let block = match taken {
            Ok(info) if info.gpu_address == 0 || info.len < len => {
                let _ = source.release_block(info.id);
                Err(TakeError::Refused)
            }
            Ok(info) => {
                // `len` is a multiple of TAKE_ALIGN no longer than the block's length.
                let whole = (info.len.checked_rem(TAKE_ALIGN))
                    .and_then(|rest| info.len.checked_sub(rest))
                    .unwrap_or(len);
                Ok((info.id, info.gpu_address, whole))
            }
            Err(_) => Err(TakeError::Refused),
        };
        (block, took)
    }
}

/// Counts a take of `len` bytes for `context` its worker took from the source in `took`, with
/// the first such take's line in every boot's log.
fn sourced(context: u32, len: u64, took: Duration) {
    count_kind(SOURCED_CODE);
    count_kind_by(SOURCED_US_CODE, micros(took));
    log::info!(
        kind:% = SOURCED_CODE;
        "private blocks: context {context}'s take of {} MiB took {:.1} ms from the source, on \
         its worker",
        len >> 20,
        took.as_secs_f64() * 1000.0
    );
}

/// The longest a take held the thread that carried it out, in microseconds, counted as
/// [`LANE_US_MOST_CODE`]'s rise to it.
static LANE_US_MOST: AtomicU64 = AtomicU64::new(0);

/// Counts a take that held the thread that carried it out for `took`, where it is the longest
/// yet.
fn lane_time(took: Duration) {
    let us = micros(took).max(1);
    let before = LANE_US_MOST.fetch_max(us, Ordering::Relaxed);
    if us > before {
        count_kind_by(LANE_US_MOST_CODE, us.saturating_sub(before));
    }
}

/// `d` in whole microseconds, saturated.
fn micros(d: Duration) -> u64 {
    u64::try_from(d.as_micros()).unwrap_or(u64::MAX)
}

#[cfg(test)]
// Test scaffolding: real time bounds the waits, and helper threads stand in for the device's.
#[allow(clippy::disallowed_methods)]
mod tests {
    use std::sync::LazyLock;
    use std::sync::atomic::AtomicU32;
    use std::sync::mpsc::{Receiver, RecvTimeoutError};
    use std::time::Instant;

    use super::super::addresses::{PRIVATE_BASE, PRIVATE_END};
    use super::super::{BlockError, BlockInfo, MonoTime};
    use super::*;

    /// The length of the takes a [`Source`] holds while its gate is shut: 16 MiB.
    const GATED: u64 = 16 << 20;

    /// The worker's thread's name.
    const WORKER: &str = "reims-vgpu.graphics-private-take";

    /// A source of Private blocks at addresses from 2^40, a block of `len` bytes rounded up to
    /// 1 MiB, plus 1 KiB, at the next multiple of 1 GiB, which records what it hands out and is
    /// given back; a take of [`GATED`] bytes waits while its gate is shut.
    #[derive(Default)]
    struct Source {
        next: AtomicU32,
        out: Mutex<Vec<BlockId>>,
        back: Mutex<Vec<BlockId>>,
        refuse: std::sync::atomic::AtomicBool,
        gate: Mutex<Gate>,
        changed: Condvar,
    }

    /// A [`Source`]'s gate: whether it is shut, and how many takes wait at it.
    #[derive(Default)]
    struct Gate {
        shut: bool,
        waiting: usize,
    }

    impl Source {
        fn shut() -> Arc<Source> {
            let source = Arc::new(Source::default());
            source.gate.lock().shut = true;
            source
        }

        fn open(&self) {
            self.gate.lock().shut = false;
            self.changed.notify_all();
        }

        /// Once `n` takes wait at the gate, or the test fails after 10 s.
        fn wait_waiting(&self, n: usize) {
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut gate = self.gate.lock();
            while gate.waiting < n {
                assert!(!self.changed.wait_until(&mut gate, deadline).timed_out());
            }
        }

        /// Once `n` blocks are back, or the test fails after 10 s.
        fn wait_back(&self, n: usize) -> Vec<BlockId> {
            let deadline = Instant::now() + Duration::from_secs(10);
            loop {
                let back = self.back.lock().clone();
                if back.len() >= n {
                    return back;
                }
                assert!(Instant::now() < deadline, "{back:?} back, not {n}");
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }

    impl BlockSource for Source {
        fn take_block(&self, class: MemoryClass, len: u64) -> Result<BlockInfo, BlockError> {
            assert_eq!(class, MemoryClass::Private);
            if len == GATED {
                let mut gate = self.gate.lock();
                gate.waiting += 1;
                self.changed.notify_all();
                while gate.shut {
                    self.changed.wait(&mut gate);
                }
                gate.waiting -= 1;
            }
            if self.refuse.load(Ordering::Relaxed) {
                return Err(BlockError::OutOfMemory);
            }
            let n = self.next.fetch_add(1, Ordering::Relaxed) + 1;
            let id = BlockId(n);
            self.out.lock().push(id);
            Ok(BlockInfo {
                id,
                class,
                len: len.next_multiple_of(1 << 20) + 0x400,
                gpu_address: (1 << 40) + u64::from(n) * (1 << 30),
                contents: None,
                zeroed: false,
            })
        }
        fn release_block(&self, block: BlockId) -> Result<(), BlockError> {
            self.back.lock().push(block);
            Ok(())
        }
    }

    /// The tests' monotonic clock: nanoseconds since its first reading.
    fn now() -> PoolNow {
        static START: LazyLock<Instant> = LazyLock::new(Instant::now);
        Box::new(|| MonoTime::from_nanos(u64::try_from(START.elapsed().as_nanos()).unwrap()))
    }

    fn ledger(source: Option<Arc<Source>>) -> PrivateBlocks {
        PrivateBlocks::new(source.map(|s| s as Arc<dyn BlockSource>), now())
    }

    /// A take's answer and the name of the thread that gave it.
    type Got = Receiver<(Result<(u64, u64), TakeError>, Option<String>)>;

    /// Context `context`'s take of `length` bytes, its answer to come on the receiver, which
    /// disconnects where the take ends unanswered.
    fn take(blocks: &PrivateBlocks, context: u32, length: u64) -> Got {
        let (tx, rx) = std::sync::mpsc::channel();
        blocks.take(
            context,
            length,
            Box::new(move |taken| {
                let name = std::thread::current().name().map(str::to_owned);
                let _ = tx.send((taken, name));
            }),
        );
        rx
    }

    /// The take's answer, given before the take returned.
    fn at_once(blocks: &PrivateBlocks, context: u32, length: u64) -> Result<(u64, u64), TakeError> {
        let (taken, name) = (take(blocks, context, length).try_recv()).expect("answered at once");
        assert_ne!(name.as_deref(), Some(WORKER));
        taken
    }

    /// The take's answer, from its context's worker, within 10 s.
    fn answered(got: &Got) -> Result<(u64, u64), TakeError> {
        let (taken, name) = got.recv_timeout(Duration::from_secs(10)).expect("answered");
        assert_eq!(name.as_deref(), Some(WORKER));
        taken
    }

    /// The take ended unanswered: its answer was dropped, never called, within 10 s.
    fn unanswered(got: &Got) {
        match got.recv_timeout(Duration::from_secs(10)) {
            Err(RecvTimeoutError::Disconnected) => {}
            other => panic!("answered or not ended: {other:?}"),
        }
    }

    fn counted(code: &str) -> u64 {
        (reims_vgpu_vulkan::answer::counted_kinds().into_iter())
            .find(|&(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// With no source, a take is the host's pick in the Private range, its length rounded up to
    /// 64 KiB, answered at once on the thread that carried it out; a length of 0 or past 1 TiB
    /// is invalid, at once; and each block is the taker's alone. No worker runs.
    #[test]
    fn a_take_with_no_source_is_a_pick_in_the_private_range() {
        let blocks = ledger(None);
        assert_eq!(at_once(&blocks, 3, 0x1000), Ok((PRIVATE_BASE, 0x1_0000)));
        assert_eq!(
            at_once(&blocks, 4, 0x10_0000),
            Ok((PRIVATE_BASE + 0x1_8000, 0x10_0000))
        );
        assert_eq!(at_once(&blocks, 3, 0), Err(TakeError::Invalid));
        assert_eq!(at_once(&blocks, 3, TAKE_MAX + 1), Err(TakeError::Invalid));
        assert_eq!(
            blocks.held(),
            [
                (PRIVATE_BASE, 3, 0x1_0000, false),
                (PRIVATE_BASE + 0x1_8000, 4, 0x10_0000, false)
            ]
        );
        assert!(blocks.held().iter().all(|&(at, ..)| at < PRIVATE_END));
        assert_eq!(blocks.workers(), 0);
        assert_eq!(counted(SOURCED_CODE), 0);
        assert!(counted(LANE_US_MOST_CODE) > 0);
    }

    /// A give names the block by its address and length, and only its context's: another's, an
    /// unknown address or a wrong length is not found, and so is one going already. A block an
    /// object lies in stays held. A given block is freed only at its free, and its address is
    /// picked again only once the picks wrap.
    #[test]
    fn a_give_frees_the_block_only_at_its_free_and_only_where_no_object_lies_in_it() {
        let blocks = ledger(None);
        let (at, len) = at_once(&blocks, 3, 0x1_0000).unwrap();
        assert_eq!(blocks.give(4, at, len, |_, _| false), PrivateGive::NotFound);
        assert_eq!(
            blocks.give(3, at + 0x8000, len, |_, _| false),
            PrivateGive::NotFound
        );
        assert_eq!(
            blocks.give(3, at, 0x2_0000, |_, _| false),
            PrivateGive::NotFound
        );
        let mut asked = None;
        assert_eq!(
            blocks.give(3, at, len, |a, l| {
                asked = Some((a, l));
                true
            }),
            PrivateGive::InUse
        );
        assert_eq!(asked, Some((at, len)));
        assert_eq!(blocks.held(), [(at, 3, len, false)]);
        let PrivateGive::Given(name) = blocks.give(3, at, len, |_, _| false) else {
            panic!("given");
        };
        assert_eq!(blocks.held(), [(at, 3, len, true)]);
        assert_eq!(blocks.give(3, at, len, |_, _| false), PrivateGive::NotFound);
        blocks.free(&[name]);
        assert!(blocks.held().is_empty());
        // Freed twice frees nothing.
        blocks.free(&[name]);
        assert_ne!(at_once(&blocks, 3, 0x1_0000).unwrap().0, at);
    }

    /// A close hands back every block of the context not going already, and frees them at its
    /// free; the device's reset frees every block at once, and a free of a block the reset freed
    /// frees nothing taken since.
    #[test]
    fn a_close_hands_back_the_context_s_blocks_and_a_reset_frees_every_block() {
        let blocks = ledger(None);
        let (a, _) = at_once(&blocks, 3, 0x1_0000).unwrap();
        let (b, _) = at_once(&blocks, 3, 0x2_0000).unwrap();
        let (c, _) = at_once(&blocks, 4, 0x1_0000).unwrap();
        let PrivateGive::Given(given) = blocks.give(3, a, 0x1_0000, |_, _| false) else {
            panic!("given");
        };
        let closed = blocks.context_closed(3);
        assert_eq!(closed.len(), 1, "only the block not going already");
        assert_eq!(
            blocks.held(),
            [
                (a, 3, 0x1_0000, true),
                (b, 3, 0x2_0000, true),
                (c, 4, 0x1_0000, false)
            ]
        );
        blocks.free(&closed);
        blocks.free(&[given]);
        assert_eq!(blocks.held(), [(c, 4, 0x1_0000, false)]);
        let gone = blocks.context_closed(4);
        blocks.reset();
        assert!(blocks.held().is_empty());
        let (d, _) = at_once(&blocks, 5, 0x1_0000).unwrap();
        blocks.free(&gone);
        assert_eq!(blocks.held(), [(d, 5, 0x1_0000, false)]);
    }

    /// With a source, a take is one of the source's Private blocks, at its address, as long as
    /// the block rounded down to 64 KiB, answered on its context's worker; a refusal of the
    /// source's is the take's; an invalid length is answered at once; a block is given back to
    /// the source at its free and at a reset, and never twice.
    #[test]
    fn a_take_with_a_source_is_its_whole_block_at_its_address() {
        let source = Arc::new(Source::default());
        let blocks = ledger(Some(source.clone()));
        let (a, len) = answered(&take(&blocks, 3, 0x2_1000)).unwrap();
        assert_eq!((a, len), ((1 << 40) + (1 << 30), 0x10_0000));
        let (b, _) = answered(&take(&blocks, 3, 0x1_0000)).unwrap();
        assert_eq!(b, (1 << 40) + (2 << 30));
        assert_eq!(at_once(&blocks, 3, 0), Err(TakeError::Invalid));
        source.refuse.store(true, Ordering::Relaxed);
        assert_eq!(
            answered(&take(&blocks, 3, 0x1_0000)),
            Err(TakeError::Refused)
        );
        let PrivateGive::Given(name) = blocks.give(3, a, len, |_, _| false) else {
            panic!("given");
        };
        assert!(source.back.lock().is_empty(), "not before its free");
        blocks.free(&[name]);
        assert_eq!(*source.back.lock(), [BlockId(1)]);
        blocks.free(&[name]);
        blocks.reset();
        assert_eq!(*source.back.lock(), [BlockId(1), BlockId(2)]);
        assert!(blocks.held().is_empty());
        assert_eq!(
            counted(SOURCED_CODE),
            2,
            "the takes the source gave a block"
        );
    }

    /// A take of the source's holds the thread that carried it out for none of the source's
    /// time: it returns while the source waits, and the worker answers once the source gives
    /// the block; counted, with the source's time, and the lane's longest stays short.
    #[test]
    fn a_take_from_the_source_waits_on_its_worker_never_on_the_lane() {
        let source = Source::shut();
        let blocks = ledger(Some(source.clone()));
        let got = take(&blocks, 3, GATED);
        source.wait_waiting(1);
        assert_eq!(got.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty));
        std::thread::sleep(Duration::from_millis(50));
        source.open();
        let (at, len) = answered(&got).unwrap();
        assert_eq!((at, len), ((1 << 40) + (1 << 30), GATED));
        assert_eq!(blocks.held(), [(at, 3, len, false)]);
        assert_eq!(counted(SOURCED_CODE), 1);
        assert!(
            counted(SOURCED_US_CODE) >= 50_000,
            "the source's time, summed"
        );
        let lane = counted(LANE_US_MOST_CODE);
        assert!((1..50_000).contains(&lane), "the lane's longest: {lane} us");
    }

    /// A context's takes are answered in its order: a take behind its context's earlier one
    /// waits for it, counted, while another context's take waits for neither.
    #[test]
    fn a_context_s_takes_are_answered_in_its_order_and_another_s_wait_for_none_of_them() {
        let source = Source::shut();
        let blocks = ledger(Some(source.clone()));
        let first = take(&blocks, 3, GATED);
        source.wait_waiting(1);
        let second = take(&blocks, 3, 0x1_0000);
        let other = take(&blocks, 4, 0x1_0000);
        assert_eq!(answered(&other).unwrap().0, (1 << 40) + (1 << 30));
        assert_eq!(second.try_recv(), Err(std::sync::mpsc::TryRecvError::Empty));
        assert_eq!(
            counted(QUEUED_CODE),
            1,
            "only the take behind its context's"
        );
        source.open();
        assert_eq!(answered(&first).unwrap().0, (1 << 40) + (2 << 30));
        assert_eq!(answered(&second).unwrap().0, (1 << 40) + (3 << 30));
        assert_eq!(blocks.workers(), 2);
    }

    /// A take its context's close or the device's reset ended is never answered: the block its
    /// worker was taking goes back to the source, and the takes queued behind it are dropped.
    /// A take of the context's after the close is answered as any.
    #[test]
    fn a_take_its_context_closed_or_the_reset_ended_is_never_answered_and_its_block_goes_back() {
        let source = Source::shut();
        let blocks = ledger(Some(source.clone()));
        let taking = take(&blocks, 3, GATED);
        source.wait_waiting(1);
        let queued = take(&blocks, 3, 0x1_0000);
        assert!(blocks.context_closed(3).is_empty());
        unanswered(&queued);
        source.open();
        unanswered(&taking);
        assert_eq!(source.wait_back(1), [BlockId(1)]);
        assert!(blocks.held().is_empty());
        let (after, _) = answered(&take(&blocks, 3, 0x1_0000)).unwrap();
        assert_eq!(blocks.held().len(), 1);

        source.gate.lock().shut = true;
        let taking = take(&blocks, 5, GATED);
        source.wait_waiting(1);
        blocks.reset();
        source.open();
        unanswered(&taking);
        let back = source.wait_back(3);
        assert_eq!(back.len(), 3, "the reset's and the stale take's: {back:?}");
        assert!(blocks.held().is_empty(), "{after:#x} freed at the reset");
        assert_eq!(counted(SOURCED_CODE), 1);
    }

    /// The device's release waits for the workers to go: the take a worker was carrying out
    /// ends unanswered and its block goes back before the release returns; a take after it is
    /// never answered and starts no worker. The blocks the contexts hold stay held.
    #[test]
    fn the_release_waits_for_the_workers_and_gives_back_the_block_in_flight() {
        let source = Source::shut();
        let blocks = ledger(Some(source.clone()));
        let (held, len) = answered(&take(&blocks, 4, 0x1_0000)).unwrap();
        let taking = take(&blocks, 3, GATED);
        source.wait_waiting(1);
        let opener = {
            let source = Arc::clone(&source);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(50));
                source.open();
            })
        };
        blocks.close();
        assert_eq!(
            *source.back.lock(),
            [BlockId(2)],
            "back before the release returned"
        );
        assert_eq!(blocks.workers(), 0);
        unanswered(&taking);
        unanswered(&take(&blocks, 3, 0x1_0000));
        assert_eq!(blocks.workers(), 0);
        assert_eq!(blocks.held(), [(held, 4, len, false)]);
        opener.join().unwrap();
    }

    /// The device's release returns only once every worker's thread has let go of the ledger:
    /// the ledger's last handle is then the caller's, and the source drops with it on the
    /// caller's thread, never on a worker's past the release.
    #[test]
    fn the_release_returns_once_no_worker_holds_the_ledger() {
        for _ in 0..50 {
            let source = Arc::new(Source::default());
            let weak = Arc::downgrade(&source);
            let blocks = ledger(Some(source));
            let takes: Vec<Got> = (1..=4)
                .map(|context| take(&blocks, context, 0x1_0000))
                .collect();
            for got in &takes {
                answered(got).unwrap();
            }
            blocks.close();
            assert_eq!(
                Arc::strong_count(&blocks.inner),
                1,
                "a worker's thread still holds the ledger"
            );
            drop(blocks);
            assert!(weak.upgrade().is_none(), "the source outlived the ledger");
        }
    }

    /// A set of known blocks that keeps none.
    struct Unknown;

    impl super::super::rt_blocks::Knowing for Unknown {
        fn know(&self, _: metal2vulkan_rt::Block) {}
        fn forget(&self, _: metal2vulkan_rt::BlockId) {}
    }

    /// On rt's device, a take is one of rt's Private blocks at its own address, answered on its
    /// context's worker, a take longer than rt's unit among them; a block goes back to rt at its
    /// free, and the device's release lets every worker go.
    #[test]
    #[cfg_attr(
        target_os = "macos",
        ignore = "MoltenVK: rt's floor refuses it (no nullDescriptor), so no rt device on the Mac before KosmicKrisp at the cutover; ledger: rt on the Mac"
    )]
    fn on_vulkan_rt_a_private_take_is_an_rt_block_answered_on_its_worker() {
        let (device, name) = reims_vgpu_vulkan::seam::test_device().unwrap();
        println!("rt's device: {name}");
        let source = super::super::rt_blocks::RtBlocks::new(device, Box::new(Unknown));
        let blocks = PrivateBlocks::new(Some(Arc::new(source)), now());
        let (a, a_len) = answered(&take(&blocks, 3, 0x1_8000)).unwrap();
        assert_ne!(a, 0);
        assert!(
            a_len >= 0x2_0000 && a_len.is_multiple_of(TAKE_ALIGN),
            "{a_len:#x}"
        );
        let (b, b_len) = answered(&take(&blocks, 3, 64 << 20)).unwrap();
        assert!(b_len >= 64 << 20, "{b_len:#x}");
        assert!(
            a + a_len <= b || b + b_len <= a,
            "{a:#x}+{a_len:#x} and {b:#x}+{b_len:#x}"
        );
        let (c, _) = answered(&take(&blocks, 4, 0x1_0000)).unwrap();
        assert_eq!(blocks.held().len(), 3);
        let PrivateGive::Given(given) = blocks.give(3, a, a_len, |_, _| false) else {
            panic!("given");
        };
        blocks.free(&[given]);
        assert_eq!(blocks.held().len(), 2);
        assert!(blocks.held().iter().any(|&(at, ..)| at == c));
        blocks.close();
        assert_eq!(blocks.workers(), 0);
        assert_eq!(counted(SOURCED_CODE), 3);
        println!(
            "the source's time for the takes: {} us",
            counted(SOURCED_US_CODE)
        );
    }

    /// A worker goes once its context has had no take for its idle time, and a later take
    /// starts another.
    #[test]
    fn an_idle_worker_goes_and_a_later_take_starts_another() {
        let blocks = ledger(Some(Arc::new(Source::default())));
        answered(&take(&blocks, 3, 0x1_0000)).unwrap();
        let deadline = Instant::now() + WORKER_IDLE + Duration::from_secs(10);
        while blocks.workers() > 0 {
            assert!(Instant::now() < deadline, "the worker stayed");
            std::thread::sleep(Duration::from_millis(20));
        }
        answered(&take(&blocks, 3, 0x1_0000)).unwrap();
        assert_eq!(blocks.held().len(), 2);
    }
}
