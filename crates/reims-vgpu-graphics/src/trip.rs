//! The trip of a command buffer's completion down to the guest, and of the guest's next buffer up
//! to rt, timed by stage: a diagnostic, off until [`turn_on`], which reims-vgpu calls where its
//! log names its trip line's target at debug (`reims_vgpu::trip`).
//!
//! Each stage's spans go into a histogram of [`BUCKETS`] buckets with their sum, and each event
//! counted into a count, all relaxed atomics: a stage costs two reads of the device's clock and
//! three atomic adds, with nothing allocated and no lock taken. Off, each site costs one relaxed
//! load. [`take`] reads every histogram and count since the take before, and zeroes them.
//!
//! The stages, each on the device's clock ([`crate::clock`]), which the guest's times map to:
//! - down, from rt's completed handler to the guest's interrupt: [`Stage::HandlerHop`] (rt's
//!   handler entered, to the decoder's), [`Stage::Ours`] (the decoder's handler to its turn of its
//!   queue's completion posts finished), [`Stage::Order`] (that turn to its post's run),
//!   [`Stage::Post`] (the post to the record on the ring), [`Stage::Raise`] (the interrupt's
//!   raise), and [`Stage::Down`], rt's handler entered to the interrupt raised (or the record
//!   written where the guest's ring was not armed);
//! - the guest's turn, where its buffers carry their times ([`crate::wire`]'s
//!   `RGFX_FEATURE_GUEST_TIMES`): [`Stage::Gap`], a queue's last completion's interrupt to its
//!   next buffer's take with none of the queue's buffers in flight, split by the guest's stamps
//!   into [`Stage::RaiseToBegan`] and [`Stage::RaiseToCommit`] (to the next buffer's make and
//!   to its commit call), [`Stage::Encode`] (make to commit call), [`Stage::PostWait`] (commit
//!   call to the post on the ring, the bundle's own), [`Stage::Queued`], and the transport (the
//!   post, or the thread's return to the ring, to the take), [`Stage::TransportWoken`] for a
//!   take the queue's thread was woken for and [`Stage::TransportAwake`] for one it found awake;
//!   and [`Stage::CommitToTake`], the commit call to the take;
//! - up, from the take to rt's commit: [`Stage::Woke`] (the thread's wake to its take),
//!   [`Stage::Gate`] (the take, through ring 0's gate, to the host), [`Stage::Begin`] (the host's
//!   begin to the walk's start), [`Stage::LaneHop`] (the hand to the lane's thread to its start),
//!   [`Stage::Walk`] (the walk's start to the platform's commit), [`Stage::Commit`] (that commit
//!   to rt's), [`Stage::RtCommit`] (rt's commit call), and [`Stage::Up`], the take to rt's commit.
//!
//! Nothing here changes what the device does, or anything the guest or the host observes.

use std::cell::Cell;
use std::fmt::{self, Write as _};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::Relaxed};

/// The histograms' buckets: one under 64 ns, then eight a power of two from 64 ns, the last
/// holding every span of 2^31 ns (about 2.1 s) or more ([`bucket`]).
pub const BUCKETS: usize = 202;

/// The stages timed (the module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// rt's completed handler entered, to the decoder's.
    HandlerHop,
    /// The decoder's completed handler, to its turn finished.
    Ours,
    /// The turn finished, to its post's run.
    Order,
    /// The post, to the record written on the guest's ring.
    Post,
    /// The interrupt's raise.
    Raise,
    /// rt's completed handler entered, to the interrupt raised or the record written.
    Down,
    /// A queue's last completion's interrupt, to its next buffer's take with none in flight.
    Gap,
    /// That interrupt, to the next buffer's make in the guest.
    RaiseToBegan,
    /// That interrupt, to the next buffer's commit call in the guest.
    RaiseToCommit,
    /// The guest's make of a buffer to its commit call.
    Encode,
    /// The guest's commit call to its post on the ring.
    PostWait,
    /// The post to the thread's return to the ring, behind the buffer before it.
    Queued,
    /// The post (or the return) to the take, for a take the thread was woken for.
    TransportWoken,
    /// The post (or the return) to the take, for a take the thread found awake.
    TransportAwake,
    /// The guest's commit call to the take.
    CommitToTake,
    /// The queue's thread's wake, to its first take after it.
    Woke,
    /// The take, through ring 0's gate, to the host's begin.
    Gate,
    /// The host's begin, to the walk's start.
    Begin,
    /// The hand to the lane's thread, to the walk's start there.
    LaneHop,
    /// The walk's start, to the platform's commit.
    Walk,
    /// The platform's commit, to rt's.
    Commit,
    /// rt's commit call.
    RtCommit,
    /// The take, to rt's commit.
    Up,
}

/// Every stage, in the line's order.
pub const STAGES: [Stage; 23] = [
    Stage::HandlerHop,
    Stage::Ours,
    Stage::Order,
    Stage::Post,
    Stage::Raise,
    Stage::Down,
    Stage::Gap,
    Stage::RaiseToBegan,
    Stage::RaiseToCommit,
    Stage::Encode,
    Stage::PostWait,
    Stage::Queued,
    Stage::TransportWoken,
    Stage::TransportAwake,
    Stage::CommitToTake,
    Stage::Woke,
    Stage::Gate,
    Stage::Begin,
    Stage::LaneHop,
    Stage::Walk,
    Stage::Commit,
    Stage::RtCommit,
    Stage::Up,
];

impl Stage {
    /// Its name in the line.
    pub fn name(self) -> &'static str {
        match self {
            Stage::HandlerHop => "d2_handler",
            Stage::Ours => "d3_ours",
            Stage::Order => "d4_order",
            Stage::Post => "d5_post",
            Stage::Raise => "d6_raise",
            Stage::Down => "down",
            Stage::Gap => "gap",
            Stage::RaiseToBegan => "g_began",
            Stage::RaiseToCommit => "g_commit",
            Stage::Encode => "g_encode",
            Stage::PostWait => "g_post",
            Stage::Queued => "g_queued",
            Stage::TransportWoken => "u1_woken",
            Stage::TransportAwake => "u1_awake",
            Stage::CommitToTake => "c_take",
            Stage::Woke => "u2_wake",
            Stage::Gate => "u3_gate",
            Stage::Begin => "u4_begin",
            Stage::LaneHop => "u4_lane",
            Stage::Walk => "u5_walk",
            Stage::Commit => "u6_commit",
            Stage::RtCommit => "u7_rt",
            Stage::Up => "up",
        }
    }
}

/// The events counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Count {
    /// Command buffers taken off the queues' rings.
    Takes,
    /// Those taken with none of their queue's in flight (its completions all written).
    Serial,
    /// Those the queue's thread was woken for (its first take after a sleep).
    Woken,
    /// The queue's threads' sleeps that ended.
    Sleeps,
    /// The doorbells a queue's bell counted as its thread woke (routed writes).
    Bells,
    /// The doorbells written through the device's MMIO.
    MmioBells,
    /// Records written onto the queues' completion rings.
    Records,
    /// Those whose post raised the queue's interrupt.
    Raised,
    /// Those posted with the ring not armed: no interrupt.
    Unarmed,
    /// `CB_COMPLETE` records written.
    Completions,
    /// Completed handlers rt's thread handed on to another thread.
    HandedOn,
    /// Completion posts parked behind a turn before theirs.
    Parked,
    /// Completion posts handed over to the queue's thread.
    HandedOver,
    /// Buffers walked on the queue's thread.
    Inline,
    /// Buffers walked on the lane's thread.
    Lane,
    /// Takes that waited at ring 0's gate.
    GateWaits,
    /// rt commits timed.
    Commits,
    /// Serial takes whose next buffer the guest made before the interrupt.
    BeganBefore,
    /// rt commits with none of their queue's buffers committed and not completed.
    Depth0,
    /// ... with one.
    Depth1,
    /// ... with two.
    Depth2,
    /// ... with three or more.
    Depth3,
}

/// Every count, in the line's order.
pub const COUNTS: [Count; 22] = [
    Count::Takes,
    Count::Serial,
    Count::Woken,
    Count::Sleeps,
    Count::Bells,
    Count::MmioBells,
    Count::Records,
    Count::Raised,
    Count::Unarmed,
    Count::Completions,
    Count::HandedOn,
    Count::Parked,
    Count::HandedOver,
    Count::Inline,
    Count::Lane,
    Count::GateWaits,
    Count::Commits,
    Count::BeganBefore,
    Count::Depth0,
    Count::Depth1,
    Count::Depth2,
    Count::Depth3,
];

impl Count {
    /// Its name in the line.
    pub fn name(self) -> &'static str {
        match self {
            Count::Takes => "takes",
            Count::Serial => "serial",
            Count::Woken => "woken",
            Count::Sleeps => "sleeps",
            Count::Bells => "bells",
            Count::MmioBells => "mmio_bells",
            Count::Records => "records",
            Count::Raised => "raised",
            Count::Unarmed => "unarmed",
            Count::Completions => "completions",
            Count::HandedOn => "handed_on",
            Count::Parked => "parked",
            Count::HandedOver => "handed_over",
            Count::Inline => "inline",
            Count::Lane => "lane",
            Count::GateWaits => "gate_waits",
            Count::Commits => "commits",
            Count::BeganBefore => "began_before",
            Count::Depth0 => "depth0",
            Count::Depth1 => "depth1",
            Count::Depth2 => "depth2",
            Count::Depth3 => "depth3",
        }
    }
}

/// A stage's histogram and sum.
struct Hist {
    buckets: [AtomicU32; BUCKETS],
    sum: AtomicU64,
}

impl Hist {
    const fn new() -> Hist {
        Hist {
            buckets: [const { AtomicU32::new(0) }; BUCKETS],
            sum: AtomicU64::new(0),
        }
    }

    fn add(&self, ns: u64) {
        self.buckets[bucket(ns)].fetch_add(1, Relaxed);
        self.sum.fetch_add(ns, Relaxed);
    }

    /// Its spans since the take before, zeroed.
    fn take(&self) -> Spread {
        Spread {
            buckets: std::array::from_fn(|i| self.buckets[i].swap(0, Relaxed)),
            sum: self.sum.swap(0, Relaxed),
        }
    }
}

/// The idle buckets' bounds ([`idle`]), in nanoseconds: under 100 us, 100 to 200, 200 to 325,
/// 325 to 400, and 400 or more.
pub const IDLE_BOUNDS_NS: [u64; 4] = [100_000, 200_000, 325_000, 400_000];

/// The idle buckets' names in the line.
const IDLE_NAMES: [&str; 5] = ["lt100", "100_200", "200_325", "325_400", "ge400"];

/// The fine idle histogram's step, in nanoseconds: 20 us ([`FineIdle`]).
pub const FINE_STEP_NS: u64 = 20_000;

/// The fine idle histogram's buckets: one a step from 0 to 400 us, then one of 400 us or more.
pub const FINE_BUCKETS: usize = 21;

/// The command buffers' idle at one depth, in steps of [`FINE_STEP_NS`], and in all.
struct Fine {
    buckets: [AtomicU64; FINE_BUCKETS],
    idle_ns: AtomicU64,
}

impl Fine {
    const fn new() -> Fine {
        Fine {
            buckets: [const { AtomicU64::new(0) }; FINE_BUCKETS],
            idle_ns: AtomicU64::new(0),
        }
    }

    /// Its buckets and idle since the take before, zeroed.
    fn take(&self) -> FineIdle {
        FineIdle {
            buckets: std::array::from_fn(|i| self.buckets[i].swap(0, Relaxed)),
            sum: self.idle_ns.swap(0, Relaxed),
        }
    }
}

/// An idle bucket's idle in all, and its command buffers' pickups.
struct Idle {
    idle_ns: AtomicU64,
    pickup: Hist,
}

impl Idle {
    const fn new() -> Idle {
        Idle {
            idle_ns: AtomicU64::new(0),
            pickup: Hist::new(),
        }
    }
}

static ON: AtomicBool = AtomicBool::new(false);
static HISTS: [Hist; STAGES.len()] = [const { Hist::new() }; STAGES.len()];
static COUNTERS: [AtomicU64; COUNTS.len()] = [const { AtomicU64::new(0) }; COUNTS.len()];
/// By the depth at the commit (0, then 1 or more), then by idle bucket.
static IDLES: [[Idle; 5]; 2] = [const { [const { Idle::new() }; 5] }; 2];
/// By the depth at the commit (0, then 1 or more).
static FINE: [Fine; 2] = [const { Fine::new() }; 2];

thread_local! {
    /// When rt's handler entered, for the completion this thread posts next ([`set_down`]).
    static DOWN: Cell<u64> = const { Cell::new(0) };
    /// The take and the host's begin of the buffer this queue's thread hands to the host.
    static UP: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
    /// The take and the walk's start of the buffer this thread carries out ([`set_job`]).
    static JOB: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
    /// When this thread's last post wrote its record, and when its raise ended (the write's time
    /// where it raised nothing).
    static POSTED: Cell<(u64, u64)> = const { Cell::new((0, 0)) };
}

/// Whether the trip is timed.
pub fn on() -> bool {
    ON.load(Relaxed)
}

/// Times the trip from now on.
pub fn turn_on() {
    ON.store(true, Relaxed);
}

/// The device's clock, in nanoseconds.
pub fn now_ns() -> u64 {
    crate::clock::now_ns()
}

/// The device's clock where the trip is timed; else 0, which no stage takes as a start.
pub fn stamp() -> u64 {
    if on() { now_ns() } else { 0 }
}

/// The bucket of a span of `ns` nanoseconds ([`BUCKETS`]).
pub fn bucket(ns: u64) -> usize {
    if ns < 64 {
        return 0;
    }
    let octave = 63 - ns.leading_zeros() as usize;
    let eighth = ((ns >> (octave - 3)) & 7) as usize;
    ((octave - 6) * 8 + eighth + 1).min(BUCKETS - 1)
}

/// The least span bucket `i` holds, in nanoseconds.
pub fn bucket_floor(i: usize) -> u64 {
    if i == 0 {
        return 0;
    }
    let octave = (i - 1) / 8 + 6;
    let eighth = ((i - 1) % 8) as u64;
    (8 + eighth) << (octave - 3)
}

/// Adds a span of `ns` nanoseconds to `stage`.
pub fn time(stage: Stage, ns: u64) {
    HISTS[stage as usize].add(ns);
}

/// A command buffer committed with `depth` of its queue's buffers committed and not completed,
/// whose GPU sat idle `idle_ns` before it and picked it up `pickup_ns` after rt scheduled it, the
/// submit call that carries it included (both from rt's times of it): its pickup in its idle
/// bucket, and its idle in its step of [`FINE_STEP_NS`], by its depth 0 or more.
pub fn idle(depth: usize, idle_ns: u64, pickup_ns: u64) {
    let at = IDLE_BOUNDS_NS
        .iter()
        .take_while(|&&bound| idle_ns >= bound)
        .count();
    let cell = &IDLES[usize::from(depth > 0)][at];
    cell.idle_ns.fetch_add(idle_ns, Relaxed);
    cell.pickup.add(pickup_ns);
    let fine = &FINE[usize::from(depth > 0)];
    let step = (idle_ns / FINE_STEP_NS).min(FINE_BUCKETS as u64 - 1);
    fine.buckets[step as usize].fetch_add(1, Relaxed);
    fine.idle_ns.fetch_add(idle_ns, Relaxed);
}

/// Adds the span from `from` to now to `stage`, where `from` is not 0.
pub fn since(stage: Stage, from: u64) {
    if from != 0 {
        time(stage, now_ns().saturating_sub(from));
    }
}

/// Counts one of `count`.
pub fn count(count: Count) {
    add(count, 1);
}

/// Counts `n` of `count`.
pub fn add(count: Count, n: u64) {
    COUNTERS[count as usize].fetch_add(n, Relaxed);
}

/// Counts an rt commit made with `n` of its queue's buffers committed and not completed.
pub fn depth(n: usize) {
    count(match n {
        0 => Count::Depth0,
        1 => Count::Depth1,
        2 => Count::Depth2,
        _ => Count::Depth3,
    });
}

/// rt's handler of the completion this thread posts next entered at `ns` (0: none).
pub fn set_down(ns: u64) {
    DOWN.set(ns);
}

/// When rt's handler of the completion this thread posts now entered, taken: 0 for a post that
/// is no buffer's completion, or with the trip off.
pub(crate) fn take_down() -> u64 {
    DOWN.replace(0)
}

/// The take and the host's begin of the buffer this queue's thread hands to the host now, (0, 0)
/// with the trip off.
pub fn up() -> (u64, u64) {
    UP.get()
}

/// Sets [`up`].
pub(crate) fn set_up(taken: u64, begun: u64) {
    UP.set((taken, begun));
}

/// The buffer this thread carries out now was taken at `taken` and its walk started at `start`
/// ((0, 0): none).
pub fn set_job(taken: u64, start: u64) {
    JOB.set((taken, start));
}

/// [`set_job`]'s take and start on this thread.
pub fn job() -> (u64, u64) {
    JOB.get()
}

/// This thread's post wrote its record at `written`, and its raise ended at `end`.
pub(crate) fn set_posted(written: u64, end: u64) {
    POSTED.set((written, end));
}

/// [`set_posted`]'s times of this thread's last post.
pub(crate) fn posted() -> (u64, u64) {
    POSTED.get()
}

/// A queue's own part of the trip: its buffers taken and completions written, its last
/// completion's end and its thread's last wake.
#[derive(Debug, Default)]
pub(crate) struct Queue {
    taken: AtomicU64,
    done: AtomicU64,
    down_ns: AtomicU64,
    woke_ns: AtomicU64,
}

/// What a take of a buffer found ([`Queue::took`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Take {
    /// When the queue's last completion's interrupt was raised, where none of the queue's
    /// buffers was in flight; else 0.
    pub(crate) serial_from: u64,
    /// The queue's thread was woken for it.
    pub(crate) woken: bool,
}

impl Queue {
    /// A completion written onto the queue's ring, its interrupt raised (or nothing raised) at
    /// `end`.
    pub(crate) fn completed(&self, end: u64) {
        self.down_ns.store(end, Relaxed);
        self.done.fetch_add(1, Relaxed);
    }

    /// The queue's thread woke at `at` from a sleep.
    pub(crate) fn woke(&self, at: u64) {
        self.woke_ns.store(at, Relaxed);
    }

    /// A buffer taken off the queue's ring at `taken`, by its thread: counted, with its wake and
    /// its gap timed.
    pub(crate) fn took(&self, taken: u64) -> Take {
        let before = self.taken.fetch_add(1, Relaxed);
        let in_flight = before.saturating_sub(self.done.load(Relaxed));
        count(Count::Takes);
        let mut take = Take::default();
        let down = self.down_ns.load(Relaxed);
        if in_flight == 0 && down != 0 && taken >= down {
            count(Count::Serial);
            time(Stage::Gap, taken - down);
            take.serial_from = down;
        }
        let woke = self.woke_ns.swap(0, Relaxed);
        if woke != 0 {
            count(Count::Woken);
            time(Stage::Woke, taken.saturating_sub(woke));
            take.woken = true;
        }
        take
    }
}

/// The guest's times of a buffer on the host's clock, where the guest held its clock's offset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct OnHost {
    /// The post to the thread's return to the ring.
    pub(crate) queued: u64,
    /// The later of the post and the return, to the take.
    pub(crate) transport: u64,
    /// The guest's make of the buffer.
    pub(crate) began: u64,
    /// The guest's commit call.
    pub(crate) committed: u64,
}

/// The guest's times of a buffer `take` found (the module doc's guest's turn): its make to its
/// commit call (`encode`), that call to its post (`post_wait`), and those on the host's clock
/// where the guest held its clock's offset (`host`), taken at `taken`.
pub(crate) fn guest(encode: u64, post_wait: u64, host: Option<OnHost>, taken: u64, take: Take) {
    time(Stage::Encode, encode);
    time(Stage::PostWait, post_wait);
    let Some(host) = host else {
        return;
    };
    time(Stage::Queued, host.queued);
    let transport = if take.woken {
        Stage::TransportWoken
    } else {
        Stage::TransportAwake
    };
    time(transport, host.transport);
    time(Stage::CommitToTake, taken.saturating_sub(host.committed));
    if take.serial_from == 0 {
        return;
    }
    if host.began < take.serial_from {
        count(Count::BeganBefore);
    } else {
        time(Stage::RaiseToBegan, host.began - take.serial_from);
    }
    if host.committed >= take.serial_from {
        time(Stage::RaiseToCommit, host.committed - take.serial_from);
    }
}

/// A stage's spans since the take before.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spread {
    /// How many spans in each bucket.
    pub buckets: [u32; BUCKETS],
    /// Their sum, in nanoseconds.
    pub sum: u64,
}

impl Spread {
    /// How many spans.
    pub fn n(&self) -> u64 {
        self.buckets.iter().map(|&n| u64::from(n)).sum()
    }

    /// The span at fraction `q` of them by nearest rank, placed within its bucket by its rank
    /// there, in nanoseconds; 0 with none.
    pub fn quantile(&self, q: f64) -> u64 {
        let n = self.n();
        if n == 0 {
            return 0;
        }
        let rank = ((q * n as f64).ceil() as u64).clamp(1, n);
        let mut below = 0;
        for (i, &c) in self.buckets.iter().enumerate() {
            let c = u64::from(c);
            if below + c >= rank {
                let floor = bucket_floor(i);
                let top = if i + 1 < BUCKETS {
                    bucket_floor(i + 1)
                } else {
                    floor
                };
                // The rank's middle within the bucket, so the span stays in it.
                let into = ((rank - below) as f64 - 0.5) / c as f64;
                return floor + ((top - floor) as f64 * into) as u64;
            }
            below += c;
        }
        0
    }
}

/// The command buffers' idle at one depth since the take before, in steps of [`FINE_STEP_NS`]
/// ([`idle`]).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FineIdle {
    /// Each step's command buffers: bucket `i` holds an idle from `i` steps to under `i + 1`, the
    /// last every idle of 400 us or more.
    pub buckets: [u64; FINE_BUCKETS],
    /// Their idle in all, in nanoseconds.
    pub sum: u64,
}

impl FineIdle {
    /// How many command buffers.
    pub fn n(&self) -> u64 {
        self.buckets.iter().sum()
    }

    /// Their mean idle, in nanoseconds; 0 with none.
    pub fn mean(&self) -> u64 {
        self.sum.checked_div(self.n()).unwrap_or(0)
    }

    /// The idle at fraction `q` of them by nearest rank, placed within its step by its rank there,
    /// in nanoseconds, or the last bucket's floor for one in it; 0 with none.
    pub fn quantile(&self, q: f64) -> u64 {
        let n = self.n();
        if n == 0 {
            return 0;
        }
        let rank = ((q * n as f64).ceil() as u64).clamp(1, n);
        let mut below = 0;
        for (i, &c) in self.buckets.iter().enumerate() {
            if below + c >= rank {
                let floor = i as u64 * FINE_STEP_NS;
                if i + 1 == FINE_BUCKETS {
                    return floor;
                }
                // The rank's middle within the step, so the idle stays in it.
                let into = ((rank - below) as f64 - 0.5) / c as f64;
                return floor + (FINE_STEP_NS as f64 * into) as u64;
            }
            below += c;
        }
        0
    }
}

/// Every stage's spread and count since the take before ([`take`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Taken {
    /// Each stage's, in [`STAGES`]' order.
    pub stages: Vec<Spread>,
    /// Each count, in [`COUNTS`]' order.
    pub counts: Vec<u64>,
    /// Each idle bucket's idle in all, in nanoseconds, and its pickups ([`idle`]): those at depth
    /// 0 in the buckets' order, then those at 1 or more.
    pub idle: Vec<(u64, Spread)>,
    /// The idle in steps of [`FINE_STEP_NS`] ([`idle`]): of those at depth 0, then at 1 or more.
    pub fine: Vec<FineIdle>,
}

/// Every stage's spans and every count since the take before, each zeroed.
pub fn take() -> Taken {
    Taken {
        stages: HISTS.iter().map(Hist::take).collect(),
        counts: COUNTERS.iter().map(|n| n.swap(0, Relaxed)).collect(),
        idle: (IDLES.iter().flatten())
            .map(|cell| (cell.idle_ns.swap(0, Relaxed), cell.pickup.take()))
            .collect(),
        fine: FINE.iter().map(Fine::take).collect(),
    }
}

/// `ns` nanoseconds in microseconds, to a tenth.
fn us(ns: u64) -> f64 {
    ns as f64 / 1e3
}

/// `spread` as `n sum_us p50_us p90_us [bucket:n,...]`, its buckets with spans by index
/// ([`bucket_floor`]).
fn write_spread(f: &mut fmt::Formatter<'_>, spread: &Spread) -> fmt::Result {
    write!(
        f,
        "{} {:.1} {:.1} {:.1} [",
        spread.n(),
        us(spread.sum),
        us(spread.quantile(0.5)),
        us(spread.quantile(0.9))
    )?;
    let mut sep = "";
    for (i, &c) in spread.buckets.iter().enumerate() {
        if c != 0 {
            write!(f, "{sep}{i}:{c}")?;
            sep = ",";
        }
    }
    f.write_char(']')
}

impl fmt::Display for Taken {
    /// `count=n ...`, then for each stage with spans `| name n sum_us p50_us p90_us
    /// [bucket:n,...]` ([`write_spread`]), then for each idle bucket with command buffers
    /// `| idle d0|d1 bucket idle_us n pickup_sum_us pickup_p50_us pickup_p90_us [...]`, then for
    /// each depth with command buffers `| idle20 d0|d1 n idle_us mean_us p50_us p90_us
    /// [step:n,...]` ([`FineIdle`]).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        for (count, n) in COUNTS.iter().zip(&self.counts) {
            if !first {
                f.write_char(' ')?;
            }
            first = false;
            write!(f, "{}={n}", count.name())?;
        }
        for (stage, spread) in STAGES.iter().zip(&self.stages) {
            if spread.n() != 0 {
                write!(f, " | {} ", stage.name())?;
                write_spread(f, spread)?;
            }
        }
        for (i, (idle_ns, pickup)) in self.idle.iter().enumerate() {
            if pickup.n() != 0 {
                let (depth, at) = (i / IDLE_NAMES.len(), i % IDLE_NAMES.len());
                write!(
                    f,
                    " | idle d{depth} {} {:.1} ",
                    IDLE_NAMES[at],
                    us(*idle_ns)
                )?;
                write_spread(f, pickup)?;
            }
        }
        for (depth, fine) in self.fine.iter().enumerate() {
            if fine.n() != 0 {
                write!(
                    f,
                    " | idle20 d{depth} {} {:.1} {:.1} {:.1} {:.1} [",
                    fine.n(),
                    us(fine.sum),
                    us(fine.mean()),
                    us(fine.quantile(0.5)),
                    us(fine.quantile(0.9))
                )?;
                let mut sep = "";
                for (i, &c) in fine.buckets.iter().enumerate() {
                    if c != 0 {
                        write!(f, "{sep}{i}:{c}")?;
                        sep = ",";
                    }
                }
                f.write_char(']')?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each bucket's floor falls in it, its last span below the next's floor
    #[test]
    fn buckets_hold_their_floors() {
        assert_eq!(bucket(0), 0);
        assert_eq!(bucket(63), 0);
        assert_eq!(bucket(64), 1);
        assert_eq!(bucket(127), 8);
        assert_eq!(bucket(128), 9);
        for i in 1..BUCKETS {
            assert_eq!(bucket(bucket_floor(i)), i, "floor of {i}");
            assert_eq!(bucket(bucket_floor(i) - 1), i - 1, "below the floor of {i}");
        }
        assert_eq!(bucket(u64::MAX), BUCKETS - 1);
        assert_eq!(bucket_floor(BUCKETS - 1), 1 << 31);
    }

    /// A spread's quantiles fall in the buckets their ranks do
    #[test]
    fn quantiles_fall_in_their_ranks_buckets() {
        let mut spread = Spread {
            buckets: [0; BUCKETS],
            sum: 0,
        };
        assert_eq!(spread.quantile(0.5), 0);
        // Nine spans of 10 us and one of 100 us.
        spread.buckets[bucket(10_000)] = 9;
        spread.buckets[bucket(100_000)] = 1;
        let p50 = spread.quantile(0.5);
        assert_eq!(bucket(p50), bucket(10_000), "{p50}");
        let p90 = spread.quantile(0.9);
        assert_eq!(bucket(p90), bucket(10_000), "{p90}");
        let p99 = spread.quantile(0.99);
        assert_eq!(bucket(p99), bucket(100_000), "{p99}");
    }

    /// A command buffer's idle goes in its bucket by its depth
    #[test]
    fn an_idle_goes_in_its_bucket_by_depth() {
        idle(0, 99_999, 30_000);
        idle(2, 400_000, 180_000);
        let taken = take();
        assert_eq!(taken.idle[0].0, 99_999);
        assert_eq!(taken.idle[0].1.n(), 1);
        assert_eq!(taken.idle[IDLE_NAMES.len() + 4].0, 400_000);
        assert_eq!(taken.idle[IDLE_NAMES.len() + 4].1.sum, 180_000);
        assert_eq!(take().idle[0].1.n(), 0, "taken once");
    }

    /// A command buffer's idle goes in its 20 us step by its depth, the last step holding every
    /// idle of 400 us or more
    #[test]
    fn an_idle_goes_in_its_20_us_step_by_depth() {
        idle(0, 19_999, 0);
        idle(0, 20_000, 0);
        idle(1, 399_999, 0);
        idle(3, 400_000, 0);
        idle(3, 5_000_000, 0);
        let fine = take().fine;
        assert_eq!(fine[0].buckets[0], 1);
        assert_eq!(fine[0].buckets[1], 1);
        assert_eq!(fine[0].sum, 39_999);
        assert_eq!(fine[1].buckets[19], 1);
        assert_eq!(fine[1].buckets[FINE_BUCKETS - 1], 2);
        assert_eq!(fine[1].n(), 3);
        assert_eq!(take().fine[1].n(), 0, "taken once");
    }

    /// A fine idle's mean is its sum over its count, and its quantiles fall in their ranks' steps,
    /// one in the last at its floor
    #[test]
    fn fine_idle_quantiles_fall_in_their_ranks_steps() {
        let mut fine = FineIdle::default();
        assert_eq!((fine.mean(), fine.quantile(0.5)), (0, 0));
        // Nine of 110 us and one of 500 us.
        fine.buckets[5] = 9;
        fine.buckets[FINE_BUCKETS - 1] = 1;
        fine.sum = 9 * 110_000 + 500_000;
        assert_eq!(fine.mean(), 149_000);
        assert_eq!(fine.quantile(0.5) / FINE_STEP_NS, 5);
        assert_eq!(fine.quantile(0.9) / FINE_STEP_NS, 5);
        assert_eq!(fine.quantile(0.99), 400_000);
    }

    /// A queue's take with none in flight times the gap from its last completion, and its wake
    #[test]
    fn a_serial_take_times_its_gap() {
        let queue = Queue::default();
        // The first take: no completion before it.
        assert_eq!(queue.took(1_000), Take::default());
        queue.completed(2_000);
        queue.woke(2_500);
        assert_eq!(
            queue.took(3_000),
            Take {
                serial_from: 2_000,
                woken: true
            }
        );
        // Taken with the one before it in flight: no gap, and no wake.
        assert_eq!(queue.took(3_100), Take::default());
    }

    /// The line names each count, and each stage with spans with its buckets
    #[test]
    fn the_line_names_counts_and_stages_with_spans() {
        let mut taken = Taken {
            stages: vec![
                Spread {
                    buckets: [0; BUCKETS],
                    sum: 0,
                };
                STAGES.len()
            ],
            counts: vec![0; COUNTS.len()],
            idle: vec![
                (
                    0,
                    Spread {
                        buckets: [0; BUCKETS],
                        sum: 0,
                    }
                );
                2 * IDLE_NAMES.len()
            ],
            fine: vec![FineIdle::default(); 2],
        };
        taken.counts[Count::Takes as usize] = 3;
        // One buffer at depth 1 after 350 us idle, picked up in 150 us.
        taken.idle[IDLE_NAMES.len() + 3].0 = 350_000;
        taken.idle[IDLE_NAMES.len() + 3].1.buckets[bucket(150_000)] = 1;
        taken.idle[IDLE_NAMES.len() + 3].1.sum = 150_000;
        // Two buffers at depth 0 after 105 and 125 us idle.
        taken.fine[0].buckets[5] = 2;
        taken.fine[0].sum = 230_000;
        taken.stages[Stage::Gap as usize].buckets[bucket(20_000)] = 2;
        taken.stages[Stage::Gap as usize].sum = 40_000;
        let line = taken.to_string();
        assert!(line.starts_with("takes=3 serial=0 "), "{line}");
        assert!(line.contains(" | gap 2 40.0 "), "{line}");
        assert!(line.contains(&format!("[{}:2]", bucket(20_000))), "{line}");
        assert!(line.contains(" | idle d1 325_400 350.0 1 150.0 "), "{line}");
        assert!(
            line.ends_with(" | idle20 d0 2 230.0 115.0 105.0 115.0 [5:2]"),
            "{line}"
        );
        assert!(!line.contains("d2_handler"), "{line}");
    }
}
