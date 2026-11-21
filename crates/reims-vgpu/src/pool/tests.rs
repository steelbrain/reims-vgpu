// Test scaffolding: real time bounds the waits and runs the delays, but for those a test must take
// a step inside, which a clock the test moves runs (`Hand`); a helper thread stands in for a thread
// of the device's.
#![allow(clippy::disallowed_methods)]

use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use super::*;

const BASE: u64 = 52 << 30;
const MIB: u64 = 1 << 20;

/// What the pool asked of the host, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Call {
    Add(u64, u64),
    Remove(u64, u64),
}

/// A host that records the pool's asks and answers none by itself.
#[derive(Default)]
struct Host {
    calls: Mutex<Vec<Call>>,
    changed: Condvar,
}

impl Host {
    fn pool_host(self: &Arc<Self>) -> PoolHost {
        let add = Arc::clone(self);
        let remove = Arc::clone(self);
        PoolHost {
            add: Arc::new(move |at, _host, len| add.record(Call::Add(at, len))),
            remove: Arc::new(move |at, len| remove.record(Call::Remove(at, len))),
        }
    }

    fn record(&self, call: Call) {
        self.calls.lock().push(call);
        self.changed.notify_all();
    }

    /// The calls once there are at least `n`, or the test fails after 10 s.
    fn wait_for(&self, n: usize) -> Vec<Call> {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut calls = self.calls.lock();
        while calls.len() < n {
            assert!(
                !self.changed.wait_until(&mut calls, deadline).timed_out(),
                "the pool asked {:?}, not {n} calls",
                *calls
            );
        }
        calls.clone()
    }

    fn calls(&self) -> Vec<Call> {
        self.calls.lock().clone()
    }
}

/// The tests' monotonic clock: nanoseconds since its first reading.
fn now() -> PoolNow {
    static START: LazyLock<Instant> = LazyLock::new(Instant::now);
    Box::new(|| MonoTime::from_nanos(u64::try_from(START.elapsed().as_nanos()).unwrap()))
}

/// A clock that stands still until the test moves it on ([`Hand::on`]): a pool on it gives a
/// piece back only once the test has moved the clock past the piece's delay, so a step the test
/// takes before that comes first however slowly a loaded host runs it.
#[derive(Default)]
struct Hand(Arc<std::sync::atomic::AtomicU64>);

impl Hand {
    /// The clock, at the nanoseconds the test moved it to.
    fn clock(&self) -> PoolNow {
        let nanos = Arc::clone(&self.0);
        Box::new(move || MonoTime::from_nanos(nanos.load(std::sync::atomic::Ordering::SeqCst)))
    }

    /// Moves the clock on by `by`.
    fn on(&self, by: Duration) {
        let by = u64::try_from(by.as_nanos()).unwrap();
        self.0.fetch_add(by, std::sync::atomic::Ordering::SeqCst);
    }
}

/// The host blocks the pool takes from, kept for the tests' counts.
fn pool(range: PoolConfig, host: &Arc<Host>, delay: Duration) -> (Pool, Arc<HostBlocks>) {
    pool_at(range, host, delay, now())
}

/// [`pool`], on `clock`.
fn pool_at(
    range: PoolConfig,
    host: &Arc<Host>,
    delay: Duration,
    clock: PoolNow,
) -> (Pool, Arc<HostBlocks>) {
    let blocks = Arc::new(HostBlocks::new());
    let source: Arc<dyn BlockSource> = Arc::clone(&blocks) as _;
    let pool = Pool::with_delay(range, host.pool_host(), delay, clock, source).unwrap();
    (pool, blocks)
}

fn range(gib: u64) -> PoolConfig {
    PoolConfig {
        physical_address: BASE,
        physical_length: gib << 30,
    }
}

const LONG: Duration = Duration::from_secs(3600);

/// The pieces of the pool's headroom: [`HEADROOM`] in pieces of [`BLOCK_LEN`].
const KEPT: u64 = HEADROOM.div_ceil(BLOCK_LEN);

/// [`KEPT`], as a count of the host's calls.
const KEPT_CALLS: usize = KEPT as usize;

/// The address of the first piece past a headroom of pieces from [`BASE`] up.
const PAST: u64 = BASE + KEPT * BLOCK_LEN;

/// The pieces of a headroom from [`BASE`] up, as the pool's adds of them and as its live pieces.
fn headroom_adds() -> Vec<Call> {
    (headroom_pieces().into_iter())
        .map(|(at, len)| Call::Add(at, len))
        .collect()
}

/// [`headroom_adds`], as [`Pool::live`] lists them.
fn headroom_pieces() -> Vec<(u64, u64)> {
    (0..KEPT)
        .map(|i| (BASE + i * BLOCK_LEN, BLOCK_LEN))
        .collect()
}

/// A piece is one of metal2vulkan-runtime's units, its length read from rt, and the headroom is
/// [`HEADROOM`] bytes of them, asked ahead of the guest's first take lowest address first: a take
/// of a unit fills one, and the pool asks one more; a take longer than a unit has a piece of its
/// own length past them, and the headroom's pieces stay empty, so the pool asks none
#[test]
fn a_piece_is_one_rt_unit() {
    assert_eq!(
        BLOCK_LEN,
        metal2vulkan_runtime::MemoryClass::Shared.block_len()
    );
    assert_eq!(KEPT * BLOCK_LEN, HEADROOM);
    let host = Arc::new(Host::default());
    let (pool, _blocks) = pool(range(64), &host, LONG);
    let ledger = pool.ledger();
    assert!(ledger.keep_headroom());
    assert_eq!(host.wait_for(KEPT_CALLS), headroom_adds());
    for (at, _) in headroom_pieces() {
        pool.added(at).unwrap();
    }
    assert_eq!(answer(&take(&ledger, 1, BLOCK_LEN)), Some((0, BLOCK_LEN)));
    assert_eq!(
        host.wait_for(KEPT_CALLS + 1)[KEPT_CALLS],
        Call::Add(PAST, BLOCK_LEN)
    );
    pool.added(PAST).unwrap();
    let long = BLOCK_LEN + TAKE_ALIGN;
    let rx = take(&ledger, 2, long);
    let at = PAST + BLOCK_LEN;
    // [`HostBlocks`] rounds a block longer than a unit up to 2 MiB.
    assert_eq!(
        host.wait_for(KEPT_CALLS + 2)[KEPT_CALLS + 1],
        Call::Add(at, BLOCK_LEN + 2 * MIB)
    );
    pool.added(at).unwrap();
    assert_eq!(answer(&rx), Some((at - BASE, long)));
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(host.calls().len(), KEPT_CALLS + 2);
}

/// A take of exactly one unit needs exactly one piece: its carve starts at the piece's first byte
/// and ends at its last, with no pad, guard or rounding, so each of the headroom's pieces holds
/// one such take whole, a take of one granule after them finds nothing left over in any of them,
/// and the pool asks only unit pieces, none of a take's own length.
#[test]
fn a_take_of_one_unit_fills_one_piece_with_nothing_left_over() {
    let host = Arc::new(Host::default());
    let (pool, _blocks) = pool(range(64), &host, LONG);
    let ledger = pool.ledger();
    assert!(ledger.keep_headroom());
    assert_eq!(host.wait_for(KEPT_CALLS), headroom_adds());
    for (at, _) in headroom_pieces() {
        pool.added(at).unwrap();
    }
    for i in 0..KEPT {
        let rx = take(&ledger, 1, BLOCK_LEN);
        assert_eq!(answer(&rx), Some((i * BLOCK_LEN, BLOCK_LEN)), "take {i}");
    }
    // Each filled piece has the pool ask one more unit, past the headroom's.
    let calls = host.wait_for(2 * KEPT_CALLS);
    let more: Vec<_> = (0..KEPT)
        .map(|i| Call::Add(PAST + i * BLOCK_LEN, BLOCK_LEN))
        .collect();
    assert_eq!(calls[KEPT_CALLS..], more[..]);
    for i in 0..KEPT {
        pool.added(PAST + i * BLOCK_LEN).unwrap();
    }
    assert_eq!(
        answer(&take(&ledger, 1, TAKE_ALIGN)),
        Some((PAST - BASE, TAKE_ALIGN))
    );
    assert_eq!(
        host.wait_for(2 * KEPT_CALLS + 1)[2 * KEPT_CALLS],
        Call::Add(PAST + KEPT * BLOCK_LEN, BLOCK_LEN)
    );
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(host.calls().len(), 2 * KEPT_CALLS + 1);
}

#[test]
fn a_range_must_be_whole_large_pages_and_not_empty() {
    assert!(range(64).is_valid());
    for bad in [
        PoolConfig::default(),
        PoolConfig {
            physical_address: BASE + 4096,
            physical_length: 1 << 30,
        },
        PoolConfig {
            physical_address: BASE,
            physical_length: (1 << 30) + 4096,
        },
        PoolConfig {
            physical_address: u64::MAX - (2 << 20) + 1,
            physical_length: 4 << 20,
        },
    ] {
        assert!(!bad.is_valid(), "{bad:?}");
    }
    let host = Arc::new(Host::default());
    assert_eq!(
        Pool::new(
            PoolConfig::default(),
            host.pool_host(),
            now(),
            Arc::new(HostBlocks::new())
        )
        .err(),
        Some(Error::InvalidArgument)
    );
}

/// A pool's range is the GPU's memory heaps, summed, from the start of the addresses the host can
/// give, rounded up to whole large pages, and stops at the end of those addresses where they hold
/// less; a GPU with no memory heaps gives no range
#[test]
fn a_range_is_the_memory_heaps_rounded_to_large_pages_at_most_the_room() {
    let room = range(43);
    let sized = |heaps: u64| {
        room.sized(heaps)
            .map(|r| (r.physical_address, r.physical_length))
    };
    assert_eq!(sized(24 << 30), Some((BASE, 24 << 30)));
    assert_eq!(sized((24 << 30) - MIB), Some((BASE, 24 << 30)));
    assert_eq!(sized((24 << 30) + 1), Some((BASE, (24 << 30) + (2 << 20))));
    assert_eq!(sized(1), Some((BASE, 2 << 20)));
    // Past the room: the room.
    assert_eq!(sized(64 << 30), Some((BASE, 43 << 30)));
    assert_eq!(sized(u64::MAX), Some((BASE, 43 << 30)));
    assert_eq!(sized(0), None);
    for heaps in [1, (24 << 30) - MIB, 64 << 30, u64::MAX] {
        assert!(room.sized(heaps).unwrap().is_valid(), "{heaps:#x}");
    }
}

#[test]
fn pieces_go_out_at_ascending_addresses_from_the_range_start() {
    let host = Arc::new(Host::default());
    let (pool, _blocks) = pool(range(64), &host, LONG);
    pool.grow(BLOCK_LEN);
    pool.grow(1);
    pool.grow(BLOCK_LEN + 1);
    assert_eq!(
        host.wait_for(3),
        [
            Call::Add(BASE, BLOCK_LEN),
            Call::Add(BASE + BLOCK_LEN, BLOCK_LEN),
            // A dedicated block, rounded up to 2 MiB.
            Call::Add(BASE + 2 * BLOCK_LEN, BLOCK_LEN + 2 * MIB),
        ]
    );
}

#[test]
fn a_piece_is_found_only_once_the_host_added_it() {
    let host = Arc::new(Host::default());
    let (pool, _blocks) = pool(range(64), &host, LONG);
    pool.grow(BLOCK_LEN);
    host.wait_for(1);
    assert!(pool.find(BASE, 4096).is_none());
    assert!(pool.live().is_empty());
    pool.added(BASE).unwrap();
    let hold = pool.find(BASE + 8192, 4096).unwrap();
    assert_eq!(hold.offset, 8192);
    assert_eq!(pool.live(), [(BASE, BLOCK_LEN)]);
    // Past the piece's end, and below the range: nothing.
    assert!(pool.find(BASE + BLOCK_LEN - 4096, 8192).is_none());
    assert!(pool.find(BASE - 4096, 4096).is_none());
    // The host says it once, and only where the pool asked.
    assert_eq!(pool.added(BASE), Err(Error::InvalidArgument));
    assert_eq!(pool.added(BASE + BLOCK_LEN), Err(Error::InvalidArgument));
    assert_eq!(pool.removed(BASE + 2 * MIB), Err(Error::InvalidArgument));
}

#[test]
fn a_piece_holds_the_blocks_memory_the_host_maps() {
    let host = Arc::new(Host::default());
    let (pool, _blocks) = pool(range(64), &host, LONG);
    // The zeroer zeroes a new piece of the headroom ahead of its takes, and this test writes
    // through a lookup, which no carve holds: shut it, or its chunk's write may land after the
    // test's and read back as zero.
    pool.shared.state.lock().zero_shut = true;
    pool.grow(BLOCK_LEN);
    host.wait_for(1);
    pool.added(BASE).unwrap();
    let a = pool.find(BASE + 4096, 8).unwrap();
    // SAFETY: inside the block, which the hold keeps.
    unsafe { a.host.as_ptr().cast::<u64>().write(0x1234_5678) };
    let b = pool.find(BASE + 4096, 8).unwrap();
    // SAFETY: as above.
    assert_eq!(unsafe { b.host.as_ptr().cast::<u64>().read() }, 0x1234_5678);
}

/// The pool at `range` with the pieces of a headroom live from [`BASE`] up and one more above
/// them, at [`PAST`], every one empty: the one above, empty past `delay`, is given back once its
/// turn comes.
fn one_past(host: &Arc<Host>, delay: Duration) -> (Pool, Arc<HostBlocks>) {
    one_past_at(host, delay, now())
}

/// [`one_past`], on `clock`.
fn one_past_at(host: &Arc<Host>, delay: Duration, clock: PoolNow) -> (Pool, Arc<HostBlocks>) {
    let (pool, blocks) = pool_at(range(64), host, delay, clock);
    for _ in 0..=KEPT {
        pool.grow(BLOCK_LEN);
    }
    host.wait_for(KEPT_CALLS + 1);
    for i in 0..=KEPT {
        pool.added(BASE + i * BLOCK_LEN).unwrap();
    }
    (pool, blocks)
}

#[test]
fn a_piece_goes_back_only_once_the_host_removed_it() {
    let host = Arc::new(Host::default());
    let (pool, blocks) = one_past(&host, Duration::from_millis(50));
    let n = KEPT_CALLS + 1;
    // The piece above the headroom, empty past the delay, is asked out of the guest, and is out
    // of the lookups.
    assert_eq!(host.wait_for(n + 1)[n], Call::Remove(PAST, BLOCK_LEN));
    assert!(pool.find(PAST, 1).is_none());
    // Its block stays the pool's until the host says the piece is out.
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(blocks.counts(), (n, 0));
    pool.removed(PAST).unwrap();
    wait_counts(&blocks, KEPT_CALLS, 1);
    assert_eq!(pool.live(), headroom_pieces());
    // The address is free again: the next block goes there.
    pool.grow(BLOCK_LEN);
    assert_eq!(host.wait_for(n + 2)[n + 1], Call::Add(PAST, BLOCK_LEN));
}

#[test]
fn a_held_piece_goes_back_only_once_its_hold_ends() {
    let delay = Duration::from_millis(50);
    let (host, hand) = (Arc::new(Host::default()), Hand::default());
    let (pool, blocks) = one_past_at(&host, delay, hand.clock());
    let n = KEPT_CALLS + 1;
    // The clock stands: the piece above the headroom is live as the hold is taken.
    let hold = pool.find(PAST, 1).unwrap();
    hand.on(delay);
    // Out of the lookups past the delay, but neither asked out of the guest nor given back while
    // the hold lives.
    let deadline = Instant::now() + Duration::from_secs(10);
    while pool.find(PAST, 1).is_some() {
        assert!(
            Instant::now() < deadline,
            "still in the lookups past the delay"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(host.calls().len(), n, "no remove while a hold lives");
    assert_eq!(blocks.counts(), (n, 0));
    drop(hold);
    assert_eq!(host.wait_for(n + 1)[n], Call::Remove(PAST, BLOCK_LEN));
    assert_eq!(blocks.counts(), (n, 0));
    pool.removed(PAST).unwrap();
    wait_counts(&blocks, KEPT_CALLS, 1);
}

#[test]
fn a_piece_given_back_before_the_hosts_add_answers_still_waits_for_the_remove() {
    let host = Arc::new(Host::default());
    let (pool, blocks) = pool(range(64), &host, Duration::from_millis(50));
    let n = KEPT_CALLS + 1;
    for _ in 0..=KEPT {
        pool.grow(BLOCK_LEN);
    }
    host.wait_for(n);
    for (at, _) in headroom_pieces() {
        pool.added(at).unwrap();
    }
    // The piece above the headroom goes back before the host's add answers: its remove is asked
    // all the same.
    assert_eq!(host.wait_for(n + 1)[n], Call::Remove(PAST, BLOCK_LEN));
    // The host's add, which it queued first, then its remove.
    pool.added(PAST).unwrap();
    assert!(pool.find(PAST, 1).is_none());
    assert_eq!(blocks.counts(), (n, 0));
    pool.removed(PAST).unwrap();
    wait_counts(&blocks, KEPT_CALLS, 1);
}

#[test]
fn the_release_gives_every_block_back_and_asks_the_host_nothing() {
    let host = Arc::new(Host::default());
    let (mut pool, blocks) = pool(range(64), &host, LONG);
    pool.grow(BLOCK_LEN);
    pool.grow(BLOCK_LEN);
    host.wait_for(2);
    pool.added(BASE).unwrap();
    // The host removed both pieces before the release: it says so of the live one, as QEMU's
    // pieces' ends do; of the other it says nothing, and the release stands for it.
    pool.removed(BASE).unwrap();
    pool.close();
    assert_eq!(blocks.counts(), (0, 2));
    assert_eq!(host.calls().len(), 2);
}

#[test]
fn the_release_gives_a_held_piece_back_only_once_its_hold_ends() {
    let host = Arc::new(Host::default());
    let (mut pool, blocks) = pool(range(64), &host, LONG);
    pool.grow(BLOCK_LEN);
    host.wait_for(1);
    pool.added(BASE).unwrap();
    let hold = pool.find(BASE, 1).unwrap();
    pool.removed(BASE).unwrap();
    let close = std::thread::spawn(move || {
        pool.close();
        pool
    });
    std::thread::sleep(Duration::from_millis(100));
    assert!(!close.is_finished(), "the release waits for the hold");
    assert_eq!(blocks.counts(), (1, 0));
    drop(hold);
    let pool = close.join().unwrap();
    assert_eq!(blocks.counts(), (0, 1));
    drop(pool);
    assert_eq!(host.calls().len(), 1);
}

#[test]
fn a_full_range_gives_the_block_back() {
    let host = Arc::new(Host::default());
    let (pool, blocks) = pool(
        PoolConfig {
            physical_address: BASE,
            physical_length: BLOCK_LEN + 2 * MIB,
        },
        &host,
        LONG,
    );
    pool.grow(BLOCK_LEN);
    host.wait_for(1);
    pool.grow(BLOCK_LEN);
    let deadline = Instant::now() + Duration::from_secs(10);
    while blocks.counts() != (1, 1) {
        assert!(Instant::now() < deadline, "{:?}", blocks.counts());
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_eq!(host.calls().len(), 1);
}

/// Waits until the source holds `out` blocks handed out and `ready` ready.
fn wait_counts(blocks: &HostBlocks, out: usize, ready: usize) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while blocks.counts() != (out, ready) {
        assert!(Instant::now() < deadline, "{:?}", blocks.counts());
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn an_empty_piece_above_the_headroom_goes_back_after_the_delay() {
    let delay = Duration::from_millis(300);
    let host = Arc::new(Host::default());
    let (pool, blocks) = pool(range(64), &host, delay);
    let n = KEPT_CALLS + 1;
    let start = Instant::now();
    for _ in 0..=KEPT {
        pool.grow(BLOCK_LEN);
    }
    host.wait_for(n);
    for i in 0..=KEPT {
        pool.added(BASE + i * BLOCK_LEN).unwrap();
    }
    // The one above the headroom is asked out of the guest, then goes back once the host says
    // it is out.
    assert_eq!(host.wait_for(n + 1)[n], Call::Remove(PAST, BLOCK_LEN));
    assert!(start.elapsed() >= delay, "{:?}", start.elapsed());
    pool.removed(PAST).unwrap();
    wait_counts(&blocks, KEPT_CALLS, 1);
    assert_eq!(pool.live(), headroom_pieces());
    assert!(pool.find(PAST, 1).is_none());
    assert_eq!(pool.piece_used(PAST), Err(Error::InvalidArgument));
    assert_eq!(host.calls().len(), n + 1);
}

#[test]
fn the_lowest_empty_pieces_stay_as_headroom() {
    let delay = Duration::from_millis(50);
    let host = Arc::new(Host::default());
    let (pool, blocks) = pool(range(64), &host, delay);
    for _ in 0..KEPT {
        pool.grow(BLOCK_LEN);
    }
    host.wait_for(KEPT_CALLS);
    for (at, _) in headroom_pieces() {
        pool.added(at).unwrap();
    }
    std::thread::sleep(delay * 6);
    assert_eq!(blocks.counts(), (KEPT_CALLS, 0));
    assert_eq!(pool.live(), headroom_pieces());
}

#[test]
fn a_piece_used_again_within_the_delay_is_kept() {
    let delay = Duration::from_millis(50);
    let (host, hand) = (Arc::new(Host::default()), Hand::default());
    let (pool, blocks) = pool_at(range(64), &host, delay, hand.clock());
    for _ in 0..KEPT {
        pool.grow(BLOCK_LEN);
    }
    host.wait_for(KEPT_CALLS);
    for (at, _) in headroom_pieces() {
        pool.added(at).unwrap();
    }
    // Using a piece of the headroom asked for a block more, so the headroom holds its bytes.
    pool.piece_used(BASE).unwrap();
    assert_eq!(
        host.wait_for(KEPT_CALLS + 1)[KEPT_CALLS],
        Call::Add(PAST, BLOCK_LEN)
    );
    pool.added(PAST).unwrap();
    // Emptied, the used piece is of the headroom and the highest is above it; used again within
    // the delay, the highest is of the headroom again and kept. The clock moves only as the test
    // moves it; the real wait after gives a pool that kept the wrong piece the time to give it
    // back.
    pool.piece_empty(BASE).unwrap();
    hand.on(delay / 2);
    pool.piece_used(BASE).unwrap();
    hand.on(delay * 2);
    std::thread::sleep(delay * 2);
    assert_eq!(blocks.counts(), (KEPT_CALLS + 1, 0));
    // Emptied for good: the lowest empty pieces, this one among them, stay as the headroom, and
    // the one above them, empty past the delay, goes.
    pool.piece_empty(BASE).unwrap();
    assert_eq!(
        host.wait_for(KEPT_CALLS + 2)[KEPT_CALLS + 1],
        Call::Remove(PAST, BLOCK_LEN)
    );
    pool.removed(PAST).unwrap();
    wait_counts(&blocks, KEPT_CALLS, 1);
    assert_eq!(pool.live(), headroom_pieces());
    assert_eq!(host.calls().len(), KEPT_CALLS + 2);
}

#[test]
fn a_block_taken_again_is_mapped_anew() {
    let host = Arc::new(Host::default());
    let (pool, blocks) = one_past(&host, Duration::from_millis(50));
    let n = KEPT_CALLS + 1;
    assert_eq!(host.wait_for(n + 1)[n], Call::Remove(PAST, BLOCK_LEN));
    pool.removed(PAST).unwrap();
    wait_counts(&blocks, KEPT_CALLS, 1);
    // A piece of the headroom is used: the pool asks for a block more, the source hands the ready
    // one out again, and the pool has the host map it at the lowest free address, where it was.
    pool.piece_used(BASE).unwrap();
    assert_eq!(host.wait_for(n + 2)[n + 1], Call::Add(PAST, BLOCK_LEN));
    wait_counts(&blocks, n, 0);
    assert_eq!(pool.live(), headroom_pieces());
    pool.added(PAST).unwrap();
    let mut live = headroom_pieces();
    live.push((PAST, BLOCK_LEN));
    assert_eq!(pool.live(), live);
}

#[test]
fn ready_blocks_past_two_are_freed_with_no_word_to_the_host() {
    let delay = Duration::from_millis(30);
    let host = Arc::new(Host::default());
    let (pool, blocks) = pool(range(64), &host, delay);
    let n = KEPT_CALLS + 3;
    for _ in 0..KEPT + 3 {
        pool.grow(BLOCK_LEN);
    }
    host.wait_for(n);
    for i in 0..KEPT + 3 {
        pool.added(BASE + i * BLOCK_LEN).unwrap();
    }
    // The three above the headroom go back, lowest first, each asked out of the guest before it
    // goes.
    for i in 0..3 {
        let at = PAST + i * BLOCK_LEN;
        let calls = n + 1 + usize::try_from(i).unwrap();
        assert_eq!(host.wait_for(calls)[calls - 1], Call::Remove(at, BLOCK_LEN));
        pool.removed(at).unwrap();
    }
    // The source keeps two ready and frees the oldest, asking no one.
    wait_counts(&blocks, KEPT_CALLS, 2);
    std::thread::sleep(delay * 4);
    assert_eq!(host.calls().len(), n + 3);
    assert!(pool.source().is_some());
}

#[test]
fn a_piece_given_back_and_taken_again_reads_zero() {
    let delay = Duration::from_millis(50);
    let (host, hand) = (Arc::new(Host::default()), Hand::default());
    let (pool, blocks) = one_past_at(&host, delay, hand.clock());
    let n = KEPT_CALLS + 1;
    // The guest writes into the piece above the headroom, then empties it; it goes back past the
    // delay, which the clock, standing still until the test moves it, cannot pass before the
    // piece is used.
    pool.piece_used(PAST).unwrap();
    {
        let hold = pool.find(PAST + 4096, 8).unwrap();
        // SAFETY: inside the block, which the hold keeps.
        unsafe { hold.host.as_ptr().cast::<u64>().write(0xdead_beef) };
    }
    pool.piece_empty(PAST).unwrap();
    hand.on(delay);
    assert_eq!(host.wait_for(n + 1)[n], Call::Remove(PAST, BLOCK_LEN));
    pool.removed(PAST).unwrap();
    wait_counts(&blocks, KEPT_CALLS, 1);
    // A piece of the headroom is used: the ready block goes out again, mapped anew at its
    // address, zeroed.
    pool.piece_used(BASE).unwrap();
    assert_eq!(host.wait_for(n + 2)[n + 1], Call::Add(PAST, BLOCK_LEN));
    pool.added(PAST).unwrap();
    wait_counts(&blocks, n, 0);
    let hold = pool.find(PAST + 4096, 8).unwrap();
    // SAFETY: as above.
    assert_eq!(unsafe { hold.host.as_ptr().cast::<u64>().read() }, 0);
    assert_eq!(host.calls().len(), n + 2);
}

/// The pool at `range(64)` with one live piece, at `BASE`, and a ledger of it.
fn live_pool(host: &Arc<Host>, delay: Duration) -> (Pool, Arc<HostBlocks>, PoolLedger) {
    let (pool, blocks) = pool(range(64), host, delay);
    pool.grow(BLOCK_LEN);
    host.wait_for(1);
    pool.added(BASE).unwrap();
    let ledger = pool.ledger();
    (pool, blocks, ledger)
}

/// [`live_pool`] with its zeroer shut before the piece is placed, so it zeroes nothing until
/// the test opens it ([`open_zeroer`]): the takes zero every byte themselves.
fn live_pool_unzeroed(host: &Arc<Host>, delay: Duration) -> (Pool, Arc<HostBlocks>, PoolLedger) {
    let (pool, blocks) = pool(range(64), host, delay);
    pool.shared.state.lock().zero_shut = true;
    pool.grow(BLOCK_LEN);
    host.wait_for(1);
    pool.added(BASE).unwrap();
    let ledger = pool.ledger();
    (pool, blocks, ledger)
}

/// Opens `pool`'s zeroer, shut by [`live_pool_unzeroed`], making one where none runs.
fn open_zeroer(pool: &Pool) {
    let mut st = pool.shared.state.lock();
    st.zero_shut = false;
    let start = st.start_zeroer();
    drop(st);
    pool.shared.wake();
    if start {
        takes::spawn_zeroer(&pool.shared);
    }
}

/// The bytes of the piece at `at` the zeroer's record says read zero.
fn zeroed(pool: &Pool, at: u64) -> u64 {
    let st = pool.shared.state.lock();
    st.pieces
        .get(&at)
        .map_or(0, |p| p.zeros.iter().map(|(first, end)| end - first).sum())
}

/// Waits until the zeroer's record of the piece at `at` holds `len` bytes, or the test fails
/// after 10 s.
fn wait_zeroed(pool: &Pool, at: u64, len: u64) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while zeroed(pool, at) != len {
        assert!(
            Instant::now() < deadline,
            "{} of {len} bytes zeroed",
            zeroed(pool, at)
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The chunk the zeroer claimed once it claimed one, or the test fails after 10 s.
fn wait_claim(pool: &Pool) -> (u64, std::ops::Range<u64>) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(claim) = pool.shared.state.lock().claim.clone() {
            return claim;
        }
        assert!(Instant::now() < deadline, "the zeroer claimed no chunk");
        std::thread::sleep(Duration::from_millis(1));
    }
}

/// The bytes the take workers of `pool` zeroed, then its zeroer.
fn zeroed_by(pool: &Pool) -> [u64; 2] {
    pool.shared.state.lock().zeroed_by
}

/// The count of `code` so far.
fn counted(code: &str) -> u64 {
    (reims_vgpu_vulkan::answer::counted_kinds().into_iter())
        .find(|(c, _)| *c == code)
        .map_or(0, |(_, n)| n)
}

type Answers = std::sync::mpsc::Receiver<Option<Taken>>;

/// Context `context`'s take of `length` bytes, its answer sent to the receiver returned.
fn take(ledger: &PoolLedger, context: u32, length: u64) -> Answers {
    let (tx, rx) = std::sync::mpsc::channel();
    let answer: TakeAnswer = Box::new(move |block| {
        let _ = tx.send(block);
    });
    ledger.take(context, length, answer).unwrap();
    rx
}

/// A take's answer, or the test fails after 10 s.
fn taken(rx: &Answers) -> Option<Taken> {
    rx.recv_timeout(Duration::from_secs(10))
        .expect("the take is answered")
}

/// A take's answer's offset and length, or the test fails after 10 s.
fn answer(rx: &Answers) -> Option<(u64, u64)> {
    taken(rx).map(|t| (t.offset, t.length))
}

/// Fills the `len` bytes at pool offset `offset` with `byte`, through the piece's mapping.
fn fill(pool: &Pool, offset: u64, len: u64, byte: u8) {
    let hold = pool.find(BASE + offset, len).unwrap();
    // SAFETY: inside the live piece, which the hold keeps.
    unsafe { std::ptr::write_bytes(hold.host.as_ptr(), byte, usize::try_from(len).unwrap()) };
}

/// Whether the `len` bytes at pool offset `offset` are all `byte`.
fn all(pool: &Pool, offset: u64, len: u64, byte: u8) -> bool {
    let hold = pool.find(BASE + offset, len).unwrap();
    // SAFETY: as in `fill`.
    let bytes =
        unsafe { std::slice::from_raw_parts(hold.host.as_ptr(), usize::try_from(len).unwrap()) };
    bytes.iter().all(|&b| b == byte)
}

#[test]
fn a_take_is_carved_zeroed_and_answered_with_its_length_rounded_up() {
    let host = Arc::new(Host::default());
    let (pool, _blocks, ledger) = live_pool_unzeroed(&host, LONG);
    fill(&pool, 0, 3 * TAKE_ALIGN, 0xa5);
    assert_eq!(
        answer(&take(&ledger, 1, 100 << 10)),
        Some((0, 2 * TAKE_ALIGN))
    );
    // The block's bytes are zero, and those past it as they were.
    assert!(all(&pool, 0, 2 * TAKE_ALIGN, 0));
    assert!(all(&pool, 2 * TAKE_ALIGN, TAKE_ALIGN, 0xa5));
    assert_eq!(ledger.counts(), (1, 2 * TAKE_ALIGN));
}

/// A take zeroes every byte of its carve no zeroer zeroed: over a given-back carve's bytes and
/// past them, bytes no carve held before, which the source's block says nothing of, are zeroed
/// as well; the bytes past the take are left as they are
#[test]
fn a_take_over_a_given_back_carve_and_past_it_zeroes_all_of_it() {
    let host = Arc::new(Host::default());
    let (pool, _blocks, ledger) = live_pool_unzeroed(&host, LONG);
    assert_eq!(answer(&take(&ledger, 1, TAKE_ALIGN)), Some((0, TAKE_ALIGN)));
    // The carve's bytes as the guest wrote them, and past them as the source's block holds them.
    fill(&pool, 0, 3 * TAKE_ALIGN, 0xa5);
    let Give::Given(carve) = ledger.give(1, 0, TAKE_ALIGN, |_, _| false) else {
        panic!("not given");
    };
    ledger.free(&[carve]);
    let len = 2 * TAKE_ALIGN;
    assert_eq!(answer(&take(&ledger, 2, len)), Some((0, len)));
    assert!(all(&pool, 0, len, 0));
    assert!(all(&pool, len, TAKE_ALIGN, 0xa5));
}

/// A freed block's bytes, in a piece another block still holds, are zeroed by the pool's zeroer
/// with no take, and the carved block beside them is left as it is; a take over them then finds
/// them zero and leaves them, counted as zeroed ahead
#[test]
fn a_freed_blocks_bytes_are_zeroed_ahead_and_a_take_over_them_leaves_them() {
    let host = Arc::new(Host::default());
    let (pool, _blocks, ledger) = live_pool(&host, LONG);
    assert_eq!(answer(&take(&ledger, 1, TAKE_ALIGN)), Some((0, TAKE_ALIGN)));
    assert_eq!(
        answer(&take(&ledger, 1, TAKE_ALIGN)),
        Some((TAKE_ALIGN, TAKE_ALIGN))
    );
    fill(&pool, 0, TAKE_ALIGN, 0xa5);
    fill(&pool, TAKE_ALIGN, TAKE_ALIGN, 0x5a);
    let zeroed = counted(ZEROED_AHEAD_BYTES_CODE);
    let Give::Given(carve) = ledger.give(1, 0, TAKE_ALIGN, |_, _| false) else {
        panic!("not given");
    };
    ledger.free(&[carve]);
    let start = Instant::now();
    while !all(&pool, 0, TAKE_ALIGN, 0) {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "the freed block's bytes are zeroed"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    while counted(ZEROED_AHEAD_BYTES_CODE) < zeroed + TAKE_ALIGN {
        assert!(start.elapsed() < Duration::from_secs(10), "counted");
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        counted(ZERO_HOLD_MAX_NS_CODE) > 0,
        "the zeroer's hold counted"
    );
    assert!(all(&pool, TAKE_ALIGN, TAKE_ALIGN, 0x5a));
    // A byte the pool's record says reads zero, made 0x77 here: the take leaves it.
    fill(&pool, 0, 1, 0x77);
    let prezeroed = counted(TAKE_ZERO_PREZEROED_BYTES_CODE);
    assert_eq!(answer(&take(&ledger, 2, TAKE_ALIGN)), Some((0, TAKE_ALIGN)));
    assert!(all(&pool, 0, 1, 0x77));
    assert!(all(&pool, 1, TAKE_ALIGN - 1, 0));
    assert!(counted(TAKE_ZERO_PREZEROED_BYTES_CODE) >= prezeroed + TAKE_ALIGN);
}

/// The device's release returns once neither a take worker nor the zeroer holds the pool: their
/// threads hold its clock, which holds the platform, so none of them drops it past the release
#[test]
fn the_release_returns_once_no_pool_thread_holds_the_pool() {
    for _ in 0..50 {
        let host = Arc::new(Host::default());
        let (mut pool, _blocks, ledger) = live_pool(&host, LONG);
        let takes: Vec<Answers> = (1..=4)
            .map(|context| take(&ledger, context, TAKE_ALIGN))
            .collect();
        for rx in &takes {
            assert!(taken(rx).is_some());
        }
        // One of the piece's four blocks freed: the zeroer runs.
        let Give::Given(carve) = ledger.give(1, 0, TAKE_ALIGN, |_, _| false) else {
            panic!("not given");
        };
        ledger.free(&[carve]);
        drop(ledger);
        pool.close();
        assert_eq!(
            Arc::strong_count(&pool.shared),
            1,
            "a pool thread still holds the pool"
        );
    }
}

#[test]
fn a_take_of_no_bytes_or_past_the_longest_is_an_invalid_argument() {
    let host = Arc::new(Host::default());
    let (_pool, _blocks, ledger) = live_pool(&host, LONG);
    for length in [0, TAKE_MAX + 1] {
        let answer: TakeAnswer = Box::new(|_| panic!("answered"));
        assert_eq!(ledger.take(1, length, answer), Err(Error::InvalidArgument));
    }
    assert_eq!(ledger.counts(), (0, 0));
}

#[test]
fn a_contexts_takes_are_answered_in_its_order_and_share_a_piece_first_fit() {
    let host = Arc::new(Host::default());
    let (_pool, _blocks, ledger) = live_pool(&host, LONG);
    let (tx, rx) = std::sync::mpsc::channel();
    for (n, length) in [TAKE_ALIGN, 2 * TAKE_ALIGN, TAKE_ALIGN]
        .into_iter()
        .enumerate()
    {
        let tx = tx.clone();
        let answer: TakeAnswer = Box::new(move |block: Option<Taken>| {
            let _ = tx.send((n, block.map(|t| (t.offset, t.length))));
        });
        ledger.take(1, length, answer).unwrap();
    }
    let got: Vec<_> = (0..3)
        .map(|_| rx.recv_timeout(Duration::from_secs(10)).unwrap())
        .collect();
    assert_eq!(
        got,
        [
            (0, Some((0, TAKE_ALIGN))),
            (1, Some((TAKE_ALIGN, 2 * TAKE_ALIGN))),
            (2, Some((3 * TAKE_ALIGN, TAKE_ALIGN))),
        ]
    );
    // Another context's take goes past them in the same piece.
    assert_eq!(
        answer(&take(&ledger, 2, TAKE_ALIGN)),
        Some((4 * TAKE_ALIGN, TAKE_ALIGN))
    );
}

#[test]
fn a_take_that_fits_no_piece_grows_the_pool_and_waits_for_the_hosts_map() {
    let host = Arc::new(Host::default());
    let (pool, _blocks) = pool(range(64), &host, LONG);
    let ledger = pool.ledger();
    let rx = take(&ledger, 1, TAKE_ALIGN);
    assert_eq!(host.wait_for(1)[0], Call::Add(BASE, BLOCK_LEN));
    std::thread::sleep(Duration::from_millis(50));
    assert!(
        rx.try_recv().is_err(),
        "answered before the host mapped the piece"
    );
    pool.added(BASE).unwrap();
    assert_eq!(answer(&rx), Some((0, TAKE_ALIGN)));
    // Its piece used, the pool asks for the headroom's pieces, the lowest next to it.
    assert_eq!(host.wait_for(2)[1], Call::Add(BASE + BLOCK_LEN, BLOCK_LEN));
}

/// A take that fits no piece is carved at its own piece's first byte as the pool's thread places
/// the piece, and its worker zeroes the carve while the host maps the piece: by the host's add's
/// answer every byte of it is zeroed, and the take is answered only once the host mapped it
#[test]
fn a_take_of_a_piece_of_its_own_is_zeroed_while_the_host_maps_it() {
    let host = Arc::new(Host::default());
    let (pool, _blocks) = pool(range(64), &host, LONG);
    let ledger = pool.ledger();
    let long = BLOCK_LEN + TAKE_ALIGN;
    let rx = take(&ledger, 1, long);
    // [`HostBlocks`] rounds a block longer than a unit up to 2 MiB.
    assert_eq!(host.wait_for(1)[0], Call::Add(BASE, BLOCK_LEN + 2 * MIB));
    let deadline = Instant::now() + Duration::from_secs(10);
    while zeroed_by(&pool)[0] < long {
        let zeroed = zeroed_by(&pool)[0];
        assert!(
            Instant::now() < deadline,
            "{zeroed} of {long} bytes zeroed before the map"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        rx.try_recv().is_err(),
        "answered before the host mapped the piece"
    );
    pool.added(BASE).unwrap();
    assert_eq!(answer(&rx), Some((0, long)));
    assert_eq!(zeroed_by(&pool)[0], long, "the take zeroed bytes again");
}

/// A zeroer's hold of the pool's lock longer than [`ZERO_HOLD_LONG`] is counted with the
/// nanoseconds of each of its parts, the chunk's pick and its mark, and each part's longest rises
/// to that part at least
#[test]
fn a_zeroers_long_hold_is_counted_with_its_parts() {
    let [pick, mark] = ZERO_LONG_HOLD_PART_NS_CODES;
    let codes = [ZERO_HOLDS_LONG_CODE, pick, mark];
    let before = codes.map(counted);
    let parts = [10_000, 45_000];
    takes::zero_held(parts.map(Duration::from_nanos));
    let after = codes.map(counted);
    // Another test's zeroer may count a long hold meanwhile, so the counts rise at least by this
    // hold's.
    assert!(after[0] > before[0], "the long hold counted");
    for (i, ns) in parts.into_iter().enumerate() {
        assert!(after[i + 1] - before[i + 1] >= ns, "{}", codes[i + 1]);
        assert!(
            counted(ZERO_HOLD_PART_MAX_NS_CODES[i]) >= ns,
            "{}",
            ZERO_HOLD_PART_MAX_NS_CODES[i]
        );
    }
    assert!(
        counted(ZERO_HOLD_MAX_NS_CODE) >= 55_000,
        "the hold's longest"
    );
}

/// The zeroer zeroes a new piece, of the headroom, ahead of every take, from its place on: a take
/// carved there once it ran zeroes nothing, and reads zero
#[test]
fn a_take_in_a_new_piece_the_zeroer_ran_over_zeroes_nothing() {
    let host = Arc::new(Host::default());
    let (pool, _blocks, ledger) = live_pool_unzeroed(&host, LONG);
    // The source's bytes, which it says nothing of.
    fill(&pool, 0, BLOCK_LEN, 0xa5);
    open_zeroer(&pool);
    wait_zeroed(&pool, BASE, BLOCK_LEN);
    assert_eq!(zeroed_by(&pool), [0, BLOCK_LEN]);
    let len = 4 * MIB;
    assert_eq!(answer(&take(&ledger, 1, len)), Some((0, len)));
    assert_eq!(zeroed_by(&pool)[0], 0, "the take zeroed bytes");
    assert!(all(&pool, 0, BLOCK_LEN, 0));
}

/// An emptied piece is zeroed once it is of the headroom, the lowest empty pieces, and kept; one
/// emptied above the headroom, which goes back whole past its delay, is not: its bytes are left
/// as the guest wrote them until it is out of the guest
#[test]
fn an_emptied_piece_is_zeroed_as_the_headroom_and_not_where_it_goes_back_whole() {
    let delay = Duration::from_millis(50);
    let (host, hand) = (Arc::new(Host::default()), Hand::default());
    let (pool, _blocks) = pool_at(range(64), &host, delay, hand.clock());
    let ledger = pool.ledger();
    let piece = |i: u64| BASE + i * BLOCK_LEN;
    // The pieces of a headroom and one more, the lowest and the highest mapped, those between
    // still being mapped: a take carves only mapped pieces.
    for _ in 0..=KEPT {
        pool.grow(BLOCK_LEN);
    }
    host.wait_for(KEPT_CALLS + 1);
    pool.added(piece(0)).unwrap();
    pool.added(PAST).unwrap();
    // The lowest taken whole, the highest is of the headroom; taken whole too, it leaves the
    // headroom a piece short, which the pool asks for.
    assert_eq!(answer(&take(&ledger, 1, BLOCK_LEN)), Some((0, BLOCK_LEN)));
    assert_eq!(
        answer(&take(&ledger, 2, BLOCK_LEN)),
        Some((PAST - BASE, BLOCK_LEN))
    );
    let next = PAST + BLOCK_LEN;
    assert_eq!(
        host.wait_for(KEPT_CALLS + 2)[KEPT_CALLS + 1],
        Call::Add(next, BLOCK_LEN)
    );
    for i in 1..KEPT {
        pool.added(piece(i)).unwrap();
        wait_zeroed(&pool, piece(i), BLOCK_LEN);
    }
    pool.added(next).unwrap();
    wait_zeroed(&pool, next, BLOCK_LEN);
    fill(&pool, 0, BLOCK_LEN, 0xa5);
    fill(&pool, PAST - BASE, BLOCK_LEN, 0x5a);
    // The first emptied is the lowest empty piece, of the headroom: zeroed.
    let Give::Given(first) = ledger.give(1, 0, BLOCK_LEN, |_, _| false) else {
        panic!("not given");
    };
    ledger.free(&[first]);
    wait_zeroed(&pool, BASE, BLOCK_LEN);
    assert!(all(&pool, 0, BLOCK_LEN, 0));
    // The second emptied lies above the headroom's pieces: it goes back whole, and nothing zeroes
    // it.
    let before = zeroed_by(&pool);
    let Give::Given(second) = ledger.give(2, PAST - BASE, BLOCK_LEN, |_, _| false) else {
        panic!("not given");
    };
    ledger.free(&[second]);
    std::thread::sleep(Duration::from_millis(50));
    assert!(all(&pool, PAST - BASE, BLOCK_LEN, 0x5a));
    assert_eq!(zeroed(&pool, PAST), 0);
    hand.on(delay);
    assert_eq!(
        host.wait_for(KEPT_CALLS + 3)[KEPT_CALLS + 2],
        Call::Remove(PAST, BLOCK_LEN)
    );
    assert_eq!(zeroed_by(&pool), before, "a piece going back was zeroed");
}

/// A take carved while the zeroer writes a chunk of the same piece is carved off that chunk, and
/// reads zero at its answer, its worker zeroing it; the zeroer's write lands on its chunk alone,
/// so the bytes the guest then writes stay as written
#[test]
fn a_take_racing_the_zeroer_reads_zero_and_keeps_what_the_guest_writes() {
    let host = Arc::new(Host::default());
    let (pool, _blocks) = pool(range(64), &host, LONG);
    pool.shared.state.lock().zero_paused = true;
    let ledger = pool.ledger();
    assert!(ledger.keep_headroom());
    host.wait_for(1);
    pool.added(BASE).unwrap();
    // The zeroer claimed the piece's top chunk, and waits before its write.
    let top = BLOCK_LEN - takes::ZERO_CHUNK;
    assert_eq!(wait_claim(&pool), (BASE, top..BLOCK_LEN));
    fill(&pool, 0, BLOCK_LEN, 0xa5);
    assert_eq!(answer(&take(&ledger, 1, top)), Some((0, top)));
    assert!(all(&pool, 0, top, 0));
    assert_eq!(zeroed_by(&pool), [top, 0]);
    fill(&pool, 0, top, 0x5a);
    pool.shared.state.lock().zero_paused = false;
    pool.shared.wake();
    wait_zeroed(&pool, BASE, takes::ZERO_CHUNK);
    assert!(all(&pool, 0, top, 0x5a));
    assert!(all(&pool, top, takes::ZERO_CHUNK, 0));
    // Its own bytes zeroed once: the take's worker zeroed none of the zeroer's chunk.
    assert_eq!(zeroed_by(&pool)[0], top);
}

/// A take whose first fit holds bytes of the zeroer's claimed chunk waits for the chunk's write,
/// counted, and is carved at that first fit, as with no zeroer: the zeroer moves no carve, and the
/// take's worker zeroes only the bytes the zeroer did not
#[test]
fn a_take_over_the_claimed_chunk_waits_for_its_write_and_keeps_its_first_fit() {
    let host = Arc::new(Host::default());
    let (pool, _blocks, ledger) = live_pool_unzeroed(&host, LONG);
    assert_eq!(
        answer(&take(&ledger, 1, 2 * TAKE_ALIGN)),
        Some((0, 2 * TAKE_ALIGN))
    );
    assert_eq!(
        answer(&take(&ledger, 2, TAKE_ALIGN)),
        Some((2 * TAKE_ALIGN, TAKE_ALIGN))
    );
    let Give::Given(carve) = ledger.give(1, 0, 2 * TAKE_ALIGN, |_, _| false) else {
        panic!("not given");
    };
    ledger.free(&[carve]);
    // The zeroer claims the top chunk of the freed bytes, the lowest run, and waits before its
    // write.
    pool.shared.state.lock().zero_paused = true;
    open_zeroer(&pool);
    assert_eq!(wait_claim(&pool), (BASE, TAKE_ALIGN..2 * TAKE_ALIGN));
    fill(&pool, 0, 2 * TAKE_ALIGN, 0xa5);
    let before = zeroed_by(&pool)[0];
    let waits = counted(TAKE_CLAIM_WAITS_CODE);
    let rx = take(&ledger, 3, 2 * TAKE_ALIGN);
    let deadline = Instant::now() + Duration::from_secs(10);
    while pool.shared.state.lock().claim_waits == 0 {
        assert!(
            Instant::now() < deadline,
            "the take did not wait for the chunk"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(rx.try_recv().is_err(), "answered past the claimed chunk");
    pool.shared.state.lock().zero_paused = false;
    pool.shared.wake();
    assert_eq!(answer(&rx), Some((0, 2 * TAKE_ALIGN)));
    assert!(all(&pool, 0, 2 * TAKE_ALIGN, 0));
    assert_eq!(zeroed_by(&pool)[0] - before, TAKE_ALIGN);
    assert!(counted(TAKE_CLAIM_WAITS_CODE) > waits);
}

/// A take that only the zeroer's claimed chunk keeps out of a piece waits for the chunk's write
/// rather than asking for a piece, counted, and is carved over it once it is written; its worker
/// zeroes the rest
#[test]
fn a_take_only_the_claimed_chunk_keeps_out_waits_for_its_write_and_asks_no_piece() {
    let host = Arc::new(Host::default());
    let (pool, _blocks) = pool(range(64), &host, LONG);
    pool.shared.state.lock().zero_paused = true;
    let ledger = pool.ledger();
    assert!(ledger.keep_headroom());
    host.wait_for(1);
    pool.added(BASE).unwrap();
    let top = BLOCK_LEN - takes::ZERO_CHUNK;
    assert_eq!(wait_claim(&pool), (BASE, top..BLOCK_LEN));
    fill(&pool, 0, BLOCK_LEN, 0xa5);
    let waits = counted(TAKE_CLAIM_WAITS_CODE);
    let rx = take(&ledger, 1, BLOCK_LEN);
    let deadline = Instant::now() + Duration::from_secs(10);
    while pool.shared.state.lock().claim_waits == 0 {
        assert!(
            Instant::now() < deadline,
            "the take did not wait for the chunk"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(rx.try_recv().is_err(), "answered over the claimed chunk");
    pool.shared.state.lock().zero_paused = false;
    pool.shared.wake();
    assert_eq!(answer(&rx), Some((0, BLOCK_LEN)));
    assert!(all(&pool, 0, BLOCK_LEN, 0));
    // The worker zeroed all but the zeroer's chunk.
    assert_eq!(zeroed_by(&pool)[0], top);
    assert!(counted(TAKE_CLAIM_WAITS_CODE) > waits);
    assert!(counted(TAKE_CLAIM_WAIT_MAX_NS_CODE) > 0);
    // No piece was asked for the take: past the headroom's is the one more the headroom asked
    // once the take used this.
    let mut adds = headroom_adds();
    adds.push(Call::Add(PAST, BLOCK_LEN));
    assert_eq!(host.wait_for(KEPT_CALLS + 1), adds);
}

/// An answered take's wait is counted by its cause ([`TAKE_WAIT_CODES`]), the take counted once
/// its waits are: a take that fits no piece waits for one of its own, which the host maps 50 ms
/// after its ask, and that wait, but for the zeroing its worker does meanwhile, is its growth
/// part's and at most the longest's; its parts lie within its whole wait
#[test]
fn a_takes_wait_is_counted_by_its_cause() {
    let codes = || {
        [
            TAKE_WAITS_CODE,
            TAKE_WAIT_CODES[0].0,
            TAKE_WAIT_CODES[1].0,
            TAKE_WAIT_CODES[2].0,
            TAKE_WAIT_CODES[3].0,
            TAKE_WAIT_CODES[2].1,
        ]
        .map(counted)
    };
    let before = codes();
    let host = Arc::new(Host::default());
    let (pool, _blocks) = pool(range(64), &host, LONG);
    let ledger = pool.ledger();
    let rx = take(&ledger, 1, TAKE_ALIGN);
    assert_eq!(host.wait_for(1)[0], Call::Add(BASE, BLOCK_LEN));
    let held = Duration::from_millis(50);
    std::thread::sleep(held);
    pool.added(BASE).unwrap();
    assert_eq!(answer(&rx), Some((0, TAKE_ALIGN)));
    // Counted on the take's worker, just after its answer.
    let start = Instant::now();
    while counted(TAKE_WAITS_CODE) == before[0] {
        assert!(
            start.elapsed() < Duration::from_secs(10),
            "the take is counted"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    let after = codes();
    let grew = |i: usize| after[i] - before[i];
    let held = u64::try_from(held.as_nanos()).unwrap();
    // Its zeroing ran while the host mapped the piece: that time is the zeroing's.
    let piece = grew(3) + grew(4);
    assert!(piece >= held, "waited for a piece, zeroing: {piece} ns");
    assert!(after[5] + grew(4) >= held, "the longest wait for a piece");
    assert!(
        grew(2) + grew(3) + grew(4) <= grew(1),
        "the parts within the whole"
    );
}

/// The guest's driver's take of the pool asks for the headroom's pieces ahead of the first take,
/// which that take then finds, answered with no piece of its own asked or mapped; taken again
/// with the pieces still empty, as after a reset, it asks none
#[test]
fn the_pieces_asked_ahead_of_the_first_take_serve_it_with_no_wait() {
    let host = Arc::new(Host::default());
    let (pool, _blocks) = pool(range(64), &host, LONG);
    let ledger = pool.ledger();
    assert!(ledger.keep_headroom());
    assert_eq!(host.wait_for(KEPT_CALLS), headroom_adds());
    assert!(
        !ledger.keep_headroom(),
        "asked again while the host maps them"
    );
    for (at, _) in headroom_pieces() {
        pool.added(at).unwrap();
    }
    assert!(!ledger.keep_headroom(), "asked again");
    assert_eq!(answer(&take(&ledger, 1, TAKE_ALIGN)), Some((0, TAKE_ALIGN)));
    // Its piece used, the pool asks for one more as headroom, which the take did not wait for.
    assert_eq!(
        host.wait_for(KEPT_CALLS + 1)[KEPT_CALLS],
        Call::Add(PAST, BLOCK_LEN)
    );
}

/// A take carved from a piece of the headroom has the pool's thread make the next at once, not
/// once the take's worker went idle: on a clock that stands still, as a test platform's does, the
/// worker never goes
#[test]
fn a_take_that_uses_a_headroom_piece_has_the_next_made_at_once() {
    let host = Arc::new(Host::default());
    let still = (now())();
    let clock: PoolNow = Box::new(move || still);
    let source: Arc<dyn BlockSource> = Arc::new(HostBlocks::new());
    let pool = Pool::with_delay(range(64), host.pool_host(), LONG, clock, source).unwrap();
    let ledger = pool.ledger();
    assert!(ledger.keep_headroom());
    assert_eq!(host.wait_for(KEPT_CALLS), headroom_adds());
    pool.added(BASE).unwrap();
    assert_eq!(answer(&take(&ledger, 1, TAKE_ALIGN)), Some((0, TAKE_ALIGN)));
    assert_eq!(
        host.wait_for(KEPT_CALLS + 1)[KEPT_CALLS],
        Call::Add(PAST, BLOCK_LEN)
    );
}

/// A source whose takes wait until the test opens its gate.
struct Gated {
    blocks: HostBlocks,
    open: Mutex<bool>,
    opened: Condvar,
}

impl BlockSource for Gated {
    fn take_block(&self, class: MemoryClass, len: u64) -> Result<BlockInfo, BlockError> {
        let mut open = self.open.lock();
        while !*open {
            self.opened.wait(&mut open);
        }
        drop(open);
        self.blocks.take_block(class, len)
    }

    fn release_block(&self, block: BlockId) -> Result<(), BlockError> {
        self.blocks.release_block(block)
    }
}

/// A source whose takes of the lengths `gated` names wait until the test opens its gate, counted
/// as they wait; the rest are [`HostBlocks`]'.
struct Held {
    blocks: HostBlocks,
    gated: fn(u64) -> bool,
    open: Mutex<bool>,
    waiting: Mutex<usize>,
    changed: Condvar,
}

impl Held {
    fn new(gated: fn(u64) -> bool) -> Arc<Held> {
        Arc::new(Held {
            blocks: HostBlocks::new(),
            gated,
            open: Mutex::new(false),
            waiting: Mutex::new(0),
            changed: Condvar::new(),
        })
    }

    /// The pool over `range(64)` on `host` and `clock`, its blocks from `self`.
    fn pool(self: &Arc<Self>, host: &Arc<Host>, delay: Duration, clock: PoolNow) -> Pool {
        let source: Arc<dyn BlockSource> = Arc::clone(self) as _;
        Pool::with_delay(range(64), host.pool_host(), delay, clock, source).unwrap()
    }

    /// Waits until `n` takes wait at the gate, or the test fails after 10 s.
    fn wait_waiting(&self, n: usize) {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut waiting = self.waiting.lock();
        while *waiting < n {
            assert!(
                !self.changed.wait_until(&mut waiting, deadline).timed_out(),
                "{} takes wait at the gate, not {n}",
                *waiting
            );
        }
    }

    /// Opens the gate.
    fn open(&self) {
        *self.open.lock() = true;
        self.changed.notify_all();
    }

    /// A guard that opens the gate at its drop: made after the pool, it drops first, so a test
    /// that fails with a take at the gate still closes its pool.
    fn opener(self: &Arc<Self>) -> Opener {
        Opener(Arc::clone(self))
    }
}

/// [`Held::opener`].
struct Opener(Arc<Held>);

impl Drop for Opener {
    fn drop(&mut self) {
        self.0.open();
    }
}

impl BlockSource for Held {
    fn take_block(&self, class: MemoryClass, len: u64) -> Result<BlockInfo, BlockError> {
        if (self.gated)(len) {
            *self.waiting.lock() += 1;
            self.changed.notify_all();
            let mut open = self.open.lock();
            while !*open {
                self.changed.wait(&mut open);
            }
        }
        self.blocks.take_block(class, len)
    }

    fn release_block(&self, block: BlockId) -> Result<(), BlockError> {
        self.blocks.release_block(block)
    }
}

/// A take that fits no piece takes a block of its own length on its context's worker, and the
/// pool's thread places and maps it ahead of the headroom's: it waits behind none of the blocks
/// the pool's taker takes, which here wait at the source
#[test]
fn a_take_of_a_piece_of_its_own_waits_behind_no_headroom_block() {
    let host = Arc::new(Host::default());
    let source = Held::new(|len| len <= BLOCK_LEN);
    let pool = source.pool(&host, LONG, now());
    let _open = source.opener();
    let ledger = pool.ledger();
    assert!(ledger.keep_headroom());
    source.wait_waiting(1);
    let long = BLOCK_LEN + TAKE_ALIGN;
    let rx = take(&ledger, 1, long);
    // [`HostBlocks`] rounds a block longer than a unit up to 2 MiB.
    let own = BLOCK_LEN + 2 * MIB;
    assert_eq!(host.wait_for(1), [Call::Add(BASE, own)]);
    pool.added(BASE).unwrap();
    assert_eq!(answer(&rx), Some((0, long)));
    // The headroom's blocks come once the source gives them, placed past the take's.
    source.open();
    let adds: Vec<_> = (0..KEPT)
        .map(|i| Call::Add(BASE + own + i * BLOCK_LEN, BLOCK_LEN))
        .collect();
    assert_eq!(host.wait_for(1 + KEPT_CALLS)[1..], adds[..]);
}

/// While a take's own block is being made (here: it waits at the source), the rest of the pool
/// goes on: a piece empty past its delay is asked out of the guest, its remove left unanswered by
/// the host; another context's take is carved and answered; and the next headroom piece its take
/// asked is taken, placed past the piece going back, and mapped. The take's block, once it comes,
/// is placed past them and answered, and the piece going back goes once the host removed it
#[test]
fn a_takes_own_block_holds_up_no_other_take_headroom_or_give_back() {
    let delay = Duration::from_millis(50);
    let (host, hand) = (Arc::new(Host::default()), Hand::default());
    let source = Held::new(|len| len > BLOCK_LEN);
    let pool = source.pool(&host, delay, hand.clock());
    let _open = source.opener();
    for _ in 0..=KEPT {
        pool.grow(BLOCK_LEN);
    }
    let n = KEPT_CALLS + 1;
    host.wait_for(n);
    for i in 0..=KEPT {
        pool.added(BASE + i * BLOCK_LEN).unwrap();
    }
    let ledger = pool.ledger();
    let long = BLOCK_LEN + TAKE_ALIGN;
    let rx = take(&ledger, 1, long);
    source.wait_waiting(1);
    hand.on(delay);
    assert_eq!(host.wait_for(n + 1)[n], Call::Remove(PAST, BLOCK_LEN));
    assert_eq!(answer(&take(&ledger, 2, TAKE_ALIGN)), Some((0, TAKE_ALIGN)));
    let next = PAST + BLOCK_LEN;
    assert_eq!(host.wait_for(n + 2)[n + 1], Call::Add(next, BLOCK_LEN));
    pool.added(next).unwrap();
    assert!(rx.try_recv().is_err(), "answered before its block came");
    source.open();
    let at = next + BLOCK_LEN;
    assert_eq!(
        host.wait_for(n + 3)[n + 2],
        Call::Add(at, BLOCK_LEN + 2 * MIB)
    );
    pool.added(at).unwrap();
    assert_eq!(answer(&rx), Some((at - BASE, long)));
    pool.removed(PAST).unwrap();
    wait_counts(&source.blocks, n + 1, 1);
    assert_eq!(host.calls().len(), n + 3);
}

/// A block the source hands out once the device's release began goes back to it, and the host
/// is asked to map nothing: it removed its pieces before the release
#[test]
fn a_block_made_during_the_release_goes_back_with_no_add() {
    let host = Arc::new(Host::default());
    let gated = Arc::new(Gated {
        blocks: HostBlocks::new(),
        open: Mutex::new(false),
        opened: Condvar::new(),
    });
    let source: Arc<dyn BlockSource> = Arc::clone(&gated) as _;
    let pool = Pool::with_delay(range(64), host.pool_host(), LONG, now(), source).unwrap();
    pool.grow(BLOCK_LEN);
    // The pool's thread waits in the source's take.
    let deadline = Instant::now() + Duration::from_secs(10);
    while !pool.shared.state.lock().asks.is_empty() {
        assert!(Instant::now() < deadline, "the ask was not taken up");
        std::thread::sleep(Duration::from_millis(1));
    }
    let shared = Arc::clone(&pool.shared);
    let release = std::thread::spawn(move || drop(pool));
    while !shared.state.lock().closed {
        assert!(Instant::now() < deadline, "the release did not begin");
        std::thread::sleep(Duration::from_millis(1));
    }
    *gated.open.lock() = true;
    gated.opened.notify_all();
    release.join().unwrap();
    assert!(host.calls().is_empty(), "the pool asked {:?}", host.calls());
    assert_eq!(gated.blocks.counts().0, 0, "the block went back");
}

#[test]
fn a_take_whose_context_closed_before_its_answer_is_given_back_unanswered() {
    let host = Arc::new(Host::default());
    let (pool, _blocks) = pool(range(64), &host, LONG);
    let ledger = pool.ledger();
    let rx = take(&ledger, 1, TAKE_ALIGN);
    host.wait_for(1);
    assert_eq!(ledger.context_closed(1), []);
    assert_eq!(
        rx.recv_timeout(Duration::from_secs(10)),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected)
    );
    pool.added(BASE).unwrap();
    assert_eq!(ledger.counts(), (0, 0));
    // The context's number, given out again, takes anew.
    assert_eq!(answer(&take(&ledger, 1, TAKE_ALIGN)), Some((0, TAKE_ALIGN)));
}

#[test]
fn a_give_finds_only_the_contexts_block_as_answered_and_frees_it_at_its_free() {
    let host = Arc::new(Host::default());
    let (_pool, _blocks, ledger) = live_pool(&host, LONG);
    let len = 2 * TAKE_ALIGN;
    assert_eq!(answer(&take(&ledger, 1, len)), Some((0, len)));
    let free = |_, _| false;
    assert_eq!(ledger.give(2, 0, len, free), Give::NotFound);
    assert_eq!(ledger.give(1, 0, TAKE_ALIGN, free), Give::NotFound);
    assert_eq!(ledger.give(1, TAKE_ALIGN, TAKE_ALIGN, free), Give::NotFound);
    // An object in it keeps it the context's.
    let in_use = ledger.give(1, 0, len, |offset, length| {
        assert_eq!((offset, length), (0, len));
        true
    });
    assert_eq!(in_use, Give::InUse);
    assert!(ledger.over_held(1, 0, len, |_, _| ()).is_some());
    let Give::Given(carve) = ledger.give(1, 0, len, free) else {
        panic!("not given");
    };
    // Going: no heap is made over it, and no second give finds it, until its free.
    assert!(ledger.over_held(1, 0, len, |_, _| ()).is_none());
    assert_eq!(ledger.give(1, 0, len, free), Give::NotFound);
    assert_eq!(ledger.counts(), (1, len));
    ledger.free(&[carve]);
    assert_eq!(ledger.counts(), (0, 0));
    assert_eq!(answer(&take(&ledger, 3, TAKE_ALIGN)), Some((0, TAKE_ALIGN)));
}

#[test]
fn a_heap_is_made_only_over_bytes_in_one_block_the_context_holds() {
    let host = Arc::new(Host::default());
    let (pool, _blocks, ledger) = live_pool(&host, LONG);
    let len = 2 * TAKE_ALIGN;
    assert_eq!(answer(&take(&ledger, 1, len)), Some((0, len)));
    let first = pool.find(BASE + TAKE_ALIGN, 1).unwrap().host;
    assert_eq!(
        ledger.over_held(1, TAKE_ALIGN, TAKE_ALIGN, |p, _| p),
        Some(first)
    );
    assert_eq!(ledger.over_held(1, TAKE_ALIGN, len, |p, _| p), None);
    assert_eq!(ledger.over_held(2, 0, TAKE_ALIGN, |p, _| p), None);
    assert_eq!(ledger.over_held(1, len, TAKE_ALIGN, |p, _| p), None);
}

#[test]
fn a_close_lets_go_of_every_block_of_its_context_and_no_other() {
    let host = Arc::new(Host::default());
    let (_pool, _blocks, ledger) = live_pool(&host, LONG);
    answer(&take(&ledger, 1, TAKE_ALIGN)).unwrap();
    answer(&take(&ledger, 2, TAKE_ALIGN)).unwrap();
    answer(&take(&ledger, 1, TAKE_ALIGN)).unwrap();
    let gone = ledger.context_closed(1);
    assert_eq!(gone.len(), 2);
    assert_eq!(ledger.give(1, 0, TAKE_ALIGN, |_, _| false), Give::NotFound);
    assert_eq!(ledger.counts(), (3, 3 * TAKE_ALIGN));
    ledger.free(&gone);
    assert_eq!(ledger.counts(), (1, TAKE_ALIGN));
    assert!(
        ledger
            .over_held(2, TAKE_ALIGN, TAKE_ALIGN, |_, _| ())
            .is_some()
    );
}

#[test]
fn a_free_after_the_reset_frees_nothing_carved_since() {
    let host = Arc::new(Host::default());
    let (_pool, _blocks, ledger) = live_pool(&host, LONG);
    answer(&take(&ledger, 1, TAKE_ALIGN)).unwrap();
    let Give::Given(carve) = ledger.give(1, 0, TAKE_ALIGN, |_, _| false) else {
        panic!("not given");
    };
    ledger.reset();
    assert_eq!(ledger.counts(), (0, 0));
    assert_eq!(answer(&take(&ledger, 3, TAKE_ALIGN)), Some((0, TAKE_ALIGN)));
    ledger.free(&[carve]);
    assert_eq!(ledger.counts(), (1, TAKE_ALIGN));
}

/// Waits, yielding, until `done`, or the test fails after 10 s, saying `what`.
fn spin_until(what: &str, done: impl Fn() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !done() {
        assert!(Instant::now() < deadline, "{what}");
        std::thread::yield_now();
    }
}

/// A keep of a piece's block, which an import of its bytes holds until the platform frees the
/// import, late, holds the block at the release's end as a hold does: the release's end sleeps
/// on the pool's bell while the keep lives, and gives the block back to the source once the
/// keep's drop rang it
#[test]
fn the_release_gives_a_kept_piece_back_only_once_its_keep_goes() {
    let host = Arc::new(Host::default());
    let (pool, blocks, ledger) = live_pool(&host, LONG);
    answer(&take(&ledger, 1, TAKE_ALIGN)).unwrap();
    let keep = ledger.over_held(1, 0, TAKE_ALIGN, |_, keep| keep);
    let keep = keep.expect("a keep of the block");
    // The headroom's pieces, asked once the take used the first.
    host.wait_for(KEPT_CALLS + 1);
    let bell = Arc::clone(&pool.shared.bell);
    // The pool's thread ends in the device's release: from here only the release's end sleeps on
    // the bell, and only while a piece is held or kept.
    drop(pool);
    let slept = bell.slept();
    let release = std::thread::spawn(move || drop(ledger));
    spin_until("the release's end sleeps on the keep", || {
        bell.slept() > slept
    });
    assert!(!release.is_finished(), "the release waits for the keep");
    assert_eq!(blocks.counts(), (KEPT_CALLS + 1, 0));
    drop(keep);
    release.join().unwrap();
    assert_eq!(blocks.counts(), (0, 2));
}

/// A piece whose block a keep holds, as an import of its bytes does until the platform frees the
/// import, late: on its way back past the delay and out of the guest, it stays the pool's, its
/// block out of the source and its address taken, until the keep's drop, after which the pool's
/// thread gives the block back
#[test]
fn a_kept_piece_goes_back_only_once_its_keep_goes() {
    let delay = Duration::from_millis(50);
    let (host, hand) = (Arc::new(Host::default()), Hand::default());
    let (pool, blocks) = pool_at(range(64), &host, delay, hand.clock());
    // The headroom's pieces and two above it: `PAST`, which the guest uses, and `kept`.
    let n = KEPT_CALLS + 2;
    for _ in 0..n {
        pool.grow(BLOCK_LEN);
    }
    host.wait_for(n);
    for i in 0..KEPT + 2 {
        pool.added(BASE + i * BLOCK_LEN).unwrap();
    }
    let kept = PAST + BLOCK_LEN;
    pool.piece_used(PAST).unwrap();
    let keep = (pool.shared.state.lock()).keep_of(kept, &pool.shared.bell);
    let keep = keep.expect("a keep of the block");
    // Past the delay, the empty piece above the headroom is asked out of the guest.
    hand.on(delay);
    assert_eq!(host.wait_for(n + 1)[n], Call::Remove(kept, BLOCK_LEN));
    pool.removed(kept).unwrap();
    // The pool's thread looks for a block to give back before it sends a piece due on its way, and
    // `PAST` comes due only once the host removed `kept`: by `PAST`'s remove, it found `kept`'s
    // block kept.
    pool.piece_empty(PAST).unwrap();
    hand.on(delay);
    assert_eq!(host.wait_for(n + 2)[n + 1], Call::Remove(PAST, BLOCK_LEN));
    assert_eq!(blocks.counts(), (n, 0));
    pool.removed(PAST).unwrap();
    spin_until("`PAST`'s block back", || blocks.counts() == (n - 1, 1));
    // `kept`'s address stays taken: of two blocks more, the first goes where `PAST` was, and the
    // second past `kept`.
    pool.grow(BLOCK_LEN);
    pool.grow(BLOCK_LEN);
    assert_eq!(
        host.wait_for(n + 4)[n + 2..],
        [
            Call::Add(PAST, BLOCK_LEN),
            Call::Add(kept + BLOCK_LEN, BLOCK_LEN),
        ]
    );
    assert_eq!(blocks.counts(), (n + 1, 0));
    assert!(pool.shared.state.lock().pieces.contains_key(&kept));
    drop(keep);
    spin_until("`kept`'s block back", || blocks.counts() == (n, 1));
    assert!(!pool.shared.state.lock().pieces.contains_key(&kept));
    assert_eq!(host.calls().len(), n + 4);
}

#[test]
fn a_piece_with_a_block_stays_and_goes_back_once_its_last_block_is_freed() {
    let delay = Duration::from_millis(50);
    let host = Arc::new(Host::default());
    let (pool, _blocks, ledger) = live_pool(&host, delay);
    // The first piece full, the pool asks for the headroom's pieces, the lowest of which the next
    // take uses.
    assert_eq!(answer(&take(&ledger, 1, BLOCK_LEN)), Some((0, BLOCK_LEN)));
    let calls = host.wait_for(KEPT_CALLS + 1);
    for i in 1..=KEPT {
        let at = BASE + i * BLOCK_LEN;
        assert_eq!(calls[usize::try_from(i).unwrap()], Call::Add(at, BLOCK_LEN));
        pool.added(at).unwrap();
    }
    assert_eq!(
        answer(&take(&ledger, 1, TAKE_ALIGN)),
        Some((BLOCK_LEN, TAKE_ALIGN))
    );
    let top = PAST + BLOCK_LEN;
    assert_eq!(
        host.wait_for(KEPT_CALLS + 2)[KEPT_CALLS + 1],
        Call::Add(top, BLOCK_LEN)
    );
    pool.added(top).unwrap();
    // Both used pieces stay past the delay, and the empty ones are the headroom.
    std::thread::sleep(delay * 6);
    assert_eq!(host.calls().len(), KEPT_CALLS + 2);
    // The first emptied, it is the lowest empty piece: the highest, above the headroom, goes back.
    let Give::Given(carve) = ledger.give(1, 0, BLOCK_LEN, |_, _| false) else {
        panic!("not given");
    };
    ledger.free(&[carve]);
    assert_eq!(
        host.wait_for(KEPT_CALLS + 3)[KEPT_CALLS + 2],
        Call::Remove(top, BLOCK_LEN)
    );
    assert_eq!(host.calls().len(), KEPT_CALLS + 3);
}

#[test]
fn the_release_leaves_the_blocks_with_a_live_ledger_until_its_drop() {
    let host = Arc::new(Host::default());
    let (pool, blocks, ledger) = live_pool(&host, LONG);
    answer(&take(&ledger, 1, TAKE_ALIGN)).unwrap();
    // The headroom's pieces, asked once the take used the first.
    host.wait_for(KEPT_CALLS + 1);
    drop(pool);
    assert_eq!(blocks.counts(), (KEPT_CALLS + 1, 0));
    // A take after the release is never answered.
    let rx = take(&ledger, 2, TAKE_ALIGN);
    assert!(rx.recv_timeout(Duration::from_secs(1)).is_err());
    drop(ledger);
    // The source keeps two ready and frees the rest.
    assert_eq!(blocks.counts(), (0, 2));
}

/// A source whose blocks have GPU addresses, as metal2vulkan-runtime's do: [`HostBlocks`]' blocks,
/// the `n`th handed out at `ADDRESSED + n * BLOCK_LEN`.
struct Addressed {
    blocks: HostBlocks,
    handed: Mutex<u64>,
}

const ADDRESSED: u64 = 0x7_0000_0000;

impl BlockSource for Addressed {
    fn take_block(&self, class: MemoryClass, len: u64) -> Result<BlockInfo, BlockError> {
        let mut block = self.blocks.take_block(class, len)?;
        let mut handed = self.handed.lock();
        block.gpu_address = ADDRESSED + *handed * BLOCK_LEN;
        *handed += 1;
        Ok(block)
    }

    fn release_block(&self, block: BlockId) -> Result<(), BlockError> {
        self.blocks.release_block(block)
    }
}

/// On a source whose blocks have no GPU address, the host picks each block's from 2^45 up, next
/// fit, each holding its pages and a guard page on the 32 KiB granule, and a freed block's
/// address is not picked again before the picks wrap
#[test]
fn a_block_of_a_source_with_no_addresses_takes_the_hosts_next_pick() {
    let host = Arc::new(Host::default());
    let (_pool, _blocks, ledger) = live_pool(&host, LONG);
    let first = taken(&take(&ledger, 1, 100 << 10)).unwrap();
    assert_eq!((first.offset, first.gpu_address), (0, addresses::HOST_BASE));
    let second = taken(&take(&ledger, 2, TAKE_ALIGN)).unwrap();
    assert_eq!(second.offset, 2 * TAKE_ALIGN);
    assert_eq!(second.gpu_address, addresses::HOST_BASE + 0x2_8000);
    let Give::Given(carve) = ledger.give(1, 0, 2 * TAKE_ALIGN, |_, _| false) else {
        panic!("not given");
    };
    ledger.free(&[carve]);
    // The freed first block's bytes are carved again, at the next pick's address.
    let third = taken(&take(&ledger, 3, TAKE_ALIGN)).unwrap();
    assert_eq!(third.offset, 0);
    assert_eq!(
        third.gpu_address,
        addresses::HOST_BASE + 0x2_8000 + 0x1_8000
    );
    // A reset frees every pick with its carve.
    ledger.reset();
    assert_eq!(ledger.picked(), 0);
}

/// On a source whose blocks have GPU addresses, a block's address is its piece's block's at the
/// block's offset in the piece, so the guest's heap over it is a range of the source's block
#[test]
fn a_block_of_a_source_with_addresses_lies_at_its_offset_in_its_piece_s_block() {
    let host = Arc::new(Host::default());
    let source = Arc::new(Addressed {
        blocks: HostBlocks::new(),
        handed: Mutex::new(0),
    });
    let pool = Pool::with_delay(range(64), host.pool_host(), LONG, now(), source).unwrap();
    pool.grow(BLOCK_LEN);
    host.wait_for(1);
    pool.added(BASE).unwrap();
    let ledger = pool.ledger();
    let first = taken(&take(&ledger, 1, TAKE_ALIGN)).unwrap();
    assert_eq!((first.offset, first.gpu_address), (0, ADDRESSED));
    let second = taken(&take(&ledger, 1, 3 * TAKE_ALIGN)).unwrap();
    assert_eq!(second.offset, TAKE_ALIGN);
    assert_eq!(second.gpu_address, ADDRESSED + TAKE_ALIGN);
    // The rest of the first piece, then the headroom's lowest piece, the second block handed
    // out.
    let rest = BLOCK_LEN - 4 * TAKE_ALIGN;
    assert_eq!(
        answer(&take(&ledger, 1, rest)),
        Some((4 * TAKE_ALIGN, rest))
    );
    host.wait_for(2);
    pool.added(BASE + BLOCK_LEN).unwrap();
    let next = taken(&take(&ledger, 2, TAKE_ALIGN)).unwrap();
    assert_eq!(next.offset, BLOCK_LEN);
    assert_eq!(next.gpu_address, ADDRESSED + BLOCK_LEN);
    assert_eq!(ledger.picked(), 0);
}

/// A source whose blocks' mappings start `skew` bytes into the host blocks', with the blocks given
/// back.
struct Skewed {
    blocks: HostBlocks,
    skew: usize,
    back: Mutex<Vec<BlockId>>,
}

impl BlockSource for Skewed {
    fn take_block(&self, class: MemoryClass, len: u64) -> Result<BlockInfo, BlockError> {
        let mut block = self.blocks.take_block(class, len)?;
        if let Some(c) = block.contents {
            // SAFETY: inside the block, which is longer than the skew.
            block.contents = Some(unsafe { NonNull::new_unchecked(c.as_ptr().add(self.skew)) });
        }
        Ok(block)
    }

    fn release_block(&self, block: BlockId) -> Result<(), BlockError> {
        self.back.lock().push(block);
        self.blocks.release_block(block)
    }
}

/// A block whose mapping does not start at a host page goes back to the source unmapped, counted
/// (`rt-pool-block-unaligned`), and the host is asked nothing; one at a page is mapped.
#[test]
fn a_block_off_the_host_s_pages_goes_back_unmapped() {
    let counted = || {
        (reims_vgpu_vulkan::answer::counted_kinds().into_iter())
            .find(|&(code, _)| code == UNALIGNED_CODE)
            .map_or(0, |(_, n)| n)
    };
    for (skew, mapped) in [(64, false), (0, true)] {
        let host = Arc::new(Host::default());
        let source = Arc::new(Skewed {
            blocks: HostBlocks::new(),
            skew,
            back: Mutex::default(),
        });
        let before = counted();
        let given: Arc<dyn BlockSource> = Arc::clone(&source) as _;
        let pool = Pool::with_delay(range(4), host.pool_host(), LONG, now(), given).unwrap();
        pool.grow(BLOCK_LEN);
        if mapped {
            assert_eq!(host.wait_for(1), [Call::Add(BASE, BLOCK_LEN)]);
            assert!(source.back.lock().is_empty());
            continue;
        }
        let deadline = Instant::now() + Duration::from_secs(10);
        while source.back.lock().is_empty() {
            assert!(Instant::now() < deadline, "no block went back");
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(source.back.lock().len(), 1);
        assert!(counted() > before);
        assert!(host.calls().is_empty());
    }
}
