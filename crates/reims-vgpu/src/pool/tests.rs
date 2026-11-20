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

/// The pool at `range` with two live pieces, the upper one empty past `delay` and so given back
/// once its turn comes.
fn two_pieces(host: &Arc<Host>, delay: Duration) -> (Pool, Arc<HostBlocks>) {
    two_pieces_at(host, delay, now())
}

/// [`two_pieces`], on `clock`.
fn two_pieces_at(host: &Arc<Host>, delay: Duration, clock: PoolNow) -> (Pool, Arc<HostBlocks>) {
    let (pool, blocks) = pool_at(range(64), host, delay, clock);
    pool.grow(BLOCK_LEN);
    pool.grow(BLOCK_LEN);
    host.wait_for(2);
    pool.added(BASE).unwrap();
    pool.added(BASE + BLOCK_LEN).unwrap();
    (pool, blocks)
}

#[test]
fn a_piece_goes_back_only_once_the_host_removed_it() {
    let host = Arc::new(Host::default());
    let (pool, blocks) = two_pieces(&host, Duration::from_millis(50));
    // The upper piece, empty past the delay, is asked out of the guest, and is out of the
    // lookups.
    assert_eq!(
        host.wait_for(3)[2],
        Call::Remove(BASE + BLOCK_LEN, BLOCK_LEN)
    );
    assert!(pool.find(BASE + BLOCK_LEN, 1).is_none());
    // Its block stays the pool's until the host says the piece is out.
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(blocks.counts(), (2, 0));
    pool.removed(BASE + BLOCK_LEN).unwrap();
    wait_counts(&blocks, 1, 1);
    assert_eq!(pool.live(), [(BASE, BLOCK_LEN)]);
    // The address is free again: the next block goes there.
    pool.grow(BLOCK_LEN);
    assert_eq!(host.wait_for(4)[3], Call::Add(BASE + BLOCK_LEN, BLOCK_LEN));
}

#[test]
fn a_held_piece_goes_back_only_once_its_hold_ends() {
    let delay = Duration::from_millis(50);
    let (host, hand) = (Arc::new(Host::default()), Hand::default());
    let (pool, blocks) = two_pieces_at(&host, delay, hand.clock());
    // The clock stands: the upper piece is live as the hold is taken.
    let hold = pool.find(BASE + BLOCK_LEN, 1).unwrap();
    hand.on(delay);
    // Out of the lookups past the delay, but neither asked out of the guest nor given back while
    // the hold lives.
    let deadline = Instant::now() + Duration::from_secs(10);
    while pool.find(BASE + BLOCK_LEN, 1).is_some() {
        assert!(
            Instant::now() < deadline,
            "still in the lookups past the delay"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(host.calls().len(), 2, "no remove while a hold lives");
    assert_eq!(blocks.counts(), (2, 0));
    drop(hold);
    assert_eq!(
        host.wait_for(3)[2],
        Call::Remove(BASE + BLOCK_LEN, BLOCK_LEN)
    );
    assert_eq!(blocks.counts(), (2, 0));
    pool.removed(BASE + BLOCK_LEN).unwrap();
    wait_counts(&blocks, 1, 1);
}

#[test]
fn a_piece_given_back_before_the_hosts_add_answers_still_waits_for_the_remove() {
    let host = Arc::new(Host::default());
    let (pool, blocks) = pool(range(64), &host, Duration::from_millis(50));
    pool.grow(BLOCK_LEN);
    pool.grow(BLOCK_LEN);
    host.wait_for(2);
    pool.added(BASE).unwrap();
    // The upper piece goes back before the host's add answers: its remove is asked all the same.
    assert_eq!(
        host.wait_for(3)[2],
        Call::Remove(BASE + BLOCK_LEN, BLOCK_LEN)
    );
    // The host's add, which it queued first, then its remove.
    pool.added(BASE + BLOCK_LEN).unwrap();
    assert!(pool.find(BASE + BLOCK_LEN, 1).is_none());
    assert_eq!(blocks.counts(), (2, 0));
    pool.removed(BASE + BLOCK_LEN).unwrap();
    wait_counts(&blocks, 1, 1);
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
    let start = Instant::now();
    pool.grow(BLOCK_LEN);
    pool.grow(BLOCK_LEN);
    host.wait_for(2);
    pool.added(BASE).unwrap();
    pool.added(BASE + BLOCK_LEN).unwrap();
    // The higher one is asked out of the guest, then goes back once the host says it is out.
    assert_eq!(
        host.wait_for(3)[2],
        Call::Remove(BASE + BLOCK_LEN, BLOCK_LEN)
    );
    assert!(start.elapsed() >= delay, "{:?}", start.elapsed());
    pool.removed(BASE + BLOCK_LEN).unwrap();
    wait_counts(&blocks, 1, 1);
    assert_eq!(pool.live(), [(BASE, BLOCK_LEN)]);
    assert!(pool.find(BASE + BLOCK_LEN, 1).is_none());
    assert_eq!(
        pool.piece_used(BASE + BLOCK_LEN),
        Err(Error::InvalidArgument)
    );
    assert_eq!(host.calls().len(), 3);
}

#[test]
fn the_lowest_empty_piece_stays_as_headroom() {
    let delay = Duration::from_millis(50);
    let host = Arc::new(Host::default());
    let (pool, blocks) = pool(range(64), &host, delay);
    pool.grow(BLOCK_LEN);
    host.wait_for(1);
    pool.added(BASE).unwrap();
    std::thread::sleep(delay * 6);
    assert_eq!(blocks.counts(), (1, 0));
    assert_eq!(pool.live(), [(BASE, BLOCK_LEN)]);
}

#[test]
fn a_piece_used_again_within_the_delay_is_kept() {
    let delay = Duration::from_millis(50);
    let (host, hand) = (Arc::new(Host::default()), Hand::default());
    let (pool, blocks) = pool_at(range(64), &host, delay, hand.clock());
    pool.grow(BLOCK_LEN);
    pool.grow(BLOCK_LEN);
    host.wait_for(2);
    pool.added(BASE).unwrap();
    pool.added(BASE + BLOCK_LEN).unwrap();
    pool.piece_used(BASE).unwrap();
    pool.piece_used(BASE + BLOCK_LEN).unwrap();
    // Using the last empty piece asked for a block more: the headroom.
    assert_eq!(
        host.wait_for(3)[2],
        Call::Add(BASE + 2 * BLOCK_LEN, BLOCK_LEN)
    );
    pool.added(BASE + 2 * BLOCK_LEN).unwrap();
    // Emptied, then used again within the delay: kept. The clock moves only as the test moves
    // it; the real wait after gives a pool that kept the wrong piece the time to give it back.
    pool.piece_empty(BASE + BLOCK_LEN).unwrap();
    hand.on(delay / 2);
    pool.piece_used(BASE + BLOCK_LEN).unwrap();
    hand.on(delay * 2);
    std::thread::sleep(delay * 2);
    assert_eq!(blocks.counts(), (3, 0));
    // Emptied for good: the lowest empty piece, now this one, stays as the headroom, and the
    // one above it, empty past the delay, goes.
    pool.piece_empty(BASE + BLOCK_LEN).unwrap();
    assert_eq!(
        host.wait_for(4)[3],
        Call::Remove(BASE + 2 * BLOCK_LEN, BLOCK_LEN)
    );
    pool.removed(BASE + 2 * BLOCK_LEN).unwrap();
    wait_counts(&blocks, 2, 1);
    assert_eq!(
        pool.live(),
        [(BASE, BLOCK_LEN), (BASE + BLOCK_LEN, BLOCK_LEN)]
    );
    assert_eq!(host.calls().len(), 4);
}

#[test]
fn a_block_taken_again_is_mapped_anew() {
    let host = Arc::new(Host::default());
    let (pool, blocks) = two_pieces(&host, Duration::from_millis(50));
    assert_eq!(
        host.wait_for(3)[2],
        Call::Remove(BASE + BLOCK_LEN, BLOCK_LEN)
    );
    pool.removed(BASE + BLOCK_LEN).unwrap();
    wait_counts(&blocks, 1, 1);
    // The headroom piece is used: the pool asks for a block more, the source hands the ready one
    // out again, and the pool has the host map it at the lowest free address, where it was.
    pool.piece_used(BASE).unwrap();
    assert_eq!(host.wait_for(4)[3], Call::Add(BASE + BLOCK_LEN, BLOCK_LEN));
    wait_counts(&blocks, 2, 0);
    assert_eq!(pool.live(), [(BASE, BLOCK_LEN)]);
    pool.added(BASE + BLOCK_LEN).unwrap();
    assert_eq!(
        pool.live(),
        [(BASE, BLOCK_LEN), (BASE + BLOCK_LEN, BLOCK_LEN)]
    );
}

#[test]
fn ready_blocks_past_two_are_freed_with_no_word_to_the_host() {
    let delay = Duration::from_millis(30);
    let host = Arc::new(Host::default());
    let (pool, blocks) = pool(range(64), &host, delay);
    for _ in 0..4 {
        pool.grow(BLOCK_LEN);
    }
    host.wait_for(4);
    for i in 0..4 {
        pool.added(BASE + i * BLOCK_LEN).unwrap();
    }
    // Three go back above the headroom, lowest first, each asked out of the guest before it goes.
    for i in 1..4 {
        let at = BASE + i * BLOCK_LEN;
        let n = usize::try_from(4 + i).unwrap();
        assert_eq!(host.wait_for(n)[n - 1], Call::Remove(at, BLOCK_LEN));
        pool.removed(at).unwrap();
    }
    // The source keeps two ready and frees the oldest, asking no one.
    wait_counts(&blocks, 1, 2);
    std::thread::sleep(delay * 4);
    assert_eq!(host.calls().len(), 7);
    assert!(pool.source().is_some());
}

#[test]
fn a_piece_given_back_and_taken_again_reads_zero() {
    let delay = Duration::from_millis(50);
    let (host, hand) = (Arc::new(Host::default()), Hand::default());
    let (pool, blocks) = pool_at(range(64), &host, delay, hand.clock());
    pool.grow(BLOCK_LEN);
    pool.grow(BLOCK_LEN);
    host.wait_for(2);
    pool.added(BASE).unwrap();
    pool.added(BASE + BLOCK_LEN).unwrap();
    // The guest writes into the upper piece, then empties it; it goes back past the delay, which
    // the clock, standing still until the test moves it, cannot pass before the piece is used.
    pool.piece_used(BASE + BLOCK_LEN).unwrap();
    {
        let hold = pool.find(BASE + BLOCK_LEN + 4096, 8).unwrap();
        // SAFETY: inside the block, which the hold keeps.
        unsafe { hold.host.as_ptr().cast::<u64>().write(0xdead_beef) };
    }
    pool.piece_empty(BASE + BLOCK_LEN).unwrap();
    hand.on(delay);
    assert_eq!(
        host.wait_for(3)[2],
        Call::Remove(BASE + BLOCK_LEN, BLOCK_LEN)
    );
    pool.removed(BASE + BLOCK_LEN).unwrap();
    wait_counts(&blocks, 1, 1);
    // The headroom piece is used: the ready block goes out again, mapped anew at its address,
    // zeroed.
    pool.piece_used(BASE).unwrap();
    assert_eq!(host.wait_for(4)[3], Call::Add(BASE + BLOCK_LEN, BLOCK_LEN));
    pool.added(BASE + BLOCK_LEN).unwrap();
    wait_counts(&blocks, 2, 0);
    let hold = pool.find(BASE + BLOCK_LEN + 4096, 8).unwrap();
    // SAFETY: as above.
    assert_eq!(unsafe { hold.host.as_ptr().cast::<u64>().read() }, 0);
    assert_eq!(host.calls().len(), 4);
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
    pool.shared.changed.notify_all();
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
    // Its piece used, the pool asks for another as headroom.
    assert_eq!(host.wait_for(2)[1], Call::Add(BASE + BLOCK_LEN, BLOCK_LEN));
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

/// The zeroer zeroes a new piece, the headroom, ahead of every take, from its place on: a take
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

/// An emptied piece is zeroed once it is the headroom, the lowest empty piece, and kept; one
/// emptied above the headroom, which goes back whole past its delay, is not: its bytes are left
/// as the guest wrote them until it is out of the guest
#[test]
fn an_emptied_piece_is_zeroed_as_the_headroom_and_not_where_it_goes_back_whole() {
    let delay = Duration::from_millis(50);
    let (host, hand) = (Arc::new(Host::default()), Hand::default());
    let (pool, _blocks) = pool_at(range(64), &host, delay, hand.clock());
    let ledger = pool.ledger();
    assert!(ledger.keep_headroom());
    host.wait_for(1);
    pool.added(BASE).unwrap();
    wait_zeroed(&pool, BASE, BLOCK_LEN);
    // The first piece taken whole, the next is the headroom, zeroed; taken whole too, the third.
    for (context, at, calls) in [(1, 0, 2), (2, BLOCK_LEN, 3)] {
        assert_eq!(
            answer(&take(&ledger, context, BLOCK_LEN)),
            Some((at, BLOCK_LEN))
        );
        let next = BASE + at + BLOCK_LEN;
        assert_eq!(host.wait_for(calls)[calls - 1], Call::Add(next, BLOCK_LEN));
        pool.added(next).unwrap();
        wait_zeroed(&pool, next, BLOCK_LEN);
    }
    fill(&pool, 0, BLOCK_LEN, 0xa5);
    fill(&pool, BLOCK_LEN, BLOCK_LEN, 0x5a);
    // The first emptied is the lowest empty piece, the headroom: zeroed.
    let Give::Given(first) = ledger.give(1, 0, BLOCK_LEN, |_, _| false) else {
        panic!("not given");
    };
    ledger.free(&[first]);
    wait_zeroed(&pool, BASE, BLOCK_LEN);
    assert!(all(&pool, 0, BLOCK_LEN, 0));
    // The second emptied lies above it: it goes back whole, and nothing zeroes it.
    let before = zeroed_by(&pool);
    let Give::Given(second) = ledger.give(2, BLOCK_LEN, BLOCK_LEN, |_, _| false) else {
        panic!("not given");
    };
    ledger.free(&[second]);
    std::thread::sleep(Duration::from_millis(50));
    assert!(all(&pool, BLOCK_LEN, BLOCK_LEN, 0x5a));
    assert_eq!(zeroed(&pool, BASE + BLOCK_LEN), 0);
    hand.on(delay);
    assert_eq!(
        host.wait_for(4)[3],
        Call::Remove(BASE + BLOCK_LEN, BLOCK_LEN)
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
    pool.shared.changed.notify_all();
    wait_zeroed(&pool, BASE, takes::ZERO_CHUNK);
    assert!(all(&pool, 0, top, 0x5a));
    assert!(all(&pool, top, takes::ZERO_CHUNK, 0));
    // Its own bytes zeroed once: the take's worker zeroed none of the zeroer's chunk.
    assert_eq!(zeroed_by(&pool)[0], top);
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
    pool.shared.changed.notify_all();
    assert_eq!(answer(&rx), Some((0, BLOCK_LEN)));
    assert!(all(&pool, 0, BLOCK_LEN, 0));
    // The worker zeroed all but the zeroer's chunk.
    assert_eq!(zeroed_by(&pool)[0], top);
    assert!(counted(TAKE_CLAIM_WAITS_CODE) > waits);
    // No piece was asked for the take: the next is the headroom, asked once the take used this.
    assert_eq!(
        host.wait_for(2),
        [
            Call::Add(BASE, BLOCK_LEN),
            Call::Add(BASE + BLOCK_LEN, BLOCK_LEN)
        ]
    );
}

/// An answered take's wait is counted by its cause ([`TAKE_WAIT_CODES`]), the take counted once
/// its waits are: a take that fits no piece waits for one of its own, which the host maps 50 ms
/// after its ask, and that wait is its growth part's and at most the longest's; its parts lie
/// within its whole wait
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
    assert!(grew(3) >= held, "waited for a piece: {} ns", grew(3));
    assert!(after[5] >= held, "the longest wait for a piece");
    assert!(
        grew(2) + grew(3) + grew(4) <= grew(1),
        "the parts within the whole"
    );
}

/// The guest's driver's take of the pool asks for a piece ahead of the first take, which that
/// take then finds, answered with no piece of its own asked or mapped; taken again with the piece
/// still empty, as after a reset, it asks none
#[test]
fn a_piece_asked_ahead_of_the_first_take_serves_it_with_no_wait() {
    let host = Arc::new(Host::default());
    let (pool, _blocks) = pool(range(64), &host, LONG);
    let ledger = pool.ledger();
    assert!(ledger.keep_headroom());
    assert_eq!(host.wait_for(1), [Call::Add(BASE, BLOCK_LEN)]);
    assert!(
        !ledger.keep_headroom(),
        "asked again while the host maps it"
    );
    pool.added(BASE).unwrap();
    assert!(!ledger.keep_headroom(), "asked again");
    assert_eq!(answer(&take(&ledger, 1, TAKE_ALIGN)), Some((0, TAKE_ALIGN)));
    // Its piece used, the pool asks for the next as headroom, which the take did not wait for.
    assert_eq!(host.wait_for(2)[1], Call::Add(BASE + BLOCK_LEN, BLOCK_LEN));
}

/// A take carved from the headroom piece has the pool's thread make the next at once, not once
/// the take's worker went idle: on a clock that stands still, as a test platform's does, the
/// worker never goes
#[test]
fn a_take_that_uses_the_headroom_piece_has_the_next_made_at_once() {
    let host = Arc::new(Host::default());
    let still = (now())();
    let clock: PoolNow = Box::new(move || still);
    let source: Arc<dyn BlockSource> = Arc::new(HostBlocks::new());
    let pool = Pool::with_delay(range(64), host.pool_host(), LONG, clock, source).unwrap();
    let ledger = pool.ledger();
    assert!(ledger.keep_headroom());
    assert_eq!(host.wait_for(1), [Call::Add(BASE, BLOCK_LEN)]);
    pool.added(BASE).unwrap();
    assert_eq!(answer(&take(&ledger, 1, TAKE_ALIGN)), Some((0, TAKE_ALIGN)));
    assert_eq!(host.wait_for(2)[1], Call::Add(BASE + BLOCK_LEN, BLOCK_LEN));
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
    assert!(ledger.over_held(1, 0, len, |_| ()).is_some());
    let Give::Given(carve) = ledger.give(1, 0, len, free) else {
        panic!("not given");
    };
    // Going: no heap is made over it, and no second give finds it, until its free.
    assert!(ledger.over_held(1, 0, len, |_| ()).is_none());
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
        ledger.over_held(1, TAKE_ALIGN, TAKE_ALIGN, |p| p),
        Some(first)
    );
    assert_eq!(ledger.over_held(1, TAKE_ALIGN, len, |p| p), None);
    assert_eq!(ledger.over_held(2, 0, TAKE_ALIGN, |p| p), None);
    assert_eq!(ledger.over_held(1, len, TAKE_ALIGN, |p| p), None);
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
            .over_held(2, TAKE_ALIGN, TAKE_ALIGN, |_| ())
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

#[test]
fn a_piece_with_a_block_stays_and_goes_back_once_its_last_block_is_freed() {
    let delay = Duration::from_millis(50);
    let host = Arc::new(Host::default());
    let (pool, _blocks, ledger) = live_pool(&host, delay);
    // The first piece full, the pool asks for headroom, which the next take uses.
    assert_eq!(answer(&take(&ledger, 1, BLOCK_LEN)), Some((0, BLOCK_LEN)));
    assert_eq!(host.wait_for(2)[1], Call::Add(BASE + BLOCK_LEN, BLOCK_LEN));
    pool.added(BASE + BLOCK_LEN).unwrap();
    assert_eq!(
        answer(&take(&ledger, 1, TAKE_ALIGN)),
        Some((BLOCK_LEN, TAKE_ALIGN))
    );
    assert_eq!(
        host.wait_for(3)[2],
        Call::Add(BASE + 2 * BLOCK_LEN, BLOCK_LEN)
    );
    pool.added(BASE + 2 * BLOCK_LEN).unwrap();
    // Both used pieces stay past the delay, and the empty one is the headroom.
    std::thread::sleep(delay * 6);
    assert_eq!(host.calls().len(), 3);
    // The first emptied, it is the lowest empty piece: the third goes back.
    let Give::Given(carve) = ledger.give(1, 0, BLOCK_LEN, |_, _| false) else {
        panic!("not given");
    };
    ledger.free(&[carve]);
    assert_eq!(
        host.wait_for(4)[3],
        Call::Remove(BASE + 2 * BLOCK_LEN, BLOCK_LEN)
    );
    assert_eq!(host.calls().len(), 4);
}

#[test]
fn the_release_leaves_the_blocks_with_a_live_ledger_until_its_drop() {
    let host = Arc::new(Host::default());
    let (pool, blocks, ledger) = live_pool(&host, LONG);
    answer(&take(&ledger, 1, TAKE_ALIGN)).unwrap();
    host.wait_for(2);
    drop(pool);
    assert_eq!(blocks.counts(), (2, 0));
    // A take after the release is never answered.
    let rx = take(&ledger, 2, TAKE_ALIGN);
    assert!(rx.recv_timeout(Duration::from_secs(1)).is_err());
    drop(ledger);
    assert_eq!(blocks.counts(), (0, 2));
}

/// A source whose blocks have GPU addresses, as metal2vulkan-rt's do: [`HostBlocks`]' blocks, the
/// `n`th handed out at `ADDRESSED + n * BLOCK_LEN`.
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
    // The rest of the first piece, then the headroom piece, the second block handed out.
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
