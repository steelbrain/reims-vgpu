// Test scaffolding: real time bounds the waits and runs the delays, and a helper thread stands in
// for a thread of the device's.
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

/// The host blocks the pool takes from, kept for the tests' counts.
fn pool(range: PoolConfig, host: &Arc<Host>, delay: Duration) -> (Pool, Arc<HostBlocks>) {
    let blocks = Arc::new(HostBlocks::new());
    let source: Arc<dyn BlockSource> = Arc::clone(&blocks) as _;
    let pool = Pool::with_delay(range, host.pool_host(), delay, now(), source).unwrap();
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
    let (pool, blocks) = pool(range(64), host, delay);
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
    let host = Arc::new(Host::default());
    let (pool, blocks) = two_pieces(&host, delay);
    let hold = pool.find(BASE + BLOCK_LEN, 1).unwrap();
    std::thread::sleep(delay * 6);
    // Out of the lookups past the delay, but neither asked out of the guest nor given back while
    // the hold lives.
    assert!(pool.find(BASE + BLOCK_LEN, 1).is_none());
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
    let delay = Duration::from_millis(400);
    let host = Arc::new(Host::default());
    let (pool, blocks) = pool(range(64), &host, delay);
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
    // Emptied, then used again within the delay: kept.
    pool.piece_empty(BASE + BLOCK_LEN).unwrap();
    std::thread::sleep(delay / 4);
    pool.piece_used(BASE + BLOCK_LEN).unwrap();
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
    let host = Arc::new(Host::default());
    let (pool, blocks) = pool(range(64), &host, delay);
    pool.grow(BLOCK_LEN);
    pool.grow(BLOCK_LEN);
    host.wait_for(2);
    pool.added(BASE).unwrap();
    pool.added(BASE + BLOCK_LEN).unwrap();
    // The guest writes into the upper piece, then empties it; it goes back past the delay.
    pool.piece_used(BASE + BLOCK_LEN).unwrap();
    {
        let hold = pool.find(BASE + BLOCK_LEN + 4096, 8).unwrap();
        // SAFETY: inside the block, which the hold keeps.
        unsafe { hold.host.as_ptr().cast::<u64>().write(0xdead_beef) };
    }
    pool.piece_empty(BASE + BLOCK_LEN).unwrap();
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
    let (pool, _blocks, ledger) = live_pool(&host, LONG);
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
    assert_eq!(third.gpu_address, addresses::HOST_BASE + 0x2_8000 + 0x1_8000);
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
    assert_eq!(answer(&take(&ledger, 1, rest)), Some((4 * TAKE_ALIGN, rest)));
    host.wait_for(2);
    pool.added(BASE + BLOCK_LEN).unwrap();
    let next = taken(&take(&ledger, 2, TAKE_ALIGN)).unwrap();
    assert_eq!(next.offset, BLOCK_LEN);
    assert_eq!(next.gpu_address, ADDRESSED + BLOCK_LEN);
    assert_eq!(ledger.picked(), 0);
}
