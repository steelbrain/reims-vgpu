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
