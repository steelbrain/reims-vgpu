use std::sync::Arc;
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

/// The host blocks the pool takes from, kept for the tests' counts.
fn pool(range: PoolConfig, host: &Arc<Host>, delay: Duration) -> (Pool, Arc<HostBlocks>) {
    let kept: Arc<Mutex<Option<Arc<HostBlocks>>>> = Arc::default();
    let k = Arc::clone(&kept);
    let pool = Pool::with_delay(range, host.pool_host(), delay, move |m| {
        let blocks = Arc::new(HostBlocks::new(m));
        *k.lock() = Some(Arc::clone(&blocks));
        let source: Arc<dyn BlockSource> = blocks;
        source
    })
    .unwrap();
    let blocks = kept.lock().take().unwrap();
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
        Pool::new(PoolConfig::default(), host.pool_host(), |m| Arc::new(
            HostBlocks::new(m)
        ))
        .err(),
        Some(Error::InvalidArgument)
    );
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

/// Gives every block back through the pool's hook, as the source does when it frees surplus
/// blocks, and returns the thread doing it.
fn unmap_in_background(pool: &Pool, block: BlockId) -> std::thread::JoinHandle<()> {
    let mappings = Mappings(Arc::downgrade(&pool.shared));
    std::thread::spawn(move || mappings.unmap(block))
}

#[test]
fn an_unmap_returns_only_once_the_host_removed_the_piece() {
    let host = Arc::new(Host::default());
    let (pool, _blocks) = pool(range(64), &host, LONG);
    pool.grow(BLOCK_LEN);
    host.wait_for(1);
    pool.added(BASE).unwrap();
    let id = pool.find(BASE, 1).unwrap().block;
    let unmap = unmap_in_background(&pool, id);
    assert_eq!(host.wait_for(2)[1], Call::Remove(BASE, BLOCK_LEN));
    // Out of the lookups as soon as the unmap began.
    assert!(pool.find(BASE, 1).is_none());
    std::thread::sleep(Duration::from_millis(100));
    assert!(!unmap.is_finished());
    pool.removed(BASE).unwrap();
    unmap.join().unwrap();
    // The address is free again: the next block goes there.
    pool.grow(BLOCK_LEN);
    assert_eq!(host.wait_for(3)[2], Call::Add(BASE, BLOCK_LEN));
}

#[test]
fn an_unmap_asks_the_host_only_once_no_hold_is_left() {
    let host = Arc::new(Host::default());
    let (pool, _blocks) = pool(range(64), &host, LONG);
    pool.grow(BLOCK_LEN);
    host.wait_for(1);
    pool.added(BASE).unwrap();
    let hold = pool.find(BASE, 1).unwrap();
    let unmap = unmap_in_background(&pool, hold.block);
    std::thread::sleep(Duration::from_millis(100));
    assert_eq!(host.calls().len(), 1, "no remove while a hold lives");
    drop(hold);
    assert_eq!(host.wait_for(2)[1], Call::Remove(BASE, BLOCK_LEN));
    pool.removed(BASE).unwrap();
    unmap.join().unwrap();
}

#[test]
fn an_unmap_before_the_hosts_add_answers_still_waits_for_the_remove() {
    let host = Arc::new(Host::default());
    let (pool, blocks) = pool(range(64), &host, LONG);
    pool.grow(BLOCK_LEN);
    host.wait_for(1);
    let id = BlockId(1);
    assert_eq!(blocks.counts(), (1, 0));
    let unmap = unmap_in_background(&pool, id);
    assert_eq!(host.wait_for(2)[1], Call::Remove(BASE, BLOCK_LEN));
    // The host's add, which it queued first, then its remove.
    pool.added(BASE).unwrap();
    assert!(pool.find(BASE, 1).is_none());
    pool.removed(BASE).unwrap();
    unmap.join().unwrap();
}

#[test]
fn after_the_release_the_sources_unmaps_ask_the_host_nothing() {
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
    drop(blocks);
    assert_eq!(host.calls().len(), 2);
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
    wait_counts(&blocks, 1, 1);
    assert!(start.elapsed() >= delay, "{:?}", start.elapsed());
    // The higher one went; it stays mapped until the source frees it, and nothing finds it.
    assert_eq!(pool.live(), [(BASE, BLOCK_LEN)]);
    assert!(pool.find(BASE + BLOCK_LEN, 1).is_none());
    assert_eq!(
        pool.piece_used(BASE + BLOCK_LEN),
        Err(Error::InvalidArgument)
    );
    assert_eq!(host.calls().len(), 2);
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
    assert_eq!(host.wait_for(3)[2], Call::Add(BASE + 2 * BLOCK_LEN, BLOCK_LEN));
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
    wait_counts(&blocks, 2, 1);
    assert_eq!(pool.live(), [(BASE, BLOCK_LEN), (BASE + BLOCK_LEN, BLOCK_LEN)]);
    assert_eq!(host.calls().len(), 3);
}

#[test]
fn a_block_taken_again_keeps_its_address_and_asks_the_host_nothing() {
    let delay = Duration::from_millis(50);
    let host = Arc::new(Host::default());
    let (pool, blocks) = pool(range(64), &host, delay);
    pool.grow(BLOCK_LEN);
    pool.grow(BLOCK_LEN);
    host.wait_for(2);
    pool.added(BASE).unwrap();
    pool.added(BASE + BLOCK_LEN).unwrap();
    wait_counts(&blocks, 1, 1);
    // The headroom piece is used: the pool asks for a block more, and the source hands the ready
    // one out again.
    pool.piece_used(BASE).unwrap();
    wait_counts(&blocks, 2, 0);
    assert_eq!(
        pool.live(),
        [(BASE, BLOCK_LEN), (BASE + BLOCK_LEN, BLOCK_LEN)]
    );
    assert_eq!(host.calls().len(), 2);
}

#[test]
fn ready_blocks_past_two_are_unmapped_before_they_are_freed() {
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
    // Three go back above the headroom; the source keeps two ready and frees the oldest, whose
    // unmap waits for the host.
    let calls = host.wait_for(5);
    let Call::Remove(at, len) = calls[4] else {
        panic!("{calls:?}");
    };
    assert_eq!(len, BLOCK_LEN);
    assert_eq!(blocks.counts(), (1, 2));
    pool.removed(at).unwrap();
    wait_counts(&blocks, 1, 2);
    std::thread::sleep(delay * 4);
    assert_eq!(host.calls().len(), 5);
    assert!(pool.source().is_some());
}
