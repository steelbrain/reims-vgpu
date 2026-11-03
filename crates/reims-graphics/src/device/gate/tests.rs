//! The gate's wakes: each waiting thread woken only once what it waits for holds, and none left
//! asleep.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parking_lot::Mutex;

use super::*;

/// What a test's waiters wait for: the control thread's count, and each context's lane's.
#[derive(Default)]
struct Progress {
    count: AtomicU64,
    lanes: Mutex<HashMap<u32, u64>>,
}

impl Progress {
    /// What a waiter for count `count` and context `context`'s lane through `own` finds.
    fn look(&self, count: u64, context: u32, own: u64) -> Look {
        if self.count.load(Ordering::Acquire) < count {
            Look::Count(count)
        } else if self.lanes.lock().get(&context).copied().unwrap_or(0) < own {
            Look::Lane(context, own)
        } else {
            Look::Done
        }
    }

    /// Whether it holds.
    fn holds(&self, count: u64, context: u32, own: u64) -> bool {
        self.look(count, context, own) == Look::Done
    }
}

/// Starts a thread waiting at `gate` for count `count` and context `context`'s lane through
/// `own`; it returns how many times it looked, having checked that what it waits for holds.
fn waiter(
    gate: &Arc<Gate>,
    progress: &Arc<Progress>,
    (count, context, own): (u64, u32, u64),
) -> std::thread::JoinHandle<u32> {
    let (gate, progress) = (Arc::clone(gate), Arc::clone(progress));
    std::thread::Builder::new()
        .name(format!("waiter-{context}-{count}-{own}"))
        .spawn(move || {
            let mut looks = 0;
            gate.wait(|| {
                looks += 1;
                progress.look(count, context, own)
            });
            assert!(
                progress.holds(count, context, own),
                "a waiter for count {count} and context {context}'s lane through {own} went early"
            );
            looks
        })
        .unwrap()
}

/// Waits about 2 s for `n` threads to wait at `gate`.
fn waiting(gate: &Gate, n: usize) {
    for _ in 0..2000 {
        if gate.counts().1 == n {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("{n} threads never waited: {:?}", gate.counts());
}

/// A thread waiting for the control thread's count is woken by no move short of it, and by the
/// move to it; one waiting for its context's lane by no other context's lane and no move of its
/// own short of what it waits for.
#[test]
fn a_waiter_is_woken_only_once_what_it_waits_for_holds() {
    let gate = Arc::new(Gate::default());
    let progress = Arc::new(Progress::default());
    let handle = waiter(&gate, &progress, (10, 3, 7));
    waiting(&gate, 1);
    for done in [2, 5, 9] {
        progress.count.store(done, Ordering::Release);
        gate.counted(done);
    }
    assert_eq!(gate.counts(), (0, 1), "woken short of its count");
    progress.count.store(10, Ordering::Release);
    gate.counted(10);
    waiting(&gate, 1);
    assert_eq!(
        gate.counts(),
        (1, 1),
        "woken once, now waiting for its lane"
    );
    progress.lanes.lock().insert(4, u64::MAX);
    gate.carried_out(4, u64::MAX);
    progress.lanes.lock().insert(3, 6);
    gate.carried_out(3, 6);
    assert_eq!(
        gate.counts(),
        (1, 1),
        "woken by another lane or short of its own"
    );
    progress.lanes.lock().insert(3, 7);
    gate.carried_out(3, 7);
    assert_eq!(handle.join().unwrap(), 3);
    assert_eq!(gate.counts(), (2, 0));
}

/// A wake of every thread (a reset, the end) wakes each, which looks again.
#[test]
fn every_wakes_each_waiter_to_look_again() {
    let gate = Arc::new(Gate::default());
    let progress = Arc::new(Progress::default());
    let handles: Vec<_> = (0..4)
        .map(|n| waiter(&gate, &progress, (u64::from(n) + 1, n, 1)))
        .collect();
    waiting(&gate, 4);
    gate.every();
    waiting(&gate, 4);
    assert_eq!(gate.counts(), (4, 4));
    progress.count.store(u64::MAX, Ordering::Release);
    progress.lanes.lock().extend((0..4).map(|c| (c, u64::MAX)));
    gate.every();
    for handle in handles {
        assert_eq!(handle.join().unwrap(), 3);
    }
}

/// SplitMix64: the fuzz's random numbers, the same for a seed on every host.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Many threads of several contexts wait for random counts and lanes while the control thread's
/// count and the lanes move in random steps, each move made before its wake, as the device makes
/// them. Each thread goes only once what it waits for holds, is woken at most once for its count
/// and once for its lane (so looks at most three times), and none is left asleep once
/// everything has moved past what they wait for. `RGFX_GATE_FUZZ_SEED` and
/// `RGFX_GATE_FUZZ_CASES` pick other cases.
#[test]
fn the_gate_wakes_each_waiter_once_what_it_waits_for_holds() {
    let seed = std::env::var("RGFX_GATE_FUZZ_SEED")
        .ok()
        .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        .unwrap_or(0x9a7e_5eed);
    let cases: u64 = std::env::var("RGFX_GATE_FUZZ_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    for case in 0..cases {
        let mut rng = Rng(seed ^ case.wrapping_mul(0x5851_f42d_4c95_7f2d));
        let contexts = 1 + rng.below(8) as u32;
        let waiters = 1 + rng.below(48) as usize;
        let top = 1 + rng.below(400);
        let gate = Arc::new(Gate::default());
        let progress = Arc::new(Progress::default());
        let handles: Vec<_> = (0..waiters)
            .map(|_| {
                let count = rng.below(top + 1);
                let context = rng.below(u64::from(contexts)) as u32;
                let own = rng.below(top + 1);
                waiter(&gate, &progress, (count, context, own))
            })
            .collect();
        // Some look before anything moves, some after.
        if rng.below(2) == 0 {
            std::thread::sleep(Duration::from_millis(2));
        }
        let mut count = 0;
        let mut lanes: HashMap<u32, u64> = HashMap::new();
        loop {
            let lanes_done = (0..contexts).all(|c| lanes.get(&c).copied().unwrap_or(0) >= top);
            if count >= top && lanes_done {
                break;
            }
            if rng.below(2) == 0 && count < top {
                count = (count + 1 + rng.below(20)).min(top);
                progress.count.store(count, Ordering::Release);
                gate.counted(count);
            } else {
                let context = rng.below(u64::from(contexts)) as u32;
                let done = lanes.entry(context).or_insert(0);
                *done = (*done + 1 + rng.below(20)).min(top);
                let done = *done;
                progress.lanes.lock().insert(context, done);
                gate.carried_out(context, done);
            }
            if rng.below(8) == 0 {
                std::thread::yield_now();
            }
        }
        for (n, handle) in handles.into_iter().enumerate() {
            let looks = handle.join().unwrap();
            assert!(
                looks <= 3,
                "case {case} (seed {seed:#x}): waiter {n} looked {looks} times"
            );
        }
        let (woken, left) = gate.counts();
        assert_eq!(left, 0, "case {case} (seed {seed:#x}): waiters left asleep");
        assert!(
            woken <= 2 * waiters as u64,
            "case {case} (seed {seed:#x}): {woken} wakes for {waiters} waiters"
        );
    }
}
