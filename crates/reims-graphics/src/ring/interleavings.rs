//! Every interleaving of the three parties to a command queue's completion ring, walked to its
//! end, for the host's post with [`rearm_or_raise`] and for the post before it, which raised
//! wherever its clear read 1:
//! - the host, which publishes each record, then clears `armed`, reads `consumer` where the clear
//!   read 1, sets `armed` again where the guest took every record, and raises the vector later,
//!   outside the ring's lock, where it decided to; one post at a time, as the ring's lock has it;
//! - the guest's kext, whose work loop takes every vector raised so far in one pass, and wakes
//!   the client where the ring's producer moved since its last look and the ring holds records
//!   the client has not taken (`RgfxAccelerator::queueInterrupt`); a vector raised for something
//!   else (a vblank, ring 0, another queue's ring) may start a pass too;
//! - the process's completion thread (`RgfxCompletionThread`), which takes the ring's records,
//!   arms it, reads `producer` again, and sleeps in the kext until a wake moves the client's
//!   count, which it reads as it returns.
//!
//! Each step is one access to shared memory, in sequential consistency: each side fences between
//! its write and its read, and the host's clear is a locked swap. Where the host has posted every
//! record, the kext has looked after every vector and the thread sleeps, the thread must have
//! taken every record and sleep armed, or the next record raises nothing and waits for a vector
//! for something else, which an idle guest may never have.

use std::collections::HashSet;

use super::{Wake, rearm_or_raise};

/// The host's post.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Post {
    /// Raises wherever its clear read 1.
    Clearing,
    /// [`rearm_or_raise`].
    Rearming,
}

/// Where the host is in its post.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
enum HostAt {
    /// Between posts.
    #[default]
    Idle,
    /// Published; the clear of `armed` next.
    Clear,
    /// Cleared a 1; the read of `consumer` next.
    ReadConsumer,
    /// The store of `armed` 1 next.
    Rearm,
}

/// Where the kext's pass is.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
enum KextAt {
    #[default]
    Idle,
    ReadProducer,
    ReadConsumer,
    Decide,
}

/// Where the completion thread is.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
enum ThreadAt {
    /// The pump's read of `producer` next.
    #[default]
    PumpRead,
    /// The pump's store of `consumer` next.
    PumpTake,
    /// The store of `armed` 1 next.
    Arm,
    /// The read of `producer` after the arm next.
    ArmLook,
    /// The wait in the kext next.
    Wait,
    /// Asleep in the kext.
    Asleep,
    /// Woken: the read of the client's count as the wait returns next.
    Woken,
}

/// One state of the three, counts in records.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default)]
struct State {
    // The ring.
    producer: u8,
    consumer: u8,
    armed: u8,
    // The host: records published, where its post is, what its clear read, vectors decided and
    // not yet raised, and raised and not yet taken by a pass.
    posted: u8,
    host: HostAt,
    cleared: u8,
    owed: u8,
    raised: u8,
    // The kext: where its pass is, what it read, the producer its last look saw, and the vectors
    // for something else still to come.
    kext: KextAt,
    k_producer: u8,
    k_consumer: u8,
    last: u8,
    foreign: u8,
    // The thread: where it is, the producer its pump read, what it took, the client's count it
    // last saw; and the client's count.
    thread: ThreadAt,
    pumped: u8,
    taken: u8,
    seen: u8,
    count: u8,
}

/// What the walk found.
#[derive(Debug, Default)]
struct Outcome {
    /// States reached.
    states: usize,
    /// Quiescent ones: every record posted, every vector looked after, the thread asleep.
    quiescent: usize,
    /// Of them, those with a record the thread never took, and states stuck short of quiescence.
    lost: usize,
    /// Quiescent states with the thread asleep unarmed.
    unarmed: usize,
}

/// The states after `s`.
fn next(s: State, post: Post, records: u8) -> Vec<State> {
    let mut out = Vec::new();
    // The host, under the ring's lock: one post at a time.
    match s.host {
        HostAt::Idle if s.posted < records => {
            let mut n = s;
            n.posted += 1;
            n.producer = n.posted;
            n.host = HostAt::Clear;
            out.push(n);
        }
        HostAt::Idle => {}
        HostAt::Clear => {
            let mut n = s;
            n.cleared = n.armed;
            n.armed = 0;
            n.host = HostAt::Idle;
            if n.cleared == 1 {
                match post {
                    Post::Clearing => n.owed += 1,
                    Post::Rearming => n.host = HostAt::ReadConsumer,
                }
            }
            out.push(n);
        }
        HostAt::ReadConsumer => {
            let mut n = s;
            n.host = HostAt::Idle;
            match rearm_or_raise(
                u32::from(n.cleared),
                u64::from(n.consumer),
                u64::from(n.posted),
            ) {
                Wake::Rearm => n.host = HostAt::Rearm,
                Wake::Raise => n.owed += 1,
                Wake::Unarmed => {}
            }
            out.push(n);
        }
        HostAt::Rearm => {
            let mut n = s;
            n.armed = 1;
            n.host = HostAt::Idle;
            out.push(n);
        }
    }
    // The vector, raised outside the lock.
    if s.owed > 0 {
        let mut n = s;
        n.owed -= 1;
        n.raised += 1;
        out.push(n);
    }
    // The kext: a pass takes every vector raised before it.
    match s.kext {
        KextAt::Idle => {
            if s.raised > 0 {
                let mut n = s;
                n.raised = 0;
                n.kext = KextAt::ReadProducer;
                out.push(n);
            }
            if s.foreign > 0 {
                let mut n = s;
                n.foreign -= 1;
                n.raised = 0;
                n.kext = KextAt::ReadProducer;
                out.push(n);
            }
        }
        KextAt::ReadProducer => {
            let mut n = s;
            n.k_producer = n.producer;
            n.kext = KextAt::ReadConsumer;
            out.push(n);
        }
        KextAt::ReadConsumer => {
            let mut n = s;
            n.k_consumer = n.consumer;
            n.kext = KextAt::Decide;
            out.push(n);
        }
        KextAt::Decide => {
            let mut n = s;
            let moved = n.k_producer != n.last;
            n.last = n.k_producer;
            if moved && n.k_producer != n.k_consumer {
                // wakeCompletions: the count moves, and a sleeping thread wakes.
                n.count += 1;
                if n.thread == ThreadAt::Asleep {
                    n.thread = ThreadAt::Woken;
                }
            }
            n.kext = KextAt::Idle;
            out.push(n);
        }
    }
    // The thread.
    let mut n = s;
    match s.thread {
        ThreadAt::PumpRead => {
            n.pumped = n.producer;
            n.thread = ThreadAt::PumpTake;
        }
        ThreadAt::PumpTake => {
            n.taken = n.pumped;
            n.consumer = n.pumped;
            n.thread = ThreadAt::Arm;
        }
        ThreadAt::Arm => {
            n.armed = 1;
            n.thread = ThreadAt::ArmLook;
        }
        ThreadAt::ArmLook => {
            n.thread = if n.producer != n.taken {
                ThreadAt::PumpRead
            } else {
                ThreadAt::Wait
            };
        }
        ThreadAt::Wait => {
            // Under the client's lock, as the kext's wake moves the count.
            if n.count != n.seen {
                n.seen = n.count;
                n.thread = ThreadAt::PumpRead;
            } else {
                n.thread = ThreadAt::Asleep;
            }
        }
        ThreadAt::Woken => {
            n.seen = n.count;
            n.thread = ThreadAt::PumpRead;
        }
        ThreadAt::Asleep => {}
    }
    if s.thread != ThreadAt::Asleep {
        out.push(n);
    }
    out
}

/// Every state reachable with `records` posted and `foreign` vectors for something else.
fn walk(post: Post, records: u8, foreign: u8) -> Outcome {
    let mut out = Outcome::default();
    let mut seen = HashSet::new();
    let mut todo = vec![State {
        foreign,
        ..State::default()
    }];
    while let Some(s) = todo.pop() {
        if !seen.insert(s) {
            continue;
        }
        out.states += 1;
        let after = next(s, post, records);
        let quiescent = s.posted == records
            && s.host == HostAt::Idle
            && s.owed == 0
            && s.raised == 0
            && s.kext == KextAt::Idle
            && s.thread == ThreadAt::Asleep;
        if quiescent {
            out.quiescent += 1;
            if s.taken != records {
                out.lost += 1;
            }
            if s.armed != 1 {
                out.unarmed += 1;
            }
        } else if after.is_empty() {
            out.lost += 1;
        }
        todo.extend(after);
    }
    out
}

/// The post that only clears leaves the thread asleep with a record it never takes, with no
/// vector for anything else coming: the stall has no bound on an idle guest
#[test]
fn a_post_that_only_clears_loses_a_record() {
    let one = walk(Post::Clearing, 1, 0);
    assert!(one.lost == 0 && one.unarmed > 0, "{one:?}");
    for (records, foreign) in [(2, 0), (2, 2), (3, 2)] {
        let o = walk(Post::Clearing, records, foreign);
        assert!(
            o.lost > 0 && o.unarmed > 0,
            "{records} records, {foreign} foreign: {o:?}"
        );
    }
}

/// With [`rearm_or_raise`], every interleaving of up to 3 records and 2 vectors for something
/// else ends with every record taken and the thread asleep armed, under the kext's look as it is
#[test]
fn every_interleaving_ends_with_the_records_taken_and_the_ring_armed() {
    for records in 1..=3 {
        for foreign in 0..=2 {
            let o = walk(Post::Rearming, records, foreign);
            assert!(
                o.quiescent > 0,
                "{records} records, {foreign} foreign: {o:?}"
            );
            assert_eq!(
                (o.lost, o.unarmed),
                (0, 0),
                "{records} records, {foreign} foreign: {o:?}"
            );
        }
    }
}
