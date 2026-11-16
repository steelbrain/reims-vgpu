//! The submission rings' two wake handshakes, raced on two threads over a ring in memory: the
//! host's own code on one thread, the guest's steps as its kext and Metal bundle make them on the
//! other, counting the wakes lost.
//!
//! Each handshake is two store, fence, load sequences. The side that goes to sleep stores its
//! flag, fences, and looks once more; the side that publishes stores its count, fences, and reads
//! the flag. Either the look sees the publish or the read sees the flag, but only while each side
//! has a full fence between its store and its load: without one, x86 lets the load pass the store
//! (the store waits in the store buffer while the load reads the cache), and arm64 lets a
//! load-acquire pass an earlier store-release. Then both can miss the other, and the sleeper
//! waits for a wake the publisher never sends.
//! - **The doorbell.** The guest publishes as the kext's `rgfxQueuePublish` and
//!   `rgfxControlPublishWantsBell` and the bundle's `RgfxRingWriter::tryPost` do: `producer`
//!   stored (release), a full fence, `consumer_idle` read (acquire), and the doorbell rung where
//!   it reads [`RGFX_CONSUMER_ASLEEP`]. The host takes every message, then
//!   [`Consumer::sleep_or_pending`], and where that answers false sleeps until the doorbell, then
//!   [`Consumer::awake`], as the device's control thread does.
//! - **The room.** A guest post that finds no room asks for it as the kext's `askForRoom` and the
//!   bundle's `RgfxRingWriter::post` do: `producer_waiting` 1 stored (release), a full fence, and
//!   the ring's room looked at once more (`consumer`, acquire), waiting for a `SUBMIT_ROOM` record
//!   where there is none. The host takes each message, then [`Consumer::room_wanted`], and posts
//!   the record where it answers true.
//!
//! A wake is lost where the sleeper waits with work no wake will come for: the host asleep, no
//! doorbell rung, while the guest has decided about a message the host has not taken; the guest
//! waiting, no record posted, while the host has decided about a take past what the guest's look
//! saw. Each side stores how far it has decided after its handshake, and the sleeper reads that,
//! then its wake, so it counts a sleep as lost only once nothing can end it; with the fences none
//! is. It then ends the sleep, as the protocol's backstop would, and the race goes on.
//!
//! The guest publishes one message at a time: in the doorbell's race once the host has taken the
//! last, in the room's into a ring the last fills. Before its handshake each side spins for a
//! random time, so the two handshakes overlap across the whole window, and stores to a line of
//! memory not in its cache, so its handshake's store waits behind that one in the store buffer
//! while its load is answered, as a store behind a command buffer's bodies does: the window a
//! missing fence opens is then the memory's latency, not a cache's. The guest also reads
//! `consumer_idle` before it writes the message, so its read after the publish is a hit. Each
//! race runs for [`RACE_NS`] or [`MOST`] messages, whichever ends first.
//!
//! The controls (`#[ignore]`, run with `--run-ignored only`) race the same with one side's fence
//! dropped, its store release and its load acquire: the guest's, as the guest driver of issue
//! #158 published, and the host's. Wakes lost per million messages, the guest's fence dropped /
//! the host's, one run each on 2026-11-15 (jobs m3-max-34985, -34986 and -34987):
//!
//! | CPU | the doorbell | the room |
//! |---|---|---|
//! | Apple M3 Max (arm64, macOS 27) | 118k / 154k | 90k / 121k |
//! | Ryzen 7 7840HS (x86-64, Zen 4) | 31 / 306 | 16k / 17k |
//! | Core i9-13900H (x86-64, Raptor Lake) | 100 / 85 | 156 / 39 |
//!
//! With both fences, none in any run: about 700k to 1M messages a race.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering, fence};

use zerocopy::IntoBytes;

use super::{CONSUMER, CONSUMER_IDLE, Consumer, PRODUCER, PRODUCER_WAITING, RingAt};
use crate::region::Region;
use crate::wire::{RGFX_CONSUMER_ASLEEP, RgfxMsgHeader};

/// How long a race runs at most, in nanoseconds.
const RACE_NS: u64 = 1_000_000_000;

/// How many messages a race publishes at most.
const MOST: u64 = 1_000_000;

/// The most each side spins before its handshake, in turns of [`Lead::spin`].
const SPIN: u64 = 128;

/// The memory each side stores to before its handshake, in `u64`s: 64 MiB, past the caches of
/// the hosts the race runs on, its lines picked at random.
const FAR: usize = 64 << 17;

/// The doorbell's ring: 16 messages of 16 bytes, so none needs a pad.
const BELL_AT: RingAt = RingAt {
    header: 0,
    data_size: 256,
};

/// The room's ring: one message fills it, so each post after the first finds it full until the
/// host takes the last.
const ROOM_AT: RingAt = RingAt {
    header: 0,
    data_size: 64,
};

/// What a race came to.
#[derive(Clone, Copy, Debug, Default)]
struct Raced {
    /// Messages the guest published.
    published: u64,
    /// Wakes sent: doorbells rung, or room records posted.
    woken: u64,
    /// Handshakes the sleeper's second look settled: the host's sleep finding a message, or the
    /// guest's second look finding room.
    looked: u64,
    /// Sleeps no wake would have ended.
    lost: u64,
}

/// What a side does before its handshake: a spin of a random length, and a store to a random
/// line of memory not in its cache.
struct Lead {
    /// A xorshift sequence's state.
    state: u64,
    /// [`FAR`] `u64`s.
    far: Vec<AtomicU64>,
}

impl Lead {
    fn new(state: u64) -> Lead {
        Lead {
            state,
            far: (0..FAR).map(|_| AtomicU64::new(0)).collect(),
        }
    }

    fn next(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// A spin of up to [`SPIN`] turns.
    fn spin(&mut self) {
        for turn in 0..self.next() % SPIN {
            std::hint::black_box(turn);
        }
    }

    /// A store to a line of `far` picked at random: a store after it waits behind it.
    fn store_far(&mut self) {
        let line = (self.next() >> 7) as usize % (FAR / 8);
        self.far[line * 8].store(self.state, Ordering::Relaxed);
    }
}

/// Waits a turn: a pause for the first thousand, then a yield, so a loaded host that runs both
/// threads on one CPU still moves on.
fn wait_turn(turns: &mut u32) {
    *turns = turns.saturating_add(1);
    if *turns < 1024 {
        std::hint::spin_loop();
    } else {
        std::thread::yield_now();
    }
}

/// Waits until the ring's `consumer` reads `count`.
fn wait_for_take(region: &Region, at: RingAt, count: u64) {
    let mut turns = 0;
    while region
        .atomic_u64(at.header + CONSUMER)
        .load(Ordering::Acquire)
        != count
    {
        wait_turn(&mut turns);
    }
}

/// The guest's publish, as the kext and the bundle make it: `producer` stored, a full fence,
/// `consumer_idle` read; whether it rings the doorbell.
fn publish(region: &Region, produced: u64) -> bool {
    region
        .atomic_u64(BELL_AT.header + PRODUCER)
        .store(produced, Ordering::Release);
    fence(Ordering::SeqCst);
    region
        .atomic_u32(BELL_AT.header + CONSUMER_IDLE)
        .load(Ordering::Acquire)
        == RGFX_CONSUMER_ASLEEP
}

/// The control: [`publish`] without its fence, as the driver of issue #158 published.
fn publish_unfenced(region: &Region, produced: u64) -> bool {
    region
        .atomic_u64(BELL_AT.header + PRODUCER)
        .store(produced, Ordering::Release);
    region
        .atomic_u32(BELL_AT.header + CONSUMER_IDLE)
        .load(Ordering::Acquire)
        == RGFX_CONSUMER_ASLEEP
}

/// The control: [`Consumer::sleep_or_pending`] with its fence dropped and its store release.
fn sleep_unfenced(consumer: &Consumer, region: &Region) -> bool {
    let idle = region.atomic_u32(consumer.at.header + CONSUMER_IDLE);
    idle.store(RGFX_CONSUMER_ASLEEP, Ordering::Release);
    let pending = consumer.has_pending(region).unwrap_or(true);
    if pending {
        idle.store(0, Ordering::Release);
    }
    pending
}

/// The guest's ask for room, as the kext makes it: `producer_waiting` 1 stored, a full fence.
fn ask(region: &Region) {
    region
        .atomic_u32(ROOM_AT.header + PRODUCER_WAITING)
        .store(1, Ordering::Release);
    fence(Ordering::SeqCst);
}

/// The control: [`ask`] without its fence.
fn ask_unfenced(region: &Region) {
    region
        .atomic_u32(ROOM_AT.header + PRODUCER_WAITING)
        .store(1, Ordering::Release);
}

/// The control: [`Consumer::room_wanted`] with its fence dropped and its load acquire.
fn room_unfenced(consumer: &Consumer, region: &Region) -> bool {
    let waiting = region.atomic_u32(consumer.at.header + PRODUCER_WAITING);
    waiting.load(Ordering::Acquire) == 1 && waiting.swap(0, Ordering::SeqCst) == 1
}

/// Whether a race that has published `published` messages since `start` goes on.
fn races_on(published: u64, start: u64) -> bool {
    published < MOST && (published % 1024 != 0 || crate::clock::now_ns() - start < RACE_NS)
}

/// The doorbell's race: the guest's `publish` against the host's take and `sleep` (the host's
/// [`Consumer::sleep_or_pending`], or the control's).
fn race_the_sleep(
    publish: fn(&Region, u64) -> bool,
    sleep: fn(&Consumer, &Region) -> bool,
) -> Raced {
    let region = Region::new(4096).unwrap();
    let mut consumer = Consumer::new(BELL_AT);
    consumer.reset(&region);
    // The doorbell, rung by the guest and taken by the host's sleep; and how far the guest has
    // decided whether to ring, its `producer` once it rang or not.
    let bell = AtomicBool::new(false);
    let decided = AtomicU64::new(0);
    let done = AtomicBool::new(false);
    let (region, bell, decided, done) = (&region, &bell, &decided, &done);
    std::thread::scope(|scope| {
        let host = scope.spawn(move || {
            let mut out = Raced::default();
            let mut lead = Lead::new(0x2545_f491_4f6c_dd1d);
            loop {
                while consumer.next(region).unwrap().is_some() {}
                lead.spin();
                lead.store_far();
                if sleep(&consumer, region) {
                    out.looked += 1;
                    continue;
                }
                // Asleep until the doorbell.
                let mut turns = 0;
                loop {
                    let through = decided.load(Ordering::SeqCst);
                    if bell.swap(false, Ordering::SeqCst) {
                        break;
                    }
                    if through > consumer.consumed() {
                        out.lost += 1;
                        break;
                    }
                    if done.load(Ordering::SeqCst) {
                        return out;
                    }
                    wait_turn(&mut turns);
                }
                consumer.awake(region);
            }
        });
        let mut out = Raced::default();
        let mut lead = Lead::new(0x9e37_79b9_7f4a_7c15);
        let mut produced = 0u64;
        let start = crate::clock::now_ns();
        while races_on(out.published, start) {
            wait_for_take(region, BELL_AT, produced);
            lead.spin();
            out.published += 1;
            let header = RgfxMsgHeader {
                r#type: 1,
                size: size_of::<RgfxMsgHeader>() as u32,
                seqno: out.published,
            };
            std::hint::black_box(
                region
                    .atomic_u32(BELL_AT.header + CONSUMER_IDLE)
                    .load(Ordering::Relaxed),
            );
            lead.store_far();
            region.write(BELL_AT.data_at(produced), header.as_bytes());
            produced += size_of::<RgfxMsgHeader>() as u64;
            if publish(region, produced) {
                bell.store(true, Ordering::SeqCst);
                out.woken += 1;
            }
            decided.store(produced, Ordering::SeqCst);
        }
        wait_for_take(region, BELL_AT, produced);
        done.store(true, Ordering::SeqCst);
        let host = host.join().unwrap();
        Raced {
            looked: host.looked,
            lost: host.lost,
            ..out
        }
    })
}

/// The room's race: the guest's posts, each asking for room with `ask` where the ring is full,
/// against the host's takes, each followed by `room` (the host's [`Consumer::room_wanted`], or
/// the control's).
fn race_the_take(ask: fn(&Region), room: fn(&Consumer, &Region) -> bool) -> Raced {
    let region = Region::new(4096).unwrap();
    let mut consumer = Consumer::new(ROOM_AT);
    consumer.reset(&region);
    // The room records the host posted, and how far it has decided whether to post one, its
    // count after a take once it posted or not.
    let records = AtomicU64::new(0);
    let decided = AtomicU64::new(0);
    let done = AtomicBool::new(false);
    let (region, records, decided, done) = (&region, &records, &decided, &done);
    let len = ROOM_AT.data_size;
    std::thread::scope(|scope| {
        let host = scope.spawn(move || {
            let mut out = Raced::default();
            let mut lead = Lead::new(0x2545_f491_4f6c_dd1d);
            let mut turns = 0;
            loop {
                if let Some(message) = consumer.peek(region).unwrap() {
                    lead.spin();
                    lead.store_far();
                    consumer.take(region, &message);
                    if room(&consumer, region) {
                        records.fetch_add(1, Ordering::SeqCst);
                        out.woken += 1;
                    }
                    decided.store(consumer.consumed(), Ordering::SeqCst);
                    turns = 0;
                } else if done.load(Ordering::SeqCst) {
                    return out;
                } else {
                    wait_turn(&mut turns);
                }
            }
        });
        let mut out = Raced::default();
        let mut lead = Lead::new(0x9e37_79b9_7f4a_7c15);
        let mut produced = 0u64;
        let mut message = vec![0u8; len as usize];
        let has_room =
            |consumer: u64, produced: u64| produced - consumer + len <= ROOM_AT.data_size;
        let start = crate::clock::now_ns();
        while races_on(out.published, start) {
            lead.spin();
            // The post, as the bundle's: the records so far read before a look, the look, and
            // where it finds no room the ask and a second look, then a wait for a record.
            loop {
                let seen = records.load(Ordering::SeqCst);
                let consumed = region
                    .atomic_u64(ROOM_AT.header + CONSUMER)
                    .load(Ordering::Acquire);
                if has_room(consumed, produced) {
                    break;
                }
                lead.store_far();
                ask(region);
                let looked = region
                    .atomic_u64(ROOM_AT.header + CONSUMER)
                    .load(Ordering::Acquire);
                if has_room(looked, produced) {
                    out.looked += 1;
                    break;
                }
                let mut turns = 0;
                loop {
                    let through = decided.load(Ordering::SeqCst);
                    if records.load(Ordering::SeqCst) != seen {
                        break;
                    }
                    if through > looked {
                        out.lost += 1;
                        break;
                    }
                    wait_turn(&mut turns);
                }
            }
            out.published += 1;
            message[..16].copy_from_slice(
                RgfxMsgHeader {
                    r#type: 1,
                    size: len as u32,
                    seqno: out.published,
                }
                .as_bytes(),
            );
            region.write(ROOM_AT.data_at(produced), &message);
            produced += len;
            region
                .atomic_u64(ROOM_AT.header + PRODUCER)
                .store(produced, Ordering::Release);
        }
        wait_for_take(region, ROOM_AT, produced);
        done.store(true, Ordering::SeqCst);
        let host = host.join().unwrap();
        Raced {
            woken: host.woken,
            ..out
        }
    })
}

/// A guest's publish racing the host's look before it sleeps never leaves the host asleep with a
/// message it was not rung for: the guest's fence and the host's in [`Consumer::sleep_or_pending`]
/// keep either the host's look seeing the message or the guest's read seeing the host asleep
#[test]
fn a_publish_racing_the_hosts_sleep_never_loses_its_wakeup() {
    let raced = race_the_sleep(publish, Consumer::sleep_or_pending);
    eprintln!("the doorbell: {raced:?}");
    assert_eq!(raced.lost, 0, "{raced:?}");
}

/// A guest's ask for room racing the host's take never leaves the guest waiting for a record
/// after the host took: the guest's fence and the host's in [`Consumer::room_wanted`] keep either
/// the guest's second look seeing the take or the host's read seeing the ask
#[test]
fn a_room_ask_racing_the_hosts_take_never_loses_its_record() {
    let raced = race_the_take(ask, Consumer::room_wanted);
    eprintln!("the room: {raced:?}");
    assert_eq!(raced.lost, 0, "{raced:?}");
}

/// The control of [`a_publish_racing_the_hosts_sleep_never_loses_its_wakeup`]: without the
/// guest's fence, and without the host's, the race loses wakes
#[test]
#[ignore = "a control that loses wakes on purpose: run for a change to the race or to the fences it guards"]
fn without_a_fence_a_publish_racing_the_hosts_sleep_loses_wakeups() {
    let guest = race_the_sleep(publish_unfenced, Consumer::sleep_or_pending);
    let host = race_the_sleep(publish, sleep_unfenced);
    eprintln!("the doorbell, the guest unfenced: {guest:?}");
    eprintln!("the doorbell, the host unfenced: {host:?}");
    assert!(guest.lost > 0 && host.lost > 0, "{guest:?} {host:?}");
}

/// The control of [`a_room_ask_racing_the_hosts_take_never_loses_its_record`]: without the
/// guest's fence, and without the host's, the race loses records
#[test]
#[ignore = "a control that loses wakes on purpose: run for a change to the race or to the fences it guards"]
fn without_a_fence_a_room_ask_racing_the_hosts_take_loses_records() {
    let guest = race_the_take(ask_unfenced, Consumer::room_wanted);
    let host = race_the_take(ask, room_unfenced);
    eprintln!("the room, the guest unfenced: {guest:?}");
    eprintln!("the room, the host unfenced: {host:?}");
    assert!(guest.lost > 0 && host.lost > 0, "{guest:?} {host:?}");
}
