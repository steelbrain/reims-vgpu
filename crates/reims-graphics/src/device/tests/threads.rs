//! Ring 0's threads (`lanes`, "Threads"), on a clock the tests move: a steady load's lanes keep
//! the threads it ran at once through its pauses, and the threads only a burst ran go once they
//! have waited [`KEEP`].

use parking_lot::Condvar;

use super::*;
use crate::device::lanes::Ring0Threads;
use crate::keep::{KEEP, Keep, TestClock};

/// A host that holds each `BUFFER_CREATE` in its call while shut, counting those it holds, and
/// carries out every other message doing nothing, counting every message it is handed.
#[derive(Clone, Default)]
struct GateHost(Arc<Gate>);

#[derive(Default)]
struct Gate {
    state: Mutex<Gated>,
    moved: Condvar,
}

#[derive(Default)]
struct Gated {
    shut: bool,
    held: usize,
    seen: usize,
}

impl GateHost {
    fn shut(&self) {
        self.0.state.lock().shut = true;
    }

    fn open(&self) {
        self.0.state.lock().shut = false;
        self.0.moved.notify_all();
    }

    /// How many messages it was handed.
    fn seen(&self) -> usize {
        self.0.state.lock().seen
    }

    /// Waits for `n` creates to be held.
    fn wait_held(&self, n: usize) {
        let mut gated = self.0.state.lock();
        while gated.held < n {
            let waited = self.0.moved.wait_for(&mut gated, Duration::from_secs(10));
            assert!(!waited.timed_out(), "{} creates held of {n}", gated.held);
        }
    }
}

impl Host for GateHost {
    fn raise_interrupt(&self, _vector: u32) {}
    fn stage3_message(&self, _completion: &CompletionPoster, message: &[u8]) {
        let (header, _) = RgfxMsgHeader::read_from_prefix(message).unwrap();
        let mut gated = self.0.state.lock();
        gated.seen += 1;
        if header.r#type != RGFX_CMD_BUFFER_CREATE {
            return;
        }
        gated.held += 1;
        self.0.moved.notify_all();
        while gated.shut {
            self.0.moved.wait(&mut gated);
        }
        gated.held -= 1;
    }
}

/// A rig over `host` whose ring 0's threads wait [`KEEP`] of `clock`'s time for a lane.
fn rig_on(clock: &Arc<TestClock>, host: GateHost) -> Rig {
    let rig = Rig::with_lanes(Box::new(host), Lanes::with_keep(Keep::on(clock, KEEP)));
    rig.start();
    rig
}

/// Waits about 10 s for `done`.
fn until(what: &str, done: impl Fn() -> bool) {
    for _ in 0..10_000 {
        if done() {
            return;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    panic!("{what}");
}

/// Object `n` of `context`.
fn obj(context: u64, n: u64) -> u64 {
    (context << RGFX_CONTEXT_OBJ_ID_SHIFT) | n
}

/// Waits until `host` was handed `seen` messages, the lanes have carried out what they were
/// handed, and `spares` of ring 0's threads wait for a lane.
fn settled(host: &GateHost, lanes: &Lanes, seen: usize, spares: usize) {
    until(
        &format!("{seen} messages, {spares} threads waiting"),
        || host.seen() >= seen && lanes.idle() && lanes.threads().spares == spares,
    );
}

/// Contexts' lanes busy one at a time, each after a pause longer than a lane's thread was once
/// kept (1 s), are served by the one thread the first made, and no lane holds it in between; with
/// no lane busy for [`KEEP`], it goes.
#[test]
fn a_steady_load_s_lanes_are_served_by_one_thread_through_its_pauses() {
    let clock = TestClock::new();
    let host = GateHost::default();
    let mut rig = rig_on(&clock, host.clone());
    let lanes = Arc::clone(&rig.device.shared.lanes);
    for n in 0..30 {
        let context = 4 + n % 3;
        rig.send(object_message(RGFX_CMD_EVENT_CREATE, obj(context, n + 1)));
        settled(&host, &lanes, n as usize + 1, 1);
        assert!(
            !lanes.served(Key::Context(context as u32)),
            "an idle lane holds its thread"
        );
        clock.advance(Duration::from_secs(2));
    }
    let threads = Ring0Threads {
        live: 1,
        spares: 1,
        made: 1,
        gone: 0,
    };
    assert_eq!(lanes.threads(), threads);
    clock.advance(KEEP);
    until("the idle thread did not go", || lanes.threads().live == 0);
    assert_eq!(
        lanes.threads(),
        Ring0Threads {
            live: 0,
            spares: 0,
            made: 1,
            gone: 1
        }
    );
    // The next message makes a thread again.
    rig.send(object_message(RGFX_CMD_EVENT_CREATE, obj(4, 99)));
    settled(&host, &lanes, 31, 1);
    assert_eq!(lanes.threads().made, 2);
}

/// Three contexts' lanes busy at once have a thread each; then one lane at a time, each after a
/// pause of 2 s, is served by the thread that went idle last, and the other two go once they have
/// waited [`KEEP`] since the burst, while the one that serves stays.
#[test]
fn a_burst_s_other_threads_go_once_they_have_waited_keep() {
    let clock = TestClock::new();
    let host = GateHost::default();
    let mut rig = rig_on(&clock, host.clone());
    let lanes = Arc::clone(&rig.device.shared.lanes);
    host.shut();
    for context in [4, 5, 6] {
        rig.send(object_message(RGFX_CMD_BUFFER_CREATE, obj(context, 1)));
    }
    host.wait_held(3);
    assert_eq!(lanes.threads().live, 3, "a thread a busy lane");
    host.open();
    settled(&host, &lanes, 3, 3);
    let pauses = KEEP.as_secs() / 2;
    for n in 0..pauses {
        clock.advance(Duration::from_secs(2));
        rig.send(object_message(RGFX_CMD_EVENT_CREATE, obj(7, n + 1)));
        if n + 1 < pauses {
            settled(&host, &lanes, 4 + n as usize, 3);
            assert_eq!(lanes.threads().gone, 0, "{} s after the burst", 2 * (n + 1));
        }
    }
    let seen = 3 + pauses as usize;
    until("the burst's other threads stayed", || {
        host.seen() >= seen
            && lanes.idle()
            && lanes.threads()
                == Ring0Threads {
                    live: 1,
                    spares: 1,
                    made: 3,
                    gone: 2,
                }
    });
    clock.advance(KEEP);
    until("the last thread stayed", || lanes.threads().live == 0);
    assert_eq!(lanes.threads().gone, 3);
}

/// A device's end ends the threads that wait for a lane, however long they had left to wait.
#[test]
fn the_device_s_end_ends_the_threads_that_wait_for_a_lane() {
    let clock = TestClock::new();
    let host = GateHost::default();
    let mut rig = rig_on(&clock, host.clone());
    let lanes = Arc::clone(&rig.device.shared.lanes);
    rig.send(object_message(RGFX_CMD_EVENT_CREATE, obj(4, 1)));
    settled(&host, &lanes, 1, 1);
    rig.device.shared.end();
    until("the waiting thread stayed past the end", || {
        lanes.threads().live == 0
    });
}

/// What ring 0's threads do under the wake patterns of a WindowServer probe and of a game, in real
/// time: a measurement, run by hand on macOS (`--run-ignored only --no-capture`).
#[cfg(target_os = "macos")]
mod churn {
    use std::collections::HashMap;
    use std::time::Instant;

    use super::*;

    /// This process's threads whose names begin with `prefix`, by their ids, which the kernel
    /// gives each thread once.
    #[allow(deprecated)] // libc points its Mach calls at the mach2 crate; a measurement's
    fn threads_named(prefix: &str) -> Vec<u64> {
        unsafe extern "C" {
            fn mach_port_deallocate(
                task: libc::mach_port_t,
                name: libc::mach_port_t,
            ) -> libc::kern_return_t;
        }
        let mut ids = Vec::new();
        // SAFETY: the task's own port.
        let task = unsafe { libc::mach_task_self() };
        let mut list: libc::thread_act_array_t = std::ptr::null_mut();
        let mut count: libc::mach_msg_type_number_t = 0;
        // SAFETY: writes the task's threads' ports and how many.
        if unsafe { libc::task_threads(task, &mut list, &mut count) } != libc::KERN_SUCCESS {
            return ids;
        }
        for i in 0..count as usize {
            // SAFETY: `list` holds `count` ports.
            let port = unsafe { *list.add(i) };
            // SAFETY: all zeroes is a valid record of each.
            let mut id: libc::thread_identifier_info = unsafe { std::mem::zeroed() };
            let mut ext: libc::thread_extended_info = unsafe { std::mem::zeroed() };
            let (mut n, mut m) = (
                libc::THREAD_IDENTIFIER_INFO_COUNT,
                libc::THREAD_EXTENDED_INFO_COUNT,
            );
            // SAFETY: each record holds as many integers as its count says.
            let known = unsafe {
                libc::thread_info(
                    port,
                    libc::THREAD_IDENTIFIER_INFO as libc::thread_flavor_t,
                    (&raw mut id).cast(),
                    &mut n,
                ) == libc::KERN_SUCCESS
                    && libc::thread_info(
                        port,
                        libc::THREAD_EXTENDED_INFO as libc::thread_flavor_t,
                        (&raw mut ext).cast(),
                        &mut m,
                    ) == libc::KERN_SUCCESS
            };
            if known {
                let name: Vec<u8> = (ext.pth_name.iter())
                    .take_while(|&&c| c != 0)
                    .map(|&c| c as u8)
                    .collect();
                if name.starts_with(prefix.as_bytes()) {
                    ids.push(id.thread_id);
                }
            }
            // SAFETY: the send right `task_threads` gave.
            unsafe { mach_port_deallocate(task, port) };
        }
        let size = count as usize * size_of::<libc::thread_act_t>();
        // SAFETY: the list `task_threads` allocated.
        unsafe { libc::vm_deallocate(task, list as libc::vm_address_t, size as libc::vm_size_t) };
        ids
    }

    /// Each thread named `prefix*` seen while `run` ran, sampled every 2 ms: when it was first and
    /// last seen, in seconds from the start, and whether it was there at the start or the end.
    struct Census {
        seen: HashMap<u64, (f64, f64)>,
        at_start: Vec<u64>,
        at_end: Vec<u64>,
    }

    #[allow(clippy::disallowed_methods)] // a measurement: real time is what it measures
    fn census(prefix: &'static str, run: impl FnOnce()) -> Census {
        let stop = Arc::new(AtomicBool::new(false));
        let at_start = threads_named(prefix);
        let start = Instant::now();
        let sampler = {
            let stop = Arc::clone(&stop);
            std::thread::Builder::new()
                .name("churn-census".into())
                .spawn(move || {
                    let mut seen: HashMap<u64, (f64, f64)> = HashMap::new();
                    while !stop.load(Ordering::SeqCst) {
                        let t = start.elapsed().as_secs_f64();
                        for id in threads_named(prefix) {
                            seen.entry(id).or_insert((t, t)).1 = t;
                        }
                        std::thread::sleep(Duration::from_millis(2));
                    }
                    seen
                })
                .unwrap()
        };
        run();
        stop.store(true, Ordering::SeqCst);
        let seen = sampler.join().unwrap();
        Census {
            seen,
            at_start,
            at_end: threads_named(prefix),
        }
    }

    impl Census {
        /// Threads made and ended in each `window` of `length` seconds.
        fn per_window(&self, length: f64, windows: usize) -> (Vec<usize>, Vec<usize>) {
            let mut made = vec![0; windows];
            let mut ended = vec![0; windows];
            let at = |t: f64| ((t / length) as usize).min(windows - 1);
            for (id, &(first, last)) in &self.seen {
                if !self.at_start.contains(id) {
                    made[at(first)] += 1;
                }
                if !self.at_end.contains(id) {
                    ended[at(last)] += 1;
                }
            }
            (made, ended)
        }
    }

    /// A message ring 0 is handed at a time in a pattern.
    #[derive(Clone, Copy, Debug)]
    enum Wake {
        /// A context's object-lifecycle message: its lane.
        Context(u64),
        /// A block's pages: its lane.
        Pages(u64),
        /// A present: the presents' lane.
        Present,
    }

    /// A pseudo-random number in `[0, 1)`, the same sequence each run.
    struct Rand(u64);

    impl Rand {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }
    }

    /// The pattern of a WindowServer probe that scrolls pages in Safari, over `secs`: presents at
    /// 60 Hz; 10 processes' contexts each posting at gaps of 1.1 to 3 s; and a block of a new
    /// surface every 150 ms, whose pages are handed over again 1.5 s later.
    fn b1(secs: f64) -> Vec<(f64, Wake)> {
        let mut rand = Rand(0x9e37_79b9_7f4a_7c15);
        let mut wakes = Vec::new();
        let mut t = 0.0;
        while t < secs {
            wakes.push((t, Wake::Present));
            t += 1.0 / 60.0;
        }
        for context in 10..20 {
            let mut t = rand.next();
            while t < secs {
                wakes.push((t, Wake::Context(context)));
                t += 1.1 + 1.9 * rand.next();
            }
        }
        let mut block = 0;
        let mut t = 0.0;
        while t < secs {
            let offset = RGFX_CONTEXT_ALLOCATIONS_OFFSET + block * 0x1_0000;
            wakes.push((t, Wake::Pages(offset)));
            if t + 1.5 < secs {
                wakes.push((t + 1.5, Wake::Pages(offset)));
            }
            block += 1;
            t += 0.15;
        }
        wakes.sort_by(|a, b| a.0.total_cmp(&b.0));
        wakes
    }

    /// A game's pattern, over `secs`: presents at 60 Hz, its context posting every 1.5 s, and a
    /// block's pages handed over every 2.5 s.
    fn game(secs: f64) -> Vec<(f64, Wake)> {
        let mut wakes = Vec::new();
        let mut t = 0.0;
        while t < secs {
            wakes.push((t, Wake::Present));
            t += 1.0 / 60.0;
        }
        let mut t = 0.2;
        while t < secs {
            wakes.push((t, Wake::Context(4)));
            t += 1.5;
        }
        let mut t = 0.4;
        while t < secs {
            wakes.push((t, Wake::Pages(RGFX_CONTEXT_ALLOCATIONS_OFFSET)));
            t += 2.5;
        }
        wakes.sort_by(|a, b| a.0.total_cmp(&b.0));
        wakes
    }

    /// Takes every answer the device has posted, waiting for none.
    fn drain(rig: &mut Rig) {
        let data = RGFX_CONTROL_RING_DATA_SIZE as u64;
        let at = |count: u64| RGFX_CONTROL_COMPLETE_OFFSET + 256 + (count % data);
        loop {
            let produced = (rig.region())
                .atomic_u64(RGFX_CONTROL_COMPLETE_OFFSET)
                .load(Ordering::SeqCst);
            if produced == rig.consumed {
                return;
            }
            let head = rig.region().read(at(rig.consumed), 8).unwrap();
            rig.consumed += u64::from(u32::from_le_bytes(head[4..8].try_into().unwrap()));
            (rig.region())
                .atomic_u64(RGFX_CONTROL_COMPLETE_OFFSET + 0x40)
                .store(rig.consumed, Ordering::SeqCst);
        }
    }

    /// Hands ring 0 each message of `wakes` at its time, from now.
    #[allow(clippy::disallowed_methods)] // a measurement: real time is what it measures
    fn replay(rig: &mut Rig, wakes: &[(f64, Wake)]) {
        let start = Instant::now();
        let mut n = 0u64;
        for &(t, wake) in wakes {
            let left = t - start.elapsed().as_secs_f64();
            if left > 0.0 {
                std::thread::sleep(Duration::from_secs_f64(left));
            }
            n += 1;
            let message = match wake {
                Wake::Context(context) => object_message(RGFX_CMD_EVENT_CREATE, obj(context, n)),
                Wake::Pages(block) => surface_pages_message(30, block, 0),
                Wake::Present => present_surface(0, 0),
            };
            rig.send(message);
            drain(rig);
        }
        rig.lanes_done();
        drain(rig);
    }

    /// Ring 0's threads made and ended in each 10 s of 30 s of each pattern, as a host's device
    /// runs them.
    #[test]
    #[ignore = "a measurement, run by hand on macOS with --no-capture"]
    fn ring_0_s_threads_under_a_windowserver_probe_s_and_a_game_s_wakes() {
        const SECS: f64 = 30.0;
        for (name, wakes) in [("b1", b1(SECS)), ("game", game(SECS))] {
            let mut rig = Rig::with_host(Box::new(GateHost::default()));
            rig.start();
            let census = census("reims-vgpu.graphics-", || replay(&mut rig, &wakes));
            let (made, ended) = census.per_window(10.0, 3);
            println!(
                "ring 0 {name}: {} messages; threads made per 10 s {made:?}, ended {ended:?}; \
                 {} at the start, {} at the end",
                wakes.len(),
                census.at_start.len(),
                census.at_end.len(),
            );
        }
    }
}
