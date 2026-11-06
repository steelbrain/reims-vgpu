//! What a command queue's lane and the decoder's workers do with their threads under the wake
//! patterns of a game and of a WindowServer probe, in real time: a measurement, run by hand on
//! macOS (`--run-ignored only --no-capture`), which counts the threads made and ended in each
//! 10 s.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use super::lane::{Budget, LANE_THREADS, Lane};
use super::worker::Worker;

/// This process's threads whose names begin with `prefix`, by their ids, which the kernel gives
/// each thread once.
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

/// Each thread named `prefix*` seen while a run ran, sampled every 2 ms: when it was first and
/// last seen, in seconds from the start, and those there at the start and at the end.
struct Census {
    seen: HashMap<u64, (f64, f64)>,
    at_start: Vec<u64>,
    at_end: Vec<u64>,
}

impl Census {
    #[allow(clippy::disallowed_methods)] // a measurement: real time is what it measures
    fn of(prefix: &'static str, run: impl FnOnce()) -> Census {
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

    /// Threads made and ended in each of `windows` windows of `length` seconds.
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

/// A job handed at a time in a pattern, that takes `.1` of time.
#[derive(Clone, Copy, Debug)]
enum Wake {
    /// A walk on queue `.0`'s lane.
    Walk(usize, Duration),
    /// A job of key `.0` on the surface worker.
    Surface(u64, Duration),
    /// A job of key `.0` on the closes' worker.
    Close(u64, Duration),
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

/// Every `period` s from `from` to `secs`, `wake`.
fn every(wakes: &mut Vec<(f64, Wake)>, from: f64, period: f64, secs: f64, wake: Wake) {
    let mut t = from;
    while t < secs {
        wakes.push((t, wake));
        t += period;
    }
}

/// A game's pattern, over `secs`: its queue's walks of 6 ms every 10 ms on one thread, with a
/// walk beside one every 2.6 s, which takes a second thread; a surface job of 1 ms every 2 s, and
/// a close's of 1 ms every 5 s.
fn game(secs: f64) -> Vec<(f64, Wake)> {
    let ms = Duration::from_millis;
    let mut wakes = Vec::new();
    every(&mut wakes, 0.0, 0.010, secs, Wake::Walk(0, ms(6)));
    every(&mut wakes, 0.005, 2.6, secs, Wake::Walk(0, ms(6)));
    every(&mut wakes, 0.3, 2.0, secs, Wake::Surface(1, ms(1)));
    every(&mut wakes, 0.7, 5.0, secs, Wake::Close(1, ms(1)));
    wakes.sort_by(|a, b| a.0.total_cmp(&b.0));
    wakes
}

/// A WindowServer probe that scrolls pages in Safari, over `secs`: 8 queues, each with bursts of
/// 1 to 3 walks of 2 ms at gaps of 0.3 to 2.5 s; a surface job of 2 ms for a new block every
/// 150 ms, and another for it 1.5 s later; and a close's job of 1 ms every 3 s.
fn b1(secs: f64) -> Vec<(f64, Wake)> {
    let ms = Duration::from_millis;
    let mut rand = Rand(0x9e37_79b9_7f4a_7c15);
    let mut wakes = Vec::new();
    for queue in 0..8 {
        let mut t = rand.next();
        while t < secs {
            let walks = 1 + (3.0 * rand.next()) as usize;
            for _ in 0..walks {
                wakes.push((t, Wake::Walk(queue, ms(2))));
            }
            t += 0.3 + 2.2 * rand.next();
        }
    }
    let mut block = 0;
    let mut t = 0.0;
    while t < secs {
        wakes.push((t, Wake::Surface(block, ms(2))));
        if t + 1.5 < secs {
            wakes.push((t + 1.5, Wake::Surface(block, ms(2))));
        }
        block += 1;
        t += 0.15;
    }
    every(&mut wakes, 0.5, 3.0, secs, Wake::Close(1, ms(1)));
    wakes.sort_by(|a, b| a.0.total_cmp(&b.0));
    wakes
}

/// Hands each job of `wakes` over at its time, from now, and waits for the workers' last to end.
#[allow(clippy::disallowed_methods)] // a measurement: real time is what it measures
fn replay(lanes: &[Arc<Lane>], surfaces: &Worker, closes: &Worker, wakes: &[(f64, Wake)]) {
    let start = Instant::now();
    for &(t, wake) in wakes {
        let left = t - start.elapsed().as_secs_f64();
        if left > 0.0 {
            std::thread::sleep(Duration::from_secs_f64(left));
        }
        match wake {
            Wake::Walk(queue, takes) => {
                lanes[queue].run(Box::new(move || std::thread::sleep(takes)));
            }
            Wake::Surface(key, takes) => {
                surfaces.submit(key, Box::new(move || std::thread::sleep(takes)));
            }
            Wake::Close(key, takes) => {
                closes.submit(key, Box::new(move || std::thread::sleep(takes)));
            }
        }
    }
    surfaces.flush();
    closes.flush();
}

/// The lanes' and workers' threads made and ended in each 10 s of 30 s of each pattern.
#[test]
#[ignore = "a measurement, run by hand on macOS with --no-capture"]
fn lane_and_worker_threads_under_a_game_s_and_a_windowserver_probe_s_wakes() {
    const SECS: f64 = 30.0;
    for (name, wakes) in [("game", game(SECS)), ("b1", b1(SECS))] {
        let budget = Budget::new(2 * LANE_THREADS * 8);
        let lanes: Vec<Arc<Lane>> = (0..8)
            .map(|queue| Lane::new(0x100 + queue, Arc::clone(&budget)))
            .collect();
        let surfaces = Worker::new("reims-vgpu.graphics-surfaces");
        let closes = Worker::new("reims-vgpu.graphics-closes");
        let census = Census::of("reims-vgpu.graphics-", || {
            replay(&lanes, &surfaces, &closes, &wakes);
        });
        let (made, ended) = census.per_window(10.0, 3);
        println!(
            "decode {name}: {} jobs; threads made per 10 s {made:?}, ended {ended:?}; {} at the \
             end",
            wakes.len(),
            census.at_end.len(),
        );
        for lane in &lanes {
            lane.close();
        }
    }
}
