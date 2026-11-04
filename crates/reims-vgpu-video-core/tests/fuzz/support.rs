//! The fuzzers' own harness: a seeded generator, byte mutators, an
//! allocator that counts, and a runner that holds each input to a CPU
//! time and a memory bound and catches its panics. It is ours, so that the
//! workspace needs no fuzzing crate from a registry and no nightly
//! compiler: each fuzzer is a plain `#[test]` over a seeded generator, in
//! the debug test build, whose overflow checks are part of the point.
//!
//! Core's fuzz target (`tests/fuzz`) and the codec crate's unit tests
//! include this file by path.
//!
//! A run is reproducible from its seed. Input `i` of a target is made
//! from a seed of its own, a mix of the run's seed, the target's name and
//! `i`, so `RVGV_FUZZ_SEED` and `RVGV_FUZZ_INDEX` replay one input, and a
//! failure prints both. `RVGV_FUZZ_CASES` sizes a long run, and
//! `RVGV_FUZZ_SECONDS` caps it.

#![allow(dead_code)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::io::Write;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Once;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use reims_vgpu_video_core::runtime::{Runtime, StdRuntime};
use reims_vgpu_video_core::sync::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// SplitMix64 (Steele, Lea and Flood, 2014): small, fast and seedable.
#[derive(Clone, Debug)]
pub struct Rng(u64);

/// SplitMix64's output function, which mixes one word into another.
pub const fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix(self.0)
    }

    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// A value in `0..n`, 0 when `n` is 0.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// A value in `low..=high`.
    pub fn range(&mut self, low: u64, high: u64) -> u64 {
        match (high - low).checked_add(1) {
            Some(span) => low + self.next_u64() % span,
            None => self.next_u64(),
        }
    }

    /// A u32 in `low..=high`.
    pub fn range_u32(&mut self, low: u32, high: u32) -> u32 {
        self.range(u64::from(low), u64::from(high)) as u32
    }

    /// True one time in `n`.
    pub fn one_in(&mut self, n: usize) -> bool {
        self.below(n) == 0
    }

    pub fn bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }

    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }

    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next_u64() as u8).collect()
    }

    /// Bytes of a length [`Rng::count`] picks.
    pub fn some_bytes(&mut self, max: usize) -> Vec<u8> {
        let n = self.count(max);
        self.bytes(n)
    }

    /// A u32 that is often a boundary: 0, 1, a power of two and its
    /// neighbours, the top of a signed or unsigned range.
    pub fn interesting_u32(&mut self) -> u32 {
        match self.below(8) {
            0 => 0,
            1 => 1,
            2 => u32::MAX,
            3 => i32::MAX as u32,
            4 => 0x8000_0000,
            5 => {
                let shift = self.below(32);
                (1u32 << shift)
                    .wrapping_add(self.range(0, 2) as u32)
                    .wrapping_sub(1)
            }
            6 => self.below(64) as u32,
            _ => self.next_u32(),
        }
    }

    /// A u64 that is often a boundary, as [`Rng::interesting_u32`].
    pub fn interesting_u64(&mut self) -> u64 {
        match self.below(6) {
            0 => u64::MAX,
            1 => i64::MAX as u64,
            2 => 1 << 63,
            3 => {
                let shift = self.below(64);
                (1u64 << shift)
                    .wrapping_add(self.range(0, 2))
                    .wrapping_sub(1)
            }
            4 => self.next_u64(),
            _ => u64::from(self.interesting_u32()),
        }
    }

    /// An i32 that is often a boundary.
    pub fn interesting_i32(&mut self) -> i32 {
        match self.below(4) {
            0 => i32::MIN,
            1 => -(self.below(64) as i32),
            _ => self.interesting_u32().cast_signed(),
        }
    }

    /// A small count, most often 0 to 3, now and then up to `max`.
    pub fn count(&mut self, max: usize) -> usize {
        if self.one_in(8) {
            self.below(max + 1)
        } else {
            self.below(max.min(3) + 1)
        }
    }
}

/// Mutates `bytes` once, in one of the ways a fuzzer mutates: flip a bit,
/// set a byte or a big-endian u32 to a boundary value, insert, delete,
/// duplicate or splice a range, or truncate.
pub fn mutate(rng: &mut Rng, bytes: &mut Vec<u8>, donor: &[u8]) {
    let len = bytes.len();
    match rng.below(11) {
        0 if len > 0 => {
            let at = rng.below(len);
            bytes[at] ^= 1 << rng.below(8);
        }
        1 if len > 0 => {
            let at = rng.below(len);
            bytes[at] = *rng.pick(&[0, 1, 2, 3, 0x7f, 0x80, 0xfe, 0xff, 0x0f, 0x10]);
        }
        2 | 3 if len >= 4 => {
            // Atoms and parameters lead with big-endian u32 sizes: aligned
            // offsets hit them most often.
            let at = if rng.bool() {
                rng.below(len / 4) * 4
            } else {
                rng.below(len - 3)
            };
            let value = if rng.bool() {
                rng.interesting_u32()
            } else {
                // A size near what the bytes hold.
                (len as u32)
                    .wrapping_add(rng.range(0, 16) as u32)
                    .wrapping_sub(8)
            };
            let value = if rng.bool() {
                value.to_be_bytes()
            } else {
                value.to_le_bytes()
            };
            bytes[at..at + 4].copy_from_slice(&value);
        }
        4 => {
            let at = rng.below(len + 1);
            let n = rng.range(1, 16) as usize;
            let insert = rng.bytes(n);
            bytes.splice(at..at, insert);
        }
        5 if len > 0 => {
            let at = rng.below(len);
            let end = (at + rng.range(1, 64) as usize).min(len);
            bytes.drain(at..end);
        }
        6 if len > 0 => {
            let at = rng.below(len);
            let end = (at + rng.range(1, 256) as usize).min(len);
            let copy = bytes[at..end].to_vec();
            let to = rng.below(len + 1);
            bytes.splice(to..to, copy);
        }
        7 if !donor.is_empty() => {
            let at = rng.below(donor.len());
            let end = (at + rng.range(1, 512) as usize).min(donor.len());
            let to = rng.below(len + 1);
            bytes.splice(to..to, donor[at..end].iter().copied());
        }
        8 if len > 0 => bytes.truncate(rng.below(len)),
        9 if len > 0 => {
            // A run of one byte, as zeros or 0xff padding.
            let at = rng.below(len);
            let end = (at + rng.range(1, 32) as usize).min(len);
            let fill = *rng.pick(&[0, 0xff, 0x80]);
            bytes[at..end].fill(fill);
        }
        _ if len > 0 => {
            let at = rng.below(len);
            bytes[at] = rng.next_u64() as u8;
        }
        _ => bytes.push(rng.next_u64() as u8),
    }
}

/// Mutates `bytes` `rounds` times.
pub fn mutate_rounds(rng: &mut Rng, bytes: &mut Vec<u8>, donor: &[u8], rounds: usize) {
    for _ in 0..rounds {
        mutate(rng, bytes, donor);
    }
}

/// Mutates `bytes` a few times, most often once, or leaves them whole
/// one time in `whole`.
pub fn maybe_mutate(rng: &mut Rng, bytes: &mut Vec<u8>, donor: &[u8], whole: usize) {
    if !rng.one_in(whole) {
        let rounds = 1 + rng.count(8);
        mutate_rounds(rng, bytes, donor, rounds);
    }
}

/// Counts what the process, and each thread, has allocated, so that an
/// input that allocates past its bound is caught. An allocation that takes
/// the process past [`HARD_LIMIT`] prints the input and aborts, before a
/// bomb takes the machine.
pub struct CountingAllocator;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);
/// The input in hand, its target's name and its seed and index, for the
/// abort's message. Of the last input to start, where targets run side by
/// side.
static CASE_NAME: AtomicPtr<&'static str> = AtomicPtr::new(std::ptr::null_mut());
static CASE_SEED: AtomicU64 = AtomicU64::new(0);
static CASE_INDEX: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// What this thread has allocated less what it has freed, which goes
    /// below 0 when it frees what another allocated.
    static THREAD_LIVE: Cell<isize> = const { Cell::new(0) };
    static THREAD_PEAK: Cell<isize> = const { Cell::new(0) };
}

/// The most a test process may hold at once.
pub const HARD_LIMIT: usize = 6 << 30;

fn count_thread(delta: isize) {
    // `try_with`: a thread's own counters may be gone as it exits.
    let _ = THREAD_LIVE.try_with(|live| {
        let now = live.get().wrapping_add(delta);
        live.set(now);
        let _ = THREAD_PEAK.try_with(|peak| {
            if now > peak.get() {
                peak.set(now);
            }
        });
    });
}

fn count_grown(size: usize) {
    let live = LIVE.fetch_add(size, Ordering::Relaxed) + size;
    if live > HARD_LIMIT {
        over_the_limit(size);
    }
    PEAK.fetch_max(live, Ordering::Relaxed);
    count_thread(size.cast_signed());
}

fn count_shrunk(size: usize) {
    LIVE.fetch_sub(size, Ordering::Relaxed);
    count_thread(-size.cast_signed());
}

// SAFETY: every call is passed to the system allocator as it came; the
// counting around it neither allocates nor touches the memory.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count_grown(layout.size());
        // SAFETY: as the caller asked of us.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count_grown(layout.size());
        // SAFETY: as the caller asked of us.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        count_shrunk(layout.size());
        // SAFETY: as the caller asked of us.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > layout.size() {
            count_grown(new_size - layout.size());
        } else {
            count_shrunk(layout.size() - new_size);
        }
        // SAFETY: as the caller asked of us.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

/// Prints the input in hand without allocating, and aborts.
#[expect(
    clippy::disallowed_methods,
    reason = "the counting allocator ends the process past its limit: a panic would allocate"
)]
fn over_the_limit(size: usize) -> ! {
    let mut line = [0u8; 200];
    let mut cursor = std::io::Cursor::new(&mut line[..]);
    let _ = writeln!(
        cursor,
        "fuzz: input {} of seed {:#x} took the process past {} bytes ({} asked)",
        CASE_INDEX.load(Ordering::Relaxed),
        CASE_SEED.load(Ordering::Relaxed),
        HARD_LIMIT,
        size
    );
    let end = usize::try_from(cursor.position()).unwrap_or(0);
    let _ = std::io::stderr().write_all(&line[..end]);
    std::process::abort();
}

/// Names the input in hand on stderr as the process aborts, as it does
/// after a thread overflows its stack (Rust's handler prints the thread
/// and aborts) and past [`HARD_LIMIT`]: abort raises SIGABRT, whose
/// handler this is, and ends the process once it returns.
#[cfg(unix)]
fn install_abort_report() {
    unsafe extern "C" {
        fn signal(signal: i32, handler: extern "C" fn(i32)) -> usize;
    }
    /// SIGABRT, 6 on Linux and on macOS.
    const ABORT: i32 = 6;
    // SAFETY: the handler touches only atomics and a buffer on its own
    // stack, and writes it to stderr.
    unsafe {
        signal(ABORT, report_abort);
    }
}

#[cfg(not(unix))]
fn install_abort_report() {}

/// SIGABRT's handler: the input in hand, written without allocating.
#[cfg(unix)]
extern "C" fn report_abort(_: i32) {
    unsafe extern "C" {
        fn write(fd: std::ffi::c_int, bytes: *const std::ffi::c_void, count: usize) -> isize;
    }
    let name = CASE_NAME.load(Ordering::SeqCst);
    // SAFETY: null, or a name `run` leaked, which lives for ever.
    let name = unsafe { name.as_ref() }.map_or(&b""[..], |name| name.as_bytes());
    let mut line = [0u8; 300];
    let mut cursor = std::io::Cursor::new(&mut line[..]);
    let _ = cursor.write_all(b"fuzz ");
    let _ = cursor.write_all(name);
    let _ = writeln!(
        cursor,
        ": the process aborted in input {} of seed {:#x}; RVGV_FUZZ_SEED and \
         RVGV_FUZZ_INDEX replay it",
        CASE_INDEX.load(Ordering::SeqCst),
        CASE_SEED.load(Ordering::SeqCst),
    );
    let end = usize::try_from(cursor.position()).unwrap_or(0);
    // SAFETY: `end` bytes of `line` are initialised.
    unsafe {
        write(2, line.as_ptr().cast(), end);
    }
}

fn thread_live() -> isize {
    THREAD_LIVE.try_with(Cell::get).unwrap_or(0)
}

fn reset_thread_peak() -> isize {
    let live = thread_live();
    let _ = THREAD_PEAK.try_with(|peak| peak.set(live));
    live
}

fn thread_peak() -> isize {
    THREAD_PEAK.try_with(Cell::get).unwrap_or(0)
}

static HOOK: Once = Once::new();
/// Set by a panic on any thread: a host session's own threads too.
static PANICKED: AtomicBool = AtomicBool::new(false);
static PANIC_MESSAGE: Mutex<String> = Mutex::new(String::new());

/// Records each panic, then hands it to the hook that was there, so that
/// the other tests of a binary print theirs as before.
fn install_panic_hook() {
    HOOK.call_once(|| {
        install_abort_report();
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            PANICKED.store(true, Ordering::SeqCst);
            *PANIC_MESSAGE.lock() = format!("{info}");
            previous(info);
        }));
    });
}

/// Inputs of targets that count per thread run side by side; one that
/// counts the whole process runs alone among the fuzzers of its binary.
static EXCLUSIVE: RwLock<()> = RwLock::new(());

/// Where a target's allocations and CPU time are counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Accounting {
    /// On the thread that runs the input: its work stays there, and the
    /// binary's other tests do not disturb the count.
    Thread,
    /// In the whole process, for work on threads of its own, as a host
    /// session's. Its panics are caught on every thread. Such a target
    /// runs alone among the binary's fuzzers, and the binary must have no
    /// other tests that allocate much or take the CPU meanwhile.
    Process,
}

impl Accounting {
    /// The CPU time spent so far where it counts: the calling thread's,
    /// or the process's, all its threads together. Unlike the wall clock,
    /// it stops while the host runs other work, so a host under load does
    /// not make an input slow.
    fn cpu_time(self) -> Duration {
        let clock = match self {
            Accounting::Thread => libc::CLOCK_THREAD_CPUTIME_ID,
            Accounting::Process => libc::CLOCK_PROCESS_CPUTIME_ID,
        };
        let mut now = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        // SAFETY: `now` is a timespec the call writes, and nothing else.
        let read = unsafe { libc::clock_gettime(clock, &mut now) };
        assert_eq!(read, 0, "clock_gettime({clock}) failed");
        Duration::new(
            u64::try_from(now.tv_sec).unwrap_or(0),
            u32::try_from(now.tv_nsec).unwrap_or(0),
        )
    }
}

/// A fuzz target and its bounds.
#[derive(Clone, Copy, Debug)]
pub struct Target {
    /// Its name, which seeds its inputs.
    pub name: &'static str,
    pub accounting: Accounting,
    /// The most CPU time one input may take, counted as its accounting
    /// counts ([`Accounting::cpu_time`]).
    pub input_time: Duration,
    /// The most one input may allocate beyond what was live before it.
    pub input_bytes: usize,
}

impl Target {
    /// A target counted per thread, an input taking 2 s of CPU time and
    /// 256 MiB at most: far past what any input of a few kilobytes needs.
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            accounting: Accounting::Thread,
            input_time: Duration::from_secs(2),
            input_bytes: 256 << 20,
        }
    }

    pub const fn process(self) -> Self {
        Self {
            accounting: Accounting::Process,
            ..self
        }
    }

    pub const fn input_time(self, input_time: Duration) -> Self {
        Self { input_time, ..self }
    }

    pub const fn input_bytes(self, input_bytes: usize) -> Self {
        Self {
            input_bytes,
            ..self
        }
    }
}

/// What a run did.
#[derive(Debug)]
pub struct Report {
    pub inputs: u64,
    pub elapsed: Duration,
    /// The most CPU time an input took.
    pub slowest: Duration,
    pub most_bytes: usize,
}

fn env(name: &str) -> Option<u64> {
    let value = std::env::var(name).ok()?;
    let value = value.trim();
    match value.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16).ok(),
        None => value.parse().ok(),
    }
}

fn name_hash(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3)
    })
}

/// The seed of input `index` of `name` in a run of seed `seed`.
pub fn input_seed(seed: u64, name: &str, index: u64) -> u64 {
    mix(mix(seed ^ name_hash(name)) ^ index)
}

/// The quick run `make check` makes: `inputs` inputs of seed 0, or of
/// `RVGV_FUZZ_SEED`.
pub fn quick(target: &Target, inputs: u64, case: impl Fn(&mut Rng)) -> Report {
    let seed = env("RVGV_FUZZ_SEED").unwrap_or(0);
    run(target, seed, inputs, None, case)
}

/// A long run, for an ignored test: `RVGV_FUZZ_CASES` inputs (100,000 by
/// default), for `RVGV_FUZZ_SECONDS` at most (none by default), of
/// `RVGV_FUZZ_SEED`, or of a seed from the clock, which it prints.
#[expect(
    clippy::disallowed_methods,
    reason = "a long run without a seed takes one from the wall clock, and prints it"
)]
pub fn long(target: &Target, case: impl Fn(&mut Rng)) -> Report {
    let seed = env("RVGV_FUZZ_SEED").unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(1, |since| mix(since.as_nanos() as u64))
    });
    let inputs = env("RVGV_FUZZ_CASES").unwrap_or(100_000);
    let time = env("RVGV_FUZZ_SECONDS").map(Duration::from_secs);
    eprintln!("fuzz {}: seed {seed:#x}, {inputs} inputs", target.name);
    run(target, seed, inputs, time, case)
}

/// Runs inputs `0..inputs` of `seed`, or `RVGV_FUZZ_INDEX` alone, until
/// `time` of the wall clock is spent, and fails at the first input that
/// panics, on any thread a process-counted target watches, or passes its
/// CPU time or memory bound. The failure names the seed and the input,
/// and the command that replays it.
#[expect(
    clippy::disallowed_methods,
    reason = "the fuzzer catches an input's panic, to name the input that made it"
)]
pub fn run(
    target: &Target,
    seed: u64,
    inputs: u64,
    time: Option<Duration>,
    case: impl Fn(&mut Rng),
) -> Report {
    install_panic_hook();
    let _turn = match target.accounting {
        Accounting::Thread => Turn::Shared(EXCLUSIVE.read()),
        Accounting::Process => Turn::Alone(EXCLUSIVE.write()),
    };
    let indices = match env("RVGV_FUZZ_INDEX") {
        Some(index) => index..index.saturating_add(1),
        None => 0..inputs,
    };
    // Where the abort's handler finds the target's name: a few bytes a run.
    let name: *mut &'static str = Box::leak(Box::new(target.name));
    let clock = StdRuntime;
    let started = clock.now();
    let mut report = Report {
        inputs: 0,
        elapsed: Duration::ZERO,
        slowest: Duration::ZERO,
        most_bytes: 0,
    };
    for index in indices {
        if time.is_some_and(|time| clock.elapsed_since(started) >= time) {
            break;
        }
        CASE_NAME.store(name, Ordering::SeqCst);
        CASE_SEED.store(seed, Ordering::SeqCst);
        CASE_INDEX.store(index, Ordering::SeqCst);
        if target.accounting == Accounting::Process {
            PANICKED.store(false, Ordering::SeqCst);
        }
        let before_thread = reset_thread_peak();
        let before_process = LIVE.load(Ordering::SeqCst);
        PEAK.store(before_process, Ordering::SeqCst);
        let input_started = target.accounting.cpu_time();
        let mut rng = Rng::new(input_seed(seed, target.name, index));
        let outcome = catch_unwind(AssertUnwindSafe(|| case(&mut rng)));
        let took = target.accounting.cpu_time().saturating_sub(input_started);
        let grew = match target.accounting {
            Accounting::Thread => thread_peak()
                .saturating_sub(before_thread)
                .max(0)
                .cast_unsigned(),
            Accounting::Process => PEAK.load(Ordering::SeqCst).saturating_sub(before_process),
        };
        report.inputs += 1;
        report.slowest = report.slowest.max(took);
        report.most_bytes = report.most_bytes.max(grew);
        let failure = match outcome {
            Err(payload) => Some(format!("panicked: {}", panic_message(&*payload))),
            Ok(())
                if target.accounting == Accounting::Process
                    && PANICKED.swap(false, Ordering::SeqCst) =>
            {
                let message = PANIC_MESSAGE.lock().clone();
                Some(format!("panicked on another thread: {message}"))
            }
            Ok(()) if took > target.input_time => Some(format!(
                "took {took:?} of CPU time, past {:?}",
                target.input_time
            )),
            Ok(()) if grew > target.input_bytes => Some(format!(
                "allocated {grew} bytes, past {}",
                target.input_bytes
            )),
            Ok(()) => None,
        };
        assert!(
            failure.is_none(),
            "fuzz {}: input {index} of seed {seed:#x} {}\n\
             replay: RVGV_FUZZ_SEED={seed:#x} RVGV_FUZZ_INDEX={index} \
             cargo test --offline -p {} -- --exact {} --include-ignored",
            target.name,
            failure.as_deref().unwrap_or_default(),
            env!("CARGO_PKG_NAME"),
            std::thread::current().name().unwrap_or("the test"),
        );
    }
    report.elapsed = clock.elapsed_since(started);
    eprintln!(
        "fuzz {}: {} inputs in {:.1?}, slowest {:.1?} of CPU time, most allocated {} KiB",
        target.name,
        report.inputs,
        report.elapsed,
        report.slowest,
        report.most_bytes >> 10
    );
    report
}

/// A panic's message, from its payload.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_owned()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "a panic of another type".to_owned()
    }
}

/// A run's turn among the binary's fuzzers.
enum Turn<'a> {
    Shared(RwLockReadGuard<'a, ()>),
    Alone(RwLockWriteGuard<'a, ()>),
}
