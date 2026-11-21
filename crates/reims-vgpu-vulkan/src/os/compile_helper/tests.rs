//! A helper's start, caps, seats and end, with this test binary as the helper: each test runs
//! itself again with [`STAND_IN`] naming what the stand-in does ([`stand_in`]). No GPU.

use std::ffi::OsString;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use super::{
    Beside, Footprints, GUEST_RAM, Helper, HelperEnd, HelperRun, Limits, MEMORY_FLOOR, MEMORY_MOST,
    Seats, TIME_MOST, admits, end_with_parent, hold_guest_ram, host, seat_cap,
};
use crate::core::platform::os::Clock;
use crate::os::OsClock;

/// The variable that makes this test binary a stand-in helper, and names what it does.
const STAND_IN: &str = "RVG_COMPILE_HELPER_STAND_IN";

/// The caps of the tests whose helper stays well inside them.
const ROOMY: Limits = Limits {
    memory: 1 << 30,
    time: Duration::from_secs(60),
};

/// How long a stand-in sleeps: past every test's wait, so a helper that ends in a test was ended.
const STAND_IN_SLEEP: Duration = Duration::from_secs(60);

/// The test `test`'s full name, as the harness takes it.
fn full_name(test: &str) -> String {
    format!("os::compile_helper::tests::{test}")
}

/// Where this process is a stand-in, does what [`STAND_IN`] names, and never returns; otherwise
/// returns at once. A stand-in arms its end with its parent first, as the helper's main does.
/// - `echo`: writes back what its stdin carries, then exits with 7.
/// - `grow`: dirties 32 MiB more every 10 ms, up to 2 GiB, then sleeps.
/// - `sleep`: sleeps.
/// - `parent`: starts a `sleep` stand-in of `test`, prints `helper-pid <pid>`, and exits at once,
///   dropping nothing.
fn stand_in(test: &str) {
    let Some(mode) = std::env::var_os(STAND_IN) else {
        return;
    };
    if mode != "parent" {
        end_with_parent().unwrap();
    }
    let mode = mode.to_string_lossy();
    match &*mode {
        "echo" => {
            let mut request = Vec::new();
            io::stdin().read_to_end(&mut request).unwrap();
            let mut stdout = io::stdout();
            stdout.write_all(&request).unwrap();
            stdout.flush().unwrap();
            std::process::exit(7);
        }
        "grow" => {
            let mut held = Vec::new();
            for _ in 0..64 {
                let mut chunk = vec![0_u8; 32 << 20];
                for page in chunk.chunks_mut(4096) {
                    page[0] = 1;
                }
                held.push(chunk);
                std::thread::sleep(Duration::from_millis(10));
            }
            std::thread::sleep(STAND_IN_SLEEP);
            drop(held);
        }
        "sleep" => std::thread::sleep(STAND_IN_SLEEP),
        "parent" => {
            let seats = one_seat("parent");
            let helper =
                Helper::start(run_of(test, "sleep", ROOMY, &seats), drain, |_, _, _| {}).unwrap();
            let pid = pid_of(&helper);
            let mut stdout = io::stdout();
            writeln!(stdout, "helper-pid {pid}").unwrap();
            stdout.flush().unwrap();
            std::process::exit(0);
        }
        other => panic!("no stand-in mode {other}"),
    }
    std::process::exit(0);
}

/// A run of this test binary as a stand-in of `test` in `mode`, on `seats`.
fn run_of(test: &str, mode: &str, limits: Limits, seats: &Seats) -> HelperRun {
    HelperRun {
        program: std::env::current_exe().unwrap(),
        args: [
            "--exact",
            full_name(test).as_str(),
            "--nocapture",
            "--test-threads=1",
        ]
        .map(OsString::from)
        .to_vec(),
        envs: vec![(OsString::from(STAND_IN), OsString::from(mode))],
        limits,
        seats: seats.clone(),
        label: format!("test {test}"),
    }
}

/// `count` seats of this test process's own, named `tag`, beside a room that `room` says.
fn seats_of(
    tag: &str,
    count: usize,
    room: impl Fn() -> Option<u64> + Send + Sync + 'static,
) -> Seats {
    Seats {
        guard: guard_of(std::process::id(), tag),
        count,
        room: Arc::new(room),
    }
}

/// One seat of this test process's own, named `tag`, beside no room said.
fn one_seat(tag: &str) -> Seats {
    seats_of(tag, 1, || None)
}

/// The seats' guard named `tag` of the test process `pid`.
fn guard_of(pid: u32, tag: &str) -> PathBuf {
    std::env::temp_dir().join(format!("rvg-compile-helper-{pid}-{tag}.lock"))
}

/// A talk that says nothing and reads the stand-in's stdout to its end, which the harness writes
/// its own lines to: a pipe closed early would fail them.
fn drain(stdin: ChildStdin, mut stdout: ChildStdout) -> u64 {
    drop(stdin);
    io::copy(&mut stdout, &mut io::sink()).unwrap_or(0)
}

/// The helper's process's id, once it has started.
fn pid_of(helper: &Helper) -> u32 {
    let clock = OsClock::new();
    let start = clock.now_mono();
    loop {
        if let Some(pid) = helper.pid() {
            return pid;
        }
        let waited = clock.now_mono().saturating_duration_since(start);
        assert!(waited < Duration::from_secs(30), "the helper started");
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// A process's exit code, where it ran to its end.
fn code(end: &HelperEnd) -> Option<i32> {
    match end {
        HelperEnd::Ended(status) => status.code(),
        _ => None,
    }
}

/// The talk has the helper's stdin and stdout, and the end and the talk's result reach `on_end`
/// once the process has ended with its own code.
#[test]
fn a_helper_talks_over_its_pipes_and_ends_with_its_code() {
    let test = "a_helper_talks_over_its_pipes_and_ends_with_its_code";
    stand_in(test);
    let seats = one_seat("talk");
    // Past a pipe's buffer, so the talk's write waits for the helper's reads.
    let request: Vec<u8> = (0_u32..1 << 20)
        .map(|i| i.to_le_bytes()[0] ^ i.to_le_bytes()[1])
        .collect();
    let sent = request.clone();
    let (ends, ended) = mpsc::channel();
    let helper = Helper::start(
        run_of(test, "echo", ROOMY, &seats),
        move |mut stdin, mut stdout| {
            stdin.write_all(&sent).unwrap();
            drop(stdin);
            let mut back = Vec::new();
            stdout.read_to_end(&mut back).unwrap();
            back
        },
        move |end, _, said| ends.send((end.clone(), said)).unwrap(),
    )
    .unwrap();
    let end = helper.wait();
    assert_eq!(code(&end), Some(7), "{end:?}");
    let (seen, said) = ended.recv_timeout(Duration::from_secs(30)).unwrap();
    assert_eq!(seen, end);
    // The harness's own lines come first.
    assert!(said.unwrap().ends_with(&request));
    seats.remove_files();
}

/// A helper whose footprint passes its cap at the ceiling, with no room said, is killed over
/// memory, and the end says the footprint read that crossed it, which is the peak of the
/// footprints its reads saw.
#[test]
fn a_helper_past_its_memory_cap_is_killed_over_memory() {
    let test = "a_helper_past_its_memory_cap_is_killed_over_memory";
    stand_in(test);
    let seats = one_seat("memory");
    let limits = Limits {
        memory: 256 << 20,
        time: Duration::from_secs(60),
    };
    let (ends, ended) = mpsc::channel();
    let on_end = move |_: &HelperEnd, seen, _| ends.send(seen).unwrap();
    let helper = Helper::start(run_of(test, "grow", limits, &seats), drain, on_end).unwrap();
    let pid = pid_of(&helper);
    let seen: Footprints = ended.recv_timeout(Duration::from_secs(30)).unwrap();
    match helper.wait() {
        HelperEnd::OverMemory { bytes } => {
            assert!(bytes > limits.memory, "{bytes}");
            assert_eq!(seen.cap, limits.memory, "{seen:?}");
            assert_eq!(seen.peak, bytes, "{seen:?}");
            assert!(seen.reads > 1 && seen.most_growth > 0, "{seen:?}");
            eprintln!(
                "killed at {} MiB, the cap {} MiB, the most growth in a poll {} MiB",
                bytes >> 20,
                limits.memory >> 20,
                seen.most_growth >> 20
            );
        }
        end => panic!("{end:?}"),
    }
    assert!(!host::alive(pid));
    seats.remove_files();
}

/// A helper alone is capped by the room read at its seat where that is below the ceiling, and
/// killed past it over the room, not over memory; with the room above the ceiling, the ceiling
/// holds it, and its kill is over memory.
#[test]
fn a_helper_alone_is_capped_by_the_room_at_its_seat() {
    let test = "a_helper_alone_is_capped_by_the_room_at_its_seat";
    stand_in(test);
    // The stand-in grows to 2 GiB, past the room's cap and short of the ceiling.
    let limits = Limits {
        memory: 4 << 30,
        time: Duration::from_secs(60),
    };
    let room = MEMORY_FLOOR + (64 << 20);
    let seats = seats_of("roomcap", 1, move || Some(room));
    let (ends, ended) = mpsc::channel();
    let on_end = move |_: &HelperEnd, seen, _| ends.send(seen).unwrap();
    let helper = Helper::start(run_of(test, "grow", limits, &seats), drain, on_end).unwrap();
    let pid = pid_of(&helper);
    let seen: Footprints = ended.recv_timeout(Duration::from_secs(30)).unwrap();
    match helper.wait() {
        HelperEnd::OverRoom { bytes, cap } => {
            assert_eq!(cap, room);
            assert!(bytes > cap && bytes < limits.memory, "{bytes}");
            assert_eq!((seen.cap, seen.peak), (cap, bytes), "{seen:?}");
        }
        end => panic!("{end:?}"),
    }
    assert!(!host::alive(pid));
    seats.remove_files();
    let small = Limits {
        memory: 256 << 20,
        time: Duration::from_secs(60),
    };
    let roomy = seats_of("roomcap-ceiling", 2, || Some(1 << 40));
    let helper = Helper::start(run_of(test, "grow", small, &roomy), drain, |_, _, _| {}).unwrap();
    let end = helper.wait();
    assert!(
        matches!(end, HelperEnd::OverMemory { bytes } if bytes > small.memory),
        "{end:?}"
    );
    roomy.remove_files();
}

/// A helper still running at its time cap is killed there.
#[test]
fn a_helper_past_its_time_cap_is_killed_over_time() {
    let test = "a_helper_past_its_time_cap_is_killed_over_time";
    stand_in(test);
    let seats = one_seat("time");
    let limits = Limits {
        memory: ROOMY.memory,
        time: Duration::from_secs(1),
    };
    let clock = OsClock::new();
    let start = clock.now_mono();
    let helper = Helper::start(run_of(test, "sleep", limits, &seats), drain, |_, _, _| {}).unwrap();
    let pid = pid_of(&helper);
    assert_eq!(helper.wait(), HelperEnd::OverTime);
    let took = clock.now_mono().saturating_duration_since(start);
    assert!(took >= limits.time, "{took:?}");
    assert!(took < Duration::from_secs(20), "{took:?}");
    assert!(!host::alive(pid));
    seats.remove_files();
}

/// A cancel kills the helper at once, and its process is gone when the end is known.
#[test]
fn a_cancelled_helper_is_killed_at_once() {
    let test = "a_cancelled_helper_is_killed_at_once";
    stand_in(test);
    let seats = one_seat("cancel");
    let helper = Helper::start(run_of(test, "sleep", ROOMY, &seats), drain, |_, _, _| {}).unwrap();
    let pid = pid_of(&helper);
    assert!(host::alive(pid));
    let clock = OsClock::new();
    let start = clock.now_mono();
    helper.cancel();
    assert_eq!(helper.wait(), HelperEnd::Cancelled);
    let took = clock.now_mono().saturating_duration_since(start);
    assert!(took < Duration::from_secs(5), "{took:?}");
    assert!(!host::alive(pid));
    seats.remove_files();
}

/// Where the OS says no room, helpers run one at a time, free seats or not: the second starts only
/// once the first has ended.
#[test]
fn with_no_room_said_helpers_run_one_at_a_time() {
    let test = "with_no_room_said_helpers_run_one_at_a_time";
    stand_in(test);
    let seats = seats_of("turns", 3, || None);
    let first = Helper::start(run_of(test, "sleep", ROOMY, &seats), drain, |_, _, _| {}).unwrap();
    pid_of(&first);
    let second = Helper::start(run_of(test, "sleep", ROOMY, &seats), drain, |_, _, _| {}).unwrap();
    std::thread::sleep(Duration::from_secs(1));
    assert_eq!(second.pid(), None, "the second started beside the first");
    first.cancel();
    assert_eq!(first.wait(), HelperEnd::Cancelled);
    pid_of(&second);
    second.cancel();
    assert_eq!(second.wait(), HelperEnd::Cancelled);
    seats.remove_files();
}

/// A helper cancelled while it waits for a seat never starts.
#[test]
fn a_helper_cancelled_while_it_waits_for_a_seat_never_starts() {
    let test = "a_helper_cancelled_while_it_waits_for_a_seat_never_starts";
    stand_in(test);
    let seats = one_seat("waiting");
    let first = Helper::start(run_of(test, "sleep", ROOMY, &seats), drain, |_, _, _| {}).unwrap();
    pid_of(&first);
    let (ends, ended) = mpsc::channel();
    let second = Helper::start(
        run_of(test, "sleep", ROOMY, &seats),
        drain,
        move |end, seen, said| ends.send((end.clone(), seen, said)).unwrap(),
    )
    .unwrap();
    std::thread::sleep(Duration::from_millis(300));
    second.cancel();
    assert_eq!(second.wait(), HelperEnd::Cancelled);
    assert_eq!(second.pid(), None);
    let ends = ended.recv_timeout(Duration::from_secs(30)).unwrap();
    assert_eq!(ends, (HelperEnd::Cancelled, Footprints::default(), None));
    first.cancel();
    assert_eq!(first.wait(), HelperEnd::Cancelled);
    seats.remove_files();
}

/// With room for every cap, as many helpers run side by side as there are seats, and the next
/// waits for a seat.
#[test]
fn with_room_for_their_caps_helpers_run_side_by_side_on_every_seat() {
    let test = "with_room_for_their_caps_helpers_run_side_by_side_on_every_seat";
    stand_in(test);
    let seats = seats_of("side", 3, || Some(1 << 40));
    let start =
        || Helper::start(run_of(test, "sleep", ROOMY, &seats), drain, |_, _, _| {}).unwrap();
    let first = start();
    let others = [start(), start()];
    pid_of(&first);
    others.iter().for_each(|helper| _ = pid_of(helper));
    let fourth = start();
    std::thread::sleep(Duration::from_secs(1));
    assert_eq!(fourth.pid(), None, "a fourth started on three seats");
    first.cancel();
    assert_eq!(first.wait(), HelperEnd::Cancelled);
    pid_of(&fourth);
    for helper in others.iter().chain([&fourth]) {
        helper.cancel();
        assert_eq!(helper.wait(), HelperEnd::Cancelled);
    }
    seats.remove_files();
}

/// Beside running helpers, a helper takes a free seat only where all their caps and its own fit in
/// the room: with room for two caps and a half, two run, and the third waits for one to end.
#[test]
fn helpers_run_side_by_side_only_as_their_caps_fit_the_room() {
    let test = "helpers_run_side_by_side_only_as_their_caps_fit_the_room";
    stand_in(test);
    // The stand-ins hold a few MiB each, far from half a cap.
    let seats = seats_of("room", 3, || Some(ROOMY.memory * 5 / 2));
    let start =
        || Helper::start(run_of(test, "sleep", ROOMY, &seats), drain, |_, _, _| {}).unwrap();
    let first = start();
    let second = start();
    pid_of(&first);
    pid_of(&second);
    let third = start();
    std::thread::sleep(Duration::from_secs(2));
    assert_eq!(third.pid(), None, "a third started past the room");
    first.cancel();
    assert_eq!(first.wait(), HelperEnd::Cancelled);
    pid_of(&third);
    for helper in [&second, &third] {
        helper.cancel();
        assert_eq!(helper.wait(), HelperEnd::Cancelled);
    }
    seats.remove_files();
}

/// A process's helpers take seats in the order they were started: of two that wait, the one
/// started first runs first.
#[test]
fn helpers_take_seats_in_the_order_they_were_started() {
    let test = "helpers_take_seats_in_the_order_they_were_started";
    stand_in(test);
    let seats = one_seat("order");
    let start =
        || Helper::start(run_of(test, "sleep", ROOMY, &seats), drain, |_, _, _| {}).unwrap();
    let first = start();
    pid_of(&first);
    let second = start();
    let third = start();
    std::thread::sleep(Duration::from_millis(300));
    first.cancel();
    assert_eq!(first.wait(), HelperEnd::Cancelled);
    pid_of(&second);
    std::thread::sleep(Duration::from_secs(1));
    assert_eq!(
        third.pid(),
        None,
        "the third took the seat before the second"
    );
    second.cancel();
    assert_eq!(second.wait(), HelperEnd::Cancelled);
    pid_of(&third);
    third.cancel();
    assert_eq!(third.wait(), HelperEnd::Cancelled);
    seats.remove_files();
}

/// A helper takes a seat beside others only where the caps of all, its own among them, fit in the
/// room and what this process's running helpers hold now, which the room no longer counts; with
/// no room said, never beside another.
#[test]
fn a_helper_takes_a_seat_beside_others_only_where_all_their_caps_fit() {
    const GIB: u64 = 1 << 30;
    let cap = 8 * GIB;
    assert!(!admits(1, cap, None, 4 * GIB));
    assert!(admits(1, cap, Some(16 * GIB), 0));
    assert!(!admits(1, cap, Some(16 * GIB - 1), 0));
    assert!(admits(1, cap, Some(12 * GIB), 4 * GIB));
    assert!(!admits(2, cap, Some(20 * GIB), 3 * GIB));
    assert!(admits(2, cap, Some(21 * GIB), 3 * GIB));
    assert!(super::room_of_host().is_some(), "the host says no room");
}

/// Alone, a helper always takes a seat, at a cap no larger than the room read at its seat: the
/// smaller of the ceiling and the room, and no less than the floor (nor more than the ceiling)
/// where the room reads less; at the ceiling where no room is said, and beside others.
#[test]
fn a_helper_alone_always_runs_at_a_cap_no_larger_than_the_room() {
    const GIB: u64 = 1 << 30;
    let ceiling = 8 * GIB;
    assert!(admits(0, ceiling, None, 0));
    assert!(admits(0, ceiling, Some(0), 0));
    let alone = |room| Beside {
        held: 0,
        room,
        ours: 0,
    };
    assert_eq!(seat_cap(ceiling, alone(Some(16 * GIB))), ceiling);
    assert_eq!(seat_cap(ceiling, alone(Some(ceiling))), ceiling);
    assert_eq!(seat_cap(ceiling, alone(Some(5 * GIB))), 5 * GIB);
    assert_eq!(seat_cap(ceiling, alone(Some(GIB + 1))), GIB + 1);
    assert_eq!(MEMORY_FLOOR, GIB);
    assert_eq!(seat_cap(ceiling, alone(Some(GIB / 2))), MEMORY_FLOOR);
    assert_eq!(seat_cap(ceiling, alone(Some(0))), MEMORY_FLOOR);
    assert_eq!(seat_cap(GIB / 2, alone(Some(0))), GIB / 2);
    assert_eq!(seat_cap(ceiling, alone(None)), ceiling);
    let beside = Beside {
        held: 1,
        room: Some(20 * GIB),
        ours: GIB,
    };
    assert_eq!(seat_cap(ceiling, beside), ceiling);
}

/// The guest's RAM is named to the room while its hold lives, and no longer once it drops.
#[test]
fn the_guests_ram_is_named_to_the_room_while_its_hold_lives() {
    let named = |held: &[(u64, u64)]| GUEST_RAM.lock().iter().any(|(_, ranges)| ranges == held);
    let ranges = vec![(0x1000_0000, 1 << 20), (0x2000_0000, 2 << 20)];
    let hold = hold_guest_ram(ranges.clone());
    assert!(named(&ranges));
    drop(hold);
    assert!(!named(&ranges));
}

/// The guest's pages it has not touched are not resident, so the room leaves them to it; those
/// it has touched are.
#[cfg(unix)]
#[test]
fn the_guests_untouched_pages_are_unresident() {
    const LEN: usize = 64 << 20;
    // SAFETY: a new anonymous private mapping of LEN bytes, unmapped below.
    let at = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            LEN,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    assert_ne!(at, libc::MAP_FAILED);
    let len = u64::try_from(LEN).unwrap();
    let range = [(u64::try_from(at.addr()).unwrap(), len)];
    assert_eq!(host::unresident(&range), len);
    // SAFETY: the first half lies in the mapping, which nothing else holds.
    unsafe { std::ptr::write_bytes(at.cast::<u8>(), 1, LEN / 2) };
    // A transparent huge page can take up to 2 MiB more past the touched half.
    let left = host::unresident(&range);
    assert!((len / 2 - (2 << 20)..=len / 2).contains(&left), "{left}");
    // SAFETY: the mapping made above, which nothing refers to past here.
    assert_eq!(unsafe { libc::munmap(at, LEN) }, 0);
}

/// A helper ends with the process that started it, which exits without a word to it: on Linux
/// by `PR_SET_PDEATHSIG`, on macOS by the helper's watch of its parent, on Windows by the Job
/// Object's close.
#[test]
fn a_helper_ends_with_the_process_that_started_it() {
    let test = "a_helper_ends_with_the_process_that_started_it";
    stand_in(test);
    let mut parent = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            full_name(test).as_str(),
            "--nocapture",
            "--test-threads=1",
        ])
        .env(STAND_IN, "parent")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap();
    let mut said = Vec::new();
    let pid: Option<u32> = BufReader::new(parent.stdout.take().unwrap())
        .lines()
        .map_while(Result::ok)
        .find_map(|line| {
            // The harness names the test first, on the same line.
            let pid = line
                .split_once("helper-pid ")
                .and_then(|(_, pid)| pid.trim().parse().ok());
            said.push(line);
            pid
        });
    let status = parent.wait().unwrap();
    let Some(pid) = pid else {
        panic!("the parent named no helper, and ended {status}: {said:?}");
    };
    assert!(status.success(), "{status}");
    Seats {
        guard: guard_of(parent.id(), "parent"),
        count: 1,
        room: Arc::new(|| None),
    }
    .remove_files();
    let clock = OsClock::new();
    let start = clock.now_mono();
    while host::alive(pid) {
        let waited = clock.now_mono().saturating_duration_since(start);
        assert!(
            waited < Duration::from_secs(10),
            "the helper outlived its parent"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// A program that cannot start is said so, and no helper runs.
#[test]
fn a_helper_whose_program_is_missing_never_starts() {
    let seats = one_seat("missing");
    let run = HelperRun {
        program: std::env::temp_dir().join("rvg-no-such-helper"),
        args: Vec::new(),
        envs: Vec::new(),
        limits: ROOMY,
        seats: seats.clone(),
        label: "test missing".to_owned(),
    };
    let helper = Helper::start(run, drain, |_, _, _| {}).unwrap();
    let end = helper.wait();
    assert!(matches!(end, HelperEnd::NotStarted(_)), "{end:?}");
    assert_eq!(helper.pid(), None);
    seats.remove_files();
}

/// The memory cap is the smaller of 8 GiB and half the host's memory, and 8 GiB where the OS does
/// not say; the time cap is 10 minutes.
#[test]
fn the_caps_are_the_smaller_of_8_gib_and_half_the_hosts_memory_and_10_minutes() {
    const GIB: u64 = 1 << 30;
    let memory = |total| Limits::for_host_memory(total).memory;
    assert_eq!(memory(Some(12 * GIB)), 6 * GIB);
    assert_eq!(memory(Some(16 * GIB)), 8 * GIB);
    assert_eq!(memory(Some(64 * GIB)), 8 * GIB);
    assert_eq!(memory(None), MEMORY_MOST);
    assert_eq!(MEMORY_MOST, 8 * GIB);
    assert_eq!(Limits::for_host_memory(None).time, Duration::from_secs(600));
    assert_eq!(TIME_MOST, Duration::from_secs(600));
    let host = Limits::of_host();
    assert!((1..=8 * GIB).contains(&host.memory), "{host:?}");
}

/// The footprints' reads keep the largest footprint, and the largest growth from one read to the
/// next, a fall counting as none.
#[test]
fn footprints_keep_the_peak_and_the_most_growth_between_two_reads() {
    let mut seen = Footprints::default();
    for bytes in [100, 130, 125, 170, 40, 60] {
        seen.read(bytes);
    }
    assert_eq!((seen.peak, seen.most_growth, seen.reads), (170, 45, 6));
    let mut one = Footprints::default();
    one.read(500);
    assert_eq!((one.peak, one.most_growth, one.reads), (500, 0, 1));
}
