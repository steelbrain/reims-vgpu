//! A helper's start, caps, lock and end, with this test binary as the helper: each test runs
//! itself again with [`STAND_IN`] naming what the stand-in does ([`stand_in`]). No GPU.

use std::ffi::OsString;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use super::{
    Footprints, Helper, HelperEnd, HelperRun, Limits, MEMORY_MOST, TIME_MOST, end_with_parent, host,
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
            let lock = lock_file("parent");
            let helper =
                Helper::start(run_of(test, "sleep", ROOMY, &lock), drain, |_, _, _| {}).unwrap();
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

/// A run of this test binary as a stand-in of `test` in `mode`.
fn run_of(test: &str, mode: &str, limits: Limits, lock: &Path) -> HelperRun {
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
        lock: lock.to_path_buf(),
    }
}

/// A lock file of this test process's own, named `tag`.
fn lock_file(tag: &str) -> PathBuf {
    lock_file_of(std::process::id(), tag)
}

/// The lock file named `tag` of the test process `pid`.
fn lock_file_of(pid: u32, tag: &str) -> PathBuf {
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
    let lock = lock_file("talk");
    // Past a pipe's buffer, so the talk's write waits for the helper's reads.
    let request: Vec<u8> = (0_u32..1 << 20)
        .map(|i| i.to_le_bytes()[0] ^ i.to_le_bytes()[1])
        .collect();
    let sent = request.clone();
    let (ends, ended) = mpsc::channel();
    let helper = Helper::start(
        run_of(test, "echo", ROOMY, &lock),
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
    let _ = std::fs::remove_file(lock);
}

/// A helper whose footprint passes its cap is killed, and the end says the footprint read that
/// crossed it, which is the peak of the footprints its reads saw.
#[test]
fn a_helper_past_its_memory_cap_is_killed_over_memory() {
    let test = "a_helper_past_its_memory_cap_is_killed_over_memory";
    stand_in(test);
    let lock = lock_file("memory");
    let limits = Limits {
        memory: 256 << 20,
        time: Duration::from_secs(60),
    };
    let (ends, ended) = mpsc::channel();
    let on_end = move |_: &HelperEnd, seen, _| ends.send(seen).unwrap();
    let helper = Helper::start(run_of(test, "grow", limits, &lock), drain, on_end).unwrap();
    let pid = pid_of(&helper);
    let seen: Footprints = ended.recv_timeout(Duration::from_secs(30)).unwrap();
    match helper.wait() {
        HelperEnd::OverMemory { bytes } => {
            assert!(bytes > limits.memory, "{bytes}");
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
    let _ = std::fs::remove_file(lock);
}

/// A helper still running at its time cap is killed there.
#[test]
fn a_helper_past_its_time_cap_is_killed_over_time() {
    let test = "a_helper_past_its_time_cap_is_killed_over_time";
    stand_in(test);
    let lock = lock_file("time");
    let limits = Limits {
        memory: ROOMY.memory,
        time: Duration::from_secs(1),
    };
    let clock = OsClock::new();
    let start = clock.now_mono();
    let helper = Helper::start(run_of(test, "sleep", limits, &lock), drain, |_, _, _| {}).unwrap();
    let pid = pid_of(&helper);
    assert_eq!(helper.wait(), HelperEnd::OverTime);
    let took = clock.now_mono().saturating_duration_since(start);
    assert!(took >= limits.time, "{took:?}");
    assert!(took < Duration::from_secs(20), "{took:?}");
    assert!(!host::alive(pid));
    let _ = std::fs::remove_file(lock);
}

/// A cancel kills the helper at once, and its process is gone when the end is known.
#[test]
fn a_cancelled_helper_is_killed_at_once() {
    let test = "a_cancelled_helper_is_killed_at_once";
    stand_in(test);
    let lock = lock_file("cancel");
    let helper = Helper::start(run_of(test, "sleep", ROOMY, &lock), drain, |_, _, _| {}).unwrap();
    let pid = pid_of(&helper);
    assert!(host::alive(pid));
    let clock = OsClock::new();
    let start = clock.now_mono();
    helper.cancel();
    assert_eq!(helper.wait(), HelperEnd::Cancelled);
    let took = clock.now_mono().saturating_duration_since(start);
    assert!(took < Duration::from_secs(5), "{took:?}");
    assert!(!host::alive(pid));
    let _ = std::fs::remove_file(lock);
}

/// Helpers on one lock file run one at a time: the second starts only once the first has ended.
#[test]
fn helpers_on_one_lock_run_one_at_a_time() {
    let test = "helpers_on_one_lock_run_one_at_a_time";
    stand_in(test);
    let lock = lock_file("turns");
    let first = Helper::start(run_of(test, "sleep", ROOMY, &lock), drain, |_, _, _| {}).unwrap();
    pid_of(&first);
    let second = Helper::start(run_of(test, "sleep", ROOMY, &lock), drain, |_, _, _| {}).unwrap();
    std::thread::sleep(Duration::from_secs(1));
    assert_eq!(second.pid(), None, "the second started beside the first");
    first.cancel();
    assert_eq!(first.wait(), HelperEnd::Cancelled);
    pid_of(&second);
    second.cancel();
    assert_eq!(second.wait(), HelperEnd::Cancelled);
    let _ = std::fs::remove_file(lock);
}

/// A helper cancelled while it waits for the lock never starts.
#[test]
fn a_helper_cancelled_while_it_waits_for_the_lock_never_starts() {
    let test = "a_helper_cancelled_while_it_waits_for_the_lock_never_starts";
    stand_in(test);
    let lock = lock_file("waiting");
    let first = Helper::start(run_of(test, "sleep", ROOMY, &lock), drain, |_, _, _| {}).unwrap();
    pid_of(&first);
    let (ends, ended) = mpsc::channel();
    let second = Helper::start(
        run_of(test, "sleep", ROOMY, &lock),
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
    let _ = std::fs::remove_file(lock);
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
    let _ = std::fs::remove_file(lock_file_of(parent.id(), "parent"));
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
    let lock = lock_file("missing");
    let run = HelperRun {
        program: std::env::temp_dir().join("rvg-no-such-helper"),
        args: Vec::new(),
        envs: Vec::new(),
        limits: ROOMY,
        lock: lock.clone(),
    };
    let helper = Helper::start(run, drain, |_, _, _| {}).unwrap();
    let end = helper.wait();
    assert!(matches!(end, HelperEnd::NotStarted(_)), "{end:?}");
    assert_eq!(helper.pid(), None);
    let _ = std::fs::remove_file(lock);
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
