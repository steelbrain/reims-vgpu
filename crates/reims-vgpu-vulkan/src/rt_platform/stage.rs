//! A stage's fill on the rt platform: a helper process that compiles it ([`helper_start`]).
//!
//! rt refuses, in the VM's process, a stage whose compile is not in its store and weighs past its
//! bound, and the device's fills ([`StageFills`](crate::core::platform::stage::StageFills)) start
//! one fill a stage. Each fill here is a helper process (`os::compile_helper`): the caller's
//! `talk` is the fill, which rt carries out over the helper's stdin and stdout, and the caller's
//! `on_end` maps how the helper ended to the fill's end, recording a breach with rt where the
//! helper was killed at a cap. A cancel of the fill kills the helper.
//!
//! Where there is no helper to run, no fill starts, and the stage's fill ends
//! [`FillEnd::Failed`] at once: nothing is recorded, and a make of the stage refuses as before.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "its caller, the handler of rt's stage refusal, lands with the helper's binary"
    )
)]

use std::fmt;
use std::process::{ChildStdin, ChildStdout};
use std::sync::Arc;

use crate::core::platform::stage::{Cancel, FillDone, FillEnd, Start};
use crate::os::compile_helper::{Helper, HelperEnd, HelperRun};

/// A start of fills ([`Start`]) that runs each in a helper process: what `run` names, or no fill
/// where it fails, as it says. The helper's talk is `talk`, and the fill's end is what `on_end`
/// makes of the helper's end and the talk's result.
pub(crate) fn helper_start<K, T>(
    run: impl Fn() -> Result<HelperRun, String> + Send + Sync + 'static,
    talk: impl Fn(&K, ChildStdin, ChildStdout) -> T + Send + Sync + 'static,
    on_end: impl Fn(&K, &HelperEnd, Option<T>) -> FillEnd + Send + Sync + 'static,
) -> Start<K>
where
    K: Clone + fmt::Debug + Send + 'static,
    T: Send + 'static,
{
    let talk = Arc::new(talk);
    let on_end = Arc::new(on_end);
    Box::new(move |key: &K, done: FillDone| -> Cancel {
        let run = match run() {
            Ok(run) => run,
            Err(why) => {
                log::warn!(
                    target: super::TARGET,
                    "stage {key:?}: no helper process compiles it: {why}"
                );
                done.end(FillEnd::Failed);
                return Box::new(|| {});
            }
        };
        let (talk, on_end) = (Arc::clone(&talk), Arc::clone(&on_end));
        let (talks, ends) = (key.clone(), key.clone());
        let started = Helper::start(
            run,
            move |to_helper, from_helper| talk(&talks, to_helper, from_helper),
            move |end, said| done.end(on_end(&ends, end, said)),
        );
        match started {
            // Dropped at the fill's end, which comes after the helper's: a cancel then is nothing.
            Ok(helper) => Box::new(move || helper.cancel()),
            // `done` went with the thread's closure, and ended the fill failed.
            Err(error) => {
                log::warn!(
                    target: super::TARGET,
                    "stage {key:?}: the thread of the helper process that compiles it did not \
                     start: {error}"
                );
                Box::new(|| {})
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::io::{self, Read, Write};
    use std::path::PathBuf;
    use std::sync::mpsc;
    use std::time::Duration;

    use super::*;
    use crate::core::platform::stage::StageFills;
    use crate::os::compile_helper::{Limits, end_with_parent};

    /// The variable that makes this test binary a stand-in helper, and names what it does.
    const STAND_IN: &str = "RVG_STAGE_STAND_IN";

    /// How long a test waits for a helper's end.
    const WAIT: Duration = Duration::from_secs(30);

    /// Where this process is a stand-in, does what [`STAND_IN`] names, and never returns; otherwise
    /// returns at once. A stand-in arms its end with its parent first, as the helper's main does.
    /// - `echo`: writes back what its stdin carries, then exits with 0.
    /// - `sleep`: sleeps for a minute, past every test's wait.
    fn stand_in() {
        let Some(mode) = std::env::var_os(STAND_IN) else {
            return;
        };
        end_with_parent().unwrap();
        if mode == "echo" {
            let mut request = Vec::new();
            io::stdin().read_to_end(&mut request).unwrap();
            let mut stdout = io::stdout();
            stdout.write_all(&request).unwrap();
            stdout.flush().unwrap();
        } else {
            std::thread::sleep(Duration::from_secs(60));
        }
        std::process::exit(0);
    }

    /// A run of this test binary as a stand-in of `test` in `mode`, with a lock file of the test's
    /// own.
    fn run_of(test: &str, mode: &str) -> HelperRun {
        HelperRun {
            program: std::env::current_exe().unwrap(),
            args: [
                "--exact",
                format!("rt_platform::stage::tests::{test}").as_str(),
                "--nocapture",
                "--test-threads=1",
            ]
            .map(OsString::from)
            .to_vec(),
            envs: vec![(OsString::from(STAND_IN), OsString::from(mode))],
            limits: Limits {
                memory: 1 << 30,
                time: Duration::from_secs(60),
            },
            lock: lock_file(test),
        }
    }

    /// The lock file of this test process's `test`.
    fn lock_file(test: &str) -> PathBuf {
        let pid = std::process::id();
        std::env::temp_dir().join(format!("rvg-stage-{pid}-{test}.lock"))
    }

    /// A talk that writes the stage's key, then reads what the helper writes to its end.
    fn echo(key: &u32, mut stdin: ChildStdin, mut stdout: ChildStdout) -> Vec<u8> {
        stdin.write_all(format!("stage {key}").as_bytes()).unwrap();
        drop(stdin);
        let mut said = Vec::new();
        stdout.read_to_end(&mut said).unwrap();
        said
    }

    /// Where no helper can run, the fill ends failed at once, with no talk and no end to map.
    #[test]
    fn a_stage_with_no_helper_fails_its_fill() {
        let (ends, ended) = mpsc::channel();
        let fills = StageFills::new(helper_start(
            || Err("no helper here".to_owned()),
            |_: &u32, _, _| panic!("a talk with no helper"),
            move |_, end: &HelperEnd, _: Option<()>| {
                ends.send(end.clone()).unwrap();
                FillEnd::Breached
            },
        ));
        let fill = fills.fill(1);
        assert_eq!(fill.end(), Some(FillEnd::Failed));
        assert_eq!(fills.fill(1).end(), Some(FillEnd::Failed));
        assert!(ended.try_recv().is_err());
    }

    /// A helper whose program cannot start has no talk, and its end maps to the fill's end.
    #[test]
    fn a_helper_that_does_not_start_ends_its_fill_as_on_end_says() {
        let test = "a_helper_that_does_not_start_ends_its_fill_as_on_end_says";
        let lock = lock_file(test);
        let (ends, ended) = mpsc::channel();
        let fills = StageFills::new(helper_start(
            move || {
                let mut run = run_of(test, "echo");
                run.program = PathBuf::from("/nonexistent/rvg-stage-helper");
                Ok(run)
            },
            |_: &u32, _, _| panic!("a talk with no process"),
            move |key, end, said: Option<()>| {
                ends.send((*key, end.clone(), said)).unwrap();
                FillEnd::Failed
            },
        ));
        let fill = fills.fill(4);
        assert_eq!(fill.wait(), FillEnd::Failed);
        let (key, end, said) = ended.recv_timeout(WAIT).unwrap();
        assert_eq!(key, 4);
        assert!(matches!(end, HelperEnd::NotStarted(_)), "{end:?}");
        assert_eq!(said, None);
        let _ = std::fs::remove_file(lock);
    }

    /// The fill's talk runs with the helper's pipes and the stage's key, and what `on_end` makes
    /// of the helper's end and the talk's result is the fill's end, kept for its key.
    #[test]
    fn a_fill_talks_to_its_helper_and_ends_as_on_end_says() {
        let test = "a_fill_talks_to_its_helper_and_ends_as_on_end_says";
        stand_in();
        let (ends, ended) = mpsc::channel();
        let fills = StageFills::new(helper_start(
            move || Ok(run_of(test, "echo")),
            echo,
            move |key, end, said| {
                ends.send((*key, end.clone(), said)).unwrap();
                FillEnd::Filled
            },
        ));
        let fill = fills.fill(9);
        assert_eq!(fill.wait(), FillEnd::Filled);
        let (key, end, said) = ended.recv_timeout(WAIT).unwrap();
        assert_eq!(key, 9);
        assert!(
            matches!(&end, HelperEnd::Ended(status) if status.success()),
            "{end:?}"
        );
        // The harness's own lines come first.
        assert!(said.unwrap().ends_with(b"stage 9"));
        assert_eq!(fills.fill(9).end(), Some(FillEnd::Filled));
        let _ = std::fs::remove_file(lock_file(test));
    }

    /// A cancel of the fills ends the fill at once and kills its helper, whose end then reaches
    /// `on_end` without changing the fill's.
    #[test]
    fn a_cancelled_fill_kills_its_helper() {
        let test = "a_cancelled_fill_kills_its_helper";
        stand_in();
        let (ends, ended) = mpsc::channel();
        let fills = StageFills::new(helper_start(
            move || Ok(run_of(test, "sleep")),
            |_: &u32, stdin, mut stdout| {
                drop(stdin);
                io::copy(&mut stdout, &mut io::sink()).unwrap_or(0)
            },
            move |_, end, _| {
                ends.send(end.clone()).unwrap();
                FillEnd::Breached
            },
        ));
        let fill = fills.fill(2);
        assert_eq!(fill.end(), None);
        fills.cancel_all();
        assert_eq!(fill.end(), Some(FillEnd::Cancelled));
        assert_eq!(ended.recv_timeout(WAIT).unwrap(), HelperEnd::Cancelled);
        assert_eq!(fill.end(), Some(FillEnd::Cancelled));
        let _ = std::fs::remove_file(lock_file(test));
    }
}
