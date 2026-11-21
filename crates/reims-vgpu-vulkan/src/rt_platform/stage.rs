//! A stage's fill on the rt platform: a helper process that compiles it ([`fills`]).
//!
//! rt refuses, in the VM's process, a kernel's `VkPipeline` whose module weighs past its stage
//! bound and whose compile its cache directory does not hold (`Error::StageUnfilled`), naming the
//! pipeline by its request ([`StageKey`], [`stage_of`]), and the device's fills ([`StageFills`])
//! start one fill a request. Each fill here is a helper process (`os::compile_helper`), the
//! program [`HELPER`] beside the process's executable ([`helper_run`]):
//! - **The talk** is rt's `Device::fill_stage` over the helper's stdin and stdout, which keeps the
//!   compile's cache data in the cache directory for the pipeline's later makes, and answers the
//!   helper's own times (`StageSpans`): its device's creation and its compile, logged with the
//!   talk's wall time and the module's weight (`StageRequest::weight`).
//! - **The fill's end** ([`fill_end`]) is the talk's answer first: data kept fills the stage, and
//!   a compile the driver failed is a breach rt kept. Otherwise it is how the helper ended: killed
//!   at its memory or time cap, a breach recorded with rt (`Device::record_stage_breach`), so the
//!   pipeline's makes refuse it (`Gap::StageSize`) with no helper asked again; cancelled; or
//!   failed, with nothing recorded (no helper, one of another build's wire, one that found no
//!   device of the request's or enabled other features, a cache directory that took no entry,
//!   a helper that ended before its reply).
//! - A cancel of the fill kills the helper.
//!
//! Where there is no helper to run, no fill starts, and the stage's fill ends
//! [`FillEnd::Failed`] at once: nothing is recorded, and a make of the stage refuses as before.
//!
//! **The helper's side** is [`helper_main`], the main of the program `reims-vgpu-compile`
//! (`src/bin/reims-vgpu-compile.rs`), which the dist tree and the installs put beside the
//! library's host (QEMU). It opens its device on the card the VM's device is on, which its one
//! argument names by `deviceUUID`, with the options the VM's device was opened with
//! (`super::device::shared_options`): rt takes its compile only from a device that enabled what
//! the VM's did.
//!
//! **Each helper's footprint** is logged as it ends, with its fill's key ([`footprints_line`]):
//! the largest its reads saw, the largest growth from one read to the next, and how far past the
//! cap it got, which a run reads the cap's margin from.
//!
//! Each fill's start is logged with the site that asked for it ([`Site`]), and rt's refusals are
//! counted by their site, each under a code of its own, apart from the refusals the calls end
//! with: a state's make refused ([`UNFILLED_AT_MAKE`]), and a dispatch refused
//! ([`UNFILLED_AT_DISPATCH`]) whether or not it is made after its fill. The make ahead's refusals
//! are not counted: what the guest meets is its dispatch's. A dispatch its stage's fill skips, one
//! that failed or breached, is counted under its end's code ([`SKIPPED_FAILED`],
//! [`SKIPPED_BREACHED`], [`skipped`]), and its command buffer fails with `Internal` (1)
//! (`super::compute`).

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::io::Write as _;
use std::process::{ChildStdin, ChildStdout, ExitCode};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use ash::vk;
use metal2vulkan_runtime as rt;

use crate::core::platform::os::Clock;
use crate::core::platform::stage::{Cancel, FillDone, FillEnd, Site, StageFills, Start};
use crate::os::OsClock;
use crate::os::compile_helper::{Footprints, Helper, HelperEnd, HelperRun, Limits, Seats};

/// The helper's program, beside the process's executable, with the OS's executable suffix.
pub(crate) const HELPER: &str = "reims-vgpu-compile";

/// The guard of the host's helpers' seats ([`Seats::guard`]), beside which the seats' lock files
/// lie, in the library's cache directory (`crate::os::cache_directory`).
const LOCK: &str = "compile-helper.lock";

/// What a stage's refusal names, and a device's fills are keyed by: rt's request of the
/// `VkPipeline`, equal for every refusal of one pipeline.
pub type StageKey = rt::StageRequest;

/// The degradation code of a pipeline state's make that rt refused for a stage whose compile is
/// not in its store (`Error::StageUnfilled`): the make fails, and the stage's fill starts.
pub(crate) const UNFILLED_AT_MAKE: &str = "stage-unfilled-make";

/// The degradation code of a dispatch that rt refused so: it waits for its stage's fill, and is
/// made after a fill that filled the store, else refused.
pub(crate) const UNFILLED_AT_DISPATCH: &str = "stage-unfilled-dispatch";

/// The degradation code of a dispatch skipped after its stage's fill failed (nothing recorded, the
/// stage not filled): its command buffer fails with `Internal` (1) ([`skipped`]).
pub(crate) const SKIPPED_FAILED: &str = "stage-skipped-failed";

/// The degradation code of a dispatch skipped after its stage's fill breached (the breach
/// recorded): its command buffer fails with `Internal` (1) ([`skipped`]).
pub(crate) const SKIPPED_BREACHED: &str = "stage-skipped-breached";

/// Whether a dispatch at `sel` whose stage's fill ended `end` is skipped: where the fill failed
/// or breached, counted under [`SKIPPED_FAILED`] or [`SKIPPED_BREACHED`], the process's first of
/// each code writing its line on stderr. A dispatch skipped so encodes nothing, and its command
/// buffer fails with `Internal` (1) (`super::compute`).
pub(crate) fn skipped(sel: &'static str, end: FillEnd) -> bool {
    static SAID_FAILED: AtomicBool = AtomicBool::new(false);
    static SAID_BREACHED: AtomicBool = AtomicBool::new(false);
    let (code, said, how) = match end {
        FillEnd::Failed => (SKIPPED_FAILED, &SAID_FAILED, "failed"),
        FillEnd::Breached => (SKIPPED_BREACHED, &SAID_BREACHED, "breached"),
        FillEnd::Filled | FillEnd::Cancelled => return false,
    };
    crate::answer::count_kind(code);
    if !said.swap(true, Ordering::Relaxed) {
        super::unbuilt::say(&format!(
            "reims-vgpu: {sel}: skipped, its stage's fill {how}: its command buffer fails with \
             Internal (1) ({code}), the first of its kind\n"
        ));
    }
    true
}

/// The request whose fill rt's refusal `error` waits for, where it is such a refusal
/// (`Error::StageUnfilled`).
pub(crate) fn stage_of(error: &rt::Error) -> Option<StageKey> {
    match error {
        rt::Error::StageUnfilled(request) => Some(request.clone()),
        _ => None,
    }
}

/// `stage`, the stage a refusal at the site counted under `code` names ([`stage_of`]), counted
/// under `code` where there is one.
pub(crate) fn counted<K>(code: &'static str, stage: Option<K>) -> Option<K> {
    if stage.is_some() {
        crate::answer::count_kind(code);
    }
    stage
}

/// The request whose fill a dispatch's refusal `error` waits for ([`stage_of`]), counted under
/// [`UNFILLED_AT_DISPATCH`] where there is one.
pub(crate) fn unfilled_at_dispatch(error: &rt::Error) -> Option<StageKey> {
    counted(UNFILLED_AT_DISPATCH, stage_of(error))
}

/// The fills of `device`, whose physical device's `deviceUUID` is `uuid`: each runs the helper
/// ([`helper_run`]), hands its pipes to rt's `fill_stage`, and records the breach of a helper
/// killed at a cap (the module's docs).
pub(crate) fn fills(device: rt::Device, uuid: [u8; 16]) -> StageFills<StageKey> {
    let talks = device.clone();
    StageFills::new(stage_start(
        move || helper_run(uuid),
        move |request: &StageKey, to_helper, from_helper| {
            let clock = OsClock::new();
            let began = clock.now_mono();
            let filled = talks.fill_stage(request, to_helper, from_helper);
            if let Ok(spans) = &filled {
                let took = clock.now_mono().saturating_duration_since(began);
                log::info!(
                    target: super::TARGET,
                    "stage {request:?}: filled by its helper in {} ms: weight {}, its device {} \
                     ms, its compile {} ms",
                    took.as_millis(),
                    request.weight(),
                    spans.device_nanos / 1_000_000,
                    spans.compile_nanos / 1_000_000
                );
            }
            filled
        },
        move |request, breach| device.record_stage_breach(request, breach),
    ))
}

/// What runs the helper of a device whose `deviceUUID` is `uuid`: the program [`HELPER`] beside
/// the process's executable, given the UUID in hex, at the host's caps, taking one of the host's
/// seats, whose guard is [`LOCK`] in the library's cache directory, which it makes where it is
/// missing.
///
/// Err: why there is no helper to run, as said.
fn helper_run(uuid: [u8; 16]) -> Result<HelperRun, String> {
    let exe = std::env::current_exe().map_err(|e| format!("the process's executable: {e}"))?;
    let name = format!("{HELPER}{}", std::env::consts::EXE_SUFFIX);
    let program = exe.with_file_name(name);
    if !program.is_file() {
        return Err(format!("{} is not there", program.display()));
    }
    let dir = crate::os::cache_directory().ok_or("the host names no cache directory")?;
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    Ok(HelperRun {
        program,
        args: vec![OsString::from(hex(&uuid))],
        envs: Vec::new(),
        limits: Limits::of_host(),
        seats: Seats::of_host(dir.join(LOCK)),
        label: String::new(),
    })
}

/// `uuid` in lowercase hex, as the helper's argument names its device's `deviceUUID`.
fn hex(uuid: &[u8; 16]) -> String {
    uuid.iter().map(|byte| format!("{byte:02x}")).collect()
}

/// The `deviceUUID` the helper's argument `arg` names in hex ([`hex`]).
///
/// Err: why `arg` names none.
fn uuid_of(arg: Option<&OsStr>) -> Result<[u8; 16], String> {
    let text = arg.ok_or("it has no argument naming a deviceUUID")?;
    let text = (text.to_str())
        .filter(|text| text.len() == 32 && text.bytes().all(|b| b.is_ascii_hexdigit()))
        .ok_or_else(|| format!("its argument {text:?} names no deviceUUID in hex"))?;
    let mut uuid = [0_u8; 16];
    for (byte, at) in uuid.iter_mut().zip((0..32).step_by(2)) {
        let pair = text.get(at..at + 2).unwrap_or_default();
        *byte = u8::from_str_radix(pair, 16).map_err(|e| format!("its argument {text}: {e}"))?;
    }
    Ok(uuid)
}

/// The `deviceUUID` of `physical`, one of `instance`'s devices.
fn device_uuid(instance: &rt::Instance, physical: &rt::PhysicalDevice) -> [u8; 16] {
    let (_, raw) = instance.raw();
    let mut id = vk::PhysicalDeviceIDProperties::default();
    let mut asked = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
    // SAFETY: the instance listed the device and is alive; rt's instance is at Vulkan 1.3, where
    // the structure is core.
    unsafe { raw.get_physical_device_properties2(physical.raw(), &mut asked) };
    id.device_uuid
}

/// The helper process's main, the program [`HELPER`]'s (the module's docs, "The helper's
/// side"): arms its end with the library's process where the OS needs the helper's part
/// (`os::compile_helper::end_with_parent`), makes rt's instance as the library's device does,
/// finds the device whose `deviceUUID` its argument names ([`helper_run`]), and serves one fill
/// over its stdin and stdout, rt's `serve_stage`, with the options the library's device was
/// opened with. Every outcome of the compile is in its reply. Its lines go to stderr: its warnings
/// and errors, or what `REIMS_VGPU_LOG` names where it is set (`crate::core::logging`).
///
/// A failure where it served no reply, as its line says; the library's fill then ends failed
/// ([`fill_end`]).
pub fn helper_main() -> ExitCode {
    let hook: crate::core::logging::Hook = Box::new(|level, target, message| {
        let line = crate::core::logging::stderr_line(level, target, message);
        let _ = std::io::stderr().write_all(line.as_bytes());
    });
    let _ = crate::core::logging::set_hook(log::LevelFilter::Warn, hook);
    match serve_helper(std::env::args_os().nth(1).as_deref()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(why) => {
            log::error!(target: super::TARGET, "the stage helper served no fill: {why}");
            ExitCode::FAILURE
        }
    }
}

/// [`helper_main`]'s work, for the device its argument `arg` names.
///
/// Err: why it served no reply.
fn serve_helper(arg: Option<&OsStr>) -> Result<(), String> {
    crate::os::compile_helper::end_with_parent()
        .map_err(|e| format!("its watch of the library's process: {e}"))?;
    let uuid = uuid_of(arg)?;
    let instance = crate::os::rt_instance(&crate::os::rt_instance_extensions())
        .map_err(|e| format!("rt's instance: {e}"))?;
    let listed = (instance.physical_devices()).map_err(|e| format!("rt's devices: {e}"))?;
    let physical = (listed.iter())
        .find(|physical| device_uuid(&instance, physical) == uuid)
        .ok_or_else(|| format!("no device has the deviceUUID {}", hex(&uuid)))?;
    let (options, _) = super::device::shared_options(&instance, physical);
    let mut to_parent = std::io::stdout().lock();
    rt::serve_stage(&instance, options, std::io::stdin().lock(), &mut to_parent)
        .and_then(|()| to_parent.flush())
        .map_err(|e| format!("its pipes: {e}"))
}

/// A start of fills of stages past rt's bound: each runs the helper `run` names
/// ([`helper_start`]), whose talk is `talk`, rt's fill or a test's; its end is [`fill_end`]'s,
/// and a breach it names is recorded with `record`.
fn stage_start<K, S>(
    run: impl Fn() -> Result<HelperRun, String> + Send + Sync + 'static,
    talk: impl Fn(&K, ChildStdin, ChildStdout) -> Result<S, rt::StageFill> + Send + Sync + 'static,
    record: impl Fn(&K, rt::StageBreach) + Send + Sync + 'static,
) -> Start<K>
where
    K: Clone + fmt::Debug + Send + 'static,
    S: Send + 'static,
{
    helper_start(run, talk, move |key: &K, end, said| {
        let (fill, breach) = fill_end(end, said.as_ref());
        if let Some(breach) = breach {
            record(key, breach);
        }
        if fill != FillEnd::Filled {
            let said = match said {
                Some(Err(refused)) => format!("; its fill: {refused}"),
                Some(Ok(_)) | None => String::new(),
            };
            log::warn!(
                target: super::TARGET,
                "stage {key:?}: its helper's fill ended {fill:?}, the helper {end:?}{said}"
            );
        }
        fill
    })
}

/// The end of a stage's fill whose helper ended as `end`, and whose talk answered `said` (none
/// where no talk ran), with the breach to record with rt, if any (the module's docs).
fn fill_end<S>(
    end: &HelperEnd,
    said: Option<&Result<S, rt::StageFill>>,
) -> (FillEnd, Option<rt::StageBreach>) {
    match (said, end) {
        (Some(Ok(_)), _) => (FillEnd::Filled, None),
        // rt kept the breach itself.
        (Some(Err(rt::StageFill::Compile(_))), _) => (FillEnd::Breached, None),
        (_, HelperEnd::OverMemory { .. }) => (FillEnd::Breached, Some(rt::StageBreach::Memory)),
        (_, HelperEnd::OverTime) => (FillEnd::Breached, Some(rt::StageBreach::Time)),
        (_, HelperEnd::Cancelled) => (FillEnd::Cancelled, None),
        (_, HelperEnd::Ended(_) | HelperEnd::NotStarted(_) | HelperEnd::Unwatched(_)) => {
            (FillEnd::Failed, None)
        }
    }
}

/// A start of fills ([`Start`]) that runs each in a helper process: what `run` names, or no fill
/// where it fails, as it says. The helper's talk is `talk`, and the fill's end is what `on_end`
/// makes of the helper's end and the talk's result.
fn helper_start<K, T>(
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
    Box::new(move |key: &K, site: Site, done: FillDone| -> Cancel {
        let mut run = match run() {
            Ok(run) => run,
            Err(why) => {
                log::warn!(
                    target: super::TARGET,
                    "stage {key:?}: no helper process compiles it for {site}: {why}"
                );
                done.end(FillEnd::Failed);
                return Box::new(|| {});
            }
        };
        log::info!(
            target: super::TARGET,
            "stage {key:?}: its helper's fill starts for {site}"
        );
        run.label = format!("stage {key:?}");
        let (talk, on_end) = (Arc::clone(&talk), Arc::clone(&on_end));
        let (talks, ends) = (key.clone(), key.clone());
        let cap = run.limits.memory;
        let started = Helper::start(
            run,
            move |to_helper, from_helper| talk(&talks, to_helper, from_helper),
            move |end, seen, said| {
                if seen.reads > 0 {
                    footprints_line(&ends, end, seen, cap);
                }
                done.end(on_end(&ends, end, said));
            },
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

/// Writes the footprint of the helper of `key`'s fill that ended as `end`, as its reads `seen`
/// it, beside its memory cap `cap` (`os::compile_helper`'s docs, "The memory cap").
fn footprints_line<K: fmt::Debug>(key: &K, end: &HelperEnd, seen: Footprints, cap: u64) {
    let mib = |bytes: u64| bytes >> 20;
    log::info!(
        target: super::TARGET,
        "stage {key:?}: its helper ended {end:?}: its peak footprint {} MiB, its most growth \
         between two reads {} MiB, its cap {} MiB, past the cap by {} MiB, in {} reads",
        mib(seen.peak),
        mib(seen.most_growth),
        mib(cap),
        mib(seen.peak.saturating_sub(cap)),
        seen.reads
    );
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

    /// A refusal of a stage is counted under its site's code, and one of no stage is not
    #[test]
    fn a_stage_s_refusal_is_counted_under_its_site_s_code() {
        let count = |code: &str| {
            (crate::answer::counted_kinds().into_iter())
                .find(|(kind, _)| *kind == code)
                .map_or(0, |(_, n)| n)
        };
        assert_eq!(counted(UNFILLED_AT_DISPATCH, Some(1)), Some(1));
        assert_eq!(counted::<u32>(UNFILLED_AT_MAKE, None), None);
        assert_eq!(unfilled_at_dispatch(&rt::Error::Quiesced), None);
        assert_eq!(count(UNFILLED_AT_DISPATCH), 1);
        assert_eq!(count(UNFILLED_AT_MAKE), 0);
    }

    /// A dispatch whose stage's fill failed or breached is skipped, counted under its end's code;
    /// one after a fill that filled the store or was cancelled is not, and counts nothing.
    #[test]
    fn a_dispatch_is_skipped_after_a_failed_or_breached_fill_counted_by_its_end() {
        let count = |code: &str| {
            (crate::answer::counted_kinds().into_iter())
                .find(|(kind, _)| *kind == code)
                .map_or(0, |(_, n)| n)
        };
        let sel = "dispatchThreads:threadsPerThreadgroup:";
        assert!(!skipped(sel, FillEnd::Filled));
        assert!(!skipped(sel, FillEnd::Cancelled));
        assert_eq!((count(SKIPPED_FAILED), count(SKIPPED_BREACHED)), (0, 0));
        assert!(skipped(sel, FillEnd::Failed));
        assert!(skipped(sel, FillEnd::Breached));
        assert!(skipped(sel, FillEnd::Breached));
        assert_eq!((count(SKIPPED_FAILED), count(SKIPPED_BREACHED)), (1, 2));
    }

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
            seats: seats_of(test),
            label: format!("test {test}"),
        }
    }

    /// The seats of this test process's `test`: one, so its helpers run one at a time.
    fn seats_of(test: &str) -> Seats {
        let pid = std::process::id();
        Seats {
            guard: std::env::temp_dir().join(format!("rvg-stage-{pid}-{test}.lock")),
            count: 1,
            room: Arc::new(|| None),
        }
    }

    /// An exit status of `code`.
    fn exit_status(code: i32) -> std::process::ExitStatus {
        #[cfg(unix)]
        use std::os::unix::process::ExitStatusExt;
        #[cfg(windows)]
        use std::os::windows::process::ExitStatusExt;
        #[cfg(unix)]
        let raw = code << 8;
        #[cfg(windows)]
        let raw = code.cast_unsigned();
        std::process::ExitStatus::from_raw(raw)
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
        let fill = fills.fill(1, Site::Dispatch);
        assert_eq!(fill.end(), Some(FillEnd::Failed));
        assert_eq!(fills.fill(1, Site::Dispatch).end(), Some(FillEnd::Failed));
        assert!(ended.try_recv().is_err());
    }

    /// A helper whose program cannot start has no talk, and its end maps to the fill's end.
    #[test]
    fn a_helper_that_does_not_start_ends_its_fill_as_on_end_says() {
        let test = "a_helper_that_does_not_start_ends_its_fill_as_on_end_says";
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
        let fill = fills.fill(4, Site::Dispatch);
        assert_eq!(fill.wait(), FillEnd::Failed);
        let (key, end, said) = ended.recv_timeout(WAIT).unwrap();
        assert_eq!(key, 4);
        assert!(matches!(end, HelperEnd::NotStarted(_)), "{end:?}");
        assert_eq!(said, None);
        seats_of(test).remove_files();
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
        let fill = fills.fill(9, Site::Dispatch);
        assert_eq!(fill.wait(), FillEnd::Filled);
        let (key, end, said) = ended.recv_timeout(WAIT).unwrap();
        assert_eq!(key, 9);
        assert!(
            matches!(&end, HelperEnd::Ended(status) if status.success()),
            "{end:?}"
        );
        // The harness's own lines come first.
        assert!(said.unwrap().ends_with(b"stage 9"));
        assert_eq!(fills.fill(9, Site::Dispatch).end(), Some(FillEnd::Filled));
        seats_of(test).remove_files();
    }

    /// The talk's answer decides first: data kept fills the stage, and a failed compile is a
    /// breach rt kept. Otherwise a cap's kill is a breach to record, a cancel is a cancel, and
    /// every other end fails the fill with nothing recorded.
    #[test]
    fn a_fill_ends_by_its_talk_first_then_by_its_helper() {
        use rt::{StageBreach, StageFill};
        type Said = Result<(), StageFill>;
        let ok: Said = Ok(());
        let compile: Said = Err(StageFill::Compile(ash::vk::Result::ERROR_DEVICE_LOST));
        let (version, ended): (Said, Said) = (Err(StageFill::Version), Err(StageFill::Ended));
        let over_memory = HelperEnd::OverMemory { bytes: 9 << 30 };
        let exited = || HelperEnd::Ended(exit_status(0));
        let not_started = HelperEnd::NotStarted("no lock".to_owned());
        assert_eq!(fill_end(&exited(), Some(&ok)), (FillEnd::Filled, None));
        assert_eq!(fill_end(&over_memory, Some(&ok)), (FillEnd::Filled, None));
        assert_eq!(
            fill_end(&exited(), Some(&compile)),
            (FillEnd::Breached, None)
        );
        assert_eq!(
            fill_end(&HelperEnd::OverTime, Some(&compile)),
            (FillEnd::Breached, None)
        );
        let memory = (FillEnd::Breached, Some(StageBreach::Memory));
        assert_eq!(fill_end(&over_memory, Some(&ended)), memory);
        assert_eq!(fill_end::<()>(&over_memory, None), memory);
        let time = (FillEnd::Breached, Some(StageBreach::Time));
        assert_eq!(fill_end(&HelperEnd::OverTime, Some(&ended)), time);
        let cancelled = (FillEnd::Cancelled, None);
        assert_eq!(fill_end(&HelperEnd::Cancelled, Some(&ended)), cancelled);
        let failed = (FillEnd::Failed, None);
        assert_eq!(fill_end(&exited(), Some(&version)), failed);
        assert_eq!(fill_end(&exited(), Some(&ended)), failed);
        assert_eq!(fill_end::<()>(&not_started, None), failed);
        let unwatched = HelperEnd::Unwatched("its status".to_owned());
        assert_eq!(fill_end(&unwatched, Some(&ended)), failed);
    }

    /// A helper of another build's wire, which rt's fill answers `StageFill::Version`, fails the
    /// stage's fill with no breach recorded, and the end is kept: no helper is asked again.
    #[test]
    fn a_helper_of_another_wire_version_fails_its_fill_with_nothing_recorded() {
        let test = "a_helper_of_another_wire_version_fails_its_fill_with_nothing_recorded";
        stand_in();
        let (records, recorded) = mpsc::channel();
        let (talks, talked) = mpsc::channel();
        let fills = StageFills::new(stage_start(
            move || Ok(run_of(test, "echo")),
            move |key: &u32, stdin, stdout| {
                talks.send(echo(key, stdin, stdout)).unwrap();
                Err::<(), _>(rt::StageFill::Version)
            },
            move |key, breach| records.send((*key, breach)).unwrap(),
        ));
        assert_eq!(fills.fill(5, Site::Dispatch).wait(), FillEnd::Failed);
        assert!(talked.recv_timeout(WAIT).unwrap().ends_with(b"stage 5"));
        assert_eq!(fills.fill(5, Site::Dispatch).end(), Some(FillEnd::Failed));
        assert!(talked.try_recv().is_err());
        assert!(recorded.try_recv().is_err());
        seats_of(test).remove_files();
    }

    /// A helper killed at its time cap breaches its stage: the breach is recorded, and the fill
    /// ends breached.
    #[test]
    fn a_helper_past_its_time_cap_records_a_time_breach() {
        let test = "a_helper_past_its_time_cap_records_a_time_breach";
        stand_in();
        let (records, recorded) = mpsc::channel();
        let fills = StageFills::new(stage_start(
            move || {
                let mut run = run_of(test, "sleep");
                run.limits.time = Duration::from_millis(300);
                Ok(run)
            },
            |_: &u32, stdin, mut stdout| {
                drop(stdin);
                io::copy(&mut stdout, &mut io::sink()).unwrap_or(0);
                Err::<(), _>(rt::StageFill::Ended)
            },
            move |key, breach| records.send((*key, breach)).unwrap(),
        ));
        assert_eq!(fills.fill(3, Site::Dispatch).wait(), FillEnd::Breached);
        assert_eq!(
            recorded.recv_timeout(WAIT).unwrap(),
            (3, rt::StageBreach::Time)
        );
        seats_of(test).remove_files();
    }

    /// A refusal other than a stage's names no stage.
    #[test]
    fn only_a_stage_s_refusal_names_a_stage() {
        assert!(stage_of(&rt::Error::Quiesced).is_none());
    }

    /// Where no helper lies beside the process's executable, as beside a test binary, there is
    /// none to run, and it says where it looked.
    #[test]
    fn a_process_with_no_helper_beside_it_runs_none() {
        let why = helper_run([7; 16]).unwrap_err();
        assert!(why.contains(HELPER), "{why}");
    }

    /// The helper's argument names its device's `deviceUUID` in hex, and an argument that names
    /// none is refused, said so.
    #[test]
    fn a_device_uuid_goes_to_the_helper_in_hex_and_back() {
        let uuid: [u8; 16] = std::array::from_fn(|i| (i as u8).wrapping_mul(37) ^ 0xa5);
        let named = hex(&uuid);
        assert_eq!(named.len(), 32);
        assert_eq!(uuid_of(Some(OsStr::new(&named))), Ok(uuid));
        assert_eq!(uuid_of(Some(OsStr::new(&named.to_uppercase()))), Ok(uuid));
        let short = &named[..30];
        let signed = format!("+{}", &named[1..]);
        let not_hex = format!("zz{}", &named[2..]);
        for bad in ["", short, signed.as_str(), not_hex.as_str()] {
            let refused = uuid_of(Some(OsStr::new(bad))).unwrap_err();
            assert!(refused.contains("names no deviceUUID"), "{bad}: {refused}");
        }
        assert!(uuid_of(None).unwrap_err().contains("no argument"));
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
        let fill = fills.fill(2, Site::Dispatch);
        assert_eq!(fill.end(), None);
        fills.cancel_all();
        assert_eq!(fill.end(), Some(FillEnd::Cancelled));
        assert_eq!(ended.recv_timeout(WAIT).unwrap(), HelperEnd::Cancelled);
        assert_eq!(fill.end(), Some(FillEnd::Cancelled));
        seats_of(test).remove_files();
    }
}
