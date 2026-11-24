//! The trip of each command buffer's completion down to the guest and of the guest's next buffer
//! up to rt, by stage, in the host's log: every [`PERIOD`], while the log names [`TARGET`] at
//! debug, one line of what the period held. A diagnostic: which lines the host writes is free
//! under the contract, and the stamps change nothing the guest or the host observes.
//!
//! - **The line** is `trip at <wall s.ms> over <s> s: <ours> || <rt's> || <window's> || imports
//!   pointer <n> dma_buf <n> copy <n>`: the wall time (`CLOCK_REALTIME`, seconds since the Unix
//!   epoch to the millisecond), then the device's counts and stage histograms
//!   (reims-vgpu-graphics' `trip`, whose doc names each stage and count, on the device's clock),
//!   then rt's (`reims_vgpu_vulkan::rt_platform::trip`, on rt's host clock), once rt's device is
//!   made, then the windows' own times in the period, as the summary's `window:` part writes its
//!   ten seconds' (`reims_vgpu_vulkan::rt_platform::window_times`, counted apart from the
//!   summary's), then the imports of the host's memory live at the line, by how rt took them
//!   (`reims_vgpu_vulkan::rt_platform::buffer::live_imports`). A period that counted nothing
//!   writes no line.
//! - **rt's metrics** follow each such line, whole: `rt metrics at <wall s.ms>: <rt's Metrics,
//!   {:?}>`, as `rt_platform::trip::Stages` read them, counted since rt's device was made.
//! - **The card's lines** follow, `card at <wall s.ms> dev <ns> n <records>: <record> ...`, each
//!   record a command buffer whose rt handler entered since the lines before
//!   (`reims_vgpu_vulkan::core::trip::CardRecord`), [`CARD_RECORDS`] at most a line
//!   ([`card_lines`]).
//! - **The stamps**: the device's own, at its queues' takes, sleeps and posts; the platform's
//!   (`reims_vgpu_vulkan::core::trip`), as rt's completed handler enters and around each commit's
//!   call into rt, on the device's clock, which [`start`] installs; and the decoder's, as its
//!   completed handler enters and finishes its turn (`decode::cmdbuf`), as a completion's post
//!   runs in its turn (`decode::completions`), as a buffer's walk starts on its thread
//!   ([`super::GraphicsControl`]'s `carry_out`) and around its commit (`decode::cmdbuf`'s
//!   `commit_holding`), with the depth of its queue's buffers committed and not completed.
//! - **Asked for by name**: [`start`] turns the stamps on only where `REIMS_VGPU_LOG` names the
//!   target itself at debug as the first decoder is made
//!   (`reims_vgpu_vulkan::core::logging::names`), as `decode::work` does, and makes its thread
//!   then. Off, each site costs one relaxed load.

use std::fmt::Write as _;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use reims_vgpu_graphics::trip;
use reims_vgpu_vulkan::core::trip::CardRecord;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::os::clock::{Clock as _, Deadline};
use reims_vgpu_vulkan::rt_platform::window_times;

/// The log target of the lines, at debug, which the filter names to start them ([`start`]).
pub(crate) const TARGET: &str = "reims_vgpu::trip";

/// How often the thread writes a line.
pub(crate) const PERIOD: Duration = Duration::from_secs(1);

/// Turns the stamps on and starts the thread that writes the lines (the module docs), once, where
/// the filter names the target at debug and its lines are written, its ticks on `platform`'s
/// clock.
pub(crate) fn start<P: Platform>(platform: &Arc<P>) {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        if !(reims_vgpu_vulkan::core::logging::names(TARGET, log::Level::Debug)
            && log::log_enabled!(target: TARGET, log::Level::Debug))
        {
            return;
        }
        let platform = Arc::clone(platform);
        let made = std::thread::Builder::new()
            .name("reims-vgpu-trip".into())
            .spawn(move || {
                let clock = platform.clock();
                let mut rt = reims_vgpu_vulkan::rt_platform::trip::Stages::default();
                let mut since = clock.now_mono();
                loop {
                    clock.sleep_until(Deadline::Mono(since.saturating_add(PERIOD)));
                    let now = clock.now_mono();
                    let over = now.saturating_duration_since(since);
                    since = now;
                    let ms = clock.now_wall().as_nanos() / 1_000_000;
                    let ours = trip::take();
                    let part = rt.take();
                    let window = window_times::line(&window_times::take_trip());
                    let imports = reims_vgpu_vulkan::rt_platform::buffer::live_imports();
                    if let Some(line) = line(ms, over, &ours, part.as_deref(), &window, imports) {
                        log::debug!(target: TARGET, "{line}");
                        if let Some(metrics) = rt.metrics() {
                            log::debug!(
                                target: TARGET,
                                "rt metrics at {}.{:03}: {metrics}",
                                ms / 1_000,
                                ms % 1_000
                            );
                        }
                    }
                    let cards = reims_vgpu_vulkan::core::trip::take_cards();
                    for line in card_lines(ms, trip::now_ns(), &cards) {
                        log::debug!(target: TARGET, "{line}");
                    }
                }
            });
        if made.is_ok() {
            reims_vgpu_vulkan::core::trip::set_clock(trip::now_ns);
            trip::turn_on();
            window_times::count_trip();
        }
    });
}

/// The card's records a line carries at most ([`card_lines`]).
const CARD_RECORDS: usize = 64;

/// The card's `records`, taken at wall time `ms` (milliseconds since the Unix epoch) and at `now`
/// on the device's clock, in lines of at most [`CARD_RECORDS`]: `card at <wall s.ms> dev <now>
/// n <records>: <record> ...`, each record `start,end,returned,taken,committed,raised,gpu_end,
/// depth` (`CardRecord` and its chain), each time in nanoseconds before `now`, `-` where
/// unknown, and its depth a count.
fn card_lines(ms: u64, now: u64, records: &[CardRecord]) -> Vec<String> {
    let ago = |line: &mut String, t: u64| {
        if t == 0 {
            line.push('-');
        } else {
            let _ = write!(line, "{}", i128::from(now) - i128::from(t));
        }
    };
    let mut lines = Vec::new();
    for chunk in records.chunks(CARD_RECORDS) {
        let mut line = format!(
            "card at {}.{:03} dev {now} n {}:",
            ms / 1_000,
            ms % 1_000,
            chunk.len()
        );
        for record in chunk {
            let chain = record.chain;
            let times = [
                record.start,
                record.end,
                record.returned,
                chain.taken,
                chain.committed,
                chain.raised,
                chain.gpu_end,
            ];
            for (i, t) in times.into_iter().enumerate() {
                line.push(if i == 0 { ' ' } else { ',' });
                ago(&mut line, t);
            }
            let _ = write!(line, ",{}", record.depth);
        }
        lines.push(line);
    }
    lines
}

/// A buffer's walk started at `start`, begun on its queue's thread at `begun` and handed to its
/// lane's thread at `handed` (0 where it is walked on the queue's thread): its begin, its hop and
/// where it is walked.
pub(crate) fn started(start: u64, begun: u64, handed: u64) {
    trip::time(trip::Stage::Begin, start.saturating_sub(begun));
    if handed != 0 {
        trip::count(trip::Count::Lane);
        trip::time(trip::Stage::LaneHop, start.saturating_sub(handed));
    } else {
        trip::count(trip::Count::Inline);
    }
}

/// The line of a period `over` that time, ending at wall time `ms` (milliseconds since the Unix
/// epoch), of the device's `ours`, rt's part `rt`, the windows' part `window` and the live
/// `imports` by pointer, dma-buf and copy; `None` where the device counted nothing.
fn line(
    ms: u64,
    over: Duration,
    ours: &trip::Taken,
    rt: Option<&str>,
    window: &str,
    [pointer, dma_buf, copy]: [u64; 3],
) -> Option<String> {
    if ours.counts.iter().all(|&n| n == 0) {
        return None;
    }
    Some(format!(
        "trip at {}.{:03} over {:.3} s: {ours} || {} || {window} || imports pointer {pointer} \
         dma_buf {dma_buf} copy {copy}",
        ms / 1_000,
        ms % 1_000,
        over.as_secs_f64(),
        rt.unwrap_or("rt: none")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The card's records go in lines of at most [`CARD_RECORDS`], each time before the line's
    /// device time and `-` where unknown
    #[test]
    fn the_card_lines_carry_each_record_before_now() {
        assert!(card_lines(1_000, 5_000, &[]).is_empty());
        let record = CardRecord {
            start: 4_000,
            end: 4_500,
            depth: 2,
            returned: 3_000,
            chain: reims_vgpu_vulkan::core::trip::Chain {
                gpu_end: 0,
                raised: 1_000,
                committed: 2_000,
                taken: 2_500,
            },
        };
        let lines = card_lines(1_700_000_000_123, 5_000, &[record; CARD_RECORDS + 1]);
        assert_eq!(lines.len(), 2);
        let one = "1000,500,2000,2500,3000,4000,-,2";
        let head = format!("card at 1700000000.123 dev 5000 n 64: {one} {one} ");
        assert!(lines[0].starts_with(&head), "{}", lines[0]);
        assert_eq!(
            lines[1],
            format!("card at 1700000000.123 dev 5000 n 1: {one}")
        );
        let line = &card_lines(1_000, 5_000, &[CardRecord::default()])[0];
        assert_eq!(line, "card at 1.000 dev 5000 n 1: -,-,-,-,-,-,-,0");
        assert!(!lines[0].contains("trip at "));
    }

    /// A period that counted something writes its line, stamped, with rt's part after ours, then
    /// the windows' part, and the live imports last; one that counted nothing writes none
    #[test]
    fn a_period_that_counted_writes_its_line() {
        let mut ours = trip::Taken {
            stages: Vec::new(),
            counts: vec![0; trip::COUNTS.len()],
            idle: Vec::new(),
            fine: Vec::new(),
        };
        assert_eq!(
            line(1_500, PERIOD, &ours, Some("rt: x"), "window: w", [0; 3]),
            None
        );
        ours.counts[0] = 2;
        let line = line(
            1_700_000_000_123,
            PERIOD,
            &ours,
            None,
            "window: w",
            [3, 1, 0],
        )
        .unwrap();
        assert!(
            line.starts_with("trip at 1700000000.123 over 1.000 s: takes=2 "),
            "{line}"
        );
        assert!(
            line.ends_with(" || rt: none || window: w || imports pointer 3 dma_buf 1 copy 0"),
            "{line}"
        );
    }
}
