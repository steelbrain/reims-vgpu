//! The trip of each command buffer's completion down to the guest and of the guest's next buffer
//! up to rt, by stage, in the host's log: every [`PERIOD`], while the log names [`TARGET`] at
//! debug, one line of what the period held. A diagnostic: which lines the host writes is free
//! under the contract, and the stamps change nothing the guest or the host observes.
//!
//! - **The line** is `trip at <wall s.ms> over <s> s: <ours> || <rt's> || imports pointer <n>
//!   dma_buf <n> copy <n>`: the wall time (`CLOCK_REALTIME`, seconds since the Unix epoch to the
//!   millisecond), then the device's counts and stage histograms (reims-vgpu-graphics' `trip`,
//!   whose doc names each stage and count, on the device's clock), then rt's
//!   (`reims_vgpu_vulkan::rt_platform::trip`, on rt's host clock), once rt's device is made, then
//!   the imports of the host's memory live at the line, by how rt took them
//!   (`reims_vgpu_vulkan::rt_platform::buffer::live_imports`). A period that counted nothing
//!   writes no line.
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

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use reims_vgpu_graphics::trip;
use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::os::clock::{Clock as _, Deadline};

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
                    let rt = rt.take();
                    let imports = reims_vgpu_vulkan::rt_platform::buffer::live_imports();
                    if let Some(line) = line(ms, over, &ours, rt.as_deref(), imports) {
                        log::debug!(target: TARGET, "{line}");
                    }
                }
            });
        if made.is_ok() {
            reims_vgpu_vulkan::core::trip::set_clock(trip::now_ns);
            trip::turn_on();
        }
    });
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
/// epoch), of the device's `ours`, rt's part `rt` and the live `imports` by pointer, dma-buf and
/// copy; `None` where the device counted nothing.
fn line(
    ms: u64,
    over: Duration,
    ours: &trip::Taken,
    rt: Option<&str>,
    [pointer, dma_buf, copy]: [u64; 3],
) -> Option<String> {
    if ours.counts.iter().all(|&n| n == 0) {
        return None;
    }
    Some(format!(
        "trip at {}.{:03} over {:.3} s: {ours} || {} || imports pointer {pointer} dma_buf \
         {dma_buf} copy {copy}",
        ms / 1_000,
        ms % 1_000,
        over.as_secs_f64(),
        rt.unwrap_or("rt: none")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A period that counted something writes its line, stamped, with rt's part after ours and
    /// the live imports last; one that counted nothing writes none
    #[test]
    fn a_period_that_counted_writes_its_line() {
        let mut ours = trip::Taken {
            stages: Vec::new(),
            counts: vec![0; trip::COUNTS.len()],
            idle: Vec::new(),
        };
        assert_eq!(line(1_500, PERIOD, &ours, Some("rt: x"), [0; 3]), None);
        ours.counts[0] = 2;
        let line = line(1_700_000_000_123, PERIOD, &ours, None, [3, 1, 0]).unwrap();
        assert!(
            line.starts_with("trip at 1700000000.123 over 1.000 s: takes=2 "),
            "{line}"
        );
        assert!(
            line.ends_with(" || rt: none || imports pointer 3 dma_buf 1 copy 0"),
            "{line}"
        );
    }
}
