//! Each guest context's command buffers and render encoders in the host's log, for the lanes'
//! benches: every [`PERIOD`], while the target [`TARGET`] is enabled at debug, one line of what
//! each context made since the line before. WindowServer's context is the first one its session
//! opens (context 1 on the lane's guests), so its rates per frame or per second fall out, and
//! every other app's beside them. The runtime's own counts go to the same target, on the same
//! tick, from the Vulkan platform (`reims-vgpu-vulkan`'s `work_log`). A diagnostic: which lines
//! the host writes is free under the contract.
//!
//! - **Counted**: a command buffer made for a context's `COMMAND_BUFFER` message (its queue gave
//!   one, whatever its walk then does), and a render encoder opened for its `ENCODER_BEGIN_RENDER`.
//! - **Asked for by name**: [`start`] turns counting on only where `REIMS_VGPU_LOG` names the
//!   target itself at debug as the first decoder is made (`reims_vgpu_vulkan::core::logging::names`),
//!   and makes the thread then: a lone level, such as a bug report's `debug`, or a pair of a
//!   target above it (`reims_vgpu=debug`) does not, as for the runtime's counts on the same
//!   target, whose reading costs the draws they count.
//! - **Off**, it costs one relaxed load a count.
//! - **A line** is `graphics work over S s: context C: N command buffers, M render encoders;
//!   ...`, the contexts in their numbers' order, each that made something since the line before.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use reims_vgpu_vulkan::core::platform::Platform;
use reims_vgpu_vulkan::core::platform::os::clock::{Clock as _, Deadline};
use reims_vgpu_vulkan::core::sync::Mutex;

/// The log target of the lines, at debug, which the filter names to start them ([`start`]).
pub(crate) const TARGET: &str = "reims_vgpu::work";

/// How often the thread writes a line.
pub(crate) const PERIOD: Duration = Duration::from_secs(1);

/// What one context made.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Made {
    pub(crate) command_buffers: u64,
    pub(crate) render_encoders: u64,
}

/// The counts since the last line, by context.
#[derive(Default)]
pub(crate) struct Counts {
    by_context: Mutex<BTreeMap<u32, Made>>,
}

impl Counts {
    fn add(&self, context: u32, made: Made) {
        let mut by = self.by_context.lock();
        let at = by.entry(context).or_default();
        at.command_buffers += made.command_buffers;
        at.render_encoders += made.render_encoders;
    }

    /// The line for what was counted since the line before, `over` that time, or None where
    /// nothing was; the counts start again from 0.
    pub(crate) fn take_line(&self, over: Duration) -> Option<String> {
        let by = std::mem::take(&mut *self.by_context.lock());
        if by.is_empty() {
            return None;
        }
        let mut line = format!("graphics work over {:.1} s:", over.as_secs_f64());
        for (i, (context, made)) in by.iter().enumerate() {
            let _ = write!(
                line,
                "{} context {context}: {} command buffers, {} render encoders",
                if i == 0 { "" } else { ";" },
                made.command_buffers,
                made.render_encoders
            );
        }
        Some(line)
    }
}

/// Whether counting is on ([`start`]).
static ON: AtomicBool = AtomicBool::new(false);

/// The process's counts.
fn counts() -> &'static Counts {
    static COUNTS: OnceLock<Counts> = OnceLock::new();
    COUNTS.get_or_init(Counts::default)
}

/// Turns counting on and starts the thread that writes the lines (the module docs), once, where
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
            .name("reims-vgpu-gfx-work".into())
            .spawn(move || {
                let clock = platform.clock();
                let mut since = clock.now_mono();
                loop {
                    clock.sleep_until(Deadline::Mono(since.saturating_add(PERIOD)));
                    let now = clock.now_mono();
                    if let Some(line) = counts().take_line(now.saturating_duration_since(since)) {
                        log::debug!(target: TARGET, "{line}");
                    }
                    since = now;
                }
            });
        if made.is_ok() {
            ON.store(true, Ordering::Relaxed);
        }
    });
}

/// Counts a command buffer made for `context`'s `COMMAND_BUFFER` message.
pub(crate) fn command_buffer(context: Option<u32>) {
    if let Some(context) = context.filter(|_| ON.load(Ordering::Relaxed)) {
        counts().add(
            context,
            Made {
                command_buffers: 1,
                render_encoders: 0,
            },
        );
    }
}

/// Counts a render encoder opened for `context`'s `ENCODER_BEGIN_RENDER`.
pub(crate) fn render_encoder(context: Option<u32>) {
    if let Some(context) = context.filter(|_| ON.load(Ordering::Relaxed)) {
        counts().add(
            context,
            Made {
                command_buffers: 0,
                render_encoders: 1,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What each context made shows in the next line, by context, and the counts start again.
    #[test]
    fn a_contexts_command_buffers_and_encoders_show_in_the_next_line() {
        let counts = Counts::default();
        assert_eq!(counts.take_line(PERIOD), None, "nothing made: no line");
        counts.add(
            7,
            Made {
                command_buffers: 1,
                render_encoders: 0,
            },
        );
        counts.add(
            1,
            Made {
                command_buffers: 2,
                render_encoders: 0,
            },
        );
        counts.add(
            1,
            Made {
                command_buffers: 0,
                render_encoders: 3,
            },
        );
        assert_eq!(
            counts.take_line(PERIOD).as_deref(),
            Some(
                "graphics work over 1.0 s: context 1: 2 command buffers, 3 render encoders; \
                 context 7: 1 command buffers, 0 render encoders"
            )
        );
        assert_eq!(counts.take_line(PERIOD), None, "counted from 0 again");
    }
}
