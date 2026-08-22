//! What the device is inside a host driver call, and for how long.
//!
//! # The failure this exists for
//!
//! A shader compile inside `vkCreateGraphicsPipelines` runs on the drain thread,
//! and the drain thread holds the device lock for the whole call. Nothing else
//! takes that lock: every FIFO, every present, every doorbell the guest rings
//! waits behind it. So a driver that does not come back does not degrade this
//! device, it stops it — and it stops it in the one shape no census can report,
//! because every census line in the crate is emitted at the *end* of a drain
//! tranche that is never going to end.
//!
//! That is not a hypothetical. A macOS 15 guest's CoreAnimation uber fragment
//! shader reaches this device as a 1 MB SPIR-V module whose dispatch loop has
//! ~2 566 predecessors, and NVIDIA's compiler has been observed inside one
//! `vkCreateGraphicsPipelines` on it for over ten minutes. The whole VM was
//! dead. The fail log said nothing at all: `display_vbl` kept ticking from the
//! contended poll and the host window kept redrawing its last frame, so the
//! boot read as *healthy* right up to the moment somebody attached a debugger.
//!
//! # Why the tick comes from the poll
//!
//! The reporter cannot be the thread that is stuck, so it is
//! `reims-vgpu::device::device_poll` — QEMU's display-timer callback, which runs on
//! a different thread and, on the contended path, takes no lock at all. That is
//! the same property `vbl_contended_pulse` relies on to keep the guest's time
//! base alive while the drain is busy, and it is what makes this report arrive
//! during the wedge rather than after it.
//!
//! # Not a deadline the device enforces
//!
//! Nothing here cancels or times out a driver call: Vulkan has no such
//! operation, and a call abandoned mid-flight would leave the driver's state
//! undefined. This makes the wedge **visible and attributable**, which is the
//! difference between a boot that explains itself and one that needs `gdb`. The
//! wedge itself is fixed by not handing a driver a module it cannot compile.
//!
//! # What arms it
//!
//! `reims-vgpu-vulkan::engine::driver_breadcrumb::DriverBreadcrumb`, which
//! already brackets exactly the calls that can end or hang the process. It
//! writes the module to disk for the crash case; this watches the clock for the
//! hang case. One arming, two failure modes.

use std::sync::Mutex;
use std::time::Instant;

/// A driver call outstanding longer than this has already cost the guest more
/// than it can absorb.
///
/// The number is the guest's, not ours: macOS's CoreDisplay asserts on a display
/// pipe that has been unready for **ten seconds** and takes WindowServer down
/// with `SIGABRT` (see `kb/macos-11-rail-wedge.md`, where that abort was
/// observed on a driven session over a blocked pipe). Past this point the guest
/// compositor is entitled to give up, so a call still running is already a lost
/// session however it ends.
pub const DRIVER_CALL_DEADLINE_S: u64 = 10;

/// Once the deadline is crossed, one line per this many seconds.
///
/// The first line says a call is late; the repeats say it is *still* late, which
/// is the only thing that distinguishes a slow compile that eventually landed
/// from one that never did. A minute is slow enough that a wedge left up for an
/// hour costs sixty lines.
pub const DRIVER_CALL_REPORT_PERIOD_S: u64 = 60;

/// One outstanding call, and when it was last reported on.
///
/// Kept as a plain struct with a pure [`Self::tick`] so the reporting rule is
/// testable without a clock or a log sink — the same shape the drain censuses
/// use, and for the same reason.
#[derive(Debug)]
struct Outstanding {
    what: String,
    started: Instant,
    /// Elapsed seconds at which the next line is due.
    next_report_s: u64,
}

impl Outstanding {
    /// The line this tick owes the reader, if any.
    fn tick(&mut self, elapsed_s: u64) -> Option<String> {
        if elapsed_s < self.next_report_s {
            return None;
        }
        self.next_report_s = elapsed_s.saturating_add(DRIVER_CALL_REPORT_PERIOD_S);
        Some(format!(
            "driver_call reason=driver_call_outstanding what={} elapsed_s={elapsed_s} \
             deadline_s={DRIVER_CALL_DEADLINE_S} (the drain thread holds the device lock \
             inside this call, so no FIFO, present or doorbell can make progress until it \
             returns)",
            self.what
        ))
    }
}

/// The one slot. A single slot rather than a stack because the calls that arm it
/// are leaf driver entry points that do not nest; [`enter`] declining to
/// displace an existing owner is what keeps that assumption from silently
/// becoming false.
static OUTSTANDING: Mutex<Option<Outstanding>> = Mutex::new(None);

fn lock() -> std::sync::MutexGuard<'static, Option<Outstanding>> {
    OUTSTANDING.lock().unwrap_or_else(|e| e.into_inner())
}

/// Start watching a driver call. Returns whether this caller owns the slot.
///
/// A caller that does not own it must not [`leave`]: an inner call clearing an
/// outer one's entry would stop the watch on the very call whose duration
/// covers both, which is the one that matters.
#[must_use]
pub fn enter(what: String) -> bool {
    let mut slot = lock();
    if slot.is_some() {
        return false;
    }
    *slot = Some(Outstanding {
        what,
        started: Instant::now(),
        next_report_s: DRIVER_CALL_DEADLINE_S,
    });
    true
}

/// The call returned. Only the owner from [`enter`] may call this.
pub fn leave() {
    *lock() = None;
}

/// What the device is inside right now, if anything.
///
/// The read side of the slot. The arming is one line inside
/// `reims-vgpu-vulkan::engine::driver_breadcrumb::DriverBreadcrumb::arm`
/// and nothing else observes it, so without this the coupling between the two
/// modules has no test — and a lost arming is invisible exactly the way the
/// wedge it reports is.
#[must_use]
pub fn watching() -> Option<String> {
    lock().as_ref().map(|o| o.what.clone())
}

/// Emit a line if the outstanding call is past its deadline.
///
/// Called from the device poll, which is not the stuck thread. Cheap enough for
/// a display-timer cadence: one uncontended mutex and one `Instant::elapsed`.
pub fn note_tick() {
    let line = {
        let mut slot = lock();
        match slot.as_mut() {
            Some(o) => {
                let elapsed_s = o.started.elapsed().as_secs();
                o.tick(elapsed_s)
            }
            None => None,
        }
    };
    if let Some(line) = line {
        crate::fail(line);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outstanding(what: &str) -> Outstanding {
        Outstanding {
            what: what.to_string(),
            started: Instant::now(),
            next_report_s: DRIVER_CALL_DEADLINE_S,
        }
    }

    /// A call that returns inside the deadline owes the reader nothing. This is
    /// every pipeline compile on a healthy boot, several a second at its
    /// busiest, so a line here would be the flood the sink's own doc forbids.
    #[test]
    fn a_call_inside_the_deadline_says_nothing() {
        let mut o = outstanding("create_graphics_pipelines");
        for elapsed_s in 0..DRIVER_CALL_DEADLINE_S {
            assert_eq!(o.tick(elapsed_s), None, "elapsed_s={elapsed_s}");
        }
    }

    /// Crossing the deadline reports once, and then not again until a whole
    /// period has passed — the poll ticks at the display rate, so a rule that
    /// reported on every tick past the deadline would write ~60 lines a second.
    #[test]
    fn crossing_the_deadline_reports_once_per_period() {
        let mut o = outstanding("create_graphics_pipelines frag_words=261597");
        let first = o
            .tick(DRIVER_CALL_DEADLINE_S)
            .expect("the deadline reports");
        assert!(first.contains("reason=driver_call_outstanding"));
        assert!(first.contains("what=create_graphics_pipelines frag_words=261597"));
        assert!(first.contains(&format!("elapsed_s={DRIVER_CALL_DEADLINE_S}")));

        for elapsed_s in
            DRIVER_CALL_DEADLINE_S + 1..DRIVER_CALL_DEADLINE_S + DRIVER_CALL_REPORT_PERIOD_S
        {
            assert_eq!(o.tick(elapsed_s), None, "elapsed_s={elapsed_s}");
        }
        let second = o
            .tick(DRIVER_CALL_DEADLINE_S + DRIVER_CALL_REPORT_PERIOD_S)
            .expect("the period reports again");
        assert!(second.contains(&format!(
            "elapsed_s={}",
            DRIVER_CALL_DEADLINE_S + DRIVER_CALL_REPORT_PERIOD_S
        )));
    }

    /// A tick that arrives well past the due time schedules the next one from
    /// *now*, not from the due time. The poll's cadence is not guaranteed — the
    /// thread that drives it competes with everything else on the host — and a
    /// rule that added a fixed period to the missed deadline would fire a burst
    /// of back-dated lines to catch up.
    #[test]
    fn a_late_tick_does_not_owe_a_backlog() {
        let mut o = outstanding("create_shader_module");
        assert!(o.tick(600).is_some(), "600 s is past the deadline");
        assert_eq!(o.tick(601), None, "the next line is a period away from 600");
        assert!(o.tick(600 + DRIVER_CALL_REPORT_PERIOD_S).is_some());
    }

    /// The outer call keeps the slot. If these ever nest, the inner one must not
    /// take ownership, because `leave` from the inner would stop the watch while
    /// the outer call is still running — and the outer one is the call whose
    /// duration contains the wedge.
    #[test]
    fn an_inner_call_does_not_displace_the_outer_one() {
        // Serialized against the other global-slot test by running the whole
        // suite with `--test-threads=1`, which this crate requires anyway.
        leave();
        assert!(
            enter("outer".into()),
            "an empty slot admits the first caller"
        );
        assert!(!enter("inner".into()), "a taken slot refuses the second");
        assert_eq!(
            lock().as_ref().map(|o| o.what.clone()),
            Some("outer".to_string()),
            "the slot still names the outer call"
        );
        leave();
        assert!(lock().is_none());
    }
}
