//! Always-on **present-path census and draw-side drop proxies**.
//!
//! These do **not** change product behavior. They record compact, fail-visible
//! signals so a log census can name a class without opening screenshots.
//!
//! ## Proxies (always-on `observe::fail` lines)
//!
//! | Proxy | Meaning |
//! | --- | --- |
//! | `stale_online_pending` | A post-ack display IRQ raised with the shared-page ONLINE bit still pending |
//! | `secondary_mrt_drop` | A multi-RT draw degraded to single-RT |
//! | `empty_sample` | A resolved fragment/vertex sample whose payload was all-zero |
//!
//! [`window_publish`] emits one line per window and stays silent while its
//! counters are zero.

use std::sync::Mutex;

use crate::observe;

struct ThrashState {
    /// Dedup for `secondary_mrt_drop`: (reason_code, width, height) already
    /// reported this boot, so a per-draw MRT-secondary drop fires once per
    /// distinct combo, never per frame. Names which build path silently degraded
    /// a multi-RT draw to single-RT — the vibrancy coverage-mask drop that leaves
    /// a later material sample reading zero alpha (transparent tooltip / frosted
    /// pass-through class). Bounded by the small set of
    /// (reason, geometry) combinations a boot produces.
    secondary_mrt_drop_seen: std::collections::BTreeSet<(u8, u32, u32)>,
    /// Latch so the (per-VBL, ~60 Hz) stale-online line fires once per boot.
    ///
    /// A post-**ack** display IRQ raised while the shared-page ONLINE bit (bit2)
    /// was still pending makes the guest re-read it and re-run `process_online`
    /// → `connectionChange` → boot-progress overlay rebuild
    /// — the host-driven strobe source the RE named ("re-signals bit2 every
    /// frame"). This is a first-occurrence alarm, not a rate: the line says the
    /// class happened at all, which is the whole question, and it has never
    /// fired on any recorded boot.
    stale_online_logged: bool,
}

/// Which multi-RT build check refused a secondary attachment on the Vulkan arm.
///
/// The driving case is a vibrancy tile whose slot-1 RG16Float coverage mask
/// cannot be built: a later material draw samples that mask GVA, and if the
/// draw had run without the attachment it would find no rendered resident and
/// read zero alpha — the see-through frosted-material class.
///
/// **This used to degrade the draw to single-RT and execute it**, which is why
/// the class above was reachable. It now refuses the whole draw through
/// [`crate::runtime::draw::DrawPreparationDecline::SecondaryTargetUnbuildable`],
/// so a guest that asks for N render targets never gets 1 without being told.
/// entry of the same colour list by its own slot number, so Metal already
/// rendered what Vulkan was dropping, and the divergence — not a fresh argument
/// about what Vulkan ought to do — is the finding.
///
/// The reasons are still reported through [`note_secondary_mrt_drop`] as well
/// as carried in the refusal, because the census answers a question the decline
/// cannot: which check bails, at what geometry, across a whole boot.
pub use reims_vgpu_core::MrtDrop;

/// Which secondary colour attachment this device could not build, and why.
///
/// A type rather than a `(u32, MrtDrop)` pair because both halves travel
/// together from the producer to the refusal that names them, and the slot is
/// the field a reader needs first.
pub use reims_vgpu_core::SecondaryMrtRefusal;

impl ThrashState {
    const fn new() -> Self {
        Self {
            secondary_mrt_drop_seen: std::collections::BTreeSet::new(),
            stale_online_logged: false,
        }
    }
}

/// Single mutex so unit tests and concurrent presents cannot interleave counters.
static STATE: Mutex<ThrashState> = Mutex::new(ThrashState::new());

/// Always-on visibility for an MRT attachment that cannot be represented.
/// The whole draw is refused; it never degrades to a single attachment.
///
/// `reason` is a stable slug for WHICH build check bailed; deduped on
/// `(reason, w, h)` so it fires once per distinct combination per boot (never per
/// frame). Runs on the render/drain worker (`runtime::draw`), never the QEMU main loop.
pub fn note_secondary_mrt_drop(reason: MrtDrop, width: u32, height: u32) {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if !st
        .secondary_mrt_drop_seen
        .insert((reason.code(), width, height))
    {
        return;
    }
    drop(st);
    observe::Emit::decline("secondary_mrt_drop", &reason)
        .field("geom", format!("{width}x{height}"))
        .fail();
}

/// Test-only isolation: proxy state is process-global, so parallel tests that
/// reset a device (`lib.rs device_reset` → [`reset_for_device`]) or drive
/// product note paths mutate counters and anchors out from under a multi-call
/// sequence assertion. Sequence tests hold the write side for their whole
/// body; product entry points take short scoped read guards in test builds.
/// Compiled out of product builds.
#[cfg(test)]
pub(crate) static TEST_STATE_ISOLATION: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Exclusive proxy-state guard for tests that assert multi-call sequences or
/// exact counter values (also used by other modules' proxy-asserting tests).
/// While holding it, call proxy `note_*`/`reset_for_test` directly only —
/// product paths take [`test_shared`] and would self-deadlock.
#[cfg(test)]
pub(crate) fn test_exclusive() -> std::sync::RwLockWriteGuard<'static, ()> {
    TEST_STATE_ISOLATION
        .write()
        .unwrap_or_else(|e| e.into_inner())
}

/// Shared-side guard for product paths that feed the proxy (capture, present,
/// draw notes). Scoped and never nested — recursive reads on `std` RwLock can
/// deadlock against a queued writer.
#[cfg(test)]
pub(crate) fn test_shared() -> std::sync::RwLockReadGuard<'static, ()> {
    TEST_STATE_ISOLATION
        .read()
        .unwrap_or_else(|e| e.into_inner())
}

/// Clear diagnostic state at a device lifetime boundary.
pub fn reset_for_device() {
    #[cfg(test)]
    let _shared = TEST_STATE_ISOLATION
        .read()
        .unwrap_or_else(|e| e.into_inner());
    reset_state_inner();
}

fn reset_state_inner() {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    *st = ThrashState::new();
}

/// Window-publish outcome: did the captured guest frame actually reach the host
/// window, or was it dropped before the window ever saw it?
///
/// The macOS/MoltenVK publish path drops a captured frame outright when no
/// candidate resident has landed content, so the window keeps showing its
/// previous (or slate) contents. A sustained drop run is the "desktop frozen
/// but the device is alive" class, so it needs a name and a count.
pub mod window_publish {
    use crate::observe;
    use std::sync::atomic::{AtomicU64, Ordering};

    static PUBLISHED: AtomicU64 = AtomicU64::new(0);
    static DROPPED: AtomicU64 = AtomicU64::new(0);
    static WINDOW_START_MS: AtomicU64 = AtomicU64::new(0);

    const WINDOW_MS: u64 = 1000;

    /// Count one publish decision: `published`=the frame was handed to the
    /// window, else it was dropped because no resident carried its content.
    pub fn note(published: bool) {
        if published {
            PUBLISHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DROPPED.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(line) = maybe_line_at(observe::elapsed_ms() as u64) {
            observe::off(line);
        }
    }

    fn maybe_line_at(now: u64) -> Option<String> {
        let start = WINDOW_START_MS.load(Ordering::Relaxed);
        if start == 0 {
            let _ = WINDOW_START_MS.compare_exchange(0, now, Ordering::Relaxed, Ordering::Relaxed);
            return None;
        }
        let dt = now.saturating_sub(start);
        if dt < WINDOW_MS {
            return None;
        }
        if WINDOW_START_MS
            .compare_exchange(start, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {
            return None;
        }
        let published = PUBLISHED.swap(0, Ordering::Relaxed);
        let dropped = DROPPED.swap(0, Ordering::Relaxed);
        if published.saturating_add(dropped) == 0 {
            return None;
        }
        Some(format_line(dt, published, dropped))
    }

    /// Why the host window published nothing for a frame it was asked to show.
    ///
    /// One variant today, and a type rather than a literal because bare
    /// `resident_not_ready` is a question several subsystems answer; the slug
    /// says which one. A second drop cause gets its own variant here.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum WindowPublishDrop {
        /// The engine had no content-ready resident to hand the window.
        ResidentNotReady,
    }

    impl crate::observe::Decline for WindowPublishDrop {
        fn slug(&self) -> &'static str {
            match self {
                Self::ResidentNotReady => "window_publish_resident_not_ready",
            }
        }
    }

    /// One line per active second. `reason=` is present only when frames were
    /// actually dropped, so a grep for the reason slug finds exactly the
    /// windows where the host window went stale.
    fn format_line(dt: u64, published: u64, dropped: u64) -> String {
        if dropped == 0 {
            return format!(
                "window_publish window_ms={dt} published={published} dropped={dropped}"
            );
        }
        observe::Emit::decline("window_publish", &WindowPublishDrop::ResidentNotReady)
            .field("window_ms", dt)
            .field("published", published)
            .field("dropped", dropped)
            .render()
    }

    #[cfg(test)]
    pub(crate) fn reset() {
        for a in [&PUBLISHED, &DROPPED, &WINDOW_START_MS] {
            a.store(0, Ordering::Relaxed);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// A clean window names no reason — a reader greps the slug to find
        /// only the windows where the window actually went stale.
        #[test]
        fn healthy_window_carries_no_reason() {
            let line = format_line(1000, 120, 0);
            assert!(line.contains("published=120 dropped=0"), "{line}");
            assert!(!line.contains("reason="), "{line}");
        }

        /// Any drop names the class, so the silent-freeze case is greppable.
        #[test]
        fn dropped_frames_name_the_reason() {
            let line = format_line(1000, 0, 118);
            assert!(line.contains("dropped=118"), "{line}");
            assert!(
                line.contains("reason=window_publish_resident_not_ready"),
                "{line}"
            );
        }

        /// An idle second emits nothing at all — the proxy must not flood a log
        /// with empty windows while the guest is not presenting.
        #[test]
        fn idle_window_emits_nothing() {
            reset();
            assert_eq!(maybe_line_at(0), None, "first call only arms the window");
            assert_eq!(maybe_line_at(WINDOW_MS + 1), None, "no samples, no line");
        }

        /// The window only closes once WINDOW_MS has elapsed.
        #[test]
        fn line_waits_for_the_full_window() {
            reset();
            assert_eq!(maybe_line_at(10), None);
            PUBLISHED.store(5, Ordering::Relaxed);
            assert_eq!(maybe_line_at(10 + WINDOW_MS - 1), None);
            let line = maybe_line_at(10 + WINDOW_MS).expect("window closed");
            assert!(line.contains("published=5"), "{line}");
        }
    }
}

/// Record that a post-ack display IRQ (`src` = vbl|present) was raised while the
/// shared-page ONLINE bit (bit2) was still pending — meaning the guest will
/// re-dispatch `process_online` → `connectionChange` and re-composite the
/// boot-progress overlay (the host-driven strobe source).
///
/// Emits an always-on line the **first** time per boot and stays silent
/// afterwards: VBL runs ~60 Hz, so a per-call line would flood. Measure-only;
/// never gates the IRQ.
pub fn note_stale_online_pending(src: &str, pending: u32) {
    let mut st = STATE.lock().unwrap_or_else(|e| e.into_inner());
    if !st.stale_online_logged {
        st.stale_online_logged = true;
        drop(st);
        observe::fail(format!(
            "stale_online_pending src={src} pending={pending:#x}"
        ));
    }
}

/// Test-only reset (unit tests). Safe while holding [`test_exclusive`]
/// (unlike [`reset_for_device`], which takes the shared side of the guard).
#[cfg(test)]
pub fn reset_for_test() {
    reset_state_inner();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialize all present_proxy tests — they share global thrash STATE —
    /// and exclude parallel device resets ([`test_exclusive`]).
    fn test_lock() -> std::sync::RwLockWriteGuard<'static, ()> {
        test_exclusive()
    }

    /// `note_stale_online_pending` names the class the first time it happens and
    /// stays silent afterwards: VBL is ~60 Hz, so a per-call line would flood.
    ///
    /// The latch is the whole mechanism, so the log is the only place to assert
    /// it — which is also the only place production ever reads it from.
    #[test]
    fn stale_online_pending_logs_once_per_boot() {
        let _g = test_lock();
        reset_for_test();
        use crate::model::DISPLAY_ONLINE_EVENT_MASK;
        // The fail log does not exist until something logs, and under
        // `cfg(test)` it is per-process — so a delta over this test's own calls
        // is exact.
        let count = || {
            std::fs::read_to_string(observe::fail_log_path())
                .unwrap_or_default()
                .matches("stale_online_pending src=")
                .count()
        };
        let before = count();
        note_stale_online_pending("vbl", DISPLAY_ONLINE_EVENT_MASK);
        let after_first = count();
        assert_eq!(
            after_first,
            before + 1,
            "first stale-online must log the always-on line"
        );
        note_stale_online_pending("vbl", DISPLAY_ONLINE_EVENT_MASK);
        note_stale_online_pending("present", DISPLAY_ONLINE_EVENT_MASK);
        assert_eq!(count(), after_first, "later IRQs must not re-log");
        let log = std::fs::read_to_string(observe::fail_log_path()).expect("fail log");
        assert!(
            log.contains("stale_online_pending src=vbl pending=0x4"),
            "the line must name the source and the pending mask"
        );
    }

    /// secondary_mrt_drop: a silently-degraded multi-RT draw fires an always-on
    /// line naming the reason, deduped on (reason, geometry) so a per-frame drop
    /// reports once per distinct combo (never floods). Distinct reasons and
    /// distinct geometries are independent episodes.
    #[test]
    fn secondary_mrt_drop_dedups_per_reason_and_geometry() {
        let _g = test_lock();
        reset_for_test();
        let path = observe::fail_log_path();
        let count = |needle: &str| {
            std::fs::read_to_string(path)
                .unwrap_or_default()
                .matches(needle)
                .count()
        };
        let l0 = count("secondary_mrt_drop reason=mrt_drop_unknown_format geom=214x54");

        // First drop at a geometry fires once.
        note_secondary_mrt_drop(MrtDrop::UnknownFormat, 214, 54);
        assert_eq!(
            count("secondary_mrt_drop reason=mrt_drop_unknown_format geom=214x54"),
            l0 + 1
        );
        // Same reason+geometry → deduped (no per-frame re-fire).
        note_secondary_mrt_drop(MrtDrop::UnknownFormat, 214, 54);
        note_secondary_mrt_drop(MrtDrop::UnknownFormat, 214, 54);
        assert_eq!(
            count("secondary_mrt_drop reason=mrt_drop_unknown_format geom=214x54"),
            l0 + 1,
            "same reason+geometry must dedup"
        );
        // A different reason at the same geometry is its own episode.
        let n_identity = count("secondary_mrt_drop reason=mrt_drop_no_identity geom=214x54");
        note_secondary_mrt_drop(MrtDrop::NoIdentity, 214, 54);
        assert_eq!(
            count("secondary_mrt_drop reason=mrt_drop_no_identity geom=214x54"),
            n_identity + 1
        );
        // A different geometry, same reason, is its own episode.
        let n_other = count("secondary_mrt_drop reason=mrt_drop_unknown_format geom=100x40");
        note_secondary_mrt_drop(MrtDrop::UnknownFormat, 100, 40);
        assert_eq!(
            count("secondary_mrt_drop reason=mrt_drop_unknown_format geom=100x40"),
            n_other + 1
        );
    }
}
