//! The one builder that renders a decline into the always-on log.
//!
//! # Why a builder rather than `format!`
//!
//! Before this existed there were 522 hand-rolled `format!` sites and **fewer
//! than half carried a `reason=` field at all** — the rule `AGENTS.md` states
//! was simply not mechanically reachable. Eleven census modules under
//! `runtime/census/` each formatted their own line shape, sharing no
//! vocabulary, so a grep that worked on one told you nothing about the others.
//!
//! [`Emit`] closes that by construction: the only way to start a line is to
//! hand it something implementing [`Decline`], so the `reason=<slug>` is not a
//! convention an author may forget — it is the constructor argument.
//!
//! # Line shape
//!
//! ```text
//! <event> reason=<slug> k=v k=v …
//! ```
//!
//! `event` is the greppable class prefix that already exists in the log
//! (`export_present`, `vk_engine_draw`, …); `reason` is the specific check.
//! Both matter: the event says which subsystem, the reason says which check.
//!
//! # Where it goes
//!
//! - [`Emit::fail`] → `/tmp/reims-vgpu-fail.log` as a `FAIL` line. Genuine failures.
//! - [`Emit::off`] → the same file as an `OFF` line. Always-on census and
//!   degradation notices that are not, in themselves, failures.
//!
//! Neither is gated behind `REIMS_VGPU_DRAW_LOG`; a decline logged only through
//! [`super::line`] is invisible on a normal boot and does not satisfy I2.

use super::decline::{Decline, Refusal};
use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};

/// A single always-on log line, under construction.
///
/// Built from a [`Decline`], so a line without a reason slug is not
/// representable.
#[must_use = "an Emit that is never sent to fail() or off() logs nothing, \
              which is the silent failure this type exists to prevent"]
pub struct Emit {
    event: &'static str,
    reason: &'static str,
    fields: Vec<(&'static str, String)>,
}

impl Emit {
    /// Start a line for `event`, taking the reason and its load-bearing fields
    /// from `decline`.
    pub fn decline(event: &'static str, decline: &dyn Decline) -> Self {
        Self {
            event,
            reason: decline.slug(),
            fields: decline.fields(),
        }
    }

    /// Start a line for a status-shaped value, or `None` when the status is
    /// control flow rather than a refusal.
    ///
    /// The `Option` is the point: a caller cannot log an `Ok` or a "not ready
    /// yet" by accident, because there is no line to send. The idiom is
    ///
    /// ```ignore
    /// if let Some(e) = Emit::refusal("blit", &status) {
    ///     e.field("dst", dst_ref).fail();
    /// }
    /// ```
    ///
    /// which reads as "log it if it refused" and cannot be written the other way
    /// round.
    pub fn refusal(event: &'static str, status: &dyn Refusal) -> Option<Self> {
        Some(Self {
            event,
            reason: status.refusal()?,
            fields: status.fields(),
        })
    }

    /// Add one `k=v` pair. Values are rendered with `Display`, so they must not
    /// contain whitespace — the log is parsed by splitting on spaces.
    pub fn field(mut self, key: &'static str, value: impl std::fmt::Display) -> Self {
        self.fields.push((key, value.to_string()));
        self
    }

    /// Render the line without sending it. Exposed for tests and for callers
    /// that need to embed the rendering in a larger line.
    pub fn render(&self) -> String {
        let mut s = String::with_capacity(32 + self.fields.len() * 16);
        s.push_str(self.event);
        s.push_str(" reason=");
        s.push_str(self.reason);
        for (k, v) in &self.fields {
            s.push(' ');
            s.push_str(k);
            s.push('=');
            s.push_str(v);
        }
        s
    }

    /// Send as a genuine failure.
    pub fn fail(self) {
        super::fail(self.render());
    }

    /// Send as a genuine failure, but only the first time this
    /// `(reason, discriminant)` pair is seen this boot.
    ///
    /// For a refusal on a path the guest **re-attempts**: an unclassified
    /// compute opcode arrives once per encode, a malformed render record once
    /// per draw. Logging every one would bury the log and trip the sink's own
    /// flood detector, and the second line carries no information the first did
    /// not — magnitude belongs to a counter, not to repetition.
    ///
    /// `discriminant` separates instances that are genuinely different events:
    /// pass the opcode, the ref, the format. Pass `0` to latch on the reason
    /// alone. The class fires once per distinct value, so a guest cycling
    /// through five unknown opcodes still gets five lines rather than one.
    ///
    /// This is flood-proofing, **not** a way to make a noisy path quiet: if a
    /// refusal fires on a healthy boot at all, that is the bug, and latching it
    /// only hides how often.
    pub fn fail_once(self, discriminant: u64) {
        if first_sight(self.reason, discriminant) {
            super::fail(self.render());
        }
    }

    /// Send as an always-on notice (census, degradation) rather than a failure.
    pub fn off(self) {
        super::off(self.render());
    }
}

/// `true` the first time this `(reason, discriminant)` pair is seen this boot.
///
/// Split out from [`Emit::fail_once`] so the latch — the load-bearing part — is
/// unit-testable without capturing the sink. Unbounded by design: the key space
/// is (registered slug × a wire value), and a guest that walks enough distinct
/// values to matter has a bug the log should be shouting about.
///
/// Public so a caller on a hot path can take the latch *before* building the
/// line. [`Emit::field`] renders eagerly, so a census sited inside a per-span
/// resolver would allocate on every call and keep throwing the result away —
/// a probe paying a cost proportional to the traffic it is measuring. Callers
/// that latch here must then send with [`Emit::fail`], not [`Emit::fail_once`]:
/// this call consumes the latch.
pub fn first_sight(reason: &'static str, discriminant: u64) -> bool {
    SEEN.get_or_init(|| Mutex::new(HashSet::new()))
        .lock()
        .map(|mut s| s.insert((reason, discriminant)))
        .unwrap_or(true)
}

/// Every `(reason, discriminant)` [`first_sight`] has been asked about.
///
// A code span below, not a link: `forget_all_latches` is `#[cfg(test)]`, and
// rustdoc documents no `cfg(test)` item, so a link to it cannot resolve on any
// arm and reads as rot in the intra-doc pass.
/// A file-level static rather than a `fn`-local one so `forget_all_latches`
/// can reach it. Nothing outside this module touches it directly.
static SEEN: OnceLock<Mutex<HashSet<(&'static str, u64)>>> = OnceLock::new();

/// `true` when `state` differs from the last state recorded for `subject` under
/// `reason`. Records it either way, and is `true` on the first sighting.
///
/// [`first_sight`] answers "has this ever happened"; this answers "has this
/// *changed*". They are different questions and the difference matters when a
/// subject is served by one of several rungs over its life and the thing worth
/// reporting is the switch: a first-sighting latch cannot see a switch at all,
/// because it goes quiet after the first one. An undeduped line on a per-bind
/// path floods instead. A transition report is bounded by the number of real
/// changes, which is what makes it cheap enough to leave on.
pub fn state_changed(reason: &'static str, subject: u64, state: u64) -> bool {
    LAST.get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .map(|mut m| m.insert((reason, subject), state) != Some(state))
        .unwrap_or(true)
}

/// The last state [`state_changed`] recorded for each `(reason, subject)`.
///
/// File-level for the same reason as [`SEEN`].
static LAST: OnceLock<Mutex<HashMap<(&'static str, u64), u64>>> = OnceLock::new();

/// Drop everything [`first_sight`] and [`state_changed`] remember.
///
/// # Why this exists
///
/// Both registries are process-global and live for the whole boot, which is
/// exactly right in a device and exactly wrong in a test binary: one process
/// runs the entire suite, so a latch a test claims is still claimed for every
/// test that runs after it. The failure is silent in the worst way — the second
/// test's emitter runs, decides it has already said this, and returns; the test
/// then asserts on a line that was never printed and reports "expected exactly
/// one line, got []" while pointing at code that is working.
///
/// Two tests collide whenever they compute the same discriminant, which for
/// keys built from fixture values (a mapping id and a PFN, a task id, a texture
/// ref) is as easy as picking the same round number. Whether they collide
/// depends on which runs first, and libtest orders by name — so the suite's
/// greenness rested on the alphabet. Moving a large test module changed the
/// ordering and turned one such pair red.
///
/// [`crate::sink::FailCapture::start`] calls this, so any test that
/// captures the sink starts from an empty latch and is order-independent by
/// construction. That is the whole cure: it is not possible to write the bug
/// above in a capturing test, and no fixture needs a value chosen to dodge
/// another test's.
///
/// A test that wants a latch *claimed* — to prove an emitter stays quiet, or
/// that two namespaces do not suppress each other — claims it after `start()`
/// rather than before.
#[cfg(any(test, feature = "test-fixtures"))]
pub(crate) fn forget_all_latches() {
    if let Some(s) = SEEN.get() {
        s.lock().unwrap_or_else(|p| p.into_inner()).clear();
    }
    if let Some(m) = LAST.get() {
        m.lock().unwrap_or_else(|p| p.into_inner()).clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake;
    impl Decline for Fake {
        fn slug(&self) -> &'static str {
            "fake_reason"
        }
        fn fields(&self) -> Vec<(&'static str, String)> {
            vec![("ref", "7".to_string())]
        }
    }

    /// The reason is the constructor argument, so it cannot be omitted, and the
    /// decline's own fields arrive without the caller restating them.
    #[test]
    fn a_line_always_carries_its_reason_and_the_declines_fields() {
        let line = Emit::decline("vk_engine_draw", &Fake).render();
        assert_eq!(line, "vk_engine_draw reason=fake_reason ref=7");
    }

    /// Caller fields append after the decline's own, so the load-bearing values
    /// stay adjacent to the reason that produced them.
    #[test]
    fn caller_fields_append_after_the_declines_own() {
        let line = Emit::decline("blit", &Fake)
            .field("w", 4)
            .field("h", 8)
            .render();
        assert_eq!(line, "blit reason=fake_reason ref=7 w=4 h=8");
    }

    /// The latch keys on the *pair*, so a guest cycling through several unknown
    /// opcodes gets one line each rather than one line total — which is the
    /// difference between flood-proofing and hiding the second failure class.
    #[test]
    fn the_flood_latch_fires_once_per_reason_and_discriminant() {
        assert!(first_sight("latch_test_reason", 0x40));
        assert!(!first_sight("latch_test_reason", 0x40));
        assert!(
            first_sight("latch_test_reason", 0x41),
            "a different discriminant is a different event"
        );
        assert!(
            first_sight("latch_test_other", 0x40),
            "a different reason is a different event"
        );
    }

    /// The property that separates this from [`first_sight`]: a subject that
    /// returns to a state it has already been in must report the switch, since
    /// the switch is the event. A first-sighting latch reports it once and then
    /// never again, which is exactly the blindness this exists to fix.
    #[test]
    fn a_state_that_returns_is_still_a_transition() {
        assert!(state_changed("flip_test", 1, 100), "first sighting reports");
        assert!(!state_changed("flip_test", 1, 100), "same state is quiet");
        assert!(state_changed("flip_test", 1, 200), "a switch reports");
        assert!(
            state_changed("flip_test", 1, 100),
            "switching back is a switch too"
        );
        assert!(
            state_changed("flip_test", 2, 100),
            "a different subject keeps its own state"
        );
        assert!(
            state_changed("flip_other", 1, 100),
            "a different reason keeps its own state"
        );
    }

    /// Arming a capture must hand the test an empty latch, for both registries.
    ///
    /// Without this, a capturing test inherits whatever the tests before it
    /// claimed, and its emitter goes quiet on a key it has never itself seen.
    /// The symptom is `cap.one(..)` finding no line while the code under test
    /// is correct, and whether it happens at all depends on libtest's
    /// name ordering — which is why it stayed hidden until a module rename
    /// reshuffled the suite. Asserted here rather than in `sink.rs` because the
    /// registries live here and a future reader adding a third one needs to
    /// find this test next to them.
    #[test]
    fn arming_a_capture_hands_back_an_unclaimed_latch() {
        assert!(first_sight("capture_reset_reason", 0x99));
        assert!(!first_sight("capture_reset_reason", 0x99));
        assert!(state_changed("capture_reset_flip", 1, 7));
        assert!(!state_changed("capture_reset_flip", 1, 7));

        let _cap = crate::sink::FailCapture::start();

        assert!(
            first_sight("capture_reset_reason", 0x99),
            "the capture must not inherit an earlier test's first-sight claim"
        );
        assert!(
            state_changed("capture_reset_flip", 1, 7),
            "nor an earlier test's recorded state, which suppresses the same way"
        );
    }
}
