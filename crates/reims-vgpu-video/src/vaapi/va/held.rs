//! On Windows, the count of a display's live contexts, the teardowns held while one lives, and
//! the decoders' contexts kept then for a decoder of their shape
//! ([`super::display::Display::release_teardown`], [`super::display::Display::context_let_go`]):
//! the accounting alone, apart from libva and its driver, so that its tests run on every host.

use super::display::Teardown;

/// A display's contexts made and not yet let go, and what was let go while one lived, in the
/// order let go: teardowns held (`R`, a teardown's release) and contexts kept (`S`).
pub(super) struct Held<R, S> {
    live: usize,
    kept: Vec<Kept<R, S>>,
}

/// What a display keeps while a context lives on it.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Kept<R, S> {
    /// A teardown, held: made once no context lives.
    Release(R),
    /// A context let go, kept: taken back by a context of its shape ([`Held::take`]), else
    /// destroyed once no context lives.
    Context(S),
}

/// What [`Held::ask`] and [`Held::let_go`] make of what is let go.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Asked<R, S> {
    /// Kept: another context lives on the display.
    Kept,
    /// Made now: what was kept before it, in the order let go, once no context lives, and it
    /// last.
    Now(Vec<Kept<R, S>>),
}

impl<R, S> Held<R, S> {
    pub(super) const fn new() -> Self {
        Held {
            live: 0,
            kept: Vec::new(),
        }
    }

    /// Counts a context made on the display.
    pub(super) fn context_made(&mut self) {
        self.live += 1;
    }

    /// The teardown `kind`'s `release`, asked for: held while a context lives on the display (a
    /// context's, while another does), else made now, behind what was kept before it once none
    /// does.
    pub(super) fn ask(&mut self, kind: Teardown, release: R) -> Asked<R, S> {
        if kind == Teardown::Context {
            return self.ended(Kept::Release(release));
        }
        self.held(Kept::Release(release))
    }

    /// A context let go that a context of its shape may take back: kept while another context
    /// lives on the display, else made now, as [`Held::ask`] makes a context's teardown.
    pub(super) fn let_go(&mut self, context: S) -> Asked<R, S> {
        self.ended(Kept::Context(context))
    }

    /// The first context kept that `fits`, taken back and counted live.
    pub(super) fn take(&mut self, fits: impl Fn(&S) -> bool) -> Option<S> {
        let at = self
            .kept
            .iter()
            .position(|kept| matches!(kept, Kept::Context(context) if fits(context)))?;
        let Kept::Context(context) = self.kept.remove(at) else {
            unreachable!("the position of a context kept")
        };
        self.live += 1;
        Some(context)
    }

    /// The end of a context, `it`: kept while another lives, else made now behind what was kept.
    fn ended(&mut self, it: Kept<R, S>) -> Asked<R, S> {
        // A context not counted (none is) leaves the count at 0, not below it.
        self.live = self.live.saturating_sub(1);
        self.held(it)
    }

    /// `it`: kept while a context lives, else made now behind what was kept.
    fn held(&mut self, it: Kept<R, S>) -> Asked<R, S> {
        if self.live > 0 {
            self.kept.push(it);
            return Asked::Kept;
        }
        let mut now = std::mem::take(&mut self.kept);
        now.push(it);
        Asked::Now(now)
    }
}

#[cfg(test)]
mod tests {
    use super::{Asked, Held, Kept, Teardown};

    type Counted = Held<u32, &'static str>;

    #[test]
    fn a_teardown_asked_for_while_no_context_lives_is_made_now() {
        let mut held = Counted::new();
        assert_eq!(
            held.ask(Teardown::Surfaces, 1),
            Asked::Now(vec![Kept::Release(1)])
        );
        assert_eq!(
            held.ask(Teardown::Config, 2),
            Asked::Now(vec![Kept::Release(2)])
        );
    }

    #[test]
    fn the_last_contexts_teardown_is_made_now() {
        let mut held = Counted::new();
        held.context_made();
        assert_eq!(
            held.ask(Teardown::Context, 1),
            Asked::Now(vec![Kept::Release(1)])
        );
        assert_eq!(
            held.ask(Teardown::Config, 2),
            Asked::Now(vec![Kept::Release(2)])
        );
    }

    #[test]
    fn a_contexts_teardown_asked_for_while_another_context_lives_is_held() {
        let mut held = Counted::new();
        held.context_made();
        held.context_made();
        assert_eq!(held.ask(Teardown::Context, 1), Asked::Kept);
    }

    #[test]
    fn a_configs_and_a_pools_teardown_asked_for_while_a_context_lives_are_held() {
        let mut held = Counted::new();
        held.context_made();
        held.context_made();
        assert_eq!(held.ask(Teardown::Context, 1), Asked::Kept);
        assert_eq!(held.ask(Teardown::Config, 2), Asked::Kept);
        assert_eq!(held.ask(Teardown::Surfaces, 3), Asked::Kept);
        assert_eq!(
            held.ask(Teardown::Context, 4),
            Asked::Now(vec![
                Kept::Release(1),
                Kept::Release(2),
                Kept::Release(3),
                Kept::Release(4)
            ])
        );
    }

    #[test]
    fn a_teardown_held_while_one_context_lives_is_made_as_it_is_let_go() {
        let mut held = Counted::new();
        held.context_made();
        assert_eq!(held.ask(Teardown::Config, 1), Asked::Kept);
        assert_eq!(
            held.let_go("4k"),
            Asked::Now(vec![Kept::Release(1), Kept::Context("4k")])
        );
    }

    #[test]
    fn the_held_teardowns_are_made_in_the_order_asked_before_the_last_contexts() {
        let mut held = Counted::new();
        held.context_made();
        held.context_made();
        held.context_made();
        assert_eq!(held.ask(Teardown::Context, 1), Asked::Kept);
        assert_eq!(held.ask(Teardown::Context, 2), Asked::Kept);
        assert_eq!(
            held.ask(Teardown::Context, 3),
            Asked::Now(vec![Kept::Release(1), Kept::Release(2), Kept::Release(3)])
        );
        assert_eq!(
            held.ask(Teardown::Config, 4),
            Asked::Now(vec![Kept::Release(4)])
        );
    }

    #[test]
    fn a_context_made_after_a_hold_keeps_it_held_until_it_ends_too() {
        let mut held = Counted::new();
        held.context_made();
        held.context_made();
        assert_eq!(held.ask(Teardown::Context, 1), Asked::Kept);
        held.context_made();
        assert_eq!(held.ask(Teardown::Context, 2), Asked::Kept);
        assert_eq!(
            held.ask(Teardown::Context, 3),
            Asked::Now(vec![Kept::Release(1), Kept::Release(2), Kept::Release(3)])
        );
    }

    #[test]
    fn a_context_teardown_with_none_counted_leaves_the_count_at_none() {
        let mut held = Counted::new();
        assert_eq!(
            held.ask(Teardown::Context, 1),
            Asked::Now(vec![Kept::Release(1)])
        );
        held.context_made();
        assert_eq!(
            held.ask(Teardown::Context, 2),
            Asked::Now(vec![Kept::Release(2)])
        );
    }

    #[test]
    fn a_context_let_go_while_another_lives_is_kept_and_taken_back_by_one_that_fits() {
        let mut held = Counted::new();
        held.context_made();
        held.context_made();
        assert_eq!(held.let_go("4k"), Asked::Kept);
        assert_eq!(held.take(|kept| *kept == "1080p"), None);
        assert_eq!(held.take(|kept| *kept == "4k"), Some("4k"));
        assert_eq!(held.take(|kept| *kept == "4k"), None);
    }

    #[test]
    fn a_context_taken_back_is_counted_live() {
        let mut held = Counted::new();
        held.context_made();
        held.context_made();
        assert_eq!(held.let_go("4k"), Asked::Kept);
        assert_eq!(held.take(|kept| *kept == "4k"), Some("4k"));
        assert_eq!(held.let_go("4k"), Asked::Kept);
        assert_eq!(
            held.ask(Teardown::Context, 1),
            Asked::Now(vec![Kept::Context("4k"), Kept::Release(1)])
        );
    }

    #[test]
    fn the_last_context_let_go_is_destroyed_now_behind_what_was_kept_in_the_order_let_go() {
        let mut held = Counted::new();
        held.context_made();
        held.context_made();
        held.context_made();
        assert_eq!(held.let_go("a"), Asked::Kept);
        assert_eq!(held.ask(Teardown::Context, 1), Asked::Kept);
        assert_eq!(
            held.let_go("b"),
            Asked::Now(vec![
                Kept::Context("a"),
                Kept::Release(1),
                Kept::Context("b")
            ])
        );
        assert_eq!(held.take(|_| true), None);
    }

    #[test]
    fn a_context_let_go_with_none_other_living_is_destroyed_now() {
        let mut held = Counted::new();
        held.context_made();
        assert_eq!(held.let_go("4k"), Asked::Now(vec![Kept::Context("4k")]));
        assert_eq!(held.take(|_| true), None);
    }
}
