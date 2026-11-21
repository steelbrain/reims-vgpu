//! On Windows, the count of a display's live contexts and the teardowns held while one lives
//! ([`super::display::Display::release_teardown`]): the accounting alone, apart from libva and
//! its driver, so that its tests run on every host.

use super::display::Teardown;

/// A display's contexts made and not yet let go, and the teardowns asked for while one lived and
/// held, in the order asked. `R` is a teardown's release.
pub(super) struct Held<R> {
    live: usize,
    held: Vec<R>,
}

/// What [`Held::ask`] makes of a teardown asked for.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Asked<R> {
    /// Held: another context lives on the display.
    Held,
    /// Made now: the teardowns held before it, in the order asked, and it last.
    Now(Vec<R>),
}

impl<R> Held<R> {
    pub(super) const fn new() -> Self {
        Held {
            live: 0,
            held: Vec::new(),
        }
    }

    /// Counts a context made on the display.
    pub(super) fn context_made(&mut self) {
        self.live += 1;
    }

    /// The teardown `kind`'s `release`, asked for: held while a context other than the one it
    /// ends lives on the display and `kind` is held then ([`held_while_a_context_lives`]), else
    /// made now, behind those held before it once no context lives.
    pub(super) fn ask(&mut self, kind: Teardown, release: R) -> Asked<R> {
        if kind == Teardown::Context {
            // A context not counted (none is) leaves the count at 0, not below it.
            self.live = self.live.saturating_sub(1);
        }
        if self.live > 0 && held_while_a_context_lives(kind) {
            self.held.push(release);
            return Asked::Held;
        }
        let mut now = if self.live == 0 {
            std::mem::take(&mut self.held)
        } else {
            Vec::new()
        };
        now.push(release);
        Asked::Now(now)
    }
}

/// Whether a teardown of `kind` asked for while another context lives on the display is held
/// until none does ([`super::display::Display::release_teardown`] says why).
const fn held_while_a_context_lives(kind: Teardown) -> bool {
    match kind {
        Teardown::Context | Teardown::Config | Teardown::Surfaces => true,
    }
}

#[cfg(test)]
mod tests {
    use super::{Asked, Held, Teardown};

    #[test]
    fn a_teardown_asked_for_while_no_context_lives_is_made_now() {
        let mut held = Held::new();
        assert_eq!(held.ask(Teardown::Surfaces, 1), Asked::Now(vec![1]));
        assert_eq!(held.ask(Teardown::Config, 2), Asked::Now(vec![2]));
    }

    #[test]
    fn the_last_contexts_teardown_is_made_now() {
        let mut held = Held::new();
        held.context_made();
        assert_eq!(held.ask(Teardown::Context, 1), Asked::Now(vec![1]));
        assert_eq!(held.ask(Teardown::Config, 2), Asked::Now(vec![2]));
    }

    #[test]
    fn every_teardown_asked_for_while_another_context_lives_is_held() {
        let mut held = Held::new();
        held.context_made();
        held.context_made();
        assert_eq!(held.ask(Teardown::Context, 1), Asked::Held);
        assert_eq!(held.ask(Teardown::Config, 2), Asked::Held);
        assert_eq!(held.ask(Teardown::Surfaces, 3), Asked::Held);
    }

    #[test]
    fn the_held_teardowns_are_made_in_the_order_asked_before_the_last_contexts() {
        let mut held = Held::new();
        held.context_made();
        held.context_made();
        assert_eq!(held.ask(Teardown::Context, 1), Asked::Held);
        assert_eq!(held.ask(Teardown::Surfaces, 2), Asked::Held);
        assert_eq!(held.ask(Teardown::Context, 3), Asked::Now(vec![1, 2, 3]));
        assert_eq!(held.ask(Teardown::Config, 4), Asked::Now(vec![4]));
    }

    #[test]
    fn a_context_made_after_a_hold_keeps_it_held_until_it_ends_too() {
        let mut held = Held::new();
        held.context_made();
        held.context_made();
        assert_eq!(held.ask(Teardown::Context, 1), Asked::Held);
        held.context_made();
        assert_eq!(held.ask(Teardown::Context, 2), Asked::Held);
        assert_eq!(held.ask(Teardown::Context, 3), Asked::Now(vec![1, 2, 3]));
    }

    #[test]
    fn a_context_teardown_with_none_counted_leaves_the_count_at_none() {
        let mut held = Held::new();
        assert_eq!(held.ask(Teardown::Context, 1), Asked::Now(vec![1]));
        held.context_made();
        assert_eq!(held.ask(Teardown::Context, 2), Asked::Now(vec![2]));
    }
}
