//! [`OnFault`]: what a raise does in a function reached both under a handler and with none.
//!
//! Every raise is decided by its site: a site no path reaches without a handler returns `Err`
//! (`raise!(id)`), one no path reaches under a handler dies there, before any `Drop` runs
//! ([`Raise::fatal`]), and a function its callers reach both ways takes an [`OnFault`] from its
//! caller, which says which of the two the call is (`raise!(on => id)`, [`OnFault::throw`]).

use crate::core::raise::{Raise, RaiseResult};

/// What a raise does in a function its callers reach both under a handler and with none: the
/// caller, which knows which of the two it is, says so. A caller reached only under a handler
/// passes [`OnFault::Return`], one reached only with none [`OnFault::Die`], and one reached both
/// ways passes on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OnFault {
    /// No handler is live: the raise dies at its site, before any `Drop` runs.
    Die,
    /// A handler is live: the raise is `Err`, the caller's to catch.
    Return,
}

impl OnFault {
    /// The raise `r`, decided by this value alone: [`OnFault::Die`] dies here, before any `Drop`
    /// runs (lock guards stay held, nothing more is sent or written), and [`OnFault::Return`] is
    /// `Err(r)`.
    pub fn throw<T>(self, r: Raise) -> RaiseResult<T> {
        match self {
            OnFault::Die => r.fatal(),
            OnFault::Return => Err(r),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::raise::RaiseId;
    use crate::core::raise::death::{DeathKind, DeathSite};
    use crate::core::testkit::capture::catch_death;

    fn a_raise() -> Raise {
        Raise::new(RaiseId::Test { name: "on_fault" })
    }

    /// `OnFault::throw` decides by its value alone: `Return` is `Err`, `Die`
    /// dies at the raise's site
    #[test]
    fn on_fault_throw_decides_by_its_value() {
        let r = OnFault::Return.throw::<()>(a_raise());
        assert_eq!(r.map_err(|r| r.id()), Err(a_raise().id()));
        let d = catch_death(|| OnFault::Die.throw::<()>(a_raise()));
        assert_eq!(
            d,
            Err(DeathSite {
                kind: DeathKind::UnhandledRaise,
                file: "on_fault",
                line: 0
            })
        );
    }

    /// `raise!(id)` returns `Err` from the enclosing function, and `raise!(on => id)` does what
    /// `on` says, each with or without a detail
    #[test]
    fn the_macro_returns_err_or_decides_by_on() {
        fn caught(fail: bool) -> RaiseResult<u32> {
            if fail {
                crate::raise!(RaiseId::Test { name: "caught" }, "bad");
            }
            Ok(1)
        }
        fn bare() -> RaiseResult<()> {
            crate::raise!(RaiseId::Test { name: "bare" });
        }
        fn both(on: OnFault) -> RaiseResult<()> {
            crate::raise!(on => RaiseId::Test { name: "both" }, "bad");
        }
        fn both_bare(on: OnFault) -> RaiseResult<()> {
            crate::raise!(on => RaiseId::Test { name: "both_bare" });
        }
        let r = caught(true).unwrap_err();
        assert_eq!(
            (r.id(), r.detail()),
            (RaiseId::Test { name: "caught" }, Some("bad"))
        );
        assert_eq!(caught(false), Ok(1));
        let r = bare().unwrap_err();
        assert_eq!((r.id(), r.detail()), (RaiseId::Test { name: "bare" }, None));
        let r = both(OnFault::Return).unwrap_err();
        assert_eq!(
            (r.id(), r.detail()),
            (RaiseId::Test { name: "both" }, Some("bad"))
        );
        let d = catch_death(|| both(OnFault::Die)).unwrap_err();
        assert_eq!((d.kind, d.file), (DeathKind::UnhandledRaise, "both"));
        assert!(both_bare(OnFault::Return).is_err());
        let d = catch_death(|| both_bare(OnFault::Die)).unwrap_err();
        assert_eq!((d.kind, d.file), (DeathKind::UnhandledRaise, "both_bare"));
    }

    /// A raise's own death dies at its site, and a platform raise's at its selector
    #[test]
    fn a_fatal_raise_dies_at_its_site() {
        let d = catch_death(|| -> RaiseResult<()> { a_raise().fatal() });
        assert_eq!(
            d.map_err(|s| (s.kind, s.file)),
            Err((DeathKind::UnhandledRaise, "on_fault"))
        );
        let d = catch_death(|| -> RaiseResult<()> { Raise::platform("commit").fatal() });
        assert_eq!(
            d,
            Err(DeathSite {
                kind: DeathKind::UncaughtObjc,
                file: "commit",
                line: 0
            })
        );
    }
}
