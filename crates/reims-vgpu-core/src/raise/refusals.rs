//! The refusal scope, [`refusing`]: where a death on this thread refuses the op that met it, and
//! the process goes on.
//!
//! The reims-graphics lane's host side runs each guest op inside a scope
//! (docs/graphics-kext/design.md, "Guest input the host refuses"). There, a death that would end
//! the process, a platform refusing a create or an encode among them, unwinds to the innermost
//! scope instead, which answers [`Refusal`]. The lane then refuses the op to the guest. The
//! paravirt device never opens a scope, so its deaths stay where its contract puts them: outside
//! a scope [`trap`] returns at once, and the death writes its line and ends the process.
//!
//! **Panics.** A panic is a bug (architecture 6.5), and one inside a scope is refused as a death
//! there is: the library's panic hook writes its line and lets it unwind ([`refuses_panics`]),
//! and the scope answers it as a [`Refusal`] of [`DeathId::Panic`], logged as the bug it is. So
//! no guest input ends the process through a panic in the lane's code either. Under the test
//! capture a panic is a test's failure, and goes on past every scope, as any other unwind does:
//! a test's assertion in work a scope runs fails the test.
//!
//! **Unwinding.** The thread unwinds from the death to the scope, and the `Drop`s between them
//! run: a handle dropped there is released. A `Drop` the unwind runs may open a scope of its own
//! (a release refused alone, [`Carried::run`]), and a death inside it unwinds to that scope, as
//! one does on a thread that is not unwinding: the scope answers it, and the unwind it was opened
//! in goes on to its own scope. A death in such a `Drop` outside any scope it opened is not
//! trapped and ends the process, since no unwind may leave a `Drop` an unwind runs. So does a
//! death on a thread that runs no scope (a platform's own threads), but where the scope was
//! carried there.
//!
//! **Carried scopes.** Work an op registers to run later on another thread (a command buffer's
//! completed handler, which the platform runs on its own thread after it reads the buffer's
//! failure) takes the scope with it: [`carry`] at the registration notes whether the op runs in a
//! scope, and [`Carried::run`] runs the work in one there too. So a death in the work, the
//! platform's read of a failure no Metal answer matches among them, refuses that work and the
//! process goes on. Work registered outside every scope, all of the paravirt device's, runs as it
//! always did.

#![expect(
    clippy::disallowed_methods,
    reason = "the refusal scope: a death in the graphics lane's op unwinds to its scope (6.4)"
)]

use std::cell::Cell;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use crate::raise::death::DeathId;

thread_local! {
    /// How many scopes this thread is inside.
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    /// Whether the innermost scope was entered while the thread was unwinding already: one a
    /// `Drop` the unwind runs opened, which a death inside it may unwind to (module doc,
    /// "Unwinding").
    static ENTERED_UNWINDING: Cell<bool> = const { Cell::new(false) };
}

/// A death a scope caught: what died, and the detail its line would have carried.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refusal {
    /// What died.
    pub id: DeathId,
    /// What the death said after its id: the platform's refusal, for a platform call.
    pub detail: Option<String>,
}

impl fmt::Display for Refusal {
    /// The death's line without its prefix: `<id>` and `: <detail>` where there is one.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.id)?;
        if let Some(detail) = &self.detail {
            write!(f, ": {detail}")?;
        }
        Ok(())
    }
}

/// Leaves the scope as the thread leaves `refusing`, whether `f` returned or unwound, and puts
/// back what the scope it was inside was entered as.
struct Depth {
    /// [`ENTERED_UNWINDING`] of the scope this one is inside.
    outer_entered_unwinding: bool,
}

impl Depth {
    fn enter() -> Depth {
        DEPTH.with(|d| d.set(d.get().saturating_add(1)));
        let unwinding = std::thread::panicking();
        Depth {
            outer_entered_unwinding: ENTERED_UNWINDING.with(|e| e.replace(unwinding)),
        }
    }
}

impl Drop for Depth {
    fn drop(&mut self) {
        DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
        ENTERED_UNWINDING.with(|e| e.set(self.outer_entered_unwinding));
    }
}

/// Runs `f` in a refusal scope: `Ok` with what `f` returns, or `Err` with the [`Refusal`] of a
/// death on this thread inside it, or of a panic there ([`refuses_panics`]), which ended `f`
/// there (module doc). Any other unwind goes on past the scope.
pub fn refusing<T>(f: impl FnOnce() -> T) -> Result<T, Refusal> {
    let depth = Depth::enter();
    let r = catch_unwind(AssertUnwindSafe(f));
    drop(depth);
    match r {
        Ok(v) => Ok(v),
        Err(payload) => match payload.downcast::<Refusal>() {
            Ok(refusal) => Err(*refusal),
            Err(other) => match panic_message(&*other) {
                Some(message) if panics_refused() => {
                    log::error!(
                        target: "reims_vgpu::refused",
                        "a panic in a refusal scope, a bug, refused the op it met: {message}"
                    );
                    Err(Refusal {
                        id: DeathId::Panic,
                        detail: Some(message),
                    })
                }
                _ => resume_unwind(other),
            },
        },
    }
}

/// What a panic says: the message of `panic!` and of the panics of std (an index out of bounds,
/// an overflow, an `unwrap`), whose payload is a string; `None` for any other payload, an unwind
/// of the library's own (a test capture's death, an isolated task's stop) among them.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> Option<String> {
    match payload.downcast_ref::<&'static str>() {
        Some(message) => Some((*message).to_owned()),
        None => payload.downcast_ref::<String>().cloned(),
    }
}

/// Whether a scope answers a panic: always, but under the test capture (the module doc,
/// "Panics"), where a panic is a test's failure.
fn panics_refused() -> bool {
    #[cfg(any(test, feature = "testkit"))]
    if crate::testkit::capture::capture_enabled() {
        return false;
    }
    true
}

/// The panic hook's question (architecture 6.5): whether the panic on this thread now is a
/// scope's to refuse, so the hook writes its line and lets it unwind there instead of ending the
/// process: the thread is inside a scope ([`refusing`]), and scopes answer panics. A panic in a
/// `Drop` an unwind runs, outside any scope the `Drop` opened, still ends the process once the
/// hook returns, as Rust ends it for any panic that would leave such a `Drop`.
pub fn refuses_panics() -> bool {
    is_refusing() && panics_refused()
}

/// Whether this thread is inside a refusal scope.
pub fn is_refusing() -> bool {
    DEPTH.with(Cell::get) > 0
}

/// Whether work registered now runs in a refusal scope where it runs later ([`carry`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use]
pub struct Carried(bool);

/// The scope of the op registering work now, for the thread that runs the work later (module
/// doc, "Carried scopes"): a scope where this thread is inside one, none otherwise.
pub fn carry() -> Carried {
    Carried(is_refusing())
}

impl Carried {
    /// Whether the work runs in a scope.
    pub fn refuses(self) -> bool {
        self.0
    }

    /// Runs `f`: in a refusal scope where the registration was in one ([`refusing`]), so a death
    /// in it is its `Err`; else plainly, so a death in it ends the process, as outside every
    /// scope.
    pub fn run<T>(self, f: impl FnOnce() -> T) -> Result<T, Refusal> {
        if self.0 { refusing(f) } else { Ok(f()) }
    }
}

/// Whether a death on this thread now unwinds to the innermost scope: the thread is inside one,
/// and is not unwinding, or that scope was entered as it unwound, inside a `Drop` the unwind runs
/// (module doc, "Unwinding").
fn takes_a_death() -> bool {
    is_refusing() && (!std::thread::panicking() || ENTERED_UNWINDING.with(Cell::get))
}

/// `raise::death`'s hook: inside a scope that can take it ([`takes_a_death`]), the death unwinds
/// to it as a [`Refusal`]. Returns where the thread is in no scope, or is unwinding already
/// toward it: the process then ends as it always did.
pub(crate) fn trap(id: DeathId, detail: Option<&str>) {
    if !takes_a_death() {
        return;
    }
    resume_unwind(Box::new(Refusal {
        id,
        detail: detail.map(str::to_owned),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::raise::death::{DeathKind, DeathSite, fatal_with_detail};
    use crate::raise::{Raise, RaiseResult};
    use crate::testkit::capture::catch_death;

    const SITE: DeathSite = DeathSite {
        kind: DeathKind::Abort,
        file: "refusals.test",
        line: 0,
    };

    /// A platform call's death inside a scope refuses: `refusing` answers what died and its
    /// detail, the `Drop`s between them run, and the thread leaves the scope
    #[test]
    fn a_death_in_a_scope_is_its_refusal() {
        struct Dropped<'a>(&'a Cell<bool>);
        impl Drop for Dropped<'_> {
            fn drop(&mut self) {
                self.0.set(true);
            }
        }
        let dropped = Cell::new(false);
        let r: Result<(), Refusal> = refusing(|| {
            let _held = Dropped(&dropped);
            assert!(is_refusing());
            fatal_with_detail(
                DeathId::Platform {
                    kind: DeathKind::Abort,
                    call: "newRenderPipelineStateWithDescriptor:error:",
                },
                "the runtime refused it",
            )
        });
        let refusal = r.unwrap_err();
        assert_eq!(
            refusal.id,
            DeathId::Platform {
                kind: DeathKind::Abort,
                call: "newRenderPipelineStateWithDescriptor:error:",
            }
        );
        assert_eq!(
            refusal.to_string(),
            "platform call newRenderPipelineStateWithDescriptor:error:: the runtime refused it"
        );
        assert!(dropped.get());
        assert!(!is_refusing());
    }

    /// Outside every scope a death is a death, as before any scope existed; a scope's work that
    /// returns is its value
    #[test]
    fn outside_a_scope_a_death_is_a_death() {
        assert_eq!(refusing(|| 7), Ok(7));
        assert!(!is_refusing());
        let d = catch_death(|| crate::fatal!(SITE));
        assert_eq!(d, Err(SITE));
        let d = catch_death(|| -> RaiseResult<()> { Raise::platform("commit").fatal() });
        assert_eq!(d.map_err(|s| s.kind), Err(DeathKind::UncaughtObjc));
    }

    /// An uncaught raise inside a scope refuses too, and the innermost scope takes the death
    #[test]
    fn the_innermost_scope_takes_the_death() {
        let outer = refusing(|| {
            let inner: Result<(), Refusal> =
                refusing(|| Raise::platform("setRenderPipelineState:").fatal());
            let refusal = inner.unwrap_err();
            assert!(refusal.to_string().contains("setRenderPipelineState:"));
            assert!(is_refusing());
            5
        });
        assert_eq!(outer, Ok(5));
    }

    /// Work registered inside a scope runs in one on the thread that runs it, and its death is
    /// its refusal there; work registered outside every scope runs plainly, and its death is a
    /// death
    #[test]
    fn a_carried_scope_refuses_on_the_thread_that_runs_it() {
        let carried = refusing(carry).unwrap();
        assert!(carried.refuses());
        let refused = std::thread::spawn(move || {
            let r = carried.run(|| -> u32 { Raise::platform("addCompletedHandler:").fatal() });
            (r, is_refusing())
        })
        .join()
        .unwrap();
        let refusal = refused.0.unwrap_err();
        assert!(
            refusal.to_string().contains("addCompletedHandler:"),
            "{refusal}"
        );
        assert!(!refused.1);
        assert_eq!(carried.run(|| 3), Ok(3));

        let plain = carry();
        assert!(!plain.refuses());
        assert_eq!(plain.run(|| 4), Ok(4));
        let d = catch_death(|| plain.run(|| crate::fatal!(SITE)));
        assert_eq!(d.map(|_| ()), Err(SITE));
    }

    /// A death in a `Drop` an unwind runs, inside a scope that `Drop` opened, is that scope's
    /// refusal, and the unwind goes on to its own scope, which answers the first death; outside
    /// every scope the `Drop` opened, the thread is unwinding toward a scope no unwind can reach
    /// from there, and no scope takes the death
    #[test]
    fn a_death_while_unwinding_is_refused_in_a_scope_opened_since() {
        /// A release that dies, refused alone in a scope of its own, as the lane's releases are.
        struct RefusedRelease<'a>(&'a Cell<Option<String>>);
        impl Drop for RefusedRelease<'_> {
            fn drop(&mut self) {
                assert!(std::thread::panicking() && is_refusing());
                let refused: Result<(), Refusal> = refusing(|| Raise::platform("release").fatal());
                self.0.set(refused.err().map(|r| r.to_string()));
            }
        }
        let released = Cell::new(None);
        let r: Result<(), Refusal> = refusing(|| {
            let _held = RefusedRelease(&released);
            Raise::platform("commit").fatal()
        });
        assert!(r.unwrap_err().to_string().contains("commit"));
        let released = released.take().expect("the release ran in the unwind");
        assert!(released.contains("release"), "{released}");
        assert!(!is_refusing());
    }

    /// A panic in a scope is its refusal, a bug's, where scopes answer panics (the product, and the
    /// subprocess harness with the capture off); under the capture it is a test's failure, and
    /// passes every scope by
    #[test]
    fn a_panic_in_a_scope_is_its_refusal() {
        use crate::testkit::capture::{capture_enabled, set_capture_enabled};
        assert!(capture_enabled());
        let r = catch_unwind(|| refusing(|| -> u32 { panic!("a test's assertion") }));
        let payload = r.unwrap_err();
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"a test's assertion"));
        if std::env::var_os("RVG_REFUSAL_PANICS_CHILD").is_none() {
            let out = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "raise::refusals::tests::a_panic_in_a_scope_is_its_refusal",
                    "--nocapture",
                    "--test-threads=1",
                ])
                .env("RVG_REFUSAL_PANICS_CHILD", "1")
                .output()
                .unwrap();
            assert!(out.status.success(), "{out:?}");
            return;
        }
        // In a child: as the product, a scope answers a panic.
        set_capture_enabled(false);
        assert!(!refuses_panics());
        let held = Cell::new(false);
        let r = refusing(|| {
            assert!(refuses_panics());
            held.set(true);
            let words = [1u32, 2];
            let at = words.len() + usize::from(held.get());
            words[at]
        });
        let refusal = r.unwrap_err();
        assert_eq!(refusal.id, DeathId::Panic);
        assert!(
            refusal
                .to_string()
                .starts_with("a panic: index out of bounds"),
            "{refusal}"
        );
        let r: Result<(), Refusal> = refusing(|| panic!("formatted {}", 7));
        assert_eq!(r.unwrap_err().to_string(), "a panic: formatted 7");
        // A carried scope answers one too, on the thread that runs the work.
        let carried = refusing(carry).unwrap();
        let r = std::thread::spawn(move || carried.run(|| -> u32 { panic!("in a handler") }))
            .join()
            .unwrap();
        assert_eq!(r.unwrap_err().to_string(), "a panic: in a handler");
        assert!(!is_refusing());
    }

    /// An unwind that is not a death passes the scope by
    #[test]
    fn another_unwind_passes_the_scope() {
        let r = catch_unwind(|| refusing(|| resume_unwind(Box::new(41u32))));
        let payload = r.unwrap_err();
        assert_eq!(payload.downcast_ref::<u32>(), Some(&41));
        assert!(!is_refusing());
    }
}
