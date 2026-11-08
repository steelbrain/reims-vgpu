//! Deaths: [`DeathId`], [`fatal`], [`fatal!`](crate::fatal) and
//! [`fatal_unless!`](crate::fatal_unless). A death is the one fatal path: one line on stderr,
//! `reims-vgpu: fatal: <what died>`, then `abort()`. Nothing unwinds and no `Drop` runs, so the
//! process ends at the guest op that caused it and nothing after it reaches the host or the
//! guest. Where the process dies is the contract; the message and the signal are diagnostics.
//! The one exception unwinds instead: a death inside a refusal scope refuses the op it met
//! ([`crate::core::raise::refusals`]), which only the graphics lane opens.
//!
//! [`DeathId`] names what died: an area's reason (its `Display` a dotted id), a raise nothing may
//! catch, a platform call, or a host call that raised. [`DeathSite`] and [`DeathKind`] are the
//! test capture's identity only: production never computes them. Under
//! `cfg(any(test, feature = "testkit"))` with capture on (the default), `fatal` records the
//! site for `testkit::capture` and panics with a `testkit::capture::Death` payload instead; with
//! capture off (the subprocess harness, `testkit::death`) its line also carries the site.

use std::fmt;

use crate::core::raise::{Origin, RaiseId};

/// A death as the test capture names it: its [`DeathKind`], and the dotted id, call or selector
/// it names with a line (0 for a named id, the Rust line for a raise that dies where it was
/// raised). Test identity only: `fatal`'s message is the [`DeathId`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeathSite {
    /// What kind of death it is.
    pub kind: DeathKind,
    /// The dotted id, call or file it names.
    pub file: &'static str,
    /// The line, or 0.
    pub line: u32,
}

/// The kinds of death, as the test capture classes them. Part of a death's identity in the test
/// capture and the subprocess harness only; every death ends the same way, in `abort()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeathKind {
    /// A failed assertion (`__assert_rtn`).
    Assert,
    /// `abort()`.
    Abort,
    /// A call through a host block that is nil.
    NilBlock,
    /// A NULL or near-NULL access.
    NullDeref,
    /// An out-of-bounds read that faults.
    OutOfBounds,
    /// A raise of core's own that dies at its site ([`DeathId::Raised`]): no handler is live
    /// there, or the site lets none take it ([`crate::core::raise::Raise::fatal`]).
    UnhandledRaise,
    /// A platform exception with no handler live, or at a send that no handler may take.
    UncaughtObjc,
    /// libdispatch's own crash (`dispatch_sync` onto a queue the thread already owns).
    DispatchMisuse,
    /// A stamp or task completion called a second time.
    DoubleCompletion,
    /// The host's code raised out of a call into it: `file` names the call.
    HostRaised,
}

/// A death's identity. Areas add one variant each (a `DeathKind` of their own, whose `Display`
/// prints a dotted id).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeathId {
    /// A raise that nothing may catch: one with no handler live, or one at a frame that cannot
    /// carry it (`Raise::fatal`).
    Raised {
        /// The raise's identity.
        id: RaiseId,
        /// Where it was raised.
        origin: Origin,
    },
    /// A death the platform causes at a call it makes, named by the call: an exception at a send
    /// no handler may take, or a system call that fails in a way that ends the process there.
    Platform {
        /// How the process ends.
        kind: DeathKind,
        /// The selector or C function.
        call: &'static str,
    },
    /// The host's code raised out of a call into it: a host block or host object that raises
    /// ends the process at the call.
    HostRaised {
        /// The call: the block's SDK property name.
        call: &'static str,
    },
    /// Memory's (`crate::core::memory::DeathKind`).
    Memory(crate::core::memory::DeathKind),
    /// A display's pipelines'.
    Display(crate::core::display::DeathKind),
    /// A panic in a refusal scope, which the scope refuses as a death there
    /// ([`crate::core::raise::refusals`], "Panics"): no death site raises it.
    Panic,
    /// A test's own death at a site it names (the testkit's fakes and the tests of the death
    /// machinery).
    #[cfg(any(test, feature = "testkit"))]
    Test {
        /// How the process ends.
        kind: DeathKind,
        /// The site's name.
        name: &'static str,
    },
}

/// A test's site as a death of its own: its kind, named by the site's name. A test's site is
/// named, not numbered, so its line is 0.
#[cfg(any(test, feature = "testkit"))]
impl From<DeathSite> for DeathId {
    fn from(s: DeathSite) -> DeathId {
        assert_eq!(s.line, 0, "a test's death site {} has a line", s.file);
        DeathId::Test {
            kind: s.kind,
            name: s.file,
        }
    }
}

impl DeathId {
    /// The site the test capture records: an area's death by its dotted name at line 0.
    pub fn site(self) -> DeathSite {
        match self {
            DeathId::Raised { id, origin } => crate::core::raise::death_site(id, origin),
            DeathId::Platform { kind, call } => DeathSite {
                kind,
                file: call,
                line: 0,
            },
            #[cfg(any(test, feature = "testkit"))]
            DeathId::Test { kind, name } => DeathSite {
                kind,
                file: name,
                line: 0,
            },
            DeathId::HostRaised { call } => DeathSite {
                kind: DeathKind::HostRaised,
                file: call,
                line: 0,
            },
            DeathId::Memory(k) => k.site(),
            DeathId::Display(k) => DeathSite {
                kind: k.class(),
                file: k.id(),
                line: 0,
            },
            // A panic outside every scope ends the process in the panic hook's `abort()` (6.5).
            DeathId::Panic => DeathSite {
                kind: DeathKind::Abort,
                file: "panic",
                line: 0,
            },
        }
    }
}

impl fmt::Display for DeathId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeathId::Raised { id, origin } => {
                write!(f, "uncaught raise {id}")?;
                match origin {
                    Origin::Core(at) => write!(f, " at {}:{}", at.file(), at.line()),
                    Origin::Platform(sel) => write!(f, " at the send of {sel}"),
                    Origin::Boundary(site) => write!(f, " at {site}"),
                }
            }
            DeathId::Platform { call, .. } => write!(f, "platform call {call}"),
            #[cfg(any(test, feature = "testkit"))]
            DeathId::Test { kind, name } => write!(f, "test.{kind:?}.{name}"),
            DeathId::HostRaised { call } => write!(f, "the host raised out of {call}"),
            DeathId::Memory(k) => k.fmt(f),
            DeathId::Display(k) => write!(f, "{k}"),
            DeathId::Panic => f.write_str("a panic"),
        }
    }
}

/// Ends the process with `id`, here: its line on stderr, then `abort()`. No `Drop` runs.
pub fn fatal(id: impl Into<DeathId>) -> ! {
    die(id.into(), None)
}

/// [`fatal`] with a detail after the id: what the platform said, for a death at a platform call.
pub fn fatal_with_detail(id: impl Into<DeathId>, detail: &str) -> ! {
    die(id.into(), Some(detail))
}

/// [`fatal`] with a detail after the id (a raise's).
pub(crate) fn die(id: DeathId, detail: Option<&str>) -> ! {
    // A death inside a refusal scope (the graphics lane's ops) refuses the op instead.
    crate::core::raise::refusals::trap(id, detail);
    #[cfg(any(test, feature = "testkit"))]
    if crate::core::testkit::capture::capture_enabled() {
        // A captured death unwinds quietly, before its line below is written: the line goes to
        // stderr first, so a test that dies where it should not shows what it died of.
        let _ = std::io::Write::write_all(
            &mut std::io::stderr(),
            format!("{}\n", message(&id, detail)).as_bytes(),
        );
        crate::core::testkit::capture::capture(id.site())
    }
    let mut line = message(&id, detail);
    // The subprocess harness names the death by its site (`testkit::death::parse_death_line`).
    #[cfg(any(test, feature = "testkit"))]
    {
        let DeathSite {
            kind,
            file,
            line: at,
        } = id.site();
        line = format!("{line} [site {kind:?} {file}:{at}]");
    }
    line.push('\n');
    use std::io::Write as _;
    // One write, so the line never interleaves with another thread's.
    let _ = std::io::stderr().write_all(line.as_bytes());
    // The trace's last lines, the events that led here (a diagnostic).
    crate::core::trace::flush_before_death();
    #[allow(clippy::disallowed_methods)] // the one fatal path's abort (6.4)
    std::process::abort()
}

/// The death's line: `reims-vgpu: fatal: <id>`, and `: <detail>` when there is one.
fn message(id: &DeathId, detail: Option<&str>) -> String {
    match detail {
        Some(detail) => format!("reims-vgpu: fatal: {id}: {detail}"),
        None => format!("reims-vgpu: fatal: {id}"),
    }
}

/// `fatal(id)`: ends the process here. Evaluates to `!`.
#[macro_export]
macro_rules! fatal {
    ($id:expr) => {
        $crate::core::raise::death::fatal($id)
    };
}

/// Ends the process with `id` unless `cond` holds.
#[macro_export]
macro_rules! fatal_unless {
    ($cond:expr, $id:expr) => {
        if !($cond) {
            $crate::core::raise::death::fatal($id)
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::testkit::capture::catch_death;

    /// `fatal` dies with the id's site; a test site names itself
    #[test]
    fn fatal_dies_at_the_site() {
        let site = DeathSite {
            kind: DeathKind::Assert,
            file: "test.site",
            line: 0,
        };
        assert_eq!(catch_death(|| fatal(site)), Err(site));
        assert_eq!(DeathId::from(site).to_string(), "test.Assert.test.site");
        let platform = DeathId::Platform {
            kind: DeathKind::Abort,
            call: "malloc",
        };
        assert_eq!(
            catch_death(|| fatal(platform)),
            Err(DeathSite {
                kind: DeathKind::Abort,
                file: "malloc",
                line: 0
            })
        );
        assert_eq!(platform.to_string(), "platform call malloc");
    }

    /// `fatal!` and `fatal_unless!` spell `fatal`; a host raise names its call
    #[test]
    fn the_macros_and_host_raised() {
        let host = DeathId::HostRaised { call: "mapMemory" };
        let d = catch_death(|| -> u32 { crate::fatal!(host) }).unwrap_err();
        assert_eq!(
            (d.kind, d.file, d.line),
            (DeathKind::HostRaised, "mapMemory", 0)
        );
        assert_eq!(host.to_string(), "the host raised out of mapMemory");
        assert_eq!(
            catch_death(|| crate::fatal_unless!(1 + 1 == 2, host)),
            Ok(())
        );
        assert!(catch_death(|| crate::fatal_unless!(1 + 1 == 3, host)).is_err());
    }

    /// The line names what died, and a raise's detail and origin
    #[test]
    fn the_line_says_what_died() {
        let raised = DeathId::Raised {
            id: RaiseId::Platform,
            origin: Origin::Boundary("setPurgeableState:"),
        };
        assert_eq!(
            message(&raised, Some("the reason")),
            "reims-vgpu: fatal: uncaught raise platform.raised at setPurgeableState:: the reason"
        );
        let area = DeathId::from(crate::core::display::DeathKind::NoLibrary);
        assert_eq!(
            message(&area, None),
            "reims-vgpu: fatal: display.pipeline.no_library"
        );
    }
}
