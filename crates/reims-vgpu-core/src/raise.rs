//! The raise channel: an exception as a plain value.
//!
//! [`Raise`] is what a raise carries and [`RaiseResult`] the channel. Each raise site decides its
//! raise ([`on_fault`]): `Err` where a handler is always live ([`raise!`](crate::raise!)`(id)`), a
//! death where none ever is ([`Raise::fatal`]), before any `Drop` runs, so lock guards stay held
//! and nothing more is sent or written, and its caller's [`OnFault`] where both are reached
//! ([`raise!`](crate::raise!)`(on => id)`). Deaths are [`death`], values kept for good
//! [`kept`], and the scope where a death refuses its op instead [`refusals`].

pub mod death;
pub mod kept;
pub mod on_fault;
pub mod refusals;

use std::fmt;
use std::panic::Location;

pub use death::{DeathId, DeathKind, DeathSite, fatal};
pub use kept::{Kept, keep_forever};
pub use on_fault::OnFault;

use crate::rt::held::Deferred;

/// A raise's identity: what raised and why. Areas add one variant each (a `RaiseKind` of their
/// own, whose `Display` prints a dotted id such as `device.wait.circular`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RaiseId {
    /// An exception the platform raised at a send (a Metal or Foundation exception); the
    /// origin names the selector.
    Platform,
    /// The tasks area's.
    Tasks(crate::tasks::RaiseKind),
    /// The deserializer's.
    Deserializer(crate::deserializer::RaiseKind),
    /// Memory's (`crate::memory::RaiseKind`).
    Memory(crate::memory::RaiseKind),
    /// The resources area's.
    Resources(crate::resources::RaiseKind),
    /// A display's.
    Display(crate::display::RaiseKind),
    /// A FIFO command's.
    Fifo(crate::fifo::RaiseKind),
    /// The GPU task service's.
    GpuTask(crate::gpu_task::RaiseKind),
    /// The device's (`crate::device::DeviceRaise`).
    Device(crate::device::DeviceRaise),
    /// The resource manager's.
    Manager(crate::resources::manager::RaiseKind),
    /// A guest display's.
    GuestDisplay(crate::display::guest_display::RaiseKind),
    /// The device's display ports'.
    Displays(crate::device::displays::RaiseKind),
    /// A test's own raise, which dies uncaught at the site it names.
    #[cfg(any(test, feature = "testkit"))]
    Test {
        /// The site's name.
        name: &'static str,
    },
}

impl RaiseId {
    /// An area's raise by its dotted name, which an uncaught one's death names (at line 0).
    /// `None` for a platform id, whose site is the selector's, for a resources raise, which dies
    /// where it was raised, and for a test's.
    pub fn name(&self) -> Option<&'static str> {
        match self {
            RaiseId::Platform | RaiseId::Resources(_) => None,
            #[cfg(any(test, feature = "testkit"))]
            RaiseId::Test { .. } => None,
            RaiseId::Tasks(k) => Some(k.id()),
            RaiseId::Deserializer(k) => Some(k.name()),
            RaiseId::Memory(k) => Some(k.name()),
            RaiseId::Display(k) => Some(k.id()),
            RaiseId::Fifo(k) => Some(k.name()),
            RaiseId::GpuTask(k) => Some(k.id()),
            RaiseId::Device(k) => Some(k.name()),
            RaiseId::Manager(k) => Some(k.name()),
            RaiseId::GuestDisplay(k) => Some(k.id()),
            RaiseId::Displays(k) => Some(k.id()),
        }
    }
}

impl From<crate::deserializer::RaiseKind> for RaiseId {
    fn from(k: crate::deserializer::RaiseKind) -> RaiseId {
        RaiseId::Deserializer(k)
    }
}

impl fmt::Display for RaiseId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(any(test, feature = "testkit"))]
            RaiseId::Test { name } => write!(f, "test.{name}"),
            RaiseId::Platform => f.write_str("platform.raised"),
            RaiseId::Tasks(k) => write!(f, "{k}"),
            RaiseId::Deserializer(k) => f.write_str(k.name()),
            RaiseId::Memory(k) => k.fmt(f),
            RaiseId::Resources(k) => write!(f, "resources.{k}"),
            RaiseId::Display(k) => write!(f, "{k}"),
            RaiseId::Fifo(k) => write!(f, "{k}"),
            RaiseId::GpuTask(k) => write!(f, "{k}"),
            RaiseId::Device(k) => f.write_str(k.name()),
            RaiseId::Manager(k) => f.write_str(k.name()),
            RaiseId::GuestDisplay(k) => write!(f, "{k}"),
            RaiseId::Displays(k) => write!(f, "{k}"),
        }
    }
}

/// Where a raise came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Origin {
    /// Core's own raise, at this call.
    Core(&'static Location<'static>),
    /// An exception the platform caught at the send of this selector (or Metal C function,
    /// with `()`).
    Platform(&'static str),
    /// A foreign frame or send named by its site alone: a boundary's frame (Metal's resolver
    /// callback), or a send named by its selector (a fake platform's).
    Boundary(&'static str),
}

/// A raise: its identity, an optional detail (diagnostics only) and its origin, and the values it
/// carries outward ([`Raise::defer`]). A plain value: building one decides nothing, so tests
/// build them freely. Two raises are equal by identity, detail and origin.
pub struct Raise {
    id: RaiseId,
    detail: Option<Box<str>>,
    origin: Origin,
    /// Released where a catch ends the raise, in its hold's holder
    /// ([`Hold::land`](crate::rt::Hold::land)).
    deferred: Deferred,
}

impl Clone for Raise {
    /// The same identity, detail and origin: a kept answer thrown again (the device's
    /// capabilities). A raise that carries values is never cloned, since each is released once.
    fn clone(&self) -> Raise {
        debug_assert!(self.deferred.is_empty(), "a raise that carries values");
        Raise {
            id: self.id,
            detail: self.detail.clone(),
            origin: self.origin,
            deferred: Deferred::default(),
        }
    }
}

impl fmt::Debug for Raise {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut d = f.debug_struct("Raise");
        d.field("id", &self.id)
            .field("detail", &self.detail)
            .field("origin", &self.origin);
        if !self.deferred.is_empty() {
            d.field("deferred", &self.deferred);
        }
        d.finish()
    }
}

impl PartialEq for Raise {
    fn eq(&self, other: &Raise) -> bool {
        (self.id, &self.detail, self.origin) == (other.id, &other.detail, other.origin)
    }
}

impl Eq for Raise {}

/// The raise channel. There is no `From` between it and an expected error or a death.
pub type RaiseResult<T> = Result<T, Raise>;

impl Raise {
    /// A raise of `id` at the caller. Pure.
    #[track_caller]
    pub fn new(id: impl Into<RaiseId>) -> Raise {
        Raise {
            id: id.into(),
            detail: None,
            origin: Origin::Core(Location::caller()),
            deferred: Deferred::default(),
        }
    }

    /// An exception the platform raised at the send of `sel` (the message layer's raise).
    pub fn platform(sel: &'static str) -> Raise {
        Raise {
            id: RaiseId::Platform,
            detail: None,
            origin: Origin::Platform(sel),
            deferred: Deferred::default(),
        }
    }

    /// An exception foreign code raised at `site`, named by its selector alone: it dies at
    /// `site` as [`Raise::platform`]'s does at its selector.
    pub fn foreign(site: &'static str) -> Raise {
        Raise {
            id: RaiseId::Platform,
            detail: None,
            origin: Origin::Boundary(site),
            deferred: Deferred::default(),
        }
    }

    /// The same raise with its detail.
    pub fn with_detail(mut self, detail: impl fmt::Display) -> Raise {
        self.detail = Some(detail.to_string().into_boxed_str());
        self
    }

    /// The same raise, carrying `v` outward: `v` is released where a catch ends the raise, in its
    /// hold's holder ([`Hold::land`](crate::rt::Hold::land)), after the catcher's report, and
    /// never if it dies. It takes its place among that holder's values at the point it was
    /// deferred.
    pub fn defer<T: std::any::Any + Send>(mut self, v: T) -> Raise {
        self.deferred.push(v);
        self
    }

    /// What it carries, for a landing frame to add its values to ([`crate::rt::Held::raising`])
    /// and a catch to land ([`Hold::land`](crate::rt::Hold::land)).
    pub(crate) fn deferred_mut(&mut self) -> &mut Deferred {
        &mut self.deferred
    }

    /// Its identity.
    pub fn id(&self) -> RaiseId {
        self.id
    }

    /// Its detail, if it has one.
    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }

    /// Where it came from.
    pub fn origin(&self) -> Origin {
        self.origin
    }

    /// The test capture's site of this raise's death when nothing catches it.
    pub fn site(&self) -> DeathSite {
        death_site(self.id, self.origin)
    }

    /// Dies with this raise, even under a handler: a frame that cannot carry a raise. What it
    /// carries is never released.
    pub fn fatal(mut self) -> ! {
        self.deferred.forget();
        let id = DeathId::Raised {
            id: self.id,
            origin: self.origin,
        };
        death::die(id, self.detail())
    }

    /// This raise in a log line: its detail, empty when it has none.
    pub fn exception(&self) -> &str {
        self.detail().unwrap_or("")
    }
}

/// The test capture's site of an uncaught raise of `id` from `origin`: the selector at line 0 for
/// the platform's, an area's dotted name at line 0 for its own, and for a resources raise the
/// Rust site it was raised at.
pub(crate) fn death_site(id: RaiseId, origin: Origin) -> DeathSite {
    match (id, origin) {
        (_, Origin::Platform(sel)) => DeathSite {
            kind: DeathKind::UncaughtObjc,
            file: sel,
            line: 0,
        },
        #[cfg(any(test, feature = "testkit"))]
        (RaiseId::Test { name }, _) => DeathSite {
            kind: DeathKind::UnhandledRaise,
            file: name,
            line: 0,
        },
        (RaiseId::Tasks(k), _) => DeathSite {
            kind: DeathKind::UnhandledRaise,
            file: k.id(),
            line: 0,
        },
        (RaiseId::GpuTask(k), _) => DeathSite {
            kind: DeathKind::UnhandledRaise,
            file: k.id(),
            line: 0,
        },
        (RaiseId::Platform, Origin::Core(at)) => DeathSite {
            kind: DeathKind::UncaughtObjc,
            file: at.file(),
            line: at.line(),
        },
        (RaiseId::Platform, Origin::Boundary(site)) => DeathSite {
            kind: DeathKind::UncaughtObjc,
            file: site,
            line: 0,
        },
        (RaiseId::Deserializer(k), _) => DeathSite {
            kind: DeathKind::UnhandledRaise,
            file: k.name(),
            line: 0,
        },
        (RaiseId::Memory(k), _) => DeathSite {
            kind: DeathKind::UnhandledRaise,
            file: k.name(),
            line: 0,
        },
        (RaiseId::Display(k), _) => DeathSite {
            kind: DeathKind::UnhandledRaise,
            file: k.id(),
            line: 0,
        },
        (RaiseId::Fifo(k), _) => DeathSite {
            kind: DeathKind::UnhandledRaise,
            file: k.name(),
            line: 0,
        },
        (RaiseId::Device(k), _) => DeathSite {
            kind: DeathKind::UnhandledRaise,
            file: k.name(),
            line: 0,
        },
        (RaiseId::Manager(k), _) => DeathSite {
            kind: DeathKind::UnhandledRaise,
            file: k.name(),
            line: 0,
        },
        (RaiseId::GuestDisplay(k), _) => DeathSite {
            kind: DeathKind::UnhandledRaise,
            file: k.id(),
            line: 0,
        },
        (RaiseId::Displays(k), _) => DeathSite {
            kind: DeathKind::UnhandledRaise,
            file: k.id(),
            line: 0,
        },
        // A resources raise dies where it was raised.
        (_, Origin::Core(at)) => DeathSite {
            kind: DeathKind::UnhandledRaise,
            file: at.file(),
            line: at.line(),
        },
        (_, Origin::Boundary(site)) => DeathSite {
            kind: DeathKind::UnhandledRaise,
            file: site,
            line: 0,
        },
    }
}

impl fmt::Display for Raise {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.id)?;
        if let Some(d) = &self.detail {
            write!(f, ": {d}")?;
        }
        Ok(())
    }
}

/// A raise of `id`, with an optional `detail`, returned from the enclosing function:
/// - `raise!(id)` or `raise!(id, detail)` at a site no path reaches without a handler, which
///   returns `Err`, the caller's to catch;
/// - `raise!(on => id)` or `raise!(on => id, detail)` in a function reached both ways, whose
///   caller says by `on` which way this call is ([`OnFault::throw`]): [`OnFault::Die`] dies here,
///   before any `Drop` runs, and [`OnFault::Return`] returns `Err`.
///
/// A site no path reaches under a handler dies there, written out:
/// `Raise::new(id).with_detail(detail).fatal()`.
#[macro_export]
macro_rules! raise {
    ($on:expr => $id:expr) => {
        return $on.throw($crate::raise::Raise::new($id))
    };
    ($on:expr => $id:expr, $detail:expr) => {
        return $on.throw($crate::raise::Raise::new($id).with_detail($detail))
    };
    ($id:expr) => {
        return Err($crate::raise::Raise::new($id))
    };
    ($id:expr, $detail:expr) => {
        return Err($crate::raise::Raise::new($id).with_detail($detail))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Building a raise decides nothing; its detail and id read back
    #[test]
    fn new_is_pure() {
        let r = Raise::new(RaiseId::Test { name: "raise" }).with_detail("no task 7");
        assert_eq!(r.detail(), Some("no task 7"));
        assert_eq!(r.to_string(), "test.raise: no task 7");
        assert!(matches!(r.origin(), Origin::Core(_)));
    }
}
