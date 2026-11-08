//! The display's pipelines ([`pipeline`]), which the graphics device's window draws the guest's
//! scanout with, and the deaths they end the process on ([`DeathKind`]).

pub mod pipeline;

use std::fmt;

use crate::raise::death::DeathId;

/// Where a display's pipelines end the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeathKind {
    /// Metal gave no default library for the pipelines.
    NoLibrary,
    /// The library has no vertex function.
    NoVertexFunction,
    /// The library has no fragment function.
    NoFragmentFunction,
    /// The library has no gamma fragment function.
    NoGammaFragmentFunction,
}

impl DeathKind {
    /// The dotted id.
    pub const fn id(self) -> &'static str {
        match self {
            DeathKind::NoLibrary => "display.pipeline.no_library",
            DeathKind::NoVertexFunction => "display.pipeline.no_vertex_function",
            DeathKind::NoFragmentFunction => "display.pipeline.no_fragment_function",
            DeathKind::NoGammaFragmentFunction => "display.pipeline.no_gamma_fragment_function",
        }
    }

    /// The kind of death.
    pub(crate) const fn class(self) -> crate::raise::death::DeathKind {
        crate::raise::death::DeathKind::Assert
    }

    /// The site a death of this kind names (tests).
    #[cfg(test)]
    pub(crate) fn site(self) -> crate::raise::death::DeathSite {
        crate::raise::death::DeathId::from(self).site()
    }
}

impl fmt::Display for DeathKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id())
    }
}

impl From<DeathKind> for DeathId {
    fn from(k: DeathKind) -> DeathId {
        DeathId::Display(k)
    }
}
