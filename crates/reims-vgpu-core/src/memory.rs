//! Guest memory as the host hands it over: the memory map and the mappings made from it.
//!
//! - [`descriptor`]: the host's ranges, in the order the host added them.
//! - [`map`]: [`MemoryMap`](map::MemoryMap), the guest's physical ranges sorted at init, and the
//!   mappings it makes (`mach_vm_remap` of each piece into a fresh or given region).
//! - [`mapping`]: [`Mapping`](mapping::Mapping), a host range over guest memory, and its teardown.
//!
//! # Edges the host sees
//!
//! Each of these is pinned by a test, and the code notes it where it happens.
//!
//! - **Map validation order** (`map`): the descriptor's ranges are summed in the host's order
//!   (an overflow refuses the map before anything is sorted), sorted by start and then length
//!   (zero-length ranges before the range with a length that shares their start), then
//!   checked against the previous range's end only.
//! - **Lookups** (`map`): the first containing range decides and a straddle is refused, while a
//!   range mapping splits across contiguous ranges. Guest physical 0 is refused even with
//!   length 0. A failed protect deallocates and still returns the mapping; a length of 0 passes
//!   the page assert.
//! - **Mappings** (`mapping`): the last release runs the teardown.
//!
//! # Outcomes
//!
//! Deaths are [`DeathKind`]s.

// Guest data flows through here: arithmetic is spelt out.
#![cfg_attr(not(test), deny(clippy::arithmetic_side_effects))]

pub mod descriptor;
pub mod map;
pub mod mapping;

use std::fmt;

use crate::raise::DeathId;
use crate::raise::death::{self, DeathSite};

/// The granule a mapping's length is whole multiples of ([`map::MemoryMap::new_range_mapping`]):
/// 16 KiB, 4 KiB in an x86-64 build.
pub const PAGE_SIZE: u64 = if cfg!(target_arch = "x86_64") {
    4096
} else {
    16384
};

/// Where memory ends the process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeathKind {
    /// A range mapping of a length that is not whole pages.
    MappingLengthUnaligned,
}

impl DeathKind {
    /// The dotted id.
    pub const fn name(self) -> &'static str {
        match self {
            DeathKind::MappingLengthUnaligned => "memory.map.mapping_length_unaligned",
        }
    }

    /// The kind of death there (diagnostics: every death aborts).
    const fn fault(self) -> death::DeathKind {
        match self {
            DeathKind::MappingLengthUnaligned => death::DeathKind::Assert,
        }
    }

    /// The site a death here records (`testkit::capture`) and names on stderr: the dotted id, at
    /// line 0.
    pub const fn site(self) -> DeathSite {
        DeathSite {
            kind: self.fault(),
            file: self.name(),
            line: 0,
        }
    }
}

impl fmt::Display for DeathKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl From<DeathKind> for DeathId {
    fn from(k: DeathKind) -> DeathId {
        DeathId::Memory(k)
    }
}
