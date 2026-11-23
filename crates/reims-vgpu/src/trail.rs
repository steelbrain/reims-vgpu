//! The trail: the last [`LEN`] events of the host's that placed or took away a hold of the
//! device's on the guest's memory, kept in memory and written to the log at a device reset that
//! dropped live objects. A diagnostic: the guest sees nothing of it.
//!
//! A guest whose kernel finds a page table damaged after a page it had reused was written leaves
//! nothing on the host that names what of the device's could still reach that page. The trail
//! keeps, for each such event, its wall-clock time (milliseconds since the Unix epoch), its kind,
//! the guest's context where the event has one, the region offset of the block it concerns where
//! it has one, and the guest-physical runs it covers where the host holds them at the event: the
//! first [`RUNS`] of them, with how many runs and bytes there are in all where there are more.
//! The kinds ([`Kind`]):
//!
//! - an import of a view of a block's `SURFACE_PAGES` runs (an IOSurface's block or a pages
//!   block) made, a cached view's import taken by a block over the same runs, and the import
//!   freed by the platform, which it does once the command buffers that used it completed
//!   (`graphics::decode::surface_pages`);
//! - a `BLOCK_RELEASE` asked, and its answer, OK or refused, with how long it waited;
//! - a `POOL_GIVE`'s and a `PRIVATE_GIVE`'s answer OK, with the block's offset in the pool or
//!   GPU address and its length;
//! - a piece of the Shared pool mapped into the guest at its guest-physical address, and sent
//!   back to the source (`pool`);
//! - the scanout set, at its region offset with its span, or blanked;
//! - the display's hold of the block of a surface it shows begun, and ended (`DisplayHold`).
//!
//! **What was in flight at a release's answer** ([`InFlight`]). A release's answer OK (a
//! `BLOCK_RELEASE`'s, a `POOL_GIVE`'s, a `PRIVATE_GIVE`'s) comes once every command buffer the
//! decoder began for the releasing context has finished, and the guest may then reuse the
//! block's memory. Its entry says what could still reach memory then: the holds of the other
//! contexts' work on their drains (`graphics::decode::Drain`), 0 where no command buffer of
//! another context was in flight, and the releasing context's queues with a command buffer
//! posted on their rings and not yet committed, which its drain does not wait for. It is read as
//! the answer is given, on the thread that gives it (the last command buffer's completion
//! thread, often), once a release: nothing is read per command buffer
//! (`graphics::decode::InFlightRead`).
//!
//! Recording an event takes the trail's lock once and copies the entry into a ring of [`LEN`]
//! entries, the runs inline: no allocation but the ring's own, once, and no lookup the event's
//! site did not make already, but a release answer's read of what was in flight.
//!
//! **When it is written.** At a device reset whose reset dropped live objects (the
//! `the device was reset: N objects of M contexts dropped` line, N above 0, which follows the
//! trail): a guest rebooting after a panic or the host's hard reset. The trail is written at info
//! under its own target, [`TARGET`], oldest first, one line an entry, each beginning `trail:`,
//! after a line that says how many follow, and is then empty. A reset that dropped nothing writes
//! nothing and keeps the trail.

use std::collections::VecDeque;
use std::fmt;

use reims_vgpu_vulkan::core::num::ToU64 as _;
use reims_vgpu_vulkan::core::platform::os::clock::Clock as _;
use reims_vgpu_vulkan::core::sync::Mutex;

/// The events the trail keeps: the last ones.
pub(crate) const LEN: usize = 512;

/// The guest-physical runs an entry keeps; past them it keeps how many runs and bytes there are.
pub(crate) const RUNS: usize = 16;

/// The trail's log target.
pub(crate) const TARGET: &str = "reims_vgpu::trail";

/// What an event did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// An import of a view of a block's runs made.
    ImportMade,
    /// A cached view's import taken by a block over the same runs.
    ImportTaken,
    /// The platform freed an import of a view, made for the block the entry names.
    ImportFreed,
    /// A `BLOCK_RELEASE` asked, of a block of `size` bytes.
    ReleaseAsked { size: u64 },
    /// A `BLOCK_RELEASE` answered, refused or OK, `waited_us` after it was asked; one answered OK
    /// with what was in flight then, where the decoder was there to read it.
    ReleaseAnswered {
        refused: bool,
        waited_us: u64,
        in_flight: Option<InFlight>,
    },
    /// A `POOL_GIVE` or `PRIVATE_GIVE` of `block`, of `length` bytes, answered OK, with what was
    /// in flight then, where the decoder was there to read it.
    GiveAnswered {
        block: Given,
        length: u64,
        in_flight: Option<InFlight>,
    },
    /// A piece of the Shared pool mapped into the guest.
    PoolMapped,
    /// A piece of the Shared pool sent back to the source.
    PoolBack,
    /// The scanout set, of `span` bytes; blanked where the entry names no offset.
    Scanout { span: u64 },
    /// The display's hold of `size` bytes of a block begun.
    HoldBegun { size: u64 },
    /// The display's hold of `size` bytes of a block ended.
    HoldEnded { size: u64 },
}

/// The block a give named.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Given {
    /// A block of the Shared pool, at its offset in the pool (`POOL_GIVE`).
    Pool { pool_offset: u64 },
    /// A Private block, at its GPU address (`PRIVATE_GIVE`).
    Private { gpu_address: u64 },
}

/// What of the guest's work was in flight as a release was answered (the module doc).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct InFlight {
    /// The holds on the other contexts' drains, each one's current drain and a closed one's last,
    /// but for each context's own hold of its current drain: one a command buffer of theirs begun
    /// and not through its completion handler, a message of theirs being carried out, a range
    /// message waiting for its commit, or a release's or close's work waiting on one; and one
    /// where an earlier drain of the context, which such a hold keeps, holds it. 0 where no
    /// command buffer of another context was in flight; past 0, holds, not command buffers.
    pub(crate) others: u64,
    /// The releasing context's queues on whose rings a command buffer was posted and is not yet
    /// committed (a queue counts once, however many it has).
    pub(crate) uncommitted: u64,
}

impl fmt::Display for InFlight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "in flight: {} holds in other contexts, {} of its queues uncommitted",
            self.others, self.uncommitted
        )
    }
}

/// Guest-physical runs, `(address, length)` each: the first [`RUNS`], with how many there are and
/// their bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Extents {
    runs: [(u64, u64); RUNS],
    /// The runs of `runs` kept.
    kept: usize,
    /// The runs in all.
    count: u64,
    /// Their bytes in all.
    bytes: u64,
}

impl Extents {
    /// No runs.
    const NONE: Extents = Extents {
        runs: [(0, 0); RUNS],
        kept: 0,
        count: 0,
        bytes: 0,
    };

    /// `runs`, the first [`RUNS`] kept.
    fn of(runs: impl IntoIterator<Item = (u64, u64)>) -> Extents {
        let mut extents = Extents::NONE;
        for (address, length) in runs {
            if let Some(slot) = extents.runs.get_mut(extents.kept) {
                *slot = (address, length);
                extents.kept += 1;
            }
            extents.count = extents.count.saturating_add(1);
            extents.bytes = extents.bytes.saturating_add(length);
        }
        extents
    }

    /// The runs kept.
    fn kept(&self) -> &[(u64, u64)] {
        self.runs.get(..self.kept).unwrap_or_default()
    }
}

impl fmt::Display for Extents {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (n, (address, length)) in self.kept().iter().enumerate() {
            let gap = if n == 0 { "" } else { " " };
            write!(f, "{gap}{address:#x}+{length:#x}")?;
        }
        if self.count > self.kept.to_u64() {
            write!(
                f,
                " ... of {} runs, {:#x} bytes in all",
                self.count, self.bytes
            )?;
        }
        Ok(())
    }
}

/// What an event names: the guest's context, the block's region offset and the guest-physical
/// runs, each where it has them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Held {
    context: Option<u32>,
    offset: Option<u64>,
    extents: Extents,
}

impl Held {
    /// Context `context`'s block at region offset `offset`, over the guest-physical `runs`.
    pub(crate) fn over(
        context: Option<u32>,
        offset: Option<u64>,
        runs: impl IntoIterator<Item = (u64, u64)>,
    ) -> Held {
        Held {
            context,
            offset,
            extents: Extents::of(runs),
        }
    }

    /// Context `context`'s block at region offset `offset`, with no runs.
    pub(crate) fn at(context: Option<u32>, offset: Option<u64>) -> Held {
        Held {
            context,
            offset,
            extents: Extents::NONE,
        }
    }
}

/// One event: when, what, and what it names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Entry {
    /// Milliseconds since the Unix epoch.
    at_ms: u64,
    kind: Kind,
    held: Held,
}

impl fmt::Display for Entry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Held {
            context,
            offset,
            extents,
        } = &self.held;
        write!(f, "trail: {}", self.at_ms)?;
        if let Some(context) = context {
            write!(f, ", context {context}")?;
        }
        if let Some(offset) = offset {
            write!(f, ", at {offset:#x}")?;
        }
        match self.kind {
            Kind::ImportMade => write!(f, ": a view's import made")?,
            Kind::ImportTaken => write!(f, ": a cached view's import taken")?,
            Kind::ImportFreed => write!(f, ": the import of a view made for it freed")?,
            Kind::ReleaseAsked { size } => write!(f, ": release asked, {size:#x} bytes")?,
            Kind::ReleaseAnswered {
                refused,
                waited_us,
                in_flight,
            } => {
                let answer = if refused { "REFUSED" } else { "OK" };
                write!(f, ": release answered {answer} after {waited_us} us")?;
                if let Some(in_flight) = in_flight {
                    write!(f, "; {in_flight}")?;
                }
            }
            Kind::GiveAnswered {
                block,
                length,
                in_flight,
            } => {
                match block {
                    Given::Pool { pool_offset } => write!(
                        f,
                        ": Shared pool give answered OK, {length:#x} bytes at pool offset \
                         {pool_offset:#x}"
                    )?,
                    Given::Private { gpu_address } => write!(
                        f,
                        ": Private give answered OK, {length:#x} bytes at GPU address \
                         {gpu_address:#x}"
                    )?,
                }
                if let Some(in_flight) = in_flight {
                    write!(f, "; {in_flight}")?;
                }
            }
            Kind::PoolMapped => write!(f, ": a Shared pool piece mapped into the guest")?,
            Kind::PoolBack => write!(f, ": a Shared pool piece goes back to the source")?,
            Kind::Scanout { .. } if offset.is_none() => write!(f, ": the scanout blanked")?,
            Kind::Scanout { span } => write!(f, ": the scanout set, {span:#x} bytes")?,
            Kind::HoldBegun { size } => write!(f, ": a display hold begun, {size:#x} bytes")?,
            Kind::HoldEnded { size } => write!(f, ": a display hold ended, {size:#x} bytes")?,
        }
        if extents.count > 0 {
            write!(f, ", over {extents}")?;
        }
        Ok(())
    }
}

/// The last [`LEN`] entries, the oldest first.
struct Trail {
    entries: VecDeque<Entry>,
}

impl Trail {
    /// No entries.
    const fn new() -> Trail {
        Trail {
            entries: VecDeque::new(),
        }
    }

    /// `entry`, the latest, the oldest let go of past [`LEN`]: the ring is made at the first
    /// entry, and after each [`Self::take`].
    fn push(&mut self, entry: Entry) {
        if self.entries.capacity() < LEN {
            self.entries
                .reserve_exact(LEN.saturating_sub(self.entries.len()));
        }
        if self.entries.len() >= LEN {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// Every entry, the oldest first; the trail is then empty.
    fn take(&mut self) -> VecDeque<Entry> {
        std::mem::take(&mut self.entries)
    }
}

/// The process's trail: one device's, as the process has one.
static TRAIL: Mutex<Trail> = Mutex::new(Trail::new());

/// An event of `kind`, naming `held`, now.
pub(crate) fn record(kind: Kind, held: Held) {
    let wall = reims_vgpu_vulkan::os::OsClock::new().now_wall();
    let entry = Entry {
        at_ms: wall.as_nanos() / 1_000_000,
        kind,
        held,
    };
    TRAIL.lock().push(entry);
}

/// Writes the trail, oldest first, and empties it (the module doc): at a device reset that
/// dropped live objects.
pub(crate) fn write() {
    write_from(&TRAIL);
}

/// Writes `trail`'s entries, oldest first, with no lock held, and empties it.
fn write_from(trail: &Mutex<Trail>) {
    let entries = trail.lock().take();
    if entries.is_empty() {
        return;
    }
    log::info!(
        target: TARGET,
        "trail: the last {} events that placed or took away a hold on the guest's memory, the \
         oldest first, at milliseconds since the Unix epoch",
        entries.len()
    );
    for entry in &entries {
        log::info!(target: TARGET, "{entry}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An entry of `kind` at `at_ms`, naming context 1's block at 0x1000 with no runs.
    fn entry(at_ms: u64, kind: Kind) -> Entry {
        Entry {
            at_ms,
            kind,
            held: Held::at(Some(1), Some(0x1000)),
        }
    }

    /// The ring keeps the last [`LEN`] entries, the oldest first, and is empty once taken
    #[test]
    fn the_trail_keeps_the_last_entries_oldest_first_until_taken() {
        let mut trail = Trail::new();
        for at in 0..LEN.to_u64() + 88 {
            trail.push(entry(at, Kind::ImportMade));
        }
        let taken = trail.take();
        assert_eq!(taken.len(), LEN);
        let times: Vec<u64> = taken.iter().map(|e| e.at_ms).collect();
        assert_eq!(times, (88..LEN.to_u64() + 88).collect::<Vec<_>>());
        assert!(trail.take().is_empty(), "taken, the trail is empty");
        trail.push(entry(7, Kind::PoolBack));
        assert_eq!(trail.take().len(), 1);
    }

    /// Recording an entry allocates nothing once the ring is made: its runs are kept inline
    #[test]
    fn recording_allocates_nothing_once_the_ring_is_made() {
        let mut trail = Trail::new();
        trail.push(entry(0, Kind::ImportMade));
        let runs = (0..40u64).map(|n| (n << 16, 0x4000));
        let ((), (allocations, _)) = crate::counted::allocator_calls_in(|| {
            for at in 0..LEN.to_u64() * 2 {
                trail.push(Entry {
                    at_ms: at,
                    kind: Kind::ImportFreed,
                    held: Held::over(Some(2), Some(0x2000), runs.clone()),
                });
            }
        });
        assert_eq!(allocations, 0);
    }

    /// An entry keeps the first [`RUNS`] runs, and says how many runs and bytes there are in all
    /// where there are more; one of fewer runs says no count
    #[test]
    fn an_entry_keeps_its_first_runs_and_counts_the_rest() {
        let many = Held::over(Some(3), Some(0x8000), (0..20u64).map(|n| (n << 20, 0x1000)));
        assert_eq!(many.extents.kept().len(), RUNS);
        assert_eq!((many.extents.count, many.extents.bytes), (20, 20 * 0x1000));
        let line = Entry {
            at_ms: 5,
            kind: Kind::ImportMade,
            held: many,
        }
        .to_string();
        assert!(
            line.starts_with(
                "trail: 5, context 3, at 0x8000: a view's import made, over 0x0+0x1000 \
                 0x100000+0x1000 "
            ),
            "{line}"
        );
        assert!(
            line.ends_with(" 0xf00000+0x1000 ... of 20 runs, 0x14000 bytes in all"),
            "{line}"
        );
        let few = Entry {
            at_ms: 6,
            kind: Kind::PoolMapped,
            held: Held::over(None, None, [(0x4_0000_0000, 0x20_0000)]),
        };
        assert_eq!(
            few.to_string(),
            "trail: 6: a Shared pool piece mapped into the guest, over 0x400000000+0x200000"
        );
    }

    /// A release's answer OK says what was in flight as it was given: the holds of other
    /// contexts' work, and the context's queues with a buffer posted past their commits; a
    /// refused one, or one answered with no decoder to read, says nothing of it. A give's answer
    /// names its block by its offset in the Shared pool or its GPU address, with its length
    #[test]
    fn a_release_answer_s_entry_says_what_was_in_flight() {
        let in_flight = Some(InFlight {
            others: 3,
            uncommitted: 1,
        });
        let line = |at_ms, kind, offset| {
            let held = Held::at(Some(2), offset);
            Entry { at_ms, kind, held }.to_string()
        };
        let answered = |refused, in_flight| Kind::ReleaseAnswered {
            refused,
            waited_us: 15,
            in_flight,
        };
        assert_eq!(
            line(1, answered(false, in_flight), Some(0x8000)),
            "trail: 1, context 2, at 0x8000: release answered OK after 15 us; in flight: 3 holds \
             in other contexts, 1 of its queues uncommitted"
        );
        assert_eq!(
            line(2, answered(true, None), Some(0x8000)),
            "trail: 2, context 2, at 0x8000: release answered REFUSED after 15 us"
        );
        let none = Some(InFlight {
            others: 0,
            uncommitted: 0,
        });
        let pool = Kind::GiveAnswered {
            block: Given::Pool {
                pool_offset: 0x20_0000,
            },
            length: 0x1_0000,
            in_flight: none,
        };
        assert_eq!(
            line(3, pool, None),
            "trail: 3, context 2: Shared pool give answered OK, 0x10000 bytes at pool offset \
             0x200000; in flight: 0 holds in other contexts, 0 of its queues uncommitted"
        );
        let private = Kind::GiveAnswered {
            block: Given::Private {
                gpu_address: 0x1_0000_0000,
            },
            length: 0x4000,
            in_flight: None,
        };
        assert_eq!(
            line(4, private, None),
            "trail: 4, context 2: Private give answered OK, 0x4000 bytes at GPU address \
             0x100000000"
        );
    }

    /// The trail is written at info under its own target, a line that counts the entries and
    /// then one line an entry, the oldest first, each beginning `trail:`; written, it is empty,
    /// and an empty trail writes nothing
    #[test]
    fn the_trail_is_written_oldest_first_and_emptied() {
        let trail = Mutex::new(Trail::new());
        trail
            .lock()
            .push(entry(1, Kind::ReleaseAsked { size: 0x4000 }));
        let answered = Kind::ReleaseAnswered {
            refused: false,
            waited_us: 40,
            in_flight: None,
        };
        trail.lock().push(entry(2, answered));
        let logs = reims_vgpu_vulkan::core::testkit::logs::capture();
        write_from(&trail);
        let lines = logs.take();
        assert!(
            lines
                .iter()
                .all(|(level, target, line)| *level == log::Level::Info
                    && target == TARGET
                    && line.starts_with("trail: ")),
            "{lines:?}"
        );
        let lines: Vec<&str> = lines.iter().map(|(_, _, line)| line.as_str()).collect();
        assert_eq!(
            lines[1..],
            [
                "trail: 1, context 1, at 0x1000: release asked, 0x4000 bytes",
                "trail: 2, context 1, at 0x1000: release answered OK after 40 us",
            ]
        );
        assert!(
            lines[0].starts_with("trail: the last 2 events "),
            "{}",
            lines[0]
        );
        assert!(
            trail.lock().take().is_empty(),
            "written, the trail is empty"
        );
        write_from(&trail);
        assert!(logs.take().is_empty(), "an empty trail writes nothing");
    }
}
