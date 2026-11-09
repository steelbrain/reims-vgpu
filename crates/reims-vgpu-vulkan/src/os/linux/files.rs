//! The process's backing table: which file, from which offset, each byte of the memory the
//! library may make a view of shows, so that a view is made by mapping files and never by
//! remapping addresses, which Linux has no call for (Mach's `mach_vm_remap`).
//!
//! Two kinds of entry:
//! - **Registrations** of memory the library does not map itself: the guest's RAM, a shared
//!   mapping of the files the host names in its memory map (the `guest_memory` lease of the
//!   memory map's device), and the memory of IOSurface's stand-in. One registered again over the
//!   same descriptor and offset is counted, and goes with the last. The C API duplicates the
//!   host's descriptors for each device it makes, so devices over the same RAM register it each
//!   under its own: a byte resolves through the first registration that holds it, and every one
//!   shows the same file pages.
//! - **Runs** the library mapped: every view [`super::vm::LinuxVm`] made, and every range of
//!   [`super::vm::allocate_aliasable`]. A run shows `end - first` bytes of its file from its
//!   offset, with the protection last set on it. Runs never overlap: a new run replaces what of
//!   another it covers, and an unmap or an overwrite drops what of the runs it covers.
//!
//! A byte in a run is found there first; a registration is looked at only for bytes no run
//! holds. Because a view of a view resolves to the file, never to the first view's address, the
//! second view keeps showing the pages after the first goes, as Mach's aliases do.
//!
//! **Mappings.** The table counts the kernel's mappings (VMAs) its runs take ([`mappings`]): one
//! per run, less one for each pair of neighbours the kernel holds as one, a run that starts where
//! the one before it ends, in the same file at the offset that one ends at, with the same
//! protection ([`joined`]). The kernel merges such neighbours whenever it maps or protects one of
//! them, and splits a mapping where part of it is unmapped or protected apart, as the table cuts
//! its runs, so the count follows every view, protect and unmap. Registrations map nothing.
//!
//! **Runs mapped through a dma-buf.** A view of many runs of file pages may be mapped as a few
//! dma-bufs of those pages instead, one mapping each whatever the pages' order
//! ([`super::vm`]'s module docs, "Views through dma-bufs"). Each of its runs still names the file
//! and offset it shows, which is what a view of it or a dma-buf of its pages is made of, and also
//! the dma-buf it is mapped through and its offset there ([`Through`]): two such neighbours are
//! one mapping where they lie in the same dma-buf, the second at the offset the first ends at,
//! with the same protection, whatever their files; a run mapped through a dma-buf and one mapped
//! from its file never are.
//!
//! The table is a leaf lock: nothing is called under it.

use std::collections::BTreeMap;
use std::os::fd::OwnedFd;
use std::sync::Arc;

use crate::core::num::ToU64 as _;
use crate::core::sync::Mutex;

use crate::os::GuestRamFile;

/// The file behind a range: an open descriptor, and the file offset of the range's first byte.
#[derive(Clone, Debug)]
pub(crate) struct Backing {
    /// The file, held while any entry or view names it.
    pub(crate) fd: Arc<OwnedFd>,
    /// The file offset of the range's first byte.
    pub(crate) offset: u64,
}

/// A run of the library's own mapping: its end, its file, the protection set on it (`PROT_*`),
/// and the dma-buf it is mapped through, if it is (module docs).
#[derive(Clone, Debug)]
struct Run {
    end: u64,
    backing: Backing,
    prot: i32,
    through: Option<Through>,
}

/// Where a run mapped through a dma-buf lies in it: the dma-buf, by a number of the process's
/// own (one per mapping of one, never used again), and the offset of the run's first byte there
/// (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Through {
    pub(crate) buf: u64,
    pub(crate) offset: u64,
}

impl Through {
    /// A number for a mapping of a new dma-buf, unique in the process.
    pub(crate) fn new_buf() -> u64 {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }
}

/// A registration: `first..end` shows `backing`, read and write, `count` times registered, `ram`
/// of them as the guest's RAM ([`lease_guest_memory`]).
#[derive(Debug)]
struct Registered {
    first: u64,
    end: u64,
    backing: Backing,
    count: usize,
    ram: usize,
}

/// One piece of a range as the table resolves it: `length` bytes, `at` bytes into the range,
/// showing `backing`, with protection `prot`.
#[derive(Clone, Debug)]
pub(crate) struct Piece {
    pub(crate) at: u64,
    pub(crate) length: u64,
    pub(crate) backing: Backing,
    pub(crate) prot: i32,
}

/// The table (module docs).
#[derive(Debug, Default)]
struct Table {
    registered: Vec<Registered>,
    runs: BTreeMap<u64, Run>,
    /// The pairs of neighbouring runs the kernel holds as one mapping ([`joined`]).
    joins: usize,
}

/// The process's table.
static TABLE: Mutex<Table> = Mutex::new(Table {
    registered: Vec::new(),
    runs: BTreeMap::new(),
    joins: 0,
});

/// Whether the kernel holds the run `a` from `a_first` and the run `b` from `b_first` as one
/// mapping: `b` starts where `a` ends, with the same protection, mapped from the same file at the
/// offset `a` ends at, or through the same dma-buf at the offset `a` ends at there (module docs).
fn joined(a_first: u64, a: &Run, b_first: u64, b: &Run) -> bool {
    let length = a.end - a_first;
    let follows = match (a.through, b.through) {
        (None, None) => {
            Arc::ptr_eq(&a.backing.fd, &b.backing.fd)
                && a.backing.offset + length == b.backing.offset
        }
        (Some(a), Some(b)) => a.buf == b.buf && a.offset + length == b.offset,
        _ => false,
    };
    a.end == b_first && a.prot == b.prot && follows
}

impl Table {
    /// The kernel's mappings the runs take (module docs).
    fn mappings(&self) -> usize {
        self.runs.len() - self.joins
    }

    /// How many of the runs on either side of `first..run.end` join a run there ([`joined`]).
    fn joins_beside(&self, first: u64, run: &Run) -> usize {
        let before = (self.runs.range(..first).next_back())
            .is_some_and(|(&at, before)| joined(at, before, first, run));
        let after = (self.runs.range(run.end..).next())
            .is_some_and(|(&at, after)| joined(first, run, at, after));
        usize::from(before) + usize::from(after)
    }

    /// Adds a run over bytes no run holds, counting the neighbours it joins.
    fn insert(&mut self, first: u64, run: Run) {
        self.joins += self.joins_beside(first, &run);
        self.runs.insert(first, run);
    }

    /// Removes the run from `first`, uncounting the neighbours it joined.
    fn remove(&mut self, first: u64) -> Option<Run> {
        let run = self.runs.remove(&first)?;
        self.joins -= self.joins_beside(first, &run);
        Some(run)
    }

    /// The run holding byte `at`: its first byte and itself.
    fn run_at(&self, at: u64) -> Option<(u64, &Run)> {
        (self.runs.range(..=at).next_back())
            .filter(|(_, run)| at < run.end)
            .map(|(&first, run)| (first, run))
    }

    /// The first run starting after `at`, before `end`: its first byte.
    fn next_run(&self, at: u64, end: u64) -> Option<u64> {
        self.runs.range(at..end).next().map(|(&first, _)| first)
    }

    /// The registration holding byte `at`.
    fn registered_at(&self, at: u64) -> Option<&Registered> {
        self.registered.iter().find(|r| r.first <= at && at < r.end)
    }

    /// Where the byte at `offset` of the file `fd` lies in a range of the guest's RAM a lease
    /// registered: its host address there, and the bytes from it to the range's end.
    fn ram_at(&self, fd: &Arc<OwnedFd>, offset: u64) -> Option<(u64, u64)> {
        (self.registered.iter())
            .filter(|r| r.ram > 0 && Arc::ptr_eq(&r.backing.fd, fd))
            .find_map(|r| {
                let into = offset.checked_sub(r.backing.offset)?;
                let length = r.end - r.first;
                (into < length).then(|| (r.first + into, length - into))
            })
    }

    /// The file behind byte `at`: the run or registration holding it, the file offset of `at`, its
    /// protection, and the end of the bytes from `at` that show the same file on, with the same
    /// protection, up to `end` (a registration's stop where a run starts); `None` where neither
    /// holds it.
    fn file_at(&self, at: u64, end: u64) -> Option<(&Arc<OwnedFd>, u64, i32, u64)> {
        if let Some((first, run)) = self.run_at(at) {
            let offset = run.backing.offset + (at - first);
            return Some((&run.backing.fd, offset, run.prot, run.end.min(end)));
        }
        let registered = self.registered_at(at)?;
        let to = (registered.end.min(end)).min(self.next_run(at, end).unwrap_or(end));
        let offset = registered.backing.offset + (at - registered.first);
        Some((&registered.backing.fd, offset, READ_WRITE, to))
    }

    /// Cuts the run holding `at` in two there, where `at` lies past its first byte: two runs the
    /// kernel holds as one mapping, so the count of mappings stays.
    fn cut(&mut self, at: u64) {
        let Some(first) = self.run_at(at).map(|(first, _)| first).filter(|&f| f < at) else {
            return;
        };
        let Some(run) = self.remove(first) else {
            return;
        };
        let tail = Run {
            end: run.end,
            backing: Backing {
                fd: Arc::clone(&run.backing.fd),
                offset: run.backing.offset + (at - first),
            },
            prot: run.prot,
            through: run.through.map(|t| Through {
                buf: t.buf,
                offset: t.offset + (at - first),
            }),
        };
        self.insert(first, Run { end: at, ..run });
        self.insert(at, tail);
    }

    /// Drops what of the runs lies in `first..end`, keeping the rest of each.
    fn forget(&mut self, first: u64, end: u64) {
        if first >= end {
            return;
        }
        self.cut(first);
        self.cut(end);
        while let Some(at) = self.runs.range(first..end).next().map(|(&at, _)| at) {
            self.remove(at);
        }
    }

    /// `first..end` shows `backing` with protection `prot` from now on, mapped through the
    /// dma-buf `through` names where it does (`record`).
    fn record(
        &mut self,
        first: u64,
        end: u64,
        backing: Backing,
        prot: i32,
        through: Option<Through>,
    ) {
        self.forget(first, end);
        if first < end {
            self.insert(
                first,
                Run {
                    end,
                    backing,
                    prot,
                    through,
                },
            );
        }
    }

    /// Appends to `into` the pieces of `length` bytes from `first` ([`pieces`]), the first `at`
    /// bytes into the range `into` gathers, joined to the last piece there where it follows on in
    /// its file with its protection; false where a byte has no file behind it, the pieces of the
    /// bytes before it appended.
    fn pieces_into(&self, first: u64, length: u64, at: u64, into: &mut Vec<Piece>) -> bool {
        let Some(end) = first.checked_add(length) else {
            return false;
        };
        let mut from = first;
        while from < end {
            let Some((fd, offset, prot, to)) = self.file_at(from, end) else {
                return false;
            };
            match into.last_mut() {
                Some(last)
                    if Arc::ptr_eq(&last.backing.fd, fd)
                        && last.backing.offset + last.length == offset
                        && last.prot == prot =>
                {
                    last.length += to - from;
                }
                _ => into.push(Piece {
                    at: at + (from - first),
                    length: to - from,
                    backing: Backing {
                        fd: Arc::clone(fd),
                        offset,
                    },
                    prot,
                }),
            }
            from = to;
        }
        true
    }

    /// Whether no run holds a byte of `first..end`.
    fn holds_none(&self, first: u64, end: u64) -> bool {
        self.run_at(first).is_none() && self.next_run(first, end).is_none()
    }

    /// [`Self::record`] of each of `pieces` from `target`, `piece.at` bytes in, over bytes no run
    /// holds ([`Self::holds_none`]), in order: nothing of another run to cut or drop, and each
    /// piece joins only the one before it, the first the run before the range and the last the
    /// run after it, as [`Self::insert`] counts them.
    fn record_fresh(&mut self, target: u64, pieces: Vec<Piece>) {
        let mut before: Option<u64> = (self.runs.range(..target).next_back()).map(|(&at, _)| at);
        let mut last: Option<u64> = None;
        for piece in pieces {
            let first = target.saturating_add(piece.at);
            let end = first.saturating_add(piece.length);
            if first >= end {
                continue;
            }
            let run = Run {
                end,
                backing: piece.backing,
                prot: piece.prot,
                through: None,
            };
            if let Some(at) = before.take()
                && let Some(prior) = self.runs.get(&at)
                && joined(at, prior, first, &run)
            {
                self.joins += 1;
            }
            self.runs.insert(first, run);
            before = Some(first);
            last = Some(first);
        }
        if let Some(at) = last
            && let Some(run) = self.runs.get(&at)
            && let Some((&next, after)) = self.runs.range(run.end..).next()
            && joined(at, run, next, after)
        {
            self.joins += 1;
        }
    }

    /// Sets the protection of what of the runs lies in `first..end` (`protect`), cutting a run it
    /// changes where the range ends inside it. A run that has the protection already stays as it
    /// is, as the kernel leaves a mapping that a protect does not change.
    fn protect(&mut self, first: u64, end: u64, prot: i32) {
        if first >= end {
            return;
        }
        for at in [first, end] {
            if self.run_at(at).is_some_and(|(_, run)| run.prot != prot) {
                self.cut(at);
            }
        }
        let mut from = first;
        while let Some(at) = (self.runs.range(from..end))
            .find(|(_, run)| run.prot != prot)
            .map(|(&at, _)| at)
        {
            let Some(mut run) = self.remove(at) else {
                break;
            };
            from = run.end;
            run.prot = prot;
            self.insert(at, run);
        }
    }

    /// Whether every byte of `first..end` has `bit` (`PROT_*`) in its protection: `None` where
    /// a byte lies in neither a run nor a registration.
    fn all_have(&self, first: u64, end: u64, bit: i32) -> Option<bool> {
        let mut all = true;
        let mut at = first;
        while at < end {
            let (_, _, prot, to) = self.file_at(at, end)?;
            all &= prot & bit != 0;
            at = to;
        }
        Some(all)
    }
}

/// The protection of a registration's bytes: read and write.
const READ_WRITE: i32 = libc::PROT_READ | libc::PROT_WRITE;

/// The kernel's mappings the table's runs take (module docs): every view the library made and
/// every range a view can be made of.
pub(crate) fn mappings() -> u64 {
    TABLE.lock().mappings().to_u64()
}

/// The kernel's mappings the table's runs in `first..end` take, as [`mappings`] counts them:
/// what a fold of them would free (`super::vm::fold_idle`), and in tests a range's count to set
/// beside the kernel's.
pub(crate) fn mappings_between(first: u64, end: u64) -> u64 {
    let table = TABLE.lock();
    let mut count = 0u64;
    let mut before: Option<(u64, &Run)> = None;
    for (&at, run) in table.runs.range(first..end) {
        if !before.is_some_and(|(b_at, b)| joined(b_at, b, at, run)) {
            count += 1;
        }
        before = Some((at, run));
    }
    count
}

/// The spans of the table's runs mapped from their files, in order, each of such runs that follow
/// one another in the address space with none through a dma-buf between: what a fold may map
/// through dma-bufs instead (`super::vm::fold_idle`).
pub(crate) fn file_spans() -> Vec<std::ops::Range<u64>> {
    let table = TABLE.lock();
    let mut spans: Vec<std::ops::Range<u64>> = Vec::new();
    for (&at, run) in &table.runs {
        if run.through.is_some() {
            continue;
        }
        match spans.last_mut() {
            Some(span) if span.end == at => span.end = run.end,
            _ => spans.push(at..run.end),
        }
    }
    spans
}

/// The runs the table holds in `first..end`, each a remap's piece or what a cut left of one
/// (tests).
#[cfg(test)]
pub(crate) fn runs_between(first: u64, end: u64) -> usize {
    TABLE.lock().runs.range(first..end).count()
}

/// `length` bytes from `first` show `backing` from now on, with protection `prot`: what of
/// another run they cover goes.
pub(crate) fn record(first: u64, length: u64, backing: Backing, prot: i32) {
    TABLE
        .lock()
        .record(first, first.saturating_add(length), backing, prot, None);
}

/// `length` bytes from `first` show no file from now on.
pub(crate) fn forget(first: u64, length: u64) {
    TABLE.lock().forget(first, first.saturating_add(length));
}

/// The pieces of `length` bytes from `first`, in order, each bytes of one file that follow on in
/// it with one protection, through the runs and registrations over them (one piece where the runs
/// a view of following pages left follow on, as the kernel maps them as one); `None` where a byte
/// has no file behind it.
pub(crate) fn pieces(first: u64, length: u64) -> Option<Vec<Piece>> {
    let mut pieces: Vec<Piece> = Vec::new();
    TABLE
        .lock()
        .pieces_into(first, length, 0, &mut pieces)
        .then_some(pieces)
}

/// How many sources [`view_pieces`] resolves under one hold of the table's lock: a view of a
/// surface's scattered pages has thousands, each one or two lookups, and the lock is taken by
/// every import's resolve of its pages too.
const SOURCES_HELD: usize = 256;

/// The pieces of a view of `sources`, each (address, length), in order: [`pieces`] of each, `at`
/// its bytes into the view, the pieces of one source and of its neighbours that follow on in one
/// file with one protection joined, as the kernel maps them as one. `Err` with the index of the
/// first source a byte of which has no file behind it, and the pieces of the sources before it.
pub(crate) fn view_pieces(sources: &[(u64, u64)]) -> Result<Vec<Piece>, (usize, Vec<Piece>)> {
    let mut pieces: Vec<Piece> = Vec::new();
    let mut at = 0u64;
    for (chunk, held) in sources.chunks(SOURCES_HELD).enumerate() {
        let table = TABLE.lock();
        for (i, &(first, length)) in held.iter().enumerate() {
            if !table.pieces_into(first, length, at, &mut pieces) {
                // What the source added before the byte with no file goes: its own pieces, and
                // what it joined to the last piece of the sources before it.
                pieces.truncate(pieces.partition_point(|p| p.at < at));
                if let Some(last) = pieces.last_mut() {
                    last.length = last.length.min(at - last.at);
                }
                return Err((chunk * SOURCES_HELD + i, pieces));
            }
            at += length;
        }
    }
    Ok(pieces)
}

/// `pieces` of a range, as [`pieces`] gave them, show from `target` on from now on, each
/// `piece.at` bytes in: what of other runs they cover goes (a remap's record, under one lock).
/// Over bytes no run holds, as a view's over core's fresh reservation, the pieces in order are
/// added as they are, with no run to cut ([`Table::record_fresh`]).
pub(crate) fn record_pieces(target: u64, pieces: Vec<Piece>) {
    let mut table = TABLE.lock();
    let end = (pieces.iter())
        .map(|p| target.saturating_add(p.at).saturating_add(p.length))
        .max();
    let in_order = (pieces.windows(2)).all(|w| w[0].at.saturating_add(w[0].length) <= w[1].at);
    if let Some(end) = end
        && in_order
        && table.holds_none(target, end)
    {
        table.record_fresh(target, pieces);
        return;
    }
    for piece in pieces {
        let first = target.saturating_add(piece.at);
        let end = first.saturating_add(piece.length);
        table.record(first, end, piece.backing, piece.prot, None);
    }
}

/// As [`record_pieces`], for pieces mapped through dma-bufs, each where its [`Through`] says
/// (module docs).
pub(crate) fn record_through(target: u64, pieces: Vec<(Piece, Through)>) {
    let mut table = TABLE.lock();
    for (piece, through) in pieces {
        let first = target.saturating_add(piece.at);
        let end = first.saturating_add(piece.length);
        table.record(first, end, piece.backing, piece.prot, Some(through));
    }
}

/// The runs in `first..end` mapped through a dma-buf, cut to it, in order, as pieces from
/// `first`: what an import's unfold maps from their files (`super::vm::unfold`).
pub(crate) fn through_pieces(first: u64, end: u64) -> Vec<Piece> {
    let table = TABLE.lock();
    let from = table.run_at(first).map_or(first, |(at, _)| at);
    (table.runs.range(from..end))
        .filter(|(_, run)| run.through.is_some() && run.end > first)
        .map(|(&at, run)| {
            let start = at.max(first);
            Piece {
                at: start - first,
                length: run.end.min(end) - start,
                backing: Backing {
                    fd: Arc::clone(&run.backing.fd),
                    offset: run.backing.offset + (start - at),
                },
                prot: run.prot,
            }
        })
        .collect()
}

/// Sets the protection of what of the runs lies in `length` bytes from `first` to `prot`,
/// cutting a run it changes at its ends.
pub(crate) fn protect(first: u64, length: u64, prot: i32) {
    TABLE
        .lock()
        .protect(first, first.saturating_add(length), prot);
}

/// Whether every byte of `length` bytes from `first` lies in a run whose protection is `prot`
/// already, so a protect of them to `prot` changes no mapping: the table's runs have the
/// protections the kernel's mappings of them have, as each map and protect of them records what
/// it set. Bytes of a registration, which the table does not map, never are.
pub(crate) fn protected(first: u64, length: u64, prot: i32) -> bool {
    let Some(end) = first.checked_add(length) else {
        return false;
    };
    let table = TABLE.lock();
    let mut at = first;
    while at < end {
        match table.run_at(at) {
            Some((_, run)) if run.prot == prot => at = run.end,
            _ => return false,
        }
    }
    true
}

/// Whether the table knows every byte of `length` bytes from `first`: `Some(readable)` where
/// each lies in a run or a registration, readable where each has read access; `None` where one
/// lies in neither.
pub(crate) fn readable(first: u64, length: u64) -> Option<bool> {
    let end = first.checked_add(length)?;
    TABLE.lock().all_have(first, end, libc::PROT_READ)
}

/// As [`readable`], for write access.
pub(crate) fn writable(first: u64, length: u64) -> Option<bool> {
    let end = first.checked_add(length)?;
    TABLE.lock().all_have(first, end, libc::PROT_WRITE)
}

/// Whether a byte of `length` bytes from `first` lies in a registration.
pub(crate) fn meets_registration(first: u64, length: u64) -> bool {
    let end = first.saturating_add(length);
    let table = TABLE.lock();
    table
        .registered
        .iter()
        .any(|r| r.first < end && first < r.end)
}

/// A registration of `length` bytes from `first` as showing `backing` (module docs), until it
/// drops.
#[derive(Debug)]
pub(crate) struct Registration {
    first: u64,
    end: u64,
    backing: Backing,
    /// Whether it registers a range of the guest's RAM ([`lease_guest_memory`]).
    ram: bool,
}

impl Registration {
    /// Registers `length` bytes from `first` as showing `backing`: counted again where the same
    /// bytes are registered over the same file and offset.
    pub(crate) fn new(first: u64, length: u64, backing: Backing) -> Registration {
        Registration::made(first, length, backing, false)
    }

    /// [`Registration::new`], of a range of the guest's RAM.
    fn guest_ram(first: u64, length: u64, backing: Backing) -> Registration {
        Registration::made(first, length, backing, true)
    }

    fn made(first: u64, length: u64, backing: Backing, ram: bool) -> Registration {
        let end = first.saturating_add(length);
        let mut table = TABLE.lock();
        match table
            .registered
            .iter_mut()
            .find(|r| r.is(first, end, &backing))
        {
            Some(r) => {
                r.count += 1;
                r.ram += usize::from(ram);
            }
            None => table.registered.push(Registered {
                first,
                end,
                backing: backing.clone(),
                count: 1,
                ram: usize::from(ram),
            }),
        }
        Registration {
            first,
            end,
            backing,
            ram,
        }
    }
}

/// The guest's RAM behind `length` bytes from `first`, as the leases registered it: runs of the
/// RAM's own host addresses, each an address and a length, in the bytes' order, neighbours that
/// follow on joined; `None` where a byte shows no page of the guest's RAM (memory of the
/// library's own, a file of zeros, an IOSurface stand-in's), shows nothing, or is mapped without
/// write permission: a device pins imported pages for writing, so read-only memory keeps the
/// imports it has (`super::import`'s module docs).
pub(crate) fn guest_ram_runs(first: u64, length: u64) -> Option<Vec<(u64, u64)>> {
    let end = first.checked_add(length)?;
    let table = TABLE.lock();
    let mut runs: Vec<(u64, u64)> = Vec::new();
    let mut at = first;
    while at < end {
        let (fd, offset, prot, to) = table.file_at(at, end)?;
        if prot & libc::PROT_WRITE == 0 {
            return None;
        }
        let (host, room) = table.ram_at(fd, offset)?;
        let take = (to - at).min(room);
        match runs.last_mut() {
            Some(last) if last.0 + last.1 == host => last.1 += take,
            _ => runs.push((host, take)),
        }
        at += take;
    }
    Some(runs)
}

/// The guest's RAM registered while a device made over it lives: the `guest_memory` lease its
/// device holds, made of the host's memory map, and dropped after core's teardown. Views of the
/// guest's pages are made of these files.
#[derive(Debug)]
pub struct GuestMemoryLease {
    /// The RAM as the devices that take it whole are handed it (`super::host_ram`), let go of
    /// first.
    _ram: Arc<super::host_ram::LeasedRam>,
    _registrations: Vec<Registration>,
}

/// Registers each range of the guest's RAM the host names a file behind: counted where the same
/// range was registered already over the same descriptor (module docs). Warns where the views'
/// budget of mappings is short of the guest's RAM (the `budget` module).
pub fn lease_guest_memory(ranges: impl IntoIterator<Item = GuestRamFile>) -> GuestMemoryLease {
    let ranges: Vec<GuestRamFile> = ranges.into_iter().collect();
    let ram: Vec<(u64, u64)> = ranges.iter().map(|r| (r.host, r.length)).collect();
    let registrations: Vec<Registration> = ranges
        .into_iter()
        .map(|r| {
            let backing = Backing {
                fd: r.fd,
                offset: r.offset,
            };
            Registration::guest_ram(r.host, r.length, backing)
        })
        .collect();
    let guest_ram = registrations
        .iter()
        .map(|r| r.end - r.first)
        .fold(0u64, u64::saturating_add);
    super::budget::warn_if_short(guest_ram);
    GuestMemoryLease {
        _ram: super::host_ram::lease(&ram),
        _registrations: registrations,
    }
}

impl Registered {
    /// Whether this registers `first..end` over `backing`.
    fn is(&self, first: u64, end: u64, backing: &Backing) -> bool {
        (self.first, self.end, self.backing.offset) == (first, end, backing.offset)
            && Arc::ptr_eq(&self.backing.fd, &backing.fd)
    }
}

impl Drop for Registration {
    /// Counts the registration down, and drops it at the last.
    fn drop(&mut self) {
        let mut table = TABLE.lock();
        if let Some(at) =
            (table.registered.iter()).position(|r| r.is(self.first, self.end, &self.backing))
        {
            let r = &mut table.registered[at];
            r.count -= 1;
            r.ram -= usize::from(self.ram);
            if r.count == 0 {
                table.registered.remove(at);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A file for the tests' backings.
    fn file() -> Arc<OwnedFd> {
        let name = c"rvg-files-test";
        // SAFETY: a NUL-terminated name; the descriptor is owned from here on.
        let fd = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
        assert!(fd >= 0);
        // SAFETY: a descriptor this test just made, owned once.
        Arc::new(unsafe { <OwnedFd as std::os::fd::FromRawFd>::from_raw_fd(fd) })
    }

    /// The pieces' (offset in the range, length, file offset).
    fn shape(pieces: &[Piece]) -> Vec<(u64, u64, u64)> {
        pieces
            .iter()
            .map(|p| (p.at, p.length, p.backing.offset))
            .collect()
    }

    // The table is the process's: each test uses address ranges of its own, far apart.

    /// A range resolves through the runs over it, cut where they end, then the registrations;
    /// a byte of neither leaves it unresolved
    #[test]
    fn a_range_resolves_through_runs_then_registrations() {
        let f = file();
        let base = 0x7a00_0000_0000;
        let backing = |offset| Backing {
            fd: Arc::clone(&f),
            offset,
        };
        let held = Registration::new(base, 0x10_000, backing(0x100_000));
        record(base + 0x20_000, 0x2000, backing(0x5000), libc::PROT_READ);
        assert_eq!(
            shape(&pieces(base + 0x1000, 0x2000).unwrap()),
            [(0, 0x2000, 0x101_000)]
        );
        assert_eq!(
            shape(&pieces(base + 0x20_000, 0x2000).unwrap()),
            [(0, 0x2000, 0x5000)]
        );
        assert!(pieces(base + 0x10_000, 0x1000).is_none());
        assert!(pieces(base + 0x1f_000, 0x2000).is_none());
        assert_eq!(readable(base + 0x20_000, 0x2000), Some(true));
        assert_eq!(readable(base, 0x10_000), Some(true));
        protect(base + 0x21_000, 0x1000, libc::PROT_NONE);
        assert_eq!(readable(base + 0x20_000, 0x1000), Some(true));
        assert_eq!(readable(base + 0x20_000, 0x2000), Some(false));
        assert_eq!(
            shape(&pieces(base + 0x20_000, 0x2000).unwrap()),
            [(0, 0x1000, 0x5000), (0x1000, 0x1000, 0x6000)]
        );
        assert_eq!(readable(base + 0x30_000, 1), None);
        drop(held);
        assert!(pieces(base, 0x1000).is_none());
        forget(base + 0x20_000, 0x2000);
    }

    /// A run replaces what of another it covers, and an unmap drops what it covers, keeping
    /// each run's other bytes at their file offsets
    #[test]
    fn runs_never_overlap() {
        let f = file();
        let base = 0x7b00_0000_0000;
        let backing = |offset| Backing {
            fd: Arc::clone(&f),
            offset,
        };
        record(
            base,
            0x4000,
            backing(0x10_000),
            libc::PROT_READ | libc::PROT_WRITE,
        );
        record(base + 0x1000, 0x1000, backing(0x90_000), libc::PROT_READ);
        assert_eq!(
            shape(&pieces(base, 0x4000).unwrap()),
            [
                (0, 0x1000, 0x10_000),
                (0x1000, 0x1000, 0x90_000),
                (0x2000, 0x2000, 0x12_000)
            ]
        );
        forget(base + 0x2000, 0x1000);
        assert!(pieces(base, 0x4000).is_none());
        assert_eq!(
            shape(&pieces(base + 0x3000, 0x1000).unwrap()),
            [(0, 0x1000, 0x13_000)]
        );
        forget(base, 0x4000);
        assert!(pieces(base, 1).is_none());
    }

    /// A lease registers each range of the guest's RAM at its file offset, until it drops
    #[test]
    fn a_lease_registers_the_guests_ram_while_held() {
        let f = file();
        let base = 0x7d00_0000_0000;
        let ram = |host, length, offset| GuestRamFile {
            host,
            guest: offset,
            length,
            fd: Arc::clone(&f),
            offset,
        };
        let lease =
            lease_guest_memory([ram(base, 0x4000, 0), ram(base + 0x10_000, 0x2000, 0x4000)]);
        assert_eq!(
            shape(&pieces(base + 0x1000, 0x1000).unwrap()),
            [(0, 0x1000, 0x1000)]
        );
        assert_eq!(
            shape(&pieces(base + 0x11_000, 0x1000).unwrap()),
            [(0, 0x1000, 0x5000)]
        );
        assert!(pieces(base + 0x4000, 0x1000).is_none());
        drop(lease);
        assert!(pieces(base, 0x1000).is_none());
        assert!(pieces(base + 0x10_000, 0x1000).is_none());
    }

    /// The guest's RAM behind a view resolves to the leased ranges' own addresses, a run per stretch
    /// that follows on in one range, in the view's order; bytes of other memory resolve to none
    #[test]
    fn a_views_bytes_resolve_to_the_guests_ram() {
        let f = file();
        let other = file();
        let base = 0x7e00_0000_0000;
        let ram = |host, length, offset| GuestRamFile {
            host,
            guest: offset,
            length,
            fd: Arc::clone(&f),
            offset,
        };
        let lease =
            lease_guest_memory([ram(base, 0x4000, 0), ram(base + 0x10_000, 0x4000, 0x4000)]);
        let backing = |offset| Backing {
            fd: Arc::clone(&f),
            offset,
        };
        // A view of five pages: the second range's first two, then the first range's last page,
        // then its first two backwards.
        let view = base + 0x40_000;
        let rw = libc::PROT_READ | libc::PROT_WRITE;
        record(view, 0x2000, backing(0x4000), rw);
        record(view + 0x2000, 0x1000, backing(0x3000), rw);
        record(view + 0x3000, 0x1000, backing(0x1000), rw);
        record(view + 0x4000, 0x1000, backing(0), rw);
        assert_eq!(
            guest_ram_runs(view + 0x800, 0x4000),
            Some(vec![
                (base + 0x10_800, 0x1800),
                (base + 0x3000, 0x1000),
                (base + 0x1000, 0x1000),
                (base, 0x800)
            ])
        );
        // The RAM's own bytes resolve to themselves, across a page.
        assert_eq!(
            guest_ram_runs(base + 0xff0, 0x20),
            Some(vec![(base + 0xff0, 0x20)])
        );
        // A read-only page of the RAM is left out, as is a page of another file, or of nothing.
        protect(view + 0x2000, 0x1000, libc::PROT_READ);
        assert_eq!(guest_ram_runs(view, 0x3000), None);
        protect(view + 0x2000, 0x1000, rw);
        record(
            view + 0x5000,
            0x1000,
            Backing {
                fd: Arc::clone(&other),
                offset: 0,
            },
            rw,
        );
        assert_eq!(guest_ram_runs(view + 0x4000, 0x2000), None);
        assert_eq!(guest_ram_runs(view + 0x6000, 0x1000), None);
        // A stand-in's registration, of a file of its own, is no RAM either.
        let stand_in = Registration::new(
            base + 0x80_000,
            0x1000,
            Backing {
                fd: Arc::clone(&other),
                offset: 0,
            },
        );
        assert_eq!(guest_ram_runs(base + 0x80_000, 0x1000), None);
        drop(stand_in);
        drop(lease);
        assert_eq!(guest_ram_runs(view, 0x1000), None);
        forget(view, 0x6000);
    }

    /// The table counts its runs' mappings as the kernel holds them: neighbours in one file at
    /// following offsets with one protection are one, and a protect or an unmap inside one cuts
    /// it
    #[test]
    fn the_table_counts_the_kernels_mappings() {
        let (f, g) = (file(), file());
        let mut t = Table::default();
        let base = 0x1000_0000;
        let page = 0x1000;
        let rw = libc::PROT_READ | libc::PROT_WRITE;
        let at = |i: u64| base + i * page;
        let b = |fd: &Arc<OwnedFd>, offset| Backing {
            fd: Arc::clone(fd),
            offset,
        };
        // Four pages of one file in order: one mapping.
        for i in 0..4 {
            t.record(at(i), at(i + 1), b(&f, i * page), rw, None);
        }
        assert_eq!(t.mappings(), 1);
        // A page of the file out of order after them, then a page of another file: three.
        t.record(at(4), at(5), b(&f, 0x9000), rw, None);
        t.record(at(5), at(6), b(&g, 0xa000), rw, None);
        assert_eq!(t.mappings(), 3);
        // The second page read-only cuts the first mapping in three; read and write again, one.
        t.protect(at(1), at(2), libc::PROT_READ);
        assert_eq!(t.mappings(), 5);
        t.protect(at(1), at(2), rw);
        assert_eq!(t.mappings(), 3);
        // An unmap of the second page cuts it in two; the page mapped again as it was, one.
        t.forget(at(1), at(2));
        assert_eq!(t.mappings(), 4);
        t.record(at(1), at(2), b(&f, page), rw, None);
        assert_eq!(t.mappings(), 3);
        // Another page of the file over the third: the first mapping in three.
        t.record(at(2), at(3), b(&f, 0x20_000), rw, None);
        assert_eq!(t.mappings(), 5);
        t.forget(base, at(6));
        assert_eq!((t.mappings(), t.joins, t.runs.len()), (0, 0, 0));
    }

    /// Runs mapped through a dma-buf are one mapping where each lies in the same dma-buf at the
    /// offset the one before ends at, whatever their files: a protect or an unmap cuts it as it
    /// cuts a file's mapping, and a run of the dma-buf never joins one mapped from its file
    #[test]
    fn runs_through_a_dma_buf_count_as_its_mappings() {
        let (f, g) = (file(), file());
        let mut t = Table::default();
        let page = 0x1000;
        let at = |i: u64| 0x3000_0000 + i * page;
        let rw = libc::PROT_READ | libc::PROT_WRITE;
        let b = |fd: &Arc<OwnedFd>, offset| Backing {
            fd: Arc::clone(fd),
            offset,
        };
        // Pages of two files backwards, through one dma-buf, then through a second: two mappings.
        let (one, two) = (Through::new_buf(), Through::new_buf());
        for i in 0..6 {
            let fd = if i % 2 == 0 { &f } else { &g };
            let (buf, offset) = if i < 4 {
                (one, i * page)
            } else {
                (two, (i - 4) * page)
            };
            let through = Through { buf, offset };
            t.record(at(i), at(i + 1), b(fd, (9 - i) * page), rw, Some(through));
        }
        assert_eq!(t.mappings(), 2);
        // Its second page read-only: three more; read and write again, as before.
        t.protect(at(1), at(2), libc::PROT_READ);
        assert_eq!(t.mappings(), 4);
        t.protect(at(1), at(2), rw);
        assert_eq!(t.mappings(), 2);
        // A cut keeps the count, and the tail lies further into the dma-buf.
        t.cut(at(0) + page / 2);
        assert_eq!(t.mappings(), 2);
        let tail = t.run_at(at(0) + page / 2).unwrap().1.through.unwrap();
        assert_eq!(
            tail,
            Through {
                buf: one,
                offset: page / 2
            }
        );
        // The third page unmapped, then mapped from its file at the offset that follows the
        // second's: apart from both, as the kernel holds a file's mapping and a dma-buf's.
        t.forget(at(2), at(3));
        assert_eq!(t.mappings(), 3);
        t.record(at(2), at(3), b(&f, 7 * page), rw, None);
        assert_eq!(t.mappings(), 4);
        t.forget(at(0), at(6));
        assert_eq!((t.mappings(), t.joins, t.runs.len()), (0, 0, 0));
    }

    /// A protect to the protection a run has leaves it whole; one that changes part of a run cuts
    /// it there, and the runs it leaves count as the mappings the kernel holds
    #[test]
    fn a_protect_cuts_only_the_runs_it_changes() {
        let f = file();
        let mut t = Table::default();
        let at = |i: u64| 0x2000_0000 + i * 0x1000;
        let rw = libc::PROT_READ | libc::PROT_WRITE;
        let backing = Backing {
            fd: Arc::clone(&f),
            offset: 0,
        };
        t.record(at(0), at(4), backing, rw, None);
        t.protect(at(1), at(3), rw);
        assert_eq!((t.runs.len(), t.mappings()), (1, 1), "unchanged");
        t.protect(at(1), at(2), libc::PROT_READ);
        assert_eq!((t.runs.len(), t.mappings()), (3, 3), "its second page");
        t.protect(at(0), at(4), libc::PROT_READ);
        assert_eq!(t.mappings(), 1, "all of it read-only: one mapping");
        assert!(t.runs.values().all(|run| run.prot == libc::PROT_READ));
        t.forget(at(0), at(4));
        assert_eq!((t.mappings(), t.joins, t.runs.len()), (0, 0, 0));
    }

    /// Runs that follow on in one file with one protection resolve as one piece, as the kernel
    /// maps them as one; a protection or a file apart cuts it
    #[test]
    fn runs_that_follow_on_resolve_as_one_piece() {
        let (f, g) = (file(), file());
        let base = 0x7e00_0000_0000;
        let rw = libc::PROT_READ | libc::PROT_WRITE;
        let backing = |fd: &Arc<OwnedFd>, offset| Backing {
            fd: Arc::clone(fd),
            offset,
        };
        for i in 0..4 {
            record(
                base + i * 0x1000,
                0x1000,
                backing(&f, 0x8000 + i * 0x1000),
                rw,
            );
        }
        assert_eq!(
            shape(&pieces(base + 0x800, 0x3000).unwrap()),
            [(0, 0x3000, 0x8800)]
        );
        protect(base + 0x2000, 0x1000, libc::PROT_READ);
        record(base + 0x3000, 0x1000, backing(&g, 0xb000), rw);
        assert_eq!(
            shape(&pieces(base, 0x4000).unwrap()),
            [
                (0, 0x2000, 0x8000),
                (0x2000, 0x1000, 0xa000),
                (0x3000, 0x1000, 0xb000)
            ]
        );
        forget(base, 0x4000);
    }

    /// A range is protected where every byte lies in a run of that protection: not where one
    /// lies in a registration or in neither
    #[test]
    fn protected_is_runs_of_the_protection_alone() {
        let f = file();
        let base = 0x7f00_0000_0000;
        let rw = libc::PROT_READ | libc::PROT_WRITE;
        let backing = || Backing {
            fd: Arc::clone(&f),
            offset: 0,
        };
        record(base, 0x2000, backing(), rw);
        record(base + 0x2000, 0x1000, backing(), rw);
        assert!(protected(base, 0x3000, rw));
        assert!(!protected(base, 0x3000, libc::PROT_READ));
        assert!(!protected(base, 0x4000, rw), "a byte of no run");
        let held = Registration::new(base + 0x3000, 0x1000, backing());
        assert!(!protected(base, 0x4000, rw), "a byte of a registration");
        assert_eq!(writable(base, 0x4000), Some(true));
        protect(base + 0x1000, 0x1000, libc::PROT_READ);
        assert_eq!(
            (readable(base, 0x4000), writable(base, 0x4000)),
            (Some(true), Some(false))
        );
        assert_eq!(writable(base, 0x5000), None);
        drop(held);
        forget(base, 0x3000);
    }

    /// The same bytes registered again over the same file and offset are counted: they stay
    /// until the last registration drops
    #[test]
    fn registrations_are_counted() {
        let f = file();
        let base = 0x7c00_0000_0000;
        let backing = || Backing {
            fd: Arc::clone(&f),
            offset: 0,
        };
        let first = Registration::new(base, 0x1000, backing());
        let second = Registration::new(base, 0x1000, backing());
        drop(first);
        assert!(pieces(base, 0x1000).is_some());
        drop(second);
        assert!(pieces(base, 0x1000).is_none());
    }

    /// A view's pieces recorded over bytes no run holds leave the table as recording them one at
    /// a time does: the same runs, and the same mappings counted, the neighbours before and after
    /// the range joining its first and last pieces in some rounds, and pieces that follow on in a
    /// file, a file apart or a protection apart in between
    #[test]
    fn a_fresh_views_record_is_the_one_at_a_time_record() {
        let (f, g) = (file(), file());
        let page = 0x1000;
        let at = |i: u64| 0x5000_0000 + i * page;
        let rw = libc::PROT_READ | libc::PROT_WRITE;
        let b = |fd: &Arc<OwnedFd>, offset| Backing {
            fd: Arc::clone(fd),
            offset,
        };
        let shape = |t: &Table| {
            let runs: Vec<(u64, u64, bool, u64, i32)> = (t.runs.iter())
                .map(|(&first, r)| {
                    (
                        first,
                        r.end,
                        Arc::ptr_eq(&r.backing.fd, &f),
                        r.backing.offset,
                        r.prot,
                    )
                })
                .collect();
            (runs, t.joins, t.mappings())
        };
        let mut seed: u64 = 0x9e37_79b9_7f4a_7c15;
        let mut next = move |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        for _ in 0..200 {
            // The pieces of pages 2 to 30, one to three pages each.
            let mut pieces = Vec::new();
            let mut page_at = 2;
            let mut offset = 0x40 * page;
            while page_at < 30 {
                let pages = (1 + next(3)).min(30 - page_at);
                let fd = if next(5) == 0 { &g } else { &f };
                offset = match next(3) {
                    0 => offset,
                    _ => next(0x40) * page,
                };
                let prot = if next(6) == 0 { libc::PROT_READ } else { rw };
                pieces.push(Piece {
                    at: (page_at - 2) * page,
                    length: pages * page,
                    backing: b(fd, offset),
                    prot,
                });
                offset += pages * page;
                page_at += pages;
            }
            // Neighbours: page 1 before the range and page 30 after it, each joining the piece
            // beside it in some rounds; page 0 apart.
            let first = &pieces[0];
            let last = &pieces[pieces.len() - 1];
            let before = match next(2) {
                0 => first.backing.offset.saturating_sub(page),
                _ => 0x80 * page,
            };
            let after = match next(2) {
                0 => last.backing.offset + last.length,
                _ => 0x90 * page,
            };
            let mut one = Table::default();
            let mut fresh = Table::default();
            for t in [&mut one, &mut fresh] {
                t.record(at(0), at(1), b(&g, 0xa0 * page), rw, None);
                t.record(at(1), at(2), b(&first.backing.fd, before), first.prot, None);
                t.record(at(30), at(31), b(&last.backing.fd, after), last.prot, None);
            }
            assert!(fresh.holds_none(at(2), at(30)));
            for piece in pieces.clone() {
                let start = at(2) + piece.at;
                one.record(start, start + piece.length, piece.backing, piece.prot, None);
            }
            fresh.record_fresh(at(2), pieces);
            assert_eq!(shape(&fresh), shape(&one));
        }
    }

    /// A view's sources resolve as each would alone, laid out one after another in the view and
    /// joined where they follow on in a file across sources; a source with a byte of no file
    /// answers its index and the pieces of the sources before it, a piece that ran into it cut
    /// back
    #[test]
    fn a_views_sources_resolve_in_one_pass() {
        let f = file();
        let base = 0x7980_0000_0000;
        let page = 0x1000;
        let backing = |offset| Backing {
            fd: Arc::clone(&f),
            offset,
        };
        let held = Registration::new(base, 0x10 * page, backing(0x100 * page));
        let src = |i: u64| base + i * page;
        // Pages 3, 4 (following on), 2 (backwards), 9 and 10 as one source.
        let sources = [
            (src(3), page),
            (src(4), page),
            (src(2), page),
            (src(9), 2 * page),
        ];
        let all = view_pieces(&sources).unwrap();
        assert_eq!(
            shape(&all),
            [
                (0, 2 * page, 0x103 * page),
                (2 * page, page, 0x102 * page),
                (3 * page, 2 * page, 0x109 * page)
            ]
        );
        // Many sources, past one hold of the lock: as each alone.
        let many: Vec<(u64, u64)> = (0..(3 * SOURCES_HELD as u64))
            .map(|i| (src(i % 0x10), page))
            .collect();
        let mut alone = Vec::new();
        for (i, &(first, length)) in many.iter().enumerate() {
            for p in pieces(first, length).unwrap() {
                alone.push((i as u64 * page + p.at, p.length, p.backing.offset));
            }
        }
        let joined: Vec<(u64, u64, u64)> = shape(&view_pieces(&many).unwrap());
        let total: u64 = joined.iter().map(|&(_, length, _)| length).sum();
        assert_eq!(total, many.len() as u64 * page);
        for &(at, length, offset) in &alone {
            assert!(
                joined.iter().any(|&(j, jl, jo)| j <= at
                    && at + length <= j + jl
                    && jo + (at - j) == offset),
                "the page {at:#x} of the view shows the file at {offset:#x}"
            );
        }
        // Page 0x10 has no file: the index of the source that runs into it, and the pieces
        // before it, the second source's piece, which the third's first page joined, cut back to
        // its own page.
        let refused = [
            (src(5), page),
            (src(0xe), page),
            (src(0xf), 2 * page),
            (src(1), page),
        ];
        let (i, before) = view_pieces(&refused).unwrap_err();
        assert_eq!(i, 2);
        assert_eq!(
            shape(&before),
            [(0, page, 0x105 * page), (page, page, 0x10e * page)]
        );
        drop(held);
    }
}
