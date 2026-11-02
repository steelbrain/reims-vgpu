//! The dirty source's side on Linux: which guest pages lie behind an import, and the per-import
//! answers the runtime asks of a device's dirty log (`reims_vgpu_core::memory::dirty`).
//!
//! **The harvester's interface** is the log's own ([`DirtyLog`], re-exported here): the device
//! makes it over its RAM with a [`HarvestWanted`] hook, and the host's harvester runs
//! [`DirtyLog::begin`], clears the watched runs it gives in its own log, and publishes with
//! [`DirtyLog::end`], one bitmap word per [`GRANULE`].
//!
//! **Behind an import.** A view of the guest's pages maps the files the host named for the
//! guest's RAM, so the backing table (`files`) resolves each byte of an import to a file and an
//! offset. The lease of a device made with a dirty source registers here each range of its RAM:
//! its file, the file offsets it covers, the guest physical address of the first, and the
//! device's log. A piece of an import over a registered range resolves to guest physical
//! addresses; any other byte (memory the library allocated, a file of zeros, another device's
//! RAM) resolves to none, and reads as written at every stamp.
//!
//! **Per import** ([`ImportDirty`], the runtime's `DirtyPages`): the runtime's questions, in
//! offsets from the import's pointer. Nothing is watched until the runtime watches a range (each
//! copy of the bytes it keeps, for the copy's life), so an import read in place costs the guest
//! nothing: the host re-protects only watched granules. A watch's range is resolved once, as it
//! begins; `changed` answers within the live watches only.
//!
//! **Writers.** The runtime's named writes ([`DirtyLog::wrote_as`]) and questions
//! ([`DirtyLog::changed_except`]) go to the log, so a shadow's own copy back or merge never comes
//! back to it as a change, while the guest's and any other writer's do.
//! `REIMS_VGPU_DIRTY_WRITERS=0` answers them as the plain ones, every write counting for every
//! asker.
//!
//! **Host writes.** The resolver ([`resolve`]) is core's (`memory::dirty::install_resolver`): it
//! stamps each write core makes through `GuestPtr`, and each the platform makes into the guest's
//! RAM by a host address (`memory::dirty::host_wrote` and `host_write`, at each such site).
//!
//! The registry's lock is a leaf: logs are called after it is let go.

use std::num::NonZeroU64;
use std::ops::Range;
use std::os::fd::OwnedFd;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use metal2vulkan_runtime as rt;
pub use reims_vgpu_core::memory::dirty::{
    DirtyCounts, DirtyLog, DirtyStamp, GRANULE, HarvestWanted, PAGE_SIZE, WAIT_TIMEOUT,
};
use reims_vgpu_core::memory::dirty::{Segment, install_resolver, set_resolving};
use reims_vgpu_core::sync::Mutex;

use super::{LinuxClock, files};
use crate::os::GuestRamFile;

/// One range of a device's RAM, registered by its lease.
#[derive(Debug)]
struct Entry {
    /// The registration's number, which its [`Registered`] drops.
    id: u64,
    fd: Arc<OwnedFd>,
    /// The file offsets the range covers.
    file: Range<u64>,
    /// The guest physical address of `file.start`.
    guest: u64,
    log: Weak<DirtyLog>,
}

static REGISTRY: Mutex<Vec<Entry>> = Mutex::new(Vec::new());
static NEXT_ID: AtomicU64 = AtomicU64::new(0);

/// The ranges of one device's RAM registered for its log, until it drops.
#[derive(Debug)]
pub(crate) struct Registered {
    ids: Vec<u64>,
}

/// Registers `ranges` of a device's RAM for `log` (module docs), and installs the resolver where
/// it is not yet.
pub(crate) fn register(ranges: &[GuestRamFile], log: &Arc<DirtyLog>) -> Registered {
    install_resolver(Box::new(resolve));
    let mut registry = REGISTRY.lock();
    let ids = ranges
        .iter()
        .map(|r| {
            let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
            registry.push(Entry {
                id,
                fd: Arc::clone(&r.fd),
                file: r.offset..r.offset.saturating_add(r.length),
                guest: r.guest,
                log: Arc::downgrade(log),
            });
            id
        })
        .collect();
    set_resolving(!registry.is_empty());
    Registered { ids }
}

impl Drop for Registered {
    fn drop(&mut self) {
        let mut registry = REGISTRY.lock();
        registry.retain(|e| !self.ids.contains(&e.id));
        set_resolving(!registry.is_empty());
    }
}

/// One resolved piece of a host range: `length` bytes, `at` bytes into it, in the guest's RAM
/// from `guest` of the device whose log is `log`, or in none.
struct Resolved {
    at: u64,
    length: u64,
    ram: Option<(Arc<DirtyLog>, u64)>,
}

/// A piece's device log and guest address, before the log is upgraded.
type WeakRam = (Weak<DirtyLog>, u64);

/// The `length` bytes at host address `at`, piece by piece: empty where a byte has no file.
fn resolved(at: u64, length: u64) -> Vec<Resolved> {
    let Some(pieces) = files::pieces(at, length) else {
        return Vec::new();
    };
    // Each piece as `Resolved`'s parts, its log not yet upgraded: that waits for the lock's drop.
    let mut found: Vec<(u64, u64, Option<WeakRam>)> = Vec::new();
    {
        let registry = REGISTRY.lock();
        for piece in &pieces {
            let first = piece.backing.offset;
            let end = first.saturating_add(piece.length);
            let mut offset = first;
            while offset < end {
                let entry = (registry.iter())
                    .find(|e| Arc::ptr_eq(&e.fd, &piece.backing.fd) && e.file.contains(&offset));
                let (to, ram) = match entry {
                    Some(e) => (
                        end.min(e.file.end),
                        Some((
                            Weak::clone(&e.log),
                            e.guest.saturating_add(offset.saturating_sub(e.file.start)),
                        )),
                    ),
                    None => {
                        let next = (registry.iter())
                            .filter(|e| Arc::ptr_eq(&e.fd, &piece.backing.fd))
                            .map(|e| e.file.start)
                            .filter(|&s| s > offset)
                            .min()
                            .unwrap_or(end);
                        (end.min(next), None)
                    }
                };
                found.push((
                    piece.at.saturating_add(offset.saturating_sub(first)),
                    to.saturating_sub(offset),
                    ram,
                ));
                offset = to;
            }
        }
    }
    found
        .into_iter()
        .map(|(at, length, ram)| Resolved {
            at,
            length,
            ram: ram.and_then(|(log, guest)| Some((log.upgrade()?, guest))),
        })
        .collect()
}

/// Core's resolver: the guest physical runs behind `length` bytes at host address `at`, each
/// with its device's log.
fn resolve(at: u64, length: u64, f: &mut dyn FnMut(&Arc<DirtyLog>, Range<u64>)) {
    for r in resolved(at, length) {
        if let Some((log, guest)) = &r.ram {
            f(log, *guest..guest.saturating_add(r.length));
        }
    }
}

/// Whether the writers' answers are on (`REIMS_VGPU_DIRTY_WRITERS=0` turns them off).
fn writers() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| writers_on(std::env::var_os("REIMS_VGPU_DIRTY_WRITERS").as_deref()))
}

/// The writers' switch as `REIMS_VGPU_DIRTY_WRITERS` reads: off at `0` alone.
fn writers_on(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_none_or(|v| v != "0")
}

/// One live watch of an import: its pieces as segments of the import.
#[derive(Debug)]
struct Watched {
    id: u64,
    segments: Arc<[Segment]>,
}

#[derive(Debug)]
struct Shared {
    log: Arc<DirtyLog>,
    /// The host address of the import's pointer, as the backing table knows it.
    at: u64,
    watches: Mutex<Vec<Watched>>,
}

/// The runtime's questions about one import (module docs), over its device's log.
#[derive(Debug)]
pub struct ImportDirty {
    shared: Arc<Shared>,
}

/// A watch of an import's range, ended by its drop.
#[derive(Debug)]
pub struct ImportWatch {
    shared: Arc<Shared>,
    id: u64,
    /// The guest physical runs it watches in the log.
    runs: Vec<Range<u64>>,
}

impl Drop for ImportWatch {
    fn drop(&mut self) {
        self.shared.log.unwatch(&self.runs);
        self.shared.watches.lock().retain(|w| w.id != self.id);
    }
}

impl ImportDirty {
    /// The answers for an import whose pointer is host address `at` (as the backing table knows
    /// it), or `None` where its first byte is in no guest RAM of a device with a dirty log: then
    /// the runtime copies it whole, as before.
    pub fn new(at: u64) -> Option<ImportDirty> {
        let first = resolved(at, 1).into_iter().next()?;
        let (log, _) = first.ram?;
        Some(ImportDirty {
            shared: Arc::new(Shared {
                log,
                at,
                watches: Mutex::new(Vec::new()),
            }),
        })
    }

    /// The answers for the whole of a guest's RAM, its offsets host addresses (the runtime's
    /// `HostRam::with_dirty_pages`, whose log's pointer is 0), where host address `first`, a byte
    /// of the RAM, lies in the RAM of a device with a dirty log; `None` otherwise.
    pub(crate) fn of_ram(first: u64) -> Option<ImportDirty> {
        let first = resolved(first, 1).into_iter().next()?;
        let (log, _) = first.ram?;
        Some(ImportDirty {
            shared: Arc::new(Shared {
                log,
                at: 0,
                watches: Mutex::new(Vec::new()),
            }),
        })
    }

    /// The import's bytes `range` as segments of the import, in this log's RAM or none.
    fn segments(&self, range: &Range<u64>) -> Vec<Segment> {
        let length = range.end.saturating_sub(range.start);
        let pieces = resolved(self.shared.at.saturating_add(range.start), length);
        if pieces.is_empty() {
            return vec![Segment {
                offset: range.start,
                length,
                guest: None,
            }];
        }
        pieces
            .into_iter()
            .map(|r| Segment {
                offset: range.start.saturating_add(r.at),
                length: r.length,
                guest: r
                    .ram
                    .filter(|(log, _)| Arc::ptr_eq(log, &self.shared.log))
                    .map(|(_, guest)| guest),
            })
            .collect()
    }

    /// Appends the changed runs of the live watches, by anyone or by anyone but `writer`.
    fn changed_by(&self, since: DirtyStamp, writer: Option<NonZeroU64>, out: &mut Vec<Range<u64>>) {
        let watched: Vec<Arc<[Segment]>> = (self.shared.watches.lock().iter())
            .map(|w| Arc::clone(&w.segments))
            .collect();
        let log = &self.shared.log;
        let ask = |segments: &[Segment], out: &mut Vec<Range<u64>>| match writer {
            Some(w) => log.changed_except(segments, since, w, out),
            None => log.changed(segments, since, out),
        };
        match watched.as_slice() {
            [] => {}
            [one] => ask(one, out),
            many => {
                let mut all = Vec::new();
                for segments in many {
                    ask(segments, &mut all);
                }
                all.sort_by_key(|r| r.start);
                for r in all {
                    match out.last_mut() {
                        Some(last) if r.start <= last.end => last.end = last.end.max(r.end),
                        _ => out.push(r),
                    }
                }
            }
        }
    }

    /// The guest runs of this log behind the import's bytes `range`.
    fn runs(&self, range: &Range<u64>) -> impl Iterator<Item = Range<u64>> {
        (self.segments(range).into_iter())
            .filter_map(|s| s.guest.map(|g| g..g.saturating_add(s.length)))
    }
}

/// The runtime's questions, in offsets from the import's pointer (module docs).
impl rt::DirtyPages for ImportDirty {
    type Watch = ImportWatch;

    fn watch(&self, range: Range<u64>) -> ImportWatch {
        let segments: Arc<[Segment]> = self.segments(&range).into();
        let runs: Vec<Range<u64>> = (segments.iter())
            .filter_map(|s| s.guest.map(|g| g..g.saturating_add(s.length)))
            .collect();
        self.shared.log.watch(&runs);
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        self.shared.watches.lock().push(Watched { id, segments });
        ImportWatch {
            shared: Arc::clone(&self.shared),
            id,
            runs,
        }
    }

    fn request(&self) -> rt::DirtyStamp {
        rt::DirtyStamp(self.shared.log.request())
    }

    fn wait(&self, stamp: rt::DirtyStamp) -> bool {
        self.shared.log.wait(stamp.0, &LinuxClock)
    }

    fn changed(&self, since: rt::DirtyStamp, out: &mut Vec<Range<u64>>) {
        self.changed_by(since.0, None, out);
    }

    fn wrote(&self, range: Range<u64>) {
        for run in self.runs(&range) {
            self.shared.log.wrote(run);
        }
    }

    fn wrote_as(&self, range: Range<u64>, writer: NonZeroU64) {
        if !writers() {
            return rt::DirtyPages::wrote(self, range);
        }
        for run in self.runs(&range) {
            self.shared.log.wrote_as(run, writer);
        }
    }

    fn changed_except(&self, since: rt::DirtyStamp, writer: NonZeroU64, out: &mut Vec<Range<u64>>) {
        self.changed_by(since.0, writers().then_some(writer), out);
    }
}

#[cfg(test)]
mod tests;
