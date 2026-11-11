//! A guest surface's memory as a device reaches it, for a backend whose GPU
//! writes a decoded frame into the guest's pages itself rather than leaving
//! it for the CPU to copy (ours).
//!
//! - [`DeviceMemory`] is what a host gives of a surface's pages: a dma-buf,
//!   for a GPU that imports one (VA-API), or one contiguous view in the
//!   host's address space, for one that registers host memory (NVIDIA's
//!   `cuMemHostRegister`). Each is the same pages the guest reads, so what a
//!   device writes there is what the guest reads.
//! - [`MemoryIdentity`] names one mapping of one surface's pages: the same
//!   for every hold on that mapping, and never the same for another mapping,
//!   before or after, so an import kept by it can never be used for a
//!   surface the guest has mapped over other pages since. The host makes a
//!   [`MemoryLife`] with each mapping and drops it with the mapping, which
//!   tells everything that kept something by its identity.
//! - [`DeviceImports`] keeps a backend's imports by identity and a variant of
//!   the backend's own, made once for each and dropped, on a thread of its
//!   own, the runtime's
//!   ([`crate::core::runtime`]), as soon as that mapping is gone: never kept past
//!   it, and never dropped on the host's thread, which may hold the host's
//!   locks.
//! - [`GuestTarget`] is a surface's layout checked against its memory, so
//!   that a device told to write its planes writes nowhere else.
//!
//! What may be written, and when, is the CPU copy's: only while the host
//! holds the surface, in the window the copy wrote it in. A hold keeps the
//! mapping and its pages as they are.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::hash::Hash;
#[cfg(unix)]
use std::os::fd::OwnedFd;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Weak};

use crate::core::backend::PixelFormat;
use crate::core::num::ToU64;
use crate::core::runtime::{Runtime, park_for_good};
use crate::core::surfaces::{PlaneLayout, SurfaceError, SurfaceLayout};
use crate::core::sync::Mutex;

/// The next identity's key: one counter for the process, so no two
/// mappings ever share one.
static NEXT_KEY: AtomicU64 = AtomicU64::new(1);

/// What is told when a mapping goes.
type Gone = Box<dyn FnOnce() + Send>;

/// One mapping's life, held by the host for as long as the mapping lives
/// (module docs). Dropping it runs what [`MemoryIdentity::when_gone`] was
/// given, on the dropping thread.
pub struct MemoryLife(Arc<LifeState>);

struct LifeState {
    key: u64,
    /// What to tell when the mapping goes; `None` once it has gone.
    gone: Mutex<Option<Vec<Gone>>>,
}

impl MemoryLife {
    /// A new mapping's life, with a key no other mapping in the process has
    /// had.
    pub fn new() -> Self {
        Self(Arc::new(LifeState {
            key: NEXT_KEY.fetch_add(1, Ordering::Relaxed),
            gone: Mutex::new(Some(Vec::new())),
        }))
    }

    /// Its identity, which does not keep it alive.
    pub fn identity(&self) -> MemoryIdentity {
        MemoryIdentity {
            key: self.0.key,
            life: Arc::downgrade(&self.0),
        }
    }
}

impl Default for MemoryLife {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for MemoryLife {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("MemoryLife").field(&self.0.key).finish()
    }
}

impl Drop for MemoryLife {
    fn drop(&mut self) {
        let gone = self.0.gone.lock().take().unwrap_or_default();
        for tell in gone {
            tell();
        }
    }
}

/// Which mapping of which pages a surface's memory is (module docs).
#[derive(Clone)]
pub struct MemoryIdentity {
    key: u64,
    life: Weak<LifeState>,
}

impl MemoryIdentity {
    /// Its key: unique to this mapping in the process, never reused.
    pub fn key(&self) -> u64 {
        self.key
    }

    /// Whether the mapping still lives.
    pub fn is_live(&self) -> bool {
        self.life
            .upgrade()
            .is_some_and(|life| life.gone.lock().is_some())
    }

    /// Runs `gone` once the mapping is gone, or at once if it is. It runs on
    /// the thread that drops the mapping, which may hold the host's locks,
    /// so it must only pass the word on, as a channel's send does: never call a driver, and never
    /// wait.
    pub fn when_gone(&self, gone: impl FnOnce() + Send + 'static) {
        let gone: Gone = Box::new(gone);
        let Some(life) = self.life.upgrade() else {
            gone();
            return;
        };
        let mut waiting = life.gone.lock();
        match &mut *waiting {
            Some(list) => list.push(gone),
            None => {
                drop(waiting);
                gone();
            }
        }
    }
}

impl fmt::Debug for MemoryIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryIdentity")
            .field("key", &self.key)
            .field("live", &self.is_live())
            .finish()
    }
}

/// A held surface's memory as a device reaches it, outside a lend (module
/// docs). It is valid while the hold it came from lives.
pub trait DeviceMemory: Send + Sync {
    /// Which mapping of which pages it is.
    fn identity(&self) -> MemoryIdentity;

    /// The bytes from the surface's first byte to the end of its
    /// allocation.
    fn length(&self) -> usize;

    /// A dma-buf of `ranges` of the surface's bytes, each an offset from
    /// its first byte and a length, in ascending order and apart, made for
    /// this call: the whole pages each range lies in, in order, a page two
    /// ranges share given once, and where each range's first byte lies in
    /// it. The caller keeps it by [`DeviceMemory::identity`]. `Err` says why
    /// not: [`DmaBufError::TooLarge`] where the pages are more than one
    /// dma-buf holds here, as where they lie in more runs than udmabuf
    /// takes in one, so that ranges of fewer pages may still be had.
    #[cfg(unix)]
    fn dma_buf_of(&self, ranges: &[(usize, usize)]) -> Result<SurfaceDmaBuf, DmaBufError> {
        let _ = ranges;
        Err(DmaBufError::Refused(
            "this host makes no dma-buf of a surface".to_owned(),
        ))
    }

    /// One contiguous view of the surface's pages in the host's address
    /// space, as the CPU reaches them, and a hold that keeps it
    /// mapped: the caller keeps both by [`DeviceMemory::identity`]. `Err`,
    /// with why, where the host has none.
    fn host_view(&self) -> Result<HostView, SurfaceError> {
        Err(SurfaceError::Unreachable(
            "this host lends no view of a surface's pages".to_owned(),
        ))
    }
}

/// A dma-buf of ranges of a surface's bytes ([`DeviceMemory::dma_buf_of`]).
#[cfg(unix)]
#[derive(Debug)]
pub struct SurfaceDmaBuf {
    /// The dma-buf.
    pub fd: OwnedFd,
    /// Its bytes: the ranges' pages, whole.
    pub size: u64,
    /// Where each range's first byte lies in it, in the ranges' order.
    pub offsets: Vec<u64>,
}

/// Why no dma-buf was made of a surface's ranges
/// ([`DeviceMemory::dma_buf_of`]).
#[cfg(unix)]
#[derive(Debug, thiserror::Error)]
pub enum DmaBufError {
    /// The ranges' pages are more than one dma-buf holds here: `pieces`
    /// runs of pages and `bytes` bytes, where one holds `most_pieces` runs
    /// and `most_bytes` bytes (udmabuf's `list_limit` and `size_limit_mb`).
    /// Ranges of fewer pages may fit.
    #[error("{why}")]
    TooLarge {
        /// The ranges' runs of pages.
        pieces: u64,
        /// Their bytes, whole pages.
        bytes: u64,
        /// The most runs one dma-buf holds.
        most_pieces: u64,
        /// The most bytes one dma-buf holds.
        most_bytes: u64,
        /// The host's words, for the log.
        why: String,
    },
    /// No dma-buf of these pages is to be had, with why.
    #[error("{0}")]
    Refused(String),
}

/// A view of a surface's whole pages in the host's address space
/// ([`DeviceMemory::host_view`]), mapped while `hold` lives.
pub struct HostView {
    /// The view's first byte: a page's.
    pub first: NonNull<u8>,
    /// Its bytes: the surface's pages, whole.
    pub size: usize,
    /// Where the surface's first byte lies in it.
    pub offset: usize,
    /// What keeps it mapped.
    pub hold: Box<dyn Send + Sync>,
}

// SAFETY: the view is the host's mapping of guest pages, which any thread
// may reach while `hold` keeps it mapped; this carries its address only.
#[expect(unsafe_code, reason = "a mapping of guest pages crosses threads")]
unsafe impl Send for HostView {}
// SAFETY: as above; nothing here is written through `&self`.
#[expect(unsafe_code, reason = "a mapping of guest pages crosses threads")]
unsafe impl Sync for HostView {}

impl fmt::Debug for HostView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HostView")
            .field("first", &self.first)
            .field("size", &self.size)
            .field("offset", &self.offset)
            .finish_non_exhaustive()
    }
}

/// One plane of a [`GuestTarget`]: where it lies in the surface's memory,
/// its bytes from one row to the next, and its size in samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TargetPlane {
    /// Its first byte's offset from the surface's first byte.
    pub offset: usize,
    /// The bytes from one row to the next.
    pub bytes_per_row: usize,
    /// Its rows.
    pub rows: u32,
    /// The bytes of each row a frame as wide as the surface fills: its
    /// samples, an interleaved Cb and Cr pair for every two luma samples.
    pub row_bytes: usize,
}

/// A guest buffer as a backend's device may write it: its memory, and its
/// layout checked against that memory (module docs).
///
/// What [`GuestTarget::new`] checks, so that a device that writes a plane's
/// rows writes inside the surface's allocation and nowhere else: a 4:2:0
/// format in two planes, luma then interleaved chroma, in ascending order
/// without overlapping, each row at least as long as the plane's samples,
/// and the last row of each ending inside the memory. A backend checks besides what
/// its driver takes (pitches, alignments, sizes) and how far its device
/// writes, and declines what it cannot hold to these bounds.
pub struct GuestTarget<'a> {
    memory: &'a dyn DeviceMemory,
    format: PixelFormat,
    width: u32,
    height: u32,
    planes: [TargetPlane; 2],
}

impl fmt::Debug for GuestTarget<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GuestTarget")
            .field("identity", &self.memory.identity())
            .field("length", &self.memory.length())
            .field("format", &self.format)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("planes", &self.planes)
            .finish()
    }
}

impl<'a> GuestTarget<'a> {
    /// `layout` over `memory`, checked (type docs); `Err` says why not.
    pub fn new(layout: &SurfaceLayout, memory: &'a dyn DeviceMemory) -> Result<Self, String> {
        let format = PixelFormat::from_fourcc(layout.pixel_format).ok_or_else(|| {
            format!(
                "the buffer is '{}', which no device writes",
                layout.pixel_format
            )
        })?;
        let [luma, chroma] = layout.planes[..] else {
            return Err(format!(
                "the buffer has {} planes, not two",
                layout.planes.len()
            ));
        };
        let sample = format.bytes_per_sample();
        let width = usize::try_from(layout.width).map_err(|_| "too wide".to_owned())?;
        let row_bytes = [
            width.checked_mul(sample),
            width.div_ceil(2).checked_mul(2 * sample),
        ];
        let rows = [layout.height, layout.height.div_ceil(2)];
        let length = memory.length();
        let mut planes = [TargetPlane {
            offset: 0,
            bytes_per_row: 0,
            rows: 0,
            row_bytes: 0,
        }; 2];
        let mut after = 0usize;
        for (((plane, (row_bytes, rows)), name), slot) in [luma, chroma]
            .into_iter()
            .zip(row_bytes.into_iter().zip(rows))
            .zip(["luma", "chroma"])
            .zip(&mut planes)
        {
            let row_bytes = row_bytes.ok_or_else(|| format!("{name}: rows too long"))?;
            let PlaneLayout {
                offset,
                bytes_per_row,
                height,
                ..
            } = plane;
            if bytes_per_row < row_bytes {
                return Err(format!(
                    "{name}: {bytes_per_row} bytes a row hold no {row_bytes} bytes of samples"
                ));
            }
            if height < rows {
                return Err(format!(
                    "{name}: {height} rows, where the picture has {rows}"
                ));
            }
            if offset < after {
                return Err(format!("{name}: at {offset}, inside the plane before it"));
            }
            let target = TargetPlane {
                offset,
                bytes_per_row,
                rows,
                row_bytes,
            };
            // The plane's whole rows, padding included, as its layout gives
            // them: a device may write any byte of a row it writes.
            let end = (usize::try_from(height).ok())
                .and_then(|height| bytes_per_row.checked_mul(height))
                .and_then(|bytes| bytes.checked_add(offset))
                .ok_or_else(|| format!("{name}: past the address space"))?;
            if end > length {
                return Err(format!(
                    "{name}: ends at {end}, past the surface's {length} bytes"
                ));
            }
            *slot = target;
            after = end;
        }
        Ok(Self {
            memory,
            format,
            width: layout.width,
            height: layout.height,
            planes,
        })
    }

    /// The memory the planes lie in.
    pub fn memory(&self) -> &'a dyn DeviceMemory {
        self.memory
    }

    /// The buffer's pixel format.
    pub fn pixel_format(&self) -> PixelFormat {
        self.format
    }

    /// The buffer's width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// The buffer's height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Its luma plane, then its chroma plane.
    pub fn planes(&self) -> [TargetPlane; 2] {
        self.planes
    }
}

/// What [`DeviceImports`] keeps an import by: the mapping's key, and a
/// variant of the backend's own, such as the size of frame and the place of
/// the planes in the mapping it was made for.
type ImportKey<V> = (u64, V);

/// A backend's variant of an import ([`DeviceImports`]): what besides its
/// mapping an import is made for, compared whole.
pub trait ImportVariant: Copy + Eq + Hash + Send + 'static {}

impl<V: Copy + Eq + Hash + Send + 'static> ImportVariant for V {}

/// What a backend made of guest surfaces' memory, by identity (module
/// docs): made once for each mapping and variant, at most `capacity` at
/// once, and dropped on a thread of its own once the mapping is gone, or
/// when the most imports are kept and another is made, the one used
/// longest ago.
pub struct DeviceImports<T: Send + Sync + 'static, V: ImportVariant = u64> {
    name: &'static str,
    capacity: usize,
    shared: Arc<Shared<T, V>>,
    /// Where the reaper's thread comes from.
    runtime: Arc<dyn Runtime>,
}

/// The name of the thread that drops imports whose mapping went (ours).
const REAPER_THREAD_NAME: &str = "reims-vgpu-video.imports";

struct Shared<T, V> {
    state: Mutex<State<T, V>>,
}

struct State<T, V> {
    entries: HashMap<ImportKey<V>, (Arc<T>, u64)>,
    /// The mappings told to say when they go.
    watched: HashSet<u64>,
    /// The order of use: each use's count.
    uses: u64,
    /// The thread that drops imports whose mapping went, started with the
    /// first watch.
    reaper: Option<Sender<Reap>>,
    /// Imports dropped since made, for the log.
    dropped: u64,
}

enum Reap {
    Gone(u64),
    Stop,
}

impl<T: Send + Sync + 'static, V: ImportVariant> DeviceImports<T, V> {
    /// No imports yet, at most `capacity` at once, dropped on a thread
    /// `runtime` starts; `name` names its log lines.
    pub fn new(runtime: &Arc<dyn Runtime>, name: &'static str, capacity: usize) -> Self {
        Self {
            name,
            runtime: Arc::clone(runtime),
            capacity: capacity.max(1),
            shared: Arc::new(Shared {
                state: Mutex::new(State {
                    entries: HashMap::new(),
                    watched: HashSet::new(),
                    uses: 0,
                    reaper: None,
                    dropped: 0,
                }),
            }),
        }
    }

    /// The import of `identity`'s memory for `variant`: the one kept, or
    /// one `make` makes now, which is kept until the mapping goes. `make`
    /// runs on the caller's thread with nothing of this held, so it may call
    /// a driver. A mapping already gone gets none, and `gone` makes the
    /// error.
    pub fn get_or_make<E>(
        &self,
        identity: &MemoryIdentity,
        variant: V,
        make: impl FnOnce() -> Result<T, E>,
        gone: impl FnOnce() -> E,
    ) -> Result<Arc<T>, E> {
        let key = (identity.key(), variant);
        {
            let mut state = self.shared.state.lock();
            state.uses += 1;
            let uses = state.uses;
            if let Some((import, used)) = state.entries.get_mut(&key) {
                *used = uses;
                return Ok(Arc::clone(import));
            }
        }
        if !identity.is_live() {
            return Err(gone());
        }
        let import = Arc::new(make()?);
        let mut evicted = Vec::new();
        let watch = {
            let mut state = self.shared.state.lock();
            while state.entries.len() >= self.capacity {
                let Some(oldest) = (state.entries.iter())
                    .min_by_key(|(_, (_, used))| *used)
                    .map(|(key, _)| *key)
                else {
                    break;
                };
                if let Some((import, _)) = state.entries.remove(&oldest) {
                    evicted.push(import);
                    state.dropped += 1;
                }
            }
            state.uses += 1;
            let uses = state.uses;
            // Another thread made one meanwhile: it goes with the evicted.
            if let Some((made, _)) = state.entries.insert(key, (Arc::clone(&import), uses)) {
                evicted.push(made);
            }
            if state.watched.insert(key.0) {
                Some(self.reaper(&mut state))
            } else {
                None
            }
        };
        // Dropped here, outside the lock: an evicted import's drop may call
        // its driver.
        drop(evicted);
        if let Some(Some(reaper)) = watch {
            identity.when_gone(move || {
                // An Err is a reaper that has stopped: nothing is kept.
                let _ = reaper.send(Reap::Gone(key.0));
            });
        }
        // Gone while it was made: its watch has run, or runs now, and drops
        // what was kept; this caller still writes under its hold.
        Ok(import)
    }

    /// The reaper's sender, starting it if it has not started; `None` where
    /// no thread starts, and imports are then dropped as others are made.
    fn reaper(&self, state: &mut State<T, V>) -> Option<Sender<Reap>> {
        if let Some(reaper) = &state.reaper {
            return Some(reaper.clone());
        }
        let (send, receive) = mpsc::channel::<Reap>();
        let shared = Arc::downgrade(&self.shared);
        let name = self.name;
        let runtime = Arc::clone(&self.runtime);
        let started = self.runtime.spawn(
            REAPER_THREAD_NAME,
            Box::new(move || {
                while let Ok(Reap::Gone(key)) = receive.recv() {
                    let Some(shared) = shared.upgrade() else {
                        return;
                    };
                    // The imports' drops call their driver: a step of the host's exit gate.
                    let Some(_step) = runtime.exit_step() else {
                        park_for_good()
                    };
                    let dropped: Vec<Arc<T>> = {
                        let mut state = shared.state.lock();
                        state.watched.remove(&key);
                        let keys: Vec<ImportKey<V>> = (state.entries.keys())
                            .filter(|(mapping, _)| *mapping == key)
                            .copied()
                            .collect();
                        let dropped: Vec<Arc<T>> = (keys.iter())
                            .filter_map(|k| state.entries.remove(k).map(|(import, _)| import))
                            .collect();
                        state.dropped += dropped.len().to_u64();
                        dropped
                    };
                    // Each import's drop, outside the lock, on this thread.
                    drop(dropped);
                }
            }),
        );
        match started {
            Ok(()) => {
                state.reaper = Some(send.clone());
                Some(send)
            }
            Err(error) => {
                log::warn!("{name}: no thread to drop imports whose surfaces go: {error}");
                None
            }
        }
    }

    /// The imports kept now.
    pub fn len(&self) -> usize {
        self.shared.state.lock().entries.len()
    }

    /// Whether none is kept.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The imports dropped since this was made: their mappings gone, or
    /// evicted.
    pub fn dropped(&self) -> u64 {
        self.shared.state.lock().dropped
    }
}

impl<T: Send + Sync + 'static, V: ImportVariant> Drop for DeviceImports<T, V> {
    /// Drops every import kept, on this thread, and stops the reaper.
    fn drop(&mut self) {
        let (entries, reaper) = {
            let mut state = self.shared.state.lock();
            (std::mem::take(&mut state.entries), state.reaper.take())
        };
        if let Some(reaper) = reaper {
            let _ = reaper.send(Reap::Stop);
        }
        drop(entries);
    }
}

impl<T: Send + Sync + 'static, V: ImportVariant> fmt::Debug for DeviceImports<T, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceImports")
            .field("name", &self.name)
            .field("kept", &self.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
