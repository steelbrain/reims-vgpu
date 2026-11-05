//! Libraries: their creates, in the three steps the device's host runs on two threads, and the
//! device's cache of the libraries it made, by content key.
//!
//! **Creates.** `LIBRARY_CREATE`, and the last chunk of a `LIBRARY_FROM_REGION` library, are
//! prepared on its context's lane ([`GraphicsDecoder::prepare_library`]): the message is read and
//! its bytes copied out, so every refusal of the message is at its op, and the library is marked
//! being made. The library is then made with no lock held ([`make_library`]), on a compile thread
//! where the host has one, and stored ([`GraphicsDecoder::finish_library`]). Each create is
//! answered by `COMPLETE`, with the platform's verdict.
//!
//! **Order.** A pipeline create that names a library being made takes the library's
//! [`LibrarySlot`] and waits on it where it compiles, not where it is prepared. An
//! `OBJECT_DESTROY` of a library being made, or its context's close, cancels its store: the
//! library is not stored under its id, but a pipeline prepared before the destroy still gets it
//! through the slot, so ring order holds.
//!
//! **The cache.** A library made from keyed bytes is kept under its key ([`verified_key`]) where
//! the key the host computes from the bytes is the key the guest sent; a `LIBRARY_LOOKUP` of that
//! key binds a new id to the same platform library, and no bytes cross. Each id is a handle of its
//! own in the table. An entry no id names any more is released into a least-recently-released
//! list bounded by [`RELEASED_BUDGET`] bytes of keyed bytes, so a process that loads the library
//! again (WindowServer after a restart) still finds it.
//!
//! **What the key names.** The key leaves the bitcode section out, and names each function's
//! bitcode by its `HASH` tag, a SHA-256 of it. The host vouches for a key only where it checked
//! every `HASH` against its bitcode and the functions' bitcode covers the whole section, so a
//! lookup can only ever bind content with that key; bytes whose tags lie are made for their
//! request, uncached. Every AIR `.metallib` of macOS 27's system, 210 of them, covers its section
//! (measured).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use reims_graphics::wire::{
    RGFX_CMD_LIBRARY_CREATE, RGFX_CMD_LIBRARY_FROM_REGION, RGFX_EVT_COMPLETE,
    RGFX_LIBRARY_BYTES_OFFSET, RGFX_LIBRARY_FLAG_KEYED, RGFX_STATUS_INVALID_ARGUMENT,
    RGFX_STATUS_NOT_FOUND, RGFX_STATUS_OK, RGFX_STATUS_REFUSED, RgfxComplete, RgfxLibraryCreate,
    RgfxLibraryFromRegion, RgfxLibraryLookup, RgfxMsgHeader,
};
use reims_vgpu_core::platform::device::GpuDevice;
use reims_vgpu_core::platform::{Handle, Platform};
use reims_vgpu_core::raise::RaiseResult;
use reims_vgpu_core::sync::{Condvar, Mutex};
use sha2::{Digest, Sha256};

use reims_graphics::wire::RGFX_CONTEXT_OBJ_ID_SHIFT;

use super::tables::Entry;
use super::{DecodeError, GraphicsDecoder, ObjHandle, read};

/// A library's content key ([`RgfxLibraryLookup`]).
pub(crate) type LibraryKey = [u8; 32];

/// The bytes the key's digest starts with ([`RgfxLibraryLookup`]).
const KEY_PREFIX: &[u8] = b"rgfx-library-v1";

/// The most bytes of keyed libraries the cache keeps once no id names them.
pub(crate) const RELEASED_BUDGET: u64 = 64 << 20;

/// A `.metallib` header's length, and where its sections' `(offset, size)` pairs lie.
const MTLB_HEADER_LEN: usize = 0x58;
const MTLB_FUNCTION_LIST: usize = 0x18;
const MTLB_BITCODE: usize = 0x48;

/// The content key of `.metallib` bytes, where the host can vouch that it names them: the bytes are
/// a `.metallib` whose every function's `HASH` is the SHA-256 of its bitcode, and whose functions'
/// bitcode covers the bitcode section, each byte once (functions may share one range whole).
/// `None` for anything else; such bytes are made, never cached.
pub(crate) fn verified_key(bytes: &[u8]) -> Option<LibraryKey> {
    let (bitcode_at, bitcode_len) = bitcode_section(bytes)?;
    let section = bytes.get(bitcode_at..bitcode_at.checked_add(bitcode_len)?)?;
    let mut ranges = function_bitcode(bytes)?;
    ranges.sort_unstable();
    ranges.dedup();
    // Covered whole, in order, with no gap and no overlap; a range two functions share is one
    // range once deduplicated (with two hashes it is two, which overlap).
    let mut at = 0usize;
    for &(start, len, hash) in &ranges {
        if start != at {
            return None;
        }
        let end = start.checked_add(len)?;
        let digest: [u8; 32] = Sha256::digest(section.get(start..end)?).into();
        if digest != hash {
            return None;
        }
        at = end;
    }
    (at == section.len()).then(|| content_key(bytes, bitcode_at, bitcode_len))
}

/// SHA-256 of [`KEY_PREFIX`] and `bytes` without `[bitcode_at, bitcode_at + bitcode_len)`.
fn content_key(bytes: &[u8], bitcode_at: usize, bitcode_len: usize) -> LibraryKey {
    let mut hasher = Sha256::new();
    hasher.update(KEY_PREFIX);
    hasher.update(&bytes[..bitcode_at]);
    hasher.update(&bytes[bitcode_at + bitcode_len..]);
    hasher.finalize().into()
}

/// The bitcode section of `.metallib` bytes, as `(offset, size)` inside them.
fn bitcode_section(bytes: &[u8]) -> Option<(usize, usize)> {
    if bytes.len() < MTLB_HEADER_LEN || bytes.get(..4)? != b"MTLB" {
        return None;
    }
    let at = usize::try_from(le64(bytes, MTLB_BITCODE)?).ok()?;
    let len = usize::try_from(le64(bytes, MTLB_BITCODE + 8)?).ok()?;
    (at >= MTLB_HEADER_LEN && at.checked_add(len)? <= bytes.len()).then_some((at, len))
}

/// Each function's bitcode range, relative to the bitcode section, with its `HASH`: from the
/// function list's entries, each a `u32` size counting itself and a tag list (a four-byte name,
/// a `u16` length and that many bytes, up to `ENDT`). `OFFT`'s third `u64` is the bitcode's
/// offset, `MDSZ` its size. `None` where an entry lacks one of the three or has one twice, or the
/// list does not parse.
fn function_bitcode(bytes: &[u8]) -> Option<Vec<(usize, usize, [u8; 32])>> {
    let list_at = usize::try_from(le64(bytes, MTLB_FUNCTION_LIST)?).ok()?;
    let list_len = usize::try_from(le64(bytes, MTLB_FUNCTION_LIST + 8)?).ok()?;
    let count = le32(bytes, list_at)?;
    let list_end = list_at.checked_add(4)?.checked_add(list_len)?;
    if list_end > bytes.len() {
        return None;
    }
    let mut out = Vec::new();
    let mut at = list_at + 4;
    for _ in 0..count {
        let size = usize::try_from(le32(bytes, at)?).ok()?;
        let end = at
            .checked_add(size)
            .filter(|&e| size >= 8 && e <= list_end)?;
        let entry = &bytes[..end];
        let (mut hash, mut len, mut offset) = (None, None, None);
        let mut tag = at + 4;
        loop {
            let name = entry.get(tag..tag.checked_add(4)?)?;
            if name == b"ENDT" {
                break;
            }
            let data_len = usize::from(u16::from_le_bytes(
                entry.get(tag + 4..tag + 6)?.try_into().ok()?,
            ));
            let data = entry.get(tag + 6..(tag + 6).checked_add(data_len)?)?;
            let twice = match name {
                b"HASH" => hash.replace(<[u8; 32]>::try_from(data).ok()?).is_some(),
                b"MDSZ" => len.replace(le64_exact(data)?).is_some(),
                b"OFFT" if data.len() == 24 => offset.replace(le64(data, 16)?).is_some(),
                b"OFFT" => return None,
                _ => false,
            };
            if twice {
                return None;
            }
            tag += 6 + data_len;
        }
        let (offset, len) = (usize::try_from(offset?).ok()?, usize::try_from(len?).ok()?);
        out.push((offset, len, hash?));
        at = end;
    }
    Some(out)
}

/// A tag's `u64`, which is the whole of its data.
fn le64_exact(data: &[u8]) -> Option<u64> {
    Some(u64::from_le_bytes(data.try_into().ok()?))
}

fn le32(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(at..at.checked_add(4)?)?.try_into().ok()?,
    ))
}

fn le64(bytes: &[u8], at: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(at..at.checked_add(8)?)?.try_into().ok()?,
    ))
}

/// The libraries the device made from keyed bytes, by key: each with its keyed bytes' length and
/// how many ids name it. An entry no id names waits in the released list, oldest first, until the
/// released entries' bytes pass [`RELEASED_BUDGET`].
pub(crate) struct LibraryCache<P: Platform> {
    entries: HashMap<LibraryKey, CacheEntry<P>>,
    /// The released entries, by the order they were released in.
    released: BTreeMap<u64, LibraryKey>,
    /// The bytes of the released entries.
    released_bytes: u64,
    /// The next release's place in `released`.
    tick: u64,
    budget: u64,
}

struct CacheEntry<P: Platform> {
    library: P::Library,
    size: u64,
    /// The ids that name it.
    holders: u32,
    /// Its place in the released list, where no id names it.
    released_at: Option<u64>,
}

impl<P: Platform> LibraryCache<P> {
    /// An empty cache that keeps `budget` bytes of released libraries.
    pub(crate) fn new(budget: u64) -> Self {
        LibraryCache {
            entries: HashMap::new(),
            released: BTreeMap::new(),
            released_bytes: 0,
            tick: 0,
            budget,
        }
    }

    /// The library under `key` of `size` keyed bytes, now named by one id more.
    fn take(&mut self, key: &LibraryKey, size: u64) -> Option<P::Library> {
        let entry = self.entries.get_mut(key).filter(|e| e.size == size)?;
        if let Some(at) = entry.released_at.take() {
            self.released.remove(&at);
            self.released_bytes -= entry.size;
        }
        entry.holders += 1;
        Some(entry.library.clone_ref())
    }

    /// Keeps `library` under `key`, named by one id where `held`; false, keeping nothing, where
    /// the key has an entry (another process made the same bytes meanwhile).
    fn insert(&mut self, key: LibraryKey, size: u64, library: &P::Library, held: bool) -> bool {
        if self.entries.contains_key(&key) {
            return false;
        }
        self.entries.insert(
            key,
            CacheEntry {
                library: library.clone_ref(),
                size,
                holders: u32::from(held),
                released_at: None,
            },
        );
        if !held {
            self.release_entry(key);
        }
        true
    }

    /// One id that named the entry under `key` no longer does.
    fn release(&mut self, key: &LibraryKey) {
        let Some(entry) = self.entries.get_mut(key) else {
            return;
        };
        entry.holders = entry.holders.saturating_sub(1);
        if entry.holders == 0 {
            self.release_entry(*key);
        }
    }

    /// Puts the entry under `key` on the released list, then drops the oldest released entries
    /// while their bytes pass the budget.
    fn release_entry(&mut self, key: LibraryKey) {
        let tick = self.tick;
        self.tick += 1;
        let Some(entry) = self.entries.get_mut(&key) else {
            return;
        };
        entry.released_at = Some(tick);
        self.released.insert(tick, key);
        self.released_bytes += entry.size;
        while self.released_bytes > self.budget {
            let Some((_, oldest)) = self.released.pop_first() else {
                break;
            };
            if let Some(gone) = self.entries.remove(&oldest) {
                self.released_bytes -= gone.size;
            }
        }
    }

    /// The entries, named or released (tests).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }
}

/// A library being made: where its make leaves the library (or `None`, refused) for the pipeline
/// creates prepared meanwhile.
pub(crate) struct LibrarySlot<P: Platform> {
    made: Mutex<Option<Option<P::Library>>>,
    done: Condvar,
}

impl<P: Platform> LibrarySlot<P> {
    fn new() -> Arc<Self> {
        Arc::new(LibrarySlot {
            made: Mutex::new(None),
            done: Condvar::new(),
        })
    }

    fn fill(&self, library: Option<&P::Library>) {
        *self.made.lock() = Some(library.map(Handle::clone_ref));
        self.done.notify_all();
    }

    /// The library once it is made; `None` where the platform refused it.
    pub(crate) fn wait(&self) -> Option<P::Library> {
        let mut made = self.made.lock();
        loop {
            if let Some(library) = &*made {
                return library.as_ref().map(Handle::clone_ref);
            }
            self.done.wait(&mut made);
        }
    }
}

/// A library a pipeline create names: in the table, or being made.
pub(crate) enum LibraryRef<P: Platform> {
    Ready(P::Library),
    Making(Arc<LibrarySlot<P>>),
}

impl<P: Platform> LibraryRef<P> {
    /// The library, waiting for its make where it is being made; `None` where that was refused.
    pub(crate) fn resolve(self) -> Option<P::Library> {
        match self {
            LibraryRef::Ready(library) => Some(library),
            LibraryRef::Making(slot) => slot.wait(),
        }
    }
}

/// A library create the decoder has prepared: its bytes, owned, and where its library goes.
pub(crate) struct LibraryJob<P: Platform> {
    /// The library's guest-picked id.
    pub(crate) obj_id: u64,
    /// The create's seqno, which its `COMPLETE` carries.
    pub(crate) seqno: u64,
    bytes: Vec<u8>,
    /// The key the guest sent with the bytes.
    guest_key: Option<LibraryKey>,
    slot: Arc<LibrarySlot<P>>,
    /// The device's resets when it was prepared.
    resets: u64,
}

/// What preparing a library message came to.
pub(crate) enum LibraryStart<P: Platform> {
    /// A library to make.
    Make(LibraryJob<P>),
    /// A chunk taken, before the library's last.
    Chunk,
    /// The message was not carried out; `answer` is the seqno of a create the guest waits on
    /// (one with no answer owed is `None`).
    Refused {
        error: DecodeError,
        answer: Option<u64>,
    },
}

/// A made library, to store.
pub(crate) struct MadeLibrary<P: Platform> {
    obj_id: u64,
    library: Option<P::Library>,
    /// The key the host vouches for, and the bytes' length, where it is the guest's.
    key: Option<(LibraryKey, u64)>,
    slot: Arc<LibrarySlot<P>>,
    /// The device's resets when its create was prepared.
    resets: u64,
    /// Why the platform made no library, for the host's log; `None` where it made one.
    refusal: Option<String>,
}

impl<P: Platform> MadeLibrary<P> {
    /// A make that raised or was refused, for `reason`: no library.
    pub(crate) fn refused(job: LibraryJob<P>, reason: impl Into<String>) -> Self {
        MadeLibrary {
            obj_id: job.obj_id,
            library: None,
            key: None,
            slot: job.slot,
            resets: job.resets,
            refusal: Some(reason.into()),
        }
    }

    /// Why the platform made no library; `None` where it made one.
    pub(crate) fn refusal(&self) -> Option<&str> {
        self.refusal.as_deref()
    }
}

/// The middle step of a library create, with no lock held: the key checked and the library made
/// from the bytes, unchanged. A platform raise propagates; the caller then stores
/// [`MadeLibrary::refused`].
pub(crate) fn make_library<P: Platform>(
    device: &P::Device,
    job: &LibraryJob<P>,
) -> RaiseResult<MadeLibrary<P>> {
    let key = job.guest_key.and_then(|guest| {
        let host = verified_key(&job.bytes)?;
        (host == guest).then_some((host, job.bytes.len() as u64))
    });
    // A nil library and a build failure both mean no usable library.
    let (library, refusal) = match device.new_library(&job.bytes)? {
        Ok(Some(library)) => (Some(library), None),
        Ok(None) => (None, Some("the platform answered nil".to_owned())),
        Err(e) => (
            None,
            Some(format!("the platform answered an error: {}", e.description)),
        ),
    };
    Ok(MadeLibrary {
        obj_id: job.obj_id,
        library,
        key,
        slot: Arc::clone(&job.slot),
        resets: job.resets,
        refusal,
    })
}

/// A `COMPLETE` for the create at `seqno`.
pub(crate) fn complete(seqno: u64, status: u32) -> RgfxComplete {
    RgfxComplete {
        header: RgfxMsgHeader {
            r#type: RGFX_EVT_COMPLETE,
            size: size_of::<RgfxComplete>() as u32,
            seqno,
        },
        status,
        ..RgfxComplete::default()
    }
}

/// Whether a chunk is its library's last, the one answered: its end reaches the length, or
/// overflows.
fn last_chunk(m: &RgfxLibraryFromRegion) -> bool {
    m.chunk_offset
        .checked_add(m.byte_length)
        .is_none_or(|end| end >= m.total_length)
}

fn keyed(flags: u32, key: LibraryKey) -> Option<LibraryKey> {
    (flags & RGFX_LIBRARY_FLAG_KEYED != 0).then_some(key)
}

/// What the decoder keeps of libraries beside its table, under one lock: held across a library's
/// store, so a destroy, which forgets these first, comes wholly before or after it.
pub(crate) struct Libraries<P: Platform> {
    /// The bytes of each `LIBRARY_FROM_REGION` taken so far, by library id.
    chunks: HashMap<u64, Vec<u8>>,
    /// The libraries being made, by id: a pipeline create naming one waits for its slot.
    making: HashMap<u64, Arc<LibrarySlot<P>>>,
    /// The libraries kept under their keys.
    cache: LibraryCache<P>,
    /// The key of the cache entry each library id names.
    cached: HashMap<u64, LibraryKey>,
    /// The device's resets so far ([`Self::take_cache`]): a library whose make began before one
    /// is not kept in the cache the reset made anew.
    resets: u64,
}

impl<P: Platform> Libraries<P> {
    /// None yet.
    pub(crate) fn new() -> Self {
        Libraries {
            chunks: HashMap::new(),
            making: HashMap::new(),
            cache: LibraryCache::new(RELEASED_BUDGET),
            cached: HashMap::new(),
            resets: 0,
        }
    }

    /// The ids of context `context`'s libraries these keep anything of.
    pub(crate) fn ids_of(&self, context: u32) -> Vec<u64> {
        let ours = |id: &u64| (id >> RGFX_CONTEXT_OBJ_ID_SHIFT) == u64::from(context);
        (self.making.keys())
            .chain(self.chunks.keys())
            .chain(self.cached.keys())
            .copied()
            .filter(ours)
            .collect()
    }

    /// The contexts of the libraries these keep anything of, each once.
    pub(crate) fn contexts(&self) -> Vec<u32> {
        let mut contexts: Vec<u32> = (self.making.keys())
            .chain(self.chunks.keys())
            .chain(self.cached.keys())
            .map(|id| (id >> RGFX_CONTEXT_OBJ_ID_SHIFT) as u32)
            .collect();
        contexts.sort_unstable();
        contexts.dedup();
        contexts
    }

    /// Lets go of the libraries kept under their keys, at the device's reset, once no id names
    /// one, and counts the reset: what it held, for the caller to let go of in a refusal scope.
    pub(crate) fn take_cache(&mut self) -> LibraryCache<P> {
        debug_assert!(self.cached.is_empty());
        self.resets += 1;
        std::mem::replace(&mut self.cache, LibraryCache::new(RELEASED_BUDGET))
    }

    /// Forgets what these keep of library `obj_id`: its make, which is cancelled, its chunks, and
    /// the cache entry it named.
    fn forget(&mut self, obj_id: u64) {
        self.making.remove(&obj_id);
        self.chunks.remove(&obj_id);
        if let Some(key) = self.cached.remove(&obj_id) {
            self.cache.release(&key);
        }
    }
}

impl<P: Platform> GraphicsDecoder<P> {
    /// `LIBRARY_LOOKUP`: binds the message's id to the library kept under its key, where there is
    /// one of its size. The create's seqno (`None` for a message too short to carry one) and its
    /// `COMPLETE` status.
    pub(crate) fn lookup_library(&self, message: &[u8]) -> (Option<u64>, u32) {
        let Some(m) = read::<RgfxLibraryLookup>(message) else {
            let seqno = read::<RgfxMsgHeader>(message).map(|h| h.seqno);
            return (seqno, RGFX_STATUS_INVALID_ARGUMENT);
        };
        let mut libraries = self.shared.libraries.lock();
        let Some(library) = libraries.cache.take(&m.key, m.slice_size) else {
            return (Some(m.header.seqno), RGFX_STATUS_NOT_FOUND);
        };
        libraries.forget(m.obj_id);
        self.put(m.obj_id, Entry::of(ObjHandle::Library(library)));
        libraries.cached.insert(m.obj_id, m.key);
        (Some(m.header.seqno), RGFX_STATUS_OK)
    }

    /// The first step of a `LIBRARY_CREATE` or `LIBRARY_FROM_REGION`, on its context's lane: the
    /// message read and its bytes copied out (a chunk's acked), each refusal here at its op as
    /// [`Self::decode`]'s; a library's last bytes are marked being made, so pipeline creates that
    /// name it wait for it ([`LibraryRef`]).
    pub(crate) fn prepare_library(&self, message: &[u8]) -> LibraryStart<P> {
        let header = read::<RgfxMsgHeader>(message);
        let refused = |error, answer| LibraryStart::Refused { error, answer };
        let (obj_id, seqno, bytes, guest_key) = match header.map(|h| h.r#type) {
            Some(RGFX_CMD_LIBRARY_CREATE) => {
                let seqno = header.map(|h| h.seqno);
                let Some(m) = read::<RgfxLibraryCreate>(message) else {
                    return refused(DecodeError::Malformed, seqno);
                };
                // The `.metallib`/AIR bytes follow the header.
                let start = RGFX_LIBRARY_BYTES_OFFSET as usize;
                let Some(bytes) = usize::try_from(m.byte_length)
                    .ok()
                    .and_then(|len| len.checked_add(start))
                    .and_then(|end| message.get(start..end))
                else {
                    return refused(DecodeError::Malformed, seqno);
                };
                (
                    m.obj_id,
                    m.header.seqno,
                    bytes.to_vec(),
                    keyed(m.flags, m.key),
                )
            }
            Some(RGFX_CMD_LIBRARY_FROM_REGION) => {
                let Some(m) = read::<RgfxLibraryFromRegion>(message) else {
                    return refused(DecodeError::Malformed, None);
                };
                let answer = last_chunk(&m).then_some(m.header.seqno);
                match self.take_library_chunk(&m) {
                    Err(error) => return refused(error, answer),
                    Ok(None) => return LibraryStart::Chunk,
                    Ok(Some(bytes)) => (m.obj_id, m.header.seqno, bytes, keyed(m.flags, m.key)),
                }
            }
            _ => return refused(DecodeError::Malformed, None),
        };
        let slot = LibrarySlot::new();
        let resets = {
            let mut libraries = self.shared.libraries.lock();
            libraries.making.insert(obj_id, Arc::clone(&slot));
            libraries.resets
        };
        LibraryStart::Make(LibraryJob {
            obj_id,
            seqno,
            bytes,
            guest_key,
            slot,
            resets,
        })
    }

    /// One `LIBRARY_FROM_REGION` chunk: its bytes are copied out once (so the platform parses
    /// memory the guest cannot change under it), its ack is written so the guest may reuse the
    /// block, and the library's whole bytes are handed back once the chunks reach its length. A
    /// span in no block of the sender's is a bad reference; a chunk out of order or past the
    /// length is malformed and drops what came before.
    fn take_library_chunk(
        &self,
        m: &RgfxLibraryFromRegion,
    ) -> Result<Option<Vec<u8>>, DecodeError> {
        let mut libraries = self.shared.libraries.lock();
        let chunks = &mut libraries.chunks;
        let Some(chunk) = usize::try_from(m.byte_length)
            .ok()
            .filter(|_| self.in_own_block(m.region_offset, m.byte_length))
            .and_then(|len| self.region.read(m.region_offset, len))
        else {
            chunks.remove(&m.obj_id);
            return Err(DecodeError::BadReference);
        };
        // The bytes are out of the block: the guest may write it again.
        if !m.ack_offset.is_multiple_of(8)
            || !self.in_own_block(m.ack_offset, 8)
            || !self.region.write(m.ack_offset, &m.ack_value.to_le_bytes())
        {
            chunks.remove(&m.obj_id);
            return Err(DecodeError::BadReference);
        }
        let mut bytes = if m.chunk_offset == 0 {
            chunks.remove(&m.obj_id);
            Vec::with_capacity(usize::try_from(m.total_length).unwrap_or(0).min(1 << 30))
        } else {
            match chunks.remove(&m.obj_id) {
                Some(b) => b,
                None => return Err(DecodeError::Malformed),
            }
        };
        if bytes.len() as u64 != m.chunk_offset
            || m.chunk_offset
                .checked_add(m.byte_length)
                .is_none_or(|end| end > m.total_length)
        {
            return Err(DecodeError::Malformed);
        }
        bytes.extend_from_slice(&chunk);
        if (bytes.len() as u64) < m.total_length {
            chunks.insert(m.obj_id, bytes);
            return Ok(None);
        }
        Ok(Some(bytes))
    }

    /// The last step of a library create: the library handed to the pipeline creates waiting on
    /// it, stored under its id where no destroy or close cancelled it meanwhile, and kept under
    /// its key where the host vouches for the guest's (cancelled, it is kept released), unless the
    /// device was reset since its create was prepared: the rebooted guest's lookups then miss, as
    /// on a new device. Its `COMPLETE` status.
    pub(crate) fn finish_library(&self, made: MadeLibrary<P>) -> u32 {
        made.slot.fill(made.library.as_ref());
        let mut libraries = self.shared.libraries.lock();
        let live =
            (libraries.making.get(&made.obj_id)).is_some_and(|slot| Arc::ptr_eq(slot, &made.slot));
        if live {
            libraries.making.remove(&made.obj_id);
        }
        let Some(library) = made.library else {
            return RGFX_STATUS_REFUSED;
        };
        let cached = made.resets == libraries.resets
            && (made.key)
                .is_some_and(|(key, size)| libraries.cache.insert(key, size, &library, live));
        if live {
            libraries.forget(made.obj_id);
            self.put(made.obj_id, Entry::of(ObjHandle::Library(library)));
            if let (true, Some((key, _))) = (cached, made.key) {
                libraries.cached.insert(made.obj_id, key);
            }
        }
        RGFX_STATUS_OK
    }

    /// A library create carried out at once, as the device's host runs its three steps
    /// ([`Self::prepare_library`], [`make_library`], [`Self::finish_library`]); the `COMPLETE` it
    /// owes is not posted.
    pub(super) fn decode_library(
        &self,
        device: &P::Device,
        message: &[u8],
    ) -> RaiseResult<Result<(), DecodeError>> {
        let job = match self.prepare_library(message) {
            LibraryStart::Make(job) => job,
            LibraryStart::Chunk => return Ok(Ok(())),
            LibraryStart::Refused { error, .. } => return Ok(Err(error)),
        };
        let made = match super::refusing(|| make_library::<P>(device, &job)) {
            Ok(made) => made,
            Err(raise) => {
                self.finish_library(MadeLibrary::refused(job, raise.to_string()));
                return Err(raise);
            }
        };
        Ok(match self.finish_library(made) {
            RGFX_STATUS_OK => Ok(()),
            _ => Err(DecodeError::Refused),
        })
    }

    /// The library a pipeline create names: in the table, or being made, and the current
    /// context's.
    pub(super) fn library_ref(&self, obj_id: u64) -> Result<LibraryRef<P>, DecodeError> {
        match self.named(obj_id) {
            Some(ObjHandle::Library(library)) => Ok(LibraryRef::Ready(library.clone_ref())),
            _ => (self.shared.libraries.lock().making.get(&obj_id))
                .filter(|_| self.owned(obj_id))
                .map(|slot| LibraryRef::Making(Arc::clone(slot)))
                .ok_or(DecodeError::BadReference),
        }
    }

    /// Forgets what the table keeps of library `obj_id` beside its handle: its make, which is
    /// cancelled, its chunks, and the cache entry it named.
    pub(super) fn forget_library(&self, obj_id: u64) {
        self.shared.libraries.lock().forget(obj_id);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A `.metallib` of `functions` (each a bitcode range, relative to the section, and its
    /// `HASH`) over `bitcode`, with a function list entry per function.
    pub(crate) fn metallib(bitcode: &[u8], functions: &[(u64, u64, [u8; 32])]) -> Vec<u8> {
        let mut list = Vec::new();
        for (i, &(offset, len, hash)) in functions.iter().enumerate() {
            let mut tags = Vec::new();
            let mut tag = |name: &[u8; 4], data: &[u8]| {
                tags.extend_from_slice(name);
                tags.extend_from_slice(&(data.len() as u16).to_le_bytes());
                tags.extend_from_slice(data);
            };
            tag(b"NAME", format!("f{i}\0").as_bytes());
            tag(b"HASH", &hash);
            tag(b"MDSZ", &len.to_le_bytes());
            let mut offt = [0u8; 24];
            offt[16..].copy_from_slice(&offset.to_le_bytes());
            tag(b"OFFT", &offt);
            tags.extend_from_slice(b"ENDT");
            list.extend_from_slice(&(tags.len() as u32 + 4).to_le_bytes());
            list.extend_from_slice(&tags);
        }
        let list_at = MTLB_HEADER_LEN as u64;
        let bitcode_at = list_at + 4 + list.len() as u64 + 4; // and the extended header's ENDT
        let mut out = vec![0u8; MTLB_HEADER_LEN];
        out[..4].copy_from_slice(b"MTLB");
        out[0x18..0x20].copy_from_slice(&list_at.to_le_bytes());
        out[0x20..0x28].copy_from_slice(&(list.len() as u64).to_le_bytes());
        out[0x48..0x50].copy_from_slice(&bitcode_at.to_le_bytes());
        out[0x50..0x58].copy_from_slice(&(bitcode.len() as u64).to_le_bytes());
        out.extend_from_slice(&(functions.len() as u32).to_le_bytes());
        out.extend_from_slice(&list);
        out.extend_from_slice(b"ENDT");
        out.extend_from_slice(bitcode);
        out.extend_from_slice(b"tail");
        let total = out.len() as u64;
        out[0x10..0x18].copy_from_slice(&total.to_le_bytes());
        out
    }

    pub(crate) fn sha(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    /// Two functions whose hashes match and whose bitcode covers the section: the key is the
    /// digest of the prefix and the bytes around the section.
    #[test]
    fn vouches_for_a_covered_library() {
        let bitcode = b"first-functionsecond";
        let lib = metallib(
            bitcode,
            &[(0, 14, sha(&bitcode[..14])), (14, 6, sha(&bitcode[14..]))],
        );
        let at = lib.len() - 4 - bitcode.len();
        let mut expect = Sha256::new();
        expect.update(b"rgfx-library-v1");
        expect.update(&lib[..at]);
        expect.update(&lib[at + bitcode.len()..]);
        let expect: [u8; 32] = expect.finalize().into();
        assert_eq!(verified_key(&lib), Some(expect));
    }

    /// The key of the bundle's synthetic `.metallib` (`guest/bundle/tests/serializer_tests.cpp`,
    /// `test_library_key`) is the digest the bundle's test pins: both sides key alike.
    #[test]
    fn keys_as_the_bundle_does() {
        let mut lib = vec![0u8; MTLB_HEADER_LEN];
        lib[..4].copy_from_slice(b"MTLB");
        lib[0x48..0x50].copy_from_slice(&0x60u64.to_le_bytes());
        lib[0x50..0x58].copy_from_slice(&0x10u64.to_le_bytes());
        lib.extend_from_slice(b"funclistBITCODE-BITCODE!tail-end");
        assert_eq!(bitcode_section(&lib), Some((0x60, 0x10)));
        let expect = [
            0xc0, 0x99, 0x93, 0x08, 0x93, 0xa6, 0x91, 0xa1, 0xaa, 0xed, 0xfa, 0x6d, 0xee, 0xd4,
            0x1a, 0x58, 0x09, 0x95, 0x3e, 0x36, 0xa8, 0xe6, 0x9e, 0xd9, 0x86, 0xe1, 0xb2, 0x11,
            0xff, 0x4e, 0x21, 0x7d,
        ];
        assert_eq!(content_key(&lib, 0x60, 0x10), expect);
    }

    /// A tag that lies, a gap in the section, an overlap, and bytes that are no `.metallib` get
    /// no key; two functions sharing one range whole do.
    #[test]
    fn vouches_for_nothing_else() {
        let bitcode = b"0123456789";
        let lying = metallib(bitcode, &[(0, 10, sha(b"something else"))]);
        assert_eq!(verified_key(&lying), None);
        let gap = metallib(
            bitcode,
            &[(0, 4, sha(&bitcode[..4])), (5, 5, sha(&bitcode[5..]))],
        );
        assert_eq!(verified_key(&gap), None);
        let overlap = metallib(
            bitcode,
            &[(0, 6, sha(&bitcode[..6])), (4, 6, sha(&bitcode[4..]))],
        );
        assert_eq!(verified_key(&overlap), None);
        let short = metallib(bitcode, &[(0, 9, sha(&bitcode[..9]))]);
        assert_eq!(verified_key(&short), None);
        assert_eq!(
            verified_key(
                b"not a metallib at all, long enough to hold a header ..........................."
            ),
            None
        );
        let shared = metallib(bitcode, &[(0, 10, sha(bitcode)), (0, 10, sha(bitcode))]);
        assert!(verified_key(&shared).is_some());
    }

    /// Released entries wait oldest first until their bytes pass the budget; an entry taken again
    /// leaves the list, and named entries are never dropped
    #[test]
    fn the_released_list_keeps_to_its_budget() {
        use reims_vgpu_core::testkit::FakePlatform;
        let platform = FakePlatform::new();
        let device = platform.device();
        let library = || device.new_library(&[1]).unwrap().unwrap().unwrap();
        let mut cache = LibraryCache::<FakePlatform>::new(100);
        let (a, b, c) = ([1u8; 32], [2u8; 32], [3u8; 32]);
        assert!(cache.insert(a, 40, &library(), true));
        assert!(cache.insert(b, 40, &library(), true));
        assert!(cache.insert(c, 40, &library(), false));
        assert!(!cache.insert(a, 40, &library(), true));
        cache.release(&a);
        assert_eq!(cache.len(), 3);
        // 120 released bytes: the oldest released, c, goes.
        cache.release(&b);
        assert_eq!(cache.len(), 2);
        assert!(cache.take(&c, 40).is_none());
        assert!(cache.take(&a, 40).is_some());
        cache.release(&a);
        assert_eq!(cache.len(), 2);
        assert!(cache.take(&b, 40).is_some());
        assert!(cache.take(&a, 40).is_some());
    }
}
