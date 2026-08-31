//! Host surface cache for Linux/Vulkan discrete-GPU present (kb tahoe-x86 §8.5).
//!
//! On Apple Metal hosts, GPU Stores land in guest IOSurface pages (unified
//! memory). On this Linux product rail guest type-4 pages are **not** filled by
//! the host GPU until encode writeback; historical product painted from a
//! **host render-cache** keyed by surface_id. This module is that cache.
//!
//! Namespace split (2026-07-13 live x86):
//! - [`store`] / [`get`] — **type-4 surface_id / mapping_id** only (`host_surfaces`)
//! - [`store_texture`] / [`get_texture`] — type-2/3 color targets by task/object ref
//! - [`store_gva_owned`] / [`get_gva`] — type-2/3 by target GVA (survives ref rebinding)
//!
//! Never put texture_ref into `host_surfaces`: list ids collide with mids and
//! recycled refs return stale full-frame blacks as multi-bind samples.
//!
//! Write paths (clear Store, metal2vulkan encode writeback) call the matching
//! store; [`crate::runtime::scanout::capture_present_frame`] prefers surface_id
//! cache content when present so scanout matches what the host executed.

use crate::contract::pixel_format::RGBA8_BPP;
use crate::model::{scanout_extent_ok, DeviceState, GvaBacking, HostSurface};
use crate::runtime::host::HostMemory;

/// `generation` is issued by
/// [`DeviceState::next_sampled_content_generation`] and is never derived from
/// the entry being replaced: an entry-local counter restarts whenever the entry
/// is re-created, and half of this cache's identity contract is that a
/// generation names one content for the life of the device.
fn store_into<K: Ord>(
    map: &mut std::collections::BTreeMap<K, HostSurface>,
    id: K,
    width: u32,
    height: u32,
    bgra: std::sync::Arc<Vec<u8>>,
    generation: u64,
) {
    if !scanout_extent_ok(width, height) {
        return;
    }
    let need = (height as usize)
        .saturating_mul(width as usize)
        .saturating_mul(RGBA8_BPP as usize);
    if bgra.len() < need {
        return;
    }
    let entry = map.entry(id).or_default();
    entry.host_gen = generation;
    entry.width = width;
    entry.height = height;
    entry.bgra = bgra;
    // These two maps carry no byte cap, so nothing here ever consults the flag.
    // Set anyway rather than left at `Default`: `false` is the value that makes
    // the GVA cap refuse to evict, and an entry that inherits it from a derive
    // would be claiming something no store here has checked.
    entry.guest_holds_bytes = true;
}

fn get_from<'a, K: Ord>(
    map: &'a std::collections::BTreeMap<K, HostSurface>,
    id: &K,
    width: u32,
    height: u32,
) -> Option<&'a [u8]> {
    get_from_with_gen(map, id, width, height).map(|(bgra, _)| bgra)
}

fn get_from_with_gen<'a, K: Ord>(
    map: &'a std::collections::BTreeMap<K, HostSurface>,
    id: &K,
    width: u32,
    height: u32,
) -> Option<(&'a [u8], u64)> {
    let e = map.get(id)?;
    if e.width != width || e.height != height || e.bgra.is_empty() {
        return None;
    }
    let need = (height as usize)
        .saturating_mul(width as usize)
        .saturating_mul(RGBA8_BPP as usize);
    if e.bgra.len() < need {
        return None;
    }
    Some((&e.bgra[..need], e.host_gen))
}

/// Insert/replace host-cache pixels for `surface_id` (type-4 present id).
pub fn store(state: &mut DeviceState, surface_id: u32, width: u32, height: u32, bgra: Vec<u8>) {
    store_shared(state, surface_id, width, height, std::sync::Arc::new(bgra));
}

/// [`store`] for a frame already held behind an `Arc` — the type-11 render Store
/// arms its deferred window with the same allocation, so the frame is stored
/// once and referenced twice.
pub fn store_shared(
    state: &mut DeviceState,
    surface_id: u32,
    width: u32,
    height: u32,
    bgra: std::sync::Arc<Vec<u8>>,
) {
    if surface_id == 0 {
        return;
    }
    let generation = state.next_sampled_content_generation();
    store_into(
        &mut state.host_surfaces,
        surface_id,
        width,
        height,
        bgra,
        generation,
    );
}

/// [`store`] from rows the caller only has as a borrow, reusing the entry's own
/// allocation whenever nothing else is looking at it.
///
/// The caller that mattered is the deferred render writeback, which reaches here
/// holding a frame it does not own and had to build a whole second copy of just
/// to call [`store`]. On the composite surface that is a fresh 8.29 MB
/// `Vec` about 95 times a second, and the allocation is the expensive half, not
/// the copy: a multi-megabyte `vec![0u8; n]` comes back as untouched pages and
/// the fill faults every one of them in, then the buffer is dropped and the next
/// flush does it again. Measured at 1.21 ms per flush — 32% of the writeback,
/// against 0.72 ms for landing the frame in the guest's pages.
///
/// Reuse is conditional on [`std::sync::Arc::get_mut`], so it happens only when
/// the strong count is exactly one and no window, sampled binding or present
/// capture can observe the mutation. When something does hold the frame, this
/// allocates as before and the old bytes stay intact for their holder.
pub fn store_rows(
    state: &mut DeviceState,
    surface_id: u32,
    width: u32,
    height: u32,
    src: &[u8],
    src_stride: u32,
) {
    let generation = state.next_sampled_content_generation();
    if surface_id == 0 || !scanout_extent_ok(width, height) {
        return;
    }
    let row = (width as usize).saturating_mul(RGBA8_BPP as usize);
    let need = (height as usize).saturating_mul(row);
    if need == 0 || src.len() < (height as usize).saturating_mul(src_stride as usize) {
        return;
    }
    let entry = state.host_surfaces.entry(surface_id).or_default();
    match std::sync::Arc::get_mut(&mut entry.bgra) {
        Some(buf) if buf.len() == need => fill_tight_rows(buf, src, src_stride, row, height),
        _ => {
            let mut buf = vec![0u8; need];
            fill_tight_rows(&mut buf, src, src_stride, row, height);
            entry.bgra = std::sync::Arc::new(buf);
        }
    }
    entry.host_gen = generation;
    entry.width = width;
    entry.height = height;
    // The same reason [`store_into`] sets it: this map has no byte cap, so
    // nothing reads the flag, but `false` is the value that means "the cap must
    // not evict this" and an entry reaching it through `or_default` would be
    // asserting something no store here has checked.
    entry.guest_holds_bytes = true;
}

/// Copy `height` rows of `row` bytes out of `src` at `src_stride` pitch into a
/// tightly packed `dst`.
///
/// One `copy_from_slice` when the source is already tight, which is the shape
/// every readback arrives in — a per-row loop over an 8 MB frame is 1079 calls
/// that a single memcpy expresses exactly.
fn fill_tight_rows(dst: &mut [u8], src: &[u8], src_stride: u32, row: usize, height: u32) {
    if src_stride as usize == row {
        let n = dst.len().min(src.len());
        dst[..n].copy_from_slice(&src[..n]);
        return;
    }
    for y in 0..height as usize {
        let so = y.saturating_mul(src_stride as usize);
        let doff = y.saturating_mul(row);
        if so + row <= src.len() && doff + row <= dst.len() {
            dst[doff..doff + row].copy_from_slice(&src[so..so + row]);
        }
    }
}

/// Borrow host-cache frame when geom matches request (surface_id namespace).
pub fn get(state: &DeviceState, surface_id: u32, width: u32, height: u32) -> Option<&[u8]> {
    get_from(&state.host_surfaces, &surface_id, width, height)
}

/// [`get_shared`], plus the generation that names these exact bytes.
///
/// The generation is the entry's `host_gen`, and it is a sampled-content
/// identity rather than provenance: every writer of this map — [`store_into`],
/// [`store_rows`] and [`cede_surface_to_resident`] — takes a fresh
/// [`DeviceState::next_sampled_content_generation`] in the same breath as it
/// changes the bytes, and [`forget`] removes the entry outright. So a repeated
/// `(surface_id, generation)` pair is a statement that the bytes have not moved,
/// which is what lets the sampled cache skip re-hashing a frame it already holds.
///
/// The caller is the type-11 sampled ladder's host-cache rung, which without
/// this had no identity to offer and drove every bind through the content
/// digest: 116 lookups a second over 201 MB, hashed twice each. It takes the
/// frame as a handle rather than a slice because it hands the bytes to the
/// engine, which outlives the borrow of `state` — so the rung costs a refcount
/// and not a full-frame copy.
pub fn get_shared_with_gen(
    state: &DeviceState,
    surface_id: u32,
    width: u32,
    height: u32,
) -> Option<(std::sync::Arc<Vec<u8>>, u64)> {
    let (_, host_gen) = get_from_with_gen(&state.host_surfaces, &surface_id, width, height)?;
    // Deliberately delegated rather than reimplemented: `get_shared` owns the
    // rule for a stored buffer carrying slop past `width * height * 4`, and a
    // second copy of that rule here could drift from it.
    Some((get_shared(state, surface_id, width, height)?, host_gen))
}

/// Cede this mapping's cached frame to the engine resident a deferred type-11
/// render Store just pinned: the entry keeps its geometry and its `host_gen`
/// lineage, and holds no bytes.
///
/// The emptiness **is** the cession, and [`get_from`]'s `bgra.is_empty()` gate is
/// what enforces it — so every reader that goes through [`get`] or [`get_shared`]
/// misses and falls through to the source that does hold the frame:
/// [`crate::runtime::scanout::capture_present_frame`] to
/// `try_capture_from_resident`, and the type-11 LOAD seed to the surface's own
/// guest pages, which lands this window first. Nothing has to be taught about a
/// new state.
///
/// Retaining the stale bytes as a fallback would be worse than missing. A Store
/// that skipped its readback has already superseded them, and a consumer served
/// the previous frame renders a whole compositing layer one frame behind with no
/// report — which is the class `deferred_flush_lost reason=cache_miss` cost 15
/// layers in one boot to close.
///
/// Returns false for a geometry this cache would not have stored anyway, so the
/// caller can refuse to arm rather than leave a live entry contradicting a
/// resident-authoritative window.
pub fn cede_surface_to_resident(
    state: &mut DeviceState,
    surface_id: u32,
    width: u32,
    height: u32,
) -> bool {
    if surface_id == 0 || !scanout_extent_ok(width, height) {
        return false;
    }
    let generation = state.next_sampled_content_generation();
    let entry = state.host_surfaces.entry(surface_id).or_default();
    entry.host_gen = generation;
    entry.width = width;
    entry.height = height;
    entry.bgra = std::sync::Arc::new(Vec::new());
    true
}

/// Drop this mapping's cache entry outright.
///
/// Distinct from [`cede_surface_to_resident`], and the difference is which
/// source the reader is being sent to. A cession says "the engine resident holds
/// this frame"; this says "nothing host-side does — read the surface's own
/// pages". It is what a writeback that deliberately left some of the guest's own
/// bytes in place has to do, because after one of those neither the cache nor the
/// resident holds the mapping's content: they hold the frame the device rendered,
/// and the pages hold that frame with the guest's stores still in it.
///
/// Removes rather than emptying, so [`surface_ceded_to_resident`] does not read
/// the result as a cession and report a decline that names the wrong source.
pub fn forget(state: &mut DeviceState, surface_id: u32) {
    state.host_surfaces.remove(&surface_id);
}

/// Whether this mapping's cache entry is the ceded shell
/// [`cede_surface_to_resident`] leaves behind: present at exactly this geometry
/// and carrying no bytes.
///
/// Read by the type-11 LOAD seed's decline classifier so a ceded entry is named
/// as such instead of being reported as a stale-geometry hit — `get`'s miss is
/// the same either way, and the two have different fixes.
pub fn surface_ceded_to_resident(
    state: &DeviceState,
    surface_id: u32,
    width: u32,
    height: u32,
) -> bool {
    state
        .host_surfaces
        .get(&surface_id)
        .is_some_and(|e| e.bgra.is_empty() && e.width == width && e.height == height)
}

/// [`get`] as a shared handle, for a caller that needs to own the frame past the
/// borrow of `state` — a Load seed does, and taking it this way costs a refcount
/// rather than a full-framebuffer copy.
///
/// Hits exactly when [`get`] hits. A handle cannot be truncated the way [`get`]'s
/// slice is, so the refcount is only taken when the stored buffer is *exactly*
/// `width * height * 4`; a store carrying slop past that is copied instead.
///
/// The copy is not reachable today — every producer of `host_surfaces` allocates
/// exactly that — but returning `None` there would be a silent seed loss, and a
/// missing Load seed renders the pass onto a cleared target, which is a
/// compositing layer going solid black. Matching [`get`] means a future producer
/// with slop costs a copy rather than a defect.
pub fn get_shared(
    state: &DeviceState,
    surface_id: u32,
    width: u32,
    height: u32,
) -> Option<std::sync::Arc<Vec<u8>>> {
    let need = get_from(&state.host_surfaces, &surface_id, width, height)?.len();
    let e = state.host_surfaces.get(&surface_id)?;
    Some(if e.bgra.len() == need {
        std::sync::Arc::clone(&e.bgra)
    } else {
        std::sync::Arc::new(e.bgra[..need].to_vec())
    })
}

/// Type-2/3 encode cache by task-local texture object ref (not surface_id).
///
/// `source_gva` is the address the producing Store rendered into, kept so
/// [`texture_source_gva`] can tell a later serve whether this entry's pixels
/// came from the allocation it is about to be served as. Zero when the producer
/// had no GVA.
pub fn store_texture(
    state: &mut DeviceState,
    task_id: u32,
    texture_ref: u32,
    width: u32,
    height: u32,
    bgra: Vec<u8>,
    source_gva: u64,
) {
    if texture_ref == 0 {
        return;
    }
    let generation = state.next_sampled_content_generation();
    store_into(
        &mut state.host_texture_surfaces,
        (task_id, texture_ref),
        width,
        height,
        std::sync::Arc::new(bgra),
        generation,
    );
    if let Some(e) = state.host_texture_surfaces.get_mut(&(task_id, texture_ref)) {
        e.source_gva = source_gva;
    }
}

pub fn get_texture(
    state: &DeviceState,
    task_id: u32,
    texture_ref: u32,
    width: u32,
    height: u32,
) -> Option<&[u8]> {
    get_from(
        &state.host_texture_surfaces,
        &(task_id, texture_ref),
        width,
        height,
    )
}

/// The address the entry behind [`get_texture`] was produced over, or `None`
/// when no entry answers at this geometry.
///
/// Separate from [`get_texture`] rather than returned beside it because the
/// serve site holds a shared borrow of `state` across the pixel slice, and the
/// question is asked once per serve while the slice is used per texel.
pub fn texture_source_gva(
    state: &DeviceState,
    task_id: u32,
    texture_ref: u32,
    width: u32,
    height: u32,
) -> Option<u64> {
    get_from(
        &state.host_texture_surfaces,
        &(task_id, texture_ref),
        width,
        height,
    )?;
    state
        .host_texture_surfaces
        .get(&(task_id, texture_ref))
        .map(|e| e.source_gva)
}

pub fn evict_texture(state: &mut DeviceState, task_id: u32, texture_ref: u32) {
    state.host_texture_surfaces.remove(&(task_id, texture_ref));
}

/// The identity of one linear compute window: which object it is, where its
/// guest backing sits, and the shape the guest declared over it.
///
/// Every `DeviceState::host_linear_textures` entry is *keyed* by
/// `(task_id, texture_ref)` and *described* by the remaining five fields, and
/// the cache serves an entry only while the asking window still describes it.
/// A window the guest re-declared at another address, format or geometry is a
/// different window; serving the old bytes for it hands a shader the content of
/// something else.
///
/// It travels as one value because the five descriptor fields are meaningless
/// apart — every operation here needs all of them, and the four that used to
/// take them positionally could be, and were, called with two of them swapped
/// without anything noticing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearWindow {
    pub task_id: u32,
    pub texture_ref: u32,
    pub gva: u64,
    pub pixel_format: u16,
    pub width: u32,
    pub height: u32,
    pub row_stride: u64,
}

impl LinearWindow {
    /// The `host_linear_textures` key this window addresses.
    fn key(&self) -> (u32, u32) {
        (self.task_id, self.texture_ref)
    }

    /// Bytes per pixel, but only for a window that can hold content at all:
    /// a named object over a real address, with a nonzero extent and a stride
    /// that reaches at least one tight row.
    ///
    /// This is the whole precondition both store paths share. `None` is a
    /// refusal to create an entry, not a missing format lookup.
    fn storable_bpp(&self) -> Option<u32> {
        let bpp = crate::contract::pixel_format::bytes_per_pixel(self.pixel_format)?;
        let ok = self.texture_ref != 0
            && self.gva != 0
            && self.width != 0
            && self.height != 0
            && self.row_stride >= (self.width as u64).saturating_mul(bpp as u64);
        ok.then_some(bpp)
    }

    /// Length of one tightly packed image of this window, or `None` on overflow.
    fn tight_len(&self, bpp: u32) -> Option<usize> {
        (self.width as usize)
            .checked_mul(self.height as usize)?
            .checked_mul(bpp as usize)
    }

    /// Whether `entry` is the window this describes, rather than a later one
    /// the guest re-declared over the same `(task, ref)` key.
    fn describes(&self, entry: &crate::model::HostLinearTexture) -> bool {
        entry.gva == self.gva
            && entry.pixel_format == self.pixel_format
            && entry.width == self.width
            && entry.height == self.height
            && entry.row_stride == self.row_stride
    }

    /// Take this window as `entry`'s descriptor, dropping whatever content the
    /// previous descriptor held. Both store paths then set their own
    /// generations; nothing else may reach these fields.
    fn adopt(&self, entry: &mut crate::model::HostLinearTexture) {
        entry.gva = self.gva;
        entry.pixel_format = self.pixel_format;
        entry.width = self.width;
        entry.height = self.height;
        entry.row_stride = self.row_stride;
        entry.bytes.clear();
    }
}

/// Store tight raw compute content for a type-2/3 texture object.
///
/// This is the discrete GPU-private body. It deliberately survives
/// MapMemory2/UnmapMemory; the guest GVA pages are only a pageable alias.
pub fn store_linear_texture(state: &mut DeviceState, w: &LinearWindow, bytes: &[u8]) -> bool {
    let Some(need) = w.storable_bpp().and_then(|bpp| w.tight_len(bpp)) else {
        return false;
    };
    if bytes.len() < need {
        return false;
    }
    let entry = state.host_linear_textures.entry(w.key()).or_default();
    entry.host_gen = entry.host_gen.wrapping_add(1);
    if entry.host_gen == 0 {
        entry.host_gen = 1;
    }
    w.adopt(entry);
    entry.bytes.extend_from_slice(&bytes[..need]);
    entry.resident_gen = 0;
    true
}

/// Deferred linear writeback: the engine's pinned resident storage image at
/// `generation` becomes the authoritative content for this window; no bytes are
/// stored. Same window precondition as [`store_linear_texture`].
pub fn note_linear_texture_resident(
    state: &mut DeviceState,
    w: &LinearWindow,
    generation: u32,
) -> bool {
    if generation == 0 || w.storable_bpp().is_none() {
        return false;
    }
    let entry = state.host_linear_textures.entry(w.key()).or_default();
    entry.host_gen = generation;
    w.adopt(entry);
    entry.resident_gen = generation;
    true
}

/// Resident generation of a linear window when the entry is still the one this
/// window describes and is resident-authoritative (deferred writeback).
pub fn linear_texture_resident_gen(state: &DeviceState, w: &LinearWindow) -> Option<u32> {
    let entry = state.host_linear_textures.get(&w.key())?;
    if entry.resident_gen == 0 || !w.describes(entry) {
        return None;
    }
    Some(entry.resident_gen)
}

/// Why [`materialize_linear_resident`] did not land the flushed bytes.
///
/// The distinction the caller needs is whether the cache entry ends up holding
/// this frame. `Superseded` is the one arm where it does not matter — the defer
/// has already been overtaken, so there is no frame here to lose. Every other
/// arm means a live entry stayed empty, and `flush_linear_one` then decides
/// whether to write guest pages on the assumption that it did not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearMaterializeDecline {
    /// The (task, ref) entry is gone, or a newer defer replaced this
    /// generation. Expected control flow: the content this flush carries is no
    /// longer the content anything would read.
    Superseded { resident_gen: u32 },
    /// The entry's own format has no bytes-per-pixel, so the tight size it
    /// wants cannot be computed.
    FormatUnsized { pixel_format: u16 },
    /// `width * height * bpp` overflowed `usize`.
    TightSizeOverflow { width: u32, height: u32, bpp: u32 },
    /// The engine returned fewer bytes than one tight image.
    ReadbackShort { got: usize, need: usize },
}

impl crate::observe::Decline for LinearMaterializeDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::Superseded { .. } => "linear_materialize_superseded",
            Self::FormatUnsized { .. } => "linear_materialize_format_unsized",
            Self::TightSizeOverflow { .. } => "linear_materialize_tight_size_overflow",
            Self::ReadbackShort { .. } => "linear_materialize_readback_short",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Superseded { resident_gen } => {
                vec![("resident_gen", resident_gen.to_string())]
            }
            Self::FormatUnsized { pixel_format } => {
                vec![("pixel_format", format!("{pixel_format:#x}"))]
            }
            Self::TightSizeOverflow { width, height, bpp } => vec![
                ("width", width.to_string()),
                ("height", height.to_string()),
                ("bpp", bpp.to_string()),
            ],
            Self::ReadbackShort { got, need } => {
                vec![("got", got.to_string()), ("need", need.to_string())]
            }
        }
    }
}

/// Land flushed resident bytes into the entry (tight rows), clearing the
/// resident-authoritative marker.
///
/// The `Err` is load-bearing rather than informational. `flush_linear_one` may
/// go on to refuse the guest write, and it is allowed to do that only because
/// this call left the authoritative bytes in the cache — so a caller that
/// cannot tell success from failure is asserting the one thing it needs to
/// know. Every arm but `Superseded` means a live entry was left empty.
pub fn materialize_linear_resident(
    state: &mut DeviceState,
    task_id: u32,
    texture_ref: u32,
    generation: u32,
    bytes: &[u8],
) -> Result<(), LinearMaterializeDecline> {
    let Some(entry) = state.host_linear_textures.get_mut(&(task_id, texture_ref)) else {
        return Err(LinearMaterializeDecline::Superseded { resident_gen: 0 });
    };
    if entry.resident_gen != generation {
        return Err(LinearMaterializeDecline::Superseded {
            resident_gen: entry.resident_gen,
        });
    }
    let Some(bpp) = crate::contract::pixel_format::bytes_per_pixel(entry.pixel_format) else {
        return Err(LinearMaterializeDecline::FormatUnsized {
            pixel_format: entry.pixel_format,
        });
    };
    let Some(need) = (entry.width as usize)
        .checked_mul(entry.height as usize)
        .and_then(|n| n.checked_mul(bpp as usize))
    else {
        return Err(LinearMaterializeDecline::TightSizeOverflow {
            width: entry.width,
            height: entry.height,
            bpp,
        });
    };
    if bytes.len() < need {
        return Err(LinearMaterializeDecline::ReadbackShort {
            got: bytes.len(),
            need,
        });
    }
    entry.bytes.clear();
    entry.bytes.extend_from_slice(&bytes[..need]);
    entry.resident_gen = 0;
    Ok(())
}

/// Borrow a raw compute encode only while the entry is still the window this
/// describes.
pub fn get_linear_texture<'a>(state: &'a DeviceState, w: &LinearWindow) -> Option<&'a [u8]> {
    let entry = state.host_linear_textures.get(&w.key())?;
    if !w.describes(entry) {
        return None;
    }
    let bpp = crate::contract::pixel_format::bytes_per_pixel(w.pixel_format)?;
    let need = w.tight_len(bpp)?;
    (entry.bytes.len() >= need).then(|| &entry.bytes[..need])
}

/// True when [`mirror_linear_color_cache`] would republish this format into
/// the BGRA render-sample caches. Deferred linear writebacks are gated on
/// `!linear_mirrorable` so render-side consumers never lose the mirror.
pub fn linear_mirrorable(pixel_format: u16) -> bool {
    use crate::contract::pixel_format::{
        MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_BGRA8_UNORM_SRGB, MTL_FORMAT_RGBA8_UNORM,
        MTL_FORMAT_RGBA8_UNORM_SRGB,
    };
    matches!(
        pixel_format,
        MTL_FORMAT_RGBA8_UNORM
            | MTL_FORMAT_RGBA8_UNORM_SRGB
            | MTL_FORMAT_BGRA8_UNORM
            | MTL_FORMAT_BGRA8_UNORM_SRGB
    )
}

/// Mirror normalized 8-bit compute output into the established BGRA sample
/// caches so a later render view over the same object/GVA observes the encode.
///
/// Reads every field of `w` but `row_stride`: the mirrored caches hold tight
/// rows, so the source stride is the one part of the window identity that does
/// not travel with the content.
pub fn mirror_linear_color_cache<M: HostMemory + crate::runtime::host::HostOps>(
    state: &mut DeviceState,
    host: &mut M,
    w: &LinearWindow,
    bytes: &[u8],
) {
    use crate::contract::pixel_format::{
        MTL_FORMAT_BGRA8_UNORM, MTL_FORMAT_BGRA8_UNORM_SRGB, MTL_FORMAT_RGBA8_UNORM,
        MTL_FORMAT_RGBA8_UNORM_SRGB,
    };
    let Some(need) = (w.width as usize)
        .checked_mul(w.height as usize)
        .and_then(|n| n.checked_mul(RGBA8_BPP as usize))
    else {
        return;
    };
    if bytes.len() < need {
        return;
    }
    let mut bgra = bytes[..need].to_vec();
    match w.pixel_format {
        MTL_FORMAT_RGBA8_UNORM | MTL_FORMAT_RGBA8_UNORM_SRGB => {
            for px in bgra.chunks_exact_mut(RGBA8_BPP as usize) {
                px.swap(0, 2);
            }
        }
        MTL_FORMAT_BGRA8_UNORM | MTL_FORMAT_BGRA8_UNORM_SRGB => {}
        _ => return,
    }
    store_texture(
        state,
        w.task_id,
        w.texture_ref,
        w.width,
        w.height,
        bgra.clone(),
        w.gva,
    );
    let backing = gva_backing(state, host, w.task_id, w.gva, w.width, w.height);
    // `false`: this runs *before* the caller's guest write, so at this instant
    // the guest's pages do not hold these bytes. The caller calls
    // [`note_gva_landed`] once its write reports `Written`.
    store_gva_owned(state, w.gva, w.width, w.height, bgra, 0, backing, false);
}

/// Record that the guest's own pages now hold the bytes cached at `gva`, so the
/// byte cap may evict the entry.
///
/// The counterpart to storing with `guest_holds_bytes = false`. A writeback path
/// that caches before it writes cannot know the outcome at store time, and
/// leaving the entry permanently unevictable would turn every successful
/// writeback into a permanent cap reservation. Call this on the success arm
/// only — see [`crate::model::HostSurface::guest_holds_bytes`].
///
/// Quiet on a `gva` with no entry: a store that was refused upstream (zero
/// geometry, short buffer) leaves nothing to mark, and that is not a failure.
pub fn note_gva_landed(state: &mut DeviceState, gva: u64) {
    if let Some(entry) = state.host_gva_surfaces.get_mut(&gva) {
        entry.guest_holds_bytes = true;
    }
}

/// The guest page currently backing `gva` under `task_id`, page-aligned.
///
/// Returns `None` when the walk cannot name the backing at all — a zero or
/// degenerate geometry, a dead task, or an address that does not translate. A
/// `None` backing means the entry is simply not validatable, never that it is
/// fresh.
///
/// The same call [`gva_backing_state`] makes to check the entry later, so the
/// producer and the consumer cannot disagree about what names an allocation.
pub fn gva_backing<M: HostMemory>(
    state: &DeviceState,
    host: &M,
    task_id: u32,
    gva: u64,
    width: u32,
    height: u32,
) -> Option<GvaBacking> {
    if gva == 0 || width == 0 || height == 0 {
        return None;
    }
    // Resolved by slot index, which is what the dense walk this replaced did
    // (`visit_task_gva_pages`) and what `gva_backing_state` does when it
    // re-asks. `translate_task_gva` applies the `active`/`directory_pfn` test
    // itself.
    let task = state.tasks.get(task_id)?;
    let gpa = crate::runtime::gva_mem::translate_task_gva(host, task, gva, state.page_shift)?;
    Some(GvaBacking {
        task_id,
        first_gpa: gpa & page_mask(state.page_shift),
    })
}

/// Mask that clears the page offset for this guest page geometry.
const fn page_mask(page_shift: u32) -> u64 {
    !((1u64 << page_shift) - 1)
}

/// Store a type-2/3 encode in the GVA-keyed cache, with the decoded object
/// identity that produced it.
///
/// Type-2/type-3 wrappers are the same linear texture storage family when the
/// GVA and geometry match; unrelated nonzero object-type transitions still
/// identify a different resource class.
///
/// On discrete hosts this cache is the **GPU-private** texture content for that
/// VA. Guest MapMemory2 unmap/remap changes PFNs under the same GVA but does
/// **not** destroy the encode: nothing on the Unmap path touches this map,
/// deliberately — an unmapped VA is the normal state of the wallpaper class this
/// cache holds. [`gva_backing_state`] is what says whether the key still names
/// these pages.
#[allow(
    clippy::too_many_arguments,
    reason = "the cache identity is the GVA, its geometry, its producer, and its guest backing"
)]
pub fn store_gva_owned(
    state: &mut DeviceState,
    gva: u64,
    width: u32,
    height: u32,
    bgra: Vec<u8>,
    object_type: u8,
    backing: Option<GvaBacking>,
    guest_holds_bytes: bool,
) {
    if gva == 0 || !scanout_extent_ok(width, height) {
        return;
    }
    let need = (height as usize)
        .saturating_mul(width as usize)
        .saturating_mul(RGBA8_BPP as usize);
    if bgra.len() < need {
        return;
    }
    let generation = state.next_sampled_content_generation();
    // A store re-populates the identity, so any miss the byte cap could have
    // been charged for it is now a different question. Retiring the witness key
    // here keeps `gva_cap_wanted` a count of lookups the cap actually cost,
    // rather than one that keeps accruing against content that came back.
    state.gva_eviction_witness.note_restored(gva, width, height);
    let touch = state.next_gva_touch();
    let entry = state.host_gva_surfaces.entry(gva).or_default();
    entry.last_touch = touch;
    entry.host_gen = generation;
    entry.width = width;
    entry.height = height;
    // One of the two sites that change this map's byte total; see
    // [`DeviceState::gva_cache_bytes`]. The replaced entry's bytes are
    // reclaimed before the new ones are charged, so a store at an existing key
    // nets to the difference instead of double-counting. Applied to the device
    // below, once this borrow of the entry has ended.
    let reclaimed = entry.bgra.len();
    entry.bgra = std::sync::Arc::new(bgra);
    let charged = entry.bgra.len();
    entry.producer_object_type = object_type;
    // These bytes came from *this* backing, so it replaces whatever the
    // previous store recorded — including a `None` that says the walk could
    // not name it. Carrying the old list forward would let a validated entry
    // vouch for pixels it did not produce.
    // The old token was taken for the old page list. If these bytes came from
    // exactly the same pages it still watches exactly the right memory, and it
    // has to be kept: the host holds a freshly tracked set at "generation
    // unreadable" for a two-harvest startup window, so a token retired and
    // re-taken on every store never survives long enough to become readable.
    // Retiring unconditionally is why `gvac_gw_clean` was 0 of 201 331 lookups
    // on a 300 s boot — not because the guest had rewritten every entry, which
    // is how that zero was read, but because no set this rail ever created
    // outlived its own arming window.
    entry.backing = backing;
    // Per store, not sticky: a later flush that *does* reach guest RAM makes
    // the same address evictable again, and a `true` left over from an earlier
    // store would let the cap take pixels this one never landed.
    entry.guest_holds_bytes = guest_holds_bytes;
    charge_gva_cache_bytes(state, reclaimed, charged);
    enforce_gva_cache_cap(state, gva);
}

/// Move [`DeviceState::gva_cache_bytes`] by one entry's replacement.
///
/// Reclaim before charge so the running total never transiently exceeds the
/// real one, and saturating so a bookkeeping slip can only under-report — an
/// over-report would make the cap evict content it never needed to.
fn charge_gva_cache_bytes(state: &mut DeviceState, reclaimed: usize, charged: usize) {
    state.gva_cache_bytes = state
        .gva_cache_bytes
        .saturating_sub(reclaimed)
        .saturating_add(charged);
}

/// The GVA encode cache is over its byte cap and every entry is excluded from
/// eviction.
///
/// Not a loss — the opposite. It is the cap declining to take the only copy of
/// pixels the guest never received, which is what a GPU with no free memory
/// does: refuse, rather than discard a surface the client still holds. It is on
/// the fail channel because an over-cap map is a condition a reader has to be
/// able to see; a silent one would be indistinguishable from a cap that is
/// holding.
///
/// `bytes` above `cap` with a small `entries` means one or a few oversized
/// surfaces (`MAX_SCANOUT_DIM` admits 256 MiB each, so a single entry can exceed
/// the whole cap); with a large `entries` it means a workload whose unlanded
/// working set genuinely exceeds the bound, which is the reading that would
/// justify raising it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GvaCapDecline {
    NothingEvictable {
        bytes: usize,
        cap: usize,
        entries: usize,
    },
}

impl crate::observe::Decline for GvaCapDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::NothingEvictable { .. } => "gva_cache_cap_nothing_evictable",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::NothingEvictable {
                bytes,
                cap,
                entries,
            } => vec![
                ("bytes", bytes.to_string()),
                ("cap", cap.to_string()),
                ("entries", entries.to_string()),
            ],
        }
    }
}

impl GvaCapDecline {
    /// Latched on the **binary magnitude** of the overshoot, not on the byte
    /// total.
    ///
    /// The total moves on every store, so latching on it makes a map that is
    /// steadily growing over its cap emit one line per store — the flood this
    /// device's `fail_once` exists to prevent. The magnitude moves only when the
    /// overshoot doubles, so the log gets a line at 1x over, 2x, 4x, and so on:
    /// bounded to a few dozen lines for the life of the process, while still
    /// showing a condition that is getting worse rather than one that has
    /// settled.
    fn emit(self) {
        let Self::NothingEvictable { bytes, cap, .. } = self;
        let over = bytes.saturating_sub(cap).max(1) as u64;
        crate::observe::Emit::decline("gva_cache_cap", &self)
            .fail_once(u64::from(over.next_power_of_two().trailing_zeros()));
    }
}

/// Hold [`DeviceState::host_gva_surfaces`] at or under
/// [`GVA_ENCODE_CACHE_BYTE_CAP`], evicting the least-recently-**used** entries
/// first.
///
/// Runs from [`store_gva_owned`], which is the map's only insert path, so the
/// bound is enforced exactly where it can be crossed. Two things it deliberately
/// does not do:
///
/// - **It never bulk-clears.** Draining to a 7/8 low-water mark, the same shape
///   [`crate::model::LruBytesMemo`] uses, means a steady insert stream evicts in
///   occasional batches with headroom instead of one-for-one at the boundary,
///   and a cap crossing never dumps the hot set — the re-encode cliff that
///   pattern exists to avoid.
/// - **It never evicts an entry the guest's pages do not also hold.** See
///   [`crate::model::HostSurface::guest_holds_bytes`]. That exclusion
///   only covers an address whose writeback has not run yet; once it runs and
///   *refuses* — which the page-ownership guard permits precisely because this
///   cache keeps the content — the entry would otherwise become an ordinary
///   candidate while still being the only copy. Same exclusion, one step later
///   in the same lifetime.
///
/// When those exclusions leave nothing to take, the map stays over its cap and
/// says so rather than evicting into them. That is the intended shape: a GPU
/// whose memory is full refuses, it does not discard a surface the client still
/// holds. The refusal is fail-visible so the over-cap condition is a reading
/// rather than a silence.
///
/// `protect` is the address the store that triggered this just wrote, and it is
/// never evicted. Without it a single entry bigger than the low-water mark is
/// dropped by its own store — the map holds one entry, that entry is over, and
/// it is the only eviction candidate — so the surface is never cached at all.
/// That is reachable rather than theoretical: `MAX_SCANOUT_DIM` is 8192, so an
/// entry may be up to 256 MiB against a 112 MiB low-water mark. An oversized
/// entry therefore rides alone and over the cap, matching the sibling memo,
/// because refusing to cache a surface for being big is how a 4K wallpaper
/// stops being cached at all.
fn enforce_gva_cache_cap(state: &mut DeviceState, protect: u64) {
    let cap = state.gva_cache_byte_cap;
    let low_water = cap - cap / 8;
    // The running total, not a fresh sum: this runs on the store path, which is
    // the draw path. See [`DeviceState::gva_cache_bytes`] — the census
    // recomputes the real figure once a second and reports any divergence, so
    // trusting it here is checked rather than assumed.
    if state.gva_cache_bytes <= low_water {
        return;
    }
    // Coldest first. This only runs at the cap boundary, never on the steady
    // store path, so one ordered pass over the keys is acceptable.
    let mut by_touch: Vec<(u64, u64)> = state
        .host_gva_surfaces
        .iter()
        .filter(|(gva, e)| **gva != protect && e.guest_holds_bytes)
        .map(|(&gva, e)| (e.last_touch, gva))
        .collect();
    by_touch.sort_unstable();
    if by_touch.is_empty() {
        GvaCapDecline::NothingEvictable {
            bytes: state.gva_cache_bytes,
            cap,
            entries: state.host_gva_surfaces.len(),
        }
        .emit();
        return;
    }
    for (_, gva) in by_touch {
        // `evict_gva` maintains the running total, so this reads the live
        // figure each round rather than tracking a second copy of it.
        if state.gva_cache_bytes <= low_water {
            break;
        }
        let Some(e) = state.host_gva_surfaces.get(&gva) else {
            continue;
        };
        let (width, height) = (e.width, e.height);
        state.gva_eviction_witness.note_evicted(gva, width, height);
        evict_gva(state, gva);
    }
}

/// The one selection rule every GVA-cache read goes through: exact key, exact
/// geometry, enough bytes for it. Returns the entry and the byte length a
/// serve would hand out.
///
/// Pure — it does **not** charge the byte cap's harm witness. Probes that ask
/// "would this hit" ([`has_gva`], [`touch_gva`]) go through here directly, so
/// only a read that actually wanted the pixels is counted as harm; charging
/// here instead would count two or three times for one frame's single logical
/// lookup and make the figure uninterpretable.
fn lookup_gva(
    state: &DeviceState,
    gva: u64,
    width: u32,
    height: u32,
) -> Option<(&crate::model::HostSurface, usize)> {
    let need = (height as usize)
        .saturating_mul(width as usize)
        .saturating_mul(RGBA8_BPP as usize);
    let e = state.host_gva_surfaces.get(&gva)?;
    (e.width == width && e.height == height && !e.bgra.is_empty() && e.bgra.len() >= need)
        .then_some((e, need))
}

/// [`lookup_gva`] for the paths that want the bytes, charging a miss to the
/// byte cap when the cap is what removed this exact identity.
///
/// A key that was never cached, or whose geometry never matched, is an ordinary
/// miss and is not counted — see [`crate::model::GvaEvictionWitness`].
fn read_gva(
    state: &DeviceState,
    gva: u64,
    width: u32,
    height: u32,
) -> Option<(&crate::model::HostSurface, usize)> {
    let hit = lookup_gva(state, gva, width, height);
    if hit.is_none() {
        state.gva_eviction_witness.note_miss(gva, width, height);
    }
    hit
}

/// Mark a GVA entry most-recently-used, so the byte cap's eviction reaches only
/// entries nothing is reading.
///
/// Call on a **confirmed serve**, not on an attempted one: this is the half of
/// the recency signal that keeps a stored-once-sampled-forever entry (the
/// retained wallpaper class) alive, and charging recency for lookups that were
/// then refused would keep entries warm that nothing can actually use.
pub fn touch_gva(state: &mut DeviceState, gva: u64, width: u32, height: u32) {
    if lookup_gva(state, gva, width, height).is_none() {
        return;
    }
    let stamp = state.next_gva_touch();
    if let Some(e) = state.host_gva_surfaces.get_mut(&gva) {
        e.last_touch = stamp;
    }
}

pub fn get_gva(state: &DeviceState, gva: u64, width: u32, height: u32) -> Option<&[u8]> {
    get_gva_with_gen(state, gva, width, height).map(|(bgra, _)| bgra)
}

/// Whether a [`get_gva`] for this key would hit, without borrowing the bytes.
///
/// Lets a caller that needs `&mut DeviceState` (backing revalidation) find out
/// first whether there is anything to revalidate.
pub fn has_gva(state: &DeviceState, gva: u64, width: u32, height: u32) -> bool {
    lookup_gva(state, gva, width, height).is_some()
}

/// Borrow a GVA encode plus its producer generation.
///
/// This is diagnostic provenance for the linear-sample loss proxy; selection
/// semantics are identical to [`get_gva`].
fn get_gva_with_gen(
    state: &DeviceState,
    gva: u64,
    width: u32,
    height: u32,
) -> Option<(&[u8], u64)> {
    let (e, need) = read_gva(state, gva, width, height)?;
    Some((&e.bgra[..need], e.host_gen))
}

/// Explicit drop (tests / object delete). Unmap does **not** come through here;
/// see [`store_gva_owned`] for why the map is retained across it.
pub fn evict_gva(state: &mut DeviceState, gva: u64) {
    if let Some(entry) = state.host_gva_surfaces.remove(&gva) {
        // The other site that changes this map's byte total; see
        // [`DeviceState::gva_cache_bytes`].
        state.gva_cache_bytes = state.gva_cache_bytes.saturating_sub(entry.bgra.len());
    }
}

/// Drop both host-side pixel copies that can name one linear texture target.
///
/// Once a writer publishes new pixels into the guest pages, those pages are
/// authoritative. Keeping either the address-keyed copy or the object-keyed
/// copy would let a later sample observe the frame that preceded the write.
pub fn forget_gva_copies(state: &mut DeviceState, task_id: u32, target_gva: u64, texture_ref: u32) {
    evict_gva(state, target_gva);
    if texture_ref != 0 {
        evict_texture(state, task_id, texture_ref);
    }
}

/// Entry count and resident bytes of one host-side pixel cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheLevel {
    pub entries: u64,
    pub bytes: u64,
    /// Bytes held by the single largest entry — the figure that separates "many
    /// small surfaces" from "a few 4K ones", which cost ~4x a 1080p entry each.
    pub largest: u64,
}

impl CacheLevel {
    fn of<'a, K: 'a, V: 'a>(
        map: &'a std::collections::BTreeMap<K, V>,
        len: impl Fn(&V) -> usize,
    ) -> Self {
        let mut level = Self {
            entries: map.len() as u64,
            ..Self::default()
        };
        for value in map.values() {
            let bytes = len(value) as u64;
            level.bytes += bytes;
            level.largest = level.largest.max(bytes);
        }
        level
    }
}

/// Resident size of every host-side pixel cache, right now.
///
/// # These are LEVELS, not per-interval counts
///
/// The opposite convention from `store_routes`, whose every field is a count for
/// one census interval and must be summed across lines. Summing these instead
/// would multiply a steady cache by the census cadence and report a leak that is
/// not there. **Take the last line for the current size, and `peak_bytes` for the
/// high-water mark**; the trend across lines is the thing to read.
///
/// `peak_bytes` is carried because a single last line cannot show a transient
/// spike, and a spike is what a resolution change produces: every geometry
/// change orphans the previous geometry's entries until something replaces or
/// evicts them.
///
/// # Why this exists
///
/// None of these maps has a size cap, and until now none had a counter either,
/// so "the host caches grow without bound" was neither refuted nor measurable —
/// `host_surfaces` alone is keyed by surface id with `remove()` on unmap/delete
/// and no bound on how many live ids there may be. This is the proxy for that
/// class, added before any attempt to cap it, because a cap chosen without a
/// measurement is a magic number.
///
/// Measure-only. Nothing may read this back to decide what to cache or evict:
/// that would make a resource gauge into a content heuristic.
///
/// `bytes` sums `Arc<Vec<u8>>` lengths, and a deferred render window can share
/// an entry's allocation rather than copying it — so a cache figure is the size
/// of the pixels reachable through the cache, not memory additional to the
/// windows.
///
/// # The surface tier reads zero by construction, and that zero is the finding
///
/// `surfaces`/`surface_bytes`/`surface_largest` are 0 on every census line of
/// every driven boot — 4 896 samples — while `gva` and `linear` beside them hold
/// 100 MB and 70 MB. That is not a dead tier and not a broken counter. It is
/// where the census samples:
///
/// - `note_cache_levels` runs in `lib.rs` at the tail of a drain tranche, after
///   `Device::drain` has returned.
/// - Inside that drain, every render Store lands its frame in guest pages
///   before the guest is told the work is done.
/// - Every one of those landings takes the leased frame — `render_flush_copied`
///   has never fired, `render_flush_leased` fires on every census line — so each
///   one writes through `mapping_write::write_bgra8_uncached`, whose
///   `CacheOutcome::Invalidate` calls [`forget`].
///
/// So every entry an arm put in has been reclaimed before the reader looks, and
/// the tier is guaranteed empty at exactly the instant it is measured. A
/// **non-zero** reading is therefore the alarm: it means a cache entry outlived
/// the window that armed it, which is the leak this counter was added to catch
/// and which nothing else in the device would report.
///
/// Two consequences a reader has to carry: `total_bytes` and `peak_bytes`
/// exclude this tier by construction, so they understate the host-side pixel
/// footprint by whatever the surface cache holds mid-drain (an 8.29 MB composite
/// frame, at ~95 a second). And anyone wanting the tier's real occupancy needs a
/// high-water mark maintained at insert time — a level read here cannot answer
/// it, and reading zero here is not evidence that it is small.
fn cache_levels(state: &DeviceState) -> (CacheLevel, CacheLevel, CacheLevel) {
    (
        CacheLevel::of(&state.host_surfaces, |e| e.bgra.len()),
        CacheLevel::of(&state.host_gva_surfaces, |e| e.bgra.len()),
        CacheLevel::of(&state.host_linear_textures, |e| e.bytes.len()),
    )
}

/// GVA-keyed entries whose key no longer translates to the backing the pixels
/// were produced from. Returns `(moved, unmapped, checked)`.
///
/// This replaced a `gva_cache_staleness` probe that counted dead-task and
/// unbacked entries. Both of its fields read zero on every census line of
/// every boot — 0 of 331 recorded at `model/state.rs`, and 0 across all 151
/// lines of a later driven boot — so task death cannot be the eviction rule
/// and the question moved to the backing itself. Do not reintroduce it.
///
/// A `GVA` is only a name for whatever the owning task's page table points it at
/// now. `GvaBacking::gpas` records what it pointed at when the pixels were
/// stored, and `get_gva_with_gen` serves on `(gva, exact geometry)` — so an entry
/// whose key now walks somewhere else is one no correct lookup can use: the
/// name has been handed to a different allocation. That is the same "drop only
/// what could never be served" standard the dead-task rule was reaching for,
/// applied where the evidence says to look.
///
/// - **`moved`** — the key translates, to a different page than recorded. The
///   guest reused the address.
/// - **`unmapped`** — the key does not translate at all. Counted apart because
///   the two are different guest actions and, more importantly, because a
///   transient walk failure looks exactly like this: `d455c3e`'s whole finding
///   was that the device answers before the guest has finished mapping, so a
///   *failure to translate* must never on its own authorise dropping content.
///   Only `moved` carries positive evidence that the address belongs to someone
///   else now.
/// - **`checked`** — entries with a usable backing and a live task, i.e. the
///   denominator. Without it a reader cannot tell "nothing moved" from "nothing
///   was examined", which is the failure direction that reads as a clean result.
///
/// Cost is one page-table walk per entry per census interval — the **first**
/// recorded page only, not the whole list. A whole-list walk of a 4K entry is
/// ~2 025 walks and this runs on the drain thread; the first page is enough to
/// tell a reused address from a retained one, and this is a measurement rather
/// than the authorisation for a write.
///
/// Measure-only. Nothing may evict on this yet: it exists to size the rule
/// before the rule is written.
fn gva_backing_moved<H: HostMemory>(state: &DeviceState, host: &H) -> (u64, u64, u64) {
    let (mut moved, mut unmapped, mut checked) = (0, 0, 0);
    for &gva in state.host_gva_surfaces.keys() {
        match gva_backing_state(state, host, gva) {
            GvaBackingState::Unrecorded => {}
            GvaBackingState::Same => checked += 1,
            GvaBackingState::Unmapped => {
                checked += 1;
                unmapped += 1;
            }
            GvaBackingState::Moved => {
                checked += 1;
                moved += 1;
            }
        }
    }
    (moved, unmapped, checked)
}

/// Whether one GVA-keyed entry's key still translates to the pages its pixels
/// were produced from.
///
/// The single spelling of that question. [`gva_backing_moved`] sums it over the
/// whole map for the level census; the colour LOAD seed asks it about the one
/// entry it is about to serve. Two spellings of "did this address move" would be
/// two answers, and the serve-side reading is only worth having if it is the
/// same reading the census reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GvaBackingState {
    /// No backing was recorded, or the task that recorded it is gone. The
    /// question cannot be asked, which is not the same as answering "fresh".
    Unrecorded,
    /// The key translates to the page it was stored over.
    Same,
    /// The key does not translate at all. Not evidence of reuse on its own —
    /// `d455c3e` found the device answers before the guest has finished
    /// mapping, and a transient walk failure looks exactly like this.
    Unmapped,
    /// The key translates, to a different page than recorded: the guest handed
    /// this address to another allocation. The only state that carries positive
    /// evidence these pixels belong to someone else.
    Moved,
}

/// [`GvaBackingState`] for one key. First recorded page only — enough to tell a
/// reused address from a retained one, and a whole-list walk of a 4K entry is
/// ~2 025 walks.
pub fn gva_backing_state<H: HostMemory>(
    state: &DeviceState,
    host: &H,
    gva: u64,
) -> GvaBackingState {
    let page_shift = state.page_shift;
    let Some(entry) = state.host_gva_surfaces.get(&gva) else {
        return GvaBackingState::Unrecorded;
    };
    let Some(backing) = entry.backing.as_ref() else {
        return GvaBackingState::Unrecorded;
    };
    let recorded = backing.first_gpa;
    // Same liveness test the walk itself applies: present in the table AND
    // flagged active. A dead task's page table cannot answer the question.
    let Some(task) = state.tasks.get(backing.task_id).filter(|t| t.active) else {
        return GvaBackingState::Unrecorded;
    };
    match crate::runtime::gva_mem::translate_task_gva(host, task, gva, page_shift) {
        None => GvaBackingState::Unmapped,
        Some(live) if (live & page_mask(page_shift)) != recorded => GvaBackingState::Moved,
        Some(_) => GvaBackingState::Same,
    }
}

/// Whether the GVA door may serve this key as `task_id`'s LOAD seed.
///
/// # Why the door needs a verdict it did not have
///
/// A LOAD seed is the attachment's *prior content*, and the matching Store
/// writes the composite back — so a door that hands a pass another allocation's
/// picture arms the next frame to load what this one stored. That is not a
/// one-frame flicker; it persists until something else repaints.
///
/// The ref door has always been gated on exactly that (`texture_source_gva ==
/// target_gva`). The GVA door was not, on the argument that "its key *is* the
/// allocation". **That argument has two holes and this closes both.**
///
/// - **A GVA is only an allocation inside one task's address space, and
///   [`crate::model::state::DeviceState::host_gva_surfaces`] is keyed by the
///   address alone.** Every sibling structure here carries the task and says
///   why — `guest_linear_memo` keys on `(task_id, gva, …)`, and `node_guard`'s
///   own doc puts it as "these pages belong to the task's address space, so a
///   reused id inheriting them would be watching memory that is now somebody
///   else's". This one neither carried it nor explained its absence.
/// - **[`gva_backing_state`] cannot see that collision**, so its measured zero
///   is not evidence against it. It resolves the page table from
///   `backing.task_id` — the task that *stored* the entry — so when another task
///   asks at the same address, the walk uses the storing task's table, finds the
///   page unchanged and answers [`GvaBackingState::Same`]. It is blind in
///   exactly the direction that reads as healthy.
///
/// So the freshness question and the ownership question are different questions,
/// and this asks both. `Moved` and `Unmapped` are refused for the reason the
/// sampled rung already refuses them — the guest handed the address to another
/// allocation and this cache is the stale side, where serving would be the
/// corruption rather than the repair. Refusing costs a guest re-read, never a
/// lost seed: the guest's own pages are the authoritative source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GvaSeedVerdict {
    /// The entry belongs to the asking task and its address still names the
    /// pages it was stored over.
    Admit,
    /// Recorded by a different task. A GVA means nothing across address spaces.
    OtherTask,
    /// The address now names different pages: positive evidence of reuse.
    Moved,
    /// The address does not translate, so ownership cannot be established.
    Unmapped,
    /// Nothing recorded, or the recording task is gone.
    Unrecorded,
    /// The guest's own pages hold these same bytes, so this copy can only be
    /// older than they are and never newer.
    ///
    /// Nothing in this map witnesses a guest CPU write — see
    /// [`crate::model::HostSurface::guest_holds_bytes`], where that gap is
    /// recorded. For an entry the guest's pages do *not* hold, the gap is a
    /// price worth paying: the copy is the only place those pixels exist and
    /// refusing it loses them. For an entry they do hold, there is nothing to
    /// buy. Both sources start equal, only one of them tracks the guest CPU,
    /// and the guest may write it with no device operation at all — so serving
    /// the copy can differ from the truth only in the direction of the past.
    ///
    /// Live shape of that difference: a Store lands the frame in the guest's
    /// pages and publishes it here, the guest CPU rasterizes into part of the
    /// layer, and the next pass's `MTLLoadActionLoad` seed takes this copy and
    /// loses everything the CPU wrote. The pass then Stores the result back
    /// over the guest's own bytes, so the loss is not a stale read that the
    /// next frame corrects — it is written into the layer.
    GuestHolds,
}

impl GvaSeedVerdict {
    /// The route name for this verdict, so the refusals are counted and a future
    /// session can price what the gate costs rather than argue it.
    pub fn route(self) -> &'static str {
        match self {
            Self::Admit => "gva_seed_admit",
            Self::OtherTask => "gva_seed_refused_other_task",
            Self::Moved => "gva_seed_refused_moved",
            Self::Unmapped => "gva_seed_refused_unmapped",
            Self::Unrecorded => "gva_seed_refused_unrecorded",
            Self::GuestHolds => "gva_seed_refused_guest_holds",
        }
    }
}

/// [`GvaSeedVerdict`] for one key, asked by the task that wants to serve it.
pub fn gva_seed_verdict<H: HostMemory>(
    state: &DeviceState,
    host: &H,
    task_id: u32,
    gva: u64,
) -> GvaSeedVerdict {
    let Some(entry) = state.host_gva_surfaces.get(&gva) else {
        return GvaSeedVerdict::Unrecorded;
    };
    let Some(backing) = entry.backing.as_ref() else {
        return GvaSeedVerdict::Unrecorded;
    };
    // Ownership before freshness: a walk of another task's page table answers a
    // question about another task's memory, however fresh it says the pages are.
    if backing.task_id != task_id {
        return GvaSeedVerdict::OtherTask;
    }
    // Before freshness, because it is not a freshness question: an entry the
    // guest's pages also hold has no answer this door needs. See
    // [`GvaSeedVerdict::GuestHolds`].
    if entry.guest_holds_bytes {
        return GvaSeedVerdict::GuestHolds;
    }
    match gva_backing_state(state, host, gva) {
        GvaBackingState::Same => GvaSeedVerdict::Admit,
        GvaBackingState::Moved => GvaSeedVerdict::Moved,
        GvaBackingState::Unmapped => GvaSeedVerdict::Unmapped,
        GvaBackingState::Unrecorded => GvaSeedVerdict::Unrecorded,
    }
}

/// Emit [`cache_levels`] at most once per census interval.
///
/// Shares the one-second cadence the drain census already runs on, so a boot's
/// cache trend lines up row-for-row with `store_routes` and `drain_duty`.
pub fn note_cache_levels<H: HostMemory>(state: &DeviceState, host: &H) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static LAST_MS: AtomicU64 = AtomicU64::new(0);
    static PEAK_BYTES: AtomicU64 = AtomicU64::new(0);

    let (surfaces, gva, linear) = cache_levels(state);
    let total = surfaces.bytes + gva.bytes + linear.bytes;
    let peak = PEAK_BYTES.fetch_max(total, Ordering::Relaxed).max(total);

    let now = crate::observe::elapsed_ms() as u64;
    let last = LAST_MS.load(Ordering::Relaxed);
    if now.saturating_sub(last) < 1000 {
        return;
    }
    // Losing the race only costs a skipped interval, never a double line.
    if LAST_MS
        .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return;
    }
    let (moved, unmapped, checked) = gva_backing_moved(state, host);
    // `gva_cap_*` are the only running totals on this line — the eviction
    // witness accumulates for the life of the device, where every other field
    // here is a level. Take the last line for all of them either way; do not
    // sum any of it.
    let (cap_evicted, cap_wanted, cap_forgotten) = state.gva_eviction_witness.counts();
    // The running total the cap actually tests against, minus the real sum this
    // census just computed for `gva_bytes`. Always 0; anything else means a
    // mutation site changed `bgra` without telling `gva_cache_bytes`, and the
    // cap is bounding a number that has stopped describing the map.
    let cap_drift = state.gva_cache_bytes as i64 - gva.bytes as i64;
    crate::observe::off(format!(
        "host_cache_levels (levels, not per-interval) total_bytes={total} peak_bytes={peak} \
         surfaces={} surface_bytes={} surface_largest={} \
         gva={} gva_bytes={} gva_largest={} \
         gva_backing_moved={moved} gva_backing_unmapped={unmapped} \
         gva_backing_checked={checked} \
         gva_cap_bytes={} gva_cap_drift={cap_drift} \
         gva_cap_evicted={cap_evicted} gva_cap_wanted={cap_wanted} \
         gva_cap_forgotten={cap_forgotten} \
         linear={} linear_bytes={} linear_largest={} \
         task_id_max={} task_id_cap=none mapping_id_max={} mapping_id_cap=none",
        surfaces.entries,
        surfaces.bytes,
        surfaces.largest,
        gva.entries,
        gva.bytes,
        gva.largest,
        state.gva_cache_byte_cap,
        linear.entries,
        linear.bytes,
        linear.largest,
        // The reach census for the id spaces — see
        // [`DeviceState::max_task_id_seen`]. Here rather than in a line of their
        // own because this is already the device's "levels, not per-interval"
        // line, and these are levels: the highest id the guest has named, beside
        // the bound that would have refused it.
        //
        // Both caps read the literal `none` because neither bound exists any
        // more: `is_mapping_id` refuses only the unbound sentinel, and the task
        // table is a map keyed by the guest's `u32`. The two reaches stay, and
        // are now occupancy readings on those maps rather than distances to a
        // refusal — they are the only thing that says how far the guest spreads
        // either id space.
        state.max_task_id_seen,
        state.max_mapping_id_seen,
    ));
}

#[cfg(test)]
mod tests;

/// The GVA encode cache's byte cap: what it bounds, what it refuses to touch,
/// and what it costs.
///
/// Every test here sets [`DeviceState::gva_cache_byte_cap`] to a size a test can
/// allocate. The policy under test is identical at 128 MiB; only the arithmetic
/// scales.
#[cfg(test)]
mod cap_tests;
