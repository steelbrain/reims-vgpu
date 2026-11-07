//! What GPU work may write: the host memory each object it binds or makes resident lies in
//! ([`Memory`]), and the slots its pipeline state's functions may store through, by the runtime's
//! slot writes of the state's descriptor ([`Reflection`], [`Slots`]), with whether a write
//! through a slot happens ([`Whether`]).
//!
//! A write through an object on a surface whose GPU writes never reach its memory
//! ([`crate::os::gpu_read_only`]) dies where Metal faults or drops it otherwise than this platform
//! ([`crate::os::dropped_write`]): a blit's (`crate::gpu::encode_blit`), a dispatch's
//! (`crate::gpu::predication`) and a draw's (`crate::gpu::encode_render`). So does a dispatch's
//! or a draw's write through a texture of a 4:4:4 two-plane YCbCr format whose writes the runtime
//! refuses ([`crate::gpu::ycbcr::refused_write`]).
//!
//! Read-only memory ([`Memory::kept`]) is of two kinds on Linux, by what the runtime is given for
//! it (`crate::gpu::backing::ImportSite::twins`):
//! - **A surface's read-only alias, and a range core maps read-only and makes a buffer over**
//!   ([`Kept::Always`], [`crate::os::TWINS_READ_ONLY`]; [`Kept::Until`] of a twinned buffer,
//!   [`crate::gpu::backing::Backing::twinned_now`]), which the runtime imports through a writable
//!   twin of its pages, in place, so a write through an object over it would reach those pages,
//!   where Metal's GPU drops it or faults: the command that may make it dies where it is encoded,
//!   at every extent ([`kept_write`]): a blit into such an object; a dispatch or a draw through one
//!   bound at a slot its function may store through, by its slot writes (every slot they leave
//!   undetermined), or made resident with write usage; and a pass whose attachment lies in one,
//!   whatever its actions, or that resolves into one or keeps its visibility results in one
//!   (`crate::gpu::encode_blit`, `crate::gpu::predication`, `crate::gpu::encode_render`); and a CPU
//!   write of such a texture (`replaceRegion:`, `crate::gpu::resource`). Only a slot the slot
//!   writes answer the function cannot store through ([`SlotWritability::Never`]) takes a read-only
//!   object without a death: they describe no type, so an argument whose type Metal's reflection is
//!   unmeasured for, which refuses the whole state's reflection, leaves every other slot answered.
//!   A function's write through an address it loads (an argument buffer's, a pointer's) into such
//!   an object not made resident for writing names nothing and is not seen: it reaches the pages.
//!   Core's own work, the one writer of its ranges, never writes them through the GPU, so no death
//!   comes there, and a copy would cost every commit and completion a compare of all their bytes
//!   (measured in `ImportSite::twins`'s docs).
//! - **The guest's read-only objects** (a read-only replace, a heap made with `MTLResourceOptions`
//!   bit 24 over address ranges, a no-copy buffer or heap made with it or a descriptor's over pages
//!   mapped read-only; [`Kept::Until`]; [`crate::os::COPIES_READ_ONLY`]), which the runtime makes
//!   over a copy of the pages whose GPU writes it drops, as Metal's GPU drops them: a write through
//!   a buffer over it is sent (a shader's store through a slot or a loaded address, a blit's fill
//!   or copy into the buffer, a pass's visibility results), and so is a pass whose attachment lies
//!   in it but does not store it (`DontCare`, `MultisampleResolve`); a pass that stores such an
//!   attachment (`Store`, `StoreAndMultisampleResolve`) fails its command buffer with Metal's page
//!   fault, and every later command buffer of its task dies at its commit, as Metal fails each with
//!   a code nothing recorded (`crate::gpu::encode_render`, `crate::gpu::command`); and a resolve
//!   into such a texture, a pass that keeps it by `CustomSampleDepthStore`, a write of a texture
//!   over it by a shader or a blit, and a CPU write of one die as [`kept_write`], as what Metal
//!   does there is unmeasured, or ends the process.
//!
//! On macOS neither is taken: MoltenVK hands Metal the pages, or a read-only view of them made
//! with bit 24, and Metal's GPU answers every write there itself.
//!
//! What Metal does there, measured on an M3 Max (macOS 27.0.1) by metal2vulkan's probes
//! (`read_only_view.rs`, `probes/readonly_gpu_writes.m`), 2026-10-25 and since: it completes a
//! draw or a dispatch whose shader stores may write through read-only memory (through a bound
//! slot or a loaded address, `drawIndexedPrimitives` among them), a blit fill into it and a pass
//! that keeps its visibility results in it, and drops those writes, over 8,314 runs; a colour,
//! depth or stencil attachment of it stored back (`Store`), a multisample primary among them,
//! fails the command buffer (`MTLCommandBufferErrorDomain`, a page fault) every run, the process
//! going on with every later command buffer failing too, with a code not recorded; the same
//! attachments under `DontCare` or `Load` complete, as does a resolve texture attached and not
//! resolved into and a resolve out of a read-only primary. A resolve into such memory is
//! unknown (no verdict over 520 processes), and so are a blit's copy into it and a texture's
//! writes by a shader or a blit there, which no probe made.
use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use metal2vulkan::metal::{BindingType, PixelFormat};
use metal2vulkan::pipeline::ComputePipelineDescriptor;
use metal2vulkan::reflection::{SlotWritability, SlotWrite, SlotWriteGap};
use reims_vgpu_core::num::ToU64;

use crate::answer::plain;
use crate::gpu::backing::Pages;
use crate::gpu::resource::{BufferObject, HeapObject, RenderDescriptor, TextureObject, Under};
use crate::os::OsSurface;

/// The memory an object's bytes lie in, which a write of the object may change.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Memory {
    /// The host memory: the first byte and end of each range. An object of none (a `Private`
    /// buffer, a texture of memory of its own) is written nowhere a guest's surface lies.
    pub ranges: Vec<Range<u64>>,
    /// Whether the GPU's writes through the object never reach the memory: the surface it was
    /// made on is one of [`crate::os::gpu_read_only`].
    pub dropped: bool,
    /// The format of the object, a texture over a surface, where the runtime refuses the GPU's
    /// writes through it: one of a 4:4:4 format of the two-plane YCbCr family whose writes it
    /// does not store as Metal's GPU does ([`crate::gpu::ycbcr::writes_refused`]).
    pub refused: Option<PixelFormat>,
    /// Whether the memory is read-only where the runtime does not answer the GPU's writes through
    /// the object as Metal's GPU does (the module docs).
    pub kept: Kept,
}

/// Whether an object's memory is read-only where the runtime does not answer the GPU's writes
/// through it as Metal's GPU does ([`Memory::kept`]), asked as work that may write through the
/// object is encoded ([`Kept::holds`]).
#[derive(Clone, Debug, Default)]
pub enum Kept {
    /// It is not.
    #[default]
    No,
    /// For the object's life: a surface's read-only memory, which the runtime imports through a
    /// writable twin of its pages, so the GPU's writes through the object reach them
    /// ([`surface_kept`]).
    Always,
    /// Until a detach leaves the buffer or heap in no memory: read-only memory the runtime
    /// imports through a writable twin of its pages, a range core maps read-only and makes a
    /// buffer over ([`crate::gpu::backing::Backing::twinned_now`]), or, for a texture over a
    /// buffer or heap, makes the buffer or heap over a copy of, dropping the GPU's writes, where
    /// what Metal does with a texture's writes is unmeasured
    /// ([`crate::gpu::backing::Backing::read_only_now`]); no replace gives it other memory while
    /// work encoded holds it, as the runtime refuses the replace (`crate::gpu::backing`). A
    /// buffer's own memory the runtime copies is never `Until`: the runtime drops its writes, as
    /// Metal's GPU does.
    Until(Arc<Pages>),
}

impl Kept {
    /// [`Kept::Always`] where `kept`, else [`Kept::No`].
    pub fn always(kept: bool) -> Kept {
        if kept { Kept::Always } else { Kept::No }
    }

    /// Whether the object's memory is read-only so now.
    pub fn holds(&self) -> bool {
        match self {
            Kept::No => false,
            Kept::Always => true,
            Kept::Until(pages) => !pages.zeroed(),
        }
    }
}

impl PartialEq for Kept {
    fn eq(&self, other: &Kept) -> bool {
        match (self, other) {
            (Kept::No, Kept::No) | (Kept::Always, Kept::Always) => true,
            (Kept::Until(a), Kept::Until(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl Eq for Kept {}

impl Memory {
    /// Whether it holds no memory.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// The memory of an object made on `surface`: all of the surface's
    /// ([`crate::os::surface_memory`]).
    pub fn of_surface(surface: &OsSurface) -> Memory {
        Memory {
            ranges: crate::os::surface_memory(surface),
            dropped: crate::os::gpu_read_only(surface),
            refused: None,
            kept: Kept::always(surface_kept(surface)),
        }
    }

    /// The memory of the bytes `bytes` (a first byte and an end, from its base address) of
    /// `surface` ([`crate::os::surface_bytes_memory`]).
    pub fn of_surface_bytes(surface: &OsSurface, bytes: Range<u64>) -> Memory {
        Memory {
            ranges: crate::os::surface_bytes_memory(surface, std::slice::from_ref(&bytes)),
            dropped: crate::os::gpu_read_only(surface),
            refused: None,
            kept: Kept::always(surface_kept(surface)),
        }
    }

    /// The memory of an object made over the host memory `ranges`, read-only as `kept` says.
    pub fn over(ranges: Vec<Range<u64>>, kept: Kept) -> Memory {
        Memory {
            ranges,
            dropped: false,
            refused: None,
            kept,
        }
    }

    /// What a bind keeps of it ([`Bound`]).
    pub fn bound(self) -> Bound {
        Bound {
            meets: !self.is_empty(),
            dropped: self.dropped,
            refused: self.refused,
            kept: self.kept,
        }
    }
}

/// What a bind or a residency keeps of an object's [`Memory`]: whether it lies in any, and what
/// the GPU's writes through it do there. What work encoded later may write is decided from these
/// alone (`crate::gpu::encode_render`'s `Reach`), never from the ranges.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Bound {
    /// Whether it lies in any memory: [`Memory::is_empty`] answers `false`.
    pub meets: bool,
    /// [`Memory::dropped`].
    pub dropped: bool,
    /// [`Memory::refused`].
    pub refused: Option<PixelFormat>,
    /// [`Memory::kept`].
    pub kept: Kept,
}

impl Bound {
    /// [`Memory::over`]'s, of memory it lies in where `meets`.
    fn over(meets: bool, kept: Kept) -> Bound {
        Bound {
            meets,
            kept,
            ..Bound::default()
        }
    }
}

/// Whether the runtime imports the memory of an object made on `surface` through a writable
/// twin of read-only pages: one whose GPU writes never reach its memory
/// ([`crate::os::gpu_read_only`]), every buffer over which lies over a read-only alias of its
/// allocation ([`crate::os::surface_alias`]), where the runtime twins a surface's read-only
/// alias ([`crate::os::TWINS_READ_ONLY`]).
fn surface_kept(surface: &OsSurface) -> bool {
    crate::os::TWINS_READ_ONLY && crate::os::gpu_read_only(surface)
}

/// Whether the writes through `t` are not answered as Metal's GPU answers them, its memory being
/// read-only ([`Memory::kept`]), as [`texture_memory`] answers, without the ranges: a surface's
/// or a buffer's the runtime twins, or a buffer's or heap's the runtime copies.
pub fn texture_kept(t: &TextureObject) -> bool {
    match (&t.surface, &t.under) {
        (Some(s), _) => surface_kept(s),
        (None, Some(Under::Buffer(b, _))) => under_buffer_kept(b).holds(),
        (None, Some(Under::Heap(h, _))) => h.backing.read_only_now(),
        _ => false,
    }
}

/// Whether the GPU's writes through `t` reach read-only memory through a writable twin of its
/// pages: `t` lies on a surface whose memory the runtime twins, or over a buffer whose memory it
/// twins ([`buffer_kept`]).
pub fn texture_twinned(t: &TextureObject) -> bool {
    match (&t.surface, &t.under) {
        (Some(s), _) => surface_kept(s),
        (None, Some(Under::Buffer(b, _))) => buffer_kept(b),
        _ => false,
    }
}

/// Whether the writes through `b` are not answered as Metal's GPU answers them, its memory being
/// read-only ([`Memory::kept`]), as [`buffer_memory`] answers, without the ranges: a surface's,
/// or a range's core made it over, the runtime twins. A buffer over read-only memory the runtime
/// copies is not: its GPU writes are dropped, as Metal's are.
pub fn buffer_kept(b: &BufferObject) -> bool {
    match (&b.surface, &b.heap) {
        (Some(s), _) => surface_kept(s),
        (None, Some(_)) => false,
        (None, None) => b.backing.twinned_now(),
    }
}

/// What a texture over `b` keeps of `b`'s memory ([`Memory::kept`]): a surface's twin, or the
/// read-only memory the runtime imports `b` over a twin of, or makes `b`, or the heap it lies
/// in, over a copy of.
fn under_buffer_kept(b: &BufferObject) -> Kept {
    match (&b.surface, &b.heap) {
        (Some(s), _) => Kept::always(surface_kept(s)),
        (None, Some(h)) => h.backing.kept(),
        (None, None) => b.backing.kept(),
    }
}

/// How many objects [`texture_kept`] and [`buffer_kept`] may answer true through now: buffers
/// and heaps over read-only memory the runtime twins or copies ([`set_kept`]), and surfaces whose
/// GPU writes never reach their memory where the runtime twins read-only memory (`Keeping`, on
/// Linux). Each is counted before it answers true and uncounted once it no longer can, so none
/// counted means they answer false for every object.
static KEPT_NOW: AtomicUsize = AtomicUsize::new(0);

/// Whether [`texture_kept`] or [`buffer_kept`] may answer true for any object now. Where not, they
/// answer false for every object, and a walk of what a command writes for an object over
/// read-only pages ([`kept_write`]) finds none.
pub(crate) fn any_kept() -> bool {
    KEPT_NOW.load(Ordering::SeqCst) != 0
}

/// Sets `flag`, whether a buffer's or heap's memory is read-only memory the runtime twins or
/// copies ([`crate::gpu::backing::Backing::read_only_now`]), to `kept`, keeping the count
/// ([`KEPT_NOW`]): counted before it becomes true, and uncounted once it becomes false.
pub(crate) fn set_kept(flag: &AtomicBool, kept: bool) {
    if kept {
        KEPT_NOW.fetch_add(1, Ordering::SeqCst);
        if flag.swap(true, Ordering::SeqCst) {
            KEPT_NOW.fetch_sub(1, Ordering::SeqCst);
        }
    } else if flag.swap(false, Ordering::SeqCst) {
        KEPT_NOW.fetch_sub(1, Ordering::SeqCst);
    }
}

/// One surface counted in [`KEPT_NOW`] while this lives: one whose GPU writes never reach its
/// memory ([`surface_kept`]), which every object made on it holds.
#[cfg(target_os = "linux")]
#[derive(Debug)]
pub(crate) struct Keeping(());

#[cfg(target_os = "linux")]
impl Keeping {
    /// A count of one, from now on.
    pub(crate) fn new() -> Keeping {
        KEPT_NOW.fetch_add(1, Ordering::SeqCst);
        Keeping(())
    }
}

#[cfg(target_os = "linux")]
impl Drop for Keeping {
    fn drop(&mut self) {
        KEPT_NOW.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Dies at `sel`, a command that may write `what` an object over read-only memory whose write
/// this platform does not answer as Metal's GPU does ([`Memory::kept`]): a surface the GPU only
/// reads, or a range core maps read-only, which the runtime imports through a writable twin of
/// its pages, where the write would land in those pages, and later reads see it; or, for a
/// texture's write, a resolve into one or a CPU write, read-only guest pages (a read-only replace
/// or heap, a resource made with `MTLResourceOptions` bit 24) the runtime copies, dropping the
/// write, where what Metal does is unmeasured. Metal's GPU drops a shader's store, a fill's or
/// a visibility result's there and completes the command, and fails the command buffer of a
/// pass that stores an attachment back to it (the module's measurements), and its CPU's write
/// through the pages' read-only mapping ends the process. Ending the process is this platform's
/// divergence from Metal (`docs/accuracy-ledger.md`, `read-only-pages-written`).
pub(crate) fn kept_write(sel: &'static str, what: &str) -> ! {
    crate::answer::unrepresentable(
        sel,
        &format!(
            "{what} read-only memory (a surface the GPU only reads or a guest range, which this \
             host's runtime imports through a writable twin, so the write would reach it; or, \
             for a texture, a read-only replace, heap or resource, whose copy drops it), where \
             Metal's GPU drops the write or faults, or what it does is unmeasured, and its CPU's \
             write through the read-only mapping ends the process"
        ),
    )
}

/// The host memory `t`'s texels lie in, which a write of `t` may change: its plane's bytes of its
/// surface ([`plane_bytes`]), or its bytes of its buffer's or its heap's ([`Under`]); none for one
/// of memory of its own.
pub fn texture_memory(t: &TextureObject) -> Memory {
    match (&t.surface, &t.under) {
        (Some(s), _) => match &t.plane {
            Some(p) => {
                // A rotated texture's rows may run along either side.
                let (width, height) = plain(|| (t.raw.width(), t.raw.height()));
                let rows = if plain(|| t.raw.rotation()) == 0 {
                    height
                } else {
                    width.max(height)
                };
                let pf = plain(|| t.raw.pixel_format());
                Memory {
                    refused: crate::gpu::ycbcr::writes_refused(pf).then_some(pf),
                    ..Memory::of_surface_bytes(s, plane_bytes(p.offset, p.bytes_per_row, rows))
                }
            }
            None => Memory::of_surface(s),
        },
        (None, Some(Under::Buffer(b, bytes))) => Memory {
            kept: under_buffer_kept(b),
            ..buffer_bytes_memory(b, bytes.clone())
        },
        (None, Some(Under::Heap(h, bytes))) => heap_bytes_memory(h, bytes.clone()),
        _ => Memory::default(),
    }
}

/// The bytes of a surface, from its base address, that a texture of `rows` rows over a plane
/// at `offset` with rows `pitch` bytes apart (its `bytesPerRow`) reads and writes: its rows from
/// the offset, `pitch` apart, as the CPU lays them out, and from the offset rounded down to 16,
/// rows `pitch` rounded down to 16 apart, or 16 where that is 0, as the GPU does (the runtime's
/// `linear_plan.rs`, measured).
fn plane_bytes(offset: u64, pitch: u64, rows: u64) -> Range<u64> {
    let start = offset - offset % 16;
    let gpu_pitch = match pitch - pitch % 16 {
        0 => 16,
        p => p,
    };
    let cpu_end = offset.saturating_add(pitch.saturating_mul(rows));
    let gpu_end = start.saturating_add(gpu_pitch.saturating_mul(rows));
    start..cpu_end.max(gpu_end)
}

/// The host memory `b`'s bytes lie in now ([`buffer_bytes_memory`]).
pub fn buffer_memory(b: &BufferObject) -> Memory {
    buffer_bytes_memory(b, 0..plain(|| b.raw.length()))
}

/// The host memory `b`'s bytes `bytes` (a first byte and an end) lie in now: in its surface's
/// memory (`newBufferWithIOSurface:`, from the surface's base address), at its place in its
/// heap's, in the guest's pages of its last replace, or in the memory it was made over; none
/// where a detach left it in no memory, or for a `Private` one.
pub fn buffer_bytes_memory(b: &BufferObject, bytes: Range<u64>) -> Memory {
    if let Some(s) = &b.surface {
        return Memory::of_surface_bytes(s, bytes);
    }
    if let Some(h) = &b.heap {
        // A heap buffer's GPU address is its heap's plus its offset there.
        let at = plain(|| b.raw.gpu_address().0.saturating_sub(h.raw.gpu_address().0));
        return Memory {
            kept: Kept::No,
            ..heap_bytes_memory(
                h,
                at.saturating_add(bytes.start)..at.saturating_add(bytes.end),
            )
        };
    }
    // Its writes into read-only memory the runtime copies are dropped, as Metal's GPU drops them:
    // kept only where it twins them ([`Kept::Until`]).
    let kept = b.backing.twin_kept();
    let paged = b.backing.in_pages(&bytes);
    if paged.is_some() || b.backing.kept_contents().is_some() {
        return Memory::over(paged.unwrap_or_default(), kept);
    }
    let contents = plain(|| b.raw.contents()).map(|first| {
        let first = first.as_ptr().addr().to_u64();
        std::iter::once(first.saturating_add(bytes.start)..first.saturating_add(bytes.end))
            .collect()
    });
    Memory::over(contents.unwrap_or_default(), kept)
}

/// [`buffer_bound`] where it may decide what work through `b` asks: `b` lies on a surface, or
/// over memory the runtime twins ([`buffer_kept`]); elsewhere, where its [`Bound`] holds no
/// surface, no refused format and nothing kept, which a slot's bind keeps nothing of, the
/// default one, of no memory, read without asking the runtime, as each draw binds its buffers
/// anew.
pub fn buffer_bind(b: &BufferObject) -> Bound {
    if buffer_bind_reads(b) {
        buffer_bound(b)
    } else {
        Bound::default()
    }
}

/// Whether [`buffer_bind`] reads `b`'s [`buffer_bound`]: `b` lies on a surface, or over memory
/// the runtime twins. Else it answers the default one: a buffer over read-only memory the runtime
/// copies keeps nothing either, as its GPU writes are dropped, as Metal's are.
pub fn buffer_bind_reads(b: &BufferObject) -> bool {
    b.surface.is_some() || buffer_kept(b)
}

/// [`texture_bound`] where it may decide what work through `t` asks, as [`buffer_bind`] reads a
/// buffer's: `t` lies on a surface, over a buffer on one, or over a buffer or in a heap over
/// read-only memory the runtime copies; elsewhere the default one, of no memory.
pub fn texture_bind(t: &TextureObject) -> Bound {
    let kept = match (&t.surface, &t.under) {
        (Some(_), _) => true,
        (None, Some(Under::Buffer(b, _))) => b.surface.is_some() || under_buffer_kept(b).holds(),
        (None, Some(Under::Heap(h, _))) => h.backing.read_only_now(),
        (None, None) => false,
    };
    if kept {
        texture_bound(t)
    } else {
        Bound::default()
    }
}

/// What a bind keeps of `b`'s memory: [`buffer_memory`]'s [`Bound`], read as it reads it, but
/// without collecting the ranges of its bytes in the guest's pages, one a page run, as each draw
/// binds its buffers anew.
pub fn buffer_bound(b: &BufferObject) -> Bound {
    buffer_bytes_bound(b, 0..plain(|| b.raw.length()))
}

/// [`buffer_bytes_memory`]'s [`Bound`] of `b`'s bytes `bytes`, read as it reads them, without
/// collecting their ranges ([`buffer_bound`]).
fn buffer_bytes_bound(b: &BufferObject, bytes: Range<u64>) -> Bound {
    if let Some(s) = &b.surface {
        return surface_bound(s, None);
    }
    if let Some(h) = &b.heap {
        // As `buffer_bytes_memory`'s: its bytes at its place in its heap's, nothing kept.
        let at = plain(|| b.raw.gpu_address().0.saturating_sub(h.raw.gpu_address().0));
        return Bound {
            kept: Kept::No,
            ..heap_bytes_bound(
                h,
                &(at.saturating_add(bytes.start)..at.saturating_add(bytes.end)),
            )
        };
    }
    let kept = b.backing.twin_kept();
    let paged = b.backing.meets_pages(&bytes);
    if paged.is_some() || b.backing.kept_contents().is_some() {
        return Bound::over(paged.unwrap_or(false), kept);
    }
    Bound::over(plain(|| b.raw.contents()).is_some(), kept)
}

/// [`heap_bytes_memory`]'s [`Bound`] of `h`'s bytes `bytes`, without collecting their ranges.
fn heap_bytes_bound(h: &HeapObject, bytes: &Range<u64>) -> Bound {
    Bound::over(
        h.backing.meets_pages(bytes).unwrap_or(false),
        h.backing.kept(),
    )
}

/// The [`Bound`] of an object made on `surface`, of the `refused` format where the runtime
/// refuses its writes: [`Memory::of_surface`]'s and [`Memory::of_surface_bytes`]'s, which always
/// hold the surface's own allocation ([`crate::os::surface_memory`]), so it meets memory.
fn surface_bound(surface: &OsSurface, refused: Option<PixelFormat>) -> Bound {
    Bound {
        meets: true,
        dropped: crate::os::gpu_read_only(surface),
        refused,
        kept: Kept::always(surface_kept(surface)),
    }
}

/// What a bind or a residency keeps of `t`'s memory: [`texture_memory`]'s [`Bound`], read as it
/// reads it, without collecting the ranges of its texels in the guest's pages, one a page run,
/// as each draw binds its textures anew.
pub fn texture_bound(t: &TextureObject) -> Bound {
    match (&t.surface, &t.under) {
        (Some(s), _) => {
            let refused = t.plane.as_ref().and_then(|_| {
                let pf = plain(|| t.raw.pixel_format());
                crate::gpu::ycbcr::writes_refused(pf).then_some(pf)
            });
            surface_bound(s, refused)
        }
        (None, Some(Under::Buffer(b, bytes))) => Bound {
            kept: under_buffer_kept(b),
            ..buffer_bytes_bound(b, bytes.clone())
        },
        (None, Some(Under::Heap(h, bytes))) => heap_bytes_bound(h, bytes),
        _ => Bound::default(),
    }
}

/// The host memory of `h`'s bytes `bytes`, a first byte and an end: in the guest's pages it was
/// made over or last replaced with; none for one of memory of its own. What it keeps is a
/// texture's in it ([`Kept::Until`]); a buffer in it keeps nothing ([`buffer_bytes_memory`]).
pub(crate) fn heap_bytes_memory(h: &HeapObject, bytes: Range<u64>) -> Memory {
    Memory::over(
        h.backing.in_pages(&bytes).unwrap_or_default(),
        h.backing.kept(),
    )
}

/// What one stage's function may store through each slot its arguments bind, by its pipeline's
/// slot writes (the runtime's `slot_writes`, which Metal's reflection answers as each binding's
/// `access`): one at a texture slot it declares `access::write` or `access::read_write`, and at a
/// buffer slot it declares a `device` pointer that is not `const` (an array's slots, each of its
/// elements'); none at a slot it declares read-only (a texture read or sampled, a `constant` or
/// `const device` buffer), or that no argument it references binds. Unlike the reflection, which
/// the runtime refuses for the whole state where Metal's answer for one argument's type is
/// unmeasured, the slot writes describe no type: they leave undetermined only the slots an
/// argument whose own slot or role is undetermined may cover ([`SlotWriteGap`]).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Slots {
    /// The function's arguments that bind a slot, in its order, but one it never references
    /// whose answer is determined, which it stores through nowhere. One whose answer is
    /// undetermined is kept, as the runtime keeps one in its texture accesses.
    args: Vec<SlotWrite>,
}

impl Slots {
    /// The slots one stage's function may store through, of its slot writes `args`.
    pub fn of(args: &[SlotWrite]) -> Slots {
        Slots {
            args: (args.iter())
                .filter(|a| a.used || a.writes.gap().is_some())
                .cloned()
                .collect(),
        }
    }

    /// Whether the function may store through the `kind` slot `slot` ([`SlotWrite::may_write`]).
    /// A slot past `u32`, which no argument's index reaches, and whose bind dies before any work
    /// runs (`crate::gpu::encode_render`'s `small`), answers that it may.
    fn at(&self, kind: BindingType, slot: u64) -> SlotWritability {
        match u32::try_from(slot) {
            Ok(slot) => SlotWrite::may_write(&self.args, kind, slot),
            Err(_) => SlotWritability::Maybe,
        }
    }

    /// Whether the function may store through texture slot `slot`.
    pub fn texture(&self, slot: u64) -> SlotWritability {
        self.at(BindingType::Texture, slot)
    }

    /// Whether the function may store through buffer slot `slot`.
    pub fn buffer(&self, slot: u64) -> SlotWritability {
        self.at(BindingType::Buffer, slot)
    }
}

/// The vertex function's index in a render pipeline state's [`Reflection::slots`].
pub const VERTEX: usize = 0;
/// The fragment function's.
pub const FRAGMENT: usize = 1;
/// The tile function's.
pub const TILE: usize = 2;
/// A mesh render pipeline state's object function's.
pub const OBJECT: usize = 3;
/// A mesh render pipeline state's mesh function's.
pub const MESH: usize = 4;
/// How many render stages [`Reflection::slots`] answers for: [`VERTEX`] to [`MESH`].
pub const RENDER_STAGES: usize = 5;

/// The stages of a render or tile pipeline state whose functions bind slots, in the order a
/// draw's writes are asked ([`Reflection::stages`]); also those of a state no reflection here
/// tells, as an execution's own states.
pub(crate) const DRAWING_STAGES: &[usize] = &[VERTEX, FRAGMENT, TILE];

/// The stages of a mesh render pipeline state, in pipeline order.
const MESH_STAGES: &[usize] = &[OBJECT, MESH, FRAGMENT];

/// The descriptor a pipeline state was made of, whose slot writes tell its functions' [`Slots`].
#[derive(Debug)]
pub enum Reflected {
    /// A compute pipeline state's: its kernel's.
    Compute(Box<ComputePipelineDescriptor>),
    /// A render, tile or mesh render pipeline state's: its vertex, fragment, tile, object and
    /// mesh functions', in that order ([`VERTEX`] to [`MESH`]).
    Render(RenderDescriptor),
}

/// The [`Slots`] of a pipeline state's functions, by its descriptor's slot writes, read the
/// first time something asks. The state holds it, and so does an encoder that set the state, as
/// Metal's encoder holds the state it runs, the guest's state gone or not.
#[derive(Debug)]
pub struct Reflection {
    /// The descriptor; `None` for a state no creator of the guest's made.
    descriptor: Option<Reflected>,
    /// Each function's slots, read once.
    slots: std::sync::OnceLock<Result<Vec<Slots>, String>>,
}

impl Reflection {
    /// The reflection of a state made of `descriptor`.
    pub fn new(descriptor: Option<Reflected>) -> Arc<Reflection> {
        Arc::new(Reflection {
            descriptor,
            slots: std::sync::OnceLock::new(),
        })
    }

    /// The descriptor the state was made of; `None` for a state no creator of the guest's made.
    pub fn descriptor(&self) -> Option<&Reflected> {
        self.descriptor.as_ref()
    }

    /// Each function's [`Slots`]: a compute state's kernel's alone, a render state's vertex,
    /// fragment, tile, object and mesh functions' ([`VERTEX`] to [`MESH`]), none for a stage the
    /// state has no function of; `Err` naming why the runtime answers none, for a state of no
    /// guest descriptor's, or where it cannot decode or specialize a function, which it refuses
    /// alone: a function whose argument types Metal's reflection is unmeasured for still answers,
    /// where its reflection is refused. A render or mesh render state with rasterization off runs
    /// no fragment function, whose slots it writes none of, though its descriptor keeps the
    /// function (`crate::gpu::serialized`).
    pub fn slots(&self) -> Result<&[Slots], &str> {
        let read = self.slots.get_or_init(|| {
            let budget = metal2vulkan::Budget::standard();
            match &self.descriptor {
                None => Err("a state of no guest descriptor's".to_owned()),
                Some(Reflected::Compute(d)) => plain(|| d.slot_writes(&budget))
                    .map(|w| vec![Slots::of(w.kernel_slots())])
                    .map_err(|e| e.to_string()),
                Some(Reflected::Render(d)) => {
                    let (writes, rasterizes) = match d {
                        RenderDescriptor::Render(d) => {
                            (plain(|| d.slot_writes(&budget)), d.rasterization_enabled)
                        }
                        RenderDescriptor::Tile(d) => (plain(|| d.slot_writes(&budget)), true),
                        RenderDescriptor::Mesh(m) => (
                            plain(|| m.descriptor.slot_writes(&budget)),
                            m.descriptor.rasterization_enabled,
                        ),
                    };
                    writes
                        .map(|w| {
                            let fragment = match rasterizes {
                                true => Slots::of(w.fragment_slots()),
                                false => Slots::default(),
                            };
                            vec![
                                Slots::of(w.vertex_slots()),
                                fragment,
                                Slots::of(w.tile_slots()),
                                Slots::of(w.object_slots()),
                                Slots::of(w.mesh_slots()),
                            ]
                        })
                        .map_err(|e| e.to_string())
                }
            }
        });
        read.as_deref().map_err(String::as_str)
    }

    /// The render stages whose functions may store through what a draw binds at their slots, in
    /// the order a draw's writes are asked: a mesh render pipeline state's object, mesh and
    /// fragment stages ([`OBJECT`], [`MESH`], [`FRAGMENT`]), which take none of the vertex and
    /// tile stages' binds; any other state's vertex, fragment and tile stages, as a state of no
    /// guest descriptor's.
    pub fn stages(&self) -> &'static [usize] {
        match &self.descriptor {
            Some(Reflected::Render(RenderDescriptor::Mesh(_))) => MESH_STAGES,
            _ => DRAWING_STAGES,
        }
    }

    /// Whether the function of `s` may store through its slot; `Err` naming why the runtime
    /// answers none for the state ([`Reflection::slots`]). A stage the state has no function of
    /// stores through none.
    fn writability(&self, s: &Slot) -> Result<SlotWritability, &str> {
        let slots = self.slots()?;
        Ok(match slots.get(s.stage) {
            Some(f) if s.texture => f.texture(s.slot),
            Some(f) => f.buffer(s.slot),
            None => SlotWritability::Never,
        })
    }
}

/// Whether work writes an object it binds or holds resident, where it may.
#[derive(Clone, Debug)]
pub enum Whether {
    /// It may, whatever its functions: a resource made resident with write usage, and a binding
    /// whose functions are not known.
    Writes,
    /// Where a function of a pipeline state may store through the slot the object is bound at,
    /// by the state's slot writes ([`Reflection::slots`]).
    Slot(Arc<Reflection>, Slot),
}

/// A slot of a function of a pipeline state, and whether the function writes it where the
/// state's slot writes cannot tell ([`Whether::Slot`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    /// The function's index in [`Reflection::slots`].
    pub stage: usize,
    /// A texture slot, or a buffer slot.
    pub texture: bool,
    /// The slot.
    pub slot: u64,
    /// Whether the function writes it where the slot writes cannot tell: the runtime answers
    /// none for the state, or leaves the slot undetermined.
    pub untold: bool,
}

/// What `gap` leaves undetermined, for a death that names it.
fn gap_named(gap: SlotWriteGap) -> String {
    let what = match gap {
        SlotWriteGap::UnresolvedIndex => {
            "an argument whose slot only a function constant states, which the state's values do \
             not fold"
        }
        SlotWriteGap::UnmeasuredPayloadStage => {
            "a payload argument of a stage Metal's access for one is unmeasured in"
        }
        SlotWriteGap::MemberRole => "an argument of a role only an argument buffer's member takes",
        _ => "an argument the runtime leaves undetermined",
    };
    format!("{what}: {gap:?}")
}

impl Whether {
    /// What work writes an object through, where it does ([`Whether::holds`]), for a death that
    /// names it, with why the state's slot writes cannot tell where they cannot. A buffer's slot
    /// is written where its function's argument there is a `device` pointer that is not `const`,
    /// which Metal's reflection answers read-write whether or not the function stores through it.
    pub(crate) fn through(&self) -> Cow<'static, str> {
        match self {
            Whether::Writes => "a resource made resident for writing, or a buffer bound for \
                                indirect commands that run pipeline states of their own, which \
                                no reflection here tells"
                .into(),
            Whether::Slot(reflection, s) => match (reflection.writability(s), s.texture) {
                (Err(why), _) => format!(
                    "a texture or buffer at a slot the function may write, which the pipeline \
                     state's slot writes cannot tell ({why})"
                )
                .into(),
                (Ok(SlotWritability::Unknown(gap)), _) => format!(
                    "a texture or buffer at a slot the function may write, which the pipeline \
                     state's slot writes leave undetermined ({})",
                    gap_named(gap)
                )
                .into(),
                (Ok(_), true) => "a texture at a slot the function writes (`access::write` or \
                                  `access::read_write`)"
                    .into(),
                (Ok(_), false) => "a buffer at a slot whose argument the function declares a \
                                   `device` pointer that is not `const`, read-write by Metal's \
                                   reflection, whether or not it stores through it"
                    .into(),
            },
        }
    }

    /// Whether the work writes the object: only a slot the function cannot store through
    /// ([`SlotWritability::Never`]) holds no write, and one the slot writes leave undetermined
    /// holds one where [`Slot::untold`].
    pub(crate) fn holds(&self) -> bool {
        match self {
            Whether::Writes => true,
            Whether::Slot(reflection, s) => match reflection.writability(s) {
                Ok(SlotWritability::Never) => false,
                Ok(SlotWritability::Unknown(_)) | Err(_) => s.untold,
                Ok(_) => true,
            },
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use reims_vgpu_core::platform::Platform as _;
    use reims_vgpu_core::platform::device::GpuDevice as _;
    use reims_vgpu_core::platform::resource::{Buffer as _, Heap as _};
    use reims_vgpu_core::platform::values::TextureDesc;
    use reims_vgpu_core::raise::OnFault;

    use super::*;
    use crate::gpu::resource::tests::{guest_pages, heap_at, over_pages};

    /// A death over a slot whose state's slot writes cannot tell names why they cannot; one they
    /// tell names the declaration that writes
    #[test]
    fn a_write_the_slot_writes_cannot_tell_names_why() {
        let slot = Slot {
            stage: 0,
            texture: false,
            slot: 3,
            untold: true,
        };
        let untold = Whether::Slot(Reflection::new(None), slot);
        assert!(untold.holds());
        assert!(
            untold
                .through()
                .ends_with("slot writes cannot tell (a state of no guest descriptor's)"),
            "{}",
            untold.through()
        );
        assert!(
            Whether::Writes
                .through()
                .starts_with("a resource made resident")
        );
    }

    /// The tests' functions of `ycbcr_probe.metal`, `read_only_beside_nested_vertex` and
    /// `read_only_beside_nested_fragment` among them.
    pub(crate) const PROBE: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/ycbcr_probe.metallib"));

    /// A pipeline of `read_only_beside_nested_vertex` and `read_only_beside_nested_fragment`
    /// ([`PROBE`]), each of which takes an argument buffer of a type Metal's reflection is
    /// unmeasured for (an array of structs that hold a struct and no constant), which the
    /// runtime refuses the whole reflection of, still answers each slot by its slot writes: the
    /// `const` buffers at the vertex and fragment functions' slots 0 and 1, a buffer slot no
    /// argument binds and a texture slot are unwritten, the vertex function's `device` buffer at
    /// slot 2 written, named as such; every slot of a state the runtime answers nothing for is
    /// written where it is untold
    #[test]
    fn slots_beside_an_argument_the_reflection_refuses_are_each_answered() {
        use metal2vulkan::error::{Error, Feature};
        use metal2vulkan::library::Library;
        use metal2vulkan::pipeline::{FunctionDescriptor, RenderPipelineDescriptor, StageSource};

        let library = Library::from_metallib(PROBE).unwrap();
        let stage =
            |name| StageSource::Function(FunctionDescriptor::new(library.function(name).unwrap()));
        let mut desc = RenderPipelineDescriptor::new(stage("read_only_beside_nested_vertex"));
        desc.fragment = Some(stage("read_only_beside_nested_fragment"));
        match desc.reflection(&metal2vulkan::Budget::standard()) {
            Err(Error::Unsupported(u)) => assert_eq!(u.feature, Feature::Reflection),
            other => panic!("{other:?}"),
        }
        let reflection = Reflection::new(Some(Reflected::Render(RenderDescriptor::Render(
            Box::new(desc),
        ))));
        let at = |stage, texture, slot| {
            let slot = Slot {
                stage,
                texture,
                slot,
                untold: true,
            };
            Whether::Slot(Arc::clone(&reflection), slot)
        };
        let unwritten = [
            (VERTEX, false, 0),
            (VERTEX, false, 1),
            (VERTEX, false, 3),
            (VERTEX, true, 1),
            (FRAGMENT, false, 0),
            (FRAGMENT, false, 1),
            (FRAGMENT, false, 2),
            (TILE, false, 1),
        ];
        for (stage, texture, slot) in unwritten {
            assert!(
                !at(stage, texture, slot).holds(),
                "{stage} {texture} {slot}"
            );
        }
        let written = at(VERTEX, false, 2);
        assert!(written.holds());
        assert!(
            written
                .through()
                .starts_with("a buffer at a slot whose argument the function declares"),
            "{}",
            written.through()
        );
        let none = Slot {
            stage: VERTEX,
            texture: false,
            slot: 1,
            untold: true,
        };
        assert!(Whether::Slot(Reflection::new(None), none).holds());
    }

    /// A buffer or heap over read-only memory the runtime copies is counted once while it is, set
    /// so again or not, and uncounted as it goes, which releases it; none counted, none is kept
    /// (in one process per test, as nextest runs them). Linux's: on macOS the runtime copies none
    #[cfg(target_os = "linux")]
    #[test]
    fn kept_backings_are_counted_while_they_are_kept() {
        use crate::gpu::backing::Backing;
        let count = || KEPT_NOW.load(Ordering::SeqCst);
        assert!(!any_kept());
        let (a, b) = (Backing::fixed(), Backing::fixed());
        a.imported(false, false);
        assert_eq!(count(), 0);
        a.imported(true, false);
        a.imported(true, false);
        assert_eq!(count(), 1);
        assert!(any_kept());
        b.imported(true, false);
        assert_eq!(count(), 2);
        a.imported(false, false);
        a.imported(false, false);
        assert_eq!(count(), 1);
        drop(b);
        assert!(!any_kept());
        a.imported(true, false);
        drop(a);
        assert!(!any_kept());
    }

    /// What a bind keeps of a buffer's or a texture's memory, read without collecting ranges, is
    /// what the memory collected answers: for a buffer of the device's memory, one placed in a
    /// heap over the guest's pages (at its bytes and past the pages), one on a surface, textures
    /// placed in the heap, over a buffer, on surfaces over the guest's pages and over memory of
    /// their own, and one of memory of its own; and once the heap is detached, in no memory
    #[test]
    fn bound_memory_is_the_memory_collected() {
        let (p, d) = (
            crate::platform::tests::platform(),
            crate::gpu::device::tests::dev(),
        );
        let bgra = |options: u64| {
            p.texture_descriptor(&TextureDesc {
                pixel_format: 80,
                width: 16,
                height: 16,
                usage: 0x1,
                resource_options: Some(options),
                ..TextureDesc::default()
            })
        };
        let size = over_pages();
        let heap = heap_at(&p, &d, guest_pages());
        let options = heap.resource_options();
        let placed = heap
            .new_buffer_of_length(0x1000, options, 0x2000)
            .unwrap()
            .unwrap();
        let in_heap = heap.new_texture(&bgra(options), 0x4000).unwrap().unwrap();
        let own = d
            .new_buffer_of_length(OnFault::Die, 0x4000, 0)
            .unwrap()
            .unwrap();
        let over_own = own
            .new_texture(OnFault::Die, &bgra(0), 0, 0x40)
            .unwrap()
            .unwrap();
        let alone = d.new_texture(OnFault::Die, &bgra(0)).unwrap().unwrap();
        let surface = crate::os::tests::bgra_surface(16, 16);
        let (over, pages, length) = crate::os::tests::bgra_surface_over_at(16, 16, 0x40);
        let on_surface = (d.new_texture_on_surface(&bgra(0), Some(&surface), 0))
            .unwrap()
            .unwrap();
        let on_pages = (d.new_texture_on_surface(&bgra(0), Some(&over), 0))
            .unwrap()
            .unwrap();
        let surface_buffer = d.new_buffer_on_surface(Some(&surface)).unwrap().unwrap();
        let agree = |when: &str| {
            for (name, b) in [
                ("placed", &placed),
                ("own", &own),
                ("surface", &surface_buffer),
            ] {
                assert_eq!(buffer_bound(b), buffer_memory(b).bound(), "{when}: {name}");
                for bytes in [0..0x10, 0xff0..0x1010, size..size + 0x10, 0x10..0x10] {
                    assert_eq!(
                        buffer_bytes_bound(b, bytes.clone()),
                        buffer_bytes_memory(b, bytes.clone()).bound(),
                        "{when}: {name} {bytes:x?}"
                    );
                }
            }
            for (name, t) in [
                ("in the heap", &in_heap),
                ("over a buffer", &over_own),
                ("of its own", &alone),
                ("on a surface", &on_surface),
                ("on the guest's pages", &on_pages),
            ] {
                assert_eq!(
                    texture_bound(t),
                    texture_memory(t).bound(),
                    "{when}: {name}"
                );
            }
        };
        agree("attached");
        assert!(buffer_bound(&placed).meets && texture_bound(&in_heap).meets);
        heap.detach_backing().unwrap();
        agree("detached");
        assert!(!buffer_bound(&placed).meets && !texture_bound(&in_heap).meets);
        drop((on_pages, over));
        crate::os::tests::free(pages, length);
    }

    /// A texture over a surface's plane reaches the plane's rows where the CPU lays them out and
    /// where the GPU reads them, to whichever ends later: from the offset rounded down to 16, a
    /// GPU's row `bytesPerRow` rounded down to 16, or 16 where that is 0
    #[test]
    fn a_plane_texture_reaches_its_rows_where_the_cpu_and_the_gpu_lay_them() {
        // Four rows of 0x48 bytes from 0x4008: the CPU's end is the later.
        assert_eq!(super::plane_bytes(0x4008, 0x48, 4), 0x4000..0x4128);
        // Four rows of 8 bytes from 0x4008, 16 apart on the GPU: its end is the later.
        assert_eq!(super::plane_bytes(0x4008, 8, 4), 0x4000..0x4040);
    }
}
