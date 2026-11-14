//! What GPU work may write: the host memory each object it binds or makes resident lies in
//! ([`Memory`]), and the slots its pipeline state's functions may store through, by the runtime's
//! slot writes of the state's descriptor ([`Reflection`], [`Slots`]), with whether a write
//! through a slot happens ([`Whether`]).
//!
//! A dispatch's or a draw's write through a texture over a surface of a 4:4:4 two-plane YCbCr
//! format whose writes the runtime refuses dies where it is encoded
//! ([`crate::gpu::ycbcr::refused_write`]; `crate::gpu::predication`, `crate::gpu::encode_render`):
//! only a slot the slot writes answer the function cannot store through
//! ([`SlotWritability::Never`]) takes such a texture without a death.
//!
//! Read-only memory ([`Memory::kept`]) on Linux is that of **the guest's read-only objects** (a
//! no-copy buffer or heap made with `MTLResourceOptions` bit 24 or over pages mapped read-only;
//! [`Kept::Copied`]; [`crate::os::COPIES_READ_ONLY`]), which the runtime makes over a copy of the
//! pages whose GPU writes it drops: every GPU write there is sent and completes, its write dropped
//! (a shader's store through a slot or a loaded address, a draw's or a kernel's write of a
//! texture, a blit's fill, copy or mipmaps, a pass's visibility results, an attachment the pass
//! does not store, a resolve the pass does not make), but a render pass's store-out, which fails
//! its command buffer with Metal's page fault, as the runtime faults it itself, the fault counting
//! against its queue as Metal's does: the queue's second revokes it (`crate::gpu::encode_render`,
//! `crate::gpu::command`). A store-out is an attachment stored (`Store`,
//! `StoreAndMultisampleResolve`, and a depth attachment's `CustomSampleDepthStore`), or a resolve
//! into such a texture (a colour or stencil attachment's `MultisampleResolve` or
//! `StoreAndMultisampleResolve`, a depth attachment's `MultisampleResolve`, into the descriptor's
//! resolve texture or a late setter's). A depth attachment's `StoreAndMultisampleResolve` into such
//! a texture completes, its resolve dropped and the texture unchanged, as Metal's GPU completes it,
//! but beside a stencil attachment that resolves, a pairing nothing measured, which dies where it
//! is encoded ([`kept_write`]). A CPU write of a texture over such memory (`replaceRegion:`)
//! writes the pages where the process can write them, and dies over pages it maps read-only, as
//! Metal's ends the process there (`crate::gpu::resource`).
//!
//! On macOS it is not taken: MoltenVK hands Metal the pages, or a read-only view of them made
//! with bit 24, and Metal's GPU answers every write there itself.
//!
//! What Metal does there, measured by metal2vulkan's probes (`read_only_view.rs`,
//! `probes/readonly_gpu_writes.m`) on the M3 Max (macOS 27.0.1), 2026-10-25 and since, and by
//! this platform's probe of every write it answered by ending the process (2026-11-14, plain
//! Metal, one shape a process, against a writable control) on the M3 Max and on the dGPU oracle's
//! Radeon Pro 5300M and UHD Graphics 630 (macOS 26.7.1, the x86 guests' line; m3-max-29292 to
//! m3-max-29294 and m3-max-29309 to m3-max-29311; `docs/graphics-kext/dgpu-oracle.md`):
//! - **On the M3 Max** bit 24, and pages the process maps read-only alike, keep the GPU's writes
//!   out. It completes a draw or a dispatch whose shader stores may write through such memory
//!   (through a bound slot or a loaded address, `drawIndexedPrimitives` among them), a kernel's
//!   `write()` of a texture there, a blit's fill, copy, colour fill or mipmaps into it and a pass
//!   that keeps its visibility results in it, and drops those writes, over 8,314 runs and every
//!   run since. A pass's store-out into it fails the command buffer
//!   (`MTLCommandBufferErrorPageFault`) every run, its signal firing, the next command buffer of
//!   its queue and of a new queue running, and every command buffer of the queue after its second
//!   fault failing (`MTLCommandBufferErrorAccessRevoked`; m3-max-29110, m3-max-28796,
//!   m3-max-28797, m3-max-29292, m3-max-29309; the runtime's `command_buffer/revocation.rs`), and
//!   so does a fragment or vertex function's `write()` of a texture there. A depth attachment's
//!   `StoreAndMultisampleResolve` into it completes (4 runs of 6; the other 2 failing as innocent
//!   victims of a recovery, code 1), the resolve texture unchanged. The same attachments under
//!   `DontCare` or `Load` complete, as does a resolve texture attached and not resolved into, a
//!   resolve out of a read-only primary and a resolve of a format pair Metal leaves untouched.
//! - **On the Radeon Pro 5300M and the UHD Graphics 630** bit 24 changes nothing: every write
//!   above lands in the pages and every command buffer completes, as without the bit. Over pages
//!   the process maps read-only, every GPU write measured (a pass's store and resolves, a draw's
//!   and a kernel's texture writes, a blit's copy) completes, the pages unchanged.
//! - **On all three** a CPU `replaceRegion:` of a texture over a bit-24 buffer writes the pages,
//!   and of one over pages the process maps read-only ends the process with a bus error, with the
//!   bit or without; and a colour or stencil attachment that the pass descriptor keeps by
//!   `CustomSampleDepthStore` ends the process in the descriptor's validation, with the bit or
//!   without.
//!
//! Where these GPUs differ, the envelope holds each answer, and this platform takes the cheaper:
//! the runtime's fault at a store-out (the M3 Max's), and completion with the writes dropped
//! elsewhere.
use std::borrow::Cow;
use std::ops::Range;
use std::sync::Arc;

use crate::core::num::ToU64;
use metal2vulkan::metal::{BindingType, PixelFormat};
use metal2vulkan::pipeline::ComputePipelineDescriptor;
use metal2vulkan::reflection::{SlotWritability, SlotWrite, SlotWriteGap};

use crate::answer::plain;
use crate::gpu::resource::{BufferObject, HeapObject, RenderDescriptor, TextureObject, Under};
use crate::os::OsSurface;

/// The memory an object's bytes lie in, which a write of the object may change.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Memory {
    /// The host memory: the first byte and end of each range. An object of none (a `Private`
    /// buffer, a texture of memory of its own) is written nowhere a guest's surface lies.
    pub ranges: Vec<Range<u64>>,
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
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum Kept {
    /// It is not.
    #[default]
    No,
    /// For the object's life: for a texture over a buffer or heap, read-only memory the runtime
    /// makes the buffer or heap over a copy of, dropping the GPU's writes
    /// ([`crate::gpu::backing::Backing::read_only_now`]): a render pass's store or resolve into
    /// it may fault the command buffer as Metal's GPU does (`crate::gpu::encode_render`), and
    /// every other GPU write of it is sent and dropped. A buffer's own memory the runtime copies
    /// is never `Copied`: the runtime drops its writes, as Metal's GPU does.
    Copied,
}

impl Kept {
    /// Whether the object's memory is read-only so now.
    pub fn holds(&self) -> bool {
        matches!(self, Kept::Copied)
    }
}

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
            refused: None,
            kept: Kept::No,
        }
    }

    /// The memory of the bytes `bytes` (a first byte and an end, from its base address) of
    /// `surface` ([`crate::os::surface_bytes_memory`]).
    pub fn of_surface_bytes(surface: &OsSurface, bytes: Range<u64>) -> Memory {
        Memory {
            ranges: crate::os::surface_bytes_memory(surface, std::slice::from_ref(&bytes)),
            refused: None,
            kept: Kept::No,
        }
    }

    /// The memory of an object made over the host memory `ranges`, read-only as `kept` says.
    pub fn over(ranges: Vec<Range<u64>>, kept: Kept) -> Memory {
        Memory {
            ranges,
            refused: None,
            kept,
        }
    }

    /// What a bind keeps of it ([`Bound`]).
    pub fn bound(self) -> Bound {
        Bound {
            meets: !self.is_empty(),
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

/// Whether the writes through `t` are not answered as Metal's GPU answers them, its memory being
/// read-only ([`Memory::kept`]), as [`texture_memory`] answers, without the ranges: that of the
/// buffer `t` lies over, or of its heap, or of `t`'s heap, which the runtime copies.
pub fn texture_kept(t: &TextureObject) -> bool {
    match (&t.surface, &t.under) {
        (None, Some(Under::Buffer(b, _))) => under_buffer_kept(b).holds(),
        (None, Some(Under::Heap(h, _))) => h.backing.read_only_now(),
        _ => false,
    }
}

/// Whether `t` lies over read-only memory the runtime copies whose pages are mapped read-only
/// ([`crate::gpu::backing::Backing::pages_read_only`]): the memory of the buffer `t` lies over,
/// or of that buffer's heap, or of `t`'s heap.
pub fn texture_pages_read_only(t: &TextureObject) -> bool {
    match (&t.surface, &t.under) {
        (None, Some(Under::Buffer(b, _))) if b.surface.is_none() => match &b.heap {
            Some(h) => h.backing.pages_read_only(),
            None => b.backing.pages_read_only(),
        },
        (None, Some(Under::Heap(h, _))) => h.backing.pages_read_only(),
        _ => false,
    }
}

/// What a texture over `b` keeps of `b`'s memory ([`Memory::kept`]): the read-only memory the
/// runtime makes `b`, or the heap it lies in, over a copy of.
fn under_buffer_kept(b: &BufferObject) -> Kept {
    match (&b.surface, &b.heap) {
        (Some(_), _) => Kept::No,
        (None, Some(h)) => h.backing.kept(),
        (None, None) => b.backing.kept(),
    }
}

/// Dies at `sel`, a command that may write `what` an object over read-only memory whose write
/// this platform does not answer as Metal's GPU does ([`Memory::kept`]): a depth attachment's
/// `StoreAndMultisampleResolve` into a texture over read-only guest pages (a read-only heap, a
/// resource made with `MTLResourceOptions` bit 24) the runtime copies, beside a stencil
/// attachment that resolves, which the runtime faults where nothing measured Metal's GPU, and a
/// late resolve texture there of an attachment the pass lacks (`crate::gpu::encode_render`'s
/// `stencil_may_resolve` and `late_resolve`). Ending the process is this platform's divergence
/// from Metal (`docs/accuracy-ledger.md`, `read-only-pages-written`).
pub(crate) fn kept_write(sel: &'static str, what: &str) -> ! {
    crate::answer::unrepresentable(
        sel,
        &format!(
            "{what} read-only memory (a read-only heap or resource the runtime copies), where \
             nothing measured what Metal's GPU does"
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
        (None, Some(Under::Heap(h, _))) => heap_memory(h),
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
/// memory (`newBufferWithIOSurface:`, from the surface's base address), or in the memory it was
/// made over; none for a `Private` one, or for one in a heap, as for the heap ([`heap_memory`]).
pub fn buffer_bytes_memory(b: &BufferObject, bytes: Range<u64>) -> Memory {
    if let Some(s) = &b.surface {
        return Memory::of_surface_bytes(s, bytes);
    }
    if b.heap.is_some() {
        // None, as for its heap, and nothing kept.
        return Memory::default();
    }
    // Its writes into read-only memory the runtime copies are dropped, as Metal's GPU drops them:
    // nothing kept.
    let contents = plain(|| b.raw.contents()).map(|first| {
        let first = first.as_ptr().addr().to_u64();
        std::iter::once(first.saturating_add(bytes.start)..first.saturating_add(bytes.end))
            .collect()
    });
    Memory::over(contents.unwrap_or_default(), Kept::No)
}

/// [`buffer_bound`] where it may decide what work through `b` asks: `b` lies on a surface;
/// elsewhere, where its [`Bound`] holds no surface, no refused format and nothing kept, which a
/// slot's bind keeps nothing of, the default one, of no memory, read without asking the runtime,
/// as each draw binds its buffers anew.
pub fn buffer_bind(b: &BufferObject) -> Bound {
    if buffer_bind_reads(b) {
        buffer_bound(b)
    } else {
        Bound::default()
    }
}

/// Whether [`buffer_bind`] reads `b`'s [`buffer_bound`]: `b` lies on a surface. Else it answers
/// the default one: a buffer over read-only memory the runtime copies keeps nothing, as its GPU
/// writes are dropped, as Metal's are.
pub fn buffer_bind_reads(b: &BufferObject) -> bool {
    b.surface.is_some()
}

/// [`texture_bound`] where it may decide what work through `t` asks, as [`buffer_bind`] reads a
/// buffer's: `t` lies on a surface, or over a buffer on one; elsewhere the default one, of no
/// memory. A texture over read-only memory the runtime copies decides nothing a draw asks: its
/// writes are sent, and dropped by the copy.
pub fn texture_bind(t: &TextureObject) -> Bound {
    let kept = match (&t.surface, &t.under) {
        (Some(_), _) => true,
        (None, Some(Under::Buffer(b, _))) => b.surface.is_some(),
        (None, Some(Under::Heap(..)) | None) => false,
    };
    if kept {
        texture_bound(t)
    } else {
        Bound::default()
    }
}

/// What a bind keeps of `b`'s memory: [`buffer_memory`]'s [`Bound`], read as it reads it, but
/// without collecting the ranges of its bytes in the guest's pages, one a page run, as each draw
/// binds its buffers anew; the same of any of its bytes ([`buffer_bytes_memory`]).
pub fn buffer_bound(b: &BufferObject) -> Bound {
    if b.surface.is_some() {
        return surface_bound(None);
    }
    if b.heap.is_some() {
        // As `buffer_bytes_memory`'s: none, and nothing kept.
        return Bound::default();
    }
    // Nothing kept, as `buffer_bytes_memory`'s.
    Bound::over(plain(|| b.raw.contents()).is_some(), Kept::No)
}

/// [`heap_memory`]'s [`Bound`] of `h`.
fn heap_bound(h: &HeapObject) -> Bound {
    Bound::over(false, h.backing.kept())
}

/// The [`Bound`] of an object made on a surface, of the `refused` format where the runtime
/// refuses its writes: [`Memory::of_surface`]'s and [`Memory::of_surface_bytes`]'s, which always
/// hold the surface's own allocation ([`crate::os::surface_memory`]), so it meets memory.
fn surface_bound(refused: Option<PixelFormat>) -> Bound {
    Bound {
        meets: true,
        refused,
        kept: Kept::No,
    }
}

/// What a bind or a residency keeps of `t`'s memory: [`texture_memory`]'s [`Bound`], read as it
/// reads it, without collecting the ranges of its texels in the guest's pages, one a page run,
/// as each draw binds its textures anew.
pub fn texture_bound(t: &TextureObject) -> Bound {
    match (&t.surface, &t.under) {
        (Some(_), _) => {
            let refused = t.plane.as_ref().and_then(|_| {
                let pf = plain(|| t.raw.pixel_format());
                crate::gpu::ycbcr::writes_refused(pf).then_some(pf)
            });
            surface_bound(refused)
        }
        (None, Some(Under::Buffer(b, _))) => Bound {
            kept: under_buffer_kept(b),
            ..buffer_bound(b)
        },
        (None, Some(Under::Heap(h, _))) => heap_bound(h),
        _ => Bound::default(),
    }
}

/// The host memory of `h`'s bytes: none, as this platform records no memory of a heap's, its own
/// or the host memory it was made over. What it keeps is a texture's in it ([`Kept::Copied`]); a
/// buffer in it keeps nothing ([`buffer_bytes_memory`]).
pub(crate) fn heap_memory(h: &HeapObject) -> Memory {
    Memory::over(Vec::new(), h.backing.kept())
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
    /// function.
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
    use crate::core::platform::Platform as _;
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::resource::{Buffer as _, Heap as _};
    use crate::core::platform::values::TextureDesc;
    use crate::core::raise::OnFault;

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

    /// A death over a slot the state's slot writes leave undetermined names what leaves it so,
    /// and the runtime's gap: a slot only a function constant states, a payload argument of an
    /// unmeasured stage, an argument buffer member's role. No committed function leaves a slot
    /// undetermined, so this reads the naming alone, not a death through a reflection
    #[test]
    fn an_undetermined_slot_names_its_gap() {
        for (gap, what) in [
            (
                SlotWriteGap::UnresolvedIndex,
                "an argument whose slot only a function constant states",
            ),
            (
                SlotWriteGap::UnmeasuredPayloadStage,
                "a payload argument of a stage",
            ),
            (
                SlotWriteGap::MemberRole,
                "an argument of a role only an argument buffer's member takes",
            ),
        ] {
            let named = gap_named(gap);
            assert!(named.starts_with(what), "{named}");
            assert!(named.ends_with(&format!(": {gap:?}")), "{named}");
        }
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

    /// A buffer or heap over read-only memory the runtime copies is read-only while its memory
    /// is, set so again or not, and no longer once it is not. Of such memory, only pages mapped
    /// read-only refuse a CPU write, not memory of bit 24 alone. Linux's: on macOS the runtime
    /// copies none
    #[cfg(target_os = "linux")]
    #[test]
    fn backings_over_copied_memory_are_read_only_while_it_is() {
        use crate::gpu::backing::Backing;
        let (a, b) = (Backing::new(false), Backing::new(false));
        a.imported(false, false);
        assert!(!a.read_only_now());
        a.imported(true, false);
        a.imported(true, false);
        assert!(a.read_only_now() && a.pages_read_only());
        b.imported(false, true);
        assert!(b.read_only_now() && !b.pages_read_only());
        a.imported(false, false);
        a.imported(false, false);
        assert!(!a.read_only_now() && !a.pages_read_only());
        let bit_24 = Backing::new(true);
        bit_24.imported(false, false);
        assert!(bit_24.read_only_now() && !bit_24.pages_read_only());
    }

    /// What a bind keeps of a buffer's or a texture's memory, read without collecting ranges, is
    /// what the memory collected answers: for a buffer of the device's memory, one placed in a
    /// heap over the guest's memory (at its bytes and past the heap's), one on a surface, textures
    /// placed in the heap, over a buffer, on surfaces over the guest's pages and over memory of
    /// their own, and one of memory of its own. A heap's buffer and texture lie in no memory a
    /// surface's write check looks for
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
        let heap = heap_at::<crate::platform::VulkanPlatform>(&d, guest_pages());
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
                        buffer_bound(b),
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
        agree("made");
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
