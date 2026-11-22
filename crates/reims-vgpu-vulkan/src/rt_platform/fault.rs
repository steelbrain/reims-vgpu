//! A lost device's fault, named and placed. rt reads the driver's report of the fault
//! (`VK_EXT_device_fault`) once, at its device's first loss of its own, and keeps it with the loss
//! (rt's `DeviceLoss::fault`), with rt's own placement of each address among rt's ranges
//! (`DeviceFault::places`). After the loss's line
//! (`super::command`'s docs, "A lost device's line"), the device writes the fault's lines on
//! stderr ([`FaultWatch::lines`]), each starting `reims-vgpu: the rt platform's device on <GPU>
//! faulted: `:
//!
//! - **The report**: the driver's description, its vendor records (each one's text, code and
//!   data), and how many accesses and instruction pointers it names. Where rt read none (the
//!   device has no `VK_EXT_device_fault`, the read failed, or a window's swapchain met the loss
//!   first and reported it to rt, with no report), the one line says so.
//! - **Each access** the report names: its kind, address and precision as rt prints them, the
//!   bytes it names ([`fault_range`]: the address rounded down to its precision, a power of two,
//!   and that many bytes on), and where it lies ([`place`], [`Placement::text`]): inside which
//!   live objects and in which mapped blocks, each with the address's offset in it, and which of
//!   them the bytes reach without the address. Where no live object holds the address, the line
//!   names the nearest live object below it, with the bytes from that object's end to the
//!   address, and the nearest above it, with the bytes from the address to that object's first
//!   byte, or none ([`Nearest`]); and so of the mapped blocks, where none holds it. Of two at the
//!   same distance, the shorter is named, a block of an allocation before the allocation. An
//!   object is named by its kind, guest object id, context, storage mode, length and GPU address;
//!   a block by what it is (rt's pool allocation, a known block, a heap's block or a buffer's
//!   import, by rt's numbers), its storage mode, the kind of import it is, its length and its GPU
//!   address. Then where the address lies in the history of released ranges ([`history`],
//!   [`History::text`]): the latest released range the history keeps that held it, with the
//!   address's offset in it, or, where none did, the nearest below and above, as of the live
//!   objects; each named as at the loss, with when it was released, in seconds since 1970 on the
//!   wall clock, and how long before the loss that was, or after it, for one released as the
//!   line was made.
//! - **Each instruction pointer**, placed as an access is, with the note that none of the ranges
//!   the host sees holds shader code: a pipeline's code lies in the driver's own memory, of which
//!   the host sees nothing, so the placement says only what lies around the address.
//! - **rt's placement of each address** (`DeviceFault::places`), one line each, as rt's
//!   `FaultPlace` displays it: the ranges of rt's own that meet the bytes (its pool's
//!   allocations, its parameter ring's blocks, imports, exports, each by its block's number
//!   (`export block N`, rt's `BlockInfo::id`), surfaces' buffers and indirect command buffers'
//!   preprocess regions), live or among the last it freed, and the nearest below and above. It
//!   answers another question than the host's placement above, which places among the guest's
//!   objects and the blocks the host sees: both are written. Of each exported block the line
//!   names, the line then names the tables' live objects that lie in it ([`ExportedObject`]),
//!   which the host's placement cannot place, each by its offset in the block, or says that none
//!   does: an object no table stores, the host's own, is not among them.
//! - **What the placement saw**: the live objects of each kind, the decoders and contexts whose
//!   tables were walked and those not walked, the live textures with no GPU range the platform
//!   knows, the mapped blocks from each source, what the host does not see, and the released
//!   ranges the history keeps of each kind, with how many it keeps at most and how many older ones
//!   it dropped past that.
//!
//! What the host sees at the loss:
//!
//! 1. The live objects of the decoders' tables whose GPU range is known, walked at the loss
//!    ([`LiveObjects`], `reims_vgpu`'s `graphics::fault_walk`): buffers, heaps, textures over a
//!    buffer, and the `Shared` buffers over two-copy `Managed` textures' CPU copies, each removed
//!    entry kept for an open view among them. A deferred buffer rt has not made yet has no range,
//!    nor has a buffer in an exported block of its own (`super::export`), whose address the
//!    placement never asks of rt (`RtBuffer::fault_address`), nor a texture over such a buffer:
//!    the walk keeps those by their exported block's number and their offset in it
//!    (`RtBuffer::exported_block`), for rt's placement's lines (above).
//!    A texture rt made as an image of its own, in a heap or not, has no GPU range the platform
//!    knows: one in a heap lies in its heap's range; any other's memory is rt's alone.
//! 2. The blocks still mapped that the host sees ([`MappedBlock`]): rt's pool allocations
//!    ([`Allocations`]); the blocks the device knows (`super::buffer`'s `KnownBlocks`: the
//!    guest's pool and `Private` blocks, the imports of a pages block's views, the copies of host
//!    pages); the device's live dma-buf imports; and the blocks of the tables' live objects (a
//!    heap's block, the import a buffer lies in).
//!
//! What the host does not see, so an access there reads as in no mapped block: rt's parameter
//! ring, zero source and staging region, which rt allocates beside its pool; the images rt makes
//! with memory of their own; exports and surfaces (D3D11 planes among them, on Windows), by their
//! ranges, though the tables' objects in an exported block are kept by its number for rt's lines
//! (above); and an import by the pages' pointer that no live object or known block lies in (one rt
//! still holds for work in flight after its buffer went).
//!
//! **The history of released ranges** ([`Releases`]): the device's last [`RELEASES_KEPT`]
//! released ranges of the tables' objects, oldest first; a release past them drops the oldest,
//! counted. An object's range is recorded as its platform object goes (the guest released it and
//! the host let go of it): a buffer rt made, a heap, a two-copy `Managed` texture's CPU-copy
//! buffer, and a texture over a buffer while that buffer is there (one whose buffer went first is
//! not recorded: the buffer's range, recorded as it went, holds its bytes); not a buffer in an
//! exported block, nor a texture over one, which have no range (above). Each entry
//! ([`Release`]) holds the object's kind, its guest object id, whose bits from 40 are its context
//! ([`Named`]: told as a decoder's table stores the object, `reims_vgpu`'s
//! `graphics::fault_walk::name`), its storage mode, its range's first byte's GPU address and
//! length, and when it was released, in nanoseconds since 1970 on the wall clock. An object no
//! table stored, the host's own, is not recorded. rt frees an object's range once no work in
//! flight holds it, so a range released just before the loss may still have been work in
//! flight's. rt's own ranges, its pool's allocations and blocks, imports among them, are not in
//! it.
//!
//! A release costs a read of the wall clock and one push under the history's own lock, which is
//! held for that alone. The loss's time is the wall clock's as the loss's line is written, by the
//! first command buffer whose error names the loss.
//!
//! **The command buffers in flight** ([`Flights`]): a command buffer joins its device's list as
//! its commit hands it to rt, and leaves it as its first completed handler runs (`super::command`'s
//! commit), so the list holds those committed and not complete. Each is kept with the object id of
//! the guest's command buffer (`CommandBuffer::names_work`), whose bits from 40 are its context, or
//! none for the host's own; its commit's time on the wall clock; and its encoders' names
//! ([`EncoderNames`]), in their order. An encoder's names are its kind; a render encoder's pass's
//! attachments, each by its texture's guest id; and, by guest id, the heaps its `useHeap:` calls
//! declare, the buffers and textures its `useResource:` calls declare, and the buffers its
//! `set*Buffer:` calls bind, refused or not. Of each list an encoder keeps the first [`IDS_KEPT`]
//! ids, each once, and counts the names of others past them, and those of objects with no guest
//! id ([`Ids`]). An encoder's names join its command buffer's as it ends, or as it goes unended
//! ([`Naming`]). A command buffer cut in parts is one entry for each part rt has.
//!
//! The list's numbers are its device's commit order, across all its queues, which the platform's
//! own texture and sampler indices go back by (`super::texture`'s `OwnIndices`): the number of the
//! last command buffer committed ([`Flights::last`]), and the highest number up to which every
//! command buffer has left the list ([`Flights::done_through`]).
//!
//! At the loss, two lines follow the fault's ([`Flights::lines`]): the first [`FLIGHTS_LISTED`]
//! command buffers committed and not complete, the oldest committed first, each with its context,
//! its commit's time against the loss and its encoders' kinds in order (runs of one kind counted,
//! the first [`KINDS_LISTED`] runs listed), and a count of the rest; then the oldest's render
//! passes, the first [`PASSES_LISTED`], and the resources its encoders name, the first
//! [`IDS_KEPT`] of each list. The list costs, for each command buffer, a read of the wall clock
//! and an entry added and removed under the list's lock; for each encoder, one push under its
//! command buffer's lock; and for each buffer bound or object declared, under the encoder's lock,
//! which each command takes already, a load of its id and a look through the at most [`IDS_KEPT`]
//! ids its list keeps.
//!
//! **What is kept before the loss.** The tables are walked at the loss, with no record of them
//! kept before it beside the history of their objects' releases. The rest is kept as it changes,
//! one entry a range: rt's pool allocations' ranges, which [`Allocations`] keeps as rt's
//! `BlockMappings` tells it of them (it maps nothing and refuses nothing, so rt's allocations and
//! their mappings are as without it), and each import's range, beside the record of where it was
//! imported (`super::diag`'s `imported`).
//!
//! **Bounded waits.** The lines are written on the thread that reads the first failed command
//! buffer's error, rt's completion thread among them, which a lock's holder may be waiting for: a
//! decoder's walk waits [`BOUND`] in all for its locks, the map of its contexts and their tables,
//! and the device's known blocks [`BOUND`] for theirs. What was not had in time is named on the
//! line as not walked.
//!
//! Measured on an RTX 3090 (NVIDIA's driver, Linux, 2026-11-21): a loss's report had an empty
//! description and no vendor records, and named one access, `READ_INVALID` at a precision of
//! 0x1000, and one instruction pointer, `INSTRUCTION_POINTER_FAULT` at a precision of 0x10. The
//! access lay in no live object and no mapped block the host saw.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use metal2vulkan_runtime as rt;

use crate::core::num::rem0;
use crate::core::platform::os::Clock as _;
use crate::core::platform::resource::{GpuResource as _, Heap as _};
use crate::core::sync::Mutex;

use super::RtObj;
use super::buffer::{KnownBlocks, RtBuffer};
use super::diag::Point;
use super::heap::RtHeap;
use super::texture::RtTexture;

/// How long the walk at the loss waits for the locks it takes (the module's docs, "Bounded
/// waits"): a decoder's, in all, and the device's known blocks'.
pub const BOUND: Duration = Duration::from_millis(500);

/// What a live object of the tables is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectKind {
    /// `MTLBuffer`.
    Buffer,
    /// `MTLTexture`, over a buffer.
    Texture,
    /// `MTLHeap`.
    Heap,
    /// The `Shared` buffer over a two-copy `Managed` texture's CPU copy, under the texture's id.
    ManagedCopy,
}

/// A live object of a decoder's tables whose GPU range is known, as the walk at the loss found it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LiveObject {
    /// Its context.
    pub context: u32,
    /// The guest's object id.
    pub id: u64,
    /// What it is.
    pub kind: ObjectKind,
    /// Its `MTLStorageMode`.
    pub storage: u64,
    /// Its first byte's GPU address.
    pub gpu_address: u64,
    /// Its length in bytes.
    pub len: u64,
    /// Removed by the guest, and kept for an open view of the table that names it.
    pub gone: bool,
}

/// A live object of a decoder's tables that lies in an exported block, whose GPU range the
/// placement never asks of rt (`RtBuffer::fault_address`): a buffer in an exported block of its
/// own (`super::export`), or a texture over one, as the walk at the loss found it, by the block's
/// number, which rt's fault line names it by (`export block N`, rt's `BlockInfo::id`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportedObject {
    /// The exported block's number (rt's `BlockId::get`).
    pub block: u32,
    /// Its context.
    pub context: u32,
    /// The guest's object id.
    pub id: u64,
    /// What it is.
    pub kind: ObjectKind,
    /// Its `MTLStorageMode`.
    pub storage: u64,
    /// Its first byte's offset in the block.
    pub offset: u64,
    /// Its length in bytes.
    pub len: u64,
    /// Removed by the guest, and kept for an open view of the table that names it.
    pub gone: bool,
}

/// A block the host sees still mapped at the loss.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MappedBlock {
    /// What it is, as the line names it: "rt's pool allocation 12", "known block 345 of
    /// allocation 12", "heap 0x10000000a's block 77 of allocation 12 (the heap at offset 0x0)".
    pub what: String,
    /// Its storage mode.
    pub class: rt::MemoryClass,
    /// The kind of import it is, `None` for none.
    pub import: Option<&'static str>,
    /// Its first byte's GPU address.
    pub gpu_address: u64,
    /// Its length in bytes.
    pub len: u64,
}

/// A range of GPU addresses the placement knows: a live object's or a mapped block's.
trait Ranged {
    /// Its first byte's GPU address and its length in bytes.
    fn span(&self) -> (u64, u64);
}

impl Ranged for LiveObject {
    fn span(&self) -> (u64, u64) {
        (self.gpu_address, self.len)
    }
}

impl Ranged for MappedBlock {
    fn span(&self) -> (u64, u64) {
        (self.gpu_address, self.len)
    }
}

/// The bytes `[start, end)` an access names: its address rounded down to its precision, a power
/// of two, and that many bytes on (rt's `FaultAddress`); a precision of 0 is read as 1.
pub fn fault_range(address: u64, precision: u64) -> (u64, u64) {
    let precision = precision.max(1);
    let start = address - rem0(address, precision);
    (start, start.saturating_add(precision))
}

/// The nearest ranges of a kind below and above an address none of them holds ([`place`]).
#[derive(Debug, PartialEq, Eq)]
pub struct Nearest<'a, T> {
    /// The range that ends nearest below the address, with the bytes from its end (the byte after
    /// its last) to the address: 0 where the address is the byte after its last.
    pub below: Option<(&'a T, u64)>,
    /// The range that starts nearest above the address, with the bytes from the address to its
    /// first byte.
    pub above: Option<(&'a T, u64)>,
}

/// The nearest of `ranges` below and above `address`, which none of them holds: of two at the same
/// distance, the shorter, then the first.
fn nearest<T: Ranged>(ranges: &[T], address: u64) -> Nearest<'_, T> {
    let nearer = |kept: Option<(&T, u64)>, range: &T, gap: u64| match kept {
        Some((kept, at)) => gap < at || (gap == at && range.span().1 < kept.span().1),
        None => true,
    };
    let (mut below, mut above) = (None, None);
    for range in ranges {
        let (start, len) = range.span();
        let end = start.saturating_add(len);
        if end <= address {
            let gap = address - end;
            if nearer(below, range, gap) {
                below = Some((range, gap));
            }
        } else if start > address {
            let gap = start - address;
            if nearer(above, range, gap) {
                above = Some((range, gap));
            }
        }
    }
    Nearest { below, above }
}

impl<T> Nearest<'_, T> {
    /// The two as the line says them, each named by `name`.
    fn text(&self, name: impl Fn(&T) -> String) -> String {
        let below = match self.below {
            Some((range, gap)) => {
                format!("{}, the address {gap:#x} bytes past its end", name(range))
            }
            None => "none".to_owned(),
        };
        let above = match self.above {
            Some((range, gap)) => {
                format!(
                    "{}, its first byte {gap:#x} bytes past the address",
                    name(range)
                )
            }
            None => "none".to_owned(),
        };
        format!("nearest below: {below}; nearest above: {above}")
    }
}

/// Where an access lies ([`place`]).
#[derive(Debug, PartialEq, Eq)]
pub struct Placement<'a> {
    /// The bytes it names ([`fault_range`]).
    pub range: (u64, u64),
    /// The live objects that hold its address, each with the address's offset in it.
    pub objects: Vec<(&'a LiveObject, u64)>,
    /// The live objects its bytes reach that do not hold its address.
    pub near_objects: Vec<&'a LiveObject>,
    /// The nearest live objects below and above its address, where none holds it.
    pub nearest_objects: Option<Nearest<'a, LiveObject>>,
    /// The mapped blocks that hold its address, each with the address's offset in it.
    pub blocks: Vec<(&'a MappedBlock, u64)>,
    /// The mapped blocks its bytes reach that do not hold its address.
    pub near_blocks: Vec<&'a MappedBlock>,
    /// The nearest mapped blocks below and above its address, where none holds it.
    pub nearest_blocks: Option<Nearest<'a, MappedBlock>>,
}

/// Whether `[start, start + len)` holds `address`: the offset in it where it does.
fn offset_in(start: u64, len: u64, address: u64) -> Option<u64> {
    let offset = address.checked_sub(start)?;
    (offset < len).then_some(offset)
}

/// Whether `[start, start + len)` and `[from, to)` share a byte.
fn overlaps(start: u64, len: u64, (from, to): (u64, u64)) -> bool {
    start < to && from < start.saturating_add(len)
}

/// Of `ranges`, those that hold `address`, each with the address's offset in it, and those the
/// bytes `range` reach that do not hold it.
fn holding<T: Ranged>(ranges: &[T], address: u64, range: (u64, u64)) -> (Vec<(&T, u64)>, Vec<&T>) {
    let (mut holds, mut reach) = (Vec::new(), Vec::new());
    for item in ranges {
        let (start, len) = item.span();
        match offset_in(start, len, address) {
            Some(offset) => holds.push((item, offset)),
            None if overlaps(start, len, range) => reach.push(item),
            None => {}
        }
    }
    (holds, reach)
}

/// Where the access at `address`, of precision `precision`, lies among `objects` and `blocks`.
pub fn place<'a>(
    address: u64,
    precision: u64,
    objects: &'a [LiveObject],
    blocks: &'a [MappedBlock],
) -> Placement<'a> {
    let range = fault_range(address, precision);
    let (held_objects, near_objects) = holding(objects, address, range);
    let (held_blocks, near_blocks) = holding(blocks, address, range);
    Placement {
        range,
        nearest_objects: held_objects.is_empty().then(|| nearest(objects, address)),
        objects: held_objects,
        near_objects,
        nearest_blocks: held_blocks.is_empty().then(|| nearest(blocks, address)),
        blocks: held_blocks,
        near_blocks,
    }
}

/// `MTLStorageMode`'s name.
fn storage_name(mode: u64) -> String {
    match mode {
        0 => "Shared".to_owned(),
        1 => "Managed".to_owned(),
        2 => "Private".to_owned(),
        3 => "Memoryless".to_owned(),
        mode => format!("storage mode {mode}"),
    }
}

/// An object's kind as the line names it.
fn kind_name(kind: ObjectKind) -> &'static str {
    match kind {
        ObjectKind::Buffer => "buffer",
        ObjectKind::Texture => "texture",
        ObjectKind::Heap => "heap",
        ObjectKind::ManagedCopy => "the CPU-copy buffer of Managed texture",
    }
}

/// The words of an object removed by the guest and kept for an open view, where `gone`.
fn gone_words(gone: bool) -> &'static str {
    if gone {
        ", removed and kept for an open view"
    } else {
        ""
    }
}

/// An object as the line names it.
fn object_text(object: &LiveObject) -> String {
    format!(
        "{} {:#x} of context {} ({}, length {:#x}, at {:#x}{})",
        kind_name(object.kind),
        object.id,
        object.context,
        storage_name(object.storage),
        object.len,
        object.gpu_address,
        gone_words(object.gone)
    )
}

/// An object in an exported block as the line names it, by its offset in the block.
fn exported_text(object: &ExportedObject) -> String {
    format!(
        "{} {:#x} of context {} ({}, length {:#x}, at offset {:#x} in it{})",
        kind_name(object.kind),
        object.id,
        object.context,
        storage_name(object.storage),
        object.len,
        object.offset,
        gone_words(object.gone)
    )
}

/// The numbers of the exported blocks among `ranges`, ranges of rt's placement of an address
/// (`FaultPlace`'s holders, then the nearest below and above), each once, in their order.
fn export_blocks<'a>(ranges: impl IntoIterator<Item = &'a rt::FaultRange>) -> Vec<u32> {
    let mut blocks = Vec::new();
    for range in ranges {
        if let rt::RangeKind::Export(block) = range.kind
            && !blocks.contains(&block.get())
        {
            blocks.push(block.get());
        }
    }
    blocks
}

/// What rt's placement's line adds of the exported blocks `blocks` it names: for each, the live
/// objects of the tables that lie in it, of `exported`, or that none does.
fn exports_text(blocks: &[u32], exported: &[ExportedObject]) -> String {
    let mut text = String::new();
    for &block in blocks {
        let held = (exported.iter())
            .filter(|object| object.block == block)
            .map(exported_text)
            .collect::<Vec<_>>();
        let held = if held.is_empty() {
            "no live object of the tables".to_owned()
        } else {
            held.join(", ")
        };
        text.push_str(&format!("; export block {block} holds {held}"));
    }
    text
}

/// A block as the line names it.
fn block_text(block: &MappedBlock) -> String {
    let class = match block.class {
        rt::MemoryClass::Private => "Private",
        rt::MemoryClass::Shared => "Shared",
    };
    let import = block.import.unwrap_or("not an import");
    format!(
        "{} ({class}, {import}, length {:#x}, at {:#x})",
        block.what, block.len, block.gpu_address
    )
}

/// The wall clock now, in nanoseconds since 1970: a release's time and a commit's, and the
/// loss's (the module's docs).
pub(super) fn wall_now() -> u64 {
    crate::os::OsClock::new().now_wall().as_nanos()
}

/// `nanos`, nanoseconds, as the line gives seconds: to the microsecond.
fn seconds(nanos: u64) -> String {
    format!(
        "{}.{:06}",
        nanos / 1_000_000_000,
        nanos % 1_000_000_000 / 1_000
    )
}

/// The time `at` against the loss's, `loss`, both on the wall clock, as the line says it.
fn against(at: u64, loss: u64) -> String {
    match loss.checked_sub(at) {
        Some(before) => format!("{} s before the loss", seconds(before)),
        None => format!("{} s after the loss", seconds(at - loss)),
    }
}

/// The bits an object id's context lies above (the wire's `RGFX_CONTEXT_OBJ_ID_SHIFT`): a
/// context's object ids are its number shifted so far, and up.
const CONTEXT_SHIFT: u32 = 40;

/// The context the guest's object id `id` lies in.
fn context_of(id: u64) -> u32 {
    u32::try_from(id >> CONTEXT_SHIFT).unwrap_or(u32::MAX)
}

/// The guest's object id of a buffer, texture or heap of the platform's, told as a decoder's table
/// stores the object (`reims_vgpu`'s `graphics::fault_walk::name`, [`name_buffer`] and the rest),
/// which the history names it by as it goes ([`Releases`]), and a command buffer's line where an
/// encoder named it ([`EncoderNames`]): 0, none, for one no table stored, the host's own. The last
/// id told is kept. The `Shared` buffer over a two-copy `Managed` texture's CPU copy is told the
/// texture's id.
#[derive(Debug, Default)]
pub struct Named {
    /// The guest's object id, 0 for none.
    id: AtomicU64,
    /// Whether it is the CPU-copy buffer of the `Managed` texture whose id it holds.
    copy: AtomicBool,
}

impl Named {
    /// The guest's object id, 0 for none.
    pub fn id(&self) -> u64 {
        self.id.load(Ordering::Relaxed)
    }

    /// The kind of object it is as the history records it: `object`, or a `Managed` texture's
    /// CPU-copy buffer where it was told it is one.
    pub fn kind(&self, object: ObjectKind) -> ObjectKind {
        match self.copy.load(Ordering::Relaxed) {
            true => ObjectKind::ManagedCopy,
            false => object,
        }
    }

    /// Tells it `id`: one store.
    fn name(&self, id: u64) {
        self.id.store(id, Ordering::Relaxed);
    }

    /// Tells it that it is the CPU-copy buffer of `Managed` texture `id`: two stores.
    fn name_copy(&self, id: u64) {
        self.copy.store(true, Ordering::Relaxed);
        self.id.store(id, Ordering::Relaxed);
    }
}

/// Tells `buffer` the guest's id `id`, under which a decoder's table stores it ([`Named`]).
pub fn name_buffer(buffer: &RtObj<RtBuffer>, id: u64) {
    buffer.named().name(id);
}

/// Tells `buffer` that it is the `Shared` buffer over the CPU copy of the two-copy `Managed`
/// texture `id`, which a decoder's table stores under that id ([`Named`]).
pub fn name_managed_copy(buffer: &RtObj<RtBuffer>, id: u64) {
    buffer.named().name_copy(id);
}

/// Tells `texture` the guest's id `id`, under which a decoder's table stores it ([`Named`]).
pub fn name_texture(texture: &RtObj<RtTexture>, id: u64) {
    texture.named().name(id);
}

/// Tells `heap` the guest's id `id`, under which a decoder's table stores it ([`Named`]).
pub fn name_heap(heap: &RtObj<RtHeap>, id: u64) {
    heap.named().name(id);
}

/// How many released ranges a device's history keeps ([`Releases`]): the last so many, each older
/// one dropped, counted.
pub const RELEASES_KEPT: usize = 4096;

/// A released range as the history keeps it ([`Releases`]): an object of the tables, as its
/// platform object went.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Release {
    /// What it was.
    pub kind: ObjectKind,
    /// The guest's object id, whose bits from 40 are its context.
    pub id: u64,
    /// Its `MTLStorageMode`.
    pub storage: u64,
    /// Its first byte's GPU address.
    pub gpu_address: u64,
    /// Its length in bytes.
    pub len: u64,
    /// When it was released: nanoseconds since 1970, on the wall clock.
    pub at: u64,
}

impl Release {
    /// The context it was of.
    pub fn context(&self) -> u32 {
        context_of(self.id)
    }
}

impl Ranged for Release {
    fn span(&self) -> (u64, u64) {
        (self.gpu_address, self.len)
    }
}

/// The released ranges kept, oldest first, and how many older ones were dropped past them.
#[derive(Default)]
struct Ring {
    /// The last [`RELEASES_KEPT`], oldest first.
    kept: VecDeque<Release>,
    /// How many older ones were dropped past them.
    dropped: u64,
}

/// A device's history of released GPU ranges (the module's docs, "The history of released
/// ranges"): the last [`RELEASES_KEPT`], and how many older ones were dropped past them. A handle:
/// its clones are the same history, which the device's fault watch holds
/// ([`FaultWatch::releases`]).
#[derive(Clone, Default)]
pub struct Releases(Arc<Mutex<Ring>>);

impl Releases {
    /// Records the release of the object `id` of `kind` and storage mode `storage`, its range
    /// the `len` bytes at GPU address `gpu_address`, now: a read of the wall clock and one push
    /// under the history's lock, held for that alone; past [`RELEASES_KEPT`], the oldest is
    /// dropped, counted.
    pub fn push(&self, kind: ObjectKind, id: u64, storage: u64, (gpu_address, len): (u64, u64)) {
        self.push_at(Release {
            kind,
            id,
            storage,
            gpu_address,
            len,
            at: wall_now(),
        });
    }

    /// Records `release`: the history's room for [`RELEASES_KEPT`] is made at its first.
    fn push_at(&self, release: Release) {
        let mut ring = self.0.lock();
        if ring.kept.capacity() < RELEASES_KEPT {
            ring.kept.reserve_exact(RELEASES_KEPT);
        }
        if ring.kept.len() >= RELEASES_KEPT {
            ring.kept.pop_front();
            ring.dropped += 1;
        }
        ring.kept.push_back(release);
    }

    /// The released ranges kept, the newest first, and how many older ones were dropped.
    pub fn kept(&self) -> (Vec<Release>, u64) {
        let ring = self.0.lock();
        (ring.kept.iter().rev().copied().collect(), ring.dropped)
    }
}

/// A released range as the line names it, released at `release.at`, against the loss at `loss`.
fn release_text(release: &Release, loss: u64) -> String {
    let what = object_text(&LiveObject {
        context: release.context(),
        id: release.id,
        kind: release.kind,
        storage: release.storage,
        gpu_address: release.gpu_address,
        len: release.len,
        gone: false,
    });
    format!(
        "{what}, released at {} s since 1970 ({})",
        seconds(release.at),
        against(release.at, loss)
    )
}

/// Where an address lies among the released ranges the history keeps ([`history`]).
#[derive(Debug, PartialEq, Eq)]
pub struct History<'a> {
    /// The latest released range that held it, with the address's offset in it.
    pub held: Option<(&'a Release, u64)>,
    /// The nearest released ranges below and above it, where none held it.
    pub nearest: Option<Nearest<'a, Release>>,
}

/// Where `address` lies among `releases`, the history's, the newest first ([`Releases::kept`]):
/// the latest that held it, else the nearest below and above, of two at the same distance the
/// shorter, then the later released.
pub fn history(address: u64, releases: &[Release]) -> History<'_> {
    let held = (releases.iter()).find_map(|release| {
        let offset = offset_in(release.gpu_address, release.len, address)?;
        Some((release, offset))
    });
    History {
        held,
        nearest: held.is_none().then(|| nearest(releases, address)),
    }
}

impl History<'_> {
    /// Where the address lies among the released ranges as the line says it, each range's release
    /// against the loss at `loss`.
    pub fn text(&self, loss: u64) -> String {
        let name = |release: &Release| release_text(release, loss);
        match (&self.held, &self.nearest) {
            (Some((release, offset)), _) => format!(
                "in the history of released ranges, the last kept that held it, at offset \
                 {offset:#x}: {}",
                name(release)
            ),
            (None, Some(nearest)) => format!(
                "in the history of released ranges, none kept held it ({})",
                nearest.text(name)
            ),
            (None, None) => "in the history of released ranges, none kept held it".to_owned(),
        }
    }
}

/// What the history keeps, as the line says it: how many of each kind, and how many it keeps at
/// most and dropped past that.
fn kept_text(released: &[Release], dropped: u64) -> String {
    let count = |kind| released.iter().filter(|r| r.kind == kind).count();
    format!(
        "The history of released ranges keeps the last {RELEASES_KEPT}, and holds {} ({dropped} \
         older dropped past them): {} buffers, {} textures over a buffer, {} heaps, {} two-copy \
         Managed textures' CPU-copy buffers",
        released.len(),
        count(ObjectKind::Buffer),
        count(ObjectKind::Texture),
        count(ObjectKind::Heap),
        count(ObjectKind::ManagedCopy),
    )
}

/// How many command buffers committed and not complete at a loss its line lists, the oldest
/// first ([`Flights::lines`]); the rest are counted.
pub const FLIGHTS_LISTED: usize = 16;

/// How many runs of one kind of encoder the line lists of a command buffer's encoders; the rest
/// are counted.
pub const KINDS_LISTED: usize = 32;

/// How many ids of each list an encoder keeps ([`Ids`]), and the oldest command buffer's line
/// lists of each of its lists.
pub const IDS_KEPT: usize = 32;

/// How many render passes the oldest command buffer's line lists; the rest are counted.
pub const PASSES_LISTED: usize = 16;

/// The kind of an encoder of a command buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EncoderKind {
    /// `MTLBlitCommandEncoder`.
    Blit,
    /// `MTLComputeCommandEncoder`.
    Compute,
    /// `MTLRenderCommandEncoder`.
    Render,
}

impl EncoderKind {
    /// Its name on the line.
    fn name(self) -> &'static str {
        match self {
            EncoderKind::Blit => "blit",
            EncoderKind::Compute => "compute",
            EncoderKind::Render => "render",
        }
    }
}

/// The guest's object ids an encoder names in one of its lists ([`EncoderNames`]): each once, in
/// the order first named, the first [`IDS_KEPT`]; past them, the names of others counted; and the
/// names of objects with no guest id ([`Named`]), counted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Ids {
    /// The ids named, each once, in the order first named: the first [`IDS_KEPT`].
    kept: Vec<u64>,
    /// How many names of ids past those kept.
    past: u64,
    /// How many names of objects with no guest id.
    unnumbered: u64,
}

impl Ids {
    /// Notes a name of `id`, 0 for an object with no guest id: a look through the at most
    /// [`IDS_KEPT`] ids kept.
    pub fn add(&mut self, id: u64) {
        if id == 0 {
            self.unnumbered += 1;
        } else if !self.kept.contains(&id) {
            match self.kept.len() < IDS_KEPT {
                true => self.kept.push(id),
                false => self.past += 1,
            }
        }
    }

    /// Notes `other`'s ids and counts in this list.
    fn join(&mut self, other: &Ids) {
        for &id in &other.kept {
            self.add(id);
        }
        self.past += other.past;
        self.unnumbered += other.unnumbered;
    }

    /// The list as the line says it.
    fn text(&self) -> String {
        if self.kept.is_empty() && self.past == 0 && self.unnumbered == 0 {
            return "none".to_owned();
        }
        let mut parts = (self.kept.iter())
            .map(|id| format!("{id:#x}"))
            .collect::<Vec<_>>();
        if self.past > 0 {
            parts.push(format!(
                "{} more names past the first {IDS_KEPT} ids",
                self.past
            ));
        }
        if self.unnumbered > 0 {
            parts.push(format!(
                "{} names of objects with no guest id",
                self.unnumbered
            ));
        }
        parts.join(", ")
    }
}

/// What an encoder of a command buffer names (the module's docs, "The command buffers in
/// flight").
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncoderNames {
    /// Its kind.
    pub kind: EncoderKind,
    /// A render encoder's pass's attachments: each one's point and its texture's guest id, 0 for
    /// one with none.
    pub attachments: Vec<(Point, u64)>,
    /// The heaps its `useHeap:` calls declare.
    pub heaps: Ids,
    /// The buffers its `useResource:` calls declare.
    pub declared_buffers: Ids,
    /// The textures its `useResource:` calls declare.
    pub declared_textures: Ids,
    /// The buffers its `set*Buffer:` calls bind (`setTessellationFactorBuffer:` among them).
    pub bound_buffers: Ids,
}

impl EncoderNames {
    /// An encoder of `kind` that names nothing yet.
    pub fn new(kind: EncoderKind) -> EncoderNames {
        EncoderNames {
            kind,
            attachments: Vec::new(),
            heaps: Ids::default(),
            declared_buffers: Ids::default(),
            declared_textures: Ids::default(),
            bound_buffers: Ids::default(),
        }
    }
}

/// A command buffer's encoders' names, in their order: each joins as its encoder ends or goes
/// ([`Naming`]), and the commit takes them ([`Flight`]). A handle: its clones, which its encoders
/// hold, are the same list.
#[derive(Clone, Default)]
pub struct Encoded(Arc<Mutex<Vec<EncoderNames>>>);

impl Encoded {
    /// Adds an encoder's names: one push under the list's lock.
    pub fn push(&self, names: EncoderNames) {
        self.0.lock().push(names);
    }

    /// The names so far, taken.
    pub fn take(&self) -> Vec<EncoderNames> {
        std::mem::take(&mut *self.0.lock())
    }
}

/// An encoder's names from its make, which join its command buffer's ([`Encoded`]) as it ends,
/// or as it goes unended.
pub struct Naming {
    /// Its command buffer's list.
    into: Encoded,
    /// Its names, until they join the list.
    names: Option<EncoderNames>,
}

impl Naming {
    /// The names `names` of an encoder made now, which join `into` as it ends or goes.
    pub fn new(into: &Encoded, names: EncoderNames) -> Naming {
        Naming {
            into: into.clone(),
            names: Some(names),
        }
    }

    /// Its names, until they joined the list.
    pub fn names(&mut self) -> Option<&mut EncoderNames> {
        self.names.as_mut()
    }

    /// The encoder ended: its names join the list, once.
    pub fn end(&mut self) {
        if let Some(names) = self.names.take() {
            self.into.push(names);
        }
    }
}

impl Drop for Naming {
    fn drop(&mut self) {
        self.end();
    }
}

/// A command buffer committed and not complete (the module's docs, "The command buffers in
/// flight").
#[derive(Debug)]
pub struct Flight {
    /// The object id of the guest's command buffer, `None` for one of the host's own.
    pub cb: Option<u64>,
    /// When it was committed: nanoseconds since 1970, on the wall clock.
    pub committed: u64,
    /// Its encoders' names, in their order.
    pub encoders: Vec<EncoderNames>,
}

impl Flight {
    /// The command buffer as the line names it.
    fn who(&self) -> String {
        match self.cb {
            Some(cb) => format!(
                "the guest's command buffer {cb:#x} of context {}",
                context_of(cb)
            ),
            None => "a command buffer of the host's own".to_owned(),
        }
    }

    /// Its encoders' kinds in order, as the line says them: each run of one kind once, with its
    /// count where more than one; the first [`KINDS_LISTED`] runs, then a count of the rest.
    fn kinds(&self) -> String {
        if self.encoders.is_empty() {
            return "with no encoders".to_owned();
        }
        let mut runs: Vec<(EncoderKind, usize)> = Vec::new();
        for encoder in &self.encoders {
            match runs.last_mut() {
                Some((kind, n)) if *kind == encoder.kind => *n += 1,
                _ => runs.push((encoder.kind, 1)),
            }
        }
        let listed = (runs.iter().take(KINDS_LISTED))
            .map(|&(kind, n)| match n {
                1 => kind.name().to_owned(),
                n => format!("{n} {}", kind.name()),
            })
            .collect::<Vec<_>>();
        let rest = (runs.iter().skip(KINDS_LISTED))
            .map(|&(_, n)| n)
            .sum::<usize>();
        let rest = match rest {
            0 => String::new(),
            rest => format!(", and {rest} more not listed"),
        };
        format!(
            "its encoders ({}): {}{rest}",
            self.encoders.len(),
            listed.join(", ")
        )
    }

    /// What its encoders name, as the oldest's line says it: its render passes' attachments by
    /// texture, the first [`PASSES_LISTED`] passes, then the heaps and resources its encoders
    /// declare and the buffers they bind.
    fn names(&self) -> String {
        let (mut passes, mut more_passes) = (Vec::new(), 0);
        let mut lists: [Ids; 4] = Default::default();
        for (at, encoder) in self.encoders.iter().enumerate() {
            let of = [
                &encoder.heaps,
                &encoder.declared_buffers,
                &encoder.declared_textures,
                &encoder.bound_buffers,
            ];
            for (list, of) in lists.iter_mut().zip(of) {
                list.join(of);
            }
            if encoder.kind != EncoderKind::Render {
                continue;
            }
            if passes.len() >= PASSES_LISTED {
                more_passes += 1;
                continue;
            }
            let attachments = (encoder.attachments.iter())
                .map(|&(point, id)| {
                    let point = match point {
                        Point::Color(i) => format!("colour {i}"),
                        Point::Depth => "depth".to_owned(),
                        Point::Stencil => "stencil".to_owned(),
                    };
                    match id {
                        0 => format!("{point} a texture with no guest id"),
                        id => format!("{point} texture {id:#x}"),
                    }
                })
                .collect::<Vec<_>>();
            let attachments = match attachments.is_empty() {
                true => "no attachments".to_owned(),
                false => attachments.join(", "),
            };
            passes.push(format!("encoder {}: {attachments}", at + 1));
        }
        let passes = match (passes.is_empty(), more_passes) {
            (true, _) => "no render passes".to_owned(),
            (false, 0) => format!("its render passes' attachments: {}", passes.join("; ")),
            (false, more) => format!(
                "its render passes' attachments: {}; and {more} more passes not listed",
                passes.join("; ")
            ),
        };
        let [heaps, buffers, textures, bound] = lists.map(|list| list.text());
        format!(
            "{passes}; the heaps its encoders declare (useHeap:): {heaps}; the buffers they \
             declare (useResource:): {buffers}; the textures they declare (useResource:): \
             {textures}; the buffers they bind: {bound}"
        )
    }
}

/// A device's command buffers committed and not complete (the module's docs, "The command buffers
/// in flight"), by the number of their commit: each added before rt has it, and removed as its
/// first completed handler runs, or as rt refuses its commit.
#[derive(Default)]
pub struct Flights(Mutex<(u64, BTreeMap<u64, Flight>)>);

impl Flights {
    /// Adds `flight`, committed now: its number, which removes it, never 0.
    pub fn add(&self, flight: Flight) -> u64 {
        let mut flights = self.0.lock();
        flights.0 += 1;
        let number = flights.0;
        flights.1.insert(number, flight);
        number
    }

    /// Removes the command buffer of number `number`, 0 for none: it completed, or rt refused
    /// it. What it kept is let go of after the lock.
    pub fn remove(&self, number: u64) {
        if number == 0 {
            return;
        }
        let gone = self.0.lock().1.remove(&number);
        drop(gone);
    }

    /// The number of the last command buffer committed, 0 where none was: a wait for it is one
    /// for every command buffer committed up to now ([`Flights::done_through`]).
    pub fn last(&self) -> u64 {
        self.0.lock().0
    }

    /// The highest number N such that every command buffer numbered N or less has left the list,
    /// its first completed handler run or its commit refused, whatever the order they left in,
    /// across all its device's queues: the oldest's in it less one, or [`Flights::last`] where it
    /// is empty. A later command buffer that completed first moves it nowhere.
    pub fn done_through(&self) -> u64 {
        let flights = self.0.lock();
        (flights.1.keys().next()).map_or(flights.0, |&oldest| oldest.saturating_sub(1))
    }

    /// How many command buffers are committed and not complete.
    pub fn len(&self) -> usize {
        self.0.lock().1.len()
    }

    /// Whether none is.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The lines of the device on `gpu` lost at `loss`, on the wall clock, each with its own `\n`
    /// (the module's docs, "The command buffers in flight"): the command buffers committed and
    /// not complete, the oldest first, and then, where there is one, what the oldest's encoders
    /// name.
    pub fn lines(&self, gpu: &str, loss: u64) -> Vec<String> {
        let flights = self.0.lock();
        let open = &flights.1;
        let listed = (open.values().take(FLIGHTS_LISTED))
            .map(|flight| {
                format!(
                    "{}, committed at {} s since 1970 ({}), {}",
                    flight.who(),
                    seconds(flight.committed),
                    against(flight.committed, loss),
                    flight.kinds()
                )
            })
            .collect::<Vec<_>>();
        let rest = match open.len().saturating_sub(FLIGHTS_LISTED) {
            0 => String::new(),
            rest => format!("; and {rest} more, not listed"),
        };
        let mut lines = vec![match open.len() {
            0 => format!(
                "reims-vgpu: the rt platform's device on {gpu}, at its loss, had no command \
                 buffers committed and not complete\n"
            ),
            n => format!(
                "reims-vgpu: the rt platform's device on {gpu}, at its loss, had {n} command \
                 buffers committed and not complete, the oldest committed first: {}{rest}\n",
                listed.join("; ")
            ),
        }];
        if let Some(oldest) = open.values().next() {
            lines.push(format!(
                "reims-vgpu: the oldest command buffer committed and not complete at the loss of \
                 the rt platform's device on {gpu}, {}: {}\n",
                oldest.who(),
                oldest.names()
            ));
        }
        lines
    }
}

impl Placement<'_> {
    /// The placement as the line says it: inside which live objects, each with the address's
    /// offset in it, or in none and the nearest below and above, then those the access's bytes
    /// reach beside; then the same of the mapped blocks.
    pub fn text(&self) -> String {
        let mut objects = Vec::new();
        for (object, offset) in &self.objects {
            let object = object_text(object);
            objects.push(format!("inside {object} at offset {offset:#x}"));
        }
        if let Some(nearest) = &self.nearest_objects {
            objects.push(format!("in no live object ({})", nearest.text(object_text)));
        }
        for object in &self.near_objects {
            let object = object_text(object);
            objects.push(format!("its bytes reach {object} (not its address)"));
        }
        let mut blocks = Vec::new();
        for (block, offset) in &self.blocks {
            let block = block_text(block);
            blocks.push(format!("in mapped block {block} at offset {offset:#x}"));
        }
        if let Some(nearest) = &self.nearest_blocks {
            let nearest = nearest.text(block_text);
            blocks.push(format!("in no mapped block we know ({nearest})"));
        }
        for block in &self.near_blocks {
            let block = format!("mapped block {}", block_text(block));
            blocks.push(format!("its bytes reach {block} (not its address)"));
        }
        format!("{}, {}", objects.join("; "), blocks.join("; "))
    }
}

/// What the walk of a decoder's tables at the loss found ([`LiveObjects::walk`]).
#[derive(Default)]
pub struct Walked {
    /// The live objects whose GPU range is known.
    pub objects: Vec<LiveObject>,
    /// The live objects that lie in an exported block, whose GPU range the walk never asks.
    pub exported: Vec<ExportedObject>,
    /// The blocks of live objects: a heap's block, the import a buffer lies in.
    pub blocks: Vec<MappedBlock>,
    /// How many contexts' tables were walked.
    pub contexts: usize,
    /// The contexts whose table's lock was not had in time ([`BOUND`]).
    pub unwalked: Vec<u32>,
    /// Whether the decoder's map of its contexts was not had in time, so no table was walked.
    pub contexts_busy: bool,
    /// The live textures with no GPU range the platform knows, nor an exported block they lie in.
    pub unranged_textures: u64,
}

/// What an object of the tables adds to the walk ([`buffer_parts`], [`texture_parts`],
/// [`heap_parts`], [`managed_copy_parts`]).
pub struct Parts {
    /// The object, where its GPU range is known.
    pub object: Option<LiveObject>,
    /// The object, where it lies in an exported block instead.
    pub exported: Option<ExportedObject>,
    /// Its block, where it is one the host sees no other way: a heap's, or the import a buffer
    /// lies in.
    pub block: Option<MappedBlock>,
}

impl Walked {
    /// Adds what an object of the tables adds, `texture` where it is a texture.
    pub fn add(&mut self, parts: Parts, texture: bool) {
        match (parts.object, &parts.exported) {
            (Some(object), _) => self.objects.push(object),
            (None, None) if texture => self.unranged_textures += 1,
            (None, _) => {}
        }
        self.exported.extend(parts.exported);
        self.blocks.extend(parts.block);
    }
}

/// A decoder's tables, walked at the loss: a graphics device's (`reims_vgpu`'s
/// `graphics::fault_walk`), told to the device as its decoder is made ([`FaultWatch::watch`]).
pub trait LiveObjects: Send + Sync {
    /// The tables' live objects now, each lock waited for [`BOUND`] in all at most.
    fn walk(&self) -> Walked;
    /// Whether the tables are still there: false once their decoder went.
    fn live(&self) -> bool;
}

/// The kind of import block `id` is: a live dma-buf import of the device's, a copy of host pages
/// (a `Shared` block of rt's pool), an import of the pages by their pointer; `None` where it is
/// none (`super::diag`'s record of the imports).
fn import_kind(known: &KnownBlocks, id: rt::BlockId) -> Option<&'static str> {
    if known.is_live_dma_buf(id) {
        return Some("a dma-buf import of host pages");
    }
    match super::diag::import_range(id)?.copy {
        true => Some("a copy of host pages, a Shared block of rt's pool"),
        false => Some("an import of host pages by their pointer"),
    }
}

/// What a live buffer of the tables adds: its range, and the import it lies in where it lies in
/// one (its block, which the host sees no other way); or, for one in an exported block of its
/// own, the block's number and its offset in it ([`ExportedObject`]). A deferred buffer rt has
/// not made yet adds nothing.
pub fn buffer_parts(context: u32, id: u64, gone: bool, buffer: &RtObj<RtBuffer>) -> Parts {
    let exported = (buffer.exported_block()).map(|(block, offset)| ExportedObject {
        block,
        context,
        id,
        kind: ObjectKind::Buffer,
        storage: buffer.storage_mode(),
        offset,
        len: buffer.made_rt().map_or(0, rt::Buffer::length),
        gone,
    });
    let (Some(made), Some(gpu_address)) = (buffer.made_rt(), buffer.fault_address()) else {
        return Parts {
            object: None,
            exported,
            block: None,
        };
    };
    let object = LiveObject {
        context,
        id,
        kind: ObjectKind::Buffer,
        storage: buffer.storage_mode(),
        gpu_address,
        len: made.length(),
        gone,
    };
    let block = super::diag::import_range(made.block()).map(|import| MappedBlock {
        what: format!("buffer {id:#x}'s block {}", made.block().get()),
        class: import.class,
        import: import_kind(buffer.device_known(), made.block()),
        gpu_address: import.gpu_address,
        len: import.len,
    });
    Parts {
        object: Some(object),
        exported: None,
        block,
    }
}

/// What the `Shared` buffer over a two-copy `Managed` texture's CPU copy adds, named by the
/// texture's id: as [`buffer_parts`].
pub fn managed_copy_parts(context: u32, id: u64, gone: bool, buffer: &RtObj<RtBuffer>) -> Parts {
    let mut parts = buffer_parts(context, id, gone, buffer);
    if let Some(object) = &mut parts.object {
        object.kind = ObjectKind::ManagedCopy;
    }
    if let Some(object) = &mut parts.exported {
        object.kind = ObjectKind::ManagedCopy;
    }
    parts
}

/// What a live texture of the tables adds: its range where it lies over a buffer (its bytes in
/// the buffer); for one over a buffer in an exported block, the block's number and its bytes'
/// offset in it ([`ExportedObject`]); none for any other.
pub fn texture_parts(context: u32, id: u64, gone: bool, texture: &RtObj<RtTexture>) -> Parts {
    let (mut range, mut exported) = (None, None);
    if let Some((buffer, bytes)) = texture.over()
        && let Some(buffer) = buffer.upgrade()
        && let Some(len) = bytes.end.checked_sub(bytes.start)
    {
        range = (buffer.fault_address())
            .and_then(|address| address.checked_add(bytes.start))
            .map(|gpu_address| (gpu_address, len));
        exported = buffer.exported_block().and_then(|(block, offset)| {
            Some(ExportedObject {
                block,
                context,
                id,
                kind: ObjectKind::Texture,
                storage: texture.storage_mode(),
                offset: offset.checked_add(bytes.start)?,
                len,
                gone,
            })
        });
    }
    Parts {
        object: range.map(|(gpu_address, len)| LiveObject {
            context,
            id,
            kind: ObjectKind::Texture,
            storage: texture.storage_mode(),
            gpu_address,
            len,
            gone,
        }),
        exported,
        block: None,
    }
}

/// What a live heap of the tables adds: its range, and its block.
pub fn heap_parts(context: u32, id: u64, gone: bool, heap: &RtObj<RtHeap>) -> Parts {
    let (block, block_address, offset, size) = heap.fault_span();
    Parts {
        object: Some(LiveObject {
            context,
            id,
            kind: ObjectKind::Heap,
            storage: heap.storage_mode(),
            gpu_address: heap.gpu_address(),
            len: size,
            gone,
        }),
        exported: None,
        block: Some(MappedBlock {
            what: format!(
                "heap {id:#x}'s block {} of allocation {} (the heap at offset {offset:#x})",
                block.id.get(),
                block.allocation.get()
            ),
            class: block.class,
            import: import_kind(heap.device_known(), block.id),
            gpu_address: block_address,
            len: block.len,
        }),
    }
}

/// rt's pool allocations still mapped, by number: their storage mode, first byte's GPU address
/// and length. As rt's `BlockMappings`, the device's (`DeviceOptions::block_mappings`), it maps
/// nothing and refuses nothing: it notes each allocation rt makes, which rt tells it of before it
/// hands out a block of it, counting it on the thread rt made it on
/// (`super::buffer`'s `pool_allocation_made_here`) and noting when its make began and ended, its
/// length, storage mode and thread (`super::overlap::pool_made`), and forgets each, which rt tells
/// it of before it frees it.
#[derive(Default)]
pub struct Allocations(Mutex<BTreeMap<u32, (rt::MemoryClass, u64, u64)>>);

impl rt::BlockMappings for Allocations {
    fn map(&self, allocation: &rt::AllocationInfo) -> bool {
        super::buffer::pool_allocation_made_here();
        super::overlap::pool_made(
            allocation.make_start,
            allocation.make_nanos,
            allocation.len,
            allocation.class,
        );
        let at = (allocation.class, allocation.gpu_address, allocation.len);
        self.0.lock().insert(allocation.id.get(), at);
        true
    }

    fn unmap(&self, allocation: rt::AllocationId) {
        self.0.lock().remove(&allocation.get());
    }
}

/// A device's watch for its loss's fault: rt's pool allocations, the blocks the device knows, its
/// history of released ranges, and the decoders' tables to walk at the loss. A handle: its clones
/// are the same watch.
#[derive(Clone, Default)]
pub struct FaultWatch {
    /// rt's pool allocations still mapped.
    allocations: Arc<Allocations>,
    /// The device's known blocks, its set's handle.
    known: KnownBlocks,
    /// The device's history of released ranges.
    releases: Releases,
    /// The decoders' tables.
    tables: Arc<Mutex<Vec<Arc<dyn LiveObjects>>>>,
}

/// The note on an instruction pointer's line (the module's docs).
const NO_CODE_SEEN: &str = "none of the ranges the host sees holds shader code (a pipeline's \
                            code lies in the driver's own memory, of which the host sees \
                            nothing), so this says only what lies around the address";

impl FaultWatch {
    /// A watch over `allocations`, the device's `BlockMappings`, and its known blocks `known`.
    pub fn new(allocations: Arc<Allocations>, known: KnownBlocks) -> FaultWatch {
        FaultWatch {
            allocations,
            known,
            releases: Releases::default(),
            tables: Arc::default(),
        }
    }

    /// The device's history of released ranges, which the tables' objects record their releases
    /// in as they go (`super::device`'s `RtDevice::releases`).
    pub fn releases(&self) -> &Releases {
        &self.releases
    }

    /// Walks `tables` too at the loss, and no longer those whose decoder went.
    pub fn watch(&self, tables: Arc<dyn LiveObjects>) {
        let mut all = self.tables.lock();
        all.retain(|t| t.live());
        all.push(tables);
    }

    /// The fault's lines for the device on `gpu`, lost at `loss` on the wall clock, its report
    /// `fault` (rt's `DeviceLoss::fault`), each with its own `\n` (the module's docs): the report,
    /// each instruction pointer and each access placed, among what is live and what was released,
    /// rt's placement of each address, and what the placement saw; one line where rt read no
    /// report.
    pub fn lines(&self, gpu: &str, fault: Option<&rt::DeviceFault>, loss: u64) -> Vec<String> {
        let head = format!("reims-vgpu: the rt platform's device on {gpu} faulted: ");
        let Some(fault) = fault else {
            return vec![format!(
                "{head}rt read no report of the fault (the device has no VK_EXT_device_fault, \
                 the read failed, or a window's swapchain reported the loss)\n"
            )];
        };
        let vendor = (fault.vendor.iter())
            .map(|(text, code, data)| format!("{text:?} code {code:#x} data {data:#x}"))
            .collect::<Vec<_>>();
        let mut lines = vec![format!(
            "{head}the driver's report: description {:?}, {} vendor records{}{}, {} accesses, {} \
             instruction pointers\n",
            fault.description,
            vendor.len(),
            if vendor.is_empty() { "" } else { ": " },
            vendor.join("; "),
            fault.accesses.len(),
            fault.instructions.len()
        )];
        let (released, dropped) = self.releases.kept();
        let (objects, exported, blocks, seen) = self.seen();
        let placed = |at: &rt::FaultAddress| {
            let placement = place(at.address, at.precision, &objects, &blocks);
            let (start, end) = placement.range;
            let text = placement.text();
            let history = history(at.address, &released).text(loss);
            format!("{at:?}, the bytes [{start:#x}, {end:#x}): {text}; {history}")
        };
        for at in &fault.instructions {
            let at = placed(at);
            lines.push(format!("{head}instruction {at}; {NO_CODE_SEEN}\n"));
        }
        for at in &fault.accesses {
            lines.push(format!("{head}access {}\n", placed(at)));
        }
        // rt's own placement of each address among its ranges, live and recently freed, with
        // the tables' objects in each exported block it names.
        for place in &fault.places {
            let ranges = (place.holders.iter()).chain(&place.below).chain(&place.above);
            let exports = exports_text(&export_blocks(ranges), &exported);
            lines.push(format!("{head}rt places {place}{exports}\n"));
        }
        let kept = kept_text(&released, dropped);
        lines.push(format!("{head}{seen}. {kept}\n"));
        lines
    }

    /// What the host sees now: the live objects of the tables' walk, every mapped block it sees,
    /// and what it saw and did not, as the line says it.
    fn seen(&self) -> (Vec<LiveObject>, Vec<ExportedObject>, Vec<MappedBlock>, String) {
        let tables = (self.tables.lock().iter())
            .filter(|t| t.live())
            .cloned()
            .collect::<Vec<_>>();
        let (mut objects, mut exported, mut of_objects) = (Vec::new(), Vec::new(), Vec::new());
        let (mut contexts, mut unwalked, mut busy, mut unranged) = (0, Vec::new(), 0, 0);
        for table in &tables {
            let mut walked = table.walk();
            objects.append(&mut walked.objects);
            exported.append(&mut walked.exported);
            of_objects.append(&mut walked.blocks);
            contexts += walked.contexts;
            unwalked.append(&mut walked.unwalked);
            busy += usize::from(walked.contexts_busy);
            unranged += walked.unranged_textures;
        }
        let mut blocks = Vec::new();
        let pool = self.allocations.0.lock().clone();
        for (&id, &(class, gpu_address, len)) in &pool {
            blocks.push(MappedBlock {
                what: format!("rt's pool allocation {id}"),
                class,
                import: None,
                gpu_address,
                len,
            });
        }
        let mut ids = BTreeSet::new();
        let known = self.known.fault_blocks(BOUND);
        let known_text = match &known {
            Some(known) => format!("{} known blocks", known.len()),
            None => "no known blocks (their lock was not had in time)".to_owned(),
        };
        for (gpu_address, info) in known.unwrap_or_default() {
            ids.insert(info.id);
            blocks.push(MappedBlock {
                what: format!(
                    "known block {} of allocation {}",
                    info.id.get(),
                    info.allocation.get()
                ),
                class: info.class,
                import: import_kind(&self.known, info.id),
                gpu_address,
                len: info.len,
            });
        }
        let mut dma_bufs = 0;
        for id in self.known.live_dma_bufs() {
            let Some(import) = super::diag::import_range(id) else {
                continue;
            };
            if ids.insert(id) {
                dma_bufs += 1;
                blocks.push(MappedBlock {
                    what: format!("dma-buf import block {}", id.get()),
                    class: import.class,
                    import: Some("a dma-buf import of host pages"),
                    gpu_address: import.gpu_address,
                    len: import.len,
                });
            }
        }
        let of_objects_count = of_objects.len();
        blocks.append(&mut of_objects);
        let count = |kind| (objects.iter()).filter(|o| o.kind == kind).count();
        let mut not_walked = String::new();
        if busy > 0 {
            not_walked += &format!(
                "; the map of contexts of {busy} decoders was not had in time, so none of its \
                 tables was walked"
            );
        }
        if !unwalked.is_empty() {
            not_walked += &format!(
                "; the tables of contexts {unwalked:?} were not had in time and not walked"
            );
        }
        let seen = format!(
            "placed among {} live objects with a GPU range ({} buffers, {} textures over a \
             buffer, {} heaps, {} two-copy Managed textures' CPU-copy buffers) of {} decoders' {} \
             contexts' tables, walked at the loss{not_walked}, beside {} live objects in an \
             exported block, which rt's lines name by its number, and {unranged} live textures \
             with no GPU range the platform knows (images of rt's own: one in a heap lies in its \
             heap's range); and among {} mapped blocks the host sees: {} of rt's pool \
             allocations, {known_text}, {dma_bufs} more live dma-buf imports, {of_objects_count} \
             blocks of live objects (heaps' blocks, buffers' imports). Not seen: rt's parameter \
             ring, zero source and staging region, images rt makes with memory of their own, \
             exports and surfaces, and an import by the pages' pointer that no live object or \
             known block lies in",
            objects.len(),
            count(ObjectKind::Buffer),
            count(ObjectKind::Texture),
            count(ObjectKind::Heap),
            count(ObjectKind::ManagedCopy),
            tables.len(),
            contexts,
            exported.len(),
            blocks.len(),
            pool.len(),
        );
        (objects, exported, blocks, seen)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ash::vk::DeviceFaultAddressTypeEXT as Kind;

    /// A loss's time on the wall clock, as a test gives it: 1795560000.25 s since 1970, a quarter
    /// second past 2026-11-24 22:40 UTC.
    const LOSS: u64 = 1_795_560_000_250_000_000;

    /// What an access's line says of the history where it keeps nothing.
    const NONE_RELEASED: &str = "in the history of released ranges, none kept held it (nearest \
                                 below: none; nearest above: none)";

    fn buffer(id: u64, gpu_address: u64, len: u64) -> LiveObject {
        LiveObject {
            context: 3,
            id,
            kind: ObjectKind::Buffer,
            storage: 2,
            gpu_address,
            len,
            gone: false,
        }
    }

    fn exported(block: u32, kind: ObjectKind, id: u64, offset: u64, len: u64) -> ExportedObject {
        ExportedObject {
            block,
            context: 3,
            id,
            kind,
            storage: 2,
            offset,
            len,
            gone: false,
        }
    }

    /// Of each exported block rt's placement names, its line names the tables' live objects that
    /// lie in it, by kind, guest id, context, storage mode, length and offset in the block, one
    /// removed and kept for an open view said so, in the order the walk found them; a block none
    /// lies in says so, and a placement that names no exported block adds nothing
    #[test]
    fn an_exported_block_rt_names_is_joined_to_the_objects_in_it() {
        let held = [
            exported(12, ObjectKind::Buffer, 0x1_0000_0005, 0xe0, 0x1000),
            exported(14, ObjectKind::Buffer, 0x1_0000_0006, 0, 0x1000),
            ExportedObject {
                gone: true,
                ..exported(12, ObjectKind::Texture, 0x1_0000_0007, 0x100, 0x800)
            },
        ];
        assert_eq!(
            exports_text(&[12, 13], &held),
            "; export block 12 holds buffer 0x100000005 of context 3 (Private, length 0x1000, at \
             offset 0xe0 in it), texture 0x100000007 of context 3 (Private, length 0x800, at \
             offset 0x100 in it, removed and kept for an open view); export block 13 holds no \
             live object of the tables"
        );
        assert_eq!(exports_text(&[], &held), "");
    }

    fn block(what: &str, gpu_address: u64, len: u64, import: Option<&'static str>) -> MappedBlock {
        MappedBlock {
            what: what.to_owned(),
            class: rt::MemoryClass::Shared,
            import,
            gpu_address,
            len,
        }
    }

    /// An access is rounded down to its precision; one inside a live buffer names it, its
    /// storage mode, length and the address's offset in it, and the mapped blocks that hold it;
    /// one whose bytes reach an object or block without its address says so; one in neither says
    /// it is in no live object and no mapped block we know.
    #[test]
    fn a_fault_s_access_is_placed_among_live_objects_and_mapped_blocks() {
        assert_eq!(
            fault_range(0xdef63a3123, 0x1000),
            (0xdef63a3000, 0xdef63a4000)
        );
        assert_eq!(fault_range(0x10, 0), (0x10, 0x11));
        let objects = [
            buffer(0x1_0000_0001, 0x10_0000, 0x400),
            LiveObject {
                kind: ObjectKind::Heap,
                id: 0x1_0000_0002,
                storage: 0,
                gone: true,
                ..buffer(0, 0x10_1000, 0x1000)
            },
        ];
        let blocks = [
            block("rt's pool allocation 7", 0x10_0000, 0x100_0000, None),
            block(
                "known block 9 of allocation 12",
                0x200_0000,
                0x1_0000,
                Some("an import"),
            ),
        ];
        // Inside the buffer and the pool allocation.
        let inside = place(0x10_0123, 0x10, &objects, &blocks);
        assert_eq!(inside.range, (0x10_0120, 0x10_0130));
        assert_eq!(inside.objects, [(&objects[0], 0x123)]);
        assert!(inside.near_objects.is_empty() && inside.near_blocks.is_empty());
        assert_eq!(inside.blocks, [(&blocks[0], 0x123)]);
        assert_eq!(
            (inside.nearest_objects, inside.nearest_blocks),
            (None, None)
        );
        assert_eq!(
            place(0x10_0123, 0x10, &objects, &blocks).text(),
            "inside buffer 0x100000001 of context 3 (Private, length 0x400, at 0x100000) at offset \
             0x123, in mapped block rt's pool allocation 7 (Shared, not an import, length \
             0x1000000, at 0x100000) at offset 0x123"
        );
        // Past the buffer's end, in the page that holds it and the removed heap.
        let past = place(0x10_0800, 0x1000, &objects, &blocks);
        assert_eq!(past.range, (0x10_0000, 0x10_1000));
        assert!(past.objects.is_empty());
        assert_eq!(past.near_objects, [&objects[0]]);
        assert_eq!(past.blocks, [(&blocks[0], 0x800)]);
        assert!(past.text().ends_with(
            "; its bytes reach buffer 0x100000001 of context 3 (Private, length 0x400, at \
             0x100000) (not its address), in mapped block rt's pool allocation 7 (Shared, not an \
             import, length 0x1000000, at 0x100000) at offset 0x800"
        ));
        let heap = place(0x10_1ff0, 0x10, &objects, &blocks);
        assert_eq!(heap.objects, [(&objects[1], 0xff0)]);
        assert!(heap.text().starts_with(
            "inside heap 0x100000002 of context 3 (Shared, length 0x1000, at 0x101000, removed \
             and kept for an open view) at offset 0xff0, in mapped block"
        ));
        // In the import alone.
        let import = place(0x200_8000, 0x1000, &objects, &blocks);
        assert_eq!(import.blocks, [(&blocks[1], 0x8000)]);
        assert!(import.text().ends_with(
            ", in mapped block known block 9 of allocation 12 (Shared, an import, length \
             0x10000, at 0x2000000) at offset 0x8000"
        ));
        // In neither: a freed allocation's address.
        let nowhere = place(0xdef63a3000, 0x1000, &objects, &blocks);
        assert!(nowhere.objects.is_empty() && nowhere.blocks.is_empty());
        assert!(nowhere.near_objects.is_empty() && nowhere.near_blocks.is_empty());
        assert!(nowhere.nearest_objects.is_some() && nowhere.nearest_blocks.is_some());
        // The address just before a block: its bytes end at the block's first byte, which they
        // do not hold; at a coarser precision they reach both blocks, the address in neither.
        let before = place(0x1ff_fff0, 0x100, &objects, &blocks);
        assert_eq!(before.range, (0x1ff_ff00, 0x200_0000));
        assert!(before.blocks.is_empty() && before.near_blocks.is_empty());
        let reaching = place(0x1ff_fff0, 0x400_0000, &objects, &blocks);
        assert_eq!(reaching.range, (0, 0x400_0000));
        assert!(reaching.blocks.is_empty());
        assert_eq!(reaching.near_blocks, [&blocks[0], &blocks[1]]);
        assert!(reaching.text().contains("its bytes reach mapped block"));
    }

    /// An access in no live object names the nearest below the address, with the bytes from its
    /// end to the address, and the nearest above, with the bytes from the address to its first
    /// byte, or none; so of the mapped blocks; of two at the same distance, the shorter.
    #[test]
    fn an_access_in_no_range_names_the_nearest_below_and_above() {
        let objects = [
            buffer(0x1_0000_0001, 0x10_0000, 0x400),
            buffer(0x1_0000_0002, 0x30_0000, 0x1000),
            buffer(0x1_0000_0003, 0x50_0000, 0x10),
        ];
        let blocks = [
            block("rt's pool allocation 7", 0x10_0000, 0x10_0000, None),
            block("known block 9 of allocation 7", 0x10_0000, 0x10_0000, None),
            block("rt's pool allocation 8", 0x40_0000, 0x100_0000, None),
        ];
        // Between the first two buffers, past the first allocation's end, below the second.
        let between = place(0x20_0400, 0x1000, &objects, &blocks);
        let objects_near = between.nearest_objects.as_ref().unwrap();
        assert_eq!(
            objects_near.below,
            Some((&objects[0], 0x20_0400 - 0x10_0400))
        );
        assert_eq!(
            objects_near.above,
            Some((&objects[1], 0x30_0000 - 0x20_0400))
        );
        let blocks_near = between.nearest_blocks.as_ref().unwrap();
        // The known block spans its whole allocation: of the two, the first.
        assert_eq!(blocks_near.below, Some((&blocks[0], 0x400)));
        assert_eq!(blocks_near.above, Some((&blocks[2], 0x40_0000 - 0x20_0400)));
        assert_eq!(
            between.text(),
            "in no live object (nearest below: buffer 0x100000001 of context 3 (Private, length \
             0x400, at 0x100000), the address 0x100000 bytes past its end; nearest above: buffer \
             0x100000002 of context 3 (Private, length 0x1000, at 0x300000), its first byte \
             0xffc00 bytes past the address), in no mapped block we know (nearest below: rt's \
             pool allocation 7 (Shared, not an import, length 0x100000, at 0x100000), the \
             address 0x400 bytes past its end; nearest above: rt's pool allocation 8 (Shared, not \
             an import, length 0x1000000, at 0x400000), its first byte 0x1ffc00 bytes past the \
             address)"
        );
        // A block shorter than another that ends at the same byte is named first.
        let shorter = [
            block("rt's pool allocation 7", 0x10_0000, 0x10_0000, None),
            block("known block 9 of allocation 7", 0x18_0000, 0x8_0000, None),
        ];
        let after = place(0x20_0000, 0x10, &objects, &shorter);
        let near = after.nearest_blocks.unwrap();
        assert_eq!(near.below, Some((&shorter[1], 0)));
        assert_eq!(near.above, None);
        // Inside a buffer and above every block: the blocks' nearest alone.
        let inside = place(0x50_0008, 0x10, &objects, &blocks[..2]);
        assert_eq!(inside.nearest_objects, None);
        let near = inside.nearest_blocks.unwrap();
        assert_eq!(near.below, Some((&blocks[0], 0x30_0008)));
        assert_eq!(near.above, None);
        // Below every range, and in an empty set.
        let low = place(0x1000, 0x1000, &objects, &[]);
        let near = low.nearest_objects.as_ref().unwrap();
        assert_eq!(
            (near.below, near.above),
            (None, Some((&objects[0], 0xf_f000)))
        );
        assert_eq!(
            low.nearest_blocks,
            Some(Nearest {
                below: None,
                above: None
            })
        );
        assert!(
            low.text().ends_with(
                ", in no mapped block we know (nearest below: none; nearest above: none)"
            )
        );
    }

    /// A decoder's tables as a test makes them.
    struct Tables {
        objects: Vec<LiveObject>,
        exported: Vec<ExportedObject>,
        blocks: Vec<MappedBlock>,
        unwalked: Vec<u32>,
        live: bool,
    }

    impl LiveObjects for Tables {
        fn walk(&self) -> Walked {
            let mut walked = Walked {
                contexts: 2,
                unwalked: self.unwalked.clone(),
                ..Walked::default()
            };
            for object in &self.objects {
                let texture = object.kind == ObjectKind::Texture;
                walked.add(
                    Parts {
                        object: Some(object.clone()),
                        exported: None,
                        block: None,
                    },
                    texture,
                );
            }
            for object in &self.exported {
                let texture = object.kind == ObjectKind::Texture;
                walked.add(
                    Parts {
                        object: None,
                        exported: Some(object.clone()),
                        block: None,
                    },
                    texture,
                );
            }
            walked.add(
                Parts {
                    object: None,
                    exported: None,
                    block: None,
                },
                true,
            );
            walked.blocks.extend(self.blocks.iter().cloned());
            walked
        }

        fn live(&self) -> bool {
            self.live
        }
    }

    /// A lost device's fault lines name the driver's report, place each instruction pointer with
    /// the note that the host sees no shader code, place each access among the tables' live
    /// objects and the blocks the host sees, and say what the placement saw; a decoder that went
    /// is not walked.
    #[test]
    fn a_lost_device_s_lines_name_the_driver_s_fault_and_place_it() {
        let allocations = Arc::new(Allocations::default());
        let private = (rt::MemoryClass::Private, 0x52_0000_0000, 0x100_0000);
        let shared = (rt::MemoryClass::Shared, 0x53_0000_0000, 0x100_0000);
        allocations.0.lock().extend([(7, private), (8, shared)]);
        let watch = FaultWatch::new(Arc::clone(&allocations), KnownBlocks::default());
        watch.watch(Arc::new(Tables {
            objects: vec![buffer(0x9, 0x1000, 0x1000)],
            exported: Vec::new(),
            blocks: Vec::new(),
            unwalked: Vec::new(),
            live: false,
        }));
        watch.watch(Arc::new(Tables {
            objects: vec![
                buffer(0x1_0000_0001, 0x52_0000_0000, 0x1000),
                LiveObject {
                    kind: ObjectKind::Texture,
                    ..buffer(0x1_0000_0002, 0x52_0000_0800, 0x400)
                },
            ],
            exported: vec![exported(12, ObjectKind::Texture, 0x1_0000_0004, 0x100, 0x800)],
            blocks: vec![block(
                "buffer 0x100000003's block 44",
                0x60_0000_0000,
                0x1_0000,
                Some("an import of host pages by their pointer"),
            )],
            unwalked: vec![5],
            live: true,
        }));
        let at = |kind, address, precision| rt::FaultAddress {
            kind,
            address,
            precision,
        };
        let fault = rt::DeviceFault {
            description: "a fault".to_owned(),
            accesses: vec![
                at(Kind::READ_INVALID, 0x52_0000_0810, 0x10),
                at(Kind::WRITE_INVALID, 0x52_1af0_0000, 0x1000),
            ],
            instructions: vec![at(Kind::INSTRUCTION_POINTER_FAULT, 0xde_fe1f_bef0, 0x10)],
            vendor: vec![("mmu".to_owned(), 0x1f, 0x2)],
            places: Vec::new(),
        };
        let lines = watch.lines("GPU (10de:2204)", Some(&fault), LOSS);
        let head = "reims-vgpu: the rt platform's device on GPU (10de:2204) faulted: ";
        assert_eq!(lines.len(), 5);
        for line in &lines {
            assert!(line.starts_with(head) && line.ends_with('\n'), "{line}");
        }
        assert_eq!(
            lines[0],
            format!(
                "{head}the driver's report: description \"a fault\", 1 vendor records: \"mmu\" \
                 code 0x1f data 0x2, 2 accesses, 1 instruction pointers\n"
            )
        );
        assert_eq!(
            lines[1],
            format!(
                "{head}instruction INSTRUCTION_POINTER_FAULT 0xdefe1fbef0 precision 0x10, the \
                 bytes [0xdefe1fbef0, 0xdefe1fbf00): in no live object (nearest below: buffer \
                 0x100000001 of context 3 (Private, length 0x1000, at 0x5200000000), the address \
                 0x8cfe1faef0 bytes past its end; nearest above: none), in no mapped block we \
                 know (nearest below: buffer 0x100000003's block 44 (Shared, an import of host \
                 pages by their pointer, length 0x10000, at 0x6000000000), the address \
                 0x7efe1ebef0 bytes past its end; nearest above: none); {NONE_RELEASED}; \
                 {NO_CODE_SEEN}\n"
            )
        );
        assert_eq!(
            lines[2],
            format!(
                "{head}access READ_INVALID 0x5200000810 precision 0x10, the bytes [0x5200000810, \
                 0x5200000820): inside buffer 0x100000001 of context 3 (Private, length 0x1000, \
                 at 0x5200000000) at offset 0x810; inside texture 0x100000002 of context 3 \
                 (Private, length 0x400, at 0x5200000800) at offset 0x10, in mapped block rt's \
                 pool allocation 7 (Private, not an import, length 0x1000000, at 0x5200000000) at \
                 offset 0x810; {NONE_RELEASED}\n"
            )
        );
        assert_eq!(
            lines[3],
            format!(
                "{head}access WRITE_INVALID 0x521af00000 precision 0x1000, the bytes \
                 [0x521af00000, 0x521af01000): in no live object (nearest below: buffer \
                 0x100000001 of context 3 (Private, length 0x1000, at 0x5200000000), the address \
                 0x1aeff000 bytes past its end; nearest above: none), in no mapped block we know \
                 (nearest below: rt's pool allocation 7 (Private, not an import, length \
                 0x1000000, at 0x5200000000), the address 0x19f00000 bytes past its end; nearest \
                 above: rt's pool allocation 8 (Shared, not an import, length 0x1000000, at \
                 0x5300000000), its first byte 0xe5100000 bytes past the address); \
                 {NONE_RELEASED}\n"
            )
        );
        assert_eq!(
            lines[4],
            format!(
                "{head}placed among 2 live objects with a GPU range (1 buffers, 1 textures over a \
                 buffer, 0 heaps, 0 two-copy Managed textures' CPU-copy buffers) of 1 decoders' 2 \
                 contexts' tables, walked at the loss; the tables of contexts [5] were not had in \
                 time and not walked, beside 1 live objects in an exported block, which rt's lines \
                 name by its number, and 1 live textures with no GPU range the platform knows \
                 (images of rt's own: one in a heap lies in its heap's range); and among 3 mapped \
                 blocks the host sees: 2 of rt's pool allocations, 0 known blocks, 0 more live \
                 dma-buf imports, 1 blocks of live objects (heaps' blocks, buffers' imports). Not \
                 seen: rt's parameter ring, zero source and staging region, images rt makes with \
                 memory of their own, exports and surfaces, and an import by the pages' pointer \
                 that no live object or known block lies in. The history of released ranges keeps \
                 the last 4096, and holds 0 (0 older dropped past them): 0 buffers, 0 textures \
                 over a buffer, 0 heaps, 0 two-copy Managed textures' CPU-copy buffers\n"
            )
        );
        // An allocation rt unmapped is no longer among the blocks.
        allocations.0.lock().remove(&8);
        let lines = watch.lines("GPU (10de:2204)", Some(&fault), LOSS);
        assert!(lines[3].ends_with(&format!(
            "nearest above: buffer 0x100000003's block 44 (Shared, an import of host pages by \
             their pointer, length 0x10000, at 0x6000000000), its first byte 0xde5100000 bytes \
             past the address); {NONE_RELEASED}\n"
        )));
    }

    /// A millisecond, in nanoseconds.
    const MS: u64 = 1_000_000;

    fn release(kind: ObjectKind, id: u64, storage: u64, (gpu_address, len): (u64, u64)) -> Release {
        Release {
            kind,
            id,
            storage,
            gpu_address,
            len,
            at: 0,
        }
    }

    /// The release of `release`, `before` nanoseconds before the loss.
    fn before(release: Release, before: u64) -> Release {
        Release {
            at: LOSS - before,
            ..release
        }
    }

    /// An address a released range held names the latest the history keeps that held it, with
    /// the address's offset in it, by the object's kind, guest id, context, storage mode, length
    /// and GPU address, and when it was released against the loss. A fault's access line says so,
    /// and the line of what the placement saw counts the history.
    #[test]
    fn a_fault_s_address_names_the_last_released_range_that_held_it() {
        let releases = Releases::default();
        let heap = release(
            ObjectKind::Heap,
            0x300_0000_0001,
            2,
            (0x52_0000_0000, 0x10_0000),
        );
        releases.push_at(before(heap, 100 * MS));
        let older = release(
            ObjectKind::Buffer,
            0x300_0000_0002,
            2,
            (0x52_0000_0000, 0x1000),
        );
        releases.push_at(before(older, 50 * MS));
        let newer = release(
            ObjectKind::Buffer,
            0x500_0000_0003,
            0,
            (0x52_0000_0000, 0x800),
        );
        releases.push_at(before(newer, 3 * MS));
        let texture = release(
            ObjectKind::Texture,
            0x300_0000_0004,
            2,
            (0x60_0000_0100, 0x100),
        );
        releases.push_at(before(texture, 2 * MS));
        let copy = release(
            ObjectKind::ManagedCopy,
            0x300_0000_0005,
            0,
            (0x70_0000_0000, 0x1_0000),
        );
        releases.push_at(before(copy, MS));
        let (kept, dropped) = releases.kept();
        assert_eq!((kept.len(), dropped), (5, 0));
        // The newest first.
        let ids = kept.iter().map(|r| r.id & 0xff).collect::<Vec<_>>();
        assert_eq!(ids, [5, 4, 3, 2, 1]);
        let contexts = kept.iter().map(Release::context).collect::<Vec<_>>();
        assert_eq!(contexts, [3, 3, 5, 3, 3]);

        let held = history(0x52_0000_0010, &kept);
        assert_eq!(
            held,
            History {
                held: Some((&kept[2], 0x10)),
                nearest: None
            }
        );
        let held = held.text(LOSS);
        assert_eq!(
            held,
            "in the history of released ranges, the last kept that held it, at offset 0x10: \
             buffer 0x50000000003 of context 5 (Shared, length 0x800, at 0x5200000000), released \
             at 1795560000.247000 s since 1970 (0.003000 s before the loss)"
        );
        let text = |address| history(address, &kept).text(LOSS);
        // Past the newer buffer's end, the older's; past both, the heap's.
        assert_eq!(
            text(0x52_0000_0900),
            "in the history of released ranges, the last kept that held it, at offset 0x900: \
             buffer 0x30000000002 of context 3 (Private, length 0x1000, at 0x5200000000), \
             released at 1795560000.200000 s since 1970 (0.050000 s before the loss)"
        );
        assert_eq!(
            text(0x52_000f_0000),
            "in the history of released ranges, the last kept that held it, at offset 0xf0000: \
             heap 0x30000000001 of context 3 (Private, length 0x100000, at 0x5200000000), \
             released at 1795560000.150000 s since 1970 (0.100000 s before the loss)"
        );
        assert_eq!(
            text(0x60_0000_01ff),
            "in the history of released ranges, the last kept that held it, at offset 0xff: \
             texture 0x30000000004 of context 3 (Private, length 0x100, at 0x6000000100), \
             released at 1795560000.248000 s since 1970 (0.002000 s before the loss)"
        );
        assert_eq!(
            text(0x70_0000_fff0),
            "in the history of released ranges, the last kept that held it, at offset 0xfff0: \
             the CPU-copy buffer of Managed texture 0x30000000005 of context 3 (Shared, length \
             0x10000, at 0x7000000000), released at 1795560000.249000 s since 1970 (0.001000 s \
             before the loss)"
        );

        // A fault's access line, and the line of what the placement saw.
        let watch = FaultWatch::default();
        for release in kept.iter().rev() {
            watch.releases().push_at(*release);
        }
        let fault = rt::DeviceFault {
            description: "a fault".to_owned(),
            accesses: vec![rt::FaultAddress {
                kind: Kind::READ_INVALID,
                address: 0x52_0000_0010,
                precision: 0x10,
            }],
            instructions: Vec::new(),
            vendor: Vec::new(),
            places: Vec::new(),
        };
        let lines = watch.lines("GPU (10de:2204)", Some(&fault), LOSS);
        assert_eq!(lines.len(), 3, "{lines:#?}");
        assert!(lines[1].ends_with(&format!("; {held}\n")), "{}", lines[1]);
        assert!(
            lines[2].ends_with(
                ". The history of released ranges keeps the last 4096, and holds 5 (0 older \
                 dropped past them): 2 buffers, 1 textures over a buffer, 1 heaps, 1 two-copy \
                 Managed textures' CPU-copy buffers\n"
            ),
            "{}",
            lines[2]
        );
    }

    /// An address no released range held names the nearest the history keeps below it, with the
    /// bytes from its end to the address, and above it, with the bytes from the address to its
    /// first byte: of two at the same distance, the shorter, then the later released. A range
    /// released after the loss's time, as its lines were made, is said to be.
    #[test]
    fn otherwise_the_nearest_released_range_below_and_above_is_named() {
        let releases = Releases::default();
        let buffer = |id, range| release(ObjectKind::Buffer, id, 0, range);
        let first = buffer(0x300_0000_0001, (0x10_0000, 0x400));
        releases.push_at(before(first, 300 * MS));
        let again = buffer(0x300_0000_0002, (0x10_0000, 0x400));
        releases.push_at(before(again, 100 * MS));
        let short = release(ObjectKind::Heap, 0x300_0000_0003, 2, (0x40_0000, 0x1000));
        releases.push_at(Release {
            at: LOSS + 20 * MS,
            ..short
        });
        let long = buffer(0x300_0000_0004, (0x40_0000, 0x10_0000));
        releases.push_at(before(long, 10 * MS));
        let (kept, _) = releases.kept();
        let ids = kept.iter().map(|r| r.id & 0xff).collect::<Vec<_>>();
        assert_eq!(ids, [4, 3, 2, 1]);
        let between = history(0x20_0000, &kept);
        assert_eq!(between.held, None);
        let nearest = between.nearest.as_ref().expect("the nearest");
        assert_eq!(nearest.below, Some((&kept[2], 0xf_fc00)));
        assert_eq!(nearest.above, Some((&kept[1], 0x20_0000)));
        assert_eq!(
            between.text(LOSS),
            "in the history of released ranges, none kept held it (nearest below: buffer \
             0x30000000002 of context 3 (Shared, length 0x400, at 0x100000), released at \
             1795560000.150000 s since 1970 (0.100000 s before the loss), the address 0xffc00 \
             bytes past its end; nearest above: heap 0x30000000003 of context 3 (Private, length \
             0x1000, at 0x400000), released at 1795560000.270000 s since 1970 (0.020000 s after \
             the loss), its first byte 0x200000 bytes past the address)"
        );
        // Below every range kept, and above every one.
        let low = history(0x1000, &kept).nearest.expect("the nearest");
        assert_eq!((low.below, low.above), (None, Some((&kept[2], 0xf_f000))));
        let high = history(0x100_0000, &kept).nearest.expect("the nearest");
        assert_eq!(
            (high.below, high.above),
            (Some((&kept[0], 0xb0_0000)), None)
        );
        // An empty history.
        assert_eq!(history(0x1000, &[]).text(LOSS), NONE_RELEASED);
    }

    /// The history keeps the last [`RELEASES_KEPT`] releases: each past them drops the oldest,
    /// counted, and the line says how many it keeps at most, holds and dropped.
    #[test]
    fn the_history_keeps_its_last_releases_and_counts_those_dropped() {
        let releases = Releases::default();
        let n = RELEASES_KEPT as u64 + 3;
        for id in 1..=n {
            let range = (id << 20, 0x1000);
            let released = release(ObjectKind::Buffer, 0x300_0000_0000 + id, 0, range);
            releases.push_at(before(released, (n - id) * MS));
        }
        let (kept, dropped) = releases.kept();
        assert_eq!((kept.len(), dropped), (RELEASES_KEPT, 3));
        let id = |release: &Release| release.id - 0x300_0000_0000;
        assert_eq!((id(&kept[0]), id(&kept[RELEASES_KEPT - 1])), (n, 4));
        // An address the oldest dropped held is in none kept.
        let dropped_held = history(1 << 20, &kept);
        assert_eq!(dropped_held.held, None);
        let above = dropped_held.nearest.and_then(|n| n.above);
        assert_eq!(above.map(|(r, gap)| (id(r), gap)), Some((4, 0x30_0000)));
        assert_eq!(
            kept_text(&kept, dropped),
            "The history of released ranges keeps the last 4096, and holds 4096 (3 older dropped \
             past them): 4096 buffers, 0 textures over a buffer, 0 heaps, 0 two-copy Managed \
             textures' CPU-copy buffers"
        );
    }

    /// On the GPU the test holds, a buffer and a heap a decoder's table named go into their
    /// device's history as their platform objects go, each by its guest id, whose bits from 40
    /// are its context, its storage mode and its GPU range; a buffer named the CPU-copy buffer of
    /// a `Managed` texture as that; a buffer no table named, the host's own, not at all.
    #[test]
    fn a_released_object_s_history_entry_carries_its_guest_id_and_context_on_vulkan() {
        use crate::core::platform::device::GpuDevice as _;
        use crate::core::platform::resource::Buffer as _;
        use crate::core::platform::values::{HeapDesc, OptionSet};
        use crate::core::raise::OnFault;

        let platform = super::super::RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let of = |length, options| {
            let made = device.new_buffer_of_length(on, length, options);
            made.expect("never a raise").expect("a buffer")
        };
        let range = |buffer: &RtObj<RtBuffer>| (buffer.gpu_address(), buffer.length());
        let from = wall_now();

        let private = of(0x1000, 0x20);
        name_buffer(&private, 0x300_0000_0007);
        let private_range = range(&private);
        drop(private);
        let copy = of(0x100, 0);
        name_managed_copy(&copy, 0x500_0000_0008);
        let copy_range = range(&copy);
        drop(copy);
        drop(of(0x100, 0));
        let desc = HeapDesc {
            size: 0x10000,
            heap_type: 1,
            options: vec![OptionSet::ResourceOptions(0x20)],
            pinned_gpu_address: None,
        };
        let heap = device.new_heap(&desc).expect("never a raise").heap;
        let heap = heap.expect("a Private placement heap");
        name_heap(&heap, 0x600_0000_0009);
        let heap_range = (heap.gpu_address(), 0x10000);
        drop(heap);

        let (kept, dropped) = device.releases().kept();
        assert_eq!(dropped, 0);
        let entry = |r: &Release| (r.kind, r.id, r.context(), r.storage, (r.gpu_address, r.len));
        assert_eq!(
            kept.iter().map(entry).collect::<Vec<_>>(),
            [
                (ObjectKind::Heap, 0x600_0000_0009, 6, 2, heap_range),
                (ObjectKind::ManagedCopy, 0x500_0000_0008, 5, 0, copy_range),
                (ObjectKind::Buffer, 0x300_0000_0007, 3, 2, private_range),
            ]
        );
        let now = wall_now();
        assert!(
            kept.iter().all(|r| (from..=now).contains(&r.at)),
            "{kept:?}"
        );
    }

    fn ids(of: &[u64]) -> Ids {
        let mut ids = Ids::default();
        for &id in of {
            ids.add(id);
        }
        ids
    }

    /// An encoder's names join its command buffer's list as it ends, once, or as it goes unended,
    /// in the order the encoders end; the commit takes them.
    #[test]
    fn an_encoder_s_names_join_its_command_buffer_s_as_it_ends_or_goes() {
        let encoded = Encoded::default();
        let mut blit = Naming::new(&encoded, EncoderNames::new(EncoderKind::Blit));
        blit.end();
        blit.end();
        assert!(blit.names().is_none());
        drop(blit);
        let mut compute = Naming::new(&encoded, EncoderNames::new(EncoderKind::Compute));
        let names = compute.names().expect("its names, until it ends");
        names.bound_buffers.add(0x7);
        drop(compute);
        let taken = encoded.take();
        let kinds = taken.iter().map(|e| e.kind).collect::<Vec<_>>();
        assert_eq!(kinds, [EncoderKind::Blit, EncoderKind::Compute]);
        assert_eq!(taken[1].bound_buffers, ids(&[0x7]));
        assert!(encoded.take().is_empty());
    }

    /// The command buffers committed and not complete at a loss are listed the oldest first, each
    /// by the guest's id and its context, or as the host's own, with its commit's time against the
    /// loss and its encoders' kinds in order; the oldest's line names its passes' attachments by
    /// texture and what its encoders declare and bind, each id once. One removed, completed, is
    /// not listed; with none, the line says so.
    #[test]
    fn the_command_buffers_in_flight_at_a_loss_are_listed_oldest_first() {
        let flights = Flights::default();
        let gpu = "GPU (10de:2204)";
        assert_eq!(
            flights.lines(gpu, LOSS),
            [
                "reims-vgpu: the rt platform's device on GPU (10de:2204), at its loss, had no \
                 command buffers committed and not complete\n"
            ]
        );
        let render = EncoderNames {
            attachments: vec![
                (Point::Color(0), 0x300_0000_0020),
                (Point::Color(1), 0),
                (Point::Depth, 0x300_0000_0021),
            ],
            heaps: ids(&[0x300_0000_0030]),
            bound_buffers: ids(&[0x300_0000_0040, 0, 0x300_0000_0040]),
            ..EncoderNames::new(EncoderKind::Render)
        };
        let compute = EncoderNames {
            declared_buffers: ids(&[0x300_0000_0041]),
            declared_textures: ids(&[0x300_0000_0022]),
            bound_buffers: ids(&[0x300_0000_0040, 0x300_0000_0042]),
            ..EncoderNames::new(EncoderKind::Compute)
        };
        let oldest = flights.add(Flight {
            cb: Some(0x300_0000_0010),
            committed: LOSS - 40 * MS,
            encoders: vec![
                render,
                compute,
                EncoderNames::new(EncoderKind::Compute),
                EncoderNames::new(EncoderKind::Blit),
            ],
        });
        let completed = flights.add(Flight {
            cb: Some(0x400_0000_0010),
            committed: LOSS - 30 * MS,
            encoders: Vec::new(),
        });
        flights.add(Flight {
            cb: None,
            committed: LOSS - 20 * MS,
            encoders: vec![EncoderNames::new(EncoderKind::Blit)],
        });
        flights.add(Flight {
            cb: Some(0x400_0000_0011),
            committed: LOSS - 10 * MS,
            encoders: Vec::new(),
        });
        assert!(oldest != 0 && completed != 0);
        flights.remove(completed);
        flights.remove(0);
        assert_eq!(flights.len(), 3);
        assert_eq!(
            flights.lines(gpu, LOSS),
            [
                "reims-vgpu: the rt platform's device on GPU (10de:2204), at its loss, had 3 \
                 command buffers committed and not complete, the oldest committed first: the \
                 guest's command buffer 0x30000000010 of context 3, committed at \
                 1795560000.210000 s since 1970 (0.040000 s before the loss), its encoders (4): \
                 render, 2 compute, blit; a command buffer of the host's own, committed at \
                 1795560000.230000 s since 1970 (0.020000 s before the loss), its encoders (1): \
                 blit; the guest's command buffer 0x40000000011 of context 4, committed at \
                 1795560000.240000 s since 1970 (0.010000 s before the loss), with no encoders\n",
                "reims-vgpu: the oldest command buffer committed and not complete at the loss of \
                 the rt platform's device on GPU (10de:2204), the guest's command buffer \
                 0x30000000010 of context 3: its render passes' attachments: encoder 1: colour 0 \
                 texture 0x30000000020, colour 1 a texture with no guest id, depth texture \
                 0x30000000021; the heaps its encoders declare (useHeap:): 0x30000000030; the \
                 buffers they declare (useResource:): 0x30000000041; the textures they declare \
                 (useResource:): 0x30000000022; the buffers they bind: 0x30000000040, \
                 0x30000000042, 1 names of objects with no guest id\n",
            ]
        );
        flights.remove(oldest);
        let lines = flights.lines(gpu, LOSS);
        assert!(
            lines[0].contains(", had 2 command buffers "),
            "{}",
            lines[0]
        );
        assert!(
            lines[1].ends_with(
                ", a command buffer of the host's own: no render passes; the heaps its encoders \
                 declare (useHeap:): none; the buffers they declare (useResource:): none; the \
                 textures they declare (useResource:): none; the buffers they bind: none\n"
            ),
            "{}",
            lines[1]
        );
    }

    /// The in-flight lines list the first [`FLIGHTS_LISTED`] command buffers, a command buffer's
    /// first [`KINDS_LISTED`] runs of encoders of one kind, the oldest's first [`PASSES_LISTED`]
    /// passes and, of each of its lists, the first [`IDS_KEPT`] ids, and count the rest.
    #[test]
    fn the_in_flight_lines_cap_their_lists_and_count_the_rest() {
        let flights = Flights::default();
        let kind = |at: usize| match at % 2 {
            0 => EncoderKind::Render,
            _ => EncoderKind::Compute,
        };
        let mut encoders = (0..KINDS_LISTED + 2)
            .map(|at| EncoderNames::new(kind(at)))
            .collect::<Vec<_>>();
        let many = (1..=IDS_KEPT as u64 + 3).collect::<Vec<_>>();
        encoders[0].bound_buffers = ids(&many);
        flights.add(Flight {
            cb: Some(0x100_0000_0001),
            committed: LOSS - 50 * MS,
            encoders,
        });
        for at in 0..FLIGHTS_LISTED as u64 + 1 {
            flights.add(Flight {
                cb: Some(0x100_0000_0002 + at),
                committed: LOSS - 40 * MS,
                encoders: Vec::new(),
            });
        }
        let lines = flights.lines("GPU (10de:2204)", LOSS);
        assert_eq!(lines.len(), 2);
        let runs = ["render", "compute"].repeat(KINDS_LISTED / 2).join(", ");
        let encoders = format!(
            "its encoders ({}): {runs}, and 2 more not listed;",
            KINDS_LISTED + 2
        );
        assert!(lines[0].contains(&encoders), "{}", lines[0]);
        assert_eq!(
            lines[0].matches("the guest's command buffer ").count(),
            FLIGHTS_LISTED
        );
        assert!(lines[0].contains(", had 18 command buffers committed and not complete, "));
        assert!(
            lines[0].ends_with("; and 2 more, not listed\n"),
            "{}",
            lines[0]
        );
        assert!(
            lines[1].contains("; encoder 31: no attachments; and 1 more passes not listed; "),
            "{}",
            lines[1]
        );
        assert!(!lines[1].contains("encoder 33"), "{}", lines[1]);
        let bound = (1..=IDS_KEPT)
            .map(|id| format!("{id:#x}"))
            .collect::<Vec<_>>();
        assert!(
            lines[1].ends_with(&format!(
                "; the buffers they bind: {}, 3 more names past the first 32 ids\n",
                bound.join(", ")
            )),
            "{}",
            lines[1]
        );
    }

    /// A decoder's table of one buffer, walked at the loss as a graphics device's is
    /// ([`buffer_parts`]).
    struct OneBuffer {
        id: u64,
        buffer: RtObj<RtBuffer>,
    }

    impl LiveObjects for OneBuffer {
        fn walk(&self) -> Walked {
            let mut walked = Walked {
                contexts: 1,
                ..Walked::default()
            };
            walked.add(buffer_parts(3, self.id, false, &self.buffer), false);
            walked
        }

        fn live(&self) -> bool {
            true
        }
    }

    /// On the GPU the test holds, an access in the import a live no-copy buffer lies in is placed
    /// in the import's block at the loss, named with the kind of import it is, after another
    /// import was made and released, its record gone (`super::super::diag::import_released`).
    #[test]
    fn a_live_import_s_fault_is_placed_in_its_block_past_a_released_one_on_vulkan() {
        use std::alloc::Layout;
        use std::sync::atomic::Ordering::{Acquire, Release};

        use super::super::diag::import_range;
        use crate::core::host::HostPtr;
        use crate::core::platform::device::{GpuDevice as _, NoCopyDeallocator};
        use crate::core::platform::host_range::HostRange;
        use crate::core::raise::OnFault;

        let platform = super::super::RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let layout = Layout::from_size_align(0x20000, 0x10000).expect("a layout");
        // SAFETY: the layout's size is not 0.
        let memory = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!memory.is_null());
        // A no-copy buffer of a page `offset` bytes into the memory, `freed` once its deallocator
        // ran.
        let over = |offset: usize, freed: &Arc<AtomicBool>| {
            let done = Arc::clone(freed);
            let deallocator: NoCopyDeallocator = Box::new(move || done.store(true, Release));
            let at = HostPtr::from_ptr(memory.wrapping_add(offset));
            // SAFETY: the memory stays allocated until both deallocators ran (below).
            let bytes = unsafe { HostRange::new(at, 0x1000) };
            let made = device.new_buffer_over_host_memory(on, bytes, 0, Some(deallocator));
            made.expect("never a raise").expect("a no-copy buffer")
        };
        let block_of = |buffer: &RtObj<RtBuffer>| buffer.made_rt().expect("made by rt").block();
        let waits_for = |done: &dyn Fn() -> bool| {
            for _ in 0..1000 {
                if done() {
                    return true;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            done()
        };

        // One made and released: its record goes.
        let gone_freed = Arc::new(AtomicBool::new(false));
        let gone = over(0x1_0000, &gone_freed);
        let gone_block = block_of(&gone);
        assert!(
            import_range(gone_block).is_some(),
            "no record of the live import"
        );
        drop(gone);
        assert!(
            waits_for(&|| gone_freed.load(Acquire)),
            "the deallocator never ran"
        );
        assert!(
            import_range(gone_block).is_none(),
            "the released import's record stayed"
        );

        // The live one's, at the loss.
        let freed = Arc::new(AtomicBool::new(false));
        let live = over(0, &freed);
        let block = block_of(&live);
        let import = import_range(block).expect("the live import's record");
        let kind = import_kind(live.device_known(), block).expect("the kind of import it is");
        let id = 0x300_0000_0007;
        let watch = FaultWatch::default();
        watch.watch(Arc::new(OneBuffer { id, buffer: live }));
        let fault = rt::DeviceFault {
            description: "a fault".to_owned(),
            accesses: vec![rt::FaultAddress {
                kind: Kind::READ_INVALID,
                address: import.gpu_address,
                precision: 0x10,
            }],
            instructions: Vec::new(),
            vendor: Vec::new(),
            places: Vec::new(),
        };
        let lines = watch.lines("GPU (10de:2204)", Some(&fault), LOSS);
        let named = MappedBlock {
            what: format!("buffer {id:#x}'s block {}", block.get()),
            class: import.class,
            import: Some(kind),
            gpu_address: import.gpu_address,
            len: import.len,
        };
        let placed = format!("in mapped block {} at offset 0x0", block_text(&named));
        let access = lines.iter().find(|l| l.contains("READ_INVALID"));
        let access = access.expect("the access's line");
        assert!(access.contains(&placed), "{placed:?} not in {access}");
        drop(watch);
        assert!(
            waits_for(&|| freed.load(Acquire)),
            "the deallocator never ran"
        );
        // SAFETY: both deallocators ran: rt holds no import of the memory.
        unsafe { std::alloc::dealloc(memory, layout) };
    }

    /// Where rt read no report of the fault, its one line says so.
    #[test]
    fn a_loss_with_no_fault_report_says_so() {
        let lines = FaultWatch::default().lines("GPU (1002:744c)", None, LOSS);
        assert_eq!(
            lines,
            [
                "reims-vgpu: the rt platform's device on GPU (1002:744c) faulted: rt read no \
                 report of the fault (the device has no VK_EXT_device_fault, the read failed, or \
                 a window's swapchain reported the loss)\n"
            ]
        );
    }
}
