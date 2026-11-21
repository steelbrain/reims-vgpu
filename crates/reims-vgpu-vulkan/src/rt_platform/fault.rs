//! A lost device's fault, named and placed. rt reads the driver's report of the fault
//! (`VK_EXT_device_fault`) once, at its device's first loss of its own, and keeps it with the loss
//! (rt's `DeviceLoss::fault`), placing none of its addresses. After the loss's line
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
//!   address.
//! - **Each instruction pointer**, placed as an access is, with the note that none of the ranges
//!   the host sees holds shader code: a pipeline's code lies in the driver's own memory, of which
//!   the host sees nothing, so the placement says only what lies around the address.
//! - **What the placement saw**: the live objects of each kind, the decoders and contexts whose
//!   tables were walked and those not walked, the live textures with no GPU range the platform
//!   knows, the mapped blocks from each source, and what the host does not see.
//!
//! What the host sees at the loss:
//!
//! 1. The live objects of the decoders' tables whose GPU range is known, walked at the loss
//!    ([`LiveObjects`], `reims_vgpu`'s `graphics::fault_walk`): buffers, heaps, textures over a
//!    buffer, and the `Shared` buffers over two-copy `Managed` textures' CPU copies, each removed
//!    entry kept for an open view among them. A deferred buffer rt has not made yet has no range.
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
//! with memory of their own; exports and surfaces (D3D11 planes among them, on Windows); and an
//! import by the pages' pointer that no live object or known block lies in (one rt still holds for
//! work in flight after its buffer went).
//!
//! **What is kept before the loss.** The tables are walked at the loss, with no record of them
//! kept before it. The rest is kept as it changes, one entry a range: rt's pool allocations'
//! ranges, which [`Allocations`] keeps as rt's `BlockMappings` tells it of them (it maps nothing
//! and refuses nothing, so rt's allocations and their mappings are as without it), and each
//! import's range, beside the record of where it was imported (`super::diag`'s `imported`).
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

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

use metal2vulkan_runtime as rt;

use crate::core::num::rem0;
use crate::core::platform::resource::{GpuResource as _, Heap as _};
use crate::core::sync::Mutex;

use super::RtObj;
use super::buffer::{KnownBlocks, RtBuffer};
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

/// An object as the line names it.
fn object_text(object: &LiveObject) -> String {
    let kind = match object.kind {
        ObjectKind::Buffer => "buffer",
        ObjectKind::Texture => "texture",
        ObjectKind::Heap => "heap",
        ObjectKind::ManagedCopy => "the CPU-copy buffer of Managed texture",
    };
    let gone = if object.gone {
        ", removed and kept for an open view"
    } else {
        ""
    };
    format!(
        "{kind} {:#x} of context {} ({}, length {:#x}, at {:#x}{gone})",
        object.id,
        object.context,
        storage_name(object.storage),
        object.len,
        object.gpu_address
    )
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
    /// The blocks of live objects: a heap's block, the import a buffer lies in.
    pub blocks: Vec<MappedBlock>,
    /// How many contexts' tables were walked.
    pub contexts: usize,
    /// The contexts whose table's lock was not had in time ([`BOUND`]).
    pub unwalked: Vec<u32>,
    /// Whether the decoder's map of its contexts was not had in time, so no table was walked.
    pub contexts_busy: bool,
    /// The live textures with no GPU range the platform knows.
    pub unranged_textures: u64,
}

/// What an object of the tables adds to the walk ([`buffer_parts`], [`texture_parts`],
/// [`heap_parts`], [`managed_copy_parts`]).
pub struct Parts {
    /// The object, where its GPU range is known.
    pub object: Option<LiveObject>,
    /// Its block, where it is one the host sees no other way: a heap's, or the import a buffer
    /// lies in.
    pub block: Option<MappedBlock>,
}

impl Walked {
    /// Adds what an object of the tables adds, `texture` where it is a texture.
    pub fn add(&mut self, parts: Parts, texture: bool) {
        match parts.object {
            Some(object) => self.objects.push(object),
            None if texture => self.unranged_textures += 1,
            None => {}
        }
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
/// one (its block, which the host sees no other way). A deferred buffer rt has not made yet adds
/// nothing.
pub fn buffer_parts(context: u32, id: u64, gone: bool, buffer: &RtObj<RtBuffer>) -> Parts {
    let Some(made) = buffer.made_rt() else {
        return Parts {
            object: None,
            block: None,
        };
    };
    let object = LiveObject {
        context,
        id,
        kind: ObjectKind::Buffer,
        storage: buffer.storage_mode(),
        gpu_address: made.gpu_address(),
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
    parts
}

/// What a live texture of the tables adds: its range where it lies over a buffer (its bytes in
/// the buffer); none for any other.
pub fn texture_parts(context: u32, id: u64, gone: bool, texture: &RtObj<RtTexture>) -> Parts {
    let range = texture.over().and_then(|(buffer, bytes)| {
        let buffer = buffer.upgrade()?;
        let made = buffer.made_rt()?;
        let len = bytes.end.checked_sub(bytes.start)?;
        Some((made.gpu_address().checked_add(bytes.start)?, len))
    });
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
        block: None,
    }
}

/// What a live heap of the tables adds: its range, and its block.
pub fn heap_parts(context: u32, id: u64, gone: bool, heap: &RtObj<RtHeap>) -> Parts {
    let (block, offset, size) = heap.fault_span();
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
        block: Some(MappedBlock {
            what: format!(
                "heap {id:#x}'s block {} of allocation {} (the heap at offset {offset:#x})",
                block.id.get(),
                block.allocation.get()
            ),
            class: block.class,
            import: import_kind(heap.device_known(), block.id),
            gpu_address: block.gpu_address,
            len: block.len,
        }),
    }
}

/// rt's pool allocations still mapped, by number: their storage mode, first byte's GPU address
/// and length. As rt's `BlockMappings`, the device's (`DeviceOptions::block_mappings`), it maps
/// nothing and refuses nothing: it notes each allocation rt makes, which rt tells it of before it
/// hands out a block of it, and forgets each, which rt tells it of before it frees it.
#[derive(Default)]
pub struct Allocations(Mutex<BTreeMap<u32, (rt::MemoryClass, u64, u64)>>);

impl rt::BlockMappings for Allocations {
    fn map(&self, allocation: &rt::AllocationInfo) -> bool {
        let at = (allocation.class, allocation.gpu_address, allocation.len);
        self.0.lock().insert(allocation.id.get(), at);
        true
    }

    fn unmap(&self, allocation: rt::AllocationId) {
        self.0.lock().remove(&allocation.get());
    }
}

/// A device's watch for its loss's fault: rt's pool allocations, the blocks the device knows, and
/// the decoders' tables to walk at the loss. A handle: its clones are the same watch.
#[derive(Clone, Default)]
pub struct FaultWatch {
    /// rt's pool allocations still mapped.
    allocations: Arc<Allocations>,
    /// The device's known blocks, its set's handle.
    known: KnownBlocks,
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
            tables: Arc::default(),
        }
    }

    /// Walks `tables` too at the loss, and no longer those whose decoder went.
    pub fn watch(&self, tables: Arc<dyn LiveObjects>) {
        let mut all = self.tables.lock();
        all.retain(|t| t.live());
        all.push(tables);
    }

    /// The fault's lines for the device on `gpu`, its report `fault` (rt's `DeviceLoss::fault`),
    /// each with its own `\n` (the module's docs): the report, each instruction pointer and each
    /// access placed, and what the placement saw; one line where rt read no report.
    pub fn lines(&self, gpu: &str, fault: Option<&rt::DeviceFault>) -> Vec<String> {
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
        let (objects, blocks, seen) = self.seen();
        let placed = |at: &rt::FaultAddress| {
            let placement = place(at.address, at.precision, &objects, &blocks);
            let (start, end) = placement.range;
            let text = placement.text();
            format!("{at:?}, the bytes [{start:#x}, {end:#x}): {text}")
        };
        for at in &fault.instructions {
            let at = placed(at);
            lines.push(format!("{head}instruction {at}; {NO_CODE_SEEN}\n"));
        }
        for at in &fault.accesses {
            lines.push(format!("{head}access {}\n", placed(at)));
        }
        lines.push(format!("{head}{seen}\n"));
        lines
    }

    /// What the host sees now: the live objects of the tables' walk, every mapped block it sees,
    /// and what it saw and did not, as the line says it.
    fn seen(&self) -> (Vec<LiveObject>, Vec<MappedBlock>, String) {
        let tables = (self.tables.lock().iter())
            .filter(|t| t.live())
            .cloned()
            .collect::<Vec<_>>();
        let (mut objects, mut of_objects) = (Vec::new(), Vec::new());
        let (mut contexts, mut unwalked, mut busy, mut unranged) = (0, Vec::new(), 0, 0);
        for table in &tables {
            let mut walked = table.walk();
            objects.append(&mut walked.objects);
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
        for info in known.unwrap_or_default() {
            ids.insert(info.id);
            blocks.push(MappedBlock {
                what: format!(
                    "known block {} of allocation {}",
                    info.id.get(),
                    info.allocation.get()
                ),
                class: info.class,
                import: import_kind(&self.known, info.id),
                gpu_address: info.gpu_address,
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
             contexts' tables, walked at the loss{not_walked}, beside {unranged} live textures \
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
            blocks.len(),
            pool.len(),
        );
        (objects, blocks, seen)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ash::vk::DeviceFaultAddressTypeEXT as Kind;

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
                        block: None,
                    },
                    texture,
                );
            }
            walked.add(
                Parts {
                    object: None,
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
        };
        let lines = watch.lines("GPU (10de:2204)", Some(&fault));
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
                 0x7efe1ebef0 bytes past its end; nearest above: none); {NO_CODE_SEEN}\n"
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
                 offset 0x810\n"
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
                 0x5300000000), its first byte 0xe5100000 bytes past the address)\n"
            )
        );
        assert_eq!(
            lines[4],
            format!(
                "{head}placed among 2 live objects with a GPU range (1 buffers, 1 textures over a \
                 buffer, 0 heaps, 0 two-copy Managed textures' CPU-copy buffers) of 1 decoders' 2 \
                 contexts' tables, walked at the loss; the tables of contexts [5] were not had in \
                 time and not walked, beside 1 live textures with no GPU range the platform knows \
                 (images of rt's own: one in a heap lies in its heap's range); and among 3 mapped \
                 blocks the host sees: 2 of rt's pool allocations, 0 known blocks, 0 more live \
                 dma-buf imports, 1 blocks of live objects (heaps' blocks, buffers' imports). Not \
                 seen: rt's parameter ring, zero source and staging region, images rt makes with \
                 memory of their own, exports and surfaces, and an import by the pages' pointer \
                 that no live object or known block lies in\n"
            )
        );
        // An allocation rt unmapped is no longer among the blocks.
        allocations.0.lock().remove(&8);
        let lines = watch.lines("GPU (10de:2204)", Some(&fault));
        assert!(lines[3].ends_with(
            "nearest above: buffer 0x100000003's block 44 (Shared, an import of host pages by \
             their pointer, length 0x10000, at 0x6000000000), its first byte 0xde5100000 bytes \
             past the address)\n"
        ));
    }

    /// Where rt read no report of the fault, its one line says so.
    #[test]
    fn a_loss_with_no_fault_report_says_so() {
        let lines = FaultWatch::default().lines("GPU (1002:744c)", None);
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
