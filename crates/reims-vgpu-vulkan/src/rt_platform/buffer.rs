//! The rt platform's buffers ([`RtBuffer`]): rt's buffers, each a range of a block, made by the
//! device's `newBufferWithLength:options:` ([`of_length`]), `newBufferWithDescriptor:`
//! ([`of_desc`]) and `newBufferWithBytesNoCopy:length:options:deallocator:`
//! ([`over_host_memory`]), and by heaps (`super::heap`), and the textures over their bytes
//! (`super::texture::make_over`).
//!
//! **Blocks of its own.** A buffer the device makes takes blocks of its own (rt's `take_block`):
//! a Shared or Private one of its storage mode, and for a Managed buffer a Shared one of its
//! bytes and, on a GPU with memory of its own (rt's `has_unified_memory` false), a Private one of
//! its device copy. A take is 64 KiB at least, whatever the buffer's length: placing small buffers
//! in shared blocks is the guest's, by address ("A pinned address"). Its bytes are zeros, as
//! Metal's are: rt zeroes a Shared buffer's through the mapping and a Private one's by a GPU fill
//! ahead of the next command buffer, the platform zeroes a Managed buffer's own bytes, and rt its
//! device copy. The buffer holds its blocks; the platform's handles of them go as it is made.
//!
//! **A pinned address.** The guest picks a buffer's GPU address (`setPinnedGPUAddress:`), and rt
//! places no buffer at a chosen address: a buffer's address is Vulkan's device address. A pinned
//! Shared or Private buffer whose range lies in a block the device knows of its class
//! ([`KnownBlocks`], which the graphics lane fills with the blocks the guest sub-allocates) is that
//! range of the block, at the pinned address. Its bytes are the block's as they are, Metal's zeros
//! the block's owner's to give (at its take) and the guest's (in the ranges it takes again), but a
//! Private buffer's, which the guest cannot reach, zeroed (rt's GPU fill). A no-copy buffer is such
//! a range only where its bytes are the block's own mapping there; its deallocator then runs as the
//! buffer goes, the block's memory held by rt for the work that still uses it. A no-copy Managed
//! buffer on a GPU with memory of its own, whose address is its device copy's, keeps that copy at
//! the pin where a known Private block holds it, its bytes where they lie (below); on a GPU that
//! shares the CPU's memory it keeps no copy, so a pin lies at its bytes, as a Shared one's does.
//! Any other pinned buffer is made as an unpinned one, at rt's address, which the guest observes as
//! a `gpuAddress` other than the one it pinned: counted ([`UNPINNED_CODE`]), the process's first
//! with a line. A no-copy buffer's bytes that a known Shared block maps are a range of that block,
//! with no import of their own: a Managed buffer's in a slab of the pool or in a pages block's
//! view, and those of a buffer whose pin no known block keeps, which then lies at the block's
//! address plus their offset.
//!
//! **Imports.** A no-copy buffer whose bytes no known block maps is a range of rt's import of the
//! host's pages that hold its bytes (rt's `import_host_memory`): the bytes rounded out to the
//! host's pages, which are mapped as the bytes are, the buffer at the bytes' offset in them;
//! Shared, or Managed over them, rt keeping the device copy where the GPU has memory of its own,
//! whose bytes past `zero_from` start as zeros. Its deallocator runs once rt freed the import,
//! after the buffer and every command buffer that used it are gone, as Metal runs it once the
//! buffer is freed; one of a buffer never made never runs, and its captures are kept. Where the
//! GPU keeps a device copy, a Managed one's bytes that its making names for that copy (all but
//! those past `zero_from`) hold the import too, until the next commit of a command buffer with
//! commands has copied them and its copy has completed (rt's `didModifyRange:` capture), so its
//! deallocator runs no sooner, though the buffer went before. Each import
//! is one Vulkan allocation; each is counted ([`IMPORTS_CODE`]), and the most live at once as that
//! code's highest value ([`IMPORTS_LIVE_MOST_CODE`]); each again, with the most live at once,
//! under the kind of the make that asked for it on its thread (`crate::answer::making_as`), or
//! [`IMPORTS_UNNAMED`] where no make named one, so the kinds' counts sum to the total. Option bit
//! 24 (read-only) is taken and answered without, as Metal does, but the GPU's writes reach the
//! bytes, where Metal drops them: counted ([`READ_ONLY_CODE`]), the process's first with a line.
//!
//! **Answers.** A buffer's resource options are the asked ones without bits 20 and 24, Metal's
//! default hazard tracking resolved to tracked, as the Vulkan platform's runtime answers them. A
//! Private buffer's `contents` is NULL. `didModifyRange:` of a Managed buffer is rt's, which copies
//! the range into the device copy at the next commit; of any other it does nothing, as Metal's.
//!
//! **Refusals.** What Metal refuses is nil, counted as Metal's refusal (`rt-refused-metal`): a
//! length of 0 or past rt's longest buffer, an option value Metal has none of, a Memoryless
//! buffer, option bit 24 on a buffer the device makes, a Private no-copy buffer, a deallocator of a
//! buffer that is not no-copy. What rt refuses is nil, counted under rt's kind. A Private buffer of
//! bytes, which a copy on the GPU fills, and a tiled texture over a buffer are unbuilt
//! ([`Unbuilt::Buffer`]).

use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use metal2vulkan_rt as rt;

use crate::core::host::HostPtr;
use crate::core::num::{ToU64 as _, ToUsize as _};
use crate::core::platform::Handle as _;
use crate::core::platform::device::NoCopyDeallocator;
use crate::core::platform::host_range::HostRange;
use crate::core::platform::resource::{Buffer, GpuResource};
use crate::core::platform::types::NsRange;
use crate::core::platform::values::{BufferDeallocator, BufferDesc, SurfacePlane};
use crate::core::raise::{OnFault, RaiseResult};
use crate::core::sync::Mutex;
use crate::gpu::device::descriptor_options;

use super::device::RtDevice;
use super::diag::{Made, Shape};
use super::texture::{self, RtTexture, refused_metal};
use super::unbuilt::{Unbuilt, refuse_create, rt_refused, rt_refused_create};
use super::{RtObj, RtPlatform, TextureFields};

/// The log target of the buffers' lines.
const TARGET: &str = "reims_vgpu_vulkan::rt_platform";

/// The code of a buffer or heap the guest pinned at an address in no block the device knows, made
/// at rt's address (the module's docs), counted per buffer or heap.
pub const UNPINNED_CODE: &str = "rt-gpu-address-unpinned";

/// The code of an import of the host's memory (a no-copy buffer's or a heap's), counted per import.
pub const IMPORTS_CODE: &str = "rt-imports";

/// The code whose count is the most imports of the host's memory live at once.
pub const IMPORTS_LIVE_MOST_CODE: &str = "rt-imports-live-most";

/// The codes of the imports of the host's memory made where no make named its kind
/// (`crate::answer::making_as`), counted per import, and the most live at once.
pub const IMPORTS_UNNAMED: crate::answer::MadeCodes = crate::answer::MadeCodes {
    made: "rt-imports-unnamed",
    live_most: "rt-imports-unnamed-live-most",
};

/// The code of a no-copy buffer or a heap of option bit 24, which is taken and answered without:
/// the GPU's writes reach a no-copy buffer's bytes, where Metal drops them. Counted per buffer or
/// heap.
pub const READ_ONLY_CODE: &str = "rt-no-copy-read-only";

/// A buffer descriptor's `resourceOptions` before any setter: `Shared`.
const BUFFER_DESCRIPTOR_OPTIONS: u64 = 0;

/// The resource options' CPU cache, storage and hazard tracking modes: bits 0 to 9.
pub(super) const FIELDS: u64 = 0x3ff;

/// Resource option bit 20, which Metal takes and answers without (`ResourceOptions::BIT_20` of
/// metal2vulkan-runtime, measured).
pub(super) const BIT_20: u64 = 1 << 20;

/// Resource option bit 24, read-only, which Metal takes for a no-copy buffer and answers without,
/// and at which a buffer the device makes ends Metal's process (`ResourceOptions::PRIVATE_BIT` of
/// metal2vulkan-runtime, measured).
pub(super) const READ_ONLY: u64 = 1 << 24;

/// `MTLStorageModeManaged`, in the resource options' bits 4 to 7.
const MANAGED: u64 = 1;

/// `id<MTLBuffer>` on metal2vulkan-rt: rt's buffer and what the platform answers of it.
pub struct RtBuffer {
    /// rt's buffer.
    rt: rt::Buffer,
    /// The device that made it.
    device: RtObj<RtDevice>,
    /// The resource options it answers (the module's docs).
    options: u64,
    /// Its purgeable state, as Metal keeps it (`super::texture::set_purgeable`).
    purgeable: AtomicU64,
    /// Whether a heap placed it (`super::heap`), which keeps no purgeable state of its own.
    in_heap: bool,
    /// A no-copy buffer's deallocator that runs as the buffer goes: one over a known block's
    /// mapping (the module's docs, "A pinned address").
    deallocator: Option<Slot>,
}

impl RtBuffer {
    /// rt's buffer.
    pub fn rt(&self) -> &rt::Buffer {
        &self.rt
    }
}

impl Drop for RtBuffer {
    fn drop(&mut self) {
        if let Some(slot) = self.deallocator.take() {
            run(&slot);
        }
    }
}

/// The blocks the device knows by their GPU address, which a pinned buffer or heap inside one is a
/// range of (the module's docs, "A pinned address"): the graphics lane's, which knows a block as it
/// hands its range to the guest and forgets it as it takes it back, just before it gives the block
/// back to rt. Each held block is a handle of it, so it lives while known. A handle of the device's
/// set: its clones are the same set (`RtDevice::known_blocks`).
#[derive(Default, Clone)]
pub struct KnownBlocks(Arc<Mutex<BTreeMap<u64, rt::Block>>>);

impl KnownBlocks {
    /// Knows `block` by its GPU address, in place of a block known there before.
    pub fn know(&self, block: rt::Block) {
        self.0.lock().insert(block.gpu_address, block);
    }

    /// Forgets the block `id`, giving back the handle it held; `None` where none is known.
    pub fn forget(&self, id: rt::BlockId) -> Option<rt::Block> {
        let mut known = self.0.lock();
        let at = known
            .iter()
            .find(|(_, block)| block.id == id)
            .map(|(&at, _)| at)?;
        known.remove(&at)
    }

    /// The known block that holds the `length` bytes at GPU address `address`, and their offset
    /// in it.
    pub fn at(&self, address: u64, length: u64) -> Option<(rt::Block, u64)> {
        let known = self.0.lock();
        let (&start, block) = known.range(..=address).next_back()?;
        let offset = address - start;
        let inside = offset
            .checked_add(length)
            .is_some_and(|end| end <= block.len);
        inside.then(|| (block.clone(), offset))
    }

    /// The known Shared block whose mapping holds the `length` bytes at `first`, and their offset
    /// in it.
    pub fn mapping(&self, first: HostPtr, length: u64) -> Option<(rt::Block, u64)> {
        let first = first.addr().to_u64();
        let known = self.0.lock();
        known.values().find_map(|block| {
            let start = block.contents?.as_ptr().addr().to_u64();
            let offset = first.checked_sub(start)?;
            let inside = offset
                .checked_add(length)
                .is_some_and(|end| end <= block.len);
            let shared = block.class == rt::MemoryClass::Shared;
            (shared && inside).then(|| (block.clone(), offset))
        })
    }
}

/// A buffer's storage mode, as its options give it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Storage {
    /// `MTLStorageModeShared`.
    Shared,
    /// `MTLStorageModeManaged`.
    Managed,
    /// `MTLStorageModePrivate`.
    Private,
}

/// What a buffer's raw `MTLResourceOptions` make of it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Options {
    /// Its storage mode.
    storage: Storage,
    /// The options it answers (the module's docs).
    answered: u64,
    /// Whether option bit 24 is set.
    read_only: bool,
}

/// The options `raw` of a buffer, no-copy where `no_copy`; `Err` says what Metal refuses of them
/// (the module's docs, "Refusals").
fn options(raw: u64, no_copy: bool) -> Result<Options, String> {
    if raw & !(FIELDS | BIT_20 | READ_ONLY) != 0 {
        return Err(format!("resource options {raw:#x}"));
    }
    let (cache, hazard) = (raw & 0xf, (raw >> 8) & 0x3);
    if cache > 1 {
        return Err(format!("CPU cache mode {cache}"));
    }
    if hazard > 2 {
        return Err(format!("hazard tracking mode {hazard}"));
    }
    let storage = match ((raw >> 4) & 0xf, no_copy) {
        (0, _) => Storage::Shared,
        (1, _) => Storage::Managed,
        (2, false) => Storage::Private,
        (2, true) => return Err("a Private buffer over the host's memory".to_owned()),
        (mode, _) => return Err(format!("storage mode {mode} of a buffer")),
    };
    let read_only = raw & READ_ONLY != 0;
    if read_only && !no_copy {
        return Err("option bit 24 of a buffer the device makes".to_owned());
    }
    Ok(Options {
        storage,
        answered: texture::tracked(raw & !(BIT_20 | READ_ONLY)),
        read_only,
    })
}

/// `newBufferWithLength:options:` on `device`, for `sel`: a buffer of zeros in blocks of its own
/// (the module's docs). Nil where Metal or rt refuses it, counted.
pub fn of_length(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    length: u64,
    raw: u64,
) -> Option<RtObj<RtBuffer>> {
    match options(raw, false) {
        Ok(o) => made(device, sel, length, o, None, None),
        Err(why) => refused_metal(sel, &why),
    }
}

/// `newBufferWithDescriptor:` of `desc` on `device`: a no-copy one as [`over_host_memory`]'s, its
/// deallocator given the bytes' address and the length; another as [`of_length`]'s, at the
/// pinned address where it lies in a known block, and holding a copy of the contents where given.
/// Nil where Metal or rt refuses it, counted, the deallocator never run.
pub fn of_desc(device: &RtObj<RtDevice>, desc: BufferDesc<'_>) -> Option<RtObj<RtBuffer>> {
    const SEL: &str = "newBufferWithDescriptor:";
    let raw = descriptor_options(BUFFER_DESCRIPTOR_OPTIONS, &desc.options);
    let BufferDesc {
        length,
        pinned_gpu_address: pin,
        contents,
        no_copy,
        deallocator,
        zero_from,
        ..
    } = desc;
    match (contents, no_copy, deallocator) {
        (Some(bytes), true, deallocator) => {
            let at = bytes.ptr();
            let release = deallocator.map(|d| with_bytes(d, at, length));
            if bytes.len() < length {
                std::mem::forget(release);
                let why = format!("contents of {} bytes for {length}", bytes.len());
                return refused_metal(SEL, &why);
            }
            over_host_memory(device, SEL, bytes, length, raw, zero_from, pin, release)
        }
        (None, true, deallocator) => {
            std::mem::forget(deallocator);
            refused_metal(SEL, &"a no-copy buffer of no contents")
        }
        (_, false, Some(deallocator)) => {
            std::mem::forget(deallocator);
            refused_metal(SEL, &"a deallocator of a buffer that is not no-copy")
        }
        (contents, false, None) => {
            let o = match options(raw, false) {
                Ok(o) => o,
                Err(why) => return refused_metal(SEL, &why),
            };
            let Some(bytes) = contents else {
                return made(device, SEL, length, o, pin, zero_from);
            };
            if o.storage == Storage::Private {
                refuse_create(Unbuilt::Buffer, SEL);
                return None;
            }
            if bytes.len() != length {
                let why = format!("contents of {} bytes for {length}", bytes.len());
                return refused_metal(SEL, &why);
            }
            let buffer = made(device, SEL, length, o, pin, None)?;
            copy_in(SEL, &buffer, bytes);
            Some(buffer)
        }
    }
}

/// `deallocator`, which runs with the bytes' address `at` and the buffer's `length`.
fn with_bytes(deallocator: BufferDeallocator, at: HostPtr, length: u64) -> NoCopyDeallocator {
    Box::new(move || deallocator(at, length))
}

/// Copies `bytes`, guest memory, into `buffer`'s contents, as long as its length, and names them
/// to a Managed buffer's device copy. They are read once, through the pointer; a range at or near
/// NULL dies, as Metal's read of it faults (`crate::gpu::resource::first_bytes`).
fn copy_in(sel: &'static str, buffer: &RtObj<RtBuffer>, bytes: HostRange<'_>) {
    let (first, length) = (bytes.ptr().addr(), bytes.len().to_usize());
    if length == 0 {
        return;
    }
    if first < crate::core::host::ZERO_REGION_END || first.checked_add(length).is_none() {
        crate::answer::metal_reads_near_null(sel, &format!("{length} bytes at {first:#x}"));
    }
    let Some(contents) = buffer.rt.contents() else {
        return;
    };
    // SAFETY: `HostRange::new`'s contract keeps the `length` bytes at `first` mapped and readable
    // for the call; the buffer's contents are `length` bytes of a block the platform took for it
    // alone, mapped for the block's life, so the two do not overlap. The copy is made through raw
    // pointers alone: no reference covers the guest's bytes, which may change under it.
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.ptr().as_ptr::<u8>(), contents.as_ptr(), length);
    }
    if buffer.storage_mode() == MANAGED
        && let Err(error) = buffer.rt.did_modify_range(0, bytes.len())
    {
        rt_refused(sel, &error);
    }
}

/// A buffer the device makes, of `length` bytes and options `o`, for `sel`: at the guest's `pin`
/// where it lies in a known block of the buffer's class, zeroed from `zero_from` alone, a Private
/// one from its first byte; else in blocks of its own, zeroed, a pin counted as not kept (the
/// module's docs). Nil where Metal or rt refuses it, counted.
fn made(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    length: u64,
    o: Options,
    pin: Option<u64>,
    zero_from: Option<u64>,
) -> Option<RtObj<RtBuffer>> {
    if length == 0 || length > rt::MAX_BUFFER_LENGTH {
        return refused_metal(sel, &format!("a buffer of {length} bytes"));
    }
    if let Some(pin) = pin {
        let zero_from = match o.storage {
            Storage::Private => Some(0),
            Storage::Shared | Storage::Managed => zero_from,
        };
        if let Some(made) = in_known(device, sel, length, o, pin, zero_from) {
            return made;
        }
        unpinned(sel, pin);
    }
    let made = match o.storage {
        Storage::Shared => own(device, sel, rt::MemoryClass::Shared, length),
        Storage::Private => own(device, sel, rt::MemoryClass::Private, length),
        Storage::Managed => own_managed(device, sel, length),
    }?;
    Some(buffer(device, made, o.answered))
}

/// The buffer's handle, of rt's buffer `rt` answering `options`.
fn buffer(device: &RtObj<RtDevice>, rt: rt::Buffer, options: u64) -> RtObj<RtBuffer> {
    buffer_of(device, rt, options, false, None)
}

/// The buffer's handle, of rt's buffer `rt` answering `options`, placed by a heap where `in_heap`,
/// running `deallocator` as it goes where given.
pub(super) fn buffer_of(
    device: &RtObj<RtDevice>,
    rt: rt::Buffer,
    options: u64,
    in_heap: bool,
    deallocator: Option<Slot>,
) -> RtObj<RtBuffer> {
    RtObj::new(RtBuffer {
        rt,
        device: device.clone_ref(),
        options,
        purgeable: texture::purgeable_state(),
        in_heap,
        deallocator,
    })
}

/// The known block of `class` that holds the `length` bytes at GPU address `pin`, and their
/// offset in it; where `bytes` is given, only one whose mapping holds them at that offset.
pub(super) fn known_at(
    device: &RtObj<RtDevice>,
    pin: u64,
    length: u64,
    class: rt::MemoryClass,
    bytes: Option<HostPtr>,
) -> Option<(rt::Block, u64)> {
    let (block, offset) = device.known_blocks().at(pin, length)?;
    let mapped = |at: HostPtr| {
        let first = block.contents.map(|c| c.as_ptr().addr().to_u64());
        first.and_then(|first| first.checked_add(offset)) == Some(at.addr().to_u64())
    };
    let fits = block.class == class && bytes.is_none_or(mapped);
    fits.then_some((block, offset))
}

/// The buffer at `pin` of `length` bytes and options `o`, where the range lies in a known block
/// of its class (Shared or Private), zeroed from `zero_from` alone: `Some(None)` where rt refuses
/// it, counted; `None` where no such block holds it.
fn in_known(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    length: u64,
    o: Options,
    pin: u64,
    zero_from: Option<u64>,
) -> Option<Option<RtObj<RtBuffer>>> {
    let class = match o.storage {
        Storage::Shared => rt::MemoryClass::Shared,
        Storage::Private => rt::MemoryClass::Private,
        Storage::Managed => return None,
    };
    let (block, offset) = known_at(device, pin, length, class, None)?;
    match device.rt().new_buffer(block.id, offset, length, zero_from) {
        Ok(made) => Some(Some(buffer(device, made, o.answered))),
        Err(error) => {
            rt_refused_create(sel, &error);
            Some(None)
        }
    }
}

/// A block of `class` for `length` bytes, for `sel`; `None` where rt refuses it, counted.
pub(super) fn take(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    class: rt::MemoryClass,
    length: u64,
) -> Option<rt::Block> {
    match device.rt().take_block(class, length) {
        Ok(block) => Some(block),
        Err(error) => {
            rt_refused_create(sel, &error);
            None
        }
    }
}

/// rt's buffer of zeros, of `length` bytes, in a block of `class` of its own, for `sel`; `None`
/// where rt refuses it, counted.
fn own(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    class: rt::MemoryClass,
    length: u64,
) -> Option<rt::Buffer> {
    let block = take(device, sel, class, length)?;
    match device.rt().new_buffer(block.id, 0, length, Some(0)) {
        Ok(made) => Some(made),
        Err(error) => {
            rt_refused_create(sel, &error);
            None
        }
    }
}

/// rt's Managed buffer of zeros, of `length` bytes, its bytes and its device copy (where the GPU
/// keeps one) in blocks of their own, for `sel`; `None` where rt refuses it, counted.
fn own_managed(device: &RtObj<RtDevice>, sel: &'static str, length: u64) -> Option<rt::Buffer> {
    let bytes = take(device, sel, rt::MemoryClass::Shared, length)?;
    if let Some(at) = bytes.contents {
        // SAFETY: the block is this call's alone, mapped for its life, and its `len`, the length
        // rounded up, holds the `length` bytes.
        unsafe { at.as_ptr().write_bytes(0, length.to_usize()) };
    }
    let copy = match device.rt().has_unified_memory() {
        true => None,
        false => Some(take(device, sel, rt::MemoryClass::Private, length)?),
    };
    let at = copy.as_ref().map(|copy| (copy.id, 0));
    match device.rt().new_managed_buffer((bytes.id, 0), at, length, 0) {
        Ok(made) => Some(made),
        Err(error) => {
            rt_refused_create(sel, &error);
            None
        }
    }
}

/// The slot of a no-copy buffer's deallocator, which runs it once: rt's release of its import, or
/// the buffer's drop over a known block.
pub(super) type Slot = Arc<Mutex<Option<NoCopyDeallocator>>>;

/// Runs the deallocator in `slot`, where it is still there.
fn run(slot: &Slot) {
    let deallocator = slot.lock().take();
    if let Some(deallocator) = deallocator {
        deallocator();
    }
}

/// The deallocator in `slot`, never to run: taken out, and its captures kept for good, as Metal
/// runs none for a buffer it does not make.
fn keep_unrun(slot: &Slot) {
    if let Some(deallocator) = slot.lock().take() {
        std::mem::forget(deallocator);
    }
}

/// `newBufferWithBytesNoCopy:length:options:deallocator:` on `device`, for `sel`: a buffer of
/// `length` bytes at `bytes`' first byte (no more than `bytes` covers), at the guest's `pin` where
/// the bytes are a known block's mapping there, else over rt's import of their pages at rt's
/// address (the module's docs, "A pinned address", "Imports"); a Managed one's bytes from
/// `zero_from` read as zeros by its device copy. Nil where Metal or rt refuses it, counted,
/// `deallocator` then never run.
#[expect(
    clippy::too_many_arguments,
    reason = "the descriptor's no-copy fields, which the creators share"
)]
pub fn over_host_memory(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    bytes: HostRange<'_>,
    length: u64,
    raw: u64,
    zero_from: Option<u64>,
    pin: Option<u64>,
    deallocator: Option<NoCopyDeallocator>,
) -> Option<RtObj<RtBuffer>> {
    let slot: Slot = Arc::new(Mutex::new(deallocator));
    let length = length.min(bytes.len());
    let made = no_copy(device, sel, bytes.ptr(), length, raw, zero_from, pin, &slot);
    if made.is_none() {
        keep_unrun(&slot);
    }
    made
}

/// [`over_host_memory`]'s buffer over the `length` bytes at `first`, its deallocator in `slot`.
/// Where the buffer is not made, the deallocator stays in the slot.
#[expect(
    clippy::too_many_arguments,
    reason = "over_host_memory's arguments and the slot"
)]
fn no_copy(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    first: HostPtr,
    length: u64,
    raw: u64,
    zero_from: Option<u64>,
    pin: Option<u64>,
    slot: &Slot,
) -> Option<RtObj<RtBuffer>> {
    let o = match options(raw, true) {
        Ok(o) => o,
        Err(why) => return refused_metal(sel, &why),
    };
    if length == 0 || length > rt::MAX_BUFFER_LENGTH {
        return refused_metal(sel, &format!("a buffer of {length} bytes"));
    }
    // A Managed buffer's GPU copy, where the GPU keeps one, is what its address is of: at the pin
    // where a known Private block holds it, its bytes then the import of their pages.
    let two_copies = o.storage == Storage::Managed && !device.rt().has_unified_memory();
    let (shared, private) = (rt::MemoryClass::Shared, rt::MemoryClass::Private);
    let copy = pin.filter(|_| two_copies);
    let copy = copy.and_then(|pin| known_at(device, pin, length, private, None));
    let at = |pin| known_at(device, pin, length, shared, Some(first));
    let (block, offset, deallocator) = match pin.filter(|_| !two_copies).and_then(at) {
        Some((block, offset)) => (block, offset, Some(Arc::clone(slot))),
        None => {
            if let Some(pin) = pin
                && copy.is_none()
            {
                unpinned(sel, pin);
            }
            match device.known_blocks().mapping(first, length) {
                Some((block, offset)) => (block, offset, Some(Arc::clone(slot))),
                None => {
                    let held = Arc::clone(slot);
                    let made = import_pages(device, sel, first, length, move || run(&held));
                    let (block, offset) = made?;
                    (block, offset, None)
                }
            }
        }
    };
    let copy = copy.map(|(block, offset)| (block.id, offset));
    let (bytes, storage) = ((&block, offset), o.storage);
    // The deallocator is kept unrun before the import goes, whose release would run it.
    let Some(made) = range_of(device, sel, bytes, length, storage, zero_from, copy) else {
        keep_unrun(slot);
        return None;
    };
    if o.read_only {
        read_only(sel);
    }
    Some(buffer_of(device, made, o.answered, false, deallocator))
}

/// rt's no-copy buffer of `length` bytes at `bytes`, an offset in a block, of `storage`, for `sel`:
/// Shared over the bytes, or Managed over them with a device copy where the GPU keeps one, at
/// `copy` (an offset in a known Private block) where given, else in a block of its own, whose
/// bytes from `zero_from` start as zeros. `None` where rt refuses it, counted.
fn range_of(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    bytes: (&rt::Block, u64),
    length: u64,
    storage: Storage,
    zero_from: Option<u64>,
    copy: Option<(rt::BlockId, u64)>,
) -> Option<rt::Buffer> {
    let (block, offset) = bytes;
    let made = match storage {
        Storage::Managed => {
            let own = match device.rt().has_unified_memory() || copy.is_some() {
                true => None,
                false => Some(take(device, sel, rt::MemoryClass::Private, length)?),
            };
            let copied = zero_from.map_or(length, |from| from.min(length));
            let at = copy.or(own.as_ref().map(|own| (own.id, 0)));
            (device.rt()).new_managed_buffer((block.id, offset), at, length, copied)
        }
        Storage::Shared | Storage::Private => {
            (device.rt()).new_buffer(block.id, offset, length, None)
        }
    };
    match made {
        Ok(made) => Some(made),
        Err(error) => {
            rt_refused_create(sel, &error);
            None
        }
    }
}

/// rt's import of the host's pages that hold the `length` bytes at `first`, for `sel` (the module's
/// docs, "Imports"), counted under the kind of the make running on the thread: the block, and the
/// bytes' offset in it. rt's release of the import runs
/// `then`, after the import is freed. `None` where Metal or rt refuses it, counted, `then` dropped
/// unrun.
pub(super) fn import_pages(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    first: HostPtr,
    length: u64,
    then: impl FnOnce() + Send + 'static,
) -> Option<(rt::Block, u64)> {
    let first = first.addr().to_u64();
    let page = device.facts().page_size.max(1);
    let start = first - first % page;
    let end = (first.checked_add(length)).and_then(|end| end.checked_next_multiple_of(page));
    let at = NonNull::new(HostPtr::from_addr(start.to_usize()).as_ptr::<u8>());
    let (Some(end), Some(at)) = (end, at) else {
        return refused_metal(sel, &format!("{length} bytes at {first:#x}"));
    };
    let codes = crate::answer::making().unwrap_or(IMPORTS_UNNAMED);
    let release: NoCopyDeallocator = Box::new(move || {
        LIVE_IMPORTS.fetch_sub(1, Ordering::Relaxed);
        crate::answer::count_gone(codes);
        then();
    });
    // SAFETY: the creators' `HostRange` contracts keep the bytes mapped and readable until their
    // deallocator runs, which runs only from `release`, once rt freed the import, or as the
    // object over a known block goes, which holds no import; with none, for the object's life,
    // which holds the import. The pages they lie in, which the import covers, are mapped as they
    // are. They are mapped writable too: the no-copy creators' bytes are memory the host maps
    // readable and writable (the device's region, the host's own).
    let imported = unsafe { (device.rt()).import_host_memory(at, end - start, Some(release)) };
    match imported {
        Ok((block, _)) => {
            imports_rise();
            super::diag::imported(block.id, sel);
            crate::answer::count_made(codes);
            Some((block, first - start))
        }
        Err(error) => {
            rt_refused_create(sel, &error);
            None
        }
    }
}

/// rt's import of the host's pages that hold the `length` bytes at `first`, as a no-copy buffer's
/// is made and counted ([`import_pages`]), known to `device` from now (`KnownBlocks::know`): the
/// block and the bytes' offset in it. A no-copy buffer over the bytes pinned at the block's
/// address plus their offset is then that range of the block ("A pinned address"), with no import
/// of its own. The caller forgets the block (`KnownBlocks::forget`) as the last such buffer goes;
/// rt frees the import once no buffer over it and no command buffer that used it lives, then
/// runs `then`. `None` where Metal or rt refuses it, counted, `then` dropped unrun.
pub fn import_known(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    first: HostPtr,
    length: u64,
    then: impl FnOnce() + Send + 'static,
) -> Option<(rt::Block, u64)> {
    let (block, offset) = import_pages(device, sel, first, length, then)?;
    device.known_blocks().know(block.clone());
    Some((block, offset))
}

/// The imports of the host's memory live now.
static LIVE_IMPORTS: AtomicU64 = AtomicU64::new(0);

/// The most imports of the host's memory live at once.
static MOST_IMPORTS: AtomicU64 = AtomicU64::new(0);

/// Counts an import made ([`IMPORTS_CODE`]), one more live, and a new most as
/// [`IMPORTS_LIVE_MOST_CODE`]'s rise to it.
fn imports_rise() {
    crate::answer::count_kind(IMPORTS_CODE);
    let live = LIVE_IMPORTS.fetch_add(1, Ordering::Relaxed) + 1;
    let most = MOST_IMPORTS.fetch_max(live, Ordering::Relaxed);
    if live > most {
        crate::answer::count_kind_by(IMPORTS_LIVE_MOST_CODE, live - most);
    }
}

/// Whether the process made a buffer or heap at rt's address for a pin yet.
static UNPINNED: AtomicBool = AtomicBool::new(false);

/// Counts a buffer or heap pinned at `pin` and made at rt's address ([`UNPINNED_CODE`]); the
/// process's first writes its line.
pub(super) fn unpinned(sel: &'static str, pin: u64) {
    crate::answer::count_kind(UNPINNED_CODE);
    if !UNPINNED.swap(true, Ordering::Relaxed) {
        log::warn!(
            target: TARGET,
            "{sel}: the guest pinned it at {pin:#x}, in no block the device knows, so it lies at \
             metal2vulkan-rt's address, which the guest's pointers to it do not hold \
             ({UNPINNED_CODE}), counted from here"
        );
    }
}

/// Whether the process made a no-copy buffer or a heap of option bit 24 yet.
static READ_ONLY_SAID: AtomicBool = AtomicBool::new(false);

/// Counts a no-copy buffer or a heap of option bit 24 ([`READ_ONLY_CODE`]); the process's first
/// writes its line.
pub(super) fn read_only(sel: &'static str) {
    crate::answer::count_kind(READ_ONLY_CODE);
    if !READ_ONLY_SAID.swap(true, Ordering::Relaxed) {
        log::warn!(
            target: TARGET,
            "{sel}: option bit 24 (read-only) is taken and answered without, but the GPU's writes \
             reach a no-copy buffer's bytes, where Metal drops them ({READ_ONLY_CODE}), counted \
             from here"
        );
    }
}

impl RtObj<RtBuffer> {
    /// The texture `sel` makes of `descriptor` over the buffer's bytes from `offset`, rows
    /// `bytes_per_row` apart (`super::texture::make_over`).
    fn texture(
        &self,
        sel: &'static str,
        descriptor: &RtObj<TextureFields>,
        offset: u64,
        bytes_per_row: u64,
    ) -> Option<RtObj<RtTexture>> {
        let over = |d: &rt::TextureDescriptor, index: Option<u32>| {
            self.rt.new_texture(d, index, offset, bytes_per_row)
        };
        let held = (self.options, texture::STORAGE_BITS);
        let diag = |shape: &Shape| Made::over_buffer(&self.rt, shape, offset, bytes_per_row);
        texture::make_over(&self.device, sel, &descriptor.desc(), held, over, diag)
    }
}

impl GpuResource<RtPlatform> for RtObj<RtBuffer> {
    fn device(&self) -> Option<RtObj<RtDevice>> {
        Some(self.device.clone_ref())
    }

    /// The asked ones without bits 20 and 24, hazard tracking resolved (the module's docs).
    fn resource_options(&self) -> u64 {
        self.options
    }

    /// Labels are diagnostics: none is kept.
    fn set_label(&self, label: &str) -> RaiseResult<()> {
        let _ = label;
        Ok(())
    }

    /// `super::texture::set_purgeable`'s; a heap's buffer answers `NonVolatile` to every state and
    /// changes nothing, as Metal answers where its validation stops each (the Vulkan platform's
    /// runtime's answer).
    fn set_purgeable_state(&self, state: u64) -> RaiseResult<u64> {
        if self.in_heap {
            return Ok(texture::NON_VOLATILE);
        }
        Ok(texture::set_purgeable(&self.purgeable, state))
    }

    /// Bits 4 to 7 of its resource options.
    fn storage_mode(&self) -> u64 {
        (self.options >> 4) & 0xf
    }
}

impl Buffer<RtPlatform> for RtObj<RtBuffer> {
    /// rt's: NULL for a Private buffer, as Metal's.
    fn contents(&self) -> HostPtr {
        match self.rt.contents() {
            Some(at) => HostPtr::from_ptr(at.as_ptr()),
            None => HostPtr::NULL,
        }
    }

    /// Of a Managed buffer, rt's, which copies the range into its device copy at the next commit;
    /// a range past the buffer changes nothing, counted, where Metal's validation aborts. Of any
    /// other, nothing, as Metal.
    fn did_modify_range(&self, on: OnFault, range: NsRange) -> RaiseResult<()> {
        let _ = on;
        if self.storage_mode() == MANAGED
            && let Err(error) = self.rt.did_modify_range(range.location, range.length)
        {
            rt_refused("didModifyRange:", &error);
        }
        Ok(())
    }

    /// rt's: the Vulkan device address of its first byte (of a Managed buffer's device copy where
    /// it has one).
    fn gpu_address(&self) -> u64 {
        self.rt.gpu_address()
    }

    fn length(&self) -> u64 {
        self.rt.length()
    }

    /// As `newTextureWithDescriptor:offset:bytesPerRow:`: rt makes a texture over a buffer of one
    /// image alone (2D, one level), whose `bytes_per_image` reads nothing.
    fn new_linear_texture(
        &self,
        descriptor: &RtObj<TextureFields>,
        offset: u64,
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> RaiseResult<Option<RtObj<RtTexture>>> {
        let _ = bytes_per_image;
        let sel = "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:";
        Ok(self.texture(sel, descriptor, offset, bytes_per_row))
    }

    /// rt's texture over the buffer's bytes (rt's `Buffer::new_texture`), at an index as
    /// `newTextureWithDescriptor:`'s, of the buffer's storage mode. Nil where rt refuses it (a
    /// texture other than 2D of one level, rows or an offset off the format's texels), counted.
    fn new_texture(
        &self,
        on: OnFault,
        descriptor: &RtObj<TextureFields>,
        offset: u64,
        bytes_per_row: u64,
    ) -> RaiseResult<Option<RtObj<RtTexture>>> {
        let _ = on;
        let sel = "newTextureWithDescriptor:offset:bytesPerRow:";
        Ok(self.texture(sel, descriptor, offset, bytes_per_row))
    }

    /// Unbuilt: nil ([`Unbuilt::Buffer`]): a tiled texture is a copy of the bytes, taken at its
    /// first use, which rt's texture over a buffer is not.
    fn new_tiled_texture(
        &self,
        descriptor: &RtObj<TextureFields>,
        offset: u64,
        bytes_per_row: u64,
    ) -> RaiseResult<Option<RtObj<RtTexture>>> {
        let _ = (descriptor, offset, bytes_per_row);
        let sel = "newTiledTextureWithDescriptor:offset:bytesPerRow:";
        refuse_create(Unbuilt::Buffer, sel);
        Ok(None)
    }

    /// Unbuilt: nil ([`Unbuilt::IosurfaceTexture`]), as `newTextureWithDescriptor:iosurface:plane:`
    /// answers on the platform.
    fn new_texture_over_surface_plane(
        &self,
        on: OnFault,
        descriptor: &RtObj<TextureFields>,
        plane: SurfacePlane,
    ) -> RaiseResult<Option<RtObj<RtTexture>>> {
        let _ = (on, descriptor, plane);
        let sel = "newTextureWithDescriptor:iosurface:plane:";
        refuse_create(Unbuilt::IosurfaceTexture, sel);
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use std::alloc::Layout;
    use std::time::Duration;

    use super::*;
    use crate::core::platform::Platform as _;
    use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
    use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::values::{OptionSet, TextureDesc};
    use crate::rt_platform::RtRefusal;

    /// A buffer's options: Shared, Private and Managed, Metal's default hazard tracking answered
    /// tracked, bits 20 and 24 answered without; Metal's refusals refused.
    #[test]
    fn a_buffer_s_options_are_metal_s() {
        let made = |raw, no_copy| options(raw, no_copy).map(|o| (o.storage, o.answered));
        assert_eq!(made(0, false), Ok((Storage::Shared, 0x200)));
        assert_eq!(made(0x20, false), Ok((Storage::Private, 0x220)));
        assert_eq!(made(0x10, true), Ok((Storage::Managed, 0x210)));
        assert_eq!(made(0x101 | BIT_20, false), Ok((Storage::Shared, 0x101)));
        let read_only = options(READ_ONLY | 0x200, true);
        assert_eq!(
            read_only.map(|o| (o.answered, o.read_only)),
            Ok((0x200, true))
        );
        for (raw, no_copy) in [
            (READ_ONLY, false),
            (0x20, true),
            (0x30, false),
            (0x2, false),
            (0x300, false),
            (1 << 30, false),
        ] {
            assert!(options(raw, no_copy).is_err(), "{raw:#x} {no_copy}");
        }
    }

    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// `length` bytes at `at`, read.
    fn read(at: HostPtr, length: usize) -> Vec<u8> {
        // SAFETY: the callers' pointers are a live buffer's contents or a block's mapping, at
        // least `length` bytes long.
        unsafe { std::slice::from_raw_parts(at.as_ptr::<u8>(), length) }.to_vec()
    }

    /// A Shared or Private buffer descriptor of 64 bytes pinned at `pin`, no-copy over `contents`
    /// where given.
    fn pinned(pin: u64, options: u64, contents: Option<HostRange<'_>>) -> BufferDesc<'_> {
        BufferDesc {
            length: 64,
            options: vec![OptionSet::ResourceOptions(options)],
            pinned_gpu_address: Some(pin),
            no_copy: contents.is_some(),
            contents,
            deallocator: None,
            zero_from: None,
        }
    }

    /// Asks `done` every 10 ms, up to 1000 times (about 10 s).
    fn waits_for(done: impl Fn() -> bool) -> bool {
        for _ in 0..1000 {
            if done() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        done()
    }

    /// Buffers on the GPU the test holds: Shared, Private and Managed ones of zeros, answering
    /// their options, a Private one's contents NULL; Metal's refusals nil and counted; a Managed
    /// buffer's range past its end counted, a Shared one's not; no-copy buffers over the host's
    /// memory at its bytes, their deallocators run once rt freed the import, with the bytes and
    /// the length through a descriptor; a pin in no known block made at rt's address, counted;
    /// pins in known blocks at the pin, a Shared one keeping the bytes the CPU wrote there, a
    /// no-copy one over the block's mapping its deallocator run as it goes, and one pinned
    /// elsewhere a range of the block still, with no import; a Managed no-copy one's device copy
    /// in a Private block where the GPU keeps one, which a blit reads, its deallocator run once
    /// the commit captured its bytes and the buffer went; a texture over a buffer.
    #[test]
    fn buffers_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let of = |length, options| device.new_buffer_of_length(on, length, options);
        let of = |length, options| of(length, options).expect("never a raise");

        for (options, answered) in [(0, 0x200), (0x20, 0x220), (0x10, 0x210)] {
            let buffer = of(100, options).expect("a buffer");
            let mode = (options >> 4) & 0xf;
            assert_eq!(buffer.length(), 100);
            assert_eq!(
                (buffer.resource_options(), buffer.storage_mode()),
                (answered, mode)
            );
            assert_ne!(buffer.gpu_address(), 0);
            match options {
                0x20 => assert_eq!(buffer.contents(), HostPtr::NULL),
                _ => assert_eq!(read(buffer.contents(), 100), [0; 100]),
            }
        }

        let metal = RtRefusal::Refused.code();
        let before = counted(metal);
        assert!(of(0, 0).is_none() && of(64, 0x30).is_none() && of(64, READ_ONLY).is_none());
        assert_eq!(counted(metal), before + 3);
        let managed = of(64, 0x10).expect("a Managed buffer");
        let shared = of(64, 0).expect("a Shared buffer");
        assert_eq!(managed.did_modify_range(on, NsRange::new(0, 64)), Ok(()));
        assert_eq!(shared.did_modify_range(on, NsRange::new(32, 64)), Ok(()));
        assert_eq!(counted(metal), before + 3);
        assert_eq!(managed.did_modify_range(on, NsRange::new(32, 64)), Ok(()));
        assert_eq!(counted(metal), before + 4);

        // A no-copy buffer 16 bytes into memory of the host's own.
        let layout = Layout::from_size_align(0x20000, 0x10000).expect("a layout");
        // SAFETY: the layout's size is not 0.
        let memory = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!memory.is_null());
        let at = HostPtr::from_ptr(memory.wrapping_add(16));
        let imports = (counted(IMPORTS_CODE), counted(IMPORTS_LIVE_MOST_CODE));
        let unnamed = counted(IMPORTS_UNNAMED.made);
        let freed = Arc::new(AtomicBool::new(false));
        let done = Arc::clone(&freed);
        let deallocator: NoCopyDeallocator = Box::new(move || done.store(true, Ordering::Release));
        // SAFETY: the memory stays allocated until the deallocator ran (below).
        let bytes = unsafe { HostRange::new(at, 1000) };
        let made = device.new_buffer_over_host_memory(on, bytes, 0, Some(deallocator));
        let buffer = made.expect("never a raise").expect("a no-copy buffer");
        assert_eq!((buffer.contents(), buffer.length()), (at, 1000));
        assert_eq!(counted(IMPORTS_CODE), imports.0 + 1);
        assert!(counted(IMPORTS_LIVE_MOST_CODE) >= imports.1.max(1));
        assert_eq!(counted(IMPORTS_UNNAMED.made), unnamed + 1);
        assert_eq!(crate::answer::live_now(IMPORTS_UNNAMED), 1);
        assert!(!freed.load(Ordering::Acquire));
        drop(buffer);
        let ran = || freed.load(Ordering::Acquire);
        assert!(waits_for(ran), "the deallocator ran");
        assert_eq!(crate::answer::live_now(IMPORTS_UNNAMED), 0);

        // Through a descriptor, read-only: the deallocator takes the bytes and the length.
        let given = Arc::new(Mutex::new(None));
        let take = Arc::clone(&given);
        let read_only_before = counted(READ_ONLY_CODE);
        // Made as a kind of make of the test's, which its import counts under.
        const MADE: crate::answer::MadeCodes = crate::answer::MadeCodes {
            made: "rt-imports-test",
            live_most: "rt-imports-test-live-most",
        };
        let (imports, unnamed) = (counted(IMPORTS_CODE), counted(IMPORTS_UNNAMED.made));
        // SAFETY: as above.
        let bytes = unsafe { HostRange::new(at, 1000) };
        let made = crate::answer::making_as(MADE, || {
            device.new_buffer(BufferDesc {
                length: 1000,
                options: vec![OptionSet::ResourceOptions(READ_ONLY)],
                pinned_gpu_address: None,
                contents: Some(bytes),
                no_copy: true,
                deallocator: Some(Box::new(move |at, length| {
                    *take.lock() = Some((at.addr(), length));
                })),
                zero_from: None,
            })
        });
        let buffer = made
            .expect("never a raise")
            .expect("a read-only no-copy buffer");
        assert_eq!(buffer.resource_options(), 0x200);
        assert_eq!(counted(IMPORTS_CODE), imports + 1);
        assert_eq!(counted(MADE.made), 1);
        assert_eq!(counted(IMPORTS_UNNAMED.made), unnamed);
        assert_eq!(counted(READ_ONLY_CODE), read_only_before + 1);
        drop(buffer);
        assert!(waits_for(|| given.lock().is_some()));
        assert_eq!(*given.lock(), Some((at.addr(), 1000)));
        // SAFETY: both deallocators ran: rt holds no import of the memory.
        unsafe { std::alloc::dealloc(memory, layout) };

        // A pin in no known block.
        let unpinned = counted(UNPINNED_CODE);
        let pin = 1 << 46;
        let made = device
            .new_buffer(pinned(pin, 0, None))
            .expect("never a raise");
        assert_ne!(made.expect("a buffer at rt's address").gpu_address(), pin);
        assert_eq!(counted(UNPINNED_CODE), unpinned + 1);

        // Pins in known blocks.
        let shared = (device.rt()).take_block(rt::MemoryClass::Shared, 0x10000);
        let private = (device.rt()).take_block(rt::MemoryClass::Private, 0x10000);
        let (shared, private) = (
            shared.expect("a Shared block"),
            private.expect("a Private one"),
        );
        let known = device.known_blocks().clone();
        known.know(shared.clone());
        known.know(private.clone());
        let mapping = shared.contents.expect("a Shared block's mapping").as_ptr();
        let written = HostPtr::from_ptr(mapping.wrapping_add(0x100));
        // SAFETY: the block is mapped while it lives, and 0x100 + 64 bytes lie in it.
        unsafe { written.as_ptr::<u8>().write_bytes(0xab, 64) };
        let pin = shared.gpu_address + 0x100;
        let made = device
            .new_buffer(pinned(pin, 0, None))
            .expect("never a raise");
        let made = made.expect("a buffer in the known block");
        assert_eq!((made.gpu_address(), made.contents()), (pin, written));
        assert_eq!(read(made.contents(), 64), [0xab; 64]);
        let pin = private.gpu_address + 0x200;
        let made = device
            .new_buffer(pinned(pin, 0x20, None))
            .expect("never a raise");
        assert_eq!(
            made.expect("a Private buffer in its block").gpu_address(),
            pin
        );
        let pin = shared.gpu_address + 0x100;
        let made = device
            .new_buffer(pinned(pin, 0x20, None))
            .expect("never a raise");
        assert_ne!(made.expect("a Private buffer, unpinned").gpu_address(), pin);

        // A no-copy one over the known block's mapping.
        let ran = Arc::new(AtomicBool::new(false));
        let run = Arc::clone(&ran);
        // SAFETY: the block's mapping lives while the test holds the block.
        let bytes = unsafe { HostRange::new(written, 64) };
        let desc = BufferDesc {
            deallocator: Some(Box::new(move |_, _| run.store(true, Ordering::Release))),
            ..pinned(pin, 0, Some(bytes))
        };
        let imports = counted(IMPORTS_CODE);
        let made = device.new_buffer(desc).expect("never a raise");
        let made = made.expect("a no-copy buffer in the known block");
        assert_eq!((made.gpu_address(), made.contents()), (pin, written));
        assert_eq!(counted(IMPORTS_CODE), imports);
        assert!(!ran.load(Ordering::Acquire));
        drop(made);
        assert!(ran.load(Ordering::Acquire));
        assert_eq!(counted(UNPINNED_CODE), unpinned + 2);

        // Over the Shared block's mapping, pinned elsewhere: a range of the block with no import, a
        // Shared one at the block's address, counted as not kept; a Managed one's device copy at
        // the pin in the Private block where the GPU keeps one, else counted as not kept too.
        let unified = device.rt().has_unified_memory();
        let (unpinned, imports) = (counted(UNPINNED_CODE), counted(IMPORTS_CODE));
        let pin = private.gpu_address + 0x800;
        for (options, kept) in [(0, false), (0x10, !unified)] {
            // SAFETY: the block's mapping lives while the test holds the block.
            let bytes = unsafe { HostRange::new(written, 64) };
            let made = device.new_buffer(pinned(pin, options, Some(bytes)));
            let made = made
                .expect("never a raise")
                .expect("a buffer over the block");
            let at = if kept {
                pin
            } else {
                shared.gpu_address + 0x100
            };
            assert_eq!((made.gpu_address(), made.contents()), (at, written));
        }
        assert_eq!(counted(IMPORTS_CODE), imports);
        assert_eq!(counted(UNPINNED_CODE), unpinned + 1 + u64::from(unified));

        // A Managed no-copy one over memory of the host's own, pinned in the Private block: its
        // device copy at the pin where the GPU keeps one; else lying at its bytes' import, the
        // pin not kept, counted. A blit reads the bytes the CPU wrote. Its deallocator runs once
        // rt freed the import: where the GPU keeps a copy, after the commit that captured the
        // bytes its making named (the module's docs, "Imports").
        let unpinned = counted(UNPINNED_CODE);
        // SAFETY: the layout's size is not 0.
        let memory = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!memory.is_null());
        let at = HostPtr::from_ptr(memory.wrapping_add(16));
        // SAFETY: the 64 bytes at `at` lie in the memory, which nothing else uses yet.
        unsafe { at.as_ptr::<u8>().write_bytes(0xcd, 64) };
        let freed = Arc::new(AtomicBool::new(false));
        let done = Arc::clone(&freed);
        let pin = private.gpu_address + 0x400;
        // SAFETY: the memory stays allocated until the deallocator ran (below).
        let bytes = unsafe { HostRange::new(at, 64) };
        let desc = BufferDesc {
            deallocator: Some(Box::new(move |_, _| done.store(true, Ordering::Release))),
            ..pinned(pin, 0x10, Some(bytes))
        };
        let made = device.new_buffer(desc).expect("never a raise");
        let made = made.expect("a Managed no-copy buffer");
        assert_eq!(made.contents(), at);
        assert_eq!(made.gpu_address() == pin, !unified);
        assert_eq!(counted(UNPINNED_CODE), unpinned + u64::from(unified));
        let out = of(64, 0).expect("a buffer");
        let queue = device.new_command_queue().expect("a command queue");
        let commands = queue.command_buffer(on).expect("never a raise");
        let commands = commands.expect("a command buffer");
        let encoder = commands.blit_encoder(on).expect("never a raise");
        let encoder = encoder.expect("a blit encoder");
        let copy = BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(&made),
            source_offset: 0,
            destination_buffer: Some(&out),
            destination_offset: 0,
            size: 64,
        };
        assert_eq!(encoder.encode(on, copy), Ok(()));
        assert_eq!(encoder.encode(on, BlitCmd::EndEncoding), Ok(()));
        assert_eq!(commands.commit(on), Ok(()));
        assert_eq!(commands.wait_until_completed(on), Ok(()));
        assert_eq!(read(out.contents(), 64), [0xcd; 64]);
        drop((encoder, commands, made));
        let ran = || freed.load(Ordering::Acquire);
        assert!(waits_for(ran), "the deallocator ran");
        // SAFETY: the deallocator ran: rt holds no import of the memory.
        unsafe { std::alloc::dealloc(memory, layout) };
        assert!(known.forget(shared.id).is_some() && known.forget(private.id).is_some());
        assert!(known.forget(shared.id).is_none());

        // A texture over a Shared buffer.
        let buffer = of(64 * 64 * 4, 0).expect("a buffer");
        let descriptor = platform.texture_descriptor(&TextureDesc {
            width: 64,
            height: 64,
            usage: 0x1,
            ..TextureDesc::default()
        });
        let texture = buffer
            .new_texture(on, &descriptor, 0, 256)
            .expect("never a raise");
        assert!(texture.is_some());
    }
}
