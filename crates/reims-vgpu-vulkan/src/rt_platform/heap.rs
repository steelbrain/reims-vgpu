//! The rt platform's heaps ([`RtHeap`]): each a range of a block, made by the device's
//! `newHeapWithDescriptor:` ([`of_desc`]) and over the host's memory ([`over_host_memory`]), and
//! the buffers and textures placed in them.
//!
//! **Its block.** A heap pinned at a GPU address (`setPinnedGPUAddress:`) inside a block the device
//! knows of its storage mode ([`KnownBlocks`](super::KnownBlocks)) is that range of the block, at
//! the pinned address, its bytes the block's as they are (the block's owner and the guest zero
//! them); one over the host's memory only where the memory is the block's own mapping there. Any
//! other heap of the device's takes a block of its own (rt's `take_block`), its bytes zeroed as
//! the Vulkan platform's runtime zeroes a new heap's (a Shared one's counted by the byte,
//! [`buffer::ZEROED_BYTES_CODE`]), and any other heap over the host's memory is a range of rt's
//! import of its pages (`super::buffer::import_pages`, counted as the no-copy buffers' imports
//! are), its bytes as they are. A pin neither keeps is counted
//! (`rt-gpu-address-unpinned`, `super::buffer`'s docs): the heap lies at rt's address. The heap's
//! size is the descriptor's rounded up to 16 KiB, as Metal's is; the heap holds its block.
//!
//! **What is placed in it.** A buffer is the heap's range at its offset (rt's `new_heap_buffer`,
//! its bytes as they are, as Metal's heap buffers are); a texture is rt's image bound there (rt's
//! `new_heap_texture`, which places textures in Private heaps alone: a Shared heap's texture is
//! rt's refusal, counted). Each takes the heap's storage, CPU cache and hazard tracking modes and
//! answers them; a heap's buffer answers `NonVolatile` to every purgeable state.
//!
//! **Hazards.** A heap's buffers and textures are one resource to rt's compute encoders' hazard
//! sets, the heap, whatever bytes each reaches: Metal tracks a heap's resources as one ("tracking
//! is done at the granularity of the heap", `MTLHeap.h`), so within a command buffer a write
//! through one orders every later use of another.
//!
//! **Answers.** A heap's resource options are the asked ones without bits 20 and 24 (bit 24
//! counted, `rt-no-copy-read-only`), a heap's default hazard tracking resolved to untracked, as
//! Metal's heaps answer it; its GPU address is its first byte's in the block.
//!
//! **Refusals.** What Metal refuses is nil, counted as Metal's refusal (`rt-refused-metal`): a
//! size of 0, an option value Metal has none of, a Managed or Memoryless heap, a heap over the
//! host's memory that is not Shared or that its bytes do not cover, a heap type Metal has none of;
//! a buffer or texture of a mode other than its heap's, in an automatic heap, at an offset off
//! 256 bytes (a buffer) or past the heap's end. What rt refuses is nil, counted under rt's kind; a
//! buffer of no bytes, which Metal places, is rt's refusal. A texture Metal places that rt refuses
//! on its place in the heap's block (a heap pinned at an offset of its block off the image's
//! alignment, which rt binds the image at) is counted apart from Metal's refusals
//! (`rt-refused-heap-texture-place`, [`super::unbuilt::HEAP_TEXTURE_PLACE`]): the guest pins its
//! Private heaps at 256 KiB from their block's first byte, the largest image alignment measured. A
//! sparse heap is unbuilt ([`Unbuilt::Heap`]).

use std::sync::atomic::AtomicU64;

use metal2vulkan_rt as rt;

use crate::core::num::ToUsize as _;
use crate::core::platform::Handle as _;
use crate::core::platform::host_range::HostRange;
use crate::core::platform::resource::Heap;
use crate::core::platform::values::HeapDesc;
use crate::core::raise::RaiseResult;

use super::buffer::{self, BIT_20, FIELDS, READ_ONLY, RtBuffer, known_at, take, unpinned};
use super::device::RtDevice;
use super::diag::{self, Call, Made};
use super::texture::{self, RtTexture, refused_metal};
use super::unbuilt::{Unbuilt, refuse_create, rt_refused_create};
use super::{RtObj, RtPlatform, TextureFields};

/// Metal's heap page: a heap's size is its descriptor's rounded up to it.
const HEAP_PAGE: u64 = 16 * 1024;

/// The offset a heap's buffer is placed at is a multiple of it, or Metal answers nil.
const PLACEMENT_ALIGNMENT: u64 = 256;

/// `MTLHeapTypeAutomatic`.
const AUTOMATIC: i64 = 0;

/// `MTLHeapTypePlacement`.
const PLACEMENT: i64 = 1;

/// `MTLHeapTypeSparse`.
const SPARSE: i64 = 2;

/// `MTLHazardTrackingModeUntracked`, in the resource options' bits 8 and 9.
const UNTRACKED: u64 = 0x100;

/// The CPU cache and storage modes, which a heap's resources share with it: bits 0 to 7.
const MODES: u64 = 0xff;

/// `id<MTLHeap>` on metal2vulkan-rt: its range of a block and what the platform answers of it.
pub struct RtHeap {
    /// The block it is a range of, held.
    block: rt::Block,
    /// Its first byte in the block.
    offset: u64,
    /// Its size: the descriptor's, rounded up to [`HEAP_PAGE`].
    size: u64,
    /// Whether it is a placement heap (an automatic one places nothing at an offset).
    placement: bool,
    /// The resource options it answers (the module's docs).
    options: u64,
    /// The device that made it.
    device: RtObj<RtDevice>,
    /// Its purgeable state, as Metal keeps it (`super::texture::set_purgeable`).
    purgeable: AtomicU64,
}

impl RtHeap {
    /// The number of the block it is a range of, which `useHeap:` keeps.
    pub fn block_id(&self) -> rt::BlockId {
        self.block.id
    }
}

/// What a heap's descriptor makes of it: its storage mode's block class, its size, whether it is
/// a placement heap, the options it answers and whether option bit 24 is set.
type Shape = (rt::MemoryClass, u64, bool, u64, bool);

/// The shape of a heap of `desc` and the descriptor's options `held` (the module's docs); `Err`
/// where Metal refuses it, saying what, and `Ok(None)` for a sparse heap, which is unbuilt.
fn shape(desc: &HeapDesc, held: u64) -> Result<Option<Shape>, String> {
    if held & !(FIELDS | BIT_20 | READ_ONLY) != 0 {
        return Err(format!("resource options {held:#x}"));
    }
    let (cache, hazard) = (held & 0xf, (held >> 8) & 0x3);
    if cache > 1 {
        return Err(format!("CPU cache mode {cache}"));
    }
    if hazard > 2 {
        return Err(format!("hazard tracking mode {hazard}"));
    }
    let class = match (held >> 4) & 0xf {
        0 => rt::MemoryClass::Shared,
        2 => rt::MemoryClass::Private,
        mode => return Err(format!("storage mode {mode} of a heap")),
    };
    let placement = match desc.heap_type {
        AUTOMATIC => false,
        PLACEMENT => true,
        SPARSE => return Ok(None),
        other => return Err(format!("heap type {other}")),
    };
    let size = desc.size.checked_next_multiple_of(HEAP_PAGE).unwrap_or(0);
    if size == 0 {
        return Err(format!("a heap of {} bytes", desc.size));
    }
    let answered = match hazard {
        0 => (held & !(BIT_20 | READ_ONLY)) | UNTRACKED,
        _ => held & !(BIT_20 | READ_ONLY),
    };
    Ok(Some((
        class,
        size,
        placement,
        answered,
        held & READ_ONLY != 0,
    )))
}

/// `newHeapWithDescriptor:` of `desc` on `device`, its descriptor's options `held`: at the pin
/// where it lies in a known block of its storage mode, else in a block of its own, zeroed (the
/// module's docs). Nil where Metal or rt refuses it, or a sparse heap, counted.
pub fn of_desc(device: &RtObj<RtDevice>, desc: &HeapDesc, held: u64) -> Option<RtObj<RtHeap>> {
    const SEL: &str = "newHeapWithDescriptor:";
    let (class, size, placement, options, read_only) = match shape(desc, held) {
        Ok(Some(shape)) => shape,
        Ok(None) => {
            refuse_create(Unbuilt::Heap, SEL);
            return None;
        }
        Err(why) => return refused_metal(SEL, &why),
    };
    let pin = desc.pinned_gpu_address;
    let at = |pin| known_at(device, pin, size, class, None);
    let (block, offset) = match pin.and_then(at) {
        Some(known) => known,
        None => {
            if let Some(pin) = pin {
                unpinned(SEL, pin);
            }
            (zeroed(device, SEL, class, size)?, 0)
        }
    };
    if read_only {
        buffer::read_only(SEL);
    }
    Some(heap(device, block, offset, size, placement, options))
}

/// A block of `class` of the `size` bytes of a heap of its own, for `sel`, zeroed: a Shared one
/// through its mapping, a Private one by rt's GPU fill. `None` where rt refuses it, counted.
fn zeroed(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    class: rt::MemoryClass,
    size: u64,
) -> Option<rt::Block> {
    let block = take(device, sel, class, size)?;
    match block.contents {
        Some(at) => {
            // SAFETY: the block is this call's alone, mapped for its life, and its `len`, the size
            // rounded up, holds the `size` bytes.
            unsafe { at.as_ptr().write_bytes(0, size.to_usize()) };
            crate::answer::count_kind_by(buffer::ZEROED_BYTES_CODE, size);
        }
        None => {
            if let Err(error) = device.rt().zero_block_range(block.id, 0, size) {
                rt_refused_create(sel, &error);
                return None;
            }
        }
    }
    Some(block)
}

/// A heap over the host's memory of `desc` on `device` (core's `new_heap_over_host_memory`), its
/// descriptor's options `held`: Shared, over `bytes`, at the pin where they are a known block's
/// mapping there, else over rt's import of their pages (the module's docs). Nil where Metal or rt
/// refuses it, counted.
pub fn over_host_memory(
    device: &RtObj<RtDevice>,
    bytes: HostRange<'_>,
    desc: &HeapDesc,
    held: u64,
) -> Option<RtObj<RtHeap>> {
    const SEL: &str = "newHeapWithDescriptor:";
    let (class, size, placement, options, read_only) = match shape(desc, held) {
        Ok(Some(shape)) => shape,
        Ok(None) => return refused_metal(SEL, &"a sparse heap over the host's memory"),
        Err(why) => return refused_metal(SEL, &why),
    };
    if class != rt::MemoryClass::Shared {
        return refused_metal(SEL, &"a Private heap over the host's memory");
    }
    if bytes.len() < size {
        let why = format!("{} bytes for a heap of {size}", bytes.len());
        return refused_metal(SEL, &why);
    }
    let first = bytes.ptr();
    let pin = desc.pinned_gpu_address;
    let at = |pin| known_at(device, pin, size, class, Some(first));
    let (block, offset) = match pin.and_then(at) {
        Some(known) => known,
        None => {
            if let Some(pin) = pin {
                unpinned(SEL, pin);
            }
            buffer::import_pages(device, SEL, first, size, || {})?
        }
    };
    if read_only {
        buffer::read_only(SEL);
    }
    Some(heap(device, block, offset, size, placement, options))
}

/// The heap's handle: `size` bytes at `offset` in `block`, answering `options`.
fn heap(
    device: &RtObj<RtDevice>,
    block: rt::Block,
    offset: u64,
    size: u64,
    placement: bool,
    options: u64,
) -> RtObj<RtHeap> {
    RtObj::new(RtHeap {
        block,
        offset,
        size,
        placement,
        options,
        device: device.clone_ref(),
        purgeable: texture::purgeable_state(),
    })
}

impl RtObj<RtHeap> {
    /// Its range of its block, which rt's heap resources are placed in and tracked by.
    fn range(&self) -> rt::HeapRange {
        rt::HeapRange {
            block: self.block.id,
            offset: self.offset,
            len: self.size,
        }
    }

    /// Why Metal answers nil for a resource of `length` bytes (`None`: unknown, a texture's) at
    /// `offset` in the heap, whose own options are `asked`; `None` where it places one there.
    fn misplaced(&self, asked: Option<u64>, offset: u64, length: Option<u64>) -> Option<String> {
        if !self.placement {
            return Some("a resource placed in an automatic heap".to_owned());
        }
        if asked.is_some_and(|asked| (asked ^ self.options) & MODES != 0) {
            return Some("a resource of a mode other than its heap's".to_owned());
        }
        let past = match length {
            Some(length) => (offset.checked_add(length)).is_none_or(|end| end > self.size),
            None => offset >= self.size,
        };
        let bytes = length.map_or("a texture".to_owned(), |length| format!("{length} bytes"));
        past.then(|| format!("{bytes} at {offset:#x} of a heap of {}", self.size))
    }
}

impl Heap<RtPlatform> for RtObj<RtHeap> {
    /// Its first byte's: Vulkan's device address in its block.
    fn gpu_address(&self) -> u64 {
        self.block.gpu_address + self.offset
    }

    /// rt's buffer of the heap's range at `offset`, its bytes as they are, tracked as the heap and
    /// answering the heap's options (the module's docs). Nil where Metal or rt refuses it, counted.
    fn new_buffer_of_length(
        &self,
        length: u64,
        options: u64,
        offset: u64,
    ) -> RaiseResult<Option<RtObj<RtBuffer>>> {
        const SEL: &str = "newBufferWithLength:options:offset:";
        if options & !(FIELDS | BIT_20 | READ_ONLY) != 0 {
            return Ok(refused_metal(
                SEL,
                &format!("resource options {options:#x}"),
            ));
        }
        let misplaced = match offset.is_multiple_of(PLACEMENT_ALIGNMENT) {
            true => self.misplaced(Some(options), offset, Some(length)),
            false => Some(format!("offset {offset:#x}")),
        };
        if let Some(why) = misplaced {
            return Ok(refused_metal(SEL, &why));
        }
        if length > rt::MAX_BUFFER_LENGTH {
            return Ok(refused_metal(SEL, &format!("a buffer of {length} bytes")));
        }
        let made = match (self.device.rt()).new_heap_buffer(self.range(), offset, length) {
            Ok(made) => made,
            Err(error) => {
                rt_refused_create(SEL, &error);
                return Ok(None);
            }
        };
        let buffer = buffer::buffer_of(&self.device, made, self.options, true, None);
        Ok(Some(buffer))
    }

    /// rt's texture of `descriptor` bound at `offset` in the heap (rt's `new_heap_texture`), at an
    /// index as `newTextureWithDescriptor:`'s, answering the heap's modes (the module's docs). Nil
    /// where Metal or rt refuses it (a Shared heap's, an offset off the image's alignment or past
    /// the heap), counted.
    fn new_texture(
        &self,
        descriptor: &RtObj<TextureFields>,
        offset: u64,
    ) -> RaiseResult<Option<RtObj<RtTexture>>> {
        const SEL: &str = "newTextureWithDescriptor:offset:";
        let desc = descriptor.desc();
        if let Some(why) = self.misplaced(desc.resource_options, offset, None) {
            return Ok(refused_metal(SEL, &why));
        }
        let range = self.range();
        let over = |d: &rt::TextureDescriptor, index: Option<u32>| {
            (self.device.rt()).new_heap_texture(d, index, range, offset)
        };
        let held = (self.options, FIELDS);
        let made = |shape: &diag::Shape| Made::new(Call::InHeap, shape);
        Ok(texture::make_over(
            &self.device,
            SEL,
            &desc,
            held,
            over,
            made,
        ))
    }

    /// The asked ones without bits 20 and 24, a default hazard tracking resolved to untracked.
    fn resource_options(&self) -> u64 {
        self.options
    }

    /// `super::texture::set_purgeable`'s.
    fn set_purgeable_state(&self, state: u64) -> RaiseResult<u64> {
        Ok(texture::set_purgeable(&self.purgeable, state))
    }

    /// Bits 4 to 7 of its resource options.
    fn storage_mode(&self) -> u64 {
        (self.options >> 4) & 0xf
    }
}

#[cfg(test)]
mod tests {
    use std::alloc::Layout;

    use super::*;
    use crate::core::host::HostPtr;
    use crate::core::platform::Platform as _;
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::resource::{Buffer as _, GpuResource as _, Texture as _};
    use crate::core::platform::types::SizeAndAlign;
    use crate::core::platform::values::{OptionSet, TextureDesc};
    use crate::rt_platform::RtRefusal;
    use crate::rt_platform::buffer::{IMPORTS_CODE, UNPINNED_CODE};
    use crate::rt_platform::unbuilt::HEAP_TEXTURE_PLACE;

    fn desc(size: u64, heap_type: i64) -> HeapDesc {
        HeapDesc {
            size,
            heap_type,
            options: vec![OptionSet::ResourceOptions(0)],
            pinned_gpu_address: None,
        }
    }

    /// A placement heap of `size` bytes and `options`, pinned at `pin` where given.
    fn placement(size: u64, options: u64, pin: Option<u64>) -> HeapDesc {
        HeapDesc {
            options: vec![OptionSet::ResourceOptions(options)],
            pinned_gpu_address: pin,
            ..desc(size, PLACEMENT)
        }
    }

    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// Heaps on the GPU the test holds: a Private one of its own, its buffers at their offsets
    /// answering its options and `NonVolatile`, Metal's refusals of a placement nil and counted, a
    /// texture placed in it; a Shared heap's texture rt's refusal; a heap pinned in a known block
    /// at the pin, and one pinned elsewhere at rt's address, counted; a heap over the host's
    /// memory over rt's import of it, its buffers' bytes the host's, and one over a known block's
    /// mapping at the pin with no import.
    #[test]
    fn heaps_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let heap = |desc: &HeapDesc| device.new_heap(desc).expect("never a raise").heap;
        let size = 0x40000;

        let private = heap(&placement(size, 0x20, None)).expect("a Private heap");
        assert_eq!(
            (private.resource_options(), private.storage_mode()),
            (0x120, 2)
        );
        assert_ne!(private.gpu_address(), 0);
        let place = |length, options, offset| {
            let made = private.new_buffer_of_length(length, options, offset);
            made.expect("never a raise")
        };
        let buffer = place(1000, 0x220, 0x100).expect("a buffer in the heap");
        assert_eq!(buffer.gpu_address(), private.gpu_address() + 0x100);
        assert_eq!((buffer.resource_options(), buffer.length()), (0x120, 1000));
        assert_eq!(buffer.set_purgeable_state(3), Ok(2));
        let metal = RtRefusal::Refused.code();
        let before = counted(metal);
        assert!(place(64, 0x20, 100).is_none());
        assert!(place(512, 0x20, size - 256).is_none());
        assert!(place(64, 0, 0).is_none());
        assert_eq!(counted(metal), before + 3);
        assert!(place(256, 0x20, size - 256).is_some());
        let texture = TextureDesc {
            width: 64,
            height: 64,
            resource_options: Some(0x20),
            ..TextureDesc::default()
        };
        let texture = platform.texture_descriptor(&texture);
        let placed = private.new_texture(&texture, 0).expect("never a raise");
        let placed = placed.expect("a texture in the heap");
        assert_eq!((placed.width(), placed.resource_options()), (64, 0x120));

        let shared = heap(&placement(size, 0, None)).expect("a Shared heap");
        assert_eq!(shared.resource_options(), 0x100);
        let gap = RtRefusal::Gap.code();
        let gaps = counted(gap);
        let texture = TextureDesc {
            resource_options: Some(0),
            ..TextureDesc::default()
        };
        let texture = platform.texture_descriptor(&texture);
        assert!(shared.new_texture(&texture, 0).is_ok_and(|t| t.is_none()));
        assert_eq!(counted(gap), gaps + 1);

        // Pinned heaps.
        let block = (device.rt()).take_block(rt::MemoryClass::Shared, size);
        let block = block.expect("a Shared block");
        device.known_blocks().know(block.clone());
        let unpinned = counted(UNPINNED_CODE);
        let pin = block.gpu_address + 0x4000;
        let known = heap(&placement(0x4000, 0, Some(pin))).expect("a heap in the block");
        assert_eq!(known.gpu_address(), pin);
        let other = heap(&placement(0x4000, 0x20, Some(pin))).expect("a Private heap");
        assert_ne!(other.gpu_address(), pin);
        assert_eq!(counted(UNPINNED_CODE), unpinned + 1);

        // Over the host's memory: memory of its own, then the known block's mapping.
        let layout = Layout::from_size_align(0x10000, 0x10000).expect("a layout");
        // SAFETY: the layout's size is not 0.
        let memory = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!memory.is_null());
        let imports = counted(IMPORTS_CODE);
        // SAFETY: the memory stays allocated until the test's end, past the heap's life and rt's
        // import's, which the device's teardown frees.
        let bytes = unsafe { HostRange::new(HostPtr::from_ptr(memory), 0x10000) };
        let made = device.new_heap_over_host_memory(bytes, &placement(0x10000, 0, None));
        let over = made
            .expect("never a raise")
            .heap
            .expect("a heap over the host's memory");
        assert_eq!(counted(IMPORTS_CODE), imports + 1);
        let buffer = over
            .new_buffer_of_length(64, 0, 0x200)
            .expect("never a raise");
        let buffer = buffer.expect("a buffer in it");
        assert_eq!(
            buffer.contents(),
            HostPtr::from_ptr(memory.wrapping_add(0x200))
        );
        let mapping = block.contents.expect("a Shared block's mapping").as_ptr();
        let first = HostPtr::from_ptr(mapping.wrapping_add(0x8000));
        // SAFETY: the block's mapping lives while the test holds the block.
        let bytes = unsafe { HostRange::new(first, 0x4000) };
        let pin = block.gpu_address + 0x8000;
        let made = device.new_heap_over_host_memory(bytes, &placement(0x4000, 0, Some(pin)));
        let over_block = made
            .expect("never a raise")
            .heap
            .expect("a heap over the block");
        assert_eq!(over_block.gpu_address(), pin);
        assert_eq!(counted(IMPORTS_CODE), imports + 1);
        let buffer = over_block
            .new_buffer_of_length(64, 0, 0x100)
            .expect("never a raise");
        assert_eq!(
            buffer.map(|b| b.contents()),
            Some(first.wrapping_add(0x100))
        );
        assert!(device.known_blocks().forget(block.id).is_some());
        // The memory is left allocated: rt's import of it goes with the device, at the process's
        // end.
    }

    /// How far from its Private block's first byte the guest pins a Private heap, a multiple of
    /// it (`kHeapAddressAlign`, `guest/bundle/src/RgfxSerializer.hpp`).
    const GUEST_HEAP_ALIGN: u64 = 0x40000;

    /// Private heaps pinned in a known Private block, a texture at offset 0 of each, of the most
    /// aligned of a few shapes on the GPU the test holds: in a heap pinned 32 KiB into the block,
    /// where the guest pinned its heaps before, Metal places it and rt refuses it where the image's
    /// alignment is more than 32 KiB, counted apart from Metal's refusals; in one pinned at a
    /// multiple of the guest's heap alignment from the block's first byte it is made.
    #[test]
    fn a_heap_pinned_in_a_block_places_textures_at_the_guest_s_alignment_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let heap = |desc: &HeapDesc| device.new_heap(desc).expect("never a raise").heap;
        let shape = |texture_type, pixel_format, side, sample_count| TextureDesc {
            texture_type,
            pixel_format,
            width: side,
            height: side,
            sample_count,
            usage: 4,
            resource_options: Some(0x20),
            ..TextureDesc::default()
        };
        // 4-sample Depth32Float and RGBA16Float targets, the shapes a GPU aligns most, and a
        // single-sample RGBA8Unorm one.
        let shapes = [
            shape(4, 252, 256, 4),
            shape(4, 115, 256, 4),
            shape(2, 70, 512, 1),
        ];
        let (texture, sized) = (shapes.iter())
            .filter_map(|desc| {
                let texture = platform.texture_descriptor(desc);
                let sized = device.heap_texture_size_and_align(&texture).ok()?;
                Some((texture, sized))
            })
            .max_by_key(|(_, sized)| (sized.align, std::cmp::Reverse(sized.size)))
            .expect("rt sizes a shape");
        let SizeAndAlign { size, align } = sized;
        assert!(
            align <= GUEST_HEAP_ALIGN,
            "the GPU aligns a texture to {align:#x} bytes: the guest's heap alignment is too small"
        );
        let size = size.next_multiple_of(HEAP_PAGE);
        let block = (device.rt()).take_block(rt::MemoryClass::Private, 0x80_0000);
        let block = block.expect("a Private block");
        device.known_blocks().know(block.clone());

        let metal = RtRefusal::Refused.code();
        let (places, metals) = (counted(HEAP_TEXTURE_PLACE), counted(metal));
        let pin = block.gpu_address + 0x8000;
        let off = heap(&placement(size, 0x20, Some(pin))).expect("a heap in the block");
        assert_eq!(off.gpu_address(), pin);
        let placed = off.new_texture(&texture, 0).expect("never a raise");
        if align > 0x8000 {
            assert!(placed.is_none(), "a {size:#x}-byte texture at {align:#x}");
            assert_eq!(counted(HEAP_TEXTURE_PLACE), places + 1);
        } else {
            assert!(placed.is_some(), "a {size:#x}-byte texture at {align:#x}");
            assert_eq!(counted(HEAP_TEXTURE_PLACE), places);
        }
        assert_eq!(counted(metal), metals);

        let pin = block.gpu_address + 11 * GUEST_HEAP_ALIGN;
        let at = heap(&placement(size, 0x20, Some(pin))).expect("a heap in the block");
        assert_eq!(at.gpu_address(), pin);
        let placed = at.new_texture(&texture, 0).expect("never a raise");
        let placed = placed.expect("a texture in the heap at the guest's alignment");
        assert_eq!(placed.resource_options(), 0x120);
        assert!(device.known_blocks().forget(block.id).is_some());
    }

    /// A heap's descriptor: its size rounded up to 16 KiB, Shared or Private, a default hazard
    /// tracking answered untracked, bits 20 and 24 answered without; Metal's refusals refused, and
    /// a sparse heap left to the unbuilt piece.
    #[test]
    fn a_heap_s_shape_is_metal_s() {
        let placement = desc(100, PLACEMENT);
        let shared = (rt::MemoryClass::Shared, HEAP_PAGE, true, 0x100, false);
        assert_eq!(shape(&placement, 0), Ok(Some(shared)));
        let private = (rt::MemoryClass::Private, 2 * HEAP_PAGE, false, 0x220, true);
        let automatic = desc(HEAP_PAGE + 1, AUTOMATIC);
        assert_eq!(
            shape(&automatic, 0x220 | BIT_20 | READ_ONLY),
            Ok(Some(private))
        );
        assert_eq!(shape(&desc(HEAP_PAGE, SPARSE), 0x20), Ok(None));
        for (d, held) in [
            (desc(0, PLACEMENT), 0),
            (desc(1, 3), 0),
            (desc(1, PLACEMENT), 0x10),
            (desc(1, PLACEMENT), 0x30),
            (desc(1, PLACEMENT), 0x2),
            (desc(1, PLACEMENT), 0x300),
            (desc(1, PLACEMENT), 1 << 30),
            (desc(u64::MAX, PLACEMENT), 0),
        ] {
            assert!(shape(&d, held).is_err(), "{d:?} {held:#x}");
        }
    }
}
