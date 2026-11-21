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
//! [`buffer::ZEROED_BYTES_CODE`]): rt says nothing of a block's bytes as it is taken. Any other
//! heap over the host's memory is a range of rt's import of its pages
//! (`super::buffer::import_pages`, counted as the no-copy buffers' imports are), its bytes as they
//! are; of option bit 24 (read-only), of a copy of them whose GPU writes are dropped
//! (`super::buffer`'s docs, "Read-only memory"). A pin neither keeps is counted
//! (`rt-gpu-address-unpinned`, `super::buffer`'s docs): the heap lies at rt's address. The heap's
//! size is the descriptor's rounded up to 16 KiB, as Metal's is; the heap holds its block.
//!
//! **Its deallocator.** A heap over the host's memory runs the deallocator its creator gave once
//! nothing reads or writes the bytes: as the heap goes where it lies in a known block, whose
//! mapping they are, and otherwise from rt's release of the import, once rt freed it, which may
//! be well after the heap went (a command buffer or a resource that used it holds the import
//! until it retires or goes). So what the deallocator holds keeps the bytes mapped for as long
//! as rt's import of them lives, as `import_host_memory`'s contract asks.
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
//! **Answers.** A heap's resource options are the asked ones without bits 20 and 24 (bit 24 counted
//! as `rt-no-copy-read-only` where the GPU's writes reach the heap's bytes: in a block of its own,
//! or pinned in a known block that drops no writes), a heap's default hazard tracking resolved to
//! untracked, as Metal's heaps answer it; its GPU address is its first byte's in the block.
//!
//! **Refusals.** What Metal refuses is nil, counted as Metal's refusal (`rt-refused-metal`): a
//! size of 0, an option value Metal has none of, a Managed or Memoryless heap, a heap over the
//! host's memory that is not Shared or that its bytes do not cover, a heap type Metal has none of;
//! a buffer or texture of a mode other than its heap's, in an automatic heap, at an offset off
//! 256 bytes (a buffer) or past the heap's end. What rt refuses is nil, counted under rt's kind; a
//! buffer of no bytes, which Metal places, is rt's refusal. A texture Metal places that rt refuses
//! on its place in the heap's block (a heap pinned at an offset of its block off the image's
//! alignment, which rt binds the image at) is counted apart from Metal's refusals
//! (`rt-refused-heap-texture-place`, [`super::unbuilt::HEAP_TEXTURE_PLACE`]). A sparse heap is
//! unbuilt ([`Unbuilt::Heap`]).
//!
//! **Where a heap texture lies.** The guest pins its Private heaps at a multiple of 256 KiB from
//! their block's first byte (`kHeapAddressAlign`, `guest/bundle/src/RgfxSerializer.hpp`), and rt
//! starts its 16 MiB blocks at a multiple of 2 MiB in their allocation, so a texture Metal places
//! there lies at the image's alignment wherever the image asks at most 256 KiB. A heap of its own
//! starts at a multiple of its length rounded up to a power of two (64 KiB at least, 2 MiB at
//! most), and no texture is aligned past the start of a heap of its own that holds it. The largest
//! alignment rt answers for a heap texture (`heapTextureSizeAndAlignWithDescriptor:`), over each
//! of the SDK's pixel formats and four usages, every texture type rt makes, 1, 2, 4 and 8 samples
//! and one level or every level (measured on 2026-11-20 at rt `d83fb05`): 256 KiB on an RX 7900
//! XTX (RADV, Mesa 26.2.6; a 4-sample texture) and on an RX 550 (RADV, Mesa 26.2.6; a 256³ 3D
//! texture); 64 KiB on a Radeon 780M (RADV, Mesa 26.2.6) and on an Iris Xe (ANV, Mesa 26.2.6; a
//! 2-sample texture); 16 KiB on an M3 Max (KosmicKrisp, Mesa 26.3.1); 1 KiB on an RTX 2080 and an
//! RTX 3090 (NVIDIA 620.70.06).
//! `a_heap_pinned_in_a_block_places_textures_at_the_guest_s_alignment_on_vulkan` measures it on
//! the GPU it holds, and fails where the GPU asks more than 256 KiB.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use metal2vulkan_runtime as rt;

use crate::core::num::ToUsize as _;
use crate::core::platform::Handle as _;
use crate::core::platform::device::NoCopyDeallocator;
use crate::core::platform::host_range::HostRange;
use crate::core::platform::resource::Heap;
use crate::core::platform::values::{HeapDesc, PagesAccess};
use crate::core::raise::RaiseResult;
use crate::core::sync::Mutex;

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

/// `id<MTLHeap>` on metal2vulkan-runtime: its range of a block and what the platform answers of it.
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
    /// The copy of host pages it lies over (`super::copied`), which a command buffer that uses it,
    /// or a resource of it, syncs; `None` for any other.
    copied: Option<Arc<super::copied::Copied>>,
    /// The deallocator of a heap over the host's memory that lies in a known block, which runs
    /// as the heap goes; `None` for any other (one over rt's import of the pages runs it from the
    /// import's release: the module's docs, "Its deallocator"). Behind a lock, as the heap is
    /// shared between threads and the deallocator is only `Send`.
    deallocator: Mutex<Option<NoCopyDeallocator>>,
}

impl Drop for RtHeap {
    fn drop(&mut self) {
        if let Some(deallocator) = self.deallocator.get_mut().take() {
            deallocator();
        }
    }
}

impl RtHeap {
    /// The copy of host pages it lies over, where it lies over one (`super::copied`).
    pub(super) fn copied(&self) -> Option<&Arc<super::copied::Copied>> {
        self.copied.as_ref()
    }

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
    if read_only && !super::copied::drops(block.id) {
        buffer::read_only(SEL);
    }
    Some(heap(device, block, offset, size, placement, options, None))
}

/// A block of `class` of the `size` bytes of a heap of its own, for `sel`, zeroed: a Shared one
/// through its mapping, counted by the byte, a Private one by rt's GPU fill. `None` where rt
/// refuses it, counted.
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
/// mapping there, else over rt's import of their pages (the module's docs), `deallocator` run
/// once nothing reads or writes them ("Its deallocator"). Nil where Metal or rt refuses it,
/// counted, `deallocator` then dropped unrun.
pub fn over_host_memory(
    device: &RtObj<RtDevice>,
    bytes: HostRange<'_>,
    desc: &HeapDesc,
    held: u64,
    deallocator: Option<NoCopyDeallocator>,
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
    let (block, offset, deallocator) = match pin.and_then(at) {
        Some((block, offset)) => (block, offset, deallocator),
        None => {
            if let Some(pin) = pin {
                unpinned(SEL, pin);
            }
            let access = match read_only {
                true => PagesAccess::ReadOnly,
                false => PagesAccess::Writable,
            };
            // rt's release of the import runs the deallocator, once rt freed it.
            let then = move || {
                if let Some(deallocator) = deallocator {
                    deallocator();
                }
            };
            let (block, offset) = buffer::import_pages(device, SEL, first, size, access, then)?;
            (block, offset, None)
        }
    };
    // A copy just made is the heap's to hold from here (`super::copied::Fresh`).
    let _fresh = super::copied::Fresh(block.id);
    if read_only && !super::copied::drops(block.id) {
        buffer::read_only(SEL);
    }
    Some(heap(
        device,
        block,
        offset,
        size,
        placement,
        options,
        deallocator,
    ))
}

/// The heap's handle: `size` bytes at `offset` in `block`, answering `options`, running
/// `deallocator` as it goes where given.
fn heap(
    device: &RtObj<RtDevice>,
    block: rt::Block,
    offset: u64,
    size: u64,
    placement: bool,
    options: u64,
    deallocator: Option<NoCopyDeallocator>,
) -> RtObj<RtHeap> {
    RtObj::new(RtHeap {
        offset,
        size,
        placement,
        options,
        device: device.clone_ref(),
        purgeable: texture::purgeable_state(),
        copied: super::copied::of(block.id),
        block,
        deallocator: Mutex::new(deallocator),
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
        let recorded_call = crate::recorder::rec_call!(
            "newBufferWithLength:options:offset:",
            self,
            length,
            options,
            offset
        );
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<Option<RtObj<RtBuffer>>> {
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
            let placed = (true, self.copied.clone());
            let buffer = buffer::buffer_of(&self.device, made, self.options, placed, None);
            Ok(Some(buffer))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
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
        let recorded_call = crate::recorder::rec_call!(
            "newTextureWithDescriptor:offset:",
            self,
            descriptor,
            offset
        );
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<Option<RtObj<RtTexture>>> {
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
                self.copied.as_ref(),
            ))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// The asked ones without bits 20 and 24, a default hazard tracking resolved to untracked.
    fn resource_options(&self) -> u64 {
        self.options
    }

    /// `super::texture::set_purgeable`'s.
    fn set_purgeable_state(&self, state: u64) -> RaiseResult<u64> {
        let recorded_call = crate::recorder::rec_call!("setPurgeableState:", self, state);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<u64> {
            Ok(texture::set_purgeable(&self.purgeable, state))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Bits 4 to 7 of its resource options.
    fn storage_mode(&self) -> u64 {
        (self.options >> 4) & 0xf
    }

    /// The copy of host pages it lies over (`super::copied`), where it lies over one.
    fn escaped_copy(&self) -> Option<crate::core::platform::command::EscapedCopy> {
        let copied = self.copied.as_ref()?;
        Some(Arc::clone(copied) as crate::core::platform::command::EscapedCopy)
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
    use crate::core::platform::values::{OptionSet, TextureDesc};
    use crate::core::testkit::DeallocatorWatch;
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

    /// A heap over host memory Metal or rt refuses on the GPU the test holds drops its deallocator
    /// unrun before the make returns, on the caller's thread ([`over_host_memory`]): a Private
    /// one, and one over fewer bytes than its size.
    #[test]
    fn a_refused_heap_over_host_memory_drops_its_deallocator_unrun_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let memory = vec![0_u8; 0x4000];
        let bytes = HostRange::of_slice(&memory);
        for (what, desc) in [
            ("Private", placement(0x4000, 0x20, None)),
            ("fewer bytes", placement(0x8000, 0, None)),
        ] {
            let watch = DeallocatorWatch::new();
            let made = device.new_heap_over_host_memory(bytes, &desc, Some(watch.deallocator()));
            assert!(made.expect("never a raise").heap.is_none(), "{what}");
            assert!(watch.dropped_unrun_here(), "{what}");
        }
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
        // SAFETY: the memory is never freed, so it stays allocated for the rest of the process,
        // as a heap over it with no deallocator asks.
        let bytes = unsafe { HostRange::new(HostPtr::from_ptr(memory), 0x10000) };
        let made = device.new_heap_over_host_memory(bytes, &placement(0x10000, 0, None), None);
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
        let made = device.new_heap_over_host_memory(bytes, &placement(0x4000, 0, Some(pin)), None);
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

    /// The guest's Private block, which its heaps are pinned in (`kPrivateBlockBytes`,
    /// `guest/bundle/src/RgfxObjectModel.hpp`).
    const GUEST_BLOCK: u64 = 16 << 20;

    /// The bytes a texture of the sweep ([`heap_shapes`]) may take at most, counted at 16 bytes a
    /// texel and sample: Vulkan's least `maxResourceSize`, 2 GiB.
    const SHAPE_BYTES: u64 = 1 << 31;

    /// The guest pins its heaps at the alignment, and in the blocks, the tests here check the
    /// GPU's answers against.
    #[test]
    fn the_guest_s_heap_alignment_and_block_are_the_tests() {
        let serializer = include_str!("../../../../guest/bundle/src/RgfxSerializer.hpp");
        let line = format!("kHeapAddressAlign = {GUEST_HEAP_ALIGN:#x};");
        assert!(serializer.contains(&line), "the serializer has no `{line}`");
        let object_model = include_str!("../../../../guest/bundle/src/RgfxObjectModel.hpp");
        let line = format!(
            "kPrivateBlockBytes = uint64_t{{{}}} << 20;",
            GUEST_BLOCK >> 20
        );
        assert!(
            object_model.contains(&line),
            "the object model has no `{line}`"
        );
    }

    /// The textures a heap can hold whose alignments the GPU's answers range over: of each of the
    /// SDK's pixel formats, Private, read by shaders, read and written, rendered to or of every
    /// usage (Metal's unknown usage), as 2D textures from 8192 to 16 texels a side, 2D multisample
    /// ones of 2, 4 and 8 samples alike, 2D and 2D multisample arrays, 1D textures and arrays,
    /// cubes and cube arrays and 3D textures, each of one level and, where it takes more, of every
    /// level; each of at most [`SHAPE_BYTES`].
    fn heap_shapes() -> Vec<TextureDesc> {
        let mut shapes = Vec::new();
        for &pixel_format in &crate::core::platform::caps::SDK_PIXEL_FORMATS {
            for usage in [0x1, 0x3, 0x5, 0] {
                let shape = |texture_type, (width, height, depth), sample_count, array_length| {
                    TextureDesc {
                        texture_type,
                        pixel_format,
                        width,
                        height,
                        depth,
                        sample_count,
                        array_length,
                        usage,
                        resource_options: Some(0x20),
                        ..TextureDesc::default()
                    }
                };
                let mut one = |d: TextureDesc| {
                    let levels = u64::from(64 - d.width.max(d.height).max(d.depth).leading_zeros());
                    shapes.push(d);
                    if d.sample_count == 1 && levels > 1 {
                        shapes.push(TextureDesc {
                            mipmap_level_count: levels,
                            ..d
                        });
                    }
                };
                for side in [8192, 4096, 1024, 256, 16] {
                    one(shape(2, (side, side, 1), 1, 1));
                    for sample_count in [2, 4, 8] {
                        one(shape(4, (side, side, 1), sample_count, 1));
                    }
                }
                one(shape(3, (1024, 1024, 1), 1, 64));
                for sample_count in [2, 4, 8] {
                    one(shape(8, (1024, 1024, 1), sample_count, 4));
                }
                for width in [16384, 16] {
                    one(shape(0, (width, 1, 1), 1, 1));
                    one(shape(1, (width, 1, 1), 1, 16));
                }
                for side in [4096, 256] {
                    one(shape(5, (side, side, 1), 1, 1));
                    one(shape(6, (side, side, 1), 1, 2));
                }
                for side in [512, 256, 16] {
                    one(shape(7, (side, side, side), 1, 1));
                }
            }
        }
        let faces = |d: &TextureDesc| {
            if matches!(d.texture_type, 5 | 6) {
                6
            } else {
                1
            }
        };
        shapes.retain(|d| {
            let texels = d.width * d.height * d.depth * d.array_length * faces(d);
            texels * d.sample_count * 16 <= SHAPE_BYTES
        });
        shapes
    }

    /// Each shape of [`heap_shapes`] that rt sizes for a heap on `device`, with rt's answer, as
    /// `heapTextureSizeAndAlignWithDescriptor:` answers the guest; and how many rt sizes none of,
    /// by the kind of its refusal ([`diag::refusal_name`]), with the formats of each.
    fn sized_shapes(device: &RtObj<RtDevice>) -> (Vec<(TextureDesc, rt::SizeAndAlign)>, Unsized) {
        let mut sized = Vec::new();
        let mut not = Unsized::new();
        for desc in heap_shapes() {
            let answer = super::super::descriptor::rt_texture_descriptor(&desc)
                .map_err(|_| "descriptor".to_owned())
                .and_then(|d| {
                    (device.rt().heap_texture_size_and_align(&d))
                        .map_err(|e| diag::refusal_name(&e).unwrap_or_else(|| format!("{e:?}")))
                });
            match answer {
                Ok(answer) => sized.push((desc, answer)),
                Err(kind) => {
                    let (n, formats) = not.entry(kind).or_default();
                    *n += 1;
                    formats.insert(desc.pixel_format);
                }
            }
        }
        (sized, not)
    }

    /// The shapes rt sizes none of, by the kind of its refusal: how many, and their formats.
    type Unsized = std::collections::BTreeMap<String, (usize, std::collections::BTreeSet<u64>)>;

    /// Every alignment the GPU the test holds asks of a heap texture, over the shapes of
    /// [`heap_shapes`], is at most the guest's heap alignment, and the guest's blocks lie in rt's
    /// memory at a multiple of the largest: a texture of each alignment, the smallest shape that
    /// asks it, at offset 0 and at the alignment of a Private heap pinned in a known Private block
    /// at a multiple of the guest's heap alignment from the block's first byte (where the guest
    /// pins its heaps), and of a heap of its own (where a heap the guest pins outside its blocks
    /// lies), is made. In a heap pinned 32 KiB into the block, where the guest pinned its heaps
    /// before, Metal places the texture at offset 0 and rt refuses one aligned to more than 32 KiB,
    /// counted apart from Metal's refusals. The test's output lists each alignment asked and the
    /// smallest shape that asks it.
    #[test]
    fn a_heap_pinned_in_a_block_places_textures_at_the_guest_s_alignment_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let heap = |desc: &HeapDesc| device.new_heap(desc).expect("never a raise").heap;
        let (sized, not) = sized_shapes(&device);
        let mut asked: std::collections::BTreeMap<u64, (TextureDesc, u64)> = Default::default();
        for &(desc, rt::SizeAndAlign { size, align }) in &sized {
            let smallest = asked.entry(align).or_insert((desc, size));
            if size < smallest.1 {
                *smallest = (desc, size);
            }
        }
        let name = &device.facts().name;
        println!(
            "{name}: {} shapes sized of {}",
            sized.len(),
            heap_shapes().len()
        );
        for (kind, (n, formats)) in &not {
            println!("{name}: {n} not sized, {kind}: formats {formats:?}");
        }
        for (align, (d, size)) in &asked {
            println!(
                "{name}: align {align:#x}: format {} type {} {}x{}x{} levels {} samples {} \
                 layers {} usage {:#x}, {size:#x} bytes",
                d.pixel_format,
                d.texture_type,
                d.width,
                d.height,
                d.depth,
                d.mipmap_level_count,
                d.sample_count,
                d.array_length,
                d.usage
            );
        }
        let worst = *asked.keys().last().expect("rt sizes a shape");
        assert!(
            worst <= GUEST_HEAP_ALIGN,
            "{name} aligns a heap texture to {worst:#x} bytes: the guest's heap alignment is too small"
        );
        // A heap of its own is a block's first byte, which rt places at a multiple of its length
        // rounded up to a power of two (64 KiB at least, 2 MiB at most) in its allocation.
        let past: Vec<_> = (sized.iter())
            .filter(|(_, s)| s.align > s.size.next_power_of_two().max(0x10000))
            .collect();
        assert!(past.is_empty(), "aligned past their size: {past:?}");

        let metal = RtRefusal::Refused.code();
        let (places, metals) = (counted(HEAP_TEXTURE_PLACE), counted(metal));
        let mut refused = 0;
        for (&align, (desc, size)) in &asked {
            let texture = platform.texture_descriptor(desc);
            let size = (align + size).next_multiple_of(HEAP_PAGE);
            let pinned = 11 * GUEST_HEAP_ALIGN;
            let len = GUEST_BLOCK.max(pinned + size);
            let block = (device.rt()).take_block(rt::MemoryClass::Private, len);
            let block = block.expect("a Private block");
            assert!(
                block.offset.is_multiple_of(worst),
                "a block of {len:#x} bytes at {:#x} in its allocation",
                block.offset
            );
            device.known_blocks().know(block.clone());
            let what = format!("a {size:#x}-byte heap's texture at {align:#x}");

            let off = heap(&placement(size, 0x20, Some(block.gpu_address + 0x8000)));
            let off = off.expect("a heap in the block");
            let placed = off.new_texture(&texture, 0).expect("never a raise");
            assert_eq!(placed.is_some(), align <= 0x8000, "{what}, 32 KiB in");
            refused += u64::from(align > 0x8000);

            let at = heap(&placement(size, 0x20, Some(block.gpu_address + pinned)));
            let at = at.expect("a heap in the block");
            assert_eq!(at.gpu_address(), block.gpu_address + pinned);
            let own = heap(&placement(size, 0x20, None)).expect("a heap of its own");
            for (placed_in, offset) in [(&at, 0), (&at, align), (&own, 0), (&own, align)] {
                let placed = placed_in
                    .new_texture(&texture, offset)
                    .expect("never a raise");
                let placed = placed.unwrap_or_else(|| panic!("{what}, placed at {offset:#x}"));
                assert_eq!(placed.resource_options(), 0x120);
            }
            assert!(device.known_blocks().forget(block.id).is_some());
        }
        assert_eq!(counted(HEAP_TEXTURE_PLACE), places + refused);
        assert_eq!(counted(metal), metals);
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
