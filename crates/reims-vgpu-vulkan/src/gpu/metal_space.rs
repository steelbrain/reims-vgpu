//! Metal's GPU address space as this process's Metal lays it out: the model this platform
//! places guest-visible buffers, heaps and textures by, so that every address the guest reads (a
//! placement entry's first word, the replies to the info commands 461 and 463, the buffer
//! addresses an indirect command buffer's write-back holds) is the one Metal hands out. The
//! runtime's own choice starts at 2^32 and packs ranges page after page, which no recording
//! shows.
//!
//! Every buffer and heap a creator of the platform makes that Metal gives a range of its own takes
//! its range here and is made at it (`AddressPlacement::At`): plain ones where the space places
//! them, the guest's pins where it pins them ([`placing`]). One the space has no range for is
//! made at an address the runtime refuses as it checks the address, after the rest of the
//! descriptor ([`REFUSED`]). A buffer Metal suballocates from its pools takes a slot in one and is
//! made at it, in the runtime's reservation of the pool ([`buffer_space`],
//! [`Placing::pool_slot`]). A buffer placed in a heap takes none, and neither do the platform's
//! own helper buffers, which lie outside Metal's space. Textures take theirs as Metal's do
//! ([`texture_space`], [`surface`]).
//! What Metal makes for its own work takes ranges too, at the first use of each kind of work
//! ([`FirstUse`]). A buffer's or heap's range stays held until the runtime frees its own range of
//! the resource, which drops the runtime's range token for it, a texture's over a surface until
//! the runtime frees the buffer it is over too, and any other texture's while the platform's
//! object lives ([`Hold`]).
//!
//! One model per process ([`process`]): Metal has one device per GPU, so every object of the
//! process takes its range after those made before it, whoever made them.
//!
//! What the probes (`vaprobe`, plain Metal) and the recordings of the suite's isolate cases
//! settle, on the M3 Max, macOS 26.7 (the table test plays every row):
//! - **The start.** The device holds a range at [`BASE`] from its creation, and its first queue
//!   or texture makes Metal's pools after it, so a guest's first buffer or heap is at [`START`]
//!   in every case (`compute_pipeline_placement`, `info_replies`, `icb_sync`). Shader
//!   libraries, pipeline states, more queues, events, samplers, depth-stencil states, rate maps
//!   and indirect command buffers take none (`vaprobe bringup`, `vaprobe textures`).
//! - **Ranges.** A range holds whole 16-KiB pages and one guard page after them, rounded up to,
//!   and starting on, a 32-KiB boundary ([`SLOT`], [`MetalSpace::place`]): `vaprobe sizes`, for
//!   buffers of every storage mode, no-copy ones and heaps, of 1 to 64 pages.
//! - **Reuse.** A released range is handed out again, the lowest free one that holds the new
//!   range first, split where it is larger ([`MetalSpace::place`]): `vaprobe reuse` and
//!   `vaprobe order`, and `resources_gap_placement_delete`, whose buffer deleted and made again
//!   takes its old address.
//! - **Pins.** A buffer or heap the guest pins (its placement entry's first word) holds what a
//!   plain range of its pages holds, from its first page: the pages its bytes touch and one
//!   guard page, rounded up to 32 KiB (`vaprobe pins pin3-10000` and `pin3-18000`: after a
//!   three-page pin, pins at +0x10000 and +0x18000 are made). Its first page starts a 32-KiB
//!   granule, or Metal makes nothing: every pin whose first page is a granule's second was nil,
//!   even far from every range (`far-4000`, `far-4010`, `guard3c`, `guard3c-bringup`,
//!   `pin3-c000`, `pin3-14000`), and one 0x10 bytes into a granule's first page is made at its
//!   address (`far-10`). A pin is refused where its span meets a held range (`vaprobe pins`: a
//!   plain range's guard page and the page rounding adds are held, the slot after it is free).
//!   A released pin frees its whole span (measured on an M4: a pin at
//!   its first page is made again at once). Pins far from every range are made
//!   (`textures_placement_forced`, `mutgap2_res_icb_bind_at_buffer_end`).
//! - **Pools.** Metal suballocates buffers of their own memory under a page from pools, a class
//!   for each storage mode, and textures of their own memory whose heap size is under a page
//!   alike ([`OWN_LENGTH`], [`OWN_HEAP_SIZE`]): `vaprobe small`, `vaprobe smallpool`, `vaprobe
//!   texsizes`, `vaprobe texpool2` and `vaprobe smalltex`. The shared and private pools take
//!   their ranges at the first queue or texture, or at the first such buffer where none came
//!   before ([`FirstUse`]), and the managed one at its first slot; a full one makes another,
//!   which takes a range ([`PoolClass`]).
//! - **Descriptors.** Every texture takes a descriptor in Metal's mempool, whose arrays take slots
//!   in the shared pool, then ranges, as the textures grow in number ([`Elements`]); every ID
//!   of the device's sampler ID pool one in another, which grows alike as the pool doubles, the
//!   sampler states' and those pipeline states number ([`Samplers`]).
//! - **Textures.** A texture of its own memory of a page or more holds a range of its heap size,
//!   `heapTextureSizeAndAlignWithDescriptor:`'s answer, and a shareable one of its texels and
//!   [`SHAREABLE_EXTRA`] (`vaprobe texsizes` and `vaprobe shared`). The answer is Metal's as the
//!   runtime models it, on every device, where the device's own may be larger: a depth or
//!   stencil one without GPU-optimized contents, and any on NVIDIA ([`own_heap_size`]).
//! - **Metal's own work.** Its first command queue, compute encoder, blit command, render
//!   encoder and tiler submission each make buffers and heaps of Metal's own, once per process
//!   ([`FirstUse`]), which lie among the guest's ranges. How they grow with more work is not
//!   modelled.

mod gaps;

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;
use std::sync::{Arc, OnceLock, Weak};

use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::types::SizeAndAlign;
use reims_vgpu_core::sync::{Mutex, MutexGuard};

/// Metal's page on this host: a range holds whole pages.
pub const PAGE: u64 = 0x4000;

/// The boundary a plain range starts on, and the multiple its pages and guard pages are rounded
/// up to.
pub const SLOT: u64 = 0x8000;

/// Where the first plain range lies once the device has a queue ([`FirstUse::Queue`]): every
/// isolate case's first buffer or heap.
pub const START: u64 = 0x100_0008_0000;

/// The end of the space: the runtime's Metal space ends here.
pub const END: u64 = 1 << 47;

/// A kind of texture, as what it takes of the space depends on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureKind {
    /// Over a buffer's memory.
    OverBuffer,
    /// In a heap.
    InHeap,
    /// A view of another texture: it shares that texture's hold ([`Hold`]).
    View,
    /// Memoryless.
    Memoryless,
    /// Of memory of its own, made from a descriptor (core's paged textures) ([`own_memory`]).
    OwnMemory {
        /// Its bytes in a heap, `heapTextureSizeAndAlignWithDescriptor:`'s answer for its
        /// descriptor.
        heap_size: u64,
        /// The pool class of its storage, which decides the pool of one Metal suballocates
        /// ([`PoolClass`]).
        class: PoolClass,
    },
    /// A shareable texture (`newSharedTextureWithDescriptor:`) ([`shared_backing`]).
    SharedBacking {
        /// Its bytes by Apple's layout ([`texture_allocation`]).
        allocation: u64,
    },
    /// Opened from another texture's shared handle: it takes none of its own and shares that
    /// texture's, which it keeps while it lives, past the texture and its handle (`vaprobe
    /// sharedrelease`) ([`Hold`]).
    FromSharedHandle,
    /// Over an IOSurface: the surface's range ([`surface`]).
    OverSurface,
}

/// What a texture takes of the space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureSpace {
    /// None of it: a texture over a buffer or in a heap, a view, a memoryless one, and one opened
    /// from a shared handle.
    None,
    /// A plain range of that many bytes, its own.
    Own(u64),
    /// A slot of that many bytes in one of Metal's pools ([`MetalSpace::slot`]): one Metal
    /// suballocates.
    Slot {
        /// Its pool class ([`PoolClass`]).
        class: PoolClass,
        /// Its bytes, before the slot's rounding.
        bytes: u64,
    },
    /// Its surface's range, shared with every texture and buffer over the surface
    /// ([`surface`]).
    Surface,
}

/// The least heap size of a texture of memory of its own Metal gives a range of its own, of that
/// many bytes: it suballocates smaller ones, of every format, from its pools ([`PoolClass`])
/// (`vaprobe texsizes`, private, after a bring-up: 4x4 Depth32Float, Depth16Unorm and Stencil8
/// textures of heap size 0x80 and a 32x64 RGBA8 one of 0x2000 took none; a 63x63 RGBA8 one of
/// 0x4000 took 0x8000; a 4x4 Depth32Float_Stencil8 one of 0x4080, and 64x64 RGBA8 ones of 0x4080
/// read or a render target and of 0x5680 with mipmaps, 0x10000; 128x128 RGBA8 of 0x10200
/// 0x18000, and 256x256 of 0x40800 0x48000. `vaprobe textures`: 4x4 RGBA8, private, shared or a
/// render target, took none). One takes a slot of its heap size in the pool of its storage's
/// class, as a small buffer does (`vaprobe texpool2`: private 4x4 RGBA8 0x100, 16x16 0x400,
/// 32x64 0x2000; `vaprobe smalltex`: 4x4 RGBA8 of shared storage in the shared pool, of managed
/// storage in a managed pool, of private storage in the private pool, 0x100 each). Metal's
/// descriptors of every texture take arrays of their own, which grow with the textures
/// ([`Elements`]).
pub const OWN_HEAP_SIZE: u64 = PAGE;

/// The bytes Metal allocates for a texture of this shape, by the runtime's model of Apple's layout
/// (its `attributes.rs`): for each level, its extent in blocks of `block_extent` texels,
/// each axis rounded up to a power of two (the depth only for a `volume`, a 3D texture), times
/// `block_bytes`; summed over the `levels`, times the `layers` (a cube's faces each) and the
/// `samples`. Apple's own sizes (`allocatedSize`) differ from it by compression metadata and
/// alignment (a 4x4 RGBA8 texture's is 0x80 where this is 0x40).
pub fn texture_allocation(
    block_extent: [u64; 2],
    block_bytes: u64,
    extent: [u64; 3],
    volume: bool,
    levels: u64,
    layers: u64,
    samples: u64,
) -> u64 {
    let blocks = |texels: u64, block: u64| texels.div_ceil(block.max(1)).max(1);
    let level_bytes = (0..levels.clamp(1, 64)).fold(0u64, |sum, level| {
        let [w, h, d] = extent.map(|axis| (axis >> level).max(1));
        let depth = if volume { d } else { 1 };
        let bytes = blocks(w, block_extent[0])
            .checked_next_power_of_two()
            .and_then(|b| b.checked_mul(blocks(h, block_extent[1]).checked_next_power_of_two()?))
            .and_then(|b| b.checked_mul(depth.checked_next_power_of_two()?))
            .and_then(|b| b.checked_mul(block_bytes))
            .unwrap_or(u64::MAX);
        sum.saturating_add(bytes)
    });
    level_bytes
        .saturating_mul(layers.max(1))
        .saturating_mul(samples.max(1))
}

/// The bytes of a texture of `desc` by Apple's layout ([`texture_allocation`]), for `sel`, the
/// call that asks, from Metal's description of its format ([`crate::gpu::format_info::block`]).
fn allocation_of(sel: &'static str, desc: &rt::TextureDescriptor) -> u64 {
    use metal2vulkan::metal::TextureType;
    let raw = u64::from(desc.pixel_format.as_raw());
    let block = crate::gpu::format_info::block(sel, raw);
    let layers = match desc.texture_type {
        TextureType::Cube | TextureType::CubeArray => desc.array_length.max(1).saturating_mul(6),
        _ => desc.array_length,
    };
    texture_allocation(
        block.extent(),
        block.bytes(),
        [desc.width, desc.height, desc.depth],
        desc.texture_type == TextureType::Type3D,
        desc.mipmap_level_count,
        layers,
        desc.sample_count,
    )
}

/// A texture of memory of its own of `desc`, made by `sel`, its heap size `heap_size`: Metal's
/// answer to `heapTextureSizeAndAlignWithDescriptor:` for `desc` ([`own_heap_size`]), which the
/// device's tests compare with Metal's, or, where none is answered, its bytes by Apple's layout
/// ([`texture_allocation`]).
pub fn own_memory(
    sel: &'static str,
    desc: &rt::TextureDescriptor,
    heap_size: Option<u64>,
) -> TextureKind {
    TextureKind::OwnMemory {
        heap_size: heap_size.unwrap_or_else(|| allocation_of(sel, desc)),
        class: PoolClass::of(desc.resource_options.storage_mode().ok()),
    }
}

/// The heap size Metal's space takes for a texture of its own memory of `desc`
/// ([`own_memory`]): `metal`, the runtime's model of Metal's answer to
/// `heapTextureSizeAndAlignWithDescriptor:` for it
/// (`crate::gpu::device::VulkanDevice::metal_answer_at`), where the model has one. That is
/// Metal's on every device, where the device's own answer, which a heap places the texture by,
/// may be larger (`heap-texture-larger-than-metal`: every texture on NVIDIA, whose
/// `bufferImageGranularity` is 1024 bytes) or is for a depth or stencil texture without
/// GPU-optimized contents (`optimized-depth-stencil-texture`), each of which the model answers.
/// Where the model has none (a format or a sample count whose layout is not measured), of `answer`,
/// the runtime's answer to the query for `desc`, which the runtime answers as its device's image's,
/// larger or smaller than Metal's, and announces so (`unmeasured-heap-texture-size`,
/// [`crate::answer::HEARD`]: 8 samples, which are refused first), at which a device's texture
/// dies ([`unmeasured_size`]).
///
/// The space takes a range or a slot by that size as for any texture ([`texture_space`]):
/// `vaprobe depthsizes` took a slot of the size rounded up to [`SLOT_STEP`] in the private pool
/// for each of its textures under a page, and a range of it for each of a page or more. The
/// runtime's image lies in none of Metal's space, so it takes no more of it.
pub fn own_heap_size(
    desc: &rt::TextureDescriptor,
    metal: Option<SizeAndAlign>,
    answer: impl Fn(&rt::TextureDescriptor) -> Option<SizeAndAlign>,
) -> Option<u64> {
    metal.or_else(|| answer(desc)).map(|answer| answer.size)
}

/// The relay at `sel`, or its death in strict mode ([`crate::answer::inexact`]), of `what` a
/// texture of `desc` whose layout in Metal's heaps the runtime does not model, which `sel` heard
/// the runtime announce (`unmeasured-heap-texture-size`, [`crate::answer::HEARD`]: 8 samples,
/// which its descriptor check refuses first on every device), whose size and alignment the
/// runtime answers as its device's image's, larger or smaller than Metal's, and the call goes on
/// with them. The line names the texture.
pub fn unmeasured_size(sel: &'static str, what: &str, desc: &rt::TextureDescriptor) {
    crate::answer::inexact(sel, crate::answer::Inexact::HeapLayout, || {
        format!(
            "{what} a texture whose layout in Metal's heaps the runtime does not model, which it \
             sizes as its device's image, larger or smaller than Metal's \
             (unmeasured-heap-texture-size): a {:?} texture of pixel format {}, {}x{}x{}, {} \
             levels, {} samples, {} slices",
            desc.texture_type,
            desc.pixel_format.as_raw(),
            desc.width,
            desc.height,
            desc.depth,
            desc.mipmap_level_count,
            desc.sample_count,
            desc.array_length
        )
    });
}

/// A shareable texture of `desc`, made by `sel`.
pub fn shared_backing(sel: &'static str, desc: &rt::TextureDescriptor) -> TextureKind {
    TextureKind::SharedBacking {
        allocation: allocation_of(sel, desc),
    }
}

/// The bytes a shareable texture's allocation holds past its texels, which its range holds
/// (`vaprobe shared`, BGRA8, private: `allocatedSize` 0x410 for 16x16, 0x1010 32x32, 0x4010 63x64
/// and 64x64, 0x8010 65x64, 0x10010 128x128 and 0x40010 256x256, each the texels by Apple's
/// layout and these bytes, and a range of that span each; 0x90 for 4x4, whose 0x80 bytes of
/// texels the model takes as 0x40, which holds the same range).
pub const SHAREABLE_EXTRA: u64 = 0x10;

/// What a texture of `kind` takes of the space.
pub const fn texture_space(kind: TextureKind) -> TextureSpace {
    match kind {
        TextureKind::OverBuffer
        | TextureKind::InHeap
        | TextureKind::View
        | TextureKind::Memoryless
        | TextureKind::FromSharedHandle => TextureSpace::None,
        TextureKind::OverSurface => TextureSpace::Surface,
        TextureKind::OwnMemory { heap_size, class } if heap_size < OWN_HEAP_SIZE => {
            TextureSpace::Slot {
                class,
                bytes: heap_size,
            }
        }
        TextureKind::OwnMemory { heap_size, .. } => TextureSpace::Own(heap_size),
        TextureKind::SharedBacking { allocation } => {
            TextureSpace::Own(allocation.saturating_add(SHAREABLE_EXTRA))
        }
    }
}

/// Where Metal's space begins on this host: the heap Metal makes with its device lies here, a
/// range of 0x18000 (`vaprobe bringup`: the device alone, then a buffer at 0x100_0001_8000).
pub const BASE: u64 = 0x100_0000_0000;

/// The bytes of the heap Metal makes with its device, at [`BASE`].
const DEVICE_HEAP: u64 = 0x1_0000;

/// A use of the device after which Metal has made its own allocations for it, once per process
/// ([`MetalSpace::first_use`]). Each takes the allocations [`allocations`] lists that no earlier
/// use took, in that order, as plain ranges held for good. Read from the VA traces of the probes
/// and of the suite's cases (`decode2.py` over `vatrace`'s calls), on the M3 Max, macOS 26.7.
///
/// Not modelled: Metal's own buffers grow with the work it records, past what a first use takes
/// (another compute or blit data buffer of 0xa980 bytes, another blit dispatch buffer of 0x400,
/// another heap of 0x40000 for its programs once the first is full), which no simple function of
/// the uses gives: `interplay_decodewalk_42` (0x10000), `interplay_decodewalk_52` (0x60000) and
/// `mutgap2_res_placement_forced_used` (0x48000) place their later objects past those.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirstUse {
    /// A command queue: Metal's pools for small shared and private buffers, and a heap, 0x68000
    /// of the space (`vaprobe bringup`: three queues; `vaprobe cmdbuf`: one). The library's
    /// device makes its queues as it starts, so these lie below [`START`].
    Queue,
    /// A texture of memory of its own, suballocated or not, a shareable one, or one over a
    /// surface: the same pools as [`FirstUse::Queue`] (`vaprobe textures` and `vaprobe
    /// surfaces`, a texture before any queue).
    Texture,
    /// A compute encoder made: six of the command pool's buffers and a data buffer, 0x70000
    /// (`vaprobe bringup`).
    ComputeEncoder,
    /// A blit encoder's first command (an empty blit encoder takes none): seven of the command
    /// pool's buffers, a data buffer and a dispatch buffer, 0x88000 (`vaprobe cmdbuf`'s fill,
    /// `info_reply_before_error`, `textures_placement_shared`'s copy).
    BlitCommand,
    /// A render encoder made: the heap of Metal's programs, nine of the command pool's buffers
    /// and two buffers of its own, 0x2d8000 where it is the first encoder
    /// (`mutgap2_rdec_use_heaps_count_placement`), less what other encoders took before it
    /// (`vaprobe cmdbuf`: 0x278000 after blit fills; `vaprobe bringup`: 0x2a8000 after a compute
    /// encoder).
    RenderEncoder,
    /// A blit's reset or copy of an indirect command buffer, which Metal runs on its tiler: the
    /// heap of Metal's programs (`interplay_decodewalk_42` and `_52`).
    TilerBlit,
    /// A command buffer's work for the tiler (a render encoder, or a [`FirstUse::TilerBlit`])
    /// submitted to the GPU: 0x208000 that no call the trace sees makes (`vaprobe cmdbuf`, after
    /// the render pass completed; `interplay_decodewalk_42`, `_52`,
    /// `mutgap2_res_placement_forced_used`). Metal takes it as it submits the work, not as the
    /// command buffer is committed (`vaprobe rendersubmit`: none of it at the commit of a render
    /// pass behind an event wait, all of it by the command buffer's completion). So the platform
    /// takes it in the command buffer's submission handler, which the runtime runs as it submits
    /// the command buffer (`command::tiler_handler`).
    TilerSubmission,
    /// A buffer of shared storage Metal suballocates ([`buffer_space`]): its pool,
    /// where no queue made it before (`vaprobe sizes`: a 1-byte buffer in a process of its
    /// device alone took 0x28000 and lay at the pool's first byte).
    SmallShared,
    /// A buffer of private storage Metal suballocates: its pool. Before any queue, unmeasured:
    /// the model takes the pool alone, which the library's queues, made as it starts, make moot.
    SmallPrivate,
}

/// One of Metal's own allocations ([`FirstUse`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Own {
    /// The buffer Metal suballocates small shared buffers from (0x20000 bytes; `vaprobe small`:
    /// shared buffers of 1 to 0x3fff bytes at 0x100-byte steps from 0x100_0001_c500).
    SharedPool,
    /// A heap of Metal's own (0x10000 bytes).
    PoolHeap,
    /// The buffer Metal suballocates small private buffers from (0x20000 bytes; `vaprobe small`:
    /// private buffers of 1 to 0x3fff bytes from 0x100_0005_8100).
    PrivatePool,
    /// One of the command pool's buffers (`IOGPUMetalResourcePool`, 0x8000 bytes each), by
    /// [`streams`]' index.
    Command(u8),
    /// A compute encoder's data buffer (0xa980 bytes).
    ComputeData,
    /// A blit encoder's data buffer (0xa980 bytes).
    BlitData,
    /// A blit encoder's dispatch buffer (0x400 bytes).
    BlitDispatch,
    /// The heap Metal loads its own programs into (0x74000 bytes).
    ProgramHeap,
    /// A render encoder's buffers of 0x100000 and 0xc0000 bytes.
    RenderPool,
    /// See [`Own::RenderPool`].
    RenderData,
    /// What a first tiler submission takes: 0x208000 of the space with its guard page, made
    /// by no call the trace sees, so its bytes are unseen; the model holds the span.
    TilerWork,
}

impl Own {
    /// Its bytes.
    const fn bytes(self) -> u64 {
        match self {
            Own::SharedPool | Own::PrivatePool => 0x2_0000,
            Own::PoolHeap => 0x1_0000,
            Own::Command(_) => 0x8000,
            Own::ComputeData | Own::BlitData => 0xa980,
            Own::BlitDispatch => 0x400,
            Own::ProgramHeap => 0x7_4000,
            Own::RenderPool => 0x10_0000,
            Own::RenderData => 0xc_0000,
            // The span less its guard page ([`Own::TilerWork`]).
            Own::TilerWork => 0x20_8000 - PAGE,
        }
    }
}

/// The command pool's buffers by which encoder kinds use them: twelve, of which compute encoders
/// use six, blit commands seven and render encoders nine, shared so that a blit command after a
/// compute encoder takes three more, a compute encoder after a blit command two, a render
/// encoder after blit commands three and after a compute encoder six, and a blit command after a
/// render encoder one. One assignment gives every count measured (`vaprobe bringup`, `vaprobe
/// cmdbuf`, `interplay_decodewalk_42`, `_44`, `_52`, `mutgap2_rdec_use_heaps_count_placement`,
/// `mutgap2_res_placement_forced_used`): three that all three kinds use, three that blit and
/// render encoders use, one that blit and compute encoders use, two of compute encoders alone and
/// three of render encoders alone. The order within a kind is unmeasured where it lands no
/// differently; render encoders take four of theirs before [`Own::RenderPool`] and five after
/// (`vaprobe bringup`: two then four after a compute encoder).
mod streams {
    use super::Own;
    /// Used by every kind.
    pub const ALL: [Own; 3] = [Own::Command(0), Own::Command(1), Own::Command(2)];
    /// Used by blit and render encoders.
    pub const BLIT_RENDER: [Own; 3] = [Own::Command(3), Own::Command(4), Own::Command(5)];
    /// Used by blit and compute encoders.
    pub const BLIT_COMPUTE: Own = Own::Command(6);
    /// Used by compute encoders alone.
    pub const COMPUTE: [Own; 2] = [Own::Command(7), Own::Command(8)];
    /// Used by render encoders alone.
    pub const RENDER: [Own; 3] = [Own::Command(9), Own::Command(10), Own::Command(11)];
}

/// Metal's own allocations a first `use` makes, in order ([`FirstUse`]).
fn allocations(first: FirstUse) -> Vec<Own> {
    use streams::*;
    match first {
        FirstUse::Queue | FirstUse::Texture => {
            vec![Own::SharedPool, Own::PoolHeap, Own::PrivatePool]
        }
        FirstUse::ComputeEncoder => [
            ALL.as_slice(),
            &[BLIT_COMPUTE],
            &COMPUTE,
            &[Own::ComputeData],
        ]
        .concat(),
        FirstUse::BlitCommand => [
            ALL.as_slice(),
            &BLIT_RENDER,
            &[BLIT_COMPUTE, Own::BlitData, Own::BlitDispatch],
        ]
        .concat(),
        FirstUse::RenderEncoder => [
            [
                Own::ProgramHeap,
                BLIT_RENDER[0],
                BLIT_RENDER[1],
                ALL[0],
                ALL[1],
            ]
            .as_slice(),
            &[Own::RenderPool, BLIT_RENDER[2], ALL[2]],
            RENDER.as_slice(),
            &[Own::RenderData],
        ]
        .concat(),
        FirstUse::TilerBlit => vec![Own::ProgramHeap],
        FirstUse::TilerSubmission => vec![Own::TilerWork],
        FirstUse::SmallShared => vec![Own::SharedPool],
        FirstUse::SmallPrivate => vec![Own::PrivatePool],
    }
}

/// The bytes of the space one resource holds, from its creation to its release
/// ([`MetalSpace::release`]).
#[derive(Debug, PartialEq, Eq)]
pub struct Held {
    /// Its address: what `gpuAddress` answers.
    address: u64,
    /// Its pages.
    pages: Range<u64>,
    /// The guard pages after them: its guard page, and the page rounding adds.
    guard: Range<u64>,
    /// Whether the guest pinned it, which Metal answers nil for where it cannot honour the pin
    /// ([`Placing::nil_at_conflict`]).
    pinned: bool,
}

impl Held {
    /// Its address: what `gpuAddress` answers.
    pub const fn address(&self) -> u64 {
        self.address
    }

    /// The pages it holds.
    pub fn pages(&self) -> Range<u64> {
        self.pages.clone()
    }

    /// The guard pages after them.
    pub fn guard(&self) -> Range<u64> {
        self.guard.clone()
    }
}

/// Why a pin is refused, where Metal makes nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PinRefused {
    /// Its span meets a held range; `held` is the first held byte it meets.
    Held {
        /// The span the pin needs: its pages and guard pages.
        requested: Range<u64>,
        /// The first held byte it meets.
        held: u64,
    },
    /// Its first page is the second of a 32-KiB granule ([`SLOT`]).
    OffGranule {
        /// The span the pin needs.
        requested: Range<u64>,
    },
    /// Its end passes the end of the 64-bit space.
    Outside,
}

/// The bytes of each of Metal's pools of small allocations ([`PoolClass`]): a buffer Metal makes
/// of its own, a plain range of the space (`vaprobe smallpool`: buffers of 0x20000 bytes, each a
/// range of 0x28000).
pub const POOL_BYTES: u64 = 0x2_0000;

/// The step of a pool's slots: each slot's offset and bytes are a multiple of it (`vaprobe
/// smallpool`: buffers of 1 to 0x100 bytes took 0x100, of 0x101 to 0x200 0x200, of 0x300 0x300,
/// of 0x1001 0x1100 and of 0x3fff 0x4000, each at the step after the one before, a buffer of
/// 0x3000 bytes at 0x100_0002_2100).
pub const SLOT_STEP: u64 = 0x100;

/// Which of Metal's pools of small allocations: it suballocates buffers of their own memory under
/// a page ([`OWN_LENGTH`]) and textures of a heap size under a page ([`OWN_HEAP_SIZE`]) from
/// pools of [`POOL_BYTES`], a class for each storage mode (`vaprobe smallpool`, `vaprobe
/// texpool2`, `vaprobe smalltex`).
///
/// A slot is the first run of free bytes of the pools of its class, in the order Metal made them,
/// that holds it, lowest offset first, rounded up to [`SLOT_STEP`]; a slot freed merges with the
/// free bytes either side (`vaprobe smallpool`: a released 0x100-byte slot was the next 0x100
/// buffer's, and a 0x200-byte buffer passed a lone 0x100-byte hole by, which the next 0x100 one
/// took; `vaprobe texpool2`: an array of 0x800 bytes of Metal's own took a hole two adjacent
/// slots freed). Where no pool of its class holds it, Metal makes another, at the lowest free
/// range of the space, and takes the slot at its first byte (512 buffers of 0x100 bytes to a
/// pool; 512 private 4x4 RGBA8 textures, 128 of 16x16 and 16 of 32x64). One it made for a slot
/// goes with its last slot, where one a first use made stays ([`FirstUse`]; `vaprobe texpool2`:
/// with every texture released, a buffer took the second pool's range again; when it goes past
/// the release of its last slot is unmeasured).
///
/// The first queue's setup takes slots of its own: 0x4500 bytes of the shared pool, the texture
/// and sampler descriptor arrays' first among them ([`Elements`], [`Samplers`]), the rest for
/// good, and 0x100 of the private one, for good (the first shared buffer after it at 0x4500 into
/// its pool, the first private one at 0x100). The managed pool comes at its first slot, as a
/// full one's next does (`vaprobe smalltex`: the first managed 4x4 RGBA8 texture, after a
/// bring-up, took a new pool at 0x100_0008_0000 and the slot at its first byte; `vaprobe
/// smallpool managed`: so did the first managed buffer of 0x100 bytes, and managed buffers
/// took slots, reused and merged, and further pools as shared and private ones do).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolClass {
    /// Shared storage's, which Metal's own arrays take slots in too.
    Shared,
    /// Private storage's.
    Private,
    /// Managed storage's.
    Managed,
}

impl PoolClass {
    /// The class of a resource of `storage`: private's, managed's, else shared's.
    pub fn of(storage: Option<rt::StorageMode>) -> PoolClass {
        match storage {
            Some(rt::StorageMode::Private) => PoolClass::Private,
            Some(rt::StorageMode::Managed) => PoolClass::Managed,
            _ => PoolClass::Shared,
        }
    }
}

/// The bytes of the shared pool Metal's setup takes for good, past the texture and sampler
/// descriptor arrays' first ([`PoolClass`]).
const SHARED_SETUP: u64 = 0x3e00;

/// The bytes of the private pool Metal's setup takes ([`PoolClass`]).
const PRIVATE_SETUP: u64 = 0x100;

/// One of Metal's pools of small allocations ([`PoolClass`]).
#[derive(Debug)]
struct Pool {
    /// Its class.
    class: PoolClass,
    /// Its range.
    held: Held,
    /// Its runs of free bytes, by their first offset, to their ends.
    free: BTreeMap<u64, u64>,
    /// The bytes of its slots taken.
    used: u64,
    /// Whether a first use made it, which Metal keeps for good.
    kept: bool,
    /// The runtime's reservations of its range (`rt::Device::reserve_address_pool`), one on each
    /// device a buffer lies in it on, which those buffers' holds keep ([`Placing::pool_slot`]),
    /// each beside the device it was made through, named by the address of its properties
    /// ([`Pool::reservation`]).
    reserved: Vec<(Weak<rt::AddressPool>, usize)>,
}

impl Pool {
    /// A pool of `class` at `held`, all of it free.
    fn new(class: PoolClass, held: Held, kept: bool) -> Pool {
        Pool {
            class,
            held,
            free: BTreeMap::from([(0, POOL_BYTES)]),
            used: 0,
            kept,
            reserved: Vec::new(),
        }
    }

    /// The runtime's reservation of its range on `device`: the one a buffer that lies in it on
    /// that device holds, or one made now at its address, of [`POOL_BYTES`], which holds the span
    /// a pin of that length holds, its range's here ([`MetalSpace::place`]). The runtime's
    /// refusal where it refuses it.
    ///
    /// A device is one runtime device, not one physical device: each has a space of its own,
    /// and refuses a slot of another's reservation (`ForeignObject`). Their buffers' holds keep
    /// them, so a device keeps its reservation while a buffer of it lies in the pool, and none
    /// after. A reservation is named by the address of the properties of the device it was made
    /// through, the platform's handle, which every clone of that handle answers: a live
    /// reservation holds a handle of its device's generation, which keeps that device and its
    /// properties where they are, so the address names it alone. The reservation's own device
    /// (`rt::AddressPool::device`) is that generation's handle, whose properties are a copy of
    /// its own at another address.
    fn reservation(&mut self, device: &rt::Device) -> Result<Arc<rt::AddressPool>, rt::Error> {
        let named = std::ptr::from_ref(device.properties()).addr();
        self.reserved
            .retain(|(reserved, _)| reserved.strong_count() > 0);
        let live = (self.reserved.iter()).filter(|&&(_, of)| of == named);
        if let Some(reserved) = live.filter_map(|(reserved, _)| reserved.upgrade()).next() {
            return Ok(reserved);
        }
        let at = rt::AddressPlacement::At(rt::GpuAddress(self.held.address()));
        let reserved = Arc::new(device.reserve_address_pool(at, POOL_BYTES)?);
        self.reserved.push((Arc::downgrade(&reserved), named));
        Ok(reserved)
    }

    /// The lowest offset with `bytes` free, now taken.
    fn take(&mut self, bytes: u64) -> Option<u64> {
        let (start, end) = self
            .free
            .iter()
            .map(|(&start, &end)| (start, end))
            .find(|&(start, end)| end - start >= bytes)?;
        self.free.remove(&start);
        if end - start > bytes {
            self.free.insert(start + bytes, end);
        }
        self.used += bytes;
        Some(start)
    }

    /// The `bytes` at `offset` free again, merged with the free runs either side.
    fn give(&mut self, offset: u64, bytes: u64) {
        let (mut start, mut end) = (offset, offset + bytes);
        if let Some((&before, &before_end)) = self.free.range(..offset).next_back()
            && before_end == offset
        {
            self.free.remove(&before);
            start = before;
        }
        if let Some(after_end) = self.free.remove(&end) {
            end = after_end;
        }
        self.free.insert(start, end);
        self.used -= bytes;
    }
}

/// A slot in one of Metal's pools ([`PoolClass`]): a small buffer's or texture's bytes, or an
/// array of Metal's own.
#[derive(Debug, PartialEq, Eq)]
pub struct Slot {
    /// Its pool's address.
    pool: u64,
    /// Its offset into the pool.
    offset: u64,
    /// Its bytes, a multiple of [`SLOT_STEP`].
    bytes: u64,
}

impl Slot {
    /// Its address: what `gpuAddress` answers for a buffer of it on Metal.
    pub const fn address(&self) -> u64 {
        self.pool + self.offset
    }
}

/// What a resource takes of the space: a plain range of its own, a slot in one of Metal's pools,
/// or neither.
#[derive(Debug, Default, PartialEq, Eq)]
pub enum Holding {
    /// Neither.
    #[default]
    Nothing,
    /// A plain range of its own.
    Range(Held),
    /// A slot in one of Metal's pools.
    Slot(Slot),
}

/// The bytes of each of the descriptor mempool's two main arrays as the first queue's setup makes
/// them ([`Elements`]).
const FIRST_ARRAY: u64 = 0x200;

/// The bytes of a main array for each texture it serves ([`Elements`]).
const ELEMENT_BYTES: u64 = 0x20;

/// Metal's mempool of texture descriptors: an element for each texture, in two main arrays and a
/// third a quarter their size (`vaprobe texpool2`: 4x4, 16x16 and 32x64 RGBA8 private textures,
/// made and kept). The first queue's setup makes them of [`FIRST_ARRAY`], [`FIRST_ARRAY`] and a
/// quarter of it in the shared pool. The texture whose count reaches the main arrays' bytes over
/// [`ELEMENT_BYTES`] (the 16th, then the 32nd, the 64th and so on) makes three of twice the size,
/// in order, then frees the old three, then takes its own range or slot: the 16th made arrays of
/// 0x400, 0x400 and 0x100 bytes, the 256th of 0x4000, 0x4000 and 0x1000, the 2048th of 0x20000,
/// 0x20000 and 0x8000. An array under a page takes a slot in the shared pool ([`PoolClass`]),
/// and one of a page or more a plain range (the 1024th 4x4 texture's arrays took ranges at
/// 0x100_000d_8000, 0x100_000f_0000 and 0x100_0008_0000, then freed the 512th's, and the texture
/// then took a new pool at 0x100_0008_8000, which held the 512th's).
///
/// Every texture takes an element, whatever made it, and the count is of textures live, an
/// element freed with its texture (`vaprobe texkinds`: 254 views, textures over a buffer, in a
/// heap, on a surface or opened from a shared handle, kept, grew the arrays as textures of their
/// own memory do; `vaprobe texrelease`: 200 made and released grew nothing). That the arrays
/// never shrink is unmeasured.
#[derive(Debug, Default)]
struct Elements {
    /// The textures live.
    live: u64,
    /// The bytes of each main array: none before the first queue's setup.
    array: u64,
    /// The arrays.
    arrays: Vec<Holding>,
}

/// Metal's mempool of sampler descriptors: an element for each ID of the device's sampler ID
/// pool, in one array, which the first queue's setup makes of [`FIRST_ARRAY`], [`FIRST_POOL`]
/// IDs, in the shared pool. The pool's IDs are the sampler states', the device's own sampler's,
/// which that setup takes, and those pipeline states number for their functions (each constexpr
/// sampler, a texture read, a texture write, once device-wide while a state, function object or
/// library holds it: the sampler-ID rule, `sidprobe`). Its size is the runtime's model of
/// Metal's pool (`rt::Device::sampler_id_pool_size`: 16 on a new device, doubled by the plain ID
/// that finds none free below it and by a reservation at or past it, and never shrunk).
///
/// As the pool doubles, an array of [`ELEMENT_BYTES`] for each of its IDs is made, then the old
/// one freed (`vaprobe samplers`: sampler states of argument buffer support, each of its own LOD
/// clamp, made and kept after three queues: the 15th, 31st, 63rd and so on after the device's
/// own, which double the pool, made slots in the shared pool of 0x400, 0x800, 0x1000 and 0x2000
/// bytes at 0x100_0001_c500, 0x100_0001_c900, 0x100_0001_d100 and 0x100_0001_e100, then ranges
/// of 0x4000, 0x8000 and 0x10000 bytes at 0x100_0008_0000, 0x100_0008_8000 and
/// 0x100_0009_8000, as a texture descriptor array does, [`Elements`]). A sampler state placed at
/// an ID at or past the pool's size grows neither the runtime's pool nor the array, as Metal's
/// grows nothing (`vaprobe placedsampler`: after three queues, a state placed at 15, 16, 20, 32,
/// 600 or 2000 moved neither the next range nor the next small shared buffer's slot).
/// Unmeasured, and taken as follows: a pool that doubles more than once in one call makes one
/// array of its new size.
///
/// The space reads the pool's size of the process's device ([`count_sampler_ids`]) each time it
/// is locked ([`process`]), before anything else of it changes: every growth since is placed
/// first, where Metal placed it, at the call that took the ID.
#[derive(Debug)]
struct Samplers {
    /// The pool's size last read: [`FIRST_POOL`] IDs before any.
    pool: u64,
    /// The array's bytes: none before the first queue's setup.
    array: u64,
    /// The array.
    held: Holding,
}

/// The IDs of a new device's sampler ID pool, which the first array holds ([`Samplers`]).
const FIRST_POOL: u64 = FIRST_ARRAY / ELEMENT_BYTES;

impl Default for Samplers {
    fn default() -> Samplers {
        Samplers {
            pool: FIRST_POOL,
            array: 0,
            held: Holding::Nothing,
        }
    }
}

/// One process's space: every run of it Metal holds, by its first byte.
#[derive(Debug, Default)]
pub struct MetalSpace {
    /// Each held run's first byte and its end. Runs never overlap.
    held: BTreeMap<u64, u64>,
    /// The gaps between the held runs, for the lowest start a range fits at
    /// ([`MetalSpace::lowest_gap`]), changed with `held` alone ([`MetalSpace::hold`],
    /// [`MetalSpace::unhold`]).
    gaps: gaps::Gaps,
    /// Whether the device's heap is held ([`BASE`]): from the space's first use.
    device: bool,
    /// Metal's own allocations made ([`FirstUse`]).
    own: BTreeSet<Own>,
    /// Metal's pools of small allocations, in the order it made them ([`PoolClass`]).
    pools: Vec<Pool>,
    /// Metal's mempool of texture descriptors ([`Elements`]).
    elements: Elements,
    /// Metal's mempool of sampler descriptors ([`Samplers`]).
    samplers: Samplers,
}

impl MetalSpace {
    /// A space where nothing is held.
    pub const fn new() -> MetalSpace {
        MetalSpace {
            held: BTreeMap::new(),
            gaps: gaps::Gaps::new(),
            device: false,
            own: BTreeSet::new(),
            pools: Vec::new(),
            elements: Elements {
                live: 0,
                array: 0,
                arrays: Vec::new(),
            },
            samplers: Samplers {
                pool: FIRST_POOL,
                array: 0,
                held: Holding::Nothing,
            },
        }
    }

    /// The device's heap, held from the space's first use: Metal makes it with the device.
    fn device(&mut self) {
        if !self.device {
            self.device = true;
            let (_, span) = MetalSpace::plain_span(DEVICE_HEAP).unwrap_or_default();
            self.hold(BASE, BASE + span);
        }
    }

    /// Whether Metal's own allocations for `first` are made.
    #[cfg(test)]
    pub(crate) fn took(&self, first: FirstUse) -> bool {
        allocations(first).iter().all(|own| self.own.contains(own))
    }

    /// The pools of `class` held ([`PoolClass`]).
    #[cfg(test)]
    #[cfg(target_os = "macos")]
    pub(crate) fn pools_of(&self, class: PoolClass) -> usize {
        (self.pools.iter())
            .filter(|pool| pool.class == class)
            .count()
    }

    /// The addresses of the pools of `class` held, in the order Metal made them ([`PoolClass`]).
    #[cfg(test)]
    #[cfg(target_os = "macos")]
    pub(crate) fn pool_addresses(&self, class: PoolClass) -> Vec<u64> {
        (self.pools.iter())
            .filter(|pool| pool.class == class)
            .map(|pool| pool.held.address())
            .collect()
    }

    /// The textures live ([`Elements`]).
    #[cfg(test)]
    pub(crate) fn textures_live(&self) -> u64 {
        self.elements.live
    }

    /// The device used so for the first time: Metal's own allocations for it that no earlier use
    /// made, each a plain range held for good ([`FirstUse`]), its pools among them
    /// ([`PoolClass`]); and at a queue or a texture, the setup's slots in the pools
    /// ([`MetalSpace::setup`]).
    pub fn first_use(&mut self, first: FirstUse) {
        for own in allocations(first) {
            if self.own.insert(own) {
                let held = self.place(own.bytes());
                let class = match own {
                    Own::SharedPool => Some(PoolClass::Shared),
                    Own::PrivatePool => Some(PoolClass::Private),
                    _ => None,
                };
                if let (Some(class), Some(held)) = (class, held) {
                    self.pools.push(Pool::new(class, held, true));
                }
            }
        }
        if let FirstUse::Queue | FirstUse::Texture = first {
            self.setup();
        }
    }

    /// The first queue's setup, once: the texture descriptor mempool's first arrays
    /// ([`Elements`]), the sampler descriptor mempool's, with the device's own sampler
    /// ([`Samplers`]), and Metal's other slots of its own in the pools, kept for good
    /// ([`PoolClass`]). Measured at a queue; where a texture comes before any queue, the model
    /// takes it there too.
    fn setup(&mut self) {
        if self.elements.array > 0 {
            return;
        }
        self.elements.array = FIRST_ARRAY;
        let arrays = [FIRST_ARRAY, FIRST_ARRAY, FIRST_ARRAY / 4].map(|bytes| self.own_array(bytes));
        self.elements.arrays = arrays.into_iter().collect();
        self.samplers.array = FIRST_ARRAY;
        self.samplers.held = self.own_array(FIRST_ARRAY);
        self.slot(PoolClass::Shared, SHARED_SETUP);
        self.slot(PoolClass::Private, PRIVATE_SETUP);
        self.sampler_pool(self.samplers.pool);
    }

    /// A slot of `length` bytes, rounded up to [`SLOT_STEP`], in the first pool of `class` that
    /// holds it, or at the first byte of a new pool where none does, which takes a plain range
    /// ([`PoolClass`]). `None` where the space has no room for a new pool, or past a pool's bytes.
    pub fn slot(&mut self, class: PoolClass, length: u64) -> Option<Slot> {
        let bytes = length.max(1).checked_next_multiple_of(SLOT_STEP)?;
        if bytes > POOL_BYTES {
            return None;
        }
        let found = (self.pools.iter_mut())
            .filter(|pool| pool.class == class)
            .find_map(|pool| Some((pool.held.address(), pool.take(bytes)?)));
        let (pool, offset) = match found {
            Some(found) => found,
            None => {
                let mut pool = Pool::new(class, self.place(POOL_BYTES)?, false);
                let found = (pool.held.address(), pool.take(bytes)?);
                self.pools.push(pool);
                found
            }
        };
        Some(Slot {
            pool,
            offset,
            bytes,
        })
    }

    /// `slot` free again; a pool made for a slot goes with its last ([`PoolClass`]).
    pub fn free_slot(&mut self, slot: Slot) {
        let Some(i) = (self.pools.iter()).position(|pool| pool.held.address() == slot.pool) else {
            return;
        };
        let pool = &mut self.pools[i];
        pool.give(slot.offset, slot.bytes);
        if pool.used == 0 && !pool.kept {
            let pool = self.pools.remove(i);
            self.release(pool.held);
        }
    }

    /// What `holding` takes, free again.
    pub fn free(&mut self, holding: Holding) {
        match holding {
            Holding::Nothing => {}
            Holding::Range(held) => self.release(held),
            Holding::Slot(slot) => self.free_slot(slot),
        }
    }

    /// One of Metal's own arrays of `bytes` ([`Elements`]): a slot in the shared pool under a
    /// page, else a plain range.
    fn own_array(&mut self, bytes: u64) -> Holding {
        let made = match bytes < PAGE {
            true => self.slot(PoolClass::Shared, bytes).map(Holding::Slot),
            false => self.place(bytes).map(Holding::Range),
        };
        made.unwrap_or(Holding::Nothing)
    }

    /// A texture made: its descriptor, which grows the mempool where its count reaches the
    /// arrays' ([`Elements`]).
    pub fn texture_made(&mut self) {
        self.elements.live += 1;
        let array = self.elements.array;
        if array > 0 && self.elements.live >= array / ELEMENT_BYTES {
            let array = array * 2;
            self.elements.array = array;
            let arrays = [array, array, array / 4].map(|bytes| self.own_array(bytes));
            let old = std::mem::replace(&mut self.elements.arrays, arrays.into_iter().collect());
            old.into_iter().for_each(|held| self.free(held));
        }
    }

    /// A texture gone: its descriptor free again ([`Elements`]).
    pub fn texture_gone(&mut self) {
        self.elements.live = self.elements.live.saturating_sub(1);
    }

    /// The device's sampler ID pool of `size` IDs: past the array's, an array of `size`
    /// elements made, then the old one freed ([`Samplers`]). Before the first queue's setup, the
    /// size is kept for it.
    pub fn sampler_pool(&mut self, size: u64) {
        self.samplers.pool = self.samplers.pool.max(size);
        let bytes = self.samplers.pool.saturating_mul(ELEMENT_BYTES);
        if self.samplers.array > 0 && bytes > self.samplers.array {
            let made = self.own_array(bytes);
            self.samplers.array = bytes;
            let old = std::mem::replace(&mut self.samplers.held, made);
            self.free(old);
        }
    }

    /// The bytes of pages and the whole span a plain range of `length` bytes holds: its pages
    /// and one guard page, rounded up to [`SLOT`] (`vaprobe sizes`: 1 to 64 pages, buffers of
    /// every storage mode, no-copy ones and heaps; three pages take 0x10000, five 0x18000). A
    /// pin's span is the same ([`MetalSpace::pin`]). `None` past the 64-bit space.
    fn plain_span(length: u64) -> Option<(u64, u64)> {
        let pages = length.max(1).checked_next_multiple_of(PAGE)?;
        let span = pages.checked_add(PAGE)?.checked_next_multiple_of(SLOT)?;
        Some((pages, span))
    }

    /// The first held byte of `range`, if any.
    fn first_held(&self, range: &Range<u64>) -> Option<u64> {
        // Runs never overlap, so the runs reaching into the range are the last ones that start
        // before its end, down to the first that ends at or before its start.
        self.held
            .range(..range.end)
            .rev()
            .take_while(|(_, end)| **end > range.start)
            .last()
            .map(|(&start, _)| start.max(range.start))
    }

    /// `start..end` held from now on, in place of a run held from `start` before.
    fn hold(&mut self, start: u64, end: u64) {
        if self.held.contains_key(&start) {
            self.unhold(start);
        }
        self.held.insert(start, end);
        self.gaps.made(&self.held, start);
    }

    /// The run held from `start`, if any, free from now on.
    fn unhold(&mut self, start: u64) {
        if self.held.remove(&start).is_some() {
            self.gaps.gone(&self.held, start);
        }
    }

    /// The lowest [`SLOT`]-aligned start from [`BASE`] where `span` bytes are free: walking the
    /// held runs in order, [`BASE`] or the first run's end rounded up to [`SLOT`] whose bytes end
    /// at or before the next run's start, or past every run; `None` where they would pass [`END`].
    /// The gaps answer the walk's start in O(log n) ([`gaps`]).
    fn lowest_gap(&self, span: u64) -> Option<u64> {
        let at = self.gaps.lowest(span)?;
        (at.checked_add(span)? <= END).then_some(at)
    }

    /// [`MetalSpace::lowest_gap`] by the walk itself, which the gaps answer as.
    #[cfg(test)]
    fn walked_gap(&self, span: u64) -> Option<u64> {
        let mut at = BASE;
        for (&start, &end) in &self.held {
            if end <= at {
                continue;
            }
            if start >= at.checked_add(span)? {
                break;
            }
            at = end.checked_next_multiple_of(SLOT)?;
        }
        (at.checked_add(span)? <= END).then_some(at)
    }

    /// A plain range for a buffer, heap or other resource of `length` bytes, now held, at the
    /// lowest free start that holds it, a free run split where it is larger (`vaprobe reuse` and
    /// `vaprobe order`: whichever was released first, a lower freed range goes first, and a
    /// freed two-page range is split for a one-page one before a freed one-page range above it
    /// is taken); `None` when the space has no room for it.
    pub fn place(&mut self, length: u64) -> Option<Held> {
        self.device();
        let (pages, span) = MetalSpace::plain_span(length)?;
        let start = self.lowest_gap(span)?;
        let end = start.checked_add(span)?;
        self.hold(start, end);
        let pages_end = start.checked_add(pages)?;
        Some(Held {
            address: start,
            pages: start..pages_end,
            guard: pages_end..end,
            pinned: false,
        })
    }

    /// The guest's pin of `length` bytes at `address`, now held: from its first page, the span
    /// a plain range of its pages holds ([`MetalSpace::plain_span`]). Refused where its first
    /// page is a granule's second or its span meets a held range, in that order, as the runtime
    /// refuses a pin (`PlacementConflict`).
    pub fn pin(&mut self, address: u64, length: u64) -> Result<Held, PinRefused> {
        self.device();
        let first = address & !(PAGE - 1);
        let end = address
            .checked_add(length.max(1))
            .and_then(|end| end.checked_next_multiple_of(PAGE))
            .ok_or(PinRefused::Outside)?;
        let span_end = MetalSpace::plain_span(end - first)
            .and_then(|(_, span)| first.checked_add(span))
            .ok_or(PinRefused::Outside)?;
        let requested = first..span_end;
        if first & (SLOT - 1) != 0 {
            return Err(PinRefused::OffGranule { requested });
        }
        if let Some(held) = self.first_held(&requested) {
            return Err(PinRefused::Held { requested, held });
        }
        self.hold(first, span_end);
        Ok(Held {
            address,
            pages: first..end,
            guard: end..span_end,
            pinned: true,
        })
    }

    /// A texture of `kind` made: its descriptor ([`MetalSpace::texture_made`]), then the range
    /// or slot it holds of its own, now held: nothing for one that takes none or its surface's
    /// ([`texture_space`]), or where the space has no room for it. A texture of memory of its
    /// own, suballocated or not, or a shareable one, is a first use ([`FirstUse::Texture`]),
    /// before both.
    pub fn place_texture(&mut self, kind: TextureKind) -> Holding {
        if let TextureKind::OwnMemory { .. } | TextureKind::SharedBacking { .. } = kind {
            self.first_use(FirstUse::Texture);
        }
        self.texture_made();
        let made = match texture_space(kind) {
            TextureSpace::None | TextureSpace::Surface => None,
            TextureSpace::Own(bytes) => self.place(bytes).map(Holding::Range),
            TextureSpace::Slot { class, bytes } => self.slot(class, bytes).map(Holding::Slot),
        };
        made.unwrap_or(Holding::Nothing)
    }

    /// The range of a surface of `alloc_size` bytes ([`SurfaceRange`]), now held, after the
    /// first use a texture over it is ([`FirstUse::Texture`]). The first use a buffer over a
    /// surface is, made before any queue or texture, is unmeasured: the model takes it as a
    /// texture's, which the library's queues, made as it starts, make moot.
    pub fn place_surface(&mut self, alloc_size: u64) -> Option<Held> {
        self.first_use(FirstUse::Texture);
        self.place(alloc_size)
    }

    /// `held`'s resource released, or never made after all: its whole span free again, a
    /// pin's as a plain range's.
    pub fn release(&mut self, held: Held) {
        self.unhold(held.pages.start);
    }
}

/// The process's space.
static PROCESS: Mutex<MetalSpace> = Mutex::new(MetalSpace::new());

/// The device whose sampler ID pool Metal's sampler mempool follows ([`Samplers`]): the
/// process's first, the one Metal's device is.
static SAMPLER_IDS: OnceLock<rt::Device> = OnceLock::new();

/// `device` made: the process's sampler mempool follows its sampler ID pool, if it is the
/// process's first ([`Samplers`]). Another runtime device of the process (a test's, with a shader
/// cache of its own) has a pool of its own, which the space does not follow.
pub fn count_sampler_ids(device: &rt::Device) {
    let _ = SAMPLER_IDS.set(device.clone());
}

/// The process's space, locked, after its sampler mempool grew to the process's device's
/// sampler ID pool ([`Samplers`]).
pub fn process() -> MutexGuard<'static, MetalSpace> {
    let mut space = PROCESS.lock();
    if let Some(device) = SAMPLER_IDS.get() {
        space.sampler_pool(device.sampler_id_pool_size());
    }
    space
}

/// Whether the process's space took `first` through [`first_use`]: what it takes is held for
/// good, so a later use takes nothing.
pub fn taken(first: FirstUse) -> bool {
    FIRST_USES.load(std::sync::atomic::Ordering::Acquire) & first.bit() != 0
}

/// The process's device used so ([`FirstUse`]): Metal's own allocations for it, the first time.
/// Each blit command, encoder and tiler submission is one, so a use the process's space made
/// already ([`FIRST_USES`]) is answered without its lock: it changes nothing of the space, and
/// the lock's other work, the sampler mempool's growth ([`process`]), is placed by the next
/// lock that changes it, before that change, where this one would have placed it.
pub fn first_use(first: FirstUse) {
    let bit = first.bit();
    if FIRST_USES.load(std::sync::atomic::Ordering::Acquire) & bit != 0 {
        return;
    }
    let mut space = process();
    space.first_use(first);
    FIRST_USES.fetch_or(bit, std::sync::atomic::Ordering::Release);
}

/// The first uses the process's space made through [`first_use`], a bit each
/// ([`FirstUse::bit`]): each made once, with every allocation it takes held for good, so a later
/// one of the same kind is no change.
static FIRST_USES: std::sync::atomic::AtomicU16 = std::sync::atomic::AtomicU16::new(0);

impl FirstUse {
    /// Its bit in [`FIRST_USES`].
    const fn bit(self) -> u16 {
        1 << match self {
            FirstUse::Queue => 0,
            FirstUse::Texture => 1,
            FirstUse::ComputeEncoder => 2,
            FirstUse::BlitCommand => 3,
            FirstUse::RenderEncoder => 4,
            FirstUse::TilerBlit => 5,
            FirstUse::TilerSubmission => 6,
            FirstUse::SmallShared => 7,
            FirstUse::SmallPrivate => 8,
        }
    }
}

/// A resource's hold on the process's space: its ranges, given back when it goes.
///
/// A buffer's or heap's is the runtime's range token ([`Placing::token`]), which the runtime
/// drops right after it frees the resource's range: after the completed handlers of the last
/// command buffer that used it and before that command buffer's waiters wake, as Metal frees it
/// (`vaprobe inflight` and `vaprobe handler`: a buffer or heap released while a committed command
/// buffer uses it keeps its range inside the completed handler, and the range is free once
/// `waitUntilCompleted` returns), and before a creation the runtime refuses returns. So the
/// space places a range again once the runtime has freed it too. A resource the runtime leaves
/// to the process's exit keeps its range, as Metal's does.
///
/// A texture over a surface holds its surface's range, which every texture and buffer over the
/// surface shares ([`SurfaceRange`]), through the range token of the buffer it is over too, so
/// the range goes once the runtime frees the last of them, as a buffer's does.
///
/// A texture of its own memory's range and a shareable texture's, which the runtime holds no
/// range of, are placed before the runtime makes the texture and held by its range token too,
/// which the runtime drops once the texture, every view of it and every texture opened from its
/// handles, and the command buffers that hold any of them, are gone, as Metal frees a released
/// texture's range; their objects share the hold, for its address. A sparse heap's ranges are
/// its range token's, which the runtime drops once the heap, its textures and the command
/// buffers that hold any of them are gone.
///
/// A small buffer's or texture's slot in one of Metal's pools goes as a range does, through its
/// range token ([`PoolClass`]). A texture's hold counts its descriptor in Metal's mempool while
/// it lives ([`Elements`]): a texture's of its own memory or a shareable one's until the runtime
/// frees it, as its range; a view's, a texture's over a buffer or in a heap and one's opened
/// from a shared handle with its object.
#[derive(Debug, Default)]
pub struct Hold(Kept, Descriptor);

/// Whether a [`Hold`] counts a texture's descriptor in Metal's mempool ([`Elements`]), which
/// goes with the hold.
#[derive(Debug, Default)]
struct Descriptor(bool);

/// What a [`Hold`] keeps.
#[derive(Debug, Default)]
enum Kept {
    /// Nothing.
    #[default]
    Nothing,
    /// Ranges of its own: one, or a sparse heap's two ([`SPARSE_STATE`]).
    Own(Vec<Held>),
    /// Its surface's range, with whether it is a buffer at the range's address
    /// ([`SurfaceRange::buffers`]).
    Surface(Arc<SurfaceRange>, bool),
    /// A slot in one of Metal's pools ([`PoolClass`]), with the runtime's reservation of the pool
    /// a buffer in it lies in ([`Placing::pool_slot`]).
    Slot(Slot, Option<Arc<rt::AddressPool>>),
    /// Another texture's hold, which a view or a texture opened from a shared handle shares.
    Shares(Arc<Hold>),
}

impl Hold {
    /// No hold: a buffer placed in a heap, which takes none of the space.
    pub const NONE: Hold = Hold(Kept::Nothing, Descriptor(false));

    /// A buffer's or heap's hold of `kept`.
    const fn of(kept: Kept) -> Hold {
        Hold(kept, Descriptor(false))
    }

    /// A texture's hold of `kept`, counting its descriptor, which its caller took
    /// ([`MetalSpace::texture_made`]).
    const fn texture(kept: Kept) -> Hold {
        Hold(kept, Descriptor(true))
    }

    /// The hold of a view of the texture that holds `of`, or of a texture opened from its shared
    /// handle: it shares `of`, and takes a descriptor of its own ([`Elements`]).
    pub fn sharing(of: Arc<Hold>) -> Hold {
        process().texture_made();
        Hold::texture(Kept::Shares(of))
    }

    /// The address it holds the space at, if any: a slot's is Metal's address in its pool.
    pub fn address(&self) -> Option<u64> {
        match &self.0 {
            Kept::Nothing => None,
            Kept::Own(held) => held.first().map(Held::address),
            Kept::Surface(range, _) => Some(range.address()),
            Kept::Slot(slot, _) => Some(slot.address()),
            Kept::Shares(of) => of.address(),
        }
    }
}

impl Drop for Hold {
    fn drop(&mut self) {
        match std::mem::take(&mut self.0) {
            Kept::Nothing | Kept::Shares(_) => {}
            Kept::Own(held) => {
                let mut space = process();
                held.into_iter().for_each(|held| space.release(held));
            }
            Kept::Slot(slot, reserved) => {
                // The reservation goes with the space locked, so no buffer reserves the pool on
                // the device again before the runtime lets its span go.
                let mut space = process();
                space.free_slot(slot);
                drop(reserved);
            }
            Kept::Surface(range, buffer) => {
                if buffer {
                    *range.buffers.lock() -= 1;
                }
            }
        }
        if std::mem::take(&mut self.1.0) {
            process().texture_gone();
        }
    }
}

/// An IOSurface's range of the space: one plain range of the surface's allocation size, which
/// every texture over the surface and every buffer over it (`newBufferWithIOSurface:`) shares
/// while one lives, and which goes back with the last (`vaprobe surfaces`: a texture over a
/// surface of 0x4000 bytes took 0x8000 of the space, one of 0x40000 bytes 0x48000, and the
/// buffer made over the same surface after it answered the texture's address and took none).
///
/// A surface is one `IOSurfaceRef` here: whether Metal shares the range with another reference
/// to the same surface (`IOSurfaceLookupFromMachPort`) is unmeasured.
#[derive(Debug)]
pub struct SurfaceRange {
    /// The surface's key ([`surface`]).
    key: u64,
    /// The range, none where the space had no room for it.
    held: Option<Held>,
    /// The buffers over the surface at the range's address, until the runtime frees each one's
    /// range: the first is made at the address, and each later one shares its range with a live
    /// one ([`surface_buffer`]).
    buffers: Mutex<u64>,
}

impl SurfaceRange {
    /// Its address, 0 where the space had no room for it.
    pub fn address(&self) -> u64 {
        self.held.as_ref().map_or(0, Held::address)
    }
}

impl Drop for SurfaceRange {
    fn drop(&mut self) {
        let mut surfaces = SURFACES.lock();
        if let Some(live) = surfaces.get(&self.key)
            && std::ptr::eq(live.as_ptr(), self)
        {
            surfaces.remove(&self.key);
        }
        drop(surfaces);
        if let Some(held) = self.held.take() {
            process().release(held);
        }
    }
}

/// Each surface's range while a texture or buffer over it lives, by the surface's key.
static SURFACES: Mutex<BTreeMap<u64, Weak<SurfaceRange>>> = Mutex::new(BTreeMap::new());

/// The range of the surface `key` (its `IOSurfaceRef`), of `alloc_size` bytes: the live one, or a
/// new one placed now.
fn surface_range(key: u64, alloc_size: u64) -> Arc<SurfaceRange> {
    let mut surfaces = SURFACES.lock();
    if let Some(live) = surfaces.get(&key).and_then(Weak::upgrade) {
        return live;
    }
    let range = Arc::new(SurfaceRange {
        key,
        held: process().place_surface(alloc_size),
        buffers: Mutex::new(0),
    });
    surfaces.insert(key, Arc::downgrade(&range));
    range
}

/// A texture's hold on the surface `key`'s range, of `alloc_size` bytes ([`SurfaceRange`]), after
/// its descriptor ([`MetalSpace::texture_made`]).
pub fn surface(key: u64, alloc_size: u64) -> Hold {
    {
        let mut space = process();
        space.first_use(FirstUse::Texture);
        space.texture_made();
    }
    Hold::texture(Kept::Surface(surface_range(key, alloc_size), false))
}

/// Where a buffer over a surface is made ([`surface_buffer`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfacePlace {
    /// At the placement: the surface's address, or where the runtime picks where the space had
    /// no room for the surface's range.
    At(rt::AddressPlacement),
    /// At the address another buffer over the surface holds, sharing its range
    /// (`BufferDescriptor::with_range_of`): the range is the runtime's once one buffer holds it,
    /// and stays while one holds it.
    Shared,
}

/// Where a buffer over the surface `key`, of `alloc_size` bytes, is made, and its hold on the
/// surface's range ([`SurfaceRange`]), which its range token keeps ([`Hold`]): at the range's
/// address while no buffer over the surface holds it, sharing the range of one that does where
/// one does, or where the runtime picks where the space had no room for the range.
///
/// Metal answers the surface's address for every buffer over it (`vaprobe surfacetwo`: two
/// buffers over one surface, both live, answered its address, with a texture over it made first
/// and with none). The runtime holds one buffer at an address, so a buffer made while another
/// holds it shares that one's range ([`SurfacePlace::Shared`]): a buffer holds the address while
/// it lives, and after its release until the runtime frees its range, once the command buffers
/// that used it completed.
pub fn surface_buffer(key: u64, alloc_size: u64) -> (SurfacePlace, Hold) {
    let range = surface_range(key, alloc_size);
    let mut buffers = range.buffers.lock();
    let place = match (&range.held, *buffers) {
        (Some(_), 1..) => SurfacePlace::Shared,
        (Some(held), 0) => {
            SurfacePlace::At(rt::AddressPlacement::At(rt::GpuAddress(held.address())))
        }
        (None, _) => {
            drop(buffers);
            return (
                SurfacePlace::At(rt::AddressPlacement::Any),
                Hold::of(Kept::Surface(range, false)),
            );
        }
    };
    *buffers += 1;
    drop(buffers);
    (place, Hold::of(Kept::Surface(range, true)))
}

/// The hold of a texture of `kind`: its descriptor ([`Elements`]), and a range of its own, a slot
/// in one of Metal's pools, or neither; for [`TextureKind::OverSurface`], its caller takes the
/// surface's ([`surface`]).
pub fn texture(kind: TextureKind) -> Hold {
    let kept = match process().place_texture(kind) {
        Holding::Nothing => Kept::Nothing,
        Holding::Range(held) => Kept::Own(vec![held]),
        Holding::Slot(slot) => Kept::Slot(slot, None),
    };
    Hold::texture(kept)
}

/// A buffer or heap being made: the ranges it holds until the runtime's range token takes them
/// ([`Placing::token`]) or its object does ([`Placing::made`]), given back where neither does
/// (its drop: a raise out of the creation before the runtime saw the token).
#[derive(Debug)]
pub struct Placing {
    /// Its ranges: its own, and a sparse heap's second ([`SPARSE_STATE`]); none where the space
    /// refused it or Metal suballocates it.
    held: Vec<Held>,
    /// Its slot in one of Metal's pools, where Metal suballocates it ([`PoolClass`]).
    slot: Option<Slot>,
    /// The runtime's reservation of its slot's pool ([`Placing::pool_slot`]).
    reserved: Option<Arc<rt::AddressPool>>,
    /// Whether the space placed it at its pin, a range or a slot, `Some(pinned)`.
    placed: Option<bool>,
    /// Why the space refused the guest's pin, where it did.
    pin_refused: Option<PinRefused>,
}

impl Placing {
    /// Whether Metal answers nil where the runtime refuses the address
    /// (`PlacementConflict`): for a guest's pin, which Metal answers nil for where it cannot
    /// honour it, and for one the space refused ([`REFUSED`]); not for a range or slot the space
    /// placed, where Metal makes the resource.
    pub fn nil_at_conflict(&self) -> bool {
        self.placed.is_none_or(|pinned| pinned)
    }

    /// Why the space refused the guest's pin, where it did ([`MetalSpace::pin`]): the creator
    /// answers nil for it.
    pub fn pin_refused(&self) -> Option<&PinRefused> {
        self.pin_refused.as_ref()
    }

    /// Whether the space refused a heap ([`placing`] with [`Bytes::Heap`]): a pin off a granule or
    /// whose span meets a held range, or no room for its range ([`REFUSED`]).
    pub fn refused(&self) -> bool {
        self.placed.is_none()
    }

    /// Where the runtime makes a buffer Metal suballocates on `device`
    /// (`BufferDescriptor::with_pool_slot`): the runtime's reservation on the device of its slot's
    /// pool ([`PoolClass`]), the one a buffer that lies in the pool on the device holds or one
    /// made now, and the slot's offset in it. `None` where it holds no slot, as before its token
    /// took it ([`Placing::token`]); the runtime's refusal of the reservation where it refuses
    /// it, which no range of the runtime's in the pool's span should cause: the space holds the
    /// span, and places every other range of Metal's space.
    ///
    /// Its hold keeps the reservation with the slot ([`Hold`]). The runtime frees a slot as it
    /// frees a buffer's range, before it drops the buffer's range token, so the space hands a
    /// slot out again once the runtime's is free too, and the last buffer of a device in the
    /// pool lets the runtime's span go: every device of the process places its buffers in the
    /// pools of the one model, as Metal's one device does.
    pub fn pool_slot(
        &mut self,
        device: &rt::Device,
    ) -> Option<Result<(Arc<rt::AddressPool>, u64), rt::Error>> {
        let slot = self.slot.as_ref()?;
        let mut space = process();
        let pool = (space.pools.iter_mut()).find(|pool| pool.held.address() == slot.pool)?;
        let made = pool.reservation(device);
        drop(space);
        Some(made.map(|reserved| {
            self.reserved = Some(Arc::clone(&reserved));
            (reserved, slot.offset)
        }))
    }

    /// What it holds: its ranges, or its slot ([`Hold`]); `None` for neither.
    fn kept(&mut self) -> Option<Kept> {
        let held = std::mem::take(&mut self.held);
        let reserved = self.reserved.take();
        match (held.is_empty(), self.slot.take()) {
            (false, slot) => {
                drop(slot.map(|slot| Hold::of(Kept::Slot(slot, reserved))));
                Some(Kept::Own(held))
            }
            (true, Some(slot)) => Some(Kept::Slot(slot, reserved)),
            (true, None) => None,
        }
    }

    /// Its ranges or slot, as the runtime's range token for the buffer or heap ([`Hold`]): none
    /// where it holds neither.
    pub fn token(&mut self) -> Option<rt::RangeToken> {
        let kept = self.kept()?;
        Some(Box::new(Hold::of(kept)))
    }

    /// Made: what its object holds, what no token took.
    pub fn made(mut self) -> Hold {
        self.kept().map_or(Hold::NONE, Hold::of)
    }
}

impl Drop for Placing {
    /// Not made, and what it holds with neither its token nor its object: nothing of it stays
    /// held.
    fn drop(&mut self) {
        drop(self.kept().map(Hold::of));
    }
}

/// Where the runtime makes a buffer or heap the space has no range for: below the runtime's
/// space, which it refuses (`PlacementConflict`) as it checks the address. It checks the
/// descriptor's options, storage, length and contents first, so a descriptor it refuses for
/// those dies or is nil as it was before the space, and the rest is nil
/// ([`Placing::nil_at_conflict`]). A sparse heap takes no address, and the runtime makes it.
pub const REFUSED: rt::GpuAddress = rt::GpuAddress(0);

/// What a buffer's or heap's bytes are, as where Metal puts it depends on them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bytes {
    /// A buffer's own memory.
    Own {
        /// The pool class of its storage ([`PoolClass`]).
        class: PoolClass,
    },
    /// The caller's memory (a no-copy buffer), its first byte this many bytes into its page: the
    /// buffer's address keeps that offset into its range's first page.
    NoCopy(u64),
    /// A heap's own.
    Heap,
}

/// Where Metal puts a buffer or heap ([`buffer_space`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferSpace {
    /// In a range of its own.
    Range,
    /// In a slot of one of Metal's pools of that class ([`PoolClass`]).
    Pool(PoolClass),
}

/// The least length of a buffer of its own memory Metal gives a range of its own: it suballocates
/// shorter ones from its pools, a pool class for each storage mode ([`PoolClass`])
/// (`vaprobe small`: shared and private buffers of 1, 0x100, 0x1000, 0x2000 and 0x3fff bytes, of
/// `newBufferWithLength:options:` and of the descriptor path, took none, at 0x100-byte steps in a
/// pool, and ones of 0x4000 bytes a range each; `vaprobe smallpool managed`: managed buffers in a
/// pool of their own).
pub const OWN_LENGTH: u64 = PAGE;

/// A creator's hold of the runtime's reservation of a pool ([`Placing::pool_slot`]), let go with
/// the space locked, as a buffer's hold lets it go ([`Hold`]): the last one of a device lets the
/// runtime's span go before another buffer can reserve the pool on that device again.
pub fn let_go<T>(reserved: Option<(Arc<rt::AddressPool>, T)>) {
    let space = process();
    drop(reserved);
    drop(space);
}

/// Where Metal puts a buffer or heap of `length` bytes of `bytes`, `pinned` by the guest or not:
/// a buffer of its own memory shorter than [`OWN_LENGTH`] in a pool, and anything else in a range
/// of its own. A pinned one takes its pin's range, which no pool's bytes hold (unmeasured for a
/// buffer under a page), and one of no bytes a range, which Metal answers nil for before it
/// allocates.
pub const fn buffer_space(length: u64, bytes: Bytes, pinned: bool) -> BufferSpace {
    match bytes {
        Bytes::Own { class } if !pinned && length > 0 && length < OWN_LENGTH => {
            BufferSpace::Pool(class)
        }
        Bytes::Own { .. } | Bytes::NoCopy(_) | Bytes::Heap => BufferSpace::Range,
    }
}

/// The first use a buffer of its own memory of `class` that Metal suballocates is: its pool's,
/// where no queue made it before; none for the managed pool, which its first slot makes
/// ([`PoolClass`]).
const fn small_first_use(class: PoolClass) -> Option<FirstUse> {
    match class {
        PoolClass::Shared => Some(FirstUse::SmallShared),
        PoolClass::Private => Some(FirstUse::SmallPrivate),
        PoolClass::Managed => None,
    }
}

/// Where a buffer or heap of `length` bytes of `bytes` is made, at the guest's `pin` if it has
/// one: the runtime's placement and the range it holds meanwhile. Where Metal makes none (a pin
/// off a granule or whose span meets a held range, or no room for the range), at [`REFUSED`],
/// holding nothing.
///
/// One Metal suballocates ([`BufferSpace::Pool`]) takes a slot in its pool, which its range
/// token holds ([`PoolClass`]), and is made at the slot's address, Metal's, in the runtime's
/// reservation of the pool ([`Placing::pool_slot`]); at [`REFUSED`] where the space has no room
/// for a new pool.
pub fn placing(length: u64, bytes: Bytes, pin: Option<u64>) -> (rt::AddressPlacement, Placing) {
    if let BufferSpace::Pool(class) = buffer_space(length, bytes, pin.is_some()) {
        let mut space = process();
        if let Some(first) = small_first_use(class) {
            space.first_use(first);
        }
        let slot = space.slot(class, length);
        let address = slot
            .as_ref()
            .map_or(REFUSED, |slot| rt::GpuAddress(slot.address()));
        let placing = Placing {
            held: Vec::new(),
            placed: slot.as_ref().map(|_| false),
            slot,
            reserved: None,
            pin_refused: None,
        };
        return (rt::AddressPlacement::At(address), placing);
    }
    let mut space = process();
    let mut pin_refused = None;
    let placed = match pin {
        Some(address) => match space.pin(address, length) {
            Ok(held) => Some((address, held)),
            Err(refused) => {
                pin_refused = Some(refused);
                None
            }
        },
        None => {
            let offset = match bytes {
                Bytes::NoCopy(offset) => offset & (PAGE - 1),
                Bytes::Own { .. } | Bytes::Heap => 0,
            };
            length
                .checked_add(offset)
                .and_then(|length| space.place(length))
                .map(|held| (held.address() + offset, held))
        }
    };
    let (address, placing) = match placed {
        Some((address, held)) => (
            rt::GpuAddress(address),
            Placing {
                placed: Some(held.pinned),
                held: vec![held],
                slot: None,
                reserved: None,
                pin_refused: None,
            },
        ),
        None => (
            REFUSED,
            Placing {
                held: Vec::new(),
                slot: None,
                reserved: None,
                placed: None,
                pin_refused,
            },
        ),
    };
    (rt::AddressPlacement::At(address), placing)
}

/// The bytes of the buffer Metal makes with a sparse heap, after the heap's own range: one page
/// (`vaprobe surfaces`: a sparse heap of 0x40000 bytes made a buffer of its size, a range of
/// 0x48000, then one of 0x4000, a range of 0x8000, and answered `gpuAddress` 0; `vaprobe
/// sparse`: one page at every size from 0x10000 to 0x1000000 bytes).
pub const SPARSE_STATE: u64 = PAGE;

/// Where a sparse heap of `size` bytes is made, at the guest's `pin` if it has one, and the
/// ranges it holds meanwhile: unpinned, a plain range of its size, then one of [`SPARSE_STATE`]
/// bytes (`vaprobe surfaces` and `vaprobe sparse`); pinned, its pin's range alone, as no probe
/// pinned a sparse heap; none where the space refuses it ([`Placing::refused`]), which its
/// creator dies at, as the runtime makes a sparse heap wherever the space places it and Metal's
/// answer there is not measured. The runtime makes a sparse heap at no address, as Metal's
/// answers 0, so the placement is the space's alone, and the runtime holds no range of it: its
/// range token holds them ([`Placing::token`], [`Hold`]).
pub fn placing_sparse(size: u64, pin: Option<u64>) -> (rt::AddressPlacement, Placing) {
    let (placement, mut placing) = placing(size, Bytes::Heap, pin);
    if placing.placed == Some(false) {
        placing.held.extend(process().place(SPARSE_STATE));
    }
    (placement, placing)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;

    impl MetalSpace {
        /// A space after the library's bring-up: the device and its queues ([`FirstUse::Queue`]).
        fn brought_up() -> MetalSpace {
            let mut space = MetalSpace::new();
            space.first_use(FirstUse::Queue);
            space
        }

        /// `span` bytes held at the lowest [`SLOT`]-aligned gap, guard pages included: what a
        /// recording shows Metal took before an object that the model does not make.
        fn taken(&mut self, span: u64) -> u64 {
            self.device();
            let start = self.lowest_gap(span).unwrap();
            self.hold(start, start + span);
            start
        }
    }

    /// One step of a recorded case or a probe's row, in the order its Metal made it.
    enum Step {
        /// A buffer or heap of that length, placed by the space.
        Make(&'static str, u64),
        /// A buffer or heap of that length, placed by the space, kept, at that address.
        At(u64, u64),
        /// The address the next one-page buffer takes, released again: a probe's `step`.
        Next(u64),
        /// A buffer or heap the guest pinned at that address, of that length, made there.
        Pin(&'static str, u64, u64),
        /// A pin the space accepts, of a resource Metal then does not make for another reason.
        PinUnmade(u64, u64),
        /// A pin at that address, of that length, that Metal refuses (nil).
        Refused(u64, u64),
        /// What Metal took before the next object, its guard pages included, read from the
        /// recording, that the model does not make: the growth of Metal's own buffers past
        /// their first use ([`FirstUse`]).
        Taken(u64),
        /// A first use of the device of that kind ([`FirstUse`]).
        Use(FirstUse),
        /// A buffer of its own memory of that length and pool class, which Metal suballocates,
        /// kept.
        Small(u64, PoolClass),
        /// The same, named, its slot at that address: the buffer's `gpuAddress` on Metal.
        SmallAt(&'static str, u64, PoolClass, u64),
        /// Every texture of a [`Step::Tex`] released.
        ReleaseTextures,
        /// A resource released.
        Release(&'static str),
        /// A value the recording shows: the named resource's address plus an offset.
        Recorded(&'static str, u64, u64),
        /// A texture of that kind, made and kept.
        Tex(TextureKind),
        /// The same, its slot or range at that address.
        TexAt(TextureKind, u64),
        /// A sampler state, made and kept: the runtime's sampler ID pool as it answers after it
        /// ([`Samplers`]).
        Sampler,
        /// The sampler descriptor mempool's array at that address ([`Samplers`]).
        SamplerArray(u64),
        /// A sampler state placed at an ID, made and kept: the runtime's sampler ID pool as it
        /// answers after it, which a placed ID does not grow ([`Samplers`]).
        PlacedSampler,
        /// A texture over a surface of that allocation size, made and kept ([`surface`]).
        Surface(&'static str, u64),
    }

    use FirstUse::{
        BlitCommand, ComputeEncoder, Queue, RenderEncoder, Texture, TilerBlit, TilerSubmission,
    };
    use Step::{
        At, Make, Next, Pin, PinUnmade, PlacedSampler, Recorded, Refused, Release, ReleaseTextures,
        Sampler, SamplerArray, Small, SmallAt, Surface, Taken, Tex, TexAt, Use,
    };

    /// A texture of private memory of its own of that heap size.
    const fn own(heap_size: u64) -> TextureKind {
        TextureKind::OwnMemory {
            heap_size,
            class: PoolClass::Private,
        }
    }

    /// A texture of shared memory of its own of that heap size.
    const fn own_shared(heap_size: u64) -> TextureKind {
        TextureKind::OwnMemory {
            heap_size,
            class: PoolClass::Shared,
        }
    }

    /// A texture of managed memory of its own of that heap size.
    const fn own_managed(heap_size: u64) -> TextureKind {
        TextureKind::OwnMemory {
            heap_size,
            class: PoolClass::Managed,
        }
    }

    /// The pool class of a buffer of private storage or not.
    const fn class(private: bool) -> PoolClass {
        match private {
            true => PoolClass::Private,
            false => PoolClass::Shared,
        }
    }

    /// A shareable texture of that allocation by Apple's layout.
    const fn shareable(allocation: u64) -> TextureKind {
        TextureKind::SharedBacking { allocation }
    }

    /// A 4x4 RGBA8 texture of memory of its own, which Metal suballocates: its `allocatedSize`
    /// as its heap size, which is under a page.
    const RGBA8_4X4: TextureKind = own(0x80);

    /// A buffer of its own shared memory.
    const SHARED: Bytes = Bytes::Own {
        class: PoolClass::Shared,
    };

    const P1: u64 = PAGE;
    const P2: u64 = 2 * PAGE;
    const P4: u64 = 4 * PAGE;
    const PINNED: u64 = 0x100_0040_0000;

    /// The objects of `icb::ICBTask`, the indirect command buffer cases' task: its output
    /// buffer, the buffer under its texture and its range buffer.
    const ICB_TASK: [Step; 3] = [Make("out", P1), Make("tb", P1), Make("rng", P1)];

    /// The objects of one decode walk task (`decodewalk_util::DTask`, an isolate walk), as every
    /// exec lists them: seven one-page buffers, the two-page reply buffer and the one-page heap.
    const fn walk_task(
        a: &'static str,
        b: &'static str,
        src: &'static str,
        icbout: &'static str,
        heap: &'static str,
    ) -> [Step; 9] {
        [
            Make(a, P1),
            Make(b, P1),
            Make(src, P1),
            Make("args", P1),
            Make("tba", P1),
            Make("tbb", P1),
            Make(icbout, P1),
            Make("reply", P2),
            Make(heap, P1),
        ]
    }

    /// Every address the 37 isolate cases that differ only in addresses record, and the two
    /// passing cases that pin, as the space places them. Steps whose resource no record shows
    /// are kept where they move the recorded ones.
    fn cases() -> Vec<(&'static str, Vec<Step>)> {
        let first_buffer = |name| (name, vec![Make("b", P1), Recorded("b", 0, START)]);
        let icb_four = |name, records: Vec<Step>| {
            let mut steps: Vec<Step> = ICB_TASK.into_iter().collect();
            steps.push(Make("fourth", P1));
            steps.extend(records);
            (name, steps)
        };
        let placed_kinds = |name| {
            (
                name,
                vec![
                    // The 4x4 base texture is made first, suballocated.
                    Tex(RGBA8_4X4),
                    Make("sh", P1),
                    Make("h", P4),
                    Make("b", P1),
                    Recorded("sh", 0, 0x100_0008_0000),
                    Recorded("h", 0, 0x100_0008_8000),
                    Recorded("b", 0, 0x100_000a_0000),
                ],
            )
        };
        let walk = |task: [Step; 9]| -> Vec<Step> { task.into_iter().collect() };
        let concat = |parts: Vec<Vec<Step>>| -> Vec<Step> { parts.into_iter().flatten().collect() };
        vec![
            first_buffer("compute_pipeline_placement"),
            icb_four(
                "icb_evict",
                vec![
                    Recorded("out", 0x10, 0x100_0008_0010),
                    Recorded("tb", 0x20, 0x100_0008_8020),
                    Recorded("out", 0x30, 0x100_0008_0030),
                    Recorded("out", 0x40, 0x100_0008_0040),
                    Recorded("tb", 0x50, 0x100_0008_8050),
                    Recorded("fourth", 2, 0x100_0009_8002),
                    Recorded("fourth", 8, 0x100_0009_8008),
                    Recorded("fourth", 0, 0x100_0009_8000),
                ],
            ),
            icb_four(
                "icb_sync",
                vec![
                    Recorded("out", 0x10, 0x100_0008_0010),
                    Recorded("tb", 0x20, 0x100_0008_8020),
                    Recorded("out", 0x30, 0x100_0008_0030),
                    Recorded("out", 0x40, 0x100_0008_0040),
                    Recorded("tb", 0x50, 0x100_0008_8050),
                    Recorded("fourth", 2, 0x100_0009_8002),
                    Recorded("fourth", 8, 0x100_0009_8008),
                    Recorded("fourth", 0, 0x100_0009_8000),
                ],
            ),
            (
                "icb_sync_after_blit",
                vec![Make("out", P1), Recorded("out", 0, 0x100_0008_0000)],
            ),
            (
                // The reply buffer is read through a cursor and made by no one; the pipelines,
                // the rate map and the indirect command buffer made among the replies take none.
                "info_replies",
                vec![
                    Make("data", P1),
                    Make("heap", P4),
                    Recorded("data", 0, 0x100_0008_0000),
                    Recorded("heap", 0, 0x100_0008_8000),
                ],
            ),
            (
                "info_replies_more",
                vec![
                    Make("data", P1),
                    Make("heap", P4),
                    Recorded("data", 0, 0x100_0008_0000),
                    Recorded("heap", 0, 0x100_0008_8000),
                ],
            ),
            (
                // The fill's two-page buffer, then the first blit's work, then the reply's.
                "info_reply_before_error",
                vec![
                    Make("param", P2),
                    Use(BlitCommand),
                    Make("data", P1),
                    Recorded("data", 0, 0x100_0011_8000),
                ],
            ),
            (
                "interplay_decodewalk_41",
                concat(vec![
                    walk(walk_task("a", "b", "src", "icbout", "heap")),
                    vec![
                        Recorded("heap", 0, 0x100_000c_8000),
                        Recorded("a", 0, 0x100_0008_0000),
                    ],
                ]),
            ),
            (
                "interplay_decodewalk_42",
                concat(vec![
                    walk(walk_task("a", "b", "src", "icbout", "heap")),
                    // Task 1's work; the blit resets an indirect command buffer. Then Metal's
                    // own buffers grow by 0x10000, which the model does not make.
                    vec![
                        Use(ComputeEncoder),
                        Use(BlitCommand),
                        Use(TilerBlit),
                        Taken(0x1_0000),
                        Use(TilerSubmission),
                    ],
                    walk(walk_task("a2", "b2", "src2", "icbout2", "heap2")),
                    vec![
                        Recorded("a", 0, 0x100_0008_0000),
                        Recorded("icbout2", 0, 0x100_0044_8000),
                        Recorded("icbout", 0, 0x100_000b_0000),
                        Recorded("heap", 0, 0x100_000c_8000),
                        Recorded("b", 0, 0x100_0008_8000),
                        Recorded("heap2", 0, 0x100_0046_0000),
                        Recorded("src", 0, 0x100_0009_0000),
                    ],
                ]),
            ),
            (
                "interplay_decodewalk_43",
                concat(vec![
                    walk(walk_task("a", "b", "src", "icbout", "heap")),
                    vec![
                        Recorded("heap", 0, 0x100_000c_8000),
                        Recorded("icbout", 0, 0x100_000b_0000),
                        Recorded("src", 0, 0x100_0009_0000),
                        Recorded("b", 0, 0x100_0008_8000),
                        Recorded("a", 0, 0x100_0008_0000),
                    ],
                ]),
            ),
            (
                // Task 2 runs first.
                "interplay_decodewalk_44",
                concat(vec![
                    walk(walk_task("a2", "b2", "src2", "icbout2", "heap2")),
                    vec![Use(BlitCommand), Use(ComputeEncoder)],
                    walk(walk_task("a", "b", "src", "icbout", "heap")),
                    vec![
                        Recorded("icbout", 0, 0x100_001b_8000),
                        Recorded("heap", 0, 0x100_001d_0000),
                        Recorded("icbout2", 0, 0x100_000b_0000),
                        Recorded("b", 0, 0x100_0019_0000),
                        Recorded("a2", 0, 0x100_0008_0000),
                        Recorded("heap2", 0, 0x100_000c_8000),
                        Recorded("b2", 0, 0x100_0008_8000),
                        Recorded("src", 0, 0x100_0019_8000),
                        Recorded("src2", 0, 0x100_0009_0000),
                        Recorded("a", 0, 0x100_0018_8000),
                    ],
                ]),
            ),
            (
                "interplay_decodewalk_51",
                concat(vec![
                    walk(walk_task("a", "b", "src", "icbout", "heap")),
                    vec![
                        Recorded("icbout", 0, 0x100_000b_0000),
                        Recorded("b", 0, 0x100_0008_8000),
                        Recorded("heap", 0, 0x100_000c_8000),
                        Recorded("a", 0, 0x100_0008_0000),
                    ],
                ]),
            ),
            (
                // Task 2 runs first.
                "interplay_decodewalk_52",
                concat(vec![
                    walk(walk_task("a2", "b2", "src2", "icbout2", "heap2")),
                    // Task 2's work: a blit resets an indirect command buffer, Metal's own
                    // buffers grow by 0x60000, which the model does not make, and a render
                    // encoder.
                    vec![
                        Use(BlitCommand),
                        Use(TilerBlit),
                        Taken(0x6_0000),
                        Use(TilerSubmission),
                        Use(RenderEncoder),
                    ],
                    walk(walk_task("a", "b", "src", "icbout", "heap")),
                    vec![
                        Recorded("icbout2", 0, 0x100_000b_0000),
                        Recorded("heap2", 0, 0x100_000c_8000),
                        Recorded("a", 0, 0x100_0063_8000),
                        Recorded("icbout", 0, 0x100_0066_8000),
                        Recorded("heap", 0, 0x100_0068_0000),
                        Recorded("src2", 0, 0x100_0009_0000),
                    ],
                ]),
            ),
            first_buffer("interplay_diff_info_opcodes"),
            (
                "mutgap_mem_icb_binds_at_end",
                vec![Make("out", P1), Recorded("out", 0x10, 0x100_0008_0010)],
            ),
            icb_four(
                "mutgap_tex_icb_sync_more",
                vec![
                    Recorded("fourth", 8, 0x100_0009_8008),
                    Recorded("out", 0x40, 0x100_0008_0040),
                    Recorded("tb", 0x50, 0x100_0008_8050),
                    Recorded("out", 0x30, 0x100_0008_0030),
                    Recorded("tb", 0x40, 0x100_0008_8040),
                ],
            ),
            first_buffer("mutgap_tex_placement_delete_function"),
            first_buffer("mutgap_tex_placement_reservations"),
            first_buffer("mutgap_tex_placement_views_used"),
            (
                // The target's buffer, then the first of six render execs that fault before
                // their commit, then heap 1.
                "mutgap2_rdec_use_heaps_count_placement",
                vec![
                    Make("target", P1),
                    Use(RenderEncoder),
                    Make("heap 1", P1),
                    Recorded("heap 1", 0, 0x100_0036_0000),
                ],
            ),
            (
                // The bind is at 0x10 of a heap buffer at 0x100 of the heap.
                "mutgap2_res_icb_heap_buffer_bind",
                concat(vec![
                    ICB_TASK.into_iter().collect(),
                    vec![Make("h", P4), Recorded("h", 0x110, 0x100_0009_8110)],
                ]),
            ),
            (
                "mutgap2_res_icb_render_two_strides",
                concat(vec![
                    ICB_TASK.into_iter().collect(),
                    vec![
                        Recorded("out", 0x40, 0x100_0008_0040),
                        Recorded("tb", 0x50, 0x100_0008_8050),
                    ],
                ]),
            ),
            first_buffer("mutgap2_res_mapper_rotation_zero_entry"),
            first_buffer("mutgap2_res_placement_delete_objects"),
            (
                // The target's buffer, the shared buffer and the copies' buffer; a render pass,
                // Metal's own buffers grown by 0x48000, which the model does not make, and two
                // blits; then the heap.
                "mutgap2_res_placement_forced_used",
                vec![
                    Make("target", P1),
                    Make("sh", P1),
                    Make("b", P1),
                    Use(RenderEncoder),
                    Use(TilerSubmission),
                    Taken(0x4_8000),
                    Use(BlitCommand),
                    Make("h", P4),
                    Recorded("target", 0, 0x100_0008_0000),
                    Recorded("sh", 0, 0x100_0008_8000),
                    Recorded("h", 0, 0x100_005e_8000),
                    Recorded("b", 0, 0x100_0009_0000),
                ],
            ),
            first_buffer("mutgap2_res_resource_lists_placement"),
            (
                // The first 4x4 shared backing, made before the buffer.
                "mutgap2_res_shared_backing_placed",
                vec![
                    Tex(shareable(0x40)),
                    Make("b", P1),
                    Recorded("b", 0, 0x100_0008_8000),
                ],
            ),
            (
                "mutgap3_res_icb_sync_spare_slot",
                vec![Make("out", P1), Recorded("out", 0x10, 0x100_0008_0010)],
            ),
            icb_four(
                "mutgap4_res_icb_sync_patch_start_tgmem",
                vec![
                    Recorded("fourth", 0, 0x100_0009_8000),
                    Recorded("out", 0, 0x100_0008_0000),
                ],
            ),
            first_buffer("mutgap4_svc_placement_unmakeable_buffer"),
            (
                // The 4x4 texture is made first, suballocated; the buffer is deleted (40,
                // operation 1000) and made again.
                "resources_gap_placement_delete",
                vec![
                    Tex(RGBA8_4X4),
                    Make("bf", P1),
                    Make("b", P1),
                    Recorded("bf", 0, 0x100_0008_0000),
                    Recorded("b", 0, 0x100_0008_8000),
                    Release("bf"),
                    Recorded("b", 0, 0x100_0008_8000),
                    Make("bf", P1),
                    Recorded("bf", 0, 0x100_0008_0000),
                    Recorded("b", 0, 0x100_0008_8000),
                ],
            ),
            first_buffer("resources_mapper_placement"),
            placed_kinds("textures_gap_misplaced_sync"),
            (
                // The placed kinds, then the guest's pinned buffer, made by the last exec.
                "textures_placement_forced",
                concat(vec![
                    placed_kinds("").1,
                    vec![
                        Pin("pinned", PINNED, P1),
                        Recorded("pinned", 0, 0x100_0040_0000),
                    ],
                ]),
            ),
            placed_kinds("textures_placement_misplaced"),
            (
                // Task 0's shareable 4x4 backing, the no-copy buffer over the paged texture's
                // memory, then the paging blit's work.
                "textures_placement_shared",
                vec![
                    Tex(shareable(0x40)),
                    Make("paged", P1),
                    Use(BlitCommand),
                    Make("b", P1),
                    Recorded("b", 0, 0x100_0011_8000),
                ],
            ),
            (
                // Two 4x4 textures, suballocated, and a 4x4 Depth32Float_Stencil8 one, of
                // 0x4080 bytes.
                "textures_placement_written",
                vec![
                    Tex(RGBA8_4X4),
                    Tex(RGBA8_4X4),
                    Tex(own(0x4080)),
                    Make("sh", P1),
                    Make("h", P4),
                    Make("b", P1),
                    Recorded("sh", 0, 0x100_0009_0000),
                    Recorded("h", 0, 0x100_0009_8000),
                    Recorded("b", 0, 0x100_000b_0000),
                ],
            ),
            (
                // Passing: both pins are accepted, and Metal makes neither of 2^36 bytes.
                "mutgap2_res_heap_pinned_huge",
                vec![
                    Make("b", P1),
                    PinUnmade(PINNED + 0x20_0000, 1 << 36),
                    PinUnmade(PINNED + 0x40_0000, 1 << 36),
                ],
            ),
            (
                // Passing: the pin far from the task's buffers is made.
                "mutgap2_res_icb_bind_at_buffer_end",
                concat(vec![
                    ICB_TASK.into_iter().collect(),
                    vec![
                        Pin("p", PINNED + 0x30_0000, P1),
                        Recorded("p", 0x4000, 0x100_0070_4000),
                    ],
                ]),
            ),
        ]
    }

    /// The pools after the first a full shared or private one makes in `vaprobe smallpool`, with
    /// where the probe after each lay.
    const EXTRA_POOLS: [(u64, u64); 4] = [
        (0x8_0000, 0xa_8000),
        (0xa_8000, 0xd_0000),
        (0xd_0000, 0xf_8000),
        (0xf_8000, 0x12_0000),
    ];

    /// `vaprobe smallpool`'s row, of buffers of pool class `class`, after three queues
    /// (`MetalSpace::slot`): a 0x100-byte one at `first`; each of the lengths `pairs` gives, with
    /// a 0x100-byte one after it, at its slots; four of 0x100 bytes from `reuse`, the second
    /// released, then one of 0x100 at its slot, the third released, then one of 0x200 past the
    /// fourth and one of 0x100 at the third's slot; then ones of 0x100 bytes, each at the step
    /// after the one before to the first pool's end `pool_end`, then at the first byte of each of
    /// the four `pools`, 512 to a pool, a buffer made and released at its probe after each move,
    /// and once more after the last.
    fn smallpool(
        class: PoolClass,
        first: u64,
        pairs: [(u64, u64, u64); 14],
        reuse: u64,
        pool_end: u64,
        pools: [(u64, u64); 4],
    ) -> Vec<Step> {
        let small = |name, length, offset| SmallAt(name, length, class, BASE + offset);
        let mut steps = vec![Use(Queue), small("first", 0x100, first)];
        for (length, at, next) in pairs {
            steps.extend([small("length", length, at), small("next", 0x100, next)]);
        }
        steps.extend([
            small("0", 0x100, reuse),
            small("1", 0x100, reuse + 0x100),
            small("2", 0x100, reuse + 0x200),
            small("3", 0x100, reuse + 0x300),
            Release("1"),
            small("1 again", 0x100, reuse + 0x100),
            Release("2"),
            small("0x200", 0x200, reuse + 0x400),
            small("2 again", 0x100, reuse + 0x200),
        ]);
        let fill = |from: u64, end: u64| {
            (from..end)
                .step_by(0x100)
                .map(|at| small("fill", 0x100, at))
        };
        steps.extend(fill(reuse + 0x600, pool_end));
        let [first_more, second, third, (last, probe)] = pools;
        for (pool, probe) in [first_more, second, third] {
            steps.extend([small("fill", 0x100, pool), Next(BASE + probe)]);
            steps.extend(fill(pool + 0x100, pool + POOL_BYTES));
        }
        steps.extend([
            small("fill", 0x100, last),
            Next(BASE + probe),
            Next(BASE + probe),
        ]);
        steps
    }

    /// `vaprobe texpool2`'s row: after three queues, a buffer made and released; `count` private
    /// RGBA8 textures of `heap_size`, made and kept, a buffer made and released after each, at the
    /// address the last of `moves` at or before that texture gives; then every texture released,
    /// and a buffer made and released at `released` (`Elements`, `PoolClass`).
    fn texpool(heap_size: u64, count: u64, moves: &[(u64, u64)], released: u64) -> Vec<Step> {
        let mut steps = vec![Use(Queue), Next(BASE + 0x8_0000)];
        let mut probe = 0x8_0000;
        for n in 1..=count {
            if let Some(&(_, at)) = moves.iter().find(|&&(moved, _)| moved == n) {
                probe = at;
            }
            steps.extend([Tex(own(heap_size)), Next(BASE + probe)]);
        }
        steps.extend([ReleaseTextures, Next(BASE + released)]);
        steps
    }

    /// `vaprobe samplers`' row: after three queues, the sampler descriptor mempool's first array
    /// at the setup's; 1024 sampler states made and kept, a one-page buffer made and released
    /// after each, and the array where each growth put it (`Samplers`).
    fn samplers() -> Vec<Step> {
        let mut steps = vec![
            Use(Queue),
            SamplerArray(BASE + 0x1_8500),
            Next(BASE + 0x8_0000),
        ];
        let grown = [
            (15, 0x1_c500),
            (31, 0x1_c900),
            (63, 0x1_d100),
            (127, 0x1_e100),
            (255, 0x8_0000),
            (511, 0x8_8000),
            (1023, 0x9_8000),
        ];
        let mut probe = 0x8_0000;
        for n in 1..=1024 {
            steps.push(Sampler);
            if let Some(&(_, array)) = grown.iter().find(|&&(at, _)| at == n) {
                steps.push(SamplerArray(BASE + array));
            }
            probe = match n {
                255 => 0x8_8000,
                511 => 0x8_0000,
                _ => probe,
            };
            steps.push(Next(BASE + probe));
        }
        steps
    }

    /// `vaprobe placedsampler`'s row, one process for each ID: after three queues, the next
    /// one-page range and a 0x100-byte shared buffer's slot, made and released, then a sampler
    /// state of argument buffer support placed at the ID (15, 16, 20, 32, 600 or 2000), then the
    /// same two again, which neither moved (`Samplers`).
    fn placed_sampler() -> Vec<Step> {
        let slot = |name| SmallAt(name, 0x100, PoolClass::Shared, BASE + 0x1_c500);
        vec![
            Use(Queue),
            Next(BASE + 0x8_0000),
            slot("before"),
            Release("before"),
            PlacedSampler,
            Next(BASE + 0x8_0000),
            slot("after"),
            Release("after"),
        ]
    }

    /// Every row of the probes that measured the space, from a process that made nothing
    /// before it but its device (`vaprobe`, plain Metal, one process each), as the space places
    /// them. Addresses are [`BASE`] plus the probe's offsets.
    fn probe_rows() -> Vec<(&'static str, Vec<Step>)> {
        let at = |length, offset| At(length, BASE + offset);
        let next = |offset| Next(BASE + offset);
        let refused = |offset| Refused(BASE + offset, P1);
        let pin = |name, offset| Pin(name, BASE + offset, P1);
        let plain = |length| vec![at(length, 0x1_8000)];
        // The bring-up, and a one-page buffer at [`START`].
        let far = || vec![Use(Queue), at(P1, 0x8_0000)];
        let pin3 = || {
            vec![
                Use(Queue),
                Pin("3", BASE + 0x40_0000, 3 * PAGE),
                Recorded("3", 0, BASE + 0x40_0000),
            ]
        };
        let concat = |parts: Vec<Vec<Step>>| -> Vec<Step> { parts.into_iter().flatten().collect() };
        let six = |second| {
            vec![
                Make("0", P1),
                Make("1", second),
                Make("2", P1),
                Make("3", P1),
                Make("4", P1),
                Make("5", P1),
            ]
        };
        let rgba8 = |side: u64| own(side * side * 4);
        vec![
            (
                // Each buffer kept; the lengths each with a 0x100-byte buffer after it.
                "vaprobe smallpool shared",
                smallpool(
                    PoolClass::Shared,
                    0x1_c500,
                    [
                        (1, 0x1_c600, 0x1_c700),
                        (0x40, 0x1_c800, 0x1_c900),
                        (0x80, 0x1_ca00, 0x1_cb00),
                        (0xff, 0x1_cc00, 0x1_cd00),
                        (0x100, 0x1_ce00, 0x1_cf00),
                        (0x101, 0x1_d000, 0x1_d200),
                        (0x180, 0x1_d300, 0x1_d500),
                        (0x200, 0x1_d600, 0x1_d800),
                        (0x300, 0x1_d900, 0x1_dc00),
                        (0x1000, 0x1_dd00, 0x1_ed00),
                        (0x1001, 0x1_ee00, 0x1_ff00),
                        (0x2000, 0x2_0000, 0x2_2000),
                        (0x3000, 0x2_2100, 0x2_5100),
                        (0x3fff, 0x2_5200, 0x2_9200),
                    ],
                    0x2_9300,
                    0x3_8000,
                    EXTRA_POOLS,
                ),
            ),
            (
                "vaprobe smallpool private",
                smallpool(
                    PoolClass::Private,
                    0x5_8100,
                    [
                        (1, 0x5_8200, 0x5_8300),
                        (0x40, 0x5_8400, 0x5_8500),
                        (0x80, 0x5_8600, 0x5_8700),
                        (0xff, 0x5_8800, 0x5_8900),
                        (0x100, 0x5_8a00, 0x5_8b00),
                        (0x101, 0x5_8c00, 0x5_8e00),
                        (0x180, 0x5_8f00, 0x5_9100),
                        (0x200, 0x5_9200, 0x5_9400),
                        (0x300, 0x5_9500, 0x5_9800),
                        (0x1000, 0x5_9900, 0x5_a900),
                        (0x1001, 0x5_aa00, 0x5_bb00),
                        (0x2000, 0x5_bc00, 0x5_dc00),
                        (0x3000, 0x5_dd00, 0x6_0d00),
                        (0x3fff, 0x6_0e00, 0x6_4e00),
                    ],
                    0x6_4f00,
                    0x7_8000,
                    EXTRA_POOLS,
                ),
            ),
            (
                // The managed pool is made at the first buffer, past the setup's ranges.
                "vaprobe smallpool managed",
                smallpool(
                    PoolClass::Managed,
                    0x8_0000,
                    [
                        (1, 0x8_0100, 0x8_0200),
                        (0x40, 0x8_0300, 0x8_0400),
                        (0x80, 0x8_0500, 0x8_0600),
                        (0xff, 0x8_0700, 0x8_0800),
                        (0x100, 0x8_0900, 0x8_0a00),
                        (0x101, 0x8_0b00, 0x8_0d00),
                        (0x180, 0x8_0e00, 0x8_1000),
                        (0x200, 0x8_1100, 0x8_1300),
                        (0x300, 0x8_1400, 0x8_1700),
                        (0x1000, 0x8_1800, 0x8_2800),
                        (0x1001, 0x8_2900, 0x8_3a00),
                        (0x2000, 0x8_3b00, 0x8_5b00),
                        (0x3000, 0x8_5c00, 0x8_8c00),
                        (0x3fff, 0x8_8d00, 0x8_cd00),
                    ],
                    0x8_ce00,
                    0xa_0000,
                    [
                        (0xa_8000, 0xd_0000),
                        (0xd_0000, 0xf_8000),
                        (0xf_8000, 0x12_0000),
                        (0x12_0000, 0x14_8000),
                    ],
                ),
            ),
            (
                // 4x4, a heap size of 0x80: 511 to the first pool, 512 to each after it.
                "vaprobe texpool2 4x4",
                texpool(
                    0x80,
                    2048,
                    &[
                        (256, 0x9_0000),
                        (512, 0x8_0000),
                        (1024, 0x10_8000),
                        (1536, 0x13_0000),
                        (2048, 0x8_0000),
                    ],
                    0x8_0000,
                ),
            ),
            (
                // 16x16, 0x400: 127 to the first pool, 128 to each after it.
                "vaprobe texpool2 16x16",
                texpool(
                    0x400,
                    2048,
                    &[
                        (128, 0xa_8000),
                        (256, 0xe_0000),
                        (384, 0x10_8000),
                        (512, 0xa_8000),
                        (1024, 0xb_0000),
                        (2048, 0xa_8000),
                    ],
                    0x8_0000,
                ),
            ),
            (
                // 32x64, 0x2000: 15 to the first pool, 16 to each after it; the probe stopped
                // at eight moves.
                "vaprobe texpool2 32x64",
                texpool(
                    0x2000,
                    128,
                    &[
                        (16, 0xa_8000),
                        (32, 0xd_0000),
                        (48, 0xf_8000),
                        (64, 0x12_0000),
                        (80, 0x14_8000),
                        (96, 0x17_0000),
                        (112, 0x19_8000),
                        (128, 0x1c_0000),
                    ],
                    0x8_0000,
                ),
            ),
            (
                // After three queues, four 4x4 RGBA8 textures of each storage mode, kept: shared
                // ones past the setup's slots in the shared pool, managed ones in a pool of their
                // own made at the first, private ones past the setup's slot in the private pool;
                // then a one-page buffer past the managed pool.
                "vaprobe smalltex",
                [
                    vec![Use(Queue)],
                    (0..4)
                        .map(|i| TexAt(own_shared(0x80), BASE + 0x1_c500 + i * 0x100))
                        .collect(),
                    (0..4)
                        .map(|i| TexAt(own_managed(0x80), BASE + 0x8_0000 + i * 0x100))
                        .collect(),
                    (0..4)
                        .map(|i| TexAt(own(0x80), BASE + 0x5_8100 + i * 0x100))
                        .collect(),
                    vec![Next(BASE + 0xa_8000)],
                ]
                .into_iter()
                .flatten()
                .collect(),
            ),
            (
                // After three queues, sampler states made and kept, a one-page buffer made and
                // released after each: the array grows at the 15th, 31st and so on, the device's
                // own sampler counted, in the shared pool to 0x2000 bytes, then in ranges.
                "vaprobe samplers",
                samplers(),
            ),
            ("vaprobe placedsampler 15", placed_sampler()),
            ("vaprobe placedsampler 16", placed_sampler()),
            ("vaprobe placedsampler 20", placed_sampler()),
            ("vaprobe placedsampler 32", placed_sampler()),
            ("vaprobe placedsampler 600", placed_sampler()),
            ("vaprobe placedsampler 2000", placed_sampler()),
            (
                // Each buffer and heap kept.
                "vaprobe sizes",
                vec![
                    // The 1-byte buffer lies in Metal's pool of small shared buffers, which it
                    // made for it.
                    Small(1, PoolClass::Shared),
                    at(0x4000, 0x4_0000),
                    at(0x4001, 0x4_8000),
                    at(0x8000, 0x5_8000),
                    at(0x8001, 0x6_8000),
                    at(0xc000, 0x7_8000),
                    at(0x1_0000, 0x8_8000),
                    at(0x1_4000, 0xa_0000),
                    at(0x1_c000, 0xb_8000),
                    at(0x4_0000, 0xd_8000),
                    at(0x10_0000, 0x12_0000),
                    at(0x4000, 0x22_8000),
                    // Private.
                    at(0x4000, 0x23_0000),
                    at(0x8000, 0x23_8000),
                    at(0xc000, 0x24_8000),
                    at(0x1_4000, 0x25_8000),
                    // No-copy.
                    at(0x8000, 0x27_0000),
                    // Heaps of one to five pages.
                    at(P1, 0x28_0000),
                    at(P2, 0x28_8000),
                    at(3 * PAGE, 0x29_8000),
                    at(P4, 0x2a_8000),
                    at(5 * PAGE, 0x2c_0000),
                    at(P1, 0x2d_8000),
                ],
            ),
            (
                // After a queue: shared buffers under a page, each followed by a one-page buffer,
                // kept; then private ones, then shared ones of the descriptor path.
                "vaprobe small",
                concat(vec![
                    vec![Use(Queue), at(P1, 0x8_0000)],
                    [false, true, false]
                        .into_iter()
                        .zip([0x8_8000, 0xc_0000, 0xf_8000])
                        .flat_map(|(private, from)| {
                            let lengths: &[u64] = match from {
                                0xf_8000 => &[0x100, 0x3fff],
                                _ => &[1, 0x100, 0x1000, 0x2000, 0x3fff],
                            };
                            let count = u64::try_from(lengths.len()).unwrap();
                            lengths
                                .iter()
                                .zip(0..)
                                .flat_map(move |(&length, i)| {
                                    [Small(length, class(private)), at(P1, from + i * SLOT)]
                                })
                                .chain([
                                    at(0x4000, from + count * SLOT),
                                    at(P1, from + count * SLOT + SLOT),
                                ])
                                .collect::<Vec<_>>()
                        })
                        .collect(),
                ]),
            ),
            (
                "vaprobe reuse",
                vec![
                    Make("a", P1),
                    Make("b", P1),
                    Make("c", P1),
                    Make("x", P2),
                    Make("y", P1),
                    Recorded("a", 0, BASE + 0x1_8000),
                    Recorded("y", 0, BASE + 0x4_0000),
                    Release("b"),
                    at(P1, 0x2_0000),
                    Release("a"),
                    at(P2, 0x4_8000),
                    at(P1, 0x1_8000),
                    Release("x"),
                    at(P1, 0x3_0000),
                    at(P1, 0x3_8000),
                    at(3 * PAGE, 0x5_8000),
                ],
            ),
            (
                "vaprobe order lowhigh",
                concat(vec![
                    six(P1),
                    vec![
                        Recorded("5", 0, BASE + 0x4_0000),
                        Release("1"),
                        Release("3"),
                        at(P1, 0x2_0000),
                        at(P1, 0x3_0000),
                        at(P1, 0x4_8000),
                    ],
                ]),
            ),
            (
                "vaprobe order highlow",
                concat(vec![
                    six(P1),
                    vec![
                        Release("3"),
                        Release("1"),
                        at(P1, 0x2_0000),
                        at(P1, 0x3_0000),
                        at(P1, 0x4_8000),
                    ],
                ]),
            ),
            (
                // A freed two-page range below a freed one-page one: the first one-page buffer
                // splits the lower.
                "vaprobe order fit",
                concat(vec![
                    six(P2),
                    vec![
                        Recorded("5", 0, BASE + 0x4_8000),
                        Release("1"),
                        Release("3"),
                        at(P1, 0x2_0000),
                        at(P1, 0x2_8000),
                        at(P1, 0x3_8000),
                    ],
                ]),
            ),
            (
                "vaprobe pins before",
                vec![
                    at(P1, 0x1_8000),
                    at(P1, 0x2_0000),
                    // Its guard page on the second's first page, then a page lower.
                    refused(0x1_c000),
                    refused(0x1_8000),
                    next(0x2_8000),
                ],
            ),
            (
                "vaprobe pins guard1",
                concat(vec![plain(P1), vec![refused(0x1_c000), next(0x2_0000)]]),
            ),
            (
                "vaprobe pins slot1",
                concat(vec![plain(P1), vec![pin("p", 0x2_0000), next(0x2_8000)]]),
            ),
            (
                "vaprobe pins guard2",
                concat(vec![plain(P2), vec![refused(0x2_0000), next(0x2_8000)]]),
            ),
            (
                "vaprobe pins spare2",
                concat(vec![plain(P2), vec![refused(0x2_4000), next(0x2_8000)]]),
            ),
            (
                "vaprobe pins guard3a",
                concat(vec![
                    plain(3 * PAGE),
                    vec![refused(0x2_4000), next(0x2_8000)],
                ]),
            ),
            (
                "vaprobe pins guard3b",
                concat(vec![
                    plain(3 * PAGE),
                    vec![pin("p", 0x2_8000), next(0x3_0000)],
                ]),
            ),
            (
                // Free, but off its granule.
                "vaprobe pins guard3c",
                concat(vec![
                    plain(3 * PAGE),
                    vec![refused(0x2_c000), next(0x2_8000)],
                ]),
            ),
            (
                "vaprobe pins guard3c-bringup",
                vec![
                    Use(Queue),
                    at(3 * PAGE, 0x8_0000),
                    refused(0x9_4000),
                    next(0x9_0000),
                ],
            ),
            (
                "vaprobe pins guard5",
                concat(vec![
                    plain(5 * PAGE),
                    vec![pin("p", 0x3_0000), next(0x3_8000)],
                ]),
            ),
            // Far from every range: on a granule's first page, at its first byte and 0x10 into
            // it, and on its second page.
            (
                "vaprobe pins far-0",
                concat(vec![
                    far(),
                    vec![
                        pin("p", 0x40_0000),
                        Recorded("p", 0, BASE + 0x40_0000),
                        next(0x8_8000),
                    ],
                ]),
            ),
            (
                "vaprobe pins far-10",
                concat(vec![
                    far(),
                    vec![
                        pin("p", 0x40_0010),
                        Recorded("p", 0, BASE + 0x40_0010),
                        next(0x8_8000),
                    ],
                ]),
            ),
            (
                "vaprobe pins far-4000",
                concat(vec![far(), vec![refused(0x40_4000), next(0x8_8000)]]),
            ),
            (
                "vaprobe pins far-4010",
                concat(vec![far(), vec![refused(0x40_4010), next(0x8_8000)]]),
            ),
            // A three-page pin, then a one-page pin after it: inside its span, at its end, off
            // a granule past it, and a granule past it.
            (
                "vaprobe pins pin3-c000",
                concat(vec![pin3(), vec![refused(0x40_c000), next(0x8_0000)]]),
            ),
            (
                "vaprobe pins pin3-10000",
                concat(vec![pin3(), vec![pin("p", 0x41_0000), next(0x8_0000)]]),
            ),
            (
                "vaprobe pins pin3-14000",
                concat(vec![pin3(), vec![refused(0x41_4000), next(0x8_0000)]]),
            ),
            (
                "vaprobe pins pin3-18000",
                concat(vec![pin3(), vec![pin("p", 0x41_8000), next(0x8_0000)]]),
            ),
            (
                "vaprobe pins released",
                vec![
                    Make("a", P1),
                    Recorded("a", 0, BASE + 0x1_8000),
                    at(P1, 0x2_0000),
                    Release("a"),
                    // Its old guard page, whose own guard page is the second's first page; then
                    // its old first page.
                    refused(0x1_c000),
                    pin("p", 0x1_8000),
                    next(0x2_8000),
                ],
            ),
            (
                // Each texture kept, then a buffer, kept.
                "vaprobe textures",
                vec![
                    at(P1, 0x1_8000),
                    // 4x4 RGBA8 private, the first texture: Metal's pools.
                    Tex(RGBA8_4X4),
                    at(P1, 0x8_8000),
                    // Shared, and a render target.
                    Tex(own_shared(0x80)),
                    at(P1, 0x9_0000),
                    Tex(RGBA8_4X4),
                    at(P1, 0x9_8000),
                    Tex(rgba8(64)),
                    at(P1, 0xa_8000),
                    Tex(rgba8(256)),
                    at(P1, 0xf_8000),
                    Tex(own(0x4080)),
                    at(P1, 0x11_0000),
                    Tex(TextureKind::Memoryless),
                    at(P1, 0x11_8000),
                    // A 4x4 base, a buffer, its view, a buffer.
                    Tex(RGBA8_4X4),
                    at(P1, 0x12_0000),
                    Tex(TextureKind::View),
                    at(P1, 0x12_8000),
                    // A buffer, a buffer, a texture over the first, a buffer.
                    at(P1, 0x13_0000),
                    at(P1, 0x13_8000),
                    Tex(TextureKind::OverBuffer),
                    at(P1, 0x14_0000),
                    // Over a 4x4 surface.
                    Surface("s", PAGE),
                    Recorded("s", 0, BASE + 0x14_8000),
                    at(P1, 0x15_0000),
                    // A sampler, a depth-stencil state, an indirect command buffer, a library, a
                    // compute pipeline state, then a queue.
                    at(P1, 0x15_8000),
                    at(P1, 0x16_0000),
                    at(P1, 0x16_8000),
                    at(P1, 0x17_0000),
                    at(P1, 0x17_8000),
                    Use(Queue),
                    at(P1, 0x18_0000),
                ],
            ),
            (
                // After a bring-up, a buffer; then each texture of its heap size, private, kept,
                // then a buffer, kept (`OWN_HEAP_SIZE`'s rows).
                "vaprobe texsizes",
                concat(vec![
                    vec![Use(Queue), at(P1, 0x8_0000)],
                    [
                        (0x80, 0x8_8000),
                        (0x80, 0x9_0000),
                        (0x80, 0x9_8000),
                        (0x4080, 0xb_0000),
                        (0x2000, 0xb_8000),
                        (0x4000, 0xc_8000),
                        (0x4080, 0xe_0000),
                        (0x4080, 0xf_8000),
                        (0x5680, 0x11_0000),
                        (0x1_0200, 0x13_0000),
                        (0x4_0800, 0x18_0000),
                    ]
                    .into_iter()
                    .flat_map(|(heap_size, next)| [Tex(own(heap_size)), at(P1, next)])
                    .collect(),
                ]),
            ),
            (
                // After a bring-up, a buffer; then each shareable BGRA8 texture by its
                // allocation, kept, then a buffer, kept (`SHAREABLE_EXTRA`'s rows).
                "vaprobe shared",
                concat(vec![
                    vec![Use(Queue), at(P1, 0x8_0000)],
                    [
                        (0x40, 0x9_0000),
                        (0x400, 0xa_0000),
                        (0x1000, 0xb_0000),
                        (0x4000, 0xc_8000),
                        (0x4000, 0xe_0000),
                        (0x8000, 0xf_8000),
                        (0x1_0000, 0x11_8000),
                        (0x4_0000, 0x16_8000),
                    ]
                    .into_iter()
                    .flat_map(|(allocation, next)| [Tex(shareable(allocation)), at(P1, next)])
                    .collect(),
                ]),
            ),
            (
                // Each made, a buffer over each surface after its texture (at the surface's
                // address), each kept, then a buffer, kept.
                "vaprobe surfaces",
                vec![
                    at(P1, 0x1_8000),
                    // A 4x4 surface, the first texture: Metal's pools.
                    Surface("4x4", PAGE),
                    Recorded("4x4", 0, BASE + 0x8_8000),
                    at(P1, 0x9_0000),
                    at(P1, 0x9_8000),
                    Surface("64x64", PAGE),
                    Recorded("64x64", 0, BASE + 0xa_0000),
                    at(P1, 0xa_8000),
                    at(P1, 0xb_0000),
                    Surface("256x256", 0x4_0000),
                    Recorded("256x256", 0, BASE + 0xb_8000),
                    at(P1, 0x10_0000),
                    at(P1, 0x10_8000),
                    // A shareable 64x64 BGRA8 texture, and one opened from its handle.
                    Tex(shareable(0x4000)),
                    at(P1, 0x12_0000),
                    Tex(TextureKind::FromSharedHandle),
                    at(P1, 0x12_8000),
                    // A sparse heap of 0x40000 bytes: its range, then its page.
                    Make("sparse", 0x4_0000),
                    Make("sparse state", SPARSE_STATE),
                    at(P1, 0x18_0000),
                ],
            ),
            (
                // The library's bring-up, an isolated task's, then a compute, a blit and a
                // render encoder's command buffers, each committed and completed.
                "vaprobe bringup",
                vec![
                    next(0x1_8000),
                    Use(Queue),
                    next(0x8_0000),
                    Use(ComputeEncoder),
                    next(0xf_0000),
                    // An empty blit encoder takes none, nor does a 4x4 render target.
                    Tex(RGBA8_4X4),
                    next(0xf_0000),
                    Use(RenderEncoder),
                    next(0x39_8000),
                    Use(TilerSubmission),
                    next(0x5a_0000),
                ],
            ),
            (
                // After a bring-up, a buffer; a render encoder, its end, its command buffer
                // committed behind an event wait, and completed once the event is signalled, a
                // buffer made and kept after each.
                "vaprobe rendersubmit",
                vec![
                    Use(Queue),
                    at(P1, 0x8_0000),
                    Use(RenderEncoder),
                    at(P1, 0x36_0000),
                    at(P1, 0x36_8000),
                    at(P1, 0x37_0000),
                    Use(TilerSubmission),
                    at(P1, 0x58_0000),
                ],
            ),
            (
                // A queue; a buffer and the target, kept; three command buffers that fill the
                // target, a buffer made and kept after each stage; then a render pass.
                "vaprobe cmdbuf",
                concat(vec![
                    vec![
                        Use(Queue),
                        at(P1, 0x8_0000),
                        at(P1, 0x8_8000),
                        // A command buffer, then a blit encoder: none.
                        at(P1, 0x9_0000),
                        at(P1, 0x9_8000),
                        // The fill.
                        Use(BlitCommand),
                        at(P1, 0x12_8000),
                        // Committed and completed, then released: none.
                        at(P1, 0x13_0000),
                        at(P1, 0x13_8000),
                    ],
                    // The second and third command buffers: none.
                    (0..10).map(|i| at(P1, 0x14_0000 + i * SLOT)).collect(),
                    vec![
                        Tex(RGBA8_4X4),
                        at(P1, 0x19_0000),
                        Use(RenderEncoder),
                        at(P1, 0x41_0000),
                        Use(TilerSubmission),
                        at(P1, 0x62_0000),
                        // An uncommitted empty blit encoder, released; a second queue's.
                        at(P1, 0x62_8000),
                        at(P1, 0x63_0000),
                        Use(Queue),
                        at(P1, 0x63_8000),
                    ],
                ]),
            ),
        ]
    }

    fn play(mut space: MetalSpace, case: &str, steps: &[Step]) {
        let mut made: BTreeMap<&str, Holding> = BTreeMap::new();
        let mut kept = Vec::new();
        let mut textures = Vec::new();
        // The runtime's sampler ID pool: its IDs held, the device's own among them, and its
        // size, doubled by the ID that brings the count to it (its `sampler_id_pool_size`: the
        // 15th, 31st, 63rd and 127th states after the device's own give 32, 64, 128 and 256).
        let (mut ids, mut pool) = (1, FIRST_POOL);
        for (i, step) in steps.iter().enumerate() {
            let at = format!("{case}, step {i}");
            match *step {
                Make(name, length) => {
                    let held = space.place(length).unwrap();
                    made.insert(name, Holding::Range(held));
                }
                At(length, value) => {
                    let held = space.place(length).unwrap();
                    assert_eq!(held.address(), value, "{at}");
                    kept.push(Holding::Range(held));
                }
                Next(value) => {
                    let held = space.place(PAGE).unwrap();
                    assert_eq!(held.address(), value, "{at}");
                    space.release(held);
                }
                Pin(name, address, length) => {
                    let held = space
                        .pin(address, length)
                        .unwrap_or_else(|e| panic!("{at}: {e:?}"));
                    made.insert(name, Holding::Range(held));
                }
                PinUnmade(address, length) => {
                    let held = space
                        .pin(address, length)
                        .unwrap_or_else(|e| panic!("{at}: {e:?}"));
                    space.release(held);
                }
                Refused(address, length) => {
                    let refused = space.pin(address, length);
                    assert!(refused.is_err(), "{at}: {refused:?}");
                }
                Taken(span) => {
                    space.taken(span);
                }
                Use(first) => space.first_use(first),
                Small(length, class) => {
                    kept.push(Holding::Slot(small(&mut space, &at, length, class)));
                }
                SmallAt(name, length, class, value) => {
                    let slot = small(&mut space, &at, length, class);
                    assert_eq!(slot.address(), value, "{at}: {name}");
                    made.insert(name, Holding::Slot(slot));
                }
                Release(name) => space.free(made.remove(name).unwrap()),
                Tex(kind) => textures.push(space.place_texture(kind)),
                TexAt(kind, value) => {
                    let texture = space.place_texture(kind);
                    let address = match &texture {
                        Holding::Range(held) => held.address(),
                        Holding::Slot(slot) => slot.address(),
                        Holding::Nothing => panic!("{at}: no space"),
                    };
                    assert_eq!(address, value, "{at}");
                    textures.push(texture);
                }
                Sampler => {
                    ids += 1;
                    if ids >= pool {
                        pool *= 2;
                    }
                    space.sampler_pool(pool);
                }
                PlacedSampler => space.sampler_pool(pool),
                SamplerArray(value) => {
                    let address = match &space.samplers.held {
                        Holding::Range(held) => held.address(),
                        Holding::Slot(slot) => slot.address(),
                        Holding::Nothing => panic!("{at}: no array"),
                    };
                    assert_eq!(address, value, "{at}");
                }
                ReleaseTextures => {
                    for texture in textures.drain(..) {
                        space.free(texture);
                        space.texture_gone();
                    }
                }
                Surface(name, alloc_size) => {
                    space.first_use(Texture);
                    space.texture_made();
                    let held = space.place_surface(alloc_size).unwrap();
                    made.insert(name, Holding::Range(held));
                }
                Recorded(name, offset, value) => {
                    let address = match made.get(name) {
                        Some(Holding::Range(held)) => held.address(),
                        Some(Holding::Slot(slot)) => slot.address(),
                        _ => panic!("{at}: {name}"),
                    };
                    assert_eq!(address + offset, value, "{at}: {name}");
                }
            }
        }
    }

    /// A buffer of its own memory of `length` bytes, private or not, which Metal suballocates:
    /// its slot, after its pool's first use.
    fn small(space: &mut MetalSpace, at: &str, length: u64, class: PoolClass) -> Slot {
        match buffer_space(length, Bytes::Own { class }, false) {
            BufferSpace::Pool(class) => small_first_use(class).map(|first| space.first_use(first)),
            BufferSpace::Range => panic!("{at}: a range"),
        };
        space
            .slot(class, length)
            .unwrap_or_else(|| panic!("{at}: no slot"))
    }

    #[test]
    fn every_recorded_address_is_placed() {
        let cases = cases();
        assert_eq!(cases.len(), 39);
        for (case, steps) in &cases {
            play(MetalSpace::brought_up(), case, steps);
        }
    }

    /// A depth or stencil texture without GPU-optimized contents takes the runtime's model of
    /// Metal's heap size where the model has one, whatever the runtime's own answer, of every
    /// format, sample count, storage and slice count; where the model has none, the runtime's
    /// answer; where neither answers, none
    #[test]
    fn depth_and_stencil_textures_take_metals_heap_size() {
        use metal2vulkan::metal::{PixelFormat, TextureType};
        /// What the runtime answers here for a descriptor: not Metal's.
        const RUNTIMES: u64 = 0x7_0000;
        let answer = |_: &rt::TextureDescriptor| {
            Some(SizeAndAlign {
                size: RUNTIMES,
                align: 0x80,
            })
        };
        let none = |_: &rt::TextureDescriptor| None;
        let modelled = SizeAndAlign {
            size: 0x4080,
            align: 0x80,
        };
        let desc = |pf, samples: u64| {
            let mut desc = rt::TextureDescriptor::texture_2d(pf, 64, 64, false);
            if samples > 1 {
                desc.texture_type = TextureType::Type2DMultisample;
                desc.sample_count = samples;
            }
            desc.resource_options = rt::ResourceOptions::private();
            desc.allow_gpu_optimized_contents = false;
            desc
        };
        let mut slices = desc(PixelFormat::Depth32FloatStencil8, 1);
        slices.texture_type = TextureType::Type2DArray;
        slices.array_length = 2;
        let mut shared = desc(PixelFormat::Depth32Float, 1);
        shared.resource_options = rt::ResourceOptions::shared();
        let mut descs = vec![slices, shared];
        for pf in [
            PixelFormat::Depth16Unorm,
            PixelFormat::Depth32Float,
            PixelFormat::Stencil8,
            PixelFormat::Depth32FloatStencil8,
            PixelFormat::X32Stencil8,
        ] {
            descs.extend([desc(pf, 1), desc(pf, 4)]);
        }
        for desc in &descs {
            let size = own_heap_size(desc, Some(modelled), answer);
            assert_eq!(size, Some(0x4080), "the model's: {desc:?}");
            let size = own_heap_size(desc, Some(modelled), none);
            assert_eq!(size, Some(0x4080), "the model's, no answer: {desc:?}");
            let size = own_heap_size(desc, None, answer);
            assert_eq!(size, Some(RUNTIMES), "the runtime's: {desc:?}");
            assert_eq!(own_heap_size(desc, None, none), None, "none: {desc:?}");
        }
    }

    #[test]
    fn every_probe_row_is_placed() {
        for (row, steps) in &probe_rows() {
            play(MetalSpace::new(), row, steps);
        }
    }

    /// The first uses' spans as the probes measured them: a first use that comes after another
    /// takes what that one did not.
    #[test]
    fn each_first_use_takes_what_no_earlier_one_took() {
        let span = |earlier: &[FirstUse], first| {
            let mut space = MetalSpace::brought_up();
            earlier.iter().for_each(|&use_| space.first_use(use_));
            let before = space.lowest_gap(SLOT).unwrap();
            space.first_use(first);
            space.lowest_gap(SLOT).unwrap() - before
        };
        let rows = [
            (&[][..], Queue, 0),
            (&[], Texture, 0),
            (&[], ComputeEncoder, 0x7_0000),
            (&[], BlitCommand, 0x8_8000),
            (&[], RenderEncoder, 0x2d_8000),
            (&[ComputeEncoder], BlitCommand, 0x4_8000),
            (&[BlitCommand], ComputeEncoder, 0x3_0000),
            (&[ComputeEncoder], RenderEncoder, 0x2a_8000),
            (&[BlitCommand], RenderEncoder, 0x27_8000),
            (&[RenderEncoder], BlitCommand, 0x2_8000),
            (&[], TilerBlit, 0x7_8000),
            (&[TilerBlit], RenderEncoder, 0x26_0000),
            (&[], TilerSubmission, 0x20_8000),
        ];
        for (earlier, first, taken) in rows {
            assert_eq!(span(earlier, first), taken, "{earlier:?} then {first:?}");
        }
        // The device's heap, then the pools: the first plain range lies at [`START`].
        let mut bare = MetalSpace::new();
        bare.first_use(Texture);
        assert_eq!(bare.place(PAGE).unwrap().address(), START);
    }

    #[test]
    fn a_one_page_buffer_leaves_a_free_page_after_it() {
        let mut space = MetalSpace::brought_up();
        let a = space.place(1).unwrap();
        let b = space.place(PAGE).unwrap();
        assert_eq!((a.address(), b.address()), (START, START + SLOT));
        assert_eq!(
            (a.pages(), a.guard()),
            (START..START + PAGE, START + PAGE..START + SLOT)
        );
    }

    /// One guard page, rounded up to a slot: three and five pages take what two and four do.
    #[test]
    fn a_range_takes_one_guard_page_rounded_up_to_a_slot() {
        let mut space = MetalSpace::brought_up();
        let spans: Vec<u64> = (1..=5)
            .map(|pages| {
                let held = space.place(pages * PAGE).unwrap();
                held.guard().end - held.address()
            })
            .collect();
        assert_eq!(spans, [0x8000, 0x10000, 0x10000, 0x18000, 0x18000]);
    }

    /// The gaps answer the lowest start where a range fits as the walk of the held runs does
    /// ([`MetalSpace::lowest_gap`]), over random sequences of placements, pins (below the base,
    /// over it, among the ranges and past the space's end) and releases, at spans of whole slots,
    /// of parts of one, of none and past the space; and the space places the same ranges
    #[test]
    fn the_gaps_answer_the_lowest_start_as_the_walk_does() {
        let mut seed = 0x2545_f491_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for round in 0..8 {
            let mut space = MetalSpace::new();
            let mut held: Vec<Held> = Vec::new();
            for step in 0..3000 {
                let roll = next() % 16;
                if roll < 8 {
                    // A placement, mostly of a few pages, now and then of many.
                    let pages = if next() % 8 == 0 {
                        next() % 512
                    } else {
                        next() % 6
                    };
                    let length = pages * PAGE + next() % PAGE;
                    // A placement holds the device's range first.
                    space.device();
                    let expected =
                        MetalSpace::plain_span(length).and_then(|(_, span)| space.walked_gap(span));
                    let placed = space.place(length);
                    let lowest = placed.as_ref().map(Held::address);
                    assert_eq!(lowest, expected, "round {round} step {step}");
                    held.extend(placed);
                } else if roll < 10 {
                    // A pin, aligned or not, below the base, among the ranges or far past them.
                    let around = match next() % 4 {
                        0 => BASE - (next() % 64) * SLOT,
                        1 => START + (next() % 4096) * SLOT,
                        2 => END - (next() % 8) * SLOT,
                        _ => START + (next() % 4096) * PAGE,
                    };
                    let length = (next() % 8) * PAGE + next() % 0x100;
                    if let Ok(pinned) = space.pin(around + next() % 2 * 0x10, length) {
                        held.push(pinned);
                    }
                } else if !held.is_empty() {
                    let at = usize::try_from(next()).unwrap() % held.len();
                    space.release(held.swap_remove(at));
                }
                for span in [
                    0,
                    1,
                    SLOT - 1,
                    SLOT,
                    SLOT + 1,
                    2 * SLOT,
                    3 * SLOT,
                    (next() % 64) * SLOT,
                    (next() % 1024) * SLOT,
                    next() % (64 * SLOT),
                    END - BASE,
                    END,
                    u64::MAX - next() % SLOT,
                    u64::MAX,
                ] {
                    assert_eq!(
                        space.lowest_gap(span),
                        space.walked_gap(span),
                        "round {round} step {step} span {span:#x}"
                    );
                }
            }
        }
    }

    /// The lowest freed range goes first, not the one freed last.
    #[test]
    fn the_lowest_freed_range_goes_first() {
        let mut space = MetalSpace::brought_up();
        let a = space.place(PAGE).unwrap();
        let b = space.place(PAGE).unwrap();
        let _c = space.place(PAGE).unwrap();
        space.release(a);
        space.release(b);
        assert_eq!(space.place(PAGE).unwrap().address(), START);
    }

    #[test]
    fn a_freed_range_too_small_is_passed_over() {
        let mut space = MetalSpace::brought_up();
        let a = space.place(PAGE).unwrap();
        let _b = space.place(PAGE).unwrap();
        space.release(a);
        assert_eq!(space.place(2 * PAGE).unwrap().address(), START + 2 * SLOT);
        assert_eq!(space.place(PAGE).unwrap().address(), START);
    }

    #[test]
    fn plain_ranges_go_around_pins() {
        let mut space = MetalSpace::brought_up();
        let pin = space.pin(START + SLOT, PAGE).unwrap();
        assert_eq!(
            (pin.pages(), pin.guard()),
            (
                START + SLOT..START + SLOT + PAGE,
                START + SLOT + PAGE..START + 2 * SLOT
            )
        );
        assert_eq!(space.place(PAGE).unwrap().address(), START);
        assert_eq!(space.place(PAGE).unwrap().address(), START + 2 * SLOT);
    }

    /// A pin holds a plain range's span from its first page: 0x10 bytes across a page take two
    /// pages and a guard page, rounded up to a slot; three pages take four (`vaprobe pins
    /// pin3-10000`).
    #[test]
    fn a_pin_holds_the_span_of_its_pages() {
        let mut space = MetalSpace::brought_up();
        let at = PINNED + 0x3ff8;
        let held = space.pin(at, 0x10).unwrap();
        assert_eq!(held.address(), at);
        assert_eq!(held.pages(), PINNED..PINNED + 2 * PAGE);
        assert_eq!(held.guard(), PINNED + 2 * PAGE..PINNED + 4 * PAGE);
        let three = space.pin(PINNED + 2 * SLOT, 3 * PAGE).unwrap();
        assert_eq!(
            three.guard(),
            PINNED + 2 * SLOT + 3 * PAGE..PINNED + 4 * SLOT
        );
        assert!(space.pin(PINNED + 4 * SLOT, PAGE).is_ok());
    }

    /// A pin whose first page is a granule's second is refused, far from every range, before
    /// what it meets is looked at; one 0x10 bytes into a granule's first page is made there
    /// (`vaprobe pins far-4000`, `far-4010`, `far-10`).
    #[test]
    fn a_pin_off_a_granule_is_refused() {
        let mut space = MetalSpace::brought_up();
        // One page, and a page from 0x10 into it, which touches two.
        for (at, span) in [(PINNED + PAGE, SLOT), (PINNED + PAGE + 0x10, 2 * SLOT)] {
            assert_eq!(
                space.pin(at, PAGE),
                Err(PinRefused::OffGranule {
                    requested: PINNED + PAGE..PINNED + PAGE + span,
                }),
                "{at:#x}"
            );
        }
        assert_eq!(
            space.pin(PINNED + 0x10, PAGE).unwrap().address(),
            PINNED + 0x10
        );
        // Off its granule and into a held range: off its granule.
        assert!(matches!(
            space.pin(START + PAGE, PAGE),
            Err(PinRefused::OffGranule { .. })
        ));
    }

    #[test]
    fn a_pin_meeting_a_held_range_is_refused() {
        let mut space = MetalSpace::brought_up();
        // A range with a free slot below it.
        let below = space.place(PAGE).unwrap();
        let a = space.place(PAGE).unwrap().address();
        space.release(below);
        // At its first page, and two pages below it, whose span reaches its first page.
        for (at, length) in [(a, PAGE), (a - SLOT, 2 * PAGE)] {
            let refused = space.pin(at, length);
            assert!(
                matches!(refused, Err(PinRefused::Held { held, .. }) if held == a),
                "{at:#x}: {refused:?}"
            );
        }
        // A page below it, whose span ends where it starts, and a pin past its span.
        assert!(space.pin(a - SLOT, PAGE).is_ok());
        assert!(space.pin(a + SLOT, PAGE).is_ok());
    }

    /// A released pin frees its whole span, and one never made holds nothing.
    #[test]
    fn a_released_pin_frees_its_whole_span() {
        let mut space = MetalSpace::brought_up();
        let pin = space.pin(PINNED, PAGE).unwrap();
        space.release(pin);
        let again = space.pin(PINNED, 3 * PAGE).unwrap();
        space.release(again);
        assert!(space.pin(PINNED + SLOT, PAGE).is_ok());
    }

    #[test]
    fn a_pin_past_the_end_of_the_64_bit_space_is_refused() {
        let mut space = MetalSpace::brought_up();
        assert_eq!(space.pin(u64::MAX - 4, 0x10), Err(PinRefused::Outside));
    }

    #[test]
    fn a_range_past_the_end_of_the_space_is_not_placed() {
        let mut space = MetalSpace::brought_up();
        assert!(space.place(END).is_none());
        assert!(space.place(u64::MAX).is_none());
        assert_eq!(space.place(PAGE).unwrap().address(), START);
    }

    /// The address a placing asks the runtime for.
    fn at(placement: rt::AddressPlacement) -> u64 {
        match placement {
            rt::AddressPlacement::At(rt::GpuAddress(at)) => at,
            other => panic!("{other:?}"),
        }
    }

    /// A buffer's range stays held while the runtime holds its token, and is free once the
    /// token goes; the object holds nothing of it.
    #[test]
    fn a_made_buffer_holds_its_range_until_its_token_goes() {
        let (placement, mut placing) = placing(PAGE, SHARED, None);
        let address = at(placement);
        assert!(!placing.nil_at_conflict());
        // Past the device's range: the process's space has had no queue in a unit test.
        assert_eq!((address % SLOT, address > BASE), (0, true));
        let token = placing.token().unwrap();
        assert_eq!(placing.made().address(), None);
        assert!(process().pin(address, 1).is_err());
        drop(token);
        let pin = process().pin(address, PAGE).unwrap();
        process().release(pin);
    }

    /// `vaprobe surfaces`: a sparse heap of 0x40000 bytes took 0x48000 of the space and then
    /// 0x8000 for its second buffer; one not made gives both back.
    #[test]
    fn a_sparse_heap_takes_its_range_and_a_page() {
        let mut space = MetalSpace::brought_up();
        let heap = space.place(0x40000).unwrap();
        let state = space.place(SPARSE_STATE).unwrap();
        assert_eq!(
            (
                heap.address(),
                state.address(),
                space.place(PAGE).unwrap().address()
            ),
            (START, START + 0x48000, START + 0x50000)
        );
        // Its token holds both, and gives them back as it goes.
        let (placement, mut placing) = placing_sparse(0x40000, None);
        let at = super::tests::at(placement);
        let token = placing.token().unwrap();
        assert_eq!(placing.made().address(), None);
        assert!(process().pin(at + 0x48000, PAGE).is_err());
        drop(token);
        let pin = process().pin(at, 0x40000 + 0x8000).unwrap();
        process().release(pin);
    }

    #[test]
    fn a_buffer_not_made_gives_its_range_back() {
        let (placement, placing) = placing(PAGE, SHARED, None);
        drop(placing);
        let pin = process().pin(at(placement), PAGE).unwrap();
        process().release(pin);
    }

    #[test]
    fn a_no_copy_buffer_keeps_its_pointers_offset_into_its_page() {
        let (placement, placing) = placing(PAGE, Bytes::NoCopy(3 * PAGE + 0x10), None);
        assert_eq!(at(placement) % SLOT, 0x10);
        drop(placing);
    }

    /// `vaprobe small`: shared and private buffers under a page lie in a pool each, of either
    /// creator; a page or more, no bytes, a pin, a no-copy buffer and a heap take a range.
    #[test]
    fn buffers_under_a_page_lie_in_metals_pools() {
        use BufferSpace::{Pool, Range};
        use PoolClass::{Managed, Private, Shared};
        let private = Bytes::Own { class: Private };
        let managed = Bytes::Own { class: Managed };
        let rows = [
            (1, SHARED, false, Pool(Shared)),
            (0x3fff, SHARED, false, Pool(Shared)),
            (0x4000, SHARED, false, Range),
            (1, private, false, Pool(Private)),
            (0x3fff, private, false, Pool(Private)),
            (0x4000, private, false, Range),
            (0x3fff, managed, false, Pool(Managed)),
            (0x4000, managed, false, Range),
            (0, SHARED, false, Range),
            (0x100, SHARED, true, Range),
            (0x100, Bytes::NoCopy(0), false, Range),
            (0x100, Bytes::Heap, false, Range),
        ];
        for (length, bytes, pinned, space) in rows {
            assert_eq!(
                buffer_space(length, bytes, pinned),
                space,
                "{length:#x} {bytes:?} {pinned}"
            );
        }
    }

    /// A buffer Metal suballocates is made at its slot in its pool, a conflict there not nil,
    /// holding the slot until its range token or its hold goes: the next buffer of the slot's
    /// bytes takes the step after it meanwhile, and the slot itself after.
    #[test]
    fn a_suballocated_buffer_holds_a_slot_in_its_pool() {
        let slot_of = |token: &rt::RangeToken| token.downcast_ref::<Hold>().unwrap().address();
        let (placement, mut first) = placing(0x100, SHARED, None);
        assert!(!first.nil_at_conflict());
        let token = first.token().unwrap();
        let at = slot_of(&token).unwrap();
        assert_eq!(placement, rt::AddressPlacement::At(rt::GpuAddress(at)));
        assert_eq!(at % SLOT_STEP, 0);
        let (_, second) = placing(0x100, SHARED, None);
        let second = second.made();
        assert_eq!(second.address(), Some(at + SLOT_STEP));
        drop(token);
        let (_, mut again) = placing(0x80, SHARED, None);
        assert_eq!(slot_of(&again.token().unwrap()), Some(at));
        drop(second);
        let (_, made) = placing(0x100, SHARED, None);
        assert_eq!(made.made().address(), Some(at));
    }

    #[test]
    fn a_guest_pin_meeting_a_held_range_is_made_where_the_runtime_refuses_it_as_nil() {
        let (placement, placing) = placing(PAGE, SHARED, None);
        let hold = placing.made();
        let address = at(placement);
        // In its range, and off a granule.
        for pin in [address, address + SLOT + PAGE] {
            let (refused, nothing) = super::placing(PAGE, SHARED, Some(pin));
            assert_eq!((at(refused), nothing.nil_at_conflict()), (REFUSED.0, true));
            assert_eq!(nothing.made().address(), None);
        }
        let (pinned, pin) = super::placing(PAGE, SHARED, Some(address + SLOT));
        assert_eq!((at(pinned), pin.nil_at_conflict()), (address + SLOT, true));
        drop(pin);
        drop(hold);
    }

    #[test]
    fn a_range_with_no_room_is_made_where_the_runtime_refuses_it_as_nil() {
        let (refused, nothing) = placing(u64::MAX, SHARED, None);
        assert_eq!((at(refused), nothing.nil_at_conflict()), (REFUSED.0, true));
        let (refused, nothing) = placing(END, SHARED, None);
        assert_eq!((at(refused), nothing.nil_at_conflict()), (REFUSED.0, true));
    }

    /// `vaprobe textures`, `vaprobe texsizes`, `vaprobe shared` and `vaprobe surfaces`, a row
    /// each: what a texture of each kind took of the space, a plain range's span where it took
    /// one. A texture of memory of its own by its heap size, a shareable one by its allocation.
    #[test]
    fn every_texture_row_is_what_the_kind_takes() {
        let span = |space| match space {
            TextureSpace::Own(bytes) => {
                let (_, span) = MetalSpace::plain_span(bytes).unwrap();
                Some(span)
            }
            _ => None,
        };
        let rows = [
            // 4x4 RGBA8, private, shared and a render target, suballocated.
            (RGBA8_4X4, None),
            // `vaprobe texsizes`: 4x4 Depth32Float, Depth16Unorm and Stencil8, and 32x64 RGBA8.
            (own(0x80), None),
            (own(0x2000), None),
            // 63x63 RGBA8; 4x4 Depth32Float_Stencil8 and 64x64 RGBA8 read or a render target;
            // 64x64 RGBA8 with mipmaps; 128x128 and 256x256 RGBA8.
            (own(0x4000), Some(0x8000)),
            (own(0x4080), Some(0x10000)),
            (own(0x5680), Some(0x10000)),
            (own(0x1_0200), Some(0x18000)),
            (own(0x4_0800), Some(0x48000)),
            // `vaprobe textures`: 256x256 RGBA8.
            (own(0x4_0000), Some(0x48000)),
            (TextureKind::Memoryless, None),
            (TextureKind::View, None),
            (TextureKind::OverBuffer, None),
            (TextureKind::InHeap, None),
            (TextureKind::FromSharedHandle, None),
            // Shareable BGRA8 (`vaprobe shared`): 4x4 (`mutgap2_res_shared_backing_placed` too),
            // 16x16, 32x32, 63x64 and 64x64, 65x64, 128x128 and 256x256.
            (shareable(0x40), Some(0x8000)),
            (shareable(0x400), Some(0x8000)),
            (shareable(0x1000), Some(0x8000)),
            (shareable(0x4000), Some(0x10000)),
            (shareable(0x8000), Some(0x10000)),
            (shareable(0x1_0000), Some(0x18000)),
            (shareable(0x4_0000), Some(0x48000)),
        ];
        for (kind, taken) in rows {
            assert_eq!(span(texture_space(kind)), taken, "{kind:?}");
        }
        assert_eq!(
            texture_space(TextureKind::OverSurface),
            TextureSpace::Surface
        );
    }

    /// Apple's layout as the runtime models it: 4x4 RGBA8 is 0x40 bytes (Apple's
    /// `allocatedSize` 0x80), 64x64 0x4000, a 4x4 cube of six faces 0x180, and a 5x3 texture's
    /// axes round up to 8 by 4.
    #[test]
    fn texture_allocations_round_each_axis_to_a_power_of_two() {
        assert_eq!(
            texture_allocation([1, 1], 4, [4, 4, 1], false, 1, 1, 1),
            0x40
        );
        assert_eq!(
            texture_allocation([1, 1], 4, [64, 64, 1], false, 1, 1, 1),
            0x4000
        );
        assert_eq!(
            texture_allocation([1, 1], 4, [4, 4, 1], false, 1, 6, 1),
            0x180
        );
        assert_eq!(
            texture_allocation([1, 1], 4, [5, 3, 1], false, 1, 1, 1),
            8 * 4 * 4
        );
        // Levels 64 by 64 down to 1 by 1.
        let levels: u64 = (0..7).map(|l| (64u64 >> l).pow(2) * 4).sum();
        assert_eq!(
            texture_allocation([1, 1], 4, [64, 64, 1], false, 7, 1, 1),
            levels
        );
        // A 4x4 block format of 16 bytes a block, 16x16 texels: 4x4 blocks.
        assert_eq!(
            texture_allocation([4, 4], 16, [16, 16, 1], false, 1, 1, 1),
            0x100
        );
        // Past every size: saturates.
        assert_eq!(
            texture_allocation([1, 1], 16, [u64::MAX, u64::MAX, 1], false, 1, 1, 1),
            u64::MAX
        );
    }

    /// Every texture and buffer over one surface shares its range, which goes back with the last;
    /// the first buffer over it is made at its address, and each made while one holds it shares
    /// that one's range, until the last goes.
    #[test]
    fn a_surface_has_one_range_while_anything_over_it_lives() {
        const KEY: u64 = 0xdead_0000;
        let texture = surface(KEY, PAGE);
        let at = texture.address().unwrap();
        let (placement, buffer) = surface_buffer(KEY, PAGE);
        let pinned = SurfacePlace::At(rt::AddressPlacement::At(rt::GpuAddress(at)));
        assert_eq!(placement, pinned);
        assert_eq!(buffer.address(), Some(at));
        let (shared, second) = surface_buffer(KEY, PAGE);
        assert_eq!(shared, SurfacePlace::Shared);
        assert_eq!(second.address(), Some(at));
        let other = surface(KEY, PAGE);
        assert_eq!(other.address(), Some(at));
        drop(texture);
        drop(buffer);
        // The second buffer holds the address still: a third shares it.
        let (third, held) = surface_buffer(KEY, PAGE);
        assert_eq!(third, SurfacePlace::Shared);
        drop((second, held));
        // The other texture keeps the range; no buffer holds its address.
        let (again, held) = surface_buffer(KEY, PAGE);
        assert_eq!(again, pinned);
        drop((other, held));
        // Nothing over the surface: its range is free.
        let pin = process().pin(at, PAGE).unwrap();
        process().release(pin);
        // A new surface range for the key, wherever the space places it.
        assert!(surface(KEY, PAGE).address().is_some());
    }
}
