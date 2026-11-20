//! The rt platform's buffers ([`RtBuffer`]): rt's buffers, each a range of a block, made by the
//! device's `newBufferWithLength:options:` ([`of_length`]), `newBufferWithDescriptor:`
//! ([`of_desc`]) and `newBufferWithBytesNoCopy:length:options:deallocator:`
//! ([`over_host_memory`]), and by heaps (`super::heap`), and the textures over their bytes
//! (`super::texture::make_over`), over a plane of an IOSurface the buffer stands for among them
//! (`newTextureWithDescriptor:iosurface:plane:`, at the plane's offset and rows).
//!
//! **Blocks of its own.** A buffer the device makes takes blocks of its own (rt's `take_block`):
//! a Shared or Private one of its storage mode, and for a Managed buffer a Shared one of its
//! bytes and, on a GPU with memory of its own (rt's `has_unified_memory` false), a Private one of
//! its device copy. A take is 64 KiB at least, whatever the buffer's length: placing small buffers
//! in shared blocks is the guest's, by address ("A pinned address"). Its bytes are zeros, as
//! Metal's are: rt zeroes a Shared buffer's through the mapping and a Private one's by a GPU fill
//! ahead of the next command buffer, the platform zeroes a Managed buffer's own bytes (counted by
//! the byte, [`ZEROED_BYTES_CODE`], with a heap's it zeroes, `super::heap`), and rt its device
//! copy: rt says nothing of a block's bytes as it is taken. The buffer holds its blocks; the
//! platform's handles of them go as it is made.
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
//! 24 (read-only) is taken and answered without, as Metal does ("Read-only memory").
//!
//! **Read-only memory.** Metal's GPU drops a kernel's and a blit's writes to a no-copy buffer or
//! heap of option bit 24, and to the pages a guest process maps read-only, and the CPU and later
//! command buffers read the pages' bytes (measured on the M3 Max, the 5300M and the UHD 630,
//! `crate::gpu::command`'s docs); Vulkan has no memory that drops writes, and rt no read-only
//! import. So such pages are imported as a copy whose GPU writes are dropped at each completion
//! (`super::copied`, "Writes dropped"), never by their pointer or a dma-buf, which the GPU would
//! write: a pages block's view its process maps read-only (`import_known`'s `read_only`), whose
//! address the guest's buffers over the block are pinned in, and a buffer or heap of option bit 24
//! with no pin, over a copy of its own. Each such import is counted ([`IMPORTS_READ_ONLY`]), and
//! as a copy ([`IMPORTS_COPIED`]). A render pass's store into a texture over such a copy completes
//! and is dropped with the copy's other writes, as the 5300M and the UHD 630 answer; the M3 Max
//! faults it: counted at the pass ([`super::copied::STORE_OUTS_CODE`]). A buffer or heap of option
//! bit 24 pinned in a known block that is no such copy stays there, at the guest's address, where
//! the GPU's writes reach the bytes: counted ([`READ_ONLY_CODE`]), the process's first with a
//! line.
//!
//! **Dma-bufs of the pages** (Linux). A GPU may take no pointer to some pages: RADV refuses the
//! pages of a shared mapping of a memfd at the import, as the guest's RAM and the views of its
//! pages are (rt's `import_host_memory`). There the import is of a dma-buf of the files behind the
//! pages instead (`crate::os::dma_buf_of_host`, rt's `import_dma_buf`), one dma-buf a block, so
//! the bytes are the same pages the CPU's mapping reaches. One holds as many runs of file pages as
//! udmabuf's `list_limit` takes, and as many bytes as its `size_limit_mb`: at the stock limit of
//! 1,024 runs about 4 MiB of a guest buffer, whose pages are nearly each a run of their own
//! (`crate::os::linux::device_lend`'s measured note). Pages past either go as several dma-bufs,
//! cut where one's limits end at multiples of rt's granule and the last padded with zeroed memory
//! of its own (`crate::os`'s `dma_bufs_of_host`), which rt binds behind one buffer, its addresses
//! running across them as over one range (rt's `import_dma_bufs`), counted
//! ([`IMPORTS_DMA_BUFS`]). Where rt binds none so on the device (no granule: Mesa's Intel driver), the host makes none, or
//! rt refuses them, counted ([`IMPORTS_DMA_BUFS_REFUSED`]), the pages are imported as a copy ("A
//! copy at the refusal point"), counted as too large with their runs and bytes
//! ([`IMPORTS_TOO_LARGE`], `super::diag::import_path`); so are pages with no file behind them. rt
//! answers no `contents` for a dma-buf's block: until it takes the mapping, the device keeps the
//! host address of the pages the dma-buf holds for it ([`KnownBlocks`]), which a buffer in the
//! block answers as its `contents` and a pinned buffer over those bytes is a range of, as over a
//! block rt maps. A Managed buffer over such a block is a range of it too, counted
//! ([`MANAGED_OVER_DMA_BUF`]): where the GPU keeps a device copy, rt fills it by a GPU copy from
//! the block at the commit that takes the ranges named since, as over any Shared block. Each
//! import is counted by the way it was made or refused ([`IMPORTS_POINTER`], [`IMPORTS_DMA_BUF`],
//! [`IMPORTS_COPIED`], [`IMPORTS_REFUSED`]).
//!
//! **Pages no pin takes** (Linux). A view of a guest's pages of more than one run may be mapped
//! through udmabuf's dma-bufs (`crate::os::linux`'s `vm` docs, "Views through dma-bufs"): a
//! `VM_PFNMAP` mapping, which no kernel pin takes. Its pointer never goes to rt: NVIDIA's driver
//! refused rt's import of such a pointer (`VK_ERROR_OUT_OF_DEVICE_MEMORY`, RTX 3090, driver
//! 620.70.06, 2026-11-19), and a boot whose views were so mapped then had 32 of rt's Vulkan calls
//! refused, a pipeline's among them, and a black desktop, where one with no such views had none
//! (the Vulkan platform's runtime records the same of the 615 driver, its
//! `UnpinnablePagesBreakHostMemory`). Pages a byte of which lies in such a mapping
//! (`crate::os`'s `through_dma_bufs`) are imported as a dma-buf of the files behind them first,
//! or as several past one's limits, as above; where the host makes none or rt refuses them, they
//! are mapped from their files (`crate::os`'s `map_for_pointer`) and imported by their pointer,
//! counted ([`IMPORTS_FROM_FILES`]), and each as why no dma-buf was imported too: too large for
//! one, with their runs ([`IMPORTS_FROM_FILES_TOO_LARGE`], `super::diag`'s import paths), refused
//! by rt ([`IMPORTS_FROM_FILES_DMA_BUF_REFUSED`]), or none made by the host
//! ([`IMPORTS_FROM_FILES_NO_DMA_BUF`]), the process's first of each with a line saying why, rt's
//! refusal named; where they cannot all be mapped so (near the views' budget), or rt refuses that
//! pointer, they are imported as a copy, with no pointer of theirs asked of rt while a page of
//! them lies in a dma-buf's mapping. So that no view is mapped twice, through dma-bufs and then
//! from its files, views that would be are mapped from their files at once, as the Vulkan
//! platform's are where its device's probe finds them imported so, by rt's own answers: where the
//! device binds no several dma-bufs behind one buffer (rt's `dma_buf_granule`, `None`, read as the
//! device is made: `crate::os`'s `learn_dma_buf_granule`), each view more than one dma-buf holds,
//! from the first; and from rt's first refusal of a view's one dma-buf on, which rt answers only
//! at the call, every view (`crate::os`'s `views_need_files`). Such a view's import then takes
//! its pointer with nothing mapped again. On NVIDIA's driver rt refuses the first view's one
//! dma-buf so, by its rule for dma-bufs (it imports one only into a host-visible, host-coherent
//! type): measured on an RTX 3090 (driver 620.70.06, 2026-11-20), the query of a udmabuf's fd
//! offers types 0x11, none of them both, so a boot there counts one
//! [`IMPORTS_FROM_FILES_DMA_BUF_REFUSED`], at the process's first such view, and maps every later
//! view from its files, which the guest does not see. Pages in no dma-buf's mapping go by their
//! pointer first.
//! From before an import asks where its pages lie until rt frees an import of their pointer, the
//! platform holds their mappings as they are (`crate::os`'s `MappingsHeld`), so no fold of the
//! views maps them through dma-bufs under it (`vm`'s docs, "Folding back").
//!
//! **A copy at the refusal point.** Pages that rt imports neither by their pointer nor as
//! dma-bufs of their files (on Linux past one dma-buf's limits where no several are imported, with
//! no file behind them, near the views' budget, or refused by the driver; elsewhere wherever rt
//! refuses their pointer) are
//! imported as a copy (`super::copied`), as rt refused nothing: memory of the platform's own that
//! holds their bytes, which rt imports by its pointer. A buffer in the copy's block answers the
//! pages as its `contents`, and a pinned buffer over the pages is a range of the block, as over a
//! dma-buf's ([`KnownBlocks`]). The copy is synced byte by byte: at each commit the bytes the CPU
//! changed in the pages since they and the copy last agreed are written into the copy, and at each
//! completion, before the command buffer's completed handlers run or a waiter returns, the bytes
//! the GPU changed in the copy are written into the pages where the CPU did not change them since,
//! a byte both changed keeping the CPU's. Counted ([`IMPORTS_COPIED`]), the process's first with a
//! line; a copy rt refuses too is rt's refusal of the pages, counted, and nil. Pages copied as rt
//! refused both a dma-buf of theirs and their pointer, in either order, are counted as such too
//! ([`IMPORTS_DMA_BUF_AND_POINTER_REFUSED`]), the process's first with a line naming both
//! refusals. Where it differs
//! from Metal's GPU, which reads and writes the pages in place: the GPU sees a CPU write made after
//! a commit only at the next commit, of any queue; a command buffer rt holds at its commit for a
//! shared event's value reads the bytes as they were at its commit; an event value the GPU signals
//! may be seen before its command buffer's writes are merged back; and rt's own reads and writes
//! of the block's bytes (a texture's over the buffer, on the CPU) are of the copy.
//!
//! **Answers.** A buffer's resource options are the asked ones without bits 20 and 24, Metal's
//! default hazard tracking resolved to tracked, as the Vulkan platform's runtime answers them. A
//! Private buffer's `contents` is NULL. `didModifyRange:` of a Managed buffer is rt's, which copies
//! the range into the device copy at the next commit; of any other it does nothing, as Metal's.
//!
//! **Refusals.** What Metal refuses is nil, counted as Metal's refusal (`rt-refused-metal`): a
//! length of 0 or past rt's longest buffer, an option value Metal has none of, a Memoryless
//! buffer, option bit 24 on a buffer the device makes, a Private no-copy buffer, a deallocator of a
//! buffer that is not no-copy, a texture over a plane past 0 of a surface without planes. What rt
//! refuses is nil, counted under rt's kind. A Private buffer of bytes, which a copy on the GPU
//! fills, is unbuilt ([`Unbuilt::Buffer`]), and so is a texture over a surface's plane of a format
//! that reads beside the plane ([`Unbuilt::IosurfaceTexture`]).
//!
//! **Tiled textures** (`newTiledTextureWithDescriptor:offset:bytesPerRow:`): of a 2D texture of
//! more than one level, a copy of the bytes, taken at its first use (`super::tiled`); of any other,
//! the texture over the bytes `newTextureWithDescriptor:offset:bytesPerRow:` makes.
//!
//! **Deferred memory** ([`deferred_of_length`],
//! `GpuDevice::new_deferred_buffer_of_length`). A deferred buffer is a Private one of the host's
//! own whose rt buffer, in a block of its own as any Private buffer's, is made at its first use
//! (`Buffer::make_memory`), once, as zeros: rt zeroes it ahead of the next command buffer, as any
//! Private buffer's. Until then it answers its length and options, NULL contents and GPU address 0.
//! Its callers make it ahead of each use that can answer a refusal; a texture over its bytes makes
//! it first, and is nil where rt refuses it. Any other use makes it there, and where rt refuses its
//! memory then, the process ends: that use has no refusal to answer ([`deferred_refused`]). Each
//! make is rt's take of a Private block on the thread of the use, which allocates there where no
//! free range of rt's pool fits the length.
//!
//! Each deferred buffer is counted as it is made and gone, with the most live at once
//! ([`DEFERRED`]); each whose memory is made ([`DEFERRED_MADE`]), and of those each made by the use
//! itself, which answers no refusal ([`DEFERRED_ON_USE_CODE`]), with the nanoseconds the makes took
//! on their threads ([`DEFERRED_MADE_NS_CODE`]) and the longest ([`DEFERRED_MADE_NS_MOST_CODE`]),
//! and the makes and the longest by what each was made for ([`DEFERRED_FOR_CODES`]);
//! each that goes unmade ([`DEFERRED_GONE_UNMADE_CODE`]); and each refusal of the memory
//! ([`DEFERRED_REFUSED_CODE`]).

use std::collections::BTreeMap;
use std::ops::Range;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use metal2vulkan_rt as rt;

use crate::core::host::HostPtr;
use crate::core::num::{ToU64 as _, ToUsize as _};
use crate::core::platform::Handle as _;
use crate::core::platform::device::NoCopyDeallocator;
use crate::core::platform::host_range::HostRange;
use crate::core::platform::os::Clock as _;
#[cfg(unix)]
use crate::core::platform::resource::ExportedMemory;
use crate::core::platform::resource::{Buffer, GpuResource, MemoryFor};
use crate::core::platform::types::NsRange;
use crate::core::platform::values::{BufferDeallocator, BufferDesc, SurfacePlane, TextureDesc};
use crate::core::raise::death::{DeathId, DeathKind, fatal_with_detail};
use crate::core::raise::{OnFault, RaiseResult};
use crate::core::sync::Mutex;
use crate::gpu::device::descriptor_options;

use super::device::RtDevice;
use super::diag::{Call, ImportPath, Made, Shape};
use super::texture::{self, RtTexture, refused_metal};
use super::tiled::{self, Tiled};
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

/// The codes of a deferred buffer (the module's docs, "Deferred memory"), counted as it is made and
/// as it goes, and the most live at once.
pub const DEFERRED: crate::answer::MadeCodes = crate::answer::MadeCodes {
    made: "rt-deferred-buffers",
    live_most: "rt-deferred-buffers-live-most",
};

/// The codes of a deferred buffer whose memory was made, counted as it is made and as the buffer
/// goes, and the most live at once.
pub const DEFERRED_MADE: crate::answer::MadeCodes = crate::answer::MadeCodes {
    made: "rt-deferred-buffers-made",
    live_most: "rt-deferred-buffers-made-live-most",
};

/// The code whose count is the nanoseconds a deferred buffer's memory took to make, on the thread
/// of its first use, summed.
pub const DEFERRED_MADE_NS_CODE: &str = "rt-deferred-buffers-made-ns";

/// The code whose count is the longest a deferred buffer's memory took to make, in nanoseconds.
pub const DEFERRED_MADE_NS_MOST_CODE: &str = "rt-deferred-buffers-made-ns-most";

/// The code of a deferred buffer's memory rt refused, counted per refusal.
pub const DEFERRED_REFUSED_CODE: &str = "rt-deferred-buffers-refused";

/// The code of a deferred buffer whose memory was made by the use itself, which answers no
/// refusal ([`RtBuffer::rt`]), not ahead of it, counted per buffer.
pub const DEFERRED_ON_USE_CODE: &str = "rt-deferred-buffers-made-on-use";

/// The code of a deferred buffer that went with its memory never made, counted per buffer.
pub const DEFERRED_GONE_UNMADE_CODE: &str = "rt-deferred-buffers-released-unmade";

/// The codes of a deferred buffer whose memory is not made, counted as it is made and as its
/// memory is made or it goes unmade, and the most so at once.
pub const DEFERRED_UNMADE: crate::answer::MadeCodes = crate::answer::MadeCodes {
    made: "rt-deferred-buffers-unmade",
    live_most: "rt-deferred-buffers-unmade-live-most",
};

/// The code whose count is the longest block take of a deferred buffer's memory, in nanoseconds
/// (`Device::take_block`, which allocates where no free range fits).
pub const DEFERRED_TAKE_NS_MOST_CODE: &str = "rt-deferred-buffers-take-ns-most";

/// The code whose count is the longest make of a deferred buffer's rt buffer over its block, in
/// nanoseconds (`Device::new_buffer`, which asks its zeros).
pub const DEFERRED_BUFFER_NS_MOST_CODE: &str = "rt-deferred-buffers-buffer-ns-most";

/// The code whose count is the nanoseconds of rt's allocations on the consumer's threads while the
/// block takes of deferred buffers' memory ran (`Metrics::consumer_allocation_nanos`, every
/// thread's: an allocation of another thread's beside a take counts too).
pub const DEFERRED_TAKE_ALLOCATION_NS_CODE: &str = "rt-deferred-buffers-take-allocation-ns";

/// The code whose count is the most bytes of rt's Private allocations (`Gauges::
/// private_allocation_bytes`: the blocks handed out, the free ranges between them and the idle
/// allocations) seen after a take of a Private block ([`take`]), where they rise: the peak of
/// the memory the platform's Private blocks hold, as of their takes.
pub const PRIVATE_ALLOCATION_MOST_CODE: &str = "rt-private-allocation-bytes-most";

/// The most bytes of rt's Private allocations seen after a take ([`PRIVATE_ALLOCATION_MOST_CODE`]).
static PRIVATE_ALLOCATION_MOST: AtomicU64 = AtomicU64::new(0);

/// The codes of the deferred buffers' makes by what they were made for: each [`MemoryFor`]'s, by
/// its discriminant, then the use's own ([`DEFERRED_ON_USE_CODE`]): the count of makes, and the
/// longest one, in nanoseconds.
pub const DEFERRED_FOR_CODES: [(&str, &str); 10] = [
    (
        "rt-deferred-buffers-made-for-reconcile",
        "rt-deferred-buffers-made-for-reconcile-ns-most",
    ),
    (
        "rt-deferred-buffers-made-for-early-copy",
        "rt-deferred-buffers-made-for-early-copy-ns-most",
    ),
    (
        "rt-deferred-buffers-made-for-fill",
        "rt-deferred-buffers-made-for-fill-ns-most",
    ),
    (
        "rt-deferred-buffers-made-for-fill-shapes",
        "rt-deferred-buffers-made-for-fill-shapes-ns-most",
    ),
    (
        "rt-deferred-buffers-made-for-write-back",
        "rt-deferred-buffers-made-for-write-back-ns-most",
    ),
    (
        "rt-deferred-buffers-made-for-download",
        "rt-deferred-buffers-made-for-download-ns-most",
    ),
    (
        "rt-deferred-buffers-made-for-release",
        "rt-deferred-buffers-made-for-release-ns-most",
    ),
    (
        "rt-deferred-buffers-made-for-command",
        "rt-deferred-buffers-made-for-command-ns-most",
    ),
    (
        "rt-deferred-buffers-made-for-texture",
        "rt-deferred-buffers-made-for-texture-ns-most",
    ),
    (
        DEFERRED_ON_USE_CODE,
        "rt-deferred-buffers-made-on-use-ns-most",
    ),
];

/// The code of a no-copy buffer or a heap of option bit 24 pinned in a known block that is no copy
/// whose writes are dropped (the module's docs, "Read-only memory"): the GPU's writes reach its
/// bytes, where Metal's GPU drops a kernel's and a blit's writes and faults a render pass's store.
/// Counted per buffer or heap.
pub const READ_ONLY_CODE: &str = "rt-no-copy-read-only";

/// The code of an import of the host's memory made as a copy whose GPU writes are dropped, of
/// read-only pages (the module's docs, "Read-only memory"), counted per import, each counted as
/// [`IMPORTS_COPIED`] too.
pub const IMPORTS_READ_ONLY: &str = "rt-imports-read-only";

/// The code of an import of the host's memory made by its pointer (rt's `import_host_memory`),
/// counted per import.
pub const IMPORTS_POINTER: &str = "rt-imports-pointer";

/// The code of an import of the host's memory made as a dma-buf of the files behind its pages, the
/// GPU taking no pointer to them (the module's docs, "Dma-bufs of the pages"), counted per import.
pub const IMPORTS_DMA_BUF: &str = "rt-imports-dma-buf";

/// The code of an import of the host's memory made as several dma-bufs of the files behind its
/// pages bound behind one buffer (rt's `import_dma_bufs`), its pages more runs of file pages, or
/// more bytes, than one dma-buf holds (the module's docs, "Dma-bufs of the pages"), counted per
/// import, each counted as [`IMPORTS_DMA_BUF`] too.
pub const IMPORTS_DMA_BUFS: &str = "rt-imports-dma-bufs";

/// The code of an import of the host's memory whose pages are more runs of file pages, or more
/// bytes, than one dma-buf holds, and of which no several dma-bufs were imported: rt binds none
/// behind one buffer on the device (no granule), the host made none, or rt refused them (the
/// module's docs, "Dma-bufs of the pages"), counted per import asked, the process's first of each
/// with a line naming why. Each goes on as one dma-buf's too large ([`IMPORTS_TOO_LARGE`],
/// [`IMPORTS_FROM_FILES_TOO_LARGE`]).
pub const IMPORTS_DMA_BUFS_REFUSED: &str = "rt-imports-dma-bufs-refused";

/// The code of an import of the host's memory made by its pointer once the pages of a view mapped
/// through dma-bufs, of which no dma-buf was imported, were mapped from their files (the module's
/// docs, "Pages no pin takes"), counted per import, each counted as [`IMPORTS_POINTER`] too.
pub const IMPORTS_FROM_FILES: &str = "rt-imports-from-files";

/// The code of an import of the host's memory made by its pointer once the pages of a view mapped
/// through dma-bufs were mapped from their files, as more runs of file pages, or more bytes, than
/// one dma-buf holds (the module's docs, "Pages no pin takes"), counted per import, each counted
/// as [`IMPORTS_FROM_FILES`] too.
pub const IMPORTS_FROM_FILES_TOO_LARGE: &str = "rt-imports-from-files-too-large";

/// The code of an import of the host's memory made by its pointer once the pages of a view mapped
/// through dma-bufs were mapped from their files, rt having refused the dma-buf of them the host
/// made (the module's docs, "Pages no pin takes"), counted per import, each counted as
/// [`IMPORTS_FROM_FILES`] too.
pub const IMPORTS_FROM_FILES_DMA_BUF_REFUSED: &str = "rt-imports-from-files-dma-buf-refused";

/// The code of an import of the host's memory made by its pointer once the pages of a view mapped
/// through dma-bufs were mapped from their files, the host having made no dma-buf of them (a byte
/// with no file behind it, or udmabuf's refusal; the module's docs, "Pages no pin takes"), counted
/// per import, each counted as [`IMPORTS_FROM_FILES`] too.
pub const IMPORTS_FROM_FILES_NO_DMA_BUF: &str = "rt-imports-from-files-no-dma-buf";

/// The code of an import of the host's memory whose pages are more runs of file pages, or more
/// bytes, than one dma-buf holds, of which no several dma-bufs were imported
/// ([`IMPORTS_DMA_BUFS_REFUSED`]), the GPU taking no pointer to them (the module's docs, "Dma-bufs
/// of the pages"), made as a copy where one is ([`IMPORTS_COPIED`]), counted per import asked.
pub const IMPORTS_TOO_LARGE: &str = "rt-imports-too-large";

/// The code of an import of the host's memory that neither the pages' pointer nor a dma-buf of
/// theirs took otherwise, made as a copy where one is ([`IMPORTS_COPIED`]), or refused, counted
/// per import asked.
pub const IMPORTS_REFUSED: &str = "rt-imports-refused";

/// The code of an import of the host's memory made as a copy of its pages (the module's docs, "A
/// copy at the refusal point"), counted per import, each counted as [`IMPORTS_TOO_LARGE`] or
/// [`IMPORTS_REFUSED`] too.
pub const IMPORTS_COPIED: &str = "rt-imports-copied";

/// The code of an import of the host's memory whose pages rt refused both as a dma-buf of the
/// files behind them and by their pointer, in either order (on Linux; the module's docs, "A copy
/// at the refusal point"), counted per import asked, each counted as [`IMPORTS_REFUSED`] too.
pub const IMPORTS_DMA_BUF_AND_POINTER_REFUSED: &str = "rt-imports-dma-buf-and-pointer-refused";

/// The code of a Managed no-copy buffer asked over a dma-buf's block, a block rt does not map (the
/// module's docs, "Dma-bufs of the pages"), counted per buffer.
pub const MANAGED_OVER_DMA_BUF: &str = "rt-managed-over-dma-buf";

/// The code of the bytes the platform zeroed through a block's mapping: a Managed buffer's own
/// bytes, and a Shared heap's of its own (`super::heap`). Counted by the byte; rt's zeroing is in
/// rt's metrics.
pub const ZEROED_BYTES_CODE: &str = "rt-zeroed-bytes";

/// A buffer descriptor's `resourceOptions` before any setter: `Shared`.
pub(super) const BUFFER_DESCRIPTOR_OPTIONS: u64 = 0;

/// The resource options' CPU cache, storage and hazard tracking modes: bits 0 to 9.
pub(super) const FIELDS: u64 = 0x3ff;

/// Resource option bit 20, which Metal takes and answers without (measured).
pub(super) const BIT_20: u64 = 1 << 20;

/// Resource option bit 24, read-only, which Metal takes for a no-copy buffer and answers without,
/// and at which a buffer the device makes ends Metal's process (measured).
pub(super) const READ_ONLY: u64 = 1 << 24;

/// `MTLStorageModeManaged`, in the resource options' bits 4 to 7.
const MANAGED: u64 = 1;

/// `MTLTextureType2D`.
const TEXTURE_TYPE_2D: u64 = 2;

/// `id<MTLBuffer>` on metal2vulkan-rt: rt's buffer and what the platform answers of it.
pub struct RtBuffer {
    /// rt's buffer: a deferred one's from its first use (the module's docs, "Deferred memory").
    rt: OnceLock<rt::Buffer>,
    /// A deferred buffer's make, `None` for any other.
    deferred: Option<Deferred>,
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
    /// An exportable buffer's export (`super::export`).
    #[cfg(unix)]
    exported: Option<super::export::Exported>,
    /// The copy of host pages it lies over (`super::copied`), which a command buffer that uses it
    /// syncs; `None` for any other.
    copied: Option<Arc<super::copied::Copied>>,
}

/// What a deferred buffer's memory is made of at its first use (the module's docs, "Deferred
/// memory").
struct Deferred {
    /// Its length.
    length: u64,
    /// Held while its memory is made, so that one make is made.
    making: Mutex<()>,
}

impl RtBuffer {
    /// rt's buffer; a deferred one's made now where it is not yet, and where rt refuses its
    /// memory the process ends ([`deferred_refused`]): the caller's use answers no refusal.
    pub fn rt(&self) -> &rt::Buffer {
        match self.made(None) {
            Some(made) => made,
            None => deferred_refused(self.deferred.as_ref().map_or(0, |d| d.length)),
        }
    }

    /// rt's buffer; a deferred one's made now where it is not yet, `None` where rt refuses its
    /// memory, counted (the module's docs, "Deferred memory") under what it is made for, `None`
    /// where the caller is the use itself, which answers no refusal.
    fn made(&self, made_for: Option<MemoryFor>) -> Option<&rt::Buffer> {
        if let Some(made) = self.rt.get() {
            return Some(made);
        }
        let deferred = self.deferred.as_ref()?;
        let _making = deferred.making.lock();
        if let Some(made) = self.rt.get() {
            return Some(made);
        }
        // As `own`'s, its two calls of rt's timed apart ([`DeferredSplit`]).
        let length = deferred.length;
        let allocated = allocations_now();
        let start = rt::host_time();
        let block = take(&self.device, DEFERRED_SEL, rt::MemoryClass::Private, length);
        let taken = rt::host_time();
        let made = block.and_then(|block| {
            match (self.device.rt()).new_buffer(block.id, 0, length, Some(0)) {
                Ok(made) => Some(made),
                Err(error) => {
                    rt_refused_create(DEFERRED_SEL, &error);
                    None
                }
            }
        });
        let end = rt::host_time();
        let took = end.saturating_sub(start);
        let Some(made) = made else {
            crate::answer::count_kind(DEFERRED_REFUSED_CODE);
            return None;
        };
        DeferredSplit {
            length,
            made_for,
            take: taken.saturating_sub(start),
            buffer: end.saturating_sub(taken),
            allocations: (allocations_now().zip(allocated))
                .map(|(a, b)| (a.0.saturating_sub(b.0), a.1.saturating_sub(b.1))),
        }
        .count();
        crate::answer::count_made(DEFERRED_MADE);
        crate::answer::count_gone(DEFERRED_UNMADE);
        crate::answer::count_kind_by(DEFERRED_MADE_NS_CODE, took);
        let most = DEFERRED_MOST_NS.fetch_max(took, Ordering::Relaxed);
        crate::answer::count_kind_by(DEFERRED_MADE_NS_MOST_CODE, took.saturating_sub(most));
        let made_for = made_for.map_or(DEFERRED_FOR_CODES.len() - 1, |made_for| made_for as usize);
        let (count, most_code) = DEFERRED_FOR_CODES[made_for];
        crate::answer::count_kind(count);
        let most = DEFERRED_FOR_MOST_NS[made_for].fetch_max(took, Ordering::Relaxed);
        crate::answer::count_kind_by(most_code, took.saturating_sub(most));
        Some(self.rt.get_or_init(|| made))
    }

    /// The copy of host pages it lies over, where it lies over one (`super::copied`).
    pub(super) fn copied(&self) -> Option<&Arc<super::copied::Copied>> {
        self.copied.as_ref()
    }
}

/// The longest a deferred buffer's memory took to make, in nanoseconds
/// ([`DEFERRED_MADE_NS_MOST_CODE`]).
static DEFERRED_MOST_NS: AtomicU64 = AtomicU64::new(0);

/// The longest take, buffer's make and whole make of a deferred buffer's memory so far, which
/// [`DEFERRED_TAKE_NS_MOST_CODE`] and [`DEFERRED_BUFFER_NS_MOST_CODE`] rise to, and past which a
/// make is logged ([`DeferredSplit`]).
static DEFERRED_SPLIT_MOST: [AtomicU64; 3] =
    [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// One deferred buffer's memory made (the module's docs, "Deferred memory"), of `length` bytes,
/// for `made_for` (`None`, the use itself): its two calls of rt's apart, the block's take
/// (`Device::take_block`, which allocates where no free range fits) and the buffer's make over it
/// (`Device::new_buffer`, which asks its zeros), in nanoseconds; and rt's allocations on the
/// consumer's threads while the take ran, how many and their nanoseconds.
struct DeferredSplit {
    length: u64,
    made_for: Option<MemoryFor>,
    take: u64,
    buffer: u64,
    allocations: Option<(u64, u64)>,
}

impl DeferredSplit {
    /// Counted ([`DEFERRED_TAKE_NS_MOST_CODE`], [`DEFERRED_BUFFER_NS_MOST_CODE`],
    /// [`DEFERRED_TAKE_ALLOCATION_NS_CODE`]), and logged where the make is the longest so far.
    fn count(&self) {
        let rise = |at: usize, value: u64| {
            let was = DEFERRED_SPLIT_MOST[at].fetch_max(value, Ordering::Relaxed);
            value.saturating_sub(was)
        };
        let (allocations, allocation_ns) = self.allocations.unwrap_or((0, 0));
        crate::answer::count_kind_by(DEFERRED_TAKE_NS_MOST_CODE, rise(0, self.take));
        crate::answer::count_kind_by(DEFERRED_BUFFER_NS_MOST_CODE, rise(1, self.buffer));
        crate::answer::count_kind_by(DEFERRED_TAKE_ALLOCATION_NS_CODE, allocation_ns);
        let whole = self.take.saturating_add(self.buffer);
        if rise(2, whole) > 0 {
            let ms = |ns: u64| ns as f64 / 1e6;
            let made_for = self
                .made_for
                .map_or("its use".to_owned(), |m| format!("{m:?}"));
            log::info!(
                target: TARGET,
                "a deferred buffer's memory of {} KiB took {:.3} ms to make for {made_for}, the \
                 longest yet: its block's take {:.3} ms ({allocations} allocations of rt's on \
                 the consumer's threads, {:.3} ms, while it ran), the buffer over it {:.3} ms",
                self.length.div_ceil(1024),
                ms(whole),
                ms(self.take),
                ms(allocation_ns),
                ms(self.buffer)
            );
        }
    }
}

/// The longest a deferred buffer's memory took to make for each use, in nanoseconds, by the index
/// of its codes in [`DEFERRED_FOR_CODES`].
static DEFERRED_FOR_MOST_NS: [AtomicU64; DEFERRED_FOR_CODES.len()] =
    [const { AtomicU64::new(0) }; DEFERRED_FOR_CODES.len()];

/// The selector a deferred buffer's memory is made for, as its buffer stands for one
/// `newBufferWithLength:options:` makes.
const DEFERRED_SEL: &str = "newBufferWithLength:options:";

/// The death at a use of a deferred buffer of `length` bytes that answers no refusal, where rt
/// refused its memory (the module's docs, "Deferred memory").
#[cold]
#[inline(never)]
fn deferred_refused(length: u64) -> ! {
    fatal_with_detail(
        DeathId::Platform {
            kind: DeathKind::Abort,
            call: DEFERRED_SEL,
        },
        &format!(
            "metal2vulkan-rt refused the memory of a deferred buffer of {length} bytes at a use \
             that answers no refusal ({DEFERRED_REFUSED_CODE})"
        ),
    )
}

impl Drop for RtBuffer {
    fn drop(&mut self) {
        if let Some(slot) = self.deallocator.take() {
            run(&slot);
        }
        if self.deferred.is_some() {
            crate::answer::count_gone(DEFERRED);
            match self.rt.get() {
                Some(_) => crate::answer::count_gone(DEFERRED_MADE),
                None => {
                    crate::answer::count_kind(DEFERRED_GONE_UNMADE_CODE);
                    crate::answer::count_gone(DEFERRED_UNMADE);
                }
            }
        }
    }
}

/// The blocks the device knows by their GPU address, which a pinned buffer or heap inside one is a
/// range of (the module's docs, "A pinned address"): the graphics lane's, which knows a block as it
/// hands its range to the guest and forgets it as it takes it back, just before it gives the block
/// back to rt. Each held block is a handle of it, so it lives while known. A handle of the
/// device's set: its clones are the same set
/// (`RtDevice::known_blocks`). With them, the host's mappings of the device's dma-buf imports and
/// the pages of its copies ([`HostMappings`]), known or not.
#[derive(Default, Clone)]
pub struct KnownBlocks {
    /// The blocks known, by their GPU address.
    known: Arc<Mutex<BTreeMap<u64, rt::Block>>>,
    /// The host's mapping of each live dma-buf import.
    mapped: HostMappings,
    /// The pages each live copy's import holds a copy of (the module's docs, "A copy at the
    /// refusal point"), the block's `contents` in place of rt's mapping of the copy.
    homes: HostMappings,
}

/// Interim, until rt takes the mapping of a dma-buf it imports, which it answers no `contents`
/// for: the host address of the first byte of each live dma-buf import of the device's
/// ([`import_pages`], the module's docs, "Dma-bufs of the pages"). It is the mapping whose pages
/// the dma-buf holds (the guest's RAM's, or a view's of it), so the CPU's writes through it and
/// the GPU's through the import reach the same pages, and each side sees the other's. An import's
/// entry goes as rt frees it, from its release, which holds this map alone: no handle of rt's.
#[derive(Default, Clone)]
struct HostMappings(Arc<Mutex<BTreeMap<rt::BlockId, u64>>>);

impl HostMappings {
    /// The host address of the first byte of the dma-buf import `id`, where it is one.
    fn of(&self, id: rt::BlockId) -> Option<u64> {
        self.0.lock().get(&id).copied()
    }
}

impl KnownBlocks {
    /// Knows `block` by its GPU address, in place of a block known there before.
    pub fn know(&self, block: rt::Block) {
        self.known.lock().insert(block.gpu_address, block);
    }

    /// Forgets the block `id`, giving back the handle it held; `None` where none is known.
    pub fn forget(&self, id: rt::BlockId) -> Option<rt::Block> {
        let mut known = self.known.lock();
        let at = known
            .iter()
            .find(|(_, block)| block.id == id)
            .map(|(&at, _)| at)?;
        known.remove(&at)
    }

    /// The known block that holds the `length` bytes at GPU address `address`, and their offset
    /// in it.
    pub fn at(&self, address: u64, length: u64) -> Option<(rt::Block, u64)> {
        let known = self.known.lock();
        let (&start, block) = known.range(..=address).next_back()?;
        let offset = address - start;
        let inside = offset
            .checked_add(length)
            .is_some_and(|end| end <= block.len);
        inside.then(|| (block.clone(), offset))
    }

    /// The known Shared block whose mapping holds the `length` bytes at `first`, and their offset
    /// in it: rt's, or the host's of a dma-buf import ([`Self::contents_of`]).
    pub fn mapping(&self, first: HostPtr, length: u64) -> Option<(rt::Block, u64)> {
        let first = first.addr().to_u64();
        let known = self.known.lock();
        known.values().find_map(|block| {
            let start = self.contents_of(block)?;
            let offset = first.checked_sub(start)?;
            let inside = offset
                .checked_add(length)
                .is_some_and(|end| end <= block.len);
            let shared = block.class == rt::MemoryClass::Shared;
            (shared && inside).then(|| (block.clone(), offset))
        })
    }

    /// The host address of `block`'s first byte: for a copy's block the pages it copies
    /// ([`Self::home_of`]), else rt's mapping of it, else the host's mapping of the pages a dma-buf
    /// import of the device's holds ([`HostMappings`]); `None` for a block with none.
    pub(super) fn contents_of(&self, block: &rt::Block) -> Option<u64> {
        if let Some(home) = self.home_of(block.id) {
            return Some(home);
        }
        match block.contents {
            Some(at) => Some(at.as_ptr().addr().to_u64()),
            None => self.mapped.of(block.id),
        }
    }

    /// The host address of the first byte of the pages the copy's import `id` holds a copy of,
    /// where it is one (the module's docs, "A copy at the refusal point").
    fn home_of(&self, id: rt::BlockId) -> Option<u64> {
        super::copied::any().then(|| self.homes.of(id)).flatten()
    }

    /// Whether `id` is a live dma-buf import of the device's ([`HostMappings`]).
    fn is_dma_buf(&self, id: rt::BlockId) -> bool {
        self.mapped.of(id).is_some()
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

/// [`of_length`]'s buffer whose bytes its maker writes before anything reads them
/// ([`GpuDevice::new_scratch_buffer_of_length`](crate::core::platform::device::GpuDevice::new_scratch_buffer_of_length)):
/// a Shared one in a block of its own, as [`own`]'s, with its bytes as the block held them, not
/// zeroed; any other as `of_length`'s.
pub fn scratch_of_length(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    length: u64,
    raw: u64,
) -> Option<RtObj<RtBuffer>> {
    let o = match options(raw, false) {
        Ok(o) => o,
        Err(why) => return refused_metal(sel, &why),
    };
    if o.storage != Storage::Shared || length == 0 || length > rt::MAX_BUFFER_LENGTH {
        return made(device, sel, length, o, None, None);
    }
    let clock = crate::os::OsClock::new();
    let allocated = allocations_now();
    let began = clock.now_mono();
    let block = take(device, sel, rt::MemoryClass::Shared, length)?;
    let taken = clock.now_mono();
    let made = device.rt().new_buffer(block.id, 0, length, None);
    let split = ScratchSplit {
        length,
        take: nanos(taken.saturating_duration_since(began)),
        buffer: nanos(clock.now_mono().saturating_duration_since(taken)),
        allocations: (allocations_now().zip(allocated))
            .map(|(a, b)| (a.0.saturating_sub(b.0), a.1.saturating_sub(b.1))),
    };
    split.count();
    match made {
        Ok(made) => Some(buffer(device, made, o.answered)),
        Err(error) => {
            rt_refused_create(sel, &error);
            None
        }
    }
}

/// The Shared scratch buffers made ([`scratch_of_length`]); the nanoseconds their makes' two
/// calls of rt's took in all, and the longest of each: the block's take (`Device::take_block`,
/// which allocates where no free range fits) and the buffer's make over it
/// (`Device::new_buffer`); and the nanoseconds of rt's allocations on the consumer's threads
/// while the takes ran (`Metrics::consumer_allocation_nanos`, every thread's: an allocation of
/// another thread's beside a take counts too). Each new longest make is logged with its split.
pub const SCRATCH_MAKES_CODE: &str = "rt-scratch-makes";
/// The nanoseconds the block takes of [`SCRATCH_MAKES_CODE`] took in all.
pub const SCRATCH_TAKE_NS_CODE: &str = "rt-scratch-take-ns";
/// The longest block take of [`SCRATCH_MAKES_CODE`], which its count rises to.
pub const SCRATCH_TAKE_MAX_NS_CODE: &str = "rt-scratch-take-max-ns";
/// The nanoseconds of rt's allocations on the consumer's threads while the takes of
/// [`SCRATCH_MAKES_CODE`] ran.
pub const SCRATCH_TAKE_ALLOCATION_NS_CODE: &str = "rt-scratch-take-allocation-ns";
/// The nanoseconds the buffers' makes over their blocks of [`SCRATCH_MAKES_CODE`] took in all.
pub const SCRATCH_BUFFER_NS_CODE: &str = "rt-scratch-buffer-ns";
/// The longest buffer's make of [`SCRATCH_MAKES_CODE`], which its count rises to.
pub const SCRATCH_BUFFER_MAX_NS_CODE: &str = "rt-scratch-buffer-max-ns";

/// The longest take, buffer's make and whole make of [`scratch_of_length`]'s so far, which
/// [`SCRATCH_TAKE_MAX_NS_CODE`] and [`SCRATCH_BUFFER_MAX_NS_CODE`] rise to, and past which a make
/// is logged.
static SCRATCH_MOST: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];

/// rt's allocations on the consumer's threads so far, and their nanoseconds
/// (`Metrics::consumer_allocations`, `consumer_allocation_nanos`); `None` before the device is
/// made.
fn allocations_now() -> Option<(u64, u64)> {
    super::device::metrics().map(|m| (m.consumer_allocations, m.consumer_allocation_nanos))
}

/// `took` in nanoseconds.
fn nanos(took: std::time::Duration) -> u64 {
    u64::try_from(took.as_nanos()).unwrap_or(u64::MAX)
}

/// One scratch buffer's make of `length` bytes: its take's and its buffer's nanoseconds, and rt's
/// allocations on the consumer's threads while the take ran, how many and their nanoseconds
/// ([`SCRATCH_MAKES_CODE`]).
struct ScratchSplit {
    length: u64,
    take: u64,
    buffer: u64,
    allocations: Option<(u64, u64)>,
}

impl ScratchSplit {
    /// Counted, and logged where the make is the longest so far.
    fn count(&self) {
        let rise = |at: usize, value: u64| {
            let was = SCRATCH_MOST[at].fetch_max(value, Ordering::Relaxed);
            value.saturating_sub(was)
        };
        let (allocations, allocation_ns) = self.allocations.unwrap_or((0, 0));
        crate::answer::count_kind(SCRATCH_MAKES_CODE);
        crate::answer::count_kind_by(SCRATCH_TAKE_NS_CODE, self.take);
        crate::answer::count_kind_by(SCRATCH_TAKE_MAX_NS_CODE, rise(0, self.take));
        crate::answer::count_kind_by(SCRATCH_TAKE_ALLOCATION_NS_CODE, allocation_ns);
        crate::answer::count_kind_by(SCRATCH_BUFFER_NS_CODE, self.buffer);
        crate::answer::count_kind_by(SCRATCH_BUFFER_MAX_NS_CODE, rise(1, self.buffer));
        let whole = self.take.saturating_add(self.buffer);
        if rise(2, whole) > 0 {
            let ms = |ns: u64| ns as f64 / 1e6;
            log::info!(
                target: TARGET,
                "a scratch buffer of {} KiB took {:.3} ms, the longest yet: its block's take \
                 {:.3} ms ({allocations} allocations of rt's on the consumer's threads, \
                 {:.3} ms, while it ran), the buffer over it {:.3} ms",
                self.length.div_ceil(1024),
                ms(whole),
                ms(self.take),
                ms(allocation_ns),
                ms(self.buffer)
            );
        }
    }
}

/// [`of_length`]'s buffer whose memory is made at its first use
/// ([`GpuDevice::new_deferred_buffer_of_length`](crate::core::platform::device::GpuDevice::new_deferred_buffer_of_length),
/// the module's docs, "Deferred memory"): a Private one; any other as `of_length`'s. Nil where Metal
/// refuses it, counted.
pub fn deferred_of_length(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    length: u64,
    raw: u64,
) -> Option<RtObj<RtBuffer>> {
    let o = match options(raw, false) {
        Ok(o) => o,
        Err(why) => return refused_metal(sel, &why),
    };
    if o.storage != Storage::Private || length == 0 || length > rt::MAX_BUFFER_LENGTH {
        return made(device, sel, length, o, None, None);
    }
    crate::answer::count_made(DEFERRED);
    crate::answer::count_made(DEFERRED_UNMADE);
    Some(RtObj::new(RtBuffer {
        rt: OnceLock::new(),
        deferred: Some(Deferred {
            length,
            making: Mutex::new(()),
        }),
        device: device.clone_ref(),
        options: o.answered,
        purgeable: texture::purgeable_state(),
        in_heap: false,
        deallocator: None,
        #[cfg(unix)]
        exported: None,
        copied: None,
    }))
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
    let Some(contents) = buffer.rt().contents() else {
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
        && let Err(error) = buffer.rt().did_modify_range(0, bytes.len())
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
    buffer_of(device, rt, options, (false, None), None)
}

/// The buffer's handle, of rt's buffer `rt` answering `options`, placed by a heap where `in_heap`,
/// lying over `copied` where it lies over a copy of host pages (`super::copied`), running
/// `deallocator` as it goes where given.
pub(super) fn buffer_of(
    device: &RtObj<RtDevice>,
    rt: rt::Buffer,
    options: u64,
    (in_heap, copied): (bool, Option<Arc<super::copied::Copied>>),
    deallocator: Option<Slot>,
) -> RtObj<RtBuffer> {
    RtObj::new(RtBuffer {
        rt: OnceLock::from(rt),
        deferred: None,
        device: device.clone_ref(),
        options,
        purgeable: texture::purgeable_state(),
        in_heap,
        deallocator,
        #[cfg(unix)]
        exported: None,
        copied,
    })
}

/// The handle of an exportable buffer (`super::export`), of rt's buffer `rt` answering `options`,
/// keeping `exported`.
#[cfg(unix)]
pub(super) fn with_export(
    device: &RtObj<RtDevice>,
    rt: rt::Buffer,
    options: u64,
    exported: super::export::Exported,
) -> RtObj<RtBuffer> {
    RtObj::new(RtBuffer {
        rt: OnceLock::from(rt),
        deferred: None,
        device: device.clone_ref(),
        options,
        purgeable: texture::purgeable_state(),
        in_heap: false,
        deallocator: None,
        exported: Some(exported),
        copied: None,
    })
}

/// The options a buffer of the raw options `raw` answers, where they are Private ones Metal takes
/// for a buffer the device makes; `None` otherwise.
#[cfg(unix)]
pub(super) fn private_options(raw: u64) -> Option<u64> {
    let o = options(raw, false).ok()?;
    (o.storage == Storage::Private).then_some(o.answered)
}

/// The known block of `class` that holds the `length` bytes at GPU address `pin`, and their
/// offset in it; where `bytes` is given, only one whose mapping holds them at that offset (rt's,
/// or the host's of a dma-buf import, `KnownBlocks::contents_of`).
pub(super) fn known_at(
    device: &RtObj<RtDevice>,
    pin: u64,
    length: u64,
    class: rt::MemoryClass,
    bytes: Option<HostPtr>,
) -> Option<(rt::Block, u64)> {
    let known = device.known_blocks();
    let (block, offset) = known.at(pin, length)?;
    let mapped = |at: HostPtr| {
        let first = known.contents_of(&block);
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

/// A block of `class` for `length` bytes, for `sel`; `None` where rt refuses it, counted. A
/// Private one's take counts rt's Private allocations where they rise past their most
/// ([`PRIVATE_ALLOCATION_MOST_CODE`]).
pub(super) fn take(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    class: rt::MemoryClass,
    length: u64,
) -> Option<rt::Block> {
    match device.rt().take_block(class, length) {
        Ok(block) => {
            if class == rt::MemoryClass::Private {
                let now = device.rt().counters().gauges().private_allocation_bytes;
                let was = PRIVATE_ALLOCATION_MOST.fetch_max(now, Ordering::Relaxed);
                crate::answer::count_kind_by(PRIVATE_ALLOCATION_MOST_CODE, now.saturating_sub(was));
            }
            Some(block)
        }
        Err(error) => {
            rt_refused_create(sel, &error);
            None
        }
    }
}

/// rt's buffer of zeros, of `length` bytes, in a block of `class` of its own, for `sel`, which rt
/// zeroes; `None` where rt refuses it, counted.
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
        crate::answer::count_kind_by(ZEROED_BYTES_CODE, length);
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
            // A read-only buffer with no pin kept lies over a copy of its own, which drops the
            // GPU's writes ("Read-only memory").
            let mapping = (!o.read_only).then(|| device.known_blocks().mapping(first, length));
            match mapping.flatten() {
                Some((block, offset)) => (block, offset, Some(Arc::clone(slot))),
                None => {
                    let held = Arc::clone(slot);
                    let then = move || run(&held);
                    let made = import_pages(device, sel, first, length, o.read_only, then);
                    let (block, offset) = made?;
                    (block, offset, None)
                }
            }
        }
    };
    let copy = copy.map(|(block, offset)| (block.id, offset));
    // In place in a known block whose writes are not dropped.
    let in_place = o.read_only && !super::copied::drops(block.id);
    let (bytes, storage) = ((&block, offset), o.storage);
    // The deallocator is kept unrun before the import goes, whose release would run it.
    let Some(made) = range_of(device, sel, bytes, length, storage, zero_from, copy) else {
        keep_unrun(slot);
        return None;
    };
    if in_place {
        read_only(sel);
    }
    let placed = (false, super::copied::of(block.id));
    Some(buffer_of(device, made, o.answered, placed, deallocator))
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
            if device.known_blocks().is_dma_buf(block.id) {
                crate::answer::count_kind(MANAGED_OVER_DMA_BUF);
            }
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
/// bytes' offset in it. Where the GPU takes no pointer to the pages, on Linux, the import is of a
/// dma-buf of the files behind them ("Dma-bufs of the pages"); the pages of a view mapped through
/// dma-bufs go to rt as a dma-buf first, and by their pointer only once mapped from their files
/// ("Pages no pin takes"). Pages whose GPU writes Metal drops (`read_only`) are imported as a
/// copy that drops them, and never otherwise ("Read-only memory"). rt's release of the import runs
/// `then`, after the import is freed. `None` where Metal or rt refuses it, counted, `then` dropped
/// unrun.
pub(super) fn import_pages(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    first: HostPtr,
    length: u64,
    read_only: bool,
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
    let len = end - start;
    let then: Then = Arc::new(Mutex::new(Some(Box::new(then))));
    if read_only {
        let why = "pages whose GPU writes are dropped";
        let made = copied(
            device,
            sel,
            codes,
            (at, len),
            &then,
            &why,
            true,
            |uncopied| {
                refused_uncopied(sel, uncopied);
            },
        )?;
        crate::answer::count_kind(IMPORTS_READ_ONLY);
        return Some(made_import(sel, codes, made, first - start));
    }
    // Held from before the pages' mappings are asked about, so no fold maps them through dma-bufs
    // before rt takes their pointer, nor while rt holds it.
    let held = Held::of(at, len);
    #[cfg(target_os = "linux")]
    {
        if crate::os::through_dma_bufs(at, len) {
            let made = unpinnable(device, sel, codes, (at, len), &then, held)?;
            return Some(made_import(sel, codes, made, first - start));
        }
    }
    let made = by_pointer(device, sel, codes, (at, len), &then, held)?;
    Some(made_import(sel, codes, made, first - start))
}

/// [`import_pages`]'s import made, `made` its block and the offset of the pages' first byte in it,
/// the bytes `skip` past that byte: counted, the block and the bytes' offset in it.
fn made_import(
    sel: &'static str,
    codes: crate::answer::MadeCodes,
    (block, offset): (rt::Block, u64),
    skip: u64,
) -> (rt::Block, u64) {
    imports_rise();
    super::diag::imported(block.id, sel);
    crate::answer::count_made(codes);
    (block, offset + skip)
}

/// What keeps the pages of an import that rt may take by their pointer mapped as they are, from
/// before [`import_pages`] asks where they lie until rt frees the import (`crate::os`'s
/// `MappingsHeld`, on Linux, where views are mapped through dma-bufs); nothing elsewhere.
#[derive(Debug)]
struct Held {
    /// The hold of the pages' mappings, until it drops.
    #[cfg(target_os = "linux")]
    _mappings: crate::os::MappingsHeld,
}

impl Held {
    /// The hold of the `len` bytes of whole pages at `at`.
    fn of(at: NonNull<u8>, len: u64) -> Held {
        #[cfg(not(target_os = "linux"))]
        let _ = (at, len);
        Held {
            #[cfg(target_os = "linux")]
            _mappings: crate::os::hold_mappings(at, len),
        }
    }
}

/// `release`, then `held` dropped: the pages' mappings kept as they are until rt frees the
/// import, whose release this is.
fn holding(release: NoCopyDeallocator, held: Held) -> NoCopyDeallocator {
    Box::new(move || {
        release();
        let _held = held;
    })
}

/// [`import_pages`]'s import of pages that lie in no dma-buf's mapping, the `len` bytes of whole
/// pages at their address, by their pointer, `held` until rt frees it; where rt refuses that, as a
/// dma-buf of the files behind them, and else as a copy ([`by_dma_buf`]).
fn by_pointer(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    codes: crate::answer::MadeCodes,
    (at, len): (NonNull<u8>, u64),
    then: &Then,
    held: Held,
) -> Option<(rt::Block, u64)> {
    let release = holding(released(codes, then, None), held);
    let imported = match pointer_refused() {
        Some(error) => Err(error),
        // SAFETY: the creators' `HostRange` contracts keep the bytes mapped and readable until
        // their deallocator runs, which runs only from the release, once rt freed the import, or
        // as the object over a known block goes, which holds no import; with none, for the
        // object's life, which holds the import. The pages they lie in, which the import covers,
        // are mapped as they are. They are mapped writable too: the no-copy creators' bytes are
        // memory the host maps readable and writable (the device's region, the host's own).
        None => unsafe { (device.rt()).import_host_memory(at, len, Some(release)) },
    };
    match imported {
        Ok((block, _)) => {
            crate::answer::count_kind(IMPORTS_POINTER);
            super::diag::import_path(codes.made, ImportPath::Pointer, len);
            Some((block, 0))
        }
        Err(error) => by_dma_buf(device, sel, codes, (at, len), then, &error),
    }
}

/// [`import_pages`]'s import of pages that lie in a view mapped through dma-bufs, whose pointer
/// no kernel pin takes, the `len` bytes of whole pages at their address (the module's docs,
/// "Pages no pin takes"): as a dma-buf of the files behind them, and where none is made or rt
/// refuses it, by their pointer once they are mapped from their files (`crate::os`'s
/// `map_for_pointer`), `held` until rt frees that import; where they cannot be, or rt refuses
/// that, as a copy ([`refused`]). `None` where the pages are refused, counted: their pointer is
/// never handed to rt while a page of them lies in a dma-buf's mapping.
#[cfg(target_os = "linux")]
fn unpinnable(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    codes: crate::answer::MadeCodes,
    (at, len): (NonNull<u8>, u64),
    then: &Then,
    held: Held,
) -> Option<(rt::Block, u64)> {
    let kind = codes.made;
    let no = match dma_buf(device, sel, (at, len), released_into(codes, then)) {
        Ok(made) => {
            dma_buf_made(sel, kind, len, &UNPINNABLE);
            return Some(made);
        }
        Err(DmaBufNo::Rt(error @ rt::Error::Quiesced)) => {
            return import_refused(sel, kind, len, &error);
        }
        Err(no) => no,
    };
    // rt's refusal of the view's one dma-buf, which it answers only at the call: the views made
    // from now on are mapped from their files at once, so their imports map none of their pages
    // again (the module's docs, "Pages no pin takes").
    if let DmaBufNo::Rt(_) = no {
        crate::os::views_need_files(&no.why());
    }
    if !crate::os::map_for_pointer(at, len) || crate::os::through_dma_bufs(at, len) {
        return refused(device, sel, codes, (at, len), then, &no, None);
    }
    let release = holding(released(codes, then, None), held);
    let imported = match pointer_refused() {
        Some(error) => Err(error),
        // SAFETY: as `by_pointer`'s: the pages are the same, mapped from their files now.
        None => unsafe { (device.rt()).import_host_memory(at, len, Some(release)) },
    };
    match imported {
        Ok((block, _)) => {
            crate::answer::count_kind(IMPORTS_POINTER);
            crate::answer::count_kind(IMPORTS_FROM_FILES);
            let (path, code, said) = match &no {
                DmaBufNo::TooLarge { path, .. } => {
                    (path.from_files(), IMPORTS_FROM_FILES_TOO_LARGE, 0)
                }
                DmaBufNo::Rt(_) => (ImportPath::Pointer, IMPORTS_FROM_FILES_DMA_BUF_REFUSED, 1),
                DmaBufNo::Refused => (ImportPath::Pointer, IMPORTS_FROM_FILES_NO_DMA_BUF, 2),
            };
            crate::answer::count_kind(code);
            super::diag::import_path(kind, path, len);
            if !FROM_FILES_SAID[said].swap(true, Ordering::Relaxed) {
                log::info!(
                    target: TARGET,
                    "{sel}: no dma-buf of the pages of a view mapped through dma-bufs was \
                     imported ({}); each such view is mapped from its files and imported by its \
                     pointer ({IMPORTS_FROM_FILES}, {code}), counted from here",
                    no.why()
                );
            }
            Some((block, 0))
        }
        Err(error) => {
            if let DmaBufNo::Rt(refusal) = &no {
                both_refused(sel, refusal, &error);
            }
            refused(device, sel, codes, (at, len), then, &no, Some(&error))
        }
    }
}

/// Why the pages of a view mapped through dma-bufs go to rt as a dma-buf, which the process's
/// first such import logs ([`dma_buf_made`]).
#[cfg(target_os = "linux")]
const UNPINNABLE: &str = "the pages lie in a view mapped through dma-bufs, whose pointer no \
                          kernel pin takes";

/// Why the pages of a view mapped through dma-bufs were refused with no pointer asked of rt: they
/// could not all be mapped from their files, the views' budget near (`crate::os`'s
/// `map_for_pointer`), or the kernel refusing a map.
#[cfg(target_os = "linux")]
const UNMAPPED: &str = "the pages of a view mapped through dma-bufs, whose dma-buf was not \
                        imported, could not be mapped from their files for their pointer";

/// Whether the process logged its first import of a view's pages mapped from their files for
/// their pointer ([`unpinnable`]), for each reason no dma-buf of them was imported: too large for
/// one, refused by rt, none made by the host.
#[cfg(target_os = "linux")]
static FROM_FILES_SAID: [AtomicBool; 3] = [const { AtomicBool::new(false) }; 3];

/// What an import's release runs, once: taken from its slot, which the release of an import
/// refused and that of the one made in its place share ([`import_pages`]).
type Then = Arc<Mutex<Option<Box<dyn FnOnce() + Send>>>>;

/// The release of an import for a make of `codes`, which runs `then`: one fewer import live, and,
/// for a dma-buf's or a copy's, whose block `forget` names once it is made, the host's mapping of
/// it, or the pages it copies, forgotten first ([`HostMappings`]). It makes no call into rt and
/// drops no handle of rt's.
fn released(
    codes: crate::answer::MadeCodes,
    then: &Then,
    forget: Option<(HostMappings, Arc<OnceLock<rt::BlockId>>)>,
) -> NoCopyDeallocator {
    let then = Arc::clone(then);
    Box::new(move || {
        LIVE_IMPORTS.fetch_sub(1, Ordering::Relaxed);
        crate::answer::count_gone(codes);
        if let Some((mapped, id)) = forget
            && let Some(id) = id.get()
        {
            mapped.0.lock().remove(id);
        }
        let then = then.lock().take();
        if let Some(then) = then {
            then();
        }
    })
}

/// [`released`] for a dma-buf's import, given the host's mappings it forgets its block from and
/// the slot its block's number goes into once it is made.
#[cfg(target_os = "linux")]
type Released = Box<dyn FnOnce(HostMappings, Arc<OnceLock<rt::BlockId>>) -> NoCopyDeallocator>;

/// The release of a dma-buf's import for a make of `codes` that runs `then` ([`released`]).
#[cfg(target_os = "linux")]
fn released_into(codes: crate::answer::MadeCodes, then: &Then) -> Released {
    let then = Arc::clone(then);
    Box::new(move |mapped, id| released(codes, &then, Some((mapped, id))))
}

/// [`by_pointer`]'s import of `pages`, the `len` bytes of whole pages at their address, as a
/// dma-buf of the files behind them, where rt refused their pointer with `error` (the module's
/// docs, "Dma-bufs of the pages"), for a make of `codes`, rt's release of it running `then`: the
/// block, and the offset of the pages' first byte in it, whose host address the device keeps for
/// the block ([`HostMappings`]) until rt frees it; past what one dma-buf holds, as several
/// ([`dma_bufs`]). Where none of those is imported, with no file behind them, or where rt refuses
/// the dma-buf, as a copy ([`refused`]). `None` where the pages are
/// refused, counted: as rt's last refusal of them.
#[cfg(target_os = "linux")]
fn by_dma_buf(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    codes: crate::answer::MadeCodes,
    (at, len): (NonNull<u8>, u64),
    then: &Then,
    error: &rt::Error,
) -> Option<(rt::Block, u64)> {
    let kind = codes.made;
    if matches!(error, rt::Error::Quiesced) {
        return import_refused(sel, kind, len, error);
    }
    match dma_buf(device, sel, (at, len), released_into(codes, then)) {
        Ok(made) => {
            dma_buf_made(sel, kind, len, error);
            Some(made)
        }
        Err(no) => {
            let last = match &no {
                DmaBufNo::Rt(refusal) => {
                    both_refused(sel, refusal, error);
                    refusal
                }
                DmaBufNo::TooLarge { .. } | DmaBufNo::Refused => error,
            };
            refused(device, sel, codes, (at, len), then, &no, Some(last))
        }
    }
}

/// [`import_pages`]'s import of `pages` where rt refused their pointer with `error`, off Linux: as
/// a copy ([`copied`]), counted as refused ([`IMPORTS_REFUSED`]); `None` where rt refuses that too,
/// counted as rt's refusal of the pointer.
#[cfg(not(target_os = "linux"))]
fn by_dma_buf(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    codes: crate::answer::MadeCodes,
    (at, len): (NonNull<u8>, u64),
    then: &Then,
    error: &rt::Error,
) -> Option<(rt::Block, u64)> {
    let kind = codes.made;
    if matches!(error, rt::Error::Quiesced) {
        return import_refused(sel, kind, len, error);
    }
    count_refused(kind, len);
    let why = format!("metal2vulkan-rt refused their pointer: {error}");
    copied(device, sel, codes, (at, len), then, &why, false, |_| {
        rt_refused_create(sel, error);
    })
}

/// Why a dma-buf of an import's pages was not imported ([`dma_buf`]).
#[cfg(target_os = "linux")]
#[derive(Debug)]
enum DmaBufNo {
    /// More runs of file pages, or more bytes, than one dma-buf holds.
    TooLarge {
        /// The import path counted, with the pages' runs and one dma-buf's limits.
        path: ImportPath,
        /// The pages' bytes.
        bytes: u64,
    },
    /// No dma-buf of them: a byte with no file behind it, or udmabuf's refusal.
    Refused,
    /// rt's refusal of the dma-buf.
    Rt(rt::Error),
}

#[cfg(target_os = "linux")]
impl DmaBufNo {
    /// The refusal, in words.
    fn why(&self) -> String {
        match self {
            DmaBufNo::TooLarge { .. } => "more runs or bytes than one dma-buf holds".to_owned(),
            DmaBufNo::Refused => "the host made no dma-buf of them".to_owned(),
            DmaBufNo::Rt(error) => format!("metal2vulkan-rt refused it: {error}"),
        }
    }
}

/// rt's import of a dma-buf of the files behind `pages`, the `len` bytes of whole pages at their
/// address, for `sel`, rt's release of it from `release`, uncounted: the block, and the offset of
/// the pages' first byte in it, whose host address the device keeps for the block
/// ([`HostMappings`]) until rt frees it; else why none was imported, `release` dropped unrun.
/// Pages more than one dma-buf holds go as several bound behind one buffer ([`dma_bufs`]),
/// counted there.
#[cfg(target_os = "linux")]
fn dma_buf(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    (at, len): (NonNull<u8>, u64),
    release: Released,
) -> Result<(rt::Block, u64), DmaBufNo> {
    use crate::os::LendRefusal;
    #[cfg(test)]
    if HOST_MAKES_NO_DMA_BUF.get() {
        return Err(DmaBufNo::Refused);
    }
    let lent = match crate::os::dma_buf_of_host(at, len, &[(0, len)]) {
        Ok(lent) => lent,
        Err(LendRefusal::TooLarge {
            pieces,
            bytes,
            most_pieces,
            most_bytes,
            ..
        }) => {
            let path = ImportPath::TooLarge {
                pieces,
                most_pieces,
                most_bytes,
            };
            let too_large = DmaBufNo::TooLarge { path, bytes };
            return dma_bufs(device, sel, (at, len), release, too_large);
        }
        Err(LendRefusal::Refused(_)) => return Err(DmaBufNo::Refused),
    };
    #[cfg(test)]
    if DMA_BUF_REFUSED.get() {
        return Err(DmaBufNo::Rt(rt::Error::Vulkan {
            call: "vkAllocateMemory",
            result: ash::vk::Result::ERROR_INVALID_EXTERNAL_HANDLE,
        }));
    }
    // The pages' first byte lies `skip` into the dma-buf, whose first byte the host maps `skip`
    // before it: the host's pages are whole in it.
    let skip = lent.offsets.first().copied().unwrap_or(0);
    let mapped = device.known_blocks().mapped.clone();
    let id = Arc::new(OnceLock::new());
    let release = release(mapped.clone(), Arc::clone(&id));
    match device
        .rt()
        .import_dma_buf(lent.fd, lent.size, Some(release))
    {
        Ok((block, _)) => {
            let _ = id.set(block.id);
            let first = at.as_ptr().addr().to_u64().wrapping_sub(skip);
            mapped.0.lock().insert(block.id, first);
            Ok((block, skip))
        }
        Err(refusal) => Err(DmaBufNo::Rt(refusal)),
    }
}

/// [`dma_buf`]'s import of `pages`, the `len` bytes of whole pages at their address, for `sel`,
/// where one dma-buf does not hold them, as `too_large` says: rt's import of several dma-bufs of
/// the files behind them, cut at rt's granule and the last padded, bound behind one buffer (rt's
/// `import_dma_bufs`; the module's docs, "Dma-bufs of the pages"), counted ([`IMPORTS_DMA_BUFS`]),
/// the process's first with a line; the block, whose host address the device keeps for it from
/// its first byte ([`HostMappings`]) until rt frees it, and 0. Where rt binds no dma-bufs behind
/// one buffer on the device, the host makes none, or rt refuses them, counted
/// ([`IMPORTS_DMA_BUFS_REFUSED`]), and `too_large`; rt's quiescing is rt's refusal. `release`
/// dropped unrun where none was imported.
#[cfg(target_os = "linux")]
fn dma_bufs(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    (at, len): (NonNull<u8>, u64),
    release: Released,
    too_large: DmaBufNo,
) -> Result<(rt::Block, u64), DmaBufNo> {
    let Some(granule) = device.rt().dma_buf_granule() else {
        let why = "metal2vulkan-rt binds no dma-bufs behind one buffer here";
        dma_bufs_refused(sel, 0, &why);
        return Err(too_large);
    };
    let chunks = match crate::os::dma_bufs_of_host(at, len, granule) {
        Ok(chunks) => chunks,
        Err(refusal) => {
            let why = match refusal {
                crate::os::LendRefusal::TooLarge { why, .. } => why,
                crate::os::LendRefusal::Refused(why) => why,
            };
            dma_bufs_refused(sel, 1, &format_args!("the host made none: {why}"));
            return Err(too_large);
        }
    };
    let (count, bytes) = (chunks.len(), chunks.iter().map(|&(_, l)| l).sum::<u64>());
    let mapped = device.known_blocks().mapped.clone();
    let id = Arc::new(OnceLock::new());
    let release = release(mapped.clone(), Arc::clone(&id));
    // Pages one dma-buf holds by now (udmabuf's limits raised since) are that one dma-buf, of their
    // own length, which rt binds as a plain buffer.
    let imported = match <[_; 1]>::try_from(chunks) {
        Ok([(fd, size)]) => device.rt().import_dma_buf(fd, size, Some(release)),
        Err(chunks) => device.rt().import_dma_bufs(chunks, Some(release)),
    };
    match imported {
        Ok((block, _)) => {
            let _ = id.set(block.id);
            let first = at.as_ptr().addr().to_u64();
            mapped.0.lock().insert(block.id, first);
            crate::answer::count_kind(IMPORTS_DMA_BUFS);
            if !DMA_BUFS_SAID.swap(true, Ordering::Relaxed) {
                log::info!(
                    target: TARGET,
                    "{sel}: the pages of a no-copy buffer or heap, more than one dma-buf holds, \
                     were imported as {count} dma-bufs of {bytes} bytes bound behind one buffer \
                     ({granule}-byte granule); each such import is counted from here \
                     ({IMPORTS_DMA_BUFS})"
                );
            }
            Ok((block, 0))
        }
        Err(error @ rt::Error::Quiesced) => Err(DmaBufNo::Rt(error)),
        Err(error) => {
            let why = format!("metal2vulkan-rt refused {count} of {bytes} bytes: {error}");
            dma_bufs_refused(sel, 2, &why);
            Err(too_large)
        }
    }
}

/// Counts an import for `sel` whose pages one dma-buf does not hold, of which no several dma-bufs
/// were imported ([`IMPORTS_DMA_BUFS_REFUSED`]), for the reason numbered `said` (rt binds none
/// here, the host made none, rt refused them); the process's first of each logged, `why` in
/// words.
#[cfg(target_os = "linux")]
fn dma_bufs_refused(sel: &'static str, said: usize, why: &dyn std::fmt::Display) {
    crate::answer::count_kind(IMPORTS_DMA_BUFS_REFUSED);
    if !DMA_BUFS_REFUSED_SAID[said].swap(true, Ordering::Relaxed) {
        log::info!(
            target: TARGET,
            "{sel}: the pages of a no-copy buffer or heap, more than one dma-buf holds, were not \
             imported as several dma-bufs ({why}); each such import is counted from here \
             ({IMPORTS_DMA_BUFS_REFUSED}) and goes on as too large for one"
        );
    }
}

/// Whether the process logged its first import made as several dma-bufs ([`dma_bufs`]).
#[cfg(target_os = "linux")]
static DMA_BUFS_SAID: AtomicBool = AtomicBool::new(false);

/// Whether the process logged its first import not made as several dma-bufs ([`dma_bufs`]), for
/// each reason: rt binds none here, the host made none, rt refused them.
#[cfg(target_os = "linux")]
static DMA_BUFS_REFUSED_SAID: [AtomicBool; 3] = [const { AtomicBool::new(false) }; 3];

/// Counts an import of `len` bytes for `sel` made as a dma-buf ([`dma_buf`]), for a make whose
/// imports are counted as `kind`; the process's first logged, with `why` the pages went so.
#[cfg(target_os = "linux")]
fn dma_buf_made(sel: &'static str, kind: &'static str, len: u64, why: &dyn std::fmt::Display) {
    crate::answer::count_kind(IMPORTS_DMA_BUF);
    super::diag::import_path(kind, ImportPath::DmaBuf, len);
    if !DMA_BUF_SAID.swap(true, Ordering::Relaxed) {
        log::info!(
            target: TARGET,
            "{sel}: the pages of a no-copy buffer or heap were imported as a dma-buf of the \
             files behind them ({why}); each such import is counted from here \
             ({IMPORTS_DMA_BUF})"
        );
    }
}

/// Counts an import for `sel` whose pages rt refused both as a dma-buf, with `dma_buf`, and by
/// their pointer, with `pointer`, in either order ([`IMPORTS_DMA_BUF_AND_POINTER_REFUSED`]); the
/// process's first logged, naming both. Nothing where either is rt's quiescing.
#[cfg(target_os = "linux")]
fn both_refused(sel: &'static str, dma_buf: &rt::Error, pointer: &rt::Error) {
    if matches!(dma_buf, rt::Error::Quiesced) || matches!(pointer, rt::Error::Quiesced) {
        return;
    }
    crate::answer::count_kind(IMPORTS_DMA_BUF_AND_POINTER_REFUSED);
    if !BOTH_REFUSED_SAID.swap(true, Ordering::Relaxed) {
        log::info!(
            target: TARGET,
            "{sel}: metal2vulkan-rt refused both a dma-buf of the pages of a no-copy buffer or \
             heap ({dma_buf}) and their pointer ({pointer}); each such import is counted from \
             here ({IMPORTS_DMA_BUF_AND_POINTER_REFUSED})"
        );
    }
}

/// Whether the process logged its first import whose dma-buf and pointer rt both refused
/// ([`both_refused`]).
#[cfg(target_os = "linux")]
static BOTH_REFUSED_SAID: AtomicBool = AtomicBool::new(false);

/// The import of `pages`, the `len` bytes of whole pages at their address, for `sel`, for a make
/// of `codes`, whose dma-buf was refused as `no`, rt's release of it running `then`: counted as
/// too large, with its runs, or as refused, and made as a copy ([`copied`]). `None` where rt
/// refuses the copy too, or `last` is rt's quiescing, counted as rt's last refusal of the pages,
/// `last`, or where rt was asked for nothing past the dma-buf (the pages left in a dma-buf's
/// mapping, [`UNMAPPED`]), as rt's limit, a gap of the platform's own.
#[cfg(target_os = "linux")]
fn refused(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    codes: crate::answer::MadeCodes,
    (at, len): (NonNull<u8>, u64),
    then: &Then,
    no: &DmaBufNo,
    last: Option<&rt::Error>,
) -> Option<(rt::Block, u64)> {
    let kind = codes.made;
    match *no {
        DmaBufNo::TooLarge { path, bytes } => {
            crate::answer::count_kind(IMPORTS_TOO_LARGE);
            super::diag::import_path(kind, path, bytes);
        }
        DmaBufNo::Refused | DmaBufNo::Rt(_) => count_refused(kind, len),
    }
    let refuse = || match last {
        Some(error) => {
            rt_refused_create(sel, error);
        }
        None => {
            crate::core::platform::device::note_relayed_gap();
            super::unbuilt::refused_as(sel, super::unbuilt::RtRefusal::Limit, &UNMAPPED);
        }
    };
    if matches!(last, Some(rt::Error::Quiesced)) {
        refuse();
        return None;
    }
    copied(
        device,
        sel,
        codes,
        (at, len),
        then,
        &no.why(),
        false,
        |_| refuse(),
    )
}

/// [`import_pages`]'s import of `pages`, the `len` bytes of whole pages at their address, for
/// `sel`, for a make of `codes`, that rt took neither by their pointer nor as a dma-buf, `why` in
/// words, or whose GPU writes are dropped (`drops`, "Read-only memory"): as a copy (the module's
/// docs, "A copy at the refusal point"), whose pages the device keeps for the block
/// ([`KnownBlocks`]) until rt frees it, rt's release of it running `then`; counted
/// ([`IMPORTS_COPIED`]), the process's first with a line. The block, and the offset of the pages'
/// first byte in it. `None` where no copy is made or rt refuses it, counted by `refuse`, given why.
#[expect(
    clippy::too_many_arguments,
    reason = "the import's pages, make and release, and how the copy is made or refused"
)]
fn copied(
    device: &RtObj<RtDevice>,
    sel: &'static str,
    codes: crate::answer::MadeCodes,
    (at, len): (NonNull<u8>, u64),
    then: &Then,
    why: &dyn std::fmt::Display,
    drops: bool,
    refuse: impl FnOnce(&super::copied::Uncopied),
) -> Option<(rt::Block, u64)> {
    let homes = device.known_blocks().homes.clone();
    let id = Arc::new(OnceLock::new());
    let release = released(codes, then, Some((homes.clone(), Arc::clone(&id))));
    // SAFETY: the creators' `HostRange` contracts keep the bytes mapped until their deallocator
    // runs, which runs only from `then`, after the copy's release, once rt freed it; the pages
    // they lie in are mapped readable, as `by_pointer`'s are, and writable unless `drops`, where
    // nothing writes them.
    match unsafe { super::copied::import(device.rt(), (at, len), drops, release) } {
        Ok(block) => {
            let _ = id.set(block.id);
            (homes.0.lock()).insert(block.id, at.as_ptr().addr().to_u64());
            super::diag::copy_made(block.id);
            crate::answer::count_kind(IMPORTS_COPIED);
            if !COPIED_SAID.swap(true, Ordering::Relaxed) {
                log::info!(
                    target: TARGET,
                    "{sel}: the pages of a no-copy buffer or heap were imported as a copy ({why}), \
                     synced at each commit and completion; each such import is counted from here \
                     ({IMPORTS_COPIED})"
                );
            }
            Some((block, 0))
        }
        Err(uncopied) => {
            if !UNCOPIED_SAID.swap(true, Ordering::Relaxed) {
                log::warn!(
                    target: TARGET,
                    "{sel}: the pages of a no-copy buffer or heap were imported neither by their \
                     pointer nor as a dma-buf ({why}), nor as a copy ({uncopied}): refused"
                );
            }
            refuse(&uncopied);
            None
        }
    }
}

/// Whether the process logged its first import made as a copy ([`copied`]).
static COPIED_SAID: AtomicBool = AtomicBool::new(false);

/// Whether the process logged its first copy refused ([`copied`]).
static UNCOPIED_SAID: AtomicBool = AtomicBool::new(false);

/// Counts the refusal of a copy of read-only pages for `sel` ([`import_pages`]), `uncopied` why:
/// rt's refusal of its import, or the host's memory, a limit.
fn refused_uncopied(sel: &'static str, uncopied: &super::copied::Uncopied) {
    match uncopied {
        super::copied::Uncopied::Rt(error) => {
            rt_refused_create(sel, error);
        }
        super::copied::Uncopied::NoMemory => {
            super::unbuilt::refused_as(sel, super::unbuilt::RtRefusal::Limit, uncopied);
        }
    }
}

/// Whether the process logged its first import made as a dma-buf.
#[cfg(target_os = "linux")]
static DMA_BUF_SAID: AtomicBool = AtomicBool::new(false);

/// Counts rt's refusal `error` of an import of `len` bytes for `sel`, for a make whose imports are
/// counted as `kind` ([`IMPORTS_REFUSED`]). `None`.
fn import_refused<T>(
    sel: &'static str,
    kind: &'static str,
    len: u64,
    error: &rt::Error,
) -> Option<T> {
    count_refused(kind, len);
    rt_refused_create(sel, error);
    None
}

/// Counts an import of `len` bytes, for a make whose imports are counted as `kind`, that neither
/// the pages' pointer nor a dma-buf of theirs took ([`IMPORTS_REFUSED`]).
fn count_refused(kind: &'static str, len: u64) {
    crate::answer::count_kind(IMPORTS_REFUSED);
    super::diag::import_path(kind, ImportPath::Refused, len);
}

#[cfg(test)]
thread_local! {
    /// Whether this thread's imports find their pointer refused, with nothing asked of rt, as
    /// RADV refuses a memfd's pages at the import (the tests').
    static POINTER_REFUSED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(all(test, target_os = "linux"))]
thread_local! {
    /// Whether this thread's imports find the host making no dma-buf of their pages (the tests').
    static HOST_MAKES_NO_DMA_BUF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Whether this thread's imports find rt refusing the dma-buf the host made of their pages,
    /// with nothing asked of rt (the tests').
    static DMA_BUF_REFUSED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The tests' refusal of the thread's pointers (`POINTER_REFUSED`), RADV's to a memfd's pages;
/// never outside the tests.
fn pointer_refused() -> Option<rt::Error> {
    #[cfg(test)]
    if POINTER_REFUSED.get() {
        return Some(rt::Error::Vulkan {
            call: "vkAllocateMemory",
            result: ash::vk::Result::ERROR_INVALID_EXTERNAL_HANDLE,
        });
    }
    None
}

/// rt's import of the host's pages that hold the `length` bytes at `first`, as a no-copy buffer's
/// is made and counted ([`import_pages`]), as a copy that drops the GPU's writes where `read_only`
/// (the module's docs, "Read-only memory"), known to `device` from now (`KnownBlocks::know`): the
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
    read_only: bool,
    then: impl FnOnce() + Send + 'static,
) -> Option<(rt::Block, u64)> {
    let (block, offset) = import_pages(device, sel, first, length, read_only, then)?;
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
             reach a no-copy buffer's bytes, where Metal's GPU drops a kernel's and a blit's \
             writes and faults a render pass's store ({READ_ONLY_CODE}), counted from here"
        );
    }
}

impl RtBuffer {
    /// The host address of its first byte, as `contents` answers it: NULL for a Private buffer,
    /// as Metal's, and for a deferred one whose memory is not made; a buffer in a dma-buf's block
    /// answers the host's mapping of its pages, and one in a copy's block the pages it copies
    /// (`KnownBlocks::contents_of`).
    pub(crate) fn host_contents(&self) -> HostPtr {
        // A deferred buffer, Private, whose memory is not made: none to answer.
        let Some(made) = self.rt.get() else {
            return HostPtr::NULL;
        };
        let known = self.device.known_blocks();
        let home = known.home_of(made.block());
        if let Some(at) = home.and_then(|first| first.checked_add(made.offset())) {
            return HostPtr::from_addr(at.to_usize());
        }
        if let Some(at) = made.contents() {
            return HostPtr::from_ptr(at.as_ptr());
        }
        let mapped = known.mapped.of(made.block());
        match mapped.and_then(|first| first.checked_add(made.offset())) {
            Some(at) => HostPtr::from_addr(at.to_usize()),
            None => HostPtr::NULL,
        }
    }

    /// Whether the GPU's work writes a copy of it apart from the CPU's: a `Managed` buffer of a
    /// device without unified memory, which rt gives a device copy.
    pub(crate) fn gpu_copy_apart(&self) -> bool {
        (self.options >> 4) & 0xf == MANAGED && !self.device.rt().has_unified_memory()
    }

    /// Its length, as `length` answers it: a deferred buffer's own, whose memory is not made.
    pub(crate) fn byte_length(&self) -> u64 {
        match (self.rt.get(), &self.deferred) {
            (Some(made), _) => made.length(),
            (None, Some(deferred)) => deferred.length,
            (None, None) => 0,
        }
    }
}

/// The bytes of a buffer of `length` bytes that a texture of `desc` over it from `offset`, rows
/// `bytes_per_row` apart, lies over: its image, each slice of a 2D array's a 128-byte multiple
/// apart, a texture buffer's from its offset's 16-byte boundary, as the Vulkan platform reads one.
/// For the recorder, which writes them as the buffer's.
fn texel_bytes(desc: &TextureDesc, offset: u64, bytes_per_row: u64, length: u64) -> Range<u64> {
    /// `MTLTextureType2DArray`.
    const TYPE_2D_ARRAY: u64 = 3;
    /// `MTLTextureTypeTextureBuffer`.
    const TYPE_TEXTURE_BUFFER: u64 = 9;
    let image = bytes_per_row.saturating_mul(desc.height.max(1));
    let (slices, pitch) = match desc.texture_type {
        TYPE_2D_ARRAY => (
            desc.array_length.max(1),
            image.checked_next_multiple_of(128).unwrap_or(u64::MAX),
        ),
        _ => (1, image),
    };
    let start = match desc.texture_type {
        TYPE_TEXTURE_BUFFER => offset - offset % 16,
        _ => offset,
    };
    let span = (pitch.saturating_mul(slices - 1)).saturating_add(image);
    start.min(length)..offset.saturating_add(span).min(length)
}

impl RtObj<RtBuffer> {
    /// The texture `sel` makes of `descriptor` over the buffer's bytes from `offset`, rows
    /// `bytes_per_row` apart (`super::texture::make_over`), which keeps what bytes of the buffer
    /// it lies over (`RtTexture::over`).
    fn texture(
        &self,
        sel: &'static str,
        descriptor: &RtObj<TextureFields>,
        offset: u64,
        bytes_per_row: u64,
    ) -> Option<RtObj<RtTexture>> {
        // A deferred buffer's memory first: nil where rt refuses it, counted.
        let made = self.made(Some(MemoryFor::Texture))?;
        let over = |d: &rt::TextureDescriptor, index: Option<u32>| {
            made.new_texture(d, index, offset, bytes_per_row)
        };
        let held = (self.options, texture::STORAGE_BITS);
        let diag = |shape: &Shape| Made::over_buffer(made, shape, offset, bytes_per_row);
        let (desc, copied) = (descriptor.desc(), self.copied.as_ref());
        let texture = texture::make_over(&self.device, sel, &desc, held, over, diag, copied);
        self.lies_under(texture.as_ref(), &desc, offset, bytes_per_row);
        texture
    }

    /// Notes that `texture`, where made, of `desc` from `offset`, rows `bytes_per_row` apart,
    /// lies over this buffer's bytes ([`texel_bytes`]).
    fn lies_under(
        &self,
        texture: Option<&RtObj<RtTexture>>,
        desc: &TextureDesc,
        offset: u64,
        bytes_per_row: u64,
    ) {
        if let Some(texture) = texture {
            let bytes = texel_bytes(desc, offset, bytes_per_row, self.byte_length());
            texture.set_over(RtObj::downgrade(self), bytes);
        }
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
        let recorded_call = crate::recorder::rec_call!("setPurgeableState:", self, state);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<u64> {
            if self.in_heap {
                return Ok(texture::NON_VOLATILE);
            }
            Ok(texture::set_purgeable(&self.purgeable, state))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Bits 4 to 7 of its resource options.
    fn storage_mode(&self) -> u64 {
        (self.options >> 4) & 0xf
    }
}

impl Buffer<RtPlatform> for RtObj<RtBuffer> {
    /// rt's: NULL for a Private buffer, as Metal's. A buffer in a dma-buf's block, which rt maps
    /// no bytes of, answers the host's mapping of its pages (`KnownBlocks::contents_of`, interim
    /// until rt takes that mapping), and one in a copy's block the pages it copies.
    fn contents(&self) -> HostPtr {
        self.host_contents()
    }

    /// Of a Managed buffer, rt's, which copies the range into its device copy at the next commit;
    /// a range past the buffer changes nothing, counted, where Metal's validation aborts. Of any
    /// other, nothing, as Metal. In a process that records, the recorder first writes each chunk
    /// of the range whose bytes differ from those the stream last recorded
    /// ([`crate::recorder::before_modify`]).
    fn did_modify_range(&self, on: OnFault, range: NsRange) -> RaiseResult<()> {
        let recorded_call = crate::recorder::rec_call!("didModifyRange:", self, range);
        crate::recorder::before_modify(&recorded_call, self, &range);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            let _ = on;
            if self.storage_mode() == MANAGED
                && let Err(error) = self.rt().did_modify_range(range.location, range.length)
            {
                rt_refused("didModifyRange:", &error);
            }
            Ok(())
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's: the Vulkan device address of its first byte (of a Managed buffer's device copy where
    /// it has one); 0 for a deferred buffer whose memory is not made (the module's docs,
    /// "Deferred memory").
    fn gpu_address(&self) -> u64 {
        self.rt.get().map_or(0, rt::Buffer::gpu_address)
    }

    /// The copy of host pages it lies over (`super::copied`), where it lies over one.
    fn escaped_copy(&self) -> Option<crate::core::platform::command::EscapedCopy> {
        let copied = self.copied.as_ref()?;
        Some(Arc::clone(copied) as crate::core::platform::command::EscapedCopy)
    }

    /// rt's; a deferred buffer's own, whose memory is not made.
    fn length(&self) -> u64 {
        self.byte_length()
    }

    /// A deferred buffer's memory, made now where it is not yet: `false` where rt refuses it,
    /// counted (the module's docs, "Deferred memory"). Any other buffer holds its memory.
    fn make_memory(&self, made_for: MemoryFor) -> bool {
        self.made(Some(made_for)).is_some()
    }

    /// Whether it holds its memory: a deferred buffer's once made (the module's docs, "Deferred
    /// memory").
    fn memory_made(&self) -> bool {
        self.deferred.is_none() || self.rt.get().is_some()
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
        let recorded_call = crate::recorder::rec_call!(
            "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:",
            self,
            descriptor,
            offset,
            bytes_per_row,
            bytes_per_image
        );
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<Option<RtObj<RtTexture>>> {
            let _ = bytes_per_image;
            let sel = "newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:";
            Ok(self.texture(sel, descriptor, offset, bytes_per_row))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
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
        let recorded_call = crate::recorder::rec_call!(
            "newTextureWithDescriptor:offset:bytesPerRow:",
            self,
            descriptor,
            offset,
            bytes_per_row
        );
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<Option<RtObj<RtTexture>>> {
            let _ = on;
            let sel = "newTextureWithDescriptor:offset:bytesPerRow:";
            Ok(self.texture(sel, descriptor, offset, bytes_per_row))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// A 2D texture of more than one level: rt's texture of the descriptor, at an index as
    /// `newTextureWithDescriptor:`'s, of the buffer's storage mode, which takes the bytes from
    /// `offset`, rows `bytes_per_row` apart, at its first use and every level from them
    /// (`super::tiled`); nil, counted, for rows Metal's validation stops (`tiled::bad_layout`).
    /// Any other is the texture `newTextureWithDescriptor:offset:bytesPerRow:` makes, which rt
    /// refuses for more than one level. Nil where rt refuses it, counted.
    fn new_tiled_texture(
        &self,
        descriptor: &RtObj<TextureFields>,
        offset: u64,
        bytes_per_row: u64,
    ) -> RaiseResult<Option<RtObj<RtTexture>>> {
        let recorded_call = crate::recorder::rec_call!(
            "newTiledTextureWithDescriptor:offset:bytesPerRow:",
            self,
            descriptor,
            offset,
            bytes_per_row
        );
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<Option<RtObj<RtTexture>>> {
            let sel = "newTiledTextureWithDescriptor:offset:bytesPerRow:";
            let desc = descriptor.desc();
            if desc.mipmap_level_count <= 1 || desc.texture_type != TEXTURE_TYPE_2D {
                return Ok(self.texture(sel, descriptor, offset, bytes_per_row));
            }
            let (rt_device, length) = (self.device.rt(), self.length());
            if let Some(why) = tiled::bad_layout(rt_device, &desc, length, offset, bytes_per_row) {
                return Ok(refused_metal(sel, &why));
            }
            // A deferred buffer's memory first, which the take reads: nil where rt refuses it,
            // counted.
            if self.made(Some(MemoryFor::Texture)).is_none() {
                return Ok(None);
            }
            let own =
                |d: &rt::TextureDescriptor, index: Option<u32>| rt_device.new_texture(d, index);
            let held = (self.options, texture::STORAGE_BITS);
            let diag = |shape: &Shape| Made::new(Call::Tiled, shape);
            let copied = self.copied.as_ref();
            let made = texture::make_over(&self.device, sel, &desc, held, own, diag, copied);
            if let Some(texture) = &made {
                let tiled = Tiled::new(rt_device, texture.rt(), self, offset, bytes_per_row);
                texture.set_tiled(tiled);
            }
            self.lies_under(made.as_ref(), &desc, offset, bytes_per_row);
            Ok(made)
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// An exportable buffer's memory as a new opaque fd (`super::export`); `Err` for any other.
    #[cfg(unix)]
    fn export_memory(&self) -> Result<ExportedMemory, String> {
        let exported = (self.exported.as_ref()).ok_or("the buffer was not made exportable")?;
        exported.export(rt::ExportHandle::OpaqueFd, self.rt().length())
    }

    /// A dma-buf buffer's memory as a new dma-buf (`super::export`); `Err` for any other.
    #[cfg(unix)]
    fn export_dma_buf(&self) -> Result<ExportedMemory, String> {
        let exported = (self.exported.as_ref()).ok_or("the buffer was not made exportable")?;
        exported.export(rt::ExportHandle::DmaBuf, self.rt().length())
    }

    /// rt's acquire of what another API wrote into an exportable buffer's block
    /// (`super::export`, "The acquire"); `Err` for any other buffer, which no other API writes.
    #[cfg(unix)]
    fn external_write_done(&self) -> Result<(), String> {
        let recorded_call = crate::recorder::rec_call!(crate::recorder::EXTERNAL_WRITE_DONE, self);
        let recorded_answer = crate::recorder::run(|| -> Result<(), String> {
            let exported = (self.exported.as_ref()).ok_or("the buffer was not made exportable")?;
            exported.acquire()
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// rt's texture over the buffer's bytes at the plane's offset, rows its bytes per row apart,
    /// as [`Self::new_texture`] makes one: the buffer stands for the surface. Of the surface's
    /// facts, Metal's answers: a plane past 0 of a surface without planes is Metal's refusal
    /// (`rt-refused-metal`), where Metal asserts, and a plane past a planar surface's last is nil,
    /// as Metal's. A format that reads beside its plane over a surface of two planes or more (a
    /// two-plane YCbCr format the Vulkan platform's runtime reads, a two-plane XR10 one:
    /// `crate::gpu::device::plane_facts_taken`) is unbuilt: nil ([`Unbuilt::IosurfaceTexture`]),
    /// as rt's texture over a buffer reads one plane. Without facts the texture is the buffer's.
    fn new_texture_over_surface_plane(
        &self,
        on: OnFault,
        descriptor: &RtObj<TextureFields>,
        plane: SurfacePlane,
    ) -> RaiseResult<Option<RtObj<RtTexture>>> {
        let recorded_call = crate::recorder::rec_call!(
            "newTextureWithDescriptor:iosurface:plane:",
            self,
            descriptor,
            plane
        );
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<Option<RtObj<RtTexture>>> {
            let _ = on;
            let sel = "newTextureWithDescriptor:iosurface:plane:";
            if let Some(facts) = plane.facts {
                if plane.plane != 0 && facts.plane_count == 0 {
                    let what = format!("plane {} of a surface without planes", plane.plane);
                    return Ok(texture::refused_metal(sel, &what));
                }
                if facts.plane_count != 0 && plane.plane >= facts.plane_count {
                    return Ok(None);
                }
                let raw = u32::try_from(descriptor.desc().pixel_format).ok();
                let pf = raw.and_then(|raw| metal2vulkan::metal::PixelFormat::try_from(raw).ok());
                let beside = pf.and_then(crate::gpu::device::plane_facts_taken);
                if beside.is_some() && facts.plane_count >= 2 {
                    refuse_create(Unbuilt::IosurfaceTexture, sel);
                    return Ok(None);
                }
            }
            Ok(self.texture(sel, descriptor, plane.offset, plane.bytes_per_row))
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
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
    use crate::core::platform::compute::{ComputeCmd, ComputeEncoder as _};
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

        // Through a descriptor, read-only, with no pin: over a copy of its own whose GPU writes are
        // dropped, which no in-place count counts; the deallocator takes the bytes and the length.
        let given = Arc::new(Mutex::new(None));
        let take = Arc::clone(&given);
        let read_only_before = counted(READ_ONLY_CODE);
        let copies = (counted(IMPORTS_READ_ONLY), counted(IMPORTS_COPIED));
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
        assert_eq!(counted(READ_ONLY_CODE), read_only_before);
        let copies_now = (counted(IMPORTS_READ_ONLY), counted(IMPORTS_COPIED));
        assert_eq!(copies_now, (copies.0 + 1, copies.1 + 1));
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

    /// Runs `cmds` in a blit encoder of a new command buffer of `queue`, and waits for it.
    fn blit_and_wait(
        queue: &<RtPlatform as crate::core::platform::Platform>::CommandQueue,
        cmds: Vec<BlitCmd<'_, RtPlatform>>,
    ) {
        blit_and_wait_of(queue, None, cmds);
    }

    /// [`blit_and_wait`], the command buffer of the process whose escaped copies are `escapes`
    /// where given (`reaches_escaped`).
    fn blit_and_wait_of(
        queue: &<RtPlatform as crate::core::platform::Platform>::CommandQueue,
        escapes: Option<&crate::core::platform::command::EscapedCopies>,
        cmds: Vec<BlitCmd<'_, RtPlatform>>,
    ) {
        let on = OnFault::Return;
        let commands = queue.command_buffer(on).expect("never a raise");
        let commands = commands.expect("a command buffer");
        if let Some(escapes) = escapes {
            commands.reaches_escaped(escapes);
        }
        let encoder = commands.blit_encoder(on).expect("never a raise");
        let encoder = encoder.expect("a blit encoder");
        for cmd in cmds {
            assert_eq!(encoder.encode(on, cmd), Ok(()));
        }
        assert_eq!(encoder.encode(on, BlitCmd::EndEncoding), Ok(()));
        assert_eq!(commands.commit(on), Ok(()));
        assert_eq!(commands.wait_until_completed(on), Ok(()));
    }

    /// The length of the deferred buffers the tests make: 128 KiB.
    const DEFERRED_LENGTH: u64 = 0x2_0000;

    /// The counts of deferred buffers made, made with their memory, made by a use itself, and
    /// gone unmade (the module's docs, "Deferred memory").
    fn deferred_counts() -> [u64; 4] {
        let codes = [
            DEFERRED.made,
            DEFERRED_MADE.made,
            DEFERRED_ON_USE_CODE,
            DEFERRED_GONE_UNMADE_CODE,
        ];
        codes.map(counted)
    }

    /// `device`'s deferred buffer of [`DEFERRED_LENGTH`] bytes.
    fn deferred(device: &RtObj<RtDevice>) -> RtObj<RtBuffer> {
        let made = device.new_deferred_buffer_of_length(OnFault::Return, DEFERRED_LENGTH, 0x20);
        made.expect("never a raise").expect("a deferred buffer")
    }

    /// On the GPU the test holds: a deferred buffer holds no memory until it is made, answering
    /// its length, NULL contents and GPU address 0; made, once, ahead of its first use, a read,
    /// it reads zeros, though a Private buffer of its length held other bytes just before it;
    /// counted as made ahead of its use, and one gone unmade counted so
    #[test]
    fn a_deferred_buffer_reads_zeros_before_any_write_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let shared = |options| {
            let made = device.new_buffer_of_length(on, DEFERRED_LENGTH, options);
            made.expect("never a raise").expect("a buffer")
        };
        let out = shared(0);
        let earlier = shared(0x20);
        let fill = BlitCmd::FillBuffer {
            buffer: &earlier,
            range: NsRange::new(0, DEFERRED_LENGTH),
            value: 0xab,
        };
        blit_and_wait(&queue, vec![fill]);
        drop(earlier);
        let before = deferred_counts();

        let buffer = deferred(&device);
        assert!(!buffer.memory_made());
        assert_eq!(
            (buffer.length(), buffer.gpu_address()),
            (DEFERRED_LENGTH, 0)
        );
        assert!(buffer.contents().is_null());
        assert!(buffer.make_memory(MemoryFor::Fill));
        assert!(buffer.memory_made());
        assert!(buffer.make_memory(MemoryFor::Fill), "made already");
        assert_ne!(buffer.gpu_address(), 0);
        let copy = BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(&buffer),
            source_offset: 0,
            destination_buffer: Some(&out),
            destination_offset: 0,
            size: DEFERRED_LENGTH,
        };
        blit_and_wait(&queue, vec![copy]);
        let read_out = read(out.contents(), DEFERRED_LENGTH.to_usize());
        assert!(read_out.iter().all(|&b| b == 0), "zeros");
        drop(deferred(&device));
        let after = deferred_counts();
        let counts: Vec<u64> = after.iter().zip(before).map(|(a, b)| a - b).collect();
        assert_eq!(
            counts,
            [2, 1, 0, 1],
            "made, with memory, on a use, gone unmade"
        );
    }

    /// On the GPU the test holds: a deferred buffer whose first use writes it, a copy of two
    /// planes' bytes into it that makes its memory there, reads those bytes after, its zeroing
    /// ahead of that write; counted as made by the use itself
    #[test]
    fn a_deferred_buffer_holds_the_planes_bytes_after_a_write_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let shared = || {
            let made = device.new_buffer_of_length(on, DEFERRED_LENGTH, 0);
            made.expect("never a raise").expect("a Shared buffer")
        };
        let (from, out) = (shared(), shared());
        let length = DEFERRED_LENGTH.to_usize();
        // A luma plane of two thirds of the bytes, then a chroma plane, each of its own bytes.
        let luma = DEFERRED_LENGTH / 3 * 2;
        let bytes: Vec<u8> = (0..length)
            .map(|i| match i < luma.to_usize() {
                true => u8::try_from(i % 251).expect("a byte"),
                false => u8::try_from(i % 239).expect("a byte") ^ 0x5a,
            })
            .collect();
        // SAFETY: a Shared buffer's contents are mapped, `length` bytes long.
        unsafe {
            (from.contents().as_ptr::<u8>()).copy_from_nonoverlapping(bytes.as_ptr(), length)
        };
        let before = deferred_counts();

        let buffer = deferred(&device);
        assert!(!buffer.memory_made());
        let plane = |at: u64, size: u64| BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(&from),
            source_offset: at,
            destination_buffer: Some(&buffer),
            destination_offset: at,
            size,
        };
        blit_and_wait(
            &queue,
            vec![plane(0, luma), plane(luma, DEFERRED_LENGTH - luma)],
        );
        assert!(buffer.memory_made(), "made by the write");
        let copy = BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(&buffer),
            source_offset: 0,
            destination_buffer: Some(&out),
            destination_offset: 0,
            size: DEFERRED_LENGTH,
        };
        blit_and_wait(&queue, vec![copy]);
        assert!(read(out.contents(), length) == bytes, "the planes' bytes");
        let after = deferred_counts();
        let counts: Vec<u64> = after.iter().zip(before).map(|(a, b)| a - b).collect();
        assert_eq!(
            counts,
            [1, 1, 1, 0],
            "made, with memory, on a use, gone unmade"
        );
    }

    /// On the GPU the test holds: a Private buffer pinned in a known block is zeroed at each
    /// placement, rt saying nothing of the block's bytes: the block's first, one over an earlier
    /// one's written bytes, one straddling them, and one past every earlier placement all read
    /// zero
    #[test]
    fn pinned_private_buffers_are_zeroed_at_each_placement_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let block = (device.rt()).take_block(rt::MemoryClass::Private, 0x10000);
        let block = block.expect("a Private block");
        let known = device.known_blocks().clone();
        known.know(block.clone());
        let queue = device.new_command_queue().expect("a command queue");
        let out = device
            .new_buffer_of_length(on, 64, 0)
            .expect("never a raise");
        let out = out.expect("a Shared buffer");
        let private = |at: u64| {
            let made = device.new_buffer(pinned(block.gpu_address + at, 0x20, None));
            let made = made
                .expect("never a raise")
                .expect("a Private buffer in its block");
            assert_eq!(made.gpu_address(), block.gpu_address + at);
            made
        };
        let read_back = |buffer: &RtObj<RtBuffer>| {
            let copy = BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(buffer),
                source_offset: 0,
                destination_buffer: Some(&out),
                destination_offset: 0,
                size: 64,
            };
            blit_and_wait(&queue, vec![copy]);
            read(out.contents(), 64)
        };

        // The block's first placement.
        let first = private(0x100);
        assert_eq!(read_back(&first), [0; 64]);
        let fill = BlitCmd::FillBuffer {
            buffer: &first,
            range: NsRange::new(0, 64),
            value: 0xab,
        };
        blit_and_wait(&queue, vec![fill]);
        assert_eq!(read_back(&first), [0xab; 64]);
        drop(first);

        // Over the first one's bytes.
        let again = private(0x100);
        assert_eq!(read_back(&again), [0; 64]);
        drop(again);
        // Straddling them: some of its bytes were the first one's.
        let straddling = private(0x120);
        assert_eq!(read_back(&straddling), [0; 64]);
        // Past every placement.
        let past = private(0x1000);
        assert_eq!(read_back(&past), [0; 64]);
        drop((straddling, past));
        assert!(known.forget(block.id).is_some());
    }

    /// On the GPU the test holds: a scratch buffer is a Shared buffer of the length asked,
    /// answering its options, its bytes mapped and none of them zeroed, where
    /// `new_buffer_of_length`'s all read zero, zeroed ([`own`]); a Private one is made as
    /// `new_buffer_of_length` makes it, so; Metal's refusals nil.
    #[test]
    fn scratch_buffers_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let zeroed = || device.rt().counters().metrics().bytes_zeroed;
        let scratch = |length, options| {
            (device.new_scratch_buffer_of_length(on, length, options)).expect("never a raise")
        };

        let before = zeroed();
        let buffer = scratch(100_000, 0).expect("a Shared scratch buffer");
        assert_eq!(
            (
                buffer.length(),
                buffer.storage_mode(),
                buffer.resource_options()
            ),
            (100_000, 0, 0x200)
        );
        assert_ne!(buffer.contents(), HostPtr::NULL);
        assert_ne!(buffer.gpu_address(), 0);
        assert_eq!(zeroed(), before, "a scratch buffer's bytes are not zeroed");

        let before = zeroed();
        let plain = (device.new_buffer_of_length(on, 100_000, 0)).expect("never a raise");
        let plain = plain.expect("a Shared buffer");
        assert!(
            zeroed() >= before + 100_000,
            "a new buffer's bytes are zeros"
        );
        assert!(read(plain.contents(), 100_000).iter().all(|&b| b == 0));

        let before = zeroed();
        let private = scratch(4096, 0x20).expect("a Private scratch buffer");
        assert_eq!(
            (private.storage_mode(), private.contents()),
            (2, HostPtr::NULL)
        );
        assert!(
            zeroed() >= before + 4096,
            "a Private one is made as any, its bytes zeros"
        );

        assert!(scratch(0, 0).is_none() && scratch(64, 0x30).is_none());
    }

    /// Textures over the planes of a two-plane surface a Shared buffer stands for, on the GPU the
    /// test holds: each reads its plane's texels at the plane's offset and rows, the CPU's writes
    /// after it was made among them. Of the surface's facts, Metal's answers: a plane past 0 of a
    /// surface without planes is Metal's refusal, counted, and a plane past the last nil; a
    /// two-plane YCbCr format over plane 0 is unbuilt, counted; without facts the texture is the
    /// buffer's.
    #[test]
    fn textures_over_surface_planes_on_vulkan() {
        use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
        use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
        use crate::core::platform::types::{CommandBufferStatus, MtlOrigin, MtlSize};
        use crate::core::platform::values::SurfaceFacts;

        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let of = |length| {
            let made = device.new_buffer_of_length(on, length, 0);
            made.expect("never a raise").expect("a Shared buffer")
        };
        // An 8 x 4 'x420' surface: plane 0, of 16-bit luma, 32 bytes in, and plane 1, of 2 x
        // 16-bit chroma, after it, rows 32 bytes apart.
        let surface = of(4096);
        let facts = SurfaceFacts {
            pixel_format: 0x7834_3230,
            plane_count: 2,
            width: 8,
            height: 4,
            chroma_offset: 160,
            chroma_bytes_per_row: 32,
            ycbcr_matrix: None,
        };
        let plane = |plane, offset, facts| SurfacePlane {
            plane,
            offset,
            bytes_per_row: 32,
            planar: true,
            facts,
        };
        let descriptor = |pixel_format, width, height| {
            platform.texture_descriptor(&TextureDesc {
                pixel_format,
                width,
                height,
                usage: 0x1,
                ..TextureDesc::default()
            })
        };
        let over = |desc: RtObj<TextureFields>, at: SurfacePlane| {
            let made = surface.new_texture_over_surface_plane(on, &desc, at);
            made.expect("never a raise")
        };
        // R16Unorm and RG16Unorm.
        let luma = over(descriptor(20, 8, 4), plane(0, 32, Some(facts)));
        let chroma = over(descriptor(60, 4, 2), plane(1, 160, Some(facts)));
        let (luma, chroma) = (
            luma.expect("a texture over plane 0"),
            chroma.expect("a texture over plane 1"),
        );

        // The planes' texels as the GPU reads them, each row's 16 bytes packed.
        let out = of(96);
        let gpu_texels = || {
            let made = queue.command_buffer(on).expect("never a raise");
            let made = made.expect("a command buffer");
            let encoder = made.blit_encoder(on).expect("never a raise");
            let encoder = encoder.expect("a blit encoder");
            for (texture, (width, height), at) in [(&luma, (8, 4), 0), (&chroma, (4, 2), 64)] {
                let cmd = BlitCmd::CopyTextureToBuffer {
                    source_texture: texture,
                    source_slice: 0,
                    source_level: 0,
                    source_origin: MtlOrigin::default(),
                    source_size: MtlSize::new(width, height, 1),
                    destination_buffer: &out,
                    destination_offset: at,
                    destination_bytes_per_row: 16,
                    destination_bytes_per_image: 16 * height,
                };
                assert_eq!(encoder.encode(on, cmd), Ok(()));
            }
            assert_eq!(encoder.encode(on, BlitCmd::EndEncoding), Ok(()));
            assert_eq!(made.commit(on), Ok(()));
            assert_eq!(made.wait_until_completed(on), Ok(()));
            assert_eq!(made.status(), CommandBufferStatus::COMPLETED);
            read(out.contents(), 96)
        };
        // Writes the surface's bytes from `seed`: its planes' texels, each row's packed.
        let fill = |seed: u8| {
            let bytes: Vec<u8> = (0..4096_usize)
                .map(|i| u8::try_from(i % 251).expect("a byte").wrapping_add(seed))
                .collect();
            // SAFETY: a Shared buffer's contents are mapped, 4096 bytes long.
            unsafe {
                let at = surface.contents().as_ptr::<u8>();
                at.copy_from_nonoverlapping(bytes.as_ptr(), bytes.len());
            }
            let rows = |offset: usize, count: usize| -> Vec<u8> {
                (0..count)
                    .flat_map(|row| bytes[offset + 32 * row..][..16].to_vec())
                    .collect()
            };
            [rows(32, 4), rows(160, 2)].concat()
        };
        let texels = fill(3);
        assert_eq!(gpu_texels(), texels);
        let texels = fill(101);
        assert_eq!(gpu_texels(), texels);

        // Surface textures over a Shared buffer of rt's own are in place: no copy is counted.
        let diag = crate::rt_platform::diag::surface_texture::<RtObj<RtTexture>>;
        let (copies, copy_bytes) = (
            crate::rt_platform::diag::SURFACE_PLANE_COPIES_CODE,
            crate::rt_platform::diag::SURFACE_PLANE_COPY_BYTES_CODE,
        );
        diag(Some(&luma), 0, (0x5a00_0001, 0), (0x1, 128), "over-buffer");
        diag(Some(&chroma), 0, (0x5a00_0001, 1), (0x1, 64), "over-buffer");
        let before = (counted(copies), counted(copy_bytes));
        assert_eq!(gpu_texels(), texels);
        assert_eq!((counted(copies), counted(copy_bytes)), before);

        // Over a buffer in an imported block, rt copies a texture into an image of its own at each
        // submission that uses it: counted once a committed command buffer where it is a surface
        // texture, with its rows' bytes, 32 apart times 4; not before the decoder noted it as one.
        let layout = Layout::from_size_align(0x10000, 0x10000).expect("a layout");
        // SAFETY: the layout's size is not 0.
        let memory = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!memory.is_null());
        for i in 0..4096 {
            // SAFETY: within the allocation.
            unsafe { memory.add(i).write(u8::try_from(i % 239).expect("a byte")) };
        }
        let at = memory.expose_provenance();
        let free: NoCopyDeallocator = Box::new(move || {
            // SAFETY: allocated above with `layout`, and freed once, as rt lets the import go.
            unsafe { std::alloc::dealloc(std::ptr::with_exposed_provenance_mut(at), layout) }
        });
        // SAFETY: the memory stays allocated until the deallocator runs.
        let bytes = unsafe { HostRange::new(HostPtr::from_ptr(memory), 4096) };
        let made = device.new_buffer_over_host_memory(on, bytes, 0, Some(free));
        let imported = made.expect("never a raise").expect("a no-copy buffer");
        let made = imported.new_texture_over_surface_plane(
            on,
            &descriptor(20, 8, 4),
            plane(0, 32, Some(facts)),
        );
        let luma = made
            .expect("never a raise")
            .expect("a texture over an import's plane 0");
        let copied = || {
            let made = queue.command_buffer(on).expect("never a raise");
            let made = made.expect("a command buffer");
            let encoder = made.blit_encoder(on).expect("never a raise");
            let encoder = encoder.expect("a blit encoder");
            // Twice, which counts once.
            for _ in 0..2 {
                let cmd = BlitCmd::CopyTextureToBuffer {
                    source_texture: &luma,
                    source_slice: 0,
                    source_level: 0,
                    source_origin: MtlOrigin::default(),
                    source_size: MtlSize::new(8, 4, 1),
                    destination_buffer: &out,
                    destination_offset: 0,
                    destination_bytes_per_row: 16,
                    destination_bytes_per_image: 64,
                };
                assert_eq!(encoder.encode(on, cmd), Ok(()));
            }
            assert_eq!(encoder.encode(on, BlitCmd::EndEncoding), Ok(()));
            assert_eq!(made.commit(on), Ok(()));
            assert_eq!(made.wait_until_completed(on), Ok(()));
            assert_eq!(made.status(), CommandBufferStatus::COMPLETED);
            read(out.contents(), 64)
        };
        let rows: Vec<u8> = (0..4_usize)
            .flat_map(|row| {
                (32 + 32 * row..48 + 32 * row).map(|i| u8::try_from(i % 239).expect("a byte"))
            })
            .collect();
        assert_eq!(copied(), rows);
        assert_eq!((counted(copies), counted(copy_bytes)), before);
        diag(Some(&luma), 0, (0x5a00_0002, 0), (0x1, 128), "over-buffer");
        assert_eq!(copied(), rows);
        assert_eq!(
            (counted(copies), counted(copy_bytes)),
            (before.0 + 1, before.1 + 128)
        );

        let (metal, unbuilt) = (RtRefusal::Refused.code(), Unbuilt::IosurfaceTexture.code());
        let before = (counted(metal), counted(unbuilt));
        let lone = SurfaceFacts {
            plane_count: 0,
            chroma_offset: 0,
            chroma_bytes_per_row: 0,
            ..facts
        };
        let lone = SurfacePlane {
            planar: false,
            ..plane(1, 0, Some(lone))
        };
        assert!(over(descriptor(20, 8, 4), lone).is_none());
        assert_eq!(counted(metal), before.0 + 1);
        assert!(over(descriptor(20, 8, 4), plane(2, 160, Some(facts))).is_none());
        // YCbCr8_420_2P.
        assert!(over(descriptor(500, 8, 4), plane(0, 32, Some(facts))).is_none());
        assert_eq!(
            (counted(metal), counted(unbuilt)),
            (before.0 + 1, before.1 + 1)
        );
        assert!(over(descriptor(20, 8, 4), plane(0, 32, None)).is_some());
    }

    /// On Linux, on the GPU the test holds, where the GPU takes no pointer to a no-copy buffer's
    /// pages (`POINTER_REFUSED`, as RADV refuses a memfd's): over pages with no file behind them,
    /// the buffer is made over a copy of them, counted as refused and as copied, its `contents`
    /// the pages, a deallocator run once rt freed the copy. Over pages in more runs of file pages
    /// than one dma-buf takes, it is a range of rt's import of several dma-bufs of them, counted,
    /// where rt binds dma-bufs behind one buffer and imports one of a page; else it is made over a
    /// copy, counted as too large, as copied and as no several dma-bufs imported. Over pages with
    /// a file behind them, the buffer is a
    /// range of rt's import of a dma-buf of them, counted, its `contents` its bytes, which the
    /// CPU's writes and a blit's reach both ways; a known import's block holds a no-copy buffer
    /// pinned over its bytes, with no import of its own, and a Managed one over them too, counted,
    /// its `contents` the bytes; each release runs once rt freed its import, and the device then
    /// forgets its mapping. Where the device imports no dma-buf, the buffer is made over a copy,
    /// counted, and the test says so.
    #[cfg(target_os = "linux")]
    #[test]
    fn dma_bufs_of_pages_on_vulkan() {
        use crate::core::num::ToU64 as _;
        use crate::core::platform::os::vm::HostVm as _;

        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        POINTER_REFUSED.set(true);
        let page = device.facts().page_size;
        let flag = || {
            let ran = Arc::new(AtomicBool::new(false));
            let run = Arc::clone(&ran);
            (ran, move || run.store(true, Ordering::Release))
        };
        let codes = [
            IMPORTS_POINTER,
            IMPORTS_DMA_BUF,
            IMPORTS_REFUSED,
            IMPORTS_TOO_LARGE,
            IMPORTS_COPIED,
            IMPORTS_DMA_BUFS,
            IMPORTS_DMA_BUFS_REFUSED,
        ];
        let counts = || codes.map(counted);
        let plus = |before: [u64; 7], counted: &[usize]| {
            let mut after = before;
            counted.iter().for_each(|&i| after[i] += 1);
            after
        };

        // Pages with no file behind them.
        let layout = Layout::from_size_align(0x4000, 0x1000).expect("a layout");
        // SAFETY: the layout's size is not 0.
        let memory = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!memory.is_null());
        let before = counts();
        let (freed, deallocator) = flag();
        // SAFETY: the memory is never freed.
        let bytes = unsafe { HostRange::new(HostPtr::from_ptr(memory), 0x1000) };
        let made = device.new_buffer_over_host_memory(on, bytes, 0, Some(Box::new(deallocator)));
        let copy = made.expect("never a raise").expect("a buffer over a copy");
        assert_eq!(counts(), plus(before, &[2, 4]));
        assert_eq!(copy.contents(), HostPtr::from_ptr(memory));
        drop(copy);
        assert!(
            waits_for(|| freed.load(Ordering::Acquire)),
            "its deallocator ran"
        );

        // Pages with a file behind them, the test's own, which the process ends with: 0x11s, 16
        // bytes in. Whether the host makes a dma-buf of them (udmabuf).
        let pages = crate::os::allocate_aliasable(4 * page).expect("pages of a file");
        let at = pages.wrapping_add(16);
        // SAFETY: fresh pages of the test's own, which nothing uses yet.
        unsafe { at.as_ptr::<u8>().write_bytes(0x11, 128) };
        let first = NonNull::new(pages.as_ptr::<u8>()).expect("mapped pages");
        let lent = crate::os::dma_buf_of_host(first, page, &[(0, page)]).is_ok();
        // Whether rt binds dma-bufs behind one buffer here, and imports one of a page.
        let one = crate::os::dma_buf_of_host(first, page, &[(0, page)]);
        let imports =
            |one: crate::os::LentDmaBuf| device.rt().import_dma_buf(one.fd, one.size, None);
        let takes =
            device.rt().dma_buf_granule().is_some() && one.is_ok_and(|one| imports(one).is_ok());

        // Pages in more runs than one dma-buf takes (udmabuf's `list_limit` as the host has it):
        // a view of every other page of a file. Where it takes more runs than the test makes, or
        // the host makes no dma-buf, unchecked.
        let limit = std::fs::read_to_string("/sys/module/udmabuf/parameters/list_limit");
        let limit = limit
            .ok()
            .and_then(|limit| limit.trim().parse::<u64>().ok());
        match limit.filter(|&limit| lent && limit < 4096) {
            Some(limit) => {
                let n = limit + 1;
                let ram = crate::os::allocate_aliasable(2 * n * page).expect("pages of a file");
                let runs: Vec<_> = (0..n)
                    .map(|i| (ram.wrapping_add(2 * i * page), page))
                    .collect();
                let vm = crate::os::OsVm::new();
                let view = vm.allocate_anywhere(n * page).expect("a range for a view");
                assert!(vm.remap_runs_shared_overwrite(view, &runs).is_ok());
                let before = counts();
                // SAFETY: the view and its pages are the test's own, never freed.
                let bytes = unsafe { HostRange::new(view, n * page) };
                let made = device.new_buffer_over_host_memory(on, bytes, 0, None);
                let buffer = made
                    .expect("never a raise")
                    .expect("a buffer over the view");
                let several = device.known_blocks().is_dma_buf(buffer.rt().block());
                assert_eq!(several, takes, "several dma-bufs where rt takes them");
                let made = if takes { &[1, 5][..] } else { &[3, 4, 6] };
                assert_eq!(counts(), plus(before, made));
                assert_eq!(buffer.contents(), view);
                eprintln!("a view of {n} runs: imported as several dma-bufs {several}");
            }
            None => eprintln!("udmabuf takes {limit:?} runs, dma-bufs {lent}: unchecked past it"),
        }

        // Over the pages with a file behind them.
        let before = counts();
        let (freed, deallocator) = flag();
        // SAFETY: the pages are the test's own, never freed.
        let bytes = unsafe { HostRange::new(at, 1000) };
        let made = device.new_buffer_over_host_memory(on, bytes, 0, Some(Box::new(deallocator)));
        let buffer = made
            .expect("never a raise")
            .expect("a buffer over the pages");
        if !device.known_blocks().is_dma_buf(buffer.rt().block()) {
            assert_eq!(counts(), plus(before, &[2, 4]));
            eprintln!("no dma-buf imported here: made over a copy, counted; the rest unchecked");
            return;
        }
        assert_eq!(counts(), plus(before, &[1]));
        assert_eq!((buffer.contents(), buffer.length()), (at, 1000));
        let id = buffer.rt().block();
        assert!(device.known_blocks().is_dma_buf(id));
        assert_eq!(read(buffer.contents(), 128), [0x11; 128]);

        // A blit reads the CPU's bytes through the import, and writes the pages the CPU reads.
        let of = |length, options| device.new_buffer_of_length(on, length, options);
        let of = |length| of(length, 0).expect("never a raise").expect("a buffer");
        let (out, source) = (of(64), of(64));
        // SAFETY: the Shared buffer's 64 bytes, mapped while it lives.
        unsafe { source.contents().as_ptr::<u8>().write_bytes(0x77, 64) };
        let queue = device.new_command_queue().expect("a command queue");
        let commands = queue.command_buffer(on).expect("never a raise");
        let commands = commands.expect("a command buffer");
        let encoder = commands.blit_encoder(on).expect("never a raise");
        let encoder = encoder.expect("a blit encoder");
        for (from, to, offset) in [(&buffer, &out, 0), (&source, &buffer, 64)] {
            let copy = BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(from),
                source_offset: 0,
                destination_buffer: Some(to),
                destination_offset: offset,
                size: 64,
            };
            assert_eq!(encoder.encode(on, copy), Ok(()));
        }
        assert_eq!(encoder.encode(on, BlitCmd::EndEncoding), Ok(()));
        assert_eq!(commands.commit(on), Ok(()));
        assert_eq!(commands.wait_until_completed(on), Ok(()));
        assert_eq!(read(out.contents(), 64), [0x11; 64]);
        assert_eq!(read(at.wrapping_add(64), 64), [0x77; 64]);
        drop((encoder, commands, buffer));
        let ran = || freed.load(Ordering::Acquire);
        assert!(waits_for(ran), "the deallocator ran");
        assert!(!device.known_blocks().is_dma_buf(id));

        // A known import of the pages: a no-copy buffer pinned over its bytes is a range of its
        // block, with no import of its own; so is a Managed one, counted.
        let at = pages.wrapping_add(2 * page + 32);
        let (released, then) = flag();
        let known = import_known(&device, "a test's import", at, 256, false, then);
        let (block, offset) = known.expect("a known import, as the buffer's above");
        let mapped = device.known_blocks().contents_of(&block);
        assert_eq!(block.contents, None);
        assert_eq!(mapped.map(|first| first + offset), Some(at.addr().to_u64()));
        let (pin, imports) = (block.gpu_address + offset, counted(IMPORTS_CODE));
        // SAFETY: the pages are the test's own, never freed.
        let bytes = unsafe { HostRange::new(at, 64) };
        let made = device.new_buffer(pinned(pin, 0, Some(bytes)));
        let made = made.expect("never a raise").expect("a buffer in the block");
        assert_eq!((made.gpu_address(), made.contents()), (pin, at));
        let managed = counted(MANAGED_OVER_DMA_BUF);
        // SAFETY: as above.
        let bytes = unsafe { HostRange::new(at, 64) };
        let over = device.new_buffer(pinned(pin, 0x10, Some(bytes)));
        let over = over
            .expect("never a raise")
            .expect("a Managed buffer in the block");
        assert_eq!(over.contents(), at);
        assert_eq!(counted(MANAGED_OVER_DMA_BUF), managed + 1);
        assert_eq!(counted(IMPORTS_CODE), imports);
        drop((over, made));
        let id = block.id;
        assert!(device.known_blocks().forget(id).is_some());
        drop(block);
        let ran = || released.load(Ordering::Acquire);
        assert!(waits_for(ran), "the known import's release ran");
        assert!(!device.known_blocks().is_dma_buf(id));
    }

    /// On Linux, on the GPU the test holds, a no-copy buffer over a view of two runs of a file's
    /// pages, out of order, which the library maps through a dma-buf where udmabuf is the
    /// process's ("Pages no pin takes"): its pointer never goes to rt while a page of it lies in
    /// that mapping. It is a range of rt's import of a dma-buf of the pages where the GPU takes
    /// that (RADV's), and else of rt's import of its pointer once the view is mapped from its
    /// files (NVIDIA's driver, which refused the dma-buf), counted, as refused by rt or as none
    /// made by the host, and not as too large for one dma-buf; its bytes are the pages the view
    /// shows, which a blit reads and writes. The device then makes buffers, and imports a run of
    /// pages, as before.
    #[cfg(target_os = "linux")]
    #[test]
    fn views_of_two_runs_on_vulkan() {
        use crate::core::num::ToUsize as _;
        use crate::core::platform::os::vm::HostVm as _;

        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let page = device.facts().page_size;
        let codes = [
            IMPORTS_POINTER,
            IMPORTS_DMA_BUF,
            IMPORTS_REFUSED,
            IMPORTS_TOO_LARGE,
            IMPORTS_FROM_FILES,
            IMPORTS_FROM_FILES_TOO_LARGE,
            IMPORTS_FROM_FILES_DMA_BUF_REFUSED,
            IMPORTS_FROM_FILES_NO_DMA_BUF,
        ];
        let counts = || codes.map(counted);

        // Two pages of a file, 0x22s then 0x33s, and a view of them, the second first.
        let ram = crate::os::allocate_aliasable(2 * page).expect("pages of a file");
        // SAFETY: fresh pages of the test's own, which nothing uses yet.
        unsafe {
            ram.as_ptr::<u8>().write_bytes(0x22, page.to_usize());
            (ram.wrapping_add(page).as_ptr::<u8>()).write_bytes(0x33, page.to_usize());
        }
        let vm = crate::os::OsVm::new();
        let view = vm.allocate_anywhere(2 * page).expect("a range for a view");
        let runs = [(ram.wrapping_add(page), page), (ram, page)];
        assert!(vm.remap_runs_shared_overwrite(view, &runs).is_ok());
        let first = NonNull::new(view.as_ptr::<u8>()).expect("a mapped view");
        let through = crate::os::through_dma_bufs(first, 2 * page);

        let before = counts();
        let freed = Arc::new(AtomicBool::new(false));
        let free = Arc::clone(&freed);
        let deallocator = move || free.store(true, Ordering::Release);
        // SAFETY: the view and its pages are the test's own, never freed.
        let bytes = unsafe { HostRange::new(view, 2 * page) };
        let made = device.new_buffer_over_host_memory(on, bytes, 0, Some(Box::new(deallocator)));
        let buffer = made
            .expect("never a raise")
            .expect("a buffer over the view");
        let after = counts();
        let delta: [u64; 8] = std::array::from_fn(|i| after[i] - before[i]);
        let as_dma_buf = delta == [0, 1, 0, 0, 0, 0, 0, 0];
        let from_files = delta == [1, 0, 0, 0, 1, 0, 1, 0] || delta == [1, 0, 0, 0, 1, 0, 0, 1];
        let by_pointer = delta == [1, 0, 0, 0, 0, 0, 0, 0];
        assert!(
            as_dma_buf || (through && from_files) || (!through && by_pointer),
            "imports {delta:?}, the view through a dma-buf {through}"
        );
        let id = buffer.rt().block();
        assert_eq!(device.known_blocks().is_dma_buf(id), as_dma_buf);
        if through && !as_dma_buf {
            assert!(
                !crate::os::through_dma_bufs(first, 2 * page),
                "mapped from its files"
            );
        }
        eprintln!(
            "the view through a dma-buf {through}: imported as {}",
            match (as_dma_buf, from_files, delta[6] == 1) {
                (true, ..) => "a dma-buf",
                (false, true, true) => "its pointer from its files, rt refusing the dma-buf",
                (false, true, false) => "its pointer from its files, no dma-buf made",
                (false, false, _) => "its pointer",
            }
        );
        assert_eq!((buffer.contents(), buffer.length()), (view, 2 * page));
        assert_eq!(read(view, 16), [0x33; 16]);
        assert_eq!(read(view.wrapping_add(page), 16), [0x22; 16]);

        // Made after the import: buffers of the device's own, and an import of one run of pages.
        let of = |length, options| device.new_buffer_of_length(on, length, options);
        let of = |length| of(length, 0).expect("never a raise").expect("a buffer");
        let (out, source) = (of(64), of(64));
        // SAFETY: the Shared buffer's 64 bytes, mapped while it lives.
        unsafe { source.contents().as_ptr::<u8>().write_bytes(0x77, 64) };
        let run = crate::os::allocate_aliasable(page).expect("a page of a file");
        // SAFETY: the page is the test's own, never freed.
        let bytes = unsafe { HostRange::new(run, page) };
        let one = device.new_buffer_over_host_memory(on, bytes, 0, None);
        let one = one.expect("never a raise").expect("a buffer over one run");

        // A blit reads the view's pages through the import, and writes the pages it shows: its
        // second page, the file's first.
        let queue = device.new_command_queue().expect("a command queue");
        let commands = queue.command_buffer(on).expect("never a raise");
        let commands = commands.expect("a command buffer");
        let encoder = commands.blit_encoder(on).expect("never a raise");
        let encoder = encoder.expect("a blit encoder");
        let copies = [
            (&buffer, &out, 0),
            (&source, &buffer, page),
            (&source, &one, 0),
        ];
        for (from, to, offset) in copies {
            let copy = BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(from),
                source_offset: 0,
                destination_buffer: Some(to),
                destination_offset: offset,
                size: 64,
            };
            assert_eq!(encoder.encode(on, copy), Ok(()));
        }
        assert_eq!(encoder.encode(on, BlitCmd::EndEncoding), Ok(()));
        assert_eq!(commands.commit(on), Ok(()));
        assert_eq!(commands.wait_until_completed(on), Ok(()));
        assert_eq!(read(out.contents(), 64), [0x33; 64]);
        assert_eq!(read(ram, 64), [0x77; 64]);
        assert_eq!(read(run, 64), [0x77; 64]);
        drop((encoder, commands, buffer, one));
        let ran = || freed.load(Ordering::Acquire);
        assert!(waits_for(ran), "the deallocator ran");
        assert!(!device.known_blocks().is_dma_buf(id));
    }

    /// On Linux, on the GPU the test holds, a no-copy buffer over a view of more runs of a file's
    /// pages than one dma-buf takes (udmabuf's `list_limit` as the host has it), every other page
    /// of the file ("Pages no pin takes"): made by the view's pointer where the GPU takes it; else
    /// as several dma-bufs of the pages where rt takes them, counted, its bytes the pages, which a
    /// blit reads across every cut and writes across the first; else by the view's pointer once
    /// the view is mapped from its files where it lay in dma-bufs' mappings, counted as from its
    /// files, too large for one dma-buf and as no several dma-bufs imported; else made over a
    /// copy, counted as too large, copied and as no several dma-bufs imported. Its `contents` are
    /// the view, whose last page is the file's last run. Where the host makes no dma-buf, or
    /// udmabuf takes more runs than the test makes, unchecked.
    #[cfg(target_os = "linux")]
    #[test]
    fn views_past_one_dma_buf_on_vulkan() {
        use crate::core::num::ToUsize as _;
        use crate::core::platform::os::vm::HostVm as _;

        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let page = device.facts().page_size;
        let codes = [
            IMPORTS_POINTER,
            IMPORTS_FROM_FILES,
            IMPORTS_FROM_FILES_TOO_LARGE,
            IMPORTS_TOO_LARGE,
            IMPORTS_REFUSED,
            IMPORTS_COPIED,
            IMPORTS_FROM_FILES_DMA_BUF_REFUSED,
            IMPORTS_FROM_FILES_NO_DMA_BUF,
            IMPORTS_DMA_BUF,
            IMPORTS_DMA_BUFS,
            IMPORTS_DMA_BUFS_REFUSED,
        ];
        let counts = || codes.map(counted);
        let one = crate::os::allocate_aliasable(page).expect("a page of a file");
        let first = NonNull::new(one.as_ptr::<u8>()).expect("a mapped page");
        let lent = crate::os::dma_buf_of_host(first, page, &[(0, page)]).is_ok();
        let limit = std::fs::read_to_string("/sys/module/udmabuf/parameters/list_limit");
        let limit = limit
            .ok()
            .and_then(|limit| limit.trim().parse::<u64>().ok());
        let Some(n) = limit.filter(|&limit| lent && limit < 4096).map(|l| l + 1) else {
            eprintln!("udmabuf takes {limit:?} runs, dma-bufs {lent}: unchecked");
            return;
        };

        // A view of every other page of a file, the last of them 0x44s and each before it the
        // byte its place in the view gives ([`run_byte`]).
        let ram = crate::os::allocate_aliasable(2 * n * page).expect("pages of a file");
        let last = ram.wrapping_add(2 * (n - 1) * page);
        let run_byte = |i: u64| (i % 199 + 1).to_le_bytes()[0];
        for i in 0..n - 1 {
            let run = ram.wrapping_add(2 * i * page).as_ptr::<u8>();
            // SAFETY: a fresh page of the test's own, which nothing uses yet.
            unsafe { run.write_bytes(run_byte(i), page.to_usize()) };
        }
        // SAFETY: a fresh page of the test's own, which nothing uses yet.
        unsafe { last.as_ptr::<u8>().write_bytes(0x44, page.to_usize()) };
        let runs: Vec<_> = (0..n)
            .map(|i| (ram.wrapping_add(2 * i * page), page))
            .collect();
        let vm = crate::os::OsVm::new();
        let view = vm.allocate_anywhere(n * page).expect("a range for a view");
        assert!(vm.remap_runs_shared_overwrite(view, &runs).is_ok());
        let at = NonNull::new(view.as_ptr::<u8>()).expect("a mapped view");
        let through = crate::os::through_dma_bufs(at, n * page);

        let before = counts();
        // SAFETY: the view and its pages are the test's own, never freed.
        let bytes = unsafe { HostRange::new(view, n * page) };
        let made = device.new_buffer_over_host_memory(on, bytes, 0, None);
        let buffer = made
            .expect("never a raise")
            .expect("a buffer over the view");
        let after = counts();
        let delta: [u64; 11] = std::array::from_fn(|i| after[i] - before[i]);
        let several = delta == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0];
        let from_files = delta == [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1];
        let by_pointer = delta == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let copied = delta == [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1];
        assert!(
            several || (through && from_files) || (!through && by_pointer) || copied,
            "imports {delta:?}, the view through dma-bufs {through}"
        );
        let block = buffer.rt().block();
        assert_eq!(device.known_blocks().is_dma_buf(block), several);
        eprintln!(
            "a view of {n} runs, through dma-bufs {through}: imported as {}",
            match (several, from_files, by_pointer) {
                (true, ..) => "several dma-bufs",
                (_, true, _) => "its pointer, mapped from its files",
                (.., true) => "its pointer",
                _ => "a copy",
            }
        );
        assert_eq!((buffer.contents(), buffer.length()), (view, n * page));
        assert_eq!(read(view.wrapping_add((n - 1) * page), 16), [0x44; 16]);
        let Some(granule) = device.rt().dma_buf_granule().filter(|_| several) else {
            return;
        };

        // A blit reads every page through the dma-bufs, and writes two pages across the first
        // cut, where the first dma-buf's runs end, which the view then shows.
        let of = |length| device.new_buffer_of_length(on, length, 0);
        let of = |length| of(length).expect("never a raise").expect("a buffer");
        let (out, source) = (of(n * page), of(2 * page));
        // SAFETY: the Shared buffer's bytes, mapped while it lives.
        unsafe { (source.contents().as_ptr::<u8>()).write_bytes(0x99, (2 * page).to_usize()) };
        let cut = (n - 1) * page / granule * granule;
        assert!(cut >= 2 * page, "a cut inside the view");
        let queue = device.new_command_queue().expect("a command queue");
        // The read, then the write, each its own command buffer, waited for.
        let copies = [
            (&buffer, 0, &out, 0, n * page),
            (&source, 0, &buffer, cut - page, 2 * page),
        ];
        for (from, source_offset, to, destination_offset, size) in copies {
            let commands = queue.command_buffer(on).expect("never a raise");
            let commands = commands.expect("a command buffer");
            let encoder = commands.blit_encoder(on).expect("never a raise");
            let encoder = encoder.expect("a blit encoder");
            let copy = BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(from),
                source_offset,
                destination_buffer: Some(to),
                destination_offset,
                size,
            };
            assert_eq!(encoder.encode(on, copy), Ok(()));
            assert_eq!(encoder.encode(on, BlitCmd::EndEncoding), Ok(()));
            assert_eq!(commands.commit(on), Ok(()));
            assert_eq!(commands.wait_until_completed(on), Ok(()));
        }
        for i in 0..n {
            let byte = if i == n - 1 { 0x44 } else { run_byte(i) };
            let at = out.contents().wrapping_add(i * page);
            assert_eq!(
                read(at, page.to_usize()),
                vec![byte; page.to_usize()],
                "page {i}"
            );
        }
        let written = read(view.wrapping_add(cut - page), (2 * page).to_usize());
        assert_eq!(written, vec![0x99; (2 * page).to_usize()]);
        let run = ram.wrapping_add(2 * (cut / page) * page);
        assert_eq!(read(run, 16), [0x99; 16], "the run after the cut");
        let before_cut = read(view.wrapping_add(cut - page - 16), 16);
        assert_eq!(before_cut, vec![run_byte(cut / page - 2); 16]);
    }

    /// On Linux, on the GPU the test holds, a no-copy buffer over a view of two runs of a file's
    /// pages, where the host makes no dma-buf of them (`HOST_MAKES_NO_DMA_BUF`), and again where
    /// rt refuses the one it makes (`DMA_BUF_REFUSED`): where the view lay in a dma-buf's mapping,
    /// made by its pointer once it is mapped from its files, counted as from its files and as why
    /// no dma-buf went, each once; else by its pointer. Where rt refuses that pointer too (RADV's,
    /// of a file's pages), made over a copy, counted as refused and copied. Its bytes are the
    /// pages the view shows.
    #[cfg(target_os = "linux")]
    #[test]
    fn views_whose_dma_buf_went_nowhere_count_why_on_vulkan() {
        use crate::core::num::ToUsize as _;
        use crate::core::platform::os::vm::HostVm as _;

        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let page = device.facts().page_size;
        let codes = [
            IMPORTS_POINTER,
            IMPORTS_DMA_BUF,
            IMPORTS_FROM_FILES,
            IMPORTS_FROM_FILES_NO_DMA_BUF,
            IMPORTS_FROM_FILES_DMA_BUF_REFUSED,
            IMPORTS_FROM_FILES_TOO_LARGE,
            IMPORTS_REFUSED,
            IMPORTS_COPIED,
        ];
        let counts = || codes.map(counted);
        let vm = crate::os::OsVm::new();
        let hooks = [(&HOST_MAKES_NO_DMA_BUF, 3), (&DMA_BUF_REFUSED, 4)];
        for (hook, why) in hooks {
            // Two pages of a file, 0x55s then 0x66s, and a view of them, the second first.
            let ram = crate::os::allocate_aliasable(2 * page).expect("pages of a file");
            // SAFETY: fresh pages of the test's own, which nothing uses yet.
            unsafe {
                ram.as_ptr::<u8>().write_bytes(0x55, page.to_usize());
                (ram.wrapping_add(page).as_ptr::<u8>()).write_bytes(0x66, page.to_usize());
            }
            let view = vm.allocate_anywhere(2 * page).expect("a range for a view");
            let runs = [(ram.wrapping_add(page), page), (ram, page)];
            assert!(vm.remap_runs_shared_overwrite(view, &runs).is_ok());
            let first = NonNull::new(view.as_ptr::<u8>()).expect("a mapped view");
            let through = crate::os::through_dma_bufs(first, 2 * page);

            let before = counts();
            hook.set(true);
            // SAFETY: the view and its pages are the test's own, never freed.
            let bytes = unsafe { HostRange::new(view, 2 * page) };
            let made = device.new_buffer_over_host_memory(on, bytes, 0, None);
            hook.set(false);
            let buffer = made
                .expect("never a raise")
                .expect("a buffer over the view");
            let after = counts();
            let delta: [u64; 8] = std::array::from_fn(|i| after[i] - before[i]);
            let mut from_files = [1, 0, 1, 0, 0, 0, 0, 0];
            from_files[why] = 1;
            let by_pointer = [1, 0, 0, 0, 0, 0, 0, 0];
            let copied = [0, 0, 0, 0, 0, 0, 1, 1];
            let taken = if through { from_files } else { by_pointer };
            assert!(
                delta == taken || delta == copied,
                "imports {delta:?}, the view through a dma-buf {through}"
            );
            assert!(!device.known_blocks().is_dma_buf(buffer.rt().block()));
            assert_eq!((buffer.contents(), buffer.length()), (view, 2 * page));
            assert_eq!(read(view, 16), [0x66; 16]);
            assert_eq!(read(view.wrapping_add(page), 16), [0x55; 16]);
        }
    }

    /// On Linux, on the GPU the test holds, where views are mapped through dma-bufs: the first
    /// import of one whose one dma-buf rt refuses (`DMA_BUF_REFUSED`) maps every view made after
    /// it from its files at once (`crate::os`'s `views_need_files`). A view of two runs made then
    /// lies in no dma-buf's mapping, and its no-copy buffer maps none of its pages from their
    /// files again: by its pointer, or, where rt refuses that (RADV's, of a file's pages), as a
    /// dma-buf of the files behind them, or a copy, none counted as from its files. Its bytes are
    /// the pages the view shows. Where views are not mapped through dma-bufs here, unchecked.
    #[cfg(target_os = "linux")]
    #[test]
    fn views_after_one_refused_through_dma_bufs_are_mapped_from_files_on_vulkan() {
        use crate::core::num::ToUsize as _;
        use crate::core::platform::os::vm::HostVm as _;

        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let page = device.facts().page_size;
        let vm = crate::os::OsVm::new();
        // Two pages of a file, `first`s then `second`s, and a view of them, the second first.
        let view = |first: u8, second: u8| {
            let ram = crate::os::allocate_aliasable(2 * page).expect("pages of a file");
            // SAFETY: fresh pages of the test's own, which nothing uses yet.
            unsafe {
                ram.as_ptr::<u8>().write_bytes(first, page.to_usize());
                (ram.wrapping_add(page).as_ptr::<u8>()).write_bytes(second, page.to_usize());
            }
            let view = vm.allocate_anywhere(2 * page).expect("a range for a view");
            let runs = [(ram.wrapping_add(page), page), (ram, page)];
            assert!(vm.remap_runs_shared_overwrite(view, &runs).is_ok());
            let at = NonNull::new(view.as_ptr::<u8>()).expect("a mapped view");
            (view, crate::os::through_dma_bufs(at, 2 * page))
        };
        let (refused, through) = view(0x55, 0x66);
        if !through {
            eprintln!("views are not mapped through dma-bufs here: unchecked");
            return;
        }
        assert!(
            !crate::os::imports_need_files(),
            "no import refused a view before"
        );

        DMA_BUF_REFUSED.set(true);
        // SAFETY: the view and its pages are the test's own, never freed.
        let bytes = unsafe { HostRange::new(refused, 2 * page) };
        let made = device.new_buffer_over_host_memory(on, bytes, 0, None);
        DMA_BUF_REFUSED.set(false);
        let first = made
            .expect("never a raise")
            .expect("a buffer over the view");
        assert!(
            crate::os::imports_need_files(),
            "views are mapped from their files now"
        );

        let (later, through) = view(0x77, 0x88);
        assert!(
            !through,
            "a view made after it lies in no dma-buf's mapping"
        );
        let from_files = counted(IMPORTS_FROM_FILES);
        // SAFETY: the view and its pages are the test's own, never freed.
        let bytes = unsafe { HostRange::new(later, 2 * page) };
        let made = device.new_buffer_over_host_memory(on, bytes, 0, None);
        let buffer = made
            .expect("never a raise")
            .expect("a buffer over the view");
        assert_eq!(
            counted(IMPORTS_FROM_FILES),
            from_files,
            "nothing mapped again"
        );
        assert_eq!((buffer.contents(), buffer.length()), (later, 2 * page));
        assert_eq!(read(later, 16), [0x88; 16]);
        assert_eq!(read(later.wrapping_add(page), 16), [0x77; 16]);
        drop((first, buffer));
    }

    /// On Linux, on the GPU the test holds: the rt platform's device, as it is made, learns from
    /// rt's `dma_buf_granule` whether it binds no several dma-bufs behind one buffer
    /// (`crate::os`'s `learn_dma_buf_granule`); where it binds them, the test gives the answer of a
    /// device that binds none (NVIDIA's driver's). A view of one run more than one dma-buf holds
    /// (udmabuf's `list_limit` and one), made then, the first of the process, lies in no dma-buf's
    /// mapping, and its no-copy buffer maps none of its pages from their files again: by its
    /// pointer, or, where rt refuses that (RADV's, of a file's pages), as dma-bufs of the files
    /// behind them, or a copy, none counted as from its files. Nothing set views from their files
    /// for every view. Its bytes are the pages the view shows. Where the host makes no dma-buf, or
    /// udmabuf takes 4,096 runs or more, unchecked.
    #[cfg(target_os = "linux")]
    #[test]
    fn views_past_one_dma_buf_are_mapped_from_files_where_none_several_bind_on_vulkan() {
        use crate::core::num::ToUsize as _;
        use crate::core::platform::os::vm::HostVm as _;

        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let page = device.facts().page_size;
        let granule = device.rt().dma_buf_granule();
        assert_eq!(
            crate::os::several_dma_bufs_refused(),
            granule.is_none(),
            "the device's answer, learned as it was made"
        );
        crate::os::learn_dma_buf_granule(None);
        assert!(crate::os::several_dma_bufs_refused());
        let one = crate::os::allocate_aliasable(page).expect("a page of a file");
        let first = NonNull::new(one.as_ptr::<u8>()).expect("a mapped page");
        let lent = crate::os::dma_buf_of_host(first, page, &[(0, page)]).is_ok();
        let limit = std::fs::read_to_string("/sys/module/udmabuf/parameters/list_limit");
        let limit = limit
            .ok()
            .and_then(|limit| limit.trim().parse::<u64>().ok());
        let Some(n) = limit.filter(|&limit| lent && limit < 4096).map(|l| l + 1) else {
            eprintln!("udmabuf takes {limit:?} runs, dma-bufs {lent}: unchecked");
            return;
        };

        // A view of every other page of a file, each the byte its place in the view gives.
        let ram = crate::os::allocate_aliasable(2 * n * page).expect("pages of a file");
        let run_byte = |i: u64| (i % 199 + 1).to_le_bytes()[0];
        for i in 0..n {
            let run = ram.wrapping_add(2 * i * page).as_ptr::<u8>();
            // SAFETY: a fresh page of the test's own, which nothing uses yet.
            unsafe { run.write_bytes(run_byte(i), page.to_usize()) };
        }
        let runs: Vec<_> = (0..n)
            .map(|i| (ram.wrapping_add(2 * i * page), page))
            .collect();
        let vm = crate::os::OsVm::new();
        let view = vm.allocate_anywhere(n * page).expect("a range for a view");
        assert!(vm.remap_runs_shared_overwrite(view, &runs).is_ok());
        let at = NonNull::new(view.as_ptr::<u8>()).expect("a mapped view");
        assert!(
            !crate::os::through_dma_bufs(at, n * page),
            "the view lies in no dma-buf's mapping"
        );

        let from_files = counted(IMPORTS_FROM_FILES);
        // SAFETY: the view and its pages are the test's own, never freed.
        let bytes = unsafe { HostRange::new(view, n * page) };
        let made = device.new_buffer_over_host_memory(on, bytes, 0, None);
        let buffer = made
            .expect("never a raise")
            .expect("a buffer over the view");
        assert_eq!(
            counted(IMPORTS_FROM_FILES),
            from_files,
            "nothing mapped again"
        );
        assert!(
            !crate::os::imports_need_files(),
            "views of one dma-buf as before"
        );
        assert_eq!((buffer.contents(), buffer.length()), (view, n * page));
        for i in [0, 1, n / 2, n - 1] {
            let at = view.wrapping_add(i * page);
            assert_eq!(read(at, 16), [run_byte(i); 16], "page {i}");
        }
    }

    /// On Linux, on the GPU the test holds, where rt refuses both pages' pointer
    /// (`POINTER_REFUSED`) and the dma-buf the host makes of them (`DMA_BUF_REFUSED`): a no-copy
    /// buffer over a view of two runs of a file's pages, whose dma-buf goes first where the view
    /// lies in a dma-buf's mapping, and one over a run of a file's pages, whose pointer goes
    /// first, are each made over a copy, counted as refused, copied, and as both refused once.
    /// Their `contents` are the pages. Where the host makes no dma-buf, unchecked.
    #[cfg(target_os = "linux")]
    #[test]
    fn imports_whose_dma_buf_and_pointer_rt_refuses_count_so_on_vulkan() {
        use crate::core::num::ToUsize as _;
        use crate::core::platform::os::vm::HostVm as _;

        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let page = device.facts().page_size;
        let codes = [
            IMPORTS_POINTER,
            IMPORTS_DMA_BUF,
            IMPORTS_FROM_FILES,
            IMPORTS_REFUSED,
            IMPORTS_COPIED,
            IMPORTS_DMA_BUF_AND_POINTER_REFUSED,
        ];
        let counts = || codes.map(counted);

        // Two pages of a file, 0x55s then 0x66s, a view of them, the second first, and a run.
        let ram = crate::os::allocate_aliasable(2 * page).expect("pages of a file");
        // SAFETY: fresh pages of the test's own, which nothing uses yet.
        unsafe {
            ram.as_ptr::<u8>().write_bytes(0x55, page.to_usize());
            (ram.wrapping_add(page).as_ptr::<u8>()).write_bytes(0x66, page.to_usize());
        }
        let vm = crate::os::OsVm::new();
        let view = vm.allocate_anywhere(2 * page).expect("a range for a view");
        let runs = [(ram.wrapping_add(page), page), (ram, page)];
        assert!(vm.remap_runs_shared_overwrite(view, &runs).is_ok());
        let run = crate::os::allocate_aliasable(page).expect("a page of a file");
        // SAFETY: a fresh page of the test's own, which nothing uses yet.
        unsafe { run.as_ptr::<u8>().write_bytes(0x77, page.to_usize()) };
        let one = NonNull::new(run.as_ptr::<u8>()).expect("a mapped page");
        if crate::os::dma_buf_of_host(one, page, &[(0, page)]).is_err() {
            eprintln!("no dma-buf made here: unchecked");
            return;
        }

        POINTER_REFUSED.set(true);
        DMA_BUF_REFUSED.set(true);
        for (at, length, first) in [(view, 2 * page, 0x66), (run, page, 0x77)] {
            let before = counts();
            // SAFETY: the pages are the test's own, never freed.
            let bytes = unsafe { HostRange::new(at, length) };
            let made = device.new_buffer_over_host_memory(on, bytes, 0, None);
            let buffer = made.expect("never a raise").expect("a buffer over a copy");
            let after = counts();
            let delta: [u64; 6] = std::array::from_fn(|i| after[i] - before[i]);
            assert_eq!(delta, [0, 0, 0, 1, 1, 1], "{length} bytes");
            assert_eq!((buffer.contents(), buffer.length()), (at, length));
            assert_eq!(read(at, 16), [first; 16]);
        }
        POINTER_REFUSED.set(false);
        DMA_BUF_REFUSED.set(false);
    }

    /// On the GPU the test holds, where the GPU takes no pointer to pages with no file behind them
    /// (`POINTER_REFUSED`, as RADV refuses a memfd's) and no dma-buf of them is made: a no-copy
    /// buffer over them is made over a copy ("A copy at the refusal point"), counted, its
    /// `contents` the pages and rt's mapping the copy. The CPU's writes before a commit reach a
    /// blit's read; the blit's writes reach the pages once its command buffer completed, held for a
    /// shared event's value meanwhile, and the bytes beside them that the CPU wrote while it was
    /// held keep the CPU's. A known import made over a copy holds a no-copy buffer pinned over its
    /// bytes, with no import of its own. Each deallocator runs once rt freed its copy, whose pages
    /// the device then forgets.
    #[test]
    fn copies_of_pages_on_vulkan() {
        use crate::core::num::{ToU64 as _, ToUsize as _};
        use crate::core::platform::event::{EventRef, SharedEvent as _};

        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        POINTER_REFUSED.set(true);
        let page = device.facts().page_size;
        let flag = || {
            let ran = Arc::new(AtomicBool::new(false));
            let run = Arc::clone(&ran);
            (ran, move || run.store(true, Ordering::Release))
        };

        // Pages of the process's own, with no file behind them: 0x11s.
        let layout = Layout::from_size_align((4 * page).to_usize(), page.to_usize());
        let layout = layout.expect("a layout");
        // SAFETY: the layout's size is not 0.
        let memory = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!memory.is_null());
        let at = HostPtr::from_ptr(memory);
        // SAFETY: the memory is the test's own, never freed.
        unsafe { memory.write_bytes(0x11, (4 * page).to_usize()) };
        let copied = counted(IMPORTS_COPIED);
        let (freed, deallocator) = flag();
        // SAFETY: the memory is never freed.
        let bytes = unsafe { HostRange::new(at, 2 * page) };
        let made = device.new_buffer_over_host_memory(on, bytes, 0, Some(Box::new(deallocator)));
        let buffer = made.expect("never a raise").expect("a buffer over a copy");
        assert_eq!(counted(IMPORTS_COPIED), copied + 1);
        assert_eq!((buffer.contents(), buffer.length()), (at, 2 * page));
        let mapping = buffer.rt().contents().expect("rt's mapping of the copy");
        let mapping = HostPtr::from_ptr(mapping.as_ptr());
        assert_ne!(mapping, at);
        assert_eq!(read(mapping, 64), [0x11; 64]);
        let id = buffer.rt().block();

        // The CPU writes 0x22s at 128 after the make, before the commit.
        let page_at = |offset: u64| at.wrapping_add(offset).as_ptr::<u8>();
        // SAFETY: inside the test's memory.
        unsafe { page_at(128).write_bytes(0x22, 64) };
        let of = |length, options| device.new_buffer_of_length(on, length, options);
        let of = |length| of(length, 0).expect("never a raise").expect("a buffer");
        let (out, source) = (of(64), of(64));
        // SAFETY: the Shared buffer's 64 bytes, mapped while it lives.
        unsafe { source.contents().as_ptr::<u8>().write_bytes(0x77, 64) };

        // A blit reads them, and writes 0x77s at the second page, held for a shared event's value;
        // meanwhile the CPU writes 0x99s beside the blit's bytes.
        let shared = device.new_shared_event().expect("a shared event");
        let queue = device.new_command_queue().expect("a command queue");
        let commands = queue.command_buffer(on).expect("never a raise");
        let commands = commands.expect("a command buffer");
        let wait = commands.encode_wait(Some(EventRef::Shared(&shared)), 1);
        assert_eq!(wait, Ok(()));
        let encoder = commands.blit_encoder(on).expect("never a raise");
        let encoder = encoder.expect("a blit encoder");
        let copies = [(&buffer, 128, &out, 0), (&source, 0, &buffer, page)];
        for (from, source_offset, to, destination_offset) in copies {
            let copy = BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(from),
                source_offset,
                destination_buffer: Some(to),
                destination_offset,
                size: 64,
            };
            assert_eq!(encoder.encode(on, copy), Ok(()));
        }
        assert_eq!(encoder.encode(on, BlitCmd::EndEncoding), Ok(()));
        assert_eq!(commands.commit(on), Ok(()));
        // SAFETY: inside the test's memory.
        unsafe { page_at(page + 64).write_bytes(0x99, 8) };
        assert_eq!(shared.set_signaled_value(1), Ok(()));
        assert_eq!(commands.wait_until_completed(on), Ok(()));
        assert_eq!(read(out.contents(), 64), [0x22; 64]);
        assert_eq!(read(at.wrapping_add(page), 64), [0x77; 64]);
        let beside = read(at.wrapping_add(page + 64), 16);
        assert_eq!(beside, [[0x99; 8], [0x11; 8]].concat());

        // A known import of the pages, made over a copy: a no-copy buffer pinned over its bytes
        // is a range of its block, with no import of its own.
        let first = at.wrapping_add(2 * page + 32);
        let (released, then) = flag();
        let known = import_known(&device, "a test's import", first, 256, false, then);
        let (block, offset) = known.expect("a known import over a copy");
        let home = device.known_blocks().contents_of(&block);
        assert_eq!(home.map(|home| home + offset), Some(first.addr().to_u64()));
        let (pin, imports) = (block.gpu_address + offset, counted(IMPORTS_CODE));
        // SAFETY: the memory is never freed.
        let bytes = unsafe { HostRange::new(first, 64) };
        let made = device.new_buffer(pinned(pin, 0, Some(bytes)));
        let made = made.expect("never a raise").expect("a buffer in the block");
        assert_eq!((made.gpu_address(), made.contents()), (pin, first));
        assert_eq!(counted(IMPORTS_CODE), imports);
        drop(made);
        let known = block.id;
        assert!(device.known_blocks().forget(known).is_some());
        drop(block);
        assert!(
            waits_for(|| released.load(Ordering::Acquire)),
            "the release ran"
        );
        assert_eq!(device.known_blocks().homes.of(known), None);

        drop((encoder, commands, buffer));
        assert!(
            waits_for(|| freed.load(Ordering::Acquire)),
            "the deallocator ran"
        );
        assert_eq!(device.known_blocks().homes.of(id), None);
    }

    /// On the GPU the test holds: a no-copy buffer of option bit 24 (read-only) with no pin lies
    /// over a copy of its own whose GPU writes are dropped ("Read-only memory"), counted as such
    /// and not as one whose writes reach its bytes. A blit's fill and copy into it complete, and
    /// once their command buffer completed the pages hold the CPU's bytes alone, as after
    /// Metal's; a later command buffer reads those, a CPU write made before its commit among them.
    /// A render pass that stores into a texture over it completes with its store dropped too,
    /// counted at the pass, as the 5300M and the UHD 630 answer.
    #[test]
    fn read_only_copies_drop_the_gpu_s_writes_on_vulkan() {
        use super::super::copied::{DROPPED_BYTES_CODE, STORE_OUTS_CODE};
        use crate::core::num::ToUsize as _;
        use crate::core::platform::descriptor::RenderPassDescriptor as _;
        use crate::core::platform::render::{RenderCmd, RenderEncoder as _};
        use crate::core::platform::types::{ClearColor, CommandBufferStatus, NsRange};
        use crate::core::platform::values::ColorAttachmentSet;
        use crate::rt_platform::PassFields;

        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let page = device.facts().page_size;
        let layout = Layout::from_size_align((2 * page).to_usize(), page.to_usize());
        let layout = layout.expect("a layout");
        // SAFETY: the layout's size is not 0.
        let memory = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!memory.is_null());
        let at = HostPtr::from_ptr(memory);
        // SAFETY: the test's own memory, freed once the deallocator ran.
        unsafe { memory.write_bytes(0x11, (2 * page).to_usize()) };
        let in_place = counted(READ_ONLY_CODE);
        let copies = (counted(IMPORTS_READ_ONLY), counted(IMPORTS_COPIED));
        let freed = Arc::new(AtomicBool::new(false));
        let done = Arc::clone(&freed);
        // SAFETY: the memory stays allocated until the deallocator ran (below).
        let bytes = unsafe { HostRange::new(at, 2 * page) };
        let made = device.new_buffer(BufferDesc {
            length: 2 * page,
            options: vec![OptionSet::ResourceOptions(READ_ONLY)],
            pinned_gpu_address: None,
            contents: Some(bytes),
            no_copy: true,
            deallocator: Some(Box::new(move |_, _| done.store(true, Ordering::Release))),
            zero_from: None,
        });
        let buffer = made
            .expect("never a raise")
            .expect("a read-only no-copy buffer");
        assert_eq!(buffer.contents(), at);
        assert_eq!(counted(READ_ONLY_CODE), in_place, "none in place");
        let copies_now = (counted(IMPORTS_READ_ONLY), counted(IMPORTS_COPIED));
        assert_eq!(copies_now, (copies.0 + 1, copies.1 + 1));

        // A fill of its first 64 bytes and a copy of 0x77s into the next 64: both dropped.
        let queue = device.new_command_queue().expect("a command queue");
        let of = |length| device.new_buffer_of_length(on, length, 0);
        let of = |length| of(length).expect("never a raise").expect("a buffer");
        let (source, out) = (of(64), of(128));
        // SAFETY: the Shared buffer's 64 bytes, mapped while it lives.
        unsafe { source.contents().as_ptr::<u8>().write_bytes(0x77, 64) };
        let dropped = counted(DROPPED_BYTES_CODE);
        let fill = BlitCmd::FillBuffer {
            buffer: &buffer,
            range: NsRange::new(0, 64),
            value: 0x55,
        };
        let copy = BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(&source),
            source_offset: 0,
            destination_buffer: Some(&buffer),
            destination_offset: 64,
            size: 64,
        };
        blit_and_wait(&queue, vec![fill, copy]);
        assert_eq!(read(at, 128), [0x11; 128], "the pages as the CPU left them");
        assert!(
            counted(DROPPED_BYTES_CODE) > dropped,
            "the GPU's bytes put back"
        );

        // The CPU writes 0x22s at 128; a later command buffer reads them beside the dropped bytes.
        // SAFETY: inside the test's memory.
        unsafe { memory.wrapping_add(128).write_bytes(0x22, 64) };
        let copy = BlitCmd::CopyBufferToBuffer {
            source_buffer: Some(&buffer),
            source_offset: 64,
            destination_buffer: Some(&out),
            destination_offset: 0,
            size: 128,
        };
        blit_and_wait(&queue, vec![copy]);
        assert_eq!(read(out.contents(), 128), [[0x11; 64], [0x22; 64]].concat());

        // A render pass that clears and stores a texture over the second page: it completes, its
        // store dropped, counted at the pass.
        let descriptor = platform.texture_descriptor(&TextureDesc {
            width: 64,
            height: 4,
            usage: 0x4,
            ..TextureDesc::default()
        });
        let texture = buffer.new_texture(on, &descriptor, page, 256);
        let texture = texture
            .expect("never a raise")
            .expect("a texture over the buffer");
        let pass = RtObj::new(PassFields::default());
        let set = ColorAttachmentSet {
            texture: Some(Some(&texture)),
            load_action: Some(2),
            store_action: Some(1),
            clear_color: Some(ClearColor {
                red: 1.0,
                alpha: 1.0,
                ..ClearColor::default()
            }),
            ..ColorAttachmentSet::default()
        };
        assert_eq!(pass.set_color_attachment(on, 0, &set), Ok(()));
        let store_outs = counted(STORE_OUTS_CODE);
        let commands = queue.command_buffer(on).expect("never a raise");
        let commands = commands.expect("a command buffer");
        let encoder = commands.render_encoder(on, &pass).expect("never a raise");
        let encoder = encoder.expect("a render encoder");
        assert_eq!(counted(STORE_OUTS_CODE), store_outs + 1);
        assert_eq!(encoder.encode(on, RenderCmd::EndEncoding), Ok(()));
        assert_eq!(commands.commit(on), Ok(()));
        assert_eq!(commands.wait_until_completed(on), Ok(()));
        assert_eq!(commands.status(), CommandBufferStatus::COMPLETED);
        let second = at.wrapping_add(page);
        assert_eq!(read(second, 1024), [0x11; 1024], "the store dropped");

        drop((encoder, commands, pass, texture, buffer));
        assert!(
            waits_for(|| freed.load(Ordering::Acquire)),
            "the deallocator ran"
        );
        // SAFETY: the deallocator ran: rt holds no copy of the memory.
        unsafe { std::alloc::dealloc(memory, layout) };
    }

    /// A buffer over a copy of two pages of the test's own, 0x11s, and rt's mapping of the copy
    /// (`super::super::copied`'s docs).
    struct OverACopy {
        /// The buffer, whose `contents` are the pages.
        buffer: RtObj<RtBuffer>,
        /// The pages.
        pages: HostPtr,
        /// rt's mapping of the copy.
        copy: HostPtr,
        /// Whether the GPU's writes to the copy are dropped.
        drops: bool,
    }

    /// A buffer over a copy of two fresh pages of 0x11s on `device`: read-only (option bit 24),
    /// whose GPU writes are dropped, where `drops`, else one whose pointer the GPU takes not
    /// (`POINTER_REFUSED`). The pages are never freed.
    fn over_a_copy(device: &RtObj<RtDevice>, drops: bool) -> OverACopy {
        use crate::core::num::ToUsize as _;

        let on = OnFault::Return;
        let page = device.facts().page_size;
        let layout = Layout::from_size_align((2 * page).to_usize(), page.to_usize());
        let layout = layout.expect("a layout");
        // SAFETY: the layout's size is not 0.
        let memory = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!memory.is_null());
        // SAFETY: the test's own memory.
        unsafe { memory.write_bytes(0x11, (2 * page).to_usize()) };
        let pages = HostPtr::from_ptr(memory);
        let copied = counted(IMPORTS_COPIED);
        // SAFETY: the memory is never freed.
        let bytes = unsafe { HostRange::new(pages, 2 * page) };
        let made = match drops {
            true => device.new_buffer(BufferDesc {
                length: 2 * page,
                options: vec![OptionSet::ResourceOptions(READ_ONLY)],
                pinned_gpu_address: None,
                contents: Some(bytes),
                no_copy: true,
                deallocator: None,
                zero_from: None,
            }),
            false => {
                POINTER_REFUSED.set(true);
                let made = device.new_buffer_over_host_memory(on, bytes, 0, None);
                POINTER_REFUSED.set(false);
                made
            }
        };
        let buffer = made.expect("never a raise").expect("a buffer over a copy");
        assert_eq!(counted(IMPORTS_COPIED), copied + 1);
        let copy = buffer.rt().contents().expect("rt's mapping of the copy");
        let copy = HostPtr::from_ptr(copy.as_ptr());
        assert_ne!(copy, pages);
        OverACopy {
            buffer,
            pages,
            copy,
            drops,
        }
    }

    impl OverACopy {
        /// The CPU writes 0x22s at the pages' first 64 bytes, and 0x33s at the copy's 64 from
        /// 128, as the GPU writes it.
        fn mark(&self) {
            // SAFETY: inside the test's pages and the copy, which no command buffer in flight
            // writes.
            unsafe {
                self.pages.as_ptr::<u8>().write_bytes(0x22, 64);
                self.copy
                    .wrapping_add(128)
                    .as_ptr::<u8>()
                    .write_bytes(0x33, 64);
            }
        }

        /// What a commit and completion did with the copy since [`OverACopy::mark`]: whether
        /// the CPU's bytes reached the copy, and whether the copy's reached the pages, or where
        /// its writes are dropped, were put back from the pages' bytes. The pages of a copy
        /// whose writes are dropped never change.
        fn synced(&self) -> (bool, bool) {
            let copied_in = read(self.copy, 64) == [0x22; 64];
            let back = match self.drops {
                false => read(self.pages.wrapping_add(128), 64) == [0x33; 64],
                true => {
                    assert_eq!(read(self.pages.wrapping_add(128), 64), [0x11; 64]);
                    read(self.copy.wrapping_add(128), 64) == [0x11; 64]
                }
            };
            (copied_in, back)
        }
    }

    /// Runs `cmds` in a compute encoder of a new command buffer of `queue`, and waits for it.
    fn compute_and_wait(
        queue: &<RtPlatform as crate::core::platform::Platform>::CommandQueue,
        cmds: Vec<ComputeCmd<'_, RtPlatform>>,
    ) {
        let on = OnFault::Return;
        let commands = queue.command_buffer(on).expect("never a raise");
        let commands = commands.expect("a command buffer");
        let encoder = commands.compute_encoder(on).expect("never a raise");
        let encoder = encoder.expect("a compute encoder");
        for cmd in cmds {
            assert_eq!(encoder.encode(on, cmd), Ok(()));
        }
        assert_eq!(encoder.encode(on, ComputeCmd::EndEncoding), Ok(()));
        assert_eq!(commands.commit(on), Ok(()));
        assert_eq!(commands.wait_until_completed(on), Ok(()));
    }

    /// On the GPU the test holds: a copy a compute encoder binds with `setBuffers:` is synced for
    /// its command buffer (`super::super::copied`'s docs, "Synced for the command buffers that use
    /// it"): a dispatch reads the bytes the CPU wrote before the commit, and once the command
    /// buffer completed the bytes a dispatch wrote into the copy are in the pages, or, where the
    /// copy's writes are dropped, put back, counted.
    #[test]
    fn a_copy_bound_by_set_buffers_is_synced_at_commit_and_completion_on_vulkan() {
        use super::super::copied::DROPPED_BYTES_CODE;
        use crate::core::platform::resource::Library as _;
        use crate::core::platform::types::{MtlSize, NsRange};

        const ADD: &[u8] = include_bytes!("../../../../tests/shaders/compute_add.metallib");
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let library = device.new_library(ADD).expect("never a raise");
        let library = library.expect("a metallib").expect("a library");
        let function = library.new_function(on, "add_one").expect("never a raise");
        let function = function.expect("a function of the table");
        let add = device.new_compute_pipeline_state_for_function(Some(&function));
        let add = add.expect("a compute state").expect("a compute state");
        let of = |length| device.new_buffer_of_length(on, length, 0);
        let of = |length| of(length).expect("never a raise").expect("a buffer");
        for drops in [false, true] {
            let c = over_a_copy(&device, drops);
            let (out, zeros) = (of(16), of(16));
            // SAFETY: inside the test's pages.
            unsafe { c.pages.as_ptr::<u8>().write_bytes(0x22, 16) };
            let dropped = counted(DROPPED_BYTES_CODE);
            let threads = || ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(4, 1, 1),
                threads_per_threadgroup: MtlSize::new(4, 1, 1),
            };
            // Reads the copy's first 4 words into `out`, then writes 1s into its 4 from 128.
            compute_and_wait(
                &queue,
                vec![
                    ComputeCmd::SetComputePipelineState { state: Some(&add) },
                    ComputeCmd::SetBuffers {
                        buffers: &[Some(&out), Some(&c.buffer)],
                        offsets: &[0, 0],
                        range: NsRange::new(0, 2),
                    },
                    threads(),
                    ComputeCmd::SetBuffers {
                        buffers: &[Some(&c.buffer), Some(&zeros)],
                        offsets: &[128, 0],
                        range: NsRange::new(0, 2),
                    },
                    threads(),
                ],
            );
            assert_eq!(read(out.contents(), 16), [0x23, 0x22, 0x22, 0x22].repeat(4));
            let back = read(c.pages.wrapping_add(128), 16);
            match drops {
                false => assert_eq!(back, [1, 0, 0, 0].repeat(4)),
                true => {
                    assert_eq!(back, [0x11; 16]);
                    assert!(
                        counted(DROPPED_BYTES_CODE) > dropped,
                        "the GPU's bytes put back"
                    );
                }
            }
        }
    }

    /// On the GPU the test holds: a copy a blit names is synced for its command buffer as one an
    /// encoder binds: a copy out of it reads the bytes the CPU wrote before the commit, and once
    /// the command buffer completed a copy into it is in the pages, or dropped.
    #[test]
    fn a_copy_bound_by_a_blit_is_synced_at_commit_and_completion_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let of = |length| device.new_buffer_of_length(on, length, 0);
        let of = |length| of(length).expect("never a raise").expect("a buffer");
        for drops in [false, true] {
            let c = over_a_copy(&device, drops);
            let (out, source) = (of(64), of(64));
            // SAFETY: the Shared buffer's 64 bytes, mapped while it lives, and the test's pages.
            unsafe {
                source.contents().as_ptr::<u8>().write_bytes(0x77, 64);
                c.pages.as_ptr::<u8>().write_bytes(0x22, 64);
            }
            let copy = |(from, to, destination_offset)| BlitCmd::CopyBufferToBuffer {
                source_buffer: Some(from),
                source_offset: 0,
                destination_buffer: Some(to),
                destination_offset,
                size: 64,
            };
            let cmds = [(&c.buffer, &out, 0), (&source, &c.buffer, 128)];
            blit_and_wait(&queue, cmds.into_iter().map(copy).collect());
            assert_eq!(read(out.contents(), 64), [0x22; 64]);
            let back = read(c.pages.wrapping_add(128), 64);
            assert_eq!(back, [if drops { 0x11 } else { 0x77 }; 64]);
        }
    }

    /// On the GPU the test holds: a copy a compute encoder declares with `useResources:`, as a
    /// shader that reaches it through an argument buffer's address asks, and binds not, is synced
    /// for its command buffer: the CPU's bytes copied in at the commit, the copy's merged back,
    /// or put back, at the completion.
    #[test]
    fn a_copy_declared_with_use_resources_is_synced_on_vulkan() {
        use crate::core::platform::blit::ResourceRef;

        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let queue = device.new_command_queue().expect("a command queue");
        for drops in [false, true] {
            let c = over_a_copy(&device, drops);
            c.mark();
            let declare = ComputeCmd::UseResources {
                resources: &[Some(ResourceRef::Buffer(&c.buffer))],
                usage: 1,
            };
            compute_and_wait(&queue, vec![declare]);
            assert_eq!(c.synced(), (true, true), "drops: {drops}");
        }
    }

    /// On the GPU the test holds: a command buffer that uses no copy, of a process none of whose
    /// copies escaped, syncs none, as the platform's own command buffers, its fills and
    /// write-backs, use none, though another process's copy escaped: a live copy keeps what it
    /// holds and its pages theirs, and no sync is counted.
    #[test]
    fn a_command_buffer_that_uses_no_copy_syncs_none_on_vulkan() {
        use super::super::copied::{BACK_BYTES_CODE, IN_BYTES_CODE, SYNC_NS_CODE};
        use crate::core::platform::command::EscapedCopies;
        use crate::core::platform::types::NsRange;

        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let other = device.new_buffer_of_length(on, 64, 0);
        let other = other.expect("never a raise").expect("a buffer");
        let codes = [IN_BYTES_CODE, BACK_BYTES_CODE, SYNC_NS_CODE];
        for drops in [false, true] {
            let c = over_a_copy(&device, drops);
            let (ours, others) = (EscapedCopies::default(), EscapedCopies::default());
            others.add(&c.buffer.escaped_copy().expect("the copy under the buffer"));
            c.mark();
            let counts = codes.map(counted);
            let fill = BlitCmd::FillBuffer {
                buffer: &other,
                range: NsRange::new(0, 64),
                value: 0x55,
            };
            blit_and_wait_of(&queue, Some(&ours), vec![fill]);
            assert_eq!(read(other.contents(), 64), [0x55; 64]);
            assert_eq!(c.synced(), (false, false), "drops: {drops}");
            assert_eq!(codes.map(counted), counts, "no sync counted");
        }
    }

    /// On the GPU the test holds: a copy whose identity escaped (`super::super::copied`'s docs,
    /// "And for every command buffer of its process once it escaped"), as an app's read of its
    /// buffer's `gpuAddress`, or of the `gpuResourceID` of a texture over it, makes it, is synced
    /// at each commit of its process's command buffers though none of their encoders sets or
    /// declares it: the CPU's bytes copied in at the commit, the copy's merged back, or put back,
    /// at the completion. No kernel of ours shows it on rt, which reaches no copy a command buffer
    /// leaves undeclared: its translator lowers no address a function loads (its README, "Known
    /// gaps of pipeline states"), a texture over the copy, a buffer in an imported block, is an
    /// image of its own that rt copies the buffer's bytes into only before the submissions that
    /// use it, and a storage read through a resource ID no `use_resource` declared answers zeros.
    /// So the test reads the copy and the pages, as its sibling of another process's escape does
    /// ([`a_command_buffer_that_uses_no_copy_syncs_none_on_vulkan`]).
    #[test]
    fn an_escaped_copy_no_encoder_names_is_synced_at_its_process_s_commits_on_vulkan() {
        use crate::core::platform::command::EscapedCopies;
        use crate::core::platform::types::NsRange;

        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let other = device.new_buffer_of_length(on, 64, 0);
        let other = other.expect("never a raise").expect("a buffer");
        for drops in [false, true] {
            let c = over_a_copy(&device, drops);
            let escapes = EscapedCopies::default();
            escapes.add(&c.buffer.escaped_copy().expect("the copy under the buffer"));
            c.mark();
            let fill = BlitCmd::FillBuffer {
                buffer: &other,
                range: NsRange::new(0, 64),
                value: 0x55,
            };
            blit_and_wait_of(&queue, Some(&escapes), vec![fill]);
            assert_eq!(read(other.contents(), 64), [0x55; 64]);
            assert_eq!(c.synced(), (true, true), "drops: {drops}");
        }
    }
}
