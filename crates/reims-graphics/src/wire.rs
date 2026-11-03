//! The device's wire format, as the guest and the host share it: the PCI identity, the registers
//! of BAR 0, the layout of BAR 2 (the region), the rings' header, and the control ring's commands
//! and events. cbindgen writes it into `include/reims_graphics_wire.h`, the header the guest's
//! kext and option ROM include (`cargo xtask header`).
//!
//! Every value is little-endian, as the guest and every host the library builds for are. A
//! message's struct is the whole message, its header included, so a writer fills in one struct;
//! a `u64` field that would follow a `u32` at an offset that is not a multiple of 8 has a
//! `reserved` field before it, so the structs are laid out the same in C and in Rust, with no
//! padding the compiler adds.
//!
//! **Rings.** A ring is a [`RgfxRingHeader`] and a byte ring of [`RGFX_CONTROL_RING_DATA_SIZE`]
//! bytes after it. `producer` and `consumer` are free-running byte counts: a message is at
//! `count % size` in the byte ring. A message starts with an [`RgfxMsgHeader`], its `size` a
//! multiple of 8 that includes the header, and never wraps: where the bytes left before the
//! ring's end are fewer than the message, the producer pads them with a message of type
//! [`RGFX_MSG_PAD`], whose `size` covers them. The bytes left are a multiple of 8, and may be 8,
//! so a pad's header is its `type` and `size` alone: its `seqno` is not written or read.
//!
//! **Doorbell and interrupts.** The producer of a submission ring (the guest) writes a message,
//! publishes `producer` (a release store), issues a full fence and reads `consumer_idle`; if it
//! reads 1 it writes the ring's id to [`RGFX_REG_DOORBELL`]. The consumer (the host) sets
//! `consumer_idle` before it sleeps, issues a full fence and reads `producer` again, so no wake
//! is lost, and clears it when it wakes. The producer of a completion ring (the host) publishes a
//! message, issues a full fence and reads `armed`; if it reads 1, it clears it, reads `consumer`,
//! and raises the ring's MSI-X vector, or, where the guest has taken every message published, sets
//! `armed` again and raises nothing: the 1 it cleared was an arm the guest made after its take.
//! The guest stores `consumer` as it drains the ring, then sets `armed`, issues a full fence and
//! reads `producer` again.

use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout};

/// The device's PCI vendor ID: a placeholder until the owner picks the device's own.
pub const RGFX_PCI_VENDOR_ID: u16 = 0x1234;
/// The device's PCI device ID: a placeholder, as the vendor ID is.
pub const RGFX_PCI_DEVICE_ID: u16 = 0x5247;
/// The device's PCI class: a display controller of class "other" (0x0380).
pub const RGFX_PCI_CLASS: u16 = 0x0380;
/// The BAR of the registers: 4 KiB of MMIO, with the MSI-X table and PBA from
/// [`RGFX_MSIX_OFFSET`].
pub const RGFX_PCI_BAR_REGISTERS: u32 = 0;
/// The BAR of the region: 64-bit and prefetchable, over host memory the guest maps write-back.
pub const RGFX_PCI_BAR_REGION: u32 = 2;
/// The size of BAR 0.
pub const RGFX_REGISTERS_SIZE: u32 = 0x1000;
/// Where in BAR 0 the MSI-X table starts, the PBA after it.
pub const RGFX_MSIX_OFFSET: u32 = 0x800;
/// The device's MSI-X vectors: vector 0 is the control completion ring's.
pub const RGFX_MSIX_VECTORS: u32 = 1;

/// `MAGIC`, RO: [`RGFX_MAGIC`].
pub const RGFX_REG_MAGIC: u32 = 0x000;
/// `VERSION`, RO: [`RGFX_VERSION`].
pub const RGFX_REG_VERSION: u32 = 0x004;
/// `FEATURES_LO`, RO: feature bits 0 to 31, the ones the host offers: [`RGFX_FEATURE_SURFACE_PAGES`],
/// [`RGFX_FEATURE_SURFACE_CACHE`] and [`RGFX_FEATURE_CB_SUBMITTED`], and from bit 3 the ones the
/// guest takes through [`RGFX_REG_DRIVER_FEATURES_LO`] ([`RGFX_FEATURE_QUIET_SCHEDULED`] to
/// [`RGFX_FEATURE_QUEUE_VECTORS`], [`RGFX_FEATURE_SURFACE_ORDER`],
/// [`RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS`], [`RGFX_FEATURE_UNSHARED_QUEUES`] and
/// [`RGFX_FEATURE_MANAGED_AT_OFFSET`]). Bits 9, 10 and 15 are reserved.
pub const RGFX_REG_FEATURES_LO: u32 = 0x008;
/// `FEATURES_HI`, RO: feature bits 32 to 63.
pub const RGFX_REG_FEATURES_HI: u32 = 0x00c;
/// `FEATURES_LO` bit 0, `SURFACE_PAGES`: the host maps an IOSurface's own guest pages for its GPU
/// to use in place (`SURFACE_PAGES`, `BUFFER_OVER_SURFACE_PAGES`); without it a surface's GPU
/// view is a copy of its pages in a block of the region.
pub const RGFX_FEATURE_SURFACE_PAGES: u32 = 1 << 0;
/// `FEATURES_LO` bit 1, `SURFACE_CACHE`, beside [`RGFX_FEATURE_SURFACE_PAGES`]: the host keeps
/// the GPU's copy of an IOSurface whose block the kext marks cached ([`RGFX_SURFACE_PAGES_CACHED`])
/// in its own GPU's memory, the surface's device cache, as a GPU that does not share the CPU's
/// memory keeps one. The cache is filled from the surface's pages at the first command buffer
/// that uses it after the CPU's generation of the surface moved (`RGFX_OP_SURFACE_USE`), and the
/// pages from the cache only at a [`RGFX_CMD_SURFACE_DOWNLOAD`] and at the block's
/// `BLOCK_RELEASE`; a write the host's own devices make into the pages (a video device's frame)
/// has the next use fill it again. Set where the host's GPU does not share the CPU's memory; the
/// kext alone reads it, and nothing the guest's processes see changes but the time copies take.
pub const RGFX_FEATURE_SURFACE_CACHE: u32 = 1 << 1;
/// `FEATURES_LO` bit 2, `CB_SUBMITTED`: a command buffer that holds an
/// [`RGFX_OP_SUBMITTED`](crate::ops::RGFX_OP_SUBMITTED) gets a [`RGFX_EVT_CB_SUBMITTED`], at which
/// the guest's `commitAndWaitUntilSubmitted` returns. Without it the guest sends no such op, and
/// that call returns at the buffer's `CB_SCHEDULED`.
pub const RGFX_FEATURE_CB_SUBMITTED: u32 = 1 << 2;
/// `FEATURES_LO` bit 3, `QUIET_SCHEDULED`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`]: on a queue created with [`RGFX_QUEUE_QUIET_SCHEDULED`], a
/// `CB_SCHEDULED` raises the ring's vector only where its buffer holds
/// [`RGFX_OP_SCHEDULED_WANTED`](crate::ops::RGFX_OP_SCHEDULED_WANTED) or the ring's `raise_all` is
/// nonzero. No host offers it yet.
pub const RGFX_FEATURE_QUIET_SCHEDULED: u32 = 1 << 3;
/// `FEATURES_LO` bit 4, `CONSUMED_DOORBELL`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`]: a host that finds a completion ring full sets the ring's
/// `producer_waiting` and waits for a [`RGFX_REG_CONSUMED`] write of the ring's offset, on ring 0's
/// completion ring and on a queue's created with [`RGFX_QUEUE_CONSUMED_DOORBELL`]. No host offers
/// it yet.
pub const RGFX_FEATURE_CONSUMED_DOORBELL: u32 = 1 << 4;
/// `FEATURES_LO` bit 5, `QUEUE_BARRIER`: the host decodes
/// [`RGFX_OP_QUEUE_BARRIER`](crate::ops::RGFX_OP_QUEUE_BARRIER). No host offers it yet.
pub const RGFX_FEATURE_QUEUE_BARRIER: u32 = 1 << 5;
/// `FEATURES_LO` bit 6, `CB_PARTS`: a queue's ring carries [`RGFX_CMD_COMMAND_BUFFER_PART`], and
/// the host decodes [`RGFX_OP_RENDER_RESET_STATE`](crate::ops::render::RGFX_OP_RENDER_RESET_STATE).
/// No host offers it yet.
pub const RGFX_FEATURE_CB_PARTS: u32 = 1 << 6;
/// `FEATURES_LO` bit 7, `PAGES_BUFFERS`: ring 0 carries [`RGFX_CMD_BUFFER_OVER_PAGES`], and a
/// `SURFACE_PAGES` may carry [`RGFX_SURFACE_PAGES_PRIVATE`]. No host offers it yet.
pub const RGFX_FEATURE_PAGES_BUFFERS: u32 = 1 << 7;
/// `FEATURES_LO` bit 8, `QUEUE_VECTORS`: the device has an MSI-X vector for each context slot
/// after vector 0, and a queue's completion ring raises its context's. No host offers it yet.
pub const RGFX_FEATURE_QUEUE_VECTORS: u32 = 1 << 8;
/// `FEATURES_LO` bit 11, `SURFACE_ORDER`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`]: the host decodes
/// [`RGFX_OP_SURFACE_ORDER`](crate::ops::RGFX_OP_SURFACE_ORDER), so a command buffer that uses an
/// IOSurface runs after the buffers of other queues committed before it that use the surface,
/// where one of the two writes it. Every host offers it.
pub const RGFX_FEATURE_SURFACE_ORDER: u32 = 1 << 11;
/// `FEATURES_LO` bit 12, `INDIRECT_COMMAND_BUFFERS`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`]: ring 0 carries [`RGFX_CMD_ICB_CREATE`], answered
/// [`RGFX_EVT_ICB_STATE`], and [`RGFX_CMD_ICB_WRITE`]. The host decodes both wherever it offers
/// the bit; the kext takes it where it answers the create, and its bundle makes indirect command
/// buffers only where it was taken.
pub const RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS: u32 = 1 << 12;
/// `FEATURES_LO` bit 13, `UNSHARED_QUEUES`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`]: a queue created with [`RGFX_QUEUE_UNSHARED`] is the
/// platform's queue that shares its GPU queue with no other command queue, so no other process's
/// work joins it there. The kext asks it for every queue of WindowServer's. Every host offers it:
/// where the platform shares no GPU queue between command queues (Metal, a Vulkan device that
/// keeps a queue of its own for each), the queue is the one any create makes.
pub const RGFX_FEATURE_UNSHARED_QUEUES: u32 = 1 << 13;
/// `FEATURES_LO` bit 14, `MANAGED_AT_OFFSET`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`]: ring 0 carries [`RGFX_CMD_BUFFER_MANAGED_AT_OFFSET`], a
/// `Managed` buffer whose CPU copy is placed in a heap over host memory, so no buffer imports
/// host memory of its own. Offered where a `Managed` buffer is two copies: where the host's GPU
/// does not share the CPU's memory.
pub const RGFX_FEATURE_MANAGED_AT_OFFSET: u32 = 1 << 14;
/// `STATUS`, RW: writing [`RGFX_STATUS_RESET`] resets the device; the guest writes
/// [`RGFX_STATUS_DRIVER_READY`] once it has read the rings' headers.
pub const RGFX_REG_STATUS: u32 = 0x010;
/// `REGION_SIZE_LO`, RO: BAR 2's size, its low 32 bits.
pub const RGFX_REG_REGION_SIZE_LO: u32 = 0x014;
/// `REGION_SIZE_HI`, RO: BAR 2's size, its high 32 bits.
pub const RGFX_REG_REGION_SIZE_HI: u32 = 0x018;
/// `DOORBELL`, WO: writing a ring's id wakes its consumer.
pub const RGFX_REG_DOORBELL: u32 = 0x020;
/// `DRIVER_FEATURES_LO`, RW: the feature bits 0 to 31 the guest's driver uses. A write takes the
/// bits written that `FEATURES_LO` offers and the host knows, and a read answers the bits taken.
/// A write while `STATUS` reads [`RGFX_STATUS_DRIVER_READY`] is ignored, and a reset clears both
/// halves. Bits 0 to 2 need no write: the host acts on them as it offers them. A guest that never
/// writes it (the option ROM, a kext built before it) gets the wire it had before.
pub const RGFX_REG_DRIVER_FEATURES_LO: u32 = 0x024;
/// `DRIVER_FEATURES_HI`, RW: feature bits 32 to 63, as [`RGFX_REG_DRIVER_FEATURES_LO`].
pub const RGFX_REG_DRIVER_FEATURES_HI: u32 = 0x028;
/// `CONSUMED`, WO: writing a completion ring's region offset wakes that ring's producer waiting
/// for room ([`RGFX_FEATURE_CONSUMED_DOORBELL`]). The host does not handle it yet: it reads 0 and
/// ignores a write, as at any offset it does not know.
pub const RGFX_REG_CONSUMED: u32 = 0x02c;
/// What `MAGIC` reads: "RGFX", little-endian.
pub const RGFX_MAGIC: u32 = 0x5846_4752;
/// What `VERSION` reads: the wire's version, which every change a guest built before it would
/// misread moves on (docs/graphics-kext/design.md, "The wire's version"). The kext and the option
/// ROM leave a device of any other version.
pub const RGFX_VERSION: u32 = 2;
/// `STATUS`: the device is reset, and takes no message.
pub const RGFX_STATUS_RESET: u32 = 0;
/// `STATUS`: the guest's driver has read the rings' headers; the device takes its messages.
pub const RGFX_STATUS_DRIVER_READY: u32 = 1;

/// Where the control submission ring is in the region: ring id [`RGFX_CONTROL_RING_ID`], guest
/// to host.
pub const RGFX_CONTROL_SUBMIT_OFFSET: u64 = 0x0_0000;
/// Where the control completion ring is in the region: host to guest, MSI-X vector
/// [`RGFX_CONTROL_VECTOR`].
pub const RGFX_CONTROL_COMPLETE_OFFSET: u64 = 0x1_0000;
/// The room each control ring has in the region, its header included.
pub const RGFX_CONTROL_RING_SPAN: u64 = 0x1_0000;
/// Where the guest's allocations start in the region: the framebuffer's scanout first
/// ([`RGFX_SCANOUT_SPAN`] bytes), then the accelerator kext's blocks
/// ([`RGFX_CONTEXT_ALLOCATIONS_OFFSET`]), then the cursor image at the region's end
/// ([`RGFX_CURSOR_SPAN`]). Everything from here to the region's end is the guest's to place.
pub const RGFX_ALLOCATIONS_OFFSET: u64 = 0x40_0000;
/// The room the framebuffer keeps for its scanout at [`RGFX_ALLOCATIONS_OFFSET`]: one 8K
/// (7680 x 4320) BGRA8 frame with 256-byte rows fits.
pub const RGFX_SCANOUT_SPAN: u64 = 0x800_0000;
/// The room the framebuffer keeps for its cursor image at the region's end: 256 x 256 BGRA8.
pub const RGFX_CURSOR_SPAN: u64 = 0x4_0000;
/// Where the accelerator kext's blocks start: the processes' rings, buffers, heaps and surfaces,
/// which the kext hands out and frees (each process's at its end), up to the cursor's span.
pub const RGFX_CONTEXT_ALLOCATIONS_OFFSET: u64 = RGFX_ALLOCATIONS_OFFSET + RGFX_SCANOUT_SPAN;
/// The object ids of context `c` (a process's, numbered from 1 by the kext) are
/// `c << RGFX_CONTEXT_OBJ_ID_SHIFT` and up, below the next context's: the guest bundle picks them
/// there, and `CONTEXT_CLOSE` drops the context's.
pub const RGFX_CONTEXT_OBJ_ID_SHIFT: u32 = 40;
/// How many processes hold the device at once: each holds one slot of the GPU address space,
/// which the kext hands out with its context.
pub const RGFX_CONTEXT_SLOTS: u32 = 32;
/// Where slot 0's GPU addresses start; slot `s`'s start [`RGFX_CONTEXT_VA_SPAN`] bytes a slot
/// later. The slots fill the top half of the Metal address space, `[2^46, 2^47)`, which no range
/// the host places itself reaches: the platform's runtime places its own buffers of Metal's space
/// lowest first from 2^32, and the Vulkan platform's model of Metal's layout places the host's
/// unpinned buffers and textures lowest first from 2^40, each as far as its live ranges reach,
/// which the host's memory bounds far below 2^46. A guest's pin meets only the guest's own
/// ranges then.
pub const RGFX_CONTEXT_VA_BASE: u64 = 0x4000_0000_0000;
/// The GPU addresses each slot picks from: 2 TiB, so a process's picks, which go forward through
/// its slot before they come round to a range given back, come round only after 2^26 picks of
/// the least span.
pub const RGFX_CONTEXT_VA_SPAN: u64 = 0x200_0000_0000;
/// The first texture resource id the host refuses: every process's textures share the host
/// device's one space of ids, and Metal ends the process for a texture placed at 16381 or more
/// (measured on an M4 and an M3 Max; metal2vulkan refuses it, as it imitates that).
pub const RGFX_RESOURCE_ID_LIMIT: u64 = 16381;
/// The texture resource ids in one chunk. The kext hands the processes their ids in chunks, from
/// one pool for the guest ([`RGFX_RESOURCE_ID_CHUNKS`]): chunk `k` holds ids `1 + k * 64` to
/// `64 + k * 64`, so a chunk's free ids fit in one 64-bit mask in the bundle.
pub const RGFX_RESOURCE_ID_CHUNK: u64 = 64;
/// How many chunks of texture resource ids the pool holds: every whole chunk below
/// [`RGFX_RESOURCE_ID_LIMIT`].
pub const RGFX_RESOURCE_ID_CHUNKS: u32 = 255;
/// The control submission ring's id, which the guest writes to `DOORBELL`.
pub const RGFX_CONTROL_RING_ID: u32 = 0;
/// The control completion ring's MSI-X vector.
pub const RGFX_CONTROL_VECTOR: u32 = 0;

/// The size of a ring's header ([`RgfxRingHeader`]); its byte ring follows it.
pub const RGFX_RING_HEADER_SIZE: u32 = 256;
/// The size of a control ring's byte ring: the largest power of two that fits in its span after
/// the header, 32 KiB. The rest of the span is unused.
pub const RGFX_CONTROL_RING_DATA_SIZE: u32 = 0x8000;

/// Where the device-info caps blob ([`RgfxDeviceInfo`]) sits in the region: the host writes it
/// once at the device's init, and the guest bundle reads it before it makes any queue. It lies in
/// the unused tail of the control completion ring's span (whose byte ring is a power of two, 32
/// KiB, smaller than the 64 KiB span), after the ring and before the guest's allocations, so it
/// adds no offset the rings or the allocations move for. It is a flat caps page, never a ring.
pub const RGFX_DEVICE_INFO_OFFSET: u64 = RGFX_CONTROL_COMPLETE_OFFSET
    + RGFX_RING_HEADER_SIZE as u64
    + RGFX_CONTROL_RING_DATA_SIZE as u64;

/// Where the video caps ([`RgfxVideoCaps`]) sit in the region: right after the device-info blob
/// (whose size is 1432 bytes and 64 per format), written with it at the device's init. A host
/// older than them leaves the bytes zero, which read as no codec.
pub const RGFX_VIDEO_CAPS_OFFSET: u64 =
    RGFX_DEVICE_INFO_OFFSET + 1432 + 64 * RGFX_DEVICE_INFO_FORMATS as u64;

/// A ring's header, at the start of the ring: each field in a 64-byte line of its own, so the
/// producer's and the consumer's writes never share a cache line.
#[repr(C)]
#[derive(Clone, Copy, Debug, FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct RgfxRingHeader {
    /// The producer's free-running byte count, written by the producer.
    pub producer: u64,
    /// On a queue's completion ring, the seqno of the last command buffer that finished with
    /// [`RGFX_CB_OK`], written by the host before the buffer's [`RgfxCbComplete`] is published:
    /// every buffer of the queue up to it has finished, its writes in the memory it names. A
    /// failed buffer leaves it, since the host may post a failure ahead of the queue's earlier
    /// buffers. 0 until one has finished, and on a host that predates the field. The guest's
    /// kext reads it to let the CPU locks of IOSurfaces drawn in place go on
    /// (docs/graphics-kext/design.md, "Surface locks").
    pub completed_seqno: u64,
    /// 1 while the host waits for room on a completion ring created to wake it
    /// ([`RGFX_FEATURE_CONSUMED_DOORBELL`]), written by the host; 0 on every other ring.
    pub producer_waiting: u32,
    /// The rest of `producer`'s line.
    pub reserved_14: [u8; 44],
    /// The consumer's free-running byte count, written by the consumer.
    pub consumer: u64,
    /// The rest of `consumer`'s line.
    pub reserved_48: [u8; 56],
    /// 1 while the consumer sleeps until a doorbell, written by the consumer.
    pub consumer_idle: u32,
    /// The rest of `consumer_idle`'s line.
    pub reserved_84: [u8; 60],
    /// 1 when the guest wants an interrupt, set by the guest on a completion ring and cleared by
    /// the host as it raises one; the host sets it again where the guest had taken every message
    /// as the host cleared it.
    pub armed: u32,
    /// On a completion ring created with [`RGFX_QUEUE_QUIET_SCHEDULED`], how many guest threads
    /// wait for a quiet record, written by the guest: nonzero makes every record raise. The host
    /// reads it on no other ring.
    pub raise_all: u32,
    /// The rest of `armed`'s line.
    pub reserved_c8: [u8; 56],
}

/// A pad to the ring's end: its `size` covers the bytes left, 8 or more, and it has no `seqno`.
pub const RGFX_MSG_PAD: u32 = 0;
/// Command `GET_DISPLAYS` ([`RgfxGetDisplays`]): answered by [`RGFX_EVT_DISPLAYS`].
pub const RGFX_CMD_GET_DISPLAYS: u32 = 1;
/// Command `SET_SCANOUT` ([`RgfxSetScanout`]): answered by [`RGFX_EVT_COMPLETE`].
pub const RGFX_CMD_SET_SCANOUT: u32 = 2;
/// Command `SET_CURSOR` ([`RgfxSetCursor`]): answered by [`RGFX_EVT_COMPLETE`].
pub const RGFX_CMD_SET_CURSOR: u32 = 3;
/// Command `MOVE_CURSOR` ([`RgfxMoveCursor`]): not answered.
pub const RGFX_CMD_MOVE_CURSOR: u32 = 4;
/// Command `SET_VBLANK` ([`RgfxSetVblank`]): answered by [`RGFX_EVT_COMPLETE`].
pub const RGFX_CMD_SET_VBLANK: u32 = 5;
/// Command `PRESENT_SURFACE` ([`RgfxPresentSurface`]): answered by [`RGFX_EVT_COMPLETE`].
pub const RGFX_CMD_PRESENT_SURFACE: u32 = 6;
/// Event `COMPLETE` ([`RgfxComplete`]): a command's answer, with its `seqno`.
pub const RGFX_EVT_COMPLETE: u32 = 0x100;
/// Event `DISPLAYS` ([`RgfxDisplays`]): `GET_DISPLAYS`'s answer, with its `seqno`.
pub const RGFX_EVT_DISPLAYS: u32 = 0x101;
/// Event `DISPLAY_CHANGED` ([`RgfxDisplayChanged`]): the guest asks `GET_DISPLAYS` again.
pub const RGFX_EVT_DISPLAY_CHANGED: u32 = 0x102;
/// Event `VBLANK` ([`RgfxVblank`]): a refresh of the display's host screen.
pub const RGFX_EVT_VBLANK: u32 = 0x103;

/// `COMPLETE`'s status: the command was done.
pub const RGFX_STATUS_OK: u32 = 0;
/// `COMPLETE`'s status: the device has no command of that type; nothing was done.
pub const RGFX_STATUS_UNKNOWN_COMMAND: u32 = 1;
/// `COMPLETE`'s status: the device has no display of that id; nothing was done.
pub const RGFX_STATUS_NO_DISPLAY: u32 = 2;
/// `COMPLETE`'s status: an argument is out of range (an image that does not lie inside the
/// guest's allocations, a stride or a size the device refuses, a message shorter than its
/// command); nothing was done.
pub const RGFX_STATUS_INVALID_ARGUMENT: u32 = 3;
/// `COMPLETE`'s status: the device holds nothing the command names (a `LIBRARY_LOOKUP` whose key
/// it holds no library for); nothing was done.
pub const RGFX_STATUS_NOT_FOUND: u32 = 4;
/// `COMPLETE`'s status: the device's GPU refused to make the object (a library whose bytes it
/// could not build); nothing was made.
pub const RGFX_STATUS_REFUSED: u32 = 5;
/// `COMPLETE`'s status: the answer does not fit the block the guest gave for it, whose first
/// bytes now say how many it needs ([`RgfxArgumentLayoutReply::needed`]); nothing else was
/// written.
pub const RGFX_STATUS_TOO_SMALL: u32 = 6;

/// Scanout pixels: 32-bit BGRA8, `kIO32BGRAPixelFormat`, the one format there is.
pub const RGFX_SCANOUT_BYTES_PER_PIXEL: u32 = 4;
/// The largest cursor image, in pixels each way.
pub const RGFX_CURSOR_MAX_SIZE: u32 = 256;
/// The displays the device has: display 0.
pub const RGFX_DISPLAY_COUNT: u32 = 1;

// ---------------------------------------------------------------------------
// Stage 3: the Metal-call stream (docs/graphics-kext/design.md, "Stage 3 wire
// protocol"). The guest's Metal GPU bundle serialises every Metal call into
// these messages; the host decodes them onto the library's `Platform`. Every
// guest object has a 64-bit `obj_id` the guest picks, unique per device; the
// host keeps an `obj_id` -> handle table. The guest also picks each resource's
// GPU address and the host pins it. The transport (rings, header, doorbell and
// interrupt handshake, `RgfxMsgHeader`) is the stage-2 one, unchanged.
// ---------------------------------------------------------------------------

/// Device-level object lifecycle (ring 0, which the accelerator kext posts for
/// the guest's processes). Creates with a guest-picked id or address are
/// fire-and-forget; pipeline creates are answered with reflection on the
/// control completion ring, and `QUEUE_CREATE` with `COMPLETE`.
/// Command `QUEUE_CREATE` ([`RgfxQueueCreate`]): make a submission+completion
/// ring pair for a command queue; answered by [`RGFX_EVT_COMPLETE`].
pub const RGFX_CMD_QUEUE_CREATE: u32 = 0x200;
/// Command `OBJECT_DESTROY` ([`RgfxObjectDestroy`]): drop the host handle for an
/// `obj_id` of any kind (the handle table knows the kind). Not answered.
pub const RGFX_CMD_OBJECT_DESTROY: u32 = 0x201;
/// Command `HEAP_CREATE` ([`RgfxHeapCreate`]): a device heap at a guest address.
pub const RGFX_CMD_HEAP_CREATE: u32 = 0x202;
/// Command `HEAP_OVER_HOST_MEMORY` ([`RgfxHeapOverHostMemory`]): a Shared heap
/// over a block of the region.
pub const RGFX_CMD_HEAP_OVER_HOST_MEMORY: u32 = 0x203;
/// Command `BUFFER_CREATE` ([`RgfxBufferCreate`]): a device buffer at a guest
/// address.
pub const RGFX_CMD_BUFFER_CREATE: u32 = 0x204;
/// Command `BUFFER_OVER_HOST_MEMORY` ([`RgfxBufferOverHostMemory`]): a Shared
/// buffer over a block of the region.
pub const RGFX_CMD_BUFFER_OVER_HOST_MEMORY: u32 = 0x205;
/// Command `BUFFER_AT_OFFSET` ([`RgfxBufferAtOffset`]): a buffer at an offset in
/// a heap.
pub const RGFX_CMD_BUFFER_AT_OFFSET: u32 = 0x206;
/// Command `TEXTURE_CREATE`: a texture from a descriptor. Struct in a later
/// increment (carries the texture descriptor schema).
pub const RGFX_CMD_TEXTURE_CREATE: u32 = 0x207;
/// Command `TEXTURE_FROM_BUFFER`: a texture over a buffer. Struct later.
pub const RGFX_CMD_TEXTURE_FROM_BUFFER: u32 = 0x208;
/// Command `TEXTURE_FROM_IOSURFACE` ([`RgfxTextureFromIOSurface`]): a texture
/// over a guest process's copy of an IOSurface plane, named by the surface's
/// global id.
pub const RGFX_CMD_TEXTURE_FROM_IOSURFACE: u32 = 0x209;
/// Command `SAMPLER_CREATE` ([`RgfxSamplerCreate`]): a sampler from a descriptor, answered by
/// [`RGFX_EVT_SAMPLER_STATE`] with the `gpuResourceID` the host's device numbers it by.
pub const RGFX_CMD_SAMPLER_CREATE: u32 = 0x20a;
/// Command `DEPTH_STENCIL_CREATE`: a depth/stencil state from a descriptor.
/// Struct later.
pub const RGFX_CMD_DEPTH_STENCIL_CREATE: u32 = 0x20b;
/// Command `LIBRARY_CREATE` ([`RgfxLibraryCreate`]): a library from `.metallib`/AIR bytes,
/// forwarded unchanged; answered by [`RGFX_EVT_COMPLETE`].
pub const RGFX_CMD_LIBRARY_CREATE: u32 = 0x20c;
/// Command `RENDER_PIPELINE_CREATE`: a render pipeline from a descriptor;
/// answered by [`RGFX_EVT_PIPELINE_REFLECTION`]. Struct later.
pub const RGFX_CMD_RENDER_PIPELINE_CREATE: u32 = 0x20d;
/// Command `COMPUTE_PIPELINE_CREATE`: a compute pipeline; answered by
/// [`RGFX_EVT_PIPELINE_REFLECTION`]. Struct later.
pub const RGFX_CMD_COMPUTE_PIPELINE_CREATE: u32 = 0x20e;
/// Command `EVENT_CREATE` ([`RgfxObjectCreate`]): a Metal event.
pub const RGFX_CMD_EVENT_CREATE: u32 = 0x20f;
/// Command `SHARED_EVENT_CREATE` ([`RgfxSharedEventCreate`]): a shared event
/// with an initial value.
pub const RGFX_CMD_SHARED_EVENT_CREATE: u32 = 0x210;
/// Command `FENCE_CREATE` ([`RgfxObjectCreate`]): a Metal fence.
pub const RGFX_CMD_FENCE_CREATE: u32 = 0x211;
/// Command `LIBRARY_FROM_REGION` ([`RgfxLibraryFromRegion`]): one chunk of a
/// library's `.metallib`/AIR bytes the guest placed in a block of the device's
/// region, for a library larger than one ring message. The library's last chunk is
/// answered by [`RGFX_EVT_COMPLETE`]; the others are not answered.
pub const RGFX_CMD_LIBRARY_FROM_REGION: u32 = 0x212;
/// Command `LIBRARY_LOOKUP` ([`RgfxLibraryLookup`]): a library by its content key, from the
/// libraries the device made for any process of the guest; answered by [`RGFX_EVT_COMPLETE`].
pub const RGFX_CMD_LIBRARY_LOOKUP: u32 = 0x213;
/// Command `BUFFER_DID_MODIFY_RANGE` ([`RgfxBufferDidModifyRange`]): the guest wrote `length`
/// bytes at `location` of a `Managed` buffer's CPU copy (`didModifyRange:`). Every command buffer
/// of the context committed after it, on any queue, sees them: ring 0 is carried out before any
/// later command buffer. Not answered; a range past the buffer, or a buffer not `Managed`, is
/// refused.
pub const RGFX_CMD_BUFFER_DID_MODIFY_RANGE: u32 = 0x214;
/// Command `MANAGED_TEXTURE_CREATE` ([`RgfxManagedTextureCreate`]): a `Managed` texture, whose
/// CPU copy is a Shared buffer of the sender's over its own region block, in the linear layout
/// [`RgfxManagedTextureCreate`] describes. Not answered.
pub const RGFX_CMD_MANAGED_TEXTURE_CREATE: u32 = 0x215;
/// Command `TEXTURE_REPLACE_REGION` ([`RgfxTextureReplaceRegion`]): the guest wrote a region of
/// a level and slice of a `Managed` texture's CPU copy (`replaceRegion:`). Every command buffer of
/// the context committed after it, on any queue, sees them. Not answered.
pub const RGFX_CMD_TEXTURE_REPLACE_REGION: u32 = 0x216;
/// Command `TEXTURE_VIEW` ([`RgfxTextureView`]): a view of a texture the guest made, over the
/// same texels in another pixel format, and, where the guest named them, another type, levels,
/// slices and swizzle (`newTextureViewWithPixelFormat:` and its longer forms). Not answered; a
/// view the platform refuses, as Metal refuses it, makes nothing.
pub const RGFX_CMD_TEXTURE_VIEW: u32 = 0x217;
/// Command `HEAP_TEXTURE_SIZE_AND_ALIGN` ([`RgfxHeapTextureSizeAndAlign`]): the size and alignment
/// a texture of a shape takes in a heap (`heapTextureSizeAndAlignWithDescriptor:`), as the host's
/// device answers it; answered by [`RGFX_EVT_SIZE_AND_ALIGN`]. Makes nothing.
pub const RGFX_CMD_HEAP_TEXTURE_SIZE_AND_ALIGN: u32 = 0x218;
/// Command `ARGUMENT_LAYOUT` ([`RgfxArgumentLayoutQuery`]): the layout of an argument buffer
/// (`MTLArgumentEncoder`'s), from argument descriptors (`newArgumentEncoderWithArguments:`) or
/// from a function's buffer argument (`newArgumentEncoderWithBufferIndex:`), written into a block
/// of the sender's as [`RgfxArgumentLayoutReply`] says, then answered by [`RGFX_EVT_COMPLETE`].
/// Nothing is made: the guest's encoder writes the arguments where the layout puts them.
pub const RGFX_CMD_ARGUMENT_LAYOUT: u32 = 0x219;
/// Command `SHARED_EVENT_SET_VALUE` ([`RgfxSharedEventSetValue`]): the CPU sets a shared event's
/// value (`setSignaledValue:`), which only rises, as Metal's does. Not answered.
pub const RGFX_CMD_SHARED_EVENT_SET_VALUE: u32 = 0x21a;
/// Command `LINEAR_TEXTURE_FROM_BUFFER` ([`RgfxLinearTextureFromBuffer`]): a texture over a
/// buffer as `newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:` makes one, a 2D
/// array's among them, which `TEXTURE_FROM_BUFFER`'s form does not make. Not answered; a texture
/// the platform refuses, as Metal refuses it, makes nothing.
pub const RGFX_CMD_LINEAR_TEXTURE_FROM_BUFFER: u32 = 0x21b;
/// Command `BUFFER_OVER_SURFACE_PAGES` ([`RgfxBufferOverSurfacePages`]), where the device has
/// [`RGFX_FEATURE_SURFACE_PAGES`]: a Shared buffer of `length` bytes over an IOSurface's own guest
/// pages, from their first byte: those the `SURFACE_PAGES` of the shared block at `block_offset`
/// named, whose first `block_size` bytes, the part the processes map, the sender's context
/// holds. The buffer lies over that part as a `BUFFER_OVER_HOST_MEMORY` buffer lies over its
/// block: a `TEXTURE_FROM_IOSURFACE` over it finds its writers there, and the block's
/// `BLOCK_RELEASE` is refused while the buffer lives. Not answered; refused, making nothing,
/// where the block has no view of its pages or `length` passes them.
pub const RGFX_CMD_BUFFER_OVER_SURFACE_PAGES: u32 = 0x21c;
/// Command `VIDEO_SESSION_CREATE` ([`RgfxVideoSessionCreate`]): a video session of one codec and
/// direction, which command buffers' video encoders name
/// ([`RGFX_OP_ENCODER_BEGIN_VIDEO`](crate::ops::RGFX_OP_ENCODER_BEGIN_VIDEO)); `OBJECT_DESTROY`
/// drops it. What the host's hardware takes is in the video caps ([`RgfxVideoCaps`]). Not
/// answered: where the host has no such session, the command buffers that name it fail.
pub const RGFX_CMD_VIDEO_SESSION_CREATE: u32 = 0x21d;
/// Command `BUFFER_OVER_PAGES` ([`RgfxBufferOverPages`]), where the guest took
/// [`RGFX_FEATURE_PAGES_BUFFERS`]: a buffer over a range of a pages block of the sender's context,
/// the guest's own pages in place. No host offers the feature yet, and a host without it refuses
/// the message as one it does not know.
pub const RGFX_CMD_BUFFER_OVER_PAGES: u32 = 0x21e;
/// Command `ICB_CREATE` ([`RgfxIcbCreate`]), where the guest took
/// [`RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS`]: indirect command buffer `obj_id`, made by the
/// platform's `newIndirectCommandBufferWithDescriptor:maxCommandCount:options:` from the guest's
/// descriptor. Answered at once by [`RGFX_EVT_ICB_STATE`], with the `gpuResourceID` the host's
/// device numbers it by, which the guest's argument buffers hold, so a kernel that encodes
/// through one reaches the host's buffer. `OBJECT_DESTROY` drops it.
pub const RGFX_CMD_ICB_CREATE: u32 = 0x228;
/// Command `ICB_WRITE` ([`RgfxIcbWrite`], then its records, [`crate::icb`]), where the guest took
/// [`RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS`]: the CPU's encodes into indirect command buffer
/// `obj_id`, one record a setter call, in call order. The host carries each out on the
/// platform's buffer, the one copy of its commands, which the GPU's own encodes also write.
/// Ring 0 is carried out before any command buffer the process posts after it, so a command
/// buffer sees every encode posted before it. Not answered: a record the host cannot carry out
/// is skipped alone.
pub const RGFX_CMD_ICB_WRITE: u32 = 0x229;
/// Command `BUFFER_MANAGED_AT_OFFSET` ([`RgfxBufferManagedAtOffset`]), where the guest took
/// [`RGFX_FEATURE_MANAGED_AT_OFFSET`]: a `Managed` buffer at the guest's address, its CPU copy a
/// `Shared` buffer placed at an offset in a heap over host memory of the sender's
/// ([`RGFX_CMD_HEAP_OVER_HOST_MEMORY`]), whose bytes are the guest's copy. A host without the
/// feature refuses the message as one it does not know.
pub const RGFX_CMD_BUFFER_MANAGED_AT_OFFSET: u32 = 0x22b;

/// `CONTEXT_CLOSE` ([`RgfxContextClose`]), on ring 0 from the accelerator kext: the process that
/// held context `context` closed the device or died. The host ends the context's command queues
/// and drops every object whose id lies in its range ([`RGFX_CONTEXT_OBJ_ID_SHIFT`]), then answers
/// [`RGFX_EVT_COMPLETE`] once every command buffer of the context it had committed has finished;
/// the kext destroyed its queues before it, and frees its blocks once it is answered, and only
/// then.
pub const RGFX_CMD_CONTEXT_CLOSE: u32 = 0x2f0;
/// `RESOURCE_ID_CHUNK` ([`RgfxResourceIdChunk`]), on ring 0 from the accelerator kext: the kext
/// gave context `context` chunk `chunk` of the texture resource ids ([`RGFX_RESOURCE_ID_CHUNK`]),
/// which it holds until its `CONTEXT_CLOSE`. Posted before the process learns of the chunk, so it
/// comes before every create naming one of its ids. Not answered. A texture create whose id lies
/// in no chunk of its context is refused.
pub const RGFX_CMD_RESOURCE_ID_CHUNK: u32 = 0x2f1;
/// `BLOCK_RELEASE` ([`RgfxBlockRelease`]), on ring 0 from the accelerator kext: context `context`
/// freed its region block of `size` bytes at `offset`, which the kext gives out again only once
/// this is answered [`RGFX_EVT_COMPLETE`] with [`RGFX_STATUS_OK`]. The host answers
/// [`RGFX_STATUS_REFUSED`] at once where an object it holds, or a live queue's ring, still lies in
/// the block; otherwise it answers OK once every command buffer of the context it had committed
/// has finished. Posted after every message the process posted before its free, and nothing the
/// process posts after it can name the block. A block over an IOSurface's pages with a device
/// cache has the cache copied into the pages first, where a command buffer wrote it since the last
/// copy, unless [`RGFX_BLOCK_RELEASE_NO_CLIENT`] says no process can read them.
pub const RGFX_CMD_BLOCK_RELEASE: u32 = 0x2f2;
/// `BLOCK_GRANT` ([`RgfxBlockGrant`]), on ring 0 from the accelerator kext: it gave context
/// `context` the region block of `size` bytes at `offset`. Posted before the process learns of the
/// block, so it comes before every message naming it. Not answered. A span a message names, on
/// ring 0 or on a queue's ring, must lie in one block its context holds: from its grant until its
/// `BLOCK_RELEASE` or the context's `CONTEXT_CLOSE`. A block lies in the processes' blocks
/// ([`RGFX_CONTEXT_ALLOCATIONS_OFFSET`] up to the cursor's span), or in the scanout's span, which
/// the kext grants to a context whose process draws the display into it; a grant of a span that
/// overlaps another context's block takes it from that context.
pub const RGFX_CMD_BLOCK_GRANT: u32 = 0x2f3;
/// `BLOCK_SHARE` ([`RgfxBlockGrant`]), on ring 0 from the accelerator kext: context `context`
/// holds the shared region block of `size` bytes at `offset` too, beside the contexts that hold
/// it already: the one block of an IOSurface's plane every process that holds the surface draws
/// and samples (`guest/bundle/src/RgfxIOSurface.h`). Posted before the process learns of the
/// block. Not answered. A span a message names may lie in it from then until the context's
/// `BLOCK_UNSHARE` or `CONTEXT_CLOSE`. A share of a span that meets a block other than a shared
/// one of the same offset and size takes the span from its holders, as a grant does.
pub const RGFX_CMD_BLOCK_SHARE: u32 = 0x2f4;
/// `BLOCK_UNSHARE` ([`RgfxBlockGrant`]), on ring 0 from the accelerator kext: context `context`
/// no longer holds the shared block of `size` bytes at `offset`, which other contexts still hold.
/// Posted after every message the process posted before it let the block go. Not answered. The
/// block's last holder lets it go with a `BLOCK_RELEASE`, which ends every hold of it, and is
/// refused while an object of any context lies in the block.
pub const RGFX_CMD_BLOCK_UNSHARE: u32 = 0x2f5;
/// `SURFACE_PAGES` ([`RgfxSurfacePages`]), on ring 0 from the accelerator kext, where the device
/// has [`RGFX_FEATURE_SURFACE_PAGES`]: the guest RAM an IOSurface's `length` bytes lie in, whose
/// shared block of `block_size` bytes at `block_offset` the kext made for the GPU to use the
/// surface's own pages in place: `run_count` [`RgfxPageRun`]s, in the surface's byte order, at
/// region offset `runs_offset`, inside the block, past the part the processes map. Posted before
/// the block's first `BLOCK_SHARE`; the kext keeps the pages wired while a process holds the
/// block. Answered [`RGFX_EVT_COMPLETE`]: [`RGFX_STATUS_OK`] once the host has mapped the pages
/// as one view, which `BUFFER_OVER_SURFACE_PAGES` buffers lie over until the block's
/// `BLOCK_RELEASE`; [`RGFX_STATUS_REFUSED`] where it made none, for a run that is not whole pages
/// from a page of guest RAM, runs whose lengths do not sum to `length`, or a view the host could
/// not map, and [`RGFX_STATUS_INVALID_ARGUMENT`] where the runs do not lie in the block. The kext
/// gives a surface whose pages were refused a block that holds a copy of them instead. Where the
/// device has [`RGFX_FEATURE_SURFACE_CACHE`], an [`RgfxSurfacePagesCache`] may follow the message
/// at [`RGFX_SURFACE_PAGES_CACHE_OFFSET`], whose [`RGFX_SURFACE_PAGES_CACHED`] gives the block a
/// device cache.
pub const RGFX_CMD_SURFACE_PAGES: u32 = 0x2f6;
/// `SURFACE_DOWNLOAD` ([`RgfxSurfaceDownload`]), on ring 0 from the accelerator kext, where the
/// device has [`RGFX_FEATURE_SURFACE_CACHE`]: a CPU lock of the IOSurface whose cached block is at
/// `block_offset` follows a GPU write of it, and the command buffers that wrote it have finished.
/// The host copies the surface's device cache into its pages once every command buffer that used
/// the cache before this message was taken has finished, and answers [`RGFX_EVT_COMPLETE`] with
/// [`RGFX_STATUS_OK`] once the pages hold it: at once where no command buffer wrote the cache
/// since it was last copied either way, or the block has no cache.
pub const RGFX_CMD_SURFACE_DOWNLOAD: u32 = 0x2f7;

/// Command `COMMAND_BUFFER` (a per-queue submission ring): one command buffer,
/// its `obj_id` followed by the encoder-op byte stream. Struct and op stream in
/// a later increment.
pub const RGFX_CMD_COMMAND_BUFFER: u32 = 0x300;

/// Command `COMMAND_BUFFER_IN_REGION` ([`RgfxCommandBufferInRegion`], a per-queue submission
/// ring): one command buffer too large for its queue's ring, whose whole `COMMAND_BUFFER` message
/// lies in a block of the process's own. The host takes the message from the block in place of
/// the ring, with this reference's `seqno`, and carries it out as it does one from the ring; the
/// completion names that seqno. The process keeps the block until the buffer completes.
pub const RGFX_CMD_COMMAND_BUFFER_IN_REGION: u32 = 0x301;

/// Command `COMMAND_BUFFER_PART` ([`RgfxCommandBufferPart`], a per-queue submission ring), where
/// the host offers [`RGFX_FEATURE_CB_PARTS`]: one part of a command buffer's ops, in a block of the
/// process's own; the part flagged [`RGFX_CB_PART_LAST`] commits the buffer, and its seqno is the
/// one the completion names. No host offers the feature yet, and a host without it refuses the
/// message as one the ring may not carry.
pub const RGFX_CMD_COMMAND_BUFFER_PART: u32 = 0x302;
/// [`RgfxCommandBufferPart::flags`]: the last part; the host commits the buffer after it.
pub const RGFX_CB_PART_LAST: u32 = 1 << 0;
/// [`RgfxCommandBufferPart::flags`]: drop the buffer uncommitted, with no completion;
/// `region_offset` and `byte_length` are 0.
pub const RGFX_CB_PART_ABORT: u32 = 1 << 1;
/// How many buffers a queue holds sent in part but not yet committed, under
/// [`RGFX_FEATURE_CB_PARTS`]: a part past it fails its buffer at its last part.
pub const RGFX_CB_PARTS_MAX_PARTIAL: u32 = 64;

/// Event `CB_COMPLETE` ([`RgfxCbComplete`], a per-queue completion ring): a
/// command buffer finished, with its status and GPU times.
pub const RGFX_EVT_CB_COMPLETE: u32 = 0x400;
/// Event `EVENT_SIGNALLED` ([`RgfxEventSignalled`]): a shared event reached a
/// value.
pub const RGFX_EVT_EVENT_SIGNALLED: u32 = 0x401;
/// Event `PIPELINE_REFLECTION`: a pipeline create's answer, with the reflection
/// the guest needs. Struct in a later increment.
pub const RGFX_EVT_PIPELINE_REFLECTION: u32 = 0x402;
/// Event `SIZE_AND_ALIGN` ([`RgfxSizeAndAlign`], on the control completion ring): a
/// `HEAP_TEXTURE_SIZE_AND_ALIGN`'s answer.
pub const RGFX_EVT_SIZE_AND_ALIGN: u32 = 0x403;
/// Event `SAMPLER_STATE` ([`RgfxSamplerState`], on the control completion ring): a
/// `SAMPLER_CREATE`'s answer.
pub const RGFX_EVT_SAMPLER_STATE: u32 = 0x404;
/// Event `CB_SCHEDULED` ([`RgfxCbScheduled`], a per-queue completion ring): a command buffer is
/// scheduled, as Metal's becomes `Scheduled`: every wait of it is satisfied and the host's
/// platform has submitted its work. Posted ahead of the buffer's [`RgfxCbComplete`], once.
pub const RGFX_EVT_CB_SCHEDULED: u32 = 0x405;
/// Event `CB_SUBMITTED` ([`RgfxCbSubmitted`], a per-queue completion ring), where the device has
/// [`RGFX_FEATURE_CB_SUBMITTED`]: a command buffer that asked for it
/// ([`RGFX_OP_SUBMITTED`](crate::ops::RGFX_OP_SUBMITTED)) is submitted, as Metal's is when its
/// `commitAndWaitUntilSubmitted` returns. Posted once, ahead of the buffer's [`RgfxCbScheduled`]
/// and [`RgfxCbComplete`]: as soon as the host takes it, before it carries out its ops, where it
/// waits for no event, the host fills no device cache it uses first and no buffer before it on its
/// queue waits; else once its waits are satisfied, at its scheduling.
pub const RGFX_EVT_CB_SUBMITTED: u32 = 0x406;
/// Event `ICB_STATE` ([`RgfxIcbState`], on the control completion ring): an `ICB_CREATE`'s
/// answer.
pub const RGFX_EVT_ICB_STATE: u32 = 0x407;

/// A command buffer's status in [`RgfxCbComplete`]: finished with no error.
pub const RGFX_CB_OK: u32 = 0;
/// A command buffer's status: a GPU page fault (IOGPU page fault).
pub const RGFX_CB_PAGE_FAULT: u32 = 1;
/// A command buffer's status: the GPU hung and the work timed out.
pub const RGFX_CB_TIMEOUT: u32 = 2;
/// A command buffer's status: an internal error (a GPU recovery made this work
/// an innocent victim).
pub const RGFX_CB_INTERNAL: u32 = 3;
/// A command buffer's status: the queue's submissions are ignored after a second
/// fault (access revoked).
pub const RGFX_CB_ACCESS_REVOKED: u32 = 4;
/// A command buffer's status: a resource the work named was invalid.
pub const RGFX_CB_INVALID_RESOURCE: u32 = 5;

/// Every message's header.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxMsgHeader {
    /// The message's type: an `RGFX_CMD_*`, an `RGFX_EVT_*` or [`RGFX_MSG_PAD`].
    pub r#type: u32,
    /// The message's bytes, this header included: a multiple of 8.
    pub size: u32,
    /// A command's number, from 1 up on each ring, which the guest picks; an answer carries its
    /// command's, and an event that answers nothing carries 0.
    pub seqno: u64,
}

/// `GET_DISPLAYS`: no payload.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxGetDisplays {
    /// [`RGFX_CMD_GET_DISPLAYS`].
    pub header: RgfxMsgHeader,
}

/// `SET_SCANOUT`: the display shows `height` rows of `width` BGRA8 pixels, each row `stride`
/// bytes after the one before, from `offset` in the region. A `width` of 0 blanks the display.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSetScanout {
    /// [`RGFX_CMD_SET_SCANOUT`].
    pub header: RgfxMsgHeader,
    /// The display's id.
    pub display: u32,
    /// Pixels a row; 0 blanks the display.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// Bytes from one row's start to the next's: a multiple of 4, at least `width * 4`.
    pub stride: u32,
    /// Where the first row starts in the region: a multiple of 4, the whole scanout in the
    /// scanout's room ([`RGFX_SCANOUT_SPAN`] bytes at [`RGFX_ALLOCATIONS_OFFSET`]).
    pub offset: u64,
}

/// `SET_CURSOR`: the display's cursor is `height` rows of `width` premultiplied BGRA8 pixels,
/// `width * 4` bytes a row, from `offset` in the region, its hot spot at (`hot_x`, `hot_y`). A
/// `width` of 0 removes the image.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSetCursor {
    /// [`RGFX_CMD_SET_CURSOR`].
    pub header: RgfxMsgHeader,
    /// The display's id.
    pub display: u32,
    /// Pixels a row, up to [`RGFX_CURSOR_MAX_SIZE`]; 0 removes the image.
    pub width: u32,
    /// Rows, up to [`RGFX_CURSOR_MAX_SIZE`].
    pub height: u32,
    /// The hot spot's column, below `width`.
    pub hot_x: u32,
    /// The hot spot's row, below `height`.
    pub hot_y: u32,
    /// Keeps `offset` 8-aligned; 0.
    pub reserved_24: u32,
    /// Where the image starts in the region: a multiple of 4, in the guest's allocations.
    pub offset: u64,
}

/// `MOVE_CURSOR`: the cursor's hot spot is at (`x`, `y`) in the display's pixels, and shows
/// where `visible` is not 0.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxMoveCursor {
    /// [`RGFX_CMD_MOVE_CURSOR`].
    pub header: RgfxMsgHeader,
    /// The display's id.
    pub display: u32,
    /// The hot spot's column.
    pub x: i32,
    /// The hot spot's row.
    pub y: i32,
    /// Whether the cursor shows: 0 hides it.
    pub visible: u32,
}

/// `SET_VBLANK`: the device sends `VBLANK` at each refresh of the display's host screen while
/// `enabled` is not 0.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSetVblank {
    /// [`RGFX_CMD_SET_VBLANK`].
    pub header: RgfxMsgHeader,
    /// The display's id.
    pub display: u32,
    /// Whether `VBLANK` is sent: 0 stops it.
    pub enabled: u32,
}

/// `PRESENT_SURFACE`: the display shows the IOSurface of global ID `iosurface_id` (plane 0), the
/// display pipe's swap, in place of the scanout, until the next `PRESENT_SURFACE`; 0 shows the
/// scanout again. The host shows the texture the device's decoder made over that surface
/// (`TEXTURE_FROM_IOSURFACE`), and again each time a command buffer completes while it is shown,
/// so the GPU's writes reach the window. Answered [`RGFX_STATUS_INVALID_ARGUMENT`] where the
/// decoder holds no texture over that surface: the display keeps what it showed.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxPresentSurface {
    /// [`RGFX_CMD_PRESENT_SURFACE`].
    pub header: RgfxMsgHeader,
    /// The display's id.
    pub display: u32,
    /// The IOSurface's global ID; 0 shows the scanout.
    pub iosurface_id: u32,
    /// The display pipe's transaction ID, for the host's log.
    pub transaction_id: u32,
    /// Pads the message to a multiple of 8; 0.
    pub reserved_1c: u32,
}

/// Where a `PRESENT_SURFACE` message's [`RgfxPresentSurfaceGen`] starts: right after its
/// [`RgfxPresentSurface`]. A message whose `size` is at least
/// `RGFX_PRESENT_SURFACE_GEN_OFFSET + size_of::<RgfxPresentSurfaceGen>()` carries it.
pub const RGFX_PRESENT_SURFACE_GEN_OFFSET: u32 = 32;

/// The CPU's generation of the presented IOSurface, where its block has a device cache
/// ([`RGFX_SURFACE_PAGES_CACHED`]), after the [`RgfxPresentSurface`] at
/// [`RGFX_PRESENT_SURFACE_GEN_OFFSET`]: as an `RGFX_OP_SURFACE_USE`'s, the display shows the
/// pages as of that generation. A present without it shows the cache as it is.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxPresentSurfaceGen {
    /// The kext's count of the surface's read-write CPU locks.
    pub cpu_gen: u64,
}

/// `COMPLETE`: a command's answer.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComplete {
    /// [`RGFX_EVT_COMPLETE`], with the command's `seqno`.
    pub header: RgfxMsgHeader,
    /// An `RGFX_STATUS_*`: 0 for success.
    pub status: u32,
    /// Pads the message to a multiple of 8; 0.
    pub reserved_14: u32,
}

/// `DISPLAYS`: `GET_DISPLAYS`'s answer. `count` [`RgfxDisplayEntry`] records follow it, at
/// [`RGFX_DISPLAYS_ENTRIES_OFFSET`], each followed by its `mode_count` [`RgfxModeEntry`] records;
/// the message's `size` is rounded up to a multiple of 8 after the last. The struct is the
/// message's start: C's `sizeof` pads it past `count`, where the first record already is.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RgfxDisplays {
    /// [`RGFX_EVT_DISPLAYS`], with `GET_DISPLAYS`' `seqno`.
    pub header: RgfxMsgHeader,
    /// The displays that follow.
    pub count: u32,
}

/// Where a `DISPLAYS` message's first display record starts: right after `count`.
pub const RGFX_DISPLAYS_ENTRIES_OFFSET: u32 = 20;

/// One display in a `DISPLAYS` message.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxDisplayEntry {
    /// The display's id, which the commands name.
    pub id: u32,
    /// 1 where a display is connected, else 0.
    pub connected: u32,
    /// The index of the mode the display prefers, among the modes that follow.
    pub preferred_mode: u32,
    /// The modes that follow this record.
    pub mode_count: u32,
}

/// One mode of a display in a `DISPLAYS` message.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxModeEntry {
    /// Pixels a row.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// The refresh rate, in thousandths of a hertz.
    pub refresh_mhz: u32,
}

/// `DISPLAY_CHANGED`: what the display offers has changed; the guest asks `GET_DISPLAYS` again.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxDisplayChanged {
    /// [`RGFX_EVT_DISPLAY_CHANGED`], `seqno` 0.
    pub header: RgfxMsgHeader,
    /// The display's id.
    pub display: u32,
    /// Pads the message to a multiple of 8; 0.
    pub reserved_14: u32,
}

/// `VBLANK`: the display's host screen refreshed.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVblank {
    /// [`RGFX_EVT_VBLANK`], `seqno` 0.
    pub header: RgfxMsgHeader,
    /// The display's id.
    pub display: u32,
    /// Keeps `count` 8-aligned; 0.
    pub reserved_14: u32,
    /// The display's refreshes since it was made, this one included.
    pub count: u64,
    /// The host's monotonic clock at the refresh, in nanoseconds.
    pub host_time_ns: u64,
}

// The layout the tables of the design give.
const _: () = {
    use core::mem::{offset_of, size_of};
    assert!(size_of::<RgfxRingHeader>() == RGFX_RING_HEADER_SIZE as usize);
    assert!(offset_of!(RgfxRingHeader, consumer) == 0x40);
    assert!(offset_of!(RgfxRingHeader, consumer_idle) == 0x80);
    assert!(offset_of!(RgfxRingHeader, armed) == 0xc0);
    assert!(offset_of!(RgfxRingHeader, producer_waiting) == 0x10);
    assert!(offset_of!(RgfxRingHeader, raise_all) == 0xc4);
    assert!(size_of::<RgfxMsgHeader>() == 16);
    assert!(size_of::<RgfxGetDisplays>() == 16);
    assert!(size_of::<RgfxSetScanout>() == 40);
    assert!(size_of::<RgfxSetCursor>() == 48);
    assert!(size_of::<RgfxMoveCursor>() == 32);
    assert!(size_of::<RgfxSetVblank>() == 24);
    assert!(size_of::<RgfxPresentSurface>() == 32);
    assert!(size_of::<RgfxComplete>() == 24);
    assert!(offset_of!(RgfxDisplays, count) + 4 == RGFX_DISPLAYS_ENTRIES_OFFSET as usize);
    assert!(size_of::<RgfxDisplayEntry>() == 16);
    assert!(size_of::<RgfxModeEntry>() == 12);
    assert!(size_of::<RgfxDisplayChanged>() == 24);
    assert!(size_of::<RgfxVblank>() == 40);
    assert!(RGFX_CONTROL_RING_DATA_SIZE.is_power_of_two());
    assert!(
        RGFX_RING_HEADER_SIZE as u64 + RGFX_CONTROL_RING_DATA_SIZE as u64 <= RGFX_CONTROL_RING_SPAN
    );
    assert!(RGFX_CONTROL_COMPLETE_OFFSET + RGFX_CONTROL_RING_SPAN <= RGFX_ALLOCATIONS_OFFSET);
    assert!(RGFX_ALLOCATIONS_OFFSET.is_multiple_of(0x1000));
    assert!(RGFX_CONTEXT_ALLOCATIONS_OFFSET.is_multiple_of(0x1000));
    assert!(size_of::<RgfxContextClose>() == 24);
    assert!(size_of::<RgfxResourceIdChunk>() == 24);
    assert!(size_of::<RgfxCommandBufferInRegion>() == 32);
    assert!(size_of::<RgfxBlockRelease>() == 40);
    assert!(size_of::<RgfxBlockGrant>() == 40);
    assert!(size_of::<RgfxPageRun>() == 16);
    assert!(size_of::<RgfxSurfacePages>() == 56);
    assert!(size_of::<RgfxSurfacePages>() == RGFX_SURFACE_PAGES_CACHE_OFFSET as usize);
    assert!(size_of::<RgfxSurfacePagesCache>() == 8);
    assert!(size_of::<RgfxSurfaceDownload>() == 24);
    assert!(size_of::<RgfxPresentSurface>() == RGFX_PRESENT_SURFACE_GEN_OFFSET as usize);
    assert!(size_of::<RgfxPresentSurfaceGen>() == 8);
    assert!(size_of::<RgfxBufferOverSurfacePages>() == 56);
    assert!(size_of::<RgfxBufferOverPages>() == 72);
    assert!(offset_of!(RgfxBufferOverPages, options) == 64);
    // Every chunk's ids lie below the limit (the last chunk's last id is CHUNKS * CHUNK), and
    // no whole chunk more fits there.
    assert!(RGFX_RESOURCE_ID_CHUNKS as u64 * RGFX_RESOURCE_ID_CHUNK < RGFX_RESOURCE_ID_LIMIT);
    assert!(
        1 + (RGFX_RESOURCE_ID_CHUNKS as u64 + 1) * RGFX_RESOURCE_ID_CHUNK > RGFX_RESOURCE_ID_LIMIT
    );
    // The bundle keeps a chunk's free ids in one 64-bit mask.
    assert!(RGFX_RESOURCE_ID_CHUNK == 64);
    // Every slot's GPU addresses lie below the Metal address space's end, 2^47.
    assert!(RGFX_CONTEXT_VA_BASE + RGFX_CONTEXT_SLOTS as u64 * RGFX_CONTEXT_VA_SPAN <= 1u64 << 47);
    // ... and at 2^46 or above, which the host's own ranges never reach (RGFX_CONTEXT_VA_BASE).
    assert!(RGFX_CONTEXT_VA_BASE >= 1u64 << 46);
};

// ---------------------------------------------------------------------------
// Stage 3 messages (first increment): the stable, descriptor-free object
// lifecycle and completion records. The descriptor-carrying creates (textures,
// pipelines, samplers, depth/stencil, library bytes) and the command-buffer
// encoder-op stream follow in later increments.
// ---------------------------------------------------------------------------

/// `OBJECT_DESTROY`: drop the host handle for `obj_id`. The handle table knows
/// the kind, so one message destroys an object of any kind.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxObjectDestroy {
    /// [`RGFX_CMD_OBJECT_DESTROY`].
    pub header: RgfxMsgHeader,
    /// The object to drop.
    pub obj_id: u64,
}

/// `EVENT_CREATE` and `FENCE_CREATE`: a sync object with no payload but its id.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxObjectCreate {
    /// [`RGFX_CMD_EVENT_CREATE`] or [`RGFX_CMD_FENCE_CREATE`].
    pub header: RgfxMsgHeader,
    /// The new object's guest-picked id.
    pub obj_id: u64,
}

/// `SHARED_EVENT_CREATE`: a shared event starting at `initial_value`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSharedEventCreate {
    /// [`RGFX_CMD_SHARED_EVENT_CREATE`].
    pub header: RgfxMsgHeader,
    /// The new event's guest-picked id.
    pub obj_id: u64,
    /// The event's value when it is made.
    pub initial_value: u64,
}

/// `SHARED_EVENT_SET_VALUE`: the shared event `obj_id` set to `value` from the CPU.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSharedEventSetValue {
    /// [`RGFX_CMD_SHARED_EVENT_SET_VALUE`].
    pub header: RgfxMsgHeader,
    /// The shared event.
    pub obj_id: u64,
    /// The value `setSignaledValue:` was given.
    pub value: u64,
}

/// `QUEUE_CREATE`: a command queue and its submission+completion ring pair, each
/// a [`RgfxRingHeader`] followed by a byte ring of the given data size (a power
/// of two) at the given region offset. The completion ring raises `vector`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxQueueCreate {
    /// [`RGFX_CMD_QUEUE_CREATE`].
    pub header: RgfxMsgHeader,
    /// The new queue's guest-picked id.
    pub obj_id: u64,
    /// Where the submission ring (guest to host) starts in the region.
    pub submit_offset: u64,
    /// Where the completion ring (host to guest) starts in the region.
    pub complete_offset: u64,
    /// The submission ring's byte-ring size, after its header: a power of two.
    pub submit_data_size: u32,
    /// The completion ring's byte-ring size, after its header: a power of two.
    pub complete_data_size: u32,
    /// The queue's priority, 0 normal and higher higher. The bundle sends 0 for
    /// every queue (a `setGPUPriority:` stays in the guest), and the host reads
    /// none: each queue is the platform's plain `newCommandQueue`.
    pub priority: u32,
    /// The MSI-X vector the completion ring raises.
    pub vector: u32,
}

/// Where a `QUEUE_CREATE` message's [`RgfxQueueCreateFlags`] starts: right after its
/// [`RgfxQueueCreate`]. A message whose `size` is at least
/// `RGFX_QUEUE_CREATE_FLAGS_OFFSET + size_of::<RgfxQueueCreateFlags>()` carries it; one of
/// `RGFX_QUEUE_CREATE_FLAGS_OFFSET` none, and its queue has no flag. The kext sends it only with a
/// flag set, so a host older than it never sees one.
pub const RGFX_QUEUE_CREATE_FLAGS_OFFSET: u32 = 56;

/// [`RgfxQueueCreateFlags::flags`]: the queue's `CB_SCHEDULED` raises only where asked
/// ([`RGFX_FEATURE_QUIET_SCHEDULED`]).
pub const RGFX_QUEUE_QUIET_SCHEDULED: u32 = 1 << 0;
/// [`RgfxQueueCreateFlags::flags`]: the host waits for room on the queue's completion ring until a
/// [`RGFX_REG_CONSUMED`] write ([`RGFX_FEATURE_CONSUMED_DOORBELL`]).
pub const RGFX_QUEUE_CONSUMED_DOORBELL: u32 = 1 << 1;
/// [`RgfxQueueCreateFlags::flags`]: the queue shares its GPU queue with no other command queue
/// ([`RGFX_FEATURE_UNSHARED_QUEUES`]). The kext sets it itself, for every queue of
/// WindowServer's, and never for a process that asks for it. Bit 2 is reserved.
pub const RGFX_QUEUE_UNSHARED: u32 = 1 << 3;

/// The modes a `QUEUE_CREATE` asks for its queue, after the [`RgfxQueueCreate`] at
/// [`RGFX_QUEUE_CREATE_FLAGS_OFFSET`]. A flag whose feature the guest did not take
/// ([`RGFX_REG_DRIVER_FEATURES_LO`]), or one the host does not carry out, refuses the create with
/// [`RGFX_STATUS_INVALID_ARGUMENT`]; the host carries out [`RGFX_QUEUE_UNSHARED`] alone.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxQueueCreateFlags {
    /// `RGFX_QUEUE_*`: [`RGFX_QUEUE_QUIET_SCHEDULED`], [`RGFX_QUEUE_CONSUMED_DOORBELL`] and
    /// [`RGFX_QUEUE_UNSHARED`].
    pub flags: u32,
    /// Zero.
    pub reserved_04: u32,
}

/// `CONTEXT_CLOSE`: context `context`'s process is gone.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxContextClose {
    /// [`RGFX_CMD_CONTEXT_CLOSE`].
    pub header: RgfxMsgHeader,
    /// The context.
    pub context: u32,
    /// Zero.
    pub reserved_14: u32,
}

/// `COMMAND_BUFFER_IN_REGION`: the `COMMAND_BUFFER` message of `byte_length` bytes at region
/// offset `region_offset`. The message there is whole: its header's `type` is
/// `COMMAND_BUFFER` and its `size` is `byte_length`; its `seqno` is this reference's. A span
/// outside the processes' blocks, or a message there that is not such a `COMMAND_BUFFER`, is a
/// guest protocol error, refused as a message the ring may not carry.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxCommandBufferInRegion {
    /// [`RGFX_CMD_COMMAND_BUFFER_IN_REGION`].
    pub header: RgfxMsgHeader,
    /// Where the message starts in the device's region: 8-aligned.
    pub region_offset: u64,
    /// The message's length: a multiple of 8.
    pub byte_length: u64,
}

/// `COMMAND_BUFFER_PART`: part `part` of command buffer `cb_obj_id`'s ops, the `byte_length` bytes
/// at region offset `region_offset`, in one block of the process's own. Its header's `seqno` is
/// the ring's; the completion names the last part's.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxCommandBufferPart {
    /// [`RGFX_CMD_COMMAND_BUFFER_PART`].
    pub header: RgfxMsgHeader,
    /// The queue's id.
    pub queue_obj_id: u64,
    /// The command buffer's guest-picked id.
    pub cb_obj_id: u64,
    /// The part's number within its buffer, from 0.
    pub part: u32,
    /// `RGFX_CB_PART_*`: [`RGFX_CB_PART_LAST`] and [`RGFX_CB_PART_ABORT`].
    pub flags: u32,
    /// Where the part's ops start in the device's region: 8-aligned.
    pub region_offset: u64,
    /// The part's length: a multiple of 8.
    pub byte_length: u64,
}

/// `RESOURCE_ID_CHUNK`: context `context` holds chunk `chunk` of the texture resource ids.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxResourceIdChunk {
    /// [`RGFX_CMD_RESOURCE_ID_CHUNK`].
    pub header: RgfxMsgHeader,
    /// The context.
    pub context: u32,
    /// The chunk, below [`RGFX_RESOURCE_ID_CHUNKS`]: ids `1 + chunk * RGFX_RESOURCE_ID_CHUNK` and
    /// up.
    pub chunk: u32,
}

/// `BLOCK_RELEASE`: context `context` freed its region block of `size` bytes at `offset`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlockRelease {
    /// [`RGFX_CMD_BLOCK_RELEASE`].
    pub header: RgfxMsgHeader,
    /// The context that held the block.
    pub context: u32,
    /// `RGFX_BLOCK_RELEASE_*` bits; zero from a kext that sets none.
    pub flags: u32,
    /// The block's region offset.
    pub offset: u64,
    /// The block's size in bytes.
    pub size: u64,
}

/// [`RgfxBlockRelease::flags`]: the block's IOSurface has no client left, no process holding it,
/// so no reader of its pages: the host drops the block's device cache with it, rather than
/// copying what command buffers wrote of it into the pages first. Without it the host copies.
pub const RGFX_BLOCK_RELEASE_NO_CLIENT: u32 = 1 << 0;

/// A run of an IOSurface's guest RAM, which `SURFACE_PAGES` lists: `length` bytes at guest
/// physical address `address`, both multiples of the guest's 4 KiB page.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxPageRun {
    /// The run's first guest physical address.
    pub address: u64,
    /// The run's bytes.
    pub length: u64,
}

/// `SURFACE_PAGES`: the guest RAM of the IOSurface whose shared block is the `block_size` bytes
/// at `block_offset`, made for context `context`: `run_count` [`RgfxPageRun`]s at region offset
/// `runs_offset`, holding the surface's `length` bytes in order.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSurfacePages {
    /// [`RGFX_CMD_SURFACE_PAGES`].
    pub header: RgfxMsgHeader,
    /// The context the block was made for, its first holder.
    pub context: u32,
    /// How many runs lie at `runs_offset`.
    pub run_count: u32,
    /// The shared block's region offset.
    pub block_offset: u64,
    /// The shared block's size in bytes, its runs included.
    pub block_size: u64,
    /// Where the runs start in the region, inside the block.
    pub runs_offset: u64,
    /// The surface's bytes, which the runs hold.
    pub length: u64,
}

/// Where a `SURFACE_PAGES` message's [`RgfxSurfacePagesCache`] starts: right after its
/// [`RgfxSurfacePages`]. A message whose `size` is at least
/// `RGFX_SURFACE_PAGES_CACHE_OFFSET + size_of::<RgfxSurfacePagesCache>()` carries it; one of
/// `RGFX_SURFACE_PAGES_CACHE_OFFSET` none, and its block has no device cache.
pub const RGFX_SURFACE_PAGES_CACHE_OFFSET: u32 = 56;

/// [`RgfxSurfacePagesCache::flags`]: the block's surface has a device cache, where the device has
/// [`RGFX_FEATURE_SURFACE_CACHE`]: its `BUFFER_OVER_SURFACE_PAGES` buffers lie over the cache, every
/// command buffer that uses the surface names it in an `RGFX_OP_SURFACE_USE` with the CPU's
/// generation of the surface, and the kext asks a [`RGFX_CMD_SURFACE_DOWNLOAD`] before a CPU lock
/// that follows a GPU write. Without the feature the host gives the block none.
pub const RGFX_SURFACE_PAGES_CACHED: u32 = 1 << 0;
/// [`RgfxSurfacePagesCache::flags`], where the guest took [`RGFX_FEATURE_PAGES_BUFFERS`]: the
/// block is a process's pages block, for its `BUFFER_OVER_PAGES` buffers, not an IOSurface's, and
/// its view stays out of the host's surface-view cache. No host offers the feature yet.
pub const RGFX_SURFACE_PAGES_PRIVATE: u32 = 1 << 1;

/// What a `SURFACE_PAGES` says of its block's device cache, after the [`RgfxSurfacePages`] at
/// [`RGFX_SURFACE_PAGES_CACHE_OFFSET`].
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSurfacePagesCache {
    /// `RGFX_SURFACE_PAGES_*`: [`RGFX_SURFACE_PAGES_CACHED`] and [`RGFX_SURFACE_PAGES_PRIVATE`].
    pub flags: u32,
    /// Zero.
    pub reserved_04: u32,
}

/// `SURFACE_DOWNLOAD`: the device cache of the IOSurface whose cached block is at
/// `block_offset`, copied into the surface's pages.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSurfaceDownload {
    /// [`RGFX_CMD_SURFACE_DOWNLOAD`].
    pub header: RgfxMsgHeader,
    /// The surface's shared block's region offset.
    pub block_offset: u64,
}

/// `BUFFER_OVER_SURFACE_PAGES`: a Shared buffer of `length` bytes over the IOSurface pages the
/// shared block of `block_size` bytes at `block_offset` names.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBufferOverSurfacePages {
    /// [`RGFX_CMD_BUFFER_OVER_SURFACE_PAGES`].
    pub header: RgfxMsgHeader,
    /// The new buffer's guest-picked id.
    pub obj_id: u64,
    /// The shared block's region offset.
    pub block_offset: u64,
    /// The shared block's size in bytes, as its holders know it.
    pub block_size: u64,
    /// The buffer's length in bytes, from the surface's first byte.
    pub length: u64,
    /// `MTLResourceOptions`: storage mode, cache mode, hazard tracking.
    pub options: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_34: u32,
}

/// `BUFFER_OVER_PAGES`: a buffer of `length` bytes at `offset` in the view of the pages block of
/// `block_size` bytes at `block_offset`, at GPU address `gpu_address`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBufferOverPages {
    /// [`RGFX_CMD_BUFFER_OVER_PAGES`].
    pub header: RgfxMsgHeader,
    /// The new buffer's guest-picked id.
    pub obj_id: u64,
    /// The buffer's guest-picked GPU address.
    pub gpu_address: u64,
    /// The pages block's region offset, a block of the sender's context.
    pub block_offset: u64,
    /// The pages block's size in bytes.
    pub block_size: u64,
    /// Where the buffer starts in the block's view, in bytes.
    pub offset: u64,
    /// The buffer's length in bytes.
    pub length: u64,
    /// `MTLResourceOptions`: storage mode, cache mode, hazard tracking.
    pub options: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_44: u32,
}

/// `BLOCK_GRANT`: the kext gave context `context` the region block of `size` bytes at `offset`.
/// `BLOCK_SHARE` and `BLOCK_UNSHARE` have its layout: context `context` holds, or no longer
/// holds, the shared block of `size` bytes at `offset`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBlockGrant {
    /// [`RGFX_CMD_BLOCK_GRANT`].
    pub header: RgfxMsgHeader,
    /// The context given the block.
    pub context: u32,
    /// Zero.
    pub reserved_14: u32,
    /// The block's region offset.
    pub offset: u64,
    /// The block's size in bytes.
    pub size: u64,
}

/// `HEAP_CREATE`: a device heap of `size_bytes` at the guest-picked
/// `gpu_address`. `options` carries the heap's storage and cache modes and type.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxHeapCreate {
    /// [`RGFX_CMD_HEAP_CREATE`].
    pub header: RgfxMsgHeader,
    /// The new heap's guest-picked id.
    pub obj_id: u64,
    /// The heap's guest-picked GPU address.
    pub gpu_address: u64,
    /// The heap's size in bytes.
    pub size_bytes: u64,
    /// `MTLResourceOptions`: storage mode, cache mode, hazard tracking.
    pub options: u32,
    /// `MTLHeapType` (Automatic, Placement or Sparse); widened on decode.
    pub heap_type: i32,
}

/// `HEAP_OVER_HOST_MEMORY`: a Shared heap of `size_bytes` over the block at
/// `region_offset` in the region, at the guest-picked `gpu_address`. The block is one
/// of the creating process's own, which the kext checks; a span outside the processes'
/// blocks ([`RGFX_CONTEXT_ALLOCATIONS_OFFSET`] up to the cursor's span) is refused.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxHeapOverHostMemory {
    /// [`RGFX_CMD_HEAP_OVER_HOST_MEMORY`].
    pub header: RgfxMsgHeader,
    /// The new heap's guest-picked id.
    pub obj_id: u64,
    /// The heap's guest-picked GPU address.
    pub gpu_address: u64,
    /// Where the heap's block starts in the region.
    pub region_offset: u64,
    /// The heap's size in bytes.
    pub size_bytes: u64,
    /// `MTLResourceOptions`: storage mode, cache mode, hazard tracking.
    pub options: u32,
    /// `MTLHeapType` (Automatic, Placement or Sparse); widened on decode.
    pub heap_type: i32,
}

/// `BUFFER_CREATE`: a device buffer of `length` bytes at the guest-picked
/// `gpu_address`. `options` carries `MTLResourceOptions`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBufferCreate {
    /// [`RGFX_CMD_BUFFER_CREATE`].
    pub header: RgfxMsgHeader,
    /// The new buffer's guest-picked id.
    pub obj_id: u64,
    /// The buffer's guest-picked GPU address.
    pub gpu_address: u64,
    /// The buffer's length in bytes.
    pub length: u64,
    /// `MTLResourceOptions`: storage mode, cache mode, hazard tracking.
    pub options: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_2c: u32,
}

/// `BUFFER_OVER_HOST_MEMORY`: a Shared buffer of `length` bytes over the block at
/// `region_offset` in the region, at the guest-picked `gpu_address`. The block is one
/// of the creating process's own, as [`RgfxHeapOverHostMemory`]'s is.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBufferOverHostMemory {
    /// [`RGFX_CMD_BUFFER_OVER_HOST_MEMORY`].
    pub header: RgfxMsgHeader,
    /// The new buffer's guest-picked id.
    pub obj_id: u64,
    /// The buffer's guest-picked GPU address.
    pub gpu_address: u64,
    /// Where the buffer's block starts in the region.
    pub region_offset: u64,
    /// The buffer's length in bytes.
    pub length: u64,
    /// `MTLResourceOptions`: storage mode, cache mode, hazard tracking.
    pub options: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_34: u32,
}

/// `BUFFER_AT_OFFSET`: a buffer of `length` bytes at `offset` in the heap
/// `heap_obj_id`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBufferAtOffset {
    /// [`RGFX_CMD_BUFFER_AT_OFFSET`].
    pub header: RgfxMsgHeader,
    /// The new buffer's guest-picked id.
    pub obj_id: u64,
    /// The heap the buffer lies in.
    pub heap_obj_id: u64,
    /// The buffer's offset in the heap.
    pub offset: u64,
    /// The buffer's length in bytes.
    pub length: u64,
    /// `MTLResourceOptions`: storage mode, cache mode, hazard tracking.
    pub options: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_34: u32,
}

/// `BUFFER_MANAGED_AT_OFFSET`: a `Managed` buffer of `length` bytes at the guest-picked
/// `gpu_address`, its CPU copy the `length` bytes at `offset` in the heap `heap_obj_id`, a heap
/// over host memory of the sender's. The host's GPU copy starts as the CPU copy's bytes;
/// `BUFFER_DID_MODIFY_RANGE` and a synchronize copy between the two as for a
/// [`RgfxBufferOverHostMemory`] of a `Managed` buffer.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBufferManagedAtOffset {
    /// [`RGFX_CMD_BUFFER_MANAGED_AT_OFFSET`].
    pub header: RgfxMsgHeader,
    /// The new buffer's guest-picked id.
    pub obj_id: u64,
    /// The buffer's guest-picked GPU address.
    pub gpu_address: u64,
    /// The heap its CPU copy lies in.
    pub heap_obj_id: u64,
    /// The CPU copy's offset in the heap.
    pub offset: u64,
    /// The buffer's length in bytes.
    pub length: u64,
    /// `MTLResourceOptions` of the `Managed` buffer: storage mode, cache mode, hazard tracking.
    pub options: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_3c: u32,
}

/// `CB_COMPLETE`: a command buffer finished. `status` is an `RGFX_CB_*`, and
/// `error_code` carries the host's raw `MTLCommandBufferError` code for
/// diagnostics (0 when `status` is [`RGFX_CB_OK`]).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxCbComplete {
    /// [`RGFX_EVT_CB_COMPLETE`], with the command buffer's submission `seqno`.
    pub header: RgfxMsgHeader,
    /// The command buffer that finished.
    pub cb_obj_id: u64,
    /// The host's monotonic clock when the GPU started the work, in nanoseconds;
    /// 0 when unknown (e.g. a recovery victim).
    pub gpu_start_ns: u64,
    /// The host's monotonic clock when the GPU finished the work, in nanoseconds.
    pub gpu_end_ns: u64,
    /// An `RGFX_CB_*`.
    pub status: u32,
    /// The host's raw `MTLCommandBufferError` code behind `status`, for
    /// diagnostics; 0 on success.
    pub error_code: u32,
}

/// `CB_SCHEDULED`, on the command buffer's queue's completion ring: the buffer is scheduled (its
/// platform command buffer's scheduled handler ran), ahead of its [`RgfxCbComplete`]. A buffer the
/// host could not carry out has none, nor does one of a host that predates the event: its
/// `CB_COMPLETE` stands for both. The guest reads a record of at least this size.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxCbScheduled {
    /// [`RGFX_EVT_CB_SCHEDULED`], with the command buffer's submission `seqno`.
    pub header: RgfxMsgHeader,
    /// The command buffer that is scheduled.
    pub cb_obj_id: u64,
}

/// `CB_SUBMITTED`, on the command buffer's queue's completion ring: the buffer, which asked for it
/// ([`RGFX_OP_SUBMITTED`](crate::ops::RGFX_OP_SUBMITTED)), is submitted: the host has taken it and
/// carries it out whole before any message the guest sends after this event, so that later work
/// comes after the buffer's commit to the host's platform, and its waits for events are satisfied.
/// Ahead of its [`RgfxCbScheduled`] and its [`RgfxCbComplete`]. A buffer the host could not take
/// has none: its `CB_COMPLETE` stands for it. One whose ops the host then cannot carry out has it,
/// and its failure at its `CB_COMPLETE`, as Metal reports a command buffer's errors at its
/// completion. The guest reads a record of at least this size.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxCbSubmitted {
    /// [`RGFX_EVT_CB_SUBMITTED`], with the command buffer's submission `seqno`.
    pub header: RgfxMsgHeader,
    /// The command buffer that is submitted.
    pub cb_obj_id: u64,
}

/// `EVENT_SIGNALLED`, on the completion ring of the queue whose command buffer signalled it, ahead
/// of that buffer's [`RgfxCbComplete`]: the shared event `obj_id` was signalled to `value`
/// (`encodeSignalEvent:value:`) by the buffer, now finished.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxEventSignalled {
    /// [`RGFX_EVT_EVENT_SIGNALLED`], `seqno` 0.
    pub header: RgfxMsgHeader,
    /// The shared event that fired.
    pub obj_id: u64,
    /// The value the buffer signalled it to.
    pub value: u64,
}

/// A texture's shape, as `MTLTextureDescriptor` gives it. Each enum is its Metal
/// raw value (`MTLPixelFormat`, `MTLTextureType`, `MTLTextureUsage`,
/// `MTLResourceOptions`); `swizzle` packs the four `MTLTextureSwizzle` channels,
/// one byte each, red in the low byte.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxTextureDescriptor {
    /// `MTLPixelFormat`.
    pub pixel_format: u32,
    /// `MTLTextureType`.
    pub texture_type: u32,
    /// Pixels a row.
    pub width: u32,
    /// Rows.
    pub height: u32,
    /// Slices, for a 3D texture; 1 otherwise.
    pub depth: u32,
    /// Mip levels.
    pub mipmap_level_count: u32,
    /// Array elements, for an array texture; 1 otherwise.
    pub array_length: u32,
    /// Samples a pixel, for a multisample texture; 1 otherwise.
    pub sample_count: u32,
    /// `MTLTextureUsage`.
    pub usage: u32,
    /// `MTLResourceOptions`: the storage, cache and hazard-tracking modes.
    pub options: u32,
    /// The four `MTLTextureSwizzle` channels, one byte each, red lowest.
    pub swizzle: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_2c: u32,
}

/// `HEAP_TEXTURE_SIZE_AND_ALIGN`: the size and alignment a texture of shape `desc` takes in a heap.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxHeapTextureSizeAndAlign {
    /// [`RGFX_CMD_HEAP_TEXTURE_SIZE_AND_ALIGN`].
    pub header: RgfxMsgHeader,
    /// A new id of the sender's context, which names no object: the query makes none, and the id
    /// tells whose it is.
    pub obj_id: u64,
    /// The texture's shape, as `TEXTURE_CREATE` carries it; its `swizzle` is not read.
    pub desc: RgfxTextureDescriptor,
}

/// `SIZE_AND_ALIGN`: a `HEAP_TEXTURE_SIZE_AND_ALIGN`'s answer. `status` is [`RGFX_STATUS_OK`]
/// with the device's `size` and `align` (a size of 0 for a shape no texture has, as Metal answers
/// it), or [`RGFX_STATUS_REFUSED`], with both 0, where the device gave no answer.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSizeAndAlign {
    /// [`RGFX_EVT_SIZE_AND_ALIGN`], with the query's `seqno`.
    pub header: RgfxMsgHeader,
    /// An `RGFX_STATUS_*`.
    pub status: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_14: u32,
    /// `MTLSizeAndAlign.size`.
    pub size: u64,
    /// `MTLSizeAndAlign.align`.
    pub align: u64,
}

/// `SAMPLER_STATE`: a `SAMPLER_CREATE`'s answer. `status` is [`RGFX_STATUS_OK`] with the sampler's
/// `gpu_resource_id`, or [`RGFX_STATUS_REFUSED`] (the device made none) or
/// [`RGFX_STATUS_INVALID_ARGUMENT`] (the create was short), with 0. The host's device numbers its
/// samplers itself, every process's and its own in one space, and answers one sampler, and one
/// id, for equal descriptors, so the guest takes the id from here rather than picking one.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSamplerState {
    /// [`RGFX_EVT_SAMPLER_STATE`], with the create's `seqno`.
    pub header: RgfxMsgHeader,
    /// An `RGFX_STATUS_*`.
    pub status: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_14: u32,
    /// `MTLSamplerState.gpuResourceID`: what an argument buffer holds of the sampler.
    pub gpu_resource_id: u64,
}

/// [`RgfxIcbCreate`] `flags`: `inheritPipelineState`.
pub const RGFX_ICB_INHERIT_PIPELINE_STATE: u32 = 1 << 0;
/// [`RgfxIcbCreate`] `flags`: `inheritBuffers`.
pub const RGFX_ICB_INHERIT_BUFFERS: u32 = 1 << 1;
/// [`RgfxIcbCreate`] `flags`: `supportRayTracing`.
pub const RGFX_ICB_SUPPORT_RAY_TRACING: u32 = 1 << 2;
/// [`RgfxIcbCreate`] `flags`: `supportDynamicAttributeStride`.
pub const RGFX_ICB_SUPPORT_DYNAMIC_ATTRIBUTE_STRIDE: u32 = 1 << 3;
/// [`RgfxIcbCreate`] `flags`: `supportColorAttachmentMapping`. The platform's descriptor has no
/// such field: a buffer that supports it executes as any other in an encoder that maps no
/// attachments, every encoder of the lane's.
pub const RGFX_ICB_SUPPORT_COLOR_ATTACHMENT_MAPPING: u32 = 1 << 4;
/// [`RgfxIcbCreate`] `flags`: `inheritDepthStencilState`.
pub const RGFX_ICB_INHERIT_DEPTH_STENCIL_STATE: u32 = 1 << 5;
/// [`RgfxIcbCreate`] `flags`: `inheritDepthBias`.
pub const RGFX_ICB_INHERIT_DEPTH_BIAS: u32 = 1 << 6;
/// [`RgfxIcbCreate`] `flags`: `inheritDepthClipMode`.
pub const RGFX_ICB_INHERIT_DEPTH_CLIP_MODE: u32 = 1 << 7;
/// [`RgfxIcbCreate`] `flags`: `inheritCullMode`.
pub const RGFX_ICB_INHERIT_CULL_MODE: u32 = 1 << 8;
/// [`RgfxIcbCreate`] `flags`: `inheritFrontFacingWinding`.
pub const RGFX_ICB_INHERIT_FRONT_FACING_WINDING: u32 = 1 << 9;
/// [`RgfxIcbCreate`] `flags`: `inheritTriangleFillMode`.
pub const RGFX_ICB_INHERIT_TRIANGLE_FILL_MODE: u32 = 1 << 10;
/// [`RgfxIcbCreate`] `flags`: `inheritStencilReferenceValues` (SPI).
pub const RGFX_ICB_INHERIT_STENCIL_REFERENCE_VALUES: u32 = 1 << 11;
/// [`RgfxIcbCreate`] `flags`: `inheritDepthTestBounds` (SPI).
pub const RGFX_ICB_INHERIT_DEPTH_TEST_BOUNDS: u32 = 1 << 12;
/// [`RgfxIcbCreate`] `flags`: `inheritScissorRects` (SPI).
pub const RGFX_ICB_INHERIT_SCISSOR_RECTS: u32 = 1 << 13;
/// [`RgfxIcbCreate`] `flags`: `inheritViewports` (SPI).
pub const RGFX_ICB_INHERIT_VIEWPORTS: u32 = 1 << 14;
/// [`RgfxIcbCreate`] `flags`: `inheritBlendColor` (SPI).
pub const RGFX_ICB_INHERIT_BLEND_COLOR: u32 = 1 << 15;

/// `ICB_CREATE`: indirect command buffer `obj_id` of `max_command_count` commands, with
/// `MTLResourceOptions` `options`, from the guest's `MTLIndirectCommandBufferDescriptor`: every
/// field its answer, the SPI ones (`max_scissor_rect_count`, `max_viewport_count`,
/// `allow_override_render_states` and the SPI inherits) where it answers them, else a fresh
/// descriptor's (1, 1, 0 and inherited, measured on an AMD 5300M and an Intel 630).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbCreate {
    /// [`RGFX_CMD_ICB_CREATE`].
    pub header: RgfxMsgHeader,
    /// The new buffer's guest-picked id.
    pub obj_id: u64,
    /// `maxCommandCount`.
    pub max_command_count: u64,
    /// `MTLResourceOptions`.
    pub options: u64,
    /// `MTLIndirectCommandType` bits.
    pub command_types: u32,
    /// `RGFX_ICB_*` bits.
    pub flags: u32,
    /// `maxVertexBufferBindCount`.
    pub max_vertex_buffer_bind_count: u32,
    /// `maxFragmentBufferBindCount`.
    pub max_fragment_buffer_bind_count: u32,
    /// `maxKernelBufferBindCount`.
    pub max_kernel_buffer_bind_count: u32,
    /// `maxKernelThreadgroupMemoryBindCount`.
    pub max_kernel_threadgroup_memory_bind_count: u32,
    /// `maxObjectBufferBindCount`.
    pub max_object_buffer_bind_count: u32,
    /// `maxMeshBufferBindCount`.
    pub max_mesh_buffer_bind_count: u32,
    /// `maxObjectThreadgroupMemoryBindCount`.
    pub max_object_threadgroup_memory_bind_count: u32,
    /// `maxScissorRectCount` (SPI).
    pub max_scissor_rect_count: u32,
    /// `maxViewportCount` (SPI).
    pub max_viewport_count: u32,
    /// `allowOverrideRenderStates` (SPI).
    pub allow_override_render_states: u32,
}

/// `ICB_STATE`: an `ICB_CREATE`'s answer. `status` is [`RGFX_STATUS_OK`] with the buffer's
/// `gpu_resource_id`, or [`RGFX_STATUS_REFUSED`] (the device made none, or the sender's context
/// failed) or [`RGFX_STATUS_INVALID_ARGUMENT`] (the create was short), with 0. The host's device
/// numbers its indirect command buffers itself, every process's in one space, so the guest takes
/// the id from here rather than picking one.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbState {
    /// [`RGFX_EVT_ICB_STATE`], with the create's `seqno`.
    pub header: RgfxMsgHeader,
    /// An `RGFX_STATUS_*`.
    pub status: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_14: u32,
    /// `MTLIndirectCommandBuffer.gpuResourceID`: what an argument buffer holds of the buffer.
    pub gpu_resource_id: u64,
}

/// `ICB_WRITE`: the CPU's encodes into indirect command buffer `obj_id`, the records
/// ([`crate::icb::RgfxIcbRecord`]) that follow, to the message's `size`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbWrite {
    /// [`RGFX_CMD_ICB_WRITE`].
    pub header: RgfxMsgHeader,
    /// The indirect command buffer.
    pub obj_id: u64,
}

/// `BUFFER_DID_MODIFY_RANGE`: the guest wrote `length` bytes at `location` of `Managed` buffer
/// `obj_id`'s CPU copy, the region block its `BUFFER_OVER_HOST_MEMORY` named.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBufferDidModifyRange {
    /// [`RGFX_CMD_BUFFER_DID_MODIFY_RANGE`].
    pub header: RgfxMsgHeader,
    /// The buffer.
    pub obj_id: u64,
    /// Where the bytes start in the buffer.
    pub location: u64,
    /// How many.
    pub length: u64,
}

/// The rows of a level of a `Managed` texture's CPU copy start at multiples of this, in bytes
/// ([`RgfxManagedTextureCreate`]).
pub const RGFX_MANAGED_ROW_ALIGN: u64 = 256;

/// [`RgfxManagedTextureCreate::flags`]: the pixel format is one a texture over a buffer can
/// have (uncompressed, one texel a block, neither depth nor stencil).
pub const RGFX_MANAGED_TEXTURE_LINEAR: u32 = 1;

/// `MANAGED_TEXTURE_CREATE`: a `Managed` texture of shape `desc` at the guest-picked
/// `gpu_address`, with the guest-picked `resource_id`, whose CPU copy is the Shared buffer
/// `cpu_buffer_obj_id` over a block of the sender's.
///
/// The CPU copy holds every level in order; each level its slices in order (`array_length`, six
/// faces a cube); each slice its images (the level's depth, for a 3D texture; else one); each
/// image its rows of texel blocks. A level of `w` by `h` texels (the texture's halved `level`
/// times, at least 1) has rows of `ceil(w / block_width) * bytes_per_block` bytes rounded up to
/// [`RGFX_MANAGED_ROW_ALIGN`], and `ceil(h / block_height)` of them an image. The guest and the
/// host each compute it from these fields alone (`reims_graphics::managed`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxManagedTextureCreate {
    /// [`RGFX_CMD_MANAGED_TEXTURE_CREATE`].
    pub header: RgfxMsgHeader,
    /// The new texture's guest-picked id.
    pub obj_id: u64,
    /// The texture's guest-picked GPU address.
    pub gpu_address: u64,
    /// The texture's guest-picked resource id.
    pub resource_id: u64,
    /// The Shared buffer over the block that holds the texture's CPU copy.
    pub cpu_buffer_obj_id: u64,
    /// The bytes of one block of texels of the pixel format.
    pub bytes_per_block: u32,
    /// A block's width in texels: 1 for an uncompressed format.
    pub block_width: u32,
    /// A block's height in texels.
    pub block_height: u32,
    /// [`RGFX_MANAGED_TEXTURE_LINEAR`], or 0.
    pub flags: u32,
    /// The texture's shape.
    pub desc: RgfxTextureDescriptor,
}

/// `TEXTURE_REPLACE_REGION`: the guest wrote the texels of `size` at `origin` of `level` and
/// `slice` of `Managed` texture `obj_id`'s CPU copy, at their places in its layout
/// ([`RgfxManagedTextureCreate`]).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxTextureReplaceRegion {
    /// [`RGFX_CMD_TEXTURE_REPLACE_REGION`].
    pub header: RgfxMsgHeader,
    /// The texture.
    pub obj_id: u64,
    /// The mip level.
    pub level: u32,
    /// The slice: the array element, times six plus the face for a cube.
    pub slice: u32,
    /// The region's origin, x, y and z, in texels.
    pub origin: [u32; 3],
    /// The region's width, height and depth, in texels.
    pub size: [u32; 3],
}

/// [`RgfxTextureView::flags`]: the view names its type, levels and slices
/// (`newTextureViewWithPixelFormat:textureType:levels:slices:`). Without it the view is of the
/// whole texture in its own type (`newTextureViewWithPixelFormat:`).
pub const RGFX_TEXTURE_VIEW_SUBRANGE: u32 = 1;
/// [`RgfxTextureView::flags`]: the view names its swizzle as well
/// (`newTextureViewWithPixelFormat:textureType:levels:slices:swizzle:`); only with
/// [`RGFX_TEXTURE_VIEW_SUBRANGE`].
pub const RGFX_TEXTURE_VIEW_SWIZZLE: u32 = 2;

/// `TEXTURE_VIEW`: view `obj_id`, with the guest-picked `resource_id`, of texture
/// `parent_obj_id` (any texture the guest made, a view among them), made by the selector the
/// `RGFX_TEXTURE_VIEW_*` flags name.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxTextureView {
    /// [`RGFX_CMD_TEXTURE_VIEW`].
    pub header: RgfxMsgHeader,
    /// The view's guest-picked id.
    pub obj_id: u64,
    /// The texture it views.
    pub parent_obj_id: u64,
    /// The view's guest-picked resource id.
    pub resource_id: u64,
    /// `MTLPixelFormat`.
    pub pixel_format: u32,
    /// The `RGFX_TEXTURE_VIEW_*` flags.
    pub flags: u32,
    /// `MTLTextureType`, with [`RGFX_TEXTURE_VIEW_SUBRANGE`]; 0 without it.
    pub texture_type: u32,
    /// The first of the parent's mip levels the view takes, with
    /// [`RGFX_TEXTURE_VIEW_SUBRANGE`]; 0 without it.
    pub level_start: u32,
    /// How many levels, with [`RGFX_TEXTURE_VIEW_SUBRANGE`]; 0 without it.
    pub level_count: u32,
    /// The first of the parent's slices the view takes (an array element, times six plus the
    /// face for a cube), with [`RGFX_TEXTURE_VIEW_SUBRANGE`]; 0 without it.
    pub slice_start: u32,
    /// How many slices, with [`RGFX_TEXTURE_VIEW_SUBRANGE`]; 0 without it.
    pub slice_count: u32,
    /// The four `MTLTextureSwizzle` channels, one byte each, red lowest, with
    /// [`RGFX_TEXTURE_VIEW_SWIZZLE`]; 0 without it.
    pub swizzle: u32,
}

/// [`RgfxArgumentLayoutQuery::source`]: the layout of `count` [`RgfxArgumentDescriptor`]s
/// (`newArgumentEncoderWithArguments:`).
pub const RGFX_ARGUMENT_SOURCE_DESCRIPTORS: u32 = 0;
/// [`RgfxArgumentLayoutQuery::source`]: the layout of the argument buffer a library's function
/// takes at `buffer_index` (`newArgumentEncoderWithBufferIndex:`).
pub const RGFX_ARGUMENT_SOURCE_FUNCTION: u32 = 1;

/// `ARGUMENT_LAYOUT`: the layout of one argument buffer, from the source `source` names, written
/// into the `reply_length` bytes at `reply_offset` of a block of the sender's. `obj_id` is an id
/// of the sender's context naming the query; nothing is made under it.
/// - [`RGFX_ARGUMENT_SOURCE_DESCRIPTORS`]: `count` [`RgfxArgumentDescriptor`]s at
///   `descriptors_offset`, in the order the guest gave them.
/// - [`RGFX_ARGUMENT_SOURCE_FUNCTION`]: function `name_offset`/`name_len` of library
///   `library_obj_id`, specialised by the `constant_count` [`RgfxFunctionConstant`]s at
///   `constants_offset` where there are any, and the argument buffer it takes at
///   `buffer_index`.
///
/// Offsets are bytes from the message start; the message's `size` is the head plus the tail,
/// rounded up to a multiple of 8. Answered by [`RGFX_EVT_COMPLETE`]: [`RGFX_STATUS_OK`] with the
/// layout written; [`RGFX_STATUS_TOO_SMALL`] with only the bytes needed written; and
/// [`RGFX_STATUS_REFUSED`] where the platform makes no encoder of the source (a function with no
/// argument buffer at the index, descriptors Metal refuses), with nothing written.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxArgumentLayoutQuery {
    /// [`RGFX_CMD_ARGUMENT_LAYOUT`].
    pub header: RgfxMsgHeader,
    /// An id of the sender's context, naming the query.
    pub obj_id: u64,
    /// The answer's block: its offset in the region.
    pub reply_offset: u64,
    /// The answer's block: its length.
    pub reply_length: u64,
    /// The function's library, for [`RGFX_ARGUMENT_SOURCE_FUNCTION`]; 0 otherwise.
    pub library_obj_id: u64,
    /// The `RGFX_ARGUMENT_SOURCE_*` the layout comes from.
    pub source: u32,
    /// The [`RgfxArgumentDescriptor`]s, for [`RGFX_ARGUMENT_SOURCE_DESCRIPTORS`].
    pub count: u32,
    /// The descriptors' byte offset from the message start.
    pub descriptors_offset: u32,
    /// The function's buffer index, for [`RGFX_ARGUMENT_SOURCE_FUNCTION`].
    pub buffer_index: u32,
    /// The function name's byte offset from the message start.
    pub name_offset: u32,
    /// The function name's length in bytes.
    pub name_len: u32,
    /// The function constants' byte offset from the message start.
    pub constants_offset: u32,
    /// The number of [`RgfxFunctionConstant`]s.
    pub constant_count: u32,
}

/// One `MTLArgumentDescriptor` of an [`RgfxArgumentLayoutQuery`]; each enum is its Metal raw
/// value.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxArgumentDescriptor {
    /// `dataType` (`MTLDataType`).
    pub data_type: u32,
    /// `index`.
    pub index: u32,
    /// `arrayLength`.
    pub array_length: u32,
    /// `access` (`MTLBindingAccess`).
    pub access: u32,
    /// `textureType` (`MTLTextureType`).
    pub texture_type: u32,
    /// `constantBlockAlignment`.
    pub constant_block_alignment: u32,
}

/// [`RgfxArgumentMemberRecord::kind`]: a buffer's address, which may point to a nested argument
/// buffer ([`RgfxArgumentMemberRecord::layout`]).
pub const RGFX_ARGUMENT_BUFFER: u32 = 0;
/// [`RgfxArgumentMemberRecord::kind`]: a texture's resource id.
pub const RGFX_ARGUMENT_TEXTURE: u32 = 1;
/// [`RgfxArgumentMemberRecord::kind`]: a sampler state's resource id.
pub const RGFX_ARGUMENT_SAMPLER: u32 = 2;
/// [`RgfxArgumentMemberRecord::kind`]: plain bytes (`constantDataAtIndex:`).
pub const RGFX_ARGUMENT_CONSTANT: u32 = 3;
/// [`RgfxArgumentMemberRecord::kind`]: an indirect command buffer's resource id.
pub const RGFX_ARGUMENT_INDIRECT_COMMAND_BUFFER: u32 = 4;
/// [`RgfxArgumentMemberRecord::kind`]: a compute pipeline state's resource id.
pub const RGFX_ARGUMENT_COMPUTE_PIPELINE: u32 = 5;
/// [`RgfxArgumentMemberRecord::kind`]: a render pipeline state's resource id.
pub const RGFX_ARGUMENT_RENDER_PIPELINE: u32 = 6;
/// [`RgfxArgumentMemberRecord::kind`]: a visible function table's resource id.
pub const RGFX_ARGUMENT_VISIBLE_FUNCTION_TABLE: u32 = 7;
/// [`RgfxArgumentMemberRecord::kind`]: an intersection function table's resource id.
pub const RGFX_ARGUMENT_INTERSECTION_FUNCTION_TABLE: u32 = 8;
/// [`RgfxArgumentMemberRecord::kind`]: an acceleration structure's resource id.
pub const RGFX_ARGUMENT_ACCELERATION_STRUCTURE: u32 = 9;
/// [`RgfxArgumentMemberRecord::kind`]: a depth/stencil state.
pub const RGFX_ARGUMENT_DEPTH_STENCIL: u32 = 10;
/// [`RgfxArgumentMemberRecord::kind`]: a struct whose members are arguments, laid out as
/// [`RgfxArgumentMemberRecord::layout`] says, from the element's offset and argument index.
pub const RGFX_ARGUMENT_STRUCT: u32 = 11;
/// [`RgfxArgumentMemberRecord::layout`]: no layout.
pub const RGFX_ARGUMENT_NO_LAYOUT: u32 = 0xffff_ffff;

/// An [`RgfxArgumentLayoutQuery`]'s answer, at the start of the guest's block: `layout_count`
/// [`RgfxArgumentLayoutRecord`]s follow it, the first the argument buffer's own, then
/// `member_count` [`RgfxArgumentMemberRecord`]s, which the layouts index.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxArgumentLayoutReply {
    /// The bytes the whole answer takes; the only field written with
    /// [`RGFX_STATUS_TOO_SMALL`].
    pub needed: u64,
    /// The layouts.
    pub layout_count: u32,
    /// The members of every layout.
    pub member_count: u32,
}

/// One layout of an [`RgfxArgumentLayoutReply`]: the argument buffer's, a nested argument
/// buffer's, or one element's of a struct member.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxArgumentLayoutRecord {
    /// `encodedLength`: the bytes one argument buffer of the layout takes; a struct element's
    /// size.
    pub encoded_length: u64,
    /// `alignment`; 1 for a struct element's.
    pub alignment: u64,
    /// Its first member's index among the reply's members.
    pub first_member: u32,
    /// Its members, in layout order.
    pub member_count: u32,
}

/// One member of a layout: an argument, or an array of them. Argument index `i` lands in the
/// last member of the layout whose indices hold it: element `(i - id) / ids_per_element`, at
/// `offset` plus `stride` times the element; a struct element's members hold
/// `(i - id) % ids_per_element` from that offset on.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxArgumentMemberRecord {
    /// The argument index (`[[id(n)]]`) of element 0.
    pub id: u32,
    /// Elements: 1 for a single argument.
    pub count: u32,
    /// Argument indices each element takes: 1, or a struct's span.
    pub ids_per_element: u32,
    /// The `RGFX_ARGUMENT_*` kind.
    pub kind: u32,
    /// Byte offset of element 0.
    pub offset: u64,
    /// Bytes from one element to the next.
    pub stride: u64,
    /// Bytes of one element.
    pub size: u64,
    /// The layout of a nested argument buffer a [`RGFX_ARGUMENT_BUFFER`] points to, or of one
    /// element of a [`RGFX_ARGUMENT_STRUCT`], by its index among the reply's layouts;
    /// [`RGFX_ARGUMENT_NO_LAYOUT`] otherwise.
    pub layout: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_2c: u32,
}

/// `TEXTURE_CREATE`: a texture of shape `desc` at the guest-picked `gpu_address`
/// (device-placed, `heap_obj_id` 0) or at `heap_offset` in `heap_obj_id`, with
/// the guest-picked `resource_id`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxTextureCreate {
    /// [`RGFX_CMD_TEXTURE_CREATE`].
    pub header: RgfxMsgHeader,
    /// The new texture's guest-picked id.
    pub obj_id: u64,
    /// The texture's guest-picked GPU address when device-placed.
    pub gpu_address: u64,
    /// The texture's guest-picked resource id.
    pub resource_id: u64,
    /// The heap the texture lies in, or 0 when device-placed at `gpu_address`.
    pub heap_obj_id: u64,
    /// The texture's offset in the heap, when `heap_obj_id` is not 0.
    pub heap_offset: u64,
    /// The texture's shape.
    pub desc: RgfxTextureDescriptor,
}

/// `TEXTURE_FROM_BUFFER`: a texture of shape `desc` over the buffer
/// `buffer_obj_id` from `offset`, each row `bytes_per_row` apart, with the
/// guest-picked `resource_id`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxTextureFromBuffer {
    /// [`RGFX_CMD_TEXTURE_FROM_BUFFER`].
    pub header: RgfxMsgHeader,
    /// The new texture's guest-picked id.
    pub obj_id: u64,
    /// The new texture's guest-picked resource id.
    pub resource_id: u64,
    /// The buffer the texture reads.
    pub buffer_obj_id: u64,
    /// The texture's first byte in the buffer.
    pub offset: u64,
    /// Bytes from one row's start to the next's.
    pub bytes_per_row: u64,
    /// The texture's shape.
    pub desc: RgfxTextureDescriptor,
}

/// `LINEAR_TEXTURE_FROM_BUFFER`: a texture of shape `desc` over the buffer
/// `buffer_obj_id` from `offset`, each row `bytes_per_row` apart, with the
/// guest-picked `resource_id`, made by the platform's
/// `newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:` with
/// `bytes_per_image`. Metal's spaces a 2D array's slices `bytes_per_row` times
/// the height apart whatever `bytes_per_image` (measured on the M3 Max under
/// macOS 27); the message carries it as the guest asked it.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxLinearTextureFromBuffer {
    /// [`RGFX_CMD_LINEAR_TEXTURE_FROM_BUFFER`].
    pub header: RgfxMsgHeader,
    /// The new texture's guest-picked id.
    pub obj_id: u64,
    /// The new texture's guest-picked resource id.
    pub resource_id: u64,
    /// The buffer the texture reads.
    pub buffer_obj_id: u64,
    /// The texture's first byte in the buffer.
    pub offset: u64,
    /// Bytes from one row's start to the next's.
    pub bytes_per_row: u64,
    /// The bytes between a 2D array's slices the guest asked for.
    pub bytes_per_image: u64,
    /// The texture's shape.
    pub desc: RgfxTextureDescriptor,
}

/// `TEXTURE_FROM_IOSURFACE`: a texture of shape `desc` over plane `plane` of
/// the IOSurface `iosurface_id` (its global `IOSurfaceID`). The texture lies
/// over the guest process's copy of that plane, the buffer `buffer_obj_id` from
/// `offset`, each row `bytes_per_row` apart, as `TEXTURE_FROM_BUFFER` places
/// one. The guest syncs that copy with the surface's own pages: from the pages
/// when the CPU wrote the surface since, before a commit, and back to them when
/// a command buffer that wrote the texture completes. The host names the
/// texture by the surface's id, so a present of the surface finds it.
///
/// `writers_offset` is where in the buffer the copy's [`RgfxSurfaceWriters`]
/// lie ([`RGFX_SURFACE_NO_WRITERS`] for none): the command buffers committed
/// that write the copy, which a present of the surface waits for.
///
/// The surface's facts ([`RgfxSurfaceFacts`]) may follow the struct, at
/// [`RGFX_SURFACE_FACTS_OFFSET`]: a message whose `size` is at least
/// `RGFX_SURFACE_FACTS_OFFSET + size_of::<RgfxSurfaceFacts>()` carries them, one
/// of `RGFX_SURFACE_FACTS_OFFSET` none, and a size between the two is
/// malformed.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxTextureFromIOSurface {
    /// [`RGFX_CMD_TEXTURE_FROM_IOSURFACE`].
    pub header: RgfxMsgHeader,
    /// The new texture's guest-picked id.
    pub obj_id: u64,
    /// The new texture's guest-picked resource id.
    pub resource_id: u64,
    /// The buffer over the guest's copy of the plane, which the texture reads.
    pub buffer_obj_id: u64,
    /// The texture's first byte in the buffer.
    pub offset: u64,
    /// Bytes from one row's start to the next's in the buffer.
    pub bytes_per_row: u64,
    /// The surface's global `IOSurfaceID`.
    pub iosurface_id: u32,
    /// The surface's plane the texture shows (0 for a surface of one plane).
    pub plane: u32,
    /// The texture's shape.
    pub desc: RgfxTextureDescriptor,
    /// The copy's [`RgfxSurfaceWriters`], bytes into the buffer;
    /// [`RGFX_SURFACE_NO_WRITERS`] for none.
    pub writers_offset: u64,
}

/// Where a `TEXTURE_FROM_IOSURFACE` message's [`RgfxSurfaceFacts`] start: right
/// after its [`RgfxTextureFromIOSurface`].
pub const RGFX_SURFACE_FACTS_OFFSET: u32 = 120;

/// The facts of the surface a `TEXTURE_FROM_IOSURFACE` texture lies over, as
/// the guest's IOSurface API gives them when the texture is made, after the
/// [`RgfxTextureFromIOSurface`] at [`RGFX_SURFACE_FACTS_OFFSET`]: what a texture
/// over a surface reads of the surface besides its plane, as Metal's does (a
/// two-plane YCbCr texture over plane 0 reads plane 1's chroma, converted by
/// the surface's range and matrix). `chroma_offset` and `chroma_bytes_per_row`
/// place plane 1 in the guest's copy of the surface, the message's buffer, as
/// `offset` and `bytes_per_row` place the texture's own plane.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSurfaceFacts {
    /// `IOSurfaceGetPixelFormat`: the surface's OSType, its four characters
    /// big-endian (`'x420'` is `0x7834_3230`).
    pub surface_pixel_format: u32,
    /// `IOSurfaceGetPlaneCount`: 0 for a surface without planes.
    pub plane_count: u32,
    /// `IOSurfaceGetWidth`: the surface's width, its plane 0's.
    pub width: u32,
    /// `IOSurfaceGetHeight`: the surface's height, its plane 0's.
    pub height: u32,
    /// Plane 1's first byte in the message's buffer; 0 for a surface of fewer
    /// than two planes.
    pub chroma_offset: u64,
    /// Bytes from one of plane 1's rows to the next in the message's buffer; 0
    /// for a surface of fewer than two planes.
    pub chroma_bytes_per_row: u64,
    /// The surface's YCbCr matrix as the texture is made, one of
    /// `RGFX_YCBCR_MATRIX_*`.
    pub ycbcr_matrix: u32,
    /// Zero.
    pub reserved_24: u32,
}

/// [`RgfxSurfaceFacts::ycbcr_matrix`] of a surface with no
/// `IOSurfaceYCbCrMatrix` value.
pub const RGFX_YCBCR_MATRIX_NONE: u32 = 0;
/// [`RgfxSurfaceFacts::ycbcr_matrix`]: the value `ITU_R_601_4`.
pub const RGFX_YCBCR_MATRIX_ITU_R_601_4: u32 = 1;
/// [`RgfxSurfaceFacts::ycbcr_matrix`]: the value `ITU_R_709_2`.
pub const RGFX_YCBCR_MATRIX_ITU_R_709_2: u32 = 2;
/// [`RgfxSurfaceFacts::ycbcr_matrix`]: the value `ITU_R_2020`.
pub const RGFX_YCBCR_MATRIX_ITU_R_2020: u32 = 3;
/// [`RgfxSurfaceFacts::ycbcr_matrix`]: the value `SMPTE_240M_1995`.
pub const RGFX_YCBCR_MATRIX_SMPTE_240M_1995: u32 = 4;
/// [`RgfxSurfaceFacts::ycbcr_matrix`]: the value `DCI_P3`.
pub const RGFX_YCBCR_MATRIX_DCI_P3: u32 = 5;
/// [`RgfxSurfaceFacts::ycbcr_matrix`]: the value `P3_D65`.
pub const RGFX_YCBCR_MATRIX_P3_D65: u32 = 6;
/// [`RgfxSurfaceFacts::ycbcr_matrix`]: any other value, a string of another
/// text or a value that is no string.
pub const RGFX_YCBCR_MATRIX_OTHER: u32 = 7;

/// [`RgfxTextureFromIOSurface::writers_offset`] of a copy with no
/// [`RgfxSurfaceWriters`].
pub const RGFX_SURFACE_NO_WRITERS: u64 = u64::MAX;

/// The slots of [`RgfxSurfaceWriters`].
pub const RGFX_SURFACE_WRITER_SLOTS: usize = 4;

/// One command queue's latest command buffer that writes a surface plane's
/// copy: the queue's id and the buffer's `seqno` on the queue's submission ring.
/// A slot no queue holds has `queue_obj_id` 0.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSurfaceWriter {
    /// The queue's id; 0 for a free slot.
    pub queue_obj_id: u64,
    /// The latest `COMMAND_BUFFER` on the queue's ring that writes the copy.
    pub seqno: u64,
}

/// The command buffers committed that write a surface plane's copy, one slot
/// per queue, kept by the guest processes that hold the copy, in the copy's
/// block ([`RgfxTextureFromIOSurface::writers_offset`]). Each committing process
/// raises its queue's `seqno` once the buffer is on the queue's ring, before its
/// commit returns, so a present the process (or one it hands the surface to)
/// makes after that finds it; a present of the surface waits for each named
/// buffer's completion before the display reads the copy.
///
/// The guest writes a slot's `seqno` with 0 before it gives the slot to another
/// queue and the new seqno after, so a reader that reads `queue_obj_id`, then
/// `seqno`, then `queue_obj_id` again, and finds the two ids alike, reads a
/// queue's own seqno (or 0).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSurfaceWriters {
    /// The slots.
    pub writers: [RgfxSurfaceWriter; RGFX_SURFACE_WRITER_SLOTS],
}

/// `SAMPLER_CREATE`: a sampler as `MTLSamplerDescriptor` gives it. Each enum is
/// its Metal raw value; `lod_min_clamp_bits`/`lod_max_clamp_bits` are the `f32`
/// clamps' bits (so the wire stays all-integer and comparable); `flags` packs
/// the booleans.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSamplerCreate {
    /// [`RGFX_CMD_SAMPLER_CREATE`].
    pub header: RgfxMsgHeader,
    /// The new sampler's guest-picked id.
    pub obj_id: u64,
    /// `MTLSamplerMinMagFilter` for minification.
    pub min_filter: u32,
    /// `MTLSamplerMinMagFilter` for magnification.
    pub mag_filter: u32,
    /// `MTLSamplerMipFilter`.
    pub mip_filter: u32,
    /// `MTLSamplerAddressMode` for s.
    pub s_address_mode: u32,
    /// `MTLSamplerAddressMode` for t.
    pub t_address_mode: u32,
    /// `MTLSamplerAddressMode` for r.
    pub r_address_mode: u32,
    /// `MTLSamplerBorderColor`.
    pub border_color: u32,
    /// `MTLCompareFunction` for the sampler's compare.
    pub compare_function: u32,
    /// The maximum anisotropy.
    pub max_anisotropy: u32,
    /// The minimum LOD clamp, as `f32` bits.
    pub lod_min_clamp_bits: u32,
    /// The maximum LOD clamp, as `f32` bits.
    pub lod_max_clamp_bits: u32,
    /// Bit 0: normalized coordinates. Bit 1: LOD average. Bit 2: usable from an
    /// argument buffer.
    pub flags: u32,
}

/// Where a `SAMPLER_CREATE` message's [`RgfxSamplerCreateExt`] starts: right after its
/// [`RgfxSamplerCreate`]. A message whose `size` is at least
/// `RGFX_SAMPLER_CREATE_EXT_OFFSET + size_of::<RgfxSamplerCreateExt>()` carries it; one of
/// `RGFX_SAMPLER_CREATE_EXT_OFFSET` none, and its sampler has Metal's defaults for both fields.
/// The host does not read it yet.
pub const RGFX_SAMPLER_CREATE_EXT_OFFSET: u32 = 72;

/// The descriptor fields a `SAMPLER_CREATE` carries after the [`RgfxSamplerCreate`] at
/// [`RGFX_SAMPLER_CREATE_EXT_OFFSET`].
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSamplerCreateExt {
    /// `MTLSamplerReductionMode`.
    pub reduction_mode: u32,
    /// `lodBias`, as `f32` bits.
    pub lod_bias_bits: u32,
}

/// One face's stencil state, as `MTLStencilDescriptor` gives it.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxStencilDescriptor {
    /// `MTLCompareFunction` for the stencil test.
    pub stencil_compare: u32,
    /// `MTLStencilOperation` when the stencil test fails.
    pub stencil_fail_op: u32,
    /// `MTLStencilOperation` when the depth test fails.
    pub depth_fail_op: u32,
    /// `MTLStencilOperation` when both tests pass.
    pub pass_op: u32,
    /// The bits the stencil test reads.
    pub read_mask: u32,
    /// The bits a stencil write may change.
    pub write_mask: u32,
}

/// `DEPTH_STENCIL_CREATE`: a depth/stencil state as `MTLDepthStencilDescriptor`
/// gives it.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxDepthStencilCreate {
    /// [`RGFX_CMD_DEPTH_STENCIL_CREATE`].
    pub header: RgfxMsgHeader,
    /// The new state's guest-picked id.
    pub obj_id: u64,
    /// `MTLCompareFunction` for the depth test.
    pub depth_compare: u32,
    /// Whether a depth write is kept: 0 discards it.
    pub depth_write_enabled: u32,
    /// The front faces' stencil state.
    pub front: RgfxStencilDescriptor,
    /// The back faces' stencil state.
    pub back: RgfxStencilDescriptor,
}

/// `LIBRARY_CREATE`: a library from `byte_length` bytes of `.metallib` or AIR
/// that follow this header at [`RGFX_LIBRARY_BYTES_OFFSET`], forwarded to the
/// host unchanged. The message's `size` is `RGFX_LIBRARY_BYTES_OFFSET +
/// byte_length` rounded up to a multiple of 8. Answered by [`RGFX_EVT_COMPLETE`]
/// once the library is made: [`RGFX_STATUS_OK`], [`RGFX_STATUS_REFUSED`] where the
/// GPU could not build the bytes, or [`RGFX_STATUS_INVALID_ARGUMENT`] for a message
/// shorter than its bytes.
///
/// With [`RGFX_LIBRARY_FLAG_KEYED`] in `flags`, `key` is the bytes' content key
/// ([`RgfxLibraryLookup`]): the device keeps the library under it, for later lookups,
/// where the key it computes from the bytes is the same.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxLibraryCreate {
    /// [`RGFX_CMD_LIBRARY_CREATE`].
    pub header: RgfxMsgHeader,
    /// The new library's guest-picked id.
    pub obj_id: u64,
    /// The length of the `.metallib`/AIR bytes that follow.
    pub byte_length: u64,
    /// [`RGFX_LIBRARY_FLAG_KEYED`], or 0.
    pub flags: u32,
    /// Zero.
    pub reserved_24: u32,
    /// The bytes' content key, where `flags` has [`RGFX_LIBRARY_FLAG_KEYED`].
    pub key: [u8; 32],
}

/// `LIBRARY_FROM_REGION`: one chunk of a library's `.metallib`/AIR bytes, which
/// the guest copied to `region_offset` in the device's region: the `byte_length`
/// bytes at `chunk_offset` of the library's `total_length`. A library goes in one
/// or more chunks in order, the first at `chunk_offset` 0; the host appends each
/// and makes the library, unchanged bytes, when the chunks reach `total_length`.
/// Once the host has copied a chunk's bytes out of the region it writes
/// `ack_value` as a little-endian `u64` at `ack_offset` (8-aligned, in the
/// region), so the guest may write that block again: the ring's consumer count
/// moves when the host takes the message, not when it carries it out. The chunk
/// and the ack slot lie in the process's own blocks, as [`RgfxHeapOverHostMemory`]'s
/// block does; a span outside the processes' blocks, or a chunk out of order, is a
/// guest protocol error.
///
/// The library's last chunk, the one whose `chunk_offset + byte_length` reaches
/// `total_length` (or overflows), is answered as [`RgfxLibraryCreate`] is, and its
/// `flags` and `key` are the library's as there; the other chunks are not answered.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxLibraryFromRegion {
    /// [`RGFX_CMD_LIBRARY_FROM_REGION`].
    pub header: RgfxMsgHeader,
    /// The new library's guest-picked id.
    pub obj_id: u64,
    /// The chunk's bytes' offset in the device's region.
    pub region_offset: u64,
    /// The chunk's length.
    pub byte_length: u64,
    /// The whole library's length.
    pub total_length: u64,
    /// Where the chunk lies in the library.
    pub chunk_offset: u64,
    /// Where the host writes `ack_value` once it has copied the chunk out.
    pub ack_offset: u64,
    /// The value the host writes at `ack_offset`.
    pub ack_value: u64,
    /// [`RGFX_LIBRARY_FLAG_KEYED`], or 0.
    pub flags: u32,
    /// Zero.
    pub reserved_4c: u32,
    /// The library's content key, where `flags` has [`RGFX_LIBRARY_FLAG_KEYED`].
    pub key: [u8; 32],
}

/// Where a `LIBRARY_CREATE` message's bytes start: right after its header.
pub const RGFX_LIBRARY_BYTES_OFFSET: u32 = 72;

/// A library create's `flags`: its `key` is the bytes' content key.
pub const RGFX_LIBRARY_FLAG_KEYED: u32 = 1;

/// `LIBRARY_LOOKUP`: library `obj_id` from the libraries the device holds, by their content key.
///
/// The bytes keyed are a `.metallib` (`MTLB`), the AIR slice of a fat file. `key` is the SHA-256
/// of the 15 bytes `rgfx-library-v1`, then the bytes before the bitcode section, then the bytes
/// after it; the section is the `u64` offset and size at 0x48 and 0x50 of the `.metallib`'s
/// header. Each function's bitcode is named by its `HASH` tag, its SHA-256, which lies outside
/// the section, so the key names every byte a function is built from. `slice_size` is the bytes'
/// length.
///
/// Answered by [`RGFX_EVT_COMPLETE`]: [`RGFX_STATUS_OK`] where the device holds a library of
/// `slice_size` bytes under `key`, which `obj_id` now names; [`RGFX_STATUS_NOT_FOUND`] where it
/// holds none, and the guest makes the library from its bytes; [`RGFX_STATUS_INVALID_ARGUMENT`]
/// for a message shorter than this record.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxLibraryLookup {
    /// [`RGFX_CMD_LIBRARY_LOOKUP`].
    pub header: RgfxMsgHeader,
    /// The library's guest-picked id.
    pub obj_id: u64,
    /// The keyed bytes' length.
    pub slice_size: u64,
    /// The bytes' content key.
    pub key: [u8; 32],
}

/// The version of the [`RgfxDeviceInfo`] layout the host writes.
pub const RGFX_DEVICE_INFO_VERSION: u32 = 3;

/// The most bytes of a device name [`RgfxDeviceInfo`] carries: Vulkan's
/// `VK_MAX_PHYSICAL_DEVICE_NAME_SIZE`, so no host GPU's name is cut short.
pub const RGFX_DEVICE_NAME_LEN: usize = 256;

/// The most bytes of the device's `architecture.name` [`RgfxDeviceInfo`] carries.
pub const RGFX_DEVICE_ARCHITECTURE_NAME_LEN: usize = 64;

/// The most families [`RgfxDeviceInfo`] answers `supportsFamily:` of.
pub const RGFX_DEVICE_INFO_FAMILIES: usize = 32;

/// The most feature sets [`RgfxDeviceInfo`] answers `supportsFeatureSet:` of.
pub const RGFX_DEVICE_INFO_FEATURE_SETS: usize = 32;

/// The integer identity answers [`RgfxDeviceInfo::sizes`] holds.
pub const RGFX_DEVICE_INFO_SIZES: usize = 19;

/// The `BOOL` identity answers [`RgfxDeviceInfo::identity_flags`] holds.
pub const RGFX_DEVICE_INFO_IDENTITY_FLAGS: usize = 19;

/// The counter sampling points [`RgfxDeviceInfo::counter_sampling`] answers.
pub const RGFX_DEVICE_INFO_COUNTER_POINTS: usize = 5;

/// The most pixel formats [`RgfxDeviceInfo::formats`] answers.
pub const RGFX_DEVICE_INFO_FORMATS: usize = 160;

/// The sparse tiles [`RgfxFormatAnswers::sparse_tiles`] holds: sample counts 1, 2 and 4, each
/// at pages of 16, 64 and 256 KiB.
pub const RGFX_DEVICE_INFO_SPARSE_TILES: usize = 9;

/// One pixel format's answers in [`RgfxDeviceInfo`].
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxFormatAnswers {
    /// The raw `MTLPixelFormat`.
    pub format: u32,
    /// `minimumTextureBufferAlignmentForPixelFormat:`; 0 for a format the host has no answer
    /// of (one Metal asserts on, among them).
    pub texture_buffer_alignment: u32,
    /// `sparseTileSizeWithTextureType:pixelFormat:sampleCount:sparsePageSize:`, the same for
    /// every texture type: width, height and depth, at sample counts 1, 2 and 4 (major) and
    /// pages of 16, 64 and 256 KiB; all 0 where the host has no answer.
    pub sparse_tiles: [[u16; 3]; RGFX_DEVICE_INFO_SPARSE_TILES],
    /// Zero.
    pub reserved_3e: [u8; 2],
}

/// The device-info caps blob the host writes once at device init, read by the
/// guest bundle before it makes any queue. It holds every answer of the guest's
/// `MTLDevice` that does not change while the device lives: the host device's
/// capabilities (what Metal or metal2vulkan answers), but where the lane's own
/// device answers otherwise (the host's `device_info` table says where). The
/// bundle answers `supportsFamily:`, `supportsFeatureSet:`, the identity, the
/// limits and the feature queries from it. A query the host platform cannot
/// answer (a runtime gap) reads 0 here, never a spurious yes. This is a flat
/// caps page, not a ring message.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout)]
pub struct RgfxDeviceInfo {
    /// [`RGFX_DEVICE_INFO_VERSION`].
    pub version: u32,
    /// The bytes of `name` in use, at most [`RGFX_DEVICE_NAME_LEN`].
    pub name_len: u32,
    /// The bytes of `architecture_name` in use, at most
    /// [`RGFX_DEVICE_ARCHITECTURE_NAME_LEN`]; 0 where the host has no answer.
    pub architecture_name_len: u32,
    /// The entries of `family_values` and `family_answers` in use.
    pub family_count: u32,
    /// The entries of `feature_set_values` and `feature_set_answers` in use.
    pub feature_set_count: u32,
    /// The entries of `formats` in use.
    pub format_count: u32,
    /// `registryID`.
    pub registry_id: u64,
    /// `gpuCoreCount`, or 0 when the device does not answer it.
    pub gpu_core_count: u64,
    /// `maxThreadsPerThreadgroup`: width, height, depth.
    pub max_threads_per_threadgroup: [u64; 3],
    /// `maxVertexAmplificationCount`.
    pub max_vertex_amplification_count: u64,
    /// `heapBufferSizeAndAlignWithLength:options:`(`vm_page_size`, Private): the
    /// size then the alignment; both 0 when the query is a host gap.
    pub heap_buffer_page_private: [u64; 2],
    /// `minimumLinearTextureAlignmentForPixelFormat:`(BGRA8Unorm); 0 on a gap.
    pub min_linear_alignment_bgra8: u64,
    /// The 27 `NSUInteger` limits, in the host `Limits` field order
    /// (`device_linear_texture_alignment_bytes` first, `max_viewport_count`
    /// last); the host fills them in that order.
    pub limits: [u64; 27],
    /// The integer identity answers, in the host `IdentitySizes` field order
    /// (`recommended_max_working_set_size` first, `double_fp_config` last); 0
    /// on a gap.
    pub sizes: [u64; RGFX_DEVICE_INFO_SIZES],
    /// The raw `MTLGPUFamily` values `family_answers` answers, by position.
    pub family_values: [i64; RGFX_DEVICE_INFO_FAMILIES],
    /// The raw `MTLFeatureSet` values `feature_set_answers` answers, by position.
    pub feature_set_values: [u64; RGFX_DEVICE_INFO_FEATURE_SETS],
    /// `supportsFamily:` of each of `family_values`, 1 or 0; a family the blob does
    /// not list names no family the host knows, and is no.
    pub family_answers: [u8; RGFX_DEVICE_INFO_FAMILIES],
    /// `supportsFeatureSet:` of each of `feature_set_values`, 1 or 0; a feature set
    /// the blob does not list is no.
    pub feature_set_answers: [u8; RGFX_DEVICE_INFO_FEATURE_SETS],
    /// `supportsTextureSampleCount:` of 1, 2, 4, 8 by position; a gap reads 0.
    pub sample_counts: [u8; 4],
    /// `supportsPrimitiveType:` of 0..10 by position; a gap reads 0.
    pub primitive_types: [u8; 10],
    /// `supportsTextureWriteRoundingMode:` of 1, 2 by position; a gap reads 0.
    pub write_rounding_modes: [u8; 2],
    /// The 25 `BOOL` properties, in the host `Flags` field order
    /// (`are_raster_order_groups_supported` first, `supports_tile_shaders` last),
    /// 1 or 0 by position.
    pub flags: [u8; 25],
    /// The `BOOL` identity answers, in the host `IdentityFlags` field order
    /// (`is_low_power` first, `supports_placement_sparse` last), 1 or 0 by
    /// position; a gap reads 0.
    pub identity_flags: [u8; RGFX_DEVICE_INFO_IDENTITY_FLAGS],
    /// `supportsCounterSampling:` of the stage, draw, dispatch, tile and blit
    /// boundaries by position; a gap reads 0.
    pub counter_sampling: [u8; RGFX_DEVICE_INFO_COUNTER_POINTS],
    /// `FEATURES_LO`'s low byte as the device answers it (`RGFX_FEATURE_*`), for the processes,
    /// which read no register; 0 from a host older than the field, which wrote it as zero.
    pub features_lo: u8,
    /// `samplerReductionModeSupport` as the host's platform answers it, from a host that reads
    /// the sampler's [`RgfxSamplerCreateExt`]; 0 from every other host, which wrote it as zero.
    /// No host writes it yet.
    pub sampler_reduction_support: u8,
    /// Zero.
    pub reserved_453: [u8; 5],
    /// The device `name`, UTF-8, `name_len` bytes used, the rest 0.
    pub name: [u8; RGFX_DEVICE_NAME_LEN],
    /// The device's `architecture.name`, UTF-8, `architecture_name_len` bytes
    /// used, the rest 0.
    pub architecture_name: [u8; RGFX_DEVICE_ARCHITECTURE_NAME_LEN],
    /// Each pixel format's answers, `format_count` in use; a format the blob does
    /// not list has none.
    pub formats: [RgfxFormatAnswers; RGFX_DEVICE_INFO_FORMATS],
}

impl Default for RgfxDeviceInfo {
    /// Every byte zero.
    fn default() -> Self {
        <Self as zerocopy::FromZeros>::new_zeroed()
    }
}

/// The version of the [`RgfxVideoCaps`] layout the host writes; 0, a host's that writes none.
pub const RGFX_VIDEO_CAPS_VERSION: u32 = 1;

/// The most codec answers [`RgfxVideoCaps`] holds.
pub const RGFX_VIDEO_CAPS_CODECS: usize = 16;

/// [`RgfxVideoCodecCaps::codec`] and [`RgfxVideoSessionCreate::codec`]: H.264.
pub const RGFX_VIDEO_CODEC_H264: u32 = 1;
/// [`RgfxVideoCodecCaps::codec`] and [`RgfxVideoSessionCreate::codec`]: HEVC.
pub const RGFX_VIDEO_CODEC_HEVC: u32 = 2;

/// [`RgfxVideoCodecCaps::direction`] and [`RgfxVideoSessionCreate::direction`]: decode.
pub const RGFX_VIDEO_DECODE: u32 = 1;
/// [`RgfxVideoCodecCaps::direction`] and [`RgfxVideoSessionCreate::direction`]: encode.
pub const RGFX_VIDEO_ENCODE: u32 = 2;

/// [`RgfxVideoCodecCaps::profiles`] of H.264: Baseline (`profile_idc` 66; a Constrained
/// Baseline stream decodes as one).
pub const RGFX_VIDEO_H264_BASELINE: u32 = 1 << 0;
/// [`RgfxVideoCodecCaps::profiles`] of H.264: Main (77).
pub const RGFX_VIDEO_H264_MAIN: u32 = 1 << 1;
/// [`RgfxVideoCodecCaps::profiles`] of H.264: High (100).
pub const RGFX_VIDEO_H264_HIGH: u32 = 1 << 2;

/// [`RgfxVideoCodecCaps::profiles`] of HEVC: Main (`general_profile_idc` 1; a Main Still
/// Picture stream decodes as one).
pub const RGFX_VIDEO_HEVC_MAIN: u32 = 1 << 0;
/// [`RgfxVideoCodecCaps::profiles`] of HEVC: Main 10 (2).
pub const RGFX_VIDEO_HEVC_MAIN_10: u32 = 1 << 1;

/// [`RgfxVideoCodecCaps::formats`]: 8-bit 4:2:0 in two planes, NV12 (`'420v'`, `'420f'`).
pub const RGFX_VIDEO_FORMAT_NV12: u32 = 1 << 0;
/// [`RgfxVideoCodecCaps::formats`]: 10-bit 4:2:0 in two planes, P010 (`'x420'`, `'xf20'`).
pub const RGFX_VIDEO_FORMAT_P010: u32 = 1 << 1;
/// [`RgfxVideoCaps::flags`]: the host lands a decoded picture in its output surface's device
/// cache ([`RGFX_FEATURE_SURFACE_CACHE`]), in the GPU's own memory, where the surface's block has
/// one. The guest's video driver then asks its output surfaces' blocks cached, so a picture
/// reaches the pages only at a CPU lock of the surface, and the GPU's other uses of it read it
/// where it landed. Without it, a picture lands in the surface's pages.
pub const RGFX_VIDEO_CAPS_OUTPUT_CACHED: u32 = 1 << 0;

/// What the host's hardware takes of one codec in one direction, as its video API answers it.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVideoCodecCaps {
    /// An `RGFX_VIDEO_CODEC_*`.
    pub codec: u32,
    /// [`RGFX_VIDEO_DECODE`] or [`RGFX_VIDEO_ENCODE`].
    pub direction: u32,
    /// The profiles taken, `RGFX_VIDEO_<codec>_*` bits.
    pub profiles: u32,
    /// The highest level taken, as the codec numbers it (H.264's `level_idc`, 52 for 5.2; HEVC's
    /// `general_level_idc`, 153 for 5.1).
    pub max_level: u32,
    /// The smallest coded picture: width, height.
    pub min_size: [u32; 2],
    /// The largest coded picture: width, height.
    pub max_size: [u32; 2],
    /// The most pictures the decoded picture buffer holds, the current one included.
    pub max_dpb_pictures: u32,
    /// The most pictures one picture references.
    pub max_references: u32,
    /// The output formats taken, `RGFX_VIDEO_FORMAT_*` bits.
    pub formats: u32,
    /// What a decode reads its access unit in place at: where the access unit starts at a
    /// multiple of this many bytes in its buffer, and its length, zeros padding its end, is one
    /// too; elsewhere the host copies it. 1 where any place and length do; 0 for an encode.
    pub bitstream_alignment: u32,
}

/// The video caps the host writes at the device's init, at [`RGFX_VIDEO_CAPS_OFFSET`]: one
/// [`RgfxVideoCodecCaps`] for each codec and direction its hardware takes, which the guest's
/// video driver lists, and none for the rest, which the guest's software codecs serve.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVideoCaps {
    /// [`RGFX_VIDEO_CAPS_VERSION`]; 0 from a host that writes none.
    pub version: u32,
    /// The entries of `codecs` in use.
    pub codec_count: u32,
    /// The codecs' answers, `codec_count` in use.
    pub codecs: [RgfxVideoCodecCaps; RGFX_VIDEO_CAPS_CODECS],
    /// `RGFX_VIDEO_CAPS_*` bits; 0 from a host older than them.
    pub flags: u32,
    /// Zero.
    pub reserved_30c: u32,
}

/// `VIDEO_SESSION_CREATE`: a video session.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVideoSessionCreate {
    /// [`RGFX_CMD_VIDEO_SESSION_CREATE`].
    pub header: RgfxMsgHeader,
    /// The session's guest-picked id.
    pub obj_id: u64,
    /// An `RGFX_VIDEO_CODEC_*`.
    pub codec: u32,
    /// [`RGFX_VIDEO_DECODE`] or [`RGFX_VIDEO_ENCODE`].
    pub direction: u32,
}

// ---------------------------------------------------------------------------
// Stage 3: render and compute pipeline creates and their reflection reply
// (RGFX_CMD_RENDER_PIPELINE_CREATE 0x20d, RGFX_CMD_COMPUTE_PIPELINE_CREATE
// 0x20e, RGFX_EVT_PIPELINE_REFLECTION 0x402). A pipeline names its functions by
// library `obj_id` + entry-point name bytes + function-constant values; the
// create blocks on the host's reflection reply, matched to the create by its
// `seqno`. Each create carries the whole descriptor the guest read from the
// Metal pipeline descriptor, so the wire is the stable guest ABI; which parts
// the host applies today is bounded by the host platform's pipeline creators
// (see the host decoder and docs).
// ---------------------------------------------------------------------------

/// A pipeline reflection reply's status: the pipeline was created.
pub const RGFX_PIPELINE_OK: u32 = 0;
/// A pipeline reflection reply's status: the host made no pipeline (a nil or
/// invalid function, or Metal refused the descriptor); no handle was stored.
pub const RGFX_PIPELINE_REFUSED: u32 = 1;

/// A pipeline reflection reply's kind: a render pipeline.
pub const RGFX_PIPELINE_KIND_RENDER: u32 = 0;
/// A pipeline reflection reply's kind: a compute pipeline.
pub const RGFX_PIPELINE_KIND_COMPUTE: u32 = 1;

/// [`RgfxPipelineReflection`] `flags`: the state supports indirect command
/// buffers (`supportIndirectCommandBuffers`).
pub const RGFX_PIPELINE_REFLECT_SUPPORT_ICB: u32 = 1 << 0;
/// [`RgfxPipelineReflection`] `flags`: a render pipeline's threadgroup size
/// matches its tile size (`threadgroupSizeMatchesTileSize`).
pub const RGFX_PIPELINE_REFLECT_THREADGROUP_MATCHES_TILE: u32 = 1 << 1;

/// [`RgfxRenderPipelineCreate`] `flags`: `alphaToCoverageEnabled`.
pub const RGFX_RENDER_PIPELINE_ALPHA_TO_COVERAGE: u32 = 1 << 0;
/// [`RgfxRenderPipelineCreate`] `flags`: `alphaToOneEnabled`.
pub const RGFX_RENDER_PIPELINE_ALPHA_TO_ONE: u32 = 1 << 1;
/// [`RgfxRenderPipelineCreate`] `flags`: `rasterizationEnabled` (set when the
/// pipeline rasterizes; a vertex-only pipeline clears it).
pub const RGFX_RENDER_PIPELINE_RASTERIZATION_ENABLED: u32 = 1 << 2;
/// [`RgfxRenderPipelineCreate`] `flags`: `supportIndirectCommandBuffers`.
pub const RGFX_RENDER_PIPELINE_SUPPORT_ICB: u32 = 1 << 3;

/// [`RgfxComputePipelineCreate`] `flags`:
/// `threadGroupSizeIsMultipleOfThreadExecutionWidth`.
pub const RGFX_COMPUTE_PIPELINE_TG_MULTIPLE_OF_EXEC_WIDTH: u32 = 1 << 0;
/// [`RgfxComputePipelineCreate`] `flags`: `supportIndirectCommandBuffers`.
pub const RGFX_COMPUTE_PIPELINE_SUPPORT_ICB: u32 = 1 << 1;

/// The bytes of [`RgfxFunctionConstant::value`]: the widest constant's, a `long4` or `ulong4`.
pub const RGFX_FUNCTION_CONSTANT_VALUE_BYTES: usize = 32;

/// One function-constant value a pipeline specialises its function with, by the
/// constant's index. The guest resolves every value its process set on the
/// `MTLFunctionConstantValues` (by index, by range or by name) against the
/// function's own constants, as Metal does, and sends one record for each of the
/// function's constants it gives a value: the function's constants it declares
/// and nothing else.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxFunctionConstant {
    /// The constant's index (`[[function_constant(index)]]`).
    pub index: u32,
    /// `MTLDataType` of the constant's value, which is the constant's own type.
    pub data_type: u32,
    /// The value's bytes, packed as the type's lanes are (a `float3` is 12 bytes), from the
    /// first; the rest are zero.
    pub value: [u8; RGFX_FUNCTION_CONSTANT_VALUE_BYTES],
}

/// One colour attachment of a render pipeline
/// (`MTLRenderPipelineColorAttachmentDescriptor`). Each enum is its Metal raw
/// value (`MTLPixelFormat`, `MTLBlendFactor`, `MTLBlendOperation`,
/// `MTLColorWriteMask`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxColorAttachmentDesc {
    /// `MTLPixelFormat`; 0 (`Invalid`) for an unused attachment.
    pub pixel_format: u32,
    /// Whether blending is enabled: 0 off.
    pub blending_enabled: u32,
    /// `MTLBlendFactor` for the source RGB.
    pub source_rgb_blend_factor: u32,
    /// `MTLBlendFactor` for the destination RGB.
    pub destination_rgb_blend_factor: u32,
    /// `MTLBlendOperation` for RGB.
    pub rgb_blend_operation: u32,
    /// `MTLBlendFactor` for the source alpha.
    pub source_alpha_blend_factor: u32,
    /// `MTLBlendFactor` for the destination alpha.
    pub destination_alpha_blend_factor: u32,
    /// `MTLBlendOperation` for alpha.
    pub alpha_blend_operation: u32,
    /// `MTLColorWriteMask`.
    pub write_mask: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_24: u32,
}

/// One vertex buffer layout of a render pipeline's vertex descriptor
/// (`MTLVertexBufferLayoutDescriptor`), at `buffer_index`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVertexBufferLayout {
    /// The vertex buffer index this layout describes.
    pub buffer_index: u32,
    /// Bytes from one element to the next.
    pub stride: u32,
    /// `MTLVertexStepFunction`.
    pub step_function: u32,
    /// The step rate.
    pub step_rate: u32,
}

/// One vertex attribute of a render pipeline's vertex descriptor
/// (`MTLVertexAttributeDescriptor`), at `attribute_index`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVertexAttribute {
    /// The attribute index this record describes.
    pub attribute_index: u32,
    /// `MTLVertexFormat`.
    pub format: u32,
    /// The attribute's byte offset in its vertex buffer.
    pub offset: u32,
    /// The vertex buffer the attribute reads.
    pub buffer_index: u32,
}

/// [`RgfxLinkedFunction`] `list`: `MTLLinkedFunctions.functions`.
pub const RGFX_LINKED_FUNCTIONS: u32 = 0;
/// [`RgfxLinkedFunction`] `list`: `MTLLinkedFunctions.privateFunctions`.
pub const RGFX_LINKED_PRIVATE_FUNCTIONS: u32 = 1;
/// [`RgfxLinkedFunction`] `list`: `MTLLinkedFunctions.binaryFunctions`.
pub const RGFX_LINKED_BINARY_FUNCTIONS: u32 = 2;
/// [`RgfxLinkedFunction`] `list`: one of `MTLLinkedFunctions.groups`, the group its
/// `group_*` name.
pub const RGFX_LINKED_GROUP: u32 = 3;

/// One function a pipeline stage links (`MTLLinkedFunctions`), named as a stage's
/// function is: a library `obj_id`, an entry-point name in the message tail and a run
/// of [`RgfxFunctionConstant`] in the tail, with the `MTLFunctionOptions` the guest
/// made it with and, where it gave one, the descriptor's `specializedName`, the name
/// the function answers and a pipeline's visible-function references resolve against.
/// A stage's linked functions are a run of these in the tail, each list's in its
/// order; a group's functions carry the group's name.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxLinkedFunction {
    /// The library the function comes from.
    pub library_obj_id: u64,
    /// The entry-point name's byte offset from the message start.
    pub name_offset: u32,
    /// The entry-point name's length in bytes.
    pub name_len: u32,
    /// The function constants' byte offset from the message start.
    pub constants_offset: u32,
    /// The number of [`RgfxFunctionConstant`] for the function; 0 for a function made
    /// with none.
    pub constant_count: u32,
    /// The list the function is in: an `RGFX_LINKED_*`.
    pub list: u32,
    /// The `MTLFunctionOptions` the function was made with
    /// (`newFunctionWithDescriptor:`); 0 for the other forms.
    pub options: u32,
    /// For [`RGFX_LINKED_GROUP`], the group name's byte offset from the message
    /// start; 0 otherwise.
    pub group_offset: u32,
    /// For [`RGFX_LINKED_GROUP`], the group name's length in bytes; 0 otherwise.
    pub group_len: u32,
    /// The `specializedName`'s byte offset from the message start; 0 for none.
    pub specialized_name_offset: u32,
    /// The `specializedName`'s length in bytes; 0 for none.
    pub specialized_name_len: u32,
}

/// `RENDER_PIPELINE_CREATE`: a render pipeline from a descriptor, answered by
/// [`RgfxPipelineReflection`] on the queue's completion ring with this create's
/// `seqno`.
///
/// The vertex and fragment functions are each named by a library `obj_id` and an
/// entry-point name in the message tail (`*_name_offset`/`*_name_len`, byte
/// offsets from the message start), optionally specialised by a run of
/// [`RgfxFunctionConstant`] in the tail (`*_constants_offset`/`*_constant_count`).
/// The vertex descriptor's layouts and attributes are runs of
/// [`RgfxVertexBufferLayout`] and [`RgfxVertexAttribute`] in the tail. The colour
/// attachments are inline, `color_attachment_count` of
/// [`RgfxColorAttachmentDesc`] in use. The vertex and fragment stages' linked
/// functions (`vertexLinkedFunctions`, `fragmentLinkedFunctions`) are runs of
/// [`RgfxLinkedFunction`] in the tail. The optional pipeline `label` is tail
/// bytes (`label_len` 0 for none). The whole message's `size` is the fixed head
/// plus the tail, rounded up to a multiple of 8.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxRenderPipelineCreate {
    /// [`RGFX_CMD_RENDER_PIPELINE_CREATE`].
    pub header: RgfxMsgHeader,
    /// The new pipeline's guest-picked id.
    pub obj_id: u64,
    /// The library the vertex function comes from.
    pub vertex_library_obj_id: u64,
    /// The library the fragment function comes from.
    pub fragment_library_obj_id: u64,
    /// The vertex entry-point name's byte offset from the message start.
    pub vertex_name_offset: u32,
    /// The vertex entry-point name's length in bytes.
    pub vertex_name_len: u32,
    /// The fragment entry-point name's byte offset from the message start.
    pub fragment_name_offset: u32,
    /// The fragment entry-point name's length in bytes.
    pub fragment_name_len: u32,
    /// The pipeline label's byte offset from the message start (0-length: none).
    pub label_offset: u32,
    /// The pipeline label's length in bytes; 0 for no label.
    pub label_len: u32,
    /// The vertex function constants' byte offset from the message start.
    pub vertex_constants_offset: u32,
    /// The number of [`RgfxFunctionConstant`] for the vertex function.
    pub vertex_constant_count: u32,
    /// The fragment function constants' byte offset from the message start.
    pub fragment_constants_offset: u32,
    /// The number of [`RgfxFunctionConstant`] for the fragment function.
    pub fragment_constant_count: u32,
    /// The vertex buffer layouts' byte offset from the message start.
    pub vertex_layouts_offset: u32,
    /// The number of [`RgfxVertexBufferLayout`] in the vertex descriptor.
    pub vertex_layout_count: u32,
    /// The vertex attributes' byte offset from the message start.
    pub vertex_attributes_offset: u32,
    /// The number of [`RgfxVertexAttribute`] in the vertex descriptor.
    pub vertex_attribute_count: u32,
    /// `rasterSampleCount` (the raster/colour sample count).
    pub raster_sample_count: u32,
    /// The colour attachments in use, up to 8 (the length of `colors`).
    pub color_attachment_count: u32,
    /// `depthAttachmentPixelFormat` (`MTLPixelFormat`; 0 for none).
    pub depth_pixel_format: u32,
    /// `stencilAttachmentPixelFormat` (`MTLPixelFormat`; 0 for none).
    pub stencil_pixel_format: u32,
    /// The `RGFX_RENDER_PIPELINE_*` boolean flags.
    pub flags: u32,
    /// `inputPrimitiveTopology` (`MTLPrimitiveTopologyClass`).
    pub input_primitive_topology: u32,
    /// `maxVertexAmplificationCount`.
    pub max_vertex_amplification_count: u32,
    /// Pads `colors` to an 8-aligned offset; 0.
    pub reserved_7c: u32,
    /// The eight colour attachments; `color_attachment_count` are in use, the
    /// rest zeroed.
    pub colors: [RgfxColorAttachmentDesc; 8],
    /// The vertex stage's linked functions' byte offset from the message start.
    pub vertex_linked_offset: u32,
    /// The number of [`RgfxLinkedFunction`] the vertex stage links.
    pub vertex_linked_count: u32,
    /// The fragment stage's linked functions' byte offset from the message start.
    pub fragment_linked_offset: u32,
    /// The number of [`RgfxLinkedFunction`] the fragment stage links.
    pub fragment_linked_count: u32,
}

/// `COMPUTE_PIPELINE_CREATE`: a compute pipeline from a function, answered by
/// [`RgfxPipelineReflection`] with this create's `seqno`. The compute function is
/// named by `function_library_obj_id` and an entry-point name in the tail
/// (`name_offset`/`name_len`, byte offsets from the message start), optionally
/// specialised by `constant_count` [`RgfxFunctionConstant`] in the tail. The
/// stage-input descriptor's (`stageInputDescriptor`) layouts and attributes are
/// runs of [`RgfxVertexBufferLayout`] and [`RgfxVertexAttribute`] in the tail,
/// their `step_function` an `MTLStepFunction` and their `format` an
/// `MTLAttributeFormat`; the kernel's linked functions (`linkedFunctions`) are a
/// run of [`RgfxLinkedFunction`]. The message's `size` is the fixed head plus the
/// tail, rounded up to a multiple of 8.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxComputePipelineCreate {
    /// [`RGFX_CMD_COMPUTE_PIPELINE_CREATE`].
    pub header: RgfxMsgHeader,
    /// The new pipeline's guest-picked id.
    pub obj_id: u64,
    /// The library the compute function comes from.
    pub function_library_obj_id: u64,
    /// The compute entry-point name's byte offset from the message start.
    pub name_offset: u32,
    /// The compute entry-point name's length in bytes.
    pub name_len: u32,
    /// The function constants' byte offset from the message start.
    pub constants_offset: u32,
    /// The number of [`RgfxFunctionConstant`] for the function.
    pub constant_count: u32,
    /// The `RGFX_COMPUTE_PIPELINE_*` boolean flags.
    pub flags: u32,
    /// `maxTotalThreadsPerThreadgroup` the guest set, or 0 to leave Metal's.
    pub max_total_threads_per_threadgroup: u32,
    /// `maxCallStackDepth`.
    pub max_call_stack_depth: u32,
    /// The stage-input layouts' byte offset from the message start.
    pub stage_input_layouts_offset: u32,
    /// The number of stage-input layouts ([`RgfxVertexBufferLayout`]).
    pub stage_input_layout_count: u32,
    /// The stage-input attributes' byte offset from the message start.
    pub stage_input_attributes_offset: u32,
    /// The number of stage-input attributes ([`RgfxVertexAttribute`]).
    pub stage_input_attribute_count: u32,
    /// The stage-input descriptor's `indexType` (`MTLIndexType`).
    pub stage_input_index_type: u32,
    /// The stage-input descriptor's `indexBufferIndex`.
    pub stage_input_index_buffer_index: u32,
    /// The kernel's linked functions' byte offset from the message start.
    pub linked_offset: u32,
    /// The number of [`RgfxLinkedFunction`] the kernel links.
    pub linked_count: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_5c: u32,
}

/// `PIPELINE_REFLECTION`: a pipeline create's answer, with the pipeline state's
/// own queries (the host platform exposes no per-argument binding reflection, so
/// the argument tables Metal's `MTLRenderPipelineReflection` would carry are not
/// here — a platform gap recorded with the host decoder). On a refusal `status`
/// is [`RGFX_PIPELINE_REFUSED`] and the metrics are 0.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxPipelineReflection {
    /// [`RGFX_EVT_PIPELINE_REFLECTION`], with the create's `seqno`.
    pub header: RgfxMsgHeader,
    /// The pipeline the create named.
    pub obj_id: u64,
    /// `gpuResourceID` (`MTLResourceID._impl`).
    pub gpu_resource_id: u64,
    /// `uniqueIdentifier`, as the guest reads it (the state's identity).
    pub unique_identifier: u64,
    /// `maxTotalThreadsPerThreadgroup`.
    pub max_total_threads_per_threadgroup: u64,
    /// `staticThreadgroupMemoryLength` (compute; 0 for render).
    pub static_threadgroup_memory_length: u64,
    /// `threadExecutionWidth` (compute; 0 for render).
    pub thread_execution_width: u32,
    /// `imageblockSampleLength` (render; 0 for compute).
    pub imageblock_sample_length: u32,
    /// `resourceIndex` (the state's argument-buffer index).
    pub resource_index: u32,
    /// An `RGFX_PIPELINE_*` status.
    pub status: u32,
    /// An `RGFX_PIPELINE_KIND_*`.
    pub kind: u32,
    /// The `RGFX_PIPELINE_REFLECT_*` flags.
    pub flags: u32,
}

// The stage-3 messages' layout.
const _: () = {
    use core::mem::size_of;
    assert!(size_of::<RgfxObjectDestroy>() == 24);
    assert!(size_of::<RgfxObjectCreate>() == 24);
    assert!(size_of::<RgfxSharedEventCreate>() == 32);
    assert!(size_of::<RgfxSharedEventSetValue>() == 32);
    assert!(size_of::<RgfxQueueCreate>() == 56);
    assert!(RGFX_QUEUE_CREATE_FLAGS_OFFSET as usize == size_of::<RgfxQueueCreate>());
    assert!(size_of::<RgfxQueueCreateFlags>() == 8);
    assert!(size_of::<RgfxCommandBufferPart>() == 56);
    assert!(core::mem::offset_of!(RgfxCommandBufferPart, region_offset) == 40);
    assert!(size_of::<RgfxHeapCreate>() == 48);
    assert!(size_of::<RgfxHeapOverHostMemory>() == 56);
    assert!(size_of::<RgfxBufferCreate>() == 48);
    assert!(size_of::<RgfxBufferOverHostMemory>() == 56);
    assert!(size_of::<RgfxBufferAtOffset>() == 56);
    assert!(size_of::<RgfxBufferManagedAtOffset>() == 64);
    assert!(size_of::<RgfxCbComplete>() == 48);
    assert!(size_of::<RgfxCbScheduled>() == 24);
    assert!(size_of::<RgfxCbSubmitted>() == 24);
    assert!(size_of::<RgfxEventSignalled>() == 32);
    assert!(size_of::<RgfxTextureDescriptor>() == 48);
    assert!(size_of::<RgfxTextureCreate>() == 104);
    assert!(size_of::<RgfxTextureFromBuffer>() == 104);
    assert!(size_of::<RgfxLinearTextureFromBuffer>() == 112);
    assert!(size_of::<RgfxTextureFromIOSurface>() == 120);
    assert!(RGFX_SURFACE_FACTS_OFFSET as usize == size_of::<RgfxTextureFromIOSurface>());
    assert!(size_of::<RgfxSurfaceFacts>() == 40);
    assert!(core::mem::offset_of!(RgfxSurfaceFacts, chroma_offset) == 0x10);
    assert!(core::mem::offset_of!(RgfxSurfaceFacts, ycbcr_matrix) == 0x20);
    assert!(size_of::<RgfxSurfaceWriters>() == 64);
    assert!(size_of::<RgfxSamplerCreate>() == 72);
    assert!(RGFX_SAMPLER_CREATE_EXT_OFFSET as usize == size_of::<RgfxSamplerCreate>());
    assert!(size_of::<RgfxSamplerCreateExt>() == 8);
    assert!(size_of::<RgfxStencilDescriptor>() == 24);
    assert!(size_of::<RgfxDepthStencilCreate>() == 80);
    assert!(size_of::<RgfxLibraryCreate>() == 72);
    assert!(size_of::<RgfxLibraryFromRegion>() == 112);
    assert!(size_of::<RgfxLibraryLookup>() == 64);
    assert!(size_of::<RgfxBufferDidModifyRange>() == 40);
    assert!(size_of::<RgfxManagedTextureCreate>() == 112);
    assert!(size_of::<RgfxTextureReplaceRegion>() == 56);
    assert!(size_of::<RgfxTextureView>() == 72);
    assert!(size_of::<RgfxHeapTextureSizeAndAlign>() == 72);
    assert!(size_of::<RgfxSizeAndAlign>() == 40);
    assert!(size_of::<RgfxSamplerState>() == 32);
    assert!(size_of::<RgfxIcbCreate>() == 88);
    assert!(size_of::<RgfxIcbState>() == 32);
    assert!(size_of::<RgfxIcbWrite>() == 24);
    assert!(RGFX_LIBRARY_BYTES_OFFSET as usize == size_of::<RgfxLibraryCreate>());
    assert!(size_of::<RgfxFormatAnswers>() == 64);
    assert!(size_of::<RgfxDeviceInfo>() == 1432 + 64 * RGFX_DEVICE_INFO_FORMATS);
    assert!(core::mem::offset_of!(RgfxDeviceInfo, formats) == 1432);
    assert!(core::mem::offset_of!(RgfxDeviceInfo, family_answers) == 976);
    assert!(core::mem::offset_of!(RgfxDeviceInfo, name) == 1112);
    assert!(core::mem::offset_of!(RgfxDeviceInfo, features_lo) == 0x451);
    assert!(core::mem::offset_of!(RgfxDeviceInfo, sampler_reduction_support) == 0x452);
    // The device-info blob sits past the completion ring and before the allocations.
    assert!(
        RGFX_DEVICE_INFO_OFFSET
            == RGFX_CONTROL_COMPLETE_OFFSET
                + RGFX_RING_HEADER_SIZE as u64
                + RGFX_CONTROL_RING_DATA_SIZE as u64
    );
    assert!(
        RGFX_DEVICE_INFO_OFFSET + size_of::<RgfxDeviceInfo>() as u64 <= RGFX_ALLOCATIONS_OFFSET
    );
    // The video caps follow it, still in the completion ring's span.
    assert!(size_of::<RgfxVideoCodecCaps>() == 48);
    assert!(size_of::<RgfxVideoCaps>() == 8 + 48 * RGFX_VIDEO_CAPS_CODECS + 8);
    assert!(core::mem::offset_of!(RgfxVideoCaps, reserved_30c) == 0x30c);
    assert!(RGFX_VIDEO_CAPS_OFFSET == RGFX_DEVICE_INFO_OFFSET + size_of::<RgfxDeviceInfo>() as u64);
    assert!(RGFX_VIDEO_CAPS_OFFSET.is_multiple_of(8));
    assert!(
        RGFX_VIDEO_CAPS_OFFSET + size_of::<RgfxVideoCaps>() as u64
            <= RGFX_CONTROL_COMPLETE_OFFSET + RGFX_CONTROL_RING_SPAN
    );
    assert!(size_of::<RgfxVideoSessionCreate>() == 32);
    // Pipeline creates and reflection.
    assert!(size_of::<RgfxFunctionConstant>() == 40);
    assert!(size_of::<RgfxColorAttachmentDesc>() == 40);
    assert!(size_of::<RgfxVertexBufferLayout>() == 16);
    assert!(size_of::<RgfxVertexAttribute>() == 16);
    assert!(size_of::<RgfxLinkedFunction>() == 48);
    assert!(size_of::<RgfxRenderPipelineCreate>() == 464);
    assert!(core::mem::offset_of!(RgfxRenderPipelineCreate, colors) == 128);
    assert!(size_of::<RgfxComputePipelineCreate>() == 96);
    assert!(size_of::<RgfxPipelineReflection>() == 80);
    // Each exact size above is a multiple of 8, as a ring message must be.
    // The stage-3 opcodes do not collide with the stage-2 ones or each other.
    assert!(RGFX_CMD_QUEUE_CREATE > RGFX_EVT_VBLANK);
    assert!(RGFX_CMD_COMMAND_BUFFER > RGFX_CMD_FENCE_CREATE);
    assert!(RGFX_EVT_CB_COMPLETE > RGFX_CMD_COMMAND_BUFFER_IN_REGION);
    assert!(RGFX_CMD_COMMAND_BUFFER_IN_REGION > RGFX_CMD_COMMAND_BUFFER);
    assert!(RGFX_CMD_COMMAND_BUFFER_PART > RGFX_CMD_COMMAND_BUFFER_IN_REGION);
    assert!(RGFX_EVT_CB_COMPLETE > RGFX_CMD_COMMAND_BUFFER_PART);
    assert!(RGFX_CMD_BUFFER_OVER_PAGES > RGFX_CMD_VIDEO_SESSION_CREATE);
    assert!(RGFX_CMD_BUFFER_OVER_PAGES < RGFX_CMD_CONTEXT_CLOSE);
};

#[cfg(test)]
mod tests {
    use super::*;
    use zerocopy::{FromBytes, IntoBytes};

    /// A message written to bytes and read back is the same, and its header's
    /// `size` matches the struct, as the ring writer fills it in.
    #[test]
    fn stage3_messages_round_trip() {
        let destroy = RgfxObjectDestroy {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_OBJECT_DESTROY,
                size: size_of::<RgfxObjectDestroy>() as u32,
                seqno: 7,
            },
            obj_id: 0xdead_beef,
        };
        let bytes = destroy.as_bytes();
        assert_eq!(bytes.len(), 24);
        assert_eq!(RgfxObjectDestroy::read_from_bytes(bytes).unwrap(), destroy);

        let queue = RgfxQueueCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_QUEUE_CREATE,
                size: size_of::<RgfxQueueCreate>() as u32,
                seqno: 1,
            },
            obj_id: 1,
            submit_offset: RGFX_ALLOCATIONS_OFFSET,
            complete_offset: RGFX_ALLOCATIONS_OFFSET + 0x1_0000,
            submit_data_size: 0x1_0000,
            complete_data_size: 0x1_0000,
            priority: 2,
            vector: 1,
        };
        assert!(queue.submit_data_size.is_power_of_two());
        assert_eq!(
            RgfxQueueCreate::read_from_bytes(queue.as_bytes()).unwrap(),
            queue
        );

        let done = RgfxCbComplete {
            header: RgfxMsgHeader {
                r#type: RGFX_EVT_CB_COMPLETE,
                size: size_of::<RgfxCbComplete>() as u32,
                seqno: 42,
            },
            cb_obj_id: 99,
            gpu_start_ns: 1_000,
            gpu_end_ns: 2_000,
            status: RGFX_CB_OK,
            error_code: 0,
        };
        assert_eq!(done.header.size as usize, size_of::<RgfxCbComplete>());
        assert_eq!(
            RgfxCbComplete::read_from_bytes(done.as_bytes()).unwrap(),
            done
        );

        let texture = RgfxTextureCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_TEXTURE_CREATE,
                size: size_of::<RgfxTextureCreate>() as u32,
                seqno: 3,
            },
            obj_id: 5,
            gpu_address: 0x1_0000,
            resource_id: 11,
            heap_obj_id: 0,
            heap_offset: 0,
            desc: RgfxTextureDescriptor {
                width: 1920,
                height: 1080,
                depth: 1,
                mipmap_level_count: 1,
                array_length: 1,
                sample_count: 1,
                ..RgfxTextureDescriptor::default()
            },
        };
        assert_eq!(texture.header.size as usize, size_of::<RgfxTextureCreate>());
        assert_eq!(
            RgfxTextureCreate::read_from_bytes(texture.as_bytes()).unwrap(),
            texture
        );

        let depth = RgfxDepthStencilCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_DEPTH_STENCIL_CREATE,
                size: size_of::<RgfxDepthStencilCreate>() as u32,
                seqno: 4,
            },
            obj_id: 8,
            depth_compare: 1,
            depth_write_enabled: 1,
            front: RgfxStencilDescriptor {
                read_mask: 0xff,
                write_mask: 0xff,
                ..RgfxStencilDescriptor::default()
            },
            back: RgfxStencilDescriptor::default(),
        };
        assert_eq!(
            RgfxDepthStencilCreate::read_from_bytes(depth.as_bytes()).unwrap(),
            depth
        );

        // A library message's bytes follow its header; the header's size covers
        // both, rounded up to a multiple of 8.
        let lib = RgfxLibraryCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_LIBRARY_CREATE,
                size: RGFX_LIBRARY_BYTES_OFFSET + 8,
                seqno: 5,
            },
            obj_id: 12,
            byte_length: 6,
            ..RgfxLibraryCreate::default()
        };
        assert_eq!(
            RgfxLibraryCreate::read_from_bytes(lib.as_bytes()).unwrap(),
            lib
        );
        assert_eq!(
            RGFX_LIBRARY_BYTES_OFFSET as usize,
            size_of::<RgfxLibraryCreate>()
        );
    }

    /// The device-info caps blob writes to bytes and reads back unchanged; its
    /// name sits inline up to its cap.
    #[test]
    fn device_info_round_trips() {
        let mut info = RgfxDeviceInfo {
            version: RGFX_DEVICE_INFO_VERSION,
            registry_id: 0x1234,
            gpu_core_count: 10,
            max_threads_per_threadgroup: [1024, 1024, 1024],
            max_vertex_amplification_count: 2,
            family_count: 2,
            sample_counts: [1, 1, 1, 0],
            ..RgfxDeviceInfo::default()
        };
        info.family_values[..2].copy_from_slice(&[1009, -1009]);
        info.family_answers[..2].copy_from_slice(&[1, 0]);
        // Longer than v1's 32-byte name, which cut it short.
        let name = b"NVIDIA GeForce RTX 5080 Laptop GPU";
        info.name[..name.len()].copy_from_slice(name);
        info.name_len = name.len() as u32;
        info.format_count = 1;
        info.formats[0].format = 80;
        info.formats[0].sparse_tiles[0] = [64, 64, 1];
        assert_eq!(info.as_bytes().len(), 1432 + 64 * RGFX_DEVICE_INFO_FORMATS);
        assert_eq!(
            RgfxDeviceInfo::read_from_bytes(info.as_bytes()).unwrap(),
            info
        );
        assert_eq!(&info.name[..info.name_len as usize], name);
    }

    /// The pipeline create heads and the reflection reply round-trip; a render
    /// create's fixed head reads back with its colour attachments and its tail
    /// names lie where the head's offsets say.
    #[test]
    fn pipeline_messages_round_trip() {
        // A render create: one colour attachment, a vertex and a fragment name in
        // the tail right after the fixed head.
        let head = size_of::<RgfxRenderPipelineCreate>() as u32;
        let (vname, fname) = (b"vmain".as_slice(), b"frag_main".as_slice());
        let mut colors = [RgfxColorAttachmentDesc::default(); 8];
        colors[0] = RgfxColorAttachmentDesc {
            pixel_format: 80, // BGRA8Unorm
            blending_enabled: 1,
            source_rgb_blend_factor: 1,      // One
            destination_rgb_blend_factor: 5, // OneMinusSourceAlpha
            rgb_blend_operation: 0,          // Add
            source_alpha_blend_factor: 1,
            destination_alpha_blend_factor: 5,
            alpha_blend_operation: 0,
            write_mask: 0xf,
            reserved_24: 0,
        };
        let mut render = RgfxRenderPipelineCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_RENDER_PIPELINE_CREATE,
                size: 0,
                seqno: 9,
            },
            obj_id: 0x30,
            vertex_library_obj_id: 0x10,
            fragment_library_obj_id: 0x10,
            vertex_name_offset: head,
            vertex_name_len: vname.len() as u32,
            fragment_name_offset: head + vname.len() as u32,
            fragment_name_len: fname.len() as u32,
            raster_sample_count: 1,
            color_attachment_count: 1,
            depth_pixel_format: 252, // Depth32Float
            flags: RGFX_RENDER_PIPELINE_RASTERIZATION_ENABLED,
            colors,
            ..RgfxRenderPipelineCreate::default()
        };
        let total = ((head as usize + vname.len() + fname.len()) + 7) & !7;
        render.header.size = total as u32;
        let mut msg = render.as_bytes().to_vec();
        msg.extend_from_slice(vname);
        msg.extend_from_slice(fname);
        msg.resize(total, 0);

        let (back, _) = RgfxRenderPipelineCreate::read_from_prefix(&msg).unwrap();
        assert_eq!(back, render);
        assert_eq!(back.colors[0].pixel_format, 80);
        assert_eq!(back.colors[0].destination_rgb_blend_factor, 5);
        let vs = back.vertex_name_offset as usize;
        assert_eq!(&msg[vs..vs + back.vertex_name_len as usize], vname);
        let fs = back.fragment_name_offset as usize;
        assert_eq!(&msg[fs..fs + back.fragment_name_len as usize], fname);

        // A compute create head.
        let compute = RgfxComputePipelineCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_COMPUTE_PIPELINE_CREATE,
                size: size_of::<RgfxComputePipelineCreate>() as u32,
                seqno: 10,
            },
            obj_id: 0x31,
            function_library_obj_id: 0x10,
            name_offset: size_of::<RgfxComputePipelineCreate>() as u32,
            name_len: 4,
            flags: RGFX_COMPUTE_PIPELINE_SUPPORT_ICB,
            max_total_threads_per_threadgroup: 256,
            ..RgfxComputePipelineCreate::default()
        };
        assert_eq!(
            RgfxComputePipelineCreate::read_from_bytes(compute.as_bytes()).unwrap(),
            compute
        );

        // A function constant and the reflection reply.
        let k = RgfxFunctionConstant {
            index: 3,
            data_type: 84, // Long4
            value: core::array::from_fn(|i| i as u8 + 1),
        };
        assert_eq!(
            RgfxFunctionConstant::read_from_bytes(k.as_bytes()).unwrap(),
            k
        );
        let reflection = RgfxPipelineReflection {
            header: RgfxMsgHeader {
                r#type: RGFX_EVT_PIPELINE_REFLECTION,
                size: size_of::<RgfxPipelineReflection>() as u32,
                seqno: 9,
            },
            obj_id: 0x30,
            gpu_resource_id: 0xabc,
            unique_identifier: 0xdef,
            max_total_threads_per_threadgroup: 1024,
            imageblock_sample_length: 4,
            resource_index: 7,
            status: RGFX_PIPELINE_OK,
            kind: RGFX_PIPELINE_KIND_RENDER,
            flags: RGFX_PIPELINE_REFLECT_SUPPORT_ICB,
            ..RgfxPipelineReflection::default()
        };
        assert_eq!(
            RgfxPipelineReflection::read_from_bytes(reflection.as_bytes()).unwrap(),
            reflection
        );
    }
}
