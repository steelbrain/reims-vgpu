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
//! reads [`RGFX_CONSUMER_ASLEEP`] it writes the ring's id to [`RGFX_REG_DOORBELL`], and so it does
//! on reading [`RGFX_CONSUMER_WALKING`] for a command buffer that asks to be answered at the
//! doorbell, whose id it writes with [`RGFX_DOORBELL_ASKS`] where it took
//! [`RGFX_FEATURE_DOORBELL_ASKS`]. The consumer (the host) sets `consumer_idle` before it sleeps, issues a full fence
//! and reads `producer` again, so no wake is lost, and clears it when it wakes. The producer of a completion ring (the host) publishes a
//! message, issues a full fence and reads `armed`; if it reads 1, it clears it, reads `consumer`,
//! and raises the ring's MSI-X vector, or, where the guest has taken every message published, sets
//! `armed` again and raises nothing: the 1 it cleared was an arm the guest made after its take.
//! The guest stores `consumer` as it drains the ring, then sets `armed`, issues a full fence and
//! reads `producer` again.

#[cfg(doc)]
use crate::ops::video::RgfxVideoOpDecodeH264;
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
/// The BAR of the doorbell pages: 32-bit MMIO, not prefetchable, [`RGFX_DOORBELLS_SIZE`] bytes,
/// [`RGFX_DOORBELL_PAGES`] pages of [`RGFX_DOORBELL_PAGE_SIZE`] bytes each, where the host offers
/// [`RGFX_FEATURE_DOORBELL_PAGES`]. A page bound to a context ([`RGFX_REG_DOORBELL_PAGE_BIND`]) is
/// that context's own `DOORBELL`, which the kext maps into the context's process: a 32-bit write
/// at the page's first byte of a ring's id, with [`RGFX_DOORBELL_ASKS`] as `DOORBELL` takes it,
/// wakes that ring where it is one of the context's command queues' rings. Every other write (an
/// unbound page, another offset, ring 0, a ring of no queue of the context's) is dropped, and
/// every read answers 0: the page holds no register a process could reset or reconfigure the
/// device with.
pub const RGFX_PCI_BAR_DOORBELLS: u32 = 4;
/// The bytes of one doorbell page: 16 KiB, the largest page a guest maps (an arm64 guest's), so
/// a process's mapping of its page reaches no other page.
pub const RGFX_DOORBELL_PAGE_SIZE: u32 = 0x4000;
/// How many doorbell pages BAR 4 holds: a busy desktop holds 74 contexts at once, and a context
/// past them, or one whose page the kext could not map, rings through the kext's doorbell call.
pub const RGFX_DOORBELL_PAGES: u32 = 256;
/// The size of BAR 4: [`RGFX_DOORBELL_PAGES`] pages of [`RGFX_DOORBELL_PAGE_SIZE`] bytes, 4 MiB.
pub const RGFX_DOORBELLS_SIZE: u32 = RGFX_DOORBELL_PAGES * RGFX_DOORBELL_PAGE_SIZE;

/// `MAGIC`, RO: [`RGFX_MAGIC`].
pub const RGFX_REG_MAGIC: u32 = 0x000;
/// `VERSION`, RO: [`RGFX_VERSION`].
pub const RGFX_REG_VERSION: u32 = 0x004;
/// `FEATURES_LO`, RO: feature bits 0 to 31, the ones the host offers:
/// [`RGFX_FEATURE_SURFACE_PAGES`], [`RGFX_FEATURE_SURFACE_CACHE`] and
/// [`RGFX_FEATURE_CB_SUBMITTED`], and from bit 3 the ones the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`] ([`RGFX_FEATURE_QUIET_SCHEDULED`] to
/// [`RGFX_FEATURE_QUEUE_VECTORS`], [`RGFX_FEATURE_IMPORT_ADDRESSES_LATER`],
/// [`RGFX_FEATURE_SURFACE_ORDER`],
/// [`RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS`], [`RGFX_FEATURE_UNSHARED_QUEUES`],
/// [`RGFX_FEATURE_MANAGED_AT_OFFSET`], [`RGFX_FEATURE_HEAP_OVER_PAGES`],
/// [`RGFX_FEATURE_SUBMIT_ROOM`], [`RGFX_FEATURE_ICB_SLOTS`], [`RGFX_FEATURE_COMMIT_ORDER`],
/// [`RGFX_FEATURE_SURFACE_DEFERRED`], [`RGFX_FEATURE_HOST_TIME`],
/// [`RGFX_FEATURE_MANAGED_TEXTURE_SLABS`], [`RGFX_FEATURE_MANAGED_BUFFER_SLABS`],
/// [`RGFX_FEATURE_SMALL_ID_CHUNKS`], [`RGFX_FEATURE_DISPLAY_COLOR`],
/// [`RGFX_FEATURE_MESH_PIPELINES`], [`RGFX_FEATURE_SHARED_EVENT_NAMES`],
/// [`RGFX_FEATURE_OPENCL_MODE`], [`RGFX_FEATURE_PRIVATE_RENDER_STATE`],
/// [`RGFX_FEATURE_SURFACE_CACHE_REFUSED`] and [`RGFX_FEATURE_SURFACE_DEFERRED_ASK`]). Bit 10 is
/// reserved.
pub const RGFX_REG_FEATURES_LO: u32 = 0x008;
/// `FEATURES_HI`, RO: feature bits 32 to 63, the ones the host offers, which the guest takes
/// through [`RGFX_REG_DRIVER_FEATURES_HI`]: [`RGFX_FEATURE_SURFACE_DEFERRED_CARRY`],
/// [`RGFX_FEATURE_SCHEDULED_ASK`], [`RGFX_FEATURE_PAGES_16K`], [`RGFX_FEATURE_DISPLAY_SCALE`],
/// [`RGFX_FEATURE_TEXTURE_DID_MODIFY_DATA`], [`RGFX_FEATURE_DISPLAY_POWER`],
/// [`RGFX_FEATURE_RATE_MAPS`], [`RGFX_FEATURE_FUNCTION_POINTERS`],
/// [`RGFX_FEATURE_DYNAMIC_LIBRARIES`], [`RGFX_FEATURE_PIPELINE_KEYS`],
/// [`RGFX_FEATURE_ICB_SLOTS_IN_PAGES`], [`RGFX_FEATURE_DOORBELL_ASKS`],
/// [`RGFX_FEATURE_GUEST_TIMES`], [`RGFX_FEATURE_READ_ONLY_PAGES`],
/// [`RGFX_FEATURE_DOORBELL_PAGES`] and [`RGFX_FEATURE_PIPELINE_ANSWERS`].
/// On a host that offers `PAGES_16K`, `FEATURES_LO` and `_HI` read the guest RAM features only
/// once the guest took it.
pub const RGFX_REG_FEATURES_HI: u32 = 0x00c;
/// `FEATURES_LO` bit 0, `SURFACE_PAGES`: the host maps an IOSurface's own guest pages for its GPU
/// to use in place (`SURFACE_PAGES`, `BUFFER_OVER_SURFACE_PAGES`); without it a surface's GPU
/// view is a copy of its pages in a block of the region.
pub const RGFX_FEATURE_SURFACE_PAGES: u32 = 1 << 0;
/// `FEATURES_LO` bit 1, `SURFACE_CACHE`, beside [`RGFX_FEATURE_SURFACE_PAGES`]: the host keeps
/// the GPU's copy of an IOSurface whose block the kext marks cached ([`RGFX_SURFACE_PAGES_CACHED`])
/// apart from its pages, the surface's device cache: in its GPU's own memory, or in the host's RAM
/// beside the pages where the GPU shares the CPU's memory. The cache is filled from the surface's
/// pages at the first command buffer that uses it after the CPU's generation of the surface moved
/// (`RGFX_OP_SURFACE_USE`), and the pages from the cache only at a [`RGFX_CMD_SURFACE_DOWNLOAD`]
/// and at the block's `BLOCK_RELEASE`; a write the host's own devices make into the pages (a video
/// device's frame) has the next use fill it again. Set on every host that offers `SURFACE_PAGES`,
/// whether or not its GPU shares the CPU's memory; the kext alone reads it, and nothing the guest's
/// processes see changes but the time copies take.
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
/// `SURFACE_PAGES` may carry [`RGFX_SURFACE_PAGES_PRIVATE`]. A host offers it where it offers
/// [`RGFX_FEATURE_SURFACE_PAGES`].
pub const RGFX_FEATURE_PAGES_BUFFERS: u32 = 1 << 7;
/// `FEATURES_LO` bit 8, `QUEUE_VECTORS`: the device has an MSI-X vector for each context slot
/// after vector 0, and a queue's completion ring raises its context's. No host offers it yet.
pub const RGFX_FEATURE_QUEUE_VECTORS: u32 = 1 << 8;
/// `FEATURES_LO` bit 9, `IMPORT_ADDRESSES_LATER`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`] beside [`RGFX_FEATURE_IMPORT_ADDRESSES`] and
/// [`RGFX_FEATURE_PAGES_BUFFERS`]: a pages block's `SURFACE_PAGES` ([`RGFX_SURFACE_PAGES_PRIVATE`])
/// is answered once the host has checked and kept its runs, before it makes the block's view and
/// gives the view its address, which it does after the answer; the answer's `gpu_address`
/// ([`RgfxPagesAnswer`]) is 0. The sender's messages after it that name the block are carried out
/// once the view has its address, so the wait for it is the host's, not the guest's call's. The
/// guest makes each `BUFFER_OVER_PAGES` over such a block at `gpu_address` 0, which the host
/// places at the view's address plus the buffer's `offset`, where it would lie had the answer
/// carried the address, and asks a buffer's address with [`RGFX_CMD_PAGES_BUFFER_ADDRESS`] where
/// it needs it. A host offers it where a view's address is that of an import of the block's pages
/// (metal2vulkan-rt's device), and else not, so a guest there sees `IMPORT_ADDRESSES` alone.
pub const RGFX_FEATURE_IMPORT_ADDRESSES_LATER: u32 = 1 << 9;
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
/// `FEATURES_LO` bit 15, `HEAP_OVER_PAGES`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`]: ring 0 carries [`RGFX_CMD_HEAP_OVER_PAGES`], a heap over the
/// one run of guest RAM a pages block of the process's holds ([`RGFX_FEATURE_PAGES_BUFFERS`]), so
/// the buffers placed in it lie in the process's own pages. Offered wherever `PAGES_BUFFERS` is.
pub const RGFX_FEATURE_HEAP_OVER_PAGES: u32 = 1 << 15;
/// `FEATURES_LO` bit 16, `SUBMIT_ROOM`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`]: a guest producer that finds a submission ring full, ring 0 or
/// a queue's, may store 1 in the ring's `producer_waiting`, fence, and look for room once more
/// before it waits. After each take that moves a submission ring's `consumer`, the host fences and
/// reads the flag; where it reads 1, it stores 0 and posts an [`RGFX_EVT_SUBMIT_ROOM`] on the
/// ring's completion ring: ring 0's for ring 0, the queue's for a queue's. Either the guest's look
/// sees the take's room, or the host's read sees the flag. Every host offers it, the device
/// carrying it out.
pub const RGFX_FEATURE_SUBMIT_ROOM: u32 = 1 << 16;
/// `FEATURES_LO` bit 17, `ICB_SLOTS`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`], and which means nothing without
/// [`RGFX_FEATURE_INDIRECT_COMMAND_BUFFERS`]: ring 0 carries [`RGFX_CMD_ICB_SLOTS`], the CPU's
/// encodes into indirect command buffers as slots in the process's region ([`crate::icb`],
/// "Slots"), which the host reads where the message names them. The guest still sends `ICB_WRITE`
/// for a buffer it keeps no slots for.
pub const RGFX_FEATURE_ICB_SLOTS: u32 = 1 << 17;
/// `FEATURES_LO` bit 18, `COMMIT_ORDER`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`]: the host decodes
/// [`RGFX_OP_COMMIT_ORDER`](crate::ops::RGFX_OP_COMMIT_ORDER), so a command buffer is committed to
/// the platform after the buffers of its process's other queues posted before it, as Metal orders a
/// tracked resource across queues by commit.
pub const RGFX_FEATURE_COMMIT_ORDER: u32 = 1 << 18;
/// `FEATURES_LO` bit 19, `SURFACE_DEFERRED`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`], and which means nothing without
/// [`RGFX_FEATURE_SURFACE_CACHE`]: a `SURFACE_PAGES` may carry [`RGFX_SURFACE_PAGES_DEFERRED`], a
/// cached block whose surface has no pages yet, its contents in the device cache alone, and a later
/// one [`RGFX_SURFACE_PAGES_ATTACH`], which names its pages. The host offers it where it offers
/// `SURFACE_CACHE`, and says in [`RGFX_REG_DEFERRED_BUDGET_LO`] how many bytes of its GPU's memory
/// such caches may hold.
pub const RGFX_FEATURE_SURFACE_DEFERRED: u32 = 1 << 19;
/// `FEATURES_LO` bit 20, `HOST_TIME`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`]: [`RGFX_REG_HOST_TIME_LO`] and `_HI` read the host's monotonic
/// clock, the one a `VBLANK`'s `host_time_ns` is on. A read between two of the guest's own clock's
/// bounds the two clocks' offset by its round trip, whatever an interrupt's delivery takes. Every
/// host offers it.
pub const RGFX_FEATURE_HOST_TIME: u32 = 1 << 20;
/// `FEATURES_LO` bit 21, `MANAGED_TEXTURE_SLABS`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`]: a [`RgfxManagedTextureCreate`]'s CPU copy may be any `Shared`
/// buffer over the guest's memory the sender holds, and the host reads and writes the copy where
/// that buffer's contents lie: over a block of the sender's (`BUFFER_OVER_HOST_MEMORY`), placed in
/// a heap over one or over a pages block's guest RAM (`BUFFER_AT_OFFSET`), or over a pages block's
/// view (`BUFFER_OVER_PAGES`). So the guest can lay the copies in its buffers' slabs, its own pages
/// among them, and not in the region. A host without it reads the copy in the region at the
/// buffer's span, which holds the copy only where the buffer lies over its own block. Every host
/// offers it.
pub const RGFX_FEATURE_MANAGED_TEXTURE_SLABS: u32 = 1 << 21;
/// `FEATURES_LO` bit 22, `MANAGED_BUFFER_SLABS`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`]: ring 0 carries [`RGFX_CMD_MANAGED_BUFFER_CREATE`], a `Managed`
/// buffer whose CPU copy is any `Shared` buffer over the guest's memory the sender holds, read
/// and written where that buffer's contents lie, as [`RGFX_FEATURE_MANAGED_TEXTURE_SLABS`] reads a
/// texture's. So the guest lays its `Managed` buffers' CPU copies in its Shared buffers' slabs,
/// its own pages among them, on every host, and not in the region. Every host offers it.
pub const RGFX_FEATURE_MANAGED_BUFFER_SLABS: u32 = 1 << 22;
/// `FEATURES_LO` bit 23, `SMALL_ID_CHUNKS`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`]: ring 0 carries [`RGFX_CMD_RESOURCE_ID_SMALL_CHUNK`], a grant
/// of a chunk of [`RGFX_RESOURCE_ID_SMALL_CHUNK`] texture resource ids, so the kext hands each
/// process its ids 16 at a time from the same pool, and a guest of many processes holds as many
/// chunks as it has contexts. Every host offers it.
pub const RGFX_FEATURE_SMALL_ID_CHUNKS: u32 = 1 << 23;
/// `FEATURES_LO` bit 24, `DISPLAY_COLOR`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`]: ring 0 carries [`RGFX_CMD_DISPLAY_COLOR`], the colour stages a
/// display pipe transaction sets (its gamma tables and its colour matrix), and a
/// `PRESENT_SURFACE` may carry [`RGFX_PRESENT_KEEP`], the present of a transaction that shows no
/// new surface. Every host offers it; one with no window answers its presents at once.
pub const RGFX_FEATURE_DISPLAY_COLOR: u32 = 1 << 24;
/// `FEATURES_LO` bit 25, `MESH_PIPELINES`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`]: ring 0 carries [`RGFX_CMD_MESH_RENDER_PIPELINE_CREATE`],
/// answered by a [`RgfxPipelineReflection`] of [`RGFX_PIPELINE_KIND_MESH`], and command buffers
/// carry the object and mesh stages' bindings and the mesh draws
/// ([`RGFX_OP_RENDER_SET_OBJECT_BUFFERS`](crate::ops::render::RGFX_OP_RENDER_SET_OBJECT_BUFFERS)
/// to [`RGFX_OP_RENDER_DRAW_MESH_THREADGROUPS_INDIRECT`](crate::ops::render::RGFX_OP_RENDER_DRAW_MESH_THREADGROUPS_INDIRECT)).
/// A host without it refuses the create as one it does not know, and faults a buffer at those
/// ops. Every host offers it; where the host's platform makes no mesh pipeline, the create is
/// refused as one the platform refused.
pub const RGFX_FEATURE_MESH_PIPELINES: u32 = 1 << 25;
/// `FEATURES_LO` bit 26, `SHARED_EVENT_NAMES`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`]: ring 0 carries [`RGFX_CMD_SHARED_EVENT_NAME`] and
/// [`RGFX_CMD_SHARED_EVENT_IMPORT`], which the accelerator kext posts for a process's shared event
/// another process makes an event of its own from, and [`RGFX_CMD_SHARED_EVENT_UNNAME`]: both
/// name one host event, which every holder's command buffers signal and wait on, and the values
/// one holder's buffers and CPU give it are posted to every other holder
/// ([`RgfxSharedEventName::listen_queue_obj_id`]).
pub const RGFX_FEATURE_SHARED_EVENT_NAMES: u32 = 1 << 26;
/// `FEATURES_LO` bit 27, `OPENCL_MODE`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`]: a [`RgfxComputePipelineCreate`] may carry
/// [`RGFX_COMPUTE_PIPELINE_OPENCL_MODE`], and the host makes that pipeline of OpenCL mode (Metal's
/// `openCLModeEnabled`): its function is a kernel of the AIR Metal's compiler writes for OpenCL C
/// (`__entry_<name>`), linked as Metal links it, with three buffers of the dispatcher's ahead of
/// the kernel's own. A host offers it only where its platform makes such pipelines. A host
/// without it reads the flag as no flag, so the guest sends the flag only where it took the bit.
pub const RGFX_FEATURE_OPENCL_MODE: u32 = 1 << 27;
/// `FEATURES_LO` bit 28, `PRIVATE_RENDER_STATE`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`]: a [`RgfxRenderPipelineCreate`] may carry
/// [`RGFX_RENDER_PIPELINE_PRIVATE_STATE`] and, after its head, the
/// [`RgfxRenderPipelinePrivateState`] of the guest's descriptor: the private state (Apple SPI)
/// Apple's OpenGL renderer sets, which the host's platform makes the pipeline with. A host offers
/// it only where its platform takes that state. A host without it reads the flag as no flag and
/// the record as tail no offset names, so the guest sends them only where it took the bit.
pub const RGFX_FEATURE_PRIVATE_RENDER_STATE: u32 = 1 << 28;
/// `FEATURES_LO` bit 29, `SURFACE_CACHE_REFUSED`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`], beside [`RGFX_FEATURE_SURFACE_PAGES`]: the host's GPU shares
/// the CPU's memory, and keeps a device cache only of the IOSurfaces whose pages its textures
/// cannot use in place, which it answers surface by surface. A kext that took it asks a new block
/// in place with [`RGFX_SURFACE_PAGES_ASK_CACHE`] and the surface's layout
/// ([`RgfxSurfacePagesLayout`]); the host answers [`RGFX_STATUS_CACHE`] where its textures over
/// those pages, for any access a texture over the surface may take, would not lie in place, and
/// the kext then names the block cached ([`RGFX_SURFACE_PAGES_CACHED`]), the host keeping its
/// device cache as [`RGFX_FEATURE_SURFACE_CACHE`] says. Every other block stays in place as on a
/// host without it. No host offers it now: one whose GPU shares the CPU's memory offers
/// `SURFACE_CACHE` and [`RGFX_FEATURE_SURFACE_DEFERRED`], as every other does.
pub const RGFX_FEATURE_SURFACE_CACHE_REFUSED: u32 = 1 << 29;
/// `FEATURES_LO` bit 30, `SURFACE_DEFERRED_ASK`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_LO`], beside [`RGFX_FEATURE_SURFACE_CACHE_REFUSED`]: a kext that took
/// both may defer a new block in place of a fresh surface ([`RGFX_SURFACE_PAGES_DEFERRED`]), as
/// [`RGFX_FEATURE_SURFACE_DEFERRED`] lets it on a host that keeps caches, asking with the
/// surface's layout ([`RGFX_SURFACE_PAGES_ASK_CACHE`]). The host defers it, a device cache with no
/// pages, only where the block would take a device cache whatever pages the surface comes to
/// have: textures over them would lie in place over no import of them the host makes. It refuses
/// every other, whose pages the kext then names asking, as without the feature. Such a surface's
/// pages are wired, and copied into its cache, only once a CPU lock needs them, as on a GPU of its
/// own memory, not at its first texture. The host says in [`RGFX_REG_DEFERRED_BUDGET_LO`] how many
/// bytes of its GPU's memory such caches may hold. A host offers it where it offers
/// `SURFACE_CACHE_REFUSED`.
pub const RGFX_FEATURE_SURFACE_DEFERRED_ASK: u32 = 1 << 30;
/// `FEATURES_HI` bit 0, feature bit 32, `SURFACE_DEFERRED_CARRY`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_HI`], and which means nothing without
/// [`RGFX_FEATURE_SURFACE_DEFERRED`] or [`RGFX_FEATURE_SURFACE_DEFERRED_ASK`]: a deferral
/// ([`RGFX_SURFACE_PAGES_DEFERRED`]) of a surface no CPU lock has touched, a few of whose pages
/// are resident all the same (written by a process without a lock), may carry those pages, its
/// runs the surface's bytes in order, each run of pages that are not resident
/// [`RGFX_PAGE_RUN_ZEROS`]. The host makes the cache of zeros as for any deferral, and fills it
/// from the carried pages, the zeros runs left as they are, as a new cache's first fill: before
/// what reads the cache next, or never where a decode writes the surface's planes whole first.
/// The kext keeps the carried pages wired until their block's pages are named
/// ([`RGFX_SURFACE_PAGES_ATTACH`]) or its release is answered. A host offers it where it offers
/// either deferral; one that does not never gets a deferral with runs.
pub const RGFX_FEATURE_SURFACE_DEFERRED_CARRY: u64 = 0x1_0000_0000;
/// `FEATURES_HI` bit 1, feature bit 33, `SCHEDULED_ASK`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_HI`]: a queue's ring carries [`RGFX_CMD_SCHEDULED_ASK`], a thread
/// of the process waiting for a command buffer of the queue to be scheduled, and the host posts
/// the buffer's [`RGFX_EVT_CB_SCHEDULED`] as it takes the ask where the buffer may be scheduled
/// then, as it posts that of a buffer that asks for its `CB_SUBMITTED` at its take. Every host
/// offers it.
pub const RGFX_FEATURE_SCHEDULED_ASK: u64 = 0x2_0000_0000;
/// `FEATURES_HI` bit 2, feature bit 34, `PAGES_16K`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_HI`]: every run of guest RAM the guest names (a `SURFACE_PAGES`'
/// runs, a pages block's) is whole 16 KiB pages from a 16 KiB page's first byte, and the length
/// of a `SURFACE_PAGES` is whole such pages too, as an arm64 guest's pages are. A host whose page
/// is 16 KiB offers it where it maps the guest's RAM, and offers the features over guest RAM there
/// ([`RGFX_FEATURE_SURFACE_PAGES`], [`RGFX_FEATURE_PAGES_BUFFERS`],
/// [`RGFX_FEATURE_HEAP_OVER_PAGES`] and those that go with them) only to a guest that took it:
/// `FEATURES_LO` and `_HI` read them from the write that takes `PAGES_16K` on, that write and the
/// ones after it may take them, and a reset, which clears the take, hides them again. A guest that
/// never takes it, as an x86-64 guest's kext, reads every other bit and register as from a host
/// that maps no guest RAM. A host whose page is 4 KiB never offers it: any guest's runs are whole
/// pages of its own there.
pub const RGFX_FEATURE_PAGES_16K: u64 = 0x4_0000_0000;
/// `FEATURES_HI` bit 3, feature bit 35, `DISPLAY_SCALE`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_HI`]: each [`RgfxDisplayEntry`] of a `DISPLAYS` answer is followed
/// by an [`RgfxDisplayScale`], then by its modes: the scale of the host screen that shows the
/// display, from which the guest states the display's pixels per inch. Every host offers it.
pub const RGFX_FEATURE_DISPLAY_SCALE: u64 = 0x8_0000_0000;

/// `FEATURES_HI` bit 4, feature bit 36, `TEXTURE_DID_MODIFY_DATA`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_HI`]: ring 0 carries [`RGFX_CMD_TEXTURE_DID_MODIFY_DATA`], a
/// texture's `didModifyData` (SPI), which a mipmapped texture over a buffer's bytes needs to take
/// them again. Every host offers it.
pub const RGFX_FEATURE_TEXTURE_DID_MODIFY_DATA: u64 = 0x10_0000_0000;

/// `FEATURES_HI` bit 5, feature bit 37, `DISPLAY_POWER`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_HI`]: ring 0 carries [`RGFX_CMD_SET_DISPLAY_POWER`], which turns
/// the display's panel off while the guest's display sleeps and on again at its wake, so the host
/// shows black while it sleeps. Every host offers it.
pub const RGFX_FEATURE_DISPLAY_POWER: u64 = 0x20_0000_0000;

/// `FEATURES_HI` bit 6, feature bit 38, `RATE_MAPS`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_HI`]: ring 0 carries [`RGFX_CMD_RATE_MAP_CREATE`],
/// [`RGFX_CMD_RATE_MAP_RESET`] and [`RGFX_CMD_RATE_MAP_QUERY`], the device's rasterization rate
/// maps, and a render pass may name one
/// ([`RGFX_PASS_FLAG_RATE_MAP`](crate::ops::RGFX_PASS_FLAG_RATE_MAP)). Every host offers it; the
/// layer counts the device makes maps of are the device info's
/// (`largest_rasterization_rate_map_layer_count`).
pub const RGFX_FEATURE_RATE_MAPS: u64 = 0x40_0000_0000;
/// `FEATURES_HI` bit 7, feature bit 39, `FUNCTION_POINTERS`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_HI`]: ring 0 carries [`RGFX_CMD_FUNCTION_HANDLE_CREATE`],
/// [`RGFX_CMD_FUNCTION_TABLE_CREATE`] and [`RGFX_CMD_FUNCTION_TABLE_SET`], a compute pipeline
/// state's function handles and visible function tables, and a compute encoder binds a table
/// ([`RGFX_OP_COMPUTE_SET_VISIBLE_FUNCTION_TABLE`](crate::ops::compute::RGFX_OP_COMPUTE_SET_VISIBLE_FUNCTION_TABLE)
/// and
/// [`RGFX_OP_COMPUTE_SET_VISIBLE_FUNCTION_TABLES`](crate::ops::compute::RGFX_OP_COMPUTE_SET_VISIBLE_FUNCTION_TABLES)).
/// Every host offers it; whether the device's kernels call through tables is the device info's
/// (`supports_function_pointers`).
pub const RGFX_FEATURE_FUNCTION_POINTERS: u64 = 0x80_0000_0000;
/// `FEATURES_HI` bit 8, feature bit 40, `DYNAMIC_LIBRARIES`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_HI`]: ring 0 carries [`RGFX_CMD_DYNAMIC_LIBRARY_CREATE`], a
/// dynamic library of a library the guest made, and a [`RgfxComputePipelineCreate`] names the
/// dynamic libraries its kernel links, preloaded or loaded under an install name
/// ([`RgfxPipelineLibrary`]). Every host offers it; whether the device's kernels link dynamic
/// libraries is the device info's (`supports_dynamic_libraries`).
pub const RGFX_FEATURE_DYNAMIC_LIBRARIES: u64 = 0x100_0000_0000;
/// `FEATURES_HI` bit 9, feature bit 41, `PIPELINE_KEYS`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_HI`]: a render or compute pipeline create whose flags carry
/// [`RGFX_RENDER_PIPELINE_ANSWER_KEY`] or [`RGFX_COMPUTE_PIPELINE_ANSWER_KEY`] is answered, where
/// the device made it and can key it, with its pipeline key after its reflection
/// ([`RGFX_PIPELINE_REFLECT_KEYED`], [`RGFX_PIPELINE_REFLECTION_KEY_OFFSET`]). Every host offers
/// it.
pub const RGFX_FEATURE_PIPELINE_KEYS: u64 = 0x200_0000_0000;
/// `FEATURES_HI` bit 10, feature bit 42, `ICB_SLOTS_IN_PAGES`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_HI`], and which means nothing without [`RGFX_FEATURE_ICB_SLOTS`] and
/// [`RGFX_FEATURE_PAGES_BUFFERS`]: ring 0 carries [`RGFX_CMD_ICB_SLOTS_IN_PAGES`], slots as
/// [`RGFX_CMD_ICB_SLOTS`]'s that lie in the pages of a pages block of the sender's, which the host
/// reads in place, in the guest's RAM: the process's own memory, which its kernel maps cached,
/// where it maps the region's blocks uncached on an AMD host's guest ([`crate::icb`], "Slots").
/// A host offers it where it offers [`RGFX_FEATURE_PAGES_BUFFERS`].
pub const RGFX_FEATURE_ICB_SLOTS_IN_PAGES: u64 = 0x400_0000_0000;
/// `FEATURES_HI` bit 11, feature bit 43, `DOORBELL_ASKS`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_HI`]: the guest rings a command queue's doorbell for a
/// [`RGFX_CMD_COMMAND_BUFFER`] whose last op asks for its [`RGFX_EVT_CB_SUBMITTED`]
/// ([`RGFX_OP_SUBMITTED`](crate::ops::RGFX_OP_SUBMITTED)) with the ring's id and
/// [`RGFX_DOORBELL_ASKS`], and for any other message with the ring's id alone. The host may then
/// take a ring's plain doorbell where no thread of its runs the write (QEMU: a KVM `ioeventfd`
/// that wakes the queue's thread), and answers a buffer at its doorbell only for the write that
/// holds [`RGFX_DOORBELL_ASKS`], as it did for any write before. Every host offers it.
pub const RGFX_FEATURE_DOORBELL_ASKS: u64 = 0x800_0000_0000;
/// `FEATURES_HI` bit 12, feature bit 44, `GUEST_TIMES`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_HI`]: a diagnostic of the time before the host takes a command
/// buffer. Each [`RGFX_CMD_COMMAND_BUFFER`]'s op stream starts with one
/// [`RGFX_OP_GUEST_TIMES`](crate::ops::RGFX_OP_GUEST_TIMES), the guest's clock as the buffer was
/// made, committed and published on its ring, and the kext keeps ring 0's
/// [`RgfxRingHeader::guest_clock_offset_ns`] and [`RgfxRingHeader::guest_clock_bound_ns`], the
/// guest's clock less the host's, from its reading of [`RGFX_REG_HOST_TIME_LO`] at each `VBLANK`.
/// The host splits the time from a buffer's making to its take into the guest's encode, the
/// guest's wait to post it, the wait behind the queue's buffer before it and the transport, for
/// its trace and its log; it changes nothing else the device does. Every host offers it.
pub const RGFX_FEATURE_GUEST_TIMES: u64 = 0x1000_0000_0000;
/// `FEATURES_HI` bit 13, feature bit 45, `READ_ONLY_PAGES`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_HI`], and which means nothing without
/// [`RGFX_FEATURE_PAGES_BUFFERS`]: a pages block's `SURFACE_PAGES` may carry
/// [`RGFX_SURFACE_PAGES_READ_ONLY`], a block of pages its process maps read-only, and the host
/// keeps every object over such a block from writing its pages: a `BUFFER_OVER_PAGES` over one
/// carries `MTLResourceOptions` bit 24, whose GPU writes the host drops as Metal does, and nothing
/// else lies over one. The flag is where the host holds that line, not a hint: a process can post
/// any message on its rings, so a kext makes a pages block of read-only memory only where the host
/// offers the bit, and a host that does not know the flag never gets one. A host offers it where
/// it offers [`RGFX_FEATURE_PAGES_BUFFERS`].
pub const RGFX_FEATURE_READ_ONLY_PAGES: u64 = 0x2000_0000_0000;
/// `FEATURES_HI` bit 14, feature bit 46, `DOORBELL_PAGES`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_HI`]: BAR 4 ([`RGFX_PCI_BAR_DOORBELLS`]) holds the doorbell pages,
/// which the guest binds to its contexts with [`RGFX_REG_DOORBELL_PAGE_BIND`]. A page's write
/// wakes only the bound context's own rings, so the kext maps each context's page into its
/// process, which rings its queues there with no call into the kext. The host routes a queue's
/// plain doorbell on its context's page where the context has one, and on `DOORBELL` otherwise.
/// A host offers it where it maps BAR 4 and hands the device its writes.
pub const RGFX_FEATURE_DOORBELL_PAGES: u64 = 0x4000_0000_0000;
/// `FEATURES_HI` bit 15, feature bit 47, `PIPELINE_ANSWERS`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_HI`], and which means nothing without
/// [`RGFX_FEATURE_PIPELINE_KEYS`]: the device keeps the answers it gave keyed pipeline creates
/// ([`RGFX_PIPELINE_REFLECT_KEYED`]) from one run of the host to the next, under the identity of
/// what decides them (its platform's runtime, translation profile and driver), and hands them to
/// the kext on [`RGFX_CMD_PIPELINE_ANSWERS`]. A kept answer is the device's own to a create of
/// that key, which it accepted: a refused create is never kept, and a kept answer whose create the
/// device later refuses, or answers otherwise, is dropped. Every host offers it.
pub const RGFX_FEATURE_PIPELINE_ANSWERS: u64 = 0x8000_0000_0000;
/// `FEATURES_HI` bit 16, feature bit 48, `MULTI_DISPLAY`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_HI`]: the device has two displays or more, up to
/// [`RGFX_DISPLAYS_MAX`], each with its own scanout, modes, `VBLANK`s, cursor, presents, colour and
/// panel, which each display message names by the display's id (docs/graphics-kext/design.md,
/// "Multiple displays"). The device offers it only where its host made it with two displays or
/// more (`RvgGraphicsDeviceConfig`'s `display_count`, QEMU's `displays` property). A guest that
/// took it is answered `DISPLAYS` with every display, ids 0 up, and is sent the events of each; a
/// guest that did not sees display 0 alone, as from a device of one display, and is sent no event
/// of another. Each display's scanout lies in a room of its own ([`RGFX_SCANOUT_ROOM_ALIGN`]).
pub const RGFX_FEATURE_MULTI_DISPLAY: u64 = 0x1_0000_0000_0000;
/// `FEATURES_HI` bit 17, feature bit 49, `POINTER_OVER`: the guest moves its cursor onto the
/// display the host's pointer is over when [`RGFX_EVT_POINTER_OVER`] says so. The device offers
/// it beside [`RGFX_FEATURE_MULTI_DISPLAY`], only where its host made it with two displays or
/// more, and sends `POINTER_OVER` only to a guest that took both: one that took `MULTI_DISPLAY`
/// alone (a kext built before this bit, Ambrosia's module) is never sent it.
pub const RGFX_FEATURE_POINTER_OVER: u64 = 0x2_0000_0000_0000;
/// `FEATURES_HI` bit 18, feature bit 50, `DISPLAY_BOUNDS`: ring 0 carries
/// [`RGFX_CMD_DISPLAY_BOUNDS`], each display's rectangle in the guest's desktop, by which the
/// host carries its captured pointer, held at a display's edge where another display meets it,
/// onto that display (docs/graphics-kext/design.md, "Multiple displays"). The device offers it
/// beside [`RGFX_FEATURE_MULTI_DISPLAY`] and [`RGFX_FEATURE_POINTER_OVER`], only where its host
/// made it with two displays or more, and carries the pointer past a display's edge only for a
/// guest that took all three and sent the displays' rectangles.
pub const RGFX_FEATURE_DISPLAY_BOUNDS: u64 = 0x4_0000_0000_0000;
/// `FEATURES_HI` bit 19, feature bit 51, `POINTER_MOTION`: the guest's driver publishes a
/// relative pointing device of its own, a mouse, which [`RGFX_EVT_POINTER_MOTION`] moves and
/// clicks, beside the host's absolute one (QEMU's USB tablet). The device offers it where its
/// host made it with a window, and sends `POINTER_MOTION` only to a guest that took it, while the
/// window's captured pointer drives a game's hidden cursor (docs/graphics-kext/design.md, "A
/// pointer for games"): a guest that did not take it (a kext built before this bit, the option ROM,
/// Ambrosia's module) publishes no device and is never sent it.
pub const RGFX_FEATURE_POINTER_MOTION: u64 = 0x8_0000_0000_0000;
/// `FEATURES_HI` bit 20, feature bit 52, `VT_MESSAGES`: ring 0 carries
/// [`RGFX_CMD_VT_MESSAGE`], a message of the guest's VideoToolbox to its host side, which the host
/// answers [`RGFX_EVT_VT_STATUS`], and the control completion ring carries
/// [`RGFX_EVT_VT_MESSAGE`], a message of the host's side to the guest's VideoToolbox
/// (docs/graphics-kext/design.md, "Decided", 17). The arm64e kext takes it where it publishes the
/// paravirtual VideoToolbox service; the device sends `VT_MESSAGE` only to a guest that took it.
pub const RGFX_FEATURE_VT_MESSAGES: u64 = 0x10_0000_0000_0000;
/// `FEATURES_HI` bit 21, feature bit 53, `UPLOADS_WRITTEN`: a command buffer's stream may carry
/// [`RGFX_OP_UPLOADS_WRITTEN`](crate::ops::RGFX_OP_UPLOADS_WRITTEN), the bytes its encoders wrote
/// in the `Managed` chunks of its `set*Bytes:` data, which the host brings to their GPU copies as
/// it begins the buffer (docs/graphics-kext/design.md, bit 53). Every host offers it, and both
/// kexts take it; the bundle names the bytes there where its chunks are `Managed` (its device
/// answers `hasUnifiedMemory` NO), and a guest that did not take it tells the host each chunk's
/// bytes with a [`RGFX_CMD_BUFFER_DID_MODIFY_RANGE`] on ring 0 instead.
pub const RGFX_FEATURE_UPLOADS_WRITTEN: u64 = 0x20_0000_0000_0000;
/// `FEATURES_HI` bit 22, feature bit 54, `VT_RING`, which means something only with
/// [`RGFX_FEATURE_VT_MESSAGES`]: the host's side's messages to the guest's VideoToolbox go into
/// the VideoToolbox ring ([`RGFX_VT_RING_OFFSET`]), each followed on the control completion ring
/// by a [`RGFX_EVT_VT_RING`] that carries the ring's count through it, instead of onto the control
/// completion ring as `VT_MESSAGE` events. The guest takes the ring's messages through that count
/// where it takes the event, so they keep their place among the control ring's events, and the
/// control ring carries 24 bytes for each instead of up to 16 KiB: an encode's frames, two or
/// more such messages each at 1080p and 8 Mbit/s, filled that ring, which then held back the
/// display's `VBLANK`s and the presents' answers. A message that finds the VideoToolbox ring
/// with no room goes onto the control completion ring as a `VT_MESSAGE` event, as for a guest
/// that did not take the bit (docs/graphics-kext/design.md, "Decided", 17). The device offers it
/// where its host offers `VT_MESSAGES`; the arm64e kext takes it with `VT_MESSAGES`.
pub const RGFX_FEATURE_VT_RING: u64 = 0x40_0000_0000_0000;
/// `FEATURES_HI` bit 23, feature bit 55, `CURSOR_COLOR`: ring 0 carries
/// [`RGFX_CMD_CURSOR_COLOR`], the colour chain a display's cursor plane converts its pixels
/// through, and the host shows the cursor's image converted (docs/graphics-kext/design.md, "The
/// cursor's colour"). An arm64 WindowServer gives its cursor's layer a chain of its own, from
/// standard sRGB to the display's blend space, the space the guest's desktop is drawn in. Every
/// host offers it (the device converts the image before the host's screen is handed it); the
/// arm64e kext takes it, and takes the cursor layer's chain from WindowServer only where the host
/// took it. A guest that did not take it (the x86-64 kext, the option ROM, Ambrosia's module) has
/// its cursor shown as its pixels are.
pub const RGFX_FEATURE_CURSOR_COLOR: u64 = 0x80_0000_0000_0000;
/// `FEATURES_HI` bit 24, feature bit 56, `BUFFER_ESCAPED`: ring 0 carries
/// [`RGFX_CMD_BUFFER_ESCAPED`], the guest's note that a no-copy buffer the host reads in place
/// handed its identity out with no message of the device's naming it: its `gpuAddress` read, an
/// argument encoder's write of it, the `gpuResourceID` of a texture over it read, or the buffer,
/// or such a texture, added to a residency set (docs/graphics-kext/design.md, "A buffer's escaped
/// identity"). The guest posts it once for each such buffer, before the identity leaves it, and
/// waits for no answer. Every host offers it; the x86-64 and arm64e kexts take it, and the bundle
/// posts it only where they did.
pub const RGFX_FEATURE_BUFFER_ESCAPED: u64 = 0x100_0000_0000_0000;
/// `FEATURES_HI` bit 25, feature bit 57, `POOL`: the device has a Shared pool, guest physical
/// addresses past the guest's RAM that the host fills with its own memory as the guest's buffers
/// need it (`POOL_BASE_LO` to `POOL_LENGTH_HI` read its range). Ring 0 carries
/// [`RGFX_CMD_POOL_TAKE`] and [`RGFX_CMD_POOL_GIVE`], a context's ask for a block of the pool and
/// its giving one back, and a context's ring carries [`RGFX_CMD_HEAP_OVER_POOL`], a Shared heap
/// over a block it holds (docs/graphics-kext/design.md, "The Shared pool"). The host offers it
/// where it reserved a pool, but on Windows; the x86-64 and arm64e kexts take it, and a guest that
/// did not take it is refused every one of those messages and keeps its slabs in the region.
pub const RGFX_FEATURE_POOL: u64 = 0x200_0000_0000_0000;
/// `FEATURES_HI` bit 26, feature bit 58, `SURFACE_REFILL`: an
/// [`RGFX_OP_SURFACE_USE`](crate::ops::RGFX_OP_SURFACE_USE) may carry
/// [`RGFX_SURFACE_USE_REFILL`](crate::ops::RGFX_SURFACE_USE_REFILL), the guest's word that a CPU
/// wrote a cached surface's pages after the read-write lock that moved its generation, so the
/// host fills the cache again at the generation it holds (docs/graphics-kext/design.md, Decided
/// 14, "The device cache"). The guest's process tracks each cached surface's unlock seed while a
/// buffer is committed during a CPU's write lock, and sets the flag at the next buffer that uses
/// the surface once the unlock has come. Every host offers it, and carries the flag out wherever
/// a use carries it (the host reads no feature for a command buffer's ops); the x86-64 and arm64e
/// kexts take it, and the bundle sets the flag only where they did, so a guest that did not take
/// it never sets it, and a host that does not offer it never sees it.
pub const RGFX_FEATURE_SURFACE_REFILL: u64 = 0x400_0000_0000_0000;
/// `FEATURES_HI` bit 27, feature bit 59, `POOL_ADDRESSES`: the host gives each block of the Shared
/// pool its GPU address. Where the guest took it beside [`RGFX_FEATURE_POOL`], a `POOL_TAKE`'s
/// answer ([`RgfxPoolBlock`]) carries the GPU address of the block's first byte, and the guest
/// places the heaps it makes over the block ([`RGFX_CMD_HEAP_OVER_POOL`]) at that address plus
/// their spans' offsets, rather than at addresses it picks in its context's range
/// ([`RGFX_CONTEXT_VA_BASE`]). On metal2vulkan-rt's device the address is the block's Vulkan
/// device address; on the Vulkan platform before it, an address the host picks in
/// `[2^45, 2^46)`, below every context's range, and pins the heaps there. Every host with a pool
/// offers it; the x86-64 and arm64e kexts take it.
pub const RGFX_FEATURE_POOL_ADDRESSES: u64 = 0x800_0000_0000_0000;
/// `FEATURES_HI` bit 28, feature bit 60, `PRIVATE_BLOCKS`: the host gives each context blocks of
/// GPU addresses for its Private objects. Ring 0 carries [`RGFX_CMD_PRIVATE_TAKE`] and
/// [`RGFX_CMD_PRIVATE_GIVE`], a context's ask for a block and its giving one back. Where the
/// guest took it, the guest places its Private buffers and its heaps that are not Shared
/// ([`RGFX_CMD_BUFFER_CREATE`], [`RGFX_CMD_HEAP_CREATE`]) and its Managed buffers' GPU copies
/// ([`RGFX_CMD_MANAGED_BUFFER_CREATE`], [`RGFX_CMD_BUFFER_MANAGED_AT_OFFSET`]) at addresses inside
/// blocks its context holds, rather than at addresses it picks in its context's range
/// ([`RGFX_CONTEXT_VA_BASE`]). On metal2vulkan-rt's device a block is device memory no one maps,
/// and its address the memory's Vulkan device address; on the Vulkan platform before it, a range
/// of addresses the host picks in `[2^44, 2^45)`, below the Shared pool's, with no memory of its
/// own: the platform pins each object made there, as it pins the guest's picks. Every host offers
/// it; the x86-64 and arm64e kexts take it.
pub const RGFX_FEATURE_PRIVATE_BLOCKS: u64 = 0x1000_0000_0000_0000;
/// `FEATURES_HI` bit 29, feature bit 61, `IMPORT_ADDRESSES`: the host gives each pages block of a
/// process's own memory ([`RGFX_SURFACE_PAGES_PRIVATE`]) its GPU address. Where the guest took it
/// beside [`RGFX_FEATURE_PAGES_BUFFERS`], such a block's `SURFACE_PAGES` is answered
/// [`RgfxPagesAnswer`], which carries the GPU address of the block's view's first byte, and the
/// guest places the `BUFFER_OVER_PAGES` buffers it makes over the block at that address plus
/// their offsets in the view, rather than at addresses it picks in its context's range
/// ([`RGFX_CONTEXT_VA_BASE`]); buffers over the same bytes, which share a block, share their
/// address. Where the guest took [`RGFX_FEATURE_IMPORT_ADDRESSES_LATER`] too, the answer comes
/// before the view has its address and carries 0, and the host places the buffers at it, the
/// guest asking each one's ([`RGFX_CMD_PAGES_BUFFER_ADDRESS`]). On metal2vulkan-rt's device the
/// address is the Vulkan device address of the import of
/// the block's pages; on the Vulkan platform before it, an address the host picks in
/// `[2^43, 2^44)`, holding Metal's span of the view's length, where the platform pins the first
/// buffer made over the block, which then reaches the block's pages for the buffers made over the
/// same bytes after it until the block's release. A block of pages its process maps read-only
/// ([`RGFX_SURFACE_PAGES_READ_ONLY`]) is given none there: its buffers lie over copies of the
/// pages. On metal2vulkan-rt's device such a block is given the address of a copy of its pages,
/// whose GPU writes are dropped as each command buffer completes, as Metal drops a GPU write to
/// memory mapped read-only, so the pages keep the bytes their process wrote. Every host offers
/// it; the x86-64 and arm64e kexts take it.
pub const RGFX_FEATURE_IMPORT_ADDRESSES: u64 = 0x2000_0000_0000_0000;
/// `FEATURES_HI` bit 30, feature bit 62, `SURFACE_CACHE_EVERY`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_HI`], beside [`RGFX_FEATURE_SURFACE_PAGES`]: the host's GPU form of
/// every IOSurface is its device cache. It keeps one for every block in place the kext names
/// cached ([`RGFX_SURFACE_PAGES_CACHED`]), lays no texture over a surface's pages, and refuses a
/// block in place named uncached ([`RGFX_STATUS_REFUSED`]). A copy block of a surface is then a
/// second copy beside the cache the surface's other holders use. A kext that took it names every
/// block in place cached, and refuses before it posts anything an ask in place it cannot cache;
/// its tables of the surfaces' blocks, of a block's holders and of the cached surfaces grow
/// rather than refuse; and it and its bundle count each copy block made, by why the surface's
/// view is not in place, logging each reason the first time. The host acts on nothing more. A
/// host offers it where its platform caches every surface (metal2vulkan-rt's per-shape form) and
/// it offers [`RGFX_FEATURE_POOL`]; the x86-64 and arm64e kexts take it.
pub const RGFX_FEATURE_SURFACE_CACHE_EVERY: u64 = 0x4000_0000_0000_0000;
/// `FEATURES_HI` bit 31, feature bit 63, `MANAGED_AT_COPY`, which the guest takes through
/// [`RGFX_REG_DRIVER_FEATURES_HI`]: on a GPU that shares the CPU's memory, a `Managed` buffer is
/// its bytes, so its GPU address is its CPU copy's. Where the guest took it, a
/// [`RGFX_CMD_MANAGED_BUFFER_CREATE`]'s `gpu_address` is the address of the first byte of its CPU
/// copy, the `Shared` buffer `cpu_buffer_obj_id`, as that buffer's create gave it, and the guest
/// takes no address of its own for it, in a Private block ([`RGFX_FEATURE_PRIVATE_BLOCKS`]) or in
/// its context's range. The host makes the buffer there, a range of the block the CPU copy lies in;
/// where its platform makes it at another address (a CPU copy in no block it knows), the host keeps
/// the buffer there, as it makes it for a guest that did not take the bit, and counts it, and the
/// create is not refused. A host offers it where its GPU shares the CPU's memory, its platform
/// keeps a pin only inside a block it knows (metal2vulkan-rt's), whose Shared blocks the CPU copies
/// lie in, and it offers [`RGFX_FEATURE_POOL`], whose blocks the slabs' copies lie in: on the
/// Vulkan platform before it the guest's own pick is pinned, and a pin inside the CPU copy's range
/// would be refused. The x86-64 and arm64e kexts take it.
pub const RGFX_FEATURE_MANAGED_AT_COPY: u64 = 0x8000_0000_0000_0000;
/// `STATUS`, RW: writing [`RGFX_STATUS_RESET`] resets the device; the guest writes
/// [`RGFX_STATUS_DRIVER_READY`] once it has read the rings' headers.
pub const RGFX_REG_STATUS: u32 = 0x010;
/// `REGION_SIZE_LO`, RO: BAR 2's size, its low 32 bits.
pub const RGFX_REG_REGION_SIZE_LO: u32 = 0x014;
/// `REGION_SIZE_HI`, RO: BAR 2's size, its high 32 bits.
pub const RGFX_REG_REGION_SIZE_HI: u32 = 0x018;
/// `DOORBELL`, WO: writing a ring's id wakes its consumer. Where the guest took
/// [`RGFX_FEATURE_DOORBELL_ASKS`], a command queue's ring's id with [`RGFX_DOORBELL_ASKS`] set
/// wakes it too, and asks the host to answer the buffer it rang for inside the write.
pub const RGFX_REG_DOORBELL: u32 = 0x020;
/// Set in a `DOORBELL` value beside a command queue's ring's id, where the guest took
/// [`RGFX_FEATURE_DOORBELL_ASKS`]: the guest rang for a buffer that asks for its
/// [`RGFX_EVT_CB_SUBMITTED`], which the host may answer inside the write. A ring's id is its
/// region offset, a multiple of 8, so the bit names no ring.
pub const RGFX_DOORBELL_ASKS: u32 = 1;
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
/// `DEFERRED_BUDGET_LO`, RO: the bytes of the host GPU's memory the device caches of deferred
/// blocks ([`RGFX_FEATURE_SURFACE_DEFERRED`], [`RGFX_FEATURE_SURFACE_DEFERRED_ASK`]) may hold at
/// once, held and dormant, its low 32 bits:
/// a quarter of the budget of the GPU's device-local memory (`VK_EXT_memory_budget`'s heap budget
/// where the device has it, else a quarter of the heap's size), read once at the device's start
/// and the same for its life, so the kext's one read at its start holds. The kext defers no block
/// past it, and lets dormant deferred blocks go past half of it. It reads 0 where the host offers
/// neither feature, as at any offset a host does not know.
pub const RGFX_REG_DEFERRED_BUDGET_LO: u32 = 0x030;
/// `DEFERRED_BUDGET_HI`, RO: [`RGFX_REG_DEFERRED_BUDGET_LO`]'s high 32 bits.
pub const RGFX_REG_DEFERRED_BUDGET_HI: u32 = 0x034;
/// `HOST_TIME_LO`, RO: where the guest took [`RGFX_FEATURE_HOST_TIME`], the host's monotonic clock
/// at the read, in nanoseconds, its low 32 bits: the clock a `VBLANK`'s `host_time_ns` is on. Each
/// read takes the clock afresh, and nothing is latched for [`RGFX_REG_HOST_TIME_HI`]: the guest
/// reads `HI`, `LO` and `HI` again, and takes the second `HI` where `LO` is below 2^31, else the
/// first. It reads 0 where the guest did not take the feature, as at any offset a host does not
/// know.
pub const RGFX_REG_HOST_TIME_LO: u32 = 0x038;
/// `HOST_TIME_HI`, RO: the host's monotonic clock at the read, its high 32 bits, as
/// [`RGFX_REG_HOST_TIME_LO`].
pub const RGFX_REG_HOST_TIME_HI: u32 = 0x03c;
/// `DOORBELL_PAGE_BIND`, WO: where the guest took [`RGFX_FEATURE_DOORBELL_PAGES`], binds doorbell
/// page `value & 0xff` to context `value >> RGFX_DOORBELL_PAGE_BIND_CONTEXT_SHIFT`, or unbinds it
/// for context 0: from the write on, the page's writes wake the rings of that context's command
/// queues alone, and its queues made after it have their plain doorbells routed on the page. A
/// page past [`RGFX_DOORBELL_PAGES`] is ignored, and a reset unbinds every page. It reads 0.
pub const RGFX_REG_DOORBELL_PAGE_BIND: u32 = 0x040;
/// Where a [`RGFX_REG_DOORBELL_PAGE_BIND`] value's context starts: bits 8 to 31, so a context of
/// 2^24 or more (whose object ids would pass 64 bits) gets no page.
pub const RGFX_DOORBELL_PAGE_BIND_CONTEXT_SHIFT: u32 = 8;
/// `POOL_BASE_LO`, RO: where the guest took [`RGFX_FEATURE_POOL`], the guest physical address of
/// the Shared pool's first byte, its low 32 bits; 0 otherwise. The range is the same for the
/// device's life, so the kext's one read at its start holds.
pub const RGFX_REG_POOL_BASE_LO: u32 = 0x044;
/// `POOL_BASE_HI`, RO: as [`RGFX_REG_POOL_BASE_LO`], its high 32 bits.
pub const RGFX_REG_POOL_BASE_HI: u32 = 0x048;
/// `POOL_LENGTH_LO`, RO: where the guest took [`RGFX_FEATURE_POOL`], the Shared pool's length in
/// bytes, its low 32 bits; 0 otherwise.
pub const RGFX_REG_POOL_LENGTH_LO: u32 = 0x04c;
/// `POOL_LENGTH_HI`, RO: as [`RGFX_REG_POOL_LENGTH_LO`], its high 32 bits.
pub const RGFX_REG_POOL_LENGTH_HI: u32 = 0x050;
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
/// Where the guest's GPU addresses start. The kext hands each context (each process) a range of
/// its own from `[RGFX_CONTEXT_VA_BASE, RGFX_CONTEXT_VA_END)`, the top half of the Metal address
/// space, `[2^46, 2^47)`, which no range the host places itself reaches: the platform's runtime
/// places its own buffers of Metal's space lowest first from 2^32, and the Vulkan platform's
/// model of Metal's layout places the host's unpinned buffers and textures lowest first from
/// 2^40, each as far as its live ranges reach, which the host's memory bounds far below 2^46. A
/// guest's pin meets only the guest's own ranges then. The host takes each range as given: it
/// knows no context's range, only that the guest's lie here. The addresses the host gives the
/// Shared pool's blocks on that platform ([`RGFX_FEATURE_POOL_ADDRESSES`]) lie below them, in
/// `[2^45, 2^46)`, and those of the contexts' Private blocks ([`RGFX_FEATURE_PRIVATE_BLOCKS`])
/// below those, in `[2^44, 2^45)`.
pub const RGFX_CONTEXT_VA_BASE: u64 = 0x4000_0000_0000;
/// Where the guest's GPU addresses end: 2^47, the Metal address space's end.
pub const RGFX_CONTEXT_VA_END: u64 = 0x8000_0000_0000;
/// The unit the kext hands the contexts' ranges out in: 64 GiB, so the guest's addresses hold
/// [`RGFX_CONTEXT_VA_UNITS`] of them. A context holds an aligned run of units, a power of two of
/// them, from its open to its close (`guest/kext/src/RgfxContextSlots.hpp`).
pub const RGFX_CONTEXT_VA_UNIT: u64 = 0x10_0000_0000;
/// How many units the guest's GPU addresses hold: 1024.
pub const RGFX_CONTEXT_VA_UNITS: u32 = 1024;
/// The GPU addresses a context holds where its bundle takes no span from the kext (a kext or a
/// bundle without the kext's `ContextSpan` feature, whose `OpenContext` answers five values):
/// 2 TiB from a base 2 TiB aligned, as each of the 32 slots of a kext before it held.
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
/// The texture resource ids in one small chunk ([`RGFX_FEATURE_SMALL_ID_CHUNKS`]): small chunk
/// `k` holds ids `1 + k * 16` to `16 + k * 16`, so chunk `k` of [`RGFX_RESOURCE_ID_CHUNK`] ids is
/// small chunks `4k` to `4k + 3`.
pub const RGFX_RESOURCE_ID_SMALL_CHUNK: u64 = 16;
/// How many small chunks the pool holds: the same ids as its [`RGFX_RESOURCE_ID_CHUNKS`] chunks.
pub const RGFX_RESOURCE_ID_SMALL_CHUNKS: u32 = 1020;
/// The control submission ring's id, which the guest writes to `DOORBELL`.
pub const RGFX_CONTROL_RING_ID: u32 = 0;
/// The control completion ring's MSI-X vector.
pub const RGFX_CONTROL_VECTOR: u32 = 0;

/// [`RgfxRingHeader::consumer_idle`] while the consumer sleeps until a doorbell: the producer
/// rings for each message it publishes.
pub const RGFX_CONSUMER_ASLEEP: u32 = 1;
/// [`RgfxRingHeader::consumer_idle`] while a queue's consumer carries out a command buffer on its
/// own thread, after which it takes the ring again with no doorbell: the producer rings only for
/// a [`RGFX_CMD_COMMAND_BUFFER`] whose last op asks for its [`RGFX_EVT_CB_SUBMITTED`]
/// ([`RGFX_OP_SUBMITTED`](crate::ops::RGFX_OP_SUBMITTED)), which the host may answer inside the
/// doorbell's write. A producer that rings only for [`RGFX_CONSUMER_ASLEEP`] rings for none,
/// and such a buffer is answered once the consumer takes it.
pub const RGFX_CONSUMER_WALKING: u32 = 2;

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

/// Where the VideoToolbox ring is in the region ([`RGFX_FEATURE_VT_RING`]): host to guest, right
/// after the control completion ring's span, in the room before the guest's allocations that no
/// other ring or record takes and the kexts map with the control rings. It is a ring's header
/// ([`RgfxRingHeader`]) and its byte ring of [`RGFX_VT_RING_DATA_SIZE`] bytes. The guest never
/// arms it: each [`RGFX_EVT_VT_RING`] on the control completion ring says what to take. It is the
/// device's memory, as the control rings are: nothing of the guest's RAM is wired for it.
pub const RGFX_VT_RING_OFFSET: u64 = RGFX_CONTROL_COMPLETE_OFFSET + RGFX_CONTROL_RING_SPAN;
/// The size of the VideoToolbox ring's byte ring: 2 MiB, the largest power of two that fits
/// after its header before the guest's allocations. It holds 127 of the longest messages, 16 KiB
/// each: an encode's session sends a frame's messages one after another and waits for the guest
/// only at its last, so a frame of up to 127 of them, about 2 MiB, goes in before the guest takes
/// any of it.
pub const RGFX_VT_RING_DATA_SIZE: u32 = 0x20_0000;

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
    /// ([`RGFX_FEATURE_CONSUMED_DOORBELL`]), written by the host; on a submission ring, 1 while a
    /// guest producer waits for room ([`RGFX_FEATURE_SUBMIT_ROOM`]), written 1 by the guest and 0
    /// by the host as it posts the ring's [`RGFX_EVT_SUBMIT_ROOM`]; 0 on every other ring.
    pub producer_waiting: u32,
    /// Zero.
    pub reserved_14: u32,
    /// On the control submission ring (ring 0), where the guest took
    /// [`RGFX_FEATURE_GUEST_TIMES`]: the guest's clock (`mach_absolute_time`, in nanoseconds)
    /// less the host's (the clock [`RGFX_REG_HOST_TIME_LO`] reads), as the kext's reading of
    /// `HOST_TIME` between two reads of its own clock found it, written by the guest before
    /// [`Self::guest_clock_bound_ns`]. 0 on every other ring.
    pub guest_clock_offset_ns: i64,
    /// On ring 0, where the guest took [`RGFX_FEATURE_GUEST_TIMES`]: how far
    /// [`Self::guest_clock_offset_ns`] may be off, in nanoseconds (half its reading's round trip,
    /// widened by the reading's age), stored by the guest with release after the offset; 0 while
    /// the guest holds no reading, and on every other ring.
    pub guest_clock_bound_ns: u32,
    /// The rest of `producer`'s line.
    pub reserved_24: [u8; 28],
    /// The consumer's free-running byte count, written by the consumer.
    pub consumer: u64,
    /// The rest of `consumer`'s line.
    pub reserved_48: [u8; 56],
    /// [`RGFX_CONSUMER_ASLEEP`] while the consumer sleeps until a doorbell,
    /// [`RGFX_CONSUMER_WALKING`] while a queue's consumer carries out a command buffer and comes
    /// back to the ring after it, 0 while it takes the ring; written by the consumer.
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
/// Command `DISPLAY_COLOR` ([`RgfxDisplayColor`]), where the guest took
/// [`RGFX_FEATURE_DISPLAY_COLOR`]: not answered.
pub const RGFX_CMD_DISPLAY_COLOR: u32 = 7;
/// Command `SET_DISPLAY_POWER` ([`RgfxSetDisplayPower`]), where the guest took
/// [`RGFX_FEATURE_DISPLAY_POWER`]: answered by [`RGFX_EVT_COMPLETE`].
pub const RGFX_CMD_SET_DISPLAY_POWER: u32 = 8;
/// Command `DISPLAY_BOUNDS` ([`RgfxDisplayBounds`]), where the guest took
/// [`RGFX_FEATURE_DISPLAY_BOUNDS`]: not answered.
pub const RGFX_CMD_DISPLAY_BOUNDS: u32 = 9;
/// Command `CURSOR_COLOR` ([`RgfxCursorColor`]), where the guest took
/// [`RGFX_FEATURE_CURSOR_COLOR`]: not answered.
pub const RGFX_CMD_CURSOR_COLOR: u32 = 10;
/// Event `COMPLETE` ([`RgfxComplete`]): a command's answer, with its `seqno`.
pub const RGFX_EVT_COMPLETE: u32 = 0x100;
/// Event `DISPLAYS` ([`RgfxDisplays`]): `GET_DISPLAYS`'s answer, with its `seqno`.
pub const RGFX_EVT_DISPLAYS: u32 = 0x101;
/// Event `DISPLAY_CHANGED` ([`RgfxDisplayChanged`]): the guest asks `GET_DISPLAYS` again.
pub const RGFX_EVT_DISPLAY_CHANGED: u32 = 0x102;
/// Event `VBLANK` ([`RgfxVblank`]): a refresh of the display's host screen.
pub const RGFX_EVT_VBLANK: u32 = 0x103;
/// Event `POINTER_OVER` ([`RgfxPointerOver`]), sent only to a guest that took
/// [`RGFX_FEATURE_POINTER_OVER`] and [`RGFX_FEATURE_MULTI_DISPLAY`]: the host's pointer is over a
/// display the guest's cursor is not on, and the guest moves its cursor there.
pub const RGFX_EVT_POINTER_OVER: u32 = 0x104;
/// Event `POINTER_MOTION` ([`RgfxPointerMotion`]), sent only to a guest that took
/// [`RGFX_FEATURE_POINTER_MOTION`]: the host's pointing device moved, turned its wheel or pressed
/// or let go of a button, and the guest's relative pointing device does the same.
pub const RGFX_EVT_POINTER_MOTION: u32 = 0x105;

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
/// `COMPLETE`'s status, to a `SURFACE_PAGES` that asks whether its block needs a device cache
/// ([`RGFX_SURFACE_PAGES_ASK_CACHE`]): the host's textures over the surface's pages would not lie
/// in place for every access, so the kext names the block cached
/// ([`RGFX_SURFACE_PAGES_CACHED`]) before any process holds it; the block has no view meanwhile.
pub const RGFX_STATUS_CACHE: u32 = 7;

/// Scanout pixels: 32-bit BGRA8, `kIO32BGRAPixelFormat`, the one format there is.
pub const RGFX_SCANOUT_BYTES_PER_PIXEL: u32 = 4;
/// The largest cursor image, in pixels each way.
pub const RGFX_CURSOR_MAX_SIZE: u32 = 256;
/// The displays a device has where its host names no other count (`RvgGraphicsDeviceConfig`'s
/// `display_count` of 0 or 1): display 0 alone. A host gives it up to [`RGFX_DISPLAYS_MAX`].
pub const RGFX_DISPLAY_COUNT: u32 = 1;
/// The most displays a device has. A device of two or more offers [`RGFX_FEATURE_MULTI_DISPLAY`],
/// and has them from its start to its end: none is added or removed while it runs.
pub const RGFX_DISPLAYS_MAX: u32 = 4;
/// What a display's scanout room is a multiple of on a device of `count` displays: the scanout's
/// room ([`RGFX_SCANOUT_SPAN`] bytes at [`RGFX_ALLOCATIONS_OFFSET`]) split evenly among them,
/// `room = (RGFX_SCANOUT_SPAN / count)` rounded down to a multiple of this, display `i`'s from
/// `RGFX_ALLOCATIONS_OFFSET + i * room`: 128 MiB for one display, as before any display had a
/// room of its own, 64 MiB each for two, 0x2aa_0000 bytes each for three and 32 MiB each for
/// four. A display's `SET_SCANOUT` lies in its own room, so display 0's starts where the one
/// display's always has, and the firmware's framebuffer lies in display 0's.
pub const RGFX_SCANOUT_ROOM_ALIGN: u64 = 0x1_0000;
/// The row alignment a display's modes are sized for on a device of more than one display
/// ([`RGFX_FEATURE_MULTI_DISPLAY`]): the device offers each display only modes whose rows,
/// `width * RGFX_SCANOUT_BYTES_PER_PIXEL` bytes rounded up to a multiple of this, fit its room
/// ([`RGFX_SCANOUT_ROOM_ALIGN`]) `height` times, as the guest's kexts and the option ROM lay
/// their scanouts out.
pub const RGFX_SCANOUT_STRIDE_ALIGN: u32 = 256;

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
/// the guest's own pages in place. A host without the feature refuses the message as one it does
/// not know. Over a read-only pages block ([`RGFX_SURFACE_PAGES_READ_ONLY`]) its options carry
/// `MTLResourceOptions` bit 24, or it is refused.
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
/// Command `HEAP_OVER_PAGES` ([`RgfxHeapOverPages`]), where the guest took
/// [`RGFX_FEATURE_HEAP_OVER_PAGES`]: a Shared heap over the guest's RAM a pages block of the
/// sender's context holds, which must be one run, in place; buffers are placed in it as in a heap
/// over host memory (`BUFFER_AT_OFFSET`, `BUFFER_MANAGED_AT_OFFSET`). A host without the feature
/// refuses the message as one it does not know.
pub const RGFX_CMD_HEAP_OVER_PAGES: u32 = 0x22a;
/// Command `ICB_SLOTS` ([`RgfxIcbSlots`], then its runs, [`RgfxIcbSlotRun`]), where the guest took
/// [`RGFX_FEATURE_ICB_SLOTS`]: for each run, the slots of `command_count` commands of indirect
/// command buffer `icb_obj_id` from `first_command`, `slot_bytes` apart from `region_offset`,
/// which must lie in one block the sender holds, as they stand as the host reads them
/// ([`crate::icb`], "Slots"). Each command's bits are carried out on the platform's buffer as its
/// records would be; a run the host cannot read (an object the sender does not hold, a span
/// outside its blocks, a shape it does not know) is skipped alone, as is a command the platform
/// refuses. Ring 0 is carried out before any command buffer the process posts after it, as
/// `ICB_WRITE` is. Not answered.
pub const RGFX_CMD_ICB_SLOTS: u32 = 0x22c;
/// Command `MANAGED_BUFFER_CREATE` ([`RgfxManagedBufferCreate`]), where the guest took
/// [`RGFX_FEATURE_MANAGED_BUFFER_SLABS`]: a `Managed` buffer at the guest's address whose CPU copy
/// is the first bytes of a `Shared` buffer over the guest's memory the sender holds. A host
/// without the feature refuses the message as one it does not know. Not answered.
pub const RGFX_CMD_MANAGED_BUFFER_CREATE: u32 = 0x22d;
/// Command `MESH_RENDER_PIPELINE_CREATE` ([`RgfxMeshRenderPipelineCreate`]), where the guest took
/// [`RGFX_FEATURE_MESH_PIPELINES`]: a mesh render pipeline from a descriptor
/// (`newRenderPipelineStateWithMeshDescriptor:options:reflection:error:`), answered by
/// [`RGFX_EVT_PIPELINE_REFLECTION`] of [`RGFX_PIPELINE_KIND_MESH`] with its mesh tail
/// ([`RgfxMeshPipelineReflection`]). `OBJECT_DESTROY` drops it, as a render pipeline. A host
/// without the feature refuses the message as one it does not know.
pub const RGFX_CMD_MESH_RENDER_PIPELINE_CREATE: u32 = 0x22e;
/// Command `SHARED_EVENT_NAME` ([`RgfxSharedEventName`]), which the accelerator kext posts for a
/// process's `SharedEventExport`, where the guest took [`RGFX_FEATURE_SHARED_EVENT_NAMES`]: the
/// process's shared event `obj_id` is named `name` device-wide, so another process may make an
/// event of its own from the name ([`RGFX_CMD_SHARED_EVENT_IMPORT`]), until the kext lets go of
/// the name ([`RGFX_CMD_SHARED_EVENT_UNNAME`]). Answered by
/// [`RGFX_EVT_COMPLETE`]: [`RGFX_STATUS_OK`], or [`RGFX_STATUS_NOT_FOUND`] where the sender holds
/// no shared event of that id (one whose create the host refused among them) or no queue of
/// `listen_queue_obj_id`'s; the kext hands the process the name's port only once it is answered
/// OK, so a later import finds the name. A host without the feature refuses the message as one it
/// does not know.
pub const RGFX_CMD_SHARED_EVENT_NAME: u32 = 0x22f;
/// Command `SHARED_EVENT_IMPORT` ([`RgfxSharedEventImport`]), which the accelerator kext posts for
/// a process's `SharedEventImport` of a port another process's `SharedEventExport` gave out, where
/// the guest took [`RGFX_FEATURE_SHARED_EVENT_NAMES`]: the process's new shared event `obj_id` is
/// the host event named `name`. Answered by [`RGFX_EVT_SHARED_EVENT_VALUE`], with the event's
/// value, or [`RGFX_STATUS_NOT_FOUND`] where the name was let go of, or never given (nothing is
/// made then). A host without the feature refuses the message as one it does not know.
pub const RGFX_CMD_SHARED_EVENT_IMPORT: u32 = 0x230;
/// Command `SHARED_EVENT_UNNAME` ([`RgfxSharedEventUnname`]), which the accelerator kext posts
/// once the port a `SHARED_EVENT_NAME` gave out is gone (no process holds it, or its connection
/// was closed), where the guest took [`RGFX_FEATURE_SHARED_EVENT_NAMES`]: no process can import
/// the name from then on. A name holds its event as a holder does: the host's event goes once
/// its last name and its last holder are gone. Context 0's, which no process is. Not answered. A
/// host without the feature refuses the message as one it does not know.
pub const RGFX_CMD_SHARED_EVENT_UNNAME: u32 = 0x231;
/// Command `TEXTURE_DID_MODIFY_DATA` ([`RgfxTextureDidModifyData`]), where the guest took
/// [`RGFX_FEATURE_TEXTURE_DID_MODIFY_DATA`]: the CPU changed the bytes a texture over a buffer lies
/// over (`didModifyData`, SPI). A mipmapped one (`TEXTURE_FROM_BUFFER` of more than one level)
/// holds a copy of them, every level made from them, taken at its first use; it takes them again
/// at its first use in a command buffer of the context begun after this, on any queue, and a CPU
/// write before that use is in the copy. Any other texture reads the bytes as they lie, and the
/// message changes nothing for it. Not answered; a texture the host does not hold is refused. A
/// host without the feature refuses the message as one it does not know.
pub const RGFX_CMD_TEXTURE_DID_MODIFY_DATA: u32 = 0x232;
/// Command `RATE_MAP_CREATE` ([`RgfxRateMapCreate`]), where the guest took
/// [`RGFX_FEATURE_RATE_MAPS`]: a rasterization rate map of the descriptor the message carries
/// (`newRasterizationRateMapWithDescriptor:`), made under `obj_id`, answered by
/// [`RGFX_EVT_RATE_MAP_STATE`]. A host without the feature refuses the message as one it does
/// not know.
pub const RGFX_CMD_RATE_MAP_CREATE: u32 = 0x233;
/// Command `RATE_MAP_RESET` ([`RgfxRateMapCreate`], its `obj_id` the map), where the guest took
/// [`RGFX_FEATURE_RATE_MAPS`]: the map compiled again from the descriptor the message carries
/// (`resetUsingDescriptor:`, SPI), its mutability kept, answered by [`RGFX_EVT_RATE_MAP_STATE`]
/// with the map's new state. A pass that took the map and whose command buffer the host has not
/// yet submitted draws by the map as reset. Makes nothing.
pub const RGFX_CMD_RATE_MAP_RESET: u32 = 0x234;
/// Command `RATE_MAP_QUERY` ([`RgfxRateMapQuery`]), where the guest took
/// [`RGFX_FEATURE_RATE_MAPS`]: one of a map's answers that takes an argument
/// (`RGFX_RATE_MAP_QUERY_*`), answered by [`RGFX_EVT_RATE_MAP_ANSWER`]. Makes nothing.
pub const RGFX_CMD_RATE_MAP_QUERY: u32 = 0x235;
/// Command `FUNCTION_HANDLE_CREATE` ([`RgfxFunctionHandleCreate`]), where the guest took
/// [`RGFX_FEATURE_FUNCTION_POINTERS`]: compute pipeline state `pipeline_obj_id`'s handle of the
/// function the message names (`functionHandleWithFunction:`), made under `obj_id`, answered by
/// [`RGFX_EVT_FUNCTION_POINTER_STATE`]: [`RGFX_STATUS_NOT_FOUND`] where Metal answers nil (a
/// function the state's kernel does not link), and nothing is made. A pipeline still being
/// compiled is waited for. A host without the feature refuses the message as one it does not
/// know.
pub const RGFX_CMD_FUNCTION_HANDLE_CREATE: u32 = 0x236;
/// Command `FUNCTION_TABLE_CREATE` ([`RgfxFunctionTableCreate`]), where the guest took
/// [`RGFX_FEATURE_FUNCTION_POINTERS`]: a visible function table of `function_count` null entries
/// of compute pipeline state `pipeline_obj_id` (`newVisibleFunctionTableWithDescriptor:`), made
/// under `obj_id`, answered by [`RGFX_EVT_FUNCTION_POINTER_STATE`] with the table's
/// `gpuResourceID`, which the guest's argument buffers hold, its `allocatedSize` and its
/// `resourceOptions`. A pipeline still being compiled is waited for. `OBJECT_DESTROY` drops it.
pub const RGFX_CMD_FUNCTION_TABLE_CREATE: u32 = 0x237;
/// Command `FUNCTION_TABLE_SET` ([`RgfxFunctionTableSet`], then `count` handle ids), where the
/// guest took [`RGFX_FEATURE_FUNCTION_POINTERS`]: entries `first` to `first + count - 1` of table
/// `obj_id` call the handles the message names, in order, 0 for a null entry
/// (`setFunction:atIndex:`, `setFunctions:withRange:`). Ring 0 is carried out before any command
/// buffer the process posts after it, so a dispatch sees every entry set before its commit, as
/// Metal's does. Not answered: a set the host cannot carry out (a table or handle it does not
/// hold, a range past the table, a handle the platform refuses) sets none of its entries.
pub const RGFX_CMD_FUNCTION_TABLE_SET: u32 = 0x238;
/// Command `DYNAMIC_LIBRARY_CREATE` ([`RgfxDynamicLibraryCreate`]), where the guest took
/// [`RGFX_FEATURE_DYNAMIC_LIBRARIES`]: library `library_obj_id` as a dynamic library
/// (`newDynamicLibrary:error:`), made under `obj_id`, answered by [`RGFX_EVT_COMPLETE`]:
/// [`RGFX_STATUS_OK`] where it was made; [`RGFX_STATUS_NOT_FOUND`] for a library the sender does
/// not hold; [`RGFX_STATUS_REFUSED`] where the library was refused or the platform refuses it as
/// a dynamic library (one of another type, of no install name, or whose code reads a function
/// constant); [`RGFX_STATUS_INVALID_ARGUMENT`] for a message too short or an id not the
/// sender's. Nothing is made but on success. A library still being made is waited for.
/// `OBJECT_DESTROY` drops it; a pipeline state made with it keeps what it linked.
pub const RGFX_CMD_DYNAMIC_LIBRARY_CREATE: u32 = 0x239;
/// Command `ICB_SLOTS_IN_PAGES` ([`RgfxIcbSlots`], then its runs, [`RgfxIcbSlotPagesRun`]),
/// where the guest took [`RGFX_FEATURE_ICB_SLOTS_IN_PAGES`]: as [`RGFX_CMD_ICB_SLOTS`], each run's
/// slots `slot_bytes` apart from `pages_offset` in the pages of the pages block at `block_offset`,
/// a block of the sender's context described by a `SURFACE_PAGES` of
/// [`RGFX_SURFACE_PAGES_PRIVATE`] and holding at least `block_size` bytes, which the host reads in
/// place ([`crate::icb`], "Slots"). A run whose block is no such block of the sender's, or whose
/// slots pass its pages, is skipped alone, as a run of `ICB_SLOTS` the host cannot read is. Ring 0
/// is carried out before any command buffer the process posts after it, as `ICB_WRITE` is. Not
/// answered.
pub const RGFX_CMD_ICB_SLOTS_IN_PAGES: u32 = 0x23a;
/// Command `BUFFER_ESCAPED` ([`RgfxBufferEscaped`]), where the guest took
/// [`RGFX_FEATURE_BUFFER_ESCAPED`]: the identity of buffer `obj_id`, a no-copy buffer the host
/// reads in place (one `BUFFER_OVER_PAGES` or `BUFFER_OVER_HOST_MEMORY` made, not `Managed`), left
/// the guest's Metal with no message naming the buffer, by the act `kind` names
/// (`RGFX_BUFFER_ESCAPED_*`): from now on the GPU may reach the buffer through memory alone. Sent
/// once in the buffer's life, before the identity leaves, so ring 0 carries it out before any
/// command buffer of the process that could use the identity. Not answered, and it changes nothing
/// the guest observes: the host marks the buffer and counts it. A note for an object that is no
/// such buffer is refused as a bad reference. A host without the feature refuses the message as one
/// it does not know.
pub const RGFX_CMD_BUFFER_ESCAPED: u32 = 0x23b;
/// Command `HEAP_OVER_POOL` ([`RgfxHeapOverHostMemory`], its `region_offset` the heap's offset in
/// the Shared pool), where the guest took [`RGFX_FEATURE_POOL`]: a Shared heap over `size_bytes`
/// of a pool block the sender's context holds ([`RGFX_CMD_POOL_TAKE`]), at that offset, as
/// [`RGFX_CMD_HEAP_OVER_HOST_MEMORY`] makes one over a region block. The heap holds the block:
/// the block's `POOL_GIVE` is refused while the heap lives. A span that does not lie in one block
/// of the sender's makes nothing, as an unknown id does. Its `gpu_address` is where the heap lies:
/// where the guest took [`RGFX_FEATURE_POOL_ADDRESSES`], the block's GPU address as its take's
/// answer gave it plus the span's offset in the block, else the guest's own pick. Not answered.
pub const RGFX_CMD_HEAP_OVER_POOL: u32 = 0x23c;
/// Command `PAGES_BUFFER_ADDRESS` ([`RgfxPagesBufferAddress`]), where the guest took
/// [`RGFX_FEATURE_IMPORT_ADDRESSES_LATER`]: the GPU address of buffer `obj_id`, one a
/// `BUFFER_OVER_PAGES` of the sender's made, answered by an [`RgfxPagesAnswer`], an
/// [`RGFX_EVT_COMPLETE`] of 32 bytes: [`RGFX_STATUS_OK`] and the address the buffer lies at, which
/// its `gpuAddress` answers; [`RGFX_STATUS_REFUSED`] and 0 for a buffer whose create the host
/// refused; [`RGFX_STATUS_NOT_FOUND`] and 0 where `obj_id` names no buffer over a pages block of
/// the sender's (another context's object among them); [`RGFX_STATUS_INVALID_ARGUMENT`] and 0
/// for a message too short. Carried out after the sender's messages before it, the buffer's
/// create among them, which waits for its block's view's address: the answer comes once the
/// buffer lies there. Makes nothing.
pub const RGFX_CMD_PAGES_BUFFER_ADDRESS: u32 = 0x23d;

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
/// `RESOURCE_ID_SMALL_CHUNK` ([`RgfxResourceIdChunk`]), on ring 0 from the accelerator kext,
/// where the guest took [`RGFX_FEATURE_SMALL_ID_CHUNKS`]: as `RESOURCE_ID_CHUNK`, of small chunk
/// `chunk` ([`RGFX_RESOURCE_ID_SMALL_CHUNK`] ids, below [`RGFX_RESOURCE_ID_SMALL_CHUNKS`]). A host
/// without the feature refuses the message as one it does not know.
pub const RGFX_CMD_RESOURCE_ID_SMALL_CHUNK: u32 = 0x2f8;
/// `PIPELINE_ANSWERS` ([`RgfxPipelineAnswersQuery`]), on ring 0 from the kext, where the guest
/// took [`RGFX_FEATURE_PIPELINE_ANSWERS`]: the answers the device keeps from earlier runs, newest
/// first, from the `first`th, at most `most` of them. Each goes on the control completion ring as
/// a [`RGFX_EVT_PIPELINE_REFLECTION`] with this message's `seqno`: a made pipeline's keyed reply
/// (its `status` [`RGFX_PIPELINE_OK`], [`RGFX_PIPELINE_REFLECT_KEYED`] set, its key after it at
/// [`RGFX_PIPELINE_REFLECTION_KEY_OFFSET`]), with its `obj_id`, `gpu_resource_id`,
/// `unique_identifier` and `resource_index` 0, as the kext keeps an answer. Then an
/// [`RGFX_EVT_COMPLETE`] of the same `seqno` whose `status` is how many went; fewer than `most`
/// where none is left. No command buffer waits for it, and it makes nothing.
pub const RGFX_CMD_PIPELINE_ANSWERS: u32 = 0x2f9;
/// `BLOCK_RELEASE` ([`RgfxBlockRelease`]), on ring 0 from the accelerator kext: context `context`
/// freed its region block of `size` bytes at `offset`, which the kext gives out again only once
/// this is answered [`RGFX_EVT_COMPLETE`] with [`RGFX_STATUS_OK`]. The host answers
/// [`RGFX_STATUS_REFUSED`] at once where an object it holds, or a live queue's ring, still lies in
/// the block; otherwise it answers OK once every command buffer of the context it had committed
/// has finished, and, of a shared block, every one each other holder had committed before its
/// [`RGFX_CMD_BLOCK_UNSHARE`], its `CONTEXT_CLOSE`, or this release where neither came first (an
/// unshare posted behind another holder's release, or never posted), once the host's display no
/// longer shows or draws an IOSurface's texture over the block, as a compositor holds the surface
/// it shows: until the display pipe's next present. Posted after every message the process posted
/// before its free, and nothing the process posts after it can name the block. A block over an
/// IOSurface's pages with a device cache has the cache copied into the pages first, where a command
/// buffer wrote it since the last copy, whatever [`RGFX_BLOCK_RELEASE_NO_CLIENT`] says, but where
/// [`RGFX_BLOCK_RELEASE_UNHELD`] says no process can read them.
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
/// refused while an object of any context lies in the block; its answer waits for the command
/// buffers this context committed before its unshare too, which may still use the block.
pub const RGFX_CMD_BLOCK_UNSHARE: u32 = 0x2f5;
/// `POOL_TAKE` ([`RgfxPoolTake`]), on ring 0 from the accelerator kext, where the guest took
/// [`RGFX_FEATURE_POOL`]: context `context` asks for a block of the Shared pool of at least
/// `length` bytes. Answered [`RGFX_EVT_POOL_BLOCK`] with this message's `seqno`: the block's offset
/// in the pool and its length, `length` rounded up to [`RGFX_POOL_BLOCK_ALIGN`], every byte of it
/// zero, and the block the context's from then until its `POOL_GIVE` or the context's
/// `CONTEXT_CLOSE`; or [`RGFX_STATUS_INVALID_ARGUMENT`] for a `length` of 0 or past
/// [`RGFX_POOL_TAKE_MAX`], and [`RGFX_STATUS_REFUSED`] where the pool has no room for it. The
/// answer may come after later messages' answers: a take that finds no room in the pool's memory
/// waits for the host to add more, and each context's takes are answered in its own order, never
/// behind another context's. A take of a context closed before its answer is given back by the
/// host, and its answer is not sent.
pub const RGFX_CMD_POOL_TAKE: u32 = 0x2fb;
/// `POOL_GIVE` ([`RgfxPoolGive`]), on ring 0 from the accelerator kext: context `context` gives
/// back its pool block of `length` bytes at pool offset `pool_offset`, which no process maps any
/// more. Answered [`RGFX_EVT_COMPLETE`]: [`RGFX_STATUS_REFUSED`] at once where an object the host
/// holds (a heap) still lies in the block, or [`RGFX_STATUS_NOT_FOUND`] where the context holds no
/// such block; otherwise [`RGFX_STATUS_OK`] once every command buffer the context had committed
/// has finished, and only then does the block go back to the pool. Posted after every message the
/// process posted before it gave the block up.
pub const RGFX_CMD_POOL_GIVE: u32 = 0x2fc;
/// A pool block's offset and length are multiples of this: 64 KiB.
pub const RGFX_POOL_BLOCK_ALIGN: u64 = 0x1_0000;
/// The longest block a `POOL_TAKE` may ask for: 4 GiB.
pub const RGFX_POOL_TAKE_MAX: u64 = 0x1_0000_0000;
/// `PRIVATE_TAKE` ([`RgfxPrivateTake`]), on ring 0 from the accelerator kext, where the guest took
/// [`RGFX_FEATURE_PRIVATE_BLOCKS`]: context `context` asks for a block of GPU addresses for its
/// Private objects of at least `length` bytes. Answered [`RGFX_EVT_PRIVATE_BLOCK`] with this
/// message's `seqno`: the GPU address of the block's first byte and its length, at least
/// `length` rounded up to [`RGFX_PRIVATE_BLOCK_ALIGN`], the block the context's from then until
/// its `PRIVATE_GIVE` or the context's `CONTEXT_CLOSE`; or [`RGFX_STATUS_INVALID_ARGUMENT`] for a
/// `length` of 0 or past [`RGFX_PRIVATE_TAKE_MAX`], and [`RGFX_STATUS_REFUSED`] where the host has
/// no block or address left for it. A buffer made in the block reads zero from its create, as one
/// made at an address of the context's own range does, wherever in the block it lies and whatever
/// lay there before. The answer may come after later messages' answers.
pub const RGFX_CMD_PRIVATE_TAKE: u32 = 0x2fd;
/// `PRIVATE_GIVE` ([`RgfxPrivateGive`]), on ring 0 from the accelerator kext: context `context`
/// gives back its Private block of `length` bytes at GPU address `gpu_address`. Answered
/// [`RGFX_EVT_COMPLETE`]: [`RGFX_STATUS_REFUSED`] at once where an object the host holds still
/// lies in the block, or [`RGFX_STATUS_NOT_FOUND`] where the context holds no such block;
/// otherwise [`RGFX_STATUS_OK`] once every command buffer the context had committed has finished,
/// and only then does the block go back to the host. Posted after every message the process
/// posted before it gave the block up.
pub const RGFX_CMD_PRIVATE_GIVE: u32 = 0x2fe;
/// A Private block's length is a multiple of this: 64 KiB.
pub const RGFX_PRIVATE_BLOCK_ALIGN: u64 = 0x1_0000;
/// The longest block a `PRIVATE_TAKE` may ask for: 1 TiB.
pub const RGFX_PRIVATE_TAKE_MAX: u64 = 0x100_0000_0000;
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
/// device cache, and the surface's layout may follow that, which places the cache
/// ([`RGFX_SURFACE_PAGES_CACHED`]'s doc).
///
/// Where the guest took [`RGFX_FEATURE_SURFACE_DEFERRED`], a cached block may be deferred
/// ([`RGFX_SURFACE_PAGES_DEFERRED`]): its message has `run_count` 0, and the host makes its device
/// cache, zeros, with no view, and answers [`RGFX_STATUS_OK`], or [`RGFX_STATUS_REFUSED`] where it
/// made no cache, and the kext then names the block's pages as above. The kext keeps the surface's
/// pages unwired meanwhile. Its first CPU lock that needs the pages posts a second `SURFACE_PAGES`
/// of the block, [`RGFX_SURFACE_PAGES_ATTACH`], with the runs, and the host takes them for the
/// cache, which keeps its bytes and the CPU's generation it holds. Where the guest took
/// [`RGFX_FEATURE_SURFACE_DEFERRED_ASK`] instead, the deferred block asks too
/// ([`RGFX_SURFACE_PAGES_ASK_CACHE`], with the layout), and the host answers [`RGFX_STATUS_OK`]
/// only where the block would take a device cache whatever its pages, else
/// [`RGFX_STATUS_REFUSED`], with nothing made; the kext then names its pages asking, as below.
///
/// Where the guest took [`RGFX_FEATURE_SURFACE_CACHE_REFUSED`], a block not cached may ask
/// whether it needs a device cache ([`RGFX_SURFACE_PAGES_ASK_CACHE`]), with the surface's layout
/// after the cache part ([`RgfxSurfacePagesLayout`] at [`RGFX_SURFACE_PAGES_LAYOUT_OFFSET`]). The
/// host maps the pages, and answers [`RGFX_STATUS_OK`], the block in place as any, or
/// [`RGFX_STATUS_CACHE`], with no view kept for the block, where its textures over the pages
/// would not lie in place: the kext then posts the block's `SURFACE_PAGES` again, cached.
///
/// Where the guest took [`RGFX_FEATURE_IMPORT_ADDRESSES`], a pages block's `SURFACE_PAGES`
/// ([`RGFX_SURFACE_PAGES_PRIVATE`]) is answered [`RgfxPagesAnswer`], an [`RGFX_EVT_COMPLETE`] of 32
/// bytes, whose `gpu_address` is the GPU address of the first byte of the block's view where the
/// status is [`RGFX_STATUS_OK`], and 0 otherwise or where the host has no address for it; the
/// address holds until the block's `BLOCK_RELEASE` is carried out, or its context's
/// `CONTEXT_CLOSE`. A reader of the 24 bytes of an [`RgfxComplete`] reads the answer as before.
/// Where it took [`RGFX_FEATURE_IMPORT_ADDRESSES_LATER`] too, such a block is answered once the
/// host has checked and kept its runs, its `gpu_address` 0, and the host gives the view its
/// address after the answer; the sender's later messages that name the block are carried out
/// after that, and a `BUFFER_OVER_PAGES` at `gpu_address` 0 lies at it plus its `offset`.
pub const RGFX_CMD_SURFACE_PAGES: u32 = 0x2f6;
/// `SURFACE_DOWNLOAD` ([`RgfxSurfaceDownload`]), on ring 0 from the accelerator kext, where the
/// device has [`RGFX_FEATURE_SURFACE_CACHE`]: a CPU lock of the IOSurface whose cached block is at
/// `block_offset` follows a GPU write of it, and the command buffers that wrote it have finished.
/// The host copies the surface's device cache into its pages once every command buffer that used
/// the cache before this message was taken has finished, and answers [`RGFX_EVT_COMPLETE`] with
/// [`RGFX_STATUS_OK`] once the pages hold it: at once where no command buffer wrote the cache
/// since it was last copied either way, or the block has no cache.
pub const RGFX_CMD_SURFACE_DOWNLOAD: u32 = 0x2f7;
/// `VT_MESSAGE` ([`RgfxVtMessage`], then its `run_count` [`RgfxPageRun`]s at
/// [`RGFX_VT_MESSAGE_RUNS_OFFSET`]), on ring 0 from the arm64e kext, where the guest took
/// [`RGFX_FEATURE_VT_MESSAGES`]: one message of the guest's VideoToolbox to its host side, a `vtpv`
/// message as VideoToolbox writes it (docs/graphics-kext/design.md, "Decided", 17), its `length`
/// bytes in the guest RAM the runs name, in order, from the first run's first byte. The
/// IOSurfaces the message lends follow its runs, `surface_count` of them, each an
/// [`RgfxVtSurface`] and then its own `run_count` runs, which hold its `length` bytes, or, with
/// [`RGFX_VT_SURFACE_RUN_TABLE`], its run table; `surfaces` names each by its lend, in the same
/// order, the rest 0. The kext holds each lent surface, its
/// pages wired, until the host lets go of it with a `rels` naming its lend (the `VT_MESSAGE`
/// event), and the host writes its pages only meanwhile. No command buffer waits for it, and it
/// waits for none. Answered [`RGFX_EVT_VT_STATUS`] with its `seqno`: the status VideoToolbox's host
/// side returns for the message, or [`RGFX_VT_STATUS_MALFORMED`] where the runs do not hold
/// `length` bytes of guest RAM, `length` is under the 48 bytes of a `vtpv` header or over
/// [`RGFX_VT_MESSAGE_MAX`], the surfaces' records do not read, or a run table does not hold its
/// surface's bytes; the host lets go of the surfaces of a message it does not take.
pub const RGFX_CMD_VT_MESSAGE: u32 = 0x2fa;
/// Where a `VT_MESSAGE`'s runs start: right after its [`RgfxVtMessage`].
pub const RGFX_VT_MESSAGE_RUNS_OFFSET: u32 = 72;
/// The most surfaces a `VT_MESSAGE` lends: VideoToolbox's five.
pub const RGFX_VT_MESSAGE_SURFACES: usize = 5;
/// The longest message a `VT_MESSAGE` carries: 10 MiB, the most the kext's message buffer holds,
/// as Apple's kext of the service sizes its own.
pub const RGFX_VT_MESSAGE_MAX: u64 = 0xa0_0000;
/// [`RgfxVtSurface::run_count`]'s flag for a surface whose runs are more than its `VT_MESSAGE`
/// holds (a 16 KiB command on ring 0): the runs that follow its record, `run_count` without the
/// flag of them, hold its run table instead, [`RgfxPageRun`]s in guest RAM from the table's first
/// byte, which hold the surface's `length` bytes in order, as many as do. The kext keeps the
/// table, wired, while it holds the lend; the host reads it as it takes the message.
pub const RGFX_VT_SURFACE_RUN_TABLE: u32 = 0x8000_0000;
/// The longest message a [`RGFX_EVT_VT_MESSAGE`] carries: 16 KiB, the buffer VideoToolbox's
/// guest side takes each host message into.
pub const RGFX_VT_HOST_MESSAGE_MAX: u32 = 0x4000;
/// [`RgfxVtStatus::status`] for a `VT_MESSAGE` the device could not read: VideoToolbox's
/// `kVTParameterErr`, as the guest's support framework answers a message it cannot send.
pub const RGFX_VT_STATUS_MALFORMED: i32 = -12902;

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
/// Command `SCHEDULED_ASK` ([`RgfxScheduledAsk`], a per-queue submission ring), where the guest
/// took [`RGFX_FEATURE_SCHEDULED_ASK`]: a thread of the queue's process waits for the queue's
/// command buffer `cb_obj_id`, posted on the ring before the ask at `cb_seqno`, to be scheduled
/// (`waitUntilScheduled`). Where the buffer waits for no event, follows no deferred copy of a
/// device cache, is behind no buffer of its queue that waits for one, and found no buffer before
/// it on its queue owing its `CB_SCHEDULED` as it was taken, the host posts the buffer's
/// [`RGFX_EVT_CB_SCHEDULED`] as it takes the ask, whether or not the buffer's work has reached
/// its platform, and commits every command buffer the process sends after the record after that
/// buffer, as after a [`RGFX_EVT_CB_SUBMITTED`]. An ask of any other buffer, of one already
/// scheduled, or of one the host does not hold changes nothing: the buffer is scheduled as
/// without it. The host answers nothing for the ask itself.
pub const RGFX_CMD_SCHEDULED_ASK: u32 = 0x303;

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
/// and [`RgfxCbComplete`]: as soon as the host takes it, before it carries out its ops and before
/// the fills of device caches it uses first that the host makes at once, where it waits for no
/// event, follows no deferred copy of a device cache, and has no buffer before it on its queue
/// that waits for one and is not yet scheduled; else once its waits are satisfied, at its
/// scheduling.
pub const RGFX_EVT_CB_SUBMITTED: u32 = 0x406;
/// Event `ICB_STATE` ([`RgfxIcbState`], on the control completion ring): an `ICB_CREATE`'s
/// answer.
pub const RGFX_EVT_ICB_STATE: u32 = 0x407;
/// Event `SUBMIT_ROOM` (a header alone, its `seqno` 0), where the guest took
/// [`RGFX_FEATURE_SUBMIT_ROOM`]: on a submission ring's completion ring, the host read the ring's
/// `producer_waiting` as 1 after a take that moved its `consumer`, and stored 0 there. A producer
/// waiting for room on the ring looks again.
pub const RGFX_EVT_SUBMIT_ROOM: u32 = 0x408;
/// Event `SHARED_EVENT_VALUE` ([`RgfxSharedEventValue`], on the control completion ring): a
/// `SHARED_EVENT_IMPORT`'s answer.
pub const RGFX_EVT_SHARED_EVENT_VALUE: u32 = 0x409;
/// Event `RATE_MAP_STATE` ([`RgfxRateMapState`], on the control completion ring): a
/// `RATE_MAP_CREATE`'s or `RATE_MAP_RESET`'s answer.
pub const RGFX_EVT_RATE_MAP_STATE: u32 = 0x40a;
/// Event `RATE_MAP_ANSWER` ([`RgfxRateMapAnswer`], on the control completion ring): a
/// `RATE_MAP_QUERY`'s answer.
pub const RGFX_EVT_RATE_MAP_ANSWER: u32 = 0x40b;
/// Event `FUNCTION_POINTER_STATE` ([`RgfxFunctionPointerState`], on the control completion
/// ring): a `FUNCTION_HANDLE_CREATE`'s or `FUNCTION_TABLE_CREATE`'s answer.
pub const RGFX_EVT_FUNCTION_POINTER_STATE: u32 = 0x40c;
/// Event `VT_STATUS` ([`RgfxVtStatus`], on the control completion ring): a `VT_MESSAGE`'s
/// answer, with its `seqno`.
pub const RGFX_EVT_VT_STATUS: u32 = 0x40d;
/// Event `VT_MESSAGE` ([`RgfxVtHostMessage`], then its `length` bytes at
/// [`RGFX_VT_HOST_MESSAGE_BYTES_OFFSET`], on the control completion ring), `seqno` 0, sent only to
/// a guest that took [`RGFX_FEATURE_VT_MESSAGES`]: one message of VideoToolbox's host side to the
/// guest's, of at most [`RGFX_VT_HOST_MESSAGE_MAX`] bytes, a `vtpv` message or a `rels` box. The
/// message's `size` is rounded up to a multiple of 8 after its bytes.
pub const RGFX_EVT_VT_MESSAGE: u32 = 0x40e;
/// Where a `VT_MESSAGE` event's bytes start: right after its [`RgfxVtHostMessage`].
pub const RGFX_VT_HOST_MESSAGE_BYTES_OFFSET: u32 = 24;
/// Event `VT_RING` ([`RgfxVtRing`], on the control completion ring), `seqno` 0, sent only to a
/// guest that took [`RGFX_FEATURE_VT_RING`]: the VideoToolbox ring holds the host's messages
/// through `produced`, the ring's count after its latest. Each is a `VT_MESSAGE` record as the
/// event of that name is ([`RgfxVtHostMessage`], then its bytes, its `size` a multiple of 8),
/// with a pad to the ring's end ([`RGFX_MSG_PAD`]) where the bytes left there are fewer than the
/// record. The guest takes them in order where it takes the event, then stores `produced` to the
/// ring's `consumer`; a `VT_RING` whose count it has taken already takes nothing.
pub const RGFX_EVT_VT_RING: u32 = 0x40f;
/// Event `POOL_BLOCK` ([`RgfxPoolBlock`], on the control completion ring), with a
/// [`RGFX_CMD_POOL_TAKE`]'s `seqno`: its answer. `status` [`RGFX_STATUS_OK`] with the block's
/// `pool_offset` and `length`, and, where the guest took [`RGFX_FEATURE_POOL_ADDRESSES`], its
/// `gpu_address`; otherwise each 0.
pub const RGFX_EVT_POOL_BLOCK: u32 = 0x410;
/// Event `PRIVATE_BLOCK` ([`RgfxPrivateBlock`], on the control completion ring), with a
/// [`RGFX_CMD_PRIVATE_TAKE`]'s `seqno`: its answer. `status` [`RGFX_STATUS_OK`] with the block's
/// `gpu_address` and `length`; otherwise each 0.
pub const RGFX_EVT_PRIVATE_BLOCK: u32 = 0x411;

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

/// `SET_DISPLAY_POWER`: the display's panel is off (`on` 0) or on, as a display's power state
/// turns a monitor's panel off and on (DPMS). While it is off, the display shows black, with no
/// cursor, whatever its scanout or presented surface; everything else goes on as while it is on:
/// the guest's `SET_SCANOUT`, `PRESENT_SURFACE` (answered as the display shows the surface, only
/// not lit), cursor commands and `VBLANK`s. Turned on, it shows its scanout or presented surface,
/// and the cursor, again. A reset turns it on.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSetDisplayPower {
    /// [`RGFX_CMD_SET_DISPLAY_POWER`].
    pub header: RgfxMsgHeader,
    /// The display's id.
    pub display: u32,
    /// Whether the panel is on: 0 turns it off.
    pub on: u32,
}

/// `DISPLAY_BOUNDS`: display `display` lies at (`min_x`, `min_y`) up to (`max_x`, `max_y`), the
/// corner past it, of the guest's desktop: WindowServer's global space, in points, where the
/// guest arranged its displays (the x86-64 kext's framebuffer's `getBoundingRect`). The guest
/// sends each display's whenever it changes, and the host keeps the latest until a reset. The
/// host's captured pointer, held at an edge of a display's window where another display's
/// rectangle meets that display's, goes on onto that display, as the guest's cursor does on a
/// Mac (docs/graphics-kext/design.md, "Multiple displays"). A rectangle of no area, or one of a
/// display the guest does not see, is dropped.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxDisplayBounds {
    /// [`RGFX_CMD_DISPLAY_BOUNDS`].
    pub header: RgfxMsgHeader,
    /// The display's id.
    pub display: u32,
    /// The rectangle's left edge, in the desktop's points.
    pub min_x: i32,
    /// Its top edge.
    pub min_y: i32,
    /// Its right edge, past its last column.
    pub max_x: i32,
    /// Its bottom edge, past its last row.
    pub max_y: i32,
    /// Keeps the size a multiple of 8; 0.
    pub reserved_24: u32,
}

/// [`RgfxCursorCurve::form`]: `y = x`, no curve; its parameters are not read.
pub const RGFX_CURVE_BYPASS: u32 = 0;
/// [`RgfxCursorCurve::form`]: `y = x^g`.
pub const RGFX_CURVE_TYPE0: u32 = 1;
/// [`RgfxCursorCurve::form`]: `y = (a x + b)^g` where `a x + b >= 0`, else 0.
pub const RGFX_CURVE_TYPE1: u32 = 2;
/// [`RgfxCursorCurve::form`]: `y = (a x + b)^g + c` where `a x + b >= 0`, else `c`.
pub const RGFX_CURVE_TYPE2: u32 = 3;
/// [`RgfxCursorCurve::form`]: `y = (a x + b)^g` where `x >= d`, else `c x`.
pub const RGFX_CURVE_TYPE3: u32 = 4;
/// [`RgfxCursorCurve::form`]: `y = (a x + b)^g + e` where `x >= d`, else `c x + f`.
pub const RGFX_CURVE_TYPE4: u32 = 5;

/// One channel's curve of a [`RgfxCursorColor`]: one of the ICC's parametric curves, in the
/// forms and with the numbers an arm64 WindowServer's ICC curve blocks give them (QuartzCore's
/// `Bypass` and `Type0` to `Type4`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxCursorCurve {
    /// An `RGFX_CURVE_*` form.
    pub form: u32,
    /// The parameters `g`, `a`, `b`, `c`, `d`, `e` and `f`, `f32`s as bits; those the form does
    /// not use are not read.
    pub params: [u32; 7],
}

/// `CURSOR_COLOR`: display `display`'s cursor converts its pixels through the chain that follows
/// where `on` is not 0, and shows them as they are where it is 0, its state from the device's
/// start. Each pixel's straight colour, its red, green and blue in [0, 1], goes through
/// `source_curves` channel by channel, then `to_xyz` and `from_xyz` (each three rows of three
/// `f32`s, as bits: output channel `i` is row `i` times the input), then `blend_curves` channel
/// by channel, and is clamped to [0, 1]; its alpha is kept. The host converts the image the
/// guest's last `SET_CURSOR` set, and each one it sets after, until the next `CURSOR_COLOR` of the
/// display; a reset turns it off. A curve of a form the host does not know, a parameter or entry
/// that is not finite, or a display the guest does not see, drops the message, and the cursor
/// keeps its chain. Not answered.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxCursorColor {
    /// [`RGFX_CMD_CURSOR_COLOR`].
    pub header: RgfxMsgHeader,
    /// The display's id.
    pub display: u32,
    /// Whether the chain applies: 0 shows the cursor's pixels as they are.
    pub on: u32,
    /// The source's curves, red, green and blue: WindowServer's `Src2XYZ` curve.
    pub source_curves: [RgfxCursorCurve; 3],
    /// The source's matrix to XYZ, row by row: WindowServer's `Src2XYZ` matrix.
    pub to_xyz: [u32; 9],
    /// The matrix from XYZ to the blend space, row by row: WindowServer's `XYZ2Blend` matrix.
    pub from_xyz: [u32; 9],
    /// The blend space's curves, red, green and blue: WindowServer's `XYZ2Blend` curve.
    pub blend_curves: [RgfxCursorCurve; 3],
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
    /// [`RGFX_PRESENT_KEEP`], where the guest took [`RGFX_FEATURE_DISPLAY_COLOR`]; 0 otherwise.
    /// A present with any other bit is refused [`RGFX_STATUS_INVALID_ARGUMENT`].
    pub flags: u32,
}

/// [`RgfxPresentSurface::flags`]: the display keeps what it shows, a surface or the scanout, and
/// `iosurface_id` is 0. The present changes only the colour the `DISPLAY_COLOR`s before it staged,
/// and is answered as any present is, once the display shows it. One with a nonzero
/// `iosurface_id` is refused [`RGFX_STATUS_INVALID_ARGUMENT`].
pub const RGFX_PRESENT_KEEP: u32 = 1 << 0;

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

/// [`RgfxDisplayColor::stage`]: plane 0's gamma table, the display pipe's
/// `IOAccelDisplayPlaneGammaTable`, the first stage.
pub const RGFX_COLOR_PLANE_GAMMA: u32 = 0;
/// [`RgfxDisplayColor::stage`]: the pipe's gamma table before its colour matrix
/// (`TransactionSetPipePreGammaTable`), the second stage.
pub const RGFX_COLOR_PRE_GAMMA: u32 = 1;
/// [`RgfxDisplayColor::stage`]: the pipe's colour matrix (`TransactionSetPipeColorMatrix`), the
/// third stage: three entries, the transaction's twelve floats as it carries them, a row each:
/// output channel `i` is `entry[i][0] * red + entry[i][1] * green + entry[i][2] * blue +
/// entry[i][3]`.
pub const RGFX_COLOR_MATRIX: u32 = 2;
/// [`RgfxDisplayColor::stage`]: the pipe's gamma table after its colour matrix
/// (`TransactionSetPipePostGammaTable`), the last stage.
pub const RGFX_COLOR_POST_GAMMA: u32 = 3;
/// The most entries one `DISPLAY_COLOR` message carries.
pub const RGFX_COLOR_PART_ENTRIES: u32 = 256;
/// The most entries a stage's table holds: the display pipe's own limit.
pub const RGFX_COLOR_MAX_ENTRIES: u32 = 0x20000;

/// `DISPLAY_COLOR`: part of one colour stage of display `display`, set by display pipe
/// transaction `transaction_id`: the `part_entries` [`RgfxColorEntry`]s that follow the message's
/// fields, entries `first_entry` on of the stage's `entry_count`. A stage's parts come in order,
/// the first at entry 0; a stage of 0 entries is off (it passes its input through). The host
/// stages each stage whole, and the next `PRESENT_SURFACE` it accepts sets the stages staged
/// since the last one, as the display shows that present (a refused one leaves them staged): a
/// stage no `DISPLAY_COLOR` sets keeps its table. A gamma stage's entry `k` of `n` is the output
/// at input `k / (n - 1)`, its `x`, `y` and `z` red, green and blue; the host interpolates
/// between entries, and clamps its input to [0, 1]. The stages apply in their order, after the
/// surface's own colour: plane gamma, pre-gamma, matrix, post-gamma. A part out of order, past
/// `RGFX_COLOR_MAX_ENTRIES` or of more than `RGFX_COLOR_PART_ENTRIES` entries, or a matrix of
/// other than 3 or 0 entries, drops its stage's staged parts, and the stage keeps its table. A
/// present that comes before all of a stage's parts drops them too. Not answered.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxDisplayColor {
    /// [`RGFX_CMD_DISPLAY_COLOR`].
    pub header: RgfxMsgHeader,
    /// The display's id.
    pub display: u32,
    /// The display pipe's transaction ID, for the host's log.
    pub transaction_id: u32,
    /// An `RGFX_COLOR_*` stage.
    pub stage: u32,
    /// The stage's entries in all.
    pub entry_count: u32,
    /// This part's first entry.
    pub first_entry: u32,
    /// The entries this part carries, after the message's fields.
    pub part_entries: u32,
}

/// One entry of a [`RgfxDisplayColor`] stage: four `f32`s, as bits.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxColorEntry {
    /// The first `f32`, as bits: a gamma entry's red.
    pub x: u32,
    /// The second, as bits: green.
    pub y: u32,
    /// The third, as bits: blue.
    pub z: u32,
    /// The fourth, as bits: unused by a gamma entry.
    pub w: u32,
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

/// A pages block's `SURFACE_PAGES` answer where the guest took
/// [`RGFX_FEATURE_IMPORT_ADDRESSES`]: an [`RgfxComplete`] and the GPU address of the block's
/// view's first byte. Also a `PAGES_BUFFER_ADDRESS`'s answer
/// ([`RGFX_CMD_PAGES_BUFFER_ADDRESS`]), with the buffer's address.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxPagesAnswer {
    /// [`RGFX_EVT_COMPLETE`], with the `SURFACE_PAGES`' or `PAGES_BUFFER_ADDRESS`' `seqno`.
    pub header: RgfxMsgHeader,
    /// An `RGFX_STATUS_*`, as an [`RgfxComplete`]'s.
    pub status: u32,
    /// Zero.
    pub reserved_14: u32,
    /// The GPU address of the first byte of the block's view, never 0 where given; 0 unless
    /// the status is [`RGFX_STATUS_OK`], where the host gives the block none, and where the
    /// guest took [`RGFX_FEATURE_IMPORT_ADDRESSES_LATER`], whose block's view is given its
    /// address after the answer. A `PAGES_BUFFER_ADDRESS`' answer: the buffer's GPU address, 0
    /// unless the status is [`RGFX_STATUS_OK`].
    pub gpu_address: u64,
}

/// `DISPLAYS`: `GET_DISPLAYS`'s answer. `count` [`RgfxDisplayEntry`] records follow it, at
/// [`RGFX_DISPLAYS_ENTRIES_OFFSET`], each followed by its [`RgfxDisplayScale`] where the guest
/// took [`RGFX_FEATURE_DISPLAY_SCALE`], then by its `mode_count` [`RgfxModeEntry`] records; the
/// message's `size` is rounded up to a multiple of 8 after the last. The struct is the message's
/// start: C's `sizeof` pads it past `count`, where the first record already is.
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

/// The scale of the host screen that shows a display, after its [`RgfxDisplayEntry`] in a
/// `DISPLAYS` message where the guest took [`RGFX_FEATURE_DISPLAY_SCALE`]: how many of the
/// screen's pixels show one of its points, which is what the person at the host sees the
/// guest's pixels at.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxDisplayScale {
    /// The screen's pixels a point, in 120ths: 120 on a screen at 1x, 240 at 2x, 252 at 2.1; 0
    /// where no screen shows the display (a display with no window).
    pub scale_120ths: u32,
    /// Zero.
    pub reserved_4: u32,
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

/// `POINTER_OVER`: the host's pointer is over display `display`, at (`x`, `y`) in the display's
/// pixels of its mode, and the guest's cursor shows on another display. The guest moves its
/// cursor onto display `display` at that point, so the host's pointer positions that follow, the
/// USB tablet's scaled over that display's pixels, land on it: macOS scales the tablet over the
/// display its cursor is on (docs/graphics-kext/design.md, "Multiple displays", X1). The host
/// tells the guest once for each display the pointer comes over while the cursor is elsewhere,
/// and again only after the cursor has shown on another display; only a guest that took
/// [`RGFX_FEATURE_POINTER_OVER`] is told. `buttons` are the pointer's buttons held, so the move
/// the guest makes is a drag where one is held.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxPointerOver {
    /// [`RGFX_EVT_POINTER_OVER`], `seqno` 0.
    pub header: RgfxMsgHeader,
    /// The display's id.
    pub display: u32,
    /// The pointer's column, 0 up to the mode's width.
    pub x: i32,
    /// The pointer's row, 0 up to the mode's height.
    pub y: i32,
    /// The buttons held: `RGFX_POINTER_BUTTON_*` bits.
    pub buttons: u32,
}

/// `POINTER_MOTION`: the host's pointing device moved by (`dx`, `dy`), right and down, in its own
/// counts, as no acceleration has changed them, turned its wheel `wheel` notches up (away from the
/// person), and holds the `RGFX_POINTER_BUTTON_*` `buttons` after it. The guest's driver hands it
/// to its relative pointing device as one report of a USB mouse, so the guest accelerates the
/// motion as its own settings say, once, and a game that reads the device's counts gets the
/// host's. Only a guest that took [`RGFX_FEATURE_POINTER_MOTION`] is sent it; motion that finds
/// the ring held or full is added to the next event of the same buttons, so a press or a release
/// is never lost.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxPointerMotion {
    /// [`RGFX_EVT_POINTER_MOTION`], `seqno` 0.
    pub header: RgfxMsgHeader,
    /// The motion to the right, in the device's counts.
    pub dx: i32,
    /// The motion down, in the device's counts.
    pub dy: i32,
    /// Wheel notches, above 0 up (away from the person).
    pub wheel: i32,
    /// The buttons held after it: `RGFX_POINTER_BUTTON_*` bits.
    pub buttons: u32,
}

/// [`RgfxPointerOver`]'s `buttons`: the left button held.
pub const RGFX_POINTER_BUTTON_LEFT: u32 = 1 << 0;
/// [`RgfxPointerOver`]'s `buttons`: the right button held.
pub const RGFX_POINTER_BUTTON_RIGHT: u32 = 1 << 1;
/// [`RgfxPointerOver`]'s `buttons`: the middle button held.
pub const RGFX_POINTER_BUTTON_MIDDLE: u32 = 1 << 2;
/// [`RgfxPointerOver`]'s `buttons`: the side button held (back).
pub const RGFX_POINTER_BUTTON_SIDE: u32 = 1 << 3;
/// [`RgfxPointerOver`]'s `buttons`: the extra button held (forward).
pub const RGFX_POINTER_BUTTON_EXTRA: u32 = 1 << 4;

// The layout the tables of the design give.
const _: () = {
    use core::mem::{offset_of, size_of};
    assert!(size_of::<RgfxRingHeader>() == RGFX_RING_HEADER_SIZE as usize);
    assert!(offset_of!(RgfxRingHeader, consumer) == 0x40);
    assert!(offset_of!(RgfxRingHeader, consumer_idle) == 0x80);
    assert!(offset_of!(RgfxRingHeader, armed) == 0xc0);
    assert!(offset_of!(RgfxRingHeader, producer_waiting) == 0x10);
    assert!(offset_of!(RgfxRingHeader, guest_clock_offset_ns) == 0x18);
    assert!(offset_of!(RgfxRingHeader, guest_clock_bound_ns) == 0x20);
    assert!(offset_of!(RgfxRingHeader, raise_all) == 0xc4);
    assert!(size_of::<RgfxMsgHeader>() == 16);
    assert!(size_of::<RgfxPoolTake>() == 32);
    assert!(size_of::<RgfxPoolGive>() == 40);
    assert!(size_of::<RgfxPoolBlock>() == 48);
    assert!(size_of::<RgfxVtMessage>() == RGFX_VT_MESSAGE_RUNS_OFFSET as usize);
    assert!(size_of::<RgfxVtSurface>() == 88);
    assert!(size_of::<RgfxVtSurfacePlane>() == 24);
    assert!(size_of::<RgfxVtStatus>() == 24);
    assert!(size_of::<RgfxVtHostMessage>() == RGFX_VT_HOST_MESSAGE_BYTES_OFFSET as usize);
    assert!(size_of::<RgfxVtRing>() == 24);
    // The VideoToolbox ring lies past the video caps and the control completion ring's span, and
    // before the guest's allocations.
    assert!(
        RGFX_VIDEO_CAPS_OFFSET + size_of::<RgfxVideoCaps>() as u64
            <= RGFX_CONTROL_COMPLETE_OFFSET + RGFX_CONTROL_RING_SPAN
    );
    assert!(RGFX_VT_RING_OFFSET >= RGFX_CONTROL_COMPLETE_OFFSET + RGFX_CONTROL_RING_SPAN);
    assert!(
        RGFX_VT_RING_OFFSET + RGFX_RING_HEADER_SIZE as u64 + RGFX_VT_RING_DATA_SIZE as u64
            <= RGFX_ALLOCATIONS_OFFSET
    );
    assert!(RGFX_VT_RING_DATA_SIZE.is_power_of_two());
    assert!(size_of::<RgfxGetDisplays>() == 16);
    assert!(size_of::<RgfxSetScanout>() == 40);
    assert!(size_of::<RgfxSetCursor>() == 48);
    assert!(size_of::<RgfxMoveCursor>() == 32);
    assert!(size_of::<RgfxSetVblank>() == 24);
    assert!(size_of::<RgfxSetDisplayPower>() == 24);
    assert!(size_of::<RgfxPresentSurface>() == 32);
    assert!(size_of::<RgfxDisplayColor>() == 40);
    assert!(size_of::<RgfxColorEntry>() == 16);
    assert!(size_of::<RgfxComplete>() == 24);
    assert!(offset_of!(RgfxDisplays, count) + 4 == RGFX_DISPLAYS_ENTRIES_OFFSET as usize);
    assert!(size_of::<RgfxDisplayEntry>() == 16);
    assert!(size_of::<RgfxDisplayScale>() == 8);
    assert!(size_of::<RgfxModeEntry>() == 12);
    assert!(size_of::<RgfxDisplayChanged>() == 24);
    assert!(size_of::<RgfxVblank>() == 40);
    assert!(size_of::<RgfxPointerOver>() == 32);
    assert!(size_of::<RgfxPointerMotion>() == 32);
    assert!(offset_of!(RgfxPointerMotion, wheel) == 24);
    assert!(offset_of!(RgfxPointerMotion, buttons) == 28);
    assert!(size_of::<RgfxDisplayBounds>() == 40);
    assert!(RGFX_CONTROL_RING_DATA_SIZE.is_power_of_two());
    assert!(
        RGFX_RING_HEADER_SIZE as u64 + RGFX_CONTROL_RING_DATA_SIZE as u64 <= RGFX_CONTROL_RING_SPAN
    );
    assert!(RGFX_CONTROL_COMPLETE_OFFSET + RGFX_CONTROL_RING_SPAN <= RGFX_ALLOCATIONS_OFFSET);
    assert!(RGFX_ALLOCATIONS_OFFSET.is_multiple_of(0x1000));
    assert!(RGFX_CONTEXT_ALLOCATIONS_OFFSET.is_multiple_of(0x1000));
    assert!(size_of::<RgfxContextClose>() == 24);
    assert!(size_of::<RgfxResourceIdChunk>() == 24);
    assert!(size_of::<RgfxPipelineAnswersQuery>() == 24);
    assert!(size_of::<RgfxCommandBufferInRegion>() == 32);
    assert!(size_of::<RgfxBlockRelease>() == 40);
    assert!(size_of::<RgfxBlockGrant>() == 40);
    assert!(size_of::<RgfxPageRun>() == 16);
    assert!(size_of::<RgfxSurfacePages>() == 56);
    assert!(size_of::<RgfxSurfacePages>() == RGFX_SURFACE_PAGES_CACHE_OFFSET as usize);
    assert!(size_of::<RgfxSurfacePagesCache>() == 8);
    assert!(
        RGFX_SURFACE_PAGES_CACHE_OFFSET as usize + size_of::<RgfxSurfacePagesCache>()
            == RGFX_SURFACE_PAGES_LAYOUT_OFFSET as usize
    );
    assert!(size_of::<RgfxSurfacePlaneLayout>() == 32);
    assert!(size_of::<RgfxSurfacePagesLayout>() == 136);
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
    // The small chunks are the same pool, four to a chunk.
    assert!(
        RGFX_RESOURCE_ID_SMALL_CHUNKS as u64 * RGFX_RESOURCE_ID_SMALL_CHUNK
            == RGFX_RESOURCE_ID_CHUNKS as u64 * RGFX_RESOURCE_ID_CHUNK
    );
    assert!(RGFX_RESOURCE_ID_CHUNK.is_multiple_of(RGFX_RESOURCE_ID_SMALL_CHUNK));
    // The units fill the guest's GPU addresses, which end at the Metal address space's end,
    // 2^47, and start at 2^46 or above, which the host's own ranges never reach
    // (RGFX_CONTEXT_VA_BASE).
    assert!(
        RGFX_CONTEXT_VA_BASE + RGFX_CONTEXT_VA_UNITS as u64 * RGFX_CONTEXT_VA_UNIT
            == RGFX_CONTEXT_VA_END
    );
    assert!(RGFX_CONTEXT_VA_END <= 1u64 << 47);
    assert!(RGFX_CONTEXT_VA_BASE >= 1u64 << 46);
    // A context of the old span holds a whole number of units, a power of two of them, so it is
    // an aligned run of them.
    assert!(RGFX_CONTEXT_VA_SPAN.is_multiple_of(RGFX_CONTEXT_VA_UNIT));
    assert!((RGFX_CONTEXT_VA_SPAN / RGFX_CONTEXT_VA_UNIT).is_power_of_two());
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

/// `SHARED_EVENT_NAME`: the sender's shared event `obj_id` named `name` device-wide. Every holder of
/// the event (its maker, and each process that made an event from the name) is posted the values
/// the others' command buffers signal it to and their CPUs set it to, as [`RgfxEventSignalled`]
/// for its own id, on the completion ring of its queue `listen_queue_obj_id`, as the host learns
/// of them: a signal where the host posts it to the signaller, a CPU set as the host carries it
/// out. The kext builds the message: `name` is the kext's, never the process's.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSharedEventName {
    /// [`RGFX_CMD_SHARED_EVENT_NAME`].
    pub header: RgfxMsgHeader,
    /// The sender's shared event.
    pub obj_id: u64,
    /// Its name, unique to the device from the kext's start: never 0.
    pub name: u64,
    /// The sender's queue whose completion ring takes the values other holders give the event.
    pub listen_queue_obj_id: u64,
}

/// `SHARED_EVENT_IMPORT`: the sender's new shared event `obj_id` is the host event named `name`, its
/// values posted on the completion ring of the sender's queue `listen_queue_obj_id`, as
/// [`RgfxSharedEventName`]'s are. The kext builds the message from the port the process gave it.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSharedEventImport {
    /// [`RGFX_CMD_SHARED_EVENT_IMPORT`].
    pub header: RgfxMsgHeader,
    /// The sender's new shared event's guest-picked id.
    pub obj_id: u64,
    /// The name the event was given ([`RgfxSharedEventName::name`]).
    pub name: u64,
    /// The sender's queue whose completion ring takes the values other holders give the event.
    pub listen_queue_obj_id: u64,
}

/// `SHARED_EVENT_UNNAME`: the name `name` is no longer the kext's to give out.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSharedEventUnname {
    /// [`RGFX_CMD_SHARED_EVENT_UNNAME`].
    pub header: RgfxMsgHeader,
    /// 0: the message is context 0's, which no process is, so it waits for no process's
    /// messages.
    pub reserved_10: u64,
    /// The name ([`RgfxSharedEventName::name`]).
    pub name: u64,
}

/// `SHARED_EVENT_VALUE`: a `SHARED_EVENT_IMPORT`'s answer, with its `seqno`: the event's value as
/// the host's event has it, the value the new event starts at.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSharedEventValue {
    /// [`RGFX_EVT_SHARED_EVENT_VALUE`], with the import's `seqno`.
    pub header: RgfxMsgHeader,
    /// An `RGFX_STATUS_*`: [`RGFX_STATUS_OK`] where the event is made.
    pub status: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_14: u32,
    /// The event's value; 0 where it was not made.
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

/// `SCHEDULED_ASK`: a thread waits for the queue's command buffer `cb_obj_id`, posted on the ring
/// at `cb_seqno`, to be scheduled. Its header's `seqno` is the ring's.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxScheduledAsk {
    /// [`RGFX_CMD_SCHEDULED_ASK`].
    pub header: RgfxMsgHeader,
    /// The command buffer's guest-picked id.
    pub cb_obj_id: u64,
    /// The seqno of the buffer's `COMMAND_BUFFER` or `COMMAND_BUFFER_IN_REGION` on the ring.
    pub cb_seqno: u64,
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
    /// up; of a `RESOURCE_ID_SMALL_CHUNK`, below [`RGFX_RESOURCE_ID_SMALL_CHUNKS`]: ids
    /// `1 + chunk * RGFX_RESOURCE_ID_SMALL_CHUNK` and up.
    pub chunk: u32,
}

/// `PIPELINE_ANSWERS`: the kept answers the kext asks for, from the `first`th, at most `most`
/// ([`RGFX_CMD_PIPELINE_ANSWERS`]).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxPipelineAnswersQuery {
    /// [`RGFX_CMD_PIPELINE_ANSWERS`].
    pub header: RgfxMsgHeader,
    /// The first answer wanted, counted from the newest.
    pub first: u32,
    /// The most answers wanted.
    pub most: u32,
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

/// [`RgfxBlockRelease::flags`]: the block's IOSurface had no client left, no process holding it,
/// as the release was posted. The host copies the block's device cache into the pages all the
/// same, and only counts it: a surface with no client can still be looked up again, from a mach
/// port or an XPC object that holds it (how one process hands a surface it drew to another), and
/// the process that looks it up reads the pages.
pub const RGFX_BLOCK_RELEASE_NO_CLIENT: u32 = 1 << 0;

/// [`RgfxBlockRelease::flags`]: besides having no client, the block's IOSurface was held by
/// nothing but the kext as the release was posted (no mach port or XPC object): no process holds
/// it to read its pages. The host drops a device cache a command buffer wrote without
/// copying it into the pages, where the block's header still counts no client of the surface as
/// the host makes its copy: the kext sets the count past 0 where a process takes the surface
/// again before the answer, and the host then copies. The kext holds the pages wired until the
/// answer all the same. A host that does not know the bit copies, as for any release.
pub const RGFX_BLOCK_RELEASE_UNHELD: u32 = 1 << 1;

/// [`RgfxPageRun::address`] of a run of a carrying deferral's surface
/// ([`RGFX_FEATURE_SURFACE_DEFERRED_CARRY`]) whose pages are not resident: the cache holds its
/// make's zeros there. No page's address: it is no multiple of the guest's 4 KiB page.
pub const RGFX_PAGE_RUN_ZEROS: u64 = u64::MAX;

/// A run of an IOSurface's guest RAM, which `SURFACE_PAGES` lists: `length` bytes at guest
/// physical address `address`, both multiples of the guest's 4 KiB page (but a carrying
/// deferral's [`RGFX_PAGE_RUN_ZEROS`]).
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
/// that follows a GPU write. Without the feature the host gives the block none. Where the guest
/// took no [`RGFX_FEATURE_SURFACE_CACHE_REFUSED`], the message may carry the surface's layout, as
/// an ask does ([`RgfxSurfacePagesLayout`] at [`RGFX_SURFACE_PAGES_LAYOUT_OFFSET`]), a deferred
/// block's ([`RGFX_SURFACE_PAGES_DEFERRED`]) among them: the host places the cache by it alone, its
/// surface's first plane's first byte where the host's GPU's textures over the cache may start in
/// place (as CoreVideo lays a frame's planes 32 bytes into their surface), and answers the
/// message as one with no layout; a message too short for it carries none, and a layout whose
/// planes do not lie in the surface's bytes places nothing. Its copies between the cache and the
/// pages move the surface's bytes as without it.
pub const RGFX_SURFACE_PAGES_CACHED: u32 = 1 << 0;
/// [`RgfxSurfacePagesCache::flags`], where the guest took [`RGFX_FEATURE_PAGES_BUFFERS`]: the
/// block is a process's pages block, for its `BUFFER_OVER_PAGES` buffers, not an IOSurface's, and
/// its view stays out of the host's surface-view cache. The block holds the runs alone, of pages
/// the kext wired for the block's context, and the host takes no device cache for it.
pub const RGFX_SURFACE_PAGES_PRIVATE: u32 = 1 << 1;
/// [`RgfxSurfacePagesCache::flags`], with [`RGFX_SURFACE_PAGES_CACHED`], where the guest took
/// [`RGFX_FEATURE_SURFACE_DEFERRED`], or [`RGFX_FEATURE_SURFACE_DEFERRED_ASK`] and then with
/// [`RGFX_SURFACE_PAGES_ASK_CACHE`]: the block's surface has no pages yet (a fresh surface no CPU
/// lock and no other device has touched), and its contents live in the device cache alone, as a
/// GPU of its own memory keeps a surface only the GPU uses. The message has `run_count` 0, or,
/// where the guest took [`RGFX_FEATURE_SURFACE_DEFERRED_CARRY`], runs of the surface's bytes in
/// order that carry its resident pages, the others [`RGFX_PAGE_RUN_ZEROS`]; the host makes the
/// cache, fills it with zeros before any use, holds the CPU's generation 0 (or, carrying, owes
/// the carried pages' fill, as a new cache owes its first), and makes no view and watches no
/// pages but the carried ones. Its `BLOCK_RELEASE` copies nothing, there being no pages
/// (the kext names them first where the surface still lives). Answered [`RGFX_STATUS_REFUSED`]
/// where the host made no cache (its GPU's memory, or its budget, is spent), and, asking, where
/// the surface's textures might lie in place over its pages. One with `ASK_CACHE` where the host
/// does not offer `SURFACE_DEFERRED_ASK`, or without it where it does, is
/// [`RGFX_STATUS_INVALID_ARGUMENT`], and so is one with runs where the host does not offer
/// `SURFACE_DEFERRED_CARRY`, or whose runs are not whole pages summing to the surface's bytes, or
/// carry none.
pub const RGFX_SURFACE_PAGES_DEFERRED: u32 = 1 << 2;
/// [`RgfxSurfacePagesCache::flags`], with [`RGFX_SURFACE_PAGES_CACHED`]: the pages of a deferred
/// block ([`RGFX_SURFACE_PAGES_DEFERRED`]), which the kext wired at the surface's first CPU lock
/// that needs them, in `run_count` runs as for any block. The host takes them for the block's
/// cache, which keeps its bytes, the CPU's generation it holds and whether a buffer wrote it since,
/// so the `SURFACE_DOWNLOAD` that follows copies what the GPU wrote. Answered
/// [`RGFX_STATUS_INVALID_ARGUMENT`] for a block with no deferred cache.
pub const RGFX_SURFACE_PAGES_ATTACH: u32 = 1 << 3;
/// [`RgfxSurfacePagesCache::flags`], without [`RGFX_SURFACE_PAGES_CACHED`], where the guest took
/// [`RGFX_FEATURE_SURFACE_CACHE_REFUSED`]: the kext asks whether the block's surface needs a
/// device cache. The message carries the surface's layout ([`RgfxSurfacePagesLayout`] at
/// [`RGFX_SURFACE_PAGES_LAYOUT_OFFSET`]); a message too short for it is
/// [`RGFX_STATUS_INVALID_ARGUMENT`]. Answered [`RGFX_STATUS_OK`], the block in place, or
/// [`RGFX_STATUS_CACHE`] (the command's doc). With [`RGFX_SURFACE_PAGES_CACHED`] and
/// [`RGFX_SURFACE_PAGES_DEFERRED`], where the guest took [`RGFX_FEATURE_SURFACE_DEFERRED_ASK`], a
/// fresh surface's block asks to be deferred, carrying the layout the same way.
pub const RGFX_SURFACE_PAGES_ASK_CACHE: u32 = 1 << 4;
/// [`RgfxSurfacePagesCache::flags`], with [`RGFX_SURFACE_PAGES_PRIVATE`], where the guest took
/// [`RGFX_FEATURE_READ_ONLY_PAGES`]: the pages block's pages are ones its process maps read-only,
/// which the kext wired for reading alone. Their memory may be a file's, or another process's
/// that it writes; the guest's RAM under them is still the host's to map. Every object over the
/// block must leave its pages as they are: a `BUFFER_OVER_PAGES` over it that does not carry
/// `MTLResourceOptions` bit 24 is refused, as is a `HEAP_OVER_PAGES` over it, an
/// `ICB_SLOTS_IN_PAGES` that names it, and a `MANAGED_BUFFER_CREATE` or `MANAGED_TEXTURE_CREATE`
/// whose CPU copy lies over it. One without [`RGFX_SURFACE_PAGES_PRIVATE`] is
/// [`RGFX_STATUS_INVALID_ARGUMENT`].
pub const RGFX_SURFACE_PAGES_READ_ONLY: u32 = 1 << 5;

/// Where a `SURFACE_PAGES` message's [`RgfxSurfacePagesLayout`] starts: right after its
/// [`RgfxSurfacePagesCache`].
pub const RGFX_SURFACE_PAGES_LAYOUT_OFFSET: u32 = 64;

/// The most planes an [`RgfxSurfacePagesLayout`] holds.
pub const RGFX_SURFACE_PAGES_PLANES: u32 = 4;

/// One plane of an IOSurface as its pages hold it ([`RgfxSurfacePagesLayout`]), from the
/// surface's first byte.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSurfacePlaneLayout {
    /// The plane's first byte, from the surface's.
    pub offset: u64,
    /// Bytes from one of the plane's rows to the next (`IOSurfaceGetBytesPerRowOfPlane`).
    pub bytes_per_row: u64,
    /// The bytes of a row's texels: its elements' (`IOSurfaceGetWidthOfPlane` over
    /// `IOSurfaceGetElementWidthOfPlane`, rounded up, times `IOSurfaceGetBytesPerElementOfPlane`).
    pub row_bytes: u64,
    /// The plane's rows of elements (`IOSurfaceGetHeightOfPlane` over
    /// `IOSurfaceGetElementHeightOfPlane`, rounded up).
    pub rows: u64,
}

/// The layout of an IOSurface whose block asks whether it needs a device cache
/// ([`RGFX_SURFACE_PAGES_ASK_CACHE`]), after the [`RgfxSurfacePagesCache`] at
/// [`RGFX_SURFACE_PAGES_LAYOUT_OFFSET`]: its planes, as every texture over it lies over them.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSurfacePagesLayout {
    /// The planes `planes` holds, 1 to [`RGFX_SURFACE_PAGES_PLANES`]: 1 for a surface without
    /// planes.
    pub plane_count: u32,
    /// Zero.
    pub reserved_04: u32,
    /// The planes in plane order; those past `plane_count` zero.
    pub planes: [RgfxSurfacePlaneLayout; RGFX_SURFACE_PAGES_PLANES as usize],
}

/// What a `SURFACE_PAGES` says of its block's device cache, after the [`RgfxSurfacePages`] at
/// [`RGFX_SURFACE_PAGES_CACHE_OFFSET`].
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxSurfacePagesCache {
    /// `RGFX_SURFACE_PAGES_*`: [`RGFX_SURFACE_PAGES_CACHED`], [`RGFX_SURFACE_PAGES_PRIVATE`],
    /// [`RGFX_SURFACE_PAGES_DEFERRED`], [`RGFX_SURFACE_PAGES_ATTACH`],
    /// [`RGFX_SURFACE_PAGES_ASK_CACHE`] and [`RGFX_SURFACE_PAGES_READ_ONLY`].
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

/// `VT_MESSAGE`: one message of the guest's VideoToolbox, `length` bytes in the guest RAM of the
/// `run_count` [`RgfxPageRun`]s that follow, lending the `surface_count` IOSurfaces whose
/// [`RgfxVtSurface`]s follow the runs.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVtMessage {
    /// [`RGFX_CMD_VT_MESSAGE`].
    pub header: RgfxMsgHeader,
    /// The message's bytes.
    pub length: u64,
    /// How many runs follow, at [`RGFX_VT_MESSAGE_RUNS_OFFSET`].
    pub run_count: u32,
    /// How many surfaces the message lends, at most [`RGFX_VT_MESSAGE_SURFACES`].
    pub surface_count: u32,
    /// The lends of the surfaces the message lends ([`RgfxVtSurface::lend`]), in their records'
    /// order, the rest 0.
    pub surfaces: [u64; RGFX_VT_MESSAGE_SURFACES],
}

/// The planes an [`RgfxVtSurface`] describes: a 4:2:0 picture's two.
pub const RGFX_VT_SURFACE_PLANES: usize = 2;

/// One IOSurface a `VT_MESSAGE` lends, as the kext reads it of the surface: then its `run_count`
/// [`RgfxPageRun`]s, which hold its `length` bytes from its first, or those of its run table
/// ([`RGFX_VT_SURFACE_RUN_TABLE`]).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVtSurface {
    /// The lend's name, which a `rels` names it by: the kext's own, nonzero, never the same as
    /// another lend's it holds.
    pub lend: u64,
    /// The surface's bytes, `IOSurface::getAllocSize`.
    pub length: u64,
    /// `IOSurface::getSurfaceID`.
    pub surface_id: u32,
    /// `IOSurface::getPixelFormat`: the surface's OSType, its four characters big-endian.
    pub pixel_format: u32,
    /// `IOSurface::getWidth`.
    pub width: u32,
    /// `IOSurface::getHeight`.
    pub height: u32,
    /// `IOSurface::getPlaneCount`: 0 for a surface without planes, whose plane 0 describes it
    /// whole.
    pub plane_count: u32,
    /// How many runs follow; with [`RGFX_VT_SURFACE_RUN_TABLE`], the runs of the surface's run
    /// table.
    pub run_count: u32,
    /// Its planes, those past `plane_count` (past 1 for a surface without planes) zero.
    pub planes: [RgfxVtSurfacePlane; RGFX_VT_SURFACE_PLANES],
}

/// One plane of an [`RgfxVtSurface`].
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVtSurfacePlane {
    /// The plane's first byte in the surface, `IOSurface::getPlaneOffset`; 0 for a surface
    /// without planes.
    pub offset: u64,
    /// Bytes from one of its rows to the next, `IOSurface::getPlaneBytesPerRow` (the surface's
    /// `getBytesPerRow` for one without planes).
    pub bytes_per_row: u32,
    /// Its width in pixels, `IOSurface::getPlaneWidth`.
    pub width: u32,
    /// Its height in pixels, `IOSurface::getPlaneHeight`.
    pub height: u32,
    /// Zero.
    pub reserved_14: u32,
}

/// `VT_STATUS`: a `VT_MESSAGE`'s answer.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVtStatus {
    /// [`RGFX_EVT_VT_STATUS`], with the message's `seqno`.
    pub header: RgfxMsgHeader,
    /// The status VideoToolbox's host side returns for the message (an `OSStatus`: 0 for one it
    /// took), or [`RGFX_VT_STATUS_MALFORMED`].
    pub status: i32,
    /// Pads the message to a multiple of 8; 0.
    pub reserved_14: u32,
}

/// `VT_RING`: the VideoToolbox ring holds the host's messages through `produced`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVtRing {
    /// [`RGFX_EVT_VT_RING`], `seqno` 0.
    pub header: RgfxMsgHeader,
    /// The ring's count after the host's latest message there, which the guest takes through.
    pub produced: u64,
}

/// The `VT_MESSAGE` event: one message of VideoToolbox's host side, its `length` bytes after
/// this struct.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVtHostMessage {
    /// [`RGFX_EVT_VT_MESSAGE`], `seqno` 0.
    pub header: RgfxMsgHeader,
    /// The message's bytes, at most [`RGFX_VT_HOST_MESSAGE_MAX`].
    pub length: u32,
    /// Pads the bytes to a multiple of 8; 0.
    pub reserved_14: u32,
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
    /// The buffer's guest-picked GPU address. 0, where the guest took
    /// [`RGFX_FEATURE_IMPORT_ADDRESSES_LATER`], for the host's: the address the host gave the
    /// block's view plus `offset`, and where it gave the view none, where the platform places the
    /// buffer ([`RGFX_CMD_PAGES_BUFFER_ADDRESS`] answers it).
    pub gpu_address: u64,
    /// The pages block's region offset, a block of the sender's context.
    pub block_offset: u64,
    /// The pages block's size in bytes.
    pub block_size: u64,
    /// Where the buffer starts in the block's view, in bytes. Where the guest took
    /// [`RGFX_FEATURE_IMPORT_ADDRESSES`] and the block's `SURFACE_PAGES` answered a GPU address
    /// ([`RgfxPagesAnswer`]), `gpu_address` is that address plus this.
    pub offset: u64,
    /// The buffer's length in bytes.
    pub length: u64,
    /// `MTLResourceOptions`: storage mode, cache mode, hazard tracking, and bit 24, Metal's
    /// private bit, a buffer whose GPU writes are dropped (the only kind over a read-only pages
    /// block, [`RGFX_SURFACE_PAGES_READ_ONLY`]).
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

/// `POOL_TAKE`: context `context` asks for a block of the Shared pool of at least `length` bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxPoolTake {
    /// [`RGFX_CMD_POOL_TAKE`].
    pub header: RgfxMsgHeader,
    /// The asking context.
    pub context: u32,
    /// Zero.
    pub reserved_14: u32,
    /// The bytes asked for.
    pub length: u64,
}

/// `POOL_GIVE`: context `context` gives back its pool block of `length` bytes at `pool_offset`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxPoolGive {
    /// [`RGFX_CMD_POOL_GIVE`].
    pub header: RgfxMsgHeader,
    /// The giving context.
    pub context: u32,
    /// Zero.
    pub reserved_14: u32,
    /// The block's offset in the pool.
    pub pool_offset: u64,
    /// The block's length, as its `POOL_BLOCK` answered it.
    pub length: u64,
}

/// Event `POOL_BLOCK`: a `POOL_TAKE`'s answer, with its `seqno`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxPoolBlock {
    /// [`RGFX_EVT_POOL_BLOCK`], the take's `seqno`.
    pub header: RgfxMsgHeader,
    /// [`RGFX_STATUS_OK`], [`RGFX_STATUS_INVALID_ARGUMENT`] or [`RGFX_STATUS_REFUSED`].
    pub status: u32,
    /// Zero.
    pub reserved_14: u32,
    /// The block's offset in the pool; 0 unless OK.
    pub pool_offset: u64,
    /// The block's length; 0 unless OK.
    pub length: u64,
    /// The GPU address of the block's first byte, never 0, where the guest took
    /// [`RGFX_FEATURE_POOL_ADDRESSES`] and the status is OK; otherwise 0.
    pub gpu_address: u64,
}

/// `PRIVATE_TAKE`: context `context` asks for a Private block of at least `length` bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxPrivateTake {
    /// [`RGFX_CMD_PRIVATE_TAKE`].
    pub header: RgfxMsgHeader,
    /// The asking context.
    pub context: u32,
    /// Zero.
    pub reserved_14: u32,
    /// The bytes asked for.
    pub length: u64,
}

/// `PRIVATE_GIVE`: context `context` gives back its Private block of `length` bytes at
/// `gpu_address`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxPrivateGive {
    /// [`RGFX_CMD_PRIVATE_GIVE`].
    pub header: RgfxMsgHeader,
    /// The giving context.
    pub context: u32,
    /// Zero.
    pub reserved_14: u32,
    /// The GPU address of the block's first byte, as its `PRIVATE_BLOCK` answered it.
    pub gpu_address: u64,
    /// The block's length, as its `PRIVATE_BLOCK` answered it.
    pub length: u64,
}

/// Event `PRIVATE_BLOCK`: a `PRIVATE_TAKE`'s answer, with its `seqno`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxPrivateBlock {
    /// [`RGFX_EVT_PRIVATE_BLOCK`], the take's `seqno`.
    pub header: RgfxMsgHeader,
    /// [`RGFX_STATUS_OK`], [`RGFX_STATUS_INVALID_ARGUMENT`] or [`RGFX_STATUS_REFUSED`].
    pub status: u32,
    /// Zero.
    pub reserved_14: u32,
    /// The GPU address of the block's first byte, never 0; 0 unless OK.
    pub gpu_address: u64,
    /// The block's length; 0 unless OK.
    pub length: u64,
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

/// `HEAP_OVER_PAGES`: a heap of `size_bytes` at the guest-picked `gpu_address` over the guest's
/// RAM of the pages block of `block_size` bytes at `block_offset`, a block of the sender's
/// context whose pages are one run, at least `size_bytes` rounded up to Metal's 16 KiB page.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxHeapOverPages {
    /// [`RGFX_CMD_HEAP_OVER_PAGES`].
    pub header: RgfxMsgHeader,
    /// The new heap's guest-picked id.
    pub obj_id: u64,
    /// The heap's guest-picked GPU address.
    pub gpu_address: u64,
    /// The pages block's region offset, a block of the sender's context.
    pub block_offset: u64,
    /// The pages block's size in bytes.
    pub block_size: u64,
    /// The heap's size in bytes.
    pub size_bytes: u64,
    /// `MTLResourceOptions`: storage mode, cache mode, hazard tracking.
    pub options: u32,
    /// `MTLHeapType` (Automatic, Placement or Sparse); widened on decode.
    pub heap_type: i32,
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

/// `MANAGED_BUFFER_CREATE`: a `Managed` buffer of `length` bytes at the guest-picked
/// `gpu_address`, its CPU copy the first `length` bytes of the `Shared` buffer
/// `cpu_buffer_obj_id`, which lies over the guest's memory: over a block of the sender's
/// (`BUFFER_OVER_HOST_MEMORY`), placed in a heap over one or over a pages block's guest RAM
/// (`BUFFER_AT_OFFSET`), or over a pages block's view (`BUFFER_OVER_PAGES`). The host's GPU copy,
/// where it keeps one, starts as the CPU copy's bytes; `BUFFER_DID_MODIFY_RANGE` and a
/// synchronize copy between the two as for a [`RgfxBufferOverHostMemory`] of a `Managed` buffer.
/// The buffer lies over what its CPU copy lies over, and keeps the copy's memory while it lives.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxManagedBufferCreate {
    /// [`RGFX_CMD_MANAGED_BUFFER_CREATE`].
    pub header: RgfxMsgHeader,
    /// The new buffer's guest-picked id.
    pub obj_id: u64,
    /// The buffer's guest-picked GPU address.
    pub gpu_address: u64,
    /// The `Shared` buffer whose first `length` bytes are the CPU copy.
    pub cpu_buffer_obj_id: u64,
    /// The buffer's length in bytes.
    pub length: u64,
    /// `MTLResourceOptions` of the `Managed` buffer: storage mode, cache mode, hazard tracking.
    pub options: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_34: u32,
}

/// Where a `MANAGED_BUFFER_CREATE` message's [`RgfxManagedBufferZero`] starts: right after its
/// [`RgfxManagedBufferCreate`]. A message whose `size` is at least
/// `RGFX_MANAGED_BUFFER_ZERO_OFFSET + size_of::<RgfxManagedBufferZero>()` carries it.
pub const RGFX_MANAGED_BUFFER_ZERO_OFFSET: u32 = 56;

/// The bytes of a `MANAGED_BUFFER_CREATE`'s CPU copy that read zero as the guest posts it, after
/// the [`RgfxManagedBufferCreate`] at [`RGFX_MANAGED_BUFFER_ZERO_OFFSET`]: every byte from
/// `zero_from` to the buffer's `length`, as a buffer the guest made with no bytes, or with fewer
/// than its length, holds. The host's GPU copy, where it keeps one, starts with zeros there that
/// the host's GPU writes, rather than with the CPU copy's bytes copied in, so a guest write
/// there reaches it through a `BUFFER_DID_MODIFY_RANGE`, as Metal's GPU copy of a `Managed`
/// buffer takes a CPU write made after the buffer's making. A `zero_from` at or past `length`,
/// and a create without it, say nothing of the bytes: the GPU copy starts as the CPU copy's
/// bytes.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxManagedBufferZero {
    /// The first byte of the CPU copy from which every byte to the buffer's length reads zero.
    pub zero_from: u64,
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
    /// 0 when unknown (e.g. a recovery victim). For a command buffer the host
    /// committed in parts, its first part's start.
    pub gpu_start_ns: u64,
    /// `gpu_start_ns` plus the GPU time of the work, in nanoseconds of the host's
    /// monotonic clock: when the GPU finished the work, for a command buffer
    /// committed whole; for one committed in parts, its parts' GPU time summed,
    /// so no later than its last part's end. The guest reads only
    /// `gpu_end_ns - gpu_start_ns`, as the run's length (docs/graphics-kext/design.md,
    /// "Completion").
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
/// (`encodeSignalEvent:value:`) by the buffer. Where the buffer has more work after the signal (an
/// encoder, or a wait for an event), it is posted as the host's event reaches the value, while the
/// buffer may still run, or wait to be scheduled; else once the buffer finishes. Each signal is
/// posted once, whatever the buffer's outcome.
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

/// `ICB_SLOTS`: `run_count` [`RgfxIcbSlotRun`]s follow, to the message's `size`; and
/// `ICB_SLOTS_IN_PAGES`: `run_count` [`RgfxIcbSlotPagesRun`]s.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbSlots {
    /// [`RGFX_CMD_ICB_SLOTS`] or [`RGFX_CMD_ICB_SLOTS_IN_PAGES`].
    pub header: RgfxMsgHeader,
    /// The first run's indirect command buffer, which places the message in its context.
    pub obj_id: u64,
    /// How many runs follow.
    pub run_count: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_1c: u32,
}

/// [`RgfxIcbSlotRun`] `kind`: render slots ([`crate::icb::RgfxIcbRenderSlot`]).
pub const RGFX_ICB_SLOTS_RENDER: u16 = 1;
/// [`RgfxIcbSlotRun`] `kind`: compute slots ([`crate::icb::RgfxIcbComputeSlot`]).
pub const RGFX_ICB_SLOTS_COMPUTE: u16 = 2;

/// One run of an `ICB_SLOTS`: the slots of commands `first_command` to `first_command +
/// command_count` of `icb_obj_id`, the first at `region_offset`, `slot_bytes` apart, each of
/// `kind` with `bindings_a` bindings of the first kind (vertex or kernel buffers) and `bindings_b`
/// of the second (fragment buffers or threadgroup memory lengths).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbSlotRun {
    /// The indirect command buffer.
    pub icb_obj_id: u64,
    /// The first slot's place in the region: 8-aligned.
    pub region_offset: u64,
    /// The first slot's command.
    pub first_command: u32,
    /// How many commands' slots.
    pub command_count: u32,
    /// The bytes from one slot to the next: a multiple of 8, at least the slot's
    /// ([`crate::icb::slot_bytes`]).
    pub slot_bytes: u32,
    /// [`RGFX_ICB_SLOTS_RENDER`] or [`RGFX_ICB_SLOTS_COMPUTE`].
    pub kind: u16,
    /// The slot's bindings of the first kind, at most 32.
    pub bindings_a: u8,
    /// The slot's bindings of the second kind, at most 32.
    pub bindings_b: u8,
}

/// One run of an `ICB_SLOTS_IN_PAGES`: as [`RgfxIcbSlotRun`], the first slot `pages_offset`
/// bytes into the pages of the pages block at `block_offset`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxIcbSlotPagesRun {
    /// The indirect command buffer.
    pub icb_obj_id: u64,
    /// The pages block's region offset.
    pub block_offset: u64,
    /// The pages block's size in the region.
    pub block_size: u64,
    /// The first slot's place in the block's pages: 8-aligned.
    pub pages_offset: u64,
    /// The first slot's command.
    pub first_command: u32,
    /// How many commands' slots.
    pub command_count: u32,
    /// The bytes from one slot to the next: a multiple of 8, at least the slot's
    /// ([`crate::icb::slot_bytes`]).
    pub slot_bytes: u32,
    /// [`RGFX_ICB_SLOTS_RENDER`] or [`RGFX_ICB_SLOTS_COMPUTE`].
    pub kind: u16,
    /// The slot's bindings of the first kind, at most 32.
    pub bindings_a: u8,
    /// The slot's bindings of the second kind, at most 32.
    pub bindings_b: u8,
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

/// `TEXTURE_DID_MODIFY_DATA`: the CPU changed the bytes texture `obj_id` lies over
/// (`didModifyData`, SPI).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxTextureDidModifyData {
    /// [`RGFX_CMD_TEXTURE_DID_MODIFY_DATA`].
    pub header: RgfxMsgHeader,
    /// The texture.
    pub obj_id: u64,
}

/// `BUFFER_ESCAPED`: buffer `obj_id`'s identity left the guest's Metal by the act `kind`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxBufferEscaped {
    /// [`RGFX_CMD_BUFFER_ESCAPED`].
    pub header: RgfxMsgHeader,
    /// The buffer.
    pub obj_id: u64,
    /// The act that handed its identity out: `RGFX_BUFFER_ESCAPED_*`.
    pub kind: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_1c: u32,
}

/// `PAGES_BUFFER_ADDRESS`: the GPU address of buffer `obj_id`, answered by an
/// [`RgfxPagesAnswer`].
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxPagesBufferAddress {
    /// [`RGFX_CMD_PAGES_BUFFER_ADDRESS`].
    pub header: RgfxMsgHeader,
    /// The buffer, one a `BUFFER_OVER_PAGES` of the sender's made.
    pub obj_id: u64,
}

/// [`RgfxBufferEscaped::kind`]: the app read the buffer's `gpuAddress`, or that of a texture over
/// it (SPI), which answers the buffer's address.
pub const RGFX_BUFFER_ESCAPED_GPU_ADDRESS: u32 = 0;
/// [`RgfxBufferEscaped::kind`]: an argument encoder wrote the buffer's address
/// (`setBuffer:offset:atIndex:`, `setBuffers:offsets:withRange:`), or the `gpuResourceID` of a
/// texture over it (`setTexture:atIndex:`, `setTextures:withRange:`), into an argument buffer.
pub const RGFX_BUFFER_ESCAPED_ARGUMENT_ENCODER: u32 = 1;
/// [`RgfxBufferEscaped::kind`]: the app read the `gpuResourceID` of a texture, or texture view,
/// that lies over the buffer.
pub const RGFX_BUFFER_ESCAPED_RESOURCE_ID: u32 = 2;
/// [`RgfxBufferEscaped::kind`]: the buffer, or a texture over it, joined a residency set
/// (`addAllocation:`, `addAllocations:count:`).
pub const RGFX_BUFFER_ESCAPED_RESIDENCY_SET: u32 = 3;

/// [`RgfxRateMapCreate::flags`]: the descriptor's `mutability` (SPI) was set, and
/// [`RgfxRateMapCreate::mutability`] holds it; without the flag the map is of Metal's default
/// mutability, immutable.
pub const RGFX_RATE_MAP_MUTABILITY_SET: u32 = 1 << 0;

/// `RATE_MAP_CREATE` and `RATE_MAP_RESET`: a rasterization rate map's descriptor
/// (`MTLRasterizationRateMapDescriptor`). `layer_count` [`RgfxRateMapLayer`]s lie at
/// `layers_offset`, the descriptor's layers from index 0, each naming its samples. Offsets are
/// bytes from the message start; the message's `size` is the head plus the tail, rounded up to a
/// multiple of 8. The descriptor's minimum factor (SPI) is not carried: it changes nothing a map
/// answers or draws (`reims-vgpu-vulkan`'s `gpu/rate_map.rs`), and neither is its label, which
/// the guest's map answers itself.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxRateMapCreate {
    /// [`RGFX_CMD_RATE_MAP_CREATE`] or [`RGFX_CMD_RATE_MAP_RESET`].
    pub header: RgfxMsgHeader,
    /// The map: the new one's guest-picked id, or the one a reset compiles again.
    pub obj_id: u64,
    /// `screenSize.width`.
    pub screen_width: u64,
    /// `screenSize.height`.
    pub screen_height: u64,
    /// `screenSize.depth`.
    pub screen_depth: u64,
    /// The `RGFX_RATE_MAP_*` flags.
    pub flags: u32,
    /// `mutability` (`MTLMutability`, SPI), with [`RGFX_RATE_MAP_MUTABILITY_SET`]; 0 without it.
    /// A reset keeps the map's own, and ignores it.
    pub mutability: u32,
    /// The [`RgfxRateMapLayer`]s at `layers_offset`.
    pub layer_count: u32,
    /// The layers' byte offset from the message start.
    pub layers_offset: u32,
}

/// One layer of an [`RgfxRateMapCreate`] (`MTLRasterizationRateLayerDescriptor`): its
/// `sampleCount`'s width and height, and its samples at `samples_offset`, `horizontal_count`
/// horizontal ones then `vertical_count` vertical ones, each the bits of its `f32`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxRateMapLayer {
    /// `sampleCount.width`: the horizontal samples.
    pub horizontal_count: u32,
    /// `sampleCount.height`: the vertical samples.
    pub vertical_count: u32,
    /// The samples' byte offset from the message start.
    pub samples_offset: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_0c: u32,
}

/// `RATE_MAP_STATE`: a `RATE_MAP_CREATE`'s or `RATE_MAP_RESET`'s answer. `status` is
/// [`RGFX_STATUS_OK`] with the map's answers that take no argument; else every other field is 0:
/// [`RGFX_STATUS_REFUSED`] where the platform made no map, or refused the reset, which leaves the
/// map as it was; [`RGFX_STATUS_NOT_FOUND`] for a reset of a map the sender does not hold; and
/// [`RGFX_STATUS_INVALID_ARGUMENT`] for a message whose layers or samples it does not hold.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxRateMapState {
    /// [`RGFX_EVT_RATE_MAP_STATE`], with the message's `seqno`.
    pub header: RgfxMsgHeader,
    /// An `RGFX_STATUS_*`.
    pub status: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_14: u32,
    /// `screenSize.width`.
    pub screen_width: u64,
    /// `screenSize.height`.
    pub screen_height: u64,
    /// `screenSize.depth`.
    pub screen_depth: u64,
    /// `physicalGranularity.width`.
    pub granularity_width: u64,
    /// `physicalGranularity.height`.
    pub granularity_height: u64,
    /// `physicalGranularity.depth`.
    pub granularity_depth: u64,
    /// `layerCount`.
    pub layer_count: u64,
    /// `mutability` (`MTLMutability`, SPI).
    pub mutability: u64,
    /// `parameterBufferSizeAndAlign.size`.
    pub parameter_size: u64,
    /// `parameterBufferSizeAndAlign.align`.
    pub parameter_align: u64,
}

/// [`RgfxRateMapQuery::query`]: `physicalSizeForLayer:` of layer `at`, answered as the width,
/// height and depth.
pub const RGFX_RATE_MAP_QUERY_PHYSICAL_SIZE: u32 = 0;
/// [`RgfxRateMapQuery::query`]: `mapScreenToPhysicalCoordinates:forLayer:` of (`x`, `y`) in
/// layer `at`, answered as the coordinate's `x` and `y`.
pub const RGFX_RATE_MAP_QUERY_SCREEN_TO_PHYSICAL: u32 = 1;
/// [`RgfxRateMapQuery::query`]: `mapPhysicalToScreenCoordinates:forLayer:` of (`x`, `y`) in
/// layer `at`, answered as the coordinate's `x` and `y`.
pub const RGFX_RATE_MAP_QUERY_PHYSICAL_TO_SCREEN: u32 = 2;
/// [`RgfxRateMapQuery::query`]: `copyParameterDataToBuffer:offset:` into buffer
/// `buffer_obj_id` at offset `at`: the platform writes the map's parameter data where the
/// buffer's contents lie, before the answer, which carries no values.
pub const RGFX_RATE_MAP_QUERY_COPY_PARAMETER_DATA: u32 = 3;

/// `RATE_MAP_QUERY`: map `obj_id`'s answer `query` (`RGFX_RATE_MAP_QUERY_*`).
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxRateMapQuery {
    /// [`RGFX_CMD_RATE_MAP_QUERY`].
    pub header: RgfxMsgHeader,
    /// The map.
    pub obj_id: u64,
    /// The layer, or for [`RGFX_RATE_MAP_QUERY_COPY_PARAMETER_DATA`] the buffer's offset.
    pub at: u64,
    /// The buffer, for [`RGFX_RATE_MAP_QUERY_COPY_PARAMETER_DATA`]: one of the sender's; 0
    /// otherwise.
    pub buffer_obj_id: u64,
    /// The `RGFX_RATE_MAP_QUERY_*` asked.
    pub query: u32,
    /// The coordinate's `x`, as the bits of its `f32`; 0 for any other query.
    pub x_bits: u32,
    /// The coordinate's `y`, as the bits of its `f32`; 0 for any other query.
    pub y_bits: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_34: u32,
}

/// `RATE_MAP_ANSWER`: a `RATE_MAP_QUERY`'s answer. `status` is [`RGFX_STATUS_OK`] with the
/// answer in `values`: a physical size's width, height and depth, or a coordinate's `x` and `y`
/// as the bits of their `f32` in `values[0]` and `values[1]`, or nothing for a copy.
/// [`RGFX_STATUS_NOT_FOUND`] for a map or buffer the sender does not hold,
/// [`RGFX_STATUS_INVALID_ARGUMENT`] for a query the wire does not define or a message too short,
/// and [`RGFX_STATUS_REFUSED`] where the platform refused the query, where Metal's process ends (a
/// coordinate of a map of no layers or of an empty screen, physical to screen of a mutable map
/// not yet reset, a copy into a `Private` buffer or past a buffer's end), each with `values` 0.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxRateMapAnswer {
    /// [`RGFX_EVT_RATE_MAP_ANSWER`], with the query's `seqno`.
    pub header: RgfxMsgHeader,
    /// An `RGFX_STATUS_*`.
    pub status: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_14: u32,
    /// The answer.
    pub values: [u64; 3],
}

/// `FUNCTION_HANDLE_CREATE`: compute pipeline state `pipeline_obj_id`'s handle of `function`, a
/// function named as a linked function is (its library, its entry point's name, its constants,
/// options and `specializedName`, offsets from the message start; its `list` is
/// [`RGFX_LINKED_FUNCTIONS`]), made under `obj_id`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxFunctionHandleCreate {
    /// [`RGFX_CMD_FUNCTION_HANDLE_CREATE`].
    pub header: RgfxMsgHeader,
    /// The handle's id.
    pub obj_id: u64,
    /// The compute pipeline state: one of the sender's.
    pub pipeline_obj_id: u64,
    /// The function.
    pub function: RgfxLinkedFunction,
}

/// `FUNCTION_TABLE_CREATE`: a visible function table of `function_count` null entries of compute
/// pipeline state `pipeline_obj_id`, made under `obj_id`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxFunctionTableCreate {
    /// [`RGFX_CMD_FUNCTION_TABLE_CREATE`].
    pub header: RgfxMsgHeader,
    /// The table's id.
    pub obj_id: u64,
    /// The compute pipeline state: one of the sender's.
    pub pipeline_obj_id: u64,
    /// `MTLVisibleFunctionTableDescriptor.functionCount`.
    pub function_count: u64,
}

/// `FUNCTION_TABLE_SET`: entries `first` to `first + count - 1` of table `obj_id` call the
/// `count` handles whose ids (`u64`, 0 for a null entry) follow this head, in order.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxFunctionTableSet {
    /// [`RGFX_CMD_FUNCTION_TABLE_SET`].
    pub header: RgfxMsgHeader,
    /// The table: one of the sender's.
    pub obj_id: u64,
    /// The first entry set.
    pub first: u64,
    /// How many entries are set: how many handle ids follow.
    pub count: u64,
}

/// `DYNAMIC_LIBRARY_CREATE`: library `library_obj_id` as a dynamic library, made under `obj_id`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxDynamicLibraryCreate {
    /// [`RGFX_CMD_DYNAMIC_LIBRARY_CREATE`].
    pub header: RgfxMsgHeader,
    /// The dynamic library's id.
    pub obj_id: u64,
    /// The library: one of the sender's.
    pub library_obj_id: u64,
}

/// `FUNCTION_POINTER_STATE`: a `FUNCTION_HANDLE_CREATE`'s or `FUNCTION_TABLE_CREATE`'s answer.
/// `status` is [`RGFX_STATUS_OK`] where the object was made: for a table, with its
/// `gpu_resource_id`, `allocated_size` and `resource_options`, and 0 in each for a handle.
/// [`RGFX_STATUS_NOT_FOUND`] for a handle Metal answers nil (a function the state's kernel does
/// not link) and for a pipeline state or library the sender does not hold,
/// [`RGFX_STATUS_INVALID_ARGUMENT`] for a message too short, and [`RGFX_STATUS_REFUSED`] where
/// the platform made none, each with the values 0 and nothing made.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxFunctionPointerState {
    /// [`RGFX_EVT_FUNCTION_POINTER_STATE`], with the create's `seqno`.
    pub header: RgfxMsgHeader,
    /// An `RGFX_STATUS_*`.
    pub status: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_14: u32,
    /// A table's `gpuResourceID`: what an argument buffer holds of it.
    pub gpu_resource_id: u64,
    /// A table's `allocatedSize`.
    pub allocated_size: u64,
    /// A table's `resourceOptions`.
    pub resource_options: u64,
}

/// The rows of a level of a `Managed` texture's CPU copy start at multiples of this, in bytes
/// ([`RgfxManagedTextureCreate`]).
pub const RGFX_MANAGED_ROW_ALIGN: u64 = 256;

/// [`RgfxManagedTextureCreate::flags`]: the pixel format is one a texture over a buffer can
/// have (uncompressed, one texel a block, neither depth nor stencil).
pub const RGFX_MANAGED_TEXTURE_LINEAR: u32 = 1;

/// `MANAGED_TEXTURE_CREATE`: a `Managed` texture of shape `desc` at the guest-picked
/// `gpu_address`, with the guest-picked `resource_id`, whose CPU copy is the Shared buffer
/// `cpu_buffer_obj_id` over a block of the sender's, or, where the guest took
/// [`RGFX_FEATURE_MANAGED_TEXTURE_SLABS`], any `Shared` buffer over the guest's memory it holds.
///
/// The CPU copy holds every level in order; each level its slices in order (`array_length`, six
/// faces a cube); each slice its images (the level's depth, for a 3D texture; else one); each
/// image its rows of texel blocks. A level of `w` by `h` texels (the texture's halved `level`
/// times, at least 1) has rows of `ceil(w / block_width) * bytes_per_block` bytes rounded up to
/// [`RGFX_MANAGED_ROW_ALIGN`], and `ceil(h / block_height)` of them an image. The guest and the
/// host each compute it from these fields alone (`reims_vgpu_graphics::managed`).
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
/// guest-picked `resource_id`. A `desc` of more than one level, a 2D texture
/// of Metal's `newTiledTextureWithDescriptor:offset:bytesPerRow:`, holds a
/// copy of those bytes, every level made from them, taken at its first use
/// and again after [`RGFX_CMD_TEXTURE_DID_MODIFY_DATA`].
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
/// The bundle sends it with every create, and the host makes the sampler with both fields.
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
/// of the 15 bytes `rgfx-library-v2`, then of the SHA-256 of each 256 KiB piece, in order (the
/// last the rest), of the bytes before the bitcode section followed by the bytes after it; the
/// section is the `u64` offset and size at 0x48 and 0x50 of the `.metallib`'s header. Each
/// function's bitcode is named by its `HASH` tag, its SHA-256, which lies outside the section, so
/// the key names every byte a function is built from; the pieces are hashed each on its own, so
/// a guest hashes them on its cores at once. `slice_size` is the bytes' length.
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

/// Where [`RgfxDeviceInfo::sizes`] holds `maxBufferLength`, which the kext reads to size the
/// contexts' GPU-address ranges (`guest/kext/src/RgfxContextSlots.hpp`).
pub const RGFX_DEVICE_INFO_MAX_BUFFER_LENGTH: usize = 7;

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
    /// `minimumLinearTextureAlignmentForPixelFormat:`, in bytes: the host GPU's alignment of a
    /// texture's offset and rows in a buffer for this format; 0 where the host has no answer of
    /// the format (one no buffer backs, among them) or one past 65535, where the guest answers
    /// the device's `linearTextureAlignmentBytes`, the largest of every format's.
    pub linear_alignment: u16,
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
    pub reserved_453: u8,
    /// The bytes the host's GPU aligns the offset and rows of a linear BGRA8 image to, to make
    /// one over a surface's bytes in place there, where the host offers the surfaces' pages
    /// (`SURFACE_PAGES`) to every guest, on a GPU of any memory, and its platform answers; else
    /// 0, as from a host older than the field, which wrote it as zero. The kext has IOSurface
    /// align the bytes per row and the plane offsets it works out for a surface to a multiple
    /// of it, and of 64.
    pub linear_row_alignment: u32,
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

/// How a decode of one codec lands its picture in place, in the output surface's planes
/// themselves, with no copy: the host's GPU writes the picture's whole coded rows and columns
/// there, in units of `granularity`, so a surface laid out to hold them takes it in place. The
/// guest's video driver asks AppleGVA for such surfaces: each plane's rows the frame's height
/// rounded up to a multiple of `granularity[1]` (chroma half), its rows `alignment` apart, from
/// the frame's width rounded up to a multiple of `granularity[0]`, and its first byte at a
/// multiple of `alignment` ([`RgfxVideoOpDecodeH264`]'s `output_rows` says what a surface holds).
/// All zero where the host lands a picture by a copy of its crop alone, and from a host older
/// than these: the driver then asks for planes of the frame's height.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVideoOutputLayout {
    /// The units of luma samples a decode writes in place: width, height.
    pub granularity: [u32; 2],
    /// What each plane's first byte, and the bytes from one of its rows to the next, are a
    /// multiple of.
    pub alignment: u32,
    /// Zero.
    pub reserved_0c: u32,
}

/// What the host's hardware takes of one encode codec beside its [`RgfxVideoCodecCaps`], by the
/// same index ([`RgfxVideoCaps::encode`]). All zero for a decode entry, and from a host older
/// than these: a guest reads an encode entry whose `rate_controls` is 0 as no encoder.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVideoEncodeCaps {
    /// The rate controls taken, a bit `1 << RGFX_VIDEO_RATE_*` each
    /// ([`RGFX_VIDEO_RATE_CQP`](crate::ops::video::RGFX_VIDEO_RATE_CQP)), as the host's API lists
    /// them for the codec's profiles.
    pub rate_controls: u32,
    /// The most consecutive B pictures an encode takes
    /// ([`RgfxVideoEncodeSettings::b_frames`](crate::ops::video::RgfxVideoEncodeSettings)).
    pub max_b_frames: u32,
    /// Zero: reserved for intra refresh and the like.
    pub flags: u32,
    /// Zero.
    pub reserved_0c: u32,
}

/// What the host's hardware takes of one decode codec beside its [`RgfxVideoCodecCaps`], by the
/// same index ([`RgfxVideoCaps::decode`]): the bounds of a picture the host refuses, which the
/// guest's video driver checks before it schedules the picture. All zero for an encode entry,
/// and from a host older than these: no bound beyond the entry's.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxVideoDecodeCaps {
    /// The narrowest tile column a decode takes, in luma samples: HEVC's
    /// `ColumnWidthInLumaSamples`, a column's CTBs times their size; 0 where any is.
    pub min_tile_width: u32,
    /// Zero.
    pub reserved_04: u32,
    /// Zero.
    pub reserved_08: u32,
    /// Zero.
    pub reserved_0c: u32,
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
    /// How each of `codecs` lands its pictures in place, by its index there; all zero for a codec
    /// whose pictures land by a copy, and from a host older than them. For an encode entry, what a
    /// source must be laid out as to be read in place, by the GPU in one pass: its planes' first
    /// bytes and rows at multiples of `alignment`, in units of `granularity`; a source of other
    /// pitches still encodes, through a copy by the CPU.
    pub layouts: [RgfxVideoOutputLayout; RGFX_VIDEO_CAPS_CODECS],
    /// What each encode entry of `codecs` takes beyond them, by its index there.
    pub encode: [RgfxVideoEncodeCaps; RGFX_VIDEO_CAPS_CODECS],
    /// What each decode entry of `codecs` takes beyond them, by its index there.
    pub decode: [RgfxVideoDecodeCaps; RGFX_VIDEO_CAPS_CODECS],
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
/// A pipeline reflection reply's kind: a mesh render pipeline
/// ([`RGFX_CMD_MESH_RENDER_PIPELINE_CREATE`]). A reply of this kind whose `status` is
/// [`RGFX_PIPELINE_OK`] carries an [`RgfxMeshPipelineReflection`] at
/// [`RGFX_PIPELINE_REFLECTION_MESH_OFFSET`], and its header's `size` says so; a refusal is the
/// plain reply.
pub const RGFX_PIPELINE_KIND_MESH: u32 = 2;

/// [`RgfxPipelineReflection`] `flags`: the state supports indirect command
/// buffers (`supportIndirectCommandBuffers`).
pub const RGFX_PIPELINE_REFLECT_SUPPORT_ICB: u32 = 1 << 0;
/// [`RgfxPipelineReflection`] `flags`: a render pipeline's threadgroup size
/// matches its tile size (`threadgroupSizeMatchesTileSize`).
pub const RGFX_PIPELINE_REFLECT_THREADGROUP_MATCHES_TILE: u32 = 1 << 1;
/// [`RgfxPipelineReflection`] `flags`: the reply carries the create's pipeline key, 32 bytes at
/// [`RGFX_PIPELINE_REFLECTION_KEY_OFFSET`], and its header's `size` counts them. Set only on a
/// made pipeline's reply to a create that asked for it ([`RGFX_RENDER_PIPELINE_ANSWER_KEY`],
/// [`RGFX_COMPUTE_PIPELINE_ANSWER_KEY`]) whose libraries the device holds under their content
/// keys.
///
/// The key is the SHA-256 of the 16 bytes `rgfx-pipeline-v1`; then the create's whole message,
/// its header's `size` bytes, with its header's `seqno`, its `obj_id` and each library id it names
/// zeroed; then the content key ([`RgfxLibraryLookup`]) of each library it names, in the same
/// order: a compute create's `function_library_obj_id`, then each of its linked functions'
/// `library_obj_id` in their run's order; a render create's `vertex_library_obj_id`,
/// `fragment_library_obj_id`, each of its vertex linked functions' and then each of its fragment
/// linked functions'. A library id of 0, a render create's with no fragment function, adds 32
/// zero bytes. So equal creates of any processes, whose libraries hold equal bytes, have one key.
/// A mesh create, and a compute create that links dynamic libraries, has none.
pub const RGFX_PIPELINE_REFLECT_KEYED: u32 = 1 << 2;

/// [`RgfxRenderPipelineCreate`] `flags`: `alphaToCoverageEnabled`.
pub const RGFX_RENDER_PIPELINE_ALPHA_TO_COVERAGE: u32 = 1 << 0;
/// [`RgfxRenderPipelineCreate`] `flags`: `alphaToOneEnabled`.
pub const RGFX_RENDER_PIPELINE_ALPHA_TO_ONE: u32 = 1 << 1;
/// [`RgfxRenderPipelineCreate`] `flags`: `rasterizationEnabled` (set when the
/// pipeline rasterizes; a vertex-only pipeline clears it).
pub const RGFX_RENDER_PIPELINE_RASTERIZATION_ENABLED: u32 = 1 << 2;
/// [`RgfxRenderPipelineCreate`] `flags`: `supportIndirectCommandBuffers`.
pub const RGFX_RENDER_PIPELINE_SUPPORT_ICB: u32 = 1 << 3;
/// [`RgfxRenderPipelineCreate`] `flags`: an [`RgfxRenderPipelinePrivateState`] follows the
/// create's head, where the guest took [`RGFX_FEATURE_PRIVATE_RENDER_STATE`]. Bits 4 and 5 are
/// the mesh create's.
pub const RGFX_RENDER_PIPELINE_PRIVATE_STATE: u32 = 1 << 6;
/// [`RgfxRenderPipelineCreate`] `flags`: answer the create with its pipeline key
/// ([`RGFX_PIPELINE_REFLECT_KEYED`]), where the guest took [`RGFX_FEATURE_PIPELINE_KEYS`].
pub const RGFX_RENDER_PIPELINE_ANSWER_KEY: u32 = 1 << 7;
/// [`RgfxRenderPipelineCreate`] `flags`: `tessellationFactorScaleEnabled`.
pub const RGFX_RENDER_PIPELINE_TESSELLATION_FACTOR_SCALE: u32 = 1 << 8;

/// [`RgfxRenderPipelinePrivateState`] `flags`: `openGLModeEnabled`.
pub const RGFX_PRIVATE_RENDER_OPENGL_MODE: u32 = 1 << 0;
/// [`RgfxRenderPipelinePrivateState`] `flags`: `sampleCoverageInvert`.
pub const RGFX_PRIVATE_RENDER_SAMPLE_COVERAGE_INVERT: u32 = 1 << 1;
/// [`RgfxRenderPipelinePrivateState`] `flags`: `logicOperationEnabled`.
pub const RGFX_PRIVATE_RENDER_LOGIC_OPERATION: u32 = 1 << 2;
/// [`RgfxRenderPipelinePrivateState`] `flags`: `depthStencilWriteDisabled`.
pub const RGFX_PRIVATE_RENDER_DEPTH_STENCIL_WRITE_DISABLED: u32 = 1 << 3;

/// [`RgfxMeshRenderPipelineCreate`] `flags`, beside the `RGFX_RENDER_PIPELINE_*` ones:
/// `objectThreadgroupSizeIsMultipleOfThreadExecutionWidth`.
pub const RGFX_MESH_PIPELINE_OBJECT_TG_MULTIPLE_OF_EXEC_WIDTH: u32 = 1 << 4;
/// [`RgfxMeshRenderPipelineCreate`] `flags`:
/// `meshThreadgroupSizeIsMultipleOfThreadExecutionWidth`.
pub const RGFX_MESH_PIPELINE_MESH_TG_MULTIPLE_OF_EXEC_WIDTH: u32 = 1 << 5;

/// [`RgfxComputePipelineCreate`] `flags`:
/// `threadGroupSizeIsMultipleOfThreadExecutionWidth`.
pub const RGFX_COMPUTE_PIPELINE_TG_MULTIPLE_OF_EXEC_WIDTH: u32 = 1 << 0;
/// [`RgfxComputePipelineCreate`] `flags`: `supportIndirectCommandBuffers`.
pub const RGFX_COMPUTE_PIPELINE_SUPPORT_ICB: u32 = 1 << 1;
/// [`RgfxComputePipelineCreate`] `flags`: `openCLModeEnabled` (Apple SPI), where the guest took
/// [`RGFX_FEATURE_OPENCL_MODE`].
pub const RGFX_COMPUTE_PIPELINE_OPENCL_MODE: u32 = 1 << 2;
/// [`RgfxComputePipelineCreate`] `flags`: answer the create with its pipeline key
/// ([`RGFX_PIPELINE_REFLECT_KEYED`]), where the guest took [`RGFX_FEATURE_PIPELINE_KEYS`].
pub const RGFX_COMPUTE_PIPELINE_ANSWER_KEY: u32 = 1 << 3;

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
/// bytes (`label_len` 0 for none). With [`RGFX_RENDER_PIPELINE_PRIVATE_STATE`] in
/// `flags`, an [`RgfxRenderPipelinePrivateState`] follows the head, ahead of the
/// tail. The whole message's `size` is the fixed head plus the tail, rounded up
/// to a multiple of 8.
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
    /// `tessellationPartitionMode` (`MTLTessellationPartitionMode`).
    pub tessellation_partition_mode: u32,
    /// `maxTessellationFactor`.
    pub max_tessellation_factor: u32,
    /// `tessellationFactorFormat` (`MTLTessellationFactorFormat`).
    pub tessellation_factor_format: u32,
    /// `tessellationControlPointIndexType` (`MTLTessellationControlPointIndexType`).
    pub tessellation_control_point_index_type: u32,
    /// `tessellationFactorStepFunction` (`MTLTessellationFactorStepFunction`).
    pub tessellation_factor_step_function: u32,
    /// `tessellationOutputWindingOrder` (`MTLWinding`).
    pub tessellation_output_winding_order: u32,
    /// `maxVertexCallStackDepth`.
    pub max_vertex_call_stack_depth: u32,
    /// `maxFragmentCallStackDepth`.
    pub max_fragment_call_stack_depth: u32,
}

/// A render pipeline descriptor's private state (Apple SPI), as Apple's OpenGL renderer sets it,
/// after an [`RgfxRenderPipelineCreate`]'s head where its `flags` carry
/// [`RGFX_RENDER_PIPELINE_PRIVATE_STATE`]: each field the guest's descriptor's value. Measured on
/// the Intel 630 (macOS 26.7.1): every pipeline of the renderer sets `openGLModeEnabled`, a
/// `sampleMask` of `0xffffffff` (or the GL sample mask) and a `logicOperation` of copy (2); a GL
/// logic operation sets `logicOperationEnabled` and its operation, GL's sample coverage
/// `sampleCoverage`, and a pipeline that writes no depth or stencil `depthStencilWriteDisabled`.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxRenderPipelinePrivateState {
    /// `sampleMask`.
    pub sample_mask: u64,
    /// `sampleCoverage`, its `f32` bits.
    pub sample_coverage: u32,
    /// `logicOperation`, the raw value (GL's copy is 2 and its XOR 10, measured).
    pub logic_operation: u32,
    /// The `RGFX_PRIVATE_RENDER_*` flags.
    pub flags: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_14: u32,
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
/// run of [`RgfxLinkedFunction`]. The dynamic libraries it links, where the guest
/// took [`RGFX_FEATURE_DYNAMIC_LIBRARIES`], are the run of `library_count`
/// [`RgfxPipelineLibrary`] that ends the message: its last `16 * library_count`
/// bytes. The message's `size` is the fixed head plus the tail, rounded up to a
/// multiple of 8.
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
    /// The number of [`RgfxPipelineLibrary`] that end the message: the dynamic libraries the
    /// kernel links; 0 for none, and from a guest that did not take
    /// [`RGFX_FEATURE_DYNAMIC_LIBRARIES`].
    pub library_count: u32,
}

/// One dynamic library a compute pipeline's kernel links, of the run that ends its
/// [`RgfxComputePipelineCreate`]: dynamic library `obj_id`, one of the sender's, preloaded
/// (`preloadedLibraries`, searched first, in the run's order) where `name_len` is 0, or else the
/// library the install name at `name_offset` (from the message start) loads, which Metal reads
/// from the file the name names: the guest reads it, and names every install name the kernel's
/// libraries and the dynamic libraries depend on, so the host loads none from a file of its own.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxPipelineLibrary {
    /// The dynamic library: one of the sender's.
    pub obj_id: u64,
    /// The install name's byte offset from the message start; 0 for a preloaded library.
    pub name_offset: u32,
    /// The install name's length in bytes; 0 for a preloaded library.
    pub name_len: u32,
}

/// `MESH_RENDER_PIPELINE_CREATE`: a mesh render pipeline from a descriptor
/// (`MTLMeshRenderPipelineDescriptor`), answered by [`RgfxPipelineReflection`] of
/// [`RGFX_PIPELINE_KIND_MESH`] with this create's `seqno`.
///
/// The object, mesh and fragment functions are each named as a render pipeline's are
/// ([`RgfxRenderPipelineCreate`]): a library `obj_id` (0, with a name of 0 bytes, for no
/// function), an entry-point name in the message tail and a run of [`RgfxFunctionConstant`] in the
/// tail; each stage's linked functions (`objectLinkedFunctions`, `meshLinkedFunctions`,
/// `fragmentLinkedFunctions`) are runs of [`RgfxLinkedFunction`]. The colour attachments are
/// inline, `color_attachment_count` of them in use. The optional `label` is tail bytes. The
/// whole message's `size` is the fixed head plus the tail, rounded up to a multiple of 8.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxMeshRenderPipelineCreate {
    /// [`RGFX_CMD_MESH_RENDER_PIPELINE_CREATE`].
    pub header: RgfxMsgHeader,
    /// The new pipeline's guest-picked id.
    pub obj_id: u64,
    /// The library the object function comes from; 0 for no object function.
    pub object_library_obj_id: u64,
    /// The library the mesh function comes from; 0 for no mesh function.
    pub mesh_library_obj_id: u64,
    /// The library the fragment function comes from; 0 for no fragment function.
    pub fragment_library_obj_id: u64,
    /// The object entry-point name's byte offset from the message start.
    pub object_name_offset: u32,
    /// The object entry-point name's length in bytes.
    pub object_name_len: u32,
    /// The mesh entry-point name's byte offset from the message start.
    pub mesh_name_offset: u32,
    /// The mesh entry-point name's length in bytes.
    pub mesh_name_len: u32,
    /// The fragment entry-point name's byte offset from the message start.
    pub fragment_name_offset: u32,
    /// The fragment entry-point name's length in bytes.
    pub fragment_name_len: u32,
    /// The pipeline label's byte offset from the message start (0-length: none).
    pub label_offset: u32,
    /// The pipeline label's length in bytes; 0 for no label.
    pub label_len: u32,
    /// The object function constants' byte offset from the message start.
    pub object_constants_offset: u32,
    /// The number of [`RgfxFunctionConstant`] for the object function.
    pub object_constant_count: u32,
    /// The mesh function constants' byte offset from the message start.
    pub mesh_constants_offset: u32,
    /// The number of [`RgfxFunctionConstant`] for the mesh function.
    pub mesh_constant_count: u32,
    /// The fragment function constants' byte offset from the message start.
    pub fragment_constants_offset: u32,
    /// The number of [`RgfxFunctionConstant`] for the fragment function.
    pub fragment_constant_count: u32,
    /// The object stage's linked functions' byte offset from the message start.
    pub object_linked_offset: u32,
    /// The number of [`RgfxLinkedFunction`] the object stage links.
    pub object_linked_count: u32,
    /// The mesh stage's linked functions' byte offset from the message start.
    pub mesh_linked_offset: u32,
    /// The number of [`RgfxLinkedFunction`] the mesh stage links.
    pub mesh_linked_count: u32,
    /// The fragment stage's linked functions' byte offset from the message start.
    pub fragment_linked_offset: u32,
    /// The number of [`RgfxLinkedFunction`] the fragment stage links.
    pub fragment_linked_count: u32,
    /// `rasterSampleCount`.
    pub raster_sample_count: u32,
    /// The colour attachments in use, up to 8 (the length of `colors`).
    pub color_attachment_count: u32,
    /// `depthAttachmentPixelFormat` (`MTLPixelFormat`; 0 for none).
    pub depth_pixel_format: u32,
    /// `stencilAttachmentPixelFormat` (`MTLPixelFormat`; 0 for none).
    pub stencil_pixel_format: u32,
    /// The `RGFX_RENDER_PIPELINE_*` flags it has (`alphaToCoverageEnabled`, `alphaToOneEnabled`,
    /// `rasterizationEnabled`, `supportIndirectCommandBuffers`) and the `RGFX_MESH_PIPELINE_*`
    /// ones.
    pub flags: u32,
    /// `maxTotalThreadsPerObjectThreadgroup`; 0 leaves Metal's.
    pub max_total_threads_per_object_threadgroup: u32,
    /// `maxTotalThreadsPerMeshThreadgroup`; 0 leaves Metal's.
    pub max_total_threads_per_mesh_threadgroup: u32,
    /// `maxTotalThreadgroupsPerMeshGrid`; 0 leaves Metal's.
    pub max_total_threadgroups_per_mesh_grid: u32,
    /// `payloadMemoryLength`; 0 leaves Metal's.
    pub payload_memory_length: u32,
    /// `maxVertexAmplificationCount`.
    pub max_vertex_amplification_count: u32,
    /// `requiredThreadsPerObjectThreadgroup`, width, height and depth; all 0 for none.
    pub required_threads_per_object_threadgroup: [u32; 3],
    /// `requiredThreadsPerMeshThreadgroup`, width, height and depth; all 0 for none.
    pub required_threads_per_mesh_threadgroup: [u32; 3],
    /// The eight colour attachments; `color_attachment_count` are in use, the rest zeroed.
    pub colors: [RgfxColorAttachmentDesc; 8],
}

/// Where a [`RGFX_PIPELINE_KIND_MESH`] reply's [`RgfxMeshPipelineReflection`] starts: right after
/// its [`RgfxPipelineReflection`].
pub const RGFX_PIPELINE_REFLECTION_MESH_OFFSET: u32 = 80;

/// Where a [`RGFX_PIPELINE_REFLECT_KEYED`] reply's pipeline key starts: right after its
/// [`RgfxPipelineReflection`]. No mesh reply is keyed.
pub const RGFX_PIPELINE_REFLECTION_KEY_OFFSET: u32 = 80;

/// The mesh stages' answers a made mesh render pipeline's reply carries after its
/// [`RgfxPipelineReflection`], at [`RGFX_PIPELINE_REFLECTION_MESH_OFFSET`]: the state's own
/// queries, as the host's platform answers them.
#[repr(C)]
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, FromBytes, IntoBytes, Immutable, KnownLayout,
)]
pub struct RgfxMeshPipelineReflection {
    /// `maxTotalThreadsPerObjectThreadgroup`.
    pub max_total_threads_per_object_threadgroup: u32,
    /// `maxTotalThreadsPerMeshThreadgroup`.
    pub max_total_threads_per_mesh_threadgroup: u32,
    /// `objectThreadExecutionWidth`.
    pub object_thread_execution_width: u32,
    /// `meshThreadExecutionWidth`.
    pub mesh_thread_execution_width: u32,
    /// `maxTotalThreadgroupsPerMeshGrid`, `u32::MAX` for any value past it (the x86 Macs answer
    /// `NSUIntegerMax`).
    pub max_total_threadgroups_per_mesh_grid: u32,
    /// Pads to a multiple of 8; 0.
    pub reserved_14: u32,
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
    assert!(size_of::<RgfxScheduledAsk>() == 32);
    assert!(core::mem::offset_of!(RgfxCommandBufferPart, region_offset) == 40);
    assert!(size_of::<RgfxHeapCreate>() == 48);
    assert!(size_of::<RgfxHeapOverHostMemory>() == 56);
    assert!(size_of::<RgfxBufferCreate>() == 48);
    assert!(size_of::<RgfxBufferOverHostMemory>() == 56);
    assert!(size_of::<RgfxBufferAtOffset>() == 56);
    assert!(size_of::<RgfxBufferManagedAtOffset>() == 64);
    assert!(size_of::<RgfxHeapOverPages>() == 64);
    assert!(core::mem::offset_of!(RgfxHeapOverPages, heap_type) == 60);
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
    assert!(size_of::<RgfxTextureDidModifyData>() == 24);
    assert!(size_of::<RgfxBufferEscaped>() == 32);
    assert!(size_of::<RgfxPagesBufferAddress>() == 24);
    assert!(size_of::<RgfxRateMapCreate>() == 64);
    assert!(size_of::<RgfxRateMapLayer>() == 16);
    assert!(size_of::<RgfxRateMapState>() == 104);
    assert!(size_of::<RgfxRateMapQuery>() == 56);
    assert!(size_of::<RgfxRateMapAnswer>() == 48);
    assert!(size_of::<RgfxManagedTextureCreate>() == 112);
    assert!(size_of::<RgfxManagedBufferCreate>() == 56);
    assert!(size_of::<RgfxManagedBufferCreate>() == RGFX_MANAGED_BUFFER_ZERO_OFFSET as usize);
    assert!(size_of::<RgfxManagedBufferZero>() == 8);
    assert!(size_of::<RgfxTextureReplaceRegion>() == 56);
    assert!(size_of::<RgfxTextureView>() == 72);
    assert!(size_of::<RgfxHeapTextureSizeAndAlign>() == 72);
    assert!(size_of::<RgfxSizeAndAlign>() == 40);
    assert!(size_of::<RgfxSamplerState>() == 32);
    assert!(size_of::<RgfxIcbCreate>() == 88);
    assert!(size_of::<RgfxIcbState>() == 32);
    assert!(size_of::<RgfxIcbWrite>() == 24);
    assert!(size_of::<RgfxIcbSlots>() == 32);
    assert!(size_of::<RgfxIcbSlotRun>() == 32);
    assert!(RGFX_LIBRARY_BYTES_OFFSET as usize == size_of::<RgfxLibraryCreate>());
    assert!(size_of::<RgfxFormatAnswers>() == 64);
    assert!(core::mem::offset_of!(RgfxFormatAnswers, linear_alignment) == 0x3e);
    assert!(size_of::<RgfxDeviceInfo>() == 1432 + 64 * RGFX_DEVICE_INFO_FORMATS);
    assert!(core::mem::offset_of!(RgfxDeviceInfo, formats) == 1432);
    assert!(core::mem::offset_of!(RgfxDeviceInfo, family_answers) == 976);
    assert!(core::mem::offset_of!(RgfxDeviceInfo, name) == 1112);
    assert!(core::mem::offset_of!(RgfxDeviceInfo, features_lo) == 0x451);
    assert!(core::mem::offset_of!(RgfxDeviceInfo, sampler_reduction_support) == 0x452);
    assert!(core::mem::offset_of!(RgfxDeviceInfo, linear_row_alignment) == 0x454);
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
    assert!(size_of::<RgfxVideoOutputLayout>() == 16);
    assert!(
        size_of::<RgfxVideoCaps>()
            == 8 + 48 * RGFX_VIDEO_CAPS_CODECS
                + 8
                + 16 * RGFX_VIDEO_CAPS_CODECS
                + 16 * RGFX_VIDEO_CAPS_CODECS
                + 16 * RGFX_VIDEO_CAPS_CODECS
    );
    assert!(core::mem::offset_of!(RgfxVideoCaps, reserved_30c) == 0x30c);
    assert!(core::mem::offset_of!(RgfxVideoCaps, layouts) == 0x310);
    assert!(size_of::<RgfxVideoEncodeCaps>() == 16);
    assert!(core::mem::offset_of!(RgfxVideoCaps, encode) == 0x410);
    assert!(size_of::<RgfxVideoDecodeCaps>() == 16);
    assert!(core::mem::offset_of!(RgfxVideoCaps, decode) == 0x510);
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
    assert!(size_of::<RgfxRenderPipelineCreate>() == 496);
    assert!(core::mem::offset_of!(RgfxRenderPipelineCreate, colors) == 128);
    assert!(size_of::<RgfxComputePipelineCreate>() == 96);
    assert!(size_of::<RgfxPipelineLibrary>() == 16);
    assert!(size_of::<RgfxDynamicLibraryCreate>() == 32);
    assert!(size_of::<RgfxPipelineReflection>() == 80);
    assert!(size_of::<RgfxMeshRenderPipelineCreate>() == 512);
    assert!(core::mem::offset_of!(RgfxMeshRenderPipelineCreate, colors) == 192);
    assert!(RGFX_PIPELINE_REFLECTION_MESH_OFFSET as usize == size_of::<RgfxPipelineReflection>());
    assert!(size_of::<RgfxMeshPipelineReflection>() == 24);
    // Each exact size above is a multiple of 8, as a ring message must be.
    // The stage-3 opcodes do not collide with the stage-2 ones or each other.
    assert!(RGFX_CMD_QUEUE_CREATE > RGFX_EVT_POINTER_MOTION);
    assert!(RGFX_EVT_POINTER_MOTION > RGFX_EVT_POINTER_OVER);
    assert!(RGFX_CMD_COMMAND_BUFFER > RGFX_CMD_FENCE_CREATE);
    assert!(RGFX_EVT_CB_COMPLETE > RGFX_CMD_COMMAND_BUFFER_IN_REGION);
    assert!(RGFX_CMD_COMMAND_BUFFER_IN_REGION > RGFX_CMD_COMMAND_BUFFER);
    assert!(RGFX_CMD_COMMAND_BUFFER_PART > RGFX_CMD_COMMAND_BUFFER_IN_REGION);
    assert!(RGFX_CMD_SCHEDULED_ASK > RGFX_CMD_COMMAND_BUFFER_PART);
    assert!(RGFX_EVT_CB_COMPLETE > RGFX_CMD_SCHEDULED_ASK);
    assert!(RGFX_CMD_BUFFER_OVER_PAGES > RGFX_CMD_VIDEO_SESSION_CREATE);
    assert!(RGFX_CMD_BUFFER_OVER_PAGES < RGFX_CMD_CONTEXT_CLOSE);
};

// The doorbell pages: a BAR is a power of two; each page is a whole page of every guest's, so a
// process's mapping of its own reaches no other; a bind's page field names every page; and the
// bind register is one of BAR 0's, below the MSI-X table and apart from the others.
const _: () = {
    assert!(RGFX_DOORBELLS_SIZE.is_power_of_two());
    assert!(RGFX_DOORBELL_PAGE_SIZE.is_power_of_two() && RGFX_DOORBELL_PAGE_SIZE >= 0x4000);
    assert!(RGFX_DOORBELL_PAGES <= 1 << RGFX_DOORBELL_PAGE_BIND_CONTEXT_SHIFT);
    assert!(RGFX_DOORBELLS_SIZE == RGFX_DOORBELL_PAGES * RGFX_DOORBELL_PAGE_SIZE);
    assert!(RGFX_REG_DOORBELL_PAGE_BIND > RGFX_REG_HOST_TIME_HI);
    assert!(RGFX_REG_DOORBELL_PAGE_BIND < RGFX_MSIX_OFFSET);
    assert!(RGFX_PCI_BAR_DOORBELLS > RGFX_PCI_BAR_REGION + 1);
    assert!(RGFX_FEATURE_DOORBELL_PAGES == 1 << 46);
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

        // A mesh create head, its colours at their offset, and its reply with the mesh tail.
        let mesh = RgfxMeshRenderPipelineCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_MESH_RENDER_PIPELINE_CREATE,
                size: size_of::<RgfxMeshRenderPipelineCreate>() as u32,
                seqno: 11,
            },
            obj_id: 0x32,
            mesh_library_obj_id: 0x10,
            mesh_name_offset: size_of::<RgfxMeshRenderPipelineCreate>() as u32,
            mesh_name_len: 4,
            flags: RGFX_RENDER_PIPELINE_RASTERIZATION_ENABLED
                | RGFX_MESH_PIPELINE_MESH_TG_MULTIPLE_OF_EXEC_WIDTH,
            max_total_threads_per_mesh_threadgroup: 128,
            required_threads_per_mesh_threadgroup: [4, 1, 1],
            color_attachment_count: 1,
            colors: core::array::from_fn(|i| RgfxColorAttachmentDesc {
                pixel_format: if i == 0 { 80 } else { 0 },
                ..RgfxColorAttachmentDesc::default()
            }),
            ..RgfxMeshRenderPipelineCreate::default()
        };
        let bytes = mesh.as_bytes();
        assert_eq!(&bytes[192..196], &80u32.to_le_bytes());
        assert_eq!(
            RgfxMeshRenderPipelineCreate::read_from_bytes(bytes).unwrap(),
            mesh
        );
        let tail = RgfxMeshPipelineReflection {
            max_total_threads_per_object_threadgroup: 1024,
            max_total_threads_per_mesh_threadgroup: 128,
            object_thread_execution_width: 32,
            mesh_thread_execution_width: 32,
            max_total_threadgroups_per_mesh_grid: 1024,
            reserved_14: 0,
        };
        let mut reply = RgfxPipelineReflection {
            kind: RGFX_PIPELINE_KIND_MESH,
            ..reflection
        };
        reply.header.size = RGFX_PIPELINE_REFLECTION_MESH_OFFSET + size_of_val(&tail) as u32;
        let mut msg = reply.as_bytes().to_vec();
        msg.extend_from_slice(tail.as_bytes());
        assert_eq!(msg.len(), 104);
        let (head, rest) = RgfxPipelineReflection::read_from_prefix(&msg).unwrap();
        assert_eq!(head, reply);
        assert_eq!(
            RgfxMeshPipelineReflection::read_from_bytes(rest).unwrap(),
            tail
        );
    }
}
