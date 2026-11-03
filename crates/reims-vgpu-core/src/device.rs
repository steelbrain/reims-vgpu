//! The device: the host's handle ([`Device`]), its retain count ([`Handle`]) and its state
//! ([`DeviceCore`]).
//!
//! ## Two-tier lifetime
//!
//! The host holds a [`Device`], an `Arc<Handle>`; each [`Device::clone_ref`] is a retain. When
//! the last [`Handle`] goes, its drop runs [`DeviceCore::shutdown`] while the core is still
//! alive:
//! - the device's FIFOs, tasks, displays and the EFI display hold the core unretained
//!   (`Weak<DeviceCore>` through the `seams::links` aliases), and those links still upgrade
//!   during the shutdown, so a late stamp still raises its interrupt and the FIFO queue still
//!   runs the teardown's blocks;
//! - weak loads ([`DeviceCore::retain_self`], the reporting handler's) go through the
//!   `Weak<Handle>` and answer nothing from the moment the shutdown starts;
//! - mappings reach the device's unmap path through [`MappingSink`], which outlives the core's
//!   own fields, so a mapping released while the core's fields drop still unmaps.
//!
//! The core's fields drop when the last `Arc<DeviceCore>` goes, in the order its explicit
//! `Drop` takes them, which is the order the host sees their release callbacks in
//! (architecture 15.3).
//!
//! ## Locks
//!
//! Each mutex guards its own fields, and nothing more. The fields the device also reads
//! without a lock (the root FIFO's `enabled`, the EFI registers, the root node, the root page, the
//! root FIFO, the guest displays and clients) are staged in a [`StagedMutex`]: unlocked readers see
//! a guard's stores at its drop or at its explicit publish, never store by store.
//!
//! The device always answers that protected content is enabled, whatever the descriptor's
//! `enableProtectedContent` (`tests::protected_content_ignored`). `addTraceRange` without
//! `removeTraceRange` refuses the device at creation; a lone `removeTraceRange` is dropped
//! (`tests::add_trace_without_remove_is_refused`, `tests::remove_trace_alone_is_dropped`). Argument
//! buffers need isolation, detach backing, the descriptor's `enableArgumentBuffers` and a device
//! not named "Apple Paravirtual device"
//! (`tests::argument_buffers_need_isolation_and_the_descriptor`).

pub mod config;
pub mod displays;
pub mod fifos;
pub mod lifecycle;
pub mod mmio;
pub mod root;
pub mod tasks;
pub mod waits;

use std::sync::atomic::Ordering;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU8, AtomicU32, AtomicU64, AtomicUsize};
use std::sync::{Arc, Weak};

use parking_lot::{Condvar, Mutex, ReentrantMutex};

use crate::collections::{BucketOrderMap, FirstFitVa};
use crate::device::config::{DeviceConfig, MAX_DISPLAY_PORT_COUNT};
use crate::device::displays::DisplayRegistry;
use crate::display::Display;
use crate::display::efi::EfiDisplay;
use crate::display::guest_display::GuestDisplay;
use crate::fifo::Fifo;
use crate::guest::GuestPtr;
use crate::guest::abi::device_info::DeviceInfoSnapshot;
use crate::guest::records::SharedFeatures;
use crate::host::{HostBlocks, HostCallbacks, HostPtr, HostTask};
use crate::iosurface::HostDeviceCore;
use crate::memory::map::MemoryMap;
use crate::memory::mapping::Mapping;
use crate::num::ToUsize;
use crate::platform::Handle as _;
use crate::platform::Platform;
use crate::platform::device::GpuDevice;
use crate::platform::resource::Library;
use crate::platform::types::NsErrorText;
use crate::raise::RaiseResult;
use crate::raise::death::DeathKind;
use crate::raise::death::DeathSite;
use crate::raise::{OnFault, Raise};
use crate::rt::Hold;
use crate::rt::pool::Pool;
use crate::rt::queue::{Executor, SerialQueue};
use crate::rt::signal::CoalescingSignal;
use crate::sync::{Staged, StagedMutex};
use crate::tasks::mapping_task::MappingTask;
use crate::tasks::{Task, TaskInfo};

/// The root task's address space (16 MiB).
pub const ROOT_TASK_SIZE: u64 = 0x100_0000;

/// The root task's virtual address space: its last byte and its granule.
const ROOT_TASK_VA_LAST: u64 = 0xff_ffff;
const ROOT_TASK_VA_GRANULE: u64 = 0x4000;

/// The stamp waiters' slots, one per FIFO stamp index.
const WAITING_INFO_COUNT: usize = 64;

/// The child FIFO slots.
const CHILD_FIFO_SLOTS: usize = 64;

/// The device's suspend and pause state. Its value is guest-visible (MMIO slot 86).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunState {
    /// Running: interrupts are raised. Name is ours: the state at creation, after a resume and
    /// after an unpause.
    Running = 0,
    /// A suspend has started: interrupts are held.
    EnteringSuspend = 1,
    /// Suspended. Name is ours: the state `finish_suspend` leaves.
    Suspended = 2,
    /// A resume is restoring state: every mapping path dies.
    LeavingSuspendEarly = 3,
    /// Resumed, rebuilding. Name is ours: the state `did_resume` holds while it rebuilds.
    Resuming = 4,
    /// Paused: interrupts are held.
    Paused = 5,
}

impl RunState {
    /// The state's value, as the guest reads it.
    pub fn code(self) -> u32 {
        u32::from(self.raw())
    }

    fn raw(self) -> u8 {
        match self {
            RunState::Running => 0,
            RunState::EnteringSuspend => 1,
            RunState::Suspended => 2,
            RunState::LeavingSuspendEarly => 3,
            RunState::Resuming => 4,
            RunState::Paused => 5,
        }
    }

    fn from_raw(raw: u8) -> RunState {
        match raw {
            1 => RunState::EnteringSuspend,
            2 => RunState::Suspended,
            3 => RunState::LeavingSuspendEarly,
            4 => RunState::Resuming,
            5 => RunState::Paused,
            _ => RunState::Running,
        }
    }
}

/// The run state as a relaxed atomic: every thread reads and writes it without a lock.
#[derive(Debug)]
pub struct RunStateCell(AtomicU8);

impl RunStateCell {
    fn new(state: RunState) -> RunStateCell {
        RunStateCell(AtomicU8::new(state.raw()))
    }

    /// The state now.
    pub fn load(&self) -> RunState {
        RunState::from_raw(self.0.load(Ordering::Relaxed))
    }

    /// Sets the state.
    pub fn store(&self, state: RunState) {
        self.0.store(state.raw(), Ordering::Relaxed);
    }
}

/// Why [`Device::new`] refused a config: each is one check, made where the device makes it, in
/// this order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceRefusal {
    /// A display port count of 0, or above [`MAX_DISPLAY_PORT_COUNT`].
    DisplayPortCount,
    /// No Metal device.
    NoMetalDevice,
    /// A memory map beside any task or memory block.
    MemoryConflict,
    /// The memory map refused its descriptor.
    MemoryMap,
    /// `addTraceRange` without `removeTraceRange`.
    TraceWithoutRemove,
    /// Metal refused the blit pipelines.
    Metal,
    /// Process isolation without a memory map.
    IsolationWithoutMap,
}

/// The device's deaths: each ends the process (architecture 6.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceDeath {
    /// A host block the device calls without a nil check is nil: the block's name.
    NilBlock(&'static str),
    /// A mapping was made or released while a resume restores state.
    MappingWhileLeavingSuspend,
    /// A suspend began with a FIFO in child slot 0, which only the root FIFO's id names.
    ChildFifoInSlotZero,
    /// A suspend was finished without having begun.
    SuspendNotBegun,
    /// A resume whose registers enable the root FIFO could not create it.
    ResumedRootFifoMissing,
    /// A resume's walk of the child FIFO nodes left the root page (`maxFIFOCount` 0).
    ResumeWalkLeftRootPage,
    /// A resume read the root node through a near-NULL address.
    RootNodeFault,
    /// A device info pair stored into a page that is not mapped.
    DeviceInfoWrite,
    /// A task's root word written through a missing root node (never reached: the root node is
    /// checked first).
    TaskRootWord,
    // The register paths' (`device/{mmio,root,fifos,waits}.rs`).
    /// A nonzero write to the FIFO enable register while the root FIFO is enabled.
    FifoAlreadyEnabled,
    /// The root FIFO's teardown found a child in slot 0.
    ChildInSlotZero,
    /// The root is torn down with no root node.
    TeardownWithoutRootNode,
    /// The root task is torn down with no `destroyTask` block.
    TeardownWithoutDestroyTask,
    /// A range mapping whose total length is not a whole number of pages.
    UnalignedRangeMapping,
    /// A root-task mapping released at an address outside the root task.
    UnmapOutsideRootTask,
    /// A physical read copies out of a mapping that is not there.
    PhysicalReadCopy,
    /// A FIFO id past the child FIFO slots.
    PastChildFifos,
    /// A FIFO or stamp index past the waiters' entries.
    PastWaitingInfo,
    /// A child FIFO is created while its entry still waits.
    CreateWhileWaiting,
    /// A stamp written with no stamp memory (no root FIFO).
    StampWriteWithoutMemory,
    /// A stamp read by a wait with no stamp memory.
    StampReadWithoutMemory,
    /// A stamp read by the timeout report with no stamp memory.
    ReportWithoutStampMemory,
    /// A deleted child's node word written with no root node.
    DeletedNodeWithoutRoot,
    /// A waiter at or past `maxFIFOCount`.
    WaiterPastMax,
    /// A waiter whose bit is already set.
    WaiterBitTaken,
    /// A waiter whose entry still waits.
    WaiterBusy,
    /// The timeout report found a stamp index past the FIFOs.
    ReportStampPastFifos,
    /// The timeout report found stamp 0 waiting with no root FIFO.
    ReportRootWithoutFifo,
    /// The timeout report found a child under another stamp index.
    ReportStampMismatch,
    /// The timeout report ran off the reporting queue.
    ReportOffQueue,
}

impl DeviceDeath {
    /// The site a death of this kind records.
    pub(crate) fn site(self) -> DeathSite {
        let (kind, file) = match self {
            DeviceDeath::NilBlock(block) => (DeathKind::NilBlock, block),
            DeviceDeath::MappingWhileLeavingSuspend => {
                (DeathKind::Assert, "device.mapping_while_leaving_suspend")
            }
            DeviceDeath::ChildFifoInSlotZero => {
                (DeathKind::Assert, "device.child_fifo_in_slot_zero")
            }
            DeviceDeath::SuspendNotBegun => (DeathKind::Assert, "device.suspend_not_begun"),
            DeviceDeath::ResumedRootFifoMissing => {
                (DeathKind::Assert, "device.resumed_root_fifo_missing")
            }
            DeviceDeath::ResumeWalkLeftRootPage => {
                (DeathKind::OutOfBounds, "device.resume_walk_left_root_page")
            }
            DeviceDeath::RootNodeFault => (DeathKind::NullDeref, "device.root_node_fault"),
            DeviceDeath::DeviceInfoWrite => (DeathKind::NullDeref, "device.device_info_write"),
            DeviceDeath::TaskRootWord => (DeathKind::NullDeref, "device.task_root_word"),
            other => (other.register_kind(), other.register_name()),
        };
        DeathSite {
            kind,
            file,
            line: 0,
        }
    }
}

impl std::fmt::Display for DeviceDeath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceDeath::NilBlock(block) => write!(f, "device.nil_block.{block}"),
            DeviceDeath::MappingWhileLeavingSuspend => {
                f.write_str("device.mapping_while_leaving_suspend")
            }
            DeviceDeath::ChildFifoInSlotZero
            | DeviceDeath::SuspendNotBegun
            | DeviceDeath::ResumedRootFifoMissing
            | DeviceDeath::ResumeWalkLeftRootPage
            | DeviceDeath::RootNodeFault
            | DeviceDeath::DeviceInfoWrite
            | DeviceDeath::TaskRootWord => f.write_str(self.site().file),
            other => f.write_str(other.register_name()),
        }
    }
}

/// The device's raises, the `NSException` channel as a value (architecture 6.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceRaise {
    /// A child FIFO id of 0, or at or past `maxFIFOCount`.
    InvalidFifoId,
    /// A stamp barrier's index at or past `maxFIFOCount`.
    InvalidStampIndex,
    /// A stamp wait that closes a cycle of waits.
    CircularWait,
    /// A physical read past the end of its page.
    ReadPastPage,
    /// A task lookup of an id the table does not hold.
    InvalidTask,
    /// A user-task lookup of task 0.
    UserTaskZero,
    /// Task 0 found remote where the kernel task is looked up (never: task 0 is always local).
    KernelTaskNotLocal,
    /// A task id at or past the table's size, to create or delete.
    InvalidTaskId,
    /// A task created over one that exists.
    TaskExists,
    /// A local task larger than the largest.
    TaskTooLarge,
    /// A task whose build refused.
    TaskNotCreated,
    /// The device info's destination page that no mapping covers.
    DeviceInfoNotMapped,
}

impl DeviceRaise {
    /// The raise's dotted name.
    pub const fn name(self) -> &'static str {
        match self {
            DeviceRaise::InvalidFifoId => "device.fifo.invalid_id",
            DeviceRaise::InvalidStampIndex => "device.wait.invalid_stamp_index",
            DeviceRaise::CircularWait => "device.wait.circular",
            DeviceRaise::ReadPastPage => "device.read.past_page",
            DeviceRaise::InvalidTask => "device.task.invalid",
            DeviceRaise::UserTaskZero => "device.task.user_zero",
            DeviceRaise::KernelTaskNotLocal => "device.task.kernel_not_local",
            DeviceRaise::InvalidTaskId => "device.task.invalid_id",
            DeviceRaise::TaskExists => "device.task.exists",
            DeviceRaise::TaskTooLarge => "device.task.too_large",
            DeviceRaise::TaskNotCreated => "device.task.not_created",
            DeviceRaise::DeviceInfoNotMapped => "device.info.not_mapped",
        }
    }
}

impl From<DeviceRaise> for crate::raise::RaiseId {
    fn from(k: DeviceRaise) -> crate::raise::RaiseId {
        crate::raise::RaiseId::Device(k)
    }
}

/// The root FIFO's registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[expect(missing_docs, reason = "the registers' names")]
pub struct FifoRegisters {
    pub enabled: u32,
    pub base_page: u32,
    pub length: u32,
    pub start: u32,
    pub virtual_offset: u64,
    pub read: u32,
    pub written: u32,
    pub fault_offset: u32,
}

/// The EFI display's registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[expect(missing_docs, reason = "the registers' names")]
pub struct EfiRegisters {
    pub enabled: u32,
    pub start: u32,
    pub length: u32,
    pub depth: u32,
    pub mode: u32,
    pub stride: u32,
}

/// One stamp index's waiter.
#[derive(Debug, Default)]
pub struct WaitingInfo {
    /// The next waiting FIFO's index (`u32::MAX` for none).
    pub next_waiting_index: AtomicU32,
    /// The value it waits for.
    pub next_waiting_value: AtomicU32,
    /// The last pending stamp (shared with the FIFO's stamp queue).
    pub last_pending_stamp: Arc<AtomicU32>,
}

/// A pointer field kept in an `AtomicUsize` as its address, 0 being none.
fn guest_ptr_from_addr(addr: usize) -> Option<GuestPtr> {
    (addr != 0).then(|| GuestPtr::new(HostPtr::from_addr(addr)))
}

/// A `GuestPtr` field as [`guest_ptr_from_addr`] reads it back.
fn guest_ptr_addr(ptr: Option<GuestPtr>) -> usize {
    ptr.map_or(0, |p| p.addr().addr())
}

/// The root task lock's fields.
pub struct RootTaskLocked {
    /// The root page number (staged: [`RootTaskAtomics::root_page_number`]).
    pub root_page_number: u32,
    /// The root task's virtual address space.
    pub root_task_allocator: Option<FirstFitVa>,
}

/// The root task's fields read or written without its lock.
#[derive(Debug, Default)]
pub struct RootTaskAtomics {
    /// The root page number (MMIO reads and the root page write take no lock).
    pub root_page_number: AtomicU32,
    /// The host's root task (0 for none), stored under the root node lock only.
    pub root_task: AtomicUsize,
    /// The root task's base address: set by the `createTask` block, cleared under the root node
    /// lock, read before the lock by the unmap path.
    pub root_task_base: AtomicUsize,
    /// The read page (cleared by the root teardown; nothing reads it).
    pub read_page: AtomicUsize,
}

impl RootTaskAtomics {
    /// The root task and its base now.
    pub fn task_info(&self) -> TaskInfo {
        TaskInfo {
            task: HostTask::from_addr(self.root_task.load(Ordering::Relaxed)),
            base_address: HostPtr::from_addr(self.root_task_base.load(Ordering::Relaxed)),
        }
    }
}

impl Staged for RootTaskLocked {
    type Atomics = RootTaskAtomics;
    type Snapshot = u32;
    fn new_atomics(&self) -> RootTaskAtomics {
        RootTaskAtomics {
            root_page_number: AtomicU32::new(self.root_page_number),
            ..RootTaskAtomics::default()
        }
    }
    fn pull(&mut self, a: &RootTaskAtomics) {
        self.root_page_number = a.root_page_number.load(Ordering::Relaxed);
    }
    fn snapshot(&self) -> u32 {
        self.root_page_number
    }
    fn publish(&self, a: &RootTaskAtomics, since: u32) {
        if self.root_page_number != since {
            a.root_page_number
                .store(self.root_page_number, Ordering::Relaxed);
        }
    }
}

/// The root node lock's fields.
pub struct RootNodeLocked<P: Platform> {
    /// The root node's mapping.
    pub root_node_mapping: Option<Arc<Mapping<P>>>,
    /// The root node (staged: [`RootNodeAtomics::root_node`]).
    pub root_node: Option<GuestPtr>,
}

/// The root node as it is read without its lock.
#[derive(Debug, Default)]
pub struct RootNodeAtomics {
    /// The root node's address, 0 for none.
    pub root_node: AtomicUsize,
}

impl RootNodeAtomics {
    /// The root node now.
    pub fn root_node(&self) -> Option<GuestPtr> {
        guest_ptr_from_addr(self.root_node.load(Ordering::Relaxed))
    }
}

impl<P: Platform> Staged for RootNodeLocked<P> {
    type Atomics = RootNodeAtomics;
    type Snapshot = Option<GuestPtr>;
    fn new_atomics(&self) -> RootNodeAtomics {
        RootNodeAtomics {
            root_node: AtomicUsize::new(guest_ptr_addr(self.root_node)),
        }
    }
    fn pull(&mut self, a: &RootNodeAtomics) {
        self.root_node = a.root_node();
    }
    fn snapshot(&self) -> Option<GuestPtr> {
        self.root_node
    }
    fn publish(&self, a: &RootNodeAtomics, since: Option<GuestPtr>) {
        if self.root_node != since {
            a.root_node
                .store(guest_ptr_addr(self.root_node), Ordering::Relaxed);
        }
    }
}

/// The root FIFO lock's fields.
pub struct RootFifoLocked<P: Platform> {
    /// The root FIFO (staged: [`RootFifoAtomics::root_fifo`]).
    pub root_fifo: Option<Arc<Fifo<P>>>,
    /// The root FIFO's mapping.
    pub root_fifo_mapping: Option<Arc<Mapping<P>>>,
    /// The root FIFO's registers (`enabled` staged: [`RootFifoAtomics::enabled`]).
    pub fifo: FifoRegisters,
}

/// The root FIFO's `enabled` register and the root FIFO as they are read without its lock.
pub struct RootFifoAtomics<P: Platform> {
    /// The `enabled` register.
    pub enabled: AtomicU32,
    /// The root FIFO, unretained, as unlocked sends see it (a released FIFO answers as none). A
    /// leaf lock around the `Weak`'s copy only, never held across anything else, so it never
    /// waits on the device's locks.
    root_fifo: Mutex<Weak<Fifo<P>>>,
}

impl<P: Platform> RootFifoAtomics<P> {
    /// The root FIFO now, retained for the send.
    pub fn root_fifo(&self) -> Option<Arc<Fifo<P>>> {
        let weak = self.root_fifo.lock().clone();
        weak.upgrade()
    }

    /// The `enabled` register now.
    pub fn enabled(&self) -> u32 {
        self.enabled.load(Ordering::Relaxed)
    }
}

/// A FIFO's identity for [`Staged::snapshot`].
fn fifo_addr<P: Platform>(fifo: Option<&Arc<Fifo<P>>>) -> usize {
    fifo.map_or(0, |f| Arc::as_ptr(f).addr())
}

impl<P: Platform> Staged for RootFifoLocked<P> {
    type Atomics = RootFifoAtomics<P>;
    type Snapshot = (u32, usize);
    fn new_atomics(&self) -> RootFifoAtomics<P> {
        RootFifoAtomics {
            enabled: AtomicU32::new(self.fifo.enabled),
            root_fifo: Mutex::new(
                self.root_fifo
                    .as_ref()
                    .map_or_else(Weak::new, Arc::downgrade),
            ),
        }
    }
    fn pull(&mut self, a: &RootFifoAtomics<P>) {
        self.fifo.enabled = a.enabled();
    }
    fn snapshot(&self) -> (u32, usize) {
        (self.fifo.enabled, fifo_addr(self.root_fifo.as_ref()))
    }
    fn publish(&self, a: &RootFifoAtomics<P>, since: (u32, usize)) {
        let (enabled, fifo) = self.snapshot();
        if enabled != since.0 {
            a.enabled.store(enabled, Ordering::Relaxed);
        }
        if fifo != since.1 {
            let weak = self
                .root_fifo
                .as_ref()
                .map_or_else(Weak::new, Arc::downgrade);
            *a.root_fifo.lock() = weak;
        }
    }
}

/// The display lock's fields.
pub struct DisplaysLocked<P: Platform> {
    /// The ports a display is attached or parked on.
    pub display_port_busy_mask: u64,
    /// The guest displays by port (staged: [`DisplaysAtomics::guest_display`]).
    pub guest_display: [Option<Arc<GuestDisplay<P>>>; 8],
    /// The displays by port, unretained (staged: [`DisplaysAtomics::display_client`]).
    pub display_client: [Option<Weak<Display<P>>>; 8],
    /// The EFI display as the guard found it. The device stores it without the lock, so its
    /// reference lives in [`DisplaysAtomics::efi_display`]; this is the guard's copy, released
    /// with the guard, and a store here reaches the atomics when the guard publishes.
    pub efi_display: Option<Arc<EfiDisplay<P>>>,
    /// The EFI registers (staged: [`DisplaysAtomics`]).
    pub efi: EfiRegisters,
}

/// The display lock's fields as they are read (or written) without it: the EFI registers (MMIO
/// reads, the slot aliases), the guest displays (the display teardown, pause, unpause, suspend and
/// resume), the clients and the EFI display itself. Each slot below is a leaf lock around its copy
/// alone, never held across anything else, so an unlocked reader never waits on the display lock.
#[expect(missing_docs, reason = "the EFI registers' names")]
pub struct DisplaysAtomics<P: Platform> {
    pub enabled: AtomicU32,
    pub start: AtomicU32,
    pub length: AtomicU32,
    pub depth: AtomicU32,
    pub mode: AtomicU32,
    pub stride: AtomicU32,
    /// The guest displays by port, unretained: `None` for none, and a released guest display
    /// answers none.
    guest_display: [Mutex<Option<Weak<GuestDisplay<P>>>>; 8],
    /// The displays by port, unretained.
    display_client: [Mutex<Option<Weak<Display<P>>>>; 8],
    /// The EFI display: the device's reference, stored without the display lock.
    efi_display: Mutex<Option<Arc<EfiDisplay<P>>>>,
}

impl<P: Platform> DisplaysAtomics<P> {
    /// Whether port `port` has a guest display now, without taking a reference.
    pub fn has_guest_display(&self, port: u64) -> bool {
        self.guest_display
            .get(port.to_usize())
            .is_some_and(|slot| slot.lock().is_some())
    }

    /// Port `port`'s guest display now, retained for the send (`None` for none, and for a guest
    /// display whose last release has run).
    pub fn guest_display(&self, port: u64) -> Option<Arc<GuestDisplay<P>>> {
        let weak = self.guest_display.get(port.to_usize())?.lock().clone();
        weak.and_then(|w| w.upgrade())
    }

    /// Port `port`'s display now (unretained).
    pub fn display_client(&self, port: u64) -> Option<Weak<Display<P>>> {
        self.display_client.get(port.to_usize())?.lock().clone()
    }

    /// Whether the device has an EFI display now, without taking a reference.
    pub fn has_efi_display(&self) -> bool {
        self.efi_display.lock().is_some()
    }

    /// The EFI display now, retained for the send.
    pub fn efi_display(&self) -> Option<Arc<EfiDisplay<P>>> {
        self.efi_display.lock().clone()
    }

    /// Replaces the EFI display without the display lock, and hands back the previous
    /// reference for the caller to release (or keep) once the cell is let go.
    pub fn replace_efi_display(
        &self,
        efi: Option<Arc<EfiDisplay<P>>>,
    ) -> Option<Arc<EfiDisplay<P>>> {
        std::mem::replace(&mut *self.efi_display.lock(), efi)
    }
}

/// [`DisplaysLocked`]'s staged fields as a guard found them: the EFI registers, and the
/// identities of the guest displays, the clients and the EFI display.
#[derive(Clone, Copy)]
pub struct DisplaysSnapshot {
    efi: EfiRegisters,
    guest_displays: [usize; 8],
    clients: [usize; 8],
    efi_display: usize,
}

impl<P: Platform> Staged for DisplaysLocked<P> {
    type Atomics = DisplaysAtomics<P>;
    type Snapshot = DisplaysSnapshot;
    fn new_atomics(&self) -> DisplaysAtomics<P> {
        let e = self.efi;
        DisplaysAtomics {
            enabled: AtomicU32::new(e.enabled),
            start: AtomicU32::new(e.start),
            length: AtomicU32::new(e.length),
            depth: AtomicU32::new(e.depth),
            mode: AtomicU32::new(e.mode),
            stride: AtomicU32::new(e.stride),
            guest_display: std::array::from_fn(|i| {
                Mutex::new(
                    self.guest_display
                        .get(i)
                        .and_then(Option::as_ref)
                        .map(Arc::downgrade),
                )
            }),
            display_client: std::array::from_fn(|i| {
                Mutex::new(self.display_client.get(i).cloned().flatten())
            }),
            efi_display: Mutex::new(self.efi_display.clone()),
        }
    }
    fn pull(&mut self, a: &DisplaysAtomics<P>) {
        self.efi = EfiRegisters {
            enabled: a.enabled.load(Ordering::Relaxed),
            start: a.start.load(Ordering::Relaxed),
            length: a.length.load(Ordering::Relaxed),
            depth: a.depth.load(Ordering::Relaxed),
            mode: a.mode.load(Ordering::Relaxed),
            stride: a.stride.load(Ordering::Relaxed),
        };
        self.efi_display = a.efi_display();
    }
    fn snapshot(&self) -> DisplaysSnapshot {
        DisplaysSnapshot {
            efi: self.efi,
            guest_displays: self
                .guest_display
                .each_ref()
                .map(|n| n.as_ref().map_or(0, |n| Arc::as_ptr(n).addr())),
            clients: self
                .display_client
                .each_ref()
                .map(|c| c.as_ref().map_or(0, |c| Weak::as_ptr(c).addr())),
            efi_display: self
                .efi_display
                .as_ref()
                .map_or(0, |e| Arc::as_ptr(e).addr()),
        }
    }
    fn publish(&self, a: &DisplaysAtomics<P>, since: DisplaysSnapshot) {
        let now = self.snapshot();
        let (e, was) = (self.efi, since.efi);
        for (now, was, cell) in [
            (e.enabled, was.enabled, &a.enabled),
            (e.start, was.start, &a.start),
            (e.length, was.length, &a.length),
            (e.depth, was.depth, &a.depth),
            (e.mode, was.mode, &a.mode),
            (e.stride, was.stride, &a.stride),
        ] {
            if now != was {
                cell.store(now, Ordering::Relaxed);
            }
        }
        for (i, slot) in a.guest_display.iter().enumerate() {
            if now.guest_displays.get(i) != since.guest_displays.get(i) {
                let guest_display = self.guest_display.get(i).and_then(Option::as_ref);
                *slot.lock() = guest_display.map(Arc::downgrade);
            }
        }
        for (i, slot) in a.display_client.iter().enumerate() {
            if now.clients.get(i) != since.clients.get(i) {
                *slot.lock() = self.display_client.get(i).cloned().flatten();
            }
        }
        if now.efi_display != since.efi_display {
            let old = a.replace_efi_display(self.efi_display.clone());
            drop(old); // outside the cell: the guard's copy or the taker holds it still.
        }
    }
    fn unlocking(&mut self) {
        // The guard's copy of the EFI display: the device's reference is the atomics'.
        drop(self.efi_display.take());
    }
}

/// The part of the device a root task mapping or a task mapping unmaps through when it is
/// released. It outlives the core's other fields: the core holds it last, and mappings hold it
/// weakly, so a mapping released while the device shuts down or while the core's fields drop
/// still unmaps (architecture 15.2). Its fields are the core's own, shared.
pub struct MappingSink<P: Platform> {
    /// The device's host blocks. The device's last holder of them: their release here is in
    /// [`HostCallbacks`]' order.
    pub blocks: HostCallbacks,
    /// The device's memory map.
    pub memory_map: Arc<Mutex<Option<Arc<MemoryMap<P>>>>>,
    /// The device's run state.
    pub run_state: Arc<RunStateCell>,
    /// The device's root task lock and fields.
    pub root_task: Arc<StagedMutex<RootTaskLocked>>,
    /// The device's mapping tasks.
    pub mapping_tasks: Arc<Mutex<Vec<Arc<MappingTask<P>>>>>,
}

impl<P: Platform> MappingSink<P> {
    /// Whether a resume is restoring state (every mapping path dies then).
    pub(crate) fn leaving_suspend_early(&self) -> bool {
        self.run_state.load() == RunState::LeavingSuspendEarly
    }

    /// A task's unmap (a task mapping's release, and the device's own for its tasks): through
    /// its mapping task, or the host's `unmapMemory` block, whose answer this is.
    pub fn unmap_memory_in_task(&self, task: HostTask, offset: u64, length: u64) -> bool {
        crate::fatal_unless!(
            !self.leaving_suspend_early(),
            DeviceDeath::MappingWhileLeavingSuspend
        );
        if self.memory_map.lock().is_some() {
            let found = self
                .mapping_tasks
                .lock()
                .iter()
                .find(|t| mapping_task_id(t) == task)
                .cloned();
            return found.is_some_and(|t| t.unmap_memory_at_virtual_offset(offset, length));
        }
        let Some(unmap) = &self.blocks.unmap_memory else {
            crate::fatal!(DeviceDeath::NilBlock("unmapMemory"))
        };
        unmap(task, offset, length)
    }
}

impl<P: Platform> Drop for MappingSink<P> {
    /// The host blocks first (their `destroy` callbacks, in [`HostCallbacks`]' order), then the
    /// memory map, then the mapping tasks: the order the host sees their releases in.
    fn drop(&mut self) {
        drop(std::mem::take(&mut self.blocks));
        // Each taken out of its lock first, so its release runs with the lock free.
        let memory_map = self.memory_map.lock().take();
        drop(memory_map);
        let mapping_tasks = std::mem::take(&mut *self.mapping_tasks.lock());
        drop(mapping_tasks);
    }
}

/// A mapping task's identity as a task handle: the object's address. Name is ours.
pub fn mapping_task_id<P: Platform>(task: &Arc<MappingTask<P>>) -> HostTask {
    HostTask::from_addr(Arc::as_ptr(task).addr())
}

/// A raise that reaches a device entry point whose signature cannot carry it: the process dies
/// at the raise's own site, handler or not.
pub fn uncaught(raised: Raise) -> ! {
    raised.fatal()
}

/// A call whose raise nothing above it catches (see [`uncaught`]).
fn no_catch<T>(r: RaiseResult<T>) -> T {
    match r {
        Ok(v) => v,
        Err(t) => uncaught(t),
    }
}

/// A pipeline's refusal in a line: Metal's error, or that Metal gave none.
fn refusal_text(e: Option<&NsErrorText>) -> &str {
    e.map_or("Metal gave no error", |e| e.description.as_str())
}

/// The two blit pipelines, as the device makes them.
struct BlitPipelines<P: Platform> {
    /// Both were made.
    ok: bool,
    blit_in: Option<P::ComputePipeline>,
    blit_out: Option<P::ComputePipeline>,
}

/// The two blit pipelines from the platform's shader library. A missing library refuses
/// without a line, a refused pipeline logs Metal's error. A pipeline made before the refusal is
/// released with the refused device.
///
/// One real pool around the whole: what Metal autoreleases (the errors, the library, the
/// functions) drains before the device's creation goes on. A Metal raise here is uncaught and
/// ends the process at the send.
fn setup_blit_pipelines<P: Platform>(platform: &P, mtl: &P::Device) -> BlitPipelines<P> {
    // Popped at each return, after the library and the functions have gone.
    let _pool = Pool::push(platform);
    let refused = |blit_in| BlitPipelines {
        ok: false,
        blit_in,
        blit_out: None,
    };
    // Shared by the three calls: a call that answers nil without an error leaves the
    // previous one.
    let mut error: Option<NsErrorText> = None;
    let library = match mtl.new_shader_library() {
        Ok(Some(library)) => library,
        Ok(None) | Err(_) => return refused(None),
    };
    // The function is not checked: a missing one goes to Metal as nil.
    let function = no_catch(library.new_function(OnFault::Die, "BlitInRGB_2P_XR10_A8"));
    let blit_in = match mtl.new_compute_pipeline_state_for_function(function.as_ref()) {
        Ok(Some(p)) => p,
        failed => {
            if let Err(e) = failed {
                error = Some(e);
            }
            log::error!(
                "device not created: Metal refused the BlitInRGB_2P_XR10_A8 pipeline: {}",
                refusal_text(error.as_ref())
            );
            return refused(None);
        }
    };
    let function = no_catch(library.new_function(OnFault::Die, "BlitOutRGB_2P_XR10_A8"));
    let blit_out = match mtl.new_compute_pipeline_state_for_function(function.as_ref()) {
        Ok(Some(p)) => p,
        failed => {
            if let Err(e) = failed {
                error = Some(e);
            }
            log::error!(
                "device not created: Metal refused the BlitOutRGB_2P_XR10_A8 pipeline: {}",
                refusal_text(error.as_ref())
            );
            return refused(Some(blit_in));
        }
    };
    BlitPipelines {
        ok: true,
        blit_in: Some(blit_in),
        blit_out: Some(blit_out),
    }
}

/// A new command queue with its label, unshared ([`GpuDevice::new_command_queue_unshared`]): each
/// of the device's queues is on a present's path (their fields say how), so none shares its GPU
/// queue with another command queue, whose long command buffer would hold its own back. Never
/// checked, and the label a send to nil when Metal answered nil.
fn labelled_queue<P: Platform>(mtl: &P::Device, label: &str) -> Option<P::CommandQueue> {
    use crate::platform::command::CommandQueue;
    let queue = mtl.new_command_queue_unshared();
    crate::nil_send!(queue, set_label(label));
    queue
}

/// What the device supports, fixed at its creation.
#[derive(Debug, Clone, Copy)]
struct Capabilities {
    /// Tasks other than task 0 are isolated: each is run by a service of its own.
    process_isolation: bool,
    /// With `process_isolation`: only while the IOSurface device the device maps through has a
    /// descriptor table ([`DeviceConfig::process_isolation_follows_guest`]).
    isolation_follows_guest: bool,
    /// Always true, whatever the descriptor asks (module doc).
    protected_content: bool,
    shared_textures: bool,
    buffer_with_iosurface: bool,
    /// Whether the device has a memory map.
    large_user_tasks: bool,
    /// Whether the device has a memory map.
    large_kernel_tasks: bool,
    range_buffer: bool,
    shared_memory_heap: bool,
    /// The descriptor's feature level.
    feature_level: u64,
}

/// The device's state (module doc). The host reaches it through a [`Handle`]; FIFOs, tasks,
/// displays and mappings hold it unretained.
pub struct DeviceCore<P: Platform> {
    /// This core, unretained, for the links the device hands its FIFOs, tasks and displays.
    /// It upgrades through the shutdown, until the fields drop.
    weak_self: Weak<DeviceCore<P>>,
    /// The retain count, for weak loads: it stops upgrading when the shutdown starts.
    handle: Weak<Handle<P>>,
    /// The device's own recursive lock (the mapper service lookup).
    object_sync: ReentrantMutex<()>,
    platform: Arc<P>,
    /// Stamp slot writes, the interrupt stamp, the fault mask updates on the FIFO queue and the
    /// stamp waits; the host's `raiseInterrupt` runs under it.
    signal_mutex: Mutex<()>,
    signal_cond: Condvar,
    /// The tasks, in ascending task ID: every fan-out across tasks and the releases when the
    /// table empties follow it.
    tasks: Mutex<BucketOrderMap<Task<P>>>,
    /// What the device supports, fixed at creation.
    caps: Capabilities,
    display_port_count: AtomicU32,
    display_port_connection_types: AtomicU32,
    interrupt_display: AtomicU32,
    displays: StagedMutex<DisplaysLocked<P>>,
    interrupt_stamp: AtomicU32,
    stamp_memory: Mutex<Option<GuestPtr>>,
    /// Shared with [`MappingSink`].
    root_task: Arc<StagedMutex<RootTaskLocked>>,
    root_node: StagedMutex<RootNodeLocked<P>>,
    /// Read and written without a lock (the root teardown, FIFO creation, a contended task
    /// delete).
    tearing_down_root: AtomicBool,
    root_fifo: StagedMutex<RootFifoLocked<P>>,
    fifo_queue: SerialQueue,
    child_fifos: Mutex<Vec<Option<Arc<Fifo<P>>>>>,
    interrupt_fault_mask: AtomicU64,
    /// Never checked: `None` where Metal answered nil, and every send to it is `nil_send!`.
    /// Unshared ([`labelled_queue`]): a display's present pages the kernel task's backing it
    /// shows in on it.
    blit_up_queue: Option<P::CommandQueue>,
    /// Unshared ([`labelled_queue`]): an in-process task's execution syncs its resources on it
    /// before its command buffer runs, and it is that task's paging queue.
    blit_down_queue: Option<P::CommandQueue>,
    blit_in_rgb_2p_xr10_a8_pipeline: Option<P::ComputePipeline>,
    blit_out_rgb_2p_xr10_a8_pipeline: Option<P::ComputePipeline>,
    blocks: HostCallbacks,
    binary_version: AtomicI32,
    features: Arc<SharedFeatures>,
    /// Shared with [`MappingSink`].
    run_state: Arc<RunStateCell>,
    /// The process's IOSurface host device, unretained: found on the first lookup that finds
    /// the process's weak registration, and looked up again while still none. Written under
    /// `object_sync`, read without it; a leaf cell. It holds the host device's core, never the
    /// host's object.
    mapper_service: Mutex<Option<Weak<HostDeviceCore<P>>>>,
    /// The descriptor's `usingIOSurfaceMapper`.
    using_mapper_service: bool,
    waiting_stamps: AtomicU64,
    waiting_info: Vec<WaitingInfo>,
    reporting_queue: SerialQueue,
    /// A `DATA_ADD` source on the reporting queue whose handler reports stamp wait timeouts on
    /// the device it loads weakly; a stamp wait merges 1 into it.
    report_stamp_wait_timeout_source: CoalescingSignal,
    /// Shared with [`MappingSink`].
    memory_map: Arc<Mutex<Option<Arc<MemoryMap<P>>>>>,
    /// The mapping tasks made over the memory map, handed out as task handles
    /// ([`mapping_task_id`]); each is held here until its task is destroyed. Looked up only.
    /// Shared with [`MappingSink`].
    mapping_tasks: Arc<Mutex<Vec<Arc<MappingTask<P>>>>>,
    guest_device_info_max_key: AtomicI32,
    /// What the device offers isolated tasks: detach backing, isolation, a Metal device that is
    /// not paravirtual and the descriptor's `enableArgumentBuffers`. Device info answers it
    /// through [`Self::argument_buffers_answer`].
    supports_argument_buffers: AtomicBool,
    /// Where isolation follows the guest, the last argument-buffer answer device info logged: 0
    /// none yet, 1 off, 2 on.
    argument_buffers_logged: AtomicU8,
    mtl_device: P::Device,
    /// Unshared ([`labelled_queue`]): every in-process task's executions run on it, which is
    /// every task's but where isolation runs one apart, WindowServer's composites among them.
    exec_queue: Option<P::CommandQueue>,
    /// The displays' cores by the host reference's identity: a leaf lock, apart from `displays`.
    display_cores: DisplayRegistry<P>,
    /// What the device's mappings unmap through. Declared last of the fields the explicit
    /// `Drop` leaves: it outlives every other field.
    mapping_sink: Arc<MappingSink<P>>,
}

impl<P: Platform> DeviceCore<P> {
    /// Creates a device over `platform` from the host's descriptor (copied at the SDK call): the
    /// host's first retain, or the reason for a refusal.
    /// Every refusal made after something was built releases it, most recent first. The host's
    /// entry point is [`Device::new`].
    pub(crate) fn create(
        platform: Arc<P>,
        descriptor: DeviceConfig<P>,
    ) -> Result<Arc<Handle<P>>, DeviceRefusal> {
        let p = platform.as_ref();
        // A raise out of a host object here ends the process with every part built so far still
        // held. Each refusal releases the parts by name, most recent first.
        macro_rules! release {
            ($($part:ident),*) => { $(drop($part);)* };
        }
        if descriptor.display_port_count == 0
            || descriptor.display_port_count > MAX_DISPLAY_PORT_COUNT
        {
            return Err(DeviceRefusal::DisplayPortCount);
        }
        let display_port_count = descriptor.display_port_count;
        let display_port_connection_types = descriptor.display_port_connection_types;
        let mtl_device = descriptor
            .device
            .as_ref()
            .map(|d| d.clone_ref())
            .ok_or(DeviceRefusal::NoMetalDevice)?;
        // The Metal device's capabilities, read for the first time here.
        let mtl_caps = mtl_device.caps();
        let name = mtl_caps.name.as_deref();
        log::info!(
            "device starting on Metal device {}",
            name.unwrap_or("(unnamed)")
        );
        let paravirtual_device = name.is_some_and(|n| n.contains("Apple Paravirtual device"));
        let fifo_queue = SerialQueue::new(p.threads(), "reims-vgpu.fifo-control");
        mtl_device.allow_foreign_libraries();
        let blit_up_queue = labelled_queue::<P>(&mtl_device, "reims-vgpu.blit-up");
        let blit_down_queue = labelled_queue::<P>(&mtl_device, "reims-vgpu.blit-down");
        let exec_queue = labelled_queue::<P>(&mtl_device, "reims-vgpu.exec");
        let tasks = BucketOrderMap::new();
        let mut blocks = HostCallbacks::default();
        let host_blocks = &descriptor.blocks;
        let memory_map = if let Some(map_descriptor) = &descriptor.memory_map_descriptor {
            if host_blocks.create_task.is_some()
                || host_blocks.destroy_task.is_some()
                || host_blocks.map_memory.is_some()
                || host_blocks.unmap_memory.is_some()
                || host_blocks.read_memory.is_some()
            {
                log::warn!("device refused: a memory map was given beside a task or memory block");
                release!(
                    blocks,
                    tasks,
                    exec_queue,
                    blit_down_queue,
                    blit_up_queue,
                    fifo_queue,
                    mtl_device
                );
                return Err(DeviceRefusal::MemoryConflict);
            }
            let Some(map) = MemoryMap::new(platform.clone(), map_descriptor) else {
                log::warn!("device refused: its memory map descriptor is not valid");
                release!(
                    blocks,
                    tasks,
                    exec_queue,
                    blit_down_queue,
                    blit_up_queue,
                    fifo_queue,
                    mtl_device
                );
                return Err(DeviceRefusal::MemoryMap);
            };
            Some(map)
        } else {
            // Any may be nil; nothing checks them here. The platform may answer its own around
            // them.
            let kept = p.host_blocks(HostBlocks {
                create_task: host_blocks.create_task.clone(),
                destroy_task: host_blocks.destroy_task.clone(),
                map_memory: host_blocks.map_memory.clone(),
                unmap_memory: host_blocks.unmap_memory.clone(),
                read_memory: host_blocks.read_memory.clone(),
                ..HostBlocks::default()
            });
            blocks.create_task = kept.create_task;
            blocks.destroy_task = kept.destroy_task;
            blocks.map_memory = kept.map_memory;
            blocks.unmap_memory = kept.unmap_memory;
            blocks.read_memory = kept.read_memory;
            None
        };
        // May be nil: the first interrupt then dies.
        blocks.raise_interrupt = host_blocks.raise_interrupt.clone();
        if host_blocks.add_trace_range.is_some() {
            if host_blocks.remove_trace_range.is_none() {
                // Refused after this release.
                release!(
                    memory_map,
                    blocks,
                    tasks,
                    exec_queue,
                    blit_down_queue,
                    blit_up_queue,
                    fifo_queue,
                    mtl_device
                );
                return Err(DeviceRefusal::TraceWithoutRemove);
            }
            blocks.add_trace_range = host_blocks.add_trace_range.clone();
            blocks.remove_trace_range = host_blocks.remove_trace_range.clone();
        }
        let root_task_allocator = FirstFitVa::new(ROOT_TASK_VA_LAST, ROOT_TASK_VA_GRANULE);
        let using_mapper_service = descriptor.using_iosurface_mapper;
        let pipelines = setup_blit_pipelines(p, &mtl_device);
        let blit_in = pipelines.blit_in;
        let blit_out = pipelines.blit_out;
        if !pipelines.ok {
            release!(
                blit_out,
                blit_in,
                memory_map,
                blocks,
                tasks,
                exec_queue,
                blit_down_queue,
                blit_up_queue,
                fifo_queue,
                mtl_device
            );
            return Err(DeviceRefusal::Metal);
        }
        let enable_process_isolation = descriptor.enable_process_isolation;
        let isolation_follows_guest = descriptor.process_isolation_follows_guest;
        if enable_process_isolation {
            if isolation_follows_guest {
                log::info!(
                    "device isolates every task but task 0 while its IOSurface device has a \
                     descriptor table"
                );
            } else {
                log::info!("device isolates every task but task 0");
            }
        }
        let supports_shared_textures = mtl_caps.flags.supports_shared_texture_handles;
        let supports_buffer_with_iosurface = mtl_caps.flags.supports_buffer_with_iosurface;
        let supports_large_user_tasks = memory_map.is_some();
        let supports_large_kernel_tasks = memory_map.is_some();
        let supports_range_buffer = mtl_caps.flags.supports_buffer_with_address_ranges;
        let supports_shared_memory_heap = mtl_caps.flags.supports_heap_with_address_ranges;
        if enable_process_isolation && memory_map.is_none() {
            log::warn!("device refused: process isolation needs a memory map");
            release!(
                blit_out,
                blit_in,
                memory_map,
                blocks,
                tasks,
                exec_queue,
                blit_down_queue,
                blit_up_queue,
                fifo_queue,
                mtl_device
            );
            return Err(DeviceRefusal::IsolationWithoutMap);
        }
        let detach_backing = mtl_caps.flags.supports_resource_detach_backing;
        let supports_argument_buffers = detach_backing
            && enable_process_isolation
            && !paravirtual_device
            && descriptor.enable_argument_buffers;
        if detach_backing {
            log::info!(
                "argument buffers {}",
                if !supports_argument_buffers {
                    "off"
                } else if descriptor.process_isolation_follows_guest {
                    "on while the IOSurface device has a descriptor table"
                } else {
                    "on"
                }
            );
        }
        // The cells the core shares with its mapping sink.
        let run_state = Arc::new(RunStateCell::new(RunState::Running));
        let memory_map = Arc::new(Mutex::new(memory_map));
        let mapping_tasks = Arc::new(Mutex::new(Vec::new()));
        let root_task = Arc::new(StagedMutex::new(RootTaskLocked {
            root_page_number: 0,
            root_task_allocator: Some(root_task_allocator),
        }));
        let sink_blocks = blocks.clone_ref();
        let handle = Arc::new_cyclic(|weak_handle: &Weak<Handle<P>>| {
            let core = Arc::new_cyclic(|weak_core: &Weak<DeviceCore<P>>| {
                let reporting_queue = SerialQueue::new(p.threads(), "reims-vgpu.reporting");
                let target: Arc<dyn Executor> = Arc::new(SerialQueue::clone(&reporting_queue));
                let reporter = weak_handle.clone();
                let release_platform = Arc::clone(&platform);
                let report_stamp_wait_timeout_source =
                    CoalescingSignal::new_add(target, move |_| {
                        // A weak load: nothing once the shutdown has started. A host release that
                        // lands while the handler runs leaves the last reference to it, and the
                        // shutdown then runs after the handler, off the reporting queue: inside
                        // the handler its sync onto that queue would die.
                        let Some(loaded) = reporter.upgrade() else {
                            return;
                        };
                        loaded.report_stamp_wait_timeout();
                        crate::rt::held::release_off_thread(release_platform.threads(), loaded);
                    });
                // Every stamp waiter idle, no stamp pending.
                let waiting_info = (0..WAITING_INFO_COUNT)
                    .map(|_| WaitingInfo {
                        next_waiting_index: AtomicU32::new(u32::MAX),
                        next_waiting_value: AtomicU32::new(0),
                        last_pending_stamp: Arc::new(AtomicU32::new(0)),
                    })
                    .collect();
                let device_feature_level = descriptor.device_feature_level;
                crate::guest::abi::features::log_device_feature_level(device_feature_level);
                DeviceCore {
                    weak_self: weak_core.clone(),
                    handle: weak_handle.clone(),
                    object_sync: ReentrantMutex::new(()),
                    platform: platform.clone(),
                    signal_mutex: Mutex::new(()),
                    signal_cond: Condvar::new(),
                    tasks: Mutex::new(tasks),
                    caps: Capabilities {
                        process_isolation: enable_process_isolation,
                        isolation_follows_guest,
                        protected_content: true,
                        shared_textures: supports_shared_textures,
                        buffer_with_iosurface: supports_buffer_with_iosurface,
                        large_user_tasks: supports_large_user_tasks,
                        large_kernel_tasks: supports_large_kernel_tasks,
                        range_buffer: supports_range_buffer,
                        shared_memory_heap: supports_shared_memory_heap,
                        feature_level: device_feature_level,
                    },
                    display_port_count: AtomicU32::new(display_port_count),
                    display_port_connection_types: AtomicU32::new(display_port_connection_types),
                    interrupt_display: AtomicU32::new(0),
                    displays: StagedMutex::new(DisplaysLocked {
                        display_port_busy_mask: 0,
                        guest_display: Default::default(),
                        display_client: Default::default(),
                        efi_display: None,
                        efi: EfiRegisters::default(),
                    }),
                    interrupt_stamp: AtomicU32::new(0),
                    stamp_memory: Mutex::new(None),
                    root_task: root_task.clone(),
                    root_node: StagedMutex::new(RootNodeLocked {
                        root_node_mapping: None,
                        root_node: None,
                    }),
                    tearing_down_root: AtomicBool::new(false),
                    root_fifo: StagedMutex::new(RootFifoLocked {
                        root_fifo: None,
                        root_fifo_mapping: None,
                        fifo: FifoRegisters::default(),
                    }),
                    fifo_queue,
                    child_fifos: Mutex::new(vec![None; CHILD_FIFO_SLOTS]),
                    interrupt_fault_mask: AtomicU64::new(0),
                    blit_up_queue,
                    blit_down_queue,
                    blit_in_rgb_2p_xr10_a8_pipeline: blit_in,
                    blit_out_rgb_2p_xr10_a8_pipeline: blit_out,
                    blocks,
                    binary_version: AtomicI32::new(0),
                    features: Arc::new(SharedFeatures::new()),
                    run_state: run_state.clone(),
                    mapper_service: Mutex::new(None),
                    using_mapper_service,
                    waiting_stamps: AtomicU64::new(0),
                    waiting_info,
                    reporting_queue,
                    report_stamp_wait_timeout_source,
                    memory_map: memory_map.clone(),
                    mapping_tasks: mapping_tasks.clone(),
                    guest_device_info_max_key: AtomicI32::new(1),
                    supports_argument_buffers: AtomicBool::new(supports_argument_buffers),
                    argument_buffers_logged: AtomicU8::new(0),
                    mtl_device,
                    exec_queue,
                    display_cores: DisplayRegistry::new(),
                    mapping_sink: Arc::new(MappingSink {
                        blocks: sink_blocks,
                        memory_map,
                        run_state,
                        root_task,
                        mapping_tasks,
                    }),
                }
            });
            Handle { core }
        });
        Ok(handle)
    }
}

impl<P: Platform> DeviceCore<P> {
    /// A retain of the device, as a block copy or a weak load takes one: `None` once the shutdown
    /// has started (a retain then does nothing, and a weak load answers nil). When it is the last,
    /// its drop shuts the device down on that thread.
    pub fn retain_self(&self) -> Option<Device<P>> {
        self.handle.upgrade().map(Device)
    }

    /// Whether the shutdown has started (the last [`Handle`] has gone).
    pub(crate) fn is_deallocating_now(&self) -> bool {
        self.handle.strong_count() == 0
    }

    /// `body` under the device's own recursive lock.
    pub(crate) fn synchronized(&self, body: &mut dyn FnMut()) {
        let _sync = self.object_sync.lock();
        body();
    }

    /// The report budget: whether a resumed child FIFO still reports its timeouts, the process
    /// counting every resume that asks, across every device in the process.
    pub(crate) fn take_report_budget(&self) -> bool {
        self.platform.globals().take_report_budget()
    }

    /// An MMIO read at `offset`.
    pub fn mmio_read(self: &Arc<Self>, offset: u64) -> u32 {
        self.mmio_read_register(crate::device::mmio::register_index(offset))
    }

    /// The argument-buffer support device info answers, read now: what the device offers
    /// ([`Self::supports_argument_buffers`]), and, where isolation follows the guest, only while
    /// the IOSurface device it maps through has a descriptor table, as a task defined now would be
    /// isolated ([`Self::isolation_of`]): only isolated tasks serve argument buffers. Under that
    /// rule each change of answer logs one line with its reason.
    ///
    /// Its callers hold none of the device's locks: the mapper lookup takes the device's
    /// synchronized lock, and its reference to the IOSurface device may be the last.
    fn argument_buffers_answer(&self) -> bool {
        let offered = self.supports_argument_buffers.load(Ordering::Relaxed);
        if !(self.caps.process_isolation && self.caps.isolation_follows_guest) {
            return offered;
        }
        let base = self.mapper_service().map(|m| m.descriptor_base());
        let on = offered && base.is_some_and(|b| b != 0);
        let logged = if on { 2 } else { 1 };
        if self.argument_buffers_logged.swap(logged, Ordering::Relaxed) != logged {
            match base {
                _ if !offered => {
                    log::info!("device info: argument buffers off: the device offers none");
                }
                Some(base) if base != 0 => {
                    log::info!(
                        "device info: argument buffers on: the IOSurface device's descriptor \
                         table is at {base:#x}"
                    );
                }
                Some(_) => log::info!(
                    "device info: argument buffers off: the IOSurface device has no descriptor \
                     table"
                ),
                None => log::info!("device info: argument buffers off: no IOSurface device"),
            }
        }
        on
    }

    /// An MMIO write at `offset`, under the host's call's `hold`. May block: a root FIFO write
    /// waits for the FIFO thread.
    pub fn mmio_write(self: &Arc<Self>, hold: Hold<'_>, offset: u64, value: u32) {
        self.mmio_write_register(hold, crate::device::mmio::register_index(offset), value);
    }

    /// Whether a resume is restoring state: every mapping path dies then.
    pub(crate) fn leaving_suspend_early(&self) -> bool {
        self.run_state.load() == RunState::LeavingSuspendEarly
    }

    /// The device's side of the device info: its features and capabilities, read now.
    pub(crate) fn info_snapshot(&self) -> DeviceInfoSnapshot {
        DeviceInfoSnapshot {
            features: self.features.load(),
            device_feature_level: self.caps.feature_level,
            supports_argument_buffers: self.argument_buffers_answer(),
            guest_device_info_max_key: self.guest_device_info_max_key.load(Ordering::Relaxed),
            supports_buffer_with_iosurface: self.caps.buffer_with_iosurface,
            supports_shared_textures: self.caps.shared_textures,
            supports_large_user_tasks: self.caps.large_user_tasks,
            supports_large_kernel_tasks: self.caps.large_kernel_tasks,
            supports_range_buffer: self.caps.range_buffer,
            supports_shared_memory_heap: self.caps.shared_memory_heap,
        }
    }

    /// The last release: the EFI display tears down (it releases itself), the root is torn down as
    /// a root page of 0 does, and the reporting source is cancelled on the reporting queue (a
    /// shutdown on that queue is libdispatch's crash). May hang behind a FIFO's delay. The fields
    /// go when the last `Arc<DeviceCore>` does.
    fn shutdown(&self, hold: Hold<'_>) {
        let efi = self.displays.lock().efi_display.take();
        if let Some(efi) = efi {
            efi.teardown();
        }
        self.set_root_page(hold, 0);
        let source = self.report_stamp_wait_timeout_source.clone();
        self.reporting_queue.sync(move || source.cancel());
    }
}

impl<P: Platform> Drop for DeviceCore<P> {
    /// The core's parts whose release has effects, in the order the host sees: the tasks (in
    /// task ID order), the guest displays, clients and EFI display, the root node's mapping, the
    /// root FIFO and its mapping, the child FIFOs, the blit queues and pipelines, then the device's
    /// copies of the host blocks. The queues, the reporting source, the Metal device and the
    /// execution queue follow as fields; the mapping sink goes last, with the host blocks' last
    /// references, so every mapping released before it still unmaps.
    fn drop(&mut self) {
        drop(std::mem::take(self.tasks.get_mut()));
        {
            let (displays, atomics) = self.displays.get_mut();
            drop(std::mem::take(&mut displays.guest_display));
            drop(std::mem::take(&mut displays.display_client));
            drop(displays.efi_display.take());
            drop(atomics.replace_efi_display(None));
        }
        drop(self.root_node.get_mut().0.root_node_mapping.take());
        {
            let (root, _) = self.root_fifo.get_mut();
            drop(root.root_fifo.take());
            drop(root.root_fifo_mapping.take());
        }
        drop(std::mem::take(self.child_fifos.get_mut()));
        drop(self.blit_up_queue.take());
        drop(self.blit_down_queue.take());
        drop(self.blit_in_rgb_2p_xr10_a8_pipeline.take());
        drop(self.blit_out_rgb_2p_xr10_a8_pipeline.take());
        drop(std::mem::take(&mut self.blocks));
    }
}

/// The device's retain count: the host's references and the ones blocks and weak loads take.
/// Its last drop shuts the device down, on the releasing thread, while the core is still
/// alive, so the core's unretained back-links still reach it during the shutdown (module doc).
pub struct Handle<P: Platform> {
    core: Arc<DeviceCore<P>>,
}

impl<P: Platform> std::ops::Deref for Handle<P> {
    type Target = Arc<DeviceCore<P>>;

    fn deref(&self) -> &Arc<DeviceCore<P>> {
        &self.core
    }
}

impl<P: Platform> Drop for Handle<P> {
    /// The last release's shutdown, with no holder ([`Hold::NONE`]): a drop has none.
    fn drop(&mut self) {
        self.core.shutdown(Hold::NONE);
    }
}

/// A retained device: the host's handle, and the retain a block copy or a weak load takes. There is
/// no `Clone`: [`Device::clone_ref`] is a retain, and dropping the last one shuts the device down,
/// on this thread (it may hang behind a FIFO's delay, or die).
///
/// The entry points are the core's methods, reached through `Deref`: the MMIO window
/// (`mmio_read`, `mmio_write`), `new_display`, and the lifecycle (`will_suspend`,
/// `finish_suspend`, `will_resume`, `did_resume`, `pause`, `unpause`, `stop`, `reset`). Each runs
/// in the host's call, where no handler is live: a raise in one dies at the raise.
pub struct Device<P: Platform>(Arc<Handle<P>>);

impl<P: Platform> Device<P> {
    /// Creates a device over `platform` from the host's descriptor, copied at the call: the
    /// host's first retain, or the reason for a refusal ([`DeviceRefusal`], checked in its
    /// order). A raise from Metal here has no handler.
    pub fn new(platform: Arc<P>, descriptor: DeviceConfig<P>) -> Result<Device<P>, DeviceRefusal> {
        DeviceCore::create(platform, descriptor).map(Device)
    }

    /// One more retain.
    pub fn clone_ref(&self) -> Device<P> {
        Device(Arc::clone(&self.0))
    }

    /// Whether `self` and `other` are retains of one device.
    pub fn ptr_eq(&self, other: &Device<P>) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<P: Platform> std::ops::Deref for Device<P> {
    type Target = Arc<DeviceCore<P>>;

    fn deref(&self) -> &Arc<DeviceCore<P>> {
        &self.0
    }
}

// What the device's FIFOs, tasks, mappings and displays ask of it.
mod seam_impls {
    use super::*;
    use crate::guest::records::FeaturesRef;
    use crate::host::PhysRange;
    use crate::memory::mapping::Teardown;

    impl<P: Platform> DeviceCore<P> {
        /// The Metal device, retained.
        pub fn mtl_device(&self) -> P::Device {
            self.mtl_device.clone_ref()
        }

        /// A reference to the device's features, read at each use.
        pub fn features(&self) -> FeaturesRef {
            self.features.clone()
        }

        /// The device's memory map; `None` in the blocks setup.
        pub fn memory_map(&self) -> Option<Arc<MemoryMap<P>>> {
            self.memory_map.lock().clone()
        }

        /// Whether the device offers argument buffers to its isolated tasks: what an isolated
        /// task's handshake carries. Where isolation follows the guest, a task is isolated only
        /// while the IOSurface device has a descriptor table, so this is also what device info
        /// answers then (`argument_buffers_answer`).
        pub fn supports_argument_buffers(&self) -> bool {
            self.supports_argument_buffers.load(Ordering::Relaxed)
        }

        /// A task's paging queue. Retained; `None` where Metal answered nil, which is never
        /// checked: the task sends to nil through it.
        pub fn blit_down_queue(&self) -> Option<P::CommandQueue> {
            self.blit_down_queue.as_ref().map(|q| q.clone_ref())
        }

        /// Retained; `None` for nil.
        pub fn blit_up_queue(&self) -> Option<P::CommandQueue> {
            self.blit_up_queue.as_ref().map(|q| q.clone_ref())
        }

        /// Retained; `None` for nil.
        pub fn exec_queue(&self) -> Option<P::CommandQueue> {
            self.exec_queue.as_ref().map(|q| q.clone_ref())
        }

        /// Retained (not an IOSurface: no observable count).
        pub fn blit_in_rgb_2p_xr10_a8_pipeline(&self) -> Option<P::ComputePipeline> {
            self.blit_in_rgb_2p_xr10_a8_pipeline
                .as_ref()
                .map(|p| p.clone_ref())
        }

        /// Retained.
        pub fn blit_out_rgb_2p_xr10_a8_pipeline(&self) -> Option<P::ComputePipeline> {
            self.blit_out_rgb_2p_xr10_a8_pipeline
                .as_ref()
                .map(|p| p.clone_ref())
        }

        /// Opcode 58: the device info pairs into guest page `destination_page`.
        pub fn get_device_info(
            &self,
            key: u32,
            length: u32,
            destination_page: u32,
        ) -> RaiseResult<()> {
            crate::guest::abi::device_info::get_device_info(
                key,
                length,
                destination_page,
                |page, len| Ok(DeviceCore::new_contiguous_mapping(self, page, len)),
                |mapping: &Arc<Mapping<P>>, max_key, max_count| {
                    let base = GuestPtr::new(mapping.virtual_address());
                    let snapshot = self.info_snapshot();
                    // Opcode 58's F class is the handler.
                    crate::guest::abi::device_info::write_device_info(
                        OnFault::Return,
                        &self.mtl_device.caps(),
                        &snapshot,
                        max_key,
                        max_count,
                        |i, pair| write_device_info_pair(base, i, pair),
                    )
                },
                // The key is stored as an `i32`.
                |max_key| {
                    self.guest_device_info_max_key
                        .store(max_key.cast_signed(), Ordering::Relaxed)
                },
            )
        }

        /// A mapping task over the memory map, or the host's `createTask` block; whether `info`
        /// names a task.
        ///
        /// With a memory map, a refused mapping task leaves `info` as it was, so a stale task
        /// reports success.
        pub fn create_task(&self, size: u64, info: &mut TaskInfo) -> bool {
            let map = self.memory_map.lock().clone();
            if let Some(map) = map {
                if let Some(task) = MappingTask::new(self.platform.clone(), size, map) {
                    info.task = mapping_task_id(&task);
                    info.base_address = task.base_address();
                    self.mapping_tasks.lock().push(task);
                }
            } else {
                let Some(create) = &self.blocks.create_task else {
                    crate::fatal!(DeviceDeath::NilBlock("createTask"))
                };
                info.task = create(size, &mut info.base_address);
            }
            !info.task.is_null()
        }

        /// Releases the mapping task, or calls the host's `destroyTask` block. The caller
        /// clears its own copy of the task.
        pub fn destroy_task(&self, info: &TaskInfo) {
            if self.memory_map.lock().is_some() {
                if info.task.is_null() {
                    return;
                }
                let released = {
                    let mut tasks = self.mapping_tasks.lock();
                    tasks
                        .iter()
                        .position(|t| mapping_task_id(t) == info.task)
                        .map(|i| tasks.swap_remove(i))
                };
                drop(released);
            } else {
                if info.task.is_null() {
                    return;
                }
                let Some(destroy) = &self.blocks.destroy_task else {
                    crate::fatal!(DeviceDeath::NilBlock("destroyTask"))
                };
                destroy(info.task);
            }
        }

        /// Maps `ranges` at `offset` in the host task `task`: through its mapping task, or the
        /// host's `mapMemory` block, whose answer this is.
        pub fn map_memory_in_task(
            &self,
            task: HostTask,
            offset: u64,
            read_only: bool,
            ranges: &[PhysRange],
        ) -> bool {
            crate::fatal_unless!(
                !self.leaving_suspend_early(),
                DeviceDeath::MappingWhileLeavingSuspend
            );
            if self.memory_map.lock().is_some() {
                let found = self
                    .mapping_tasks
                    .lock()
                    .iter()
                    .find(|t| mapping_task_id(t) == task)
                    .cloned();
                return found
                    .is_some_and(|t| t.map_memory_at_virtual_offset(offset, ranges, read_only));
            }
            let Some(map) = &self.blocks.map_memory else {
                crate::fatal!(DeviceDeath::NilBlock("mapMemory"))
            };
            map(
                task,
                crate::num::Truncate::low_u32(ranges.len()),
                offset,
                read_only,
                ranges,
            )
        }

        /// Unmaps `length` bytes at `offset` in the host task `task`, through the
        /// [`MappingSink`].
        pub fn unmap_memory_in_task(&self, task: HostTask, offset: u64, length: u64) -> bool {
            self.mapping_sink.unmap_memory_in_task(task, offset, length)
        }

        /// What the device's mappings unmap through, unretained: a [`Mapping`]'s teardown
        /// link.
        pub(crate) fn mapping_link(&self) -> crate::seams::links::MappingDeviceLink<P> {
            Arc::downgrade(&self.mapping_sink)
        }
    }

    /// The EFI display's task mapping: from the memory map, or a range of the host task `info`
    /// names, mapped by the host's `mapMemory` block.
    fn new_task_mapping<P: Platform>(
        d: &DeviceCore<P>,
        info: &TaskInfo,
        ranges: &[PhysRange],
        total_length: u64,
        offset: u64,
        read_only: bool,
    ) -> Option<Arc<Mapping<P>>> {
        crate::fatal_unless!(
            !d.leaving_suspend_early(),
            DeviceDeath::MappingWhileLeavingSuspend
        );
        // The task and the offset are not used with a memory map.
        let map = d.memory_map.lock().clone();
        if let Some(map) = map {
            return map.new_range_mapping(ranges, total_length, u32::from(read_only));
        }
        let Some(map_memory) = &d.blocks.map_memory else {
            crate::fatal!(DeviceDeath::NilBlock("mapMemory"))
        };
        let count = crate::num::Truncate::low_u32(ranges.len());
        if !map_memory(info.task, count, offset, read_only, ranges) {
            return None;
        }
        // At `base_address + offset`; the device answers its release through the sink.
        Some(Mapping::new(
            info.base_address.wrapping_add(offset),
            total_length,
            true,
            Teardown::Task {
                device: d.mapping_link(),
                virtual_offset: offset,
                task: info.task,
            },
        ))
    }

    /// A device info pair's store into the mapped page (a NULL page faults).
    pub(crate) const DEVICE_INFO_WRITE: DeviceDeath = DeviceDeath::DeviceInfoWrite;

    /// Stores pair `i` as one 8-byte store, single-copy atomic in the page, so a guest reading
    /// a pair with one 64-bit load never sees a key without its value.
    pub(crate) fn write_device_info_pair(
        base: GuestPtr,
        i: u32,
        pair: crate::guest::abi::suspend_blob::KeyValuePair,
    ) {
        let at = u64::from(i).wrapping_mul(8);
        base.write_u64(at, device_info_pair_word(pair), DEVICE_INFO_WRITE);
    }

    /// A pair's 8 bytes as one little-endian word: the key low, the value high.
    pub(crate) fn device_info_pair_word(
        pair: crate::guest::abi::suspend_blob::KeyValuePair,
    ) -> u64 {
        u64::from(pair.value).wrapping_shl(32) | u64::from(pair.key)
    }

    // The EFI display's framebuffer.
    impl<P: Platform> DeviceCore<P> {
        /// The EFI framebuffer: `ranges` mapped into `info`'s task at `offset`. `None` when
        /// the host refuses.
        pub(crate) fn new_task_mapping(
            &self,
            info: &TaskInfo,
            ranges: &[PhysRange],
            total_length: u64,
            offset: u64,
            read_only: bool,
        ) -> Option<Arc<Mapping<P>>> {
            new_task_mapping(self, info, ranges, total_length, offset, read_only)
        }
    }
}

/// A device on the fake platform with the fake host's blocks, for the device's unit tests.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;

    use crate::device::DeviceCore;
    use crate::device::config::DeviceConfig;
    use crate::guest::GuestPtr;
    use crate::guest::records::APV_PAGE_SIZE;
    use crate::rt::Hold;
    use crate::testkit::guest_ram::RawRegion;
    use crate::testkit::host::FakeHost;
    use crate::testkit::{FakePlatform, GuestRam};

    /// The guest RAM the fake host has: physical 0 up, 256 pages.
    pub(crate) const RAM_LENGTH: u64 = 0x40_0000;

    /// The lines the device area writes while this lives, captured from the `log` crate
    /// (`testkit::logs`): it relies on nextest's one process per test, and one lives at a time.
    pub(crate) struct Lines {
        logs: crate::testkit::logs::Capture,
        seen: crate::sync::Mutex<Vec<String>>,
    }

    impl Lines {
        /// Starts capturing.
        pub(crate) fn capture() -> Lines {
            Lines {
                logs: crate::testkit::logs::capture(),
                seen: crate::sync::Mutex::new(Vec::new()),
            }
        }

        /// Every message the device area wrote since the capture began, in order.
        pub(crate) fn all(&self) -> Vec<String> {
            let area = concat!(env!("CARGO_CRATE_NAME"), "::device");
            let mut seen = self.seen.lock();
            seen.extend(
                self.logs
                    .take()
                    .into_iter()
                    .filter(|(_, target, _)| {
                        target
                            .strip_prefix(area)
                            .is_some_and(|below| below.is_empty() || below.starts_with("::"))
                    })
                    .map(|(_, _, message)| message),
            );
            seen.clone()
        }

        /// How many messages contain `needle`.
        pub(crate) fn count(&self, needle: &str) -> usize {
            self.all().iter().filter(|l| l.contains(needle)).count()
        }

        /// Whether a message contains `needle`.
        pub(crate) fn contains(&self, needle: &str) -> bool {
            self.count(needle) != 0
        }
    }

    /// The page `with_root_page` gives the root node.
    pub(crate) const ROOT_PAGE: u32 = 1;

    /// The device under test, its platform, its host and the fake RAM behind its pointers.
    pub(crate) struct TestDevice {
        pub platform: Arc<FakePlatform>,
        pub host: Arc<FakeHost>,
        pub device: Arc<crate::device::Handle<FakePlatform>>,
        /// Backs the stamp memory (64 slots) once [`TestDevice::with_stamp_memory`] made it.
        pub stamp_ram: Option<Arc<RawRegion>>,
        /// Backs the root node page once [`TestDevice::with_root_node`] made it.
        pub root_ram: Option<Arc<RawRegion>>,
    }

    /// Options for [`device`].
    pub(crate) struct Opts {
        pub ports: u32,
        pub max_fifo_count: u32,
        pub using_mapper_service: bool,
        pub connection_types: u32,
    }

    impl Default for Opts {
        fn default() -> Self {
            Opts {
                ports: 2,
                max_fifo_count: 64,
                using_mapper_service: false,
                connection_types: 0,
            }
        }
    }

    /// A device with `opts`: no root, no FIFOs, no tasks, `maxFIFOCount` as asked.
    pub(crate) fn device(opts: Opts) -> TestDevice {
        let platform = Arc::new(FakePlatform::new());
        let host = FakeHost::new(&platform, GuestRam::with_range(0, RAM_LENGTH).expect("ram"));
        let mut c = DeviceConfig::new(Some(platform.device()));
        c.display_port_count = opts.ports;
        c.blocks = host.blocks();
        c.using_iosurface_mapper = opts.using_mapper_service;
        c.display_port_connection_types = opts.connection_types;
        let device = DeviceCore::create(platform.clone(), c).expect("device");
        let mut f = device.features.load();
        f.max_fifo_count = opts.max_fifo_count;
        device.features.store(&f);
        TestDevice {
            platform,
            host,
            device,
            stamp_ram: None,
            root_ram: None,
        }
    }

    impl TestDevice {
        /// Sets the memory map the device answers.
        pub(crate) fn set_memory_map(
            &self,
            map: Option<Arc<crate::memory::map::MemoryMap<FakePlatform>>>,
        ) {
            *self.device.memory_map.lock() = map;
        }

        /// Sets the device's argument-buffer support.
        pub(crate) fn set_supports_argument_buffers(&self, on: bool) {
            self.device
                .supports_argument_buffers
                .store(on, std::sync::atomic::Ordering::Relaxed);
        }

        /// The stamp interrupt's bits.
        pub(crate) fn interrupt_stamp_bits(&self) -> u32 {
            self.device
                .interrupt_stamp
                .load(std::sync::atomic::Ordering::SeqCst)
        }

        /// The fault interrupt's bits.
        pub(crate) fn interrupt_fault_bits(&self) -> u64 {
            self.device
                .interrupt_fault_mask
                .load(std::sync::atomic::Ordering::SeqCst)
        }

        /// Gives the device 64 stamp slots in fake RAM.
        pub(crate) fn with_stamp_memory(mut self) -> Self {
            let ram = RawRegion::new(0x4000).expect("stamp ram");
            *self.device.stamp_memory.lock() = Some(GuestPtr::new(ram.addr()));
            self.stamp_ram = Some(ram);
            self
        }

        /// Gives the device a root node page in fake RAM.
        pub(crate) fn with_root_node(mut self) -> Self {
            let ram = RawRegion::new(0x4000).expect("root ram");
            self.device.root_node.lock().root_node = Some(GuestPtr::new(ram.addr()));
            self.root_ram = Some(ram);
            self
        }

        /// Sets the root page through `set_root_page` (the blocks setup: a root task, the
        /// node mapped from guest RAM page [`ROOT_PAGE`]).
        pub(crate) fn with_root_page(self) -> Self {
            self.device.set_root_page(Hold::NONE, ROOT_PAGE);
            assert!(self.device.root_node.lock().root_node.is_some());
            self
        }

        /// The root node's word `i` in guest RAM (after [`TestDevice::with_root_page`]).
        pub(crate) fn guest_root_word(&self, i: u64) -> u32 {
            self.platform.fake_vm().sync();
            self.host
                .ram()
                .read_u32(u64::from(ROOT_PAGE) * APV_PAGE_SIZE + i * 4)
                .expect("root word")
        }

        /// Writes the root node's word `i` in guest RAM.
        pub(crate) fn set_guest_root_word(&self, i: u64, v: u32) {
            self.platform.fake_vm().sync();
            assert!(
                self.host
                    .ram()
                    .write_u32(u64::from(ROOT_PAGE) * APV_PAGE_SIZE + i * 4, v)
            );
            self.platform.fake_vm().sync();
        }

        /// A task root node in guest page `page`: page-table depth 1 and `length` at +0x14.
        pub(crate) fn task_root_node(&self, page: u32, length: u64) {
            let pa = u64::from(page) * APV_PAGE_SIZE;
            self.platform.fake_vm().sync();
            assert!(self.host.ram().write_u32(pa + 4, 1));
            assert!(self.host.ram().write_u64(pa + 0x14, length));
            self.platform.fake_vm().sync();
        }

        /// Writes stamp slot `i` (as `signal_stamp` does, under the signal lock) and
        /// notifies the waiters when `notify`.
        pub(crate) fn set_stamp(&self, i: u32, v: u32, notify: bool) {
            let _g = self.device.signal_mutex.lock();
            assert!(
                self.stamp_ram
                    .as_ref()
                    .expect("stamp ram")
                    .write_u32(u64::from(i) * 4, v)
            );
            if notify {
                self.device.signal_cond.notify_all();
            }
        }

        /// Root node word `i` (a `u32` index).
        pub(crate) fn root_word(&self, i: u64) -> u32 {
            self.root_ram
                .as_ref()
                .and_then(|r| r.read_u32(i * 4))
                .expect("root word")
        }

        /// Writes root node word `i`.
        pub(crate) fn set_root_word(&self, i: u64, v: u32) {
            assert!(
                self.root_ram
                    .as_ref()
                    .expect("root ram")
                    .write_u32(i * 4, v)
            );
        }

        /// The device's EFI display now.
        pub(crate) fn efi_display(
            &self,
        ) -> Option<Arc<crate::display::efi::EfiDisplay<FakePlatform>>> {
            self.device.displays.atomics().efi_display()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::device::root::rig::rig;
    use crate::device::tasks::Isolation;
    use crate::host::HostBlocks;
    use crate::memory::descriptor::MemoryMapDescriptor;
    use crate::testkit::capture::expect_death;
    use crate::testkit::host::FakeHost;
    use crate::testkit::{FakePlatform, GuestRam};

    fn config(p: &Arc<FakePlatform>, host: &Arc<FakeHost>) -> DeviceConfig<FakePlatform> {
        let mut c = DeviceConfig::new(Some(p.device()));
        c.display_port_count = 1;
        c.blocks = host.blocks();
        c
    }

    /// A block's captured context: records its name when the block's last reference goes
    /// (the host's `destroy` at the last `Block_release`).
    struct Tag(Arc<parking_lot::Mutex<Vec<&'static str>>>, &'static str);

    impl Drop for Tag {
        fn drop(&mut self) {
            self.0.lock().push(self.1);
        }
    }

    /// All eight blocks, each capturing a [`Tag`]; the only references are the answer's.
    fn tagged_blocks(order: &Arc<parking_lot::Mutex<Vec<&'static str>>>) -> HostBlocks {
        let tag = |name| Tag(order.clone(), name);
        let (raise, create, destroy) =
            (tag("raiseInterrupt"), tag("createTask"), tag("destroyTask"));
        let (map, unmap, read) = (tag("mapMemory"), tag("unmapMemory"), tag("readMemory"));
        let (add, remove) = (tag("addTraceRange"), tag("removeTraceRange"));
        HostBlocks {
            raise_interrupt: Some(Arc::new(move |_| {
                let _ = &raise;
            })),
            create_task: Some(Arc::new(move |_, _| {
                let _ = &create;
                HostTask::NULL
            })),
            destroy_task: Some(Arc::new(move |_| {
                let _ = &destroy;
            })),
            map_memory: Some(Arc::new(move |_, _, _, _, _| {
                let _ = &map;
                false
            })),
            unmap_memory: Some(Arc::new(move |_, _, _| {
                let _ = &unmap;
                false
            })),
            read_memory: Some(Arc::new(move |_, _, _| {
                let _ = &read;
                false
            })),
            add_trace_range: Some(Arc::new(move |_, _| {
                let _ = &add;
                crate::host::HostTraceRange::NULL
            })),
            remove_trace_range: Some(Arc::new(move |_| {
                let _ = &remove;
            })),
        }
    }

    /// The order the host sees the blocks' `destroy` callbacks.
    const BLOCK_RELEASE_ORDER: [&str; 8] = [
        "createTask",
        "destroyTask",
        "mapMemory",
        "unmapMemory",
        "readMemory",
        "raiseInterrupt",
        "addTraceRange",
        "removeTraceRange",
    ];

    /// The shutdown releases createTask, destroyTask, mapMemory, unmapMemory, readMemory,
    /// raiseInterrupt, addTraceRange, removeTraceRange; with the host's descriptor gone the
    /// device's are the last references, so each block's destroy runs in that order
    #[test]
    fn the_last_release_releases_the_blocks_in_order() {
        let p = Arc::new(FakePlatform::new());
        let order = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let mut c = DeviceConfig::new(Some(p.device()));
        c.display_port_count = 1;
        c.blocks = tagged_blocks(&order);
        // The config (the host's descriptor) is consumed: the device's copies are the last.
        let d = DeviceCore::create(p.clone(), c).expect("device");
        assert!(order.lock().is_empty());
        drop(d);
        assert_eq!(*order.lock(), BLOCK_RELEASE_ORDER);
    }

    /// Without a memory map the device keeps the host's own task and memory blocks, each the
    /// same block the host gave: the platform's `host_blocks` answers them unchanged by default
    #[test]
    fn the_blocks_setup_keeps_the_host_s_own_blocks() {
        let p = Arc::new(FakePlatform::new());
        let order = Arc::new(parking_lot::Mutex::new(Vec::new()));
        let mut c = DeviceConfig::new(Some(p.device()));
        c.display_port_count = 1;
        c.blocks = tagged_blocks(&order);
        let host = c.blocks.clone();
        let d = DeviceCore::create(p.clone(), c).expect("device");
        fn same<T: ?Sized>(kept: &Option<Arc<T>>, given: &Option<Arc<T>>) -> bool {
            matches!((kept, given), (Some(k), Some(g)) if Arc::ptr_eq(k, g))
        }
        assert!(same(&d.blocks.create_task, &host.create_task));
        assert!(same(&d.blocks.destroy_task, &host.destroy_task));
        assert!(same(&d.blocks.map_memory, &host.map_memory));
        assert!(same(&d.blocks.unmap_memory, &host.unmap_memory));
        assert!(same(&d.blocks.read_memory, &host.read_memory));
        drop(host);
        drop(d);
        assert_eq!(*order.lock(), BLOCK_RELEASE_ORDER);
    }

    /// A weak load of the device (`DeviceCore::retain_self`, the page-table read block's), held
    /// to the end of its hold's holder: a host release meanwhile leaves the shutdown to that
    /// end; nothing once the shutdown has started
    #[test]
    fn a_weak_load_held_to_the_holder_s_end_shuts_down_there() {
        let r = rig();
        let device: Arc<DeviceCore<FakePlatform>> = Arc::clone(&r.device);
        let retained = device.handle.clone();
        let held = crate::rt::held::Held::new();
        Hold::of(&held).keep(device.retain_self().expect("live"));
        // The host's release: the holder's reference keeps the device from shutting down.
        drop(r);
        assert_eq!(retained.strong_count(), 1);
        drop(held);
        assert_eq!(retained.strong_count(), 0);
        assert!(device.retain_self().is_none());
    }

    /// An `Err` reaching a method without `RaiseResult` dies at the raise's own site: a core raise's
    /// site, the selector for a Metal one.
    #[test]
    fn uncaught_dies_at_the_raise() {
        expect_death(
            DeathSite {
                kind: DeathKind::UnhandledRaise,
                file: "device",
                line: 0,
            },
            || {
                no_catch::<()>(Err(crate::raise::Raise::new(crate::raise::RaiseId::Test {
                    name: "device",
                })
                .with_detail("no task 5")))
            },
        );
        expect_death(
            DeathSite {
                kind: DeathKind::UncaughtObjc,
                file: "newComputePipelineStateWithFunction:error:",
                line: 0,
            },
            || {
                uncaught(
                    crate::raise::Raise::foreign("newComputePipelineStateWithFunction:error:")
                        .with_detail("scripted"),
                )
            },
        );
    }

    #[test]
    fn new_initial_state() {
        let r = rig();
        let d = &r.device;
        assert_eq!(d.run_state.load(), RunState::Running);
        assert_eq!(d.child_fifos.lock().len(), 64);
        assert_eq!(d.waiting_info.len(), 64);
        assert!(
            d.waiting_info
                .iter()
                .all(|w| w.next_waiting_index.load(Ordering::SeqCst) == u32::MAX)
        );
        assert!(d.root_node.lock().root_node.is_none());
        assert!(d.root_task.lock().root_task_allocator.is_some());
        assert!(d.memory_map.lock().is_none());
    }

    /// The device's blit-up, blit-down and exec queues are made unshared, each labelled: each is
    /// on a present's path.
    #[test]
    fn the_device_s_queues_are_unshared() {
        use crate::testkit::impls::device::UNSHARED_QUEUE;
        let p = Arc::new(FakePlatform::new());
        let host = FakeHost::new(&p, GuestRam::new());
        let d = DeviceCore::create(p.clone(), config(&p, &host)).expect("device");
        let log = p.call_log();
        assert_eq!(log.count(UNSHARED_QUEUE), 3);
        assert_eq!(log.count("newCommandQueue"), 0);
        let labels: Vec<String> = log
            .calls_to("setLabel:")
            .iter()
            .filter_map(|c| match c.args.first() {
                Some(crate::testkit::calllog::Arg::Str(s)) => Some(s.clone()),
                _ => None,
            })
            .collect();
        for label in [
            "reims-vgpu.blit-up",
            "reims-vgpu.blit-down",
            "reims-vgpu.exec",
        ] {
            assert!(labels.iter().any(|l| l == label), "{label} in {labels:?}");
        }
        assert!(d.blit_up_queue.is_some() && d.blit_down_queue.is_some());
        assert!(d.exec_queue.is_some());
    }

    /// The device's capabilities are fields: what
    /// the creation stored from Metal's answers, the map and the descriptor.
    #[test]
    fn synthesized_getters_are_fields() {
        let p = Arc::new(FakePlatform::new());
        let host = FakeHost::new(&p, GuestRam::new());
        let mut c = config(&p, &host);
        c.device_feature_level = 3;
        let d = DeviceCore::create(p.clone(), c).expect("device");
        let m = d.mtl_device.caps();
        assert_eq!(d.binary_version.load(Ordering::SeqCst), 0);
        assert_eq!(d.caps.feature_level, 3);
        assert_eq!(
            d.caps.shared_textures,
            m.flags.supports_shared_texture_handles
        );
        assert_eq!(
            d.caps.buffer_with_iosurface,
            m.flags.supports_buffer_with_iosurface
        );
        assert_eq!(
            d.caps.range_buffer,
            m.flags.supports_buffer_with_address_ranges
        );
        assert_eq!(
            d.caps.shared_memory_heap,
            m.flags.supports_heap_with_address_ranges
        );
        // No memory map: neither large-task form.
        assert!(!d.caps.large_user_tasks && !d.caps.large_kernel_tasks);
        // Not isolated: argument buffers stay off.
        assert!(!d.supports_argument_buffers.load(Ordering::SeqCst));
    }

    #[test]
    fn refusals_name_their_check() {
        let p = Arc::new(FakePlatform::new());
        let host = FakeHost::new(&p, GuestRam::new());
        // Port count 0 and 9.
        let mut c = config(&p, &host);
        c.display_port_count = 0;
        assert_eq!(
            DeviceCore::create(p.clone(), c).err(),
            Some(DeviceRefusal::DisplayPortCount)
        );
        let mut c = config(&p, &host);
        c.display_port_count = 9;
        assert_eq!(
            DeviceCore::create(p.clone(), c).err(),
            Some(DeviceRefusal::DisplayPortCount)
        );
        // No Metal device.
        let mut c = config(&p, &host);
        c.device = None;
        assert_eq!(
            DeviceCore::create(p.clone(), c).err(),
            Some(DeviceRefusal::NoMetalDevice)
        );
        // A memory map with blocks.
        let mut c = config(&p, &host);
        c.memory_map_descriptor = Some(Arc::new(MemoryMapDescriptor::new()));
        assert_eq!(
            DeviceCore::create(p.clone(), c).err(),
            Some(DeviceRefusal::MemoryConflict)
        );
        // Isolation without a memory map.
        let mut c = config(&p, &host);
        c.enable_process_isolation = true;
        assert_eq!(
            DeviceCore::create(p.clone(), c).err(),
            Some(DeviceRefusal::IsolationWithoutMap)
        );
    }

    /// `addTraceRange` without `removeTraceRange` is refused at creation, with what was built
    /// released: the host's blocks are let go before `new` returns
    #[test]
    fn add_trace_without_remove_is_refused() {
        let p = Arc::new(FakePlatform::new());
        let host = FakeHost::new(&p, GuestRam::new());
        let mut c = config(&p, &host);
        c.blocks.remove_trace_range = None;
        let add = c.blocks.add_trace_range.clone().expect("add");
        assert_eq!(
            DeviceCore::create(p.clone(), c).err(),
            Some(DeviceRefusal::TraceWithoutRemove)
        );
        assert_eq!(Arc::strong_count(&add), 1);
    }

    #[test]
    fn remove_trace_alone_is_dropped() {
        let p = Arc::new(FakePlatform::new());
        let host = FakeHost::new(&p, GuestRam::new());
        let mut c = config(&p, &host);
        c.blocks.add_trace_range = None;
        let d = DeviceCore::create(p.clone(), c).expect("device");
        assert!(d.blocks.remove_trace_range.is_none());
    }

    /// An isolated device over a memory map of `host`'s RAM that asks for argument buffers.
    fn isolated_config(p: &Arc<FakePlatform>, host: &Arc<FakeHost>) -> DeviceConfig<FakePlatform> {
        use crate::memory::descriptor::GuestPhysicalRange;
        let mut map = MemoryMapDescriptor::new();
        for r in host.ram().ranges() {
            map.add_range(GuestPhysicalRange {
                physical_address: r.physical_address,
                physical_length: r.region.len(),
                host_address: r.host(),
            });
        }
        let mut c = DeviceConfig::new(Some(p.device()));
        c.display_port_count = 1;
        c.blocks.raise_interrupt = host.blocks().raise_interrupt;
        c.memory_map_descriptor = Some(Arc::new(map));
        c.enable_process_isolation = true;
        c.enable_argument_buffers = true;
        c
    }

    /// The device info `d` writes, as (key, value) pairs, for a guest at binary version 245 that
    /// knows every key.
    fn device_info(d: &DeviceCore<FakePlatform>) -> Vec<(u32, u32)> {
        use crate::guest::abi::device_info::write_device_info;
        use crate::guest::abi::features::features_for_version;
        d.features.store(&features_for_version(245));
        let mut out = Vec::new();
        write_device_info(
            OnFault::Die,
            &d.mtl_device.caps(),
            &d.info_snapshot(),
            45,
            64,
            |_, kv| {
                out.push((kv.key, kv.value));
            },
        )
        .expect("device info");
        out
    }

    /// The IOSurface device's descriptor table base register (`iosurface::regs`).
    const DESCRIPTOR_BASE_REGISTER: u64 = 0x1010;
    /// A descriptor table's guest physical base: a page of the test RAM.
    const TABLE: u64 = 0x10_0000;

    /// A device of [`isolated_config`] that maps through the process's IOSurface device, with
    /// isolation following the guest as `follows` asks, over RAM whose page 0x55 is a task root
    /// of 0x4000 bytes: a page table of depth 1 whose root, page 0x56, is all holes.
    fn following_device(
        follows: bool,
    ) -> (Arc<FakePlatform>, Arc<FakeHost>, Arc<Handle<FakePlatform>>) {
        use crate::guest::records::APV_PAGE_SIZE;
        let p = Arc::new(FakePlatform::new());
        let host = FakeHost::new(&p, GuestRam::with_range(0, 0x40_0000).expect("ram"));
        let mut c = isolated_config(&p, &host);
        c.using_iosurface_mapper = true;
        c.process_isolation_follows_guest = follows;
        let d = DeviceCore::create(p.clone(), c).expect("device");
        let pa = 0x55 * APV_PAGE_SIZE;
        p.fake_vm().sync();
        assert!(host.ram().write_u32(pa, 0x56));
        assert!(host.ram().write_u32(pa + 4, 1));
        assert!(host.ram().write_u64(pa + 0x14, 0x4000));
        p.fake_vm().sync();
        (p, host, d)
    }

    /// The process's IOSurface device, over `host`'s RAM: the one `following_device` maps
    /// through.
    fn iosurface_device(
        p: &Arc<FakePlatform>,
        host: &Arc<FakeHost>,
    ) -> crate::iosurface::HostDeviceHandle<FakePlatform> {
        use crate::iosurface::IoSurfaceHostDevice;
        use crate::iosurface::descriptor::IoSurfaceHostDeviceDescriptor;
        use crate::memory::descriptor::GuestPhysicalRange;
        let mut map = MemoryMapDescriptor::new();
        for r in host.ram().ranges() {
            map.add_range(GuestPhysicalRange {
                physical_address: r.physical_address,
                physical_length: r.region.len(),
                host_address: r.host(),
            });
        }
        let mut d = IoSurfaceHostDeviceDescriptor::new();
        d.memory_map_descriptor = Some(map);
        IoSurfaceHostDevice::new(p, &d).expect("IOSurface device")
    }

    /// Makes task `id` over the task root at page 0x55: whether it is remote.
    fn made_remote(d: &DeviceCore<FakePlatform>, id: u32) -> bool {
        d.create_task_id(Hold::NONE, id, 0x55, 0x4000, false)
            .expect("task");
        matches!(d.tasks.lock().get(id), Some(Task::Remote(_)))
    }

    /// Under isolation that follows the guest, a task other than task 0 is isolated only when,
    /// as it is made, the IOSurface device the device maps through has a descriptor table: with
    /// no IOSurface device, or one whose base is 0, it is local; with a base, remote; a base
    /// written back to 0 makes later tasks local and leaves the remote one as it was. Task 0 is
    /// never isolated. Each task made logs how it runs and why.
    #[test]
    fn isolation_follows_the_iosurface_devices_descriptor_table() {
        let logs = crate::testkit::logs::capture();
        let (p, host, d) = following_device(true);
        assert!(d.caps.isolation_follows_guest);
        assert_eq!(
            d.isolation_of(1),
            Isolation::Followed {
                descriptor_base: None
            }
        );
        assert!(!made_remote(&d, 1), "no IOSurface device");
        let ios = iosurface_device(&p, &host);
        let core = ios.core_weak().upgrade().expect("IOSurface core");
        assert_eq!(
            d.isolation_of(2),
            Isolation::Followed {
                descriptor_base: Some(0)
            }
        );
        assert!(!made_remote(&d, 2), "no descriptor table");
        core.mmio_write(DESCRIPTOR_BASE_REGISTER, TABLE);
        assert_eq!(d.isolation_of(0), Isolation::Off);
        assert!(made_remote(&d, 3), "a descriptor table");
        core.mmio_write(DESCRIPTOR_BASE_REGISTER, 0);
        assert!(!made_remote(&d, 4), "the table written back to 0");
        assert!(matches!(d.tasks.lock().get(3), Some(Task::Remote(_))));
        let lines: Vec<String> = logs
            .take()
            .into_iter()
            .map(|l| l.2)
            .filter(|l| l.starts_with("task "))
            .collect();
        assert_eq!(
            lines,
            [
                "task 1 runs in process: no IOSurface device".to_string(),
                "task 2 runs in process: the IOSurface device has no descriptor table".into(),
                format!(
                    "task 3 runs isolated: the IOSurface device's descriptor table is at {TABLE:#x}"
                ),
                "task 4 runs in process: the IOSurface device has no descriptor table".into(),
            ]
        );
        for id in 1..=4 {
            assert!(d.delete_task_id(id).expect("delete"));
        }
        drop(core);
        drop(ios);
    }

    /// Without isolation following the guest, every task but task 0 is isolated, with or
    /// without an IOSurface device and its table, and no task logs how it runs.
    #[test]
    fn without_following_every_task_but_task_0_is_isolated() {
        let logs = crate::testkit::logs::capture();
        let (p, host, d) = following_device(false);
        assert_eq!(d.isolation_of(0), Isolation::Off);
        assert_eq!(d.isolation_of(1), Isolation::On);
        assert!(made_remote(&d, 1), "no IOSurface device");
        let ios = iosurface_device(&p, &host);
        assert!(made_remote(&d, 2), "no descriptor table");
        let core = ios.core_weak().upgrade().expect("IOSurface core");
        core.mmio_write(DESCRIPTOR_BASE_REGISTER, TABLE);
        assert!(made_remote(&d, 3), "a descriptor table");
        assert!(
            !logs
                .take()
                .into_iter()
                .any(|l| l.2.starts_with("task ") && l.2.contains(" runs ")),
        );
        for id in 1..=3 {
            assert!(d.delete_task_id(id).expect("delete"));
        }
        drop(core);
        drop(ios);
    }

    /// Device info's argument-buffer answer: key 33's argument buffer bit (0x20) and key 42.
    fn argument_buffers_answered(d: &DeviceCore<FakePlatform>) -> (bool, Option<u32>) {
        let info = device_info(d);
        let value = |key| info.iter().find(|kv| kv.0 == key).map(|kv| kv.1);
        (value(33).is_some_and(|v| v & 0x20 != 0), value(42))
    }

    /// The device info lines about argument buffers since `logs` began.
    fn argument_buffer_lines(logs: &crate::testkit::logs::Capture) -> Vec<String> {
        logs.take()
            .into_iter()
            .map(|l| l.2)
            .filter(|l| l.starts_with("device info: argument buffers"))
            .collect()
    }

    /// Under isolation that follows the guest, device info answers argument buffers (key 33's
    /// 0x20 bit, key 42) as a task defined then would be isolated, read as the guest asks: off
    /// with no IOSurface device and with a descriptor table base of 0, on with a base, off again
    /// once it is written back to 0, the device offering them throughout. Each change of answer
    /// logs one line with its reason, and an answer repeated logs none.
    #[test]
    fn argument_buffers_follow_the_iosurface_devices_descriptor_table() {
        let logs = crate::testkit::logs::capture();
        let (p, host, d) = following_device(true);
        assert!(d.supports_argument_buffers());
        let off = (false, Some(0));
        let on = (true, Some(3));
        assert_eq!(argument_buffers_answered(&d), off, "no IOSurface device");
        assert_eq!(argument_buffers_answered(&d), off, "asked again");
        let ios = iosurface_device(&p, &host);
        let core = ios.core_weak().upgrade().expect("IOSurface core");
        assert_eq!(argument_buffers_answered(&d), off, "no descriptor table");
        core.mmio_write(DESCRIPTOR_BASE_REGISTER, TABLE);
        assert_eq!(argument_buffers_answered(&d), on, "a descriptor table");
        assert_eq!(argument_buffers_answered(&d), on, "asked again");
        core.mmio_write(DESCRIPTOR_BASE_REGISTER, 0);
        assert_eq!(
            argument_buffers_answered(&d),
            off,
            "the table written back to 0"
        );
        assert_eq!(
            argument_buffer_lines(&logs),
            [
                "device info: argument buffers off: no IOSurface device".to_string(),
                format!(
                    "device info: argument buffers on: the IOSurface device's descriptor table \
                     is at {TABLE:#x}"
                ),
                "device info: argument buffers off: the IOSurface device has no descriptor table"
                    .into(),
            ]
        );
        drop(core);
        drop(ios);
    }

    /// The first answer of a run that starts with an IOSurface device but no table says so.
    #[test]
    fn argument_buffers_first_answer_names_an_empty_table() {
        let logs = crate::testkit::logs::capture();
        let (p, host, d) = following_device(true);
        let ios = iosurface_device(&p, &host);
        assert_eq!(argument_buffers_answered(&d), (false, Some(0)));
        assert_eq!(
            argument_buffer_lines(&logs),
            ["device info: argument buffers off: the IOSurface device has no descriptor table"]
        );
        drop(ios);
    }

    /// A device that offers no argument buffers answers none under isolation that follows the
    /// guest, descriptor table or not, and says why.
    #[test]
    fn following_without_the_offer_answers_no_argument_buffers() {
        let logs = crate::testkit::logs::capture();
        let p = Arc::new(FakePlatform::new());
        let host = FakeHost::new(&p, GuestRam::with_range(0, 0x40_0000).expect("ram"));
        let mut c = isolated_config(&p, &host);
        c.using_iosurface_mapper = true;
        c.process_isolation_follows_guest = true;
        c.enable_argument_buffers = false;
        let d = DeviceCore::create(p.clone(), c).expect("device");
        let ios = iosurface_device(&p, &host);
        let core = ios.core_weak().upgrade().expect("IOSurface core");
        core.mmio_write(DESCRIPTOR_BASE_REGISTER, TABLE);
        assert_eq!(argument_buffers_answered(&d), (false, Some(0)));
        assert_eq!(
            argument_buffer_lines(&logs),
            ["device info: argument buffers off: the device offers none"]
        );
        drop(core);
        drop(ios);
    }

    /// Without isolation following the guest, device info answers argument buffers as the device
    /// offers them, with or without an IOSurface device and its table, and logs nothing about
    /// them.
    #[test]
    fn without_following_argument_buffers_are_what_the_device_offers() {
        let logs = crate::testkit::logs::capture();
        let (p, host, d) = following_device(false);
        let on = (true, Some(3));
        assert_eq!(argument_buffers_answered(&d), on, "no IOSurface device");
        let ios = iosurface_device(&p, &host);
        assert_eq!(argument_buffers_answered(&d), on, "no descriptor table");
        let core = ios.core_weak().upgrade().expect("IOSurface core");
        core.mmio_write(DESCRIPTOR_BASE_REGISTER, TABLE);
        assert_eq!(argument_buffers_answered(&d), on, "a descriptor table");
        assert!(argument_buffer_lines(&logs).is_empty());
        drop(core);
        drop(ios);
    }

    /// Argument buffers follow the descriptor's `enableArgumentBuffers` on an isolated device
    /// (the fake device supports detach backing and is not the paravirtual one).
    #[test]
    fn argument_buffers_need_isolation_and_the_descriptor() {
        for enable in [true, false] {
            let p = Arc::new(FakePlatform::new());
            let host = FakeHost::new(&p, GuestRam::with_range(0, 0x40_0000).expect("ram"));
            let mut c = isolated_config(&p, &host);
            c.enable_argument_buffers = enable;
            let d = DeviceCore::create(p.clone(), c).expect("device");
            assert!(d.caps.process_isolation);
            assert_eq!(d.supports_argument_buffers.load(Ordering::Relaxed), enable);
        }
    }

    /// A Metal device answering NO for the capabilities an older GPU lacks keeps argument
    /// buffers off on an isolated device that asks for them (it cannot detach backings), and
    /// device info reports what the device derives from the answers: keys 17 (IOSurface
    /// buffers) and 19 (shared textures) are left out, key 24 (image blocks) is 0, key 28 loses
    /// its dynamic stride (2), key 33 loses its command buffer jump (4), range buffer (8), shared
    /// memory heap (0x10), argument buffer (0x20) and shared texture (0x800) bits, and key 42
    /// is 0.
    #[test]
    fn a_device_without_the_capabilities_leaves_them_out_of_device_info() {
        for supported in [true, false] {
            let p = Arc::new(FakePlatform::new());
            if !supported {
                let mut caps = crate::platform::caps::DeviceCaps::clone(&p.device().caps());
                let f = &mut caps.flags;
                f.supports_resource_detach_backing = false;
                f.supports_heap_with_address_ranges = false;
                f.supports_buffer_with_address_ranges = false;
                f.supports_buffer_with_iosurface = false;
                f.supports_shared_texture_handles = false;
                f.supports_dynamic_attribute_stride = false;
                f.supports_image_blocks = false;
                f.supports_command_buffer_jump = false;
                p.device().set_caps(caps);
            }
            let host = FakeHost::new(&p, GuestRam::with_range(0, 0x40_0000).expect("ram"));
            let d = DeviceCore::create(p.clone(), isolated_config(&p, &host)).expect("device");
            assert_eq!(
                d.supports_argument_buffers.load(Ordering::Relaxed),
                supported
            );
            let info = device_info(&d);
            let value = |key| info.iter().find(|kv| kv.0 == key).map(|kv| kv.1);
            let want = if supported {
                [Some(1), Some(1), Some(1), Some(7), Some(0xfff), Some(3)]
            } else {
                [None, None, Some(0), Some(5), Some(0x7c3), Some(0)]
            };
            for (key, want) in [17, 19, 24, 28, 33, 42].into_iter().zip(want) {
                assert_eq!(value(key), want, "key {key}");
            }
        }
    }

    /// A Metal device named "Apple Paravirtual device", as Metal names the GPU of a macOS guest
    /// that itself hosts a VM, keeps argument buffers off on an isolated device that asks for
    /// them: device info's key 33 loses its argument buffer bit (0x20) and key 42 is 0.
    #[test]
    fn a_paravirtual_device_keeps_argument_buffers_off() {
        for (name, on) in [("FakeMTLDevice", true), ("Apple Paravirtual device", false)] {
            let p = Arc::new(FakePlatform::new());
            let mut caps = crate::platform::caps::DeviceCaps::clone(&p.device().caps());
            caps.name = Some(String::from(name));
            p.device().set_caps(caps);
            let host = FakeHost::new(&p, GuestRam::with_range(0, 0x40_0000).expect("ram"));
            let d = DeviceCore::create(p.clone(), isolated_config(&p, &host)).expect("device");
            assert_eq!(d.supports_argument_buffers.load(Ordering::Relaxed), on);
            let info = device_info(&d);
            let value = |key| info.iter().find(|kv| kv.0 == key).map(|kv| kv.1);
            assert_eq!(value(33), Some(if on { 0xfff } else { 0xfdf }), "{name}");
            assert_eq!(value(42), Some(if on { 3 } else { 0 }), "{name}");
        }
    }

    /// A nil `descriptor.device` refuses, no log
    #[test]
    fn nil_device_refuses() {
        let p = Arc::new(FakePlatform::new());
        let host = FakeHost::new(&p, GuestRam::new());
        let mut c = config(&p, &host);
        c.device = None;
        assert!(DeviceCore::create(p.clone(), c).is_err());
    }

    // `enableProtectedContent` is ignored
    #[test]
    fn protected_content_ignored() {
        let p = Arc::new(FakePlatform::new());
        let host = FakeHost::new(&p, GuestRam::new());
        let mut c = config(&p, &host);
        c.enable_protected_content = false;
        let d = DeviceCore::create(p.clone(), c).expect("device");
        assert!(d.caps.protected_content);
    }

    // `mmioLength` is never read
    #[test]
    fn mmio_length_is_ignored() {
        let p = Arc::new(FakePlatform::new());
        let host = FakeHost::new(&p, GuestRam::new());
        let mut c = config(&p, &host);
        c.mmio_length = 0;
        assert!(DeviceCore::create(p.clone(), c).is_ok());
    }

    #[test]
    fn drop_stops_the_device() {
        let r = rig();
        r.with_root_fifo(1, 2);
        let m = r.mark();
        let crate::device::root::rig::Rig {
            platform, device, ..
        } = r;
        drop(device);
        let sels: Vec<_> = platform
            .call_log()
            .since(m)
            .into_iter()
            .filter(|c| c.class == "HostBlocks")
            .map(|c| c.sel)
            .collect();
        assert!(sels.ends_with(&["unmapMemory", "destroyTask"]), "{sels:?}");
        assert_eq!(sels.iter().filter(|s| **s == "unmapMemory").count(), 2);
    }

    /// The reporting handler's weak load is let go off the reporting queue. A host release
    /// while the handler runs is not the last: the shutdown runs after the handler, on a worker
    /// that does not own the reporting queue, so the shutdown's sync onto that queue runs and the
    /// device tears down, instead of dying at the sync inside the handler.
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: real-time waits for the worker
    fn reporting_handler_lets_the_device_go_off_the_queue() {
        use crate::device::root::rig::{Rig, rig_with};
        use crate::testkit::capture::DeathSlot;
        use crate::testkit::capture::with_slot;
        use std::time::{Duration, Instant};
        fn wait_until(what: &str, f: impl Fn() -> bool) {
            let start = Instant::now();
            while !f() {
                assert!(
                    start.elapsed() < Duration::from_secs(10),
                    "timed out: {what}"
                );
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        // The device's threads (the reporting worker's included) report deaths here.
        let slot = DeathSlot::new();
        with_slot(Some(slot.clone()), || {
            // The thread each `destroyTask` (the shutdown's) ran on.
            let threads = Arc::new(parking_lot::Mutex::new(Vec::new()));
            let seen = threads.clone();
            let r = rig_with(move |b| {
                let inner = b.destroy_task.clone().expect("destroyTask");
                b.destroy_task = Some(Arc::new(move |info| {
                    let name = std::thread::current().name().map(str::to_owned);
                    seen.lock().push(name);
                    inner(info)
                }));
            });
            r.with_root_fifo(1, 2);
            let m = r.mark();
            let Rig {
                platform,
                host,
                device,
            } = r;
            let handle = Arc::downgrade(&device);
            // The core, not a retain: it outlives the handle until the shutdown ends.
            let fields: Arc<DeviceCore<FakePlatform>> = Arc::clone(&device);
            // The handler blocks in `report_stamp_wait_timeout` on the root FIFO lock.
            let root = fields.root_fifo.lock();
            fields.report_stamp_wait_timeout_source.merge(1);
            // The host's reference and the handler's load.
            wait_until("the handler's weak load", || handle.strong_count() == 2);
            drop(device); // The host's release while the handler runs.
            assert_eq!(handle.strong_count(), 1);
            let host_sels = |platform: &FakePlatform| -> Vec<&'static str> {
                platform
                    .call_log()
                    .since(m)
                    .into_iter()
                    .filter(|c| c.class == "HostBlocks")
                    .map(|c| c.sel)
                    .collect()
            };
            assert_eq!(host_sels(&platform), Vec::<&str>::new());
            drop(root);
            // The block returns; its load, the last reference, goes off the queue. The shutdown
            // ends with its block on the reporting queue, which cancels the source once the
            // root's `destroyTask` has returned; a shutdown on that queue dies at the sync.
            wait_until("the shutdown", || {
                slot.death().is_some() || fields.report_stamp_wait_timeout_source.is_cancelled()
            });
            assert_eq!(slot.death(), None);
            assert_eq!(handle.strong_count(), 0);
            // An unnamed pooled worker, not a thread named after the queue.
            assert_eq!(
                *threads.lock(),
                [None],
                "the shutdown runs on an unnamed worker"
            );
            let sels = host_sels(&platform);
            assert!(sels.ends_with(&["unmapMemory", "destroyTask"]), "{sels:?}");
            // The shutdown's block on the queue cancelled the source.
            assert!(fields.report_stamp_wait_timeout_source.is_cancelled());
            drop(fields);
            drop(host);
        });
    }

    /// A display retains its device, so the host's last release does not shut the device
    /// down; the display's release does.
    #[test]
    fn display_keeps_device_alive_past_host_release() {
        use crate::display::descriptor::DisplayDescriptor;
        use crate::testkit::host::{FakeHostQueue, QueueMode};
        let r = rig();
        r.with_root_fifo(1, 2);
        let mut desc = DisplayDescriptor::sdk_defaults();
        desc.queue = Some(FakeHostQueue::new(QueueMode::Manual));
        let display = r.device.new_display(&desc, 0, 0x5150).expect("display");
        let m = r.mark();
        let crate::device::root::rig::Rig {
            platform, device, ..
        } = r;
        let host_sels = |platform: &FakePlatform| -> Vec<&'static str> {
            platform
                .call_log()
                .since(m)
                .into_iter()
                .filter(|c| c.class == "HostBlocks")
                .map(|c| c.sel)
                .collect()
        };
        drop(device); // The host's release; the display's retain keeps the device.
        assert_eq!(host_sels(&platform), Vec::<&str>::new());
        drop(display); // The last release: the shutdown.
        let sels = host_sels(&platform);
        assert!(sels.ends_with(&["unmapMemory", "destroyTask"]), "{sels:?}");
        assert_eq!(sels.iter().filter(|s| **s == "unmapMemory").count(), 2);
    }

    /// Stamps the teardown's quiesce publishes during the shutdown reach the device, whose
    /// back-links still upgrade then.
    #[test]
    fn drop_quiesce_publishes_held_stamps() {
        use crate::device::root::rig::{Rig, rig_with};
        use crate::guest::GuestPtr;
        // The stamp memory once the root FIFO is up, and its slot 0 at each raise (the
        // block runs with the mapping still live).
        let stamp_memory: Arc<Mutex<Option<GuestPtr>>> = Arc::new(Mutex::new(None));
        let at_raise = Arc::new(Mutex::new(Vec::new()));
        let (sm, seen) = (stamp_memory.clone(), at_raise.clone());
        let r = rig_with(move |b| {
            let inner = b.raise_interrupt.clone().expect("raiseInterrupt");
            b.raise_interrupt = Some(Arc::new(move |v| {
                let at = *sm.lock();
                if let Some(at) = at {
                    let site = DeathSite {
                        kind: DeathKind::NullDeref,
                        file: "test",
                        line: 0,
                    };
                    seen.lock().push(at.read_u32(0, site));
                }
                inner(v);
            }));
        });
        r.with_root_fifo(1, 2);
        *stamp_memory.lock() = *r.device.stamp_memory.lock();
        let fifo = r
            .device
            .root_fifo
            .lock()
            .root_fifo
            .clone()
            .expect("root fifo");
        // Stamp 5 faults at the head; stamp 6 completes behind it and is held.
        fifo.push_stamp(5, 13);
        fifo.push_stamp(6, 0);
        fifo.signal_stamp_value(6);
        fifo.fault_at_offset(0, 5);
        drop(fifo);
        // The fault's `signalFaultInFIFO:` block (it retains the device) has run.
        r.device.fifo_queue.sync(|| ());
        assert_eq!(*at_raise.lock(), [0]);
        let m = r.mark();
        let Rig {
            platform, device, ..
        } = r;
        drop(device); // -> teardownFifos -> [_rootFIFO stop]
        let sels: Vec<_> = platform
            .call_log()
            .since(m)
            .into_iter()
            .filter(|c| c.class == "HostBlocks")
            .map(|c| c.sel)
            .collect();
        // Slot 0 = 6, then the interrupt, before the unmaps.
        assert_eq!(
            sels,
            [
                "raiseInterrupt",
                "unmapMemory",
                "unmapMemory",
                "destroyTask"
            ]
        );
        assert_eq!(*at_raise.lock(), [0, 6]);
    }

    /// A device dropped with a live child FIFO makes the host calls a root page of 0 makes on a
    /// live device: mappings reach the device during its shutdown, so the child's command
    /// mapping is unmapped too.
    #[test]
    #[allow(clippy::disallowed_methods)] // test scaffolding: real-time wait for the child thread
    fn drop_with_live_child_unmaps_like_set_root_page_zero() {
        use crate::device::root::rig::{Rig, rig_with};
        use std::time::Duration;
        fn child_rig() -> Rig {
            let r = rig_with(|_| {});
            let node = crate::guest::records::APV_PAGE_SIZE + 0x400;
            r.set_ram_u32(node + 0x0c, 1); // child 1 stampIndex
            r.set_ram_u32(node + 0x10, 5); // pageListPage
            r.set_ram_u32(0x5 * crate::guest::records::APV_PAGE_SIZE, 6); // page list: page 6
            r.with_root_fifo(1, 2);
            assert!(r.device.create_fifo(1).unwrap());
            r
        }
        let a = child_rig();
        let m = a.mark();
        a.device.set_root_page(Hold::NONE, 0);
        std::thread::sleep(Duration::from_millis(200));
        let live = a.host_sels(m);
        assert_eq!(
            live,
            ["unmapMemory", "unmapMemory", "unmapMemory", "destroyTask"]
        );
        let b = child_rig();
        let m = b.mark();
        let Rig {
            platform,
            host,
            device,
        } = b;
        drop(device);
        std::thread::sleep(Duration::from_millis(200));
        let log: Vec<&'static str> = platform
            .call_log()
            .since(m)
            .into_iter()
            .filter(|c| c.class == "HostBlocks")
            .map(|c| c.sel)
            .collect();
        drop(host);
        assert_eq!(log, live);
    }

    /// A refused pipeline's line carries Metal's error, or says Metal gave none when it answered
    /// nil without one.
    #[test]
    fn failed_pipeline_logs_the_error() {
        use crate::device::test_support::Lines;
        let lines = Lines::capture();
        use crate::testkit::script::Step;
        let p = Arc::new(FakePlatform::new());
        let host = FakeHost::new(&p, GuestRam::new());
        p.script().push_n(
            "newComputePipelineStateWithFunction:error:",
            1,
            Step::Error(crate::platform::types::NsErrorText {
                domain: "MTLLibraryErrorDomain".to_owned(),
                code: 3,
                description: "no such function".to_owned(),
            }),
        );
        assert!(DeviceCore::create(p.clone(), config(&p, &host)).is_err());
        // The first pipeline's line only.
        assert_eq!(
            lines.all().last().map(String::as_str),
            Some(
                "device not created: Metal refused the BlitInRGB_2P_XR10_A8 pipeline: \
                 no such function"
            )
        );
        assert_eq!(lines.count("Metal refused"), 1);
        // The second pipeline answers nil without an error: `error` is still nil.
        let p = Arc::new(FakePlatform::new());
        let host = FakeHost::new(&p, GuestRam::new());
        p.script()
            .after("newComputePipelineStateWithFunction:error:", 1, Step::Nil);
        assert!(DeviceCore::create(p.clone(), config(&p, &host)).is_err());
        assert_eq!(
            lines.all().last().map(String::as_str),
            Some(
                "device not created: Metal refused the BlitOutRGB_2P_XR10_A8 pipeline: \
                 Metal gave no error"
            )
        );
        assert_eq!(lines.count("Metal refused"), 2);
    }

    /// The blit pipelines' pool is a real one, pushed before the shader library is made and
    /// popped at each return, so what Metal autoreleases drains there and not in the host's pool.
    mod blit_pipelines_pool {
        use super::*;
        use crate::rt::pool::test_native::{events, with_test_native};
        use crate::testkit::Step;

        fn run(p: &FakePlatform) -> (bool, bool, bool, Vec<String>) {
            let mut got = (false, false, false, Vec::new());
            with_test_native(|| {
                let b = setup_blit_pipelines(p, &p.device());
                got = (b.ok, b.blit_in.is_some(), b.blit_out.is_some(), events());
            });
            got
        }

        /// both pipelines made inside one pool, popped at `return true`; the library and both
        /// functions are gone by then.
        #[test]
        fn success_pops_the_pool_after_the_pipelines() {
            let p = FakePlatform::new();
            let got = run(&p);
            assert_eq!(
                got,
                (
                    true,
                    true,
                    true,
                    vec!["push 0x100".to_owned(), "pop 0x100".to_owned()]
                )
            );
            assert!(p.world().live("MTLLibrary").is_empty());
            assert!(p.world().live("MTLFunction").is_empty());
        }

        /// a nil library returns inside the pool, which pops.
        #[test]
        fn nil_library_pops_the_pool() {
            let p = FakePlatform::new();
            p.script()
                .push("newLibraryWithSource:options:error:", Step::Nil);
            let got = run(&p);
            assert_eq!(
                got,
                (
                    false,
                    false,
                    false,
                    vec!["push 0x100".to_owned(), "pop 0x100".to_owned()]
                )
            );
        }

        /// a refused pipeline returns inside the pool; the pool pops, and the library and
        /// functions go with it.
        #[test]
        fn refused_pipeline_pops_the_pool() {
            for refused in 1..=2usize {
                let p = FakePlatform::new();
                for _ in 1..refused {
                    p.script()
                        .push("newComputePipelineStateWithFunction:error:", Step::Default);
                }
                p.script().push(
                    "newComputePipelineStateWithFunction:error:",
                    Step::Error(NsErrorText {
                        domain: "MTLLibraryErrorDomain".into(),
                        code: 3,
                        description: "no".into(),
                    }),
                );
                let got = run(&p);
                assert_eq!(
                    got,
                    (
                        false,
                        refused == 2,
                        false,
                        vec!["push 0x100".to_owned(), "pop 0x100".to_owned()]
                    ),
                    "{refused}"
                );
                assert!(p.world().live("MTLLibrary").is_empty());
                assert!(p.world().live("MTLFunction").is_empty());
            }
        }
    }

    /// Each device info pair is one 8-byte store, so a vCPU that reads a pair with one 64-bit
    /// load sees the old pair or the new one, never a key with the old value.
    mod device_info_pair {
        use super::*;
        use crate::device::seam_impls::{
            DEVICE_INFO_WRITE, device_info_pair_word, write_device_info_pair,
        };
        use crate::guest::abi::suspend_blob::KeyValuePair;
        use crate::testkit::guest_ram::RawRegion;

        /// `APVKeyValuePair` is `{uint32_t key; uint32_t value;}`: the key in the low word.
        #[test]
        fn the_pair_is_one_little_endian_word() {
            let pair = KeyValuePair {
                key: 1,
                value: 0x8765_4321,
            };
            assert_eq!(device_info_pair_word(pair), 0x8765_4321_0000_0001);
            let ram = RawRegion::new(0x4000).expect("ram");
            let base = GuestPtr::new(ram.addr());
            base.write_u64(0x10, u64::MAX, DEVICE_INFO_WRITE);
            write_device_info_pair(base, 2, pair);
            assert_eq!(base.read_u32(0x10, DEVICE_INFO_WRITE), 1);
            assert_eq!(base.read_u32(0x14, DEVICE_INFO_WRITE), 0x8765_4321);
            assert_eq!(base.read_u64(8, DEVICE_INFO_WRITE), 0);
        }

        /// A reader polling pair 0 with 64-bit loads while the pair is rewritten over 0xff
        /// bytes sees only the two whole values: never `{1, 0xffffffff}` (a key stored before
        /// its value, as two 32-bit stores would show it).
        #[test]
        #[allow(clippy::disallowed_methods)] // test scaffolding: a second thread
        fn a_64_bit_reader_never_sees_a_torn_pair() {
            let ram = RawRegion::new(0x4000).expect("ram");
            let base = GuestPtr::new(ram.addr());
            let pair = KeyValuePair { key: 1, value: 4 };
            let whole = device_info_pair_word(pair);
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let torn = Arc::new(std::sync::atomic::AtomicU64::new(0));
            let reader = {
                let (stop, torn) = (stop.clone(), torn.clone());
                let addr = base.addr().addr();
                std::thread::spawn(move || {
                    let base = GuestPtr::new(crate::host::HostPtr::from_addr(addr));
                    while !stop.load(Ordering::Relaxed) {
                        let v = base.read_u64(0, DEVICE_INFO_WRITE);
                        if v != u64::MAX && v != whole {
                            torn.store(v, Ordering::Relaxed);
                        }
                    }
                })
            };
            for _ in 0..200_000 {
                base.write_u64(0, u64::MAX, DEVICE_INFO_WRITE);
                write_device_info_pair(base, 0, pair);
            }
            stop.store(true, Ordering::Relaxed);
            reader.join().expect("reader");
            assert_eq!(torn.load(Ordering::Relaxed), 0);
        }

        /// The device's `writeDeviceInfo` as `CmdGetDeviceInfo` calls it: every pair and the
        /// `{0, 0}` end land as 8-byte records.
        #[test]
        fn get_device_info_writes_whole_pairs() {
            use crate::device::test_support::{Opts, device};
            let t = device(Opts::default()).with_root_page();
            let page = 3u32;
            let at = u64::from(page) * crate::guest::records::APV_PAGE_SIZE;
            let ones = vec![0xffu8; crate::guest::records::APV_PAGE_SIZE as usize];
            assert!(t.host.ram().write(at, &ones));
            t.platform.fake_vm().sync();
            let dev: Arc<DeviceCore<FakePlatform>> = Arc::clone(&t.device);
            dev.get_device_info(3, 0x4000, page).expect("info");
            t.platform.fake_vm().sync();
            let mut bytes = [0u8; 24];
            assert!(t.host.ram().read(at, &mut bytes));
            let word = |i: usize| {
                let mut w = [0u8; 8];
                w.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
                u64::from_le_bytes(w)
            };
            // Key 1 (the sample count) and key 2, then the end, for a guest that knows keys 1-2.
            assert_eq!(word(0) & 0xffff_ffff, 1);
            assert_eq!(word(1) & 0xffff_ffff, 2);
            assert_eq!(word(2), 0);
        }
    }
}
