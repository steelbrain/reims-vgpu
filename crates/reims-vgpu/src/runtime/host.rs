//! Host memory + action sink abstractions for the device model.
//!
//! Production: QEMU C offers HostOps callbacks; Rust enqueues HostActions for a
//! QEMU BH. Tests: [`FakeHost`] owns an in-memory GPA space and action log.

/// Only [`FakeHost`]'s sparse byte store, X-register fixtures and guest-write
/// sets use this, and all three are test-gated.
#[cfg(test)]
use std::collections::BTreeMap;

/// Guest-physical memory access error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemError {
    Unmapped,
    NoCpu,
    Overflow,
    BadArgs,
    /// The QEMU host table omitted a mandatory guest-physical read callback.
    QemuReadGpaCallbackMissing,
    /// QEMU's guest-physical read callback returned a transaction failure.
    QemuReadGpaCallbackFailed(i32),
    /// The QEMU host table omitted a mandatory guest-physical write callback.
    QemuWriteGpaCallbackMissing,
    /// QEMU's guest-physical write callback returned a transaction failure.
    QemuWriteGpaCallbackFailed(i32),
    /// The QEMU host table omitted the guest-kernel-VA debug-read callback.
    QemuReadKvaCallbackMissing,
    /// QEMU's guest-kernel-VA debug-read callback failed for a reason other
    /// than the explicitly represented [`Self::NoCpu`] state.
    QemuReadKvaCallbackFailed(i32),
    /// The host cannot expose a CPU register on this pathway.
    XregUnavailable,
    /// The QEMU host table omitted the guest-register callback.
    QemuReadXregCallbackMissing,
    /// QEMU's guest-register callback rejected the register read.
    QemuReadXregCallbackFailed(i32),
    /// The guest page-table walk refused, carrying **which** of its fifteen
    /// checks did.
    ///
    /// Every GVA path used to answer `Unmapped` here, so a malformed PTE, a
    /// zero root PFN and an out-of-range address were one value — on top of the
    /// genuinely-unmapped GPA cases that also answer `Unmapped`. That is the
    /// "one status for N checks" shape the ground rules name by example, and it
    /// sat on the guest-memory hot path.
    Unresolved(reims_vgpu_paging::resolve::ResolveStatus),
    /// The task is not active, or its directory PFN is zero, so there is no page
    /// table to walk. Distinct from [`Self::Unresolved`]: the walk never began.
    NoTaskDirectory,
    /// No page-table geometry for the guest's page shift. A create-time
    /// configuration error, not a guest one.
    UnsupportedPageShift,
    /// The task root (directory → root PFN + depth) could not be read.
    TaskRootRead,
    /// Neither the wire task id nor its `>> 1` define-task form names an active
    /// task, so there is no address space to resolve against.
    NoSuchTask,
    /// A page of the span resolves to a GPA that is not guest RAM, so no host
    /// mapping can cover it (mapper / wild-PFN class).
    NotRam,
    /// [`HostPageViews::map_pages`] refused a **packed** page run the walk had already
    /// resolved — a RAMBlock or MemoryRegion edge, not a gap in the GPA list.
    /// Fragmentation alone never reaches here: the multi-import path splits a
    /// gapped span into packed runs and maps them one at a time.
    MapPagesRefused,
    /// A packed run's copy window fell outside the bytes `map_pages` returned or
    /// outside the caller's buffer. Run arithmetic, not a guest condition.
    RunOutOfRange,
    /// The walk resolved a page the caller was not authorised to write.
    ///
    /// Only a *deferred* write carries an authorisation set: it was armed
    /// against a page list at defer time, and the write that lands it much later
    /// must reach those pages and no others. A synchronous Store has no such set
    /// — the command being executed is what names its destination — and passes
    /// `None`, which can never produce this.
    ///
    /// Distinct from every other variant here because nothing is wrong with the
    /// walk: the page table is healthy and its answer is current. What is stale
    /// is the window, and the guest has already given that memory to somebody
    /// else.
    WriteOutsideWindow,
}

impl MemError {
    /// The guest's own page table says nothing is mapped at this address.
    ///
    /// A zero PFN in a task PTE is a *decoded guest fact*: the guest owns that
    /// entry and wrote the zero. So a deferred writeback whose target answers
    /// this has no target — the guest tore the range down — and that is a
    /// different outcome from a write that failed while the target still
    /// existed. Callers landing deferred content use it to pick between
    /// "discharge the obligation" and "report lost guest work".
    ///
    /// Deliberately **only** the zero-PFN status. The other fourteen walk
    /// refusals describe a table that is malformed, out of range or unreadable,
    /// none of which is the guest saying "I unmapped this", and widening the set
    /// to make a log quieter would turn this into the exception list the ground
    /// rules forbid.
    pub fn is_guest_teardown(&self) -> bool {
        matches!(
            self,
            Self::Unresolved(reims_vgpu_paging::resolve::ResolveStatus::ErrZeroPfn)
        )
    }
}

impl crate::observe::Decline for MemError {
    fn slug(&self) -> &'static str {
        match self {
            Self::Unmapped => "mem_unmapped",
            Self::NoCpu => "mem_no_cpu",
            Self::Overflow => "mem_overflow",
            Self::BadArgs => "mem_bad_args",
            Self::QemuReadGpaCallbackMissing => "mem_qemu_read_gpa_callback_missing",
            Self::QemuReadGpaCallbackFailed(_) => "mem_qemu_read_gpa_callback_failed",
            Self::QemuWriteGpaCallbackMissing => "mem_qemu_write_gpa_callback_missing",
            Self::QemuWriteGpaCallbackFailed(_) => "mem_qemu_write_gpa_callback_failed",
            Self::QemuReadKvaCallbackMissing => "mem_qemu_read_kva_callback_missing",
            Self::QemuReadKvaCallbackFailed(_) => "mem_qemu_read_kva_callback_failed",
            Self::XregUnavailable => "mem_xreg_unavailable",
            Self::QemuReadXregCallbackMissing => "mem_qemu_read_xreg_callback_missing",
            Self::QemuReadXregCallbackFailed(_) => "mem_qemu_read_xreg_callback_failed",
            // Delegates, so the walk's own fifteen slugs stay the reason rather
            // than being flattened into one and reconstructed from a field.
            Self::Unresolved(status) => match crate::runtime::gva_refusal::slug(*status) {
                Some(slug) => slug,
                // `Unresolved(Ok)` is a construction bug, not a walk failure.
                // Naming it beats reporting a plausible walk reason for
                // something the walk never said.
                None => "mem_unresolved_ok",
            },
            Self::NoTaskDirectory => "mem_no_task_directory",
            Self::UnsupportedPageShift => "mem_unsupported_page_shift",
            Self::TaskRootRead => "mem_task_root_read",
            Self::NoSuchTask => "mem_no_such_task",
            Self::NotRam => "mem_not_ram",
            Self::MapPagesRefused => "mem_map_pages_refused",
            Self::RunOutOfRange => "mem_run_out_of_range",
            Self::WriteOutsideWindow => "mem_write_outside_window",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::QemuReadGpaCallbackFailed(rc)
            | Self::QemuWriteGpaCallbackFailed(rc)
            | Self::QemuReadKvaCallbackFailed(rc)
            | Self::QemuReadXregCallbackFailed(rc) => vec![("rc", rc.to_string())],
            _ => Vec::new(),
        }
    }
}

/// Checked guest physical memory (no scanning; directed GPA only).
pub trait HostMemory {
    fn read_gpa(&self, gpa: u64, buf: &mut [u8]) -> Result<(), MemError>;
    fn write_gpa(&mut self, gpa: u64, buf: &[u8]) -> Result<(), MemError>;
}

/// Typed actions for the QEMU main loop (or FakeHost log).
///
/// `#[repr(u32)]` because this enum *is* the `kind` word of the C
/// `ReimsVgpuHostAction` the BH pops — there is no second FFI spelling to drift
/// against. Every discriminant is pinned to its `REIMS_VGPU_HOST_ACTION_*`
/// header define by `the_abi_header_agrees_on_the_host_action_table`.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostActionKind {
    None = 0,
    IrqGfxPulse = 1,
    IrqIosfcPulse = 2,
    ScanoutUpdate = 3,
    CursorUpdate = 4,
    Trace = 5,
    /// New software cursor glyph ready in device state (C pulls via ABI).
    CursorGlyph = 6,
    // 7 is a retired wire value: it named a pre-host-window QEMU GL/dmabuf
    // scanout action that no longer exists on either side. Every discriminant
    // below is written out so removing it did not renumber the wire; do not
    // reuse 7 for a new action.
    /// A guest keyboard key from the host-owned window (see
    /// [`crate::runtime::input`]). `a0` = Linux evdev keycode (`KEY_*`),
    /// `a1` = 1 down / 0 up. The window thread maps the platform key into the
    /// stable evdev space; the C shim forwards it verbatim via
    /// `qemu_input_event_send_key_linux` (QEMU owns the evdev→qcode table), so
    /// no QEMU keycode constants leak into Rust. See [`HostAction::input_key`].
    InputKey = 8,
    /// Absolute pointer move from the host-owned window. `a0` = x pixel,
    /// `a1` = y pixel, `a2` = surface width (px), `a3` = surface height (px).
    /// Absolute (not relative) because the guest binds an absolute pointer
    /// (usb-tablet); the C shim scales into the abs axis range with
    /// `qemu_input_queue_abs` (min_in = 0, max_in = dim). See
    /// [`HostAction::input_pointer_move`].
    InputPointerMove = 9,
    /// Pointer button (including wheel) from the host-owned window. `a0` = the
    /// neutral [`crate::runtime::input::ReimsVgpuButton`] code, `a1` = 1 down / 0 up.
    /// The C shim maps the neutral code to QEMU's `InputButton`; a wheel click
    /// is a down+up pair the window thread emits, so the C side stays uniform
    /// (one `qemu_input_queue_btn` + sync per action). See
    /// [`HostAction::input_pointer_button`].
    InputPointerButton = 10,
    /// The host-owned window was closed through its UI (title-bar close /
    /// compositor close). Carries no payload; the C shim turns it into
    /// `qemu_system_shutdown_request` so closing the window shuts the VM down —
    /// the window is the VM's display, so closing it is closing the machine.
    /// See [`HostAction::window_closed`].
    WindowClosed = 11,
}

/// One queued action, in the exact layout the C `ReimsVgpuHostAction` declares.
///
/// `#[repr(C)]` so `reims_vgpu_qemu_device_pop_action` can write this type
/// straight into the caller's out-pointer. The queue is write-only from Rust —
/// the shim reads `kind` and dispatches — so no value the C side chose ever
/// reaches [`HostActionKind`], and a `#[repr(u32)]` enum in the `kind` slot
/// cannot be handed an out-of-range discriminant.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostAction {
    pub kind: HostActionKind,
    pub a0: u64,
    pub a1: u64,
    pub a2: u64,
    pub a3: u64,
}

impl Default for HostAction {
    fn default() -> Self {
        Self {
            kind: HostActionKind::None,
            a0: 0,
            a1: 0,
            a2: 0,
            a3: 0,
        }
    }
}

impl HostAction {
    pub fn irq_gfx() -> Self {
        Self {
            kind: HostActionKind::IrqGfxPulse,
            a0: 0,
            a1: 0,
            a2: 0,
            a3: 0,
        }
    }

    pub fn irq_iosfc() -> Self {
        Self {
            kind: HostActionKind::IrqIosfcPulse,
            a0: 0,
            a1: 0,
            a2: 0,
            a3: 0,
        }
    }

    pub fn scanout_gen(mapping_id: u32, width: u32, height: u32, generation: u32) -> Self {
        Self {
            kind: HostActionKind::ScanoutUpdate,
            a0: mapping_id as u64,
            a1: width as u64,
            a2: height as u64,
            a3: generation as u64,
        }
    }

    pub fn cursor(x: u16, y: u16, show: bool) -> Self {
        Self {
            kind: HostActionKind::CursorUpdate,
            a0: x as u64,
            a1: y as u64,
            a2: u64::from(show),
            a3: 0,
        }
    }

    pub fn cursor_glyph() -> Self {
        Self {
            kind: HostActionKind::CursorGlyph,
            a0: 0,
            a1: 0,
            a2: 0,
            a3: 0,
        }
    }

    /// Guest keyboard key from the host-owned window. `evdev_keycode` is a Linux
    /// `KEY_*` code (the stable neutral space the window thread maps into); the
    /// C shim hands it straight to `qemu_input_event_send_key_linux`.
    pub fn input_key(evdev_keycode: u32, down: bool) -> Self {
        Self {
            kind: HostActionKind::InputKey,
            a0: u64::from(evdev_keycode),
            a1: u64::from(down),
            a2: 0,
            a3: 0,
        }
    }

    /// Absolute pointer move from the host-owned window. `x`/`y` are pixel
    /// coordinates within a `width`x`height` surface; the C shim scales them
    /// into the abs axis range. `width`/`height` must be non-zero (the window
    /// surface is always sized before a move is emitted).
    pub fn input_pointer_move(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            kind: HostActionKind::InputPointerMove,
            a0: u64::from(x),
            a1: u64::from(y),
            a2: u64::from(width),
            a3: u64::from(height),
        }
    }

    /// Pointer button (or wheel) from the host-owned window. Wheel clicks are
    /// emitted as a `down` then `up` pair by the window thread.
    pub fn input_pointer_button(
        button: crate::runtime::input::ReimsVgpuButton,
        down: bool,
    ) -> Self {
        Self {
            kind: HostActionKind::InputPointerButton,
            a0: u64::from(button as u32),
            a1: u64::from(down),
            a2: 0,
            a3: 0,
        }
    }

    /// The host-owned window was closed through its UI. No payload; the C shim
    /// requests a VM shutdown when it applies this.
    pub fn window_closed() -> Self {
        Self {
            kind: HostActionKind::WindowClosed,
            a0: 0,
            a1: 0,
            a2: 0,
            a3: 0,
        }
    }
}

/// Which check refused the guest-RAM span enumeration.
///
/// One variant per negative `REIMS_VGPU_GUEST_RAM_ERR_*` code in the shared ABI
/// header, plus the failures that are Rust's own: a shim too old to offer the
/// callback, a code this build does not recognise, and an answer that did not
/// fit the array twice running.
///
/// This is the door to every guest-memory import, so a refusal here is not a
/// slow path — it is the device running its copying rails for the whole boot.
/// The variants stay distinct because they send a reader to different places: a
/// missing callback is a shim/staticlib version mismatch, an empty address space
/// is a machine wiring problem, and a short array is ours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuestRamRegionsError {
    /// The shim offers no `guest_ram_regions`. Every pre-v17 shim, and every
    /// fixture host.
    CallbackMissing,
    /// The shim rejected the arguments, or had more spans than its return value
    /// could carry.
    Args,
    /// The system address space holds no writable RAM span. A machine with no
    /// memory, or a call made before the board finished wiring its RAM up.
    NoRam,
    /// The shim reported more spans than the array we grew to hold them, twice
    /// running. The retry sizes itself from the first answer, so this means the
    /// span count changed underneath us — which for RAMBlock mappings it must
    /// not.
    StillTruncated { total: usize, capacity: usize },
    /// A negative code this build has no name for, which means the shim is
    /// newer than the staticlib. Carried rather than folded into another
    /// variant so the number itself reaches the log.
    UnknownCode(i32),
}

impl GuestRamRegionsError {
    /// Map a negative shim return to the check it names.
    pub fn from_code(code: i32) -> Self {
        match code {
            crate::qemu::abi::REIMS_VGPU_GUEST_RAM_ERR_ARGS => Self::Args,
            crate::qemu::abi::REIMS_VGPU_GUEST_RAM_ERR_NO_RAM => Self::NoRam,
            other => Self::UnknownCode(other),
        }
    }
}

impl crate::observe::Decline for GuestRamRegionsError {
    fn slug(&self) -> &'static str {
        match self {
            Self::CallbackMissing => "guest_ram_regions_callback_missing",
            Self::Args => "guest_ram_regions_args",
            Self::NoRam => "guest_ram_regions_no_ram",
            Self::StillTruncated { .. } => "guest_ram_regions_still_truncated",
            Self::UnknownCode(_) => "guest_ram_regions_unknown_code",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::StillTruncated { total, capacity } => vec![
                ("total", total.to_string()),
                ("capacity", capacity.to_string()),
            ],
            Self::UnknownCode(code) => vec![("code", code.to_string())],
            _ => Vec::new(),
        }
    }
}

crate::observe::decline::decline_display!(GuestRamRegionsError);

/// Clock, wake, and typed host-effect delivery.
pub trait HostControl {
    fn mono_ns(&self) -> u64;
    fn enqueue(&mut self, action: HostAction);
    fn schedule_bh(&mut self);
}

/// Directed guest CPU and kernel-virtual-address introspection.
pub trait GuestCpuAccess {
    /// Read guest kernel virtual address (cpu_memory_rw_debug). Default: fail.
    fn read_kva(&self, _kva: u64, _buf: &mut [u8]) -> Result<(), MemError> {
        Err(MemError::Unmapped)
    }

    /// Read guest CPU X-register `index` (0..30). Default: none.
    /// Used only on the MMIO path that publishes an iosfc mapper request so
    /// x19/x21/x22 still hold the directed MappingInternal handoff.
    fn read_xreg(&self, _index: u32) -> Result<u64, MemError> {
        Err(MemError::XregUnavailable)
    }
}

/// Stable RAMBlock discovery for whole-guest-memory import.
pub trait GuestRamProvider {
    fn guest_ram_regions(
        &mut self,
    ) -> Result<Vec<crate::runtime::guest_ram::GuestRamRegion>, GuestRamRegionsError> {
        Err(GuestRamRegionsError::CallbackMissing)
    }
}

/// Ownership-bearing views over directed sets of guest pages.
pub trait HostPageViews {
    /// Build one contiguous host-VA view over guest pages (page-aligned GPAs).
    ///
    /// `page_size` is the guest page size (4 KiB x86 / 16 KiB arm64e). Each
    /// `gpas[i]` is one guest page base. View length is `gpas.len() * page_size`.
    /// ParavirtualizedGraphics mapMemory model: the view aliases guest RAM so
    /// CPU/GPU access *is* guest memory. Default: unavailable.
    fn map_pages(&mut self, _gpas: &[u64], _page_size: usize) -> Option<usize> {
        None
    }

    /// Build a packed guest-page alias whose exact trailing extent is
    /// anonymous host memory rather than guest RAM.
    fn map_pages_with_padding(
        &mut self,
        _gpas: &[u64],
        _page_size: usize,
        _total_len: usize,
    ) -> Option<usize> {
        None
    }

    /// Release a view obtained from [`HostPageViews::map_pages`].
    fn unmap_pages(&mut self, _ptr: usize, _len: usize) {}

    /// True when [`HostPageViews::map_pages`] returns an alias that may be
    /// retained and imported by the backend. It stays valid until the matching
    /// [`HostPageViews::unmap_pages`] after backend access has completed.
    ///
    /// This is a claim about a CPU-side *view* only, and says nothing about the
    /// GPU rail: the backend may import this exact view, so its completion is
    /// part of the release ordering.
    ///
    /// Default `false` — the conservative answer, so a host that has not
    /// declared stability keeps the portable CPU writeback.
    fn map_pages_stable(&self) -> bool {
        false
    }

    /// Where guest RAM lives in this process, as stable spans held for the VM's
    /// lifetime.
    ///
    /// The whole guest-memory import rail starts here: the backend imports each
    /// span once and every later reference is a
    /// [`crate::runtime::guest_ram::GuestSlice`] inside one of them. Called at
    /// device init and not again — the answer does not change, and re-importing
    /// pays the driver's page pinning for an answer that is already known.
    ///
    /// Deliberately not [`HostPageViews::map_pages`] with a different return type.
    /// That call answers about specific pages and on the sysbus shim may build a
    /// transient `mach_vm_remap` view the caller has to release; this one never
    /// allocates and never releases.
    ///
    /// Default: unavailable. A host that cannot answer says so by name, and the
    /// caller runs the copying rails rather than reaching for `map_pages`.
    /// True if `gpa` is guest RAM (not MMIO / ROM / unmapped). Product QEMU
    /// implements via `address_space_translate` + `memory_region_is_ram`.
    /// Default: true (fixtures / NullHost without a RAM map).
    ///
    /// **Answers for one address.** A caller holding a span must ask
    /// [`HostPageViews::first_non_ram_page`] instead — see the sampling trap recorded
    /// there.
    fn is_ram_gpa(&self, _gpa: u64) -> bool {
        true
    }

    /// The first page of `[gpa, gpa + len)` that is not guest RAM, or `None`
    /// when every page of it is.
    ///
    /// # Two endpoints are not a span
    ///
    /// [`HostPageViews::is_ram_gpa`] answers about a single address, and a caller that
    /// wants to know whether a *range* is readable has to ask about every page
    /// of it. Testing the first and last byte reads as thorough and is not: a
    /// guest-physical range is a walk through whatever regions the machine
    /// happens to lay out, and one non-RAM page anywhere between two RAM
    /// endpoints is enough for `read_gpa` to refuse. The EFI console capture
    /// held exactly that shape — two `is_ram_gpa` calls vouching for an 8 MB
    /// framebuffer span — and a driven x86 boot refused a row 375 rows into it,
    /// after 375 completed reads, with the endpoints both answering RAM.
    ///
    /// Returning the offending page rather than a `bool` is what lets a refusal
    /// name it. A caller that only needs the verdict tests `.is_none()`.
    ///
    /// Walks at `page_size` because that is the granularity every other RAM
    /// check in this crate uses (`map_pages` checks its page list the same way),
    /// and because a memory region finer than a guest page cannot back a
    /// mapping this device would read through. Short-circuits on the first
    /// failure, so the common "this door is shut" case costs one call and not
    /// one per page.
    fn first_non_ram_page(&self, gpa: u64, len: u64, page_size: usize) -> Option<u64> {
        if len == 0 || page_size == 0 {
            return None;
        }
        let step = page_size as u64;
        let last = gpa.saturating_add(len - 1);
        // Modulo rather than a `!(step - 1)` mask: nothing here requires
        // `page_size` to be a power of two, and a caller that passed one that
        // is not would get a silently wrong base from the mask.
        let last_page = last - (last % step);
        let mut page = gpa - (gpa % step);
        while page <= last_page {
            if !self.is_ram_gpa(page) {
                return Some(page);
            }
            // An overflow here means the walk reached the top of the address
            // space with every page so far answering RAM, so `None` — no
            // offending page — is the right answer and not a lost check.
            page = page.checked_add(step)?;
        }
        None
    }
}

/// Compatibility bound for operations which genuinely consume every host
/// port. New helpers should name only the narrow traits they use.
pub trait HostOps: HostControl + GuestCpuAccess + GuestRamProvider + HostPageViews {}

impl<T: HostControl + GuestCpuAccess + GuestRamProvider + HostPageViews + ?Sized> HostOps for T {}

/// Arm64e guest page size, from the contract shift rather than a literal.
/// FakeHost / `map_pages` test fixture only — product paths use
/// `state.page_size()` / `page_size_of(page_shift)`.
///
/// Arch-qualified with no portable alias, deliberately. A bare `GUEST_PAGE_SIZE_ARM64E`
/// reads as "the guest's page size" and is 16 KiB whatever the guest is, which
/// is the spelling `model::regs` records as the cause of x86 wild writes.
pub const GUEST_PAGE_SIZE_ARM64E: usize = 1usize << reims_vgpu_paging::geometry::PAGE_SHIFT_ARM64E;

/// mach VM aliasing for FakeHost views — the same mechanism the QEMU shim
/// uses in production (`mach_vm_remap` of guest RAM), exercised for real in
/// unit tests so view coherence is tested, not simulated.
#[cfg(all(test, target_os = "macos"))]
mod mach_vm {
    #[allow(non_upper_case_globals)]
    extern "C" {
        pub static mach_task_self_: u32;
        pub fn mach_vm_allocate(task: u32, addr: *mut u64, size: u64, flags: i32) -> i32;
        pub fn mach_vm_deallocate(task: u32, addr: u64, size: u64) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn mach_vm_remap(
            target: u32,
            addr: *mut u64,
            size: u64,
            mask: u64,
            flags: i32,
            src_task: u32,
            src_addr: u64,
            copy: i32,
            cur_protection: *mut i32,
            max_protection: *mut i32,
            inheritance: u32,
        ) -> i32;
    }
    pub const VM_FLAGS_ANYWHERE: i32 = 1;
    pub const VM_FLAGS_FIXED_OVERWRITE: i32 = 0x4000;
    pub const VM_INHERIT_NONE: u32 = 2;
}

#[cfg(test)]
/// A real, 16KiB-aligned memory block backing a GPA range in [`FakeHost`].
#[derive(Debug)]
struct RealRange {
    gpa: u64,
    len: usize,
    ptr: usize,
    alloc_len: usize,
}

#[cfg(test)]
/// Combined host for unit tests: GPA store + action log + BH flag.
///
/// GPA ranges are backed by real page-aligned host memory so
/// [`HostPageViews::map_pages`] views work exactly like production (mach_vm_remap
/// aliasing): a GPU/CPU write through a view is immediately visible via
/// `read_gpa` and vice versa. Bytes outside mapped ranges live in a sparse
/// map (synthetic KVA fixtures); unmapped reads stay permissive zeros.
#[derive(Debug, Default)]
pub struct FakeHost {
    ranges: Vec<RealRange>,
    /// Sparse byte store for addresses outside real ranges.
    pages: BTreeMap<u64, u8>,
    /// Live map_pages views (ptr, len) for cleanup (mach remap or bounce).
    views: Vec<(usize, usize)>,
    /// Linux bounce buffers: host ptr must be written back to GPA on unmap.
    bounce: Vec<BounceView>,
    /// Synthetic guest X-regs for mapper capture tests.
    pub xregs: BTreeMap<u32, u64>,
    pub actions: Vec<HostAction>,
    pub mono_ns: u64,
    pub bh_scheduled: bool,
    /// When true (any host platform): `map_pages` models a host that can return
    /// only an already-packed sequential alias. The product x86 shim can also
    /// reconstruct scattered shared pages; this narrower fixture exercises the
    /// refusal and multi-run fallback arms.
    pub strict_linux_map: bool,
    /// Test-controlled answer for [`HostPageViews::map_pages_stable`]. Keep separate
    /// from `strict_linux_map`: packed shape and pointer lifetime are distinct
    /// host contracts.
    pub stable_map_pages: bool,
    /// Number of HostOps page-import attempts (test proxy for import amplification).
    pub map_pages_calls: u64,
    /// Half-open GPA ranges this host reports as **not** guest RAM, so a test
    /// can model device memory — a PCI BAR — and not only mapped vs unmapped.
    ///
    /// Empty by default, which is exactly the previous behaviour: `is_ram_gpa`
    /// answered a flat `true`, so nothing could exercise a caller's non-RAM arm.
    /// Arm it with [`FakeHost::mark_non_ram`] or, for a range that stops being
    /// RAM partway through a loop, [`FakeHost::arm_unmap_on_read`].
    ///
    /// Interior-mutable for the second of those: the fixture has to be able to
    /// change this answer from `&self`, because the loop that observes the
    /// change holds the host immutably — which is also the product's position.
    non_ram: std::cell::RefCell<Vec<(u64, u64)>>,
    /// Scripted guest page-table edits, armed by [`FakeHost::arm_rewire`].
    rewires: std::cell::RefCell<Vec<Rewire>>,
    /// Scripted mid-loop retractions of guest RAM, armed by
    /// [`FakeHost::arm_unmap_on_read`]: `(on_read_gpa, on_read_len, base, end)`.
    unmap_on_read: std::cell::RefCell<Vec<(u64, u64, u64, u64)>>,
    /// How many armed rewires have fired, so a test can assert its trigger hit.
    rewires_fired: std::cell::Cell<u64>,
}

#[cfg(test)]
/// A guest page-table edit that fires from inside a device guest read.
///
/// The corruption class this harness exists to test is defined by something the
/// guest does *while a device loop is running*: a copy takes an address from a
/// command, walks it again per row, and the guest re-points it in between. A
/// test that edits the page table before or after the call cannot express that,
/// because the interesting instant is inside it — so a bound that only ever saw
/// a settled page table would pass whether or not it worked.
///
/// The trigger is a guest read overlapping `[on_read_gpa, +on_read_len)` rather
/// than a read ordinal. Every one of these loops reads its source per row, so
/// naming the source row that must already have been read states the timing in
/// the loop's own terms; an ordinal would be a number picked by watching one run
/// and would move the moment anything else read guest memory.
///
/// Fires at most once.
#[derive(Debug)]
pub struct Rewire {
    /// Guest read address whose access fires this.
    pub on_read_gpa: u64,
    /// Length of the triggering read window.
    pub on_read_len: u64,
    /// Guest physical address of the page-table entry to overwrite.
    pub pte_gpa: u64,
    /// Bytes to write there.
    pub bytes: Vec<u8>,
}

/// The host's own page size.
///
/// Distinct from every guest page size in this crate, and the distinction is
/// load-bearing on Apple Silicon: a 16 KiB host page cannot express an x86
/// guest's 4 KiB page as an independent mapping.
///
/// Gated to macOS because its only caller is: `mach_vm_remap` is what needs host
/// page granularity, and the non-macOS arm of `map_pages` bounces instead. The
/// gate used to be `#[cfg(test)]` alone, which was invisible because `libc` is
/// declared only for macOS — so the x86_64-linux arm of the feature matrix
/// failed to compile its tests rather than reporting this as dead.
#[cfg(all(test, target_os = "macos"))]
fn host_page_size() -> usize {
    // SAFETY: `sysconf` takes an int and returns a long; it touches no memory.
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v > 0 {
        v as usize
    } else {
        GUEST_PAGE_SIZE_ARM64E
    }
}

/// Contiguous bounce for [`FakeHost::map_pages`] where an alias is not expressible.
#[cfg(test)]
#[derive(Debug)]
struct BounceView {
    ptr: usize,
    len: usize,
    gpas: Vec<u64>,
    page_sz: usize,
}

#[cfg(test)]
impl Drop for FakeHost {
    fn drop(&mut self) {
        // Bounce views are heap allocations on every platform, and they are
        // also tracked in `self.views`. Free them first and drop their
        // tracking, or the macOS arm below would hand a `Box` pointer to
        // `mach_vm_deallocate`.
        for b in self.bounce.drain(..) {
            self.views.retain(|&(p, l)| !(p == b.ptr && l == b.len));
            // SAFETY: ptr from Box::into_raw in `bounce_view`, with this len.
            unsafe {
                let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(b.ptr as *mut u8, b.len));
            }
        }
        #[cfg(target_os = "macos")]
        unsafe {
            for (ptr, len) in self.views.drain(..) {
                mach_vm::mach_vm_deallocate(mach_vm::mach_task_self_, ptr as u64, len as u64);
            }
            for r in self.ranges.drain(..) {
                mach_vm::mach_vm_deallocate(
                    mach_vm::mach_task_self_,
                    r.ptr as u64,
                    r.alloc_len as u64,
                );
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            for r in self.ranges.drain(..) {
                // The layout `alloc_block` allocated with, and no fallback.
                // `dealloc` requires the *allocating* layout, so the `align = 1`
                // fallback that used to sit here was undefined behaviour on the
                // one path it could have run — and it could not run: alloc_block
                // returns `None` when this exact layout cannot be built, so a
                // block that exists proves it can. `unwrap_or` also evaluated
                // that wrong layout on every drop, not only on failure.
                let layout =
                    std::alloc::Layout::from_size_align(r.alloc_len, GUEST_PAGE_SIZE_ARM64E)
                        .expect(
                            "alloc_block built this layout; the block could not exist otherwise",
                        );
                // SAFETY: ptr and alloc_len come from alloc_block, and the
                // layout is the one it allocated with.
                unsafe { std::alloc::dealloc(r.ptr as *mut u8, layout) };
            }
        }
    }
}

#[cfg(test)]
impl FakeHost {
    pub fn new() -> Self {
        Self::default()
    }

    /// Report `[base, base + len)` as device memory rather than guest RAM, so
    /// [`HostPageViews::is_ram_gpa`] answers `false` inside it.
    ///
    /// Models a PCI BAR. Reads and writes still work through this fixture —
    /// the point is the *classification*, which is what production QEMU refuses
    /// on (`MemTxAttrs.memory`), not whether bytes happen to be reachable here.
    pub fn mark_non_ram(&mut self, base: u64, len: u64) {
        self.non_ram
            .borrow_mut()
            .push((base, base.saturating_add(len)));
    }

    /// Stop answering RAM for `[base, base + len)` once a read touches
    /// `[on_read_gpa, on_read_gpa + on_read_len)`.
    ///
    /// The guest half of a race this device cannot close: a caller pre-flights a
    /// span, finds it all RAM, and then copies it a row at a time while the
    /// guest is free to unmap it underneath. A `Rewire` models the guest editing
    /// its page tables mid-loop; this models it retracting the memory, which
    /// `Rewire` cannot express because `non_ram` is not page-table bytes.
    ///
    /// Fires from `read_gpa`, the same point and for the same reason: it is the
    /// one operation every guest-memory loop performs, so the change lands
    /// between two iterations without the loop knowing.
    pub fn arm_unmap_on_read(&mut self, on_read_gpa: u64, on_read_len: u64, base: u64, len: u64) {
        self.unmap_on_read.borrow_mut().push((
            on_read_gpa,
            on_read_len,
            base,
            base.saturating_add(len),
        ));
    }

    fn alloc_block(len: usize) -> Option<(usize, usize)> {
        #[cfg(target_os = "macos")]
        unsafe {
            let alloc_len = len.max(1).next_multiple_of(GUEST_PAGE_SIZE_ARM64E);
            let mut addr = 0u64;
            let kr = mach_vm::mach_vm_allocate(
                mach_vm::mach_task_self_,
                &mut addr,
                alloc_len as u64,
                mach_vm::VM_FLAGS_ANYWHERE,
            );
            if kr != 0 {
                return None;
            }
            Some((addr as usize, alloc_len))
        }
        #[cfg(not(target_os = "macos"))]
        {
            // Real host pages so map_pages can return an aliasing pointer (product
            // contig write path). Align to 16 KiB so arm fixtures work.
            let alloc_len = len.max(1).next_multiple_of(GUEST_PAGE_SIZE_ARM64E);
            let layout =
                std::alloc::Layout::from_size_align(alloc_len, GUEST_PAGE_SIZE_ARM64E).ok()?;
            // SAFETY: non-zero layout.
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            if ptr.is_null() {
                return None;
            }
            Some((ptr as usize, alloc_len))
        }
    }

    fn range_containing(&self, gpa: u64) -> Option<usize> {
        self.ranges
            .iter()
            .position(|r| gpa >= r.gpa && gpa < r.gpa + r.len as u64)
    }

    /// Arm a [`Rewire`]: the guest edits its page table mid-loop.
    pub fn arm_rewire(&mut self, rewire: Rewire) {
        self.rewires.borrow_mut().push(rewire);
    }

    /// Rewires that have fired so far.
    pub fn rewires_fired(&self) -> u64 {
        self.rewires_fired.get()
    }

    /// Fire any armed [`Rewire`] whose trigger window this read touches.
    ///
    /// Called from `read_gpa` — the one operation every guest-memory loop
    /// performs — so the edit lands between two iterations without the loop
    /// knowing anything about it, which is exactly the guest's position.
    fn fire_rewires(&self, gpa: u64, len: usize) {
        self.fire_unmaps(gpa, len);
        if self.rewires.borrow().is_empty() {
            return;
        }
        let end = gpa.saturating_add(len as u64);
        let mut fired: Vec<Rewire> = Vec::new();
        self.rewires.borrow_mut().retain(|r| {
            let hit = gpa < r.on_read_gpa.saturating_add(r.on_read_len) && r.on_read_gpa < end;
            if hit {
                fired.push(Rewire {
                    on_read_gpa: r.on_read_gpa,
                    on_read_len: r.on_read_len,
                    pte_gpa: r.pte_gpa,
                    bytes: r.bytes.clone(),
                });
            }
            !hit
        });
        for r in fired {
            self.poke(r.pte_gpa, &r.bytes);
            self.rewires_fired.set(self.rewires_fired.get() + 1);
        }
    }

    /// Retract any armed range whose trigger window this read touches.
    ///
    /// Ordered *before* the read it triggers on rather than after, because the
    /// case being modelled is a read that fails: the guest unmapped the page and
    /// the caller's read of it is the operation that discovers so.
    fn fire_unmaps(&self, gpa: u64, len: usize) {
        if self.unmap_on_read.borrow().is_empty() {
            return;
        }
        let end = gpa.saturating_add(len as u64);
        let mut retracted: Vec<(u64, u64)> = Vec::new();
        self.unmap_on_read
            .borrow_mut()
            .retain(|&(on_gpa, on_len, base, range_end)| {
                let hit = gpa < on_gpa.saturating_add(on_len) && on_gpa < end;
                if hit {
                    retracted.push((base, range_end));
                }
                !hit
            });
        self.non_ram.borrow_mut().extend(retracted);
    }

    /// Write `bytes` at `gpa` through a live range, from `&self`.
    ///
    /// A guest vCPU writes its own page table without asking the device, so the
    /// harness must be able to as well. Only real ranges: a page table lives in
    /// mapped guest RAM in every fixture, and silently landing a rewire in the
    /// sparse side store would let a test that never armed anything pass.
    fn poke(&self, gpa: u64, bytes: &[u8]) {
        let Some(i) = self.range_containing(gpa) else {
            panic!("FakeHost::poke: {gpa:#x} is not in a mapped range");
        };
        let r = &self.ranges[i];
        let off = (gpa - r.gpa) as usize;
        assert!(
            off + bytes.len() <= r.len,
            "FakeHost::poke: {gpa:#x}+{} runs past its range",
            bytes.len()
        );
        // SAFETY: `off + bytes.len() <= r.len` bytes are live at `r.ptr`.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), (r.ptr + off) as *mut u8, bytes.len());
        }
    }

    /// Register a real range at `gpa`, seeding from any sparse bytes there.
    fn provision_range(&mut self, gpa: u64, len: usize) -> Option<usize> {
        let (ptr, alloc_len) = Self::alloc_block(len)?;
        // Seed from sparse bytes previously written at these addresses.
        for off in 0..len as u64 {
            if let Some(b) = self.pages.remove(&gpa.wrapping_add(off)) {
                unsafe { *((ptr + off as usize) as *mut u8) = b };
            }
        }
        self.ranges.push(RealRange {
            gpa,
            len,
            ptr,
            alloc_len,
        });
        Some(self.ranges.len() - 1)
    }

    /// Map a contiguous GPA range filled with `fill` (or zeros).
    pub fn map_range(&mut self, gpa: u64, len: usize, fill: u8) {
        if len == 0 {
            return;
        }
        // Fully inside an existing range: fill in place.
        if let Some(i) = self.range_containing(gpa) {
            let r = &self.ranges[i];
            if gpa + len as u64 <= r.gpa + r.len as u64 {
                let off = (gpa - r.gpa) as usize;
                unsafe { std::ptr::write_bytes((r.ptr + off) as *mut u8, fill, len) };
                return;
            }
        }
        debug_assert!(
            !self
                .ranges
                .iter()
                .any(|r| gpa < r.gpa + r.len as u64 && r.gpa < gpa + len as u64),
            "FakeHost::map_range partial overlap at {gpa:#x}+{len:#x}"
        );
        if let Some(i) = self.provision_range(gpa, len) {
            let r = &self.ranges[i];
            unsafe { std::ptr::write_bytes(r.ptr as *mut u8, fill, len) };
        } else {
            // Non-macOS fallback: sparse bytes.
            for i in 0..len {
                self.pages.insert(gpa.wrapping_add(i as u64), fill);
            }
        }
    }

    /// Write a LE u32 at GPA.
    pub fn put_u32(&mut self, gpa: u64, v: u32) {
        let b = v.to_le_bytes();
        let _ = self.write_gpa(gpa, &b);
    }

    /// Read a LE u32 at GPA (zero if unmapped).
    pub fn get_u32(&self, gpa: u64) -> u32 {
        let mut b = [0u8; 4];
        let _ = self.read_gpa(gpa, &mut b);
        u32::from_le_bytes(b)
    }

    /// Count actions of a given kind.
    pub fn action_count(&self, kind: HostActionKind) -> usize {
        self.actions.iter().filter(|a| a.kind == kind).count()
    }

    /// Set a synthetic X-register value (mapper capture tests).
    pub fn set_xreg(&mut self, index: u32, value: u64) {
        self.xregs.insert(index, value);
    }
}

#[cfg(test)]
impl FakeHost {
    /// Contiguous heap copy of a scattered page list, written back on unmap.
    ///
    /// The answer wherever an *aliasing* view cannot be built: a page list
    /// spanning several provisioned ranges, or one whose guest page is smaller
    /// than the host's, which `mach_vm_remap` cannot place independently. It
    /// costs a copy and it is exact, where a remap of those shapes silently
    /// returns a view of the wrong bytes.
    fn bounce_view(&mut self, gpas: &[u64], page_size: usize) -> Option<usize> {
        let total = gpas.len().checked_mul(page_size)?;
        self.bounce_view_with_len(gpas, page_size, total)
    }

    fn bounce_view_with_len(
        &mut self,
        gpas: &[u64],
        page_size: usize,
        total_len: usize,
    ) -> Option<usize> {
        let guest_len = gpas.len().checked_mul(page_size)?;
        if total_len < guest_len {
            return None;
        }
        let mut buf = vec![0u8; total_len].into_boxed_slice();
        for (i, &gpa) in gpas.iter().enumerate() {
            let off = i * page_size;
            let _ = self.read_gpa(gpa, &mut buf[off..off + page_size]);
        }
        let ptr = Box::into_raw(buf) as *mut u8 as usize;
        self.bounce.push(BounceView {
            ptr,
            len: total_len,
            gpas: gpas.to_vec(),
            page_sz: page_size,
        });
        self.views.push((ptr, total_len));
        Some(ptr)
    }

    /// Write a bounce view back to guest memory and free it. `false` if this
    /// pointer is not one.
    fn release_bounce_view(&mut self, ptr: usize, len: usize) -> bool {
        let Some(pos) = self
            .bounce
            .iter()
            .position(|b| b.ptr == ptr && b.len == len)
        else {
            return false;
        };
        let b = self.bounce.remove(pos);
        self.views.retain(|&(p, l)| !(p == ptr && l == len));
        // SAFETY: the bounce buffer is exclusively owned for this view's
        // lifetime, and `len` is the length it was allocated with.
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
        for (i, &gpa) in b.gpas.iter().enumerate() {
            let off = i * b.page_sz;
            let _ = self.write_gpa(gpa, &slice[off..off + b.page_sz]);
        }
        // SAFETY: allocated by `bounce_view` as a boxed slice of this length.
        unsafe {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr as *mut u8, len));
        }
        true
    }

    /// If `addr` is in a live bounce view, return `(bounce_base, offset, max_contig)`.
    fn bounce_slot(&self, addr: u64) -> Option<(usize, usize, usize)> {
        for b in &self.bounce {
            for (i, &pg) in b.gpas.iter().enumerate() {
                if addr >= pg && addr < pg + b.page_sz as u64 {
                    let within = (addr - pg) as usize;
                    let off = i * b.page_sz + within;
                    return Some((b.ptr, off, b.page_sz - within));
                }
            }
        }
        None
    }
}

#[cfg(test)]
impl HostMemory for FakeHost {
    fn read_gpa(&self, gpa: u64, buf: &mut [u8]) -> Result<(), MemError> {
        if buf.is_empty() {
            return Ok(());
        }
        self.fire_rewires(gpa, buf.len());
        let mut done = 0usize;
        while done < buf.len() {
            let addr = gpa.checked_add(done as u64).ok_or(MemError::Overflow)?;
            // A span this fixture calls non-RAM refuses the read, because that
            // is what the product host does: the QEMU shim reads with
            // `MemTxAttrs.memory` set, so an address-space read of device memory
            // — this device's own BAR, most of all — fails closed by design.
            // Answering bytes for one would let a test pass through a door the
            // product keeps shut. `QemuReadGpaCallbackFailed` is the error the
            // shim reports for it.
            if !self.is_ram_gpa(addr) {
                return Err(MemError::QemuReadGpaCallbackFailed(-1));
            }
            // Bounce views alias guest pages until unmap.
            if let Some((bptr, off, max)) = self.bounce_slot(addr) {
                let n = (buf.len() - done).min(max);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (bptr + off) as *const u8,
                        buf[done..].as_mut_ptr(),
                        n,
                    );
                }
                done += n;
                continue;
            }
            if let Some(i) = self.range_containing(addr) {
                let r = &self.ranges[i];
                let off = (addr - r.gpa) as usize;
                let n = (buf.len() - done).min(r.len - off);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (r.ptr + off) as *const u8,
                        buf[done..].as_mut_ptr(),
                        n,
                    );
                }
                done += n;
            } else {
                buf[done] = self.pages.get(&addr).copied().unwrap_or(0);
                done += 1;
            }
        }
        Ok(())
    }

    fn write_gpa(&mut self, gpa: u64, buf: &[u8]) -> Result<(), MemError> {
        if buf.is_empty() {
            return Ok(());
        }
        let mut done = 0usize;
        while done < buf.len() {
            let addr = gpa.checked_add(done as u64).ok_or(MemError::Overflow)?;
            if let Some((bptr, off, max)) = self.bounce_slot(addr) {
                let n = (buf.len() - done).min(max);
                unsafe {
                    std::ptr::copy_nonoverlapping(buf[done..].as_ptr(), (bptr + off) as *mut u8, n);
                }
                done += n;
                continue;
            }
            if let Some(i) = self.range_containing(addr) {
                let r = &self.ranges[i];
                let off = (addr - r.gpa) as usize;
                let n = (buf.len() - done).min(r.len - off);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        buf[done..].as_ptr(),
                        (r.ptr + off) as *mut u8,
                        n,
                    );
                }
                done += n;
            } else {
                self.pages.insert(addr, buf[done]);
                done += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
impl GuestRamProvider for FakeHost {
    /// The real ranges this fixture has mapped, as RAMBlocks.
    ///
    /// The default trait impl answers `CallbackMissing`, which puts the guest-RAM
    /// map in a standing refusal. That is the right default for a `NullHost`
    /// and it was the wrong one here: `FakeHost` models a host that *does* have
    /// guest RAM, and every test of the whole-RAMBlock rail had to latch the
    /// import limits by hand and then run against a map that had refused.
    ///
    /// Answering from `ranges` keeps the fixture honest in both directions: a
    /// test that maps nothing still gets a refusing map, and one that maps guest
    /// RAM gets a host that can import it.
    fn guest_ram_regions(
        &mut self,
    ) -> Result<Vec<crate::runtime::guest_ram::GuestRamRegion>, GuestRamRegionsError> {
        Ok(self
            .ranges
            .iter()
            .map(|r| crate::runtime::guest_ram::GuestRamRegion {
                gpa_base: r.gpa,
                host_va: r.ptr as u64,
                len: r.len as u64,
            })
            .collect())
    }
}

#[cfg(test)]
impl HostControl for FakeHost {
    fn mono_ns(&self) -> u64 {
        self.mono_ns
    }

    fn enqueue(&mut self, action: HostAction) {
        // apple-gfx new_frame_handler_bh: drop/coalesce when guest ahead of
        // encode (pending_frames). Keep only the latest ScanoutUpdate pending.
        if action.kind == HostActionKind::ScanoutUpdate {
            self.actions
                .retain(|a| a.kind != HostActionKind::ScanoutUpdate);
        }
        self.actions.push(action);
    }

    fn schedule_bh(&mut self) {
        self.bh_scheduled = true;
    }
}

#[cfg(test)]
impl GuestCpuAccess for FakeHost {
    fn read_kva(&self, kva: u64, buf: &mut [u8]) -> Result<(), MemError> {
        // Tests map "KVA" into the same sparse store as GPA.
        self.read_gpa(kva, buf)
    }

    fn read_xreg(&self, index: u32) -> Result<u64, MemError> {
        self.xregs
            .get(&index)
            .copied()
            .ok_or(MemError::XregUnavailable)
    }
}

#[cfg(test)]
impl HostPageViews for FakeHost {
    /// Everything is RAM unless a test said otherwise through
    /// [`FakeHost::mark_non_ram`].
    fn is_ram_gpa(&self, gpa: u64) -> bool {
        !self
            .non_ram
            .borrow()
            .iter()
            .any(|&(start, end)| gpa >= start && gpa < end)
    }

    fn map_pages_stable(&self) -> bool {
        self.stable_map_pages
    }

    fn map_pages_with_padding(
        &mut self,
        gpas: &[u64],
        page_size: usize,
        total_len: usize,
    ) -> Option<usize> {
        if !self.stable_map_pages
            || page_size == 0
            || !page_size.is_power_of_two()
            || !total_len.is_multiple_of(page_size)
        {
            return None;
        }
        self.map_pages_calls = self.map_pages_calls.saturating_add(1);
        self.bounce_view_with_len(gpas, page_size, total_len)
    }

    /// Contiguous host view; `page_size` is the guest page size from the device.
    fn map_pages(&mut self, gpas: &[u64], page_size: usize) -> Option<usize> {
        self.map_pages_calls = self.map_pages_calls.saturating_add(1);
        if gpas.is_empty() || page_size == 0 || !page_size.is_power_of_two() {
            return None;
        }
        if gpas.iter().any(|g| *g % page_size as u64 != 0) {
            return None;
        }
        if self.strict_linux_map {
            // Model a host limited to a packed sequential alias inside one
            // already-provisioned RAM range. No range provisioning and no
            // remap/bounce packing of fragmented lists.
            if gpas.iter().any(|&gpa| self.range_containing(gpa).is_none()) {
                return None;
            }
            let i = self.range_containing(gpas[0])?;
            let r = &self.ranges[i];
            let base_off = (gpas[0] - r.gpa) as usize;
            let need = gpas.len() * page_size;
            let packed = gpas
                .iter()
                .enumerate()
                .all(|(n, &gpa)| gpa == gpas[0] + (n * page_size) as u64);
            if base_off + need > r.len || !packed {
                return None;
            }
            return Some(r.ptr + base_off);
        }
        #[cfg(target_os = "macos")]
        {
            if self.stable_map_pages {
                for &gpa in gpas {
                    if self.range_containing(gpa).is_none() {
                        let _ = self.provision_range(gpa, page_size)?;
                    }
                }
                if let Some(i) = self.range_containing(gpas[0]) {
                    let r = &self.ranges[i];
                    let base_off = (gpas[0] - r.gpa) as usize;
                    let need = gpas.len() * page_size;
                    if base_off + need <= r.len {
                        let ok = gpas
                            .iter()
                            .enumerate()
                            .all(|(n, &gpa)| gpa == gpas[0] + (n * page_size) as u64);
                        if ok {
                            return Some(r.ptr + base_off);
                        }
                    }
                }
                return None;
            }
            for &gpa in gpas {
                if self.range_containing(gpa).is_none() {
                    let _ = self.provision_range(gpa, page_size)?;
                }
            }
            // A packed run inside one provisioned range is aliased directly,
            // exactly as the two arms below and above this one already do, and
            // as `reims_vgpu_pci_map_pages` does in the product.
            //
            // This has to come BEFORE the remap loop, because `mach_vm_remap`
            // works at *host* page granularity while `page_size` here is the
            // *guest* one. On Apple Silicon the host page is 16 KiB, so an
            // x86 fixture's 4 KiB guest page at offset 0x1000 is rounded down
            // to the enclosing host page and the view aliases offset 0. The
            // remap succeeds and hands back a valid pointer, so nothing
            // reports anything: a seed write lands one guest page early and
            // the test that reads it back sees zeros. The alignment guard the
            // loop used to carry tested `r.ptr + off` against `page_size`,
            // which is satisfied by exactly the case that breaks.
            //
            // The alias is deliberately NOT recorded in `self.views`: it points
            // into a fixture's own guest RAM, and `unmap_pages` would
            // `mach_vm_deallocate` it out from under the range.
            if let Some(i) = self.range_containing(gpas[0]) {
                let r = &self.ranges[i];
                let base_off = (gpas[0] - r.gpa) as usize;
                let need = gpas.len() * page_size;
                let packed = gpas
                    .iter()
                    .enumerate()
                    .all(|(n, &gpa)| gpa == gpas[0] + (n * page_size) as u64);
                if packed && base_off + need <= r.len {
                    return Some(r.ptr + base_off);
                }
            }
            // Fragmented, or spanning ranges. Each page has to be placed on its
            // own, and `mach_vm_remap` can only do that at host page
            // granularity: it rounds the source down and the size up to whole
            // host pages. So a page whose source is not host-page-aligned
            // aliases its neighbour, and a `page_size` below the host's makes
            // every destination slot overlap the next. Neither is expressible,
            // and both used to be attempted — silently, since the remap
            // succeeds either way.
            //
            // Where it is not expressible, copy instead. A bounce view is
            // correct rather than lucky, and `unmap_pages` writes it back.
            let host_page = host_page_size();
            let remappable = page_size >= host_page
                && gpas.iter().all(|&gpa| {
                    self.range_containing(gpa).is_some_and(|idx| {
                        let r = &self.ranges[idx];
                        let off = (gpa - r.gpa) as usize;
                        off + page_size <= r.alloc_len && (r.ptr + off).is_multiple_of(host_page)
                    })
                });
            if !remappable {
                return self.bounce_view(gpas, page_size);
            }
            let mut srcs = Vec::with_capacity(gpas.len());
            for &gpa in gpas {
                let idx = self.range_containing(gpa)?;
                let r = &self.ranges[idx];
                srcs.push(r.ptr + (gpa - r.gpa) as usize);
            }
            let len = gpas.len() * page_size;
            unsafe {
                let mut view = 0u64;
                if mach_vm::mach_vm_allocate(
                    mach_vm::mach_task_self_,
                    &mut view,
                    len as u64,
                    mach_vm::VM_FLAGS_ANYWHERE,
                ) != 0
                {
                    return None;
                }
                for (i, &src) in srcs.iter().enumerate() {
                    let mut dst = view + (i * page_size) as u64;
                    let (mut cur, mut max) = (0i32, 0i32);
                    if mach_vm::mach_vm_remap(
                        mach_vm::mach_task_self_,
                        &mut dst,
                        page_size as u64,
                        0,
                        mach_vm::VM_FLAGS_FIXED_OVERWRITE,
                        mach_vm::mach_task_self_,
                        src as u64,
                        0,
                        &mut cur,
                        &mut max,
                        mach_vm::VM_INHERIT_NONE,
                    ) != 0
                    {
                        mach_vm::mach_vm_deallocate(mach_vm::mach_task_self_, view, len as u64);
                        return None;
                    }
                }
                self.views.push((view as usize, len));
                Some(view as usize)
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            for &gpa in gpas {
                if self.range_containing(gpa).is_none() {
                    let _ = self.provision_range(gpa, page_size)?;
                }
            }
            // Fast path: single contiguous span in one RealRange → alias ptr.
            // Product Linux map_pages: page i at base + i*page only.
            if let Some(i) = self.range_containing(gpas[0]) {
                let r = &self.ranges[i];
                let base_off = (gpas[0] - r.gpa) as usize;
                let need = gpas.len() * page_size;
                if base_off + need <= r.len {
                    let mut ok = true;
                    for (n, &gpa) in gpas.iter().enumerate() {
                        if gpa != gpas[0] + (n * page_size) as u64 {
                            ok = false;
                            break;
                        }
                    }
                    if ok {
                        let ptr = r.ptr + base_off;
                        self.views.push((ptr, need));
                        return Some(ptr);
                    }
                }
            }
            // Scattered pages: bounce + write-back on unmap (test convenience).
            self.bounce_view(gpas, page_size)
        }
    }

    fn unmap_pages(&mut self, ptr: usize, len: usize) {
        // A bounce view is a heap copy on every platform, so it is released the
        // same way on every platform — and it must be checked first, because
        // handing one to `mach_vm_deallocate` would free memory Mach never
        // owned.
        if self.release_bounce_view(ptr, len) {
            return;
        }
        #[cfg(target_os = "macos")]
        {
            if let Some(pos) = self.views.iter().position(|&(p, l)| p == ptr && l == len) {
                self.views.remove(pos);
                unsafe {
                    mach_vm::mach_vm_deallocate(mach_vm::mach_task_self_, ptr as u64, len as u64);
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            // Aliasing view into a RealRange — drop tracking only.
            self.views.retain(|&(p, l)| !(p == ptr && l == len));
        }
    }
}

/// Helpers usable with any HostMemory.
pub fn read_u32<M: HostMemory>(mem: &M, gpa: u64) -> Result<u32, MemError> {
    let mut b = [0u8; 4];
    mem.read_gpa(gpa, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A span walk finds a hole its two endpoints cannot see.
    ///
    /// This is the whole reason [`HostPageViews::first_non_ram_page`] exists rather
    /// than two [`HostPageViews::is_ram_gpa`] calls at the caller, and it is the shape
    /// a driven boot hit: both ends of the EFI console framebuffer answered RAM
    /// and a page 375 rows in did not.
    #[test]
    fn a_ram_span_with_an_interior_hole_is_not_vouched_for_by_its_endpoints() {
        const PAGE: usize = 4096;
        let base = 0x8000_0000u64;
        let len = 64 * PAGE as u64;
        let hole = base + 37 * PAGE as u64;

        let mut host = FakeHost::new();
        host.mark_non_ram(hole, PAGE as u64);

        assert!(
            host.is_ram_gpa(base) && host.is_ram_gpa(base + len - 1),
            "the fixture must reproduce the trap: both endpoints answer RAM"
        );
        assert_eq!(
            host.first_non_ram_page(base, len, PAGE),
            Some(hole),
            "the walk must find the interior page and name it"
        );
        assert_eq!(
            host.first_non_ram_page(base, 37 * PAGE as u64, PAGE),
            None,
            "a span stopping short of the hole is entirely RAM"
        );
    }

    /// The walk covers the page holding the last byte, and an unaligned base
    /// does not shift the grid.
    ///
    /// A span ending one byte into a page still depends on that page, so an
    /// off-by-one that stopped at the previous one would vouch for bytes it
    /// never asked about.
    #[test]
    fn a_span_walk_covers_the_page_its_last_byte_falls_in() {
        const PAGE: usize = 4096;
        let base = 0x1_0000u64;
        let mut host = FakeHost::new();
        host.mark_non_ram(base + 2 * PAGE as u64, PAGE as u64);

        assert_eq!(
            host.first_non_ram_page(base, 2 * PAGE as u64 + 1, PAGE),
            Some(base + 2 * PAGE as u64),
            "one byte into the third page still needs the third page"
        );
        assert_eq!(
            host.first_non_ram_page(base, 2 * PAGE as u64, PAGE),
            None,
            "stopping at the page boundary does not reach it"
        );
        assert_eq!(
            host.first_non_ram_page(base + 8, 2 * PAGE as u64, PAGE),
            Some(base + 2 * PAGE as u64),
            "an unaligned base is floored to its own page and the span still \
             ends where its last byte lands, so the same length now reaches one \
             page further"
        );
        assert_eq!(
            host.first_non_ram_page(base, 0, PAGE),
            None,
            "an empty span asks about nothing"
        );
    }

    /// The ABI header's action table agrees with [`HostActionKind`], and still
    /// leaves 7 retired.
    ///
    /// This is the discriminant both shims switch on in `apply_action`, and it
    /// is the highest-stakes table crossing the boundary: a drift does not fail
    /// the pop, it runs the *wrong handler* for a real action — an IRQ pulse
    /// taken as a scanout, a window close taken as a cursor update. Nothing
    /// compared the two spellings until this test.
    ///
    /// 7 is asserted absent rather than skipped. It named a pre-host-window
    /// GL/dmabuf scanout action that exists on neither side now, and every
    /// discriminant above it is written out precisely so removing it did not
    /// renumber the wire. A header that quietly reuses 7 for something new would
    /// reintroduce the renumbering this enum's comment exists to prevent.
    #[test]
    fn the_abi_header_agrees_on_the_host_action_table() {
        use crate::qemu::abi::header_define as define;
        for (name, kind) in [
            ("REIMS_VGPU_HOST_ACTION_NONE", HostActionKind::None),
            (
                "REIMS_VGPU_HOST_ACTION_IRQ_GFX",
                HostActionKind::IrqGfxPulse,
            ),
            (
                "REIMS_VGPU_HOST_ACTION_IRQ_IOSFC",
                HostActionKind::IrqIosfcPulse,
            ),
            (
                "REIMS_VGPU_HOST_ACTION_SCANOUT",
                HostActionKind::ScanoutUpdate,
            ),
            (
                "REIMS_VGPU_HOST_ACTION_CURSOR",
                HostActionKind::CursorUpdate,
            ),
            ("REIMS_VGPU_HOST_ACTION_TRACE", HostActionKind::Trace),
            (
                "REIMS_VGPU_HOST_ACTION_CURSOR_GLYPH",
                HostActionKind::CursorGlyph,
            ),
            ("REIMS_VGPU_HOST_ACTION_INPUT_KEY", HostActionKind::InputKey),
            (
                "REIMS_VGPU_HOST_ACTION_INPUT_POINTER_MOVE",
                HostActionKind::InputPointerMove,
            ),
            (
                "REIMS_VGPU_HOST_ACTION_INPUT_POINTER_BUTTON",
                HostActionKind::InputPointerButton,
            ),
            (
                "REIMS_VGPU_HOST_ACTION_WINDOW_CLOSED",
                HostActionKind::WindowClosed,
            ),
        ] {
            assert_eq!(
                define(name),
                kind as u32,
                "{name} has drifted from HostActionKind::{kind:?}; the shims \
                 would run the wrong handler for this action"
            );
            assert_ne!(define(name), 7, "{name} took the retired wire value 7");
        }
    }

    /// Unified-memory contract: a map_pages view aliases guest RAM — writes
    /// via write_gpa are visible through the view pointer and vice versa,
    /// including scattered (non-adjacent GPA) pages.
    ///
    /// On Linux FakeHost, map_pages is optional (no remappable guest RAM
    /// allocator) — skip when unsupported.
    #[test]
    fn map_pages_view_aliases_guest_ram() {
        let mut h = FakeHost::new();
        let p = GUEST_PAGE_SIZE_ARM64E as u64;
        h.map_range(0x10 * p, GUEST_PAGE_SIZE_ARM64E, 0);
        h.map_range(0x99 * p, GUEST_PAGE_SIZE_ARM64E, 0);
        let Some(view) = h.map_pages(&[0x10 * p, 0x99 * p], GUEST_PAGE_SIZE_ARM64E) else {
            // FakeHost without contig remap: not the product QEMU path.
            return;
        };
        // write_gpa → view
        h.put_u32(0x99 * p + 8, 0xdead_beef);
        let via_view = unsafe { *((view + GUEST_PAGE_SIZE_ARM64E + 8) as *const u32) };
        assert_eq!(via_view, 0xdead_beef);
        // view → read_gpa
        unsafe { *((view + 4) as *mut u32) = 0x1122_3344 };
        assert_eq!(h.get_u32(0x10 * p + 4), 0x1122_3344);
        h.unmap_pages(view, 2 * GUEST_PAGE_SIZE_ARM64E);
    }

    #[test]
    fn padded_page_view_writes_back_only_the_guest_prefix() {
        const PAGE: usize = 1usize << reims_vgpu_paging::geometry::PAGE_SHIFT_X86;
        let mut h = FakeHost::new();
        h.stable_map_pages = true;
        let gpa = 0x71u64 * PAGE as u64;
        h.map_range(gpa, 2 * PAGE, 0);
        h.put_u32(gpa, 0x1122_3344);
        h.put_u32(gpa + PAGE as u64, 0x5566_7788);

        let view = h
            .map_pages_with_padding(&[gpa], PAGE, 2 * PAGE)
            .expect("one guest page plus one host-only page");
        unsafe {
            *(view as *mut u32) = 0xaabb_ccdd;
            *((view + PAGE) as *mut u32) = 0xfeed_face;
        }
        h.unmap_pages(view, 2 * PAGE);

        assert_eq!(h.get_u32(gpa), 0xaabb_ccdd);
        assert_eq!(
            h.get_u32(gpa + PAGE as u64),
            0x5566_7788,
            "host padding must not create guest-owned memory"
        );
    }

    /// A guest page smaller than the host's must still map to its own bytes.
    ///
    /// This is the case [`map_pages_view_aliases_guest_ram`] cannot reach: it
    /// uses `GUEST_PAGE_SIZE_ARM64E` for both the range and the page size, so
    /// every offset it produces is host-page-aligned by construction. An x86
    /// fixture uses 4 KiB guest pages, and on Apple Silicon the host page is
    /// 16 KiB — so page 1 of such a range sits at offset 0x1000, which
    /// `mach_vm_remap` rounds down to the enclosing host page. The remap
    /// succeeded and returned a valid pointer aliasing page 0, so a seed write
    /// through the view landed a page early and reported success. One drain
    /// test had been red on this host since the day it was written because of
    /// it, and its two substantive assertions never ran.
    #[test]
    fn a_guest_page_smaller_than_the_hosts_maps_to_its_own_bytes() {
        const X86_PAGE: usize = 1usize << reims_vgpu_paging::geometry::PAGE_SHIFT_X86;
        let mut h = FakeHost::new();
        let base = 0x40u64 * X86_PAGE as u64;
        h.map_range(base, 3 * X86_PAGE, 0);

        // A distinct marker in each of the three pages, written through the
        // GPA store rather than the view.
        for i in 0..3u64 {
            h.put_u32(base + i * X86_PAGE as u64, 0x1000_0000 + i as u32);
        }

        let gpas: Vec<u64> = (0..3).map(|i| base + i * X86_PAGE as u64).collect();
        let view = h
            .map_pages(&gpas, X86_PAGE)
            .expect("a packed run inside one provisioned range must map");
        for i in 0..3usize {
            let got = unsafe { *((view + i * X86_PAGE) as *const u32) };
            assert_eq!(
                got,
                0x1000_0000 + i as u32,
                "view page {i} aliases the wrong guest page"
            );
        }

        // And the other direction: a write through page 1 of the view must not
        // land in page 0, which is exactly what the rounded remap did.
        unsafe { *((view + X86_PAGE) as *mut u32) = 0xfeed_face };
        assert_eq!(h.get_u32(base + X86_PAGE as u64), 0xfeed_face);
        assert_eq!(h.get_u32(base), 0x1000_0000, "page 0 must be untouched");
        h.unmap_pages(view, 3 * X86_PAGE);
        // The alias is not a view the harness owns; unmapping it must leave the
        // range's own memory alive.
        assert_eq!(h.get_u32(base + X86_PAGE as u64), 0xfeed_face);
    }

    #[test]
    fn fake_host_roundtrip() {
        let mut h = FakeHost::new();
        h.map_range(0x1000, 16, 0);
        h.put_u32(0x1000, 0x1122_3344);
        assert_eq!(h.get_u32(0x1000), 0x1122_3344);
        h.enqueue(HostAction::irq_gfx());
        assert_eq!(h.action_count(HostActionKind::IrqGfxPulse), 1);
    }

    /// apple-gfx pending_frames coalesce: multiple presentFrame signals before
    /// encode keep only the latest ScanoutUpdate (encode current +0x188 once).
    #[test]
    fn scanout_update_enqueue_coalesces_to_latest() {
        let mut h = FakeHost::new();
        h.enqueue(HostAction::irq_gfx());
        h.enqueue(HostAction::scanout_gen(3, 1440, 1080, 10));
        h.enqueue(HostAction::cursor(1, 2, true));
        h.enqueue(HostAction::scanout_gen(4, 1440, 1080, 11));
        assert_eq!(h.action_count(HostActionKind::IrqGfxPulse), 1);
        assert_eq!(h.action_count(HostActionKind::CursorUpdate), 1);
        assert_eq!(h.action_count(HostActionKind::ScanoutUpdate), 1);
        let scan = h
            .actions
            .iter()
            .find(|a| a.kind == HostActionKind::ScanoutUpdate)
            .expect("one ScanoutUpdate");
        assert_eq!(scan.a0, 4, "latest present mid wins");
        assert_eq!(scan.a3, 11);
    }

    /// Exactly one refusal means "the guest unmapped this".
    ///
    /// `is_guest_teardown` decides whether a deferred writeback that could not
    /// land is discharged quietly or reported as lost guest work, so widening it
    /// silences real losses and is the exception-list anti-pattern in miniature.
    /// Asserted exhaustively over every walk status and every `MemError` rather
    /// than by naming the one, so a variant added later has to be classified
    /// here on purpose instead of falling into whichever side `matches!`
    /// happens to put it.
    #[test]
    fn only_a_zero_pfn_means_the_guest_tore_the_range_down() {
        use reims_vgpu_paging::resolve::ResolveStatus as R;
        const WALK: &[R] = &[
            R::Ok,
            R::ErrArgs,
            R::ErrInactiveTask,
            R::ErrNoDirectory,
            R::ErrDirectoryRead,
            R::ErrZeroRootPfn,
            R::ErrZeroDepth,
            R::ErrDepthTooDeep,
            R::ErrPageTableRead,
            R::ErrZeroPfn,
            R::ErrMalformedPte,
            R::ErrUnsupportedGeometry,
        ];
        let teardown: Vec<R> = WALK
            .iter()
            .copied()
            .filter(|r| MemError::Unresolved(*r).is_guest_teardown())
            .collect();
        assert_eq!(teardown, vec![R::ErrZeroPfn]);

        for e in [
            MemError::Unmapped,
            MemError::NoCpu,
            MemError::Overflow,
            MemError::BadArgs,
            MemError::QemuReadGpaCallbackMissing,
            MemError::QemuReadGpaCallbackFailed(-1),
            MemError::QemuWriteGpaCallbackMissing,
            MemError::QemuWriteGpaCallbackFailed(-1),
            MemError::QemuReadKvaCallbackMissing,
            MemError::QemuReadKvaCallbackFailed(-1),
            MemError::XregUnavailable,
            MemError::QemuReadXregCallbackMissing,
            MemError::QemuReadXregCallbackFailed(-1),
            MemError::NoTaskDirectory,
            MemError::UnsupportedPageShift,
            MemError::TaskRootRead,
            MemError::NoSuchTask,
            MemError::NotRam,
            MemError::MapPagesRefused,
            MemError::RunOutOfRange,
        ] {
            assert!(
                !e.is_guest_teardown(),
                "{} is not the guest saying it unmapped the range",
                crate::observe::Decline::slug(&e)
            );
        }
    }
}
