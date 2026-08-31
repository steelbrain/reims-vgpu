//! Import a bounded host mapping as `VkDeviceMemory` and bind a `VkBuffer` over
//! all of it.
//!
//! This is the one place guest memory becomes something the engine can bind.
//! Which *bytes* a draw reaches is decided before it gets here and is carried by
//! a [`GuestRef`], whose bound cannot be skipped — see
//! [`crate::runtime::guest_ram`] for why that is a type and not a review rule,
//! and [`super::super::caps::host_pointer`] for the capability that gates the
//! whole rail.
//!
//! # One import per allocation identity
//!
//! RAMBlocks are imported once for the device's life. A scattered task buffer
//! may also arrive as a stable packed host alias, created once per live guest
//! buffer reference; its many draw offsets are still only bounds checks.
//!
//! [`HostRamImports`] keys both forms by
//! [`crate::runtime::guest_ram::ImportId`], so one allocation identity is never
//! imported twice. The driver is allowed to refuse a packed alias; that answer
//! is remembered and its caller gathers instead. The census separates RAMBlock
//! entries from aliases so resource-shaped growth is visible.
//!
//! # What the import does not promise
//!
//! Freeing the memory ends the GPU's access, but nothing in the extension's
//! specification says the pages were pinned while it lived. amdgpu and the
//! NVIDIA driver call `get_user_pages` at import time in practice; that is an
//! observation about two drivers rather than a contract. The honest statement is
//! in [`crate::runtime::guest_ram`]'s module doc and is not repeated as a
//! guarantee here.

use std::collections::HashMap;

use ash::vk;

use crate::backend::vulkan::caps::host_pointer::ImportTypeRefusal;
use crate::observe::Decline;
use crate::runtime::guest_ram::{GuestRamError, GuestRamImport, GuestRef};

/// One host allocation living on the GPU as a bindable buffer, with no copy
/// between it and the guest's own view of those bytes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ImportedHostRam {
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    /// Memory type selected when the parent allocation was imported. Child
    /// images may alias this memory only when their own requirements name the
    /// same type.
    pub memory_type_index: u32,
    /// Bytes the import covers. The buffer spans all of it, so every
    /// [`crate::runtime::guest_ram::BoundRange`] inside the import is a valid
    /// offset into this one buffer.
    pub size: vk::DeviceSize,
    /// Constructed HostOps alias backing this import. RAMBlock imports have no
    /// release; packed aliases are returned only after Vulkan destruction.
    pub alias: Option<(usize, usize)>,
}

impl ImportedHostRam {
    /// Release both halves. Freeing the memory is what ends the GPU's access to
    /// guest RAM, so it must run even on a teardown path that is otherwise
    /// giving up.
    ///
    /// # Safety
    ///
    /// No submission may still reference `buffer`, and `device` must be the one
    /// the import was made against.
    pub(crate) unsafe fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
        if let Some(alias) = self.alias {
            super::release_host_alias(alias);
        }
    }
}

/// A check that stopped guest RAM from becoming a bindable buffer.
///
/// Every variant is a distinct check with its own slug. An import that fails at
/// `vkAllocateMemory` and one the device declined a memory type for are two
/// different findings — the first is usually the driver refusing the pointer's
/// backing, the second is a memory-type intersection that came out empty — and a
/// shared reason would leave a reader unable to tell them apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostRamDecline {
    /// This device cannot import a host pointer at all. Carries the rung so the
    /// log says which check refused; expected on every host without the
    /// extension and on any host where an operator turned the rail off.
    Unsupported {
        rung: crate::backend::vulkan::caps::HostPointerImport,
    },
    /// The guest ended this parent allocation's lifetime. Old child objects
    /// may finish retiring, but no new view may resurrect its import identity.
    Retired { import_id: u64 },
    /// No memory type could be named for the pointer. Carries which of the two
    /// checks refused — see [`ImportTypeRefusal`], whose doc says why they are
    /// not one finding.
    NoImportableMemoryType {
        host_base: usize,
        refusal: ImportTypeRefusal,
    },
    /// A memory type was named for the pointer and the *buffer* over the same
    /// span excludes it.
    ///
    /// A separate variant rather than a second use of the one above, which is
    /// what it was. The two are asked of different objects — the first of the
    /// host allocation, this one of `vkGetBufferMemoryRequirements` — and they
    /// have different repairs: the first is a request this device chose, and
    /// this one is two driver answers that do not intersect, which no policy
    /// here can widen. Sharing a slug meant a log line could not say which, and
    /// `bugs/bug-06` is a hundred of exactly that line.
    BufferExcludesMemoryType {
        host_base: usize,
        picked: u32,
        buffer_types: u32,
    },
    /// `vkCreateBuffer` over the whole span failed.
    CreateBuffer { result: vk::Result },
    /// The buffer the driver made needs more bytes than the span has. The import
    /// is sized to the RAMBlock exactly and may not be rounded up: the bytes past
    /// the end are this process's own memory.
    TooSmall { required: u64, available: u64 },
    /// `vkAllocateMemory` with the chained import failed. On most drivers this
    /// is the pointer being refused — not fd-backed, not aligned, or not a
    /// mapping the driver can take a reference on.
    AllocateMemory { result: vk::Result },
    /// `vkBindBufferMemory` failed after a successful import.
    BindBuffer { result: vk::Result },
    /// The reference did not survive its own bound. Carries the check that
    /// refused, from [`crate::runtime::guest_ram`].
    Bound { inner: GuestRamError },
}

impl Decline for HostRamDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::Unsupported { .. } => "host_ram_import_unsupported",
            Self::Retired { .. } => "host_ram_import_retired",
            Self::NoImportableMemoryType { .. } => "host_ram_import_no_importable_memory_type",
            Self::BufferExcludesMemoryType { .. } => "host_ram_import_buffer_excludes_memory_type",
            Self::CreateBuffer { .. } => "host_ram_import_create_buffer",
            Self::TooSmall { .. } => "host_ram_import_too_small",
            Self::AllocateMemory { .. } => "host_ram_import_allocate_memory",
            Self::BindBuffer { .. } => "host_ram_import_bind_buffer",
            // The inner check is the diagnosis. Forwarding rather than adding a
            // slug keeps one name per check across the two modules.
            Self::Bound { inner } => inner.slug(),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Unsupported { rung } => vec![("rung", rung.slug().to_string())],
            Self::Retired { import_id } => vec![("import_id", import_id.to_string())],
            Self::NoImportableMemoryType { host_base, refusal } => {
                let mut fields = vec![("host_base", format!("{host_base:#x}"))];
                match refusal {
                    ImportTypeRefusal::PointerDeclined { result } => {
                        fields.push(("check", "pointer_declined".to_string()));
                        fields.push(("result", format!("{result:?}")));
                    }
                    ImportTypeRefusal::NoTypeMeetsRequest {
                        pointer_types,
                        refusal,
                    } => {
                        // The selector's own check, not just "no type": a guest
                        // this host has nowhere to put and a host that offers no
                        // importable memory at all are different reports.
                        fields.push(("check", refusal.slug().to_string()));
                        fields.push(("detail", refusal.to_string()));
                        fields.push(("pointer_types", format!("{pointer_types:#x}")));
                    }
                }
                fields
            }
            Self::BufferExcludesMemoryType {
                host_base,
                picked,
                buffer_types,
            } => vec![
                ("host_base", format!("{host_base:#x}")),
                ("picked", picked.to_string()),
                ("buffer_types", format!("{buffer_types:#x}")),
            ],
            Self::CreateBuffer { result }
            | Self::AllocateMemory { result }
            | Self::BindBuffer { result } => vec![("result", format!("{result:?}"))],
            Self::TooSmall {
                required,
                available,
            } => vec![
                ("required", required.to_string()),
                ("available", available.to_string()),
            ],
            Self::Bound { inner } => inner.fields(),
        }
    }
}

crate::observe::decline_display!(HostRamDecline);

/// A guest-memory range the engine can bind right now.
///
/// The buffer spans the whole RAMBlock, so `offset` and `len` are the only
/// things that differ between two references into the same block. Both came out
/// of [`GuestRamImport::resolve`], which is the only producer of either.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct BoundGuestRam {
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
    pub len: vk::DeviceSize,
    /// Bytes from `offset` to the first byte the caller asked for, added by
    /// widening the range to the device's import granularity. A caller that
    /// binds `offset` and reads from byte zero of it reads this many bytes early.
    pub head: vk::DeviceSize,
}

/// Every bounded host allocation this device has imported, keyed by identity.
#[derive(Clone, Copy)]
struct LiveImport {
    allocation: ImportedHostRam,
    child_images: usize,
    retired: bool,
    retirement_fences_cleared: bool,
}

pub(crate) enum ParentRetire {
    NotImported,
    WaitingForChildren,
    Ready(ImportedHostRam),
}

#[derive(Default)]
pub(crate) struct HostRamImports {
    live: HashMap<u64, LiveImport>,
    /// `true` for RAMBlock-coordinate imports, `false` for packed task aliases.
    kinds: HashMap<u64, bool>,
    /// Driver refusals are properties of this pointer/device pair. Holding the
    /// answer prevents every draw from repeating a failed allocation.
    declined: HashMap<u64, HostRamDecline>,
}

impl HostRamImports {
    pub(crate) fn alias(
        &self,
        import_id: crate::runtime::guest_ram::ImportId,
    ) -> Option<(usize, usize)> {
        self.live
            .get(&import_id.get())
            .and_then(|live| live.allocation.alias)
    }
    fn remove_live(&mut self, key: u64) -> Option<ImportedHostRam> {
        self.kinds.remove(&key);
        self.live.remove(&key).map(|entry| entry.allocation)
    }

    /// Resolve `guest_ref` to a bindable range, importing its RAMBlock if this
    /// is the first reference into it.
    ///
    /// # Safety
    ///
    /// `ctx` must be the live device context, and the import's host base must
    /// still be a mapping in this process — which it is for the VM's lifetime,
    /// because it is QEMU's own RAMBlock mapping.
    pub(crate) unsafe fn bind(
        &mut self,
        ctx: &super::context::DeviceContext,
        guest_ref: &GuestRef,
    ) -> Result<BoundGuestRam, HostRamDecline> {
        // The bound first, so a reference that cannot name its own bytes never
        // reaches an import call.
        let range = guest_ref
            .bound()
            .map_err(|inner| HostRamDecline::Bound { inner })?;
        let (live, _) = unsafe { self.ensure(ctx, guest_ref.import()) }?;
        Ok(BoundGuestRam {
            buffer: live.buffer,
            offset: range.offset,
            len: range.len,
            head: guest_ref.head(),
        })
    }

    /// Import `import`'s RAMBlock now, if it is not imported already.
    ///
    /// [`Self::bind`] does this on whatever draw happens to reference the block
    /// first, and on a discrete host that draw pays seconds for it — see
    /// [`import_ramblock`]'s own note. This is the same work with no reference
    /// to name, so a caller that knows the RAMBlocks before the guest asks for
    /// any of their bytes can pay it where nothing is waiting on a frame.
    ///
    /// Returns whether an import was made, so the caller can tell a warm that
    /// did work from one that found the answer already there.
    ///
    /// # Safety
    ///
    /// As [`Self::bind`].
    pub(crate) unsafe fn warm(
        &mut self,
        ctx: &super::context::DeviceContext,
        import: &GuestRamImport,
    ) -> Result<bool, HostRamDecline> {
        unsafe { self.ensure(ctx, import) }.map(|(_, made)| made)
    }

    /// Return the one device-memory import owned by `import`'s allocation.
    ///
    /// Child buffers and images are views into this parent allocation; they do
    /// not import the host pointer again. The caller still has to validate the
    /// child's memory requirements and bind offset before using the handle.
    pub(crate) unsafe fn allocation(
        &mut self,
        ctx: &super::context::DeviceContext,
        import: &GuestRamImport,
    ) -> Result<ImportedHostRam, HostRamDecline> {
        unsafe { self.ensure(ctx, import) }.map(|(live, _)| live)
    }

    /// Record one child image bound into this parent allocation.
    pub(crate) fn retain_child(&mut self, import: &GuestRamImport) {
        let entry = self
            .live
            .get_mut(&import.id().get())
            .expect("a child can only retain the parent it just bound");
        entry.child_images = entry
            .child_images
            .checked_add(1)
            .expect("child image reference count overflow");
    }

    /// Release one retired child. Returns the parent allocation exactly when
    /// its guest lifetime has ended and this was its last child.
    pub(crate) fn release_child(&mut self, import: &GuestRamImport) -> Option<ImportedHostRam> {
        let key = import.id().get();
        let entry = self.live.get_mut(&key)?;
        entry.child_images = entry
            .child_images
            .checked_sub(1)
            .expect("every child release has a matching retain");
        if entry.retired && entry.child_images == 0 && entry.retirement_fences_cleared {
            return self.remove_live(key);
        }
        None
    }

    /// The submission slots that were open when the parent retired have all
    /// completed. Returns the allocation when child retirement completed too.
    pub(crate) fn retirement_fences_cleared(
        &mut self,
        import_id: crate::runtime::guest_ram::ImportId,
    ) -> Option<ImportedHostRam> {
        let key = import_id.get();
        let entry = self.live.get_mut(&key)?;
        entry.retirement_fences_cleared = true;
        if entry.retired && entry.child_images == 0 {
            return self.remove_live(key);
        }
        None
    }

    /// End a guest parent allocation's lifetime. Existing children keep its
    /// Vulkan import alive until their in-flight-safe retirement completes.
    pub(crate) fn retire(
        &mut self,
        import_id: crate::runtime::guest_ram::ImportId,
    ) -> ParentRetire {
        let key = import_id.get();
        let Some(entry) = self.live.get_mut(&key) else {
            return ParentRetire::NotImported;
        };
        entry.retired = true;
        if entry.child_images == 0 {
            return ParentRetire::Ready(self.remove_live(key).expect("the entry was found above"));
        }
        ParentRetire::WaitingForChildren
    }

    /// The one place a RAMBlock becomes an import, so "have we imported this
    /// block" is asked once and cannot be answered two ways.
    ///
    /// # Safety
    ///
    /// As [`Self::bind`].
    unsafe fn ensure(
        &mut self,
        ctx: &super::context::DeviceContext,
        import: &GuestRamImport,
    ) -> Result<(ImportedHostRam, bool), HostRamDecline> {
        let key = import.id().get();
        if import.is_retired() {
            return Err(HostRamDecline::Retired { import_id: key });
        }
        if let Some(live) = self.live.get(&key) {
            return Ok((live.allocation, false));
        }
        if let Some(decline) = self.declined.get(&key) {
            return Err(*decline);
        }
        let made = match unsafe { import_ramblock(ctx, import) } {
            Ok(made) => made,
            Err(decline) => {
                self.declined.insert(key, decline);
                return Err(decline);
            }
        };
        self.live.insert(
            key,
            LiveImport {
                allocation: made,
                child_images: 0,
                retired: false,
                retirement_fences_cleared: false,
            },
        );
        self.kinds.insert(key, import.gpa_base().is_some());
        Ok((made, true))
    }

    /// Release every import. Called on device teardown, before the device goes.
    ///
    /// # Safety
    ///
    /// No submission may still reference any imported buffer.
    pub(crate) unsafe fn destroy_all(&mut self, device: &ash::Device) {
        for (_, live) in self.live.drain() {
            unsafe { live.allocation.destroy(device) };
        }
        self.kinds.clear();
        self.declined.clear();
    }

    /// Bytes of guest RAM this device currently has imported, for the census.
    pub(crate) fn imported_bytes(&self) -> u64 {
        self.live.values().map(|l| l.allocation.size).sum()
    }

    /// How many RAMBlocks are imported. One or two on an ordinary machine, and
    /// the number that must not grow with the workload — a rising count here is
    /// the per-resource import the model exists to avoid.
    pub(crate) fn counts(&self) -> (usize, usize) {
        self.live.keys().fold((0, 0), |(ramblocks, aliases), key| {
            // Every live entry is created from the import passed to `ensure`;
            // retain its kind beside the handle so the census does not call a
            // packed task allocation a RAMBlock.
            if self.kinds.get(key).copied().unwrap_or(false) {
                (ramblocks + 1, aliases)
            } else {
                (ramblocks, aliases + 1)
            }
        })
    }
}

/// Import one RAMBlock's whole host mapping.
///
/// # Cost, and why it is timed
///
/// This is the only expensive step on the whole guest-memory rail, and it is
/// paid once per RAMBlock per device. Which of its two halves costs what is not
/// a thing a reader can assume: `vkGetMemoryHostPointerPropertiesEXT` asks the
/// driver about a pointer and `vkAllocateMemory` is where a driver that pins
/// takes its `get_user_pages` over every page of a multi-gigabyte mapping. The
/// two are timed separately and emitted once per import, because the first draw
/// of a boot pays whichever of them is slow, and a display transaction the
/// guest abandons after 1000 ms is what that draw sits inside.
///
/// # Safety
///
/// As [`HostRamImports::bind`].
unsafe fn import_ramblock(
    ctx: &super::context::DeviceContext,
    import: &GuestRamImport,
) -> Result<ImportedHostRam, HostRamDecline> {
    use crate::backend::vulkan::caps::host_pointer::GUEST_IMPORT_USAGE;
    use std::time::Instant;

    let Some(loader) = ctx.external_memory_host.as_ref() else {
        return Err(HostRamDecline::Unsupported {
            rung: ctx.caps.host_pointer.rung,
        });
    };
    const HANDLE_TYPE: vk::ExternalMemoryHandleTypeFlags =
        vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;

    let host_base = import.host_base();
    let size = import.len();

    // Which memory types will accept *this* pointer. Asked before anything is
    // created, because the answer is a property of the mapping rather than of
    // the device — and it goes through the one memory-type selector this
    // backend has, so the ranking is not restated here.
    //
    // `Upload` is the class: guest RAM is host memory the GPU reaches, which is
    // exactly what that preference describes. On a discrete host the selector
    // will land on a host-visible type, and the copy into VRAM is a separate
    // decision made by the caller, not by this import.
    //
    // The RAMBlock's whole length goes with the request, and for this call site
    // that is the load-bearing argument rather than a detail. An imported host
    // pointer's pages do not move — the memory type cannot relocate a mapping
    // this process already holds — so the only thing the pick decides is which
    // heap the driver charges a multi-gigabyte allocation to. `Upload` on a
    // `Unified` classification prefers `DEVICE_LOCAL`, and on a part whose
    // device-local heap is a carve-out smaller than the guest (an APU with 2 GiB
    // against a 16 GiB guest) that preference asks the driver to keep the entire
    // guest resident in a pool with no room for it.
    let req = ctx
        .caps
        .memory_request(crate::backend::vulkan::caps::MemoryClass::Upload);
    let probe_started = Instant::now();
    let picked = unsafe {
        crate::backend::vulkan::caps::host_pointer::import_memory_type(
            loader,
            &ctx.memory_properties,
            host_base as *const std::ffi::c_void,
            &req,
            size,
            ctx.caps.max_allocation_size,
        )
    };
    let probe_us = probe_started.elapsed().as_micros() as u64;
    // A refusal here is the whole of the heap and allocation-size admission for
    // this rail. It reaches the caller as a decline, the copying rails take the
    // guest's bytes instead, and no `vkAllocateMemory` the specification forbids
    // is ever issued — which is the difference between the two drivers this was
    // reported on, one of which returns success and then loses the device.
    let pick =
        picked.map_err(|refusal| HostRamDecline::NoImportableMemoryType { host_base, refusal })?;
    let memory_type_index = pick.index;
    let alloc_started = Instant::now();

    let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(HANDLE_TYPE);
    let create = vk::BufferCreateInfo::default()
        .size(size)
        .usage(GUEST_IMPORT_USAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push_next(&mut external);
    let buffer = unsafe { ctx.device.create_buffer(&create, None) }
        .map_err(|result| HostRamDecline::CreateBuffer { result })?;

    // From here every failure must destroy the buffer before returning, so the
    // work is done in a closure and the cleanup happens once at the end.
    let bound = (|| {
        let reqs = unsafe { ctx.device.get_buffer_memory_requirements(buffer) };
        if reqs.size > size {
            // Not rounded up. The bytes past the end of a RAMBlock are this
            // process's own memory, and handing the GPU write access to them is
            // the one stray the bound exists to prevent.
            return Err(HostRamDecline::TooSmall {
                required: reqs.size,
                available: size,
            });
        }
        if reqs.memory_type_bits & (1u32 << memory_type_index) == 0 {
            return Err(HostRamDecline::BufferExcludesMemoryType {
                host_base,
                picked: memory_type_index,
                buffer_types: reqs.memory_type_bits,
            });
        }

        let mut host_import = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(HANDLE_TYPE)
            .host_pointer(host_base as *mut std::ffi::c_void);
        let allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(size)
            .memory_type_index(memory_type_index)
            .push_next(&mut host_import);
        let memory = unsafe { ctx.device.allocate_memory(&allocate, None) }
            .map_err(|result| HostRamDecline::AllocateMemory { result })?;

        match unsafe { ctx.device.bind_buffer_memory(buffer, memory, 0) } {
            Ok(()) => Ok(ImportedHostRam {
                buffer,
                memory,
                memory_type_index,
                size,
                alias: import.gpa_base().is_none().then_some((
                    host_base,
                    usize::try_from(size).expect("host allocation size fits this process"),
                )),
            }),
            Err(result) => {
                // Freeing the memory is what ends the GPU's access to the
                // pointer, so it happens even on this failure path.
                unsafe { ctx.device.free_memory(memory, None) };
                Err(HostRamDecline::BindBuffer { result })
            }
        }
    })();

    if bound.is_err() {
        unsafe { ctx.device.destroy_buffer(buffer, None) };
    }
    // The heap is on this line and not just the type index, because "which type"
    // is not answerable from the index alone on an unfamiliar device and the
    // heap is what a report of a slow host turns on. There is no `fits=` field
    // any more and there cannot be one: a pick whose heap could not hold the
    // import is a refusal now, so every line here is an import the device was
    // allowed to make.
    crate::observe::off(format!(
        "host_ram_import id={} bytes={size} mtype={memory_type_index} heap={} heap_mb={} \
         probe_us={probe_us} alloc_us={} ok={}",
        import.id().get(),
        pick.heap_index,
        pick.heap_bytes >> 20,
        alloc_started.elapsed().as_micros() as u64,
        bound.is_ok(),
    ));
    bound
}

/// A check that stopped GPU-ordered access to the guest's own pages, so the
/// operation took its blocking CPU route instead.
///
/// Every one of these is a *routing* answer rather than a loss — the copying
/// rail still lands the frame — but each is a whole flush's worth of memcpy that
/// the device paid and did not have to, so they are named individually and
/// counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuestWriteDecline {
    /// The named resident is an ordinary device allocation rather than the
    /// guest allocation synchronization requires.
    NoSharedBacking,
    /// The device cannot import guest RAM at all. Carries the rung so the log
    /// says which check refused; expected on every host without the extension.
    Unsupported {
        rung: crate::backend::vulkan::caps::HostPointerImport,
    },
    /// Guest-memory work is outstanding, but no successful FIFO submission
    /// published a completion point for it. This is an ownership invariant
    /// failure rather than a missing host capability.
    NoCompletionPoint,
    /// The resident's physical channel order is not the order the destination
    /// stores, so landing it would need an R/B exchange — which an image→buffer
    /// copy cannot perform. The copying rail's per-row conversion is where that
    /// lives.
    ///
    /// Stated as a disagreement between the two rather than as "the resident is
    /// not BGRA", because both orders reach this call: a type-11 mapping's pages
    /// are guest scanout order and a GVA render target's are whatever the guest
    /// declared for it. A rail that spelled the rule as one fixed order refused
    /// every RGBA destination it could have served unchanged.
    /// Two whole formats and not two orders: an order stopped being a complete
    /// description of a resident once a render target could be wider than eight
    /// bits per channel, and this copy converts nothing, so a half-float
    /// destination over an eight-bit resident must be caught here.
    ResidentFormatMismatch {
        held: ash::vk::Format,
        want: ash::vk::Format,
    },
    /// The resident's geometry is not the geometry the window promised the
    /// guest. Copying anyway would land one extent's pixels under another's row
    /// pitch.
    GeometryMoved {
        resident_width: u32,
        resident_height: u32,
        want_width: u32,
        want_height: u32,
    },
    /// The frame's last byte falls past the end of the range the runtime asked
    /// for.
    ///
    /// Not the same check as the import bound, and that is why it is kept: the
    /// runtime sizes the request from the mapping's page plan and the engine
    /// computes the extent from the resident's own geometry and row pitch. Two
    /// independently-derived numbers, and a disagreement between them is a
    /// frame that would land under the wrong pitch.
    WindowTooSmall { need: u64, have: u64 },
    /// The import itself declined; the inner reason names the step.
    Import { inner: HostRamDecline },
}

impl Decline for GuestWriteDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::NoSharedBacking => "gpu_writeback_no_shared_backing",
            Self::Unsupported { .. } => "gpu_writeback_unsupported",
            Self::NoCompletionPoint => "gpu_completion_point_missing",
            Self::ResidentFormatMismatch { .. } => "gpu_writeback_resident_format_mismatch",
            Self::GeometryMoved { .. } => "gpu_writeback_geometry_moved",
            Self::WindowTooSmall { .. } => "gpu_writeback_window_too_small",
            // The inner decline's own slug, so a driver that refuses the pointer
            // and a range that is too short stay as distinguishable here as they
            // are at the import site.
            Self::Import { inner } => inner.slug(),
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::NoSharedBacking => Vec::new(),
            Self::Unsupported { rung } => vec![("rung", rung.slug().to_string())],
            Self::NoCompletionPoint => Vec::new(),
            Self::ResidentFormatMismatch { held, want } => vec![
                ("resident", format!("{held:?}")),
                ("want", format!("{want:?}")),
            ],
            Self::GeometryMoved {
                resident_width,
                resident_height,
                want_width,
                want_height,
            } => vec![
                ("resident", format!("{resident_width}x{resident_height}")),
                ("want", format!("{want_width}x{want_height}")),
            ],
            Self::WindowTooSmall { need, have } => {
                vec![("need", need.to_string()), ("have", have.to_string())]
            }
            Self::Import { inner } => inner.fields(),
        }
    }
}

crate::observe::decline::decline_display!(GuestWriteDecline);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::vulkan::caps::memory_topology::MemoryTypeRefusal;
    use crate::backend::vulkan::caps::HostPointerImport;

    /// One slug per check. Two sharing one would mean watching a slug fire and
    /// still not knowing whether the driver refused the pointer or the memory
    /// type intersection came out empty.
    #[test]
    fn every_decline_has_its_own_slug() {
        let all = [
            HostRamDecline::Unsupported {
                rung: HostPointerImport::Unqueried,
            },
            HostRamDecline::Retired { import_id: 1 },
            HostRamDecline::NoImportableMemoryType {
                host_base: 0,
                refusal: ImportTypeRefusal::NoTypeMeetsRequest {
                    pointer_types: 0,
                    refusal: MemoryTypeRefusal::NoTypeWithRequiredFlags { type_bits: 0 },
                },
            },
            HostRamDecline::BufferExcludesMemoryType {
                host_base: 0,
                picked: 0,
                buffer_types: 0,
            },
            HostRamDecline::CreateBuffer {
                result: vk::Result::ERROR_UNKNOWN,
            },
            HostRamDecline::TooSmall {
                required: 0,
                available: 0,
            },
            HostRamDecline::AllocateMemory {
                result: vk::Result::ERROR_UNKNOWN,
            },
            HostRamDecline::BindBuffer {
                result: vk::Result::ERROR_UNKNOWN,
            },
        ];
        let mut slugs: Vec<_> = all.iter().map(|d| d.slug()).collect();
        let count = slugs.len();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), count, "two checks share a slug");
        for slug in slugs {
            assert!(slug.starts_with("host_ram_import_"), "{slug}");
        }
    }

    /// A refusal that cannot say which check refused is a log line nobody can
    /// act on. `bugs/bug-06` is a hundred `no_importable_memory_type` lines at
    /// one `host_base` and no way to tell the driver declining the mapping from
    /// this device's own memory request excluding every type the driver named —
    /// the first is the host's, the second is ours, and only one of them has a
    /// repair here.
    ///
    /// Asserted on the emitted fields rather than on the variant, because the
    /// fields are what a reader greps.
    #[test]
    fn the_memory_type_refusal_names_the_check_that_refused() {
        let declined = HostRamDecline::NoImportableMemoryType {
            host_base: 0x7ff5f7e00000,
            refusal: ImportTypeRefusal::PointerDeclined {
                result: vk::Result::ERROR_INVALID_EXTERNAL_HANDLE,
            },
        };
        let unmet = HostRamDecline::NoImportableMemoryType {
            host_base: 0x7ff5f7e00000,
            refusal: ImportTypeRefusal::NoTypeMeetsRequest {
                pointer_types: 0b1010,
                refusal: MemoryTypeRefusal::EveryHeapTooSmall {
                    bytes: 6 << 30,
                    roomiest_heap: 2 << 30,
                },
            },
        };
        assert_eq!(declined.slug(), unmet.slug(), "one check, one slug");
        let check = |d: &HostRamDecline| {
            d.fields()
                .into_iter()
                .find(|(name, _)| *name == "check")
                .map(|(_, value)| value)
        };
        assert_eq!(check(&declined).as_deref(), Some("pointer_declined"));
        // The selector's own check, which is what separates "this host offers
        // no importable memory" from "this host has nowhere to put six
        // gigabytes" — one is a capability report and the other is a capacity
        // one, and only the second says the guest is too big for the machine.
        assert_eq!(
            check(&unmet).as_deref(),
            Some("vk_memory_every_heap_too_small")
        );
        // The mask rides along so an empty one — the driver accepting the
        // pointer for no type at all — is separable from an incompatible one
        // without a second boot.
        assert!(unmet
            .fields()
            .iter()
            .any(|(name, value)| *name == "pointer_types" && value == "0xa"));
    }

    #[test]
    fn a_retired_parent_waits_for_its_last_child_and_cannot_resurrect() {
        let import = GuestRamImport::new_host_allocation(0x1000, 0x4000, 0x1000)
            .expect("aligned synthetic import");
        let key = import.id().get();
        let allocation = ImportedHostRam {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            memory_type_index: 0,
            size: 0x4000,
            alias: Some((0x1000, 0x4000)),
        };
        let mut imports = HostRamImports::default();
        imports.live.insert(
            key,
            LiveImport {
                allocation,
                child_images: 0,
                retired: false,
                retirement_fences_cleared: false,
            },
        );
        imports.retain_child(&import);
        imports.retain_child(&import);
        import.retire();

        assert_eq!(imports.alias(import.id()), Some((0x1000, 0x4000)));

        assert!(matches!(
            imports.retire(import.id()),
            ParentRetire::WaitingForChildren
        ));
        assert!(
            imports.release_child(&import).is_none(),
            "one child remains"
        );
        assert!(
            imports.retirement_fences_cleared(import.id()).is_none(),
            "the retirement barrier passed but one child remains"
        );
        assert_eq!(
            imports.release_child(&import).map(|parent| parent.alias),
            Some(Some((0x1000, 0x4000))),
            "the last child hands the parent back exactly once"
        );
        assert!(!imports.live.contains_key(&key));
        assert!(import.is_retired());
    }

    #[test]
    fn parent_release_waits_when_the_last_child_finishes_before_the_fence_barrier() {
        let import = GuestRamImport::new_host_allocation(0x5000, 0x4000, 0x1000)
            .expect("aligned synthetic import");
        let key = import.id().get();
        let mut imports = HostRamImports::default();
        imports.live.insert(
            key,
            LiveImport {
                allocation: ImportedHostRam {
                    buffer: vk::Buffer::null(),
                    memory: vk::DeviceMemory::null(),
                    memory_type_index: 0,
                    size: 0x4000,
                    alias: Some((0x5000, 0x4000)),
                },
                child_images: 0,
                retired: false,
                retirement_fences_cleared: false,
            },
        );
        imports.retain_child(&import);
        import.retire();
        assert!(matches!(
            imports.retire(import.id()),
            ParentRetire::WaitingForChildren
        ));
        assert!(
            imports.release_child(&import).is_none(),
            "the retirement fence has not completed"
        );
        assert_eq!(
            imports
                .retirement_fences_cleared(import.id())
                .map(|parent| parent.alias),
            Some(Some((0x5000, 0x4000)))
        );
    }

    #[test]
    fn a_missing_shared_backing_has_its_own_sync_refusal() {
        assert_eq!(
            GuestWriteDecline::NoSharedBacking.slug(),
            "gpu_writeback_no_shared_backing"
        );
    }

    #[test]
    fn a_missing_fifo_completion_point_has_its_own_refusal() {
        assert_eq!(
            GuestWriteDecline::NoCompletionPoint.slug(),
            "gpu_completion_point_missing"
        );
    }

    /// A bound refusal keeps the inner check's name rather than being renamed
    /// at the boundary. The two modules are one rail and a reader greps one
    /// vocabulary.
    #[test]
    fn a_bound_refusal_forwards_the_check_that_refused() {
        let inner = GuestRamError::SliceEndPastImport {
            end: 0x1001,
            import_len: 0x1000,
        };
        let outer = HostRamDecline::Bound { inner };
        assert_eq!(outer.slug(), inner.slug());
        assert_eq!(outer.fields(), inner.fields());
    }

    /// The guest's handshake, and not the guest's first frame, is what pays for
    /// the import.
    ///
    /// This is the whole point of the warm rail, and it is asserted end to end —
    /// through [`crate::runtime::guest_ram_map::warm`], which is what the
    /// protocol-version handshake calls — because the mechanism being present
    /// proves nothing about it being wired. Left unwired the import lands on the
    /// first `gather` of the first draw, and on a discrete host that is seconds
    /// inside a display transaction the guest abandons after one.
    ///
    /// Skips when this host has no device or no import capability: there the
    /// copying rails are the only rails and there is nothing to warm.
    #[test]
    fn the_handshake_warm_imports_before_any_draw_references_a_byte() {
        // A device first, because it is device creation that publishes the
        // import granularity `guest_ram_map::warm` refuses without.
        {
            let mut guard = super::super::lock_engine();
            let super::super::EngineState {
                ref mut owner,
                ref counters,
                ..
            } = &mut *guard;
            let Ok(ctx) = owner.ensure(counters) else {
                eprintln!("skip: no Vulkan device");
                return;
            };
            if !ctx.caps.host_pointer.is_available() {
                eprintln!("skip: no host-pointer import on this device");
                return;
            }
        }

        // Stand-in for a RAMBlock: an allocation this process owns and never
        // frees. An import outlives this test — it is released at device
        // teardown — so freeing at the end of the test would leave the device
        // holding a buffer over memory that is no longer ours. Page-aligned
        // because that is what the import granularity asks of a base.
        const LEN: usize = 16 << 20;
        let layout = std::alloc::Layout::from_size_align(LEN, 4096).expect("valid layout");
        let base = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!base.is_null(), "allocation for the stand-in RAMBlock");

        struct OneBlock(u64);
        impl crate::runtime::host::HostOps for OneBlock {
            fn mono_ns(&self) -> u64 {
                0
            }
            fn enqueue(&mut self, _action: crate::runtime::host::HostAction) {}
            fn schedule_bh(&mut self) {}
            fn guest_ram_regions(
                &mut self,
            ) -> Result<
                Vec<crate::runtime::guest_ram::GuestRamRegion>,
                crate::runtime::host::GuestRamRegionsError,
            > {
                Ok(vec![crate::runtime::guest_ram::GuestRamRegion {
                    gpa_base: 0,
                    host_va: self.0,
                    len: LEN as u64,
                }])
            }
        }

        crate::runtime::guest_ram_map::reset();
        let before = super::super::guest_import_census().0;
        let mut host = OneBlock(base as u64);
        crate::runtime::guest_ram_map::warm(&mut host);
        let after = super::super::guest_import_census().0;
        assert_eq!(
            after - before,
            LEN as u64,
            "the handshake warm must import the whole block"
        );

        // And it is once. A warm that re-imported per call would be the
        // per-RAMBlock model turning into a per-call one.
        crate::runtime::guest_ram_map::warm(&mut host);
        assert_eq!(super::super::guest_import_census().0, after);
        crate::runtime::guest_ram_map::reset();
    }

    /// A fresh map holds nothing and reports nothing imported. The count is the
    /// reading that says whether the model held: it must stay at the number of
    /// RAMBlocks for a whole boot rather than tracking the workload.
    #[test]
    fn an_empty_map_reports_no_imports() {
        let imports = HostRamImports::default();
        assert_eq!(imports.counts(), (0, 0));
        assert_eq!(imports.imported_bytes(), 0);
    }
}
