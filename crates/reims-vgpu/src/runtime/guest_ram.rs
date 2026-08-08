//! The bound on every GPU reference to guest RAM, expressed as a pair of types.
//!
//! # What this replaces
//!
//! Guest pages reach the host GPU by importing the host virtual address range
//! that QEMU's RAMBlock already maps — `VK_EXT_external_memory_host` on Linux
//! and Windows, the same extension through MoltenVK on macOS, and
//! `newBufferWithBytesNoCopy` on the Metal-direct arm. One primitive, all three
//! hosts. What that mechanism does *not* carry is a bound: the pointer handed to
//! the driver is an ordinary host address, and an offset arithmetic slip past the
//! end of the RAMBlock reaches this process's own memory — device state, our own
//! structures, another RAMBlock — with the GPU's read *and* write access.
//!
//! The guest reaching its own RAM through shaders it authored is not an escape.
//! Straying off the end of the import is. So the entire residual security
//! argument reduces to one property:
//!
//! > **No GPU reference to guest memory may name a byte outside the RAMBlock it
//! > came from.**
//!
//! # Why it is a type and not a rule
//!
//! `AGENTS.md` bans source-grep scanners, and the test that used to hold the old
//! ban was one. Its replacement is rule 1 of the "Before A Broad Sweep" ladder —
//! *make the invariant unrepresentable*. [`GuestSlice`](crate::runtime::guest_ram::GuestSlice) has exactly one
//! constructor, [`GuestRamImport::slice`](crate::runtime::guest_ram::GuestRamImport::slice), which bounds-checks with checked
//! arithmetic against the import's own length. There is no second way to build
//! one, no public field that hands back a raw pointer or an absolute offset a
//! call site could re-add to something, and no `From` conversion. A new import
//! site cannot be written that skips the check, because there is nothing else to
//! write.
//!
//! The absolute position of a slice inside its import is obtainable only by
//! presenting the slice back to the import that made it
//! ([`GuestRamImport::resolve`](crate::runtime::guest_ram::GuestRamImport::resolve)), which is also where the cross-import check
//! lives. That is the same rule as the C shim boundary in `AGENTS.md`: export
//! what the backend binds, not the inputs it binds from.
//!
//! # What this type does not promise
//!
//! A udmabuf fd pinned the pages it named — they stopped being swappable or
//! migratable — and closing the fd revoked the GPU's access.
//! `VK_EXT_external_memory_host` makes no such promise in the specification.
//! amdgpu and the NVIDIA driver do `get_user_pages` at import time in practice,
//! but that is an observation about two drivers, not a contract. We trade a
//! kernel-enforced pin and a revocation handle for a primitive that exists on
//! all three hosts. Do not write anywhere that this is the same guarantee. If a
//! host is ever observed migrating a page under a live import, that is a real
//! defect with a measurement, not a gap in this doc.
//!
//! Page recycling is unchanged and still load-bearing: the guest reassigning a
//! GPA to a different allocation while we hold a reference over it is the
//! PTE-corruption class the surface page-ownership guards exist for.
//! It applied to the dma-buf and it applies here.
//!
//! # One import per RAMBlock, for the lifetime of the VM
//!
//! GPA → HVA is linear *within* a RAMBlock, so one import covers every guest
//! page in it and a resource becomes an `(offset, len)` pair rather than a page
//! list somebody has to export. That is what makes [`GuestRamImport::slice`](crate::runtime::guest_ram::GuestRamImport::slice)
//! free — no ioctl, no allocation, no cache, no kernel reference per page — and
//! it is why a scattered surface is not un-importable: it is N slices over one
//! import.
//!
//! **Do not import per resource.** The extension does not guarantee that
//! importing the same host allocation twice into one device works, and
//! re-importing per draw pays the driver's `get_user_pages` thousands of times a
//! second for an answer that never changes.

use crate::contract::checked::align_up_u64;
use crate::observe::{Decline, Emit};

/// One RAMBlock as the host shim describes it: where it starts in guest physical
/// address space, where QEMU mapped it, and how long it is.
///
/// This is the shape [`crate::runtime::HostOps::guest_ram_regions`] hands back,
/// and it is deliberately not `map_pages` with a different return type.
/// `map_pages` answers "give me a view of these specific pages" and may build a
/// transient one the caller must release; this answers "where does guest RAM
/// live, for the lifetime of the VM".
///
/// `#[repr(C)]` and three `u64`s because the shim writes these directly through
/// the caller's array: this declaration and `ReimsVgpuGuestRamRegion` in
/// `include/reims_vgpu_qemu_abi.h` are one struct, and
/// `crate::qemu::abi::tests::the_abi_header_agrees_on_the_guest_ram_region_layout`
/// is the only thing that compares them. The host address is a `u64` rather
/// than a `usize` so the layout does not depend on the target the shim happened
/// to be built for.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GuestRamRegion {
    /// First guest physical address this block backs.
    pub gpa_base: u64,
    /// Host virtual address QEMU mapped it at. Stable for the VM's lifetime.
    pub host_va: u64,
    /// Length in bytes, in both address spaces.
    pub len: u64,
}

/// Identity of one live import.
///
/// Process-monotonic and never reused, which is load-bearing rather than
/// cosmetic: a [`GuestSlice`] built against an import that has since been torn
/// down must not resolve against the *replacement* import over the same
/// RAMBlock. A device recreate makes a new id, so the stale slice refuses with
/// [`GuestRamError::SliceForeignImport`] instead of binding a range whose
/// meaning changed underneath it.
///
/// `u64` so exhaustion is not a case anyone has to handle — at one import per
/// nanosecond it lasts five centuries — which removes the wrap that would
/// otherwise be the one way two imports could share an identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ImportId(std::num::NonZeroU64);

impl ImportId {
    /// The next unused identity.
    fn allocate() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let raw = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // `fetch_add` from 1 cannot produce 0 short of 2^64 imports, which the
        // type doc explains is not a reachable state. `new` rather than
        // `new_unchecked` so the impossible case is a panic on a control path
        // rather than undefined behavior in a driver.
        Self(std::num::NonZeroU64::new(raw).expect("import identity space exhausted"))
    }

    /// The raw identity, for log fields only. Nothing may key a GPU resource on
    /// this without also holding the [`GuestRamImport`] it came from.
    pub fn get(self) -> u64 {
        self.0.get()
    }
}

impl std::fmt::Display for ImportId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.get())
    }
}

/// Every way a guest-memory reference can be refused, named per check.
///
/// One slug per distinct check, per [`Decline`]: watching a slug fire in
/// `/tmp/reims-vgpu-fail.log` has to say *which* bound refused, or the reader is
/// back to guessing between an overflow, an out-of-range end and a slice
/// presented to the wrong import.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuestRamError {
    /// The shim described a zero-length RAMBlock. Nothing to import.
    RegionEmpty,
    /// The shim described a RAMBlock at host address 0. Either the block is not
    /// mapped in this process or the shim answered with an uninitialized field;
    /// importing it would hand the driver a null pointer.
    RegionUnmapped,
    /// `host_va + len` or `gpa_base + len` leaves its address space. A RAMBlock
    /// cannot wrap, so this is a malformed answer from the shim rather than a
    /// large one.
    RegionWraps { host_va: u64, len: u64 },
    /// The backend reported an import granularity that is zero or not a power of
    /// two. Every alignment computation below assumes a power of two mask, so
    /// this is refused rather than worked around.
    AlignmentNotPowerOfTwo { align: u64 },
    /// The RAMBlock cannot be trimmed to the backend's import granularity and
    /// leave anything behind — the block is shorter than one granule, or the
    /// rounded-up base is already past its end. This is the
    /// `AlignmentUnsatisfiable` capability rung arriving as a refusal.
    AlignmentUnsatisfiable { align: u64, len: u64 },
    /// A zero-length slice. Nothing binds a zero-length range, and admitting one
    /// would make `offset == len` a legal reference to the byte past the end.
    SliceEmpty,
    /// `offset + len` overflowed `u64`. Refused on the overflow rather than on
    /// the comparison, because a wrapped end compares as small and would pass
    /// the range check that follows.
    SliceOverflow { offset: u64, len: u64 },
    /// The slice ends past the import. The bound this whole module exists for.
    SliceEndPastImport { end: u64, import_len: u64 },
    /// The slice fits, but widening it to the import granularity does not.
    ///
    /// A healthy zero: [`GuestRamImport::new`] rounds the import length *down*
    /// to a multiple of the granularity, so rounding an end that is already
    /// inside it up to the same granularity cannot leave it. A firing means that
    /// derivation broke, which is why this is its own slug and not folded into
    /// [`Self::SliceEndPastImport`].
    SliceAlignedEndPastImport { aligned_end: u64, import_len: u64 },
    /// A slice made by one import, presented to another. Refused before any
    /// arithmetic: the offset is meaningful only against the import that built
    /// it, and against a different one it is an arbitrary address.
    SliceForeignImport { slice: u64, import: u64 },
    /// A guest physical address that this RAMBlock does not back. Includes a GPA
    /// inside the untrimmed block but below the granularity-aligned base.
    GpaOutsideImport { gpa: u64, gpa_base: u64, len: u64 },
}

impl Decline for GuestRamError {
    fn slug(&self) -> &'static str {
        match self {
            Self::RegionEmpty => "guest_ram_region_empty",
            Self::RegionUnmapped => "guest_ram_region_unmapped",
            Self::RegionWraps { .. } => "guest_ram_region_wraps",
            Self::AlignmentNotPowerOfTwo { .. } => "guest_ram_alignment_not_power_of_two",
            Self::AlignmentUnsatisfiable { .. } => "guest_ram_alignment_unsatisfiable",
            Self::SliceEmpty => "guest_ram_slice_empty",
            Self::SliceOverflow { .. } => "guest_ram_slice_overflow",
            Self::SliceEndPastImport { .. } => "guest_ram_slice_end_past_import",
            Self::SliceAlignedEndPastImport { .. } => "guest_ram_slice_aligned_end_past_import",
            Self::SliceForeignImport { .. } => "guest_ram_slice_foreign_import",
            Self::GpaOutsideImport { .. } => "guest_ram_gpa_outside_import",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match *self {
            Self::RegionEmpty | Self::RegionUnmapped | Self::SliceEmpty => Vec::new(),
            Self::RegionWraps { host_va, len } => {
                vec![("host_va", format!("{host_va:#x}")), ("len", len.to_string())]
            }
            Self::AlignmentNotPowerOfTwo { align } => vec![("align", align.to_string())],
            Self::AlignmentUnsatisfiable { align, len } => {
                vec![("align", align.to_string()), ("len", len.to_string())]
            }
            Self::SliceOverflow { offset, len } => {
                vec![("offset", offset.to_string()), ("len", len.to_string())]
            }
            Self::SliceEndPastImport { end, import_len } => vec![
                ("end", end.to_string()),
                ("import_len", import_len.to_string()),
            ],
            Self::SliceAlignedEndPastImport {
                aligned_end,
                import_len,
            } => vec![
                ("aligned_end", aligned_end.to_string()),
                ("import_len", import_len.to_string()),
            ],
            Self::SliceForeignImport { slice, import } => vec![
                ("slice_import", slice.to_string()),
                ("import", import.to_string()),
            ],
            Self::GpaOutsideImport { gpa, gpa_base, len } => vec![
                ("gpa", format!("{gpa:#x}")),
                ("gpa_base", format!("{gpa_base:#x}")),
                ("len", len.to_string()),
            ],
        }
    }
}

crate::observe::decline_display!(GuestRamError);

/// The greppable event class every refusal in this module carries.
const EVENT: &str = "guest_ram";

impl GuestRamError {
    /// Send this refusal to the always-on sink.
    ///
    /// Emitted here rather than at each call site so a new consumer of
    /// [`GuestRamImport::slice`] cannot lose a bound violation by forgetting to
    /// log it — the same argument the type itself makes about the check.
    /// Deduped by slug, because a bad offset in a per-draw path would otherwise
    /// arrive thousands of times a second and drown the log it is evidence in.
    ///
    /// Call sites should add their own context on a *different* event rather
    /// than re-emitting this one.
    fn report(self) -> Self {
        Emit::decline(EVENT, &self).fail_once(0);
        self
    }
}

/// One RAMBlock, imported once, and the only thing that can name a byte in it.
///
/// Created at device init and held for the VM's lifetime. Not per draw, not per
/// window, not per resource — see the module doc for why re-importing is both
/// unsupported and expensive.
///
/// The backend's handle for the import (a `VkDeviceMemory` and its whole-region
/// `VkBuffer`, or an `MTLBuffer`) lives beside this in the backend, keyed by
/// [`Self::id`]. This type is deliberately backend-free: the bound is pure
/// arithmetic, and keeping it out of a backend-gated module is what lets its
/// tests run on the Metal arm's build as well as the Vulkan one. On a Linux host
/// nothing under `backend/metal/` runs its tests at all.
#[derive(Debug)]
pub struct GuestRamImport {
    id: ImportId,
    /// Guest physical address of the first byte *covered*, after any trim.
    gpa_base: u64,
    /// Host virtual address of the first byte covered, after any trim. The
    /// address handed to the backend's import call, and never a subrange of it.
    host_base: usize,
    /// Bytes covered, in both address spaces. Always a multiple of `align`.
    len: u64,
    /// The backend's import granularity — `minImportedHostPointerAlignment` on
    /// Vulkan, the host page size on Metal-direct.
    align: u64,
}

impl GuestRamImport {
    /// Take a RAMBlock the shim described and bound it to `align`.
    ///
    /// `align` is the backend's queried import granularity, never a guessed
    /// constant: `minImportedHostPointerAlignment` from
    /// `VkPhysicalDeviceExternalMemoryHostPropertiesEXT`, or the host page size
    /// on the Metal-direct arm. Both the base and the length must satisfy it,
    /// which is a requirement of the import call and not a preference of ours.
    ///
    /// Where the block's base does not meet it, the covered span is trimmed
    /// forward to the next granule and the length rounded down. The trim is
    /// reported by [`Self::gpa_base`] rather than hidden: a GPA below the
    /// covered base refuses with [`GuestRamError::GpaOutsideImport`] instead of
    /// resolving to the wrong bytes. In practice a RAMBlock is page-aligned and
    /// the trim is zero.
    pub fn new(region: GuestRamRegion, align: u64) -> Result<Self, GuestRamError> {
        if align == 0 || !align.is_power_of_two() {
            return Err(GuestRamError::AlignmentNotPowerOfTwo { align }.report());
        }
        if region.len == 0 {
            return Err(GuestRamError::RegionEmpty.report());
        }
        if region.host_va == 0 {
            return Err(GuestRamError::RegionUnmapped.report());
        }
        let wraps = GuestRamError::RegionWraps {
            host_va: region.host_va,
            len: region.len,
        };
        let host_end = region
            .host_va
            .checked_add(region.len)
            .ok_or(wraps)
            .map_err(GuestRamError::report)?;
        if host_end > usize::MAX as u64 {
            return Err(wraps.report());
        }
        region
            .gpa_base
            .checked_add(region.len)
            .ok_or(wraps)
            .map_err(GuestRamError::report)?;

        let unsatisfiable = GuestRamError::AlignmentUnsatisfiable {
            align,
            len: region.len,
        };
        let host_base = align_up_u64(region.host_va, align)
            .ok_or(unsatisfiable)
            .map_err(GuestRamError::report)?;
        let head = host_base - region.host_va;
        let len = region
            .len
            .checked_sub(head)
            .ok_or(unsatisfiable)
            .map_err(GuestRamError::report)?
            & !(align - 1);
        if len == 0 {
            return Err(unsatisfiable.report());
        }

        Ok(Self {
            id: ImportId::allocate(),
            gpa_base: region.gpa_base + head,
            host_base: host_base as usize,
            len,
            align,
        })
    }

    /// This import's identity. Every [`GuestSlice`] it makes carries it.
    pub fn id(&self) -> ImportId {
        self.id
    }

    /// Guest physical address of the first byte covered.
    pub fn gpa_base(&self) -> u64 {
        self.gpa_base
    }

    /// Bytes covered. Always a multiple of [`Self::align`].
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Never true: [`Self::new`] refuses a region that would produce it. Present
    /// because a `len` without an `is_empty` is a lint, and because a caller
    /// reading it as a liveness check should get the right answer.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The backend's import granularity this import was built against.
    pub fn align(&self) -> u64 {
        self.align
    }

    /// The host address to import, for the backend's import call and nothing
    /// else.
    ///
    /// There is deliberately no `host_ptr_for(slice)`. The whole region is what
    /// gets imported, once; a per-slice host pointer is the raw-offset-plus-base
    /// arithmetic this module exists to make unwritable.
    pub fn host_base(&self) -> usize {
        self.host_base
    }

    /// Whether `gpa` falls inside the covered span.
    pub fn contains_gpa(&self, gpa: u64) -> bool {
        gpa >= self.gpa_base && gpa - self.gpa_base < self.len
    }

    /// The only constructor of a [`GuestSlice`].
    ///
    /// `offset` and `len` are relative to this import's covered base. The span
    /// is widened outward to the import granularity so the backend can bind it;
    /// the widening is reported by [`GuestSlice::head`], so the caller can find
    /// the byte it asked for without doing address arithmetic of its own.
    ///
    /// Refuses on overflow *before* comparing, so a wrapped end cannot compare
    /// as small and pass.
    pub fn slice(&self, offset: u64, len: u64) -> Result<GuestSlice, GuestRamError> {
        if len == 0 {
            return Err(GuestRamError::SliceEmpty.report());
        }
        let end = offset
            .checked_add(len)
            .ok_or(GuestRamError::SliceOverflow { offset, len })
            .map_err(GuestRamError::report)?;
        if end > self.len {
            return Err(GuestRamError::SliceEndPastImport {
                end,
                import_len: self.len,
            }
            .report());
        }
        let aligned_offset = offset & !(self.align - 1);
        let aligned_end = align_up_u64(end, self.align)
            .ok_or(GuestRamError::SliceOverflow { offset, len })
            .map_err(GuestRamError::report)?;
        if aligned_end > self.len {
            return Err(GuestRamError::SliceAlignedEndPastImport {
                aligned_end,
                import_len: self.len,
            }
            .report());
        }
        Ok(GuestSlice {
            import: self.id,
            offset: aligned_offset,
            len: aligned_end - aligned_offset,
            head: offset - aligned_offset,
            requested: len,
        })
    }

    /// [`Self::slice`] addressed by guest physical address, which is how every
    /// decoded resource names its bytes.
    pub fn slice_for_gpa(&self, gpa: u64, len: u64) -> Result<GuestSlice, GuestRamError> {
        let outside = GuestRamError::GpaOutsideImport {
            gpa,
            gpa_base: self.gpa_base,
            len: self.len,
        };
        let offset = gpa
            .checked_sub(self.gpa_base)
            .ok_or(outside)
            .map_err(GuestRamError::report)?;
        if offset >= self.len {
            return Err(outside.report());
        }
        self.slice(offset, len)
    }

    /// The byte range inside this import that `slice` names.
    ///
    /// The only way to obtain a slice's absolute position, and therefore the one
    /// place the cross-import check can be skipped by nobody. A slice from a
    /// different import is refused before any arithmetic runs: its offset is
    /// meaningful only against the import that built it.
    ///
    /// The bound is re-checked here as well as in [`Self::slice`]. It is two
    /// comparisons on the last control path before the GPU sees the range, and
    /// it is the check that would catch a `GuestSlice` field mutated by
    /// something that had no business mutating it.
    pub fn resolve(&self, slice: &GuestSlice) -> Result<BoundRange, GuestRamError> {
        if slice.import != self.id {
            return Err(GuestRamError::SliceForeignImport {
                slice: slice.import.get(),
                import: self.id.get(),
            }
            .report());
        }
        let end = slice
            .offset
            .checked_add(slice.len)
            .ok_or(GuestRamError::SliceOverflow {
                offset: slice.offset,
                len: slice.len,
            })
            .map_err(GuestRamError::report)?;
        if end > self.len {
            return Err(GuestRamError::SliceEndPastImport {
                end,
                import_len: self.len,
            }
            .report());
        }
        Ok(BoundRange {
            offset: slice.offset,
            len: slice.len,
        })
    }
}

/// A bounded reference to guest memory inside exactly one [`GuestRamImport`].
///
/// Constructible only through [`GuestRamImport::slice`] and
/// [`GuestRamImport::slice_for_gpa`]. It exposes no absolute offset and no host
/// pointer: [`Self::head`] and [`Self::requested`] are deltas *within* the
/// slice, useless as an address on their own, and the absolute position comes
/// from [`GuestRamImport::resolve`] and nowhere else.
///
/// `Clone`/`Copy` are fine — a copy names the same bytes in the same import and
/// carries the same identity, so it is bounded by the same check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GuestSlice {
    import: ImportId,
    offset: u64,
    len: u64,
    head: u64,
    requested: u64,
}

impl GuestSlice {
    /// Which import this slice may be resolved against.
    pub fn import(&self) -> ImportId {
        self.import
    }

    /// Bytes between the start of the bound range and the first byte the caller
    /// asked for, added by widening the span to the import granularity.
    pub fn head(&self) -> u64 {
        self.head
    }

    /// Bytes the caller asked for, which is `head + requested <= bound length`.
    pub fn requested(&self) -> u64 {
        self.requested
    }

    /// Length of the bound range, granularity included. Not an address.
    pub fn bound_len(&self) -> u64 {
        self.len
    }
}

/// The import granularity the active backend published, or 0 before any
/// backend has created a device that can import at all.
///
/// # Why this is a latch and not a parameter
///
/// The granularity is measured from the GPU — `minImportedHostPointerAlignment`
/// on Vulkan, the host page size on the Metal-direct arm — and it is needed by
/// [`crate::runtime::guest_ram_map`], which runs on the runtime side and holds
/// no device context. The alternative is threading a number from the backend
/// through every decode path that might name guest memory, which is how a site
/// ends up with a default nobody measured.
///
/// Zero means *no backend has said it can import*, which is the honest answer
/// on a host without the extension and the one the map treats as "run the
/// copying rails". A backend that resolves a negative capability rung must not
/// publish, so the absence of a number is itself the gate.
static GRANULARITY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Publish the granularity a freshly created device resolved to.
///
/// Called once per device creation, including each recreate, so a rebuilt
/// device republishes rather than leaving the previous one's answer standing. A
/// backend whose capability rung refused must call [`forget_granularity`]
/// instead — publishing a number from a device that declined the handle type is
/// how the map would build imports nothing can bind.
///
/// Refuses a granularity that is not a power of two: every alignment
/// computation here masks with `align - 1`, and a non-power-of-two would make
/// those masks name arbitrary bytes rather than refusing.
pub fn latch_granularity(align: u64) {
    if align == 0 || !align.is_power_of_two() {
        Emit::decline(EVENT, &GuestRamError::AlignmentNotPowerOfTwo { align }).fail();
        forget_granularity();
        return;
    }
    GRANULARITY.store(align, std::sync::atomic::Ordering::Relaxed);
}

/// Withdraw the published granularity: no device can import guest RAM.
pub fn forget_granularity() {
    GRANULARITY.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// The published granularity, or `None` when no backend can import.
///
/// `Relaxed` on both sides: this gates whether an optimization is attempted, and
/// a reader that sees the previous value takes the copying path for one window.
/// Nothing here orders access to any other memory.
pub fn granularity() -> Option<u64> {
    match GRANULARITY.load(std::sync::atomic::Ordering::Relaxed) {
        0 => None,
        align => Some(align),
    }
}

/// A bounded guest-memory reference a backend can bind: the import that owns
/// the bytes, and the slice inside it.
///
/// This is what replaces a page list plus an exported fd. Producing one costs a
/// range check and an `Arc` clone — no ioctl, no allocation, no kernel
/// reference per page — which is why nothing caches them.
///
/// The pair travels together because neither half is usable alone: a
/// [`GuestSlice`] carries no absolute offset by construction, and the import is
/// the only thing that can turn one into a [`BoundRange`].
#[derive(Clone, Debug)]
pub struct GuestRef {
    import: std::sync::Arc<GuestRamImport>,
    slice: GuestSlice,
}

impl GuestRef {
    /// Pair a slice with the import that made it.
    ///
    /// Refuses a mismatched pair up front rather than at bind time, so a
    /// mis-plumbed call site fails where it is written instead of one layer
    /// down inside the backend.
    pub fn new(
        import: std::sync::Arc<GuestRamImport>,
        slice: GuestSlice,
    ) -> Result<Self, GuestRamError> {
        import.resolve(&slice)?;
        Ok(Self { import, slice })
    }

    /// The import to bind against. The backend keys its device handle on
    /// [`GuestRamImport::id`] and imports [`GuestRamImport::host_base`] once.
    pub fn import(&self) -> &std::sync::Arc<GuestRamImport> {
        &self.import
    }

    /// The checked byte range inside that import.
    pub fn bound(&self) -> Result<BoundRange, GuestRamError> {
        self.import.resolve(&self.slice)
    }

    /// Bytes between the start of the bound range and the first byte the caller
    /// asked for.
    pub fn head(&self) -> u64 {
        self.slice.head()
    }

    /// Bytes the caller asked for.
    pub fn requested(&self) -> u64 {
        self.slice.requested()
    }
}

/// What a backend binds: a byte range inside one import, already checked.
///
/// Produced only by [`GuestRamImport::resolve`], so holding one is proof the
/// range was checked against the import that owns those bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundRange {
    /// Byte offset from [`GuestRamImport::host_base`], a multiple of the import
    /// granularity.
    pub offset: u64,
    /// Bytes to bind, a multiple of the import granularity.
    pub len: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 4 KiB-granular import over 64 KiB at a plausible host address. The host
    /// address is only ever compared, never dereferenced.
    fn import(len: u64, align: u64) -> GuestRamImport {
        GuestRamImport::new(
            GuestRamRegion {
                gpa_base: 0x1_0000_0000,
                host_va: 0x7f00_0000_0000,
                len,
            },
            align,
        )
        .expect("region is aligned and non-empty")
    }

    /// The bound, at the byte it exists for. A two-byte slice starting at the
    /// last byte names one byte past the end, and one byte past the end of a
    /// RAMBlock is this process's own memory.
    #[test]
    fn a_slice_that_ends_one_byte_past_the_import_refuses() {
        let import = import(0x1000, 1);
        assert_eq!(import.slice(0xfff, 1).map(|s| s.bound_len()), Ok(1));
        assert_eq!(
            import.slice(0xfff, 2),
            Err(GuestRamError::SliceEndPastImport {
                end: 0x1001,
                import_len: 0x1000,
            })
        );
    }

    /// Refused on the overflow, not on the comparison. `u64::MAX + 2` wraps to
    /// 1, which compares as comfortably inside a 4 KiB import — so a check
    /// written as `offset + len > self.len` admits the largest offset there is.
    /// The slug is what distinguishes the two, which is why they are separate
    /// variants rather than one "out of range".
    #[test]
    fn an_offset_that_overflows_refuses_on_the_overflow() {
        let import = import(0x1000, 1);
        assert_eq!(
            import.slice(u64::MAX, 2),
            Err(GuestRamError::SliceOverflow {
                offset: u64::MAX,
                len: 2,
            })
        );
        assert_eq!(
            u64::MAX.wrapping_add(2),
            1,
            "the wrapped end this test exists for must compare as inside"
        );
    }

    /// A slice is meaningful only against the import that built it. Against
    /// another one its offset is an arbitrary address into an unrelated mapping,
    /// and both imports here are the same length, so no range check would catch
    /// it.
    #[test]
    fn a_slice_from_another_import_cannot_be_resolved() {
        let a = import(0x1000, 1);
        let b = import(0x1000, 1);
        let from_a = a.slice(0x100, 0x10).expect("inside a");
        assert_eq!(
            a.resolve(&from_a),
            Ok(BoundRange {
                offset: 0x100,
                len: 0x10
            })
        );
        assert_eq!(
            b.resolve(&from_a),
            Err(GuestRamError::SliceForeignImport {
                slice: a.id().get(),
                import: b.id().get(),
            })
        );
    }

    /// Identities are never reused, so a slice outliving its import does not
    /// resolve against the import that replaced it. A device recreate over the
    /// same RAMBlock is exactly that situation.
    #[test]
    fn a_torn_down_imports_slices_do_not_resolve_against_its_replacement() {
        let first = import(0x1000, 1);
        let stale = first.slice(0, 0x10).expect("inside");
        let first_id = first.id();
        // The replacement is built over the same region with the same base,
        // length and granularity — everything a range check could compare. The
        // identity is the only thing that differs, and it is what refuses.
        let replacement = import(0x1000, 1);
        assert_ne!(replacement.id(), first_id);
        assert!(matches!(
            replacement.resolve(&stale),
            Err(GuestRamError::SliceForeignImport { .. })
        ));
    }

    /// Every refusal reaches the always-on sink by itself, naming the check.
    /// A bound violation that only returns an `Err` is one a caller can drop on
    /// the floor, which is the silent-failure class `AGENTS.md` forbids.
    #[test]
    fn the_refusal_reaches_the_always_on_log_with_a_named_reason() {
        let capture = crate::observe::FailCapture::start();
        let import = import(0x1000, 1);
        let _ = import.slice(0xfff, 2);
        let line = capture.one(EVENT);
        assert!(
            line.contains("reason=guest_ram_slice_end_past_import"),
            "{line}"
        );
        assert!(line.contains("end=4097"), "{line}");
        assert!(line.contains("import_len=4096"), "{line}");
    }

    /// The widening is outward and reported, never silent. A caller asking for
    /// 4 bytes at offset 5 of a 4 KiB-granular import gets the whole first
    /// granule and is told its bytes start 5 in — it does not have to
    /// reconstruct that from an offset it was handed back.
    #[test]
    fn a_slice_is_widened_to_the_granularity_and_says_by_how_much() {
        let import = import(0x2000, 0x1000);
        let slice = import.slice(5, 4).expect("inside");
        assert_eq!(import.resolve(&slice), Ok(BoundRange { offset: 0, len: 0x1000 }));
        assert_eq!(slice.head(), 5);
        assert_eq!(slice.requested(), 4);
        assert_eq!(slice.bound_len(), 0x1000);

        // Spanning a granule boundary widens both ends.
        let spanning = import.slice(0xffe, 4).expect("inside");
        assert_eq!(
            import.resolve(&spanning),
            Ok(BoundRange {
                offset: 0,
                len: 0x2000
            })
        );
        assert_eq!(spanning.head(), 0xffe);
    }

    /// Widening cannot leave the import, because the import's own length was
    /// rounded down to the same granularity. This pins the derivation that makes
    /// `SliceAlignedEndPastImport` a healthy zero.
    #[test]
    fn widening_the_last_byte_of_the_import_stays_inside_it() {
        for align in [1u64, 0x1000, 0x4000] {
            let import = import(0x8000, align);
            assert_eq!(import.len() % align, 0, "align {align}");
            let last = import.slice(import.len() - 1, 1).expect("inside");
            let bound = import.resolve(&last).expect("inside");
            assert_eq!(
                bound.offset + bound.len,
                import.len(),
                "widening left the import at align {align}"
            );
        }
    }

    /// A GPA below the covered base is outside, even when it is inside the
    /// untrimmed block. Resolving it against offset zero would silently read
    /// the wrong bytes.
    #[test]
    fn a_gpa_the_block_does_not_cover_refuses() {
        let region = GuestRamRegion {
            gpa_base: 0x1_0000_0000,
            // Deliberately off a 4 KiB granule by 0x800, so the import trims.
            host_va: 0x7f00_0000_0800,
            len: 0x4000,
        };
        let import = GuestRamImport::new(region, 0x1000).expect("trims to fit");
        assert_eq!(import.gpa_base(), region.gpa_base + 0x800);
        assert_eq!(import.host_base(), 0x7f00_0000_1000);
        assert_eq!(import.len(), 0x3000);

        assert!(matches!(
            import.slice_for_gpa(region.gpa_base, 4),
            Err(GuestRamError::GpaOutsideImport { .. })
        ));
        assert!(matches!(
            import.slice_for_gpa(region.gpa_base + 0x800 + 0x3000, 4),
            Err(GuestRamError::GpaOutsideImport { .. })
        ));
        assert!(import.slice_for_gpa(import.gpa_base(), 4).is_ok());
        assert!(import.contains_gpa(import.gpa_base()));
        assert!(!import.contains_gpa(region.gpa_base));
    }

    /// A block the granularity cannot fit in is refused by name, rather than
    /// producing a zero-length import something downstream has to notice.
    #[test]
    fn a_block_shorter_than_one_granule_is_refused_by_name() {
        let region = GuestRamRegion {
            gpa_base: 0,
            host_va: 0x7f00_0000_0000,
            len: 0x800,
        };
        assert_eq!(
            GuestRamImport::new(region, 0x1000).err(),
            Some(GuestRamError::AlignmentUnsatisfiable {
                align: 0x1000,
                len: 0x800,
            })
        );
    }

    /// The shim's answer is not trusted. Each malformed field has its own slug,
    /// because "the shim answered wrong" is not a diagnosis and the three cases
    /// have three different causes.
    #[test]
    fn a_malformed_region_is_refused_per_field() {
        let ok = GuestRamRegion {
            gpa_base: 0,
            host_va: 0x7f00_0000_0000,
            len: 0x4000,
        };
        assert_eq!(
            GuestRamImport::new(GuestRamRegion { len: 0, ..ok }, 0x1000).err(),
            Some(GuestRamError::RegionEmpty)
        );
        assert_eq!(
            GuestRamImport::new(GuestRamRegion { host_va: 0, ..ok }, 0x1000).err(),
            Some(GuestRamError::RegionUnmapped)
        );
        assert!(matches!(
            GuestRamImport::new(
                GuestRamRegion {
                    host_va: u64::MAX - 0xfff,
                    len: 0x4000,
                    ..ok
                },
                0x1000
            ),
            Err(GuestRamError::RegionWraps { .. })
        ));
        assert!(matches!(
            GuestRamImport::new(
                GuestRamRegion {
                    gpa_base: u64::MAX - 0xfff,
                    ..ok
                },
                0x1000
            ),
            Err(GuestRamError::RegionWraps { .. })
        ));
        for align in [0u64, 3, 0x1800] {
            assert_eq!(
                GuestRamImport::new(ok, align).err(),
                Some(GuestRamError::AlignmentNotPowerOfTwo { align })
            );
        }
    }

    /// A zero-length slice is not a reference. Admitting one would make
    /// `offset == len` — the byte past the end — a legal starting point.
    #[test]
    fn a_zero_length_slice_is_refused() {
        let import = import(0x1000, 1);
        assert_eq!(import.slice(0, 0), Err(GuestRamError::SliceEmpty));
        assert_eq!(import.slice(0x1000, 0), Err(GuestRamError::SliceEmpty));
        assert!(matches!(
            import.slice(0x1000, 1),
            Err(GuestRamError::SliceEndPastImport { .. })
        ));
    }

    /// One slug per check. Two checks sharing a slug is the defect the decline
    /// vocabulary exists to prevent: you watch it fire and still cannot tell
    /// which bound refused.
    #[test]
    fn every_refusal_has_its_own_slug() {
        let all = [
            GuestRamError::RegionEmpty,
            GuestRamError::RegionUnmapped,
            GuestRamError::RegionWraps {
                host_va: 0,
                len: 0,
            },
            GuestRamError::AlignmentNotPowerOfTwo { align: 0 },
            GuestRamError::AlignmentUnsatisfiable { align: 0, len: 0 },
            GuestRamError::SliceEmpty,
            GuestRamError::SliceOverflow { offset: 0, len: 0 },
            GuestRamError::SliceEndPastImport {
                end: 0,
                import_len: 0,
            },
            GuestRamError::SliceAlignedEndPastImport {
                aligned_end: 0,
                import_len: 0,
            },
            GuestRamError::SliceForeignImport { slice: 0, import: 0 },
            GuestRamError::GpaOutsideImport {
                gpa: 0,
                gpa_base: 0,
                len: 0,
            },
        ];
        let mut slugs: Vec<_> = all.iter().map(|e| e.slug()).collect();
        let count = slugs.len();
        slugs.sort_unstable();
        slugs.dedup();
        assert_eq!(slugs.len(), count, "two checks share a slug");
        for slug in slugs {
            assert!(slug.starts_with("guest_ram_"), "{slug}");
            assert!(
                slug.bytes().all(|b| b.is_ascii_lowercase() || b == b'_'),
                "{slug}"
            );
        }
    }
}
