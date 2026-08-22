//! The bound on every GPU reference to guest RAM, expressed as a pair of types.
//!
//! # What this replaces
//!
//! Guest pages reach the host GPU by importing the host virtual address range
//! that QEMU's RAMBlock already maps — `VK_EXT_external_memory_host` on Linux
//! and Windows, the same extension through MoltenVK on macOS, and
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
//! *make the invariant unrepresentable*. `reims-vgpu::runtime::guest_ram::GuestSlice` has exactly one
//! constructor, `GuestRamImport::slice`, which bounds-checks with checked
//! arithmetic against the import's own length. There is no second way to build
//! one, no public field that hands back a raw pointer or an absolute offset a
//! call site could re-add to something, and no `From` conversion. A new import
//! site cannot be written that skips the check, because there is nothing else to
//! write.
//!
//! The absolute position of a slice inside its import is obtainable only by
//! presenting the slice back to the import that made it
//! (`GuestRamImport::resolve`), which is also where the cross-import check
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
//! Page recycling is unchanged and still load-bearing: an import is retired
//! with the guest allocation that owns it. Physical-page equality is not an
//! ownership test; distinct views may intentionally name shared storage.
//!
//! # RAMBlock imports and packed task-buffer aliases
//!
//! GPA → HVA is linear *within* a RAMBlock, so one import covers every guest
//! page in it and a resource becomes an `(offset, len)` pair rather than a page
//! list somebody has to export. That is what makes `GuestRamImport::slice`
//! free — no ioctl, no allocation, no cache, no kernel reference per page — and
//! it is why a scattered surface is not un-importable: it is N slices over one
//! import.
//!
//! A task mapping whose physical pages are scattered has no single range inside
//! a RAMBlock. On a host that can construct a stable packed alias, that alias is
//! a second kind of import: one per live mapping, sliced by the resources that
//! mapping backs and retired with the mapping. A resource-owned alias remains a
//! fallback only when no mapping import is available. Importing either alias is
//! optional; a driver may refuse it and the existing gather remains the
//! correctness path. Neither is attempted per draw.

use reims_vgpu_observe::{Decline, Emit};

/// One packed-contiguous guest-RAM span exposed through a stable host alias.
#[derive(Clone, Copy, Debug)]
pub struct GuestRun {
    /// Host VA of the span start (page-aligned base + in-page offset).
    pub host_ptr: usize,
    /// Byte length of the span.
    pub len: u64,
}

/// A bounded guest-memory window and its two equivalent transport views.
///
/// `runs` lets a CPU fallback gather from stable host aliases. `pages` names
/// the same bytes as checked RAMBlock references so a capable executor can bind
/// or copy them without re-deriving their bounds. `physical_pages` is the
/// canonical identity shared with in-flight guest writes; it is deliberately
/// independent of either host representation because distinct Vulkan objects
/// may alias the same guest pages.
#[derive(Clone, Debug)]
pub struct GuestRunSource {
    pub runs: std::sync::Arc<Vec<GuestRun>>,
    pub source_offset: u64,
    pub total_len: u64,
    /// Guest row stride in texels; zero means tightly packed rows.
    pub row_length_texels: u32,
    pub pages: Option<std::sync::Arc<Vec<GuestWindowRun>>>,
    pub physical_pages: Option<GuestPageSet>,
}

/// One guest surface plane within a stable shared host allocation.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GuestTargetBacking {
    pub allocation_host_ptr: usize,
    pub allocation_len: u64,
    pub resource_offset: u64,
    pub resource_len: u64,
    pub plane_offset: u64,
    pub row_pitch: u64,
}

impl GuestTargetBacking {
    /// Allocation-relative byte window occupied by a 2D image in this layout.
    pub fn visible_window(
        self,
        width: u32,
        height: u32,
        bytes_per_texel: u64,
    ) -> Option<std::ops::Range<u64>> {
        if width == 0 || height == 0 || bytes_per_texel == 0 {
            return None;
        }
        let tight_row = u64::from(width).checked_mul(bytes_per_texel)?;
        if self.row_pitch < tight_row {
            return None;
        }
        let end = u64::from(height - 1)
            .checked_mul(self.row_pitch)?
            .checked_add(self.plane_offset)?
            .checked_add(tight_row)?;
        let resource_end = self.resource_offset.checked_add(self.resource_len)?;
        (self.plane_offset >= self.resource_offset
            && end <= resource_end
            && resource_end <= self.allocation_len)
            .then_some(self.plane_offset..end)
    }

    /// Allocation-relative byte window occupied by a typed image layout.
    pub fn visible_image_window(
        self,
        layout: GuestImageLayout,
        bytes_per_texel: u64,
    ) -> Option<std::ops::Range<u64>> {
        let span = layout.visible_span(self.row_pitch, bytes_per_texel)?;
        let end = self.plane_offset.checked_add(span)?;
        let resource_end = self.resource_offset.checked_add(self.resource_len)?;
        (self.plane_offset >= self.resource_offset
            && end <= resource_end
            && resource_end <= self.allocation_len)
            .then_some(self.plane_offset..end)
    }
}

/// An importable guest allocation and the exact physical pages it owns.
#[derive(Clone, Debug)]
pub struct GuestTargetMemory {
    pub backing: GuestTargetBacking,
    pub import: std::sync::Arc<GuestRamImport>,
    pub footprint: GuestPageFootprint,
}

/// One image whose authoritative texels live in a guest allocation, together
/// with its complete transfer representation and optional direct import.
///
/// These are two materializations of one backing, not two content sources.
/// A backend whose image-layout equation agrees may bind `direct` directly;
/// otherwise it consumes `transfer` through its ordinary upload path. Keeping
/// both behind one semantic object prevents a draw-time capability decision
/// from replacing the resource's identity with page runs.
#[derive(Clone, Debug)]
pub struct GuestImageSource {
    /// Directly importable materialization of this allocation, when the host
    /// transport can provide one. Absence changes placement, not image
    /// semantics; `transfer` still carries the same allocation.
    pub direct: Option<GuestTargetMemory>,
    pub allocation: GuestImageAllocationLayout,
    pub view: GuestImageViewRange,
    pub transfer: GuestRunSource,
}

impl GuestImageSource {
    /// Construct the exact one-mip allocation used by the existing direct
    /// image rails. Keeping this constructor beside the general allocation
    /// model makes the current subset explicit without baking it into the
    /// backend image type.
    pub fn single_mip(
        memory: GuestTargetMemory,
        layout: GuestImageLayout,
        transfer: GuestRunSource,
    ) -> Option<Self> {
        let mip = GuestImageMipLayout {
            resource_relative_offset: memory
                .backing
                .plane_offset
                .checked_sub(memory.backing.resource_offset)?,
            row_pitch: memory.backing.row_pitch,
            layout,
        };
        Some(Self {
            direct: Some(memory),
            allocation: GuestImageAllocationLayout {
                mips: std::sync::Arc::from([mip]),
            },
            view: GuestImageViewRange {
                base_mip_level: 0,
                mip_level_count: 1,
                base_array_layer: 0,
                array_layer_count: layout.array_layers(),
            },
            transfer,
        })
    }

    pub fn viewed_base_layout(&self) -> Option<GuestImageLayout> {
        let layout = self
            .allocation
            .mips
            .get(self.view.base_mip_level as usize)
            .map(|mip| mip.layout)?;
        match layout {
            GuestImageLayout::D1Array {
                width, array_pitch, ..
            } => Some(GuestImageLayout::D1Array {
                width,
                layers: self.view.array_layer_count,
                array_pitch,
            }),
            GuestImageLayout::D2Array {
                width,
                height,
                array_pitch,
                ..
            } => Some(GuestImageLayout::D2Array {
                width,
                height,
                layers: self.view.array_layer_count,
                array_pitch,
            }),
            other => (self.view.base_array_layer == 0 && self.view.array_layer_count == 1)
                .then_some(other),
        }
    }
}

/// One mip subresource's guest-resource-relative placement and geometry.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GuestImageMipLayout {
    /// Bytes from the start of the **guest resource** to this subresource.
    ///
    /// Spelled out because the other offset in play — [`GuestTargetBacking`]'s
    /// `plane_offset` — counts from the start of the parent **allocation**, and
    /// on this device an allocation is a whole RAMBlock. The two therefore
    /// differ by hundreds of megabytes on a live boot while both being small
    /// non-negative integers in a test, which is exactly the shape a basis
    /// confusion hides in. Cross between them with [`Self::plane_in`].
    pub resource_relative_offset: u64,
    pub row_pitch: u64,
    pub layout: GuestImageLayout,
}

impl GuestImageMipLayout {
    /// This subresource's own backing inside the allocation holding its
    /// resource.
    ///
    /// This is the only correct route from a mip chain to a
    /// [`GuestTargetBacking`], and being the only route is the point. Reading
    /// `resource_relative_offset` straight into `plane_offset` names a byte
    /// short of the resource by `resource_offset`, and every bound the backing
    /// then checks — `visible_image_window`'s `plane_offset >= resource_offset`
    /// first — refuses a placement that is in fact perfectly valid.
    pub fn plane_in(self, backing: GuestTargetBacking) -> Option<GuestTargetBacking> {
        Some(GuestTargetBacking {
            plane_offset: backing
                .resource_offset
                .checked_add(self.resource_relative_offset)?,
            row_pitch: self.row_pitch,
            ..backing
        })
    }
}

/// Complete mip layout of one Vulkan-compatible guest image allocation.
///
/// Array layers remain inside each mip's [`GuestImageLayout`], where their
/// full-chain pitch is explicit. Mip offsets are relative to the guest
/// resource, independent of host import padding, and are translated and
/// validated against every Vulkan subresource layout at materialization.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct GuestImageAllocationLayout {
    pub mips: std::sync::Arc<[GuestImageMipLayout]>,
}

impl GuestImageAllocationLayout {
    /// A one-level chain whose sole subresource sits `resource_relative_offset`
    /// bytes into the guest resource — **not** into the parent allocation. A
    /// caller holding a `GuestTargetBacking` reaches this by subtracting that
    /// backing's `resource_offset` from its `plane_offset`.
    pub fn single(resource_relative_offset: u64, row_pitch: u64, layout: GuestImageLayout) -> Self {
        Self {
            mips: std::sync::Arc::from([GuestImageMipLayout {
                resource_relative_offset,
                row_pitch,
                layout,
            }]),
        }
    }

    pub fn base(&self) -> Option<GuestImageMipLayout> {
        self.mips.first().copied()
    }

    /// Allocation-relative byte window covering every level of this chain.
    ///
    /// The guest places each level independently, so the chain's extent is the
    /// union of the levels' own windows rather than level zero's window scaled
    /// by anything. `None` if any level's placement is not expressible against
    /// `backing`, which is the same refusal
    /// [`GuestTargetBacking::visible_image_window`] makes for one level.
    ///
    /// Levels are not assumed to be in address order: a chain the guest laid
    /// out smallest-first has level zero at the far end, and taking
    /// `first.start..last.end` would name a window running backwards.
    pub fn visible_chain_window(
        &self,
        backing: GuestTargetBacking,
        bytes_per_texel: u64,
    ) -> Option<std::ops::Range<u64>> {
        let mut chain: Option<std::ops::Range<u64>> = None;
        for mip in self.mips.iter() {
            let level = mip
                .plane_in(backing)?
                .visible_image_window(mip.layout, bytes_per_texel)?;
            chain = Some(match chain {
                None => level,
                Some(so_far) => so_far.start.min(level.start)..so_far.end.max(level.end),
            });
        }
        chain
    }

    pub fn mip_level_count(&self) -> Option<u32> {
        u32::try_from(self.mips.len())
            .ok()
            .filter(|count| *count != 0)
    }

    /// Whether this declaration can be one Vulkan image mip chain.
    ///
    /// The guest supplies every level independently. Vulkan derives later
    /// extents from level zero, so accepting two individually valid levels is
    /// insufficient: their dimensional family, layer domain, and halving
    /// sequence must also agree.
    pub fn is_vulkan_mip_chain(&self, bytes_per_texel: u64) -> bool {
        let Some(base) = self.base() else {
            return false;
        };
        if bytes_per_texel == 0 || base.layout.width() == 0 || base.layout.height() == 0 {
            return false;
        }
        self.mips.iter().enumerate().all(|(index, mip)| {
            let Ok(level) = u32::try_from(index) else {
                return false;
            };
            let reduced = |extent: u32| extent.checked_shr(level).unwrap_or(0).max(1);
            let geometry_matches = mip.layout.width() == reduced(base.layout.width())
                && mip.layout.height() == reduced(base.layout.height())
                && mip.layout.depth() == reduced(base.layout.depth());
            let family_matches = match (base.layout, mip.layout) {
                (GuestImageLayout::D1 { .. }, GuestImageLayout::D1 { .. })
                | (GuestImageLayout::D2 { .. }, GuestImageLayout::D2 { .. }) => true,
                (
                    GuestImageLayout::D1Array {
                        layers,
                        array_pitch,
                        ..
                    },
                    GuestImageLayout::D1Array {
                        layers: next_layers,
                        array_pitch: next_pitch,
                        ..
                    },
                )
                | (
                    GuestImageLayout::D2Array {
                        layers,
                        array_pitch,
                        ..
                    },
                    GuestImageLayout::D2Array {
                        layers: next_layers,
                        array_pitch: next_pitch,
                        ..
                    },
                ) => layers == next_layers && array_pitch == next_pitch,
                (GuestImageLayout::D3 { .. }, GuestImageLayout::D3 { .. }) => true,
                _ => false,
            };
            geometry_matches
                && family_matches
                && mip.row_pitch.is_multiple_of(bytes_per_texel)
                && mip
                    .layout
                    .visible_span(mip.row_pitch, bytes_per_texel)
                    .is_some()
        })
    }
}

/// Texture-view subresource range over a [`GuestImageAllocationLayout`].
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GuestImageViewRange {
    pub base_mip_level: u32,
    pub mip_level_count: u32,
    pub base_array_layer: u32,
    pub array_layer_count: u32,
}

impl GuestImageViewRange {
    pub fn fits(self, allocation: &GuestImageAllocationLayout) -> bool {
        let Some(mip_count) = allocation.mip_level_count() else {
            return false;
        };
        let Some(mip_end) = self.base_mip_level.checked_add(self.mip_level_count) else {
            return false;
        };
        let Some(base) = allocation.base() else {
            return false;
        };
        let Some(layer_end) = self.base_array_layer.checked_add(self.array_layer_count) else {
            return false;
        };
        self.mip_level_count != 0
            && self.array_layer_count != 0
            && mip_end <= mip_count
            && layer_end <= base.layout.array_layers()
    }
}

/// Complete dimensional and inter-subresource layout of one guest image.
///
/// The enum keeps array layers and volume slices distinct. Both are a third
/// coordinate to a shader, but Vulkan represents the former with
/// `arrayLayers`/`arrayPitch` and the latter with `extent.depth`/`depthPitch`.
/// Collapsing either to a generic layer count loses the equation required for
/// a direct image binding.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GuestImageLayout {
    D1 {
        width: u32,
    },
    D1Array {
        width: u32,
        layers: u32,
        array_pitch: u64,
    },
    D2 {
        width: u32,
        height: u32,
    },
    D2Array {
        width: u32,
        height: u32,
        layers: u32,
        array_pitch: u64,
    },
    D3 {
        width: u32,
        height: u32,
        depth: u32,
        depth_pitch: u64,
    },
}

impl GuestImageLayout {
    pub const fn width(self) -> u32 {
        match self {
            Self::D1 { width }
            | Self::D1Array { width, .. }
            | Self::D2 { width, .. }
            | Self::D2Array { width, .. }
            | Self::D3 { width, .. } => width,
        }
    }

    pub const fn height(self) -> u32 {
        match self {
            Self::D1 { .. } | Self::D1Array { .. } => 1,
            Self::D2 { height, .. } | Self::D2Array { height, .. } | Self::D3 { height, .. } => {
                height
            }
        }
    }

    pub const fn depth(self) -> u32 {
        match self {
            Self::D3 { depth, .. } => depth,
            _ => 1,
        }
    }

    pub const fn array_layers(self) -> u32 {
        match self {
            Self::D1Array { layers, .. } | Self::D2Array { layers, .. } => layers,
            _ => 1,
        }
    }

    pub const fn is_arrayed(self) -> bool {
        matches!(self, Self::D1Array { .. } | Self::D2Array { .. })
    }

    pub const fn is_volume(self) -> bool {
        matches!(self, Self::D3 { .. })
    }

    pub const fn is_one_dimensional(self) -> bool {
        matches!(self, Self::D1 { .. } | Self::D1Array { .. })
    }

    /// Final visible byte relative to the image plane's first byte.
    pub fn visible_span(self, row_pitch: u64, bytes_per_texel: u64) -> Option<u64> {
        if bytes_per_texel == 0 {
            return None;
        }
        let tight_row = u64::from(self.width()).checked_mul(bytes_per_texel)?;
        if self.width() == 0 || row_pitch < tight_row {
            return None;
        }
        let rows = u64::from(self.height().checked_sub(1)?).checked_mul(row_pitch)?;
        let slice = rows.checked_add(tight_row)?;
        match self {
            Self::D1 { .. } | Self::D2 { .. } => Some(slice),
            Self::D1Array {
                layers,
                array_pitch,
                ..
            }
            | Self::D2Array {
                layers,
                array_pitch,
                ..
            } => {
                if layers == 0 || array_pitch < slice {
                    return None;
                }
                u64::from(layers - 1)
                    .checked_mul(array_pitch)?
                    .checked_add(slice)
            }
            Self::D3 {
                depth, depth_pitch, ..
            } => {
                if depth == 0 || depth_pitch < slice {
                    return None;
                }
                u64::from(depth - 1)
                    .checked_mul(depth_pitch)?
                    .checked_add(slice)
            }
        }
    }
}

/// Backend-neutral declaration used to size a host image materialization
/// before its guest-page alias is created.
///
/// `allocation` remains guest-resource-relative while `backing` names that
/// resource's placement in a candidate host import. The backend may require
/// trailing host-only bytes, but it may not change either coordinate to make
/// its own layout fit.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct GuestImageBindingRequest {
    pub backing: GuestTargetBacking,
    pub allocation: GuestImageAllocationLayout,
    pub format: reims_vgpu_protocol::ImageFormat,
}

/// Stable resource-owned key for one backend image-layout query.
///
/// The host pointer and allocation extent are deliberately absent: adding
/// host-only binding padding may replace both without changing the image whose
/// requirement was queried. The remaining fields are every guest-layout term
/// that can affect either the Vulkan image requirements or its allocation-
/// relative binding offset.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct GuestImageBindingKey {
    pub resource_offset: u64,
    pub resource_len: u64,
    pub plane_offset: u64,
    pub row_pitch: u64,
    pub allocation: GuestImageAllocationLayout,
    pub format: reims_vgpu_protocol::ImageFormat,
}

impl GuestImageBindingRequest {
    pub fn key(&self) -> GuestImageBindingKey {
        GuestImageBindingKey {
            resource_offset: self.backing.resource_offset,
            resource_len: self.backing.resource_len,
            plane_offset: self.backing.plane_offset,
            row_pitch: self.backing.row_pitch,
            allocation: self.allocation.clone(),
            format: self.format,
        }
    }
}

/// Exact host allocation extent required to bind one declared image.
///
/// Bytes beyond the guest resource are host-only binding padding. They never
/// enter a [`GuestPageFootprint`] and cannot be published as guest content.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GuestImageBindingRequirement {
    pub allocation_len: u64,
}

/// Stable backend admission for one resource-owned image layout.
///
/// A refusal means the declared guest layout cannot be represented directly
/// and the caller must use its copying rail. Transient backend failures are
/// not values of this type and therefore cannot be retained as resource state.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GuestImageBindingDisposition {
    Direct(GuestImageBindingRequirement),
    Refused,
}

impl GuestTargetMemory {
    /// Physical pages touched by the visible image, translating the image's
    /// parent-import coordinates into this resource footprint's coordinates.
    pub fn visible_footprint(
        &self,
        width: u32,
        height: u32,
        bytes_per_texel: u64,
    ) -> Option<GuestPageFootprint> {
        self.window_footprint(
            self.backing
                .visible_window(width, height, bytes_per_texel)?,
        )
    }

    /// Physical pages touched by an allocation-relative byte window, in this
    /// resource footprint's coordinates.
    ///
    /// The two window-shaped questions below both land here so the coordinate
    /// translation — allocation-relative in, resource-footprint-relative out —
    /// is written once. Getting it wrong names pages hundreds of megabytes from
    /// the ones the image occupies, and both bases are small integers in a test.
    pub fn window_footprint(&self, window: std::ops::Range<u64>) -> Option<GuestPageFootprint> {
        let resource_head = self.backing.resource_offset % self.footprint.page_size();
        let start = window
            .start
            .checked_sub(self.backing.resource_offset)?
            .checked_add(resource_head)?;
        let end = window
            .end
            .checked_sub(self.backing.resource_offset)?
            .checked_add(resource_head)?;
        self.footprint.window(start..end)
    }

    /// Canonical page set written by this declared image window.
    pub fn visible_write_pages(
        &self,
        width: u32,
        height: u32,
        bytes_per_texel: u64,
    ) -> Option<GuestWritePages> {
        let footprint = self.visible_footprint(width, height, bytes_per_texel)?;
        GuestWritePages::new(footprint.pages())
    }

    /// Canonical page set written by every level of a declared mip chain.
    ///
    /// The whole-chain counterpart of [`Self::visible_write_pages`], and the
    /// one an aliasing image owes: such an image's birth copy writes each
    /// level, so page ownership has to be asked about each level before the
    /// image exists. For a one-level chain the two agree.
    pub fn chain_write_pages(
        &self,
        allocation: &GuestImageAllocationLayout,
        bytes_per_texel: u64,
    ) -> Option<GuestWritePages> {
        let window = allocation.visible_chain_window(self.backing, bytes_per_texel)?;
        GuestWritePages::new(self.window_footprint(window)?.pages())
    }
}

/// A render attachment's prior contents in its bounded guest-memory form.
#[derive(Clone, Debug)]
pub struct GuestTargetSeed {
    pub source: GuestRunSource,
    /// Physical texel layout of the guest bytes.
    pub format: reims_vgpu_protocol::TexelLayout,
}

/// The guest-memory side of one render attachment.
///
/// A seed is content only. A backing is the allocation the executor may bind
/// as the attachment when its native layout agrees exactly; its optional seed
/// says whether this pass must observe the guest's current bytes on LOAD.
#[derive(Clone, Debug)]
pub enum GuestTargetPlan {
    Seed(GuestTargetSeed),
    Backing {
        memory: GuestTargetMemory,
        seed: Option<GuestTargetSeed>,
    },
}

impl GuestTargetPlan {
    pub fn seed(&self) -> Option<&GuestTargetSeed> {
        match self {
            Self::Seed(seed) => Some(seed),
            Self::Backing { seed, .. } => seed.as_ref(),
        }
    }

    pub fn memory(&self) -> Option<&GuestTargetMemory> {
        match self {
            Self::Seed(_) => None,
            Self::Backing { memory, .. } => Some(memory),
        }
    }
}

/// Describe the exact guest-memory window needed to seed an attachment LOAD.
pub fn guest_target_seed(
    memory: &GuestTargetMemory,
    width: u32,
    height: u32,
    format: reims_vgpu_protocol::TexelLayout,
) -> Option<GuestTargetSeed> {
    if width == 0 || height == 0 || memory.import.is_retired() {
        return None;
    }
    let texel = u64::from(format.bytes_per_texel());
    let tight_row = u64::from(width).checked_mul(texel)?;
    let row_pitch = memory.backing.row_pitch;
    if row_pitch < tight_row || !row_pitch.is_multiple_of(texel) {
        return None;
    }
    let span = u64::from(height - 1)
        .checked_mul(row_pitch)?
        .checked_add(tight_row)?;
    let resource_end = memory
        .backing
        .resource_offset
        .checked_add(memory.backing.resource_len)?;
    let plane_end = memory.backing.plane_offset.checked_add(span)?;
    if memory.backing.plane_offset < memory.backing.resource_offset || plane_end > resource_end {
        return None;
    }
    let slice = memory
        .import
        .slice(memory.backing.plane_offset, span)
        .ok()?;
    let guest = GuestRef::new(std::sync::Arc::clone(&memory.import), slice).ok()?;
    let host_ptr = memory
        .import
        .host_base()
        .checked_add(usize::try_from(memory.backing.plane_offset).ok()?)?;
    let row_length_texels = if row_pitch == tight_row {
        0
    } else {
        u32::try_from(row_pitch / texel).ok()?
    };
    let physical_pages = memory.visible_write_pages(width, height, texel);
    Some(GuestTargetSeed {
        source: GuestRunSource {
            runs: std::sync::Arc::new(vec![GuestRun {
                host_ptr,
                len: span,
            }]),
            source_offset: 0,
            total_len: span,
            row_length_texels,
            pages: Some(std::sync::Arc::new(vec![GuestWindowRun {
                window_offset: 0,
                guest,
            }])),
            physical_pages,
        },
        format,
    })
}

/// A render or compute frame's bounded destination in guest pages.
#[derive(Debug)]
pub struct GuestPageTarget {
    pub runs: Vec<GuestWindowRun>,
    /// Guest row pitch in texels; zero/tighter-than-width means tight rows.
    pub row_length_texels: u32,
    pub width: u32,
    pub height: u32,
    /// Physical texel layout the guest declared for this destination.
    pub format: reims_vgpu_protocol::StorageImageFormat,
}

/// Topology-independent shape of a GPU landing into guest pages.
///
/// Padded rows require texel rectangles so padding remains untouched. Dense
/// rows may be detiled once and scattered as bytes. Both variants describe the
/// same guest-visible write; host topology and import capability may choose how
/// the executor materializes the declared operation, never which bytes exist.
#[derive(Clone, Copy, Debug)]
pub enum GuestPageTransferPlan {
    PitchedRectangles {
        geometry: reims_vgpu_paging::regions::WindowGeometry,
    },
    DenseScatter {
        window_bytes: u64,
    },
}

impl GuestPageTarget {
    pub fn extent_end(&self) -> u64 {
        let rows_before = u64::from(self.height.saturating_sub(1));
        rows_before * self.pitch_bytes()
            + u64::from(self.width) * self.format.bytes_per_texel() as u64
    }

    pub fn window_bytes(&self) -> u64 {
        self.runs
            .iter()
            .map(|run| run.guest.requested())
            .fold(0_u64, u64::saturating_add)
    }

    pub fn pitch_bytes(&self) -> u64 {
        u64::from(self.row_length_texels.max(self.width)) * self.format.bytes_per_texel() as u64
    }

    pub fn geometry(&self) -> reims_vgpu_paging::regions::WindowGeometry {
        reims_vgpu_paging::regions::WindowGeometry {
            pitch_bytes: self.pitch_bytes(),
            width_texels: self.width,
            height_texels: self.height,
        }
    }

    pub fn rows_are_dense(&self) -> bool {
        self.pitch_bytes() == u64::from(self.width) * self.format.bytes_per_texel() as u64
    }

    pub fn transfer_plan(&self) -> GuestPageTransferPlan {
        if self.rows_are_dense() {
            GuestPageTransferPlan::DenseScatter {
                window_bytes: self.window_bytes(),
            }
        } else {
            GuestPageTransferPlan::PitchedRectangles {
                geometry: self.geometry(),
            }
        }
    }
}

/// One stretch of a [`GuestRunSource`]'s requested window, clipped to it.
#[derive(Clone, Copy, Debug)]
pub struct WindowStretch<'a> {
    pub guest: &'a GuestRef,
    pub skip: u64,
    pub window_offset: u64,
    pub len: u64,
}

/// Allocation-free iterator over the checked stretches of one source window.
#[derive(Clone, Debug)]
pub struct WindowStretches<'a> {
    runs: std::slice::Iter<'a, GuestWindowRun>,
    source_offset: u64,
    wanted_end: u64,
}

impl<'a> Iterator for WindowStretches<'a> {
    type Item = WindowStretch<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        for run in self.runs.by_ref() {
            let run_end = run.window_offset.checked_add(run.guest.requested())?;
            let start = run.window_offset.max(self.source_offset);
            let end = run_end.min(self.wanted_end);
            if start >= end {
                continue;
            }
            return Some(WindowStretch {
                guest: &run.guest,
                skip: start - run.window_offset,
                window_offset: start - self.source_offset,
                len: end - start,
            });
        }
        None
    }
}

/// Topology-independent read shape for one bounded guest window.
///
/// `GpuVisible` means checked guest references tile the complete logical
/// window. A capable executor may bind the exact direct stretch or gather the
/// complete iterator. `CpuOnly` preserves the stable host-run path when no
/// checked import view exists; it is a transport distinction, not a content or
/// lifecycle decision.
#[derive(Clone, Debug)]
pub enum GuestReadTransferPlan<'a> {
    CpuOnly,
    GpuVisible {
        direct: Option<WindowStretch<'a>>,
        stretches: WindowStretches<'a>,
    },
}

impl<'a> GuestReadTransferPlan<'a> {
    pub fn direct(&self) -> Option<WindowStretch<'a>> {
        match self {
            Self::GpuVisible { direct, .. } => *direct,
            Self::CpuOnly => None,
        }
    }

    pub fn stretches(&self) -> Option<WindowStretches<'a>> {
        match self {
            Self::GpuVisible { stretches, .. } => Some(stretches.clone()),
            Self::CpuOnly => None,
        }
    }
}

impl GuestRunSource {
    /// The source window as one importable guest stretch, when it fits wholly
    /// inside the one checked RAMBlock reference supplied by the resolver.
    pub fn single_stretch(&self) -> Option<WindowStretch<'_>> {
        let [only] = self.pages.as_ref()?.as_slice() else {
            return None;
        };
        if only.window_offset != 0 {
            return None;
        }
        let end = self.source_offset.checked_add(self.total_len)?;
        if end > only.guest.requested() {
            return None;
        }
        Some(WindowStretch {
            guest: &only.guest,
            skip: self.source_offset,
            window_offset: 0,
            len: self.total_len,
        })
    }

    /// Every checked guest stretch touched by this source window, in window
    /// order. Returned lengths tile `total_len` when the source is valid.
    pub fn window_stretches(&self) -> Option<WindowStretches<'_>> {
        let pages = self.pages.as_ref()?;
        let wanted_end = self.source_offset.checked_add(self.total_len)?;
        Some(WindowStretches {
            runs: pages.iter(),
            source_offset: self.source_offset,
            wanted_end,
        })
    }

    /// Classify the complete window once for every executor consumer.
    pub fn transfer_plan(&self) -> GuestReadTransferPlan<'_> {
        let Some(stretches) = self.window_stretches() else {
            return GuestReadTransferPlan::CpuOnly;
        };
        let mut expected = 0u64;
        for stretch in stretches.clone() {
            if stretch.window_offset != expected {
                return GuestReadTransferPlan::CpuOnly;
            }
            let Some(next) = expected.checked_add(stretch.len) else {
                return GuestReadTransferPlan::CpuOnly;
            };
            expected = next;
        }
        if expected != self.total_len {
            return GuestReadTransferPlan::CpuOnly;
        }
        GuestReadTransferPlan::GpuVisible {
            direct: self.single_stretch(),
            stretches,
        }
    }
}

fn align_up_u64(value: u64, align: u64) -> Option<u64> {
    if !align.is_power_of_two() {
        return None;
    }
    value
        .checked_add(align - 1)
        .map(|value| value & !(align - 1))
}

/// Exact physical footprint retained with one imported guest allocation.
///
/// `pages` is the allocation order required by alias and ownership checks;
/// `runs` is the same set partitioned into physically contiguous stretches.
/// Deriving the partition once makes a resource—not each Store consumer—the
/// authority on how its scattered pages fit together.
///
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GuestPageFootprint {
    pages: std::sync::Arc<[u64]>,
    runs: std::sync::Arc<[std::ops::Range<usize>]>,
    page_size: u64,
}

/// Canonical physical-page identity shared by guest-memory readers and writers.
///
/// Construction canonicalizes the set once. Read-side alias checks and
/// submission visibility ledgers then compare the immutable result instead of
/// inferring identity from distinct host or Vulkan objects over the same bytes.
#[derive(Clone, Debug)]
pub struct GuestPageSet {
    pages: std::sync::Arc<[u64]>,
    fingerprint: u64,
}

impl PartialEq for GuestPageSet {
    fn eq(&self, other: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.pages, &other.pages)
            || (self.fingerprint == other.fingerprint && self.pages == other.pages)
    }
}

impl Eq for GuestPageSet {}

impl std::hash::Hash for GuestPageSet {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u64(self.fingerprint);
    }
}

/// Submitted-write spelling retained at ownership sites. Reads and writes use
/// the same canonical physical identity when deciding whether aliases overlap.
pub type GuestWritePages = GuestPageSet;

impl GuestPageSet {
    pub fn new(pages: &[u64]) -> Option<Self> {
        if pages.is_empty() {
            return None;
        }
        let mut pages = pages.to_vec();
        pages.sort_unstable();
        pages.dedup();
        let mut fingerprint = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&pages, &mut fingerprint);
        Some(Self {
            pages: pages.into(),
            fingerprint: std::hash::Hasher::finish(&fingerprint),
        })
    }

    pub fn pages(&self) -> &[u64] {
        &self.pages
    }

    /// Whether two canonical physical footprints share at least one page.
    ///
    /// It is the exact invalidation question for derived GPU copies of guest
    /// storage, and on the x86 rail it is asked once per draw against the
    /// colour target's own footprint.
    ///
    /// # Why this is a search and not a merge
    ///
    /// Both inputs are sorted and deduplicated at construction, which a plain
    /// linear merge uses only to avoid allocating. That is the wrong shape for
    /// the question actually asked here: the two sets are usually *disjoint
    /// ranges* -- a sampled texture and a render target are different guest
    /// allocations -- and a merge proves that by walking every page of the
    /// larger one.
    ///
    /// Measured, driven fullscreen Maps on macos-13/x86: `draw_phase`'s
    /// `post_store_us` was **1.252 us a draw**, 5.7 % of the 22.01 us the drain
    /// worker spends on a draw, and `store_routes` says the walk it did was
    /// `sampled_bindmap_write_disjoint` 0.88 times a draw with
    /// `..._overlap` and `..._unknown` both **zero**. So the whole span was a
    /// full-length merge over a 1920x1080 target's 2 025 pages, every draw,
    /// always answering "no".
    ///
    /// The sortedness answers it far more cheaply. Two disjoint ranges are
    /// separated by a single comparison of one end against the other, and where
    /// the ranges do interleave, `partition_point` skips the whole run of pages
    /// below the other side's front instead of stepping through it. Each
    /// iteration advances one side strictly past the other's current front, so
    /// the loop runs at most `2 * min(len) + 1` times, each a binary search.
    /// Nothing here is a cache or a remembered answer: it is the same total
    /// function over the same inputs, using an invariant [`Self::new`] already
    /// establishes.
    ///
    /// Measured on the same rail, two boots an arm: `post_store_us` fell from
    /// 1.196 and 1.247 us a draw to **0.396 and 0.419**, disjoint. Score a
    /// change of this size on its own span and not on `proc_us`, which has a
    /// 3.4 % coefficient of variation within one boot population and cannot
    /// resolve 3.7 % however many boots it is given.
    pub fn overlaps(&self, other: &Self) -> bool {
        let mut left = self.pages();
        let mut right = other.pages();
        loop {
            // `new` refuses an empty set, but a slice narrowed by the skips
            // below can become one, and that is the disjoint answer.
            let (Some(&left_front), Some(&right_front)) = (left.first(), right.first()) else {
                return false;
            };
            let (Some(&left_back), Some(&right_back)) = (left.last(), right.last()) else {
                return false;
            };
            // Sorted, so one range ending below the other's start settles it.
            if left_front > right_back || right_front > left_back {
                return false;
            }
            match left_front.cmp(&right_front) {
                std::cmp::Ordering::Equal => return true,
                // Skip every page below the other side's front in one step.
                // The range check above guarantees this leaves something.
                std::cmp::Ordering::Less => {
                    left = &left[left.partition_point(|&page| page < right_front)..];
                }
                std::cmp::Ordering::Greater => {
                    right = &right[right.partition_point(|&page| page < left_front)..];
                }
            }
        }
    }
}

impl GuestPageFootprint {
    pub fn new(pages: std::sync::Arc<[u64]>, page_size: u64) -> Option<Self> {
        if pages.is_empty() || !page_size.is_power_of_two() {
            return None;
        }
        let runs = reims_vgpu_paging::runs::contig_page_runs(&pages, page_size).into();
        Some(Self {
            pages,
            runs,
            page_size,
        })
    }

    pub fn pages(&self) -> &[u64] {
        &self.pages
    }

    pub fn runs(&self) -> &[std::ops::Range<usize>] {
        &self.runs
    }

    pub fn page_size(&self) -> u64 {
        self.page_size
    }

    pub fn pages_arc(&self) -> std::sync::Arc<[u64]> {
        std::sync::Arc::clone(&self.pages)
    }

    /// Physical pages intersecting one allocation-relative byte window.
    pub fn window(&self, bytes: std::ops::Range<u64>) -> Option<Self> {
        if bytes.start >= bytes.end {
            return None;
        }
        let allocation_len = (self.pages.len() as u64).checked_mul(self.page_size)?;
        if bytes.end > allocation_len {
            return None;
        }
        let first = usize::try_from(bytes.start / self.page_size).ok()?;
        let last = usize::try_from((bytes.end - 1) / self.page_size).ok()?;
        Self::new(self.pages[first..=last].into(), self.page_size)
    }

    /// Visit the exact physical byte runs reached by an allocation-relative
    /// byte window. Scatter gaps are never joined.
    pub fn visit_window(&self, off: u64, len: u64, mut visit: impl FnMut(u64, u64)) {
        if len == 0 {
            return;
        }
        let end = off.saturating_add(len);
        let first_page = off / self.page_size;
        let last_page_exclusive = ((end - 1) / self.page_size).saturating_add(1);
        for run in self.runs.iter() {
            let start = run.start.max(first_page as usize);
            let stop = run.end.min(last_page_exclusive as usize);
            if start >= stop {
                continue;
            }
            let logical_base = (run.start as u64).saturating_mul(self.page_size);
            let logical_lo = off.max((start as u64).saturating_mul(self.page_size));
            let logical_hi = end.min((stop as u64).saturating_mul(self.page_size));
            let physical_lo = self.pages[run.start].saturating_add(logical_lo - logical_base);
            if logical_lo < logical_hi {
                visit(physical_lo, logical_hi - logical_lo);
            }
        }
    }
}

/// One RAMBlock as the host shim describes it: where it starts in guest physical
/// address space, where QEMU mapped it, and how long it is.
///
/// This is the shape `reims-vgpu::runtime::host::GuestRamProvider::guest_ram_regions` hands back,
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
    /// A backend published a granularity beside a zero import budget: it says it
    /// can import guest RAM and that no heap on it can hold any. Broken rather
    /// than restrictive, and refused so the copying rails run instead of a rail
    /// whose every import is over budget.
    ImportBudgetEmpty,
    /// A backend published a span ceiling smaller than its own import
    /// granularity: no chunk of a RAMBlock could be both inside the ceiling and
    /// a whole number of granules, so the rail has no legal import size at all.
    /// Broken rather than restrictive, and refused so the copying rails run.
    ImportSpanMaxBelowGranularity { span_max: u64, align: u64 },
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
            Self::ImportBudgetEmpty => "guest_ram_import_budget_empty",
            Self::ImportSpanMaxBelowGranularity { .. } => {
                "guest_ram_import_span_max_below_granularity"
            }
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
            Self::RegionEmpty
            | Self::RegionUnmapped
            | Self::SliceEmpty
            | Self::ImportBudgetEmpty => Vec::new(),
            Self::RegionWraps { host_va, len } => {
                vec![
                    ("host_va", format!("{host_va:#x}")),
                    ("len", len.to_string()),
                ]
            }
            Self::AlignmentNotPowerOfTwo { align } => vec![("align", align.to_string())],
            Self::AlignmentUnsatisfiable { align, len } => {
                vec![("align", align.to_string()), ("len", len.to_string())]
            }
            Self::ImportSpanMaxBelowGranularity { span_max, align } => vec![
                ("span_max", span_max.to_string()),
                ("align", align.to_string()),
            ],
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

reims_vgpu_observe::decline_display!(GuestRamError);

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

/// One bounded host allocation, and the only thing that can name a byte in it.
///
/// A RAMBlock form is created at device init and held for the VM's lifetime. A
/// packed form normally belongs to one live task mapping and is shared by the
/// resources backed by that mapping; a resource may own one only when no
/// mapping import exists. Both are sliced by checked relative offsets; neither
/// is created per draw/window.
///
/// The backend's handle for the import (a `VkDeviceMemory` and its whole-region
/// `VkBuffer`, or an `MTLBuffer`) lives beside this in the backend, keyed by
/// [`Self::id`]. This type is deliberately backend-free: the bound is pure
/// arithmetic, and keeping it out of a backend-gated module is what lets its
/// target-specific modules otherwise do not run their tests at all.
#[derive(Debug)]
pub struct GuestRamImport {
    id: ImportId,
    /// Set once the guest allocation lifetime ends. The identity is never
    /// reusable, and a stale child holding this object may not recreate a
    /// backend import after retirement.
    retired: std::sync::atomic::AtomicBool,
    /// Guest physical address of the first byte covered, when this is a
    /// RAMBlock import. A packed task-VA alias has no linear GPA coordinate.
    gpa_base: Option<u64>,
    /// Host virtual address of the first byte covered, after any trim. The
    /// address handed to the backend's import call, and never a subrange of it.
    host_base: usize,
    /// Bytes covered, in both address spaces. Always a multiple of `align`.
    len: u64,
    /// The backend's import granularity — `minImportedHostPointerAlignment` on
    align: u64,
}

impl GuestRamImport {
    /// Take a RAMBlock the shim described and bound it to `align`.
    ///
    /// `align` is the backend's queried import granularity, never a guessed
    /// constant: `minImportedHostPointerAlignment` from
    /// `VkPhysicalDeviceExternalMemoryHostPropertiesEXT`, or the host page size
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
            retired: std::sync::atomic::AtomicBool::new(false),
            gpa_base: Some(region.gpa_base + head),
            host_base: host_base as usize,
            len,
            align,
        })
    }

    /// Bound an already-packed, stable host allocation for backend import.
    ///
    /// Unlike [`Self::new`], this allocation has no guest-physical coordinate:
    /// its consecutive host pages may name arbitrary GPAs in one task's virtual
    /// order. It can therefore be sliced only by relative offset, never through
    /// [`Self::slice_for_gpa`]. The host owns the mapping and must keep it live
    /// until the backend device has released every import made from it.
    pub fn new_host_allocation(
        host_base: usize,
        len: u64,
        align: u64,
    ) -> Result<Self, GuestRamError> {
        if align == 0 || !align.is_power_of_two() {
            return Err(GuestRamError::AlignmentNotPowerOfTwo { align }.report());
        }
        if host_base == 0 {
            return Err(GuestRamError::RegionUnmapped.report());
        }
        if len == 0 {
            return Err(GuestRamError::RegionEmpty.report());
        }
        let host = host_base as u64;
        host.checked_add(len)
            .filter(|end| *end <= usize::MAX as u64)
            .ok_or(GuestRamError::RegionWraps { host_va: host, len })
            .map_err(GuestRamError::report)?;
        if !host.is_multiple_of(align) || !len.is_multiple_of(align) {
            return Err(GuestRamError::AlignmentUnsatisfiable { align, len }.report());
        }
        Ok(Self {
            id: ImportId::allocate(),
            retired: std::sync::atomic::AtomicBool::new(false),
            gpa_base: None,
            host_base,
            len,
            align,
        })
    }

    /// This import's identity. Every [`GuestSlice`] it makes carries it.
    pub fn id(&self) -> ImportId {
        self.id
    }

    /// End this allocation identity. Existing backend children may finish,
    /// but no new child or import may be created from it afterward.
    pub fn retire(&self) {
        self.retired
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Whether the allocation identity has ended and may no longer acquire new
    /// backend children.
    pub fn is_retired(&self) -> bool {
        self.retired.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Guest physical address of the first byte covered.
    pub fn gpa_base(&self) -> Option<u64> {
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
        self.gpa_base
            .is_some_and(|base| gpa >= base && gpa - base < self.len)
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
        let Some(gpa_base) = self.gpa_base else {
            return Err(GuestRamError::GpaOutsideImport {
                gpa,
                gpa_base: 0,
                len: self.len,
            }
            .report());
        };
        let outside = GuestRamError::GpaOutsideImport {
            gpa,
            gpa_base,
            len: self.len,
        };
        let offset = gpa
            .checked_sub(gpa_base)
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
/// `reims-vgpu::runtime::guest_ram_map`, which runs on the runtime side and holds
/// no device context. The alternative is threading a number from the backend
/// through every decode path that might name guest memory, which is how a site
/// ends up with a default nobody measured.
///
/// Zero means *no backend has said it can import*, which is the honest answer
/// on a host without the extension and the one the map treats as "run the
/// copying rails". A backend that resolves a negative capability rung must not
/// publish, so the absence of a number is itself the gate.
static GRANULARITY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The largest import the active backend said it could hold, or 0 before any
/// backend has created a device that can import at all.
///
/// Published and withdrawn with [`GRANULARITY`] and never on its own — see
/// [`latch_import_limits`] for why the two are one call.
static IMPORT_BUDGET: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The largest single import the active backend will ask a driver for, or 0
/// before any backend has published. Published and withdrawn with the two above.
static IMPORT_SPAN_MAX: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Publish the import limits a freshly created device resolved to: the
/// granularity every import must meet, and the largest single import the device
/// could hold.
///
/// Called once per device creation, including each recreate, so a rebuilt
/// device republishes rather than leaving the previous one's answer standing. A
/// backend whose capability rung refused must call [`forget_import_limits`]
/// instead — publishing a number from a device that declined the handle type is
/// how the map would build imports nothing can bind.
///
/// **One call for both numbers**, because they have one lifetime and the failure
/// mode of two calls is a budget left standing from the previous device beside a
/// granularity from this one. A caller cannot publish half an answer.
///
/// Refuses a granularity that is not a power of two: every alignment
/// computation here masks with `align - 1`, and a non-power-of-two would make
/// those masks name arbitrary bytes rather than refusing. A zero `budget` is
/// refused with it: a device that can import guest RAM and holds nothing is not
/// a device this rail can run on, and the honest answer is the copying rails.
pub fn latch_import_limits(align: u64, budget: u64, span_max: u64) {
    if align == 0 || !align.is_power_of_two() {
        Emit::decline(EVENT, &GuestRamError::AlignmentNotPowerOfTwo { align }).fail();
        forget_import_limits();
        return;
    }
    if budget == 0 {
        Emit::decline(EVENT, &GuestRamError::ImportBudgetEmpty).fail();
        forget_import_limits();
        return;
    }
    // A span ceiling below the granularity cannot produce a single importable
    // chunk, so it is the same refusal as an unusable alignment wearing a
    // different number. Refused rather than clamped: clamping would import at a
    // size the device's own limits said no to.
    if span_max < align {
        Emit::decline(
            EVENT,
            &GuestRamError::ImportSpanMaxBelowGranularity { span_max, align },
        )
        .fail();
        forget_import_limits();
        return;
    }
    GRANULARITY.store(align, std::sync::atomic::Ordering::Relaxed);
    IMPORT_BUDGET.store(budget, std::sync::atomic::Ordering::Relaxed);
    IMPORT_SPAN_MAX.store(span_max, std::sync::atomic::Ordering::Relaxed);
}

/// Withdraw the published limits: no device can import guest RAM.
pub fn forget_import_limits() {
    GRANULARITY.store(0, std::sync::atomic::Ordering::Relaxed);
    IMPORT_BUDGET.store(0, std::sync::atomic::Ordering::Relaxed);
    IMPORT_SPAN_MAX.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// The largest single import the active backend will ask a driver for, or `None`
/// before any backend has published.
///
/// Distinct from [`import_budget`], which bounds the *sum* of every live import
/// against the roomiest heap. This bounds one `vkAllocateMemory`, and a RAMBlock
/// longer than it is imported in several. The value is derived from the
/// backend's queried allocation and buffer limits.
pub fn import_span_max() -> Option<u64> {
    match IMPORT_SPAN_MAX.load(std::sync::atomic::Ordering::Relaxed) {
        0 => None,
        span => Some(span),
    }
}

/// The largest single import the active backend can hold, or `None` when no
/// backend can import.
///
/// This is a **capacity** bound and not a residency one: a heap large enough to
/// hold an import can still be too full of this device's own images to admit it.
/// The direction it does catch is unambiguous, and it is the one that has been
/// observed killing a guest — an import charged to a heap that could not hold it
/// empty makes every submission referencing it fail validation in the kernel,
/// which surfaces as a lost device rather than as a slow one.
pub fn import_budget() -> Option<u64> {
    match IMPORT_BUDGET.load(std::sync::atomic::Ordering::Relaxed) {
        0 => None,
        budget => Some(budget),
    }
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

/// Why one stable host allocation cannot be imported as a resource backing.
///
/// This is deliberately independent of the whole-RAMBlock map. A packed task
/// or mapper allocation is already the contract-sized object the guest named;
/// whether every other byte of guest RAM fits in the same heap says nothing
/// about whether this allocation does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostAllocationImportRefusal {
    /// No backend published host-pointer import support.
    Unavailable,
    /// This allocation exceeds the largest single allocation the queried API
    /// limits permit.
    SpanTooLarge { len: u64, max: u64 },
    /// No compatible heap can hold this allocation even when empty.
    HeapTooSmall { len: u64, budget: u64 },
}

impl Decline for HostAllocationImportRefusal {
    fn slug(&self) -> &'static str {
        match self {
            Self::Unavailable => "host_allocation_import_unavailable",
            Self::SpanTooLarge { .. } => "host_allocation_import_span_too_large",
            Self::HeapTooSmall { .. } => "host_allocation_import_heap_too_small",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match *self {
            Self::Unavailable => Vec::new(),
            Self::SpanTooLarge { len, max } => {
                vec![("len", len.to_string()), ("max", max.to_string())]
            }
            Self::HeapTooSmall { len, budget } => {
                vec![("len", len.to_string()), ("budget", budget.to_string())]
            }
        }
    }
}

reims_vgpu_observe::decline::decline_display!(HostAllocationImportRefusal);

/// Record a resource-import degradation once per consumer and reason.
///
/// Resource import is optional, so a refusal belongs on the `OFF` channel
/// rather than the failure channel. It still has to be visible: otherwise a
/// host can spend the whole boot copying while the log claims the direct rail
/// is available. The event address distinguishes the small, static set of
/// consumers without inventing a guest-visible policy term.
pub fn report_host_allocation_import_refusal(
    event: &'static str,
    refusal: &HostAllocationImportRefusal,
) {
    if reims_vgpu_observe::first_sight(refusal.slug(), event.as_ptr() as u64) {
        Emit::decline(event, refusal).off();
    }
}

/// Admit one stable host allocation against the active backend's explicit
/// host-pointer limits and return the alignment its owner must preserve.
///
/// The allocation is judged on its own length. In particular, this does not
/// consult `reims-vgpu::runtime::guest_ram_map::standing_refusal`: that answer is
/// about the optional import of every RAMBlock in the VM, while this allocation
/// follows one guest resource's decoded lifetime. Coupling the two disables a
/// legal resource import whenever unrelated RAM makes the whole-VM sum too
/// large.
pub fn host_allocation_import_align(len: u64) -> Result<u64, HostAllocationImportRefusal> {
    let align = granularity().ok_or(HostAllocationImportRefusal::Unavailable)?;
    let span_max = import_span_max().ok_or(HostAllocationImportRefusal::Unavailable)?;
    if len > span_max {
        return Err(HostAllocationImportRefusal::SpanTooLarge { len, max: span_max });
    }
    let budget = import_budget().ok_or(HostAllocationImportRefusal::Unavailable)?;
    if len > budget {
        return Err(HostAllocationImportRefusal::HeapTooSmall { len, budget });
    }
    Ok(align)
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

/// One physically contiguous run within a logical guest-memory window.
#[derive(Clone, Debug)]
pub struct GuestWindowRun {
    /// Byte offset of this run's first byte within the requested window.
    pub window_offset: u64,
    /// Checked, bindable reference for this run's bytes.
    pub guest: GuestRef,
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

    /// Store one guest-visible word through this reference's checked host
    /// mapping.
    ///
    /// Completion handling uses this only after the queue point governing the
    /// reference has completed. Keeping the address derivation here preserves
    /// the module's central invariant: callers never receive a raw host pointer
    /// or reconstruct `host_base + offset` themselves.
    pub fn store_u32_release(&self, value: u32) -> bool {
        if self.requested() < std::mem::size_of::<u32>() as u64 {
            return false;
        }
        let Ok(bound) = self.bound() else {
            return false;
        };
        let Some(byte_offset) = bound.offset.checked_add(self.head()) else {
            return false;
        };
        let Some(address) = self.import.host_base().checked_add(byte_offset as usize) else {
            return false;
        };
        if !address.is_multiple_of(std::mem::align_of::<std::sync::atomic::AtomicU32>()) {
            return false;
        }
        // SAFETY: `GuestRamImport::new` validates the RAMBlock mapping and
        // `bound()` proves this four-byte word is inside it. The Arc held by
        // `GuestRef` keeps the import identity live through the store. The
        // guest polls the aligned word concurrently, so an atomic release store
        // prevents a torn value and orders the write before its interrupt.
        unsafe {
            (address as *const std::sync::atomic::AtomicU32)
                .as_ref()
                .expect("validated guest RAM address")
                .store(value.to_le(), std::sync::atomic::Ordering::Release);
        }
        true
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

    #[test]
    fn canonical_page_sets_answer_exact_overlap() {
        let a = GuestPageSet::new(&[9, 3, 9, 5]).unwrap();
        let same = GuestPageSet::new(&[5, 9, 3]).unwrap();
        let b = GuestPageSet::new(&[7, 5]).unwrap();
        let c = GuestPageSet::new(&[2, 4, 8]).unwrap();

        assert_eq!(a.pages(), &[3, 5, 9]);
        assert_eq!(a, same);
        assert_eq!(
            std::collections::HashSet::from([a.clone(), same]).len(),
            1,
            "canonical equality and the cached hash must remain one map identity"
        );
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
        assert!(!a.overlaps(&c));
    }

    #[test]
    fn target_visible_window_is_the_declared_strided_plane() {
        let backing = GuestTargetBacking {
            allocation_host_ptr: 0x1000,
            allocation_len: 0x8000,
            resource_offset: 0x1000,
            resource_len: 0x4000,
            plane_offset: 0x1200,
            row_pitch: 0x100,
        };
        assert_eq!(backing.visible_window(16, 3, 4), Some(0x1200..0x1440));
    }

    #[test]
    fn image_window_preserves_array_and_volume_pitch() {
        let backing = GuestTargetBacking {
            allocation_host_ptr: 0x1000,
            allocation_len: 0x8000,
            resource_offset: 0x1000,
            resource_len: 0x4000,
            plane_offset: 0x1200,
            row_pitch: 0x100,
        };
        let array = GuestImageLayout::D1Array {
            width: 16,
            layers: 3,
            array_pitch: 0x100,
        };
        assert_eq!(array.visible_span(backing.row_pitch, 4), Some(0x240));
        assert_eq!(backing.visible_image_window(array, 4), Some(0x1200..0x1440));

        let volume = GuestImageLayout::D3 {
            width: 16,
            height: 2,
            depth: 3,
            depth_pitch: 0x200,
        };
        assert_eq!(volume.visible_span(backing.row_pitch, 4), Some(0x540));
        assert_eq!(
            backing.visible_image_window(volume, 4),
            Some(0x1200..0x1740)
        );
    }

    #[test]
    fn a_vulkan_mip_chain_has_one_family_and_the_derived_extent_sequence() {
        let chain = GuestImageAllocationLayout {
            mips: std::sync::Arc::from([
                GuestImageMipLayout {
                    resource_relative_offset: 0x100,
                    row_pitch: 64,
                    layout: GuestImageLayout::D3 {
                        width: 16,
                        height: 8,
                        depth: 4,
                        depth_pitch: 512,
                    },
                },
                GuestImageMipLayout {
                    resource_relative_offset: 0x900,
                    row_pitch: 32,
                    layout: GuestImageLayout::D3 {
                        width: 8,
                        height: 4,
                        depth: 2,
                        depth_pitch: 128,
                    },
                },
            ]),
        };
        assert!(chain.is_vulkan_mip_chain(4));

        let mut wrong_extent = chain.clone();
        std::sync::Arc::make_mut(&mut wrong_extent.mips)[1].layout = GuestImageLayout::D3 {
            width: 8,
            height: 4,
            depth: 3,
            depth_pitch: 128,
        };
        assert!(!wrong_extent.is_vulkan_mip_chain(4));

        let mut wrong_family = chain;
        std::sync::Arc::make_mut(&mut wrong_family.mips)[1].layout = GuestImageLayout::D2 {
            width: 8,
            height: 4,
        };
        assert!(!wrong_family.is_vulkan_mip_chain(4));
    }

    #[test]
    fn image_requirement_key_survives_host_only_padding_but_not_layout_changes() {
        let request = GuestImageBindingRequest {
            backing: GuestTargetBacking {
                allocation_host_ptr: 0x1000,
                allocation_len: 0x4000,
                resource_offset: 0x100,
                resource_len: 0x2000,
                plane_offset: 0x300,
                row_pitch: 0x100,
            },
            allocation: GuestImageAllocationLayout::single(
                0x200,
                0x100,
                GuestImageLayout::D2 {
                    width: 16,
                    height: 16,
                },
            ),
            format: reims_vgpu_protocol::ImageFormat::linear(
                reims_vgpu_protocol::TexelLayout::Bgra8,
            ),
        };
        let padded = GuestImageBindingRequest {
            backing: GuestTargetBacking {
                allocation_host_ptr: 0x9000,
                allocation_len: 0x8000,
                ..request.backing
            },
            ..request.clone()
        };
        assert_eq!(request.key(), padded.key());

        let array = GuestImageBindingRequest {
            allocation: GuestImageAllocationLayout::single(
                0x200,
                0x100,
                GuestImageLayout::D2Array {
                    width: 16,
                    height: 16,
                    layers: 2,
                    array_pitch: 0x1000,
                },
            ),
            ..request.clone()
        };
        assert_ne!(request.key(), array.key());
    }

    #[test]
    fn target_visible_window_refuses_every_out_of_resource_shape() {
        let backing = GuestTargetBacking {
            allocation_host_ptr: 0x1000,
            allocation_len: 0x4000,
            resource_offset: 0x1000,
            resource_len: 0x1000,
            plane_offset: 0x1800,
            row_pitch: 0x100,
        };
        assert_eq!(backing.visible_window(0, 1, 4), None);
        assert_eq!(backing.visible_window(0x41, 1, 4), None);
        assert_eq!(backing.visible_window(0x40, 9, 4), None);
        assert_eq!(
            GuestTargetBacking {
                plane_offset: 0x800,
                ..backing
            }
            .visible_window(1, 1, 4),
            None
        );
    }

    /// A mip chain declares where its subresources sit inside the guest
    /// *resource*, and on this device the parent allocation is a whole
    /// RAMBlock — so on a live boot the resource starts hundreds of megabytes
    /// in. Every mip of such a resource must still be admissible.
    ///
    /// The regression this pins is what happens when the resource-relative
    /// offset is read straight into `plane_offset`: `visible_image_window`'s
    /// first term is `plane_offset >= resource_offset`, so a perfectly valid
    /// chain fails it for every mip whose offset is smaller than the distance
    /// to its own resource — which, at a RAMBlock's scale, is all of them. The
    /// second half of this test is that raw read, and it must stay `None`.
    #[test]
    fn a_mip_is_placed_against_its_resource_not_against_the_allocation() {
        let backing = GuestTargetBacking {
            allocation_host_ptr: 0x1000,
            allocation_len: 0x4000_0000,
            resource_offset: 0x3612_0000,
            resource_len: 0x2000,
            plane_offset: 0x3612_0000,
            row_pitch: 0x100,
        };
        let mips = [
            GuestImageMipLayout {
                resource_relative_offset: 0,
                row_pitch: 0x100,
                layout: GuestImageLayout::D2 {
                    width: 64,
                    height: 16,
                },
            },
            GuestImageMipLayout {
                resource_relative_offset: 0x1000,
                row_pitch: 0x80,
                layout: GuestImageLayout::D2 {
                    width: 32,
                    height: 8,
                },
            },
        ];

        for mip in mips {
            let placed = mip.plane_in(backing).expect("mip places in its resource");
            assert_eq!(
                placed.plane_offset,
                backing.resource_offset + mip.resource_relative_offset
            );
            assert!(
                placed.visible_image_window(mip.layout, 4).is_some(),
                "a declared mip of a resource deep in its allocation must be admissible"
            );
        }

        // The raw read, kept as the thing that must not come back.
        for mip in mips {
            let unplaced = GuestTargetBacking {
                plane_offset: mip.resource_relative_offset,
                row_pitch: mip.row_pitch,
                ..backing
            };
            assert_eq!(unplaced.visible_image_window(mip.layout, 4), None);
        }
    }

    #[test]
    fn a_page_footprint_can_name_only_the_pages_an_image_window_reaches() {
        let footprint = GuestPageFootprint::new(
            std::sync::Arc::from([0x1000, 0x9000, 0x3000, 0x4000]),
            0x1000,
        )
        .expect("footprint");
        assert_eq!(
            footprint.window(0x800..0x2800).expect("window").pages(),
            &[0x1000, 0x9000, 0x3000]
        );
        assert!(footprint.window(0x4000..0x4001).is_none());
        assert!(footprint.window(7..7).is_none());
    }

    #[test]
    fn target_footprint_translates_from_parent_import_to_resource_coordinates() {
        let import = std::sync::Arc::new(
            GuestRamImport::new_host_allocation(0x1000_0000, 0x8000, 0x1000)
                .expect("aligned import"),
        );
        let memory = GuestTargetMemory {
            backing: GuestTargetBacking {
                allocation_host_ptr: import.host_base(),
                allocation_len: import.len(),
                resource_offset: 0x3000,
                resource_len: 0x3000,
                plane_offset: 0x3800,
                row_pitch: 0x800,
            },
            import,
            footprint: GuestPageFootprint::new(
                std::sync::Arc::from([0xa000, 0xb000, 0xc000]),
                0x1000,
            )
            .expect("resource footprint"),
        };
        assert_eq!(
            memory
                .visible_footprint(64, 2, 4)
                .expect("the visible window lies in resource pages")
                .pages(),
            &[0xa000, 0xb000],
            "the 0x3000 parent-import head must not index three pages past the resource"
        );
    }

    #[test]
    fn target_footprint_retains_the_resources_offset_inside_its_first_page() {
        let import = std::sync::Arc::new(
            GuestRamImport::new_host_allocation(0x1000_0000, 0x8000, 0x1000)
                .expect("aligned import"),
        );
        let memory = GuestTargetMemory {
            backing: GuestTargetBacking {
                allocation_host_ptr: import.host_base(),
                allocation_len: import.len(),
                resource_offset: 0x3800,
                resource_len: 0x1800,
                plane_offset: 0x3800,
                row_pitch: 0x900,
            },
            import,
            footprint: GuestPageFootprint::new(std::sync::Arc::from([0xa000, 0xb000]), 0x1000)
                .expect("resource footprint"),
        };
        assert_eq!(
            memory
                .visible_footprint(64, 2, 4)
                .expect("the second row crosses into the resource's second page")
                .pages(),
            &[0xa000, 0xb000]
        );
    }

    #[test]
    fn a_target_write_page_set_is_canonicalized_once_for_submission_ownership() {
        let import = std::sync::Arc::new(
            GuestRamImport::new_host_allocation(0x1000_0000, 0x4000, 0x1000)
                .expect("aligned import"),
        );
        let memory = GuestTargetMemory {
            backing: GuestTargetBacking {
                allocation_host_ptr: import.host_base(),
                allocation_len: import.len(),
                resource_offset: 0,
                resource_len: 0x3000,
                plane_offset: 0,
                row_pitch: 0x1000,
            },
            import,
            footprint: GuestPageFootprint::new(
                std::sync::Arc::from([0xc000, 0xa000, 0xc000]),
                0x1000,
            )
            .expect("footprint"),
        };
        assert_eq!(
            memory
                .visible_write_pages(1, 3, 4)
                .expect("visible pages")
                .pages(),
            &[0xa000, 0xc000],
            "the immutable ledger value is sorted and deduplicated"
        );
    }

    /// A three-level chain the guest laid out smallest-first, so level zero —
    /// the first entry of `mips` — sits at the *far* end of the allocation.
    fn smallest_first_chain() -> GuestImageAllocationLayout {
        GuestImageAllocationLayout {
            mips: std::sync::Arc::from([
                GuestImageMipLayout {
                    resource_relative_offset: 0x1000,
                    row_pitch: 0x40,
                    layout: GuestImageLayout::D2 {
                        width: 16,
                        height: 16,
                    },
                },
                GuestImageMipLayout {
                    resource_relative_offset: 0x100,
                    row_pitch: 0x20,
                    layout: GuestImageLayout::D2 {
                        width: 8,
                        height: 8,
                    },
                },
                GuestImageMipLayout {
                    resource_relative_offset: 0,
                    row_pitch: 0x10,
                    layout: GuestImageLayout::D2 {
                        width: 4,
                        height: 4,
                    },
                },
            ]),
        }
    }

    #[test]
    fn a_chain_window_unions_every_level_whatever_order_the_guest_placed_them_in() {
        let backing = GuestTargetBacking {
            allocation_host_ptr: 0x1000_0000,
            allocation_len: 0x4000,
            resource_offset: 0,
            resource_len: 0x4000,
            plane_offset: 0,
            row_pitch: 0x40,
        };
        let chain = smallest_first_chain();

        assert_eq!(
            chain
                .visible_chain_window(backing, 4)
                .expect("chain window"),
            0..0x1400,
            "the union runs from the smallest level's start to level zero's end"
        );

        // The trap the union exists to avoid: level zero alone names only the
        // far end, and first.start..last.end would run backwards.
        let base = chain.base().expect("a chain has a level zero");
        assert_eq!(
            base.plane_in(backing)
                .expect("level zero is placed")
                .visible_image_window(base.layout, 4)
                .expect("level zero window"),
            0x1000..0x1400
        );
    }

    #[test]
    fn a_level_that_escapes_the_resource_refuses_the_whole_chain() {
        let backing = GuestTargetBacking {
            allocation_host_ptr: 0x1000_0000,
            allocation_len: 0x4000,
            resource_offset: 0,
            // Level zero ends at 0x1400, one byte past this resource.
            resource_len: 0x13ff,
            plane_offset: 0,
            row_pitch: 0x40,
        };
        assert_eq!(
            smallest_first_chain().visible_chain_window(backing, 4),
            None
        );
    }

    #[test]
    fn chain_write_pages_owns_every_page_any_level_touches() {
        let import = std::sync::Arc::new(
            GuestRamImport::new_host_allocation(0x1000_0000, 0x4000, 0x1000)
                .expect("aligned import"),
        );
        let memory = GuestTargetMemory {
            backing: GuestTargetBacking {
                allocation_host_ptr: import.host_base(),
                allocation_len: import.len(),
                resource_offset: 0,
                resource_len: 0x4000,
                plane_offset: 0,
                row_pitch: 0x40,
            },
            import,
            footprint: GuestPageFootprint::new(
                std::sync::Arc::from([0xa000, 0xb000, 0xc000, 0xd000]),
                0x1000,
            )
            .expect("resource footprint"),
        };
        let chain = smallest_first_chain();

        assert_eq!(
            memory
                .chain_write_pages(&chain, 4)
                .expect("chain pages")
                .pages(),
            &[0xa000, 0xb000],
            "the smaller levels' page must be owned alongside level zero's"
        );
        assert_eq!(
            memory
                .visible_write_pages(16, 16, 4)
                .expect("level-zero-shaped pages")
                .pages(),
            &[0xa000],
            "level zero read against the backing's own plane_offset misses the tail levels"
        );
    }

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

    fn window_run(
        import: &std::sync::Arc<GuestRamImport>,
        window_offset: u64,
        import_offset: u64,
        len: u64,
    ) -> GuestWindowRun {
        let slice = import
            .slice(import_offset, len)
            .expect("bounded test slice");
        GuestWindowRun {
            window_offset,
            guest: GuestRef::new(std::sync::Arc::clone(import), slice)
                .expect("slice belongs to import"),
        }
    }

    #[test]
    fn a_guest_run_source_exposes_one_bounded_direct_stretch() {
        let import = std::sync::Arc::new(import(0x4000, 0x1000));
        let source = GuestRunSource {
            runs: std::sync::Arc::new(vec![GuestRun {
                host_ptr: 0x7f00_0000_0000,
                len: 0x2000,
            }]),
            source_offset: 0x180,
            total_len: 0x800,
            row_length_texels: 0,
            pages: Some(std::sync::Arc::new(vec![window_run(&import, 0, 0, 0x2000)])),
            physical_pages: None,
        };

        let stretch = source.single_stretch().expect("one checked stretch");
        assert_eq!(
            (stretch.skip, stretch.window_offset, stretch.len),
            (0x180, 0, 0x800)
        );
    }

    #[test]
    fn a_guest_run_source_clips_each_scattered_stretch_to_its_window() {
        let import = std::sync::Arc::new(import(0x4000, 0x1000));
        let source = GuestRunSource {
            runs: std::sync::Arc::new(Vec::new()),
            source_offset: 0x800,
            total_len: 0x1800,
            row_length_texels: 0,
            pages: Some(std::sync::Arc::new(vec![
                window_run(&import, 0, 0, 0x1000),
                window_run(&import, 0x1000, 0x2000, 0x1000),
                window_run(&import, 0x2000, 0x3000, 0x1000),
            ])),
            physical_pages: None,
        };

        assert!(source.single_stretch().is_none());
        let stretches: Vec<_> = source
            .window_stretches()
            .expect("checked pages exist")
            .map(|stretch| (stretch.skip, stretch.window_offset, stretch.len))
            .collect();
        assert_eq!(stretches, vec![(0x800, 0, 0x800), (0, 0x800, 0x1000)]);
        let plan = source.transfer_plan();
        assert!(plan.direct().is_none());
        assert_eq!(
            plan.stretches()
                .expect("the checked pages tile the window")
                .map(|stretch| stretch.len)
                .sum::<u64>(),
            source.total_len
        );
    }

    #[test]
    fn a_read_transfer_plan_refuses_partial_checked_coverage_to_the_gpu() {
        let import = std::sync::Arc::new(import(0x4000, 0x1000));
        let source = GuestRunSource {
            runs: std::sync::Arc::new(vec![GuestRun {
                host_ptr: 0x7f00_0000_0000,
                len: 0x1800,
            }]),
            source_offset: 0,
            total_len: 0x1800,
            row_length_texels: 0,
            pages: Some(std::sync::Arc::new(vec![window_run(&import, 0, 0, 0x1000)])),
            physical_pages: None,
        };

        assert!(matches!(
            source.transfer_plan(),
            GuestReadTransferPlan::CpuOnly
        ));
    }

    #[test]
    fn a_guest_page_target_derives_pitch_extent_and_coverage_from_its_layout() {
        let import = std::sync::Arc::new(import(0x4000, 0x1000));
        let target = GuestPageTarget {
            runs: vec![window_run(&import, 0, 0, 0x40)],
            row_length_texels: 8,
            width: 4,
            height: 2,
            format: reims_vgpu_protocol::StorageImageFormat::Bgra8Unorm,
        };

        assert_eq!(target.pitch_bytes(), 32);
        assert_eq!(target.extent_end(), 48);
        assert_eq!(target.window_bytes(), 0x40);
        assert!(!target.rows_are_dense());
        assert_eq!(target.geometry().pitch_bytes, 32);
        assert!(matches!(
            target.transfer_plan(),
            GuestPageTransferPlan::PitchedRectangles { geometry }
                if geometry.pitch_bytes == 32
        ));

        let dense = GuestPageTarget {
            row_length_texels: 4,
            ..target
        };
        assert!(matches!(
            dense.transfer_plan(),
            GuestPageTransferPlan::DenseScatter { window_bytes: 0x40 }
        ));
    }

    #[test]
    fn a_guest_target_seed_derives_its_load_window_from_the_plane_contract() {
        let import = std::sync::Arc::new(
            GuestRamImport::new_host_allocation(0x1000_0000, 0x4000, 0x1000)
                .expect("aligned import"),
        );
        let memory = GuestTargetMemory {
            backing: GuestTargetBacking {
                allocation_host_ptr: import.host_base(),
                allocation_len: import.len(),
                resource_offset: 0x1000,
                resource_len: 0x2000,
                plane_offset: 0x1200,
                row_pitch: 32,
            },
            import,
            footprint: GuestPageFootprint::new(std::sync::Arc::from([0x5000, 0x6000]), 0x1000)
                .expect("footprint"),
        };

        let seed = guest_target_seed(&memory, 4, 2, reims_vgpu_protocol::TexelLayout::Rgba8)
            .expect("the plane contains two padded rows");
        assert_eq!(seed.source.total_len, 48, "one stride plus the final row");
        assert_eq!(seed.source.row_length_texels, 8);
        assert_eq!(seed.source.runs[0].host_ptr, 0x1000_1200);
        assert_eq!(seed.source.pages.as_ref().unwrap().len(), 1);

        let mut outside = memory.clone();
        outside.backing.plane_offset = 0x2ff8;
        assert!(
            guest_target_seed(&outside, 4, 2, reims_vgpu_protocol::TexelLayout::Rgba8,).is_none(),
            "a plane extending beyond its resource is not widened into its neighbour"
        );
    }

    #[test]
    fn a_page_footprint_derives_physical_runs_once_and_windows_them_exactly() {
        let pages: std::sync::Arc<[u64]> = [0x1000, 0x2000, 0x9000, 0xa000, 0xb000].into();
        let footprint = GuestPageFootprint::new(pages, 0x1000).expect("valid footprint");
        assert_eq!(footprint.runs(), &[0..2, 2..5]);

        let mut visited = Vec::new();
        footprint.visit_window(0x1800, 0x3000, |gpa, len| visited.push((gpa, len)));
        assert_eq!(
            visited,
            vec![(0x2800, 0x800), (0x9000, 0x2800)],
            "the allocation window follows physical runs without filling their gap"
        );
    }

    #[test]
    fn a_page_footprint_rejects_an_empty_or_non_page_geometry() {
        assert!(GuestPageFootprint::new(std::sync::Arc::from([]), 0x1000).is_none());
        assert!(GuestPageFootprint::new(std::sync::Arc::from([0x1000]), 0).is_none());
        assert!(GuestPageFootprint::new(std::sync::Arc::from([0x1000]), 0x1800).is_none());
    }

    #[test]
    fn retirement_is_monotonic_on_the_allocation_identity() {
        let import = import(0x4000, 0x1000);
        let id = import.id();
        assert!(!import.is_retired());
        import.retire();
        import.retire();
        assert!(import.is_retired());
        assert_eq!(import.id(), id, "retirement never creates a new identity");
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
        let capture = reims_vgpu_observe::FailCapture::start();
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
        assert_eq!(
            import.resolve(&slice),
            Ok(BoundRange {
                offset: 0,
                len: 0x1000
            })
        );
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
        assert_eq!(import.gpa_base(), Some(region.gpa_base + 0x800));
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
        let gpa_base = import.gpa_base().expect("RAMBlock coordinate");
        assert!(import.slice_for_gpa(gpa_base, 4).is_ok());
        assert!(import.contains_gpa(gpa_base));
        assert!(!import.contains_gpa(region.gpa_base));
    }

    /// A packed task-address alias is bounded like RAM, but deliberately has no
    /// GPA interpretation: adjacent bytes may come from unrelated guest frames.
    #[test]
    fn a_packed_host_allocation_slices_only_by_relative_offset() {
        let import = GuestRamImport::new_host_allocation(0x7f00_0000_0000, 0x4000, 0x1000)
            .expect("aligned stable allocation");
        assert_eq!(import.gpa_base(), None);
        let slice = import.slice(0x1800, 0x800).expect("inside allocation");
        assert_eq!(
            import.resolve(&slice),
            Ok(BoundRange {
                offset: 0x1000,
                len: 0x1000,
            })
        );
        assert!(matches!(
            import.slice_for_gpa(0x1800, 0x800),
            Err(GuestRamError::GpaOutsideImport { .. })
        ));
    }

    /// Resource admission is the exact three-term host-pointer contract: the
    /// backend published support, the allocation fits one API allocation, and
    /// a compatible heap can hold it. Each failed term keeps its own type.
    #[test]
    fn a_host_allocation_is_admitted_by_its_own_queried_limits() {
        const ALIGN: u64 = 0x1000;
        const SPAN_MAX: u64 = 0x20_0000;
        const BUDGET: u64 = 0x80_0000;

        forget_import_limits();
        assert_eq!(
            host_allocation_import_align(ALIGN),
            Err(HostAllocationImportRefusal::Unavailable)
        );

        latch_import_limits(ALIGN, BUDGET, SPAN_MAX);
        assert_eq!(host_allocation_import_align(SPAN_MAX), Ok(ALIGN));
        assert_eq!(
            host_allocation_import_align(SPAN_MAX + ALIGN),
            Err(HostAllocationImportRefusal::SpanTooLarge {
                len: SPAN_MAX + ALIGN,
                max: SPAN_MAX,
            })
        );

        latch_import_limits(ALIGN, SPAN_MAX, BUDGET);
        assert_eq!(
            host_allocation_import_align(SPAN_MAX + ALIGN),
            Err(HostAllocationImportRefusal::HeapTooSmall {
                len: SPAN_MAX + ALIGN,
                budget: SPAN_MAX,
            })
        );
        forget_import_limits();
    }

    /// A direct-import refusal degrades to copying, but the degradation remains
    /// visible and does not flood when the same consumer retries it.
    #[test]
    fn a_host_allocation_refusal_is_reported_once_per_consumer() {
        let capture = reims_vgpu_observe::FailCapture::start();
        let refusal = HostAllocationImportRefusal::SpanTooLarge {
            len: 0x30_0000,
            max: 0x20_0000,
        };
        report_host_allocation_import_refusal("test_alias_import", &refusal);
        report_host_allocation_import_refusal("test_alias_import", &refusal);
        assert_eq!(
            capture.lines(),
            vec![
                "OFF test_alias_import reason=host_allocation_import_span_too_large len=3145728 max=2097152"
            ]
        );
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

    #[test]
    fn a_guest_reference_stores_only_inside_its_checked_word() {
        let mut words = [0u32; 4];
        let import = std::sync::Arc::new(
            GuestRamImport::new_host_allocation(
                words.as_mut_ptr() as usize,
                std::mem::size_of_val(&words) as u64,
                std::mem::align_of_val(&words) as u64,
            )
            .expect("aligned test allocation"),
        );
        let slice = import
            .slice(
                std::mem::size_of::<u32>() as u64,
                std::mem::size_of::<u32>() as u64,
            )
            .expect("second word");
        let guest = GuestRef::new(import, slice).expect("matching import");

        assert!(guest.store_u32_release(0x1234_5678));
        assert_eq!(words, [0, 0x1234_5678u32.to_le(), 0, 0]);
    }

    /// One slug per check. Two checks sharing a slug is the defect the decline
    /// vocabulary exists to prevent: you watch it fire and still cannot tell
    /// which bound refused.
    #[test]
    fn every_refusal_has_its_own_slug() {
        let all = [
            GuestRamError::RegionEmpty,
            GuestRamError::RegionUnmapped,
            GuestRamError::RegionWraps { host_va: 0, len: 0 },
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
            GuestRamError::SliceForeignImport {
                slice: 0,
                import: 0,
            },
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

/// [`GuestPageSet::overlaps`] answers the same question as a plain merge.
///
/// The rewrite that made it a search is a pure algorithmic change with nothing
/// guest-visible in it, so no `conformance/` case can express the difference --
/// a passing battery is exactly what a correct rewrite and the code it replaced
/// both produce. What gates it instead is this differential: a naive merge,
/// written from the definition rather than from the implementation, run against
/// the real one over sets shaped like the ones the device asks about. A skip
/// that overshoots or a range check with the wrong comparison is a *silent*
/// stale bind -- a content defect -- so the check has to be an oracle and not
/// an assertion about the code's own reasoning.
#[cfg(test)]
mod page_set_overlap {
    use super::GuestPageSet;

    /// Overlap straight from the definition: the sets share at least one page.
    fn shares_a_page(left: &[u64], right: &[u64]) -> bool {
        left.iter().any(|page| right.contains(page))
    }

    /// A deterministic 64-bit stream, so a failure is reproducible from its
    /// seed and the suite needs no dependency to be random.
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn overlaps_agrees_with_a_naive_merge_on_every_shape() {
        let mut state = 0x5eed_1234_9abc_def1_u64;
        // The shapes that matter: a few pages against thousands is the sampled
        // texture against a fullscreen target, and the small spread is where
        // interleaving actually happens.
        for &(left_len, right_len, spread) in &[
            (1_usize, 2025_usize, 4096_u64),
            (2025, 1, 4096),
            (8, 2025, 64),
            (2025, 2025, 4096),
            (1, 1, 2),
            (17, 23, 32),
            (3, 3, 100_000),
        ] {
            for _ in 0..400 {
                let mut make = |len: usize| {
                    let pages: Vec<u64> = (0..len).map(|_| next(&mut state) % spread).collect();
                    pages
                };
                let left_pages = make(left_len);
                let right_pages = make(right_len);
                let (Some(left), Some(right)) = (
                    GuestPageSet::new(&left_pages),
                    GuestPageSet::new(&right_pages),
                ) else {
                    continue;
                };
                let expected = shares_a_page(&left_pages, &right_pages);
                assert_eq!(
                    left.overlaps(&right),
                    expected,
                    "left={:?} right={:?}",
                    left.pages(),
                    right.pages()
                );
                // The question is symmetric and the implementation is not, so
                // both directions are separate evidence.
                assert_eq!(
                    right.overlaps(&left),
                    expected,
                    "reversed left={:?} right={:?}",
                    left.pages(),
                    right.pages()
                );
            }
        }
    }

    #[test]
    fn disjoint_ranges_and_touching_ends_are_both_answered() {
        let low = GuestPageSet::new(&[1, 2, 3]).expect("non-empty");
        let high = GuestPageSet::new(&[9, 10, 11]).expect("non-empty");
        assert!(!low.overlaps(&high));
        assert!(!high.overlaps(&low));

        // The two ranges meet at exactly one page, which is the case a range
        // check with `>=` instead of `>` would drop.
        let touching = GuestPageSet::new(&[3, 40, 41]).expect("non-empty");
        assert!(low.overlaps(&touching));
        assert!(touching.overlaps(&low));

        // One set entirely inside a gap of the other: sorted ranges do not
        // separate them, so only the skip loop can answer.
        let straddling = GuestPageSet::new(&[0, 100]).expect("non-empty");
        let inside = GuestPageSet::new(&[50]).expect("non-empty");
        assert!(!straddling.overlaps(&inside));
        assert!(!inside.overlaps(&straddling));
        assert!(straddling.overlaps(&GuestPageSet::new(&[50, 100]).expect("non-empty")));
    }
}
