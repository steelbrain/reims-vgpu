//! Linear resource views over retained task mappings.
//!
//! # The shape this follows
//!
//! Apple's host resolves a guest object reference to a host buffer **once**,
//! when the object is created, and stores it on the task under that reference.
//! Its render decoder then reads a `{u32 reference, u64 offset}` record per
//! bound slot, asks the task for the buffer by reference, and hands Metal the
//! buffer and the offset. No address translation happens on the draw path at
//! all — the page-run computation on that side is reachable only from the
//! map/unmap handlers, never from a decoder.
//!
//! This device resolved the bind instead: every bound buffer of every draw
//! walked the task page table over the bound span, coalesced the GPA-contiguous
//! stretches, and asked the host to alias each one. That is the same answer
//! every time until the guest changes a mapping, and the guest changes mappings
//! about four orders of magnitude less often than it draws.
//!
//! MapMemory2 owns the normal packed allocation. This registry holds resource
//! views over it, and retains an exact-window fallback only when the mapping
//! allocation was unavailable.
//!
//! Linear textures use the same packed allocation. Their level offset and row
//! pitch become image coordinates over it, while buffer records carry their
//! offset to the bind. Both are views of one resource-owned mapping; neither
//! needs a second task-page walk after that mapping is retained.
//!
//! # What a held resolution is, and is not
//!
//! It is an **address** resolution: which host spans back this reference's
//! bytes right now. It is not the bytes. The runs point into this process's
//! import of guest RAM and the GPU reads them when the command buffer executes,
//! so a guest CPU write to those pages is picked up with nothing invalidated —
//! the same property the walking rail had, and the reason `CmdInvalidateResources`
//! and the exec resource table's validity quad do not appear anywhere here.
//! Content invalidation is not this module's business.
//!
//! Only an **address** change matters, and the guest announces every one of
//! them:
//!
//! * `CmdMapMemory2` / `CmdUnmapMemory` — the guest mutates the task page table
//!   and then notifies, carrying the exact `(task, gva, length)` that moved.
//!   Retired by range.
//! * `CmdReplacePhysical` — a GPA behind a GVA changed.
//! * `CmdSetObjectList` / `CmdDeleteObject` — a reference now names something
//!   else, or nothing.
//! * `CmdDefineTask2` / `CmdDeleteTask` — the page table root changed or the
//!   task is gone.
//!
//! `CmdReplacePhysical` and `CmdDeleteObject` carry the task-local resource
//! reference and retire that reference. `CmdSetObjectList`, `CmdDefineTask2`
//! and `CmdDeleteTask` replace task-wide naming state and retire the whole task.
//!
//! # Why the fallback key carries the offset
//!
//! Apple keys purely by reference, because their buffer covers the whole
//! allocation and the offset rides to Metal beside it. An exact-window fallback
//! resolution here covers `[gva + offset, gva + size)` — the span the bind
//! actually asked for — so two binds of one reference at different offsets are
//! two resolutions.
//!
//! The packed-alias rail resolves the whole allocation once and supplies the
//! offset beside that retained source. It bypasses this map entirely, which is
//! what keeps resource-shaped state resource-shaped. It is an optional answer
//! beside the narrower fallback: if an unmapped tail prevents the whole
//! allocation from being reconstructed, the exact offset/cap window still
//! resolves here and gathers.
//!
//! The distinction is measured rather than aesthetic. Before packed resources
//! bypassed this map, one driven x86 window-drag run reached 33,828 fallback
//! entries over 48 `(task, reference)` pairs, with one reference accounting for
//! 3,080 offsets. The same workload with buffer-plus-offset binding held zero
//! fallback entries while the packed resources remained live. The offset is
//! therefore required for correctness only on the exact-window fallback; it is
//! not a sound identity for the normal resource registry.
//!
//! # Why the key also carries the shader's extent cap
//!
//! The three fields above all describe the *bind*. The fourth describes the
//! **shader**: how far reflection proved this draw's shader can read into the
//! buffer, which is what lets the resolution cover less than the rest of the
//! allocation. A resolution walked under a narrow cap covers fewer bytes than a
//! shader with a wider one needs, and serving it across would hand the GPU a
//! short buffer — wrong pixels, no error. So the cap is part of the identity of
//! the resolution rather than a property of it, and [`Key`] says the rest.
//!
//! # No capacity
//!
//! There is no cap and no eviction. The fallback population is one entry per
//! live `(task, reference, offset, extent cap)` whose whole resource could not
//! be reconstructed; the normal population is one packed entry per
//! `(task, reference)`. Every entry leaves through one of the retirement rules
//! above or through [`BoundBuffers::clear`] at device reset. A capacity here
//! would be a second, invisible reason for a resolution to disappear, and the
//! miss it caused would read as a mapping change that never happened.
//!
//! The extent cap widens that population only where two shaders declare
//! different extents over one bind; the retirement rules are all keyed on task
//! and reference and so are indifferent to it.

use std::sync::Arc;

use crate::runtime::guest_ram_map::GuestWindowRun;
use reims_vgpu_memory::GuestRun;

/// One task buffer view over a stable, contiguous host allocation.
///
/// The allocation follows the buffer's task-virtual byte order even when its
/// guest-physical pages are scattered. Every offset bind can therefore slice
/// this one checked import instead of gathering the same pages into scratch.
#[derive(Clone, Debug)]
pub struct PackedBuffer {
    pub gva: u64,
    pub size: u64,
    /// Offset of `gva` inside the page-aligned host allocation.
    pub head: u64,
    pub import: Arc<crate::runtime::guest_ram::GuestRamImport>,
    /// Physical page list behind the packed alias, retained so resource views
    /// can build witnesses without walking the task page table again.
    pub gpas: Arc<Vec<u64>>,
    /// Physical ownership of the complete declared allocation, derived with
    /// the alias and retained for Store publication.
    pub footprint: crate::runtime::guest_ram::GuestPageFootprint,
    /// Persistent whole-buffer sources shared by every offset bind.
    pub runs: Arc<Vec<GuestRun>>,
    pub pages: Arc<Vec<GuestWindowRun>>,
    /// Backend-reported allocation extents for the sampled image views of this
    /// resource. Entries share the resource lifetime; distinct dimensions,
    /// formats, levels, and pitches remain distinct construction answers.
    pub sampled_image_requirements: std::collections::HashMap<
        reims_vgpu_memory::GuestImageBindingKey,
        reims_vgpu_memory::GuestImageBindingDisposition,
    >,
    /// Allocation owned by `map_pages` when this resource required a packed
    /// alias. RAMBlock-backed resources borrow the VM-wide import and carry no
    /// per-resource retirement.
    pub(crate) owned_alias: Option<PackedAliasRetirement>,
}

/// Immutable execution payload borrowed from one retained packed resource.
///
/// Admission results and alias-retirement ownership stay in [`PackedBuffer`].
/// A warm bind needs only these shared handles and scalar coordinates, so
/// taking a payload does not clone the resource's image-admission map.
#[derive(Clone, Debug)]
pub struct PackedBufferAccess {
    pub gva: u64,
    pub size: u64,
    pub head: u64,
    pub import: Arc<crate::runtime::guest_ram::GuestRamImport>,
    pub gpas: Arc<Vec<u64>>,
    pub footprint: crate::runtime::guest_ram::GuestPageFootprint,
    pub runs: Arc<Vec<GuestRun>>,
    pub pages: Arc<Vec<GuestWindowRun>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PackedAliasRetirement {
    pub import: reims_vgpu_memory::ImportId,
    pub ptr: usize,
    pub len: usize,
}

/// One allocation-relative byte window expressed in both host-import and
/// guest-physical coordinates.
///
/// `host_ptr` includes [`PackedBuffer::head`], which is the resource's offset
/// inside its retained host import. `gpas` is indexed by the resource GVA's
/// offset inside its first guest page. Those offsets agree only when the host
/// import begins at that guest page; a RAM-block import may begin much earlier.
pub struct PackedWitnessWindow<'a> {
    pub host_ptr: usize,
    pub gpas: &'a [u64],
}

impl PackedBuffer {
    pub fn access(&self) -> PackedBufferAccess {
        PackedBufferAccess {
            gva: self.gva,
            size: self.size,
            head: self.head,
            import: Arc::clone(&self.import),
            gpas: Arc::clone(&self.gpas),
            footprint: self.footprint.clone(),
            runs: Arc::clone(&self.runs),
            pages: Arc::clone(&self.pages),
        }
    }

    /// Resolve one allocation-relative byte window for content witnessing.
    pub fn witness_window(&self, offset: u64, span: u64) -> Option<PackedWitnessWindow<'_>> {
        if span == 0 || offset.checked_add(span)? > self.size {
            return None;
        }
        let page = self.footprint.page_size();
        let guest_start = (self.gva % page).checked_add(offset)?;
        let guest_end = guest_start.checked_add(span)?;
        let first = usize::try_from(guest_start / page).ok()?;
        let last = usize::try_from((guest_end - 1) / page).ok()?;
        let import_offset = self.head.checked_add(offset)?;
        Some(PackedWitnessWindow {
            host_ptr: self
                .import
                .host_base()
                .checked_add(usize::try_from(import_offset).ok()?)?,
            gpas: self.gpas.get(first..=last)?,
        })
    }

    /// One buffer bind inside this retained allocation. Compute and render
    /// consume the same source type; the zero row length says this is a flat
    /// byte range rather than a pitched image plane.
    pub fn buffer_source(
        &self,
        offset: u64,
        span: u64,
    ) -> Option<reims_vgpu_memory::GuestRunSource> {
        self.texel_source(offset, span, 0)
    }

    /// Physical pages touched by one allocation-relative byte window.
    pub fn window_pages(&self, offset: u64, span: u64) -> Option<std::collections::HashSet<u64>> {
        Some(
            self.witness_window(offset, span)?
                .gpas
                .iter()
                .copied()
                .collect(),
        )
    }

    /// One image plane inside this retained allocation, expressed in the form
    /// the Vulkan engine consumes. The plane borrows the allocation's one set
    /// of runs and bounded imports; only its decoded offset, extent and row
    /// stride vary per texture level.
    pub fn texel_source(
        &self,
        offset: u64,
        span: u64,
        row_length_texels: u32,
    ) -> Option<reims_vgpu_memory::GuestRunSource> {
        offset.checked_add(span).filter(|&end| end <= self.size)?;
        let physical_pages =
            reims_vgpu_memory::GuestPageSet::new(self.witness_window(offset, span)?.gpas);
        Some(reims_vgpu_memory::GuestRunSource {
            runs: Arc::clone(&self.runs),
            source_offset: offset,
            total_len: span,
            row_length_texels,
            pages: Some(Arc::clone(&self.pages)),
            physical_pages,
        })
    }
}

impl PackedBufferAccess {
    /// Resolve one allocation-relative byte window for content witnessing.
    pub fn witness_window(&self, offset: u64, span: u64) -> Option<PackedWitnessWindow<'_>> {
        if span == 0 || offset.checked_add(span)? > self.size {
            return None;
        }
        let page = self.footprint.page_size();
        let guest_start = (self.gva % page).checked_add(offset)?;
        let guest_end = guest_start.checked_add(span)?;
        let first = usize::try_from(guest_start / page).ok()?;
        let last = usize::try_from((guest_end - 1) / page).ok()?;
        let import_offset = self.head.checked_add(offset)?;
        Some(PackedWitnessWindow {
            host_ptr: self
                .import
                .host_base()
                .checked_add(usize::try_from(import_offset).ok()?)?,
            gpas: self.gpas.get(first..=last)?,
        })
    }

    pub fn texel_source(
        &self,
        offset: u64,
        span: u64,
        row_length_texels: u32,
    ) -> Option<reims_vgpu_memory::GuestRunSource> {
        offset.checked_add(span).filter(|&end| end <= self.size)?;
        let physical_pages =
            reims_vgpu_memory::GuestPageSet::new(self.witness_window(offset, span)?.gpas);
        Some(reims_vgpu_memory::GuestRunSource {
            runs: Arc::clone(&self.runs),
            source_offset: offset,
            total_len: span,
            row_length_texels,
            pages: Some(Arc::clone(&self.pages)),
            physical_pages,
        })
    }
}

/// Consumer that asked the resource registry to resolve one allocation. This
/// selects observability only; every arm creates the same retained allocation.
#[derive(Clone, Copy)]
pub enum PackedResourceUse {
    Buffer,
    LinearSample,
    LinearTarget,
    ComputeTexture,
}

fn stable_guest_alias_available<H: crate::runtime::host::HostOps>(host: &H) -> bool {
    if host.map_pages_stable() {
        return true;
    }
    static NOTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !NOTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        crate::observe::fail(String::from(
            "guest_run_rail off reason=host_page_alias_not_stable \
             (resource binds take the CPU byte loader)",
        ));
    }
    false
}

pub(crate) fn packed_scatter_band(gpas: &[u64], page: u64) -> &'static str {
    match reims_vgpu_paging::runs::contig_run_count(gpas, page) {
        0 | 1 => "zc_packed_scatter_runs_1",
        2 => "zc_packed_scatter_runs_2",
        3..=4 => "zc_packed_scatter_runs_3_4",
        5..=8 => "zc_packed_scatter_runs_5_8",
        9..=16 => "zc_packed_scatter_runs_9_16",
        17..=64 => "zc_packed_scatter_runs_17_64",
        _ => "zc_packed_scatter_runs_65_up",
    }
}

/// Resolve and retain the complete allocation named by one task-local resource.
/// Buffer offsets and texture planes are views over this object; neither walks
/// the task page table again while the guest keeps the mapping alive.
pub fn ensure_packed_resource<
    M: crate::runtime::host::HostMemory + crate::runtime::host::HostOps,
>(
    state: &mut crate::runtime::Device,
    host: &mut M,
    task_id: u32,
    resource_ref: u32,
    gva: u64,
    size: u64,
    usage: PackedResourceUse,
) -> bool {
    ensure_packed_resource_with_extent(state, host, task_id, resource_ref, gva, size, usage, None)
}

/// Resolve a resource-shaped alias whose host allocation reaches an exact
/// backend-reported image binding extent.
pub struct PackedImageBinding {
    pub task_id: u32,
    pub resource_ref: u32,
    pub gva: u64,
    pub size: u64,
    pub required_import_len: u64,
    pub usage: PackedResourceUse,
}

pub fn ensure_packed_resource_for_image<
    M: crate::runtime::host::HostMemory + crate::runtime::host::HostOps,
>(
    state: &mut crate::runtime::Device,
    host: &mut M,
    request: PackedImageBinding,
) -> bool {
    ensure_packed_resource_with_extent(
        state,
        host,
        request.task_id,
        request.resource_ref,
        request.gva,
        request.size,
        request.usage,
        Some(request.required_import_len),
    )
}

/// Ways a declared mapping view's recorded page GPAs can disagree with what the
/// task's page table resolves the same span to *now*.
enum ViewGpaDisagreement {
    /// The view names a different physical page than the page table does. The
    /// alias built from the view therefore reads and writes memory the guest
    /// does not believe is behind this address.
    Differs {
        index: usize,
        view_gpa: u64,
        table_gpa: u64,
    },
    /// The page table resolves fewer pages of the span than the view claims to
    /// cover. The tail of the alias has no guest backing at all.
    Short {
        view_pages: usize,
        table_pages: usize,
    },
}

impl ViewGpaDisagreement {
    /// Judge a view's recorded page GPAs against what the page table resolves.
    ///
    /// `table` may legitimately be *longer* than `view_gpas`: the walk covers
    /// the whole aligned span while the view covers the resource. Only a short
    /// table, or a page the two name differently, is a disagreement. The first
    /// differing page is the one reported — a later one adds no information a
    /// reader of the first does not already have.
    fn judge(view_gpas: &[u64], table: &[u64]) -> Option<Self> {
        if table.len() < view_gpas.len() {
            return Some(Self::Short {
                view_pages: view_gpas.len(),
                table_pages: table.len(),
            });
        }
        view_gpas
            .iter()
            .zip(table.iter())
            .enumerate()
            .find(|(_, (view, table))| view != table)
            .map(|(index, (view, table))| Self::Differs {
                index,
                view_gpa: *view,
                table_gpa: *table,
            })
    }
}

impl reims_vgpu_observe::Decline for ViewGpaDisagreement {
    fn slug(&self) -> &'static str {
        match self {
            Self::Differs { .. } => "zc_packed_view_gpa_differs",
            Self::Short { .. } => "zc_packed_view_gpa_short",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Differs {
                index,
                view_gpa,
                table_gpa,
            } => vec![
                ("page", index.to_string()),
                ("view_gpa", format!("{view_gpa:#x}")),
                ("table_gpa", format!("{table_gpa:#x}")),
            ],
            Self::Short {
                view_pages,
                table_pages,
            } => vec![
                ("view_pages", view_pages.to_string()),
                ("table_pages", table_pages.to_string()),
            ],
        }
    }
}

/// Judge a declared mapping view's page GPAs against the task page table.
///
/// This device resolves the physical pages behind a packed span two different
/// ways, and which one runs is decided by whether the host-pointer import is
/// available. With the import, a zero-copy alias takes the GPAs the guest
/// recorded when it *declared* the mapping view. Without it, the copying rail
/// walks the task's page table at bind time. Both are contract answers — the
/// view is the guest's own declaration and the page table is the guest's own
/// translation — so they are required to agree, and nothing until now compared
/// them.
///
/// A disagreement is a genuine loss of guest work rather than a policy choice:
/// the alias is bound over pages the guest does not currently place at that
/// address, so the GPU samples or overwrites unrelated memory while every
/// counter in the tree reads healthy. That is why this is on the fail channel
/// and not only in the census.
///
/// It runs at construction and not on reuse, which is what makes it free: the
/// packed resolution is cached, so this walks the page table about once per
/// distinct resource rather than once per bind. It therefore cannot see a page
/// table edited *after* an alias was built — a reuse-time audit would be needed
/// for that, and would need a stride to pay for itself.
fn audit_view_gpas_against_page_table<M: crate::runtime::host::HostMemory>(
    host: &M,
    state: &crate::runtime::Device,
    task_id: u32,
    page_base: u64,
    map_len: u64,
    view_gpas: &[u64],
) {
    let table = crate::runtime::gva_mem::task_gva_page_gpas(
        host,
        &state.tasks,
        task_id,
        page_base,
        map_len,
        state.page_shift,
    );
    match ViewGpaDisagreement::judge(view_gpas, &table) {
        Some(disagreement) => {
            use reims_vgpu_observe::Decline as _;
            crate::runtime::drain::note_store_route(disagreement.slug());
            reims_vgpu_observe::Emit::decline("zc_packed_view", &disagreement)
                .field("gva", format!("{page_base:#x}"))
                .fail_once(page_base);
        }
        None => crate::runtime::drain::note_store_route("zc_packed_view_gpa_agree"),
    }
}

#[allow(clippy::too_many_arguments)]
fn ensure_packed_resource_with_extent<
    M: crate::runtime::host::HostMemory + crate::runtime::host::HostOps,
>(
    state: &mut crate::runtime::Device,
    host: &mut M,
    task_id: u32,
    resource_ref: u32,
    gva: u64,
    size: u64,
    usage: PackedResourceUse,
    requested_import_len: Option<u64>,
) -> bool {
    let page = state.page_size();
    let page_base = gva & !(page - 1);
    let Some(head) = gva.checked_sub(page_base) else {
        return false;
    };
    let Some(map_len) = head
        .checked_add(size)
        .and_then(|len| reims_vgpu_protocol::align_up_u64(len, page))
    else {
        return false;
    };
    let required_import_len = requested_import_len.unwrap_or(map_len).max(map_len);
    let mut force_padded_alias = false;
    let mut previous_available = None;
    if let Some(held) = state.bound_buffers.packed(task_id, resource_ref) {
        let matches = match held {
            PackedBufferResolution::Available(buffer) => {
                let same = buffer.gva == gva && buffer.size == size;
                force_padded_alias = same && buffer.import.len() < required_import_len;
                if force_padded_alias {
                    previous_available = Some(buffer.clone());
                }
                same && !force_padded_alias
            }
            PackedBufferResolution::Unavailable {
                gva: held_gva,
                size: held_size,
                required_import_len: held_requirement,
            } => *held_gva == gva && *held_size == size && *held_requirement >= required_import_len,
        };
        if matches {
            let available = matches!(held, PackedBufferResolution::Available(_));
            return available;
        }
    }

    let unavailable = || PackedBufferResolution::Unavailable {
        gva,
        size,
        required_import_len,
    };
    let made = (|| {
        if !stable_guest_alias_available(host) {
            return None;
        }
        if !force_padded_alias {
            if let Some((import, import_offset, mapping_page_base, mapping_pages)) =
                crate::runtime::gva_view::mapping_import_for_span(state, task_id, gva, size)
            {
                if import.len() >= required_import_len {
                    let first =
                        usize::try_from(page_base.checked_sub(mapping_page_base)? / page).ok()?;
                    let count = usize::try_from(map_len / page).ok()?;
                    let gpas: Vec<u64> = mapping_pages
                        .get(first..first.checked_add(count)?)?
                        .to_vec();
                    audit_view_gpas_against_page_table(
                        host, state, task_id, page_base, map_len, &gpas,
                    );
                    let whole = import.slice(import_offset, size).ok()?;
                    let guest =
                        crate::runtime::guest_ram::GuestRef::new(Arc::clone(&import), whole)
                            .ok()?;
                    let footprint = crate::runtime::guest_ram::GuestPageFootprint::new(
                        Arc::<[u64]>::from(gpas.clone()),
                        page,
                    )?;
                    let host_ptr = import
                        .host_base()
                        .checked_add(usize::try_from(import_offset).ok()?)?;
                    crate::runtime::drain::note_store_route("zc_packed_mapping_import");
                    return Some(PackedBufferResolution::Available(PackedBuffer {
                        gva,
                        size,
                        head: import_offset,
                        import,
                        gpas: Arc::new(gpas),
                        footprint,
                        runs: Arc::new(vec![GuestRun {
                            host_ptr,
                            len: size,
                        }]),
                        pages: Arc::new(vec![GuestWindowRun {
                            window_offset: 0,
                            guest,
                        }]),
                        sampled_image_requirements: std::collections::HashMap::new(),
                        owned_alias: None,
                    }));
                }
            }
        }
        let align =
            match crate::runtime::guest_ram::host_allocation_import_align(required_import_len) {
                Ok(align) => align,
                Err(refusal) => {
                    crate::runtime::guest_ram::report_host_allocation_import_refusal(
                        "task_buffer_alias_import",
                        &refusal,
                    );
                    return None;
                }
            };
        let gpas = crate::runtime::gva_mem::task_gva_page_gpas(
            host,
            &state.tasks,
            task_id,
            page_base,
            map_len,
            state.page_shift,
        );
        if gpas.len() as u64 != map_len / page {
            return None;
        }
        let footprint = crate::runtime::guest_ram::GuestPageFootprint::new(
            Arc::<[u64]>::from(gpas.clone()),
            page,
        )?;
        if !force_padded_alias {
            let retained =
                crate::runtime::guest_ram_map::reference_for_pages(host, &gpas, page, head, size)
                    .ok()
                    .and_then(|guest| {
                        let base = guest.import().gpa_base()?;
                        let head_in_import = gpas.first()?.checked_add(head)?.checked_sub(base)?;
                        Some((Arc::clone(guest.import()), head_in_import, guest))
                    });
            if let Some((import, retained_head, guest)) = retained {
                if import.len() >= required_import_len {
                    let host_ptr = import
                        .host_base()
                        .checked_add(usize::try_from(retained_head).ok()?)?;
                    crate::runtime::drain::note_store_route("zc_packed_ramblock");
                    return Some(PackedBufferResolution::Available(PackedBuffer {
                        gva,
                        size,
                        head: retained_head,
                        import,
                        gpas: Arc::new(gpas),
                        footprint,
                        runs: Arc::new(vec![GuestRun {
                            host_ptr,
                            len: size,
                        }]),
                        pages: Arc::new(vec![GuestWindowRun {
                            window_offset: 0,
                            guest,
                        }]),
                        sampled_image_requirements: std::collections::HashMap::new(),
                        owned_alias: None,
                    }));
                }
            }
        }

        let allocation_align = align.max(page);
        let alias_len_u64 =
            reims_vgpu_protocol::align_up_u64(required_import_len, allocation_align)?;
        let alias_len = usize::try_from(alias_len_u64).ok()?;
        let host_base = if alias_len_u64 > map_len {
            host.map_pages_with_padding(&gpas, page as usize, alias_len)?
        } else {
            host.map_pages(&gpas, page as usize)?
        };
        let Some(host_ptr) = host_base.checked_add(head as usize) else {
            host.unmap_pages(host_base, alias_len);
            return None;
        };
        let import = match crate::runtime::guest_ram::GuestRamImport::new_host_allocation(
            host_base,
            alias_len_u64,
            align,
        ) {
            Ok(import) => Arc::new(import),
            Err(_) => {
                host.unmap_pages(host_base, alias_len);
                return None;
            }
        };
        let guest = match import
            .slice(head, size)
            .and_then(|whole| crate::runtime::guest_ram::GuestRef::new(Arc::clone(&import), whole))
        {
            Ok(guest) => guest,
            Err(_) => {
                import.retire();
                host.unmap_pages(host_base, alias_len);
                return None;
            }
        };
        crate::runtime::drain::note_store_route("zc_packed_alias_import");
        crate::runtime::drain::note_store_route(packed_scatter_band(&gpas, page));
        Some(PackedBufferResolution::Available(PackedBuffer {
            gva,
            size,
            head,
            import: Arc::clone(&import),
            gpas: Arc::new(gpas),
            footprint,
            runs: Arc::new(vec![GuestRun {
                host_ptr,
                len: size,
            }]),
            pages: Arc::new(vec![GuestWindowRun {
                window_offset: 0,
                guest,
            }]),
            sampled_image_requirements: previous_available
                .as_ref()
                .map(|previous| previous.sampled_image_requirements.clone())
                .unwrap_or_default(),
            owned_alias: Some(PackedAliasRetirement {
                import: import.id(),
                ptr: host_base,
                len: alias_len,
            }),
        }))
    })()
    .unwrap_or_else(unavailable);

    crate::runtime::drain::note_store_route(match (usage, &made) {
        (PackedResourceUse::Buffer, PackedBufferResolution::Available(_)) => {
            "zc_buffer_packed_alias"
        }
        (PackedResourceUse::Buffer, PackedBufferResolution::Unavailable { .. }) => {
            "zc_buffer_packed_unavailable"
        }
        (PackedResourceUse::LinearSample, PackedBufferResolution::Available(_)) => {
            "zc_lin_packed_alias"
        }
        (PackedResourceUse::LinearSample, PackedBufferResolution::Unavailable { .. }) => {
            "zc_lin_packed_unavailable"
        }
        (PackedResourceUse::LinearTarget, PackedBufferResolution::Available(_)) => {
            "zc_target_packed_alias"
        }
        (PackedResourceUse::LinearTarget, PackedBufferResolution::Unavailable { .. }) => {
            "zc_target_packed_unavailable"
        }
        (PackedResourceUse::ComputeTexture, PackedBufferResolution::Available(_)) => {
            "zc_compute_texture_packed_alias"
        }
        (PackedResourceUse::ComputeTexture, PackedBufferResolution::Unavailable { .. }) => {
            "zc_compute_texture_packed_unavailable"
        }
    });
    let available = matches!(made, PackedBufferResolution::Available(_));
    if !available && previous_available.is_some() {
        // A backend-specific extent upgrade must not destroy the exact guest
        // allocation that remains a valid copied fallback. The failed larger
        // view has no lifetime to publish; the existing resource does.
        return false;
    }
    if let Some(retired) = state
        .bound_buffers
        .insert_packed(task_id, resource_ref, made)
    {
        state
            .host_materializations
            .retire_materialization(Some((retired.ptr, retired.len)), Some(retired.import));
    }
    available
}

#[derive(Clone, Debug)]
pub enum PackedBufferResolution {
    Available(PackedBuffer),
    /// The whole declared allocation could not be mapped. Narrow, individually
    /// walkable binds remain valid and use the existing gather rail.
    Unavailable {
        gva: u64,
        size: u64,
        required_import_len: u64,
    },
}

impl PackedBufferResolution {
    fn into_alias_retirement(self) -> Option<PackedAliasRetirement> {
        match self {
            Self::Available(buffer) => buffer.owned_alias,
            Self::Unavailable { .. } => None,
        }
    }
}

pub(crate) struct BoundBufferRetirement {
    pub window_count: usize,
    pub aliases: Vec<PackedAliasRetirement>,
}

impl BoundBufferRetirement {
    fn from_registry(
        retired: reims_vgpu_core::MaterializationRetirement<PackedBufferResolution>,
    ) -> Self {
        Self {
            window_count: retired.window_count,
            aliases: retired
                .resources
                .into_iter()
                .filter_map(PackedBufferResolution::into_alias_retirement)
                .collect(),
        }
    }
}

/// A resolved bind: where this reference's bytes live, as the engine binds them.
///
/// Both lists are `Arc`ed by the producer already, so a lookup hands the draw
/// path the same allocation the walk built rather than a copy of it.
#[derive(Clone, Debug)]
pub struct BoundBuffer {
    /// Guest VA the resolution starts at (the backing's `gva + offset`).
    pub gva: u64,
    /// Byte length the runs cover, and the bind's `total_len`.
    pub span: u64,
    /// First byte of this bind inside `runs` / `pages`.
    ///
    /// Exact-window resolutions use zero. A packed resource shares its one
    /// whole-buffer source and carries the guest's bind offset here.
    pub source_offset: u64,
    /// Host-pointer spans the CPU gather walks.
    pub runs: Arc<Vec<GuestRun>>,
    /// The same bytes as bounded references into this process's import, when
    /// the host can import guest RAM at all. `None` keeps the caller on the
    /// gathering arm exactly as a fresh resolution would.
    pub pages: Option<Arc<Vec<GuestWindowRun>>>,
    /// Canonical guest-physical identity of this exact bind window.
    pub physical_pages: Option<reims_vgpu_memory::GuestPageSet>,
}

/// `(task, reference, offset, extent cap)` — see the module doc on why the
/// offset is here.
///
/// The cap is in the key because it is not a property of the bind: it is what
/// the *shader on this draw* proved about how far it can read
/// ([`reims_vgpu_vulkan::spirv_bind::reflected_buffer_extent`]). Two shaders may
/// bind one allocation at one offset and declare different extents, and a
/// resolution walked for the narrower one covers fewer bytes than the wider one
/// needs. Keyed without the cap, that resolution would be handed to the wider
/// shader as a hit and the GPU would read a short buffer — no error, wrong
/// pixels. Keyed with it, the two coexist and each pays its own walk.
///
/// `None` — the uncapped whole-allocation resolution — is a distinct key from
/// any capped one, which is what keeps the pre-existing behaviour reachable and
/// unchanged for every bind reflection does not bound.
fn owner(task: u32, object: u32) -> reims_vgpu_core::MaterializationOwner {
    reims_vgpu_core::MaterializationOwner::new(
        reims_vgpu_protocol::TaskId::new(task),
        reims_vgpu_protocol::ObjectTableRef::<reims_vgpu_protocol::ResourceObject>::new(object),
    )
}

fn window_key(
    task: u32,
    object: u32,
    offset: u64,
    cap: Option<u64>,
) -> reims_vgpu_core::BoundWindowKey {
    reims_vgpu_core::BoundWindowKey {
        owner: owner(task, object),
        offset: reims_vgpu_protocol::ByteOffset::new(offset),
        extent_cap: cap.map(reims_vgpu_protocol::ByteLength::new),
    }
}

fn address_span(gva: u64, len: u64) -> reims_vgpu_core::GuestAddressSpan {
    reims_vgpu_core::GuestAddressSpan::new(
        reims_vgpu_protocol::GuestVirtualAddress::new(gva),
        reims_vgpu_protocol::ByteLength::new(len),
    )
}

/// What [`BoundBuffers::shape`] measures. Levels, not per-interval counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegistryShape {
    /// Held resolutions.
    pub entries: usize,
    /// Distinct `(task, reference)` pairs behind them — what the registry would
    /// hold if it were keyed the way Apple's is.
    pub pairs: usize,
    /// Pairs held at more than one offset. Zero means the offset in the key
    /// never separates two live entries.
    pub multi_offset_pairs: usize,
    /// The most offsets any one pair is held at.
    pub max_offsets: u32,
}

/// Live physical-alias shape behind resource-owned packed allocations.
///
/// Mapping-owned and RAMBlock imports are excluded: those already have a
/// canonical allocation identity. This measures only the fallback aliases for
/// which sharing would change an ownership boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PackedAliasShape {
    pub resources: usize,
    pub distinct_page_plans: usize,
    pub duplicate_resources: usize,
    pub max_resources_per_plan: u32,
    pub physical_pages: usize,
    pub multiply_aliased_pages: usize,
    pub max_aliases_per_page: u32,
}

/// Report the registry's shape once per census interval, on the same one-second
/// cadence as `store_routes` so the two line up row for row.
///
/// Read against the `bb_retire_*` routes: those say how many resolutions a
/// retirement dropped, this says what the survivors look like. A miss is either
/// a retired key or a key never seen, and the two together are what decide
/// whether the 12.8x more fresh resolutions on an importing host are churn in
/// the retirement rules or churn in the keys.
pub fn note_registry_levels(state: &crate::runtime::Device) {
    use std::sync::atomic::{AtomicU64, Ordering};
    static LAST_MS: AtomicU64 = AtomicU64::new(0);
    static PEAK_ENTRIES: AtomicU64 = AtomicU64::new(0);

    let now = crate::observe::elapsed_ms() as u64;
    let last = LAST_MS.load(Ordering::Relaxed);
    if now.saturating_sub(last) < 1000 {
        return;
    }
    // Losing the race only costs a skipped interval, never a double line.
    if LAST_MS
        .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
        .is_err()
    {
        return;
    }
    let shape = state.bound_buffers.shape();
    let peak = PEAK_ENTRIES
        .fetch_max(shape.entries as u64, Ordering::Relaxed)
        .max(shape.entries as u64);
    let aliases = state.bound_buffers.packed_alias_shape();
    crate::observe::off(format!(
        "bound_buffers (levels, not per-interval) entries={} peak={} pairs={} \
         multi_offset_pairs={} max_offsets={} aliases={} alias_plans={} \
         alias_duplicates={} alias_plan_max={} alias_pages={} alias_shared_pages={} \
         alias_page_max={}",
        shape.entries,
        peak,
        shape.pairs,
        shape.multi_offset_pairs,
        shape.max_offsets,
        aliases.resources,
        aliases.distinct_page_plans,
        aliases.duplicate_resources,
        aliases.max_resources_per_plan,
        aliases.physical_pages,
        aliases.multiply_aliased_pages,
        aliases.max_aliases_per_page,
    ));
}

/// Every held bind resolution on this device.
#[derive(Default, Debug)]
pub struct BoundBuffers {
    retained: reims_vgpu_core::MaterializationRegistry<BoundBuffer, PackedBufferResolution>,
}

impl BoundBuffers {
    /// The resolution for this bind, if one is held.
    pub fn get(
        &self,
        task_id: u32,
        buffer_ref: u32,
        offset: u64,
        cap: Option<u64>,
    ) -> Option<&BoundBuffer> {
        self.retained
            .window(window_key(task_id, buffer_ref, offset, cap))
    }

    /// Hold a freshly walked resolution.
    pub fn insert(
        &mut self,
        task_id: u32,
        buffer_ref: u32,
        offset: u64,
        cap: Option<u64>,
        bound: BoundBuffer,
    ) {
        let span = address_span(bound.gva, bound.span);
        self.retained
            .insert_window(window_key(task_id, buffer_ref, offset, cap), span, bound);
    }

    pub fn packed(&self, task_id: u32, buffer_ref: u32) -> Option<&PackedBufferResolution> {
        self.retained.resource(owner(task_id, buffer_ref))
    }

    /// Borrow the retained allocation when it still describes exactly this
    /// resource construction.
    ///
    /// The geometry check matters on the narrow window between a descriptor
    /// changing and its retirement packet being consumed: returning the old
    /// allocation there would make a warm lookup observably different from a
    /// fresh resolution. Returning a reference is equally deliberate. A warm
    /// encoder bind borrows its resource object and retains only the execution
    /// payload it hands onward; it does not acquire all of the construction
    /// state merely to inspect it.
    pub fn packed_available(
        &self,
        task_id: u32,
        resource_ref: u32,
        gva: u64,
        size: u64,
    ) -> Option<&PackedBuffer> {
        match self.packed(task_id, resource_ref)? {
            PackedBufferResolution::Available(packed)
                if packed.gva == gva && packed.size == size =>
            {
                Some(packed)
            }
            PackedBufferResolution::Available(_) | PackedBufferResolution::Unavailable { .. } => {
                None
            }
        }
    }

    pub fn note_sampled_image_requirement(
        &mut self,
        task_id: u32,
        resource_ref: u32,
        gva: u64,
        size: u64,
        key: reims_vgpu_memory::GuestImageBindingKey,
        requirement: reims_vgpu_memory::GuestImageBindingDisposition,
    ) {
        if let Some(PackedBufferResolution::Available(packed)) =
            self.retained.resource_mut(owner(task_id, resource_ref))
        {
            if packed.gva == gva && packed.size == size {
                packed.sampled_image_requirements.insert(key, requirement);
            }
        }
    }

    pub(crate) fn insert_packed(
        &mut self,
        task_id: u32,
        buffer_ref: u32,
        packed: PackedBufferResolution,
    ) -> Option<PackedAliasRetirement> {
        let (gva, size) = match &packed {
            PackedBufferResolution::Available(buffer) => (buffer.gva, buffer.size),
            PackedBufferResolution::Unavailable { gva, size, .. } => (*gva, *size),
        };
        self.retained
            .insert_resource(owner(task_id, buffer_ref), address_span(gva, size), packed)
            .and_then(PackedBufferResolution::into_alias_retirement)
    }

    /// Drop everything held for one task.
    ///
    /// The answer for a page-table root change, a new object list, or a deleted
    /// task: in each case every reference may now name different bytes.
    pub(crate) fn take_task(&mut self, task_id: u32) -> BoundBufferRetirement {
        BoundBufferRetirement::from_registry(
            self.retained
                .take_task(reims_vgpu_protocol::TaskId::new(task_id)),
        )
    }

    #[cfg(test)]
    pub fn retire_task(&mut self, task_id: u32) -> usize {
        self.take_task(task_id).window_count
    }

    /// Drop everything held for one reference, at every offset.
    ///
    /// The `CmdDeleteObject` answer. That packet names the reference —
    /// `delete_object(task_id, ref_)` — and the rest of the device already
    /// scopes its response to it: the canonical resource, host copies, and its
    /// retained IOSurface relation are all keyed `(task, ref)`. This registry
    /// retiring the whole task was the outlier, and a measured expensive one:
    /// one driven boot dropped 54 109 resolutions there, 95% of every bind miss
    /// on the device.
    ///
    /// # Why the narrower rule is sound
    ///
    /// A held resolution for reference `R` is built from three things: the
    /// object-list entry at index `R`, the descriptor that entry names, and the
    /// task page-table walk over the span the descriptor declares. Deleting
    /// object `X` where `X != R` touches none of them — the list is indexed by
    /// reference so entries do not shift, `X`'s descriptor is at its own
    /// address, and no page table changes. So no resolution but `R`'s own can
    /// be stale because of this packet.
    ///
    /// Two references aliasing one allocation are safe for the same reason:
    /// deleting one does not free the pages or move the other's descriptor. If
    /// the guest then reuses that address the announcement is a different
    /// packet — `CmdMapMemory2`, `CmdUnmapMemory` or `CmdReplacePhysical` — and
    /// those rules still retire by range or by task.
    pub(crate) fn take_ref(&mut self, task_id: u32, buffer_ref: u32) -> BoundBufferRetirement {
        BoundBufferRetirement::from_registry(self.retained.take_object(owner(task_id, buffer_ref)))
    }

    #[cfg(test)]
    pub fn retire_ref(&mut self, task_id: u32, buffer_ref: u32) -> usize {
        self.take_ref(task_id, buffer_ref).window_count
    }

    /// Drop everything held for `task_id` whose bytes overlap `[gva, gva+len)`.
    ///
    /// The map/unmap answer, which carries the exact range that moved.
    pub(crate) fn take_range(&mut self, task_id: u32, gva: u64, len: u64) -> BoundBufferRetirement {
        BoundBufferRetirement::from_registry(self.retained.take_range(
            reims_vgpu_protocol::TaskId::new(task_id),
            address_span(gva, len),
        ))
    }

    #[cfg(test)]
    pub fn retire_range(&mut self, task_id: u32, gva: u64, len: u64) -> usize {
        self.take_range(task_id, gva, len).window_count
    }

    /// Drop everything. Device reset, where no guest state survives.
    pub(crate) fn take_all(&mut self) -> BoundBufferRetirement {
        BoundBufferRetirement::from_registry(self.retained.take_all())
    }

    /// How many resolutions are held, for the census.
    pub fn len(&self) -> usize {
        self.retained.window_len()
    }

    /// The registry's shape: entries, the distinct `(task, reference)` pairs
    /// behind them, how many of those pairs are held at more than one offset,
    /// and the most offsets any single pair carries.
    ///
    /// This is the instrument for the question the module doc states and does
    /// not answer. Apple keys by reference alone; this keys by
    /// `(task, reference, offset)`, on the belief that a reference is bound at
    /// one offset and the two keys therefore describe the same registry. That
    /// belief has never been counted. `pairs == entries` says it holds and the
    /// extra field is inert; `pairs < entries` says references really are bound
    /// at several offsets, each paying its own walk, and the narrower key is
    /// costing exactly `entries - pairs` resolutions.
    ///
    /// Walked once per census interval rather than tracked incrementally: a
    /// second index would have to be maintained by every retirement rule, which
    /// is a correctness surface bought for a measurement, and the population is
    /// the guest's live working set rather than anything unbounded.
    pub fn shape(&self) -> RegistryShape {
        let shape = self.retained.shape();
        RegistryShape {
            entries: shape.entries,
            pairs: shape.owners,
            multi_offset_pairs: shape.multi_offset_owners,
            max_offsets: shape.max_offsets,
        }
    }

    /// Measure whether resource-owned aliases actually duplicate one another.
    ///
    /// This deliberately derives from the owning registry when sampled. A
    /// continuously maintained reverse index would become a second lifetime
    /// graph whose retirements could disagree with the resource graph it is
    /// supposed to describe.
    pub fn packed_alias_shape(&self) -> PackedAliasShape {
        let aliases: Vec<&PackedBuffer> = self
            .retained
            .resource_values()
            .filter_map(|resolution| match resolution {
                PackedBufferResolution::Available(buffer) if buffer.owned_alias.is_some() => {
                    Some(buffer)
                }
                PackedBufferResolution::Available(_)
                | PackedBufferResolution::Unavailable { .. } => None,
            })
            .collect();
        let mut plans = std::collections::HashMap::<&[u64], u32>::new();
        let mut pages = std::collections::HashMap::<u64, u32>::new();
        for alias in &aliases {
            *plans.entry(alias.gpas.as_slice()).or_default() += 1;
            for &page in alias.gpas.iter() {
                *pages.entry(page).or_default() += 1;
            }
        }
        PackedAliasShape {
            resources: aliases.len(),
            distinct_page_plans: plans.len(),
            duplicate_resources: plans
                .values()
                .map(|count| count.saturating_sub(1) as usize)
                .sum(),
            max_resources_per_plan: plans.values().copied().max().unwrap_or(0),
            physical_pages: pages.len(),
            multiply_aliased_pages: pages.values().filter(|count| **count > 1).count(),
            max_aliases_per_page: pages.values().copied().max().unwrap_or(0),
        }
    }

    /// Whether nothing is held.
    pub fn is_empty(&self) -> bool {
        self.retained.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bound(gva: u64, span: u64) -> BoundBuffer {
        BoundBuffer {
            gva,
            span,
            source_offset: 0,
            runs: Arc::new(Vec::new()),
            pages: None,
            physical_pages: None,
        }
    }

    fn packed(gva: u64, pages: &[u64], allocation: usize) -> PackedBufferResolution {
        let page_size = 0x1000;
        let import = Arc::new(
            crate::runtime::guest_ram::GuestRamImport::new_host_allocation(
                0x10_0000 + allocation * 0x10_0000,
                pages.len() as u64 * page_size,
                page_size,
            )
            .unwrap(),
        );
        let import_id = import.id();
        PackedBufferResolution::Available(PackedBuffer {
            gva,
            size: pages.len() as u64 * page_size,
            head: 0,
            import,
            gpas: Arc::new(pages.to_vec()),
            footprint: crate::runtime::guest_ram::GuestPageFootprint::new(
                Arc::from(pages),
                page_size,
            )
            .unwrap(),
            runs: Arc::new(Vec::new()),
            pages: Arc::new(Vec::new()),
            sampled_image_requirements: std::collections::HashMap::new(),
            owned_alias: Some(PackedAliasRetirement {
                import: import_id,
                ptr: 0x10_0000 + allocation * 0x10_0000,
                len: pages.len() * page_size as usize,
            }),
        })
    }

    /// The two rails that resolve a packed span must agree, and this is the
    /// only place they are compared.
    ///
    /// The zero-copy alias takes the page GPAs the guest recorded when it
    /// declared the mapping view; the copying rail walks the task page table at
    /// bind time. A boot with the host-pointer import runs only the first, so
    /// nothing else in the tree can notice the two diverging — and the way it
    /// fails is content, which no counter reports.
    #[test]
    fn a_view_page_that_the_page_table_places_elsewhere_is_a_disagreement() {
        let view = [0x10_000, 0x20_000, 0x30_000];

        assert!(
            ViewGpaDisagreement::judge(&view, &view).is_none(),
            "identical resolutions are the healthy case"
        );

        // The walk covers the whole aligned span, so trailing pages beyond the
        // resource are expected and must not read as a disagreement.
        assert!(
            ViewGpaDisagreement::judge(&view, &[0x10_000, 0x20_000, 0x30_000, 0x40_000]).is_none(),
            "a longer page-table walk is not a disagreement"
        );

        match ViewGpaDisagreement::judge(&view, &[0x10_000, 0x99_000, 0x30_000]) {
            Some(ViewGpaDisagreement::Differs {
                index,
                view_gpa,
                table_gpa,
            }) => {
                assert_eq!((index, view_gpa, table_gpa), (1, 0x20_000, 0x99_000));
            }
            other => panic!(
                "a relocated page must be reported, got {:?}",
                other.is_none()
            ),
        }

        match ViewGpaDisagreement::judge(&view, &[0x10_000]) {
            Some(ViewGpaDisagreement::Short {
                view_pages,
                table_pages,
            }) => assert_eq!((view_pages, table_pages), (3, 1)),
            other => panic!(
                "an unbacked tail must be reported, got {:?}",
                other.is_none()
            ),
        }
    }

    /// Every arm names itself, because a shared slug is the one defect the
    /// decline vocabulary exists to prevent.
    #[test]
    fn each_view_disagreement_names_its_own_check() {
        use reims_vgpu_observe::Decline as _;
        let differs = ViewGpaDisagreement::Differs {
            index: 1,
            view_gpa: 0x20_000,
            table_gpa: 0x99_000,
        };
        let short = ViewGpaDisagreement::Short {
            view_pages: 3,
            table_pages: 1,
        };
        assert_ne!(differs.slug(), short.slug());
        assert_eq!(
            differs.fields().iter().map(|(k, _)| *k).collect::<Vec<_>>(),
            ["page", "view_gpa", "table_gpa"]
        );
    }

    #[test]
    fn packed_alias_shape_distinguishes_equal_plans_from_page_overlap() {
        let mut buffers = BoundBuffers::default();
        buffers.insert_packed(1, 1, packed(0x1000, &[0x10_000, 0x20_000], 1));
        buffers.insert_packed(1, 2, packed(0x3000, &[0x10_000, 0x20_000], 2));
        buffers.insert_packed(1, 3, packed(0x5000, &[0x20_000, 0x30_000], 3));

        assert_eq!(
            buffers.packed_alias_shape(),
            PackedAliasShape {
                resources: 3,
                distinct_page_plans: 2,
                duplicate_resources: 1,
                max_resources_per_plan: 2,
                physical_pages: 3,
                multiply_aliased_pages: 2,
                max_aliases_per_page: 3,
            }
        );
    }

    /// The lookup is keyed by all three of task, reference and offset, so no
    /// two binds can collide onto one resolution.
    #[test]
    fn a_resolution_is_found_only_by_its_own_key() {
        let mut b = BoundBuffers::default();
        b.insert(7, 3, 0, None, bound(0x1000, 0x2000));
        assert!(b.get(7, 3, 0, None).is_some());
        assert!(b.get(7, 3, 0x100, None).is_none(), "a different offset");
        assert!(b.get(7, 4, 0, None).is_none(), "a different reference");
        assert!(b.get(8, 3, 0, None).is_none(), "a different task");
    }

    /// A resolution walked under one shader's extent cap is never served to a
    /// shader that proved a different one.
    ///
    /// This is the corruption guard for the narrowing rail, and it fails in the
    /// direction that has no other alarm: a 64-byte resolution handed to a
    /// shader entitled to 4096 does not error, it reads whatever the GPU finds
    /// past the end of a short buffer and draws it. The uncapped entry is a
    /// fourth distinct key rather than a wildcard, so a bind reflection could
    /// not bound never picks up a neighbour's narrowing either.
    #[test]
    fn a_resolution_is_never_served_across_a_different_extent_cap() {
        let mut b = BoundBuffers::default();
        b.insert(1, 1, 0, Some(64), bound(0x1000, 64));

        assert!(b.get(1, 1, 0, Some(64)).is_some(), "its own cap");
        assert!(
            b.get(1, 1, 0, Some(4096)).is_none(),
            "a shader entitled to more must not get the 64-byte walk"
        );
        assert!(
            b.get(1, 1, 0, None).is_none(),
            "an unbounded bind must not get a capped walk"
        );

        // The three coexist rather than evicting one another, so neither shader
        // re-walks on every draw because the other one ran in between.
        b.insert(1, 1, 0, Some(4096), bound(0x1000, 4096));
        b.insert(1, 1, 0, None, bound(0x1000, 0x10000));
        assert_eq!(b.len(), 3);
        assert_eq!(b.get(1, 1, 0, Some(64)).map(|r| r.span), Some(64));
        assert_eq!(b.get(1, 1, 0, Some(4096)).map(|r| r.span), Some(4096));
        assert_eq!(b.get(1, 1, 0, None).map(|r| r.span), Some(0x10000));

        // A retirement is keyed on task and reference, so it takes all three.
        assert_eq!(b.retire_ref(1, 1), 3);
        assert!(b.is_empty());
    }

    /// A map/unmap notify retires exactly the resolutions whose bytes moved,
    /// and leaves the neighbours that did not.
    #[test]
    fn a_range_retire_takes_the_overlapping_resolutions_only() {
        let mut b = BoundBuffers::default();
        b.insert(1, 1, 0, None, bound(0x1000, 0x1000)); // [0x1000,0x2000)
        b.insert(1, 2, 0, None, bound(0x2000, 0x1000)); // [0x2000,0x3000)
        b.insert(1, 3, 0, None, bound(0x9000, 0x1000)); // far away
        assert_eq!(b.retire_range(1, 0x1800, 0x1000), 2, "spans the first two");
        assert!(b.get(1, 3, 0, None).is_some(), "the far one survives");
        assert_eq!(b.len(), 1);
    }

    /// Whole-buffer alias answers share the reference lifecycle even when no
    /// offset resolution has been materialized yet.
    #[test]
    fn packed_alias_answers_retire_with_their_reference_and_mapping() {
        let mut b = BoundBuffers::default();
        b.insert_packed(
            1,
            7,
            PackedBufferResolution::Unavailable {
                gva: 0x4000,
                size: 0x3000,
                required_import_len: 0x3000,
            },
        );
        b.insert_packed(
            1,
            8,
            PackedBufferResolution::Unavailable {
                gva: 0x9000,
                size: 0x1000,
                required_import_len: 0x1000,
            },
        );
        assert!(b.packed(1, 7).is_some());
        assert_eq!(b.retire_range(1, 0x5000, 0x1000), 0);
        assert!(b.packed(1, 7).is_none(), "overlapping alias answer");
        assert!(b.packed(1, 8).is_some(), "unrelated alias answer");
        assert_eq!(b.retire_ref(1, 8), 0);
        assert!(b.packed(1, 8).is_none());
    }

    #[test]
    fn one_packed_import_serves_every_offset_of_a_reference() {
        let import = Arc::new(
            crate::runtime::guest_ram::GuestRamImport::new_host_allocation(
                0x7f00_0000_0000,
                0x8000,
                0x1000,
            )
            .expect("aligned allocation"),
        );
        let id = import.id();
        let mut b = BoundBuffers::default();
        b.insert_packed(
            3,
            9,
            PackedBufferResolution::Available(PackedBuffer {
                gva: 0x10800,
                size: 0x7000,
                head: 0x800,
                import: Arc::clone(&import),
                gpas: Arc::new(Vec::new()),
                footprint: crate::runtime::guest_ram::GuestPageFootprint::new(
                    Arc::from([0x1000]),
                    0x1000,
                )
                .unwrap(),
                runs: Arc::new(Vec::new()),
                pages: Arc::new(Vec::new()),
                sampled_image_requirements: std::collections::HashMap::new(),
                owned_alias: None,
            }),
        );
        let PackedBufferResolution::Available(packed) = b.packed(3, 9).unwrap() else {
            panic!("available above")
        };
        for (offset, span) in [(0, 0x1000), (0x1800, 0x2000), (0x5000, 0x800)] {
            let slice = packed
                .import
                .slice(packed.head + offset, span)
                .expect("each bind lies in the one allocation");
            assert_eq!(slice.import(), id);
        }
        let owners = Arc::strong_count(&import);
        assert!(b.packed_available(3, 9, 0x10800, 0x7000).is_some());
        assert_eq!(
            Arc::strong_count(&import),
            owners,
            "a warm lookup borrows the resource rather than acquiring it"
        );
        assert!(
            b.packed_available(3, 9, 0x11800, 0x7000).is_none(),
            "a changed base cannot reuse the prior construction"
        );
        assert!(
            b.packed_available(3, 9, 0x10800, 0x6000).is_none(),
            "a changed allocation size cannot reuse the prior construction"
        );
        assert!(
            b.packed_available(4, 9, 0x10800, 0x7000).is_none(),
            "the same reference in another task is a different resource"
        );
    }

    #[test]
    fn packed_alias_retirement_carries_import_and_host_view_together() {
        let import = Arc::new(
            crate::runtime::guest_ram::GuestRamImport::new_host_allocation(
                0x7f00_0000_0000,
                0x2000,
                0x1000,
            )
            .unwrap(),
        );
        let expected = PackedAliasRetirement {
            import: import.id(),
            ptr: import.host_base(),
            len: 0x2000,
        };
        let mut buffers = BoundBuffers::default();
        let replaced = buffers.insert_packed(
            3,
            9,
            PackedBufferResolution::Available(PackedBuffer {
                gva: 0x1000,
                size: 0x2000,
                head: 0,
                import,
                gpas: Arc::new(vec![0x4000, 0x9000]),
                footprint: crate::runtime::guest_ram::GuestPageFootprint::new(
                    Arc::from([0x4000, 0x9000]),
                    0x1000,
                )
                .unwrap(),
                runs: Arc::new(Vec::new()),
                pages: Arc::new(Vec::new()),
                sampled_image_requirements: std::collections::HashMap::new(),
                owned_alias: Some(expected),
            }),
        );
        assert!(replaced.is_none());

        let retired = buffers.take_range(3, 0x1800, 1);
        assert_eq!(retired.window_count, 0);
        assert_eq!(retired.aliases, vec![expected]);
        assert!(buffers.is_empty());
    }

    #[test]
    fn packed_witness_separates_host_import_head_from_guest_page_index() {
        let import = Arc::new(
            crate::runtime::guest_ram::GuestRamImport::new_host_allocation(
                0x7f00_0000_0000,
                0x20_000,
                0x1000,
            )
            .expect("aligned host allocation"),
        );
        let packed = PackedBuffer {
            gva: 0x840_000,
            size: 0x1000,
            // A RAM-block import may begin many pages before this resource.
            head: 0x12_000,
            import: Arc::clone(&import),
            gpas: Arc::new(vec![0x220_000]),
            footprint: crate::runtime::guest_ram::GuestPageFootprint::new(
                Arc::from([0x220_000]),
                0x1000,
            )
            .unwrap(),
            runs: Arc::new(Vec::new()),
            pages: Arc::new(Vec::new()),
            sampled_image_requirements: std::collections::HashMap::new(),
            owned_alias: None,
        };

        let window = packed
            .witness_window(0, 4)
            .expect("the first four resource bytes are in its first guest page");
        assert_eq!(window.gpas, &[0x220_000]);
        assert_eq!(window.host_ptr, import.host_base() + 0x12_000);
        let access = packed.access();
        let access_window = access
            .witness_window(0, 4)
            .expect("an execution payload preserves the same witness coordinates");
        assert_eq!(access_window.gpas, window.gpas);
        assert_eq!(access_window.host_ptr, window.host_ptr);
        assert_eq!(
            packed.window_pages(0, 4).unwrap(),
            std::collections::HashSet::from([0x220_000])
        );
    }

    /// A range retire is scoped to its task: the same GVA under another task is
    /// a different address space and must not be touched.
    #[test]
    fn a_range_retire_does_not_cross_tasks() {
        let mut b = BoundBuffers::default();
        b.insert(1, 1, 0, None, bound(0x1000, 0x1000));
        b.insert(2, 1, 0, None, bound(0x1000, 0x1000));
        assert_eq!(b.retire_range(1, 0x1000, 0x1000), 1);
        assert!(b.get(2, 1, 0, None).is_some());
    }

    /// A zero-length notify names no bytes and must retire nothing — otherwise
    /// a malformed packet would silently drop every resolution it touched.
    #[test]
    fn a_zero_length_range_retires_nothing() {
        let mut b = BoundBuffers::default();
        b.insert(1, 1, 0, None, bound(0x1000, 0x1000));
        assert_eq!(b.retire_range(1, 0x1000, 0), 0);
        assert_eq!(b.len(), 1);
    }

    /// Ranges that merely touch at an endpoint do not overlap, so an unmap of
    /// the page after a resolution does not retire it.
    #[test]
    fn abutting_ranges_do_not_overlap() {
        let mut b = BoundBuffers::default();
        b.insert(1, 1, 0, None, bound(0x1000, 0x1000)); // [0x1000,0x2000)
        assert_eq!(b.retire_range(1, 0x2000, 0x1000), 0, "starts where it ends");
        assert_eq!(b.retire_range(1, 0x0000, 0x1000), 0, "ends where it starts");
        assert_eq!(b.len(), 1);
    }

    /// A reference retire takes that reference at **every** offset, and nothing
    /// else.
    ///
    /// Both halves matter and they fail in opposite directions. Leaving one of
    /// the deleted reference's offsets behind serves bytes from an object the
    /// guest destroyed; taking a neighbour's is the whole-task rule this
    /// replaced, which is merely expensive. The offsets are the ones a driven
    /// boot actually produces — a single reference is held at 233 of them.
    #[test]
    fn a_reference_retire_takes_every_offset_of_that_reference_only() {
        let mut b = BoundBuffers::default();
        for off in [0u64, 0x400, 0x1000, 0x9000] {
            b.insert(1, 7, off, None, bound(0x1000 + off, 0x400));
        }
        // A neighbouring reference on the same task, and the same reference
        // under another task: neither is named by this packet.
        b.insert(1, 8, 0, None, bound(0x8000, 0x400));
        b.insert(2, 7, 0, None, bound(0x1000, 0x400));
        assert_eq!(b.len(), 6);

        assert_eq!(b.retire_ref(1, 7), 4, "every offset of reference 7");
        assert!(b.get(1, 7, 0, None).is_none());
        assert!(b.get(1, 7, 0x9000, None).is_none());
        assert!(
            b.get(1, 8, 0, None).is_some(),
            "a sibling reference survives"
        );
        assert!(b.get(2, 7, 0, None).is_some(), "another task's survives");
        assert_eq!(b.len(), 2);

        // A reference nothing is held for is not an error, and takes nothing.
        assert_eq!(b.retire_ref(1, 7), 0);
        assert_eq!(b.retire_ref(1, 99), 0);
        assert_eq!(b.len(), 2);
    }

    /// The shape separates entries from the `(task, reference)` pairs behind
    /// them, which is the whole reason it exists: `pairs == entries` says the
    /// offset in the key never distinguishes two live entries, and anything
    /// less counts the resolutions the narrower key is paying for.
    #[test]
    fn the_shape_counts_pairs_apart_from_entries() {
        let mut b = BoundBuffers::default();
        assert_eq!(b.shape(), RegistryShape::default(), "empty");

        // One reference at one offset each: the two keys would agree.
        b.insert(1, 1, 0, None, bound(0x1000, 0x1000));
        b.insert(1, 2, 0, None, bound(0x2000, 0x1000));
        let s = b.shape();
        assert_eq!((s.entries, s.pairs), (2, 2));
        assert_eq!((s.multi_offset_pairs, s.max_offsets), (0, 1));

        // The same reference at a second offset is a second entry and not a
        // second pair — exactly the divergence from Apple's key.
        b.insert(1, 1, 0x400, None, bound(0x1400, 0x400));
        let s = b.shape();
        assert_eq!((s.entries, s.pairs), (3, 2), "one pair now holds two");
        assert_eq!((s.multi_offset_pairs, s.max_offsets), (1, 2));

        // The same reference under another task is a different pair, because a
        // GVA has no meaning apart from the table it resolves against.
        b.insert(2, 1, 0, None, bound(0x1000, 0x1000));
        let s = b.shape();
        assert_eq!((s.entries, s.pairs), (4, 3));
        assert_eq!(s.multi_offset_pairs, 1);
    }

    /// A task retire takes that task's resolutions whatever their addresses,
    /// and leaves every other task alone.
    #[test]
    fn a_task_retire_takes_the_whole_task() {
        let mut b = BoundBuffers::default();
        b.insert(1, 1, 0, None, bound(0x1000, 0x1000));
        b.insert(1, 2, 0x40, None, bound(0x8000, 0x1000));
        b.insert(2, 1, 0, None, bound(0x1000, 0x1000));
        assert_eq!(b.retire_task(1), 2);
        assert_eq!(b.len(), 1);
        assert!(b.get(2, 1, 0, None).is_some());
    }
}
