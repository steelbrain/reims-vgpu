//! Move bytes between a device-local buffer and the guest's
//! physically-discontiguous pages with one compute dispatch instead of one
//! transfer region per run — in **either** direction.
//!
//! # One kernel, two rails
//!
//! The kernel carries no direction at all. It reads `src[run.x + i]` and writes
//! `dst[run.y + i]`, and nothing in it knows which side is guest memory. Two
//! callers use it:
//!
//! | rail | `Src` | `Dst` | planner |
//! |---|---|---|---|
//! | render writeback | detiled scratch | guest pages | [`build_run_tables`] |
//! | draw buffer gather | guest pages | pooled gather slot | [`build_gather_run_tables`] |
//!
//! What differs is only which binding has to be a *window*, and that is forced
//! rather than chosen: the imported RAMBlock is wider than
//! `maxStorageBufferRange` and the pooled slot is not, so the guest side is
//! always the windowed one. [`build_gather_run_tables`] is therefore a swap over
//! [`build_run_tables`] and not a second planner — see its doc for why that
//! matters more than tidiness.
//!
//! The writeback shipped first and its measurement is the one below; the gather
//! is the same repair against a larger population, ~427 000 transfer regions a
//! second against the writeback's ~200 per frame.
//!
//! # Why this rail exists
//!
//! [`crate::runtime::render_writeback`]'s module doc carries the measurement and
//! the reasoning; the short form is that the guest backs a surface in 16 KiB
//! physically-contiguous granules, so one 1080p writeback is ~507 runs and the
//! `Linear` plan's scatter was one `VkBufferCopy` region each. Quadrupling the
//! regions for byte-identical output halved the frame rate while `record_us` did
//! not move and `slot_us` nearly tripled, so the cost is GPU-side per-region work
//! rather than the driver's recording of it, and batching the same regions into
//! fewer calls could not have touched it.
//!
//! One dispatch has no regions at all. It reads the same detiled scratch and
//! writes the same guest bytes — `uint`-for-`uint`, with no format, row or texel
//! semantics anywhere in the kernel — so the result is byte-identical to the
//! transfer form by construction rather than by measurement.
//!
//! # The transfer form stays, and this is why
//!
//! [`super::plan_guest_linear_copies`] is still the path for a host without the
//! guest-RAM import, for a run this module refuses, and for the A/B baseline that
//! ranks the two. Nothing here may become the only way a frame reaches the guest.
//!
//! # The shape of one dispatch
//!
//! One workgroup per run, which makes `groupCountX` the run count and is why
//! nothing outside `shaders/guest_scatter.comp` names its `local_size_x` — see
//! [`super::scatter_shader`].
//!
//! `Dst` is bound at an **offset**, never at zero. A word index into a whole
//! RAMBlock does not fit a `uint`: a 16 GiB guest is exactly 2^32 words, and
//! `vm/boot-x86.sh` runs `-m 16G`. [`build_run_tables`] binds the smallest
//! alignment-respecting window covering the writeback's own destinations and
//! makes every index relative to that base, which is single-digit MiB wide.
//!
//! The run table is a host-written storage buffer rather than push constants:
//! ~200 runs of 16 bytes is past every push-constant limit. It is written into a
//! mapped staging slot and read by the shader in place, so it costs no copy
//! region either — the fourth transfer this design was first sketched with is
//! not there.

use ash::vk;

use super::context::DeviceContext;
use super::scatter_shader::GUEST_SCATTER_SPIRV;
use super::types::DrawError;
use super::vk_call::{VkCall, VkOp};
use crate::observe::Decline;

/// Bytes in the word this kernel copies in. Every offset and length a run
/// carries has to be a whole number of these or the run cannot be expressed.
pub(crate) const SCATTER_WORD: u64 = 4;

/// `uvec4` per run: source word, destination word, word count, unused.
const WORDS_PER_RUN: usize = 4;

/// Binding numbers, matching `shaders/guest_scatter.comp`'s `layout(binding =)`
/// declarations. The kernel is compiled ahead of time and embedded, so nothing
/// in the toolchain relates these to the GLSL; the source-match test in
/// [`super::scatter_shader`] is what keeps the embedded module honest about the
/// file these were read from.
const BINDING_SRC: u32 = 0;
const BINDING_DST: u32 = 1;
const BINDING_RUNS: u32 = 2;

/// The one push constant: how many runs the table holds.
///
/// Redundant with `groupCountX` by construction and kept anyway. A dispatch
/// whose grid outran its table would read past the bound range, and under
/// `robustBufferAccess` that is defined-but-arbitrary rather than a fault — so
/// it would write arbitrary words into guest RAM instead of crashing. One
/// `uint` compared per workgroup is a cheap way for that to be impossible.
const PUSH_BYTES: u32 = 4;

/// A run this device cannot express as a dispatch, so the writeback took the
/// transfer regions instead.
///
/// Every one of these is a **routing** answer and not a loss: the frame still
/// lands, byte-identically, down [`super::plan_guest_linear_copies`]. They are
/// named because the region path is the expensive one and a boot silently
/// falling back to it would read as the dispatch not paying.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScatterDecline {
    /// A run's source offset, destination offset or length is not a whole
    /// number of [`SCATTER_WORD`] bytes.
    ///
    /// Run geometry is texel-aligned, which is four bytes for the eight-bit
    /// -per-channel formats this rail serves — but not for a narrower texel, so
    /// this is a check and never an assumption.
    Unaligned { src: u64, dst: u64, len: u64 },
    /// The window the writeback lands in is wider than the driver will bind as
    /// one storage buffer.
    RangeTooWide { range: u64, max: u64 },
    /// A run reads past the end of the detiled scratch it was planned against.
    ///
    /// Two independently-derived numbers disagreeing — the scratch is sized from
    /// the window's byte count and a run's extent comes from the guest's page
    /// plan — so this is the same class as `WindowTooSmall` one layer down.
    SourceOverrun { end: u64, have: u64 },
    /// The writeback named no runs at all, so there is nothing to dispatch.
    Empty,
}

impl Decline for ScatterDecline {
    fn slug(&self) -> &'static str {
        match self {
            Self::Unaligned { .. } => "scatter_run_unaligned",
            Self::RangeTooWide { .. } => "scatter_range_too_wide",
            Self::SourceOverrun { .. } => "scatter_source_overrun",
            Self::Empty => "scatter_no_runs",
        }
    }

    fn fields(&self) -> Vec<(&'static str, String)> {
        match self {
            Self::Unaligned { src, dst, len } => vec![
                ("src", src.to_string()),
                ("dst", dst.to_string()),
                ("len", len.to_string()),
            ],
            Self::RangeTooWide { range, max } => {
                vec![("range", range.to_string()), ("max", max.to_string())]
            }
            Self::SourceOverrun { end, have } => {
                vec![("end", end.to_string()), ("have", have.to_string())]
            }
            Self::Empty => Vec::new(),
        }
    }
}

crate::observe::decline::decline_display!(ScatterDecline);

/// One run as the planner sees it, before it becomes word indices.
///
/// `dst` is absolute in the imported buffer — `bound.offset + bound.head`, the
/// same re-basing every other planner here does — because the bind offset is not
/// known until every run has been seen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ScatterRun {
    pub src: u64,
    pub dst: u64,
    pub len: u64,
}

/// The word-indexed run table for one destination buffer, and the window `Dst`
/// has to be bound over for its indices to mean anything.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct RunTable {
    /// Where to bind `Dst`. Aligned down to the device's storage-buffer offset
    /// alignment, which is what makes it a legal `VkDescriptorBufferInfo::offset`.
    pub bind_offset: u64,
    /// How much of the buffer to bind, from `bind_offset`. Never `WHOLE_SIZE`:
    /// a RAMBlock is routinely wider than `maxStorageBufferRange`, and asking
    /// for the whole of one is the invalid-descriptor form of this bug.
    pub bind_range: u64,
    /// `WORDS_PER_RUN` `u32`s per run, ready to be written into a staging slot.
    pub words: Vec<u32>,
    pub run_count: u32,
}

/// Turn one destination buffer's runs into word-indexed tables and the windows
/// to bind them against — one table per dispatch.
///
/// Pure, so it is testable on every arm including the one with no GPU — which
/// matters more here than usual, because every failure mode of this arithmetic
/// is a wrong *byte* in guest RAM rather than a crash.
///
/// `bind_align` is the device's storage-buffer offset alignment
/// ([`DeviceContext::guest_bind_offset_align`]) and `max_range` its
/// `maxStorageBufferRange`. `src_have` is what the detile actually wrote.
///
/// # Why this returns a list and not one table
///
/// One writeback's runs routinely span more than `maxStorageBufferRange`, which
/// is a `uint32_t` and so at most 4 GiB, against the 16 GiB `vm/boot-x86.sh`
/// gives the guest. The guest's allocator hands out 16 KiB granules from
/// wherever it has them, so a single 1080p surface's pages can sit either side
/// of a 4 GiB boundary — a **driven macos-13 boot measured this on 6 % of its
/// writebacks**, each of which then fell back to ~200 transfer regions.
///
/// So the span is partitioned rather than refused: the runs are swept in
/// destination order and closed into a new window whenever the next one would
/// not fit. The cost is one extra dispatch and one extra descriptor set at each
/// boundary, against the ~200 regions the fallback cost. A single run wider than
/// `max_range` still refuses, because no partition can help it.
pub(crate) fn build_run_tables(
    runs: &[ScatterRun],
    bind_align: u64,
    max_range: u64,
    src_have: u64,
) -> Result<Vec<RunTable>, ScatterDecline> {
    if runs.is_empty() {
        return Err(ScatterDecline::Empty);
    }
    for run in runs {
        if run.src % SCATTER_WORD != 0 || run.dst % SCATTER_WORD != 0 || run.len % SCATTER_WORD != 0
        {
            return Err(ScatterDecline::Unaligned {
                src: run.src,
                dst: run.dst,
                len: run.len,
            });
        }
        let end = run.src.saturating_add(run.len);
        if end > src_have {
            return Err(ScatterDecline::SourceOverrun {
                end,
                have: src_have,
            });
        }
    }
    // Destination order, which is not the order the guest's page plan hands them
    // over — that is window order. The sweep below needs the former and the
    // dispatch does not care about either, because every run carries its own two
    // indices.
    let mut order: Vec<&ScatterRun> = runs.iter().collect();
    order.sort_unstable_by_key(|r| r.dst);

    let align = bind_align.max(1);
    // One window is the overwhelmingly common case — the partition below only
    // splits at a `max_range` boundary, which a driven boot met on 6 % of its
    // writebacks — so this reserves for that and grows on the rare split.
    let mut tables: Vec<RunTable> = Vec::with_capacity(1);
    let mut open: Option<(u64, Vec<u32>, u64)> = None;
    // Every run contributes exactly [`WORDS_PER_RUN`] words and a window holds
    // at most all of them, so this is the exact upper bound and the push loop
    // below cannot reallocate. It was reallocating six times per table on a
    // ~13-run gather, growing a `Vec::new()` to fifty words, and this rail plans
    // ~21 000 tables a second — see [`super::gather_phase`].
    let words_cap = runs.len() * WORDS_PER_RUN;
    for run in order {
        let end = run.dst.saturating_add(run.len);
        // `align` is at least 16 and always a power of two, so the rounded-down
        // base stays a whole number of words and every relative index is exact.
        let fresh_base = run.dst - run.dst % align;
        if end - fresh_base > max_range {
            // Not a partitioning problem: no base this run can be indexed from
            // brings its own end inside the bound.
            return Err(ScatterDecline::RangeTooWide {
                range: end - fresh_base,
                max: max_range,
            });
        }
        let base = match &open {
            Some((base, _, _)) if end - *base <= max_range => *base,
            Some(_) => {
                // Closing here and not at the top of the next iteration, so a
                // window is emitted exactly once and the `open` slot never holds
                // two.
                let (base, words, hi) = open.take().expect("just matched Some");
                tables.push(finish_table(base, words, hi)?);
                fresh_base
            }
            None => fresh_base,
        };
        let (_, words, hi) = open.get_or_insert_with(|| (base, Vec::with_capacity(words_cap), 0));
        // Every one of these divisions is exact and every result is bounded by
        // the `max_range` check above, so the truncating casts cannot lose a bit.
        words.push((run.src / SCATTER_WORD) as u32);
        words.push(((run.dst - base) / SCATTER_WORD) as u32);
        words.push((run.len / SCATTER_WORD) as u32);
        words.push(0);
        *hi = (*hi).max(end);
    }
    if let Some((base, words, hi)) = open.take() {
        tables.push(finish_table(base, words, hi)?);
    }
    Ok(tables)
}

/// [`build_run_tables`] for the other direction: guest RAM is the **source**
/// and the device-local slot is the destination.
///
/// # Why this is a swap and not a second planner
///
/// The kernel carries no direction at all — it reads `src[run.x + i]` and writes
/// `dst[run.y + i]`, and nothing in it knows which side is guest memory. What
/// differs between a writeback and a gather is only *which binding has to be a
/// window*, and that is forced: the imported RAMBlock is wider than
/// `maxStorageBufferRange` and the pooled slot is not, so the guest side is
/// always the windowed one. [`build_run_tables`] windows `dst`, so a gather
/// hands it the runs with the two sides exchanged and exchanges the two index
/// words back afterwards.
///
/// Doing it this way rather than by copying the planner is not tidiness. That
/// function carries the partitioning that fixes a measured bug — 6 % of one
/// boot's writebacks straddled a 4 GiB boundary — and a second copy of the
/// sweep is a second place for that to be got wrong, in a direction no test on
/// the other one would catch.
///
/// `dst_have` bounds the device-local slot the runs write into, taking the place
/// of `src_have`'s bound on the scratch.
pub(crate) fn build_gather_run_tables(
    runs: &[ScatterRun],
    bind_align: u64,
    max_range: u64,
    dst_have: u64,
) -> Result<Vec<RunTable>, ScatterDecline> {
    let exchanged: Vec<ScatterRun> = runs
        .iter()
        .map(|r| ScatterRun {
            src: r.dst,
            dst: r.src,
            len: r.len,
        })
        .collect();
    let mut tables = build_run_tables(&exchanged, bind_align, max_range, dst_have)?;
    for table in &mut tables {
        for run in table.words.chunks_exact_mut(WORDS_PER_RUN) {
            run.swap(0, 1);
        }
    }
    Ok(tables)
}

/// Close one window into a table, once no further run will join it.
fn finish_table(bind_offset: u64, words: Vec<u32>, hi: u64) -> Result<RunTable, ScatterDecline> {
    let run_count =
        u32::try_from(words.len() / WORDS_PER_RUN).map_err(|_| ScatterDecline::RangeTooWide {
            range: (words.len() / WORDS_PER_RUN) as u64,
            max: u64::from(u32::MAX),
        })?;
    Ok(RunTable {
        bind_offset,
        bind_range: hi - bind_offset,
        words,
        run_count,
    })
}

/// The device's own scatter pipeline, created once and held for the device's
/// life.
///
/// Not in [`super::caches`], which is keyed by guest shader digests and bounded
/// against a guest that walks pipeline space. This one is a fixture of the
/// device: exactly one exists, nothing evicts it, and a cache miss on it would
/// be a `vkCreateComputePipelines` in the middle of a writeback.
///
/// `Copy` because it is four handles and the writeback needs it while it also
/// holds `&mut ResourcePools` for the descriptor allocation and the staging
/// write. Copying the handles is what keeps that from being a borrow conflict
/// resolved by threading the pipeline through five signatures — and the owner
/// is still the single `Option` in the pools, which is what `destroy` clears.
#[derive(Clone, Copy)]
pub(crate) struct ScatterPipeline {
    module: vk::ShaderModule,
    pub(super) dsl: vk::DescriptorSetLayout,
    pub(super) layout: vk::PipelineLayout,
    pub(super) pipeline: vk::Pipeline,
}

impl ScatterPipeline {
    /// # Safety
    ///
    /// `ctx`'s device must be live, and the returned pipeline must be destroyed
    /// with [`Self::destroy`] before it.
    pub(crate) unsafe fn create(ctx: &DeviceContext) -> Result<Self, DrawError> {
        let device = &ctx.device;
        let module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default().code(&GUEST_SCATTER_SPIRV),
                None,
            )
        }
        .map_err(|e| DrawError::VkCall(VkCall::new(VkOp::ScatterCreateShaderModule, e)))?;
        // Every binding is a plain storage buffer, which is the one descriptor
        // type `desc_arena`'s blocks are sized for in quantity.
        let bindings = [BINDING_SRC, BINDING_DST, BINDING_RUNS].map(|b| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(b)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        let dsl = match unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        } {
            Ok(dsl) => dsl,
            Err(e) => {
                unsafe { device.destroy_shader_module(module, None) };
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::ScatterCreateSetLayout,
                    e,
                )));
            }
        };
        let set_layouts = [dsl];
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(PUSH_BYTES)];
        let layout = match unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&ranges),
                None,
            )
        } {
            Ok(l) => l,
            Err(e) => {
                unsafe {
                    device.destroy_descriptor_set_layout(dsl, None);
                    device.destroy_shader_module(module, None);
                }
                return Err(DrawError::VkCall(VkCall::new(
                    VkOp::ScatterCreatePipelineLayout,
                    e,
                )));
            }
        };
        let entry = c"main";
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(entry);
        let info = [vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout)];
        let pipeline =
            match unsafe { device.create_compute_pipelines(ctx.pipeline_cache, &info, None) } {
                Ok(p) => p[0],
                Err((_, e)) => {
                    unsafe {
                        device.destroy_pipeline_layout(layout, None);
                        device.destroy_descriptor_set_layout(dsl, None);
                        device.destroy_shader_module(module, None);
                    }
                    return Err(DrawError::VkCall(VkCall::new(
                        VkOp::ScatterCreatePipeline,
                        e,
                    )));
                }
            };
        Ok(Self {
            module,
            dsl,
            layout,
            pipeline,
        })
    }

    /// Consumes the copy it is called on, because the owner is the pools' single
    /// `Option` and taking it out of there is the only way to reach this.
    ///
    /// # Safety
    ///
    /// No submitted command buffer may still reference this pipeline.
    pub(crate) unsafe fn destroy(self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.dsl, None);
            device.destroy_shader_module(self.module, None);
        }
    }

    /// Write one dispatch's three bindings into an allocated set.
    ///
    /// Both `src` and `dst` are `(buffer, offset, range)` because either of them
    /// can be the imported guest RAM, and that one has to be a window: a
    /// RAMBlock is routinely wider than `maxStorageBufferRange`. The writeback
    /// windows `dst` and the gather windows `src`, and which side is which is
    /// the only difference between the two — see [`build_gather_run_tables`].
    ///
    /// `runs` is `(buffer, offset, range)` for a different reason: a whole
    /// submission's tables share one staging slot, so a dispatch's own table
    /// starts wherever [`super::stage_run_tables`] placed it. The kernel indexes
    /// from zero of its *bound range*, which is what makes the offset the only
    /// thing that has to say which of them this dispatch reads.
    ///
    /// # Safety
    ///
    /// `set` must have been allocated from [`Self::dsl`], and every buffer must
    /// be live and cover the offset/range pair given for it. Every offset must
    /// be a multiple of the device's `minStorageBufferOffsetAlignment`.
    pub(crate) unsafe fn write_set(
        device: &ash::Device,
        set: vk::DescriptorSet,
        src: (vk::Buffer, u64, u64),
        dst: (vk::Buffer, u64, u64),
        runs: (vk::Buffer, u64, u64),
    ) {
        let infos = [
            vk::DescriptorBufferInfo::default()
                .buffer(src.0)
                .offset(src.1)
                .range(src.2),
            vk::DescriptorBufferInfo::default()
                .buffer(dst.0)
                .offset(dst.1)
                .range(dst.2),
            vk::DescriptorBufferInfo::default()
                .buffer(runs.0)
                .offset(runs.1)
                .range(runs.2),
        ];
        // An array and not a collected `Vec`: this runs once per dispatch and
        // the draw-time gather issues ~40 000 of those a second, so a heap
        // allocation here is 40 000 a second for three elements whose count is
        // fixed by the layout.
        let writes = [
            (BINDING_SRC, &infos[0]),
            (BINDING_DST, &infos[1]),
            (BINDING_RUNS, &infos[2]),
        ]
        .map(|(binding, info)| {
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(info))
        });
        unsafe { device.update_descriptor_sets(&writes, &[]) };
    }

    /// Bind the kernel, once, ahead of a run of [`Self::dispatch`]es.
    ///
    /// Split out because the handle is the same on every dispatch and this rail
    /// issues ~21 000 of them a second: `record` is 39 % of a dispatch's CPU
    /// cost and this is one of its four driver calls, so a draw's 1.4 dispatches
    /// paid for 1.4 binds of one pipeline. Whoever binds is also responsible for
    /// there being no *other* pipeline bound to
    /// [`vk::PipelineBindPoint::COMPUTE`] between the bind and the last
    /// dispatch — which holds at both call sites, because each records its
    /// dispatches in one uninterrupted loop.
    ///
    /// # Safety
    ///
    /// `cb` must be recording.
    pub(crate) unsafe fn bind(&self, device: &ash::Device, cb: vk::CommandBuffer) {
        unsafe { device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline) };
    }

    /// Bind the set, push the run count and dispatch one workgroup per run.
    ///
    /// # Safety
    ///
    /// `cb` must be recording with this pipeline bound by [`Self::bind`], and
    /// `set` must name buffers live for the whole of the submission `cb`
    /// belongs to.
    pub(crate) unsafe fn dispatch(
        &self,
        device: &ash::Device,
        cb: vk::CommandBuffer,
        set: vk::DescriptorSet,
        run_count: u32,
    ) {
        unsafe {
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[set],
                &[],
            );
            device.cmd_push_constants(
                cb,
                self.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &run_count.to_ne_bytes(),
            );
            // One workgroup per run: the kernel strides its own run by its own
            // `local_size_x`, so no size from this side enters the arithmetic.
            device.cmd_dispatch(cb, run_count, 1, 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALIGN: u64 = 16;
    const MAX: u64 = u32::MAX as u64;

    fn run(src: u64, dst: u64, len: u64) -> ScatterRun {
        ScatterRun { src, dst, len }
    }

    /// The single table a set of runs that fits one window must produce.
    fn one(mut tables: Vec<RunTable>) -> RunTable {
        assert_eq!(tables.len(), 1, "these runs fit one bound window");
        tables.remove(0)
    }

    /// A gather is the writeback with the two sides exchanged, so the window it
    /// binds must land on the **guest** offsets and the device-local indices
    /// must stay absolute. Getting the swap backwards reads the guest's RAM at
    /// a slot-sized offset and writes the slot at a RAMBlock-sized one, which is
    /// wrong bytes rather than a crash — so it is asserted against the planner
    /// it delegates to rather than against a hand-written expectation.
    #[test]
    fn a_gather_windows_the_guest_side_and_leaves_the_slot_absolute() {
        let high = 8 * 1024 * 1024 * 1024u64;
        // Guest source high in the RAMBlock, device-local destination low.
        let g = one(build_gather_run_tables(
            &[run(high, 0, 4096), run(high + 65536, 4096, 4096)],
            ALIGN,
            MAX,
            1 << 20,
        )
        .expect("aligned runs plan"));
        assert_eq!(g.bind_offset, high, "the window is on the guest side");
        assert_eq!(g.bind_range, 65536 + 4096);
        assert_eq!(g.run_count, 2);
        // src index relative to the guest window, dst index absolute in the slot.
        assert_eq!(&g.words[0..3], &[0, 0, 1024]);
        assert_eq!(&g.words[4..7], &[65536 / 4, 4096 / 4, 1024]);
    }

    /// The same runs planned both ways must agree on everything but which two
    /// words hold which index — that identity is what makes the delegation safe,
    /// and it is the thing a future edit to either function could break.
    #[test]
    fn a_gather_table_is_the_writeback_table_with_its_indices_exchanged() {
        let rs = [run(0x4000_0000, 0, 8192), run(0x4001_0000, 8192, 8192)];
        let flipped: Vec<ScatterRun> = rs
            .iter()
            .map(|r| ScatterRun {
                src: r.dst,
                dst: r.src,
                len: r.len,
            })
            .collect();
        let g = one(build_gather_run_tables(&rs, ALIGN, MAX, 1 << 20).expect("gather plans"));
        let w = one(build_run_tables(&flipped, ALIGN, MAX, 1 << 20).expect("writeback plans"));
        assert_eq!(g.bind_offset, w.bind_offset);
        assert_eq!(g.bind_range, w.bind_range);
        assert_eq!(g.run_count, w.run_count);
        for (gr, wr) in g
            .words
            .chunks_exact(WORDS_PER_RUN)
            .zip(w.words.chunks_exact(WORDS_PER_RUN))
        {
            assert_eq!(gr[0], wr[1], "src index is the writeback's dst index");
            assert_eq!(gr[1], wr[0], "and the other way round");
            assert_eq!(gr[2], wr[2], "length is a length either way");
        }
    }

    /// A gather straddling the 4 GiB `maxStorageBufferRange` wall has to
    /// partition on the guest side, exactly as a writeback does — the bug that
    /// cost 6 % of one boot's writebacks ~200 transfer regions each, reached
    /// through the delegation rather than reimplemented behind it.
    #[test]
    fn a_gather_spanning_the_storage_range_wall_splits_into_two_windows() {
        let max = 4 * 1024 * 1024 * 1024u64;
        let tables = build_gather_run_tables(
            &[run(0, 0, 4096), run(max + 4096, 4096, 4096)],
            ALIGN,
            max,
            1 << 20,
        )
        .expect("aligned runs plan");
        assert_eq!(tables.len(), 2, "no one window can bind both guest offsets");
        assert_eq!(tables.iter().map(|t| t.run_count).sum::<u32>(), 2);
    }

    /// The whole point of the offset bind: a destination at the top of a 16 GiB
    /// RAMBlock still produces small indices, where an index from zero would
    /// have overflowed a `uint`.
    ///
    /// 16 GiB is `vm/boot-x86.sh`'s own `-m`, and it is exactly 2^32 words — so
    /// a word index from buffer byte zero has *no* headroom at the top of the
    /// block and a byte index has none from 4 GiB up. This test sits one word
    /// past the first of those two walls.
    #[test]
    fn indices_are_relative_to_the_bound_window_not_to_the_buffer() {
        let high = 16 * 1024 * 1024 * 1024u64;
        let mut t = build_run_tables(
            &[run(0, high, 16384), run(16384, high + 65536, 16384)],
            ALIGN,
            MAX,
            1 << 20,
        )
        .expect("aligned runs plan");
        assert_eq!(t.len(), 1, "one window covers both");
        let t = t.remove(0);
        assert_eq!(t.bind_offset, high, "already aligned, so bound where it is");
        assert_eq!(t.bind_range, 65536 + 16384);
        assert_eq!(t.run_count, 2);
        assert_eq!(t.words[0..4], [0, 0, 4096, 0]);
        assert_eq!(t.words[4..8], [4096, 16384, 4096, 0]);
        // An index from buffer byte zero would not have fitted at all.
        assert!(high / SCATTER_WORD > u64::from(u32::MAX));
    }

    /// The bound the driver states is a `uint32_t`, so a range that passes it
    /// can never produce a word index that does not fit a `uint` — which is why
    /// [`build_run_tables`] carries one check and not two.
    #[test]
    fn a_range_the_driver_admits_always_has_a_word_index_that_fits() {
        assert!(u64::from(u32::MAX) / SCATTER_WORD <= u64::from(u32::MAX));
    }

    #[test]
    fn the_bind_offset_rounds_down_to_the_alignment_and_indices_absorb_it() {
        let t = one(build_run_tables(&[run(0, 1000, 8)], 16, MAX, 1 << 20).expect("plan"));
        assert_eq!(t.bind_offset, 992, "1000 rounded down to a multiple of 16");
        assert_eq!(t.bind_range, 1008 - 992);
        // The 8 bytes the rounding put in front become 2 words of index.
        assert_eq!(t.words[1], 2);
    }

    /// A run the kernel cannot express must refuse rather than round, because
    /// rounding here writes the wrong guest bytes and reports success.
    #[test]
    fn a_run_that_is_not_a_whole_number_of_words_is_refused() {
        for bad in [run(1, 64, 16), run(0, 65, 16), run(0, 64, 15)] {
            let err = build_run_tables(&[bad], ALIGN, MAX, 1 << 20)
                .expect_err("a sub-word run must not plan");
            assert!(matches!(err, ScatterDecline::Unaligned { .. }), "{err:?}");
        }
    }

    /// A span wider than the driver binds splits into two dispatches rather than
    /// falling back to ~200 transfer regions.
    ///
    /// This is not a corner: `maxStorageBufferRange` is a `uint32_t` and the
    /// guest gets 16 GiB, and a driven macos-13 boot straddled the boundary on
    /// 6 % of its writebacks.
    #[test]
    fn a_span_wider_than_the_driver_binds_splits_into_two_windows() {
        let far = 4 * 1024 * 1024 * 1024u64;
        let tables = build_run_tables(&[run(0, 0, 16), run(16, far, 16)], ALIGN, MAX, 1 << 20)
            .expect("a straddling span partitions rather than refusing");
        assert_eq!(tables.len(), 2, "one window either side of the bound");
        assert_eq!(tables[0].bind_offset, 0);
        assert_eq!(tables[1].bind_offset, far);
        for t in &tables {
            assert_eq!(t.run_count, 1);
            assert!(t.bind_range <= MAX, "each window is inside the bound");
            // Index zero in its own window, which is the whole point of a base.
            assert_eq!(t.words[1], 0);
        }
        // The source words are preserved across the split, so no run lost its
        // half of the copy.
        assert_eq!(tables[0].words[0], 0);
        assert_eq!(tables[1].words[0], 4);
    }

    /// A single run no base can bring inside the bound is a refusal, because no
    /// partition can help it.
    #[test]
    fn one_run_wider_than_the_driver_binds_is_refused() {
        // Word-aligned, so the refusal is the range and not the alignment.
        let err = build_run_tables(&[run(0, 0, MAX + 1 + 16)], ALIGN, MAX, u64::MAX)
            .expect_err("a run past maxStorageBufferRange must not plan");
        assert!(
            matches!(err, ScatterDecline::RangeTooWide { .. }),
            "{err:?}"
        );
    }

    /// Every run must land in exactly one window, and the windows must tile the
    /// runs — a sweep that dropped the last open window, or emitted one twice,
    /// loses or duplicates guest bytes and nothing downstream could tell.
    #[test]
    fn a_partitioned_sweep_places_every_run_exactly_once() {
        let step = 3 * 1024 * 1024 * 1024u64;
        let runs: Vec<_> = (0..5u64).map(|i| run(i * 16, i * step, 16)).collect();
        let tables = build_run_tables(&runs, ALIGN, MAX, 1 << 20).expect("plan");
        assert!(tables.len() > 1, "5 runs 3 GiB apart cannot be one window");
        let placed: u32 = tables.iter().map(|t| t.run_count).sum();
        assert_eq!(placed as usize, runs.len(), "every run placed once");
        // Reconstruct each run's absolute destination from its window and check
        // the set matches, which catches a run rebased against the wrong base.
        let mut seen: Vec<u64> = tables
            .iter()
            .flat_map(|t| {
                t.words
                    .chunks_exact(WORDS_PER_RUN)
                    .map(move |w| t.bind_offset + u64::from(w[1]) * SCATTER_WORD)
            })
            .collect();
        seen.sort_unstable();
        let mut want: Vec<u64> = runs.iter().map(|r| r.dst).collect();
        want.sort_unstable();
        assert_eq!(seen, want);
    }

    /// The scratch bound is checked from this side because the descriptor's own
    /// range cannot catch it: `Src` is bound whole, so an over-long run reads
    /// defined-but-wrong words and scatters them into the guest.
    #[test]
    fn a_run_reading_past_the_scratch_is_refused() {
        let err = build_run_tables(&[run(4096, 0, 4096)], ALIGN, MAX, 4096)
            .expect_err("a run past the scratch must not plan");
        assert!(
            matches!(err, ScatterDecline::SourceOverrun { .. }),
            "{err:?}"
        );
    }

    #[test]
    fn no_runs_is_refused_rather_than_dispatched_empty() {
        assert_eq!(
            build_run_tables(&[], ALIGN, MAX, 1 << 20),
            Err(ScatterDecline::Empty)
        );
    }

    /// The table's words are what the guest's pages end up holding, so the
    /// mapping from run to `uvec4` is asserted directly rather than through the
    /// two properties above.
    ///
    /// The table is in **destination** order, which is not the order the runs
    /// arrive in — the sweep that partitions wide spans needs that ordering, and
    /// the dispatch does not care because every run carries both its indices.
    #[test]
    fn every_run_becomes_four_words_carrying_its_own_two_indices() {
        let runs = [run(0, 4096, 64), run(64, 8192, 128), run(192, 100, 32)];
        let t = one(build_run_tables(&runs, ALIGN, MAX, 1 << 20).expect("plan"));
        assert_eq!(t.words.len(), runs.len() * WORDS_PER_RUN);
        assert_eq!(t.run_count as usize, runs.len());
        let mut want: Vec<_> = runs.iter().map(|r| (r.dst, r.src, r.len)).collect();
        want.sort_unstable();
        let got: Vec<_> = t
            .words
            .chunks_exact(WORDS_PER_RUN)
            .map(|w| {
                (
                    t.bind_offset + u64::from(w[1]) * SCATTER_WORD,
                    u64::from(w[0]) * SCATTER_WORD,
                    u64::from(w[2]) * SCATTER_WORD,
                )
            })
            .collect();
        assert_eq!(got, want, "each run's own (dst, src, len), in dst order");
    }

    /// The bound window has to cover every run, including one whose destination
    /// is neither the lowest nor the highest seen so far — the guest's runs
    /// arrive in window order, which is not destination order.
    #[test]
    fn the_bound_window_covers_runs_arriving_out_of_destination_order() {
        let t = one(build_run_tables(
            &[run(0, 8192, 16), run(16, 128, 16), run(32, 4096, 16)],
            ALIGN,
            MAX,
            1 << 20,
        )
        .expect("plan"));
        assert_eq!(t.bind_offset, 128, "the lowest destination, aligned down");
        assert_eq!(t.bind_range, 8192 + 16 - 128, "up to the highest end");
        for i in 0..3 {
            let w = &t.words[i * WORDS_PER_RUN..][..WORDS_PER_RUN];
            let end = u64::from(w[1]) * SCATTER_WORD + u64::from(w[2]) * SCATTER_WORD;
            assert!(end <= t.bind_range, "run {i} lands inside the bound window");
        }
    }
}
