//! The rt platform's compute encoders ([`RtComputeEncoder`]): rt's, which records each dispatch
//! into its command buffer at once, with the barriers its hazards need.
//!
//! - **Each command is rt's call.** A list (`setBuffers:offsets:withRange:`,
//!   `setTextures:withRange:`, `setSamplerStates:withRange:` and its form with LOD clamps,
//!   `useResources:count:usage:`, `useHeaps:count:`) is one call per entry, in its order. The
//!   dispatch type is rt's: serial for `computeCommandEncoder`, the descriptor's for
//!   `computeCommandEncoderWithDescriptor:`.
//! - **A pipeline state the encoder sets is kept by its command buffer** until the command buffer
//!   completes ([`Kept`]), as rt keeps its own: the sampler indices a state numbers
//!   (`super::pipeline`) go back only once no work recorded with it can read them.
//! - **Attribute strides** (`setBufferOffset:attributeStride:atIndex:`,
//!   `setBuffers:offsets:attributeStrides:withRange:`) bind as the calls without them: only a
//!   `[[stage_in]]` attribute a kernel reads by thread position through a layout of
//!   `MTLBufferLayoutStrideDynamic` takes a stride from the encoder, and rt refuses such a kernel
//!   as its state is made (`Gap::StageIn`, kind `DynamicStride`).
//! - **The stage-in region**: `setStageInRegion:` is rt's, the region the encoder's later
//!   dispatches fetch a kernel's `[[stage_in]]` attributes by.
//!   `setStageInRegionWithIndirectBuffer:indirectBufferOffset:` is rt's call too, which refuses it
//!   (`Gap::StageIn`, kind `IndirectRegion`; `StageIn-IndirectRegion` in the boot diagnostics),
//!   counted (`rt-refused-gap`): it encodes nothing, and the encoder's region stays the one set
//!   before it, or none.
//! - **Barriers**: `memoryBarrierWithScope:` and `memoryBarrierWithResources:count:` are rt's one
//!   global barrier, whatever the scope or the resources.
//! - **Nothing to record**: `setImageblockWidth:height:` (rt makes no kernel that reads an
//!   imageblock), `insertCompressedTextureReinterpretationFlush` (rt keeps no lossless
//!   compression), and a nil visible function table (rt binds none).
//! - **Each dispatch rt encodes is counted by its form**, with one relaxed add, for the counts
//!   line, where rt counts none: direct (`dispatchThreads:threadsPerThreadgroup:` and
//!   `dispatchThreadgroups:threadsPerThreadgroup:`, [`DISPATCHES_DIRECT_CODE`]) and indirect
//!   (`dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:`,
//!   [`DISPATCHES_INDIRECT_CODE`]), amounts on the line ([`fold_dispatches`]). A dispatch counts
//!   once, however many threads or threadgroups it runs, as rt records it, after any wait for its
//!   stage's fill; one that records nothing (a refusal, a skip, a form rt has not built) counts
//!   none.
//! - **Unbuilt**: an indirect `dispatchThreads` ([`Unbuilt::ComputeCommand`]), and command
//!   predication (`encodeStart*`, `encodeEnd*`, [`Unbuilt::Predication`]): each encodes nothing,
//!   counted, and an `encodeEnd*` answers NO, as Metal's answers with nothing open.
//! - **No usage**: `useResources:count:usage:` of usage 0 declares nothing, and is no refusal: no
//!   line, no count (a nil resource of a concurrent encoder's is still Metal's refusal, below).
//!   Measured with `useResource:usage:` and the render encoder's `useResource:usage:stages:` on a
//!   Radeon Pro 5300M and a UHD 630 under macOS 26.7.1: Metal's API validation
//!   (`MTL_DEBUG_LAYER=1`) accepts usage 0 with no message and ends the process at usage 8
//!   ("usage (0x8) has an invalid value"), its GPU validation (`MTL_SHADER_VALIDATION=1`) finds
//!   the resources either names not resident, as with no declaration, and without validation both
//!   run as no declaration.
//! - **Metal's refusals**, each encoding nothing, counted (`rt-refused-metal`): a nil indirect
//!   buffer and a nil resource in a concurrent encoder's `useResources:count:usage:`, at each of
//!   which Metal ends the process; a dispatch with no pipeline state set, or with nil set since; a
//!   usage with a bit past sample (4); an index, a length or a size past rt's 32 bits; a list of
//!   another length than its range; and a command after `endEncoding`. A nil entry of a serial
//!   encoder's resources, or of a heap list, is skipped, as Metal skips it.
//! - **A stage's fill holds a dispatch.** Where rt refuses a dispatch for a stage whose compile is
//!   not in its store, counted (`super::stage`, `UNFILLED_AT_DISPATCH`), the dispatch waits for
//!   that stage's fill ([`held`]) and is made once more after the fill filled the store. After a
//!   fill that failed or breached it is skipped, encoding nothing, counted under its end's code
//!   (`super::stage::skipped`), and its command buffer fails with Metal's `Internal` (1)
//!   ([`Kept::fail_internal`]); after a cancelled fill it is rt's refusal.
//!
//! A command rt refuses encodes nothing, counted under the refusal's kind; in a list, the entries
//! before the refused one are bound.

use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use metal2vulkan_runtime as rt;

use crate::core::platform::blit::ResourceRef;
use crate::core::platform::compute::{ComputeCmd, ComputeEncoder};
use crate::core::platform::stage::{Refused, StageFills, held};
use crate::core::platform::types::{MtlSize, NsRange};
use crate::core::platform::{Handle, ObjId, Platform};
use crate::core::raise::{OnFault, RaiseResult};
use crate::core::sync::Mutex;

use super::blit::{NIL, Stop, report};
use super::command::{Kept, OpenMark};
use super::copied::Used;
use super::diag::Use;
use super::fault::{EncoderKind, EncoderNames, Naming};
use super::stage::{StageKey, skipped, unfilled_at_dispatch};
use super::unbuilt::{Unbuilt, refuse};
use super::{RtObj, RtPlatform};

/// An index, a length or a size past rt's 32 bits.
const WIDE: Stop = Stop::Metal("an index, a length or a size past 32 bits");

/// The code of the direct dispatches rt encoded, `dispatchThreads:threadsPerThreadgroup:` and
/// `dispatchThreadgroups:threadsPerThreadgroup:`, an amount on the counts line
/// ([`fold_dispatches`]).
pub const DISPATCHES_DIRECT_CODE: &str = "rt-dispatches-direct";

/// The code of the indirect dispatches rt encoded,
/// `dispatchThreadgroupsWithIndirectBuffer:indirectBufferOffset:threadsPerThreadgroup:`, an amount
/// on the counts line ([`fold_dispatches`]).
pub const DISPATCHES_INDIRECT_CODE: &str = "rt-dispatches-indirect";

/// The direct dispatches rt encoded since the last fold ([`fold_dispatches`]).
static DIRECT_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// The indirect dispatches rt encoded since the last fold ([`fold_dispatches`]).
static INDIRECT_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Counts by code (`crate::answer::count_kind_by`) the dispatches rt encoded since the last fold,
/// direct ([`DISPATCHES_DIRECT_CODE`]) and indirect ([`DISPATCHES_INDIRECT_CODE`]), so the counts
/// line carries their totals; `super::device::fold_metrics` folds them.
pub(super) fn fold_dispatches() {
    let direct = DIRECT_DISPATCHES.swap(0, Ordering::Relaxed);
    let indirect = INDIRECT_DISPATCHES.swap(0, Ordering::Relaxed);
    crate::answer::count_kind_by(DISPATCHES_DIRECT_CODE, direct);
    crate::answer::count_kind_by(DISPATCHES_INDIRECT_CODE, indirect);
}

/// `id<MTLComputeCommandEncoder>` on metal2vulkan-runtime.
pub struct RtComputeEncoder {
    /// rt's encoder and what the encoder set.
    open: Mutex<Open>,
    /// Its command buffer's mark of it open, until it ends.
    mark: OpenMark,
    /// What its command buffer keeps until it completes.
    kept: Kept,
    /// Whether its dispatch type is concurrent.
    concurrent: bool,
    /// Its device's stage fills, which its dispatches wait for.
    fills: Arc<StageFills<StageKey>>,
}

/// An encoder's rt encoder, until `endEncoding` ends it, its pipeline state, and what it names.
struct Open {
    /// rt's encoder.
    rt: Option<rt::ComputeCommandEncoder>,
    /// The pipeline state set last, which the command buffer keeps: none before one is set, or
    /// once nil is.
    pipeline: Option<ObjId>,
    /// What its commands name, for a lost device's lines (`super::fault`'s docs, "The command
    /// buffers in flight"), which join its command buffer's as it ends or goes.
    naming: Naming,
}

impl RtComputeEncoder {
    /// The encoder over rt's `encoder`, open as `mark` marks it, its pipeline states kept in
    /// `kept`; `concurrent` where its dispatch type is; its dispatches wait for `fills`.
    pub fn new(
        encoder: rt::ComputeCommandEncoder,
        mark: OpenMark,
        kept: Kept,
        concurrent: bool,
        fills: Arc<StageFills<StageKey>>,
    ) -> RtComputeEncoder {
        let names = EncoderNames::new(EncoderKind::Compute);
        RtComputeEncoder {
            open: Mutex::new(Open {
                rt: Some(encoder),
                pipeline: None,
                naming: Naming::new(kept.encoded(), names),
            }),
            mark,
            kept,
            concurrent,
            fills,
        }
    }

    /// A dispatch at `sel` that [`held`] did not make, `refused`: skipped where its stage's fill
    /// failed or breached ([`skipped`]), its command buffer failing with `Internal` (1)
    /// ([`Kept::fail_internal`]); otherwise rt's refusal.
    fn unmade(&self, sel: &'static str, refused: Refused<rt::Error>) -> Stop {
        if let Some(end) = refused.fill
            && skipped(sel, end)
        {
            self.kept.fail_internal();
            return Stop::Counted;
        }
        Stop::Rt(refused.error)
    }

    /// Encodes `cmd` but `endEncoding`, which the caller ends.
    fn command(&self, open: &mut Open, cmd: ComputeCmd<'_, RtPlatform>) -> Result<(), Stop> {
        let sel = cmd.selector();
        let Open {
            rt: encoder,
            pipeline,
            ..
        } = open;
        let Some(e) = encoder.as_mut() else {
            return Err(Stop::Metal("a command after endEncoding"));
        };
        match cmd {
            ComputeCmd::DispatchThreadgroups {
                threadgroups_per_grid,
                threads_per_threadgroup,
            } => {
                set(*pipeline)?;
                let groups = size(threadgroups_per_grid)?;
                let threads = size(threads_per_threadgroup)?;
                let dispatch = || e.dispatch_threadgroups(groups, threads);
                held(&self.fills, dispatch, unfilled_at_dispatch)
                    .map_err(|refused| self.unmade(sel, refused))?;
                DIRECT_DISPATCHES.fetch_add(1, Ordering::Relaxed);
            }
            ComputeCmd::DispatchThreadgroupsIndirect {
                indirect_buffer,
                indirect_buffer_offset,
                threads_per_threadgroup,
            } => {
                let buffer = indirect_buffer.ok_or(NIL)?;
                set(*pipeline)?;
                let threads = size(threads_per_threadgroup)?;
                let offset = indirect_buffer_offset;
                let dispatch =
                    || e.dispatch_threadgroups_with_indirect_buffer(buffer.rt(), offset, threads);
                held(&self.fills, dispatch, unfilled_at_dispatch)
                    .map_err(|refused| self.unmade(sel, refused))?;
                INDIRECT_DISPATCHES.fetch_add(1, Ordering::Relaxed);
            }
            ComputeCmd::DispatchThreads {
                threads_per_grid,
                threads_per_threadgroup,
            } => {
                set(*pipeline)?;
                let grid = size(threads_per_grid)?;
                let threads = size(threads_per_threadgroup)?;
                let dispatch = || e.dispatch_threads(grid, threads);
                held(&self.fills, dispatch, unfilled_at_dispatch)
                    .map_err(|refused| self.unmade(sel, refused))?;
                DIRECT_DISPATCHES.fetch_add(1, Ordering::Relaxed);
            }
            ComputeCmd::DispatchThreadsIndirect { .. } => {
                return Err(Stop::Unbuilt(Unbuilt::ComputeCommand));
            }
            ComputeCmd::EncodeStartDoWhile
            | ComputeCmd::EncodeStartElse
            | ComputeCmd::EncodeStartIf { .. }
            | ComputeCmd::EncodeStartWhile { .. } => {
                return Err(Stop::Unbuilt(Unbuilt::Predication));
            }
            ComputeCmd::EndEncoding
            | ComputeCmd::CompressedTextureReinterpretationFlush
            | ComputeCmd::SetImageblockWidth { .. } => {}
            ComputeCmd::ExecuteCommandsIndirect { .. }
            | ComputeCmd::ExecuteCommandsInBuffer { .. } => {
                return Err(Stop::Metal(
                    "an indirect command buffer of draws executed in a compute encoder",
                ));
            }
            ComputeCmd::MemoryBarrierWithResources { .. }
            | ComputeCmd::MemoryBarrierWithScope { .. } => e.memory_barrier(),
            ComputeCmd::SetBufferOffset { offset, index }
            | ComputeCmd::SetBufferOffsetAttributeStride { offset, index, .. } => {
                e.set_buffer_offset(offset, small(index)?)?;
            }
            ComputeCmd::SetBuffersAttributeStrides {
                buffers,
                offsets,
                strides,
                range,
            } => {
                let at = slots(range, &[buffers.len(), offsets.len(), strides.len()])?;
                for ((buffer, &offset), index) in buffers.iter().zip(offsets).zip(at) {
                    e.set_buffer(buffer.map(|b| b.rt()), offset, index)?;
                }
            }
            ComputeCmd::SetBuffers {
                buffers,
                offsets,
                range,
            } => {
                let at = slots(range, &[buffers.len(), offsets.len()])?;
                for ((buffer, &offset), index) in buffers.iter().zip(offsets).zip(at) {
                    e.set_buffer(buffer.map(|b| b.rt()), offset, index)?;
                }
            }
            ComputeCmd::SetComputePipelineState { state } => {
                let Some(state) = state else {
                    *pipeline = None;
                    return Ok(());
                };
                e.set_compute_pipeline_state(state.rt())?;
                let id = state.id();
                if *pipeline != Some(id) {
                    self.kept.keep(id, Box::new(state.clone_ref()));
                    *pipeline = Some(id);
                }
            }
            ComputeCmd::SetSamplerStatesLodClamps {
                samplers,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => {
                let lengths = [samplers.len(), lod_min_clamps.len(), lod_max_clamps.len()];
                let at = slots(range, &lengths)?;
                let clamps = lod_min_clamps.iter().zip(lod_max_clamps);
                for ((sampler, (&min, &max)), index) in samplers.iter().zip(clamps).zip(at) {
                    let sampler = sampler.map(|s| s.rt());
                    e.set_sampler_state_with_lod_clamps(sampler, min, max, index)?;
                }
            }
            ComputeCmd::SetSamplerStates { samplers, range } => {
                let at = slots(range, &[samplers.len()])?;
                for (sampler, index) in samplers.iter().zip(at) {
                    e.set_sampler_state(sampler.map(|s| s.rt()), index)?;
                }
            }
            ComputeCmd::SetStageInRegion { region } => {
                let o = region.origin;
                e.set_stage_in_region(rt::Region {
                    origin: [small(o.x)?, small(o.y)?, small(o.z)?],
                    size: size(region.size)?,
                });
            }
            ComputeCmd::SetStageInRegionIndirect {
                indirect_buffer,
                indirect_buffer_offset,
            } => {
                e.set_stage_in_region_with_indirect_buffer(
                    indirect_buffer.rt(),
                    indirect_buffer_offset,
                )?;
            }
            ComputeCmd::SetTexture { texture, index } => {
                e.set_texture(texture.map(|t| t.rt()), small(index)?)?;
                if let Some(texture) = texture {
                    self.kept.used(texture, Use::Sampled);
                }
            }
            ComputeCmd::SetTextures { textures, range } => {
                let at = slots(range, &[textures.len()])?;
                for (texture, index) in textures.iter().zip(at) {
                    e.set_texture(texture.map(|t| t.rt()), index)?;
                    if let Some(texture) = texture {
                        self.kept.used(texture, Use::Sampled);
                    }
                }
            }
            ComputeCmd::SetThreadgroupMemoryLength { length, index } => {
                e.set_threadgroup_memory_length(small(length)?, small(index)?)?;
            }
            ComputeCmd::SetVisibleFunctionTable { function_table, .. } => {
                if let Some(table) = function_table {
                    table.unreachable();
                }
            }
            ComputeCmd::SetVisibleFunctionTables {
                function_tables, ..
            } => {
                if let Some(table) = function_tables.iter().flatten().next() {
                    table.unreachable();
                }
            }
            ComputeCmd::UpdateFence { fence } => e.update_fence(fence)?,
            ComputeCmd::WaitForFence { fence } => e.wait_for_fence(fence)?,
            ComputeCmd::UseHeaps { heaps } => {
                for heap in heaps.iter().flatten() {
                    e.use_heap(heap.block_id())?;
                }
            }
            ComputeCmd::UseResources { resources, usage } => {
                let usage = resource_usage(usage)?;
                if self.concurrent && resources.iter().any(Option::is_none) {
                    return Err(Stop::Metal(
                        "a nil resource in a concurrent encoder, which ends Metal's process",
                    ));
                }
                if let Some(usage) = usage {
                    for resource in resources.iter().flatten() {
                        e.use_resource(declared(&self.kept, resource), usage)?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// `Ok` where a pipeline state is set ([`Open::pipeline`]); a dispatch without one is Metal's
/// refusal.
fn set(pipeline: Option<ObjId>) -> Result<(), Stop> {
    match pipeline {
        Some(_) => Ok(()),
        None => Err(Stop::Metal("a dispatch with no pipeline state set")),
    }
}

/// `value` as rt's 32 bits.
pub(super) fn small(value: u64) -> Result<u32, Stop> {
    u32::try_from(value).map_err(|_| WIDE)
}

/// A grid's or a threadgroup's size as rt's.
fn size(size: MtlSize) -> Result<[u32; 3], Stop> {
    Ok([small(size.width)?, small(size.height)?, small(size.depth)?])
}

/// The argument table indices of `range`, whose length each of the lists bound with it has
/// (`lengths`); another length, or an index past rt's 32 bits, is Metal's refusal.
pub(super) fn slots(range: NsRange, lengths: &[usize]) -> Result<Range<u32>, Stop> {
    let length = |n: usize| u64::try_from(n).ok();
    if lengths.iter().any(|&n| length(n) != Some(range.length)) {
        return Err(Stop::Metal("a list of another length than its range"));
    }
    let end = range.location.checked_add(range.length).ok_or(WIDE)?;
    Ok(small(range.location)?..small(end)?)
}

/// `MTLResourceUsage` as rt's: a value of read (1), write (2) and sample (4), or none for 0, which
/// declares nothing (the module's docs, "No usage"); a bit past sample is Metal's refusal. The
/// render encoder's `useResources:` takes it too (`super::render`).
pub(super) fn resource_usage(usage: u64) -> Result<Option<rt::ResourceUsage>, Stop> {
    if usage & !7 != 0 {
        return Err(Stop::Metal("a resource usage no Metal value is"));
    }
    let bits = [
        (1, rt::ResourceUsage::READ),
        (2, rt::ResourceUsage::WRITE),
        (4, rt::ResourceUsage::SAMPLE),
    ];
    Ok((bits.into_iter())
        .filter(|&(bit, _)| usage & bit != 0)
        .map(|(_, named)| named)
        .reduce(|a, b| a | b))
}

/// rt's resource of `resource`, which a compute or render encoder's `useResources:` declares to the
/// command buffer that keeps `kept`: a texture's declaration is a use of it where it is tiled
/// (`super::tiled`).
pub(super) fn declared<'r>(
    kept: &Kept,
    resource: &'r ResourceRef<'_, RtPlatform>,
) -> rt::Resource<'r> {
    match resource {
        ResourceRef::Buffer(b) => rt::Resource::Buffer(b.rt()),
        ResourceRef::Texture(t) => {
            kept.takes().used(t);
            rt::Resource::Texture(t.rt())
        }
        ResourceRef::Icb(icb) => rt::Resource::IndirectCommandBuffer(icb.rt()),
    }
}

/// Notes the copies `cmd` uses in `used` ([`Used`]): those the buffers it sets, the heaps and
/// resources it declares and its indirect buffers lie over, each in a use that can write it but
/// an indirect or condition buffer's and a resource's declared for reading alone. Its textures are
/// noted as they bind ([`Kept::used`]).
fn note_copies(used: &Used, cmd: &ComputeCmd<'_, RtPlatform>) {
    if !super::copied::any() {
        return;
    }
    match cmd {
        ComputeCmd::DispatchThreadgroupsIndirect {
            indirect_buffer: buffer,
            ..
        }
        | ComputeCmd::DispatchThreadsIndirect {
            indirect_buffer: buffer,
            ..
        }
        | ComputeCmd::EncodeStartIf { buffer, .. }
        | ComputeCmd::EncodeStartWhile { buffer, .. } => {
            buffer.iter().for_each(|b| used.buffer_read(b));
        }
        ComputeCmd::ExecuteCommandsIndirect {
            indirect_range_buffer: buffer,
            ..
        }
        | ComputeCmd::SetStageInRegionIndirect {
            indirect_buffer: buffer,
            ..
        } => used.buffer_read(buffer),
        ComputeCmd::SetBuffers { buffers, .. }
        | ComputeCmd::SetBuffersAttributeStrides { buffers, .. } => {
            buffers.iter().flatten().for_each(|b| used.buffer(b));
        }
        ComputeCmd::UseHeaps { heaps } => heaps.iter().flatten().for_each(|h| used.heap(h)),
        ComputeCmd::UseResources { resources, usage } => {
            resources
                .iter()
                .flatten()
                .for_each(|r| used.resource(r, *usage));
        }
        _ => {}
    }
}

/// Notes in `names` what `cmd` names (`super::fault`'s docs, "The command buffers in flight"): the
/// buffers it binds, and the heaps and resources it declares, by their guest ids.
fn note_names(names: &mut EncoderNames, cmd: &ComputeCmd<'_, RtPlatform>) {
    match cmd {
        ComputeCmd::SetBuffers { buffers, .. }
        | ComputeCmd::SetBuffersAttributeStrides { buffers, .. } => {
            for buffer in buffers.iter().flatten() {
                names.bound_buffers.add(buffer.named().id());
            }
        }
        ComputeCmd::UseHeaps { heaps } => {
            for heap in heaps.iter().flatten() {
                names.heaps.add(heap.named().id());
            }
        }
        ComputeCmd::UseResources { resources, .. } => note_declared(names, resources),
        _ => {}
    }
}

/// Notes in `names` the resources a `useResources:` call declares, by their guest ids (an
/// indirect command buffer, which rt builds none of, as none).
pub(super) fn note_declared(
    names: &mut EncoderNames,
    resources: &[Option<ResourceRef<'_, RtPlatform>>],
) {
    for resource in resources.iter().flatten() {
        match resource {
            ResourceRef::Buffer(buffer) => names.declared_buffers.add(buffer.named().id()),
            ResourceRef::Texture(texture) => names.declared_textures.add(texture.named().id()),
            ResourceRef::Icb(_) => {}
        }
    }
}

impl ComputeEncoder<RtPlatform> for RtObj<RtComputeEncoder> {
    /// rt's command (the module's docs); `endEncoding` ends rt's encoder.
    fn encode(&self, on: OnFault, cmd: ComputeCmd<'_, RtPlatform>) -> RaiseResult<()> {
        let recorded_call = crate::recorder::rec_call!(@cmd self, cmd);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            let _ = on;
            let sel = cmd.selector();
            let ending = matches!(cmd, ComputeCmd::EndEncoding);
            note_copies(self.kept.copies(), &cmd);
            let mut open = self.open.lock();
            if let Some(names) = open.naming.names() {
                note_names(names, &cmd);
            }
            let stopped = self.command(&mut open, cmd).err();
            if ending && let Some(e) = open.rt.take() {
                e.end_encoding();
                self.mark.end();
                open.naming.end();
            }
            drop(open);
            report(sel, stopped);
            Ok(())
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Unbuilt: NO, as with nothing open ([`Unbuilt::Predication`]).
    fn encode_end_do_while(
        &self,
        buffer: Option<&<RtPlatform as Platform>::Buffer>,
        offset: u64,
        comparison: u64,
        reference_value: u32,
    ) -> RaiseResult<bool> {
        let recorded_call = crate::recorder::rec_call!(
            "encodeEndDoWhile:offset:comparison:referenceValue:",
            self,
            buffer,
            offset,
            comparison,
            reference_value
        );
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<bool> {
            let _ = (buffer, offset, comparison, reference_value);
            let sel = "encodeEndDoWhile:offset:comparison:referenceValue:";
            refuse(Unbuilt::Predication, sel);
            Ok(false)
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Unbuilt: NO, as with nothing open ([`Unbuilt::Predication`]).
    fn encode_end_if(&self) -> RaiseResult<bool> {
        let recorded_call = crate::recorder::rec_call!("encodeEndIf", self);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<bool> {
            refuse(Unbuilt::Predication, "encodeEndIf");
            Ok(false)
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    /// Unbuilt: NO, as with nothing open ([`Unbuilt::Predication`]).
    fn encode_end_while(&self) -> RaiseResult<bool> {
        let recorded_call = crate::recorder::rec_call!("encodeEndWhile", self);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<bool> {
            refuse(Unbuilt::Predication, "encodeEndWhile");
            Ok(false)
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::core::num::ToU64 as _;
    use crate::core::platform::command::{CommandBuffer as _, CommandQueue as _};
    use crate::core::platform::device::GpuDevice as _;
    use crate::core::platform::resource::{Buffer as _, Heap as _, Library as _};
    use crate::core::platform::types::{CommandBufferStatus, MtlOrigin, MtlRegion};
    use crate::core::platform::values::{
        ComputePassDesc, HeapDesc, OptionSet, SamplerDesc, TextureDesc,
    };
    use crate::rt_platform::diag::gaps_named;
    use crate::rt_platform::unbuilt::RtRefusal;
    use crate::rt_platform::{RtBuffer, RtComputePipeline, RtQueue};

    const ADD: &[u8] = include_bytes!("../../../../tests/shaders/compute_add.metallib");
    const TGMEM: &[u8] = include_bytes!("../../../../tests/shaders/compute_tgmem.metallib");
    const TEXWRITE: &[u8] = include_bytes!("../../../../tests/shaders/compute_texwrite.metallib");
    const SAMPLE: &[u8] = include_bytes!("../../../../tests/shaders/compute_sample.metallib");
    const STAGEIN: &[u8] = include_bytes!("../../../../tests/shaders/compute_stagein.metallib");
    const NUMBERED: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/numbered_samplers.metallib"));

    fn counted(code: &str) -> u64 {
        (crate::answer::degradation_counts().into_iter())
            .find(|(c, _)| c == code)
            .map_or(0, |(_, n)| n)
    }

    /// The first `n` words of the Shared `buffer`.
    fn words(buffer: &RtObj<RtBuffer>, n: usize) -> Vec<u32> {
        assert!((4 * n).to_u64() <= buffer.length());
        // SAFETY: a Shared buffer's contents are mapped, at least `n` words long.
        unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr::<u32>(), n) }.to_vec()
    }

    /// Encodes `cmds` into one compute encoder of a command buffer of `queue`, of `pass`'s
    /// dispatch type (serial where `None`), ends it, commits the command buffer and waits for
    /// it: its status then.
    fn run(
        queue: &RtObj<RtQueue>,
        pass: Option<&ComputePassDesc>,
        cmds: Vec<ComputeCmd<'_, RtPlatform>>,
    ) -> CommandBufferStatus {
        let on = OnFault::Return;
        let made = queue.command_buffer(on).expect("never a raise");
        let made = made.expect("a command buffer");
        let encoder = match pass {
            Some(pass) => made.compute_encoder_for_pass(pass),
            None => made.compute_encoder(on),
        };
        let encoder = encoder.expect("never a raise").expect("a compute encoder");
        for cmd in cmds {
            assert_eq!(encoder.encode(on, cmd), Ok(()));
        }
        assert_eq!(encoder.encode(on, ComputeCmd::EndEncoding), Ok(()));
        assert_eq!(made.commit(on), Ok(()));
        assert_eq!(made.wait_until_completed(on), Ok(()));
        made.status()
    }

    /// A one-dimensional grid of `n` threads, in one threadgroup.
    fn threads<'a>(n: u64) -> ComputeCmd<'a, RtPlatform> {
        ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(n, 1, 1),
            threads_per_threadgroup: MtlSize::new(n, 1, 1),
        }
    }

    /// `buffers` bound from slot 0, each at offset 0.
    fn bind<'a>(buffers: &'a [Option<&'a RtObj<RtBuffer>>]) -> ComputeCmd<'a, RtPlatform> {
        static OFFSETS: [u64; 4] = [0; 4];
        ComputeCmd::SetBuffers {
            buffers,
            offsets: &OFFSETS[..buffers.len()],
            range: NsRange::new(0, buffers.len().to_u64()),
        }
    }

    /// Compute encoders run rt's dispatches in their order, serial ones with the barriers their
    /// hazards need and concurrent ones at their barriers, over buffers, threadgroup memory,
    /// textures and sampler states; their command buffers keep the pipeline states they set
    /// until they complete. What rt has no form of is counted as its piece, what rt refuses as
    /// its gap by its kind, what Metal's validation stops as Metal's refusal, and the command
    /// buffer still completes.
    #[test]
    fn compute_encoders_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let done = CommandBufferStatus::COMPLETED;
        let metal = RtRefusal::Refused.code();
        let buffer = |length: u64| {
            let made = device.new_buffer_of_length(on, length, 0);
            made.expect("never a raise").expect("a buffer")
        };
        let holding = |values: &[u32]| {
            let made = buffer(4 * values.len().to_u64());
            // SAFETY: the Shared buffer's contents are mapped, `values` long.
            unsafe {
                let at = made.contents().as_ptr::<u32>();
                at.copy_from_nonoverlapping(values.as_ptr(), values.len());
            }
            made
        };
        let state = |bytes: &[u8], name: &str| -> RtObj<RtComputePipeline> {
            let library = device.new_library(bytes).expect("never a raise");
            let library = library.expect("a metallib").expect("a library");
            let function = library.new_function(on, name).expect("never a raise");
            let function = function.expect("a function of the table");
            let made = device.new_compute_pipeline_state_for_function(Some(&function));
            made.expect("a compute state").expect("a compute state")
        };
        let add = state(ADD, "add_one");

        // Serial: the second dispatch reads what the first wrote; a heap's use declares nothing.
        let heap = device.new_heap(&HeapDesc {
            size: 0x10000,
            heap_type: 0,
            options: vec![OptionSet::ResourceOptions(0)],
            pinned_gpu_address: None,
        });
        let heap = heap.expect("never a raise").heap.expect("a heap");
        let (input, once, twice) = (holding(&[7, 8, 9, 10]), buffer(16), buffer(16));
        let status = run(
            &queue,
            None,
            vec![
                ComputeCmd::UseHeaps {
                    heaps: &[Some(&heap), None],
                },
                ComputeCmd::SetComputePipelineState { state: Some(&add) },
                bind(&[Some(&once), Some(&input)]),
                threads(4),
                bind(&[Some(&twice), Some(&once)]),
                ComputeCmd::DispatchThreadgroups {
                    threadgroups_per_grid: MtlSize::new(2, 1, 1),
                    threads_per_threadgroup: MtlSize::new(2, 1, 1),
                },
            ],
        );
        assert_eq!(status, done);
        assert_eq!(words(&once, 4), [8, 9, 10, 11]);
        assert_eq!(words(&twice, 4), [9, 10, 11, 12]);

        // Concurrent, ordered at its barrier.
        let concurrent = ComputePassDesc {
            dispatch_type: Some(1),
        };
        let (first, second) = (buffer(16), buffer(16));
        let status = run(
            &queue,
            Some(&concurrent),
            vec![
                ComputeCmd::SetComputePipelineState { state: Some(&add) },
                bind(&[Some(&first), Some(&input)]),
                threads(4),
                ComputeCmd::MemoryBarrierWithScope { scope: 1 },
                bind(&[Some(&second), Some(&first)]),
                threads(4),
            ],
        );
        assert_eq!(status, done);
        assert_eq!(words(&second, 4), [9, 10, 11, 12]);

        // Threadgroup memory of the encoder's length.
        let reversed = buffer(16);
        let status = run(
            &queue,
            None,
            vec![
                ComputeCmd::SetComputePipelineState {
                    state: Some(&state(TGMEM, "tg_reverse")),
                },
                ComputeCmd::SetThreadgroupMemoryLength {
                    length: 16,
                    index: 0,
                },
                bind(&[Some(&reversed)]),
                threads(4),
            ],
        );
        assert_eq!(status, done);
        assert_eq!(words(&reversed, 4), [9, 6, 3, 0]);

        // A texture written, then sampled left of its edge through a clamping sampler state.
        let made = device.new_texture(
            on,
            &platform.texture_descriptor(&TextureDesc {
                width: 4,
                usage: 3,
                ..TextureDesc::default()
            }),
        );
        let texture = made.expect("never a raise").expect("a texture");
        let sampler = device.new_sampler_state(&SamplerDesc::default());
        let sampler = sampler.expect("never a raise").expect("a sampler state");
        let sampled = buffer(16);
        let status = run(
            &queue,
            None,
            vec![
                ComputeCmd::SetComputePipelineState {
                    state: Some(&state(TEXWRITE, "tex_write")),
                },
                ComputeCmd::SetTexture {
                    texture: Some(&texture),
                    index: 0,
                },
                threads(4),
                ComputeCmd::SetComputePipelineState {
                    state: Some(&state(SAMPLE, "sample_k")),
                },
                ComputeCmd::SetTextures {
                    textures: &[Some(&texture)],
                    range: NsRange::new(0, 1),
                },
                ComputeCmd::SetSamplerStates {
                    samplers: &[Some(&sampler)],
                    range: NsRange::new(0, 1),
                },
                bind(&[Some(&sampled)]),
                threads(4),
            ],
        );
        assert_eq!(status, done);
        let mut texels = [0; 16];
        texture.read_pixels(&mut texels, 16);
        assert_eq!(
            texels,
            [
                0, 0, 128, 255, 1, 0, 128, 255, 2, 0, 128, 255, 3, 0, 128, 255
            ]
        );
        assert_eq!(words(&sampled, 4), [0xff80_0000; 4]);

        // A command buffer keeps the state it set, and its sampler index, until it completes.
        let held = device.numbering().held();
        let numbered = state(NUMBERED, "numbered_sample");
        assert_eq!(device.numbering().held(), held + 1);
        let made = queue.command_buffer(on).expect("never a raise");
        let made = made.expect("a command buffer");
        let encoder = made.compute_encoder(on).expect("never a raise");
        let encoder = encoder.expect("a compute encoder");
        let set = ComputeCmd::SetComputePipelineState {
            state: Some(&numbered),
        };
        assert_eq!(encoder.encode(on, set), Ok(()));
        assert_eq!(encoder.encode(on, ComputeCmd::EndEncoding), Ok(()));
        drop(numbered);
        assert_eq!(device.numbering().held(), held + 1);
        assert_eq!(made.commit(on), Ok(()));
        assert_eq!(made.wait_until_completed(on), Ok(()));
        for _ in 0..1000 {
            if device.numbering().held() == held {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(device.numbering().held(), held);

        // What rt has no form of, what Metal's validation stops, and what records nothing.
        let (unbuilt, predication) = (Unbuilt::ComputeCommand.code(), Unbuilt::Predication.code());
        let (before, unbuilt_before, predication_before) =
            (counted(metal), counted(unbuilt), counted(predication));
        let ours = (RtRefusal::Ours.code(), counted(RtRefusal::Ours.code()));
        let gap = (RtRefusal::Gap.code(), counted(RtRefusal::Gap.code()));
        let indirect_region = "StageIn-IndirectRegion";
        let indirect_region_before = gaps_named(indirect_region);
        let made = queue.command_buffer(on).expect("never a raise");
        let made = made.expect("a command buffer");
        let encoder = made.compute_encoder(on).expect("never a raise");
        let encoder = encoder.expect("a compute encoder");
        let cmds = [
            ComputeCmd::DispatchThreadsIndirect {
                indirect_buffer: Some(&input),
                indirect_buffer_offset: 0,
            },
            ComputeCmd::SetStageInRegionIndirect {
                indirect_buffer: &input,
                indirect_buffer_offset: 0,
            },
            ComputeCmd::EncodeStartIf {
                buffer: None,
                offset: 0,
                comparison: 0,
                reference_value: 0,
            },
            ComputeCmd::EncodeStartElse,
            threads(4),
            ComputeCmd::SetComputePipelineState { state: Some(&add) },
            ComputeCmd::DispatchThreadgroupsIndirect {
                indirect_buffer: None,
                indirect_buffer_offset: 0,
                threads_per_threadgroup: MtlSize::new(1, 1, 1),
            },
            ComputeCmd::SetComputePipelineState { state: None },
            threads(4),
            ComputeCmd::SetBuffers {
                buffers: &[Some(&input)],
                offsets: &[0, 0],
                range: NsRange::new(0, 1),
            },
            ComputeCmd::UseResources {
                resources: &[Some(ResourceRef::Buffer(&input))],
                usage: 8,
            },
            ComputeCmd::SetTexture {
                texture: None,
                index: u64::MAX,
            },
            ComputeCmd::SetBufferOffset {
                offset: 0,
                index: 5,
            },
            ComputeCmd::UseResources {
                resources: &[None, Some(ResourceRef::Buffer(&input))],
                usage: 1,
            },
            ComputeCmd::SetImageblockWidth {
                width: 16,
                height: 16,
            },
            ComputeCmd::CompressedTextureReinterpretationFlush,
            ComputeCmd::SetVisibleFunctionTable {
                function_table: None,
                buffer_index: 0,
            },
            ComputeCmd::SetStageInRegion {
                region: MtlRegion {
                    origin: MtlOrigin::default(),
                    size: MtlSize::new(4, 1, 1),
                },
            },
        ];
        for cmd in cmds {
            assert_eq!(encoder.encode(on, cmd), Ok(()));
        }
        assert_eq!(encoder.encode_end_if(), Ok(false));
        assert_eq!(encoder.encode_end_while(), Ok(false));
        assert_eq!(encoder.encode_end_do_while(None, 0, 0, 0), Ok(false));
        assert_eq!(encoder.encode(on, ComputeCmd::EndEncoding), Ok(()));
        let after = ComputeCmd::SetBufferOffset {
            offset: 0,
            index: 0,
        };
        assert_eq!(encoder.encode(on, after), Ok(()));
        assert_eq!(made.commit(on), Ok(()));
        assert_eq!(made.wait_until_completed(on), Ok(()));
        assert_eq!(made.status(), done);
        // The indirect dispatchThreads alone; the indirect stage-in region is rt's gap, by its
        // kind.
        assert_eq!(counted(unbuilt), unbuilt_before + 1);
        assert_eq!(counted(gap.0), gap.1 + 1);
        assert_eq!(gaps_named(indirect_region), indirect_region_before + 1);
        assert_eq!(counted(predication), predication_before + 5);
        // A dispatch with none set, a nil indirect buffer, a dispatch with nil set, lengths
        // apart, usage 8, an index past 32 bits, and a command after `endEncoding`; and rt's
        // refusal on a rule of its own of an empty slot's offset (`Refusal::NoBufferAtIndex`).
        assert_eq!(counted(metal), before + 7);
        assert_eq!(counted(ours.0), ours.1 + 1);

        // A nil resource of a concurrent encoder, and a dispatch type Metal names none of.
        let before = counted(metal);
        let nil = ComputeCmd::UseResources {
            resources: &[None],
            usage: 1,
        };
        assert_eq!(run(&queue, Some(&concurrent), vec![nil]), done);
        let made = queue.command_buffer(on).expect("never a raise");
        let made = made.expect("a command buffer");
        let unknown = ComputePassDesc {
            dispatch_type: Some(7),
        };
        assert!(matches!(made.compute_encoder_for_pass(&unknown), Ok(None)));
        assert_eq!(counted(metal), before + 2);
        drop(made);
    }

    /// Each dispatch rt encodes moves its form's count on the counts line by one, as
    /// `fold_metrics` folds them: a direct `dispatchThreads` ([`DISPATCHES_DIRECT_CODE`]) and an
    /// indirect `dispatchThreadgroups` ([`DISPATCHES_INDIRECT_CODE`]), each running its grid; a
    /// dispatch with no pipeline state set, Metal's refusal, moves neither.
    #[test]
    fn each_dispatch_rt_encodes_counts_by_its_form_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let done = CommandBufferStatus::COMPLETED;
        let holding = |values: &[u32]| {
            let length = 4 * values.len().to_u64();
            let made = device.new_buffer_of_length(on, length, 0);
            let made = made.expect("never a raise").expect("a buffer");
            // SAFETY: the Shared buffer's contents are mapped, `values` long.
            unsafe {
                let at = made.contents().as_ptr::<u32>();
                at.copy_from_nonoverlapping(values.as_ptr(), values.len());
            }
            made
        };
        let library = device.new_library(ADD).expect("never a raise");
        let library = library.expect("a metallib").expect("a library");
        let kernel = library.new_function(on, "add_one");
        let kernel = kernel.expect("never a raise").expect("the kernel");
        let made = device.new_compute_pipeline_state_for_function(Some(&kernel));
        let add = made.expect("a compute state").expect("a compute state");
        let counts = || {
            crate::rt_platform::device::fold_metrics();
            [
                counted(DISPATCHES_DIRECT_CODE),
                counted(DISPATCHES_INDIRECT_CODE),
            ]
        };
        let (input, groups) = (holding(&[7, 8, 9, 10]), holding(&[2, 1, 1]));
        let (direct, indirect) = (holding(&[0; 4]), holding(&[0; 4]));
        let [direct_before, indirect_before] = counts();

        let status = run(
            &queue,
            None,
            vec![
                threads(4),
                ComputeCmd::SetComputePipelineState { state: Some(&add) },
                bind(&[Some(&direct), Some(&input)]),
                threads(4),
            ],
        );
        assert_eq!(status, done);
        assert_eq!(words(&direct, 4), [8, 9, 10, 11]);
        assert_eq!(counts(), [direct_before + 1, indirect_before]);

        let status = run(
            &queue,
            None,
            vec![
                ComputeCmd::SetComputePipelineState { state: Some(&add) },
                bind(&[Some(&indirect), Some(&input)]),
                ComputeCmd::DispatchThreadgroupsIndirect {
                    indirect_buffer: Some(&groups),
                    indirect_buffer_offset: 0,
                    threads_per_threadgroup: MtlSize::new(2, 1, 1),
                },
            ],
        );
        assert_eq!(status, done);
        assert_eq!(words(&indirect, 4), [8, 9, 10, 11]);
        assert_eq!(counts(), [direct_before + 1, indirect_before + 1]);
    }

    /// A kernel's `[[stage_in]]` attribute of a fixed layout stride that steps by the thread's
    /// position in X is fetched by the encoder's stage-in region (`setStageInRegion:`): thread `x`
    /// reads element `origin + x / rate` (`stage_in_k` writes it plus `0x100`), at a step rate of
    /// 1 and of 2, every thread inside the region. A stage-in region read from a buffer is rt's
    /// gap, counted by its kind, and the dispatch after it fetches by the region set before it.
    /// The forms rt does not build are refused as the state is made, nil and an `NSError`, each
    /// counted by its kind: a layout of `MTLBufferLayoutStrideDynamic` (`StageIn-DynamicStride`)
    /// and an indexed step at a rate of 2 (`StageIn-IndexedStepRate`).
    #[test]
    fn a_kernel_s_stage_in_is_fetched_by_the_stage_in_region_on_vulkan() {
        use crate::core::platform::values::{
            AttributeDesc, BufferLayoutDesc, FullComputePipelineDesc, StageInputDesc,
        };
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let done = CommandBufferStatus::COMPLETED;
        let holding = |values: &[u32]| {
            let length = 4 * values.len().to_u64();
            let made = device.new_buffer_of_length(on, length, 0);
            let made = made.expect("never a raise").expect("a buffer");
            // SAFETY: the Shared buffer's contents are mapped, `values` long.
            unsafe {
                let at = made.contents().as_ptr::<u32>();
                at.copy_from_nonoverlapping(values.as_ptr(), values.len());
            }
            made
        };
        let library = device.new_library(STAGEIN).expect("never a raise");
        let library = library.expect("a metallib").expect("a library");
        let kernel = library.new_function(on, "stage_in_k");
        let kernel = kernel.expect("never a raise").expect("the kernel");
        // Attribute 0, a `UInt` at offset 0 of buffer 1, whose one layout is `layout`.
        let stage_input = |layout: BufferLayoutDesc| StageInputDesc {
            layouts: vec![layout],
            attributes: vec![AttributeDesc {
                attribute_index: 0,
                format: 36,
                offset: 0,
                buffer_index: 1,
            }],
            index_type: 1,
            index_buffer_index: 2,
        };
        // A layout of buffer 1 of `stride` bytes, stepped by `step` (`MTLStepFunction`) at `rate`.
        let layout = |stride: u64, step: u64, rate: u64| BufferLayoutDesc {
            buffer_index: 1,
            stride,
            step_function: step,
            step_rate: rate,
        };
        let make = |layout: BufferLayoutDesc| {
            let desc = FullComputePipelineDesc {
                stage_input: Some(stage_input(layout)),
                ..FullComputePipelineDesc::new(&kernel)
            };
            device.new_compute_pipeline_state_full(&desc)
        };
        let made = |layout: BufferLayoutDesc| {
            let made = make(layout).map_err(|error| error.description);
            let made = made.expect("rt makes a stage input of a fixed stride");
            made.expect("a compute state")
        };
        // Element `e` of buffer 1 holds `0x10 + e`.
        let elements: Vec<u32> = (0..16).map(|e| 0x10 + e).collect();
        let input = holding(&elements);
        let region = |x: u64, width: u64| MtlRegion {
            origin: MtlOrigin { x, y: 0, z: 0 },
            size: MtlSize::new(width, 1, 1),
        };
        // `MTLStepFunctionThreadPositionInGridX`.
        let by_x = 5;

        // Each thread's element is the region's origin plus its position over the step rate, as an
        // M3 Max answers (measured: metal2vulkan's recording of its case `stage-in-region-clamps`,
        // rows `x_o2_r4_g8` and `x_rate2_o1_r2_g8`). Every thread here is inside the region: Metal
        // leaves a thread past it undefined. By row: the step rate, the region's origin and size,
        // and what each thread writes.
        let rows: [(u64, u64, u64, &[u32]); 2] = [
            (1, 2, 4, &[0x112, 0x113, 0x114, 0x115]),
            (
                2,
                1,
                8,
                &[0x111, 0x111, 0x112, 0x112, 0x113, 0x113, 0x114, 0x114],
            ),
        ];
        for (rate, origin, width, expected) in rows {
            let state = made(layout(4, by_x, rate));
            let out = holding(&[0; 8]);
            let status = run(
                &queue,
                None,
                vec![
                    ComputeCmd::SetComputePipelineState {
                        state: Some(&state),
                    },
                    bind(&[Some(&out), Some(&input)]),
                    ComputeCmd::SetStageInRegion {
                        region: region(origin, width),
                    },
                    threads(expected.len().to_u64()),
                ],
            );
            assert_eq!(status, done, "rate {rate}");
            let written = words(&out, expected.len());
            assert_eq!(written, expected, "rate {rate}, origin {origin}");
        }

        // A region read from a buffer (origin 0, size 4) is refused, counted, and the region set
        // before it (origin 2) is the one the dispatch fetches by.
        let state = made(layout(4, by_x, 1));
        let (out, indirect) = (holding(&[0; 4]), holding(&[0, 0, 0, 4, 1, 1]));
        let refused = gaps_named("StageIn-IndirectRegion");
        let status = run(
            &queue,
            None,
            vec![
                ComputeCmd::SetComputePipelineState {
                    state: Some(&state),
                },
                bind(&[Some(&out), Some(&input)]),
                ComputeCmd::SetStageInRegion {
                    region: region(2, 4),
                },
                ComputeCmd::SetStageInRegionIndirect {
                    indirect_buffer: &indirect,
                    indirect_buffer_offset: 0,
                },
                threads(4),
            ],
        );
        assert_eq!(status, done);
        assert_eq!(gaps_named("StageIn-IndirectRegion"), refused + 1);
        assert_eq!(words(&out, 4), [0x112, 0x113, 0x114, 0x115]);

        // The forms rt does not build, each refused as the state is made, counted by its kind: a
        // layout of `MTLBufferLayoutStrideDynamic` stepped by X, and one stepped by
        // `MTLStepFunctionThreadPositionInGridXIndexed` (7) at a rate of 2.
        let gap = RtRefusal::Gap.code();
        for (refused, kind) in [
            (layout(u64::MAX, by_x, 1), "StageIn-DynamicStride"),
            (layout(4, 7, 2), "StageIn-IndexedStepRate"),
        ] {
            let (gaps, named) = (counted(gap), gaps_named(kind));
            let refusal = make(refused).err();
            let refusal = refusal.unwrap_or_else(|| panic!("{kind}: refused"));
            assert!(refusal.description.contains(gap), "{kind}: {refusal:?}");
            assert_eq!(counted(gap), gaps + 1, "{kind}");
            assert_eq!(gaps_named(kind), named + 1, "{kind}");
        }
    }

    /// A declaration of no usage (`useResources:count:usage:` of 0) declares nothing and is no
    /// refusal, in a serial and in a concurrent encoder (the module's docs, "No usage"), where a
    /// usage with a bit past sample is Metal's refusal, counted; the dispatch after either runs.
    #[test]
    fn a_declaration_of_no_usage_is_no_refusal_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let done = CommandBufferStatus::COMPLETED;
        let metal = RtRefusal::Refused.code();
        let buffer = || {
            let made = device.new_buffer_of_length(on, 16, 0);
            made.expect("never a raise").expect("a buffer")
        };
        let library = device.new_library(ADD).expect("never a raise");
        let library = library.expect("a metallib").expect("a library");
        let function = library.new_function(on, "add_one").expect("never a raise");
        let function = function.expect("a function of the table");
        let add = device.new_compute_pipeline_state_for_function(Some(&function));
        let add = add.expect("a compute state").expect("a compute state");
        let concurrent = ComputePassDesc {
            dispatch_type: Some(1),
        };
        for (pass, kind) in [(None, "serial"), (Some(&concurrent), "concurrent")] {
            for (usage, refused) in [(0, 0), (8, 1)] {
                let (output, input) = (buffer(), buffer());
                // SAFETY: the Shared buffer's contents are mapped, 4 words long.
                unsafe {
                    let at = input.contents().as_ptr::<u32>();
                    at.copy_from_nonoverlapping([1, 2, 3, 4].as_ptr(), 4);
                }
                let declared = [
                    Some(ResourceRef::Buffer(&input)),
                    Some(ResourceRef::Buffer(&output)),
                ];
                let before = counted(metal);
                let status = run(
                    &queue,
                    pass,
                    vec![
                        ComputeCmd::UseResources {
                            resources: &declared,
                            usage,
                        },
                        ComputeCmd::SetComputePipelineState { state: Some(&add) },
                        bind(&[Some(&output), Some(&input)]),
                        threads(4),
                    ],
                );
                assert_eq!(status, done, "{kind} {usage}");
                assert_eq!(counted(metal), before + refused, "{kind} {usage}");
                assert_eq!(words(&output, 4), [2, 3, 4, 5], "{kind} {usage}");
            }
        }
    }

    /// A placement heap's buffers are one resource to a serial encoder's hazards, as Metal tracks a
    /// heap's resources as one: a dispatch that reads one of its buffers after a dispatch wrote
    /// another, at other bytes, waits for it behind a barrier, where two buffers of no heap take
    /// none.
    #[test]
    fn a_heap_s_buffers_are_one_resource_on_vulkan() {
        let platform = RtPlatform::new();
        let device = (platform.system_default_device()).expect("an rt device on the GPU it holds");
        let on = OnFault::Return;
        let queue = device.new_command_queue().expect("a command queue");
        let buffer = || {
            let made = device.new_buffer_of_length(on, 16, 0);
            made.expect("never a raise").expect("a buffer")
        };
        let library = device.new_library(ADD).expect("never a raise");
        let library = library.expect("a metallib").expect("a library");
        let function = library.new_function(on, "add_one").expect("never a raise");
        let function = function.expect("a function of the table");
        let add = device.new_compute_pipeline_state_for_function(Some(&function));
        let add = add.expect("a compute state").expect("a compute state");
        // Tracked, so that Metal itself orders the heap's buffers.
        let heap = device.new_heap(&HeapDesc {
            size: 0x10000,
            heap_type: 1,
            options: vec![OptionSet::ResourceOptions(0x200)],
            pinned_gpu_address: None,
        });
        let heap = heap.expect("never a raise").heap.expect("a placement heap");
        let placed = |offset| {
            let made = heap.new_buffer_of_length(16, 0, offset);
            made.expect("never a raise").expect("a buffer in the heap")
        };
        let input = buffer();
        let barriers = || {
            crate::rt_platform::device::metrics()
                .expect("rt's metrics")
                .barriers
        };
        // The barriers of one encoder that writes `written`, then reads `read` into a buffer of
        // its own.
        let barriers_between = |written: &RtObj<RtBuffer>, read: &RtObj<RtBuffer>| {
            let out = buffer();
            let before = barriers();
            let status = run(
                &queue,
                None,
                vec![
                    ComputeCmd::SetComputePipelineState { state: Some(&add) },
                    bind(&[Some(written), Some(&input)]),
                    threads(4),
                    bind(&[Some(&out), Some(read)]),
                    threads(4),
                ],
            );
            assert_eq!(status, CommandBufferStatus::COMPLETED);
            barriers() - before
        };
        let apart = barriers_between(&buffer(), &buffer());
        let in_heap = barriers_between(&placed(0), &placed(0x100));
        assert_eq!(in_heap, apart + 1);
    }
}
