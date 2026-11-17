//! `id<MTLComputeCommandEncoder>`: core's `ComputeEncoder` over the runtime's, one runtime call
//! per `ComputeCmd`.
//!
//! - **A runtime refusal** of any command, `endEncoding` too, dies at its selector naming the
//!   runtime's error (`Policy::Dies`): Metal raises at none of them. Where Metal goes on and the
//!   runtime refuses, the adapter does as Metal does, below.
//! - **Executing an indirect command buffer** runs its commands as their records stand when the
//!   command buffer is committed, or, for a buffer the GPU may have written (a blit wrote it, or
//!   an encoder made it resident for writing), once the work before the execution completed (the
//!   runtime's host-side step), where Metal's GPU reads them as it reaches them. The runtime's
//!   refusal of the records dies naming the execution, at the commit or as the command buffer
//!   completes ([`crate::gpu::command`]); a range past its size, which Metal runs without
//!   validation, dies at the call.
//! - **Work of a pipeline that can encode indirect commands** writes the records of a buffer the
//!   encoder made resident for writing through the buffer's ID, the runtime's, Metal's
//!   ([`crate::gpu::icb`]), so it reaches them as Metal's GPU does ([`encodes_resident`]). Which
//!   pipelines can encode indirect commands is the runtime's `can_encode_indirect_commands`, read
//!   from the kernel's AIR (an indirect command buffer it takes, through an argument buffer at
//!   any depth). The work is a direct dispatch ([`dispatchable`]), and an execution's commands:
//!   where they inherit the encoder's pipeline, the execution decides at its call, as a dispatch;
//!   where they name their own, the commit reads their records, or, for records the GPU may have
//!   written, the runtime's host step asks them ([`crate::gpu::command::executions`]). A render
//!   encoder's functions go alike ([`crate::gpu::encode_render`]). The encoder's making a buffer
//!   resident for writing marks the runtime's buffer written for good (`gpu_written`), so a later
//!   execution's commands are asked at the runtime's host step, and the runtime keeps when the
//!   GPU may write its records, which decides the pipeline state a record names
//!   ([`crate::gpu::icb`]); where the pipeline can encode, a command buffer that executes the
//!   buffer as its commit read it dies at the write (`executions::gpu_writes`).
//! - **Command predication** (`encodeStart*`, `encodeEnd*`) is the adapter's ([`predication`]):
//!   every command passes it
//!   first, and the dispatches and executions are sent through it. The runtime's own
//!   predication is never asked.
//! - **A nil entry of `useResources:count:usage:`, `useHeaps:count:` or
//!   `memoryBarrierWithResources:count:`** is skipped, as Metal skips it
//!   (`compute_err_resource_lists`: a pipeline's ref and an unknown one are nil and the pass
//!   runs), but a nil resource in a concurrent encoder's residency dies: Metal ends the process at
//!   one there (`interplay_decodewalk_26`). A nil heap is skipped in a concurrent encoder too, as
//!   Metal's skips it (measured on an M3 Max under macOS 27.0.1, without API validation). A barrier
//!   of nil entries alone is a barrier over buffers and textures, as the runtime's barrier of
//!   resources is.
//! - **What the runtime has no form for dies**: a nil pipeline or indirect buffer, an index past
//!   `u32`, a slice of another length than its range, a scope or usage no Metal value is,
//!   and a command after `endEncoding`. A rotated texture bound to the kernel is the runtime's,
//!   which reads it as Metal's functions do ([`crate::gpu::resource::TextureObject::texels`]).

use std::cell::Cell;
use std::ops::Range;
use std::sync::atomic::Ordering;

use crate::core::platform::Handle as _;
use crate::core::platform::blit::ResourceRef;
use crate::core::platform::compute::{ComputeCmd, ComputeEncoder};
use crate::core::platform::types::NsRange;
use crate::core::raise::{OnFault, RaiseResult};
use metal2vulkan::metal::{BarrierScope, DispatchType, ResourceUsage};
use metal2vulkan_runtime as rt;

use crate::answer::{Policy, plain, send, unrepresentable};
use crate::gpu::command::executions::{self, Execution, Points};
use crate::gpu::icb::VulkanIcb;
use crate::gpu::predication::{self, PredicatedEncoder, Work};
use crate::gpu::resource::VulkanBuffer;
use crate::gpu::{listed, metal_value, region};
use crate::platform::{Obj, VulkanPlatform};

/// `id<MTLComputeCommandEncoder>`.
pub type VulkanComputeEncoder = Obj<PredicatedEncoder>;

/// A binding index or a length core passes as `NSUInteger`, which the runtime takes as `u32`.
fn small(sel: &'static str, value: u64) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| unrepresentable(sel, &format!("{value} as a u32")))
}

/// The first index of `range`, whose length every slice bound with it has.
fn first(sel: &'static str, range: NsRange, lengths: &[usize]) -> u32 {
    if lengths
        .iter()
        .any(|&n| u64::try_from(n).ok() != Some(range.length))
    {
        unrepresentable(sel, &format!("slices of another length than {range:?}"));
    }
    small(sel, range.location)
}

/// `object`, which core may pass as nil, for `sel`: the runtime takes no nil one.
fn some<'a, T>(sel: &'static str, object: Option<&'a T>, what: &str) -> &'a T {
    object.unwrap_or_else(|| unrepresentable(sel, &format!("a nil {what}")))
}

/// The runtime's form of core's resources, the nil ones skipped, as Metal skips them.
fn resources<'a>(
    resources: &'a [Option<ResourceRef<'a, VulkanPlatform>>],
) -> Vec<rt::ResourceRef<'a>> {
    resources
        .iter()
        .flatten()
        .map(|r| match r {
            ResourceRef::Buffer(b) => rt::ResourceRef::Buffer(&b.raw),
            ResourceRef::Texture(t) => rt::ResourceRef::Texture(&t.raw),
            ResourceRef::Icb(icb) => rt::ResourceRef::IndirectCommandBuffer(&icb.raw),
        })
        .collect()
}

/// A direct dispatch `sel` of `encoder`: where its pipeline can encode indirect commands, its
/// kernel may write the records of each buffer the encoder made resident for writing (the module
/// docs).
fn dispatchable(encoder: &VulkanComputeEncoder, sel: &'static str) {
    if plain(|| encoder.encodes_icbs.load(Ordering::Relaxed)) {
        encodes_resident(encoder, sel);
    }
}

/// Work `sel` of `encoder` runs the pipeline state the encoder set last, which can encode
/// indirect commands: it may write the records of each buffer the encoder made resident for
/// writing ([`crate::gpu::command::executions::gpu_writes`]).
fn encodes_resident(encoder: &VulkanComputeEncoder, sel: &'static str) {
    for icb in writable(encoder) {
        executions::gpu_writes(sel, &icb);
    }
}

/// Another reference to each indirect command buffer `encoder` made resident for writing.
fn writable(encoder: &VulkanComputeEncoder) -> Vec<VulkanIcb> {
    plain(|| {
        let writable = encoder.writable_icbs.lock();
        writable.iter().map(|icb| icb.clone_ref()).collect()
    })
}

/// An execution `sel` of `icb`'s commands `range` (`None`: a range a buffer holds) in `encoder`,
/// which runs nothing where it is `empty`. Where its commands inherit the encoder's pipeline
/// state, that state decides as a dispatch's does ([`encodes_resident`]). Where they name their
/// own, its commit reads them where the encoder made resident for writing any indirect command
/// buffer, whose records a command may write ([`crate::gpu::command::executions`]).
fn executes(
    encoder: &VulkanComputeEncoder,
    sel: &'static str,
    (icb, range): (&VulkanIcb, Option<Range<u64>>),
    empty: bool,
) -> Option<Execution> {
    if empty {
        return None;
    }
    if icb.inherits.pipeline {
        if plain(|| encoder.encodes_icbs.load(Ordering::Relaxed)) {
            encodes_resident(encoder, sel);
        }
        return None;
    }
    let execution = Execution {
        sel,
        icb: icb.clone_ref(),
        range,
        points: Points::Unasked,
        writable: writable(encoder),
    };
    execution.asks().then_some(execution)
}

/// `execution`, which the runtime took, is checked at the commit of `encoder`'s command buffer
/// ([`crate::gpu::command::CommandBufferObject::checked_at_commit`]).
fn checked_at_commit(encoder: &VulkanComputeEncoder, execution: Option<Execution>) {
    if let Some(execution) = execution
        && let Some(cb) = encoder.command_buffer()
    {
        cb.checked_at_commit(execution);
    }
}

/// Whether an encoder's dispatches run concurrently, for `sel`.
trait Concurrent {
    fn concurrent(&self, sel: &'static str) -> bool;
}

impl Concurrent for VulkanComputeEncoder {
    fn concurrent(&self, sel: &'static str) -> bool {
        plain(|| self.with(sel, |e| e.dispatch_type())) == DispatchType::Concurrent
    }
}

impl ComputeEncoder<VulkanPlatform> for VulkanComputeEncoder {
    fn encode(&self, _on: OnFault, cmd: ComputeCmd<'_, VulkanPlatform>) -> RaiseResult<()> {
        let mut recorded_call = crate::recorder::rec_call!(@cmd self, cmd);
        let snapshot = crate::recorder::private::before(&recorded_call, self, &cmd);
        let recorded_answer = crate::recorder::run(|| -> RaiseResult<()> {
            let sel = cmd.selector();
            predication::observe(self, &cmd);
            let caught =
                |call: &mut dyn FnMut(&mut rt::ComputeCommandEncoder) -> Result<(), rt::Error>| {
                    send(sel, Policy::Dies, || self.with(sel, |e| call(e)))
                };
            if matches!(
                cmd,
                ComputeCmd::DispatchThreadgroups { .. }
                    | ComputeCmd::DispatchThreadgroupsIndirect { .. }
                    | ComputeCmd::DispatchThreads { .. }
                    | ComputeCmd::DispatchThreadsIndirect { .. }
            ) {
                dispatchable(self, sel);
            }
            match cmd {
                ComputeCmd::DispatchThreadgroups {
                    threadgroups_per_grid,
                    threads_per_threadgroup,
                } => predication::work(
                    self,
                    sel,
                    Work::Threadgroups {
                        groups: threadgroups_per_grid,
                        threads: threads_per_threadgroup,
                    },
                ),
                ComputeCmd::DispatchThreadgroupsIndirect {
                    indirect_buffer,
                    indirect_buffer_offset,
                    threads_per_threadgroup,
                } => {
                    let buffer = some(sel, indirect_buffer, "indirect buffer");
                    let work = Work::ThreadgroupsIndirect {
                        buffer: &buffer.raw,
                        offset: indirect_buffer_offset,
                        threads: threads_per_threadgroup,
                    };
                    predication::work(self, sel, work)
                }
                ComputeCmd::DispatchThreads {
                    threads_per_grid,
                    threads_per_threadgroup,
                } => predication::work(
                    self,
                    sel,
                    Work::Threads {
                        threads: threads_per_grid,
                        per_group: threads_per_threadgroup,
                    },
                ),
                ComputeCmd::DispatchThreadsIndirect {
                    indirect_buffer,
                    indirect_buffer_offset,
                } => {
                    let buffer = some(sel, indirect_buffer, "indirect buffer");
                    let work = Work::ThreadsIndirect {
                        buffer: &buffer.raw,
                        offset: indirect_buffer_offset,
                    };
                    predication::work(self, sel, work)
                }
                ComputeCmd::EncodeStartDoWhile => predication::start_do_while(self, sel),
                ComputeCmd::EncodeStartElse => predication::start_else(self, sel),
                ComputeCmd::EncodeStartIf {
                    buffer,
                    offset,
                    comparison,
                    reference_value,
                } => predication::start_if(self, sel, buffer, offset, comparison, reference_value),
                ComputeCmd::EncodeStartWhile {
                    buffer,
                    offset,
                    comparison,
                    reference_value,
                } => {
                    predication::start_while(self, sel, buffer, offset, comparison, reference_value)
                }
                ComputeCmd::EndEncoding => self.end(sel, |e| e.end_encoding()),
                ComputeCmd::ExecuteCommandsIndirect {
                    indirect_commandbuffer,
                    indirect_range_buffer,
                    indirect_buffer_offset,
                } => {
                    let checked = executes(self, sel, (indirect_commandbuffer, None), false);
                    let progress = self.progress(sel);
                    indirect_commandbuffer.executed(sel, None, progress.as_ref());
                    let work = Work::ExecuteIndirect {
                        icb: &indirect_commandbuffer.raw,
                        buffer: &indirect_range_buffer.raw,
                        offset: indirect_buffer_offset,
                        inherits: indirect_commandbuffer.inherits,
                        check: Cell::new(checked.as_ref().and_then(Execution::host_step_check)),
                    };
                    predication::work(self, sel, work)?;
                    checked_at_commit(self, checked);
                    if let Some(progress) = progress {
                        progress.executes(sel, indirect_commandbuffer.clone_ref());
                    }
                    Ok(())
                }
                ComputeCmd::ExecuteCommandsInBuffer {
                    indirect_command_buffer,
                    execution_range,
                } => {
                    let range = crate::gpu::range(execution_range).unwrap_or_else(|| {
                        unrepresentable(sel, &format!("the range {execution_range:?}"))
                    });
                    let execution = (indirect_command_buffer, Some(range.clone()));
                    let checked = executes(self, sel, execution, range.is_empty());
                    let progress = self.progress(sel);
                    indirect_command_buffer.executed(sel, Some(range.clone()), progress.as_ref());
                    let work = Work::Execute {
                        icb: &indirect_command_buffer.raw,
                        range,
                        inherits: indirect_command_buffer.inherits,
                        check: Cell::new(checked.as_ref().and_then(Execution::host_step_check)),
                    };
                    predication::work(self, sel, work)?;
                    checked_at_commit(self, checked);
                    if let Some(progress) = progress {
                        progress.executes(sel, indirect_command_buffer.clone_ref());
                    }
                    Ok(())
                }
                ComputeCmd::CompressedTextureReinterpretationFlush => caught(&mut |e| {
                    e.insert_compressed_texture_reinterpretation_flush();
                    Ok(())
                }),
                ComputeCmd::MemoryBarrierWithResources { resources: list } => {
                    let some = resources(list);
                    if some.is_empty() && !list.is_empty() {
                        return caught(&mut |e| {
                            e.memory_barrier_with_scope(
                                BarrierScope::BUFFERS.union(BarrierScope::TEXTURES),
                            );
                            Ok(())
                        });
                    }
                    caught(&mut |e| e.memory_barrier_with_resources(&some))
                }
                ComputeCmd::MemoryBarrierWithScope { scope } => {
                    let scope: BarrierScope = metal_value(sel, "barrier scope", scope);
                    caught(&mut |e| {
                        e.memory_barrier_with_scope(scope);
                        Ok(())
                    })
                }
                ComputeCmd::SetBufferOffset { offset, index } => {
                    let index = small(sel, index);
                    caught(&mut |e| e.set_buffer_offset(offset, index))
                }
                ComputeCmd::SetBufferOffsetAttributeStride {
                    offset,
                    stride,
                    index,
                } => {
                    let index = small(sel, index);
                    caught(&mut |e| e.set_buffer_offset_with_stride(offset, stride, index))
                }
                ComputeCmd::SetBuffersAttributeStrides {
                    buffers,
                    offsets,
                    strides,
                    range,
                } => {
                    let at = first(sel, range, &[buffers.len(), offsets.len(), strides.len()]);
                    listed(
                        buffers,
                        |b| b.map(|b| &b.raw),
                        |buffers| {
                            caught(&mut |e| {
                                e.set_buffers_with_strides(buffers, offsets, strides, at)
                            })
                        },
                    )
                }
                ComputeCmd::SetBuffers {
                    buffers,
                    offsets,
                    range,
                } => {
                    let at = first(sel, range, &[buffers.len(), offsets.len()]);
                    listed(
                        buffers,
                        |b| b.map(|b| &b.raw),
                        |buffers| caught(&mut |e| e.set_buffers(buffers, offsets, at)),
                    )
                }
                ComputeCmd::SetComputePipelineState { state } => {
                    let state = some(sel, state, "pipeline state");
                    caught(&mut |e| e.set_compute_pipeline_state(state))?;
                    let encodes = plain(|| state.can_encode_indirect_commands());
                    self.encodes_icbs.store(encodes, Ordering::Relaxed);
                    self.pipeline
                        .store(state.unique_identifier, Ordering::Relaxed);
                    Ok(())
                }
                ComputeCmd::SetImageblockWidth { width, height } => {
                    caught(&mut |e| e.set_imageblock_size(width, height))
                }
                ComputeCmd::SetSamplerStatesLodClamps {
                    samplers,
                    lod_min_clamps,
                    lod_max_clamps,
                    range,
                } => {
                    let lengths = [samplers.len(), lod_min_clamps.len(), lod_max_clamps.len()];
                    let at = first(sel, range, &lengths);
                    listed(
                        samplers,
                        |s| s.map(|s| &**s),
                        |samplers| {
                            caught(&mut |e| {
                                e.set_sampler_states_with_lod_clamps(
                                    samplers,
                                    lod_min_clamps,
                                    lod_max_clamps,
                                    at,
                                )
                            })
                        },
                    )
                }
                ComputeCmd::SetSamplerStates { samplers, range } => {
                    let at = first(sel, range, &[samplers.len()]);
                    listed(
                        samplers,
                        |s| s.map(|s| &**s),
                        |samplers| caught(&mut |e| e.set_sampler_states(samplers, at)),
                    )
                }
                ComputeCmd::SetStageInRegion { region: r } => {
                    caught(&mut |e| e.set_stage_in_region(region(r)))
                }
                ComputeCmd::SetStageInRegionIndirect {
                    indirect_buffer,
                    indirect_buffer_offset,
                } => caught(&mut |e| {
                    e.set_stage_in_region_indirect(&indirect_buffer.raw, indirect_buffer_offset)
                }),
                ComputeCmd::SetTexture { texture, index } => {
                    let index = small(sel, index);
                    let texture = texture.map(|t| t.texels(sel));
                    caught(&mut |e| e.set_texture(texture, index))
                }
                ComputeCmd::SetTextures { textures, range } => {
                    let at = first(sel, range, &[textures.len()]);
                    listed(
                        textures,
                        |t| t.map(|t| t.texels(sel)),
                        |textures| caught(&mut |e| e.set_textures(textures, at)),
                    )
                }
                ComputeCmd::SetThreadgroupMemoryLength { length, index } => {
                    let (length, index) = (small(sel, length), small(sel, index));
                    caught(&mut |e| e.set_threadgroup_memory_length(length, index))
                }
                ComputeCmd::SetVisibleFunctionTable {
                    function_table,
                    buffer_index,
                } => {
                    let index = small(sel, buffer_index);
                    caught(&mut |e| {
                        e.set_visible_function_table(function_table.map(|t| &**t), index)
                    })
                }
                ComputeCmd::SetVisibleFunctionTables {
                    function_tables,
                    range,
                } => {
                    let at = first(sel, range, &[function_tables.len()]);
                    listed(
                        function_tables,
                        |t| t.map(|t| &**t),
                        |tables| caught(&mut |e| e.set_visible_function_tables(tables, at)),
                    )
                }
                ComputeCmd::UpdateFence { fence } => caught(&mut |e| e.update_fence(fence)),
                ComputeCmd::UseHeaps { heaps } => {
                    let heaps: Vec<&rt::Heap> = heaps.iter().flatten().map(|h| &h.raw).collect();
                    caught(&mut |e| e.use_heaps(&heaps))
                }
                ComputeCmd::UseResources {
                    resources: list,
                    usage,
                } => {
                    if list.iter().any(Option::is_none) && self.concurrent(sel) {
                        unrepresentable(
                            sel,
                            "a nil resource in a concurrent encoder, which ends Metal's process",
                        );
                    }
                    let some = resources(list);
                    let usage: ResourceUsage = metal_value(sel, "resource usage", usage);
                    let icbs = crate::gpu::icb::made_resident(list);
                    if !icbs.is_empty() {
                        // A kernel may copy any of their commands.
                        let progress = self.progress(sel);
                        for icb in &icbs {
                            icb.resident(
                                sel,
                                progress.as_ref(),
                                usage.contains(ResourceUsage::WRITE),
                            );
                        }
                    }
                    caught(&mut |e| e.use_resources(&some, usage))?;
                    if usage.contains(ResourceUsage::WRITE) {
                        plain(|| {
                            let mut writable = self.writable_icbs.lock();
                            writable.extend(icbs.iter().map(|icb| icb.clone_ref()));
                        });
                    }
                    if !icbs.is_empty() {
                        self.hold(sel, icbs);
                    }
                    Ok(())
                }
                ComputeCmd::WaitForFence { fence } => caught(&mut |e| e.wait_for_fence(fence)),
            }
        });
        let sent = recorded_answer.is_ok();
        crate::recorder::private::after(&mut recorded_call, self, snapshot, sent);
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    fn encode_end_do_while(
        &self,
        buffer: Option<&VulkanBuffer>,
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
            let sel = "encodeEndDoWhile:offset:comparison:referenceValue:";
            predication::end_do_while(self, sel, buffer, offset, comparison, reference_value)
        });
        crate::recorder::rec_done!(recorded_call, recorded_answer);
        recorded_answer
    }

    fn encode_end_if(&self) -> RaiseResult<bool> {
        predication::end_if(self, "encodeEndIf")
    }

    fn encode_end_while(&self) -> RaiseResult<bool> {
        predication::end_while(self, "encodeEndWhile")
    }
}

#[cfg(test)]
mod tests {
    use crate::core::platform::Platform as _;
    use crate::core::platform::command::{CommandBuffer, CommandQueue};
    use crate::core::platform::descriptor::TextureDescriptor as _;
    use crate::core::platform::device::GpuDevice;
    use crate::core::platform::resource::{Buffer as _, Library as _};
    use crate::core::platform::types::MtlSize;
    use crate::core::platform::values::{ComputePassDesc, TextureDesc};
    use crate::core::raise::death::DeathKind;
    use crate::core::testkit::capture::catch_death;
    use metal2vulkan::bind::SamplerDescriptor;
    use metal2vulkan::metal::SamplerAddressMode;

    use super::*;
    use crate::gpu::command::VulkanCommandBuffer;
    use crate::gpu::device::tests::dev;
    use crate::gpu::resource::tests::{fill, pattern, shared};
    use crate::gpu::resource::{VulkanComputePipeline, VulkanLibrary};
    use crate::platform::tests::platform;

    /// The tests' kernel that resets indirect commands through an argument buffer, which
    /// `build.rs` compiled from `icb_encode.metal`.
    const ICB_ENCODE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/icb_encode.metallib"));

    const ADD: &[u8] = include_bytes!("../../../../tests/shaders/compute_add.metallib");
    const TGMEM: &[u8] = include_bytes!("../../../../tests/shaders/compute_tgmem.metallib");
    const SAMPLE: &[u8] = include_bytes!("../../../../tests/shaders/compute_sample.metallib");

    /// `MTLCommandBufferStatusCompleted`.
    const COMPLETED: u64 = 4;

    /// The library of a metallib.
    fn library(bytes: &[u8]) -> VulkanLibrary {
        dev().new_library(bytes).unwrap().unwrap().unwrap()
    }

    /// The pipeline of the kernel `name` in a metallib.
    fn pipeline(bytes: &[u8], name: &str) -> VulkanComputePipeline {
        let f = library(bytes)
            .new_function(OnFault::Return, name)
            .unwrap()
            .unwrap();
        dev()
            .new_compute_pipeline_state_for_function(Some(&f))
            .unwrap()
            .unwrap()
    }

    /// A command buffer of a fresh queue.
    fn command_buffer() -> VulkanCommandBuffer {
        let q = dev().new_command_queue().unwrap();
        q.command_buffer(OnFault::Return).unwrap().unwrap()
    }

    /// The words of a Shared buffer.
    fn words(b: &VulkanBuffer) -> Vec<u32> {
        let n = usize::try_from(b.length() / 4).unwrap();
        // SAFETY: a Shared buffer's bytes, read after the GPU's writes completed.
        unsafe { std::slice::from_raw_parts(b.contents().as_ptr::<u32>(), n) }.to_vec()
    }

    /// A Shared buffer holding `w`.
    fn holding(w: &[u32]) -> VulkanBuffer {
        let b = shared(u64::try_from(w.len() * 4).unwrap());
        // SAFETY: the buffer's Shared bytes, as long as `w`, written before any command uses them.
        unsafe { std::ptr::copy_nonoverlapping(w.as_ptr(), b.contents().as_ptr::<u32>(), w.len()) };
        b
    }

    /// Sends `encode` to `e`, ends it, commits `cb` and waits for it to complete.
    fn run(
        cb: &VulkanCommandBuffer,
        e: &VulkanComputeEncoder,
        encode: Vec<ComputeCmd<'_, VulkanPlatform>>,
    ) {
        for cmd in encode {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        cb.commit(OnFault::Return).unwrap();
        cb.wait_until_completed(OnFault::Return).unwrap();
        assert_eq!((cb.status().0, cb.error()), (COMPLETED, None));
    }

    /// Runs `encode` in a serial compute encoder of a fresh command buffer.
    fn run_serial(encode: Vec<ComputeCmd<'_, VulkanPlatform>>) {
        let cb = command_buffer();
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        run(&cb, &e, encode);
    }

    /// The tests' kernel that writes texel (x, y) of texture 0.
    #[cfg(any(target_os = "linux", windows))]
    const TEXWRITE: &[u8] = include_bytes!("../../../../tests/shaders/compute_texwrite.metallib");

    /// A 16 × 16 `BGRA8Unorm` texture of `levels` mipmap levels and shader read and write usage at
    /// the start of a heap made with bit 24 over pages of the test's own
    /// (`resource::tests::guest_pages_of`), as the guest makes a read-only heap, with the pages and
    /// their length; the runtime makes the heap over a copy of them on Linux and Windows,
    /// announcing `copied-host-pages`, which dies in strict mode, so it is made under the default.
    #[cfg(any(target_os = "linux", windows))]
    fn bgra_in_read_only_heap(
        levels: u64,
    ) -> (
        crate::gpu::resource::VulkanTexture,
        crate::core::host::HostPtr,
        u64,
    ) {
        use crate::core::host::HostPtr;
        use crate::core::num::ToUsize as _;
        use crate::core::platform::HostRange;
        use crate::core::platform::resource::Heap as _;
        use crate::core::platform::values::{HeapDesc, OptionSet};

        use crate::gpu::resource::tests::{BIT_24, guest_pages_of, over_pages};

        let size = over_pages();
        let pages = HostPtr::from_addr(guest_pages_of(size).to_usize());
        // SAFETY: pages of the test's, left mapped until the process ends.
        let bytes = unsafe { HostRange::new(pages, size) };
        let heap = crate::answer::relaxed(|| {
            dev().new_heap_over_host_memory(
                bytes,
                &HeapDesc {
                    size,
                    heap_type: 1,
                    options: vec![
                        OptionSet::StorageMode(0),
                        OptionSet::HazardTrackingMode(0),
                        OptionSet::ResourceOptionsOr(BIT_24),
                    ],
                    pinned_gpu_address: None,
                },
            )
        })
        .unwrap()
        .heap
        .unwrap();
        let desc = platform().texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width: 16,
            height: 16,
            mipmap_level_count: levels,
            usage: 0x3,
            resource_options: Some(heap.resource_options() & !BIT_24),
            ..TextureDesc::default()
        });
        let texture = heap.new_texture(&desc, 0).unwrap().unwrap();
        (texture, pages, size)
    }

    /// A kernel's `write()` of a texture over a guest's read-only memory, a buffer made with bit 24
    /// (`resource::tests::read_only_buffer`) or a heap made with it over pages
    /// ([`bgra_in_read_only_heap`]), and a blit's copy into such a texture out of a buffer or a
    /// texture, each complete with no error, and the pages, and what a later command buffer reads
    /// of the texture, stay zeros, as Metal's GPU takes those writes nowhere (measured on the M3
    /// Max, macOS 27.0.1, metal2vulkan's `probes/readonly_gpu_writes.m`; the runtime's
    /// `read_only_view.rs`): the runtime's copy of the pages drops them. Linux's and Windows's: on
    /// macOS they go to Metal
    #[cfg(any(target_os = "linux", windows))]
    #[test]
    fn texture_writes_into_read_only_memory_are_dropped() {
        use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
        use crate::core::platform::os::vm::HostVm as _;
        use crate::core::platform::types::MtlOrigin;

        use crate::gpu::resource::tests::{bgra_over, host_bytes_at, read_only_buffer};

        let write = pipeline(TEXWRITE, "tex_write");
        let source = shared(16 * 64);
        // SAFETY: a Shared buffer of 16 rows of 64 bytes, which no command uses yet.
        unsafe { std::ptr::write_bytes(source.contents().as_ptr::<u8>(), 0x3c, 16 * 64) };
        let held = d_texture(0x5a);
        let (ro, pages) = read_only_buffer(1);
        let over = bgra_over(&ro, 0x3);
        let (in_heap, heap_pages, heap_size) = bgra_in_read_only_heap(1);
        let whole = MtlSize::new(16, 16, 1);
        let at = MtlOrigin::new(0, 0, 0);
        for (name, texture, (pages, length)) in [
            ("over a buffer", &over, (pages, 16 * 64)),
            ("in a heap", &in_heap, (heap_pages, heap_size)),
        ] {
            // The kernel's write.
            run_serial(vec![
                ComputeCmd::SetComputePipelineState {
                    state: Some(&write),
                },
                ComputeCmd::SetTexture {
                    texture: Some(texture),
                    index: 0,
                },
                ComputeCmd::DispatchThreads {
                    threads_per_grid: whole,
                    threads_per_threadgroup: MtlSize::new(4, 4, 1),
                },
            ]);
            // The blit's copies, out of a buffer and out of a texture.
            for copy in [
                BlitCmd::CopyBufferToTexture {
                    source_buffer: &source,
                    source_offset: 0,
                    source_bytes_per_row: 64,
                    source_bytes_per_image: 16 * 64,
                    source_size: whole,
                    destination_texture: texture,
                    destination_slice: 0,
                    destination_level: 0,
                    destination_origin: at,
                },
                BlitCmd::CopyTextureRegion {
                    source_texture: &held,
                    source_slice: 0,
                    source_level: 0,
                    source_origin: at,
                    source_size: whole,
                    destination_texture: texture,
                    destination_slice: 0,
                    destination_level: 0,
                    destination_origin: at,
                },
            ] {
                let cb = command_buffer();
                let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
                e.encode(OnFault::Return, copy).unwrap();
                e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
                cb.commit(OnFault::Return).unwrap();
                cb.wait_until_completed(OnFault::Return).unwrap();
                assert_eq!((cb.status().0, cb.error()), (COMPLETED, None), "{name}");
            }
            let length = usize::try_from(length).unwrap();
            assert!(
                host_bytes_at(pages, length).iter().all(|&b| b == 0),
                "{name}: the pages"
            );
            // What a later command buffer reads of the texture.
            let out = shared(16 * 64);
            let cb = command_buffer();
            let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
            let read = BlitCmd::CopyTextureToBuffer {
                source_texture: texture,
                source_slice: 0,
                source_level: 0,
                source_origin: at,
                source_size: whole,
                destination_buffer: &out,
                destination_offset: 0,
                destination_bytes_per_row: 64,
                destination_bytes_per_image: 16 * 64,
            };
            e.encode(OnFault::Return, read).unwrap();
            e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
            cb.commit(OnFault::Return).unwrap();
            cb.wait_until_completed(OnFault::Return).unwrap();
            assert!(words(&out).iter().all(|&w| w == 0), "{name}: read back");
        }
        drop((over, ro, in_heap));
        assert!(
            crate::remap::VulkanVm::new()
                .deallocate(pages, crate::os::PAGE)
                .is_success()
        );
    }

    /// A blit's other writes of a texture over a guest's read-only memory, a buffer made with bit
    /// 24 (`resource::tests::read_only_buffer`) or a heap made with it
    /// ([`bgra_in_read_only_heap`]), complete with no error, the writes dropped and the buffer's
    /// pages staying zeros, as the M3 Max's GPU drops them (plain Metal, macOS 27.0.1, m3-max-29292
    /// and m3-max-29309): a fill by a colour or by bytes and an invalidation (the `fillTexture:`
    /// and `invalidateCompressedTexture:` SPI, which the Radeon Pro 5300M and UHD Graphics 630
    /// under macOS 26.7.1 end the process at with bit 24 or without), and mipmaps' generation,
    /// after which level 1 of a heap's texture whose level 0 the CPU wrote reads zeros, where it
    /// reads level 0's texels without bit 24 (`mipmaps-shared`, m3-max-29309). The two x86 GPUs
    /// complete mipmaps' generation too, writing it. Linux's and Windows's: on macOS they go to
    /// Metal
    #[cfg(any(target_os = "linux", windows))]
    #[test]
    fn blit_writes_of_a_texture_in_read_only_memory_are_dropped() {
        use crate::core::platform::HostRange;
        use crate::core::platform::blit::{BlitCmd, BlitEncoder as _};
        use crate::core::platform::os::vm::HostVm as _;
        use crate::core::platform::resource::Texture as _;
        use crate::core::platform::types::{ClearColor, MtlOrigin, MtlRegion};

        use crate::gpu::resource::tests::{bgra_over, host_bytes_at, read_only_buffer};

        let blit = |cmd| {
            let cb = command_buffer();
            let e = cb.blit_encoder(OnFault::Return).unwrap().unwrap();
            e.encode(OnFault::Return, cmd).unwrap();
            e.encode(OnFault::Return, BlitCmd::EndEncoding).unwrap();
            cb.commit(OnFault::Return).unwrap();
            cb.wait_until_completed(OnFault::Return).unwrap();
            (cb.status().0, cb.error())
        };
        let (ro, pages) = read_only_buffer(1);
        let over = bgra_over(&ro, 0x3);
        let (in_heap, _, _) = bgra_in_read_only_heap(2);
        let at = MtlOrigin::new(0, 0, 0);
        let region = MtlRegion::new(at, MtlSize::new(16, 16, 1));
        let texel = [0x77u8; 4];
        let white = ClearColor {
            red: 1.0,
            green: 1.0,
            blue: 1.0,
            alpha: 1.0,
        };
        for (name, texture) in [("over a buffer", &over), ("in a heap", &in_heap)] {
            let fills = [
                BlitCmd::FillTextureColor {
                    texture,
                    level: 0,
                    slice: 0,
                    region,
                    color: white,
                    pixel_format: 80,
                },
                BlitCmd::FillTexture {
                    texture,
                    level: 0,
                    slice: 0,
                    region,
                    bytes: HostRange::of_slice(&texel),
                },
                BlitCmd::InvalidateCompressedTexture { texture },
            ];
            for fill in fills {
                let sel = fill.selector();
                assert_eq!(blit(fill), (COMPLETED, None), "{name}: {sel}");
            }
        }
        assert!(host_bytes_at(pages, 16 * 64).iter().all(|&b| b == 0));
        let level_0 = [0x3cu8; 16 * 64];
        in_heap
            .replace_region(region, 0, 0, HostRange::of_slice(&level_0), 64, 0)
            .unwrap();
        let mipmaps = BlitCmd::GenerateMipmaps {
            texture: Some(&in_heap),
        };
        assert_eq!(blit(mipmaps), (COMPLETED, None), "mipmaps");
        let out = shared(8 * 32);
        let read = BlitCmd::CopyTextureToBuffer {
            source_texture: &in_heap,
            source_slice: 0,
            source_level: 1,
            source_origin: at,
            source_size: MtlSize::new(8, 8, 1),
            destination_buffer: &out,
            destination_offset: 0,
            destination_bytes_per_row: 32,
            destination_bytes_per_image: 8 * 32,
        };
        assert_eq!(blit(read), (COMPLETED, None), "level 1 read back");
        assert!(words(&out).iter().all(|&w| w == 0), "level 1");
        drop((over, ro, in_heap));
        assert!(
            crate::remap::VulkanVm::new()
                .deallocate(pages, crate::os::PAGE)
                .is_success()
        );
    }

    /// A 16 × 16 `BGRA8Unorm` texture of the device's memory, every byte `byte`.
    #[cfg(any(target_os = "linux", windows))]
    fn d_texture(byte: u8) -> crate::gpu::resource::VulkanTexture {
        use crate::core::platform::HostRange;
        use crate::core::platform::resource::Texture as _;
        use crate::core::platform::types::{MtlOrigin, MtlRegion};

        let desc = platform().texture_descriptor(&TextureDesc {
            pixel_format: 80,
            width: 16,
            height: 16,
            usage: 0x1,
            resource_options: Some(0),
            ..TextureDesc::default()
        });
        let t = dev().new_texture(OnFault::Return, &desc).unwrap().unwrap();
        let region = MtlRegion::new(MtlOrigin::new(0, 0, 0), MtlSize::new(16, 16, 1));
        let texels = [byte; 16 * 64];
        t.replace_region(region, 0, 0, HostRange::of_slice(&texels), 64, 0)
            .unwrap();
        t
    }

    /// A dispatch whose kernel stores into a guest's read-only buffer (made with bit 24,
    /// `resource::tests::read_only_buffer`), bound at the slot it writes and made resident for
    /// writing, completes with no error, its stores dropped, the pages staying zeros, as Metal
    /// drops a kernel's stores there (`crate::gpu::writes`'s docs): through the runtime's copy of
    /// the pages on Linux and Windows, and Metal's read-only view on macOS. The kernel reads it as
    /// the pages
    #[test]
    fn a_kernel_store_into_a_read_only_buffer_is_dropped() {
        use crate::core::platform::os::vm::HostVm as _;

        use crate::gpu::resource::tests::{host_bytes_at, read_only_buffer};
        use crate::os::PAGE;

        let add = pipeline(ADD, "add_one");
        let (ro, pages) = read_only_buffer(1);
        let out = shared(16);
        let four = || ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(4, 1, 1),
            threads_per_threadgroup: MtlSize::new(4, 1, 1),
        };
        run_serial(vec![
            ComputeCmd::SetComputePipelineState { state: Some(&add) },
            ComputeCmd::SetBuffers {
                buffers: &[Some(&out), Some(&ro)],
                offsets: &[0, 0],
                range: NsRange::new(0, 2),
            },
            four(),
        ]);
        assert_eq!(words(&out), [1; 4], "read as the pages");
        let source = holding(&[7; 4]);
        run_serial(vec![
            ComputeCmd::SetComputePipelineState { state: Some(&add) },
            ComputeCmd::SetBuffers {
                buffers: &[Some(&ro), Some(&source)],
                offsets: &[0, 0],
                range: NsRange::new(0, 2),
            },
            ComputeCmd::UseResources {
                resources: &[Some(ResourceRef::Buffer(&ro))],
                usage: 3,
            },
            four(),
        ]);
        assert!(host_bytes_at(pages, 16).iter().all(|&b| b == 0), "dropped");
        drop((ro, source, out));
        assert!(
            crate::remap::VulkanVm::new()
                .deallocate(pages, PAGE)
                .is_success()
        );
    }

    /// Prints how long 512 one-thread dispatches take to encode after one `useResources` of 0,
    /// 1, 256 and 4096 Shared buffers, and of one buffer 4096 times: through this platform's
    /// compute encoder, and through the runtime's alone, on the same device and pipeline, the
    /// best of 3 runs each.
    ///
    /// Measured on the M3 Max (MoltenVK, a dev build), platform against runtime alone, in ms:
    /// 6.6 against 5.7 with none resident, 15.8 against 15.1 with one, 99.5 against 97.3 with
    /// 256, 1542 against 1498 with 4096, and 22.3 against 15.8 with one buffer 4096 times. The
    /// runtime takes its dispatch memo only in an encoder that made nothing resident, so each
    /// dispatch after a `useResources` settles its bind state again (about 18 us more), and
    /// that settling copies, shapes and hashes every resident resource (about 0.7 us more for
    /// each).
    #[test]
    #[ignore = "a measurement, run by hand: prints the times"]
    fn dispatches_after_a_large_residency() {
        use crate::core::platform::os::Clock as _;
        let now = || crate::os::OsClock::new().now_mono();
        let add = pipeline(ADD, "add_one");
        let (input, out) = (shared(1024), shared(1024));
        let resident: Vec<VulkanBuffer> = (0..4096).map(|_| shared(256)).collect();
        let d = dev();
        let raw_q = d.raw.new_command_queue().unwrap();
        let one = vec![&resident[0]; 4096];
        let all: Vec<_> = resident.iter().collect();
        for (name, list) in [
            ("0 buffers", &all[..0]),
            ("1 buffer", &all[..1]),
            ("256 buffers", &all[..256]),
            ("4096 buffers", &all[..]),
            ("one buffer 4096 times", &one[..]),
        ] {
            let refs: Vec<_> = (list.iter())
                .map(|b| Some(ResourceRef::Buffer(*b)))
                .collect();
            let raw_refs: Vec<_> = (list.iter())
                .map(|b| rt::ResourceRef::Buffer(&b.raw))
                .collect();
            let (mut ours, mut runtime) = (u128::MAX, u128::MAX);
            for _ in 0..3 {
                let cb = command_buffer();
                let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
                let setup = [
                    ComputeCmd::SetComputePipelineState { state: Some(&add) },
                    ComputeCmd::SetBuffers {
                        buffers: &[Some(&out), Some(&input)],
                        offsets: &[0, 0],
                        range: NsRange::new(0, 2),
                    },
                    ComputeCmd::UseResources {
                        resources: &refs,
                        usage: 1,
                    },
                ];
                for cmd in setup {
                    e.encode(OnFault::Return, cmd).unwrap();
                }
                let start = now();
                for _ in 0..512 {
                    let dispatch = ComputeCmd::DispatchThreads {
                        threads_per_grid: MtlSize::new(1, 1, 1),
                        threads_per_threadgroup: MtlSize::new(1, 1, 1),
                    };
                    e.encode(OnFault::Return, dispatch).unwrap();
                }
                ours = ours.min(now().saturating_duration_since(start).as_micros());
                e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
                cb.commit(OnFault::Return).unwrap();
                cb.wait_until_completed(OnFault::Return).unwrap();

                let cb = raw_q.command_buffer().unwrap();
                let mut e = cb.compute_command_encoder().unwrap();
                e.set_compute_pipeline_state(&add.raw).unwrap();
                e.set_buffers(&[Some(&out.raw), Some(&input.raw)], &[0, 0], 0)
                    .unwrap();
                e.use_resources(&raw_refs, ResourceUsage::READ).unwrap();
                let start = now();
                for _ in 0..512 {
                    let one = rt::Size::new(1, 1, 1);
                    e.dispatch_threads(one, one).unwrap();
                }
                runtime = runtime.min(now().saturating_duration_since(start).as_micros());
                e.end_encoding().unwrap();
                cb.commit().unwrap();
                cb.wait_until_completed().unwrap();
            }
            println!(
                "{name} resident: 512 dispatches through the platform {ours} us, through the \
                 runtime alone {runtime} us"
            );
        }
    }

    /// Dispatches run their kernels: threads over bound buffers, threadgroups after a buffer
    /// offset, threadgroups from an indirect buffer, threadgroup memory of the length set; with
    /// a barrier, a use and the reinterpretation flush between them
    #[test]
    fn dispatches_run_kernels() {
        let add = pipeline(ADD, "add_one");
        let input = holding(&(0..256).map(|i| i * 5).collect::<Vec<_>>());
        let added: Vec<u32> = (0..256).map(|i| i * 5 + 1).collect();
        let out = shared(1024);
        run_serial(vec![
            ComputeCmd::SetComputePipelineState { state: Some(&add) },
            ComputeCmd::SetBuffers {
                buffers: &[Some(&out), Some(&input)],
                offsets: &[0, 0],
                range: NsRange::new(0, 2),
            },
            ComputeCmd::UseResources {
                resources: &[Some(ResourceRef::Buffer(&input))],
                usage: 1,
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(256, 1, 1),
                threads_per_threadgroup: MtlSize::new(64, 1, 1),
            },
            ComputeCmd::MemoryBarrierWithScope { scope: 1 },
            ComputeCmd::CompressedTextureReinterpretationFlush,
        ]);
        assert_eq!(words(&out), added);
        // The second half, through offsets.
        let half = shared(1024);
        run_serial(vec![
            ComputeCmd::SetComputePipelineState { state: Some(&add) },
            ComputeCmd::SetBuffers {
                buffers: &[Some(&half), Some(&input)],
                offsets: &[0, 0],
                range: NsRange::new(0, 2),
            },
            ComputeCmd::SetBufferOffset {
                offset: 512,
                index: 0,
            },
            ComputeCmd::SetBufferOffset {
                offset: 512,
                index: 1,
            },
            ComputeCmd::DispatchThreadgroups {
                threadgroups_per_grid: MtlSize::new(2, 1, 1),
                threads_per_threadgroup: MtlSize::new(64, 1, 1),
            },
        ]);
        assert_eq!(words(&half)[..128], [0; 128]);
        assert_eq!(words(&half)[128..], added[128..]);
        // Four groups, from an indirect buffer.
        let (arguments, indirect) = (holding(&[4, 1, 1]), shared(1024));
        run_serial(vec![
            ComputeCmd::SetComputePipelineState { state: Some(&add) },
            ComputeCmd::SetBuffers {
                buffers: &[Some(&indirect), Some(&input)],
                offsets: &[0, 0],
                range: NsRange::new(0, 2),
            },
            ComputeCmd::DispatchThreadgroupsIndirect {
                indirect_buffer: Some(&arguments),
                indirect_buffer_offset: 0,
                threads_per_threadgroup: MtlSize::new(64, 1, 1),
            },
        ]);
        assert_eq!(words(&indirect), added);
        // Each group of 64 reversed through threadgroup memory 0.
        let (reverse, reversed) = (pipeline(TGMEM, "tg_reverse"), shared(1024));
        run_serial(vec![
            ComputeCmd::SetComputePipelineState {
                state: Some(&reverse),
            },
            ComputeCmd::SetBuffers {
                buffers: &[Some(&reversed)],
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
            ComputeCmd::SetThreadgroupMemoryLength {
                length: 256,
                index: 0,
            },
            ComputeCmd::DispatchThreadgroups {
                threadgroups_per_grid: MtlSize::new(4, 1, 1),
                threads_per_threadgroup: MtlSize::new(64, 1, 1),
            },
        ]);
        let expected: Vec<u32> = (0..256u32)
            .map(|gid| (gid / 64 * 64 + 63 - gid % 64) * 3)
            .collect();
        assert_eq!(words(&reversed), expected);
    }

    /// A nil entry of useResources:count:usage:, useHeaps:count: and
    /// memoryBarrierWithResources:count: is skipped, and a buffer offset at a slot of no buffer,
    /// which the runtime takes, changes nothing, and the pass runs, as on Metal; a nil heap is
    /// skipped in a concurrent encoder too, and a nil entry in useResources:count:usage: of one,
    /// which ends Metal's process, dies at its selector
    #[test]
    fn nil_resources_are_skipped_but_in_a_concurrent_encoders_residency() {
        let add = pipeline(ADD, "add_one");
        let input = holding(&[1, 2, 3, 4]);
        let out = shared(16);
        run_serial(vec![
            ComputeCmd::UseResources {
                resources: &[None, Some(ResourceRef::Buffer(&input))],
                usage: 1,
            },
            ComputeCmd::MemoryBarrierWithResources { resources: &[None] },
            ComputeCmd::UseHeaps { heaps: &[None] },
            ComputeCmd::MemoryBarrierWithResources {
                resources: &[Some(ResourceRef::Buffer(&out)), None],
            },
            ComputeCmd::SetBufferOffset {
                offset: 0,
                index: 30,
            },
            ComputeCmd::SetBufferOffsetAttributeStride {
                offset: 0,
                stride: 0,
                index: 29,
            },
            ComputeCmd::SetComputePipelineState { state: Some(&add) },
            ComputeCmd::SetBuffers {
                buffers: &[Some(&out), Some(&input)],
                offsets: &[0, 0],
                range: NsRange::new(0, 2),
            },
            ComputeCmd::DispatchThreads {
                threads_per_grid: MtlSize::new(4, 1, 1),
                threads_per_threadgroup: MtlSize::new(4, 1, 1),
            },
        ]);
        assert_eq!(words(&out), [2, 3, 4, 5]);
        let cb = command_buffer();
        let e = cb
            .compute_encoder_for_pass(&ComputePassDesc {
                dispatch_type: Some(1),
            })
            .unwrap()
            .unwrap();
        e.encode(OnFault::Return, ComputeCmd::UseHeaps { heaps: &[None] })
            .unwrap();
        let site = catch_death(|| {
            e.encode(
                OnFault::Return,
                ComputeCmd::UseResources {
                    resources: &[Some(ResourceRef::Buffer(&input)), None],
                    usage: 1,
                },
            )
        })
        .unwrap_err();
        assert_eq!(
            (site.kind, site.file),
            (DeathKind::Abort, "useResources:count:usage:")
        );
        e.encode(
            OnFault::Return,
            ComputeCmd::MemoryBarrierWithResources { resources: &[None] },
        )
        .unwrap();
    }

    /// A sampler's address mode decides what a kernel samples left of a texture: repeat reads
    /// texel i, clamp to edge texel 0. The samplers are the runtime's, as this platform's
    /// `newSamplerStateWithDescriptor:` is another's to build
    #[test]
    fn kernels_sample_through_bound_samplers() {
        let sampler = |mode: SamplerAddressMode| {
            let mut desc = SamplerDescriptor::default();
            desc.s_address_mode = mode;
            desc.t_address_mode = mode;
            Obj::new(dev().raw.new_sampler_state(&desc).unwrap())
        };
        let (repeat, edge) = (
            sampler(SamplerAddressMode::Repeat),
            sampler(SamplerAddressMode::ClampToEdge),
        );
        let desc = platform().texture_descriptor(&TextureDesc {
            pixel_format: 70,
            width: 4,
            height: 1,
            usage: 1,
            ..TextureDesc::default()
        });
        assert_eq!(desc.pixel_format(), 70);
        let t = dev().new_texture(OnFault::Return, &desc).unwrap().unwrap();
        let texels = pattern(4, 1);
        fill(&t, &texels);
        let texel = |i: usize| u32::from_le_bytes(texels[i * 4..i * 4 + 4].try_into().unwrap());
        let k = pipeline(SAMPLE, "sample_k");
        for (s, expected) in [
            (&repeat, [texel(0), texel(1), texel(2), texel(3)]),
            (&edge, [texel(0); 4]),
        ] {
            let out = shared(16);
            run_serial(vec![
                ComputeCmd::SetComputePipelineState { state: Some(&k) },
                ComputeCmd::SetTexture {
                    texture: Some(&t),
                    index: 0,
                },
                ComputeCmd::SetSamplerStates {
                    samplers: &[Some(s)],
                    range: NsRange::new(0, 1),
                },
                ComputeCmd::SetBuffers {
                    buffers: &[Some(&out)],
                    offsets: &[0],
                    range: NsRange::new(0, 1),
                },
                ComputeCmd::DispatchThreads {
                    threads_per_grid: MtlSize::new(4, 1, 1),
                    threads_per_threadgroup: MtlSize::new(4, 1, 1),
                },
            ]);
            assert_eq!(words(&out), expected);
        }
    }

    /// A device cache the host's video writes from outside the runtime (NVDEC's copy within the
    /// GPU, through CUDA's import of the cache's exported memory) reads each picture written
    /// there through a texture over it: an exportable `Private` buffer, its memory imported into
    /// a second `VkDevice` of the same physical device, which fills it with a new byte three
    /// times, each fill waited for and said done (`external_write_done`), and a texture over it
    /// laid as CoreVideo lays a frame's plane, 32 bytes in, which the runtime keeps as a shadow
    /// of the bytes rather than a linear image over them, sampled by a kernel in a command buffer
    /// of its own after each fill. Without the call the runtime's shadow kept what its first copy
    /// in read, as it does where only its own commands write a buffer, and the kernel read the
    /// first fill's byte after every later one. Linux's: MoltenVK exports no buffer's memory.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_device_cache_another_api_writes_reads_each_write_through_a_texture_over_it() {
        use ash::vk;
        use std::os::fd::IntoRawFd as _;

        use crate::core::platform::resource::Buffer as _;

        const LENGTH: u64 = 64 * 1024;
        const OFFSET: u64 = 32;
        const WIDTH: u64 = 200;
        const HEIGHT: u64 = 16;
        const BYTES_PER_ROW: u64 = 256;
        /// `MTLResourceStorageModePrivate`.
        const PRIVATE: u64 = 2 << 4;
        let d = dev();
        let made = d.new_exportable_buffer_of_length(OnFault::Return, LENGTH, PRIVATE, None);
        let cache = made.expect("no raise").expect("an exportable buffer");
        let exported = cache.export_memory().expect("its memory");
        // The export's memory type, which the import names: the runtime's own answer.
        let memory_type_index = (cache.raw.export_memory_fd())
            .expect("its memory")
            .memory_type_index;
        let desc = platform().texture_descriptor(&TextureDesc {
            pixel_format: 10,
            width: WIDTH,
            height: HEIGHT,
            usage: 1,
            resource_options: Some(PRIVATE),
            ..TextureDesc::default()
        });
        let texture = (cache.new_texture(OnFault::Return, &desc, OFFSET, BYTES_PER_ROW))
            .expect("no raise")
            .expect("an R8Unorm texture over the cache");
        let sampler = {
            let mut desc = SamplerDescriptor::default();
            desc.s_address_mode = SamplerAddressMode::ClampToEdge;
            desc.t_address_mode = SamplerAddressMode::ClampToEdge;
            Obj::new(d.raw.new_sampler_state(&desc).unwrap())
        };
        let k = pipeline(SAMPLE, "sample_k");

        // SAFETY: the handles are used while `d` lives; the second device is the test's own and
        // destroyed below, after its work.
        let handles = unsafe { d.raw.raw() };
        let (instance, physical) = (handles.instance, handles.physical_device);
        let mut features12 = vk::PhysicalDeviceVulkan12Features::default();
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut features12);
        // SAFETY: the physical device is the instance's.
        unsafe { instance.get_physical_device_features2(physical, &mut features) };
        let device_address = features12.buffer_device_address == vk::TRUE;
        let family = handles.queue_family;
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let extensions = [ash::khr::external_memory_fd::NAME.as_ptr()];
        let mut enabled12 =
            vk::PhysicalDeviceVulkan12Features::default().buffer_device_address(device_address);
        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions)
            .push_next(&mut enabled12);
        // SAFETY: the physical device is the instance's and lists the extension (the export ran).
        let second = unsafe { instance.create_device(physical, &info, None) }.expect("a device");
        // The buffer the import binds, made as the runtime makes an exportable one: a dedicated
        // import names a buffer of the same parameters.
        use vk::BufferUsageFlags as U;
        let mut usage = U::TRANSFER_SRC
            | U::TRANSFER_DST
            | U::UNIFORM_BUFFER
            | U::STORAGE_BUFFER
            | U::UNIFORM_TEXEL_BUFFER
            | U::STORAGE_TEXEL_BUFFER
            | U::INDEX_BUFFER
            | U::VERTEX_BUFFER
            | U::INDIRECT_BUFFER;
        if device_address {
            usage |= U::SHADER_DEVICE_ADDRESS;
        }
        let mut external = vk::ExternalMemoryBufferCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD);
        let buffer_info = vk::BufferCreateInfo::default()
            .size(LENGTH)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .push_next(&mut external);
        // SAFETY: the second device is live and the info valid for it.
        let imported = unsafe { second.create_buffer(&buffer_info, None) }.expect("a buffer");
        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD)
            .fd(exported.fd.into_raw_fd());
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().buffer(imported);
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let mut allocate = vk::MemoryAllocateInfo::default()
            .allocation_size(exported.allocation_size)
            .memory_type_index(memory_type_index)
            .push_next(&mut import);
        if exported.dedicated {
            allocate = allocate.push_next(&mut dedicated);
        }
        if device_address {
            allocate = allocate.push_next(&mut flags);
        }
        // SAFETY: the descriptor is the export's, of the same physical device, its size and type
        // the allocation's; the import owns it once made.
        let memory = unsafe { second.allocate_memory(&allocate, None) }.expect("the import");
        // SAFETY: the buffer and the memory are the second device's; the cache lies at the
        // export's offset in the allocation.
        unsafe { second.bind_buffer_memory(imported, memory, exported.offset) }.expect("bound");
        // SAFETY: the family is the second device's, made with one queue.
        let queue = unsafe { second.get_device_queue(family, 0) };
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        // SAFETY: as above.
        let pool = unsafe { second.create_command_pool(&pool_info, None) }.expect("a pool");
        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        // SAFETY: as above.
        let command = unsafe { second.allocate_command_buffers(&allocate_info) }.expect("one")[0];
        // The second device fills the cache's bytes with `byte` and waits for it.
        let fill = |byte: u8| {
            let begin = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            let commands = [command];
            let submit = [vk::SubmitInfo::default().command_buffers(&commands)];
            // SAFETY: the command buffer is the second device's, recorded and submitted here
            // alone, its fill within the imported buffer; the queue is waited idle before the
            // next recording.
            unsafe {
                second.begin_command_buffer(command, &begin).unwrap();
                second.cmd_fill_buffer(command, imported, 0, LENGTH, u32::from(byte) * 0x0101_0101);
                second.end_command_buffer(command).unwrap();
                second
                    .queue_submit(queue, &submit, vk::Fence::null())
                    .unwrap();
                second.queue_wait_idle(queue).unwrap();
            }
        };

        // Each fill's byte, and the words the kernel sampled after it.
        let mut read = Vec::new();
        for byte in [0x11u8, 0x22, 0x33] {
            fill(byte);
            cache.external_write_done().expect("the acquire");
            let out = shared(32);
            run_serial(vec![
                ComputeCmd::SetComputePipelineState { state: Some(&k) },
                ComputeCmd::SetTexture {
                    texture: Some(&texture),
                    index: 0,
                },
                ComputeCmd::SetSamplerStates {
                    samplers: &[Some(&sampler)],
                    range: NsRange::new(0, 1),
                },
                ComputeCmd::SetBuffers {
                    buffers: &[Some(&out)],
                    offsets: &[0],
                    range: NsRange::new(0, 1),
                },
                ComputeCmd::DispatchThreads {
                    threads_per_grid: MtlSize::new(8, 1, 1),
                    threads_per_threadgroup: MtlSize::new(8, 1, 1),
                },
            ]);
            read.push((byte, words(&out)));
        }
        // SAFETY: the second device's objects, its queue idle; the cache outlives the import.
        unsafe {
            second.destroy_command_pool(pool, None);
            second.destroy_buffer(imported, None);
            second.free_memory(memory, None);
            second.destroy_device(None);
        }
        // An R8Unorm texel sampled and packed as unorm4x8: the byte, zeros, and an alpha of one.
        let wrong: Vec<_> = (read.iter())
            .filter(|(byte, words)| words.iter().any(|&w| w != u32::from(*byte) | 0xff00_0000))
            .collect();
        assert!(
            wrong.is_empty(),
            "(the byte written, the words sampled after it) on {}: {wrong:x?}",
            d.raw.name()
        );
    }

    /// A nil pipeline, slices of another length than their range, a scope no Metal value is and
    /// a command after endEncoding, predication's too, die at their selectors
    #[test]
    fn compute_refusals() {
        let cb = command_buffer();
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        let b = shared(16);
        let dies = |cmd: ComputeCmd<'_, VulkanPlatform>| {
            let sel = cmd.selector();
            let site = catch_death(|| e.encode(OnFault::Return, cmd)).unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
        };
        dies(ComputeCmd::SetComputePipelineState { state: None });
        dies(ComputeCmd::SetBuffers {
            buffers: &[Some(&b)],
            offsets: &[0, 0],
            range: NsRange::new(0, 2),
        });
        dies(ComputeCmd::MemoryBarrierWithScope { scope: 0x80 });
        e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
        dies(ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(1, 1, 1),
            threads_per_threadgroup: MtlSize::new(1, 1, 1),
        });
        dies(ComputeCmd::EncodeStartElse);
        let site = catch_death(|| e.encode_end_if()).unwrap_err();
        assert_eq!((site.kind, site.file), (DeathKind::Abort, "encodeEndIf"));
    }

    /// A direct dispatch, in each form, of a pipeline that can encode indirect commands runs in an
    /// encoder that made resident for writing an indirect command buffer numbered past a
    /// reservation of 2 and 2^23 + 1, which the runtime takes as Metal does, so the buffers'
    /// IDs, 1 then 3, are the runtime's and Metal's alike. So do one before it, one where the
    /// buffer was made resident only for reading, one of a pipeline that cannot encode indirect
    /// commands, one whose kernel resets the buffer's commands through its ID, and an execution
    /// of the buffer, whose command dispatches nothing
    #[test]
    fn a_dispatch_after_an_icb_made_writable_runs() {
        use crate::core::platform::icb::IndirectCommandBuffer as _;
        use crate::core::platform::values::IcbDesc;

        let d = dev();
        d.reserve_resource_indices(5, &[2, (1 << 23) + 1]).unwrap();
        let (add, reset) = (
            pipeline(ADD, "add_one"),
            pipeline(ICB_ENCODE, "reset_commands"),
        );
        assert!(!add.can_encode_indirect_commands() && reset.can_encode_indirect_commands());
        let make = || {
            d.new_indirect_command_buffer(
                &IcbDesc {
                    command_types: 0x40,
                    max_kernel_buffer_bind_count: 2,
                    ..IcbDesc::default()
                },
                1,
                0,
            )
            .unwrap()
            .unwrap()
        };
        let (plain, icb) = (make(), make());
        assert_eq!((plain.gpu_resource_id(), icb.gpu_resource_id()), (1, 3));
        let (input, out) = (holding(&[1, 2, 3, 4]), holding(&[0; 4]));
        let id = plain.gpu_resource_id();
        let held = holding(&[u32::try_from(id).unwrap(), 0]);
        let bound = [Some(&out), Some(&input)];
        let the_held = [Some(&held)];
        let used = [Some(ResourceRef::Buffer(&out))];
        let the_icb = [Some(ResourceRef::Icb(&icb))];
        let the_plain = [Some(ResourceRef::Icb(&plain))];
        let set = |state| {
            vec![
                ComputeCmd::SetComputePipelineState { state: Some(state) },
                ComputeCmd::SetBuffers {
                    buffers: &bound,
                    offsets: &[0, 0],
                    range: NsRange::new(0, 2),
                },
                ComputeCmd::UseResources {
                    resources: &used,
                    usage: 3,
                },
            ]
        };
        let dispatch = || ComputeCmd::DispatchThreads {
            threads_per_grid: MtlSize::new(1, 1, 1),
            threads_per_threadgroup: MtlSize::new(1, 1, 1),
        };
        let resident = |usage| ComputeCmd::UseResources {
            resources: &the_icb,
            usage,
        };
        let execution = ComputeCmd::ExecuteCommandsInBuffer {
            indirect_command_buffer: &icb,
            execution_range: NsRange::new(0, 1),
        };
        let mut ran = set(&reset);
        ran.extend([dispatch(), resident(1), dispatch()]);
        run_serial(ran);
        let mut ran = set(&reset);
        ran.extend([resident(3), execution]);
        run_serial(ran);
        let mut ran = set(&add);
        ran.extend([resident(3), dispatch()]);
        run_serial(ran);
        let mut ran = set(&reset);
        ran.extend([
            ComputeCmd::SetBuffers {
                buffers: &the_held,
                offsets: &[0],
                range: NsRange::new(0, 1),
            },
            ComputeCmd::UseResources {
                resources: &the_plain,
                usage: 3,
            },
            dispatch(),
        ]);
        run_serial(ran);
        // Threads per grid, then per threadgroup: what the thread form reads.
        let indirect = holding(&[1; 6]);
        for (usage, last) in [
            (2, dispatch()),
            (3, dispatch()),
            (
                3,
                ComputeCmd::DispatchThreadgroups {
                    threadgroups_per_grid: MtlSize::new(1, 1, 1),
                    threads_per_threadgroup: MtlSize::new(1, 1, 1),
                },
            ),
            (
                3,
                ComputeCmd::DispatchThreadsIndirect {
                    indirect_buffer: Some(&indirect),
                    indirect_buffer_offset: 0,
                },
            ),
            (
                3,
                ComputeCmd::DispatchThreadgroupsIndirect {
                    indirect_buffer: Some(&indirect),
                    indirect_buffer_offset: 0,
                    threads_per_threadgroup: MtlSize::new(1, 1, 1),
                },
            ),
        ] {
            let mut ran = set(&reset);
            ran.extend([resident(usage), last]);
            run_serial(ran);
        }
    }

    /// What a compute encoder's `setBuffers:` costs the thread that encodes it (a measurement, not
    /// a check; run it by name with `--run-ignored only --no-capture` in a release build, holding
    /// the host's GPU): `BIND_COST_ROUNDS` sends (default 4,096) of 8 buffers at indices 0 to 7,
    /// as a kernel's binds before each of its dispatches, in two shapes. `same`: an upload chunk
    /// at index 0 at its next 256-byte offset, as the bundle sends `setBytes:`, and the same 7
    /// buffers after it each time; `fresh`: 8 buffers of a pool of 64 each time, none bound at its
    /// index the send before. Prints each shape's user-space instructions and cycles a buffer
    /// bound, the median of 5 runs, each an encoder of its own.
    #[cfg(target_os = "linux")]
    #[test]
    #[ignore = "a measurement: run it by name"]
    fn set_buffers_cost() {
        let rounds: u64 = std::env::var("BIND_COST_ROUNDS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4096);
        let chunk = shared(0x1_0000);
        let pool: Vec<VulkanBuffer> = (0..64).map(|_| shared(256)).collect();
        let offsets_of = |r: u64| -> [u64; 8] {
            let mut offsets = [0; 8];
            offsets[0] = (r * 256) % 0x1_0000;
            offsets
        };
        for shape in ["same", "fresh"] {
            let mut figures: [Vec<u64>; 2] = Default::default();
            for _ in 0..5 {
                let cb = command_buffer();
                let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
                let mut spent = [0u64; 2];
                for r in 0..rounds {
                    let mut buffers: [Option<&VulkanBuffer>; 8] = [Some(&chunk); 8];
                    for (i, slot) in buffers.iter_mut().enumerate().skip(1) {
                        let pick = match shape {
                            "same" => i,
                            _ => (r as usize * 8 + i) % pool.len(),
                        };
                        *slot = Some(&pool[pick]);
                    }
                    let offsets = offsets_of(r);
                    let cmd = ComputeCmd::SetBuffers {
                        buffers: &buffers,
                        offsets: &offsets,
                        range: NsRange::new(0, 8),
                    };
                    let before = crate::os::thread_counts::now();
                    e.encode(OnFault::Return, cmd).unwrap();
                    let after = crate::os::thread_counts::now();
                    for k in 0..2 {
                        spent[k] += after[k] - before[k];
                    }
                }
                e.encode(OnFault::Return, ComputeCmd::EndEncoding).unwrap();
                cb.commit(OnFault::Return).unwrap();
                cb.wait_until_completed(OnFault::Return).unwrap();
                for k in 0..2 {
                    figures[k].push(spent[k] / (rounds * 8));
                }
            }
            for (figure, name) in figures.iter_mut().zip(["instructions", "cycles"]) {
                figure.sort_unstable();
                eprintln!(
                    "setBuffers of 8, {shape:<5}: {name:<12} a buffer bound {:6} ({} to {})",
                    figure[2], figure[0], figure[4]
                );
            }
        }
    }

    #[cfg(target_os = "macos")]
    mod metal_tests;
}
