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

use metal2vulkan::metal::{BarrierScope, DispatchType, ResourceUsage};
use metal2vulkan_runtime as rt;
use reims_vgpu_core::platform::Handle as _;
use reims_vgpu_core::platform::blit::ResourceRef;
use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder};
use reims_vgpu_core::platform::types::NsRange;
use reims_vgpu_core::raise::{OnFault, RaiseResult};

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
            } => predication::start_while(self, sel, buffer, offset, comparison, reference_value),
            ComputeCmd::EndEncoding => self.end(sel, rt::ComputeCommandEncoder::end_encoding),
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
                        caught(&mut |e| e.set_buffers_with_strides(buffers, offsets, strides, at))
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
                        icb.resident(sel, progress.as_ref(), usage.contains(ResourceUsage::WRITE));
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
    }

    fn encode_end_do_while(
        &self,
        buffer: Option<&VulkanBuffer>,
        offset: u64,
        comparison: u64,
        reference_value: u32,
    ) -> RaiseResult<bool> {
        let sel = "encodeEndDoWhile:offset:comparison:referenceValue:";
        predication::end_do_while(self, sel, buffer, offset, comparison, reference_value)
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
    use metal2vulkan::bind::SamplerDescriptor;
    use metal2vulkan::metal::SamplerAddressMode;
    use reims_vgpu_core::platform::Platform as _;
    use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
    use reims_vgpu_core::platform::descriptor::TextureDescriptor as _;
    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::platform::resource::{Buffer as _, Library as _};
    use reims_vgpu_core::platform::types::MtlSize;
    use reims_vgpu_core::platform::values::{ComputePassDesc, TextureDesc};
    use reims_vgpu_core::raise::death::DeathKind;
    use reims_vgpu_core::testkit::capture::catch_death;

    use super::*;
    use crate::gpu::command::VulkanCommandBuffer;
    use crate::gpu::device::tests::dev;
    use crate::gpu::resource::tests::{fill, pattern, shared};
    use crate::gpu::resource::{VulkanComputePipeline, VulkanLibrary};
    use crate::platform::tests::platform;

    /// The tests' kernel that resets indirect commands through an argument buffer, which
    /// `build.rs` compiled from `icb_encode.metal`.
    const ICB_ENCODE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/icb_encode.metallib"));

    const ADD: &[u8] =
        include_bytes!("../../../../tests/shaders/compute_add.metallib");
    const TGMEM: &[u8] =
        include_bytes!("../../../../tests/shaders/compute_tgmem.metallib");
    const SAMPLE: &[u8] =
        include_bytes!("../../../../tests/shaders/compute_sample.metallib");

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

    /// A dispatch through a buffer core makes over pages mapped read-only runs where its kernel
    /// reads the buffer, at a `const` pointer, and dies where it is encoded where the kernel may
    /// write it, by its slot writes, or the buffer is made resident for writing: on Linux the
    /// runtime imports a writable twin of the pages, which the write would reach, where Metal's
    /// GPU drops it ([`crate::gpu::writes::kept_write`]). The pages stay zeros
    #[cfg(target_os = "linux")]
    #[test]
    fn a_dispatch_through_read_only_pages_dies_on_linux() {
        use reims_vgpu_core::platform::os::vm::HostVm as _;

        use crate::gpu::resource::tests::over_read_only_pages;
        use crate::os::{OsVm, PAGE};

        let add = pipeline(ADD, "add_one");
        let (ro, pages) = over_read_only_pages(1);
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
        assert_eq!(words(&out), [1; 4], "read through the twin");
        let cb = command_buffer();
        let e = cb.compute_encoder(OnFault::Return).unwrap().unwrap();
        let dies = |cmd: ComputeCmd<'_, VulkanPlatform>| {
            let sel = cmd.selector();
            let site = catch_death(|| e.encode(OnFault::Return, cmd)).unwrap_err();
            assert_eq!((site.kind, site.file), (DeathKind::Abort, sel));
        };
        for cmd in [
            ComputeCmd::SetComputePipelineState { state: Some(&add) },
            ComputeCmd::SetBuffers {
                buffers: &[Some(&ro), Some(&out)],
                offsets: &[0, 0],
                range: NsRange::new(0, 2),
            },
        ] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        dies(four());
        // Read at slot 1 again, but made resident for writing.
        for cmd in [
            ComputeCmd::SetBuffers {
                buffers: &[Some(&out), Some(&ro)],
                offsets: &[0, 0],
                range: NsRange::new(0, 2),
            },
            ComputeCmd::UseResources {
                resources: &[Some(ResourceRef::Buffer(&ro))],
                usage: 3,
            },
        ] {
            e.encode(OnFault::Return, cmd).unwrap();
        }
        dies(four());
        run(&cb, &e, Vec::new());
        // SAFETY: the page, mapped readable, which no GPU work uses now.
        let first = unsafe { pages.as_ptr::<u32>().read_volatile() };
        assert_eq!(first, 0);
        drop((cb, e, ro));
        assert!(OsVm::new().deallocate(pages, PAGE).is_success());
    }

    /// A dispatch whose kernel stores into a guest's read-only buffer (made with bit 24,
    /// `resource::tests::read_only_buffer`), bound at the slot it writes and made resident for
    /// writing, completes with no error, its stores dropped, the pages staying zeros, as Metal
    /// drops a kernel's stores there (`crate::gpu::writes`'s docs): through the runtime's copy of
    /// the pages on Linux, and Metal's read-only view on macOS. The kernel reads it as the pages
    #[test]
    fn a_kernel_store_into_a_read_only_buffer_is_dropped() {
        use reims_vgpu_core::platform::os::vm::HostVm as _;

        use crate::gpu::resource::tests::{host_bytes_at, read_only_buffer};
        use crate::os::{OsVm, PAGE};

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
        assert!(OsVm::new().deallocate(pages, PAGE).is_success());
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
        use reims_vgpu_core::platform::os::Clock as _;
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
        use reims_vgpu_core::platform::icb::IndirectCommandBuffer as _;
        use reims_vgpu_core::platform::values::IcbDesc;

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

    #[cfg(target_os = "macos")]
    mod metal_tests;
}
