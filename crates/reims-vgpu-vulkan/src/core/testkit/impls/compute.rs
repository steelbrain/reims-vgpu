//! Fake: the `platform/compute.rs` traits over the fake models (architecture 13.2).
//!
//! Every compute command is ordered: each logs its selector and arguments (`fake::calllog`) and
//! consults the script (`fake::script`); see `fake::impls` for the outcome channels.

use super::super::calllog::Arg;
use super::super::objects::{Fk, kind};
use super::super::platform::FakePlatform;
use super::res_ids;
use crate::core::platform::Platform;
use crate::core::platform::compute::*;
use crate::core::raise::{OnFault, RaiseResult};
use crate::core::testkit::script::Policy;

impl ComputeEncoder<FakePlatform> for Fk<kind::ComputeEncoder> {
    fn encode(&self, on: OnFault, cmd: ComputeCmd<'_, FakePlatform>) -> RaiseResult<()> {
        let sel = cmd.selector();
        match cmd {
            ComputeCmd::DispatchThreadgroups {
                threadgroups_per_grid,
                threads_per_threadgroup,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::val(&threadgroups_per_grid),
                    Arg::val(&threads_per_threadgroup),
                ],
            ),
            ComputeCmd::DispatchThreadgroupsIndirect {
                indirect_buffer,
                indirect_buffer_offset,
                threads_per_threadgroup,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::opt(indirect_buffer),
                    Arg::U64(indirect_buffer_offset),
                    Arg::val(&threads_per_threadgroup),
                ],
            ),
            ComputeCmd::DispatchThreads {
                threads_per_grid,
                threads_per_threadgroup,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::val(&threads_per_grid),
                    Arg::val(&threads_per_threadgroup),
                ],
            ),
            ComputeCmd::DispatchThreadsIndirect {
                indirect_buffer,
                indirect_buffer_offset,
            } => self.act(
                Policy::Caught,
                sel,
                vec![Arg::opt(indirect_buffer), Arg::U64(indirect_buffer_offset)],
            ),
            ComputeCmd::EncodeStartDoWhile => self.act(Policy::Caught, sel, vec![]),
            ComputeCmd::EncodeStartElse => self.act(Policy::Caught, sel, vec![]),
            ComputeCmd::EncodeStartIf {
                buffer,
                offset,
                comparison,
                reference_value,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::opt(buffer),
                    Arg::U64(offset),
                    Arg::U64(comparison),
                    Arg::U32(reference_value),
                ],
            ),
            ComputeCmd::EncodeStartWhile {
                buffer,
                offset,
                comparison,
                reference_value,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::opt(buffer),
                    Arg::U64(offset),
                    Arg::U64(comparison),
                    Arg::U32(reference_value),
                ],
            ),
            ComputeCmd::EndEncoding => self.act(Policy::On(on), sel, vec![]),
            ComputeCmd::ExecuteCommandsIndirect {
                indirect_commandbuffer,
                indirect_range_buffer,
                indirect_buffer_offset,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::obj(indirect_commandbuffer),
                    Arg::obj(indirect_range_buffer),
                    Arg::U64(indirect_buffer_offset),
                ],
            ),
            ComputeCmd::ExecuteCommandsInBuffer {
                indirect_command_buffer,
                execution_range,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::obj(indirect_command_buffer),
                    Arg::val(&execution_range),
                ],
            ),
            ComputeCmd::CompressedTextureReinterpretationFlush => {
                self.act(Policy::Caught, sel, vec![])
            }
            ComputeCmd::MemoryBarrierWithResources { resources } => {
                self.act(Policy::Caught, sel, vec![res_ids(resources)])
            }
            ComputeCmd::MemoryBarrierWithScope { scope } => self.act(
                Policy::Caught,
                "memoryBarrierWithScope:",
                vec![Arg::U64(scope)],
            ),
            ComputeCmd::SetBufferOffset { offset, index } => {
                self.act(Policy::Caught, sel, vec![Arg::U64(offset), Arg::U64(index)])
            }
            ComputeCmd::SetBufferOffsetAttributeStride {
                offset,
                stride,
                index,
            } => self.act(
                Policy::Caught,
                sel,
                vec![Arg::U64(offset), Arg::U64(stride), Arg::U64(index)],
            ),
            ComputeCmd::SetBuffersAttributeStrides {
                buffers,
                offsets,
                strides,
                range,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::objs(buffers),
                    Arg::U64s(offsets.to_vec()),
                    Arg::U64s(strides.to_vec()),
                    Arg::val(&range),
                ],
            ),
            ComputeCmd::SetBuffers {
                buffers,
                offsets,
                range,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::objs(buffers),
                    Arg::U64s(offsets.to_vec()),
                    Arg::val(&range),
                ],
            ),
            ComputeCmd::SetComputePipelineState { state } => {
                self.act(Policy::Caught, sel, vec![Arg::opt(state)])
            }
            ComputeCmd::SetImageblockWidth { width, height } => {
                self.act(Policy::Caught, sel, vec![Arg::U64(width), Arg::U64(height)])
            }
            ComputeCmd::SetSamplerStatesLodClamps {
                samplers,
                lod_min_clamps,
                lod_max_clamps,
                range,
            } => self.act(
                Policy::Caught,
                sel,
                vec![
                    Arg::objs(samplers),
                    Arg::F32s(lod_min_clamps.to_vec()),
                    Arg::F32s(lod_max_clamps.to_vec()),
                    Arg::val(&range),
                ],
            ),
            ComputeCmd::SetSamplerStates { samplers, range } => self.act(
                Policy::Caught,
                sel,
                vec![Arg::objs(samplers), Arg::val(&range)],
            ),
            ComputeCmd::SetStageInRegion { region } => {
                self.act(Policy::Caught, sel, vec![Arg::val(&region)])
            }
            ComputeCmd::SetStageInRegionIndirect {
                indirect_buffer,
                indirect_buffer_offset,
            } => self.act(
                Policy::Caught,
                sel,
                vec![Arg::obj(indirect_buffer), Arg::U64(indirect_buffer_offset)],
            ),
            ComputeCmd::SetTexture { texture, index } => self.act(
                Policy::Caught,
                sel,
                vec![Arg::opt(texture), Arg::U64(index)],
            ),
            ComputeCmd::SetTextures { textures, range } => self.act(
                Policy::Caught,
                sel,
                vec![Arg::objs(textures), Arg::val(&range)],
            ),
            ComputeCmd::SetThreadgroupMemoryLength { length, index } => {
                self.act(Policy::Caught, sel, vec![Arg::U64(length), Arg::U64(index)])
            }
            ComputeCmd::UpdateFence { fence } => {
                self.act(Policy::Caught, "updateFence:", vec![Arg::obj(fence)])
            }
            ComputeCmd::UseHeaps { heaps } => self.act(Policy::Caught, sel, vec![Arg::objs(heaps)]),
            ComputeCmd::UseResources { resources, usage } => self.act(
                Policy::Caught,
                sel,
                vec![res_ids(resources), Arg::U64(usage)],
            ),
            ComputeCmd::WaitForFence { fence } => {
                self.act(Policy::Caught, "waitForFence:", vec![Arg::obj(fence)])
            }
        }
    }

    fn encode_end_do_while(
        &self,
        buffer: Option<&<FakePlatform as Platform>::Buffer>,
        offset: u64,
        comparison: u64,
        reference_value: u32,
    ) -> RaiseResult<bool> {
        self.thr(
            Policy::Caught,
            "encodeEndDoWhile:offset:comparison:referenceValue:",
            vec![
                Arg::opt(buffer),
                Arg::U64(offset),
                Arg::U64(comparison),
                Arg::U32(reference_value),
            ],
            || true,
        )
    }

    fn encode_end_if(&self) -> RaiseResult<bool> {
        self.thr(Policy::Caught, "encodeEndIf", vec![], || true)
    }

    fn encode_end_while(&self) -> RaiseResult<bool> {
        self.thr(Policy::Caught, "encodeEndWhile", vec![], || true)
    }
}
