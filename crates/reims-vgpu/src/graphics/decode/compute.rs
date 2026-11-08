//! The compute-encoder op decoder: one arm per
//! [`reims_vgpu_graphics::ops::compute`] op, turning a wire op into a `ComputeCmd` on
//! the open `P::ComputeEncoder` (docs/graphics-kext/design.md, "Stage 3 wire
//! protocol"). The command-buffer walk ([`super::cmdbuf`]) routes every
//! compute-group op (`tag >> 12 == `[`RGFX_OP_GROUP_COMPUTE`]) here.
//!
//! Each arm reads its op's body with zerocopy, resolves its object references
//! through the [object-resolution API](super) (`require_*`/`get_*`), and sends the
//! matching [`ComputeCmd`] with [`ComputeEncoder::encode`]. A body too short for its
//! op faults the exec ([`super::op_fault`]); a bad reference faults through the
//! resolution API. The array-tailed ops ([`reims_vgpu_graphics::ops::compute`], "Array-tailed
//! ops") read a fixed head, then its `length`/`count` element records right after it.
//!
//! `endEncoding` has no arm here: the walk's `ENCODER_END` framing op ends the open
//! compute encoder ([`super::cmdbuf`]).
//!
//! [`RGFX_OP_GROUP_COMPUTE`]: reims_vgpu_graphics::ops::RGFX_OP_GROUP_COMPUTE

use reims_vgpu_core::platform::Platform;
use reims_vgpu_core::platform::blit::ResourceRef;
use reims_vgpu_core::platform::compute::{ComputeCmd, ComputeEncoder};
use reims_vgpu_core::platform::types::{MtlOrigin, MtlRegion, MtlSize, NsRange};
use reims_vgpu_core::raise::{OnFault, Raise, RaiseResult};
use reims_vgpu_graphics::ops::compute::{
    RGFX_OP_COMPUTE_COMPRESSED_TEXTURE_REINTERPRETATION_FLUSH,
    RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS, RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS_INDIRECT,
    RGFX_OP_COMPUTE_DISPATCH_THREADS, RGFX_OP_COMPUTE_DISPATCH_THREADS_INDIRECT,
    RGFX_OP_COMPUTE_ENCODE_START_DO_WHILE, RGFX_OP_COMPUTE_ENCODE_START_ELSE,
    RGFX_OP_COMPUTE_ENCODE_START_IF, RGFX_OP_COMPUTE_ENCODE_START_WHILE,
    RGFX_OP_COMPUTE_EXECUTE_COMMANDS_IN_BUFFER, RGFX_OP_COMPUTE_EXECUTE_COMMANDS_INDIRECT,
    RGFX_OP_COMPUTE_MEMORY_BARRIER_RESOURCES, RGFX_OP_COMPUTE_MEMORY_BARRIER_SCOPE,
    RGFX_OP_COMPUTE_SET_BUFFER_OFFSET, RGFX_OP_COMPUTE_SET_BUFFER_OFFSET_ATTRIBUTE_STRIDE,
    RGFX_OP_COMPUTE_SET_BUFFERS, RGFX_OP_COMPUTE_SET_BUFFERS_ATTRIBUTE_STRIDES,
    RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE, RGFX_OP_COMPUTE_SET_IMAGEBLOCK_WIDTH,
    RGFX_OP_COMPUTE_SET_SAMPLER_STATES, RGFX_OP_COMPUTE_SET_SAMPLER_STATES_LOD_CLAMPS,
    RGFX_OP_COMPUTE_SET_STAGE_IN_REGION, RGFX_OP_COMPUTE_SET_STAGE_IN_REGION_INDIRECT,
    RGFX_OP_COMPUTE_SET_TEXTURE, RGFX_OP_COMPUTE_SET_TEXTURES,
    RGFX_OP_COMPUTE_SET_THREADGROUP_MEMORY_LENGTH, RGFX_OP_COMPUTE_UPDATE_FENCE,
    RGFX_OP_COMPUTE_USE_HEAPS, RGFX_OP_COMPUTE_USE_RESOURCES, RGFX_OP_COMPUTE_WAIT_FOR_FENCE,
    RGFX_RESOURCE_KIND_BUFFER, RGFX_RESOURCE_KIND_ICB, RGFX_RESOURCE_KIND_NONE,
    RGFX_RESOURCE_KIND_TEXTURE, RgfxComputeBufferBinding, RgfxComputeBufferStrideBinding,
    RgfxComputeHeapBinding, RgfxComputeOpDispatchThreadgroups,
    RgfxComputeOpDispatchThreadgroupsIndirect, RgfxComputeOpDispatchThreads,
    RgfxComputeOpDispatchThreadsIndirect, RgfxComputeOpEncodeStartIf,
    RgfxComputeOpEncodeStartWhile, RgfxComputeOpExecuteCommandsInBuffer,
    RgfxComputeOpExecuteCommandsIndirect, RgfxComputeOpMemoryBarrierResources,
    RgfxComputeOpMemoryBarrierScope, RgfxComputeOpSetBufferOffset,
    RgfxComputeOpSetBufferOffsetAttributeStride, RgfxComputeOpSetBuffers,
    RgfxComputeOpSetBuffersAttributeStrides, RgfxComputeOpSetComputePipelineState,
    RgfxComputeOpSetImageblockWidth, RgfxComputeOpSetSamplerStates,
    RgfxComputeOpSetSamplerStatesLodClamps, RgfxComputeOpSetStageInRegion,
    RgfxComputeOpSetStageInRegionIndirect, RgfxComputeOpSetTexture, RgfxComputeOpSetTextures,
    RgfxComputeOpSetThreadgroupMemoryLength, RgfxComputeOpUpdateFence, RgfxComputeOpUseHeaps,
    RgfxComputeOpUseResources, RgfxComputeOpWaitForFence, RgfxComputeResourceRef,
    RgfxComputeSamplerBinding, RgfxComputeSamplerLodBinding, RgfxComputeTextureBinding,
};
use zerocopy::{FromBytes, Immutable};

use super::refused::{self, Bound, Marks, Slot, Stage};
use super::{GraphicsDecoder, op_fault};

/// Reads op body `T` from the start of `payload`, or an exec fault when it is too
/// short.
#[inline]
fn body<T: FromBytes>(tag: u16, payload: &[u8]) -> RaiseResult<T> {
    match T::read_from_prefix(payload) {
        Ok((t, _)) => Ok(t),
        Err(_) => Err(short_body(tag)),
    }
}

/// The exec fault of compute op `tag`, whose body is too short: out of line, as no well-formed
/// op meets it.
#[cold]
#[inline(never)]
fn short_body(tag: u16) -> Raise {
    op_fault(format_args!("compute op {tag:#06x}: body too short"))
}

/// Reads the `count` element records of `E` that follow a head of `head_len` bytes
/// in `payload`, or an exec fault when the body is too short for them.
fn tail<E: FromBytes + Immutable>(
    tag: u16,
    payload: &[u8],
    head_len: usize,
    count: usize,
) -> RaiseResult<Vec<E>> {
    let stride = size_of::<E>();
    let short = || {
        op_fault(format_args!(
            "compute op {tag:#06x}: body too short for its records"
        ))
    };
    let span = count.checked_mul(stride).ok_or_else(short)?;
    let end = head_len.checked_add(span).ok_or_else(short)?;
    let records = payload.get(head_len..end).ok_or_else(short)?;
    let mut out = Vec::with_capacity(count);
    for chunk in records.chunks_exact(stride) {
        out.push(E::read_from_bytes(chunk).map_err(|_| short())?);
    }
    Ok(out)
}

/// Resolves one wire resource record: a nil kind is `None`, each other kind
/// resolves `obj_id` through its resolver (a miss or wrong kind faults), and an
/// unknown kind faults the exec. An object whose create the host refused is `None`, which no
/// dispatch reads ([`refused`]).
fn resolve_resource<'a, P: Platform>(
    objs: &'a GraphicsDecoder<P>,
    rec: &RgfxComputeResourceRef,
) -> RaiseResult<Option<ResourceRef<'a, P>>> {
    match rec.kind {
        RGFX_RESOURCE_KIND_NONE => Ok(None),
        RGFX_RESOURCE_KIND_BUFFER => {
            refused::some_or_nil(objs.require_buffer(rec.obj_id).map(ResourceRef::Buffer))
        }
        RGFX_RESOURCE_KIND_TEXTURE => {
            refused::some_or_nil(objs.require_texture(rec.obj_id).map(ResourceRef::Texture))
        }
        RGFX_RESOURCE_KIND_ICB => refused::some_or_nil(
            objs.require_indirect_command_buffer(rec.obj_id)
                .map(ResourceRef::Icb),
        ),
        other => Err(op_fault(format_args!(
            "compute op: unknown resource kind {other}"
        ))),
    }
}

/// Resolves a wire resource array into the objects it names, for the encode.
fn resolve_resources<'a, P: Platform>(
    objs: &'a GraphicsDecoder<P>,
    recs: &[RgfxComputeResourceRef],
) -> RaiseResult<Vec<Option<ResourceRef<'a, P>>>> {
    recs.iter().map(|r| resolve_resource(objs, r)).collect()
}

/// Whether compute op `tag` dispatches (or executes an indirect command buffer): one dropped
/// while a slot of its encoder's state holds an object whose create the host refused
/// ([`refused`]).
pub(crate) fn dispatches(tag: u16) -> bool {
    matches!(
        tag,
        RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS
            | RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS_INDIRECT
            | RGFX_OP_COMPUTE_DISPATCH_THREADS
            | RGFX_OP_COMPUTE_DISPATCH_THREADS_INDIRECT
            | RGFX_OP_COMPUTE_EXECUTE_COMMANDS_INDIRECT
            | RGFX_OP_COMPUTE_EXECUTE_COMMANDS_IN_BUFFER
    )
}

/// The bindings at `first` on of the `recs` a resolve of each, `resolve`, names: an object whose
/// create the host refused is nil, and marks its slot of `bound` in `marks` ([`refused`]).
fn bind_each<'a, R, T>(
    recs: &[R],
    (marks, bound, first): (&Marks, Bound, u64),
    resolve: impl Fn(&R) -> RaiseResult<Option<&'a T>>,
) -> RaiseResult<Vec<Option<&'a T>>> {
    let mut named = Vec::with_capacity(recs.len());
    for (i, r) in (0u64..).zip(recs) {
        named.push(refused::bind!(
            marks,
            Slot::new(Stage::Kernel, bound, first.wrapping_add(i)),
            resolve(r)
        ));
    }
    Ok(named)
}

/// Decodes one compute-encoder op: `tag` its op, `payload` its body (the bytes
/// after the [`RgfxOpHeader`](reims_vgpu_graphics::ops::RgfxOpHeader)). `objs` resolves
/// object references, `enc` is the open compute encoder, `marks` its slots that hold an object
/// whose create the host refused, and `on` says how an encode's raise is taken. `Ok(true)` when
/// the op was handled, `Ok(false)` for an unknown tag (the walk then faults); a platform raise
/// propagates as `Err`.
pub(crate) fn decode_op<P: Platform>(
    tag: u16,
    payload: &[u8],
    objs: &GraphicsDecoder<P>,
    enc: &P::ComputeEncoder,
    marks: &Marks,
    on: OnFault,
) -> RaiseResult<bool> {
    match tag {
        RGFX_OP_COMPUTE_SET_COMPUTE_PIPELINE_STATE => {
            let m: RgfxComputeOpSetComputePipelineState = body(tag, payload)?;
            let slot = Slot::state(Bound::Pipeline);
            let state = refused::bind!(marks, slot, objs.get_compute_pipeline(m.pipeline_obj_id));
            // A refused pipeline is not set: the encoder's dispatches are dropped until the next
            // ([`refused`]).
            if marks.holds(slot) {
                return Ok(true);
            }
            enc.encode(on, ComputeCmd::SetComputePipelineState { state })?;
        }
        RGFX_OP_COMPUTE_SET_BUFFER_OFFSET => {
            let m: RgfxComputeOpSetBufferOffset = body(tag, payload)?;
            // The slot holds nil for a refused buffer ([`refused`]): nothing to offset.
            if marks.holds(Slot::new(Stage::Kernel, Bound::Buffer, m.index)) {
                return Ok(true);
            }
            enc.encode(
                on,
                ComputeCmd::SetBufferOffset {
                    offset: m.offset,
                    index: m.index,
                },
            )?;
        }
        RGFX_OP_COMPUTE_SET_BUFFER_OFFSET_ATTRIBUTE_STRIDE => {
            let m: RgfxComputeOpSetBufferOffsetAttributeStride = body(tag, payload)?;
            // The slot holds nil for a refused buffer ([`refused`]): nothing to offset.
            if marks.holds(Slot::new(Stage::Kernel, Bound::Buffer, m.index)) {
                return Ok(true);
            }
            enc.encode(
                on,
                ComputeCmd::SetBufferOffsetAttributeStride {
                    offset: m.offset,
                    stride: m.stride,
                    index: m.index,
                },
            )?;
        }
        RGFX_OP_COMPUTE_SET_BUFFERS => {
            let m: RgfxComputeOpSetBuffers = body(tag, payload)?;
            let recs: Vec<RgfxComputeBufferBinding> = tail(
                tag,
                payload,
                size_of::<RgfxComputeOpSetBuffers>(),
                m.length as usize,
            )?;
            let named: Vec<Option<&P::Buffer>> =
                bind_each(&recs, (marks, Bound::Buffer, m.location), |r| {
                    objs.get_buffer(r.buffer_obj_id)
                })?;
            let buffers: Vec<Option<&P::Buffer>> = named;
            let offsets: Vec<u64> = recs.iter().map(|r| r.offset).collect();
            enc.encode(
                on,
                ComputeCmd::SetBuffers {
                    buffers: &buffers,
                    offsets: &offsets,
                    range: NsRange::new(m.location, m.length),
                },
            )?;
        }
        RGFX_OP_COMPUTE_SET_BUFFERS_ATTRIBUTE_STRIDES => {
            let m: RgfxComputeOpSetBuffersAttributeStrides = body(tag, payload)?;
            let recs: Vec<RgfxComputeBufferStrideBinding> = tail(
                tag,
                payload,
                size_of::<RgfxComputeOpSetBuffersAttributeStrides>(),
                m.length as usize,
            )?;
            let named: Vec<Option<&P::Buffer>> =
                bind_each(&recs, (marks, Bound::Buffer, m.location), |r| {
                    objs.get_buffer(r.buffer_obj_id)
                })?;
            let buffers: Vec<Option<&P::Buffer>> = named;
            let offsets: Vec<u64> = recs.iter().map(|r| r.offset).collect();
            let strides: Vec<u64> = recs.iter().map(|r| r.stride).collect();
            enc.encode(
                on,
                ComputeCmd::SetBuffersAttributeStrides {
                    buffers: &buffers,
                    offsets: &offsets,
                    strides: &strides,
                    range: NsRange::new(m.location, m.length),
                },
            )?;
        }
        RGFX_OP_COMPUTE_SET_TEXTURE => {
            let m: RgfxComputeOpSetTexture = body(tag, payload)?;
            let texture = refused::bind!(
                marks,
                Slot::new(Stage::Kernel, Bound::Texture, m.index),
                objs.get_texture(m.texture_obj_id)
            );
            enc.encode(
                on,
                ComputeCmd::SetTexture {
                    texture,
                    index: m.index,
                },
            )?;
        }
        RGFX_OP_COMPUTE_SET_TEXTURES => {
            let m: RgfxComputeOpSetTextures = body(tag, payload)?;
            let recs: Vec<RgfxComputeTextureBinding> = tail(
                tag,
                payload,
                size_of::<RgfxComputeOpSetTextures>(),
                m.length as usize,
            )?;
            let named: Vec<Option<&P::Texture>> =
                bind_each(&recs, (marks, Bound::Texture, m.location), |r| {
                    objs.get_texture(r.texture_obj_id)
                })?;
            let textures: Vec<Option<&P::Texture>> = named;
            enc.encode(
                on,
                ComputeCmd::SetTextures {
                    textures: &textures,
                    range: NsRange::new(m.location, m.length),
                },
            )?;
        }
        RGFX_OP_COMPUTE_SET_SAMPLER_STATES => {
            let m: RgfxComputeOpSetSamplerStates = body(tag, payload)?;
            let recs: Vec<RgfxComputeSamplerBinding> = tail(
                tag,
                payload,
                size_of::<RgfxComputeOpSetSamplerStates>(),
                m.length as usize,
            )?;
            let named: Vec<Option<&P::SamplerState>> =
                bind_each(&recs, (marks, Bound::Sampler, m.location), |r| {
                    objs.get_sampler(r.sampler_obj_id)
                })?;
            let samplers: Vec<Option<&P::SamplerState>> = named;
            enc.encode(
                on,
                ComputeCmd::SetSamplerStates {
                    samplers: &samplers,
                    range: NsRange::new(m.location, m.length),
                },
            )?;
        }
        RGFX_OP_COMPUTE_SET_SAMPLER_STATES_LOD_CLAMPS => {
            let m: RgfxComputeOpSetSamplerStatesLodClamps = body(tag, payload)?;
            let recs: Vec<RgfxComputeSamplerLodBinding> = tail(
                tag,
                payload,
                size_of::<RgfxComputeOpSetSamplerStatesLodClamps>(),
                m.length as usize,
            )?;
            let named: Vec<Option<&P::SamplerState>> =
                bind_each(&recs, (marks, Bound::Sampler, m.location), |r| {
                    objs.get_sampler(r.sampler_obj_id)
                })?;
            let samplers: Vec<Option<&P::SamplerState>> = named;
            let lod_min: Vec<f32> = recs
                .iter()
                .map(|r| f32::from_bits(r.lod_min_clamp_bits))
                .collect();
            let lod_max: Vec<f32> = recs
                .iter()
                .map(|r| f32::from_bits(r.lod_max_clamp_bits))
                .collect();
            enc.encode(
                on,
                ComputeCmd::SetSamplerStatesLodClamps {
                    samplers: &samplers,
                    lod_min_clamps: &lod_min,
                    lod_max_clamps: &lod_max,
                    range: NsRange::new(m.location, m.length),
                },
            )?;
        }
        RGFX_OP_COMPUTE_SET_THREADGROUP_MEMORY_LENGTH => {
            let m: RgfxComputeOpSetThreadgroupMemoryLength = body(tag, payload)?;
            enc.encode(
                on,
                ComputeCmd::SetThreadgroupMemoryLength {
                    length: m.length,
                    index: m.index,
                },
            )?;
        }
        RGFX_OP_COMPUTE_SET_STAGE_IN_REGION => {
            let m: RgfxComputeOpSetStageInRegion = body(tag, payload)?;
            enc.encode(
                on,
                ComputeCmd::SetStageInRegion {
                    region: MtlRegion::new(
                        MtlOrigin::new(m.origin_x, m.origin_y, m.origin_z),
                        MtlSize::new(m.width, m.height, m.depth),
                    ),
                },
            )?;
        }
        RGFX_OP_COMPUTE_SET_STAGE_IN_REGION_INDIRECT => {
            let m: RgfxComputeOpSetStageInRegionIndirect = body(tag, payload)?;
            // The indirect buffer must name an object (nil faults the stream).
            let indirect_buffer = objs.require_buffer(m.indirect_buffer_obj_id)?;
            enc.encode(
                on,
                ComputeCmd::SetStageInRegionIndirect {
                    indirect_buffer,
                    indirect_buffer_offset: m.indirect_buffer_offset,
                },
            )?;
        }
        RGFX_OP_COMPUTE_SET_IMAGEBLOCK_WIDTH => {
            let m: RgfxComputeOpSetImageblockWidth = body(tag, payload)?;
            enc.encode(
                on,
                ComputeCmd::SetImageblockWidth {
                    width: m.width,
                    height: m.height,
                },
            )?;
        }
        RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS => {
            let m: RgfxComputeOpDispatchThreadgroups = body(tag, payload)?;
            enc.encode(
                on,
                ComputeCmd::DispatchThreadgroups {
                    threadgroups_per_grid: MtlSize::new(
                        m.threadgroups_width,
                        m.threadgroups_height,
                        m.threadgroups_depth,
                    ),
                    threads_per_threadgroup: MtlSize::new(
                        m.threads_width,
                        m.threads_height,
                        m.threads_depth,
                    ),
                },
            )?;
        }
        RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS_INDIRECT => {
            let m: RgfxComputeOpDispatchThreadgroupsIndirect = body(tag, payload)?;
            let indirect_buffer = objs.get_buffer(m.indirect_buffer_obj_id)?;
            enc.encode(
                on,
                ComputeCmd::DispatchThreadgroupsIndirect {
                    indirect_buffer,
                    indirect_buffer_offset: m.indirect_buffer_offset,
                    threads_per_threadgroup: MtlSize::new(
                        m.threads_width,
                        m.threads_height,
                        m.threads_depth,
                    ),
                },
            )?;
        }
        RGFX_OP_COMPUTE_DISPATCH_THREADS => {
            let m: RgfxComputeOpDispatchThreads = body(tag, payload)?;
            enc.encode(
                on,
                ComputeCmd::DispatchThreads {
                    threads_per_grid: MtlSize::new(m.grid_width, m.grid_height, m.grid_depth),
                    threads_per_threadgroup: MtlSize::new(
                        m.threads_width,
                        m.threads_height,
                        m.threads_depth,
                    ),
                },
            )?;
        }
        RGFX_OP_COMPUTE_DISPATCH_THREADS_INDIRECT => {
            let m: RgfxComputeOpDispatchThreadsIndirect = body(tag, payload)?;
            let indirect_buffer = objs.get_buffer(m.indirect_buffer_obj_id)?;
            enc.encode(
                on,
                ComputeCmd::DispatchThreadsIndirect {
                    indirect_buffer,
                    indirect_buffer_offset: m.indirect_buffer_offset,
                },
            )?;
        }
        RGFX_OP_COMPUTE_USE_RESOURCES => {
            let m: RgfxComputeOpUseResources = body(tag, payload)?;
            let recs: Vec<RgfxComputeResourceRef> = tail(
                tag,
                payload,
                size_of::<RgfxComputeOpUseResources>(),
                m.count as usize,
            )?;
            let resources = resolve_resources(objs, &recs)?;
            enc.encode(
                on,
                ComputeCmd::UseResources {
                    resources: &resources,
                    usage: m.usage,
                },
            )?;
        }
        RGFX_OP_COMPUTE_USE_HEAPS => {
            let m: RgfxComputeOpUseHeaps = body(tag, payload)?;
            let recs: Vec<RgfxComputeHeapBinding> = tail(
                tag,
                payload,
                size_of::<RgfxComputeOpUseHeaps>(),
                m.count as usize,
            )?;
            let named: Vec<Option<&P::Heap>> = recs
                .iter()
                .map(|r| refused::or_nil(objs.get_heap(r.heap_obj_id)))
                .collect::<RaiseResult<_>>()?;
            let heaps: Vec<Option<&P::Heap>> = named;
            enc.encode(on, ComputeCmd::UseHeaps { heaps: &heaps })?;
        }
        RGFX_OP_COMPUTE_MEMORY_BARRIER_RESOURCES => {
            let m: RgfxComputeOpMemoryBarrierResources = body(tag, payload)?;
            let recs: Vec<RgfxComputeResourceRef> = tail(
                tag,
                payload,
                size_of::<RgfxComputeOpMemoryBarrierResources>(),
                m.count as usize,
            )?;
            let resources = resolve_resources(objs, &recs)?;
            enc.encode(
                on,
                ComputeCmd::MemoryBarrierWithResources {
                    resources: &resources,
                },
            )?;
        }
        RGFX_OP_COMPUTE_MEMORY_BARRIER_SCOPE => {
            let m: RgfxComputeOpMemoryBarrierScope = body(tag, payload)?;
            enc.encode(on, ComputeCmd::MemoryBarrierWithScope { scope: m.scope })?;
        }
        RGFX_OP_COMPUTE_UPDATE_FENCE => {
            let m: RgfxComputeOpUpdateFence = body(tag, payload)?;
            // The fence must name an object (nil faults the stream).
            let fence = objs.require_fence(m.fence_obj_id)?;
            enc.encode(on, ComputeCmd::UpdateFence { fence })?;
        }
        RGFX_OP_COMPUTE_WAIT_FOR_FENCE => {
            let m: RgfxComputeOpWaitForFence = body(tag, payload)?;
            // The fence must name an object (nil faults the stream).
            let fence = objs.require_fence(m.fence_obj_id)?;
            enc.encode(on, ComputeCmd::WaitForFence { fence })?;
        }
        RGFX_OP_COMPUTE_EXECUTE_COMMANDS_INDIRECT => {
            let m: RgfxComputeOpExecuteCommandsIndirect = body(tag, payload)?;
            // Both the ICB and the range buffer must name an object.
            let icb = objs.require_indirect_command_buffer(m.indirect_commandbuffer_obj_id)?;
            let range_buffer = objs.require_buffer(m.indirect_range_buffer_obj_id)?;
            enc.encode(
                on,
                ComputeCmd::ExecuteCommandsIndirect {
                    indirect_commandbuffer: icb,
                    indirect_range_buffer: range_buffer,
                    indirect_buffer_offset: m.indirect_buffer_offset,
                },
            )?;
        }
        RGFX_OP_COMPUTE_EXECUTE_COMMANDS_IN_BUFFER => {
            let m: RgfxComputeOpExecuteCommandsInBuffer = body(tag, payload)?;
            // The ICB must name an object.
            let icb = objs.require_indirect_command_buffer(m.indirect_command_buffer_obj_id)?;
            enc.encode(
                on,
                ComputeCmd::ExecuteCommandsInBuffer {
                    indirect_command_buffer: icb,
                    execution_range: NsRange::new(m.location, m.length),
                },
            )?;
        }
        RGFX_OP_COMPUTE_ENCODE_START_IF => {
            let m: RgfxComputeOpEncodeStartIf = body(tag, payload)?;
            // A refused buffer's start would leave its else and its end unpaired: the encoder's
            // later ops are dropped ([`refused`]).
            let buffer = match objs.get_buffer(m.buffer_obj_id) {
                Err(raise) if refused::is_refused(&raise) => {
                    marks.skip_rest();
                    return Ok(true);
                }
                buffer => buffer?,
            };
            enc.encode(
                on,
                ComputeCmd::EncodeStartIf {
                    buffer,
                    offset: m.offset,
                    comparison: m.comparison,
                    reference_value: m.reference_value,
                },
            )?;
        }
        RGFX_OP_COMPUTE_ENCODE_START_WHILE => {
            let m: RgfxComputeOpEncodeStartWhile = body(tag, payload)?;
            // A refused buffer's start would leave its else and its end unpaired: the encoder's
            // later ops are dropped ([`refused`]).
            let buffer = match objs.get_buffer(m.buffer_obj_id) {
                Err(raise) if refused::is_refused(&raise) => {
                    marks.skip_rest();
                    return Ok(true);
                }
                buffer => buffer?,
            };
            enc.encode(
                on,
                ComputeCmd::EncodeStartWhile {
                    buffer,
                    offset: m.offset,
                    comparison: m.comparison,
                    reference_value: m.reference_value,
                },
            )?;
        }
        RGFX_OP_COMPUTE_ENCODE_START_ELSE => {
            enc.encode(on, ComputeCmd::EncodeStartElse)?;
        }
        RGFX_OP_COMPUTE_ENCODE_START_DO_WHILE => {
            enc.encode(on, ComputeCmd::EncodeStartDoWhile)?;
        }
        RGFX_OP_COMPUTE_COMPRESSED_TEXTURE_REINTERPRETATION_FLUSH => {
            enc.encode(on, ComputeCmd::CompressedTextureReinterpretationFlush)?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use reims_vgpu_core::platform::Handle;
    use reims_vgpu_core::platform::command::{CommandBuffer, CommandQueue};
    use reims_vgpu_core::platform::device::GpuDevice;
    use reims_vgpu_core::platform::values::ComputePassDesc;
    use reims_vgpu_core::raise::OnFault;
    use reims_vgpu_core::testkit::FakePlatform;
    use reims_vgpu_core::testkit::calllog::Arg;
    use reims_vgpu_graphics::wire::{
        RGFX_CMD_BUFFER_CREATE, RGFX_CMD_FENCE_CREATE, RGFX_CMD_QUEUE_CREATE,
        RGFX_CMD_TEXTURE_CREATE, RgfxBufferCreate, RgfxMsgHeader, RgfxObjectCreate,
        RgfxQueueCreate, RgfxTextureCreate, RgfxTextureDescriptor,
    };
    use zerocopy::IntoBytes;

    use super::*;

    /// A decoder with a queue (1), two buffers (2, 3), a texture (4) and a fence (5)
    /// created, plus an open compute encoder on a fresh command buffer.
    fn setup() -> (
        Arc<FakePlatform>,
        GraphicsDecoder<FakePlatform>,
        <FakePlatform as Platform>::ComputeEncoder,
    ) {
        let p = Arc::new(FakePlatform::new());
        let device = p.device();
        let mut dec = GraphicsDecoder::<FakePlatform>::new(
            Arc::clone(&p),
            Arc::new(reims_vgpu_graphics::region::Region::new(0x10000).unwrap()),
        );

        let queue = RgfxQueueCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_QUEUE_CREATE,
                size: size_of::<RgfxQueueCreate>() as u32,
                seqno: 0,
            },
            obj_id: 1,
            ..RgfxQueueCreate::default()
        };
        assert_eq!(dec.decode(&device, queue.as_bytes()).unwrap(), Ok(()));

        for id in [2, 3] {
            let b = RgfxBufferCreate {
                header: RgfxMsgHeader {
                    r#type: RGFX_CMD_BUFFER_CREATE,
                    size: size_of::<RgfxBufferCreate>() as u32,
                    seqno: 0,
                },
                obj_id: id,
                length: 65536,
                ..RgfxBufferCreate::default()
            };
            assert_eq!(dec.decode(&device, b.as_bytes()).unwrap(), Ok(()));
        }

        dec.hold_resource_ids(0, 0);
        let t = RgfxTextureCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_TEXTURE_CREATE,
                size: size_of::<RgfxTextureCreate>() as u32,
                seqno: 0,
            },
            obj_id: 4,
            desc: RgfxTextureDescriptor {
                pixel_format: 80,
                texture_type: 2,
                width: 64,
                height: 64,
                depth: 1,
                mipmap_level_count: 1,
                array_length: 1,
                sample_count: 1,
                usage: 5,
                ..RgfxTextureDescriptor::default()
            },
            resource_id: 4,
            ..RgfxTextureCreate::default()
        };
        assert_eq!(dec.decode(&device, t.as_bytes()).unwrap(), Ok(()));

        let fence = RgfxObjectCreate {
            header: RgfxMsgHeader {
                r#type: RGFX_CMD_FENCE_CREATE,
                size: size_of::<RgfxObjectCreate>() as u32,
                seqno: 0,
            },
            obj_id: 5,
        };
        assert_eq!(dec.decode(&device, fence.as_bytes()).unwrap(), Ok(()));

        // An open compute encoder on a fresh command buffer from the queue.
        let queue_handle = p.device().new_command_queue().unwrap();
        let cb = queue_handle
            .command_buffer(OnFault::Return)
            .unwrap()
            .unwrap();
        let enc = cb
            .compute_encoder_for_pass(&ComputePassDesc::default())
            .unwrap()
            .unwrap();
        (p, dec, enc)
    }

    /// One op body of bytes to the decoder, asserting it was handled.
    fn drive(
        dec: &GraphicsDecoder<FakePlatform>,
        enc: &<FakePlatform as Platform>::ComputeEncoder,
        tag: u16,
        body: &[u8],
    ) {
        assert!(decode_op(tag, body, dec, enc, &Marks::default(), OnFault::Return).unwrap());
    }

    /// Each scalar compute op drives its selector with its arguments.
    #[test]
    fn scalar_ops_drive_the_encoder() {
        let (p, dec, enc) = setup();
        let mark = p.call_log().mark();

        drive(
            &dec,
            &enc,
            RGFX_OP_COMPUTE_SET_BUFFER_OFFSET,
            RgfxComputeOpSetBufferOffset {
                offset: 128,
                index: 2,
            }
            .as_bytes(),
        );
        drive(
            &dec,
            &enc,
            RGFX_OP_COMPUTE_DISPATCH_THREADGROUPS,
            RgfxComputeOpDispatchThreadgroups {
                threadgroups_width: 16,
                threadgroups_height: 8,
                threadgroups_depth: 1,
                threads_width: 32,
                threads_height: 1,
                threads_depth: 1,
            }
            .as_bytes(),
        );
        drive(
            &dec,
            &enc,
            RGFX_OP_COMPUTE_SET_STAGE_IN_REGION,
            RgfxComputeOpSetStageInRegion {
                origin_x: 0,
                origin_y: 0,
                origin_z: 0,
                width: 64,
                height: 64,
                depth: 1,
            }
            .as_bytes(),
        );
        drive(
            &dec,
            &enc,
            RGFX_OP_COMPUTE_MEMORY_BARRIER_SCOPE,
            RgfxComputeOpMemoryBarrierScope { scope: 3 }.as_bytes(),
        );
        drive(&dec, &enc, RGFX_OP_COMPUTE_ENCODE_START_ELSE, &[]);

        let sels: Vec<&str> = p.call_log().sels_since(mark);
        assert_eq!(
            sels,
            vec![
                "setBufferOffset:atIndex:",
                "dispatchThreadgroups:threadsPerThreadgroup:",
                "setStageInRegion:",
                "memoryBarrierWithScope:",
                "encodeStartElse",
            ]
        );

        // The dispatch sent the two sizes as its arguments.
        let dispatch = p
            .call_log()
            .calls_to("dispatchThreadgroups:threadsPerThreadgroup:");
        assert_eq!(dispatch.len(), 1);
        assert_eq!(
            dispatch[0].args,
            vec![
                Arg::val(&MtlSize::new(16, 8, 1)),
                Arg::val(&MtlSize::new(32, 1, 1)),
            ]
        );
    }

    /// A compute op resolves its object references and sends the handle.
    #[test]
    fn set_texture_resolves_its_reference() {
        let (p, dec, enc) = setup();
        let mark = p.call_log().mark();
        drive(
            &dec,
            &enc,
            RGFX_OP_COMPUTE_SET_TEXTURE,
            RgfxComputeOpSetTexture {
                texture_obj_id: 4,
                index: 1,
            }
            .as_bytes(),
        );
        let calls = p.call_log().calls_to("setTexture:atIndex:");
        assert_eq!(calls.len(), 1);
        let want = Arg::Obj(Some(dec.require_texture(4).unwrap().id()));
        assert_eq!(calls[0].args, vec![want, Arg::U64(1)]);
        let _ = mark;
    }

    /// An update/wait fence op resolves its required reference.
    #[test]
    fn fence_ops_resolve_their_reference() {
        let (p, dec, enc) = setup();
        drive(
            &dec,
            &enc,
            RGFX_OP_COMPUTE_UPDATE_FENCE,
            RgfxComputeOpUpdateFence { fence_obj_id: 5 }.as_bytes(),
        );
        let want = Arg::Obj(Some(dec.require_fence(5).unwrap().id()));
        assert_eq!(p.call_log().calls_to("updateFence:")[0].args, vec![want]);
    }

    /// An array-tailed `setBuffers:` reads its head and element records, resolves
    /// each buffer (a nil entry kept), and sends the arrays and range.
    #[test]
    fn set_buffers_decodes_its_tail() {
        let (p, dec, enc) = setup();
        let mut msg = RgfxComputeOpSetBuffers {
            location: 3,
            length: 3,
        }
        .as_bytes()
        .to_vec();
        for (buffer_obj_id, offset) in [(2u64, 0u64), (0, 0), (3, 512)] {
            msg.extend_from_slice(
                RgfxComputeBufferBinding {
                    buffer_obj_id,
                    offset,
                }
                .as_bytes(),
            );
        }
        drive(&dec, &enc, RGFX_OP_COMPUTE_SET_BUFFERS, &msg);

        let calls = p.call_log().calls_to("setBuffers:offsets:withRange:");
        assert_eq!(calls.len(), 1);
        let b2 = dec.require_buffer(2).unwrap().id();
        let b3 = dec.require_buffer(3).unwrap().id();
        assert_eq!(
            calls[0].args,
            vec![
                Arg::Objs(vec![Some(b2), None, Some(b3)]),
                Arg::U64s(vec![0, 0, 512]),
                Arg::val(&NsRange::new(3, 3)),
            ]
        );
    }

    /// `setSamplerStates:lodMinClamps:lodMaxClamps:withRange:` carries the LOD clamps
    /// as f32 from their bits.
    #[test]
    fn sampler_lod_clamps_decode() {
        let (p, dec, enc) = setup();
        let mut msg = RgfxComputeOpSetSamplerStatesLodClamps {
            location: 0,
            length: 1,
        }
        .as_bytes()
        .to_vec();
        msg.extend_from_slice(
            RgfxComputeSamplerLodBinding {
                sampler_obj_id: 0, // nil sampler
                lod_min_clamp_bits: 0.5f32.to_bits(),
                lod_max_clamp_bits: 7.0f32.to_bits(),
            }
            .as_bytes(),
        );
        drive(
            &dec,
            &enc,
            RGFX_OP_COMPUTE_SET_SAMPLER_STATES_LOD_CLAMPS,
            &msg,
        );
        let calls = p
            .call_log()
            .calls_to("setSamplerStates:lodMinClamps:lodMaxClamps:withRange:");
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0].args,
            vec![
                Arg::Objs(vec![None]),
                Arg::F32s(vec![0.5]),
                Arg::F32s(vec![7.0]),
                Arg::val(&NsRange::new(0, 1)),
            ]
        );
    }

    /// A mixed `useResources:count:usage:` resolves a buffer and a texture by their
    /// wire kind, keeps a nil entry, and sends the usage.
    #[test]
    fn use_resources_decodes_mixed_kinds() {
        let (p, dec, enc) = setup();
        let mut msg = RgfxComputeOpUseResources { count: 3, usage: 1 }
            .as_bytes()
            .to_vec();
        for (kind, obj_id) in [
            (RGFX_RESOURCE_KIND_BUFFER, 2u64),
            (RGFX_RESOURCE_KIND_NONE, 0),
            (RGFX_RESOURCE_KIND_TEXTURE, 4),
        ] {
            msg.extend_from_slice(
                RgfxComputeResourceRef {
                    kind,
                    reserved_04: 0,
                    obj_id,
                }
                .as_bytes(),
            );
        }
        drive(&dec, &enc, RGFX_OP_COMPUTE_USE_RESOURCES, &msg);
        let calls = p.call_log().calls_to("useResources:count:usage:");
        assert_eq!(calls.len(), 1);
        let b2 = dec.require_buffer(2).unwrap().id();
        let t4 = dec.require_texture(4).unwrap().id();
        assert_eq!(
            calls[0].args,
            vec![Arg::Objs(vec![Some(b2), None, Some(t4)]), Arg::U64(1),]
        );
    }

    /// A body too short for its op faults the exec.
    #[test]
    fn a_short_body_faults() {
        let (_p, dec, enc) = setup();
        let r = decode_op(
            RGFX_OP_COMPUTE_SET_BUFFER_OFFSET,
            &[0u8; 4],
            &dec,
            &enc,
            &Marks::default(),
            OnFault::Return,
        );
        assert!(r.is_err());
    }

    /// An array-tailed op whose records run past the body faults the exec.
    #[test]
    fn a_short_tail_faults() {
        let (_p, dec, enc) = setup();
        // Claims three bindings but carries none.
        let msg = RgfxComputeOpSetBuffers {
            location: 0,
            length: 3,
        }
        .as_bytes()
        .to_vec();
        let r = decode_op(
            RGFX_OP_COMPUTE_SET_BUFFERS,
            &msg,
            &dec,
            &enc,
            &Marks::default(),
            OnFault::Return,
        );
        assert!(r.is_err());
    }

    /// A required reference the host does not hold faults the exec; a nil required
    /// reference faults too.
    #[test]
    fn a_bad_reference_raises() {
        let (_p, dec, enc) = setup();
        // A fence the host never made.
        let r = decode_op(
            RGFX_OP_COMPUTE_UPDATE_FENCE,
            RgfxComputeOpUpdateFence { fence_obj_id: 999 }.as_bytes(),
            &dec,
            &enc,
            &Marks::default(),
            OnFault::Return,
        );
        assert!(r.is_err());
        // A nil required indirect buffer for the stage-in region.
        let r = decode_op(
            RGFX_OP_COMPUTE_SET_STAGE_IN_REGION_INDIRECT,
            RgfxComputeOpSetStageInRegionIndirect {
                indirect_buffer_obj_id: 0,
                indirect_buffer_offset: 0,
            }
            .as_bytes(),
            &dec,
            &enc,
            &Marks::default(),
            OnFault::Return,
        );
        assert!(r.is_err());
    }

    /// An unknown resource kind faults the exec.
    #[test]
    fn an_unknown_resource_kind_faults() {
        let (_p, dec, enc) = setup();
        let mut msg = RgfxComputeOpUseResources { count: 1, usage: 0 }
            .as_bytes()
            .to_vec();
        msg.extend_from_slice(
            RgfxComputeResourceRef {
                kind: 99,
                reserved_04: 0,
                obj_id: 2,
            }
            .as_bytes(),
        );
        let r = decode_op(
            RGFX_OP_COMPUTE_USE_RESOURCES,
            &msg,
            &dec,
            &enc,
            &Marks::default(),
            OnFault::Return,
        );
        assert!(r.is_err());
    }

    /// An unknown compute tag is not handled (the walk then faults).
    #[test]
    fn an_unknown_tag_is_unhandled() {
        let (_p, dec, enc) = setup();
        assert!(
            !decode_op(
                0x2fff,
                &[0u8; 8],
                &dec,
                &enc,
                &Marks::default(),
                OnFault::Return
            )
            .unwrap()
        );
    }

    /// Every array-tailed op decodes an empty tail (count 0) without a record.
    #[test]
    fn empty_tails_decode() {
        let (p, dec, enc) = setup();
        let mark = p.call_log().mark();
        drive(
            &dec,
            &enc,
            RGFX_OP_COMPUTE_USE_HEAPS,
            RgfxComputeOpUseHeaps { count: 0 }.as_bytes(),
        );
        drive(
            &dec,
            &enc,
            RGFX_OP_COMPUTE_MEMORY_BARRIER_RESOURCES,
            RgfxComputeOpMemoryBarrierResources { count: 0 }.as_bytes(),
        );
        drive(
            &dec,
            &enc,
            RGFX_OP_COMPUTE_SET_TEXTURES,
            RgfxComputeOpSetTextures {
                location: 0,
                length: 0,
            }
            .as_bytes(),
        );
        assert_eq!(
            p.call_log().sels_since(mark),
            vec![
                "useHeaps:count:",
                "memoryBarrierWithResources:count:",
                "setTextures:withRange:",
            ]
        );
    }
}
