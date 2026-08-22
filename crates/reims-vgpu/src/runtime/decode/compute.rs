//! Compute command decoder (port of `host/utils/reims-vgpu-compute-decode`).

use reims_vgpu_core::endian::ld32;
use reims_vgpu_protocol::{HeapObject, ObjectTableRef, ResourceObject};
use reims_vgpu_wire::ops::compute as wire;

/// Shared serializer op-header length from `reims-vgpu-wire`.
use reims_vgpu_wire::OP_HEADER_LEN;

/// Residency on the compute rail.
///
/// # These are inherited, not unsupported
///
/// This doc used to say the pair had **no selector behind it at all**: the
/// compute encoder's own selector list carries no `useHeaps:`/`useResources:`,
/// so — the argument went — nothing in the serializer can produce these two
/// numbers, and `compute_noop_residency_hint` reading zero on a driven boot was
/// that conclusion confirmed.
///
/// The premise is true and the conclusion does not follow. Residency is declared
/// on the encoder base class, which every encoder derives from, in an
/// unqualified `useHeaps:count:` / `useResources:count:usage:` pair; only the
/// `stages:`-qualified overrides are declared on the render encoder, which is
/// why those are the only ones that appear in
/// [`reims_vgpu_wire::manifest`]. That manifest is built from each class's *own*
/// method list and has no row for the base class, so a base-class selector is
/// absent from it while being callable on every encoder — see the caveat in that
/// module, which this is the worked example of.
///
/// So a compute encoder does answer `useHeaps:count:`, and these are the numbers
/// it emits. The zero counter says this workload never issued one, which is the
/// ordinary reading of a healthy zero, not evidence that the arm is unreachable.
///
/// The layouts agree independently: the emitted record for `useHeaps:count:` is
/// a four-byte head and `count` four-byte refs, and for
/// `useResources:count:usage:` an eight-byte head and the same refs — which is
/// exactly [`COUNT_BASE`] and [`BIND_BASE`] below.
///
/// # The render rail is the one with the gap
///
/// `runtime::decode::render` carried `0x86`/`0x87` as its residency pair until a
/// capture replaced them with `0x1b`/`0x89`. That replacement was right for the
/// records it measured — those are the `stages:`-qualified forms — but it left
/// the render rail knowing only half the family, because a render encoder
/// inherits the unqualified pair too. An unqualified `useResources:count:usage:`
/// on a render encoder therefore reaches no render arm and is reported as
/// `render_unimplemented reason=accepted_without_executor` rather than counted
/// with its siblings under `render_noop_residency_hint`. No guest work is lost —
/// this device answers residency hints by doing nothing, for the reason
/// `runtime::exec` states — but the counter that exists to price that argument
/// sees only the qualified half.
pub const OP_USE_HEAPS: u32 = 0x86;
pub const OP_USE_RESOURCES: u32 = 0x87;

pub const REJECTED_85: u32 = 0x85;
pub const REJECTED_88: u32 = 0x88;
pub const REJECTED_C7: u32 = 0xc7;

const COUNT_BASE: usize = OP_HEADER_LEN + 4;
const BIND_BASE: usize = OP_HEADER_LEN + 8;
const REF_SIZE: usize = 4;
const BUF_ENTRY: usize = 12;
const BUF_STRIDE_ENTRY: usize = 20;
const SAMPLER_LOD_ENTRY: usize = 12;
/// A control-flow marker is the header alone; the crate that derived the
/// predicate form derived this one beside it.
///
/// `insertCompressedTextureReinterpretationFlush` shares this arm and is a
/// different family with its own derivation, so its length is asserted equal
/// rather than assumed: two records being the header alone is a fact about each
/// one, not a fact they share.
const EMPTY_LEN: usize = wire::CONTROL_FLOW_MARKER_TOTAL_LEN as usize;
const _: () = assert!(EMPTY_LEN == wire::INSERT_COMPRESSED_TEXTURE_FLUSH_TOTAL_LEN as usize);

/// Why the compute decoder refused a command.
///
/// No `Ok` and no `ErrArgs`, for the reason recorded on `blit::DecodeStatus`:
/// success is the result's own `Ok`, and a bad argument here is a payload
/// shorter than the field, which `ErrShort` already names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecodeStatus {
    ErrShort,
    ErrUnknownOpcode,
    ErrUnsupportedOpcode,
}

impl crate::observe::Refusal for DecodeStatus {
    /// Slugs carry a `compute_decode_` prefix: seven modules under
    /// `runtime/decode/` define a type called `DecodeStatus`, and five of them
    /// have an `ErrShort` that means a different read. Without the prefix the
    /// crate-wide uniqueness gate could not tell the compute decoder's refusals
    /// from any other's.
    fn refusal(&self) -> Option<&'static str> {
        Some(match self {
            Self::ErrShort => "compute_decode_short",
            Self::ErrUnknownOpcode => "compute_decode_unknown_opcode",
            Self::ErrUnsupportedOpcode => "compute_decode_unsupported_opcode",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OpcodeConfidence {
    #[default]
    Unknown = 0,
    AppleEmittedConfirmed,
    AppleRejected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Kind {
    #[default]
    Unknown = 0,
    UseHeaps,
    UseResources,
    Pipeline,
    BufferBind,
    BufferOffset,
    TextureBind,
    SamplerBind,
    SamplerLod,
    DispatchThreadgroups,
    DispatchThreadgroupsIndirect,
    DispatchThreads,
    StageInRegion,
    StageInRegionIndirect,
    ThreadgroupMemory,
    UpdateFence,
    WaitFence,
    BarrierResources,
    BarrierScope,
    ImageblockDimensions,
    BufferBindAttributeStride,
    BufferOffsetAttributeStride,
    DispatchType,
    DispatchThreadsIndirect,
    ControlStartDoWhile,
    ControlEndDoWhile,
    ControlStartWhile,
    ControlEndWhile,
    ControlStartIf,
    ControlStartElse,
    ControlEndIf,
    CompressedTextureFlush,
    ExecuteCommandsInBuffer,
    ExecuteCommandsInBufferIndirect,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Size3 {
    pub x: u64,
    pub y: u64,
    pub z: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferBinding {
    pub ref_: u32,
    pub offset: u64,
    pub attribute_stride: u64,
    pub has_attribute_stride: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RefBinding {
    pub ref_: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SamplerBinding {
    pub ref_: u32,
    pub lod_min_bits: u32,
    pub lod_max_bits: u32,
    pub has_lod_clamp: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Region3 {
    pub origin: Size3,
    pub size: Size3,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Command {
    pub opcode: u32,
    pub command_length: u32,
    pub kind: Kind,
    pub confidence: OpcodeConfidence,
    pub pipeline_ref: u32,
    pub first: u32,
    pub count: u32,
    pub buffers: Vec<BufferBinding>,
    pub textures: Vec<RefBinding>,
    pub samplers: Vec<SamplerBinding>,
    pub resources: Vec<ObjectTableRef<ResourceObject>>,
    pub heaps: Vec<ObjectTableRef<HeapObject>>,
    pub grid: Size3,
    pub threads_per_threadgroup: Size3,
    pub indirect_buffer_ref: u32,
    pub indirect_buffer_offset: u64,
    pub buffer_offset: u64,
    pub attribute_stride: u64,
    pub imageblock_width: u32,
    pub imageblock_height: u32,
    pub dispatch_type: u32,
    pub stage_in_region: Region3,
    pub stage_in_indirect_buffer_ref: u32,
    pub stage_in_indirect_buffer_offset: u64,
    pub threadgroup_memory_length: u64,
    pub threadgroup_memory_index: u32,
    pub resource_usage: u32,
    pub fence_ref: u32,
    pub barrier_scope: u16,
    pub condition_buffer_ref: u32,
    pub condition_buffer_offset: u64,
    pub condition_comparison: u32,
    pub condition_reference_value: u32,
    pub indirect_command_buffer_ref: u32,
    pub indirect_command_range_location: u64,
    pub indirect_command_range_length: u64,
    pub indirect_command_arguments_buffer_ref: u32,
    pub indirect_command_arguments_buffer_offset: u64,
}

pub fn opcode_supported(opcode: u32) -> bool {
    matches!(
        opcode,
        OP_USE_HEAPS
            | OP_USE_RESOURCES
            | wire::OPCODE_DISPATCH_THREADGROUPS
            | wire::OPCODE_DISPATCH_THREADGROUPS_INDIRECT
            | wire::OPCODE_DISPATCH_THREADS
            | wire::OPCODE_SET_BUFFER
            | wire::OPCODE_SET_SAMPLER
            | wire::OPCODE_SET_SAMPLER_LOD
            | wire::OPCODE_SET_TEXTURE
            | wire::OPCODE_SET_BUFFER_OFFSET
            | wire::OPCODE_SET_PIPELINE_STATE
            | wire::OPCODE_SET_STAGE_IN_REGION
            | wire::OPCODE_SET_STAGE_IN_REGION_INDIRECT
            | wire::OPCODE_SET_THREADGROUP_MEMORY_LENGTH
            | wire::OPCODE_UPDATE_FENCE
            | wire::OPCODE_WAIT_FOR_FENCE
            | wire::OPCODE_MEMORY_BARRIER_RESOURCES
            | wire::OPCODE_MEMORY_BARRIER_SCOPE
            | wire::OPCODE_SET_IMAGEBLOCK_SIZE
            | wire::OPCODE_SET_BUFFER_STRIDE
            | wire::OPCODE_SET_BUFFER_OFFSET_STRIDE
            | wire::OPCODE_WRITE_DESCRIPTOR
            | wire::OPCODE_START_DO_WHILE
            | wire::OPCODE_END_DO_WHILE
            | wire::OPCODE_START_WHILE
            | wire::OPCODE_END_WHILE
            | wire::OPCODE_START_IF
            | wire::OPCODE_START_ELSE
            | wire::OPCODE_END_IF
            | wire::OPCODE_INSERT_COMPRESSED_TEXTURE_FLUSH
            | wire::OPCODE_EXECUTE_COMMANDS_RANGE
            | wire::OPCODE_EXECUTE_COMMANDS_INDIRECT
            | wire::OPCODE_DISPATCH_THREADS_INDIRECT
    )
}

pub fn opcode_apple_rejected(opcode: u32) -> bool {
    matches!(opcode, REJECTED_85 | REJECTED_88 | REJECTED_C7)
}

pub fn opcode_confidence(opcode: u32) -> OpcodeConfidence {
    if opcode_supported(opcode) {
        OpcodeConfidence::AppleEmittedConfirmed
    } else if opcode_apple_rejected(opcode) {
        OpcodeConfidence::AppleRejected
    } else {
        OpcodeConfidence::Unknown
    }
}

/// Whether a variable-length record's declared length is exactly what `count`
/// entries of `stride` need.
///
/// **A bind record is bounded by its own length and by nothing else.** Seven
/// call sites used to test `count > MAX_BIND_ENTRIES` first and refuse the whole
/// record with `ErrTooManyBindings`; the constant was 128 and carried no
/// citation. `runtime::decode::render` removed the identical check for the
/// identical reason — its cap was 32 and refused a 40-slot texture bind Apple's
/// serializer really produces, dropping all forty rather than the eight that
/// would not fit — and `bind_record_len` there states the rule this doc now
/// states here.
///
/// The cap was also redundant, which is why removing it costs no safety. The
/// count is never trusted before this check, `command_length <= command.len()`
/// is established at the top of [`decode`], and the arithmetic below is done in
/// `u64` and bounded at `u32::MAX`, so nothing is read or pushed until the
/// entries are known to be inside the record the guest itself sized. What the
/// cap added was a second, lower bound with no derivation behind it.
///
/// Which index Vulkan can actually bind is a backend question. A decoder that
/// pre-empts that answer can refuse otherwise valid guest work.
///
/// **The limit those caps were reaching for has since been measured, and it is
/// three numbers rather than one** — see [`reims_vgpu_wire::ops::bind_limit`],
/// where fixtures pin them. Apple's serializer truncates a plural bind's range
/// at the stage's argument table: 128 textures, **31** buffers, 16 samplers,
/// with the bound falling on `first + count` rather than on `count`.
///
/// That is the sharper reason a single `MAX_BIND_ENTRIES` could not have been
/// right at these seven sites. The compute rail's 128 was correct for the two
/// texture sites and four times too permissive for the buffer ones; the render
/// rail's 32 was above the buffer limit and a quarter of the texture one, which
/// is the half that fired. A cap that is too permissive never refuses anything
/// and so never looks wrong.
///
/// It does **not** license reinstating them as three constants. These describe
/// what Apple's serializer *emits*; they are not a validity check on bytes
/// arriving from a guest, and refusing a record for declaring more entries than
/// Apple currently writes loses every bind in it the first time a limit moves.
/// The record's own declared length is the bound, and it always was.
fn var_len(cmd_len: usize, base: usize, count: u32, stride: usize) -> bool {
    let expected = base as u64 + (count as u64) * (stride as u64);
    expected <= u32::MAX as u64 && cmd_len == expected as usize
}

/// Transactional compute command decode.
///
/// Framing and covered layouts come from [`reims_vgpu_wire`]: [`reims_vgpu_wire::op`]
/// for the shared header and the parsers in [`reims_vgpu_wire::ops::compute`] for
/// each payload this encoder owns. Local residency opcodes (`OP_USE_HEAPS` /
/// `OP_USE_RESOURCES`) have no wire export and stay hand-read.
pub fn decode(command: &[u8]) -> Result<Command, DecodeStatus> {
    let op = reims_vgpu_wire::op(command, 0).map_err(|_| DecodeStatus::ErrShort)?;
    let opcode = op.opcode();
    let command_length = op.length() as usize;
    let confidence = opcode_confidence(opcode);
    if confidence == OpcodeConfidence::Unknown {
        return Err(DecodeStatus::ErrUnknownOpcode);
    }
    if !opcode_supported(opcode) {
        return Err(DecodeStatus::ErrUnsupportedOpcode);
    }
    let payload = op.payload;
    let mut out = Command {
        opcode,
        command_length: op.length(),
        confidence,
        ..Default::default()
    };

    match opcode {
        OP_USE_HEAPS => {
            // No wire export: Apple's compute serializer has no useHeaps: selector.
            if command_length < COUNT_BASE {
                return Err(DecodeStatus::ErrShort);
            }
            let count = ld32(&payload[0..]);
            if !var_len(command_length, COUNT_BASE, count, REF_SIZE) {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::UseHeaps;
            out.count = count;
            for i in 0..count as usize {
                out.heaps
                    .push(ObjectTableRef::new(ld32(&payload[4 + i * REF_SIZE..])));
            }
            Ok(out)
        }
        OP_USE_RESOURCES => {
            // No wire export: same gap as OP_USE_HEAPS.
            if command_length < BIND_BASE {
                return Err(DecodeStatus::ErrShort);
            }
            let count = ld32(&payload[0..]);
            if !var_len(command_length, BIND_BASE, count, REF_SIZE) {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::UseResources;
            out.count = count;
            out.resource_usage = ld32(&payload[4..]);
            for i in 0..count as usize {
                out.resources
                    .push(ObjectTableRef::new(ld32(&payload[8 + i * REF_SIZE..])));
            }
            Ok(out)
        }
        wire::OPCODE_SET_PIPELINE_STATE => {
            if command_length != wire::SET_PIPELINE_STATE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let r = wire::set_pipeline_state(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::Pipeline;
            out.pipeline_ref = r.object_ref.get();
            Ok(out)
        }
        wire::OPCODE_SET_BUFFER => {
            let (head, entries) = wire::buffer_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            if !var_len(command_length, BIND_BASE, head.count.get(), BUF_ENTRY) {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::BufferBind;
            out.first = head.first.get();
            out.count = head.count.get();
            for e in entries {
                out.buffers.push(BufferBinding {
                    ref_: e.buffer_ref.get(),
                    offset: e.offset.get(),
                    ..Default::default()
                });
            }
            Ok(out)
        }
        wire::OPCODE_SET_BUFFER_STRIDE => {
            let (head, entries) =
                wire::buffer_stride_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            if !var_len(
                command_length,
                BIND_BASE,
                head.count.get(),
                BUF_STRIDE_ENTRY,
            ) {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::BufferBindAttributeStride;
            out.first = head.first.get();
            out.count = head.count.get();
            for e in entries {
                out.buffers.push(BufferBinding {
                    ref_: e.buffer_ref.get(),
                    offset: e.offset.get(),
                    attribute_stride: e.attribute_stride.get(),
                    has_attribute_stride: true,
                });
            }
            Ok(out)
        }
        wire::OPCODE_SET_SAMPLER | wire::OPCODE_SET_TEXTURE => {
            let (head, entries) = wire::ref_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            if !var_len(command_length, BIND_BASE, head.count.get(), REF_SIZE) {
                return Err(DecodeStatus::ErrShort);
            }
            let samplers = opcode == wire::OPCODE_SET_SAMPLER;
            out.kind = if samplers {
                Kind::SamplerBind
            } else {
                Kind::TextureBind
            };
            out.first = head.first.get();
            out.count = head.count.get();
            for e in entries {
                let ref_ = e.object_ref.get();
                if samplers {
                    out.samplers.push(SamplerBinding {
                        ref_,
                        ..Default::default()
                    });
                } else {
                    out.textures.push(RefBinding { ref_ });
                }
            }
            Ok(out)
        }
        wire::OPCODE_SET_SAMPLER_LOD => {
            let (head, entries) =
                wire::sampler_lod_binds(&op).map_err(|_| DecodeStatus::ErrShort)?;
            if !var_len(
                command_length,
                BIND_BASE,
                head.count.get(),
                SAMPLER_LOD_ENTRY,
            ) {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::SamplerLod;
            out.first = head.first.get();
            out.count = head.count.get();
            for e in entries {
                out.samplers.push(SamplerBinding {
                    ref_: e.sampler_ref.get(),
                    lod_min_bits: e.lod_min_clamp.get().to_bits(),
                    lod_max_bits: e.lod_max_clamp.get().to_bits(),
                    has_lod_clamp: true,
                });
            }
            Ok(out)
        }
        wire::OPCODE_SET_BUFFER_OFFSET => {
            if command_length != wire::SET_BUFFER_OFFSET_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let b = wire::set_buffer_offset(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::BufferOffset;
            out.first = b.index.get();
            out.buffer_offset = b.offset.get();
            Ok(out)
        }
        wire::OPCODE_SET_BUFFER_OFFSET_STRIDE => {
            if command_length != wire::SET_BUFFER_OFFSET_STRIDE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let b = wire::buffer_offset_stride(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::BufferOffsetAttributeStride;
            out.first = b.index.get();
            out.buffer_offset = b.offset.get();
            out.attribute_stride = b.attribute_stride.get();
            Ok(out)
        }
        wire::OPCODE_DISPATCH_THREADGROUPS | wire::OPCODE_DISPATCH_THREADS => {
            if command_length != wire::DISPATCH_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let d = wire::dispatch(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = if opcode == wire::OPCODE_DISPATCH_THREADGROUPS {
                Kind::DispatchThreadgroups
            } else {
                Kind::DispatchThreads
            };
            out.grid = Size3 {
                x: d.groups_width.get(),
                y: d.groups_height.get(),
                z: d.groups_depth.get(),
            };
            out.threads_per_threadgroup = Size3 {
                x: d.threads_width.get(),
                y: d.threads_height.get(),
                z: d.threads_depth.get(),
            };
            Ok(out)
        }
        wire::OPCODE_DISPATCH_THREADGROUPS_INDIRECT => {
            if command_length != wire::DISPATCH_THREADGROUPS_INDIRECT_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let d = wire::dispatch_indirect(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::DispatchThreadgroupsIndirect;
            out.threads_per_threadgroup = Size3 {
                x: d.threads_width.get(),
                y: d.threads_height.get(),
                z: d.threads_depth.get(),
            };
            out.indirect_buffer_offset = d.indirect_buffer_offset.get();
            out.indirect_buffer_ref = d.indirect_buffer_ref.get();
            Ok(out)
        }
        wire::OPCODE_DISPATCH_THREADS_INDIRECT => {
            if command_length != wire::DISPATCH_THREADS_INDIRECT_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let d = wire::dispatch_threads_indirect(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::DispatchThreadsIndirect;
            out.indirect_buffer_offset = d.indirect_buffer_offset.get();
            out.indirect_buffer_ref = d.indirect_buffer_ref.get();
            Ok(out)
        }
        wire::OPCODE_SET_STAGE_IN_REGION => {
            if command_length != wire::SET_STAGE_IN_REGION_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let r = wire::set_stage_in_region(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::StageInRegion;
            out.stage_in_region.size = Size3 {
                x: r.size_width.get(),
                y: r.size_height.get(),
                z: r.size_depth.get(),
            };
            out.stage_in_region.origin = Size3 {
                x: r.origin_x.get(),
                y: r.origin_y.get(),
                z: r.origin_z.get(),
            };
            Ok(out)
        }
        wire::OPCODE_SET_STAGE_IN_REGION_INDIRECT => {
            if command_length != wire::SET_STAGE_IN_REGION_INDIRECT_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let r = wire::set_stage_in_region_indirect(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::StageInRegionIndirect;
            out.stage_in_indirect_buffer_ref = r.indirect_buffer_ref.get();
            out.stage_in_indirect_buffer_offset = r.indirect_buffer_offset.get();
            Ok(out)
        }
        wire::OPCODE_SET_THREADGROUP_MEMORY_LENGTH => {
            if command_length != wire::SET_THREADGROUP_MEMORY_LENGTH_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let t = wire::set_threadgroup_memory_length(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::ThreadgroupMemory;
            out.threadgroup_memory_length = t.length.get();
            out.threadgroup_memory_index = t.index.get();
            Ok(out)
        }
        wire::OPCODE_UPDATE_FENCE | wire::OPCODE_WAIT_FOR_FENCE => {
            if command_length != wire::FENCE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let r = wire::fence(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = if opcode == wire::OPCODE_UPDATE_FENCE {
                Kind::UpdateFence
            } else {
                Kind::WaitFence
            };
            out.fence_ref = r.object_ref.get();
            Ok(out)
        }
        wire::OPCODE_MEMORY_BARRIER_RESOURCES => {
            let (head, refs) =
                wire::memory_barrier_resources(&op).map_err(|_| DecodeStatus::ErrShort)?;
            if !var_len(command_length, COUNT_BASE, head.count.get(), REF_SIZE) {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = Kind::BarrierResources;
            out.count = head.count.get();
            for r in refs {
                out.resources.push(ObjectTableRef::new(r.object_ref.get()));
            }
            Ok(out)
        }
        wire::OPCODE_MEMORY_BARRIER_SCOPE => {
            if command_length != wire::MEMORY_BARRIER_SCOPE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let s = wire::memory_barrier_scope(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::BarrierScope;
            out.barrier_scope = s.scope.get();
            Ok(out)
        }
        wire::OPCODE_SET_IMAGEBLOCK_SIZE => {
            if command_length != wire::SET_IMAGEBLOCK_SIZE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let s = wire::set_imageblock_size(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::ImageblockDimensions;
            out.imageblock_width = s.width.get();
            out.imageblock_height = s.height.get();
            Ok(out)
        }
        wire::OPCODE_WRITE_DESCRIPTOR => {
            if command_length != wire::WRITE_DESCRIPTOR_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let d = wire::write_descriptor(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::DispatchType;
            out.dispatch_type = d.dispatch_type.get();
            Ok(out)
        }
        wire::OPCODE_START_WHILE | wire::OPCODE_START_IF | wire::OPCODE_END_DO_WHILE => {
            if command_length != wire::CONTROL_FLOW_PREDICATE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let p = wire::control_flow_predicate(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = match opcode {
                wire::OPCODE_START_WHILE => Kind::ControlStartWhile,
                wire::OPCODE_START_IF => Kind::ControlStartIf,
                _ => Kind::ControlEndDoWhile,
            };
            out.condition_buffer_ref = p.buffer_ref.get();
            out.condition_buffer_offset = p.offset.get();
            out.condition_comparison = p.comparison.get();
            out.condition_reference_value = p.reference_value.get();
            Ok(out)
        }
        wire::OPCODE_START_DO_WHILE
        | wire::OPCODE_END_WHILE
        | wire::OPCODE_START_ELSE
        | wire::OPCODE_END_IF
        | wire::OPCODE_INSERT_COMPRESSED_TEXTURE_FLUSH => {
            if command_length != EMPTY_LEN {
                return Err(DecodeStatus::ErrShort);
            }
            out.kind = match opcode {
                wire::OPCODE_START_DO_WHILE => Kind::ControlStartDoWhile,
                wire::OPCODE_END_WHILE => Kind::ControlEndWhile,
                wire::OPCODE_START_ELSE => Kind::ControlStartElse,
                wire::OPCODE_END_IF => Kind::ControlEndIf,
                _ => Kind::CompressedTextureFlush,
            };
            Ok(out)
        }
        wire::OPCODE_EXECUTE_COMMANDS_RANGE => {
            if command_length != wire::EXECUTE_COMMANDS_RANGE_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let e = wire::execute_commands_range(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::ExecuteCommandsInBuffer;
            out.indirect_command_buffer_ref = e.icb_ref.get();
            out.indirect_command_range_location = e.range_location.get();
            out.indirect_command_range_length = e.range_length.get();
            Ok(out)
        }
        wire::OPCODE_EXECUTE_COMMANDS_INDIRECT => {
            if command_length != wire::EXECUTE_COMMANDS_INDIRECT_TOTAL_LEN as usize {
                return Err(DecodeStatus::ErrShort);
            }
            let e = wire::execute_commands_indirect(&op).map_err(|_| DecodeStatus::ErrShort)?;
            out.kind = Kind::ExecuteCommandsInBufferIndirect;
            out.indirect_command_buffer_ref = e.icb_ref.get();
            out.indirect_command_arguments_buffer_ref = e.indirect_buffer_ref.get();
            out.indirect_command_arguments_buffer_offset = e.indirect_buffer_offset.get();
            Ok(out)
        }
        _ => Err(DecodeStatus::ErrUnknownOpcode),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use reims_vgpu_core::endian::st32;

    /// Header plus two `Size3`, and taken from the crate that pins it.
    const DISPATCH_DIRECT_LEN: usize = wire::DISPATCH_TOTAL_LEN as usize;
    /// Header plus two `Size3` — the region's size and its origin.
    const BARRIER_SCOPE_LEN: usize = wire::MEMORY_BARRIER_SCOPE_TOTAL_LEN as usize;
    /// The condition record's length, from the crate that derived it.
    ///
    /// Was `0x1c` written here. It is the same number, and that is the point:
    /// the two agreed by luck for as long as nobody could check, because
    /// `-setSupportsCommandBufferJump:` defaults off and the capture recorded
    /// all seven control-flow selectors as emitting nothing at all.
    const CONDITION_LEN: usize = wire::CONTROL_FLOW_PREDICATE_TOTAL_LEN as usize;
    const EXECUTE_LEN: usize = wire::EXECUTE_COMMANDS_RANGE_TOTAL_LEN as usize;
    const PIPELINE_LEN: usize = wire::SET_PIPELINE_STATE_TOTAL_LEN as usize;

    /// A malformed compute command used to be dropped at the dispatch site with no
    /// log line at all — indistinguishable from a segment carrying no compute
    /// work. Each check names itself now, `Ok` still produces nothing, and the
    /// prefix keeps them apart from the six sibling `DecodeStatus` enums.
    #[test]
    fn every_compute_decode_failure_names_its_own_check() {
        use crate::observe::Refusal;
        const ERRS: &[DecodeStatus] = &[
            DecodeStatus::ErrShort,
            DecodeStatus::ErrUnknownOpcode,
            DecodeStatus::ErrUnsupportedOpcode,
        ];
        let mut slugs: Vec<&str> = ERRS.iter().filter_map(|s| s.refusal()).collect();
        assert_eq!(slugs.len(), ERRS.len(), "every error variant refuses");
        assert!(slugs.iter().all(|s| s.starts_with("compute_decode_")));
        slugs.sort_unstable();
        let n = slugs.len();
        slugs.dedup();
        assert_eq!(slugs.len(), n, "two compute decode checks share a slug");
    }
    fn hdr(op: u32, len: usize) -> Vec<u8> {
        let mut v = vec![0u8; len];
        st32(&mut v[0..4], op);
        st32(&mut v[4..8], len as u32);
        v
    }

    /// A bind record above the old `MAX_BIND_ENTRIES = 128` cap decodes, and
    /// every entry of it survives.
    ///
    /// This is the compute half of the bug `runtime::decode::render` fixed by
    /// deleting its own cap: a well-formed record was refused whole, so a guest
    /// binding 200 textures lost all 200 rather than the 72 that would not have
    /// fit. Metal's own per-stage texture limit is 128, but that is the
    /// *backend's* bound to enforce and it is enforced there — a decoder that
    /// pre-empts it turns a bind the backend would have clamped into a decode
    /// failure with a shape slug, which the divergence instrument reads as this
    /// project having the layout wrong.
    ///
    /// 200 rather than 129 so the assertion cannot pass by an off-by-one in a
    /// bound that is supposed to be gone entirely.
    #[test]
    fn a_bind_larger_than_the_deleted_cap_decodes_every_entry() {
        const COUNT: u32 = 200;
        let len = BIND_BASE + COUNT as usize * REF_SIZE;
        let mut v = hdr(wire::OPCODE_SET_TEXTURE, len);
        st32(&mut v[8..], 0); // first
        st32(&mut v[12..], COUNT);
        for i in 0..COUNT as usize {
            st32(&mut v[BIND_BASE + i * REF_SIZE..], 1000 + i as u32);
        }
        let c = decode(&v).expect("a record the guest sized correctly must decode");
        assert_eq!(c.kind, Kind::TextureBind);
        assert_eq!(c.count, COUNT);
        assert_eq!(c.textures.len(), COUNT as usize);
        assert_eq!(c.textures[0].ref_, 1000);
        assert_eq!(c.textures[COUNT as usize - 1].ref_, 1000 + COUNT - 1);

        // The record's own length is still the bound, and it is exact: one
        // entry's worth of slack either way is `ErrShort`, not a silent
        // truncation. Deleting the cap must not have loosened this.
        let mut short = v.clone();
        st32(&mut short[12..], COUNT + 1);
        assert_eq!(decode(&short).unwrap_err(), DecodeStatus::ErrShort);
        st32(&mut short[12..], COUNT - 1);
        assert_eq!(decode(&short).unwrap_err(), DecodeStatus::ErrShort);
        // A count whose entries would overflow the length arithmetic is refused
        // by the same check rather than by a cap.
        st32(&mut short[12..], u32::MAX);
        assert_eq!(decode(&short).unwrap_err(), DecodeStatus::ErrShort);
    }

    #[test]
    fn pipeline_and_dispatch() {
        let mut v = hdr(wire::OPCODE_SET_PIPELINE_STATE, PIPELINE_LEN);
        st32(&mut v[8..], 42);
        let c = decode(&v).unwrap();
        assert_eq!(c.kind, Kind::Pipeline);
        assert_eq!(c.pipeline_ref, 42);

        let v = hdr(wire::OPCODE_DISPATCH_THREADGROUPS, DISPATCH_DIRECT_LEN);
        let c = decode(&v).unwrap();
        assert_eq!(c.kind, Kind::DispatchThreadgroups);
    }

    #[test]
    fn rejected_and_matrix() {
        assert!(opcode_supported(wire::OPCODE_SET_PIPELINE_STATE));
        assert!(opcode_apple_rejected(REJECTED_85));
        assert_eq!(
            decode(&hdr(REJECTED_85, 16)).unwrap_err(),
            DecodeStatus::ErrUnsupportedOpcode
        );
        assert_eq!(
            decode(&hdr(0x999, 16)).unwrap_err(),
            DecodeStatus::ErrUnknownOpcode
        );
    }

    #[test]
    fn set_buffers() {
        let count = 2u32;
        let len = BIND_BASE + (count as usize) * BUF_ENTRY;
        let mut v = hdr(wire::OPCODE_SET_BUFFER, len);
        st32(&mut v[8..], 1); // first
        st32(&mut v[12..], count);
        st32(&mut v[16..], 10);
        st32(&mut v[28..], 11);
        let c = decode(&v).unwrap();
        assert_eq!(c.count, 2);
        assert_eq!(c.buffers[0].ref_, 10);
        assert_eq!(c.buffers[1].ref_, 11);
    }

    #[test]
    fn property_fuzz_opcodes() {
        for op in 0x80u32..0xf0 {
            for len in [8, 12, 16, 20, 28, 0x14, 0x1c, 0x38, 0x40] {
                let _ = decode(&hdr(op, len));
            }
        }
    }

    #[test]
    fn control_do_while_start_empty_end_condition() {
        // Wire contract: start-do-while is empty; end-do-while carries condition.
        let v = hdr(wire::OPCODE_START_DO_WHILE, EMPTY_LEN);
        let c = decode(&v).unwrap();
        assert_eq!(c.kind, Kind::ControlStartDoWhile);

        let mut v = hdr(wire::OPCODE_END_DO_WHILE, CONDITION_LEN);
        st32(&mut v[8..], 1201); // buffer ref
                                 // offset u64 @ +4 payload = absolute +12
        v[12..20].copy_from_slice(&0x640u64.to_le_bytes());
        st32(&mut v[20..], 2); // comparison Equal
        st32(&mut v[24..], 0x1234_5678);
        let c = decode(&v).unwrap();
        assert_eq!(c.kind, Kind::ControlEndDoWhile);
        assert_eq!(c.condition_buffer_ref, 1201);
        assert_eq!(c.condition_buffer_offset, 0x640);
        assert_eq!(c.condition_comparison, 2);
        assert_eq!(c.condition_reference_value, 0x1234_5678);

        // Swapped lengths must fail closed.
        assert!(decode(&hdr(wire::OPCODE_START_DO_WHILE, CONDITION_LEN)).is_err());
        assert!(decode(&hdr(wire::OPCODE_END_DO_WHILE, EMPTY_LEN)).is_err());
    }

    /// The control-flow layout this device uses is Apple's, field for field.
    ///
    /// It arrived here as a port with no derivation recorded, and
    /// `compute_ctrl_seen` has never fired on a driven boot — so nothing said
    /// whether it was right, and the wire capture could not say either while
    /// `-setSupportsCommandBufferJump:` was at its default and all seven
    /// selectors looked like records Apple never writes. Driving them showed
    /// this arm had every offset and every width correct.
    ///
    /// This asserts the offsets against `offset_of!` on the derived struct
    /// rather than against the numbers that were here, so the check is against
    /// Apple's layout rather than against this module's memory of it. The
    /// interesting one is `comparison`: it is `Q` on all three selectors and
    /// four bytes on the wire, and widening it to match the API would push
    /// `reference_value` off the end.
    #[test]
    fn the_control_flow_condition_is_apples_own_layout() {
        use core::mem::offset_of;

        assert_eq!(
            CONDITION_LEN,
            OP_HEADER_LEN + core::mem::size_of::<wire::ControlFlowPredicate>(),
            "the condition record is its header plus the derived body"
        );
        assert_eq!(offset_of!(wire::ControlFlowPredicate, buffer_ref), 0);
        assert_eq!(offset_of!(wire::ControlFlowPredicate, offset), 4);
        assert_eq!(offset_of!(wire::ControlFlowPredicate, comparison), 12);
        assert_eq!(offset_of!(wire::ControlFlowPredicate, reference_value), 16);

        // All three condition-bearing opcodes read the same body. Driven per
        // opcode rather than generalized from one, because the three are
        // separate arms and only the `Kind` is supposed to differ.
        for (op, kind) in [
            (wire::OPCODE_START_IF, Kind::ControlStartIf),
            (wire::OPCODE_START_WHILE, Kind::ControlStartWhile),
            (wire::OPCODE_END_DO_WHILE, Kind::ControlEndDoWhile),
        ] {
            let mut v = hdr(op, CONDITION_LEN);
            st32(&mut v[OP_HEADER_LEN..], 5151);
            v[OP_HEADER_LEN + 4..OP_HEADER_LEN + 12].copy_from_slice(&0x1111u64.to_le_bytes());
            st32(&mut v[OP_HEADER_LEN + 12..], 0x22);
            st32(&mut v[OP_HEADER_LEN + 16..], 0x89ab_cdef);
            let c = decode(&v).unwrap_or_else(|e| panic!("op {op:#x}: {e:?}"));
            assert_eq!(c.kind, kind);
            assert_eq!(c.condition_buffer_ref, 5151);
            assert_eq!(c.condition_buffer_offset, 0x1111);
            // Outside `MTLCompareFunction`'s 0–7 on purpose: the serializer
            // carries the guest's ordinal verbatim rather than validating or
            // remapping it, so a reader must treat an out-of-range value as
            // guest data rather than as impossible.
            assert_eq!(c.condition_comparison, 0x22);
            assert_eq!(c.condition_reference_value, 0x89ab_cdef);
        }
    }

    #[test]
    fn control_if_while_and_icb_lengths() {
        let mut v = hdr(wire::OPCODE_START_IF, CONDITION_LEN);
        st32(&mut v[8..], 7);
        let c = decode(&v).unwrap();
        assert_eq!(c.kind, Kind::ControlStartIf);
        assert_eq!(c.condition_buffer_ref, 7);

        assert_eq!(
            decode(&hdr(wire::OPCODE_START_ELSE, EMPTY_LEN))
                .unwrap()
                .kind,
            Kind::ControlStartElse
        );
        assert_eq!(
            decode(&hdr(wire::OPCODE_END_IF, EMPTY_LEN)).unwrap().kind,
            Kind::ControlEndIf
        );

        let mut v = hdr(wire::OPCODE_EXECUTE_COMMANDS_RANGE, EXECUTE_LEN);
        st32(&mut v[8..], 1301);
        v[12..20].copy_from_slice(&3u64.to_le_bytes());
        v[20..28].copy_from_slice(&7u64.to_le_bytes());
        let c = decode(&v).unwrap();
        assert_eq!(c.kind, Kind::ExecuteCommandsInBuffer);
        assert_eq!(c.indirect_command_buffer_ref, 1301);
        assert_eq!(c.indirect_command_range_location, 3);
        assert_eq!(c.indirect_command_range_length, 7);
    }

    /// A compute resource barrier is the length the serializer writes.
    ///
    /// It is `12 + 4 * count`: eight bytes of header, the count, then the refs.
    /// This arm demanded `16 + 4 * count`, four bytes further on, so every
    /// barrier a guest issued was refused with `ErrShort` and every resource it
    /// named was lost — and the four bytes it skipped were the first ref, so a
    /// record long enough to pass would have read the list shifted by one.
    ///
    /// The lengths come from `reims_vgpu_wire::ops::compute`, which pins them
    /// against bytes Apple's serializer produced. Both directions are asserted:
    /// the length Apple writes is accepted, and the one this arm used to demand
    /// is refused.
    #[test]
    fn a_resource_barrier_is_the_length_the_serializer_writes() {
        use reims_vgpu_core::endian::st32;

        const COUNT: u32 = 2;
        let apple_len = COUNT_BASE + (COUNT as usize) * REF_SIZE;
        assert_eq!(apple_len, 20, "the serializer's own record is 20 bytes");

        let mut v = vec![0u8; apple_len];
        st32(&mut v[0..], wire::OPCODE_MEMORY_BARRIER_RESOURCES);
        st32(&mut v[4..], apple_len as u32);
        st32(&mut v[OP_HEADER_LEN..], COUNT);
        st32(&mut v[OP_HEADER_LEN + 4..], 5151);
        st32(&mut v[OP_HEADER_LEN + 8..], 4343);

        let c = decode(&v).expect("the serializer's own record must decode");
        assert_eq!(c.kind, Kind::BarrierResources);
        assert_eq!(c.count, COUNT);
        let refs: Vec<u32> = c.resources.iter().map(|r| r.get()).collect();
        assert_eq!(
            refs,
            vec![5151, 4343],
            "the refs start at the count, not four bytes past it"
        );

        // The length this arm used to require. Nothing writes it, so it must
        // not be the one that decodes.
        let old_len = BIND_BASE + (COUNT as usize) * REF_SIZE;
        let mut v = vec![0u8; old_len];
        st32(&mut v[0..], wire::OPCODE_MEMORY_BARRIER_RESOURCES);
        st32(&mut v[4..], old_len as u32);
        st32(&mut v[OP_HEADER_LEN..], COUNT);
        assert_eq!(decode(&v).unwrap_err(), DecodeStatus::ErrShort);
    }

    /// The scope barrier lifts only the bytes the serializer wrote.
    ///
    /// `compute_memory_barrier_scope` is `04 00 AA AA` against the oracle's
    /// poison: two bytes written, two never touched. On a guest's wire those
    /// two hold whatever the ring last contained, so a field reading them
    /// reports noise. Poison here stands in for that ring content — if a future
    /// change grew a field into those bytes, this decodes `0xAAAA` into it.
    #[test]
    fn the_scope_barrier_reads_no_byte_the_serializer_left_alone() {
        use reims_vgpu_core::endian::st32;

        let mut v = vec![0xAAu8; BARRIER_SCOPE_LEN];
        st32(&mut v[0..], wire::OPCODE_MEMORY_BARRIER_SCOPE);
        st32(&mut v[4..], BARRIER_SCOPE_LEN as u32);
        v[OP_HEADER_LEN] = 4;
        v[OP_HEADER_LEN + 1] = 0;

        let c = decode(&v).expect("the serializer's own record must decode");
        assert_eq!(c.kind, Kind::BarrierScope);
        assert_eq!(c.barrier_scope, 4);
        // Everything else the record could carry stays at its default, so no
        // field picked up the two unwritten bytes.
        assert_eq!(c.count, 0);
        assert_eq!(c.resource_usage, 0);
        assert_eq!(c.fence_ref, 0);
    }

    /// The compute encoder does not *declare* residency — it inherits it.
    ///
    /// This asserts exactly one thing: no residency selector appears among the
    /// compute encoder's own methods. That is all the manifest can say, because
    /// it is built from each class's own method list and has no row for the
    /// encoder base class where residency is declared.
    ///
    /// It used to be read as the stronger claim that these two opcodes have no
    /// producer at all, and [`OP_USE_HEAPS`] records why that does not follow.
    /// The assertion is unchanged and still worth keeping: if a future build
    /// *overrides* residency on this class, the override may carry a different
    /// record shape from the inherited one, and this fires before the arm below
    /// decodes the new shape with the old layout.
    #[test]
    fn the_compute_encoder_declares_no_residency_selector() {
        let residency: Vec<&str> = reims_vgpu_wire::manifest::MANIFEST
            .iter()
            .filter(|e| e.class == "PGSerializerComputeCommandEncoder")
            .map(|e| e.selector)
            .filter(|s| s.starts_with("useHeap") || s.starts_with("useResource"))
            .collect();
        assert!(
            residency.is_empty(),
            "the compute encoder now declares {residency:?} of its own; an \
             override may not share the inherited record shape, so the \
             OP_USE_HEAPS/OP_USE_RESOURCES layouts need a fresh capture"
        );
        // The render encoder does ship them, and its opcodes are not these.
        use reims_vgpu_wire::ops::render as wire;
        assert_ne!(OP_USE_HEAPS, wire::OPCODE_USE_HEAP);
        assert_ne!(OP_USE_RESOURCES, wire::OPCODE_USE_RESOURCE);
    }

    #[test]
    fn inherited_residency_records_preserve_their_typed_reference_arrays() {
        let mut heaps = vec![0u8; COUNT_BASE + 2 * REF_SIZE];
        st32(&mut heaps, OP_USE_HEAPS);
        st32(&mut heaps[4..], (COUNT_BASE + 2 * REF_SIZE) as u32);
        st32(&mut heaps[OP_HEADER_LEN..], 2);
        st32(&mut heaps[COUNT_BASE..], 5151);
        st32(&mut heaps[COUNT_BASE + REF_SIZE..], 4343);
        let heaps = decode(&heaps).expect("heap residency");
        assert_eq!(heaps.kind, Kind::UseHeaps);
        assert_eq!(
            heaps
                .heaps
                .iter()
                .map(|reference| reference.get())
                .collect::<Vec<_>>(),
            vec![5151, 4343]
        );

        let mut resources = vec![0u8; BIND_BASE + 2 * REF_SIZE];
        st32(&mut resources, OP_USE_RESOURCES);
        st32(&mut resources[4..], (BIND_BASE + 2 * REF_SIZE) as u32);
        st32(&mut resources[OP_HEADER_LEN..], 2);
        st32(&mut resources[COUNT_BASE..], 3);
        st32(&mut resources[BIND_BASE..], 7171);
        st32(&mut resources[BIND_BASE + REF_SIZE..], 8181);
        let resources = decode(&resources).expect("resource residency");
        assert_eq!(resources.kind, Kind::UseResources);
        assert_eq!(
            resources
                .resources
                .iter()
                .map(|reference| reference.get())
                .collect::<Vec<_>>(),
            vec![7171, 8181]
        );
    }

    /// Every compute opcode Apple's serializer emits has a constant here, and
    /// the two this module names beyond them are named as exceptions.
    ///
    /// The render sibling of this test can assert plain set equality; this one
    /// cannot, and the difference is the point. `0x86`/`0x87` are declared on
    /// the encoder base class rather than on this one, which is what
    /// `the_compute_encoder_declares_no_residency_selector` states and all it
    /// states — the manifest is built per class from each class's own methods
    /// and has no row for the base, so an inherited selector cannot appear in
    /// the set this test compares against. See [`OP_USE_HEAPS`].
    ///
    /// # This list used to hold four, and the other two were a wrong claim
    ///
    /// `0xe3`/`0xe6` sat here as "a selector Apple's serializer **refuses**,
    /// failing an assertion instead of emitting". That was measured, and it was
    /// measured in one capability state. Both selectors are gated —
    /// `insertCompressedTextureReinterpretationFlush` on
    /// `-setSupportsInsertCompressedTextureReinterpretationFlush:`,
    /// `dispatchThreadsWithIndirectBuffer:indirectBufferOffset:` on
    /// `-setSupportsDispatchThreadsIndirect:` — and with the flag forced on the
    /// serializer emits both. They are ordinary derived opcodes now, and the
    /// layouts this module already read for them turned out to be right.
    ///
    /// The lesson is about the *shape* of the old claim rather than its
    /// content: an assertion is a refusal by this harness's inputs, not by
    /// Apple, and neither is a claim any single capability state can support.
    ///
    /// Keeping the remaining exception explicit is what stops a third from being
    /// added silently.
    ///
    /// The gap direction is the one that costs guest work. An opcode Apple
    /// emits and this module does not name reaches no arm, and a compute record
    /// that reaches no arm is a dispatch or a bind that never happened.
    #[test]
    fn the_compute_opcode_table_is_apples_compute_manifest_plus_four_named_exceptions() {
        let derived: &[(u32, &str)] = &[
            (
                wire::OPCODE_DISPATCH_THREADGROUPS,
                "wire::OPCODE_DISPATCH_THREADGROUPS",
            ),
            (
                wire::OPCODE_DISPATCH_THREADGROUPS_INDIRECT,
                "wire::OPCODE_DISPATCH_THREADGROUPS_INDIRECT",
            ),
            (
                wire::OPCODE_DISPATCH_THREADS,
                "wire::OPCODE_DISPATCH_THREADS",
            ),
            (wire::OPCODE_SET_BUFFER, "wire::OPCODE_SET_BUFFER"),
            (wire::OPCODE_SET_SAMPLER, "wire::OPCODE_SET_SAMPLER"),
            (wire::OPCODE_SET_SAMPLER_LOD, "wire::OPCODE_SET_SAMPLER_LOD"),
            (wire::OPCODE_SET_TEXTURE, "wire::OPCODE_SET_TEXTURE"),
            (
                wire::OPCODE_SET_BUFFER_OFFSET,
                "wire::OPCODE_SET_BUFFER_OFFSET",
            ),
            (
                wire::OPCODE_SET_PIPELINE_STATE,
                "wire::OPCODE_SET_PIPELINE_STATE",
            ),
            (
                wire::OPCODE_SET_STAGE_IN_REGION,
                "wire::OPCODE_SET_STAGE_IN_REGION",
            ),
            (
                wire::OPCODE_SET_STAGE_IN_REGION_INDIRECT,
                "wire::OPCODE_SET_STAGE_IN_REGION_INDIRECT",
            ),
            (
                wire::OPCODE_SET_THREADGROUP_MEMORY_LENGTH,
                "wire::OPCODE_SET_THREADGROUP_MEMORY_LENGTH",
            ),
            (wire::OPCODE_UPDATE_FENCE, "wire::OPCODE_UPDATE_FENCE"),
            (wire::OPCODE_WAIT_FOR_FENCE, "wire::OPCODE_WAIT_FOR_FENCE"),
            (
                wire::OPCODE_MEMORY_BARRIER_RESOURCES,
                "wire::OPCODE_MEMORY_BARRIER_RESOURCES",
            ),
            (
                wire::OPCODE_MEMORY_BARRIER_SCOPE,
                "wire::OPCODE_MEMORY_BARRIER_SCOPE",
            ),
            (
                wire::OPCODE_SET_IMAGEBLOCK_SIZE,
                "wire::OPCODE_SET_IMAGEBLOCK_SIZE",
            ),
            (
                wire::OPCODE_SET_BUFFER_STRIDE,
                "wire::OPCODE_SET_BUFFER_STRIDE",
            ),
            (
                wire::OPCODE_SET_BUFFER_OFFSET_STRIDE,
                "wire::OPCODE_SET_BUFFER_OFFSET_STRIDE",
            ),
            (
                wire::OPCODE_WRITE_DESCRIPTOR,
                "wire::OPCODE_WRITE_DESCRIPTOR",
            ),
            (wire::OPCODE_START_DO_WHILE, "wire::OPCODE_START_DO_WHILE"),
            (wire::OPCODE_END_DO_WHILE, "wire::OPCODE_END_DO_WHILE"),
            (wire::OPCODE_START_WHILE, "wire::OPCODE_START_WHILE"),
            (wire::OPCODE_END_WHILE, "wire::OPCODE_END_WHILE"),
            (wire::OPCODE_START_IF, "wire::OPCODE_START_IF"),
            (wire::OPCODE_START_ELSE, "wire::OPCODE_START_ELSE"),
            (wire::OPCODE_END_IF, "wire::OPCODE_END_IF"),
            (
                wire::OPCODE_EXECUTE_COMMANDS_RANGE,
                "wire::OPCODE_EXECUTE_COMMANDS_RANGE",
            ),
            (
                wire::OPCODE_EXECUTE_COMMANDS_INDIRECT,
                "wire::OPCODE_EXECUTE_COMMANDS_INDIRECT",
            ),
            (
                wire::OPCODE_INSERT_COMPRESSED_TEXTURE_FLUSH,
                "wire::OPCODE_INSERT_COMPRESSED_TEXTURE_FLUSH",
            ),
            (
                wire::OPCODE_DISPATCH_THREADS_INDIRECT,
                "wire::OPCODE_DISPATCH_THREADS_INDIRECT",
            ),
        ];

        // Each exception names the selector that explains it, or `None` when
        // the class ships no selector for it at all.
        let unsupported: &[(u32, &str, Option<&str>)] = &[
            (OP_USE_HEAPS, "OP_USE_HEAPS", None),
            (OP_USE_RESOURCES, "OP_USE_RESOURCES", None),
        ];

        let rows = || {
            reims_vgpu_wire::manifest::MANIFEST
                .iter()
                .filter(|e| e.class == "PGSerializerComputeCommandEncoder")
        };
        let mut apple: Vec<u32> = rows().flat_map(|e| e.opcodes.iter().copied()).collect();
        apple.sort_unstable();
        apple.dedup();

        for (op, name) in derived {
            assert!(
                apple.contains(op),
                "{name} = {op:#x} is not an opcode Apple's compute manifest \
                 lists, so no capture supports it"
            );
        }
        for op in &apple {
            assert!(
                derived.iter().any(|(d, _)| d == op),
                "Apple's serializer emits compute opcode {op:#x} and this module \
                 names no constant for it, so every dispatch or bind carrying it \
                 reaches no arm"
            );
        }
        for (op, name, selector) in unsupported {
            assert!(
                !apple.contains(op),
                "{name} = {op:#x} now has a producer in Apple's manifest and must \
                 move to the derived roster, taking its decoder arm with it"
            );
            match selector {
                // A refused selector is a measured outcome, so the row must
                // still be there saying Apple declined to emit it.
                Some(sel) => {
                    let row = rows().find(|e| e.selector == *sel).unwrap_or_else(|| {
                        panic!("{name}: the compute class no longer ships {sel}")
                    });
                    assert!(
                        matches!(
                            row.coverage,
                            reims_vgpu_wire::manifest::Coverage::Excluded { .. }
                        ),
                        "{name}: {sel} is no longer excluded, so {op:#x} needs a \
                         capture rather than an inherited number"
                    );
                }
                // No selector at all is the weaker state, and the one the
                // residency test above pins by name.
                None => assert!(
                    !rows().any(|e| e.opcodes.contains(op)),
                    "{name}: a selector now claims {op:#x}"
                ),
            }
        }
        assert_eq!(
            derived.len(),
            apple.len(),
            "the derived roster has a duplicate entry"
        );
    }
}
